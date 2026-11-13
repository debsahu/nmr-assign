//! Probabilistic structure-based methyl NOE contact-graph assignment objective $E(\pi)$,
//! deterministic multi-replica CPU simulated annealing solver, consensus marginal + local-swap
//! Boltzmann confidence calibration, and NMR-STAR v3 writer (`docs/specs/13-methyl-trosy.md` §3).
//!
//! Provenance: Clean-room implementation from peer-reviewed biomolecular NMR literature:
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125,
//!   10120–10121 & 10420–10428 — `1H-13C` methyl-TROSY HMQC in high-MW proteins).
//! - Gans, P. et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962 — stereospecific `pro-R` / `pro-S`
//!   `Leu` / `Val` precursor labeling).
//! - Ruschak, A. M. & Kay, L. E. (*J. Biomol. NMR* 2010, 46, 75–87).
//! - Xu, Y., Lin, Z., Ho, C. & Yang, D. (*J. Am. Chem. Soc.* 2009, 131, 9480–9481 — automated
//!   methyl resonance assignment from NOE networks).
//! - Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G. (*J. Magn. Reson.* 2012, 214, 103–110).
//! - Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J.
//!   & Kay, L. E. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533 — probabilistic methyl NOE contact-graph
//!   matching).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::thread;

use crate::methyl_trosy::{
    classify_methyl_peak, hmcmcbca_log_likelihood, link_hmcmcbca_geminal_pairs,
    lookup_site_methyl_shifts, MethylClassPosterior, MethylGroupKind, MethylNoePeak,
    MethylRootPeak, MethylStructureModel,
};
use crate::nmrstar::Reference;
use crate::rng::Rng;

pub const METHYL_FORBID: f32 = -1.0e9;
pub const METHYL_FORBID_HALF: f32 = -5.0e8;

/// Scoring hyperparameters for the methyl NOE contact-graph objective $E(\pi)$.
#[derive(Clone, Debug)]
pub struct MethylScoreConfig {
    /// Upper structural `C-C` distance cutoff in Angstroms before violation penalty (`default = 10.0`).
    pub d_cutoff_angstrom: f64,
    /// Expected-observable short structural `C-C` contact radius in Angstroms (`default = 7.6`).
    pub r_obs_short_angstrom: f64,
    /// Gaussian distance-agreement tolerance $\sigma_d$ in Angstroms (`default = 1.15`).
    pub sigma_d_angstrom: f64,
    /// Flat-bottom rotamer tolerance around $\hat{d}_e$ in Angstroms (`default = 0.35`).
    pub flat_bottom_d_angstrom: f64,
    /// Maximum missing-peak penalty $\lambda_0$ at $d_{\text{CC}} = 2.5\text{ \AA}$ (`default = 2.40`).
    pub lambda_miss_max: f64,
    /// Diagonal `13C` overlap mask tolerance in ppm (`default = 0.18`).
    pub diag_mask_c_ppm: f64,
    /// Baseline unary bonus for assigning an observed methyl peak (`default = 2.50`).
    pub assign_bonus: f64,
    /// Edge weight for reciprocally corroborated (`s->t` and `t->s`) NOEs (`default = 1.35`).
    pub w_bidir_edge: f64,
    /// Edge weight for unidirectional (`s->t` or `t->s`) NOEs (`default = 0.75`).
    pub w_unidir_edge: f64,
    /// Matching tolerance `(H_ppm, C_ppm)` for mapping unindexed NOE peaks onto 2D HMQC roots.
    pub root_match_tol_h_ppm: f64,
    pub root_match_tol_c_ppm: f64,
    /// Aliphatic carbon tolerance for `HMCM(CG)CBCA` geminal linking (`default = 0.18`).
    pub hmcm_tol_aliph_ppm: f64,
}

impl Default for MethylScoreConfig {
    fn default() -> Self {
        Self {
            d_cutoff_angstrom: 10.0,
            r_obs_short_angstrom: 7.6,
            sigma_d_angstrom: 1.15,
            flat_bottom_d_angstrom: 0.35,
            lambda_miss_max: 2.40,
            diag_mask_c_ppm: 0.18,
            assign_bonus: 2.50,
            w_bidir_edge: 1.35,
            w_unidir_edge: 0.75,
            root_match_tol_h_ppm: 0.045,
            root_match_tol_c_ppm: 0.14,
            hmcm_tol_aliph_ppm: 0.18,
        }
    }
}

/// Multi-replica simulated annealing configuration for methyl contact-graph optimization.
#[derive(Clone, Debug)]
pub struct MethylAnnealConfig {
    pub replicas: usize,
    pub steps_per_root: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub seed: u64,
}

impl Default for MethylAnnealConfig {
    fn default() -> Self {
        Self {
            replicas: 56,
            steps_per_root: 235,
            t_start: 22.0,
            t_end: 0.15,
            seed: 2026,
        }
    }
}

/// Undirected observed NOE edge `s1 < s2` with unsupervised $r^{-6}$-calibrated target distance.
#[derive(Clone, Copy, Debug)]
pub struct ObsNoeEdge {
    pub s1: u32,
    pub s2: u32,
    pub d_hat: f32,
    pub weight: f32,
}

/// CSR adjacency entry for an observed NOE neighbor of root `s`.
#[derive(Clone, Copy, Debug)]
pub struct ObsAdjEntry {
    pub peer_s: u32,
    pub d_hat: f32,
    pub weight: f32,
}

/// Undirected short structural contact `m1 < m2` with missing-peak penalty $\lambda_{\text{miss}}$.
#[derive(Clone, Copy, Debug)]
pub struct StructShortEdge {
    pub m1: u32,
    pub m2: u32,
    pub lambda_miss: f32,
}

/// CSR adjacency entry for a short structural neighbor of site `m`.
#[derive(Clone, Copy, Debug)]
pub struct StructShortAdjEntry {
    pub peer_m: u32,
    pub lambda_miss: f32,
}

/// Precompiled methyl contact-graph optimization problem shared identically by CPU and WebGPU engines.
#[derive(Clone, Debug)]
pub struct MethylContactProblem {
    pub s_len: usize,
    pub m_len: usize,
    pub d_cutoff: f32,
    pub sigma_d: f32,
    pub flat_bottom_d: f32,
    /// Per-root normalization factor `1.0 / max(s_len, 1)`.
    pub norm_inv: f64,
    /// `S x M` unary matrix (`METHYL_FORBID` for incompatible pairs).
    pub u_mat: Vec<f32>,
    /// Compatible structural site indices `m` per observed root `s`.
    pub slots: Vec<Vec<u32>>,
    /// `M x M` structural `C-C` distance matrix in Angstroms.
    pub d_mat: Vec<f32>,
    /// Geminal partner index for each structural site `m` (`-1` if none).
    pub gem_site: Vec<i32>,
    /// `S x S` geminal coupling matrix $G_{\text{gem}}(s_1, s_2)$ when placed on `(m, gem_site[m])`.
    pub gem_obs_mat: Vec<f32>,
    /// `S x S` indicator (`1` if `(s1, s2)` has an observed NOE OR is diagonal-masked in `13C`).
    pub obs_explained_mask: Vec<u32>,
    pub obs_edges: Vec<ObsNoeEdge>,
    pub obs_adj_offsets: Vec<u32>,
    pub obs_adj_entries: Vec<ObsAdjEntry>,
    pub struct_short_edges: Vec<StructShortEdge>,
    pub struct_short_offsets: Vec<u32>,
    pub struct_short_entries: Vec<StructShortAdjEntry>,
    pub struct_cut_offsets: Vec<u32>,
    pub struct_cut_peers: Vec<u32>,
    pub class_posteriors: Vec<MethylClassPosterior>,
}

/// Result of CPU or WebGPU methyl contact-graph assignment.
#[derive(Clone, Debug)]
pub struct MethylAssignResult {
    /// Assigned `site_idx` (`0..M`) for each observed root `0..S`, or `-1` if unassigned.
    pub position: Vec<i32>,
    /// Calibrated assignment confidence in `[0.0, 1.0]` per observed root `0..S`.
    pub confidence: Vec<f64>,
    /// Normalized Bayesian methyl-class posteriors per observed root `0..S`.
    pub class_posteriors: Vec<MethylClassPosterior>,
    /// Per-root normalized objective $E(\pi^*)$ of the consensus assignment.
    pub score: f64,
    /// Unnormalized total log-posterior score $E_{\text{raw}}(\pi^*)$.
    pub raw_score: f64,
    /// Per-root normalized objective scores across all completed annealing replicas.
    pub run_scores: Vec<f64>,
}

/// Pairwise NOE contact-graph edge agreement $W_{\text{NOE}}(e, d_{\text{CC}})$ evaluated in exact
/// single-precision-compatible arithmetic so CPU and WGSL match to `< 1e-6`.
#[inline]
pub fn noe_edge_score_f32(
    d_cc: f32,
    d_hat: f32,
    weight: f32,
    d_cutoff: f32,
    sigma_d: f32,
    flat_bottom_d: f32,
) -> f32 {
    if d_cc <= d_cutoff {
        let diff = (d_cc - d_hat).abs() - flat_bottom_d;
        let ex = if diff > 0.0 { diff / sigma_d } else { 0.0 };
        weight * (2.60 - 0.5 * ex * ex)
    } else {
        let pen = (1.80 + 0.45 * (d_cc - d_cutoff)).min(4.20);
        -weight * pen
    }
}

/// Compile a `MethylContactProblem` from a 3D `MethylStructureModel`, observed `2D HMQC` roots,
/// and `3D C-C-H` or `4D C-H-C-H` `MethylNoePeak`s.
pub fn build_methyl_contact_problem(
    model: &MethylStructureModel,
    roots: &[MethylRootPeak],
    noe_peaks: &[MethylNoePeak],
    cfg: &MethylScoreConfig,
) -> MethylContactProblem {
    let s_len = roots.len();
    let m_len = model.sites.len();
    let norm_inv = 1.0 / (s_len.max(1) as f64);

    let class_posteriors: Vec<MethylClassPosterior> = roots
        .iter()
        .map(|rt| classify_methyl_peak(rt.h_ppm, rt.c_ppm, rt.hmcm.as_ref(), &model.scheme))
        .collect();

    // 1. Symmetrize observed NOE cross-peaks onto root pairs (s1, s2)
    // Accumulate max intensity per directed (recipient -> donor) pair
    let mut dir_inten: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for pk in noe_peaks {
        let rec_opt = pk.root_idx.filter(|&i| i < s_len).or_else(|| {
            find_closest_root_2d(
                roots,
                pk.h_root_ppm,
                pk.c_root_ppm,
                cfg.root_match_tol_h_ppm,
                cfg.root_match_tol_c_ppm,
            )
        });
        let Some(rec) = rec_opt else {
            continue;
        };

        if let Some(don) = pk.donor_root_idx.filter(|&i| i < s_len && i != rec) {
            let e = dir_inten.entry((rec, don)).or_insert(0.0);
            if pk.intensity > *e {
                *e = pk.intensity;
            }
        } else if let Some(h_n) = pk.h_noe_ppm {
            if let Some(don) = find_closest_root_2d(
                roots,
                h_n,
                pk.c_noe_ppm,
                cfg.root_match_tol_h_ppm * 1.25,
                cfg.root_match_tol_c_ppm,
            ) {
                if don != rec {
                    let e = dir_inten.entry((rec, don)).or_insert(0.0);
                    if pk.intensity > *e {
                        *e = pk.intensity;
                    }
                }
            }
        } else {
            // Unindexed 3D C-C-H peak without pre-resolved donor: match candidate roots by 13C_noe
            let cands: Vec<usize> = (0..s_len)
                .filter(|&t| {
                    t != rec && (roots[t].c_ppm - pk.c_noe_ppm).abs() <= cfg.root_match_tol_c_ppm
                })
                .collect();
            if cands.len() == 1 {
                let don = cands[0];
                let e = dir_inten.entry((rec, don)).or_insert(0.0);
                if pk.intensity > *e {
                    *e = pk.intensity;
                }
            } else if cands.len() > 1 {
                // Prefer candidates with reciprocal intensity back at C_rec; otherwise take closest C
                let mut picked = false;
                for &don in &cands {
                    let has_rev = noe_peaks.iter().any(|q| {
                        let q_rec = q.root_idx.or_else(|| {
                            find_closest_root_2d(
                                roots,
                                q.h_root_ppm,
                                q.c_root_ppm,
                                cfg.root_match_tol_h_ppm,
                                cfg.root_match_tol_c_ppm,
                            )
                        });
                        q_rec == Some(don)
                            && (q.c_noe_ppm - roots[rec].c_ppm).abs() <= cfg.root_match_tol_c_ppm
                    });
                    if has_rev {
                        let e = dir_inten.entry((rec, don)).or_insert(0.0);
                        if pk.intensity > *e {
                            *e = pk.intensity;
                        }
                        picked = true;
                    }
                }
                if !picked {
                    let best_don = *cands
                        .iter()
                        .min_by(|&&a, &&b| {
                            (roots[a].c_ppm - pk.c_noe_ppm)
                                .abs()
                                .total_cmp(&(roots[b].c_ppm - pk.c_noe_ppm).abs())
                        })
                        .unwrap();
                    let e = dir_inten.entry((rec, best_don)).or_insert(0.0);
                    if pk.intensity > *e {
                        *e = pk.intensity;
                    }
                }
            }
        }
    }

    let mut raw_undir: Vec<(usize, usize, f64, bool)> = Vec::new();
    let mut visited_pairs: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (&(a, b), &i_ab) in &dir_inten {
        let (lo, hi) = (a.min(b), a.max(b));
        if visited_pairs.contains_key(&(lo, hi)) {
            continue;
        }
        visited_pairs.insert((lo, hi), raw_undir.len());
        if let Some(&i_ba) = dir_inten.get(&(b, a)) {
            raw_undir.push((lo, hi, (i_ab * i_ba).sqrt(), true));
        } else {
            raw_undir.push((lo, hi, i_ab, false));
        }
    }

    // Pre-compute HMCM(CG)CBCA geminal links so r^-6 distance calibration matches inter-residue
    // structural contacts (`3.4-9.0 A`) without collapsing onto `2.50 A` intraresidue geminal pairs.
    let hmcm_gem_pairs = link_hmcmcbca_geminal_pairs(roots, cfg.hmcm_tol_aliph_ppm);
    let mut hmcm_partner = vec![None::<usize>; s_len];
    for &(a, b) in &hmcm_gem_pairs {
        hmcm_partner[a] = Some(b);
        hmcm_partner[b] = Some(a);
    }

    let n_struct_gem_pairs = model
        .sites
        .iter()
        .enumerate()
        .filter(|&(i, st)| st.geminal_partner.is_some_and(|p| p > i))
        .count();

    let mut sorted_intens: Vec<f64> = raw_undir
        .iter()
        .filter(|&&(s1, s2, _, _)| hmcm_partner[s1] != Some(s2))
        .map(|x| x.2)
        .collect();
    sorted_intens.sort_by(|a, b| a.total_cmp(b));
    // When HMCM is not supplied on a geminal labeling scheme, trim the ~75% non-diagonal geminal tail
    let eff_len = if hmcm_gem_pairs.is_empty() && n_struct_gem_pairs > 0 {
        sorted_intens
            .len()
            .saturating_sub((n_struct_gem_pairs * 3) / 4)
            .max(1)
    } else {
        sorted_intens.len()
    };
    let inter_intens = &sorted_intens[..eff_len.min(sorted_intens.len())];
    let i_cal = if inter_intens.is_empty() {
        1.0
    } else {
        inter_intens[(inter_intens.len() * 80 / 100).min(inter_intens.len() - 1)].max(1e-9)
    };

    let mut struct_short_dists: Vec<f64> = Vec::new();
    for m1 in 0..m_len {
        for m2 in (m1 + 1)..m_len {
            if model.sites[m1].geminal_partner == Some(m2) {
                continue;
            }
            let d = model.dist_cc(m1, m2);
            if d <= 9.0 {
                struct_short_dists.push(d);
            }
        }
    }
    struct_short_dists.sort_by(|a, b| a.total_cmp(b));
    let d_cal = if struct_short_dists.is_empty() {
        4.8
    } else {
        struct_short_dists[(struct_short_dists.len() * 20 / 100).min(struct_short_dists.len() - 1)]
            .clamp(3.6, 7.5)
    };
    let d_sat6 = 3.8_f64.powi(6);
    let deff6_cal = d_cal.powi(6) + d_sat6;

    let mut obs_edges = Vec::with_capacity(raw_undir.len());
    let mut obs_adj_lists: Vec<Vec<ObsAdjEntry>> = vec![Vec::new(); s_len];
    let mut obs_explained_mask = vec![0u32; s_len * s_len];
    let mut obs_dhat_lookup: BTreeMap<(usize, usize), f32> = BTreeMap::new();

    for i in 0..s_len {
        obs_explained_mask[i * s_len + i] = 1;
        for j in (i + 1)..s_len {
            if (roots[i].c_ppm - roots[j].c_ppm).abs() < cfg.diag_mask_c_ppm {
                obs_explained_mask[i * s_len + j] = 1;
                obs_explained_mask[j * s_len + i] = 1;
            }
        }
    }

    for (s1, s2, inten, bidir) in raw_undir {
        let deff6 = deff6_cal * (i_cal / inten.max(1e-12));
        let d_hat = (deff6 - d_sat6)
            .max(2.45_f64.powi(6))
            .powf(1.0 / 6.0)
            .clamp(2.45, 9.2) as f32;
        let weight = (if bidir {
            cfg.w_bidir_edge
        } else {
            cfg.w_unidir_edge
        }) as f32;
        obs_edges.push(ObsNoeEdge {
            s1: s1 as u32,
            s2: s2 as u32,
            d_hat,
            weight,
        });
        obs_adj_lists[s1].push(ObsAdjEntry {
            peer_s: s2 as u32,
            d_hat,
            weight,
        });
        obs_adj_lists[s2].push(ObsAdjEntry {
            peer_s: s1 as u32,
            d_hat,
            weight,
        });
        obs_explained_mask[s1 * s_len + s2] = 1;
        obs_explained_mask[s2 * s_len + s1] = 1;
        obs_dhat_lookup.insert((s1, s2), d_hat);
    }

    let mut obs_adj_offsets = Vec::with_capacity(s_len + 1);
    let mut obs_adj_entries = Vec::new();
    obs_adj_offsets.push(0);
    for list in &obs_adj_lists {
        obs_adj_entries.extend_from_slice(list);
        obs_adj_offsets.push(obs_adj_entries.len() as u32);
    }

    // 2. Geminal coupling matrix G_gem(s1, s2)

    let mut gem_obs_mat = vec![0.0f32; s_len * s_len];
    for s1 in 0..s_len {
        for s2 in (s1 + 1)..s_len {
            let val: f32 = match (hmcm_partner[s1], hmcm_partner[s2]) {
                (Some(p1), _) if p1 == s2 => 8.0,
                (Some(_), _) | (_, Some(_)) => -8.0,
                (None, None) => {
                    if let Some(&dh) = obs_dhat_lookup.get(&(s1, s2)) {
                        if dh <= 4.85 {
                            2.2
                        } else {
                            -0.8
                        }
                    } else if (roots[s1].c_ppm - roots[s2].c_ppm).abs() < cfg.diag_mask_c_ppm {
                        1.0
                    } else {
                        -1.5
                    }
                }
            };
            gem_obs_mat[s1 * s_len + s2] = val;
            gem_obs_mat[s2 * s_len + s1] = val;
        }
    }

    // 3. Unary compatibility matrix U(s, m)
    let mut u_mat = vec![METHYL_FORBID; s_len * m_len];
    let mut slots: Vec<Vec<u32>> = vec![Vec::new(); s_len];

    for (m, site) in model.sites.iter().enumerate() {
        let (mu_c0, sd_c0, mu_h0, sd_h0) = site.kind.bmrb_prior();

        // Mild conservative gamma-gauche rotamer adjustment when sidechain dihedral is present
        let c_rot_adj = match (site.kind, site.chi_deg) {
            (
                MethylGroupKind::IleCd1
                | MethylGroupKind::LeuCd1
                | MethylGroupKind::LeuCd2
                | MethylGroupKind::ValCg1
                | MethylGroupKind::ValCg2,
                Some(chi),
            ) => {
                // Trans (|chi| > 135 deg) shifts slightly downfield (+0.35 ppm), gauche (-0.25 ppm)
                if chi.abs() > 135.0 {
                    0.35
                } else if chi.abs() < 95.0 {
                    -0.25
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let mu_c = mu_c0 + c_rot_adj;
        let sd_c = (sd_c0 * sd_c0 + 0.35 * 0.35).sqrt();
        let mu_h = mu_h0 + site.ring_current_h_offset_ppm;
        let sd_h = (sd_h0 * sd_h0 + site.ring_current_h_extra_sd_ppm.powi(2) + 0.08 * 0.08).sqrt();
        let cls = site.kind.residue_class();

        for (s, rt) in roots.iter().enumerate() {
            let zc = (rt.c_ppm - mu_c) / sd_c;
            if zc.abs() > 5.2 {
                continue;
            }
            let zh = (rt.h_ppm - mu_h) / sd_h;
            if zh.abs() > 6.5 && (rt.h_ppm - mu_h0).abs() > 2.3 {
                continue;
            }
            let l_hmcm = if let Some(ref rec) = rt.hmcm {
                let ll = hmcmcbca_log_likelihood(rec, cls);
                if ll <= -9.5 {
                    continue;
                }
                ll
            } else {
                0.0
            };

            let lc = -0.5 * zc * zc - sd_c.ln();
            let lh = (-0.5 * zh * zh - sd_h.ln()).max(-7.5);
            let u_val = (cfg.assign_bonus + lc + lh + l_hmcm) as f32;
            u_mat[s * m_len + m] = u_val;
            slots[s].push(m as u32);
        }
    }

    // Fallback safety: if an extreme chemical-shift outlier had 0 slots, open all active sites of closest C class
    for (s, rt) in roots.iter().enumerate() {
        if slots[s].is_empty() && m_len > 0 {
            for (m, site) in model.sites.iter().enumerate() {
                let (mu_c0, sd_c0, _, _) = site.kind.bmrb_prior();
                let zc = (rt.c_ppm - mu_c0) / (sd_c0 * 1.5);
                if zc.abs() <= 7.5 {
                    u_mat[s * m_len + m] = (-0.5 * zc * zc - 4.0) as f32;
                    slots[s].push(m as u32);
                }
            }
            if slots[s].is_empty() {
                for m in 0..m_len {
                    u_mat[s * m_len + m] = -12.0;
                    slots[s].push(m as u32);
                }
            }
        }
    }

    // 4. Structural distance matrix, short-contact CSR, and cutoff-neighbor CSR
    let d_mat: Vec<f32> = model.d_cc.iter().map(|&d| d as f32).collect();
    let gem_site: Vec<i32> = model
        .sites
        .iter()
        .map(|st| st.geminal_partner.map(|p| p as i32).unwrap_or(-1))
        .collect();

    let mut struct_short_edges = Vec::new();
    let mut struct_short_lists: Vec<Vec<StructShortAdjEntry>> = vec![Vec::new(); m_len];
    let mut struct_cut_lists: Vec<Vec<u32>> = vec![Vec::new(); m_len];

    for m1 in 0..m_len {
        for m2 in (m1 + 1)..m_len {
            let d = model.dist_cc(m1, m2);
            if d <= cfg.d_cutoff_angstrom {
                struct_cut_lists[m1].push(m2 as u32);
                struct_cut_lists[m2].push(m1 as u32);
            }
            if d <= cfg.r_obs_short_angstrom {
                let ratio = ((cfg.r_obs_short_angstrom - d)
                    / (cfg.r_obs_short_angstrom - 2.5).max(0.5))
                .clamp(0.0, 1.5);
                let lam = (cfg.lambda_miss_max * ratio * ratio) as f32;
                struct_short_edges.push(StructShortEdge {
                    m1: m1 as u32,
                    m2: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m1].push(StructShortAdjEntry {
                    peer_m: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m2].push(StructShortAdjEntry {
                    peer_m: m1 as u32,
                    lambda_miss: lam,
                });
            }
        }
    }

    let mut struct_short_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_short_entries = Vec::new();
    struct_short_offsets.push(0);
    for list in &struct_short_lists {
        struct_short_entries.extend_from_slice(list);
        struct_short_offsets.push(struct_short_entries.len() as u32);
    }

    let mut struct_cut_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_cut_peers = Vec::new();
    struct_cut_offsets.push(0);
    for list in &struct_cut_lists {
        struct_cut_peers.extend_from_slice(list);
        struct_cut_offsets.push(struct_cut_peers.len() as u32);
    }

    MethylContactProblem {
        s_len,
        m_len,
        d_cutoff: cfg.d_cutoff_angstrom as f32,
        sigma_d: cfg.sigma_d_angstrom as f32,
        flat_bottom_d: cfg.flat_bottom_d_angstrom as f32,
        norm_inv,
        u_mat,
        slots,
        d_mat,
        gem_site,
        gem_obs_mat,
        obs_explained_mask,
        obs_edges,
        obs_adj_offsets,
        obs_adj_entries,
        struct_short_edges,
        struct_short_offsets,
        struct_short_entries,
        struct_cut_offsets,
        struct_cut_peers,
        class_posteriors,
    }
}

fn find_closest_root_2d(
    roots: &[MethylRootPeak],
    h_ppm: f64,
    c_ppm: f64,
    tol_h: f64,
    tol_c: f64,
) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (i, rt) in roots.iter().enumerate() {
        let dh = (rt.h_ppm - h_ppm) / tol_h.max(1e-6);
        let dc = (rt.c_ppm - c_ppm) / tol_c.max(1e-6);
        if dh.abs() <= 1.0 && dc.abs() <= 1.0 {
            let d2 = dh * dh + dc * dc;
            if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, i));
            }
        }
    }
    best.map(|(_, idx)| idx)
}

/// Evaluate exact unnormalized objective $E_{\text{raw}}(\pi)$ in `f64` from `pos[0..S]`.
pub fn evaluate_methyl_raw_score(prob: &MethylContactProblem, pos: &[i32]) -> f64 {
    assert_eq!(pos.len(), prob.s_len);
    let mut at = vec![-1i32; prob.m_len];
    for (s, &m) in pos.iter().enumerate() {
        if m >= 0 && (m as usize) < prob.m_len {
            if at[m as usize] >= 0 {
                // Invalid non-injective mapping
                return -1.0e12;
            }
            at[m as usize] = s as i32;
        }
    }
    evaluate_methyl_raw_from_both(prob, &at, pos)
}

/// Evaluate exact per-root normalized objective $E(\pi) = E_{\text{raw}}(\pi) / \max(S, 1)$ in `f64`.
#[inline]
pub fn evaluate_methyl_objective(prob: &MethylContactProblem, pos: &[i32]) -> f64 {
    evaluate_methyl_raw_score(prob, pos) * prob.norm_inv
}

/// Evaluate exact unnormalized objective $E_{\text{raw}}(\pi)$ given consistent `(at, pos)`.
pub fn evaluate_methyl_raw_from_both(prob: &MethylContactProblem, at: &[i32], pos: &[i32]) -> f64 {
    let s_len = prob.s_len;
    let m_len = prob.m_len;
    let mut sc = 0.0f64;

    // 1. Unary compatibility
    for (s, &m) in pos.iter().enumerate() {
        if m >= 0 {
            sc += prob.u_mat[s * m_len + (m as usize)] as f64;
        }
    }

    // 2. Geminal coupling
    for m1 in 0..m_len {
        let m2 = prob.gem_site[m1];
        if m2 > m1 as i32 {
            let s1 = at[m1];
            let s2 = at[m2 as usize];
            if s1 >= 0 && s2 >= 0 {
                sc += prob.gem_obs_mat[(s1 as usize) * s_len + (s2 as usize)] as f64;
            }
        }
    }

    // 3. Observed NOE contact-graph edge agreement
    for e in &prob.obs_edges {
        let m1 = pos[e.s1 as usize];
        let m2 = pos[e.s2 as usize];
        if m1 >= 0 && m2 >= 0 {
            let d = prob.d_mat[(m1 as usize) * m_len + (m2 as usize)];
            sc += noe_edge_score_f32(
                d,
                e.d_hat,
                e.weight,
                prob.d_cutoff,
                prob.sigma_d,
                prob.flat_bottom_d,
            ) as f64;
        }
    }

    // 4. Missing short structural contacts
    for se in &prob.struct_short_edges {
        let s1 = at[se.m1 as usize];
        let s2 = at[se.m2 as usize];
        if s1 >= 0 && s2 >= 0 {
            if prob.obs_explained_mask[(s1 as usize) * s_len + (s2 as usize)] == 0 {
                sc -= se.lambda_miss as f64;
            }
        } else {
            sc -= 0.5 * (se.lambda_miss as f64);
        }
    }

    sc
}

/// Initialized replica state `(at[0..M], pos[0..S], raw_score)`.
pub type ReplicaInitState = (Vec<i32>, Vec<i32>, f64);

/// Generate deterministic unary+degree-guided randomized initial replica states `(at, pos, raw_score)`
/// shared identically by CPU and WebGPU multi-replica simulated annealing.
pub fn build_initial_replica_states(
    prob: &MethylContactProblem,
    replicas: usize,
    seed: u64,
) -> Vec<ReplicaInitState> {
    let s_len = prob.s_len;
    let m_len = prob.m_len;
    let mut master_rng = Rng::new(seed ^ 0x4D45_5448_494E_4954);
    let mut out = Vec::with_capacity(replicas);

    let obs_deg: Vec<f64> = (0..s_len)
        .map(|s| (prob.obs_adj_offsets[s + 1] - prob.obs_adj_offsets[s]) as f64)
        .collect();
    let struct_deg: Vec<f64> = (0..m_len)
        .map(|m| (prob.struct_cut_offsets[m + 1] - prob.struct_cut_offsets[m]) as f64)
        .collect();

    // Identify mutually strongest geminal-coupled root pairs `(s1, s2)` (`gem_obs_mat > 1.5`)
    // so replicas start with intact Leu/Val geminal pairings rather than broken HMCM links.
    let mut best_gem_partner: Vec<Option<usize>> = vec![None; s_len];
    for (s1, partner_slot) in best_gem_partner.iter_mut().enumerate() {
        let mut best: Option<(f32, usize)> = None;
        for s2 in 0..s_len {
            if s1 == s2 {
                continue;
            }
            let g = prob.gem_obs_mat[s1 * s_len + s2];
            if g > 1.5 && best.is_none_or(|(bg, _)| g > bg) {
                best = Some((g, s2));
            }
        }
        *partner_slot = best.map(|(_, idx)| idx);
    }
    let mut gem_root_pairs: Vec<(usize, usize)> = Vec::new();
    let mut is_in_gem_pair = vec![false; s_len];
    for s1 in 0..s_len {
        if let Some(s2) = best_gem_partner[s1] {
            if s1 < s2 && best_gem_partner[s2] == Some(s1) {
                gem_root_pairs.push((s1, s2));
                is_in_gem_pair[s1] = true;
                is_in_gem_pair[s2] = true;
            }
        }
    }

    let struct_gem_pairs: Vec<(usize, usize)> = (0..m_len)
        .filter_map(|m1| {
            let m2 = prob.gem_site[m1];
            if m2 > m1 as i32 {
                Some((m1, m2 as usize))
            } else {
                None
            }
        })
        .collect();

    for _ in 0..replicas {
        let mut r_rng = Rng::new(master_rng.next_u64());
        let mut at = vec![-1i32; m_len];
        let mut pos = vec![-1i32; s_len];

        // 1. Place geminal-coupled root pairs `(s1, s2)` onto compatible vacant `(m1, m2)` pairs
        let mut pair_order: Vec<(f64, (usize, usize))> = gem_root_pairs
            .iter()
            .copied()
            .map(|(s1, s2)| {
                let cap = (prob.slots[s1].len().min(prob.slots[s2].len()) as f64) + r_rng.f64();
                (cap, (s1, s2))
            })
            .collect();
        pair_order.sort_by(|a, b| a.0.total_cmp(&b.0));

        for &(_, (s1, s2)) in &pair_order {
            let mut best_choice: Option<(usize, usize)> = None;
            let mut best_val = f64::NEG_INFINITY;
            for &(m1, m2) in &struct_gem_pairs {
                if at[m1] >= 0 || at[m2] >= 0 {
                    continue;
                }
                for &(ma, mb) in &[(m1, m2), (m2, m1)] {
                    let u1 = prob.u_mat[s1 * m_len + ma] as f64;
                    let u2 = prob.u_mat[s2 * m_len + mb] as f64;
                    if u1 <= (METHYL_FORBID_HALF as f64) || u2 <= (METHYL_FORBID_HALF as f64) {
                        continue;
                    }
                    let deg_pen = 0.16
                        * ((obs_deg[s1] - struct_deg[ma]).abs()
                            + (obs_deg[s2] - struct_deg[mb]).abs());
                    let noisy = u1 + u2 - deg_pen + r_rng.normal(0.0, 2.40);
                    if noisy > best_val {
                        best_val = noisy;
                        best_choice = Some((ma, mb));
                    }
                }
            }
            if let Some((ma, mb)) = best_choice {
                at[ma] = s1 as i32;
                at[mb] = s2 as i32;
                pos[s1] = ma as i32;
                pos[s2] = mb as i32;
            }
        }

        // 2. Order remaining unassigned roots from most constrained to broadest with random tie-break
        let mut order: Vec<(f64, usize)> = (0..s_len)
            .filter(|&s| pos[s] < 0)
            .map(|s| {
                let base = if is_in_gem_pair[s] { 0.0 } else { 1.0 };
                (base + (prob.slots[s].len() as f64) + r_rng.f64(), s)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));

        for &(_, s) in &order {
            let mut best_m: Option<usize> = None;
            let mut best_val = f64::NEG_INFINITY;
            for &m_u32 in &prob.slots[s] {
                let m = m_u32 as usize;
                if at[m] >= 0 {
                    continue;
                }
                let u = prob.u_mat[s * m_len + m] as f64;
                if u <= (METHYL_FORBID_HALF as f64) {
                    continue;
                }
                let deg_pen = 0.18 * (obs_deg[s] - struct_deg[m]).abs();
                let noisy_score = u - deg_pen + r_rng.normal(0.0, 1.85);
                if noisy_score > best_val {
                    best_val = noisy_score;
                    best_m = Some(m);
                }
            }
            if let Some(m) = best_m {
                at[m] = s as i32;
                pos[s] = m as i32;
            }
        }

        let raw_sc = evaluate_methyl_raw_from_both(prob, &at, &pos);
        out.push((at, pos, raw_sc));
    }

    out
}

/// Evaluate exact incremental energy contribution of a small modified subset `m_sub` ($\le 4$)
/// and `s_sub` ($\le 4$) without double-counting internal edges.
#[inline]
fn eval_subset_contrib(
    prob: &MethylContactProblem,
    at: &[i32],
    pos: &[i32],
    m_sub: &[usize],
    s_sub: &[usize],
) -> f64 {
    let s_len = prob.s_len;
    let m_len = prob.m_len;
    let mut val = 0.0f64;

    // 1. Unary on s_sub
    for &s in s_sub {
        let m = pos[s];
        if m >= 0 {
            val += prob.u_mat[s * m_len + (m as usize)] as f64;
        }
    }

    // 2. Geminal pairs touching m_sub (deduplicated across up to 4 sites)
    let mut seen_gem = [usize::MAX; 4];
    let mut n_gem = 0usize;
    for &m1 in m_sub {
        let m2 = prob.gem_site[m1];
        if m2 >= 0 {
            let lo = m1.min(m2 as usize);
            let hi = m1.max(m2 as usize);
            if !seen_gem[..n_gem].contains(&lo) {
                seen_gem[n_gem] = lo;
                n_gem += 1;
                let s1 = at[lo];
                let s2 = at[hi];
                if s1 >= 0 && s2 >= 0 {
                    val += prob.gem_obs_mat[(s1 as usize) * s_len + (s2 as usize)] as f64;
                }
            }
        }
    }

    // 3. Observed NOE edges touching s_sub
    for &s1 in s_sub {
        let m1 = pos[s1];
        if m1 < 0 {
            continue;
        }
        let start = prob.obs_adj_offsets[s1] as usize;
        let end = prob.obs_adj_offsets[s1 + 1] as usize;
        for entry in &prob.obs_adj_entries[start..end] {
            let s2 = entry.peer_s as usize;
            if s2 < s1 && s_sub.contains(&s2) {
                continue;
            }
            let m2 = pos[s2];
            if m2 >= 0 {
                let d = prob.d_mat[(m1 as usize) * m_len + (m2 as usize)];
                val += noe_edge_score_f32(
                    d,
                    entry.d_hat,
                    entry.weight,
                    prob.d_cutoff,
                    prob.sigma_d,
                    prob.flat_bottom_d,
                ) as f64;
            }
        }
    }

    // 4. Short structural contacts touching m_sub
    for &m1 in m_sub {
        let s1 = at[m1];
        let start = prob.struct_short_offsets[m1] as usize;
        let end = prob.struct_short_offsets[m1 + 1] as usize;
        for entry in &prob.struct_short_entries[start..end] {
            let m2 = entry.peer_m as usize;
            if m2 < m1 && m_sub.contains(&m2) {
                continue;
            }
            let s2 = at[m2];
            if s1 >= 0 && s2 >= 0 {
                if prob.obs_explained_mask[(s1 as usize) * s_len + (s2 as usize)] == 0 {
                    val -= entry.lambda_miss as f64;
                }
            } else {
                val -= 0.5 * (entry.lambda_miss as f64);
            }
        }
    }

    val
}

#[derive(Clone, Copy)]
struct CpuReplicaSchedule {
    total_steps: usize,
    t_start: f64,
    t_end: f64,
    rng_seed: u64,
}

fn run_single_cpu_replica(
    prob: &MethylContactProblem,
    init: ReplicaInitState,
    sched: CpuReplicaSchedule,
) -> (Vec<i32>, f64) {
    let (mut at, mut pos, mut cur) = init;
    let total_steps = sched.total_steps;
    let t_start = sched.t_start;
    let t_end = sched.t_end;
    let rng_seed = sched.rng_seed;
    let s_len = prob.s_len;
    let m_len = prob.m_len;
    if s_len == 0 || m_len == 0 || total_steps == 0 {
        return (pos, cur);
    }

    let mut rng = Rng::new(rng_seed);
    let mut best_sc = cur;
    let mut best_pos = pos.clone();

    let denom = (total_steps.saturating_sub(1)).max(1) as f64;
    let log_ratio = (t_end / t_start).ln();
    let mut m_buf = [0usize; 4];
    let mut s_buf = [0usize; 4];

    for step in 0..total_steps {
        let temp = t_start * (log_ratio * (step as f64 / denom)).exp();
        let draw = rng.f64();

        if draw < 0.44 {
            // Move 1: single-root compatible slot placement / 2-way swap / excess-root unassign
            let s = rng.usize(s_len);
            let sl = &prob.slots[s];
            if sl.is_empty() {
                continue;
            }
            let m = sl[rng.usize(sl.len())] as usize;
            let ms = pos[s];
            if ms == m as i32 {
                if s_len > m_len {
                    m_buf[0] = m;
                    s_buf[0] = s;
                    let before = eval_subset_contrib(prob, &at, &pos, &m_buf[..1], &s_buf[..1]);
                    at[m] = -1;
                    pos[s] = -1;
                    let delta =
                        eval_subset_contrib(prob, &at, &pos, &m_buf[..1], &s_buf[..1]) - before;
                    if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                        cur += delta;
                        if cur > best_sc {
                            best_sc = cur;
                            best_pos.copy_from_slice(&pos);
                        }
                    } else {
                        at[m] = s as i32;
                        pos[s] = m as i32;
                    }
                }
                continue;
            }
            let t = at[m];
            if ms >= 0
                && t >= 0
                && prob.u_mat[(t as usize) * m_len + (ms as usize)] <= METHYL_FORBID_HALF
            {
                continue;
            }
            let nm = if ms < 0 {
                m_buf[0] = m;
                1
            } else {
                m_buf[0] = m;
                m_buf[1] = ms as usize;
                2
            };
            let ns = if t < 0 {
                s_buf[0] = s;
                1
            } else {
                s_buf[0] = s;
                s_buf[1] = t as usize;
                2
            };
            let before = eval_subset_contrib(prob, &at, &pos, &m_buf[..nm], &s_buf[..ns]);
            if ms >= 0 {
                at[ms as usize] = t;
            }
            at[m] = s as i32;
            pos[s] = m as i32;
            if t >= 0 {
                pos[t as usize] = ms;
            }
            let delta = eval_subset_contrib(prob, &at, &pos, &m_buf[..nm], &s_buf[..ns]) - before;
            if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best_sc {
                    best_sc = cur;
                    best_pos.copy_from_slice(&pos);
                }
            } else {
                at[m] = t;
                if ms >= 0 {
                    at[ms as usize] = s as i32;
                }
                pos[s] = ms;
                if t >= 0 {
                    pos[t as usize] = m as i32;
                }
            }
        } else if draw < 0.72 {
            // Move 2: 3D contact-guided neighbor extension from anchored root
            let s_anc = rng.usize(s_len);
            let m_anc = pos[s_anc];
            if m_anc < 0 {
                continue;
            }
            let o0 = prob.obs_adj_offsets[s_anc] as usize;
            let o1 = prob.obs_adj_offsets[s_anc + 1] as usize;
            let c0 = prob.struct_cut_offsets[m_anc as usize] as usize;
            let c1 = prob.struct_cut_offsets[m_anc as usize + 1] as usize;
            if o0 == o1 || c0 == c1 {
                continue;
            }
            let s = prob.obs_adj_entries[o0 + rng.usize(o1 - o0)].peer_s as usize;
            let m = prob.struct_cut_peers[c0 + rng.usize(c1 - c0)] as usize;
            if prob.u_mat[s * m_len + m] <= METHYL_FORBID_HALF || pos[s] == m as i32 {
                continue;
            }
            let ms = pos[s];
            let t = at[m];
            if ms >= 0
                && t >= 0
                && prob.u_mat[(t as usize) * m_len + (ms as usize)] <= METHYL_FORBID_HALF
            {
                continue;
            }
            let nm = if ms < 0 {
                m_buf[0] = m;
                1
            } else {
                m_buf[0] = m;
                m_buf[1] = ms as usize;
                2
            };
            let ns = if t < 0 {
                s_buf[0] = s;
                1
            } else {
                s_buf[0] = s;
                s_buf[1] = t as usize;
                2
            };
            let before = eval_subset_contrib(prob, &at, &pos, &m_buf[..nm], &s_buf[..ns]);
            if ms >= 0 {
                at[ms as usize] = t;
            }
            at[m] = s as i32;
            pos[s] = m as i32;
            if t >= 0 {
                pos[t as usize] = ms;
            }
            let delta = eval_subset_contrib(prob, &at, &pos, &m_buf[..nm], &s_buf[..ns]) - before;
            if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best_sc {
                    best_sc = cur;
                    best_pos.copy_from_slice(&pos);
                }
            } else {
                at[m] = t;
                if ms >= 0 {
                    at[ms as usize] = s as i32;
                }
                pos[s] = ms;
                if t >= 0 {
                    pos[t as usize] = m as i32;
                }
            }
        } else {
            // Move 3 & 4: slot- & contact-guided 4-body geminal residue swap (78%) or prochiral flip (22%)
            let s_seed = rng.usize(s_len);
            let m1_i = pos[s_seed];
            if m1_i < 0 {
                continue;
            }
            let m1 = m1_i as usize;
            let m2_i = prob.gem_site[m1];
            if m2_i < 0 {
                continue;
            }
            let m2 = m2_i as usize;

            if rng.f64() < 0.22 {
                let s1 = at[m1];
                let s2 = at[m2];
                if s1 == s2 {
                    continue;
                }
                if (s1 >= 0 && prob.u_mat[(s1 as usize) * m_len + m2] <= METHYL_FORBID_HALF)
                    || (s2 >= 0 && prob.u_mat[(s2 as usize) * m_len + m1] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                m_buf[0] = m1;
                m_buf[1] = m2;
                let mut ns = 0usize;
                if s1 >= 0 {
                    s_buf[ns] = s1 as usize;
                    ns += 1;
                }
                if s2 >= 0 {
                    s_buf[ns] = s2 as usize;
                    ns += 1;
                }
                let before = eval_subset_contrib(prob, &at, &pos, &m_buf[..2], &s_buf[..ns]);
                at[m1] = s2;
                at[m2] = s1;
                if s1 >= 0 {
                    pos[s1 as usize] = m2 as i32;
                }
                if s2 >= 0 {
                    pos[s2 as usize] = m1 as i32;
                }
                let delta =
                    eval_subset_contrib(prob, &at, &pos, &m_buf[..2], &s_buf[..ns]) - before;
                if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                    cur += delta;
                    if cur > best_sc {
                        best_sc = cur;
                        best_pos.copy_from_slice(&pos);
                    }
                } else {
                    at[m1] = s1;
                    at[m2] = s2;
                    if s1 >= 0 {
                        pos[s1 as usize] = m1 as i32;
                    }
                    if s2 >= 0 {
                        pos[s2 as usize] = m2 as i32;
                    }
                }
            } else {
                // Contact-guided (50%) or compatible-slot-guided (50%) target geminal residue
                let mut m3_opt: Option<usize> = None;
                if rng.usize(2) == 0 {
                    let o0 = prob.obs_adj_offsets[s_seed] as usize;
                    let o1 = prob.obs_adj_offsets[s_seed + 1] as usize;
                    if o0 < o1 {
                        let s_nb = prob.obs_adj_entries[o0 + rng.usize(o1 - o0)].peer_s as usize;
                        let m_nb = pos[s_nb];
                        if m_nb >= 0 {
                            let c0 = prob.struct_cut_offsets[m_nb as usize] as usize;
                            let c1 = prob.struct_cut_offsets[m_nb as usize + 1] as usize;
                            if c0 < c1 {
                                let cand = prob.struct_cut_peers[c0 + rng.usize(c1 - c0)] as usize;
                                if prob.gem_site[cand] >= 0 {
                                    m3_opt = Some(cand);
                                }
                            }
                        }
                    }
                }
                let mut m3 = match m3_opt {
                    Some(c) => c,
                    None => {
                        let sl = &prob.slots[s_seed];
                        if sl.is_empty() {
                            continue;
                        }
                        sl[rng.usize(sl.len())] as usize
                    }
                };
                let m4_i = prob.gem_site[m3];
                if m4_i < 0 || m3 == m1 || m3 == m2 {
                    continue;
                }
                let mut m4 = m4_i as usize;
                if rng.usize(2) == 1 {
                    std::mem::swap(&mut m3, &mut m4);
                }
                let (s1, s2, s3, s4) = (at[m1], at[m2], at[m3], at[m4]);
                if (s1 >= 0 && prob.u_mat[(s1 as usize) * m_len + m3] <= METHYL_FORBID_HALF)
                    || (s2 >= 0 && prob.u_mat[(s2 as usize) * m_len + m4] <= METHYL_FORBID_HALF)
                    || (s3 >= 0 && prob.u_mat[(s3 as usize) * m_len + m1] <= METHYL_FORBID_HALF)
                    || (s4 >= 0 && prob.u_mat[(s4 as usize) * m_len + m2] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                m_buf = [m1, m2, m3, m4];
                let mut ns = 0usize;
                for &sx in &[s1, s2, s3, s4] {
                    if sx >= 0 {
                        s_buf[ns] = sx as usize;
                        ns += 1;
                    }
                }
                if ns == 0 {
                    continue;
                }
                let before = eval_subset_contrib(prob, &at, &pos, &m_buf, &s_buf[..ns]);
                at[m1] = s3;
                at[m2] = s4;
                at[m3] = s1;
                at[m4] = s2;
                if s3 >= 0 {
                    pos[s3 as usize] = m1 as i32;
                }
                if s4 >= 0 {
                    pos[s4 as usize] = m2 as i32;
                }
                if s1 >= 0 {
                    pos[s1 as usize] = m3 as i32;
                }
                if s2 >= 0 {
                    pos[s2 as usize] = m4 as i32;
                }
                let delta = eval_subset_contrib(prob, &at, &pos, &m_buf, &s_buf[..ns]) - before;
                if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                    cur += delta;
                    if cur > best_sc {
                        best_sc = cur;
                        best_pos.copy_from_slice(&pos);
                    }
                } else {
                    at[m1] = s1;
                    at[m2] = s2;
                    at[m3] = s3;
                    at[m4] = s4;
                    if s1 >= 0 {
                        pos[s1 as usize] = m1 as i32;
                    }
                    if s2 >= 0 {
                        pos[s2 as usize] = m2 as i32;
                    }
                    if s3 >= 0 {
                        pos[s3 as usize] = m3 as i32;
                    }
                    if s4 >= 0 {
                        pos[s4 as usize] = m4 as i32;
                    }
                }
            }
        }
    }

    let exact_best = evaluate_methyl_raw_score(prob, &best_pos);
    (best_pos, exact_best)
}

/// Run deterministic multi-threaded CPU multi-replica simulated annealing on `prob`.
pub fn assign_methyl_cpu(
    prob: &MethylContactProblem,
    cfg: &MethylAnnealConfig,
) -> MethylAssignResult {
    let replicas = cfg.replicas.max(1);
    if prob.s_len == 0 || prob.m_len == 0 {
        return MethylAssignResult {
            position: vec![-1; prob.s_len],
            confidence: vec![0.0; prob.s_len],
            class_posteriors: prob.class_posteriors.clone(),
            score: 0.0,
            raw_score: 0.0,
            run_scores: vec![0.0; replicas],
        };
    }

    let init_states = build_initial_replica_states(prob, replicas, cfg.seed);
    let mut seeder = Rng::new(cfg.seed ^ 0x4350_555F_414E_4E4C);
    let rep_seeds: Vec<u64> = (0..replicas).map(|_| seeder.next_u64()).collect();
    let min_floor = cfg.steps_per_root.saturating_mul(85);
    let total_steps = cfg
        .steps_per_root
        .saturating_mul(prob.s_len)
        .max(min_floor)
        .max(1);

    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;
    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(replicas)
        .max(1);

    let mut all_positions = vec![Vec::new(); replicas];
    let mut all_raw_scores = vec![0.0f64; replicas];

    if n_threads <= 1 {
        for (r, (st, rseed)) in init_states.into_iter().zip(rep_seeds).enumerate() {
            let (bp, bsc) = run_single_cpu_replica(
                prob,
                st,
                CpuReplicaSchedule {
                    total_steps,
                    t_start: cfg.t_start,
                    t_end: cfg.t_end,
                    rng_seed: rseed,
                },
            );
            all_positions[r] = bp;
            all_raw_scores[r] = bsc;
        }
    } else {
        let chunk = replicas.div_ceil(n_threads);
        let indexed: Vec<(usize, ReplicaInitState, u64)> = init_states
            .into_iter()
            .zip(rep_seeds)
            .enumerate()
            .map(|(i, (st, sd))| (i, st, sd))
            .collect();

        let results: Vec<Vec<(usize, Vec<i32>, f64)>> = thread::scope(|scope| {
            let mut handles = Vec::new();
            for slice in indexed.chunks(chunk) {
                let t_start = cfg.t_start;
                let t_end = cfg.t_end;
                handles.push(scope.spawn(move || {
                    let mut out = Vec::with_capacity(slice.len());
                    for &(idx, ref st, rseed) in slice {
                        let (bp, bsc) = run_single_cpu_replica(
                            prob,
                            st.clone(),
                            CpuReplicaSchedule {
                                total_steps,
                                t_start,
                                t_end,
                                rng_seed: rseed,
                            },
                        );
                        out.push((idx, bp, bsc));
                    }
                    out
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for batch in results {
            for (idx, bp, bsc) in batch {
                all_positions[idx] = bp;
                all_raw_scores[idx] = bsc;
            }
        }
    }

    consensus_and_calibrate_methyl(prob, &all_positions, &all_raw_scores)
}

/// Combine completed replica assignments via Boltzmann-weighted consensus marginals AND exact
/// local-swap Boltzmann specificity calibration (`docs/specs/13-methyl-trosy.md` §3.3).
pub fn consensus_and_calibrate_methyl(
    prob: &MethylContactProblem,
    positions: &[Vec<i32>],
    raw_scores: &[f64],
) -> MethylAssignResult {
    let s_len = prob.s_len;
    let m_len = prob.m_len;
    let n_runs = positions.len();
    let run_scores: Vec<f64> = raw_scores.iter().map(|&r| r * prob.norm_inv).collect();

    if s_len == 0 || m_len == 0 || n_runs == 0 {
        return MethylAssignResult {
            position: vec![-1; s_len],
            confidence: vec![0.0; s_len],
            class_posteriors: prob.class_posteriors.clone(),
            score: 0.0,
            raw_score: 0.0,
            run_scores,
        };
    }

    let mut ranked: Vec<usize> = (0..n_runs).collect();
    ranked.sort_by(|&a, &b| raw_scores[b].total_cmp(&raw_scores[a]));
    let top_k = (n_runs / 2).max(4).min(n_runs);
    let max_raw = raw_scores[ranked[0]];
    const TAU_REP: f64 = 3.5;

    let mut rep_weights = Vec::with_capacity(top_k);
    let mut w_sum = 0.0f64;
    for &idx in &ranked[..top_k] {
        let w = ((raw_scores[idx] - max_raw) / TAU_REP).max(-30.0).exp();
        rep_weights.push((idx, w));
        w_sum += w;
    }
    let w_sum = w_sum.max(1e-12);

    // Marginal candidate probabilities per (s, m)
    let mut cand_pairs: Vec<(f64, usize, i32)> = Vec::new();
    let mut unassign_prob = vec![0.0f64; s_len];
    for s in 0..s_len {
        let mut counts: BTreeMap<i32, f64> = BTreeMap::new();
        for &(r_idx, w) in &rep_weights {
            *counts.entry(positions[r_idx][s]).or_insert(0.0) += w;
        }
        for (m, wt) in counts {
            let p = wt / w_sum;
            if m < 0 {
                unassign_prob[s] = p;
            } else {
                cand_pairs.push((p, s, m));
            }
        }
    }

    // Conflict-free 1-to-1 consensus injection ordered by descending replica consensus probability
    cand_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut cons_pos = vec![-1i32; s_len];
    let mut cons_at = vec![-1i32; m_len];
    let mut cons_prob = vec![0.0f64; s_len];
    for (p, s, m) in cand_pairs {
        if cons_pos[s] < 0 && cons_at[m as usize] < 0 && p >= unassign_prob[s] {
            cons_pos[s] = m;
            cons_at[m as usize] = s as i32;
            cons_prob[s] = p;
        }
    }

    // If single top replica achieved strictly higher raw score than marginal-merged state (rare tie),
    // adopt the top replica's valid injection while preserving marginal probabilities
    let top_pos = &positions[ranked[0]];
    let merged_raw = evaluate_methyl_raw_from_both(prob, &cons_at, &cons_pos);
    if max_raw > merged_raw + 0.5 {
        cons_pos.copy_from_slice(top_pos);
        cons_at.fill(-1);
        for (s, &m) in cons_pos.iter().enumerate() {
            if m >= 0 {
                cons_at[m as usize] = s as i32;
                let mut wt = 0.0;
                for &(r_idx, w) in &rep_weights {
                    if positions[r_idx][s] == m {
                        wt += w;
                    }
                }
                cons_prob[s] = wt / w_sum;
            }
        }
    }

    let final_raw = evaluate_methyl_raw_from_both(prob, &cons_at, &cons_pos);

    // Local-swap Boltzmann specificity factor across:
    // 1. All compatible single-site swaps m' != m* (including intraresidue prochiral flip),
    // 2. Unassigning root s (m' = -1), and
    // 3. Coupled 4-body geminal residue swaps (m*, gem(m*)) <-> (m3, gem(m3)) for Leu/Val pairs!
    const TAU_SWAP: f64 = 3.0;
    const TAU_GEM4_SWAP: f64 = 4.2;
    let mut confidence = vec![0.0f64; s_len];
    let mut work_at = cons_at.clone();
    let mut work_pos = cons_pos.clone();

    for s in 0..s_len {
        let m_star_i = work_pos[s];
        if m_star_i < 0 {
            confidence[s] = 0.0;
            continue;
        }
        let m_star = m_star_i as usize;
        let mut alt_exp_sum = 0.0f64;

        // 1. Unassigning s -> -1
        {
            let m_sub = [m_star];
            let s_sub = [s];
            let before = eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub);
            work_at[m_star] = -1;
            work_pos[s] = -1;
            let delta = eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub) - before;
            work_at[m_star] = s as i32;
            work_pos[s] = m_star as i32;
            alt_exp_sum += (delta / TAU_SWAP).clamp(-40.0, 20.0).exp();
        }

        // 2. Single-site 2-body swaps m_star <-> m_alt
        for &m_alt_u in &prob.slots[s] {
            let m_alt = m_alt_u as usize;
            if m_alt == m_star {
                continue;
            }
            let t = work_at[m_alt];
            if t >= 0 && prob.u_mat[(t as usize) * m_len + m_star] <= METHYL_FORBID_HALF {
                continue;
            }
            let m_sub = [m_star, m_alt];
            let (s_sub_arr, ns) = if t >= 0 {
                ([s, t as usize], 2usize)
            } else {
                ([s, 0], 1usize)
            };
            let before = eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub_arr[..ns]);
            work_at[m_star] = t;
            work_at[m_alt] = s as i32;
            work_pos[s] = m_alt as i32;
            if t >= 0 {
                work_pos[t as usize] = m_star as i32;
            }
            let delta =
                eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub_arr[..ns]) - before;
            work_at[m_star] = s as i32;
            work_at[m_alt] = t;
            work_pos[s] = m_star as i32;
            if t >= 0 {
                work_pos[t as usize] = m_alt as i32;
            }

            alt_exp_sum += (delta / TAU_SWAP).clamp(-40.0, 20.0).exp();
        }

        // 3. Coupled 4-body geminal swaps (m_star, m_gem) <-> (m3, m4) when m_star has a geminal partner
        let m_gem_i = prob.gem_site[m_star];
        if m_gem_i >= 0 {
            let m_gem = m_gem_i as usize;
            let s_gem = work_at[m_gem];
            for &m3_u in &prob.slots[s] {
                let m3 = m3_u as usize;
                if m3 == m_star || m3 == m_gem {
                    continue;
                }
                let m4_i = prob.gem_site[m3];
                if m4_i < 0 {
                    continue;
                }
                let m4 = m4_i as usize;
                let s3 = work_at[m3];
                let s4 = work_at[m4];
                if (s_gem >= 0 && prob.u_mat[(s_gem as usize) * m_len + m4] <= METHYL_FORBID_HALF)
                    || (s3 >= 0 && prob.u_mat[(s3 as usize) * m_len + m_star] <= METHYL_FORBID_HALF)
                    || (s4 >= 0 && prob.u_mat[(s4 as usize) * m_len + m_gem] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                let m_sub = [m_star, m_gem, m3, m4];
                let mut s_sub = [0usize; 4];
                let mut ns = 0usize;
                for &sx in &[s as i32, s_gem, s3, s4] {
                    if sx >= 0 {
                        s_sub[ns] = sx as usize;
                        ns += 1;
                    }
                }
                let before = eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub[..ns]);
                work_at[m_star] = s3;
                work_at[m_gem] = s4;
                work_at[m3] = s as i32;
                work_at[m4] = s_gem;
                work_pos[s] = m3 as i32;
                if s_gem >= 0 {
                    work_pos[s_gem as usize] = m4 as i32;
                }
                if s3 >= 0 {
                    work_pos[s3 as usize] = m_star as i32;
                }
                if s4 >= 0 {
                    work_pos[s4 as usize] = m_gem as i32;
                }
                let delta =
                    eval_subset_contrib(prob, &work_at, &work_pos, &m_sub, &s_sub[..ns]) - before;
                // Restore immediately
                work_at[m_star] = s as i32;
                work_at[m_gem] = s_gem;
                work_at[m3] = s3;
                work_at[m4] = s4;
                work_pos[s] = m_star as i32;
                if s_gem >= 0 {
                    work_pos[s_gem as usize] = m_gem as i32;
                }
                if s3 >= 0 {
                    work_pos[s3 as usize] = m3 as i32;
                }
                if s4 >= 0 {
                    work_pos[s4 as usize] = m4 as i32;
                }
                alt_exp_sum += (delta / TAU_GEM4_SWAP).clamp(-40.0, 20.0).exp();
            }
        }

        let obs_deg = prob.obs_adj_offsets[s + 1] - prob.obs_adj_offsets[s];
        let deg_cap = if obs_deg == 0 && prob.slots[s].len() > 1 {
            0.72
        } else {
            1.0
        };
        let p_swap = 1.0 / (1.0 + alt_exp_sum);
        confidence[s] = (cons_prob[s] * p_swap * deg_cap).clamp(0.0, 1.0);
    }

    MethylAssignResult {
        position: cons_pos,
        confidence,
        class_posteriors: prob.class_posteriors.clone(),
        score: final_raw * prob.norm_inv,
        raw_score: final_raw,
        run_scores,
    }
}

/// Write assigned methyl `(13C, 1H)` chemical shifts and per-site confidences to an NMR-STAR v3 file.
pub fn write_methyl_star_assignments(
    path: impl AsRef<Path>,
    model: &MethylStructureModel,
    roots: &[MethylRootPeak],
    result: &MethylAssignResult,
) -> Result<(), String> {
    let mut lines = vec![
        format!(
            "# Methyl-TROSY contact-graph assignment (scheme={}, sites={}, roots={})",
            model.scheme.label(),
            model.sites.len(),
            roots.len()
        ),
        String::new(),
        "data_nmr_assign_methyl_output".to_string(),
        String::new(),
        "save_entity_1".to_string(),
        "   _Entity.Sf_category  entity".to_string(),
        "   _Entity.ID           1".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Entity_poly_seq.Entity_ID".to_string(),
        "      _Entity_poly_seq.Num".to_string(),
        "      _Entity_poly_seq.Mon_ID".to_string(),
        String::new(),
    ];
    for (sid, mon) in model.seq_ids.iter().zip(&model.sequence) {
        lines.push(format!("      1 {:5} {:3}", sid, mon));
    }
    lines.extend([
        "   stop_".to_string(),
        "save_".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Atom_chem_shift.ID".to_string(),
        "      _Atom_chem_shift.Entity_ID".to_string(),
        "      _Atom_chem_shift.Seq_ID".to_string(),
        "      _Atom_chem_shift.Comp_ID".to_string(),
        "      _Atom_chem_shift.Atom_ID".to_string(),
        "      _Atom_chem_shift.Atom_type".to_string(),
        "      _Atom_chem_shift.Val".to_string(),
        "      _Atom_chem_shift.Ambiguity_code".to_string(),
        "      _Atom_chem_shift.Details".to_string(),
        String::new(),
    ]);

    let mut assigned_by_site: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    for (s, &m) in result.position.iter().enumerate() {
        if m >= 0 && (m as usize) < model.sites.len() {
            assigned_by_site.insert(m as usize, (s, result.confidence[s]));
        }
    }

    let mut row_id = 1usize;
    for (m, site) in model.sites.iter().enumerate() {
        let Some(&(s, conf)) = assigned_by_site.get(&m) else {
            continue;
        };
        let rt = &roots[s];
        let det = format!("conf={conf:.3}");
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} C {:9.3} 1 {}",
            row_id, site.seq_id, site.res_name, site.carbon_atom, rt.c_ppm, det
        ));
        row_id += 1;
        for &h_atom in &site.kind.proton_atoms() {
            lines.push(format!(
                "      {:5} 1 {:5} {:3} {:4} H {:9.3} 1 {}",
                row_id, site.seq_id, site.res_name, h_atom, rt.h_ppm, det
            ));
            row_id += 1;
        }
    }

    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());

    fs::write(path.as_ref(), lines.join("\n"))
        .map_err(|e| format!("failed to write NMR-STAR {}: {e}", path.as_ref().display()))
}

/// Summary accuracy and high-confidence (`>= hc_threshold`) precision metrics against ground-truth shifts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MethylBenchmarkMetrics {
    pub correct: usize,
    pub total_matched_roots: usize,
    pub accuracy_pct: f64,
    pub hc_correct: usize,
    pub hc_total: usize,
    pub hc_precision_pct: f64,
}

/// Evaluate methyl assignment accuracy and high-confidence precision against a ground-truth `Reference`.
pub fn evaluate_methyl_assignment(
    model: &MethylStructureModel,
    roots: &[MethylRootPeak],
    result: &MethylAssignResult,
    true_ref: &Reference,
    hc_threshold: f64,
) -> MethylBenchmarkMetrics {
    let true_site_shifts: Vec<Option<(f64, f64)>> = model
        .sites
        .iter()
        .map(|st| lookup_site_methyl_shifts(true_ref, st))
        .collect();

    // Match each observed root to its true structural site by closest normalized (H, C) distance
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (s, rt) in roots.iter().enumerate() {
        for (m, opt_hc) in true_site_shifts.iter().enumerate() {
            if let Some((th, tc)) = *opt_hc {
                let dh = (rt.h_ppm - th) / 0.04;
                let dc = (rt.c_ppm - tc) / 0.14;
                if dh.abs() <= 1.5 && dc.abs() <= 1.5 {
                    pairs.push((dh * dh + dc * dc, s, m));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut root_truth = vec![None::<usize>; roots.len()];
    let mut site_taken = vec![false; model.sites.len()];
    for (_, s, m) in pairs {
        if root_truth[s].is_none() && !site_taken[m] {
            root_truth[s] = Some(m);
            site_taken[m] = true;
        }
    }

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut hc_correct = 0usize;
    let mut hc_total = 0usize;

    for (s, truth_opt) in root_truth.into_iter().enumerate() {
        let Some(true_m) = truth_opt else {
            continue;
        };
        total += 1;
        let pred_m = result.position[s];
        let is_ok = pred_m == true_m as i32;
        if is_ok {
            correct += 1;
        }
        if pred_m >= 0 && result.confidence[s] >= hc_threshold {
            hc_total += 1;
            if is_ok {
                hc_correct += 1;
            }
        }
    }

    let accuracy_pct = if total > 0 {
        100.0 * (correct as f64) / (total as f64)
    } else {
        0.0
    };
    let hc_precision_pct = if hc_total > 0 {
        100.0 * (hc_correct as f64) / (hc_total as f64)
    } else {
        100.0
    };

    MethylBenchmarkMetrics {
        correct,
        total_matched_roots: total,
        accuracy_pct,
        hc_correct,
        hc_total,
        hc_precision_pct,
    }
}
