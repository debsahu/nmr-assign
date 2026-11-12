//! Stoichiometry-guided evolutionary and graph self-consistency backbone assignment solver.
//!
//! Provenance: Clean-room implementation derived strictly from peer-reviewed publications:
//! 1. Schmidt, E. & Güntert, P. (2012). "A New Algorithm for Reliable and General NMR Resonance
//!    Assignment." *J. Am. Chem. Soc.* 134(30), 12817–12829 — Sections "Expected Peaks"
//!    (sequence-derived peak network stoichiometry), "Optimization of Assignments" (hybrid
//!    evolutionary recombination of contiguous sequential assignment fragments coupled with local
//!    Metropolis-Hastings simulated annealing), and "Consolidation of Assignments" (multi-island
//!    population consensus).
//! 2. Klukowski, P., Augoff, M., Zieba, M., Drabikowski, M., Gonczarek, A. & Walczak, M. J.
//!    (2018). "NMRNet: a deep learning approach to automated peak picking of protein NMR spectra."
//!    *Bioinformatics* 34(15), 2590–2597 — Section 2.4 (sequence-expected peak count ranking and
//!    spectral artifact suppression).
//! 3. Klukowski, P., Riek, R. & Güntert, P. (2022). *Nature Communications* 13, 6151 — Methods:
//!    stoichiometry-proportional peak list filtering, chemical-shift reference offset alignment,
//!    and iterative residue-graph chemical-shift self-consistency with outlier pruning.
//! 4. Klukowski, P., Damberger, F. F., Allain, F. H.-T., Iwai, H., Kadavath, H., Ramelot, T. A.,
//!    Montelione, G. T., Riek, R. & Güntert, P. (2023). *Science Advances* 9(49), eadi9323 —
//!    Methods: optional 3D structure `(phi, psi)` torsion-conditioned backbone shift expectations,
//!    parameterized via empirical conformational secondary shifts from Spera, S. & Bax, A.,
//!    *J. Am. Chem. Soc.* 1991, 113(14), 5490–5492 and Wishart, D. S. et al., *J. Biomol. NMR*
//!    1995, 5(1), 67–81.
//! 5. **Primary-sequence covalent pre-Proline shift adjustments** (`delta_covalent` on `CA` and
//!    `C'`): Wishart, D. S. et al., *J. Biomol. NMR* 1995, 5, 67–81; Wishart, D. S. et al.,
//!    *J. Biomol. NMR* 1995, 6, 135–140; Braun, D., Wider, G. & Wüthrich, K., *J. Am. Chem. Soc.*
//!    1994, 116, 8466–8469; Schwarzinger, S. et al., *J. Am. Chem. Soc.* 2001, 123, 2970–2978; and
//!    MacArthur, M. W. & Thornton, J. M., *J. Mol. Biol.* 1991, 218, 397–412.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::dihedrals::{align_by_res_names, periodic_angle_diff_deg, random_coil_shifts};
use crate::noesy::{compute_chain_torsions, parse_pdb_first_model, PdbChain};
use crate::peaks::Peak;
use crate::priors::{
    covalent_atom_offset, delta_covalent, expected_covalent_atom_offset, logpdf, one_to_three,
    Priors, STANDARD_RESIDUES,
};
use crate::qc::{apply_dataset_qc_to_assignments, evaluate_dataset_qc, DatasetQcReport};
use crate::rng::Rng;
use crate::solver::{
    candidate_llr_shifted, consensus_from_runs, pair_matrix, resolve_locked_assignments,
    AssignResult, Matrix2D, ScoreConfig, FORBID,
};
use crate::spinsystems::{populate_root_heights, SpinSystem, Tolerances};

/// Estimated uniform chemical-shift reference offsets `(obs_ppm - dss_prior_ppm)` per backbone nucleus.
///
/// Applying these offsets shifts `Priors` into the experimental spectrum's ppm frame while leaving
/// all `SpinSystem` shifts and emitted resonance assignments strictly in the experimental frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReferenceOffsets {
    pub d_ca: f64,
    pub d_cb: f64,
    pub d_c: f64,
    pub d_h: f64,
    pub d_n: f64,
}

impl ReferenceOffsets {
    #[inline]
    pub fn atom_offset(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.d_ca,
            "CB" => self.d_cb,
            "C" => self.d_c,
            "H" => self.d_h,
            "N" => self.d_n,
            _ => 0.0,
        }
    }
}

/// Backbone `(phi_deg, psi_deg)` torsion pair for a single residue (`None` when undefined).
pub type ResidueTorsion = (Option<f64>, Option<f64>);

/// Configuration for stoichiometry-guided peak filtering, spin-system assembly, two-pass
/// reference alignment, hybrid evolutionary + SA optimization, and graph self-consistency.
#[derive(Clone, Debug)]
pub struct EvoGraphConfig<'a> {
    /// `(H, N)` strip attachment and `13C` cross-experiment matching tolerances.
    pub tolerances: Tolerances,
    /// Maximum competing carbon clusters retained per `(atom, offset)` on each spin system.
    pub max_cands: usize,
    /// Underlying LLR typing and Gaussian connectivity scoring parameters.
    pub score_cfg: ScoreConfig,
    /// Global per-spectrum peak budget multiplier (`~1.4..=1.8 * N_exp(exp, sequence)`).
    pub stoichiometry_budget_mult: f64,
    /// Backbone amide `1H` chemical-shift window `(min_ppm, max_ppm)` applied when peak heights exist.
    pub hsqc_h_window: (f64, f64),
    /// Aliphatic `13C` (`CA`/`CB`) physical shift window `(min_ppm, max_ppm)` when peak heights exist.
    pub ca_cb_window: (f64, f64),
    /// Carbonyl `13C'` (`CO`) physical shift window `(min_ppm, max_ppm)` when peak heights exist.
    pub co_window: (f64, f64),
    /// Minimum per-placement null penalty preventing noise roots from hijacking exchange-broadened slots.
    pub min_assign_cost: f64,
    /// Total population size `M` evolved across islands (`e.g. 64`).
    pub population_size: usize,
    /// Number of independent evolutionary islands used for calibrated consensus confidence (`e.g. 16`).
    pub num_islands: usize,
    /// Number of evolutionary fragment-crossover + local-annealing generations per pass (`4..=6`).
    pub generations: usize,
    /// Local simulated annealing steps per spin system per generation.
    pub steps_per_system_per_gen: usize,
    /// Initial simulated annealing temperature in Generation 0.
    pub t_start: f64,
    /// Final simulated annealing temperature at the end of each local sweep.
    pub t_end: f64,
    /// Maximum contiguous block length for block-swap SA moves.
    pub max_block: usize,
    /// Tournament selection size within each evolutionary island.
    pub tournament_size: usize,
    /// Probability of applying sequential fragment crossover when generating an offspring.
    pub crossover_rate: f64,
    /// Minimum sequential connectivity score `P(s_k, s_{k+1})` defining a contiguous crossover stretch.
    pub fragment_link_threshold: f64,
    /// Enable two-pass automatic chemical-shift reference offset estimation (Mixture -> Anchor).
    pub auto_reference: bool,
    /// Minimum Pass-1 consensus (`>= 0.70`) for an assigned pair to serve as a Pass-2 reference anchor.
    pub anchor_consensus_threshold: f64,
    /// Enable 2-layer message-passing graph self-consistency check between Pass 1 and Pass 2.
    pub graph_consistency: bool,
    /// Automatically evaluate per-dataset QC (`DatasetQcReport`) and apply confidence calibration / abstention.
    pub apply_qc_calibration: bool,
    /// Optional 3D backbone `(phi_deg, psi_deg)` torsions aligned to `sequence` (`None` = sequence-only).
    pub structure_torsions: Option<&'a [(Option<f64>, Option<f64>)]>,
    /// Optional user-locked `(spin_system_idx, seq_idx)` hard assignment constraints
    /// (merged with `score_cfg.locked_assignments`; when calling `assign_evo_graph_from_peaks`,
    /// `spin_system_idx` indexes raw `HSQC` roots in `peaks`).
    pub locked_assignments: Vec<(usize, usize)>,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for EvoGraphConfig<'_> {
    fn default() -> Self {
        Self {
            tolerances: Tolerances {
                h: 0.035,
                n: 0.35,
                match_ca: 0.25,
                match_cb: 0.35,
                match_c: 0.25,
            },
            max_cands: 3,
            score_cfg: ScoreConfig::default(),
            stoichiometry_budget_mult: 1.60,
            hsqc_h_window: (5.5, 11.5),
            ca_cb_window: (8.0, 76.0),
            co_window: (165.5, 184.0),
            min_assign_cost: 1.5,
            population_size: 64,
            num_islands: 16,
            generations: 5,
            steps_per_system_per_gen: 450,
            t_start: 8.0,
            t_end: 0.04,
            max_block: 8,
            tournament_size: 3,
            crossover_rate: 0.80,
            fragment_link_threshold: 1.5,
            auto_reference: true,
            anchor_consensus_threshold: 0.70,
            graph_consistency: true,
            apply_qc_calibration: true,
            structure_torsions: None,
            locked_assignments: Vec::new(),
            seed: 0,
        }
    }
}

fn merged_config_locks(cfg: &EvoGraphConfig<'_>) -> Vec<(usize, usize)> {
    if cfg.locked_assignments.is_empty() && cfg.score_cfg.locked_assignments.is_empty() {
        return Vec::new();
    }
    let mut out =
        Vec::with_capacity(cfg.locked_assignments.len() + cfg.score_cfg.locked_assignments.len());
    out.extend_from_slice(&cfg.locked_assignments);
    out.extend_from_slice(&cfg.score_cfg.locked_assignments);
    out
}

/// Diagnostic telemetry produced by stoichiometric filtering, two-pass reference estimation,
/// residue-graph self-consistency, and unlabelled per-dataset quality control.
#[derive(Clone, Debug, Default)]
pub struct EvoGraphReport {
    pub raw_peak_count: usize,
    pub filtered_peak_count: usize,
    pub raw_hsqc_count: usize,
    pub filtered_hsqc_count: usize,
    pub nh2_removed_count: usize,
    /// Indices (`0..raw_hsqc_count`) of the input `HSQC` peaks retained as `SpinSystem` roots.
    pub kept_hsqc_indices: Vec<usize>,
    pub expected_peaks_by_exp: BTreeMap<String, usize>,
    pub kept_peaks_by_exp: BTreeMap<String, usize>,
    pub pass1_offsets: ReferenceOffsets,
    pub final_offsets: ReferenceOffsets,
    pub pass2_anchor_count: usize,
    pub graph_anchored_count: usize,
    pub graph_pruned_count: usize,
    pub pass1_best_score: f64,
    pub final_best_score: f64,
    pub qc_report: DatasetQcReport,
}

impl EvoGraphReport {
    /// Return a copy of `base` priors shifted into the experimental spectrum's ppm frame.
    pub fn shifted_priors(&self, base: &Priors) -> Priors {
        apply_offsets_to_priors(base, &self.final_offsets)
    }
}

/// Shift `Priors` means by `offsets` (`mu_exp = mu_dss + offset`) so scoring operates in the
/// experimental spectrum's ppm frame without mutating peak or spin-system coordinates.
pub fn apply_offsets_to_priors(priors: &Priors, offsets: &ReferenceOffsets) -> Priors {
    let mut table = priors.table.clone();
    for atoms in table.values_mut() {
        for (atom, stat) in atoms.iter_mut() {
            let d = offsets.atom_offset(atom);
            if d != 0.0 {
                stat.0 += d;
            }
        }
    }
    Priors { table }
}

fn normalize_sequence(sequence: &[String]) -> Vec<String> {
    sequence
        .iter()
        .map(|r| {
            let t = r.trim();
            if t.len() == 1 {
                one_to_three(t.chars().next().unwrap_or('X')).to_string()
            } else {
                t.to_ascii_uppercase()
            }
        })
        .collect()
}

fn median_f64(mut vals: Vec<f64>) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some(0.5 * (vals[mid - 1] + vals[mid]))
    }
}

#[derive(Clone, Copy, Debug)]
struct SequenceStoichiometry {
    n_amide: usize,
    n_asn_gln: usize,
    n_non_gly_intra: usize,
    n_non_gly_prev: usize,
}

fn compute_stoichiometry(seq: &[String]) -> SequenceStoichiometry {
    if seq.is_empty() {
        return SequenceStoichiometry {
            n_amide: 64,
            n_asn_gln: 6,
            n_non_gly_intra: 58,
            n_non_gly_prev: 58,
        };
    }
    let mut n_amide = 0usize;
    let mut n_non_gly_intra = 0usize;
    let mut n_non_gly_prev = 0usize;
    for k in 1..seq.len() {
        if seq[k] != "PRO" {
            n_amide += 1;
            if seq[k] != "GLY" {
                n_non_gly_intra += 1;
            }
            if seq[k - 1] != "GLY" {
                n_non_gly_prev += 1;
            }
        }
    }
    let n_asn_gln = seq
        .iter()
        .filter(|r| r.as_str() == "ASN" || r.as_str() == "GLN")
        .count();
    SequenceStoichiometry {
        n_amide: n_amide.max(1),
        n_asn_gln,
        n_non_gly_intra: n_non_gly_intra.max(1),
        n_non_gly_prev: n_non_gly_prev.max(1),
    }
}

fn expected_signed_peaks(exp: &str, sign: i32, st: &SequenceStoichiometry) -> usize {
    match exp {
        "HSQC" => st.n_amide + 2 * st.n_asn_gln,
        "HNCO" => st.n_amide + 2 * st.n_asn_gln,
        "HNCACO" => 2 * st.n_amide,
        "HNCA" => 2 * st.n_amide,
        "HNCOCA" => st.n_amide,
        "CBCACONH" => st.n_amide + st.n_non_gly_prev,
        "HNCACB" => {
            if sign >= 0 {
                2 * st.n_amide
            } else {
                st.n_non_gly_intra + st.n_non_gly_prev
            }
        }
        _ => 2 * st.n_amide,
    }
}

/// Expected number of peaks $N_{\text{exp}}(\text{exp}, \text{sequence})$ predicted directly from
/// covalent magnetization-transfer pathways along `sequence` (Schmidt & Güntert, *JACS* 2012).
pub fn expected_experiment_peaks(exp: &str, sequence: &[String]) -> usize {
    let norm = normalize_sequence(sequence);
    let st = compute_stoichiometry(&norm);
    let mut up = exp.trim().to_ascii_uppercase().replace(['(', ')'], "");
    let sign_override = if let Some(base) = up.strip_suffix('+') {
        up = base.replace('-', "");
        Some(1)
    } else if let Some(base) = up.strip_suffix('-') {
        up = base.replace('-', "");
        Some(-1)
    } else {
        up = up.replace('-', "");
        None
    };
    match (up.as_str(), sign_override) {
        ("HNCACB", None) => {
            expected_signed_peaks("HNCACB", 1, &st) + expected_signed_peaks("HNCACB", -1, &st)
        }
        ("HNCACB", Some(sgn)) => expected_signed_peaks("HNCACB", sgn, &st),
        (other, _) => expected_signed_peaks(other, 1, &st),
    }
}

#[inline]
fn per_strip_cap(exp: &str) -> usize {
    match exp {
        "HNCO" | "HNCOCA" => 2,
        "HNCA" | "HNCACO" | "CBCACONH" | "HNCACB" => 4,
        _ => 4,
    }
}

#[inline]
fn canonical_exp_key(exp: &str, sign: i32) -> (&'static str, i32) {
    match exp {
        "HSQC" => ("HSQC", 1),
        "HNCA" => ("HNCA", 1),
        "HNCOCA" => ("HNCOCA", 1),
        "CBCACONH" => ("CBCACONH", 1),
        "HNCO" => ("HNCO", 1),
        "HNCACO" => ("HNCACO", 1),
        "HNCACB" => {
            if sign >= 0 {
                ("HNCACB", 1)
            } else {
                ("HNCACB", -1)
            }
        }
        _ => ("OTHER", 1),
    }
}

#[inline]
fn exp_channel_label(exp: &str, sign: i32) -> &'static str {
    match (exp, sign >= 0) {
        ("HSQC", _) => "HSQC",
        ("HNCA", _) => "HNCA",
        ("HNCOCA", _) => "HNCOCA",
        ("CBCACONH", _) => "CBCACONH",
        ("HNCO", _) => "HNCO",
        ("HNCACO", _) => "HNCACO",
        ("HNCACB", true) => "HNCACB(+)",
        ("HNCACB", false) => "HNCACB(-)",
        _ => "OTHER",
    }
}

/// Filter raw picked 2D/3D peaks using sequence-derived stoichiometry budgets, physical chemical-shift
/// windows, indirect-axis sinc/ridge artifact suppression, and paired `NH2` doublet removal.
///
/// Preserves legacy pass-through behavior when peak `height` metadata is absent (`None`).
pub fn filter_peaks_stoichiometric(
    peaks: &[Peak],
    sequence: &[String],
    cfg: &EvoGraphConfig<'_>,
) -> Vec<Peak> {
    let mut dummy = EvoGraphReport::default();
    filter_peaks_with_report(peaks, sequence, cfg, &mut dummy)
}

fn filter_peaks_with_report(
    peaks: &[Peak],
    sequence: &[String],
    cfg: &EvoGraphConfig<'_>,
    report: &mut EvoGraphReport,
) -> Vec<Peak> {
    let norm_seq = normalize_sequence(sequence);
    let st = compute_stoichiometry(&norm_seq);

    report.raw_peak_count = peaks.len();
    report.raw_hsqc_count = peaks.iter().filter(|p| p.exp == "HSQC").count();

    let has_heights = peaks.iter().any(|p| p.height.is_some());
    if !has_heights {
        report.filtered_peak_count = peaks.len();
        report.filtered_hsqc_count = report.raw_hsqc_count;
        report.kept_hsqc_indices = (0..report.raw_hsqc_count).collect();
        return peaks.to_vec();
    }

    let raw_locks = merged_config_locks(cfg);
    let locked_raw_ords: BTreeSet<usize> = if raw_locks.is_empty() {
        BTreeSet::new()
    } else {
        resolve_locked_assignments(report.raw_hsqc_count, &norm_seq, &raw_locks)
            .into_iter()
            .map(|(s, _)| s)
            .collect()
    };

    // 1. Physical chemical-shift window gating
    let mut gated_hsqc: Vec<(usize, Peak)> = Vec::new();
    let mut gated_3d: BTreeMap<(&'static str, i32), Vec<Peak>> = BTreeMap::new();
    let mut other_peaks: Vec<Peak> = Vec::new();
    let mut raw_hsqc_ord = 0usize;

    for p in peaks {
        if p.exp == "HSQC" {
            let ord = raw_hsqc_ord;
            raw_hsqc_ord += 1;
            if locked_raw_ords.contains(&ord)
                || (p.h >= cfg.hsqc_h_window.0
                    && p.h <= cfg.hsqc_h_window.1
                    && p.n >= 96.0
                    && p.n <= 142.0)
            {
                gated_hsqc.push((ord, p.clone()));
            }
            continue;
        }

        let (ekey, sgn) = canonical_exp_key(&p.exp, p.sign);
        if ekey == "OTHER" {
            other_peaks.push(p.clone());
            continue;
        }
        let Some(c) = p.c else {
            continue;
        };
        let in_window = match ekey {
            "HNCO" | "HNCACO" => c >= cfg.co_window.0 && c <= cfg.co_window.1,
            "HNCA" | "HNCOCA" => c >= 35.0_f64.max(cfg.ca_cb_window.0) && c <= cfg.ca_cb_window.1,
            "CBCACONH" => c >= cfg.ca_cb_window.0 && c <= cfg.ca_cb_window.1,
            "HNCACB" => {
                if sgn > 0 {
                    // Positive HNCACB phase encodes CA only; values < 35 ppm are sinc sidelobes of negative CB.
                    c >= 35.0_f64.max(cfg.ca_cb_window.0) && c <= cfg.ca_cb_window.1
                } else {
                    c >= cfg.ca_cb_window.0 && c <= cfg.ca_cb_window.1
                }
            }
            _ => true,
        };
        if in_window {
            gated_3d.entry((ekey, sgn)).or_default().push(p.clone());
        }
    }

    // 2. Stoichiometric signal intensity scale I_hat(E) = median(top N_exp heights)
    let n_exp_hsqc = expected_signed_peaks("HSQC", 1, &st);
    report
        .expected_peaks_by_exp
        .insert("HSQC".to_string(), n_exp_hsqc);
    let hsqc_ref_height = {
        let mut hs: Vec<f64> = gated_hsqc
            .iter()
            .filter_map(|(_, p)| p.height.map(f64::abs))
            .collect();
        hs.sort_by(|a, b| b.total_cmp(a));
        hs.truncate(n_exp_hsqc.max(1));
        median_f64(hs).unwrap_or(1.0).max(1e-9)
    };

    let mut ref_height_3d: BTreeMap<(&'static str, i32), f64> = BTreeMap::new();
    let mut noise_floor_3d: BTreeMap<(&'static str, i32), f64> = BTreeMap::new();
    for (&(ekey, sgn), list) in &gated_3d {
        let n_exp = expected_signed_peaks(ekey, sgn, &st);
        report
            .expected_peaks_by_exp
            .insert(exp_channel_label(ekey, sgn).to_string(), n_exp);
        let mut hs: Vec<f64> = list.iter().filter_map(|p| p.height.map(f64::abs)).collect();
        hs.sort_by(|a, b| b.total_cmp(a));
        // If raw picks significantly exceed stoichiometry, the bottom quartile estimates the picker's k*MAD floor
        let p_floor = if hs.len() > (n_exp * 3) / 2 {
            hs[hs.len() * 4 / 5]
        } else {
            0.0
        };
        noise_floor_3d.insert((ekey, sgn), p_floor);
        // Use top (2/3 * N_exp) so partial-transfer pathways (e.g. HNCACO) anchor on true signals, not noise
        let ref_k = ((n_exp * 2) / 3).max(4).min(hs.len().max(1));
        hs.truncate(ref_k);
        let med = median_f64(hs).unwrap_or(1.0).max(1e-9);
        ref_height_3d.insert((ekey, sgn), med);
    }

    // 2b. Inter-spectrum (1H, 15N) alignment of 3D channels to strong HSQC roots (correcting multi-session sample drift)
    if gated_hsqc.len() >= 12 {
        let mut strong_roots: Vec<(f64, f64, f64)> = gated_hsqc
            .iter()
            .map(|(_, p)| (p.h, p.n, p.height.map(f64::abs).unwrap_or(0.0)))
            .collect();
        strong_roots.sort_by(|a, b| b.2.total_cmp(&a.2));
        strong_roots.truncate(n_exp_hsqc.max(12));

        for (&(ekey, sgn), list) in &mut gated_3d {
            let iref = *ref_height_3d.get(&(ekey, sgn)).unwrap_or(&1.0);
            let mut dhs = Vec::new();
            let mut dns = Vec::new();
            for p in list.iter() {
                if p.height.map(f64::abs).unwrap_or(0.0) < 0.35 * iref {
                    continue;
                }
                let mut best: Option<(f64, f64, f64)> = None;
                for &(rh, rn, _) in &strong_roots {
                    let d = ((p.h - rh) / 0.055).hypot((p.n - rn) / 0.55);
                    if d <= 1.0 && best.is_none_or(|b| d < b.0) {
                        best = Some((d, p.h - rh, p.n - rn));
                    }
                }
                if let Some((_, dh, dn)) = best {
                    dhs.push(dh);
                    dns.push(dn);
                }
            }
            if dhs.len() >= 12 {
                let mh = median_f64(dhs).unwrap_or(0.0);
                let mn = median_f64(dns).unwrap_or(0.0);
                let adj_h = if mh.abs() >= 0.004 && mh.abs() <= 0.045 {
                    mh
                } else {
                    0.0
                };
                let adj_n = if mn.abs() >= 0.04 && mn.abs() <= 0.45 {
                    mn
                } else {
                    0.0
                };
                if adj_h != 0.0 || adj_n != 0.0 {
                    for p in list.iter_mut() {
                        p.h -= adj_h;
                        p.n -= adj_n;
                    }
                }
            }
        }
    }

    // 2c. Reject uncorroborated noise-only HNCACO channel when HNCO is present:
    // a genuine HN(CA)CO acquisition observes C'(i-1) alongside C'(i), whereas an empty/failed
    // cube picked near k*MAD exhibits a flat truncated-Gaussian tail (top/p80 < 1.80) and near-zero
    // (< 8%) strip-level C'(i-1) overlap with HNCO.
    if let (Some(hncaco_list), Some(hnco_list)) = (
        gated_3d.get(&("HNCACO", 1)).cloned(),
        gated_3d.get(&("HNCO", 1)),
    ) {
        if hncaco_list.len() >= 16 && hnco_list.len() >= 20 {
            let mut hs: Vec<f64> = hncaco_list
                .iter()
                .filter_map(|p| p.height.map(f64::abs))
                .collect();
            hs.sort_by(|a, b| b.total_cmp(a));
            if hs.len() >= 16 {
                let top_h = hs[0];
                let p80_h = hs[hs.len() * 4 / 5].max(1e-9);
                if top_h / p80_h < 1.80 {
                    let matched = hncaco_list
                        .iter()
                        .filter(|p| {
                            let pc = p.c.unwrap_or(0.0);
                            hnco_list.iter().any(|q| {
                                let qc = q.c.unwrap_or(0.0);
                                (p.h - q.h).abs() <= 0.045
                                    && (p.n - q.n).abs() <= 0.45
                                    && (pc - qc).abs() <= 0.30
                            })
                        })
                        .count();
                    if (matched as f64) < 0.08 * (hncaco_list.len() as f64) {
                        gated_3d.remove(&("HNCACO", 1));
                    }
                }
            }
        }
    }

    // 3. Cross-sign F1 sinc sidelobe suppression in phase-sensitive HNCACB
    if let (Some(pos_list), Some(neg_list)) = (
        gated_3d.get(&("HNCACB", 1)).cloned(),
        gated_3d.get(&("HNCACB", -1)).cloned(),
    ) {
        let filter_cross_sign = |target: Vec<Peak>, opp: &[Peak]| -> Vec<Peak> {
            target
                .into_iter()
                .filter(|q| {
                    let hq = q.height.map(f64::abs).unwrap_or(0.0);
                    let cq = q.c.unwrap_or(0.0);
                    !opp.iter().any(|p| {
                        let hp = p.height.map(f64::abs).unwrap_or(0.0);
                        let cp = p.c.unwrap_or(0.0);
                        hp > hq
                            && (q.h - p.h).abs() <= 0.035
                            && (q.n - p.n).abs() <= 0.35
                            && (cq - cp).abs() <= 0.65
                            && hq < 0.16 * hp
                    })
                })
                .collect()
        };
        let clean_pos = filter_cross_sign(pos_list.clone(), &neg_list);
        let clean_neg = filter_cross_sign(neg_list, &pos_list);
        gated_3d.insert(("HNCACB", 1), clean_pos);
        gated_3d.insert(("HNCACB", -1), clean_neg);
    }

    // 4. Same-channel 3D F1/F2/F3 sinc sidelobe & strip-ridge suppression
    let mut cleaned_3d: BTreeMap<(&'static str, i32), Vec<Peak>> = BTreeMap::new();
    for ((ekey, sgn), mut list) in gated_3d {
        let iref = *ref_height_3d.get(&(ekey, sgn)).unwrap_or(&1.0);
        let p_floor = *noise_floor_3d.get(&(ekey, sgn)).unwrap_or(&0.0);
        list.sort_by(|a, b| {
            let ha = a.height.map(f64::abs).unwrap_or(0.0);
            let hb = b.height.map(f64::abs).unwrap_or(0.0);
            hb.total_cmp(&ha)
        });
        let single_peak_strip = matches!(ekey, "HNCO" | "HNCOCA");
        let sinc_dc = if matches!(ekey, "HNCO" | "HNCACO") {
            0.55
        } else {
            0.60
        };
        let min_norm = if ekey == "HNCACO" && p_floor > 0.0 {
            0.09
        } else {
            0.035
        };

        let mut kept: Vec<Peak> = Vec::with_capacity(list.len());
        for q in list {
            let hq = q.height.map(f64::abs).unwrap_or(0.0);
            let norm_hq = hq / iref;
            if norm_hq < min_norm || (p_floor > 0.0 && hq < 1.35 * p_floor) {
                continue;
            }
            let cq = q.c.unwrap_or(0.0);
            let mut is_artifact = false;
            for p in &kept {
                let hp = p.height.map(f64::abs).unwrap_or(0.0);
                let cp = p.c.unwrap_or(0.0);
                let dh = (q.h - p.h).abs();
                let dn = (q.n - p.n).abs();
                let dc = (cq - cp).abs();

                // Same-(H, N) strip F1 sidelobes and t1 strip floor
                if dh <= 0.032 && dn <= 0.32 {
                    if single_peak_strip {
                        if (dc <= 1.80 && hq < 0.24 * hp) || hq < 0.10 * hp {
                            is_artifact = true;
                            break;
                        }
                    } else if (dc <= sinc_dc && hq < 0.15 * hp) || hq < 0.05 * hp {
                        is_artifact = true;
                        break;
                    }
                }

                // Indirect F2 (15N) truncation sinc ripple at identical (1H, 13C)
                if dh <= 0.025 && dc <= 0.18 && dn <= 2.00 && hq < 0.12 * hp {
                    is_artifact = true;
                    break;
                }

                // Direct/indirect F3 (1H) ripple at identical (15N, 13C)
                if dn <= 0.22 && dc <= 0.18 && dh <= 0.20 && hq < 0.10 * hp {
                    is_artifact = true;
                    break;
                }
            }
            if !is_artifact {
                kept.push(q);
            }
        }
        cleaned_3d.insert((ekey, sgn), kept);
    }

    // 5. Evaluate 3D corroboration per HSQC root candidate
    let has_any_3d = cleaned_3d.values().any(|v| !v.is_empty());
    let has_i_minus_1_ca = cleaned_3d
        .get(&("CBCACONH", 1))
        .is_some_and(|v| !v.is_empty())
        || cleaned_3d
            .get(&("HNCOCA", 1))
            .is_some_and(|v| !v.is_empty());

    #[derive(Clone, Debug)]
    struct Root3DSummary {
        distinct_exps: usize,
        sum_best_norm_h: f64,
        has_intra_ca: bool,
        strong_carbons: Vec<(f64, &'static str)>,
    }

    let tol_h = cfg.tolerances.h;
    let tol_n = cfg.tolerances.n;

    let summarize_root_3d = |rh: f64, rn: f64| -> Root3DSummary {
        let mut exps_seen: BTreeSet<&'static str> = BTreeSet::new();
        let mut best_by_exp: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut strong_carbons: Vec<(f64, &'static str)> = Vec::new();
        let mut im1_ca_vals: Vec<f64> = Vec::new();
        let mut both_ca_vals: Vec<(f64, f64)> = Vec::new();

        for (&(ekey, sgn), list) in &cleaned_3d {
            let iref = *ref_height_3d.get(&(ekey, sgn)).unwrap_or(&1.0);
            for p in list {
                let d = ((rh - p.h) / tol_h).hypot((rn - p.n) / tol_n);
                if d > 1.0 {
                    continue;
                }
                let nh = (p.height.map(f64::abs).unwrap_or(0.0) / iref) * (-0.30 * d * d).exp();
                if nh >= 0.05 {
                    exps_seen.insert(ekey);
                    let entry = best_by_exp.entry(ekey).or_insert(0.0);
                    if nh > *entry {
                        *entry = nh;
                    }
                    if let Some(c) = p.c {
                        if nh >= 0.12 {
                            strong_carbons.push((c, ekey));
                        }
                        if (ekey == "CBCACONH" || ekey == "HNCOCA") && (37.0..=74.0).contains(&c) {
                            im1_ca_vals.push(c);
                        }
                        if (ekey == "HNCA" || (ekey == "HNCACB" && sgn > 0))
                            && (37.0..=74.0).contains(&c)
                            && nh >= 0.08
                        {
                            both_ca_vals.push((c, nh));
                        }
                    }
                }
            }
        }

        let has_intra_ca = if both_ca_vals.is_empty() {
            false
        } else if !has_i_minus_1_ca {
            both_ca_vals.iter().any(|&(_, nh)| nh >= 0.10)
        } else {
            // Intra-residue CA(i) exists either if a CA peak does not match i-1 CA, or if a superposed
            // degenerate peak has elevated amplitude (>= 1.4x typical single peak).
            both_ca_vals.iter().any(|&(c, nh)| {
                let matches_im1 = im1_ca_vals.iter().any(|&cref| (c - cref).abs() <= 0.30);
                !matches_im1 || nh >= 1.40
            })
        };

        let sum_best_norm_h: f64 = best_by_exp.values().map(|&v| v.min(2.0)).sum();
        Root3DSummary {
            distinct_exps: exps_seen.len(),
            sum_best_norm_h,
            has_intra_ca,
            strong_carbons,
        }
    };

    // 6. Sort HSQC roots by descending amplitude and suppress 2D sinc/t1/F2 ripples
    gated_hsqc.sort_by(|(_, a), (_, b)| {
        let ha = a.height.map(f64::abs).unwrap_or(0.0);
        let hb = b.height.map(f64::abs).unwrap_or(0.0);
        hb.total_cmp(&ha)
    });

    let summaries: Vec<Root3DSummary> = gated_hsqc
        .iter()
        .map(|(_, r)| summarize_root_3d(r.h, r.n))
        .collect();

    let has_independent_3d = |q_sum: &Root3DSummary, p_sum: &Root3DSummary| -> bool {
        if q_sum.distinct_exps == 0 || q_sum.sum_best_norm_h < 0.35 {
            return false;
        }
        let distinct_c = q_sum
            .strong_carbons
            .iter()
            .filter(|&&(cq, eq)| {
                !p_sum
                    .strong_carbons
                    .iter()
                    .any(|&(cp, ep)| eq == ep && (cq - cp).abs() <= 0.30)
            })
            .count();
        distinct_c >= 1 && (q_sum.distinct_exps >= 2 || q_sum.sum_best_norm_h >= 0.65)
    };

    let mut kept_roots_idx: Vec<usize> = Vec::with_capacity(gated_hsqc.len());
    for i in 0..gated_hsqc.len() {
        let (ord_q, ref rq) = gated_hsqc[i];
        if locked_raw_ords.contains(&ord_q) {
            kept_roots_idx.push(i);
            continue;
        }
        let hq = rq.height.map(f64::abs).unwrap_or(0.0);
        let norm_hq = hq / hsqc_ref_height;
        let q_sum = &summaries[i];

        if has_any_3d && norm_hq < 0.04 && q_sum.sum_best_norm_h < 0.12 {
            continue;
        }
        if has_any_3d && q_sum.distinct_exps == 0 && norm_hq < 0.20 {
            continue;
        }

        let mut suppressed = false;
        for &j in &kept_roots_idx {
            let (_, ref rp) = gated_hsqc[j];
            let hp = rp.height.map(f64::abs).unwrap_or(0.0);
            let dh = (rq.h - rp.h).abs();
            let dn = (rq.n - rp.n).abs();
            let p_sum = &summaries[j];

            // 3J_HNHa scalar-split doublet lobe or close 2D shoulder sharing identical 3D carbon strip
            if dh <= 0.022 && dn <= 0.16 && !has_independent_3d(q_sum, p_sum) {
                suppressed = true;
                break;
            }
            // Indirect F1 (15N) sinc sidelobe & t1 column noise ridge along constant 1H
            if dh <= 0.028
                && ((dn <= 1.65 && hq < 0.15 * hp) || hq < 0.055 * hp)
                && !has_independent_3d(q_sum, p_sum)
            {
                suppressed = true;
                break;
            }
            // Direct F2 (1H) sinc/baseline ripple train along constant 15N
            if dn <= 0.22
                && ((dh <= 0.90 && hq < 0.10 * hp) || hq < 0.05 * hp)
                && !has_independent_3d(q_sum, p_sum)
            {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            kept_roots_idx.push(i);
        }
    }

    // 7. Detect and remove paired Asn/Gln sidechain NH2 doublets:
    // |dN| <= 0.22 ppm (~1.5 indirect grid pts), 0.30 <= |dH| <= 1.20 ppm, N <= 120.5 ppm, lacking intra CA(i).
    let mut nh2_pairs: Vec<(f64, usize, usize)> = Vec::new();
    for pos_a in 0..kept_roots_idx.len() {
        let ia = kept_roots_idx[pos_a];
        if summaries[ia].has_intra_ca || gated_hsqc[ia].1.n > 120.5 {
            continue;
        }
        for &ib in &kept_roots_idx[(pos_a + 1)..] {
            if summaries[ib].has_intra_ca || gated_hsqc[ib].1.n > 120.5 {
                continue;
            }
            let ra = &gated_hsqc[ia].1;
            let rb = &gated_hsqc[ib].1;
            let dn = (ra.n - rb.n).abs();
            let dh = (ra.h - rb.h).abs();
            if dn <= 0.22 && (0.30..=1.20).contains(&dh) {
                let ha = ra.height.map(f64::abs).unwrap_or(1.0);
                let hb = rb.height.map(f64::abs).unwrap_or(1.0);
                let sym = ha.min(hb) / ha.max(hb).max(1e-9);
                if sym >= 0.20 {
                    let n_mid = 0.5 * (ra.n + rb.n);
                    let score = sym * (-0.5 * ((n_mid - 112.5) / 4.0).powi(2)).exp() / (dn + 0.02);
                    nh2_pairs.push((score, ia, ib));
                }
            }
        }
    }
    nh2_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let max_nh2_pairs = st.n_asn_gln + 6;
    let mut removed_nh2: BTreeSet<usize> = BTreeSet::new();
    let mut pairs_taken = 0usize;
    for (_, ia, ib) in nh2_pairs {
        if pairs_taken >= max_nh2_pairs {
            break;
        }
        if !removed_nh2.contains(&ia) && !removed_nh2.contains(&ib) {
            removed_nh2.insert(ia);
            removed_nh2.insert(ib);
            pairs_taken += 1;
        }
    }
    report.nh2_removed_count = removed_nh2.len();
    kept_roots_idx
        .retain(|&idx| locked_raw_ords.contains(&gated_hsqc[idx].0) || !removed_nh2.contains(&idx));

    // Drop roots that have zero 3D peaks when 3D experiments exist
    if has_any_3d {
        kept_roots_idx.retain(|&idx| {
            locked_raw_ords.contains(&gated_hsqc[idx].0) || summaries[idx].distinct_exps > 0
        });
    }

    // 8. Rank surviving HSQC roots by corroborated SNR + intra-CA presence and cap at post-NH2 budget
    kept_roots_idx.sort_by(|&ia, &ib| {
        let bonus_a = if summaries[ia].has_intra_ca {
            1.35
        } else {
            1.0
        };
        let bonus_b = if summaries[ib].has_intra_ca {
            1.35
        } else {
            1.0
        };
        let sa = (gated_hsqc[ia].1.height.map(f64::abs).unwrap_or(0.0) / hsqc_ref_height)
            * (1.0 + 0.90 * summaries[ia].sum_best_norm_h)
            * bonus_a;
        let sb = (gated_hsqc[ib].1.height.map(f64::abs).unwrap_or(0.0) / hsqc_ref_height)
            * (1.0 + 0.90 * summaries[ib].sum_best_norm_h)
            * bonus_b;
        sb.total_cmp(&sa)
    });
    let rem_expected_hsqc = n_exp_hsqc.saturating_sub(removed_nh2.len()).max(st.n_amide);
    let root_mult = (cfg.stoichiometry_budget_mult * 0.85).max(1.25);
    let hsqc_cap = ((root_mult * (rem_expected_hsqc as f64)).ceil() as usize).max(1);
    if locked_raw_ords.is_empty() {
        kept_roots_idx.truncate(hsqc_cap);
    } else if kept_roots_idx.len() > hsqc_cap {
        let mut preserved: Vec<usize> = Vec::with_capacity(hsqc_cap + locked_raw_ords.len());
        for &idx in &kept_roots_idx {
            if preserved.len() < hsqc_cap || locked_raw_ords.contains(&gated_hsqc[idx].0) {
                preserved.push(idx);
            }
        }
        kept_roots_idx = preserved;
    }
    // Restore original HSQC peak order for deterministic SpinSystem indexing
    kept_roots_idx.sort_by_key(|&idx| gated_hsqc[idx].0);

    let final_roots: Vec<Peak> = kept_roots_idx
        .iter()
        .map(|&idx| gated_hsqc[idx].1.clone())
        .collect();
    report.kept_hsqc_indices = kept_roots_idx
        .iter()
        .map(|&idx| gated_hsqc[idx].0)
        .collect();
    report.filtered_hsqc_count = final_roots.len();
    report
        .kept_peaks_by_exp
        .insert("HSQC".to_string(), final_roots.len());

    // 9. Per-strip top-M_exp filtering + primary-slot-protected global stoichiometric budget cap
    let mut out_peaks: Vec<Peak> = Vec::with_capacity(final_roots.len() * 10);
    out_peaks.extend(final_roots.iter().cloned());

    for ((ekey, sgn), chan_peaks) in cleaned_3d {
        let iref = *ref_height_3d.get(&(ekey, sgn)).unwrap_or(&1.0);
        let m_cap = per_strip_cap(ekey);
        let primary_slots: usize = match ekey {
            "HNCO" | "HNCOCA" => 1,
            _ => 2,
        };
        let mut selected_scores: BTreeMap<usize, f64> = BTreeMap::new();

        for r in &final_roots {
            let mut strip_hits: Vec<(usize, f64)> = Vec::new();
            for (pi, p) in chan_peaks.iter().enumerate() {
                let d = ((r.h - p.h) / tol_h).hypot((r.n - p.n) / tol_n);
                if d <= 1.0 {
                    let nh = p.height.map(f64::abs).unwrap_or(0.0) / iref;
                    if nh >= 0.035 {
                        let sc = nh * (-0.55 * d * d).exp();
                        strip_hits.push((pi, sc));
                    }
                }
            }
            strip_hits.sort_by(|a, b| b.1.total_cmp(&a.1));
            strip_hits.truncate(m_cap);
            for (rank_idx, (pi, sc)) in strip_hits.into_iter().enumerate() {
                let rank_weight = if rank_idx < primary_slots { 2.5 } else { 0.35 };
                let weighted = sc * rank_weight;
                let prev = selected_scores.entry(pi).or_insert(0.0);
                if weighted > *prev {
                    *prev = weighted;
                }
            }
        }

        let mut ranked: Vec<(usize, f64)> = selected_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n_exp = expected_signed_peaks(ekey, sgn, &st);
        let chan_cap = ((cfg.stoichiometry_budget_mult * (n_exp as f64)).ceil() as usize).max(1);
        ranked.truncate(chan_cap);
        ranked.sort_by_key(|&(pi, _)| pi);

        report
            .kept_peaks_by_exp
            .insert(exp_channel_label(ekey, sgn).to_string(), ranked.len());
        for (pi, _) in ranked {
            out_peaks.push(chan_peaks[pi].clone());
        }
    }

    out_peaks.extend(other_peaks);
    report.filtered_peak_count = out_peaks.len();
    out_peaks
}

#[derive(Clone, Debug)]
struct StoichEntry {
    exp: &'static str,
    atom: Option<&'static str>,
    offset: Option<i32>,
    val: f64,
    hn_dist: f64,
    norm_height: Option<f64>,
    both_offsets: bool,
}

#[derive(Clone, Copy, Debug)]
struct StoichClusterPoint {
    exp: &'static str,
    val: f64,
    hn_dist: f64,
    norm_height: Option<f64>,
}

fn cluster_distinct_support(pts: &[StoichClusterPoint]) -> usize {
    let mut s: BTreeSet<&'static str> = BTreeSet::new();
    for p in pts {
        s.insert(p.exp);
    }
    s.len()
}

fn cluster_mean_dist(pts: &[StoichClusterPoint]) -> f64 {
    pts.iter().map(|p| p.hn_dist).sum::<f64>() / (pts.len() as f64)
}

fn cluster_weighted_val(pts: &[StoichClusterPoint]) -> f64 {
    if pts.iter().any(|p| p.norm_height.is_some()) {
        let mut num = 0.0;
        let mut den = 0.0;
        for p in pts {
            let w = p.norm_height.unwrap_or(0.1).max(1e-6) * (-0.35 * p.hn_dist * p.hn_dist).exp();
            num += w * p.val;
            den += w;
        }
        if den > 0.0 {
            return num / den;
        }
    }
    pts.iter().map(|p| p.val).sum::<f64>() / (pts.len() as f64)
}

fn cluster_prominence(pts: &[StoichClusterPoint]) -> f64 {
    pts.iter()
        .map(|p| p.norm_height.unwrap_or(0.0) * (-0.65 * p.hn_dist * p.hn_dist).exp())
        .fold(0.0_f64, f64::max)
}

fn cluster_stoich_points(points: &[StoichClusterPoint], tol: f64) -> Vec<Vec<StoichClusterPoint>> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.val.total_cmp(&b.val));
    let mut out: Vec<Vec<StoichClusterPoint>> = Vec::new();
    for p in sorted {
        if let Some(last) = out.last_mut() {
            let center = last.iter().map(|x| x.val).sum::<f64>() / (last.len() as f64);
            if p.val - center <= tol {
                last.push(p);
                continue;
            }
        }
        out.push(vec![p]);
    }
    out
}

/// Assemble intensity-weighted `SpinSystem`s where candidate `support` equals the number of
/// **distinct** corroborating experiments and candidate clusters are ranked by
/// `(distinct_exp_support, normalized_height, -hn_dist)` (falling back to exact legacy ranking
/// when peak heights are `None`).
pub fn build_stoichiometric_spin_systems(
    peaks: &[Peak],
    tol: &Tolerances,
    max_cands: usize,
) -> Vec<SpinSystem> {
    let roots: Vec<&Peak> = peaks.iter().filter(|p| p.exp == "HSQC").collect();
    if roots.is_empty() {
        return Vec::new();
    }

    let heights_present = peaks.iter().any(|p| p.exp != "HSQC" && p.height.is_some());

    // Compute per-channel reference height from top half of peaks (already filtered or raw)
    let mut chan_ref: BTreeMap<(&'static str, i32), f64> = BTreeMap::new();
    if heights_present {
        for &(ename, sgn) in &[
            ("HNCA", 1),
            ("HNCOCA", 1),
            ("CBCACONH", 1),
            ("HNCO", 1),
            ("HNCACO", 1),
            ("HNCACB", 1),
            ("HNCACB", -1),
        ] {
            let mut hs: Vec<f64> = peaks
                .iter()
                .filter(|p| canonical_exp_key(&p.exp, p.sign) == (ename, sgn))
                .filter_map(|p| p.height.map(f64::abs))
                .collect();
            if !hs.is_empty() {
                hs.sort_by(|a, b| b.total_cmp(a));
                let keep_n = hs.len().min((roots.len() * 2).max(8));
                hs.truncate(keep_n);
                if let Some(m) = median_f64(hs) {
                    chan_ref.insert((ename, sgn), m.max(1e-9));
                }
            }
        }
    }

    let mut attached: Vec<Vec<&Peak>> = vec![Vec::new(); roots.len()];
    for p in peaks {
        if p.exp == "HSQC" || p.c.is_none() {
            continue;
        }
        if !matches!(
            p.exp.as_str(),
            "HNCA" | "HNCOCA" | "HNCACB" | "CBCACONH" | "HNCO" | "HNCACO"
        ) {
            continue;
        }
        for (j, r) in roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
            if d <= 1.0 {
                attached[j].push(p);
            }
        }
    }

    let mut systems: Vec<SpinSystem> = roots
        .iter()
        .zip(attached.iter())
        .map(|(&r, plist)| {
            resolve_stoichiometric_root(r, plist, tol, max_cands, heights_present, &chan_ref)
        })
        .collect();
    populate_root_heights(&mut systems, &roots, &attached);
    systems
}

fn resolve_stoichiometric_root(
    root: &Peak,
    plist: &[&Peak],
    tol: &Tolerances,
    max_cands: usize,
    heights_present: bool,
    chan_ref: &BTreeMap<(&'static str, i32), f64>,
) -> SpinSystem {
    let hn_dist = |p: &Peak| ((root.h - p.h) / tol.h).hypot((root.n - p.n) / tol.n);
    let norm_h = |p: &Peak| -> Option<f64> {
        let h = p.height.map(f64::abs)?;
        let key = canonical_exp_key(&p.exp, p.sign);
        let iref = chan_ref.get(&key).copied().unwrap_or(1.0);
        Some(h / iref)
    };

    // Cap per-(experiment, sign_group) peaks attached to this strip when heights exist
    let mut capped_peaks: Vec<&Peak> = Vec::with_capacity(plist.len());
    if heights_present {
        for &(ename, sgn) in &[
            ("HNCA", 1),
            ("HNCOCA", 1),
            ("CBCACONH", 1),
            ("HNCO", 1),
            ("HNCACO", 1),
            ("HNCACB", 1),
            ("HNCACB", -1),
        ] {
            let mut group: Vec<(&Peak, f64)> = plist
                .iter()
                .copied()
                .filter(|p| canonical_exp_key(&p.exp, p.sign) == (ename, sgn))
                .map(|p| {
                    let d = hn_dist(p);
                    let sc = norm_h(p).unwrap_or(0.1) * (-0.5 * d * d).exp();
                    (p, sc)
                })
                .collect();
            group.sort_by(|a, b| b.1.total_cmp(&a.1));
            group.truncate(per_strip_cap(ename));
            capped_peaks.extend(group.into_iter().map(|(p, _)| p));
        }
    } else {
        capped_peaks.extend_from_slice(plist);
    }

    let mut hncacb: Vec<StoichEntry> = capped_peaks
        .iter()
        .filter(|p| p.exp == "HNCACB")
        .filter_map(|p| {
            p.c.map(|c| StoichEntry {
                exp: "HNCACB",
                atom: Some(if p.sign >= 0 { "CA" } else { "CB" }),
                offset: None,
                val: c,
                hn_dist: hn_dist(p),
                norm_height: norm_h(p),
                both_offsets: true,
            })
        })
        .collect();

    let mut entries: Vec<StoichEntry> = Vec::new();
    for p in &capped_peaks {
        let Some(c) = p.c else { continue };
        let hd = hn_dist(p);
        let nh = norm_h(p);
        match p.exp.as_str() {
            "HNCA" => entries.push(StoichEntry {
                exp: "HNCA",
                atom: Some("CA"),
                offset: None,
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: true,
            }),
            "HNCOCA" => entries.push(StoichEntry {
                exp: "HNCOCA",
                atom: Some("CA"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: false,
            }),
            "HNCO" => entries.push(StoichEntry {
                exp: "HNCO",
                atom: Some("C"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: false,
            }),
            "HNCACO" => entries.push(StoichEntry {
                exp: "HNCACO",
                atom: Some("C"),
                offset: None,
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: true,
            }),
            _ => {}
        }
    }

    // Match CBCACONH peaks against HNCACB to assign atom type and mark matched HNCACB peak as i-1
    let mut cbcaconh_entries: Vec<StoichEntry> = Vec::new();
    for p in &capped_peaks {
        if p.exp != "CBCACONH" {
            continue;
        }
        let Some(c) = p.c else { continue };
        let hd = hn_dist(p);
        let nh = norm_h(p);
        let mut best_idx: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        for (idx, e) in hncacb.iter().enumerate() {
            let atom = e.atom.unwrap_or("CA");
            let d = (e.val - c).abs();
            if d <= tol.carbon_tol(atom) && d < best_d {
                best_d = d;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            let matched_atom = hncacb[idx].atom;
            hncacb[idx].offset = Some(-1);
            cbcaconh_entries.push(StoichEntry {
                exp: "CBCACONH",
                atom: matched_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: false,
            });
        } else {
            cbcaconh_entries.push(StoichEntry {
                exp: "CBCACONH",
                atom: None,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                norm_height: nh,
                both_offsets: false,
            });
        }
    }

    // Suppress uncorroborated negative HNCACB dispersive / F1-sinc undershoot lobes sitting
    // inside the CB-empty pure-CA band (47.8..=56.5 ppm) within 1.35 ppm of a stronger CA peak
    let ca_peaks_on_strip: Vec<(f64, f64)> = hncacb
        .iter()
        .filter(|e| e.atom == Some("CA"))
        .map(|e| (e.val, e.norm_height.unwrap_or(0.0)))
        .collect();
    hncacb.retain(|e| {
        if e.atom == Some("CB") && e.offset.is_none() && (47.8..=56.5).contains(&e.val) {
            let eh = e.norm_height.unwrap_or(0.0);
            let is_ca_undershoot = ca_peaks_on_strip
                .iter()
                .any(|&(cv, ch)| (e.val - cv).abs() <= 1.35 && eh < ch);
            if is_ca_undershoot {
                return false;
            }
        }
        true
    });

    // Disambiguate unmatched CBCACONH peaks (when weak 2J HNCACB peak dropped out!)
    let has_typed_cb_im1 = cbcaconh_entries.iter().any(|e| e.atom == Some("CB"));
    let typed_ca_im1: Vec<(f64, f64)> = cbcaconh_entries
        .iter()
        .filter(|e| e.atom == Some("CA"))
        .map(|e| (e.val, e.norm_height.unwrap_or(0.0)))
        .collect();
    let has_typed_ca_im1 = !typed_ca_im1.is_empty();
    for entry in &mut cbcaconh_entries {
        if entry.atom.is_none() {
            let v = entry.val;
            let eh = entry.norm_height.unwrap_or(0.0);
            let is_ca_sidelobe = heights_present
                && typed_ca_im1
                    .iter()
                    .any(|&(cv, ch)| (v - cv).abs() <= 2.30 && ch > 0.0 && eh < 0.30 * ch);
            if is_ca_sidelobe {
                continue;
            }
            // Check against other i-1 entries first (e.g. HNCOCA)
            let matched_ca = entries.iter().any(|x| {
                x.atom == Some("CA") && x.offset == Some(-1) && (x.val - v).abs() <= tol.match_ca
            });
            if matched_ca {
                entry.atom = Some("CA");
            } else if !(38.5..=66.0).contains(&v) {
                entry.atom = Some("CB");
            } else if has_typed_cb_im1 && !has_typed_ca_im1 && (40.5..=65.5).contains(&v) {
                entry.atom = Some("CA");
            } else if has_typed_ca_im1 && !has_typed_cb_im1 && !(42.0..=50.0).contains(&v) {
                entry.atom = Some("CB");
            } else if heights_present && (41.0..=64.5).contains(&v) {
                // Keep unambiguous mid-aliphatic unmatched CBCACONH peak as CA(i-1) candidate
                entry.atom = Some("CA");
            }
        }
    }

    let cbcaconh_saw_ca = cbcaconh_entries.iter().any(|e| e.atom == Some("CA"));
    let cbcaconh_saw_cb = cbcaconh_entries.iter().any(|e| e.atom == Some("CB"));

    entries.extend(cbcaconh_entries);
    entries.extend(hncacb);

    // Resolve offsets of "i and i-1" peaks against i-1 references
    for i in 0..entries.len() {
        if let (Some(atom), None) = (entries[i].atom, entries[i].offset) {
            let v = entries[i].val;
            let m = tol.carbon_tol(atom);
            let mut has_refs = false;
            let mut matches_ref = false;
            for (j, x) in entries.iter().enumerate() {
                if j != i && x.atom == Some(atom) && x.offset == Some(-1) {
                    has_refs = true;
                    if (x.val - v).abs() <= m {
                        matches_ref = true;
                    }
                }
            }
            if has_refs {
                entries[i].offset = Some(if matches_ref { -1 } else { 0 });
            }
        }
    }

    // Post-Glycine fix: when CBCACONH transferred CA(i-1) cleanly on this strip (`cbcaconh_saw_ca`)
    // and observed zero CB(i-1) (`!cbcaconh_saw_cb`), while HNCACB observed exactly 1 unresolved CB
    // peak, that single CB peak is intra-residue CB(i) (`offset = 0`), preventing false CB(i-1) penalties on Gly(k-1).
    if cbcaconh_saw_ca && !cbcaconh_saw_cb {
        let unresolved_cb: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.exp == "HNCACB" && e.atom == Some("CB") && e.offset.is_none())
            .map(|(idx, _)| idx)
            .collect();
        if unresolved_cb.len() == 1 {
            entries[unresolved_cb[0]].offset = Some(0);
        }
    }

    // Intensity-ratio 1J vs 2J i / i-1 disambiguation when no i-1 reference exists for `atom`
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        let any_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if any_im1 {
            continue;
        }
        for &ename in &["HNCA", "HNCACB", "HNCACO"] {
            let idxs: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.exp == ename
                        && e.atom == Some(atom)
                        && e.offset.is_none()
                        && e.hn_dist <= 0.75
                        && e.norm_height.is_some()
                })
                .map(|(idx, _)| idx)
                .collect();
            if idxs.len() == 2 {
                let (i0, i1) = (idxs[0], idxs[1]);
                if (entries[i0].val - entries[i1].val).abs() > m {
                    let h0 = entries[i0].norm_height.unwrap_or(0.0);
                    let h1 = entries[i1].norm_height.unwrap_or(0.0);
                    let (strong, weak, hi, lo) = if h0 >= h1 {
                        (i0, i1, h0, h1)
                    } else {
                        (i1, i0, h1, h0)
                    };
                    if lo > 0.0 && hi / lo >= 1.55 {
                        entries[strong].offset = Some(0);
                        entries[weak].offset = Some(-1);
                    }
                }
            }
        }
        let newly_resolved_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if newly_resolved_im1 {
            for i in 0..entries.len() {
                if entries[i].atom == Some(atom) && entries[i].offset.is_none() {
                    let v = entries[i].val;
                    let matches = entries.iter().enumerate().any(|(j, x)| {
                        j != i
                            && x.atom == Some(atom)
                            && x.offset == Some(-1)
                            && (x.val - v).abs() <= m
                    });
                    entries[i].offset = Some(if matches { -1 } else { 0 });
                }
            }
        }
    }

    let mut ss = SpinSystem::new(root.h, root.n);
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        for off in [0, -1] {
            let pts: Vec<StoichClusterPoint> = entries
                .iter()
                .filter(|e| e.atom == Some(atom) && (e.offset == Some(off) || e.offset.is_none()))
                .map(|e| StoichClusterPoint {
                    exp: e.exp,
                    val: e.val,
                    hn_dist: e.hn_dist,
                    norm_height: e.norm_height,
                })
                .collect();
            if !pts.is_empty() {
                let mut clusters = cluster_stoich_points(&pts, m);
                if heights_present {
                    clusters.sort_by(|a, b| {
                        cluster_distinct_support(b)
                            .cmp(&cluster_distinct_support(a))
                            .then_with(|| cluster_prominence(b).total_cmp(&cluster_prominence(a)))
                            .then_with(|| cluster_mean_dist(a).total_cmp(&cluster_mean_dist(b)))
                    });
                } else {
                    clusters.sort_by(|a, b| {
                        cluster_distinct_support(b)
                            .cmp(&cluster_distinct_support(a))
                            .then_with(|| cluster_mean_dist(a).total_cmp(&cluster_mean_dist(b)))
                    });
                }
                clusters.truncate(max_cands);
                let cands: Vec<(f64, usize)> = clusters
                    .into_iter()
                    .map(|c| (cluster_weighted_val(&c), cluster_distinct_support(&c)))
                    .collect();
                ss.cands.insert((atom.to_string(), off), cands);
            } else if off == 0 && heights_present {
                // Intensity-gated degenerate i == i-1 candidate (`support = 0`, connectivity only)
                let im1_both: Vec<&StoichEntry> = entries
                    .iter()
                    .filter(|e| e.atom == Some(atom) && e.offset == Some(-1) && e.both_offsets)
                    .collect();
                let mut vals_deg: Vec<f64> = Vec::new();
                for &e in &im1_both {
                    let summed_nh: f64 = im1_both
                        .iter()
                        .filter(|o| o.exp == e.exp && (o.val - e.val).abs() <= m)
                        .filter_map(|o| o.norm_height)
                        .sum();
                    if summed_nh >= 1.55 {
                        vals_deg.push(e.val);
                    }
                }
                if !vals_deg.is_empty() {
                    let avg = vals_deg.iter().sum::<f64>() / (vals_deg.len() as f64);
                    ss.cands.insert((atom.to_string(), off), vec![(avg, 0)]);
                }
            }
        }
    }
    ss
}

// ============================================================================
// Component 2: Two-Pass Reference-Offset Estimation (Mixture -> Anchor)
// ============================================================================

fn sequence_atom_weights(seq: &[String], priors: &Priors, atom: &str) -> Vec<(f64, f64, f64)> {
    let mut counts: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    if seq.is_empty() {
        for &(res, _) in &STANDARD_RESIDUES {
            if priors.has(res, atom) {
                counts.insert(res, (1, 0.0));
            }
        }
    } else {
        let start = usize::from(atom == "H" || atom == "N");
        for (idx, r) in seq.iter().enumerate().skip(start) {
            if (atom == "H" || atom == "N") && r == "PRO" {
                continue;
            }
            for &(std_r, _) in &STANDARD_RESIDUES {
                if r == std_r && priors.has(std_r, atom) {
                    let d_cov = expected_covalent_atom_offset(delta_covalent(seq, idx), atom);
                    let entry = counts.entry(std_r).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += d_cov;
                }
            }
        }
    }
    let total: usize = counts.values().map(|&(c, _)| c).sum();
    if total == 0 {
        return Vec::new();
    }
    let tf = total as f64;
    counts
        .into_iter()
        .filter_map(|(res, (cnt, cov_sum))| {
            priors.get(res, atom).map(|(mu, sd)| {
                let mean_cov = cov_sum / (cnt.max(1) as f64);
                ((cnt as f64) / tf, mu + mean_cov, sd)
            })
        })
        .collect()
}

fn atom_mixture_loglik(obs: &[f64], comps: &[(f64, f64, f64)], delta: f64) -> f64 {
    if obs.is_empty() || comps.is_empty() {
        return 0.0;
    }
    let floor_ll = -10.5_f64;
    let mut total = 0.0;
    for &v in obs {
        let mut max_lp = floor_ll;
        let mut lps = [0.0_f64; 20];
        let m = comps.len().min(20);
        for (idx, &(w, mu, sd)) in comps.iter().take(m).enumerate() {
            // Broaden mixture component slightly to account for secondary-structure dispersion
            let sd_eff = (sd * sd + 0.55 * 0.55).sqrt();
            let lp = w.ln() + logpdf(v, mu + delta, sd_eff);
            lps[idx] = lp;
            if lp > max_lp {
                max_lp = lp;
            }
        }
        let mut sum_exp = (floor_ll - max_lp).exp() * 0.01;
        for &lp in &lps[..m] {
            sum_exp += (lp - max_lp).exp();
        }
        total += max_lp + sum_exp.ln();
    }
    total
}

fn maximize_1d_grid_with_gain(
    lo: f64,
    hi: f64,
    step: f64,
    mut f: impl FnMut(f64) -> f64,
) -> (f64, f64) {
    let n = (((hi - lo) / step).round() as usize).max(2);
    let v0 = f(0.0);
    let mut best_x = 0.0;
    let mut best_v = v0;
    for i in 0..=n {
        let x = lo + (i as f64) * step;
        let v = f(x);
        if v > best_v {
            best_v = v;
            best_x = x;
        }
    }
    let fine_lo = (best_x - step).max(lo);
    let fine_hi = (best_x + step).min(hi);
    let fine_step = step * 0.1;
    let nf = (((fine_hi - fine_lo) / fine_step).round() as usize).max(2);
    for i in 0..=nf {
        let x = fine_lo + (i as f64) * fine_step;
        let v = f(x);
        if v > best_v {
            best_v = v;
            best_x = x;
        }
    }
    (best_x, best_v - v0)
}

/// Pass-1 chemical-shift reference offset estimator maximizing marginal sequence-weighted
/// mixture log-likelihood against `priors` with joint `13C` coupling (`CA + CB + C`) and
/// log-likelihood-ratio significance gating so DSS-referenced data stays at `0.0 ppm` while
/// systematic referencing offsets (e.g. `-2.75 ppm` TMS vs DSS) are captured automatically.
pub fn estimate_mixture_reference_offsets(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
) -> ReferenceOffsets {
    let norm_seq = normalize_sequence(sequence);
    if systems.len() < 6 {
        return ReferenceOffsets::default();
    }

    let collect_atom = |atom: &str| -> Vec<f64> {
        let mut vals = Vec::new();
        for ss in systems {
            if !ss.has_observed_carbons() {
                continue;
            }
            for off in [0, -1] {
                if let Some(cands) = ss.cands.get(&(atom.to_string(), off)) {
                    if let Some(&(v, sup)) = cands.first() {
                        if sup > 0 {
                            vals.push(v);
                        }
                    }
                }
            }
        }
        vals
    };

    let obs_ca = collect_atom("CA");
    let obs_cb = collect_atom("CB");
    let obs_c = collect_atom("C");
    let obs_h: Vec<f64> = systems
        .iter()
        .filter(|s| s.has_observed_carbons())
        .map(|s| s.h)
        .collect();
    let obs_n: Vec<f64> = systems
        .iter()
        .filter(|s| s.has_observed_carbons())
        .map(|s| s.n)
        .collect();

    let comp_ca = sequence_atom_weights(&norm_seq, priors, "CA");
    let comp_cb = sequence_atom_weights(&norm_seq, priors, "CB");
    let comp_c = sequence_atom_weights(&norm_seq, priors, "C");
    let comp_h = sequence_atom_weights(&norm_seq, priors, "H");
    let comp_n = sequence_atom_weights(&norm_seq, priors, "N");

    // Step A: Fit shared 13C reference offset across CA + CB + C (cancelling secondary-structure bias)
    let (d_13c_raw, gain_13c) = if obs_ca.len() >= 8 || obs_cb.len() >= 8 || obs_c.len() >= 8 {
        maximize_1d_grid_with_gain(-5.5, 5.5, 0.10, |d| {
            atom_mixture_loglik(&obs_ca, &comp_ca, d)
                + atom_mixture_loglik(&obs_cb, &comp_cb, d)
                + atom_mixture_loglik(&obs_c, &comp_c, d)
                - 0.5 * (d / 3.5).powi(2)
        })
    } else {
        (0.0, 0.0)
    };
    let d_aliph = if gain_13c >= 8.0 && d_13c_raw.abs() >= 0.65 {
        let (da, _) = maximize_1d_grid_with_gain(d_13c_raw - 0.80, d_13c_raw + 0.80, 0.05, |d| {
            atom_mixture_loglik(&obs_ca, &comp_ca, d) + atom_mixture_loglik(&obs_cb, &comp_cb, d)
                - 0.5 * ((d - d_13c_raw) / 0.35).powi(2)
        });
        da
    } else {
        0.0
    };

    // Step B: Allow small tightly-regularized differential around d_aliph when significant
    let (d_ca, d_cb) = if d_aliph == 0.0 {
        (0.0, 0.0)
    } else {
        let (dca, _) = maximize_1d_grid_with_gain(d_aliph - 0.45, d_aliph + 0.45, 0.05, |d| {
            atom_mixture_loglik(&obs_ca, &comp_ca, d) - 0.5 * ((d - d_aliph) / 0.20).powi(2)
        });
        let (dcb, _) = maximize_1d_grid_with_gain(d_aliph - 0.45, d_aliph + 0.45, 0.05, |d| {
            atom_mixture_loglik(&obs_cb, &comp_cb, d) - 0.5 * ((d - d_aliph) / 0.20).powi(2)
        });
        (dca, dcb)
    };

    // Step C: Carbonyl 13C' offset (either coupled to significant 13C shift or independently misreferenced)
    let d_c = if obs_c.len() >= 8 {
        if d_aliph != 0.0 {
            let (dc, _) = maximize_1d_grid_with_gain(d_aliph - 1.20, d_aliph + 1.20, 0.05, |d| {
                atom_mixture_loglik(&obs_c, &comp_c, d) - 0.5 * ((d - d_aliph) / 0.55).powi(2)
            });
            dc
        } else {
            let (raw_c, gain_c) = maximize_1d_grid_with_gain(-5.5, 5.5, 0.10, |d| {
                atom_mixture_loglik(&obs_c, &comp_c, d) - 0.5 * (d / 3.5).powi(2)
            });
            if gain_c >= 8.0 && raw_c.abs() >= 0.65 {
                raw_c
            } else {
                0.0
            }
        }
    } else {
        d_aliph
    };

    let d_h = if obs_h.len() >= 10 {
        let (raw_h, gain_h) = maximize_1d_grid_with_gain(-0.60, 0.60, 0.02, |d| {
            atom_mixture_loglik(&obs_h, &comp_h, d) - 0.5 * (d / 0.22).powi(2)
        });
        if gain_h >= 6.0 && raw_h.abs() >= 0.12 {
            raw_h
        } else {
            0.0
        }
    } else {
        0.0
    };

    let d_n = if obs_n.len() >= 10 {
        let (raw_n, gain_n) = maximize_1d_grid_with_gain(-3.00, 3.00, 0.08, |d| {
            atom_mixture_loglik(&obs_n, &comp_n, d) - 0.5 * (d / 1.10).powi(2)
        });
        if gain_n >= 6.0 && raw_n.abs() >= 0.90 {
            raw_n
        } else {
            0.0
        }
    } else {
        0.0
    };

    ReferenceOffsets {
        d_ca,
        d_cb,
        d_c,
        d_h,
        d_n,
    }
}

/// Pass-2 reference-offset refinement from high-consensus (`>= min_consensus`) assigned
/// `(spin_system, residue)` anchor pairs, using `0.5 * (median(dCA) + median(dCB))` coupling
/// to separate uniform spectrometer referencing from protein secondary-structure content.
pub fn refine_anchor_reference_offsets(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    pass1: &AssignResult,
    min_consensus: f64,
    init_offsets: ReferenceOffsets,
) -> (ReferenceOffsets, usize) {
    refine_anchor_reference_offsets_with_torsions(
        systems,
        sequence,
        priors,
        pass1,
        min_consensus,
        init_offsets,
        None,
    )
}

fn refine_anchor_reference_offsets_with_torsions(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    pass1: &AssignResult,
    min_consensus: f64,
    init_offsets: ReferenceOffsets,
    structure_torsions: Option<&[ResidueTorsion]>,
) -> (ReferenceOffsets, usize) {
    let norm_seq = normalize_sequence(sequence);
    let init_priors = apply_offsets_to_priors(priors, &init_offsets);

    let mut res_ca = Vec::new();
    let mut res_cb = Vec::new();
    let mut res_c = Vec::new();
    let mut res_h = Vec::new();
    let mut res_n = Vec::new();
    let mut n_anchors = 0usize;

    for (s, &k_raw) in pass1.position.iter().enumerate() {
        if k_raw <= 0 || pass1.confidence.get(s).copied().unwrap_or(0.0) < min_consensus {
            continue;
        }
        let k = k_raw as usize;
        if k >= norm_seq.len() {
            continue;
        }
        n_anchors += 1;
        let ss = &systems[s];
        let rname = &norm_seq[k];
        let prev_name = &norm_seq[k - 1];
        let cov_k = delta_covalent(&norm_seq, k);
        let cov_km1 = delta_covalent(&norm_seq, k - 1);
        let (phi_k, psi_k) = structure_torsions
            .and_then(|t| t.get(k).copied())
            .unwrap_or((None, None));
        let (phi_km1, psi_km1) = structure_torsions
            .and_then(|t| t.get(k - 1).copied())
            .unwrap_or((None, None));
        let has_tors = structure_torsions.is_some();

        if let Some((mu_h, _)) = priors.get(rname, "H") {
            let d_tors = if has_tors {
                torsion_shift_adjustment(rname, "H", phi_k, psi_k).0
            } else {
                0.0
            };
            res_h.push(ss.h - (mu_h + d_tors));
        }
        if let Some((mu_n, _)) = priors.get(rname, "N") {
            let d_tors = if has_tors {
                torsion_shift_adjustment(rname, "N", phi_k, psi_k).0
            } else {
                0.0
            };
            res_n.push(ss.n - (mu_n + cov_k.0 + d_tors));
        }
        if let Some(cands) = ss.cands.get(&("CA".to_string(), 0)) {
            let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
            if !obs.is_empty() {
                if let Some((mu_ca, _)) = priors.get(rname, "CA") {
                    let (_, idx) = if has_tors {
                        conditioned_candidate_llr(
                            &init_priors,
                            rname,
                            "CA",
                            &obs,
                            0.7,
                            (phi_k, psi_k),
                            cov_k.1,
                        )
                    } else {
                        candidate_llr_shifted(&init_priors, rname, "CA", &obs, 0.7, cov_k.1)
                    };
                    let d_tors = if has_tors {
                        torsion_shift_adjustment(rname, "CA", phi_k, psi_k).0
                    } else {
                        0.0
                    };
                    res_ca.push(
                        obs[idx].0 - (mu_ca + expected_covalent_atom_offset(cov_k, "CA") + d_tors),
                    );
                }
            }
        }
        if rname != "GLY" {
            if let Some(cands) = ss.cands.get(&("CB".to_string(), 0)) {
                let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
                if !obs.is_empty() {
                    if let Some((mu_cb, _)) = priors.get(rname, "CB") {
                        let (_, idx) = if has_tors {
                            conditioned_candidate_llr(
                                &init_priors,
                                rname,
                                "CB",
                                &obs,
                                0.7,
                                (phi_k, psi_k),
                                cov_k.2,
                            )
                        } else {
                            candidate_llr_shifted(&init_priors, rname, "CB", &obs, 0.7, cov_k.2)
                        };
                        let d_tors = if has_tors {
                            torsion_shift_adjustment(rname, "CB", phi_k, psi_k).0
                        } else {
                            0.0
                        };
                        res_cb.push(
                            obs[idx].0
                                - (mu_cb + expected_covalent_atom_offset(cov_k, "CB") + d_tors),
                        );
                    }
                }
            }
        }
        if let Some(cands) = ss.cands.get(&("C".to_string(), -1)) {
            let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
            if !obs.is_empty() {
                if let Some((mu_c, _)) = priors.get(prev_name, "C") {
                    let (_, idx) = if has_tors {
                        conditioned_candidate_llr(
                            &init_priors,
                            prev_name,
                            "C",
                            &obs,
                            0.7,
                            (phi_km1, psi_km1),
                            cov_km1.3,
                        )
                    } else {
                        candidate_llr_shifted(&init_priors, prev_name, "C", &obs, 0.7, cov_km1.3)
                    };
                    let d_tors = if has_tors {
                        torsion_shift_adjustment(prev_name, "C", phi_km1, psi_km1).0
                    } else {
                        0.0
                    };
                    res_c.push(
                        obs[idx].0 - (mu_c + expected_covalent_atom_offset(cov_km1, "C") + d_tors),
                    );
                }
            }
        } else if let Some(cands) = ss.cands.get(&("C".to_string(), 0)) {
            let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
            if !obs.is_empty() {
                if let Some((mu_c, _)) = priors.get(rname, "C") {
                    let (_, idx) = if has_tors {
                        conditioned_candidate_llr(
                            &init_priors,
                            rname,
                            "C",
                            &obs,
                            0.7,
                            (phi_k, psi_k),
                            cov_k.3,
                        )
                    } else {
                        candidate_llr_shifted(&init_priors, rname, "C", &obs, 0.7, cov_k.3)
                    };
                    let d_tors = if has_tors {
                        torsion_shift_adjustment(rname, "C", phi_k, psi_k).0
                    } else {
                        0.0
                    };
                    res_c.push(
                        obs[idx].0 - (mu_c + expected_covalent_atom_offset(cov_k, "C") + d_tors),
                    );
                }
            }
        }
    }

    if n_anchors < 8 {
        return (init_offsets, n_anchors);
    }

    let med_ca = median_f64(res_ca);
    let med_cb = median_f64(res_cb);
    let (d_ca, d_cb) = match (med_ca, med_cb) {
        (Some(mca), Some(mcb)) => {
            let shared = 0.5 * (mca + mcb);
            if init_offsets.d_ca.abs() < 0.50 && shared.abs() < 0.55 {
                (0.0, 0.0)
            } else {
                // Unbiased Pass-2 anchor refinement: 75% shared aliphatic + 25% per-nucleus median
                let ca_adj = shared + (0.25 * (mca - shared)).clamp(-0.25, 0.25);
                let cb_adj = shared + (0.25 * (mcb - shared)).clamp(-0.25, 0.25);
                (ca_adj, cb_adj)
            }
        }
        (Some(mca), None) => {
            if init_offsets.d_ca.abs() < 0.50 && mca.abs() < 0.65 {
                (0.0, 0.0)
            } else {
                (mca, mca)
            }
        }
        _ => (init_offsets.d_ca, init_offsets.d_cb),
    };

    let d_c = median_f64(res_c)
        .map(|mc| {
            if init_offsets.d_c.abs() < 0.50 && d_ca == 0.0 && mc.abs() < 0.55 {
                0.0
            } else {
                mc
            }
        })
        .unwrap_or(init_offsets.d_c);
    let d_h = median_f64(res_h)
        .map(|mh| {
            if init_offsets.d_h.abs() < 0.08 && mh.abs() < 0.14 {
                0.0
            } else {
                mh.clamp(-0.45, 0.45)
            }
        })
        .unwrap_or(init_offsets.d_h);
    let d_n = median_f64(res_n)
        .map(|mn| {
            if init_offsets.d_n.abs() < 0.50 && mn.abs() < 1.50 {
                0.0
            } else {
                mn.clamp(-2.50, 2.50)
            }
        })
        .unwrap_or(init_offsets.d_n);

    (
        ReferenceOffsets {
            d_ca,
            d_cb,
            d_c,
            d_h,
            d_n,
        },
        n_anchors,
    )
}

// ============================================================================
// Component 4B: Optional 3D Structure (phi, psi) Shift Conditioning
// ============================================================================

/// Extract `(Option<phi_deg>, Option<psi_deg>)` torsions aligned to `sequence` from a parsed `PdbChain`.
pub fn torsions_from_pdb_chain(
    chain: &PdbChain,
    sequence: &[String],
) -> Vec<(Option<f64>, Option<f64>)> {
    let norm_seq = normalize_sequence(sequence);
    let mut out = vec![(None, None); norm_seq.len()];
    if norm_seq.is_empty() || chain.residues.is_empty() {
        return out;
    }
    let exp_torsions = compute_chain_torsions(chain);
    let seq_keys: Vec<(i32, &str)> = norm_seq
        .iter()
        .enumerate()
        .map(|(i, r)| ((i + 1) as i32, r.as_str()))
        .collect();
    let exp_keys: Vec<(i32, &str)> = exp_torsions
        .iter()
        .map(|(sid, rn, _, _)| (*sid, rn.as_str()))
        .collect();
    for (si, ei) in align_by_res_names(&seq_keys, &exp_keys) {
        if si < out.len() {
            out[si] = (exp_torsions[ei].2, exp_torsions[ei].3);
        }
    }
    out
}

/// Parse the first model of a `.pdb` file and return `(Option<phi_deg>, Option<psi_deg>)` per residue of `sequence`.
pub fn torsions_from_pdb_path(
    path: impl AsRef<Path>,
    sequence: &[String],
) -> Result<Vec<ResidueTorsion>, String> {
    let chain = parse_pdb_first_model(path)?;
    Ok(torsions_from_pdb_chain(&chain, sequence))
}

/// Empirical `(phi, psi)` secondary-shift expectation `(delta_mu_ppm, sd_scale)` relative to BMRB
/// residue mean, parameterized after Spera & Bax (*JACS* 1991) and Wishart et al. (*J. Biomol. NMR* 1995).
#[inline]
fn torsion_shift_adjustment(
    res: &str,
    atom: &str,
    phi_opt: Option<f64>,
    psi_opt: Option<f64>,
) -> (f64, f64) {
    let (Some(phi), Some(psi)) = (phi_opt, psi_opt) else {
        return (0.0, 1.0);
    };
    let mut d_mu = 0.0;

    let dphi_a = periodic_angle_diff_deg(phi, -63.0);
    let dpsi_a = periodic_angle_diff_deg(psi, -42.0);
    let w_alpha = (-0.5 * ((dphi_a / 28.0).powi(2) + (dpsi_a / 28.0).powi(2))).exp();

    let dphi_b1 = periodic_angle_diff_deg(phi, -122.0);
    let dpsi_b1 = periodic_angle_diff_deg(psi, 135.0);
    let w_b1 = (-0.5 * ((dphi_b1 / 32.0).powi(2) + (dpsi_b1 / 32.0).powi(2))).exp();

    let dphi_b2 = periodic_angle_diff_deg(phi, -138.0);
    let dpsi_b2 = periodic_angle_diff_deg(psi, 162.0);
    let w_b2 = 0.90 * (-0.5 * ((dphi_b2 / 30.0).powi(2) + (dpsi_b2 / 28.0).powi(2))).exp();
    let w_beta = w_b1.max(w_b2);

    let dphi_l = periodic_angle_diff_deg(phi, 60.0);
    let dpsi_l = periodic_angle_diff_deg(psi, 35.0);
    let w_alpha_l = (-0.5 * ((dphi_l / 26.0).powi(2) + (dpsi_l / 28.0).powi(2))).exp();

    let conf_strength = w_alpha.max(w_beta);
    let mut sd_scale = 1.0;

    match atom {
        "CA" => {
            if res == "GLY" {
                d_mu += 1.05 * w_alpha - 0.65 * w_beta + 0.50 * w_alpha_l;
            } else {
                d_mu += 2.35 * w_alpha - 1.55 * w_beta + 0.65 * w_alpha_l;
            }
            sd_scale = 1.0 - 0.28 * conf_strength;
        }
        "CB" if res != "GLY" => {
            d_mu += -1.05 * w_alpha + 1.50 * w_beta + 0.15 * w_alpha_l;
            sd_scale = 1.0 - 0.24 * conf_strength;
        }
        "C" => {
            d_mu += 1.65 * w_alpha - 1.25 * w_beta + 0.45 * w_alpha_l;
            sd_scale = 1.0 - 0.24 * conf_strength;
        }
        "H" if res != "PRO" => {
            d_mu += -0.18 * w_alpha + 0.25 * w_beta;
            sd_scale = 1.0 - 0.10 * conf_strength;
        }
        _ => {}
    }

    (d_mu, sd_scale.clamp(0.68, 1.0))
}

fn residue_conditioned_llr(
    priors: &Priors,
    res: &str,
    atom: &str,
    v: f64,
    phi_opt: Option<f64>,
    psi_opt: Option<f64>,
    d_cov: f64,
) -> f64 {
    if !priors.table.contains_key(res) {
        return 0.0;
    }
    if !priors.has(res, atom) {
        return 0.02_f64.ln();
    }
    let (mu_base, sd_base) = priors.table[res][atom];
    let (d_tors, sd_scale) = torsion_shift_adjustment(res, atom, phi_opt, psi_opt);
    let mu = mu_base + d_cov + d_tors;
    let sd = (sd_base * sd_scale).max(0.25);
    (logpdf(v, mu, sd) - priors.background(atom, v)).max(-12.0)
}

fn conditioned_candidate_llr(
    priors: &Priors,
    res: &str,
    atom: &str,
    cands: &[(f64, usize)],
    support_bonus: f64,
    torsion: ResidueTorsion,
    d_cov: f64,
) -> (f64, usize) {
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &(v, sup)) in cands.iter().enumerate() {
        let sc = residue_conditioned_llr(priors, res, atom, v, torsion.0, torsion.1, d_cov)
            + support_bonus * ((sup as f64) - 1.0);
        if sc > best {
            best = sc;
            best_i = i;
        }
    }
    (best - (cands.len() as f64).ln(), best_i)
}

/// Sub-stoichiometric minor-conformer threshold (`rel_root_height < 0.15`) and major-conformer
/// reference threshold (`rel_root_height >= 0.45`).
const MINOR_CONFORMER_REL_HEIGHT: f64 = 0.15;
const MAJOR_CONFORMER_REL_HEIGHT: f64 = 0.45;

/// Compute smooth sub-stoichiometric minor-conformer unary penalties (in nats) for roots with
/// `rel_root_height < 0.15` while leaving all roots with `rel_root_height >= 0.15` (including
/// moderate exchange-broadened amides) or `None` at exact `0.0`.
///
/// Sub-stoichiometric roots (`< 15%` median root height) that lack intra-residue `CA(0)` or whose
/// observed `i`/`i-1` `13C` shifts duplicate a major-state root (`rel_root_height >= 0.45`, e.g.,
/// slow-exchange minor second-form peaks such as Ubiquitin `Gln2` at `~9–12%` amplitude) receive
/// a smooth population-weighted conformer log-prior penalty, whereas unique multi-slot C-terminal
/// or flexible Glycine roots without a major-state carbon duplicate receive only a mild `< 0.8`-nat
/// sub-stoichiometric prior.
fn compute_minor_conformer_penalties(systems: &[SpinSystem]) -> Vec<f64> {
    let mut penalties = vec![0.0_f64; systems.len()];
    let major_roots: Vec<(usize, f64)> = systems
        .iter()
        .enumerate()
        .filter_map(|(m, ss)| {
            let r = ss.rel_root_height?;
            if r >= MAJOR_CONFORMER_REL_HEIGHT && ss.has_observed_carbons() {
                Some((m, r))
            } else {
                None
            }
        })
        .collect();
    if major_roots.is_empty() {
        return penalties;
    }

    for (s, ss) in systems.iter().enumerate() {
        let Some(r) = ss.rel_root_height else {
            continue;
        };
        let has_ca_i = ss
            .cands
            .get(&("CA".to_string(), 0))
            .is_some_and(|l| l.iter().any(|c| c.1 > 0));

        let obs_slots: Vec<(&str, i32, Vec<f64>)> = ss
            .cands
            .iter()
            .filter_map(|((atom, off), raw)| {
                let vals: Vec<f64> = raw
                    .iter()
                    .filter(|&&(_, sup)| sup > 0)
                    .map(|&(v, _)| v)
                    .collect();
                if vals.is_empty() {
                    None
                } else {
                    Some((atom.as_str(), *off, vals))
                }
            })
            .collect();
        let n_slots = obs_slots.len();

        if r < MINOR_CONFORMER_REL_HEIGHT {
            let u_sub = ((MINOR_CONFORMER_REL_HEIGHT - r.max(0.0)) / MINOR_CONFORMER_REL_HEIGHT)
                .clamp(0.0, 1.0);
            let mut best_shadow_r_maj: Option<f64> = None;
            let mut best_local_conf_r_maj: Option<f64> = None;
            if n_slots > 0 {
                let req_slots = if n_slots <= 2 {
                    n_slots
                } else {
                    (n_slots * 2).div_ceil(3)
                };
                for &(m, r_maj) in &major_roots {
                    if m == s {
                        continue;
                    }
                    let sm = &systems[m];
                    let mut matched_slots = 0usize;
                    for (atom, off, s_vals) in &obs_slots {
                        let tol_c = if *atom == "C" { 0.40 } else { 0.55 };
                        if let Some(m_raw) = sm.cands.get(&(atom.to_string(), *off)) {
                            let hit = s_vals.iter().any(|&vs| {
                                m_raw
                                    .iter()
                                    .any(|&(vm, sup)| sup > 0 && (vs - vm).abs() <= tol_c)
                            });
                            if hit {
                                matched_slots += 1;
                            }
                        }
                    }
                    let near_hn = (ss.h - sm.h).abs() <= 0.35 && (ss.n - sm.n).abs() <= 2.50;
                    let hn_neighbor = near_hn && matched_slots >= 1;
                    if matched_slots >= req_slots || hn_neighbor {
                        best_shadow_r_maj =
                            Some(best_shadow_r_maj.map_or(r_maj, |prev| prev.max(r_maj)));
                    }
                    if near_hn
                        && n_slots >= 2
                        && matched_slots >= req_slots
                        && r_maj >= 6.0 * r.max(0.015)
                    {
                        best_local_conf_r_maj =
                            Some(best_local_conf_r_maj.map_or(r_maj, |prev| prev.max(r_maj)));
                    }
                }
            }

            let mut pen = 0.75 * u_sub;
            if !has_ca_i {
                pen += 2.6 * (1.0 - (-6.0 * u_sub).exp());
            }
            if let Some(r_maj) = best_local_conf_r_maj {
                let log_pop = (r_maj / r.max(0.015)).ln().max(0.0);
                pen += 5.0 * log_pop * (1.0 - (-8.0 * u_sub).exp());
            } else if let Some(r_maj) = best_shadow_r_maj {
                let log_pop = (r_maj / r.max(0.015)).ln().max(0.0);
                pen += 1.65 * log_pop * (1.0 - (-8.0 * u_sub).exp());
            }
            penalties[s] = pen;
        } else if n_slots >= 4 && has_ca_i && obs_slots.iter().any(|(_, off, _)| *off == -1) {
            // Local slow-exchange satellite / split-peak tie-breaker: only triggers when a >= 3x
            // stronger major root (`r_maj >= 0.90`, `r <= 0.35 * r_maj`) sits within `(0.20 ppm 1H,
            // 0.60 ppm 15N)` and reproduces 100% of `s`'s observed carbon peaks across all >= 4
            // intra+sequential slots within 0.25 ppm (zero distinct carbon peaks on `s`).
            let mut best_sat_ratio = 1.0_f64;
            for &(m, r_maj) in &major_roots {
                if m == s || r_maj < 0.90 || r > 0.35 * r_maj {
                    continue;
                }
                let sm = &systems[m];
                if (ss.h - sm.h).abs() > 0.20 || (ss.n - sm.n).abs() > 0.60 {
                    continue;
                }
                let all_identical = obs_slots.iter().all(|(atom, off, s_vals)| {
                    let Some(m_raw) = sm.cands.get(&(atom.to_string(), *off)) else {
                        return false;
                    };
                    s_vals.iter().all(|&vs| {
                        m_raw
                            .iter()
                            .any(|&(vm, sup)| sup > 0 && (vs - vm).abs() <= 0.25)
                    })
                });
                if all_identical {
                    best_sat_ratio = best_sat_ratio.max(r_maj / r.max(0.015));
                }
            }
            if best_sat_ratio > 1.0 {
                penalties[s] = 6.0 * best_sat_ratio.ln();
            }
        }
    }

    penalties
}

fn build_evo_unary_matrix(
    systems: &[SpinSystem],
    seq: &[String],
    shifted_priors: &Priors,
    cfg: &EvoGraphConfig<'_>,
    effective_assign_cost: f64,
    locks: &[(usize, usize)],
) -> Matrix2D {
    let s_len = systems.len();
    let k_len = seq.len();
    let mut u = Matrix2D::filled(s_len, k_len, FORBID);
    let minor_penalties = compute_minor_conformer_penalties(systems);

    let has_global_aliphatic = systems.iter().any(|ss| {
        ss.cands
            .get(&("CA".to_string(), 0))
            .is_some_and(|c| c.iter().any(|x| x.1 > 0))
            || ss
                .cands
                .get(&("CA".to_string(), -1))
                .is_some_and(|c| c.iter().any(|x| x.1 > 0))
    });
    let assignable_slots = (1..k_len).filter(|&k| seq[k] != "PRO").count();
    let surplus_roots = s_len > assignable_slots;

    for k in 1..k_len {
        let res = &seq[k];
        let prev = &seq[k - 1];
        if res == "PRO" {
            continue;
        }
        let cov_k = delta_covalent(seq, k);
        let cov_km1 = delta_covalent(seq, k - 1);
        let (phi_k, psi_k) = cfg
            .structure_torsions
            .and_then(|t| t.get(k).copied())
            .unwrap_or((None, None));
        let (phi_km1, psi_km1) = cfg
            .structure_torsions
            .and_then(|t| t.get(k - 1).copied())
            .unwrap_or((None, None));

        for (s, ss) in systems.iter().enumerate() {
            if cfg.score_cfg.forbid_zero_carbons && !ss.has_observed_carbons() {
                continue;
            }
            let has_ca_any = ss
                .cands
                .get(&("CA".to_string(), 0))
                .is_some_and(|c| c.iter().any(|x| x.1 > 0))
                || ss
                    .cands
                    .get(&("CA".to_string(), -1))
                    .is_some_and(|c| c.iter().any(|x| x.1 > 0));
            let has_cb_any = ss
                .cands
                .get(&("CB".to_string(), 0))
                .is_some_and(|c| c.iter().any(|x| x.1 > 0))
                || ss
                    .cands
                    .get(&("CB".to_string(), -1))
                    .is_some_and(|c| c.iter().any(|x| x.1 > 0));

            if has_global_aliphatic && surplus_roots && !has_ca_any && !has_cb_any {
                continue;
            }
            let has_intra_any = ["CA", "CB", "C"].iter().any(|atom| {
                ss.cands
                    .get(&((*atom).to_string(), 0))
                    .is_some_and(|c| c.iter().any(|x| x.1 > 0))
            });
            if has_global_aliphatic
                && surplus_roots
                && !has_ca_any
                && !has_intra_any
                && ss.h < 7.65
                && ss.n < 115.5
            {
                continue;
            }

            let mut sc = 0.0;

            for ((atom, off), raw_cands) in &ss.cands {
                let cands: Vec<(f64, usize)> =
                    raw_cands.iter().copied().filter(|c| c.1 > 0).collect();
                if cands.is_empty() {
                    continue;
                }
                let (target_res, phi_t, psi_t, cov) = if *off == 0 {
                    (res.as_str(), phi_k, psi_k, cov_k)
                } else {
                    (prev.as_str(), phi_km1, psi_km1, cov_km1)
                };
                let d_cov = covalent_atom_offset(cov, atom);
                if cfg.structure_torsions.is_some() {
                    sc += conditioned_candidate_llr(
                        shifted_priors,
                        target_res,
                        atom,
                        &cands,
                        cfg.score_cfg.support_bonus,
                        (phi_t, psi_t),
                        d_cov,
                    )
                    .0;
                } else {
                    sc += candidate_llr_shifted(
                        shifted_priors,
                        target_res,
                        atom,
                        &cands,
                        cfg.score_cfg.support_bonus,
                        d_cov,
                    )
                    .0;
                }
            }
            if cfg.score_cfg.use_hn {
                if cfg.structure_torsions.is_some() {
                    sc += residue_conditioned_llr(
                        shifted_priors,
                        res,
                        "N",
                        ss.n,
                        phi_k,
                        psi_k,
                        cov_k.0,
                    ) + residue_conditioned_llr(
                        shifted_priors,
                        res,
                        "H",
                        ss.h,
                        phi_k,
                        psi_k,
                        0.0,
                    );
                } else {
                    sc += shifted_priors.llr_shifted(res, "N", ss.n, cov_k.0)
                        + shifted_priors.llr(res, "H", ss.h);
                }
            }
            sc -= effective_assign_cost + minor_penalties[s];
            u.set(s, k, sc);
        }
    }

    for &(s_lock, k_lock) in locks {
        if s_lock >= s_len || k_lock == 0 || k_lock >= k_len || seq[k_lock] == "PRO" {
            continue;
        }
        let mut lock_sc = u.get(s_lock, k_lock);
        if lock_sc <= FORBID * 0.5 || !lock_sc.is_finite() {
            let ss = &systems[s_lock];
            let res = &seq[k_lock];
            let cov_k = delta_covalent(seq, k_lock);
            let mut sc = 0.0;
            if cfg.score_cfg.use_hn {
                sc += shifted_priors.llr_shifted(res, "N", ss.n, cov_k.0)
                    + shifted_priors.llr(res, "H", ss.h);
            }
            sc -= effective_assign_cost;
            lock_sc = if sc.is_finite() { sc.max(-500.0) } else { 0.0 };
        }
        for k in 0..k_len {
            u.set(s_lock, k, FORBID);
        }
        for s in 0..s_len {
            u.set(s, k_lock, FORBID);
        }
        u.set(s_lock, k_lock, lock_sc);
    }

    u
}

/// Calibrate raw multi-island consensus confidence with the local conditional Boltzmann margin
/// `1 / (1 + sum_m exp((E_alt_m - E*) / T_cal))` over unassignment, competing free spin systems,
/// empty sequence slots, and 2-cycle swaps so near-degenerate local alternatives never receive
/// overconfident `>= 0.80` probability solely from evolutionary clone convergence.
fn calibrate_boltzmann_confidence(res: &mut AssignResult, u: &Matrix2D, p: &Matrix2D) {
    let s_len = u.rows;
    let k_len = u.cols;
    if s_len == 0 || k_len == 0 {
        return;
    }
    let mut at = vec![-1i32; k_len];
    for (s, &k) in res.position.iter().enumerate() {
        if k >= 0 && (k as usize) < k_len {
            at[k as usize] = s as i32;
        }
    }

    let site_energy = |spin: usize, slot: usize, cur_at: &[i32]| -> f64 {
        let mut sc = u.get(spin, slot);
        if slot > 0 && cur_at[slot - 1] >= 0 {
            sc += p.get(cur_at[slot - 1] as usize, spin);
        }
        if slot + 1 < k_len && cur_at[slot + 1] >= 0 {
            sc += p.get(spin, cur_at[slot + 1] as usize);
        }
        sc
    };

    let t_cal = 1.65_f64;
    for s in 0..s_len {
        let k_raw = res.position[s];
        if k_raw < 0 || res.confidence[s] <= 0.0 {
            continue;
        }
        let k = k_raw as usize;
        let e_self = site_energy(s, k, &at);
        let mut alt_partition = (-e_self / t_cal).clamp(-40.0, 20.0).exp();

        // Competing unassigned spin systems `t` taking slot `k`
        for t in 0..s_len {
            if t != s && res.position[t] < 0 && u.get(t, k) > FORBID * 0.5 {
                let delta = site_energy(t, k, &at) - e_self;
                alt_partition += (delta / t_cal).clamp(-40.0, 20.0).exp();
            }
        }

        // Relocating `s` to an empty slot `k2` or 2-swapping `(s @ k) <-> (t @ k2)`
        for k2 in 1..k_len {
            if k2 == k || u.get(s, k2) <= FORBID * 0.5 {
                continue;
            }
            let t_raw = at[k2];
            if t_raw < 0 {
                let mut e_s_k2 = u.get(s, k2);
                if k2 > 0 && k2 - 1 != k && at[k2 - 1] >= 0 {
                    e_s_k2 += p.get(at[k2 - 1] as usize, s);
                }
                if k2 + 1 < k_len && k2 + 1 != k && at[k2 + 1] >= 0 {
                    e_s_k2 += p.get(s, at[k2 + 1] as usize);
                }
                let delta = e_s_k2 - e_self;
                alt_partition += (delta / t_cal).clamp(-40.0, 20.0).exp();
            } else {
                let t = t_raw as usize;
                if u.get(t, k) <= FORBID * 0.5 {
                    continue;
                }
                // Exact 2-swap delta between k and k2
                let mut before = u.get(s, k) + u.get(t, k2);
                let mut after = u.get(t, k) + u.get(s, k2);
                if k + 1 == k2 {
                    if k > 0 && at[k - 1] >= 0 {
                        before += p.get(at[k - 1] as usize, s);
                        after += p.get(at[k - 1] as usize, t);
                    }
                    before += p.get(s, t);
                    after += p.get(t, s);
                    if k2 + 1 < k_len && at[k2 + 1] >= 0 {
                        before += p.get(t, at[k2 + 1] as usize);
                        after += p.get(s, at[k2 + 1] as usize);
                    }
                } else if k2 + 1 == k {
                    if k2 > 0 && at[k2 - 1] >= 0 {
                        before += p.get(at[k2 - 1] as usize, t);
                        after += p.get(at[k2 - 1] as usize, s);
                    }
                    before += p.get(t, s);
                    after += p.get(s, t);
                    if k + 1 < k_len && at[k + 1] >= 0 {
                        before += p.get(s, at[k + 1] as usize);
                        after += p.get(t, at[k + 1] as usize);
                    }
                } else {
                    if k > 0 && at[k - 1] >= 0 {
                        before += p.get(at[k - 1] as usize, s);
                        after += p.get(at[k - 1] as usize, t);
                    }
                    if k + 1 < k_len && at[k + 1] >= 0 {
                        before += p.get(s, at[k + 1] as usize);
                        after += p.get(t, at[k + 1] as usize);
                    }
                    if k2 > 0 && at[k2 - 1] >= 0 {
                        before += p.get(at[k2 - 1] as usize, t);
                        after += p.get(at[k2 - 1] as usize, s);
                    }
                    if k2 + 1 < k_len && at[k2 + 1] >= 0 {
                        before += p.get(t, at[k2 + 1] as usize);
                        after += p.get(s, at[k2 + 1] as usize);
                    }
                }
                let delta = after - before;
                alt_partition += (delta / t_cal).clamp(-40.0, 20.0).exp();
            }
        }

        let p_boltz = 1.0 / (1.0 + alt_partition);
        res.confidence[s] = (res.confidence[s] * p_boltz).clamp(0.0, 1.0);
    }
}

/// Final reported marginal confidence calibration (Pass 2):
/// combines Pass-1/2 consensus `c0` with single-spin Boltzmann beliefs `b1` and second-order
/// belief-weighted sequential damping (`clamp_w = (b1 / 0.85)^2`), including rigid register
/// translation `[a..=b] -> [k2 .. k2 + L]` (`L = 2..=5`) for short stretches lacking a high-belief
/// (`>= 0.80`) sequential anchor at either boundary so unanchored short islands cannot inflate
/// interior residues above `0.80` via mutual clamping.
fn calibrate_final_marginal_confidence(res: &mut AssignResult, u: &Matrix2D, p: &Matrix2D) {
    let c0 = res.confidence.clone();
    calibrate_boltzmann_confidence(res, u, p);

    let k_len = u.cols;
    let s_len = u.rows;
    if k_len <= 1 || s_len == 0 {
        return;
    }

    let mut at = vec![-1i32; k_len];
    for (s, &k) in res.position.iter().enumerate() {
        if k >= 0 && (k as usize) < k_len {
            at[k as usize] = s as i32;
        }
    }

    let b1 = res.confidence.clone();
    let clamp_w = |spin_idx: usize| -> f64 {
        let r = (b1[spin_idx] / 0.85).clamp(0.0, 1.0);
        r * r
    };
    let weighted_site = |spin: usize, slot: usize| -> f64 {
        let mut sc = u.get(spin, slot);
        if slot > 0 && at[slot - 1] >= 0 {
            let sl = at[slot - 1] as usize;
            sc += clamp_w(sl) * p.get(sl, spin);
        }
        if slot + 1 < k_len && at[slot + 1] >= 0 {
            let sr = at[slot + 1] as usize;
            sc += clamp_w(sr) * p.get(spin, sr);
        }
        sc
    };

    let mut bp_unassign = vec![f64::NEG_INFINITY; s_len];
    let mut bp_frag_dest = vec![vec![f64::NEG_INFINITY; k_len]; s_len];
    let max_island_len = 5usize.min(k_len.saturating_sub(1));

    for a in 1..k_len {
        if at[a] < 0 {
            continue;
        }
        let left_anchored = a > 0
            && at[a - 1] >= 0
            && b1[at[a - 1] as usize] >= 0.80
            && p.get(at[a - 1] as usize, at[a] as usize) >= 1.5;
        if left_anchored {
            continue;
        }
        for len in 2..=max_island_len {
            let b = a + len - 1;
            if b >= k_len || at[b] < 0 {
                break;
            }
            let right_anchored = b + 1 < k_len
                && at[b + 1] >= 0
                && b1[at[b + 1] as usize] >= 0.80
                && p.get(at[b] as usize, at[b + 1] as usize) >= 1.5;
            if right_anchored {
                continue;
            }

            let mut u_self_sum = 0.0;
            let mut internal_p = 0.0;
            for off in 0..len {
                u_self_sum += u.get(at[a + off] as usize, a + off);
                if off + 1 < len {
                    internal_p += p.get(at[a + off] as usize, at[a + off + 1] as usize);
                }
            }
            let mut bnd_self = 0.0;
            if a > 0 && at[a - 1] >= 0 {
                let sl = at[a - 1] as usize;
                bnd_self += clamp_w(sl) * p.get(sl, at[a] as usize);
            }
            if b + 1 < k_len && at[b + 1] >= 0 {
                let sr = at[b + 1] as usize;
                bnd_self += clamp_w(sr) * p.get(at[b] as usize, sr);
            }
            let e_frag_self = u_self_sum + bnd_self;
            let delta_un = -(e_frag_self + internal_p);
            for &spin_raw in &at[a..=b] {
                let sj = spin_raw as usize;
                if delta_un > bp_unassign[sj] {
                    bp_unassign[sj] = delta_un;
                }
            }

            for k2 in 1..=(k_len - len) {
                if k2 == a {
                    continue;
                }
                if !(0..len).all(|off| u.get(at[a + off] as usize, k2 + off) > FORBID * 0.5) {
                    continue;
                }
                let mut u_alt_sum = 0.0;
                let mut occ_cost = 0.0;
                let in_src = |idx: usize| idx >= a && idx <= b;
                let in_dst = |idx: usize| idx >= k2 && idx < k2 + len;
                for off in 0..len {
                    let q = k2 + off;
                    u_alt_sum += u.get(at[a + off] as usize, q);
                    if !in_src(q) && at[q] >= 0 {
                        let occ = at[q] as usize;
                        let mut e_occ = u.get(occ, q);
                        // Count external left edge only when `q - 1` is neither moved nor evicted
                        if q > 0 && !in_src(q - 1) && !in_dst(q - 1) && at[q - 1] >= 0 {
                            let sl = at[q - 1] as usize;
                            e_occ += clamp_w(sl) * p.get(sl, occ);
                        }
                        // Count internal evicted edge `(q, q + 1)` once, and skip moved `[a..=b]`
                        // since `bnd_self` already accounts for `(a - 1, a)` and `(b, b + 1)`
                        if q + 1 < k_len && !in_src(q + 1) && at[q + 1] >= 0 {
                            let sr = at[q + 1] as usize;
                            let edge_w = if in_dst(q + 1) {
                                clamp_w(occ).min(clamp_w(sr))
                            } else {
                                clamp_w(occ) * clamp_w(sr)
                            };
                            occ_cost += edge_w * p.get(occ, sr).max(0.0);
                        }
                        occ_cost += clamp_w(occ) * e_occ.max(0.0);
                    }
                }
                let mut bnd_alt = 0.0;
                if k2 > 0 && (k2 - 1 < a || k2 - 1 > b) && at[k2 - 1] >= 0 {
                    let sl = at[k2 - 1] as usize;
                    bnd_alt += clamp_w(sl) * p.get(sl, at[a] as usize);
                }
                if k2 + len < k_len && (k2 + len < a || k2 + len > b) && at[k2 + len] >= 0 {
                    let sr = at[k2 + len] as usize;
                    bnd_alt += clamp_w(sr) * p.get(at[b] as usize, sr);
                }
                let delta_frag = (u_alt_sum + bnd_alt - occ_cost) - e_frag_self;
                for off in 0..len {
                    let sj = at[a + off] as usize;
                    let k_dest = k2 + off;
                    if delta_frag > bp_frag_dest[sj][k_dest] {
                        bp_frag_dest[sj][k_dest] = delta_frag;
                    }
                }
            }
        }
    }

    let t_cal = 1.68_f64;
    for s in 0..s_len {
        let k_raw = res.position[s];
        if k_raw < 0 || b1[s] <= 0.0 {
            continue;
        }
        let k = k_raw as usize;
        let e_self_w = weighted_site(s, k);
        let un_w = (-e_self_w).max(bp_unassign[s]);
        let mut alt_w = (un_w / t_cal).clamp(-40.0, 20.0).exp();

        for t in 0..s_len {
            if t != s && res.position[t] < 0 && u.get(t, k) > FORBID * 0.5 {
                let delta = weighted_site(t, k) - e_self_w;
                alt_w += (delta / t_cal).clamp(-40.0, 20.0).exp();
            }
        }
        for k2 in 1..k_len {
            if k2 == k {
                continue;
            }
            let mut best_k2 = bp_frag_dest[s][k2];
            if u.get(s, k2) > FORBID * 0.5 {
                let mut e_k2 = u.get(s, k2);
                if k2 > 0 && k2 - 1 != k && at[k2 - 1] >= 0 {
                    let sl = at[k2 - 1] as usize;
                    e_k2 += clamp_w(sl) * p.get(sl, s);
                }
                if k2 + 1 < k_len && k2 + 1 != k && at[k2 + 1] >= 0 {
                    let sr = at[k2 + 1] as usize;
                    e_k2 += clamp_w(sr) * p.get(s, sr);
                }
                let occ_cost = if at[k2] >= 0 {
                    let t = at[k2] as usize;
                    clamp_w(t) * weighted_site(t, k2).max(0.0)
                } else {
                    0.0
                };
                best_k2 = best_k2.max(e_k2 - occ_cost - e_self_w);
            }
            if best_k2 > f64::NEG_INFINITY * 0.5 {
                alt_w += (best_k2 / t_cal).clamp(-40.0, 20.0).exp();
            }
        }
        let p_bp = 1.0 / (1.0 + alt_w);
        let p_prod = b1[s].min(c0[s] * p_bp).clamp(0.0, 1.0);
        let p_boltz1 = if c0[s] > 1e-9 {
            (b1[s] / c0[s]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p_upper = c0[s].min(p_bp).min(p_boltz1);
        // Fréchet copula blend below the 0.80 HC boundary: corrects quadratic c0*p_bp
        // under-confidence for low/medium posteriors without ever promoting <0.80 above 0.80.
        let sub_hc_taper = (1.0 - p_prod / 0.80).max(0.0);
        res.confidence[s] = (p_prod + 0.56 * (p_upper - p_prod) * sub_hc_taper).clamp(0.0, 1.0);
    }
}

// ============================================================================
// Component 4A: 2-Layer Sequence-Graph Self-Consistency Check
// ============================================================================

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
fn apply_graph_self_consistency(
    u_pass2: &mut Matrix2D,
    u_pass1: &Matrix2D,
    p: &Matrix2D,
    systems: &[SpinSystem],
    seq: &[String],
    shifted_priors: &Priors,
    offsets: &ReferenceOffsets,
    pass1: &AssignResult,
    locks: &[(usize, usize)],
) -> (Vec<bool>, usize, usize) {
    let k_len = seq.len();
    let mut at = vec![-1i32; k_len];
    for (s, &k) in pass1.position.iter().enumerate() {
        if k >= 0 && (k as usize) < k_len {
            at[k as usize] = s as i32;
        }
    }

    // Extract secondary chemical-shift index chi_k = 0.55*(dCA - dCB) + 0.45*dC' at assigned nodes
    let mut chi = vec![0.0_f64; k_len];
    let mut has_chi = vec![false; k_len];
    for k in 1..k_len {
        let s_raw = at[k];
        if s_raw < 0 {
            continue;
        }
        let s = s_raw as usize;
        let ss = &systems[s];
        let rname = &seq[k];
        let next_pro = k + 1 < k_len && seq[k + 1] == "PRO";
        let rc = random_coil_shifts(rname, false, next_pro);
        let cov_k = delta_covalent(seq, k);

        let mut dca = 0.0;
        let mut dcb = 0.0;
        let mut dc = 0.0;
        let mut cnt = 0usize;

        if let Some(cands) = ss.cands.get(&("CA".to_string(), 0)) {
            let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
            if !obs.is_empty() {
                let (_, idx) =
                    candidate_llr_shifted(shifted_priors, rname, "CA", &obs, 0.7, cov_k.1);
                dca = (obs[idx].0 - offsets.d_ca) - rc.ca;
                cnt += 1;
            }
        }
        if rname != "GLY" {
            if let Some(cands) = ss.cands.get(&("CB".to_string(), 0)) {
                let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
                if !obs.is_empty() {
                    let (_, idx) =
                        candidate_llr_shifted(shifted_priors, rname, "CB", &obs, 0.7, cov_k.2);
                    dcb = (obs[idx].0 - offsets.d_cb) - rc.cb;
                    cnt += 1;
                }
            }
        } else {
            dcb = -0.35 * dca;
        }
        if let Some(cands) = ss.cands.get(&("C".to_string(), 0)) {
            let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
            if !obs.is_empty() {
                let (_, idx) =
                    candidate_llr_shifted(shifted_priors, rname, "C", &obs, 0.7, cov_k.3);
                dc = (obs[idx].0 - offsets.d_c) - rc.c;
                cnt += 1;
            }
        } else if k + 1 < k_len && at[k + 1] >= 0 {
            let ns = at[k + 1] as usize;
            if let Some(cands) = systems[ns].cands.get(&("C".to_string(), -1)) {
                let obs: Vec<(f64, usize)> = cands.iter().copied().filter(|c| c.1 > 0).collect();
                if !obs.is_empty() {
                    let (_, idx) =
                        candidate_llr_shifted(shifted_priors, rname, "C", &obs, 0.7, cov_k.3);
                    dc = (obs[idx].0 - offsets.d_c) - rc.c;
                    cnt += 1;
                }
            }
        }

        if cnt > 0 {
            chi[k] = 0.55 * (dca - dcb) + 0.45 * dc;
            has_chi[k] = true;
        }
    }

    // Edge reliability weights w(k, k+1) in [0, 1]
    let mut edge_w = vec![0.0_f64; k_len.saturating_sub(1)];
    for k in 0..k_len.saturating_sub(1) {
        if at[k] >= 0 && at[k + 1] >= 0 {
            let p_sc = p.get(at[k] as usize, at[k + 1] as usize);
            edge_w[k] = sigmoid((p_sc - 1.0) / 2.8);
        }
    }

    // Layer 1: Combine consensus, unary fit, sequential edge reliability, and secondary-shift cooperativity
    let mut h1 = vec![0.0_f64; k_len];
    for k in 1..k_len {
        let s_raw = at[k];
        if s_raw < 0 {
            continue;
        }
        let s = s_raw as usize;
        let q0 = pass1.confidence.get(s).copied().unwrap_or(0.0);
        let u_sig = sigmoid(u_pass1.get(s, k) / 4.0);

        let mut nbr_sum = 0.0;
        let mut nbr_cnt = 0usize;
        if k > 1 && seq[k - 1] != "PRO" {
            nbr_sum += edge_w[k - 1];
            nbr_cnt += 1;
        }
        if k + 1 < k_len && seq[k + 1] != "PRO" {
            nbr_sum += edge_w[k];
            nbr_cnt += 1;
        }
        let w_nbr = if nbr_cnt > 0 {
            nbr_sum / (nbr_cnt as f64)
        } else {
            0.5
        };

        // Secondary-shift cooperativity across (k-1, k, k+1)
        let mut coop = 0.5_f64;
        if has_chi[k]
            && seq[k] != "GLY"
            && k > 1
            && k + 1 < k_len
            && has_chi[k - 1]
            && has_chi[k + 1]
        {
            let c_l = chi[k - 1];
            let c_r = chi[k + 1];
            if c_l.abs() > 1.0 && c_r.abs() > 1.0 && c_l.signum() == c_r.signum() {
                let prod = chi[k] * c_l.signum();
                coop = sigmoid(prod / 1.2);
            }
        }

        h1[k] = 0.42 * q0 + 0.20 * u_sig + 0.30 * w_nbr + 0.08 * coop;
    }

    // Layer 2: Propagate node consistency across verified sequential edges
    let mut h2 = vec![0.0_f64; k_len];
    for k in 1..k_len {
        if at[k] < 0 {
            continue;
        }
        let left_msg = if k > 1 {
            edge_w[k - 1] * h1[k - 1]
        } else {
            0.0
        };
        let right_msg = if k + 1 < k_len {
            edge_w[k] * h1[k + 1]
        } else {
            0.0
        };
        let norm =
            0.55 + if k > 1 && seq[k - 1] != "PRO" {
                0.225 * edge_w[k - 1]
            } else {
                0.0
            } + if k + 1 < k_len && seq[k + 1] != "PRO" {
                0.225 * edge_w[k]
            } else {
                0.0
            };
        h2[k] = (0.55 * h1[k] + 0.225 * left_msg + 0.225 * right_msg) / norm.max(0.55);
    }

    let mut pruned_slot = vec![false; k_len];
    let mut anchored_count = 0usize;
    let mut pruned_count = 0usize;

    for k in 1..k_len {
        let s_raw = at[k];
        if s_raw < 0 {
            continue;
        }
        let s = s_raw as usize;
        let q0 = pass1.confidence.get(s).copied().unwrap_or(0.0);
        let cur_u = u_pass2.get(s, k);
        if cur_u <= FORBID * 0.5 {
            continue;
        }

        let strong_left = k > 1
            && at[k - 1] >= 0
            && p.get(at[k - 1] as usize, s) >= 3.5
            && pass1
                .confidence
                .get(at[k - 1] as usize)
                .copied()
                .unwrap_or(0.0)
                >= 0.70;
        let strong_right = k + 1 < k_len
            && at[k + 1] >= 0
            && p.get(s, at[k + 1] as usize) >= 3.5
            && pass1
                .confidence
                .get(at[k + 1] as usize)
                .copied()
                .unwrap_or(0.0)
                >= 0.70;

        let clash_left = k > 1
            && at[k - 1] >= 0
            && pass1
                .confidence
                .get(at[k - 1] as usize)
                .copied()
                .unwrap_or(0.0)
                >= 0.80
            && p.get(at[k - 1] as usize, s) < -2.5;
        let clash_right = k + 1 < k_len
            && at[k + 1] >= 0
            && pass1
                .confidence
                .get(at[k + 1] as usize)
                .copied()
                .unwrap_or(0.0)
                >= 0.80
            && p.get(s, at[k + 1] as usize) < -2.5;

        if h2[k] >= 0.78
            && q0 >= 0.75
            && (strong_left || strong_right)
            && !clash_left
            && !clash_right
        {
            // Soft-anchor verified sequential stretch
            u_pass2.set(s, k, cur_u + 1.8 * h2[k]);
            anchored_count += 1;
        } else if !locks.iter().any(|&(sl, kl)| sl == s || kl == k)
            && (clash_left || clash_right || h2[k] < 0.28)
            && q0 < 0.45
        {
            // Prune/penalize low-consistency outlier clashing with anchored neighbors
            u_pass2.set(s, k, cur_u - 2.8);
            pruned_slot[k] = true;
            pruned_count += 1;
        }
    }

    (pruned_slot, anchored_count, pruned_count)
}

// ============================================================================
// Component 3: Hybrid Evolutionary Fragment Crossover + Local Simulated Annealing
// ============================================================================

#[derive(Clone, Debug)]
struct Individual {
    /// Residue index `k` -> spin system index `s` (`-1` if unassigned).
    at: Vec<i32>,
    /// Spin system index `s` -> residue index `k` (`-1` if unassigned).
    pos: Vec<i32>,
    score: f64,
}

impl Individual {
    fn empty(s_len: usize, k_len: usize) -> Self {
        Self {
            at: vec![-1; k_len],
            pos: vec![-1; s_len],
            score: 0.0,
        }
    }

    fn from_pos(pos: &[i32], k_len: usize) -> Self {
        let s_len = pos.len();
        let mut at = vec![-1; k_len];
        let mut clean_pos = vec![-1; s_len];
        for (s, &k) in pos.iter().enumerate() {
            if k >= 0 && (k as usize) < k_len && at[k as usize] < 0 {
                at[k as usize] = s as i32;
                clean_pos[s] = k;
            }
        }
        Self {
            at,
            pos: clean_pos,
            score: 0.0,
        }
    }
}

fn evaluate_total_score(at: &[i32], u: &Matrix2D, p: &Matrix2D) -> f64 {
    let k_len = at.len();
    let mut sc = 0.0;
    for (k, &s) in at.iter().enumerate() {
        if s >= 0 {
            sc += u.get(s as usize, k);
        }
    }
    for e in 0..k_len.saturating_sub(1) {
        let a = at[e];
        let b = at[e + 1];
        if a >= 0 && b >= 0 {
            sc += p.get(a as usize, b as usize);
        }
    }
    sc
}

struct LocalAnnealer<'a> {
    u: &'a Matrix2D,
    p: &'a Matrix2D,
    slots: &'a [Vec<usize>],
    locked_s: Vec<bool>,
    locked_k: Vec<bool>,
    s_len: usize,
    k_len: usize,
    edge_marks: Vec<u32>,
    mark_gen: u32,
}

impl<'a> LocalAnnealer<'a> {
    fn new(
        u: &'a Matrix2D,
        p: &'a Matrix2D,
        slots: &'a [Vec<usize>],
        locks: &[(usize, usize)],
    ) -> Self {
        let mut locked_s = vec![false; u.rows];
        let mut locked_k = vec![false; u.cols];
        for &(s_lock, k_lock) in locks {
            if s_lock < u.rows && k_lock < u.cols {
                locked_s[s_lock] = true;
                locked_k[k_lock] = true;
            }
        }
        Self {
            u,
            p,
            slots,
            locked_s,
            locked_k,
            s_len: u.rows,
            k_len: u.cols,
            edge_marks: vec![0; u.cols],
            mark_gen: 1,
        }
    }

    #[inline]
    fn local_score(&mut self, at: &[i32], changed: &[usize]) -> f64 {
        if self.k_len == 0 {
            return 0.0;
        }
        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.edge_marks.fill(0);
            self.mark_gen = 1;
        }
        let gen = self.mark_gen;
        let mut sc = 0.0;
        let mut edges = [0usize; 64];
        let mut n_edges = 0usize;

        for &k in changed {
            let s = at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
            if k > 0 && k - 1 < self.k_len - 1 && self.edge_marks[k - 1] != gen {
                self.edge_marks[k - 1] = gen;
                edges[n_edges] = k - 1;
                n_edges += 1;
            }
            if k < self.k_len - 1 && self.edge_marks[k] != gen {
                self.edge_marks[k] = gen;
                edges[n_edges] = k;
                n_edges += 1;
            }
        }
        for &e in &edges[..n_edges] {
            let a = at[e];
            let b = at[e + 1];
            if a >= 0 && b >= 0 {
                sc += self.p.get(a as usize, b as usize);
            }
        }
        sc
    }

    #[inline]
    fn apply_changes(
        at: &mut [i32],
        pos: &mut [i32],
        changes: &[(usize, i32)],
        undo: &mut Vec<(usize, i32)>,
    ) {
        undo.clear();
        for &(k, _) in changes {
            undo.push((k, at[k]));
        }
        for &(k, _) in changes {
            let prev_s = at[k];
            if prev_s >= 0 {
                pos[prev_s as usize] = -1;
            }
        }
        for &(k, s) in changes {
            at[k] = s;
        }
        for &(k, s) in changes {
            if s >= 0 {
                pos[s as usize] = k as i32;
            }
        }
    }

    fn greedy_fill_gaps(&mut self, ind: &mut Individual, rng: &mut Rng) {
        let mut free_s: Vec<usize> = (0..self.s_len).filter(|&s| ind.pos[s] < 0).collect();
        rng.shuffle(&mut free_s);
        for s in free_s {
            if self.slots[s].is_empty() {
                continue;
            }
            let mut best_k: Option<usize> = None;
            let mut best_gain = 0.0_f64;
            for &k in &self.slots[s] {
                if ind.at[k] >= 0 {
                    continue;
                }
                let mut gain = self.u.get(s, k);
                if k > 0 && ind.at[k - 1] >= 0 {
                    gain += self.p.get(ind.at[k - 1] as usize, s);
                }
                if k + 1 < self.k_len && ind.at[k + 1] >= 0 {
                    gain += self.p.get(s, ind.at[k + 1] as usize);
                }
                if gain > best_gain {
                    best_gain = gain;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                ind.at[k] = s as i32;
                ind.pos[s] = k as i32;
            }
        }
    }

    fn anneal(
        &mut self,
        ind: &mut Individual,
        rng: &mut Rng,
        steps: usize,
        t0: f64,
        t1: f64,
        max_block: usize,
    ) {
        if self.s_len == 0 || self.k_len == 0 || steps == 0 {
            ind.score = evaluate_total_score(&ind.at, self.u, self.p);
            return;
        }
        let mut cur = evaluate_total_score(&ind.at, self.u, self.p);
        let mut best = cur;
        let mut best_at = ind.at.clone();

        let mut changes: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let mut affected: Vec<usize> = Vec::with_capacity(max_block * 2 + 4);
        let mut undo: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let mut dummy: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let denom = (steps.saturating_sub(1)).max(1) as f64;
        let ratio = (t1 / t0.max(1e-6)).clamp(1e-6, 1.0);

        for step in 0..steps {
            let temp = t0 * ratio.powf((step as f64) / denom);
            changes.clear();

            let r = rng.f64();
            if r < 0.65 {
                let s = rng.usize(self.s_len);
                if self.slots[s].is_empty() || self.locked_s[s] {
                    continue;
                }
                let k = self.slots[s][rng.usize(self.slots[s].len())];
                if self.locked_k[k] {
                    continue;
                }
                let ks = ind.pos[s];
                let t = ind.at[k];
                if ks == k as i32 {
                    changes.push((k, -1));
                } else {
                    if t >= 0 && self.locked_s[t as usize] {
                        continue;
                    }
                    changes.push((k, s as i32));
                    if ks >= 0 {
                        let ks_u = ks as usize;
                        let repl = if t >= 0 && self.u.get(t as usize, ks_u) > FORBID * 0.5 {
                            t
                        } else {
                            -1
                        };
                        changes.push((ks_u, repl));
                    }
                }
            } else if r < 0.90 {
                let max_l = max_block.min(self.k_len / 2);
                if max_l == 0 {
                    continue;
                }
                let l = rng.range_usize(1, max_l + 1);
                let k1 = rng.usize(self.k_len - l + 1);
                let k2 = rng.usize(self.k_len - l + 1);
                if k1.abs_diff(k2) < l {
                    continue;
                }
                let mut valid = true;
                for i in 0..l {
                    if self.locked_k[k1 + i] || self.locked_k[k2 + i] {
                        valid = false;
                        break;
                    }
                    let a = ind.at[k1 + i];
                    let b = ind.at[k2 + i];
                    if (b >= 0 && self.u.get(b as usize, k1 + i) <= FORBID * 0.5)
                        || (a >= 0 && self.u.get(a as usize, k2 + i) <= FORBID * 0.5)
                    {
                        valid = false;
                        break;
                    }
                    changes.push((k1 + i, b));
                    changes.push((k2 + i, a));
                }
                if !valid {
                    continue;
                }
            } else {
                let s = rng.usize(self.s_len);
                if ind.pos[s] < 0 || self.locked_s[s] {
                    continue;
                }
                changes.push((ind.pos[s] as usize, -1));
            }

            if changes.is_empty() {
                continue;
            }
            affected.clear();
            for &(k, _) in &changes {
                affected.push(k);
            }
            let before = self.local_score(&ind.at, &affected);
            Self::apply_changes(&mut ind.at, &mut ind.pos, &changes, &mut undo);
            let delta = self.local_score(&ind.at, &affected) - before;
            if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best {
                    best = cur;
                    best_at.copy_from_slice(&ind.at);
                }
            } else {
                Self::apply_changes(&mut ind.at, &mut ind.pos, &undo, &mut dummy);
            }
        }

        ind.at = best_at;
        ind.pos.fill(-1);
        for (k, &s) in ind.at.iter().enumerate() {
            if s >= 0 {
                ind.pos[s as usize] = k as i32;
            }
        }
        ind.score = best;
    }
}

/// Extract contiguous well-linked sequential fragments (`P(s_k, s_{k+1}) > link_threshold`)
/// from `donor` and splice a selected stretch cleanly into `child` (cloned from `receiver`),
/// evicting any conflicting spin systems to maintain strict injectivity.
fn crossover_sequential_fragment(
    donor: &Individual,
    receiver: &Individual,
    u: &Matrix2D,
    p: &Matrix2D,
    link_threshold: f64,
    rng: &mut Rng,
) -> Individual {
    let k_len = receiver.at.len();
    let mut child = receiver.clone();
    if k_len < 3 {
        return child;
    }

    // Identify all contiguous assigned stretches in donor where adjacent edges exceed link_threshold
    let mut fragments: Vec<(usize, usize, f64)> = Vec::new();
    let mut k = 1usize;
    while k < k_len {
        if donor.at[k] < 0 {
            k += 1;
            continue;
        }
        let start = k;
        let mut end = k;
        let mut frag_sc = u.get(donor.at[k] as usize, k);
        while end + 1 < k_len && donor.at[end + 1] >= 0 {
            let edge = p.get(donor.at[end] as usize, donor.at[end + 1] as usize);
            if edge > link_threshold {
                end += 1;
                frag_sc += edge + u.get(donor.at[end] as usize, end);
            } else {
                break;
            }
        }
        if end > start {
            // Prefer stretches that actually differ somewhere from receiver
            let differs = (start..=end).any(|idx| donor.at[idx] != receiver.at[idx]);
            if differs {
                fragments.push((start, end, frag_sc.max(0.1)));
            }
        }
        k = end + 1;
    }

    let (splice_start, splice_end) = if !fragments.is_empty() {
        // Weighted roulette selection over differing well-linked fragments
        let tot_w: f64 = fragments.iter().map(|f| f.2).sum();
        let mut pick = rng.f64() * tot_w;
        let mut chosen = (fragments[0].0, fragments[0].1);
        for &(s, e, w) in &fragments {
            if pick <= w {
                chosen = (s, e);
                break;
            }
            pick -= w;
        }
        // If the stretch is very long (> k_len / 2), optionally take a random contiguous sub-window
        let len = chosen.1 - chosen.0 + 1;
        let max_win = (k_len / 2).max(3);
        if len > max_win {
            let sub_l = rng.range_usize(2, max_win + 1);
            let sub_s = chosen.0 + rng.usize(len - sub_l + 1);
            (sub_s, sub_s + sub_l - 1)
        } else {
            chosen
        }
    } else {
        // Fallback: random contiguous sequence window [start ..= end]
        let max_w = (k_len / 3).clamp(2, 12);
        let l = rng.range_usize(2, max_w + 1).min(k_len - 1);
        let s = rng.range_usize(1, k_len - l + 1);
        (s, s + l - 1)
    };

    for idx in splice_start..=splice_end {
        let s_donor = donor.at[idx];
        if s_donor < 0 || u.get(s_donor as usize, idx) <= FORBID * 0.5 {
            continue;
        }
        // Evict occupant currently at `idx` in child
        let prev_occupant = child.at[idx];
        if prev_occupant >= 0 && prev_occupant != s_donor {
            child.pos[prev_occupant as usize] = -1;
            child.at[idx] = -1;
        }
        // Evict `s_donor` from its previous residue position in child if placed elsewhere
        let prev_k = child.pos[s_donor as usize];
        if prev_k >= 0 && prev_k as usize != idx {
            child.at[prev_k as usize] = -1;
        }
        child.at[idx] = s_donor;
        child.pos[s_donor as usize] = idx as i32;
    }

    child
}

fn tournament_select(
    pop: &[Individual],
    k_tour: usize,
    avoid: Option<usize>,
    rng: &mut Rng,
) -> usize {
    let n = pop.len();
    let mut best_idx = rng.usize(n);
    for _ in 0..k_tour.max(1) {
        let cand = rng.usize(n);
        if Some(cand) == avoid && n > 1 {
            continue;
        }
        if Some(best_idx) == avoid || pop[cand].score > pop[best_idx].score {
            best_idx = cand;
        }
    }
    best_idx
}

fn enforce_locks_on_individual(ind: &mut Individual, locks: &[(usize, usize)]) {
    for &(s_lock, k_lock) in locks {
        if s_lock >= ind.pos.len() || k_lock >= ind.at.len() {
            continue;
        }
        let prev_s = ind.at[k_lock];
        if prev_s >= 0 && (prev_s as usize) != s_lock {
            ind.pos[prev_s as usize] = -1;
        }
        let prev_k = ind.pos[s_lock];
        if prev_k >= 0 && (prev_k as usize) != k_lock {
            ind.at[prev_k as usize] = -1;
        }
        ind.at[k_lock] = s_lock as i32;
        ind.pos[s_lock] = k_lock as i32;
    }
}

#[allow(clippy::too_many_arguments)]
fn evolve_island(
    u: &Matrix2D,
    p: &Matrix2D,
    slots: &[Vec<usize>],
    island_pop_size: usize,
    cfg: &EvoGraphConfig<'_>,
    locks: &[(usize, usize)],
    warm_seed: Option<&[i32]>,
    seed: u64,
) -> Individual {
    let s_len = u.rows;
    let k_len = u.cols;
    let pop_m = island_pop_size.max(2);
    let mut rng = Rng::new(seed);
    let mut annealer = LocalAnnealer::new(u, p, slots, locks);
    let steps_gen0 = cfg.steps_per_system_per_gen.max(50) * s_len;
    let steps_gen = ((cfg.steps_per_system_per_gen * 4) / 5).max(40) * s_len;

    // Generation 0: initialize population (with optional warm survivor from Pass 1 of THIS island)
    let mut pop: Vec<Individual> = Vec::with_capacity(pop_m);
    for i in 0..pop_m {
        let mut ind = if i == 0 {
            if let Some(w_pos) = warm_seed {
                Individual::from_pos(w_pos, k_len)
            } else {
                Individual::empty(s_len, k_len)
            }
        } else {
            Individual::empty(s_len, k_len)
        };
        if !locks.is_empty() {
            enforce_locks_on_individual(&mut ind, locks);
        }
        let t0 = if i == 0 && warm_seed.is_some() {
            (cfg.t_start * 0.45).max(1.5)
        } else {
            cfg.t_start
        };
        if i == 0 && warm_seed.is_some() {
            annealer.greedy_fill_gaps(&mut ind, &mut rng);
        }
        annealer.anneal(&mut ind, &mut rng, steps_gen0, t0, cfg.t_end, cfg.max_block);
        pop.push(ind);
    }

    let gens = cfg.generations.max(1);
    for g in 1..gens {
        pop.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut next_pop: Vec<Individual> = Vec::with_capacity(pop_m);
        // Elitism: preserve the island's best individual (and polish lightly at low T)
        let mut elite = pop[0].clone();
        annealer.anneal(
            &mut elite,
            &mut rng,
            steps_gen / 3,
            (cfg.t_end * 8.0).max(0.35),
            cfg.t_end,
            cfg.max_block,
        );
        if elite.score < pop[0].score {
            elite = pop[0].clone();
        }
        next_pop.push(elite);

        let reheat_t0 = (cfg.t_start * 0.55 * 0.78_f64.powi((g - 1) as i32)).max(0.85);
        while next_pop.len() < pop_m {
            let pb = tournament_select(&pop, cfg.tournament_size, None, &mut rng);
            let mut child = if rng.f64() < cfg.crossover_rate && pop.len() >= 2 {
                let pa = tournament_select(&pop, cfg.tournament_size, Some(pb), &mut rng);
                let mut c = crossover_sequential_fragment(
                    &pop[pa],
                    &pop[pb],
                    u,
                    p,
                    cfg.fragment_link_threshold,
                    &mut rng,
                );
                if !locks.is_empty() {
                    enforce_locks_on_individual(&mut c, locks);
                }
                annealer.greedy_fill_gaps(&mut c, &mut rng);
                c
            } else {
                pop[pb].clone()
            };
            annealer.anneal(
                &mut child,
                &mut rng,
                steps_gen,
                reheat_t0,
                cfg.t_end,
                cfg.max_block,
            );
            next_pop.push(child);
        }
        pop = next_pop;
    }

    pop.into_iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap_or_else(|| Individual::empty(s_len, k_len))
}

fn run_parallel_islands(
    u: &Matrix2D,
    p: &Matrix2D,
    u_eval: &Matrix2D,
    cfg: &EvoGraphConfig<'_>,
    locks: &[(usize, usize)],
    warm_positions: Option<&[Vec<i32>]>,
    master_rng: &mut Rng,
) -> (Vec<Vec<i32>>, Vec<f64>) {
    let s_len = u.rows;
    let k_len = u.cols;
    let n_islands = cfg.num_islands.max(1);
    let island_pop = cfg.population_size.div_ceil(n_islands).max(2);

    let slots: Vec<Vec<usize>> = (0..s_len)
        .map(|s| {
            (0..k_len)
                .filter(|&k| u.get(s, k) > FORBID * 0.5)
                .collect::<Vec<usize>>()
        })
        .collect();

    let seeds: Vec<u64> = (0..n_islands).map(|_| master_rng.next_u64()).collect();
    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_islands);
    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;

    let mut winners: Vec<Individual> = Vec::with_capacity(n_islands);
    if n_threads <= 1 || n_islands <= 1 {
        for (idx, &sd) in seeds.iter().enumerate() {
            let warm = warm_positions.and_then(|wp| wp.get(idx).map(Vec::as_slice));
            winners.push(evolve_island(
                u, p, &slots, island_pop, cfg, locks, warm, sd,
            ));
        }
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let indices: Vec<usize> = (0..n_islands).collect();
            let chunk_len = n_islands.div_ceil(n_threads);
            let mut parts: Vec<Vec<Individual>> = Vec::with_capacity(n_threads);
            std::thread::scope(|scope| {
                let slots_ref = &slots;
                let seeds_ref = &seeds;
                let mut handles = Vec::new();
                for idx_chunk in indices.chunks(chunk_len) {
                    handles.push(scope.spawn(move || {
                        let mut local = Vec::with_capacity(idx_chunk.len());
                        for &isl_idx in idx_chunk {
                            let warm =
                                warm_positions.and_then(|wp| wp.get(isl_idx).map(Vec::as_slice));
                            local.push(evolve_island(
                                u,
                                p,
                                slots_ref,
                                island_pop,
                                cfg,
                                locks,
                                warm,
                                seeds_ref[isl_idx],
                            ));
                        }
                        local
                    }));
                }
                for h in handles {
                    if let Ok(part) = h.join() {
                        parts.push(part);
                    }
                }
            });
            for part in parts {
                winners.extend(part);
            }
        }
    }

    let mut positions = Vec::with_capacity(winners.len());
    let mut scores = Vec::with_capacity(winners.len());
    for w in winners {
        let phys_sc = evaluate_total_score(&w.at, u_eval, p);
        positions.push(w.pos);
        scores.push(phys_sc);
    }
    (positions, scores)
}

fn apply_locked_result_overrides(res: &mut AssignResult, k_len: usize, locks: &[(usize, usize)]) {
    for &(s_lock, k_lock) in locks {
        if s_lock < res.position.len() && k_lock < k_len {
            res.position[s_lock] = k_lock as i32;
            res.confidence[s_lock] = 1.0;
        }
    }
}

/// Run the two-pass hybrid evolutionary fragment-crossover + local simulated annealing +
/// 2-layer sequence-graph self-consistency solver on pre-built `SpinSystem`s.
pub fn assign_evo_graph(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &EvoGraphConfig<'_>,
) -> AssignResult {
    assign_evo_graph_with_report(systems, sequence, priors, cfg).0
}

/// Run `assign_evo_graph` and return both `AssignResult` and diagnostic `EvoGraphReport`.
pub fn assign_evo_graph_with_report(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &EvoGraphConfig<'_>,
) -> (AssignResult, EvoGraphReport) {
    let mut report = EvoGraphReport {
        raw_hsqc_count: systems.len(),
        filtered_hsqc_count: systems.len(),
        kept_hsqc_indices: (0..systems.len()).collect(),
        ..Default::default()
    };
    let norm_seq = normalize_sequence(sequence);
    let raw_locks = merged_config_locks(cfg);
    let valid_locks = resolve_locked_assignments(systems.len(), &norm_seq, &raw_locks);
    let mut result =
        assign_evo_graph_core(systems, sequence, priors, cfg, &valid_locks, &mut report);
    let mut qc = evaluate_dataset_qc(
        None,
        systems,
        sequence,
        priors,
        &cfg.score_cfg,
        &result,
        None,
    );
    if cfg.apply_qc_calibration {
        apply_dataset_qc_to_assignments(&mut result, &qc);
        qc.already_applied = true;
    }
    apply_locked_result_overrides(&mut result, norm_seq.len(), &valid_locks);
    report.qc_report = qc;
    (result, report)
}

fn assign_evo_graph_core(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &EvoGraphConfig<'_>,
    locks: &[(usize, usize)],
    report: &mut EvoGraphReport,
) -> AssignResult {
    let norm_seq = normalize_sequence(sequence);
    if systems.is_empty() || norm_seq.is_empty() {
        return AssignResult {
            position: vec![-1; systems.len()],
            confidence: vec![0.0; systems.len()],
            score: 0.0,
            run_scores: Vec::new(),
        };
    }

    let assignable_slots = (1..norm_seq.len())
        .filter(|&k| norm_seq[k] != "PRO")
        .count();
    let effective_assign_cost = if systems.len() > assignable_slots {
        cfg.score_cfg.assign_cost.max(cfg.min_assign_cost)
    } else {
        cfg.score_cfg.assign_cost.max(cfg.min_assign_cost * 0.5)
    };

    // Pass 1: Mixture reference-offset estimation + initial evolutionary assignment
    let pass1_offsets = if cfg.auto_reference {
        estimate_mixture_reference_offsets(systems, &norm_seq, priors)
    } else {
        ReferenceOffsets::default()
    };
    report.pass1_offsets = pass1_offsets;
    let priors_pass1 = apply_offsets_to_priors(priors, &pass1_offsets);

    let u1 = build_evo_unary_matrix(
        systems,
        &norm_seq,
        &priors_pass1,
        cfg,
        effective_assign_cost,
        locks,
    );
    let p = pair_matrix(systems, &cfg.score_cfg);

    let mut master_rng = Rng::new(cfg.seed);
    let (pos1, scores1) = run_parallel_islands(&u1, &p, &u1, cfg, locks, None, &mut master_rng);
    let mut res1 = consensus_from_runs(&pos1, scores1);
    calibrate_boltzmann_confidence(&mut res1, &u1, &p);
    apply_locked_result_overrides(&mut res1, norm_seq.len(), locks);
    report.pass1_best_score = res1.score;

    // Pass 2: High-consensus anchor reference offset refinement + 2-layer graph self-consistency
    let (final_offsets, n_anchors) = if cfg.auto_reference {
        refine_anchor_reference_offsets_with_torsions(
            systems,
            &norm_seq,
            priors,
            &res1,
            cfg.anchor_consensus_threshold,
            pass1_offsets,
            cfg.structure_torsions,
        )
    } else {
        (ReferenceOffsets::default(), 0)
    };
    report.final_offsets = final_offsets;
    report.pass2_anchor_count = n_anchors;

    let priors_pass2 = apply_offsets_to_priors(priors, &final_offsets);
    let u2_phys = build_evo_unary_matrix(
        systems,
        &norm_seq,
        &priors_pass2,
        cfg,
        effective_assign_cost,
        locks,
    );
    let mut u2_guided = u2_phys.clone();

    let mut warm_pos2 = pos1;
    if cfg.graph_consistency {
        let (pruned_slots, n_anc, n_prn) = apply_graph_self_consistency(
            &mut u2_guided,
            &u1,
            &p,
            systems,
            &norm_seq,
            &priors_pass2,
            &final_offsets,
            &res1,
            locks,
        );
        report.graph_anchored_count = n_anc;
        report.graph_pruned_count = n_prn;
        // Unlock pruned outlier slots inside each island's warm seed before Pass 2 recombination
        for wp in &mut warm_pos2 {
            for k_val in wp.iter_mut() {
                if *k_val >= 0
                    && (*k_val as usize) < pruned_slots.len()
                    && pruned_slots[*k_val as usize]
                {
                    *k_val = -1;
                }
            }
        }
    }

    let (pos2, scores2) = run_parallel_islands(
        &u2_guided,
        &p,
        &u2_phys,
        cfg,
        locks,
        Some(&warm_pos2),
        &mut master_rng,
    );
    let mut res2 = consensus_from_runs(&pos2, scores2);
    calibrate_final_marginal_confidence(&mut res2, &u2_phys, &p);
    apply_locked_result_overrides(&mut res2, norm_seq.len(), locks);
    report.final_best_score = res2.score;
    res2
}

/// End-to-end stoichiometry-guided peak filtering, intensity-weighted spin-system construction,
/// two-pass reference alignment, evolutionary fragment crossover + SA, and residue-graph
/// self-consistency directly from raw picked `Peak`s.
pub fn assign_evo_graph_from_peaks(
    peaks: &[Peak],
    sequence: &[String],
    priors: &Priors,
    cfg: &EvoGraphConfig<'_>,
) -> (Vec<SpinSystem>, AssignResult, EvoGraphReport) {
    let mut report = EvoGraphReport::default();
    let filtered = filter_peaks_with_report(peaks, sequence, cfg, &mut report);
    let systems = build_stoichiometric_spin_systems(&filtered, &cfg.tolerances, cfg.max_cands);
    let norm_seq = normalize_sequence(sequence);
    let raw_locks = merged_config_locks(cfg);
    let remapped_locks: Vec<(usize, usize)> = if raw_locks.is_empty() {
        Vec::new()
    } else {
        let valid_raw = resolve_locked_assignments(report.raw_hsqc_count, &norm_seq, &raw_locks);
        let mapped: Vec<(usize, usize)> = valid_raw
            .into_iter()
            .filter_map(|(raw_s, k)| {
                report
                    .kept_hsqc_indices
                    .iter()
                    .position(|&orig| orig == raw_s)
                    .map(|filt_s| (filt_s, k))
            })
            .collect();
        resolve_locked_assignments(systems.len(), &norm_seq, &mapped)
    };
    let mut result = assign_evo_graph_core(
        &systems,
        sequence,
        priors,
        cfg,
        &remapped_locks,
        &mut report,
    );
    let mut qc = evaluate_dataset_qc(
        Some(peaks),
        &systems,
        sequence,
        priors,
        &cfg.score_cfg,
        &result,
        None,
    );
    if cfg.apply_qc_calibration {
        apply_dataset_qc_to_assignments(&mut result, &qc);
        qc.already_applied = true;
    }
    apply_locked_result_overrides(&mut result, norm_seq.len(), &remapped_locks);
    report.qc_report = qc;
    (systems, result, report)
}
