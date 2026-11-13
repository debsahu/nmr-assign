//! Per-dataset quality-control verdict, unlabelled spectral self-consistency diagnostics,
//! per-protein confidence calibration / explicit abstention, and reliability curves.
//!
//! Provenance: Clean-room original implementation grounded in peer-reviewed calibration,
//! selective-prediction, and NMR referencing/stoichiometry self-consistency literature:
//! 1. Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities With Supervised
//!    Learning." *Proc. 22nd ICML*, 625–632 — reliability diagrams and per-bin empirical accuracy.
//! 2. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). "On Calibration of Modern Neural
//!    Networks." *Proc. 34th ICML*, PMLR 70, 1321–1330 — binned Expected Calibration Error (ECE).
//! 3. Chow, C. K. (1970). "On Optimum Recognition Error and Reject Tradeoff." *IEEE Trans. Inf.
//!    Theory* 16(1), 41–46 — explicit abstention when observable input evidence cannot support
//!    high-confidence classification.
//! 4. Zhang, H., Neal, S. & Wishart, D. S. (2003). "RefDB: a database of uniformly referenced
//!    protein chemical shifts." *J. Biomol. NMR* 25, 173–195 — cross-nucleus chemical-shift
//!    referencing consistency across `(1H, 15N, 13CA, 13CB, 13C')`.
//! 5. Schmidt, E. & Güntert, P. (2012). *J. Am. Chem. Soc.* 134(30), 12817–12829 — sequence-derived
//!    amide peak stoichiometry and sequential `(i, i-1)` connectivity self-consistency.
//! 6. Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. & Silverman, E. (1955). "An empirical
//!    distribution function for sampling with incomplete information." *Ann. Math. Statist.*
//!    26(4), 641–647; Zadrozny, B. & Elkan, C. (2002). "Transforming classifier scores into
//!    accurate multiclass probability estimates." *Proc. 8th ACM SIGKDD*, 694–699 — Pool-Adjacent-
//!    Violators (PAV) isotonic calibration.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::peaks::Peak;
use crate::priors::{
    estimate_reference_offsets_detailed, one_to_three, DetailedReferenceOffsets, OffsetFitStatus,
    Priors,
};
use crate::solver::{pair_matrix, AssignResult, ScoreConfig};
use crate::spinsystems::SpinSystem;

/// Per-dataset quality-control gate verdict computed strictly from unlabelled observables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetQcVerdict {
    /// Clean stoichiometry, 3D strip support, `(H, N)` registration, referencing, and connectivity.
    #[default]
    Pass,
    /// Multi-conformer crowding, partial 3D support, inter-spectrum drift, or offset inconsistency;
    /// per-residue confidences are conservatively damped by `confidence_scale`.
    Warn,
    /// Catastrophic 3D mismatch (`< 25%` supported roots) or sequential disconnection (`< 15%` links);
    /// all confidences are capped at `<= 0.45` so `n_high_conf == 0` (explicit abstention).
    Abstain,
}

impl DatasetQcVerdict {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Abstain => "ABSTAIN",
        }
    }
}

impl fmt::Display for DatasetQcVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unlabelled per-dataset quality-control telemetry and confidence calibration parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetQcReport {
    pub verdict: DatasetQcVerdict,
    /// Multiplicative confidence damping factor (`1.0` on `Pass`, `< 1.0` on `Warn`, `0.0` when `Abstain` caps).
    pub confidence_scale: f64,
    /// Upper confidence cap (`1.0` on `Pass` / `Warn`, `0.45` on `Abstain`).
    pub confidence_cap: f64,
    /// True once `apply_dataset_qc_to_assignments` has been applied to the `AssignResult` so
    /// downstream callers will not accidentally apply multiplicative damping a second time.
    pub already_applied: bool,
    pub n_assignable_amides: usize,
    pub n_raw_hsqc_roots: usize,
    pub n_amide_hsqc_roots: usize,
    /// Fraction of candidate amide 2D HSQC roots matched to `>= 1` 3D triple-resonance peak.
    pub hsqc_3d_support_frac: f64,
    /// Ratio of 3D-supported amide roots to sequence-assignable backbone amides (`n_supported / n_assignable`).
    pub supported_roots_per_assignable: f64,
    /// Empirical `1H` match residual RMS (ppm) between 2D HSQC roots and nearest 3D peaks across experiments.
    pub hsqc_3d_rms_dh_ppm: f64,
    /// Empirical `15N` match residual RMS (ppm) between 2D HSQC roots and nearest 3D peaks across experiments.
    pub hsqc_3d_rms_dn_ppm: f64,
    /// Candidate amide HSQC root multiplicity (`n_amide_hsqc_roots / n_assignable_amides`; flags `> 1.45`).
    pub hsqc_multiplicity_ratio: f64,
    /// Raw picked HSQC root multiplicity (`n_raw_hsqc_roots / n_assignable_amides`).
    pub raw_hsqc_multiplicity_ratio: f64,
    /// Whether fitted global chemical-shift offsets across `(H, N, CA, CB, C)` are physically self-consistent.
    pub offset_consistency_ok: bool,
    /// Per-nucleus reference offset diagnostics (`Fitted`, `NotSignificant`, `TooFewValues`).
    pub detailed_offsets: DetailedReferenceOffsets,
    /// Fraction of assigned spin systems supported by at least one sequential `(i, i-1)` link (`P >= 1.0`).
    pub connectivity_frac: f64,
    /// Fraction of assigned spin systems with strong consensus/marginal belief (`>= 0.65`).
    pub solution_margin: f64,
    /// Human-readable QC warnings or abstention triggers.
    pub reasons: Vec<String>,
}

impl Default for DatasetQcReport {
    fn default() -> Self {
        Self {
            verdict: DatasetQcVerdict::Pass,
            confidence_scale: 1.0,
            confidence_cap: 1.0,
            already_applied: false,
            n_assignable_amides: 0,
            n_raw_hsqc_roots: 0,
            n_amide_hsqc_roots: 0,
            hsqc_3d_support_frac: 1.0,
            supported_roots_per_assignable: 0.0,
            hsqc_3d_rms_dh_ppm: 0.0,
            hsqc_3d_rms_dn_ppm: 0.0,
            hsqc_multiplicity_ratio: 1.0,
            raw_hsqc_multiplicity_ratio: 1.0,
            offset_consistency_ok: true,
            detailed_offsets: DetailedReferenceOffsets::default(),
            connectivity_frac: 1.0,
            solution_margin: 1.0,
            reasons: Vec::new(),
        }
    }
}

fn normalize_seq(sequence: &[String]) -> Vec<String> {
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

/// Check physical consistency of `DetailedReferenceOffsets` across `(H, N, CA, CB, C)` (`P1-4`):
/// - Never treats `OffsetFitStatus::NotSignificant` or `OffsetFitStatus::TooFewValues` as evidence
///   of a `0.0 ppm` offset (`NotSignificant` means unknown / below deadband, not zero).
/// - Flags `aliphatic offset mismatch` strictly when **both** `dCA` and `dCB` are `Fitted` AND
///   differ by `> max(3.0 * sigma_joint, 0.85)` ppm.
/// - Flags `CA/CO offset mismatch` strictly when **both** `dCA` and `dC` are `Fitted` AND differ
///   by `> max(3.5 * sigma_ca_co, 2.0)` ppm.
/// - Flags extreme amide shifts `|dH| > 0.45 ppm` or `|dN| > 2.5 ppm` when `Fitted`.
pub fn check_offset_consistency(offsets: &DetailedReferenceOffsets) -> (bool, Vec<String>) {
    let mut issues = Vec::new();

    if offsets.ca.status == OffsetFitStatus::Fitted && offsets.cb.status == OffsetFitStatus::Fitted
    {
        let sigma_joint = offsets.aliphatic_joint_sigma_ppm();
        let thresh = (3.0 * sigma_joint).max(0.85);
        let diff = (offsets.ca.applied_ppm - offsets.cb.applied_ppm).abs();
        if diff > thresh {
            issues.push(format!(
                "aliphatic offset mismatch: {} vs {} (|Δ|={:.2} > {:.2} ppm, σ_joint={:.2})",
                offsets.ca.format_compact("dCA"),
                offsets.cb.format_compact("dCB"),
                diff,
                thresh,
                sigma_joint
            ));
        }
    }

    if offsets.ca.status == OffsetFitStatus::Fitted && offsets.c.status == OffsetFitStatus::Fitted {
        let sigma_ca_co = offsets.ca_co_joint_sigma_ppm();
        let thresh = (3.5 * sigma_ca_co).max(2.0);
        let diff = (offsets.ca.applied_ppm - offsets.c.applied_ppm).abs();
        if diff > thresh {
            issues.push(format!(
                "CA/CO offset mismatch: {} vs {} (|Δ|={:.2} > {:.2} ppm)",
                offsets.ca.format_compact("dCA"),
                offsets.c.format_compact("dC"),
                diff,
                thresh
            ));
        }
    }

    if offsets.h.status == OffsetFitStatus::Fitted && offsets.h.applied_ppm.abs() > 0.45 {
        issues.push(format!(
            "extreme 1H offset: {}",
            offsets.h.format_compact("dH")
        ));
    }
    if offsets.n.status == OffsetFitStatus::Fitted && offsets.n.applied_ppm.abs() > 2.5 {
        issues.push(format!(
            "extreme 15N offset: {}",
            offsets.n.format_compact("dN")
        ));
    }

    (issues.is_empty(), issues)
}

/// Evaluate unlabelled per-dataset quality-control metrics and compute `DatasetQcVerdict` (`Pass`,
/// `Warn`, `Abstain`) + conservative confidence calibration scale.
pub fn evaluate_dataset_qc(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    result: &AssignResult,
    opt_detailed_offsets: Option<&DetailedReferenceOffsets>,
) -> DatasetQcReport {
    let norm_seq = normalize_seq(sequence);
    let n_assignable_amides = (1..norm_seq.len())
        .filter(|&k| norm_seq[k] != "PRO")
        .count()
        .max(1);
    let n_asn_gln = norm_seq
        .iter()
        .filter(|r| r.as_str() == "ASN" || r.as_str() == "GLN")
        .count();

    let (
        n_raw_hsqc_roots,
        n_amide_hsqc_roots,
        hsqc_3d_support_frac,
        supported_roots_per_assignable,
        hsqc_3d_rms_dh_ppm,
        hsqc_3d_rms_dn_ppm,
    ) = if let Some(pks) = peaks {
        let raw_hsqc: Vec<&Peak> = pks.iter().filter(|p| p.exp == "HSQC").collect();
        let raw_3d: Vec<&Peak> = pks
            .iter()
            .filter(|p| p.exp != "HSQC" && p.c.is_some())
            .collect();
        if raw_hsqc.is_empty() {
            let n_sys = systems.len();
            let n_sup = systems.iter().filter(|s| s.has_observed_carbons()).count();
            (
                n_sys,
                n_sys,
                if n_sys > 0 {
                    (n_sup as f64) / (n_sys as f64)
                } else {
                    0.0
                },
                (n_sup as f64) / (n_assignable_amides as f64),
                0.0,
                0.0,
            )
        } else {
            let n_neg = raw_hsqc.iter().filter(|r| r.sign < 0).count();
            let maj_sign = if n_neg * 2 > raw_hsqc.len() { -1 } else { 1 };
            let has_heights = raw_hsqc.iter().any(|r| r.height.is_some());

            // Gate out opposite-sign folded Arg sidechains, out-of-window water/artifacts, and 2D sinc sidelobes,
            // WITHOUT applying any stoichiometry cap so slow-exchange minor conformers remain visible!
            let mut gated_roots: Vec<&Peak> = raw_hsqc
                .iter()
                .copied()
                .filter(|r| {
                    let sgn = if r.sign < 0 { -1 } else { 1 };
                    if sgn != maj_sign {
                        return false;
                    }
                    if has_heights {
                        (5.5..=11.5).contains(&r.h) && (96.0..=142.0).contains(&r.n)
                    } else {
                        true
                    }
                })
                .collect();

            if has_heights && !gated_roots.is_empty() {
                let mut hs: Vec<f64> = gated_roots
                    .iter()
                    .filter_map(|r| r.height.map(f64::abs))
                    .collect();
                hs.sort_by(|a, b| b.total_cmp(a));
                let n_ref = (n_assignable_amides + 2 * n_asn_gln).clamp(1, hs.len());
                let iref = median_f64(hs[..n_ref].to_vec()).unwrap_or(1.0).max(1e-9);
                let snapshot = gated_roots.clone();
                gated_roots.retain(|r| {
                    let hr = r.height.map(f64::abs).unwrap_or(0.0);
                    if hr < 0.05 * iref {
                        return false;
                    }
                    // Suppress 1D indirect/direct 2D sinc sidelobes of intense parent peaks
                    for q in &snapshot {
                        let hq = q.height.map(f64::abs).unwrap_or(0.0);
                        if hq <= hr || hq < 0.25 * iref {
                            continue;
                        }
                        let dh = (r.h - q.h).abs();
                        let dn = (r.n - q.n).abs();
                        if (dh <= 0.025 && (0.20..=3.0).contains(&dn) && hr < 0.14 * hq)
                            || (dn <= 0.22 && (0.028..=0.35).contains(&dh) && hr < 0.14 * hq)
                        {
                            return false;
                        }
                    }
                    true
                });
            }

            // Exclude paired Asn/Gln NH2 sidechain doublets that lack intra-residue CA 3D peaks
            let mut is_nh2 = vec![false; gated_roots.len()];
            for i in 0..gated_roots.len() {
                for j in (i + 1)..gated_roots.len() {
                    let dn = (gated_roots[i].n - gated_roots[j].n).abs();
                    let dh = (gated_roots[i].h - gated_roots[j].h).abs();
                    if dn <= 0.14 && (0.28..=1.25).contains(&dh) {
                        let has_ca_i = raw_3d.iter().any(|p| {
                            (p.exp == "HNCA" || (p.exp == "HNCACB" && p.sign >= 0))
                                && (p.h - gated_roots[i].h).abs() <= 0.045
                                && (p.n - gated_roots[i].n).abs() <= 0.45
                        });
                        let has_ca_j = raw_3d.iter().any(|p| {
                            (p.exp == "HNCA" || (p.exp == "HNCACB" && p.sign >= 0))
                                && (p.h - gated_roots[j].h).abs() <= 0.045
                                && (p.n - gated_roots[j].n).abs() <= 0.45
                        });
                        if !has_ca_i && !has_ca_j {
                            is_nh2[i] = true;
                            is_nh2[j] = true;
                        }
                    }
                }
            }
            let mut amide_roots: Vec<&Peak> = gated_roots
                .into_iter()
                .enumerate()
                .filter_map(|(idx, r)| (!is_nh2[idx]).then_some(r))
                .collect();
            if amide_roots.is_empty() {
                amide_roots = raw_hsqc.clone();
            }

            // Filter low-amplitude noise floor from 3D channels before computing support & (H, N) RMS
            let mut by_exp_3d: BTreeMap<&str, Vec<&Peak>> = BTreeMap::new();
            for p in &raw_3d {
                by_exp_3d.entry(p.exp.as_str()).or_default().push(*p);
            }
            let mut clean_3d_by_exp: BTreeMap<&str, Vec<&Peak>> = BTreeMap::new();
            for (ekey, list) in by_exp_3d {
                if list.iter().any(|p| p.height.is_some()) {
                    let mut hs: Vec<f64> =
                        list.iter().filter_map(|p| p.height.map(f64::abs)).collect();
                    hs.sort_by(|a, b| b.total_cmp(a));
                    let k_ref = n_assignable_amides.clamp(4, hs.len().max(1));
                    let iref = median_f64(hs[..k_ref.min(hs.len())].to_vec())
                        .unwrap_or(1.0)
                        .max(1e-9);
                    let kept: Vec<&Peak> = list
                        .into_iter()
                        .filter(|p| p.height.map(f64::abs).unwrap_or(0.0) >= 0.06 * iref)
                        .collect();
                    clean_3d_by_exp.insert(ekey, kept);
                } else {
                    clean_3d_by_exp.insert(ekey, list);
                }
            }

            // Center small uniform per-experiment lock/carrier shifts (<= 0.045 H, <= 0.45 N ppm
            // with >= 10 matches, matching solver alignment) so routine multi-day or multi-magnet
            // acquisitions are not penalized, while non-uniform scatter or large misregistration is caught.
            let mut exp_lock_shift: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
            for (&ekey, list) in &clean_3d_by_exp {
                let mut dhs = Vec::new();
                let mut dns = Vec::new();
                for r in &amide_roots {
                    let mut best: Option<(f64, f64, f64)> = None;
                    for p in list {
                        let dh = p.h - r.h;
                        let dn = p.n - r.n;
                        let d = (dh / 0.055).hypot(dn / 0.55);
                        if d <= 1.0 && best.is_none_or(|b| d < b.0) {
                            best = Some((d, dh, dn));
                        }
                    }
                    if let Some((_, dh, dn)) = best {
                        dhs.push(dh);
                        dns.push(dn);
                    }
                }
                if dhs.len() >= 10 {
                    let med_dh = median_f64(dhs).unwrap_or(0.0);
                    let med_dn = median_f64(dns).unwrap_or(0.0);
                    if med_dh.abs() <= 0.045 && med_dn.abs() <= 0.45 {
                        exp_lock_shift.insert(ekey, (med_dh, med_dn));
                    }
                }
            }

            let mut n_supported = 0usize;
            let mut sum_dh2 = 0.0f64;
            let mut sum_dn2 = 0.0f64;
            let mut n_resid = 0usize;
            for r in &amide_roots {
                let mut root_matched = false;
                for (&ekey, list) in &clean_3d_by_exp {
                    let (sh_h, sh_n) = exp_lock_shift.get(ekey).copied().unwrap_or((0.0, 0.0));
                    let mut best: Option<(f64, f64, f64)> = None;
                    for p in list {
                        let dh = (p.h - sh_h) - r.h;
                        let dn = (p.n - sh_n) - r.n;
                        let d_sup = (dh / 0.055).hypot(dn / 0.55);
                        if d_sup <= 1.0 {
                            root_matched = true;
                        }
                        let d_rms = (dh / 0.080).hypot(dn / 0.80);
                        if d_rms <= 1.0 && best.is_none_or(|b| d_rms < b.0) {
                            best = Some((d_rms, dh, dn));
                        }
                    }
                    if let Some((_, dh, dn)) = best {
                        sum_dh2 += dh * dh;
                        sum_dn2 += dn * dn;
                        n_resid += 1;
                    }
                }
                if root_matched {
                    n_supported += 1;
                }
            }

            // If 3D peaks exist yet zero pairs fell inside (0.08, 0.80) ppm, measure nearest-neighbor mismatch
            if n_resid == 0 && !raw_3d.is_empty() {
                for r in &amide_roots {
                    if let Some((dh, dn, _)) = raw_3d
                        .iter()
                        .map(|p| {
                            let dh = (p.h - r.h).clamp(-0.50, 0.50);
                            let dn = (p.n - r.n).clamp(-5.0, 5.0);
                            (dh, dn, (dh / 0.05).hypot(dn / 0.50))
                        })
                        .min_by(|a, b| a.2.total_cmp(&b.2))
                    {
                        sum_dh2 += dh * dh;
                        sum_dn2 += dn * dn;
                        n_resid += 1;
                    }
                }
            }

            let rms_dh = if n_resid > 0 {
                (sum_dh2 / (n_resid as f64)).sqrt()
            } else {
                0.0
            };
            let rms_dn = if n_resid > 0 {
                (sum_dn2 / (n_resid as f64)).sqrt()
            } else {
                0.0
            };
            let n_amide = amide_roots.len();
            let sup_frac = if n_amide > 0 {
                (n_supported as f64) / (n_amide as f64)
            } else {
                0.0
            };
            let sup_per_assign = (n_supported as f64) / (n_assignable_amides as f64);
            (
                raw_hsqc.len(),
                n_amide,
                sup_frac,
                sup_per_assign,
                rms_dh,
                rms_dn,
            )
        }
    } else {
        let n_sys = systems.len();
        let n_sup = systems.iter().filter(|s| s.has_observed_carbons()).count();
        (
            n_sys,
            n_sys,
            if n_sys > 0 {
                (n_sup as f64) / (n_sys as f64)
            } else {
                0.0
            },
            (n_sup as f64) / (n_assignable_amides as f64),
            0.0,
            0.0,
        )
    };

    let hsqc_multiplicity_ratio = (n_amide_hsqc_roots as f64) / (n_assignable_amides as f64);
    let raw_hsqc_multiplicity_ratio = (n_raw_hsqc_roots as f64) / (n_assignable_amides as f64);

    let detailed_offsets = opt_detailed_offsets
        .copied()
        .unwrap_or_else(|| estimate_reference_offsets_detailed(systems, &norm_seq, priors));
    let (offset_consistency_ok, offset_issues) = check_offset_consistency(&detailed_offsets);

    // Sequential `(i, i-1)` link coverage across assigned spin systems
    let k_len = norm_seq.len();
    let mut at = vec![-1i32; k_len];
    let mut assigned_count = 0usize;
    let mut strong_belief_count = 0usize;
    for (s, &pos) in result.position.iter().enumerate() {
        if pos >= 0 && (pos as usize) < k_len {
            at[pos as usize] = s as i32;
            assigned_count += 1;
            if result.confidence.get(s).copied().unwrap_or(0.0) >= 0.65 {
                strong_belief_count += 1;
            }
        }
    }

    let connectivity_frac = if assigned_count == 0 || systems.is_empty() {
        0.0
    } else {
        let p_mat = pair_matrix(systems, score_cfg);
        let mut linkable_assigned = 0usize;
        let mut linked_assigned = 0usize;
        for k in 1..k_len {
            let s = at[k];
            if s < 0 {
                continue;
            }
            let can_prev = k >= 2 && norm_seq[k - 1] != "PRO";
            let can_next = k + 1 < k_len && norm_seq[k + 1] != "PRO";
            if !can_prev && !can_next {
                continue;
            }
            linkable_assigned += 1;
            let prev_ok =
                can_prev && at[k - 1] >= 0 && p_mat.get(at[k - 1] as usize, s as usize) >= 1.0;
            let next_ok =
                can_next && at[k + 1] >= 0 && p_mat.get(s as usize, at[k + 1] as usize) >= 1.0;
            if prev_ok || next_ok {
                linked_assigned += 1;
            }
        }
        if linkable_assigned == 0 {
            1.0
        } else {
            (linked_assigned as f64) / (linkable_assigned as f64)
        }
    };

    let solution_margin = if assigned_count == 0 {
        0.0
    } else {
        (strong_belief_count as f64) / (assigned_count as f64)
    };

    // Determine DatasetQcVerdict (`Pass`, `Warn`, `Abstain`) & principled confidence calibration
    let mut abstain_reasons = Vec::new();
    let mut warn_reasons = Vec::new();

    if n_amide_hsqc_roots == 0 || assigned_count == 0 {
        abstain_reasons.push("zero assignable amide roots or placements".to_string());
    }
    if n_amide_hsqc_roots > 0 && hsqc_3d_support_frac < 0.25 {
        abstain_reasons.push(format!(
            "catastrophic 3D strip support ({:.1}% < 25.0%)",
            hsqc_3d_support_frac * 100.0
        ));
    }
    if assigned_count >= 6 && connectivity_frac < 0.15 {
        abstain_reasons.push(format!(
            "sequential link coverage collapse ({:.1}% < 15.0%)",
            connectivity_frac * 100.0
        ));
    }
    if assigned_count >= 12
        && (solution_margin < 0.07 || (hsqc_multiplicity_ratio > 1.60 && solution_margin < 0.16))
    {
        abstain_reasons.push(format!(
            "solver ensemble failed to converge (solution margin {:.1}%, multiplicity {:.2}x)",
            solution_margin * 100.0,
            hsqc_multiplicity_ratio
        ));
    }
    if (hsqc_3d_rms_dh_ppm > 0.065 || hsqc_3d_rms_dn_ppm > 0.65) && hsqc_3d_support_frac < 0.45 {
        abstain_reasons.push(format!(
            "severe HSQC-to-3D registration mismatch (rms dH={:.3}, dN={:.2} ppm)",
            hsqc_3d_rms_dh_ppm, hsqc_3d_rms_dn_ppm
        ));
    }

    if !abstain_reasons.is_empty() {
        return DatasetQcReport {
            verdict: DatasetQcVerdict::Abstain,
            confidence_scale: 0.0,
            confidence_cap: 0.45,
            already_applied: false,
            n_assignable_amides,
            n_raw_hsqc_roots,
            n_amide_hsqc_roots,
            hsqc_3d_support_frac,
            supported_roots_per_assignable,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            solution_margin,
            reasons: abstain_reasons,
        };
    }

    let mut scale = 1.0f64;

    if hsqc_multiplicity_ratio > 1.45 {
        let pen = ((hsqc_multiplicity_ratio - 1.45) * 0.11).clamp(0.06, 0.18);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "elevated HSQC amide multiplicity ({:.2}x > 1.45x; possible multi-conformer crowding)",
            hsqc_multiplicity_ratio
        ));
    }
    if hsqc_3d_support_frac < 0.55 {
        let pen = ((0.55 - hsqc_3d_support_frac) * 0.45).clamp(0.06, 0.18);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "low 3D strip support ({:.1}% < 55.0%)",
            hsqc_3d_support_frac * 100.0
        ));
    }
    if hsqc_3d_rms_dh_ppm > 0.032 || hsqc_3d_rms_dn_ppm > 0.32 {
        scale *= 0.88;
        warn_reasons.push(format!(
            "elevated HSQC-to-3D residual drift (rms dH={:.3}, dN={:.2} ppm)",
            hsqc_3d_rms_dh_ppm, hsqc_3d_rms_dn_ppm
        ));
    }
    if !offset_consistency_ok {
        scale *= 0.78;
        warn_reasons.extend(offset_issues);
    }
    if assigned_count >= 8 && connectivity_frac < 0.38 {
        let pen = ((0.38 - connectivity_frac) * 0.55).clamp(0.06, 0.16);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "fragmented sequential connectivity ({:.1}% < 38.0%)",
            connectivity_frac * 100.0
        ));
    }

    if warn_reasons.is_empty() {
        DatasetQcReport {
            verdict: DatasetQcVerdict::Pass,
            confidence_scale: 1.0,
            confidence_cap: 1.0,
            already_applied: false,
            n_assignable_amides,
            n_raw_hsqc_roots,
            n_amide_hsqc_roots,
            hsqc_3d_support_frac,
            supported_roots_per_assignable,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            solution_margin,
            reasons: Vec::new(),
        }
    } else {
        DatasetQcReport {
            verdict: DatasetQcVerdict::Warn,
            confidence_scale: scale.clamp(0.65, 0.92),
            confidence_cap: 1.0,
            already_applied: false,
            n_assignable_amides,
            n_raw_hsqc_roots,
            n_amide_hsqc_roots,
            hsqc_3d_support_frac,
            supported_roots_per_assignable,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            solution_margin,
            reasons: warn_reasons,
        }
    }
}

/// Apply per-dataset quality-control calibration / abstention to `result.confidence`:
/// - `DatasetQcVerdict::Abstain`: caps every confidence at `0.45` (`<= 0.45 < 0.50`) so `n_high_conf == 0`.
/// - `DatasetQcVerdict::Warn`: multiplies confidences by `qc.confidence_scale` (unless `qc.already_applied` is set).
/// - `DatasetQcVerdict::Pass`: leaves confidences untouched (`scale = 1.0`).
pub fn apply_dataset_qc_to_assignments(result: &mut AssignResult, qc: &DatasetQcReport) {
    match qc.verdict {
        DatasetQcVerdict::Pass => {}
        DatasetQcVerdict::Abstain => {
            let cap = qc.confidence_cap.min(0.45);
            for c in &mut result.confidence {
                *c = c.clamp(0.0, cap);
            }
        }
        DatasetQcVerdict::Warn => {
            if !qc.already_applied {
                let s = qc.confidence_scale.clamp(0.0, 1.0);
                let cap = qc.confidence_cap.clamp(0.0, 1.0);
                for c in &mut result.confidence {
                    *c = (*c * s).clamp(0.0, cap);
                }
            }
        }
    }
}

/// Single confidence bin in a per-protein reliability diagram (`[lower, upper)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub correct: usize,
    pub mean_conf: f64,
    pub empirical_acc: f64,
}

/// Per-protein confidence reliability diagram (`[0.0..0.5, 0.5..0.8, 0.8..0.9, 0.9..1.0]`),
/// Expected Calibration Error (`ece`), and explicit high-confidence abstention status.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProteinReliabilityCurve {
    pub bins: [ReliabilityBin; 4],
    pub total_predictions: usize,
    pub ece: f64,
    pub hc_threshold: f64,
    pub hc_assigned: usize,
    pub hc_correct: usize,
    /// `None` when `hc_assigned == 0` (abstained), `Some(hc_correct / hc_assigned)` otherwise.
    pub hc_precision: Option<f64>,
    /// Distinguishes `"Abstained (0 HC calls)"` from `"0.0% (0/N)"` when wrong HC calls occur.
    pub hc_status: String,
}

impl ProteinReliabilityCurve {
    pub const BIN_EDGES: [(f64, f64); 4] = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.0)];

    /// Construct a 4-bin `ProteinReliabilityCurve` from `(confidence, is_correct)` pairs.
    pub fn from_predictions(predictions: &[(f64, bool)], hc_threshold: f64) -> Self {
        let mut counts = [0usize; 4];
        let mut corrects = [0usize; 4];
        let mut conf_sums = [0.0f64; 4];
        let mut hc_assigned = 0usize;
        let mut hc_correct = 0usize;

        for &(raw_conf, is_ok) in predictions {
            let c = raw_conf.clamp(0.0, 1.0);
            let b = if c < 0.5 {
                0
            } else if c < 0.8 {
                1
            } else if c < 0.9 {
                2
            } else {
                3
            };
            counts[b] += 1;
            conf_sums[b] += c;
            if is_ok {
                corrects[b] += 1;
            }
            if c >= hc_threshold {
                hc_assigned += 1;
                if is_ok {
                    hc_correct += 1;
                }
            }
        }

        let total = predictions.len();
        let mut ece = 0.0f64;
        let bins = std::array::from_fn(|b| {
            let (lower, upper) = Self::BIN_EDGES[b];
            let cnt = counts[b];
            let cor = corrects[b];
            let (mean_conf, empirical_acc) = if cnt > 0 {
                let mc = conf_sums[b] / (cnt as f64);
                let ea = (cor as f64) / (cnt as f64);
                if total > 0 {
                    ece += ((cnt as f64) / (total as f64)) * (mc - ea).abs();
                }
                (mc, ea)
            } else {
                (0.0, 0.0)
            };
            ReliabilityBin {
                lower,
                upper,
                count: cnt,
                correct: cor,
                mean_conf,
                empirical_acc,
            }
        });

        let (hc_precision, hc_status) = if hc_assigned == 0 {
            (None, "Abstained (0 HC calls)".to_string())
        } else {
            let p = (hc_correct as f64) / (hc_assigned as f64);
            (
                Some(p),
                format!("{:.1}% ({}/{})", p * 100.0, hc_correct, hc_assigned),
            )
        };

        Self {
            bins,
            total_predictions: total,
            ece,
            hc_threshold,
            hc_assigned,
            hc_correct,
            hc_precision,
            hc_status,
        }
    }

    /// Construct a `ProteinReliabilityCurve` directly from an `AssignResult` and spin-system `truth` vector.
    pub fn from_assign_result(result: &AssignResult, truth: &[i32], hc_threshold: f64) -> Self {
        let mut preds = Vec::new();
        for (s, &pos) in result.position.iter().enumerate() {
            if pos >= 0 {
                let conf = result.confidence.get(s).copied().unwrap_or(0.0);
                let is_ok = truth.get(s).copied().unwrap_or(-2) == pos;
                preds.push((conf, is_ok));
            }
        }
        Self::from_predictions(&preds, hc_threshold)
    }

    /// Whether the dataset abstained from making any high-confidence (`>= hc_threshold`) call.
    #[inline]
    pub fn is_abstained(&self) -> bool {
        self.hc_assigned == 0
    }
}

/// Evaluate a monotone piecewise-linear calibration map `f: [0, 1] -> [0, 1]` over sorted
/// `(x_i, y_i)` knots (`0 = x_0 < x_1 < ... < x_m = 1`, `0 <= y_0 <= ... <= y_m <= 1`).
pub fn eval_monotone_calibration_map(p_raw: f64, knots: &[(f64, f64)]) -> f64 {
    let x = p_raw.clamp(0.0, 1.0);
    if knots.is_empty() {
        return x;
    }
    if x <= knots[0].0 {
        return knots[0].1.clamp(0.0, 1.0);
    }
    let last = knots[knots.len() - 1];
    if x >= last.0 {
        return last.1.clamp(0.0, 1.0);
    }
    for win in knots.windows(2) {
        let (x0, y0) = win[0];
        let (x1, y1) = win[1];
        if x <= x1 {
            let span = (x1 - x0).max(1e-12);
            let t = ((x - x0) / span).clamp(0.0, 1.0);
            return (y0 + t * (y1 - y0)).clamp(0.0, 1.0);
        }
    }
    last.1.clamp(0.0, 1.0)
}

/// Fit a monotone piecewise-linear calibration knot table on `grid` (`0.0 = g_0 < ... < g_m = 1.0`)
/// using Laplace-regularized Pool-Adjacent-Violators (PAV) isotonic regression (Ayer et al. 1955;
/// Zadrozny & Elkan 2002).
///
/// Each interval `[g_b, g_{b+1})` accumulates empirical `(correct, count)` plus a symmetric
/// neutral Beta(`laplace_prior / 2`, `laplace_prior / 2`) pseudocount (`0.5 * laplace_prior`) so
/// sparse bins shrink toward `0.5` rather than collapsing to `0.0`/`1.0` or inheriting raw
/// `T = 1.0` overconfidence.
pub fn fit_isotonic_monotone_knots(
    samples: &[(f64, bool)],
    grid: &[f64],
    laplace_prior: f64,
) -> Vec<(f64, f64)> {
    if grid.len() < 2 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    let n_bins = grid.len() - 1;
    let mut weights = vec![0.0_f64; n_bins];
    let mut pos_sums = vec![0.0_f64; n_bins];
    let mut x_mids = vec![0.0_f64; n_bins];
    for b in 0..n_bins {
        let mid = 0.5 * (grid[b] + grid[b + 1]);
        x_mids[b] = mid;
        weights[b] = laplace_prior.max(1e-6);
        pos_sums[b] = 0.5 * weights[b];
    }

    for &(raw_p, is_ok) in samples {
        let p = raw_p.clamp(0.0, 1.0);
        let mut idx = n_bins - 1;
        for b in 0..n_bins {
            if p < grid[b + 1] || b + 1 == n_bins {
                idx = b;
                break;
            }
        }
        weights[idx] += 1.0;
        if is_ok {
            pos_sums[idx] += 1.0;
        }
    }

    // Pool-Adjacent-Violators (PAV) stack of `(start_bin, end_bin, sum_pos, sum_weight)`
    let mut blocks: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        blocks.push((b, b, pos_sums[b], weights[b]));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let mean_prev = blocks[n - 2].2 / blocks[n - 2].3;
            let mean_curr = blocks[n - 1].2 / blocks[n - 1].3;
            if mean_prev > mean_curr {
                let last = blocks.pop().unwrap();
                let prev = blocks.last_mut().unwrap();
                prev.1 = last.1;
                prev.2 += last.2;
                prev.3 += last.3;
            } else {
                break;
            }
        }
    }

    let mut bin_vals = vec![0.0_f64; n_bins];
    for (lo, hi, sp, sw) in blocks {
        let v = (sp / sw).clamp(0.0, 1.0);
        for slot in &mut bin_vals[lo..=hi] {
            *slot = v;
        }
    }

    let mut knots = Vec::with_capacity(n_bins + 2);
    knots.push((0.0, 0.0));
    for b in 0..n_bins {
        knots.push((x_mids[b], bin_vals[b]));
    }
    let y_end = bin_vals[n_bins - 1].max(0.0);
    knots.push((1.0, y_end));
    knots
}

/// Monotone calibration knot table for `evo-graph` (`P1-2`), fitted via isotonic PAV on the
/// canonical 12-bin grid (`g12`) with Jeffreys `laplace = 0.5` smoothing across the 10-entry
/// `dev` split (zero hyperparameter search; validated via strict 10-fold Leave-One-Protein-Out
/// cross-validation over physical `T = 1.0` marginal posteriors).
pub const EVO_GRAPH_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0750, 0.1917),
    (0.2250, 0.2909),
    (0.3750, 0.4567),
    (0.5150, 0.6230),
    (0.6300, 0.6230),
    (0.7200, 0.6603),
    (0.7950, 0.6976),
    (0.8600, 0.6976),
    (0.9150, 0.8224),
    (0.9575, 0.8224),
    (0.9835, 0.8717),
    (0.9960, 0.9757),
    (1.0000, 0.9757),
];

/// Monotone calibration knot table for `standard` (`P1-6`), fitted via isotonic PAV on the
/// identical canonical 12-bin grid (`g12`) with Jeffreys `laplace = 0.5` smoothing across the
/// 10-entry `dev` split (zero hyperparameter search; validated via strict 10-fold
/// Leave-One-Protein-Out cross-validation over physical `T = 1.0` marginal posteriors).
pub const STANDARD_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0750, 0.2411),
    (0.2250, 0.5080),
    (0.3750, 0.5080),
    (0.5150, 0.5745),
    (0.6300, 0.5925),
    (0.7200, 0.7721),
    (0.7950, 0.7721),
    (0.8600, 0.8966),
    (0.9150, 0.9348),
    (0.9575, 0.9444),
    (0.9835, 0.9462),
    (0.9960, 0.9462),
    (1.0000, 0.9462),
];

/// Map raw physical (`T = 1.0`) `evo-graph` marginal belief into calibrated probability.
#[inline]
pub fn apply_evo_graph_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, EVO_GRAPH_CALIBRATION_KNOTS)
}

/// Map raw physical (`T = 1.0`) `standard` marginal belief into calibrated probability.
#[inline]
pub fn apply_standard_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, STANDARD_CALIBRATION_KNOTS)
}
