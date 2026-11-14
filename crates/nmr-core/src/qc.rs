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
//! 7. Ferro, C. A. T. & Fricker, T. E. (2012). "A bias-corrected decomposition of the Brier score."
//!    *Q. J. R. Meteorol. Soc.* 138(668), 1954–1960; Kumar, A., Liang, P. & Ma, T. (2019).
//!    "Verified Uncertainty Calibration." *Adv. Neural Inf. Process. Syst. (NeurIPS)* 32 —
//!    finite-sample plug-in calibration bias `O(1 / sqrt(n_b))` and variance-debiased ECE.

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

const fn default_one_f64() -> f64 {
    1.0
}

/// Per-dataset quality-control gate verdict computed strictly from unlabelled observables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetQcVerdict {
    /// Clean stoichiometry, 3D strip support, `(H, N)` registration, referencing, and connectivity.
    #[default]
    Pass,
    /// Multi-conformer crowding, partial 3D/aliphatic support, inter-spectrum drift, offset
    /// inconsistency, or triggered protein-level HC cap gate; per-residue confidences are damped
    /// by `confidence_scale`, bounded by `confidence_cap`, and sub-HC calls (`< 0.80`) carry
    /// `warn_sub_hc_uncalibrated = true` (`Q4`).
    Warn,
    /// Catastrophic 3D/aliphatic mismatch (`< 25%` supported roots or `< 20%` aliphatic systems)
    /// or sequential/ensemble collapse; all confidences are capped at `<= 0.45` so `n_high_conf == 0`.
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
    /// Upper confidence cap (`1.0` on `Pass` / unconstrained `Warn`, `0.79` when the protein-level
    /// HC gate triggers under `P1-7`, `0.45` on `Abstain`).
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
    /// Fraction of carbon-bearing spin systems backed by `>= 1` aliphatic (`13CA` or `13CB`) carbon observation (`P1-7`).
    #[serde(default = "default_one_f64")]
    pub aliphatic_support_frac: f64,
    /// Empirical Voronoi-nearest `1H` match residual RMS (ppm) between 2D HSQC roots and 3D peaks.
    pub hsqc_3d_rms_dh_ppm: f64,
    /// Empirical Voronoi-nearest `15N` match residual RMS (ppm) between 2D HSQC roots and 3D peaks.
    pub hsqc_3d_rms_dn_ppm: f64,
    /// Candidate amide HSQC root multiplicity (`n_amide_hsqc_roots / n_assignable_amides`; flags `> 1.45` when 3D-backed).
    pub hsqc_multiplicity_ratio: f64,
    /// Raw picked HSQC root multiplicity (`n_raw_hsqc_roots / n_assignable_amides`).
    pub raw_hsqc_multiplicity_ratio: f64,
    /// Whether fitted global chemical-shift offsets across `(H, N, CA, CB, C)` are physically self-consistent.
    pub offset_consistency_ok: bool,
    /// Per-nucleus reference offset diagnostics (`Fitted`, `NotSignificant`, `TooFewValues`).
    pub detailed_offsets: DetailedReferenceOffsets,
    /// Fraction of assigned spin systems supported by at least one sequential `(i, i-1)` link (`P >= 1.0`).
    pub connectivity_frac: f64,
    /// Fraction of linkable assigned spin systems supported by a corroborated multi-channel sequential
    /// `(i, i-1)` link (`P >= 3.5` nats; `P1-7`).
    #[serde(default = "default_one_f64")]
    pub corroborated_connectivity_frac: f64,
    /// Fraction of assigned spin systems with strong consensus/marginal belief (`>= 0.65`).
    pub solution_margin: f64,
    /// Pre-cap high-confidence (`>= 0.80`) call fraction among placed spin systems (`n_hc / n_assigned`; `P1-7`).
    #[serde(default)]
    pub hc_call_fraction: f64,
    /// True when the label-free protein-level HC gate (`P1-7`) capped post-QC confidence at `<= 0.79`.
    #[serde(default)]
    pub protein_hc_gate_triggered: bool,
    /// True under `DatasetQcVerdict::Warn` (`Q4`): explicitly marks sub-HC calls (`confidence < 0.80`)
    /// as uncalibrated under `Warn` damping rather than fitting a separate small-sample `Warn` isotonic curve.
    #[serde(default)]
    pub warn_sub_hc_uncalibrated: bool,
    /// True when the solver replica/population count matched the canonical isotonic calibration fit
    /// ([`STANDARD_CPU_CALIBRATED_RUNS`] `= 16` for CPU `standard`,
    /// [`STANDARD_GPU_CALIBRATED_RUNS`] `= 64` for WebGPU `standard`,
    /// [`EVO_GRAPH_CALIBRATED_RUNS`] `= 64` for `evo-graph`); `false` when a caller overrides
    /// `runs != fit_runs` (`P1-8`).
    pub confidence_calibrated: bool,
    /// Explicit warning when `confidence_calibrated == false` (`runs != fit_runs`, `P1-8`).
    pub calibration_warning: Option<String>,
    /// Human-readable QC warnings or abstention triggers.
    pub reasons: Vec<String>,
}

/// Canonical CPU `standard` replica count (`runs = 16`) at which [`STANDARD_CALIBRATION_KNOTS`] is fitted (`P1-8`).
pub const STANDARD_CPU_CALIBRATED_RUNS: usize = 16;

/// Canonical WebGPU `standard` replica count (`runs = 64`) at which [`STANDARD_GPU_CALIBRATION_KNOTS`] is fitted (`P1-8`).
pub const STANDARD_GPU_CALIBRATED_RUNS: usize = 64;

/// Canonical `evo-graph` island population size (`population_size = 64`, `num_islands = 16`) at which
/// [`EVO_GRAPH_CALIBRATION_KNOTS`] is fitted (`P1-8`).
pub const EVO_GRAPH_CALIBRATED_RUNS: usize = 64;

impl DatasetQcReport {
    /// Record an explicit `P1-8` replica-count mismatch (`actual_runs != fit_runs`), marking
    /// `confidence_calibrated = false` and populating `calibration_warning`.
    pub fn mark_uncalibrated_runs(
        &mut self,
        solver_label: &str,
        actual_runs: usize,
        fit_runs: usize,
    ) {
        if actual_runs != fit_runs {
            self.confidence_calibrated = false;
            self.calibration_warning = Some(format!(
                "uncalibrated replica count for {solver_label} (runs={actual_runs} != fit_runs={fit_runs}); isotonic confidence knots were fitted at runs={fit_runs} (confidence_calibrated=false)"
            ));
        }
    }

    /// Whether an individual post-QC residue confidence is certified as calibrated (`Q4`, `P1-8`):
    /// returns `false` when `confidence_calibrated == false`, when `verdict == Abstain`, or when
    /// `warn_sub_hc_uncalibrated` is set (`verdict == Warn`) and `post_qc_confidence < 0.80`.
    #[inline]
    pub fn is_call_calibrated(&self, post_qc_confidence: f64) -> bool {
        self.confidence_calibrated
            && self.verdict != DatasetQcVerdict::Abstain
            && !(self.warn_sub_hc_uncalibrated && post_qc_confidence < 0.80)
    }
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
            aliphatic_support_frac: 1.0,
            hsqc_3d_rms_dh_ppm: 0.0,
            hsqc_3d_rms_dn_ppm: 0.0,
            hsqc_multiplicity_ratio: 1.0,
            raw_hsqc_multiplicity_ratio: 1.0,
            offset_consistency_ok: true,
            detailed_offsets: DetailedReferenceOffsets::default(),
            connectivity_frac: 1.0,
            corroborated_connectivity_frac: 1.0,
            solution_margin: 1.0,
            hc_call_fraction: 0.0,
            protein_hc_gate_triggered: false,
            warn_sub_hc_uncalibrated: false,
            confidence_calibrated: true,
            calibration_warning: None,
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

/// Whether `offsets` carries a large carbon/nitrogen registration shift (`|dCA| > 3.20` or
/// `|dCB| > 3.20` or `|dN| > 2.55` ppm) backed by weak per-observation log-likelihood gain
/// (`log_likelihood_gain / n_values < min_gain_per_obs`), indicating likely spectral aliasing or
/// wrong-nucleus folding rather than uniform TMS/DSS chemical-shift referencing (`P1-7`).
pub fn has_weak_likelihood_large_offset(
    offsets: &DetailedReferenceOffsets,
    min_gain_per_obs: f64,
) -> Option<String> {
    for (label, fit, ppm_thresh) in [
        ("dN", &offsets.n, 2.55_f64),
        ("dCA", &offsets.ca, 3.20_f64),
        ("dCB", &offsets.cb, 3.20_f64),
    ] {
        if fit.status == OffsetFitStatus::Fitted && fit.applied_ppm.abs() > ppm_thresh {
            let gain_per_obs = fit.log_likelihood_gain / (fit.n_values.max(1) as f64);
            if gain_per_obs < min_gain_per_obs {
                return Some(format!(
                    "weakly supported large {label} registration shift ({}, ΔLL/n={gain_per_obs:.3} < {min_gain_per_obs:.3})",
                    fit.format_compact(label)
                ));
            }
        }
    }
    None
}

/// Fixed per-spin belief threshold (`>= 0.65`) defining the fraction of strongly supported
/// placements (`solution_margin`) in [`evaluate_dataset_qc`].
pub const QC_STRONG_BELIEF_THRESHOLD: f64 = 0.65;

/// Minimum number of assigned spin systems (`>= 12`) required before applying the ensemble
/// convergence (`solution_margin`) abstention gate in [`evaluate_dataset_qc`].
pub const QC_MIN_ASSIGNED_FOR_MARGIN_GATE: usize = 12;

/// Minimum `solution_margin` (`0.07`) required for ensemble convergence on normal-multiplicity
/// spectra (`hsqc_multiplicity_ratio <= QC_CROWDED_MULTIPLICITY_RATIO`).
pub const QC_MIN_SOLUTION_MARGIN: f64 = 0.07;

/// Amide `HSQC` multiplicity ratio (`1.60`) above which a spectrum is treated as severely
/// multi-conformer crowded in [`is_unconverged_solution_margin`].
pub const QC_CROWDED_MULTIPLICITY_RATIO: f64 = 1.60;

/// Minimum `solution_margin` (`0.16`) required for ensemble convergence on crowded spectra
/// (`hsqc_multiplicity_ratio > QC_CROWDED_MULTIPLICITY_RATIO`).
pub const QC_CROWDED_MIN_SOLUTION_MARGIN: f64 = 0.16;

/// Pairwise sequential match score threshold (`P >= 3.5` nats) defining a corroborated
/// multi-channel `(i, i-1)` link (`corroborated_connectivity_frac`; `P1-7`). Single-channel
/// accidental `13C'` or `13CA` overlaps routinely reach `P >= 1.0` nat (`85–90%` even on random
/// `9–21%`-accurate placements), whereas genuine multi-pathway sequential matches exceed `3.5` nats.
pub const QC_CORROBORATED_PAIR_THRESHOLD: f64 = 3.5;

/// Post-QC confidence ceiling (`0.79 < 0.80`) applied when the protein-level HC gate triggers (`P1-7`),
/// suppressing unreliable `>= 0.80` high-confidence commitments without collapsing intermediate
/// `[0.45, 0.79]` rankings.
pub const QC_PROTEIN_HC_CONFIDENCE_CAP: f64 = 0.79;

/// Low-DOF label-free configuration for the protein-level HC cap gate (`P1-7`), fitted strictly
/// inside 13-fold `dev` Leave-One-Protein-Out (LOPO) cross-validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProteinHcGateConfig {
    /// Minimum corroborated sequential link density (`P >= 3.5` nats) required to emit `>= 0.80` HC calls.
    pub min_corroborated_connectivity: f64,
    /// Minimum fraction of placed spin systems reaching `>= 0.80` (`n_hc / n_assigned`) required
    /// so isolated tail spikes (`1–5%` of residues) cannot emit false HC commitments.
    pub min_hc_call_fraction: f64,
    /// Minimum `solution_margin` required on severely crowded multi-conformer datasets (`supported_roots_per_assignable > 1.60`).
    pub min_crowded_solution_margin: f64,
    /// Minimum floor on the multiplicative crowding damping penalty when `hsqc_multiplicity_ratio > 1.45`
    /// and `supported_roots_per_assignable > 1.35`.
    pub multiplicity_penalty_floor: f64,
    /// Minimum `log_likelihood_gain / n_values` required when a large registration offset (`|dCA|, |dCB| > 3.2` ppm) is fitted.
    pub min_large_offset_gain_per_obs: f64,
}

/// Canonical protein-level HC gate parameters fitted on the 13-protein `dev` split (`P1-7`).
pub const CANONICAL_PROTEIN_HC_GATE: ProteinHcGateConfig = ProteinHcGateConfig {
    min_corroborated_connectivity: 0.66,
    min_hc_call_fraction: 0.08,
    min_crowded_solution_margin: 0.45,
    multiplicity_penalty_floor: 0.10,
    min_large_offset_gain_per_obs: 0.15,
};

impl Default for ProteinHcGateConfig {
    fn default() -> Self {
        CANONICAL_PROTEIN_HC_GATE
    }
}

/// Per-seed label-free protein telemetry + raw `(p_raw, is_correct)` observations used to refit
/// [`ProteinHcGateConfig`] inside each 13-fold `dev` LOPO training split with zero target leakage.
#[derive(Clone, Debug)]
pub struct ProteinHcTrainingSeed {
    pub n_assignable_amides: usize,
    pub hsqc_3d_support_frac: f64,
    pub supported_roots_per_assignable: f64,
    pub aliphatic_support_frac: f64,
    pub hsqc_multiplicity_ratio: f64,
    pub hsqc_3d_rms_dh_ppm: f64,
    pub hsqc_3d_rms_dn_ppm: f64,
    pub offset_consistency_ok: bool,
    pub weak_large_offset: bool,
    pub connectivity_frac: f64,
    pub corroborated_connectivity_frac: f64,
    pub raw_predictions: Vec<(f64, bool)>,
}

/// Evaluate a single seed's post-QC `(confidence, is_correct)` predictions from raw `(p_raw, is_ok)`
/// pairs, isotonic `knots`, and `gate_cfg`, mirroring [`evaluate_dataset_qc_with_gate`] +
/// [`apply_dataset_qc_to_assignments`] identically for zero-leakage LOPO cross-validation.
pub fn evaluate_seed_predictions_with_gate(
    seed: &ProteinHcTrainingSeed,
    knots: &[(f64, f64)],
    gate_cfg: &ProteinHcGateConfig,
) -> Vec<(f64, bool)> {
    let n_ass = seed.raw_predictions.len();
    if n_ass == 0 {
        return Vec::new();
    }
    let cals: Vec<f64> = seed
        .raw_predictions
        .iter()
        .map(|&(p, _)| eval_monotone_calibration_map(p, knots))
        .collect();
    let solution_margin = (cals
        .iter()
        .filter(|&&c| c >= QC_STRONG_BELIEF_THRESHOLD)
        .count() as f64)
        / (n_ass as f64);

    let is_abstain = seed.hsqc_3d_support_frac < 0.25
        || seed.aliphatic_support_frac < 0.20
        || (n_ass >= 6 && seed.connectivity_frac < 0.15)
        || is_unconverged_solution_margin(n_ass, solution_margin, seed.hsqc_multiplicity_ratio)
        || ((seed.hsqc_3d_rms_dh_ppm > 0.065 || seed.hsqc_3d_rms_dn_ppm > 0.65)
            && seed.hsqc_3d_support_frac < 0.45)
        || (seed.hsqc_3d_support_frac < 0.45
            && seed.supported_roots_per_assignable < 0.35
            && solution_margin < 0.20);
    if is_abstain {
        return cals
            .into_iter()
            .zip(seed.raw_predictions.iter())
            .map(|(c, &(_, ok))| (c.clamp(0.0, 0.45), ok))
            .collect();
    }

    let mut scale = 1.0_f64;
    if seed.hsqc_multiplicity_ratio > 1.45 && seed.supported_roots_per_assignable > 1.35 {
        let pen = ((seed.hsqc_multiplicity_ratio - 1.45) * 0.20)
            .clamp(gate_cfg.multiplicity_penalty_floor, 0.22);
        scale *= 1.0 - pen;
    }
    if seed.hsqc_3d_support_frac < 0.55 && seed.supported_roots_per_assignable < 0.80 {
        let pen = ((0.55 - seed.hsqc_3d_support_frac) * 0.45).clamp(0.06, 0.18);
        scale *= 1.0 - pen;
    }
    if seed.aliphatic_support_frac < 0.50 {
        scale *= 0.90;
    }
    if seed.hsqc_3d_rms_dh_ppm > 0.028 || seed.hsqc_3d_rms_dn_ppm > 0.26 {
        scale *= 0.88;
    }
    if !seed.offset_consistency_ok || seed.weak_large_offset {
        scale *= 0.78;
    }
    if n_ass >= 8 && seed.connectivity_frac < 0.38 {
        let pen = ((0.38 - seed.connectivity_frac) * 0.55).clamp(0.06, 0.16);
        scale *= 1.0 - pen;
    }
    if n_ass >= 8 && seed.corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity {
        scale *= 0.90;
    }
    if scale < 1.0 {
        scale = scale.clamp(0.65, 0.92);
    }

    let scaled: Vec<f64> = cals
        .into_iter()
        .map(|c| (c * scale).clamp(0.0, 1.0))
        .collect();
    let hc_cnt = scaled.iter().filter(|&&c| c >= 0.80).count();
    let hc_frac = (hc_cnt as f64) / (n_ass as f64);

    let trigger_hc_cap = (n_ass >= 12
        && ((hc_cnt > 0 && hc_frac < gate_cfg.min_hc_call_fraction)
            || seed.corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity
            || (seed.supported_roots_per_assignable > 1.60
                && solution_margin < gate_cfg.min_crowded_solution_margin)
            || ((!seed.offset_consistency_ok || seed.weak_large_offset)
                && solution_margin < 0.50)))
        || (hc_cnt > 0 && n_ass * 5 < seed.n_assignable_amides);
    let cap = if trigger_hc_cap {
        QC_PROTEIN_HC_CONFIDENCE_CAP
    } else {
        1.0
    };
    scaled
        .into_iter()
        .zip(seed.raw_predictions.iter())
        .map(|(c, &(_, ok))| (c.clamp(0.0, cap), ok))
        .collect()
}

/// Fit [`ProteinHcGateConfig`] strictly on training proteins (`train_proteins`, each a slice of
/// [`ProteinHcTrainingSeed`] seeds) by minimizing pooled 10-bin `ECE_10` subject to `>= 94.5%`
/// training micro HC-precision over a compact 27-point low-DOF grid (`P1-7`).
pub fn fit_protein_hc_gate_config(
    train_proteins: &[Vec<ProteinHcTrainingSeed>],
    knots: &[(f64, f64)],
) -> ProteinHcGateConfig {
    if train_proteins.is_empty() {
        return CANONICAL_PROTEIN_HC_GATE;
    }
    let mut best_cfg = CANONICAL_PROTEIN_HC_GATE;
    let mut best_obj = f64::INFINITY;
    for &min_c35 in &[0.65_f64, 0.66, 0.68] {
        for &min_hcf in &[0.06_f64, 0.08, 0.10] {
            for &mpen in &[0.08_f64, 0.10, 0.12] {
                let cand = ProteinHcGateConfig {
                    min_corroborated_connectivity: min_c35,
                    min_hc_call_fraction: min_hcf,
                    min_crowded_solution_margin: 0.45,
                    multiplicity_penalty_floor: mpen,
                    min_large_offset_gain_per_obs: 0.15,
                };
                let mut pool = Vec::new();
                for seeds in train_proteins {
                    for s in seeds {
                        pool.extend(evaluate_seed_predictions_with_gate(s, knots, &cand));
                    }
                }
                if pool.is_empty() {
                    continue;
                }
                let ece10 = compute_grid_ece(&pool, UNIFORM_CALIBRATION_GRID_10);
                let hc_tot = pool.iter().filter(|&&(c, _)| c >= 0.80).count();
                let hc_ok = pool.iter().filter(|&&(c, ok)| c >= 0.80 && ok).count();
                let hcp = if hc_tot > 0 {
                    (hc_ok as f64) / (hc_tot as f64)
                } else {
                    1.0
                };
                let hcp_pen = if hcp >= 0.945 {
                    0.0
                } else {
                    (0.945 - hcp) * 5.0
                };
                // Mild tie-breaking preference toward canonical (0.66, 0.08, 0.10) center
                let reg =
                    1e-4 * ((min_c35 - 0.66).abs() + (min_hcf - 0.08).abs() + (mpen - 0.10).abs());
                let obj = ece10 + hcp_pen + reg;
                if obj < best_obj {
                    best_obj = obj;
                    best_cfg = cand;
                }
            }
        }
    }
    best_cfg
}

/// Fixed ensemble-convergence specification predicate (`(0.65, 0.07, 1.60, 0.16)`) shared by
/// [`evaluate_dataset_qc`] and Leave-One-Protein-Out (LOPO) calibration training-pool filtering:
/// returns `true` when `assigned_count >= 12` and either `solution_margin < 0.07` or
/// `(hsqc_multiplicity_ratio > 1.60 && solution_margin < 0.16)`.
#[inline]
pub fn is_unconverged_solution_margin(
    assigned_count: usize,
    solution_margin: f64,
    hsqc_multiplicity_ratio: f64,
) -> bool {
    assigned_count >= QC_MIN_ASSIGNED_FOR_MARGIN_GATE
        && (solution_margin < QC_MIN_SOLUTION_MARGIN
            || (hsqc_multiplicity_ratio > QC_CROWDED_MULTIPLICITY_RATIO
                && solution_margin < QC_CROWDED_MIN_SOLUTION_MARGIN))
}

/// Evaluate unlabelled per-dataset quality-control metrics using the canonical protein-level HC
/// gate ([`CANONICAL_PROTEIN_HC_GATE`]).
pub fn evaluate_dataset_qc(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    result: &AssignResult,
    opt_detailed_offsets: Option<&DetailedReferenceOffsets>,
) -> DatasetQcReport {
    evaluate_dataset_qc_with_gate(
        peaks,
        systems,
        sequence,
        priors,
        score_cfg,
        result,
        opt_detailed_offsets,
        &CANONICAL_PROTEIN_HC_GATE,
    )
}

/// Evaluate unlabelled per-dataset quality-control metrics and compute `DatasetQcVerdict` (`Pass`,
/// `Warn`, `Abstain`), conservative confidence damping, `Q4` sub-HC uncalibrated flag, and
/// `P1-7` protein-level HC cap under `gate_cfg`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_dataset_qc_with_gate(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    result: &AssignResult,
    opt_detailed_offsets: Option<&DetailedReferenceOffsets>,
    gate_cfg: &ProteinHcGateConfig,
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

            // Filter low-amplitude noise floor from 3D channels before computing carrier lock shifts,
            // Asn/Gln NH2 exclusion, support, and Voronoi (H, N) RMS (`P1-7`).
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
            // with >= 10 matches, matching solver alignment) BEFORE NH2 doublet exclusion and
            // residual computation (`P1-7`).
            let mut exp_lock_shift: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
            for (&ekey, list) in &clean_3d_by_exp {
                let mut dhs = Vec::new();
                let mut dns = Vec::new();
                for r in &gated_roots {
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

            // Exclude paired Asn/Gln NH2 sidechain doublets lacking backbone CA 3D peaks
            // (including HNCOCA CA(i-1) in `c >= 39 ppm` for HNCA-absent experiment sets; `P1-7`).
            let has_backbone_ca = |root: &Peak| -> bool {
                raw_3d.iter().any(|p| {
                    let is_ca_chan = p.exp == "HNCA"
                        || (p.exp == "HNCACB" && p.sign >= 0)
                        || (p.exp == "HNCOCA" && p.c.is_some_and(|c| c >= 39.0));
                    if !is_ca_chan {
                        return false;
                    }
                    let (sh_h, sh_n) = exp_lock_shift
                        .get(p.exp.as_str())
                        .copied()
                        .unwrap_or((0.0, 0.0));
                    ((p.h - sh_h) - root.h).abs() <= 0.045 && ((p.n - sh_n) - root.n).abs() <= 0.45
                })
            };
            let mut is_nh2 = vec![false; gated_roots.len()];
            for i in 0..gated_roots.len() {
                for j in (i + 1)..gated_roots.len() {
                    let dn = (gated_roots[i].n - gated_roots[j].n).abs();
                    let dh = (gated_roots[i].h - gated_roots[j].h).abs();
                    if dn <= 0.16
                        && (0.28..=1.25).contains(&dh)
                        && !has_backbone_ca(gated_roots[i])
                        && !has_backbone_ca(gated_roots[j])
                    {
                        is_nh2[i] = true;
                        is_nh2[j] = true;
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

            // Re-center per-experiment lock shift on the cleaned `amide_roots`
            exp_lock_shift.clear();
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

            // Compute 3D support AND Voronoi-nearest `(dH, dN)` residuals (`P1-7`):
            // In lower-sensitivity experiments such as `HNCACO` (in `HNCO`-absent complementary
            // sets `HNCACB+CBCACONH+HNCOCA+HNCACO`), a root missing its own `HNCACO` peak must NOT
            // inherit a neighboring root's `HNCACO` peak `0.45–0.79 ppm` away as artificial drift.
            let mut root_supported = vec![false; amide_roots.len()];
            let mut sum_dh2 = 0.0_f64;
            let mut sum_dn2 = 0.0_f64;
            let mut n_resid = 0usize;

            for (&ekey, list) in &clean_3d_by_exp {
                let (sh_h, sh_n) = exp_lock_shift.get(ekey).copied().unwrap_or((0.0, 0.0));
                // Map each 3D peak to its Voronoi-nearest amide root
                let mut best_for_root: Vec<Option<(f64, f64, f64)>> = vec![None; amide_roots.len()];
                for p in list {
                    let ph = p.h - sh_h;
                    let pn = p.n - sh_n;
                    let mut owner: Option<(usize, f64, f64, f64)> = None;
                    for (r_idx, r) in amide_roots.iter().enumerate() {
                        let dh = ph - r.h;
                        let dn = pn - r.n;
                        if (dh / 0.055).hypot(dn / 0.55) <= 1.0 {
                            root_supported[r_idx] = true;
                        }
                        let d_vor = (dh / 0.080).hypot(dn / 0.80);
                        if d_vor <= 1.0 && owner.is_none_or(|o| d_vor < o.1) {
                            owner = Some((r_idx, d_vor, dh, dn));
                        }
                    }
                    if let Some((r_idx, d_vor, dh, dn)) = owner {
                        if best_for_root[r_idx].is_none_or(|b| d_vor < b.0) {
                            best_for_root[r_idx] = Some((d_vor, dh, dn));
                        }
                    }
                }
                for (_, dh, dn) in best_for_root.into_iter().flatten() {
                    sum_dh2 += dh * dh;
                    sum_dn2 += dn * dn;
                    n_resid += 1;
                }
            }
            let n_supported = root_supported.iter().filter(|&&ok| ok).count();

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

    // Aliphatic (`CA` / `CB`) spin-system backing fraction among carbon-bearing spin systems (`P1-7`):
    // catches `HNCO`-only or folded-aliphatic cubes that previously earned `hsqc_3d_support_frac = 1.00`
    // (`Pass`) from `13C'` carbonyl strips alone, without penalizing unpruned zero-carbon 2D noise roots.
    let aliphatic_support_frac = if systems.is_empty() {
        1.0
    } else {
        let n_carbon_backed = systems
            .iter()
            .filter(|ss| ss.has_observed_carbons())
            .count();
        if n_carbon_backed == 0 {
            0.0
        } else {
            let n_aliph = systems
                .iter()
                .filter(|ss| {
                    ss.cands
                        .iter()
                        .any(|((nuc, _), v)| !v.is_empty() && (nuc == "CA" || nuc == "CB"))
                })
                .count();
            (n_aliph as f64) / (n_carbon_backed as f64)
        }
    };

    let hsqc_multiplicity_ratio = (n_amide_hsqc_roots as f64) / (n_assignable_amides as f64);
    let raw_hsqc_multiplicity_ratio = (n_raw_hsqc_roots as f64) / (n_assignable_amides as f64);

    let detailed_offsets = opt_detailed_offsets
        .copied()
        .unwrap_or_else(|| estimate_reference_offsets_detailed(systems, &norm_seq, priors));
    let (offset_consistency_ok, offset_issues) = check_offset_consistency(&detailed_offsets);
    let weak_offset_issue =
        has_weak_likelihood_large_offset(&detailed_offsets, gate_cfg.min_large_offset_gain_per_obs);

    // Sequential `(i, i-1)` link coverage (`P >= 1.0` and corroborated `P >= 3.5`; `P1-7`)
    let k_len = norm_seq.len();
    let mut at = vec![-1i32; k_len];
    let mut assigned_count = 0usize;
    let mut strong_belief_count = 0usize;
    let mut placed_confidences = Vec::new();
    for (s, &pos) in result.position.iter().enumerate() {
        if pos >= 0 && (pos as usize) < k_len {
            at[pos as usize] = s as i32;
            assigned_count += 1;
            let c = result.confidence.get(s).copied().unwrap_or(0.0);
            placed_confidences.push(c);
            if c >= QC_STRONG_BELIEF_THRESHOLD {
                strong_belief_count += 1;
            }
        }
    }

    let (connectivity_frac, corroborated_connectivity_frac) =
        if assigned_count == 0 || systems.is_empty() {
            (0.0, 0.0)
        } else {
            let p_mat = pair_matrix(systems, score_cfg);
            let mut linkable_assigned = 0usize;
            let mut linked_assigned = 0usize;
            let mut corr_linked_assigned = 0usize;
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
                let prev_p = if can_prev && at[k - 1] >= 0 {
                    p_mat.get(at[k - 1] as usize, s as usize)
                } else {
                    f64::NEG_INFINITY
                };
                let next_p = if can_next && at[k + 1] >= 0 {
                    p_mat.get(s as usize, at[k + 1] as usize)
                } else {
                    f64::NEG_INFINITY
                };
                let best_link = prev_p.max(next_p);
                if best_link >= 1.0 {
                    linked_assigned += 1;
                }
                if best_link >= QC_CORROBORATED_PAIR_THRESHOLD {
                    corr_linked_assigned += 1;
                }
            }
            if linkable_assigned == 0 {
                (1.0, 1.0)
            } else {
                (
                    (linked_assigned as f64) / (linkable_assigned as f64),
                    (corr_linked_assigned as f64) / (linkable_assigned as f64),
                )
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
    if !systems.is_empty() && aliphatic_support_frac < 0.20 {
        abstain_reasons.push(format!(
            "catastrophic aliphatic (CA/CB) strip starvation ({:.1}% < 20.0%)",
            aliphatic_support_frac * 100.0
        ));
    }
    if assigned_count >= 6 && connectivity_frac < 0.15 {
        abstain_reasons.push(format!(
            "sequential link coverage collapse ({:.1}% < 15.0%)",
            connectivity_frac * 100.0
        ));
    }
    if is_unconverged_solution_margin(assigned_count, solution_margin, hsqc_multiplicity_ratio) {
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
    if n_amide_hsqc_roots > 0
        && hsqc_3d_support_frac < 0.45
        && supported_roots_per_assignable < 0.35
        && solution_margin < 0.20
    {
        abstain_reasons.push(format!(
            "severe 3D strip starvation with weak convergence (3D support {:.1}%, supported/assignable {:.1}%, margin {:.1}%)",
            hsqc_3d_support_frac * 100.0,
            supported_roots_per_assignable * 100.0,
            solution_margin * 100.0,
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
            aliphatic_support_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            solution_margin,
            hc_call_fraction: 0.0,
            protein_hc_gate_triggered: false,
            warn_sub_hc_uncalibrated: false,
            confidence_calibrated: true,
            calibration_warning: None,
            reasons: abstain_reasons,
        };
    }

    let mut scale = 1.0_f64;

    if hsqc_multiplicity_ratio > 1.45 && (peaks.is_none() || supported_roots_per_assignable > 1.35)
    {
        let pen = ((hsqc_multiplicity_ratio - 1.45) * 0.20)
            .clamp(gate_cfg.multiplicity_penalty_floor, 0.22);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "elevated 3D-backed HSQC amide multiplicity ({:.2}x > 1.45x, supported {:.2}x; possible multi-conformer crowding)",
            hsqc_multiplicity_ratio, supported_roots_per_assignable
        ));
    }
    if hsqc_3d_support_frac < 0.55 && supported_roots_per_assignable < 0.80 {
        let pen = ((0.55 - hsqc_3d_support_frac) * 0.45).clamp(0.06, 0.18);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "low 3D strip support ({:.1}% < 55.0%, supported/assignable {:.2}x)",
            hsqc_3d_support_frac * 100.0,
            supported_roots_per_assignable
        ));
    }
    if !systems.is_empty() && aliphatic_support_frac < 0.50 {
        scale *= 0.90;
        warn_reasons.push(format!(
            "sparse aliphatic (CA/CB) strip backing ({:.1}% < 50.0%)",
            aliphatic_support_frac * 100.0
        ));
    }
    if hsqc_3d_rms_dh_ppm > 0.028 || hsqc_3d_rms_dn_ppm > 0.26 {
        scale *= 0.88;
        warn_reasons.push(format!(
            "elevated Voronoi HSQC-to-3D residual drift (rms dH={:.3}, dN={:.2} ppm)",
            hsqc_3d_rms_dh_ppm, hsqc_3d_rms_dn_ppm
        ));
    }
    if !offset_consistency_ok {
        scale *= 0.78;
        warn_reasons.extend(offset_issues);
    }
    if let Some(ref weak_off_msg) = weak_offset_issue {
        if offset_consistency_ok {
            scale *= 0.78;
        }
        warn_reasons.push(weak_off_msg.clone());
    }
    if assigned_count >= 8 && connectivity_frac < 0.38 {
        let pen = ((0.38 - connectivity_frac) * 0.55).clamp(0.06, 0.16);
        scale *= 1.0 - pen;
        warn_reasons.push(format!(
            "fragmented sequential connectivity ({:.1}% < 38.0%)",
            connectivity_frac * 100.0
        ));
    }
    if assigned_count >= 8
        && corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity
    {
        scale *= 0.90;
        warn_reasons.push(format!(
            "weak corroborated multi-channel sequential connectivity ({:.1}% < {:.1}% at P >= 3.5 nats)",
            corroborated_connectivity_frac * 100.0,
            gate_cfg.min_corroborated_connectivity * 100.0
        ));
    }

    let applied_scale = if warn_reasons.is_empty() {
        1.0
    } else {
        scale.clamp(0.65, 0.92)
    };

    // Evaluate pre-cap HC call fraction after multiplicative `Warn` damping (`P1-7`)
    let hc_count = placed_confidences
        .iter()
        .filter(|&&c| (c * applied_scale) >= 0.80)
        .count();
    let hc_call_fraction = if assigned_count > 0 {
        (hc_count as f64) / (assigned_count as f64)
    } else {
        0.0
    };

    // Protein-level HC cap gate (`P1-7`): suppress unreliable `>= 0.80` tail calls (`cap = 0.79`)
    // when label-free global evidence shows unconverged corroborated links, isolated sparse HC
    // spikes, severe multi-conformer crowding with low global margin, or weakly supported offsets.
    let mut hc_gate_reasons = Vec::new();
    if assigned_count >= 12 {
        if hc_count > 0 && hc_call_fraction < gate_cfg.min_hc_call_fraction {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[sparse_hc_tail]: isolated HC tail ({}/{} = {:.1}% < {:.1}%)",
                hc_count,
                assigned_count,
                hc_call_fraction * 100.0,
                gate_cfg.min_hc_call_fraction * 100.0
            ));
        }
        if corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[low_corroborated_connectivity]: corroborated links {:.1}% < {:.1}%",
                corroborated_connectivity_frac * 100.0,
                gate_cfg.min_corroborated_connectivity * 100.0
            ));
        }
        if supported_roots_per_assignable > 1.60
            && solution_margin < gate_cfg.min_crowded_solution_margin
        {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[crowded_low_margin]: supported roots {:.2}x > 1.60x with margin {:.1}% < {:.1}%",
                supported_roots_per_assignable,
                solution_margin * 100.0,
                gate_cfg.min_crowded_solution_margin * 100.0
            ));
        }
        if (!offset_consistency_ok || weak_offset_issue.is_some()) && solution_margin < 0.50 {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[offset_low_margin]: suspect referencing offset with margin {:.1}% < 50.0%",
                solution_margin * 100.0
            ));
        }
    } else if hc_count > 0 && assigned_count * 5 < n_assignable_amides {
        hc_gate_reasons.push(format!(
            "protein_hc_gate[sparse_sequence_placement]: only {}/{} assignable amides placed (< 20.0%)",
            assigned_count, n_assignable_amides
        ));
    }

    let protein_hc_gate_triggered = !hc_gate_reasons.is_empty();
    let confidence_cap = if protein_hc_gate_triggered {
        QC_PROTEIN_HC_CONFIDENCE_CAP
    } else {
        1.0
    };
    warn_reasons.extend(hc_gate_reasons);

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
            aliphatic_support_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            solution_margin,
            hc_call_fraction,
            protein_hc_gate_triggered: false,
            warn_sub_hc_uncalibrated: false,
            confidence_calibrated: true,
            calibration_warning: None,
            reasons: Vec::new(),
        }
    } else {
        DatasetQcReport {
            verdict: DatasetQcVerdict::Warn,
            confidence_scale: applied_scale,
            confidence_cap,
            already_applied: false,
            n_assignable_amides,
            n_raw_hsqc_roots,
            n_amide_hsqc_roots,
            hsqc_3d_support_frac,
            supported_roots_per_assignable,
            aliphatic_support_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            solution_margin,
            hc_call_fraction,
            protein_hc_gate_triggered,
            warn_sub_hc_uncalibrated: true,
            confidence_calibrated: true,
            calibration_warning: None,
            reasons: warn_reasons,
        }
    }
}

/// Apply per-dataset quality-control calibration / abstention to `result.confidence`:
/// - `DatasetQcVerdict::Abstain`: caps every confidence at `0.45` (`<= 0.45 < 0.50`) so `n_high_conf == 0`.
/// - `DatasetQcVerdict::Warn`: multiplies confidences by `qc.confidence_scale` and clamps at
///   `qc.confidence_cap` (`<= 0.79` when `qc.protein_hc_gate_triggered`, `1.0` otherwise), unless
///   `qc.already_applied` is set.
/// - `DatasetQcVerdict::Pass`: leaves confidences untouched (`scale = 1.0, cap = 1.0`).
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

/// Standard 10-bin equal-width probability partition (`[0.0, 0.1, ..., 1.0]`, Guo et al. 2017)
/// serving as the primary Phase 13 pooled calibration gate (`ECE_10 <= 0.050`, `P1-8`).
pub const UNIFORM_CALIBRATION_GRID_10: &[f64] =
    &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Compute binned Expected Calibration Error (ECE) of `(confidence, is_correct)` pairs over an
/// arbitrary strictly increasing partition `grid` (`g_0 < g_1 < ... < g_M`, with `[g_b, g_{b+1})`
/// for `b < M - 1` and `[g_{M - 1}, g_M]` inclusive on the top bin).
pub fn compute_grid_ece(predictions: &[(f64, bool)], grid: &[f64]) -> f64 {
    if predictions.is_empty() || grid.len() < 2 {
        return 0.0;
    }
    let n_bins = grid.len() - 1;
    let mut counts = vec![0usize; n_bins];
    let mut corrects = vec![0usize; n_bins];
    let mut conf_sums = vec![0.0_f64; n_bins];
    for &(raw_conf, is_ok) in predictions {
        let c = raw_conf.clamp(0.0, 1.0);
        let mut idx = n_bins - 1;
        for b in 0..n_bins {
            if c < grid[b + 1] || b + 1 == n_bins {
                idx = b;
                break;
            }
        }
        counts[idx] += 1;
        conf_sums[idx] += c;
        if is_ok {
            corrects[idx] += 1;
        }
    }
    let total = predictions.len() as f64;
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        let cnt = counts[b];
        if cnt > 0 {
            let mc = conf_sums[b] / (cnt as f64);
            let ea = (corrects[b] as f64) / (cnt as f64);
            ece += ((cnt as f64) / total) * (mc - ea).abs();
        }
    }
    ece
}

/// Expected finite-sample plug-in `ECE` floor of an **exact Bernoulli-calibrated oracle**
/// (`Y_i ~ Bernoulli(p_i)`) over `grid` when each physical residue is repeated `repeat_factor`
/// times (`repeat_factor = 1.0` for single-seed runs, `5.0` for 5-seed per-protein evaluations;
/// Ferro & Fricker, *Q. J. R. Meteorol. Soc.* 2012; Kumar, Liang & Ma, *NeurIPS* 2019).
///
/// For bin `b` with effective independent sample size `n_eff = max(1, count_b / repeat_factor)`
/// and mean confidence `p_b`, `E[|hat_a_b - p_b|] ≈ sqrt(2 * p_b * (1 - p_b) / (pi * n_eff))`.
pub fn compute_grid_ece_bernoulli_floor(
    predictions: &[(f64, bool)],
    grid: &[f64],
    repeat_factor: f64,
) -> f64 {
    if predictions.is_empty() || grid.len() < 2 {
        return 0.0;
    }
    let rep = repeat_factor.max(1.0);
    let n_bins = grid.len() - 1;
    let mut counts = vec![0usize; n_bins];
    let mut conf_sums = vec![0.0_f64; n_bins];
    for &(raw_conf, _) in predictions {
        let c = raw_conf.clamp(0.0, 1.0);
        let mut idx = n_bins - 1;
        for b in 0..n_bins {
            if c < grid[b + 1] || b + 1 == n_bins {
                idx = b;
                break;
            }
        }
        counts[idx] += 1;
        conf_sums[idx] += c;
    }
    let total = predictions.len() as f64;
    let two_over_pi = 2.0 / std::f64::consts::PI;
    let mut floor = 0.0_f64;
    for b in 0..n_bins {
        let cnt = counts[b];
        if cnt > 0 {
            let mc = (conf_sums[b] / (cnt as f64)).clamp(0.0, 1.0);
            let n_eff = ((cnt as f64) / rep).max(1.0);
            let bin_floor = (two_over_pi * mc * (1.0 - mc) / n_eff).sqrt();
            floor += ((cnt as f64) / total) * bin_floor;
        }
    }
    floor
}

/// Finite-sample variance-debiased binned `ECE` (Ferro & Fricker, *Q. J. R. Meteorol. Soc.* 2012,
/// Eq. 7; Kumar, Liang & Ma, *NeurIPS* 2019), subtracting the finite-sample Bernoulli plug-in
/// variance `p_b * (1 - p_b) / n_eff` inside each bin before taking the root-squared calibration
/// discrepancy. Converges to [`compute_grid_ece`] as `n_eff -> inf` (`pooled_ECE_10`), while
/// removing the artificial `O(1 / sqrt(n_b))` floor on single `30–100`-residue proteins.
pub fn compute_debiased_grid_ece(
    predictions: &[(f64, bool)],
    grid: &[f64],
    repeat_factor: f64,
) -> f64 {
    if predictions.is_empty() || grid.len() < 2 {
        return 0.0;
    }
    let rep = repeat_factor.max(1.0);
    let n_bins = grid.len() - 1;
    let mut counts = vec![0usize; n_bins];
    let mut corrects = vec![0usize; n_bins];
    let mut conf_sums = vec![0.0_f64; n_bins];
    for &(raw_conf, is_ok) in predictions {
        let c = raw_conf.clamp(0.0, 1.0);
        let mut idx = n_bins - 1;
        for b in 0..n_bins {
            if c < grid[b + 1] || b + 1 == n_bins {
                idx = b;
                break;
            }
        }
        counts[idx] += 1;
        conf_sums[idx] += c;
        if is_ok {
            corrects[idx] += 1;
        }
    }
    let total = predictions.len() as f64;
    let mut deb_ece = 0.0_f64;
    for b in 0..n_bins {
        let cnt = counts[b];
        if cnt > 0 {
            let mc = (conf_sums[b] / (cnt as f64)).clamp(0.0, 1.0);
            let ea = (corrects[b] as f64) / (cnt as f64);
            let n_eff = ((cnt as f64) / rep).max(1.0);
            let var_bias = mc * (1.0 - mc) / n_eff;
            let deb_gap = ((mc - ea).powi(2) - var_bias).max(0.0).sqrt();
            deb_ece += ((cnt as f64) / total) * deb_gap;
        }
    }
    deb_ece
}

/// Per-protein confidence reliability diagram (`[0.0..0.5, 0.5..0.8, 0.8..0.9, 0.9..1.0]`),
/// 4-bin (`ece`), 10-bin (`ece_10`), finite-sample debiased 10-bin (`debiased_ece_10`), and
/// 12-bin (`ece_12`) Expected Calibration Error, plus explicit high-confidence abstention status.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProteinReliabilityCurve {
    pub bins: [ReliabilityBin; 4],
    pub total_predictions: usize,
    /// 4-bin Expected Calibration Error (`[0.0, 0.5), [0.5, 0.8), [0.8, 0.9), [0.9, 1.0]`).
    pub ece: f64,
    /// Uniform 10-bin Expected Calibration Error (`[0.0, 0.1, ..., 1.0]`, Guo et al. 2017; primary pooled gate `P1-8`).
    pub ece_10: f64,
    /// Ferro–Fricker (2012) finite-sample variance-debiased 10-bin ECE (`P1-8`).
    #[serde(default)]
    pub debiased_ece_10: f64,
    /// Analytical Half-Normal Bernoulli oracle floor on 10-bin plug-in ECE at this sample size (`P1-8`).
    #[serde(default)]
    pub ece_10_bernoulli_floor: f64,
    /// Canonical 12-bin (`g12`, [`CANONICAL_CALIBRATION_GRID_12`]) Expected Calibration Error (`P2`).
    pub ece_12: f64,
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

    /// Construct a `ProteinReliabilityCurve` (`ECE_4`, `ECE_10`, `ECE_12`) assuming independent
    /// observations (`repeat_factor = 1.0`).
    pub fn from_predictions(predictions: &[(f64, bool)], hc_threshold: f64) -> Self {
        Self::from_predictions_with_repeats(predictions, hc_threshold, 1.0)
    }

    /// Construct a `ProteinReliabilityCurve` when `predictions` pools `repeat_factor` repeated
    /// solver seeds of the same physical residue set (`repeat_factor = 5.0` for 5-seed per-protein
    /// benchmarks, `1.0` for single-seed or cross-protein pooled cohorts).
    pub fn from_predictions_with_repeats(
        predictions: &[(f64, bool)],
        hc_threshold: f64,
        repeat_factor: f64,
    ) -> Self {
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
        let ece_10 = compute_grid_ece(predictions, UNIFORM_CALIBRATION_GRID_10);
        let debiased_ece_10 =
            compute_debiased_grid_ece(predictions, UNIFORM_CALIBRATION_GRID_10, repeat_factor);
        let ece_10_bernoulli_floor = compute_grid_ece_bernoulli_floor(
            predictions,
            UNIFORM_CALIBRATION_GRID_10,
            repeat_factor,
        );
        let ece_12 = compute_grid_ece(predictions, CANONICAL_CALIBRATION_GRID_12);

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
            ece_10,
            debiased_ece_10,
            ece_10_bernoulli_floor,
            ece_12,
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

/// Canonical 12-bin (`13`-edge, `g12`) probability partition used across all solver and engine
/// isotonic PAV calibration fits (`0.0 = g_0 < g_1 < ... < g_12 = 1.0`).
pub const CANONICAL_CALIBRATION_GRID_12: &[f64] = &[
    0.0, 0.15, 0.30, 0.45, 0.58, 0.68, 0.76, 0.83, 0.89, 0.94, 0.975, 0.992, 1.0,
];

/// Canonical total pseudocount (`laplace_prior = 0.5`) per `CANONICAL_CALIBRATION_GRID_12` interval,
/// corresponding to a symmetric `Beta(0.25, 0.25)` prior (`0.25` positive + `0.25` negative).
pub const CANONICAL_CALIBRATION_LAPLACE_PRIOR: f64 = 0.5;

/// Conservative conjugate `Beta(alpha_pos = 0.75, beta_neg = 2.25)` prior (`3.0` total pseudocount,
/// prior mean `0.25`) used for CPU `standard` (`runs = 16`, `P1-8`) so sparse upper-intermediate
/// replica-consensus bins shrink conservatively rather than overcommitting on sparse proteins.
pub const STANDARD_CPU_PRIOR_ALPHA: f64 = 0.75;

/// Negative pseudocount (`beta_neg = 2.25`) paired with [`STANDARD_CPU_PRIOR_ALPHA`] (`P1-8`).
pub const STANDARD_CPU_PRIOR_BETA: f64 = 2.25;

/// Pre-registered candidate empirical-Bayes CPU shrinkage weights evaluated by inner 12-fold LOPO
/// inside each outer `dev` LOPO fold when calibrating `STANDARD_GPU` (`P1-8`).
pub const CANDIDATE_GPU_SHRINKAGE_WEIGHTS: &[f64] = &[0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0];

/// Fit a monotone piecewise-linear calibration knot table on `grid` (`0.0 = g_0 < ... < g_m = 1.0`)
/// using general conjugate `Beta(alpha_pos, beta_neg)`-smoothed Pool-Adjacent-Violators (PAV)
/// isotonic regression (Ayer et al. 1955; Zadrozny & Elkan 2002).
pub fn fit_isotonic_monotone_knots_beta(
    samples: &[(f64, bool)],
    grid: &[f64],
    alpha_pos: f64,
    beta_neg: f64,
) -> Vec<(f64, f64)> {
    if grid.len() < 2 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    let n_bins = grid.len() - 1;
    let mut weights = vec![0.0_f64; n_bins];
    let mut pos_sums = vec![0.0_f64; n_bins];
    let mut x_sums = vec![0.0_f64; n_bins];
    let a = alpha_pos.max(0.0);
    let b_neg = beta_neg.max(0.0);
    let prior_w = (a + b_neg).max(1e-6);
    let prior_pos = if a + b_neg > 0.0 { a } else { 0.5 * prior_w };

    for b in 0..n_bins {
        let mid = 0.5 * (grid[b] + grid[b + 1]);
        weights[b] = prior_w;
        pos_sums[b] = prior_pos;
        x_sums[b] = prior_w * mid;
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
        x_sums[idx] += p;
        if is_ok {
            pos_sums[idx] += 1.0;
        }
    }

    let mut x_knots = vec![0.0_f64; n_bins];
    for b in 0..n_bins {
        x_knots[b] = (x_sums[b] / weights[b]).clamp(grid[b], grid[b + 1]);
    }

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

    let mut knots = Vec::with_capacity(n_bins + 3);
    knots.push((0.0, 0.0));
    let x_left = 0.5 * x_knots[0];
    if x_left > 0.0 {
        knots.push((x_left, 0.5 * bin_vals[0]));
    }
    for b in 0..n_bins {
        knots.push((x_knots[b], bin_vals[b]));
    }
    let y_end = bin_vals[n_bins - 1].max(0.0);
    knots.push((1.0, y_end));
    knots
}

/// Fit a monotone piecewise-linear calibration knot table on `grid` (`0.0 = g_0 < ... < g_m = 1.0`)
/// using symmetric `Beta(laplace_prior / 2, laplace_prior / 2)`-smoothed PAV isotonic regression.
pub fn fit_isotonic_monotone_knots(
    samples: &[(f64, bool)],
    grid: &[f64],
    laplace_prior: f64,
) -> Vec<(f64, f64)> {
    let half = 0.5 * laplace_prior.max(1e-6);
    fit_isotonic_monotone_knots_beta(samples, grid, half, half)
}

/// Hierarchical empirical-Bayes isotonic PAV fitter (`Q2` / `P1-8`): pools primary `samples`
/// (weight `1.0` per observation) with `anchor_samples` from the matched CPU Hamiltonian
/// (`weight = anchor_weight` per observation) plus symmetric `Beta(laplace_prior / 2, laplace_prior / 2)`
/// smoothing on `grid`. When `anchor_weight <= 0.0` or `anchor_samples` is empty, reduces
/// identically to [`fit_isotonic_monotone_knots`].
pub fn fit_isotonic_monotone_knots_shrunk_toward(
    samples: &[(f64, bool)],
    anchor_samples: &[(f64, bool)],
    anchor_weight: f64,
    grid: &[f64],
    laplace_prior: f64,
) -> Vec<(f64, f64)> {
    if anchor_weight <= 0.0 || anchor_samples.is_empty() {
        return fit_isotonic_monotone_knots(samples, grid, laplace_prior);
    }
    if grid.len() < 2 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    let n_bins = grid.len() - 1;
    let mut weights = vec![0.0_f64; n_bins];
    let mut pos_sums = vec![0.0_f64; n_bins];
    let mut x_sums = vec![0.0_f64; n_bins];
    for b in 0..n_bins {
        let mid = 0.5 * (grid[b] + grid[b + 1]);
        let prior_w = laplace_prior.max(1e-6);
        weights[b] = prior_w;
        pos_sums[b] = 0.5 * prior_w;
        x_sums[b] = prior_w * mid;
    }

    let mut accum = |raw_p: f64, is_ok: bool, w: f64| {
        let p = raw_p.clamp(0.0, 1.0);
        let mut idx = n_bins - 1;
        for b in 0..n_bins {
            if p < grid[b + 1] || b + 1 == n_bins {
                idx = b;
                break;
            }
        }
        weights[idx] += w;
        x_sums[idx] += w * p;
        if is_ok {
            pos_sums[idx] += w;
        }
    };

    for &(raw_p, is_ok) in anchor_samples {
        accum(raw_p, is_ok, anchor_weight);
    }
    for &(raw_p, is_ok) in samples {
        accum(raw_p, is_ok, 1.0);
    }

    let mut x_knots = vec![0.0_f64; n_bins];
    for b in 0..n_bins {
        x_knots[b] = (x_sums[b] / weights[b]).clamp(grid[b], grid[b + 1]);
    }

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

    let mut knots = Vec::with_capacity(n_bins + 3);
    knots.push((0.0, 0.0));
    let x_left = 0.5 * x_knots[0];
    if x_left > 0.0 {
        knots.push((x_left, 0.5 * bin_vals[0]));
    }
    for b in 0..n_bins {
        knots.push((x_knots[b], bin_vals[b]));
    }
    let y_end = bin_vals[n_bins - 1].max(0.0);
    knots.push((1.0, y_end));
    knots
}

/// Nested Leave-One-Protein-Out (LOPO) selector for `STANDARD_GPU` CPU-anchor shrinkage weight
/// (`P1-8`): given an outer training set of `(gpu_seeds, cpu_raw_pairs, gpu_trainable, cpu_trainable)`
/// per protein, runs an inner `M`-fold LOPO over [`CANDIDATE_GPU_SHRINKAGE_WEIGHTS`], selecting
/// `w` that minimizes inner OOF pooled 10-bin `ECE_10` ([`UNIFORM_CALIBRATION_GRID_10`]) subject
/// to inner OOF micro HC-precision non-degradation relative to `w = 0.0` and within `0.25%` of
/// the feasible `94.5%` target. Returns `(selected_w, per_candidate_(inner_ece_10, inner_micro_hcp))`.
pub fn select_gpu_shrinkage_weight_nested_lopo(
    gpu_proteins: &[Vec<ProteinHcTrainingSeed>],
    cpu_raw_by_protein: &[Vec<(f64, bool)>],
    gpu_trainable: &[bool],
    cpu_trainable: &[bool],
    gate_cfg: &ProteinHcGateConfig,
) -> (f64, Vec<(f64, f64, f64)>) {
    let m = gpu_proteins.len();
    if m < 2 || cpu_raw_by_protein.len() != m {
        return (CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT, Vec::new());
    }
    let mut stats = Vec::with_capacity(CANDIDATE_GPU_SHRINKAGE_WEIGHTS.len());
    for &w in CANDIDATE_GPU_SHRINKAGE_WEIGHTS {
        let mut inner_pooled = Vec::new();
        for hold in 0..m {
            let mut gpu_tr = Vec::new();
            let mut cpu_tr = Vec::new();
            for j in 0..m {
                if j == hold {
                    continue;
                }
                if gpu_trainable.get(j).copied().unwrap_or(true) {
                    for s in &gpu_proteins[j] {
                        gpu_tr.extend_from_slice(&s.raw_predictions);
                    }
                }
                if cpu_trainable.get(j).copied().unwrap_or(true) {
                    cpu_tr.extend_from_slice(&cpu_raw_by_protein[j]);
                }
            }
            let inner_knots = fit_isotonic_monotone_knots_shrunk_toward(
                &gpu_tr,
                &cpu_tr,
                w,
                CANONICAL_CALIBRATION_GRID_12,
                CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            );
            for s in &gpu_proteins[hold] {
                inner_pooled.extend(evaluate_seed_predictions_with_gate(
                    s,
                    &inner_knots,
                    gate_cfg,
                ));
            }
        }
        let ece10 = compute_grid_ece(&inner_pooled, UNIFORM_CALIBRATION_GRID_10);
        let hc_tot = inner_pooled.iter().filter(|&&(c, _)| c >= 0.80).count();
        let hc_ok = inner_pooled
            .iter()
            .filter(|&&(c, ok)| c >= 0.80 && ok)
            .count();
        let micro_hcp = if hc_tot > 0 {
            (hc_ok as f64) / (hc_tot as f64)
        } else {
            1.0
        };
        stats.push((w, ece10, micro_hcp));
    }

    let hcp_w0 = stats.first().map(|x| x.2).unwrap_or(0.0);
    let max_hcp = stats.iter().map(|x| x.2).fold(0.0_f64, f64::max);
    let target_hcp = hcp_w0.max(0.945_f64.min(max_hcp - 0.0025));

    let best_w = stats
        .iter()
        .filter(|&&(_, _, hcp)| hcp + 1e-4 >= target_hcp)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .or_else(|| stats.iter().max_by(|a, b| a.2.total_cmp(&b.2)))
        .map(|x| x.0)
        .unwrap_or(CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT);

    (best_w, stats)
}

/// Canonical empirical-Bayes shrinkage weight (`w = 0.50`, `P1-8`) pooling CPU `standard` (`f64`)
/// bin counts with WebGPU `standard` (`f32`) bin counts before isotonic PAV projection.
///
/// Selected strictly via the full-13 `dev` inner-LOPO objective ([`select_gpu_shrinkage_weight_nested_lopo`]),
/// while outer OOF generalization is evaluated by **nested** 13x12 LOPO (selecting `w_i` strictly
/// inside the 12 training proteins of each outer fold `i`, with zero outer-fold leakage).
pub const CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT: f64 = 0.50;

/// Monotone calibration knot table for `evo-graph` (`population_size = 64, num_islands = 16`,
/// [`EVO_GRAPH_CALIBRATED_RUNS`]; `P1-2` / `P1-7` / `P1-8`), fitted via isotonic PAV on
/// [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) with symmetric `Beta(0.25, 0.25)` smoothing across
/// the 13-entry `dev` split.
pub const EVO_GRAPH_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0204, 0.0705),
    (0.0407, 0.1409),
    (0.2201, 0.3322),
    (0.3739, 0.3754),
    (0.5083, 0.6623),
    (0.6299, 0.6623),
    (0.7269, 0.6623),
    (0.7992, 0.7473),
    (0.8650, 0.7473),
    (0.9309, 0.8250),
    (0.9586, 0.8250),
    (0.9868, 0.8445),
    (0.9997, 0.9793),
    (1.0000, 0.9793),
];

/// Monotone calibration knot table for CPU `standard` (`runs = 16`, [`STANDARD_CPU_CALIBRATED_RUNS`];
/// `P1-6` / `P1-8`), fitted via conservative conjugate `Beta(0.75, 2.25)` isotonic PAV
/// ([`fit_isotonic_monotone_knots_beta`] with [`STANDARD_CPU_PRIOR_ALPHA`], [`STANDARD_CPU_PRIOR_BETA`])
/// on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) across the 13-entry `dev` split.
pub const STANDARD_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0273, 0.1015),
    (0.0546, 0.2030),
    (0.2180, 0.4610),
    (0.3772, 0.5099),
    (0.5083, 0.5768),
    (0.6251, 0.5768),
    (0.7216, 0.7508),
    (0.8023, 0.7508),
    (0.8698, 0.8912),
    (0.9333, 0.9394),
    (0.9586, 0.9394),
    (0.9835, 0.9409),
    (0.9995, 0.9542),
    (1.0000, 0.9542),
];

/// Monotone calibration knot table for WebGPU `standard` (`runs = 64` `f32` multi-replica elite
/// consensus, [`STANDARD_GPU_CALIBRATED_RUNS`]; `P1-8` / `Q2`), fitted via CPU-anchored isotonic
/// PAV ([`fit_isotonic_monotone_knots_shrunk_toward`] with nested-LOPO-selected
/// [`CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT`] `= 0.50`) on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`)
/// across the 13-entry `dev` split on `Apple M5 Pro` (`Metal` via `wgpu`).
pub const STANDARD_GPU_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0264, 0.0961),
    (0.0529, 0.1922),
    (0.2199, 0.4211),
    (0.3801, 0.5339),
    (0.5116, 0.6633),
    (0.6273, 0.6896),
    (0.7200, 0.8427),
    (0.7975, 0.8596),
    (0.8630, 0.9351),
    (0.9227, 0.9574),
    (0.9585, 0.9574),
    (0.9840, 0.9574),
    (0.9995, 0.9574),
    (1.0000, 0.9574),
];

/// Map raw physical (`T = 1.0`) `evo-graph` marginal belief into calibrated probability.
#[inline]
pub fn apply_evo_graph_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, EVO_GRAPH_CALIBRATION_KNOTS)
}

/// Map raw physical (`T = 1.0`) CPU `standard` marginal belief into calibrated probability.
#[inline]
pub fn apply_standard_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, STANDARD_CALIBRATION_KNOTS)
}

/// Map raw physical (`T = 1.0`) WebGPU `standard` marginal belief into calibrated probability.
#[inline]
pub fn apply_standard_gpu_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, STANDARD_GPU_CALIBRATION_KNOTS)
}
