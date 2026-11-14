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
use crate::solver::{pair_matrix, unary_matrix, AssignResult, ScoreConfig};
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
    /// Fraction of placed spin systems backed by a resolved bidirectional intra `(0)` + sequential `(-1)`
    /// aliphatic (`13CA` or `13CB`) shift pair (`|v0 - vm1| > 0.18` ppm or multi-experiment corroborated
    /// degenerate `sup0 == 0 && supm1 >= 2`; `13.1:P1-7b`).
    #[serde(default = "default_one_f64")]
    pub resolved_bidir_aliphatic_frac: f64,
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
    /// Fraction of linkable assigned spin systems belonging to a maximal contiguous `P >= 3.5`-nat
    /// sequential chain of length `>= 4.min(seg_len)` within its non-Proline sequence segment (`13.1:P1-7b`).
    #[serde(default = "default_one_f64")]
    pub chain_corroborated_frac: f64,
    /// Mean Bayesian unary-vs-random-slot specificity probability `1 / (1 + exp(-ΔU(s, k)))` in `[0, 1]`
    /// over placed spin systems (`13.1:P1-7b`).
    #[serde(default = "default_one_f64")]
    pub unary_specificity: f64,
    /// Fraction of assigned spin systems with strong consensus/marginal belief (`>= 0.65`).
    pub solution_margin: f64,
    /// Pre-cap high-confidence (`>= 0.80`) call fraction among placed spin systems (`n_hc / n_assigned`; `P1-7`).
    #[serde(default)]
    pub hc_call_fraction: f64,
    /// True when the label-free protein-level HC gate (`P1-7`) capped post-QC confidence at `<= 0.79`.
    #[serde(default)]
    pub protein_hc_gate_triggered: bool,
    /// True when one or more input spectra were zero-filled from halted/truncated raw FIDs via
    /// explicit `--allow-truncated-fid` opt-in (`F2`).
    #[serde(default)]
    pub truncated_fid_zero_filled: bool,
    /// True under `DatasetQcVerdict::Warn` (`Q4`): explicitly marks sub-HC calls (`confidence < 0.80`)
    /// as uncalibrated under `Warn` damping rather than fitting a separate small-sample `Warn` isotonic curve.
    #[serde(default)]
    pub warn_sub_hc_uncalibrated: bool,
    /// True when the solver replica/population count matched the canonical isotonic calibration fit
    /// ([`STANDARD_CPU_CALIBRATED_RUNS`] `= 16` for CPU `standard`,
    /// [`STANDARD_GPU_CALIBRATED_RUNS`] `= 64` for WebGPU `standard`,
    /// [`EVO_GRAPH_CALIBRATED_RUNS`] `= 64` for `evo-graph`,
    /// [`BAYES_SA_CPU_CALIBRATED_RUNS`] `= 32` for CPU `bayes-sa`,
    /// [`BAYES_SA_GPU_CALIBRATED_RUNS`] `= 128` for WebGPU `bayes-sa`); `false` when a caller overrides
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

/// Canonical CPU `bayes-sa` replica count (`replicas = 32`) at which [`BAYES_SA_CALIBRATION_KNOTS`] is fitted (`13.4`).
pub const BAYES_SA_CPU_CALIBRATED_RUNS: usize = 32;

/// Canonical WebGPU `bayes-sa` replica count (`replicas = 128`) at which [`BAYES_SA_GPU_CALIBRATION_KNOTS`] is fitted (`13.4`).
pub const BAYES_SA_GPU_CALIBRATED_RUNS: usize = 128;

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
            resolved_bidir_aliphatic_frac: 1.0,
            hsqc_3d_rms_dh_ppm: 0.0,
            hsqc_3d_rms_dn_ppm: 0.0,
            hsqc_multiplicity_ratio: 1.0,
            raw_hsqc_multiplicity_ratio: 1.0,
            offset_consistency_ok: true,
            detailed_offsets: DetailedReferenceOffsets::default(),
            connectivity_frac: 1.0,
            corroborated_connectivity_frac: 1.0,
            chain_corroborated_frac: 1.0,
            unary_specificity: 1.0,
            solution_margin: 1.0,
            hc_call_fraction: 0.0,
            protein_hc_gate_triggered: false,
            truncated_fid_zero_filled: false,
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

/// Whether `offsets` carries a large carbon/nitrogen registration shift (`|dN| > 1.50` or
/// `|dCA| > 3.20` or `|dCB| > 3.20` ppm) backed by weak per-observation unlabelled log-likelihood
/// gain (`log_likelihood_gain / n_values < gain_scale * min_gain_per_obs`), indicating likely
/// spectral aliasing or wrong-nucleus folding rather than uniform chemical-shift referencing (`P1-7`).
///
/// Notes:
/// - The `1.50` ppm `15N` threshold sits strictly below the unconditional `2.50` ppm
///   [`check_offset_consistency`] ceiling and scales `min_gain_per_obs` by `0.50` (`0.075` nats/obs)
///   to reflect the broader amino-acid-marginal `15N` prior dispersion relative to `13CA`/`13CB`.
/// - Pass-2 sequence-anchor-refined offsets synchronized via `sync_detailed_reference_offsets`
///   (where `applied_ppm` diverges from Pass-1 `candidate_ppm` or carries the `1.0`-nat sync floor
///   from a Pass-1 `NotSignificant` channel) are excluded because `fit.log_likelihood_gain` records
///   only the unlabelled Pass-1 marginal curve, not the Pass-2 sequence-anchored likelihood.
pub fn has_weak_likelihood_large_offset(
    offsets: &DetailedReferenceOffsets,
    min_gain_per_obs: f64,
) -> Option<String> {
    for (label, fit, ppm_thresh, gain_scale) in [
        ("dN", &offsets.n, 1.50_f64, 0.50_f64),
        ("dCA", &offsets.ca, 3.20_f64, 1.0_f64),
        ("dCB", &offsets.cb, 3.20_f64, 1.0_f64),
    ] {
        let is_pass2_anchor_synced = (fit.applied_ppm - fit.candidate_ppm).abs() > 0.05
            || (fit.log_likelihood_gain - 1.0).abs() < 1e-9;
        if fit.status == OffsetFitStatus::Fitted
            && !is_pass2_anchor_synced
            && fit.applied_ppm.abs() > ppm_thresh
        {
            let gain_per_obs = fit.log_likelihood_gain / (fit.n_values.max(1) as f64);
            let req_gain = min_gain_per_obs * gain_scale;
            if gain_per_obs < req_gain {
                return Some(format!(
                    "weakly supported large {label} registration shift ({}, ΔLL/n={gain_per_obs:.3} < {req_gain:.3})",
                    fit.format_compact(label)
                ));
            }
        }
    }
    None
}

/// Fixed per-observation log-likelihood gain threshold (`0.15` nats/observation for `13CA`/`13CB`;
/// scaled by `0.50` to `0.075` nats/observation for `15N`) required when a large Pass-1 unlabelled
/// registration offset (`|dN| > 1.50` or `|dCA|, |dCB| > 3.20` ppm) is fitted (`P1-7`).
pub const QC_MIN_LARGE_OFFSET_GAIN_PER_OBS: f64 = 0.15;

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

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_corroborated_connectivity`
/// searched inside each LOPO training split (`P1-7` / `13.1:P1-7b`; `0.0` = inactive floor).
pub const CANDIDATE_GATE_CORROBORATED_CONNECTIVITY: &[f64] = &[0.0, 0.45, 0.50, 0.55];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_hc_call_fraction`
/// searched inside each LOPO training split (`P1-7` / `13.1:P1-7b`; `0.0` = inactive floor).
pub const CANDIDATE_GATE_HC_CALL_FRACTION: &[f64] = &[0.0, 0.02, 0.04, 0.06];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_crowded_solution_margin`
/// searched inside each LOPO training split (`P1-7` / `13.1:P1-7b`; `0.0` = inactive floor).
pub const CANDIDATE_GATE_CROWDED_SOLUTION_MARGIN: &[f64] = &[0.0, 0.35, 0.45, 0.55];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::multiplicity_penalty_floor`
/// searched inside each LOPO training split (`P1-7` / `13.1:P1-7b`; `0.0` = inactive floor).
pub const CANDIDATE_GATE_MULTIPLICITY_PENALTY_FLOOR: &[f64] = &[0.0, 0.06, 0.10, 0.12];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_chain_corroborated_frac`
/// (`0.0` = inactive floor; `13.1:P1-7b`).
pub const CANDIDATE_GATE_CHAIN_CORROBORATED_FRAC: &[f64] = &[0.0, 0.25, 0.35, 0.45];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_unary_specificity`
/// (`0.0` = inactive floor; `13.1:P1-7b`).
pub const CANDIDATE_GATE_UNARY_SPECIFICITY: &[f64] = &[0.0, 0.65, 0.72, 0.78];

/// Pre-registered 0.0-anchored candidate grid for `ProteinHcGateConfig::min_bidir_aliphatic_frac`
/// gating effective 3D bidirectional aliphatic support (`hsqc_3d_support_frac * resolved_bidir_aliphatic_frac`;
/// `0.0` = inactive floor; `13.1:P1-7b`).
pub const CANDIDATE_GATE_BIDIR_ALIPHATIC_FRAC: &[f64] = &[0.0, 0.50, 0.58, 0.62, 0.65];

/// Low-DOF label-free configuration for the protein-level HC cap gate (`P1-7` / `13.1:P1-7b`),
/// fitted strictly inside Leave-One-Protein-Out (LOPO) cross-validation over the non-cold training
/// pool (`CANDIDATE_GATE_*`) with a selective-risk objective penalizing `< 85%` per-protein HC
/// overcommitment while protecting `>= 94%` clean proteins, plus a neutral monotone Occam tie-breaker
/// (`1e-4 * sum_of_thresholds`, preferring the least restrictive gate whenever accuracy ties).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProteinHcGateConfig {
    /// Minimum corroborated sequential link density (`P >= 3.5` nats) required to emit `>= 0.80` HC calls.
    pub min_corroborated_connectivity: f64,
    /// Minimum fraction of placed spin systems reaching `>= 0.80` (`n_hc / n_assigned`) required
    /// so isolated tail spikes (`1–3%` of residues) cannot emit false HC commitments.
    pub min_hc_call_fraction: f64,
    /// Minimum `solution_margin` required on severely crowded multi-conformer datasets (`supported_roots_per_assignable > 1.60`).
    pub min_crowded_solution_margin: f64,
    /// Minimum floor on the multiplicative crowding damping penalty when `hsqc_multiplicity_ratio > 1.45`
    /// and (`!peaks_present` or `supported_roots_per_assignable > 1.35`).
    pub multiplicity_penalty_floor: f64,
    /// Minimum fraction of linkable placed residues belonging to a contiguous `>= 4.min(seg_len)`
    /// corroborated (`P >= 3.5` nats) chain (`13.1:P1-7b`; `0.0` = inactive).
    #[serde(default)]
    pub min_chain_corroborated_frac: f64,
    /// Minimum mean Bayesian unary-vs-random-slot specificity probability in `[0, 1]` over placed
    /// spin systems (`13.1:P1-7b`; `0.0` = inactive).
    #[serde(default)]
    pub min_unary_specificity: f64,
    /// Minimum effective 3D bidirectional aliphatic support (`hsqc_3d_support_frac * resolved_bidir_aliphatic_frac`),
    /// i.e. joint 3D strip backing and resolved `(0, -1)` `13CA`/`13CB` shift-pair completeness (`13.1:P1-7b`; `0.0` = inactive).
    #[serde(default)]
    pub min_bidir_aliphatic_frac: f64,
}

impl ProteinHcGateConfig {
    /// Least restrictive corner (`all 0.0`s) of the canonical `0.0`-anchored `CANDIDATE_GATE_*`
    /// search space (selected by the monotone Occam tie-breaker whenever no training protein
    /// triggers selective risk).
    #[inline]
    pub const fn least_restrictive_grid_corner() -> Self {
        Self {
            min_corroborated_connectivity: 0.0,
            min_hc_call_fraction: 0.0,
            min_crowded_solution_margin: 0.0,
            multiplicity_penalty_floor: 0.0,
            min_chain_corroborated_frac: 0.0,
            min_unary_specificity: 0.0,
            min_bidir_aliphatic_frac: 0.0,
        }
    }
}

/// Canonical unified protein-level HC gate parameters fitted across all 5 solver engines on the
/// 26-protein non-cold pool (`dev` + `seen_p12` + `seen_p13`; `13.1:P1-7b`).
pub const CANONICAL_PROTEIN_HC_GATE: ProteinHcGateConfig = ProteinHcGateConfig {
    min_corroborated_connectivity: 0.45,
    min_hc_call_fraction: 0.0,
    min_crowded_solution_margin: 0.55,
    multiplicity_penalty_floor: 0.0,
    min_chain_corroborated_frac: 0.0,
    min_unary_specificity: 0.0,
    min_bidir_aliphatic_frac: 0.58,
};

impl Default for ProteinHcGateConfig {
    fn default() -> Self {
        CANONICAL_PROTEIN_HC_GATE
    }
}

/// Per-seed label-free protein telemetry + raw `(p_raw, is_correct)` observations used to refit
/// [`ProteinHcGateConfig`] inside each LOPO training split with zero target leakage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProteinHcTrainingSeed {
    pub n_assignable_amides: usize,
    pub n_amide_hsqc_roots: usize,
    pub peaks_present: bool,
    pub hsqc_3d_support_frac: f64,
    pub supported_roots_per_assignable: f64,
    pub aliphatic_support_frac: f64,
    #[serde(default = "default_one_f64")]
    pub resolved_bidir_aliphatic_frac: f64,
    pub hsqc_multiplicity_ratio: f64,
    pub hsqc_3d_rms_dh_ppm: f64,
    pub hsqc_3d_rms_dn_ppm: f64,
    pub offset_consistency_ok: bool,
    pub weak_large_offset: bool,
    pub connectivity_frac: f64,
    pub corroborated_connectivity_frac: f64,
    #[serde(default = "default_one_f64")]
    pub chain_corroborated_frac: f64,
    #[serde(default = "default_one_f64")]
    pub unary_specificity: f64,
    pub raw_predictions: Vec<(f64, bool)>,
}

impl ProteinHcTrainingSeed {
    /// Build a zero-leakage LOPO training seed directly from a knot-free [`DatasetQcReport`] and
    /// raw `(p_raw, is_correct)` placed predictions.
    pub fn from_qc_report(
        qc: &DatasetQcReport,
        peaks_present: bool,
        raw_predictions: Vec<(f64, bool)>,
    ) -> Self {
        Self {
            n_assignable_amides: qc.n_assignable_amides,
            n_amide_hsqc_roots: qc.n_amide_hsqc_roots,
            peaks_present,
            hsqc_3d_support_frac: qc.hsqc_3d_support_frac,
            supported_roots_per_assignable: qc.supported_roots_per_assignable,
            aliphatic_support_frac: qc.aliphatic_support_frac,
            resolved_bidir_aliphatic_frac: qc.resolved_bidir_aliphatic_frac,
            hsqc_multiplicity_ratio: qc.hsqc_multiplicity_ratio,
            hsqc_3d_rms_dh_ppm: qc.hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm: qc.hsqc_3d_rms_dn_ppm,
            offset_consistency_ok: qc.offset_consistency_ok,
            weak_large_offset: has_weak_likelihood_large_offset(
                &qc.detailed_offsets,
                QC_MIN_LARGE_OFFSET_GAIN_PER_OBS,
            )
            .is_some(),
            connectivity_frac: qc.connectivity_frac,
            corroborated_connectivity_frac: qc.corroborated_connectivity_frac,
            chain_corroborated_frac: qc.chain_corroborated_frac,
            unary_specificity: qc.unary_specificity,
            raw_predictions,
        }
    }
}

#[inline]
fn eval_precached_seed_with_gate(
    seed: &ProteinHcTrainingSeed,
    cals: &[f64],
    solution_margin: f64,
    gate_cfg: &ProteinHcGateConfig,
) -> Vec<(f64, bool)> {
    let n_ass = seed.raw_predictions.len();
    if n_ass == 0 {
        return Vec::new();
    }

    let is_abstain = seed.n_amide_hsqc_roots == 0
        || (seed.n_amide_hsqc_roots > 0 && seed.hsqc_3d_support_frac < 0.25)
        || seed.aliphatic_support_frac < 0.20
        || (n_ass >= 6 && seed.connectivity_frac < 0.15)
        || is_unconverged_solution_margin(n_ass, solution_margin, seed.hsqc_multiplicity_ratio)
        || ((seed.hsqc_3d_rms_dh_ppm > 0.065 || seed.hsqc_3d_rms_dn_ppm > 0.65)
            && seed.hsqc_3d_support_frac < 0.45)
        || (seed.n_amide_hsqc_roots > 0
            && seed.hsqc_3d_support_frac < 0.45
            && seed.supported_roots_per_assignable < 0.35
            && solution_margin < 0.20);
    if is_abstain {
        return cals
            .iter()
            .zip(seed.raw_predictions.iter())
            .map(|(&c, &(_, ok))| (c.clamp(0.0, 0.45), ok))
            .collect();
    }

    let mut scale = 1.0_f64;
    let mut warned = false;
    if seed.hsqc_multiplicity_ratio > 1.45
        && (!seed.peaks_present || seed.supported_roots_per_assignable > 1.35)
    {
        let pen = ((seed.hsqc_multiplicity_ratio - 1.45) * 0.20)
            .clamp(gate_cfg.multiplicity_penalty_floor, 0.22);
        scale *= 1.0 - pen;
        warned = true;
    }
    if seed.hsqc_3d_support_frac < 0.55 && seed.supported_roots_per_assignable < 0.80 {
        let pen = ((0.55 - seed.hsqc_3d_support_frac) * 0.45).clamp(0.06, 0.18);
        scale *= 1.0 - pen;
        warned = true;
    }
    if seed.aliphatic_support_frac < 0.50 {
        scale *= 0.90;
        warned = true;
    }
    if seed.hsqc_3d_rms_dh_ppm > 0.028 || seed.hsqc_3d_rms_dn_ppm > 0.26 {
        scale *= 0.88;
        warned = true;
    }
    if !seed.offset_consistency_ok || seed.weak_large_offset {
        scale *= 0.78;
        warned = true;
    }
    if n_ass >= 8 && seed.connectivity_frac < 0.38 {
        let pen = ((0.38 - seed.connectivity_frac) * 0.55).clamp(0.06, 0.16);
        scale *= 1.0 - pen;
        warned = true;
    }
    if n_ass >= 8 && seed.corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity {
        scale *= 0.90;
        warned = true;
    }
    let applied_scale = if !warned {
        1.0
    } else {
        scale.clamp(0.65, 0.92)
    };

    let scaled: Vec<f64> = cals
        .iter()
        .map(|&c| (c * applied_scale).clamp(0.0, 1.0))
        .collect();
    let hc_cnt = scaled.iter().filter(|&&c| c >= 0.80).count();
    let hc_frac = (hc_cnt as f64) / (n_ass as f64);

    let trigger_hc_cap = (n_ass >= 12
        && ((hc_cnt > 0
            && hc_frac < gate_cfg.min_hc_call_fraction
            && seed.corroborated_connectivity_frac < 0.75)
            || seed.corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity
            || seed.chain_corroborated_frac < gate_cfg.min_chain_corroborated_frac
            || seed.unary_specificity < gate_cfg.min_unary_specificity
            || (seed.hsqc_3d_support_frac * seed.resolved_bidir_aliphatic_frac)
                < gate_cfg.min_bidir_aliphatic_frac
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
    eval_precached_seed_with_gate(seed, &cals, solution_margin, gate_cfg)
}

/// Borrowed `(training_proteins, calibration_knots)` pair for one solver engine in
/// [`fit_protein_hc_gate_config_multi_engine`].
pub type EngineTrainingSlice<'a> = (&'a [Vec<ProteinHcTrainingSeed>], &'a [(f64, f64)]);

/// Evaluate `(mean_pooled_ece_10, per_engine_micro_hcp, total_capped_seeds)` for a candidate
/// [`ProteinHcGateConfig`] across `engines` (`P1-7` / `13.1:P1-7b`).
pub fn evaluate_protein_hc_gate_objective_multi_engine(
    engines: &[EngineTrainingSlice<'_>],
    cand: &ProteinHcGateConfig,
) -> (f64, Vec<f64>, usize) {
    if engines.is_empty() {
        return (0.0, Vec::new(), 0);
    }
    let mut ece_sum = 0.0_f64;
    let mut hcps = Vec::with_capacity(engines.len());
    let mut n_capped = 0usize;
    for &(train_proteins, knots) in engines {
        let mut pool = Vec::new();
        for seeds in train_proteins {
            for s in seeds {
                if s.raw_predictions.is_empty() {
                    continue;
                }
                let uncapped = evaluate_seed_predictions_with_gate(
                    s,
                    knots,
                    &ProteinHcGateConfig {
                        min_corroborated_connectivity: 0.0,
                        min_hc_call_fraction: 0.0,
                        min_crowded_solution_margin: 0.0,
                        multiplicity_penalty_floor: cand.multiplicity_penalty_floor,
                        min_chain_corroborated_frac: 0.0,
                        min_unary_specificity: 0.0,
                        min_bidir_aliphatic_frac: 0.0,
                    },
                );
                let gated = evaluate_seed_predictions_with_gate(s, knots, cand);
                let had_hc_before = uncapped.iter().any(|&(c, _)| c >= 0.80);
                let has_hc_after = gated.iter().any(|&(c, _)| c >= 0.80);
                if had_hc_before && !has_hc_after {
                    n_capped += 1;
                }
                pool.extend(gated);
            }
        }
        ece_sum += compute_grid_ece(&pool, UNIFORM_CALIBRATION_GRID_10);
        let hc_tot = pool.iter().filter(|&&(c, _)| c >= 0.80).count();
        let hc_ok = pool.iter().filter(|&&(c, ok)| c >= 0.80 && ok).count();
        hcps.push(if hc_tot > 0 {
            (hc_ok as f64) / (hc_tot as f64)
        } else {
            1.0
        });
    }
    (ece_sum / (engines.len() as f64), hcps, n_capped)
}

#[derive(Clone, Copy, Default)]
struct Bin10Hist {
    counts: [usize; 10],
    corrects: [usize; 10],
    conf_sums: [f64; 10],
    total: usize,
    hc_tot: usize,
    hc_ok: usize,
}

impl Bin10Hist {
    #[inline]
    fn from_preds(preds: &[(f64, bool)]) -> Self {
        let mut h = Self {
            total: preds.len(),
            ..Self::default()
        };
        for &(raw_conf, is_ok) in preds {
            let c = raw_conf.clamp(0.0, 1.0);
            let idx = if c >= 0.9 {
                9
            } else {
                ((c * 10.0).floor() as usize).min(9)
            };
            h.counts[idx] += 1;
            h.conf_sums[idx] += c;
            if is_ok {
                h.corrects[idx] += 1;
            }
            if c >= 0.80 {
                h.hc_tot += 1;
                if is_ok {
                    h.hc_ok += 1;
                }
            }
        }
        h
    }

    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.total += other.total;
        self.hc_tot += other.hc_tot;
        self.hc_ok += other.hc_ok;
        for b in 0..10 {
            self.counts[b] += other.counts[b];
            self.corrects[b] += other.corrects[b];
            self.conf_sums[b] += other.conf_sums[b];
        }
    }

    #[inline]
    fn ece_10(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let tot = self.total as f64;
        let mut ece = 0.0_f64;
        for b in 0..10 {
            let cnt = self.counts[b];
            if cnt > 0 {
                let mc = self.conf_sums[b] / (cnt as f64);
                let ea = (self.corrects[b] as f64) / (cnt as f64);
                ece += ((cnt as f64) / tot) * (mc - ea).abs();
            }
        }
        ece
    }
}

/// Fit [`ProteinHcGateConfig`] across `engines` over caller-supplied 7D candidate axes and
/// Occam regularization weight `occam_reg_weight` (`1e-4` in [`fit_protein_hc_gate_config_multi_engine`],
/// or `0.0` for unconstrained floor diagnostics; `13.1:P1-7b`).
///
/// Inside each training split, optimizes:
/// 1. Mean pooled 10-bin `ECE_10` across engines,
/// 2. Per-engine pooled micro-HCP `>= 94.5%` hinge (`5.0 * max(0, 0.945 - micro_hcp)`),
/// 3. Chow (1970) per-protein selective-risk hinge on training proteins emitting `< 85%` HC precision
///    (`0.35 * max(0, 0.85 - prot_hcp)` when `prot_hc_tot > 0`), while strictly protecting clean
///    training proteins whose baseline per-protein HC precision is `>= 94%` (`0.60 * capped_seed_frac`),
/// 4. Neutral monotone Occam tie-breaker (`occam_reg_weight * sum_of_thresholds`).
#[allow(clippy::too_many_arguments)]
pub fn fit_protein_hc_gate_config_7d_with_grids(
    engines: &[EngineTrainingSlice<'_>],
    grid_c35: &[f64],
    grid_hcf: &[f64],
    grid_cmarg: &[f64],
    grid_mpen: &[f64],
    grid_chain: &[f64],
    grid_uspec: &[f64],
    grid_bidir: &[f64],
    occam_reg_weight: f64,
) -> (ProteinHcGateConfig, f64) {
    struct CachedSeed<'a> {
        seed: &'a ProteinHcTrainingSeed,
        cals: Vec<f64>,
        solution_margin: f64,
        is_abstain: bool,
    }
    struct CachedProt<'a> {
        seeds: Vec<CachedSeed<'a>>,
        clean_94: bool,
    }
    struct PrecompSeed<'a> {
        seed: &'a ProteinHcTrainingSeed,
        solution_margin: f64,
        can_extra_cap: bool,
        had_raw_hc: bool,
        base_capped: bool,
        hc_frac: f64,
        hist_uncapped: Bin10Hist,
        hist_capped: Bin10Hist,
    }

    let zero_floor_gate = ProteinHcGateConfig {
        min_corroborated_connectivity: 0.0,
        min_hc_call_fraction: 0.0,
        min_crowded_solution_margin: 0.0,
        multiplicity_penalty_floor: grid_mpen.first().copied().unwrap_or(0.0),
        min_chain_corroborated_frac: 0.0,
        min_unary_specificity: 0.0,
        min_bidir_aliphatic_frac: 0.0,
    };

    let mut cached_engines: Vec<Vec<CachedProt<'_>>> = Vec::with_capacity(engines.len());
    for &(train_proteins, knots) in engines {
        let mut engine_prots = Vec::new();
        for seeds in train_proteins {
            let mut p_seeds = Vec::new();
            let mut base_hc_tot = 0usize;
            let mut base_hc_ok = 0usize;
            for s in seeds {
                if s.raw_predictions.is_empty() {
                    continue;
                }
                let n_ass = s.raw_predictions.len();
                let cals: Vec<f64> = s
                    .raw_predictions
                    .iter()
                    .map(|&(p, _)| eval_monotone_calibration_map(p, knots))
                    .collect();
                let solution_margin = (cals
                    .iter()
                    .filter(|&&c| c >= QC_STRONG_BELIEF_THRESHOLD)
                    .count() as f64)
                    / (n_ass as f64);
                let is_abstain = s.n_amide_hsqc_roots == 0
                    || (s.n_amide_hsqc_roots > 0 && s.hsqc_3d_support_frac < 0.25)
                    || s.aliphatic_support_frac < 0.20
                    || (n_ass >= 6 && s.connectivity_frac < 0.15)
                    || is_unconverged_solution_margin(
                        n_ass,
                        solution_margin,
                        s.hsqc_multiplicity_ratio,
                    )
                    || ((s.hsqc_3d_rms_dh_ppm > 0.065 || s.hsqc_3d_rms_dn_ppm > 0.65)
                        && s.hsqc_3d_support_frac < 0.45)
                    || (s.n_amide_hsqc_roots > 0
                        && s.hsqc_3d_support_frac < 0.45
                        && s.supported_roots_per_assignable < 0.35
                        && solution_margin < 0.20);
                let base_preds =
                    eval_precached_seed_with_gate(s, &cals, solution_margin, &zero_floor_gate);
                for &(c, ok) in &base_preds {
                    if c >= 0.80 {
                        base_hc_tot += 1;
                        if ok {
                            base_hc_ok += 1;
                        }
                    }
                }
                p_seeds.push(CachedSeed {
                    seed: s,
                    cals,
                    solution_margin,
                    is_abstain,
                });
            }
            if !p_seeds.is_empty() {
                let clean_94 =
                    base_hc_tot >= 1 && (base_hc_ok as f64) >= 0.94 * (base_hc_tot as f64);
                engine_prots.push(CachedProt {
                    seeds: p_seeds,
                    clean_94,
                });
            }
        }
        if !engine_prots.is_empty() {
            cached_engines.push(engine_prots);
        }
    }
    if cached_engines.is_empty() {
        return (ProteinHcGateConfig::least_restrictive_grid_corner(), 0.0);
    }

    let n_eng = cached_engines.len() as f64;
    let mut best_cfg = ProteinHcGateConfig::least_restrictive_grid_corner();
    let mut best_obj = f64::INFINITY;

    for &min_c35 in grid_c35 {
        for &mpen in grid_mpen {
            // Precompute exact uncapped/capped 10-bin histograms once per `(min_c35, mpen)` pair
            let mut pre_engines: Vec<Vec<(bool, Vec<PrecompSeed<'_>>)>> =
                Vec::with_capacity(cached_engines.len());
            for eng_prots in &cached_engines {
                let mut p_vec = Vec::with_capacity(eng_prots.len());
                for cp in eng_prots {
                    let mut s_vec = Vec::with_capacity(cp.seeds.len());
                    for cs in &cp.seeds {
                        let s = cs.seed;
                        let n_ass = s.raw_predictions.len();
                        if cs.is_abstain {
                            let preds: Vec<(f64, bool)> = cs
                                .cals
                                .iter()
                                .zip(s.raw_predictions.iter())
                                .map(|(&c, &(_, ok))| (c.clamp(0.0, 0.45), ok))
                                .collect();
                            let h = Bin10Hist::from_preds(&preds);
                            s_vec.push(PrecompSeed {
                                seed: s,
                                solution_margin: cs.solution_margin,
                                can_extra_cap: false,
                                had_raw_hc: false,
                                base_capped: false,
                                hc_frac: 0.0,
                                hist_uncapped: h,
                                hist_capped: h,
                            });
                            continue;
                        }
                        let mut scale = 1.0_f64;
                        let mut warned = false;
                        if s.hsqc_multiplicity_ratio > 1.45
                            && (!s.peaks_present || s.supported_roots_per_assignable > 1.35)
                        {
                            let pen = ((s.hsqc_multiplicity_ratio - 1.45) * 0.20).clamp(mpen, 0.22);
                            scale *= 1.0 - pen;
                            warned = true;
                        }
                        if s.hsqc_3d_support_frac < 0.55 && s.supported_roots_per_assignable < 0.80
                        {
                            let pen = ((0.55 - s.hsqc_3d_support_frac) * 0.45).clamp(0.06, 0.18);
                            scale *= 1.0 - pen;
                            warned = true;
                        }
                        if s.aliphatic_support_frac < 0.50 {
                            scale *= 0.90;
                            warned = true;
                        }
                        if s.hsqc_3d_rms_dh_ppm > 0.028 || s.hsqc_3d_rms_dn_ppm > 0.26 {
                            scale *= 0.88;
                            warned = true;
                        }
                        if !s.offset_consistency_ok || s.weak_large_offset {
                            scale *= 0.78;
                            warned = true;
                        }
                        if n_ass >= 8 && s.connectivity_frac < 0.38 {
                            let pen = ((0.38 - s.connectivity_frac) * 0.55).clamp(0.06, 0.16);
                            scale *= 1.0 - pen;
                            warned = true;
                        }
                        if n_ass >= 8 && s.corroborated_connectivity_frac < min_c35 {
                            scale *= 0.90;
                            warned = true;
                        }
                        let applied_scale = if !warned {
                            1.0
                        } else {
                            scale.clamp(0.65, 0.92)
                        };
                        let scaled_preds: Vec<(f64, bool)> = cs
                            .cals
                            .iter()
                            .zip(s.raw_predictions.iter())
                            .map(|(&c, &(_, ok))| ((c * applied_scale).clamp(0.0, 1.0), ok))
                            .collect();
                        let hc_cnt = scaled_preds.iter().filter(|&&(c, _)| c >= 0.80).count();
                        let hc_frac = (hc_cnt as f64) / (n_ass as f64);
                        let capped_preds: Vec<(f64, bool)> = scaled_preds
                            .iter()
                            .map(|&(c, ok)| (c.clamp(0.0, QC_PROTEIN_HC_CONFIDENCE_CAP), ok))
                            .collect();
                        let hist_capped = Bin10Hist::from_preds(&capped_preds);
                        let base_cap = (n_ass >= 12
                            && (s.corroborated_connectivity_frac < min_c35
                                || ((!s.offset_consistency_ok || s.weak_large_offset)
                                    && cs.solution_margin < 0.50)))
                            || (hc_cnt > 0 && n_ass * 5 < s.n_assignable_amides);
                        let hist_uncapped = if base_cap {
                            hist_capped
                        } else {
                            Bin10Hist::from_preds(&scaled_preds)
                        };
                        s_vec.push(PrecompSeed {
                            seed: s,
                            solution_margin: cs.solution_margin,
                            can_extra_cap: !base_cap && n_ass >= 12,
                            had_raw_hc: hc_cnt > 0,
                            base_capped: base_cap && hc_cnt > 0,
                            hc_frac,
                            hist_uncapped,
                            hist_capped,
                        });
                    }
                    p_vec.push((cp.clean_94, s_vec));
                }
                pre_engines.push(p_vec);
            }

            for &min_hcf in grid_hcf {
                for &min_cmarg in grid_cmarg {
                    for &min_chain in grid_chain {
                        for &min_uspec in grid_uspec {
                            for &min_bidir in grid_bidir {
                                let mut ece_sum = 0.0_f64;
                                let mut pen_sum = 0.0_f64;
                                let mut sel_sum = 0.0_f64;

                                for eng_prots in &pre_engines {
                                    let mut pool_hist = Bin10Hist::default();
                                    for (clean_94, p_seeds) in eng_prots {
                                        let mut prot_hc_tot = 0usize;
                                        let mut prot_hc_ok = 0usize;
                                        let mut prot_capped_hc_seeds = 0usize;
                                        for ps in p_seeds {
                                            let extra_cap = ps.can_extra_cap
                                                && ((ps.had_raw_hc
                                                    && ps.hc_frac < min_hcf
                                                    && ps.seed.corroborated_connectivity_frac
                                                        < 0.75)
                                                    || ps.seed.chain_corroborated_frac < min_chain
                                                    || ps.seed.unary_specificity < min_uspec
                                                    || (ps.seed.hsqc_3d_support_frac
                                                        * ps.seed.resolved_bidir_aliphatic_frac)
                                                        < min_bidir
                                                    || (ps.seed.supported_roots_per_assignable
                                                        > 1.60
                                                        && ps.solution_margin < min_cmarg));
                                            if extra_cap {
                                                pool_hist.add_assign(&ps.hist_capped);
                                                if ps.had_raw_hc {
                                                    prot_capped_hc_seeds += 1;
                                                }
                                            } else {
                                                pool_hist.add_assign(&ps.hist_uncapped);
                                                prot_hc_tot += ps.hist_uncapped.hc_tot;
                                                prot_hc_ok += ps.hist_uncapped.hc_ok;
                                                if ps.base_capped {
                                                    prot_capped_hc_seeds += 1;
                                                }
                                            }
                                        }
                                        if prot_hc_tot > 0 {
                                            let excess_false_hc = (0.85 * (prot_hc_tot as f64)
                                                - (prot_hc_ok as f64))
                                                .max(0.0);
                                            if excess_false_hc > 0.0 {
                                                sel_sum += 0.35 * (excess_false_hc / 20.0);
                                            }
                                        }
                                        if *clean_94 && prot_capped_hc_seeds > 0 {
                                            sel_sum += 0.60 * (prot_capped_hc_seeds as f64)
                                                / (p_seeds.len().max(1) as f64);
                                        }
                                    }
                                    ece_sum += pool_hist.ece_10();
                                    let hcp = if pool_hist.hc_tot > 0 {
                                        (pool_hist.hc_ok as f64) / (pool_hist.hc_tot as f64)
                                    } else {
                                        1.0
                                    };
                                    if hcp < 0.945 {
                                        pen_sum += (0.945 - hcp) * 5.0;
                                    }
                                }

                                let reg = occam_reg_weight
                                    * (min_c35
                                        + min_hcf
                                        + min_cmarg
                                        + mpen
                                        + min_chain
                                        + min_uspec
                                        + min_bidir);
                                let obj = ((ece_sum + sel_sum) / n_eng) + pen_sum + reg;
                                if obj < best_obj {
                                    best_obj = obj;
                                    best_cfg = ProteinHcGateConfig {
                                        min_corroborated_connectivity: min_c35,
                                        min_hc_call_fraction: min_hcf,
                                        min_crowded_solution_margin: min_cmarg,
                                        multiplicity_penalty_floor: mpen,
                                        min_chain_corroborated_frac: min_chain,
                                        min_unary_specificity: min_uspec,
                                        min_bidir_aliphatic_frac: min_bidir,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (best_cfg, best_obj)
}

/// Fit [`ProteinHcGateConfig`] across `engines` over caller-supplied 4D legacy axes plus the
/// canonical `13.1:P1-7b` structural axes (`CANDIDATE_GATE_CHAIN_CORROBORATED_FRAC`,
/// `CANDIDATE_GATE_UNARY_SPECIFICITY`, `CANDIDATE_GATE_BIDIR_ALIPHATIC_FRAC`).
pub fn fit_protein_hc_gate_config_with_grids(
    engines: &[EngineTrainingSlice<'_>],
    grid_c35: &[f64],
    grid_hcf: &[f64],
    grid_cmarg: &[f64],
    grid_mpen: &[f64],
    occam_reg_weight: f64,
) -> (ProteinHcGateConfig, f64) {
    fit_protein_hc_gate_config_7d_with_grids(
        engines,
        grid_c35,
        grid_hcf,
        grid_cmarg,
        grid_mpen,
        CANDIDATE_GATE_CHAIN_CORROBORATED_FRAC,
        CANDIDATE_GATE_UNARY_SPECIFICITY,
        CANDIDATE_GATE_BIDIR_ALIPHATIC_FRAC,
        occam_reg_weight,
    )
}

/// Fit a single shared [`ProteinHcGateConfig`] strictly on training proteins across one or more
/// solver engines (`engines`, each `(train_proteins, knots)`) over the `CANDIDATE_GATE_*` product
/// grid (`13.1:P1-7b`).
pub fn fit_protein_hc_gate_config_multi_engine(
    engines: &[EngineTrainingSlice<'_>],
) -> ProteinHcGateConfig {
    fit_protein_hc_gate_config_7d_with_grids(
        engines,
        CANDIDATE_GATE_CORROBORATED_CONNECTIVITY,
        CANDIDATE_GATE_HC_CALL_FRACTION,
        CANDIDATE_GATE_CROWDED_SOLUTION_MARGIN,
        CANDIDATE_GATE_MULTIPLICITY_PENALTY_FLOOR,
        CANDIDATE_GATE_CHAIN_CORROBORATED_FRAC,
        CANDIDATE_GATE_UNARY_SPECIFICITY,
        CANDIDATE_GATE_BIDIR_ALIPHATIC_FRAC,
        1e-4,
    )
    .0
}

/// Fit [`ProteinHcGateConfig`] strictly on training proteins (`train_proteins`, each a slice of
/// [`ProteinHcTrainingSeed`] seeds) for a single engine by delegating to
/// [`fit_protein_hc_gate_config_multi_engine`] (`P1-7`).
pub fn fit_protein_hc_gate_config(
    train_proteins: &[Vec<ProteinHcTrainingSeed>],
    knots: &[(f64, f64)],
) -> ProteinHcGateConfig {
    fit_protein_hc_gate_config_multi_engine(&[(train_proteins, knots)])
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
        has_weak_likelihood_large_offset(&detailed_offsets, QC_MIN_LARGE_OFFSET_GAIN_PER_OBS);

    // Sequential `(i, i-1)` link coverage (`P >= 1.0`, corroborated `P >= 3.5`, and contiguous
    // `>= 4.min(seg_len)` chain corroboration `13.1:P1-7b`)
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

    let has_resolved_bidir_aliphatic = |ss: &SpinSystem| -> bool {
        for nuc in ["CA", "CB"] {
            let Some(c0) = ss.cands.get(&(nuc.to_string(), 0)) else {
                continue;
            };
            let Some(cm1) = ss.cands.get(&(nuc.to_string(), -1)) else {
                continue;
            };
            let resolved = c0.iter().any(|&(v0, sup0)| {
                cm1.iter().any(|&(vm1, supm1)| {
                    supm1 > 0
                        && ((sup0 > 0 && (v0 - vm1).abs() > 0.18) || (sup0 == 0 && supm1 >= 2))
                })
            });
            if resolved {
                return true;
            }
        }
        false
    };

    let resolved_bidir_aliphatic_frac = if systems.is_empty() {
        1.0
    } else if assigned_count > 0 {
        let n_bidir = result
            .position
            .iter()
            .enumerate()
            .filter(|&(s, &pos)| {
                pos >= 0
                    && (pos as usize) < k_len
                    && s < systems.len()
                    && has_resolved_bidir_aliphatic(&systems[s])
            })
            .count();
        (n_bidir as f64) / (assigned_count as f64)
    } else {
        let n_carbon_backed = systems
            .iter()
            .filter(|ss| ss.has_observed_carbons())
            .count();
        if n_carbon_backed == 0 {
            0.0
        } else {
            let n_bidir = systems
                .iter()
                .filter(|ss| ss.has_observed_carbons() && has_resolved_bidir_aliphatic(ss))
                .count();
            (n_bidir as f64) / (n_carbon_backed as f64)
        }
    };

    let (connectivity_frac, corroborated_connectivity_frac, chain_corroborated_frac) =
        if assigned_count == 0 || systems.is_empty() {
            (0.0, 0.0, 0.0)
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

            // Contiguous corroborated chain coverage (`>= 4.min(seg_len)` consecutive placed residues
            // joined by `P >= 3.5` links inside each non-Proline sequence segment; `13.1:P1-7b`)
            let mut chain_assigned = 0usize;
            let mut seg_start = 1usize;
            while seg_start < k_len {
                if norm_seq[seg_start] == "PRO" {
                    seg_start += 1;
                    continue;
                }
                let mut seg_end = seg_start;
                while seg_end + 1 < k_len && norm_seq[seg_end + 1] != "PRO" {
                    seg_end += 1;
                }
                let seg_len = seg_end - seg_start + 1;
                if seg_len >= 2 {
                    let min_chain_res = 4usize.min(seg_len);
                    let mut run_edges = 0usize;
                    for k in seg_start..seg_end {
                        let edge_ok = at[k] >= 0
                            && at[k + 1] >= 0
                            && p_mat.get(at[k] as usize, at[k + 1] as usize)
                                >= QC_CORROBORATED_PAIR_THRESHOLD;
                        if edge_ok {
                            run_edges += 1;
                        } else {
                            if run_edges + 1 >= min_chain_res {
                                chain_assigned += run_edges + 1;
                            }
                            run_edges = 0;
                        }
                    }
                    if run_edges + 1 >= min_chain_res {
                        chain_assigned += run_edges + 1;
                    }
                }
                seg_start = seg_end + 1;
            }

            if linkable_assigned == 0 {
                (1.0, 1.0, 1.0)
            } else {
                (
                    (linked_assigned as f64) / (linkable_assigned as f64),
                    (corr_linked_assigned as f64) / (linkable_assigned as f64),
                    ((chain_assigned as f64) / (linkable_assigned as f64)).clamp(0.0, 1.0),
                )
            }
        };

    // Mean Bayesian unary-vs-random-slot specificity probability `1 / (1 + exp(-ΔU(s, k)))` (`13.1:P1-7b`)
    let unary_specificity = if assigned_count == 0 || systems.is_empty() || k_len <= 1 {
        0.0
    } else {
        let mut u_cfg = score_cfg.clone();
        u_cfg.auto_reference = false;
        let shifted_priors = priors.with_offsets(&detailed_offsets.to_shift_offsets());
        let u_mat = unary_matrix(systems, &norm_seq, &shifted_priors, &u_cfg);
        let mut spec_sum = 0.0_f64;
        let mut spec_n = 0usize;
        for (s, &pos) in result.position.iter().enumerate() {
            if pos <= 0 || (pos as usize) >= k_len || s >= systems.len() {
                continue;
            }
            let k_assigned = pos as usize;
            let u_sk = u_mat.get(s, k_assigned);
            if u_sk <= -1e5 {
                spec_sum += 0.0;
                spec_n += 1;
                continue;
            }
            let mut max_u = f64::NEG_INFINITY;
            let mut n_adm = 0usize;
            for (kp, res) in norm_seq.iter().enumerate().skip(1) {
                if res == "PRO" {
                    continue;
                }
                let v = u_mat.get(s, kp);
                if v > -1e5 {
                    if v > max_u {
                        max_u = v;
                    }
                    n_adm += 1;
                }
            }
            if n_adm == 0 || !max_u.is_finite() {
                spec_sum += 0.5;
            } else {
                let mut sum_exp = 0.0_f64;
                for (kp, res) in norm_seq.iter().enumerate().skip(1) {
                    if res == "PRO" {
                        continue;
                    }
                    let v = u_mat.get(s, kp);
                    if v > -1e5 {
                        sum_exp += (v - max_u).exp();
                    }
                }
                let log_mean = max_u + (sum_exp / (n_adm as f64)).max(1e-30).ln();
                let delta_u = (u_sk - log_mean).clamp(-20.0, 20.0);
                spec_sum += 1.0 / (1.0 + (-delta_u).exp());
            }
            spec_n += 1;
        }
        if spec_n == 0 {
            1.0
        } else {
            (spec_sum / (spec_n as f64)).clamp(0.0, 1.0)
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
            resolved_bidir_aliphatic_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            chain_corroborated_frac,
            unary_specificity,
            solution_margin,
            hc_call_fraction: 0.0,
            protein_hc_gate_triggered: false,
            truncated_fid_zero_filled: false,
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

    // Protein-level HC cap gate (`P1-7` / `13.1:P1-7b`): suppress unreliable `>= 0.80` tail calls
    // (`cap = 0.79`) when label-free global evidence shows unconverged corroborated links, broken
    // contiguous sequential chains, low unary specificity, unidirectional strip starvation, isolated
    // sparse HC spikes, severe multi-conformer crowding with low global margin, or suspect offsets.
    let mut hc_gate_reasons = Vec::new();
    if assigned_count >= 12 {
        if hc_count > 0
            && hc_call_fraction < gate_cfg.min_hc_call_fraction
            && corroborated_connectivity_frac < 0.75
        {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[sparse_hc_tail]: isolated HC tail ({}/{} = {:.1}% < {:.1}% with corroborated links {:.1}% < 75.0%)",
                hc_count,
                assigned_count,
                hc_call_fraction * 100.0,
                gate_cfg.min_hc_call_fraction * 100.0,
                corroborated_connectivity_frac * 100.0,
            ));
        }
        if corroborated_connectivity_frac < gate_cfg.min_corroborated_connectivity {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[low_corroborated_connectivity]: corroborated links {:.1}% < {:.1}%",
                corroborated_connectivity_frac * 100.0,
                gate_cfg.min_corroborated_connectivity * 100.0
            ));
        }
        if chain_corroborated_frac < gate_cfg.min_chain_corroborated_frac {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[low_chain_corroborated_frac]: contiguous >=4-residue corroborated chains {:.1}% < {:.1}%",
                chain_corroborated_frac * 100.0,
                gate_cfg.min_chain_corroborated_frac * 100.0
            ));
        }
        if unary_specificity < gate_cfg.min_unary_specificity {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[low_unary_specificity]: mean unary specificity {:.1}% < {:.1}%",
                unary_specificity * 100.0,
                gate_cfg.min_unary_specificity * 100.0
            ));
        }
        let effective_bidir_3d_support = hsqc_3d_support_frac * resolved_bidir_aliphatic_frac;
        if effective_bidir_3d_support < gate_cfg.min_bidir_aliphatic_frac {
            hc_gate_reasons.push(format!(
                "protein_hc_gate[low_bidir_aliphatic_frac]: effective 3D bidirectional (i, i-1) aliphatic support {:.1}% < {:.1}% (sup3d={:.1}%, bidir={:.1}%)",
                effective_bidir_3d_support * 100.0,
                gate_cfg.min_bidir_aliphatic_frac * 100.0,
                hsqc_3d_support_frac * 100.0,
                resolved_bidir_aliphatic_frac * 100.0,
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
    }
    if hc_count > 0 && assigned_count * 5 < n_assignable_amides {
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
            resolved_bidir_aliphatic_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            chain_corroborated_frac,
            unary_specificity,
            solution_margin,
            hc_call_fraction,
            protein_hc_gate_triggered: false,
            truncated_fid_zero_filled: false,
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
            resolved_bidir_aliphatic_frac,
            hsqc_3d_rms_dh_ppm,
            hsqc_3d_rms_dn_ppm,
            hsqc_multiplicity_ratio,
            raw_hsqc_multiplicity_ratio,
            offset_consistency_ok,
            detailed_offsets,
            connectivity_frac,
            corroborated_connectivity_frac,
            chain_corroborated_frac,
            unary_specificity,
            solution_margin,
            hc_call_fraction,
            protein_hc_gate_triggered,
            truncated_fid_zero_filled: false,
            warn_sub_hc_uncalibrated: true,
            confidence_calibrated: true,
            calibration_warning: None,
            reasons: warn_reasons,
        }
    }
}

/// Propagate opt-in `--allow-truncated-fid` zero-fill warnings (`F2`) into `qc` and `result`,
/// guaranteeing `qc.truncated_fid_zero_filled == true` and at least `DatasetQcVerdict::Warn`
/// (`warn_sub_hc_uncalibrated = true`, `0.90` damping when upgrading from `Pass`).
pub fn apply_truncated_fid_qc_warning(
    result: &mut AssignResult,
    qc: &mut DatasetQcReport,
    warnings: &[String],
) {
    if warnings.is_empty() {
        return;
    }
    qc.truncated_fid_zero_filled = true;
    for w in warnings {
        let msg = format!("truncated FID zero-filled (--allow-truncated-fid): {w}");
        if !qc.reasons.contains(&msg) {
            qc.reasons.push(msg);
        }
    }
    if qc.verdict == DatasetQcVerdict::Pass {
        qc.verdict = DatasetQcVerdict::Warn;
        qc.warn_sub_hc_uncalibrated = true;
        qc.confidence_scale = 0.90;
        if qc.already_applied {
            let cap = qc.confidence_cap.clamp(0.0, 1.0);
            for c in &mut result.confidence {
                if *c < 0.999_999 {
                    *c = (*c * 0.90).clamp(0.0, cap);
                }
            }
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

/// Mild conservative conjugate `Beta(alpha_pos = 0.35, beta_neg = 0.65)` prior (`1.0` total
/// pseudocount, prior mean `0.35`, intermediate between `evo-graph`'s symmetric `Beta(0.25, 0.25)`
/// mean `0.50` and CPU `standard`'s `Beta(0.75, 2.25)` mean `0.25`) used for WebGPU `standard`
/// (`runs = 64`, `P1-8`) so sparse upper-intermediate bins (`[0.68, 0.79]`) do not cross the
/// `0.80` HC boundary prematurely.
pub const STANDARD_GPU_PRIOR_ALPHA: f64 = 0.35;

/// Negative pseudocount (`beta_neg = 0.65`) paired with [`STANDARD_GPU_PRIOR_ALPHA`] (`P1-8`).
pub const STANDARD_GPU_PRIOR_BETA: f64 = 0.65;

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

/// Hierarchical empirical-Bayes isotonic PAV fitter with general conjugate `Beta(alpha_pos, beta_neg)`
/// prior (`Q2` / `P1-8`): pools primary `samples` (weight `1.0` per observation) with
/// `anchor_samples` from the matched CPU Hamiltonian (`weight = anchor_weight` per observation)
/// plus `Beta(alpha_pos, beta_neg)` smoothing on `grid`. When `anchor_weight <= 0.0` or
/// `anchor_samples` is empty, reduces identically to [`fit_isotonic_monotone_knots_beta`].
pub fn fit_isotonic_monotone_knots_shrunk_toward_beta(
    samples: &[(f64, bool)],
    anchor_samples: &[(f64, bool)],
    anchor_weight: f64,
    grid: &[f64],
    alpha_pos: f64,
    beta_neg: f64,
) -> Vec<(f64, f64)> {
    if anchor_weight <= 0.0 || anchor_samples.is_empty() {
        return fit_isotonic_monotone_knots_beta(samples, grid, alpha_pos, beta_neg);
    }
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
    let half = 0.5 * laplace_prior.max(1e-6);
    fit_isotonic_monotone_knots_shrunk_toward_beta(
        samples,
        anchor_samples,
        anchor_weight,
        grid,
        half,
        half,
    )
}

/// Conjugate `Beta(alpha_pos = 0.25, beta_neg = 0.25)` prior (`0.5` total pseudocount, matching
/// [`CANONICAL_CALIBRATION_LAPLACE_PRIOR`]) used for CPU `bayes-sa` (`replicas = 32`; `13.4`).
pub const BAYES_SA_CPU_PRIOR_ALPHA: f64 = 0.25;

/// Negative pseudocount (`beta_neg = 0.25`) paired with [`BAYES_SA_CPU_PRIOR_ALPHA`] (`13.4`).
pub const BAYES_SA_CPU_PRIOR_BETA: f64 = 0.25;

/// Mild conservative conjugate `Beta(alpha_pos = 0.35, beta_neg = 0.65)` prior used for WebGPU
/// `bayes-sa` (`replicas = 128` `f32`; `13.4`), matching [`STANDARD_GPU_PRIOR_ALPHA`] / [`STANDARD_GPU_PRIOR_BETA`].
pub const BAYES_SA_GPU_PRIOR_ALPHA: f64 = 0.35;

/// Negative pseudocount (`beta_neg = 0.65`) paired with [`BAYES_SA_GPU_PRIOR_ALPHA`] (`13.4`).
pub const BAYES_SA_GPU_PRIOR_BETA: f64 = 0.65;

/// Generalized nested Leave-One-Protein-Out (LOPO) selector for GPU CPU-anchor shrinkage weight
/// (`P1-8` / `13.4`): runs an inner `M`-fold LOPO over [`CANDIDATE_GPU_SHRINKAGE_WEIGHTS`] with
/// conjugate `Beta(prior_alpha, prior_beta)` PAV smoothing, selecting `w` that minimizes inner OOF
/// pooled 10-bin `ECE_10` subject to inner OOF micro HC-precision non-degradation relative to
/// `w = 0.0` and within `0.25%` of the feasible `94.5%` target.
#[allow(clippy::too_many_arguments)]
pub fn select_gpu_shrinkage_weight_nested_lopo_beta(
    gpu_proteins: &[Vec<ProteinHcTrainingSeed>],
    cpu_raw_by_protein: &[Vec<(f64, bool)>],
    gpu_trainable: &[bool],
    cpu_trainable: &[bool],
    gate_cfg: &ProteinHcGateConfig,
    prior_alpha: f64,
    prior_beta: f64,
    default_w: f64,
) -> (f64, Vec<(f64, f64, f64)>) {
    let m = gpu_proteins.len();
    if m < 2 || cpu_raw_by_protein.len() != m {
        return (default_w, Vec::new());
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
            let inner_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &gpu_tr,
                &cpu_tr,
                w,
                CANONICAL_CALIBRATION_GRID_12,
                prior_alpha,
                prior_beta,
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
        .unwrap_or(default_w);

    (best_w, stats)
}

/// Nested Leave-One-Protein-Out (LOPO) selector for `STANDARD_GPU` CPU-anchor shrinkage weight (`P1-8`).
pub fn select_gpu_shrinkage_weight_nested_lopo(
    gpu_proteins: &[Vec<ProteinHcTrainingSeed>],
    cpu_raw_by_protein: &[Vec<(f64, bool)>],
    gpu_trainable: &[bool],
    cpu_trainable: &[bool],
    gate_cfg: &ProteinHcGateConfig,
) -> (f64, Vec<(f64, f64, f64)>) {
    select_gpu_shrinkage_weight_nested_lopo_beta(
        gpu_proteins,
        cpu_raw_by_protein,
        gpu_trainable,
        cpu_trainable,
        gate_cfg,
        STANDARD_GPU_PRIOR_ALPHA,
        STANDARD_GPU_PRIOR_BETA,
        CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
    )
}

/// Canonical empirical-Bayes shrinkage weight (`w = 0.00`, `P1-8` / `13.1:P1-7b`) selected by
/// nested LOPO across the 26-entry non-cold pool when fitting WebGPU `standard` (`f32`) knots.
pub const CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT: f64 = 0.00;

/// Canonical empirical-Bayes shrinkage weight (`w = 0.20`, `13.4`) pooling CPU `bayes-sa` (`f64`)
/// bin counts with WebGPU `bayes-sa` (`f32`) bin counts before isotonic PAV projection across the
/// 26-entry non-cold pool.
pub const CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT: f64 = 0.20;

/// Monotone calibration knot table for `evo-graph` (`population_size = 64, num_islands = 16`,
/// [`EVO_GRAPH_CALIBRATED_RUNS`]; `P1-2` / `P1-7` / `P1-8` / `13.1:P1-7b`), fitted via isotonic
/// PAV on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) with symmetric `Beta(0.25, 0.25)` smoothing
/// across the 26-entry non-cold pool (`dev` + `seen_p12` + `seen_p13`).
pub const EVO_GRAPH_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0222, 0.0795),
    (0.0444, 0.1591),
    (0.2186, 0.3846),
    (0.3755, 0.4287),
    (0.5117, 0.6317),
    (0.6295, 0.6789),
    (0.7242, 0.7196),
    (0.8001, 0.7771),
    (0.8673, 0.7771),
    (0.9311, 0.8263),
    (0.9572, 0.8263),
    (0.9870, 0.8263),
    (0.9997, 0.9715),
    (1.0000, 0.9715),
];

/// Monotone calibration knot table for CPU `standard` (`runs = 16`, [`STANDARD_CPU_CALIBRATED_RUNS`];
/// `P1-6` / `P1-8` / `13.1:P1-7b`), fitted via conservative conjugate `Beta(0.75, 2.25)` isotonic
/// PAV on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) across the 26-entry non-cold pool.
pub const STANDARD_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0235, 0.0707),
    (0.0470, 0.1415),
    (0.2184, 0.2750),
    (0.3765, 0.3905),
    (0.5126, 0.5243),
    (0.6285, 0.6608),
    (0.7218, 0.7893),
    (0.8021, 0.7893),
    (0.8661, 0.8367),
    (0.9331, 0.9137),
    (0.9600, 0.9137),
    (0.9848, 0.9330),
    (0.9994, 0.9607),
    (1.0000, 0.9607),
];

/// Monotone calibration knot table for WebGPU `standard` (`runs = 64`, [`STANDARD_GPU_CALIBRATED_RUNS`];
/// `P1-8` / `13.1:P1-7b`), fitted on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) across the 26-entry non-cold pool.
pub const STANDARD_GPU_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0231, 0.0643),
    (0.0462, 0.1286),
    (0.2194, 0.3355),
    (0.3753, 0.5861),
    (0.5148, 0.7034),
    (0.6285, 0.7585),
    (0.7193, 0.8868),
    (0.7958, 0.9121),
    (0.8603, 0.9362),
    (0.9182, 0.9682),
    (0.9595, 0.9682),
    (0.9844, 0.9682),
    (0.9993, 0.9682),
    (1.0000, 0.9682),
];

/// Monotone calibration knot table for CPU `bayes-sa` (`replicas = 32`, [`BAYES_SA_CPU_CALIBRATED_RUNS`]; `13.4`),
/// fitted via isotonic PAV on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) across the 26-entry non-cold pool.
pub const BAYES_SA_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0567, 0.1120),
    (0.1135, 0.2240),
    (0.2237, 0.2746),
    (0.3735, 0.4740),
    (0.5156, 0.5498),
    (0.6298, 0.6150),
    (0.7224, 0.7383),
    (0.7933, 0.8298),
    (0.8610, 0.8298),
    (0.9170, 0.8298),
    (0.9576, 0.8672),
    (0.9853, 0.8672),
    (0.9948, 0.9621),
    (1.0000, 0.9621),
];

/// Monotone calibration knot table for WebGPU `bayes-sa` (`replicas = 128`, [`BAYES_SA_GPU_CALIBRATED_RUNS`]; `13.4`),
/// fitted via CPU-anchored isotonic PAV (`w = 0.20`) on [`CANONICAL_CALIBRATION_GRID_12`] (`g12`) across the 26-entry non-cold pool.
pub const BAYES_SA_GPU_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.0000, 0.0000),
    (0.0580, 0.1696),
    (0.1160, 0.3392),
    (0.2217, 0.4679),
    (0.3734, 0.6894),
    (0.5149, 0.6894),
    (0.6312, 0.8027),
    (0.7229, 0.8027),
    (0.7950, 0.8556),
    (0.8605, 0.8556),
    (0.9171, 0.8556),
    (0.9583, 0.8902),
    (0.9846, 0.8902),
    (0.9949, 0.9719),
    (1.0000, 0.9719),
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

/// Map raw physical (`T = 1.0`) CPU `bayes-sa` marginal belief into calibrated probability (`13.4`).
#[inline]
pub fn apply_bayes_sa_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, BAYES_SA_CALIBRATION_KNOTS)
}

/// Map raw physical (`T = 1.0`) WebGPU `bayes-sa` marginal belief into calibrated probability (`13.4`).
#[inline]
pub fn apply_bayes_sa_gpu_calibration_map(p_raw: f64) -> f64 {
    eval_monotone_calibration_map(p_raw, BAYES_SA_GPU_CALIBRATION_KNOTS)
}
