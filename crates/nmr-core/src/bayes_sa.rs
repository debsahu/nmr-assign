//! Clean-room Bayesian pseudo-energy and thermodynamic specific-heat ($C_v(T)$) adaptive
//! simulated annealing solver for protein NMR backbone triple-resonance assignment.
//!
//! # Provenance (100% Clean-Room)
//! Derived strictly from peer-reviewed statistical mechanics and biomolecular NMR publications,
//! using our CC0 empirical BMRB prior corpus (`data/priors/bmrb_priors.json`):
//! 1. **Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J.** (2023).
//!    "Robust automated backbone triple resonance NMR assignments of proteins using
//!    Bayesian-based simulated annealing." *Nature Communications* **14**, 1556.
//!    DOI: `10.1038/s41467-023-37219-z` — joint `(amino_acid × secondary_structure)` $(i, i-1)$
//!    Bayesian shift pseudo-energies, multi-pathway sequential overlap Bayes factors, null/dummy
//!    spin-system states, and ensemble specific-heat adaptive cooling.
//! 2. **Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P.** (1983). "Optimization by Simulated
//!    Annealing." *Science* **220**(4598), 671–680 — thermodynamic correspondence between
//!    combinatorial energy variance and specific heat $C_v(T) = (\langle E^2\rangle - \langle E\rangle^2)/T^2$,
//!    and decelerated cooling across cooperative phase transitions.
//! 3. **Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.** (1995).
//!    *J. Biomol. NMR* **5**, 67–81 & **6**, 135–140; **Spera, S. & Bax, A.** (1991),
//!    *J. Am. Chem. Soc.* **113**, 5490–5492; **MacArthur, M. W. & Thornton, J. M.** (1991),
//!    *J. Mol. Biol.* **218**, 397–412 — covalent primary-sequence neighbor shifts (including
//!    pre-Proline steric/electronic shifts) and secondary-structure chemical-shift displacements.
//! 4. **Zhang, H., Neal, S. & Wishart, D. S.** (2003). *J. Biomol. NMR* **25**, 173–195 —
//!    cross-nucleus chemical-shift referencing consistency.
//!
//! No external source code or external parameter tables were consulted.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::sync::{Barrier, Mutex};

use serde::{Deserialize, Serialize};

use crate::peaks::Peak;
use crate::priors::{
    covalent_atom_offset, delta_covalent, estimate_reference_offsets_detailed, one_to_three,
    sync_detailed_reference_offsets, DetailedReferenceOffsets, Priors, ShiftOffsets,
    COVALENT_COIL_WEIGHT, STANDARD_RESIDUES,
};
use crate::qc::{
    apply_bayes_sa_calibration_map, apply_bayes_sa_gpu_calibration_map,
    apply_dataset_qc_to_assignments, evaluate_dataset_qc, DatasetQcReport,
    BAYES_SA_CPU_CALIBRATED_RUNS, BAYES_SA_GPU_CALIBRATED_RUNS,
};
use crate::rng::Rng;
use crate::solver::{resolve_locked_assignments, AssignResult, Matrix2D, ScoreConfig};
use crate::spinsystems::{build_spin_systems, SpinSystem, Tolerances};

/// Pseudo-energy barrier (`+1e9` nats) representing structurally forbidden placements
/// (N-terminus `k = 0`, `PRO` backbone amide slots, diagonal `s == t` self-links, and hard locks).
pub const FORBID_ENERGY: f64 = 1.0e9;

/// Default parallel Metropolis–Hastings replica count on multi-threaded CPU (`R = 32`).
pub const BAYES_SA_CPU_DEFAULT_REPLICAS: usize = 32;

/// Default parallel Metropolis–Hastings replica count on WebGPU (`R = 128`).
pub const BAYES_SA_GPU_DEFAULT_REPLICAS: usize = 128;

/// Number of chemical-shift channels modeled in `bayes_sa` (`CA`, `CB`, `C`, `HA`, `N`, `H`).
pub const NUM_BAYES_CHANNELS: usize = 6;

/// Number of sequential `(i, i-1)` overlap channels (`CA`, `CB`, `C`, `HA`).
pub const NUM_LINK_CHANNELS: usize = 4;

/// Maximum intra-strip candidate hypotheses packed per `(atom, offset)` slot for CPU/GPU parity.
pub const MAX_BAYES_SLOT_CANDS: usize = 4;

/// Exact normalization constant $8 / (3\pi\sqrt{5})$ for the Student-$t(\nu = 5)$ density.
pub const STUDENT_T5_NORM: f64 = 0.379_943_800_856_852_35;

/// Ratio $\sigma_t / \sigma$ matching core Gaussian curvature for Student-$t(\nu = 5)$.
pub const STUDENT_T5_SIGMA_SCALE: f64 = 0.86;

/// Log-Bayes-factor penalty ($\ln 0.02 \approx -3.912$) when an observed `CB` peak appears on `GLY`.
pub const GLY_OBSERVED_CB_LOG_BF: f64 = -3.912_023_005_428_146;

/// Carrier-center ppm offsets subtracted in `f64` prior to evaluation so `f32` WebGPU shaders
/// retain sub-ppm-ULP precision (`~5e-7 ppm` even on `176 ppm` carbonyl `13C'`).
pub const CHANNEL_CENTERS_PPM: [f64; NUM_BAYES_CHANNELS] = [56.0, 36.0, 176.0, 4.4, 120.0, 8.25];

/// Physical spectral dispersion widths $R_a$ (ppm) defining the uniform heavy-tail outlier floor.
pub const CHANNEL_RANGES_PPM: [f64; NUM_BAYES_CHANNELS] = [36.0, 64.0, 24.0, 3.2, 38.0, 4.5];

/// Channel names ordered identically to [`CHANNEL_CENTERS_PPM`].
pub const BAYES_CHANNEL_ATOMS: [&str; NUM_BAYES_CHANNELS] = ["CA", "CB", "C", "HA", "N", "H"];

/// Three-state backbone conformational basin used in the joint `(AA_type × secondary_structure)`
/// $(i, i-1)$ Bayesian chemical-shift model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConformerClass {
    AlphaHelix = 0,
    BetaStrand = 1,
    Coil = 2,
}

impl ConformerClass {
    pub const ALL: [Self; 3] = [Self::AlphaHelix, Self::BetaStrand, Self::Coil];
}

/// Hard user-locked `(spin_system, seq_index)` backbone assignment constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockedAssignment {
    pub spin_system: usize,
    pub seq_index: usize,
}

impl LockedAssignment {
    #[inline]
    pub const fn new(spin_system: usize, seq_index: usize) -> Self {
        Self {
            spin_system,
            seq_index,
        }
    }
}

impl From<(usize, usize)> for LockedAssignment {
    #[inline]
    fn from((spin_system, seq_index): (usize, usize)) -> Self {
        Self {
            spin_system,
            seq_index,
        }
    }
}

/// Simulated annealing temperature update policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoolingScheduleKind {
    /// Thermodynamic specific-heat $C_v(T_m)$ adaptive cooling (Kirkpatrick et al. 1983;
    /// Bishop et al. 2023): decelerates across the cooperative assignment phase transition and
    /// accelerates in high-$T$ disordered and low-$T$ frozen regimes.
    #[default]
    AdaptiveSpecificHeat,
    /// Fixed geometric cooling $T_{m+1} = T_m (T_{\min} / T_{\text{start}})^{1 / (M - 1)}$
    /// over a fixed number of stages (for controlled thermodynamic ablation benchmarks).
    FixedGeometric,
}

/// Configuration for the Bayesian pseudo-energy and specific-heat adaptive SA solver.
#[derive(Clone, Debug)]
pub struct BayesSaConfig {
    /// Number of parallel Metropolis–Hastings replicas (`32` default on CPU, `128` on WebGPU).
    pub replicas: usize,
    /// Temperature cooling schedule (`AdaptiveSpecificHeat` default).
    pub cooling_schedule: CoolingScheduleKind,
    /// Initial high-temperature disordered starting temperature $T_0$ (in nats).
    pub t_start: f64,
    /// Minimum low-temperature floor $T_{\min}$ (in nats).
    pub t_min: f64,
    /// Base dimensionless cooling step $\Delta \beta_0$ in
    /// $T_{m+1} = T_m \exp(-\Delta\beta_0 / \sqrt{1 + \eta C_v(T_m) / N_{\text{active}}})$.
    pub delta_beta_0: f64,
    /// Dimensionless specific-heat deceleration gain $\eta$.
    pub specific_heat_eta: f64,
    /// MCMC proposals per active sequence slot at each temperature stage (`<= 4,096` per GPU submit).
    pub steps_per_stage_per_active: usize,
    /// Minimum number of temperature stages before post-peak convergence stopping can trigger.
    pub min_stages: usize,
    /// Maximum number of temperature stages $M_{\max}$.
    pub max_stages: usize,
    /// Optional exact total MCMC step budget per replica (distributes remaining steps cleanly
    /// so `AdaptiveSpecificHeat` vs `FixedGeometric` comparisons run at matched total steps).
    pub fixed_total_steps: Option<usize>,
    /// Post-peak thermal specific-heat per active residue threshold ($C_v / N_{\text{active}}$)
    /// for automatic frozen-regime convergence detection.
    pub convergence_cv_tol: f64,
    /// Post-peak acceptance-rate threshold for automatic frozen-regime convergence detection.
    pub convergence_acc_tol: f64,
    /// Maximum contiguous sequence window length $L_{\max}$ (`2..=L_max`) for fragment shift/swap moves.
    pub max_fragment_len: usize,
    /// Top-$M$ state-independent candidate spin systems per slot in symmetric candidate proposals.
    pub candidate_list_size: usize,
    /// Uniform heavy-tail outlier mixture weight $\epsilon_{\text{out}} \in (0, 0.2)$ in `E_type`.
    pub outlier_floor_eps: f64,
    /// Sequential inter-strip matching tolerance $\sigma_{\text{CA}}$ (ppm).
    pub sigma_ca: f64,
    /// Sequential inter-strip matching tolerance $\sigma_{\text{CB}}$ (ppm).
    pub sigma_cb: f64,
    /// Sequential inter-strip matching tolerance $\sigma_{\text{C'}}$ (ppm).
    pub sigma_c: f64,
    /// Sequential inter-strip matching tolerance $\sigma_{\text{HA}}$ (ppm).
    pub sigma_ha: f64,
    /// Per-channel sequential mismatch log-Bayes-factor floor (e.g. `-7.5` nats).
    pub link_floor: f64,
    /// Distinct-experiment corroboration weight $\gamma_{\text{sup}}$ inside multi-candidate slots.
    pub support_bonus: f64,
    /// Optional explicit override for the null/dummy chemical potential $\mu_{\text{null}}$ (`None` = calibrated).
    pub mu_null_override: Option<f64>,
    /// Include amide `(15N, 1H)` root chemical shifts in `E_type(s, k)`.
    pub use_hn: bool,
    /// Automatically estimate and apply global chemical-shift referencing offsets (`estimate_reference_offsets_detailed`).
    pub auto_reference: bool,
    /// Apply minor-conformer, water-artifact, and uncorroborated weak-root penalties.
    pub enable_weak_root_penalty: bool,
    /// Forbid assigning spin systems that have zero observed carbon/HA candidates (`!has_observed_carbons`).
    pub forbid_zero_carbons: bool,
    /// Protect non-Gly/non-post-Pro slots from `CA`-less excess noise roots when `S > N_assignable`.
    pub forbid_ca_less_excess_roots: bool,
    /// Evaluate `DatasetQcReport` and apply conservative QC confidence damping / abstention.
    pub apply_qc_calibration: bool,
    /// Peak-to-root and cross-spectrum carbon clustering tolerances used by `solve_bayes_sa_from_peaks`.
    pub tolerances: Tolerances,
    /// Maximum competing clusters retained per `(atom, offset)` slot when building spin systems.
    pub max_cands: usize,
    /// Hard user-locked `(spin_system, seq_index)` constraints.
    pub locked_assignments: Vec<LockedAssignment>,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for BayesSaConfig {
    fn default() -> Self {
        Self {
            replicas: BAYES_SA_CPU_DEFAULT_REPLICAS,
            cooling_schedule: CoolingScheduleKind::AdaptiveSpecificHeat,
            t_start: 16.0,
            t_min: 0.15,
            delta_beta_0: 0.095,
            specific_heat_eta: 10.0,
            steps_per_stage_per_active: 30,
            min_stages: 32,
            max_stages: 150,
            fixed_total_steps: None,
            convergence_cv_tol: 0.008,
            convergence_acc_tol: 0.0020,
            max_fragment_len: 28,
            candidate_list_size: 32,
            outlier_floor_eps: 0.015,
            sigma_ca: 0.12,
            sigma_cb: 0.20,
            sigma_c: 0.12,
            sigma_ha: 0.035,
            link_floor: -3.6,
            support_bonus: 0.65,
            mu_null_override: None,
            use_hn: true,
            auto_reference: true,
            enable_weak_root_penalty: true,
            forbid_zero_carbons: true,
            forbid_ca_less_excess_roots: true,
            apply_qc_calibration: true,
            tolerances: Tolerances::default(),
            max_cands: 3,
            locked_assignments: Vec::new(),
            seed: 0,
        }
    }
}

impl BayesSaConfig {
    /// Default configuration tuned for massively parallel WebGPU execution (`R = 128` replicas,
    /// multi-chunk TDR-safe `<= 4,096` steps/submit stage execution, and 16-entry candidate tables).
    pub fn default_gpu() -> Self {
        Self {
            replicas: BAYES_SA_GPU_DEFAULT_REPLICAS,
            t_start: 9.0,
            delta_beta_0: 0.180,
            steps_per_stage_per_active: 7,
            min_stages: 24,
            max_stages: 44,
            max_fragment_len: 16,
            candidate_list_size: 16,
            ..Self::default()
        }
    }

    /// Build a matching `ScoreConfig` view for `qc::evaluate_dataset_qc` compatibility.
    pub fn to_qc_score_config(&self) -> ScoreConfig {
        ScoreConfig {
            sigma_ca: self.sigma_ca,
            sigma_cb: self.sigma_cb,
            sigma_c: self.sigma_c,
            range_ppm: 30.0,
            pair_floor: self.link_floor,
            support_bonus: self.support_bonus,
            assign_cost: self.mu_null_override.unwrap_or(0.0),
            forbid_zero_carbons: self.forbid_zero_carbons,
            use_hn: self.use_hn,
            auto_reference: self.auto_reference,
            forbid_ca_less_excess_roots: self.forbid_ca_less_excess_roots,
            locked_assignments: self
                .locked_assignments
                .iter()
                .map(|l| (l.spin_system, l.seq_index))
                .collect(),
        }
    }

    #[inline]
    pub fn link_sigma(&self, chan_idx: usize) -> f64 {
        match chan_idx {
            0 => self.sigma_ca,
            1 => self.sigma_cb,
            2 => self.sigma_c,
            3 => self.sigma_ha,
            _ => 0.15,
        }
    }
}

/// Per-stage thermodynamic ensemble telemetry recorded during simulated annealing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BayesSaTempStep {
    pub stage: usize,
    pub temperature: f64,
    pub steps_per_replica: usize,
    /// Ensemble mean pseudo-energy $\langle E \rangle_{T_m}$ across all replicas (nats).
    pub mean_energy: f64,
    /// Equilibrium intra-stage thermal energy variance $\langle E^2 \rangle_{T_m} - \langle E \rangle_{T_m}^2 \ge 0$.
    pub energy_variance: f64,
    /// Static inter-replica spread of stage-mean energies across the $R$ chains ($\ge 0$).
    pub ensemble_spread_variance: f64,
    /// Dimensionless thermodynamic specific heat $C_v(T_m) = \mathrm{Var}_{T_m}(E) / T_m^2 \ge 0$.
    pub specific_heat_cv: f64,
    /// Fraction of accepted Metropolis–Hastings proposals at stage $T_m$.
    pub acceptance_rate: f64,
    /// Lowest pseudo-energy observed across all replicas up to and including stage $T_m$.
    pub min_replica_energy: f64,
}

/// Precomputed Bayesian pseudo-energy matrices and reference-offset metadata.
#[derive(Clone, Debug)]
pub struct BayesSaMatrices {
    /// Pure `(S × K)` marginal log-Bayes-factor typing pseudo-energy $E_{\text{type}}(s, k)$.
    pub e_type: Matrix2D,
    /// Effective `(S × K)` unary pseudo-energy $E_{\text{type}}(s, k) + E_{\text{weak}}(s) + \mu_{\text{null}}$.
    pub e_unary: Matrix2D,
    /// `(S × S)` multi-pathway sequential overlap pseudo-energy $E_{\text{link}}(s, t)$.
    pub e_link: Matrix2D,
    /// Per-spin-system weak-root / minor-conformer / corroboration pseudo-energy adjustment.
    pub weak_penalties: Vec<f64>,
    /// Calibrated chemical potential $\mu_{\text{null}}$ of the null/dummy assignment reservoir.
    pub mu_null: f64,
    /// Global chemical-shift reference offsets applied to `Priors` during scoring.
    pub applied_offsets: ShiftOffsets,
    /// Detailed per-nucleus reference offset fit telemetry.
    pub detailed_offsets: DetailedReferenceOffsets,
    /// Effective `Priors` shifted into the experimental spectrum's ppm frame.
    pub effective_priors: Priors,
    /// Validated injective `(spin_system, seq_index)` hard constraints.
    pub resolved_locks: Vec<(usize, usize)>,
    /// Number of active unlocked non-Proline amide sequence positions $N_{\text{active}}$.
    pub active_slots: usize,
}

/// Diagnostic and thermodynamic telemetry returned by `solve_bayes_sa` / `solve_bayes_sa_gpu`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BayesSaReport {
    /// Minimum Bayesian pseudo-energy $E(\mathbf{x}^*)$ found across all replicas (lower is better).
    pub best_energy: f64,
    /// Mean final pseudo-energy across all $R$ replicas at termination.
    pub final_mean_energy: f64,
    /// Variance of final replica pseudo-energies.
    pub final_energy_variance: f64,
    /// Best pseudo-energy reached by each individual replica (`len == replicas`).
    pub replica_best_energies: Vec<f64>,
    /// Final pseudo-energy of each individual replica (`len == replicas`).
    pub replica_final_energies: Vec<f64>,
    /// Critical phase-transition temperature $T_{\text{crit}}$ at which $C_v(T)$ peaked.
    pub t_crit: f64,
    /// Peak specific heat $\max_m C_v(T_m)$ observed along the cooling trajectory.
    pub max_cv: f64,
    /// Number of temperature stages executed.
    pub total_stages: usize,
    /// Total MCMC proposals executed per replica across all stages.
    pub total_steps_per_replica: usize,
    /// True if post-peak low-$T$ convergence criteria terminated cooling prior to `max_stages`.
    pub converged_early: bool,
    /// Calibrated null-state chemical potential $\mu_{\text{null}}$ (in nats).
    pub mu_null: f64,
    /// Applied global chemical-shift reference offsets.
    pub applied_offsets: ShiftOffsets,
    /// Per-nucleus reference offset diagnostics.
    pub detailed_offsets: DetailedReferenceOffsets,
    /// Unlabelled dataset quality-control verdict and confidence damping telemetry.
    pub qc_report: DatasetQcReport,
    /// Full stage-by-stage thermodynamic trajectory $(T_m, \langle E \rangle, \mathrm{Var}(E), C_v(T_m), \text{acc})$.
    pub cooling_trajectory: Vec<BayesSaTempStep>,
}

// ============================================================================
// Packed Host <-> WebGPU Shared Scoring & Move Structures
// ============================================================================

/// Packed read-only scoring tables shared identically between CPU `f64` reference routines and
/// `nmr-gpu::bayes_sa_gpu` (`bayes_sa.wgsl`) to guarantee `< 1e-4` (`~1e-6`) evaluation parity.
#[derive(Clone, Debug)]
pub struct BayesPackedScoringInputs {
    pub s_len: usize,
    pub k_len: usize,
    pub active_slots: usize,
    pub mu_null: f64,
    pub outlier_floor_eps: f64,
    pub link_floor: f64,
    pub use_hn: bool,
    pub sigmas_link: [f64; NUM_LINK_CHANNELS],
    /// `20 * 3 * 6` centered base conformer means $\mu_{r, c, a}^{(0)} - C_a$.
    pub bg_mu: Vec<f64>,
    /// `20 * 3 * 6` Student-$t$ scales $\sigma_{t, r, c, a}$.
    pub bg_sig_t: Vec<f64>,
    /// `20 * 6` atom presence flags (`0` for `GLY CB` and `PRO H`, else `1`).
    pub res_atom_present: Vec<u32>,
    /// `20 * 3` residue conformer weights $w(r, c)$.
    pub res_conf_w: Vec<f64>,
    /// `20 * 6` sequence-smoothed background residue weights $\pi_{\text{bg}, a}(r)$ per channel.
    pub pi_bg_chan: Vec<f64>,
    /// `20` foreground $i$-position sequence composition $\pi_i(r)$ (`0` on `PRO`, sums to `1`).
    pub pi_seq_i: [f64; 20],
    /// `20` foreground $(i-1)$-position sequence composition $\pi_{i-1}(r)$ (sums to `1`).
    pub pi_seq_im1: [f64; 20],
    /// `9` baseline joint conformer log-probabilities $\ln P_0(c_i, c_{i-1})$.
    pub log_p0_joint: [f64; 9],
    /// `K` standard residue indices (`0..20`, or `20` for unknown).
    pub seq_res_idx: Vec<u32>,
    /// `K * 3 * 6` covalent-adjusted centered conformer means $\mu_{k, c, a} - C_a$.
    pub seq_mu: Vec<f64>,
    /// `K * 3 * 6` Student-$t$ scales $\sigma_{t, k, c, a}$.
    pub seq_sig_t: Vec<f64>,
    /// `K * 9` sequence-specific joint conformer log-priors $\ln P(c_k, c_{k-1} \mid \text{aa}_k, \text{aa}_{k-1})$.
    pub seq_log_p_joint: Vec<f64>,
    /// `S * 10 * MAX_BAYES_SLOT_CANDS` centered candidate shifts (`slots 0..6` = `off 0` for `CA,CB,C,HA,N,H`;
    /// `slots 6..10` = `off -1` for `CA,CB,C,HA`) for unary typing (`sup > 0`).
    pub unary_cand_val: Vec<f64>,
    /// Matching normalized intra-slot weights $w_j$ ($\sum_j w_j = 1$) for unary typing.
    pub unary_cand_w: Vec<f64>,
    /// `S * 10` candidate counts (`0..=MAX_BAYES_SLOT_CANDS`) for unary typing.
    pub unary_cand_cnt: Vec<u32>,
    /// `S * 8 * MAX_BAYES_SLOT_CANDS` centered candidate shifts (`slots 0..4` = `off 0` for `CA,CB,C,HA`;
    /// `slots 4..8` = `off -1` for `CA,CB,C,HA`, including `sup == 0` degenerate candidates) for `E_link`.
    pub link_cand_val: Vec<f64>,
    /// Matching normalized candidate weights for `E_link`.
    pub link_cand_w: Vec<f64>,
    /// `S * 8` candidate counts for `E_link`.
    pub link_cand_cnt: Vec<u32>,
    /// `S * 8` flags (`1` if slot has only `sup == 0` degenerate candidates, else `0`).
    pub link_only_deg: Vec<u32>,
    /// `S` flags (`1` if spin system is globally forbidden from non-locked placement).
    pub sys_forbidden: Vec<u32>,
    /// `S` flags (`1` if CA-less excess root forbidden on non-Gly / non-post-Pro slots).
    pub sys_ca_less_forbid: Vec<u32>,
    /// `S` additive per-spin unary adjustments (`weak_penalties[s] + mu_null`).
    pub sys_extra_unary: Vec<f64>,
    /// `S` raw weak-root penalties (excluding `mu_null`).
    pub weak_penalties: Vec<f64>,
    /// `S` locked sequence target (`-1` if unlocked).
    pub lock_k_for_s: Vec<i32>,
    /// `K` locked spin system (`-1` if unlocked).
    pub lock_s_for_k: Vec<i32>,
    pub applied_offsets: ShiftOffsets,
    pub detailed_offsets: DetailedReferenceOffsets,
    pub effective_priors: Priors,
    pub resolved_locks: Vec<(usize, usize)>,
}

/// Precomputed state-independent tables driving the symmetric MCMC move set on CPU and WebGPU.
#[derive(Clone, Debug)]
pub struct BayesMoveTables {
    /// Active unlocked non-Proline residue indices (`k in 1..K`).
    pub free_k: Vec<u32>,
    /// Inverse lookup mapping `k -> index in free_k` (`-1` for `k=0`, `PRO`, or locked slots).
    pub free_k_idx: Vec<i32>,
    /// Admissible unlocked spin-system indices (`s in 0..S`).
    pub free_s: Vec<u32>,
    /// `1` if sequence position `k` is an unlocked assignable non-Proline slot, else `0`.
    pub unlocked_k_flag: Vec<u32>,
    /// Candidate list width `M` (`0..=32`, uniform across all `k`).
    pub cand_m: usize,
    /// Flattened `K * M` top-unary candidate spin systems per slot (`-1` when unused).
    pub cand_spins: Vec<i32>,
    /// Flattened `S * 8` top-unary sequence slots `k` per spin system (`-1` when unused).
    pub top_k8: Vec<i32>,
    /// Flattened `S * 8` top sequential successor spin systems (`E_link(s, succ) < -0.5`, else `-1`).
    pub succ8: Vec<i32>,
    /// Flattened `S * 8` top sequential predecessor spin systems (`E_link(pred, s) < -0.5`, else `-1`).
    pub pred8: Vec<i32>,
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

#[inline]
fn residue_index(res: &str) -> Option<usize> {
    STANDARD_RESIDUES
        .iter()
        .position(|&(three, _)| three == res)
}

const GLY_RES_IDX: u32 = 7;
const PRO_RES_IDX: u32 = 14;

/// Biophysical baseline conformer propensities $(w_\alpha, w_\beta, w_{\text{coil}})$ per amino acid
/// (Wishart et al., *J. Biomol. NMR* 1995; MacArthur & Thornton, *J. Mol. Biol.* 1991).
fn residue_conformer_weights(res: &str) -> [f64; 3] {
    match res {
        "PRO" => [0.14, 0.12, 0.74],
        "GLY" => [0.18, 0.18, 0.64],
        "VAL" | "ILE" | "THR" => [0.26, 0.40, 0.34],
        "CYS" | "PHE" | "TYR" | "TRP" => [0.28, 0.34, 0.38],
        "ALA" | "GLU" | "LEU" | "MET" | "LYS" | "ARG" | "GLN" => [0.42, 0.20, 0.38],
        "ASN" | "ASP" | "SER" | "HIS" => [0.26, 0.20, 0.54],
        _ => [0.34, 0.24, 0.42],
    }
}

/// Raw secondary-structure chemical-shift displacements $(\Delta_\alpha, \Delta_\beta, \Delta_{\text{coil}})$
/// in ppm relative to random coil (Spera & Bax, *J. Am. Chem. Soc.* 1991; Wishart et al., 1991/1995).
fn raw_conformer_displacements(res: &str, chan_idx: usize) -> [f64; 3] {
    let is_gly = res == "GLY";
    let is_pro = res == "PRO";
    match chan_idx {
        // CA
        0 => {
            if is_gly {
                [1.05, -0.65, 0.0]
            } else if is_pro {
                [1.75, -0.95, 0.0]
            } else {
                [2.65, -1.15, 0.0]
            }
        }
        // CB
        1 => {
            if is_gly {
                [0.0, 0.0, 0.0]
            } else if is_pro {
                [-0.55, 0.65, 0.0]
            } else {
                [-1.10, 1.45, 0.0]
            }
        }
        // C'
        2 => {
            if is_gly {
                [1.10, -0.75, 0.0]
            } else {
                [1.85, -1.05, 0.0]
            }
        }
        // HA
        3 => [-0.30, 0.32, 0.0],
        // N
        4 => {
            if is_pro {
                [0.0, 0.0, 0.0]
            } else {
                [-1.50, 1.75, 0.0]
            }
        }
        // H
        5 => {
            if is_pro {
                [0.0, 0.0, 0.0]
            } else {
                [-0.26, 0.32, 0.0]
            }
        }
        _ => [0.0, 0.0, 0.0],
    }
}

#[inline]
fn min_within_conformer_sd(chan_idx: usize) -> f64 {
    match chan_idx {
        0 => 0.95, // CA
        1 => 1.05, // CB
        2 => 0.95, // C'
        3 => 0.24, // HA
        4 => 2.65, // N
        5 => 0.44, // H
        _ => 1.00,
    }
}

/// Cooperative sequential transition matrix $T(c_{k-1} \to c_k)$ along the polypeptide chain.
fn sequential_ss_transition(curr_is_pro: bool) -> [[f64; 3]; 3] {
    if curr_is_pro {
        [[0.14, 0.08, 0.78], [0.08, 0.30, 0.62], [0.18, 0.12, 0.70]]
    } else {
        [[0.80, 0.04, 0.16], [0.04, 0.76, 0.20], [0.14, 0.14, 0.72]]
    }
}

fn joint_conformer_log_probs(res_curr: &str, res_prev: &str, curr_pre_pro: bool) -> [f64; 9] {
    let w0 = [0.34_f64, 0.24, 0.42];
    let mut w_curr = residue_conformer_weights(res_curr);
    if curr_pre_pro && res_curr != "PRO" {
        // MacArthur & Thornton (1991): pre-Proline strongly favors non-helical conformations (~85%).
        w_curr = [0.15, 0.27, 0.58];
    }
    let w_prev = residue_conformer_weights(res_prev);
    let trans = sequential_ss_transition(res_curr == "PRO");

    let mut raw = [0.0_f64; 9];
    let mut sum = 0.0_f64;
    for c_curr in 0..3 {
        for c_prev in 0..3 {
            let mod_prev = (w_prev[c_prev] / w0[c_prev]).sqrt();
            let mod_curr = (w_curr[c_curr] / w0[c_curr]).sqrt();
            let p = w0[c_prev] * trans[c_prev][c_curr] * mod_prev * mod_curr;
            raw[3 * c_curr + c_prev] = p;
            sum += p;
        }
    }
    let mut out = [0.0_f64; 9];
    for idx in 0..9 {
        out[idx] = (raw[idx] / sum).max(1e-12).ln();
    }
    out
}

#[inline]
pub fn student_t5_robust_pdf(
    x_centered: f64,
    mu_centered: f64,
    sig_t: f64,
    range_ppm: f64,
    eps: f64,
) -> f64 {
    let z = (x_centered - mu_centered) / sig_t;
    let u = 1.0 + 0.2 * z * z;
    let t_pdf = STUDENT_T5_NORM / (sig_t * u * u * u);
    (1.0 - eps) * t_pdf + eps / range_ppm
}

#[inline]
fn logsumexp_slice(vals: &[f64]) -> f64 {
    let m = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0_f64;
    for &v in vals {
        s += (v - m).exp();
    }
    m + s.ln()
}

fn median_usize(mut vals: Vec<usize>) -> usize {
    if vals.is_empty() {
        return 0;
    }
    vals.sort_unstable();
    vals[vals.len() / 2]
}

/// Compute weak-root, minor-conformer, sidechain-NH2, water-artifact, and corroboration unary
/// adjustments (`weak_penalties[s]`), plus calibrated null chemical potential $\mu_{\text{null}}$.
fn compute_root_penalties_and_mu_null(
    systems: &[SpinSystem],
    norm_seq: &[String],
    cfg: &BayesSaConfig,
) -> (Vec<f64>, Vec<u32>, Vec<u32>, f64) {
    let s_len = systems.len();
    let assignable = (1..norm_seq.len())
        .filter(|&k| norm_seq[k] != "PRO")
        .count()
        .max(1);
    let has_excess = cfg.forbid_ca_less_excess_roots && s_len > assignable;
    let any_heights = systems.iter().any(|ss| ss.root_height.is_some());

    let mut has_ca0 = vec![false; s_len];
    let mut has_cam1 = vec![false; s_len];
    let mut obs_slots = vec![0usize; s_len];
    let mut corrob_slots = vec![0usize; s_len];
    let mut max_sup = vec![0usize; s_len];
    let mut slot_counts_nonempty = Vec::with_capacity(s_len);

    for (s, ss) in systems.iter().enumerate() {
        ca0_cam1_and_slots(
            ss,
            &mut has_ca0[s],
            &mut has_cam1[s],
            &mut obs_slots[s],
            &mut corrob_slots[s],
            &mut max_sup[s],
        );
        if obs_slots[s] > 0 {
            slot_counts_nonempty.push(obs_slots[s]);
        }
    }

    let med_slots = median_usize(slot_counts_nonempty);
    let corrob_nonempty: Vec<usize> = obs_slots
        .iter()
        .zip(corrob_slots.iter())
        .filter_map(|(&n, &c)| (n > 0).then_some(c))
        .collect();
    let med_corrob = median_usize(corrob_nonempty);
    let dataset_has_corrob = corrob_slots.iter().filter(|&&c| c > 0).count() * 3 >= s_len.max(1);
    let multi_slot_dataset =
        obs_slots.iter().filter(|&&n| n >= 3).count() >= 8.min(assignable / 2 + 1);

    let mut sys_forbidden = vec![0u32; s_len];
    let mut sys_ca_less_forbid = vec![0u32; s_len];
    let mut penalties = vec![0.0_f64; s_len];

    for (s, ss) in systems.iter().enumerate() {
        if cfg.forbid_zero_carbons && !ss.has_observed_carbons() {
            sys_forbidden[s] = 1;
            continue;
        }
        // Water-suppression (< 5.35 ppm) or extreme downfield (> 12.25 ppm) artifact roots when heights exist.
        if cfg.enable_weak_root_penalty && any_heights && !(5.35..=12.25).contains(&ss.h) {
            sys_forbidden[s] = 1;
            continue;
        }
        let has_any_ca = has_ca0[s] || has_cam1[s];
        // Only hard-forbid CA-less excess roots when they also have at most 1 uncorroborated peak;
        // multi-slot strips that lost CA due to missing-peak dropout get a soft penalty instead.
        if has_excess && !has_any_ca && obs_slots[s] <= 1 && max_sup[s] <= 1 {
            sys_ca_less_forbid[s] = 1;
        }
    }

    if cfg.enable_weak_root_penalty && any_heights {
        // Detect indirect 15N truncation sinc sidelobes and t1-noise ridges along strong 1H columns
        for s in 0..s_len {
            if sys_forbidden[s] == 1 || corrob_slots[s] >= 1 {
                continue;
            }
            let Some(ht_s) = systems[s].root_height else {
                continue;
            };
            for p in 0..s_len {
                if p == s || sys_forbidden[p] == 1 || obs_slots[p] < 2 {
                    continue;
                }
                let Some(ht_p) = systems[p].root_height else {
                    continue;
                };
                let dh = (systems[s].h - systems[p].h).abs();
                let dn = (systems[s].n - systems[p].n).abs();
                let is_sinc_lobe =
                    dh <= 0.024 && (1.00..=2.05).contains(&dn) && ht_s < 0.115 * ht_p;
                let is_t1_ridge = dh <= 0.028 && dn > 0.65 && ht_s < 0.052 * ht_p && !has_ca0[s];
                if is_sinc_lobe || is_t1_ridge {
                    sys_forbidden[s] = 1;
                    break;
                }
            }
        }
    }

    if cfg.enable_weak_root_penalty {
        // Residual paired Asn/Gln NH2 sidechains with jittered 15N (<= 0.24 ppm) lacking intra-CA(0)
        for i in 0..s_len {
            if has_ca0[i] || sys_forbidden[i] == 1 {
                continue;
            }
            for j in (i + 1)..s_len {
                if has_ca0[j] || sys_forbidden[j] == 1 {
                    continue;
                }
                let dn = (systems[i].n - systems[j].n).abs();
                let dh = (systems[i].h - systems[j].h).abs();
                if dn <= 0.24 && (0.28..=1.25).contains(&dh) {
                    let c_match = match (
                        systems[i].cands.get(&("C".to_string(), -1)),
                        systems[j].cands.get(&("C".to_string(), -1)),
                    ) {
                        (Some(ci), Some(cj)) => ci.iter().any(|&(vi, si)| {
                            si > 0 && cj.iter().any(|&(vj, sj)| sj > 0 && (vi - vj).abs() <= 0.30)
                        }),
                        _ => false,
                    };
                    let cb_match = match (
                        systems[i].cands.get(&("CB".to_string(), -1)),
                        systems[j].cands.get(&("CB".to_string(), -1)),
                    ) {
                        (Some(bi), Some(bj)) => bi.iter().any(|&(vi, si)| {
                            si > 0 && bj.iter().any(|&(vj, sj)| sj > 0 && (vi - vj).abs() <= 0.45)
                        }),
                        _ => false,
                    };
                    if c_match || cb_match || (!has_cam1[i] && !has_cam1[j]) {
                        sys_forbidden[i] = 1;
                        sys_forbidden[j] = 1;
                    }
                }
            }
        }
    }

    let clean_s_len = sys_forbidden.iter().filter(|&&f| f == 0).count();
    let clean_excess_frac = if clean_s_len > assignable {
        ((clean_s_len - assignable) as f64 / (assignable as f64)).min(1.0)
    } else {
        0.0
    };

    // Calibrated chemical potential mu_null balancing missing-root gaps vs excess noise roots.
    let mu_null = if let Some(explicit) = cfg.mu_null_override {
        explicit
    } else if med_slots == 0 {
        0.15
    } else if clean_s_len <= assignable {
        0.55
    } else {
        0.65 + 0.65 * clean_excess_frac
    };

    if !cfg.enable_weak_root_penalty {
        return (penalties, sys_forbidden, sys_ca_less_forbid, mu_null);
    }

    // Major-root reference list for local slow-exchange minor-conformer detection
    let major_roots: Vec<(usize, f64)> = systems
        .iter()
        .enumerate()
        .filter_map(|(idx, ss)| {
            let r = ss.rel_root_intensity.or(ss.rel_root_height)?;
            (sys_forbidden[idx] == 0 && r >= 0.55 && obs_slots[idx] >= 2).then_some((idx, r))
        })
        .collect();

    for (s, ss) in systems.iter().enumerate() {
        if sys_forbidden[s] == 1 {
            continue;
        }
        let has_any_ca = has_ca0[s] || has_cam1[s];
        if multi_slot_dataset {
            if clean_s_len > assignable && !has_any_ca {
                penalties[s] += if obs_slots[s] <= 1 { 4.2 } else { 2.4 };
            } else if clean_s_len > assignable && obs_slots[s] <= 1 && max_sup[s] <= 1 {
                penalties[s] += 3.5;
            } else if obs_slots[s] <= 1 && max_sup[s] <= 1 {
                penalties[s] += 2.2;
            } else if dataset_has_corrob && corrob_slots[s] == 0 && max_sup[s] <= 1 {
                penalties[s] += if med_corrob >= 3 && !has_cam1[s] {
                    6.5
                } else if med_corrob >= 2 {
                    if !has_cam1[s] {
                        4.35
                    } else {
                        3.45
                    }
                } else if obs_slots[s] <= 2 {
                    1.65
                } else {
                    1.35
                };
            }
        }

        // Corroboration credit for multi-experiment validated strips
        if corrob_slots[s] > 0 {
            penalties[s] -= 0.22 * (corrob_slots[s].min(4) as f64);
        }

        // Sub-stoichiometric intensity & local slow-exchange shadow damping
        if let Some(r) = ss.rel_root_intensity.or(ss.rel_root_height) {
            if r < 0.35 {
                let u_sub = ((0.35 - r.max(0.0)) / 0.35).clamp(0.0, 1.0);
                penalties[s] += 0.80 * u_sub;
                if obs_slots[s] >= 2 {
                    let mut best_maj: Option<f64> = None;
                    for &(m, r_maj) in &major_roots {
                        if m == s || r_maj < 4.0 * r.max(0.015) {
                            continue;
                        }
                        let sm = &systems[m];
                        if (ss.h - sm.h).abs() > 0.25 || (ss.n - sm.n).abs() > 1.80 {
                            continue;
                        }
                        let shared = count_shared_carbon_slots(ss, sm);
                        if shared >= 2 {
                            best_maj = Some(best_maj.map_or(r_maj, |p| p.max(r_maj)));
                        }
                    }
                    if let Some(r_maj) = best_maj {
                        let log_pop = (r_maj / r.max(0.015)).ln().max(0.0);
                        penalties[s] += 3.5 * log_pop * (1.0 - (-8.0 * u_sub).exp());
                    }
                }
            }
        }
    }

    (penalties, sys_forbidden, sys_ca_less_forbid, mu_null)
}

fn ca0_cam1_and_slots(
    ss: &SpinSystem,
    has_ca0: &mut bool,
    has_cam1: &mut bool,
    n_slots: &mut usize,
    n_corrob: &mut usize,
    max_sup: &mut usize,
) {
    *has_ca0 = ss
        .cands
        .get(&("CA".to_string(), 0))
        .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0));
    *has_cam1 = ss
        .cands
        .get(&("CA".to_string(), -1))
        .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0));
    let mut slots = 0usize;
    let mut corrob = 0usize;
    let mut m_sup = 0usize;
    for raw in ss.cands.values() {
        let mut slot_sup = 0usize;
        for &(_, sup) in raw {
            if sup > slot_sup {
                slot_sup = sup;
            }
        }
        if slot_sup > 0 {
            slots += 1;
            if slot_sup >= 2 {
                corrob += 1;
            }
            if slot_sup > m_sup {
                m_sup = slot_sup;
            }
        }
    }
    *n_slots = slots;
    *n_corrob = corrob;
    *max_sup = m_sup;
}

fn count_shared_carbon_slots(a: &SpinSystem, b: &SpinSystem) -> usize {
    let mut hits = 0usize;
    for ((atom, off), a_list) in &a.cands {
        if !a_list.iter().any(|&(_, sup)| sup > 0) {
            continue;
        }
        let Some(b_list) = b.cands.get(&(atom.clone(), *off)) else {
            continue;
        };
        let tol = if atom == "C" { 0.15 } else { 0.22 };
        let matched = a_list.iter().any(|&(va, sa)| {
            sa > 0
                && b_list
                    .iter()
                    .any(|&(vb, sb)| sb > 0 && (va - vb).abs() <= tol)
        });
        if matched {
            hits += 1;
        }
    }
    hits
}

/// Pack all sequence, conformer-prior, background-density, and spin-system inputs for exact
/// CPU (`f64`) and WebGPU (`f32`) scoring parity.
pub fn pack_bayes_scoring_inputs(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> BayesPackedScoringInputs {
    let norm_seq = normalize_sequence(sequence);
    let s_len = systems.len();
    let k_len = norm_seq.len();

    let raw_locks: Vec<(usize, usize)> = cfg
        .locked_assignments
        .iter()
        .map(|l| (l.spin_system, l.seq_index))
        .collect();
    let resolved_locks = resolve_locked_assignments(s_len, &norm_seq, &raw_locks);

    let mut lock_k_for_s = vec![-1i32; s_len];
    let mut lock_s_for_k = vec![-1i32; k_len];
    for &(s_l, k_l) in &resolved_locks {
        lock_k_for_s[s_l] = k_l as i32;
        lock_s_for_k[k_l] = s_l as i32;
    }

    let active_slots = (1..k_len)
        .filter(|&k| norm_seq[k] != "PRO" && lock_s_for_k[k] < 0)
        .count();

    let (weak_penalties, sys_forbidden, sys_ca_less_forbid, mu_null) =
        compute_root_penalties_and_mu_null(systems, &norm_seq, cfg);
    let sys_extra_unary: Vec<f64> = weak_penalties.iter().map(|&w| w + mu_null).collect();

    // Estimate global reference offsets using only clean (non-artifact) spin systems
    let raw_detailed = if cfg.auto_reference && !systems.is_empty() && !norm_seq.is_empty() {
        let clean_systems: Vec<SpinSystem> = systems
            .iter()
            .enumerate()
            .filter(|&(s, _)| sys_forbidden[s] == 0)
            .map(|(_, ss)| ss.clone())
            .collect();
        if clean_systems.len() >= 4 {
            estimate_reference_offsets_detailed(&clean_systems, &norm_seq, priors)
        } else {
            estimate_reference_offsets_detailed(systems, &norm_seq, priors)
        }
    } else {
        DetailedReferenceOffsets::default()
    };
    let applied_offsets = if cfg.auto_reference {
        raw_detailed.to_shift_offsets()
    } else {
        ShiftOffsets::default()
    };
    let detailed_offsets = sync_detailed_reference_offsets(&applied_offsets, &raw_detailed);
    let effective_priors = priors.with_offsets(&applied_offsets);
    let all_atom_priors = Priors::all_atoms();

    // 1. Build 20-residue × 3-conformer × 6-channel centered base table
    let mut bg_mu = vec![0.0_f64; 20 * 3 * NUM_BAYES_CHANNELS];
    let mut bg_sig_t = vec![1.0_f64; 20 * 3 * NUM_BAYES_CHANNELS];
    let mut res_atom_present = vec![0u32; 20 * NUM_BAYES_CHANNELS];
    let mut res_conf_w = vec![0.0_f64; 20 * 3];

    for (r_idx, &(res_name, _)) in STANDARD_RESIDUES.iter().enumerate() {
        let w = residue_conformer_weights(res_name);
        for c in 0..3 {
            res_conf_w[r_idx * 3 + c] = w[c];
        }
        for (a_idx, &atom) in BAYES_CHANNEL_ATOMS.iter().enumerate() {
            let stat_opt = if atom == "HA" {
                all_atom_priors
                    .get(res_name, "HA")
                    .or_else(|| all_atom_priors.get(res_name, "HA2"))
                    .map(|(m, s)| (m + applied_offsets.h, s))
            } else {
                effective_priors.get(res_name, atom)
            };
            let Some((mu_bmrb, sd_bmrb)) = stat_opt else {
                continue;
            };
            res_atom_present[r_idx * NUM_BAYES_CHANNELS + a_idx] = 1;
            let raw_disp = raw_conformer_displacements(res_name, a_idx);
            let mean_disp = w[0] * raw_disp[0] + w[1] * raw_disp[1] + w[2] * raw_disp[2];
            let centered_disp = [
                raw_disp[0] - mean_disp,
                raw_disp[1] - mean_disp,
                raw_disp[2] - mean_disp,
            ];
            let var_between = w[0] * centered_disp[0].powi(2)
                + w[1] * centered_disp[1].powi(2)
                + w[2] * centered_disp[2].powi(2);
            let sd_floor = min_within_conformer_sd(a_idx);
            // Preserve oxidized/reduced broad variance on CYS CB
            let kappa = if res_name == "CYS" && atom == "CB" {
                0.20
            } else {
                0.85
            };
            let sd_within = (sd_bmrb * sd_bmrb - kappa * var_between)
                .max(sd_floor * sd_floor)
                .sqrt();
            let sig_t = (sd_within * STUDENT_T5_SIGMA_SCALE).max(0.10);
            for (c, &disp_c) in centered_disp.iter().enumerate() {
                let idx = (r_idx * 3 + c) * NUM_BAYES_CHANNELS + a_idx;
                bg_mu[idx] = (mu_bmrb + disp_c) - CHANNEL_CENTERS_PPM[a_idx];
                bg_sig_t[idx] = sig_t;
            }
        }
    }

    // 2. Sequence amino-acid composition priors (Laplace-smoothed)
    let mut cnt_all = [0usize; 20];
    let mut cnt_i = [0usize; 20];
    let mut cnt_im1 = [0usize; 20];
    for (k, r_str) in norm_seq.iter().enumerate() {
        if let Some(ri) = residue_index(r_str) {
            cnt_all[ri] += 1;
            if k >= 1 && r_str != "PRO" {
                cnt_i[ri] += 1;
                if let Some(prev_i) = residue_index(&norm_seq[k - 1]) {
                    cnt_im1[prev_i] += 1;
                }
            }
        }
    }
    let tot_i: usize = cnt_i.iter().sum();
    let tot_im1: usize = cnt_im1.iter().sum();
    let tot_all: usize = cnt_all.iter().sum();

    // Dirichlet pseudocount shrinkage (alpha_0 = 16) toward globular natural AA abundance so short
    // peptides do not over-inflate rare residue background density while full proteins track stoichiometry.
    const NAT_AA_PROB: [f64; 20] = [
        0.082, 0.055, 0.041, 0.054, 0.014, 0.039, 0.067, 0.071, 0.023, 0.059, 0.096, 0.058, 0.024,
        0.039, 0.047, 0.066, 0.053, 0.011, 0.029, 0.068,
    ];
    const DIRICHLET_ALPHA: f64 = 16.0;

    let mut pi_seq_i = [0.0_f64; 20];
    let mut pi_seq_im1 = [0.0_f64; 20];
    for r in 0..20 {
        if STANDARD_RESIDUES[r].0 == "PRO" {
            pi_seq_i[r] = 0.0;
        } else {
            pi_seq_i[r] = ((cnt_i[r] as f64) + DIRICHLET_ALPHA * NAT_AA_PROB[r])
                / ((tot_i as f64) + DIRICHLET_ALPHA);
        }
        pi_seq_im1[r] = ((cnt_im1[r] as f64) + DIRICHLET_ALPHA * NAT_AA_PROB[r])
            / ((tot_im1 as f64) + DIRICHLET_ALPHA);
    }
    let sum_pi_i: f64 = pi_seq_i.iter().sum();
    let sum_pi_im1: f64 = pi_seq_im1.iter().sum();
    for r in 0..20 {
        pi_seq_i[r] /= sum_pi_i;
        pi_seq_im1[r] /= sum_pi_im1;
    }

    let mut pi_bg_chan = vec![0.0_f64; 20 * NUM_BAYES_CHANNELS];
    for a_idx in 0..NUM_BAYES_CHANNELS {
        let mut sum_a = 0.0_f64;
        for r in 0..20 {
            if res_atom_present[r * NUM_BAYES_CHANNELS + a_idx] == 0 {
                continue;
            }
            if a_idx == 4 && STANDARD_RESIDUES[r].0 == "PRO" {
                continue;
            }
            let w = ((cnt_all[r] as f64) + DIRICHLET_ALPHA * NAT_AA_PROB[r])
                / ((tot_all as f64) + DIRICHLET_ALPHA);
            pi_bg_chan[r * NUM_BAYES_CHANNELS + a_idx] = w;
            sum_a += w;
        }
        if sum_a > 0.0 {
            for r in 0..20 {
                pi_bg_chan[r * NUM_BAYES_CHANNELS + a_idx] /= sum_a;
            }
        }
    }

    let log_p0_joint = joint_conformer_log_probs("ALA", "VAL", false);
    // Re-compute neutral baseline joint log-probabilities using global globular weights w0
    let log_p0_joint = {
        let w0 = [0.34_f64, 0.24, 0.42];
        let trans = sequential_ss_transition(false);
        let mut out = [0.0_f64; 9];
        let mut s = 0.0_f64;
        for c_curr in 0..3 {
            for c_prev in 0..3 {
                let p = w0[c_prev] * trans[c_prev][c_curr];
                out[3 * c_curr + c_prev] = p;
                s += p;
            }
        }
        for v in &mut out {
            *v = (*v / s).ln();
        }
        let _ = log_p0_joint;
        out
    };

    // 3. Per-sequence-position covalent-adjusted conformer expectations
    let mut seq_res_idx = vec![20u32; k_len];
    let mut seq_mu = vec![0.0_f64; k_len * 3 * NUM_BAYES_CHANNELS];
    let mut seq_sig_t = vec![1.0_f64; k_len * 3 * NUM_BAYES_CHANNELS];
    let mut seq_log_p_joint = vec![0.0_f64; k_len * 9];

    const NATURAL_PRO_FRAC: f64 = 91.0 / 2508.0;
    for k in 0..k_len {
        let res_str = norm_seq[k].as_str();
        let next_is_pro = k + 1 < k_len && norm_seq[k + 1] == "PRO";
        let prev_str = if k > 0 {
            norm_seq[k - 1].as_str()
        } else {
            "ALA"
        };
        let jp = joint_conformer_log_probs(res_str, prev_str, next_is_pro);
        seq_log_p_joint[k * 9..(k + 1) * 9].copy_from_slice(&jp);

        let Some(r_idx) = residue_index(res_str) else {
            continue;
        };
        seq_res_idx[k] = r_idx as u32;
        let cov = delta_covalent(&norm_seq, k);

        for (a_idx, &atom) in BAYES_CHANNEL_ATOMS.iter().enumerate() {
            let d_cov_mean = covalent_atom_offset(cov, atom);
            for c in 0..3 {
                let base_i = (r_idx * 3 + c) * NUM_BAYES_CHANNELS + a_idx;
                let dst_i = (k * 3 + c) * NUM_BAYES_CHANNELS + a_idx;
                let d_cov_conf =
                    if next_is_pro && (a_idx == 0 || a_idx == 2) && d_cov_mean.abs() >= 0.09 {
                        let raw = d_cov_mean / (COVALENT_COIL_WEIGHT * (1.0 - NATURAL_PRO_FRAC));
                        let d_non = -COVALENT_COIL_WEIGHT * NATURAL_PRO_FRAC * raw;
                        if c == 0 {
                            // Kinked alpha-helical pre-Proline experiences strongly attenuated steric shift
                            d_non + 0.20 * raw
                        } else {
                            d_non + raw
                        }
                    } else {
                        d_cov_mean
                    };
                seq_mu[dst_i] = bg_mu[base_i] + d_cov_conf;
                seq_sig_t[dst_i] = bg_sig_t[base_i];
            }
        }
    }

    // 4. Pack spin-system candidate slots (centered in f64!)
    let max_slot = cfg.max_cands.clamp(1, MAX_BAYES_SLOT_CANDS);
    let mut unary_cand_val = vec![0.0_f64; s_len * 10 * MAX_BAYES_SLOT_CANDS];
    let mut unary_cand_w = vec![0.0_f64; s_len * 10 * MAX_BAYES_SLOT_CANDS];
    let mut unary_cand_cnt = vec![0u32; s_len * 10];

    let mut link_cand_val = vec![0.0_f64; s_len * 8 * MAX_BAYES_SLOT_CANDS];
    let mut link_cand_w = vec![0.0_f64; s_len * 8 * MAX_BAYES_SLOT_CANDS];
    let mut link_cand_cnt = vec![0u32; s_len * 8];
    let mut link_only_deg = vec![0u32; s_len * 8];

    for (s, ss) in systems.iter().enumerate() {
        // Unary intra (slots 0..4 = CA,CB,C,HA @ 0; 4 = N; 5 = H) + sequential (slots 6..10 = CA,CB,C,HA @ -1)
        for a_idx in 0..4 {
            let atom = BAYES_CHANNEL_ATOMS[a_idx];
            let center = CHANNEL_CENTERS_PPM[a_idx];
            for (off, u_slot, l_slot) in [(0, a_idx, a_idx), (-1, 6 + a_idx, 4 + a_idx)] {
                if let Some(raw) = ss.cands.get(&(atom.to_string(), off)) {
                    // Unary: strictly observed (`sup > 0`)
                    let obs: Vec<(f64, usize)> = raw
                        .iter()
                        .copied()
                        .filter(|&(_, sup)| sup > 0)
                        .take(max_slot)
                        .collect();
                    if !obs.is_empty() {
                        let mut w_sum = 0.0_f64;
                        let mut ws = [0.0_f64; MAX_BAYES_SLOT_CANDS];
                        for (j, &(_, sup)) in obs.iter().enumerate() {
                            let rank_decay = 0.62_f64.powi(j as i32);
                            let w = rank_decay
                                * (cfg.support_bonus * ((sup as f64) - 1.0))
                                    .clamp(0.0, 3.0)
                                    .exp();
                            ws[j] = w;
                            w_sum += w;
                        }
                        let u_idx = s * 10 + u_slot;
                        unary_cand_cnt[u_idx] = obs.len() as u32;
                        for (j, &(v, _)) in obs.iter().enumerate() {
                            let cell = u_idx * MAX_BAYES_SLOT_CANDS + j;
                            unary_cand_val[cell] = v - center;
                            unary_cand_w[cell] = ws[j] / w_sum;
                        }
                    }

                    // Link: includes `sup == 0` degenerate candidates
                    let lnk: Vec<(f64, usize)> = raw.iter().copied().take(max_slot).collect();
                    if !lnk.is_empty() {
                        let all_deg = lnk.iter().all(|&(_, sup)| sup == 0);
                        let mut w_sum = 0.0_f64;
                        let mut ws = [0.0_f64; MAX_BAYES_SLOT_CANDS];
                        for (j, &(_, sup)) in lnk.iter().enumerate() {
                            let rank_decay = 0.65_f64.powi(j as i32);
                            let w = if sup == 0 {
                                0.35 * rank_decay
                            } else {
                                rank_decay * (1.0 + 0.50 * ((sup - 1) as f64))
                            };
                            ws[j] = w;
                            w_sum += w;
                        }
                        let l_idx = s * 8 + l_slot;
                        link_cand_cnt[l_idx] = lnk.len() as u32;
                        link_only_deg[l_idx] = u32::from(all_deg);
                        for (j, &(v, _)) in lnk.iter().enumerate() {
                            let cell = l_idx * MAX_BAYES_SLOT_CANDS + j;
                            link_cand_val[cell] = v - center;
                            link_cand_w[cell] = ws[j] / w_sum;
                        }
                    }
                }
            }
        }

        // Amide N (slot 4) and H (slot 5)
        if cfg.use_hn {
            if (95.0..=145.0).contains(&ss.n) {
                let u_n = s * 10 + 4;
                unary_cand_cnt[u_n] = 1;
                unary_cand_val[u_n * MAX_BAYES_SLOT_CANDS] = ss.n - CHANNEL_CENTERS_PPM[4];
                unary_cand_w[u_n * MAX_BAYES_SLOT_CANDS] = 1.0;
            }
            if (5.0..=12.5).contains(&ss.h) {
                let u_h = s * 10 + 5;
                unary_cand_cnt[u_h] = 1;
                unary_cand_val[u_h * MAX_BAYES_SLOT_CANDS] = ss.h - CHANNEL_CENTERS_PPM[5];
                unary_cand_w[u_h * MAX_BAYES_SLOT_CANDS] = 1.0;
            }
        }
    }

    BayesPackedScoringInputs {
        s_len,
        k_len,
        active_slots,
        mu_null,
        outlier_floor_eps: cfg.outlier_floor_eps,
        link_floor: cfg.link_floor,
        use_hn: cfg.use_hn,
        sigmas_link: [cfg.sigma_ca, cfg.sigma_cb, cfg.sigma_c, cfg.sigma_ha],
        bg_mu,
        bg_sig_t,
        res_atom_present,
        res_conf_w,
        pi_bg_chan,
        pi_seq_i,
        pi_seq_im1,
        log_p0_joint,
        seq_res_idx,
        seq_mu,
        seq_sig_t,
        seq_log_p_joint,
        unary_cand_val,
        unary_cand_w,
        unary_cand_cnt,
        link_cand_val,
        link_cand_w,
        link_cand_cnt,
        link_only_deg,
        sys_forbidden,
        sys_ca_less_forbid,
        sys_extra_unary,
        weak_penalties,
        lock_k_for_s,
        lock_s_for_k,
        applied_offsets,
        detailed_offsets,
        effective_priors,
        resolved_locks,
    }
}

#[inline]
fn eval_channel_bg_density(inp: &BayesPackedScoringInputs, a_idx: usize, x_centered: f64) -> f64 {
    let range = CHANNEL_RANGES_PPM[a_idx];
    let eps = inp.outlier_floor_eps;
    let mut sum = 0.0_f64;
    for r in 0..20 {
        let pi_r = inp.pi_bg_chan[r * NUM_BAYES_CHANNELS + a_idx];
        if pi_r <= 0.0 {
            continue;
        }
        let mut r_pdf = 0.0_f64;
        for c in 0..3 {
            let w_c = inp.res_conf_w[r * 3 + c];
            let base = (r * 3 + c) * NUM_BAYES_CHANNELS + a_idx;
            r_pdf += w_c
                * student_t5_robust_pdf(
                    x_centered,
                    inp.bg_mu[base],
                    inp.bg_sig_t[base],
                    range,
                    eps,
                );
        }
        sum += pi_r * r_pdf;
    }
    // Smooth spectral-band floor so extreme shifts have bounded background density
    (0.985 * sum + 0.015 / range).max(1e-12)
}

#[inline]
fn eval_slot_log_bf_residue(
    inp: &BayesPackedScoringInputs,
    s: usize,
    u_slot: usize,
    a_idx: usize,
    res_idx: usize,
    conf_c: usize,
    use_seq_pos: Option<usize>,
) -> f64 {
    let cnt = inp.unary_cand_cnt[s * 10 + u_slot] as usize;
    if cnt == 0 {
        return 0.0;
    }
    if inp.res_atom_present[res_idx * NUM_BAYES_CHANNELS + a_idx] == 0 {
        return if a_idx == 1 && res_idx == GLY_RES_IDX as usize {
            GLY_OBSERVED_CB_LOG_BF
        } else {
            0.0
        };
    }
    let (mu, sig_t) = if let Some(k_pos) = use_seq_pos {
        let idx = (k_pos * 3 + conf_c) * NUM_BAYES_CHANNELS + a_idx;
        (inp.seq_mu[idx], inp.seq_sig_t[idx])
    } else {
        let idx = (res_idx * 3 + conf_c) * NUM_BAYES_CHANNELS + a_idx;
        (inp.bg_mu[idx], inp.bg_sig_t[idx])
    };
    let range = CHANNEL_RANGES_PPM[a_idx];
    let eps = inp.outlier_floor_eps;
    let base_cell = (s * 10 + u_slot) * MAX_BAYES_SLOT_CANDS;
    let mut bf_sum = 0.0_f64;
    for j in 0..cnt {
        let x = inp.unary_cand_val[base_cell + j];
        let w = inp.unary_cand_w[base_cell + j];
        let p_fg = student_t5_robust_pdf(x, mu, sig_t, range, eps);
        let p_bg = eval_channel_bg_density(inp, a_idx, x);
        bf_sum += w * (p_fg / p_bg);
    }
    bf_sum.max(1e-12).ln()
}

fn compute_spin_ln_z_bg(inp: &BayesPackedScoringInputs, s: usize) -> f64 {
    let mut ln_bi = [0.0_f64; 3];
    for (c, ln_bi_c) in ln_bi.iter_mut().enumerate() {
        let mut terms = Vec::with_capacity(19);
        for r in 0..20 {
            let pi = inp.pi_seq_i[r];
            if pi <= 0.0 {
                continue;
            }
            let mut ll = 0.0_f64;
            for a_idx in 0..NUM_BAYES_CHANNELS {
                ll += eval_slot_log_bf_residue(inp, s, a_idx, a_idx, r, c, None);
            }
            terms.push(pi.ln() + ll);
        }
        *ln_bi_c = logsumexp_slice(&terms);
    }

    let mut ln_bim1 = [0.0_f64; 3];
    for (c_prev, ln_bim1_c) in ln_bim1.iter_mut().enumerate() {
        let mut terms = [0.0_f64; 20];
        for (r_prev, term_r) in terms.iter_mut().enumerate() {
            let mut ll = 0.0_f64;
            for a_idx in 0..4 {
                ll += eval_slot_log_bf_residue(inp, s, 6 + a_idx, a_idx, r_prev, c_prev, None);
            }
            *term_r = inp.pi_seq_im1[r_prev].ln() + ll;
        }
        *ln_bim1_c = logsumexp_slice(&terms);
    }

    let mut joint9 = [0.0_f64; 9];
    for c in 0..3 {
        for c_prev in 0..3 {
            joint9[3 * c + c_prev] = inp.log_p0_joint[3 * c + c_prev] + ln_bi[c] + ln_bim1[c_prev];
        }
    }
    logsumexp_slice(&joint9)
}

/// Compute `E_type`, `E_unary`, and `E_link` on the CPU from `systems`, `sequence`, and `priors`.
pub fn compute_bayes_energy_matrices(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> BayesSaMatrices {
    let inp = pack_bayes_scoring_inputs(systems, sequence, priors, cfg);
    compute_bayes_matrices_from_packed(&inp)
}

/// Evaluate `BayesSaMatrices` directly from pre-packed inputs (`f64` reference implementation).
pub fn compute_bayes_matrices_from_packed(inp: &BayesPackedScoringInputs) -> BayesSaMatrices {
    let s_len = inp.s_len;
    let k_len = inp.k_len;
    let mut e_type = Matrix2D::filled(s_len, k_len, FORBID_ENERGY);
    let mut e_unary = Matrix2D::filled(s_len, k_len, FORBID_ENERGY);
    let mut e_link = Matrix2D::filled(s_len, s_len, 0.0);

    if s_len == 0 || k_len == 0 {
        return BayesSaMatrices {
            e_type,
            e_unary,
            e_link,
            weak_penalties: inp.weak_penalties.clone(),
            mu_null: inp.mu_null,
            applied_offsets: inp.applied_offsets,
            detailed_offsets: inp.detailed_offsets,
            effective_priors: inp.effective_priors.clone(),
            resolved_locks: inp.resolved_locks.clone(),
            active_slots: inp.active_slots,
        };
    }

    let ln_z_bg: Vec<f64> = (0..s_len).map(|s| compute_spin_ln_z_bg(inp, s)).collect();

    for (s, &ln_z_bg_s) in ln_z_bg.iter().enumerate() {
        for k in 1..k_len {
            let rk = inp.seq_res_idx[k];
            if rk == PRO_RES_IDX {
                continue;
            }
            let is_exact_lock =
                inp.lock_k_for_s[s] == (k as i32) && inp.lock_s_for_k[k] == (s as i32);
            if !is_exact_lock {
                if inp.lock_k_for_s[s] >= 0 || inp.lock_s_for_k[k] >= 0 {
                    continue;
                }
                if inp.sys_forbidden[s] != 0 {
                    continue;
                }
                let rkm1 = inp.seq_res_idx[k - 1];
                if inp.sys_ca_less_forbid[s] != 0 && rk != GLY_RES_IDX && rkm1 != PRO_RES_IDX {
                    continue;
                }
            }

            let raw_et = if rk < 20 && inp.seq_res_idx[k - 1] < 20 {
                let rk_u = rk as usize;
                let rkm1_u = inp.seq_res_idx[k - 1] as usize;
                let mut ll_i = [0.0_f64; 3];
                for (c, ll_i_c) in ll_i.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for a_idx in 0..NUM_BAYES_CHANNELS {
                        acc += eval_slot_log_bf_residue(inp, s, a_idx, a_idx, rk_u, c, Some(k));
                    }
                    *ll_i_c = acc;
                }
                let mut ll_im1 = [0.0_f64; 3];
                for (c_prev, ll_im1_c) in ll_im1.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for a_idx in 0..4 {
                        acc += eval_slot_log_bf_residue(
                            inp,
                            s,
                            6 + a_idx,
                            a_idx,
                            rkm1_u,
                            c_prev,
                            Some(k - 1),
                        );
                    }
                    *ll_im1_c = acc;
                }
                let mut joint9 = [0.0_f64; 9];
                for c in 0..3 {
                    for c_prev in 0..3 {
                        let jp = inp.seq_log_p_joint[k * 9 + 3 * c + c_prev];
                        joint9[3 * c + c_prev] = jp + ll_i[c] + ll_im1[c_prev];
                    }
                }
                let ln_z_fg = logsumexp_slice(&joint9);
                -(ln_z_fg - ln_z_bg_s)
            } else {
                0.0
            };

            if is_exact_lock {
                let clamped = raw_et.clamp(-500.0, 500.0);
                e_type.set(s, k, clamped);
                e_unary.set(s, k, clamped);
            } else {
                e_type.set(s, k, raw_et);
                e_unary.set(s, k, raw_et + inp.sys_extra_unary[s]);
            }
        }
    }

    // Compute pairwise multi-pathway sequential overlap pseudo-energy E_link(s, t)
    let p_leak_std = inp.link_floor.exp();
    let p_leak_deg = (-2.5_f64).exp().max(p_leak_std);
    let sqrt_tau = TAU.sqrt();

    for s in 0..s_len {
        e_link.set(s, s, FORBID_ENERGY);
        for t in 0..s_len {
            if s == t {
                continue;
            }
            let mut link_e = 0.0_f64;
            for a_idx in 0..NUM_LINK_CHANNELS {
                let s_slot = s * 8 + a_idx;
                let t_slot = t * 8 + 4 + a_idx;
                let na = inp.link_cand_cnt[s_slot] as usize;
                let nb = inp.link_cand_cnt[t_slot] as usize;
                if na == 0 || nb == 0 {
                    continue;
                }
                let sig = inp.sigmas_link[a_idx];
                let norm_gauss = 1.0 / (sqrt_tau * sig);
                let only_deg = inp.link_only_deg[s_slot] != 0 || inp.link_only_deg[t_slot] != 0;
                let p_leak = if only_deg { p_leak_deg } else { p_leak_std };

                let s_base = s_slot * MAX_BAYES_SLOT_CANDS;
                let t_base = t_slot * MAX_BAYES_SLOT_CANDS;
                let mut overlap_sum = 0.0_f64;
                for ia in 0..na {
                    let xa = inp.link_cand_val[s_base + ia];
                    let wa = inp.link_cand_w[s_base + ia];
                    for ib in 0..nb {
                        let yb = inp.link_cand_val[t_base + ib];
                        let wb = inp.link_cand_w[t_base + ib];
                        let d = (xa - yb) / sig;
                        let g = norm_gauss * (-0.5 * d * d).exp();
                        let x_mid = 0.5 * (xa + yb);
                        let p_rand = eval_channel_bg_density(inp, a_idx, x_mid);
                        overlap_sum += (wa * wb) * (g / p_rand);
                    }
                }
                let bf = (1.0 - p_leak) * overlap_sum + p_leak;
                link_e -= bf.max(1e-12).ln();
            }
            e_link.set(s, t, link_e);
        }
    }

    BayesSaMatrices {
        e_type,
        e_unary,
        e_link,
        weak_penalties: inp.weak_penalties.clone(),
        mu_null: inp.mu_null,
        applied_offsets: inp.applied_offsets,
        detailed_offsets: inp.detailed_offsets,
        effective_priors: inp.effective_priors.clone(),
        resolved_locks: inp.resolved_locks.clone(),
        active_slots: inp.active_slots,
    }
}

/// Evaluate the exact Bayesian pseudo-energy $E(\mathbf{x})$ of a sequence-slot assignment vector
/// `at` (`len == K`, where `at[k] == s >= 0` or `-1` for null/dummy).
pub fn evaluate_state_energy(at: &[i32], matrices: &BayesSaMatrices) -> Result<f64, String> {
    let s_len = matrices.e_unary.rows;
    let k_len = matrices.e_unary.cols;
    if at.len() != k_len {
        return Err(format!("at.len() {} != k_len {}", at.len(), k_len));
    }
    let mut seen_s = vec![false; s_len];
    let mut total = 0.0_f64;
    for (k, &s_raw) in at.iter().enumerate() {
        if s_raw >= 0 {
            let s = s_raw as usize;
            if s >= s_len {
                return Err(format!("invalid spin system {s} >= {s_len} at slot {k}"));
            }
            if seen_s[s] {
                return Err(format!("duplicate spin system {s} in state vector"));
            }
            seen_s[s] = true;
            let eu = matrices.e_unary.get(s, k);
            if eu >= FORBID_ENERGY * 0.5 {
                return Err(format!(
                    "forbidden placement (s={s}, k={k}) in state vector"
                ));
            }
            total += eu;
        }
    }
    for k in 1..k_len {
        if at[k - 1] >= 0 && at[k] >= 0 {
            total += matrices.e_link.get(at[k - 1] as usize, at[k] as usize);
        }
    }
    Ok(total)
}

/// Evaluate the exact Bayesian pseudo-energy $E(\mathbf{x})$ from a spin-system `position` vector
/// (`len == S`, where `position[s] == k >= 0` or `-1` for unassigned/null).
pub fn evaluate_position_energy(
    position: &[i32],
    matrices: &BayesSaMatrices,
) -> Result<f64, String> {
    let s_len = matrices.e_unary.rows;
    let k_len = matrices.e_unary.cols;
    if position.len() != s_len {
        return Err(format!(
            "position.len() {} != s_len {}",
            position.len(),
            s_len
        ));
    }
    let mut at = vec![-1i32; k_len];
    for (s, &k_raw) in position.iter().enumerate() {
        if k_raw >= 0 {
            let k = k_raw as usize;
            if k >= k_len {
                return Err(format!("invalid residue index {k} >= {k_len}"));
            }
            if at[k] >= 0 {
                return Err(format!("duplicate residue assignment at k={k}"));
            }
            at[k] = s as i32;
        }
    }
    evaluate_state_energy(&at, matrices)
}

// ============================================================================
// Move-Table Construction & Replica Initialization
// ============================================================================

/// Precompute state-independent tables for symmetric MCMC proposals on CPU and WebGPU.
pub fn build_bayes_move_tables(matrices: &BayesSaMatrices, cfg: &BayesSaConfig) -> BayesMoveTables {
    let s_len = matrices.e_unary.rows;
    let k_len = matrices.e_unary.cols;

    let mut locked_s = vec![false; s_len];
    let mut locked_k = vec![false; k_len];
    for &(s_l, k_l) in &matrices.resolved_locks {
        if s_l < s_len && k_l < k_len {
            locked_s[s_l] = true;
            locked_k[k_l] = true;
        }
    }

    let mut unlocked_k_flag = vec![0u32; k_len];
    let mut free_k_idx = vec![-1i32; k_len];
    let mut free_k = Vec::with_capacity(k_len);
    for k in 1..k_len {
        if locked_k[k] {
            continue;
        }
        let any_ok = (0..s_len).any(|s| matrices.e_unary.get(s, k) < FORBID_ENERGY * 0.5);
        let is_non_pro = if s_len > 0 { any_ok } else { false };
        if is_non_pro {
            unlocked_k_flag[k] = 1;
            free_k_idx[k] = free_k.len() as i32;
            free_k.push(k as u32);
        }
    }

    let mut free_s = Vec::with_capacity(s_len);
    for (s, &is_locked) in locked_s.iter().enumerate() {
        if is_locked {
            continue;
        }
        let admissible = free_k
            .iter()
            .any(|&k| matrices.e_unary.get(s, k as usize) < FORBID_ENERGY * 0.5);
        if admissible {
            free_s.push(s as u32);
        }
    }

    // 1. Build sequential partner lists succ8 / pred8 first so static pentapeptide walk messages
    // can rank `cand_spins` and `top_k8` cleanly even on 480–800 aa sequences with 40+ copies per AA.
    let mut succ8 = vec![-1i32; s_len * 8];
    let mut pred8 = vec![-1i32; s_len * 8];
    for &s_u32 in &free_s {
        let s = s_u32 as usize;
        let mut succs: Vec<(u32, f64)> = free_s
            .iter()
            .copied()
            .filter(|&t| t != s_u32)
            .filter_map(|t| {
                let el = matrices.e_link.get(s, t as usize);
                (el < -0.5).then_some((t, el))
            })
            .collect();
        succs.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (j, &(t, _)) in succs.iter().take(8).enumerate() {
            succ8[s * 8 + j] = t as i32;
        }

        let mut preds: Vec<(u32, f64)> = free_s
            .iter()
            .copied()
            .filter(|&t| t != s_u32)
            .filter_map(|t| {
                let el = matrices.e_link.get(t as usize, s);
                (el < -0.5).then_some((t, el))
            })
            .collect();
        preds.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (j, &(t, _)) in preds.iter().take(8).enumerate() {
            pred8[s * 8 + j] = t as i32;
        }
    }

    // 2. Static 2-step bidirectional sequential-walk message passing over (E_unary, E_link)
    // (100% state-independent, O(S * K * 8) < 1 ms at 480 aa) to disambiguate degenerate single-AA
    // candidates via pentapeptide [k-2 .. k+2] motif compatibility.
    let mut msg_l = vec![0.0_f64; s_len * k_len];
    let mut msg_r = vec![0.0_f64; s_len * k_len];
    if !free_s.is_empty() && k_len > 2 {
        let mut lock_at_k = vec![-1i32; k_len];
        for &(sl, kl) in &matrices.resolved_locks {
            if sl < s_len && kl < k_len {
                lock_at_k[kl] = sl as i32;
            }
        }
        // 2. Static 4-step damped bidirectional sequential-walk message passing over (E_unary, E_link)
        // (100% state-independent, O(S * K * 8) < 2 ms at 480 aa) to disambiguate degenerate single-AA
        // candidates via nonapeptide [k-4 .. k+4] motif compatibility.
        let mut next_l = vec![0.0_f64; s_len * k_len];
        let mut next_r = vec![0.0_f64; s_len * k_len];
        for _iter in 0..4 {
            next_l.fill(0.0);
            next_r.fill(0.0);
            for &k_u32 in &free_k {
                let k = k_u32 as usize;
                // Left neighbor k - 1
                if k > 1 {
                    let km1 = k - 1;
                    if lock_at_k[km1] >= 0 {
                        let p = lock_at_k[km1] as usize;
                        for &s_u32 in &free_s {
                            let s = s_u32 as usize;
                            if matrices.e_unary.get(s, k) >= FORBID_ENERGY * 0.5 {
                                continue;
                            }
                            next_l[s * k_len + k] = matrices.e_link.get(p, s).min(0.0);
                        }
                    } else if unlocked_k_flag[km1] != 0 {
                        for &s_u32 in &free_s {
                            let s = s_u32 as usize;
                            if matrices.e_unary.get(s, k) >= FORBID_ENERGY * 0.5 {
                                continue;
                            }
                            let mut best = 0.0_f64;
                            for j in 0..8 {
                                let p_i32 = pred8[s * 8 + j];
                                if p_i32 < 0 {
                                    break;
                                }
                                let p = p_i32 as usize;
                                if p == s {
                                    continue;
                                }
                                let eu_p = matrices.e_unary.get(p, km1);
                                if eu_p < FORBID_ENERGY * 0.5 {
                                    let v = eu_p
                                        + matrices.e_link.get(p, s)
                                        + 0.72 * msg_l[p * k_len + km1];
                                    if v < best {
                                        best = v;
                                    }
                                }
                            }
                            next_l[s * k_len + k] = best;
                        }
                    }
                }
                // Right neighbor k + 1
                if k + 1 < k_len {
                    let kp1 = k + 1;
                    if lock_at_k[kp1] >= 0 {
                        let t = lock_at_k[kp1] as usize;
                        for &s_u32 in &free_s {
                            let s = s_u32 as usize;
                            if matrices.e_unary.get(s, k) >= FORBID_ENERGY * 0.5 {
                                continue;
                            }
                            next_r[s * k_len + k] = matrices.e_link.get(s, t).min(0.0);
                        }
                    } else if unlocked_k_flag[kp1] != 0 {
                        for &s_u32 in &free_s {
                            let s = s_u32 as usize;
                            if matrices.e_unary.get(s, k) >= FORBID_ENERGY * 0.5 {
                                continue;
                            }
                            let mut best = 0.0_f64;
                            for j in 0..8 {
                                let t_i32 = succ8[s * 8 + j];
                                if t_i32 < 0 {
                                    break;
                                }
                                let t = t_i32 as usize;
                                if t == s {
                                    continue;
                                }
                                let eu_t = matrices.e_unary.get(t, kp1);
                                if eu_t < FORBID_ENERGY * 0.5 {
                                    let v = matrices.e_link.get(s, t)
                                        + eu_t
                                        + 0.72 * msg_r[t * k_len + kp1];
                                    if v < best {
                                        best = v;
                                    }
                                }
                            }
                            next_r[s * k_len + k] = best;
                        }
                    }
                }
            }
            std::mem::swap(&mut msg_l, &mut next_l);
            std::mem::swap(&mut msg_r, &mut next_r);
        }
    }

    let guide_score = |s: usize, k: usize| -> f64 {
        matrices.e_unary.get(s, k) + 0.75 * (msg_l[s * k_len + k] + msg_r[s * k_len + k])
    };

    let cand_m = cfg.candidate_list_size.clamp(1, 32).min(free_s.len());
    let mut cand_spins = vec![-1i32; k_len * cand_m.max(1)];
    if cand_m > 0 {
        for &k_u32 in &free_k {
            let k = k_u32 as usize;
            let mut ranked: Vec<(u32, f64)> = free_s
                .iter()
                .copied()
                .filter_map(|s| {
                    let su = s as usize;
                    let eu = matrices.e_unary.get(su, k);
                    (eu < FORBID_ENERGY * 0.5).then_some((s, guide_score(su, k)))
                })
                .collect();
            ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
            for j in 0..cand_m {
                if j < ranked.len() {
                    cand_spins[k * cand_m + j] = ranked[j].0 as i32;
                }
            }
        }
    }

    let mut top_k8 = vec![-1i32; s_len * 8];
    for &s_u32 in &free_s {
        let s = s_u32 as usize;
        let mut best_ks: Vec<(u32, f64)> = free_k
            .iter()
            .copied()
            .filter_map(|k| {
                let ku = k as usize;
                let eu = matrices.e_unary.get(s, ku);
                (eu < FORBID_ENERGY * 0.5).then_some((k, guide_score(s, ku)))
            })
            .collect();
        best_ks.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (j, &(k_val, _)) in best_ks.iter().take(8).enumerate() {
            top_k8[s * 8 + j] = k_val as i32;
        }
    }

    BayesMoveTables {
        free_k,
        free_k_idx,
        free_s,
        unlocked_k_flag,
        cand_m,
        cand_spins,
        top_k8,
        succ8,
        pred8,
    }
}

/// Initial `(cur_ats, cur_positions, cur_energies, replica_seeds)` ensemble state.
pub type InitialBayesReplicas = (Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<f64>, Vec<u64>);

/// Initialize `replicas` feasible starting states `(cur_at, cur_pos, cur_energy)`
/// honoring all hard `resolved_locks` without artificial clash spikes before stage-0 equilibration.
pub fn build_initial_bayes_replicas(
    matrices: &BayesSaMatrices,
    tables: &BayesMoveTables,
    replicas: usize,
    master_rng: &mut Rng,
) -> InitialBayesReplicas {
    let s_len = matrices.e_unary.rows;
    let k_len = matrices.e_unary.cols;
    let r_cnt = replicas.max(1);

    let mut cur_ats = Vec::with_capacity(r_cnt);
    let mut cur_poss = Vec::with_capacity(r_cnt);
    let mut cur_energies = Vec::with_capacity(r_cnt);
    let mut sub_seeds = Vec::with_capacity(r_cnt);

    for _ in 0..r_cnt {
        let seed_r = master_rng.next_u64();
        sub_seeds.push(seed_r);
        let mut rng = Rng::new(seed_r ^ 0x9E37_79B9_7F4A_7C15);

        let mut at = vec![-1i32; k_len];
        let mut pos = vec![-1i32; s_len];
        for &(s_l, k_l) in &matrices.resolved_locks {
            if s_l < s_len && k_l < k_len {
                at[k_l] = s_l as i32;
                pos[s_l] = k_l as i32;
            }
        }

        // Seed ~20% sparse unary-admissible anchors so sequential chains nucleate cleanly
        // into open slots without dense random gridlock; stage 0 uses 75% burn-in before C_v.
        if !tables.free_s.is_empty() && !tables.free_k.is_empty() {
            let n_seed = (tables.free_k.len() / 5).max(1);
            for _ in 0..n_seed {
                let s = tables.free_s[rng.usize(tables.free_s.len())] as usize;
                let k = tables.free_k[rng.usize(tables.free_k.len())] as usize;
                if pos[s] < 0 && at[k] < 0 && matrices.e_unary.get(s, k) < FORBID_ENERGY * 0.5 {
                    at[k] = s as i32;
                    pos[s] = k as i32;
                }
            }
        }

        let e0 = evaluate_state_energy(&at, matrices).unwrap_or(0.0);
        cur_ats.push(at);
        cur_poss.push(pos);
        cur_energies.push(e0);
    }

    (cur_ats, cur_poss, cur_energies, sub_seeds)
}

// ============================================================================
// Thermodynamic Specific-Heat Adaptive Cooling Step Computation
// ============================================================================

/// Compute the next stage temperature $T_{m+1}$ from current stage telemetry.
#[inline]
pub fn next_stage_temperature(
    cfg: &BayesSaConfig,
    stage: usize,
    planned_stages: usize,
    current_temp: f64,
    specific_heat_cv: f64,
    active_slots: usize,
) -> f64 {
    match cfg.cooling_schedule {
        CoolingScheduleKind::AdaptiveSpecificHeat => {
            let n_act = (active_slots.max(1)) as f64;
            let cv_per_site = (specific_heat_cv / n_act).max(0.0);
            let denom = (1.0 + cfg.specific_heat_eta.max(0.0) * cv_per_site).sqrt();

            let base_beta = if cfg.fixed_total_steps.is_some() {
                let rem_stages = planned_stages.saturating_sub(stage + 1);
                if rem_stages == 0 {
                    return cfg.t_min;
                }
                if rem_stages <= 6 {
                    let rem_log = (current_temp / cfg.t_min.max(1e-6)).max(1.0).ln();
                    (rem_log / (rem_stages as f64) * denom).max(1e-4)
                } else {
                    // Constant base_beta across high-T and critical regimes guarantees strict
                    // monotonic deceleration with C_v(T_m) while arriving at low-T without a tail cliff.
                    let tot_log = (cfg.t_start / cfg.t_min.max(1e-6)).max(1.001).ln();
                    let geom_step = tot_log / ((planned_stages.saturating_sub(1)).max(1) as f64);
                    (1.56 * geom_step).max(1e-4)
                }
            } else {
                let rem_stages = planned_stages.saturating_sub(stage + 1);
                if rem_stages == 0 {
                    return cfg.t_min;
                }
                if rem_stages <= 6 {
                    let rem_log = (current_temp / cfg.t_min.max(1e-6)).max(1.0).ln();
                    (rem_log / (rem_stages as f64) * denom).max(cfg.delta_beta_0.max(1e-4))
                } else {
                    cfg.delta_beta_0.max(1e-4)
                }
            };

            let d_beta = base_beta / denom;
            (current_temp * (-d_beta).exp()).max(cfg.t_min)
        }
        CoolingScheduleKind::FixedGeometric => {
            let denom = (planned_stages.saturating_sub(1)).max(1) as f64;
            let frac = ((stage + 1) as f64 / denom).clamp(0.0, 1.0);
            let ratio = (cfg.t_min / cfg.t_start.max(1e-6)).clamp(1e-8, 1.0);
            (cfg.t_start * ratio.powf(frac)).max(cfg.t_min)
        }
    }
}

/// Determine per-stage step count per replica, scaling linearly with `active_slots` for both
/// `R >= 64` WebGPU ensembles (`[1_024, 8_192]`, dispatched via `<= 4,096` TDR-safe chunks) and
/// `R < 64` CPU ensembles (`[512, 12_288]`), honoring `fixed_total_steps` when set.
#[inline]
pub fn default_steps_per_stage(cfg: &BayesSaConfig, active_slots: usize) -> usize {
    if let Some(fixed_tot) = cfg.fixed_total_steps {
        (fixed_tot / cfg.max_stages.max(1)).max(64)
    } else if cfg.replicas >= 64 {
        (cfg.steps_per_stage_per_active.max(1) * active_slots.max(16)).clamp(1_024, 8_192)
    } else {
        (cfg.steps_per_stage_per_active.max(1) * active_slots.max(16)).clamp(512, 12_288)
    }
}

// ============================================================================
// CPU Replica Markov Kernel (Exact Detailed-Balance O(1) Moves)
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct ReplicaStageStats {
    mean_e: f64,
    m2_e: f64,
    n_samples: usize,
    n_accepted: usize,
    n_proposed: usize,
}

struct CpuReplicaState {
    at: Vec<i32>,
    pos: Vec<i32>,
    cur_e: f64,
    best_at: Vec<i32>,
    best_e: f64,
    rng: Rng,
}

#[inline]
fn site_energy(at: &[i32], e_unary: &Matrix2D, e_link: &Matrix2D, k: usize) -> f64 {
    let s = at[k];
    if s < 0 {
        return 0.0;
    }
    let su = s as usize;
    let mut e = e_unary.get(su, k);
    if k > 0 && at[k - 1] >= 0 {
        e += e_link.get(at[k - 1] as usize, su);
    }
    if k + 1 < at.len() && at[k + 1] >= 0 {
        e += e_link.get(su, at[k + 1] as usize);
    }
    e
}

#[inline]
fn two_site_energy(at: &[i32], e_unary: &Matrix2D, e_link: &Matrix2D, ka: usize, kb: usize) -> f64 {
    let (lo, hi) = if ka < kb { (ka, kb) } else { (kb, ka) };
    let slo = at[lo];
    let shi = at[hi];
    let mut e = 0.0_f64;
    if slo >= 0 {
        let su = slo as usize;
        e += e_unary.get(su, lo);
        if lo > 0 && at[lo - 1] >= 0 {
            e += e_link.get(at[lo - 1] as usize, su);
        }
        if at[lo + 1] >= 0 {
            e += e_link.get(su, at[lo + 1] as usize);
        }
    }
    if shi >= 0 {
        let su = shi as usize;
        e += e_unary.get(su, hi);
        if hi > lo + 1 && at[hi - 1] >= 0 {
            e += e_link.get(at[hi - 1] as usize, su);
        }
        if hi + 1 < at.len() && at[hi + 1] >= 0 {
            e += e_link.get(su, at[hi + 1] as usize);
        }
    }
    e
}

#[inline]
fn interval_energy(
    at: &[i32],
    e_unary: &Matrix2D,
    e_link: &Matrix2D,
    start: usize,
    len: usize,
) -> f64 {
    let end = start + len;
    let mut e = 0.0_f64;
    for (k, &s_at_k) in at.iter().enumerate().take(end).skip(start) {
        if s_at_k >= 0 {
            e += e_unary.get(s_at_k as usize, k);
        }
    }
    let edge_lo = start.max(1);
    let edge_hi = end.min(at.len().saturating_sub(1));
    for k in edge_lo..=edge_hi {
        if k < at.len() && at[k - 1] >= 0 && at[k] >= 0 {
            e += e_link.get(at[k - 1] as usize, at[k] as usize);
        }
    }
    e
}

#[inline]
fn two_disjoint_spans_energy(
    at: &[i32],
    e_unary: &Matrix2D,
    e_link: &Matrix2D,
    a_start: usize,
    a_len: usize,
    b_start: usize,
    b_len: usize,
) -> f64 {
    let (lo_s, lo_len, hi_s, hi_len) = if a_start < b_start {
        (a_start, a_len, b_start, b_len)
    } else {
        (b_start, b_len, a_start, a_len)
    };
    let mut e = interval_energy(at, e_unary, e_link, lo_s, lo_len)
        + interval_energy(at, e_unary, e_link, hi_s, hi_len);
    if lo_s + lo_len == hi_s && at[hi_s - 1] >= 0 && at[hi_s] >= 0 {
        e -= e_link.get(at[hi_s - 1] as usize, at[hi_s] as usize);
    }
    e
}

#[inline]
fn in_cand_spins(tables: &BayesMoveTables, k: usize, s: i32) -> bool {
    if s < 0 || tables.cand_m == 0 {
        return false;
    }
    let base = k * tables.cand_m;
    for j in 0..tables.cand_m {
        if tables.cand_spins[base + j] == s {
            return true;
        }
    }
    false
}

#[inline]
fn in_list8(list8: &[i32], row: i32, target: i32) -> bool {
    if row < 0 || target < 0 {
        return false;
    }
    let base = (row as usize) * 8;
    for j in 0..8 {
        if list8[base + j] == target {
            return true;
        }
    }
    false
}

#[inline]
fn pair_proposal_weight(
    at: &[i32],
    tables: &BayesMoveTables,
    k: usize,
    s: i32,
    r_glob: f64,
    r_cand: f64,
    r_ext: f64,
) -> f64 {
    let mut w = r_glob;
    if r_cand > 0.0 && in_cand_spins(tables, k, s) {
        w += r_cand;
    }
    if k > 0 && in_list8(&tables.succ8, at[k - 1], s) {
        w += r_ext;
    }
    if k + 1 < at.len() && in_list8(&tables.pred8, at[k + 1], s) {
        w += r_ext;
    }
    w
}

#[inline]
fn block_swap_proposal_weight(
    at: &[i32],
    tables: &BayesMoveTables,
    ia: usize,
    ib: usize,
    l: usize,
    m_start_f: f64,
) -> f64 {
    // Scaled by M_start (the uniform 1/M_start factor on the first index cancels in Q_rev / Q_fwd)
    let mut w = 0.80 / m_start_f;
    let r_guided = 0.15 / 8.0;

    let ka0 = tables.free_k[ia] as usize;
    let kb0 = tables.free_k[ib] as usize;
    let ka_end = tables.free_k[ia + l - 1] as usize;
    let kb_end = tables.free_k[ib + l - 1] as usize;

    // Partner-guided stitching across interval boundaries
    if ka0 > 1 && in_list8(&tables.succ8, at[ka0 - 1], at[kb0]) {
        w += r_guided;
    }
    if kb0 > 1 && in_list8(&tables.succ8, at[kb0 - 1], at[ka0]) {
        w += r_guided;
    }
    if ka_end + 1 < at.len() && in_list8(&tables.pred8, at[ka_end + 1], at[kb_end]) {
        w += r_guided;
    }
    if kb_end + 1 < at.len() && in_list8(&tables.pred8, at[kb_end + 1], at[ka_end]) {
        w += r_guided;
    }

    // Unary-home jump via top_k8 on head and tail spin systems
    if in_list8(&tables.top_k8, at[ka0], kb0 as i32) {
        w += r_guided;
    }
    if in_list8(&tables.top_k8, at[kb0], ka0 as i32) {
        w += r_guided;
    }
    if in_list8(&tables.top_k8, at[ka_end], kb_end as i32) {
        w += r_guided;
    }
    if in_list8(&tables.top_k8, at[kb_end], ka_end as i32) {
        w += r_guided;
    }
    w
}

fn run_cpu_replica_stage(
    rep: &mut CpuReplicaState,
    matrices: &BayesSaMatrices,
    tables: &BayesMoveTables,
    stage: usize,
    temp: f64,
    steps: usize,
    max_frag: usize,
) -> ReplicaStageStats {
    let n_fk = tables.free_k.len();
    let n_fs = tables.free_s.len();
    let k_len = matrices.e_unary.cols;
    let e_u = &matrices.e_unary;
    let e_l = &matrices.e_link;

    if n_fk == 0 || n_fs == 0 || steps == 0 {
        return ReplicaStageStats {
            mean_e: rep.cur_e,
            m2_e: 0.0,
            n_samples: 1,
            n_accepted: 0,
            n_proposed: 0,
        };
    }

    // 75% stage-0 burn-in (20% sparse seeding -> high-T occupancy equilibrium) and 50% thereafter
    let burn_in = if stage == 0 {
        (3 * steps) / 4
    } else {
        steps / 2
    };
    let mut stats = ReplicaStageStats::default();
    let inv_t = 1.0 / temp.max(1e-6);
    let max_w = max_frag.clamp(2, 32);

    let has_cands = tables.cand_m > 0;
    let r_glob = if has_cands {
        0.22 / (n_fs as f64)
    } else {
        0.62 / (n_fs as f64)
    };
    let r_cand = if has_cands {
        0.40 / (tables.cand_m as f64)
    } else {
        0.0
    };
    let r_ext = 0.19 / 8.0;

    for step in 0..steps {
        stats.n_proposed += 1;
        let u_move = rep.rng.f64();

        if u_move < 0.16 {
            // Move (a): Symmetric two-position spin-system / dummy swap on free_k
            if n_fk >= 2 {
                let (k1, k2) = if rep.rng.f64() < 0.55 && n_fk > 2 {
                    let i1 = rep.rng.usize(n_fk - 1);
                    let max_d = 12usize.min(n_fk - 1 - i1);
                    let d = 1 + rep.rng.usize(max_d);
                    (tables.free_k[i1] as usize, tables.free_k[i1 + d] as usize)
                } else {
                    let i1 = rep.rng.usize(n_fk);
                    let mut i2 = rep.rng.usize(n_fk - 1);
                    if i2 >= i1 {
                        i2 += 1;
                    }
                    (tables.free_k[i1] as usize, tables.free_k[i2] as usize)
                };
                let s1 = rep.at[k1];
                let s2 = rep.at[k2];
                if s1 != s2
                    && (s1 < 0 || e_u.get(s1 as usize, k2) < FORBID_ENERGY * 0.5)
                    && (s2 < 0 || e_u.get(s2 as usize, k1) < FORBID_ENERGY * 0.5)
                {
                    let before = two_site_energy(&rep.at, e_u, e_l, k1, k2);
                    rep.at[k1] = s2;
                    rep.at[k2] = s1;
                    let after = two_site_energy(&rep.at, e_u, e_l, k1, k2);
                    let de = after - before;
                    if de <= 0.0 || rep.rng.f64() < (-de * inv_t).exp() {
                        if s1 >= 0 {
                            rep.pos[s1 as usize] = k2 as i32;
                        }
                        if s2 >= 0 {
                            rep.pos[s2 as usize] = k1 as i32;
                        }
                        rep.cur_e += de;
                        stats.n_accepted += 1;
                    } else {
                        rep.at[k1] = s1;
                        rep.at[k2] = s2;
                    }
                }
            }
        } else if u_move < 0.72 {
            // Move (b): Exact Metropolis–Hastings spin-to-slot insertion/deletion/relocation/evict/swap
            let k = tables.free_k[rep.rng.usize(n_fk)] as usize;
            let sub = rep.rng.f64();
            let s_i32 = if sub >= 0.62 {
                let use_left = rep.rng.f64() < 0.5;
                let anchor = if use_left {
                    if k > 0 {
                        rep.at[k - 1]
                    } else {
                        -1
                    }
                } else if k + 1 < k_len {
                    rep.at[k + 1]
                } else {
                    -1
                };
                if anchor >= 0 {
                    let j = rep.rng.usize(8);
                    if use_left {
                        tables.succ8[(anchor as usize) * 8 + j]
                    } else {
                        tables.pred8[(anchor as usize) * 8 + j]
                    }
                } else {
                    -1
                }
            } else if sub >= 0.22 && has_cands {
                let j = rep.rng.usize(tables.cand_m);
                tables.cand_spins[k * tables.cand_m + j]
            } else {
                tables.free_s[rep.rng.usize(n_fs)] as i32
            };

            if s_i32 >= 0 {
                apply_mh_spin_slot_move(
                    rep,
                    e_u,
                    e_l,
                    tables,
                    k,
                    s_i32 as usize,
                    r_glob,
                    r_cand,
                    r_ext,
                    inv_t,
                    &mut stats,
                );
            }
        } else {
            // Move (c): Free-slot window cyclic shift (+1/-1) or exact MH non-overlapping block swap
            if n_fk >= 4 {
                if rep.rng.f64() < 0.34 {
                    // (c1) Register-only symmetric +1 / -1 cyclic shift along free_k[ia .. ia + w)
                    let max_cw = max_w.min(n_fk);
                    let w = if rep.rng.f64() < 0.55 {
                        2 + rep.rng.usize((max_cw.min(8) - 1).max(1))
                    } else {
                        2 + rep.rng.usize((max_cw - 1).max(1))
                    };
                    let ia = rep.rng.usize(n_fk + 1 - w);
                    let shift_right = rep.rng.f64() < 0.5;
                    let mut valid_u = true;
                    let mut any_change = false;
                    for off in 0..w {
                        let src_off = if shift_right {
                            if off == 0 {
                                w - 1
                            } else {
                                off - 1
                            }
                        } else if off + 1 == w {
                            0
                        } else {
                            off + 1
                        };
                        let k_dst = tables.free_k[ia + off] as usize;
                        let s_prop = rep.at[tables.free_k[ia + src_off] as usize];
                        if s_prop != rep.at[k_dst] {
                            any_change = true;
                        }
                        if s_prop >= 0 && e_u.get(s_prop as usize, k_dst) >= FORBID_ENERGY * 0.5 {
                            valid_u = false;
                            break;
                        }
                    }
                    if valid_u && any_change {
                        let k_lo = tables.free_k[ia] as usize;
                        let k_hi = tables.free_k[ia + w - 1] as usize;
                        let span = k_hi - k_lo + 1;
                        let before = interval_energy(&rep.at, e_u, e_l, k_lo, span);
                        if shift_right {
                            let last_s = rep.at[tables.free_k[ia + w - 1] as usize];
                            for off in (1..w).rev() {
                                rep.at[tables.free_k[ia + off] as usize] =
                                    rep.at[tables.free_k[ia + off - 1] as usize];
                            }
                            rep.at[tables.free_k[ia] as usize] = last_s;
                        } else {
                            let first_s = rep.at[tables.free_k[ia] as usize];
                            for off in 0..(w - 1) {
                                rep.at[tables.free_k[ia + off] as usize] =
                                    rep.at[tables.free_k[ia + off + 1] as usize];
                            }
                            rep.at[tables.free_k[ia + w - 1] as usize] = first_s;
                        }
                        let after = interval_energy(&rep.at, e_u, e_l, k_lo, span);
                        let de = after - before;
                        if de <= 0.0 || rep.rng.f64() < (-de * inv_t).exp() {
                            for off in 0..w {
                                let k_dst = tables.free_k[ia + off] as usize;
                                let sp = rep.at[k_dst];
                                if sp >= 0 {
                                    rep.pos[sp as usize] = k_dst as i32;
                                }
                            }
                            rep.cur_e += de;
                            stats.n_accepted += 1;
                        } else if shift_right {
                            let first_s = rep.at[tables.free_k[ia] as usize];
                            for off in 0..(w - 1) {
                                rep.at[tables.free_k[ia + off] as usize] =
                                    rep.at[tables.free_k[ia + off + 1] as usize];
                            }
                            rep.at[tables.free_k[ia + w - 1] as usize] = first_s;
                        } else {
                            let last_s = rep.at[tables.free_k[ia + w - 1] as usize];
                            for off in (1..w).rev() {
                                rep.at[tables.free_k[ia + off] as usize] =
                                    rep.at[tables.free_k[ia + off - 1] as usize];
                            }
                            rep.at[tables.free_k[ia] as usize] = last_s;
                        }
                    }
                } else {
                    // (c2) Exact MH non-overlapping block swap free_k[ia..ia+l) <-> free_k[ib..ib+l)
                    // combining 40% uniform, 30% sequential partner stitching, and 30% top_k8 unary-home jump
                    let max_l = max_w.min(n_fk / 2);
                    if max_l >= 2 {
                        let l = if rep.rng.f64() < 0.55 {
                            2 + rep.rng.usize((max_l.min(5) - 1).max(1))
                        } else {
                            2 + rep.rng.usize(max_l - 1)
                        };
                        let m_start = n_fk + 1 - l;
                        let ia = rep.rng.usize(m_start);
                        let ka0 = tables.free_k[ia] as usize;
                        let ka_end = tables.free_k[ia + l - 1] as usize;
                        let u_mode = rep.rng.f64();

                        let ib_opt: Option<usize> = if u_mode < 0.40 {
                            Some(rep.rng.usize(m_start))
                        } else if u_mode < 0.70 {
                            let stitch_left = rep.rng.f64() < 0.5;
                            let j = rep.rng.usize(8);
                            if stitch_left {
                                let anc = if ka0 > 1 { rep.at[ka0 - 1] } else { -1 };
                                if anc >= 0 {
                                    let sp = tables.succ8[(anc as usize) * 8 + j];
                                    if sp >= 0 {
                                        let kb = rep.pos[sp as usize];
                                        if kb >= 0 {
                                            let ib_c = tables.free_k_idx[kb as usize];
                                            if ib_c >= 0 && (ib_c as usize) < m_start {
                                                Some(ib_c as usize)
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                let anc = if ka_end + 1 < k_len {
                                    rep.at[ka_end + 1]
                                } else {
                                    -1
                                };
                                if anc >= 0 {
                                    let sp = tables.pred8[(anc as usize) * 8 + j];
                                    if sp >= 0 {
                                        let kb_e = rep.pos[sp as usize];
                                        if kb_e >= 0 {
                                            let ib_end = tables.free_k_idx[kb_e as usize];
                                            if ib_end >= (l as i32 - 1) {
                                                let ib_c = (ib_end as usize) + 1 - l;
                                                (ib_c < m_start).then_some(ib_c)
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                        } else {
                            let use_head = rep.rng.f64() < 0.5;
                            let j = rep.rng.usize(8);
                            if use_head {
                                let sh = rep.at[ka0];
                                if sh >= 0 {
                                    let kh = tables.top_k8[(sh as usize) * 8 + j];
                                    if kh >= 0 {
                                        let ib_c = tables.free_k_idx[kh as usize];
                                        (ib_c >= 0 && (ib_c as usize) < m_start)
                                            .then_some(ib_c as usize)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                let st = rep.at[ka_end];
                                if st >= 0 {
                                    let kt = tables.top_k8[(st as usize) * 8 + j];
                                    if kt >= 0 {
                                        let ib_end = tables.free_k_idx[kt as usize];
                                        if ib_end >= (l as i32 - 1) {
                                            let ib_c = (ib_end as usize) + 1 - l;
                                            (ib_c < m_start).then_some(ib_c)
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                        };

                        if let Some(ib) = ib_opt {
                            if ia.abs_diff(ib) >= l {
                                let mut ok = true;
                                let mut any_nonempty = false;
                                for off in 0..l {
                                    let ka = tables.free_k[ia + off] as usize;
                                    let kb = tables.free_k[ib + off] as usize;
                                    let sa = rep.at[ka];
                                    let sb = rep.at[kb];
                                    if sa != sb {
                                        any_nonempty = true;
                                    }
                                    if (sa >= 0 && e_u.get(sa as usize, kb) >= FORBID_ENERGY * 0.5)
                                        || (sb >= 0
                                            && e_u.get(sb as usize, ka) >= FORBID_ENERGY * 0.5)
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                                if ok && any_nonempty {
                                    let m_start_f = m_start as f64;
                                    let kb0 = tables.free_k[ib] as usize;
                                    let kb_end = tables.free_k[ib + l - 1] as usize;
                                    let a_span = ka_end - ka0 + 1;
                                    let b_span = kb_end - kb0 + 1;

                                    let w_fwd = block_swap_proposal_weight(
                                        &rep.at, tables, ia, ib, l, m_start_f,
                                    );
                                    let before = two_disjoint_spans_energy(
                                        &rep.at, e_u, e_l, ka0, a_span, kb0, b_span,
                                    );
                                    for off in 0..l {
                                        let ka = tables.free_k[ia + off] as usize;
                                        let kb = tables.free_k[ib + off] as usize;
                                        rep.at.swap(ka, kb);
                                    }
                                    let after = two_disjoint_spans_energy(
                                        &rep.at, e_u, e_l, ka0, a_span, kb0, b_span,
                                    );
                                    let w_rev = block_swap_proposal_weight(
                                        &rep.at, tables, ia, ib, l, m_start_f,
                                    );
                                    let de = after - before;
                                    let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
                                    if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
                                        for off in 0..l {
                                            let ka = tables.free_k[ia + off] as usize;
                                            let kb = tables.free_k[ib + off] as usize;
                                            let sa_new = rep.at[ka];
                                            let sb_new = rep.at[kb];
                                            if sa_new >= 0 {
                                                rep.pos[sa_new as usize] = ka as i32;
                                            }
                                            if sb_new >= 0 {
                                                rep.pos[sb_new as usize] = kb as i32;
                                            }
                                        }
                                        rep.cur_e += de;
                                        stats.n_accepted += 1;
                                    } else {
                                        for off in 0..l {
                                            let ka = tables.free_k[ia + off] as usize;
                                            let kb = tables.free_k[ib + off] as usize;
                                            rep.at.swap(ka, kb);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if rep.cur_e < rep.best_e {
            rep.best_e = rep.cur_e;
            rep.best_at.copy_from_slice(&rep.at);
        }

        if step >= burn_in {
            stats.n_samples += 1;
            let delta = rep.cur_e - stats.mean_e;
            stats.mean_e += delta / (stats.n_samples as f64);
            let delta2 = rep.cur_e - stats.mean_e;
            stats.m2_e += delta * delta2;
        }
    }

    stats
}

const P_RESTORE_EVICT_BAYES: f64 = 0.35;
const P_EVICT_WHEN_SWAP_OK: f64 = 0.18;

#[inline]
fn evict_slot_weight(
    tables: &BayesMoveTables,
    displaced_spin: i32,
    slot_m: usize,
    n_fk_f: f64,
) -> f64 {
    let mut w = 0.50 / n_fk_f;
    if in_list8(&tables.top_k8, displaced_spin, slot_m as i32) {
        w += 0.50 / 8.0;
    }
    w
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_mh_spin_slot_move(
    rep: &mut CpuReplicaState,
    e_u: &Matrix2D,
    e_l: &Matrix2D,
    tables: &BayesMoveTables,
    k: usize,
    s: usize,
    r_glob: f64,
    r_cand: f64,
    r_ext: f64,
    inv_t: f64,
    stats: &mut ReplicaStageStats,
) {
    let s_i32 = s as i32;
    let t = rep.at[k];
    let ks = rep.pos[s];
    let n_fk = tables.free_k.len();
    let n_fk_f = n_fk as f64;

    if t == s_i32 {
        // Case 1: Delete s from slot k to null (-1); neighbor slots unchanged => log_q == 0
        let before = site_energy(&rep.at, e_u, e_l, k);
        rep.at[k] = -1;
        let de = -before;
        if de <= 0.0 || rep.rng.f64() < (-de * inv_t).exp() {
            rep.pos[s] = -1;
            rep.cur_e += de;
            stats.n_accepted += 1;
        } else {
            rep.at[k] = s_i32;
        }
    } else if t == -1 && ks == -1 {
        // Case 2: Insert unassigned s into empty slot k; neighbor slots unchanged => log_q == 0
        if e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
            return;
        }
        let before = site_energy(&rep.at, e_u, e_l, k);
        rep.at[k] = s_i32;
        let after = site_energy(&rep.at, e_u, e_l, k);
        let de = after - before;
        if de <= 0.0 || rep.rng.f64() < (-de * inv_t).exp() {
            rep.pos[s] = k as i32;
            rep.cur_e += de;
            stats.n_accepted += 1;
        } else {
            rep.at[k] = -1;
        }
    } else if t >= 0 && ks == -1 {
        if e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
            return;
        }
        let t_u = t as usize;
        if n_fk >= 2 && rep.rng.f64() < P_RESTORE_EVICT_BAYES {
            // Case 3b (RestoreEvict): place unassigned s into k AND relocate displaced t into an empty
            // slot m (50% guided by top_k8[t], 50% uniform free_k), exact bijection of Case 5b (Evict).
            let m_i32 = if rep.rng.f64() < 0.50 {
                let j = rep.rng.usize(8);
                tables.top_k8[t_u * 8 + j]
            } else {
                tables.free_k[rep.rng.usize(n_fk)] as i32
            };
            if m_i32 < 0 {
                return;
            }
            let m = m_i32 as usize;
            if m == k
                || tables.unlocked_k_flag[m] == 0
                || rep.at[m] >= 0
                || e_u.get(t_u, m) >= FORBID_ENERGY * 0.5
            {
                return;
            }
            let w_fwd = pair_proposal_weight(&rep.at, tables, k, s_i32, r_glob, r_cand, r_ext)
                * P_RESTORE_EVICT_BAYES
                * evict_slot_weight(tables, t, m, n_fk_f);
            let before = two_site_energy(&rep.at, e_u, e_l, k, m);
            rep.at[k] = s_i32;
            rep.at[m] = t;
            let after = two_site_energy(&rep.at, e_u, e_l, k, m);
            let p_ev_rev = if e_u.get(s, m) < FORBID_ENERGY * 0.5 {
                P_EVICT_WHEN_SWAP_OK
            } else {
                1.0
            };
            let w_rev =
                pair_proposal_weight(&rep.at, tables, k, t, r_glob, r_cand, r_ext) * p_ev_rev;
            let de = after - before;
            let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
            if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
                rep.pos[s] = k as i32;
                rep.pos[t_u] = m as i32;
                rep.cur_e += de;
                stats.n_accepted += 1;
            } else {
                rep.at[k] = t;
                rep.at[m] = -1;
            }
        } else {
            // Case 3a: Replace t at slot k with unassigned s (sending t to null -1)
            let w_fwd = pair_proposal_weight(&rep.at, tables, k, s_i32, r_glob, r_cand, r_ext);
            let w_rev = pair_proposal_weight(&rep.at, tables, k, t, r_glob, r_cand, r_ext);
            let before = site_energy(&rep.at, e_u, e_l, k);
            rep.at[k] = s_i32;
            let after = site_energy(&rep.at, e_u, e_l, k);
            let de = after - before;
            let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
            if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
                rep.pos[t_u] = -1;
                rep.pos[s] = k as i32;
                rep.cur_e += de;
                stats.n_accepted += 1;
            } else {
                rep.at[k] = t;
            }
        }
    } else if t == -1 && ks >= 0 {
        // Case 4: Relocate s from ks to empty slot k
        let ks_u = ks as usize;
        if tables.unlocked_k_flag[ks_u] == 0 || e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
            return;
        }
        let w_fwd = pair_proposal_weight(&rep.at, tables, k, s_i32, r_glob, r_cand, r_ext);
        let before = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
        rep.at[ks_u] = -1;
        rep.at[k] = s_i32;
        let after = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
        let w_rev = pair_proposal_weight(&rep.at, tables, ks_u, s_i32, r_glob, r_cand, r_ext);
        let de = after - before;
        let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
        if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
            rep.pos[s] = k as i32;
            rep.cur_e += de;
            stats.n_accepted += 1;
        } else {
            rep.at[ks_u] = s_i32;
            rep.at[k] = -1;
        }
    } else if t >= 0 && ks >= 0 {
        let ks_u = ks as usize;
        let t_u = t as usize;
        if ks_u == k || tables.unlocked_k_flag[ks_u] == 0 || e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
            return;
        }
        let swap_ok = e_u.get(t_u, ks_u) < FORBID_ENERGY * 0.5;
        let do_evict = !swap_ok || (n_fk >= 2 && rep.rng.f64() < P_EVICT_WHEN_SWAP_OK);
        if do_evict {
            // Case 5b (Evict): Relocate s from ks -> k and evict t -> -1 (leaving ks empty -1),
            // exact reverse bijection of Case 3b (RestoreEvict) displacing s_i32 into ks_u.
            if n_fk < 2 {
                return;
            }
            let p_ev = if swap_ok { P_EVICT_WHEN_SWAP_OK } else { 1.0 };
            let w_fwd =
                pair_proposal_weight(&rep.at, tables, k, s_i32, r_glob, r_cand, r_ext) * p_ev;
            let before = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
            rep.at[ks_u] = -1;
            rep.at[k] = s_i32;
            let after = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
            let w_rev = pair_proposal_weight(&rep.at, tables, k, t, r_glob, r_cand, r_ext)
                * P_RESTORE_EVICT_BAYES
                * evict_slot_weight(tables, s_i32, ks_u, n_fk_f);
            let de = after - before;
            let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
            if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
                rep.pos[s] = k as i32;
                rep.pos[t_u] = -1;
                rep.cur_e += de;
                stats.n_accepted += 1;
            } else {
                rep.at[ks_u] = s_i32;
                rep.at[k] = t;
            }
        } else {
            // Case 5a: 2-Swap s (at ks) <-> t (at k)
            let w_fwd = pair_proposal_weight(&rep.at, tables, k, s_i32, r_glob, r_cand, r_ext)
                + pair_proposal_weight(&rep.at, tables, ks_u, t, r_glob, r_cand, r_ext);
            let before = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
            rep.at[k] = s_i32;
            rep.at[ks_u] = t;
            let after = two_site_energy(&rep.at, e_u, e_l, k, ks_u);
            let w_rev = pair_proposal_weight(&rep.at, tables, k, t, r_glob, r_cand, r_ext)
                + pair_proposal_weight(&rep.at, tables, ks_u, s_i32, r_glob, r_cand, r_ext);
            let de = after - before;
            let log_acc = -de * inv_t + (w_rev / w_fwd).ln();
            if log_acc >= 0.0 || rep.rng.f64() < log_acc.exp() {
                rep.pos[s] = k as i32;
                rep.pos[t_u] = ks;
                rep.cur_e += de;
                stats.n_accepted += 1;
            } else {
                rep.at[k] = t;
                rep.at[ks_u] = s_i32;
            }
        }
    }
}

// ============================================================================
// Ensemble Stage Aggregation, Marginal Posterior Calibration & QC Finalization
// ============================================================================

/// Aggregate per-replica intra-stage Welford moments `(mean_e, m2_e, n_samples, n_accepted, n_proposed, min_best_e)`
/// into a [`BayesSaTempStep`].
pub fn aggregate_stage_telemetry(
    stage: usize,
    temperature: f64,
    steps_per_replica: usize,
    replica_stats: &[(f64, f64, usize, usize, usize, f64)],
) -> BayesSaTempStep {
    if replica_stats.is_empty() {
        return BayesSaTempStep {
            stage,
            temperature,
            steps_per_replica,
            mean_energy: 0.0,
            energy_variance: 0.0,
            ensemble_spread_variance: 0.0,
            specific_heat_cv: 0.0,
            acceptance_rate: 0.0,
            min_replica_energy: 0.0,
        };
    }

    let r_cnt = replica_stats.len() as f64;
    let mean_energy = replica_stats.iter().map(|s| s.0).sum::<f64>() / r_cnt;
    let mut sum_intra_var = 0.0_f64;
    let mut sum_spread = 0.0_f64;
    let mut tot_acc = 0usize;
    let mut tot_prop = 0usize;
    let mut min_best = f64::INFINITY;

    for &(m_r, m2_r, n_samp, n_acc, n_prop, best_r) in replica_stats {
        let v_r = if n_samp > 0 {
            (m2_r / (n_samp as f64)).max(0.0)
        } else {
            0.0
        };
        sum_intra_var += v_r;
        sum_spread += (m_r - mean_energy).powi(2);
        tot_acc += n_acc;
        tot_prop += n_prop;
        if best_r < min_best {
            min_best = best_r;
        }
    }

    let energy_variance = (sum_intra_var / r_cnt).max(0.0);
    let ensemble_spread_variance = (sum_spread / r_cnt).max(0.0);
    let t2 = (temperature * temperature).max(1e-12);
    let specific_heat_cv = (energy_variance / t2).max(0.0);
    let acceptance_rate = if tot_prop > 0 {
        (tot_acc as f64) / (tot_prop as f64)
    } else {
        0.0
    };

    BayesSaTempStep {
        stage,
        temperature,
        steps_per_replica,
        mean_energy,
        energy_variance,
        ensemble_spread_variance,
        specific_heat_cv,
        acceptance_rate,
        min_replica_energy: if min_best.is_finite() { min_best } else { 0.0 },
    }
}

/// Check whether post-peak low-$T$ thermodynamic freezing criteria have been met.
#[inline]
pub fn check_post_peak_convergence(
    cfg: &BayesSaConfig,
    stage: usize,
    next_temp: f64,
    active_slots: usize,
    max_cv_seen: f64,
    max_cv_stage: usize,
    low_fluct_streak: usize,
) -> bool {
    if cfg.cooling_schedule == CoolingScheduleKind::FixedGeometric
        || cfg.fixed_total_steps.is_some()
    {
        return false;
    }
    if next_temp <= cfg.t_min * 1.000_001 && stage + 1 >= cfg.min_stages {
        return true;
    }
    let n_act = (active_slots.max(1)) as f64;
    let passed_peak = max_cv_seen > 0.05 * n_act && stage >= max_cv_stage + 4;
    stage + 1 >= cfg.min_stages && passed_peak && low_fluct_streak >= 3 && next_temp <= 0.45
}

/// Convert multi-replica best assignments + energies into a calibrated `(AssignResult, BayesSaReport)`
/// for CPU (`is_gpu_engine = false`).
#[allow(clippy::too_many_arguments)]
pub fn finalize_bayes_sa_from_replicas(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
    matrices: &BayesSaMatrices,
    best_ats: &[Vec<i32>],
    best_energies: Vec<f64>,
    final_energies: Vec<f64>,
    cooling_trajectory: Vec<BayesSaTempStep>,
    converged_early: bool,
) -> (AssignResult, BayesSaReport) {
    finalize_bayes_sa_from_replicas_for_engine(
        peaks,
        systems,
        sequence,
        priors,
        cfg,
        matrices,
        best_ats,
        best_energies,
        final_energies,
        cooling_trajectory,
        converged_early,
        false,
    )
}

/// Convert multi-replica best assignments + energies into a calibrated `(AssignResult, BayesSaReport)`.
/// Shared by both `nmr-core::bayes_sa` (`is_gpu_engine = false`) and
/// `nmr-gpu::bayes_sa_gpu` (`is_gpu_engine = true`).
#[allow(clippy::too_many_arguments)]
pub fn finalize_bayes_sa_from_replicas_for_engine(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
    matrices: &BayesSaMatrices,
    best_ats: &[Vec<i32>],
    best_energies: Vec<f64>,
    final_energies: Vec<f64>,
    cooling_trajectory: Vec<BayesSaTempStep>,
    converged_early: bool,
    is_gpu_engine: bool,
) -> (AssignResult, BayesSaReport) {
    let s_len = systems.len();
    let k_len = matrices.e_unary.cols;

    if s_len == 0 || k_len == 0 || best_ats.is_empty() {
        let empty_res = AssignResult {
            position: vec![-1; s_len],
            confidence: vec![0.0; s_len],
            score: 0.0,
            run_scores: Vec::new(),
        };
        let qc = if cfg.apply_qc_calibration {
            evaluate_dataset_qc(
                peaks,
                systems,
                sequence,
                priors,
                &cfg.to_qc_score_config(),
                &empty_res,
                Some(&matrices.detailed_offsets),
            )
        } else {
            DatasetQcReport::default()
        };
        let rep = BayesSaReport {
            best_energy: 0.0,
            final_mean_energy: 0.0,
            final_energy_variance: 0.0,
            replica_best_energies: Vec::new(),
            replica_final_energies: Vec::new(),
            t_crit: cfg.t_start,
            max_cv: 0.0,
            total_stages: cooling_trajectory.len(),
            total_steps_per_replica: cooling_trajectory.iter().map(|s| s.steps_per_replica).sum(),
            converged_early,
            mu_null: matrices.mu_null,
            applied_offsets: matrices.applied_offsets,
            detailed_offsets: matrices.detailed_offsets,
            qc_report: qc,
            cooling_trajectory,
        };
        return (empty_res, rep);
    }

    // Recompute exact f64 energies of every replica's best_at so CPU and GPU rank identically
    let mut exact_best_energies = Vec::with_capacity(best_ats.len());
    let mut positions = Vec::with_capacity(best_ats.len());
    for (idx, at) in best_ats.iter().enumerate() {
        let mut pos = vec![-1i32; s_len];
        for (k, &s_raw) in at.iter().enumerate() {
            if s_raw >= 0 && (s_raw as usize) < s_len {
                pos[s_raw as usize] = k as i32;
            }
        }
        let exact_e = evaluate_state_energy(at, matrices)
            .unwrap_or_else(|_| best_energies.get(idx).copied().unwrap_or(0.0));
        exact_best_energies.push(exact_e);
        positions.push(pos);
    }

    let winner_idx = exact_best_energies
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut best_energy = exact_best_energies[winner_idx];
    let mut winner_at = best_ats[winner_idx].clone();
    let mut winner_pos = positions[winner_idx].clone();

    // Synthesize Minimum-Bayes-Risk (MBR) ensemble consensus candidate across all R replicas
    // and perform exact zero-temperature coordinate descent so sub-chains solved in complementary
    // replicas combine cleanly on large 450–800 aa proteins.
    if best_ats.len() >= 2 {
        let mbr_at = synthesize_and_polish_mbr_consensus(
            &positions,
            &exact_best_energies,
            &winner_at,
            matrices,
        );
        if let Ok(mbr_e) = evaluate_state_energy(&mbr_at, matrices) {
            if mbr_e < best_energy - 1e-9 {
                best_energy = mbr_e;
                winner_at = mbr_at;
                winner_pos.fill(-1);
                for (k, &s_raw) in winner_at.iter().enumerate() {
                    if s_raw >= 0 && (s_raw as usize) < s_len {
                        winner_pos[s_raw as usize] = k as i32;
                    }
                }
            }
        }
    }

    let run_scores: Vec<f64> = exact_best_energies.iter().map(|&e| -e).collect();

    let (mut confidence, c_rep) = compute_bayes_marginal_confidence(
        &winner_at,
        &winner_pos,
        &positions,
        &exact_best_energies,
        matrices,
    );

    // Bayes-optimal abstention on unlinked singleton placements whose marginal posterior or
    // Laplace-regularized replica agreement is low, preventing spurious cherry-picking of empty
    // missing-root sequence slots by noise strips while keeping winner_at and best_energy exact.
    if cfg.enable_weak_root_penalty {
        let mut modified_winner = false;
        for s in 0..s_len {
            let k_raw = winner_pos[s];
            if k_raw <= 0 || (confidence[s] >= 0.48 && c_rep[s] >= 0.34) {
                continue;
            }
            let k = k_raw as usize;
            let is_locked = matrices
                .resolved_locks
                .iter()
                .any(|&(sl, kl)| sl == s && kl == k);
            if is_locked {
                continue;
            }
            let linked_left = k > 1
                && winner_at[k - 1] >= 0
                && matrices.e_link.get(winner_at[k - 1] as usize, s) <= -2.0;
            let linked_right = k + 1 < k_len
                && winner_at[k + 1] >= 0
                && matrices.e_link.get(s, winner_at[k + 1] as usize) <= -2.0;
            let strong_link = (linked_left && linked_right)
                || (linked_left && matrices.e_link.get(winner_at[k - 1] as usize, s) <= -4.5)
                || (linked_right && matrices.e_link.get(s, winner_at[k + 1] as usize) <= -4.5);
            if (!linked_left && !linked_right)
                || (!strong_link && confidence[s] < 0.22 && c_rep[s] < 0.20)
                || confidence[s] < 0.08
            {
                winner_pos[s] = -1;
                if winner_at[k] == s as i32 {
                    winner_at[k] = -1;
                }
                confidence[s] = 0.0;
                modified_winner = true;
            }
        }
        if modified_winner {
            if let Ok(re_e) = evaluate_state_energy(&winner_at, matrices) {
                best_energy = re_e;
            }
        }
    }

    let mut result = AssignResult {
        position: winner_pos,
        confidence,
        score: -best_energy,
        run_scores,
    };

    if cfg.apply_qc_calibration {
        for s in 0..s_len {
            if result.position[s] >= 0 {
                result.confidence[s] = if is_gpu_engine {
                    apply_bayes_sa_gpu_calibration_map(result.confidence[s])
                } else {
                    apply_bayes_sa_calibration_map(result.confidence[s])
                };
            } else {
                result.confidence[s] = 0.0;
            }
        }
    }

    // Enforce hard locked assignments prior to and following QC evaluation
    for &(s_lock, k_lock) in &matrices.resolved_locks {
        if s_lock < s_len && k_lock < k_len {
            result.position[s_lock] = k_lock as i32;
            result.confidence[s_lock] = 1.0;
        }
    }

    let mut qc_report = evaluate_dataset_qc(
        peaks,
        systems,
        sequence,
        priors,
        &cfg.to_qc_score_config(),
        &result,
        Some(&matrices.detailed_offsets),
    );
    if cfg.apply_qc_calibration {
        let fit_runs = if is_gpu_engine {
            BAYES_SA_GPU_CALIBRATED_RUNS
        } else {
            BAYES_SA_CPU_CALIBRATED_RUNS
        };
        if best_ats.len() != fit_runs {
            let engine_lbl = if is_gpu_engine {
                "WebGPU (bayes-sa)"
            } else {
                "CPU (f64, bayes-sa)"
            };
            qc_report.mark_uncalibrated_runs(engine_lbl, best_ats.len(), fit_runs);
        }
        apply_dataset_qc_to_assignments(&mut result, &qc_report);
        qc_report.already_applied = true;
        for &(s_lock, k_lock) in &matrices.resolved_locks {
            if s_lock < s_len && k_lock < k_len {
                result.position[s_lock] = k_lock as i32;
                result.confidence[s_lock] = 1.0;
            }
        }
    }

    // Extract cooperative phase-transition peak (T_crit, max_cv) across active thermal stages
    let mut max_cv = 0.0_f64;
    let mut t_crit = cooling_trajectory
        .first()
        .map(|s| s.temperature)
        .unwrap_or(cfg.t_start);
    let any_active_stage = cooling_trajectory
        .iter()
        .any(|s| s.acceptance_rate >= 0.01 && s.temperature > cfg.t_min * 1.15);
    for step in &cooling_trajectory {
        let eligible = !any_active_stage
            || (step.acceptance_rate >= 0.01 && step.temperature > cfg.t_min * 1.15);
        if eligible && step.specific_heat_cv > max_cv {
            max_cv = step.specific_heat_cv;
            t_crit = step.temperature;
        }
    }

    let n_fin = final_energies.len().max(1) as f64;
    let final_mean_energy = final_energies.iter().sum::<f64>() / n_fin;
    let final_energy_variance = final_energies
        .iter()
        .map(|&e| (e - final_mean_energy).powi(2))
        .sum::<f64>()
        / n_fin;
    let total_steps_per_replica = cooling_trajectory.iter().map(|s| s.steps_per_replica).sum();

    let report = BayesSaReport {
        best_energy,
        final_mean_energy,
        final_energy_variance,
        replica_best_energies: exact_best_energies,
        replica_final_energies: final_energies,
        t_crit,
        max_cv,
        total_stages: cooling_trajectory.len(),
        total_steps_per_replica,
        converged_early,
        mu_null: matrices.mu_null,
        applied_offsets: matrices.applied_offsets,
        detailed_offsets: matrices.detailed_offsets,
        qc_report,
        cooling_trajectory,
    };

    (result, report)
}

/// Synthesize a Minimum-Bayes-Risk (MBR) consensus assignment across all annealed replicas and
/// apply deterministic zero-temperature (`T = 0`) exact pseudo-energy polish (`O(K * S)`).
fn synthesize_and_polish_mbr_consensus(
    replica_positions: &[Vec<i32>],
    replica_energies: &[f64],
    seed_winner_at: &[i32],
    matrices: &BayesSaMatrices,
) -> Vec<i32> {
    let e_u = &matrices.e_unary;
    let e_l = &matrices.e_link;
    let s_len = e_u.rows;
    let k_len = e_u.cols;
    if s_len == 0 || k_len <= 1 || replica_positions.is_empty() {
        return seed_winner_at.to_vec();
    }

    let e_min = replica_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let n_rep = replica_positions.len() as f64;
    let tau_ens = (0.22 * (matrices.active_slots.max(1) as f64)).max(10.0);

    let mut boltz_w = Vec::with_capacity(replica_energies.len());
    let mut w_sum = 0.0_f64;
    for &er in replica_energies {
        let w = (-(er - e_min) / tau_ens).clamp(-30.0, 0.0).exp();
        boltz_w.push(w);
        w_sum += w;
    }

    // Sparse vote tally over (s, k) pairs actually visited by replicas
    let mut votes: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    for (r_idx, rp) in replica_positions.iter().enumerate() {
        let bw = boltz_w[r_idx];
        for (s, &k_raw) in rp.iter().enumerate() {
            if k_raw > 0 && (k_raw as usize) < k_len {
                let entry = votes.entry((s, k_raw as usize)).or_insert((0.0, 0.0));
                entry.0 += 1.0;
                entry.1 += bw;
            }
        }
    }

    let mut locked_s = vec![false; s_len];
    let mut locked_k = vec![false; k_len];
    let mut mbr_at = vec![-1i32; k_len];
    let mut mbr_pos = vec![-1i32; s_len];
    for &(sl, kl) in &matrices.resolved_locks {
        if sl < s_len && kl < k_len {
            locked_s[sl] = true;
            locked_k[kl] = true;
            mbr_at[kl] = sl as i32;
            mbr_pos[sl] = kl as i32;
        }
    }

    let mut ranked_pairs: Vec<(usize, usize, f64)> = votes
        .into_iter()
        .filter_map(|((s, k), (cnt, w_acc))| {
            if locked_s[s] || locked_k[k] || e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
                return None;
            }
            let f_u = cnt / n_rep;
            let f_b = w_acc / (w_sum + 2.0);
            let p_ens = 0.50 * f_u + 0.50 * f_b;
            let min_vote = if matrices.active_slots >= 180 {
                0.05
            } else {
                0.14
            };
            (p_ens >= min_vote).then_some((s, k, p_ens - 0.004 * e_u.get(s, k)))
        })
        .collect();
    ranked_pairs.sort_by(|a, b| b.2.total_cmp(&a.2));

    for (s, k, _) in ranked_pairs {
        if mbr_pos[s] < 0 && mbr_at[k] < 0 {
            mbr_at[k] = s as i32;
            mbr_pos[s] = k as i32;
        }
    }

    polish_bayes_state_zero_temp(&mut mbr_at, &mut mbr_pos, &locked_s, &locked_k, e_u, e_l);

    let mut win_at = seed_winner_at.to_vec();
    let mut win_pos = vec![-1i32; s_len];
    for (k, &s_raw) in win_at.iter().enumerate() {
        if s_raw >= 0 && (s_raw as usize) < s_len {
            win_pos[s_raw as usize] = k as i32;
        }
    }
    polish_bayes_state_zero_temp(&mut win_at, &mut win_pos, &locked_s, &locked_k, e_u, e_l);

    let e_mbr = evaluate_state_energy(&mbr_at, matrices).unwrap_or(f64::INFINITY);
    let e_win = evaluate_state_energy(&win_at, matrices).unwrap_or(f64::INFINITY);
    if e_mbr < e_win {
        mbr_at
    } else {
        win_at
    }
}

/// Deterministic zero-temperature (`T = 0`) coordinate descent over single-site insertions/deletions,
/// pairwise 2-swaps / relocations, and connected-stretch shifts/swaps, strictly preserving locks.
fn polish_bayes_state_zero_temp(
    at: &mut [i32],
    pos: &mut [i32],
    locked_s: &[bool],
    locked_k: &[bool],
    e_u: &Matrix2D,
    e_l: &Matrix2D,
) {
    let s_len = pos.len();
    let k_len = at.len();
    if s_len == 0 || k_len <= 1 {
        return;
    }

    for _pass in 0..4 {
        let mut improved = false;

        // 1. Single-slot unassigned spin insertion, deletion to null, or replacement
        for k in 1..k_len {
            if locked_k[k] {
                continue;
            }
            let cur_s = at[k];
            let cur_e = site_energy(at, e_u, e_l, k);
            let mut best_cand = cur_s;
            let mut best_de = -1e-9_f64;

            if cur_s >= 0 && -cur_e < best_de {
                best_de = -cur_e;
                best_cand = -1;
            }
            for s in 0..s_len {
                if pos[s] >= 0 || locked_s[s] || e_u.get(s, k) >= FORBID_ENERGY * 0.5 {
                    continue;
                }
                let mut cand_e = e_u.get(s, k);
                if k > 0 && at[k - 1] >= 0 {
                    cand_e += e_l.get(at[k - 1] as usize, s);
                }
                if k + 1 < k_len && at[k + 1] >= 0 {
                    cand_e += e_l.get(s, at[k + 1] as usize);
                }
                let de = cand_e - cur_e;
                if de < best_de {
                    best_de = de;
                    best_cand = s as i32;
                }
            }
            if best_cand != cur_s {
                if cur_s >= 0 {
                    pos[cur_s as usize] = -1;
                }
                at[k] = best_cand;
                if best_cand >= 0 {
                    pos[best_cand as usize] = k as i32;
                }
                improved = true;
            }
        }

        // 2. Pairwise 2-swap / relocation across unlocked non-Proline slots
        for k1 in 1..k_len {
            if locked_k[k1] {
                continue;
            }
            for k2 in (k1 + 1)..k_len {
                if locked_k[k2] {
                    continue;
                }
                let s1 = at[k1];
                let s2 = at[k2];
                if s1 == s2 {
                    continue;
                }
                if (s1 >= 0 && e_u.get(s1 as usize, k2) >= FORBID_ENERGY * 0.5)
                    || (s2 >= 0 && e_u.get(s2 as usize, k1) >= FORBID_ENERGY * 0.5)
                {
                    continue;
                }
                let before = two_site_energy(at, e_u, e_l, k1, k2);
                at[k1] = s2;
                at[k2] = s1;
                let after = two_site_energy(at, e_u, e_l, k1, k2);
                if after - before < -1e-9 {
                    if s1 >= 0 {
                        pos[s1 as usize] = k2 as i32;
                    }
                    if s2 >= 0 {
                        pos[s2 as usize] = k1 as i32;
                    }
                    improved = true;
                } else {
                    at[k1] = s1;
                    at[k2] = s2;
                }
            }
        }

        // 3. Connected-stretch non-overlapping swap to relocate mis-registered islands cleanly
        let mut k = 1usize;
        while k < k_len {
            if at[k] < 0 || locked_k[k] {
                k += 1;
                continue;
            }
            let start = k;
            while k + 1 < k_len
                && at[k + 1] >= 0
                && !locked_k[k + 1]
                && e_l.get(at[k] as usize, at[k + 1] as usize) <= -2.0
            {
                k += 1;
            }
            let len = k - start + 1;
            if len >= 2 && len * 2 <= k_len {
                let mut best_dst: Option<usize> = None;
                let mut best_de = -1e-9_f64;
                for dst in 1..=(k_len - len) {
                    if dst.abs_diff(start) < len {
                        continue;
                    }
                    let mut ok = true;
                    for off in 0..len {
                        let ka = start + off;
                        let kb = dst + off;
                        if locked_k[kb] {
                            ok = false;
                            break;
                        }
                        let sa = at[ka];
                        let sb = at[kb];
                        if (sa >= 0 && e_u.get(sa as usize, kb) >= FORBID_ENERGY * 0.5)
                            || (sb >= 0 && e_u.get(sb as usize, ka) >= FORBID_ENERGY * 0.5)
                        {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let before = two_disjoint_spans_energy(at, e_u, e_l, start, len, dst, len);
                    for off in 0..len {
                        at.swap(start + off, dst + off);
                    }
                    let after = two_disjoint_spans_energy(at, e_u, e_l, start, len, dst, len);
                    for off in 0..len {
                        at.swap(start + off, dst + off);
                    }
                    let de = after - before;
                    if de < best_de {
                        best_de = de;
                        best_dst = Some(dst);
                    }
                }
                if let Some(dst) = best_dst {
                    for off in 0..len {
                        let ka = start + off;
                        let kb = dst + off;
                        at.swap(ka, kb);
                        if at[ka] >= 0 {
                            pos[at[ka] as usize] = ka as i32;
                        }
                        if at[kb] >= 0 {
                            pos[at[kb] as usize] = kb as i32;
                        }
                    }
                    improved = true;
                }
            }
            k += 1;
        }

        if !improved {
            break;
        }
    }
}

/// Compute calibrated unit-temperature (`T = 1.0` nat) marginal assignment posteriors combining
/// Laplace-regularized Boltzmann-weighted replica consensus with conditional single-site and 2-swap
/// partition competition. Returns `(confidence, c_rep)`.
fn compute_bayes_marginal_confidence(
    at: &[i32],
    pos: &[i32],
    replica_positions: &[Vec<i32>],
    replica_energies: &[f64],
    matrices: &BayesSaMatrices,
) -> (Vec<f64>, Vec<f64>) {
    let s_len = pos.len();
    let k_len = at.len();
    let mut conf = vec![0.0_f64; s_len];
    let mut c_rep = vec![0.0_f64; s_len];
    if s_len == 0 || k_len <= 1 || replica_positions.is_empty() {
        return (conf, c_rep);
    }

    let e_min = replica_energies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let n_rep = replica_positions.len() as f64;
    let tau_ens = (0.22 * (matrices.active_slots.max(1) as f64)).max(10.0);

    // Blend uniform replica agreement with Laplace-smoothed elite-basin agreement so a single
    // outlier trajectory cannot inflate consensus on incomplete or multi-modal landscapes.
    let mut boltz_w = Vec::with_capacity(replica_energies.len());
    let mut w_sum = 0.0_f64;
    for &er in replica_energies {
        let w = (-(er - e_min) / tau_ens).clamp(-30.0, 0.0).exp();
        boltz_w.push(w);
        w_sum += w;
    }

    for s in 0..s_len {
        if pos[s] < 0 {
            continue;
        }
        let target_k = pos[s];
        let mut raw_agree = 0.0_f64;
        let mut weighted_agree = 0.0_f64;
        for (r_idx, rp) in replica_positions.iter().enumerate() {
            if rp[s] == target_k {
                raw_agree += 1.0;
                weighted_agree += boltz_w[r_idx];
            }
        }
        let f_unif = raw_agree / n_rep;
        let f_boltz = (weighted_agree / (w_sum + 2.0)).clamp(0.0, 1.0);
        c_rep[s] = (0.50 * f_unif + 0.50 * f_boltz).clamp(0.0, 1.0);
    }

    let e_u = &matrices.e_unary;
    let e_l = &matrices.e_link;

    // Identify connected sequential stretches (`E_link(at[k], at[k+1]) <= -1.5`) and cooperative
    // multi-residue island support for large multi-domain sequences where independent replicas
    // nucleate complementary subsets of sequential islands before MBR synthesis.
    let mut chain_len = vec![1usize; s_len];
    let mut eff_rep_base = c_rep.clone();
    {
        let mut k = 1usize;
        while k < k_len {
            if at[k] < 0 {
                k += 1;
                continue;
            }
            let start = k;
            let mut neg_link_sum = 0.0_f64;
            while k + 1 < k_len && at[k + 1] >= 0 {
                let el = e_l.get(at[k] as usize, at[k + 1] as usize);
                if el <= -1.5 {
                    neg_link_sum += -el;
                    k += 1;
                } else {
                    break;
                }
            }
            let clen = k - start + 1;
            let mut sum_r = 0.0_f64;
            let mut max_r = 0.0_f64;
            for &sp in &at[start..=k] {
                let su = sp as usize;
                chain_len[su] = clen;
                sum_r += c_rep[su];
                max_r = max_r.max(c_rep[su]);
            }
            if clen >= 4 && max_r >= 0.06 {
                let mean_r = sum_r / (clen as f64);
                let mean_neg_link = neg_link_sum / ((clen - 1) as f64);
                let stretch_e = interval_energy(at, e_u, e_l, start, clen);
                if stretch_e < -(1.5 * (clen as f64)) && mean_neg_link >= 3.5 {
                    let mut island_floor = 0.75 * mean_r;
                    if matrices.active_slots >= 180 && mean_r >= 0.06 && mean_neg_link >= 4.0 {
                        let coop = if mean_r >= 0.14 {
                            (0.56 + 0.06 * ((clen - 4).min(5) as f64)).min(0.86)
                        } else {
                            (0.38 + 0.06 * ((clen - 4).min(5) as f64)).min(0.72)
                        };
                        island_floor = island_floor.max(coop);
                    }
                    for &sp in &at[start..=k] {
                        let su = sp as usize;
                        eff_rep_base[su] = eff_rep_base[su].max(island_floor);
                    }
                }
            }
            k += 1;
        }
    }

    let col_assignable: Vec<bool> = (0..k_len)
        .map(|k| k > 0 && (0..s_len).any(|s| e_u.get(s, k) < FORBID_ENERGY * 0.5))
        .collect();

    let clamp_w = |spin_idx: usize| -> f64 {
        let r = (eff_rep_base[spin_idx] / 0.85).clamp(0.0, 1.0);
        r * r
    };

    let weighted_site_e = |spin: usize, slot: usize| -> f64 {
        let mut e = e_u.get(spin, slot);
        if slot > 0 && at[slot - 1] >= 0 {
            let sl = at[slot - 1] as usize;
            e += clamp_w(sl) * e_l.get(sl, spin);
        }
        if slot + 1 < k_len && at[slot + 1] >= 0 {
            let sr = at[slot + 1] as usize;
            e += clamp_w(sr) * e_l.get(spin, sr);
        }
        e
    };

    for s in 0..s_len {
        let k_raw = pos[s];
        if k_raw < 0 || eff_rep_base[s] <= 0.0 {
            continue;
        }
        let k = k_raw as usize;
        let e_self = weighted_site_e(s, k);

        // Partition sum over competing hypotheses (lower energy = higher exp(-(E_alt - E_self)))
        // 1. Null/unassigned alternative (`E_null = 0`, so `E_null - E_self = -e_self`)
        let mut alt_part = e_self.clamp(-40.0, 20.0).exp();

        // Singleton caution when linkable neighbors exist in sequence
        let can_link = (k > 1 && col_assignable[k - 1]) || (k + 1 < k_len && col_assignable[k + 1]);
        if can_link && chain_len[s] == 1 {
            alt_part += 0.22;
        }

        // 2. Unassigned spin systems competing for slot k
        for (t, &pos_t) in pos.iter().enumerate() {
            if t != s && pos_t < 0 && e_u.get(t, k) < FORBID_ENERGY * 0.5 {
                let e_t = weighted_site_e(t, k);
                alt_part += (e_self - e_t).clamp(-40.0, 20.0).exp();
            }
        }

        // 3. Alternative slots k2 != k (relocation into empty/weakly-held slot or 2-swap)
        for k2 in 1..k_len {
            if k2 == k || !col_assignable[k2] || e_u.get(s, k2) >= FORBID_ENERGY * 0.5 {
                continue;
            }
            let mut e_s_k2 = e_u.get(s, k2);
            if k2 > 0 && k2 - 1 != k && at[k2 - 1] >= 0 {
                let sl = at[k2 - 1] as usize;
                e_s_k2 += clamp_w(sl) * e_l.get(sl, s);
            }
            if k2 + 1 < k_len && k2 + 1 != k && at[k2 + 1] >= 0 {
                let sr = at[k2 + 1] as usize;
                e_s_k2 += clamp_w(sr) * e_l.get(s, sr);
            }
            let occ_protect = if at[k2] >= 0 {
                let t = at[k2] as usize;
                clamp_w(t) * (-weighted_site_e(t, k2)).max(0.0)
            } else {
                0.0
            };
            let mut best_gain = e_self - (e_s_k2 + occ_protect);

            if at[k2] >= 0 {
                let t = at[k2] as usize;
                if e_u.get(t, k) < FORBID_ENERGY * 0.5 {
                    let (a, sa, b, sb) = if k < k2 { (k, s, k2, t) } else { (k2, t, k, s) };
                    let mut e_before = e_u.get(sa, a) + e_u.get(sb, b);
                    let mut e_after = e_u.get(sb, a) + e_u.get(sa, b);
                    if a > 0 && at[a - 1] >= 0 {
                        let sl = at[a - 1] as usize;
                        let w = clamp_w(sl);
                        e_before += w * e_l.get(sl, sa);
                        e_after += w * e_l.get(sl, sb);
                    }
                    if b + 1 < k_len && at[b + 1] >= 0 {
                        let sr = at[b + 1] as usize;
                        let w = clamp_w(sr);
                        e_before += w * e_l.get(sb, sr);
                        e_after += w * e_l.get(sa, sr);
                    }
                    if a + 1 == b {
                        e_before += e_l.get(sa, sb);
                        e_after += e_l.get(sb, sa);
                    } else {
                        if a + 1 < k_len && at[a + 1] >= 0 {
                            let ma = at[a + 1] as usize;
                            let w = clamp_w(ma);
                            e_before += w * e_l.get(sa, ma);
                            e_after += w * e_l.get(sb, ma);
                        }
                        if b > 0 && at[b - 1] >= 0 {
                            let mb = at[b - 1] as usize;
                            let w = clamp_w(mb);
                            e_before += w * e_l.get(mb, sb);
                            e_after += w * e_l.get(mb, sa);
                        }
                    }
                    best_gain = best_gain.max(e_before - e_after);
                }
            }
            alt_part += best_gain.clamp(-40.0, 20.0).exp();
        }

        let p_local = 1.0 / (1.0 + alt_part);
        let base_r = eff_rep_base[s];
        let eff_c_rep = if chain_len[s] >= 3 && base_r >= 0.42 && p_local >= 0.92 {
            let boost = 0.06 * ((chain_len[s] - 2).min(4) as f64);
            (base_r + boost).clamp(0.0, 0.995)
        } else {
            base_r
        };
        let p_joint = (eff_c_rep * p_local).clamp(0.0, 1.0);
        let p_min = eff_c_rep.min(p_local);
        conf[s] = (0.5 * (p_joint + p_min)).clamp(0.0, 1.0);
    }

    (conf, c_rep)
}

// ============================================================================
// Public CPU Solver Entry Points
// ============================================================================

/// Run the Bayesian pseudo-energy and specific-heat ($C_v(T)$) adaptive simulated annealing
/// backbone solver on assembled `SpinSystem`s (`R = 32` parallel replicas on CPU by default).
pub fn solve_bayes_sa(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> (AssignResult, BayesSaReport) {
    solve_bayes_sa_with_peaks(None, systems, sequence, priors, cfg)
}

/// State machine driving the adaptive $C_v(T_m)$ or fixed-geometric temperature schedule across stages.
/// Shared by both `nmr-core::bayes_sa` (CPU) and `nmr-gpu::bayes_sa_gpu` (WebGPU).
#[derive(Debug, Clone)]
pub struct BayesScheduleDriver {
    pub temp: f64,
    pub stage: usize,
    pub stage_steps: usize,
    pub max_stages: usize,
    pub base_stage_steps: usize,
    pub active_slots: usize,
    pub done: bool,
    pub converged_early: bool,
    pub steps_used: usize,
    pub max_cv_seen: f64,
    pub max_cv_stage: usize,
    pub low_fluct_streak: usize,
    pub rep_stats: Vec<(f64, f64, usize, usize, usize, f64)>,
    pub trajectory: Vec<BayesSaTempStep>,
}

impl BayesScheduleDriver {
    pub fn new(cfg: &BayesSaConfig, active_slots: usize, num_replicas: usize) -> Self {
        let max_stages = cfg.max_stages.max(1);
        let base_stage_steps = default_steps_per_stage(cfg, active_slots);
        let first_steps = if let Some(tot) = cfg.fixed_total_steps {
            base_stage_steps.min(tot.max(1))
        } else {
            base_stage_steps
        };
        Self {
            temp: cfg.t_start.max(cfg.t_min),
            stage: 0,
            stage_steps: first_steps,
            max_stages,
            base_stage_steps,
            active_slots,
            done: max_stages == 0,
            converged_early: false,
            steps_used: 0,
            max_cv_seen: 0.0,
            max_cv_stage: 0,
            low_fluct_streak: 0,
            rep_stats: vec![(0.0, 0.0, 0, 0, 0, 0.0); num_replicas.max(1)],
            trajectory: Vec::with_capacity(max_stages),
        }
    }

    pub fn advance_after_stage(&mut self, cfg: &BayesSaConfig) {
        let step_rec =
            aggregate_stage_telemetry(self.stage, self.temp, self.stage_steps, &self.rep_stats);
        let cv = step_rec.specific_heat_cv;
        let acc = step_rec.acceptance_rate;
        if cv > self.max_cv_seen {
            self.max_cv_seen = cv;
            self.max_cv_stage = self.stage;
        }
        let n_act = (self.active_slots.max(1)) as f64;
        if cv / n_act <= cfg.convergence_cv_tol && acc <= cfg.convergence_acc_tol {
            self.low_fluct_streak += 1;
        } else {
            self.low_fluct_streak = 0;
        }

        let next_t = next_stage_temperature(
            cfg,
            self.stage,
            self.max_stages,
            self.temp,
            cv,
            self.active_slots,
        );
        let used = self.steps_used + self.stage_steps;
        self.steps_used = used;
        self.trajectory.push(step_rec);

        let next_stage = self.stage + 1;
        if let Some(fixed_tot) = cfg.fixed_total_steps {
            if used >= fixed_tot || next_stage >= self.max_stages {
                self.done = true;
                self.converged_early = false;
                return;
            }
            let rem_stages = (self.max_stages - next_stage).max(1);
            let next_steps = if rem_stages == 1 {
                fixed_tot - used
            } else {
                self.base_stage_steps.min(fixed_tot - used).max(1)
            };
            self.stage = next_stage;
            self.temp = next_t;
            self.stage_steps = next_steps;
            return;
        }

        if next_stage >= self.max_stages {
            self.done = true;
            self.converged_early = false;
            return;
        }
        if check_post_peak_convergence(
            cfg,
            self.stage,
            next_t,
            self.active_slots,
            self.max_cv_seen,
            self.max_cv_stage,
            self.low_fluct_streak,
        ) {
            self.done = true;
            self.converged_early = true;
            return;
        }
        self.stage = next_stage;
        self.temp = next_t;
        self.stage_steps = self.base_stage_steps;
    }
}

/// Variant of [`solve_bayes_sa`] forwarding optional raw `peaks` into `DatasetQcReport` evaluation.
pub fn solve_bayes_sa_with_peaks(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> (AssignResult, BayesSaReport) {
    let matrices = compute_bayes_energy_matrices(systems, sequence, priors, cfg);
    if systems.is_empty() || sequence.is_empty() {
        return finalize_bayes_sa_from_replicas(
            peaks,
            systems,
            sequence,
            priors,
            cfg,
            &matrices,
            &[],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            false,
        );
    }

    let tables = build_bayes_move_tables(&matrices, cfg);
    let r_cnt = cfg.replicas.max(1);
    let mut master_rng = Rng::new(cfg.seed);
    let (init_ats, init_poss, init_es, sub_seeds) =
        build_initial_bayes_replicas(&matrices, &tables, r_cnt, &mut master_rng);

    let mut replicas: Vec<CpuReplicaState> = (0..r_cnt)
        .map(|r| CpuReplicaState {
            best_at: init_ats[r].clone(),
            best_e: init_es[r],
            at: init_ats[r].clone(),
            pos: init_poss[r].clone(),
            cur_e: init_es[r],
            rng: Rng::new(sub_seeds[r]),
        })
        .collect();

    let shared = Mutex::new(BayesScheduleDriver::new(cfg, matrices.active_slots, r_cnt));

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(r_cnt);

    if n_threads <= 1 {
        loop {
            let (cur_stage, cur_temp, cur_steps, is_done) = {
                let st = shared.lock().unwrap();
                (st.stage, st.temp, st.stage_steps, st.done)
            };
            if is_done {
                break;
            }
            let mut local_tuples = Vec::with_capacity(r_cnt);
            for rep in &mut replicas {
                let s = run_cpu_replica_stage(
                    rep,
                    &matrices,
                    &tables,
                    cur_stage,
                    cur_temp,
                    cur_steps,
                    cfg.max_fragment_len,
                );
                local_tuples.push((
                    s.mean_e,
                    s.m2_e,
                    s.n_samples,
                    s.n_accepted,
                    s.n_proposed,
                    rep.best_e,
                ));
            }
            let mut st = shared.lock().unwrap();
            st.rep_stats = local_tuples;
            st.advance_after_stage(cfg);
        }
    } else {
        let chunk_len = r_cnt.div_ceil(n_threads);
        let mut slices: Vec<(usize, &mut [CpuReplicaState])> = Vec::new();
        let mut rem = replicas.as_mut_slice();
        let mut offset = 0usize;
        while !rem.is_empty() {
            let take = chunk_len.min(rem.len());
            let (head, tail) = rem.split_at_mut(take);
            slices.push((offset, head));
            offset += take;
            rem = tail;
        }
        let actual_workers = slices.len();
        let barrier = Barrier::new(actual_workers);
        let barrier_ref = &barrier;
        let shared_ref = &shared;
        let matrices_ref = &matrices;
        let tables_ref = &tables;

        std::thread::scope(|scope| {
            for (base_idx, rep_slice) in slices {
                scope.spawn(move || loop {
                    let (cur_stage, cur_temp, cur_steps, is_done) = {
                        let st = shared_ref.lock().unwrap();
                        (st.stage, st.temp, st.stage_steps, st.done)
                    };
                    if is_done {
                        break;
                    }
                    let mut local_out = Vec::with_capacity(rep_slice.len());
                    for rep in rep_slice.iter_mut() {
                        let s = run_cpu_replica_stage(
                            rep,
                            matrices_ref,
                            tables_ref,
                            cur_stage,
                            cur_temp,
                            cur_steps,
                            cfg.max_fragment_len,
                        );
                        local_out.push((
                            s.mean_e,
                            s.m2_e,
                            s.n_samples,
                            s.n_accepted,
                            s.n_proposed,
                            rep.best_e,
                        ));
                    }
                    {
                        let mut st = shared_ref.lock().unwrap();
                        for (i, val) in local_out.into_iter().enumerate() {
                            st.rep_stats[base_idx + i] = val;
                        }
                    }
                    let wait_res = barrier_ref.wait();
                    if wait_res.is_leader() {
                        let mut st = shared_ref.lock().unwrap();
                        st.advance_after_stage(cfg);
                    }
                    barrier_ref.wait();
                });
            }
        });
    }

    let advance_res = shared.into_inner().unwrap();
    let mut best_ats = Vec::with_capacity(r_cnt);
    let mut best_es = Vec::with_capacity(r_cnt);
    let mut final_es = Vec::with_capacity(r_cnt);
    for rep in replicas {
        best_ats.push(rep.best_at);
        best_es.push(rep.best_e);
        final_es.push(rep.cur_e);
    }

    finalize_bayes_sa_from_replicas(
        peaks,
        systems,
        sequence,
        priors,
        cfg,
        &matrices,
        &best_ats,
        best_es,
        final_es,
        advance_res.trajectory,
        advance_res.converged_early,
    )
}

/// Clean 3D triple-resonance `F1` truncation sinc sidelobes, `t1` thermal noise ridges, and
/// out-of-band `13C` artifacts while keeping **every `HSQC` peak in exact 1-to-1 order**
/// (`systems.len() == hsqc_count`). Legacy unheighted peak lists pass through with only physical
/// spectral-band checks.
pub fn condition_peaks_preserving_hsqc_roots(peaks: &[Peak], tol: &Tolerances) -> Vec<Peak> {
    let any_3d_heights = peaks.iter().any(|p| p.exp != "HSQC" && p.height.is_some());
    let hsqc_roots: Vec<(f64, f64)> = peaks
        .iter()
        .filter(|p| p.exp == "HSQC")
        .map(|p| (p.h, p.n))
        .collect();

    // 1. Physical 13C spectral-band filter on 3D peaks (never touches HSQC roots)
    let mut band_kept: Vec<usize> = Vec::with_capacity(peaks.len());
    for (idx, p) in peaks.iter().enumerate() {
        if p.exp == "HSQC" {
            band_kept.push(idx);
            continue;
        }
        if let Some(c) = p.c {
            let exp_u = p.exp.as_str();
            let valid_band = match exp_u {
                "HNCO" | "HNCACO" => (158.0..=190.0).contains(&c),
                "HNCA" | "HNCOCA" => (35.5..=76.5).contains(&c),
                "HNCACB" => {
                    if p.sign >= 0 {
                        (35.5..=76.5).contains(&c)
                    } else {
                        (10.5..=80.0).contains(&c)
                    }
                }
                "CBCACONH" => (10.5..=80.0).contains(&c),
                _ => true,
            };
            if !valid_band {
                continue;
            }
        }
        band_kept.push(idx);
    }

    if !any_3d_heights || hsqc_roots.is_empty() {
        return band_kept.into_iter().map(|i| peaks[i].clone()).collect();
    }

    let inv_th2 = 1.0 / (tol.h.max(1e-4) * tol.h.max(1e-4));
    let inv_tn2 = 1.0 / (tol.n.max(1e-3) * tol.n.max(1e-3));

    // Group heighted 3D peaks onto their nearest HSQC root strip to reject F1 sinc sidelobes & t1 ridges
    let mut drop_mask = vec![false; peaks.len()];
    let mut strip_groups: BTreeMap<(usize, String, i32), Vec<usize>> = BTreeMap::new();

    for &idx in &band_kept {
        let p = &peaks[idx];
        if p.exp == "HSQC" || p.height.is_none() || p.c.is_none() {
            continue;
        }
        let mut best_r: Option<(usize, f64)> = None;
        for (r_idx, &(rh, rn)) in hsqc_roots.iter().enumerate() {
            let dh = (p.h - rh).abs();
            let dn = (p.n - rn).abs();
            if dh <= tol.h && dn <= tol.n {
                let d2 = dh * dh * inv_th2 + dn * dn * inv_tn2;
                if best_r.is_none_or(|(_, b)| d2 < b) {
                    best_r = Some((r_idx, d2));
                }
            }
        }
        if let Some((r_idx, _)) = best_r {
            let phase_key = if p.exp == "HNCACB" {
                if p.sign >= 0 {
                    1
                } else {
                    -1
                }
            } else {
                1
            };
            strip_groups
                .entry((r_idx, p.exp.clone(), phase_key))
                .or_default()
                .push(idx);
        }
    }

    // Also suppress opposite-phase F1 sinc sidelobes in HNCACB between (+) and (-) on the same strip
    let mut hncacb_strip_all: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&(r_idx, ref exp, _), indices) in &strip_groups {
        if exp == "HNCACB" {
            hncacb_strip_all
                .entry(r_idx)
                .or_default()
                .extend(indices.iter().copied());
        }
    }
    for indices in hncacb_strip_all.values() {
        for &i_ch in indices {
            let hc = peaks[i_ch].height.unwrap_or(0.0).abs();
            let cc = peaks[i_ch].c.unwrap_or(0.0);
            for &i_pa in indices {
                if i_ch == i_pa {
                    continue;
                }
                let hp = peaks[i_pa].height.unwrap_or(0.0).abs();
                let cp = peaks[i_pa].c.unwrap_or(0.0);
                let dc = (cc - cp).abs();
                if (0.75..=2.25).contains(&dc) && hc < 0.085 * hp {
                    drop_mask[i_ch] = true;
                    break;
                }
            }
        }
    }

    for ((_, exp, _), mut indices) in strip_groups {
        indices.retain(|&i| !drop_mask[i]);
        if indices.len() <= 1 {
            continue;
        }
        indices.sort_by(|&a, &b| {
            let ha = peaks[a].height.unwrap_or(0.0).abs();
            let hb = peaks[b].height.unwrap_or(0.0).abs();
            hb.total_cmp(&ha)
        });
        let top_h = peaks[indices[0]].height.unwrap_or(0.0).abs();
        let is_co = exp == "HNCO" || exp == "HNCACO";
        let min_frac = if exp == "HNCO" { 0.08 } else { 0.055 };
        let max_keep = match exp.as_str() {
            "HNCO" | "HNCOCA" => 2usize,
            "HNCA" | "HNCACO" | "CBCACONH" | "HNCACB" => 4usize,
            _ => 6usize,
        };

        let mut kept_in_strip: Vec<usize> = Vec::with_capacity(max_keep);
        for &idx in &indices {
            let h_cur = peaks[idx].height.unwrap_or(0.0).abs();
            if h_cur < min_frac * top_h {
                drop_mask[idx] = true;
                continue;
            }
            let c_cur = peaks[idx].c.unwrap_or(0.0);
            let is_sidelobe = kept_in_strip.iter().any(|&k_prev| {
                let h_prev = peaks[k_prev].height.unwrap_or(0.0).abs();
                let c_prev = peaks[k_prev].c.unwrap_or(0.0);
                let dc = (c_cur - c_prev).abs();
                if is_co {
                    (0.14..=0.72).contains(&dc) && h_cur < 0.14 * h_prev
                } else {
                    (0.65..=2.25).contains(&dc) && h_cur < 0.095 * h_prev
                }
            });
            if is_sidelobe || kept_in_strip.len() >= max_keep {
                drop_mask[idx] = true;
            } else {
                kept_in_strip.push(idx);
            }
        }
    }

    band_kept
        .into_iter()
        .filter(|&i| !drop_mask[i])
        .map(|i| peaks[i].clone())
        .collect()
}

/// End-to-end Bayesian SA backbone assignment directly from picked `Peak`s (conditioning 3D
/// spectral artifacts while preserving 1-to-1 `HSQC` root indexing, solving, and evaluating
/// `DatasetQcReport`).
pub fn solve_bayes_sa_from_peaks(
    peaks: &[Peak],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> Result<(Vec<SpinSystem>, AssignResult, BayesSaReport), String> {
    let conditioned = if cfg.enable_weak_root_penalty {
        condition_peaks_preserving_hsqc_roots(peaks, &cfg.tolerances)
    } else {
        peaks.to_vec()
    };
    let systems = build_spin_systems(&conditioned, &cfg.tolerances, cfg.max_cands, false)?;
    let (result, report) = solve_bayes_sa_with_peaks(Some(peaks), &systems, sequence, priors, cfg);
    Ok((systems, result, report))
}
