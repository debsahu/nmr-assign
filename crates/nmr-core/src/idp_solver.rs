//! Continuous through-Proline `13C`-direct-detected IDP backbone resonance assignment solver,
//! temperature- & pH-aware `ncIDP` neighbor-corrected random-coil chemical-shift library,
//! unlabelled permutation-invariant reference offset estimator, multi-island evolutionary +
//! detailed-balance simulated annealing, and calibrated `ProteinReliabilityCurve` QC (`Phase 11`, `T11.2`).
//!
//! Provenance: Clean-room implementation derived strictly from peer-reviewed biomolecular NMR literature:
//! - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn.
//!   Reson. Spectrosc.* 2006, 48, 25–45; *J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem.
//!   Soc.* 2009, 131, 15339–15345 — protonless `2D CON`, `2D CACO`, `3D CANCO`, `CBCACON`,
//!   `CCC(CO)N`, and `NCACO` continuous through-Proline IDP backbone assignment).
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP`
//!   sequence-neighbor random-coil chemical shift library for intrinsically disordered proteins).
//! - Kjaergaard, M., Brander, S. & Poulsen, F. M. (*J. Biomol. NMR* 2011, 49, 139–149) and
//!   Kjaergaard, M. & Poulsen, F. M. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57 —
//!   temperature and sequence effects on random-coil chemical shifts in disordered proteins).
//! - Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* 2014, 60, 109–129 — pH-dependent
//!   random-coil `1H`, `13C`, and `15N` chemical shifts of ionizable residues).
//! - Schwarzinger, S. et al. (*J. Am. Chem. Soc.* 2001, 123, 2970–2978) and Wishart, D. S. et al.
//!   (*J. Biomol. NMR* 1995, 5, 67–81 — pre-Proline and sidechain aliphatic carbon shifts).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::thread;

use serde::{Deserialize, Serialize};

use crate::c13_detected::{C13DetectedExp, C13PickedDataset, CacoPeakKind};
use crate::nmrstar::Reference;
use crate::priors::one_to_three;
use crate::qc::{DatasetQcVerdict, ProteinReliabilityCurve};
use crate::rng::Rng;
use crate::solver::{AssignResult, Matrix2D, FORBID};

/// Half-forbidden sentinel (`-5e8`) matching WebGPU `idp_scoring.wgsl`.
pub const IDP_FORBID_HALF: f32 = -5.0e8;

/// Maximum per-channel candidate shifts packed per `ConDipeptideSystem` for CPU/WebGPU scoring.
pub const MAX_IDP_CANDS: usize = 6;

/// Maximum sidechain `(i-1)` aliphatic `13C` TOCSY peaks (`CCC(CO)N`) packed per system.
pub const MAX_IDP_SC_CANDS: usize = 6;

/// Peptide-bond dipeptide spin system anchored on `2D CON(i)` (`13C'(i-1)` – `15N(i)`),
/// active across ALL residue boundaries `k = 1..L-1` including `-Xaa-Pro-` and `-Pro-Pro-`!
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConDipeptideSystem {
    /// 0-based spin-system identifier (`0..S`).
    pub id: usize,
    /// Observed `15N(i)` chemical shift in ppm (`~105–131 ppm` amide, `~132–142 ppm` `Pro` imino).
    pub n_curr: f64,
    /// Observed `13C'(i-1)` chemical shift in ppm (`~168–181 ppm`, direct IPAP).
    pub c_prev: f64,
    /// Relative `2D CON` root intensity.
    pub root_height: f64,
    /// Candidate intraresidue `13CA(i)` shifts in ppm (`CANCO` / `NCACO`).
    pub ca_curr: Vec<f64>,
    /// Candidate preceding `13CA(i-1)` shifts in ppm (`CBCACON` / `CANCO` / `CCC(CO)N` / `CACO`).
    pub ca_prev: Vec<f64>,
    /// Candidate preceding `13CB(i-1)` shifts in ppm (`CBCACON` / `CCC(CO)N`).
    pub cb_prev: Vec<f64>,
    /// Candidate intraresidue `13C'(i)` shifts in ppm (`NCACO` / `CACO`), linking into `CON(i+1).c_prev`.
    pub c_curr: Vec<f64>,
    /// Candidate sequential `15N(i+1)` shifts in ppm linking into `CON(i+1).n_curr`.
    pub n_next: Vec<f64>,
    /// Additional preceding `(i-1)` aliphatic sidechain carbons (`CG*, CD*, CE*` from `CCC(CO)N`).
    pub sidechain_c_prev: Vec<f64>,
    /// Unlabelled constant-time `2D CACO` opposite-phase `Gly` indicator at `c_prev`.
    pub prev_gly_opposite_phase: bool,
    /// True when `CBCACON` strip evidence was observed/acquired for this dataset.
    pub has_cbcacon_evidence: bool,
}

impl ConDipeptideSystem {
    pub fn new(id: usize, c_prev: f64, n_curr: f64, root_height: f64) -> Self {
        Self {
            id,
            n_curr,
            c_prev,
            root_height,
            ca_curr: Vec::new(),
            ca_prev: Vec::new(),
            cb_prev: Vec::new(),
            c_curr: Vec::new(),
            n_next: Vec::new(),
            sidechain_c_prev: Vec::new(),
            prev_gly_opposite_phase: false,
            has_cbcacon_evidence: false,
        }
    }

    #[inline]
    pub fn has_3d_support(&self) -> bool {
        !self.ca_curr.is_empty()
            || !self.ca_prev.is_empty()
            || !self.cb_prev.is_empty()
            || !self.c_curr.is_empty()
            || !self.sidechain_c_prev.is_empty()
    }
}

/// Global unlabelled chemical-shift referencing offsets `(dN, dC', dCA, dCB)` in ppm
/// (`observed ≈ predicted_ncidp + offset`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IdpShiftOffsets {
    pub d_n: f64,
    pub d_c: f64,
    pub d_ca: f64,
    pub d_cb: f64,
}

/// Predicted sequence-specific random-coil shifts and coil dispersion widths for one residue `k`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NcIdpResiduePrediction {
    pub residue: String,
    pub n_ppm: f64,
    pub sigma_n: f64,
    pub c_ppm: f64,
    pub sigma_c: f64,
    pub ca_ppm: f64,
    pub sigma_ca: f64,
    pub cb_ppm: Option<f64>,
    pub sigma_cb: f64,
    /// Expected non-`CA`/`CB` aliphatic sidechain carbons (`CG*, CD*, CE*`) in `[10.0, 56.0] ppm`.
    pub sidechain_c: Vec<f64>,
}

/// Solver & scoring configuration for `13C`-detected IDP backbone assignment (`CPU` & `WebGPU`).
#[derive(Clone, Debug)]
pub struct IdpSolverConfig {
    /// Sample temperature in Kelvin (`default = 298.15`).
    pub temperature_k: f64,
    /// Sample pH (`default = 6.5`).
    pub ph: f64,
    /// Automatically estimate and apply unlabelled `(dN, dC', dCA, dCB)` reference offsets (`default = true`).
    pub auto_reference_offset: bool,
    /// Optional explicit offsets applied when `auto_reference_offset == false`.
    pub manual_offsets: IdpShiftOffsets,
    /// Core `ncIDP` random-coil standard deviations `(ppm)`.
    pub sigma_ca_rc: f64,
    pub sigma_cb_rc: f64,
    pub sigma_c_rc: f64,
    pub sigma_n_rc: f64,
    pub sigma_pro_n_rc: f64,
    /// Sequential inter-system matching tolerances `(ppm)` in `P(s, t)`.
    pub sigma_seq_ca: f64,
    pub sigma_seq_c: f64,
    pub sigma_seq_n: f64,
    /// Per-atom unary LLR floor (`default = -10.0`).
    pub llr_floor: f64,
    /// Sequential pair score floor (`default = -6.0`).
    pub pair_floor: f64,
    /// Number of trajectories (`replicas`, `default = 48`).
    pub replicas: usize,
    /// Number of evolutionary islands on CPU (`default = 4`).
    pub islands: usize,
    /// Number of evolutionary crossover generations on CPU (`default = 3`).
    pub generations: usize,
    /// MCMC steps per spin system (`default = 2_400`).
    pub steps_per_system: usize,
    pub t_start: f64,
    pub t_end: f64,
    /// High-confidence threshold for QC / `ProteinReliabilityCurve` (`default = 0.80`).
    pub hc_threshold: f64,
    pub seed: u64,
}

impl Default for IdpSolverConfig {
    fn default() -> Self {
        Self {
            temperature_k: 298.15,
            ph: 6.5,
            auto_reference_offset: true,
            manual_offsets: IdpShiftOffsets::default(),
            sigma_ca_rc: 0.45,
            sigma_cb_rc: 0.55,
            sigma_c_rc: 0.40,
            sigma_n_rc: 0.85,
            sigma_pro_n_rc: 1.15,
            sigma_seq_ca: 0.085,
            sigma_seq_c: 0.055,
            sigma_seq_n: 0.14,
            llr_floor: -5.5,
            pair_floor: -3.8,
            replicas: 48,
            islands: 4,
            generations: 3,
            steps_per_system: 2_400,
            t_start: 16.0,
            t_end: 0.08,
            hc_threshold: 0.80,
            seed: 42,
        }
    }
}

/// Complete IDP `13C`-detected assignment result, unlabelled QC verdict, and fitted offsets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdpAssignResult {
    /// Assigned sequence index `k in 1..L` for each `ConDipeptideSystem` (`-1` if unassigned).
    pub position: Vec<i32>,
    /// Monotone calibrated posterior confidence in `[0.0, 1.0]` for each `ConDipeptideSystem`.
    pub confidence: Vec<f64>,
    pub score: f64,
    pub run_scores: Vec<f64>,
    pub fitted_offsets: IdpShiftOffsets,
    pub qc_verdict: DatasetQcVerdict,
    pub support_3d_frac: f64,
    pub connectivity_frac: f64,
    pub qc_reasons: Vec<String>,
}

impl IdpAssignResult {
    /// Convert to standard `AssignResult` for interoperability with `evaluate::score` and `ProteinReliabilityCurve`.
    pub fn as_assign_result(&self) -> AssignResult {
        AssignResult {
            position: self.position.clone(),
            confidence: self.confidence.clone(),
            score: self.score,
            run_scores: self.run_scores.clone(),
        }
    }

    /// Evaluate calibrated `ProteinReliabilityCurve` against spin-system ground-truth indices `truth`.
    pub fn reliability_curve(&self, truth: &[i32], hc_threshold: f64) -> ProteinReliabilityCurve {
        ProteinReliabilityCurve::from_assign_result(&self.as_assign_result(), truth, hc_threshold)
    }
}

fn normalize_seq_3(sequence: &[String]) -> Vec<String> {
    sequence
        .iter()
        .map(|s| {
            let t = s.trim();
            if t.len() == 1 {
                one_to_three(t.chars().next().unwrap_or('X')).to_string()
            } else {
                t.to_ascii_uppercase()
            }
        })
        .collect()
}

// ============================================================================
// Temperature- & pH-Aware `ncIDP` Neighbor-Corrected Random-Coil Model
// ============================================================================

/// Baseline `ncIDP` random-coil `(N, C', CA, CB)` shifts in ppm (DSS scale, `298.15 K, pH 6.5`)
/// per Tamiola, Acar & Mulder (*J. Am. Chem. Soc.* 2010, 132, 18000–18003, Table 1) and
/// Kjaergaard & Poulsen (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57).
fn ncidp_base_shifts(aa: &str) -> (f64, f64, f64, Option<f64>) {
    match aa {
        "ALA" => (124.5, 177.8, 52.5, Some(19.1)),
        "ARG" => (121.3, 176.3, 56.0, Some(30.8)),
        "ASN" => (119.0, 175.2, 53.1, Some(38.9)),
        "ASP" => (120.7, 176.4, 54.2, Some(41.1)),
        "CYS" => (119.8, 174.6, 58.2, Some(28.0)),
        "GLN" => (120.3, 176.0, 55.8, Some(29.5)),
        "GLU" => (121.2, 176.6, 56.6, Some(30.2)),
        "GLY" => (109.7, 173.9, 45.2, None),
        "HIS" => (119.6, 175.2, 55.8, Some(29.8)),
        "ILE" => (121.2, 176.4, 61.1, Some(38.6)),
        "LEU" => (122.6, 177.4, 55.1, Some(42.3)),
        "LYS" => (121.6, 176.6, 56.3, Some(33.0)),
        "MET" => (120.6, 176.2, 55.4, Some(32.8)),
        "PHE" => (120.9, 175.8, 57.8, Some(39.6)),
        "PRO" => (136.8, 176.9, 63.2, Some(32.1)),
        "SER" => (116.5, 174.6, 58.3, Some(63.8)),
        "THR" => (115.0, 174.6, 61.8, Some(69.8)),
        "TRP" => (121.7, 176.3, 57.4, Some(29.6)),
        "TYR" => (120.8, 175.7, 57.9, Some(38.8)),
        "VAL" => (120.6, 176.3, 62.2, Some(32.8)),
        _ => (120.5, 176.0, 56.0, Some(33.0)),
    }
}

/// Preceding-neighbor `(i-1)` random-coil corrections `(d_n, d_c, d_ca, d_cb)` in ppm exerted by
/// `prev_aa` ($X_{i-1}$) onto residue $X_i$ (Tamiola, Acar & Mulder *JACS* 2010, Table 2;
/// Braun, Wider & Wüthrich *JACS* 1994), zero-centered across standard amino acids relative to
/// `ncidp_base_shifts` so global `(dN, dC', dCA, dCB)` referencing remains strictly unbiased.
fn ncidp_prev_neighbor_delta(prev_aa: &str, curr_aa: &str) -> (f64, f64, f64, f64) {
    let (raw_dn, raw_dc, raw_dca, dcb) = match prev_aa {
        "GLY" => (-0.95, 0.08, 0.12, 0.00),
        "ALA" => (-0.55, 0.05, 0.04, 0.00),
        "SER" => (0.20, -0.08, -0.05, 0.02),
        "THR" => (1.55, -0.18, -0.14, 0.04),
        "VAL" => (2.55, -0.25, -0.22, 0.05),
        "ILE" => (2.85, -0.26, -0.24, 0.05),
        "LEU" => (0.68, -0.06, -0.06, 0.00),
        "PRO" => (1.10, -0.22, -0.18, -0.05),
        "PHE" => (1.42, -0.18, -0.16, -0.06),
        "TYR" => (1.50, -0.18, -0.16, -0.06),
        "TRP" => (1.30, -0.16, -0.18, -0.08),
        "HIS" => (1.05, -0.14, -0.12, -0.03),
        "ASP" => (0.32, -0.05, -0.03, 0.02),
        "GLU" => (0.52, -0.04, -0.04, 0.02),
        "ASN" => (0.42, -0.08, -0.06, 0.00),
        "GLN" => (0.62, -0.06, -0.05, 0.00),
        "LYS" => (0.60, -0.06, -0.05, 0.00),
        "ARG" => (0.62, -0.08, -0.06, 0.00),
        "MET" => (0.60, -0.06, -0.05, 0.00),
        "CYS" => (0.80, -0.12, -0.08, 0.00),
        _ => (0.855, -0.104, -0.083, 0.0),
    };
    let mut dn = raw_dn - 0.855;
    let dc = raw_dc + 0.104;
    let dca = raw_dca + 0.083;
    // Specific `Xaa-Pro` tertiary-imide `15N_Pro` sensitivity (Bermel et al. *JACS* 2009; Tamiola et al. 2010)
    if curr_aa == "PRO" {
        dn = match prev_aa {
            "GLY" => -2.35,
            "ALA" => -0.90,
            "PRO" => -1.25,
            "SER" => -0.55,
            "THR" => 1.05,
            "VAL" => 1.85,
            "ILE" => 2.05,
            "PHE" | "TYR" | "TRP" => 0.80,
            _ => 0.75 * dn,
        };
    }
    (dn, dc, dca, dcb)
}

/// Following-neighbor `(i+1)` random-coil corrections `(d_n, d_c, d_ca, d_cb)` in ppm exerted by
/// `next_aa` ($X_{i+1}$) onto residue $X_i$, including strong steric/electronic pre-Proline shifts
/// when `next_aa == "PRO"` (Tamiola et al. *JACS* 2010; Schwarzinger et al. *JACS* 2001).
fn ncidp_next_neighbor_delta(curr_aa: &str, next_aa: &str) -> (f64, f64, f64, f64) {
    if next_aa == "PRO" {
        return match curr_aa {
            "GLY" => (1.45, -1.50, -1.05, 0.00),
            "PRO" => (0.90, -1.80, -1.80, 0.12),
            _ => (1.65, -1.95, -1.95, -0.35),
        };
    }
    match next_aa {
        "GLY" => (0.42, -0.18, 0.14, 0.00),
        "PHE" | "TYR" | "TRP" => (-0.25, -0.15, -0.14, -0.06),
        "ILE" | "VAL" | "THR" => (-0.11, -0.07, -0.04, 0.04),
        "ALA" => (-0.03, 0.13, 0.08, 0.00),
        _ => (0.05, 0.06, 0.03, 0.00),
    }
}

/// Henderson–Hasselbalch pH titration shifts `(d_n, d_c, d_ca, d_cb)` relative to `pH 6.5`
/// (Platzer, Okon & McIntosh *J. Biomol. NMR* 2014, 60, 109–129).
fn ncidp_ph_delta(aa: &str, ph: f64, is_n_term: bool, is_c_term: bool) -> (f64, f64, f64, f64) {
    let ph_c = ph.clamp(1.0, 11.0);
    let frac_prot = |pka: f64, p: f64| -> f64 { 1.0 / (1.0 + 10.0_f64.powf(p - pka)) };

    let (mut dn, mut dc, mut dca, mut dcb) = match aa {
        "ASP" => {
            let df = frac_prot(3.86, ph_c) - frac_prot(3.86, 6.5);
            (-0.70 * df, -1.80 * df, -1.90 * df, -2.90 * df)
        }
        "GLU" => {
            let df = frac_prot(4.34, ph_c) - frac_prot(4.34, 6.5);
            (-0.60 * df, -1.30 * df, -1.80 * df, -1.80 * df)
        }
        "HIS" => {
            let df = frac_prot(6.45, ph_c) - frac_prot(6.45, 6.5);
            (-0.80 * df, -1.20 * df, -df, -0.90 * df)
        }
        _ => (0.0, 0.0, 0.0, 0.0),
    };
    if is_c_term {
        // C-terminal carboxylate (+2.1 ppm C' at neutral pH vs internal; protonates below pH 3.6)
        let f_deprot = 1.0 - frac_prot(3.55, ph_c);
        dc += 2.10 * f_deprot - 0.80 * (1.0 - f_deprot);
        dca += 0.65 * f_deprot - 0.55 * (1.0 - f_deprot);
        dn += 1.20 * f_deprot;
    }
    if is_n_term {
        let f_prot = frac_prot(7.75, ph_c);
        dca -= 1.35 * f_prot;
        dc -= 1.65 * f_prot;
        dcb += 0.35 * f_prot;
    }
    (dn, dc, dca, dcb)
}

/// Linear temperature coefficients `(d_n, d_c, d_ca, d_cb)` in `ppm` for `(temperature_k - 298.15)`
/// (Kjaergaard, Brander & Poulsen *J. Biomol. NMR* 2011, 49, 139–149).
fn ncidp_temp_delta(aa: &str, temperature_k: f64) -> (f64, f64, f64, f64) {
    let dt = (temperature_k.clamp(270.0, 345.0)) - 298.15;
    if dt.abs() < 1e-6 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let slope_n = if aa == "PRO" { -0.008 } else { -0.020 };
    let slope_c = -0.0055;
    let slope_ca = match aa {
        "VAL" | "ILE" | "PRO" => 0.0020,
        "ALA" => -0.0030,
        _ => -0.0015,
    };
    let slope_cb = 0.0020;
    (dt * slope_n, dt * slope_c, dt * slope_ca, dt * slope_cb)
}

/// Expected non-`CA`/`CB` aliphatic sidechain `13C` random-coil shifts (`CG*, CD*, CE*` in ppm)
/// observable in `3D CCC(CO)N` (Wishart et al. *J. Biomol. NMR* 1995, 5, 67–81).
pub fn expected_aliphatic_sidechain_c(aa: &str) -> &'static [f64] {
    match aa {
        "PRO" => &[27.3, 50.4],
        "ILE" => &[13.0, 17.6, 27.3],
        "VAL" => &[20.9, 21.5],
        "LEU" => &[23.8, 24.9, 26.9],
        "THR" => &[21.5],
        "LYS" => &[24.8, 29.1, 42.0],
        "ARG" => &[27.2, 43.3],
        "MET" => &[17.0, 32.1],
        "GLU" => &[36.1],
        "GLN" => &[33.8],
        _ => &[],
    }
}

/// Predict temperature-, pH-, and sequence-neighbor-corrected `ncIDP` random-coil shifts across
/// `sequence[0..L]` (including `Pro` `15N` ~ `136.8 ppm` and pre-Proline corrections on `k - 1`).
pub fn predict_ncidp_shifts(
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Vec<NcIdpResiduePrediction> {
    let seq = normalize_seq_3(sequence);
    let len = seq.len();
    let mut out = Vec::with_capacity(len);

    for k in 0..len {
        let aa = seq[k].as_str();
        let (mut n_ppm, mut c_ppm, mut ca_ppm, mut cb_opt) = ncidp_base_shifts(aa);

        if k > 0 {
            let (dn, dc, dca, dcb) = ncidp_prev_neighbor_delta(&seq[k - 1], aa);
            n_ppm += dn;
            c_ppm += dc;
            ca_ppm += dca;
            if let Some(cb) = &mut cb_opt {
                *cb += dcb;
            }
            if k == 1 {
                // Inductive +NH3 effect of protonated N-terminal residue `0` on `15N(1)`
                let f_nt = 1.0 / (1.0 + 10.0_f64.powf(cfg.ph.clamp(1.0, 11.0) - 7.75));
                n_ppm += 2.20 * f_nt;
            }
        }
        if k + 1 < len {
            let (dn, dc, dca, dcb) = ncidp_next_neighbor_delta(aa, &seq[k + 1]);
            n_ppm += dn;
            c_ppm += dc;
            ca_ppm += dca;
            if let Some(cb) = &mut cb_opt {
                *cb += dcb;
            }
        }

        let (dn_ph, dc_ph, dca_ph, dcb_ph) = ncidp_ph_delta(aa, cfg.ph, k == 0, k + 1 == len);
        n_ppm += dn_ph;
        c_ppm += dc_ph;
        ca_ppm += dca_ph;
        if let Some(cb) = &mut cb_opt {
            *cb += dcb_ph;
        }

        let (dn_t, dc_t, dca_t, dcb_t) = ncidp_temp_delta(aa, cfg.temperature_k);
        n_ppm += dn_t;
        c_ppm += dc_t;
        ca_ppm += dca_t;
        if let Some(cb) = &mut cb_opt {
            *cb += dcb_t;
        }

        let sigma_n = if aa == "PRO" {
            cfg.sigma_pro_n_rc.max(0.40)
        } else {
            cfg.sigma_n_rc.max(0.35)
        };
        let sigma_cb = if aa == "CYS" {
            cfg.sigma_cb_rc.max(1.80)
        } else {
            cfg.sigma_cb_rc.max(0.25)
        };

        out.push(NcIdpResiduePrediction {
            residue: seq[k].clone(),
            n_ppm,
            sigma_n,
            c_ppm,
            sigma_c: cfg.sigma_c_rc.max(0.20),
            ca_ppm,
            sigma_ca: cfg.sigma_ca_rc.max(0.20),
            cb_ppm: cb_opt,
            sigma_cb,
            sidechain_c: expected_aliphatic_sidechain_c(aa).to_vec(),
        });
    }

    out
}

// ============================================================================
// Unlabelled Permutation-Invariant Global `(dN, dC', dCA, dCB)` Offset Estimator
// ============================================================================

/// Fit a 1D shift offset `d` maximizing the unlabelled sequence-marginal outlier-robust mixture
/// log-likelihood + Huber IRLS polish over canonically sorted observations `obs_sorted`.
fn fit_1d_unlabelled_marginal_offset(
    obs_sorted: &[f64],
    model_sorted: &[(f64, f64)],
    range_ppm: f64,
    max_abs_offset: f64,
    grid_step: f64,
) -> f64 {
    if obs_sorted.len() < 5 || model_sorted.is_empty() {
        return 0.0;
    }
    let inv_m = 1.0 / (model_sorted.len() as f64);
    let bg = 0.06 / range_ppm.max(4.0);
    let fg = 0.94;

    let eval_ll = |d: f64| -> f64 {
        let mut ll = -0.04 * d * d;
        for &v in obs_sorted {
            let mut density = 0.0f64;
            for &(mu, sd) in model_sorted {
                let s_eff = (sd * 1.15).max(0.30);
                let z = (v - (mu + d)) / s_eff;
                if z.abs() < 8.0 {
                    density += inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                }
            }
            ll += (fg * density + bg).max(1e-30).ln();
        }
        ll
    };

    let n_steps = ((2.0 * max_abs_offset) / grid_step).round() as usize;
    let mut scores = Vec::with_capacity(n_steps + 1);
    let mut best_idx = 0usize;
    let mut best_ll = f64::NEG_INFINITY;
    for i in 0..=n_steps {
        let d = -max_abs_offset + (i as f64) * grid_step;
        let sc = eval_ll(d);
        if sc > best_ll {
            best_ll = sc;
            best_idx = i;
        }
        scores.push(sc);
    }

    let base_ll = eval_ll(0.0);
    let mut d_est = -max_abs_offset + (best_idx as f64) * grid_step;
    if best_idx > 0 && best_idx < n_steps {
        let ym1 = scores[best_idx - 1];
        let y0 = scores[best_idx];
        let yp1 = scores[best_idx + 1];
        let den = ym1 - 2.0 * y0 + yp1;
        if den < -1e-12 {
            let frac = (0.5 * (ym1 - yp1) / den).clamp(-0.5, 0.5);
            d_est += frac * grid_step;
        }
    }

    // 4-step posterior-responsibility Huber IRLS polish (strictly unlabelled & order-independent)
    let huber_k = 1.345f64;
    for _ in 0..4 {
        let mut num = 0.0f64;
        let mut den = 1e-9f64;
        for &v in obs_sorted {
            let mut post_res_num = 0.0f64;
            let mut post_den = bg;
            for &(mu, sd) in model_sorted {
                let s_eff = (sd * 1.10).max(0.30);
                let r = v - (mu + d_est);
                let z = r / s_eff;
                if z.abs() < 7.0 {
                    let comp = fg * inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                    let huber_w = if z.abs() <= huber_k {
                        1.0
                    } else {
                        huber_k / z.abs()
                    };
                    let w = comp * huber_w / (s_eff * s_eff);
                    post_res_num += w * r;
                    post_den += comp;
                }
            }
            if post_den > 1e-24 {
                num += post_res_num / post_den;
                // Corresponding Fisher weight denominator
                let mut f_w = 0.0f64;
                for &(mu, sd) in model_sorted {
                    let s_eff = (sd * 1.10).max(0.30);
                    let r = v - (mu + d_est);
                    let z = r / s_eff;
                    if z.abs() < 7.0 {
                        let comp = fg * inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                        let huber_w = if z.abs() <= huber_k {
                            1.0
                        } else {
                            huber_k / z.abs()
                        };
                        f_w += comp * huber_w / (s_eff * s_eff);
                    }
                }
                den += f_w / post_den;
            }
        }
        let step = (num / den).clamp(-0.15, 0.15);
        let cand = (d_est + step).clamp(-max_abs_offset, max_abs_offset);
        if eval_ll(cand) >= eval_ll(d_est) - 1e-6 {
            d_est = cand;
        }
    }

    if eval_ll(d_est) > base_ll + 0.35 && d_est.abs() >= 0.012 {
        d_est
    } else {
        0.0
    }
}

/// Strictly unlabelled, bit-for-bit permutation-invariant global `(dN, dC', dCA, dCB)` offset
/// estimator matching unlabelled scalar peak distributions against the unlabelled `ncIDP`
/// sequence-marginal mixture.
///
/// Permuting the order of `systems` (or candidate vectors within any system) produces `0.0` change.
pub fn estimate_idp_reference_offsets(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> IdpShiftOffsets {
    if systems.len() < 5 || sequence.len() < 2 {
        return IdpShiftOffsets::default();
    }
    let preds = predict_ncidp_shifts(sequence, cfg);

    // 1. Canonically sorted unlabelled observations (guarantees 0.0 permutation sensitivity)
    let mut obs_n: Vec<f64> = systems
        .iter()
        .map(|s| s.n_curr)
        .filter(|v| v.is_finite() && (102.0..=145.0).contains(v))
        .collect();
    obs_n.sort_by(|a, b| a.total_cmp(b));

    let mut obs_c: Vec<f64> = systems
        .iter()
        .map(|s| s.c_prev)
        .filter(|v| v.is_finite() && (166.0..=183.5).contains(v))
        .collect();
    obs_c.sort_by(|a, b| a.total_cmp(b));

    let mut obs_ca: Vec<f64> = Vec::with_capacity(systems.len() * 2);
    let mut obs_cb: Vec<f64> = Vec::with_capacity(systems.len());
    for s in systems {
        let mut ca_s: Vec<f64> = s
            .ca_curr
            .iter()
            .chain(s.ca_prev.iter())
            .copied()
            .filter(|v| v.is_finite() && (40.0..=69.5).contains(v))
            .collect();
        ca_s.sort_by(|a, b| a.total_cmp(b));
        ca_s.dedup_by(|a, b| (*a - *b).abs() < 0.04);
        obs_ca.extend(ca_s);

        let mut cb_s: Vec<f64> = s
            .cb_prev
            .iter()
            .copied()
            .filter(|v| v.is_finite() && (13.0..=73.5).contains(v))
            .collect();
        cb_s.sort_by(|a, b| a.total_cmp(b));
        if let Some(&first_cb) = cb_s.first() {
            obs_cb.push(first_cb);
        }
    }
    obs_ca.sort_by(|a, b| a.total_cmp(b));
    obs_cb.sort_by(|a, b| a.total_cmp(b));

    // 2. Unlabelled sequence-marginal prior components sorted canonically
    let mut mod_n: Vec<(f64, f64)> = preds[1..].iter().map(|p| (p.n_ppm, p.sigma_n)).collect();
    mod_n.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_c: Vec<(f64, f64)> = preds[..preds.len() - 1]
        .iter()
        .map(|p| (p.c_ppm, p.sigma_c))
        .collect();
    mod_c.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_ca: Vec<(f64, f64)> = preds.iter().map(|p| (p.ca_ppm, p.sigma_ca)).collect();
    mod_ca.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_cb: Vec<(f64, f64)> = preds[..preds.len() - 1]
        .iter()
        .filter_map(|p| p.cb_ppm.map(|cb| (cb, p.sigma_cb)))
        .collect();
    mod_cb.sort_by(|a, b| a.0.total_cmp(&b.0));

    let d_n = fit_1d_unlabelled_marginal_offset(&obs_n, &mod_n, 38.0, 4.0, 0.02);
    let d_c = fit_1d_unlabelled_marginal_offset(&obs_c, &mod_c, 15.0, 4.0, 0.01);
    let d_ca = fit_1d_unlabelled_marginal_offset(&obs_ca, &mod_ca, 29.0, 4.0, 0.01);
    let d_cb = fit_1d_unlabelled_marginal_offset(&obs_cb, &mod_cb, 58.0, 4.0, 0.015);

    IdpShiftOffsets {
        d_n,
        d_c,
        d_ca,
        d_cb,
    }
}

// ============================================================================
// Unsupervised Inter-Experiment Peak Registration & Spin-System Assembly
// ============================================================================

/// Estimate 1D relative drift `delta` (`obs ≈ ref + delta`) between two unlabelled shift sets
/// via smooth Gaussian kernel cross-correlation over `[-max_shift, +max_shift]`.
fn fit_unlabelled_pairwise_drift(diffs_sorted: &[f64], max_shift: f64, bw: f64) -> f64 {
    if diffs_sorted.len() < 4 {
        return 0.0;
    }
    let eval = |d: f64| -> f64 {
        let mut s = 0.0f64;
        for &df in diffs_sorted {
            let z = (df - d) / bw;
            if z.abs() < 6.0 {
                s += (-0.5 * z * z).exp();
            }
        }
        s
    };
    let base = eval(0.0);
    let mut best_d = 0.0f64;
    let mut best_s = base;
    for step in -100i32..=100 {
        let d = (step as f64) * (max_shift / 100.0);
        let sc = eval(d);
        if sc > best_s {
            best_s = sc;
            best_d = d;
        }
    }
    let d0 = best_d;
    for step in -10i32..=10 {
        let d = (d0 + (step as f64) * 0.002).clamp(-max_shift, max_shift);
        let sc = eval(d);
        if sc > best_s {
            best_s = sc;
            best_d = d;
        }
    }
    if best_s > base * 1.05 && best_d.abs() >= 0.01 {
        best_d
    } else {
        0.0
    }
}

fn dedup_push(vec: &mut Vec<f64>, val: f64, tol: f64, max_len: usize) {
    if !val.is_finite() {
        return;
    }
    if let Some(existing) = vec.iter_mut().find(|&&mut x| (x - val).abs() <= tol) {
        *existing = 0.5 * (*existing + val);
    } else if vec.len() < max_len {
        vec.push(val);
    }
}

/// Register inter-experiment `(ΔC', ΔCA/CB)` carrier drift across `CON`, `CACO`, `CANCO`,
/// `CBCACON`, `CCC(CO)N`, and `NCACO` strictly without sequence/priors, and assemble
/// `ConDipeptideSystem` spin systems anchored on `picked.con_roots`.
pub fn register_and_build_con_dipeptide_systems(
    picked: &C13PickedDataset,
) -> Vec<ConDipeptideSystem> {
    let n_roots = picked.con_roots.len();
    if n_roots == 0 {
        return Vec::new();
    }

    // Group 3D strip peaks by `(exp, anchor_root_idx)`
    let mut canco_by_root: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_roots];
    let mut cbca_by_root: Vec<Vec<(f64, i32, f64)>> = vec![Vec::new(); n_roots];
    let mut cccon_by_root: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_roots];
    let mut ncaco_by_root: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); n_roots];

    for p in &picked.strip_peaks {
        if let Some(r_idx) = p.anchor_root_idx {
            if r_idx < n_roots {
                match p.exp {
                    C13DetectedExp::Canco3D => {
                        canco_by_root[r_idx].push((p.c_indirect_ppm, p.height.abs()));
                    }
                    C13DetectedExp::Cbcaco3D => {
                        cbca_by_root[r_idx].push((p.c_indirect_ppm, p.sign, p.height.abs()));
                    }
                    C13DetectedExp::Cccon3D => {
                        cccon_by_root[r_idx].push((p.c_indirect_ppm, p.height.abs()));
                    }
                    C13DetectedExp::Ncaco3D => {
                        ncaco_by_root[r_idx].push((
                            p.c_prime_ppm,
                            p.c_indirect_ppm,
                            p.height.abs(),
                        ));
                    }
                    C13DetectedExp::Con2D | C13DetectedExp::Caco2D => {}
                }
            }
        }
    }

    // 1. Align `CBCACON`, `CCC(CO)N`, and `NCACO` indirect aliphatic axes onto `CANCO`'s `13CA` frame
    let has_canco = canco_by_root.iter().any(|v| !v.is_empty());
    let mut diffs_cbca_vs_canco = Vec::new();
    let mut diffs_cccon_vs_ref = Vec::new();
    let mut diffs_ncaco_vs_canco = Vec::new();

    if has_canco {
        for r in 0..n_roots {
            for &(ca_cb, sgn, _) in &cbca_by_root[r] {
                if sgn >= 0 && (40.0..=69.5).contains(&ca_cb) {
                    for &(ca_cn, _) in &canco_by_root[r] {
                        let df = ca_cb - ca_cn;
                        if df.abs() <= 0.65 {
                            diffs_cbca_vs_canco.push(df);
                        }
                    }
                }
            }
            for &(_, ca_nc, _) in &ncaco_by_root[r] {
                for &(ca_cn, _) in &canco_by_root[r] {
                    let df = ca_nc - ca_cn;
                    if df.abs() <= 0.65 {
                        diffs_ncaco_vs_canco.push(df);
                    }
                }
            }
        }
    }
    diffs_cbca_vs_canco.sort_by(|a, b| a.total_cmp(b));
    diffs_ncaco_vs_canco.sort_by(|a, b| a.total_cmp(b));
    let d_cbca_aliph = fit_unlabelled_pairwise_drift(&diffs_cbca_vs_canco, 0.45, 0.045);
    let d_ncaco_ca = fit_unlabelled_pairwise_drift(&diffs_ncaco_vs_canco, 0.45, 0.045);

    for r in 0..n_roots {
        for &(c_cc, _) in &cccon_by_root[r] {
            for &(ca_cn, _) in &canco_by_root[r] {
                let df = c_cc - ca_cn;
                if df.abs() <= 0.65 {
                    diffs_cccon_vs_ref.push(df);
                }
            }
            for &(c_cb, _, _) in &cbca_by_root[r] {
                let df = c_cc - (c_cb - d_cbca_aliph);
                if df.abs() <= 0.65 {
                    diffs_cccon_vs_ref.push(df);
                }
            }
        }
    }
    diffs_cccon_vs_ref.sort_by(|a, b| a.total_cmp(b));
    let d_cccon_aliph = fit_unlabelled_pairwise_drift(&diffs_cccon_vs_ref, 0.45, 0.045);

    // 2. Align `2D CACO` `(C', CA)` onto `(CON C', CANCO/CBCACON CA)` frame
    let (d_caco_c, d_caco_ca) = if picked.caco_peaks.len() >= 5 {
        let mut diffs_c = Vec::new();
        for cp in &picked.caco_peaks {
            for r in &picked.con_roots {
                let df = cp.c_curr_ppm - r.c_prev_ppm;
                if df.abs() <= 0.55 {
                    diffs_c.push(df);
                }
            }
        }
        diffs_c.sort_by(|a, b| a.total_cmp(b));
        let dc = fit_unlabelled_pairwise_drift(&diffs_c, 0.45, 0.035);

        let mut diffs_ca = Vec::new();
        for cp in &picked.caco_peaks {
            let c_corr = cp.c_curr_ppm - dc;
            for (r_idx, r) in picked.con_roots.iter().enumerate() {
                if (c_corr - r.c_prev_ppm).abs() <= 0.06 {
                    for &(ca_cn, _) in &canco_by_root[r_idx] {
                        let df = cp.ca_curr_ppm - ca_cn;
                        if df.abs() <= 0.65 {
                            diffs_ca.push(df);
                        }
                    }
                    for &(ca_cb, sgn, _) in &cbca_by_root[r_idx] {
                        if sgn >= 0 {
                            let df = cp.ca_curr_ppm - (ca_cb - d_cbca_aliph);
                            if df.abs() <= 0.65 {
                                diffs_ca.push(df);
                            }
                        }
                    }
                }
            }
        }
        diffs_ca.sort_by(|a, b| a.total_cmp(b));
        let dca = fit_unlabelled_pairwise_drift(&diffs_ca, 0.45, 0.045);
        (dc, dca)
    } else {
        (0.0, 0.0)
    };

    let any_cbcacon = cbca_by_root.iter().any(|v| !v.is_empty());
    let has_both_signs_cbca = picked
        .strip_peaks
        .iter()
        .any(|p| p.exp == C13DetectedExp::Cbcaco3D && p.sign < 0);

    let caco_corr: Vec<(f64, f64, CacoPeakKind, f64)> = picked
        .caco_peaks
        .iter()
        .map(|p| {
            (
                p.c_curr_ppm - d_caco_c,
                p.ca_curr_ppm - d_caco_ca,
                p.kind,
                p.height.abs(),
            )
        })
        .collect();

    // 3. Assemble per-root `ConDipeptideSystem`
    let mut systems = Vec::with_capacity(n_roots);
    for (r_idx, root) in picked.con_roots.iter().enumerate() {
        let mut sys = ConDipeptideSystem::new(r_idx, root.c_prev_ppm, root.n_curr_ppm, root.height);
        sys.has_cbcacon_evidence = any_cbcacon;

        // Sort strip peaks deterministically by descending intensity (tie-break by shift)
        let mut canco = canco_by_root[r_idx].clone();
        canco.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        let mut cbca: Vec<(f64, i32, f64)> = cbca_by_root[r_idx]
            .iter()
            .map(|&(v, s, h)| (v - d_cbca_aliph, s, h))
            .collect();
        cbca.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));

        let mut cccon: Vec<(f64, f64)> = cccon_by_root[r_idx]
            .iter()
            .map(|&(v, h)| (v - d_cccon_aliph, h))
            .collect();
        cccon.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        let mut ncaco: Vec<(f64, f64, f64)> = ncaco_by_root[r_idx]
            .iter()
            .map(|&(c_p, ca_p, h)| (c_p, ca_p - d_ncaco_ca, h))
            .collect();
        ncaco.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));

        // (a) Extract `CA(i-1)` and `CB(i-1)` from `CBCACON`, suppressing <0.35x noise spikes
        // when a strong primary peak is already present in that single-atom channel
        let max_pos_cbca = cbca
            .iter()
            .filter(|&&(_, s, _)| s >= 0)
            .map(|&(_, _, h)| h)
            .fold(0.0f64, f64::max);
        let max_neg_cbca = cbca
            .iter()
            .filter(|&&(_, s, _)| s < 0)
            .map(|&(_, _, h)| h)
            .fold(0.0f64, f64::max);

        for &(val, sgn, h) in &cbca {
            if has_both_signs_cbca {
                if sgn >= 0 && (40.0..=69.5).contains(&val) {
                    if sys.ca_prev.is_empty() || h >= 0.35 * max_pos_cbca {
                        dedup_push(&mut sys.ca_prev, val, 0.09, MAX_IDP_CANDS);
                    }
                } else if sgn < 0
                    && (13.0..=73.5).contains(&val)
                    && (sys.cb_prev.is_empty() || h >= 0.35 * max_neg_cbca)
                {
                    dedup_push(&mut sys.cb_prev, val, 0.11, MAX_IDP_CANDS);
                }
            } else if (41.5..=65.5).contains(&val) && sys.ca_prev.is_empty() {
                dedup_push(&mut sys.ca_prev, val, 0.09, MAX_IDP_CANDS);
            } else if (13.0..=73.5).contains(&val) {
                dedup_push(&mut sys.cb_prev, val, 0.11, MAX_IDP_CANDS);
            }
        }

        // (b) Check `2D CACO` at `c_prev` (`|c_caco - c_prev| <= 0.045 ppm`) for `Gly` opposite-phase
        // and corroboration / missing-relay rescue of `CA(i-1)` (promoting corroborated peaks ahead
        // of low-amplitude spurious 3D noise artifacts!)
        let mut caco_at_prev: Vec<(f64, CacoPeakKind, f64)> = caco_corr
            .iter()
            .filter(|&&(cc, _, _, _)| (cc - root.c_prev_ppm).abs() <= 0.045)
            .map(|&(_, ca_v, kind, h)| (ca_v, kind, h))
            .collect();
        caco_at_prev.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));

        // Promote any `CANCO` or `CBCACON` peak corroborated by `2D CACO(c_prev)` to the front of `ca_prev`
        let mut corroborated_prev = Vec::new();
        for &p in &sys.ca_prev {
            if caco_at_prev
                .iter()
                .any(|&(ca_co, _, _)| (ca_co - p).abs() <= 0.12)
            {
                dedup_push(&mut corroborated_prev, p, 0.09, MAX_IDP_CANDS);
            }
        }
        for &(ca_cn, _) in &canco {
            if caco_at_prev
                .iter()
                .any(|&(ca_co, _, _)| (ca_co - ca_cn).abs() <= 0.12)
            {
                dedup_push(&mut corroborated_prev, ca_cn, 0.09, MAX_IDP_CANDS);
            }
        }
        if !corroborated_prev.is_empty() {
            for &old_p in &sys.ca_prev {
                if max_pos_cbca <= 0.0
                    || cbca.iter().any(|&(v, s, h)| {
                        s >= 0 && (v - old_p).abs() <= 0.09 && h >= 0.35 * max_pos_cbca
                    })
                {
                    dedup_push(&mut corroborated_prev, old_p, 0.09, MAX_IDP_CANDS);
                }
            }
            sys.ca_prev = corroborated_prev;
        } else if sys.ca_prev.is_empty() && caco_at_prev.len() <= 3 {
            for &(ca_co, _, _) in &caco_at_prev {
                dedup_push(&mut sys.ca_prev, ca_co, 0.09, MAX_IDP_CANDS);
            }
        }

        for &(ca_v, kind, _) in &caco_at_prev {
            if kind == CacoPeakKind::GlyOppositePhase
                && (sys.ca_prev.is_empty() || sys.ca_prev.iter().any(|&p| (p - ca_v).abs() <= 0.18))
            {
                sys.prev_gly_opposite_phase = true;
            }
        }
        // If `2D CACO` confirms opposite-phase `Gly` at `c_prev` (`CA < 47 ppm`), suppress weak 3D noise in `cb_prev`
        if sys.prev_gly_opposite_phase
            && sys.ca_prev.first().is_some_and(|&ca| ca < 47.0)
            && max_pos_cbca > 0.0
            && max_neg_cbca < 0.42 * max_pos_cbca
        {
            sys.cb_prev.clear();
        }

        // Suppress `< 0.32 * max_canco` 3rd+ noise spikes in `CANCO` (which physically has <= 2 peaks: CA_i & CA_{i-1})
        let max_canco_h = canco.first().map_or(0.0, |x| x.1);
        let canco_clean: Vec<(f64, f64)> = canco
            .iter()
            .enumerate()
            .filter(|&(idx, &(_, h))| idx < 2 || h >= 0.32 * max_canco_h)
            .map(|(_, &x)| x)
            .collect();

        // Corroborate `NCACO` `(C'(i), CA(i))` against `2D CACO` when `2D CACO` is present
        let max_ncaco_h = ncaco.first().map_or(0.0, |x| x.2);
        let mut corrob_ncaco: Vec<(f64, f64, f64)> = if !caco_corr.is_empty() {
            ncaco
                .iter()
                .copied()
                .filter(|&(c_i, ca_i, _)| {
                    caco_corr.iter().any(|&(cc, cac, _, _)| {
                        (cc - c_i).abs() <= 0.055 && (cac - ca_i).abs() <= 0.12
                    })
                })
                .collect()
        } else {
            ncaco
                .iter()
                .copied()
                .filter(|&(_, _, h)| h >= 0.35 * max_ncaco_h)
                .collect()
        };
        if corrob_ncaco.is_empty() {
            for &(c_i, ca_i, h) in &ncaco {
                if h >= 0.40 * max_ncaco_h {
                    corrob_ncaco.push((c_i, ca_i, h));
                }
            }
        }
        if corrob_ncaco.len() > 1 {
            corrob_ncaco.sort_by(|a, b| {
                // Prioritize `NCACO` peaks whose `CA` agrees with `canco_clean` (`<= 0.15 ppm`)
                // so unrelated `15N`-plane collisions never rank ahead of intraresidue `CA(i)`!
                let ma = canco_clean.iter().any(|&(vc, _)| (vc - a.1).abs() <= 0.15);
                let mb = canco_clean.iter().any(|&(vc, _)| (vc - b.1).abs() <= 0.15);
                if ma != mb {
                    return mb.cmp(&ma);
                }
                if !caco_corr.is_empty() {
                    let da = caco_corr
                        .iter()
                        .map(|&(cc, cac, _, _)| {
                            ((cc - a.0) / 0.04).powi(2) + ((cac - a.1) / 0.09).powi(2)
                        })
                        .fold(f64::INFINITY, f64::min);
                    let db = caco_corr
                        .iter()
                        .map(|&(cc, cac, _, _)| {
                            ((cc - b.0) / 0.04).powi(2) + ((cac - b.1) / 0.09).powi(2)
                        })
                        .fold(f64::INFINITY, f64::min);
                    da.total_cmp(&db).then_with(|| b.2.total_cmp(&a.2))
                } else {
                    b.2.total_cmp(&a.2)
                }
            });
        }

        // (c) + (d) Jointly separate `CANCO` into `ca_curr` / `ca_prev` and attach `NCACO` `(c_curr, ca_curr)`
        // without poisoning `ca_curr` with `CA(i-1)` when `CANCO` drops `CA(i)`!
        // CRITICAL: use `0.014 ppm` dedup on `c_curr` (and `0.045 ppm` on `ca_curr`) so `0.03–0.04 ppm`
        // IDP `13C'` clusters (e.g. multiple `GLU-GLY` repeats in alpha-synuclein) are NEVER merged away!
        let prev_was_empty = sys.ca_prev.is_empty();
        let non_prev_canco: Vec<f64> = canco_clean
            .iter()
            .map(|&(v, _)| v)
            .filter(|&v| !sys.ca_prev.iter().take(1).any(|&p| (p - v).abs() <= 0.13))
            .collect();

        if !non_prev_canco.is_empty() {
            for &v in &non_prev_canco {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
            }
            if prev_was_empty && canco_clean.len() >= 2 {
                // `^2J_{NCa}` (`CA(i-1)`) is weaker than `^1J_{NCa}` (`CA(i)`): put weaker peak first in `ca_prev`
                for &(v, _) in canco_clean.iter().rev() {
                    dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
                }
            } else {
                for &(v, _) in &canco_clean {
                    if !non_prev_canco.iter().any(|&c| (c - v).abs() <= 0.09) {
                        dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
                    }
                }
            }
            // Also keep a `ca_prev`-matching `CANCO` peak in `ca_curr` if `NCACO` explicitly supports it (`CA(i) ≈ CA(i-1)`)
            for &(v, _) in &canco_clean {
                if corrob_ncaco
                    .iter()
                    .any(|&(_, ca_nc, _)| (ca_nc - v).abs() <= 0.14)
                {
                    dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                }
            }
            for &(c_i, ca_i, _) in &corrob_ncaco {
                if sys.ca_curr.iter().any(|&ca_c| (ca_c - ca_i).abs() <= 0.15) {
                    dedup_push(&mut sys.c_curr, c_i, 0.014, MAX_IDP_CANDS);
                    dedup_push(&mut sys.ca_curr, ca_i, 0.045, MAX_IDP_CANDS);
                }
            }
        } else if !corrob_ncaco.is_empty() {
            // `CANCO` either missed `CA(i)` (10% dropout!) OR coalesced `CA(i) ≈ CA(i-1)`:
            // Rescue true `(C'(i), CA(i))` directly from `NCACO` FIRST before falling back to `canco_clean`!
            for &(c_i, ca_i, _) in corrob_ncaco.iter().take(5) {
                dedup_push(&mut sys.c_curr, c_i, 0.014, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_curr, ca_i, 0.045, MAX_IDP_CANDS);
            }
            for &(v, _) in &canco_clean {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
            }
        } else {
            for &(v, _) in &canco_clean {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
            }
        }

        // Unambiguous single-match fallback: ONLY rescue `c_curr` from `2D CACO` when `ca_curr[0]`
        // matches a strictly unique `CA` peak across the entire `2D CACO` spectrum (`len == 1` within `0.15 ppm`)
        if sys.c_curr.is_empty() && !sys.ca_curr.is_empty() && !caco_corr.is_empty() {
            let ca0 = sys.ca_curr[0];
            let cands_c: Vec<f64> = caco_corr
                .iter()
                .filter(|&&(_, cac, _, _)| (cac - ca0).abs() <= 0.15)
                .map(|&(cc, _, _, _)| cc)
                .collect();
            if cands_c.len() == 1
                && (caco_corr
                    .iter()
                    .map(|&(_, cac, _, _)| (cac - ca0).abs())
                    .fold(1.0, f64::min)
                    <= 0.08)
            {
                dedup_push(&mut sys.c_curr, cands_c[0], 0.014, MAX_IDP_CANDS);
            }
        }

        // (e) Incorporate `3D CCC(CO)N` aliphatic TOCSY relays for residue `(i-1)`
        for &(c_val, _) in &cccon {
            let is_ca = sys.ca_prev.iter().any(|&p| (p - c_val).abs() <= 0.16);
            let is_cb = sys.cb_prev.iter().any(|&p| (p - c_val).abs() <= 0.18);
            if is_ca || is_cb {
                continue;
            }
            if sys.ca_prev.is_empty() && (42.0..=66.0).contains(&c_val) {
                dedup_push(&mut sys.ca_prev, c_val, 0.10, MAX_IDP_CANDS);
            } else if sys.cb_prev.is_empty() && (26.0..=71.5).contains(&c_val) {
                dedup_push(&mut sys.cb_prev, c_val, 0.12, MAX_IDP_CANDS);
            } else if (9.5..=56.0).contains(&c_val) {
                dedup_push(&mut sys.sidechain_c_prev, c_val, 0.15, MAX_IDP_SC_CANDS);
            }
        }

        systems.push(sys);
    }

    systems
}

// ============================================================================
// Exact CPU / WebGPU Parity Scoring (`U` & `P` Matrices in Smooth `f32` Math)
// ============================================================================

#[inline]
fn logpdf_2comp_f32(v: f32, mu: f32, sd: f32) -> f32 {
    let s1 = sd.max(0.15);
    let s2 = 3.8 * s1;
    let z1 = (v - mu) / s1;
    let z2 = (v - mu) / s2;
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let p1 = 0.78 * inv_sqrt_2pi * (-0.5 * z1 * z1).exp() / s1;
    let p2 = 0.22 * inv_sqrt_2pi * (-0.5 * z2 * z2).exp() / s2;
    (p1 + p2).max(1.0e-30).ln()
}

/// Precomputed `20`-AA background shifts for `f32`-exact CPU/WebGPU background normalization.
pub const IDP_BG_N_MUS: [f32; 20] = [
    124.5, 121.3, 119.0, 120.7, 119.8, 120.3, 121.2, 109.7, 119.6, 121.2, 122.6, 121.6, 120.6,
    120.9, 136.8, 116.5, 115.0, 121.7, 120.8, 120.6,
];
pub const IDP_BG_C_MUS: [f32; 20] = [
    177.8, 176.3, 175.2, 176.4, 174.6, 176.0, 176.6, 173.9, 175.2, 176.4, 177.4, 176.6, 176.2,
    175.8, 176.9, 174.6, 174.6, 176.3, 175.7, 176.3,
];
pub const IDP_BG_CA_MUS: [f32; 20] = [
    52.5, 56.0, 53.1, 54.2, 58.2, 55.8, 56.6, 45.2, 55.8, 61.1, 55.1, 56.3, 55.4, 57.8, 63.2, 58.3,
    61.8, 57.4, 57.9, 62.2,
];
pub const IDP_BG_CB_MUS: [f32; 19] = [
    19.1, 30.8, 38.9, 41.1, 28.0, 29.5, 30.2, 29.8, 38.6, 42.3, 33.0, 32.8, 39.6, 32.1, 63.8, 69.8,
    29.6, 38.8, 32.8,
];
pub const IDP_BG_SC_MUS: [f32; 12] = [
    13.0, 17.3, 21.2, 24.4, 27.1, 29.1, 32.1, 33.8, 36.1, 42.0, 43.3, 50.4,
];

#[inline]
fn bg_logpdf_f32(v: f32, mus: &[f32], sd: f32) -> f32 {
    let mut sum = 0.0f32;
    let inv_n = 1.0 / (mus.len() as f32);
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let s1 = (sd * 1.25).max(0.25);
    for &m in mus {
        let z = (v - m) / s1;
        sum += inv_n * inv_sqrt_2pi * (-0.5 * z * z).exp() / s1;
    }
    (0.95 * sum + 0.05 / 45.0).max(1.0e-30).ln()
}

#[inline]
fn best_cand_llr_f32(
    cands: &[f32],
    mu: f32,
    sd: f32,
    bg_mus: &[f32],
    bg_sd: f32,
    llr_floor: f32,
) -> f32 {
    if cands.is_empty() {
        return 0.0;
    }
    let mut best = -1.0e30f32;
    for (idx, &v) in cands.iter().take(MAX_IDP_CANDS).enumerate() {
        let rank_pen = 0.25 * (idx as f32);
        let llr = (logpdf_2comp_f32(v, mu, sd) - bg_logpdf_f32(v, bg_mus, bg_sd)).max(llr_floor)
            - rank_pen;
        if llr > best {
            best = llr;
        }
    }
    best
}

/// Compute the continuous through-Proline IDP dipeptide unary scoring matrix `U` `(S x K)` on CPU.
///
/// - `k == 0` (N-terminus, no preceding peptide bond) is `FORBID` (`-1e9`).
/// - **Every `k in 1..K` — INCLUDING `PRO` at `k` and `PRO` at `k - 1` — is an active spin-system slot!**
/// - Uses smooth `f32`-compatible log-likelihood ratios matching `idp_scoring.wgsl` to `< 1e-5`.
pub fn idp_unary_matrix(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> Matrix2D {
    let s_len = systems.len();
    let k_len = sequence.len();
    let mut u = Matrix2D::filled(s_len, k_len, FORBID);
    if s_len == 0 || k_len < 2 {
        return u;
    }

    let preds = predict_ncidp_shifts(sequence, cfg);
    let dn = offsets.d_n as f32;
    let dc = offsets.d_c as f32;
    let dca = offsets.d_ca as f32;
    let dcb = offsets.d_cb as f32;
    let llr_floor = cfg.llr_floor as f32;
    let n_with_3d = systems
        .iter()
        .filter(|s| {
            !s.ca_curr.is_empty()
                || !s.ca_prev.is_empty()
                || !s.cb_prev.is_empty()
                || !s.c_curr.is_empty()
                || !s.sidechain_c_prev.is_empty()
        })
        .count();
    let multi_3d_dataset = n_with_3d * 2 >= s_len;

    // Pre-shift background arrays into the experimental frame
    let bg_n: Vec<f32> = IDP_BG_N_MUS.iter().map(|&m| m + dn).collect();
    let bg_c: Vec<f32> = IDP_BG_C_MUS.iter().map(|&m| m + dc).collect();
    let bg_ca: Vec<f32> = IDP_BG_CA_MUS.iter().map(|&m| m + dca).collect();
    let bg_cb: Vec<f32> = IDP_BG_CB_MUS.iter().map(|&m| m + dcb).collect();
    let bg_sc: Vec<f32> = IDP_BG_SC_MUS.iter().map(|&m| m + dcb).collect();

    for (s_idx, sys) in systems.iter().enumerate() {
        let n_curr = sys.n_curr as f32;
        let c_prev = sys.c_prev as f32;
        let ca_curr: Vec<f32> = sys
            .ca_curr
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let ca_prev: Vec<f32> = sys
            .ca_prev
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let cb_prev: Vec<f32> = sys
            .cb_prev
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let c_curr: Vec<f32> = sys
            .c_curr
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let n_next: Vec<f32> = sys
            .n_next
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let sc_prev: Vec<f32> = sys
            .sidechain_c_prev
            .iter()
            .take(MAX_IDP_SC_CANDS)
            .map(|&x| x as f32)
            .collect();
        let n_3d_ch = u32::from(!ca_curr.is_empty())
            + u32::from(!ca_prev.is_empty())
            + u32::from(!cb_prev.is_empty())
            + u32::from(!c_curr.is_empty())
            + u32::from(!sc_prev.is_empty());
        let sys_assign_bonus: f32 = if n_3d_ch >= 1 {
            2.65
        } else if multi_3d_dataset && s_len + 1 >= k_len {
            -3.5
        } else {
            0.5
        };

        for k in 1..k_len {
            let cur_p = &preds[k];
            let prv_p = &preds[k - 1];

            // 1. Root `15N(k)` & `13C'(k-1)`
            let mut total: f32 = sys_assign_bonus;
            total += (logpdf_2comp_f32(n_curr, (cur_p.n_ppm as f32) + dn, cur_p.sigma_n as f32)
                - bg_logpdf_f32(n_curr, &bg_n, 1.4))
            .max(llr_floor);
            total += (logpdf_2comp_f32(c_prev, (prv_p.c_ppm as f32) + dc, prv_p.sigma_c as f32)
                - bg_logpdf_f32(c_prev, &bg_c, 0.7))
            .max(llr_floor);

            // 2. `CA(k)` & `CA(k-1)`
            total += best_cand_llr_f32(
                &ca_curr,
                (cur_p.ca_ppm as f32) + dca,
                cur_p.sigma_ca as f32,
                &bg_ca,
                1.2,
                llr_floor,
            );
            total += best_cand_llr_f32(
                &ca_prev,
                (prv_p.ca_ppm as f32) + dca,
                prv_p.sigma_ca as f32,
                &bg_ca,
                1.2,
                llr_floor,
            );

            // 3. `CB(k-1)` and `Gly(k-1)` opposite-phase / missing-CB signatures
            if let Some(mu_cb) = prv_p.cb_ppm {
                if !cb_prev.is_empty() {
                    total += best_cand_llr_f32(
                        &cb_prev,
                        (mu_cb as f32) + dcb,
                        prv_p.sigma_cb as f32,
                        &bg_cb,
                        2.5,
                        llr_floor,
                    );
                } else if sys.has_cbcacon_evidence {
                    total -= 1.35;
                }
                if sys.prev_gly_opposite_phase {
                    total -= 4.50;
                }
            } else {
                // `Gly(k-1)` has no CB! Only award missing-CB bonus when `CA(k-1)` WAS observed
                if !cb_prev.is_empty() {
                    total -= 4.20;
                } else if sys.has_cbcacon_evidence && !ca_prev.is_empty() {
                    total += 2.10;
                }
                if sys.prev_gly_opposite_phase {
                    total += 3.60;
                }
            }

            // 4. Intraresidue `C'(k)` (`NCACO` / `CACO`)
            if !c_curr.is_empty() {
                total += 0.85
                    * best_cand_llr_f32(
                        &c_curr,
                        (cur_p.c_ppm as f32) + dc,
                        cur_p.sigma_c as f32,
                        &bg_c,
                        0.7,
                        llr_floor,
                    );
            }

            // 5. Optional `N(k+1)` when `k + 1 < k_len`
            if !n_next.is_empty() && k + 1 < k_len {
                let nxt_p = &preds[k + 1];
                total += 0.70
                    * best_cand_llr_f32(
                        &n_next,
                        (nxt_p.n_ppm as f32) + dn,
                        nxt_p.sigma_n as f32,
                        &bg_n,
                        1.4,
                        llr_floor,
                    );
            }

            // 6. Aliphatic sidechain `CCC(CO)N` peaks of `k - 1`
            if !sc_prev.is_empty() {
                if prv_p.sidechain_c.is_empty() {
                    total -= 1.25 * (sc_prev.len().min(3) as f32);
                } else {
                    let exp_sc: Vec<f32> = prv_p
                        .sidechain_c
                        .iter()
                        .take(4)
                        .map(|&m| (m as f32) + dcb)
                        .collect();
                    let mut sc_sum = 0.0f32;
                    for &v in &sc_prev {
                        let mut best_comp = -1.0e30f32;
                        for &mu_s in &exp_sc {
                            let lp = logpdf_2comp_f32(v, mu_s, 0.75);
                            if lp > best_comp {
                                best_comp = lp;
                            }
                        }
                        let llr = (best_comp - bg_logpdf_f32(v, &bg_sc, 1.8)).max(-4.0);
                        sc_sum += 0.55 * llr;
                    }
                    total += sc_sum;
                }
            }

            u.set(s_idx, k, total as f64);
        }
    }

    u
}

#[inline]
fn channel_pair_llr_f32(
    a_cands: &[f32],
    b_cands: &[f32],
    sigma: f32,
    range_ppm: f32,
    pair_floor: f32,
) -> f32 {
    if a_cands.is_empty() || b_cands.is_empty() {
        return 0.0;
    }
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let s = sigma.max(0.02);
    let bg_log = (1.0 / range_ppm.max(2.0)).ln();
    let mut best = -1.0e30f32;
    for (ia, &va) in a_cands.iter().take(MAX_IDP_CANDS).enumerate() {
        for (ib, &vb) in b_cands.iter().take(MAX_IDP_CANDS).enumerate() {
            let z = (va - vb) / s;
            let lp = (inv_sqrt_2pi / s).ln() - 0.5 * z * z - 0.14 * ((ia + ib) as f32);
            let llr = (lp - bg_log).max(pair_floor);
            if llr > best {
                best = llr;
            }
        }
    }
    best
}

/// Compute sequential dipeptide transition matrix `P` `(S x S)` where `P(s, t)` scores `s` at `k`
/// followed immediately by `t` at `k + 1` across THREE sequential pathways:
/// 1. `s.ca_curr` (`CA(k)`) matching `t.ca_prev` (`CA((k+1)-1)`),
/// 2. `s.c_curr` (`C'(k)`) matching `t.c_prev` (`C'((k+1)-1)` root!), and
/// 3. `s.n_next` (`N(k+1)`) matching `t.n_curr` (`N(k+1)` root!).
pub fn idp_pair_matrix(systems: &[ConDipeptideSystem], cfg: &IdpSolverConfig) -> Matrix2D {
    let s_len = systems.len();
    let mut p = Matrix2D::filled(s_len, s_len, 0.0);
    if s_len == 0 {
        return p;
    }

    let sig_ca = cfg.sigma_seq_ca as f32;
    let sig_c = cfg.sigma_seq_c as f32;
    let sig_n = cfg.sigma_seq_n as f32;
    let p_floor = cfg.pair_floor as f32;

    let ca_curr: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.ca_curr
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let ca_prev: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.ca_prev
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let c_curr: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.c_curr
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let n_next: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.n_next
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let c_prev_root: Vec<[f32; 1]> = systems.iter().map(|s| [s.c_prev as f32]).collect();
    let n_curr_root: Vec<[f32; 1]> = systems.iter().map(|s| [s.n_curr as f32]).collect();

    for s in 0..s_len {
        for t in 0..s_len {
            if s == t {
                p.set(s, t, FORBID);
                continue;
            }
            let sc_ca = channel_pair_llr_f32(&ca_curr[s], &ca_prev[t], sig_ca, 28.0, p_floor);
            let sc_c = channel_pair_llr_f32(&c_curr[s], &c_prev_root[t], sig_c, 13.0, p_floor);
            let sc_n = channel_pair_llr_f32(&n_next[s], &n_curr_root[t], sig_n, 36.0, p_floor);
            p.set(s, t, (sc_ca + sc_c + sc_n) as f64);
        }
    }

    p
}

// ============================================================================
// Multi-Island Evolutionary + Detailed-Balance Annealing & QC Calibration
// ============================================================================

/// Initialized IDP replica state `(at[0..K], pos[0..S], exact_score)`.
pub type IdpReplicaState = (Vec<i32>, Vec<i32>, f64);

/// Build valid sequence slot lists `slots[s]` (`k in 1..K` with `U(s, k) > IDP_FORBID_HALF`).
pub fn build_idp_valid_slots(u_mat: &Matrix2D) -> Vec<Vec<u32>> {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut slots = Vec::with_capacity(s_len);
    for s in 0..s_len {
        let mut v = Vec::with_capacity(k_len);
        for k in 1..k_len {
            if u_mat.get(s, k) > (IDP_FORBID_HALF as f64) {
                v.push(k as u32);
            }
        }
        slots.push(v);
    }
    slots
}

/// Evaluate exact total objective $\sum_k U(\text{at}[k], k) + \sum_k P(\text{at}[k], \text{at}[k+1])$.
pub fn evaluate_idp_total_score(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D) -> f64 {
    let k_len = at.len();
    let mut sc = 0.0f64;
    for (k, &s) in at.iter().enumerate() {
        if s >= 0 {
            sc += u_mat.get(s as usize, k);
        }
    }
    for k in 0..k_len.saturating_sub(1) {
        let a = at[k];
        let b = at[k + 1];
        if a >= 0 && b >= 0 {
            sc += p_mat.get(a as usize, b as usize);
        }
    }
    sc
}

#[inline]
fn eval_1_slot(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D, k: usize) -> f64 {
    let s = at[k];
    if s < 0 {
        return 0.0;
    }
    let su = s as usize;
    let mut sc = u_mat.get(su, k);
    if k > 0 && at[k - 1] >= 0 {
        sc += p_mat.get(at[k - 1] as usize, su);
    }
    if k + 1 < at.len() && at[k + 1] >= 0 {
        sc += p_mat.get(su, at[k + 1] as usize);
    }
    sc
}

#[inline]
fn eval_2_slots(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D, ka: usize, kb: usize) -> f64 {
    let lo = ka.min(kb);
    let hi = ka.max(kb);
    let mut sc = 0.0f64;
    let s_lo = at[lo];
    let s_hi = at[hi];
    if s_lo >= 0 {
        let u_lo = s_lo as usize;
        sc += u_mat.get(u_lo, lo);
        if lo > 0 && at[lo - 1] >= 0 {
            sc += p_mat.get(at[lo - 1] as usize, u_lo);
        }
        if at[lo + 1] >= 0 {
            sc += p_mat.get(u_lo, at[lo + 1] as usize);
        }
    }
    if s_hi >= 0 {
        let u_hi = s_hi as usize;
        sc += u_mat.get(u_hi, hi);
        if hi > lo + 1 && at[hi - 1] >= 0 {
            sc += p_mat.get(at[hi - 1] as usize, u_hi);
        }
        if hi + 1 < at.len() && at[hi + 1] >= 0 {
            sc += p_mat.get(u_hi, at[hi + 1] as usize);
        }
    }
    sc
}

fn eval_block_slots(
    at: &[i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    k1: usize,
    k2: usize,
    len: usize,
) -> f64 {
    let b0 = k1.min(k2);
    let b1 = k1.max(k2);
    let k_len = at.len();
    let mut sc = 0.0f64;
    for i in 0..len {
        if at[b0 + i] >= 0 {
            sc += u_mat.get(at[b0 + i] as usize, b0 + i);
        }
        if at[b1 + i] >= 0 {
            sc += u_mat.get(at[b1 + i] as usize, b1 + i);
        }
    }
    if k_len > 1 {
        let max_e = k_len - 1;
        let e0_start = b0.saturating_sub(1);
        let e0_end = (b0 + len).min(max_e);
        for e in e0_start..e0_end {
            if at[e] >= 0 && at[e + 1] >= 0 {
                sc += p_mat.get(at[e] as usize, at[e + 1] as usize);
            }
        }
        let e1_start = b1.saturating_sub(1).max(e0_end);
        let e1_end = (b1 + len).min(max_e);
        for e in e1_start..e1_end {
            if at[e] >= 0 && at[e + 1] >= 0 {
                sc += p_mat.get(at[e] as usize, at[e + 1] as usize);
            }
        }
    }
    sc
}

/// Deterministic zero-temperature 2-opt swap + single-relocation + contiguous block-swap polish.
pub fn polish_idp_state(
    at: &mut [i32],
    pos: &mut [i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
) -> f64 {
    let s_len = pos.len();
    let k_len = at.len();
    for _pass in 0..3 {
        let mut improved = false;
        // 1. Single-root relocation / 2-swap
        for s in 0..s_len {
            for &ku in &slots[s] {
                let k = ku as usize;
                let ks = pos[s];
                if ks == k as i32 {
                    continue;
                }
                let t = at[k];
                if ks < 0 {
                    let before = eval_1_slot(at, u_mat, p_mat, k);
                    if t >= 0 {
                        pos[t as usize] = -1;
                    }
                    at[k] = s as i32;
                    pos[s] = k as i32;
                    let delta = eval_1_slot(at, u_mat, p_mat, k) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        pos[s] = -1;
                        at[k] = t;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                } else {
                    let ksu = ks as usize;
                    if t >= 0 && u_mat.get(t as usize, ksu) <= (IDP_FORBID_HALF as f64) {
                        continue;
                    }
                    let before = eval_2_slots(at, u_mat, p_mat, k, ksu);
                    at[k] = s as i32;
                    at[ksu] = t;
                    pos[s] = k as i32;
                    if t >= 0 {
                        pos[t as usize] = ksu as i32;
                    }
                    let delta = eval_2_slots(at, u_mat, p_mat, k, ksu) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        at[k] = t;
                        at[ksu] = s as i32;
                        pos[s] = ksu as i32;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                }
            }
            // Check unassigning spurious root if score improves
            let ks = pos[s];
            if ks >= 0 {
                let ksu = ks as usize;
                let before = eval_1_slot(at, u_mat, p_mat, ksu);
                if before < -1e-6 {
                    at[ksu] = -1;
                    pos[s] = -1;
                    improved = true;
                }
            }
        }

        // 2. Short contiguous motif swaps (`L = 2..=6`) to resolve repeated dipeptide/tripeptide motifs
        for blen in 2..=6usize.min(k_len / 2) {
            if k_len <= blen + 1 {
                continue;
            }
            for k1 in 1..=(k_len - blen) {
                for k2 in (k1 + blen)..=(k_len - blen) {
                    let mut valid = true;
                    for i in 0..blen {
                        let a = at[k1 + i];
                        let b = at[k2 + i];
                        if (b >= 0 && u_mat.get(b as usize, k1 + i) <= (IDP_FORBID_HALF as f64))
                            || (a >= 0 && u_mat.get(a as usize, k2 + i) <= (IDP_FORBID_HALF as f64))
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let before = eval_block_slots(at, u_mat, p_mat, k1, k2, blen);
                    for i in 0..blen {
                        let a = at[k1 + i];
                        let b = at[k2 + i];
                        at[k1 + i] = b;
                        at[k2 + i] = a;
                        if a >= 0 {
                            pos[a as usize] = (k2 + i) as i32;
                        }
                        if b >= 0 {
                            pos[b as usize] = (k1 + i) as i32;
                        }
                    }
                    let delta = eval_block_slots(at, u_mat, p_mat, k1, k2, blen) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        for i in 0..blen {
                            let b = at[k1 + i];
                            let a = at[k2 + i];
                            at[k1 + i] = a;
                            at[k2 + i] = b;
                            if a >= 0 {
                                pos[a as usize] = (k1 + i) as i32;
                            }
                            if b >= 0 {
                                pos[b as usize] = (k2 + i) as i32;
                            }
                        }
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }
    evaluate_idp_total_score(at, u_mat, p_mat)
}

/// Build diverse initial replica states `(at, pos, score)` using randomized disjoint high-weight
/// sequential path covers on `P(s, t)` placed onto optimal sequence windows of `U(s, k)`.
/// Shared identically by `assign_idp_13c_cpu` and `assign_idp_13c_gpu`.
pub fn build_idp_initial_replicas(
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    replicas: usize,
    seed: u64,
) -> Vec<IdpReplicaState> {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut master_rng = Rng::new(seed ^ 0x4944_505F_494E_4954);
    let mut out = Vec::with_capacity(replicas);

    // Collect strong sequential candidate edges `(weight, s, t)`
    let mut strong_edges: Vec<(f64, usize, usize)> = Vec::new();
    for s in 0..s_len {
        for t in 0..s_len {
            if s != t {
                let w = p_mat.get(s, t);
                if w >= 2.4 {
                    strong_edges.push((w, s, t));
                }
            }
        }
    }

    for rep in 0..replicas {
        let mut r_rng = Rng::new(master_rng.next_u64());
        let mut at = vec![-1i32; k_len];
        let mut pos = vec![-1i32; s_len];

        // Use greedy chain seeding on 75% of replicas and pure unary-guided seeding on 25% for diversity
        if rep % 4 != 3 && !strong_edges.is_empty() && k_len >= 3 {
            let mut noisy_edges: Vec<(f64, usize, usize)> = strong_edges
                .iter()
                .map(|&(w, s, t)| (w + r_rng.normal(0.0, 0.85), s, t))
                .collect();
            noisy_edges.sort_by(|a, b| b.0.total_cmp(&a.0));

            let mut next_of: Vec<Option<usize>> = vec![None; s_len];
            let mut prev_of: Vec<Option<usize>> = vec![None; s_len];
            let mut comp_id: Vec<usize> = (0..s_len).collect();

            for &(w, s, t) in &noisy_edges {
                if w < 2.2
                    || next_of[s].is_some()
                    || prev_of[t].is_some()
                    || comp_id[s] == comp_id[t]
                {
                    continue;
                }
                // Cap initial seed chain length at `14` so one wrong link never shifts a 60-aa block
                let mut len_s = 1usize;
                let mut cur = s;
                while let Some(p) = prev_of[cur] {
                    len_s += 1;
                    cur = p;
                }
                let mut len_t = 1usize;
                cur = t;
                while let Some(nx) = next_of[cur] {
                    len_t += 1;
                    cur = nx;
                }
                if len_s + len_t > 14 {
                    continue;
                }
                next_of[s] = Some(t);
                prev_of[t] = Some(s);
                let old_c = comp_id[t];
                let new_c = comp_id[s];
                for cid in &mut comp_id {
                    if *cid == old_c {
                        *cid = new_c;
                    }
                }
            }

            // Extract chains of length >= 2, longest first
            let mut chains: Vec<Vec<usize>> = Vec::new();
            for s in 0..s_len {
                if prev_of[s].is_none() && next_of[s].is_some() {
                    let mut ch = vec![s];
                    let mut cur = s;
                    while let Some(nx) = next_of[cur] {
                        ch.push(nx);
                        cur = nx;
                    }
                    chains.push(ch);
                }
            }
            chains.sort_by_key(|b| std::cmp::Reverse(b.len()));

            for ch in &chains {
                let clen = ch.len();
                if clen == 0 || k_len <= clen {
                    continue;
                }
                let mut best_k0: Option<usize> = None;
                let mut best_sc = f64::NEG_INFINITY;
                for k0 in 1..=(k_len - clen) {
                    let mut ok = true;
                    let mut u_sum = 0.0f64;
                    for (m, &sm) in ch.iter().enumerate() {
                        if at[k0 + m] >= 0 {
                            ok = false;
                            break;
                        }
                        let u = u_mat.get(sm, k0 + m);
                        if u <= (IDP_FORBID_HALF as f64) {
                            ok = false;
                            break;
                        }
                        u_sum += u;
                    }
                    if ok {
                        let noisy = u_sum + r_rng.normal(0.0, 1.4 * (clen as f64).sqrt());
                        if noisy > best_sc {
                            best_sc = noisy;
                            best_k0 = Some(k0);
                        }
                    }
                }
                if let Some(k0) = best_k0 {
                    for (m, &sm) in ch.iter().enumerate() {
                        at[k0 + m] = sm as i32;
                        pos[sm] = (k0 + m) as i32;
                    }
                }
            }
        }

        // Fill remaining unassigned roots in randomized order
        let mut rem: Vec<(f64, usize)> = (0..s_len)
            .filter(|&s| pos[s] < 0)
            .map(|s| ((slots[s].len() as f64) + r_rng.f64(), s))
            .collect();
        rem.sort_by(|a, b| a.0.total_cmp(&b.0));

        for &(_, s) in &rem {
            let mut best_k: Option<usize> = None;
            let mut best_val = -0.5f64;
            for &ku in &slots[s] {
                let k = ku as usize;
                if at[k] >= 0 {
                    continue;
                }
                let mut sc = u_mat.get(s, k);
                if k > 1 && at[k - 1] >= 0 {
                    sc += p_mat.get(at[k - 1] as usize, s);
                }
                if k + 1 < k_len && at[k + 1] >= 0 {
                    sc += p_mat.get(s, at[k + 1] as usize);
                }
                let noisy = sc + r_rng.normal(0.0, 1.5);
                if noisy > best_val {
                    best_val = noisy;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                at[k] = s as i32;
                pos[s] = k as i32;
            }
        }

        let sc = evaluate_idp_total_score(&at, u_mat, p_mat);
        out.push((at, pos, sc));
    }

    out
}

/// Run exact detailed-balance Metropolis–Hastings simulated annealing on a single CPU replica.
#[allow(clippy::too_many_arguments)]
fn run_idp_cpu_anneal_segment(
    state: &mut IdpReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    steps: usize,
    t_start: f64,
    t_end: f64,
    rng: &mut Rng,
) -> (Vec<i32>, Vec<i32>, f64) {
    let s_len = state.1.len();
    let k_len = state.0.len();
    let at = &mut state.0;
    let pos = &mut state.1;
    let mut cur = evaluate_idp_total_score(at, u_mat, p_mat);
    let mut best_sc = cur;
    let mut best_at = at.clone();
    let mut best_pos = pos.clone();

    if s_len == 0 || k_len < 2 || steps == 0 {
        state.2 = best_sc;
        return (best_at, best_pos, best_sc);
    }

    let denom = (steps.saturating_sub(1)).max(1) as f64;
    let log_ratio = (t_end.max(1e-4) / t_start.max(1e-3)).ln();
    let max_blk = 16usize.min((k_len - 1) / 2);

    for step in 0..steps {
        let temp = t_start * (log_ratio * (step as f64 / denom)).exp();
        let draw = rng.f64();

        if draw < 0.70 || max_blk == 0 {
            // Reversible single-root move over `slots[s] U {-1}` (exact detailed balance!)
            let s = rng.usize(s_len);
            let sl_s = slots[s].len();
            if sl_s == 0 {
                continue;
            }
            let opt = rng.usize(sl_s + 1);
            let ks = pos[s];

            if opt == sl_s {
                // Propose unassign `s -> -1`
                if ks < 0 {
                    continue;
                }
                let ksu = ks as usize;
                let before = eval_1_slot(at, u_mat, p_mat, ksu);
                at[ksu] = -1;
                pos[s] = -1;
                let delta = -before;
                if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                    cur += delta;
                    if cur > best_sc {
                        best_sc = cur;
                        best_at.copy_from_slice(at);
                        best_pos.copy_from_slice(pos);
                    }
                } else {
                    at[ksu] = s as i32;
                    pos[s] = ks;
                }
            } else {
                let k = slots[s][opt] as usize;
                if ks == k as i32 {
                    continue;
                }
                let t = at[k];
                if ks < 0 {
                    let before = eval_1_slot(at, u_mat, p_mat, k);
                    let hastings_log = if t >= 0 {
                        ((sl_s + 1) as f64 / (slots[t as usize].len() + 1) as f64).ln()
                    } else {
                        0.0
                    };
                    if t >= 0 {
                        pos[t as usize] = -1;
                    }
                    at[k] = s as i32;
                    pos[s] = k as i32;
                    let delta = eval_1_slot(at, u_mat, p_mat, k) - before;
                    let log_acc = (delta / temp) + hastings_log;
                    if log_acc >= 0.0 || rng.f64() < log_acc.exp() {
                        cur += delta;
                        if cur > best_sc {
                            best_sc = cur;
                            best_at.copy_from_slice(at);
                            best_pos.copy_from_slice(pos);
                        }
                    } else {
                        pos[s] = -1;
                        at[k] = t;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                } else {
                    let ksu = ks as usize;
                    // Pure reversible 2-swap: reject if `t` is forbidden at `ksu`
                    if t >= 0 && u_mat.get(t as usize, ksu) <= (IDP_FORBID_HALF as f64) {
                        continue;
                    }
                    let before = eval_2_slots(at, u_mat, p_mat, k, ksu);
                    at[k] = s as i32;
                    at[ksu] = t;
                    pos[s] = k as i32;
                    if t >= 0 {
                        pos[t as usize] = ksu as i32;
                    }
                    let delta = eval_2_slots(at, u_mat, p_mat, k, ksu) - before;
                    if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                        cur += delta;
                        if cur > best_sc {
                            best_sc = cur;
                            best_at.copy_from_slice(at);
                            best_pos.copy_from_slice(pos);
                        }
                    } else {
                        at[k] = t;
                        at[ksu] = s as i32;
                        pos[s] = ksu as i32;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                }
            }
        } else {
            // Equal-length non-overlapping block swap on `1..k_len` (symmetric proposal!)
            let l = 1 + rng.usize(max_blk);
            let span = (k_len - 1) - l + 1;
            if span < 2 {
                continue;
            }
            let k1 = 1 + rng.usize(span);
            let k2 = 1 + rng.usize(span);
            if k1.abs_diff(k2) < l {
                continue;
            }
            let mut valid = true;
            for i in 0..l {
                let a = at[k1 + i];
                let b = at[k2 + i];
                if (b >= 0 && u_mat.get(b as usize, k1 + i) <= (IDP_FORBID_HALF as f64))
                    || (a >= 0 && u_mat.get(a as usize, k2 + i) <= (IDP_FORBID_HALF as f64))
                {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }
            let before = eval_block_slots(at, u_mat, p_mat, k1, k2, l);
            for i in 0..l {
                let a = at[k1 + i];
                let b = at[k2 + i];
                at[k1 + i] = b;
                at[k2 + i] = a;
                if a >= 0 {
                    pos[a as usize] = (k2 + i) as i32;
                }
                if b >= 0 {
                    pos[b as usize] = (k1 + i) as i32;
                }
            }
            let delta = eval_block_slots(at, u_mat, p_mat, k1, k2, l) - before;
            if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best_sc {
                    best_sc = cur;
                    best_at.copy_from_slice(at);
                    best_pos.copy_from_slice(pos);
                }
            } else {
                for i in 0..l {
                    let b = at[k1 + i];
                    let a = at[k2 + i];
                    at[k1 + i] = a;
                    at[k2 + i] = b;
                    if a >= 0 {
                        pos[a as usize] = (k1 + i) as i32;
                    }
                    if b >= 0 {
                        pos[b as usize] = (k2 + i) as i32;
                    }
                }
            }
        }
    }

    state.2 = cur;
    (best_at, best_pos, best_sc)
}

/// Evolutionary conflict-free segment crossover injecting `donor.0[k_lo..=k_hi]` into `recipient`
/// and repairing any displaced roots greedily.
pub fn crossover_idp_states(
    recipient: &IdpReplicaState,
    donor: &IdpReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    rng: &mut Rng,
) -> IdpReplicaState {
    let k_len = recipient.0.len();
    let s_len = recipient.1.len();
    if k_len < 5 || s_len == 0 {
        return recipient.clone();
    }
    let mut at = recipient.0.clone();
    let mut pos = recipient.1.clone();

    let seg_len = (k_len / 5).clamp(3, 28).min(k_len - 1);
    let k_lo = 1 + rng.usize((k_len - seg_len).max(1));
    let k_hi = (k_lo + seg_len - 1).min(k_len - 1);

    // 1. Splice donor segment `[k_lo..=k_hi]`, clearing any conflicting placements cleanly
    for slot_val in &mut at[k_lo..=k_hi] {
        let old_s = *slot_val;
        if old_s >= 0 {
            pos[old_s as usize] = -1;
            *slot_val = -1;
        }
    }
    for k in k_lo..=k_hi {
        let don_s = donor.0[k];
        if don_s >= 0 && u_mat.get(don_s as usize, k) > (IDP_FORBID_HALF as f64) {
            let prev_k = pos[don_s as usize];
            if prev_k >= 0 {
                at[prev_k as usize] = -1;
            }
            at[k] = don_s;
            pos[don_s as usize] = k as i32;
        }
    }

    // 2. Re-insert displaced unassigned roots into best available slots
    for s in 0..s_len {
        if pos[s] >= 0 {
            continue;
        }
        let mut best_k: Option<usize> = None;
        let mut best_v = -0.25f64;
        for &ku in &slots[s] {
            let k = ku as usize;
            if at[k] >= 0 {
                continue;
            }
            let mut v = u_mat.get(s, k);
            if k > 1 && at[k - 1] >= 0 {
                v += p_mat.get(at[k - 1] as usize, s);
            }
            if k + 1 < k_len && at[k + 1] >= 0 {
                v += p_mat.get(s, at[k + 1] as usize);
            }
            if v > best_v {
                best_v = v;
                best_k = Some(k);
            }
        }
        if let Some(k) = best_k {
            at[k] = s as i32;
            pos[s] = k as i32;
        }
    }

    let sc = evaluate_idp_total_score(&at, u_mat, p_mat);
    (at, pos, sc)
}

/// Monotone piecewise-linear calibration map for `13C`-detected IDP local+ensemble posteriors.
pub const IDP_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.00, 0.00),
    (0.10, 0.09),
    (0.25, 0.25),
    (0.45, 0.46),
    (0.65, 0.68),
    (0.80, 0.825),
    (0.90, 0.92),
    (0.96, 0.97),
    (1.00, 0.994),
];

#[inline]
fn eval_idp_calibration(p_raw: f64) -> f64 {
    crate::qc::eval_monotone_calibration_map(p_raw, IDP_CALIBRATION_KNOTS)
}

/// Combine completed CPU or WebGPU replica assignments via Boltzmann consensus, exact local-swap /
/// motif-swap specificity calibration, and unlabelled `DatasetQcVerdict` (`Pass`, `Warn`, `Abstain`).
#[allow(clippy::too_many_arguments)]
pub fn consensus_and_calibrate_idp(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    replica_positions: &[Vec<i32>],
    replica_scores: &[f64],
    fitted_offsets: IdpShiftOffsets,
    cfg: &IdpSolverConfig,
) -> IdpAssignResult {
    let s_len = systems.len();
    let k_len = sequence.len();
    if s_len == 0 || k_len < 2 || replica_positions.is_empty() {
        return IdpAssignResult {
            position: vec![-1; s_len],
            confidence: vec![0.0; s_len],
            score: 0.0,
            run_scores: replica_scores.to_vec(),
            fitted_offsets,
            qc_verdict: DatasetQcVerdict::Abstain,
            support_3d_frac: 0.0,
            connectivity_frac: 0.0,
            qc_reasons: vec!["Insufficient spin systems or sequence length (< 2)".to_string()],
        };
    }

    // Polish top replicas cleanly and rank by exact global score
    let mut polished: Vec<(Vec<i32>, Vec<i32>, f64)> = Vec::with_capacity(replica_positions.len());
    for pos_in in replica_positions {
        let mut at = vec![-1i32; k_len];
        let mut pos = vec![-1i32; s_len];
        for (s, &k) in pos_in.iter().enumerate().take(s_len) {
            if k >= 1 && (k as usize) < k_len && at[k as usize] < 0 {
                at[k as usize] = s as i32;
                pos[s] = k;
            }
        }
        let sc = polish_idp_state(&mut at, &mut pos, u_mat, p_mat, slots);
        polished.push((at, pos, sc));
    }

    let run_scores: Vec<f64> = polished.iter().map(|r| r.2).collect();
    let mut ranked: Vec<usize> = (0..polished.len()).collect();
    ranked.sort_by(|&a, &b| polished[b].2.total_cmp(&polished[a].2));

    let top_n = (polished.len() / 2).max(4).min(polished.len());
    let max_sc = polished[ranked[0]].2;
    const TAU_ENS: f64 = 6.5;
    let mut ens_weights = Vec::with_capacity(top_n);
    let mut w_sum = 0.0f64;
    for &idx in &ranked[..top_n] {
        let w = ((polished[idx].2 - max_sc) / TAU_ENS).max(-30.0).exp();
        ens_weights.push((idx, w));
        w_sum += w;
    }
    let w_sum = w_sum.max(1e-12);

    // 1. Conflict-free consensus injection from Boltzmann-weighted replica marginals
    let mut cand_pairs: Vec<(f64, usize, i32)> = Vec::new();
    let mut unassign_p = vec![0.0f64; s_len];
    for (s, unassign_slot) in unassign_p.iter_mut().enumerate() {
        let mut counts: BTreeMap<i32, f64> = BTreeMap::new();
        for &(r_idx, w) in &ens_weights {
            *counts.entry(polished[r_idx].1[s]).or_insert(0.0) += w;
        }
        for (k, wt) in counts {
            let p = wt / w_sum;
            if k < 0 {
                *unassign_slot = p;
            } else {
                cand_pairs.push((p, s, k));
            }
        }
    }
    cand_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut cons_pos = vec![-1i32; s_len];
    let mut cons_at = vec![-1i32; k_len];
    for &(p, s, k) in &cand_pairs {
        if cons_pos[s] < 0 && cons_at[k as usize] < 0 && p >= unassign_p[s] {
            cons_pos[s] = k;
            cons_at[k as usize] = s as i32;
        }
    }
    let mut final_sc = polish_idp_state(&mut cons_at, &mut cons_pos, u_mat, p_mat, slots);
    if max_sc > final_sc + 0.25 {
        cons_at.copy_from_slice(&polished[ranked[0]].0);
        cons_pos.copy_from_slice(&polished[ranked[0]].1);
        final_sc = max_sc;
    }

    // Recompute replica consensus support `p_ens[s]` on final `cons_pos[s]`
    let mut p_ens = vec![0.0f64; s_len];
    for s in 0..s_len {
        let k = cons_pos[s];
        if k >= 0 {
            let mut wt = 0.0f64;
            for &(r_idx, w) in &ens_weights {
                if polished[r_idx].1[s] == k {
                    wt += w;
                }
            }
            p_ens[s] = wt / w_sum;
        }
    }

    // 2. Exact local-swap + dipeptide-block-swap Boltzmann specificity `p_local[s]`
    // Evaluates all alternative placements `k' != k*` (single swap / relocation, unassign, and
    // 2-residue motif swap `(k*, k*+1) <-> (k', k'+1)`) so degenerate repeats receive honest <0.80 uncertainty!
    const TAU_LOCAL: f64 = 2.35;
    let mut confidence = vec![0.0f64; s_len];
    let mut work_at = cons_at.clone();

    for s in 0..s_len {
        let k_i = cons_pos[s];
        if k_i < 0 {
            confidence[s] = 0.0;
            continue;
        }
        let k_star = k_i as usize;
        let mut alt_sum = 0.0f64;

        // (a) Unassigning `s -> -1`
        let base_1 = eval_1_slot(&work_at, u_mat, p_mat, k_star);
        alt_sum += ((-base_1 - 2.0) / TAU_LOCAL).clamp(-35.0, 20.0).exp();

        // (b) All single-slot relocations / 2-swaps `k_star <-> k_alt`
        for &k_alt_u in &slots[s] {
            let k_alt = k_alt_u as usize;
            if k_alt == k_star {
                continue;
            }
            let t = work_at[k_alt];
            if t >= 0 && u_mat.get(t as usize, k_star) <= (IDP_FORBID_HALF as f64) {
                continue;
            }
            let before = eval_2_slots(&work_at, u_mat, p_mat, k_star, k_alt);
            work_at[k_star] = t;
            work_at[k_alt] = s as i32;
            let delta = eval_2_slots(&work_at, u_mat, p_mat, k_star, k_alt) - before;
            work_at[k_star] = s as i32;
            work_at[k_alt] = t;
            alt_sum += (delta / TAU_LOCAL).clamp(-35.0, 20.0).exp();
        }

        // (c) Best contiguous `2..=5`-residue motif swap containing `k_star` so degenerate
        // dipeptide..pentapeptide repeats receive honest `< 0.80` uncertainty without penalizing
        // uniquely flanked motifs (e.g. `-Pro-Pro-` with distinct flanking residues).
        let mut best_blk_delta = f64::NEG_INFINITY;
        for blen in 2..=5usize.min(k_len.saturating_sub(1)) {
            let win_lo = k_star.saturating_sub(blen - 1).max(1);
            let win_hi = k_star.min(k_len - blen);
            if win_lo > win_hi {
                continue;
            }
            for k_win in win_lo..=win_hi {
                for k_alt in 1..=(k_len - blen) {
                    if k_alt.abs_diff(k_win) < blen {
                        continue;
                    }
                    let mut valid = true;
                    for i in 0..blen {
                        let a = work_at[k_win + i];
                        let b = work_at[k_alt + i];
                        if (b >= 0 && u_mat.get(b as usize, k_win + i) <= (IDP_FORBID_HALF as f64))
                            || (a >= 0
                                && u_mat.get(a as usize, k_alt + i) <= (IDP_FORBID_HALF as f64))
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let before = eval_block_slots(&work_at, u_mat, p_mat, k_win, k_alt, blen);
                    for i in 0..blen {
                        work_at.swap(k_win + i, k_alt + i);
                    }
                    let delta =
                        eval_block_slots(&work_at, u_mat, p_mat, k_win, k_alt, blen) - before;
                    for i in 0..blen {
                        work_at.swap(k_win + i, k_alt + i);
                    }
                    if delta > best_blk_delta {
                        best_blk_delta = delta;
                    }
                }
            }
        }
        if best_blk_delta > -4.0 {
            alt_sum += (best_blk_delta / (1.15 * TAU_LOCAL))
                .clamp(-35.0, 20.0)
                .exp();
        }

        let p_local = 1.0 / (1.0 + alt_sum);
        // Require both local Boltzmann specificity and multi-island ensemble agreement for HC (>=0.80)
        // while preserving >=0.55 posterior on uniquely polished sequential placements.
        let p_combined = 0.64 * p_local + 0.36 * (p_local * p_ens[s]).sqrt();
        confidence[s] = eval_idp_calibration(p_combined);
    }

    // 3. Unlabelled Dataset Quality-Control Verdict (`Pass`, `Warn`, `Abstain`)
    let n_3d_sup = systems.iter().filter(|s| s.has_3d_support()).count();
    let support_3d_frac = (n_3d_sup as f64) / (s_len as f64);

    let mut n_assigned = 0usize;
    let mut n_connected = 0usize;
    for (s, &k_i) in cons_pos.iter().enumerate() {
        if k_i >= 1 {
            n_assigned += 1;
            let k = k_i as usize;
            let prev_ok =
                k > 1 && cons_at[k - 1] >= 0 && p_mat.get(cons_at[k - 1] as usize, s) >= 1.0;
            let next_ok = k + 1 < k_len
                && cons_at[k + 1] >= 0
                && p_mat.get(s, cons_at[k + 1] as usize) >= 1.0;
            if prev_ok || next_ok {
                n_connected += 1;
            }
        }
    }
    let connectivity_frac = if n_assigned > 0 {
        (n_connected as f64) / (n_assigned as f64)
    } else {
        0.0
    };

    let mut qc_reasons = Vec::new();
    let qc_verdict = if support_3d_frac < 0.25 {
        qc_reasons.push(format!(
            "Catastrophic 3D strip support loss ({:.1}% < 25.0% of CON roots supported)",
            support_3d_frac * 100.0
        ));
        DatasetQcVerdict::Abstain
    } else if s_len >= 6 && connectivity_frac < 0.15 {
        qc_reasons.push(format!(
            "Sequential dipeptide connectivity collapsed ({:.1}% < 15.0% linked)",
            connectivity_frac * 100.0
        ));
        DatasetQcVerdict::Abstain
    } else if support_3d_frac < 0.55 || (s_len >= 6 && connectivity_frac < 0.40) {
        qc_reasons.push(format!(
            "Partial 3D support ({:.1}%) or sparse sequential connectivity ({:.1}%)",
            support_3d_frac * 100.0,
            connectivity_frac * 100.0
        ));
        DatasetQcVerdict::Warn
    } else {
        DatasetQcVerdict::Pass
    };

    match qc_verdict {
        DatasetQcVerdict::Abstain => {
            let cap = (cfg.hc_threshold - 0.35).clamp(0.10, 0.45);
            for c in &mut confidence {
                *c = (*c * 0.45).min(cap);
            }
        }
        DatasetQcVerdict::Warn => {
            for c in &mut confidence {
                *c = (*c * 0.88).clamp(0.0, 0.96);
            }
        }
        DatasetQcVerdict::Pass => {}
    }

    IdpAssignResult {
        position: cons_pos,
        confidence,
        score: final_sc,
        run_scores,
        fitted_offsets,
        qc_verdict,
        support_3d_frac,
        connectivity_frac,
        qc_reasons,
    }
}

/// Multi-island evolutionary + detailed-balance simulated annealing CPU solver for continuous
/// through-Proline `13C`-detected IDP assignment (`T11.2`).
pub fn assign_idp_13c_cpu(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> IdpAssignResult {
    let s_len = systems.len();
    let k_len = sequence.len();
    let fitted_offsets = if cfg.auto_reference_offset {
        estimate_idp_reference_offsets(systems, sequence, cfg)
    } else {
        cfg.manual_offsets
    };

    if s_len == 0 || k_len < 2 {
        return consensus_and_calibrate_idp(
            systems,
            sequence,
            &Matrix2D::filled(0, 0, 0.0),
            &Matrix2D::filled(0, 0, 0.0),
            &[],
            &[],
            &[],
            fitted_offsets,
            cfg,
        );
    }

    let u_mat = idp_unary_matrix(systems, sequence, cfg, &fitted_offsets);
    let p_mat = idp_pair_matrix(systems, cfg);
    let slots = build_idp_valid_slots(&u_mat);

    let n_islands = cfg.islands.max(1);
    let total_replicas = cfg.replicas.max(n_islands).max(4);
    let gens = cfg.generations.max(1);
    let total_steps = cfg
        .steps_per_system
        .saturating_mul(s_len.max(24))
        .max(20_000);
    let steps_per_gen = (total_steps / gens).max(4_000);

    let mut population =
        build_idp_initial_replicas(&u_mat, &p_mat, &slots, total_replicas, cfg.seed);
    let mut master_rng = Rng::new(cfg.seed ^ 0x4350_555F_4944_5031);

    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;
    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(total_replicas)
        .max(1);

    for gen_idx in 0..gens {
        let gen_t_start = if gen_idx == 0 {
            cfg.t_start
        } else {
            (cfg.t_start * 0.38_f64.powi(gen_idx as i32)).max(cfg.t_end * 4.0)
        };
        let seeds: Vec<u64> = (0..total_replicas).map(|_| master_rng.next_u64()).collect();

        if n_threads <= 1 {
            for (r, sd) in seeds.into_iter().enumerate() {
                let mut r_rng = Rng::new(sd);
                let (bat, bpos, bsc) = run_idp_cpu_anneal_segment(
                    &mut population[r],
                    &u_mat,
                    &p_mat,
                    &slots,
                    steps_per_gen,
                    gen_t_start,
                    cfg.t_end,
                    &mut r_rng,
                );
                population[r] = (bat, bpos, bsc);
            }
        } else {
            let chunk_sz = total_replicas.div_ceil(n_threads);
            let u_ref = &u_mat;
            let p_ref = &p_mat;
            let sl_ref = &slots;
            let t_end = cfg.t_end;
            let indexed: Vec<(usize, IdpReplicaState, u64)> = population
                .into_iter()
                .zip(seeds)
                .enumerate()
                .map(|(i, (st, sd))| (i, st, sd))
                .collect();

            let batches: Vec<Vec<(usize, IdpReplicaState)>> = thread::scope(|scope| {
                let mut handles = Vec::new();
                for chunk in indexed.chunks(chunk_sz) {
                    handles.push(scope.spawn(move || {
                        let mut out = Vec::with_capacity(chunk.len());
                        for &(idx, ref st_in, sd) in chunk {
                            let mut st = st_in.clone();
                            let mut r_rng = Rng::new(sd);
                            let (bat, bpos, bsc) = run_idp_cpu_anneal_segment(
                                &mut st,
                                u_ref,
                                p_ref,
                                sl_ref,
                                steps_per_gen,
                                gen_t_start,
                                t_end,
                                &mut r_rng,
                            );
                            out.push((idx, (bat, bpos, bsc)));
                        }
                        out
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            population = vec![(Vec::new(), Vec::new(), 0.0); total_replicas];
            for batch in batches {
                for (idx, st) in batch {
                    population[idx] = st;
                }
            }
        }

        // Polish elites and run island evolutionary segment crossover between generations
        if gen_idx + 1 < gens && total_replicas >= 4 {
            let per_island = (total_replicas / n_islands).max(2);
            for isl in 0..n_islands {
                let lo = isl * per_island;
                let hi = if isl + 1 == n_islands {
                    total_replicas
                } else {
                    ((isl + 1) * per_island).min(total_replicas)
                };
                if hi <= lo + 1 {
                    continue;
                }
                population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                // Polish top two island elites before segment crossover
                for elite in &mut population[lo..(lo + 2).min(hi)] {
                    elite.2 = polish_idp_state(&mut elite.0, &mut elite.1, &u_mat, &p_mat, &slots);
                }
                population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                let elite_clone = population[lo].clone();
                let second_clone = population[lo + 1].clone();
                // Replace bottom half of island with segment crossovers of elites
                let mid = lo + (hi - lo) / 2;
                for idx in mid..hi {
                    let donor = if idx % 2 == 0 {
                        &elite_clone
                    } else {
                        &second_clone
                    };
                    population[idx] = crossover_idp_states(
                        &population[lo + (idx - mid) % (mid - lo)],
                        donor,
                        &u_mat,
                        &p_mat,
                        &slots,
                        &mut master_rng,
                    );
                }
            }
            // Ring migration of best elite across adjacent islands
            if n_islands >= 2 && per_island >= 3 {
                let elites: Vec<IdpReplicaState> = (0..n_islands)
                    .map(|isl| population[isl * per_island].clone())
                    .collect();
                for (isl, elite_st) in elites.into_iter().enumerate() {
                    let target_isl = (isl + 1) % n_islands;
                    let replace_idx =
                        (target_isl * per_island + per_island - 1).min(total_replicas - 1);
                    population[replace_idx] = elite_st;
                }
            }
        }
    }

    let rep_positions: Vec<Vec<i32>> = population.iter().map(|st| st.1.clone()).collect();
    let rep_scores: Vec<f64> = population.iter().map(|st| st.2).collect();

    consensus_and_calibrate_idp(
        systems,
        sequence,
        &u_mat,
        &p_mat,
        &slots,
        &rep_positions,
        &rep_scores,
        fitted_offsets,
        cfg,
    )
}

/// End-to-end convenience pipeline: register inter-experiment drift across `picked`, build
/// `ConDipeptideSystem`s, and run `assign_idp_13c_cpu`.
pub fn assign_idp_13c_from_picked(
    picked: &C13PickedDataset,
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> (Vec<ConDipeptideSystem>, IdpAssignResult) {
    let systems = register_and_build_con_dipeptide_systems(picked);
    let result = assign_idp_13c_cpu(&systems, sequence, cfg);
    (systems, result)
}

/// Convert a `13C`-detected IDP assignment (`ConDipeptideSystem`s placed on `k = 1..L-1`,
/// continuously through Prolines!) into a residue-indexed `Reference` chemical shift table.
pub fn idp_to_reference(
    result: &IdpAssignResult,
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    seq_ids: &[i32],
    min_conf: f64,
    de_offset: bool,
) -> (Reference, BTreeMap<usize, f64>) {
    let mut out = Reference::new(sequence.to_vec(), seq_ids.to_vec());
    let mut conf_map: BTreeMap<usize, f64> = BTreeMap::new();
    let off = if de_offset {
        result.fitted_offsets
    } else {
        IdpShiftOffsets::default()
    };

    let mut at: BTreeMap<usize, usize> = BTreeMap::new();
    for (s, &k_i) in result.position.iter().enumerate() {
        if k_i >= 1 && result.confidence[s] >= min_conf {
            at.insert(k_i as usize, s);
        }
    }

    for (&k, &s) in &at {
        let sys = &systems[s];
        let c = result.confidence[s];
        conf_map.insert(k, c);
        out.shifts
            .insert((k, "N".to_string()), sys.n_curr - off.d_n);
        out.shifts
            .insert((k - 1, "C".to_string()), sys.c_prev - off.d_c);
        conf_map.entry(k - 1).or_insert(c);

        // `CA(k)`: corroborate against `at[k+1].ca_prev` when `k + 1` is assigned
        if !sys.ca_curr.is_empty() {
            let ca_val = if let Some(&s_nxt) = at.get(&(k + 1)) {
                let nxt_prev = &systems[s_nxt].ca_prev;
                if !nxt_prev.is_empty() {
                    sys.ca_curr
                        .iter()
                        .copied()
                        .min_by(|&a, &b| {
                            let da = nxt_prev
                                .iter()
                                .map(|&p| (a - p).abs())
                                .fold(f64::INFINITY, f64::min);
                            let db = nxt_prev
                                .iter()
                                .map(|&p| (b - p).abs())
                                .fold(f64::INFINITY, f64::min);
                            da.total_cmp(&db)
                        })
                        .unwrap_or(sys.ca_curr[0])
                } else {
                    sys.ca_curr[0]
                }
            } else {
                sys.ca_curr[0]
            };
            out.shifts.insert((k, "CA".to_string()), ca_val - off.d_ca);
        }

        if !sys.ca_prev.is_empty() && !out.shifts.contains_key(&(k - 1, "CA".to_string())) {
            out.shifts
                .insert((k - 1, "CA".to_string()), sys.ca_prev[0] - off.d_ca);
        }
        if !sys.cb_prev.is_empty() && !sequence[k - 1].eq_ignore_ascii_case("GLY") {
            out.shifts
                .insert((k - 1, "CB".to_string()), sys.cb_prev[0] - off.d_cb);
        }
        if !sys.c_curr.is_empty() && !out.shifts.contains_key(&(k, "C".to_string())) {
            out.shifts
                .insert((k, "C".to_string()), sys.c_curr[0] - off.d_c);
        }
    }

    (out, conf_map)
}
