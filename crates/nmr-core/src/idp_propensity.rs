//! Secondary Structure Propensity (`SSP`) and 4-state conformational ensemble populations
//! (`δ2D`: α-helix, β-strand, polyproline II `PPII`, and random coil) from backbone NMR
//! chemical shifts of intrinsically disordered and folded proteins.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature:
//! 1. Marsh, J. A., Singh, V. K., Jia, Z. & Forman-Kay, J. D. (2006). "Sensitivity of secondary
//!    structure propensities to sequence differences between α- and γ-synuclein: Implications for
//!    fibrillation." *Protein Sci.* 15(12), 2795–2804 (`SSP` 5-residue weighted secondary-shift
//!    propensity score).
//! 2. Camilloni, C., De Simone, A., Vranken, W. F. & Vendruscolo, M. (2012). "Determination of
//!    secondary structure populations in disordered states of proteins using nuclear magnetic
//!    resonance chemical shifts." *Biochemistry* 51(11), 2224–2231 (`δ2D` 4-state conformational
//!    population predictor for α-helix, β-strand, `PPII`, and random coil).
//! 3. Tamiola, K., Acar, B. & Mulder, F. A. A. (2010). "Sequence-specific random coil chemical
//!    shifts of intrinsically disordered proteins." *J. Am. Chem. Soc.* 132(51), 18000–18003
//!    (`ncIDP` neighbor-corrected random-coil chemical shift library).
//! 4. Temperature and pH random-coil adjustments follow Kjaergaard, M., Brander, S. & Poulsen,
//!    F. M. (2011), *J. Biomol. NMR* 49, 139–149, and Platzer, G., Okon, M. & McIntosh, L. P.
//!    (2014), *J. Biomol. NMR* 60, 109–129 (Henderson–Hasselbalch sidechain titration shifts).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::report::SecondaryStructureClass;

/// Canonical 4-state backbone conformational basin (`Camilloni et al., Biochemistry 2012`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConformationalState {
    /// Right-handed α-helix / $3_{10}$-helix basin (`H`).
    Alpha,
    /// Extended β-strand / β-bridge basin (`E`).
    Beta,
    /// Polyproline II extended left-handed helix basin (`P`, $(\phi, \psi) \approx (-75^\circ, +145^\circ)$).
    Ppii,
    /// Unstructured random-coil basin (`C`).
    Coil,
}

impl ConformationalState {
    /// Single-character state code (`H`, `E`, `P`, `C`).
    #[inline]
    pub fn code(self) -> &'static str {
        match self {
            Self::Alpha => "H",
            Self::Beta => "E",
            Self::Ppii => "P",
            Self::Coil => "C",
        }
    }

    /// Human-readable conformational state label.
    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::Alpha => "Alpha-Helix",
            Self::Beta => "Beta-Strand",
            Self::Ppii => "Polyproline-II",
            Self::Coil => "Random-Coil",
        }
    }

    /// Project the 4-state `δ2D` classification onto classic 3-state secondary structure
    /// (`Alpha -> Helix`, `Beta -> Strand`, `Ppii | Coil -> Coil`).
    #[inline]
    pub fn to_three_state(self) -> SecondaryStructureClass {
        match self {
            Self::Alpha => SecondaryStructureClass::Helix,
            Self::Beta => SecondaryStructureClass::Strand,
            Self::Ppii | Self::Coil => SecondaryStructureClass::Coil,
        }
    }
}

/// Prior population distribution assigned to residues lacking any experimental backbone shifts
/// (`has_data == false`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnassignedPriorMode {
    /// Disordered coil/PPII baseline `(p_alpha = 0.05, p_beta = 0.05, p_ppii = 0.25, p_coil = 0.65)`.
    #[default]
    CoilBaseline,
    /// Maximum-entropy uniform simplex `(0.25, 0.25, 0.25, 0.25)`.
    Uniform,
}

/// Global chemical-shift referencing offsets `(obs_ppm - dss_rc_ppm)` per backbone nucleus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GlobalShiftOffsets {
    pub d_ca: f64,
    pub d_cb: f64,
    pub d_c: f64,
    pub d_ha: f64,
    pub d_h: f64,
    pub d_n: f64,
}

impl GlobalShiftOffsets {
    #[inline]
    pub fn get(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.d_ca,
            "CB" => self.d_cb,
            "C" => self.d_c,
            "HA" => self.d_ha,
            "H" => self.d_h,
            "N" => self.d_n,
            _ => 0.0,
        }
    }
}

/// Configuration for `SSP` and `δ2D` conformational population calculation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropensityConfig {
    /// Sample temperature in Kelvin (reference `ncIDP` temperature is `298.15 K` / `25 °C`).
    pub temperature_kelvin: f64,
    /// Sample pH (`None` defaults to `6.5`, with optional automatic acidic `Asp`/`Glu` protonation sensing).
    pub sample_ph: Option<f64>,
    /// When `sample_ph` is `None`, automatically sense low-pH carboxyl protonation (`pH < 3.5`)
    /// from simultaneous `Asp`/`Glu` upfield `CA` and `CB` shifts (`Platzer et al. 2014`).
    pub auto_acidic_ph_detection: bool,
    /// Automatically estimate and subtract uniform unlabelled chemical-shift referencing offsets
    /// (e.g. TMS vs DSS `13C` offset) using secondary-structure-invariant shift combinations.
    pub auto_offset_correction: bool,
    /// Explicit manual chemical-shift offsets added to random-coil expectations.
    pub manual_offsets: GlobalShiftOffsets,
    /// Sliding window width for `Marsh et al. (2006)` `SSP` smoothing (canonical `5` -> `i-2..=i+2`).
    pub ssp_window: usize,
    /// Prior distribution emitted on unassigned residues (`has_data == false`).
    pub unassigned_prior: UnassignedPriorMode,
}

impl Default for PropensityConfig {
    fn default() -> Self {
        Self {
            temperature_kelvin: 298.15,
            sample_ph: None,
            auto_acidic_ph_detection: true,
            auto_offset_correction: false,
            manual_offsets: GlobalShiftOffsets::default(),
            ssp_window: 5,
            unassigned_prior: UnassignedPriorMode::CoilBaseline,
        }
    }
}

impl PropensityConfig {
    pub fn with_temperature_kelvin(mut self, temp_k: f64) -> Self {
        self.temperature_kelvin = temp_k;
        self
    }

    pub fn with_sample_ph(mut self, ph: f64) -> Self {
        self.sample_ph = Some(ph);
        self
    }

    pub fn with_auto_offset_correction(mut self, enabled: bool) -> Self {
        self.auto_offset_correction = enabled;
        self
    }

    pub fn with_unassigned_prior(mut self, mode: UnassignedPriorMode) -> Self {
        self.unassigned_prior = mode;
        self
    }
}

/// Per-residue `SSP` score and normalized 4-state `δ2D` conformational populations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResiduePropensity {
    /// Zero-based index into `Reference::sequence`.
    pub seq_index: usize,
    /// Author / BMRB residue number (`Seq_ID`).
    pub seq_id: i32,
    /// Three-letter amino-acid code (`ALA`, `GLY`, `PRO`, ...).
    pub residue: String,
    /// `true` if at least one backbone chemical shift (`CA, CB, C, HA, H, N`) was observed on this residue;
    /// `false` if unassigned (in which case `(p_alpha, p_beta, p_ppii, p_coil)` carries the explicit prior).
    pub has_data: bool,
    /// Number of observed backbone nuclei contributing at this residue (`0..=6`).
    pub n_observed_nuclei: usize,
    /// `Marsh et al. (2006)` Secondary Structure Propensity score in `[-1.0, +1.0]`
    /// (`+1` = fully formed α-helix, `-1` = fully formed β-strand; `None` if window has no data).
    pub ssp_score: Option<f64>,
    /// Normalized α-helix population $p_\alpha \in [0, 1]$.
    pub p_alpha: f64,
    /// Normalized β-strand population $p_\beta \in [0, 1]$.
    pub p_beta: f64,
    /// Normalized polyproline II population $p_{\text{PPII}} \in [0, 1]$.
    pub p_ppii: f64,
    /// Normalized random-coil population $p_{\text{coil}} \in [0, 1]$.
    pub p_coil: f64,
    /// Highest-probability conformational state (`Coil` when `has_data == false`).
    pub dominant_state: ConformationalState,
    /// Neighbor-, pH-, temperature-, and offset-corrected secondary shifts $\Delta\delta_X$ (ppm).
    pub secondary_shifts: BTreeMap<String, f64>,
}

impl ResiduePropensity {
    /// Sum of the 4 conformational state populations (guaranteed `1.0 +- 1e-9`).
    #[inline]
    pub fn population_sum(&self) -> f64 {
        self.p_alpha + self.p_beta + self.p_ppii + self.p_coil
    }
}

/// Complete protein-wide `SSP` and `δ2D` ensemble propensity analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropensityReport {
    pub residues: Vec<ResiduePropensity>,
    pub effective_ph: f64,
    pub temperature_kelvin: f64,
    pub applied_offsets: GlobalShiftOffsets,
    pub n_residues_with_data: usize,
    pub mean_p_alpha: f64,
    pub mean_p_beta: f64,
    pub mean_p_ppii: f64,
    pub mean_p_coil: f64,
    pub mean_abs_ssp: f64,
    pub is_predominantly_disordered: bool,
}

#[derive(Clone, Copy, Debug)]
struct RcValues {
    ca: f64,
    cb: Option<f64>,
    c: f64,
    ha: f64,
    h: Option<f64>,
    n: Option<f64>,
}

/// Base `ncIDP` / DSS random-coil chemical shifts (ppm) at `298.15 K`, `pH 6.5`
/// (`Tamiola, Acar & Mulder, J. Am. Chem. Soc. 2010, 132, 18000–18003`).
fn base_ncidp_random_coil(res: &str, cys_oxidized: bool) -> RcValues {
    let up = res.trim().to_ascii_uppercase();
    let (ca, cb, c, ha, h, n) = match up.as_str() {
        "ALA" => (52.48, Some(19.07), 177.82, 4.32, Some(8.26), Some(123.8)),
        "ARG" => (56.02, Some(30.76), 176.35, 4.34, Some(8.26), Some(121.3)),
        "ASN" => (53.12, Some(38.74), 175.24, 4.74, Some(8.40), Some(118.9)),
        "ASP" => (54.22, Some(41.05), 176.32, 4.62, Some(8.32), Some(120.4)),
        "CYS" => {
            if cys_oxidized {
                (55.55, Some(41.10), 174.60, 4.71, Some(8.35), Some(118.8))
            } else {
                (58.25, Some(28.02), 174.65, 4.55, Some(8.32), Some(118.8))
            }
        }
        "GLN" => (55.75, Some(29.35), 176.04, 4.34, Some(8.34), Some(119.9)),
        "GLU" => (56.62, Some(30.02), 176.62, 4.33, Some(8.38), Some(120.5)),
        "GLY" => (45.14, None, 173.95, 3.97, Some(8.33), Some(109.1)),
        "HIS" => (55.52, Some(29.25), 174.85, 4.68, Some(8.38), Some(118.6)),
        "ILE" => (61.12, Some(38.68), 176.42, 4.17, Some(8.04), Some(120.2)),
        "LEU" => (55.12, Some(42.32), 177.55, 4.35, Some(8.18), Some(122.0)),
        "LYS" => (56.24, Some(32.95), 176.62, 4.32, Some(8.29), Some(121.1)),
        "MET" => (55.42, Some(32.78), 176.28, 4.48, Some(8.28), Some(119.8)),
        "PHE" => (57.74, Some(39.55), 175.82, 4.62, Some(8.28), Some(120.5)),
        "PRO" => (63.28, Some(32.02), 177.26, 4.42, None, None),
        "SER" => (58.30, Some(63.76), 174.65, 4.47, Some(8.31), Some(115.9)),
        "THR" => (61.78, Some(69.75), 174.72, 4.35, Some(8.16), Some(113.9)),
        "TRP" => (57.42, Some(29.58), 176.12, 4.66, Some(8.22), Some(121.4)),
        "TYR" => (57.88, Some(38.75), 175.88, 4.55, Some(8.14), Some(120.4)),
        "VAL" => (62.22, Some(32.82), 176.32, 4.12, Some(8.06), Some(119.6)),
        _ => (56.00, Some(33.00), 176.00, 4.35, Some(8.25), Some(120.0)),
    };
    RcValues {
        ca,
        cb,
        c,
        ha,
        h,
        n,
    }
}

/// Sequence-specific nearest-neighbor (`i-1` and `i+1`), pH Henderson–Hasselbalch, and
/// linear temperature adjustments (`Tamiola et al. 2010`; `Kjaergaard et al. 2011`; `Platzer et al. 2014`).
fn adjusted_random_coil(
    sequence: &[String],
    k: usize,
    cys_oxidized: bool,
    ph: f64,
    temp_kelvin: f64,
) -> RcValues {
    let res = sequence[k].trim().to_ascii_uppercase();
    let prev = if k > 0 {
        Some(sequence[k - 1].trim().to_ascii_uppercase())
    } else {
        None
    };
    let next = if k + 1 < sequence.len() {
        Some(sequence[k + 1].trim().to_ascii_uppercase())
    } else {
        None
    };

    let mut rc = base_ncidp_random_coil(&res, cys_oxidized);

    // 1. Succeeding residue (i+1) covalent neighbor effects (Tamiola et al. 2010, Tables S1-S6)
    if let Some(ref nx) = next {
        if nx == "PRO" {
            if res == "GLY" {
                rc.ca -= 1.00;
                rc.c -= 1.50;
            } else {
                rc.ca -= 1.95;
                rc.c -= 2.05;
                if let Some(ref mut cb) = rc.cb {
                    *cb += 0.22;
                }
            }
            rc.ha += 0.24;
            if let Some(ref mut n) = rc.n {
                *n += 1.05;
            }
        } else if matches!(nx.as_str(), "VAL" | "ILE" | "THR") {
            rc.c -= 0.22;
            if let Some(ref mut n) = rc.n {
                *n -= 0.35;
            }
        } else if matches!(nx.as_str(), "PHE" | "TYR" | "TRP") {
            rc.ca -= 0.12;
            rc.c -= 0.18;
            if let Some(ref mut h) = rc.h {
                *h -= 0.05;
            }
        } else if nx == "GLY" {
            rc.c += 0.15;
        }
    }

    // 2. Preceding residue (i-1) covalent peptide-bond effects (primarily 15N and 1HN)
    if let Some(ref pv) = prev {
        match pv.as_str() {
            "GLY" => {
                if let Some(ref mut n) = rc.n {
                    *n -= 0.95;
                }
                if let Some(ref mut h) = rc.h {
                    *h += 0.04;
                }
            }
            "VAL" | "ILE" | "THR" => {
                if let Some(ref mut n) = rc.n {
                    *n += 2.15;
                }
                if let Some(ref mut h) = rc.h {
                    *h += 0.07;
                }
                rc.ca -= 0.12;
            }
            "PHE" | "TYR" | "TRP" => {
                if let Some(ref mut n) = rc.n {
                    *n += 0.85;
                }
                if let Some(ref mut h) = rc.h {
                    *h -= 0.08;
                }
                rc.ca -= 0.15;
            }
            "PRO" => {
                if let Some(ref mut n) = rc.n {
                    *n += 0.70;
                }
                rc.ca -= 0.10;
            }
            "ALA" => {
                if let Some(ref mut n) = rc.n {
                    *n -= 0.35;
                }
            }
            _ => {}
        }
    }

    // 3. Henderson-Hasselbalch pH titration shifts relative to pH 6.5 reference (Platzer et al. 2014)
    let frac_protonated =
        |pka: f64, cur_ph: f64| -> f64 { 1.0 / (1.0 + 10.0_f64.powf(cur_ph - pka)) };
    let delta_f = |pka: f64| -> f64 { frac_protonated(pka, ph) - frac_protonated(pka, 6.5) };

    match res.as_str() {
        "ASP" => {
            let df = delta_f(3.90);
            rc.ca -= 1.48 * df;
            if let Some(ref mut cb) = rc.cb {
                *cb -= 3.25 * df;
            }
            rc.c -= 1.05 * df;
            rc.ha += 0.14 * df;
            if let Some(ref mut h) = rc.h {
                *h += 0.16 * df;
            }
            if let Some(ref mut n) = rc.n {
                *n -= 0.35 * df;
            }
        }
        "GLU" => {
            let df = delta_f(4.30);
            rc.ca -= 1.38 * df;
            if let Some(ref mut cb) = rc.cb {
                *cb -= 1.65 * df;
            }
            rc.c -= 0.75 * df;
            rc.ha += 0.09 * df;
            if let Some(ref mut h) = rc.h {
                *h += 0.10 * df;
            }
            if let Some(ref mut n) = rc.n {
                *n += 0.25 * df;
            }
        }
        "HIS" => {
            let df = delta_f(6.35);
            rc.ca -= 1.25 * df;
            if let Some(ref mut cb) = rc.cb {
                *cb -= 0.85 * df;
            }
            rc.c -= 0.75 * df;
            rc.ha += 0.12 * df;
            if let Some(ref mut h) = rc.h {
                *h += 0.22 * df;
            }
            if let Some(ref mut n) = rc.n {
                *n += 0.90 * df;
            }
        }
        _ => {}
    }

    // N-terminal and C-terminal chain-end random-coil shifts
    if k == 0 {
        rc.ca -= 0.80;
        rc.c -= 3.50;
        rc.h = None;
        rc.n = None;
    } else if k + 1 == sequence.len() {
        let f_prot_cterm = frac_protonated(3.60, ph);
        rc.ca += 0.65 * (1.0 - f_prot_cterm) - 0.85 * f_prot_cterm;
        rc.c += 2.80 * (1.0 - f_prot_cterm) - 0.40 * f_prot_cterm;
        if let Some(ref mut n) = rc.n {
            *n += 4.20 * (1.0 - f_prot_cterm) - 0.80 * f_prot_cterm;
        }
        if let Some(ref mut h) = rc.h {
            *h -= 0.25 * (1.0 - f_prot_cterm);
        }
    }

    // 4. Linear temperature coefficients relative to 298.15 K (Kjaergaard, Brander & Poulsen 2011)
    let dt = (temp_kelvin - 298.15).clamp(-40.0, 60.0);
    if dt.abs() > 1e-6 {
        if let Some(ref mut h) = rc.h {
            // Amide 1HN random-coil temperature coefficient: ~ -4.6 ppb/K
            *h += -0.0046 * dt;
        }
        if let Some(ref mut n) = rc.n {
            // Amide 15N random-coil temperature coefficient: ~ -17 ppb/K
            *n += -0.0170 * dt;
        }
        rc.ca += -0.0025 * dt;
        if let Some(ref mut cb) = rc.cb {
            *cb += 0.0030 * dt;
        }
        rc.c += -0.0055 * dt;
    }

    rc
}

fn median_f64(vals: &mut [f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len().is_multiple_of(2) {
        Some(0.5 * (vals[mid - 1] + vals[mid]))
    } else {
        Some(vals[mid])
    }
}

/// Automatically detect acidic sample pH (`pH ~ 2.0..3.5`) when `sample_ph` is `None`.
///
#[derive(Clone, Copy, Debug, Default)]
struct UnlabelledShiftTuple {
    ca: Option<f64>,
    cb: Option<f64>,
    c: Option<f64>,
    ha: Option<f64>,
    h: Option<f64>,
    n: Option<f64>,
}

fn cmp_opt_f64(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Extract empirical per-spin-system shift tuples while stripping residue index `k` and sorting
/// canonically, guaranteeing bit-for-bit (`0.0`) invariance under any permutation `k -> pi(k)`
/// of `ref_data.shifts`.
fn extract_unlabelled_tuples(shifts: &BTreeMap<(usize, String), f64>) -> Vec<UnlabelledShiftTuple> {
    let mut grouped: BTreeMap<usize, UnlabelledShiftTuple> = BTreeMap::new();
    for (&(idx, ref atom), &val) in shifts {
        if !val.is_finite() {
            continue;
        }
        let entry = grouped.entry(idx).or_default();
        match atom.as_str() {
            "CA" => entry.ca = Some(val),
            "CB" => entry.cb = Some(val),
            "C" => entry.c = Some(val),
            "HA" => entry.ha = Some(val),
            "HA2" if entry.ha.is_none() => entry.ha = Some(val),
            "H" => entry.h = Some(val),
            "N" => entry.n = Some(val),
            _ => {}
        }
    }
    let mut out: Vec<UnlabelledShiftTuple> = grouped.into_values().collect();
    out.sort_by(|a, b| {
        cmp_opt_f64(a.ca, b.ca)
            .then_with(|| cmp_opt_f64(a.cb, b.cb))
            .then_with(|| cmp_opt_f64(a.c, b.c))
            .then_with(|| cmp_opt_f64(a.ha, b.ha))
            .then_with(|| cmp_opt_f64(a.h, b.h))
            .then_with(|| cmp_opt_f64(a.n, b.n))
    });
    out
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn quantile_match_offset(obs: &mut [f64], rc: &mut [f64], q_lo: f64, q_hi: f64) -> Option<f64> {
    if obs.is_empty() || rc.is_empty() {
        return None;
    }
    obs.sort_by(f64::total_cmp);
    rc.sort_by(f64::total_cmp);
    let mut diffs = Vec::with_capacity(15);
    for m in 0..15 {
        let q = q_lo + (q_hi - q_lo) * ((m as f64) / 14.0);
        diffs.push(quantile_sorted(obs, q) - quantile_sorted(rc, q));
    }
    median_f64(&mut diffs)
}

fn sequence_rc_indices(n: usize) -> std::ops::Range<usize> {
    if n > 2 {
        1..(n - 1)
    } else {
        0..n
    }
}

/// Automatically detect strongly acidic sample conditions (`pH <= 3.5`) strictly from unlabelled
/// `(CA, CB)` shift tuples vs sequence composition when `sample_ph` is not explicitly provided.
///
/// Carboxyl protonation of `Asp` and `Glu` produces simultaneous upfield shifts on BOTH `13CA`
/// (`~ -1.4 ppm`) and `13CB` (`~ -3.3 ppm` on `Asp`, `~ -1.6 ppm` on `Glu`), shifting `(CA, CB)`
/// pairs into protonated `(Asp⁰, Glu⁰)` regions and lowering the unlabelled `13CB` quantile L1
/// discrepancy under an acidic random-coil prior.
fn infer_effective_ph(ref_data: &Reference, cfg: &PropensityConfig) -> f64 {
    if let Some(ph) = cfg.sample_ph {
        return ph.clamp(0.5, 13.5);
    }
    if !cfg.auto_acidic_ph_detection {
        return 6.5;
    }

    let n = ref_data.sequence.len();
    let idx_range = sequence_rc_indices(n);
    let n_asp_glu = idx_range
        .clone()
        .filter(|&j| {
            let up = ref_data.sequence[j].trim().to_ascii_uppercase();
            let next_pro = ref_data
                .sequence
                .get(j + 1)
                .is_some_and(|nx| nx.eq_ignore_ascii_case("PRO"));
            !next_pro && (up == "ASP" || up == "GLU")
        })
        .count();
    if n_asp_glu < 3 {
        return 6.5;
    }

    let tuples = extract_unlabelled_tuples(&ref_data.shifts);
    let mut z_obs: Vec<f64> = tuples
        .iter()
        .filter_map(|t| Some(0.48 * t.ca? + 0.52 * t.cb?))
        .collect();
    let mut z_rc_neutral: Vec<f64> = idx_range
        .clone()
        .filter_map(|j| {
            let rc =
                adjusted_random_coil(&ref_data.sequence, j, false, 6.5, cfg.temperature_kelvin);
            Some(0.48 * rc.ca + 0.52 * rc.cb?)
        })
        .collect();

    // Upper-half quantiles (q in 0.45..=0.88) sit above Asp/Glu Z-values (~47.4 / ~42.8 ppm),
    // isolating any uniform TMS/DSS 13C carrier offset without acidic bias.
    let raw_c13_off =
        quantile_match_offset(&mut z_obs, &mut z_rc_neutral, 0.45, 0.88).unwrap_or(0.0);
    let c13_off = if cfg.auto_offset_correction || raw_c13_off.abs() >= 0.75 {
        raw_c13_off
    } else {
        0.0
    };

    let mut protonated_hits = 0usize;
    let mut cb_obs_corr = Vec::new();
    for t in &tuples {
        if let (Some(ca_v), Some(cb_v)) = (t.ca, t.cb) {
            let ca_c = ca_v - c13_off;
            let cb_c = cb_v - c13_off;
            cb_obs_corr.push(cb_c);
            let is_asp_prot = (50.5..=53.45).contains(&ca_c) && (35.2..=39.2).contains(&cb_c);
            let is_glu_prot = (53.0..=55.85).contains(&ca_c) && (26.4..=29.15).contains(&cb_c);
            if is_asp_prot || is_glu_prot {
                protonated_hits += 1;
            }
        }
    }
    if cb_obs_corr.len() < 5 {
        return 6.5;
    }
    cb_obs_corr.sort_by(f64::total_cmp);

    let mut cb_rc_65: Vec<f64> = idx_range
        .clone()
        .filter_map(|j| {
            adjusted_random_coil(&ref_data.sequence, j, false, 6.5, cfg.temperature_kelvin).cb
        })
        .collect();
    let mut cb_rc_22: Vec<f64> = idx_range
        .filter_map(|j| {
            adjusted_random_coil(&ref_data.sequence, j, false, 2.2, cfg.temperature_kelvin).cb
        })
        .collect();
    cb_rc_65.sort_by(f64::total_cmp);
    cb_rc_22.sort_by(f64::total_cmp);

    let mut l1_65 = 0.0_f64;
    let mut l1_22 = 0.0_f64;
    for m in 0..15 {
        let q = 0.10 + 0.80 * ((m as f64) / 14.0);
        let o_q = quantile_sorted(&cb_obs_corr, q);
        l1_65 += (o_q - quantile_sorted(&cb_rc_65, q)).abs();
        l1_22 += (o_q - quantile_sorted(&cb_rc_22, q)).abs();
    }

    let frac = (protonated_hits as f64) / (n_asp_glu as f64);
    if frac >= 0.55 && l1_22 + 0.15 * 15.0 < l1_65 {
        let f_clamped = frac.clamp(0.65, 0.96);
        let ratio = ((1.0 - f_clamped) / f_clamped).max(0.01);
        return (4.10 + ratio.log10()).clamp(2.0, 4.2);
    }
    6.5
}

/// Estimate uniform chemical-shift referencing offsets `(obs - rc)` strictly from unlabelled
/// empirical shift tuples matched against the sequence random-coil distribution `{rc(j)}`,
/// using secondary-structure-invariant `0.48 * CA + 0.52 * CB` and `C' + 0.72 * CB` combinations
/// (`Zhang, Neal & Wishart, J. Biomol. NMR 2003`). Invariant (`0.0` difference) under arbitrary
/// residue-label permutations `k -> pi(k)` of `ref_data.shifts`.
pub fn estimate_global_rc_offsets(
    ref_data: &Reference,
    cfg: &PropensityConfig,
) -> GlobalShiftOffsets {
    let eff_ph = infer_effective_ph(ref_data, cfg);
    let n = ref_data.sequence.len();
    let tuples = extract_unlabelled_tuples(&ref_data.shifts);
    let idx_range = sequence_rc_indices(n);

    // Infer bulk Cys redox state unlabelled from reduced-CB (~28 ppm) vs oxidized-CB (~41 ppm) counts
    let n_cys = idx_range
        .clone()
        .filter(|&j| ref_data.sequence[j].eq_ignore_ascii_case("CYS"))
        .count();
    let default_cys_ox = if n_cys > 0 {
        let red_like = tuples
            .iter()
            .filter(|t| {
                t.ca.is_some_and(|ca| (55.5..=61.0).contains(&ca))
                    && t.cb.is_some_and(|cb| (24.5..=31.2).contains(&cb))
            })
            .count();
        red_like * 2 < n_cys
    } else {
        false
    };

    let rc_list: Vec<_> = idx_range
        .map(|j| {
            adjusted_random_coil(
                &ref_data.sequence,
                j,
                default_cys_ox,
                eff_ph,
                cfg.temperature_kelvin,
            )
        })
        .collect();

    // 1. Aliphatic 13C uniform offset (shared by CA and CB on common DSS/TMS 13C carrier):
    // 0.48 * CA + 0.52 * CB cancels alpha/beta secondary shifts and moves 1:1 with 13C offset.
    let mut z_obs: Vec<f64> = tuples
        .iter()
        .filter_map(|t| Some(0.48 * t.ca? + 0.52 * t.cb?))
        .collect();
    let mut z_rc: Vec<f64> = rc_list
        .iter()
        .filter_map(|rc| Some(0.48 * rc.ca + 0.52 * rc.cb?))
        .collect();

    let d_aliph = if z_obs.len() >= 3 && !z_rc.is_empty() {
        quantile_match_offset(&mut z_obs, &mut z_rc, 0.15, 0.85).unwrap_or(0.0)
    } else {
        let mut ca_obs: Vec<f64> = tuples.iter().filter_map(|t| t.ca).collect();
        let mut ca_rc: Vec<f64> = rc_list.iter().map(|rc| rc.ca).collect();
        quantile_match_offset(&mut ca_obs, &mut ca_rc, 0.15, 0.85).unwrap_or(0.0)
    };

    // 2. Carbonyl 13C' offset via secondary-structure-cancelling W = C' + 0.72 * CB:
    // E[W_obs - W_rc] = d_c + 0.72 * d_cb => d_c = QMatch(W_obs, W_rc) - 0.72 * d_aliph.
    let mut w_obs: Vec<f64> = tuples
        .iter()
        .filter_map(|t| Some(t.c? + 0.72 * t.cb?))
        .collect();
    let mut w_rc: Vec<f64> = rc_list
        .iter()
        .filter_map(|rc| Some(rc.c + 0.72 * rc.cb?))
        .collect();

    let d_co = if w_obs.len() >= 3 && !w_rc.is_empty() {
        quantile_match_offset(&mut w_obs, &mut w_rc, 0.15, 0.85)
            .map(|dw| dw - 0.72 * d_aliph)
            .unwrap_or(0.0)
    } else {
        let mut c_obs: Vec<f64> = tuples.iter().filter_map(|t| t.c).collect();
        let mut c_rc: Vec<f64> = rc_list.iter().map(|rc| rc.c).collect();
        quantile_match_offset(&mut c_obs, &mut c_rc, 0.15, 0.85).unwrap_or(0.0)
    };

    // 3. Amide 1H, 15N, and aliphatic 1HA offsets from unlabelled interior quantile matching
    let mut h_obs: Vec<f64> = tuples.iter().filter_map(|t| t.h).collect();
    let mut h_rc: Vec<f64> = rc_list.iter().filter_map(|rc| rc.h).collect();
    let d_h = quantile_match_offset(&mut h_obs, &mut h_rc, 0.15, 0.85).unwrap_or(0.0);

    let mut n_obs: Vec<f64> = tuples.iter().filter_map(|t| t.n).collect();
    let mut n_rc: Vec<f64> = rc_list.iter().filter_map(|rc| rc.n).collect();
    let d_n = quantile_match_offset(&mut n_obs, &mut n_rc, 0.15, 0.85).unwrap_or(0.0);

    let mut ha_obs: Vec<f64> = tuples.iter().filter_map(|t| t.ha).collect();
    let mut ha_rc: Vec<f64> = rc_list.iter().map(|rc| rc.ha).collect();
    let d_ha = quantile_match_offset(&mut ha_obs, &mut ha_rc, 0.15, 0.85).unwrap_or(0.0);

    GlobalShiftOffsets {
        d_ca: d_aliph + cfg.manual_offsets.d_ca,
        d_cb: d_aliph + cfg.manual_offsets.d_cb,
        d_c: d_co + cfg.manual_offsets.d_c,
        d_ha: d_ha + cfg.manual_offsets.d_ha,
        d_h: d_h + cfg.manual_offsets.d_h,
        d_n: d_n + cfg.manual_offsets.d_n,
    }
}

/// Normalize a single secondary shift $\Delta\delta_X$ for `Marsh et al. (2006)` `SSP` scoring
/// (`(normalized_ratio, inverse_sd_weight)`).
fn marsh_ssp_term(res: &str, next_is_pro: bool, atom: &str, d_ppm: f64) -> Option<(f64, f64)> {
    let up = res.trim().to_ascii_uppercase();
    let pre_pro_scale = if next_is_pro { 0.65 } else { 1.0 };

    let (alpha_ref, beta_ref, weight): (f64, f64, f64) = match atom {
        "CA" => {
            if up == "GLY" {
                (1.25, -0.85, 0.85 * pre_pro_scale)
            } else if up == "PRO" {
                (2.10, -1.20, 0.80 * pre_pro_scale)
            } else {
                (3.09, -1.48, 1.00 * pre_pro_scale)
            }
        }
        "CB" => {
            if up == "GLY" {
                return None;
            }
            // Marsh et al. (2006) Table 1: CB dispersion in helices is broad relative to mean shift,
            // so helical CB uses an effective reference amplitude preventing small negative noise spikes.
            (-1.55, 2.16, 0.74 * pre_pro_scale)
        }
        "C" => {
            if up == "GLY" {
                (1.25, -0.90, 0.72 * pre_pro_scale)
            } else {
                (2.15, -1.38, 0.84 * pre_pro_scale)
            }
        }
        "HA" => (-0.39, 0.42, 0.68),
        "H" => {
            if up == "PRO" {
                return None;
            }
            (-0.32, 0.45, 0.38)
        }
        "N" => {
            if up == "PRO" {
                return None;
            }
            (-1.95, 2.50, 0.32)
        }
        _ => return None,
    };

    // Determine whether observed shift points toward alpha_ref (positive SSP) or beta_ref (negative SSP)
    let ratio = if alpha_ref > 0.0 {
        if d_ppm >= 0.0 {
            d_ppm / alpha_ref
        } else {
            d_ppm / beta_ref.abs()
        }
    } else if d_ppm <= 0.0 {
        // alpha_ref < 0 (CB, HA, H, N): negative d_ppm is helical (+SSP)
        (-d_ppm) / alpha_ref.abs()
    } else {
        // positive d_ppm is extended (-SSP)
        -(d_ppm / beta_ref.abs())
    };

    Some((ratio.clamp(-1.35, 1.35), weight))
}

/// Asymmetric basin log-likelihood for `Camilloni et al. (2012)` `δ2D` 4-state conformational
/// scoring: quadratic inside the conformational basin, gently saturated on the outer side of
/// extreme rigid α-helical or β-strand shifts so strongly folded elements never revert to coil.
#[inline]
fn basin_atom_ll(d_ppm: f64, mu: f64, sigma: f64, outer_dir: i8) -> f64 {
    let diff = d_ppm - mu;
    let eff_sigma = if (outer_dir > 0 && diff > 0.0) || (outer_dir < 0 && diff < 0.0) {
        sigma * 2.25
    } else {
        sigma
    };
    let z = diff / eff_sigma;
    -0.5 * (z * z).min(14.0)
}

/// Compute local (unsmoothed) 4-state log-evidence `[ll_alpha, ll_beta, ll_ppii, ll_coil]`
/// at residue `k` from its observed secondary shifts.
fn local_d2d_log_evidence(res: &str, sec: &BTreeMap<String, f64>) -> Option<[f64; 4]> {
    if sec.is_empty() {
        return None;
    }
    let is_gly = res.eq_ignore_ascii_case("GLY");
    let is_pro = res.eq_ignore_ascii_case("PRO");

    let mut ll = [0.0_f64; 4]; // [Alpha, Beta, Ppii, Coil]
    let mut used = 0usize;

    if let Some(&dca) = sec.get("CA") {
        let (mu_a, mu_b, mu_p, mu_c, sd) = if is_gly {
            (1.10, -0.64, -0.18, 0.08, 0.80)
        } else if is_pro {
            (1.80, -0.97, -0.25, 0.10, 1.16)
        } else {
            (2.72, -1.09, -0.32, 0.14, 1.25)
        };
        ll[0] += basin_atom_ll(dca, mu_a, sd, 1);
        ll[1] += basin_atom_ll(dca, mu_b, sd, -1);
        ll[2] += basin_atom_ll(dca, mu_p, sd * 0.90, 0);
        ll[3] += basin_atom_ll(dca, mu_c, sd * 0.96, 0);
        used += 1;
    }

    if !is_gly {
        if let Some(&dcb) = sec.get("CB") {
            let sd = 1.48;
            ll[0] += basin_atom_ll(dcb, -1.14, sd, -1);
            ll[1] += basin_atom_ll(dcb, 1.50, sd, 1);
            ll[2] += basin_atom_ll(dcb, 0.02, sd * 0.90, 0);
            ll[3] += basin_atom_ll(dcb, -0.04, sd * 0.96, 0);
            used += 1;
        }
    }

    if let Some(&dc) = sec.get("C") {
        let (mu_a, mu_b, mu_p, mu_c, sd) = if is_gly {
            (1.10, -0.72, -0.25, 0.05, 0.96)
        } else {
            (1.88, -1.02, -0.36, 0.07, 1.25)
        };
        ll[0] += basin_atom_ll(dc, mu_a, sd, 1);
        ll[1] += basin_atom_ll(dc, mu_b, sd, -1);
        ll[2] += basin_atom_ll(dc, mu_p, sd * 0.90, 0);
        ll[3] += basin_atom_ll(dc, mu_c, sd * 0.96, 0);
        used += 1;
    }

    if let Some(&dha) = sec.get("HA") {
        let sd = 0.30;
        ll[0] += basin_atom_ll(dha, -0.30, sd, -1);
        ll[1] += basin_atom_ll(dha, 0.32, sd, 1);
        ll[2] += basin_atom_ll(dha, 0.02, sd * 0.91, 0);
        ll[3] += basin_atom_ll(dha, -0.02, sd * 0.97, 0);
        used += 1;
    }

    if !is_pro {
        if let Some(&dh) = sec.get("H") {
            let sd = 0.52;
            ll[0] += basin_atom_ll(dh, -0.27, sd, -1);
            ll[1] += basin_atom_ll(dh, 0.33, sd, 1);
            ll[2] += basin_atom_ll(dh, -0.03, sd, 0);
            ll[3] += basin_atom_ll(dh, 0.0, sd, 0);
            used += 1;
        }
        if let Some(&dn) = sec.get("N") {
            let sd = 3.10;
            ll[0] += basin_atom_ll(dn, -1.50, sd, -1);
            ll[1] += basin_atom_ll(dn, 1.82, sd, 1);
            ll[2] += basin_atom_ll(dn, -0.20, sd, 0);
            ll[3] += basin_atom_ll(dn, 0.0, sd, 0);
            used += 1;
        }
    }

    if used == 0 {
        None
    } else {
        let norm = 3.6 / (used as f64).max(3.0);
        for v in &mut ll {
            *v *= norm;
        }
        Some(ll)
    }
}

/// Exact simplex softmax normalization guaranteeing `sum == 1.0 +- 1e-15` and `p_s in [0, 1]`.
fn softmax4(logits: [f64; 4]) -> [f64; 4] {
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0_f64; 4];
    let mut sum = 0.0_f64;
    for i in 0..4 {
        let e = (logits[i] - max_l).exp();
        exps[i] = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return [0.25, 0.25, 0.25, 0.25];
    }
    let mut probs = [exps[0] / sum, exps[1] / sum, exps[2] / sum, exps[3] / sum];
    let exact_sum = probs[0] + probs[1] + probs[2];
    probs[3] = (1.0 - exact_sum).max(0.0);
    let renorm = probs[0] + probs[1] + probs[2] + probs[3];
    if (renorm - 1.0).abs() > 1e-12 {
        for p in &mut probs {
            *p /= renorm;
        }
    }
    probs
}

/// Log-sum-exp over 4 state log-weights.
#[inline]
fn logsumexp4(v: [f64; 4]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    m + ((v[0] - m).exp() + (v[1] - m).exp() + (v[2] - m).exp() + (v[3] - m).exp()).ln()
}

/// Exact 4-state cooperative Forward–Backward posterior decoding (`Camilloni et al. 2012`)
/// coupling adjacent backbone conformational states (`Alpha`, `Beta`, `Ppii`, `Coil`) along
/// hydrogen-bonded secondary-structure elements while preserving disordered coil/PPII baselines
/// when secondary shifts fluctuate randomly around zero.
fn forward_backward_d2d(
    sequence: &[String],
    local_ll: &[Option<[f64; 4]>],
) -> Vec<Option<[f64; 4]>> {
    let n = sequence.len();
    if n == 0 {
        return Vec::new();
    }

    // Emission log-potentials with helix/strand nucleation barrier encoded in the cooperative
    // transition matrix below so genuine multi-residue H-bonded elements retain crisp edges while
    // isolated 1-2 residue loop/turn dihedral spikes remain Coil/PPII.
    let mut emit = vec![[0.0_f64; 4]; n];
    for k in 0..n {
        let Some(ll_c) = local_ll[k] else {
            continue;
        };
        let is_pro = sequence[k].eq_ignore_ascii_case("PRO");
        let is_gly = sequence[k].eq_ignore_ascii_case("GLY");
        let next_pro = sequence
            .get(k + 1)
            .is_some_and(|nx| nx.eq_ignore_ascii_case("PRO"));

        let res_bias = if is_pro && next_pro {
            // Schimmel & Flory (1968): Pro preceding Pro is sterically excluded from the alpha_R basin
            [-1.60_f64, -0.85, 0.72, 0.04]
        } else if is_pro {
            [-0.10_f64, -0.78, 0.28, 0.04]
        } else if next_pro {
            [-0.42_f64, -0.08, 0.22, 0.03]
        } else if is_gly {
            [-0.36_f64, -0.28, -0.50, 0.20]
        } else {
            [-0.20_f64, -0.05, -0.22, 0.04]
        };

        for s in 0..4 {
            emit[k][s] = res_bias[s] + 1.07 * ll_c[s];
        }
    }

    // Cooperative log-transition matrix ln T(prev -> curr): high intra-element persistence
    // paired with realistic Coil->Alpha/Beta nucleation cost suppresses isolated loop false
    // positives while permitting direct Alpha<->Beta secondary-structure cap junctions.
    let trans_std: [[f64; 4]; 4] = [
        [0.81_f64.ln(), 0.07_f64.ln(), 0.04_f64.ln(), 0.08_f64.ln()],
        [0.07_f64.ln(), 0.79_f64.ln(), 0.04_f64.ln(), 0.10_f64.ln()],
        [0.04_f64.ln(), 0.09_f64.ln(), 0.47_f64.ln(), 0.40_f64.ln()],
        [0.04_f64.ln(), 0.09_f64.ln(), 0.29_f64.ln(), 0.58_f64.ln()],
    ];
    // Proline lacks an amide HN proton and blocks incoming Alpha->Alpha propagation across all k >= 1,
    // while still permitting Coil/PPII -> N-cap Pro initiation when followed by non-Pro helical residues.
    let trans_into_pro: [[f64; 4]; 4] = [
        [0.10_f64.ln(), 0.04_f64.ln(), 0.50_f64.ln(), 0.36_f64.ln()],
        [0.06_f64.ln(), 0.24_f64.ln(), 0.37_f64.ln(), 0.33_f64.ln()],
        [0.18_f64.ln(), 0.05_f64.ln(), 0.44_f64.ln(), 0.33_f64.ln()],
        [0.18_f64.ln(), 0.05_f64.ln(), 0.42_f64.ln(), 0.35_f64.ln()],
    ];
    let init_log = [0.10_f64.ln(), 0.14_f64.ln(), 0.28_f64.ln(), 0.48_f64.ln()];

    let mut fwd = vec![[0.0_f64; 4]; n];
    for s in 0..4 {
        fwd[0][s] = init_log[s] + emit[0][s];
    }
    for k in 1..n {
        let tmat = if sequence[k].eq_ignore_ascii_case("PRO") {
            &trans_into_pro
        } else {
            &trans_std
        };
        for s in 0..4 {
            let incoming = [
                fwd[k - 1][0] + tmat[0][s],
                fwd[k - 1][1] + tmat[1][s],
                fwd[k - 1][2] + tmat[2][s],
                fwd[k - 1][3] + tmat[3][s],
            ];
            fwd[k][s] = emit[k][s] + logsumexp4(incoming);
        }
    }

    let mut bwd = vec![[0.0_f64; 4]; n];
    for k in (0..n.saturating_sub(1)).rev() {
        let tmat = if sequence[k + 1].eq_ignore_ascii_case("PRO") {
            &trans_into_pro
        } else {
            &trans_std
        };
        for s_prev in 0..4 {
            let outgoing = [
                tmat[s_prev][0] + emit[k + 1][0] + bwd[k + 1][0],
                tmat[s_prev][1] + emit[k + 1][1] + bwd[k + 1][1],
                tmat[s_prev][2] + emit[k + 1][2] + bwd[k + 1][2],
                tmat[s_prev][3] + emit[k + 1][3] + bwd[k + 1][3],
            ];
            bwd[k][s_prev] = logsumexp4(outgoing);
        }
    }

    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        if local_ll[k].is_none() {
            out.push(None);
        } else {
            let logits = [
                fwd[k][0] + bwd[k][0],
                fwd[k][1] + bwd[k][1],
                fwd[k][2] + bwd[k][2],
                fwd[k][3] + bwd[k][3],
            ];
            out.push(Some(softmax4(logits)));
        }
    }
    out
}

/// Compute per-residue `SSP` scores (`Marsh et al. 2006`) and normalized 4-state `δ2D`
/// conformational populations (`Camilloni et al. 2012`: `p_alpha`, `p_beta`, `p_ppii`, `p_coil`).
pub fn compute_idp_propensity(ref_data: &Reference, cfg: &PropensityConfig) -> PropensityReport {
    let n = ref_data.sequence.len();
    let effective_ph = infer_effective_ph(ref_data, cfg);
    let applied_offsets = if cfg.auto_offset_correction {
        estimate_global_rc_offsets(ref_data, cfg)
    } else {
        cfg.manual_offsets
    };

    // Pass 1: Per-residue secondary shifts and local SSP / d2D evidence
    let mut sec_per_res: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); n];
    let mut local_ssp_num = vec![0.0_f64; n];
    let mut local_ssp_den = vec![0.0_f64; n];
    let mut local_ll: Vec<Option<[f64; 4]>> = vec![None; n];

    for k in 0..n {
        let res = &ref_data.sequence[k];
        let next_is_pro = ref_data
            .sequence
            .get(k + 1)
            .is_some_and(|nx| nx.eq_ignore_ascii_case("PRO"));
        let cys_ox = ref_data
            .shifts
            .get(&(k, "CB".to_string()))
            .is_some_and(|&v| res.eq_ignore_ascii_case("CYS") && v - applied_offsets.d_cb > 35.0);
        let rc = adjusted_random_coil(
            &ref_data.sequence,
            k,
            cys_ox,
            effective_ph,
            cfg.temperature_kelvin,
        );

        let mut sec = BTreeMap::new();
        if let Some(&v) = ref_data.shifts.get(&(k, "CA".to_string())) {
            sec.insert("CA".to_string(), v - (rc.ca + applied_offsets.d_ca));
        }
        if let (Some(&v), Some(rc_cb)) = (ref_data.shifts.get(&(k, "CB".to_string())), rc.cb) {
            sec.insert("CB".to_string(), v - (rc_cb + applied_offsets.d_cb));
        }
        if let Some(&v) = ref_data.shifts.get(&(k, "C".to_string())) {
            sec.insert("C".to_string(), v - (rc.c + applied_offsets.d_c));
        }
        let ha_obs = ref_data
            .shifts
            .get(&(k, "HA".to_string()))
            .or_else(|| ref_data.shifts.get(&(k, "HA2".to_string())));
        if let Some(&v) = ha_obs {
            sec.insert("HA".to_string(), v - (rc.ha + applied_offsets.d_ha));
        }
        if let (Some(&v), Some(rc_h)) = (ref_data.shifts.get(&(k, "H".to_string())), rc.h) {
            sec.insert("H".to_string(), v - (rc_h + applied_offsets.d_h));
        }
        if let (Some(&v), Some(rc_n)) = (ref_data.shifts.get(&(k, "N".to_string())), rc.n) {
            sec.insert("N".to_string(), v - (rc_n + applied_offsets.d_n));
        }

        for (atom, &d_ppm) in &sec {
            if let Some((ratio, w)) = marsh_ssp_term(res, next_is_pro, atom, d_ppm) {
                local_ssp_num[k] += w * ratio;
                local_ssp_den[k] += w;
            }
        }
        local_ll[k] = local_d2d_log_evidence(res, &sec);
        sec_per_res[k] = sec;
    }

    // Pass 2: Sliding-window Marsh (2006) SSP and cooperative Camilloni (2012) d2D posteriors
    let d2d_posteriors = forward_backward_d2d(&ref_data.sequence, &local_ll);
    let half_w = (cfg.ssp_window.max(1) / 2) as isize;
    let mut residues = Vec::with_capacity(n);
    let mut sum_pa = 0.0_f64;
    let mut sum_pb = 0.0_f64;
    let mut sum_pp = 0.0_f64;
    let mut sum_pc = 0.0_f64;
    let mut sum_abs_ssp = 0.0_f64;
    let mut n_ssp = 0usize;
    let mut n_with_data = 0usize;
    let mut max_rigid_run = 0usize;
    let mut cur_rigid_run = 0usize;

    for k in 0..n {
        let res = ref_data.sequence[k].clone();
        let sid = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let n_obs = sec_per_res[k].len();
        let has_data = d2d_posteriors[k].is_some();

        // Marsh et al. (2006) 5-residue window weighted average
        let mut win_num = 0.0_f64;
        let mut win_den = 0.0_f64;
        for dj in -half_w..=half_w {
            let j_isize = (k as isize) + dj;
            if j_isize < 0 || j_isize >= (n as isize) {
                continue;
            }
            let j = j_isize as usize;
            if local_ssp_den[j] <= 0.0 {
                continue;
            }
            let dist = dj.unsigned_abs();
            let pos_w = match dist {
                0 => 1.00,
                1 => 0.82,
                2 => 0.52,
                _ => 0.30,
            };
            win_num += pos_w * local_ssp_num[j];
            win_den += pos_w * local_ssp_den[j];
        }
        let ssp_score = if has_data && win_den > 0.0 {
            let s = (win_num / win_den).clamp(-1.0, 1.0);
            sum_abs_ssp += s.abs();
            n_ssp += 1;
            Some(s)
        } else {
            None
        };

        let (probs, dominant_state) = if let Some(p) = d2d_posteriors[k] {
            n_with_data += 1;
            let states = [
                ConformationalState::Alpha,
                ConformationalState::Beta,
                ConformationalState::Ppii,
                ConformationalState::Coil,
            ];
            let mut best_idx = 3usize;
            for s in 0..4 {
                if p[s] > p[best_idx] {
                    best_idx = s;
                }
            }
            sum_pa += p[0];
            sum_pb += p[1];
            sum_pp += p[2];
            sum_pc += p[3];

            if matches!(
                states[best_idx],
                ConformationalState::Alpha | ConformationalState::Beta
            ) && (p[0] + p[1]) >= 0.55
            {
                cur_rigid_run += 1;
                max_rigid_run = max_rigid_run.max(cur_rigid_run);
            } else {
                cur_rigid_run = 0;
            }

            (p, states[best_idx])
        } else {
            cur_rigid_run = 0;
            let p = match cfg.unassigned_prior {
                UnassignedPriorMode::CoilBaseline => [0.05, 0.05, 0.25, 0.65],
                UnassignedPriorMode::Uniform => [0.25, 0.25, 0.25, 0.25],
            };
            (p, ConformationalState::Coil)
        };

        residues.push(ResiduePropensity {
            seq_index: k,
            seq_id: sid,
            residue: res,
            has_data,
            n_observed_nuclei: n_obs,
            ssp_score,
            p_alpha: probs[0],
            p_beta: probs[1],
            p_ppii: probs[2],
            p_coil: probs[3],
            dominant_state,
            secondary_shifts: sec_per_res[k].clone(),
        });
    }

    let denom = (n_with_data.max(1)) as f64;
    let (mean_p_alpha, mean_p_beta, mean_p_ppii, mean_p_coil) = if n_with_data > 0 {
        (
            sum_pa / denom,
            sum_pb / denom,
            sum_pp / denom,
            sum_pc / denom,
        )
    } else {
        match cfg.unassigned_prior {
            UnassignedPriorMode::CoilBaseline => (0.05, 0.05, 0.25, 0.65),
            UnassignedPriorMode::Uniform => (0.25, 0.25, 0.25, 0.25),
        }
    };
    let mean_abs_ssp = if n_ssp > 0 {
        sum_abs_ssp / (n_ssp as f64)
    } else {
        0.0
    };
    let is_predominantly_disordered = (mean_p_coil + mean_p_ppii) >= 0.70 && max_rigid_run < 4;

    PropensityReport {
        residues,
        effective_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        applied_offsets,
        n_residues_with_data: n_with_data,
        mean_p_alpha,
        mean_p_beta,
        mean_p_ppii,
        mean_p_coil,
        mean_abs_ssp,
        is_predominantly_disordered,
    }
}

/// Format a [`PropensityReport`] as a deterministic tab-separated values (`TSV`) table.
pub fn format_propensity_tsv(report: &PropensityReport) -> String {
    let mut out = String::with_capacity(report.residues.len() * 96 + 256);
    out.push_str(&format!(
        "# nmr-core SSP / d2D Conformational Propensity Report (T={:.2} K, pH={:.2})\n",
        report.temperature_kelvin, report.effective_ph
    ));
    out.push_str("seq_id\tresidue\thas_data\tn_nuclei\tssp\tp_alpha\tp_beta\tp_ppii\tp_coil\tstate\tdCA\tdCB\tdC\tdHA\tdH\tdN\n");
    for r in &report.residues {
        let ssp_s = r
            .ssp_score
            .map(|v| format!("{v:+.4}"))
            .unwrap_or_else(|| "NA".to_string());
        let fmt_sec = |atom: &str| -> String {
            r.secondary_shifts
                .get(atom)
                .map(|v| format!("{v:+.3}"))
                .unwrap_or_else(|| "NA".to_string())
        };
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.seq_id,
            r.residue,
            r.has_data,
            r.n_observed_nuclei,
            ssp_s,
            r.p_alpha,
            r.p_beta,
            r.p_ppii,
            r.p_coil,
            r.dominant_state.code(),
            fmt_sec("CA"),
            fmt_sec("CB"),
            fmt_sec("C"),
            fmt_sec("HA"),
            fmt_sec("H"),
            fmt_sec("N"),
        ));
    }
    out
}

/// Format a [`PropensityReport`] as pretty-printed JSON.
pub fn format_propensity_json(report: &PropensityReport) -> Result<String, String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| format!("failed to serialize propensity JSON: {e}"))
}
