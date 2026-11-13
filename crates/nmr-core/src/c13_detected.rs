//! Protonless `13C`-direct-detected NMR experiment models, field-strength-invariant narrow-dispersion
//! `2D CON` / `2D CACO` / `3D` strip peak pickers, and volumetric IDP cube simulator with `1H^N`
//! solvent-exchange broadening comparison (`Phase 11`, `T11.1`).
//!
//! Provenance: Clean-room implementation derived strictly from peer-reviewed biomolecular NMR literature:
//! - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn.
//!   Reson. Spectrosc.* 2006, 48, 25–45 — protonless `13C` direct-detection NMR spectroscopy).
//! - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C. & Pierattelli, R. (*J. Am.
//!   Chem. Soc.* 2006, 128, 3918–3919 — protonless `2D CON` and `2D CACO` IPAP/virtual-decoupled
//!   experiments for intrinsically disordered proteins).
//! - Bermel, W., Bertini, I., Csizmok, V., Felli, I. C., Pierattelli, R. & Tompa, P. (*J. Magn.
//!   Reson.* 2009, 198, 275–281) and Bermel, W., Bertini, I., Felli, I. C. & Pierattelli, R.
//!   (*J. Am. Chem. Soc.* 2009, 131, 15339–15345 — `3D CANCO`, `(H)CBCACON` / `CBCACO`,
//!   `(H)CCCON` / `CCC(CO)N`, and `NCACO` experiments for crowded proline-rich IDPs).
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP`
//!   random-coil chemical shifts) and Kjaergaard, M. & Poulsen, F. M. (*Prog. Nucl. Magn. Reson.
//!   Spectrosc.* 2011, 59, 43–57 — sequence and temperature/pH behavior in disordered proteins).
//! - Bai, Y., Milne, J. S., Mayne, L. & Englander, S. W. (*Proteins* 1993, 17, 75–86 — primary
//!   sequence and pH/temperature dependence of intrinsic amide `1H^N` solvent-exchange rates).
//! - Markley, J. L. et al. (*J. Mol. Biol.* 1998, 280, 933–952 — IUPAC heteronuclear $\Xi$ frequency
//!   ratios for `13C` and `15N` relative to `1H` DSS).

use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::{LN_2, PI};

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::picker::{estimate_noise_mad, pick_hsqc_2d, PickerConfig};
use crate::rng::Rng;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

/// IUPAC-IUBMB heteronuclear $\Xi$ ratio $\nu_0(^{13}\text{C}) / \nu_0(^1\text{H})$ (DSS scale).
pub const XI_RATIO_C13_OVER_H1: f64 = 0.251_450_20;

/// IUPAC-IUBMB heteronuclear $\Xi$ ratio $\nu_0(^{15}\text{N}) / \nu_0(^1\text{H})$ (liquid $\text{NH}_3$ / DSS scale).
pub const XI_RATIO_N15_OVER_H1: f64 = 0.101_329_12;

/// Convert `1H` spectrometer frequency (`500–950 MHz`) to `13C` observation frequency in `MHz`.
#[inline]
pub fn c13_obs_mhz(proton_mhz: f64) -> f64 {
    let base = if proton_mhz.is_finite() && proton_mhz > 50.0 {
        proton_mhz
    } else {
        600.0
    };
    base * XI_RATIO_C13_OVER_H1
}

/// Convert `1H` spectrometer frequency (`500–950 MHz`) to `15N` observation frequency in `MHz`.
#[inline]
pub fn n15_obs_mhz(proton_mhz: f64) -> f64 {
    let base = if proton_mhz.is_finite() && proton_mhz > 50.0 {
        proton_mhz
    } else {
        600.0
    };
    base * XI_RATIO_N15_OVER_H1
}

/// Convert a physical frequency interval in `Hz` to `ppm` dynamically using `obs_mhz` (`Hz / MHz`).
/// Guards against non-positive or non-finite `obs_mhz` by falling back to `fallback_obs_mhz`.
#[inline]
pub fn hz_to_ppm(hz: f64, obs_mhz: f64, fallback_obs_mhz: f64) -> f64 {
    let denom = if obs_mhz.is_finite() && obs_mhz > 1.0 {
        obs_mhz
    } else {
        fallback_obs_mhz.max(1.0)
    };
    hz / denom
}

/// Supported protonless / `13C`-direct-detected multidimensional IDP experiments
/// (Bermel et al., *Prog. NMR Spectrosc.* 2006; *JACS* 2006; *JACS* 2009).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum C13DetectedExp {
    /// `2D CON`: correlates `13C'(i-1)` (`168–180 ppm`, direct IPAP) with `15N(i)` (`105–142 ppm`,
    /// including `Pro` tertiary imide `132–142 ppm`).
    Con2D,
    /// `2D CACO`: correlates intraresidue `13CA(i)` (`41–68 ppm`) with `13C'(i)` (`168–180 ppm`,
    /// direct IPAP), with optional constant-time `^1J_{CACB}` `Gly` sign inversion.
    Caco2D,
    /// `3D CANCO`: bidirectional `13CA -> 15N -> 13C'` and `13CA -> 13C' -> 15N` transfer yielding
    /// both `13CA(i)` and `13CA(i-1)` at each `CON(i)` `(13C'(i-1), 15N(i))` anchor.
    Canco3D,
    /// `3D CBCACON` (and `CBCACO` projection): unidirectional aliphatic `13CA(i-1)/13CB(i-1) ->
    /// 13C'(i-1) -> 15N(i)` transfer isolating `(i-1)` `CA` and `CB` at `(13C'(i-1), 15N(i))`.
    Cbcaco3D,
    /// `3D CCC(CO)N`: isotropic `13C–13C` TOCSY relay across the entire aliphatic sidechain of
    /// residue `(i-1)` (`CA, CB, CG*, CD*, CE*`, including `Pro` `CG ~ 27.3` & `CD ~ 50.4 ppm`)
    /// into `(13C'(i-1), 15N(i))`.
    Cccon3D,
    /// `3D NCACO`: intraresidue `15N(i) -> 13CA(i) -> 13C'(i)` correlation linking `N(i)` directly
    /// to `(CA(i), C'(i))` to bridge `CON(i)` with `CON(i+1)`.
    Ncaco3D,
}

impl C13DetectedExp {
    pub const ALL: [Self; 6] = [
        Self::Con2D,
        Self::Caco2D,
        Self::Canco3D,
        Self::Cbcaco3D,
        Self::Cccon3D,
        Self::Ncaco3D,
    ];

    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Con2D => "CON",
            Self::Caco2D => "CACO",
            Self::Canco3D => "CANCO",
            Self::Cbcaco3D => "CBCACON",
            Self::Cccon3D => "CCCON",
            Self::Ncaco3D => "NCACO",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        let norm: String = name
            .trim()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect();
        match norm.as_str() {
            "CON" | "CON2D" | "2DCON" | "HCON" => Some(Self::Con2D),
            "CACO" | "CACO2D" | "2DCACO" | "HCACO" => Some(Self::Caco2D),
            "CANCO" | "CANCO3D" | "3DCANCO" | "HCANCO" => Some(Self::Canco3D),
            "CBCACON" | "CBCACO" | "CBCACO3D" | "HCBCACON" => Some(Self::Cbcaco3D),
            "CCCON" | "CCCCON" | "CCCON3D" | "HCCCON" => Some(Self::Cccon3D),
            "NCACO" | "NCACO3D" | "3DNCACO" => Some(Self::Ncaco3D),
            _ => None,
        }
    }

    #[inline]
    pub fn is_2d(&self) -> bool {
        matches!(self, Self::Con2D | Self::Caco2D)
    }

    #[inline]
    pub fn is_3d(&self) -> bool {
        !self.is_2d()
    }

    /// Direct acquisition dimension (`F2` in 2D, `F3` in 3D): always IPAP/S3E virtual-decoupled `13C'`.
    #[inline]
    pub fn direct_nucleus(&self) -> &'static str {
        "13C'"
    }

    /// Indirect dimension labels (`[F1]` for 2D, `[F2_mid, F1_ind]` for 3D).
    #[inline]
    pub fn indirect_nuclei(&self) -> &'static [&'static str] {
        match self {
            Self::Con2D => &["15N"],
            Self::Caco2D => &["13CA"],
            Self::Canco3D => &["15N", "13CA"],
            Self::Cbcaco3D => &["15N", "13CAB"],
            Self::Cccon3D => &["15N", "13C_ALIPH"],
            Self::Ncaco3D => &["15N", "13CA"],
        }
    }

    /// Exact coherence-transfer pathway description (Bermel et al. 2006, 2009).
    pub fn pathway_description(&self) -> &'static str {
        match self {
            Self::Con2D => "13C'(i-1) -> 15N(i)(t1) -> 13C'(i-1)(t2, IPAP)",
            Self::Caco2D => "13CA(i)(t1, CT) -> 13C'(i)(t2, IPAP)",
            Self::Canco3D => {
                "13CA(i,i-1)(t1) -> 15N(i)(t2) -> 13C'(i-1)(t3, IPAP) & 13CA(i-1)(t1) -> 13C'(i-1) -> 15N(i)(t2) -> 13C'(i-1)(t3)"
            }
            Self::Cbcaco3D => {
                "13CA/13CB(i-1)(t1) -> 13CA(i-1) -> 13C'(i-1) -> 15N(i)(t2) -> 13C'(i-1)(t3, IPAP)"
            }
            Self::Cccon3D => {
                "13C_aliph(i-1)(t1) -[FLOPSY-16]-> 13CA(i-1) -> 13C'(i-1) -> 15N(i)(t2) -> 13C'(i-1)(t3, IPAP)"
            }
            Self::Ncaco3D => "15N(i)(t1) -> 13CA(i)(t2) -> 13C'(i)(t3, IPAP)",
        }
    }

    /// Physical spectral window bounds `(c_prime_ppm, n_or_none_ppm, c_aliph_or_none_ppm)`.
    #[allow(clippy::type_complexity)]
    pub fn spectral_bounds_ppm(&self) -> ((f64, f64), Option<(f64, f64)>, Option<(f64, f64)>) {
        match self {
            Self::Con2D => ((166.0, 183.0), Some((103.0, 144.5)), None),
            Self::Caco2D => ((166.0, 183.0), None, Some((40.0, 69.0))),
            Self::Canco3D => ((166.0, 183.0), Some((103.0, 144.5)), Some((40.0, 69.0))),
            Self::Cbcaco3D => ((166.0, 183.0), Some((103.0, 144.5)), Some((13.0, 73.5))),
            Self::Cccon3D => ((166.0, 183.0), Some((103.0, 144.5)), Some((9.5, 73.5))),
            Self::Ncaco3D => ((166.0, 183.0), Some((103.0, 144.5)), Some((40.0, 69.0))),
        }
    }
}

/// Picked peptide-bond root in a `2D CON` spectrum correlating `13C'(i-1)` with `15N(i)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Con2DPeak {
    /// Observed `13C'(i-1)` chemical shift in ppm (`~168–180 ppm`).
    pub c_prev_ppm: f64,
    /// Observed `15N(i)` chemical shift in ppm (`~105–131 ppm` amide, `~132–142 ppm` `Pro` imino).
    pub n_curr_ppm: f64,
    pub height: f64,
    pub volume: f64,
    /// Unlabelled spectral region flag indicating `n_curr_ppm` falls in the downfield `Pro` imino window (`>= 131.5 ppm`).
    pub is_proline_imino_region: bool,
    /// True if resolved from an overlapping multi-peak IDP island via 2D Gaussian residual peeling.
    pub deconvolved_from_cluster: bool,
}

/// Unlabelled spectral classification of a `2D CACO` `(CA(i), C'(i))` peak based strictly on
/// observed chemical-shift window and constant-time `^1J_{CACB}` phase sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacoPeakKind {
    /// Opposite sign relative to `CB`-bearing majority (`cos(pi * J_CACB * T_CA)` constant-time `Gly` signature).
    GlyOppositePhase,
    /// Same-sign spectrum with `CA` inside the upfield `Gly` region (`41.0..=47.8 ppm`).
    GlyRegion,
    /// Downfield `Ser` / `Thr` / `Pro` / `Val` / `Ile` `CA` region (`>= 58.0 ppm`).
    SerThrProDownfield,
    /// General mid-field aliphatic `CA` region (`47.8..58.0 ppm`).
    GeneralAliphatic,
}

/// Picked intraresidue `(13CA(i), 13C'(i))` correlation from a `2D CACO` spectrum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Caco2DPeak {
    pub c_curr_ppm: f64,
    pub ca_curr_ppm: f64,
    pub sign: i32,
    pub height: f64,
    pub volume: f64,
    pub kind: CacoPeakKind,
}

/// Picked 3D `13C`-detected peak from `CANCO`, `CBCACON`, `CCC(CO)N`, or `NCACO`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct C13Strip3DPeak {
    pub exp: C13DetectedExp,
    pub c_prime_ppm: f64,
    pub n_ppm: f64,
    /// Indirect aliphatic carbon frequency (`CA`, `CB`, or sidechain `CG/CD/CE` in ppm).
    pub c_indirect_ppm: f64,
    pub sign: i32,
    pub height: f64,
    pub volume: f64,
    /// Index of the anchoring `Con2DPeak` when picked from a `CON(i)` strip (`None` for unanchored / `NCACO` global).
    pub anchor_root_idx: Option<usize>,
}

/// Complete unlabelled peak collection picked from a `13C`-detected IDP experiment suite.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct C13PickedDataset {
    pub con_roots: Vec<Con2DPeak>,
    pub caco_peaks: Vec<Caco2DPeak>,
    pub strip_peaks: Vec<C13Strip3DPeak>,
}

/// Field-strength-invariant configuration for `13C`-detected 2D/3D IDP peak picking.
///
/// Every physical linewidth, IPAP decoupling residual cutoff, and strip matching radius is
/// expressed in **`Hz`** and converted to `ppm` dynamically via each axis's `obs_mhz` (`500–950 MHz`).
#[derive(Clone, Debug)]
pub struct C13PickerConfig {
    /// Primary 2D `CON` / `CACO` signal-to-noise threshold (`k * MAD`, default `5.0`).
    pub snr_2d: f32,
    /// Shoulder residual signal-to-noise threshold during crowded-island 2D deconvolution (`default = 5.2`).
    pub shoulder_snr_2d: f32,
    /// Minimum relative amplitude (`I_shoulder / I_island_max`) for splitting a crowded 2D island (`default = 0.14`).
    pub shoulder_min_rel_amp: f64,
    /// Maximum components extracted per connected 2D crowded island (`default = 16`).
    pub max_cluster_components: usize,
    /// Minimum resolved separation in `Hz` along `13C'` (`default = 3.0 Hz`, i.e. `0.024 ppm` at `500 MHz`, `0.013 ppm` at `950 MHz`).
    pub min_sep_c_prime_hz: f64,
    /// Minimum resolved separation in `Hz` along `15N` (`default = 3.5 Hz`, i.e. `0.069 ppm` at `500 MHz`, `0.036 ppm` at `950 MHz`).
    pub min_sep_n15_hz: f64,
    /// Minimum resolved separation in `Hz` along `13CA`/`13CB` (`default = 4.2 Hz`).
    pub min_sep_c_aliph_hz: f64,
    /// Nominal IPAP virtual-decoupling full `^1J_{C'CA}` residual satellite window `(lo_hz, hi_hz)`
    /// (`default = (45.0, 62.0) Hz`, centered on `^1J_{C'CA} ≈ 53.5 Hz`).
    pub ipap_j_window_hz: (f64, f64),
    /// Maximum relative height of symmetric IPAP `± ^1J_{C'CA}` residual decoupling satellites (`default = 0.105`).
    pub ipap_max_residual_frac: f64,
    /// Allow opposite-sign peaks in `2D CACO` (for constant-time `^1J_{CACB}` `Gly` inversion, `default = true`).
    pub caco_allow_negative: bool,
    /// 3D `CON`-anchored strip signal-to-noise threshold (`default = 4.5`).
    pub snr_3d: f32,
    /// Allow negative peaks in `CBCACON` (`CA` `+1`, `CB` `-1`, `default = true`).
    pub cbcacon_allow_negative: bool,
    /// `CON` strip extraction half-width along `13C'` in `Hz` (`default = 5.5 Hz`).
    pub strip_tol_c_prime_hz: f64,
    /// `CON` strip extraction half-width along `15N` in `Hz` (`default = 6.5 Hz`).
    pub strip_tol_n15_hz: f64,
    /// Automatically estimate and remove uniform inter-experiment `(ΔC', ΔN)` carrier drift on 3D cubes (`default = true`).
    pub auto_register_3d_cubes: bool,
}

impl Default for C13PickerConfig {
    fn default() -> Self {
        Self {
            snr_2d: 5.0,
            shoulder_snr_2d: 5.2,
            shoulder_min_rel_amp: 0.14,
            max_cluster_components: 16,
            min_sep_c_prime_hz: 3.0,
            min_sep_n15_hz: 3.5,
            min_sep_c_aliph_hz: 4.2,
            ipap_j_window_hz: (45.0, 62.0),
            ipap_max_residual_frac: 0.105,
            caco_allow_negative: true,
            snr_3d: 4.5,
            cbcacon_allow_negative: true,
            strip_tol_c_prime_hz: 5.5,
            strip_tol_n15_hz: 6.5,
            auto_register_3d_cubes: true,
        }
    }
}

#[inline]
fn subvoxel_1d(ym1: f64, y0: f64, yp1: f64) -> f64 {
    let a = ym1.abs();
    let b = y0.abs();
    let c = yp1.abs();
    if a > 0.0
        && b > a
        && b > c
        && c > 0.0
        && ym1.signum() == y0.signum()
        && yp1.signum() == y0.signum()
    {
        let la = a.ln();
        let lb = b.ln();
        let lc = c.ln();
        let denom = la - 2.0 * lb + lc;
        if denom.abs() > 1e-12 {
            return (0.5 * (la - lc) / denom).clamp(-0.5, 0.5);
        }
    }
    let denom = a - 2.0 * b + c;
    if denom.abs() > 1e-12 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

#[inline]
fn is_c_prime_axis(ax: &AxisPpm) -> bool {
    let hi = ax.max_ppm();
    let lo = ax.min_ppm();
    hi > 165.0 && lo > 150.0 && hi < 195.0
}

#[inline]
fn is_n15_axis(ax: &AxisPpm) -> bool {
    ax.label.to_ascii_uppercase().contains('N')
        || (ax.max_ppm() > 102.0 && ax.max_ppm() < 155.0 && ax.min_ppm() > 90.0)
}

/// Estimate empirical single-peak Gaussian standard deviations `(sigma_x_pts, sigma_y_pts)` directly
/// from isolated high-SNR 2D maxima, clamped to physical IDP `Hz` bounds converted via `obs_mhz`.
#[allow(clippy::too_many_arguments)]
fn estimate_2d_peak_sigmas_pts(
    spec: &Spectrum2D,
    maxima: &[(usize, usize, f64)],
    min_fwhm_x_hz: f64,
    max_fwhm_x_hz: f64,
    min_fwhm_y_hz: f64,
    max_fwhm_y_hz: f64,
    fallback_mhz_x: f64,
    fallback_mhz_y: f64,
) -> (f64, f64) {
    let fwhm_to_sigma = 1.0 / (8.0 * LN_2).sqrt();
    let step_x = spec.axis_x.step_ppm.abs().max(1e-6);
    let step_y = spec.axis_y.step_ppm.abs().max(1e-6);

    let lo_sx = (hz_to_ppm(min_fwhm_x_hz, spec.axis_x.obs_mhz, fallback_mhz_x) * fwhm_to_sigma
        / step_x)
        .max(0.55);
    let hi_sx = (hz_to_ppm(max_fwhm_x_hz, spec.axis_x.obs_mhz, fallback_mhz_x) * fwhm_to_sigma
        / step_x)
        .max(lo_sx + 0.25);
    let lo_sy = (hz_to_ppm(min_fwhm_y_hz, spec.axis_y.obs_mhz, fallback_mhz_y) * fwhm_to_sigma
        / step_y)
        .max(0.55);
    let hi_sy = (hz_to_ppm(max_fwhm_y_hz, spec.axis_y.obs_mhz, fallback_mhz_y) * fwhm_to_sigma
        / step_y)
        .max(lo_sy + 0.25);

    let nx = spec.nx();
    let ny = spec.ny();
    let mut sx_samples = Vec::new();
    let mut sy_samples = Vec::new();

    for &(ix, iy, amp) in maxima {
        if ix < 2 || ix + 2 >= nx || iy < 2 || iy + 2 >= ny || amp <= 0.0 {
            continue;
        }
        // Use clean isolated peaks without immediate competing maxima within 5 grid points
        let isolated = maxima.iter().all(|&(jx, jy, _)| {
            (jx == ix && jy == iy) || ix.abs_diff(jx) > 5 || iy.abs_diff(jy) > 5
        });
        if !isolated {
            continue;
        }
        let sgn = amp.signum();
        let v0 = spec.get(ix, iy) as f64 * sgn;
        let vxm = spec.get(ix - 1, iy) as f64 * sgn;
        let vxp = spec.get(ix + 1, iy) as f64 * sgn;
        if vxm > 0.2 * v0 && vxp > 0.2 * v0 && vxm < v0 && vxp < v0 {
            let d2 = 2.0 * v0.ln() - vxm.ln() - vxp.ln();
            if d2 > 1e-4 {
                let s = (1.0 / d2).sqrt();
                if (lo_sx * 0.7..=hi_sx * 1.3).contains(&s) {
                    sx_samples.push(s.clamp(lo_sx, hi_sx));
                }
            }
        }
        let vym = spec.get(ix, iy - 1) as f64 * sgn;
        let vyp = spec.get(ix, iy + 1) as f64 * sgn;
        if vym > 0.2 * v0 && vyp > 0.2 * v0 && vym < v0 && vyp < v0 {
            let d2 = 2.0 * v0.ln() - vym.ln() - vyp.ln();
            if d2 > 1e-4 {
                let s = (1.0 / d2).sqrt();
                if (lo_sy * 0.7..=hi_sy * 1.3).contains(&s) {
                    sy_samples.push(s.clamp(lo_sy, hi_sy));
                }
            }
        }
    }

    let med = |mut v: Vec<f64>, fallback: f64| -> f64 {
        if v.is_empty() {
            fallback
        } else {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        }
    };

    let default_sx = (hz_to_ppm(5.2, spec.axis_x.obs_mhz, fallback_mhz_x) * fwhm_to_sigma / step_x)
        .clamp(lo_sx, hi_sx);
    let default_sy = (hz_to_ppm(4.5, spec.axis_y.obs_mhz, fallback_mhz_y) * fwhm_to_sigma / step_y)
        .clamp(lo_sy, hi_sy);
    (med(sx_samples, default_sx), med(sy_samples, default_sy))
}

#[derive(Clone, Debug)]
struct Raw2DComponent {
    x_idx: f64,
    y_idx: f64,
    amp: f64,
    sign: i32,
    deconvolved: bool,
}

/// Core 2D narrow-dispersion local-maxima + connected-island Gaussian residual peeling picker.
/// Operates strictly on spectral intensities and axis `obs_mhz` calibration (zero sequence/priors).
#[allow(clippy::too_many_arguments)]
fn pick_narrow_2d_core(
    spec: &Spectrum2D,
    snr_main: f32,
    snr_shoulder: f32,
    shoulder_min_rel: f64,
    max_cluster_comps: usize,
    allow_negative: bool,
    bounds_x_ppm: (f64, f64),
    bounds_y_ppm: (f64, f64),
    min_sep_x_ppm: f64,
    min_sep_y_ppm: f64,
    sigma_x_pts: f64,
    sigma_y_pts: f64,
) -> Vec<Raw2DComponent> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 3 || ny < 3 || spec.data.len() < nx * ny {
        return Vec::new();
    }

    let noise = estimate_noise_mad(&spec.data).max(1e-12);
    let thresh_main = (snr_main.max(2.5) * noise) as f64;
    let thresh_shoulder = (snr_shoulder.max(3.0) * noise) as f64;

    let step_x = spec.axis_x.step_ppm.abs().max(1e-9);
    let step_y = spec.axis_y.step_ppm.abs().max(1e-9);
    let min_dx_pts = (min_sep_x_ppm / step_x).max(0.65);
    let min_dy_pts = (min_sep_y_ppm / step_y).max(0.65);

    let in_bounds = |ix: usize, iy: usize| -> bool {
        let px = spec.axis_x.index_to_ppm(ix as f64);
        let py = spec.axis_y.index_to_ppm(iy as f64);
        px >= bounds_x_ppm.0 && px <= bounds_x_ppm.1 && py >= bounds_y_ppm.0 && py <= bounds_y_ppm.1
    };

    // Per-column (indirect y) local MAD floor to suppress multiplicative t1 ridges
    let mut col_floor = vec![thresh_main; nx];
    let mut scratch = Vec::with_capacity(ny);
    for (ix, slot) in col_floor.iter_mut().enumerate() {
        scratch.clear();
        for iy in 0..ny {
            scratch.push(spec.get(ix, iy));
        }
        scratch.sort_by(|a, b| a.total_cmp(b));
        let med = scratch[ny / 2];
        for v in &mut scratch {
            *v = (*v - med).abs();
        }
        scratch.sort_by(|a, b| a.total_cmp(b));
        let mad = (1.4826 * scratch[ny / 2]).max(noise);
        *slot = thresh_main.max(4.8 * (mad as f64));
    }

    let mut visited = vec![false; nx * ny];
    let mut queue = VecDeque::new();
    let mut out = Vec::new();

    let inv_2sx2 = 0.5 / (sigma_x_pts * sigma_x_pts).max(0.15);
    let inv_2sy2 = 0.5 / (sigma_y_pts * sigma_y_pts).max(0.15);

    for sy in 1..(ny - 1) {
        for sx in 1..(nx - 1) {
            let flat0 = sy * nx + sx;
            if visited[flat0] || !in_bounds(sx, sy) {
                continue;
            }
            let v0 = spec.data[flat0] as f64;
            let floor0 = col_floor[sx];
            let sign = if v0 >= floor0 {
                1i32
            } else if allow_negative && v0 <= -floor0 {
                -1i32
            } else {
                continue;
            };

            // Flood-fill same-sign connected island above 0.75 * thresh_main
            let grow_floor = 0.75 * thresh_main;
            let mut pixels: Vec<(usize, usize, f64)> = Vec::new();
            let mut island_max = 0.0f64;
            visited[flat0] = true;
            queue.push_back((sx, sy));

            while let Some((cx, cy)) = queue.pop_front() {
                let sv = (spec.get(cx, cy) as f64) * (sign as f64);
                if sv > island_max {
                    island_max = sv;
                }
                pixels.push((cx, cy, sv));
                let x_lo = cx.saturating_sub(1).max(1);
                let x_hi = (cx + 1).min(nx - 2);
                let y_lo = cy.saturating_sub(1).max(1);
                let y_hi = (cy + 1).min(ny - 2);
                for ny_i in y_lo..=y_hi {
                    for nx_i in x_lo..=x_hi {
                        let nf = ny_i * nx + nx_i;
                        if !visited[nf] && in_bounds(nx_i, ny_i) {
                            let nv = (spec.data[nf] as f64) * (sign as f64);
                            if nv >= grow_floor {
                                visited[nf] = true;
                                queue.push_back((nx_i, ny_i));
                            }
                        }
                    }
                }
            }

            if pixels.is_empty() || island_max < thresh_main {
                continue;
            }
            // Suppress single-voxel white-noise spikes when physical linewidth spans >= 0.55 grid points
            if pixels.len() == 1
                && sigma_x_pts >= 0.50
                && sigma_y_pts >= 0.50
                && island_max < 7.2 * (noise as f64)
            {
                let (px, py, _) = pixels[0];
                let max_cardinal = [
                    spec.get(px - 1, py) * (sign as f32),
                    spec.get(px + 1, py) * (sign as f32),
                    spec.get(px, py - 1) * (sign as f32),
                    spec.get(px, py + 1) * (sign as f32),
                ]
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
                if max_cardinal < 1.8 * noise {
                    continue;
                }
            }

            // 1. Seed with all strict 8-neighbor local maxima inside this island
            let mut comps: Vec<Raw2DComponent> = Vec::new();
            for &(px, py, sv) in &pixels {
                if sv < col_floor[px] {
                    continue;
                }
                let mut is_max = true;
                'nbr: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let qx = (px as i32 + dx) as usize;
                        let qy = (py as i32 + dy) as usize;
                        let qv = (spec.get(qx, qy) as f64) * (sign as f64);
                        if qv > sv || (qv == sv && (qy, qx) < (py, px)) {
                            is_max = false;
                            break 'nbr;
                        }
                    }
                }
                if is_max {
                    let ox = subvoxel_1d(
                        spec.get(px - 1, py) as f64 * (sign as f64),
                        sv,
                        spec.get(px + 1, py) as f64 * (sign as f64),
                    );
                    let oy = subvoxel_1d(
                        spec.get(px, py - 1) as f64 * (sign as f64),
                        sv,
                        spec.get(px, py + 1) as f64 * (sign as f64),
                    );
                    comps.push(Raw2DComponent {
                        x_idx: (px as f64) + ox,
                        y_idx: (py as f64) + oy,
                        amp: sv,
                        sign,
                        deconvolved: false,
                    });
                }
            }

            if comps.is_empty() {
                if let Some(&(bx, by, bsv)) = pixels.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
                    if bsv >= col_floor[bx] {
                        comps.push(Raw2DComponent {
                            x_idx: bx as f64,
                            y_idx: by as f64,
                            amp: bsv,
                            sign,
                            deconvolved: false,
                        });
                    }
                }
            }
            if comps.is_empty() {
                continue;
            }

            // Helper: jointly refine component amplitudes (non-negative Gauss-Seidel) + sub-voxel centroids
            let refine_island = |comps: &mut [Raw2DComponent]| {
                let m = comps.len();
                if m == 0 {
                    return;
                }
                // 3 iterations of non-negative least-squares / coordinate-descent amplitude & centroid update
                for _iter in 0..3 {
                    for k in 0..m {
                        let mut num = 0.0f64;
                        let mut den = 0.0f64;
                        let mut wx_num = 0.0f64;
                        let mut wy_num = 0.0f64;
                        let mut w_den = 0.0f64;
                        let cx = comps[k].x_idx;
                        let cy = comps[k].y_idx;

                        for &(px, py, sv) in &pixels {
                            let dx = (px as f64) - cx;
                            let dy = (py as f64) - cy;
                            if dx.abs() > 3.2 * sigma_x_pts || dy.abs() > 3.2 * sigma_y_pts {
                                continue;
                            }
                            let gk = (-dx * dx * inv_2sx2 - dy * dy * inv_2sy2).exp();
                            let mut other = 0.0f64;
                            for (j, cj) in comps.iter().enumerate() {
                                if j == k {
                                    continue;
                                }
                                let djx = (px as f64) - cj.x_idx;
                                let djy = (py as f64) - cj.y_idx;
                                if djx.abs() <= 4.0 * sigma_x_pts && djy.abs() <= 4.0 * sigma_y_pts
                                {
                                    other += cj.amp
                                        * (-djx * djx * inv_2sx2 - djy * djy * inv_2sy2).exp();
                                }
                            }
                            let rem = (sv - other).max(0.0);
                            num += rem * gk;
                            den += gk * gk;
                            if dx.abs() <= 1.6 * sigma_x_pts && dy.abs() <= 1.6 * sigma_y_pts {
                                let w = rem * gk;
                                wx_num += w * (px as f64);
                                wy_num += w * (py as f64);
                                w_den += w;
                            }
                        }
                        if den > 1e-9 {
                            comps[k].amp = (num / den).max(0.0);
                        }
                        if m > 1 && w_den > 1e-9 {
                            let nx_c = (wx_num / w_den).clamp(cx - 0.65, cx + 0.65);
                            let ny_c = (wy_num / w_den).clamp(cy - 0.65, cy + 0.65);
                            comps[k].x_idx = 0.5 * cx + 0.5 * nx_c;
                            comps[k].y_idx = 0.5 * cy + 0.5 * ny_c;
                        }
                    }
                }
            };

            if pixels.len() >= 6 {
                refine_island(&mut comps);

                // 2. Iterative Gaussian residual peeling to rescue unresolved shoulders in crowded clusters
                let min_shoulder = thresh_shoulder.max(shoulder_min_rel * island_max);
                while comps.len() < max_cluster_comps {
                    let mut best_res = 0.0f64;
                    let mut best_px = 0usize;
                    let mut best_py = 0usize;

                    for &(px, py, sv) in &pixels {
                        if sv < col_floor[px].min(min_shoulder) {
                            continue;
                        }
                        let too_close = comps.iter().any(|c| {
                            let rx = ((px as f64) - c.x_idx) / min_dx_pts;
                            let ry = ((py as f64) - c.y_idx) / min_dy_pts;
                            rx * rx + ry * ry < 1.0
                        });
                        if too_close {
                            continue;
                        }
                        let mut pred = 0.0f64;
                        for c in &comps {
                            let dx = (px as f64) - c.x_idx;
                            let dy = (py as f64) - c.y_idx;
                            if dx.abs() <= 4.5 * sigma_x_pts && dy.abs() <= 4.5 * sigma_y_pts {
                                pred += c.amp * (-dx * dx * inv_2sx2 - dy * dy * inv_2sy2).exp();
                            }
                        }
                        let res = sv - pred;
                        if res > best_res {
                            best_res = res;
                            best_px = px;
                            best_py = py;
                        }
                    }

                    if best_res < min_shoulder {
                        break;
                    }

                    comps.push(Raw2DComponent {
                        x_idx: best_px as f64,
                        y_idx: best_py as f64,
                        amp: best_res,
                        sign,
                        deconvolved: true,
                    });
                    refine_island(&mut comps);
                }
            }

            // Retain components clearing threshold and coalesce any components that drifted within 0.70 * min_sep
            comps.sort_by(|a, b| b.amp.total_cmp(&a.amp));
            let mut kept: Vec<Raw2DComponent> = Vec::with_capacity(comps.len());
            for c in comps {
                if c.amp < thresh_shoulder.min(thresh_main) {
                    continue;
                }
                if let Some(existing) = kept.iter_mut().find(|k| {
                    let rx = (c.x_idx - k.x_idx) / (0.72 * min_dx_pts);
                    let ry = (c.y_idx - k.y_idx) / (0.72 * min_dy_pts);
                    rx * rx + ry * ry < 1.0
                }) {
                    let w = (existing.amp + c.amp).max(1e-12);
                    existing.x_idx = (existing.amp * existing.x_idx + c.amp * c.x_idx) / w;
                    existing.y_idx = (existing.amp * existing.y_idx + c.amp * c.y_idx) / w;
                    existing.amp = existing.amp.max(c.amp);
                } else {
                    kept.push(c);
                }
            }

            out.extend(kept);
        }
    }

    out
}

/// Suppress residual IPAP / S3E virtual-decoupling satellites at full `± ^1J_{C'CA}` (`~53–55 Hz`)
/// along `13C'` using field-strength-invariant `Hz -> ppm` conversion via `c_obs_mhz`.
///
/// In IPAP virtual homodecoupling (`IP ± AP`), shifting the selected doublet line by `∓J/2` places
/// any uncancelled opposite doublet component at full `± ^1J_{C'CA}` from the decoupled parent peak.
fn suppress_ipap_c_prime_satellites<T>(
    mut peaks: Vec<T>,
    c_obs_mhz: f64,
    other_tol_ppm: f64,
    noise_sigma: f64,
    cfg: &C13PickerConfig,
    get_coords: impl Fn(&T) -> (f64, f64, f64, i32),
) -> Vec<T> {
    if peaks.len() < 2 || cfg.ipap_max_residual_frac <= 0.0 {
        return peaks;
    }
    let lo_ppm = hz_to_ppm(cfg.ipap_j_window_hz.0, c_obs_mhz, 150.0);
    let hi_ppm = hz_to_ppm(cfg.ipap_j_window_hz.1, c_obs_mhz, 150.0);
    peaks.sort_by(|a, b| get_coords(b).2.abs().total_cmp(&get_coords(a).2.abs()));

    let n = peaks.len();
    let mut keep = vec![true; n];
    for j in 1..n {
        let (cj, oj, hj, sj) = get_coords(&peaks[j]);
        let cand_amp = hj.abs();
        let mut sum_parent_amp = 0.0_f64;
        let mut max_parent_amp = 0.0_f64;
        for i in 0..j {
            if !keep[i] {
                continue;
            }
            let (ci, oi, hi, si) = get_coords(&peaks[i]);
            let parent_amp = hi.abs();
            if parent_amp <= 0.0 || si != sj {
                continue;
            }
            let dc = (cj - ci).abs();
            let do_ppm = (oj - oi).abs();
            if do_ppm <= other_tol_ppm && dc >= lo_ppm && dc <= hi_ppm {
                sum_parent_amp += parent_amp;
                max_parent_amp = max_parent_amp.max(parent_amp);
            }
        }
        if max_parent_amp > 0.0 {
            let max_allowed = (cfg.ipap_max_residual_frac * sum_parent_amp
                + 2.5 * noise_sigma.max(0.0))
            .min(0.24 * max_parent_amp);
            if cand_amp <= max_allowed {
                keep[j] = false;
            }
        }
    }

    peaks
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| keep[i].then_some(p))
        .collect()
}

/// Pick peptide-bond `(13C'(i-1), 15N(i))` roots (including `-Xaa-Pro-` and `-Pro-Pro-` imino
/// correlations at `132–142 ppm`) from an IPAP/virtual-decoupled `2D CON` spectrum.
///
/// Strictly unlabelled: never consults primary sequence or chemical-shift priors.
pub fn pick_con_2d(spec: &Spectrum2D, cfg: &C13PickerConfig) -> Vec<Con2DPeak> {
    if spec.nx() < 3 || spec.ny() < 3 {
        return Vec::new();
    }
    let x_is_c = is_c_prime_axis(&spec.axis_x) || !is_n15_axis(&spec.axis_x);
    let (c_axis, n_axis) = if x_is_c {
        (&spec.axis_x, &spec.axis_y)
    } else {
        (&spec.axis_y, &spec.axis_x)
    };

    // Initial quick local maxima for empirical single-peak sigma estimation
    let noise = estimate_noise_mad(&spec.data).max(1e-12) as f64;
    let mut quick_max = Vec::new();
    for iy in 2..(spec.ny().saturating_sub(2)) {
        for ix in 2..(spec.nx().saturating_sub(2)) {
            let v = spec.get(ix, iy) as f64;
            if v >= 7.0 * noise
                && v >= spec.get(ix - 1, iy) as f64
                && v >= spec.get(ix + 1, iy) as f64
                && v >= spec.get(ix, iy - 1) as f64
                && v >= spec.get(ix, iy + 1) as f64
            {
                quick_max.push((ix, iy, v));
            }
        }
    }

    let (sx_pts, sy_pts) = if x_is_c {
        estimate_2d_peak_sigmas_pts(spec, &quick_max, 3.2, 11.0, 2.8, 10.0, 150.9, 60.8)
    } else {
        estimate_2d_peak_sigmas_pts(spec, &quick_max, 2.8, 10.0, 3.2, 11.0, 60.8, 150.9)
    };

    let min_sep_c_ppm = hz_to_ppm(cfg.min_sep_c_prime_hz, c_axis.obs_mhz, 150.9);
    let min_sep_n_ppm = hz_to_ppm(cfg.min_sep_n15_hz, n_axis.obs_mhz, 60.8);
    let (c_bounds, n_bounds_opt, _) = C13DetectedExp::Con2D.spectral_bounds_ppm();
    let n_bounds = n_bounds_opt.unwrap_or((103.0, 144.5));

    let raw = if x_is_c {
        pick_narrow_2d_core(
            spec,
            cfg.snr_2d,
            cfg.shoulder_snr_2d,
            cfg.shoulder_min_rel_amp,
            cfg.max_cluster_components,
            false,
            c_bounds,
            n_bounds,
            min_sep_c_ppm,
            min_sep_n_ppm,
            sx_pts,
            sy_pts,
        )
    } else {
        pick_narrow_2d_core(
            spec,
            cfg.snr_2d,
            cfg.shoulder_snr_2d,
            cfg.shoulder_min_rel_amp,
            cfg.max_cluster_components,
            false,
            n_bounds,
            c_bounds,
            min_sep_n_ppm,
            min_sep_c_ppm,
            sx_pts,
            sy_pts,
        )
    };

    let vol_scale = 2.0 * PI * sx_pts * sy_pts;
    let mut peaks: Vec<Con2DPeak> = raw
        .into_iter()
        .map(|r| {
            let (c_ppm, n_ppm) = if x_is_c {
                (
                    spec.axis_x.index_to_ppm(r.x_idx),
                    spec.axis_y.index_to_ppm(r.y_idx),
                )
            } else {
                (
                    spec.axis_y.index_to_ppm(r.y_idx),
                    spec.axis_x.index_to_ppm(r.x_idx),
                )
            };
            Con2DPeak {
                c_prev_ppm: c_ppm,
                n_curr_ppm: n_ppm,
                height: r.amp,
                volume: r.amp * vol_scale,
                is_proline_imino_region: n_ppm >= 131.5,
                deconvolved_from_cluster: r.deconvolved,
            }
        })
        .collect();

    let n_tol_ipap = hz_to_ppm(4.0, n_axis.obs_mhz, 60.8);
    peaks = suppress_ipap_c_prime_satellites(peaks, c_axis.obs_mhz, n_tol_ipap, noise, cfg, |p| {
        (p.c_prev_ppm, p.n_curr_ppm, p.height, 1)
    });

    peaks.sort_by(|a, b| {
        a.n_curr_ppm
            .total_cmp(&b.n_curr_ppm)
            .then_with(|| a.c_prev_ppm.total_cmp(&b.c_prev_ppm))
    });
    peaks
}

/// Pick intraresidue `(13CA(i), 13C'(i))` peaks from a `2D CACO` spectrum, handling constant-time
/// `^1J_{CACB}` `Gly` sign inversion as well as crowded `Ser`/`Thr`/`Pro` downfield clusters.
///
/// Strictly unlabelled: never consults primary sequence or chemical-shift priors.
pub fn pick_caco_2d(spec: &Spectrum2D, cfg: &C13PickerConfig) -> Vec<Caco2DPeak> {
    if spec.nx() < 3 || spec.ny() < 3 {
        return Vec::new();
    }
    let x_is_c = is_c_prime_axis(&spec.axis_x);
    let (c_axis, ca_axis) = if x_is_c {
        (&spec.axis_x, &spec.axis_y)
    } else {
        (&spec.axis_y, &spec.axis_x)
    };

    let noise = estimate_noise_mad(&spec.data).max(1e-12) as f64;
    let mut quick_max = Vec::new();
    for iy in 2..(spec.ny().saturating_sub(2)) {
        for ix in 2..(spec.nx().saturating_sub(2)) {
            let v = spec.get(ix, iy) as f64;
            if v.abs() >= 7.0 * noise
                && v.abs() >= (spec.get(ix - 1, iy) as f64).abs()
                && v.abs() >= (spec.get(ix + 1, iy) as f64).abs()
                && v.abs() >= (spec.get(ix, iy - 1) as f64).abs()
                && v.abs() >= (spec.get(ix, iy + 1) as f64).abs()
            {
                quick_max.push((ix, iy, v));
            }
        }
    }

    let (sx_pts, sy_pts) = if x_is_c {
        estimate_2d_peak_sigmas_pts(spec, &quick_max, 3.2, 11.0, 4.0, 13.0, 150.9, 150.9)
    } else {
        estimate_2d_peak_sigmas_pts(spec, &quick_max, 4.0, 13.0, 3.2, 11.0, 150.9, 150.9)
    };

    let min_sep_c_ppm = hz_to_ppm(cfg.min_sep_c_prime_hz, c_axis.obs_mhz, 150.9);
    let min_sep_ca_ppm = hz_to_ppm(cfg.min_sep_c_aliph_hz, ca_axis.obs_mhz, 150.9);
    let (c_bounds, _, ca_bounds_opt) = C13DetectedExp::Caco2D.spectral_bounds_ppm();
    let ca_bounds = ca_bounds_opt.unwrap_or((40.0, 69.0));

    let raw = if x_is_c {
        pick_narrow_2d_core(
            spec,
            cfg.snr_2d,
            cfg.shoulder_snr_2d,
            cfg.shoulder_min_rel_amp,
            cfg.max_cluster_components,
            cfg.caco_allow_negative,
            c_bounds,
            ca_bounds,
            min_sep_c_ppm,
            min_sep_ca_ppm,
            sx_pts,
            sy_pts,
        )
    } else {
        pick_narrow_2d_core(
            spec,
            cfg.snr_2d,
            cfg.shoulder_snr_2d,
            cfg.shoulder_min_rel_amp,
            cfg.max_cluster_components,
            cfg.caco_allow_negative,
            ca_bounds,
            c_bounds,
            min_sep_ca_ppm,
            min_sep_c_ppm,
            sx_pts,
            sy_pts,
        )
    };

    // Determine dominant sign of non-Gly aliphatic CA peaks (> 48.5 ppm)
    let mut pos_non_gly = 0usize;
    let mut neg_non_gly = 0usize;
    for r in &raw {
        let ca_ppm = if x_is_c {
            spec.axis_y.index_to_ppm(r.y_idx)
        } else {
            spec.axis_x.index_to_ppm(r.x_idx)
        };
        if ca_ppm > 48.5 {
            if r.sign >= 0 {
                pos_non_gly += 1;
            } else {
                neg_non_gly += 1;
            }
        }
    }
    let maj_sign = if neg_non_gly > pos_non_gly { -1 } else { 1 };
    let has_both_signs = pos_non_gly + neg_non_gly > 0 && raw.iter().any(|r| r.sign != maj_sign);

    let vol_scale = 2.0 * PI * sx_pts * sy_pts;
    let mut peaks: Vec<Caco2DPeak> = raw
        .into_iter()
        .map(|r| {
            let (c_ppm, ca_ppm) = if x_is_c {
                (
                    spec.axis_x.index_to_ppm(r.x_idx),
                    spec.axis_y.index_to_ppm(r.y_idx),
                )
            } else {
                (
                    spec.axis_y.index_to_ppm(r.y_idx),
                    spec.axis_x.index_to_ppm(r.x_idx),
                )
            };
            let kind = if has_both_signs && r.sign != maj_sign && ca_ppm <= 49.0 {
                CacoPeakKind::GlyOppositePhase
            } else if ca_ppm <= 47.8 {
                CacoPeakKind::GlyRegion
            } else if ca_ppm >= 58.0 {
                CacoPeakKind::SerThrProDownfield
            } else {
                CacoPeakKind::GeneralAliphatic
            };
            Caco2DPeak {
                c_curr_ppm: c_ppm,
                ca_curr_ppm: ca_ppm,
                sign: r.sign,
                height: r.amp * (r.sign as f64),
                volume: r.amp * vol_scale * (r.sign as f64),
                kind,
            }
        })
        .collect();

    let ca_tol_ipap = hz_to_ppm(4.5, ca_axis.obs_mhz, 150.9);
    peaks = suppress_ipap_c_prime_satellites(peaks, c_axis.obs_mhz, ca_tol_ipap, noise, cfg, |p| {
        (p.c_curr_ppm, p.ca_curr_ppm, p.height, p.sign)
    });

    peaks.sort_by(|a, b| {
        a.ca_curr_ppm
            .total_cmp(&b.ca_curr_ppm)
            .then_with(|| a.c_curr_ppm.total_cmp(&b.c_curr_ppm))
    });
    peaks
}

/// Identify `(c_prime_dim, n15_dim, c_aliph_dim)` (`0=x, 1=y, 2=z`) in a 3D `13C`-detected cube.
fn identify_3d_axes(spec: &Spectrum3D) -> (usize, usize, usize) {
    let axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let c_dim = (0..3).find(|&i| is_c_prime_axis(axes[i])).unwrap_or(0);
    let n_dim = (0..3)
        .find(|&i| i != c_dim && is_n15_axis(axes[i]))
        .unwrap_or(if c_dim == 1 { 0 } else { 1 });
    let aliph_dim = 3 - c_dim - n_dim;
    (c_dim, n_dim, aliph_dim)
}

#[inline]
fn get_3d_permuted(
    spec: &Spectrum3D,
    dims: (usize, usize, usize),
    ic: usize,
    in_idx: usize,
    ia: usize,
) -> f32 {
    let mut coords = [0usize; 3];
    coords[dims.0] = ic;
    coords[dims.1] = in_idx;
    coords[dims.2] = ia;
    spec.get(coords[0], coords[1], coords[2])
}

/// Estimate unlabelled uniform `(ΔC'_cube_minus_con, ΔN_cube_minus_con)` carrier offset between a
/// `CON`-anchored 3D cube (`CANCO`, `CBCACON`, `CCC(CO)N`) and `con_roots` by maximizing the
/// max-projection strip signal across a coarse-to-fine 2D `(ΔC', ΔN)` grid (`±0.35 ppm C'`, `±0.60 ppm N`).
fn estimate_3d_con_cube_offset(
    spec: &Spectrum3D,
    dims: (usize, usize, usize),
    con_roots: &[Con2DPeak],
) -> (f64, f64) {
    if con_roots.len() < 5 {
        return (0.0, 0.0);
    }
    let axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let ax_c = axes[dims.0];
    let ax_n = axes[dims.1];
    let n_aliph = axes[dims.2].size;
    if ax_c.size < 4 || ax_n.size < 4 || n_aliph < 4 {
        return (0.0, 0.0);
    }

    // Build 2D `(ic, in)` max-abs projection over the aliphatic axis once: O(nx * ny * nz)
    let nc = ax_c.size;
    let nn = ax_n.size;
    let mut proj = vec![0.0f32; nc * nn];
    for ic in 0..nc {
        for in_i in 0..nn {
            let mut m = 0.0f32;
            for ia in 0..n_aliph {
                let v = get_3d_permuted(spec, dims, ic, in_i, ia).abs();
                if v > m {
                    m = v;
                }
            }
            proj[ic * nn + in_i] = m;
        }
    }

    let eval_shift = |dc: f64, dn: f64| -> f64 {
        let mut sum = 0.0f64;
        for r in con_roots {
            let fc = ax_c.ppm_to_index(r.c_prev_ppm + dc);
            let fn_i = ax_n.ppm_to_index(r.n_curr_ppm + dn);
            if fc >= 0.0 && fc <= (nc - 1) as f64 && fn_i >= 0.0 && fn_i <= (nn - 1) as f64 {
                let ic0 = fc.floor() as usize;
                let in0 = fn_i.floor() as usize;
                let ic1 = (ic0 + 1).min(nc - 1);
                let in1 = (in0 + 1).min(nn - 1);
                let tc = fc - (ic0 as f64);
                let tn = fn_i - (in0 as f64);
                let v00 = proj[ic0 * nn + in0] as f64;
                let v10 = proj[ic1 * nn + in0] as f64;
                let v01 = proj[ic0 * nn + in1] as f64;
                let v11 = proj[ic1 * nn + in1] as f64;
                let val =
                    (1.0 - tc) * ((1.0 - tn) * v00 + tn * v01) + tc * ((1.0 - tn) * v10 + tn * v11);
                sum += val;
            }
        }
        sum
    };

    let base_score = eval_shift(0.0, 0.0);
    let mut best_score = base_score;
    let mut best_dc = 0.0f64;
    let mut best_dn = 0.0f64;

    for ic_s in -14i32..=14 {
        let dc = (ic_s as f64) * 0.025;
        for in_s in -15i32..=15 {
            let dn = (in_s as f64) * 0.04;
            let sc = eval_shift(dc, dn);
            if sc > best_score {
                best_score = sc;
                best_dc = dc;
                best_dn = dn;
            }
        }
    }

    // Fine 2D refinement around best coarse shift
    let c0 = best_dc;
    let n0 = best_dn;
    for ic_s in -6i32..=6 {
        let dc = c0 + (ic_s as f64) * 0.005;
        for in_s in -6i32..=6 {
            let dn = n0 + (in_s as f64) * 0.008;
            let sc = eval_shift(dc, dn);
            if sc > best_score {
                best_score = sc;
                best_dc = dc;
                best_dn = dn;
            }
        }
    }

    // Apply only if alignment improves projected strip intensity appreciably (> 4%)
    if best_score > 1.04 * base_score && (best_dc.abs() >= 0.015 || best_dn.abs() >= 0.025) {
        (best_dc, best_dn)
    } else {
        (0.0, 0.0)
    }
}

/// Pick 1D aliphatic carbon maxima + unresolved shoulders along a 1D strip `profile[0..n_aliph]`.
fn pick_1d_aliph_strip(
    profile: &[f64],
    aliph_axis: &AxisPpm,
    aliph_bounds: (f64, f64),
    thresh: f64,
    allow_negative: bool,
    min_sep_aliph_ppm: f64,
) -> Vec<(f64, i32, f64)> {
    let n = profile.len();
    if n < 3 {
        return Vec::new();
    }
    let step = aliph_axis.step_ppm.abs().max(1e-9);
    let min_sep_pts = (min_sep_aliph_ppm / step).max(0.75);
    let fwhm_pts = (hz_to_ppm(6.5, aliph_axis.obs_mhz, 150.9) / step).clamp(1.1, 5.0);
    let sigma_pts = (fwhm_pts / (8.0 * LN_2).sqrt()).max(0.55);
    let inv_2s2 = 0.5 / (sigma_pts * sigma_pts);

    let mut comps: Vec<(f64, i32, f64)> = Vec::new();
    for i in 1..(n - 1) {
        let ppm = aliph_axis.index_to_ppm(i as f64);
        if ppm < aliph_bounds.0 || ppm > aliph_bounds.1 {
            continue;
        }
        let v = profile[i];
        if v >= thresh
            && v >= profile[i - 1]
            && v >= profile[i + 1]
            && (v > profile[i - 1] || v > profile[i + 1])
        {
            let off = subvoxel_1d(profile[i - 1], v, profile[i + 1]);
            comps.push(((i as f64) + off, 1, v));
        } else if allow_negative
            && v <= -thresh
            && v <= profile[i - 1]
            && v <= profile[i + 1]
            && (v < profile[i - 1] || v < profile[i + 1])
        {
            let off = subvoxel_1d(-profile[i - 1], -v, -profile[i + 1]);
            comps.push(((i as f64) + off, -1, v.abs()));
        }
    }

    // Residual shoulder rescue for close `CA(i)` / `CA(i-1)` pairs in `CANCO` or `CCC(CO)N`
    let max_amp = comps.iter().fold(0.0f64, |m, c| m.max(c.2));
    if max_amp >= thresh {
        for _ in 0..4 {
            let mut best_res = 0.0f64;
            let mut best_idx = 0usize;
            let mut best_sgn = 1i32;
            for (i, &raw_v) in profile.iter().enumerate().take(n - 1).skip(1) {
                let ppm = aliph_axis.index_to_ppm(i as f64);
                if ppm < aliph_bounds.0 || ppm > aliph_bounds.1 {
                    continue;
                }
                let sgn = if raw_v >= 0.0 {
                    1
                } else if allow_negative {
                    -1
                } else {
                    continue;
                };
                if comps
                    .iter()
                    .any(|c| c.1 == sgn && ((i as f64) - c.0).abs() < min_sep_pts)
                {
                    continue;
                }
                let mut pred = 0.0f64;
                for &(cx, cs, ca) in &comps {
                    if cs == sgn {
                        let d = (i as f64) - cx;
                        if d.abs() <= 4.5 * sigma_pts {
                            pred += ca * (-d * d * inv_2s2).exp();
                        }
                    }
                }
                let res = raw_v * (sgn as f64) - pred;
                if res > best_res {
                    best_res = res;
                    best_idx = i;
                    best_sgn = sgn;
                }
            }
            if best_res < thresh.max(0.18 * max_amp) {
                break;
            }
            comps.push((best_idx as f64, best_sgn, best_res));
        }
    }

    // Suppress 1D FT truncation sinc ripples (< 4% of strip maximum)
    let sinc_floor = 0.04 * max_amp;
    comps
        .into_iter()
        .filter(|c| c.2 >= thresh.max(sinc_floor))
        .map(|(idx, sgn, amp)| (aliph_axis.index_to_ppm(idx), sgn, amp))
        .collect()
}

/// Pick 3D peaks from `CANCO`, `CBCACON`, `CCC(CO)N`, or `NCACO` anchored on `2D CON` roots.
///
/// Strictly unlabelled: never consults primary sequence or chemical-shift priors.
pub fn pick_con_anchored_3d(
    spec: &Spectrum3D,
    exp: C13DetectedExp,
    con_roots: &[Con2DPeak],
    cfg: &C13PickerConfig,
) -> Vec<C13Strip3DPeak> {
    if spec.nx() < 3 || spec.ny() < 3 || spec.nz() < 3 || con_roots.is_empty() {
        return Vec::new();
    }
    let dims = identify_3d_axes(spec);
    let axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let ax_c = axes[dims.0];
    let ax_n = axes[dims.1];
    let ax_aliph = axes[dims.2];

    let noise = estimate_noise_mad(&spec.data).max(1e-12) as f64;
    let thresh = (cfg.snr_3d.max(3.0) as f64) * noise;
    let (c_bounds, _, aliph_bounds_opt) = exp.spectral_bounds_ppm();
    let aliph_bounds = aliph_bounds_opt.unwrap_or((10.0, 72.0));
    let min_sep_aliph_ppm = hz_to_ppm(cfg.min_sep_c_aliph_hz, ax_aliph.obs_mhz, 150.9);
    let allow_neg = exp == C13DetectedExp::Cbcaco3D && cfg.cbcacon_allow_negative;

    // NCACO correlates `(N(i), CA(i), C'(i))` where `N(i)` matches `con_roots[r].n_curr_ppm`,
    // while `C'(i)` varies freely across `c_bounds`!
    if exp == C13DetectedExp::Ncaco3D {
        return pick_ncaco_3d_planes(spec, dims, con_roots, thresh, c_bounds, aliph_bounds, cfg);
    }

    let (dc_reg, dn_reg) = if cfg.auto_register_3d_cubes {
        estimate_3d_con_cube_offset(spec, dims, con_roots)
    } else {
        (0.0, 0.0)
    };

    let tol_c_ppm =
        hz_to_ppm(cfg.strip_tol_c_prime_hz, ax_c.obs_mhz, 150.9).max(0.65 * ax_c.step_ppm.abs());
    let tol_n_ppm =
        hz_to_ppm(cfg.strip_tol_n15_hz, ax_n.obs_mhz, 60.8).max(0.65 * ax_n.step_ppm.abs());

    let nc = ax_c.size;
    let nn = ax_n.size;
    let na = ax_aliph.size;
    let mut profile = vec![0.0f64; na];
    let mut out = Vec::new();

    for (r_idx, root) in con_roots.iter().enumerate() {
        let c_target = root.c_prev_ppm + dc_reg;
        let n_target = root.n_curr_ppm + dn_reg;

        let ic_a = ax_c.ppm_to_index(c_target + tol_c_ppm);
        let ic_b = ax_c.ppm_to_index(c_target - tol_c_ppm);
        let ic0 = ic_a.min(ic_b).floor().clamp(0.0, (nc - 1) as f64) as usize;
        let ic1 = ic_a.max(ic_b).ceil().clamp(0.0, (nc - 1) as f64) as usize;

        let in_a = ax_n.ppm_to_index(n_target + tol_n_ppm);
        let in_b = ax_n.ppm_to_index(n_target - tol_n_ppm);
        let in0 = in_a.min(in_b).floor().clamp(0.0, (nn - 1) as f64) as usize;
        let in1 = in_a.max(in_b).ceil().clamp(0.0, (nn - 1) as f64) as usize;

        // Weighted interpolation around `(c_target, n_target)` to suppress bleed from adjacent roots
        profile.fill(0.0);
        let mut w_sum = 0.0f64;
        for ic in ic0..=ic1 {
            let dc = (ax_c.index_to_ppm(ic as f64) - c_target) / tol_c_ppm.max(1e-6);
            for in_i in in0..=in1 {
                let dn = (ax_n.index_to_ppm(in_i as f64) - n_target) / tol_n_ppm.max(1e-6);
                let d2 = dc * dc + dn * dn;
                if d2 <= 1.35 {
                    let w = (-0.5 * d2).exp();
                    w_sum += w;
                    for (ia, slot) in profile.iter_mut().enumerate() {
                        *slot += w * (get_3d_permuted(spec, dims, ic, in_i, ia) as f64);
                    }
                }
            }
        }
        if w_sum <= 1e-9 {
            let ic_near = ax_c
                .ppm_to_index(c_target)
                .round()
                .clamp(0.0, (nc - 1) as f64) as usize;
            let in_near = ax_n
                .ppm_to_index(n_target)
                .round()
                .clamp(0.0, (nn - 1) as f64) as usize;
            for (ia, slot) in profile.iter_mut().enumerate() {
                *slot = get_3d_permuted(spec, dims, ic_near, in_near, ia) as f64;
            }
        } else {
            // Normalize by peak weight (`max w` ~ 1) while accounting for noise reduction `sqrt(sum w^2) / sum w`
            let mut w2_sum = 0.0f64;
            for ic in ic0..=ic1 {
                let dc = (ax_c.index_to_ppm(ic as f64) - c_target) / tol_c_ppm.max(1e-6);
                for in_i in in0..=in1 {
                    let dn = (ax_n.index_to_ppm(in_i as f64) - n_target) / tol_n_ppm.max(1e-6);
                    let d2 = dc * dc + dn * dn;
                    if d2 <= 1.35 {
                        let w = (-0.5 * d2).exp();
                        w2_sum += w * w;
                    }
                }
            }
            let noise_scale = w2_sum.sqrt().max(1e-9);
            for slot in &mut profile {
                *slot /= noise_scale;
            }
        }

        let hits = pick_1d_aliph_strip(
            &profile,
            ax_aliph,
            aliph_bounds,
            thresh,
            allow_neg,
            min_sep_aliph_ppm,
        );
        for (c_ind_ppm, sgn, amp) in hits {
            out.push(C13Strip3DPeak {
                exp,
                c_prime_ppm: root.c_prev_ppm,
                n_ppm: root.n_curr_ppm,
                c_indirect_ppm: c_ind_ppm,
                sign: sgn,
                height: amp * (sgn as f64),
                volume: amp * (sgn as f64),
                anchor_root_idx: Some(r_idx),
            });
        }
    }

    out
}

/// Pick 3D `NCACO` `(N(i), CA(i), C'(i))` peaks across the cube and attach/register `N(i)` to
/// `con_roots` (returning unlabelled `(c_prime_ppm, n_ppm, ca_ppm)` peaks registered into the `CON` `N`/`C'` frame).
fn pick_ncaco_3d_planes(
    spec: &Spectrum3D,
    dims: (usize, usize, usize),
    con_roots: &[Con2DPeak],
    thresh: f64,
    c_bounds: (f64, f64),
    ca_bounds: (f64, f64),
    cfg: &C13PickerConfig,
) -> Vec<C13Strip3DPeak> {
    let axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let ax_c = axes[dims.0];
    let ax_n = axes[dims.1];
    let ax_ca = axes[dims.2];
    let nc = ax_c.size;
    let nn = ax_n.size;
    let nca = ax_ca.size;

    let mut raw_3d: Vec<(f64, f64, f64, f64)> = Vec::new();
    for ic in 1..(nc - 1) {
        let c_ppm = ax_c.index_to_ppm(ic as f64);
        if c_ppm < c_bounds.0 || c_ppm > c_bounds.1 {
            continue;
        }
        for in_i in 1..(nn - 1) {
            for ica in 1..(nca - 1) {
                let v = get_3d_permuted(spec, dims, ic, in_i, ica) as f64;
                if v < thresh {
                    continue;
                }
                let ca_ppm = ax_ca.index_to_ppm(ica as f64);
                if ca_ppm < ca_bounds.0 || ca_ppm > ca_bounds.1 {
                    continue;
                }
                let mut is_max = true;
                'nbr: for dc in -1i32..=1 {
                    for dn in -1i32..=1 {
                        for dca in -1i32..=1 {
                            if dc == 0 && dn == 0 && dca == 0 {
                                continue;
                            }
                            let q = get_3d_permuted(
                                spec,
                                dims,
                                (ic as i32 + dc) as usize,
                                (in_i as i32 + dn) as usize,
                                (ica as i32 + dca) as usize,
                            ) as f64;
                            if q > v {
                                is_max = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if is_max {
                    let oc = subvoxel_1d(
                        get_3d_permuted(spec, dims, ic - 1, in_i, ica) as f64,
                        v,
                        get_3d_permuted(spec, dims, ic + 1, in_i, ica) as f64,
                    );
                    let on = subvoxel_1d(
                        get_3d_permuted(spec, dims, ic, in_i - 1, ica) as f64,
                        v,
                        get_3d_permuted(spec, dims, ic, in_i + 1, ica) as f64,
                    );
                    let oca = subvoxel_1d(
                        get_3d_permuted(spec, dims, ic, in_i, ica - 1) as f64,
                        v,
                        get_3d_permuted(spec, dims, ic, in_i, ica + 1) as f64,
                    );
                    raw_3d.push((
                        ax_c.index_to_ppm((ic as f64) + oc),
                        ax_n.index_to_ppm((in_i as f64) + on),
                        ax_ca.index_to_ppm((ica as f64) + oca),
                        v,
                    ));
                }
            }
        }
    }

    // Unsupervised 1D registration of NCACO `N` and `C'` onto `CON` `N` and `C'` distributions
    let (dc_reg, dn_reg) =
        if cfg.auto_register_3d_cubes && raw_3d.len() >= 5 && con_roots.len() >= 5 {
            let fit_1d = |obs: &[f64], ref_vals: &[f64], max_shift: f64, bw: f64| -> f64 {
                let mut best_d = 0.0f64;
                let mut best_s = -1.0f64;
                for step in -60i32..=60 {
                    let d = (step as f64) * (max_shift / 60.0);
                    let mut s = 0.0f64;
                    for &x in obs {
                        let near = ref_vals
                            .iter()
                            .map(|&y| ((x - d - y) / bw).powi(2))
                            .fold(f64::INFINITY, f64::min);
                        s += (-0.5 * near).exp();
                    }
                    if s > best_s {
                        best_s = s;
                        best_d = d;
                    }
                }
                best_d
            };
            let obs_c: Vec<f64> = raw_3d.iter().map(|p| p.0).collect();
            let ref_c: Vec<f64> = con_roots.iter().map(|r| r.c_prev_ppm).collect();
            let obs_n: Vec<f64> = raw_3d.iter().map(|p| p.1).collect();
            let ref_n: Vec<f64> = con_roots.iter().map(|r| r.n_curr_ppm).collect();
            (
                fit_1d(&obs_c, &ref_c, 0.40, 0.05),
                fit_1d(&obs_n, &ref_n, 0.60, 0.14),
            )
        } else {
            (0.0, 0.0)
        };

    let tol_n_match = hz_to_ppm(cfg.strip_tol_n15_hz * 1.6, ax_n.obs_mhz, 60.8).max(0.18);
    let mut out = Vec::with_capacity(raw_3d.len());
    for (c_raw, n_raw, ca_ppm, amp) in raw_3d {
        let c_corr = c_raw - dc_reg;
        let n_corr = n_raw - dn_reg;
        // A single `N(i)` plane in crowded IDPs can match multiple `CON` roots sharing `N(i)`;
        // emit anchored entry for every matching `CON` root within `tol_n_match` (or unanchored if none).
        let mut matched_any = false;
        for (r_idx, r) in con_roots.iter().enumerate() {
            if (r.n_curr_ppm - n_corr).abs() <= tol_n_match {
                matched_any = true;
                out.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Ncaco3D,
                    c_prime_ppm: c_corr,
                    n_ppm: r.n_curr_ppm,
                    c_indirect_ppm: ca_ppm,
                    sign: 1,
                    height: amp,
                    volume: amp,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }
        if !matched_any {
            out.push(C13Strip3DPeak {
                exp: C13DetectedExp::Ncaco3D,
                c_prime_ppm: c_corr,
                n_ppm: n_corr,
                c_indirect_ppm: ca_ppm,
                sign: 1,
                height: amp,
                volume: amp,
                anchor_root_idx: None,
            });
        }
    }
    out
}

// ============================================================================
// Volumetric 2D/3D IDP Cube Simulator & 1H^N Solvent-Exchange Comparison
// ============================================================================

/// Configuration for realistic volumetric 2D/3D `13C`-detected IDP spectrum simulation and
/// `1H^N` solvent-exchange comparison (`500–950 MHz`, `298–310 K`, `pH 6.0–8.0`).
#[derive(Clone, Debug)]
pub struct IdpCubeSimConfig {
    /// Spectrometer `1H` Larmor frequency in `MHz` (`e.g. 500.0, 600.0, 800.0, 950.0`).
    pub proton_mhz: f64,
    /// Sample temperature in Kelvin (`e.g. 298.15..=310.15`).
    pub temperature_k: f64,
    /// Sample pH (`e.g. 7.4` physiological where exposed `1H^N` amides exchange rapidly).
    pub ph: f64,
    /// Transverse linewidth (FWHM in `Hz`) of `13C'` in IPAP-decoupled IDPs (`4.0–8.0 Hz`, default `4.8 Hz`).
    pub lw_c_prime_hz: f64,
    /// Transverse linewidth (FWHM in `Hz`) of `15N` in IDPs (`3.5–7.0 Hz`, default `4.2 Hz`).
    pub lw_n15_hz: f64,
    /// Transverse linewidth (FWHM in `Hz`) of `13CA` in IDPs (`default = 6.2 Hz`).
    pub lw_ca_hz: f64,
    /// Transverse linewidth (FWHM in `Hz`) of `13CB` / sidechain carbons (`default = 7.2 Hz`).
    pub lw_cb_hz: f64,
    /// Intrinsic non-exchange `1H^N` linewidth in `Hz` (`default = 6.0 Hz` before $k_{\text{ex}}/\pi$ broadening).
    pub lw_h1_intrinsic_hz: f64,
    /// Peak signal-to-noise ratio in `2D CON` / `2D CACO` (`default = 45.0`).
    pub snr_2d: f64,
    /// Peak signal-to-noise ratio of `1J` pathways in 3D cubes (`default = 28.0`).
    pub snr_3d: f64,
    /// Fractional amplitude of residual IPAP `± ^1J_{C'CA}/2` decoupling satellites (`default = 0.025`).
    pub ipap_residual_frac: f64,
    /// Invert `Gly` `CA` sign relative to `CB`-bearing residues in `2D CACO` (`default = true`).
    pub gly_caco_opposite_phase: bool,
    /// Invert `CB` sign relative to `CA` in `3D CBCACON` (`default = true`).
    pub cbcacon_opposite_phase: bool,
    /// 2D `CON` grid dimensions `(nx_c_prime, ny_n15)` (`default = (1024, 1024)`).
    pub con_grid: (usize, usize),
    /// 2D `CACO` grid dimensions `(nx_c_prime, ny_ca)` (`default = (512, 512)`).
    pub caco_grid: (usize, usize),
    /// 3D cube grid dimensions `(nx_c_prime, ny_n15, nz_c_aliph)` (`default = (192, 192, 192)`).
    pub cube_3d_grid: (usize, usize, usize),
    /// Which `13C`-detected experiments to render.
    pub experiments: Vec<C13DetectedExp>,
    /// Per-experiment referencing offsets `(d_c_prime_ppm, d_n_ppm, d_c_aliph_ppm)`.
    pub exp_offsets: BTreeMap<C13DetectedExp, (f64, f64, f64)>,
    /// Optional model-mismatched random shift perturbation `(sd_n, sd_c_prime, sd_ca, sd_cb)` in ppm
    /// applied per residue before rendering (independent of `ncIDP`).
    pub empirical_scatter_ppm: Option<(f64, f64, f64, f64)>,
    /// Probability of dropping an individual 3D relay peak (`default = 0.0`).
    pub p_missing_3d_relay: f64,
    /// Fractional rate of spurious 3D artifact peaks (`default = 0.0`).
    pub spurious_3d_rate: f64,
    pub seed: u64,
}

impl Default for IdpCubeSimConfig {
    fn default() -> Self {
        Self {
            proton_mhz: 700.0,
            temperature_k: 298.15,
            ph: 7.4,
            lw_c_prime_hz: 4.8,
            lw_n15_hz: 4.2,
            lw_ca_hz: 6.2,
            lw_cb_hz: 7.2,
            lw_h1_intrinsic_hz: 6.0,
            snr_2d: 45.0,
            snr_3d: 28.0,
            ipap_residual_frac: 0.025,
            gly_caco_opposite_phase: true,
            cbcacon_opposite_phase: true,
            con_grid: (1024, 1024),
            caco_grid: (512, 512),
            cube_3d_grid: (192, 192, 192),
            experiments: vec![
                C13DetectedExp::Con2D,
                C13DetectedExp::Caco2D,
                C13DetectedExp::Canco3D,
                C13DetectedExp::Cbcaco3D,
                C13DetectedExp::Ncaco3D,
            ],
            exp_offsets: BTreeMap::new(),
            empirical_scatter_ppm: None,
            p_missing_3d_relay: 0.0,
            spurious_3d_rate: 0.0,
            seed: 42,
        }
    }
}

/// Simulated 2D and 3D `13C`-detected IDP spectra with ground-truth `CON(i)` coordinates.
#[derive(Clone, Debug)]
pub struct SimulatedIdpCubes {
    pub con_2d: Spectrum2D,
    pub caco_2d: Option<Spectrum2D>,
    pub cubes_3d: Vec<(C13DetectedExp, Spectrum3D)>,
    /// Ground-truth `(residue_idx_i, c_prev_ppm_in_con, n_curr_ppm_in_con, prev_is_pro, curr_is_pro)`
    /// for every peptide bond `i in 1..L` that has deposited `(C'(i-1), N(i))` shifts in `Reference`.
    pub ground_truth_con: Vec<(usize, f64, f64, bool, bool)>,
}

/// Quantitative comparison of `1H`-detected `15N-HSQC` vs protonless `13C`-detected `2D CON`
/// at elevated pH/temperature (`e.g. pH 7.4, 298–310 K`).
#[derive(Clone, Debug)]
pub struct H1ExchangeComparison {
    pub proton_mhz: f64,
    pub temperature_k: f64,
    pub ph: f64,
    pub total_peptide_bonds: usize,
    pub proline_peptide_bonds: usize,
    pub amide_bonds_broadened_below_snr: usize,
    pub hsqc_2d: Spectrum2D,
    pub picked_hsqc_roots: usize,
    pub hsqc_bond_recovery_frac: f64,
    pub picked_con_roots: usize,
    pub con_bond_recovery_frac: f64,
    pub per_residue_kex_s_inv: Vec<f64>,
}

/// Intrinsic unprotected random-coil amide `1H^N` solvent-exchange rate $k_{\text{ex}}$ ($\text{s}^{-1}$)
/// as a function of primary sequence, `pH`, and `temperature_k` (Bai, Milne, Mayne & Englander,
/// *Proteins* 1993, 17, 75–86).
pub fn intrinsic_amide_h_exchange_rate(
    prev_res: &str,
    curr_res: &str,
    ph: f64,
    temperature_k: f64,
) -> f64 {
    if curr_res.eq_ignore_ascii_case("PRO") {
        return 0.0;
    }
    // Reference base-catalyzed rate constant for poly-DL-Ala in H2O at 293.15 K:
    // k_B,ref ~ 1.26e9 M^-1 s^-1, log10(K_w(293.15 K)) = -14.17 -> k_base(pH 7.0, 293 K) ~ 8.5 s^-1
    let log_side_curr = match curr_res.to_ascii_uppercase().as_str() {
        "GLY" => 0.28,
        "SER" => 0.42,
        "THR" => 0.30,
        "HIS" => 0.52,
        "ASN" => 0.48,
        "GLN" => 0.22,
        "ARG" | "LYS" => 0.26,
        "ASP" => -0.18,
        "GLU" => -0.24,
        "ILE" | "VAL" => -0.45,
        "LEU" => -0.25,
        _ => 0.0,
    };
    let log_side_prev = match prev_res.to_ascii_uppercase().as_str() {
        "GLY" => 0.22,
        "SER" | "THR" => 0.34,
        "HIS" | "ASN" | "ARG" | "LYS" => 0.30,
        "PRO" => -0.15,
        "ASP" | "GLU" => -0.20,
        "ILE" | "VAL" => -0.32,
        _ => 0.0,
    };
    // Activation enthalpy ~ 17 kcal/mol (71.1 kJ/mol) including water ionization enthalpy (Bai et al. 1993)
    let r_kj = 0.008_314_462;
    let temp = temperature_k.clamp(273.15, 340.0);
    let thermal = (71.1 / r_kj * (1.0 / 293.15 - 1.0 / temp)).exp();

    let ph_clamped = ph.clamp(1.0, 10.5);
    let k_base = 8.5 * 10.0_f64.powf(ph_clamped - 7.0 + log_side_curr + log_side_prev) * thermal;
    let k_acid = 0.0007 * 10.0_f64.powf(3.0 - ph_clamped) * thermal;
    (k_base + k_acid).clamp(0.001, 50_000.0)
}

#[allow(clippy::too_many_arguments)]
fn splat_2d_gaussian(
    spec: &mut Spectrum2D,
    x_ppm: f64,
    y_ppm: f64,
    sigma_x_ppm: f64,
    sigma_y_ppm: f64,
    amp: f64,
) {
    let nx = spec.nx();
    let ny = spec.ny();
    let cx = spec.axis_x.ppm_to_index(x_ppm);
    let cy = spec.axis_y.ppm_to_index(y_ppm);
    let sx = (sigma_x_ppm / spec.axis_x.step_ppm.abs().max(1e-9)).max(0.45);
    let sy = (sigma_y_ppm / spec.axis_y.step_ppm.abs().max(1e-9)).max(0.45);
    let rx = (4.5 * sx).ceil() as i32;
    let ry = (4.5 * sy).ceil() as i32;
    let ix0 = ((cx.floor() as i32) - rx).max(0) as usize;
    let ix1 = ((cx.ceil() as i32) + rx).min((nx as i32) - 1).max(0) as usize;
    let iy0 = ((cy.floor() as i32) - ry).max(0) as usize;
    let iy1 = ((cy.ceil() as i32) + ry).min((ny as i32) - 1).max(0) as usize;
    if cx < -4.0 || cx > (nx + 4) as f64 || cy < -4.0 || cy > (ny + 4) as f64 {
        return;
    }
    let inv_2sx2 = 0.5 / (sx * sx);
    let inv_2sy2 = 0.5 / (sy * sy);
    for iy in iy0..=iy1 {
        let dy = (iy as f64) - cy;
        let wy = (-dy * dy * inv_2sy2).exp();
        let row = iy * nx;
        for ix in ix0..=ix1 {
            let dx = (ix as f64) - cx;
            spec.data[row + ix] += (amp * wy * (-dx * dx * inv_2sx2).exp()) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn splat_3d_gaussian(
    spec: &mut Spectrum3D,
    x_ppm: f64,
    y_ppm: f64,
    z_ppm: f64,
    sigma_x_ppm: f64,
    sigma_y_ppm: f64,
    sigma_z_ppm: f64,
    amp: f64,
) {
    let nx = spec.nx();
    let ny = spec.ny();
    let nz = spec.nz();
    let cx = spec.axis_x.ppm_to_index(x_ppm);
    let cy = spec.axis_y.ppm_to_index(y_ppm);
    let cz = spec.axis_z.ppm_to_index(z_ppm);
    if cx < -4.0
        || cx > (nx + 4) as f64
        || cy < -4.0
        || cy > (ny + 4) as f64
        || cz < -4.0
        || cz > (nz + 4) as f64
    {
        return;
    }
    let sx = (sigma_x_ppm / spec.axis_x.step_ppm.abs().max(1e-9)).max(0.45);
    let sy = (sigma_y_ppm / spec.axis_y.step_ppm.abs().max(1e-9)).max(0.45);
    let sz = (sigma_z_ppm / spec.axis_z.step_ppm.abs().max(1e-9)).max(0.45);
    let rx = (4.2 * sx).ceil() as i32;
    let ry = (4.2 * sy).ceil() as i32;
    let rz = (4.2 * sz).ceil() as i32;
    let ix0 = ((cx.floor() as i32) - rx).max(0) as usize;
    let ix1 = ((cx.ceil() as i32) + rx).min((nx as i32) - 1).max(0) as usize;
    let iy0 = ((cy.floor() as i32) - ry).max(0) as usize;
    let iy1 = ((cy.ceil() as i32) + ry).min((ny as i32) - 1).max(0) as usize;
    let iz0 = ((cz.floor() as i32) - rz).max(0) as usize;
    let iz1 = ((cz.ceil() as i32) + rz).min((nz as i32) - 1).max(0) as usize;

    let inv_2sx2 = 0.5 / (sx * sx);
    let inv_2sy2 = 0.5 / (sy * sy);
    let inv_2sz2 = 0.5 / (sz * sz);

    for iz in iz0..=iz1 {
        let dz = (iz as f64) - cz;
        let wz = (-dz * dz * inv_2sz2).exp();
        for iy in iy0..=iy1 {
            let dy = (iy as f64) - cy;
            let wyz = wz * (-dy * dy * inv_2sy2).exp();
            let base = (iz * ny + iy) * nx;
            for ix in ix0..=ix1 {
                let dx = (ix as f64) - cx;
                spec.data[base + ix] += (amp * wyz * (-dx * dx * inv_2sx2).exp()) as f32;
            }
        }
    }
}

/// Simulate realistic 2D (`CON`, `CACO`) and 3D (`CANCO`, `CBCACON`, `CCC(CO)N`, `NCACO`)
/// volumetric IDP cubes directly from deposited `Reference` shifts (`ref_data`), converting all
/// physical `Hz` linewidths and IPAP virtual-decoupling residuals via `cfg.proton_mhz`.
pub fn simulate_idp_13c_cubes(ref_data: &Reference, cfg: &IdpCubeSimConfig) -> SimulatedIdpCubes {
    let mut rng = Rng::new(cfg.seed ^ 0x13C0_4944_505F_5349);
    let c_mhz = c13_obs_mhz(cfg.proton_mhz);
    let n_mhz = n15_obs_mhz(cfg.proton_mhz);
    let fwhm_to_sigma = 1.0 / (8.0 * LN_2).sqrt();

    let sig_cp = hz_to_ppm(cfg.lw_c_prime_hz.max(2.0), c_mhz, 150.9) * fwhm_to_sigma;
    let sig_n = hz_to_ppm(cfg.lw_n15_hz.max(2.0), n_mhz, 60.8) * fwhm_to_sigma;
    let sig_ca = hz_to_ppm(cfg.lw_ca_hz.max(2.5), c_mhz, 150.9) * fwhm_to_sigma;
    let sig_cb = hz_to_ppm(cfg.lw_cb_hz.max(2.5), c_mhz, 150.9) * fwhm_to_sigma;
    // IPAP virtual homodecoupling (`IP ± AP`) shifts the retained doublet line by `∓J/2`,
    // leaving any uncancelled opposite doublet component at full `±^1J_{C'CA} ≈ 53.5 Hz`.
    let ipap_j_ppm = hz_to_ppm(53.5, c_mhz, 150.9);

    let seq_len = ref_data.sequence.len();
    // Optional per-residue empirical model-mismatch shift perturbation
    let mut shifts = ref_data.shifts.clone();
    if let Some((sd_n, sd_c, sd_ca, sd_cb)) = cfg.empirical_scatter_ppm {
        for i in 0..seq_len {
            if sd_n > 0.0 {
                if let Some(v) = shifts.get_mut(&(i, "N".to_string())) {
                    *v += rng.normal(0.0, sd_n);
                }
            }
            if sd_c > 0.0 {
                if let Some(v) = shifts.get_mut(&(i, "C".to_string())) {
                    *v += rng.normal(0.0, sd_c);
                }
            }
            if sd_ca > 0.0 {
                if let Some(v) = shifts.get_mut(&(i, "CA".to_string())) {
                    *v += rng.normal(0.0, sd_ca);
                }
            }
            if sd_cb > 0.0 {
                if let Some(v) = shifts.get_mut(&(i, "CB".to_string())) {
                    *v += rng.normal(0.0, sd_cb);
                }
            }
        }
    }

    let get_off = |exp: C13DetectedExp| -> (f64, f64, f64) {
        cfg.exp_offsets
            .get(&exp)
            .copied()
            .unwrap_or((0.0, 0.0, 0.0))
    };

    // 1. Render 2D CON: `axis_x = 13C'` (181.5..167.5 ppm), `axis_y = 15N` (143.5..104.5 ppm)
    let (nx_con, ny_con) = (cfg.con_grid.0.max(64), cfg.con_grid.1.max(64));
    let mut con_2d = Spectrum2D {
        name: "CON".to_string(),
        axis_x: AxisPpm::from_bounds("13C", nx_con, 181.5, 167.5, c_mhz),
        axis_y: AxisPpm::from_bounds("15N", ny_con, 143.5, 104.5, n_mhz),
        data: vec![0.0f32; nx_con * ny_con],
    };
    let (dc_con, dn_con, _) = get_off(C13DetectedExp::Con2D);
    let mut ground_truth_con = Vec::new();
    let i0_2d = 100.0f64;

    for i in 1..seq_len {
        let c_prev = shifts.get(&(i - 1, "C".to_string())).copied();
        let n_curr = shifts.get(&(i, "N".to_string())).copied();
        let (Some(cp_raw), Some(nc_raw)) = (c_prev, n_curr) else {
            continue;
        };
        let cp = cp_raw + dc_con;
        let nc = nc_raw + dn_con;
        let prev_pro = ref_data.sequence[i - 1].eq_ignore_ascii_case("PRO");
        let curr_pro = ref_data.sequence[i].eq_ignore_ascii_case("PRO");
        ground_truth_con.push((i, cp, nc, prev_pro, curr_pro));

        let amp = i0_2d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.70, 1.35);
        splat_2d_gaussian(&mut con_2d, cp, nc, sig_cp, sig_n, amp);
        if cfg.ipap_residual_frac > 0.0 {
            let sat_amp = amp * cfg.ipap_residual_frac;
            splat_2d_gaussian(&mut con_2d, cp + ipap_j_ppm, nc, sig_cp, sig_n, sat_amp);
            splat_2d_gaussian(&mut con_2d, cp - ipap_j_ppm, nc, sig_cp, sig_n, sat_amp);
        }
    }
    let noise_2d = i0_2d / cfg.snr_2d.max(5.0);
    for v in &mut con_2d.data {
        *v += rng.normal(0.0, noise_2d) as f32;
    }

    // 2. Optional 2D CACO: `axis_x = 13C'`, `axis_y = 13CA` (68.5..41.0 ppm)
    let caco_2d = if cfg.experiments.contains(&C13DetectedExp::Caco2D) {
        let (nx_ca, ny_ca) = (cfg.caco_grid.0.max(64), cfg.caco_grid.1.max(64));
        let mut spec = Spectrum2D {
            name: "CACO".to_string(),
            axis_x: AxisPpm::from_bounds("13C", nx_ca, 181.5, 167.5, c_mhz),
            axis_y: AxisPpm::from_bounds("13CA", ny_ca, 68.5, 41.0, c_mhz),
            data: vec![0.0f32; nx_ca * ny_ca],
        };
        let (dc_caco, _, dca_caco) = get_off(C13DetectedExp::Caco2D);
        for i in 0..seq_len {
            let Some(&c_val) = shifts.get(&(i, "C".to_string())) else {
                continue;
            };
            let Some(&ca_val) = shifts.get(&(i, "CA".to_string())) else {
                continue;
            };
            let is_gly = ref_data.sequence[i].eq_ignore_ascii_case("GLY");
            let sgn = if cfg.gly_caco_opposite_phase && is_gly {
                -1.0
            } else {
                1.0
            };
            let amp = sgn * i0_2d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.70, 1.35);
            let cp = c_val + dc_caco;
            let cap = ca_val + dca_caco;
            splat_2d_gaussian(&mut spec, cp, cap, sig_cp, sig_ca, amp);
            if cfg.ipap_residual_frac > 0.0 {
                let sat = amp * cfg.ipap_residual_frac;
                splat_2d_gaussian(&mut spec, cp + ipap_j_ppm, cap, sig_cp, sig_ca, sat);
                splat_2d_gaussian(&mut spec, cp - ipap_j_ppm, cap, sig_cp, sig_ca, sat);
            }
        }
        for v in &mut spec.data {
            *v += rng.normal(0.0, noise_2d) as f32;
        }
        Some(spec)
    } else {
        None
    };

    // 3. Render requested 3D cubes (`CANCO`, `CBCACON`, `CCC(CO)N`, `NCACO`)
    let mut cubes_3d = Vec::new();
    let (nx3, ny3, nz3) = (
        cfg.cube_3d_grid.0.max(48),
        cfg.cube_3d_grid.1.max(48),
        cfg.cube_3d_grid.2.max(48),
    );
    let i0_3d = 100.0f64;
    let noise_3d = i0_3d / cfg.snr_3d.max(5.0);

    for &exp in &cfg.experiments {
        if !exp.is_3d() {
            continue;
        }
        let (_, _, aliph_bounds_opt) = exp.spectral_bounds_ppm();
        let (alo, ahi) = aliph_bounds_opt.unwrap_or((12.0, 72.0));
        let mut cube = Spectrum3D {
            name: exp.as_str().to_string(),
            exp: exp.as_str().to_string(),
            axis_x: AxisPpm::from_bounds("13C", nx3, 181.5, 167.5, c_mhz),
            axis_y: AxisPpm::from_bounds("15N", ny3, 143.5, 104.5, n_mhz),
            axis_z: AxisPpm::from_bounds("13CA", nz3, ahi, alo, c_mhz),
            data: vec![0.0f32; nx3 * ny3 * nz3],
        };
        let (dc, dn, da) = get_off(exp);
        let mut n_true_peaks = 0usize;

        match exp {
            C13DetectedExp::Canco3D => {
                for i in 1..seq_len {
                    let (Some(&cp), Some(&nc)) = (
                        shifts.get(&(i - 1, "C".to_string())),
                        shifts.get(&(i, "N".to_string())),
                    ) else {
                        continue;
                    };
                    if let Some(&ca_i) = shifts.get(&(i, "CA".to_string())) {
                        if rng.f64() >= cfg.p_missing_3d_relay {
                            splat_3d_gaussian(
                                &mut cube,
                                cp + dc,
                                nc + dn,
                                ca_i + da,
                                sig_cp,
                                sig_n,
                                sig_ca,
                                i0_3d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
                            );
                            n_true_peaks += 1;
                        }
                    }
                    if let Some(&ca_im1) = shifts.get(&(i - 1, "CA".to_string())) {
                        if rng.f64() >= cfg.p_missing_3d_relay {
                            splat_3d_gaussian(
                                &mut cube,
                                cp + dc,
                                nc + dn,
                                ca_im1 + da,
                                sig_cp,
                                sig_n,
                                sig_ca,
                                0.68 * i0_3d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
                            );
                            n_true_peaks += 1;
                        }
                    }
                }
            }
            C13DetectedExp::Cbcaco3D => {
                for i in 1..seq_len {
                    let (Some(&cp), Some(&nc)) = (
                        shifts.get(&(i - 1, "C".to_string())),
                        shifts.get(&(i, "N".to_string())),
                    ) else {
                        continue;
                    };
                    if let Some(&ca_im1) = shifts.get(&(i - 1, "CA".to_string())) {
                        if rng.f64() >= cfg.p_missing_3d_relay {
                            splat_3d_gaussian(
                                &mut cube,
                                cp + dc,
                                nc + dn,
                                ca_im1 + da,
                                sig_cp,
                                sig_n,
                                sig_ca,
                                i0_3d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
                            );
                            n_true_peaks += 1;
                        }
                    }
                    if !ref_data.sequence[i - 1].eq_ignore_ascii_case("GLY") {
                        if let Some(&cb_im1) = shifts.get(&(i - 1, "CB".to_string())) {
                            if rng.f64() >= cfg.p_missing_3d_relay {
                                let sgn = if cfg.cbcacon_opposite_phase {
                                    -1.0
                                } else {
                                    1.0
                                };
                                splat_3d_gaussian(
                                    &mut cube,
                                    cp + dc,
                                    nc + dn,
                                    cb_im1 + da,
                                    sig_cp,
                                    sig_n,
                                    sig_cb,
                                    sgn * 0.88
                                        * i0_3d
                                        * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
                                );
                                n_true_peaks += 1;
                            }
                        }
                    }
                }
            }
            C13DetectedExp::Cccon3D => {
                let sc_atoms = ["CA", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE"];
                for i in 1..seq_len {
                    let (Some(&cp), Some(&nc)) = (
                        shifts.get(&(i - 1, "C".to_string())),
                        shifts.get(&(i, "N".to_string())),
                    ) else {
                        continue;
                    };
                    for &aname in &sc_atoms {
                        if let Some(&c_val) = shifts.get(&(i - 1, aname.to_string())) {
                            if (alo..=ahi).contains(&c_val) && rng.f64() >= cfg.p_missing_3d_relay {
                                let atten = if aname == "CA" || aname == "CB" {
                                    0.85
                                } else {
                                    0.62
                                };
                                splat_3d_gaussian(
                                    &mut cube,
                                    cp + dc,
                                    nc + dn,
                                    c_val + da,
                                    sig_cp,
                                    sig_n,
                                    sig_cb,
                                    atten * i0_3d,
                                );
                                n_true_peaks += 1;
                            }
                        }
                    }
                }
            }
            C13DetectedExp::Ncaco3D => {
                for i in 1..seq_len {
                    let (Some(&nc), Some(&ca_i), Some(&c_i)) = (
                        shifts.get(&(i, "N".to_string())),
                        shifts.get(&(i, "CA".to_string())),
                        shifts.get(&(i, "C".to_string())),
                    ) else {
                        continue;
                    };
                    if rng.f64() >= cfg.p_missing_3d_relay {
                        splat_3d_gaussian(
                            &mut cube,
                            c_i + dc,
                            nc + dn,
                            ca_i + da,
                            sig_cp,
                            sig_n,
                            sig_ca,
                            i0_3d * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
                        );
                        n_true_peaks += 1;
                    }
                }
            }
            C13DetectedExp::Con2D | C13DetectedExp::Caco2D => {}
        }

        // Inject spurious 3D noise peaks if configured
        let n_spurious = ((n_true_peaks as f64) * cfg.spurious_3d_rate.max(0.0)).round() as usize;
        for _ in 0..n_spurious {
            let sc = 169.0 + rng.f64() * 10.5;
            let sn = 106.0 + rng.f64() * 35.0;
            let sa = (alo + 2.0) + rng.f64() * (ahi - alo - 4.0).max(1.0);
            splat_3d_gaussian(&mut cube, sc, sn, sa, sig_cp, sig_n, sig_cb, 0.32 * i0_3d);
        }

        for v in &mut cube.data {
            *v += rng.normal(0.0, noise_3d) as f32;
        }
        cubes_3d.push((exp, cube));
    }

    SimulatedIdpCubes {
        con_2d,
        caco_2d,
        cubes_3d,
        ground_truth_con,
    }
}

/// Simulate side-by-side `1H`-detected `2D 15N-HSQC` (with Bai–Englander `1H^N` solvent-exchange
/// broadening + INEPT transfer loss + invisible Prolines) vs protonless `13C`-detected `2D CON`
/// at identical `(proton_mhz, temperature_k, ph)` and compare blind root recovery!
pub fn simulate_h1_exchange_vs_c13_con(
    ref_data: &Reference,
    cfg: &IdpCubeSimConfig,
    picker_cfg: &C13PickerConfig,
) -> H1ExchangeComparison {
    let mut con_cfg = cfg.clone();
    con_cfg.experiments = vec![C13DetectedExp::Con2D];
    let cubes = simulate_idp_13c_cubes(ref_data, &con_cfg);
    let con_peaks = pick_con_2d(&cubes.con_2d, picker_cfg);

    let seq_len = ref_data.sequence.len();
    let n_mhz = n15_obs_mhz(cfg.proton_mhz);
    let h_mhz = cfg.proton_mhz.max(400.0);
    let fwhm_to_sigma = 1.0 / (8.0 * LN_2).sqrt();
    let sig_n = hz_to_ppm(cfg.lw_n15_hz.max(2.0), n_mhz, 60.8) * fwhm_to_sigma;

    let mut hsqc = Spectrum2D {
        name: "HSQC".to_string(),
        axis_x: AxisPpm::from_bounds("1H", 1024, 10.5, 6.5, h_mhz),
        axis_y: AxisPpm::from_bounds("15N", 512, 134.0, 104.0, n_mhz),
        data: vec![0.0f32; 1024 * 512],
    };

    let mut rng = Rng::new(cfg.seed ^ 0x4853_5143_4558_4348);
    let i0 = 100.0f64;
    let noise_hsqc = i0 / cfg.snr_2d.max(5.0);
    // Standard refocused 1H-15N HSQC total 1H^N antiphase/refocusing INEPT delay 2*tau ~ 10.8 ms (1/(2 J_NH))
    let two_tau_inept_s = 0.0108;

    let mut total_bonds = 0usize;
    let mut pro_bonds = 0usize;
    let mut broadened_amides = 0usize;
    let mut per_residue_kex = vec![0.0f64; seq_len];

    for (i, kex_slot) in per_residue_kex.iter_mut().enumerate().take(seq_len).skip(1) {
        let is_pro = ref_data.sequence[i].eq_ignore_ascii_case("PRO");
        let has_con = ref_data.shifts.contains_key(&(i - 1, "C".to_string()))
            && ref_data.shifts.contains_key(&(i, "N".to_string()));
        if !has_con {
            continue;
        }
        total_bonds += 1;
        if is_pro {
            pro_bonds += 1;
            continue;
        }
        let kex = intrinsic_amide_h_exchange_rate(
            &ref_data.sequence[i - 1],
            &ref_data.sequence[i],
            cfg.ph,
            cfg.temperature_k,
        );
        *kex_slot = kex;

        let Some(&n_ppm) = ref_data.shifts.get(&(i, "N".to_string())) else {
            continue;
        };
        // Use deposited `1H^N` when available, otherwise typical narrow IDP amide `8.25 ppm`
        let h_ppm = ref_data
            .shifts
            .get(&(i, "H".to_string()))
            .copied()
            .unwrap_or(8.25);

        // Exchange broadening: FWHM_eff = FWHM_0 + k_ex / pi; peak height scales as (FWHM_0 / FWHM_eff) * exp(-2 * tau * k_ex)
        let lw_eff_hz = cfg.lw_h1_intrinsic_hz + kex / PI;
        let peak_atten = (cfg.lw_h1_intrinsic_hz / lw_eff_hz) * (-two_tau_inept_s * kex).exp();
        let amp = i0 * peak_atten;
        if amp < (PickerConfig::default().snr_threshold as f64) * noise_hsqc {
            broadened_amides += 1;
        }
        let sig_h = hz_to_ppm(lw_eff_hz, h_mhz, 600.0) * fwhm_to_sigma;
        splat_2d_gaussian(&mut hsqc, h_ppm, n_ppm, sig_h, sig_n, amp);
    }

    for v in &mut hsqc.data {
        *v += rng.normal(0.0, noise_hsqc) as f32;
    }

    let picked_hsqc = pick_hsqc_2d(&hsqc, &PickerConfig::default());
    // Match picked CON roots to ground-truth peptide bonds within `(0.035 ppm C', 0.14 ppm N)`
    let mut matched_con = 0usize;
    for &(_, tcp, tnc, _, _) in &cubes.ground_truth_con {
        if con_peaks
            .iter()
            .any(|p| (p.c_prev_ppm - tcp).abs() <= 0.04 && (p.n_curr_ppm - tnc).abs() <= 0.16)
        {
            matched_con += 1;
        }
    }

    let denom = total_bonds.max(1) as f64;
    H1ExchangeComparison {
        proton_mhz: cfg.proton_mhz,
        temperature_k: cfg.temperature_k,
        ph: cfg.ph,
        total_peptide_bonds: total_bonds,
        proline_peptide_bonds: pro_bonds,
        amide_bonds_broadened_below_snr: broadened_amides,
        hsqc_2d: hsqc,
        picked_hsqc_roots: picked_hsqc.len(),
        hsqc_bond_recovery_frac: (picked_hsqc.len().min(total_bonds) as f64) / denom,
        picked_con_roots: con_peaks.len(),
        con_bond_recovery_frac: (matched_con as f64) / denom,
        per_residue_kex_s_inv: per_residue_kex,
    }
}
