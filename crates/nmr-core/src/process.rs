//! Deterministic multi-dimensional NMR time-domain (FID) signal processing and
//! non-uniform sampling (NUS) reconstruction (`Spectrum2D` / `Spectrum3D`).
//!
//! Provenance: Clean-room original implementation of `docs/specs/11-fid-processing.md` section 2
//! strictly from peer-reviewed literature:
//! - Cooley, J. W. & Tukey, J. W. (*Math. Comput.* **1965**, 19, 297–301);
//!   Bluestein, L. (*IEEE Trans. Audio Electroacoust.* **1970**, 18, 451–455) —
//!   radix-2 and arbitrary-length chirp-z complex `f64` discrete Fourier transforms.
//! - Bartholdi, E. & Ernst, R. R. (*J. Magn. Reson.* **1973**, 11, 9–19) —
//!   zero-filling information preservation prior to imaginary discard.
//! - Otting, G., Widmer, H., Wagner, G. & Wüthrich, K. (*J. Magn. Reson.* **1986**, 66, 187–193) —
//!   first-point scaling (`FCOR = 0.5` when $|p_1| < 1^\circ$) to suppress DFT baseline offset.
//! - Marion, D., Ikura, M. & Bax, A. (*J. Magn. Reson.* **1989**, 84, 425–430) —
//!   time-domain Gaussian low-pass convolution subtraction of residual water resonance.
//! - Zhu, G. & Bax, A. (*J. Magn. Reson.* **1990**, 90, 405–410) —
//!   mirror-image linear prediction (`MILP`) for known-phase indirect interferograms.
//! - Hoch, J. C. & Stern, A. S. (*NMR Data Processing*, Wiley-Liss, **1996**) —
//!   sine-bell / exponential apodization, Nyquist ordering, and polynomial baseline correction.
//! - Moskau, D. (*Concepts Magn. Reson.* **2002**, 15, 164–176) —
//!   digital oversampling FIR group-delay (`GRPDLY`) integer shift and sub-sample phase alignment.
//! - Chen, L., Weng, Z., Goh, L. & Garland, M. (*J. Magn. Reson.* **2002**, 158, 164–168) —
//!   first-derivative Shannon entropy minimization with scale-invariant negative-intensity penalty
//!   for automatic phase correction.
//! - Stern, A. S., Donoho, D. L. & Hoch, J. C. (*J. Magn. Reson.* **2007**, 188, 295–300);
//!   Beck, A. & Teboulle, M. (*SIAM J. Imaging Sci.* **2009**, 2, 183–202);
//!   Hyberts, S. G., Milbradt, A. G., Wagner, A. B., Arthanari, H. & Wagner, G.
//!   (*J. Biomol. NMR* **2012**, 52, 315–327) — 1D complex and 2D hypercomplex (bicomplex)
//!   Iterative Soft Thresholding (`IST` / `FISTA`) reconstruction of non-uniformly sampled FIDs.

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fid::{
    read_fid_2d, read_fid_3d, Complex64, Fid1D, Fid2D, Fid3D, FidAxisMeta, FidVendor, QuadMode,
    StoredWindowHint,
};
use crate::ops::{
    auto_phase_1d, auto_phase_direct_nd, auto_phase_indirect_nd, hilbert_transform_in_place,
    lock_all_positive_polarity_3d, lock_biphasic_ca_polarity_3d, AutoPhaseConfig, PolarityMode,
};
pub use crate::pulprog::{
    apply_indirect_axis_transform_3d, evaluate_3d_axis_hypotheses_against_hsqc,
    evaluate_3d_axis_hypotheses_against_roots, extract_hsqc_2d_roots_for_verification,
    verify_3d_axis_hypotheses_against_hsqc, verify_3d_axis_hypotheses_against_roots,
    AxisHypothesisScore, AxisHypothesisVerification, AxisTransformOp, NamedAxisHypothesis,
    MIN_AXIS_RESCUE_LN_BF,
};
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

// ============================================================================
// Serializable Processing Recipe (`processing_recipe.json`)
// ============================================================================

/// Time-domain apodization window applied over active complex points `0..n_apod`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ApodizationWindow {
    None,
    Exponential { lb_hz: f64 },
    SineBell { offset_pi_frac: f64 },
    SquaredSineBell { offset_pi_frac: f64 },
}

impl From<StoredWindowHint> for ApodizationWindow {
    fn from(h: StoredWindowHint) -> Self {
        match h {
            StoredWindowHint::None => Self::None,
            StoredWindowHint::Exponential { lb_hz } => Self::Exponential { lb_hz },
            StoredWindowHint::SineBell { offset_pi_frac } => Self::SineBell { offset_pi_frac },
            StoredWindowHint::SquaredSineBell { offset_pi_frac } => {
                Self::SquaredSineBell { offset_pi_frac }
            }
        }
    }
}

/// Time-domain Gaussian low-pass solvent filter (Marion, Ikura & Bax 1989).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolventFilterConfig {
    pub enabled: bool,
    /// Gaussian kernel standard deviation in complex time-domain points (default `32`).
    pub half_width_points: usize,
}

impl Default for SolventFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            half_width_points: 32,
        }
    }
}

/// Frequency-domain iterative MAD-masked Legendre polynomial baseline correction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub enabled: bool,
    /// Orthogonal Legendre polynomial degree (`0..=5`, default `1`).
    pub poly_order: usize,
    /// Noise-masking threshold in MAD sigma units (default `3.0`).
    pub mad_sigma: f64,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poly_order: 1,
            mad_sigma: 3.0,
        }
    }
}

/// Provenance of the frequency-domain phase parameters `(p0_deg, p1_deg)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseSource {
    /// Explicitly supplied by caller / recipe JSON override.
    Explicit,
    /// Adopted from stored spectrometer processing metadata (`procs` / `procpar`).
    Stored,
    /// Multidimensional blind automatic phase correction via `ops.rs` (`auto_phase_1d`,
    /// `auto_phase_direct_nd`, `auto_phase_indirect_nd`, with `PolarityMode::Biphasic` on `HNCACB`).
    Auto,
    /// Solved automatically via Chen et al. (2002) entropy minimization when no phases were stored.
    AutoChen2002,
    /// Default `(0.0, 0.0)` on indirect axes when no phases were stored.
    DefaultZero,
}

/// Per-dimension processing parameters executed strictly in `docs/specs/11-fid-processing.md` §2.1 order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DimRecipe {
    pub label: String,
    /// Step 1: Remove causal digital oversampling group delay (`group_delay` complex points).
    pub remove_digital_filter: bool,
    pub group_delay: f64,
    /// Step 2: Optional time-domain Gaussian low-pass solvent subtraction (direct `1H` axis).
    pub solvent_filter: Option<SolventFilterConfig>,
    /// Active complex time-domain points before zero-filling (`None` = all valid points).
    /// Clamped to `<= axis.complex_points` whenever `linear_prediction == false`.
    pub td_eff: Option<usize>,
    /// Step 3b: Optional indirect Mirror-Image Linear Prediction (`false` by default).
    pub linear_prediction: bool,
    /// Target complex points after MILP when `linear_prediction == true` (`None` = `2 * active_td`).
    pub lp_target_points: Option<usize>,
    /// Autoregressive Burg order for MILP (`None` = `min(16, active_td / 3)`).
    pub lp_order: Option<usize>,
    /// Step 3a: Apodization window function.
    pub window: ApodizationWindow,
    /// Step 3c: First-point scaling (`0.5` when `|p1| < 1 deg` per Otting et al. 1986, else `1.0`).
    pub fcor: f64,
    /// Step 4: Complex FFT size after zero-filling.
    pub si: usize,
    /// Step 5a: Conjugate time signal / positive-exponent FT convention (Bruker vs Varian).
    pub conjugate_for_fft: bool,
    /// Step 5b: Reverse frequency axis after FFT (`##$REVERSE= yes` in Bruker `procs`).
    pub reverse: bool,
    /// Optional causal Hilbert imaginary-channel reconstruction prior to phase rotation (`--hilbert`).
    #[serde(default)]
    pub hilbert: bool,
    /// Step 6: Zero-order phase in degrees (`None` when `phase_source` is `Auto` / `AutoChen2002`).
    pub p0_deg: Option<f64>,
    /// Step 6: First-order phase in degrees across `[0, N_fft)` (`None` when `phase_source` is `Auto` / `AutoChen2002`).
    pub p1_deg: Option<f64>,
    pub phase_source: PhaseSource,
    /// Step 8: Optional polynomial baseline correction after imaginary discard.
    pub baseline: Option<BaselineConfig>,
    /// Step 9: Optional region-of-interest `[min_ppm, max_ppm]` crop (`Some([5.5, 11.5])` by default on direct `1H`).
    pub roi_ppm: Option<[f64; 2]>,
}

/// Clean-room FISTA / IST configuration for 1D and 2D hypercomplex NUS reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NusIstConfig {
    /// Number of FISTA / IST soft-thresholding iterations (default `40`).
    pub iterations: usize,
    /// Starting threshold as a fraction of maximum dirty-spectrum magnitude (default `0.85`).
    pub initial_thresh_frac: f64,
    /// Final threshold as a multiple of frequency-domain MAD noise (default `2.5`).
    pub final_noise_sigma: f64,
    /// Use Nesterov momentum extrapolation (`FISTA`, default `true`).
    pub fista_momentum: bool,
}

impl Default for NusIstConfig {
    fn default() -> Self {
        Self {
            iterations: 40,
            initial_thresh_frac: 0.85,
            final_noise_sigma: 2.5,
            fista_momentum: true,
        }
    }
}

/// Complete deterministic processing recipe for 1D, 2D, or 3D FIDs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessingRecipe {
    /// Ordered per-dimension recipes: `[dim_x]` (1D), `[dim_x, dim_y]` (2D), or `[dim_x, dim_y, dim_z]` (3D).
    pub dims: Vec<DimRecipe>,
    /// NUS IST/FISTA reconstruction parameters (populated when `nus_mask.is_some()`, or used as fallback if NUS).
    pub nus: Option<NusIstConfig>,
}

impl DimRecipe {
    /// Build default `DimRecipe` for dimension `dim_idx` (`0` = direct, `> 0` = indirect) from `FidAxisMeta`.
    pub fn from_axis_meta(axis: &FidAxisMeta, dim_idx: usize) -> Self {
        let is_direct = dim_idx == 0;
        let is_direct_proton = is_direct && axis.label.to_ascii_uppercase().contains('H');
        let sp = &axis.stored_proc;

        let remove_digital_filter = is_direct && axis.group_delay > 0.0;
        let group_delay = if is_direct {
            axis.group_delay.max(0.0)
        } else {
            0.0
        };

        // Clamp td_eff to <= axis.complex_points because linear_prediction defaults to false
        // (TopSpin writes TDeff = 2 * TD when LP was active in proc2s/proc3s).
        let td_eff = sp
            .td_eff
            .map(|t| t.min(axis.complex_points).max(1))
            .filter(|&t| t < axis.complex_points);

        let n_eff = td_eff.unwrap_or(axis.complex_points).max(1);
        let min_zf = if is_direct_proton && n_eff >= 512 {
            (2 * n_eff).max(2048).next_power_of_two()
        } else {
            (2 * n_eff).next_power_of_two()
        };
        let si = sp
            .si
            .map(|s| s.max(min_zf).next_power_of_two())
            .unwrap_or(min_zf);

        let window =
            sp.window
                .map(ApodizationWindow::from)
                .unwrap_or(ApodizationWindow::SquaredSineBell {
                    offset_pi_frac: 0.5,
                });

        let (p0_deg, p1_deg, phase_source) = match (sp.p0_deg, sp.p1_deg) {
            (Some(p0), Some(p1)) => (Some(p0), Some(p1), PhaseSource::Stored),
            (Some(p0), None) => (Some(p0), Some(0.0), PhaseSource::Stored),
            (None, Some(p1)) => (Some(0.0), Some(p1), PhaseSource::Stored),
            (None, None) => {
                if is_direct {
                    (None, None, PhaseSource::AutoChen2002)
                } else {
                    (Some(0.0), Some(0.0), PhaseSource::DefaultZero)
                }
            }
        };

        let p1_for_fcor = p1_deg.unwrap_or(0.0);
        let fcor = sp
            .fcor
            .unwrap_or(if p1_for_fcor.abs() < 1.0 { 0.5 } else { 1.0 });

        // Solvent filter: enable automatically on direct 1H when Bruker BC_mod == 3..=6
        // (`spol`/`qpol`/`sfil`/`qfil` pre-FT carrier/water suppression) OR when no stored BC_mod
        // metadata is present (raw un-processed FID).
        let solvent_filter = if is_direct_proton {
            let enable_sf = match sp.bc_mod {
                Some(3..=6) => true,
                Some(_) => false,
                None => true,
            };
            let half_width_points = if let Some(bcfw) = sp.bcfw_ppm.filter(|&w| w > 0.0) {
                // Convert BCFW (full notch width in ppm) to Gaussian time-domain sigma in points:
                // sigma_f ~ (bcfw / 4) * obs_mhz / sw_hz => sigma_t = 1 / (2 * pi * sigma_f),
                // clamped to [16, 64] so amides >= 5.5 ppm suffer zero attenuation.
                let sw_ppm = (axis.sw_hz / axis.obs_mhz.max(1e-6)).max(1.0);
                let sigma_f = (0.25 * bcfw / sw_ppm).max(1e-4);
                let sigma_t = (1.0 / (2.0 * PI * sigma_f)).round() as usize;
                sigma_t.clamp(16, 64)
            } else {
                32
            };
            Some(SolventFilterConfig {
                enabled: enable_sf,
                half_width_points,
            })
        } else {
            None
        };

        // Baseline correction: enable on direct 1H according to BC_mod (order 3 for spol/qpol BC_mod=3/4,
        // order 1 for linear/qfil/default), disabled on indirect dimensions unless BC_mod > 0.
        let baseline = if is_direct_proton {
            match sp.bc_mod {
                Some(0) => Some(BaselineConfig {
                    enabled: false,
                    poly_order: 1,
                    mad_sigma: 3.0,
                }),
                _ => Some(BaselineConfig {
                    enabled: true,
                    poly_order: 1,
                    mad_sigma: 3.0,
                }),
            }
        } else {
            match sp.bc_mod {
                Some(m) if m > 0 => Some(BaselineConfig {
                    enabled: true,
                    poly_order: if m >= 3 { 2 } else { 1 },
                    mad_sigma: 3.0,
                }),
                _ => None,
            }
        };

        // Default direct 1H ROI crop: 5.5..=11.5 ppm (if axis spans that region).
        let axis_sw_ppm = axis.sw_hz / axis.obs_mhz.max(1e-6);
        let axis_low_ppm = axis.first_ppm - axis_sw_ppm;
        let roi_ppm = if is_direct_proton && axis.first_ppm > 6.0 && axis_low_ppm < 11.0 {
            Some([5.5, 11.5])
        } else {
            None
        };

        // With descending-ppm Nyquist mapping `m = (N/2 + N - j) % N` and `-i` forward DFT:
        // - Bruker (`axis.conjugate_for_fft == true` in `fid.rs`, including Rance-Kay `MC2 = 5`
        //   `EchoAntiecho` recombined in `fid.rs`) downfield resonances oscillate as `exp(+i 2pi f n)`
        //   and require `conjugate_for_fft = false`.
        // - Varian (`axis.conjugate_for_fft == false` in `fid.rs`) downfield resonances oscillate
        //   as `exp(-i 2pi f n)` and require `conjugate_for_fft = true`.
        let conjugate_for_fft = !axis.conjugate_for_fft;

        Self {
            label: axis.label.clone(),
            remove_digital_filter,
            group_delay,
            solvent_filter,
            td_eff,
            linear_prediction: false,
            lp_target_points: None,
            lp_order: None,
            window,
            fcor,
            si,
            conjugate_for_fft,
            reverse: sp.reverse,
            hilbert: false,
            p0_deg,
            p1_deg,
            phase_source,
            baseline,
            roi_ppm,
        }
    }

    /// Explicitly override `(p0_deg, p1_deg)` on this dimension (adjusting `fcor` per Otting et al. 1986).
    pub fn set_explicit_phase(&mut self, p0_deg: f64, p1_deg: f64) {
        self.p0_deg = Some(p0_deg);
        self.p1_deg = Some(p1_deg);
        self.phase_source = PhaseSource::Explicit;
        self.fcor = if p1_deg.abs() < 1.0 { 0.5 } else { 1.0 };
    }

    /// Switch this dimension to blind multidimensional automatic phase correction (`PhaseSource::Auto`).
    pub fn set_auto_phase(&mut self) {
        self.p0_deg = None;
        self.p1_deg = None;
        self.phase_source = PhaseSource::Auto;
        self.fcor = 0.5;
    }
}

impl ProcessingRecipe {
    pub fn from_fid_1d(fid: &Fid1D) -> Self {
        Self {
            dims: vec![DimRecipe::from_axis_meta(&fid.axis_x, 0)],
            nus: None,
        }
    }

    pub fn from_fid_2d(fid: &Fid2D) -> Self {
        let mut rec = Self {
            dims: vec![
                DimRecipe::from_axis_meta(&fid.axis_x, 0),
                DimRecipe::from_axis_meta(&fid.axis_y, 1),
            ],
            nus: fid.nus_mask.as_ref().map(|_| NusIstConfig::default()),
        };
        if fid.vendor == FidVendor::Varian {
            // Varian `.fid/procpar` stores VnmrJ 1D interactive scout cursor `rp*`/`lp*` rather
            // than calibrated 2D spectral phases; solve both dimensions via blind autophasing.
            rec.set_auto_phase();
        } else {
            for d in &mut rec.dims {
                if matches!(
                    d.phase_source,
                    PhaseSource::AutoChen2002 | PhaseSource::DefaultZero
                ) {
                    d.set_auto_phase();
                }
            }
        }
        rec
    }

    pub fn from_fid_3d(fid: &Fid3D) -> Self {
        let mut rec = Self {
            dims: vec![
                DimRecipe::from_axis_meta(&fid.axis_x, 0),
                DimRecipe::from_axis_meta(&fid.axis_y, 1),
                DimRecipe::from_axis_meta(&fid.axis_z, 2),
            ],
            nus: fid.nus_mask.as_ref().map(|_| NusIstConfig::default()),
        };
        if fid.vendor == FidVendor::Varian {
            rec.set_auto_phase();
        } else {
            for d in &mut rec.dims {
                if matches!(
                    d.phase_source,
                    PhaseSource::AutoChen2002 | PhaseSource::DefaultZero
                ) {
                    d.set_auto_phase();
                }
            }
        }
        rec
    }

    /// Force blind multidimensional automatic phase correction across all dimensions (`--auto-phase`).
    pub fn set_auto_phase(&mut self) {
        for d in &mut self.dims {
            d.set_auto_phase();
        }
    }

    /// Builder helper to force blind multidimensional automatic phase correction (`--auto-phase`).
    pub fn with_auto_phase(mut self) -> Self {
        self.set_auto_phase();
        self
    }

    /// Enable or disable causal Hilbert imaginary-channel reconstruction (`--hilbert`) across dimensions.
    pub fn set_hilbert(&mut self, enabled: bool) {
        for d in &mut self.dims {
            d.hilbert = enabled;
        }
    }

    /// Configure NUS reconstruction method (`fista_momentum = true` for `FISTA`, `false` for `IST`).
    pub fn set_nus_method_fista(&mut self, fista_momentum: bool) {
        self.nus
            .get_or_insert_with(NusIstConfig::default)
            .fista_momentum = fista_momentum;
    }

    /// Disable ROI cropping across all dimensions (`--no-crop`).
    pub fn set_no_crop(&mut self) {
        for d in &mut self.dims {
            d.roi_ppm = None;
        }
    }

    /// Builder helper to disable or customize direct-axis ROI cropping.
    pub fn with_crop_h(mut self, roi_ppm: Option<[f64; 2]>) -> Self {
        if let Some(d0) = self.dims.first_mut() {
            d0.roi_ppm = roi_ppm;
        }
        self
    }

    /// Enable or disable indirect Mirror-Image Linear Prediction (`--lp`) on indirect dimensions.
    pub fn set_linear_prediction(&mut self, enabled: bool) {
        for d in self.dims.iter_mut().skip(1) {
            d.linear_prediction = enabled;
        }
    }

    /// Builder helper to toggle indirect Mirror-Image Linear Prediction.
    pub fn with_linear_prediction(mut self, enabled: bool) -> Self {
        self.set_linear_prediction(enabled);
        self
    }

    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize ProcessingRecipe: {e}"))
    }

    pub fn from_json_str(json: &str) -> Result<Self, String> {
        serde_json::from_str(json)
            .map_err(|e| format!("failed to parse ProcessingRecipe JSON: {e}"))
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
            }
        }
        let txt = self.to_json_pretty()?;
        fs::write(p, txt).map_err(|e| format!("failed to write {}: {e}", p.display()))
    }

    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let txt =
            fs::read_to_string(p).map_err(|e| format!("failed to read {}: {e}", p.display()))?;
        Self::from_json_str(&txt)
    }
}

// ============================================================================
// Clean-Room Radix-2 Cooley-Tukey + Bluestein Chirp-Z Complex f64 FFT
// ============================================================================

/// In-place complex `f64` discrete Fourier transform of arbitrary length `n`:
/// - `inverse == false`: $X[k] = \sum_{m=0}^{n-1} x[m] e^{-i 2\pi k m / n}$
/// - `inverse == true`:  $x[m] = \frac{1}{n} \sum_{k=0}^{n-1} X[k] e^{+i 2\pi k m / n}$
pub fn fft_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        fft_radix2_in_place(buf, inverse);
    } else {
        fft_bluestein_in_place(buf, inverse);
    }
}

fn fft_radix2_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let theta = sign * 2.0 * PI / (len as f64);
        // Use exact sin/cos half-angle recurrence for numerical accuracy across large stages
        let s_half = (0.5 * theta).sin();
        let wp_re = -2.0 * s_half * s_half;
        let wp_im = theta.sin();

        for base in (0..n).step_by(len) {
            let mut w_re = 1.0_f64;
            let mut w_im = 0.0_f64;
            for k in 0..half {
                let u = buf[base + k];
                let v = buf[base + k + half];
                let t_re = w_re * v[0] - w_im * v[1];
                let t_im = w_re * v[1] + w_im * v[0];
                buf[base + k] = [u[0] + t_re, u[1] + t_im];
                buf[base + k + half] = [u[0] - t_re, u[1] - t_im];

                let next_re = w_re + (w_re * wp_re - w_im * wp_im);
                let next_im = w_im + (w_im * wp_re + w_re * wp_im);
                w_re = next_re;
                w_im = next_im;
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for c in buf.iter_mut() {
            c[0] *= inv_n;
            c[1] *= inv_n;
        }
    }
}

fn fft_bluestein_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };
    let two_n = (2 * n) as u128;

    let mut chirp = vec![[0.0, 0.0]; n];
    for (k, ch) in chirp.iter_mut().enumerate() {
        let rem = ((k as u128) * (k as u128)) % two_n;
        let angle = sign * PI * (rem as f64) / (n as f64);
        let (s, c) = angle.sin_cos();
        *ch = [c, s];
    }

    let mut a = vec![[0.0, 0.0]; m];
    let mut b = vec![[0.0, 0.0]; m];
    for k in 0..n {
        let x = buf[k];
        let ch = chirp[k];
        // a[k] = x[k] * chirp[k]
        a[k] = [x[0] * ch[0] - x[1] * ch[1], x[0] * ch[1] + x[1] * ch[0]];
    }
    b[0] = [chirp[0][0], -chirp[0][1]];
    for k in 1..n {
        let ch_conj = [chirp[k][0], -chirp[k][1]];
        b[k] = ch_conj;
        b[m - k] = ch_conj;
    }

    fft_radix2_in_place(&mut a, false);
    fft_radix2_in_place(&mut b, false);
    for i in 0..m {
        let ar = a[i][0];
        let ai = a[i][1];
        let br = b[i][0];
        let bi = b[i][1];
        a[i] = [ar * br - ai * bi, ar * bi + ai * br];
    }
    fft_radix2_in_place(&mut a, true);

    let scale = if inverse { 1.0 / (n as f64) } else { 1.0 };
    for k in 0..n {
        let conv = a[k];
        let ch = chirp[k];
        buf[k] = [
            (conv[0] * ch[0] - conv[1] * ch[1]) * scale,
            (conv[0] * ch[1] + conv[1] * ch[0]) * scale,
        ];
    }
}

// ============================================================================
// Per-Dimension Building Blocks (Steps 1–9)
// ============================================================================

/// Step 1: Left-shift complex trace by `round(group_delay)` samples and return `(valid_points, dg_residual)`.
/// `dg_residual = group_delay - round(group_delay) in [-0.5, 0.5]`.
fn apply_digital_filter_shift(trace: &mut [Complex64], group_delay: f64) -> (usize, f64) {
    let len = trace.len();
    if group_delay <= 0.0 || len == 0 {
        return (len, 0.0);
    }
    let m_signed = group_delay.round();
    let dg = group_delay - m_signed;
    let m = (m_signed.max(0.0) as usize).min(len);
    let valid = len - m;
    if m > 0 {
        trace.copy_within(m..len, 0);
        trace[valid..len].fill([0.0, 0.0]);
    }
    (valid, dg)
}

/// Step 2: Marion, Ikura & Bax (*JMR* 1989) time-domain low-frequency Gaussian convolution subtraction.
fn apply_solvent_filter(trace: &mut [Complex64], valid_len: usize, half_width_points: usize) {
    if valid_len < 4 || half_width_points == 0 {
        return;
    }
    let sigma = half_width_points as f64;
    let radius = (3 * half_width_points).min(valid_len - 1);
    let mut weights = Vec::with_capacity(radius + 1);
    let inv_two_sigma2 = 0.5 / (sigma * sigma);
    for d in 0..=radius {
        weights.push((-(d as f64) * (d as f64) * inv_two_sigma2).exp());
    }

    let orig = trace[..valid_len].to_vec();
    let last = (valid_len - 1) as isize;
    for n in 0..valid_len {
        let mut sum_re = weights[0] * orig[n][0];
        let mut sum_im = weights[0] * orig[n][1];
        let mut w_tot = weights[0];
        let ni = n as isize;
        for (d, &w) in weights.iter().enumerate().skip(1) {
            let d_i = d as isize;
            // Left reflected index around 0
            let idx_l = (ni - d_i).unsigned_abs().min(valid_len - 1);
            // Right reflected index around valid_len - 1
            let raw_r = ni + d_i;
            let idx_r = if raw_r <= last {
                raw_r as usize
            } else {
                (2 * last - raw_r).max(0) as usize
            };
            sum_re += w * (orig[idx_l][0] + orig[idx_r][0]);
            sum_im += w * (orig[idx_l][1] + orig[idx_r][1]);
            w_tot += 2.0 * w;
        }
        let inv_w = 1.0 / w_tot;
        trace[n][0] = orig[n][0] - sum_re * inv_w;
        trace[n][1] = orig[n][1] - sum_im * inv_w;
    }
}

/// Step 3b: Mirror-Image Linear Prediction (`MILP`, Zhu & Bax *J. Magn. Reson.* 1990, 90, 405–410)
/// via complex Burg lattice reflection on the Hermitian-extended interferogram `[-N+1 .. N-1]`,
/// guaranteeing all characteristic roots lie strictly inside the unit circle (`|rho_m| < 1`).
fn apply_mirror_image_lp(
    trace: &[Complex64],
    active_td: usize,
    target_td: usize,
    lp_order: Option<usize>,
    known_p0_deg: f64,
) -> Vec<Complex64> {
    let n = active_td.min(trace.len());
    if target_td <= n || n < 6 {
        let mut out = trace[..n].to_vec();
        out.resize(target_td.max(n), [0.0, 0.0]);
        return out;
    }

    let order = lp_order
        .unwrap_or_else(|| (n / 3).clamp(2, 16))
        .min(n - 1)
        .max(1);

    // De-rotate by known t=0 phase so mirror symmetry s[-k] = conj(s[k]) holds around k = 0
    let theta = -known_p0_deg.to_radians();
    let (st, ct) = theta.sin_cos();
    let mut x = Vec::with_capacity(n);
    for &c in &trace[..n] {
        x.push([c[0] * ct - c[1] * st, c[0] * st + c[1] * ct]);
    }

    // Build Hermitian mirror-image sequence of length 2*n - 1
    let m_len = 2 * n - 1;
    let mut f_err = Vec::with_capacity(m_len);
    for k in (1..n).rev() {
        f_err.push([x[k][0], -x[k][1]]);
    }
    for &v in &x {
        f_err.push(v);
    }
    let mut b_err = f_err.clone();

    // Burg complex lattice recursion: x[k] = sum_{m=1..order} a[m] * x[k - m]
    let mut a = vec![[0.0, 0.0]; order + 1];
    let mut a_prev = vec![[0.0, 0.0]; order + 1];

    for m in 1..=order {
        let mut num_re = 0.0_f64;
        let mut num_im = 0.0_f64;
        let mut den = 0.0_f64;
        for k in m..m_len {
            let fk = f_err[k];
            let bk = b_err[k - 1];
            // 2 * fk * conj(bk)
            num_re += 2.0 * (fk[0] * bk[0] + fk[1] * bk[1]);
            num_im += 2.0 * (fk[1] * bk[0] - fk[0] * bk[1]);
            den += fk[0] * fk[0] + fk[1] * fk[1] + bk[0] * bk[0] + bk[1] * bk[1];
        }
        if den <= 1e-30 {
            break;
        }
        let rho = [num_re / den, num_im / den];
        a_prev[1..m].copy_from_slice(&a[1..m]);
        a[m] = rho;
        for i in 1..m {
            let ap_conj = [a_prev[m - i][0], -a_prev[m - i][1]];
            a[i] = [
                a_prev[i][0] - (rho[0] * ap_conj[0] - rho[1] * ap_conj[1]),
                a_prev[i][1] - (rho[0] * ap_conj[1] + rho[1] * ap_conj[0]),
            ];
        }
        for k in (m..m_len).rev() {
            let fk = f_err[k];
            let bk = b_err[k - 1];
            f_err[k] = [
                fk[0] - (rho[0] * bk[0] - rho[1] * bk[1]),
                fk[1] - (rho[0] * bk[1] + rho[1] * bk[0]),
            ];
            b_err[k] = [
                bk[0] - (rho[0] * fk[0] + rho[1] * fk[1]),
                bk[1] - (rho[0] * fk[1] - rho[1] * fk[0]),
            ];
        }
    }

    x.reserve(target_td - n);
    for k in n..target_td {
        let mut pred_re = 0.0_f64;
        let mut pred_im = 0.0_f64;
        for m in 1..=order {
            let prev = x[k - m];
            let am = a[m];
            pred_re += am[0] * prev[0] - am[1] * prev[1];
            pred_im += am[0] * prev[1] + am[1] * prev[0];
        }
        x.push([pred_re, pred_im]);
    }

    // Re-rotate back to original t=0 phase
    let mut out = Vec::with_capacity(target_td);
    for c in x {
        out.push([c[0] * ct + c[1] * st, -c[0] * st + c[1] * ct]);
    }
    out
}

/// Precompute apodization weights of length `n_apod` (including `fcor` on index `0`).
fn build_apodization_weights(
    window: ApodizationWindow,
    n_apod: usize,
    sw_hz: f64,
    fcor: f64,
) -> Vec<f64> {
    if n_apod == 0 {
        return Vec::new();
    }
    let n_f = n_apod as f64;
    let sw = sw_hz.max(1e-6);
    let mut w = Vec::with_capacity(n_apod);
    match window {
        ApodizationWindow::None => w.resize(n_apod, 1.0),
        ApodizationWindow::Exponential { lb_hz } => {
            let decay = -PI * lb_hz / sw;
            for n in 0..n_apod {
                w.push((decay * (n as f64)).exp());
            }
        }
        ApodizationWindow::SineBell { offset_pi_frac } => {
            let alpha = offset_pi_frac.clamp(0.0, 0.99);
            for n in 0..n_apod {
                let arg = PI * (alpha + (1.0 - alpha) * (n as f64) / n_f);
                w.push(arg.sin());
            }
        }
        ApodizationWindow::SquaredSineBell { offset_pi_frac } => {
            let alpha = offset_pi_frac.clamp(0.0, 0.99);
            for n in 0..n_apod {
                let s = (PI * (alpha + (1.0 - alpha) * (n as f64) / n_f)).sin();
                w.push(s * s);
            }
        }
    }
    w[0] *= fcor;
    w
}

/// Step 5: Execute zero-filled FFT, Nyquist shift into descending-ppm `AxisPpm` order (`0 = high_ppm`,
/// `N/2 = carrier`), sub-sample digital filter phase alignment (`dg_residual`), and optional `reverse`.
fn transform_to_ordered_spectrum(
    time_buf: &[Complex64],
    si: usize,
    conjugate_for_fft: bool,
    dg_residual: f64,
    reverse: bool,
    work: &mut Vec<Complex64>,
    out_spec: &mut Vec<Complex64>,
) {
    let n = si.max(time_buf.len()).max(2);
    work.clear();
    work.resize(n, [0.0, 0.0]);
    let copy_len = time_buf.len().min(n);

    if conjugate_for_fft {
        for i in 0..copy_len {
            work[i] = [time_buf[i][0], -time_buf[i][1]];
        }
    } else {
        work[..copy_len].copy_from_slice(&time_buf[..copy_len]);
    }

    fft_in_place(work, false);

    // Map standard DFT bins `m in 0..n` into descending-ppm AxisPpm bins `j in 0..n`:
    // - `j = 0`   -> Nyquist `+SW/2` (`m = n / 2`)
    // - `j = n/2` -> carrier `0 Hz`  (`m = 0`)
    // - `j = n/2 - k` -> positive offset `+k * SW / n` (`m = k`)
    // - `j = n/2 + k` -> negative offset `-k * SW / n` (`m = n - k`)
    // Thus `m = (n / 2 + n - j) % n`!
    let half = n / 2;
    out_spec.clear();
    out_spec.resize(n, [0.0, 0.0]);
    for (j, dst) in out_spec.iter_mut().enumerate() {
        let m = (half + n - (j % n)) % n;
        *dst = work[m];
    }

    // Apply sub-sample digital filter residual `dg_residual = group_delay - round(group_delay)`
    // pivoted at the carrier `j = n / 2` so `t = 0` aligns with sample 0 without perturbing DC phase:
    // For `conjugate_for_fft`, `work[m] = sum conj(z'[n]) exp(-i 2pi m n / n)`:
    // If a downfield peak at `+f_j` (`f_j = (half - j) / n`) had `conj(z[n]) = A* exp(+i 2pi f_j (n - G))`,
    // then `conj(z'[n]) = A* exp(+i 2pi f_j (n - dg))` => DFT at `m = +f_j * n` carries `exp(-i 2pi f_j dg)`,
    // which is cancelled by multiplying by `exp(+i 2pi f_j dg)`.
    if dg_residual.abs() > 1e-12 {
        let n_f = n as f64;
        let half_f = half as f64;
        for (j, val) in out_spec.iter_mut().enumerate() {
            let f_j = (half_f - (j as f64)) / n_f;
            let phi = 2.0 * PI * dg_residual * f_j;
            let (sp, cp) = phi.sin_cos();
            let (re, im) = (val[0], val[1]);
            *val = [re * cp - im * sp, re * sp + im * cp];
        }
    }

    // Step 5b: `##$REVERSE= yes` in Bruker `procs`/`proc2s`/`proc3s` reflects frequency bins
    // around the carrier `j = n / 2` (`j <-> (n - j) % n`), preserving `j = 0` (Nyquist) and
    // `j = n / 2` (DC / carrier) without introducing a 1-bin Nyquist offset.
    if reverse && n > 2 {
        out_spec[1..n].reverse();
    }
}

/// Step 6 & 7: Apply `(p0_deg, p1_deg)` phase rotation (`Re(S[k] * exp(-i (p0 + p1 * k / N)))`,
/// matching Bruker TopSpin `PHC0`/`PHC1` sign convention) across `0..n` and discard imaginary part.
#[inline]
fn apply_phase_and_discard_imag(
    spec: &[Complex64],
    p0_deg: f64,
    p1_deg: f64,
    out_real: &mut [f64],
) {
    let n = spec.len();
    let n_f = n.max(1) as f64;
    let p0_rad = p0_deg.to_radians();
    let p1_step_rad = p1_deg.to_radians() / n_f;
    for k in 0..n {
        let phi = p0_rad + (k as f64) * p1_step_rad;
        let (sp, cp) = phi.sin_cos();
        let (re, im) = (spec[k][0], spec[k][1]);
        out_real[k] = re * cp + im * sp;
    }
}

/// Step 6c: Automatic phase correction via Chen et al. (*J. Magn. Reson.* **2002**, 158, 164–168):
/// Minimizes first-derivative Shannon entropy plus scale-invariant negative-intensity penalty
/// over `eval_slice`:
/// $$E(p_0, p_1) = -\sum_{k=0}^{M-2} h_k \ln h_k + \gamma \frac{\sum_{|R_k| > \tau} \max(0, -R_k)^2}{\sum_k R_k^2 + \epsilon}$$
pub fn auto_phase_chen2002(spec_slice: &[Complex64]) -> (f64, f64) {
    let m = spec_slice.len();
    if m < 8 {
        return (0.0, 0.0);
    }

    // Estimate thermal noise floor from complex first-difference MAD so smooth Lorentzian 1/f wings
    // do not inflate the threshold on high-SNR spectra while baseline noise is still ignored.
    let mut diff_mags: Vec<f64> = (0..(m - 1))
        .map(|k| {
            let dr = spec_slice[k + 1][0] - spec_slice[k][0];
            let di = spec_slice[k + 1][1] - spec_slice[k][1];
            (0.5 * (dr * dr + di * di)).sqrt()
        })
        .collect();
    let sigma_est = 1.4826 * median_in_place(&mut diff_mags);
    let noise_floor = 2.0 * sigma_est;
    let m_f = m as f64;

    // Pivot `(p_center, p1)` at the spectral energy centroid `u_c` to orthogonalize `p0` and `p1`.
    let mut w_sum = 0.0_f64;
    let mut ku_sum = 0.0_f64;
    for (k, c) in spec_slice.iter().enumerate() {
        let e = c[0] * c[0] + c[1] * c[1];
        w_sum += e;
        ku_sum += e * (k as f64) / m_f;
    }
    let u_c = if w_sum > 1e-30 { ku_sum / w_sum } else { 0.5 };

    let mut r_buf = vec![0.0_f64; m];
    let mut eval_obj = |pc_deg: f64, p1_deg: f64| -> f64 {
        let p0_rad = (pc_deg - p1_deg * u_c).to_radians();
        let p1_step = p1_deg.to_radians() / m_f;
        let mut tot_sq = 1e-30_f64;
        let mut neg_sq = 0.0_f64;
        for k in 0..m {
            let phi = p0_rad + (k as f64) * p1_step;
            let (sp, cp) = phi.sin_cos();
            let rk = spec_slice[k][0] * cp + spec_slice[k][1] * sp;
            r_buf[k] = rk;
            tot_sq += rk * rk;
            if rk < -noise_floor {
                let ex = -rk - noise_floor;
                neg_sq += ex * ex;
            }
        }

        let mut deriv_sum = 0.0_f64;
        for k in 0..(m - 1) {
            deriv_sum += (r_buf[k + 1] - r_buf[k]).abs();
        }
        if deriv_sum <= 1e-30 {
            return 1e9;
        }
        let inv_d = 1.0 / deriv_sum;
        let mut entropy = 0.0_f64;
        for k in 0..(m - 1) {
            let hk = (r_buf[k + 1] - r_buf[k]).abs() * inv_d;
            if hk > 1e-18 {
                entropy -= hk * hk.ln();
            }
        }

        let penalty = 2500.0 * (neg_sq / tot_sq);
        entropy + penalty
    };

    // Coarse scan over p_center in [-180, 180) (5 deg step) and p1 in [-90, 90] (15 deg step)
    let mut best_pc = 0.0_f64;
    let mut best_p1 = 0.0_f64;
    let mut best_e = f64::INFINITY;

    for i0 in -36..36 {
        let pc = (i0 as f64) * 5.0;
        for i1 in -6..=6 {
            let p1 = (i1 as f64) * 15.0;
            let e = eval_obj(pc, p1);
            if e < best_e {
                best_e = e;
                best_pc = pc;
                best_p1 = p1;
            }
        }
    }

    // Multi-scale coordinate pattern search refinement in orthogonalized (p_center, p1) coordinates
    for &(step0, step1) in &[(3.0, 8.0), (1.0, 2.5), (0.25, 0.6), (0.05, 0.1)] {
        let mut improved = true;
        let mut guard = 0usize;
        while improved && guard < 50 {
            improved = false;
            guard += 1;
            for &(d0, d1) in &[
                (step0, 0.0),
                (-step0, 0.0),
                (0.0, step1),
                (0.0, -step1),
                (step0, step1),
                (step0, -step1),
                (-step0, step1),
                (-step0, -step1),
            ] {
                let c0 = wrap_deg_180(best_pc + d0);
                let c1 = (best_p1 + d1).clamp(-180.0, 180.0);
                let e = eval_obj(c0, c1);
                if e + 1e-12 < best_e {
                    best_e = e;
                    best_pc = c0;
                    best_p1 = c1;
                    improved = true;
                }
            }
        }
    }

    let p0_deg = wrap_deg_180(best_pc - best_p1 * u_c);
    (p0_deg, best_p1)
}

#[inline]
fn wrap_deg_180(deg: f64) -> f64 {
    let mut d = (deg + 180.0) % 360.0;
    if d < 0.0 {
        d += 360.0;
    }
    d - 180.0
}

/// Map `(p0_slice, p1_slice)` solved on `spec[idx_start..=idx_end]` back to full-spectrum `(p0, p1)`
/// across `0..n_full`.
fn lift_slice_phase_to_full(
    p0_slice: f64,
    p1_slice: f64,
    idx_start: usize,
    slice_len: usize,
    n_full: usize,
) -> (f64, f64) {
    let p1_full = p1_slice * (n_full as f64) / (slice_len.max(1) as f64);
    let p0_full = p0_slice - p1_full * (idx_start as f64) / (n_full.max(1) as f64);
    (wrap_deg_180(p0_full), p1_full)
}

/// Step 8: Iterative MAD-masked orthogonal Legendre polynomial baseline subtraction (`order = 0..=5`).
pub fn subtract_polynomial_baseline(spec: &mut [f64], cfg: &BaselineConfig) {
    let n = spec.len();
    if !cfg.enabled || n < 8 {
        return;
    }
    let order = cfg.poly_order.min(5).min(n / 4);
    let d = order + 1;

    // Precompute Legendre basis P_0..P_order on [-1, 1]
    let mut basis = vec![[0.0_f64; 6]; n];
    let denom = (n - 1).max(1) as f64;
    for (k, row) in basis.iter_mut().enumerate() {
        let x = 2.0 * (k as f64) / denom - 1.0;
        row[0] = 1.0;
        if order >= 1 {
            row[1] = x;
        }
        for m in 1..order {
            row[m + 1] =
                ((2 * m + 1) as f64 * x * row[m] - (m as f64) * row[m - 1]) / ((m + 1) as f64);
        }
    }

    let mut baseline = vec![0.0_f64; n];
    let mut resid = spec.to_vec();
    let mut scratch = vec![0.0_f64; n];
    let mut mask = vec![true; n];
    let mad_mult = cfg.mad_sigma.max(1.5);

    for iter in 0..4 {
        if iter > 0 {
            for k in 0..n {
                resid[k] = spec[k] - baseline[k];
            }
        }
        scratch.copy_from_slice(&resid);
        let med = median_in_place(&mut scratch);
        for k in 0..n {
            scratch[k] = (resid[k] - med).abs();
        }
        let sigma = 1.4826 * median_in_place(&mut scratch);
        let thresh = mad_mult * sigma;

        let mut kept = 0usize;
        if sigma > 1e-30 {
            for k in 0..n {
                let ok = (resid[k] - med).abs() <= thresh;
                mask[k] = ok;
                if ok {
                    kept += 1;
                }
            }
        }
        let use_all = kept < 2 * d;

        let mut ata = [[0.0_f64; 6]; 6];
        let mut atb = [0.0_f64; 6];
        for k in 0..n {
            if !use_all && !mask[k] {
                continue;
            }
            let b = &basis[k];
            let y = spec[k];
            for r in 0..d {
                atb[r] += b[r] * y;
                for c in r..d {
                    ata[r][c] += b[r] * b[c];
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for r in 0..d {
            ata[r][r] += 1e-12;
            for c in 0..r {
                ata[r][c] = ata[c][r];
            }
        }

        let coeffs = solve_small_linear_system(ata, atb, d);
        for k in 0..n {
            let b = &basis[k];
            let mut val = 0.0_f64;
            for m in 0..d {
                val += coeffs[m] * b[m];
            }
            baseline[k] = val;
        }
    }

    for k in 0..n {
        spec[k] -= baseline[k];
    }
}

#[allow(clippy::needless_range_loop)]
fn solve_small_linear_system(mut a: [[f64; 6]; 6], mut b: [f64; 6], d: usize) -> [f64; 6] {
    for col in 0..d {
        let mut piv = col;
        let mut max_v = a[col][col].abs();
        for row in (col + 1)..d {
            let v = a[row][col].abs();
            if v > max_v {
                max_v = v;
                piv = row;
            }
        }
        if max_v <= 1e-30 {
            continue;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let inv_diag = 1.0 / a[col][col];
        for row in (col + 1)..d {
            let factor = a[row][col] * inv_diag;
            a[row][col] = 0.0;
            for k in (col + 1)..d {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0_f64; 6];
    for col in (0..d).rev() {
        let diag = a[col][col];
        if diag.abs() <= 1e-30 {
            continue;
        }
        let mut sum = b[col];
        for k in (col + 1)..d {
            sum -= a[col][k] * x[k];
        }
        x[col] = sum / diag;
    }
    x
}

fn median_in_place(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, &mut m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    m
}

/// Construct full uncropped frequency `AxisPpm` of size `si` from `FidAxisMeta`.
fn build_full_axis_ppm(meta: &FidAxisMeta, si: usize) -> AxisPpm {
    let sw_ppm = meta.sw_hz / meta.obs_mhz.max(1e-9);
    let step_ppm = -sw_ppm / (si.max(1) as f64);
    AxisPpm {
        label: meta.label.clone(),
        size: si,
        first_ppm: meta.first_ppm,
        step_ppm,
        obs_mhz: meta.obs_mhz,
    }
}

/// Step 9: Compute inclusive ROI index bounds `(idx_start, idx_end)` and cropped `AxisPpm`.
fn compute_axis_roi(
    full_axis: &AxisPpm,
    roi_ppm: Option<[f64; 2]>,
) -> Result<(usize, usize, AxisPpm), String> {
    let Some([a, b]) = roi_ppm else {
        return Ok((0, full_axis.size.saturating_sub(1), full_axis.clone()));
    };
    let lo = a.min(b);
    let hi = a.max(b);
    if lo > full_axis.max_ppm() || hi < full_axis.min_ppm() {
        return Err(format!(
            "ROI [{lo:.2}, {hi:.2}] ppm does not overlap {} axis [{:.2}, {:.2}] ppm",
            full_axis.label,
            full_axis.min_ppm(),
            full_axis.max_ppm()
        ));
    }
    let idx_a = full_axis.ppm_to_index(hi);
    let idx_b = full_axis.ppm_to_index(lo);
    let max_idx = (full_axis.size.saturating_sub(1)) as f64;
    let start = idx_a.min(idx_b).floor().clamp(0.0, max_idx) as usize;
    let end = idx_a.max(idx_b).ceil().clamp(start as f64, max_idx) as usize;
    let new_size = end - start + 1;
    let cropped = AxisPpm {
        label: full_axis.label.clone(),
        size: new_size,
        first_ppm: full_axis.index_to_ppm(start as f64),
        step_ppm: full_axis.step_ppm,
        obs_mhz: full_axis.obs_mhz,
    };
    Ok((start, end, cropped))
}

// ============================================================================
// Single-Trace & Batch 1D Processing Engine
// ============================================================================

/// Optional causal Hilbert imaginary-channel reconstruction converted to `process.rs`'s
/// `Re(S * exp(-i phi)) = re * cos(phi) + im * sin(phi)` sign convention.
#[inline]
fn apply_optional_hilbert_proc_sign(sc: &mut [Complex64], hilbert: bool) {
    if hilbert && sc.len() > 1 {
        for c in sc.iter_mut() {
            c[1] = 0.0;
        }
        hilbert_transform_in_place(sc);
        for c in sc.iter_mut() {
            c[1] = -c[1];
        }
    }
}

/// Conjugate a `process.rs` frequency trace (`re * cos + im * sin`) into `ops.rs`'s
/// `re * cos - im * sin` phase convention so `PhaseSolution` `(p0_deg, p1_deg)` matches verbatim.
#[inline]
fn conj_trace_for_ops(sc: &[Complex64]) -> Vec<Complex64> {
    sc.iter().map(|&c| [c[0], -c[1]]).collect()
}

fn auto_phase_cfg_for_exp(exp: &str) -> AutoPhaseConfig {
    let norm = exp.trim().replace(['(', ')'], "");
    if norm.eq_ignore_ascii_case("HNCACB") {
        AutoPhaseConfig::biphasic()
    } else {
        AutoPhaseConfig::default()
    }
}

/// Select up to `max_cols` distinct local-maximum peak-apex indices separated by `>= min_sep` bins.
fn select_apex_indices(energies: &[f64], min_sep: usize, max_cols: usize) -> Vec<usize> {
    let n = energies.len();
    let mut apices: Vec<(f64, usize)> = (0..n)
        .filter(|&i| {
            let e0 = energies[i];
            let el = if i > 0 { energies[i - 1] } else { 0.0 };
            let er = if i + 1 < n { energies[i + 1] } else { 0.0 };
            e0 > 0.0 && e0 >= el && e0 >= er
        })
        .map(|i| (energies[i], i))
        .collect();
    apices.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen = Vec::new();
    for &(_, idx) in &apices {
        if chosen.iter().all(|&q: &usize| q.abs_diff(idx) >= min_sep) {
            chosen.push(idx);
            if chosen.len() >= max_cols {
                break;
            }
        }
    }
    if chosen.len() < 4 && n > 0 {
        let mut all: Vec<(f64, usize)> = energies
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, e)| e > 0.0)
            .map(|(i, e)| (e, i))
            .collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, idx) in all {
            if !chosen.contains(&idx) {
                chosen.push(idx);
                if chosen.len() >= max_cols.min(16) {
                    break;
                }
            }
        }
        if chosen.is_empty() {
            chosen.push(0);
        }
    }
    chosen
}

/// Resolve `(p0_deg, p1_deg)` for a dimension given a representative unphased spectrum `rep_spec`
/// and full `AxisPpm`.
fn resolve_dim_phase(dim: &DimRecipe, full_axis: &AxisPpm, rep_spec: &[Complex64]) -> (f64, f64) {
    match (dim.p0_deg, dim.p1_deg, dim.phase_source) {
        (Some(p0), Some(p1), _) => (p0, p1),
        (Some(p0), None, _) => (p0, 0.0),
        (None, Some(p1), _) => (0.0, p1),
        (None, None, PhaseSource::DefaultZero) => (0.0, 0.0),
        (None, None, PhaseSource::Auto) => {
            let ops_tr = conj_trace_for_ops(rep_spec);
            let mut cfg = AutoPhaseConfig::default();
            if dim.remove_digital_filter && dim.group_delay > 0.0 {
                cfg.max_abs_p1_deg = cfg.max_abs_p1_deg.min(18.0);
            }
            let sol = auto_phase_1d(&ops_tr, full_axis, &cfg);
            (sol.p0_deg, sol.p1_deg)
        }
        (None, None, _) => {
            // Evaluate Chen et al. (2002) on the amide sub-window (6.0..=10.5 ppm) when 1H spans it,
            // avoiding any residual water peak near 4.7 ppm; otherwise evaluate across [10%..90%].
            let n = rep_spec.len();
            if full_axis.label.contains('H')
                && full_axis.max_ppm() > 6.5
                && full_axis.min_ppm() < 10.0
            {
                if let Ok((s0, s1, _)) = compute_axis_roi(full_axis, Some([6.0, 10.5])) {
                    if s1 > s0 + 16 {
                        let (p0_sub, p1_sub) = auto_phase_chen2002(&rep_spec[s0..=s1]);
                        return lift_slice_phase_to_full(p0_sub, p1_sub, s0, s1 - s0 + 1, n);
                    }
                }
            }
            let s0 = n / 10;
            let s1 = (n * 9 / 10).max(s0 + 1);
            let (p0_sub, p1_sub) = auto_phase_chen2002(&rep_spec[s0..s1]);
            lift_slice_phase_to_full(p0_sub, p1_sub, s0, s1 - s0, n)
        }
    }
}

/// Transform a single 1D complex time-domain trace through Steps 1–5 (returning unphased frequency
/// spectrum of length `si` so Step 6–9 can run once phases are resolved).
fn transform_trace_steps_1_to_5(
    raw_trace: &[Complex64],
    meta: &FidAxisMeta,
    dim: &DimRecipe,
    work_td: &mut Vec<Complex64>,
    work_fft: &mut Vec<Complex64>,
    out_spec: &mut Vec<Complex64>,
) {
    let raw_n = raw_trace.len().min(meta.complex_points);
    // Clamp td_eff when linear_prediction == false
    let td_limit = match dim.td_eff {
        Some(t) => t.min(raw_n).max(1),
        None => raw_n,
    };

    work_td.clear();
    work_td.extend_from_slice(&raw_trace[..td_limit]);

    // (1) Digital filter removal
    let (valid_len, dg_residual) = if dim.remove_digital_filter && dim.group_delay > 0.0 {
        apply_digital_filter_shift(work_td, dim.group_delay)
    } else {
        (td_limit, 0.0)
    };

    // (2) Optional direct 1H time-domain Gaussian solvent filter
    if let Some(ref sf) = dim.solvent_filter {
        if sf.enabled && valid_len >= 4 {
            apply_solvent_filter(work_td, valid_len, sf.half_width_points);
        }
    }

    // (3b) Optional indirect Mirror-Image Linear Prediction (`false` by default)
    if dim.linear_prediction && !dim.remove_digital_filter && valid_len >= 6 {
        let target = dim
            .lp_target_points
            .unwrap_or(2 * valid_len)
            .min(dim.si.max(valid_len))
            .max(valid_len);
        if target > valid_len {
            let eff_p0_deg = if dim.conjugate_for_fft {
                -dim.p0_deg.unwrap_or(0.0)
            } else {
                dim.p0_deg.unwrap_or(0.0)
            };
            let extended = apply_mirror_image_lp(
                &work_td[..valid_len],
                valid_len,
                target,
                dim.lp_order,
                eff_p0_deg,
            );
            *work_td = extended;
        } else {
            work_td.truncate(valid_len);
        }
    } else {
        work_td.truncate(valid_len);
    }

    // (3a + 3c) Apodization + first-point fcor scaling
    let n_apod = work_td.len();
    let weights = build_apodization_weights(dim.window, n_apod, meta.sw_hz, dim.fcor);
    for (c, &w) in work_td.iter_mut().zip(weights.iter()) {
        c[0] *= w;
        c[1] *= w;
    }

    // (4 + 5) Zero-fill to `si`, FFT, Nyquist shift, sub-sample `dg` phase, and optional `reverse`
    let si = dim.si.max(n_apod).max(2);
    transform_to_ordered_spectrum(
        work_td,
        si,
        dim.conjugate_for_fft,
        dg_residual,
        dim.reverse,
        work_fft,
        out_spec,
    );
}

// ============================================================================
// Clean-Room 1D & 2D Hypercomplex NUS Reconstruction (FISTA / IST)
// ============================================================================

/// Enforce the `States`-recombined hypercomplex invariant prior to 1D/2D `IST` / `FISTA`
/// non-uniform sampling reconstruction: Echo-Antiecho `(S+, S-)` pairs must already be recombined
/// into separate `(cos, sin)` channels and `States-TPPI` `(-1)^k` sign alternation must already be
/// removed so every point peak corresponds to a single sparse complex / bicomplex exponential.
pub fn validate_nus_states_axis(axis: &FidAxisMeta, dim_desc: &str) -> Result<(), String> {
    if !axis.states_recombined || axis.quad_mode == QuadMode::Qf {
        return Err(format!(
            "NUS IST/FISTA on {dim_desc} ('{}', quad_mode={:?}) requires uniform States-recombined hypercomplex input prior to reconstruction",
            axis.label, axis.quad_mode
        ));
    }
    Ok(())
}

/// Reconstruct missing samples of a 1D indirect complex trace `obs[0..n1]` sampled on `mask[0..n1]`
/// using proximal-gradient `FISTA` / `IST` ($L_1$ soft-thresholding in the unitary frequency domain
/// with dynamic **residual-spectrum** $\sigma_{\text{MAD}}$ noise estimation and exact data consistency;
/// Stern et al. 2007; Beck & Teboulle 2009; Hyberts et al. 2012).
pub fn reconstruct_nus_1d_ist(
    obs: &[Complex64],
    mask: &[bool],
    cfg: &NusIstConfig,
) -> Vec<Complex64> {
    let n1 = obs.len().min(mask.len());
    if n1 == 0 || mask.iter().take(n1).all(|&m| m) {
        return obs[..n1].to_vec();
    }

    let n_fft = (2 * n1).next_power_of_two().max(4);
    let scale_fwd = 1.0 / (n_fft as f64).sqrt();
    let scale_inv = (n_fft as f64).sqrt();

    let mut y_obs = vec![[0.0, 0.0]; n1];
    for k in 0..n1 {
        if mask[k] {
            y_obs[k] = obs[k];
        }
    }

    let iters = cfg.iterations.max(1);
    let init_frac = cfg.initial_thresh_frac.clamp(0.1, 0.99);
    // Geometric continuation schedule reaches 1e-4 * max_mag so >= 100:1 dynamic-range weak peaks
    // enter cleanly once strong-peak PSF sidelobes are subtracted from the residual spectrum.
    let sched_ratio = if iters > 1 {
        (1e-4_f64 / init_frac).powf(1.0 / ((iters - 1) as f64))
    } else {
        1.0
    };
    let noise_mult = cfg.final_noise_sigma.max(0.5);

    let mut x_spec = vec![[0.0, 0.0]; n_fft];
    let mut x_next = vec![[0.0, 0.0]; n_fft];
    let mut w_spec = vec![[0.0, 0.0]; n_fft];
    let mut td_work = vec![[0.0, 0.0]; n_fft];
    let mut mags = vec![0.0_f64; n_fft];
    let mut active = vec![false; n_fft];

    let mut max_mag = 0.0_f64;
    let mut lam_start = 0.0_f64;
    let mut t_fista = 1.0_f64;
    let mut prev_res_e = f64::INFINITY;

    for it in 0..iters {
        // Compute masked time-domain residual r_td = P_Omega(y_obs - F^{-1}(w_spec))
        let mut res_e = 0.0_f64;
        if it == 0 {
            td_work.fill([0.0, 0.0]);
            for k in 0..n1 {
                if mask[k] {
                    td_work[k] = y_obs[k];
                    res_e += y_obs[k][0] * y_obs[k][0] + y_obs[k][1] * y_obs[k][1];
                }
            }
        } else {
            for m in 0..n_fft {
                td_work[m] = [w_spec[m][0] * scale_inv, w_spec[m][1] * scale_inv];
            }
            fft_in_place(&mut td_work, true);
            for k in 0..n_fft {
                if k < n1 && mask[k] {
                    let dr = y_obs[k][0] - td_work[k][0];
                    let di = y_obs[k][1] - td_work[k][1];
                    td_work[k] = [dr, di];
                    res_e += dr * dr + di * di;
                } else {
                    td_work[k] = [0.0, 0.0];
                }
            }
        }

        // Unitary residual spectrum R = F(r_td) / sqrt(n_fft)
        fft_in_place(&mut td_work, false);
        for (m, c) in td_work.iter_mut().enumerate() {
            c[0] *= scale_fwd;
            c[1] *= scale_fwd;
            let mag = (c[0] * c[0] + c[1] * c[1]).sqrt();
            mags[m] = mag;
            if it == 0 && mag > max_mag {
                max_mag = mag;
            }
        }
        if it == 0 {
            if max_mag <= 1e-30 {
                return y_obs;
            }
            lam_start = init_frac * max_mag;
        }

        // Estimate thermal / unmodeled noise floor from the CURRENT residual spectrum (not initial dirty FT!)
        let med_res = median_in_place(&mut mags);
        for v in &mut mags {
            *v = (*v - med_res).abs();
        }
        let sigma_res = 1.4826 * median_in_place(&mut mags);
        let lam_sched = lam_start * sched_ratio.powi(it as i32);
        let lam = lam_sched
            .max(noise_mult * sigma_res)
            .max(1e-5 * max_mag)
            .min(lam_start);

        let is_last = it + 1 == iters;
        for m in 0..n_fft {
            let ur = w_spec[m][0] + td_work[m][0];
            let ui = w_spec[m][1] + td_work[m][1];
            let mag = (ur * ur + ui * ui).sqrt();
            if mag > lam {
                let s = 1.0 - lam / mag;
                x_next[m] = [ur * s, ui * s];
                if is_last {
                    active[m] = true;
                }
            } else {
                x_next[m] = [0.0, 0.0];
            }
        }

        if cfg.fista_momentum {
            if res_e > prev_res_e * 1.05 {
                t_fista = 1.0;
            }
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt());
            let beta = ((t_fista - 1.0) / t_next).min(0.75);
            for m in 0..n_fft {
                w_spec[m] = [
                    x_next[m][0] + beta * (x_next[m][0] - x_spec[m][0]),
                    x_next[m][1] + beta * (x_next[m][1] - x_spec[m][1]),
                ];
            }
            t_fista = t_next;
            prev_res_e = res_e;
        } else {
            w_spec.copy_from_slice(&x_next);
        }
        x_spec.copy_from_slice(&x_next);
    }

    // Transform sparse unitary frequency solution back to time domain and enforce exact data consistency
    let mut z_cur = vec![[0.0, 0.0]; n_fft];
    for m in 0..n_fft {
        z_cur[m] = [x_spec[m][0] * scale_inv, x_spec[m][1] * scale_inv];
    }
    fft_in_place(&mut z_cur, true);
    for k in 0..n1 {
        if mask[k] {
            z_cur[k] = y_obs[k];
        }
    }

    // Debiasing support projections: eliminate L1 soft-thresholding shrinkage bias on active peaks
    if active.iter().any(|&a| a) {
        for _ in 0..8 {
            td_work.copy_from_slice(&z_cur);
            for k in 0..n1 {
                if mask[k] {
                    td_work[k] = y_obs[k];
                }
            }
            fft_in_place(&mut td_work, false);
            for (m, c) in td_work.iter_mut().enumerate() {
                if !active[m] {
                    *c = [0.0, 0.0];
                }
            }
            fft_in_place(&mut td_work, true);
            z_cur.copy_from_slice(&td_work);
            for k in 0..n1 {
                if mask[k] {
                    z_cur[k] = y_obs[k];
                }
            }
        }
    }

    z_cur.truncate(n1);
    z_cur
}

/// Reconstruct missing hypercomplex samples `(cc, cs, sc, ss)` on a 2D indirect grid `(n2, n1)`
/// (`mask[k2 * n1 + k1]`) via bicomplex `(u_+, u_-)` 2D unitary FFT + joint quaternion norm
/// proximal-gradient soft-thresholding with dynamic **residual-spectrum** $\sigma_{\text{MAD}}$
/// noise calibration (`FISTA / IST`, Stern et al. 2007; Hyberts et al. 2012).
pub fn reconstruct_nus_2d_hypercomplex_ist(
    obs_quad: &[[f64; 4]],
    n2: usize,
    n1: usize,
    mask: &[bool],
    cfg: &NusIstConfig,
) -> Vec<[f64; 4]> {
    let total = n2 * n1;
    assert_eq!(obs_quad.len(), total);
    assert_eq!(mask.len(), total);
    if total == 0 || mask.iter().all(|&m| m) {
        return obs_quad.to_vec();
    }

    // Bicomplex decomposition:
    //   u_+ = (cc - ss) + i (sc + cs) = A exp(+i a2) exp(+i a1)
    //   u_- = (cc + ss) + i (sc - cs) = A exp(+i a2) exp(-i a1)
    let mut y_plus = vec![[0.0, 0.0]; total];
    let mut y_minus = vec![[0.0, 0.0]; total];
    for idx in 0..total {
        if mask[idx] {
            let [cc, cs, sc, ss] = obs_quad[idx];
            y_plus[idx] = [cc - ss, sc + cs];
            y_minus[idx] = [cc + ss, sc - cs];
        }
    }

    let f2 = (2 * n2).next_power_of_two().max(4);
    let f1 = (2 * n1).next_power_of_two().max(4);
    let n_vox = f2 * f1;
    let scale_fwd = 1.0 / (n_vox as f64).sqrt();
    let scale_inv = (n_vox as f64).sqrt();

    let mut sp_plus = vec![[0.0, 0.0]; n_vox];
    let mut sp_minus = vec![[0.0, 0.0]; n_vox];
    let mut xp_spec = vec![[0.0, 0.0]; n_vox];
    let mut xm_spec = vec![[0.0, 0.0]; n_vox];
    let mut xp_next = vec![[0.0, 0.0]; n_vox];
    let mut xm_next = vec![[0.0, 0.0]; n_vox];
    let mut wp_spec = vec![[0.0, 0.0]; n_vox];
    let mut wm_spec = vec![[0.0, 0.0]; n_vox];
    let mut norms = vec![0.0_f64; n_vox];
    let mut col_buf = vec![[0.0, 0.0]; f2];
    let mut active_plus = vec![false; n_vox];

    let fft_2d = |grid: &mut [Complex64], col_tmp: &mut [Complex64], inverse: bool| {
        for r in 0..f2 {
            fft_in_place(&mut grid[r * f1..(r + 1) * f1], inverse);
        }
        for c in 0..f1 {
            for r in 0..f2 {
                col_tmp[r] = grid[r * f1 + c];
            }
            fft_in_place(col_tmp, inverse);
            for r in 0..f2 {
                grid[r * f1 + c] = col_tmp[r];
            }
        }
    };

    let iters = cfg.iterations.max(1);
    let init_frac = cfg.initial_thresh_frac.clamp(0.1, 0.99);
    let sched_ratio = if iters > 1 {
        (1e-4_f64 / init_frac).powf(1.0 / ((iters - 1) as f64))
    } else {
        1.0
    };
    let noise_mult = cfg.final_noise_sigma.max(0.5);

    let mut max_norm = 0.0_f64;
    let mut lam_start = 0.0_f64;
    let mut t_fista = 1.0_f64;
    let mut prev_res_e = f64::INFINITY;

    for it in 0..iters {
        let mut res_e = 0.0_f64;
        if it == 0 {
            sp_plus.fill([0.0, 0.0]);
            sp_minus.fill([0.0, 0.0]);
            for k2 in 0..n2 {
                for k1 in 0..n1 {
                    let idx = k2 * n1 + k1;
                    if mask[idx] {
                        let yp = y_plus[idx];
                        let ym = y_minus[idx];
                        sp_plus[k2 * f1 + k1] = yp;
                        sp_minus[k2 * f1 + k1] = ym;
                        res_e += yp[0] * yp[0] + yp[1] * yp[1] + ym[0] * ym[0] + ym[1] * ym[1];
                    }
                }
            }
        } else {
            for i in 0..n_vox {
                sp_plus[i] = [wp_spec[i][0] * scale_inv, wp_spec[i][1] * scale_inv];
                sp_minus[i] = [wm_spec[i][0] * scale_inv, wm_spec[i][1] * scale_inv];
            }
            fft_2d(&mut sp_plus, &mut col_buf, true);
            fft_2d(&mut sp_minus, &mut col_buf, true);
            for k2 in 0..f2 {
                for k1 in 0..f1 {
                    let g_idx = k2 * f1 + k1;
                    if k2 < n2 && k1 < n1 && mask[k2 * n1 + k1] {
                        let idx = k2 * n1 + k1;
                        let dp = [
                            y_plus[idx][0] - sp_plus[g_idx][0],
                            y_plus[idx][1] - sp_plus[g_idx][1],
                        ];
                        let dm = [
                            y_minus[idx][0] - sp_minus[g_idx][0],
                            y_minus[idx][1] - sp_minus[g_idx][1],
                        ];
                        sp_plus[g_idx] = dp;
                        sp_minus[g_idx] = dm;
                        res_e += dp[0] * dp[0] + dp[1] * dp[1] + dm[0] * dm[0] + dm[1] * dm[1];
                    } else {
                        sp_plus[g_idx] = [0.0, 0.0];
                        sp_minus[g_idx] = [0.0, 0.0];
                    }
                }
            }
        }

        // Unitary 2D residual spectrum (res_plus, res_minus)
        fft_2d(&mut sp_plus, &mut col_buf, false);
        fft_2d(&mut sp_minus, &mut col_buf, false);
        for i in 0..n_vox {
            sp_plus[i][0] *= scale_fwd;
            sp_plus[i][1] *= scale_fwd;
            sp_minus[i][0] *= scale_fwd;
            sp_minus[i][1] *= scale_fwd;
        }

        for m2 in 0..f2 {
            for m1 in 0..f1 {
                let m1_conj = (f1 - m1) % f1;
                let up = sp_plus[m2 * f1 + m1];
                let um = sp_minus[m2 * f1 + m1_conj];
                let h =
                    (0.5 * (up[0] * up[0] + up[1] * up[1] + um[0] * um[0] + um[1] * um[1])).sqrt();
                norms[m2 * f1 + m1] = h;
                if it == 0 && h > max_norm {
                    max_norm = h;
                }
            }
        }
        if it == 0 {
            if max_norm <= 1e-30 {
                return vec![[0.0; 4]; total];
            }
            lam_start = init_frac * max_norm;
        }

        // Dynamic MAD noise floor of the CURRENT quaternion residual spectrum
        let med = median_in_place(&mut norms);
        for v in &mut norms {
            *v = (*v - med).abs();
        }
        let sigma_res = 1.4826 * median_in_place(&mut norms);
        let lam_sched = lam_start * sched_ratio.powi(it as i32);
        let lam = lam_sched
            .max(noise_mult * sigma_res)
            .max(1e-5 * max_norm)
            .min(lam_start);

        let is_last = it + 1 == iters;
        // Joint group-sparse quaternion soft-thresholding on `w_spec + res_spec`
        for m2 in 0..f2 {
            for &m1 in &[0usize, f1 / 2] {
                let i_pos = m2 * f1 + m1;
                let up = [
                    wp_spec[i_pos][0] + sp_plus[i_pos][0],
                    wp_spec[i_pos][1] + sp_plus[i_pos][1],
                ];
                let um = [
                    wm_spec[i_pos][0] + sp_minus[i_pos][0],
                    wm_spec[i_pos][1] + sp_minus[i_pos][1],
                ];
                let h =
                    (0.5 * (up[0] * up[0] + up[1] * up[1] + um[0] * um[0] + um[1] * um[1])).sqrt();
                if h > lam {
                    let s = 1.0 - lam / h;
                    xp_next[i_pos] = [up[0] * s, up[1] * s];
                    xm_next[i_pos] = [um[0] * s, um[1] * s];
                    if is_last {
                        active_plus[i_pos] = true;
                    }
                } else {
                    xp_next[i_pos] = [0.0, 0.0];
                    xm_next[i_pos] = [0.0, 0.0];
                }
            }
            for m1 in 1..(f1 / 2) {
                let m1_c = f1 - m1;
                let idx_a = m2 * f1 + m1;
                let idx_b = m2 * f1 + m1_c;

                let up_a = [
                    wp_spec[idx_a][0] + sp_plus[idx_a][0],
                    wp_spec[idx_a][1] + sp_plus[idx_a][1],
                ];
                let um_a = [
                    wm_spec[idx_b][0] + sp_minus[idx_b][0],
                    wm_spec[idx_b][1] + sp_minus[idx_b][1],
                ];
                let h_a = (0.5
                    * (up_a[0] * up_a[0]
                        + up_a[1] * up_a[1]
                        + um_a[0] * um_a[0]
                        + um_a[1] * um_a[1]))
                    .sqrt();

                let up_b = [
                    wp_spec[idx_b][0] + sp_plus[idx_b][0],
                    wp_spec[idx_b][1] + sp_plus[idx_b][1],
                ];
                let um_b = [
                    wm_spec[idx_a][0] + sp_minus[idx_a][0],
                    wm_spec[idx_a][1] + sp_minus[idx_a][1],
                ];
                let h_b = (0.5
                    * (up_b[0] * up_b[0]
                        + up_b[1] * up_b[1]
                        + um_b[0] * um_b[0]
                        + um_b[1] * um_b[1]))
                    .sqrt();

                if h_a > lam {
                    let s = 1.0 - lam / h_a;
                    xp_next[idx_a] = [up_a[0] * s, up_a[1] * s];
                    xm_next[idx_b] = [um_a[0] * s, um_a[1] * s];
                    if is_last {
                        active_plus[idx_a] = true;
                    }
                } else {
                    xp_next[idx_a] = [0.0, 0.0];
                    xm_next[idx_b] = [0.0, 0.0];
                }

                if h_b > lam {
                    let s = 1.0 - lam / h_b;
                    xp_next[idx_b] = [up_b[0] * s, up_b[1] * s];
                    xm_next[idx_a] = [um_b[0] * s, um_b[1] * s];
                    if is_last {
                        active_plus[idx_b] = true;
                    }
                } else {
                    xp_next[idx_b] = [0.0, 0.0];
                    xm_next[idx_a] = [0.0, 0.0];
                }
            }
        }

        if cfg.fista_momentum {
            if res_e > prev_res_e * 1.05 {
                t_fista = 1.0;
            }
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt());
            let beta = ((t_fista - 1.0) / t_next).min(0.75);
            for i in 0..n_vox {
                wp_spec[i] = [
                    xp_next[i][0] + beta * (xp_next[i][0] - xp_spec[i][0]),
                    xp_next[i][1] + beta * (xp_next[i][1] - xp_spec[i][1]),
                ];
                wm_spec[i] = [
                    xm_next[i][0] + beta * (xm_next[i][0] - xm_spec[i][0]),
                    xm_next[i][1] + beta * (xm_next[i][1] - xm_spec[i][1]),
                ];
            }
            t_fista = t_next;
            prev_res_e = res_e;
        } else {
            wp_spec.copy_from_slice(&xp_next);
            wm_spec.copy_from_slice(&xm_next);
        }
        xp_spec.copy_from_slice(&xp_next);
        xm_spec.copy_from_slice(&xm_next);
    }

    let mut zp_cur = vec![[0.0, 0.0]; n_vox];
    let mut zm_cur = vec![[0.0, 0.0]; n_vox];
    for i in 0..n_vox {
        zp_cur[i] = [xp_spec[i][0] * scale_inv, xp_spec[i][1] * scale_inv];
        zm_cur[i] = [xm_spec[i][0] * scale_inv, xm_spec[i][1] * scale_inv];
    }
    fft_2d(&mut zp_cur, &mut col_buf, true);
    fft_2d(&mut zm_cur, &mut col_buf, true);
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            let idx = k2 * n1 + k1;
            if mask[idx] {
                zp_cur[k2 * f1 + k1] = y_plus[idx];
                zm_cur[k2 * f1 + k1] = y_minus[idx];
            }
        }
    }

    // Debiasing support projections on active quaternion frequencies
    if active_plus.iter().any(|&a| a) {
        for _ in 0..6 {
            sp_plus.copy_from_slice(&zp_cur);
            sp_minus.copy_from_slice(&zm_cur);
            for k2 in 0..n2 {
                for k1 in 0..n1 {
                    let idx = k2 * n1 + k1;
                    if mask[idx] {
                        sp_plus[k2 * f1 + k1] = y_plus[idx];
                        sp_minus[k2 * f1 + k1] = y_minus[idx];
                    }
                }
            }
            fft_2d(&mut sp_plus, &mut col_buf, false);
            fft_2d(&mut sp_minus, &mut col_buf, false);
            for m2 in 0..f2 {
                for m1 in 0..f1 {
                    let m1_c = (f1 - m1) % f1;
                    if !active_plus[m2 * f1 + m1] {
                        sp_plus[m2 * f1 + m1] = [0.0, 0.0];
                        sp_minus[m2 * f1 + m1_c] = [0.0, 0.0];
                    }
                }
            }
            fft_2d(&mut sp_plus, &mut col_buf, true);
            fft_2d(&mut sp_minus, &mut col_buf, true);
            zp_cur.copy_from_slice(&sp_plus);
            zm_cur.copy_from_slice(&sp_minus);
            for k2 in 0..n2 {
                for k1 in 0..n1 {
                    let idx = k2 * n1 + k1;
                    if mask[idx] {
                        zp_cur[k2 * f1 + k1] = y_plus[idx];
                        zm_cur[k2 * f1 + k1] = y_minus[idx];
                    }
                }
            }
        }
    }

    // Invert bicomplex mapping back to (cc, cs, sc, ss)
    let mut out = vec![[0.0; 4]; total];
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            let idx = k2 * n1 + k1;
            if mask[idx] {
                out[idx] = obs_quad[idx];
            } else {
                let up = zp_cur[k2 * f1 + k1];
                let um = zm_cur[k2 * f1 + k1];
                out[idx] = [
                    0.5 * (um[0] + up[0]),
                    0.5 * (up[1] - um[1]),
                    0.5 * (up[1] + um[1]),
                    0.5 * (um[0] - up[0]),
                ];
            }
        }
    }
    out
}

// ============================================================================
// Parallel Map Helper (`std::thread::scope` on native, sequential on `wasm32`)
// ============================================================================

fn parallel_chunks_mut<T, F>(slice: &mut [T], chunk_size: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let csize = chunk_size.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let n_chunks = slice.len().div_ceil(csize);
        if n_chunks <= 1 {
            f(0, slice);
        } else {
            std::thread::scope(|s| {
                for (ci, chunk) in slice.chunks_mut(csize).enumerate() {
                    let f_ref = &f;
                    let base = ci * csize;
                    s.spawn(move || f_ref(base, chunk));
                }
            });
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (ci, chunk) in slice.chunks_mut(csize).enumerate() {
            f(ci * csize, chunk);
        }
    }
}

fn default_chunk_size(total_items: usize) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        total_items.div_ceil(threads * 2).max(1)
    }
    #[cfg(target_arch = "wasm32")]
    {
        total_items.max(1)
    }
}

// ============================================================================
// Top-Level 1D, 2D, and 3D FID Processing Entrypoints
// ============================================================================

/// Process a 1D FID into `(AxisPpm, Vec<f32>)`, updating `recipe.dims[0]` with resolved phases if auto-phased.
pub fn process_fid_1d(
    fid: &Fid1D,
    recipe: &ProcessingRecipe,
) -> Result<(AxisPpm, Vec<f32>), String> {
    let mut rec = recipe.clone();
    process_fid_1d_mut(fid, &mut rec)
}

pub fn process_fid_1d_mut(
    fid: &Fid1D,
    recipe: &mut ProcessingRecipe,
) -> Result<(AxisPpm, Vec<f32>), String> {
    if recipe.dims.is_empty() {
        return Err("ProcessingRecipe has 0 dimensions for 1D FID".to_string());
    }
    let dim0 = &mut recipe.dims[0];
    let mut work_td = Vec::new();
    let mut work_fft = Vec::new();
    let mut spec_c = Vec::new();
    transform_trace_steps_1_to_5(
        &fid.data,
        &fid.axis_x,
        dim0,
        &mut work_td,
        &mut work_fft,
        &mut spec_c,
    );
    apply_optional_hilbert_proc_sign(&mut spec_c, dim0.hilbert);

    let full_ax = build_full_axis_ppm(&fid.axis_x, spec_c.len());
    let (p0, p1) = resolve_dim_phase(dim0, &full_ax, &spec_c);
    dim0.p0_deg = Some(p0);
    dim0.p1_deg = Some(p1);

    let mut real_spec = vec![0.0_f64; spec_c.len()];
    apply_phase_and_discard_imag(&spec_c, p0, p1, &mut real_spec);

    if let Some(ref bc) = dim0.baseline {
        subtract_polynomial_baseline(&mut real_spec, bc);
    }

    let (i0, i1, out_ax) = compute_axis_roi(&full_ax, dim0.roi_ppm)?;
    let data: Vec<f32> = real_spec[i0..=i1].iter().map(|&v| v as f32).collect();
    Ok((out_ax, data))
}

/// Process a 2D FID (`Fid2D`, uniform or 1D NUS) into a `Spectrum2D`.
pub fn process_fid_2d(fid: &Fid2D, recipe: &ProcessingRecipe) -> Result<Spectrum2D, String> {
    let mut rec = recipe.clone();
    process_fid_2d_mut(fid, &mut rec)
}

/// Process a 2D FID while recording any automatically solved `(p0_deg, p1_deg)` back into `recipe`.
pub fn process_fid_2d_mut(
    fid: &Fid2D,
    recipe: &mut ProcessingRecipe,
) -> Result<Spectrum2D, String> {
    if recipe.dims.len() < 2 {
        return Err(format!(
            "ProcessingRecipe has {} dimensions, expected 2 for Fid2D",
            recipe.dims.len()
        ));
    }

    let n0 = fid.axis_x.complex_points;
    let n1 = fid.axis_y.complex_points;
    if fid.data.len() < n1 * 2 * n0 {
        return Err(format!(
            "Fid2D data buffer truncated: got {}, expected {}",
            fid.data.len(),
            n1 * 2 * n0
        ));
    }
    if let Some(ref mask) = fid.nus_mask {
        if mask.len() != n1 {
            return Err(format!(
                "Fid2D nus_mask length mismatch: got {}, expected n1={n1}",
                mask.len()
            ));
        }
        validate_nus_states_axis(&fid.axis_y, "2D indirect F1")?;
    }

    // ------------------------------------------------------------------------
    // Dimension 0 (Direct 1H): Steps 1–9 across `n1 * 2` traces
    // ------------------------------------------------------------------------
    let dim_x = recipe.dims[0].clone();
    let si_x_est = dim_x.si.max(n0).max(2);
    let full_axis_x = build_full_axis_ppm(&fid.axis_x, si_x_est);
    let (x0, x1, axis_x_out) = compute_axis_roi(&full_axis_x, dim_x.roi_ppm)?;
    let nx_out = axis_x_out.size;

    let (p0_x, p1_x) = match (dim_x.p0_deg, dim_x.p1_deg, dim_x.phase_source) {
        (Some(p0), Some(p1), _) => (p0, p1),
        (Some(p0), None, _) => (p0, 0.0),
        (None, Some(p1), _) => (0.0, p1),
        (None, None, PhaseSource::DefaultZero) => (0.0, 0.0),
        (None, None, PhaseSource::Auto) => {
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut sample_x = Vec::new();
            let mut ax_rep: Option<AxisPpm> = None;
            'sample_2d_x: for k1 in 0..n1 {
                if let Some(ref mask) = fid.nus_mask {
                    if !mask[k1] {
                        continue;
                    }
                }
                for c1 in 0..2 {
                    transform_trace_steps_1_to_5(
                        fid.trace(k1, c1),
                        &fid.axis_x,
                        &dim_x,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_x.hilbert);
                    if ax_rep.is_none() {
                        ax_rep = Some(build_full_axis_ppm(&fid.axis_x, sc.len()));
                    }
                    sample_x.push(conj_trace_for_ops(&sc));
                    if sample_x.len() >= 64 {
                        break 'sample_2d_x;
                    }
                }
            }
            if let Some(ax_x_full) = ax_rep {
                let mut cfg_direct = AutoPhaseConfig::default();
                if dim_x.remove_digital_filter && dim_x.group_delay > 0.0 {
                    cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(18.0);
                } else if fid.axis_x.group_delay == 0.0
                    && fid.axis_x.stored_proc.p1_deg.unwrap_or(0.0).abs() <= 20.0
                {
                    cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(25.0);
                }
                let sol = auto_phase_direct_nd(&sample_x, &ax_x_full, &cfg_direct);
                (sol.p0_deg, sol.p1_deg)
            } else {
                (0.0, 0.0)
            }
        }
        (None, None, _) => {
            let first_k1 = fid
                .nus_mask
                .as_ref()
                .and_then(|m| m.iter().position(|&b| b))
                .unwrap_or(0);
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut rep_spec = Vec::new();
            transform_trace_steps_1_to_5(
                fid.trace(first_k1, 0),
                &fid.axis_x,
                &dim_x,
                &mut wt,
                &mut wf,
                &mut rep_spec,
            );
            apply_optional_hilbert_proc_sign(&mut rep_spec, dim_x.hilbert);
            let ax_rep = build_full_axis_ppm(&fid.axis_x, rep_spec.len());
            resolve_dim_phase(&dim_x, &ax_rep, &rep_spec)
        }
    };
    recipe.dims[0].p0_deg = Some(p0_x);
    recipe.dims[0].p1_deg = Some(p1_x);

    // Transform all `2 * n1` direct traces in parallel -> `direct_cropped[(k1 * 2 + c1) * nx_out + ix]`
    let n_direct_traces = n1 * 2;
    let mut direct_cropped = vec![0.0_f64; n_direct_traces * nx_out];
    let nus_mask_ref = fid.nus_mask.as_deref();

    parallel_chunks_mut(
        &mut direct_cropped,
        default_chunk_size(n_direct_traces) * nx_out,
        |base_elem, chunk| {
            let base_tr = base_elem / nx_out;
            let n_tr = chunk.len() / nx_out;
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut real_full = Vec::new();

            for local in 0..n_tr {
                let tr_idx = base_tr + local;
                let k1 = tr_idx / 2;
                let c1 = tr_idx % 2;
                let dst = &mut chunk[local * nx_out..(local + 1) * nx_out];
                if let Some(mask) = nus_mask_ref {
                    if !mask[k1] {
                        dst.fill(0.0);
                        continue;
                    }
                }
                let raw_tr = fid.trace(k1, c1);
                transform_trace_steps_1_to_5(
                    raw_tr,
                    &fid.axis_x,
                    &dim_x,
                    &mut wt,
                    &mut wf,
                    &mut sc,
                );
                apply_optional_hilbert_proc_sign(&mut sc, dim_x.hilbert);
                real_full.resize(sc.len(), 0.0);
                apply_phase_and_discard_imag(&sc, p0_x, p1_x, &mut real_full);
                dst.copy_from_slice(&real_full[x0..=x1]);
            }
        },
    );

    // ------------------------------------------------------------------------
    // Dimension 1 (Indirect F1): Optional NUS IST + Steps 3–9 across `nx_out` columns
    // ------------------------------------------------------------------------
    let mut dim_y = recipe.dims[1].clone();
    let si_y = dim_y.si.max(n1).max(2);
    let full_axis_y = build_full_axis_ppm(&fid.axis_y, si_y);
    let (y0, y1, axis_y_out) = compute_axis_roi(&full_axis_y, dim_y.roi_ppm)?;
    let ny_out = axis_y_out.size;
    let nus_cfg = recipe.nus.clone().unwrap_or_default();
    let auto_solved_y = dim_y.p0_deg.is_none()
        && dim_y.p1_deg.is_none()
        && !matches!(dim_y.phase_source, PhaseSource::DefaultZero);

    let (p0_y, p1_y) = match (dim_y.p0_deg, dim_y.p1_deg, dim_y.phase_source) {
        (Some(p0), Some(p1), _) => (p0, p1),
        (Some(p0), None, _) => (p0, 0.0),
        (None, Some(p1), _) => (0.0, p1),
        (None, None, PhaseSource::DefaultZero) => (0.0, 0.0),
        (None, None, _) => {
            // Select up to 48 distinct `1H` local-maximum apex columns for blind indirect autophasing
            let mut col_energy = vec![0.0_f64; nx_out];
            for ix in 0..nx_out {
                let mut e = 0.0;
                for k1 in 0..n1 {
                    let re = direct_cropped[(k1 * 2) * nx_out + ix];
                    let im = direct_cropped[(k1 * 2 + 1) * nx_out + ix];
                    e += re * re + im * im;
                }
                col_energy[ix] = e;
            }
            let chosen_ix = select_apex_indices(&col_energy, 3, 48);
            let mut cand_cols = Vec::with_capacity(chosen_ix.len());
            let mut col_td = vec![[0.0, 0.0]; n1];
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut ax_y_rep: Option<AxisPpm> = None;
            for &ix in &chosen_ix {
                for k1 in 0..n1 {
                    col_td[k1] = [
                        direct_cropped[(k1 * 2) * nx_out + ix],
                        direct_cropped[(k1 * 2 + 1) * nx_out + ix],
                    ];
                }
                if let Some(mask) = nus_mask_ref {
                    let rec = reconstruct_nus_1d_ist(&col_td, mask, &nus_cfg);
                    col_td.copy_from_slice(&rec);
                }
                transform_trace_steps_1_to_5(
                    &col_td,
                    &fid.axis_y,
                    &dim_y,
                    &mut wt,
                    &mut wf,
                    &mut sc,
                );
                apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                if ax_y_rep.is_none() {
                    ax_y_rep = Some(build_full_axis_ppm(&fid.axis_y, sc.len()));
                }
                cand_cols.push(conj_trace_for_ops(&sc));
            }
            if let Some(ax_y_full) = ax_y_rep {
                let sol =
                    auto_phase_indirect_nd(&cand_cols, &ax_y_full, &AutoPhaseConfig::default());
                (sol.p0_deg, sol.p1_deg)
            } else {
                (0.0, 0.0)
            }
        }
    };
    recipe.dims[1].p0_deg = Some(p0_y);
    recipe.dims[1].p1_deg = Some(p1_y);
    if auto_solved_y && p1_y.abs() > 15.0 {
        dim_y.fcor = 1.0;
        recipe.dims[1].fcor = 1.0;
    }

    // Process each `ix in 0..nx_out` column independently in parallel, storing transposed `[nx_out * ny_out]`
    // then scattering to row-major `[ny_out * nx_out]`.
    let mut cols_out = vec![0.0_f32; nx_out * ny_out];
    parallel_chunks_mut(
        &mut cols_out,
        default_chunk_size(nx_out) * ny_out,
        |base_elem, chunk| {
            let base_ix = base_elem / ny_out;
            let cnt_ix = chunk.len() / ny_out;
            let mut col_td = vec![[0.0, 0.0]; n1];
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut real_full = Vec::new();

            for local in 0..cnt_ix {
                let ix = base_ix + local;
                for k1 in 0..n1 {
                    let re = direct_cropped[(k1 * 2) * nx_out + ix];
                    let im = direct_cropped[(k1 * 2 + 1) * nx_out + ix];
                    col_td[k1] = [re, im];
                }
                let uniform_td: &[Complex64] = if let Some(mask) = nus_mask_ref {
                    let rec = reconstruct_nus_1d_ist(&col_td, mask, &nus_cfg);
                    col_td.copy_from_slice(&rec);
                    &col_td
                } else {
                    &col_td
                };

                transform_trace_steps_1_to_5(
                    uniform_td,
                    &fid.axis_y,
                    &dim_y,
                    &mut wt,
                    &mut wf,
                    &mut sc,
                );
                apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                real_full.resize(sc.len(), 0.0);
                apply_phase_and_discard_imag(&sc, p0_y, p1_y, &mut real_full);
                if let Some(ref bc) = dim_y.baseline {
                    subtract_polynomial_baseline(&mut real_full, bc);
                }
                let dst = &mut chunk[local * ny_out..(local + 1) * ny_out];
                for (dy, &v) in real_full[y0..=y1].iter().enumerate() {
                    dst[dy] = v as f32;
                }
            }
        },
    );

    let mut data = vec![0.0_f32; ny_out * nx_out];
    for ix in 0..nx_out {
        let src = &cols_out[ix * ny_out..(ix + 1) * ny_out];
        for iy in 0..ny_out {
            data[iy * nx_out + ix] = src[iy];
        }
    }
    let run_x_baseline = !(dim_x.roi_ppm.is_none() && dim_x.remove_digital_filter);
    if let Some(ref bc) = dim_x.baseline {
        if bc.enabled && run_x_baseline && nx_out >= 8 {
            parallel_chunks_mut(
                &mut data,
                default_chunk_size(ny_out) * nx_out,
                |_, chunk| {
                    let mut row_f64 = vec![0.0_f64; nx_out];
                    for row in chunk.chunks_exact_mut(nx_out) {
                        for (d, &s) in row_f64.iter_mut().zip(row.iter()) {
                            *d = s as f64;
                        }
                        subtract_polynomial_baseline(&mut row_f64, bc);
                        for (d, &s) in row.iter_mut().zip(row_f64.iter()) {
                            *d = s as f32;
                        }
                    }
                },
            );
        }
    }

    Ok(Spectrum2D {
        name: fid.name.clone(),
        axis_x: axis_x_out,
        axis_y: axis_y_out,
        data,
    })
}

/// Process a 3D FID (`Fid3D`, uniform or 2D indirect NUS) into a `Spectrum3D`.
pub fn process_fid_3d(
    fid: &Fid3D,
    exp: &str,
    recipe: &ProcessingRecipe,
) -> Result<Spectrum3D, String> {
    let mut rec = recipe.clone();
    process_fid_3d_mut(fid, exp, &mut rec)
}

/// Build indirect-focused 1D complex direct (`1H`) frequency traces for 3D blind autophasing.
///
/// On low-SNR 3D cubes (especially `HNCACB` where `CA(+)` and `CB(-)` cancel at `t_C = 0` and any
/// single raw `(k2, c2, k1, c1)` 1D trace has `SNR ~ 1–2`), raw time-slice sampling is dominated
/// by thermal noise. Applying a real-weight 2D indirect sine-bell DFT over `(k2 < m2, k1 < m1)`
/// separately to `Re_x` and `Im_x` preserves the exact direct complex phase `exp(+i phi_x(ix))`
/// (modulo `180 deg`) with zero cross-mixing of `Re_x`/`Im_x` and zero dependence on unknown
/// indirect phases, while boosting amide peak SNR by `~sqrt(m1 * m2)` (`15x–25x`) and separating
/// `CA(+)` from `CB(-)` in indirect frequency space.
fn focused_direct_traces_3d(
    fid: &Fid3D,
    dim_x: &DimRecipe,
    conj_y: bool,
    conj_z: bool,
    ap_cfg: &AutoPhaseConfig,
) -> (Option<AxisPpm>, Vec<Vec<Complex64>>) {
    let n1 = fid.axis_y.complex_points;
    let n2 = fid.axis_z.complex_points;
    let cap = if fid.nus_mask.is_some() { 48 } else { 24 };
    let m1 = n1.min(cap).max(1);
    let m2 = n2.min(cap).max(1);
    let p1_ch = if fid.axis_y.quad_mode == QuadMode::Qf {
        1
    } else {
        2
    };
    let p2_ch = if fid.axis_z.quad_mode == QuadMode::Qf {
        1
    } else {
        2
    };

    let mut coords: Vec<(usize, usize)> = Vec::new();
    for k2 in 0..m2 {
        for k1 in 0..m1 {
            if fid.nus_mask.as_ref().is_none_or(|mask| mask[k2 * n1 + k1]) {
                coords.push((k2, k1));
            }
        }
    }
    if coords.is_empty() {
        if let Some(first_idx) = fid
            .nus_mask
            .as_ref()
            .and_then(|m| m.iter().position(|&b| b))
        {
            coords.push((first_idx / n1, first_idx % n1));
        } else if n1 > 0 && n2 > 0 {
            coords.push((0, 0));
        }
    }
    if coords.is_empty() {
        return (None, Vec::new());
    }

    let sub_per_coord = p2_ch * p1_ch;
    let total_tr = coords.len() * sub_per_coord;
    let mut tr_specs: Vec<Vec<Complex64>> = vec![Vec::new(); total_tr];

    parallel_chunks_mut(
        &mut tr_specs,
        default_chunk_size(total_tr),
        |base_idx, chunk| {
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            for (local, slot) in chunk.iter_mut().enumerate() {
                let idx = base_idx + local;
                let coord_i = idx / sub_per_coord;
                let rem = idx % sub_per_coord;
                let c2 = rem / p1_ch;
                let c1 = rem % p1_ch;
                let (k2, k1) = coords[coord_i];
                transform_trace_steps_1_to_5(
                    fid.trace(k2, c2, k1, c1),
                    &fid.axis_x,
                    dim_x,
                    &mut wt,
                    &mut wf,
                    &mut sc,
                );
                apply_optional_hilbert_proc_sign(&mut sc, dim_x.hilbert);
                *slot = conj_trace_for_ops(&sc);
            }
        },
    );

    let si_x = tr_specs[0].len();
    if si_x == 0 {
        return (None, Vec::new());
    }
    let ax_x_full = build_full_axis_ppm(&fid.axis_x, si_x);

    if coords.len() < 4 {
        return (Some(ax_x_full), tr_specs);
    }

    let is_excluded = |ppm: f64| -> bool {
        ap_cfg.water_mask_ppm.is_some_and(|(a, b)| {
            let (lo, hi) = (a.min(b), a.max(b));
            ppm >= lo && ppm <= hi
        })
    };
    let mut cand_ix: Vec<usize> = (0..si_x)
        .filter(|&ix| {
            let ppm = ax_x_full.index_to_ppm(ix as f64);
            if is_excluded(ppm) {
                return false;
            }
            if let Some((r0, r1)) = ap_cfg.direct_roi_ppm {
                let (lo, hi) = (r0.min(r1), r0.max(r1));
                ppm >= lo && ppm <= hi
            } else {
                true
            }
        })
        .collect();
    if cand_ix.len() < 8 {
        cand_ix = (0..si_x)
            .filter(|&ix| !is_excluded(ax_x_full.index_to_ppm(ix as f64)))
            .collect();
    }
    if cand_ix.is_empty() {
        cand_ix = (0..si_x).collect();
    }
    let stride = (cand_ix.len() / 64).max(1);
    let sample_ix: Vec<usize> = cand_ix.into_iter().step_by(stride).take(64).collect();

    let g1 = (2 * m1).max(2);
    let g2 = (2 * m2).max(2);
    let sy = if conj_y { -1.0 } else { 1.0 };
    let sz = if conj_z { -1.0 } else { 1.0 };

    let mut w1 = vec![1.0_f64; m1];
    if m1 > 1 {
        for (k1, w) in w1.iter_mut().enumerate() {
            *w = (0.5 * PI * (k1 as f64) / (m1 as f64)).cos();
        }
        w1[0] *= 0.5;
    }
    let mut w2 = vec![1.0_f64; m2];
    if m2 > 1 {
        for (k2, w) in w2.iter_mut().enumerate() {
            *w = (0.5 * PI * (k2 as f64) / (m2 as f64)).cos();
        }
        w2[0] *= 0.5;
    }

    let mut by = vec![[0.0_f64; 4]; g1 * m1];
    for iy in 0..g1 {
        for k1 in 0..m1 {
            let theta = -2.0 * PI * (k1 as f64) * (iy as f64) / (g1 as f64);
            let (s, c) = theta.sin_cos();
            let wk = w1[k1];
            by[iy * m1 + k1] = [wk * c, -wk * sy * s, wk * s, wk * sy * c];
        }
    }
    let mut bz = vec![[0.0_f64; 4]; g2 * m2];
    for iz in 0..g2 {
        for k2 in 0..m2 {
            let theta = -2.0 * PI * (k2 as f64) * (iz as f64) / (g2 as f64);
            let (s, c) = theta.sin_cos();
            let wk = w2[k2];
            bz[iz * m2 + k2] = [wk * c, -wk * sz * s, wk * s, wk * sz * c];
        }
    }

    let mut voxel_e = vec![0.0_f64; g2 * g1 * 4];
    let mut coord_map = vec![usize::MAX; m2 * m1];
    for (ci, &(k2, k1)) in coords.iter().enumerate() {
        if k2 < m2 && k1 < m1 {
            coord_map[k2 * m1 + k1] = ci;
        }
    }

    let mut after_y = vec![[[0.0_f64; 2]; 4]; m2 * g1];
    for &ix in &sample_ix {
        after_y.fill([[0.0, 0.0]; 4]);
        for k2 in 0..m2 {
            for k1 in 0..m1 {
                let ci = coord_map[k2 * m1 + k1];
                if ci == usize::MAX {
                    continue;
                }
                let base = ci * sub_per_coord;
                let z00 = tr_specs[base][ix];
                let z01 = if p1_ch > 1 {
                    tr_specs[base + 1][ix]
                } else {
                    [0.0, 0.0]
                };
                let z10 = if p2_ch > 1 {
                    tr_specs[base + p1_ch][ix]
                } else {
                    [0.0, 0.0]
                };
                let z11 = if p2_ch > 1 && p1_ch > 1 {
                    tr_specs[base + p1_ch + 1][ix]
                } else {
                    [0.0, 0.0]
                };
                for iy in 0..g1 {
                    let b = by[iy * m1 + k1];
                    let dst = &mut after_y[k2 * g1 + iy];
                    dst[0][0] += b[0] * z00[0] + b[1] * z01[0];
                    dst[0][1] += b[0] * z00[1] + b[1] * z01[1];
                    dst[1][0] += b[2] * z00[0] + b[3] * z01[0];
                    dst[1][1] += b[2] * z00[1] + b[3] * z01[1];
                    if p2_ch > 1 {
                        dst[2][0] += b[0] * z10[0] + b[1] * z11[0];
                        dst[2][1] += b[0] * z10[1] + b[1] * z11[1];
                        dst[3][0] += b[2] * z10[0] + b[3] * z11[0];
                        dst[3][1] += b[2] * z10[1] + b[3] * z11[1];
                    }
                }
            }
        }

        for iz in 0..g2 {
            for iy in 0..g1 {
                let mut acc = [[0.0_f64; 2]; 4];
                for k2 in 0..m2 {
                    let b = bz[iz * m2 + k2];
                    let src = &after_y[k2 * g1 + iy];
                    acc[0][0] += b[0] * src[0][0] + b[1] * src[2][0];
                    acc[0][1] += b[0] * src[0][1] + b[1] * src[2][1];
                    acc[1][0] += b[0] * src[1][0] + b[1] * src[3][0];
                    acc[1][1] += b[0] * src[1][1] + b[1] * src[3][1];
                    acc[2][0] += b[2] * src[0][0] + b[3] * src[2][0];
                    acc[2][1] += b[2] * src[0][1] + b[3] * src[2][1];
                    acc[3][0] += b[2] * src[1][0] + b[3] * src[3][0];
                    acc[3][1] += b[2] * src[1][1] + b[3] * src[3][1];
                }
                let vbase = (iz * g1 + iy) * 4;
                for comp in 0..4 {
                    voxel_e[vbase + comp] +=
                        acc[comp][0] * acc[comp][0] + acc[comp][1] * acc[comp][1];
                }
            }
        }
    }

    let mut ranked: Vec<(f64, usize, usize, usize)> = Vec::with_capacity(g2 * g1 * 4);
    for iz in 0..g2 {
        for iy in 0..g1 {
            let vbase = (iz * g1 + iy) * 4;
            for comp in 0..4 {
                let e = voxel_e[vbase + comp];
                if e > 0.0 {
                    ranked.push((e, iz, iy, comp));
                }
            }
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut chosen: Vec<(usize, usize, usize)> = Vec::new();
    for &(_, iz, iy, comp) in &ranked {
        if chosen.iter().all(|&(qz, qy, _)| qz != iz || qy != iy) {
            chosen.push((iz, iy, comp));
            if chosen.len() >= 24 {
                break;
            }
        }
    }

    let mut out_traces: Vec<Vec<Complex64>> = Vec::with_capacity(1 + chosen.len());
    out_traces.push(tr_specs[0].clone());

    let mut weights = vec![0.0_f64; total_tr];
    for (iz, iy, comp) in chosen {
        let comp_z = comp / 2;
        let comp_y = comp % 2;
        let mut w_norm2 = 0.0_f64;
        weights.fill(0.0);
        for (ci, &(k2, k1)) in coords.iter().enumerate() {
            if k2 >= m2 || k1 >= m1 {
                continue;
            }
            let bz_pair = if comp_z == 0 {
                [bz[iz * m2 + k2][0], bz[iz * m2 + k2][1]]
            } else {
                [bz[iz * m2 + k2][2], bz[iz * m2 + k2][3]]
            };
            let by_pair = if comp_y == 0 {
                [by[iy * m1 + k1][0], by[iy * m1 + k1][1]]
            } else {
                [by[iy * m1 + k1][2], by[iy * m1 + k1][3]]
            };
            for c2 in 0..p2_ch {
                for c1 in 0..p1_ch {
                    let w = bz_pair[c2] * by_pair[c1];
                    weights[ci * sub_per_coord + c2 * p1_ch + c1] = w;
                    w_norm2 += w * w;
                }
            }
        }
        if w_norm2 <= 1e-30 {
            continue;
        }
        let inv_norm = 1.0 / w_norm2.sqrt();
        let mut focused = vec![[0.0_f64; 2]; si_x];
        for (t_idx, &w_raw) in weights.iter().enumerate() {
            let w = w_raw * inv_norm;
            if w.abs() <= 1e-15 {
                continue;
            }
            let src = &tr_specs[t_idx];
            for (dst, &s) in focused.iter_mut().zip(src.iter()) {
                dst[0] += w * s[0];
                dst[1] += w * s[1];
            }
        }
        out_traces.push(focused);
    }

    (Some(ax_x_full), out_traces)
}

/// Detect whether an un-annotated broad-band aliphatic `13CA/CB` axis (`CBCACONH` / `HNCACB`) is
/// carrier-mirrored (`conjugate_for_fft` inverted due to missing `procpar` `f1coef` or raw `ser`
/// custom pulse program) using universal BMRB backbone `(CA, CB)` chemical shift bounds:
/// - Canonical `CA` (`51.5..64.8 ppm`, ~92% of residues) and `Ala CB` (`15.5..22.2 ppm`, ~8%) vs
/// - Forbidden downfield region `72.2..79.5 ppm` (zero canonical amino-acid `CA`/`CB`, populated
///   only when `Ala CB` mirrors across a `43..48 ppm` carrier!).
fn is_mirrored_aliphatic_cacb_3d(
    spec: &Spectrum3D,
    full_c_axis: &AxisPpm,
    c0_offset: usize,
) -> bool {
    let nz = spec.axis_z.size;
    let ny = spec.axis_y.size;
    let nx = spec.axis_x.size;
    let si_c = full_c_axis.size;
    if nz < 16 || ny < 4 || nx < 8 || si_c < 16 || spec.data.len() != nz * ny * nx {
        return false;
    }
    let p_a = spec.axis_z.index_to_ppm(0.0);
    let p_b = spec.axis_z.index_to_ppm((nz - 1) as f64);
    let (z_min, z_max) = (p_a.min(p_b), p_a.max(p_b));
    if (z_max - z_min) < 45.0 || z_min > 21.5 || z_max < 73.0 {
        return false;
    }

    let step = (spec.data.len() / 32_768).max(1);
    let mut abs_vals: Vec<f32> = spec
        .data
        .iter()
        .step_by(step)
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if abs_vals.len() < 64 {
        return false;
    }
    let mid = abs_vals.len() / 2;
    let (_, med, _) = abs_vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let sigma = ((*med as f64) * 1.4826).max(1e-12);
    let thr = (6.0 * sigma) as f32;

    let prior = |ppm: f64| -> f64 {
        if (51.5..=64.8).contains(&ppm) {
            1.0
        } else if (15.5..=22.2).contains(&ppm) {
            1.4
        } else if (67.0..=71.5).contains(&ppm) {
            0.6
        } else if (72.2..=79.5).contains(&ppm) {
            -2.5
        } else if (10.0..=14.8).contains(&ppm) {
            -2.0
        } else {
            0.0
        }
    };

    let plane = ny * nx;
    let mut n_peaks = 0usize;
    let mut score_orig = 0.0_f64;
    let mut score_rev = 0.0_f64;
    let mut ca_orig = 0.0_f64;
    let mut ca_rev = 0.0_f64;

    for iz in 1..(nz - 1) {
        let iz_full = c0_offset + iz;
        let iz_rev = (si_c - (iz_full % si_c)) % si_c;
        let ppm_orig = full_c_axis.index_to_ppm(iz_full as f64);
        let ppm_rev = full_c_axis.index_to_ppm(iz_rev as f64);
        let pr_o = prior(ppm_orig);
        let pr_r = prior(ppm_rev);
        let in_ca_o = (51.5..=64.8).contains(&ppm_orig);
        let in_ca_r = (51.5..=64.8).contains(&ppm_rev);

        for iy in 1..(ny - 1) {
            let row_base = iz * plane + iy * nx;
            for ix in 1..(nx - 1) {
                let idx = row_base + ix;
                let av = spec.data[idx].abs();
                if av < thr {
                    continue;
                }
                if av >= spec.data[idx - 1].abs()
                    && av > spec.data[idx + 1].abs()
                    && av >= spec.data[idx - nx].abs()
                    && av >= spec.data[idx + nx].abs()
                    && av >= spec.data[idx - plane].abs()
                    && av >= spec.data[idx + plane].abs()
                {
                    let w = (((av as f64) / sigma) - 5.5).clamp(0.5, 6.0);
                    n_peaks += 1;
                    score_orig += w * pr_o;
                    score_rev += w * pr_r;
                    if in_ca_o {
                        ca_orig += w;
                    }
                    if in_ca_r {
                        ca_rev += w;
                    }
                }
            }
        }
    }

    n_peaks >= 12
        && score_rev > 20.0
        && score_rev > score_orig * 1.6 + 12.0
        && ca_rev > ca_orig * 1.35
}

/// Process a 3D FID while recording any automatically solved `(p0_deg, p1_deg)` back into `recipe`.
pub fn process_fid_3d_mut(
    fid: &Fid3D,
    exp: &str,
    recipe: &mut ProcessingRecipe,
) -> Result<Spectrum3D, String> {
    process_fid_3d_inner(fid, exp, recipe, true)
}

fn process_fid_3d_inner(
    fid: &Fid3D,
    exp: &str,
    recipe: &mut ProcessingRecipe,
    allow_cacb_unmirror: bool,
) -> Result<Spectrum3D, String> {
    if recipe.dims.len() < 3 {
        return Err(format!(
            "ProcessingRecipe has {} dimensions, expected 3 for Fid3D",
            recipe.dims.len()
        ));
    }

    let n0 = fid.axis_x.complex_points;
    let n1 = fid.axis_y.complex_points;
    let n2 = fid.axis_z.complex_points;
    let expected_len = n2 * 2 * n1 * 2 * n0;
    if fid.data.len() < expected_len {
        return Err(format!(
            "Fid3D data buffer truncated: got {}, expected {expected_len}",
            fid.data.len()
        ));
    }
    if let Some(ref mask) = fid.nus_mask {
        if mask.len() != n2 * n1 {
            return Err(format!(
                "Fid3D nus_mask length mismatch: got {}, expected n2*n1={}",
                mask.len(),
                n2 * n1
            ));
        }
        validate_nus_states_axis(&fid.axis_y, "3D indirect F2")?;
        validate_nus_states_axis(&fid.axis_z, "3D indirect F1")?;
    }

    let ap_cfg = auto_phase_cfg_for_exp(exp);
    let any_dim_auto = recipe.dims[..3].iter().any(|d| {
        d.p0_deg.is_none()
            && d.p1_deg.is_none()
            && matches!(
                d.phase_source,
                PhaseSource::Auto | PhaseSource::AutoChen2002
            )
    });

    // ------------------------------------------------------------------------
    // Step A: Direct 1H (Dim 0) processing + imaginary discard + 1H ROI crop
    // ------------------------------------------------------------------------
    let dim_x = recipe.dims[0].clone();
    let si_x_est = dim_x.si.max(n0).max(2);
    let full_axis_x = build_full_axis_ppm(&fid.axis_x, si_x_est);
    let (x0, x1, axis_x_out) = compute_axis_roi(&full_axis_x, dim_x.roi_ppm)?;
    let nx_out = axis_x_out.size;

    let (p0_x, p1_x) = match (dim_x.p0_deg, dim_x.p1_deg, dim_x.phase_source) {
        (Some(p0), Some(p1), _) => (p0, p1),
        (Some(p0), None, _) => (p0, 0.0),
        (None, Some(p1), _) => (0.0, p1),
        (None, None, PhaseSource::DefaultZero) => (0.0, 0.0),
        (None, None, PhaseSource::Auto) => {
            let (ax_rep, sample_x) = focused_direct_traces_3d(
                fid,
                &dim_x,
                recipe.dims[1].conjugate_for_fft,
                recipe.dims[2].conjugate_for_fft,
                &ap_cfg,
            );
            if let Some(ax_x_full) = ax_rep {
                let mut cfg_direct = ap_cfg.clone();
                if dim_x.remove_digital_filter && dim_x.group_delay > 0.0 {
                    cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(18.0);
                } else if fid.axis_x.group_delay == 0.0
                    && fid.axis_x.stored_proc.p1_deg.unwrap_or(0.0).abs() <= 20.0
                {
                    cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(25.0);
                }
                let sol = auto_phase_direct_nd(&sample_x, &ax_x_full, &cfg_direct);
                (sol.p0_deg, sol.p1_deg)
            } else {
                (0.0, 0.0)
            }
        }
        (None, None, _) => {
            let first_idx = fid
                .nus_mask
                .as_ref()
                .and_then(|m| m.iter().position(|&b| b))
                .unwrap_or(0);
            let fk2 = first_idx / n1;
            let fk1 = first_idx % n1;
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut rep_spec = Vec::new();
            transform_trace_steps_1_to_5(
                fid.trace(fk2, 0, fk1, 0),
                &fid.axis_x,
                &dim_x,
                &mut wt,
                &mut wf,
                &mut rep_spec,
            );
            apply_optional_hilbert_proc_sign(&mut rep_spec, dim_x.hilbert);
            let ax_rep = build_full_axis_ppm(&fid.axis_x, rep_spec.len());
            resolve_dim_phase(&dim_x, &ax_rep, &rep_spec)
        }
    };
    recipe.dims[0].p0_deg = Some(p0_x);
    recipe.dims[0].p1_deg = Some(p1_x);

    // Total hypercomplex direct traces = `n2 * 2 * n1 * 2`
    let n_direct_traces = n2 * 2 * n1 * 2;
    let mut direct_cropped = vec![0.0_f64; n_direct_traces * nx_out];
    let nus_mask_ref = fid.nus_mask.as_deref();

    parallel_chunks_mut(
        &mut direct_cropped,
        default_chunk_size(n_direct_traces) * nx_out,
        |base_elem, chunk| {
            let base_tr = base_elem / nx_out;
            let cnt_tr = chunk.len() / nx_out;
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut real_full = Vec::new();

            for local in 0..cnt_tr {
                let tr_idx = base_tr + local;
                let c1 = tr_idx % 2;
                let rem1 = tr_idx / 2;
                let k1 = rem1 % n1;
                let rem2 = rem1 / n1;
                let c2 = rem2 % 2;
                let k2 = rem2 / 2;

                let dst = &mut chunk[local * nx_out..(local + 1) * nx_out];
                if let Some(mask) = nus_mask_ref {
                    if !mask[k2 * n1 + k1] {
                        dst.fill(0.0);
                        continue;
                    }
                }
                let raw_tr = fid.trace(k2, c2, k1, c1);
                transform_trace_steps_1_to_5(
                    raw_tr,
                    &fid.axis_x,
                    &dim_x,
                    &mut wt,
                    &mut wf,
                    &mut sc,
                );
                apply_optional_hilbert_proc_sign(&mut sc, dim_x.hilbert);
                real_full.resize(sc.len(), 0.0);
                apply_phase_and_discard_imag(&sc, p0_x, p1_x, &mut real_full);
                dst.copy_from_slice(&real_full[x0..=x1]);
            }
        },
    );

    // ------------------------------------------------------------------------
    // Step B: Indirect `(F2 = dim 1, F1 = dim 2)` per direct `1H` column `ix in 0..nx_out`
    // Parallelized across `1H` columns via `std::thread::scope`!
    // ------------------------------------------------------------------------
    let mut dim_y = recipe.dims[1].clone();
    let mut dim_z = recipe.dims[2].clone();
    let si_y = dim_y.si.max(n1).max(2);
    let si_z = dim_z.si.max(n2).max(2);
    let full_axis_y = build_full_axis_ppm(&fid.axis_y, si_y);
    let full_axis_z = build_full_axis_ppm(&fid.axis_z, si_z);
    let (y0, y1, axis_y_out) = compute_axis_roi(&full_axis_y, dim_y.roi_ppm)?;
    let (z0, z1, axis_z_out) = compute_axis_roi(&full_axis_z, dim_z.roi_ppm)?;
    let ny_out = axis_y_out.size;
    let nz_out = axis_z_out.size;
    let nus_cfg = recipe.nus.clone().unwrap_or_default();

    let need_auto_y = dim_y.p0_deg.is_none()
        && dim_y.p1_deg.is_none()
        && !matches!(dim_y.phase_source, PhaseSource::DefaultZero);
    let need_auto_z = dim_z.p0_deg.is_none()
        && dim_z.p1_deg.is_none()
        && !matches!(dim_z.phase_source, PhaseSource::DefaultZero);

    let (mut p0_y, p1_y) = (dim_y.p0_deg.unwrap_or(0.0), dim_y.p1_deg.unwrap_or(0.0));
    let (mut p0_z, p1_z) = (dim_z.p0_deg.unwrap_or(0.0), dim_z.p1_deg.unwrap_or(0.0));

    let (p0_y_solved, p1_y_solved, p0_z_solved, p1_z_solved) = if need_auto_y || need_auto_z {
        // Select up to 24 distinct `1H` peak-apex planes `ix` to solve indirect phases `(dim_y, dim_z)`
        let mut ix_energy = vec![0.0_f64; nx_out];
        for k2 in 0..n2 {
            for k1 in 0..n1 {
                let b00 = (((k2 * 2) * n1 + k1) * 2) * nx_out;
                let b01 = b00 + nx_out;
                let b10 = (((k2 * 2 + 1) * n1 + k1) * 2) * nx_out;
                let b11 = b10 + nx_out;
                for ix in 0..nx_out {
                    let q0 = direct_cropped[b00 + ix];
                    let q1 = direct_cropped[b01 + ix];
                    let q2 = direct_cropped[b10 + ix];
                    let q3 = direct_cropped[b11 + ix];
                    ix_energy[ix] += q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
                }
            }
        }
        if dim_x.roi_ppm.is_none() && fid.axis_x.label.to_ascii_uppercase().contains('H') {
            let mut masked = ix_energy.clone();
            for (ix, val) in masked.iter_mut().enumerate() {
                let ppm = axis_x_out.index_to_ppm(ix as f64);
                let in_water = ap_cfg.water_mask_ppm.is_some_and(|(a, b)| {
                    let (lo, hi) = (a.min(b), a.max(b));
                    ppm >= lo && ppm <= hi
                });
                let out_amide = ap_cfg.direct_roi_ppm.is_some_and(|(r0, r1)| {
                    let (lo, hi) = (r0.min(r1), r0.max(r1));
                    ppm < lo || ppm > hi
                });
                if in_water || out_amide {
                    *val = 0.0;
                }
            }
            if masked.iter().any(|&v| v > 0.0) {
                ix_energy = masked;
            }
        }
        let sel_ix = select_apex_indices(&ix_energy, 3, 24);
        let n_sel = sel_ix.len();
        let mut sel_quads = vec![vec![[0.0_f64; 4]; n2 * n1]; n_sel];
        parallel_chunks_mut(
            &mut sel_quads,
            default_chunk_size(n_sel),
            |base_p, chunk| {
                for (local, qp) in chunk.iter_mut().enumerate() {
                    let ix = sel_ix[base_p + local];
                    for k2 in 0..n2 {
                        for k1 in 0..n1 {
                            let b00 = (((k2 * 2) * n1 + k1) * 2) * nx_out + ix;
                            qp[k2 * n1 + k1] = [
                                direct_cropped[b00],
                                direct_cropped[b00 + nx_out],
                                direct_cropped[((k2 * 2 + 1) * n1 + k1) * 2 * nx_out + ix],
                                direct_cropped[(((k2 * 2 + 1) * n1 + k1) * 2 + 1) * nx_out + ix],
                            ];
                        }
                    }
                    if let Some(mask) = nus_mask_ref {
                        *qp = reconstruct_nus_2d_hypercomplex_ist(qp, n2, n1, mask, &nus_cfg);
                    }
                }
            },
        );

        let (sy0, sy1) = if need_auto_y {
            let mut cand_y_cols = Vec::new();
            let mut td_y = vec![[0.0, 0.0]; n1];
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut ax_y_rep: Option<AxisPpm> = None;
            // Select the single dominant `(k2_best, c2_best)` slice (preferring `k2 = 0` so all columns
            // share a coherent `F1` sign without `k2 > 0` inter-residue cosine/sine sign alternation).
            let mut slice_e = Vec::new();
            for k2 in 0..n2.min(4) {
                for c2 in 0..2 {
                    let mut se = 0.0_f64;
                    for qp in &sel_quads {
                        for k1 in 0..n1 {
                            let q = qp[k2 * n1 + k1];
                            let c = if c2 == 0 { [q[0], q[1]] } else { [q[2], q[3]] };
                            se += c[0] * c[0] + c[1] * c[1];
                        }
                    }
                    slice_e.push((se, k2, c2));
                }
            }
            let max_se = slice_e.iter().map(|s| s.0).fold(0.0_f64, f64::max);
            let best_k2_0 = slice_e
                .iter()
                .filter(|s| s.1 == 0)
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .copied();
            let (_, k2_best, c2_best) = match best_k2_0 {
                Some(s0) if s0.0 >= 0.20 * max_se => s0,
                _ => slice_e
                    .into_iter()
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap_or((0.0, 0, 0)),
            };

            for qp in &sel_quads {
                for k1 in 0..n1 {
                    let q = qp[k2_best * n1 + k1];
                    td_y[k1] = if c2_best == 0 {
                        [q[0], q[1]]
                    } else {
                        [q[2], q[3]]
                    };
                }
                transform_trace_steps_1_to_5(&td_y, &fid.axis_y, &dim_y, &mut wt, &mut wf, &mut sc);
                apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                if ax_y_rep.is_none() {
                    ax_y_rep = Some(build_full_axis_ppm(&fid.axis_y, sc.len()));
                }
                cand_y_cols.push(conj_trace_for_ops(&sc));
            }
            if let Some(ax_y_full) = ax_y_rep {
                let mut cfg_y = ap_cfg.clone();
                if cfg_y.polarity == PolarityMode::Biphasic
                    && !fid.axis_y.label.to_ascii_uppercase().contains('C')
                {
                    cfg_y.polarity = PolarityMode::AllPositive;
                }
                let sol = auto_phase_indirect_nd(&cand_y_cols, &ax_y_full, &cfg_y);
                (sol.p0_deg, sol.p1_deg)
            } else {
                (0.0, 0.0)
            }
        } else {
            (p0_y, p1_y)
        };
        if need_auto_y && sy1.abs() > 15.0 {
            dim_y.fcor = 1.0;
            recipe.dims[1].fcor = 1.0;
        }

        let (sz0, sz1) = if need_auto_z {
            // Transform `dim_y` across the `n_sel` apex planes using `(sy0, sy1)` and select top `(plane, iy)` ridges
            let mut sel_after_y = vec![vec![0.0_f64; n2 * 2 * ny_out]; n_sel];
            let mut td_y = vec![[0.0, 0.0]; n1];
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut real_full = Vec::new();
            for (pi, qp) in sel_quads.iter().enumerate() {
                let ay = &mut sel_after_y[pi];
                for k2 in 0..n2 {
                    for k1 in 0..n1 {
                        let q = qp[k2 * n1 + k1];
                        td_y[k1] = [q[0], q[1]];
                    }
                    transform_trace_steps_1_to_5(
                        &td_y,
                        &fid.axis_y,
                        &dim_y,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                    real_full.resize(sc.len(), 0.0);
                    apply_phase_and_discard_imag(&sc, sy0, sy1, &mut real_full);
                    if let Some(ref bc) = dim_y.baseline {
                        subtract_polynomial_baseline(&mut real_full, bc);
                    }
                    ay[(k2 * 2) * ny_out..(k2 * 2 + 1) * ny_out]
                        .copy_from_slice(&real_full[y0..=y1]);

                    for k1 in 0..n1 {
                        let q = qp[k2 * n1 + k1];
                        td_y[k1] = [q[2], q[3]];
                    }
                    transform_trace_steps_1_to_5(
                        &td_y,
                        &fid.axis_y,
                        &dim_y,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                    real_full.resize(sc.len(), 0.0);
                    apply_phase_and_discard_imag(&sc, sy0, sy1, &mut real_full);
                    if let Some(ref bc) = dim_y.baseline {
                        subtract_polynomial_baseline(&mut real_full, bc);
                    }
                    ay[(k2 * 2 + 1) * ny_out..(k2 * 2 + 2) * ny_out]
                        .copy_from_slice(&real_full[y0..=y1]);
                }
            }

            // When `axis_y` is `13C` in `Biphasic` (`HNCACB`, e.g. Varian BioPack) and `axis_z` is `15N`,
            // restrict ridge selection to the `CA` band (`48..60 ppm`) so all `15N` columns share one sign!
            let restrict_y_to_ca = ap_cfg.polarity == PolarityMode::Biphasic
                && fid.axis_y.label.to_ascii_uppercase().contains('C')
                && !fid.axis_z.label.to_ascii_uppercase().contains('C');
            let (ca_lo, ca_hi) = (
                ap_cfg
                    .biphasic_pos_band_ppm
                    .0
                    .min(ap_cfg.biphasic_pos_band_ppm.1),
                ap_cfg
                    .biphasic_pos_band_ppm
                    .0
                    .max(ap_cfg.biphasic_pos_band_ppm.1),
            );
            let mut ridges: Vec<(f64, usize, usize)> = Vec::new();
            let mut y_col_e = vec![0.0_f64; ny_out];
            for (pi, ay) in sel_after_y.iter().enumerate() {
                for iy in 0..ny_out {
                    if restrict_y_to_ca {
                        let y_ppm = axis_y_out.index_to_ppm(iy as f64);
                        if y_ppm < ca_lo || y_ppm > ca_hi {
                            y_col_e[iy] = 0.0;
                            continue;
                        }
                    }
                    let mut e = 0.0;
                    for k2 in 0..n2 {
                        let re = ay[(k2 * 2) * ny_out + iy];
                        let im = ay[(k2 * 2 + 1) * ny_out + iy];
                        e += re * re + im * im;
                    }
                    y_col_e[iy] = e;
                }
                for iy in 1..ny_out.saturating_sub(1) {
                    let e0 = y_col_e[iy];
                    if e0 > 0.0 && e0 >= y_col_e[iy - 1] && e0 >= y_col_e[iy + 1] {
                        ridges.push((e0, pi, iy));
                    }
                }
            }
            ridges.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut chosen_ridges: Vec<(usize, usize)> = Vec::new();
            for &(_, pi, iy) in &ridges {
                if chosen_ridges
                    .iter()
                    .all(|&(qp, qy)| qp != pi || iy.abs_diff(qy) >= 2)
                {
                    chosen_ridges.push((pi, iy));
                    if chosen_ridges.len() >= 64 {
                        break;
                    }
                }
            }
            if chosen_ridges.len() < 4 {
                for (pi, ay) in sel_after_y.iter().enumerate() {
                    for iy in 0..ny_out {
                        let mut e = 0.0;
                        for k2 in 0..n2 {
                            let re = ay[(k2 * 2) * ny_out + iy];
                            let im = ay[(k2 * 2 + 1) * ny_out + iy];
                            e += re * re + im * im;
                        }
                        ridges.push((e, pi, iy));
                    }
                }
                ridges.sort_by(|a, b| b.0.total_cmp(&a.0));
                chosen_ridges = ridges
                    .into_iter()
                    .take(64)
                    .map(|(_, p, y)| (p, y))
                    .collect();
            }

            let mut cand_z_cols = Vec::with_capacity(chosen_ridges.len());
            let mut td_z = vec![[0.0, 0.0]; n2];
            let mut ax_z_rep: Option<AxisPpm> = None;
            for (pi, iy) in chosen_ridges {
                let ay = &sel_after_y[pi];
                for k2 in 0..n2 {
                    td_z[k2] = [ay[(k2 * 2) * ny_out + iy], ay[(k2 * 2 + 1) * ny_out + iy]];
                }
                transform_trace_steps_1_to_5(&td_z, &fid.axis_z, &dim_z, &mut wt, &mut wf, &mut sc);
                apply_optional_hilbert_proc_sign(&mut sc, dim_z.hilbert);
                if ax_z_rep.is_none() {
                    ax_z_rep = Some(build_full_axis_ppm(&fid.axis_z, sc.len()));
                }
                cand_z_cols.push(conj_trace_for_ops(&sc));
            }
            if let Some(ax_z_full) = ax_z_rep {
                let mut cfg_z = ap_cfg.clone();
                if cfg_z.polarity == PolarityMode::Biphasic
                    && !fid.axis_z.label.to_ascii_uppercase().contains('C')
                {
                    cfg_z.polarity = PolarityMode::AllPositive;
                }
                let sol = auto_phase_indirect_nd(&cand_z_cols, &ax_z_full, &cfg_z);
                (sol.p0_deg, sol.p1_deg)
            } else {
                (0.0, 0.0)
            }
        } else {
            (p0_z, p1_z)
        };

        (sy0, sy1, sz0, sz1)
    } else {
        (p0_y, p1_y, p0_z, p1_z)
    };

    p0_y = p0_y_solved;
    p0_z = p0_z_solved;
    let p1_y = p1_y_solved;
    let p1_z = p1_z_solved;
    if need_auto_z && p1_z.abs() > 15.0 {
        dim_z.fcor = 1.0;
        recipe.dims[2].fcor = 1.0;
    }
    recipe.dims[1].p0_deg = Some(p0_y);
    recipe.dims[1].p1_deg = Some(p1_y);
    recipe.dims[2].p0_deg = Some(p0_z);
    recipe.dims[2].p1_deg = Some(p1_z);

    let plane_out_elems = nz_out * ny_out;
    let mut planes_by_x = vec![0.0_f32; nx_out * plane_out_elems];

    parallel_chunks_mut(
        &mut planes_by_x,
        default_chunk_size(nx_out) * plane_out_elems,
        |base_elem, chunk| {
            let base_ix = base_elem / plane_out_elems;
            let cnt_ix = chunk.len() / plane_out_elems;

            let mut quad_plane = vec![[0.0_f64; 4]; n2 * n1];
            // After F2 (`axis_y`) transform + imag discard + Y crop: `[n2 * 2 * ny_out]`
            let mut after_y = vec![0.0_f64; n2 * 2 * ny_out];
            let mut td_y = vec![[0.0, 0.0]; n1];
            let mut td_z = vec![[0.0, 0.0]; n2];
            let mut wt = Vec::new();
            let mut wf = Vec::new();
            let mut sc = Vec::new();
            let mut real_full = Vec::new();

            for local in 0..cnt_ix {
                let ix = base_ix + local;

                // Gather `(cc, cs, sc, ss)` for `(k2, k1)` at column `ix`
                for k2 in 0..n2 {
                    for k1 in 0..n1 {
                        let base_00 = (((k2 * 2) * n1 + k1) * 2) * nx_out + ix;
                        let base_01 = base_00 + nx_out;
                        let base_10 = (((k2 * 2 + 1) * n1 + k1) * 2) * nx_out + ix;
                        let base_11 = base_10 + nx_out;
                        quad_plane[k2 * n1 + k1] = [
                            direct_cropped[base_00],
                            direct_cropped[base_01],
                            direct_cropped[base_10],
                            direct_cropped[base_11],
                        ];
                    }
                }

                // Reconstruct 2D hypercomplex NUS plane if `nus_mask` is active
                if let Some(mask) = nus_mask_ref {
                    quad_plane =
                        reconstruct_nus_2d_hypercomplex_ist(&quad_plane, n2, n1, mask, &nus_cfg);
                }

                // Transform Dim 1 (`F2` = `axis_y`) for both `c2 = 0` (`cos_z`) and `c2 = 1` (`sin_z`)
                for k2 in 0..n2 {
                    // c2 = 0: `(cc, cs)`
                    for k1 in 0..n1 {
                        let q = quad_plane[k2 * n1 + k1];
                        td_y[k1] = [q[0], q[1]];
                    }
                    transform_trace_steps_1_to_5(
                        &td_y,
                        &fid.axis_y,
                        &dim_y,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                    real_full.resize(sc.len(), 0.0);
                    apply_phase_and_discard_imag(&sc, p0_y, p1_y, &mut real_full);
                    if let Some(ref bc) = dim_y.baseline {
                        subtract_polynomial_baseline(&mut real_full, bc);
                    }
                    after_y[(k2 * 2) * ny_out..(k2 * 2 + 1) * ny_out]
                        .copy_from_slice(&real_full[y0..=y1]);

                    // c2 = 1: `(sc, ss)`
                    for k1 in 0..n1 {
                        let q = quad_plane[k2 * n1 + k1];
                        td_y[k1] = [q[2], q[3]];
                    }
                    transform_trace_steps_1_to_5(
                        &td_y,
                        &fid.axis_y,
                        &dim_y,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_y.hilbert);
                    real_full.resize(sc.len(), 0.0);
                    apply_phase_and_discard_imag(&sc, p0_y, p1_y, &mut real_full);
                    if let Some(ref bc) = dim_y.baseline {
                        subtract_polynomial_baseline(&mut real_full, bc);
                    }
                    after_y[(k2 * 2 + 1) * ny_out..(k2 * 2 + 2) * ny_out]
                        .copy_from_slice(&real_full[y0..=y1]);
                }

                // Transform Dim 2 (`F1` = `axis_z`) across each `iy in 0..ny_out`
                let dst_plane = &mut chunk[local * plane_out_elems..(local + 1) * plane_out_elems];
                for iy in 0..ny_out {
                    for k2 in 0..n2 {
                        let re = after_y[(k2 * 2) * ny_out + iy];
                        let im = after_y[(k2 * 2 + 1) * ny_out + iy];
                        td_z[k2] = [re, im];
                    }
                    transform_trace_steps_1_to_5(
                        &td_z,
                        &fid.axis_z,
                        &dim_z,
                        &mut wt,
                        &mut wf,
                        &mut sc,
                    );
                    apply_optional_hilbert_proc_sign(&mut sc, dim_z.hilbert);
                    real_full.resize(sc.len(), 0.0);
                    apply_phase_and_discard_imag(&sc, p0_z, p1_z, &mut real_full);
                    if let Some(ref bc) = dim_z.baseline {
                        subtract_polynomial_baseline(&mut real_full, bc);
                    }
                    for (iz, &v) in real_full[z0..=z1].iter().enumerate() {
                        dst_plane[iz * ny_out + iy] = v as f32;
                    }
                }
            }
        },
    );

    // Scatter `planes_by_x[ix * (nz_out * ny_out) + iz * ny_out + iy]` -> `Spectrum3D[(iz * ny_out + iy) * nx_out + ix]`
    // Also ensure `Spectrum3D` canonical triple-resonance order `(X=1H, Y=15N, Z=13C)`:
    // If `axis_y_out` is `13C` and `axis_z_out` is `15N` (e.g. Varian BioPack where `ni` is 13C and `ni2` is 15N),
    // transpose `(Y, Z)` so `Spectrum3D` always has `Y = 15N, Z = 13C` matching `read_ucsf_3d` / `picker`!
    let swap_yz = axis_y_out.label.contains('C') && axis_z_out.label.contains('N');
    let run_x_baseline = !(dim_x.roi_ppm.is_none() && dim_x.remove_digital_filter);

    let mut out_spec = if !swap_yz {
        let mut data = vec![0.0_f32; nz_out * ny_out * nx_out];
        for ix in 0..nx_out {
            let src = &planes_by_x[ix * plane_out_elems..(ix + 1) * plane_out_elems];
            for iz in 0..nz_out {
                let src_row = &src[iz * ny_out..(iz + 1) * ny_out];
                let dst_base = iz * ny_out * nx_out + ix;
                for iy in 0..ny_out {
                    data[dst_base + iy * nx_out] = src_row[iy];
                }
            }
        }
        if let Some(ref bc) = dim_x.baseline {
            if bc.enabled && run_x_baseline && nx_out >= 8 {
                let n_rows = nz_out * ny_out;
                parallel_chunks_mut(
                    &mut data,
                    default_chunk_size(n_rows) * nx_out,
                    |_, chunk| {
                        let mut row_f64 = vec![0.0_f64; nx_out];
                        for row in chunk.chunks_exact_mut(nx_out) {
                            for (d, &s) in row_f64.iter_mut().zip(row.iter()) {
                                *d = s as f64;
                            }
                            subtract_polynomial_baseline(&mut row_f64, bc);
                            for (d, &s) in row.iter_mut().zip(row_f64.iter()) {
                                *d = s as f32;
                            }
                        }
                    },
                );
            }
        }
        Spectrum3D {
            name: fid.name.clone(),
            exp: exp.to_string(),
            axis_x: axis_x_out,
            axis_y: axis_y_out,
            axis_z: axis_z_out,
            data,
        }
    } else {
        // Transpose `(iz_n, iy_c)` -> `(iz_new = iy_c, iy_new = iz_n)`
        let new_ny = nz_out;
        let new_nz = ny_out;
        let mut data = vec![0.0_f32; new_nz * new_ny * nx_out];
        for ix in 0..nx_out {
            let src = &planes_by_x[ix * plane_out_elems..(ix + 1) * plane_out_elems];
            for iz_n in 0..nz_out {
                for iy_c in 0..ny_out {
                    let v = src[iz_n * ny_out + iy_c];
                    data[(iy_c * new_ny + iz_n) * nx_out + ix] = v;
                }
            }
        }
        if let Some(ref bc) = dim_x.baseline {
            if bc.enabled && run_x_baseline && nx_out >= 8 {
                let n_rows = new_nz * new_ny;
                parallel_chunks_mut(
                    &mut data,
                    default_chunk_size(n_rows) * nx_out,
                    |_, chunk| {
                        let mut row_f64 = vec![0.0_f64; nx_out];
                        for row in chunk.chunks_exact_mut(nx_out) {
                            for (d, &s) in row_f64.iter_mut().zip(row.iter()) {
                                *d = s as f64;
                            }
                            subtract_polynomial_baseline(&mut row_f64, bc);
                            for (d, &s) in row.iter_mut().zip(row_f64.iter()) {
                                *d = s as f32;
                            }
                        }
                    },
                );
            }
        }
        Spectrum3D {
            name: fid.name.clone(),
            exp: exp.to_string(),
            axis_x: axis_x_out,
            axis_y: axis_z_out,
            axis_z: axis_y_out,
            data,
        }
    };

    let any_explicit = recipe.dims[..3]
        .iter()
        .any(|d| d.phase_source == PhaseSource::Explicit);
    if allow_cacb_unmirror && !any_explicit {
        let exp_u = exp.trim().to_ascii_uppercase();
        if matches!(exp_u.as_str(), "CBCACONH" | "CBCA(CO)NH" | "HNCACB") {
            let (c_dim, full_c_axis, c0_off) = if swap_yz {
                (1usize, &full_axis_y, y0)
            } else {
                (2usize, &full_axis_z, z0)
            };
            if is_mirrored_aliphatic_cacb_3d(&out_spec, full_c_axis, c0_off) {
                recipe.dims[c_dim].conjugate_for_fft = !recipe.dims[c_dim].conjugate_for_fft;
                if need_auto_y {
                    recipe.dims[1].set_auto_phase();
                }
                if need_auto_z {
                    recipe.dims[2].set_auto_phase();
                }
                return process_fid_3d_inner(fid, exp, recipe, false);
            }
        }
    }
    if !any_explicit {
        if ap_cfg.polarity == PolarityMode::Biphasic {
            if lock_biphasic_ca_polarity_3d(&mut out_spec, ap_cfg.biphasic_pos_band_ppm) {
                let flip_dim = if swap_yz { 1 } else { 2 };
                let cur = recipe.dims[flip_dim].p0_deg.unwrap_or(0.0);
                recipe.dims[flip_dim].p0_deg = Some(wrap_deg_180(cur + 180.0));
            }
        } else if (any_dim_auto || need_auto_y || need_auto_z)
            && ap_cfg.polarity == PolarityMode::AllPositive
            && lock_all_positive_polarity_3d(&mut out_spec)
        {
            let flip_dim = if swap_yz { 1 } else { 2 };
            let cur = recipe.dims[flip_dim].p0_deg.unwrap_or(0.0);
            recipe.dims[flip_dim].p0_deg = Some(wrap_deg_180(cur + 180.0));
        }
    }

    Ok(out_spec)
}

/// Align a processed 3D `(1H, 15N, 13C)` spectrum (`spec_3d`) to a reference 2D `(1H, 15N)` HSQC
/// (`hsqc_2d`) against small rigid `(dH, dN)` carrier/temperature drift across multi-spectrometer
/// experiment bundles, using unlabelled 2D sky-projection capped-SNR bilinear cross-correlation.
///
/// Updates `spec_3d.axis_x.first_ppm += dh` and `spec_3d.axis_y.first_ppm += dn` in place when a
/// statistically significant peak alignment shift is detected within `[-max_dh_ppm, +max_dh_ppm]`
/// and `[-max_dn_ppm, +max_dn_ppm]`, returning `(applied_dh_ppm, applied_dn_ppm)`.
pub fn register_3d_hn_to_hsqc_2d(
    spec_3d: &mut Spectrum3D,
    hsqc_2d: &Spectrum2D,
    max_dh_ppm: f64,
    max_dn_ppm: f64,
) -> (f64, f64) {
    let nx3 = spec_3d.axis_x.size;
    let ny3 = spec_3d.axis_y.size;
    let nz3 = spec_3d.axis_z.size;
    let nx2 = hsqc_2d.axis_x.size;
    let ny2 = hsqc_2d.axis_y.size;
    if nx3 < 4
        || ny3 < 4
        || nz3 == 0
        || nx2 < 4
        || ny2 < 4
        || max_dh_ppm <= 0.0
        || max_dn_ppm <= 0.0
    {
        return (0.0, 0.0);
    }

    // 0. Empirical `{sign, fold}` indirect-axis likelihood verification against `hsqc_2d`
    //    (`docs/specs/11-fid-processing.md` §1.5): rescues inverted (`REVERSE`) or half-SW-shifted
    //    (`States` <-> `States-TPPI`) indirect 15N/13C axes when log-Bayes factor exceeds
    //    `MIN_AXIS_RESCUE_LN_BF` (`12.0` nats), leaving canonical well-referenced cubes untouched.
    let _ = crate::pulprog::verify_3d_axis_hypotheses_against_hsqc(spec_3d, hsqc_2d, None);

    // 1. Build 2D `(15N, 1H)` max-abs sky-projection of `spec_3d` (supports both AllPositive & Biphasic cubes)
    let plane3 = ny3 * nx3;
    let mut proj = vec![0.0_f64; plane3];
    for plane_slice in spec_3d.data.chunks_exact(plane3).take(nz3) {
        for (dst, &s) in proj.iter_mut().zip(plane_slice.iter()) {
            let v = (s as f64).abs();
            if v > *dst {
                *dst = v;
            }
        }
    }
    let max3 = proj.iter().copied().fold(0.0_f64, f64::max);
    if max3 <= 0.0 {
        return (0.0, 0.0);
    }
    let mut scratch = proj.clone();
    let med3 = median_in_place(&mut scratch);
    for v in &mut scratch {
        *v = (*v - med3).abs();
    }
    let sigma3 = (1.4826 * median_in_place(&mut scratch)).max(0.02 * max3);
    let thresh3 = (med3 + 3.5 * sigma3).max(0.04 * max3);

    // Extract distinct 2D local maxima on the 3D `(1H, 15N)` sky-projection with capped SNR weights
    let mut pts3: Vec<(f64, f64, f64)> = Vec::new();
    for iy in 1..ny3.saturating_sub(1) {
        let n_ppm = spec_3d.axis_y.index_to_ppm(iy as f64);
        for ix in 1..nx3.saturating_sub(1) {
            let v = proj[iy * nx3 + ix];
            if v <= thresh3 {
                continue;
            }
            if v >= proj[(iy - 1) * nx3 + ix]
                && v >= proj[(iy + 1) * nx3 + ix]
                && v >= proj[iy * nx3 + ix - 1]
                && v >= proj[iy * nx3 + ix + 1]
            {
                let h_ppm = spec_3d.axis_x.index_to_ppm(ix as f64);
                let w = ((v - med3) / sigma3).clamp(0.5, 10.0);
                pts3.push((w, h_ppm, n_ppm));
            }
        }
    }
    if pts3.len() < 4 {
        return (0.0, 0.0);
    }
    pts3.sort_by(|a, b| b.0.total_cmp(&a.0));
    pts3.truncate(256);

    // 2. Extract distinct positive 2D local maxima from `hsqc_2d` (invariant to baseline zeroing / clean Lorentzians)
    let max2 = hsqc_2d
        .data
        .iter()
        .map(|&v| v as f64)
        .fold(0.0_f64, f64::max);
    if max2 <= 0.0 {
        return (0.0, 0.0);
    }
    let mut scratch2: Vec<f64> = hsqc_2d.data.iter().map(|&v| v as f64).collect();
    let med2 = median_in_place(&mut scratch2);
    for v in &mut scratch2 {
        *v = (*v - med2).abs();
    }
    let sigma2 = (1.4826 * median_in_place(&mut scratch2)).max(0.02 * max2);
    let thresh2 = (med2 + 3.5 * sigma2).max(0.04 * max2);

    let mut pts2: Vec<(f64, f64, f64)> = Vec::new();
    for iy in 1..ny2.saturating_sub(1) {
        let n_ppm = hsqc_2d.axis_y.index_to_ppm(iy as f64);
        for ix in 1..nx2.saturating_sub(1) {
            let v = hsqc_2d.data[iy * nx2 + ix] as f64;
            if v <= thresh2 {
                continue;
            }
            if v >= hsqc_2d.data[(iy - 1) * nx2 + ix] as f64
                && v >= hsqc_2d.data[(iy + 1) * nx2 + ix] as f64
                && v >= hsqc_2d.data[iy * nx2 + ix - 1] as f64
                && v >= hsqc_2d.data[iy * nx2 + ix + 1] as f64
            {
                let h_ppm = hsqc_2d.axis_x.index_to_ppm(ix as f64);
                let w = ((v - med2) / sigma2).clamp(0.5, 10.0);
                pts2.push((w, h_ppm, n_ppm));
            }
        }
    }
    if pts2.len() < 4 {
        return (0.0, 0.0);
    }
    pts2.sort_by(|a, b| b.0.total_cmp(&a.0));
    pts2.truncate(256);

    // Pre-filter candidate `(3D, 2D)` peak pairs within the search window plus 4 Gaussian widths
    let sig_h = 0.022_f64;
    let sig_n = 0.20_f64;
    let lim_dh = max_dh_ppm + 4.0 * sig_h;
    let lim_dn = max_dn_ppm + 4.0 * sig_n;
    let mut pairs: Vec<(f64, f64, f64)> = Vec::new();
    for &(w3, h3, n3) in &pts3 {
        for &(w2, h2, n2) in &pts2 {
            let pdh = h2 - h3;
            let pdn = n2 - n3;
            if pdh.abs() <= lim_dh && pdn.abs() <= lim_dn {
                pairs.push((w3 * w2, pdh, pdn));
            }
        }
    }
    if pairs.is_empty() {
        return (0.0, 0.0);
    }

    let inv_vh = 1.0 / (sig_h * sig_h);
    let inv_vn = 1.0 / (sig_n * sig_n);
    let eval_shift = |dh: f64, dn: f64| -> f64 {
        let mut s = 0.0_f64;
        for &(w, pdh, pdn) in &pairs {
            let eh = dh - pdh;
            let en = dn - pdn;
            let d2 = eh * eh * inv_vh + en * en * inv_vn;
            if d2 < 18.0 {
                s += w * (-0.5 * d2).exp();
            }
        }
        s
    };

    let zero_score = eval_shift(0.0, 0.0);
    let mut best_score = zero_score;
    let mut best_dh = 0.0_f64;
    let mut best_dn = 0.0_f64;

    let step_h_coarse = 0.006_f64;
    let step_n_coarse = 0.06_f64;
    let nh = (max_dh_ppm / step_h_coarse).ceil() as isize;
    let nn = (max_dn_ppm / step_n_coarse).ceil() as isize;
    for ih in -nh..=nh {
        let dh = (ih as f64 * step_h_coarse).clamp(-max_dh_ppm, max_dh_ppm);
        for in_idx in -nn..=nn {
            let dn = (in_idx as f64 * step_n_coarse).clamp(-max_dn_ppm, max_dn_ppm);
            let sc = eval_shift(dh, dn);
            if sc > best_score {
                best_score = sc;
                best_dh = dh;
                best_dn = dn;
            }
        }
    }

    let center_dh = best_dh;
    let center_dn = best_dn;
    for ih in -12..=12 {
        let dh = (center_dh + ih as f64 * 0.001).clamp(-max_dh_ppm, max_dh_ppm);
        for in_idx in -12..=12 {
            let dn = (center_dn + in_idx as f64 * 0.01).clamp(-max_dn_ppm, max_dn_ppm);
            let sc = eval_shift(dh, dn);
            if sc > best_score {
                best_score = sc;
                best_dh = dh;
                best_dn = dn;
            }
        }
    }

    if best_score > 0.0 && best_score > zero_score * 1.08 + 1e-6 {
        spec_3d.axis_x.first_ppm += best_dh;
        spec_3d.axis_y.first_ppm += best_dn;
        (best_dh, best_dn)
    } else {
        (0.0, 0.0)
    }
}

/// Convenience helper: read and process a 2D FID directory, returning `(Spectrum2D, resolved ProcessingRecipe)`.
pub fn process_fid_dir_2d(
    dir: impl AsRef<Path>,
    no_crop: bool,
    lp: bool,
) -> Result<(Spectrum2D, ProcessingRecipe), String> {
    let fid = read_fid_2d(dir)?;
    let mut recipe = ProcessingRecipe::from_fid_2d(&fid);
    if no_crop {
        recipe.set_no_crop();
    }
    if lp {
        recipe.set_linear_prediction(true);
    }
    let spec = process_fid_2d_mut(&fid, &mut recipe)?;
    Ok((spec, recipe))
}

/// Convenience helper: read and process a 3D FID directory, returning `(Spectrum3D, resolved ProcessingRecipe)`.
pub fn process_fid_dir_3d(
    dir: impl AsRef<Path>,
    exp: &str,
    no_crop: bool,
    lp: bool,
) -> Result<(Spectrum3D, ProcessingRecipe), String> {
    let fid = read_fid_3d(dir)?;
    let mut recipe = ProcessingRecipe::from_fid_3d(&fid);
    if no_crop {
        recipe.set_no_crop();
    }
    if lp {
        recipe.set_linear_prediction(true);
    }
    let spec = process_fid_3d_mut(&fid, exp, &mut recipe)?;
    Ok((spec, recipe))
}
