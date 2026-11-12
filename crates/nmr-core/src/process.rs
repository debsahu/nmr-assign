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
    read_fid_2d, read_fid_3d, Complex64, Fid1D, Fid2D, Fid3D, FidAxisMeta, StoredWindowHint,
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
    /// Step 6: Zero-order phase in degrees (`None` when `phase_source == AutoChen2002`).
    pub p0_deg: Option<f64>,
    /// Step 6: First-order phase in degrees across `[0, N_fft)` (`None` when `phase_source == AutoChen2002`).
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
            p0_deg,
            p1_deg,
            phase_source,
            baseline,
            roi_ppm,
        }
    }

    /// Explicitly override `(p0_deg, p1_deg)` on this dimension.
    pub fn set_explicit_phase(&mut self, p0_deg: f64, p1_deg: f64) {
        self.p0_deg = Some(p0_deg);
        self.p1_deg = Some(p1_deg);
        self.phase_source = PhaseSource::Explicit;
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
        Self {
            dims: vec![
                DimRecipe::from_axis_meta(&fid.axis_x, 0),
                DimRecipe::from_axis_meta(&fid.axis_y, 1),
            ],
            nus: fid.nus_mask.as_ref().map(|_| NusIstConfig::default()),
        }
    }

    pub fn from_fid_3d(fid: &Fid3D) -> Self {
        Self {
            dims: vec![
                DimRecipe::from_axis_meta(&fid.axis_x, 0),
                DimRecipe::from_axis_meta(&fid.axis_y, 1),
                DimRecipe::from_axis_meta(&fid.axis_z, 2),
            ],
            nus: fid.nus_mask.as_ref().map(|_| NusIstConfig::default()),
        }
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

/// Resolve `(p0_deg, p1_deg)` for a dimension given a representative unphased spectrum `rep_spec`
/// and full `AxisPpm`.
fn resolve_dim_phase(dim: &DimRecipe, full_axis: &AxisPpm, rep_spec: &[Complex64]) -> (f64, f64) {
    match (dim.p0_deg, dim.p1_deg, dim.phase_source) {
        (Some(p0), Some(p1), _) => (p0, p1),
        (Some(p0), None, _) => (p0, 0.0),
        (None, Some(p1), _) => (0.0, p1),
        (None, None, PhaseSource::DefaultZero) => (0.0, 0.0),
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

/// Reconstruct missing samples of a 1D indirect complex trace `obs[0..n1]` sampled on `mask[0..n1]`
/// using FISTA / IST ($L_1$ soft-thresholding in the unitary frequency domain with exact data consistency).
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

    // Compute initial dirty spectrum to calibrate geometric threshold schedule
    let mut spec = vec![[0.0, 0.0]; n_fft];
    spec[..n1].copy_from_slice(&y_obs);
    fft_in_place(&mut spec, false);
    let mut mags = Vec::with_capacity(n_fft);
    let mut max_mag = 0.0_f64;
    for c in &mut spec {
        c[0] *= scale_fwd;
        c[1] *= scale_fwd;
        let m = (c[0] * c[0] + c[1] * c[1]).sqrt();
        if m > max_mag {
            max_mag = m;
        }
        mags.push(m);
    }
    if max_mag <= 1e-30 {
        return y_obs;
    }

    let med_mag = median_in_place(&mut mags);
    for v in &mut mags {
        *v = (*v - med_mag).abs();
    }
    let mad_noise = 1.4826 * median_in_place(&mut mags);

    let iters = cfg.iterations.max(1);
    let lam_start = (cfg.initial_thresh_frac.clamp(0.1, 0.99)) * max_mag;
    let lam_end = (cfg.final_noise_sigma.max(0.5) * mad_noise)
        .max(0.002 * max_mag)
        .min(lam_start);
    let ratio = if iters > 1 && lam_start > 1e-30 {
        (lam_end / lam_start).powf(1.0 / ((iters - 1) as f64))
    } else {
        1.0
    };

    let mut z_cur = vec![[0.0, 0.0]; n_fft];
    z_cur[..n1].copy_from_slice(&y_obs);
    let mut z_prev = z_cur.clone();
    let mut w_ext = z_cur.clone();
    let mut active = vec![false; n_fft];
    let mut lam = lam_start;
    let beta_mom = if cfg.fista_momentum { 0.30 } else { 0.0 };

    for it in 0..iters {
        spec.copy_from_slice(&w_ext);
        for k in 0..n1 {
            if mask[k] {
                spec[k] = y_obs[k];
            }
        }
        fft_in_place(&mut spec, false);
        let is_last = it + 1 == iters;
        for (m, c) in spec.iter_mut().enumerate() {
            let sr = c[0] * scale_fwd;
            let si = c[1] * scale_fwd;
            let mag = (sr * sr + si * si).sqrt();
            if mag > lam {
                let shrink = (1.0 - lam / mag) * scale_inv;
                *c = [sr * shrink, si * shrink];
                if is_last {
                    active[m] = true;
                }
            } else {
                *c = [0.0, 0.0];
            }
        }
        fft_in_place(&mut spec, true);

        z_prev.copy_from_slice(&z_cur);
        z_cur.copy_from_slice(&spec);
        for k in 0..n1 {
            if mask[k] {
                z_cur[k] = y_obs[k];
            }
        }

        if beta_mom > 0.0 {
            for k in 0..n_fft {
                if k < n1 && mask[k] {
                    w_ext[k] = y_obs[k];
                } else {
                    w_ext[k] = [
                        z_cur[k][0] + beta_mom * (z_cur[k][0] - z_prev[k][0]),
                        z_cur[k][1] + beta_mom * (z_cur[k][1] - z_prev[k][1]),
                    ];
                }
            }
        } else {
            w_ext.copy_from_slice(&z_cur);
        }

        lam *= ratio;
    }

    // Debiasing support projections: eliminate L1 soft-thresholding shrinkage bias on active peaks
    if active.iter().any(|&a| a) {
        for _ in 0..8 {
            spec.copy_from_slice(&z_cur);
            for k in 0..n1 {
                if mask[k] {
                    spec[k] = y_obs[k];
                }
            }
            fft_in_place(&mut spec, false);
            for (m, c) in spec.iter_mut().enumerate() {
                if !active[m] {
                    *c = [0.0, 0.0];
                }
            }
            fft_in_place(&mut spec, true);
            z_cur.copy_from_slice(&spec);
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
/// soft-thresholding (`FISTA / IST`, Stern et al. 2007; Hyberts et al. 2012).
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
    let mut col_buf = vec![[0.0, 0.0]; f2];

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

    // Initial dirty spectrum for threshold calibration
    for k2 in 0..n2 {
        sp_plus[k2 * f1..k2 * f1 + n1].copy_from_slice(&y_plus[k2 * n1..(k2 + 1) * n1]);
        sp_minus[k2 * f1..k2 * f1 + n1].copy_from_slice(&y_minus[k2 * n1..(k2 + 1) * n1]);
    }
    fft_2d(&mut sp_plus, &mut col_buf, false);
    fft_2d(&mut sp_minus, &mut col_buf, false);

    let mut norms = Vec::with_capacity(n_vox);
    let mut max_norm = 0.0_f64;
    for m2 in 0..f2 {
        for m1 in 0..f1 {
            let m1_conj = (f1 - m1) % f1;
            let up = sp_plus[m2 * f1 + m1];
            let um = sp_minus[m2 * f1 + m1_conj];
            let e = 0.5
                * (up[0] * up[0] + up[1] * up[1] + um[0] * um[0] + um[1] * um[1])
                * (scale_fwd * scale_fwd);
            let h = e.sqrt();
            if h > max_norm {
                max_norm = h;
            }
            norms.push(h);
        }
    }
    if max_norm <= 1e-30 {
        return vec![[0.0; 4]; total];
    }

    let med = median_in_place(&mut norms);
    for v in &mut norms {
        *v = (*v - med).abs();
    }
    let mad_noise = 1.4826 * median_in_place(&mut norms);

    let iters = cfg.iterations.max(1);
    let lam_start = (cfg.initial_thresh_frac.clamp(0.1, 0.99)) * max_norm;
    let lam_end = (cfg.final_noise_sigma.max(0.5) * mad_noise)
        .max(0.002 * max_norm)
        .min(lam_start);
    let ratio = if iters > 1 && lam_start > 1e-30 {
        (lam_end / lam_start).powf(1.0 / ((iters - 1) as f64))
    } else {
        1.0
    };

    let mut zp_cur = vec![[0.0, 0.0]; n_vox];
    let mut zm_cur = vec![[0.0, 0.0]; n_vox];
    for k2 in 0..n2 {
        zp_cur[k2 * f1..k2 * f1 + n1].copy_from_slice(&y_plus[k2 * n1..(k2 + 1) * n1]);
        zm_cur[k2 * f1..k2 * f1 + n1].copy_from_slice(&y_minus[k2 * n1..(k2 + 1) * n1]);
    }
    let mut zp_prev = zp_cur.clone();
    let mut zm_prev = zm_cur.clone();
    let mut wp = zp_cur.clone();
    let mut wm = zm_cur.clone();
    let mut active_plus = vec![false; n_vox];
    let mut lam = lam_start;
    let beta_mom = if cfg.fista_momentum { 0.30 } else { 0.0 };

    for it in 0..iters {
        sp_plus.copy_from_slice(&wp);
        sp_minus.copy_from_slice(&wm);
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

        let is_last = it + 1 == iters;
        // Joint group-sparse quaternion thresholding at aligned `(m2, m1)` and `(m2, (f1 - m1) % f1)`:
        for m2 in 0..f2 {
            for &m1 in &[0usize, f1 / 2] {
                let i_pos = m2 * f1 + m1;
                let up = [sp_plus[i_pos][0] * scale_fwd, sp_plus[i_pos][1] * scale_fwd];
                let um = [
                    sp_minus[i_pos][0] * scale_fwd,
                    sp_minus[i_pos][1] * scale_fwd,
                ];
                let h =
                    (0.5 * (up[0] * up[0] + up[1] * up[1] + um[0] * um[0] + um[1] * um[1])).sqrt();
                if h > lam {
                    let s = (1.0 - lam / h) * scale_inv;
                    sp_plus[i_pos] = [up[0] * s, up[1] * s];
                    sp_minus[i_pos] = [um[0] * s, um[1] * s];
                    if is_last {
                        active_plus[i_pos] = true;
                    }
                } else {
                    sp_plus[i_pos] = [0.0, 0.0];
                    sp_minus[i_pos] = [0.0, 0.0];
                }
            }
            for m1 in 1..(f1 / 2) {
                let m1_c = f1 - m1;
                let idx_a = m2 * f1 + m1;
                let idx_b = m2 * f1 + m1_c;

                let up_a = [sp_plus[idx_a][0] * scale_fwd, sp_plus[idx_a][1] * scale_fwd];
                let um_a = [
                    sp_minus[idx_b][0] * scale_fwd,
                    sp_minus[idx_b][1] * scale_fwd,
                ];
                let h_a = (0.5
                    * (up_a[0] * up_a[0]
                        + up_a[1] * up_a[1]
                        + um_a[0] * um_a[0]
                        + um_a[1] * um_a[1]))
                    .sqrt();

                let up_b = [sp_plus[idx_b][0] * scale_fwd, sp_plus[idx_b][1] * scale_fwd];
                let um_b = [
                    sp_minus[idx_a][0] * scale_fwd,
                    sp_minus[idx_a][1] * scale_fwd,
                ];
                let h_b = (0.5
                    * (up_b[0] * up_b[0]
                        + up_b[1] * up_b[1]
                        + um_b[0] * um_b[0]
                        + um_b[1] * um_b[1]))
                    .sqrt();

                if h_a > lam {
                    let s = (1.0 - lam / h_a) * scale_inv;
                    sp_plus[idx_a] = [up_a[0] * s, up_a[1] * s];
                    sp_minus[idx_b] = [um_a[0] * s, um_a[1] * s];
                    if is_last {
                        active_plus[idx_a] = true;
                    }
                } else {
                    sp_plus[idx_a] = [0.0, 0.0];
                    sp_minus[idx_b] = [0.0, 0.0];
                }

                if h_b > lam {
                    let s = (1.0 - lam / h_b) * scale_inv;
                    sp_plus[idx_b] = [up_b[0] * s, up_b[1] * s];
                    sp_minus[idx_a] = [um_b[0] * s, um_b[1] * s];
                    if is_last {
                        active_plus[idx_b] = true;
                    }
                } else {
                    sp_plus[idx_b] = [0.0, 0.0];
                    sp_minus[idx_a] = [0.0, 0.0];
                }
            }
        }

        fft_2d(&mut sp_plus, &mut col_buf, true);
        fft_2d(&mut sp_minus, &mut col_buf, true);

        zp_prev.copy_from_slice(&zp_cur);
        zm_prev.copy_from_slice(&zm_cur);
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

        if beta_mom > 0.0 {
            for i in 0..n_vox {
                wp[i] = [
                    zp_cur[i][0] + beta_mom * (zp_cur[i][0] - zp_prev[i][0]),
                    zp_cur[i][1] + beta_mom * (zp_cur[i][1] - zp_prev[i][1]),
                ];
                wm[i] = [
                    zm_cur[i][0] + beta_mom * (zm_cur[i][0] - zm_prev[i][0]),
                    zm_cur[i][1] + beta_mom * (zm_cur[i][1] - zm_prev[i][1]),
                ];
            }
            for k2 in 0..n2 {
                for k1 in 0..n1 {
                    let idx = k2 * n1 + k1;
                    if mask[idx] {
                        wp[k2 * f1 + k1] = y_plus[idx];
                        wm[k2 * f1 + k1] = y_minus[idx];
                    }
                }
            }
        } else {
            wp.copy_from_slice(&zp_cur);
            wm.copy_from_slice(&zm_cur);
        }

        lam *= ratio;
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
    }

    // ------------------------------------------------------------------------
    // Dimension 0 (Direct 1H): Steps 1–9 across `n1 * 2` traces
    // ------------------------------------------------------------------------
    let dim_x = recipe.dims[0].clone();
    let si_x_est = dim_x.si.max(n0).max(2);
    let full_axis_x = build_full_axis_ppm(&fid.axis_x, si_x_est);
    let (x0, x1, axis_x_out) = compute_axis_roi(&full_axis_x, dim_x.roi_ppm)?;
    let nx_out = axis_x_out.size;

    // Resolve direct phase once (from stored/explicit or Chen 2002 on first sampled cosine trace)
    let (p0_x, p1_x) = if let (Some(p0), Some(p1)) = (dim_x.p0_deg, dim_x.p1_deg) {
        (p0, p1)
    } else {
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
        let ax_rep = build_full_axis_ppm(&fid.axis_x, rep_spec.len());
        resolve_dim_phase(&dim_x, &ax_rep, &rep_spec)
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
                real_full.resize(sc.len(), 0.0);
                apply_phase_and_discard_imag(&sc, p0_x, p1_x, &mut real_full);
                dst.copy_from_slice(&real_full[x0..=x1]);
            }
        },
    );

    // ------------------------------------------------------------------------
    // Dimension 1 (Indirect F1): Optional NUS IST + Steps 3–9 across `nx_out` columns
    // ------------------------------------------------------------------------
    let dim_y = recipe.dims[1].clone();
    let si_y = dim_y.si.max(n1).max(2);
    let full_axis_y = build_full_axis_ppm(&fid.axis_y, si_y);
    let (y0, y1, axis_y_out) = compute_axis_roi(&full_axis_y, dim_y.roi_ppm)?;
    let ny_out = axis_y_out.size;

    let (p0_y, p1_y) = (dim_y.p0_deg.unwrap_or(0.0), dim_y.p1_deg.unwrap_or(0.0));
    recipe.dims[1].p0_deg = Some(p0_y);
    recipe.dims[1].p1_deg = Some(p1_y);

    let nus_cfg = recipe.nus.clone().unwrap_or_default();

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

/// Process a 3D FID while recording any automatically solved `(p0_deg, p1_deg)` back into `recipe`.
pub fn process_fid_3d_mut(
    fid: &Fid3D,
    exp: &str,
    recipe: &mut ProcessingRecipe,
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
    }

    // ------------------------------------------------------------------------
    // Step A: Direct 1H (Dim 0) processing + imaginary discard + 1H ROI crop
    // ------------------------------------------------------------------------
    let dim_x = recipe.dims[0].clone();
    let si_x_est = dim_x.si.max(n0).max(2);
    let full_axis_x = build_full_axis_ppm(&fid.axis_x, si_x_est);
    let (x0, x1, axis_x_out) = compute_axis_roi(&full_axis_x, dim_x.roi_ppm)?;
    let nx_out = axis_x_out.size;

    let (p0_x, p1_x) = if let (Some(p0), Some(p1)) = (dim_x.p0_deg, dim_x.p1_deg) {
        (p0, p1)
    } else {
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
        let ax_rep = build_full_axis_ppm(&fid.axis_x, rep_spec.len());
        resolve_dim_phase(&dim_x, &ax_rep, &rep_spec)
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
    let dim_y = recipe.dims[1].clone();
    let dim_z = recipe.dims[2].clone();
    let si_y = dim_y.si.max(n1).max(2);
    let si_z = dim_z.si.max(n2).max(2);
    let full_axis_y = build_full_axis_ppm(&fid.axis_y, si_y);
    let full_axis_z = build_full_axis_ppm(&fid.axis_z, si_z);
    let (y0, y1, axis_y_out) = compute_axis_roi(&full_axis_y, dim_y.roi_ppm)?;
    let (z0, z1, axis_z_out) = compute_axis_roi(&full_axis_z, dim_z.roi_ppm)?;
    let ny_out = axis_y_out.size;
    let nz_out = axis_z_out.size;

    let (p0_y, p1_y) = (dim_y.p0_deg.unwrap_or(0.0), dim_y.p1_deg.unwrap_or(0.0));
    let (p0_z, p1_z) = (dim_z.p0_deg.unwrap_or(0.0), dim_z.p1_deg.unwrap_or(0.0));
    recipe.dims[1].p0_deg = Some(p0_y);
    recipe.dims[1].p1_deg = Some(p1_y);
    recipe.dims[2].p0_deg = Some(p0_z);
    recipe.dims[2].p1_deg = Some(p1_z);

    let nus_cfg = recipe.nus.clone().unwrap_or_default();
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

    if !swap_yz {
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
        Ok(Spectrum3D {
            name: fid.name.clone(),
            exp: exp.to_string(),
            axis_x: axis_x_out,
            axis_y: axis_y_out,
            axis_z: axis_z_out,
            data,
        })
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
        Ok(Spectrum3D {
            name: fid.name.clone(),
            exp: exp.to_string(),
            axis_x: axis_x_out,
            axis_y: axis_z_out,
            axis_z: axis_y_out,
            data,
        })
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
