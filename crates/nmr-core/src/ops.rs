//! Composable multidimensional NMR digital signal processing (DSP) operators,
//! autoregressive linear prediction, causal Hilbert transform, iterative baseline
//! correction, non-uniform sampling (NUS) reconstruction, 1D/2D/3D automatic phase
//! correction (`AllPositive` and `Biphasic`), and `.ft1` / multi-plane `.ft3` I/O.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed
//! literature and public binary header specifications (zero external source code, macros,
//! or parameter files inspected):
//! - Composable multidimensional vector-stream DSP model & 512-float binary header layout:
//!   Delaglio, F., Grzesiek, S., Vuister, G. W., Zhu, G., Pfeifer, J. & Bax, A.,
//!   *J. Biomol. NMR* **1995**, 6, 277–293.
//! - Apodization windows, causal Hilbert transform, discrete Fourier transform, phase &
//!   polynomial baseline correction: Hoch, J. C. & Stern, A. S., *NMR Data Processing*,
//!   Wiley-Liss, New York, **1996**; Otting, G., Widmer, H., Wagner, G. & Wüthrich, K.,
//!   *J. Magn. Reson.* **1986**, 66, 187–193 (`first_scale = 0.5` baseline offset removal);
//!   Ferrige, A. G. & Lindon, J. C., *J. Magn. Reson.* **1978**, 31, 337–340 (Lorentz-to-Gauss).
//! - Time-domain low-frequency convolution solvent filters:
//!   Marion, D., Ikura, M. & Bax, A., *J. Magn. Reson.* **1989**, 84, 425–430.
//! - Forward, backward & mirror-image Burg/autoregressive linear prediction with unit-circle
//!   root reflection: Barkhuijsen, H., de Beer, R., Bovée, W. M. M. J. & van Ormondt, D.,
//!   *J. Magn. Reson.* **1985**, 61, 465–481; Zhu, G. & Bax, A., *J. Magn. Reson.* **1990**,
//!   90, 405–410 & *J. Magn. Reson.* **1992**, 98, 192–199.
//! - Automatic phase correction (first-derivative Shannon entropy + dispersive baseline-wing
//!   symmetry): Chen, L., Weng, Z., Goh, L. & Garland, M., *J. Magn. Reson.* **2002**, 158,
//!   164–168; de Brouwer, H., *J. Magn. Reson.* **2009**, 201, 230–238.
//! - Model-free sliding-median baseline correction:
//!   Friedrichs, M. S., *J. Biomol. NMR* **1995**, 5, 147–153.
//! - Iterative soft thresholding (IST / FISTA) & Virtual-Echo NUS reconstruction:
//!   Stern, A. S., Donoho, D. L. & Hoch, J. C., *J. Magn. Reson.* **2007**, 188, 295–300;
//!   Hyberts, S. G., Milbradt, A. G., Wagner, A. B., Arthanari, H. & Wagner, G.,
//!   *J. Biomol. NMR* **2012**, 52, 315–327; Mayzel, M., Kazimierczuk, K. & Orekhov, V. Y.,
//!   *Chem. Commun.* **2014**, 50, 8947–8950.

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fid::{Complex64, Fid2D, Fid3D, FidAxisMeta, StoredWindowHint};
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

// ============================================================================
// 1. Core Pipeline & Operator Types (`StageOp` + `AxisPipeline` + `Trace1D`)
// ============================================================================

/// Kernel shape for time-domain low-frequency convolution solvent subtraction
/// (Marion, Ikura & Bax, *J. Magn. Reson.* 1989, 84, 425–430).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolventKernel {
    Boxcar,
    SineBell,
    Gaussian,
    PolyLowPass,
}

/// Autoregressive linear-prediction mode (Barkhuijsen et al. 1985; Zhu & Bax 1990, 1992).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LpMode {
    /// Append `n_pred` forward-extrapolated complex points `[N .. N + n_pred]` using AR(`order`).
    Forward { n_pred: usize, order: usize },
    /// Reconstruct the first `n_before` corrupted complex points `[0 .. n_before]` backward
    /// from the uncorrupted segment `[n_before .. N]` using AR(`order`).
    Backward { n_before: usize, order: usize },
    /// Mirror-image linear prediction (Zhu & Bax 1990, 1992) for known-phase indirect
    /// interferograms: `half_dwell = false` for `(0, 0)` initial delay (`t_0 = 0`),
    /// `half_dwell = true` for `(-90, +180)` / `(+90, -180)` half-dwell delay (`t_0 = dt / 2`).
    MirrorImage {
        n_pred: usize,
        order: usize,
        half_dwell: bool,
    },
}

/// Non-uniform sampling (NUS) 1D compressed-sensing solver algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NusMethod {
    /// Iterative Soft Thresholding (Stern, Donoho & Hoch 2007; Hyberts et al. 2012).
    Ist,
    /// Fast Iterative Shrinkage-Thresholding with Nesterov momentum (Mayzel et al. 2014).
    Fista,
}

/// Configuration for 1D IST / FISTA + optional Virtual-Echo NUS reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NusConfig {
    pub method: NusMethod,
    pub max_iter: usize,
    /// Relative stopping threshold relative to peak spectrum amplitude (e.g. `0.005`).
    pub stop_rel_thresh: f64,
    /// Per-iteration geometric threshold schedule factor in `(0.5, 0.999)` for `Ist` (e.g. `0.95`).
    pub schedule_decay: f64,
    /// Negate imaginary part before forward DFT (matching indirect `[cos, -sin]` convention).
    pub negate_imag: bool,
    /// First-point scaling factor (`0.5` when `|p1| < 1 deg`, else `1.0`).
    pub first_scale: f64,
    /// Optional known indirect `(p0_deg, p1_deg)` enabling Virtual-Echo real-absorption
    /// constraint (Mayzel, Kazimierczuk & Orekhov, *Chem. Commun.* 2014, 50, 8947–8950).
    pub virtual_echo_phase: Option<(f64, f64)>,
}

impl Default for NusConfig {
    fn default() -> Self {
        Self {
            method: NusMethod::Fista,
            max_iter: 80,
            stop_rel_thresh: 0.005,
            schedule_decay: 0.95,
            negate_imag: true,
            first_scale: 0.5,
            virtual_echo_phase: None,
        }
    }
}

/// Composable per-dimension NMR signal-processing stage (Delaglio et al. 1995; Hoch & Stern 1996).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StageOp {
    // --- Time-domain conditioning & solvent suppression ---
    GroupDelay {
        delay_pts: f64,
    },
    SolventFilter {
        shape: SolventKernel,
        half_width: usize,
        mirror_edges: bool,
    },
    TimeDcOffset {
        tail_frac: f64,
    },

    // --- Apodization windows ---
    SineWindow {
        off: f64,
        end: f64,
        pow: f64,
        first_scale: f64,
    },
    ExpWindow {
        lb_hz: f64,
        first_scale: f64,
    },
    LorentzToGauss {
        g1_hz: f64,
        g2_hz: f64,
        g3_center: f64,
        first_scale: f64,
    },
    Trapezoid {
        t1: usize,
        t2: usize,
        first_scale: f64,
    },
    Triangle {
        peak: usize,
        first_scale: f64,
    },
    JModulation {
        j_hz: f64,
        phase_deg: f64,
        sin_mode: bool,
    },

    // --- Linear prediction & NUS ---
    LinearPredict {
        mode: LpMode,
    },
    NusReconstruct {
        mask: Vec<bool>,
        config: NusConfig,
    },

    // --- Transforms, Hilbert transform & coordinate/sign manipulations ---
    ZeroFill {
        target_size: Option<usize>,
        factor: Option<usize>,
        pad: Option<usize>,
        auto_pow2: bool,
    },
    FourierTransform {
        inverse: bool,
        negate_imag: bool,
        sign_alt: bool,
        real_only: bool,
    },
    /// Reconstruct causal imaginary channel from a real-only frequency trace via
    /// `IFFT -> analytic causal multiplier [1, 2..2, 1, 0..0] -> FFT`.
    HilbertTransform,
    Phase {
        p0_deg: f64,
        p1_deg: f64,
        pivot_frac: f64,
    },
    DeleteImaginary,
    Magnitude,
    /// Exact DFT frequency-reversal convention (`out[0] = in[0]`, `out[k] = in[N - k]` for `k >= 1`),
    /// preserving Nyquist bin `0` and carrier bin `N / 2`.
    Reverse,
    CircularShift {
        points: isize,
    },
    CircularShiftPpm {
        delta_ppm: f64,
    },
    Scale {
        factor: f64,
    },
    ExtractPpm {
        min_ppm: f64,
        max_ppm: f64,
    },
    /// Half-open index slice `[start, end)` updating `AxisPpm` origin and size.
    ExtractIndex {
        start: usize,
        end: usize,
    },

    // --- Frequency-domain baseline correction ---
    PolyBaseline {
        order: usize,
        sigma_mask: f64,
    },
    MedianBaseline {
        half_window_pts: usize,
    },
    NodeSplineBaseline {
        node_indices: Vec<usize>,
    },
    CarrierNotch {
        center_ppm: f64,
        width_ppm: f64,
    },
}

/// Ordered per-dimension operator pipeline.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AxisPipeline {
    pub stages: Vec<StageOp>,
}

impl AxisPipeline {
    pub fn new(stages: Vec<StageOp>) -> Self {
        Self { stages }
    }

    pub fn push(&mut self, op: StageOp) -> &mut Self {
        self.stages.push(op);
        self
    }

    pub fn apply(&self, trace: &mut Trace1D) -> Result<(), String> {
        for stage in &self.stages {
            apply_stage(trace, stage)?;
        }
        Ok(())
    }
}

/// Mutable 1D vector state with calibrated spectral axis metadata.
#[derive(Clone, Debug)]
pub struct Trace1D {
    pub axis: AxisPpm,
    pub sw_hz: f64,
    pub data: Vec<Complex64>,
    pub is_freq_domain: bool,
    pub has_imaginary: bool,
    /// Residual sub-sample group delay (`delay_pts - round(delay_pts)`) applied as an exact
    /// linear phase across the zero-filled frequency grid during forward `FourierTransform`.
    pub pending_group_delay_pts: f64,
}

impl Trace1D {
    /// Construct a time-domain complex trace from `FidAxisMeta` and complex samples.
    pub fn from_fid_axis(meta: &FidAxisMeta, samples: Vec<Complex64>) -> Self {
        let n = samples.len().max(1);
        let sw_ppm = meta.sw_hz / meta.obs_mhz.max(1e-9);
        let axis = AxisPpm {
            label: meta.label.clone(),
            size: n,
            first_ppm: meta.first_ppm,
            step_ppm: -sw_ppm / (n as f64),
            obs_mhz: meta.obs_mhz,
        };
        Self {
            axis,
            sw_hz: meta.sw_hz,
            data: samples,
            is_freq_domain: false,
            has_imaginary: true,
            pending_group_delay_pts: 0.0,
        }
    }

    /// Construct a time-domain complex trace with explicit `sw_hz`, `obs_mhz`, and `first_ppm`.
    pub fn new_time_domain(
        label: impl Into<String>,
        sw_hz: f64,
        obs_mhz: f64,
        first_ppm: f64,
        data: Vec<Complex64>,
    ) -> Self {
        let n = data.len().max(1);
        let sw_ppm = sw_hz / obs_mhz.max(1e-9);
        Self {
            axis: AxisPpm {
                label: label.into(),
                size: n,
                first_ppm,
                step_ppm: -sw_ppm / (n as f64),
                obs_mhz,
            },
            sw_hz,
            data,
            is_freq_domain: false,
            has_imaginary: true,
            pending_group_delay_pts: 0.0,
        }
    }

    /// Construct a frequency-domain trace from an `AxisPpm` and complex bins.
    pub fn new_freq_complex(axis: AxisPpm, data: Vec<Complex64>) -> Self {
        let sw_hz = axis.sw_ppm() * axis.obs_mhz;
        Self {
            axis,
            sw_hz,
            data,
            is_freq_domain: true,
            has_imaginary: true,
            pending_group_delay_pts: 0.0,
        }
    }

    /// Construct a real-only frequency-domain trace from an `AxisPpm` and real bins.
    pub fn new_freq_real(axis: AxisPpm, real_bins: &[f64]) -> Self {
        let sw_hz = axis.sw_ppm() * axis.obs_mhz;
        let data = real_bins.iter().map(|&r| [r, 0.0]).collect();
        Self {
            axis,
            sw_hz,
            data,
            is_freq_domain: true,
            has_imaginary: false,
            pending_group_delay_pts: 0.0,
        }
    }

    /// Extract the real channel as `Vec<f64>`.
    pub fn real_vec(&self) -> Vec<f64> {
        self.data.iter().map(|c| c[0]).collect()
    }

    fn sync_axis_size(&mut self) {
        let n = self.data.len().max(1);
        let sw_ppm = self.sw_hz / self.axis.obs_mhz.max(1e-9);
        self.axis.size = n;
        self.axis.step_ppm = -sw_ppm / (n as f64);
    }
}

// ============================================================================
// 2. Complex Algebra & Arbitrary-N `f64` FFT (Radix-2 + Bluestein Chirp-Z)
// ============================================================================

#[inline]
fn c_add(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
fn c_sub(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn c_mul(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

#[inline]
fn c_conj(a: Complex64) -> Complex64 {
    [a[0], -a[1]]
}

#[inline]
fn c_abs2(a: Complex64) -> f64 {
    a[0] * a[0] + a[1] * a[1]
}

#[inline]
fn c_div(a: Complex64, b: Complex64) -> Complex64 {
    let d = c_abs2(b);
    if d <= 1e-60 {
        [0.0, 0.0]
    } else {
        [
            (a[0] * b[0] + a[1] * b[1]) / d,
            (a[1] * b[0] - a[0] * b[1]) / d,
        ]
    }
}

fn fft_radix2(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
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
        let ang = sign * 2.0 * PI / (len as f64);
        let wlen = [ang.cos(), ang.sin()];
        for start in (0..n).step_by(len) {
            let mut w = [1.0, 0.0];
            for k in 0..half {
                let u = buf[start + k];
                let v = c_mul(buf[start + k + half], w);
                buf[start + k] = c_add(u, v);
                buf[start + k + half] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in buf.iter_mut() {
            v[0] *= inv_n;
            v[1] *= inv_n;
        }
    }
}

/// Exact `f64` complex Discrete Fourier Transform for arbitrary length `N >= 1`
/// (in-place radix-2 Cooley–Tukey for power-of-two `N`, Bluestein chirp-z otherwise).
/// Forward (`inverse == false`): `X[m] = sum_{k=0}^{N-1} x[k] exp(-2 pi i k m / N)`.
/// Inverse (`inverse == true`): `x[k] = (1 / N) sum_{m=0}^{N-1} X[m] exp(+2 pi i k m / N)`.
pub fn fft_arbitrary_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        fft_radix2(buf, inverse);
        return;
    }

    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };
    let two_n = (2 * n) as u128;

    let mut chirp = Vec::with_capacity(n);
    for k in 0..n {
        let k2_mod = ((k as u128) * (k as u128) % two_n) as f64;
        let ang = sign * PI * k2_mod / (n as f64);
        chirp.push([ang.cos(), ang.sin()]);
    }

    let mut a = vec![[0.0, 0.0]; m];
    let mut b = vec![[0.0, 0.0]; m];
    for k in 0..n {
        a[k] = c_mul(buf[k], chirp[k]);
    }
    b[0] = c_conj(chirp[0]);
    for k in 1..n {
        let cc = c_conj(chirp[k]);
        b[k] = cc;
        b[m - k] = cc;
    }

    fft_radix2(&mut a, false);
    fft_radix2(&mut b, false);
    for i in 0..m {
        a[i] = c_mul(a[i], b[i]);
    }
    fft_radix2(&mut a, true);

    let norm = if inverse { 1.0 / (n as f64) } else { 1.0 };
    for k in 0..n {
        let prod = c_mul(a[k], chirp[k]);
        buf[k] = [prod[0] * norm, prod[1] * norm];
    }
}

/// Shift zero-frequency bin `0` to the center (`N / 2`).
#[inline]
pub fn fftshift_in_place(buf: &mut [Complex64]) {
    if buf.len() > 1 {
        buf.rotate_right(buf.len() / 2);
    }
}

/// Inverse of `fftshift_in_place`: move center (`N / 2`) back to bin `0`.
#[inline]
pub fn ifftshift_in_place(buf: &mut [Complex64]) {
    if buf.len() > 1 {
        buf.rotate_left(buf.len() / 2);
    }
}

/// Reconstruct the causal imaginary channel of a shifted frequency-domain trace `buf`
/// from its real channel `buf[k][0]` via `IFFT -> [1, 2..2, 1, 0..0] -> FFT`
/// (Hoch & Stern, *NMR Data Processing*, Wiley-Liss, 1996).
pub fn hilbert_transform_in_place(buf: &mut [Complex64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let mut work: Vec<Complex64> = buf.iter().map(|c| [c[0], 0.0]).collect();
    ifftshift_in_place(&mut work);
    fft_arbitrary_in_place(&mut work, true);

    if n.is_multiple_of(2) {
        let half = n / 2;
        for c in work.iter_mut().take(half).skip(1) {
            c[0] *= 2.0;
            c[1] *= 2.0;
        }
        // work[0] and work[half] keep factor 1.0
        for c in work.iter_mut().skip(half + 1) {
            *c = [0.0, 0.0];
        }
    } else {
        let half = n / 2;
        for c in work.iter_mut().take(half + 1).skip(1) {
            c[0] *= 2.0;
            c[1] *= 2.0;
        }
        for c in work.iter_mut().skip(half + 1) {
            *c = [0.0, 0.0];
        }
    }

    fft_arbitrary_in_place(&mut work, false);
    fftshift_in_place(&mut work);
    for (dst, src) in buf.iter_mut().zip(work.iter()) {
        dst[1] = src[1];
    }
}

// ============================================================================
// 3. Burg Autoregressive Linear Prediction + Unit-Circle Root Reflection
// ============================================================================

const LP_MAX_ROOT_RADIUS: f64 = 0.9999;

/// Fit complex Burg autoregressive coefficients `a[0..order]` (`y[n] + sum_{j=1}^p a[j-1] y[n-j] = 0`)
/// and stabilize all characteristic roots inside `|z| <= 0.9999` via Aberth–Ehrlich root reflection
/// (Barkhuijsen et al. 1985; Zhu & Bax 1990, 1992).
pub fn fit_burg_ar_stabilized(signal: &[Complex64], order: usize) -> Vec<Complex64> {
    let m_len = signal.len();
    let p = order.min(m_len.saturating_sub(2));
    if p == 0 {
        return Vec::new();
    }

    let total_energy: f64 = signal.iter().map(|&c| c_abs2(c)).sum();
    if !total_energy.is_finite() || total_energy <= 1e-30 {
        return vec![[0.0, 0.0]; p];
    }

    let mut f_err = signal.to_vec();
    let mut b_err = signal.to_vec();
    let mut a = vec![[0.0, 0.0]; p];

    for stage in 0..p {
        let mut num = [0.0, 0.0];
        let mut den = 0.0;
        for n in (stage + 1)..m_len {
            let fn_val = f_err[n];
            let bn_prev = b_err[n - 1];
            num = c_add(num, c_mul(fn_val, c_conj(bn_prev)));
            den += c_abs2(fn_val) + c_abs2(bn_prev);
        }
        if den <= 1e-30 {
            break;
        }
        let mut k_ref = [-2.0 * num[0] / den, -2.0 * num[1] / den];
        let k_mag = k_ref[0].hypot(k_ref[1]);
        if k_mag > LP_MAX_ROOT_RADIUS {
            let sc = LP_MAX_ROOT_RADIUS / k_mag;
            k_ref[0] *= sc;
            k_ref[1] *= sc;
        }

        // Levinson–Durbin update of AR coefficients
        let mut a_next = a.clone();
        a_next[stage] = k_ref;
        for j in 0..stage {
            a_next[j] = c_add(a[j], c_mul(k_ref, c_conj(a[stage - 1 - j])));
        }
        a = a_next;

        // Update forward and backward prediction errors for `n = m_len - 1` down to `stage + 1`
        for n in ((stage + 1)..m_len).rev() {
            let f_old = f_err[n];
            let b_old = b_err[n - 1];
            f_err[n] = c_add(f_old, c_mul(k_ref, b_old));
            b_err[n] = c_add(b_old, c_mul(c_conj(k_ref), f_old));
        }
    }

    stabilize_ar_polynomial_roots(&a)
}

/// Find all roots of `P(z) = z^p + a[0] z^{p-1} + ... + a[p-1]` using Aberth–Ehrlich cubic
/// iteration, reflect any root outside the unit circle (`z -> 1 / conj(z)`), clamp modulus to
/// `<= 0.9999`, and reconstruct stabilized monic coefficients `a_stab[0..p]`.
fn stabilize_ar_polynomial_roots(a: &[Complex64]) -> Vec<Complex64> {
    let p = a.len();
    if p == 0 {
        return Vec::new();
    }
    // Trim trailing exact zeros (roots at z = 0 are already strictly inside the unit circle)
    let mut p_eff = p;
    while p_eff > 0 && c_abs2(a[p_eff - 1]) <= 1e-30 {
        p_eff -= 1;
    }
    if p_eff == 0 {
        return vec![[0.0, 0.0]; p];
    }
    if p_eff == 1 {
        // P(z) = z + a[0] => root z0 = -a[0]
        let mut z0 = [-a[0][0], -a[0][1]];
        z0 = reflect_and_clamp_root(z0);
        let mut out = vec![[0.0, 0.0]; p];
        out[0] = [-z0[0], -z0[1]];
        return out;
    }

    let mut roots: Vec<Complex64> = (0..p_eff)
        .map(|j| {
            let r = 0.85 + 0.07 * (((j * 7) % 5) as f64 / 5.0);
            let ang = 2.0 * PI * (j as f64) / (p_eff as f64) + 0.137;
            [r * ang.cos(), r * ang.sin()]
        })
        .collect();

    let mut converged = false;
    for _iter in 0..80 {
        let mut max_step2 = 0.0f64;
        for i in 0..p_eff {
            let zi = roots[i];
            // Horner evaluation of P(zi) and P'(zi)
            let mut val = [1.0, 0.0];
            let mut deriv = [0.0, 0.0];
            for &coef in &a[..p_eff] {
                deriv = c_add(c_mul(deriv, zi), val);
                val = c_add(c_mul(val, zi), coef);
            }
            if c_abs2(val) <= 1e-28 {
                continue;
            }
            if c_abs2(deriv) <= 1e-50 {
                continue;
            }
            let newton = c_div(val, deriv);
            let mut sum_inv = [0.0, 0.0];
            for (j, &zj) in roots.iter().enumerate() {
                if i != j {
                    let diff = c_sub(zi, zj);
                    if c_abs2(diff) > 1e-50 {
                        sum_inv = c_add(sum_inv, c_div([1.0, 0.0], diff));
                    }
                }
            }
            let denom = c_sub([1.0, 0.0], c_mul(newton, sum_inv));
            let step = if c_abs2(denom) > 1e-50 {
                c_div(newton, denom)
            } else {
                newton
            };
            roots[i] = c_sub(zi, step);
            max_step2 = max_step2.max(c_abs2(step));
        }
        if max_step2 < 1e-24 {
            converged = true;
            break;
        }
    }

    if !converged && roots.iter().any(|z| !z[0].is_finite() || !z[1].is_finite()) {
        // Burg's Levinson coefficients with |k_m| <= 0.9999 are already Schur-stable
        return a.to_vec();
    }

    for z in &mut roots {
        if !z[0].is_finite() || !z[1].is_finite() {
            return a.to_vec();
        }
        *z = reflect_and_clamp_root(*z);
    }

    // Reconstruct monic polynomial prod_{j=0}^{p_eff - 1} (z - z_j) = z^{p_eff} + c_1 z^{p_eff - 1} + ...
    let mut poly = Vec::with_capacity(p_eff + 1);
    poly.push([1.0, 0.0]);
    for &zj in &roots {
        let neg_zj = [-zj[0], -zj[1]];
        let prev_len = poly.len();
        poly.push([0.0, 0.0]);
        for k in (1..=prev_len).rev() {
            poly[k] = c_add(poly[k], c_mul(poly[k - 1], neg_zj));
        }
    }

    let mut out = vec![[0.0, 0.0]; p];
    out[..p_eff].copy_from_slice(&poly[1..(p_eff + 1)]);
    out
}

#[inline]
fn reflect_and_clamp_root(mut z: Complex64) -> Complex64 {
    let r2 = c_abs2(z);
    if r2 > 1.0 {
        // Unit-circle reflection: 1 / conj(z) = z / |z|^2 (preserves exact frequency arg(z))
        z = [z[0] / r2, z[1] / r2];
    }
    let r = z[0].hypot(z[1]);
    if r > LP_MAX_ROOT_RADIUS {
        let sc = LP_MAX_ROOT_RADIUS / r;
        z = [z[0] * sc, z[1] * sc];
    }
    z
}

fn predict_forward_with_ar(
    history: &[Complex64],
    ar: &[Complex64],
    n_pred: usize,
) -> Vec<Complex64> {
    let mut buf = Vec::with_capacity(history.len() + n_pred);
    buf.extend_from_slice(history);
    for _ in 0..n_pred {
        let cur = buf.len();
        let mut next = [0.0, 0.0];
        for (j, &aj) in ar.iter().enumerate() {
            if cur > j {
                next = c_sub(next, c_mul(aj, buf[cur - 1 - j]));
            }
        }
        buf.push(next);
    }
    buf.split_off(history.len())
}

/// Fit damped forward linear-prediction AR coefficients `a[0..order]`
/// (`y[n] + sum_{j=1}^p a[j-1] y[n-j] = 0`) via complex covariance normal equations with
/// Burg fallback, then reflect & clamp all characteristic roots inside `|z| <= 0.9999`
/// (Barkhuijsen et al. 1985; Zhu & Bax 1990, 1992).
pub fn fit_damped_ar_stabilized(signal: &[Complex64], order: usize) -> Vec<Complex64> {
    let m_len = signal.len();
    let p = order.min(m_len.saturating_sub(2));
    if p == 0 {
        return Vec::new();
    }
    let total_energy: f64 = signal.iter().map(|&c| c_abs2(c)).sum();
    if !total_energy.is_finite() || total_energy <= 1e-30 {
        return vec![[0.0, 0.0]; p];
    }

    // Build complex Hermitian covariance system `C * a = -r` for `n = p .. m_len`
    let mut c_mat = vec![vec![[0.0, 0.0]; p]; p];
    let mut rhs = vec![[0.0, 0.0]; p];
    for n in p..m_len {
        let yn = signal[n];
        for row in 0..p {
            let y_row_conj = c_conj(signal[n - 1 - row]);
            rhs[row] = c_sub(rhs[row], c_mul(y_row_conj, yn));
            for col in row..p {
                c_mat[row][col] = c_add(c_mat[row][col], c_mul(y_row_conj, signal[n - 1 - col]));
            }
        }
    }
    let mut diag_sum = 0.0;
    for (i, row_v) in c_mat.iter().enumerate().take(p) {
        diag_sum += row_v[i][0];
    }
    let reg = (1e-10 * (diag_sum / (p as f64))).max(1e-24);
    for (row, row_v) in c_mat.iter_mut().enumerate().take(p) {
        row_v[row][0] += reg;
    }
    for row in 1..p {
        let (upper, lower) = c_mat.split_at_mut(row);
        for (col, up_row) in upper.iter().enumerate() {
            lower[0][col] = c_conj(up_row[row]);
        }
    }

    if let Some(a_cov) = solve_complex_gauss(&mut c_mat, &mut rhs) {
        if a_cov.iter().all(|c| c[0].is_finite() && c[1].is_finite()) {
            return stabilize_ar_polynomial_roots(&a_cov);
        }
    }
    fit_burg_ar_stabilized(signal, order)
}

fn solve_complex_gauss(a: &mut [Vec<Complex64>], b: &mut [Complex64]) -> Option<Vec<Complex64>> {
    let m = b.len();
    for col in 0..m {
        let mut piv = col;
        let mut best = c_abs2(a[col][col]);
        for (row, a_row) in a.iter().enumerate().take(m).skip(col + 1) {
            let mag2 = c_abs2(a_row[col]);
            if mag2 > best {
                best = mag2;
                piv = row;
            }
        }
        if best <= 1e-40 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        let (top, bot) = a.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for (offset, a_row) in bot.iter_mut().enumerate() {
            let row = col + 1 + offset;
            let factor = c_div(a_row[col], diag);
            b[row] = c_sub(b[row], c_mul(factor, b[col]));
            for (dst, &pv) in a_row[col..m].iter_mut().zip(&pivot_row[col..m]) {
                *dst = c_sub(*dst, c_mul(factor, pv));
            }
        }
    }
    let mut x = vec![[0.0, 0.0]; m];
    for row in (0..m).rev() {
        let mut s = b[row];
        for k in (row + 1)..m {
            s = c_sub(s, c_mul(a[row][k], x[k]));
        }
        x[row] = c_div(s, a[row][row]);
    }
    Some(x)
}

/// Apply `LpMode` linear prediction to a 1D complex FID vector.
pub fn apply_linear_prediction(data: &mut Vec<Complex64>, mode: LpMode) {
    let n = data.len();
    match mode {
        LpMode::Forward { n_pred, order } => {
            if n_pred == 0 || n < 3 || order == 0 {
                data.resize(n + n_pred, [0.0, 0.0]);
                return;
            }
            let ar = fit_damped_ar_stabilized(data, order);
            let tail = predict_forward_with_ar(data, &ar, n_pred);
            data.extend_from_slice(&tail);
        }
        LpMode::Backward { n_before, order } => {
            if n_before == 0 || n <= n_before + 2 || order == 0 {
                return;
            }
            // Reverse and conjugate the uncorrupted segment `[n_before .. n]`, forward-predict
            // `n_before` points, then conjugate and reverse back into `[0 .. n_before]`.
            let clean = &data[n_before..n];
            let rev_conj: Vec<Complex64> = clean.iter().rev().map(|&c| c_conj(c)).collect();
            let ar = fit_burg_ar_stabilized(&rev_conj, order);
            let pred_rev = predict_forward_with_ar(&rev_conj, &ar, n_before);
            for (k, &val) in pred_rev.iter().enumerate() {
                // k = 0 corresponds to index `n_before - 1`, k = n_before - 1 to index `0`
                data[n_before - 1 - k] = c_conj(val);
            }
        }
        LpMode::MirrorImage {
            n_pred,
            order,
            half_dwell,
        } => {
            if n_pred == 0 || n < 2 || order == 0 {
                data.resize(n + n_pred, [0.0, 0.0]);
                return;
            }
            // Zhu & Bax (1990, 1992) conjugate-symmetric extended sequence:
            // - `half_dwell == false` (`t_k = k * dt`): `[z*[N-1], ..., z*[1], z[0], ..., z[N-1]]` (len 2N - 1)
            // - `half_dwell == true` (`t_k = (k + 0.5) * dt`): `[z*[N-1], ..., z*[0], z[0], ..., z[N-1]]` (len 2N)
            let mut mirror = Vec::with_capacity(if half_dwell { 2 * n } else { 2 * n - 1 });
            let skip_zero = usize::from(!half_dwell);
            for k in (skip_zero..n).rev() {
                mirror.push(c_conj(data[k]));
            }
            mirror.extend_from_slice(data);
            let ar = fit_burg_ar_stabilized(&mirror, order);
            let tail = predict_forward_with_ar(&mirror, &ar, n_pred);
            data.extend_from_slice(&tail);
        }
    }
}

// ============================================================================
// 4. Non-Uniform Sampling (NUS): IST / FISTA + Virtual-Echo Reconstruction
// ============================================================================

/// Reconstruct a 1D complex time-domain trace sampled on `mask` (`mask.len() == trace.len()`)
/// via IST or FISTA (Stern, Donoho & Hoch 2007; Hyberts et al. 2012; Mayzel et al. 2014).
/// Returns a dense time-domain vector of length `trace.len()` matching `trace` on sampled indices.
pub fn reconstruct_nus_1d(
    trace: &[Complex64],
    mask: &[bool],
    config: &NusConfig,
) -> Vec<Complex64> {
    let n = trace.len();
    if n <= 1 || mask.len() != n || mask.iter().all(|&b| b) {
        return trace.to_vec();
    }
    if !mask.iter().any(|&b| b) {
        return vec![[0.0, 0.0]; n];
    }

    let n_fft = (2 * n).next_power_of_two();
    let sign_im = if config.negate_imag { -1.0 } else { 1.0 };

    // Forward operator `A(x_spec) -> time_masked` and adjoint `A*(time_res) -> spec_grad`
    let time_to_spec = |td: &[Complex64]| -> Vec<Complex64> {
        let mut buf = vec![[0.0, 0.0]; n_fft];
        for (k, &c) in td.iter().enumerate() {
            buf[k] = [c[0], sign_im * c[1]];
        }
        if !buf.is_empty() {
            buf[0][0] *= config.first_scale;
            buf[0][1] *= config.first_scale;
        }
        fft_arbitrary_in_place(&mut buf, false);
        fftshift_in_place(&mut buf);
        if let Some((p0, p1)) = config.virtual_echo_phase {
            apply_phase_slice(&mut buf, p0, p1, 0.0);
        }
        buf
    };

    let spec_to_time = |spec: &[Complex64]| -> Vec<Complex64> {
        let mut buf = spec.to_vec();
        if let Some((p0, p1)) = config.virtual_echo_phase {
            // Virtual-Echo (Mayzel et al. 2014): reconstruct causal partner from real absorption
            for c in &mut buf {
                c[1] = 0.0;
            }
            hilbert_transform_in_place(&mut buf);
            apply_phase_slice(&mut buf, -p0, -p1, 0.0);
        }
        ifftshift_in_place(&mut buf);
        fft_arbitrary_in_place(&mut buf, true);
        let mut td = Vec::with_capacity(n);
        for (k, &c) in buf.iter().enumerate().take(n) {
            let inv_f = if k == 0 && config.first_scale.abs() > 1e-6 {
                1.0 / config.first_scale
            } else {
                1.0
            };
            td.push([c[0] * inv_f, sign_im * c[1] * inv_f]);
        }
        td
    };

    // Masked observation `b`
    let mut b_obs = vec![[0.0, 0.0]; n];
    for k in 0..n {
        if mask[k] {
            b_obs[k] = trace[k];
        }
    }

    let spec0 = time_to_spec(&b_obs);
    let max_amp = spec0
        .iter()
        .map(|c| {
            if config.virtual_echo_phase.is_some() {
                c[0].abs()
            } else {
                c[0].hypot(c[1])
            }
        })
        .fold(0.0_f64, f64::max);
    if max_amp <= 1e-30 {
        return b_obs;
    }

    let tau_min = (config.stop_rel_thresh.clamp(1e-5, 0.5)) * max_amp;
    let decay = config.schedule_decay.clamp(0.5, 0.999);
    let max_it = config.max_iter.max(1);

    let mut x_spec = vec![[0.0, 0.0]; n_fft];
    let mut y_spec = vec![[0.0, 0.0]; n_fft];
    let mut t_fista = 1.0_f64;

    for it in 0..max_it {
        let base_spec = match config.method {
            NusMethod::Ist => &x_spec,
            NusMethod::Fista => &y_spec,
        };
        let pred_td = spec_to_time(base_spec);
        let mut res_td = vec![[0.0, 0.0]; n];
        for k in 0..n {
            if mask[k] {
                res_td[k] = c_sub(b_obs[k], pred_td[k]);
            }
        }
        let grad_spec = time_to_spec(&res_td);

        let tau = match config.method {
            NusMethod::Ist => (max_amp * decay.powi((it + 1) as i32)).max(tau_min),
            NusMethod::Fista => {
                let frac = (it as f64) / ((max_it.saturating_sub(1)).max(1) as f64);
                (max_amp * (tau_min / max_amp).powf(frac)).max(tau_min)
            }
        };

        let mut x_next = vec![[0.0, 0.0]; n_fft];
        for i in 0..n_fft {
            let u = c_add(base_spec[i], grad_spec[i]);
            if config.virtual_echo_phase.is_some() {
                let r_abs = u[0].abs();
                let r_shrunk = if r_abs > tau {
                    u[0].signum() * (r_abs - tau)
                } else {
                    0.0
                };
                x_next[i] = [r_shrunk, 0.0];
            } else {
                let mag = u[0].hypot(u[1]);
                if mag > tau {
                    let sc = (mag - tau) / mag;
                    x_next[i] = [u[0] * sc, u[1] * sc];
                }
            }
        }

        if config.method == NusMethod::Fista {
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_fista * t_fista).sqrt());
            let mom = (t_fista - 1.0) / t_next;
            for i in 0..n_fft {
                y_spec[i] = [
                    x_next[i][0] + mom * (x_next[i][0] - x_spec[i][0]),
                    x_next[i][1] + mom * (x_next[i][1] - x_spec[i][1]),
                ];
            }
            t_fista = t_next;
        }
        x_spec = x_next;

        if config.method == NusMethod::Ist && tau <= tau_min * 1.001 {
            break;
        }
    }

    let mut out_td = spec_to_time(&x_spec);
    // Enforce exact data fidelity on measured schedule coordinates
    for k in 0..n {
        if mask[k] {
            out_td[k] = trace[k];
        }
    }
    out_td
}

// ============================================================================
// 5. Execution of Individual `StageOp` Stages
// ============================================================================

#[inline]
fn apply_phase_slice(buf: &mut [Complex64], p0_deg: f64, p1_deg: f64, pivot_frac: f64) {
    let n = buf.len();
    if n == 0 || (p0_deg.abs() <= 1e-12 && p1_deg.abs() <= 1e-12) {
        return;
    }
    let nf = n as f64;
    for (k, c) in buf.iter_mut().enumerate() {
        let phi = (p0_deg + p1_deg * ((k as f64) / nf - pivot_frac)).to_radians();
        let (sp, cp) = phi.sin_cos();
        let (re, im) = (c[0], c[1]);
        *c = [re * cp - im * sp, re * sp + im * cp];
    }
}

/// Apply a single `StageOp` in place to `trace`.
pub fn apply_stage(trace: &mut Trace1D, op: &StageOp) -> Result<(), String> {
    let n = trace.data.len();
    if n == 0 {
        return Ok(());
    }
    match op {
        StageOp::GroupDelay { delay_pts } => {
            if delay_pts.abs() <= 1e-12 {
                return Ok(());
            }
            let m = delay_pts.round() as isize;
            let delta_g = *delay_pts - (m as f64);
            if m > 0 {
                let mu = (m as usize).min(n);
                trace.data.copy_within(mu..n, 0);
                for c in &mut trace.data[(n - mu)..n] {
                    *c = [0.0, 0.0];
                }
            } else if m < 0 {
                let mu = ((-m) as usize).min(n);
                trace.data.copy_within(0..(n - mu), mu);
                for c in &mut trace.data[0..mu] {
                    *c = [0.0, 0.0];
                }
            }
            trace.pending_group_delay_pts += delta_g;
        }

        StageOp::SolventFilter {
            shape,
            half_width,
            mirror_edges,
        } => {
            apply_solvent_filter(&mut trace.data, *shape, *half_width, *mirror_edges);
        }

        StageOp::TimeDcOffset { tail_frac } => {
            let frac = tail_frac.clamp(0.01, 1.0);
            let count = ((n as f64 * frac).round() as usize).clamp(1, n);
            let start = n - count;
            let mut mean = [0.0, 0.0];
            for &c in &trace.data[start..n] {
                mean[0] += c[0];
                mean[1] += c[1];
            }
            mean[0] /= count as f64;
            mean[1] /= count as f64;
            for c in &mut trace.data {
                c[0] -= mean[0];
                c[1] -= mean[1];
            }
        }

        StageOp::SineWindow {
            off,
            end,
            pow,
            first_scale,
        } => {
            let denom = (n.saturating_sub(1)).max(1) as f64;
            for (k, c) in trace.data.iter_mut().enumerate() {
                let frac = if n > 1 { (k as f64) / denom } else { 0.0 };
                let arg = PI * (off + (end - off) * frac);
                let s = arg.sin();
                let w = if (*pow - 1.0).abs() <= 1e-12 {
                    s
                } else if (*pow - 2.0).abs() <= 1e-12 {
                    s * s
                } else {
                    s.max(0.0).powf(*pow)
                };
                c[0] *= w;
                c[1] *= w;
            }
            trace.data[0][0] *= *first_scale;
            trace.data[0][1] *= *first_scale;
        }

        StageOp::ExpWindow { lb_hz, first_scale } => {
            let sw = trace.sw_hz.max(1e-9);
            for (k, c) in trace.data.iter_mut().enumerate() {
                let w = (-PI * lb_hz * (k as f64) / sw).exp();
                c[0] *= w;
                c[1] *= w;
            }
            trace.data[0][0] *= *first_scale;
            trace.data[0][1] *= *first_scale;
        }

        StageOp::LorentzToGauss {
            g1_hz,
            g2_hz,
            g3_center,
            first_scale,
        } => {
            let sw = trace.sw_hz.max(1e-9);
            // FWHM-to-Gaussian coefficient: alpha = pi / (2 * sqrt(ln 2)) * g2_hz
            let gauss_alpha = (PI / (2.0 * 2.0_f64.ln().sqrt())) * g2_hz;
            let k_center = g3_center * ((n.saturating_sub(1)) as f64);
            for (k, c) in trace.data.iter_mut().enumerate() {
                let t = (k as f64) / sw;
                let dt_center = ((k as f64) - k_center) / sw;
                let e_term = PI * g1_hz * t;
                let g_term = gauss_alpha * dt_center;
                let w = (e_term - g_term * g_term).exp();
                c[0] *= w;
                c[1] *= w;
            }
            trace.data[0][0] *= *first_scale;
            trace.data[0][1] *= *first_scale;
        }

        StageOp::Trapezoid {
            t1,
            t2,
            first_scale,
        } => {
            let last = n.saturating_sub(1);
            let k1 = (*t1).min(last);
            let k2 = (*t2).clamp(k1, last);
            for (k, c) in trace.data.iter_mut().enumerate() {
                let w = if k < k1 {
                    (k as f64) / (k1 as f64)
                } else if k <= k2 || last == k2 {
                    1.0
                } else {
                    ((last - k) as f64) / ((last - k2) as f64)
                };
                c[0] *= w;
                c[1] *= w;
            }
            trace.data[0][0] *= *first_scale;
            trace.data[0][1] *= *first_scale;
        }

        StageOp::Triangle { peak, first_scale } => {
            let last = n.saturating_sub(1);
            let kp = (*peak).min(last);
            for (k, c) in trace.data.iter_mut().enumerate() {
                let w = if k < kp {
                    (k as f64) / (kp as f64)
                } else if k == kp || last == kp {
                    1.0
                } else {
                    ((last - k) as f64) / ((last - kp) as f64)
                };
                c[0] *= w;
                c[1] *= w;
            }
            trace.data[0][0] *= *first_scale;
            trace.data[0][1] *= *first_scale;
        }

        StageOp::JModulation {
            j_hz,
            phase_deg,
            sin_mode,
        } => {
            let sw = trace.sw_hz.max(1e-9);
            let phi0 = phase_deg.to_radians();
            for (k, c) in trace.data.iter_mut().enumerate() {
                let ang = PI * j_hz * (k as f64) / sw + phi0;
                let m = if *sin_mode { ang.sin() } else { ang.cos() };
                c[0] *= m;
                c[1] *= m;
            }
        }

        StageOp::LinearPredict { mode } => {
            apply_linear_prediction(&mut trace.data, *mode);
            trace.sync_axis_size();
        }

        StageOp::NusReconstruct { mask, config } => {
            if mask.len() != n {
                return Err(format!(
                    "NUS mask length ({}) does not match trace length ({n})",
                    mask.len()
                ));
            }
            trace.data = reconstruct_nus_1d(&trace.data, mask, config);
        }

        StageOp::ZeroFill {
            target_size,
            factor,
            pad,
            auto_pow2,
        } => {
            let base = if let Some(ts) = target_size {
                *ts
            } else if let Some(f) = factor {
                n.saturating_mul((*f).max(1))
            } else if let Some(p) = pad {
                n.saturating_add(*p)
            } else {
                n
            }
            .max(1);
            let new_n = if *auto_pow2 {
                base.next_power_of_two()
            } else {
                base
            };
            trace.data.resize(new_n, [0.0, 0.0]);
            trace.sync_axis_size();
        }

        StageOp::FourierTransform {
            inverse,
            negate_imag,
            sign_alt,
            real_only,
        } => {
            if !inverse {
                if *real_only {
                    for c in &mut trace.data {
                        c[1] = 0.0;
                    }
                }
                if *sign_alt {
                    for (k, c) in trace.data.iter_mut().enumerate() {
                        if k % 2 == 1 {
                            c[0] = -c[0];
                            c[1] = -c[1];
                        }
                    }
                }
                if *negate_imag {
                    for c in &mut trace.data {
                        c[1] = -c[1];
                    }
                }
                fft_arbitrary_in_place(&mut trace.data, false);
                fftshift_in_place(&mut trace.data);
                if trace.pending_group_delay_pts.abs() > 1e-12 {
                    let dg = trace.pending_group_delay_pts;
                    let nf = trace.data.len() as f64;
                    for (r, c) in trace.data.iter_mut().enumerate() {
                        let f_norm = (r as f64) / nf - 0.5;
                        let phi = 2.0 * PI * dg * f_norm;
                        let (sp, cp) = phi.sin_cos();
                        let (re, im) = (c[0], c[1]);
                        *c = [re * cp - im * sp, re * sp + im * cp];
                    }
                    trace.pending_group_delay_pts = 0.0;
                }
                trace.is_freq_domain = true;
                trace.has_imaginary = true;
                trace.sync_axis_size();
            } else {
                ifftshift_in_place(&mut trace.data);
                fft_arbitrary_in_place(&mut trace.data, true);
                if *negate_imag {
                    for c in &mut trace.data {
                        c[1] = -c[1];
                    }
                }
                if *sign_alt {
                    for (k, c) in trace.data.iter_mut().enumerate() {
                        if k % 2 == 1 {
                            c[0] = -c[0];
                            c[1] = -c[1];
                        }
                    }
                }
                if *real_only {
                    for c in &mut trace.data {
                        c[1] = 0.0;
                    }
                }
                trace.is_freq_domain = false;
                trace.has_imaginary = !*real_only;
            }
        }

        StageOp::HilbertTransform => {
            hilbert_transform_in_place(&mut trace.data);
            trace.has_imaginary = true;
        }

        StageOp::Phase {
            p0_deg,
            p1_deg,
            pivot_frac,
        } => {
            apply_phase_slice(&mut trace.data, *p0_deg, *p1_deg, *pivot_frac);
        }

        StageOp::DeleteImaginary => {
            for c in &mut trace.data {
                c[1] = 0.0;
            }
            trace.has_imaginary = false;
        }

        StageOp::Magnitude => {
            for c in &mut trace.data {
                c[0] = c[0].hypot(c[1]);
                c[1] = 0.0;
            }
            trace.has_imaginary = false;
        }

        StageOp::Reverse => {
            if n > 2 {
                trace.data[1..].reverse();
            }
        }

        StageOp::CircularShift { points } => {
            let k = points.rem_euclid(n as isize) as usize;
            if k > 0 {
                trace.data.rotate_right(k);
            }
            trace.axis.first_ppm -= (*points as f64) * trace.axis.step_ppm;
        }

        StageOp::CircularShiftPpm { delta_ppm } => {
            if trace.axis.step_ppm.abs() > 1e-15 {
                let points = (-delta_ppm / trace.axis.step_ppm).round() as isize;
                let k = points.rem_euclid(n as isize) as usize;
                if k > 0 {
                    trace.data.rotate_right(k);
                }
                trace.axis.first_ppm -= (points as f64) * trace.axis.step_ppm;
            }
        }

        StageOp::Scale { factor } => {
            for c in &mut trace.data {
                c[0] *= *factor;
                c[1] *= *factor;
            }
        }

        StageOp::ExtractPpm { min_ppm, max_ppm } => {
            let lo = min_ppm.min(*max_ppm);
            let hi = min_ppm.max(*max_ppm);
            let i0 = trace.axis.ppm_to_index(hi);
            let i1 = trace.axis.ppm_to_index(lo);
            let max_idx = (n - 1) as f64;
            let s = i0.min(i1).floor().clamp(0.0, max_idx) as usize;
            let e = i0.max(i1).ceil().clamp(0.0, max_idx) as usize;
            let new_first = trace.axis.index_to_ppm(s as f64);
            trace.data = trace.data[s..=e].to_vec();
            trace.axis.first_ppm = new_first;
            trace.axis.size = trace.data.len();
            trace.sw_hz = trace.axis.sw_ppm() * trace.axis.obs_mhz;
        }

        StageOp::ExtractIndex { start, end } => {
            let s = (*start).min(n - 1);
            let e = (*end).clamp(s + 1, n);
            let new_first = trace.axis.index_to_ppm(s as f64);
            trace.data = trace.data[s..e].to_vec();
            trace.axis.first_ppm = new_first;
            trace.axis.size = trace.data.len();
            trace.sw_hz = trace.axis.sw_ppm() * trace.axis.obs_mhz;
        }

        StageOp::PolyBaseline { order, sigma_mask } => {
            let mut re: Vec<f64> = trace.data.iter().map(|c| c[0]).collect();
            subtract_poly_baseline(&mut re, *order, *sigma_mask);
            for (c, &r) in trace.data.iter_mut().zip(re.iter()) {
                c[0] = r;
            }
            if trace.has_imaginary {
                let mut im: Vec<f64> = trace.data.iter().map(|c| c[1]).collect();
                subtract_poly_baseline(&mut im, *order, *sigma_mask);
                for (c, &v) in trace.data.iter_mut().zip(im.iter()) {
                    c[1] = v;
                }
            }
        }

        StageOp::MedianBaseline { half_window_pts } => {
            let mut re: Vec<f64> = trace.data.iter().map(|c| c[0]).collect();
            subtract_median_baseline(&mut re, *half_window_pts);
            for (c, &r) in trace.data.iter_mut().zip(re.iter()) {
                c[0] = r;
            }
            if trace.has_imaginary {
                let mut im: Vec<f64> = trace.data.iter().map(|c| c[1]).collect();
                subtract_median_baseline(&mut im, *half_window_pts);
                for (c, &v) in trace.data.iter_mut().zip(im.iter()) {
                    c[1] = v;
                }
            }
        }

        StageOp::NodeSplineBaseline { node_indices } => {
            let mut re: Vec<f64> = trace.data.iter().map(|c| c[0]).collect();
            subtract_node_spline_baseline(&mut re, node_indices);
            for (c, &r) in trace.data.iter_mut().zip(re.iter()) {
                c[0] = r;
            }
            if trace.has_imaginary {
                let mut im: Vec<f64> = trace.data.iter().map(|c| c[1]).collect();
                subtract_node_spline_baseline(&mut im, node_indices);
                for (c, &v) in trace.data.iter_mut().zip(im.iter()) {
                    c[1] = v;
                }
            }
        }

        StageOp::CarrierNotch {
            center_ppm,
            width_ppm,
        } => {
            let half_w = (*width_ppm * 0.5).max(1e-6);
            for (k, c) in trace.data.iter_mut().enumerate() {
                let ppm = trace.axis.index_to_ppm(k as f64);
                let u = (ppm - center_ppm) / half_w;
                let gain = 1.0 - (-0.5 * u * u).exp();
                c[0] *= gain;
                c[1] *= gain;
            }
        }
    }
    Ok(())
}

// ============================================================================
// 6. Solvent Filters, Baseline Algorithms & 2D/3D Transpose Helpers
// ============================================================================

fn reflected_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let last = (len - 1) as isize;
    let mut k = idx;
    if k < 0 {
        k = -k;
    }
    if k > last {
        k = 2 * last - k;
    }
    k.clamp(0, last) as usize
}

/// Time-domain low-pass convolution subtraction solvent filter (Marion, Ikura & Bax 1989).
pub fn apply_solvent_filter(
    data: &mut [Complex64],
    shape: SolventKernel,
    half_width: usize,
    mirror_edges: bool,
) {
    let n = data.len();
    if n == 0 || half_width == 0 {
        return;
    }
    let orig = data.to_vec();
    let w = half_width.min(n);

    if shape == SolventKernel::PolyLowPass {
        // Local unweighted quadratic Savitzky–Golay / least-squares low-pass fit over `[i - w, i + w]`
        for (i, dst_c) in data.iter_mut().enumerate() {
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut s3 = 0.0;
            let mut s4 = 0.0;
            let mut y0 = [0.0, 0.0];
            let mut y1 = [0.0, 0.0];
            let mut y2 = [0.0, 0.0];
            let wi = w as isize;
            for d in -wi..=wi {
                let idx_raw = (i as isize) + d;
                let src = if mirror_edges {
                    orig[reflected_index(idx_raw, n)]
                } else if idx_raw >= 0 && (idx_raw as usize) < n {
                    orig[idx_raw as usize]
                } else {
                    continue;
                };
                let x = (d as f64) / (w.max(1) as f64);
                let x2 = x * x;
                s0 += 1.0;
                s1 += x;
                s2 += x2;
                s3 += x2 * x;
                s4 += x2 * x2;
                y0[0] += src[0];
                y0[1] += src[1];
                y1[0] += x * src[0];
                y1[1] += x * src[1];
                y2[0] += x2 * src[0];
                y2[1] += x2 * src[1];
            }
            let a_mat = [
                [s0 + 1e-12, s1, s2],
                [s1, s2 + 1e-12, s3],
                [s2, s3, s4 + 1e-12],
            ];
            let mut b_re = [y0[0], y1[0], y2[0]];
            let mut b_im = [y0[1], y1[1], y2[1]];
            let c_re = solve_linear_system_gauss(&mut a_mat.map(|r| r.to_vec()), &mut b_re)
                .map_or(y0[0] / s0.max(1.0), |v| v[0]);
            let c_im = solve_linear_system_gauss(&mut a_mat.map(|r| r.to_vec()), &mut b_im)
                .map_or(y0[1] / s0.max(1.0), |v| v[0]);
            dst_c[0] = orig[i][0] - c_re;
            dst_c[1] = orig[i][1] - c_im;
        }
        return;
    }

    let sigma = ((w as f64) * 0.5).max(1.0);
    let weights: Vec<f64> = (0..=w)
        .map(|d| match shape {
            SolventKernel::Boxcar => 1.0,
            SolventKernel::SineBell => (PI * (d as f64) / (2.0 * ((w + 1) as f64))).cos(),
            SolventKernel::Gaussian => (-0.5 * ((d as f64) / sigma).powi(2)).exp(),
            SolventKernel::PolyLowPass => 1.0,
        })
        .collect();

    let wi = w as isize;
    for (i, dst_c) in data.iter_mut().enumerate() {
        let mut sum = [0.0, 0.0];
        let mut wsum = 0.0;
        for d in -wi..=wi {
            let wt = weights[d.unsigned_abs()];
            let idx_raw = (i as isize) + d;
            if mirror_edges {
                let src = orig[reflected_index(idx_raw, n)];
                sum[0] += wt * src[0];
                sum[1] += wt * src[1];
                wsum += wt;
            } else if idx_raw >= 0 && (idx_raw as usize) < n {
                let src = orig[idx_raw as usize];
                sum[0] += wt * src[0];
                sum[1] += wt * src[1];
                wsum += wt;
            }
        }
        if wsum > 1e-30 {
            dst_c[0] = orig[i][0] - sum[0] / wsum;
            dst_c[1] = orig[i][1] - sum[1] / wsum;
        }
    }
}

fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len().is_multiple_of(2) {
        0.5 * (vals[mid - 1] + vals[mid])
    } else {
        vals[mid]
    }
}

fn robust_mad_sigma(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut work = vals.to_vec();
    let med = median_f64(&mut work);
    for v in &mut work {
        *v = (*v - med).abs();
    }
    1.4826 * median_f64(&mut work)
}

fn solve_linear_system_gauss(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let m = b.len();
    for col in 0..m {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().take(m).skip(col + 1) {
            if a_row[col].abs() > best {
                best = a_row[col].abs();
                piv = row;
            }
        }
        if best <= 1e-18 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        let (top, bot) = a.split_at_mut(col + 1);
        let pivot_row = &top[col];
        for (offset, a_row) in bot.iter_mut().enumerate() {
            let row = col + 1 + offset;
            let factor = a_row[col] / diag;
            b[row] -= factor * b[col];
            for (dst, &pv) in a_row[col..m].iter_mut().zip(&pivot_row[col..m]) {
                *dst -= factor * pv;
            }
        }
    }
    let mut x = vec![0.0; m];
    for row in (0..m).rev() {
        let mut s = b[row];
        for k in (row + 1)..m {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

fn eval_legendre_basis(x: f64, max_order: usize, out: &mut [f64]) {
    out[0] = 1.0;
    if max_order >= 1 {
        out[1] = x;
    }
    for m in 1..max_order {
        out[m + 1] = ((2 * m + 1) as f64 * x * out[m] - (m as f64) * out[m - 1]) / ((m + 1) as f64);
    }
}

/// Iterative MAD-masked Legendre polynomial baseline subtraction (`order = 0..=8`).
pub fn subtract_poly_baseline(y: &mut [f64], order: usize, sigma_mask: f64) {
    let n = y.len();
    if n == 0 {
        return;
    }
    let p = order.min(8).min(n.saturating_sub(1));
    let m_dim = p + 1;
    let denom = (n.saturating_sub(1)).max(1) as f64;

    let mut basis = vec![vec![0.0; m_dim]; n];
    for (k, row) in basis.iter_mut().enumerate() {
        let x = if n > 1 {
            2.0 * (k as f64) / denom - 1.0
        } else {
            0.0
        };
        eval_legendre_basis(x, p, row);
    }

    let mut mask = vec![true; n];
    let mut coeffs = vec![0.0; m_dim];
    let min_keep = (m_dim + 2).max(n / 8).min(n);

    for _iter in 0..6 {
        let mut g = vec![vec![0.0; m_dim]; m_dim];
        let mut rhs = vec![0.0; m_dim];
        for k in 0..n {
            if !mask[k] {
                continue;
            }
            let bk = &basis[k];
            let yk = y[k];
            for a_i in 0..m_dim {
                rhs[a_i] += bk[a_i] * yk;
                for b_i in a_i..m_dim {
                    g[a_i][b_i] += bk[a_i] * bk[b_i];
                }
            }
        }
        for (a_i, row_g) in g.iter_mut().enumerate().take(m_dim) {
            row_g[a_i] += 1e-12;
        }
        for a_i in 1..m_dim {
            let (upper, lower) = g.split_at_mut(a_i);
            for (b_i, up_row) in upper.iter().enumerate() {
                lower[0][b_i] = up_row[a_i];
            }
        }
        let Some(c_fit) = solve_linear_system_gauss(&mut g, &mut rhs) else {
            break;
        };
        coeffs = c_fit;

        if sigma_mask <= 0.0 {
            break;
        }

        let mut masked_res = Vec::with_capacity(n);
        let mut all_abs_res = Vec::with_capacity(n);
        for k in 0..n {
            let fit: f64 = basis[k].iter().zip(&coeffs).map(|(b, c)| b * c).sum();
            let r = y[k] - fit;
            if mask[k] {
                masked_res.push(r);
            }
            all_abs_res.push((r.abs(), k));
        }
        let sigma = robust_mad_sigma(&masked_res);
        if sigma <= 1e-15 {
            break;
        }
        let thresh = sigma_mask * sigma;
        let mut next_mask = vec![false; n];
        let mut kept = 0usize;
        for (k, &(abs_r, _)) in all_abs_res.iter().enumerate() {
            if abs_r <= thresh {
                next_mask[k] = true;
                kept += 1;
            }
        }
        if kept < min_keep {
            all_abs_res.sort_by(|a, b| a.0.total_cmp(&b.0));
            next_mask.fill(false);
            for &(_, idx) in all_abs_res.iter().take(min_keep) {
                next_mask[idx] = true;
            }
        }
        if next_mask == mask {
            break;
        }
        mask = next_mask;
    }

    for (k, yk) in y.iter_mut().enumerate() {
        let fit: f64 = basis[k].iter().zip(&coeffs).map(|(b, c)| b * c).sum();
        *yk -= fit;
    }
}

/// Friedrichs (*J. Biomol. NMR* 1995, 5, 147–153) model-free peak-masked sliding-median
/// baseline subtraction with raised-cosine smoothing.
pub fn subtract_median_baseline(y: &mut [f64], half_window_pts: usize) {
    let n = y.len();
    if n == 0 || half_window_pts == 0 {
        return;
    }
    let w = half_window_pts.min(n.saturating_sub(1)).max(1);
    let mut coarse = vec![0.0; n];
    let mut scratch = Vec::with_capacity(2 * w + 1);
    for (k, c_slot) in coarse.iter_mut().enumerate().take(n) {
        let lo = k.saturating_sub(w);
        let hi = (k + w).min(n - 1);
        scratch.clear();
        scratch.extend_from_slice(&y[lo..=hi]);
        *c_slot = median_f64(&mut scratch);
    }

    // Mask peak excursions exceeding 3.0 MAD above/below coarse median (Friedrichs 1995 §2)
    let res: Vec<f64> = y.iter().zip(&coarse).map(|(a, b)| a - b).collect();
    let sigma = robust_mad_sigma(&res);
    let mut med = coarse.clone();
    if sigma > 1e-15 {
        let thresh = 3.0 * sigma;
        let is_base: Vec<bool> = res.iter().map(|&r| r.abs() <= thresh).collect();
        for (k, m_slot) in med.iter_mut().enumerate().take(n) {
            let lo = k.saturating_sub(w);
            let hi = (k + w).min(n - 1);
            scratch.clear();
            for j in lo..=hi {
                if is_base[j] {
                    scratch.push(y[j]);
                }
            }
            if scratch.len() >= 4 {
                *m_slot = median_f64(&mut scratch);
            }
        }
    }

    let sw = (w / 2).max(1);
    let weights: Vec<f64> = (0..=sw)
        .map(|d| 0.5 * (1.0 + (PI * (d as f64) / ((sw + 1) as f64)).cos()))
        .collect();
    let swi = sw as isize;
    for (k, yk) in y.iter_mut().enumerate() {
        let mut s = 0.0;
        let mut wt_sum = 0.0;
        for d in -swi..=swi {
            let wt = weights[d.unsigned_abs()];
            let idx = reflected_index((k as isize) + d, n);
            s += wt * med[idx];
            wt_sum += wt;
        }
        *yk -= s / wt_sum.max(1e-30);
    }
}

/// Natural cubic / piecewise linear spline baseline subtraction through user-specified `node_indices`.
pub fn subtract_node_spline_baseline(y: &mut [f64], node_indices: &[usize]) {
    let n = y.len();
    if n == 0 || node_indices.is_empty() {
        return;
    }
    let mut nodes: Vec<usize> = node_indices.iter().map(|&idx| idx.min(n - 1)).collect();
    nodes.sort_unstable();
    nodes.dedup();

    if nodes.len() == 1 {
        let c = y[nodes[0]];
        for v in y.iter_mut() {
            *v -= c;
        }
        return;
    }
    if nodes.len() == 2 {
        let (x0, x1) = (nodes[0] as f64, nodes[1] as f64);
        let (y0, y1) = (y[nodes[0]], y[nodes[1]]);
        let slope = (y1 - y0) / (x1 - x0).max(1.0);
        for (k, v) in y.iter_mut().enumerate() {
            *v -= y0 + slope * ((k as f64) - x0);
        }
        return;
    }

    // Natural cubic spline (`m` knots, second derivatives `m2[0] = m2[m - 1] = 0`)
    let m = nodes.len();
    let xs: Vec<f64> = nodes.iter().map(|&i| i as f64).collect();
    let ys: Vec<f64> = nodes.iter().map(|&i| y[i]).collect();
    let mut h = vec![0.0; m - 1];
    for i in 0..(m - 1) {
        h[i] = (xs[i + 1] - xs[i]).max(1.0);
    }

    let inner = m - 2;
    let mut a_mat = vec![vec![0.0; inner]; inner];
    let mut rhs = vec![0.0; inner];
    for i in 0..inner {
        let k = i + 1;
        a_mat[i][i] = 2.0 * (h[k - 1] + h[k]);
        if i > 0 {
            a_mat[i][i - 1] = h[k - 1];
        }
        if i + 1 < inner {
            a_mat[i][i + 1] = h[k];
        }
        rhs[i] = 6.0 * ((ys[k + 1] - ys[k]) / h[k] - (ys[k] - ys[k - 1]) / h[k - 1]);
    }
    let mut m2 = vec![0.0; m];
    if let Some(sol) = solve_linear_system_gauss(&mut a_mat, &mut rhs) {
        for (i, &val) in sol.iter().enumerate() {
            m2[i + 1] = val;
        }
    }

    let mut seg = 0usize;
    for (k, v) in y.iter_mut().enumerate() {
        let x = k as f64;
        let base = if x <= xs[0] {
            let slope0 = (ys[1] - ys[0]) / h[0] - h[0] * (2.0 * m2[0] + m2[1]) / 6.0;
            ys[0] + slope0 * (x - xs[0])
        } else if x >= xs[m - 1] {
            let slopen =
                (ys[m - 1] - ys[m - 2]) / h[m - 2] + h[m - 2] * (m2[m - 2] + 2.0 * m2[m - 1]) / 6.0;
            ys[m - 1] + slopen * (x - xs[m - 1])
        } else {
            while seg + 1 < m - 1 && x > xs[seg + 1] {
                seg += 1;
            }
            let hi = h[seg];
            let a_t = (xs[seg + 1] - x) / hi;
            let b_t = (x - xs[seg]) / hi;
            a_t * ys[seg]
                + b_t * ys[seg + 1]
                + ((a_t * a_t * a_t - a_t) * m2[seg] + (b_t * b_t * b_t - b_t) * m2[seg + 1])
                    * (hi * hi / 6.0)
        };
        *v -= base;
    }
}

/// Transpose a 2D spectrum (`axis_x <-> axis_y`, `[iy * nx + ix] -> [ix * ny + iy]`).
pub fn transpose_2d(spec: &Spectrum2D) -> Spectrum2D {
    let nx = spec.nx();
    let ny = spec.ny();
    let mut out = vec![0.0f32; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            out[ix * ny + iy] = spec.data[iy * nx + ix];
        }
    }
    Spectrum2D {
        name: spec.name.clone(),
        axis_x: spec.axis_y.clone(),
        axis_y: spec.axis_x.clone(),
        data: out,
    }
}

/// Permute the axes `(0 = x, 1 = y, 2 = z)` of a `Spectrum3D` according to `perm`
/// (a permutation of `[0, 1, 2]`, e.g. `[1, 0, 2]` for `TP` X<->Y, `[2, 1, 0]` for X<->Z, `[0, 2, 1]` for Y<->Z).
pub fn transpose_3d(spec: &Spectrum3D, perm: [usize; 3]) -> Result<Spectrum3D, String> {
    let mut check = perm;
    check.sort_unstable();
    if check != [0, 1, 2] {
        return Err(format!(
            "invalid 3D axis permutation {perm:?} (expected a permutation of [0, 1, 2])"
        ));
    }
    let old_axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let old_dims = [spec.nx(), spec.ny(), spec.nz()];
    let new_axes = [
        old_axes[perm[0]].clone(),
        old_axes[perm[1]].clone(),
        old_axes[perm[2]].clone(),
    ];
    let (nnx, nny, nnz) = (new_axes[0].size, new_axes[1].size, new_axes[2].size);
    let mut out = vec![0.0f32; nnx * nny * nnz];

    let mut coords = [0usize; 3];
    for iz in 0..old_dims[2] {
        coords[2] = iz;
        for iy in 0..old_dims[1] {
            coords[1] = iy;
            let base_old = (iz * old_dims[1] + iy) * old_dims[0];
            for ix in 0..old_dims[0] {
                coords[0] = ix;
                let nx_i = coords[perm[0]];
                let ny_i = coords[perm[1]];
                let nz_i = coords[perm[2]];
                out[(nz_i * nny + ny_i) * nnx + nx_i] = spec.data[base_old + ix];
            }
        }
    }

    Ok(Spectrum3D {
        name: spec.name.clone(),
        exp: spec.exp.clone(),
        axis_x: new_axes[0].clone(),
        axis_y: new_axes[1].clone(),
        axis_z: new_axes[2].clone(),
        data: out,
    })
}

/// Swap `axis_x` and `axis_y` of a 3D cube (`perm = [1, 0, 2]`).
pub fn transpose_3d_xy(spec: &Spectrum3D) -> Spectrum3D {
    transpose_3d(spec, [1, 0, 2]).expect("valid XY permutation")
}

/// Swap `axis_x` and `axis_z` of a 3D cube (`perm = [2, 1, 0]`).
pub fn transpose_3d_xz(spec: &Spectrum3D) -> Spectrum3D {
    transpose_3d(spec, [2, 1, 0]).expect("valid XZ permutation")
}

/// Swap `axis_y` and `axis_z` of a 3D cube (`perm = [0, 2, 1]`).
pub fn transpose_3d_yz(spec: &Spectrum3D) -> Spectrum3D {
    transpose_3d(spec, [0, 2, 1]).expect("valid YZ permutation")
}

// ============================================================================
// 7. Full Multi-Dimensional Autophasing Engine (`1D` / `2D` / `3D`)
// ============================================================================

/// Expected resonance sign polarity for automatic phase correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolarityMode {
    /// All true resonances have positive sign (`HSQC`, `HNCO`, `HNCA`, `CBCACONH`, 1D `1H`/`13C`).
    AllPositive,
    /// Resonances have both positive and negative signs (`HNCACB`, DEPT, edited HSQC);
    /// optimizes dispersive-free Lorentzian symmetry and locks global `0 deg` vs `180 deg`
    /// polarity so `biphasic_pos_band_ppm` (`CA`: `48..66 ppm`) is positive.
    Biphasic,
}

/// Configuration for 1D, direct N-D, and indirect N-D automatic phase correction
/// (Chen et al. 2002; de Brouwer 2009).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoPhaseConfig {
    pub polarity: PolarityMode,
    /// Coarse scan step in degrees for zero-order phase `p0` over `[-180, 180)`.
    pub coarse_step_deg: f64,
    /// Maximum magnitude of first-order phase `|p1_deg|` explored on direct axes.
    pub max_abs_p1_deg: f64,
    /// Enable continuous refinement of first-order phase `p1_deg`.
    pub refine_p1: bool,
    /// Weight of negative-absorption penalty in `PolarityMode::AllPositive` (Chen et al. 2002).
    pub neg_penalty_weight: f64,
    /// Weight of local peak-wing antisymmetry penalty (de Brouwer 2009).
    pub wing_asymmetry_weight: f64,
    /// Excluded water/solvent carrier notch `(min_ppm, max_ppm)` on `1H` axes.
    pub water_mask_ppm: Option<(f64, f64)>,
    /// Preferred downfield amide evaluation window `(min_ppm, max_ppm)` on `1H` axes
    /// (falls back automatically to full non-water axis when `< 5%` energy lies inside).
    pub direct_roi_ppm: Option<(f64, f64)>,
    /// Positive-polarity reference band `(min_ppm, max_ppm)` in `PolarityMode::Biphasic`
    /// (`CA` chemical-shift band `48.0..66.0 ppm` for `HNCACB`).
    pub biphasic_pos_band_ppm: (f64, f64),
    /// Maximum number of high-SNR traces/columns selected for N-D autophasing.
    pub top_traces: usize,
}

impl Default for AutoPhaseConfig {
    fn default() -> Self {
        Self {
            polarity: PolarityMode::AllPositive,
            coarse_step_deg: 5.0,
            max_abs_p1_deg: 90.0,
            refine_p1: true,
            neg_penalty_weight: 2.0,
            wing_asymmetry_weight: 4.5,
            water_mask_ppm: Some((4.55, 5.05)),
            direct_roi_ppm: Some((5.80, 10.60)),
            biphasic_pos_band_ppm: (48.0, 66.0),
            top_traces: 32,
        }
    }
}

impl AutoPhaseConfig {
    /// Preset for `HNCACB` / biphasic spectra (`PolarityMode::Biphasic`, `CA` positive in `48..66 ppm`).
    pub fn biphasic() -> Self {
        Self {
            polarity: PolarityMode::Biphasic,
            ..Self::default()
        }
    }
}

/// Recovered zero-order and first-order phase angles (`pivot_frac = 0.0`, i.e. index `0` pivot)
/// and final objective score.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseSolution {
    pub p0_deg: f64,
    pub p1_deg: f64,
    pub score: f64,
}

/// Normalize an angle in degrees into `[-180.0, 180.0)`.
#[inline]
pub fn wrap_deg_180(deg: f64) -> f64 {
    let mut x = (deg + 180.0).rem_euclid(360.0) - 180.0;
    if (x - 180.0).abs() < 1e-12 {
        x = -180.0;
    }
    x
}

/// Shortest signed angular difference `wrap_deg_180(a_deg - b_deg)` in `[-180, 180)`.
#[inline]
pub fn phase_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    wrap_deg_180(a_deg - b_deg)
}

/// Unsigned angular difference modulo `180 deg` in `[0, 90]`.
#[inline]
pub fn phase_diff_mod180_deg(a_deg: f64, b_deg: f64) -> f64 {
    let d1 = phase_diff_deg(a_deg, b_deg).abs();
    let d2 = phase_diff_deg(a_deg + 180.0, b_deg).abs();
    d1.min(d2)
}

fn is_proton_like_axis(axis: &AxisPpm) -> bool {
    let up = axis.label.to_ascii_uppercase();
    if up.contains('N') || up.contains('C') || up.contains('P') || up.contains('F') {
        return false;
    }
    up.contains('H') || (axis.obs_mhz >= 200.0 && axis.max_ppm() <= 25.0 && axis.min_ppm() >= -8.0)
}

fn build_active_indices(
    traces: &[&[Complex64]],
    axis: &AxisPpm,
    config: &AutoPhaseConfig,
    is_direct: bool,
) -> Vec<usize> {
    let n = axis.size;
    if n == 0 {
        return Vec::new();
    }
    let proton = is_direct && is_proton_like_axis(axis);
    let in_water = |ppm: f64| -> bool {
        if !proton {
            return false;
        }
        if let Some((w0, w1)) = config.water_mask_ppm {
            let lo = w0.min(w1);
            let hi = w0.max(w1);
            ppm >= lo && ppm <= hi
        } else {
            false
        }
    };

    let non_water: Vec<usize> = (0..n)
        .filter(|&k| !in_water(axis.index_to_ppm(k as f64)))
        .collect();
    if non_water.len() < 8 {
        return (0..n).collect();
    }

    if proton {
        if let Some((r0, r1)) = config.direct_roi_ppm {
            let lo = r0.min(r1);
            let hi = r0.max(r1);
            let roi: Vec<usize> = non_water
                .iter()
                .copied()
                .filter(|&k| {
                    let ppm = axis.index_to_ppm(k as f64);
                    ppm >= lo && ppm <= hi
                })
                .collect();
            if roi.len() >= 16 {
                let e_non_water: f64 = traces
                    .iter()
                    .take(16)
                    .map(|tr| non_water.iter().map(|&k| c_abs2(tr[k])).sum::<f64>())
                    .sum();
                let e_roi: f64 = traces
                    .iter()
                    .take(16)
                    .map(|tr| roi.iter().map(|&k| c_abs2(tr[k])).sum::<f64>())
                    .sum();
                if e_roi >= 0.04 * e_non_water && e_roi > 1e-30 {
                    return roi;
                }
            }
        }
    }
    non_water
}

/// Fast 2-pass robust quadratic detrend (`1, x, x^2`) on an active slice `r` to prevent any
/// residual unphased solvent skirt from biasing amide lineshape symmetry.
fn detrend_quadratic_in_place(r: &mut [f64]) {
    let m = r.len();
    if m < 64 {
        let mut tmp = r.to_vec();
        let med = median_f64(&mut tmp);
        for v in r.iter_mut() {
            *v -= med;
        }
        return;
    }

    let denom = (m - 1) as f64;
    let fit_masked = |vals: &[f64], mask: Option<&[bool]>| -> [f64; 3] {
        let mut g = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for (j, &y) in vals.iter().enumerate() {
            if let Some(mk) = mask {
                if !mk[j] {
                    continue;
                }
            }
            let x = 2.0 * (j as f64) / denom - 1.0;
            let p = [1.0, x, 1.5 * x * x - 0.5];
            for a_i in 0..3 {
                b[a_i] += p[a_i] * y;
                for b_i in a_i..3 {
                    g[a_i][b_i] += p[a_i] * p[b_i];
                }
            }
        }
        for (a_i, row_g) in g.iter_mut().enumerate() {
            row_g[a_i] += 1e-12;
        }
        g[1][0] = g[0][1];
        g[2][0] = g[0][2];
        g[2][1] = g[1][2];
        let mut gv: Vec<Vec<f64>> = g.iter().map(|row| row.to_vec()).collect();
        let mut bv = b;
        if let Some(sol) = solve_linear_system_gauss(&mut gv, &mut bv) {
            [sol[0], sol[1], sol[2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    };

    let c1 = fit_masked(r, None);
    let mut res = Vec::with_capacity(m);
    for (j, &y) in r.iter().enumerate() {
        let x = 2.0 * (j as f64) / denom - 1.0;
        let fit = c1[0] + c1[1] * x + c1[2] * (1.5 * x * x - 0.5);
        res.push(y - fit);
    }
    let sig = robust_mad_sigma(&res);
    let coeffs = if sig > 1e-15 {
        let thresh = 2.5 * sig;
        let mask: Vec<bool> = res.iter().map(|&v| v.abs() <= thresh).collect();
        let kept = mask.iter().filter(|&&b| b).count();
        if kept >= 12 {
            fit_masked(r, Some(&mask))
        } else {
            c1
        }
    } else {
        c1
    };

    for (j, y) in r.iter_mut().enumerate() {
        let x = 2.0 * (j as f64) / denom - 1.0;
        *y -= coeffs[0] + coeffs[1] * x + coeffs[2] * (1.5 * x * x - 0.5);
    }
}

#[derive(Clone, Copy, Debug)]
struct EnvelopePeak {
    /// Sub-bin peak center in `0 .. m` (`active_idx` coordinate).
    center: f64,
    /// Baseline-corrected complex magnitude envelope at `center`.
    amp: f64,
    /// Square-root amplitude weight so edge and central peaks anchor `p1` evenly.
    weight: f64,
    /// Largest symmetric step `d` (`1 <= d <= max_step`) free of neighbor peaks or ROI gaps.
    max_step: usize,
    /// Symmetric foot offset `w_foot` (`2.0`, `4.0`, `6.0`, or `1.0`) for `Z(x0) - 0.5*(Z(x0+w)+Z(x0-w))`.
    w_foot: f64,
}

#[inline]
fn interp_catmull_rom(buf: &[f64], x: f64) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let xf = x.clamp(0.0, (n - 1) as f64);
    let i1 = (xf.floor() as usize).min(n - 1);
    let t = xf - (i1 as f64);
    if t <= 1e-12 {
        return buf[i1];
    }
    let i0 = i1.saturating_sub(1);
    let i2 = (i1 + 1).min(n - 1);
    let i3 = (i1 + 2).min(n - 1);
    let (p0, p1, p2, p3) = (buf[i0], buf[i1], buf[i2], buf[i3]);
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Zero-mean symmetric raised-cosine inner product of `buf` over `[center - w, center + w]`:
/// `sum_{k} (cos(pi u_k / (2 w)) - mean_cos) * [buf(center + u_k) + buf(center - u_k)]`.
///
/// Cancels constant offsets (including `1 / df` neighbor dispersion skirts) and linear slopes to
/// machine precision (`0.0`), avoids single-point boundary spikes, and remains strictly positive
/// for symmetric peaks of arbitrary zero-filling width by Chebyshev's inequality.
#[inline]
fn symmetric_zero_mean_cosine_proj(buf: &[f64], center: f64, w: f64) -> f64 {
    const N_HALF: usize = 8;
    let n = buf.len();
    if n < 3 || w <= 1e-9 {
        return 0.0;
    }
    let max_x = (n - 1) as f64;
    let w_eff = w.min(center).min(max_x - center);
    if w_eff <= 1e-6 {
        return 0.0;
    }

    let mut c_sum = 0.0;
    let mut c_wt = [0.0f64; N_HALF];
    for (k, cw) in c_wt.iter_mut().enumerate() {
        let frac = ((k as f64) + 0.5) / (N_HALF as f64);
        *cw = (0.5 * PI * frac).cos();
        c_sum += *cw;
    }
    let c_mean = c_sum / (N_HALF as f64);

    let mut acc = 0.0;
    for (k, &cw) in c_wt.iter().enumerate() {
        let u = (((k as f64) + 0.5) / (N_HALF as f64)) * w_eff;
        let pair_sum = interp_catmull_rom(buf, center + u) + interp_catmull_rom(buf, center - u);
        acc += (cw - c_mean) * pair_sum;
    }
    acc
}

fn detect_envelope_peaks(cplx: &[Complex64], active_idx: &[usize]) -> Vec<EnvelopePeak> {
    let m = active_idx.len();
    if m < 8 {
        return Vec::new();
    }
    let mut re_tmp: Vec<f64> = active_idx.iter().map(|&k| cplx[k][0]).collect();
    let mut im_tmp: Vec<f64> = active_idx.iter().map(|&k| cplx[k][1]).collect();
    let med_re = median_f64(&mut re_tmp);
    let med_im = median_f64(&mut im_tmp);

    let env: Vec<f64> = active_idx
        .iter()
        .map(|&k| (cplx[k][0] - med_re).hypot(cplx[k][1] - med_im))
        .collect();

    let mut env_copy = env.clone();
    let q50 = median_f64(&mut env_copy);
    for v in &mut env_copy {
        *v = (*v - q50).abs();
    }
    let mad_env = 1.4826 * median_f64(&mut env_copy);
    let pk_thresh = (q50 + 3.6 * mad_env).max(1e-20);

    let mut cands: Vec<(f64, usize)> = Vec::new();
    for j in 2..(m - 2) {
        if active_idx[j + 2] - active_idx[j - 2] == 4 {
            let e0 = env[j];
            if e0 >= pk_thresh && e0 >= env[j - 1] && e0 > env[j + 1] {
                cands.push((e0, j));
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));

    let wing_cap = (m / 24).clamp(4, 10);
    let mut selected_j: Vec<usize> = Vec::new();
    let mut peaks = Vec::new();

    for &(_, jp) in &cands {
        if selected_j.iter().any(|&q| q.abs_diff(jp) < 2) {
            continue;
        }
        let (em, e0, ep) = (
            env[jp - 1].max(1e-30),
            env[jp].max(1e-30),
            env[jp + 1].max(1e-30),
        );
        // Exact Lorentzian sub-bin centering: `1 / env(x)^2 = (1 + ((x - x0)/gamma)^2) / A^2` is
        // an exact quadratic parabola in `x`!
        let (um, u0, up) = (1.0 / (em * em), 1.0 / (e0 * e0), 1.0 / (ep * ep));
        let u_denom = um - 2.0 * u0 + up;
        let delta = if u_denom > 1e-30 {
            (0.5 * (um - up) / u_denom).clamp(-0.5, 0.5)
        } else {
            let denom = em - 2.0 * e0 + ep;
            if denom.abs() > 1e-30 {
                (0.5 * (em - ep) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        };
        let center = (jp as f64) + delta;
        let u_min = (u0 - 0.25 * (um - up) * delta).max(0.25 * u0);
        let amp = (1.0 / u_min.sqrt()).max(e0).max(1e-20);

        let mut max_step = 0usize;
        let mut ep_prev = amp;
        let mut em_prev = amp;
        for d in 1..=wing_cap {
            let df = d as f64;
            if center - df - 1.0 < 0.0 || center + df + 1.0 > (m - 1) as f64 {
                break;
            }
            let lo = ((center - df).floor() as usize).saturating_sub(1);
            let hi = (((center + df).ceil() as usize) + 1).min(m - 1);
            if active_idx[hi] - active_idx[lo] != hi - lo {
                break;
            }
            let ep_d = interp_catmull_rom(&env, center + df);
            let em_d = interp_catmull_rom(&env, center - df);
            if d >= 2
                && (ep_d > ep_prev * 1.03
                    || em_d > em_prev * 1.03
                    || (ep_d - em_d).abs() > 0.20 * amp)
            {
                break;
            }
            max_step = d;
            ep_prev = ep_d;
            em_prev = em_d;
        }

        if max_step >= 1 {
            let w_foot = (max_step.min(9) as f64).max(1.25);
            selected_j.push(jp);
            peaks.push(EnvelopePeak {
                center,
                amp,
                weight: amp.sqrt(),
                max_step,
                w_foot,
            });
            if peaks.len() >= 32 {
                break;
            }
        }
    }

    peaks
}

struct PreparedTrace<'a> {
    cplx: &'a [Complex64],
    weight: f64,
    peaks: Vec<EnvelopePeak>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_phase_objective(
    traces: &[PreparedTrace<'_>],
    n_full: usize,
    active_idx: &[usize],
    p0_deg: f64,
    p1_deg: f64,
    config: &AutoPhaseConfig,
    apply_neg_penalty: bool,
    detrend_quad: bool,
) -> f64 {
    let m = active_idx.len();
    if m < 4 || traces.is_empty() || n_full == 0 {
        return 0.0;
    }

    let nf = n_full as f64;
    let mut cos_tab = Vec::with_capacity(m);
    let mut sin_tab = Vec::with_capacity(m);
    for &k in active_idx {
        let phi = (p0_deg + p1_deg * ((k as f64) / nf)).to_radians();
        let (sp, cp) = phi.sin_cos();
        cos_tab.push(cp);
        sin_tab.push(sp);
    }

    let mut r_buf = vec![0.0; m];
    let mut q_buf = vec![0.0; m];
    let mut d_buf = Vec::with_capacity(m);
    let mut total_cost = 0.0;

    for tr in traces {
        for j in 0..m {
            let k = active_idx[j];
            let c = tr.cplx[k];
            r_buf[j] = c[0] * cos_tab[j] - c[1] * sin_tab[j];
            q_buf[j] = c[0] * sin_tab[j] + c[1] * cos_tab[j];
        }

        if detrend_quad {
            detrend_quadratic_in_place(&mut r_buf);
            detrend_quadratic_in_place(&mut q_buf);
        } else {
            let mut tmp = r_buf.clone();
            let med = median_f64(&mut tmp);
            for v in &mut r_buf {
                *v -= med;
            }
            let mut qtmp = q_buf.clone();
            let qmed = median_f64(&mut qtmp);
            for v in &mut q_buf {
                *v -= qmed;
            }
        }

        // 1. Noise-thresholded first-derivative Shannon entropy (Chen et al. 2002):
        // Used ONLY alongside the negative-pixel penalty (`apply_neg_penalty == true`); without
        // `p_neg` (`Biphasic` or multi-increment oscillating cosine traces), `|dR|` entropy alone
        // favors sharp zero-crossing dispersive slopes (`D'(0) = 1 / gamma`).
        let h_acme = if apply_neg_penalty {
            d_buf.clear();
            for j in 0..(m - 1) {
                if active_idx[j + 1] == active_idx[j] + 1 {
                    d_buf.push((r_buf[j + 1] - r_buf[j]).abs());
                }
            }
            if d_buf.len() >= 4 {
                let sigma_d = robust_mad_sigma(&d_buf);
                let thr_d = 2.5 * sigma_d;
                let mut d_sum = 0.0;
                for &dv in &d_buf {
                    if dv > thr_d {
                        d_sum += dv - thr_d;
                    }
                }
                if d_sum > 1e-30 {
                    let inv = 1.0 / d_sum;
                    let mut ent = 0.0;
                    for &dv in &d_buf {
                        if dv > thr_d {
                            let h = (dv - thr_d) * inv;
                            if h > 1e-30 {
                                ent -= h * h.ln();
                            }
                        }
                    }
                    ent / (d_buf.len() as f64).ln()
                } else {
                    let raw_sum: f64 = d_buf.iter().sum();
                    if raw_sum > 1e-30 {
                        let inv = 1.0 / raw_sum;
                        let mut ent = 0.0;
                        for &dv in &d_buf {
                            let h = dv * inv;
                            if h > 1e-30 {
                                ent -= h * h.ln();
                            }
                        }
                        ent / (d_buf.len() as f64).ln()
                    } else {
                        1.0
                    }
                }
            } else {
                1.0
            }
        } else {
            0.0
        };

        // 2. Sub-bin Kramers–Kronig zero-mean symmetric raised-cosine phasor + flank symmetry
        // (Hoch & Stern 1996; de Brouwer 2009):
        // Cancels constant offsets (including `1 / df` neighbor dispersive skirts) and linear
        // baselines while remaining strictly positive across arbitrary zero-filling ratios.
        // At pure absorption (`psi == 0 deg` for `+CA` or `psi == 180 deg` for `-CB`), `s_im == 0`,
        // whereas at dispersion (`psi == +-90 deg`), `s_im^2 / (s_re^2 + s_im^2) == 1.0`!
        let sigma = robust_mad_sigma(&r_buf);
        let mut wing_wsum = 0.0;
        let mut wing_acc = 0.0;
        for pk in &tr.peaks {
            let wf = pk.w_foot;
            let s_re = symmetric_zero_mean_cosine_proj(&r_buf, pk.center, wf);
            let s_im = symmetric_zero_mean_cosine_proj(&q_buf, pk.center, wf);
            let s_mag2 = s_re * s_re + s_im * s_im;
            let disp_sin2 = if s_mag2 > 1e-30 {
                s_im * s_im / s_mag2
            } else {
                0.0
            };
            let neg_phasor = if apply_neg_penalty && s_mag2 > 1e-30 {
                let s_mag = s_mag2.sqrt();
                // `0.5 * (1 - cos(psi)) = sin^2(psi / 2)`: 0 at +absorption, 0.5 at +-90, 1.0 at -inverted!
                0.75 * (1.0 - (s_re / s_mag).clamp(-1.0, 1.0))
            } else {
                0.0
            };

            let n_flank = pk.max_step.min(3);
            let mut flank_err = 0.0;
            for d in 1..=n_flank {
                let df = d as f64;
                let r_plus = interp_catmull_rom(&r_buf, pk.center + df);
                let r_minus = interp_catmull_rom(&r_buf, pk.center - df);
                let diff_norm = ((r_plus - r_minus) / pk.amp).clamp(-1.5, 1.5);
                flank_err += diff_norm * diff_norm;
            }
            flank_err /= n_flank as f64;

            wing_acc += pk.weight * (disp_sin2 + 0.35 * flank_err + neg_phasor);
            wing_wsum += pk.weight;
        }

        // 3. Negative-absorption penalty (`AllPositive` single-sign traces only)
        let p_neg = if apply_neg_penalty {
            let tau = (2.5 * sigma).max(1e-20);
            let mut neg_e = 0.0;
            let mut tot_e = 0.0;
            for &rv in &r_buf {
                let av = rv.abs();
                if av > tau {
                    let ex = av - tau;
                    tot_e += ex * ex;
                    if rv < -tau {
                        neg_e += ex * ex;
                    }
                }
            }
            if tot_e > 1e-30 {
                neg_e / tot_e
            } else {
                let n2: f64 = r_buf.iter().filter(|&&v| v < 0.0).map(|&v| v * v).sum();
                let t2: f64 = r_buf.iter().map(|&v| v * v).sum();
                if t2 > 1e-30 {
                    n2 / t2
                } else {
                    0.0
                }
            }
        } else {
            0.0
        };

        let acme_w = if apply_neg_penalty { 0.15 } else { 0.0 };
        let trace_cost = if wing_wsum > 1e-30 {
            let a_wing = wing_acc / wing_wsum;
            acme_w * h_acme
                + config.wing_asymmetry_weight * a_wing
                + config.neg_penalty_weight * p_neg
        } else {
            h_acme + config.neg_penalty_weight * p_neg
        };
        total_cost += tr.weight * trace_cost;
    }

    total_cost
}

fn compute_energy_centroid_frac(
    traces: &[PreparedTrace<'_>],
    n_full: usize,
    active_idx: &[usize],
) -> f64 {
    if n_full == 0 || active_idx.is_empty() {
        return 0.5;
    }
    let nf = n_full as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for tr in traces {
        if !tr.peaks.is_empty() {
            let pk_wsum: f64 = tr.peaks.iter().map(|pk| pk.weight).sum();
            if pk_wsum > 1e-30 {
                for pk in &tr.peaks {
                    let i0 = (pk.center.floor() as usize).min(active_idx.len() - 1);
                    let i1 = (i0 + 1).min(active_idx.len() - 1);
                    let f = pk.center - (i0 as f64);
                    let k_interp =
                        (active_idx[i0] as f64) * (1.0 - f) + (active_idx[i1] as f64) * f;
                    let w = tr.weight * (pk.weight / pk_wsum);
                    num += w * (k_interp / nf);
                    den += w;
                }
                continue;
            }
        }
        let tr_e2: f64 = active_idx.iter().map(|&k| c_abs2(tr.cplx[k])).sum();
        if tr_e2 > 1e-30 {
            for &k in active_idx {
                let w = tr.weight * (c_abs2(tr.cplx[k]) / tr_e2);
                num += w * ((k as f64) / nf);
                den += w;
            }
        }
    }
    if den > 1e-30 {
        (num / den).clamp(0.05, 0.95)
    } else {
        0.5
    }
}

/// Centroid-decoupled Powell / pattern refinement around `(p0_init, p1_init)`.
/// Optimizes along orthogonalized directions `(1, 0)` (pure zero-order shift) and
/// `(-f_bar, 1)` (pure first-order tilt around signal centroid `f_bar`).
#[allow(clippy::too_many_arguments)]
fn refine_phase_powell(
    traces: &[PreparedTrace<'_>],
    n_full: usize,
    active_idx: &[usize],
    p0_init: f64,
    p1_init: f64,
    p1_anchor: f64,
    p1_reg_weight: f64,
    p1_bounds: (f64, f64),
    config: &AutoPhaseConfig,
    apply_neg_penalty: bool,
    detrend_quad: bool,
) -> PhaseSolution {
    let f_bar = compute_energy_centroid_frac(traces, n_full, active_idx);
    let eval = |p0: f64, p1: f64| -> f64 {
        let base = evaluate_phase_objective(
            traces,
            n_full,
            active_idx,
            p0,
            p1,
            config,
            apply_neg_penalty,
            detrend_quad,
        );
        let dp1 = (p1 - p1_anchor) / 90.0;
        base + p1_reg_weight * dp1 * dp1
    };

    let mut p0 = wrap_deg_180(p0_init);
    let mut p1 = p1_init.clamp(p1_bounds.0, p1_bounds.1);
    let mut best = eval(p0, p1);

    let steps: &[(f64, f64)] = &[
        (4.0, 12.0),
        (2.0, 6.0),
        (0.8, 2.5),
        (0.25, 0.8),
        (0.05, 0.2),
    ];

    for &(s0, s1) in steps {
        for _pass in 0..8 {
            let mut improved = false;

            // Direction 1: pure `p0`
            for sign in [-1.0, 1.0] {
                let cand_p0 = wrap_deg_180(p0 + sign * s0);
                let c = eval(cand_p0, p1);
                if c + 1e-12 < best {
                    best = c;
                    p0 = cand_p0;
                    improved = true;
                }
            }

            // Direction 2: centroid-decoupled `p1` tilt (`dp0 = -f_bar * dp1`)
            if config.refine_p1 && p1_bounds.1 > p1_bounds.0 + 1e-6 {
                for sign in [-1.0, 1.0] {
                    let dp1 = sign * s1;
                    let cand_p1 = (p1 + dp1).clamp(p1_bounds.0, p1_bounds.1);
                    let actual_dp1 = cand_p1 - p1;
                    if actual_dp1.abs() <= 1e-9 {
                        continue;
                    }
                    let cand_p0 = wrap_deg_180(p0 - f_bar * actual_dp1);
                    let c = eval(cand_p0, cand_p1);
                    if c + 1e-12 < best {
                        best = c;
                        p0 = cand_p0;
                        p1 = cand_p1;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }
    }

    PhaseSolution {
        p0_deg: wrap_deg_180(p0),
        p1_deg: p1,
        score: best,
    }
}

/// Compute net signed cubic/quadratic strong-peak energy over `indices` at `(p0_deg, p1_deg)`.
/// Positive return value indicates dominant positive absorption peaks; negative indicates inverted (`180 deg`).
fn net_signed_peak_polarity(
    traces: &[&[Complex64]],
    n_full: usize,
    indices: &[usize],
    p0_deg: f64,
    p1_deg: f64,
    detrend_quad: bool,
) -> f64 {
    let m = indices.len();
    if m == 0 || traces.is_empty() || n_full == 0 {
        return 0.0;
    }
    let nf = n_full as f64;
    let mut r_buf = vec![0.0; m];
    let mut net = 0.0;
    for tr in traces {
        for (j, &k) in indices.iter().enumerate() {
            let phi = (p0_deg + p1_deg * ((k as f64) / nf)).to_radians();
            let (sp, cp) = phi.sin_cos();
            r_buf[j] = tr[k][0] * cp - tr[k][1] * sp;
        }
        if detrend_quad {
            detrend_quadratic_in_place(&mut r_buf);
        } else {
            let mut tmp = r_buf.clone();
            let med = median_f64(&mut tmp);
            for v in &mut r_buf {
                *v -= med;
            }
        }
        let sigma = robust_mad_sigma(&r_buf);
        let tau = (3.0 * sigma).max(1e-20);
        let mut tr_net = 0.0;
        let mut any_above = false;
        for &rv in &r_buf {
            if rv.abs() > tau {
                any_above = true;
                let ex = rv.abs() - tau;
                tr_net += rv.signum() * ex * ex;
            }
        }
        if !any_above {
            for &rv in &r_buf {
                tr_net += rv.signum() * rv * rv;
            }
        }
        net += tr_net;
    }
    net
}

/// Automatically phase a single complex 1D frequency-domain spectrum `spec_cplx`
/// (`spec_cplx.len() == axis.size`).
pub fn auto_phase_1d(
    spec_cplx: &[Complex64],
    axis: &AxisPpm,
    config: &AutoPhaseConfig,
) -> PhaseSolution {
    if spec_cplx.len() < 4 {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }
    let owned_ht: Vec<Complex64>;
    let has_im = spec_cplx.iter().any(|c| c[1].abs() > 1e-15);
    let cplx_ref: &[Complex64] = if has_im {
        spec_cplx
    } else {
        let mut h = spec_cplx.to_vec();
        hilbert_transform_in_place(&mut h);
        owned_ht = h;
        &owned_ht
    };

    let slice_refs = [cplx_ref];
    let active_idx = build_active_indices(&slice_refs, axis, config, true);
    let prep = [PreparedTrace {
        cplx: cplx_ref,
        weight: 1.0,
        peaks: detect_envelope_peaks(cplx_ref, &active_idx),
    }];
    let apply_neg = config.polarity == PolarityMode::AllPositive;
    let max_p1 = if config.refine_p1 {
        config.max_abs_p1_deg.abs()
    } else {
        0.0
    };
    let step = config.coarse_step_deg.clamp(1.0, 30.0);

    let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
    let p1_grid: Vec<f64> = if max_p1 <= 1e-6 {
        vec![0.0]
    } else {
        let mut g = vec![0.0];
        let mut v = 30.0;
        while v <= max_p1 + 1e-6 {
            g.push(v);
            g.push(-v);
            v += 30.0;
        }
        g
    };

    let n_p0 = ((360.0 / step).round() as usize).max(12);
    for &p1_c in &p1_grid {
        for i in 0..n_p0 {
            let p0_c = -180.0 + (i as f64) * (360.0 / (n_p0 as f64));
            let sc = evaluate_phase_objective(
                &prep,
                cplx_ref.len(),
                &active_idx,
                p0_c,
                p1_c,
                config,
                apply_neg,
                true,
            ) + 2e-5 * (p1_c / 90.0).powi(2);
            seeds.push((sc, p0_c, p1_c));
        }
    }
    seeds.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Keep top 4 well-separated seeds for local Powell refinement
    let mut distinct: Vec<(f64, f64)> = Vec::new();
    for &(_, s_p0, s_p1) in &seeds {
        if distinct
            .iter()
            .all(|&(q0, q1)| phase_diff_deg(s_p0, q0).abs() > 15.0 || (s_p1 - q1).abs() > 25.0)
        {
            distinct.push((s_p0, s_p1));
            if distinct.len() >= 4 {
                break;
            }
        }
    }

    let mut best_sol = PhaseSolution {
        p0_deg: 0.0,
        p1_deg: 0.0,
        score: f64::INFINITY,
    };
    for (s_p0, s_p1) in distinct {
        let sol = refine_phase_powell(
            &prep,
            cplx_ref.len(),
            &active_idx,
            s_p0,
            s_p1,
            0.0,
            2e-5,
            (-max_p1, max_p1),
            config,
            apply_neg,
            true,
        );
        if sol.score < best_sol.score {
            best_sol = sol;
        }
    }

    if config.polarity == PolarityMode::AllPositive {
        let net = net_signed_peak_polarity(
            &slice_refs,
            cplx_ref.len(),
            &active_idx,
            best_sol.p0_deg,
            best_sol.p1_deg,
            true,
        );
        if net < 0.0 {
            best_sol.p0_deg = wrap_deg_180(best_sol.p0_deg + 180.0);
        }
    } else {
        // Biphasic 1D: if axis overlaps `biphasic_pos_band_ppm`, lock positive band
        let (b0, b1) = (
            config
                .biphasic_pos_band_ppm
                .0
                .min(config.biphasic_pos_band_ppm.1),
            config
                .biphasic_pos_band_ppm
                .0
                .max(config.biphasic_pos_band_ppm.1),
        );
        let pos_idx: Vec<usize> = (0..cplx_ref.len())
            .filter(|&k| {
                let ppm = axis.index_to_ppm(k as f64);
                ppm >= b0 && ppm <= b1
            })
            .collect();
        if pos_idx.len() >= 4 {
            let net = net_signed_peak_polarity(
                &slice_refs,
                cplx_ref.len(),
                &pos_idx,
                best_sol.p0_deg,
                best_sol.p1_deg,
                false,
            );
            if net < 0.0 {
                best_sol.p0_deg = wrap_deg_180(best_sol.p0_deg + 180.0);
            }
        }
    }

    best_sol
}

/// Automatically phase a real-only 1D frequency trace by reconstructing its causal imaginary
/// partner via `HilbertTransform` first.
pub fn auto_phase_1d_real(
    spec_real: &[f64],
    axis: &AxisPpm,
    config: &AutoPhaseConfig,
) -> PhaseSolution {
    let mut cplx: Vec<Complex64> = spec_real.iter().map(|&r| [r, 0.0]).collect();
    hilbert_transform_in_place(&mut cplx);
    auto_phase_1d(&cplx, axis, config)
}

fn select_high_snr_traces<'a>(
    traces: &'a [Vec<Complex64>],
    active_idx: &[usize],
    max_keep: usize,
) -> Vec<PreparedTrace<'a>> {
    let mut scored: Vec<(f64, &'a [Complex64])> = traces
        .iter()
        .filter_map(|tr| {
            let e2: f64 = active_idx.iter().map(|&k| c_abs2(tr[k])).sum();
            (e2.is_finite() && e2 > 1e-30).then_some((e2, tr.as_slice()))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(max_keep.max(1));

    // Mild `e2^0.15` weighting prevents a single flexible C-terminal residue (`10x` taller)
    // from dominating the remaining globular residues.
    let w_sum: f64 = scored.iter().map(|(e2, _)| e2.powf(0.15)).sum();
    if w_sum <= 1e-30 {
        return Vec::new();
    }
    scored
        .into_iter()
        .map(|(e2, cplx)| PreparedTrace {
            cplx,
            weight: e2.powf(0.15) / w_sum,
            peaks: detect_envelope_peaks(cplx, active_idx),
        })
        .collect()
}

/// Jointly autophase the direct (`1H`) dimension of a 2D/3D dataset from its complex
/// direct-dimension frequency traces (`traces_freq_cplx`, where `traces_freq_cplx[0]` is the
/// first `t_ind = 0` cosine trace).
///
/// Note: Across indirect increments `k1 > 0`, `cos(omega_N * t1)` oscillates in sign across
/// residues even in `AllPositive` experiments! Therefore dispersive-free Lorentzian lineshape
/// symmetry (`ACME` derivative entropy + de Brouwer wing symmetry) is optimized across the
/// top `config.top_traces` high-SNR traces, and global `0 deg` vs `180 deg` sign in
/// `AllPositive` mode is locked on the `t_ind = 0` cosine increment (`traces_freq_cplx[0]`).
pub fn auto_phase_direct_nd(
    traces_freq_cplx: &[Vec<Complex64>],
    axis_x: &AxisPpm,
    config: &AutoPhaseConfig,
) -> PhaseSolution {
    let n = axis_x.size;
    if traces_freq_cplx.is_empty() || n < 4 {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    let sample_refs: Vec<&[Complex64]> = traces_freq_cplx
        .iter()
        .take(32)
        .filter(|t| t.len() == n)
        .map(Vec::as_slice)
        .collect();
    if sample_refs.is_empty() {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    let active_idx = build_active_indices(&sample_refs, axis_x, config, true);
    let valid_traces: Vec<Vec<Complex64>> = traces_freq_cplx
        .iter()
        .filter(|t| t.len() == n)
        .map(|t| {
            if t.iter().any(|c| c[1].abs() > 1e-15) {
                t.clone()
            } else {
                let mut h = t.clone();
                hilbert_transform_in_place(&mut h);
                h
            }
        })
        .collect();

    let prep = select_high_snr_traces(&valid_traces, &active_idx, config.top_traces);
    if prep.is_empty() {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    // Optimize dispersive-free lineshape (`apply_neg_penalty = false` on multi-increment traces)
    let max_p1 = if config.refine_p1 {
        config.max_abs_p1_deg.abs()
    } else {
        0.0
    };
    let step = config.coarse_step_deg.clamp(1.0, 30.0);
    // Because lineshape symmetry is 180-deg periodic, scanning `[-90, 90)` at step `step`
    // covers the full modulo-180 manifold without duplicate mirrors!
    let n_p0 = ((180.0 / step).round() as usize).max(12);
    let p1_grid: Vec<f64> = if max_p1 <= 1e-6 {
        vec![0.0]
    } else {
        let mut g = vec![0.0];
        let mut v = 30.0;
        while v <= max_p1 + 1e-6 {
            g.push(v);
            g.push(-v);
            v += 30.0;
        }
        g
    };

    let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
    for &p1_c in &p1_grid {
        for i in 0..n_p0 {
            let p0_c = -90.0 + (i as f64) * (180.0 / (n_p0 as f64));
            let sc =
                evaluate_phase_objective(&prep, n, &active_idx, p0_c, p1_c, config, false, true)
                    + 8e-3 * (p1_c / 90.0).powi(2);
            seeds.push((sc, p0_c, p1_c));
        }
    }
    seeds.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut distinct: Vec<(f64, f64)> = Vec::new();
    for &(_, s_p0, s_p1) in &seeds {
        if distinct
            .iter()
            .all(|&(q0, q1)| phase_diff_mod180_deg(s_p0, q0) > 12.0 || (s_p1 - q1).abs() > 25.0)
        {
            distinct.push((s_p0, s_p1));
            if distinct.len() >= 4 {
                break;
            }
        }
    }

    let mut best_sol = PhaseSolution {
        p0_deg: 0.0,
        p1_deg: 0.0,
        score: f64::INFINITY,
    };
    for (s_p0, s_p1) in distinct {
        let sol = refine_phase_powell(
            &prep,
            n,
            &active_idx,
            s_p0,
            s_p1,
            0.0,
            8e-3,
            (-max_p1, max_p1),
            config,
            false,
            true,
        );
        if sol.score < best_sol.score {
            best_sol = sol;
        }
    }

    // Lock `0 deg` vs `180 deg` polarity on the `t_ind = 0` first cosine increment (`valid_traces[0]`)
    if config.polarity == PolarityMode::AllPositive {
        let first_ref = [valid_traces[0].as_slice()];
        let net0 = net_signed_peak_polarity(
            &first_ref,
            n,
            &active_idx,
            best_sol.p0_deg,
            best_sol.p1_deg,
            true,
        );
        if net0 < 0.0 {
            best_sol.p0_deg = wrap_deg_180(best_sol.p0_deg + 180.0);
        }
    } else {
        // In `Biphasic` mode on direct 1H, choose the canonical representative with smaller |p0|
        // (e.g. `0 deg` over `180 deg`), or positive first-trace net energy if tied; final 3D
        // `CA`-band (`48..66 ppm`) polarity is locked unambiguously on the `13C` axis!
        let alt_p0 = wrap_deg_180(best_sol.p0_deg + 180.0);
        if alt_p0.abs() + 1e-6 < best_sol.p0_deg.abs() {
            best_sol.p0_deg = alt_p0;
        }
    }

    best_sol
}

/// Autophase an indirect (`15N` or `13C`) dimension from 1D indirect frequency columns
/// (accepts either complex columns or real-only columns — real-only columns are automatically
/// Hilbert-transformed to reconstruct their causal imaginary channel).
///
/// Evaluates canonical pulse-sequence sampling-delay phase states `(0, 0)`, `(-90, +180)`
/// (half-dwell), `(+90, -180)`, `(180, 0)`, `(-180, +360)` (full-dwell), and `(+180, -360)`,
/// plus a full coarse scan, followed by continuous fine `(p0, p1)` refinement.
/// In `PolarityMode::Biphasic`, when `axis_ind` overlaps `config.biphasic_pos_band_ppm`
/// (`48..66 ppm` `CA` band), global `0 deg` vs `180 deg` polarity is locked so `CA` is positive!
pub fn auto_phase_indirect_nd(
    columns_freq: &[Vec<Complex64>],
    axis_ind: &AxisPpm,
    config: &AutoPhaseConfig,
) -> PhaseSolution {
    let n = axis_ind.size;
    if columns_freq.is_empty() || n < 4 {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    let valid_cols: Vec<Vec<Complex64>> = columns_freq
        .iter()
        .filter(|c| c.len() == n)
        .map(|col| {
            let max_re = col.iter().map(|v| v[0].abs()).fold(0.0_f64, f64::max);
            let max_im = col.iter().map(|v| v[1].abs()).fold(0.0_f64, f64::max);
            if max_im <= 1e-10 * max_re.max(1e-30) {
                let mut h = col.clone();
                hilbert_transform_in_place(&mut h);
                h
            } else {
                col.clone()
            }
        })
        .collect();
    if valid_cols.is_empty() {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    let active_idx: Vec<usize> = (0..n).collect();
    let prep = select_high_snr_traces(&valid_cols, &active_idx, config.top_traces);
    if prep.is_empty() {
        return PhaseSolution {
            p0_deg: 0.0,
            p1_deg: 0.0,
            score: 0.0,
        };
    }

    let apply_neg = config.polarity == PolarityMode::AllPositive;

    let top_refs: Vec<&[Complex64]> = prep.iter().map(|p| p.cplx).collect();
    let (ca_lo, ca_hi) = (
        config
            .biphasic_pos_band_ppm
            .0
            .min(config.biphasic_pos_band_ppm.1),
        config
            .biphasic_pos_band_ppm
            .0
            .max(config.biphasic_pos_band_ppm.1),
    );
    let ca_band_idx: Vec<usize> = if config.polarity == PolarityMode::Biphasic {
        (0..n)
            .filter(|&k| {
                let ppm = axis_ind.index_to_ppm(k as f64);
                ppm >= ca_lo && ppm <= ca_hi
            })
            .collect()
    } else {
        Vec::new()
    };
    let cb_band_idx: Vec<usize> = if config.polarity == PolarityMode::Biphasic {
        (0..n)
            .filter(|&k| {
                let ppm = axis_ind.index_to_ppm(k as f64);
                (14.0..=38.0).contains(&ppm)
            })
            .collect()
    } else {
        Vec::new()
    };
    let check_biphasic_opp = ca_band_idx.len() >= 4 && cb_band_idx.len() >= 4;
    let biphasic_same_sign_pen = |p0: f64, p1: f64| -> f64 {
        if !check_biphasic_opp {
            return 0.0;
        }
        let n_ca = net_signed_peak_polarity(&top_refs, n, &ca_band_idx, p0, p1, false);
        let n_cb = net_signed_peak_polarity(&top_refs, n, &cb_band_idx, p0, p1, false);
        if n_ca.abs() > 1e-30 && n_cb.abs() > 1e-30 && n_ca * n_cb >= 0.0 {
            1.5
        } else {
            0.0
        }
    };

    // 1. Canonical indirect sampling-delay phase states + coarse grid
    // Indirect axes have no FIR digital filter; physical `p1` arises from initial evolution delay
    // `t0 / dt in {0, 0.5, 1.0}` (`p1 in {0, +-180, +-360}`), with zero-delay `p1 = 0` most common.
    let delay_prior = |p1: f64| -> f64 {
        let a = p1.abs();
        if a < 1.0 {
            0.0
        } else if (a - 180.0).abs() <= 20.0 {
            0.025
        } else {
            0.12 + 0.02 * (a / 180.0)
        }
    };

    let canonical: &[(f64, f64)] = &[
        (0.0, 0.0),
        (-180.0, 0.0),
        (-90.0, 180.0),
        (90.0, -180.0),
        (-180.0, 360.0),
        (180.0, -360.0),
    ];

    let mut candidates: Vec<(f64, f64, f64)> = Vec::new();
    for &(cp0, cp1) in canonical {
        let sc =
            evaluate_phase_objective(&prep, n, &active_idx, cp0, cp1, config, apply_neg, false)
                + delay_prior(cp1)
                + biphasic_same_sign_pen(cp0, cp1)
                - 5e-4;
        candidates.push((sc, cp0, cp1));
    }

    let step = config.coarse_step_deg.clamp(2.0, 30.0);
    let n_p0 = ((360.0 / step).round() as usize).max(12);
    let coarse_p1s: &[f64] = &[0.0, 180.0, -180.0, 360.0, -360.0];
    for &p1_c in coarse_p1s {
        let pr = delay_prior(p1_c);
        for i in 0..n_p0 {
            let p0_c = -180.0 + (i as f64) * (360.0 / (n_p0 as f64));
            let sc = evaluate_phase_objective(
                &prep,
                n,
                &active_idx,
                p0_c,
                p1_c,
                config,
                apply_neg,
                false,
            ) + pr
                + biphasic_same_sign_pen(p0_c, p1_c);
            candidates.push((sc, p0_c, p1_c));
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut seeds: Vec<(f64, f64)> = Vec::new();
    for &(_, s_p0, s_p1) in &candidates {
        if seeds
            .iter()
            .all(|&(q0, q1)| phase_diff_deg(s_p0, q0).abs() > 15.0 || (s_p1 - q1).abs() > 45.0)
        {
            seeds.push((s_p0, s_p1));
            if seeds.len() >= 5 {
                break;
            }
        }
    }

    let mut best_sol = PhaseSolution {
        p0_deg: 0.0,
        p1_deg: 0.0,
        score: f64::INFINITY,
    };
    for (s_p0, s_p1) in seeds {
        let p1_win = if config.refine_p1 { 15.0 } else { 0.0 };
        let mut sol = refine_phase_powell(
            &prep,
            n,
            &active_idx,
            s_p0,
            s_p1,
            s_p1,
            8e-3,
            (s_p1 - p1_win, s_p1 + p1_win),
            config,
            apply_neg,
            false,
        );
        sol.score += delay_prior(s_p1) + biphasic_same_sign_pen(sol.p0_deg, sol.p1_deg);
        if sol.score < best_sol.score {
            best_sol = sol;
        }
    }

    if config.polarity == PolarityMode::AllPositive {
        let net = net_signed_peak_polarity(
            &top_refs,
            n,
            &active_idx,
            best_sol.p0_deg,
            best_sol.p1_deg,
            false,
        );
        if net < 0.0 {
            best_sol.p0_deg = wrap_deg_180(best_sol.p0_deg + 180.0);
        }
    } else {
        // Biphasic mode (e.g. `13C` in `HNCACB`): lock `CA` band (`48..66 ppm`) positive!
        let (b0, b1) = (
            config
                .biphasic_pos_band_ppm
                .0
                .min(config.biphasic_pos_band_ppm.1),
            config
                .biphasic_pos_band_ppm
                .0
                .max(config.biphasic_pos_band_ppm.1),
        );
        let ca_idx: Vec<usize> = (0..n)
            .filter(|&k| {
                let ppm = axis_ind.index_to_ppm(k as f64);
                ppm >= b0 && ppm <= b1
            })
            .collect();
        if ca_idx.len() >= 4 {
            let net_ca = net_signed_peak_polarity(
                &top_refs,
                n,
                &ca_idx,
                best_sol.p0_deg,
                best_sol.p1_deg,
                false,
            );
            if net_ca < 0.0 {
                best_sol.p0_deg = wrap_deg_180(best_sol.p0_deg + 180.0);
            }
        } else {
            let alt_p0 = wrap_deg_180(best_sol.p0_deg + 180.0);
            if alt_p0.abs() + 1e-6 < best_sol.p0_deg.abs() {
                best_sol.p0_deg = alt_p0;
            }
        }
    }

    best_sol
}

/// Verify and lock global `0 deg` vs `180 deg` polarity of a 3D `Biphasic` (`HNCACB`) cube
/// so that strong peaks in `ca_band_ppm` (`48..66 ppm` on the `13C` axis) are strictly positive.
/// Returns `true` if the cube was inverted (`* -1.0`).
pub fn lock_biphasic_ca_polarity_3d(spec: &mut Spectrum3D, ca_band_ppm: (f64, f64)) -> bool {
    let (lo, hi) = (
        ca_band_ppm.0.min(ca_band_ppm.1),
        ca_band_ppm.0.max(ca_band_ppm.1),
    );
    let z_is_c = spec.axis_z.label.to_ascii_uppercase().contains('C')
        || (spec.axis_z.min_ppm() <= lo && spec.axis_z.max_ppm() >= hi);
    let y_is_c = !z_is_c
        && (spec.axis_y.label.to_ascii_uppercase().contains('C')
            || (spec.axis_y.min_ppm() <= lo && spec.axis_y.max_ppm() >= hi));
    if !z_is_c && !y_is_c {
        return false;
    }

    // Estimate global cube MAD from a strided subsample
    let step = (spec.data.len() / 16_384).max(1);
    let mut sub: Vec<f64> = spec.data.iter().step_by(step).map(|&v| v as f64).collect();
    let med = median_f64(&mut sub);
    for v in &mut sub {
        *v = (*v - med).abs();
    }
    let sigma = (1.4826 * median_f64(&mut sub)).max(1e-12);
    let thresh = (4.5 * sigma) as f32;

    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    let mut net_ca = 0.0f64;
    if z_is_c {
        for iz in 0..nz {
            let ppm = spec.axis_z.index_to_ppm(iz as f64);
            if ppm < lo || ppm > hi {
                continue;
            }
            let z_off = iz * ny * nx;
            for &v in &spec.data[z_off..z_off + ny * nx] {
                if v.abs() > thresh {
                    let vf = v as f64;
                    net_ca += vf.signum() * vf * vf;
                }
            }
        }
    } else {
        for iz in 0..nz {
            for iy in 0..ny {
                let ppm = spec.axis_y.index_to_ppm(iy as f64);
                if ppm < lo || ppm > hi {
                    continue;
                }
                let off = (iz * ny + iy) * nx;
                for &v in &spec.data[off..off + nx] {
                    if v.abs() > thresh {
                        let vf = v as f64;
                        net_ca += vf.signum() * vf * vf;
                    }
                }
            }
        }
    }

    if net_ca < 0.0 {
        for v in &mut spec.data {
            *v = -*v;
        }
        true
    } else {
        false
    }
}

fn default_apod_op(meta: &FidAxisMeta, first_scale: f64) -> StageOp {
    match meta.stored_proc.window {
        Some(StoredWindowHint::Exponential { lb_hz }) if lb_hz > 0.0 => {
            StageOp::ExpWindow { lb_hz, first_scale }
        }
        Some(StoredWindowHint::SineBell { offset_pi_frac }) => StageOp::SineWindow {
            off: offset_pi_frac,
            end: 1.0,
            pow: 1.0,
            first_scale,
        },
        Some(StoredWindowHint::SquaredSineBell { offset_pi_frac }) => StageOp::SineWindow {
            off: offset_pi_frac,
            end: 1.0,
            pow: 2.0,
            first_scale,
        },
        _ => StageOp::SineWindow {
            off: 0.5,
            end: 1.0,
            pow: 2.0,
            first_scale,
        },
    }
}

/// Blindly autophase and process a raw 2D `Fid2D` **completely ignoring stored `procs`/`proc2s`
/// phases (`PHC0`, `PHC1`)**, returning `(direct_phase, indirect_phase, spectrum_2d)`.
pub fn blind_autophase_fid_2d(
    fid: &Fid2D,
    config: &AutoPhaseConfig,
) -> Result<(PhaseSolution, PhaseSolution, Spectrum2D), String> {
    let n1 = fid.axis_y.complex_points;
    let si_x = fid
        .axis_x
        .stored_proc
        .si
        .unwrap_or_else(|| (2 * fid.axis_x.complex_points).next_power_of_two());
    let si_y = fid
        .axis_y
        .stored_proc
        .si
        .unwrap_or_else(|| (2 * n1).next_power_of_two());

    // 1. Transform direct 1H traces WITHOUT phase (`GroupDelay -> SolventFilter -> Cosine2 -> ZeroFill -> FFT`)
    let direct_pipe = AxisPipeline::new(vec![
        StageOp::GroupDelay {
            delay_pts: fid.axis_x.group_delay,
        },
        StageOp::SolventFilter {
            shape: SolventKernel::Gaussian,
            half_width: 32,
            mirror_edges: true,
        },
        default_apod_op(&fid.axis_x, 0.5),
        StageOp::ZeroFill {
            target_size: Some(si_x),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: fid.axis_x.conjugate_for_fft,
            sign_alt: false,
            real_only: false,
        },
    ]);

    let mut x_cos_cplx = Vec::with_capacity(n1);
    let mut x_sin_cplx = Vec::with_capacity(n1);
    let mut axis_x_out: Option<AxisPpm> = None;
    for k1 in 0..n1 {
        let mut tc = Trace1D::from_fid_axis(&fid.axis_x, fid.trace(k1, 0).to_vec());
        direct_pipe.apply(&mut tc)?;
        if axis_x_out.is_none() {
            axis_x_out = Some(tc.axis.clone());
        }
        x_cos_cplx.push(tc.data);

        let mut ts = Trace1D::from_fid_axis(&fid.axis_x, fid.trace(k1, 1).to_vec());
        direct_pipe.apply(&mut ts)?;
        x_sin_cplx.push(ts.data);
    }
    let axis_x = axis_x_out.ok_or("empty 2D FID")?;
    let nx = axis_x.size;

    // 2. Blindly autophase direct `1H` on the cosine traces (`x_cos_cplx[0]` = `t1 = 0`).
    // When FIR `group_delay` has already been compensated, residual physical `DE` first-order phase is small (`<= 30 deg`).
    let mut cfg_direct = config.clone();
    if fid.axis_x.group_delay > 0.0 {
        cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(18.0);
    }
    let sol_x = auto_phase_direct_nd(&x_cos_cplx, &axis_x, &cfg_direct);

    // Apply `sol_x` + `DeleteImaginary` + linear baseline to all `2 * n1` traces
    let mut x_cos_re = vec![vec![0.0; nx]; n1];
    let mut x_sin_re = vec![vec![0.0; nx]; n1];
    for k1 in 0..n1 {
        apply_phase_slice(&mut x_cos_cplx[k1], sol_x.p0_deg, sol_x.p1_deg, 0.0);
        apply_phase_slice(&mut x_sin_cplx[k1], sol_x.p0_deg, sol_x.p1_deg, 0.0);
        for ix in 0..nx {
            x_cos_re[k1][ix] = x_cos_cplx[k1][ix][0];
            x_sin_re[k1][ix] = x_sin_cplx[k1][ix][0];
        }
    }

    // 3. Transform indirect `F1` (`15N`/`13C`) columns WITHOUT phase
    let mut ind_pipe = AxisPipeline::new(vec![
        default_apod_op(&fid.axis_y, 0.5),
        StageOp::ZeroFill {
            target_size: Some(si_y),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: fid.axis_y.conjugate_for_fft,
            sign_alt: false,
            real_only: false,
        },
    ]);
    if fid.axis_y.stored_proc.reverse {
        ind_pipe.push(StageOp::Reverse);
    }

    let mut cols_cplx: Vec<Vec<Complex64>> = Vec::with_capacity(nx);
    let mut axis_y_out: Option<AxisPpm> = None;
    for ix in 0..nx {
        let mut raw_col = Vec::with_capacity(n1);
        for k1 in 0..n1 {
            raw_col.push([x_cos_re[k1][ix], x_sin_re[k1][ix]]);
        }
        if let Some(mask) = &fid.nus_mask {
            raw_col = reconstruct_nus_1d(&raw_col, mask, &NusConfig::default());
        }
        let mut ty = Trace1D::from_fid_axis(&fid.axis_y, raw_col);
        ind_pipe.apply(&mut ty)?;
        if axis_y_out.is_none() {
            axis_y_out = Some(ty.axis.clone());
        }
        cols_cplx.push(ty.data);
    }
    let axis_y = axis_y_out.ok_or("empty indirect axis")?;
    let ny = axis_y.size;

    // Select indirect autophasing columns at distinct 1D `1H` local-maximum apices!
    let sample_x_refs: Vec<&[Complex64]> = x_cos_cplx.iter().take(8).map(Vec::as_slice).collect();
    let active_ix = build_active_indices(&sample_x_refs, &axis_x, config, true);
    let mut col_e: Vec<f64> = vec![0.0; nx];
    for &ix in &active_ix {
        col_e[ix] = cols_cplx[ix].iter().map(|&c| c_abs2(c)).sum();
    }
    let mut apex_ix: Vec<(f64, usize)> = active_ix
        .iter()
        .copied()
        .filter(|&ix| {
            let e0 = col_e[ix];
            let el = if ix > 0 { col_e[ix - 1] } else { 0.0 };
            let er = if ix + 1 < nx { col_e[ix + 1] } else { 0.0 };
            e0 > 0.0 && e0 >= el && e0 >= er
        })
        .map(|ix| (col_e[ix], ix))
        .collect();
    apex_ix.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen_ix: Vec<usize> = Vec::new();
    for &(_, ix) in &apex_ix {
        if chosen_ix.iter().all(|&q| q.abs_diff(ix) >= 3) {
            chosen_ix.push(ix);
            if chosen_ix.len() >= 48 {
                break;
            }
        }
    }
    let cand_cols: Vec<Vec<Complex64>> = if chosen_ix.len() >= 4 {
        chosen_ix.iter().map(|&ix| cols_cplx[ix].clone()).collect()
    } else {
        active_ix.iter().map(|&ix| cols_cplx[ix].clone()).collect()
    };
    let sol_y = auto_phase_indirect_nd(&cand_cols, &axis_y, config);

    let mut data = vec![0.0f32; ny * nx];
    for (ix, col) in cols_cplx.iter_mut().enumerate() {
        apply_phase_slice(col, sol_y.p0_deg, sol_y.p1_deg, 0.0);
        for (iy, c) in col.iter().enumerate() {
            data[iy * nx + ix] = c[0] as f32;
        }
    }

    Ok((
        sol_x,
        sol_y,
        Spectrum2D {
            name: fid.name.clone(),
            axis_x,
            axis_y,
            data,
        },
    ))
}

/// Blindly autophase and process a raw 3D `Fid3D` **completely ignoring stored `procs`/`proc2s`/`proc3s`
/// phases (`PHC0`, `PHC1`)**, returning `(sol_x, sol_y, sol_z, spectrum_3d)`.
pub fn blind_autophase_fid_3d(
    fid: &Fid3D,
    exp: &str,
    config: &AutoPhaseConfig,
) -> Result<(PhaseSolution, PhaseSolution, PhaseSolution, Spectrum3D), String> {
    let n1 = fid.axis_y.complex_points;
    let n2 = fid.axis_z.complex_points;
    let si_x = fid
        .axis_x
        .stored_proc
        .si
        .unwrap_or_else(|| (2 * fid.axis_x.complex_points).next_power_of_two());
    let si_y = fid
        .axis_y
        .stored_proc
        .si
        .unwrap_or_else(|| (2 * n1).next_power_of_two());
    let si_z = fid
        .axis_z
        .stored_proc
        .si
        .unwrap_or_else(|| (2 * n2).next_power_of_two());

    let direct_pipe = AxisPipeline::new(vec![
        StageOp::GroupDelay {
            delay_pts: fid.axis_x.group_delay,
        },
        StageOp::SolventFilter {
            shape: SolventKernel::Gaussian,
            half_width: 32,
            mirror_edges: true,
        },
        default_apod_op(&fid.axis_x, 0.5),
        StageOp::ZeroFill {
            target_size: Some(si_x),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: fid.axis_x.conjugate_for_fft,
            sign_alt: false,
            real_only: false,
        },
    ]);

    // 1. Sample early `(k2, k1)` cosine-cosine direct traces (`(0,0)` first!) to autophase `1H`
    let mut sample_x_traces: Vec<Vec<Complex64>> = Vec::new();
    let mut axis_x_out: Option<AxisPpm> = None;
    for k2 in 0..n2.min(8) {
        for k1 in 0..n1.min(8) {
            if let Some(mask) = &fid.nus_mask {
                if !mask[k2 * n1 + k1] {
                    continue;
                }
            }
            let mut tr = Trace1D::from_fid_axis(&fid.axis_x, fid.trace(k2, 0, k1, 0).to_vec());
            direct_pipe.apply(&mut tr)?;
            if axis_x_out.is_none() {
                axis_x_out = Some(tr.axis.clone());
            }
            sample_x_traces.push(tr.data);
        }
    }
    let axis_x = axis_x_out.ok_or("no sampled traces in 3D FID")?;
    let nx = axis_x.size;
    let mut cfg_direct = config.clone();
    if fid.axis_x.group_delay > 0.0 {
        cfg_direct.max_abs_p1_deg = cfg_direct.max_abs_p1_deg.min(18.0);
    }
    let sol_x = auto_phase_direct_nd(&sample_x_traces, &axis_x, &cfg_direct);

    let sample_x_refs: Vec<&[Complex64]> =
        sample_x_traces.iter().take(8).map(Vec::as_slice).collect();
    let active_ix = build_active_indices(&sample_x_refs, &axis_x, config, true);

    // Transform & phase all `4 * n1 * n2` direct traces, discarding `1H` imaginary channel
    let total_planes = n2 * 2 * n1 * 2;
    let mut x_real = vec![0.0f64; total_planes * nx];
    for k2 in 0..n2 {
        for c2 in 0..2 {
            for k1 in 0..n1 {
                if let Some(mask) = &fid.nus_mask {
                    if !mask[k2 * n1 + k1] {
                        continue;
                    }
                }
                for c1 in 0..2 {
                    let mut tr =
                        Trace1D::from_fid_axis(&fid.axis_x, fid.trace(k2, c2, k1, c1).to_vec());
                    direct_pipe.apply(&mut tr)?;
                    apply_phase_slice(&mut tr.data, sol_x.p0_deg, sol_x.p1_deg, 0.0);
                    let p_idx = ((k2 * 2 + c2) * n1 + k1) * 2 + c1;
                    let dst = &mut x_real[p_idx * nx..(p_idx + 1) * nx];
                    for (ix, c) in tr.data.iter().enumerate() {
                        dst[ix] = c[0];
                    }
                }
            }
        }
    }

    // 2. Transform `axis_y` (`F2`, e.g. `15N`) along `k1`
    let mut pipe_y = AxisPipeline::new(vec![
        default_apod_op(&fid.axis_y, 0.5),
        StageOp::ZeroFill {
            target_size: Some(si_y),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: fid.axis_y.conjugate_for_fft,
            sign_alt: false,
            real_only: false,
        },
    ]);
    if fid.axis_y.stored_proc.reverse {
        pipe_y.push(StageOp::Reverse);
    }

    // Collect candidate `y` columns from `k2 = 0..min(4, n2), c2 = 0` at distinct `1H` local-maximum apices
    let mut ix_y_energy = vec![0.0f64; nx];
    for k2 in 0..n2.min(4) {
        for &ix in &active_ix {
            let mut e = 0.0;
            for k1 in 0..n1 {
                let p0 = ((k2 * 2) * n1 + k1) * 2;
                let re = x_real[p0 * nx + ix];
                let im = x_real[(p0 + 1) * nx + ix];
                e += re * re + im * im;
            }
            ix_y_energy[ix] += e;
        }
    }
    let mut y_apex_ix: Vec<(f64, usize)> = active_ix
        .iter()
        .copied()
        .filter(|&ix| {
            let e0 = ix_y_energy[ix];
            let el = if ix > 0 { ix_y_energy[ix - 1] } else { 0.0 };
            let er = if ix + 1 < nx {
                ix_y_energy[ix + 1]
            } else {
                0.0
            };
            e0 > 0.0 && e0 >= el && e0 >= er
        })
        .map(|ix| (ix_y_energy[ix], ix))
        .collect();
    y_apex_ix.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut sel_y_ix: Vec<usize> = Vec::new();
    for &(_, ix) in &y_apex_ix {
        if sel_y_ix.iter().all(|&q| q.abs_diff(ix) >= 3) {
            sel_y_ix.push(ix);
            if sel_y_ix.len() >= 32 {
                break;
            }
        }
    }
    if sel_y_ix.len() < 4 {
        sel_y_ix = active_ix.clone();
    }

    let mut cand_y_cols: Vec<Vec<Complex64>> = Vec::new();
    let mut axis_y_out: Option<AxisPpm> = None;
    for k2 in 0..n2.min(4) {
        for &ix in &sel_y_ix {
            let mut raw_y = Vec::with_capacity(n1);
            for k1 in 0..n1 {
                let p0 = ((k2 * 2) * n1 + k1) * 2;
                let p1 = p0 + 1;
                raw_y.push([x_real[p0 * nx + ix], x_real[p1 * nx + ix]]);
            }
            let mut ty = Trace1D::from_fid_axis(&fid.axis_y, raw_y);
            pipe_y.apply(&mut ty)?;
            if axis_y_out.is_none() {
                axis_y_out = Some(ty.axis.clone());
            }
            cand_y_cols.push(ty.data);
        }
    }
    let axis_y = axis_y_out.ok_or("empty axis_y in 3D FID")?;
    let ny = axis_y.size;
    let mut sol_y = auto_phase_indirect_nd(&cand_y_cols, &axis_y, config);

    // Transform & phase `axis_y` across the full cube -> `[n2 * 2 * ny * nx]` real values
    let mut xy_real = vec![0.0f64; n2 * 2 * ny * nx];
    for k2 in 0..n2 {
        for c2 in 0..2 {
            for ix in 0..nx {
                let mut raw_y = Vec::with_capacity(n1);
                for k1 in 0..n1 {
                    let p0 = ((k2 * 2 + c2) * n1 + k1) * 2;
                    let p1 = p0 + 1;
                    raw_y.push([x_real[p0 * nx + ix], x_real[p1 * nx + ix]]);
                }
                let mut ty = Trace1D::from_fid_axis(&fid.axis_y, raw_y);
                pipe_y.apply(&mut ty)?;
                apply_phase_slice(&mut ty.data, sol_y.p0_deg, sol_y.p1_deg, 0.0);
                for (iy, c) in ty.data.iter().enumerate() {
                    xy_real[((k2 * 2 + c2) * ny + iy) * nx + ix] = c[0];
                }
            }
        }
    }
    drop(x_real);

    // 3. Transform `axis_z` (`F1`, e.g. `13C`) along `k2`
    let mut pipe_z = AxisPipeline::new(vec![
        default_apod_op(&fid.axis_z, 0.5),
        StageOp::ZeroFill {
            target_size: Some(si_z),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: fid.axis_z.conjugate_for_fft,
            sign_alt: false,
            real_only: false,
        },
    ]);
    if fid.axis_z.stored_proc.reverse {
        pipe_z.push(StageOp::Reverse);
    }

    // Compute 2D `(iy, ix)` `k2` energy map and select distinct 2D amide peak apices via 2D NMS
    // so `cand_z_cols` samples diverse residues across the entire `13C` chemical-shift range
    // at `D_x(0) == 0, D_y(0) == 0` rather than 64 adjacent pixels of a single tail peak!
    let mut e_grid = vec![0.0f64; ny * nx];
    for iy in 0..ny {
        for &ix in &active_ix {
            let mut e = 0.0;
            for k2 in 0..n2 {
                let re = xy_real[((k2 * 2) * ny + iy) * nx + ix];
                let im = xy_real[((k2 * 2 + 1) * ny + iy) * nx + ix];
                e += re * re + im * im;
            }
            e_grid[iy * nx + ix] = e;
        }
    }
    let mut yx_maxima: Vec<(f64, usize, usize)> = Vec::new();
    for iy in 1..ny.saturating_sub(1) {
        for &ix in &active_ix {
            if ix == 0 || ix + 1 >= nx {
                continue;
            }
            let e0 = e_grid[iy * nx + ix];
            if e0 <= 0.0 {
                continue;
            }
            let is_max = e0 >= e_grid[(iy - 1) * nx + ix]
                && e0 >= e_grid[(iy + 1) * nx + ix]
                && e0 >= e_grid[iy * nx + ix - 1]
                && e0 >= e_grid[iy * nx + ix + 1];
            if is_max {
                yx_maxima.push((e0, iy, ix));
            }
        }
    }
    yx_maxima.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut yx_selected: Vec<(f64, usize, usize)> = Vec::new();
    for &(e0, iy, ix) in &yx_maxima {
        if yx_selected
            .iter()
            .all(|&(_, qy, qx)| iy.abs_diff(qy) >= 2 || ix.abs_diff(qx) >= 4)
        {
            yx_selected.push((e0, iy, ix));
            if yx_selected.len() >= 64 {
                break;
            }
        }
    }
    if yx_selected.len() < 4 {
        for iy in 0..ny {
            for &ix in &active_ix {
                yx_selected.push((e_grid[iy * nx + ix], iy, ix));
            }
        }
        yx_selected.sort_by(|a, b| b.0.total_cmp(&a.0));
        yx_selected.truncate(64);
    }

    let mut cand_z_cols: Vec<Vec<Complex64>> = Vec::with_capacity(yx_selected.len());
    let mut axis_z_out: Option<AxisPpm> = None;
    for &(_, iy, ix) in &yx_selected {
        let mut raw_z = Vec::with_capacity(n2);
        for k2 in 0..n2 {
            let re = xy_real[((k2 * 2) * ny + iy) * nx + ix];
            let im = xy_real[((k2 * 2 + 1) * ny + iy) * nx + ix];
            raw_z.push([re, im]);
        }
        let mut tz = Trace1D::from_fid_axis(&fid.axis_z, raw_z);
        pipe_z.apply(&mut tz)?;
        if axis_z_out.is_none() {
            axis_z_out = Some(tz.axis.clone());
        }
        cand_z_cols.push(tz.data);
    }
    let axis_z = axis_z_out.ok_or("empty axis_z in 3D FID")?;
    let nz = axis_z.size;
    let mut sol_z = auto_phase_indirect_nd(&cand_z_cols, &axis_z, config);

    let mut cube_data = vec![0.0f32; nz * ny * nx];
    for iy in 0..ny {
        for ix in 0..nx {
            let mut raw_z = Vec::with_capacity(n2);
            for k2 in 0..n2 {
                let re = xy_real[((k2 * 2) * ny + iy) * nx + ix];
                let im = xy_real[((k2 * 2 + 1) * ny + iy) * nx + ix];
                raw_z.push([re, im]);
            }
            let mut tz = Trace1D::from_fid_axis(&fid.axis_z, raw_z);
            pipe_z.apply(&mut tz)?;
            apply_phase_slice(&mut tz.data, sol_z.p0_deg, sol_z.p1_deg, 0.0);
            for (iz, c) in tz.data.iter().enumerate() {
                cube_data[(iz * ny + iy) * nx + ix] = c[0] as f32;
            }
        }
    }

    let mut spec3 = Spectrum3D {
        name: fid.name.clone(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data: cube_data,
    };

    if config.polarity == PolarityMode::Biphasic
        && lock_biphasic_ca_polarity_3d(&mut spec3, config.biphasic_pos_band_ppm)
    {
        // Record the 180-deg global flip on the non-13C indirect axis (`sol_y`) or `sol_z`
        if spec3.axis_z.label.to_ascii_uppercase().contains('C') {
            sol_z.p0_deg = wrap_deg_180(sol_z.p0_deg + 180.0);
        } else {
            sol_y.p0_deg = wrap_deg_180(sol_y.p0_deg + 180.0);
        }
    }

    Ok((sol_x, sol_y, sol_z, spec3))
}

// ============================================================================
// 8. 1D `.ft1` & Multi-File 3D Z-Plane Series (`test%03d.ft3`) I/O Helpers
// ============================================================================

const FT_HEADER_FLOATS: usize = 512;
const FT_HEADER_BYTES: usize = 2048;
const FT_SW_IDX: [usize; 4] = [100, 229, 11, 29];
const FT_ORIG_IDX: [usize; 4] = [101, 249, 12, 30];
const FT_OBS_IDX: [usize; 4] = [119, 218, 10, 28];
const FT_LABEL_IDX: [usize; 4] = [16, 18, 20, 22];

/// Processed 1D NMR spectrum supporting real-only or complex (`imag`) storage.
#[derive(Clone, Debug)]
pub struct Spectrum1D {
    pub name: String,
    pub axis_x: AxisPpm,
    pub data: Vec<f32>,
    pub imag: Option<Vec<f32>>,
}

impl Spectrum1D {
    #[inline]
    pub fn nx(&self) -> usize {
        self.axis_x.size
    }
}

fn decode_ft_header(bytes: &[u8]) -> Result<(Vec<f32>, bool), String> {
    if bytes.len() < FT_HEADER_BYTES {
        return Err("binary spectrum file shorter than 2048-byte 512-float header".to_string());
    }
    let order_le = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let order_be = f32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let big_endian = if (order_le - 2.345).abs() < 1e-2 {
        false
    } else if (order_be - 2.345).abs() < 1e-2 {
        true
    } else {
        return Err(format!(
            "unrecognized 512-float header byte-order magic (le={order_le}, be={order_be})"
        ));
    };

    let mut hdr = Vec::with_capacity(FT_HEADER_FLOATS);
    for chunk in bytes[..FT_HEADER_BYTES].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        hdr.push(if big_endian {
            f32::from_be_bytes(arr)
        } else {
            f32::from_le_bytes(arr)
        });
    }
    Ok((hdr, big_endian))
}

fn read_ft_label(raw_header: &[u8], slot: usize) -> String {
    let start = FT_LABEL_IDX[slot] * 4;
    let slice = &raw_header[start..start + 8];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(8);
    let s = String::from_utf8_lossy(&slice[..end]).trim().to_string();
    if s.is_empty() {
        format!("F{}", slot + 1)
    } else {
        s
    }
}

fn write_ft_label(raw_header: &mut [u8], slot: usize, label: &str) {
    let start = FT_LABEL_IDX[slot] * 4;
    let mut raw = [0u8; 8];
    let b = label.as_bytes();
    let len = b.len().min(8);
    raw[..len].copy_from_slice(&b[..len]);
    raw_header[start..start + 8].copy_from_slice(&raw);
}

fn decode_ft_axis(hdr: &[f32], raw_header: &[u8], slot: usize, size: usize) -> AxisPpm {
    let sw_hz = (hdr[FT_SW_IDX[slot]] as f64).max(1e-3);
    let orig_hz = hdr[FT_ORIG_IDX[slot]] as f64;
    let obs_mhz = (hdr[FT_OBS_IDX[slot]] as f64).max(1e-3);
    let n = size.max(1) as f64;
    let first_hz = orig_hz + sw_hz * ((n - 1.0) / n);
    AxisPpm {
        label: read_ft_label(raw_header, slot),
        size,
        first_ppm: first_hz / obs_mhz,
        step_ppm: -sw_hz / (obs_mhz * n),
        obs_mhz,
    }
}

fn encode_ft_axis_slot(hdr: &mut [f32; FT_HEADER_FLOATS], slot: usize, ax: &AxisPpm) {
    let n = ax.size.max(1) as f64;
    let sw_hz = ax.sw_ppm() * ax.obs_mhz;
    let last_ppm = ax.index_to_ppm(n - 1.0);
    let orig_hz = last_ppm * ax.obs_mhz;
    hdr[FT_SW_IDX[slot]] = sw_hz as f32;
    hdr[FT_ORIG_IDX[slot]] = orig_hz as f32;
    hdr[FT_OBS_IDX[slot]] = ax.obs_mhz as f32;
}

fn ft_slot_for_dim(hdr: &[f32], dim_idx: usize) -> usize {
    match hdr[24 + dim_idx].round() as isize {
        2 => 0,
        1 => 1,
        3 => 2,
        4 => 3,
        _ => dim_idx.min(3),
    }
}

/// Write a 1D `.ft1` spectrum file (512-float 2048-byte header + real `[+ optional imag]` `f32` vector).
pub fn write_ft1_1d(path: impl AsRef<Path>, spec: &Spectrum1D) -> Result<(), String> {
    let nx = spec.nx();
    if spec.data.len() != nx {
        return Err(format!(
            "Spectrum1D real data length ({}) != axis_x.size ({nx})",
            spec.data.len()
        ));
    }
    if let Some(im) = &spec.imag {
        if im.len() != nx {
            return Err(format!(
                "Spectrum1D imag data length ({}) != axis_x.size ({nx})",
                im.len()
            ));
        }
    }

    let is_complex = spec.imag.is_some();
    let mut hdr = [0.0f32; FT_HEADER_FLOATS];
    hdr[0] = 0.0;
    hdr[1] = f32::from_bits(0xeeee_eeee);
    hdr[2] = 2.345;
    hdr[9] = 1.0; // FDDIMCOUNT = 1
    hdr[24] = 2.0; // dim 0 -> slot 0 (F2)
    hdr[25] = 1.0;
    hdr[26] = 3.0;
    hdr[27] = 4.0;
    hdr[56] = if is_complex { 0.0 } else { 1.0 }; // FDQUADFLAG (0 = complex, 1 = real)
    hdr[106] = 1.0; // FDF2FTFLAG = frequency domain
    hdr[99] = nx as f32; // FDSIZE
    hdr[219] = 1.0; // FDSPECNUM
    hdr[15] = 1.0; // FDZSIZE
    hdr[442] = 1.0; // FDFILECOUNT
    encode_ft_axis_slot(&mut hdr, 0, &spec.axis_x);

    let n_floats = if is_complex { 2 * nx } else { nx };
    let mut bytes = Vec::with_capacity(FT_HEADER_BYTES + n_floats * 4);
    for &w in &hdr {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    write_ft_label(&mut bytes[..FT_HEADER_BYTES], 0, &spec.axis_x.label);

    for &v in &spec.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(im) = &spec.imag {
        for &v in im {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }

    fs::write(path.as_ref(), bytes)
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

/// Read a 1D `.ft1` spectrum file (supports both little-endian and big-endian, real and complex).
pub fn read_ft1_1d(path: impl AsRef<Path>) -> Result<Spectrum1D, String> {
    let p = path.as_ref();
    let bytes = fs::read(p).map_err(|e| format!("failed to read {}: {e}", p.display()))?;
    let (hdr, big_endian) = decode_ft_header(&bytes)?;

    let nx = (hdr[99].round() as usize).max(1);
    let slot_x = ft_slot_for_dim(&hdr, 0);
    let axis_x = decode_ft_axis(&hdr, &bytes[..FT_HEADER_BYTES], slot_x, nx);
    let is_complex = (hdr[56].round() as i32) == 0 && bytes.len() >= FT_HEADER_BYTES + 2 * nx * 4;

    let body = &bytes[FT_HEADER_BYTES..];
    let needed = if is_complex { 2 * nx } else { nx };
    if body.len() < needed * 4 {
        return Err(format!(
            "1D .ft1 body truncated: expected {needed} floats, got {} bytes",
            body.len()
        ));
    }

    let decode_slice = |raw: &[u8], count: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(count);
        for chunk in raw[..count * 4].chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            v.push(if big_endian {
                f32::from_be_bytes(arr)
            } else {
                f32::from_le_bytes(arr)
            });
        }
        v
    };

    let data = decode_slice(body, nx);
    let imag = if is_complex {
        Some(decode_slice(&body[nx * 4..], nx))
    } else {
        None
    };

    Ok(Spectrum1D {
        name: p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("spectrum.ft1")
            .to_string(),
        axis_x,
        data,
        imag,
    })
}

/// Expand a `printf`-style integer plane template (e.g. `"test%03d.ft3"`, `"plane_%04d.ft3"`,
/// or `"spec%d.ft3"`) for a 1-based plane index `one_based_index`.
pub fn format_plane_template(
    template: impl AsRef<Path>,
    one_based_index: usize,
) -> Result<PathBuf, String> {
    let s = template.as_ref().to_string_lossy();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'%' {
                i += 2;
                continue;
            }
            let start = i;
            i += 1;
            let zero_pad = if i < bytes.len() && bytes[i] == b'0' {
                i += 1;
                true
            } else {
                false
            };
            let w_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let width: usize = if i > w_start {
                s[w_start..i]
                    .parse()
                    .map_err(|e| format!("invalid template width in '{s}': {e}"))?
            } else {
                0
            };
            if i < bytes.len() && (bytes[i] == b'd' || bytes[i] == b'i') {
                let end = i + 1;
                let formatted_num = if width == 0 {
                    format!("{one_based_index}")
                } else if zero_pad {
                    format!("{one_based_index:0width$}")
                } else {
                    format!("{one_based_index:width$}")
                };
                let mut out = String::with_capacity(s.len() + formatted_num.len());
                out.push_str(&s[..start]);
                out.push_str(&formatted_num);
                out.push_str(&s[end..]);
                return Ok(PathBuf::from(out));
            }
        }
        i += 1;
    }
    Err(format!(
        "3D plane series template '{s}' is missing a '%d' or '%0Nd' integer placeholder"
    ))
}

/// Write a `Spectrum3D` as a template-numbered series of `nz` Z-plane files (`1..=nz`, e.g. `test%03d.ft3`),
/// each carrying a complete 512-float 3D header (`FDFILECOUNT = nz`) and `nx * ny` `f32` values.
pub fn write_ft3_plane_series(
    template_path: impl AsRef<Path>,
    spec: &Spectrum3D,
) -> Result<Vec<PathBuf>, String> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if spec.data.len() != nx * ny * nz {
        return Err(format!(
            "Spectrum3D data length ({}) != nx * ny * nz ({})",
            spec.data.len(),
            nx * ny * nz
        ));
    }

    let mut base_hdr = [0.0f32; FT_HEADER_FLOATS];
    base_hdr[0] = 0.0;
    base_hdr[1] = f32::from_bits(0xeeee_eeee);
    base_hdr[2] = 2.345;
    base_hdr[9] = 3.0; // FDDIMCOUNT = 3
    base_hdr[24] = 2.0; // dim 0 -> slot 0 (F2)
    base_hdr[25] = 1.0; // dim 1 -> slot 1 (F1)
    base_hdr[26] = 3.0; // dim 2 -> slot 2 (F3)
    base_hdr[27] = 4.0;
    base_hdr[56] = 1.0; // real
    base_hdr[55] = 1.0;
    base_hdr[50] = 1.0;
    base_hdr[106] = 1.0;
    base_hdr[222] = 1.0;
    base_hdr[99] = nx as f32; // FDSIZE
    base_hdr[219] = ny as f32; // FDSPECNUM
    base_hdr[15] = nz as f32; // FDZSIZE
    base_hdr[442] = nz as f32; // FDFILECOUNT = nz (multi-file 3D plane series)

    encode_ft_axis_slot(&mut base_hdr, 0, &spec.axis_x);
    encode_ft_axis_slot(&mut base_hdr, 1, &spec.axis_y);
    encode_ft_axis_slot(&mut base_hdr, 2, &spec.axis_z);

    let plane_floats = nx * ny;
    let mut written = Vec::with_capacity(nz);
    for iz in 0..nz {
        let p = format_plane_template(template_path.as_ref(), iz + 1)?;
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create directory {}: {e}", parent.display()))?;
            }
        }
        let mut hdr = base_hdr;
        hdr[342] = (iz + 1) as f32;

        let mut bytes = Vec::with_capacity(FT_HEADER_BYTES + plane_floats * 4);
        for &w in &hdr {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        write_ft_label(&mut bytes[..FT_HEADER_BYTES], 0, &spec.axis_x.label);
        write_ft_label(&mut bytes[..FT_HEADER_BYTES], 1, &spec.axis_y.label);
        write_ft_label(&mut bytes[..FT_HEADER_BYTES], 2, &spec.axis_z.label);

        let slice = &spec.data[iz * plane_floats..(iz + 1) * plane_floats];
        for &v in slice {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&p, bytes).map_err(|e| format!("failed to write {}: {e}", p.display()))?;
        written.push(p);
    }

    Ok(written)
}

/// Read a template-numbered 3D Z-plane series (`1..=nz`, e.g. `test%03d.ft3`) into a `Spectrum3D`.
pub fn read_ft3_plane_series(
    template_path: impl AsRef<Path>,
    exp: &str,
) -> Result<Spectrum3D, String> {
    let first_path = format_plane_template(template_path.as_ref(), 1)?;
    let first_bytes = fs::read(&first_path).map_err(|e| {
        format!(
            "failed to read first 3D plane {}: {e}",
            first_path.display()
        )
    })?;
    let (hdr0, _) = decode_ft_header(&first_bytes)?;

    let nx = (hdr0[99].round() as usize).max(1);
    let ny = (hdr0[219].round() as usize).max(1);
    let nz = (hdr0[15].round() as usize)
        .max(hdr0[442].round().max(1.0) as usize)
        .max(1);

    let slot_x = ft_slot_for_dim(&hdr0, 0);
    let slot_y = ft_slot_for_dim(&hdr0, 1);
    let slot_z = ft_slot_for_dim(&hdr0, 2);

    let axis_x = decode_ft_axis(&hdr0, &first_bytes[..FT_HEADER_BYTES], slot_x, nx);
    let axis_y = decode_ft_axis(&hdr0, &first_bytes[..FT_HEADER_BYTES], slot_y, ny);
    let axis_z = decode_ft_axis(&hdr0, &first_bytes[..FT_HEADER_BYTES], slot_z, nz);

    let plane_floats = nx * ny;
    let mut data = Vec::with_capacity(plane_floats * nz);

    for iz in 0..nz {
        let p = format_plane_template(template_path.as_ref(), iz + 1)?;
        let bytes = if iz == 0 {
            first_bytes.clone()
        } else {
            fs::read(&p).map_err(|e| format!("failed to read 3D plane {}: {e}", p.display()))?
        };
        let (_, big_endian) = decode_ft_header(&bytes)?;
        let body = &bytes[FT_HEADER_BYTES..];
        if body.len() < plane_floats * 4 {
            return Err(format!(
                "3D plane {} truncated: expected {plane_floats} floats, got {} bytes",
                p.display(),
                body.len()
            ));
        }
        for chunk in body[..plane_floats * 4].chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            data.push(if big_endian {
                f32::from_be_bytes(arr)
            } else {
                f32::from_le_bytes(arr)
            });
        }
    }

    Ok(Spectrum3D {
        name: template_path
            .as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("series.ft3")
            .to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    })
}
