//! Separable multi-scale 2D/3D convolutional neural peak picker, synthetic NMR lineshape
//! generator, analytical-backpropagation Adam trainer, and blind end-to-end benchmark.
//!
//! Provenance: Clean-room original implementation designed from public signal-processing and
//! deep-learning literature:
//! - Li, D.-W., Hansen, A. L., Yuan, C., Bruschweiler-Li, L. & Brüschweiler, R. (2021),
//!   *J. Biomol. NMR* 75, 79–89 — methodological reference for convolutional NMR peak
//!   classification and sub-voxel coordinate regression trained on synthetic pseudo-Voigt spectra.
//! - Klukowski, P., Augoff, M., Zięba, M., Drwal, M., Gonczarek, A. & Walczak, M. J. (2018),
//!   *Bioinformatics* 34(15), 2590–2597 — methodological reference for convolutional peak
//!   discrimination in the presence of indirect-dimension ($t_1$) ridge artefacts.
//! - Chollet, F. (2017), *CVPR*, 1251–1258 ("Xception: Deep Learning with Depthwise Separable
//!   Convolutions") & Yu, F. & Koltun, V. (2016), *ICLR* ("Multi-Scale Context Aggregation by
//!   Dilated Convolutions") — multi-scale 1D/3D separable dilated residual convolutions.
//! - Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017), *ICCV*, 2980–2988
//!   ("Focal Loss for Dense Object Detection") — class-imbalanced focal loss on sparse peak grids.
//! - Kingma, D. P. & Ba, J. (2015), *ICLR* ("Adam: A Method for Stochastic Optimization").

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::time::Instant;

use crate::evaluate::{compare_shifts, to_reference};
use crate::nmrstar::Reference;
use crate::peaks::{experiment_rules, Peak};
use crate::picker::{estimate_noise_mad, pick_hsqc_2d, pick_triple_3d, PickerConfig};
use crate::priors::Priors;
use crate::rng::Rng;
use crate::solver::{assign, AnnealConfig, ScoreConfig};
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use crate::spinsystems::{build_spin_systems, Tolerances};

// ============================================================================
// Architecture Constants & Flat Weight Offsets (shared with WebGPU conv3d.wgsl)
// ============================================================================

pub const HIDDEN_CH: usize = 16;

// 2D Separable CNN layout: 885 f32 parameters
pub const CONV2D_OUT_CH: usize = 5;
pub const OFF2_STEM_KX: usize = 0; // 16 * 5 = 80
pub const OFF2_STEM_KY: usize = 80; // 16 * 5 = 80
pub const OFF2_PW1_W: usize = 160; // 16 * 16 = 256
pub const OFF2_PW1_B: usize = 416; // 16
pub const OFF2_DW2_KX: usize = 432; // 16 * 3 = 48
pub const OFF2_DW2_KY: usize = 480; // 16 * 3 = 48
pub const OFF2_PW2_W: usize = 528; // 16 * 16 = 256
pub const OFF2_PW2_B: usize = 784; // 16
pub const OFF2_HEAD_W: usize = 800; // 5 * 16 = 80
pub const OFF2_HEAD_B: usize = 880; // 5
pub const CONV2D_NUM_WEIGHTS: usize = 885;

// 3D Separable CNN layout: 1,030 f32 parameters
pub const CONV3D_OUT_CH: usize = 6;
pub const OFF3_STEM_KX: usize = 0; // 16 * 5 = 80
pub const OFF3_STEM_KY: usize = 80; // 16 * 5 = 80
pub const OFF3_STEM_KZ: usize = 160; // 16 * 5 = 80
pub const OFF3_PW1_W: usize = 240; // 16 * 16 = 256
pub const OFF3_PW1_B: usize = 496; // 16
pub const OFF3_DW2_KX: usize = 512; // 16 * 3 = 48
pub const OFF3_DW2_KY: usize = 560; // 16 * 3 = 48
pub const OFF3_DW2_KZ: usize = 608; // 16 * 3 = 48
pub const OFF3_PW2_W: usize = 656; // 16 * 16 = 256
pub const OFF3_PW2_B: usize = 912; // 16
pub const OFF3_HEAD_W: usize = 928; // 6 * 16 = 96
pub const OFF3_HEAD_B: usize = 1024; // 6
pub const CONV3D_NUM_WEIGHTS: usize = 1030;

const NORM_SIGMA_DIV: f32 = 10.0;
const NORM_CLAMP: f32 = 8.0;
const BINARY_MAGIC_2D: &[u8; 8] = b"NMRCN2D\x01";
const BINARY_MAGIC_3D: &[u8; 8] = b"NMRCN3D\x01";

#[inline]
fn chan_dilation(c: usize) -> isize {
    if c < 8 {
        1
    } else {
        2
    }
}

#[inline]
fn clamp_idx(v: isize, dim: usize) -> usize {
    v.clamp(0, (dim as isize) - 1) as usize
}

#[inline]
pub fn safe_tanh_f32(x: f32) -> f32 {
    let xc = x.clamp(-9.5, 9.5);
    let e2 = (2.0 * xc).exp();
    (e2 - 1.0) / (e2 + 1.0)
}

/// Clamped tanh-GELU matching WebGPU `conv3d.wgsl` to `< 1e-6` across all `f32` inputs.
#[inline]
pub fn gelu_f32(x: f32) -> f32 {
    let inner = (0.797_884_6 * (x + 0.044_715 * x * x * x)).clamp(-9.0, 9.0);
    0.5 * x * (1.0 + safe_tanh_f32(inner))
}

#[inline]
fn gelu_grad_f32(x: f32) -> f32 {
    let raw = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    if raw >= 9.0 {
        return 1.0;
    }
    if raw <= -9.0 {
        return 0.0;
    }
    let t = safe_tanh_f32(raw);
    let d_inner = 0.797_884_6 * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
}

#[inline]
pub fn sigmoid_f32(x: f32) -> f32 {
    let xc = x.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-xc).exp())
}

#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    if x > 18.0 {
        x
    } else if x < -18.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn median_f32_local(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Wavelet/Haar first-difference MAD estimator (`1.4826 * median(|x[i+1] - x[i]| / sqrt(2))`),
/// invariant to smooth baseline roll, indirect `t1` ridges, and high peak density.
pub fn estimate_thermal_noise_haar(data: &[f32], nx: usize) -> f32 {
    if data.len() < 4 || nx < 2 {
        return estimate_noise_mad(data).max(1e-6);
    }
    let n_rows = data.len() / nx;
    let row_stride = (n_rows / 512).max(1);
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut diffs = Vec::with_capacity((n_rows / row_stride + 1) * (nx - 1));
    for r in (0..n_rows).step_by(row_stride) {
        let row = &data[r * nx..(r + 1) * nx];
        let x_stride = (nx / 128).max(1);
        let mut ix = 0usize;
        while ix + 1 < nx {
            diffs.push((row[ix + 1] - row[ix]).abs() * inv_sqrt2);
            ix += x_stride;
        }
    }
    if diffs.len() < 8 {
        return estimate_noise_mad(data).max(1e-6);
    }
    let mad0 = (1.4826 * median_f32_local(&mut diffs)).max(1e-6);
    let mut bg: Vec<f32> = diffs.into_iter().filter(|&d| d <= 2.5 * mad0).collect();
    if bg.len() < 8 {
        return mad0;
    }
    // Correction factor 1.014 accounts for 2.5-sigma half-normal tail truncation
    (1.4826 * 1.014 * median_f32_local(&mut bg)).max(1e-6)
}

/// Estimate `(median_baseline, thermal_noise_sigma)` deterministically and normalize intensities.
pub fn normalize_spectrum_data_with_nx(data: &[f32], nx: usize) -> (Vec<f32>, f32, f32) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let stride = (data.len() / 32_768).max(1);
    let mut sample: Vec<f32> = data.iter().step_by(stride).copied().collect();
    let med = median_f32_local(&mut sample);
    let noise = estimate_thermal_noise_haar(data, nx);
    let inv = 1.0 / (NORM_SIGMA_DIV * noise);
    let norm = data
        .iter()
        .map(|&v| ((v - med) * inv).clamp(-NORM_CLAMP, NORM_CLAMP))
        .collect();
    (norm, med, noise)
}

pub fn normalize_spectrum_data(data: &[f32]) -> (Vec<f32>, f32, f32) {
    let nx_guess = if data.len() >= 64 {
        64
    } else {
        data.len().max(2)
    };
    normalize_spectrum_data_with_nx(data, nx_guess)
}

// ============================================================================
// Synthetic 2D & 3D Spectral Generators (Voigt, +/- Phase, Shoulders, Ridges)
// ============================================================================

#[derive(Clone, Debug)]
pub struct GroundTruthPeak2D {
    pub x_idx: f64,
    pub y_idx: f64,
    pub h_ppm: f64,
    pub n_ppm: f64,
    pub sign: i32,
    pub amplitude: f32,
}

#[derive(Clone, Debug)]
pub struct SyntheticSpectrum2D {
    pub spectrum: Spectrum2D,
    pub ground_truth: Vec<GroundTruthPeak2D>,
    pub peaks: Vec<Peak>,
    pub noise_sigma: f32,
}

#[derive(Clone, Debug)]
pub struct SyntheticGenConfig2D {
    pub nx: usize,
    pub ny: usize,
    pub n_peaks: usize,
    pub allow_negative: bool,
    pub min_snr: f32,
    pub max_snr: f32,
    pub noise_sigma: f32,
    pub ridge_prob: f64,
    pub ridge_rel_amp: f32,
    pub baseline_roll_sigma: f32,
    pub shoulder_prob: f64,
    pub j_split_prob: f64,
    pub seed: u64,
}

impl Default for SyntheticGenConfig2D {
    fn default() -> Self {
        Self {
            nx: 64,
            ny: 64,
            n_peaks: 10,
            allow_negative: true,
            min_snr: 4.0,
            max_snr: 28.0,
            noise_sigma: 1.0,
            ridge_prob: 0.45,
            ridge_rel_amp: 0.14,
            baseline_roll_sigma: 2.2,
            shoulder_prob: 0.30,
            j_split_prob: 0.25,
            seed: 101,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroundTruthPeak3D {
    pub x_idx: f64,
    pub y_idx: f64,
    pub z_idx: f64,
    pub h_ppm: f64,
    pub n_ppm: f64,
    pub c_ppm: f64,
    pub sign: i32,
    pub amplitude: f32,
}

#[derive(Clone, Debug)]
pub struct SyntheticSpectrum3D {
    pub spectrum: Spectrum3D,
    pub ground_truth: Vec<GroundTruthPeak3D>,
    pub peaks: Vec<Peak>,
    pub noise_sigma: f32,
}

#[derive(Clone, Debug)]
pub struct SyntheticGenConfig3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub exp: String,
    pub n_peaks: usize,
    pub allow_negative: bool,
    pub min_snr: f32,
    pub max_snr: f32,
    pub noise_sigma: f32,
    pub ridge_prob: f64,
    pub ridge_rel_amp: f32,
    pub baseline_roll_sigma: f32,
    pub shoulder_prob: f64,
    pub j_split_prob: f64,
    pub seed: u64,
}

impl Default for SyntheticGenConfig3D {
    fn default() -> Self {
        Self {
            nx: 24,
            ny: 24,
            nz: 24,
            exp: "HNCACB".to_string(),
            n_peaks: 7,
            allow_negative: true,
            min_snr: 3.8,
            max_snr: 26.0,
            noise_sigma: 1.0,
            ridge_prob: 0.45,
            ridge_rel_amp: 0.14,
            baseline_roll_sigma: 2.0,
            shoulder_prob: 0.30,
            j_split_prob: 0.20,
            seed: 202,
        }
    }
}

#[inline]
fn pseudo_voigt_1d(delta_pts: f64, fwhm_pts: f64, gauss_mix: f64) -> f64 {
    let u = delta_pts / (0.5 * fwhm_pts.max(0.25));
    let g = (-std::f64::consts::LN_2 * u * u).exp();
    let l = 1.0 / (1.0 + u * u);
    gauss_mix * g + (1.0 - gauss_mix) * l
}

/// Generate a realistic synthetic 2D (`15N-HSQC`) spectrum/patch with pseudo-Voigt lineshapes,
/// positive/negative phase, unresolved J-splittings, overlapping shoulders, smooth baseline roll,
/// `t1` indirect ridges, and Gaussian thermal noise.
pub fn generate_synthetic_spectrum_2d(cfg: &SyntheticGenConfig2D) -> SyntheticSpectrum2D {
    let nx = cfg.nx.max(8);
    let ny = cfg.ny.max(8);
    // Use realistic NMR digital resolution (~0.0095 ppm/pt in 1H, ~0.13 ppm/pt in 15N)
    let sw_h = ((nx as f64) * 0.0095).min(4.8);
    let sw_n = ((ny as f64) * 0.135).min(34.0);
    let axis_x = AxisPpm::from_bounds("1H", nx, 8.2 + 0.5 * sw_h, 8.2 - 0.5 * sw_h, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 119.0 + 0.5 * sw_n, 119.0 - 0.5 * sw_n, 60.81);
    let mut rng = Rng::new(cfg.seed);
    let sigma = cfg.noise_sigma.max(1e-4);

    let mut data = vec![0.0f32; ny * nx];

    // 1. Smooth 2D baseline roll (sinusoidal + gentle quadratic drift)
    if cfg.baseline_roll_sigma > 0.0 {
        let amp_roll = (cfg.baseline_roll_sigma * sigma) as f64 * rng.uniform(0.35, 1.0);
        let fx = rng.uniform(0.5, 1.6) * TAU / (nx as f64);
        let fy = rng.uniform(0.4, 1.3) * TAU / (ny as f64);
        let px = rng.uniform(0.0, TAU);
        let py = rng.uniform(0.0, TAU);
        let slope_x = rng.uniform(-0.4, 0.4) * (sigma as f64);
        for iy in 0..ny {
            let cy = (fy * (iy as f64) + py).cos();
            for ix in 0..nx {
                let u = (ix as f64) / (nx as f64) - 0.5;
                let v = amp_roll * (fx * (ix as f64) + px).sin() * cy + slope_x * u;
                data[iy * nx + ix] += v as f32;
            }
        }
    }

    // 2. Place ground-truth pseudo-Voigt peaks (+ optional shoulder pairs & J-coupling splittings)
    let margin = 3.0_f64.min((nx.min(ny) as f64) * 0.2);
    let mut gt: Vec<GroundTruthPeak2D> = Vec::new();
    let mut col_strong_x: Vec<(f64, f64, f64)> = Vec::new(); // (cx, fwhm_x, amp)

    let add_voigt_2d = |data: &mut [f32],
                        cx: f64,
                        cy: f64,
                        amp_signed: f64,
                        fwhm_x: f64,
                        fwhm_y: f64,
                        eta: f64,
                        j_split_x: f64| {
        let rx = (4.0 * fwhm_x).ceil() as isize + 2;
        let ry = (4.0 * fwhm_y).ceil() as isize + 2;
        let x0 = ((cx.floor() as isize) - rx).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + rx).min(nx as isize - 1).max(0) as usize;
        let y0 = ((cy.floor() as isize) - ry).max(0) as usize;
        let y1 = ((cy.ceil() as isize) + ry).min(ny as isize - 1).max(0) as usize;

        // Normalize J-doublet so peak top equals `amp_signed`
        let norm_x = if j_split_x > 0.0 {
            1.0 / (2.0 * pseudo_voigt_1d(0.5 * j_split_x, fwhm_x, eta)).max(0.5)
        } else {
            1.0
        };
        for iy in y0..=y1 {
            let wy = pseudo_voigt_1d((iy as f64) - cy, fwhm_y, eta);
            for ix in x0..=x1 {
                let dx = (ix as f64) - cx;
                let wx = if j_split_x > 0.0 {
                    0.5 * norm_x
                        * (pseudo_voigt_1d(dx - 0.5 * j_split_x, fwhm_x, eta)
                            + pseudo_voigt_1d(dx + 0.5 * j_split_x, fwhm_x, eta))
                } else {
                    pseudo_voigt_1d(dx, fwhm_x, eta)
                };
                data[iy * nx + ix] += (amp_signed * wx * wy) as f32;
            }
        }
    };

    let mut attempts = 0usize;
    while gt.len() < cfg.n_peaks && attempts < cfg.n_peaks * 16 {
        attempts += 1;
        let cx = rng.uniform(margin, (nx as f64) - 1.0 - margin);
        let cy = rng.uniform(margin, (ny as f64) - 1.0 - margin);
        if gt.iter().any(|p| (p.x_idx - cx).hypot(p.y_idx - cy) < 2.2) {
            continue;
        }
        let u_snr = rng.f64();
        let snr = (cfg.min_snr as f64) * ((cfg.max_snr / cfg.min_snr.max(0.5)) as f64).powf(u_snr);
        let amp = (snr as f32) * sigma;
        let sign = if cfg.allow_negative && rng.f64() < 0.42 {
            -1
        } else {
            1
        };
        let fwhm_x = rng.uniform(1.8, 3.4);
        let fwhm_y = rng.uniform(1.7, 3.2);
        let eta = rng.uniform(0.30, 0.80);
        let j_split = if rng.f64() < cfg.j_split_prob {
            rng.uniform(0.25, 0.55) * fwhm_x
        } else {
            0.0
        };

        add_voigt_2d(
            &mut data,
            cx,
            cy,
            (amp as f64) * (sign as f64),
            fwhm_x,
            fwhm_y,
            eta,
            j_split,
        );
        gt.push(GroundTruthPeak2D {
            x_idx: cx,
            y_idx: cy,
            h_ppm: axis_x.index_to_ppm(cx),
            n_ppm: axis_y.index_to_ppm(cy),
            sign,
            amplitude: amp,
        });
        if snr > 12.0 {
            col_strong_x.push((cx, fwhm_x, amp as f64));
        }

        // Inject closely spaced overlapping shoulder companion peak
        if gt.len() < cfg.n_peaks && rng.f64() < cfg.shoulder_prob {
            let angle = rng.uniform(0.0, TAU);
            let dist = rng.uniform(2.3, 3.6);
            let sx = cx + dist * angle.cos();
            let sy = cy + dist * angle.sin();
            if sx >= margin
                && sx <= (nx as f64) - 1.0 - margin
                && sy >= margin
                && sy <= (ny as f64) - 1.0 - margin
                && !gt.iter().any(|p| (p.x_idx - sx).hypot(p.y_idx - sy) < 2.1)
            {
                let s_amp = (amp * (rng.uniform(0.42, 0.78) as f32)).max(cfg.min_snr * sigma);
                let s_sign = if cfg.allow_negative && rng.f64() < 0.35 {
                    -sign
                } else {
                    sign
                };
                add_voigt_2d(
                    &mut data,
                    sx,
                    sy,
                    (s_amp as f64) * (s_sign as f64),
                    fwhm_x * 0.95,
                    fwhm_y * 0.95,
                    eta,
                    0.0,
                );
                gt.push(GroundTruthPeak2D {
                    x_idx: sx,
                    y_idx: sy,
                    h_ppm: axis_x.index_to_ppm(sx),
                    n_ppm: axis_y.index_to_ppm(sy),
                    sign: s_sign,
                    amplitude: s_amp,
                });
            }
        }
    }

    // 3. Indirect-dimension (t1 / 15N) ridges along columns of intense peaks (or solvent column)
    if cfg.ridge_rel_amp > 0.0 && rng.f64() < cfg.ridge_prob {
        if col_strong_x.is_empty() {
            for _ in 0..3 {
                col_strong_x.push((
                    rng.uniform(margin, (nx as f64) - 1.0 - margin),
                    rng.uniform(1.8, 2.8),
                    rng.uniform(14.0, 24.0) * (sigma as f64),
                ));
            }
        }
        col_strong_x.sort_by(|a, b| b.2.total_cmp(&a.2));
        for &(rx_c, fwhm_x, base_amp) in col_strong_x.iter().take(3) {
            let ridge_amp = base_amp * (cfg.ridge_rel_amp as f64) * rng.uniform(0.6, 1.25);
            let dc_offset = ridge_amp * rng.uniform(-0.45, 0.65);
            let x0 = ((rx_c.floor() as isize) - 4).max(0) as usize;
            let x1 = ((rx_c.ceil() as isize) + 4).min(nx as isize - 1).max(0) as usize;
            for iy in 0..ny {
                let streak = dc_offset + rng.normal(0.0, ridge_amp);
                for ix in x0..=x1 {
                    let wx = pseudo_voigt_1d((ix as f64) - rx_c, fwhm_x, 0.6);
                    data[iy * nx + ix] += (streak * wx) as f32;
                }
            }
        }
    }

    // 4. Additive Gaussian thermal noise
    for v in &mut data {
        *v += rng.normal(0.0, sigma as f64) as f32;
    }

    let peaks = gt
        .iter()
        .map(|g| Peak {
            exp: "HSQC".to_string(),
            h: g.h_ppm,
            n: g.n_ppm,
            c: None,
            sign: g.sign,
            height: Some(g.amplitude as f64),
            volume: Some((g.amplitude as f64) * 4.2),
            h2: None,
        })
        .collect();

    SyntheticSpectrum2D {
        spectrum: Spectrum2D {
            name: "Synthetic 2D HSQC".to_string(),
            axis_x,
            axis_y,
            data,
        },
        ground_truth: gt,
        peaks,
        noise_sigma: sigma,
    }
}

/// Generate a realistic synthetic 3D triple-resonance (`HNCA`, `HNCACB`, `HNCO`) spectrum/patch
/// with pseudo-Voigt lineshapes, positive/negative phase (`+CA`/`-CB`), overlapping shoulders,
/// baseline roll, indirect-dimension ridges, and Gaussian thermal noise.
pub fn generate_synthetic_spectrum_3d(cfg: &SyntheticGenConfig3D) -> SyntheticSpectrum3D {
    let nx = cfg.nx.max(8);
    let ny = cfg.ny.max(8);
    let nz = cfg.nz.max(8);
    // Realistic NMR digital resolution per voxel even on cropped subvolume patches
    let sw_h = ((nx as f64) * 0.014).min(4.8);
    let sw_n = ((ny as f64) * 0.22).min(34.0);
    let (c_mid, c_step, c_max_sw) = if cfg.exp == "HNCO" || cfg.exp == "HNCACO" {
        (175.5, 0.16, 15.0)
    } else if cfg.exp == "HNCACB" || cfg.exp == "CBCACONH" {
        (44.5, 0.36, 62.0)
    } else {
        (55.0, 0.24, 30.0)
    };
    let sw_c = ((nz as f64) * c_step).min(c_max_sw);
    let axis_x = AxisPpm::from_bounds("1H", nx, 8.2 + 0.5 * sw_h, 8.2 - 0.5 * sw_h, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 119.0 + 0.5 * sw_n, 119.0 - 0.5 * sw_n, 60.81);
    let axis_z = AxisPpm::from_bounds("13C", nz, c_mid + 0.5 * sw_c, c_mid - 0.5 * sw_c, 150.90);

    let mut rng = Rng::new(cfg.seed);
    let sigma = cfg.noise_sigma.max(1e-4);
    let mut data = vec![0.0f32; nz * ny * nx];

    // 1. Smooth 3D baseline roll
    if cfg.baseline_roll_sigma > 0.0 {
        let amp_roll = (cfg.baseline_roll_sigma * sigma) as f64 * rng.uniform(0.35, 1.0);
        let fx = rng.uniform(0.5, 1.5) * TAU / (nx as f64);
        let fy = rng.uniform(0.4, 1.2) * TAU / (ny as f64);
        let fz = rng.uniform(0.4, 1.2) * TAU / (nz as f64);
        let px = rng.uniform(0.0, TAU);
        let py = rng.uniform(0.0, TAU);
        let pz = rng.uniform(0.0, TAU);
        for iz in 0..nz {
            let cz = (fz * (iz as f64) + pz).cos();
            for iy in 0..ny {
                let cyz = cz * (fy * (iy as f64) + py).cos();
                let base = (iz * ny + iy) * nx;
                for ix in 0..nx {
                    data[base + ix] += (amp_roll * (fx * (ix as f64) + px).sin() * cyz) as f32;
                }
            }
        }
    }

    let margin = 2.5_f64.min((nx.min(ny).min(nz) as f64) * 0.2);
    let mut gt: Vec<GroundTruthPeak3D> = Vec::new();
    let mut ridge_anchors: Vec<(f64, f64, f64, f64, f64)> = Vec::new(); // (cx, cy, fx, fy, amp)

    let add_voigt_3d = |data: &mut [f32],
                        cx: f64,
                        cy: f64,
                        cz: f64,
                        amp_signed: f64,
                        fx: f64,
                        fy: f64,
                        fz: f64,
                        eta: f64,
                        j_split_z: f64| {
        let rx = (3.5 * fx).ceil() as isize + 2;
        let ry = (3.5 * fy).ceil() as isize + 2;
        let rz = (3.5 * fz).ceil() as isize + 2;
        let x0 = ((cx.floor() as isize) - rx).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + rx).min(nx as isize - 1).max(0) as usize;
        let y0 = ((cy.floor() as isize) - ry).max(0) as usize;
        let y1 = ((cy.ceil() as isize) + ry).min(ny as isize - 1).max(0) as usize;
        let z0 = ((cz.floor() as isize) - rz).max(0) as usize;
        let z1 = ((cz.ceil() as isize) + rz).min(nz as isize - 1).max(0) as usize;

        let norm_z = if j_split_z > 0.0 {
            1.0 / (2.0 * pseudo_voigt_1d(0.5 * j_split_z, fz, eta)).max(0.5)
        } else {
            1.0
        };
        for iz in z0..=z1 {
            let dz = (iz as f64) - cz;
            let wz = if j_split_z > 0.0 {
                0.5 * norm_z
                    * (pseudo_voigt_1d(dz - 0.5 * j_split_z, fz, eta)
                        + pseudo_voigt_1d(dz + 0.5 * j_split_z, fz, eta))
            } else {
                pseudo_voigt_1d(dz, fz, eta)
            };
            for iy in y0..=y1 {
                let wyz = wz * pseudo_voigt_1d((iy as f64) - cy, fy, eta);
                let base = (iz * ny + iy) * nx;
                for ix in x0..=x1 {
                    let wx = pseudo_voigt_1d((ix as f64) - cx, fx, eta);
                    data[base + ix] += (amp_signed * wx * wyz) as f32;
                }
            }
        }
    };

    let mut attempts = 0usize;
    while gt.len() < cfg.n_peaks && attempts < cfg.n_peaks * 16 {
        attempts += 1;
        let cx = rng.uniform(margin, (nx as f64) - 1.0 - margin);
        let cy = rng.uniform(margin, (ny as f64) - 1.0 - margin);
        let cz = rng.uniform(margin, (nz as f64) - 1.0 - margin);
        if gt.iter().any(|p| {
            let d2 = (p.x_idx - cx).powi(2) + (p.y_idx - cy).powi(2) + (p.z_idx - cz).powi(2);
            d2 < 5.0
        }) {
            continue;
        }
        let u_snr = rng.f64();
        let snr = (cfg.min_snr as f64) * ((cfg.max_snr / cfg.min_snr.max(0.5)) as f64).powf(u_snr);
        let amp = (snr as f32) * sigma;
        let sign = if (cfg.allow_negative || cfg.exp == "HNCACB") && rng.f64() < 0.46 {
            -1
        } else {
            1
        };
        let fx = rng.uniform(1.7, 3.1);
        let fy = rng.uniform(1.6, 3.0);
        let fz = rng.uniform(1.7, 3.1);
        let eta = rng.uniform(0.35, 0.80);
        let j_split = if rng.f64() < cfg.j_split_prob {
            rng.uniform(0.22, 0.50) * fz
        } else {
            0.0
        };

        add_voigt_3d(
            &mut data,
            cx,
            cy,
            cz,
            (amp as f64) * (sign as f64),
            fx,
            fy,
            fz,
            eta,
            j_split,
        );
        gt.push(GroundTruthPeak3D {
            x_idx: cx,
            y_idx: cy,
            z_idx: cz,
            h_ppm: axis_x.index_to_ppm(cx),
            n_ppm: axis_y.index_to_ppm(cy),
            c_ppm: axis_z.index_to_ppm(cz),
            sign,
            amplitude: amp,
        });
        if snr > 11.0 {
            ridge_anchors.push((cx, cy, fx, fy, amp as f64));
        }

        // Overlapping shoulder peak along carbon (strip) axis or nearby 3D coordinate
        if gt.len() < cfg.n_peaks && rng.f64() < cfg.shoulder_prob {
            let dir_sign = if rng.f64() < 0.5 { -1.0 } else { 1.0 };
            let sz = cz + dir_sign * rng.uniform(2.3, 3.5);
            let sx = (cx + rng.uniform(-0.4, 0.4)).clamp(margin, (nx as f64) - 1.0 - margin);
            let sy = (cy + rng.uniform(-0.4, 0.4)).clamp(margin, (ny as f64) - 1.0 - margin);
            if sz >= margin
                && sz <= (nz as f64) - 1.0 - margin
                && !gt.iter().any(|p| {
                    (p.x_idx - sx).powi(2) + (p.y_idx - sy).powi(2) + (p.z_idx - sz).powi(2) < 4.5
                })
            {
                let s_amp = (amp * (rng.uniform(0.40, 0.75) as f32)).max(cfg.min_snr * sigma);
                let s_sign = if (cfg.allow_negative || cfg.exp == "HNCACB") && rng.f64() < 0.40 {
                    -sign
                } else {
                    sign
                };
                add_voigt_3d(
                    &mut data,
                    sx,
                    sy,
                    sz,
                    (s_amp as f64) * (s_sign as f64),
                    fx,
                    fy,
                    fz * 0.95,
                    eta,
                    0.0,
                );
                gt.push(GroundTruthPeak3D {
                    x_idx: sx,
                    y_idx: sy,
                    z_idx: sz,
                    h_ppm: axis_x.index_to_ppm(sx),
                    n_ppm: axis_y.index_to_ppm(sy),
                    c_ppm: axis_z.index_to_ppm(sz),
                    sign: s_sign,
                    amplitude: s_amp,
                });
            }
        }
    }

    // 3. Indirect 13C / 15N ridge streaks at fixed (1H, 15N) amide coordinates
    if cfg.ridge_rel_amp > 0.0 && rng.f64() < cfg.ridge_prob {
        if ridge_anchors.is_empty() {
            for _ in 0..2 {
                ridge_anchors.push((
                    rng.uniform(margin, (nx as f64) - 1.0 - margin),
                    rng.uniform(margin, (ny as f64) - 1.0 - margin),
                    rng.uniform(1.8, 2.5),
                    rng.uniform(1.7, 2.4),
                    rng.uniform(14.0, 24.0) * (sigma as f64),
                ));
            }
        }
        ridge_anchors.sort_by(|a, b| b.4.total_cmp(&a.4));
        for &(rx_c, ry_c, fx, fy, base_amp) in ridge_anchors.iter().take(2) {
            let ridge_amp = base_amp * (cfg.ridge_rel_amp as f64) * rng.uniform(0.65, 1.25);
            let dc = ridge_amp * rng.uniform(-0.5, 0.65);
            let x0 = ((rx_c.floor() as isize) - 3).max(0) as usize;
            let x1 = ((rx_c.ceil() as isize) + 3).min(nx as isize - 1).max(0) as usize;
            let y0 = ((ry_c.floor() as isize) - 3).max(0) as usize;
            let y1 = ((ry_c.ceil() as isize) + 3).min(ny as isize - 1).max(0) as usize;
            for iz in 0..nz {
                let streak = dc + rng.normal(0.0, ridge_amp);
                for iy in y0..=y1 {
                    let wy = pseudo_voigt_1d((iy as f64) - ry_c, fy, 0.6);
                    let base = (iz * ny + iy) * nx;
                    for ix in x0..=x1 {
                        let wx = pseudo_voigt_1d((ix as f64) - rx_c, fx, 0.6);
                        data[base + ix] += (streak * wx * wy) as f32;
                    }
                }
            }
        }
    }

    // 4. Additive Gaussian thermal noise
    for v in &mut data {
        *v += rng.normal(0.0, sigma as f64) as f32;
    }

    let peaks = gt
        .iter()
        .map(|g| Peak {
            exp: cfg.exp.clone(),
            h: g.h_ppm,
            n: g.n_ppm,
            c: Some(g.c_ppm),
            sign: g.sign,
            height: Some(g.amplitude as f64),
            volume: Some((g.amplitude as f64) * 5.8),
            h2: None,
        })
        .collect();

    SyntheticSpectrum3D {
        spectrum: Spectrum3D {
            name: format!("Synthetic 3D {}", cfg.exp),
            exp: cfg.exp.clone(),
            axis_x,
            axis_y,
            axis_z,
            data,
        },
        ground_truth: gt,
        peaks,
        noise_sigma: sigma,
    }
}

// ============================================================================
// Separable 2D & 3D Convolutional Networks (Forward, Analytical Grad, Serde)
// ============================================================================

#[derive(Clone, Debug)]
pub struct SeparableConvNet2D {
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SeparableConvNet3D {
    pub weights: Vec<f32>,
}

fn weights_to_json_string(weights: &[f32]) -> String {
    let mut s = String::with_capacity(16 + weights.len() * 12);
    s.push_str("{\"weights\":[");
    for (i, &w) in weights.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{w:.8}"));
    }
    s.push_str("]}");
    s
}

fn weights_from_json_string(text: &str, expected_len: usize) -> Result<Vec<f32>, String> {
    let start = text
        .find('[')
        .ok_or_else(|| "JSON parse error: missing '['".to_string())?;
    let end = text[start + 1..]
        .find(']')
        .map(|k| start + 1 + k)
        .ok_or_else(|| "JSON parse error: missing ']'".to_string())?;
    let body = text[start + 1..end].trim();
    if body.is_empty() {
        return Err("JSON parse error: empty weights array".to_string());
    }
    let mut out = Vec::with_capacity(expected_len);
    for tok in body.split(',') {
        let v: f32 = tok
            .trim()
            .parse()
            .map_err(|e| format!("JSON parse float error: {e}"))?;
        out.push(v);
    }
    if out.len() != expected_len {
        return Err(format!(
            "invalid weight length in JSON: expected {expected_len}, got {}",
            out.len()
        ));
    }
    Ok(out)
}

#[allow(clippy::needless_range_loop)]
impl SeparableConvNet2D {
    /// Structured signal-processing + Kaiming initialization (matched Voigt, 1st/2nd derivatives,
    /// multi-scale dilated background estimators, and paired +/- gating).
    pub fn init_seeded(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut w = vec![0.0f32; CONV2D_NUM_WEIGHTS];

        let sym_narrow = [0.12f32, 0.35, 0.56, 0.35, 0.12];
        let dog = [-0.30f32, 0.12, 0.36, 0.12, -0.30];
        let flank = [-0.25f32, 0.50, 0.0, 0.50, -0.25];
        let deriv1 = [-0.22f32, -0.48, 0.0, 0.48, 0.22];
        let dc_flat = [0.20f32, 0.20, 0.20, 0.20, 0.20];

        let templates: [([f32; 5], [f32; 5]); 16] = [
            (dog, dog),
            (flank, flank),
            (dog, flank),
            (flank, dog),
            (deriv1, dog),
            (dog, deriv1),
            (sym_narrow, dog),
            (dog, sym_narrow),
            // Dilated d=2 wide context & ridge discriminators
            (dc_flat, dc_flat),
            (dog, dc_flat), // ridge along Y detector
            (dc_flat, dog), // ridge along X detector
            (flank, dc_flat),
            (deriv1, flank),
            (flank, deriv1),
            (dog, dog),
            (flank, flank),
        ];

        for (c, (tx, ty)) in templates.iter().enumerate() {
            for m in 0..5 {
                w[OFF2_STEM_KX + c * 5 + m] = tx[m] + rng.normal(0.0, 0.04) as f32;
                w[OFF2_STEM_KY + c * 5 + m] = ty[m] + rng.normal(0.0, 0.04) as f32;
            }
        }
        for i in 0..(HIDDEN_CH * HIDDEN_CH) {
            w[OFF2_PW1_W + i] = rng.normal(0.0, 0.18) as f32;
            w[OFF2_PW2_W + i] = rng.normal(0.0, 0.14) as f32;
        }
        // Paired +/- feature routing in PW1
        for c in 0..8 {
            w[OFF2_PW1_W + (2 * c) * HIDDEN_CH + c] += 0.65;
            w[OFF2_PW1_W + (2 * c + 1) * HIDDEN_CH + c] -= 0.65;
        }
        for c in 0..HIDDEN_CH {
            let k3 = [
                -0.12 + rng.normal(0.0, 0.04) as f32,
                0.70 + rng.normal(0.0, 0.05) as f32,
                -0.12 + rng.normal(0.0, 0.04) as f32,
            ];
            for m in 0..3 {
                w[OFF2_DW2_KX + c * 3 + m] = k3[m];
                w[OFF2_DW2_KY + c * 3 + m] = k3[m];
            }
        }
        for i in 0..(CONV2D_OUT_CH * HIDDEN_CH) {
            w[OFF2_HEAD_W + i] = rng.normal(0.0, 0.15) as f32;
        }
        // Focal-loss prior bias p0 ~ 0.09 (Lin et al. 2017)
        w[OFF2_HEAD_B] = -2.3;
        w[OFF2_HEAD_B + 1] = -2.3;
        w[OFF2_HEAD_B + 4] = 0.2;
        Self { weights: w }
    }

    pub fn from_flat_weights(weights: &[f32]) -> Result<Self, String> {
        if weights.len() != CONV2D_NUM_WEIGHTS {
            return Err(format!(
                "expected {CONV2D_NUM_WEIGHTS} weights for SeparableConvNet2D, got {}",
                weights.len()
            ));
        }
        Ok(Self {
            weights: weights.to_vec(),
        })
    }

    pub fn to_flat_weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn to_json(&self) -> Result<String, String> {
        Ok(weights_to_json_string(&self.weights))
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let weights = weights_from_json_string(text, CONV2D_NUM_WEIGHTS)?;
        Ok(Self { weights })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + CONV2D_NUM_WEIGHTS * 4);
        out.extend_from_slice(BINARY_MAGIC_2D);
        for &v in &self.weights {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 8 + CONV2D_NUM_WEIGHTS * 4 || &bytes[..8] != BINARY_MAGIC_2D {
            return Err("invalid binary header or length for SeparableConvNet2D".to_string());
        }
        let mut weights = Vec::with_capacity(CONV2D_NUM_WEIGHTS);
        for chunk in bytes[8..].chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { weights })
    }

    /// Multi-threaded 2D forward pass returning `[ny * nx * 5]` (`prob_pos, prob_neg, dx, dy, amp_norm`).
    pub fn forward_2d(&self, input: &[f32], nx: usize, ny: usize) -> Vec<f32> {
        let w = &self.weights;
        let n_pix = nx * ny;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min((ny / 16).max(1));

        // Pass 1: Stem X -> sx [ny * nx * 16]
        let mut sx = vec![0.0f32; n_pix * HIDDEN_CH];
        let rows_per_t = ny.div_ceil(n_threads);
        std::thread::scope(|s| {
            for (ti, chunk) in sx.chunks_mut(rows_per_t * nx * HIDDEN_CH).enumerate() {
                let y_start = ti * rows_per_t;
                let local_rows = chunk.len() / (nx * HIDDEN_CH);
                s.spawn(move || {
                    for ly in 0..local_rows {
                        let iy = y_start + ly;
                        let in_row = &input[iy * nx..(iy + 1) * nx];
                        for ix in 0..nx {
                            let dst = &mut chunk
                                [(ly * nx + ix) * HIDDEN_CH..(ly * nx + ix + 1) * HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF2_STEM_KX + c * 5..OFF2_STEM_KX + (c + 1) * 5];
                                let mut acc = 0.0f32;
                                for m in 0..5 {
                                    let gx = clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                                    acc += kw[m] * in_row[gx];
                                }
                                dst[c] = acc;
                            }
                        }
                    }
                });
            }
        });

        // Pass 2: Stem Y + PW1 + GELU -> a1 [ny * nx * 16]
        let mut a1 = vec![0.0f32; n_pix * HIDDEN_CH];
        let sx_ref = &sx;
        std::thread::scope(|s| {
            for (ti, chunk) in a1.chunks_mut(rows_per_t * nx * HIDDEN_CH).enumerate() {
                let y_start = ti * rows_per_t;
                let local_rows = chunk.len() / (nx * HIDDEN_CH);
                s.spawn(move || {
                    for ly in 0..local_rows {
                        let iy = y_start + ly;
                        for ix in 0..nx {
                            let mut sy = [0.0f32; HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF2_STEM_KY + c * 5..OFF2_STEM_KY + (c + 1) * 5];
                                let mut acc = 0.0f32;
                                for m in 0..5 {
                                    let gy = clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                                    acc += kw[m] * sx_ref[(gy * nx + ix) * HIDDEN_CH + c];
                                }
                                sy[c] = acc;
                            }
                            let dst = &mut chunk
                                [(ly * nx + ix) * HIDDEN_CH..(ly * nx + ix + 1) * HIDDEN_CH];
                            for co in 0..HIDDEN_CH {
                                let pw = &w[OFF2_PW1_W + co * HIDDEN_CH
                                    ..OFF2_PW1_W + (co + 1) * HIDDEN_CH];
                                let mut sum = w[OFF2_PW1_B + co];
                                for ci in 0..HIDDEN_CH {
                                    sum += pw[ci] * sy[ci];
                                }
                                dst[co] = gelu_f32(sum);
                            }
                        }
                    }
                });
            }
        });

        // Pass 3: DW2 X -> rx [ny * nx * 16]
        let mut rx = vec![0.0f32; n_pix * HIDDEN_CH];
        let a1_ref = &a1;
        std::thread::scope(|s| {
            for (ti, chunk) in rx.chunks_mut(rows_per_t * nx * HIDDEN_CH).enumerate() {
                let y_start = ti * rows_per_t;
                let local_rows = chunk.len() / (nx * HIDDEN_CH);
                s.spawn(move || {
                    for ly in 0..local_rows {
                        let iy = y_start + ly;
                        for ix in 0..nx {
                            let dst = &mut chunk
                                [(ly * nx + ix) * HIDDEN_CH..(ly * nx + ix + 1) * HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF2_DW2_KX + c * 3..OFF2_DW2_KX + (c + 1) * 3];
                                let mut acc = 0.0f32;
                                for m in 0..3 {
                                    let gx = clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                                    acc += kw[m] * a1_ref[(iy * nx + gx) * HIDDEN_CH + c];
                                }
                                dst[c] = acc;
                            }
                        }
                    }
                });
            }
        });

        // Pass 4: DW2 Y + PW2 + Skip + GELU + Head -> out [ny * nx * 5]
        let mut out = vec![0.0f32; n_pix * CONV2D_OUT_CH];
        let rx_ref = &rx;
        std::thread::scope(|s| {
            for (ti, chunk) in out.chunks_mut(rows_per_t * nx * CONV2D_OUT_CH).enumerate() {
                let y_start = ti * rows_per_t;
                let local_rows = chunk.len() / (nx * CONV2D_OUT_CH);
                s.spawn(move || {
                    for ly in 0..local_rows {
                        let iy = y_start + ly;
                        for ix in 0..nx {
                            let mut ry = [0.0f32; HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF2_DW2_KY + c * 3..OFF2_DW2_KY + (c + 1) * 3];
                                let mut acc = 0.0f32;
                                for m in 0..3 {
                                    let gy = clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                                    acc += kw[m] * rx_ref[(gy * nx + ix) * HIDDEN_CH + c];
                                }
                                ry[c] = acc;
                            }
                            let skip =
                                &a1_ref[(iy * nx + ix) * HIDDEN_CH..(iy * nx + ix + 1) * HIDDEN_CH];
                            let mut a2 = [0.0f32; HIDDEN_CH];
                            for co in 0..HIDDEN_CH {
                                let pw = &w[OFF2_PW2_W + co * HIDDEN_CH
                                    ..OFF2_PW2_W + (co + 1) * HIDDEN_CH];
                                let mut sum = skip[co] + w[OFF2_PW2_B + co];
                                for ci in 0..HIDDEN_CH {
                                    sum += pw[ci] * ry[ci];
                                }
                                a2[co] = gelu_f32(sum);
                            }
                            let mut raw_h = [0.0f32; CONV2D_OUT_CH];
                            for k in 0..CONV2D_OUT_CH {
                                let hw = &w[OFF2_HEAD_W + k * HIDDEN_CH
                                    ..OFF2_HEAD_W + (k + 1) * HIDDEN_CH];
                                let mut sum = w[OFF2_HEAD_B + k];
                                for ci in 0..HIDDEN_CH {
                                    sum += hw[ci] * a2[ci];
                                }
                                raw_h[k] = sum;
                            }
                            let dst = &mut chunk[(ly * nx + ix) * CONV2D_OUT_CH
                                ..(ly * nx + ix + 1) * CONV2D_OUT_CH];
                            dst[0] = sigmoid_f32(raw_h[0]);
                            dst[1] = sigmoid_f32(raw_h[1]);
                            dst[2] = 0.5 * safe_tanh_f32(raw_h[2].clamp(-12.0, 12.0));
                            dst[3] = 0.5 * safe_tanh_f32(raw_h[3].clamp(-12.0, 12.0));
                            dst[4] = softplus_f32(raw_h[4]);
                        }
                    }
                });
            }
        });
        out
    }

    /// Exact analytical forward + backward pass wrt all `885` weights given upstream `d_loss / d_raw_head`.
    pub fn forward_backward_raw_2d(
        &self,
        input: &[f32],
        nx: usize,
        ny: usize,
        grad_from_raw_head: impl Fn(usize, &[f32; CONV2D_OUT_CH], &mut [f32; CONV2D_OUT_CH]) -> f64,
    ) -> (f64, Vec<f32>) {
        let w = &self.weights;
        let n_pix = nx * ny;
        let mut sx = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut sy = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut pre1 = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut a1 = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut rx = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut ry = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut pre2 = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut a2 = vec![0.0f32; n_pix * HIDDEN_CH];

        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..5 {
                        let gx = clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                        acc += w[OFF2_STEM_KX + c * 5 + m] * input[iy * nx + gx];
                    }
                    sx[p * HIDDEN_CH + c] = acc;
                }
            }
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..5 {
                        let gy = clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                        acc += w[OFF2_STEM_KY + c * 5 + m] * sx[(gy * nx + ix) * HIDDEN_CH + c];
                    }
                    sy[p * HIDDEN_CH + c] = acc;
                }
                for co in 0..HIDDEN_CH {
                    let mut sum = w[OFF2_PW1_B + co];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF2_PW1_W + co * HIDDEN_CH + ci] * sy[p * HIDDEN_CH + ci];
                    }
                    pre1[p * HIDDEN_CH + co] = sum;
                    a1[p * HIDDEN_CH + co] = gelu_f32(sum);
                }
            }
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..3 {
                        let gx = clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                        acc += w[OFF2_DW2_KX + c * 3 + m] * a1[(iy * nx + gx) * HIDDEN_CH + c];
                    }
                    rx[p * HIDDEN_CH + c] = acc;
                }
            }
        }

        let mut total_loss = 0.0f64;
        let mut gw = vec![0.0f32; CONV2D_NUM_WEIGHTS];
        let mut g_ry = vec![0.0f32; n_pix * HIDDEN_CH];
        let mut g_a1 = vec![0.0f32; n_pix * HIDDEN_CH];

        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..3 {
                        let gy = clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                        acc += w[OFF2_DW2_KY + c * 3 + m] * rx[(gy * nx + ix) * HIDDEN_CH + c];
                    }
                    ry[p * HIDDEN_CH + c] = acc;
                }
                for co in 0..HIDDEN_CH {
                    let mut sum = a1[p * HIDDEN_CH + co] + w[OFF2_PW2_B + co];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF2_PW2_W + co * HIDDEN_CH + ci] * ry[p * HIDDEN_CH + ci];
                    }
                    pre2[p * HIDDEN_CH + co] = sum;
                    a2[p * HIDDEN_CH + co] = gelu_f32(sum);
                }
                let mut raw_h = [0.0f32; CONV2D_OUT_CH];
                for k in 0..CONV2D_OUT_CH {
                    let mut sum = w[OFF2_HEAD_B + k];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF2_HEAD_W + k * HIDDEN_CH + ci] * a2[p * HIDDEN_CH + ci];
                    }
                    raw_h[k] = sum;
                }
                let mut g_raw = [0.0f32; CONV2D_OUT_CH];
                total_loss += grad_from_raw_head(p, &raw_h, &mut g_raw);

                let mut g_a2_p = [0.0f32; HIDDEN_CH];
                for k in 0..CONV2D_OUT_CH {
                    let gr = g_raw[k];
                    if gr == 0.0 {
                        continue;
                    }
                    gw[OFF2_HEAD_B + k] += gr;
                    for ci in 0..HIDDEN_CH {
                        gw[OFF2_HEAD_W + k * HIDDEN_CH + ci] += gr * a2[p * HIDDEN_CH + ci];
                        g_a2_p[ci] += gr * w[OFF2_HEAD_W + k * HIDDEN_CH + ci];
                    }
                }
                for co in 0..HIDDEN_CH {
                    let g_p2 = g_a2_p[co] * gelu_grad_f32(pre2[p * HIDDEN_CH + co]);
                    g_a1[p * HIDDEN_CH + co] += g_p2;
                    gw[OFF2_PW2_B + co] += g_p2;
                    for ci in 0..HIDDEN_CH {
                        gw[OFF2_PW2_W + co * HIDDEN_CH + ci] += g_p2 * ry[p * HIDDEN_CH + ci];
                        g_ry[p * HIDDEN_CH + ci] += g_p2 * w[OFF2_PW2_W + co * HIDDEN_CH + ci];
                    }
                }
            }
        }

        // Backprop DW2 Y -> rx
        let mut g_rx = vec![0.0f32; n_pix * HIDDEN_CH];
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let gr = g_ry[p * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..3 {
                        let gy = clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                        let src = (gy * nx + ix) * HIDDEN_CH + c;
                        gw[OFF2_DW2_KY + c * 3 + m] += gr * rx[src];
                        g_rx[src] += gr * w[OFF2_DW2_KY + c * 3 + m];
                    }
                }
            }
        }
        // Backprop DW2 X -> a1
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let gr = g_rx[p * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..3 {
                        let gx = clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                        let src = (iy * nx + gx) * HIDDEN_CH + c;
                        gw[OFF2_DW2_KX + c * 3 + m] += gr * a1[src];
                        g_a1[src] += gr * w[OFF2_DW2_KX + c * 3 + m];
                    }
                }
            }
        }
        // Backprop PW1 + GELU -> sy
        let mut g_sy = vec![0.0f32; n_pix * HIDDEN_CH];
        for p in 0..n_pix {
            for co in 0..HIDDEN_CH {
                let g_p1 = g_a1[p * HIDDEN_CH + co] * gelu_grad_f32(pre1[p * HIDDEN_CH + co]);
                gw[OFF2_PW1_B + co] += g_p1;
                for ci in 0..HIDDEN_CH {
                    gw[OFF2_PW1_W + co * HIDDEN_CH + ci] += g_p1 * sy[p * HIDDEN_CH + ci];
                    g_sy[p * HIDDEN_CH + ci] += g_p1 * w[OFF2_PW1_W + co * HIDDEN_CH + ci];
                }
            }
        }
        // Backprop Stem Y -> sx
        let mut g_sx = vec![0.0f32; n_pix * HIDDEN_CH];
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let gr = g_sy[p * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..5 {
                        let gy = clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                        let src = (gy * nx + ix) * HIDDEN_CH + c;
                        gw[OFF2_STEM_KY + c * 5 + m] += gr * sx[src];
                        g_sx[src] += gr * w[OFF2_STEM_KY + c * 5 + m];
                    }
                }
            }
        }
        // Backprop Stem X -> stem_kx
        for iy in 0..ny {
            for ix in 0..nx {
                let p = iy * nx + ix;
                for c in 0..HIDDEN_CH {
                    let gr = g_sx[p * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..5 {
                        let gx = clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                        gw[OFF2_STEM_KX + c * 5 + m] += gr * input[iy * nx + gx];
                    }
                }
            }
        }

        (total_loss, gw)
    }
}

#[allow(clippy::needless_range_loop)]
impl SeparableConvNet3D {
    /// Structured 3D signal-processing + Kaiming initialization.
    pub fn init_seeded(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut w = vec![0.0f32; CONV3D_NUM_WEIGHTS];

        let sym = [0.14f32, 0.36, 0.56, 0.36, 0.14];
        let dog = [-0.30f32, 0.12, 0.36, 0.12, -0.30];
        let flank = [-0.25f32, 0.50, 0.0, 0.50, -0.25];
        let d1 = [-0.22f32, -0.48, 0.0, 0.48, 0.22];
        let dc = [0.20f32, 0.20, 0.20, 0.20, 0.20];

        let templates: [([f32; 5], [f32; 5], [f32; 5]); 16] = [
            (dog, dog, dog),
            (flank, flank, flank),
            (dog, dog, flank),
            (flank, flank, dog),
            (d1, dog, dog),
            (dog, d1, dog),
            (dog, dog, d1),
            (sym, sym, dog),
            // Dilated d=2 channels: wide context & Z/Y ridge suppressors
            (dc, dc, dc),
            (dog, dog, dc), // Z-ridge detector (constant along 13C)
            (dog, dc, dog), // Y-ridge detector (constant along 15N)
            (flank, flank, dc),
            (d1, flank, flank),
            (flank, d1, flank),
            (flank, flank, d1),
            (dog, dog, dog),
        ];

        for (c, (tx, ty, tz)) in templates.iter().enumerate() {
            for m in 0..5 {
                w[OFF3_STEM_KX + c * 5 + m] = tx[m] + rng.normal(0.0, 0.04) as f32;
                w[OFF3_STEM_KY + c * 5 + m] = ty[m] + rng.normal(0.0, 0.04) as f32;
                w[OFF3_STEM_KZ + c * 5 + m] = tz[m] + rng.normal(0.0, 0.04) as f32;
            }
        }
        for i in 0..(HIDDEN_CH * HIDDEN_CH) {
            w[OFF3_PW1_W + i] = rng.normal(0.0, 0.18) as f32;
            w[OFF3_PW2_W + i] = rng.normal(0.0, 0.14) as f32;
        }
        for c in 0..8 {
            w[OFF3_PW1_W + (2 * c) * HIDDEN_CH + c] += 0.65;
            w[OFF3_PW1_W + (2 * c + 1) * HIDDEN_CH + c] -= 0.65;
        }
        for c in 0..HIDDEN_CH {
            for m in 0..3 {
                let base = if m == 1 { 0.72 } else { -0.10 };
                w[OFF3_DW2_KX + c * 3 + m] = base + rng.normal(0.0, 0.04) as f32;
                w[OFF3_DW2_KY + c * 3 + m] = base + rng.normal(0.0, 0.04) as f32;
                w[OFF3_DW2_KZ + c * 3 + m] = base + rng.normal(0.0, 0.04) as f32;
            }
        }
        for i in 0..(CONV3D_OUT_CH * HIDDEN_CH) {
            w[OFF3_HEAD_W + i] = rng.normal(0.0, 0.15) as f32;
        }
        w[OFF3_HEAD_B] = -2.4;
        w[OFF3_HEAD_B + 1] = -2.4;
        w[OFF3_HEAD_B + 5] = 0.2;
        Self { weights: w }
    }

    pub fn from_flat_weights(weights: &[f32]) -> Result<Self, String> {
        if weights.len() != CONV3D_NUM_WEIGHTS {
            return Err(format!(
                "expected {CONV3D_NUM_WEIGHTS} weights for SeparableConvNet3D, got {}",
                weights.len()
            ));
        }
        Ok(Self {
            weights: weights.to_vec(),
        })
    }

    pub fn to_flat_weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn to_json(&self) -> Result<String, String> {
        Ok(weights_to_json_string(&self.weights))
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let weights = weights_from_json_string(text, CONV3D_NUM_WEIGHTS)?;
        Ok(Self { weights })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + CONV3D_NUM_WEIGHTS * 4);
        out.extend_from_slice(BINARY_MAGIC_3D);
        for &v in &self.weights {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 8 + CONV3D_NUM_WEIGHTS * 4 || &bytes[..8] != BINARY_MAGIC_3D {
            return Err("invalid binary header or length for SeparableConvNet3D".to_string());
        }
        let mut weights = Vec::with_capacity(CONV3D_NUM_WEIGHTS);
        for chunk in bytes[8..].chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { weights })
    }

    /// Multi-threaded 3D forward pass across Z-slabs returning `[nz * ny * nx * 6]`
    /// (`prob_pos, prob_neg, dx, dy, dz, amp_norm`), matching WebGPU `conv3d.wgsl` 1:1.
    pub fn forward_3d(&self, input: &[f32], nx: usize, ny: usize, nz: usize) -> Vec<f32> {
        let w = &self.weights;
        let plane = nx * ny;
        let n_vox = plane * nz;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(nz.max(1));
        let z_per_t = nz.div_ceil(n_threads);

        // Pass 1 & 2 (per Z-plane): Stem X -> Stem Y -> sy [nz * ny * nx * 16]
        let mut sy = vec![0.0f32; n_vox * HIDDEN_CH];
        std::thread::scope(|s| {
            for (ti, z_chunk) in sy.chunks_mut(z_per_t * plane * HIDDEN_CH).enumerate() {
                let z0 = ti * z_per_t;
                let z_len = z_chunk.len() / (plane * HIDDEN_CH);
                s.spawn(move || {
                    let mut plane_sx = vec![0.0f32; plane * HIDDEN_CH];
                    for lz in 0..z_len {
                        let iz = z0 + lz;
                        let in_plane = &input[iz * plane..(iz + 1) * plane];
                        for iy in 0..ny {
                            let row = &in_plane[iy * nx..(iy + 1) * nx];
                            for ix in 0..nx {
                                let dst = &mut plane_sx
                                    [(iy * nx + ix) * HIDDEN_CH..(iy * nx + ix + 1) * HIDDEN_CH];
                                for c in 0..HIDDEN_CH {
                                    let d = chan_dilation(c);
                                    let kw = &w[OFF3_STEM_KX + c * 5..OFF3_STEM_KX + (c + 1) * 5];
                                    let mut acc = 0.0f32;
                                    for m in 0..5 {
                                        let gx =
                                            clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                                        acc += kw[m] * row[gx];
                                    }
                                    dst[c] = acc;
                                }
                            }
                        }
                        let dst_plane =
                            &mut z_chunk[lz * plane * HIDDEN_CH..(lz + 1) * plane * HIDDEN_CH];
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let dst = &mut dst_plane
                                    [(iy * nx + ix) * HIDDEN_CH..(iy * nx + ix + 1) * HIDDEN_CH];
                                for c in 0..HIDDEN_CH {
                                    let d = chan_dilation(c);
                                    let kw = &w[OFF3_STEM_KY + c * 5..OFF3_STEM_KY + (c + 1) * 5];
                                    let mut acc = 0.0f32;
                                    for m in 0..5 {
                                        let gy =
                                            clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                                        acc += kw[m] * plane_sx[(gy * nx + ix) * HIDDEN_CH + c];
                                    }
                                    dst[c] = acc;
                                }
                            }
                        }
                    }
                });
            }
        });

        // Pass 3: Stem Z + PW1 + GELU -> a1 [nz * ny * nx * 16]
        let mut a1 = vec![0.0f32; n_vox * HIDDEN_CH];
        let sy_ref = &sy;
        std::thread::scope(|s| {
            for (ti, z_chunk) in a1.chunks_mut(z_per_t * plane * HIDDEN_CH).enumerate() {
                let z0 = ti * z_per_t;
                let z_len = z_chunk.len() / (plane * HIDDEN_CH);
                s.spawn(move || {
                    for lz in 0..z_len {
                        let iz = z0 + lz;
                        for p in 0..plane {
                            let mut sz = [0.0f32; HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF3_STEM_KZ + c * 5..OFF3_STEM_KZ + (c + 1) * 5];
                                let mut acc = 0.0f32;
                                for m in 0..5 {
                                    let gz = clamp_idx((iz as isize) + d * ((m as isize) - 2), nz);
                                    acc += kw[m] * sy_ref[(gz * plane + p) * HIDDEN_CH + c];
                                }
                                sz[c] = acc;
                            }
                            let dst = &mut z_chunk
                                [(lz * plane + p) * HIDDEN_CH..(lz * plane + p + 1) * HIDDEN_CH];
                            for co in 0..HIDDEN_CH {
                                let pw = &w[OFF3_PW1_W + co * HIDDEN_CH
                                    ..OFF3_PW1_W + (co + 1) * HIDDEN_CH];
                                let mut sum = w[OFF3_PW1_B + co];
                                for ci in 0..HIDDEN_CH {
                                    sum += pw[ci] * sz[ci];
                                }
                                dst[co] = gelu_f32(sum);
                            }
                        }
                    }
                });
            }
        });

        // Pass 4 & 5 (per Z-plane): DW2 X -> DW2 Y -> ry [nz * ny * nx * 16]
        let mut ry = vec![0.0f32; n_vox * HIDDEN_CH];
        let a1_ref = &a1;
        std::thread::scope(|s| {
            for (ti, z_chunk) in ry.chunks_mut(z_per_t * plane * HIDDEN_CH).enumerate() {
                let z0 = ti * z_per_t;
                let z_len = z_chunk.len() / (plane * HIDDEN_CH);
                s.spawn(move || {
                    let mut plane_rx = vec![0.0f32; plane * HIDDEN_CH];
                    for lz in 0..z_len {
                        let iz = z0 + lz;
                        let a1_plane =
                            &a1_ref[iz * plane * HIDDEN_CH..(iz + 1) * plane * HIDDEN_CH];
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let dst = &mut plane_rx
                                    [(iy * nx + ix) * HIDDEN_CH..(iy * nx + ix + 1) * HIDDEN_CH];
                                for c in 0..HIDDEN_CH {
                                    let d = chan_dilation(c);
                                    let kw = &w[OFF3_DW2_KX + c * 3..OFF3_DW2_KX + (c + 1) * 3];
                                    let mut acc = 0.0f32;
                                    for m in 0..3 {
                                        let gx =
                                            clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                                        acc += kw[m] * a1_plane[(iy * nx + gx) * HIDDEN_CH + c];
                                    }
                                    dst[c] = acc;
                                }
                            }
                        }
                        let dst_plane =
                            &mut z_chunk[lz * plane * HIDDEN_CH..(lz + 1) * plane * HIDDEN_CH];
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let dst = &mut dst_plane
                                    [(iy * nx + ix) * HIDDEN_CH..(iy * nx + ix + 1) * HIDDEN_CH];
                                for c in 0..HIDDEN_CH {
                                    let d = chan_dilation(c);
                                    let kw = &w[OFF3_DW2_KY + c * 3..OFF3_DW2_KY + (c + 1) * 3];
                                    let mut acc = 0.0f32;
                                    for m in 0..3 {
                                        let gy =
                                            clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                                        acc += kw[m] * plane_rx[(gy * nx + ix) * HIDDEN_CH + c];
                                    }
                                    dst[c] = acc;
                                }
                            }
                        }
                    }
                });
            }
        });

        // Pass 6: DW2 Z + PW2 + Skip + GELU + Head -> out [nz * ny * nx * 6]
        let mut out = vec![0.0f32; n_vox * CONV3D_OUT_CH];
        let ry_ref = &ry;
        std::thread::scope(|s| {
            for (ti, z_chunk) in out.chunks_mut(z_per_t * plane * CONV3D_OUT_CH).enumerate() {
                let z0 = ti * z_per_t;
                let z_len = z_chunk.len() / (plane * CONV3D_OUT_CH);
                s.spawn(move || {
                    for lz in 0..z_len {
                        let iz = z0 + lz;
                        for p in 0..plane {
                            let mut rz = [0.0f32; HIDDEN_CH];
                            for c in 0..HIDDEN_CH {
                                let d = chan_dilation(c);
                                let kw = &w[OFF3_DW2_KZ + c * 3..OFF3_DW2_KZ + (c + 1) * 3];
                                let mut acc = 0.0f32;
                                for m in 0..3 {
                                    let gz = clamp_idx((iz as isize) + d * ((m as isize) - 1), nz);
                                    acc += kw[m] * ry_ref[(gz * plane + p) * HIDDEN_CH + c];
                                }
                                rz[c] = acc;
                            }
                            let skip = &a1_ref
                                [(iz * plane + p) * HIDDEN_CH..(iz * plane + p + 1) * HIDDEN_CH];
                            let mut a2 = [0.0f32; HIDDEN_CH];
                            for co in 0..HIDDEN_CH {
                                let pw = &w[OFF3_PW2_W + co * HIDDEN_CH
                                    ..OFF3_PW2_W + (co + 1) * HIDDEN_CH];
                                let mut sum = skip[co] + w[OFF3_PW2_B + co];
                                for ci in 0..HIDDEN_CH {
                                    sum += pw[ci] * rz[ci];
                                }
                                a2[co] = gelu_f32(sum);
                            }
                            let mut raw_h = [0.0f32; CONV3D_OUT_CH];
                            for k in 0..CONV3D_OUT_CH {
                                let hw = &w[OFF3_HEAD_W + k * HIDDEN_CH
                                    ..OFF3_HEAD_W + (k + 1) * HIDDEN_CH];
                                let mut sum = w[OFF3_HEAD_B + k];
                                for ci in 0..HIDDEN_CH {
                                    sum += hw[ci] * a2[ci];
                                }
                                raw_h[k] = sum;
                            }
                            let dst = &mut z_chunk[(lz * plane + p) * CONV3D_OUT_CH
                                ..(lz * plane + p + 1) * CONV3D_OUT_CH];
                            dst[0] = sigmoid_f32(raw_h[0]);
                            dst[1] = sigmoid_f32(raw_h[1]);
                            dst[2] = 0.5 * safe_tanh_f32(raw_h[2].clamp(-12.0, 12.0));
                            dst[3] = 0.5 * safe_tanh_f32(raw_h[3].clamp(-12.0, 12.0));
                            dst[4] = 0.5 * safe_tanh_f32(raw_h[4].clamp(-12.0, 12.0));
                            dst[5] = softplus_f32(raw_h[5]);
                        }
                    }
                });
            }
        });
        out
    }

    /// Exact halo-padded subvolume forward pass over `[x0..=x1, y0..=y1, 0..nz]`.
    /// Because Stem max radius is 4 and Block 2 max radius is 2 (total halo = 6 voxels),
    /// every voxel inside `[x0..=x1, y0..=y1, 0..nz]` is bit-for-bit identical to `forward_3d`!
    #[allow(clippy::too_many_arguments)]
    pub fn forward_3d_subregion(
        &self,
        full_input: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
    ) -> (usize, usize, usize, usize, Vec<f32>) {
        let hx0 = x0.saturating_sub(6);
        let hx1 = (x1 + 6).min(nx - 1);
        let hy0 = y0.saturating_sub(6);
        let hy1 = (y1 + 6).min(ny - 1);
        let snx = hx1 - hx0 + 1;
        let sny = hy1 - hy0 + 1;
        let mut sub_in = vec![0.0f32; nz * sny * snx];
        for iz in 0..nz {
            for ly in 0..sny {
                let gy = hy0 + ly;
                let src_base = (iz * ny + gy) * nx + hx0;
                let dst_base = (iz * sny + ly) * snx;
                sub_in[dst_base..dst_base + snx]
                    .copy_from_slice(&full_input[src_base..src_base + snx]);
            }
        }
        let sub_out = self.forward_3d(&sub_in, snx, sny, nz);
        (hx0, hy0, snx, sny, sub_out)
    }

    /// Exact analytical forward + backward pass wrt all `1,030` weights.
    pub fn forward_backward_raw_3d(
        &self,
        input: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        grad_from_raw_head: impl Fn(usize, &[f32; CONV3D_OUT_CH], &mut [f32; CONV3D_OUT_CH]) -> f64,
    ) -> (f64, Vec<f32>) {
        let w = &self.weights;
        let plane = nx * ny;
        let n_vox = plane * nz;

        let mut sx = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut sy = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut sz = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut pre1 = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut a1 = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut rx = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut ry = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut rz = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut pre2 = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut a2 = vec![0.0f32; n_vox * HIDDEN_CH];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let d = chan_dilation(c);
                        let mut acc = 0.0f32;
                        for m in 0..5 {
                            let gx = clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                            acc += w[OFF3_STEM_KX + c * 5 + m] * input[(iz * ny + iy) * nx + gx];
                        }
                        sx[v * HIDDEN_CH + c] = acc;
                    }
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let d = chan_dilation(c);
                        let mut acc = 0.0f32;
                        for m in 0..5 {
                            let gy = clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                            acc += w[OFF3_STEM_KY + c * 5 + m]
                                * sx[((iz * ny + gy) * nx + ix) * HIDDEN_CH + c];
                        }
                        sy[v * HIDDEN_CH + c] = acc;
                    }
                }
            }
        }

        for iz in 0..nz {
            for p in 0..plane {
                let v = iz * plane + p;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..5 {
                        let gz = clamp_idx((iz as isize) + d * ((m as isize) - 2), nz);
                        acc += w[OFF3_STEM_KZ + c * 5 + m] * sy[(gz * plane + p) * HIDDEN_CH + c];
                    }
                    sz[v * HIDDEN_CH + c] = acc;
                }
                for co in 0..HIDDEN_CH {
                    let mut sum = w[OFF3_PW1_B + co];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF3_PW1_W + co * HIDDEN_CH + ci] * sz[v * HIDDEN_CH + ci];
                    }
                    pre1[v * HIDDEN_CH + co] = sum;
                    a1[v * HIDDEN_CH + co] = gelu_f32(sum);
                }
            }
        }

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let d = chan_dilation(c);
                        let mut acc = 0.0f32;
                        for m in 0..3 {
                            let gx = clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                            acc += w[OFF3_DW2_KX + c * 3 + m]
                                * a1[((iz * ny + iy) * nx + gx) * HIDDEN_CH + c];
                        }
                        rx[v * HIDDEN_CH + c] = acc;
                    }
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let d = chan_dilation(c);
                        let mut acc = 0.0f32;
                        for m in 0..3 {
                            let gy = clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                            acc += w[OFF3_DW2_KY + c * 3 + m]
                                * rx[((iz * ny + gy) * nx + ix) * HIDDEN_CH + c];
                        }
                        ry[v * HIDDEN_CH + c] = acc;
                    }
                }
            }
        }

        let mut total_loss = 0.0f64;
        let mut gw = vec![0.0f32; CONV3D_NUM_WEIGHTS];
        let mut g_rz = vec![0.0f32; n_vox * HIDDEN_CH];
        let mut g_a1 = vec![0.0f32; n_vox * HIDDEN_CH];

        for iz in 0..nz {
            for p in 0..plane {
                let v = iz * plane + p;
                for c in 0..HIDDEN_CH {
                    let d = chan_dilation(c);
                    let mut acc = 0.0f32;
                    for m in 0..3 {
                        let gz = clamp_idx((iz as isize) + d * ((m as isize) - 1), nz);
                        acc += w[OFF3_DW2_KZ + c * 3 + m] * ry[(gz * plane + p) * HIDDEN_CH + c];
                    }
                    rz[v * HIDDEN_CH + c] = acc;
                }
                for co in 0..HIDDEN_CH {
                    let mut sum = a1[v * HIDDEN_CH + co] + w[OFF3_PW2_B + co];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF3_PW2_W + co * HIDDEN_CH + ci] * rz[v * HIDDEN_CH + ci];
                    }
                    pre2[v * HIDDEN_CH + co] = sum;
                    a2[v * HIDDEN_CH + co] = gelu_f32(sum);
                }
                let mut raw_h = [0.0f32; CONV3D_OUT_CH];
                for k in 0..CONV3D_OUT_CH {
                    let mut sum = w[OFF3_HEAD_B + k];
                    for ci in 0..HIDDEN_CH {
                        sum += w[OFF3_HEAD_W + k * HIDDEN_CH + ci] * a2[v * HIDDEN_CH + ci];
                    }
                    raw_h[k] = sum;
                }
                let mut g_raw = [0.0f32; CONV3D_OUT_CH];
                total_loss += grad_from_raw_head(v, &raw_h, &mut g_raw);

                let mut g_a2_v = [0.0f32; HIDDEN_CH];
                for k in 0..CONV3D_OUT_CH {
                    let gr = g_raw[k];
                    if gr == 0.0 {
                        continue;
                    }
                    gw[OFF3_HEAD_B + k] += gr;
                    for ci in 0..HIDDEN_CH {
                        gw[OFF3_HEAD_W + k * HIDDEN_CH + ci] += gr * a2[v * HIDDEN_CH + ci];
                        g_a2_v[ci] += gr * w[OFF3_HEAD_W + k * HIDDEN_CH + ci];
                    }
                }
                for co in 0..HIDDEN_CH {
                    let g_p2 = g_a2_v[co] * gelu_grad_f32(pre2[v * HIDDEN_CH + co]);
                    g_a1[v * HIDDEN_CH + co] += g_p2;
                    gw[OFF3_PW2_B + co] += g_p2;
                    for ci in 0..HIDDEN_CH {
                        gw[OFF3_PW2_W + co * HIDDEN_CH + ci] += g_p2 * rz[v * HIDDEN_CH + ci];
                        g_rz[v * HIDDEN_CH + ci] += g_p2 * w[OFF3_PW2_W + co * HIDDEN_CH + ci];
                    }
                }
            }
        }

        // Backprop DW2 Z -> ry
        let mut g_ry = vec![0.0f32; n_vox * HIDDEN_CH];
        for iz in 0..nz {
            for p in 0..plane {
                let v = iz * plane + p;
                for c in 0..HIDDEN_CH {
                    let gr = g_rz[v * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..3 {
                        let gz = clamp_idx((iz as isize) + d * ((m as isize) - 1), nz);
                        let src = (gz * plane + p) * HIDDEN_CH + c;
                        gw[OFF3_DW2_KZ + c * 3 + m] += gr * ry[src];
                        g_ry[src] += gr * w[OFF3_DW2_KZ + c * 3 + m];
                    }
                }
            }
        }
        // Backprop DW2 Y -> rx
        let mut g_rx = vec![0.0f32; n_vox * HIDDEN_CH];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let gr = g_ry[v * HIDDEN_CH + c];
                        let d = chan_dilation(c);
                        for m in 0..3 {
                            let gy = clamp_idx((iy as isize) + d * ((m as isize) - 1), ny);
                            let src = ((iz * ny + gy) * nx + ix) * HIDDEN_CH + c;
                            gw[OFF3_DW2_KY + c * 3 + m] += gr * rx[src];
                            g_rx[src] += gr * w[OFF3_DW2_KY + c * 3 + m];
                        }
                    }
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let gr = g_rx[v * HIDDEN_CH + c];
                        let d = chan_dilation(c);
                        for m in 0..3 {
                            let gx = clamp_idx((ix as isize) + d * ((m as isize) - 1), nx);
                            let src = ((iz * ny + iy) * nx + gx) * HIDDEN_CH + c;
                            gw[OFF3_DW2_KX + c * 3 + m] += gr * a1[src];
                            g_a1[src] += gr * w[OFF3_DW2_KX + c * 3 + m];
                        }
                    }
                }
            }
        }
        // Backprop PW1 + GELU -> sz
        let mut g_sz = vec![0.0f32; n_vox * HIDDEN_CH];
        for v in 0..n_vox {
            for co in 0..HIDDEN_CH {
                let g_p1 = g_a1[v * HIDDEN_CH + co] * gelu_grad_f32(pre1[v * HIDDEN_CH + co]);
                gw[OFF3_PW1_B + co] += g_p1;
                for ci in 0..HIDDEN_CH {
                    gw[OFF3_PW1_W + co * HIDDEN_CH + ci] += g_p1 * sz[v * HIDDEN_CH + ci];
                    g_sz[v * HIDDEN_CH + ci] += g_p1 * w[OFF3_PW1_W + co * HIDDEN_CH + ci];
                }
            }
        }
        // Backprop Stem Z -> sy
        let mut g_sy = vec![0.0f32; n_vox * HIDDEN_CH];
        for iz in 0..nz {
            for p in 0..plane {
                let v = iz * plane + p;
                for c in 0..HIDDEN_CH {
                    let gr = g_sz[v * HIDDEN_CH + c];
                    let d = chan_dilation(c);
                    for m in 0..5 {
                        let gz = clamp_idx((iz as isize) + d * ((m as isize) - 2), nz);
                        let src = (gz * plane + p) * HIDDEN_CH + c;
                        gw[OFF3_STEM_KZ + c * 5 + m] += gr * sy[src];
                        g_sy[src] += gr * w[OFF3_STEM_KZ + c * 5 + m];
                    }
                }
            }
        }
        // Backprop Stem Y -> sx -> Stem X
        let mut g_sx = vec![0.0f32; n_vox * HIDDEN_CH];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let gr = g_sy[v * HIDDEN_CH + c];
                        let d = chan_dilation(c);
                        for m in 0..5 {
                            let gy = clamp_idx((iy as isize) + d * ((m as isize) - 2), ny);
                            let src = ((iz * ny + gy) * nx + ix) * HIDDEN_CH + c;
                            gw[OFF3_STEM_KY + c * 5 + m] += gr * sx[src];
                            g_sx[src] += gr * w[OFF3_STEM_KY + c * 5 + m];
                        }
                    }
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = (iz * ny + iy) * nx + ix;
                    for c in 0..HIDDEN_CH {
                        let gr = g_sx[v * HIDDEN_CH + c];
                        let d = chan_dilation(c);
                        for m in 0..5 {
                            let gx = clamp_idx((ix as isize) + d * ((m as isize) - 2), nx);
                            gw[OFF3_STEM_KX + c * 5 + m] += gr * input[(iz * ny + iy) * nx + gx];
                        }
                    }
                }
            }
        }

        (total_loss, gw)
    }
}

// ============================================================================
// In-Rust Adam + Focal-Loss + Sub-Voxel Huber Offset Trainer
// ============================================================================

#[derive(Clone, Debug)]
pub struct TrainerConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr_start: f32,
    pub lr_end: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            steps: 720,
            batch_size: 8,
            lr_start: 0.006,
            lr_end: 0.00025,
            weight_decay: 1e-5,
            grad_clip: 5.0,
            seed: 2026,
        }
    }
}

#[inline]
fn focal_bce_and_grad(raw_logit: f32, target_heat: f32, norm_pos: f32) -> (f64, f32) {
    let p = sigmoid_f32(raw_logit).clamp(1e-6, 1.0 - 1e-6);
    let gamma = 2.0f32;
    if target_heat >= 0.999 {
        let alpha_pos = 2.8f32 / norm_pos;
        let one_m_p = 1.0 - p;
        let w = alpha_pos * one_m_p.powf(gamma);
        let loss = -(w as f64) * (p as f64).ln();
        let grad = w * (gamma * p * p.ln() - one_m_p);
        (loss, grad)
    } else {
        let neg_w = (1.0 - target_heat).powi(4) / norm_pos;
        let p_gam = p.powf(gamma);
        let one_m_p = 1.0 - p;
        let loss = -(neg_w as f64) * (p_gam as f64) * (one_m_p as f64).ln();
        let grad = neg_w * p_gam * (p - gamma * one_m_p * one_m_p.ln());
        (loss, grad)
    }
}

#[inline]
fn huber_offset_and_grad(raw_t: f32, target_off: f32, scale: f32) -> (f64, f32) {
    let tc = raw_t.clamp(-12.0, 12.0);
    let th = safe_tanh_f32(tc);
    let pred = 0.5 * th;
    let err = pred - target_off;
    let beta = 0.05f32;
    let (loss, d_pred) = if err.abs() < beta {
        (0.5 * (err * err / beta) as f64, err / beta)
    } else {
        ((err.abs() - 0.5 * beta) as f64, err.signum())
    };
    let d_raw = 0.5 * (1.0 - th * th);
    (loss * (scale as f64), d_pred * d_raw * scale)
}

#[inline]
fn huber_amp_and_grad(raw_a: f32, target_amp: f32, scale: f32) -> (f64, f32) {
    let pred = softplus_f32(raw_a);
    let err = pred - target_amp;
    let beta = 0.15f32;
    let (loss, d_pred) = if err.abs() < beta {
        (0.5 * (err * err / beta) as f64, err / beta)
    } else {
        ((err.abs() - 0.5 * beta) as f64, err.signum())
    };
    let d_raw = sigmoid_f32(raw_a);
    (loss * (scale as f64), d_pred * d_raw * scale)
}

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl AdamState {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    fn step(&mut self, weights: &mut [f32], grads: &mut [f32], lr: f32, wd: f32, clip: f32) {
        self.t += 1;
        let norm_sq: f32 = grads.iter().map(|g| g * g).sum();
        let norm = norm_sq.sqrt();
        if norm > clip && norm.is_finite() {
            let s = clip / norm;
            for g in grads.iter_mut() {
                *g *= s;
            }
        }
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..weights.len() {
            let g = grads[i];
            if !g.is_finite() {
                continue;
            }
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            weights[i] -= lr * (m_hat / (v_hat.sqrt() + 1e-8) + wd * weights[i]);
        }
    }
}

/// Train `SeparableConvNet2D` from scratch on synthetic Voigt/ridge/baseline/shoulder patches
/// using Adam + Focal Loss + Sub-Voxel Huber Offset Loss. Returns `(net, initial_loss, final_loss)`.
pub fn train_weights_2d(cfg: &TrainerConfig) -> (SeparableConvNet2D, f64, f64) {
    let mut net = SeparableConvNet2D::init_seeded(cfg.seed);
    let mut adam = AdamState::new(CONV2D_NUM_WEIGHTS);
    let nx = 32usize;
    let ny = 32usize;
    let bsz = cfg.batch_size.max(1);
    let mut first_loss = 0.0f64;
    let mut last_loss = 0.0f64;

    for step in 0..cfg.steps.max(1) {
        let progress = (step as f32) / (cfg.steps.saturating_sub(1).max(1) as f32);
        let cos_w = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        let lr = cfg.lr_end + (cfg.lr_start - cfg.lr_end) * cos_w;

        let net_ref = &net;
        let sample_results: Vec<(f64, Vec<f32>)> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(bsz);
            for b in 0..bsz {
                let p_seed = cfg
                    .seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(((step * 16 + b) as u64) * 1442695040888963407);
                handles.push(s.spawn(move || {
                    let syn = generate_synthetic_spectrum_2d(&SyntheticGenConfig2D {
                        nx,
                        ny,
                        n_peaks: 5 + (b % 4) * 2,
                        allow_negative: true,
                        min_snr: 3.5,
                        max_snr: 28.0,
                        noise_sigma: 1.0,
                        ridge_prob: 0.78,
                        ridge_rel_amp: 0.18,
                        baseline_roll_sigma: 2.5,
                        shoulder_prob: 0.40,
                        j_split_prob: 0.25,
                        seed: p_seed,
                    });
                    let (norm_in, _, est_noise) =
                        normalize_spectrum_data_with_nx(&syn.spectrum.data, nx);

                    let n_pix = nx * ny;
                    let mut heat_pos = vec![0.0f32; n_pix];
                    let mut heat_neg = vec![0.0f32; n_pix];
                    let mut off_x = vec![0.0f32; n_pix];
                    let mut off_y = vec![0.0f32; n_pix];
                    let mut amp_t = vec![0.0f32; n_pix];
                    let mut has_peak = vec![false; n_pix];

                    for g in &syn.ground_truth {
                        let ix0 = g.x_idx.round().clamp(1.0, (nx - 2) as f64) as usize;
                        let iy0 = g.y_idx.round().clamp(1.0, (ny - 2) as f64) as usize;
                        let p0 = iy0 * nx + ix0;
                        if g.sign >= 0 {
                            heat_pos[p0] = 1.0;
                        } else {
                            heat_neg[p0] = 1.0;
                        }
                        off_x[p0] = ((g.x_idx - (ix0 as f64)) as f32).clamp(-0.49, 0.49);
                        off_y[p0] = ((g.y_idx - (iy0 as f64)) as f32).clamp(-0.49, 0.49);
                        amp_t[p0] = g.amplitude / (NORM_SIGMA_DIV * est_noise);
                        has_peak[p0] = true;

                        for dy in -2isize..=2 {
                            for dx in -2isize..=2 {
                                let gx = (ix0 as isize + dx).clamp(0, nx as isize - 1) as usize;
                                let gy = (iy0 as isize + dy).clamp(0, ny as isize - 1) as usize;
                                let gp = gy * nx + gx;
                                if gp == p0 {
                                    continue;
                                }
                                let d2 = ((gx as f64 - g.x_idx).powi(2)
                                    + (gy as f64 - g.y_idx).powi(2))
                                    as f32;
                                let hval = (-d2 / (2.0 * 0.65 * 0.65)).exp().min(0.95);
                                if g.sign >= 0 {
                                    heat_pos[gp] = heat_pos[gp].max(hval);
                                } else {
                                    heat_neg[gp] = heat_neg[gp].max(hval);
                                }
                            }
                        }
                    }

                    let norm_pos = (syn.ground_truth.len().max(1) as f32) * (bsz as f32);
                    net_ref.forward_backward_raw_2d(&norm_in, nx, ny, |p, raw_h, g_raw| {
                        let (lp, gp) = focal_bce_and_grad(raw_h[0], heat_pos[p], norm_pos);
                        let (ln, gn) = focal_bce_and_grad(raw_h[1], heat_neg[p], norm_pos);
                        g_raw[0] = gp;
                        g_raw[1] = gn;
                        let mut l_tot = lp + ln;
                        if has_peak[p] {
                            let s = 3.0 / norm_pos;
                            let (lx, gx_o) = huber_offset_and_grad(raw_h[2], off_x[p], s);
                            let (ly, gy_o) = huber_offset_and_grad(raw_h[3], off_y[p], s);
                            let (la, ga_o) = huber_amp_and_grad(raw_h[4], amp_t[p], 0.8 / norm_pos);
                            g_raw[2] = gx_o;
                            g_raw[3] = gy_o;
                            g_raw[4] = ga_o;
                            l_tot += lx + ly + la;
                        }
                        l_tot
                    })
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut batch_grad = vec![0.0f32; CONV2D_NUM_WEIGHTS];
        let mut batch_loss = 0.0f64;
        for (loss, gw) in sample_results {
            batch_loss += loss;
            for i in 0..CONV2D_NUM_WEIGHTS {
                batch_grad[i] += gw[i];
            }
        }

        if step < 4 {
            first_loss += batch_loss / 4.0;
        }
        if step + 4 >= cfg.steps {
            last_loss += batch_loss / (cfg.steps.min(4) as f64);
        }
        adam.step(
            &mut net.weights,
            &mut batch_grad,
            lr,
            cfg.weight_decay,
            cfg.grad_clip,
        );
    }

    (net, first_loss, last_loss)
}

/// Train default 2D CNN weights (`SeparableConvNet2D`).
pub fn train_default_weights_2d() -> SeparableConvNet2D {
    train_weights_2d(&TrainerConfig::default()).0
}

/// Train `SeparableConvNet3D` from scratch on synthetic 3D Voigt/ridge/baseline/shoulder cubes
/// using Adam + Focal Loss + Sub-Voxel Huber Offset Loss. Returns `(net, initial_loss, final_loss)`.
pub fn train_weights_3d(cfg: &TrainerConfig) -> (SeparableConvNet3D, f64, f64) {
    let mut net = SeparableConvNet3D::init_seeded(cfg.seed ^ 0x55AA_55AA);
    let mut adam = AdamState::new(CONV3D_NUM_WEIGHTS);
    let (nx, ny, nz) = (16usize, 16usize, 20usize);
    let bsz = cfg.batch_size.max(1);
    let mut first_loss = 0.0f64;
    let mut last_loss = 0.0f64;

    for step in 0..cfg.steps.max(1) {
        let progress = (step as f32) / (cfg.steps.saturating_sub(1).max(1) as f32);
        let cos_w = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        let lr = cfg.lr_end + (cfg.lr_start - cfg.lr_end) * cos_w;

        let net_ref = &net;
        let sample_results: Vec<(f64, Vec<f32>)> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(bsz);
            for b in 0..bsz {
                let p_seed = cfg
                    .seed
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(((step * 16 + b) as u64) * 3037000493);
                handles.push(s.spawn(move || {
                    let syn = generate_synthetic_spectrum_3d(&SyntheticGenConfig3D {
                        nx,
                        ny,
                        nz,
                        exp: "HNCACB".to_string(),
                        n_peaks: 5 + (b % 3) * 2,
                        allow_negative: true,
                        min_snr: 3.5,
                        max_snr: 26.0,
                        noise_sigma: 1.0,
                        ridge_prob: 0.78,
                        ridge_rel_amp: 0.18,
                        baseline_roll_sigma: 2.4,
                        shoulder_prob: 0.40,
                        j_split_prob: 0.20,
                        seed: p_seed,
                    });
                    let (norm_in, _, est_noise) =
                        normalize_spectrum_data_with_nx(&syn.spectrum.data, nx);

                    let n_vox = nx * ny * nz;
                    let mut heat_pos = vec![0.0f32; n_vox];
                    let mut heat_neg = vec![0.0f32; n_vox];
                    let mut off_x = vec![0.0f32; n_vox];
                    let mut off_y = vec![0.0f32; n_vox];
                    let mut off_z = vec![0.0f32; n_vox];
                    let mut amp_t = vec![0.0f32; n_vox];
                    let mut has_peak = vec![false; n_vox];

                    for g in &syn.ground_truth {
                        let ix0 = g.x_idx.round().clamp(1.0, (nx - 2) as f64) as usize;
                        let iy0 = g.y_idx.round().clamp(1.0, (ny - 2) as f64) as usize;
                        let iz0 = g.z_idx.round().clamp(1.0, (nz - 2) as f64) as usize;
                        let v0 = (iz0 * ny + iy0) * nx + ix0;
                        if g.sign >= 0 {
                            heat_pos[v0] = 1.0;
                        } else {
                            heat_neg[v0] = 1.0;
                        }
                        off_x[v0] = ((g.x_idx - (ix0 as f64)) as f32).clamp(-0.49, 0.49);
                        off_y[v0] = ((g.y_idx - (iy0 as f64)) as f32).clamp(-0.49, 0.49);
                        off_z[v0] = ((g.z_idx - (iz0 as f64)) as f32).clamp(-0.49, 0.49);
                        amp_t[v0] = g.amplitude / (NORM_SIGMA_DIV * est_noise);
                        has_peak[v0] = true;

                        for dz in -2isize..=2 {
                            for dy in -2isize..=2 {
                                for dx in -2isize..=2 {
                                    let gx = (ix0 as isize + dx).clamp(0, nx as isize - 1) as usize;
                                    let gy = (iy0 as isize + dy).clamp(0, ny as isize - 1) as usize;
                                    let gz = (iz0 as isize + dz).clamp(0, nz as isize - 1) as usize;
                                    let gv = (gz * ny + gy) * nx + gx;
                                    if gv == v0 {
                                        continue;
                                    }
                                    let d2 = ((gx as f64 - g.x_idx).powi(2)
                                        + (gy as f64 - g.y_idx).powi(2)
                                        + (gz as f64 - g.z_idx).powi(2))
                                        as f32;
                                    let hval = (-d2 / (2.0 * 0.65 * 0.65)).exp().min(0.95);
                                    if g.sign >= 0 {
                                        heat_pos[gv] = heat_pos[gv].max(hval);
                                    } else {
                                        heat_neg[gv] = heat_neg[gv].max(hval);
                                    }
                                }
                            }
                        }
                    }

                    let norm_pos = (syn.ground_truth.len().max(1) as f32) * (bsz as f32);
                    net_ref.forward_backward_raw_3d(&norm_in, nx, ny, nz, |v, raw_h, g_raw| {
                        let (lp, gp) = focal_bce_and_grad(raw_h[0], heat_pos[v], norm_pos);
                        let (ln, gn) = focal_bce_and_grad(raw_h[1], heat_neg[v], norm_pos);
                        g_raw[0] = gp;
                        g_raw[1] = gn;
                        let mut l_tot = lp + ln;
                        if has_peak[v] {
                            let s = 3.0 / norm_pos;
                            let (lx, gx_o) = huber_offset_and_grad(raw_h[2], off_x[v], s);
                            let (ly, gy_o) = huber_offset_and_grad(raw_h[3], off_y[v], s);
                            let (lz, gz_o) = huber_offset_and_grad(raw_h[4], off_z[v], s);
                            let (la, ga_o) = huber_amp_and_grad(raw_h[5], amp_t[v], 0.8 / norm_pos);
                            g_raw[2] = gx_o;
                            g_raw[3] = gy_o;
                            g_raw[4] = gz_o;
                            g_raw[5] = ga_o;
                            l_tot += lx + ly + lz + la;
                        }
                        l_tot
                    })
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut batch_grad = vec![0.0f32; CONV3D_NUM_WEIGHTS];
        let mut batch_loss = 0.0f64;
        for (loss, gw) in sample_results {
            batch_loss += loss;
            for i in 0..CONV3D_NUM_WEIGHTS {
                batch_grad[i] += gw[i];
            }
        }

        if step < 4 {
            first_loss += batch_loss / 4.0;
        }
        if step + 4 >= cfg.steps {
            last_loss += batch_loss / (cfg.steps.min(4) as f64);
        }
        adam.step(
            &mut net.weights,
            &mut batch_grad,
            lr,
            cfg.weight_decay,
            cfg.grad_clip,
        );
    }

    (net, first_loss, last_loss)
}

/// Train default 3D CNN weights (`SeparableConvNet3D`).
pub fn train_default_weights_3d() -> SeparableConvNet3D {
    train_weights_3d(&TrainerConfig::default()).0
}

// ============================================================================
// NeuralPicker & Public Peak-Picking Entry Points
// ============================================================================

#[derive(Clone, Debug)]
pub struct NeuralPickerConfig {
    /// Peak center probability cutoff in `(0, 1)` (default `0.46`).
    pub prob_threshold: f32,
    /// Minimum normalized `|z| = |v - med| / (10 * sigma)` guard (default `0.22` = `2.2 sigma`).
    pub min_snr_norm: f32,
    /// Allow negative peaks (`-1`, auto-enabled for `HNCACB`).
    pub allow_negative: bool,
    /// Optional HSQC root peaks to restrict 3D picking to `(H, N)` strips (`strip_tol_h`, `strip_tol_n`).
    pub hsqc_roots: Vec<Peak>,
    pub strip_tol_h: f64,
    pub strip_tol_n: f64,
}

impl Default for NeuralPickerConfig {
    fn default() -> Self {
        Self {
            prob_threshold: 0.45,
            min_snr_norm: 0.08,
            allow_negative: false,
            hsqc_roots: Vec::new(),
            strip_tol_h: 0.035,
            strip_tol_n: 0.35,
        }
    }
}

impl NeuralPickerConfig {
    pub fn with_roots(mut self, roots: &[Peak]) -> Self {
        self.hsqc_roots = roots.to_vec();
        self
    }
}

#[derive(Clone, Debug)]
pub struct NeuralPicker {
    pub net2d: SeparableConvNet2D,
    pub net3d: SeparableConvNet3D,
}

impl NeuralPicker {
    /// Load converged clean-room pretrained 2D & 3D weights in `< 0.1 ms` (embedded constants).
    pub fn pretrained() -> Self {
        if PRETRAINED_WEIGHTS_2D_READY && PRETRAINED_WEIGHTS_3D_READY {
            Self {
                net2d: SeparableConvNet2D {
                    weights: PRETRAINED_WEIGHTS_2D.to_vec(),
                },
                net3d: SeparableConvNet3D {
                    weights: PRETRAINED_WEIGHTS_3D.to_vec(),
                },
            }
        } else {
            Self {
                net2d: train_default_weights_2d(),
                net3d: train_default_weights_3d(),
            }
        }
    }

    pub fn pick_2d(&self, spec: &Spectrum2D, cfg: &NeuralPickerConfig) -> Vec<Peak> {
        let (nx, ny) = (spec.nx(), spec.ny());
        if nx < 3 || ny < 3 {
            return Vec::new();
        }
        let (norm_in, med, noise) = normalize_spectrum_data_with_nx(&spec.data, nx);
        let out = self.net2d.forward_2d(&norm_in, nx, ny);
        // Bayes spatial-sparsity calibration for large full-sweep 2D grids (> 8,192 pixels)
        let p_clamped = cfg.prob_threshold.clamp(0.02, 0.98);
        let base_logit = (p_clamped / (1.0 - p_clamped)).ln();
        let area_shift = 0.21 * (((nx * ny) as f32 / 8192.0).ln()).max(0.0);
        let eff_prob_thresh = sigmoid_f32(base_logit + area_shift);
        let mut peaks = Vec::new();

        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let p0 = iy * nx + ix;
                let base0 = p0 * CONV2D_OUT_CH;

                for &(ch, sign) in &[(0usize, 1i32), (1usize, -1i32)] {
                    if sign < 0 && !cfg.allow_negative {
                        continue;
                    }
                    let prob = out[base0 + ch];
                    if prob < eff_prob_thresh {
                        continue;
                    }
                    let z_val = norm_in[p0];
                    if (sign > 0 && z_val < cfg.min_snr_norm)
                        || (sign < 0 && z_val > -cfg.min_snr_norm)
                    {
                        continue;
                    }
                    // 3x3 Non-Maximum Suppression on probability map (deterministic index tie-break)
                    let mut is_max = true;
                    'nms2: for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let p1 =
                                ((iy as isize + dy) as usize) * nx + ((ix as isize + dx) as usize);
                            let np = out[p1 * CONV2D_OUT_CH + ch];
                            if np > prob || (np == prob && p1 < p0) {
                                is_max = false;
                                break 'nms2;
                            }
                        }
                    }
                    if !is_max {
                        continue;
                    }

                    let dx = out[base0 + 2] as f64;
                    let dy = out[base0 + 3] as f64;
                    let h = spec.axis_x.index_to_ppm((ix as f64) + dx);
                    let n = spec.axis_y.index_to_ppm((iy as f64) + dy);

                    let cnn_height = (out[base0 + 4] * NORM_SIGMA_DIV * noise) as f64;
                    let raw_height = ((spec.get(ix, iy) - med).abs()) as f64;
                    let height = (0.85 * cnn_height + 0.15 * raw_height).max(1e-6);

                    let mut vol = 0.0f64;
                    for wy in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = spec
                                .get((ix as isize + wx) as usize, (iy as isize + wy) as usize)
                                - med;
                            if (sign > 0 && sv > 0.0) || (sign < 0 && sv < 0.0) {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                    peaks.push(Peak {
                        exp: "HSQC".to_string(),
                        h,
                        n,
                        c: None,
                        sign,
                        height: Some(height),
                        volume: Some(vol.max(height)),
                        h2: None,
                    });
                }
            }
        }
        peaks
    }

    pub fn pick_3d(
        &self,
        spec: &Spectrum3D,
        experiment: &str,
        cfg: &NeuralPickerConfig,
    ) -> Vec<Peak> {
        let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
        if nx < 3 || ny < 3 || nz < 3 {
            return Vec::new();
        }
        let exp_name = if experiment.is_empty() {
            spec.exp.as_str()
        } else {
            experiment
        };
        let allow_neg = cfg.allow_negative || exp_name == "HNCACB";
        let y_is_n = spec.axis_y.label.contains('N')
            || (spec.axis_y.max_ppm() > 95.0
                && spec.axis_y.min_ppm() < 145.0
                && !spec.axis_z.label.contains('N'));

        let (norm_in, med, noise) = normalize_spectrum_data_with_nx(&spec.data, nx);
        let eff_prob_thresh = cfg.prob_threshold;

        // Fast exact halo-subregion strip execution when HSQC roots are supplied (and y_is_n)
        if !cfg.hsqc_roots.is_empty() && y_is_n {
            // Merge overlapping (H, N) root boxes onto a boolean active mask [ny * nx]
            let mut active = vec![false; ny * nx];
            let mut root_windows: Vec<(usize, usize, usize, usize)> = Vec::new();
            for r in &cfg.hsqc_roots {
                let x_a = spec.axis_x.ppm_to_index(r.h + cfg.strip_tol_h);
                let x_b = spec.axis_x.ppm_to_index(r.h - cfg.strip_tol_h);
                let y_a = spec.axis_y.ppm_to_index(r.n + cfg.strip_tol_n);
                let y_b = spec.axis_y.ppm_to_index(r.n - cfg.strip_tol_n);
                let x0 = (x_a.min(x_b).floor() as isize).clamp(1, nx as isize - 2) as usize;
                let x1 = (x_a.max(x_b).ceil() as isize).clamp(1, nx as isize - 2) as usize;
                let y0 = (y_a.min(y_b).floor() as isize).clamp(1, ny as isize - 2) as usize;
                let y1 = (y_a.max(y_b).ceil() as isize).clamp(1, ny as isize - 2) as usize;
                for iy in y0..=y1 {
                    for ix in x0..=x1 {
                        active[iy * nx + ix] = true;
                    }
                }
                root_windows.push((x0, x1, y0, y1));
            }

            // Group nearby windows or evaluate each strip tile while deduplicating via `visited`
            let mut visited = vec![false; ny * nx];
            let mut indexed_peaks: Vec<((usize, usize, usize, i32), Peak)> = Vec::new();
            for &(wx0, wx1, wy0, wy1) in &root_windows {
                let mut need = false;
                for iy in wy0..=wy1 {
                    for ix in wx0..=wx1 {
                        if !visited[iy * nx + ix] {
                            need = true;
                            break;
                        }
                    }
                }
                if !need {
                    continue;
                }
                // Expand slightly to cover immediate 1-voxel NMS halo
                let ex0 = wx0.saturating_sub(1);
                let ex1 = (wx1 + 1).min(nx - 1);
                let ey0 = wy0.saturating_sub(1);
                let ey1 = (wy1 + 1).min(ny - 1);
                let (hx0, hy0, snx, sny, sub_out) = self
                    .net3d
                    .forward_3d_subregion(&norm_in, nx, ny, nz, ex0, ex1, ey0, ey1);

                for iy in wy0..=wy1 {
                    for ix in wx0..=wx1 {
                        if visited[iy * nx + ix] || !active[iy * nx + ix] {
                            continue;
                        }
                        visited[iy * nx + ix] = true;
                        let h_grid = spec.axis_x.index_to_ppm(ix as f64);
                        let n_grid = spec.axis_y.index_to_ppm(iy as f64);
                        if !cfg.hsqc_roots.iter().any(|r| {
                            (r.h - h_grid).abs() <= cfg.strip_tol_h
                                && (r.n - n_grid).abs() <= cfg.strip_tol_n
                        }) {
                            continue;
                        }
                        let lx = ix - hx0;
                        let ly = iy - hy0;
                        for iz in 1..nz - 1 {
                            let sv0 = (iz * sny + ly) * snx + lx;
                            let gv0 = (iz * ny + iy) * nx + ix;
                            let base0 = sv0 * CONV3D_OUT_CH;

                            for &(ch, sign) in &[(0usize, 1i32), (1usize, -1i32)] {
                                if sign < 0 && !allow_neg {
                                    continue;
                                }
                                let prob = sub_out[base0 + ch];
                                if prob < eff_prob_thresh {
                                    continue;
                                }
                                let z_val = norm_in[gv0];
                                if (sign > 0 && z_val < cfg.min_snr_norm)
                                    || (sign < 0 && z_val > -cfg.min_snr_norm)
                                {
                                    continue;
                                }
                                let mut is_max = true;
                                'nms_s: for dz in -1isize..=1 {
                                    for dy in -1isize..=1 {
                                        for dx in -1isize..=1 {
                                            if dx == 0 && dy == 0 && dz == 0 {
                                                continue;
                                            }
                                            let sv1 = (((iz as isize + dz) as usize) * sny
                                                + ((ly as isize + dy) as usize))
                                                * snx
                                                + ((lx as isize + dx) as usize);
                                            let gv1 = (((iz as isize + dz) as usize) * ny
                                                + ((iy as isize + dy) as usize))
                                                * nx
                                                + ((ix as isize + dx) as usize);
                                            let np = sub_out[sv1 * CONV3D_OUT_CH + ch];
                                            if np > prob || (np == prob && gv1 < gv0) {
                                                is_max = false;
                                                break 'nms_s;
                                            }
                                        }
                                    }
                                }
                                if !is_max {
                                    continue;
                                }
                                let dx = sub_out[base0 + 2] as f64;
                                let dy = sub_out[base0 + 3] as f64;
                                let dz = sub_out[base0 + 4] as f64;
                                let h = spec.axis_x.index_to_ppm((ix as f64) + dx);
                                let n = spec.axis_y.index_to_ppm((iy as f64) + dy);
                                let c = spec.axis_z.index_to_ppm((iz as f64) + dz);

                                let cnn_height =
                                    (sub_out[base0 + 5] * NORM_SIGMA_DIV * noise) as f64;
                                let raw_height = ((spec.get(ix, iy, iz) - med).abs()) as f64;
                                let height = (0.85 * cnn_height + 0.15 * raw_height).max(1e-6);

                                let mut vol = 0.0f64;
                                for wz in -1isize..=1 {
                                    for wy in -1isize..=1 {
                                        for wx in -1isize..=1 {
                                            let sv = spec.get(
                                                (ix as isize + wx) as usize,
                                                (iy as isize + wy) as usize,
                                                (iz as isize + wz) as usize,
                                            ) - med;
                                            if (sign > 0 && sv > 0.0) || (sign < 0 && sv < 0.0) {
                                                vol += (sv.abs()) as f64;
                                            }
                                        }
                                    }
                                }
                                indexed_peaks.push((
                                    (iz, iy, ix, -sign),
                                    Peak {
                                        exp: exp_name.to_string(),
                                        h,
                                        n,
                                        c: Some(c),
                                        sign,
                                        height: Some(height),
                                        volume: Some(vol.max(height)),
                                        h2: None,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            indexed_peaks.sort_by_key(|a| a.0);
            return indexed_peaks.into_iter().map(|(_, p)| p).collect();
        }

        // Full blind 3D volume execution
        let out = self.net3d.forward_3d(&norm_in, nx, ny, nz);
        extract_3d_peaks_from_dense_output(
            spec, exp_name, &norm_in, med, noise, &out, cfg, allow_neg, y_is_n,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn extract_3d_peaks_from_dense_output(
    spec: &Spectrum3D,
    exp_name: &str,
    norm_in: &[f32],
    med: f32,
    noise: f32,
    out: &[f32],
    cfg: &NeuralPickerConfig,
    allow_neg: bool,
    y_is_n: bool,
) -> Vec<Peak> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    let eff_prob_thresh = cfg.prob_threshold;
    let mut peaks = Vec::new();

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            let n_grid = if y_is_n {
                spec.axis_y.index_to_ppm(iy as f64)
            } else {
                spec.axis_z.index_to_ppm(iz as f64)
            };
            if !cfg.hsqc_roots.is_empty()
                && !cfg
                    .hsqc_roots
                    .iter()
                    .any(|r| (r.n - n_grid).abs() <= cfg.strip_tol_n * 1.5)
            {
                continue;
            }
            for ix in 1..nx - 1 {
                if !cfg.hsqc_roots.is_empty() {
                    let h_grid = spec.axis_x.index_to_ppm(ix as f64);
                    if !cfg.hsqc_roots.iter().any(|r| {
                        (r.h - h_grid).abs() <= cfg.strip_tol_h
                            && (r.n - n_grid).abs() <= cfg.strip_tol_n
                    }) {
                        continue;
                    }
                }
                let v0 = (iz * ny + iy) * nx + ix;
                let base0 = v0 * CONV3D_OUT_CH;

                for &(ch, sign) in &[(0usize, 1i32), (1usize, -1i32)] {
                    if sign < 0 && !allow_neg {
                        continue;
                    }
                    let prob = out[base0 + ch];
                    if prob < eff_prob_thresh {
                        continue;
                    }
                    let z_val = norm_in[v0];
                    if (sign > 0 && z_val < cfg.min_snr_norm)
                        || (sign < 0 && z_val > -cfg.min_snr_norm)
                    {
                        continue;
                    }
                    let mut is_max = true;
                    'nms3: for dz in -1isize..=1 {
                        for dy in -1isize..=1 {
                            for dx in -1isize..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let v1 = (((iz as isize + dz) as usize) * ny
                                    + ((iy as isize + dy) as usize))
                                    * nx
                                    + ((ix as isize + dx) as usize);
                                let np = out[v1 * CONV3D_OUT_CH + ch];
                                if np > prob || (np == prob && v1 < v0) {
                                    is_max = false;
                                    break 'nms3;
                                }
                            }
                        }
                    }
                    if !is_max {
                        continue;
                    }

                    let dx = out[base0 + 2] as f64;
                    let dy = out[base0 + 3] as f64;
                    let dz = out[base0 + 4] as f64;
                    let h = spec.axis_x.index_to_ppm((ix as f64) + dx);
                    let (n, c) = if y_is_n {
                        (
                            spec.axis_y.index_to_ppm((iy as f64) + dy),
                            spec.axis_z.index_to_ppm((iz as f64) + dz),
                        )
                    } else {
                        (
                            spec.axis_z.index_to_ppm((iz as f64) + dz),
                            spec.axis_y.index_to_ppm((iy as f64) + dy),
                        )
                    };

                    let cnn_height = (out[base0 + 5] * NORM_SIGMA_DIV * noise) as f64;
                    let raw_height = ((spec.get(ix, iy, iz) - med).abs()) as f64;
                    let height = (0.85 * cnn_height + 0.15 * raw_height).max(1e-6);

                    let mut vol = 0.0f64;
                    for wz in -1isize..=1 {
                        for wy in -1isize..=1 {
                            for wx in -1isize..=1 {
                                let sv = spec.get(
                                    (ix as isize + wx) as usize,
                                    (iy as isize + wy) as usize,
                                    (iz as isize + wz) as usize,
                                ) - med;
                                if (sign > 0 && sv > 0.0) || (sign < 0 && sv < 0.0) {
                                    vol += (sv.abs()) as f64;
                                }
                            }
                        }
                    }
                    peaks.push(Peak {
                        exp: exp_name.to_string(),
                        h,
                        n,
                        c: Some(c),
                        sign,
                        height: Some(height),
                        volume: Some(vol.max(height)),
                        h2: None,
                    });
                }
            }
        }
    }
    peaks
}

/// Pick 2D `15N-HSQC` peaks using the pretrained separable 2D neural network.
pub fn pick_hsqc_2d_neural(spec: &Spectrum2D, cfg: &NeuralPickerConfig) -> Vec<Peak> {
    NeuralPicker::pretrained().pick_2d(spec, cfg)
}

/// Pick 3D triple-resonance peaks (`HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`, `HNCO`, `HNCACO`)
/// using the pretrained separable 3D neural network.
pub fn pick_triple_3d_neural(
    spec: &Spectrum3D,
    experiment: &str,
    cfg: &NeuralPickerConfig,
) -> Vec<Peak> {
    NeuralPicker::pretrained().pick_3d(spec, experiment, cfg)
}

/// Convenience entry point matching `picker::pick_triple_3d` ergonomics with explicit `hsqc_roots`.
pub fn pick_triple_3d_neural_with_roots(
    spec: &Spectrum3D,
    experiment: &str,
    hsqc_roots: &[Peak],
    cfg: &NeuralPickerConfig,
) -> Vec<Peak> {
    let mut cfg_r = cfg.clone();
    cfg_r.hsqc_roots = hsqc_roots.to_vec();
    NeuralPicker::pretrained().pick_3d(spec, experiment, &cfg_r)
}

// ============================================================================
// Blind End-to-End Spectra -> Picked Peaks -> Assignment Benchmark
// ============================================================================

#[derive(Clone, Debug)]
pub struct PeakEvalMetrics {
    pub experiment: String,
    pub true_peaks: usize,
    pub picked_peaks: usize,
    pub matched_tp: usize,
    pub precision: f64,
    pub recall: f64,
    pub fdr: f64,
    pub f1: f64,
    pub mean_pos_err_ppm: f64,
}

#[derive(Clone, Debug)]
pub struct PipelineEvalSummary {
    pub picker_name: String,
    pub per_experiment: Vec<PeakEvalMetrics>,
    pub overall_peaks: PeakEvalMetrics,
    pub assignable_residues: usize,
    pub assigned_residues: usize,
    pub correct_residues: usize,
    pub assignment_accuracy: f64,
    pub assignment_recall: f64,
    pub hc_assigned: usize,
    pub hc_correct: usize,
    pub hc_precision: f64,
    pub hc_recall: f64,
    pub shift_agreement: BTreeMap<String, (usize, usize)>,
    pub runtime_ms: f64,
}

#[derive(Clone, Debug)]
pub struct E2eBenchmarkConfig {
    pub nx_h_2d: usize,
    pub ny_n_2d: usize,
    pub nx_h_3d: usize,
    pub ny_n_3d: usize,
    pub nz_c_3d: usize,
    /// Thermal noise standard deviation relative to `100.0` `1J` peak amplitude.
    pub thermal_noise_2d: f32,
    pub thermal_noise_3d: f32,
    /// Indirect-dimension `t1` ridge fraction of intense amide roots.
    pub ridge_rel_amp: f32,
    /// Smooth sinusoidal baseline roll amplitude in noise sigma units.
    pub baseline_roll_sigma: f32,
    pub classical_snr: f32,
    pub neural_prob: f32,
    pub anneal_runs: usize,
    pub anneal_steps: usize,
    pub seed: u64,
}

impl Default for E2eBenchmarkConfig {
    fn default() -> Self {
        Self {
            nx_h_2d: 512,
            ny_n_2d: 256,
            nx_h_3d: 320,
            ny_n_3d: 128,
            nz_c_3d: 192,
            thermal_noise_2d: 8.0,
            thermal_noise_3d: 8.2,
            ridge_rel_amp: 0.16,
            baseline_roll_sigma: 2.4,
            classical_snr: 5.0,
            neural_prob: 0.455,
            anneal_runs: 16,
            anneal_steps: 1800,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct E2eBenchmarkComparison {
    pub classical: PipelineEvalSummary,
    pub neural: PipelineEvalSummary,
}

impl E2eBenchmarkComparison {
    pub fn format_table(&self) -> String {
        let c = &self.classical;
        let n = &self.neural;
        format!(
            "| Metric | Classical (`picker`) | Neural (`neural_picker`) | Delta |\n\
             |---|---:|---:|---:|\n\
             | Picked Peaks (Total) | {} / {} true | {} / {} true | {:+} |\n\
             | Peak Precision | {:.1}% | {:.1}% | {:+.1}% |\n\
             | Peak Recall | {:.1}% | {:.1}% | {:+.1}% |\n\
             | Peak False Discovery Rate (FDR) | {:.1}% | {:.1}% | {:+.1}% |\n\
             | **Peak F1** | **{:.3}** | **{:.3}** | **{:+.3}** |\n\
             | Mean Coord Error (ppm) | {:.4} | {:.4} | {:+.4} |\n\
             | Downstream Assigned / Assignable | {} / {} | {} / {} | {:+} |\n\
             | **Assignment Accuracy** (`correct/assigned`) | **{:.1}%** | **{:.1}%** | **{:+.1}%** |\n\
             | **Assignment Recall** (`correct/assignable`) | **{:.1}%** | **{:.1}%** | **{:+.1}%** |\n\
             | **High-Conf (`>=0.8`) Precision** | **{:.1}%** ({}/{}) | **{:.1}%** ({}/{}) | **{:+.1}%** |\n\
             | Pipeline Runtime (ms) | {:.1} ms | {:.1} ms | {:+.1} ms |",
            c.overall_peaks.picked_peaks,
            c.overall_peaks.true_peaks,
            n.overall_peaks.picked_peaks,
            n.overall_peaks.true_peaks,
            (n.overall_peaks.picked_peaks as isize) - (c.overall_peaks.picked_peaks as isize),
            c.overall_peaks.precision * 100.0,
            n.overall_peaks.precision * 100.0,
            (n.overall_peaks.precision - c.overall_peaks.precision) * 100.0,
            c.overall_peaks.recall * 100.0,
            n.overall_peaks.recall * 100.0,
            (n.overall_peaks.recall - c.overall_peaks.recall) * 100.0,
            c.overall_peaks.fdr * 100.0,
            n.overall_peaks.fdr * 100.0,
            (n.overall_peaks.fdr - c.overall_peaks.fdr) * 100.0,
            c.overall_peaks.f1,
            n.overall_peaks.f1,
            n.overall_peaks.f1 - c.overall_peaks.f1,
            c.overall_peaks.mean_pos_err_ppm,
            n.overall_peaks.mean_pos_err_ppm,
            n.overall_peaks.mean_pos_err_ppm - c.overall_peaks.mean_pos_err_ppm,
            c.assigned_residues,
            c.assignable_residues,
            n.assigned_residues,
            n.assignable_residues,
            (n.assigned_residues as isize) - (c.assigned_residues as isize),
            c.assignment_accuracy * 100.0,
            n.assignment_accuracy * 100.0,
            (n.assignment_accuracy - c.assignment_accuracy) * 100.0,
            c.assignment_recall * 100.0,
            n.assignment_recall * 100.0,
            (n.assignment_recall - c.assignment_recall) * 100.0,
            c.hc_precision * 100.0,
            c.hc_correct,
            c.hc_assigned,
            n.hc_precision * 100.0,
            n.hc_correct,
            n.hc_assigned,
            (n.hc_precision - c.hc_precision) * 100.0,
            c.runtime_ms,
            n.runtime_ms,
            n.runtime_ms - c.runtime_ms,
        )
    }
}

/// Coalesce physically superposed identical-sign peaks within 0.65 linewidth so ground-truth
/// peak counting reflects physically distinct extrema in the rendered spectrum.
fn coalesce_observable_peaks(peaks: &[Peak], tol_h: f64, tol_n: f64, tol_c: f64) -> Vec<Peak> {
    let mut out: Vec<Peak> = Vec::new();
    for p in peaks {
        if let Some(existing) = out.iter_mut().find(|q| {
            q.exp == p.exp
                && q.sign == p.sign
                && (q.h - p.h).abs() <= tol_h
                && (q.n - p.n).abs() <= tol_n
                && match (q.c, p.c) {
                    (Some(ca), Some(cb)) => (ca - cb).abs() <= tol_c,
                    (None, None) => true,
                    _ => false,
                }
        }) {
            let w1 = existing.height.unwrap_or(100.0);
            let w2 = p.height.unwrap_or(100.0);
            let wt = (w1 + w2).max(1e-6);
            existing.h = (existing.h * w1 + p.h * w2) / wt;
            existing.n = (existing.n * w1 + p.n * w2) / wt;
            if let (Some(ca), Some(cb)) = (existing.c, p.c) {
                existing.c = Some((ca * w1 + cb * w2) / wt);
            }
            existing.height = Some(w1 + w2);
        } else {
            out.push(p.clone());
        }
    }
    out
}

/// Match picked peaks greedily 1-to-1 against observable ground-truth peaks for one experiment.
pub fn evaluate_picked_peaks(exp: &str, picked: &[Peak], truth: &[Peak]) -> PeakEvalMetrics {
    let tol_h = 0.04;
    let tol_n = 0.40;
    let tol_c = if exp == "HNCO" || exp == "HNCACO" {
        0.28
    } else {
        0.42
    };

    let mut pairs: Vec<(f64, f64, usize, usize)> = Vec::new();
    for (pi, p) in picked.iter().enumerate() {
        for (ti, t) in truth.iter().enumerate() {
            if p.sign != t.sign {
                continue;
            }
            let dh = (p.h - t.h).abs();
            let dn = (p.n - t.n).abs();
            if dh > tol_h || dn > tol_n {
                continue;
            }
            let (d_norm_sq, raw_ppm_err) = match (p.c, t.c) {
                (Some(pc), Some(tc)) => {
                    let dc = (pc - tc).abs();
                    if dc > tol_c {
                        continue;
                    }
                    (
                        (dh / tol_h).powi(2) + (dn / tol_n).powi(2) + (dc / tol_c).powi(2),
                        (dh + 0.1 * dn + 0.25 * dc) / 3.0,
                    )
                }
                (None, None) => (
                    (dh / tol_h).powi(2) + (dn / tol_n).powi(2),
                    0.5 * (dh + 0.1 * dn),
                ),
                _ => continue,
            };
            pairs.push((d_norm_sq, raw_ppm_err, pi, ti));
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut used_p = vec![false; picked.len()];
    let mut used_t = vec![false; truth.len()];
    let mut tp = 0usize;
    let mut err_sum = 0.0f64;
    for &(_, raw_err, pi, ti) in &pairs {
        if !used_p[pi] && !used_t[ti] {
            used_p[pi] = true;
            used_t[ti] = true;
            tp += 1;
            err_sum += raw_err;
        }
    }

    let n_p = picked.len();
    let n_t = truth.len();
    let precision = if n_p > 0 {
        (tp as f64) / (n_p as f64)
    } else {
        0.0
    };
    let recall = if n_t > 0 {
        (tp as f64) / (n_t as f64)
    } else {
        0.0
    };
    let fdr = if n_p > 0 { 1.0 - precision } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    PeakEvalMetrics {
        experiment: exp.to_string(),
        true_peaks: n_t,
        picked_peaks: n_p,
        matched_tp: tp,
        precision,
        recall,
        fdr,
        f1,
        mean_pos_err_ppm: if tp > 0 { err_sum / (tp as f64) } else { 0.0 },
    }
}

fn render_realistic_hsqc_2d(
    peaks: &[Peak],
    nx: usize,
    ny: usize,
    noise_sigma: f32,
    ridge_rel_amp: f32,
    baseline_roll_sigma: f32,
    seed: u64,
) -> Spectrum2D {
    let axis_x = AxisPpm::from_bounds("1H", nx, 10.6, 5.8, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 136.0, 102.0, 60.81);
    let mut rng = Rng::new(seed);
    let mut data = vec![0.0f32; ny * nx];

    // Smooth baseline roll
    if baseline_roll_sigma > 0.0 {
        let roll = (baseline_roll_sigma * noise_sigma) as f64;
        for iy in 0..ny {
            let cy = (TAU * (iy as f64) / (ny as f64) + 0.6).cos();
            for ix in 0..nx {
                let cx = (1.3 * TAU * (ix as f64) / (nx as f64) + 0.3).sin();
                data[iy * nx + ix] += (roll * cx * cy) as f32;
            }
        }
    }

    let lw_h = 0.024;
    let lw_n = 0.28;
    let fwhm_x = lw_h / axis_x.step_ppm.abs();
    let fwhm_y = lw_n / axis_y.step_ppm.abs();
    let rx = (4.0 * fwhm_x).ceil() as isize + 2;
    let ry = (4.0 * fwhm_y).ceil() as isize + 2;

    let mut strong_cols: Vec<(f64, f64)> = Vec::new();
    for p in peaks.iter().filter(|p| p.exp == "HSQC") {
        let cx = axis_x.ppm_to_index(p.h);
        let cy = axis_y.ppm_to_index(p.n);
        let amp = p.height.unwrap_or(100.0) * (p.sign as f64);
        if amp.abs() > 115.0 {
            strong_cols.push((cx, amp.abs()));
        }
        let x0 = ((cx.floor() as isize) - rx).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + rx).min(nx as isize - 1).max(0) as usize;
        let y0 = ((cy.floor() as isize) - ry).max(0) as usize;
        let y1 = ((cy.ceil() as isize) + ry).min(ny as isize - 1).max(0) as usize;
        for iy in y0..=y1 {
            let wy = pseudo_voigt_1d((iy as f64) - cy, fwhm_y, 0.55);
            for ix in x0..=x1 {
                let wx = pseudo_voigt_1d((ix as f64) - cx, fwhm_x, 0.55);
                data[iy * nx + ix] += (amp * wx * wy) as f32;
            }
        }
    }

    // Add t1 (15N) indirect ridges under the strongest amide signals
    strong_cols.sort_by(|a, b| b.1.total_cmp(&a.1));
    for &(cx, amp) in strong_cols.iter().take(6) {
        let r_amp = amp * (ridge_rel_amp as f64);
        let dc = 0.35 * r_amp;
        let x0 = ((cx.floor() as isize) - 4).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + 4).min(nx as isize - 1).max(0) as usize;
        for iy in 0..ny {
            let streak = dc + rng.normal(0.0, r_amp);
            for ix in x0..=x1 {
                let wx = pseudo_voigt_1d((ix as f64) - cx, fwhm_x, 0.6);
                data[iy * nx + ix] += (streak * wx) as f32;
            }
        }
    }

    for v in &mut data {
        *v += rng.normal(0.0, noise_sigma as f64) as f32;
    }

    Spectrum2D {
        name: "Synthetic Blind 15N-HSQC".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_realistic_triple_3d(
    exp: &str,
    peaks: &[Peak],
    nx: usize,
    ny: usize,
    nz: usize,
    noise_sigma: f32,
    ridge_rel_amp: f32,
    baseline_roll_sigma: f32,
    seed: u64,
) -> Spectrum3D {
    let (c_hi, c_lo, lw_c) = if exp == "HNCO" || exp == "HNCACO" {
        (183.0, 168.0, 0.28)
    } else if exp == "HNCACB" || exp == "CBCACONH" {
        (76.0, 13.0, 0.78)
    } else {
        (70.0, 40.0, 0.45)
    };
    let axis_x = AxisPpm::from_bounds("1H", nx, 10.6, 5.8, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 136.0, 102.0, 60.81);
    let axis_z = AxisPpm::from_bounds("13C", nz, c_hi, c_lo, 150.90);

    let mut rng = Rng::new(seed);
    let mut data = vec![0.0f32; nz * ny * nx];

    if baseline_roll_sigma > 0.0 {
        let roll = (baseline_roll_sigma * noise_sigma) as f64;
        for iz in 0..nz {
            let cz = (TAU * (iz as f64) / (nz as f64) + 0.4).cos();
            for iy in 0..ny {
                let cyz = cz * (TAU * (iy as f64) / (ny as f64) + 0.2).cos();
                let base = (iz * ny + iy) * nx;
                for ix in 0..nx {
                    let cx = (1.2 * TAU * (ix as f64) / (nx as f64) + 0.5).sin();
                    data[base + ix] += (roll * cx * cyz) as f32;
                }
            }
        }
    }

    let fwhm_x = 0.030 / axis_x.step_ppm.abs();
    let fwhm_y = 0.48 / axis_y.step_ppm.abs();
    let fwhm_z = lw_c / axis_z.step_ppm.abs();
    let rx = (3.5 * fwhm_x).ceil() as isize + 2;
    let ry = (3.5 * fwhm_y).ceil() as isize + 2;
    let rz = (3.5 * fwhm_z).ceil() as isize + 2;

    let mut strong_strips: Vec<(f64, f64, f64)> = Vec::new();
    for p in peaks.iter().filter(|p| p.exp == exp) {
        let Some(c_ppm) = p.c else { continue };
        let cx = axis_x.ppm_to_index(p.h);
        let cy = axis_y.ppm_to_index(p.n);
        let cz = axis_z.ppm_to_index(c_ppm);
        let amp = p.height.unwrap_or(90.0) * (p.sign as f64);
        if amp.abs() > 105.0 {
            strong_strips.push((cx, cy, amp.abs()));
        }

        let x0 = ((cx.floor() as isize) - rx).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + rx).min(nx as isize - 1).max(0) as usize;
        let y0 = ((cy.floor() as isize) - ry).max(0) as usize;
        let y1 = ((cy.ceil() as isize) + ry).min(ny as isize - 1).max(0) as usize;
        let z0 = ((cz.floor() as isize) - rz).max(0) as usize;
        let z1 = ((cz.ceil() as isize) + rz).min(nz as isize - 1).max(0) as usize;

        for iz in z0..=z1 {
            let wz = pseudo_voigt_1d((iz as f64) - cz, fwhm_z, 0.55);
            for iy in y0..=y1 {
                let wyz = wz * pseudo_voigt_1d((iy as f64) - cy, fwhm_y, 0.55);
                let base = (iz * ny + iy) * nx;
                for ix in x0..=x1 {
                    let wx = pseudo_voigt_1d((ix as f64) - cx, fwhm_x, 0.55);
                    data[base + ix] += (amp * wx * wyz) as f32;
                }
            }
        }
    }

    // Add indirect 13C ridge noise along intense amide strips
    strong_strips.sort_by(|a, b| b.2.total_cmp(&a.2));
    for &(cx, cy, amp) in strong_strips.iter().take(10) {
        let r_amp = amp * (ridge_rel_amp as f64);
        let dc = 0.30 * r_amp;
        let x0 = ((cx.floor() as isize) - 3).max(0) as usize;
        let x1 = ((cx.ceil() as isize) + 3).min(nx as isize - 1).max(0) as usize;
        let y0 = ((cy.floor() as isize) - 3).max(0) as usize;
        let y1 = ((cy.ceil() as isize) + 3).min(ny as isize - 1).max(0) as usize;
        for iz in 0..nz {
            let streak = dc + rng.normal(0.0, r_amp);
            for iy in y0..=y1 {
                let wy = pseudo_voigt_1d((iy as f64) - cy, fwhm_y, 0.6);
                let base = (iz * ny + iy) * nx;
                for ix in x0..=x1 {
                    let wx = pseudo_voigt_1d((ix as f64) - cx, fwhm_x, 0.6);
                    data[base + ix] += (streak * wx * wy) as f32;
                }
            }
        }
    }

    for v in &mut data {
        *v += rng.normal(0.0, noise_sigma as f64) as f32;
    }

    Spectrum3D {
        name: format!("Synthetic Blind 3D {exp}"),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_blind_pipeline(
    picker_name: &str,
    picked_hsqc: Vec<Peak>,
    picked_3d_by_exp: &[(&str, Vec<Peak>)],
    obs_truth_by_exp: &BTreeMap<String, Vec<Peak>>,
    true_backbone_roots: &[(usize, f64, f64)],
    ref_data: &Reference,
    anneal_cfg: &AnnealConfig,
    runtime_ms: f64,
) -> Result<PipelineEvalSummary, String> {
    let mut per_exp = Vec::new();
    let mut tot_true = 0usize;
    let mut tot_picked = 0usize;
    let mut tot_tp = 0usize;
    let mut weighted_err = 0.0f64;

    let hsqc_eval = evaluate_picked_peaks("HSQC", &picked_hsqc, &obs_truth_by_exp["HSQC"]);
    tot_true += hsqc_eval.true_peaks;
    tot_picked += hsqc_eval.picked_peaks;
    tot_tp += hsqc_eval.matched_tp;
    weighted_err += hsqc_eval.mean_pos_err_ppm * (hsqc_eval.matched_tp as f64);
    per_exp.push(hsqc_eval);

    let mut all_peaks = picked_hsqc.clone();
    for (exp, pks) in picked_3d_by_exp {
        let m = evaluate_picked_peaks(exp, pks, &obs_truth_by_exp[*exp]);
        tot_true += m.true_peaks;
        tot_picked += m.picked_peaks;
        tot_tp += m.matched_tp;
        weighted_err += m.mean_pos_err_ppm * (m.matched_tp as f64);
        per_exp.push(m);
        all_peaks.extend(pks.iter().cloned());
    }

    let overall_prec = if tot_picked > 0 {
        (tot_tp as f64) / (tot_picked as f64)
    } else {
        0.0
    };
    let overall_rec = if tot_true > 0 {
        (tot_tp as f64) / (tot_true as f64)
    } else {
        0.0
    };
    let overall_peaks = PeakEvalMetrics {
        experiment: "ALL".to_string(),
        true_peaks: tot_true,
        picked_peaks: tot_picked,
        matched_tp: tot_tp,
        precision: overall_prec,
        recall: overall_rec,
        fdr: if tot_picked > 0 {
            1.0 - overall_prec
        } else {
            0.0
        },
        f1: if overall_prec + overall_rec > 0.0 {
            2.0 * overall_prec * overall_rec / (overall_prec + overall_rec)
        } else {
            0.0
        },
        mean_pos_err_ppm: if tot_tp > 0 {
            weighted_err / (tot_tp as f64)
        } else {
            0.0
        },
    };

    // Match picked HSQC roots 1-to-1 to true backbone amide residues for ground-truth evaluation
    let mut root_pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (si, p) in picked_hsqc.iter().enumerate() {
        for (ti, &(_, th, tn)) in true_backbone_roots.iter().enumerate() {
            let dh = (p.h - th).abs();
            let dn = (p.n - tn).abs();
            if dh <= 0.045 && dn <= 0.45 {
                let d2 = (dh / 0.045).powi(2) + (dn / 0.45).powi(2);
                root_pairs.push((d2, si, ti));
            }
        }
    }
    root_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_s = vec![false; picked_hsqc.len()];
    let mut used_t = vec![false; true_backbone_roots.len()];
    let mut root_truth = vec![-1i32; picked_hsqc.len()];
    for &(_, si, ti) in &root_pairs {
        if !used_s[si] && !used_t[ti] {
            used_s[si] = true;
            used_t[ti] = true;
            root_truth[si] = true_backbone_roots[ti].0 as i32;
        }
    }

    let assignable_residues = true_backbone_roots.len();
    if picked_hsqc.is_empty() {
        return Ok(PipelineEvalSummary {
            picker_name: picker_name.to_string(),
            per_experiment: per_exp,
            overall_peaks,
            assignable_residues,
            assigned_residues: 0,
            correct_residues: 0,
            assignment_accuracy: 0.0,
            assignment_recall: 0.0,
            hc_assigned: 0,
            hc_correct: 0,
            hc_precision: 0.0,
            hc_recall: 0.0,
            shift_agreement: BTreeMap::new(),
            runtime_ms,
        });
    }

    let systems = build_spin_systems(&all_peaks, &Tolerances::default(), 3, false)?;
    let priors = Priors::default();
    let res = assign(
        &systems,
        &ref_data.sequence,
        &priors,
        &ScoreConfig::default(),
        anneal_cfg,
    );

    let mut assigned = 0usize;
    let mut correct = 0usize;
    let mut hc_assigned = 0usize;
    let mut hc_correct = 0usize;
    for (s, &pos) in res.position.iter().enumerate() {
        if pos >= 0 {
            assigned += 1;
            let ok = pos == root_truth[s];
            if ok {
                correct += 1;
            }
            if res.confidence[s] >= 0.8 {
                hc_assigned += 1;
                if ok {
                    hc_correct += 1;
                }
            }
        }
    }

    let (pred_ref, _) = to_reference(
        &res,
        &systems,
        &ref_data.sequence,
        &ref_data.seq_ids,
        0.0,
        Some(&priors),
    );
    let shift_agreement = compare_shifts(&pred_ref, ref_data);

    Ok(PipelineEvalSummary {
        picker_name: picker_name.to_string(),
        per_experiment: per_exp,
        overall_peaks,
        assignable_residues,
        assigned_residues: assigned,
        correct_residues: correct,
        assignment_accuracy: if assigned > 0 {
            (correct as f64) / (assigned as f64)
        } else {
            0.0
        },
        assignment_recall: if assignable_residues > 0 {
            (correct as f64) / (assignable_residues as f64)
        } else {
            0.0
        },
        hc_assigned,
        hc_correct,
        hc_precision: if hc_assigned > 0 {
            (hc_correct as f64) / (hc_assigned as f64)
        } else {
            0.0
        },
        hc_recall: if assignable_residues > 0 {
            (hc_correct as f64) / (assignable_residues as f64)
        } else {
            0.0
        },
        shift_agreement,
        runtime_ms,
    })
}

/// Synthesize noisy 2D `HSQC` + 3D `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`, `HNCO` volumetric grids
/// from `ref_data` (with realistic `1J`/`2J` transfer intensities, thermal noise, baseline roll,
/// and `t1` ridges), pick peaks blindly with both classical `picker` and `neural_picker`, assemble
/// spin systems, run the `nmr-core` solver, and compare peak F1 & downstream assignment accuracy.
pub fn run_e2e_blind_benchmark(
    ref_data: &Reference,
    cfg: &E2eBenchmarkConfig,
) -> Result<E2eBenchmarkComparison, String> {
    let mut rng = Rng::new(cfg.seed);
    let triple_exps = ["HNCA", "HNCOCA", "HNCACB", "CBCACONH", "HNCO"];
    let mut raw_peaks: Vec<Peak> = Vec::new();
    let mut true_backbone_roots: Vec<(usize, f64, f64)> = Vec::new();

    for (k, res) in ref_data.sequence.iter().enumerate() {
        if k == 0 || res == "PRO" {
            continue;
        }
        let Some(&h_val) = ref_data.shifts.get(&(k, "H".to_string())) else {
            continue;
        };
        let Some(&n_val) = ref_data.shifts.get(&(k, "N".to_string())) else {
            continue;
        };
        true_backbone_roots.push((k, h_val, n_val));

        let root_ht = 110.0 * (rng.normal(0.0, 0.16)).exp();
        raw_peaks.push(Peak::hsqc(h_val, n_val).with_intensity(root_ht, Some(root_ht * 4.2)));

        for &exp in &triple_exps {
            let Some(rules) = experiment_rules(exp) else {
                continue;
            };
            let has_both = rules.iter().any(|r| r.1 == 0) && rules.iter().any(|r| r.1 == -1);
            for &(atom, off, sign) in rules {
                let target = (k as isize) + (off as isize);
                if target < 0 {
                    continue;
                }
                let Some(&c_val) = ref_data.shifts.get(&(target as usize, atom.to_string())) else {
                    continue;
                };
                // Realistic transfer efficiency: 2J sequential (i-1) is ~0.43x of 1J intraresidue (i)
                let base_amp = match (exp, atom, has_both, off) {
                    ("HNCACB", "CB", true, -1) => 39.0,
                    ("HNCACB", "CA", true, -1) => 43.0,
                    ("HNCACB", "CB", true, 0) => 82.0,
                    ("HNCACB", "CA", true, 0) => 96.0,
                    (_, _, true, -1) => 44.0,
                    (_, _, true, 0) => 98.0,
                    _ => 86.0,
                };
                let ht = base_amp * (rng.normal(0.0, 0.14)).exp();
                raw_peaks.push(
                    Peak::triple(exp, h_val, n_val, c_val, sign).with_intensity(ht, Some(ht * 5.8)),
                );
            }
        }
    }

    // Build observable ground-truth peak sets per experiment
    let mut obs_truth_by_exp: BTreeMap<String, Vec<Peak>> = BTreeMap::new();
    let hsqc_raw: Vec<Peak> = raw_peaks
        .iter()
        .filter(|p| p.exp == "HSQC")
        .cloned()
        .collect();
    obs_truth_by_exp.insert(
        "HSQC".to_string(),
        coalesce_observable_peaks(&hsqc_raw, 0.015, 0.18, 0.0),
    );
    for &exp in &triple_exps {
        let exp_raw: Vec<Peak> = raw_peaks.iter().filter(|p| p.exp == exp).cloned().collect();
        let c_coal = if exp == "HNCO" { 0.18 } else { 0.45 };
        obs_truth_by_exp.insert(
            exp.to_string(),
            coalesce_observable_peaks(&exp_raw, 0.020, 0.28, c_coal),
        );
    }

    // Render volumetric 2D HSQC + 5 3D cubes
    let hsqc_spec = render_realistic_hsqc_2d(
        &raw_peaks,
        cfg.nx_h_2d,
        cfg.ny_n_2d,
        cfg.thermal_noise_2d,
        cfg.ridge_rel_amp,
        cfg.baseline_roll_sigma,
        cfg.seed.wrapping_add(10),
    );
    let mut specs_3d: Vec<(&str, Spectrum3D)> = Vec::with_capacity(triple_exps.len());
    for (idx, &exp) in triple_exps.iter().enumerate() {
        let nz = if exp == "HNCO" {
            96
        } else if exp == "HNCACB" || exp == "CBCACONH" {
            cfg.nz_c_3d
        } else {
            128
        };
        specs_3d.push((
            exp,
            render_realistic_triple_3d(
                exp,
                &raw_peaks,
                cfg.nx_h_3d,
                cfg.ny_n_3d,
                nz,
                cfg.thermal_noise_3d,
                cfg.ridge_rel_amp,
                cfg.baseline_roll_sigma,
                cfg.seed.wrapping_add(100 + idx as u64),
            ),
        ));
    }

    let anneal_cfg = AnnealConfig {
        runs: cfg.anneal_runs,
        steps_per_system: cfg.anneal_steps,
        seed: cfg.seed,
        ..AnnealConfig::default()
    };

    // 1. Classical Picker + Solver
    let t_cls = Instant::now();
    let cls_cfg = PickerConfig {
        snr_threshold: cfg.classical_snr,
        ..PickerConfig::default()
    };
    let cls_hsqc = pick_hsqc_2d(&hsqc_spec, &cls_cfg);
    let mut cls_3d = Vec::with_capacity(specs_3d.len());
    for (exp, s3) in &specs_3d {
        cls_3d.push((*exp, pick_triple_3d(s3, &cls_hsqc, &cls_cfg)));
    }
    let classical = evaluate_blind_pipeline(
        "Classical",
        cls_hsqc,
        &cls_3d,
        &obs_truth_by_exp,
        &true_backbone_roots,
        ref_data,
        &anneal_cfg,
        t_cls.elapsed().as_secs_f64() * 1000.0,
    )?;

    // 2. Neural Picker + Solver
    let neural_model = NeuralPicker::pretrained();
    let t_neu = Instant::now();
    let neu_cfg_2d = NeuralPickerConfig {
        prob_threshold: cfg.neural_prob,
        ..NeuralPickerConfig::default()
    };
    let neu_hsqc = neural_model.pick_2d(&hsqc_spec, &neu_cfg_2d);
    let neu_cfg_3d = NeuralPickerConfig {
        prob_threshold: cfg.neural_prob,
        hsqc_roots: neu_hsqc.clone(),
        ..NeuralPickerConfig::default()
    };
    let mut neu_3d = Vec::with_capacity(specs_3d.len());
    for (exp, s3) in &specs_3d {
        neu_3d.push((*exp, neural_model.pick_3d(s3, exp, &neu_cfg_3d)));
    }
    let neural = evaluate_blind_pipeline(
        "Neural",
        neu_hsqc,
        &neu_3d,
        &obs_truth_by_exp,
        &true_backbone_roots,
        ref_data,
        &anneal_cfg,
        t_neu.elapsed().as_secs_f64() * 1000.0,
    )?;

    Ok(E2eBenchmarkComparison { classical, neural })
}

// ============================================================================
// Embedded Pretrained Weights (populated by clean-room in-Rust Adam trainer)
// ============================================================================

const PRETRAINED_WEIGHTS_2D_READY: bool = true;
const PRETRAINED_WEIGHTS_3D_READY: bool = true;

#[allow(clippy::excessive_precision)]
pub const PRETRAINED_WEIGHTS_2D: [f32; CONV2D_NUM_WEIGHTS] = [
    -0.2146277_f32,
    0.1948847_f32,
    0.7323077_f32,
    0.0783146_f32,
    -0.3126267_f32,
    -0.3114331_f32,
    0.6041722_f32,
    0.2593740_f32,
    0.4867543_f32,
    -0.2114435_f32,
    -0.2759137_f32,
    0.0907911_f32,
    0.4945682_f32,
    -0.2612922_f32,
    -0.1083304_f32,
    -0.1738271_f32,
    0.4826044_f32,
    0.2553663_f32,
    0.5486093_f32,
    -0.1836446_f32,
    -0.0842912_f32,
    -0.7813724_f32,
    0.1130893_f32,
    0.8433690_f32,
    -0.0328963_f32,
    -0.1565378_f32,
    0.4891901_f32,
    0.9732361_f32,
    0.3599181_f32,
    -0.2200564_f32,
    0.1201186_f32,
    0.3919755_f32,
    0.8826428_f32,
    0.5075855_f32,
    0.1765913_f32,
    -0.2738461_f32,
    0.0782821_f32,
    0.6002045_f32,
    0.2064383_f32,
    -0.2328384_f32,
    0.0134399_f32,
    0.0750887_f32,
    0.3388617_f32,
    0.1311210_f32,
    0.1380321_f32,
    -0.3902813_f32,
    0.0019049_f32,
    0.6364166_f32,
    0.0677333_f32,
    -0.2799860_f32,
    0.1736157_f32,
    0.1379636_f32,
    0.4690105_f32,
    0.0726570_f32,
    0.2254670_f32,
    -0.3418913_f32,
    0.4405755_f32,
    0.1655529_f32,
    0.5289069_f32,
    -0.1722380_f32,
    -0.0005677_f32,
    -0.5934600_f32,
    0.2633798_f32,
    0.4989794_f32,
    0.2281159_f32,
    -0.2021999_f32,
    0.3608699_f32,
    0.6842986_f32,
    0.3965389_f32,
    -0.1456621_f32,
    -0.2278971_f32,
    0.1721412_f32,
    0.7656280_f32,
    0.1710946_f32,
    -0.2988094_f32,
    -0.2420121_f32,
    0.4071423_f32,
    0.1994149_f32,
    0.3758734_f32,
    -0.3576963_f32,
    -0.2346678_f32,
    0.1817639_f32,
    0.5636820_f32,
    0.2318683_f32,
    -0.1832155_f32,
    -0.3379844_f32,
    0.4559376_f32,
    0.3137151_f32,
    0.4604347_f32,
    -0.4620122_f32,
    0.0720965_f32,
    0.4693735_f32,
    0.2248601_f32,
    0.6617651_f32,
    -0.0306246_f32,
    -0.1668683_f32,
    0.0920008_f32,
    0.4753403_f32,
    0.1762978_f32,
    -0.3112275_f32,
    -0.0653323_f32,
    0.4693933_f32,
    0.9692724_f32,
    0.3333306_f32,
    -0.1528178_f32,
    -0.1657626_f32,
    -0.8974739_f32,
    -0.1668168_f32,
    0.7677419_f32,
    0.1870480_f32,
    -0.3279592_f32,
    0.0489466_f32,
    0.5069447_f32,
    0.3666048_f32,
    -0.4278699_f32,
    0.1983274_f32,
    0.4532893_f32,
    0.8500383_f32,
    0.6533948_f32,
    0.2527737_f32,
    0.0845218_f32,
    0.3145253_f32,
    0.3602846_f32,
    0.1975683_f32,
    -0.0503851_f32,
    0.3152286_f32,
    0.3711959_f32,
    0.4222503_f32,
    0.2714787_f32,
    0.1579732_f32,
    -0.5741391_f32,
    0.0001064_f32,
    0.4817956_f32,
    0.1871126_f32,
    -0.3709290_f32,
    0.0329826_f32,
    0.1421247_f32,
    0.2019077_f32,
    -0.0102419_f32,
    0.0496252_f32,
    0.1566594_f32,
    0.5393243_f32,
    0.3287773_f32,
    0.4937411_f32,
    -0.0173617_f32,
    0.2000101_f32,
    -0.3520860_f32,
    -0.3894611_f32,
    0.4592337_f32,
    0.4777776_f32,
    -0.5763918_f32,
    0.1438929_f32,
    0.6809527_f32,
    0.1358456_f32,
    -0.3639783_f32,
    -0.3124095_f32,
    0.4627777_f32,
    -0.1166645_f32,
    0.3729329_f32,
    -0.2022273_f32,
    1.0712966_f32,
    -0.1829918_f32,
    0.2990588_f32,
    -0.0610357_f32,
    -0.2064049_f32,
    0.2791505_f32,
    -0.0649153_f32,
    0.1650106_f32,
    -0.2008512_f32,
    0.1595080_f32,
    0.2397721_f32,
    -0.0174362_f32,
    -0.0574076_f32,
    0.4676085_f32,
    0.0697709_f32,
    -0.0023418_f32,
    -0.9214345_f32,
    0.3728228_f32,
    -0.2503293_f32,
    0.2455467_f32,
    0.5998048_f32,
    -0.4761130_f32,
    0.1588697_f32,
    -0.4364184_f32,
    0.2718149_f32,
    -0.0082701_f32,
    0.0181143_f32,
    0.0147462_f32,
    -0.0948586_f32,
    0.2515706_f32,
    0.2403224_f32,
    0.0530153_f32,
    0.0805464_f32,
    0.6672501_f32,
    -0.2729866_f32,
    0.1174940_f32,
    0.5434281_f32,
    -0.1652041_f32,
    0.6251859_f32,
    -0.1056742_f32,
    -0.3357079_f32,
    -0.3253934_f32,
    0.4496128_f32,
    -0.0378856_f32,
    -0.3076421_f32,
    -0.2267510_f32,
    0.5703098_f32,
    -0.2785186_f32,
    -0.2758358_f32,
    -0.7416850_f32,
    -0.4798740_f32,
    -0.2880633_f32,
    0.5406829_f32,
    0.2254844_f32,
    -0.1895158_f32,
    -0.3865466_f32,
    -0.0402444_f32,
    -0.4627867_f32,
    -0.1656133_f32,
    -0.3008384_f32,
    0.0885587_f32,
    0.2643234_f32,
    0.0752010_f32,
    0.2757024_f32,
    -0.3514659_f32,
    -0.0045698_f32,
    0.1032072_f32,
    -0.0973945_f32,
    0.3060010_f32,
    0.3151793_f32,
    -0.2676326_f32,
    -0.2663818_f32,
    -0.4476368_f32,
    -0.1685927_f32,
    0.0837804_f32,
    -0.2740333_f32,
    -0.0517543_f32,
    0.3105326_f32,
    -0.4932445_f32,
    0.0593956_f32,
    0.0733739_f32,
    0.1843609_f32,
    -0.7574058_f32,
    -0.2006990_f32,
    0.9871190_f32,
    -0.0059359_f32,
    0.1517770_f32,
    0.1013237_f32,
    -0.2113503_f32,
    0.1109528_f32,
    -0.5659032_f32,
    -0.0092579_f32,
    -0.1511132_f32,
    0.4440317_f32,
    -0.4629427_f32,
    -0.3228692_f32,
    -0.7331061_f32,
    -0.1413799_f32,
    -0.0065128_f32,
    0.5280128_f32,
    -0.4476948_f32,
    0.7294081_f32,
    0.0924322_f32,
    -0.4018382_f32,
    -0.1325589_f32,
    -0.1081440_f32,
    -0.2044090_f32,
    -0.1183696_f32,
    0.1707261_f32,
    0.0376167_f32,
    -0.0568408_f32,
    0.5247779_f32,
    -0.0439216_f32,
    0.1284592_f32,
    -0.2002333_f32,
    -1.1225898_f32,
    -0.5387912_f32,
    0.8159195_f32,
    -0.1067660_f32,
    -0.4627225_f32,
    0.2240128_f32,
    -0.1806615_f32,
    -0.3180066_f32,
    -0.2411349_f32,
    0.2031887_f32,
    0.0534138_f32,
    -0.4929486_f32,
    -0.0178485_f32,
    -0.1321874_f32,
    -0.0492824_f32,
    -0.0558971_f32,
    -0.0521039_f32,
    0.7514173_f32,
    0.4118249_f32,
    0.0919885_f32,
    0.1248900_f32,
    0.0488819_f32,
    0.0057365_f32,
    -0.1142916_f32,
    -0.1903341_f32,
    -0.4434960_f32,
    -0.1829799_f32,
    -0.0181317_f32,
    0.0897250_f32,
    0.3446325_f32,
    0.6064649_f32,
    0.6365182_f32,
    0.6759505_f32,
    -0.4832193_f32,
    0.6001827_f32,
    0.5959999_f32,
    0.7281485_f32,
    0.4094977_f32,
    0.4048692_f32,
    0.2581286_f32,
    0.2452670_f32,
    0.3597245_f32,
    -0.3820243_f32,
    0.4377221_f32,
    -0.0652346_f32,
    0.2397740_f32,
    -0.0907613_f32,
    -0.1289027_f32,
    -0.0342863_f32,
    -0.3972160_f32,
    1.3168485_f32,
    0.1498241_f32,
    0.2630439_f32,
    -0.2202694_f32,
    -0.0933568_f32,
    0.0003414_f32,
    0.1573588_f32,
    0.0600986_f32,
    -0.0098712_f32,
    -0.0767392_f32,
    0.0386357_f32,
    0.2364048_f32,
    0.0360745_f32,
    0.1871115_f32,
    0.4499965_f32,
    0.2358683_f32,
    -0.6838037_f32,
    0.0865083_f32,
    0.2040747_f32,
    0.0831776_f32,
    -0.0041183_f32,
    0.0383883_f32,
    0.0373325_f32,
    -0.1158590_f32,
    0.0575543_f32,
    0.3871205_f32,
    0.1657442_f32,
    -0.2287867_f32,
    -0.1933277_f32,
    0.2092062_f32,
    -0.1791161_f32,
    0.2158479_f32,
    0.1219160_f32,
    0.8973831_f32,
    0.3567778_f32,
    0.0842252_f32,
    0.1739689_f32,
    0.1068349_f32,
    0.0952528_f32,
    -0.0802474_f32,
    0.4088983_f32,
    -0.1019101_f32,
    -0.2734480_f32,
    -0.3458942_f32,
    -0.0050226_f32,
    -0.4836260_f32,
    -0.2517236_f32,
    0.2354169_f32,
    0.1334950_f32,
    -0.5128261_f32,
    -0.3864053_f32,
    -0.3376497_f32,
    -0.4878160_f32,
    -0.2517805_f32,
    0.1754701_f32,
    -0.1594248_f32,
    0.0462619_f32,
    -0.3131674_f32,
    0.2022748_f32,
    0.0172800_f32,
    -0.0715557_f32,
    -0.0237383_f32,
    -0.0492992_f32,
    0.7193154_f32,
    0.0487587_f32,
    0.2265796_f32,
    0.8509867_f32,
    0.4326213_f32,
    0.1341462_f32,
    0.3418274_f32,
    -0.1007864_f32,
    0.0431320_f32,
    0.1299205_f32,
    0.3656634_f32,
    -0.0097131_f32,
    -0.2410164_f32,
    -0.2759169_f32,
    -0.2897049_f32,
    -0.2197131_f32,
    -0.0954538_f32,
    -0.1787136_f32,
    0.1065784_f32,
    -0.9523298_f32,
    -0.0589473_f32,
    0.0367692_f32,
    -0.3811677_f32,
    -0.2402422_f32,
    0.1273800_f32,
    -0.0782573_f32,
    0.0291717_f32,
    0.3016334_f32,
    0.3884457_f32,
    0.0986787_f32,
    0.1154014_f32,
    0.1980808_f32,
    0.1354404_f32,
    -0.0905403_f32,
    0.0029736_f32,
    0.4326778_f32,
    0.7665668_f32,
    -0.0378043_f32,
    -0.3136245_f32,
    0.2918659_f32,
    0.4917302_f32,
    0.2896284_f32,
    0.3414194_f32,
    0.1755235_f32,
    0.1256914_f32,
    0.8362176_f32,
    -0.0850143_f32,
    -0.1915376_f32,
    0.8911695_f32,
    -0.2347842_f32,
    0.1153269_f32,
    0.9195465_f32,
    0.1465120_f32,
    0.1094130_f32,
    0.8572093_f32,
    -0.0642137_f32,
    0.0955163_f32,
    0.8063645_f32,
    0.0849710_f32,
    0.0166415_f32,
    1.0403491_f32,
    -0.0234925_f32,
    0.2404096_f32,
    1.4401023_f32,
    0.1861568_f32,
    -0.0063704_f32,
    0.8736117_f32,
    0.1597628_f32,
    -0.4685430_f32,
    1.3109937_f32,
    -0.3537644_f32,
    -0.2804514_f32,
    0.8868063_f32,
    -0.1191867_f32,
    -0.0529350_f32,
    1.2485931_f32,
    -0.0165536_f32,
    -0.2303593_f32,
    0.9339324_f32,
    -0.2217678_f32,
    0.0167008_f32,
    0.8557013_f32,
    -0.0821191_f32,
    -0.1236264_f32,
    0.9315248_f32,
    -0.0426213_f32,
    -0.1151546_f32,
    1.0171108_f32,
    -0.3587087_f32,
    -0.4622723_f32,
    0.7624879_f32,
    -0.6127324_f32,
    0.4529485_f32,
    0.7064562_f32,
    -0.2278626_f32,
    0.0683081_f32,
    0.9216824_f32,
    -0.0943920_f32,
    0.2522501_f32,
    0.8600348_f32,
    0.3385496_f32,
    0.0463168_f32,
    0.8210467_f32,
    0.2932384_f32,
    0.1066161_f32,
    0.7502869_f32,
    0.3249068_f32,
    0.2662823_f32,
    0.9370281_f32,
    0.5685434_f32,
    -0.1563947_f32,
    0.7786831_f32,
    -0.4448317_f32,
    0.2253473_f32,
    0.8202156_f32,
    0.3013254_f32,
    -0.1286284_f32,
    1.4950407_f32,
    0.0213983_f32,
    -0.5589684_f32,
    0.7342592_f32,
    -0.5923732_f32,
    0.2948483_f32,
    1.2851812_f32,
    0.1327806_f32,
    -0.4794369_f32,
    0.8888700_f32,
    -0.0186413_f32,
    -0.2643828_f32,
    0.7781571_f32,
    -0.3094357_f32,
    0.0741948_f32,
    0.8275464_f32,
    0.4724047_f32,
    0.1846227_f32,
    1.0604148_f32,
    -0.0322030_f32,
    -0.0562487_f32,
    0.9108424_f32,
    -0.1458843_f32,
    0.3096041_f32,
    -0.3645234_f32,
    0.2215872_f32,
    -0.0151118_f32,
    -0.5854853_f32,
    -0.6571886_f32,
    0.0864064_f32,
    0.0554400_f32,
    -0.3441442_f32,
    0.4304998_f32,
    -0.1222365_f32,
    0.0876744_f32,
    0.4922293_f32,
    -0.3979136_f32,
    0.1105895_f32,
    -0.3491219_f32,
    0.3223356_f32,
    0.0369487_f32,
    0.2591593_f32,
    -0.3001738_f32,
    0.0398930_f32,
    -0.1349438_f32,
    -0.7392935_f32,
    -0.6872801_f32,
    -0.2623200_f32,
    -0.3164063_f32,
    -0.1295253_f32,
    0.1862662_f32,
    0.0289635_f32,
    -0.2232736_f32,
    0.3891093_f32,
    -0.0557593_f32,
    -0.0413456_f32,
    -0.0159676_f32,
    0.3946977_f32,
    -0.1217797_f32,
    -0.0803580_f32,
    0.3152435_f32,
    0.0477589_f32,
    0.4891334_f32,
    -0.5131742_f32,
    -0.1916895_f32,
    -0.1551324_f32,
    0.0474123_f32,
    -0.2053715_f32,
    0.1274302_f32,
    0.0492342_f32,
    -0.0304459_f32,
    -0.1913846_f32,
    0.5012505_f32,
    0.1695333_f32,
    0.1867280_f32,
    -0.1281780_f32,
    0.6096939_f32,
    0.2222236_f32,
    0.0298640_f32,
    0.7593144_f32,
    0.0770387_f32,
    -0.4008400_f32,
    -0.0714862_f32,
    0.0167303_f32,
    0.2760931_f32,
    -0.1931106_f32,
    0.2804477_f32,
    -0.7210483_f32,
    0.1684941_f32,
    -0.6283905_f32,
    -0.0454596_f32,
    -0.0843967_f32,
    -0.1113806_f32,
    0.3514598_f32,
    0.2006767_f32,
    0.5269167_f32,
    0.1725719_f32,
    -1.0348382_f32,
    -0.2215080_f32,
    -0.3607944_f32,
    -0.1206737_f32,
    -0.4058563_f32,
    0.3563080_f32,
    -0.0332124_f32,
    -0.0592861_f32,
    0.4153375_f32,
    0.0820642_f32,
    -0.1031251_f32,
    0.1106107_f32,
    -0.0247060_f32,
    -0.3649762_f32,
    0.7183764_f32,
    0.1683688_f32,
    0.0303138_f32,
    0.3100908_f32,
    0.5331777_f32,
    -0.1956033_f32,
    0.4588501_f32,
    -0.3364586_f32,
    -0.5548948_f32,
    0.0451455_f32,
    -0.3270112_f32,
    0.0437352_f32,
    0.2753385_f32,
    -0.0983998_f32,
    0.3128051_f32,
    0.1247400_f32,
    0.3125540_f32,
    0.0822725_f32,
    0.0793890_f32,
    -0.5979553_f32,
    -0.2662173_f32,
    -0.0564144_f32,
    -0.4582370_f32,
    0.0002683_f32,
    0.1775119_f32,
    -0.1461324_f32,
    -0.4339251_f32,
    0.0302637_f32,
    0.1299003_f32,
    -0.4851443_f32,
    0.4359943_f32,
    0.1683162_f32,
    -0.3972986_f32,
    0.1142817_f32,
    -0.5957053_f32,
    -0.1050091_f32,
    0.0753336_f32,
    -0.1346129_f32,
    -0.7221989_f32,
    0.4510674_f32,
    -0.0578670_f32,
    -0.2112773_f32,
    -0.0083005_f32,
    -0.0738409_f32,
    -0.1090840_f32,
    -0.3477055_f32,
    -0.3030507_f32,
    -0.2871751_f32,
    -0.3044738_f32,
    0.3027350_f32,
    0.0891814_f32,
    -0.1884308_f32,
    0.1081063_f32,
    -0.0675538_f32,
    -0.1116969_f32,
    -0.1190678_f32,
    -0.2371047_f32,
    0.5949025_f32,
    -0.1305167_f32,
    0.2564025_f32,
    0.1116424_f32,
    -0.0168304_f32,
    -0.7866987_f32,
    -0.0406741_f32,
    -0.1460024_f32,
    0.4093287_f32,
    0.4530105_f32,
    -0.4282092_f32,
    -0.0267042_f32,
    0.2151620_f32,
    0.0284337_f32,
    0.2842640_f32,
    -0.2811238_f32,
    0.3157822_f32,
    -0.3574869_f32,
    -0.1247506_f32,
    0.0588843_f32,
    -0.1908293_f32,
    -0.2792206_f32,
    -0.2501483_f32,
    -0.1265196_f32,
    0.2515440_f32,
    -0.0969961_f32,
    -0.0159997_f32,
    -0.2115234_f32,
    0.0800075_f32,
    -0.1518004_f32,
    0.0044863_f32,
    0.2053478_f32,
    -0.3542536_f32,
    0.2434062_f32,
    -0.2495996_f32,
    -0.4486803_f32,
    -0.5918149_f32,
    -0.1320861_f32,
    -0.3035212_f32,
    -0.7077368_f32,
    0.4684443_f32,
    0.0237804_f32,
    0.1202459_f32,
    0.2529677_f32,
    -0.1249781_f32,
    0.0918812_f32,
    -0.2291193_f32,
    0.4185495_f32,
    -0.2576239_f32,
    0.1335485_f32,
    -0.1627650_f32,
    -0.4758283_f32,
    -0.3517751_f32,
    0.2403213_f32,
    -0.4022510_f32,
    0.0838502_f32,
    -0.0119172_f32,
    -0.5610569_f32,
    -0.0859921_f32,
    -0.0031227_f32,
    -0.3412203_f32,
    0.0504075_f32,
    -0.1236619_f32,
    -0.2203480_f32,
    0.2982153_f32,
    -0.7395763_f32,
    0.2010967_f32,
    0.1400406_f32,
    -0.3920124_f32,
    -0.2172706_f32,
    0.3050216_f32,
    -0.1462787_f32,
    0.3729965_f32,
    -1.2519497_f32,
    -0.3060545_f32,
    -0.3249845_f32,
    0.0969560_f32,
    -0.4802394_f32,
    0.5640393_f32,
    -0.1228747_f32,
    -0.1662870_f32,
    -0.0364632_f32,
    -0.3074961_f32,
    -0.5146228_f32,
    -0.6102420_f32,
    0.3158281_f32,
    -0.3759804_f32,
    0.2283993_f32,
    0.7201392_f32,
    -0.4507204_f32,
    0.2044829_f32,
    0.0373349_f32,
    -0.5981894_f32,
    -0.0964030_f32,
    0.1017566_f32,
    0.0043341_f32,
    0.4357173_f32,
    0.0367781_f32,
    0.0148635_f32,
    0.2298010_f32,
    0.1781911_f32,
    -0.5266766_f32,
    0.2479423_f32,
    -0.4672582_f32,
    -0.3794813_f32,
    -0.3248662_f32,
    0.2598347_f32,
    -0.4777493_f32,
    0.2176101_f32,
    0.2960969_f32,
    0.0409489_f32,
    0.4908971_f32,
    0.0950925_f32,
    0.0181677_f32,
    0.3403699_f32,
    0.2864241_f32,
    0.0716156_f32,
    -0.1061966_f32,
    0.4810382_f32,
    0.4539103_f32,
    0.2031232_f32,
    -0.2867885_f32,
    0.2347226_f32,
    0.4990519_f32,
    0.4056346_f32,
    0.2795332_f32,
    -0.0259779_f32,
    0.3319466_f32,
    0.0838146_f32,
    -0.2497751_f32,
    -0.5224835_f32,
    -0.5761951_f32,
    0.2009215_f32,
    -0.1481455_f32,
    -0.6383340_f32,
    -0.0944971_f32,
    -0.3149903_f32,
    0.1876137_f32,
    0.2914854_f32,
    0.1854835_f32,
    -0.7588136_f32,
    0.4462815_f32,
    -0.4747460_f32,
    -0.8405462_f32,
    0.0534748_f32,
    -0.2749191_f32,
    0.0433015_f32,
    0.2221867_f32,
    -0.4260472_f32,
    0.1377477_f32,
    0.2262120_f32,
    -0.2580610_f32,
    0.0202426_f32,
    0.1734364_f32,
    -0.4811826_f32,
    -1.1210270_f32,
    0.3414957_f32,
    -0.6142725_f32,
    -0.0892315_f32,
    -0.1683281_f32,
    0.0374473_f32,
    0.1082361_f32,
    -0.2813449_f32,
    -0.1094415_f32,
    0.4468488_f32,
    0.0213857_f32,
    0.2675095_f32,
    0.0212803_f32,
    -0.1818333_f32,
    -0.2336493_f32,
    -0.2187248_f32,
    -0.0170278_f32,
    0.0421963_f32,
    0.1106600_f32,
    0.1116559_f32,
    -0.0006415_f32,
    -0.3291442_f32,
    -0.1009353_f32,
    -0.0764048_f32,
    -0.0781274_f32,
    0.0613166_f32,
    -0.2995985_f32,
    -0.0813332_f32,
    0.2040483_f32,
    0.3940564_f32,
    0.1274501_f32,
    -0.3028534_f32,
    -0.0316262_f32,
    0.1179285_f32,
    0.1061768_f32,
    0.1772182_f32,
    0.0099443_f32,
    -0.1703282_f32,
    -0.0844002_f32,
    -0.1490322_f32,
    0.0601355_f32,
    0.2348998_f32,
    0.4595267_f32,
    -0.1586762_f32,
    -0.5247776_f32,
    0.2075666_f32,
    0.7142223_f32,
    0.2137341_f32,
    0.1633020_f32,
    0.0313845_f32,
    -0.1315275_f32,
    0.2615398_f32,
    -2.2702773_f32,
    -2.2665317_f32,
    -0.0407574_f32,
    0.0161953_f32,
    -0.3513536_f32,
];

#[allow(clippy::excessive_precision)]
pub const PRETRAINED_WEIGHTS_3D: [f32; CONV3D_NUM_WEIGHTS] = [
    -0.0976156_f32,
    0.5865893_f32,
    0.8234336_f32,
    0.2408835_f32,
    -0.2319170_f32,
    -0.2722685_f32,
    0.6327428_f32,
    0.0787241_f32,
    0.3480960_f32,
    -0.3548083_f32,
    -0.5219154_f32,
    0.0518068_f32,
    0.7113128_f32,
    -0.1180177_f32,
    -0.3362499_f32,
    -0.3052263_f32,
    0.6335716_f32,
    0.2855617_f32,
    0.4766995_f32,
    -0.3466614_f32,
    -0.3927867_f32,
    -0.5134649_f32,
    0.3798009_f32,
    0.8324884_f32,
    0.1745117_f32,
    -0.2205904_f32,
    0.4338102_f32,
    0.9855352_f32,
    0.4009320_f32,
    -0.1842854_f32,
    -0.1493649_f32,
    0.4415535_f32,
    0.8436815_f32,
    0.2740238_f32,
    -0.0615668_f32,
    0.1511941_f32,
    0.4939968_f32,
    0.8286315_f32,
    0.2815271_f32,
    0.0447646_f32,
    0.2338325_f32,
    0.3033748_f32,
    0.4804524_f32,
    0.2842765_f32,
    0.1340881_f32,
    0.1129794_f32,
    -0.0064097_f32,
    0.6412024_f32,
    0.2260302_f32,
    0.0010504_f32,
    -0.0082433_f32,
    -0.0678198_f32,
    0.7106975_f32,
    0.0693463_f32,
    -0.0517540_f32,
    -0.3900950_f32,
    0.5274752_f32,
    -0.0000433_f32,
    0.6177756_f32,
    -0.3107450_f32,
    -0.2382209_f32,
    -0.2905028_f32,
    0.1979266_f32,
    0.5679480_f32,
    0.1051923_f32,
    -0.4432760_f32,
    0.4213407_f32,
    0.4768032_f32,
    0.3656780_f32,
    -0.2262964_f32,
    -0.2029778_f32,
    0.5645167_f32,
    -0.1208719_f32,
    0.6428103_f32,
    -0.1205166_f32,
    -0.0461320_f32,
    -0.1601584_f32,
    0.7572544_f32,
    0.1403813_f32,
    -0.0153732_f32,
    -0.3437098_f32,
    0.1340311_f32,
    0.8786735_f32,
    0.2098457_f32,
    -0.2447941_f32,
    -0.2215833_f32,
    0.6158163_f32,
    0.2351247_f32,
    0.5362384_f32,
    -0.1726981_f32,
    -0.1634198_f32,
    0.2523359_f32,
    0.7854782_f32,
    0.4353702_f32,
    -0.1381640_f32,
    -0.1386765_f32,
    0.5758839_f32,
    0.3000023_f32,
    0.5665652_f32,
    -0.0761982_f32,
    -0.1201167_f32,
    0.3217549_f32,
    0.8152915_f32,
    0.1893322_f32,
    -0.2406658_f32,
    -0.1896923_f32,
    -0.8737372_f32,
    -0.1065906_f32,
    0.8102896_f32,
    0.0703356_f32,
    -0.1620375_f32,
    0.2782011_f32,
    0.8909982_f32,
    0.2599207_f32,
    -0.2571162_f32,
    0.0016676_f32,
    0.3060810_f32,
    0.8802185_f32,
    0.4253758_f32,
    0.0465740_f32,
    0.3486545_f32,
    0.4238868_f32,
    0.4213827_f32,
    -0.1044520_f32,
    -0.3038613_f32,
    -0.3072591_f32,
    -0.0746676_f32,
    0.6192414_f32,
    0.0558429_f32,
    -0.0516612_f32,
    0.1684330_f32,
    0.1895531_f32,
    0.5129414_f32,
    0.1392495_f32,
    0.1197830_f32,
    -0.1272034_f32,
    0.7765920_f32,
    0.0053432_f32,
    0.5507826_f32,
    -0.1876212_f32,
    0.0622656_f32,
    0.4971855_f32,
    0.1337866_f32,
    0.3911787_f32,
    -0.2688237_f32,
    -0.3814436_f32,
    -0.1303153_f32,
    0.4369792_f32,
    0.7616761_f32,
    0.1425049_f32,
    -0.1626107_f32,
    0.5860379_f32,
    0.1345080_f32,
    0.6731585_f32,
    -0.0675683_f32,
    -0.1192611_f32,
    0.0700530_f32,
    0.7379736_f32,
    0.1812315_f32,
    -0.2239110_f32,
    -0.2289306_f32,
    0.0667208_f32,
    0.9463195_f32,
    0.4623603_f32,
    -0.2334162_f32,
    -0.2841312_f32,
    0.6144496_f32,
    0.1782960_f32,
    0.4873763_f32,
    0.0348867_f32,
    0.0065047_f32,
    0.5175835_f32,
    0.6589286_f32,
    0.7973474_f32,
    -0.1398593_f32,
    -0.2602083_f32,
    -0.0386915_f32,
    0.4439891_f32,
    0.1089773_f32,
    -0.4623799_f32,
    -0.2631274_f32,
    0.3415903_f32,
    0.7526602_f32,
    0.0694701_f32,
    -0.2168191_f32,
    -0.1620105_f32,
    0.4868574_f32,
    1.0822198_f32,
    0.4511348_f32,
    -0.1894724_f32,
    -0.0686353_f32,
    -0.9254017_f32,
    -0.3919806_f32,
    0.6672061_f32,
    -0.0583248_f32,
    -0.2564615_f32,
    0.0127624_f32,
    0.5679355_f32,
    0.1263388_f32,
    -0.3435050_f32,
    0.2367501_f32,
    0.3234884_f32,
    0.4205286_f32,
    0.3900122_f32,
    0.2552914_f32,
    0.2912583_f32,
    0.2267634_f32,
    0.4627742_f32,
    -0.0386288_f32,
    0.1283334_f32,
    -0.3029073_f32,
    -0.0437350_f32,
    0.7341929_f32,
    -0.1067808_f32,
    -0.2245992_f32,
    -0.0739648_f32,
    0.0167732_f32,
    0.5925903_f32,
    0.1866255_f32,
    0.0552297_f32,
    -0.2081706_f32,
    0.5078833_f32,
    0.1198902_f32,
    0.5085612_f32,
    -0.1255057_f32,
    -0.2017633_f32,
    0.6756870_f32,
    0.1850348_f32,
    0.6249605_f32,
    -0.1539433_f32,
    -0.1262822_f32,
    -0.2154289_f32,
    -0.5459570_f32,
    0.3874536_f32,
    0.1423837_f32,
    -0.3235873_f32,
    -0.0605569_f32,
    0.6992477_f32,
    -0.1169560_f32,
    -0.2522258_f32,
    0.3295299_f32,
    -0.3683593_f32,
    0.1404348_f32,
    0.1555665_f32,
    -0.1742305_f32,
    0.6005692_f32,
    -0.0992569_f32,
    0.2098382_f32,
    0.0954537_f32,
    -0.0306030_f32,
    -0.2254187_f32,
    -0.0183872_f32,
    0.2550589_f32,
    -0.2957007_f32,
    -0.0606671_f32,
    0.5447669_f32,
    -0.9808239_f32,
    0.1788936_f32,
    -0.4684070_f32,
    -0.3323285_f32,
    -0.0977209_f32,
    0.7490460_f32,
    -0.3596503_f32,
    -0.4497676_f32,
    0.2939382_f32,
    -0.5171903_f32,
    -0.4038871_f32,
    0.5890323_f32,
    0.1678278_f32,
    -0.1270953_f32,
    -0.0669885_f32,
    -0.2784708_f32,
    0.3654064_f32,
    0.6405777_f32,
    0.3173735_f32,
    0.3852752_f32,
    -0.3027392_f32,
    -0.3359242_f32,
    -0.4470619_f32,
    0.0698214_f32,
    -0.1243540_f32,
    0.1074519_f32,
    0.3111192_f32,
    -0.2466942_f32,
    -0.1169045_f32,
    0.1794460_f32,
    0.2515127_f32,
    0.3134565_f32,
    0.4485229_f32,
    -0.8794997_f32,
    0.2827999_f32,
    -0.0194973_f32,
    0.0856703_f32,
    0.5510954_f32,
    -0.1493742_f32,
    0.2606383_f32,
    0.4092523_f32,
    0.0689814_f32,
    0.1591952_f32,
    -0.0210568_f32,
    -0.4427639_f32,
    -0.0302895_f32,
    0.0433559_f32,
    0.0388531_f32,
    0.2595844_f32,
    -0.2038149_f32,
    0.9508591_f32,
    0.0999254_f32,
    0.4772177_f32,
    0.3338193_f32,
    0.1679785_f32,
    0.0742715_f32,
    0.3292704_f32,
    0.6394325_f32,
    0.3857028_f32,
    -0.4370784_f32,
    0.0670566_f32,
    -0.2150397_f32,
    0.1300691_f32,
    0.3379788_f32,
    -0.1362541_f32,
    -0.4149824_f32,
    -0.8558764_f32,
    -0.4167784_f32,
    -0.6275173_f32,
    -0.0568384_f32,
    0.4070663_f32,
    0.1882032_f32,
    -0.0056704_f32,
    -0.0432768_f32,
    -0.3740177_f32,
    0.1730565_f32,
    0.2446702_f32,
    0.2627119_f32,
    -0.2891131_f32,
    -0.8225421_f32,
    0.5479049_f32,
    0.0044434_f32,
    0.4826599_f32,
    0.8075740_f32,
    -0.6497988_f32,
    -0.3197050_f32,
    0.1088915_f32,
    0.1556042_f32,
    0.0067669_f32,
    0.6280903_f32,
    0.3950166_f32,
    -0.0274145_f32,
    0.3299229_f32,
    -0.0922509_f32,
    0.1307300_f32,
    0.2303542_f32,
    0.6658716_f32,
    0.2861496_f32,
    0.2928735_f32,
    -0.0015334_f32,
    0.5825924_f32,
    0.6548495_f32,
    -0.4409434_f32,
    0.2050192_f32,
    0.1693599_f32,
    -0.0897386_f32,
    0.2834574_f32,
    -0.3524189_f32,
    -0.2532374_f32,
    -0.5332743_f32,
    0.5541626_f32,
    0.2111126_f32,
    0.1639944_f32,
    -0.0504616_f32,
    0.2012189_f32,
    0.2970550_f32,
    0.8520983_f32,
    -0.4731808_f32,
    -0.2962967_f32,
    0.1936362_f32,
    -0.0864200_f32,
    0.2232141_f32,
    -0.0017433_f32,
    -0.1040500_f32,
    -0.3220062_f32,
    0.1639892_f32,
    0.1717254_f32,
    0.2288501_f32,
    0.0568804_f32,
    0.0806665_f32,
    0.0706644_f32,
    0.0936192_f32,
    -0.5810406_f32,
    0.3389675_f32,
    0.6937408_f32,
    -0.1861841_f32,
    -0.1289704_f32,
    -0.0061798_f32,
    -0.1203282_f32,
    -0.0997128_f32,
    0.3518953_f32,
    -0.2008733_f32,
    -0.2065027_f32,
    0.4065139_f32,
    0.3805785_f32,
    -0.0245618_f32,
    0.3401976_f32,
    0.2946540_f32,
    -0.4101917_f32,
    0.5233698_f32,
    -0.1983267_f32,
    -0.0206596_f32,
    0.3854667_f32,
    -0.2092257_f32,
    0.1022226_f32,
    0.1209470_f32,
    0.1731847_f32,
    -0.2183768_f32,
    -0.0077827_f32,
    0.2340731_f32,
    0.1125831_f32,
    0.1515527_f32,
    -0.0629936_f32,
    -0.0611408_f32,
    -0.2723598_f32,
    -0.4103260_f32,
    -0.6316774_f32,
    -0.2907023_f32,
    -0.4240564_f32,
    -0.2788835_f32,
    -0.0724356_f32,
    0.0131696_f32,
    -0.1515085_f32,
    0.1750832_f32,
    -0.0209173_f32,
    0.0671740_f32,
    -0.6108940_f32,
    0.1488806_f32,
    -0.0303117_f32,
    -0.0237769_f32,
    -0.3380297_f32,
    0.1213495_f32,
    1.0656165_f32,
    -0.5583879_f32,
    -0.0406271_f32,
    -0.2514542_f32,
    -0.0514522_f32,
    0.1313706_f32,
    -0.0971740_f32,
    -0.0539927_f32,
    -0.0893971_f32,
    -0.2700705_f32,
    -0.0160460_f32,
    -0.1110049_f32,
    0.3340858_f32,
    0.0117367_f32,
    -0.0104371_f32,
    0.1511707_f32,
    -1.3839163_f32,
    0.1453659_f32,
    0.0415906_f32,
    0.0681526_f32,
    0.0577984_f32,
    -0.1572631_f32,
    0.0018150_f32,
    -0.1501347_f32,
    0.4057209_f32,
    0.0725810_f32,
    0.0487302_f32,
    -0.1111329_f32,
    0.0595756_f32,
    0.1608834_f32,
    0.0848768_f32,
    -0.4654472_f32,
    -0.0619625_f32,
    0.9370192_f32,
    0.0030927_f32,
    -0.0252680_f32,
    0.3937866_f32,
    -0.0868804_f32,
    -0.0888709_f32,
    0.0342504_f32,
    0.2865414_f32,
    0.4626017_f32,
    -0.0664817_f32,
    -0.2966340_f32,
    -0.5574431_f32,
    -0.1566837_f32,
    0.5652665_f32,
    -0.8642877_f32,
    0.1088534_f32,
    -0.8720545_f32,
    -0.3298033_f32,
    -0.0193032_f32,
    -0.2575068_f32,
    0.0013762_f32,
    -0.2579965_f32,
    0.0500300_f32,
    -0.0332654_f32,
    -0.3436339_f32,
    0.0961687_f32,
    0.3507096_f32,
    0.1452819_f32,
    0.4173290_f32,
    0.2038103_f32,
    0.0421715_f32,
    0.0803799_f32,
    -0.4728044_f32,
    0.2023124_f32,
    0.4054636_f32,
    0.3039753_f32,
    0.1126815_f32,
    0.4041777_f32,
    0.4002688_f32,
    0.2833641_f32,
    0.1227788_f32,
    0.1290662_f32,
    0.8913703_f32,
    -0.0819709_f32,
    0.3683386_f32,
    1.1088820_f32,
    0.1723389_f32,
    0.1418617_f32,
    0.8664063_f32,
    -0.0926033_f32,
    0.2200187_f32,
    0.9676481_f32,
    -0.0868889_f32,
    -0.0287097_f32,
    1.0583351_f32,
    0.6951579_f32,
    -0.2709865_f32,
    0.3163892_f32,
    -1.2430345_f32,
    -0.0445386_f32,
    1.0022820_f32,
    -0.0017845_f32,
    0.5329359_f32,
    1.1495661_f32,
    0.6924145_f32,
    -0.0907409_f32,
    0.8943833_f32,
    -0.2234907_f32,
    0.2558363_f32,
    0.8782756_f32,
    -0.2773184_f32,
    0.1716597_f32,
    1.1096650_f32,
    -0.0169593_f32,
    0.1608080_f32,
    1.0827579_f32,
    0.2286979_f32,
    0.3645399_f32,
    0.8390363_f32,
    0.1331834_f32,
    -0.0719916_f32,
    0.9933913_f32,
    -0.0044286_f32,
    0.0583530_f32,
    0.9611533_f32,
    -0.1725870_f32,
    0.1505117_f32,
    1.0512412_f32,
    0.0215237_f32,
    0.1491656_f32,
    1.0468957_f32,
    -0.1678671_f32,
    0.0649399_f32,
    1.0711814_f32,
    0.2711818_f32,
    -0.1086336_f32,
    0.8506011_f32,
    0.1267437_f32,
    0.0378131_f32,
    0.9945403_f32,
    -0.2168279_f32,
    0.2839020_f32,
    1.1097300_f32,
    -0.0229765_f32,
    0.3212028_f32,
    1.0971128_f32,
    0.1764092_f32,
    0.0193826_f32,
    0.8760764_f32,
    0.1285975_f32,
    0.4890027_f32,
    1.1460613_f32,
    0.7786202_f32,
    -0.0626174_f32,
    0.8631148_f32,
    -0.0611514_f32,
    -0.2474871_f32,
    0.9575630_f32,
    0.0698243_f32,
    -0.1563008_f32,
    1.0598850_f32,
    -0.0022975_f32,
    -0.3100439_f32,
    1.0589292_f32,
    -0.5999134_f32,
    -0.5350822_f32,
    0.8955259_f32,
    -0.3289533_f32,
    0.0320311_f32,
    0.9984024_f32,
    0.0836964_f32,
    -0.0767166_f32,
    0.8861030_f32,
    0.1190345_f32,
    0.1941680_f32,
    1.1190816_f32,
    0.0339921_f32,
    0.0122775_f32,
    0.9701337_f32,
    0.2830221_f32,
    0.2137495_f32,
    0.9747545_f32,
    0.4714881_f32,
    0.0635100_f32,
    0.7753871_f32,
    0.1939915_f32,
    0.0953922_f32,
    0.9393375_f32,
    0.3287779_f32,
    0.3957763_f32,
    1.0679222_f32,
    0.5971149_f32,
    0.2515192_f32,
    1.0563650_f32,
    0.5500914_f32,
    0.0454987_f32,
    0.9670420_f32,
    0.4008161_f32,
    0.6394243_f32,
    1.1475577_f32,
    0.5707502_f32,
    0.0835870_f32,
    0.9485882_f32,
    0.0248572_f32,
    -0.3437831_f32,
    0.8994003_f32,
    -0.1906862_f32,
    0.1739060_f32,
    1.1674629_f32,
    0.2980099_f32,
    -0.0545066_f32,
    1.0888529_f32,
    -0.1651553_f32,
    0.2676092_f32,
    0.7580549_f32,
    0.5553530_f32,
    -0.3227307_f32,
    0.9823096_f32,
    -0.1327482_f32,
    -0.1396336_f32,
    0.9205542_f32,
    -0.0759557_f32,
    0.2259490_f32,
    1.0587139_f32,
    0.2116114_f32,
    -0.2150586_f32,
    -0.4815771_f32,
    -0.2345996_f32,
    0.0715430_f32,
    0.1694490_f32,
    0.6277893_f32,
    -0.2980029_f32,
    -0.3116676_f32,
    -0.2534522_f32,
    -0.0138473_f32,
    0.8790613_f32,
    -0.6601876_f32,
    -0.3273432_f32,
    -0.0320922_f32,
    0.2296826_f32,
    -0.7639822_f32,
    -0.2126144_f32,
    0.3112076_f32,
    0.1890715_f32,
    -0.4402646_f32,
    -0.1989510_f32,
    0.3819954_f32,
    0.1802935_f32,
    -0.4343880_f32,
    0.1462527_f32,
    0.0455597_f32,
    0.0330499_f32,
    -0.3809013_f32,
    0.0917431_f32,
    -0.0742168_f32,
    0.1854619_f32,
    -0.3023599_f32,
    0.0821447_f32,
    -0.4640500_f32,
    -0.0656499_f32,
    0.3148834_f32,
    -0.0142385_f32,
    -0.1224921_f32,
    0.2085187_f32,
    -0.3357277_f32,
    0.0071951_f32,
    -0.3801588_f32,
    0.4073377_f32,
    -0.2737229_f32,
    -0.6480386_f32,
    0.1416543_f32,
    0.2305917_f32,
    -0.4069017_f32,
    0.0930619_f32,
    -0.3361472_f32,
    0.0197008_f32,
    0.2601102_f32,
    0.4998631_f32,
    0.7458162_f32,
    -0.3030633_f32,
    -0.3106394_f32,
    0.1557774_f32,
    -0.1393241_f32,
    0.3388757_f32,
    -0.7485161_f32,
    -0.2825306_f32,
    0.3942816_f32,
    -0.0023692_f32,
    -0.8545148_f32,
    0.0209450_f32,
    -0.7245241_f32,
    0.3064304_f32,
    0.2281581_f32,
    -0.1104448_f32,
    0.2878368_f32,
    0.0564405_f32,
    -0.0034715_f32,
    0.0213236_f32,
    -0.3463767_f32,
    -0.1118070_f32,
    -0.3310691_f32,
    -0.2943667_f32,
    0.1948387_f32,
    0.1440035_f32,
    -0.4818816_f32,
    -0.0789752_f32,
    -0.0475191_f32,
    -0.1980652_f32,
    -0.5627871_f32,
    -1.1011760_f32,
    -0.2705984_f32,
    -0.6317919_f32,
    -0.0953277_f32,
    -0.2671058_f32,
    0.3563909_f32,
    -0.0501725_f32,
    0.1251185_f32,
    -0.0024093_f32,
    -0.5946230_f32,
    -0.2770158_f32,
    0.0868469_f32,
    0.4372998_f32,
    -1.0561750_f32,
    0.2019041_f32,
    0.4024456_f32,
    0.2089194_f32,
    0.0663882_f32,
    0.1930386_f32,
    -0.2509571_f32,
    -0.0155761_f32,
    -0.1245154_f32,
    0.2880098_f32,
    -0.4039198_f32,
    -0.4449839_f32,
    0.0360742_f32,
    0.0785074_f32,
    -0.7333779_f32,
    0.0374439_f32,
    -0.6991382_f32,
    -0.0062087_f32,
    0.1952604_f32,
    -0.0794176_f32,
    0.6325591_f32,
    0.1258151_f32,
    -0.2446987_f32,
    0.0741554_f32,
    -0.7450882_f32,
    -0.0141401_f32,
    -0.1925968_f32,
    -0.2425244_f32,
    0.4810871_f32,
    0.1200433_f32,
    -0.5432531_f32,
    0.3180644_f32,
    -0.8195585_f32,
    0.0018904_f32,
    0.2279391_f32,
    0.3819911_f32,
    0.7607157_f32,
    0.0227216_f32,
    -0.2518524_f32,
    0.4207601_f32,
    -0.4938162_f32,
    -0.3393400_f32,
    -0.1476648_f32,
    -0.5783122_f32,
    0.1096253_f32,
    0.0586957_f32,
    -0.2941149_f32,
    -0.2041496_f32,
    -0.2823389_f32,
    -0.7590166_f32,
    0.0984770_f32,
    -0.2234334_f32,
    -0.0723187_f32,
    -0.6239209_f32,
    -0.6325903_f32,
    -0.2222456_f32,
    -0.0244784_f32,
    0.2966291_f32,
    0.2093343_f32,
    -0.2688833_f32,
    -0.0791881_f32,
    -0.8672296_f32,
    -0.0247865_f32,
    0.0127916_f32,
    -0.3957820_f32,
    0.4069268_f32,
    0.0186687_f32,
    -0.1345360_f32,
    0.0709840_f32,
    0.1383424_f32,
    -0.5773191_f32,
    0.0646583_f32,
    -0.4039402_f32,
    0.2468977_f32,
    0.0070791_f32,
    -0.4549448_f32,
    0.2705908_f32,
    -0.0364471_f32,
    -0.1968273_f32,
    -0.4171826_f32,
    0.1853591_f32,
    -0.3599790_f32,
    -0.3924109_f32,
    -0.9873305_f32,
    -0.0236451_f32,
    -0.4369704_f32,
    -0.3435808_f32,
    -0.3827062_f32,
    -0.3815695_f32,
    -0.2316054_f32,
    0.2810324_f32,
    -0.0667219_f32,
    0.1555770_f32,
    0.0738921_f32,
    0.0186481_f32,
    -0.5534586_f32,
    -0.0189945_f32,
    -0.6178082_f32,
    -0.6310515_f32,
    -1.0482125_f32,
    0.6715516_f32,
    -0.6172658_f32,
    -0.7239385_f32,
    -0.4223104_f32,
    0.3821667_f32,
    -0.3781609_f32,
    -0.4107505_f32,
    -0.0463715_f32,
    -0.4596776_f32,
    -0.4936227_f32,
    -0.3267912_f32,
    0.1723617_f32,
    -0.1801065_f32,
    0.1414004_f32,
    0.2754991_f32,
    -0.0445292_f32,
    0.1011325_f32,
    -0.1429617_f32,
    -0.1663917_f32,
    0.1718503_f32,
    -0.4029871_f32,
    0.1995195_f32,
    0.0507600_f32,
    -0.1068991_f32,
    0.6494417_f32,
    0.2228922_f32,
    -0.4733713_f32,
    -0.2036799_f32,
    -0.3645424_f32,
    0.0340878_f32,
    0.0463455_f32,
    0.1899979_f32,
    0.4913444_f32,
    0.1001289_f32,
    -0.6297565_f32,
    0.2036585_f32,
    -0.2957630_f32,
    -0.1077577_f32,
    0.1951456_f32,
    -0.6914741_f32,
    0.0110312_f32,
    -0.0962586_f32,
    -0.0358197_f32,
    -0.4012583_f32,
    0.0231706_f32,
    -0.1281997_f32,
    -0.8421248_f32,
    0.1666217_f32,
    0.3005572_f32,
    -0.1027572_f32,
    -0.1166396_f32,
    -0.0727212_f32,
    -0.0180009_f32,
    -0.3706886_f32,
    0.3216677_f32,
    0.3744043_f32,
    -0.2427881_f32,
    0.0365849_f32,
    0.0474909_f32,
    0.1623246_f32,
    0.2930891_f32,
    0.2488571_f32,
    0.4307851_f32,
    -0.0908276_f32,
    -0.1336632_f32,
    -0.0631434_f32,
    0.1080624_f32,
    0.0419672_f32,
    0.3945646_f32,
    0.3952877_f32,
    -0.1056918_f32,
    0.2305445_f32,
    0.1911329_f32,
    0.3260361_f32,
    0.1858367_f32,
    0.0440293_f32,
    -0.6557166_f32,
    0.1156181_f32,
    -0.0017963_f32,
    0.3244708_f32,
    0.1014456_f32,
    0.2468570_f32,
    0.3840420_f32,
    0.2116503_f32,
    -0.8463494_f32,
    0.0452271_f32,
    0.0122542_f32,
    -0.6610509_f32,
    0.1553790_f32,
    -0.0081018_f32,
    -0.3137966_f32,
    -0.3779987_f32,
    0.1755779_f32,
    -0.5357634_f32,
    -0.5429783_f32,
    -0.0517890_f32,
    0.1552144_f32,
    -0.2603098_f32,
    -0.0602923_f32,
    -0.1586821_f32,
    -0.1995614_f32,
    -0.6059815_f32,
    0.2292956_f32,
    0.4575188_f32,
    -0.4609511_f32,
    -0.7025276_f32,
    -0.0529122_f32,
    0.0936021_f32,
    -0.0662461_f32,
    -0.3161530_f32,
    0.1855838_f32,
    -0.0045533_f32,
    0.3807741_f32,
    -0.3142389_f32,
    0.1079214_f32,
    0.3304892_f32,
    0.1348565_f32,
    -0.1116893_f32,
    0.2480519_f32,
    -0.1326695_f32,
    -0.1445942_f32,
    0.1271062_f32,
    -0.3148151_f32,
    0.3086464_f32,
    -0.2716174_f32,
    -0.1267714_f32,
    0.2351022_f32,
    0.1043204_f32,
    0.0513768_f32,
    -0.0031377_f32,
    0.0185488_f32,
    -0.2646657_f32,
    0.0786427_f32,
    0.0316442_f32,
    0.2512519_f32,
    -0.0425524_f32,
    0.0146502_f32,
    -0.2090448_f32,
    0.3714326_f32,
    0.1231930_f32,
    0.1179512_f32,
    0.0037010_f32,
    -0.0021364_f32,
    0.0855395_f32,
    -0.2811730_f32,
    0.2779074_f32,
    -0.2193812_f32,
    0.0505008_f32,
    -0.1677244_f32,
    -0.0521841_f32,
    0.6761827_f32,
    -0.2363487_f32,
    -0.3807964_f32,
    0.0283764_f32,
    -0.0172157_f32,
    -0.0192487_f32,
    0.1193792_f32,
    -0.1898157_f32,
    -0.1632355_f32,
    0.3336088_f32,
    0.2504867_f32,
    0.2001165_f32,
    -0.1812013_f32,
    0.0634891_f32,
    -0.3439041_f32,
    -0.3072314_f32,
    -0.2081489_f32,
    -0.0836721_f32,
    0.1509564_f32,
    0.0470818_f32,
    0.1288994_f32,
    -2.7429733_f32,
    -2.6276667_f32,
    -0.0047758_f32,
    0.0667859_f32,
    0.0743652_f32,
    -0.2614106_f32,
];
