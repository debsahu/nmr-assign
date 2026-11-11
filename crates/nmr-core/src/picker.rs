//! Classical baseline 2D and 3D NMR peak picker.
//!
//! Provenance: Original implementation of `docs/specs/02-peak-picking.md` Stage A:
//! robust MAD noise estimation on empty regions, local extrema detection with phase sign,
//! sub-voxel 3-point log-Gaussian / parabolic centre refinement, and HSQC strip restriction.

use crate::peaks::Peak;
use crate::spectrum::{Spectrum2D, Spectrum3D};

#[derive(Clone, Debug)]
pub struct PickerConfig {
    /// Signal-to-noise threshold multiplier (`k * noise`, default `5.0`).
    pub snr_threshold: f32,
    /// Allow negative peaks (required for phase-sensitive `HNCACB`).
    pub allow_negative: bool,
    /// Strip half-widths `(H_ppm, N_ppm)` when restricting 3D picking under HSQC roots.
    pub strip_tol_h: f64,
    pub strip_tol_n: f64,
}

impl Default for PickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.0,
            allow_negative: false,
            strip_tol_h: 0.035,
            strip_tol_n: 0.35,
        }
    }
}

fn median_f32(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Robust MAD noise estimate (`1.4826 * median(|x - median(x)|)`) after trimming signal outliers.
pub fn estimate_noise_mad(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    // Stride large 3D arrays for fast deterministic O(N/stride) noise estimation
    let stride = (data.len() / 65_536).max(1);
    let mut sample: Vec<f32> = data.iter().step_by(stride).copied().collect();
    let med0 = median_f32(&mut sample);

    let mut dev0: Vec<f32> = sample.iter().map(|&v| (v - med0).abs()).collect();
    let mad0 = (1.4826 * median_f32(&mut dev0)).max(1e-12);

    // Exclude peak regions (> 3.5 sigma) and recompute MAD on baseline noise
    let mut bg: Vec<f32> = sample
        .into_iter()
        .filter(|&v| (v - med0).abs() <= 3.5 * mad0)
        .collect();
    if bg.len() < 16 {
        return mad0;
    }
    let med1 = median_f32(&mut bg);
    let mut dev1: Vec<f32> = bg.into_iter().map(|v| (v - med1).abs()).collect();
    (1.4826 * median_f32(&mut dev1)).max(1e-12)
}

/// Sub-voxel 3-point peak offset in `[-0.5, 0.5]` using log-Gaussian fit (with parabolic fallback).
#[inline]
fn subvoxel_offset(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.abs()) as f64;
    let b = (y0.abs()) as f64;
    let c = (yp1.abs()) as f64;
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

/// Pick 2D `HSQC` root peaks from a processed 2D spectrum.
pub fn pick_hsqc_2d(spec: &Spectrum2D, cfg: &PickerConfig) -> Vec<Peak> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 3 || ny < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let mut peaks = Vec::new();

    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let v = spec.get(ix, iy);
            if v.abs() < thresh || (!cfg.allow_negative && v < 0.0) {
                continue;
            }
            let sign = if v >= 0.0 { 1 } else { -1 };
            let mut is_extremum = true;
            'nbr: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = spec.get((ix as isize + dx) as usize, (iy as isize + dy) as usize);
                    if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                        is_extremum = false;
                        break 'nbr;
                    }
                }
            }
            if !is_extremum {
                continue;
            }
            let dx = subvoxel_offset(spec.get(ix - 1, iy), v, spec.get(ix + 1, iy));
            let dy = subvoxel_offset(spec.get(ix, iy - 1), v, spec.get(ix, iy + 1));
            let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
            let n = spec.axis_y.index_to_ppm(iy as f64 + dy);
            let mut vol = 0.0f64;
            for wy in -1isize..=1 {
                for wx in -1isize..=1 {
                    let sv = spec.get((ix as isize + wx) as usize, (iy as isize + wy) as usize);
                    if sv.signum() == v.signum() {
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
                height: Some((v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            });
        }
    }
    peaks
}

/// Pick 2D `C13HSQC` peaks `(1H, 13C)` from an aliphatic or aromatic `13C`-HSQC spectrum.
pub fn pick_c13_hsqc_2d(spec: &Spectrum2D, cfg: &PickerConfig) -> Vec<Peak> {
    let mut base = pick_hsqc_2d(spec, cfg);
    for p in &mut base {
        p.exp = "C13HSQC".to_string();
        p.c = Some(p.n);
        p.n = 0.0;
    }
    base
}

/// Pick 3D triple-resonance peaks restricted to `(H, N)` strips around `hsqc_roots`.
pub fn pick_triple_3d(spec: &Spectrum3D, hsqc_roots: &[Peak], cfg: &PickerConfig) -> Vec<Peak> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let allow_neg = cfg.allow_negative || spec.exp == "HNCACB";

    let y_is_n = spec.axis_y.label.contains('N')
        || (spec.axis_y.max_ppm() > 95.0
            && spec.axis_y.min_ppm() < 145.0
            && !spec.axis_z.label.contains('N'));

    let mut out = Vec::new();

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            let n_ppm_grid = if y_is_n {
                spec.axis_y.index_to_ppm(iy as f64)
            } else {
                spec.axis_z.index_to_ppm(iz as f64)
            };
            // Quick filter along 15N before scanning 1H row
            if !hsqc_roots.is_empty()
                && !hsqc_roots
                    .iter()
                    .any(|r| (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n * 1.5)
            {
                continue;
            }

            for ix in 1..nx - 1 {
                let v = spec.get(ix, iy, iz);
                if v.abs() < thresh || (!allow_neg && v < 0.0) {
                    continue;
                }
                let h_ppm_grid = spec.axis_x.index_to_ppm(ix as f64);
                if !hsqc_roots.is_empty()
                    && !hsqc_roots.iter().any(|r| {
                        (r.h - h_ppm_grid).abs() <= cfg.strip_tol_h
                            && (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n
                    })
                {
                    continue;
                }

                let sign = if v >= 0.0 { 1 } else { -1 };
                let mut is_extremum = true;
                'nbr: for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nv = spec.get(
                                (ix as isize + dx) as usize,
                                (iy as isize + dy) as usize,
                                (iz as isize + dz) as usize,
                            );
                            if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                                is_extremum = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if !is_extremum {
                    continue;
                }

                let dx = subvoxel_offset(spec.get(ix - 1, iy, iz), v, spec.get(ix + 1, iy, iz));
                let dy = subvoxel_offset(spec.get(ix, iy - 1, iz), v, spec.get(ix, iy + 1, iz));
                let dz = subvoxel_offset(spec.get(ix, iy, iz - 1), v, spec.get(ix, iy, iz + 1));

                let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
                let (n, c) = if y_is_n {
                    (
                        spec.axis_y.index_to_ppm(iy as f64 + dy),
                        spec.axis_z.index_to_ppm(iz as f64 + dz),
                    )
                } else {
                    (
                        spec.axis_z.index_to_ppm(iz as f64 + dz),
                        spec.axis_y.index_to_ppm(iy as f64 + dy),
                    )
                };

                let mut vol = 0.0f64;
                for wz in -1isize..=1 {
                    for wy in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = spec.get(
                                (ix as isize + wx) as usize,
                                (iy as isize + wy) as usize,
                                (iz as isize + wz) as usize,
                            );
                            if sv.signum() == v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                out.push(Peak {
                    exp: spec.exp.clone(),
                    h,
                    n,
                    c: Some(c),
                    sign,
                    height: Some((v.abs()) as f64),
                    volume: Some(vol),
                    h2: None,
                });
            }
        }
    }
    out
}
