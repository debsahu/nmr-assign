//! Classical baseline 2D and 3D NMR peak picker with real-spectrum artifact hygiene.
//!
//! Provenance: Original clean-room implementation of `docs/specs/02-peak-picking.md` Stage A:
//! two-pass outlier-trimmed MAD noise estimation (`1.4826 * median(|x - median(x)|)`),
//! amide `1H` and experiment-specific `13C` spectral window gating, local indirect-axis column/strip
//! MAD `t1`-ridge thresholding and FT truncation sinc-sidelobe suppression, automatic `180°` phase
//! polarity detection for single-sign 3D triple-resonance spectra, resolved `^3J_HNHa` `1H` doublet
//! coalescing, local extrema detection, and 3-point log-Gaussian / parabolic sub-voxel refinement.

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
    /// Amide `1H` window `(min_ppm, max_ppm)` applied to `15N-HSQC` when the `1H` axis spans wide
    /// sweep widths (`min_ppm < 4.5`) or when customized (`None` disables gating, e.g. `13C-HSQC`).
    pub amide_h_window: Option<(f64, f64)>,
    /// Multiplier on local indirect-axis column/strip MAD for `t1` noise ridge suppression (default `5.5`).
    pub ridge_snr: f32,
    /// Minimum fraction of column/strip parent amplitude for FT truncation sinc suppression (default `0.025`).
    pub sinc_frac: f32,
    /// Coalesce resolved `^3J_HNHa` `1H` doublets (`|dH| * obs_mhz <= max_j_hnha_hz`, `|dN| <= 0.08 ppm`)
    /// in high-digital-resolution `15N-HSQC` spectra (`|step_ppm| * obs_mhz <= 6.5 Hz`, default `true`).
    pub coalesce_j_doublets: bool,
    /// Maximum `^3J_HNHa` scalar splitting in Hz for `15N-HSQC` doublet coalescing (default `9.5` Hz),
    /// converted to ppm via the `1H` axis spectrometer frequency `obs_mhz`.
    pub max_j_hnha_hz: f64,
}

impl Default for PickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.0,
            allow_negative: false,
            strip_tol_h: 0.035,
            strip_tol_n: 0.35,
            amide_h_window: Some((5.5, 11.5)),
            ridge_snr: 5.5,
            sinc_frac: 0.025,
            coalesce_j_doublets: true,
            max_j_hnha_hz: 9.5,
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

/// Compute two-pass outlier-trimmed local MAD (floored at `fallback_mad`) and `max_abs` for a 1D slice.
fn slice_trimmed_mad_and_max(
    vals: &[f32],
    scratch: &mut Vec<f32>,
    fallback_mad: f32,
) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    for &v in vals {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if vals.len() < 24 {
        return (fallback_mad, max_abs);
    }
    scratch.clear();
    scratch.extend_from_slice(vals);
    let med0 = median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med0).abs();
    }
    let mad0 = (1.4826 * median_f32(scratch)).max(1e-12);

    scratch.clear();
    scratch.extend(
        vals.iter()
            .copied()
            .filter(|&v| (v - med0).abs() <= 3.5 * mad0),
    );
    if scratch.len() < 12 {
        return (mad0.max(fallback_mad), max_abs);
    }
    let med1 = median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med1).abs();
    }
    let mad1 = (1.4826 * median_f32(scratch)).max(fallback_mad);
    (mad1, max_abs)
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

#[inline]
fn active_amide_window(axis_min_ppm: f64, cfg: &PickerConfig) -> Option<(f64, f64)> {
    let (w0, w1) = cfg.amide_h_window?;
    let (lo, hi) = (w0.min(w1), w0.max(w1));
    let custom = (lo - 5.5).abs() > 1e-6 || (hi - 11.5).abs() > 1e-6;
    if axis_min_ppm < 4.5 || custom {
        Some((lo, hi))
    } else {
        None
    }
}

#[inline]
fn carbon_bounds_for_exp(exp: &str) -> Option<(f64, f64)> {
    match exp {
        "HNCACB" | "CBCACONH" | "HNCA" | "HNCOCA" => Some((8.0, 76.0)),
        "HNCO" | "HNCACO" => Some((164.0, 187.0)),
        _ => None,
    }
}

/// Coalesce symmetric `^3J_HNHa` `1H` doublets (`|dN| <= 0.08 ppm`, `|dH| <= max_j_ppm`,
/// `min_ht >= 0.40 * max_ht`) in high-resolution `15N-HSQC` spectra into intensity-weighted centroids,
/// where `max_j_ppm = max_j_hnha_hz / obs_mhz`.
fn coalesce_hsqc_j_doublets(mut peaks: Vec<Peak>, max_j_ppm: f64) -> Vec<Peak> {
    if peaks.len() < 2 || max_j_ppm <= 0.0 {
        return peaks;
    }
    peaks.sort_by(|a, b| b.height.unwrap_or(0.0).total_cmp(&a.height.unwrap_or(0.0)));
    let n = peaks.len();
    let mut merged = vec![false; n];
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        if merged[i] {
            continue;
        }
        let pi = &peaks[i];
        let hi = pi.height.unwrap_or(0.0);
        let mut best_j: Option<usize> = None;
        let mut best_dh = f64::INFINITY;
        for j in (i + 1)..n {
            if merged[j] || peaks[j].sign != pi.sign {
                continue;
            }
            let pj = &peaks[j];
            let hj = pj.height.unwrap_or(0.0);
            let dn = (pi.n - pj.n).abs();
            let dh = (pi.h - pj.h).abs();
            if dn <= 0.08 && dh <= max_j_ppm && hj >= 0.40 * hi && dh < best_dh {
                best_dh = dh;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            merged[j] = true;
            let pj = &peaks[j];
            let hj = pj.height.unwrap_or(0.0);
            let w_sum = (hi + hj).max(1e-12);
            let h_cen = (hi * pi.h + hj * pj.h) / w_sum;
            let n_cen = (hi * pi.n + hj * pj.n) / w_sum;
            let vol = match (pi.volume, pj.volume) {
                (Some(va), Some(vb)) => Some(va + vb),
                (a, b) => a.or(b),
            };
            out.push(Peak {
                exp: pi.exp.clone(),
                h: h_cen,
                n: n_cen,
                c: pi.c,
                sign: pi.sign,
                height: Some(hi + hj),
                volume: vol,
                h2: pi.h2,
            });
        } else {
            out.push(pi.clone());
        }
    }
    out
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
    let h_gate = active_amide_window(spec.axis_x.min_ppm(), cfg);

    // Identify active 1H columns within the optional amide window (plus 1-point margin for subvoxel fit)
    let step_h = spec.axis_x.step_ppm.abs();
    let mut active_x = vec![false; nx];
    let mut active_ix_list = Vec::with_capacity(nx);
    let mut spec_max_abs = 0.0f32;
    for (ix, slot) in active_x.iter_mut().enumerate().take(nx - 1).skip(1) {
        if let Some((lo, hi)) = h_gate {
            let hp = spec.axis_x.index_to_ppm(ix as f64);
            if hp < lo - step_h || hp > hi + step_h {
                continue;
            }
        }
        *slot = true;
        active_ix_list.push(ix);
    }
    for iy in 0..ny {
        for &ix in &active_ix_list {
            let a = spec.get(ix, iy).abs();
            if a > spec_max_abs {
                spec_max_abs = a;
            }
        }
    }

    // Activate local ridge MAD and FT truncation sinc suppression in high-dynamic-range spectra
    let high_dr = spec_max_abs > 40.0 * noise;
    let mut col_thresh = vec![thresh; nx];
    let mut row_thresh = vec![thresh; ny];
    if high_dr {
        let mut slice_buf = Vec::with_capacity(nx.max(ny));
        let mut scratch = Vec::with_capacity(nx.max(ny));
        for &ix in &active_ix_list {
            slice_buf.clear();
            for iy in 0..ny {
                slice_buf.push(spec.get(ix, iy));
            }
            let (cmad, cmax) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            col_thresh[ix] = thresh.max(cfg.ridge_snr * cmad).max(cfg.sinc_frac * cmax);
        }
        for (iy, rt) in row_thresh.iter_mut().enumerate().take(ny - 1).skip(1) {
            slice_buf.clear();
            for &ix in &active_ix_list {
                slice_buf.push(spec.get(ix, iy));
            }
            let (rmad, _) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            *rt = thresh.max(cfg.ridge_snr * rmad);
        }
    }

    let mut peaks = Vec::new();

    for (iy, &r_thr) in row_thresh.iter().enumerate().take(ny - 1).skip(1) {
        for &ix in &active_ix_list {
            let v = spec.get(ix, iy);
            let loc_thr = col_thresh[ix].max(r_thr);
            if v.abs() < loc_thr || (!cfg.allow_negative && v < 0.0) {
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
            if let Some((lo, hi)) = h_gate {
                if h < lo || h > hi {
                    continue;
                }
            }
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

    let obs_mhz = if spec.axis_x.obs_mhz > 10.0 {
        spec.axis_x.obs_mhz
    } else {
        600.0
    };
    let max_j_ppm = cfg.max_j_hnha_hz.max(0.0) / obs_mhz;
    let min_f2_sidelobe_dh = (12.0 / obs_mhz).max(if cfg.coalesce_j_doublets {
        max_j_ppm
    } else {
        0.0
    });

    // Suppress near-parent F1 (indirect) and F2 (direct) truncation sinc sidelobes
    if high_dr && peaks.len() > 1 {
        let min_parent = (40.0 * noise) as f64;
        let base_sinc = cfg.sinc_frac as f64;
        let mut keep = vec![true; peaks.len()];
        for i in 0..peaks.len() {
            let pi = &peaks[i];
            let hti = pi.height.unwrap_or(0.0);
            for (j, pj) in peaks.iter().enumerate() {
                if i == j || pj.sign != pi.sign {
                    continue;
                }
                let htj = pj.height.unwrap_or(0.0);
                if htj <= hti || htj < min_parent {
                    continue;
                }
                let dh = (pi.h - pj.h).abs();
                let dn = (pi.n - pj.n).abs();
                // F1 indirect sidelobe along the same 1H column
                if dh <= 0.025 && dn > 0.08 {
                    let env = base_sinc + 0.16 / (1.0 + (dn / 0.65).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
                // F2 direct sidelobe along the same indirect row (above ^3J_HNHa / 10 Hz in ppm)
                if dn <= 0.25 && dh > min_f2_sidelobe_dh && dh <= 0.35 {
                    let env = 0.015 + 0.12 / (1.0 + (dh / 0.06).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        let mut filtered = Vec::with_capacity(peaks.len());
        for (p, k) in peaks.into_iter().zip(keep) {
            if k {
                filtered.push(p);
            }
        }
        peaks = filtered;
    }

    // Require digital resolution <= 6.5 Hz/pt (e.g. SI=2048 over 14 ppm up to 950 MHz)
    // to resolve ^3J_HNHa doublets without merging coarse-grid peaks
    if cfg.coalesce_j_doublets
        && cfg.max_j_hnha_hz > 0.0
        && cfg.amide_h_window.is_some()
        && step_h * obs_mhz <= 6.5
    {
        peaks = coalesce_hsqc_j_doublets(peaks, max_j_ppm);
    }

    peaks
}

/// Pick 2D `C13HSQC` peaks `(1H, 13C)` from an aliphatic or aromatic `13C`-HSQC spectrum.
pub fn pick_c13_hsqc_2d(spec: &Spectrum2D, cfg: &PickerConfig) -> Vec<Peak> {
    let mut c13_cfg = cfg.clone();
    c13_cfg.amide_h_window = None;
    c13_cfg.coalesce_j_doublets = false;
    let mut base = pick_hsqc_2d(spec, &c13_cfg);
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
    let c_bounds = carbon_bounds_for_exp(&spec.exp);
    let h_gate = if hsqc_roots.is_empty() && c_bounds.is_some() {
        active_amide_window(spec.axis_x.min_ppm(), cfg)
    } else {
        None
    };

    let y_is_n = spec.axis_y.label.contains('N')
        || (spec.axis_y.max_ppm() > 95.0
            && spec.axis_y.min_ppm() < 145.0
            && !spec.axis_z.label.contains('N'));

    let (n_n, n_c) = if y_is_n { (ny, nz) } else { (nz, ny) };
    let axis_n = if y_is_n { &spec.axis_y } else { &spec.axis_z };
    let axis_c = if y_is_n { &spec.axis_z } else { &spec.axis_y };

    let step_c = axis_c.step_ppm.abs();
    let mut active_c = Vec::with_capacity(n_c);
    for ic in 1..n_c - 1 {
        let cp = axis_c.index_to_ppm(ic as f64);
        if let Some((clo, chi)) = c_bounds {
            if cp < clo - step_c || cp > chi + step_c {
                continue;
            }
        }
        active_c.push(ic);
    }
    if active_c.is_empty() {
        return Vec::new();
    }

    // Preselect active (ix, in_idx) strip columns matching hsqc_roots (or h_gate when rootless)
    let step_h = spec.axis_x.step_ppm.abs();
    let mut active_strips: Vec<(usize, usize)> = Vec::new();
    for in_idx in 1..n_n - 1 {
        let n_ppm_grid = axis_n.index_to_ppm(in_idx as f64);
        if !hsqc_roots.is_empty()
            && !hsqc_roots
                .iter()
                .any(|r| (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n)
        {
            continue;
        }
        for ix in 1..nx - 1 {
            let h_ppm_grid = spec.axis_x.index_to_ppm(ix as f64);
            if let Some((lo, hi)) = h_gate {
                if h_ppm_grid < lo - step_h || h_ppm_grid > hi + step_h {
                    continue;
                }
            }
            if !hsqc_roots.is_empty()
                && !hsqc_roots.iter().any(|r| {
                    (r.h - h_ppm_grid).abs() <= cfg.strip_tol_h
                        && (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n
                })
            {
                continue;
            }
            active_strips.push((ix, in_idx));
        }
    }
    if active_strips.is_empty() {
        return Vec::new();
    }

    let get_voxel = |ix: usize, in_idx: usize, ic: usize| -> f32 {
        if y_is_n {
            spec.get(ix, in_idx, ic)
        } else {
            spec.get(ix, ic, in_idx)
        }
    };

    // Scan active strips to measure peak dynamic range and detect 180-degree inverted single-sign phase
    let mut max_pos = 0.0f32;
    let mut max_neg = 0.0f32;
    for &(ix, in_idx) in &active_strips {
        for &ic in &active_c {
            let v = get_voxel(ix, in_idx, ic);
            if v > max_pos {
                max_pos = v;
            } else if -v > max_neg {
                max_neg = -v;
            }
        }
    }
    let single_sign_backbone = !allow_neg
        && matches!(
            spec.exp.as_str(),
            "HNCA" | "HNCOCA" | "CBCACONH" | "HNCO" | "HNCACO"
        );
    let invert_polarity = single_sign_backbone && max_neg > 2.5 * max_pos.max(thresh);
    let spec_max_abs = if invert_polarity {
        max_neg
    } else if allow_neg {
        max_pos.max(max_neg)
    } else {
        max_pos
    };
    let high_dr = spec_max_abs > 40.0 * noise;
    let far_sinc_frac = (cfg.sinc_frac * 0.4).min(0.012);

    let mut strip_buf = Vec::with_capacity(n_c);
    let mut scratch = Vec::with_capacity(n_c);
    let mut out = Vec::new();

    for &(ix, in_idx) in &active_strips {
        let loc_thr = if high_dr {
            strip_buf.clear();
            for ic in 0..n_c {
                strip_buf.push(get_voxel(ix, in_idx, ic));
            }
            let (smad, smax) = slice_trimmed_mad_and_max(&strip_buf, &mut scratch, noise);
            thresh.max(cfg.ridge_snr * smad).max(far_sinc_frac * smax)
        } else {
            thresh
        };

        for &ic in &active_c {
            let (iy, iz) = if y_is_n { (in_idx, ic) } else { (ic, in_idx) };
            let raw_v = spec.get(ix, iy, iz);
            let eff_v = if invert_polarity { -raw_v } else { raw_v };
            if eff_v.abs() < loc_thr || (!allow_neg && eff_v < 0.0) {
                continue;
            }

            let raw_sign = if raw_v >= 0.0 { 1 } else { -1 };
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
                        if (raw_sign > 0 && nv >= raw_v) || (raw_sign < 0 && nv <= raw_v) {
                            is_extremum = false;
                            break 'nbr;
                        }
                    }
                }
            }
            if !is_extremum {
                continue;
            }

            let dx = subvoxel_offset(spec.get(ix - 1, iy, iz), raw_v, spec.get(ix + 1, iy, iz));
            let dy = subvoxel_offset(spec.get(ix, iy - 1, iz), raw_v, spec.get(ix, iy + 1, iz));
            let dz = subvoxel_offset(spec.get(ix, iy, iz - 1), raw_v, spec.get(ix, iy, iz + 1));

            let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
            if let Some((lo, hi)) = h_gate {
                if h < lo || h > hi {
                    continue;
                }
            }
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
            if let Some((clo, chi)) = c_bounds {
                if c < clo || c > chi {
                    continue;
                }
            }

            let mut vol = 0.0f64;
            for wz in -1isize..=1 {
                for wy in -1isize..=1 {
                    for wx in -1isize..=1 {
                        let sv = spec.get(
                            (ix as isize + wx) as usize,
                            (iy as isize + wy) as usize,
                            (iz as isize + wz) as usize,
                        );
                        if sv.signum() == raw_v.signum() {
                            vol += (sv.abs()) as f64;
                        }
                    }
                }
            }

            let sign = if invert_polarity || raw_v >= 0.0 {
                1
            } else {
                -1
            };
            out.push(Peak {
                exp: spec.exp.clone(),
                h,
                n,
                c: Some(c),
                sign,
                height: Some((raw_v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            });
        }
    }

    // Suppress near-parent 13C/15N truncation sinc sidelobes within the same (H, N) strip
    if high_dr && out.len() > 1 {
        let min_parent = (40.0 * noise) as f64;
        let half_sinc = 0.5 * (cfg.sinc_frac as f64);
        let mut keep = vec![true; out.len()];
        for i in 0..out.len() {
            let pi = &out[i];
            let Some(ci) = pi.c else { continue };
            let hti = pi.height.unwrap_or(0.0);
            for (j, pj) in out.iter().enumerate() {
                if i == j {
                    continue;
                }
                let Some(cj) = pj.c else { continue };
                let htj = pj.height.unwrap_or(0.0);
                if htj <= hti || htj < min_parent {
                    continue;
                }
                let dh = (pi.h - pj.h).abs();
                let dn = (pi.n - pj.n).abs();
                let dc = (ci - cj).abs();
                // 13C indirect sinc ripple along the same (H, N) strip
                if dh <= 0.025 && dn <= 0.30 && dc > 0.25 && dc <= 6.0 {
                    let env = half_sinc + 0.16 / (1.0 + (dc / 0.85).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
                // 15N indirect sinc ripple at identical (H, C) plane
                if dh <= 0.020 && dc <= 0.25 && dn > 0.20 && dn <= 4.0 && pi.sign == pj.sign {
                    let env = half_sinc + 0.14 / (1.0 + (dn / 0.70).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        let mut filtered = Vec::with_capacity(out.len());
        for (p, k) in out.into_iter().zip(keep) {
            if k {
                filtered.push(p);
            }
        }
        out = filtered;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::AxisPpm;

    fn render_two_peak_hsqc(obs_mhz_h: f64, dh_ppm: f64) -> Spectrum2D {
        // 1024 points across 2.0 ppm => 0.00195 ppm/pt (~0.98 Hz/pt at 500 MHz, ~1.56 Hz/pt at 800 MHz)
        let nx = 1024usize;
        let ny = 64usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 9.5, 7.5, obs_mhz_h);
        let axis_y = AxisPpm::from_bounds("15N", ny, 128.0, 112.0, obs_mhz_h * 0.101329);
        let mut data = vec![0.0f32; nx * ny];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 41) as f32 - 20.0) * 0.05;
        }
        let h_center = 8.40f64;
        let n_center = 120.0f64;
        for &h_pk in &[h_center - 0.5 * dh_ppm, h_center + 0.5 * dh_ppm] {
            let cx = axis_x.ppm_to_index(h_pk);
            let cy = axis_y.ppm_to_index(n_center);
            let ix0 = (cx.floor() as isize - 4).max(0) as usize;
            let ix1 = (cx.ceil() as isize + 4).min(nx as isize - 1) as usize;
            let iy0 = (cy.floor() as isize - 3).max(0) as usize;
            let iy1 = (cy.ceil() as isize + 3).min(ny as isize - 1) as usize;
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    let dx = (ix as f64 - cx) / 1.4;
                    let dy = (iy as f64 - cy) / 1.2;
                    data[iy * nx + ix] += 500.0 * (-0.5 * (dx * dx + dy * dy) as f32).exp();
                }
            }
        }
        Spectrum2D {
            name: "HSQC".to_string(),
            axis_x,
            axis_y,
            data,
        }
    }

    #[test]
    fn test_hsqc_j_doublet_threshold_scales_with_spectrometer_mhz() {
        let cfg = PickerConfig::default(); // max_j_hnha_hz = 9.5 Hz

        // 0.015 ppm separation = 7.5 Hz at 500 MHz (<= 9.5 Hz -> coalesced to 1 centroid)
        let spec_500 = render_two_peak_hsqc(500.0, 0.015);
        let pks_500 = pick_hsqc_2d(&spec_500, &cfg);
        assert_eq!(
            pks_500.len(),
            1,
            "0.015 ppm (7.5 Hz at 500 MHz) must coalesce into 1 doublet centroid, got {pks_500:?}"
        );
        assert!((pks_500[0].h - 8.40).abs() < 0.002);

        // Same 0.015 ppm separation = 12.0 Hz at 800 MHz (> 9.5 Hz -> two distinct resonances!)
        let spec_800 = render_two_peak_hsqc(800.0, 0.015);
        let pks_800 = pick_hsqc_2d(&spec_800, &cfg);
        assert_eq!(
            pks_800.len(),
            2,
            "0.015 ppm (12.0 Hz at 800 MHz) exceeds 9.5 Hz and must remain 2 distinct peaks, got {pks_800:?}"
        );
    }
}
