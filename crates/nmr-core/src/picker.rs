//! Classical baseline 2D and 3D NMR peak picker with real-spectrum artifact hygiene
//! and two-pass experiment-corroborated 3D strip rescue.
//!
//! Provenance: Original clean-room implementation of `docs/specs/02-peak-picking.md` Stages A & A2:
//! - Pass 1 (Stage A): two-pass outlier-trimmed MAD noise estimation (`1.4826 * median(|x - median(x)|)`),
//!   amide `1H` and experiment-specific `13C` spectral window gating, local indirect-axis column/strip
//!   MAD `t1`-ridge thresholding and FT truncation sinc-sidelobe suppression, automatic `180°` phase
//!   polarity detection for single-sign 3D triple-resonance spectra, resolved `^3J_HNHa` `1H` doublet
//!   coalescing, local extrema detection, and 3-point log-Gaussian / parabolic sub-voxel refinement.
//! - Pass 2 (Stage A2, `StripBank3D` + `corroborated_re_pick_3d`): original clean-room two-pass
//!   experiment-guided corroborated re-picking in compact 3D `(H, N)` halo strip cubes, exploiting
//!   complementary triple-resonance coherence-transfer pathways (`HNCA`/`HN(CO)CA`,
//!   `HNCACB`/`CBCA(CO)NH`, `HNCO`/`HN(CA)CO`; Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990,
//!   89, 496–514; Grzesiek & Bax, *J. Am. Chem. Soc.* 1992, 114, 6291–6293) strictly without
//!   sequence or chemical-shift prior inputs.

use std::collections::BTreeMap;

use crate::peaks::Peak;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

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

/// Configuration for Pass-2 experiment-corroborated 3D strip rescue (`corroborated_re_pick_3d`).
///
/// Guard against confirmation bias: Pass 2 accepts no sequence or prior shift inputs and inspects
/// raw 3D strips strictly inside narrow `+-ca_cb_window_ppm` (`0.15 ppm`) or `+-co_window_ppm`
/// (`0.12 ppm`) windows pinned by high-SNR (`>= min_anchor_snr`, default `5.5 sigma`) Pass-1
/// experimental peaks.
#[derive(Clone, Debug)]
pub struct CorroboratedRepickConfig {
    /// Minimum local-strip MAD SNR (`height / strip_mad`) required for a Pass-1 peak to serve
    /// as a corroborating anchor (default `5.5`).
    pub min_anchor_snr: f32,
    /// Conservative local-strip MAD floor (`k * strip_mad`, default `3.8`) for intra-strip
    /// cross-experiment rescue at a single pinned carbon shift within the same `(H, N)` strip.
    pub intra_strip_snr_floor: f32,
    /// Conservative local-strip MAD floor (`k * strip_mad`, default `4.2`) for cross-strip
    /// `0`-peak sequential pool rescue on multi-experiment validated backbone strips.
    pub cross_strip_snr_floor: f32,
    /// Half-width of the `13CA` / `13CB` corroboration window in ppm (default `0.15`).
    pub ca_cb_window_ppm: f64,
    /// Half-width of the `13C'` carbonyl corroboration window in ppm (default `0.12`).
    pub co_window_ppm: f64,
    /// Enable cross-strip sequential pool rescue on `0`-peak channels of validated backbone strips
    /// (default `true`).
    pub enable_cross_strip: bool,
}

impl Default for CorroboratedRepickConfig {
    fn default() -> Self {
        Self {
            min_anchor_snr: 5.5,
            intra_strip_snr_floor: 3.8,
            cross_strip_snr_floor: 4.2,
            ca_cb_window_ppm: 0.15,
            co_window_ppm: 0.12,
            enable_cross_strip: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveStripCol {
    ix: usize,
    in_idx: usize,
    smad: f32,
    smax: f32,
}

/// Compact 1-voxel `(H, N)` halo strip representation of a 3D triple-resonance cube around
/// `hsqc_roots`, retaining bit-exact `3x3x3` local-extremum, sub-voxel refinement, volume, and
/// local-strip MAD statistics in ~1–5% of the memory of a full `Spectrum3D`.
#[derive(Clone, Debug)]
pub struct StripCube3D {
    exp: String,
    axis_h: AxisPpm,
    axis_n: AxisPpm,
    axis_c: AxisPpm,
    noise: f32,
    allow_neg: bool,
    invert_polarity: bool,
    high_dr: bool,
    far_sinc_frac: f32,
    c_bounds: Option<(f64, f64)>,
    ic_store_lo: usize,
    ic_store_len: usize,
    active_strips: Vec<ActiveStripCol>,
    halo_keys: Vec<u64>,
    col_data: Vec<f32>,
}

impl StripCube3D {
    /// Extract a compact `(H, N)` halo strip cube around `hsqc_roots` from a full `Spectrum3D`.
    pub fn from_spectrum_3d(spec: &Spectrum3D, hsqc_roots: &[Peak], cfg: &PickerConfig) -> Self {
        let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
        let noise = estimate_noise_mad(&spec.data);
        let thresh = cfg.snr_threshold * noise;
        let allow_neg = cfg.allow_negative || spec.exp == "HNCACB";
        let c_bounds = carbon_bounds_for_exp(&spec.exp);

        let y_is_n = spec.axis_y.label.contains('N')
            || (spec.axis_y.max_ppm() > 95.0
                && spec.axis_y.min_ppm() < 145.0
                && !spec.axis_z.label.contains('N'));
        let (n_n, n_c) = if y_is_n { (ny, nz) } else { (nz, ny) };
        let axis_n = if y_is_n {
            spec.axis_y.clone()
        } else {
            spec.axis_z.clone()
        };
        let axis_c = if y_is_n {
            spec.axis_z.clone()
        } else {
            spec.axis_y.clone()
        };

        if nx < 3 || ny < 3 || nz < 3 || hsqc_roots.is_empty() {
            return Self {
                exp: spec.exp.clone(),
                axis_h: spec.axis_x.clone(),
                axis_n,
                axis_c,
                noise,
                allow_neg,
                invert_polarity: false,
                high_dr: false,
                far_sinc_frac: (cfg.sinc_frac * 0.4).min(0.012),
                c_bounds,
                ic_store_lo: 0,
                ic_store_len: 0,
                active_strips: Vec::new(),
                halo_keys: Vec::new(),
                col_data: Vec::new(),
            };
        }

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

        let mut raw_active: Vec<(usize, usize)> = Vec::new();
        for in_idx in 1..n_n - 1 {
            let np = axis_n.index_to_ppm(in_idx as f64);
            if !hsqc_roots
                .iter()
                .any(|r| (r.n - np).abs() <= cfg.strip_tol_n)
            {
                continue;
            }
            for ix in 1..nx - 1 {
                let hp = spec.axis_x.index_to_ppm(ix as f64);
                if hsqc_roots.iter().any(|r| {
                    (r.h - hp).abs() <= cfg.strip_tol_h && (r.n - np).abs() <= cfg.strip_tol_n
                }) {
                    raw_active.push((ix, in_idx));
                }
            }
        }

        if active_c.is_empty() || raw_active.is_empty() {
            return Self {
                exp: spec.exp.clone(),
                axis_h: spec.axis_x.clone(),
                axis_n,
                axis_c,
                noise,
                allow_neg,
                invert_polarity: false,
                high_dr: false,
                far_sinc_frac: (cfg.sinc_frac * 0.4).min(0.012),
                c_bounds,
                ic_store_lo: 0,
                ic_store_len: 0,
                active_strips: Vec::new(),
                halo_keys: Vec::new(),
                col_data: Vec::new(),
            };
        }

        let get_voxel = |ix: usize, in_idx: usize, ic: usize| -> f32 {
            if y_is_n {
                spec.get(ix, in_idx, ic)
            } else {
                spec.get(ix, ic, in_idx)
            }
        };

        let mut max_pos = 0.0f32;
        let mut max_neg = 0.0f32;
        for &(ix, in_idx) in &raw_active {
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

        // Compute full-column outlier-trimmed MAD & max identically to Pass 1 before cropping 13C storage
        let mut strip_buf = Vec::with_capacity(n_c);
        let mut scratch = Vec::with_capacity(n_c);
        let mut active_strips = Vec::with_capacity(raw_active.len());
        for &(ix, in_idx) in &raw_active {
            strip_buf.clear();
            for ic in 0..n_c {
                strip_buf.push(get_voxel(ix, in_idx, ic));
            }
            let (smad, smax) = slice_trimmed_mad_and_max(&strip_buf, &mut scratch, noise);
            active_strips.push(ActiveStripCol {
                ix,
                in_idx,
                smad,
                smax,
            });
        }

        let ic_min = *active_c.first().unwrap_or(&1);
        let ic_max = *active_c.last().unwrap_or(&(n_c - 2));
        let ic_store_lo = ic_min.saturating_sub(1);
        let ic_store_hi = (ic_max + 1).min(n_c - 1);
        let ic_store_len = ic_store_hi - ic_store_lo + 1;

        let mut halo_keys = Vec::with_capacity(raw_active.len() * 4);
        for &(ix, in_idx) in &raw_active {
            for dn in -1isize..=1 {
                for dx in -1isize..=1 {
                    let jx = (ix as isize + dx) as u64;
                    let jn = (in_idx as isize + dn) as u64;
                    halo_keys.push((jx << 32) | jn);
                }
            }
        }
        halo_keys.sort_unstable();
        halo_keys.dedup();

        let mut col_data = Vec::with_capacity(halo_keys.len() * ic_store_len);
        for &key in &halo_keys {
            let ix = (key >> 32) as usize;
            let in_idx = (key & 0xFFFF_FFFF) as usize;
            for ic in ic_store_lo..=ic_store_hi {
                col_data.push(get_voxel(ix, in_idx, ic));
            }
        }

        Self {
            exp: spec.exp.clone(),
            axis_h: spec.axis_x.clone(),
            axis_n,
            axis_c,
            noise,
            allow_neg,
            invert_polarity,
            high_dr,
            far_sinc_frac,
            c_bounds,
            ic_store_lo,
            ic_store_len,
            active_strips,
            halo_keys,
            col_data,
        }
    }

    /// Experiment identifier (e.g. `"HNCACB"`, `"CBCACONH"`, `"HNCA"`, `"HNCOCA"`, `"HNCO"`, `"HNCACO"`).
    pub fn exp(&self) -> &str {
        &self.exp
    }

    /// Approximate heap footprint of the extracted strip cube in bytes.
    pub fn byte_size(&self) -> usize {
        self.col_data.len() * std::mem::size_of::<f32>()
            + self.halo_keys.len() * std::mem::size_of::<u64>()
            + self.active_strips.len() * std::mem::size_of::<ActiveStripCol>()
    }

    #[inline]
    fn get(&self, ix: usize, in_idx: usize, ic: usize) -> f32 {
        let key = ((ix as u64) << 32) | (in_idx as u64);
        let slot = self
            .halo_keys
            .binary_search(&key)
            .expect("1-voxel halo column present in StripCube3D");
        self.col_data[slot * self.ic_store_len + (ic - self.ic_store_lo)]
    }

    fn strip_mad_at(&self, h: f64, n: f64) -> f32 {
        if self.active_strips.is_empty() {
            return self.noise;
        }
        let max_ix = (self.axis_h.size.saturating_sub(1)) as f64;
        let max_in = (self.axis_n.size.saturating_sub(1)) as f64;
        let ix = self.axis_h.ppm_to_index(h).round().clamp(0.0, max_ix) as usize;
        let in_idx = self.axis_n.ppm_to_index(n).round().clamp(0.0, max_in) as usize;
        let mut best = self.noise;
        let mut best_d = usize::MAX;
        for s in &self.active_strips {
            let d = s.ix.abs_diff(ix) + s.in_idx.abs_diff(in_idx);
            if d < best_d {
                best_d = d;
                best = s.smad;
                if d == 0 {
                    break;
                }
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn search_corroborated_window(
        &self,
        h_r: f64,
        n_r: f64,
        c_target: f64,
        win_ppm: f64,
        sign_req: Option<i32>,
        min_local_snr: f32,
        existing_exp_peaks: &[&Peak],
        cfg: &PickerConfig,
    ) -> Option<(Peak, f32)> {
        if self.ic_store_len < 3 || self.active_strips.is_empty() {
            return None;
        }
        let step_c = self.axis_c.step_ppm.abs();
        let f0 = self.axis_c.ppm_to_index(c_target - win_ppm - step_c);
        let f1 = self.axis_c.ppm_to_index(c_target + win_ppm + step_c);
        if !f0.is_finite() || !f1.is_finite() {
            return None;
        }
        let i0 = f0.floor() as isize;
        let i1 = f1.ceil() as isize;
        let ic_min_valid = (self.ic_store_lo + 1) as isize;
        let ic_max_valid = (self.ic_store_lo + self.ic_store_len - 2) as isize;
        let lo_isize = i0.min(i1).max(ic_min_valid);
        let hi_isize = i0.max(i1).min(ic_max_valid);
        if lo_isize < 0 || lo_isize > hi_isize {
            return None;
        }
        let ic_lo = lo_isize as usize;
        let ic_hi = hi_isize as usize;

        let snr_floor = min_local_snr.max(3.8);
        let mut best: Option<(Peak, f32, f64)> = None;

        for scol in &self.active_strips {
            let hp_grid = self.axis_h.index_to_ppm(scol.ix as f64);
            let np_grid = self.axis_n.index_to_ppm(scol.in_idx as f64);
            if (hp_grid - h_r).abs() > cfg.strip_tol_h || (np_grid - n_r).abs() > cfg.strip_tol_n {
                continue;
            }
            // Strictly enforce local-strip MAD floor (scol.smad >= self.noise) regardless of high_dr
            let base_thr = snr_floor * scol.smad;
            let loc_thr = if self.high_dr {
                base_thr.max(self.far_sinc_frac * scol.smax)
            } else {
                base_thr
            };

            for ic in ic_lo..=ic_hi {
                let raw_v = self.get(scol.ix, scol.in_idx, ic);
                let eff_v = if self.invert_polarity { -raw_v } else { raw_v };
                if eff_v.abs() < loc_thr || (!self.allow_neg && eff_v < 0.0) {
                    continue;
                }
                let sign = if self.invert_polarity || raw_v >= 0.0 {
                    1
                } else {
                    -1
                };
                if let Some(sr) = sign_req {
                    if sign != sr {
                        continue;
                    }
                }

                let raw_sign = if raw_v >= 0.0 { 1 } else { -1 };
                let mut is_extremum = true;
                'nbr: for dc in -1isize..=1 {
                    for dn in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dn == 0 && dc == 0 {
                                continue;
                            }
                            let nv = self.get(
                                (scol.ix as isize + dx) as usize,
                                (scol.in_idx as isize + dn) as usize,
                                (ic as isize + dc) as usize,
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

                let dx = subvoxel_offset(
                    self.get(scol.ix - 1, scol.in_idx, ic),
                    raw_v,
                    self.get(scol.ix + 1, scol.in_idx, ic),
                );
                let dn = subvoxel_offset(
                    self.get(scol.ix, scol.in_idx - 1, ic),
                    raw_v,
                    self.get(scol.ix, scol.in_idx + 1, ic),
                );
                let dc = subvoxel_offset(
                    self.get(scol.ix, scol.in_idx, ic - 1),
                    raw_v,
                    self.get(scol.ix, scol.in_idx, ic + 1),
                );

                let h = self.axis_h.index_to_ppm(scol.ix as f64 + dx);
                let n = self.axis_n.index_to_ppm(scol.in_idx as f64 + dn);
                let c = self.axis_c.index_to_ppm(ic as f64 + dc);

                if (c - c_target).abs() > win_ppm {
                    continue;
                }
                if let Some((clo, chi)) = self.c_bounds {
                    if c < clo || c > chi {
                        continue;
                    }
                }
                // Require tight centering on the target HSQC root so neighbor bleed is never rescued
                let hd = ((h - h_r) / 0.028).hypot((n - n_r) / 0.28);
                if hd > 1.0 {
                    continue;
                }

                let hti = (raw_v.abs()) as f64;
                if self.high_dr {
                    let min_parent = (40.0 * self.noise) as f64;
                    let half_sinc = 0.5 * (cfg.sinc_frac as f64);
                    let mut is_sidelobe = false;
                    for pj in existing_exp_peaks {
                        let Some(cj) = pj.c else { continue };
                        let htj = pj.height.unwrap_or(0.0);
                        if htj <= hti || htj < min_parent {
                            continue;
                        }
                        let dhp = (h - pj.h).abs();
                        let dnp = (n - pj.n).abs();
                        let dcp = (c - cj).abs();
                        if dhp <= 0.025 && dnp <= 0.30 && dcp > 0.25 && dcp <= 6.0 {
                            let env = half_sinc + 0.16 / (1.0 + (dcp / 0.85).powi(2));
                            if hti < htj * env {
                                is_sidelobe = true;
                                break;
                            }
                        }
                        if dhp <= 0.020
                            && dcp <= 0.25
                            && dnp > 0.20
                            && dnp <= 4.0
                            && sign == pj.sign
                        {
                            let env = half_sinc + 0.14 / (1.0 + (dnp / 0.70).powi(2));
                            if hti < htj * env {
                                is_sidelobe = true;
                                break;
                            }
                        }
                        // Cross-sign F1 truncation sinc sidelobe guard in phase-sensitive HNCACB
                        if self.allow_neg
                            && sign != pj.sign
                            && dhp <= 0.035
                            && dnp <= 0.35
                            && dcp <= 0.65
                            && hti < 0.16 * htj
                        {
                            is_sidelobe = true;
                            break;
                        }
                    }
                    if is_sidelobe {
                        continue;
                    }
                }

                // Skip if Pass 1 already picked this extremum
                if existing_exp_peaks.iter().any(|pj| {
                    pj.sign == sign
                        && (pj.h - h).abs() <= 0.030
                        && (pj.n - n).abs() <= 0.30
                        && (pj.c.unwrap_or(0.0) - c).abs() <= 0.20
                }) {
                    continue;
                }

                let mut vol = 0.0f64;
                for wc in -1isize..=1 {
                    for wn in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = self.get(
                                (scol.ix as isize + wx) as usize,
                                (scol.in_idx as isize + wn) as usize,
                                (ic as isize + wc) as usize,
                            );
                            if sv.signum() == raw_v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                let lsnr = eff_v.abs() / scol.smad.max(1e-12);
                let rank = (lsnr as f64) * (-0.35 * hd * hd).exp();
                if best.as_ref().is_none_or(|b| rank > b.2) {
                    best = Some((
                        Peak {
                            exp: self.exp.clone(),
                            h,
                            n,
                            c: Some(c),
                            sign,
                            height: Some(hti),
                            volume: Some(vol),
                            h2: None,
                        },
                        lsnr,
                        rank,
                    ));
                }
            }
        }
        best.map(|(pk, lsnr, _)| (pk, lsnr))
    }
}

/// Multi-experiment bank of compact 3D strip cubes (`StripCube3D`) accumulated during streaming
/// Pass-1 3D picking so Pass 2 (`corroborated_re_pick_3d`) can inspect all experiments jointly
/// without keeping full 3D cubes (`~130–540 MB` each) in memory simultaneously.
#[derive(Clone, Debug, Default)]
pub struct StripBank3D {
    cubes: BTreeMap<String, StripCube3D>,
}

impl StripBank3D {
    /// Create an empty multi-experiment strip bank.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a pre-extracted `StripCube3D` keyed by its experiment name.
    pub fn insert_cube(&mut self, cube: StripCube3D) {
        self.cubes.insert(cube.exp.clone(), cube);
    }

    /// Run Pass-1 `pick_triple_3d` on `spec` and store its compact `StripCube3D` in this bank.
    pub fn pick_and_record_3d(
        &mut self,
        spec: &Spectrum3D,
        hsqc_roots: &[Peak],
        cfg: &PickerConfig,
    ) -> Vec<Peak> {
        let p1 = pick_triple_3d(spec, hsqc_roots, cfg);
        if !hsqc_roots.is_empty() {
            self.insert_cube(StripCube3D::from_spectrum_3d(spec, hsqc_roots, cfg));
        }
        p1
    }

    /// Look up the strip cube for an experiment name (e.g. `"HNCACB"`).
    pub fn get(&self, exp: &str) -> Option<&StripCube3D> {
        self.cubes.get(exp)
    }

    /// Number of 3D experiments stored in this bank.
    pub fn len(&self) -> usize {
        self.cubes.len()
    }

    /// Whether no 3D strip cubes have been recorded.
    pub fn is_empty(&self) -> bool {
        self.cubes.is_empty()
    }

    /// Total heap memory used across all stored strip cubes in bytes.
    pub fn byte_size(&self) -> usize {
        self.cubes.values().map(StripCube3D::byte_size).sum()
    }
}

/// Pass-2 experiment-corroborated 3D strip rescue (`docs/specs/02-peak-picking.md` Stage A2).
///
/// Returns ONLY the newly rescued Pass-2 3D peaks (caller may extend `pass1_peaks` with the result).
/// Never uses amino-acid sequence or chemical-shift priors; every search window is pinned by a
/// high-SNR (`>= repick_cfg.min_anchor_snr`) Pass-1 experimental peak.
pub fn corroborated_re_pick_3d(
    bank: &StripBank3D,
    hsqc_roots: &[Peak],
    pass1_peaks: &[Peak],
    cfg: &PickerConfig,
    repick_cfg: &CorroboratedRepickConfig,
) -> Vec<Peak> {
    if bank.is_empty() || hsqc_roots.is_empty() {
        return Vec::new();
    }

    let tol_h = 0.030;
    let tol_n = 0.30;
    let anchor_min_snr = repick_cfg.min_anchor_snr.max(4.5);
    let intra_snr = repick_cfg.intra_strip_snr_floor.max(3.8);
    let cross_snr = repick_cfg.cross_strip_snr_floor.max(3.8);
    let ca_cb_win = repick_cfg.ca_cb_window_ppm.clamp(0.05, 0.25);
    let co_win = repick_cfg.co_window_ppm.clamp(0.04, 0.20);
    let min_amide_h = cfg
        .amide_h_window
        .map(|(a, b)| a.min(b).max(6.0))
        .unwrap_or(0.0);

    // Identify paired sidechain NH2 doublets (`|dN| <= 0.12 ppm`, `0.30 <= |dH| <= 1.20 ppm`)
    let mut is_nh2_pair = vec![false; hsqc_roots.len()];
    for i in 0..hsqc_roots.len() {
        for j in (i + 1)..hsqc_roots.len() {
            let dn = (hsqc_roots[i].n - hsqc_roots[j].n).abs();
            let dh = (hsqc_roots[i].h - hsqc_roots[j].h).abs();
            if dn <= 0.12 && (0.30..=1.20).contains(&dh) {
                is_nh2_pair[i] = true;
                is_nh2_pair[j] = true;
            }
        }
    }

    let mut by_exp: BTreeMap<String, Vec<&Peak>> = BTreeMap::new();
    for p in pass1_peaks {
        if p.exp != "HSQC" && p.c.is_some() {
            by_exp.entry(p.exp.clone()).or_default().push(p);
        }
    }

    #[derive(Clone)]
    struct RootPeak {
        exp: String,
        c: f64,
        sign: i32,
        lsnr: f32,
    }

    let mut root_peaks: Vec<Vec<RootPeak>> = vec![Vec::new(); hsqc_roots.len()];
    for p in pass1_peaks {
        let Some(c) = p.c else { continue };
        if p.exp == "HSQC" {
            continue;
        }
        let Some(cube) = bank.get(&p.exp) else {
            continue;
        };
        let smad = cube.strip_mad_at(p.h, p.n);
        let lsnr = (p.height.unwrap_or(0.0).abs() as f32) / smad.max(1e-12);
        for (ri, r) in hsqc_roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol_h).hypot((r.n - p.n) / tol_n);
            if d <= 1.0 {
                root_peaks[ri].push(RootPeak {
                    exp: p.exp.clone(),
                    c,
                    sign: p.sign,
                    lsnr,
                });
            }
        }
    }

    let has_hncoca_cube = bank.get("HNCOCA").is_some();
    let mut rescued: Vec<Peak> = Vec::new();

    let has_near = |ri: usize, exp: &str, c0: f64, sgn: Option<i32>, dc: f64, extra: &[Peak]| {
        root_peaks[ri]
            .iter()
            .any(|rp| rp.exp == exp && sgn.is_none_or(|s| rp.sign == s) && (rp.c - c0).abs() <= dc)
            || extra.iter().any(|ep| {
                ep.exp == exp
                    && sgn.is_none_or(|s| ep.sign == s)
                    && ((ep.h - hsqc_roots[ri].h) / tol_h).hypot((ep.n - hsqc_roots[ri].n) / tol_n)
                        <= 1.0
                    && (ep.c.unwrap_or(0.0) - c0).abs() <= dc
            })
    };

    // Pass 2A: Intra-strip cross-experiment corroboration at the same HSQC root
    for (ri, r) in hsqc_roots.iter().enumerate() {
        if r.h < min_amide_h
            || (is_nh2_pair[ri]
                && !root_peaks[ri].iter().any(|rp| {
                    (rp.exp == "HNCA" || rp.exp == "HNCACB") && rp.lsnr >= anchor_min_snr
                }))
        {
            continue;
        }

        let n_hnca = root_peaks[ri].iter().filter(|rp| rp.exp == "HNCA").count();

        // 1a. HNCOCA (>= anchor_min_snr) -> HNCACB(+1), and -> HNCA(+1) when strip has 0 HNCA peaks
        let hncoca_anchors: Vec<f64> = root_peaks[ri]
            .iter()
            .filter(|rp| {
                rp.exp == "HNCOCA" && rp.lsnr >= anchor_min_snr && (35.0..=76.0).contains(&rp.c)
            })
            .map(|rp| rp.c)
            .collect();
        for c0 in hncoca_anchors {
            if let Some(cube) = bank.get("HNCACB") {
                if !has_near(ri, "HNCACB", c0, Some(1), 0.30, &rescued) {
                    let existing = by_exp.get("HNCACB").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c0,
                        ca_cb_win,
                        Some(1),
                        intra_snr,
                        existing,
                        cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
            if n_hnca == 0 {
                if let Some(cube) = bank.get("HNCA") {
                    if !has_near(ri, "HNCA", c0, Some(1), 0.30, &rescued) {
                        let existing = by_exp.get("HNCA").map(Vec::as_slice).unwrap_or(&[]);
                        if let Some((pk, _)) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            c0,
                            ca_cb_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            rescued.push(pk);
                        }
                    }
                }
            }
        }

        // 1b. CBCACONH (>= anchor_min_snr) -> HNCACB (+CA / -CB typing), and -> HNCA when HNCOCA is absent or 0 HNCA
        let cbca_anchors: Vec<f64> = root_peaks[ri]
            .iter()
            .filter(|rp| rp.exp == "CBCACONH" && rp.lsnr >= anchor_min_snr)
            .map(|rp| rp.c)
            .collect();
        for c0 in cbca_anchors {
            if let Some(cube) = bank.get("HNCACB") {
                if !has_near(ri, "HNCACB", c0, None, 0.30, &rescued) {
                    let hncoca_match = root_peaks[ri]
                        .iter()
                        .any(|rp| rp.exp == "HNCOCA" && (rp.c - c0).abs() <= 0.25);
                    let sign_req = if hncoca_match || c0 > 73.0 {
                        Some(1)
                    } else if c0 < 38.0 {
                        Some(-1)
                    } else {
                        None
                    };
                    let existing = by_exp.get("HNCACB").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h, r.n, c0, ca_cb_win, sign_req, intra_snr, existing, cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
            if (!has_hncoca_cube || n_hnca == 0) && (38.5..=76.0).contains(&c0) {
                if let Some(cube) = bank.get("HNCA") {
                    if !has_near(ri, "HNCA", c0, Some(1), 0.30, &rescued) {
                        let existing = by_exp.get("HNCA").map(Vec::as_slice).unwrap_or(&[]);
                        if let Some((pk, _)) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            c0,
                            ca_cb_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            rescued.push(pk);
                        }
                    }
                }
            }
        }

        // 1c. HNCA / HNCACB(+1) (>= anchor_min_snr) -> HNCOCA when strip has 0 HNCOCA peaks
        if !root_peaks[ri].iter().any(|rp| rp.exp == "HNCOCA") {
            if let Some(cube) = bank.get("HNCOCA") {
                let existing = by_exp.get("HNCOCA").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                let floor_ca_to_im1 = intra_snr.max(4.0);
                for rp in &root_peaks[ri] {
                    if (rp.exp == "HNCA" || (rp.exp == "HNCACB" && rp.sign > 0))
                        && rp.lsnr >= anchor_min_snr
                        && (38.0..=76.0).contains(&rp.c)
                    {
                        if let Some(cand) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            rp.c,
                            ca_cb_win,
                            Some(1),
                            floor_ca_to_im1,
                            existing,
                            cfg,
                        ) {
                            if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                                best_cand = Some(cand);
                            }
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }

        // 2a. HNCO (>= anchor_min_snr) -> HNCACO (corroborates C'(i-1) in HNCACO)
        if let Some(cube) = bank.get("HNCACO") {
            for rp in &root_peaks[ri] {
                if rp.exp == "HNCO"
                    && rp.lsnr >= anchor_min_snr
                    && !has_near(ri, "HNCACO", rp.c, Some(1), 0.18, &rescued)
                {
                    let existing = by_exp.get("HNCACO").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        rp.c,
                        co_win,
                        Some(1),
                        intra_snr,
                        existing,
                        cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
        }

        // 2b. HNCACO (>= anchor_min_snr) -> HNCO when strip has 0 HNCO peaks
        if !root_peaks[ri].iter().any(|rp| rp.exp == "HNCO") {
            if let Some(cube) = bank.get("HNCO") {
                let existing = by_exp.get("HNCO").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for rp in &root_peaks[ri] {
                    if rp.exp == "HNCACO" && rp.lsnr >= anchor_min_snr {
                        if let Some(cand) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            rp.c,
                            co_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                                best_cand = Some(cand);
                            }
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }
    }

    if !repick_cfg.enable_cross_strip {
        return rescued;
    }

    // Build global high-SNR experimental pools strictly from Pass-1 non-NH2 backbone peaks
    let mut pool_co_im1: Vec<f64> = Vec::new();
    let mut pool_ca_im1: Vec<f64> = Vec::new();
    for (ri, rps) in root_peaks.iter().enumerate() {
        if hsqc_roots[ri].h < min_amide_h || is_nh2_pair[ri] {
            continue;
        }
        for rp in rps {
            if rp.lsnr < anchor_min_snr {
                continue;
            }
            match rp.exp.as_str() {
                "HNCO" => pool_co_im1.push(rp.c),
                "HNCOCA" => pool_ca_im1.push(rp.c),
                "CBCACONH" if (38.5..=76.0).contains(&rp.c) => {
                    let is_cb = rps
                        .iter()
                        .any(|q| q.exp == "HNCACB" && q.sign < 0 && (q.c - rp.c).abs() <= 0.25);
                    if !is_cb {
                        pool_ca_im1.push(rp.c);
                    }
                }
                _ => {}
            }
        }
    }
    let dedup_shifts = |mut v: Vec<f64>, min_sep: f64| -> Vec<f64> {
        v.sort_by(|a, b| a.total_cmp(b));
        let mut out: Vec<f64> = Vec::new();
        for x in v {
            if out.last().is_none_or(|&last| (x - last) > min_sep) {
                out.push(x);
            }
        }
        out
    };
    let pool_co_im1 = dedup_shifts(pool_co_im1, 0.05);
    let pool_ca_im1 = dedup_shifts(pool_ca_im1, 0.06);

    // Pass 2B: Cross-strip 0-peak rescue on multi-experiment validated backbone strips
    for (ri, r) in hsqc_roots.iter().enumerate() {
        if r.h < min_amide_h || is_nh2_pair[ri] {
            continue;
        }
        let mut distinct_exps: Vec<&str> = root_peaks[ri]
            .iter()
            .filter(|rp| rp.lsnr >= anchor_min_snr)
            .map(|rp| rp.exp.as_str())
            .collect();
        distinct_exps.sort_unstable();
        distinct_exps.dedup();
        let has_intra_backbone =
            distinct_exps.contains(&"HNCA") || distinct_exps.contains(&"HNCACB");
        if distinct_exps.len() < 2 || (!has_intra_backbone && !distinct_exps.contains(&"HNCO")) {
            continue;
        }

        // Cross-strip CA(i) rescue ONLY when HNCA or HNCACB(+1) has 0 positive peaks
        for target_exp in ["HNCA", "HNCACB"] {
            let Some(cube) = bank.get(target_exp) else {
                continue;
            };
            let n_pos = root_peaks[ri]
                .iter()
                .filter(|rp| rp.exp == target_exp && rp.sign > 0)
                .count()
                + rescued
                    .iter()
                    .filter(|ep| {
                        ep.exp == target_exp
                            && ep.sign > 0
                            && ((ep.h - r.h) / tol_h).hypot((ep.n - r.n) / tol_n) <= 1.0
                    })
                    .count();
            if n_pos == 0 {
                let existing = by_exp.get(target_exp).map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for &c_cand in &pool_ca_im1 {
                    if let Some(cand) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c_cand,
                        ca_cb_win,
                        Some(1),
                        cross_snr,
                        existing,
                        cfg,
                    ) {
                        if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                            best_cand = Some(cand);
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }

        // Cross-strip HNCO rescue ONLY when strip has 0 HNCO peaks and >= 2 aliphatic backbone anchors
        if has_intra_backbone
            && !root_peaks[ri].iter().any(|rp| rp.exp == "HNCO")
            && !rescued.iter().any(|ep| {
                ep.exp == "HNCO" && ((ep.h - r.h) / tol_h).hypot((ep.n - r.n) / tol_n) <= 1.0
            })
        {
            if let Some(cube) = bank.get("HNCO") {
                let existing = by_exp.get("HNCO").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for &c_cand in &pool_co_im1 {
                    if let Some(cand) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c_cand,
                        co_win,
                        Some(1),
                        cross_snr,
                        existing,
                        cfg,
                    ) {
                        if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                            best_cand = Some(cand);
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }
    }

    rescued
}

/// Convenience two-pass 3D triple-resonance peak picker across a bundle of in-memory 3D cubes:
/// runs Pass 1 (`pick_triple_3d`) on every cube while recording a compact `StripBank3D`, then
/// appends Pass-2 corroborated rescues (`corroborated_re_pick_3d`).
pub fn pick_triple_3d_two_pass(
    spectra: &[&Spectrum3D],
    hsqc_roots: &[Peak],
    cfg: &PickerConfig,
    repick_cfg: &CorroboratedRepickConfig,
) -> Vec<Peak> {
    let mut bank = StripBank3D::new();
    let mut peaks = Vec::new();
    for spec in spectra {
        peaks.extend(bank.pick_and_record_3d(spec, hsqc_roots, cfg));
    }
    let rescued = corroborated_re_pick_3d(&bank, hsqc_roots, &peaks, cfg, repick_cfg);
    peaks.extend(rescued);
    peaks
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
