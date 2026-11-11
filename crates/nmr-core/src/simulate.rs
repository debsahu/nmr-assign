//! Simulate triple-resonance peak lists and synthetic 2D/3D NMR spectra from a reference assignment.
//!
//! Provenance: Original method implementing `docs/specs/05-benchmark.md` and the synthetic
//! spectrum renderer planned in `docs/specs/02-peak-picking.md`.

use crate::nmrstar::Reference;
use crate::peaks::{experiment_rules, Peak, EXPERIMENT_NAMES};
use crate::rng::Rng;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub experiments: Vec<String>,
    pub noise_h: f64,
    pub noise_n: f64,
    pub noise_c: f64,
    pub p_missing_peak: f64,
    pub p_missing_root: f64,
    pub spurious_rate: f64,
    pub sidechain_nh2: bool,
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            experiments: EXPERIMENT_NAMES.iter().map(|s| (*s).to_string()).collect(),
            noise_h: 0.008,
            noise_n: 0.06,
            noise_c: 0.04,
            p_missing_peak: 0.06,
            p_missing_root: 0.03,
            spurious_rate: 0.04,
            sidechain_nh2: true,
            seed: 0,
        }
    }
}

fn carbon_range(atom: &str) -> (f64, f64) {
    match atom {
        "CA" => (42.0, 68.0),
        "CB" => (15.0, 72.0),
        "C" => (170.0, 181.0),
        _ => (15.0, 181.0),
    }
}

/// Simulate blind peak lists from `ref_data`. Returns `(peaks, truth_per_hsqc_root)`.
pub fn simulate(ref_data: &Reference, cfg: &SimConfig) -> (Vec<Peak>, Vec<i32>) {
    let mut rng = Rng::new(cfg.seed);
    let mut peaks: Vec<Peak> = Vec::new();
    let mut truth: Vec<i32> = Vec::new();
    let mut roots: Vec<(f64, f64)> = Vec::new();

    for (k, res) in ref_data.sequence.iter().enumerate() {
        let h = ref_data.shifts.get(&(k, "H".to_string())).copied();
        let n = ref_data.shifts.get(&(k, "N".to_string())).copied();
        let (Some(h_val), Some(n_val)) = (h, n) else { continue };
        if k == 0 || res == "PRO" {
            continue;
        }
        if rng.f64() < cfg.p_missing_root {
            continue;
        }
        roots.push((h_val, n_val));
        let hj = h_val + rng.normal(0.0, cfg.noise_h);
        let nj = n_val + rng.normal(0.0, cfg.noise_n);
        peaks.push(Peak::hsqc(hj, nj));
        truth.push(k as i32);

        for exp in &cfg.experiments {
            let Some(rules) = experiment_rules(exp) else { continue };
            for &(atom, off, sign) in rules {
                let target = (k as isize) + (off as isize);
                if target < 0 {
                    continue;
                }
                let Some(&c_val) = ref_data.shifts.get(&(target as usize, atom.to_string())) else {
                    continue;
                };
                if rng.f64() < cfg.p_missing_peak {
                    continue;
                }
                let ph = h_val + rng.normal(0.0, cfg.noise_h);
                let pn = n_val + rng.normal(0.0, cfg.noise_n);
                let pc = c_val + rng.normal(0.0, cfg.noise_c);
                peaks.push(Peak::triple(exp.clone(), ph, pn, pc, sign));
            }
        }
    }

    if cfg.sidechain_nh2 {
        for res in &ref_data.sequence {
            if res == "ASN" || res == "GLN" {
                let n_sc = rng.normal(112.5, 1.0);
                for h_mean in [6.85, 7.55] {
                    let h_sc = rng.normal(h_mean, 0.1);
                    peaks.push(Peak::hsqc(h_sc, n_sc));
                    truth.push(-1);
                }
            }
        }
    }

    if !roots.is_empty() {
        for exp in &cfg.experiments {
            if exp == "HSQC" {
                continue;
            }
            let Some(rules) = experiment_rules(exp) else { continue };
            if rules.is_empty() {
                continue;
            }
            let n_real = peaks.iter().filter(|p| &p.exp == exp).count();
            let n_spur = rng.poisson(cfg.spurious_rate * (n_real as f64));
            for _ in 0..n_spur {
                let (rh, rn) = roots[rng.usize(roots.len())];
                let &(atom, _, sign) = &rules[rng.usize(rules.len())];
                let (lo, hi) = carbon_range(atom);
                let ph = rh + rng.normal(0.0, cfg.noise_h);
                let pn = rn + rng.normal(0.0, cfg.noise_n);
                let pc = rng.uniform(lo, hi);
                peaks.push(Peak::triple(exp.clone(), ph, pn, pc, sign));
            }
        }
    }

    let hsqc_indices: Vec<usize> = peaks
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (p.exp == "HSQC").then_some(i))
        .collect();
    let perm = rng.permutation(hsqc_indices.len());
    let hsqc_peaks: Vec<Peak> = perm.iter().map(|&j| peaks[hsqc_indices[j]].clone()).collect();
    let shuffled_truth: Vec<i32> = perm.iter().map(|&j| truth[j]).collect();

    let mut others: Vec<Peak> = peaks.into_iter().filter(|p| p.exp != "HSQC").collect();
    rng.shuffle(&mut others);

    let mut combined = hsqc_peaks;
    combined.extend(others);
    (combined, shuffled_truth)
}

#[derive(Clone, Debug)]
pub struct SpectrumRenderConfig {
    pub nx_h: usize,
    pub ny_n: usize,
    pub nz_c: usize,
    pub h_high_ppm: f64,
    pub h_low_ppm: f64,
    pub n_high_ppm: f64,
    pub n_low_ppm: f64,
    pub lw_h_ppm: f64,
    pub lw_n_ppm: f64,
    pub lw_c_ppm: f64,
    /// Thermal baseline noise standard deviation (peak amplitude = `100.0`).
    pub noise_level: f32,
    /// Fraction Gaussian vs Lorentzian lineshape (`0.0` = pure Lorentzian, `1.0` = pure Gaussian).
    pub gaussian_mix: f64,
    pub seed: u64,
}

impl Default for SpectrumRenderConfig {
    fn default() -> Self {
        Self {
            nx_h: 512,
            ny_n: 256,
            nz_c: 128,
            h_high_ppm: 10.6,
            h_low_ppm: 5.8,
            n_high_ppm: 136.0,
            n_low_ppm: 102.0,
            lw_h_ppm: 0.022,
            lw_n_ppm: 0.22,
            lw_c_ppm: 0.28,
            noise_level: 1.5,
            gaussian_mix: 0.65,
            seed: 42,
        }
    }
}

#[inline]
fn lineshape_1d(delta_ppm: f64, lw_ppm: f64, gauss_mix: f64) -> f64 {
    let u = delta_ppm / (0.5 * lw_ppm.max(1e-6));
    let gauss = (-0.693_147_18 * u * u).exp();
    let lorentz = 1.0 / (1.0 + u * u);
    gauss_mix * gauss + (1.0 - gauss_mix) * lorentz
}

/// Render a synthetic 2D `1H-15N` HSQC spectrum from peaks or a reference assignment.
pub fn render_hsqc_2d(peaks: &[Peak], cfg: &SpectrumRenderConfig) -> Spectrum2D {
    let axis_x = AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, 60.81);
    let mut rng = Rng::new(cfg.seed);

    let mut data = vec![0.0f32; cfg.ny_n * cfg.nx_h];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.5 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let rad_y = ((3.5 * cfg.lw_n_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(3);

    for p in peaks.iter().filter(|p| p.exp == "HSQC") {
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(p.n).round() as isize;
        let amp = 100.0 * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;

        for iy in y0..=y1 {
            let py = axis_y.index_to_ppm(iy as f64);
            let wy = lineshape_1d(py - p.n, cfg.lw_n_ppm, cfg.gaussian_mix);
            for ix in x0..=x1 {
                let px = axis_x.index_to_ppm(ix as f64);
                let wx = lineshape_1d(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix);
                data[iy * cfg.nx_h + ix] += (amp * wx * wy) as f32;
            }
        }
    }

    Spectrum2D {
        name: "Synthetic 1H-15N HSQC".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

/// Render a synthetic 3D triple-resonance spectrum (`HNCACB`, `HNCA`, `HNCO`, etc.) from peaks.
pub fn render_triple_3d(exp: &str, peaks: &[Peak], cfg: &SpectrumRenderConfig) -> Spectrum3D {
    let (c_high, c_low) = if exp == "HNCO" || exp == "HNCACO" {
        (183.0, 168.0)
    } else if exp == "HNCACB" || exp == "CBCACONH" {
        (75.0, 14.0)
    } else {
        (70.0, 40.0)
    };

    let axis_x = AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, 60.81);
    let axis_z = AxisPpm::from_bounds("13C", cfg.nz_c, c_high, c_low, 150.90);
    let mut rng = Rng::new(cfg.seed.wrapping_add(1337));

    let mut data = vec![0.0f32; cfg.nz_c * cfg.ny_n * cfg.nx_h];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.0 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(2);
    let rad_y = ((3.0 * cfg.lw_n_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(2);
    let rad_z = ((3.0 * cfg.lw_c_ppm / axis_z.step_ppm.abs()).ceil() as isize).max(2);

    for p in peaks.iter().filter(|p| p.exp == exp) {
        let Some(c_ppm) = p.c else { continue };
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(p.n).round() as isize;
        let cz = axis_z.ppm_to_index(c_ppm).round() as isize;
        let amp = 90.0 * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
        let z0 = (cz - rad_z).max(0) as usize;
        let z1 = (cz + rad_z).min(cfg.nz_c as isize - 1).max(0) as usize;

        for iz in z0..=z1 {
            let pz = axis_z.index_to_ppm(iz as f64);
            let wz = lineshape_1d(pz - c_ppm, cfg.lw_c_ppm, cfg.gaussian_mix);
            for iy in y0..=y1 {
                let py = axis_y.index_to_ppm(iy as f64);
                let wy = lineshape_1d(py - p.n, cfg.lw_n_ppm, cfg.gaussian_mix);
                for ix in x0..=x1 {
                    let px = axis_x.index_to_ppm(ix as f64);
                    let wx = lineshape_1d(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix);
                    data[(iz * cfg.ny_n + iy) * cfg.nx_h + ix] += (amp * wx * wy * wz) as f32;
                }
            }
        }
    }

    Spectrum3D {
        name: format!("Synthetic 3D {exp}"),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}
