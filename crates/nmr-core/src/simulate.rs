//! Simulate triple-resonance peak lists and synthetic 2D/3D NMR spectra from a reference assignment.
//!
//! Provenance: Clean-room original method implementing `docs/specs/05-benchmark.md` and the
//! synthetic volumetric spectrum renderer in `docs/specs/02-peak-picking.md`:
//! - Kay, L. E., Ikura, M., Tschudin, R. & Bax, A., *J. Magn. Reson.* 1990, 89, 496–514
//!   (`1J_NCa` vs `2J_NCa` intraresidue/sequential transfer amplitude ratios).
//! - Grzesiek, S. & Bax, A., *J. Magn. Reson.* 1992, 99, 201–207; Wittekind, M. & Mueller, L.,
//!   *J. Magn. Reson. B* 1993, 101, 201–205 (`CBCA(CO)NH` and `HNCACB` transfer signs/amplitudes).
//! - Hoch, J. C. & Stern, A. S., *NMR Data Processing*, Wiley-Liss, 1996, Ch. 3–4 (finite-`TD`
//!   indirect-dimension truncation `sinc` sidelobes damped by cosine-bell / exponential apodization).
//! - Mehlkopf, A. F., Korbee, D., Tiggelman, T. A. & Freeman, R., *J. Magn. Reson.* 1984, 58,
//!   315–323 (multiplicative indirect-dimension `t1` ridge noise proportional to signal intensity).
//! - Wishart, D. S. et al., *J. Biomol. NMR* 1995, 6, 135–140 (DSS vs TMS chemical-shift
//!   referencing offsets across `1H`, `15N`, and `13C`).

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI, TAU};

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
    pub include_intensities: bool,
    pub seed: u64,
    /// Systematic `1H` chemical-shift referencing offset (ppm) added to simulated peaks.
    pub ref_offset_h: f64,
    /// Systematic `15N` chemical-shift referencing offset (ppm) added to simulated peaks.
    pub ref_offset_n: f64,
    /// Systematic `13CA` referencing offset (ppm, e.g. `-2.75` for TMS vs DSS).
    pub ref_offset_ca: f64,
    /// Systematic `13CB` referencing offset (ppm, e.g. `-2.75` for TMS vs DSS).
    pub ref_offset_cb: f64,
    /// Systematic `13C'` carbonyl referencing offset (ppm, e.g. `-2.75` for TMS vs DSS).
    pub ref_offset_c: f64,
    /// Log-normal per-residue conformational-exchange / relaxation intensity dispersion (`0.0` = none).
    pub exchange_sigma: f64,
    /// Relative amplitude of F1/F2 truncation `sinc` sidelobe false peaks (`0.0` = none).
    pub sinc_sidelobe_frac: f64,
    /// Rate of `t1`-ridge spurious peaks along intense parent amide columns/strips (`0.0` = none).
    pub t1_ridge_spurious_rate: f64,
    /// Emit realistic Asn/Gln sidechain `NH2` 3D peaks (`HNCO`/`CBCACONH`) and realistic `15N` pair jitter.
    pub nh2_sidechain_3d: bool,
    /// Number of residual water-suppression (`~4.77 ppm`) and upfield (`< 4.3 ppm`) spurious HSQC roots.
    pub water_artifact_roots: usize,
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
            include_intensities: true,
            seed: 0,
            ref_offset_h: 0.0,
            ref_offset_n: 0.0,
            ref_offset_ca: 0.0,
            ref_offset_cb: 0.0,
            ref_offset_c: 0.0,
            exchange_sigma: 0.0,
            sinc_sidelobe_frac: 0.0,
            t1_ridge_spurious_rate: 0.0,
            nh2_sidechain_3d: false,
            water_artifact_roots: 0,
        }
    }
}

impl SimConfig {
    /// Realistic 5-spectrum Bruker triple-resonance configuration (`HSQC`, `HNCACB`, `CBCACONH`,
    /// `HNCO`, `HNCACO` — no `HNCA` or `HNCOCA`), with `-2.75 ppm` TMS-vs-DSS `13C` referencing
    /// offset, wide log-normal conformational-exchange intensity dispersion, `~4%` exchange-broadened
    /// missing backbone amides, paired Asn/Gln `NH2` sidechain doublets, F1/F2 `sinc` sidelobes,
    /// indirect `t1`-ridge spurious peaks, and residual water/aliphatic artifacts.
    pub fn realistic_bruker() -> Self {
        Self {
            experiments: vec![
                "HSQC".to_string(),
                "HNCACB".to_string(),
                "CBCACONH".to_string(),
                "HNCO".to_string(),
                "HNCACO".to_string(),
            ],
            noise_h: 0.010,
            noise_n: 0.07,
            noise_c: 0.05,
            p_missing_peak: 0.07,
            p_missing_root: 0.04,
            spurious_rate: 0.05,
            sidechain_nh2: true,
            include_intensities: true,
            seed: 0,
            ref_offset_h: -0.02,
            ref_offset_n: 0.25,
            ref_offset_ca: -2.75,
            ref_offset_cb: -2.75,
            ref_offset_c: -2.75,
            exchange_sigma: 0.60,
            sinc_sidelobe_frac: 0.042,
            t1_ridge_spurious_rate: 0.35,
            nh2_sidechain_3d: true,
            water_artifact_roots: 4,
        }
    }

    #[inline]
    fn has_ref_offsets(&self) -> bool {
        self.ref_offset_h != 0.0
            || self.ref_offset_n != 0.0
            || self.ref_offset_ca != 0.0
            || self.ref_offset_cb != 0.0
            || self.ref_offset_c != 0.0
    }

    #[inline]
    fn carbon_offset(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.ref_offset_ca,
            "CB" => self.ref_offset_cb,
            "C" => self.ref_offset_c,
            _ => 0.0,
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

fn uncropped_bruker_carbon_range(exp: &str) -> (f64, f64) {
    if exp == "HNCO" || exp == "HNCACO" {
        (165.0, 187.0)
    } else {
        (1.63, 76.5)
    }
}

/// Simulate blind peak lists from `ref_data`. Returns `(peaks, truth_per_hsqc_root)`.
pub fn simulate(ref_data: &Reference, cfg: &SimConfig) -> (Vec<Peak>, Vec<i32>) {
    let mut rng = Rng::new(cfg.seed);
    // Separate sub-stream for peak intensities so coordinate/missing/spurious PRNG draws remain invariant.
    let mut irng = Rng::new(cfg.seed ^ 0x9E37_79B9_7F4A_7C15);
    // Dedicated third sub-stream for optional realistic Bruker physics so `SimConfig::default()`
    // preserves exact bit-for-bit PRNG parity across all existing tests and benchmarks.
    let mut arng = Rng::new(cfg.seed ^ 0xD1B5_4A32_D192_ED03);

    let mut peaks: Vec<Peak> = Vec::new();
    let mut truth: Vec<i32> = Vec::new();
    let mut roots: Vec<(f64, f64)> = Vec::new();
    let mut root_amps: Vec<(f64, f64, f64)> = Vec::new();
    let mut parent_3d_peaks: Vec<(String, f64, f64, f64, i32, f64)> = Vec::new();
    let apply_offsets = cfg.has_ref_offsets();

    for (k, res) in ref_data.sequence.iter().enumerate() {
        let h = ref_data.shifts.get(&(k, "H".to_string())).copied();
        let n = ref_data.shifts.get(&(k, "N".to_string())).copied();
        let (Some(h_raw), Some(n_raw)) = (h, n) else {
            continue;
        };
        if k == 0 || res == "PRO" {
            continue;
        }
        if rng.f64() < cfg.p_missing_root {
            continue;
        }
        let h_val = if apply_offsets {
            h_raw + cfg.ref_offset_h
        } else {
            h_raw
        };
        let n_val = if apply_offsets {
            n_raw + cfg.ref_offset_n
        } else {
            n_raw
        };
        let res_scale = if cfg.exchange_sigma > 0.0 {
            (arng.normal(0.0, cfg.exchange_sigma))
                .exp()
                .clamp(0.12, 2.6)
        } else {
            1.0
        };

        roots.push((h_val, n_val));
        let hj = h_val + rng.normal(0.0, cfg.noise_h);
        let nj = n_val + rng.normal(0.0, cfg.noise_n);
        let mut root_pk = Peak::hsqc(hj, nj);
        let mut root_ht = 120.0;
        if cfg.include_intensities {
            root_ht = 120.0 * (irng.normal(0.0, 0.18)).exp() * res_scale;
            root_pk = root_pk.with_intensity(root_ht, Some(root_ht * 4.2));
        }
        root_amps.push((hj, nj, root_ht));
        peaks.push(root_pk);
        truth.push(k as i32);

        for exp in &cfg.experiments {
            let Some(rules) = experiment_rules(exp) else {
                continue;
            };
            let has_both = rules.iter().any(|r| r.1 == 0) && rules.iter().any(|r| r.1 == -1);
            // Detect coincident i / i-1 carbon shifts so simulated intensity sums realistically
            for &(atom, off, sign) in rules {
                let target = (k as isize) + (off as isize);
                if target < 0 {
                    continue;
                }
                let Some(&c_raw) = ref_data.shifts.get(&(target as usize, atom.to_string())) else {
                    continue;
                };
                if rng.f64() < cfg.p_missing_peak {
                    continue;
                }
                let c_val = if apply_offsets {
                    c_raw + cfg.carbon_offset(atom)
                } else {
                    c_raw
                };
                let ph = h_val + rng.normal(0.0, cfg.noise_h);
                let pn = n_val + rng.normal(0.0, cfg.noise_n);
                let pc = c_val + rng.normal(0.0, cfg.noise_c);
                let mut pk = Peak::triple(exp.clone(), ph, pn, pc, sign);
                let mut ht = 90.0;
                if cfg.include_intensities {
                    // Intraresidue (i) 1J transfer is ~2.2x stronger than sequential (i-1) 2J transfer
                    let base = if !has_both {
                        90.0
                    } else if off == 0 {
                        100.0
                    } else {
                        44.0
                    };
                    let exp_attenuation = if cfg.exchange_sigma > 0.0 && exp == "HNCACO" {
                        0.42
                    } else {
                        1.0
                    };
                    ht = base * (irng.normal(0.0, 0.20)).exp() * res_scale * exp_attenuation;
                    pk = pk.with_intensity(ht, Some(ht * 5.8));
                }
                parent_3d_peaks.push((exp.clone(), ph, pn, pc, sign, ht));
                peaks.push(pk);
            }
        }
    }

    if cfg.sidechain_nh2 {
        let has_hnco = cfg.experiments.iter().any(|e| e == "HNCO");
        let has_cbcaconh = cfg.experiments.iter().any(|e| e == "CBCACONH");
        for res in &ref_data.sequence {
            if res == "ASN" || res == "GLN" {
                let n_sc_raw = rng.normal(112.5, 1.0);
                let n_sc = if apply_offsets {
                    n_sc_raw + cfg.ref_offset_n
                } else {
                    n_sc_raw
                };
                let mut pair_coords: [(f64, f64, f64); 2] = [(0.0, 0.0, 85.0); 2];
                for (pair_idx, h_mean) in [6.85, 7.55].into_iter().enumerate() {
                    let h_sc_raw = rng.normal(h_mean, 0.1);
                    let h_sc = if apply_offsets {
                        h_sc_raw + cfg.ref_offset_h
                    } else {
                        h_sc_raw
                    };
                    let n_pick = if cfg.nh2_sidechain_3d {
                        n_sc + arng.normal(0.0, 0.022)
                    } else {
                        n_sc
                    };
                    let mut ht = 85.0;
                    if cfg.include_intensities {
                        ht = 85.0 * (irng.normal(0.0, 0.20)).exp();
                    }
                    pair_coords[pair_idx] = (h_sc, n_pick, ht);
                }

                // Ensure realistic NH2 doublet separation (0.35 <= |dH| <= 1.15 ppm) in realistic mode
                if cfg.nh2_sidechain_3d {
                    let mid = 0.5 * (pair_coords[0].0 + pair_coords[1].0);
                    let sep = (pair_coords[1].0 - pair_coords[0].0)
                        .abs()
                        .clamp(0.42, 1.05);
                    pair_coords[0].0 = mid - 0.5 * sep;
                    pair_coords[1].0 = mid + 0.5 * sep;
                }

                for &(h_sc, n_pick, ht) in &pair_coords {
                    let mut pk = Peak::hsqc(h_sc, n_pick);
                    if cfg.include_intensities {
                        pk = pk.with_intensity(ht, Some(ht * 4.0));
                    }
                    peaks.push(pk);
                    truth.push(-1);
                }

                // Sidechain NH2 transfer in HNCO (CG/CD carbonyl) and CBCACONH (CB/CG aliphatic),
                // lacking any intra-residue CA(i) peak in HNCACB.
                if cfg.nh2_sidechain_3d {
                    let c_co = if res == "ASN" { 177.2 } else { 180.1 }
                        + cfg.ref_offset_c
                        + arng.normal(0.0, 0.35);
                    let c_aliph = if res == "ASN" { 38.6 } else { 33.8 }
                        + cfg.ref_offset_cb
                        + arng.normal(0.0, 0.45);
                    for &(h_sc, n_pick, ht) in &pair_coords {
                        if has_hnco {
                            let mut p_co = Peak::triple(
                                "HNCO",
                                h_sc + arng.normal(0.0, cfg.noise_h),
                                n_pick + arng.normal(0.0, cfg.noise_n),
                                c_co + arng.normal(0.0, cfg.noise_c),
                                1,
                            );
                            if cfg.include_intensities {
                                let h_co = ht * arng.uniform(0.65, 0.95);
                                p_co = p_co.with_intensity(h_co, Some(h_co * 5.2));
                            }
                            peaks.push(p_co);
                        }
                        if has_cbcaconh {
                            let mut p_cb = Peak::triple(
                                "CBCACONH",
                                h_sc + arng.normal(0.0, cfg.noise_h),
                                n_pick + arng.normal(0.0, cfg.noise_n),
                                c_aliph + arng.normal(0.0, cfg.noise_c),
                                1,
                            );
                            if cfg.include_intensities {
                                let h_cb = ht * arng.uniform(0.45, 0.80);
                                p_cb = p_cb.with_intensity(h_cb, Some(h_cb * 5.2));
                            }
                            peaks.push(p_cb);
                        }
                    }
                }
            }
        }
    }

    if !roots.is_empty() {
        for exp in &cfg.experiments {
            if exp == "HSQC" {
                continue;
            }
            let Some(rules) = experiment_rules(exp) else {
                continue;
            };
            if rules.is_empty() {
                continue;
            }
            let n_real = peaks.iter().filter(|p| &p.exp == exp).count();
            let n_spur = rng.poisson(cfg.spurious_rate * (n_real as f64));
            for _ in 0..n_spur {
                let (rh, rn) = roots[rng.usize(roots.len())];
                let &(atom, _, sign) = &rules[rng.usize(rules.len())];
                let (lo, hi) = carbon_range(atom);
                let c_off = if apply_offsets {
                    cfg.carbon_offset(atom)
                } else {
                    0.0
                };
                let ph = rh + rng.normal(0.0, cfg.noise_h);
                let pn = rn + rng.normal(0.0, cfg.noise_n);
                let pc = rng.uniform(lo + c_off, hi + c_off);
                let mut pk = Peak::triple(exp.clone(), ph, pn, pc, sign);
                if cfg.include_intensities {
                    let ht = 28.0 * (irng.normal(0.0, 0.25)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 5.0));
                }
                peaks.push(pk);
            }
        }
    }

    // Realistic Bruker artifacts: F1/F2 truncation sinc sidelobes, indirect t1 ridges,
    // and residual water / upfield aliphatic false HSQC roots.
    if cfg.sinc_sidelobe_frac > 0.0
        || cfg.t1_ridge_spurious_rate > 0.0
        || cfg.water_artifact_roots > 0
    {
        for &(rh, rn, rht) in &root_amps {
            // Intense parent amides produce F1 (15N) sinc sidelobes and t1-ridge false roots
            if cfg.sinc_sidelobe_frac > 0.0 && rht >= 135.0 {
                for dir in [-1.0_f64, 1.0_f64] {
                    if arng.f64() < 0.65 {
                        let dn = dir * arng.uniform(1.15, 1.85);
                        let sh = rh + arng.normal(0.0, cfg.noise_h * 0.6);
                        let sn = (rn + dn).clamp(100.4, 137.3);
                        let mut spk = Peak::hsqc(sh, sn);
                        if cfg.include_intensities {
                            let sht = rht * cfg.sinc_sidelobe_frac * arng.uniform(0.65, 1.25);
                            spk = spk.with_intensity(sht, Some(sht * 3.4));
                        }
                        peaks.push(spk);
                        truth.push(-1);
                    }
                }
            }
            if cfg.t1_ridge_spurious_rate > 0.0
                && rht >= 145.0
                && arng.f64() < cfg.t1_ridge_spurious_rate
            {
                let sh = rh + arng.normal(0.0, cfg.noise_h * 0.8);
                let sn = arng.uniform(100.5, 137.2);
                if (sn - rn).abs() > 0.8 {
                    let mut spk = Peak::hsqc(sh, sn);
                    if cfg.include_intensities {
                        let sht = rht * arng.uniform(0.016, 0.036);
                        spk = spk.with_intensity(sht, Some(sht * 3.2));
                    }
                    peaks.push(spk);
                    truth.push(-1);
                }
            }
        }

        for idx in 0..cfg.water_artifact_roots {
            let (wh, wn) = if idx % 2 == 0 {
                (
                    4.77 + cfg.ref_offset_h + arng.normal(0.0, 0.035),
                    arng.uniform(104.0, 133.0),
                )
            } else {
                (arng.uniform(0.95, 4.05), arng.uniform(103.0, 134.0))
            };
            let mut wpk = Peak::hsqc(wh, wn);
            if cfg.include_intensities {
                let wht = arng.uniform(11.0, 26.0);
                wpk = wpk.with_intensity(wht, Some(wht * 3.8));
            }
            peaks.push(wpk);
            truth.push(-1);
        }

        // 3D F1 (13C) sinc sidelobes and t1-ridge spurious peaks along strong strips
        for &(ref exp, ph, pn, pc, sign, ht) in &parent_3d_peaks {
            let is_co = exp == "HNCO" || exp == "HNCACO";
            let (c_lo, c_hi) = uncropped_bruker_carbon_range(exp);
            if cfg.sinc_sidelobe_frac > 0.0 && ht >= 95.0 {
                let delta_acq = if is_co { 0.24 } else { 0.78 };
                for dir in [-1.0_f64, 1.0_f64] {
                    if arng.f64() < 0.55 {
                        // First negative sidelobe (~1.43 * delta_acq) in HNCACB, or second positive
                        // sidelobe (~2.46 * delta_acq) in positive-only 3D spectra.
                        let (lobe_mult, lobe_sign, rel_scale) =
                            if exp == "HNCACB" && arng.f64() < 0.55 {
                                (1.43, -sign, 1.15)
                            } else {
                                (2.46, if exp == "HNCACB" { sign } else { 1 }, 0.80)
                            };
                        let sc = (pc + dir * lobe_mult * delta_acq * arng.uniform(0.92, 1.08))
                            .clamp(c_lo + 0.3, c_hi - 0.3);
                        let mut spk = Peak::triple(
                            exp.clone(),
                            ph + arng.normal(0.0, cfg.noise_h * 0.5),
                            pn + arng.normal(0.0, cfg.noise_n * 0.5),
                            sc,
                            lobe_sign,
                        );
                        if cfg.include_intensities {
                            let sht =
                                ht * cfg.sinc_sidelobe_frac * rel_scale * arng.uniform(0.75, 1.25);
                            spk = spk.with_intensity(sht, Some(sht * 4.0));
                        }
                        peaks.push(spk);
                    }
                }
            }
            if cfg.t1_ridge_spurious_rate > 0.0
                && ht >= 110.0
                && arng.f64() < cfg.t1_ridge_spurious_rate
            {
                let rc = arng.uniform(c_lo, c_hi);
                let r_sign = if exp == "HNCACB" && arng.f64() < 0.5 {
                    -1
                } else {
                    1
                };
                let mut rpk = Peak::triple(
                    exp.clone(),
                    ph + arng.normal(0.0, cfg.noise_h * 0.6),
                    pn + arng.normal(0.0, cfg.noise_n * 0.6),
                    rc,
                    r_sign,
                );
                if cfg.include_intensities {
                    let rht = ht * arng.uniform(0.015, 0.035);
                    rpk = rpk.with_intensity(rht, Some(rht * 4.0));
                }
                peaks.push(rpk);
            }
        }
    }

    let hsqc_indices: Vec<usize> = peaks
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (p.exp == "HSQC").then_some(i))
        .collect();
    let perm = rng.permutation(hsqc_indices.len());
    let hsqc_peaks: Vec<Peak> = perm
        .iter()
        .map(|&j| peaks[hsqc_indices[j]].clone())
        .collect();
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
    pub c_aliph_high_ppm: f64,
    pub c_aliph_low_ppm: f64,
    pub c_ca_high_ppm: f64,
    pub c_ca_low_ppm: f64,
    pub c_co_high_ppm: f64,
    pub c_co_low_ppm: f64,
    pub sf_h_mhz: f64,
    pub sf_n_mhz: f64,
    pub sf_c_mhz: f64,
    pub lw_h_ppm: f64,
    pub lw_n_ppm: f64,
    pub lw_c_ppm: f64,
    pub lw_c_co_ppm: f64,
    /// Thermal baseline noise standard deviation (peak amplitude = `100.0`).
    pub noise_level: f32,
    /// Fraction Gaussian vs Lorentzian lineshape (`0.0` = pure Lorentzian, `1.0` = pure Gaussian).
    pub gaussian_mix: f64,
    /// Bruker indirect `15N` time-domain size `TD` (`0` = infinite/no truncation ripples).
    pub td_n_indirect: usize,
    /// Bruker indirect `13C` time-domain size `TD` (`0` = infinite/no truncation ripples).
    pub td_c_indirect: usize,
    /// Truncation `sinc` blend weight (`0.0` = none; `~0.36` produces `2–6%` apodized sidelobes).
    pub sinc_amp: f64,
    /// Multiplicative indirect `t1` ridge noise RMS as a fraction of column/strip signal (`0.0` = none).
    pub t1_ridge_fraction: f64,
    /// Residual water chemical shift in ppm (`~4.77 ppm`).
    pub water_ppm: f64,
    /// Residual water suppression ridge/artifact amplitude (`0.0` = none).
    pub water_artifact_amp: f64,
    /// Upfield `< 4.3 ppm` aliphatic/methyl breakthrough artifact amplitude (`0.0` = none).
    pub aliphatic_bleed_amp: f64,
    /// Gentle baseline roll amplitude across spectral axes (`0.0` = none).
    pub baseline_roll_amp: f64,
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
            c_aliph_high_ppm: 75.0,
            c_aliph_low_ppm: 14.0,
            c_ca_high_ppm: 70.0,
            c_ca_low_ppm: 40.0,
            c_co_high_ppm: 183.0,
            c_co_low_ppm: 168.0,
            sf_h_mhz: 600.13,
            sf_n_mhz: 60.81,
            sf_c_mhz: 150.90,
            lw_h_ppm: 0.022,
            lw_n_ppm: 0.22,
            lw_c_ppm: 0.28,
            lw_c_co_ppm: 0.28,
            noise_level: 1.5,
            gaussian_mix: 0.65,
            td_n_indirect: 0,
            td_c_indirect: 0,
            sinc_amp: 0.0,
            t1_ridge_fraction: 0.0,
            water_ppm: 4.77,
            water_artifact_amp: 0.0,
            aliphatic_bleed_amp: 0.0,
            baseline_roll_amp: 0.0,
            seed: 42,
        }
    }
}

impl SpectrumRenderConfig {
    /// Realistic Bruker 500 MHz processed spectrum configuration (`pdata` `2rr`/`3rrr` physics):
    /// full uncropped sweep widths (`11.75 .. -2.25 ppm` `1H`, `137.5 .. 100.2 ppm` `15N`,
    /// `76.5 .. 1.63 ppm` aliphatic `13C`, `187.0 .. 165.0 ppm` carbonyl `13C`), high dynamic
    /// range (`peak_amp / thermal_noise ~ 2,500–4,000`), finite-`TD` apodized `sinc` truncation
    /// ripples (`2–6%` positive/negative sidelobes), multiplicative `t1` noise ridges, `~4.77 ppm`
    /// residual water suppression stripe, upfield `< 4.3 ppm` aliphatic bleed, and gentle baseline roll.
    pub fn realistic_bruker() -> Self {
        Self {
            nx_h: 768,
            ny_n: 128,
            nz_c: 256,
            h_high_ppm: 11.75,
            h_low_ppm: -2.25,
            n_high_ppm: 137.5,
            n_low_ppm: 100.2,
            c_aliph_high_ppm: 76.5,
            c_aliph_low_ppm: 1.63,
            c_ca_high_ppm: 76.5,
            c_ca_low_ppm: 1.63,
            c_co_high_ppm: 187.0,
            c_co_low_ppm: 165.0,
            sf_h_mhz: 500.13,
            sf_n_mhz: 50.68,
            sf_c_mhz: 125.76,
            lw_h_ppm: 0.026,
            lw_n_ppm: 0.34,
            lw_c_ppm: 0.42,
            lw_c_co_ppm: 0.13,
            noise_level: 0.035,
            gaussian_mix: 0.60,
            td_n_indirect: 96,
            td_c_indirect: 192,
            sinc_amp: 0.36,
            t1_ridge_fraction: 0.0085,
            water_ppm: 4.77,
            water_artifact_amp: 18.0,
            aliphatic_bleed_amp: 12.0,
            baseline_roll_amp: 0.020,
            seed: 42,
        }
    }

    #[inline]
    fn has_realistic_artifacts(&self) -> bool {
        self.sinc_amp > 0.0
            || self.td_n_indirect > 0
            || self.td_c_indirect > 0
            || self.t1_ridge_fraction > 0.0
            || self.water_artifact_amp > 0.0
            || self.aliphatic_bleed_amp > 0.0
            || self.baseline_roll_amp > 0.0
    }
}

#[inline]
fn lineshape_1d(delta_ppm: f64, lw_ppm: f64, gauss_mix: f64) -> f64 {
    let u = delta_ppm / (0.5 * lw_ppm.max(1e-6));
    let gauss = (-LN_2 * u * u).exp();
    let lorentz = 1.0 / (1.0 + u * u);
    gauss_mix * gauss + (1.0 - gauss_mix) * lorentz
}

/// Indirect-dimension lineshape combining the intrinsic pseudo-Voigt peak with finite-`TD`
/// truncation `sinc(pi * dx) = sin(pi * dx) / (pi * dx)` damped by cosine-bell and exponential
/// apodization (`sinc_amp = 0.0` reduces identically to `lineshape_1d`).
#[inline]
fn indirect_lineshape_1d(
    delta_ppm: f64,
    lw_ppm: f64,
    gauss_mix: f64,
    delta_acq_ppm: f64,
    sinc_amp: f64,
) -> f64 {
    let core = lineshape_1d(delta_ppm, lw_ppm, gauss_mix);
    if sinc_amp <= 0.0 || delta_acq_ppm <= 0.0 {
        return core;
    }
    let dx = delta_ppm / delta_acq_ppm;
    let abs_dx = dx.abs();
    const MAX_LOBES: f64 = 11.0;
    if abs_dx >= MAX_LOBES {
        return (1.0 - sinc_amp) * core;
    }
    let sinc = if abs_dx < 1e-7 {
        1.0
    } else {
        let pix = PI * dx;
        pix.sin() / pix
    };
    let cos_bell = 0.5 * (1.0 + (PI * (abs_dx / MAX_LOBES)).cos());
    let exp_damp = (-abs_dx / 7.5).exp();
    (1.0 - sinc_amp) * core + sinc_amp * sinc * cos_bell * exp_damp
}

#[inline]
fn indirect_delta_acq_ppm(sw_ppm: f64, axis_len: usize, td_indirect: usize) -> f64 {
    if td_indirect == 0 || axis_len < 8 {
        return 0.0;
    }
    // Ensure >= ~2.67 zero-filled points per Rayleigh interval so sinc sidelobes are resolved extrema
    let max_nacq = (axis_len * 3 / 8).max(4);
    let n_acq = (td_indirect / 2).clamp(4, max_nacq);
    sw_ppm.abs() / (n_acq as f64)
}

/// Fast deterministic Gaussian thermal noise filler for large realistic 3D cubes: pre-draws a
/// 65,536-entry Box-Muller table from `rng` and sums two independent LCG-indexed entries divided
/// by `sqrt(2)` (`N(0, sigma^2)` with > 4e9 combinations, zero spatial stride aliasing, ~15x faster).
fn fill_gaussian_noise_fast(data: &mut [f32], sigma: f32, rng: &mut Rng) {
    if sigma <= 0.0 || data.is_empty() {
        return;
    }
    const LUT_BITS: usize = 16;
    const LUT_LEN: usize = 1 << LUT_BITS;
    let mut lut = vec![0.0f32; LUT_LEN];
    let scale = (sigma as f64) * FRAC_1_SQRT_2;
    for slot in &mut lut {
        *slot = rng.normal(0.0, scale) as f32;
    }
    let mut state = (rng.f64().to_bits()) ^ 0xA076_1D64_78BD_642F;
    for v in data.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i1 = ((state >> 48) as usize) & (LUT_LEN - 1);
        let i2 = ((state >> 32) as usize) & (LUT_LEN - 1);
        *v += lut[i1] + lut[i2];
    }
}

/// Render a synthetic 2D `1H-15N` HSQC spectrum from peaks or a reference assignment.
pub fn render_hsqc_2d(peaks: &[Peak], cfg: &SpectrumRenderConfig) -> Spectrum2D {
    if !cfg.has_realistic_artifacts() {
        let axis_x =
            AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, cfg.sf_h_mhz);
        let axis_y =
            AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, cfg.sf_n_mhz);
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
            if cx + rad_x < 0
                || cx - rad_x >= cfg.nx_h as isize
                || cy + rad_y < 0
                || cy - rad_y >= cfg.ny_n as isize
            {
                continue;
            }
            let amp = p.height.unwrap_or(100.0) * (p.sign as f64);

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

        return Spectrum2D {
            name: "Synthetic 1H-15N HSQC".to_string(),
            axis_x,
            axis_y,
            data,
        };
    }

    let axis_x = AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, cfg.sf_h_mhz);
    let axis_y = AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, cfg.sf_n_mhz);
    let mut rng = Rng::new(cfg.seed);
    let mut data = vec![0.0f32; cfg.ny_n * cfg.nx_h];

    let sw_n = (cfg.n_high_ppm - cfg.n_low_ppm).abs();
    let delta_acq_n = indirect_delta_acq_ppm(sw_n, cfg.ny_n, cfg.td_n_indirect);
    let rad_x = ((4.0 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let core_rad_y = ((4.0 * cfg.lw_n_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(3);
    let sinc_rad_y = if cfg.sinc_amp > 0.0 && delta_acq_n > 0.0 {
        (11.0 * delta_acq_n / axis_y.step_ppm.abs()).ceil() as isize
    } else {
        0
    };
    let rad_y = core_rad_y.max(sinc_rad_y);

    let mut col_anchors: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    let mut wx_buf = Vec::new();
    let mut wy_buf = Vec::new();

    // 1. Render backbone + sidechain peaks with indirect 15N finite-TD apodized sinc truncation ripples
    for p in peaks.iter().filter(|p| p.exp == "HSQC") {
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(p.n).round() as isize;
        if cx + rad_x < 0
            || cx - rad_x >= cfg.nx_h as isize
            || cy + rad_y < 0
            || cy - rad_y >= cfg.ny_n as isize
        {
            continue;
        }
        let amp = p.height.unwrap_or(100.0) * (p.sign as f64);
        if cx >= 0 && (cx as usize) < cfg.nx_h {
            let entry = col_anchors.entry(cx as usize).or_insert((p.h, 0.0));
            if amp.abs() > entry.1 {
                entry.0 = p.h;
            }
            entry.1 = entry.1.hypot(amp.abs());
        }

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
        if x0 > x1 || y0 > y1 {
            continue;
        }

        wx_buf.clear();
        for ix in x0..=x1 {
            let px = axis_x.index_to_ppm(ix as f64);
            wx_buf.push(lineshape_1d(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix));
        }
        wy_buf.clear();
        for iy in y0..=y1 {
            let py = axis_y.index_to_ppm(iy as f64);
            wy_buf.push(indirect_lineshape_1d(
                py - p.n,
                cfg.lw_n_ppm,
                cfg.gaussian_mix,
                delta_acq_n,
                cfg.sinc_amp,
            ));
        }

        for (yi, &wy) in wy_buf.iter().enumerate() {
            let row_off = (y0 + yi) * cfg.nx_h;
            let amp_wy = amp * wy;
            for (xi, &wx) in wx_buf.iter().enumerate() {
                data[row_off + x0 + xi] += (amp_wy * wx) as f32;
            }
        }
    }

    // 2. Multiplicative indirect t1 (15N) ridge noise along columns containing strong signals
    if cfg.t1_ridge_fraction > 0.0 {
        let min_parent = (cfg.noise_level as f64) * 80.0;
        for (&cx_u, &(h_center, col_amp)) in &col_anchors {
            if col_amp < min_parent {
                continue;
            }
            let r_sigma = cfg.t1_ridge_fraction * col_amp;
            let dc = r_sigma * rng.uniform(-0.25, 0.35);
            let cx = cx_u as isize;
            let x0 = (cx - rad_x).max(0) as usize;
            let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
            wx_buf.clear();
            for ix in x0..=x1 {
                let px = axis_x.index_to_ppm(ix as f64);
                wx_buf.push(lineshape_1d(px - h_center, cfg.lw_h_ppm, cfg.gaussian_mix));
            }
            for iy in 0..cfg.ny_n {
                let streak = dc + rng.normal(0.0, r_sigma);
                let row_off = iy * cfg.nx_h;
                for (xi, &wx) in wx_buf.iter().enumerate() {
                    data[row_off + x0 + xi] += (streak * wx) as f32;
                }
            }
        }
    }

    let h_lo = cfg.h_low_ppm.min(cfg.h_high_ppm);
    let h_hi = cfg.h_low_ppm.max(cfg.h_high_ppm);

    // 3. Residual ~4.77 ppm water suppression stripe & radiation-damping dispersive glitches
    if cfg.water_artifact_amp > 0.0 && cfg.water_ppm > h_lo + 0.15 && cfg.water_ppm < h_hi - 0.15 {
        let w_cx = axis_x.ppm_to_index(cfg.water_ppm).round() as isize;
        let w_lw = (cfg.lw_h_ppm * 2.2).max(0.055);
        let w_rad = ((4.5 * w_lw / axis_x.step_ppm.abs()).ceil() as isize).max(4);
        let x0 = (w_cx - w_rad).max(0) as usize;
        let x1 = (w_cx + w_rad).min(cfg.nx_h as isize - 1).max(0) as usize;
        if x0 <= x1 {
            for iy in 0..cfg.ny_n {
                let absorptive = rng.normal(0.0, 0.32 * cfg.water_artifact_amp);
                let dispersive = rng.normal(0.0, 0.28 * cfg.water_artifact_amp);
                let row_off = iy * cfg.nx_h;
                for ix in x0..=x1 {
                    let dh = axis_x.index_to_ppm(ix as f64) - cfg.water_ppm;
                    let u = dh / (0.5 * w_lw);
                    let abs_prof = 1.0 / (1.0 + u * u);
                    let disp_prof = u / (1.0 + u * u);
                    data[row_off + ix] += (absorptive * abs_prof + dispersive * disp_prof) as f32;
                }
            }
            // Localized intense residual solvent axial spikes near ~4.77 ppm
            for k in 0..4usize {
                let spike_y = ((k * 2 + 1) * cfg.ny_n) / 8;
                let spike_sign = if k % 2 == 0 { 1.0 } else { -0.85 };
                let spike_amp = cfg.water_artifact_amp * spike_sign * rng.uniform(0.75, 1.15);
                let spike_h = cfg.water_ppm + (if k < 2 { 0.015 } else { -0.018 });
                let y0 = spike_y.saturating_sub(3);
                let y1 = (spike_y + 3).min(cfg.ny_n - 1);
                for iy in y0..=y1 {
                    let py = axis_y.index_to_ppm(iy as f64);
                    let wy = lineshape_1d(
                        py - axis_y.index_to_ppm(spike_y as f64),
                        cfg.lw_n_ppm * 1.3,
                        0.5,
                    );
                    for ix in x0..=x1 {
                        let px = axis_x.index_to_ppm(ix as f64);
                        let wx = lineshape_1d(px - spike_h, cfg.lw_h_ppm * 1.4, 0.5);
                        data[iy * cfg.nx_h + ix] += (spike_amp * wx * wy) as f32;
                    }
                }
            }
        }
    }

    // 4. Upfield (< 4.3 ppm) methyl / aliphatic breakthrough peaks and streaks
    if cfg.aliphatic_bleed_amp > 0.0 && h_lo < 4.1 {
        let up_lo = h_lo.max(0.65);
        let up_hi = 4.15_f64.min(h_hi - 0.2);
        if up_lo < up_hi {
            let n_lo = cfg.n_low_ppm.min(cfg.n_high_ppm) + 1.5;
            let n_hi = cfg.n_low_ppm.max(cfg.n_high_ppm) - 1.5;
            for _ in 0..12 {
                let bh = rng.uniform(up_lo, up_hi);
                let bn = rng.uniform(n_lo, n_hi);
                let b_amp = cfg.aliphatic_bleed_amp * rng.uniform(0.45, 1.15);
                let cx = axis_x.ppm_to_index(bh).round() as isize;
                let cy = axis_y.ppm_to_index(bn).round() as isize;
                let x0 = (cx - rad_x).max(0) as usize;
                let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
                let y0 = (cy - core_rad_y).max(0) as usize;
                let y1 = (cy + core_rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
                for iy in y0..=y1 {
                    let py = axis_y.index_to_ppm(iy as f64);
                    let wy = lineshape_1d(py - bn, cfg.lw_n_ppm * 1.2, 0.5);
                    for ix in x0..=x1 {
                        let px = axis_x.index_to_ppm(ix as f64);
                        let wx = lineshape_1d(px - bh, cfg.lw_h_ppm * 1.2, 0.5);
                        data[iy * cfg.nx_h + ix] += (b_amp * wx * wy) as f32;
                    }
                }
            }
        }
    }

    // 5. Gentle baseline roll across direct & indirect dimensions
    if cfg.baseline_roll_amp > 0.0 {
        let roll = cfg.baseline_roll_amp;
        let inv_nx = 1.0 / (cfg.nx_h.max(1) as f64);
        let inv_ny = 1.0 / (cfg.ny_n.max(1) as f64);
        for iy in 0..cfg.ny_n {
            let cy = (TAU * (iy as f64) * inv_ny + 0.4).cos();
            let row_off = iy * cfg.nx_h;
            for ix in 0..cfg.nx_h {
                let u = (ix as f64) * inv_nx - 0.5;
                let bx = (1.25 * TAU * (ix as f64) * inv_nx + 0.25).sin();
                data[row_off + ix] += (roll * (0.75 * bx * cy + 0.35 * u)) as f32;
            }
        }
    }

    // 6. Thermal receiver noise
    fill_gaussian_noise_fast(&mut data, cfg.noise_level, &mut rng);

    Spectrum2D {
        name: "Synthetic 1H-15N HSQC".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

/// Render a synthetic 3D triple-resonance spectrum (`HNCACB`, `HNCA`, `HNCO`, etc.) from peaks.
pub fn render_triple_3d(exp: &str, peaks: &[Peak], cfg: &SpectrumRenderConfig) -> Spectrum3D {
    let is_co = exp == "HNCO" || exp == "HNCACO";
    let (c_high, c_low) = if is_co {
        (cfg.c_co_high_ppm, cfg.c_co_low_ppm)
    } else if exp == "HNCACB" || exp == "CBCACONH" {
        (cfg.c_aliph_high_ppm, cfg.c_aliph_low_ppm)
    } else {
        (cfg.c_ca_high_ppm, cfg.c_ca_low_ppm)
    };

    if !cfg.has_realistic_artifacts() {
        let axis_x =
            AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, cfg.sf_h_mhz);
        let axis_y =
            AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, cfg.sf_n_mhz);
        let axis_z = AxisPpm::from_bounds("13C", cfg.nz_c, c_high, c_low, cfg.sf_c_mhz);
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
            if cx + rad_x < 0
                || cx - rad_x >= cfg.nx_h as isize
                || cy + rad_y < 0
                || cy - rad_y >= cfg.ny_n as isize
                || cz + rad_z < 0
                || cz - rad_z >= cfg.nz_c as isize
            {
                continue;
            }
            let amp = p.height.unwrap_or(90.0) * (p.sign as f64);

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

        return Spectrum3D {
            name: format!("Synthetic 3D {exp}"),
            exp: exp.to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        };
    }

    let axis_x = AxisPpm::from_bounds("1H", cfg.nx_h, cfg.h_high_ppm, cfg.h_low_ppm, cfg.sf_h_mhz);
    let axis_y = AxisPpm::from_bounds("15N", cfg.ny_n, cfg.n_high_ppm, cfg.n_low_ppm, cfg.sf_n_mhz);
    let axis_z = AxisPpm::from_bounds("13C", cfg.nz_c, c_high, c_low, cfg.sf_c_mhz);
    let mut rng = Rng::new(cfg.seed.wrapping_add(1337));
    let mut data = vec![0.0f32; cfg.nz_c * cfg.ny_n * cfg.nx_h];

    let lw_c = if is_co { cfg.lw_c_co_ppm } else { cfg.lw_c_ppm };
    let sw_n = (cfg.n_high_ppm - cfg.n_low_ppm).abs();
    let sw_c = (c_high - c_low).abs();
    let delta_acq_n = indirect_delta_acq_ppm(sw_n, cfg.ny_n, cfg.td_n_indirect);
    let delta_acq_c = indirect_delta_acq_ppm(sw_c, cfg.nz_c, cfg.td_c_indirect);

    let rad_x = ((3.5 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(2);
    let core_rad_y = ((3.5 * cfg.lw_n_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(2);
    let core_rad_z = ((3.5 * lw_c / axis_z.step_ppm.abs()).ceil() as isize).max(2);
    let sinc_rad_y = if cfg.sinc_amp > 0.0 && delta_acq_n > 0.0 {
        (11.0 * delta_acq_n / axis_y.step_ppm.abs()).ceil() as isize
    } else {
        0
    };
    let sinc_rad_z = if cfg.sinc_amp > 0.0 && delta_acq_c > 0.0 {
        (11.0 * delta_acq_c / axis_z.step_ppm.abs()).ceil() as isize
    } else {
        0
    };
    let rad_y = core_rad_y.max(sinc_rad_y);
    let rad_z = core_rad_z.max(sinc_rad_z);

    let mut strip_anchors: BTreeMap<(usize, usize), (f64, f64, f64)> = BTreeMap::new();
    let mut wx_buf = Vec::new();
    let mut wy_buf = Vec::new();
    let mut wz_buf = Vec::new();

    // 1. Render 3D peaks with separable finite-TD apodized sinc ripples along indirect 15N & 13C
    for p in peaks.iter().filter(|p| p.exp == exp) {
        let Some(c_ppm) = p.c else { continue };
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(p.n).round() as isize;
        let cz = axis_z.ppm_to_index(c_ppm).round() as isize;
        if cx + rad_x < 0
            || cx - rad_x >= cfg.nx_h as isize
            || cy + rad_y < 0
            || cy - rad_y >= cfg.ny_n as isize
            || cz + rad_z < 0
            || cz - rad_z >= cfg.nz_c as isize
        {
            continue;
        }
        let amp = p.height.unwrap_or(90.0) * (p.sign as f64);
        if cx >= 0 && (cx as usize) < cfg.nx_h && cy >= 0 && (cy as usize) < cfg.ny_n {
            let entry = strip_anchors
                .entry((cx as usize, cy as usize))
                .or_insert((p.h, p.n, 0.0));
            if amp.abs() > entry.2 {
                entry.0 = p.h;
                entry.1 = p.n;
            }
            entry.2 = entry.2.hypot(amp.abs());
        }

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
        let z0 = (cz - rad_z).max(0) as usize;
        let z1 = (cz + rad_z).min(cfg.nz_c as isize - 1).max(0) as usize;
        if x0 > x1 || y0 > y1 || z0 > z1 {
            continue;
        }

        wx_buf.clear();
        for ix in x0..=x1 {
            let px = axis_x.index_to_ppm(ix as f64);
            wx_buf.push(lineshape_1d(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix));
        }
        wy_buf.clear();
        for iy in y0..=y1 {
            let py = axis_y.index_to_ppm(iy as f64);
            wy_buf.push(indirect_lineshape_1d(
                py - p.n,
                cfg.lw_n_ppm,
                cfg.gaussian_mix,
                delta_acq_n,
                cfg.sinc_amp,
            ));
        }
        wz_buf.clear();
        for iz in z0..=z1 {
            let pz = axis_z.index_to_ppm(iz as f64);
            wz_buf.push(indirect_lineshape_1d(
                pz - c_ppm,
                lw_c,
                cfg.gaussian_mix,
                delta_acq_c,
                cfg.sinc_amp,
            ));
        }

        for (zi, &wz) in wz_buf.iter().enumerate() {
            let amp_wz = amp * wz;
            let z_plane = (z0 + zi) * cfg.ny_n;
            for (yi, &wy) in wy_buf.iter().enumerate() {
                let wyz = amp_wz * wy;
                let base = (z_plane + y0 + yi) * cfg.nx_h + x0;
                for (xi, &wx) in wx_buf.iter().enumerate() {
                    data[base + xi] += (wyz * wx) as f32;
                }
            }
        }
    }

    // 2. Multiplicative indirect t1 (13C) ridge noise along amide strips containing strong signals
    if cfg.t1_ridge_fraction > 0.0 {
        let min_parent = (cfg.noise_level as f64) * 80.0;
        for (&(cx_u, cy_u), &(h_center, n_center, strip_amp)) in &strip_anchors {
            if strip_amp < min_parent {
                continue;
            }
            let r_sigma = cfg.t1_ridge_fraction * strip_amp;
            let dc = r_sigma * rng.uniform(-0.20, 0.30);
            let cx = cx_u as isize;
            let cy = cy_u as isize;
            let x0 = (cx - rad_x).max(0) as usize;
            let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
            let y0 = (cy - core_rad_y).max(0) as usize;
            let y1 = (cy + core_rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
            if x0 > x1 || y0 > y1 {
                continue;
            }
            wx_buf.clear();
            for ix in x0..=x1 {
                let px = axis_x.index_to_ppm(ix as f64);
                wx_buf.push(lineshape_1d(px - h_center, cfg.lw_h_ppm, cfg.gaussian_mix));
            }
            wy_buf.clear();
            for iy in y0..=y1 {
                let py = axis_y.index_to_ppm(iy as f64);
                wy_buf.push(lineshape_1d(py - n_center, cfg.lw_n_ppm, cfg.gaussian_mix));
            }
            for iz in 0..cfg.nz_c {
                let streak = dc + rng.normal(0.0, r_sigma);
                let z_plane = iz * cfg.ny_n;
                for (yi, &wy) in wy_buf.iter().enumerate() {
                    let s_wy = streak * wy;
                    let base = (z_plane + y0 + yi) * cfg.nx_h + x0;
                    for (xi, &wx) in wx_buf.iter().enumerate() {
                        data[base + xi] += (s_wy * wx) as f32;
                    }
                }
            }
        }
    }

    let h_lo = cfg.h_low_ppm.min(cfg.h_high_ppm);
    let h_hi = cfg.h_low_ppm.max(cfg.h_high_ppm);

    // 3. Residual ~4.77 ppm water suppression ridges & axial artifacts in 3D
    if cfg.water_artifact_amp > 0.0 && cfg.water_ppm > h_lo + 0.15 && cfg.water_ppm < h_hi - 0.15 {
        let w_cx = axis_x.ppm_to_index(cfg.water_ppm).round() as isize;
        let w_lw = (cfg.lw_h_ppm * 2.2).max(0.055);
        let w_rad = ((4.0 * w_lw / axis_x.step_ppm.abs()).ceil() as isize).max(3);
        let x0 = (w_cx - w_rad).max(0) as usize;
        let x1 = (w_cx + w_rad).min(cfg.nx_h as isize - 1).max(0) as usize;
        if x0 <= x1 {
            wx_buf.clear();
            for ix in x0..=x1 {
                let dh = axis_x.index_to_ppm(ix as f64) - cfg.water_ppm;
                let u = dh / (0.5 * w_lw);
                wx_buf.push(1.0 / (1.0 + u * u));
            }
            for k in 0..6usize {
                let wy_idx = ((k * 2 + 1) * cfg.ny_n) / 12;
                let y0 = wy_idx.saturating_sub(2);
                let y1 = (wy_idx + 2).min(cfg.ny_n - 1);
                for iz in 0..cfg.nz_c {
                    let streak = rng.normal(0.0, 0.35 * cfg.water_artifact_amp);
                    let z_plane = iz * cfg.ny_n;
                    for iy in y0..=y1 {
                        let base = (z_plane + iy) * cfg.nx_h + x0;
                        for (xi, &wx) in wx_buf.iter().enumerate() {
                            data[base + xi] += (streak * wx) as f32;
                        }
                    }
                }
            }
        }
    }

    // 4. Upfield (< 4.3 ppm) aliphatic/methyl breakthrough artifacts in 3D
    if cfg.aliphatic_bleed_amp > 0.0 && h_lo < 4.1 {
        let up_lo = h_lo.max(0.65);
        let up_hi = 4.15_f64.min(h_hi - 0.2);
        if up_lo < up_hi {
            let n_lo = cfg.n_low_ppm.min(cfg.n_high_ppm) + 2.0;
            let n_hi = cfg.n_low_ppm.max(cfg.n_high_ppm) - 2.0;
            let c_min = c_low.min(c_high) + 1.0;
            let c_max = c_low.max(c_high) - 1.0;
            for _ in 0..8 {
                let bh = rng.uniform(up_lo, up_hi);
                let bn = rng.uniform(n_lo, n_hi);
                let bc = rng.uniform(c_min, c_max);
                let b_amp = cfg.aliphatic_bleed_amp * rng.uniform(0.5, 1.1);
                let cx = axis_x.ppm_to_index(bh).round() as isize;
                let cy = axis_y.ppm_to_index(bn).round() as isize;
                let cz = axis_z.ppm_to_index(bc).round() as isize;
                let x0 = (cx - rad_x).max(0) as usize;
                let x1 = (cx + rad_x).min(cfg.nx_h as isize - 1).max(0) as usize;
                let y0 = (cy - core_rad_y).max(0) as usize;
                let y1 = (cy + core_rad_y).min(cfg.ny_n as isize - 1).max(0) as usize;
                let z0 = (cz - core_rad_z).max(0) as usize;
                let z1 = (cz + core_rad_z).min(cfg.nz_c as isize - 1).max(0) as usize;
                for iz in z0..=z1 {
                    let wz = lineshape_1d(axis_z.index_to_ppm(iz as f64) - bc, lw_c, 0.5);
                    for iy in y0..=y1 {
                        let wyz = wz
                            * lineshape_1d(axis_y.index_to_ppm(iy as f64) - bn, cfg.lw_n_ppm, 0.5);
                        let base = (iz * cfg.ny_n + iy) * cfg.nx_h;
                        for ix in x0..=x1 {
                            let wx = lineshape_1d(
                                axis_x.index_to_ppm(ix as f64) - bh,
                                cfg.lw_h_ppm,
                                0.5,
                            );
                            data[base + ix] += (b_amp * wx * wyz) as f32;
                        }
                    }
                }
            }
        }
    }

    // 5. Gentle 3D baseline roll
    if cfg.baseline_roll_amp > 0.0 {
        let roll = cfg.baseline_roll_amp;
        let inv_nx = 1.0 / (cfg.nx_h.max(1) as f64);
        let inv_ny = 1.0 / (cfg.ny_n.max(1) as f64);
        let inv_nz = 1.0 / (cfg.nz_c.max(1) as f64);
        wx_buf.clear();
        for ix in 0..cfg.nx_h {
            wx_buf.push((1.2 * TAU * (ix as f64) * inv_nx + 0.35).sin());
        }
        for iz in 0..cfg.nz_c {
            let cz = (TAU * (iz as f64) * inv_nz + 0.4).cos();
            let z_plane = iz * cfg.ny_n;
            for iy in 0..cfg.ny_n {
                let cyz = roll * cz * (TAU * (iy as f64) * inv_ny + 0.2).cos();
                let base = (z_plane + iy) * cfg.nx_h;
                for (xi, &bx) in wx_buf.iter().enumerate() {
                    data[base + xi] += (cyz * bx) as f32;
                }
            }
        }
    }

    // 6. Thermal receiver noise
    fill_gaussian_noise_fast(&mut data, cfg.noise_level, &mut rng);

    Spectrum3D {
        name: format!("Synthetic 3D {exp}"),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}
