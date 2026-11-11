//! Integration tests for realistic Bruker NMR peak-list and volumetric 2D/3D spectrum simulation,
//! plus regression checks confirming `SimConfig::default()` and `SpectrumRenderConfig::default()`
//! preserve exact deterministic PRNG parity.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nmr_core::neural_picker::{run_e2e_blind_benchmark, E2eBenchmarkConfig};
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::Peak;
use nmr_core::picker::{estimate_noise_mad, pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::Priors;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

fn ubiquitin_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2]
}

#[test]
fn default_sim_and_render_preserve_exact_regression_parity() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read synthetic_ubiquitin.str");
    let (peaks1, truth1) = simulate(&ref_data, &SimConfig::default());
    let (peaks2, truth2) = simulate(&ref_data, &SimConfig::default());

    assert_eq!(
        peaks1, peaks2,
        "SimConfig::default() must be strictly deterministic"
    );
    assert_eq!(truth1, truth2);
    assert_eq!(
        peaks1.len(),
        855,
        "exact baseline synthetic_ubiquitin peak count"
    );
    assert_eq!(
        truth1.len(),
        84,
        "exact baseline synthetic_ubiquitin HSQC root count"
    );
    assert_eq!(
        truth1.iter().filter(|&&t| t >= 0).count(),
        68,
        "exact baseline assignable root count"
    );

    let systems =
        build_spin_systems(&peaks1, &Tolerances::default(), 3, false).expect("spin systems");
    let res = assign(
        &systems,
        &ref_data.sequence,
        &Priors::default(),
        &ScoreConfig::default(),
        &AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let assigned = res.position.iter().filter(|&&p| p >= 0).count();
    let correct = res
        .position
        .iter()
        .enumerate()
        .filter(|&(s, &p)| p >= 0 && p == truth1[s])
        .count();
    let acc = (correct as f64) / (assigned as f64);
    assert!(
        acc >= 0.957,
        "baseline CPU accuracy regressed: {acc:.4} ({correct}/{assigned})"
    );

    let dcfg = SpectrumRenderConfig::default();
    let hsqc = render_hsqc_2d(&peaks1, &dcfg);
    assert!((hsqc.axis_x.first_ppm - 10.6).abs() < 1e-6);
    assert!((hsqc.axis_x.last_ppm() - 5.8).abs() < 1e-6);
    assert!((hsqc.axis_y.first_ppm - 136.0).abs() < 1e-6);
    assert!((hsqc.axis_y.last_ppm() - 102.0).abs() < 1e-6);
}

#[test]
fn realistic_bruker_peak_simulation_physics() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read synthetic_ubiquitin.str");
    let rcfg = SimConfig::realistic_bruker();
    let (peaks, truth) = simulate(&ref_data, &rcfg);

    // 1. Exact 5-spectrum Bruker experiment set (no HNCA, no HNCOCA)
    let exps: BTreeSet<&str> = peaks.iter().map(|p| p.exp.as_str()).collect();
    let expected: BTreeSet<&str> = ["HSQC", "HNCACB", "CBCACONH", "HNCO", "HNCACO"]
        .into_iter()
        .collect();
    assert_eq!(exps, expected);
    assert!(!peaks.iter().any(|p| p.exp == "HNCA" || p.exp == "HNCOCA"));

    // 2. Verify -2.75 ppm 13C referencing offset (and H/N offsets) on true backbone roots
    let hsqc_peaks: Vec<&Peak> = peaks.iter().filter(|p| p.exp == "HSQC").collect();
    assert_eq!(hsqc_peaks.len(), truth.len());

    let mut dh_list = Vec::new();
    let mut dn_list = Vec::new();
    let mut dco_list = Vec::new();
    let mut dca_list = Vec::new();
    let mut true_heights = Vec::new();

    for (pk, &res_idx) in hsqc_peaks.iter().zip(truth.iter()) {
        if res_idx >= 0 {
            let k = res_idx as usize;
            let th = ref_data.shifts[&(k, "H".to_string())];
            let tn = ref_data.shifts[&(k, "N".to_string())];
            dh_list.push(pk.h - th);
            dn_list.push(pk.n - tn);
            true_heights.push(pk.height.expect("realistic intensity"));

            if k > 0 {
                if let Some(&tc_prev) = ref_data.shifts.get(&(k - 1, "C".to_string())) {
                    // Find closest strong HNCO peak in this root's strip
                    if let Some(best_co) = peaks
                        .iter()
                        .filter(|q| {
                            q.exp == "HNCO"
                                && (q.h - pk.h).abs() <= 0.04
                                && (q.n - pk.n).abs() <= 0.30
                                && q.height.unwrap_or(0.0) >= 20.0
                        })
                        .max_by(|a, b| a.height.unwrap_or(0.0).total_cmp(&b.height.unwrap_or(0.0)))
                    {
                        dco_list.push(best_co.c.unwrap() - tc_prev);
                    }
                }
            }
            if let Some(&tca) = ref_data.shifts.get(&(k, "CA".to_string())) {
                if let Some(best_ca) = peaks
                    .iter()
                    .filter(|q| {
                        q.exp == "HNCACB"
                            && q.sign > 0
                            && (q.h - pk.h).abs() <= 0.04
                            && (q.n - pk.n).abs() <= 0.30
                            && q.height.unwrap_or(0.0) >= 25.0
                    })
                    .min_by(|a, b| {
                        (a.c.unwrap() - (tca - 2.75))
                            .abs()
                            .total_cmp(&(b.c.unwrap() - (tca - 2.75)).abs())
                    })
                {
                    dca_list.push(best_ca.c.unwrap() - tca);
                }
            }
        }
    }

    let med_dh = median(&mut dh_list);
    let med_dn = median(&mut dn_list);
    let med_dco = median(&mut dco_list);
    let med_dca = median(&mut dca_list);
    assert!(
        (med_dh - (-0.02)).abs() < 0.01,
        "expected ~ -0.02 ppm 1H offset, got {med_dh:.4}"
    );
    assert!(
        (med_dn - 0.25).abs() < 0.04,
        "expected ~ +0.25 ppm 15N offset, got {med_dn:.4}"
    );
    assert!(
        (med_dco - (-2.75)).abs() < 0.06,
        "expected -2.75 ppm HNCO 13C offset, got {med_dco:.4}"
    );
    assert!(
        (med_dca - (-2.75)).abs() < 0.06,
        "expected -2.75 ppm HNCACB 13CA offset, got {med_dca:.4}"
    );

    // 3. Wide conformational-exchange intensity dispersion & missing roots
    true_heights.sort_by(|a, b| a.total_cmp(b));
    let p10 = true_heights[true_heights.len() / 10];
    let p90 = true_heights[true_heights.len() * 9 / 10];
    assert!(
        p90 / p10 > 3.5,
        "expected wide log-normal exchange dispersion (p90/p10 > 3.5), got {:.2}",
        p90 / p10
    );

    // 4. Paired Asn/Gln NH2 doublets (ubiquitin has 8 Asn+Gln residues)
    let spurious_hsqc: Vec<&Peak> = hsqc_peaks
        .iter()
        .zip(truth.iter())
        .filter_map(|(&p, &t)| (t < 0).then_some(p))
        .collect();
    let mut nh2_pairs = 0usize;
    for i in 0..spurious_hsqc.len() {
        for j in (i + 1)..spurious_hsqc.len() {
            let a = spurious_hsqc[i];
            let b = spurious_hsqc[j];
            let dn = (a.n - b.n).abs();
            let dh = (a.h - b.h).abs();
            if dn <= 0.12
                && (0.30..=1.20).contains(&dh)
                && a.h > 6.2
                && a.h < 8.4
                && b.h > 6.2
                && b.h < 8.4
            {
                nh2_pairs += 1;
            }
        }
    }
    assert!(
        nh2_pairs >= 6,
        "expected >= 6 paired Asn/Gln NH2 doublets sharing 15N, found {nh2_pairs}"
    );

    // 5. Water (~4.77 ppm) and upfield (< 4.3 ppm) spurious HSQC roots
    assert!(
        spurious_hsqc.iter().any(|p| (p.h - 4.75).abs() < 0.18),
        "expected residual water spurious HSQC root near ~4.77 ppm"
    );
    assert!(
        spurious_hsqc.iter().any(|p| p.h < 4.3),
        "expected upfield < 4.3 ppm aliphatic spurious HSQC root"
    );
}

#[test]
fn realistic_bruker_volumetric_sweep_widths_and_dynamic_range() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read synthetic_ubiquitin.str");
    let (peaks, _) = simulate(&ref_data, &SimConfig::realistic_bruker());
    let rcfg = SpectrumRenderConfig::realistic_bruker();

    let hsqc = render_hsqc_2d(&peaks, &rcfg);
    assert_eq!((hsqc.axis_x.size, hsqc.axis_y.size), (768, 128));
    assert!((hsqc.axis_x.first_ppm - 11.75).abs() < 1e-6);
    assert!((hsqc.axis_x.last_ppm() - (-2.25)).abs() < 1e-6);
    assert!((hsqc.axis_y.first_ppm - 137.5).abs() < 1e-6);
    assert!((hsqc.axis_y.last_ppm() - 100.2).abs() < 1e-6);

    let mad_2d = estimate_noise_mad(&hsqc.data);
    let max_2d = hsqc.data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let dyn_2d = max_2d / mad_2d;
    assert!(
        dyn_2d > 1000.0,
        "expected 2D HSQC dynamic range max_abs/MAD > 1000, got {dyn_2d:.1} (max={max_2d}, mad={mad_2d})"
    );

    let hncacb = render_triple_3d("HNCACB", &peaks, &rcfg);
    assert!((hncacb.axis_x.first_ppm - 11.75).abs() < 1e-6);
    assert!((hncacb.axis_x.last_ppm() - (-2.25)).abs() < 1e-6);
    assert!((hncacb.axis_y.first_ppm - 137.5).abs() < 1e-6);
    assert!((hncacb.axis_y.last_ppm() - 100.2).abs() < 1e-6);
    assert!((hncacb.axis_z.first_ppm - 76.5).abs() < 1e-6);
    assert!((hncacb.axis_z.last_ppm() - 1.63).abs() < 1e-6);

    let mad_cb = estimate_noise_mad(&hncacb.data);
    let max_cb = hncacb.data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    assert!(
        max_cb / mad_cb > 1000.0,
        "expected 3D HNCACB dynamic range max_abs/MAD > 1000, got {:.1}",
        max_cb / mad_cb
    );

    let hnco = render_triple_3d("HNCO", &peaks, &rcfg);
    assert!((hnco.axis_z.first_ppm - 187.0).abs() < 1e-6);
    assert!((hnco.axis_z.last_ppm() - 165.0).abs() < 1e-6);
    let mad_co = estimate_noise_mad(&hnco.data);
    let max_co = hnco.data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    assert!(
        max_co / mad_co > 1000.0,
        "expected 3D HNCO dynamic range max_abs/MAD > 1000, got {:.1}",
        max_co / mad_co
    );
}

#[test]
fn realistic_bruker_indirect_sinc_ripples_and_t1_ridges_above_5_mad() {
    // Isolate a single intense parent amide peak at (8.35 ppm, 120.0 ppm) to measure
    // pure indirect sinc sidelobes and t1 ridge noise unambiguously.
    let single_peaks = vec![
        Peak::hsqc(8.35, 120.0).with_intensity(160.0, Some(672.0)),
        Peak::triple("HNCACB", 8.35, 120.0, 54.0, 1).with_intensity(150.0, Some(870.0)),
        Peak::triple("HNCO", 8.35, 120.0, 175.0, 1).with_intensity(160.0, Some(928.0)),
    ];
    let rcfg = SpectrumRenderConfig {
        nx_h: 512,
        ny_n: 128,
        nz_c: 256,
        ..SpectrumRenderConfig::realistic_bruker()
    };

    // --- 2D HSQC: indirect 15N sinc ripples + t1 ridge + ~4.77 ppm water stripe ---
    let hsqc = render_hsqc_2d(&single_peaks, &rcfg);
    let mad_2d = estimate_noise_mad(&hsqc.data);
    let peak_ix = hsqc.axis_x.ppm_to_index(8.35).round() as usize;
    let empty_ix = hsqc.axis_x.ppm_to_index(9.60).round() as usize;

    // Sinc sidelobes within 0.85..3.2 ppm in 15N of 120.0 ppm
    let mut min_sinc_2d = 0.0f32;
    let mut max_pos_sidelobe_2d = 0.0f32;
    let mut max_far_ridge_2d = 0.0f32;
    let mut ridge_sq_sum = 0.0f64;
    let mut empty_sq_sum = 0.0f64;
    let mut far_count = 0usize;

    for iy in 0..rcfg.ny_n {
        let n_ppm = hsqc.axis_y.index_to_ppm(iy as f64);
        let dn = (n_ppm - 120.0).abs();
        let val = hsqc.data[iy * rcfg.nx_h + peak_ix];
        let empty_val = hsqc.data[iy * rcfg.nx_h + empty_ix];
        if (0.85..=1.60).contains(&dn) {
            min_sinc_2d = min_sinc_2d.min(val);
        }
        if (1.65..=2.55).contains(&dn) {
            max_pos_sidelobe_2d = max_pos_sidelobe_2d.max(val);
        }
        if dn > 10.0 {
            max_far_ridge_2d = max_far_ridge_2d.max(val.abs());
            ridge_sq_sum += (val as f64) * (val as f64);
            empty_sq_sum += (empty_val as f64) * (empty_val as f64);
            far_count += 1;
        }
    }

    assert!(
        min_sinc_2d < -30.0 * mad_2d,
        "expected strong negative 1st sinc sidelobe (< -30 * MAD), got {:.3} (MAD={:.4})",
        min_sinc_2d,
        mad_2d
    );
    assert!(
        max_pos_sidelobe_2d > 30.0 * mad_2d,
        "expected strong positive 2nd sinc sidelobe (> +30 * MAD), got {:.3} (MAD={:.4})",
        max_pos_sidelobe_2d,
        mad_2d
    );
    // Relative sidelobe ratios (~2% to ~7.5% of the 160.0 parent peak)
    let neg_frac = (min_sinc_2d.abs() as f64) / 160.0;
    let pos_frac = (max_pos_sidelobe_2d as f64) / 160.0;
    assert!(
        (0.02..=0.08).contains(&neg_frac) && (0.015..=0.07).contains(&pos_frac),
        "expected ~2-6% sinc sidelobe ratios, got neg={neg_frac:.3}, pos={pos_frac:.3}"
    );
    let ridge_rms = (ridge_sq_sum / (far_count as f64)).sqrt();
    let empty_rms = (empty_sq_sum / (far_count as f64)).sqrt();
    assert!(
        max_far_ridge_2d > 5.0 * mad_2d && ridge_rms > 5.0 * empty_rms,
        "expected t1 ridge > 5 * MAD along strong column (max_far={max_far_ridge_2d:.3}, 5*MAD={:.3}, ridge_rms={ridge_rms:.3}, empty_rms={empty_rms:.3})",
        5.0 * mad_2d
    );

    // Verify residual water suppression artifact around ~4.77 ppm in 2D HSQC
    let mut max_water_2d = 0.0f32;
    let mut max_upfield_2d = 0.0f32;
    for iy in 0..rcfg.ny_n {
        for ix in 0..rcfg.nx_h {
            let h_ppm = hsqc.axis_x.index_to_ppm(ix as f64);
            let v = hsqc.data[iy * rcfg.nx_h + ix].abs();
            if (4.65..=4.90).contains(&h_ppm) {
                max_water_2d = max_water_2d.max(v);
            } else if h_ppm < 4.2 && h_ppm > 0.5 {
                max_upfield_2d = max_upfield_2d.max(v);
            }
        }
    }
    assert!(
        max_water_2d > 50.0 * mad_2d,
        "expected prominent ~4.77 ppm water artifact (> 50 * MAD), got {max_water_2d:.3} (MAD={mad_2d:.4})"
    );
    assert!(
        max_upfield_2d > 20.0 * mad_2d,
        "expected upfield < 4.3 ppm aliphatic bleed (> 20 * MAD), got {max_upfield_2d:.3}"
    );

    // --- 3D HNCACB: indirect 13C sinc ripples + t1 strip ridge + ~4.77 ppm water artifact ---
    let hncacb = render_triple_3d("HNCACB", &single_peaks, &rcfg);
    let mad_3d = estimate_noise_mad(&hncacb.data);
    let px = hncacb.axis_x.ppm_to_index(8.35).round() as usize;
    let py = hncacb.axis_y.ppm_to_index(120.0).round() as usize;

    let mut min_sinc_3d = 0.0f32;
    let mut max_pos_sidelobe_3d = 0.0f32;
    let mut max_far_ridge_3d = 0.0f32;
    for iz in 0..rcfg.nz_c {
        let c_ppm = hncacb.axis_z.index_to_ppm(iz as f64);
        let dc = (c_ppm - 54.0).abs();
        let val = hncacb.data[(iz * rcfg.ny_n + py) * rcfg.nx_h + px];
        if (0.85..=1.55).contains(&dc) {
            min_sinc_3d = min_sinc_3d.min(val);
        }
        if (1.60..=2.45).contains(&dc) {
            max_pos_sidelobe_3d = max_pos_sidelobe_3d.max(val);
        }
        if dc > 12.0 {
            max_far_ridge_3d = max_far_ridge_3d.max(val.abs());
        }
    }
    assert!(
        min_sinc_3d < -25.0 * mad_3d,
        "expected 3D 13C negative 1st sinc sidelobe (< -25 * MAD), got {min_sinc_3d:.3} (MAD={mad_3d:.4})"
    );
    assert!(
        max_pos_sidelobe_3d > 20.0 * mad_3d,
        "expected 3D 13C positive 2nd sinc sidelobe (> +20 * MAD), got {max_pos_sidelobe_3d:.3} (MAD={mad_3d:.4})"
    );
    assert!(
        max_far_ridge_3d > 5.0 * mad_3d,
        "expected 3D 13C strip t1 ridge > 5 * MAD far from peak, got {max_far_ridge_3d:.3} (MAD={mad_3d:.4})"
    );

    let wx = hncacb.axis_x.ppm_to_index(4.77).round() as usize;
    let mut max_water_3d = 0.0f32;
    for iz in 0..rcfg.nz_c {
        for iy in 0..rcfg.ny_n {
            max_water_3d =
                max_water_3d.max(hncacb.data[(iz * rcfg.ny_n + iy) * rcfg.nx_h + wx].abs());
        }
    }
    assert!(
        max_water_3d > 20.0 * mad_3d,
        "expected 3D water artifact at ~4.77 ppm (> 20 * MAD), got {max_water_3d:.3}"
    );

    // Also verify 3D HNCO carbonyl indirect 13C sinc sidelobes
    let hnco = render_triple_3d("HNCO", &single_peaks, &rcfg);
    let mad_co = estimate_noise_mad(&hnco.data);
    let mut min_sinc_co = 0.0f32;
    let mut max_pos_co = 0.0f32;
    for iz in 0..rcfg.nz_c {
        let c_ppm = hnco.axis_z.index_to_ppm(iz as f64);
        let dc = (c_ppm - 175.0).abs();
        let val = hnco.data[(iz * rcfg.ny_n + py) * rcfg.nx_h + px];
        if (0.24..=0.46).contains(&dc) {
            min_sinc_co = min_sinc_co.min(val);
        }
        if (0.47..=0.72).contains(&dc) {
            max_pos_co = max_pos_co.max(val);
        }
    }
    assert!(
        min_sinc_co < -25.0 * mad_co && max_pos_co > 20.0 * mad_co,
        "expected HNCO 13C sinc ripples above 20 * MAD (min={min_sinc_co:.3}, pos={max_pos_co:.3}, MAD={mad_co:.4})"
    );

    // Verify picking operates cleanly without panicking on uncropped realistic spectra
    let picked_2d = pick_hsqc_2d(&hsqc, &PickerConfig::default());
    assert!(!picked_2d.is_empty());
    let picked_3d = pick_triple_3d(
        &hncacb,
        &[Peak::hsqc(8.35, 120.0)],
        &PickerConfig {
            allow_negative: true,
            ..PickerConfig::default()
        },
    );
    assert!(!picked_3d.is_empty());
}

#[test]
fn realistic_bruker_e2e_benchmark_config_executes_cleanly() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read synthetic_ubiquitin.str");
    let defaults = E2eBenchmarkConfig::realistic_bruker();
    assert!(defaults.realistic_bruker_mode);
    assert!((defaults.ref_offset_c_ppm - (-2.75)).abs() < 1e-9);
    assert!(defaults.include_sidechain_nh2);
    assert!(defaults.sinc_amp > 0.0);

    // Use moderate grid resolution so CPU neural + classical 3D inference runs in < 2 seconds
    let fast_realistic = E2eBenchmarkConfig {
        nx_h_2d: 256,
        ny_n_2d: 96,
        nx_h_3d: 160,
        ny_n_3d: 64,
        nz_c_3d: 128,
        anneal_runs: 6,
        anneal_steps: 600,
        ..defaults
    };
    let cmp = run_e2e_blind_benchmark(&ref_data, &fast_realistic).expect("realistic e2e benchmark");
    let exp_names: Vec<&str> = cmp
        .classical
        .per_experiment
        .iter()
        .map(|m| m.experiment.as_str())
        .collect();
    assert_eq!(
        exp_names,
        vec!["HSQC", "HNCACB", "CBCACONH", "HNCO", "HNCACO"],
        "realistic Bruker e2e benchmark must use exact 5-spectrum set without HNCA/HNCOCA"
    );
    assert!(cmp.classical.assignable_residues >= 60);
    assert!(!cmp.classical.overall_peaks.f1.is_nan());
    assert!(!cmp.neural.overall_peaks.f1.is_nan());
}
