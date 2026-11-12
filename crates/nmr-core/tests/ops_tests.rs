//! Comprehensive unit and real-data blind autophasing tests for `nmr_core::ops`.

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nmr_core::fid::{read_fid_2d, read_fid_3d, Complex64};
use nmr_core::ops::{
    apply_linear_prediction, auto_phase_1d, auto_phase_1d_real, auto_phase_direct_nd,
    auto_phase_indirect_nd, blind_autophase_fid_2d, blind_autophase_fid_3d, fft_arbitrary_in_place,
    fftshift_in_place, fit_burg_ar_stabilized, format_plane_template, hilbert_transform_in_place,
    phase_diff_deg, phase_diff_mod180_deg, read_ft1_1d, read_ft3_plane_series, reconstruct_nus_1d,
    transpose_2d, transpose_3d, transpose_3d_xy, transpose_3d_xz, transpose_3d_yz, write_ft1_1d,
    write_ft3_plane_series, AutoPhaseConfig, AxisPipeline, LpMode, NusConfig, NusMethod,
    PolarityMode, SolventKernel, Spectrum1D, StageOp, Trace1D,
};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, AxisPpm, Spectrum2D, Spectrum3D,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve local git-ignored `data/` directory via `NMR_REAL_DATA_DIR`, workspace `data/`,
/// or `--git-common-dir` parent `data/` when running inside a git worktree.
fn resolve_local_real_data_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(custom);
        return p.is_dir().then_some(p);
    }
    let ws_data = workspace_root().join("data");
    if ws_data.join("ubiquitin").is_dir() {
        return Some(ws_data);
    }
    let out = Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .current_dir(workspace_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let git_common = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let common_data = git_common.parent()?.join("data");
    common_data
        .join("ubiquitin")
        .is_dir()
        .then_some(common_data)
}

fn temp_test_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("nmrassign_ops_test_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("create temp test dir");
    p
}

#[test]
fn apodization_windows_and_j_modulation_behave_as_derived() {
    let n = 128usize;
    let sw_hz = 2000.0;
    let ones = vec![[1.0, 0.0]; n];

    // 1. SineWindow: pure sine bell (`off = 0, end = 1, pow = 1`) and cosine-squared (`off = 0.5, end = 1, pow = 2`)
    let mut t_sine = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::SineWindow {
        off: 0.0,
        end: 1.0,
        pow: 1.0,
        first_scale: 1.0,
    }])
    .apply(&mut t_sine)
    .unwrap();
    assert!(t_sine.data[0][0].abs() < 1e-12);
    assert!(t_sine.data[n - 1][0].abs() < 1e-12);
    assert!((t_sine.data[n / 2][0] - 1.0).abs() < 2e-3);

    let mut t_cos2 = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::SineWindow {
        off: 0.5,
        end: 1.0,
        pow: 2.0,
        first_scale: 0.5,
    }])
    .apply(&mut t_cos2)
    .unwrap();
    assert!((t_cos2.data[0][0] - 0.5).abs() < 1e-12);
    assert!(t_cos2.data[n - 1][0].abs() < 1e-12);

    // 2. ExpWindow: positive broadening vs negative narrowing
    let mut t_em = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::ExpWindow {
        lb_hz: 10.0,
        first_scale: 0.5,
    }])
    .apply(&mut t_em)
    .unwrap();
    assert!((t_em.data[0][0] - 0.5).abs() < 1e-12);
    assert!(t_em.data[64][0] < t_em.data[1][0]);

    let mut t_em_neg = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::ExpWindow {
        lb_hz: -5.0,
        first_scale: 1.0,
    }])
    .apply(&mut t_em_neg)
    .unwrap();
    assert!(t_em_neg.data[32][0] > 1.0);

    // 3. LorentzToGauss: narrows a decaying exponential FID peak FWHM in frequency domain
    let fid_decay: Vec<Complex64> = (0..256)
        .map(|k| {
            let t = (k as f64) / sw_hz;
            [(-PI * 8.0 * t).exp(), 0.0]
        })
        .collect();
    let mut tr_raw = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, fid_decay.clone());
    AxisPipeline::new(vec![
        StageOp::ExpWindow {
            lb_hz: 0.0,
            first_scale: 0.5,
        },
        StageOp::ZeroFill {
            target_size: Some(1024),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: false,
            sign_alt: false,
            real_only: false,
        },
    ])
    .apply(&mut tr_raw)
    .unwrap();

    let mut tr_gm = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, fid_decay);
    AxisPipeline::new(vec![
        StageOp::LorentzToGauss {
            g1_hz: 8.0,
            g2_hz: 5.0,
            g3_center: 0.0,
            first_scale: 0.5,
        },
        StageOp::ZeroFill {
            target_size: Some(1024),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: false,
            sign_alt: false,
            real_only: false,
        },
    ])
    .apply(&mut tr_gm)
    .unwrap();

    let fwhm = |re: &[f64]| -> usize {
        let peak = re[512];
        re.iter().filter(|&&v| v >= 0.5 * peak).count()
    };
    assert!(
        fwhm(&tr_gm.real_vec()) < fwhm(&tr_raw.real_vec()),
        "LorentzToGauss should narrow FWHM relative to unapodized Lorentzian"
    );

    // 4. Trapezoid & Triangle
    let mut t_trap = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::Trapezoid {
        t1: 16,
        t2: 96,
        first_scale: 1.0,
    }])
    .apply(&mut t_trap)
    .unwrap();
    assert!((t_trap.data[0][0] - 0.0).abs() < 1e-12);
    assert!((t_trap.data[8][0] - 0.5).abs() < 1e-12);
    assert!((t_trap.data[48][0] - 1.0).abs() < 1e-12);
    assert!((t_trap.data[127][0] - 0.0).abs() < 1e-12);

    let mut t_tri = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, ones.clone());
    AxisPipeline::new(vec![StageOp::Triangle {
        peak: 64,
        first_scale: 1.0,
    }])
    .apply(&mut t_tri)
    .unwrap();
    assert!((t_tri.data[64][0] - 1.0).abs() < 1e-12);
    assert!((t_tri.data[32][0] - 0.5).abs() < 1e-12);

    // 5. JModulation (`j_hz = 50 Hz`, `sw_hz = 2000 Hz` -> first null of cos(pi J t) at k = 20)
    let mut t_j = Trace1D::new_time_domain("13C", sw_hz, 125.0, 180.0, ones);
    AxisPipeline::new(vec![StageOp::JModulation {
        j_hz: 50.0,
        phase_deg: 0.0,
        sin_mode: false,
    }])
    .apply(&mut t_j)
    .unwrap();
    assert!((t_j.data[0][0] - 1.0).abs() < 1e-12);
    assert!(t_j.data[20][0].abs() < 1e-12);
}

#[test]
fn group_delay_subsample_and_solvent_filters_preserve_amide_peaks() {
    let n = 512usize;
    let sw_hz = 6000.0;
    let delay_pts = 67.73; // non-integer group delay!
    let f_amide = 0.21; // normalized frequency well clear of carrier DC

    // Delayed DQD FID (`negate_imag = true` convention: raw = exp(-2 pi i f (k - G)) for k >= 68)
    let mut raw = vec![[0.0, 0.0]; n];
    for (k, slot) in raw.iter_mut().enumerate().skip(68) {
        let u = (k as f64) - delay_pts;
        let env = (-u / 180.0).exp();
        let ang = -2.0 * PI * f_amide * u;
        // Superimpose a 50x stronger zero-frequency solvent signal
        *slot = [env * ang.cos() + 50.0 * (-u / 400.0).exp(), env * ang.sin()];
    }

    for shape in [
        SolventKernel::Gaussian,
        SolventKernel::SineBell,
        SolventKernel::Boxcar,
        SolventKernel::PolyLowPass,
    ] {
        let mut tr = Trace1D::new_time_domain("1H", sw_hz, 600.0, 10.0, raw.clone());
        let pipe = AxisPipeline::new(vec![
            StageOp::GroupDelay { delay_pts },
            StageOp::SolventFilter {
                shape,
                half_width: 24,
                mirror_edges: true,
            },
            StageOp::ExpWindow {
                lb_hz: 2.0,
                first_scale: 0.5,
            },
            StageOp::ZeroFill {
                target_size: Some(1024),
                factor: None,
                pad: None,
                auto_pow2: true,
            },
            StageOp::FourierTransform {
                inverse: false,
                negate_imag: true,
                sign_alt: false,
                real_only: false,
            },
        ]);
        pipe.apply(&mut tr).unwrap();

        let re = tr.real_vec();
        // Carrier sits at bin 512; amide sits near bin `512 + round(0.21 * 1024) = 727`
        let carrier_amp = re[512].abs();
        let amide_peak_idx = (720..735).max_by(|&a, &b| re[a].total_cmp(&re[b])).unwrap();
        let amide_amp = re[amide_peak_idx];
        assert!(
            amide_amp > 50.0 && carrier_amp < 0.08 * amide_amp,
            "SolventFilter {shape:?} failed: amide_amp={amide_amp}, carrier_amp={carrier_amp}"
        );
        // Sub-sample GroupDelay compensation must yield symmetric zero-phase absorption around `amide_peak_idx`
        let wing_asym = (re[amide_peak_idx + 3] - re[amide_peak_idx - 3]).abs() / amide_amp;
        assert!(
            wing_asym < 0.06,
            "Sub-sample GroupDelay left dispersive asymmetry {wing_asym} for {shape:?}"
        );
    }
}

#[test]
fn linear_prediction_forward_backward_mirror_image_and_root_stability() {
    // 1. Forward LP on two damped complex exponentials: extrapolates 64 -> 128 accurately
    let n_orig = 64usize;
    let synth = |k: usize| -> Complex64 {
        let kf = k as f64;
        let a1 = (-0.008 * kf).exp();
        let p1 = 2.0 * PI * 0.125 * kf + 0.3;
        let a2 = 0.7 * (-0.010 * kf).exp();
        let p2 = -2.0 * PI * 0.28 * kf - 0.5;
        [a1 * p1.cos() + a2 * p2.cos(), a1 * p1.sin() + a2 * p2.sin()]
    };
    let mut fwd_vec: Vec<Complex64> = (0..n_orig).map(synth).collect();
    apply_linear_prediction(
        &mut fwd_vec,
        LpMode::Forward {
            n_pred: 64,
            order: 8,
        },
    );
    assert_eq!(fwd_vec.len(), 128);
    let err_fwd: f64 = (64..96)
        .map(|k| {
            let t = synth(k);
            (fwd_vec[k][0] - t[0]).hypot(fwd_vec[k][1] - t[1])
        })
        .sum::<f64>()
        / 32.0;
    assert!(err_fwd < 1e-3, "Forward LP mean error too high: {err_fwd}");

    // 2. Backward LP reconstructs 4 corrupted initial points (`1e4` spikes) cleanly
    let mut bwd_vec: Vec<Complex64> = (0..64).map(synth).collect();
    let truth_head: Vec<Complex64> = bwd_vec[0..4].to_vec();
    for c in &mut bwd_vec[0..4] {
        *c = [1.0e4, -7.5e3];
    }
    apply_linear_prediction(
        &mut bwd_vec,
        LpMode::Backward {
            n_before: 4,
            order: 8,
        },
    );
    for k in 0..4 {
        let err = (bwd_vec[k][0] - truth_head[k][0]).hypot(bwd_vec[k][1] - truth_head[k][1]);
        assert!(
            err < 0.05,
            "Backward LP failed at k={k}: got {:?} vs {:?}",
            bwd_vec[k],
            truth_head[k]
        );
    }

    // 3. Mirror-Image LP (`half_dwell = false` AND `half_dwell = true`, including biphasic +CA/-CB!)
    for half_dwell in [false, true] {
        let dt_off = if half_dwell { 0.5 } else { 0.0 };
        let milp_sig = |k: usize| -> Complex64 {
            let t = (k as f64) + dt_off;
            // +1.0 (`CA`) and -0.8 (`CB`) constant-time undamped interferogram
            let th1 = 2.0 * PI * 0.15 * t;
            let th2 = -2.0 * PI * 0.31 * t;
            [
                1.0 * th1.cos() - 0.8 * th2.cos(),
                1.0 * th1.sin() - 0.8 * th2.sin(),
            ]
        };
        let mut milp_vec: Vec<Complex64> = (0..24).map(milp_sig).collect();
        apply_linear_prediction(
            &mut milp_vec,
            LpMode::MirrorImage {
                n_pred: 24,
                order: 10,
                half_dwell,
            },
        );
        assert_eq!(milp_vec.len(), 48);
        for (k, &pred) in milp_vec.iter().enumerate().skip(24).take(24) {
            let tr = milp_sig(k);
            let err = (pred[0] - tr[0]).hypot(pred[1] - tr[1]);
            assert!(
                err < 2e-2,
                "MirrorImage LP (half_dwell={half_dwell}) error at k={k}: {err}"
            );
        }
    }

    // 4. Unit-circle root reflection stability test on an exponentially GROWING noisy input!
    let mut growing: Vec<Complex64> = (0..48)
        .map(|k| {
            let kf = k as f64;
            let amp = (0.09 * kf).exp(); // unstable growing input!
            let jitter = ((k * 73 + 19) % 29) as f64 * 0.01;
            [
                amp * (0.7 * kf).cos() + jitter,
                amp * (0.7 * kf).sin() - jitter,
            ]
        })
        .collect();
    let max_input = growing
        .iter()
        .map(|c| c[0].hypot(c[1]))
        .fold(0.0_f64, f64::max);
    let ar = fit_burg_ar_stabilized(&growing, 16);
    assert_eq!(ar.len(), 16);
    apply_linear_prediction(
        &mut growing,
        LpMode::Forward {
            n_pred: 256,
            order: 16,
        },
    );
    // Tail after 256 steps must decay / stay strictly bounded (never blow up!)
    let tail_amp = growing[303][0].hypot(growing[303][1]);
    assert!(
        tail_amp.is_finite() && tail_amp <= max_input * 1.05,
        "Root reflection failed to stabilize growing signal: tail_amp={tail_amp}, max_input={max_input}"
    );
}

#[test]
fn bluestein_arbitrary_fft_and_hilbert_transform_equivalence_under_1e3() {
    // 1. Prime-length Bluestein FFT -> IFFT round-trip (`N = 127` and `N = 120`)
    for n in [120usize, 127usize] {
        let orig: Vec<Complex64> = (0..n)
            .map(|k| {
                let x = k as f64;
                [(0.37 * x).cos() - 0.4 * (1.1 * x).sin(), (0.83 * x).sin()]
            })
            .collect();
        let mut buf = orig.clone();
        fft_arbitrary_in_place(&mut buf, false);
        fft_arbitrary_in_place(&mut buf, true);
        let max_err = orig
            .iter()
            .zip(&buf)
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "Bluestein round-trip error for N={n}: {max_err}"
        );
    }

    // 2. `HilbertTransform -> Phase -> DeleteImaginary` exact equivalence to complex phasing (`< 1e-3`!)
    let n_td = 1024usize;
    let sw_hz = 5000.0;
    let fid: Vec<Complex64> = (0..n_td)
        .map(|k| {
            let t = (k as f64) / sw_hz;
            let s1 = (-PI * 3.0 * t).exp();
            let a1 = 2.0 * PI * 620.0 * t + 1.15;
            let s2 = 0.65 * (-PI * 4.0 * t).exp();
            let a2 = -2.0 * PI * 1140.0 * t - 0.72;
            [s1 * a1.cos() + s2 * a2.cos(), s1 * a1.sin() + s2 * a2.sin()]
        })
        .collect();

    let base_ops = vec![
        StageOp::SineWindow {
            off: 0.35,
            end: 1.0,
            pow: 2.0,
            first_scale: 0.5,
        },
        StageOp::ZeroFill {
            target_size: Some(4096),
            factor: None,
            pad: None,
            auto_pow2: true,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: false,
            sign_alt: false,
            real_only: false,
        },
    ];

    let mut tr_complex = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, fid.clone());
    let mut p_complex = AxisPipeline::new(base_ops.clone());
    p_complex
        .push(StageOp::Phase {
            p0_deg: -137.4,
            p1_deg: 42.5,
            pivot_frac: 0.25,
        })
        .push(StageOp::DeleteImaginary);
    p_complex.apply(&mut tr_complex).unwrap();

    let mut tr_hilbert = Trace1D::new_time_domain("1H", sw_hz, 500.0, 10.0, fid);
    let mut p_hilbert = AxisPipeline::new(base_ops);
    p_hilbert
        .push(StageOp::DeleteImaginary)
        .push(StageOp::HilbertTransform)
        .push(StageOp::Phase {
            p0_deg: -137.4,
            p1_deg: 42.5,
            pivot_frac: 0.25,
        })
        .push(StageOp::DeleteImaginary);
    p_hilbert.apply(&mut tr_hilbert).unwrap();

    let ref_re = tr_complex.real_vec();
    let ht_re = tr_hilbert.real_vec();
    let peak_max = ref_re.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_rel_err = ref_re
        .iter()
        .zip(&ht_re)
        .map(|(a, b)| (a - b).abs() / peak_max)
        .fold(0.0_f64, f64::max);
    assert!(
        max_rel_err < 1e-3,
        "HilbertTransform -> Phase -> DeleteImaginary relative error {max_rel_err} >= 1e-3"
    );
}

#[test]
fn circular_shift_ppm_unwraps_nyquist_aliased_peaks_and_reverse_transpose_work() {
    // 256-pt 15N axis spanning `[100.0, 132.0)` (`first_ppm = 132.0`, `step_ppm = -0.125`)
    let axis = AxisPpm {
        label: "15N".to_string(),
        size: 256,
        first_ppm: 132.0,
        step_ppm: -0.125,
        obs_mhz: 50.68,
    };
    let mut bins = vec![0.0; 256];
    // Peak A at 120.0 ppm -> index `(120 - 132) / -0.125 = 96` (unaliased)
    bins[96] = 10.0;
    // Peak B has true shift 135.0 ppm, Nyquist-aliased into `[100, 132]` at `135 - 32 = 103.0 ppm` -> index `232`
    bins[232] = 8.0;

    let mut tr = Trace1D::new_freq_real(axis, &bins);
    AxisPipeline::new(vec![StageOp::CircularShiftPpm { delta_ppm: 5.0 }])
        .apply(&mut tr)
        .unwrap();

    let re = tr.real_vec();
    let idx_a = (0..256).find(|&i| (re[i] - 10.0).abs() < 1e-9).unwrap();
    let idx_b = (0..256).find(|&i| (re[i] - 8.0).abs() < 1e-9).unwrap();
    assert!((tr.axis.index_to_ppm(idx_a as f64) - 120.0).abs() < 1e-12);
    assert!((tr.axis.index_to_ppm(idx_b as f64) - 135.0).abs() < 1e-12);

    // Reverse convention: index 0 and N/2 stay fixed, applying twice is exact identity
    let orig_data = tr.data.clone();
    AxisPipeline::new(vec![StageOp::Reverse])
        .apply(&mut tr)
        .unwrap();
    assert_eq!(tr.data[0], orig_data[0]);
    assert_eq!(tr.data[128], orig_data[128]);
    AxisPipeline::new(vec![StageOp::Reverse])
        .apply(&mut tr)
        .unwrap();
    assert_eq!(tr.data, orig_data);

    // ExtractPpm crops cleanly and preserves exact ppm coordinates
    AxisPipeline::new(vec![StageOp::ExtractPpm {
        min_ppm: 118.0,
        max_ppm: 136.0,
    }])
    .apply(&mut tr)
    .unwrap();
    let cropped = tr.real_vec();
    let idx_a2 = (0..cropped.len())
        .find(|&i| (cropped[i] - 10.0).abs() < 1e-9)
        .unwrap();
    assert!((tr.axis.index_to_ppm(idx_a2 as f64) - 120.0).abs() < 1e-12);

    // 2D and 3D transpose round-trips
    let s2 = Spectrum2D {
        name: "t2".into(),
        axis_x: AxisPpm::from_bounds("1H", 5, 10.0, 6.0, 600.0),
        axis_y: AxisPpm::from_bounds("15N", 3, 130.0, 105.0, 60.8),
        data: (0..15).map(|i| i as f32).collect(),
    };
    let s2_t = transpose_2d(&s2);
    assert_eq!((s2_t.nx(), s2_t.ny()), (3, 5));
    assert_eq!(s2_t.get(2, 4), s2.get(4, 2));
    assert_eq!(transpose_2d(&s2_t).data, s2.data);

    let s3 = Spectrum3D {
        name: "t3".into(),
        exp: "HNCO".into(),
        axis_x: AxisPpm::from_bounds("1H", 4, 10.0, 6.0, 600.0),
        axis_y: AxisPpm::from_bounds("15N", 3, 130.0, 105.0, 60.8),
        axis_z: AxisPpm::from_bounds("13C", 2, 180.0, 170.0, 150.9),
        data: (0..24).map(|i| i as f32).collect(),
    };
    assert_eq!(transpose_3d_xy(&transpose_3d_xy(&s3)).data, s3.data);
    assert_eq!(transpose_3d_xz(&transpose_3d_xz(&s3)).data, s3.data);
    assert_eq!(transpose_3d_yz(&transpose_3d_yz(&s3)).data, s3.data);
    let cyc = transpose_3d(&s3, [1, 2, 0]).unwrap();
    assert_eq!((cyc.nx(), cyc.ny(), cyc.nz()), (3, 2, 4));
    assert_eq!(cyc.get(2, 1, 3), s3.get(3, 2, 1));
}

#[test]
fn poly_median_spline_baselines_and_nus_fista_virtual_echo() {
    let n = 512usize;
    let axis = AxisPpm::from_bounds("1H", n, 11.0, 5.0, 600.0);
    let peak_fn = |k: usize| -> f64 {
        let d1 = (k as f64 - 160.0) / 2.5;
        let d2 = (k as f64 - 340.0) / 2.5;
        25.0 / (1.0 + d1 * d1) + 18.0 / (1.0 + d2 * d2)
    };
    let base_fn = |k: usize| -> f64 {
        let x = 2.0 * (k as f64) / ((n - 1) as f64) - 1.0;
        3.5 - 2.2 * x + 4.1 * x * x - 1.3 * x * x * x
    };
    let raw: Vec<f64> = (0..n).map(|k| base_fn(k) + peak_fn(k)).collect();

    // 1. PolyBaseline (`order = 4`)
    let mut tr_poly = Trace1D::new_freq_real(axis.clone(), &raw);
    AxisPipeline::new(vec![StageOp::PolyBaseline {
        order: 4,
        sigma_mask: 2.5,
    }])
    .apply(&mut tr_poly)
    .unwrap();
    let re_poly = tr_poly.real_vec();
    assert!((re_poly[50]).abs() < 0.05);
    assert!((re_poly[160] - 25.0).abs() < 0.1);

    // 2. MedianBaseline (Friedrichs 1995)
    let mut tr_med = Trace1D::new_freq_real(axis.clone(), &raw);
    AxisPipeline::new(vec![StageOp::MedianBaseline {
        half_window_pts: 40,
    }])
    .apply(&mut tr_med)
    .unwrap();
    let re_med = tr_med.real_vec();
    assert!(re_med[80].abs() < 0.25);
    assert!((re_med[160] - 25.0).abs() < 0.35);

    // 3. NodeSplineBaseline
    let mut tr_sp = Trace1D::new_freq_real(axis, &raw);
    let pipe_sp = AxisPipeline::new(vec![StageOp::NodeSplineBaseline {
        node_indices: vec![0, 80, 250, 430, 511],
    }]);
    // Verify serde JSON round-trip of AxisPipeline simultaneously
    let json = serde_json::to_string(&pipe_sp).unwrap();
    let pipe_round: AxisPipeline = serde_json::from_str(&json).unwrap();
    assert_eq!(pipe_sp, pipe_round);
    pipe_round.apply(&mut tr_sp).unwrap();
    let re_sp = tr_sp.real_vec();
    assert!(re_sp[80].abs() < 1e-9);
    assert!((re_sp[160] - 25.0).abs() < 0.25);

    // 4. 1D NUS FISTA + Virtual-Echo reconstruction (`35%` sampling of 64 complex points)
    let n_nus = 64usize;
    let dense: Vec<Complex64> = (0..n_nus)
        .map(|k| {
            let t = k as f64;
            let env = (-0.012 * t).exp();
            let a1 = -2.0 * PI * 0.18 * t;
            let a2 = 2.0 * PI * 0.31 * t;
            [
                env * (a1.cos() + 0.6 * a2.cos()),
                env * (a1.sin() + 0.6 * a2.sin()),
            ]
        })
        .collect();
    let mask: Vec<bool> = (0..n_nus)
        .map(|k| k < 4 || (k * 37 + 11) % 10 < 3)
        .collect();
    let mut sparse = vec![[0.0, 0.0]; n_nus];
    for k in 0..n_nus {
        if mask[k] {
            sparse[k] = dense[k];
        }
    }
    let cfg = NusConfig {
        method: NusMethod::Fista,
        max_iter: 80,
        stop_rel_thresh: 0.004,
        virtual_echo_phase: Some((0.0, 0.0)),
        ..NusConfig::default()
    };
    let rec = reconstruct_nus_1d(&sparse, &mask, &cfg);
    let num_err: f64 = dense
        .iter()
        .zip(&rec)
        .map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
        .sum();
    let den_e: f64 = dense.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum();
    assert!(
        (num_err / den_e).sqrt() < 0.15,
        "FISTA + Virtual-Echo relative L2 error too large: {}",
        (num_err / den_e).sqrt()
    );
}

#[test]
fn ft1_1d_and_ft3_plane_series_round_trip_losslessly() {
    let dir = temp_test_dir("ft1_ft3_io");

    // 1. Real and Complex 1D `.ft1`
    let s1_cplx = Spectrum1D {
        name: "amide.ft1".into(),
        axis_x: AxisPpm::from_bounds("1H", 64, 11.5, 5.5, 600.13),
        data: (0..64).map(|k| (k as f32) * 0.5 - 4.0).collect(),
        imag: Some((0..64).map(|k| -(k as f32) * 0.25).collect()),
    };
    let p_ft1 = dir.join("amide.ft1");
    write_ft1_1d(&p_ft1, &s1_cplx).unwrap();
    let r1 = read_ft1_1d(&p_ft1).unwrap();
    assert_eq!(r1.nx(), 64);
    assert_eq!(r1.axis_x.label, "1H");
    assert!((r1.axis_x.first_ppm - s1_cplx.axis_x.first_ppm).abs() < 1e-4);
    assert_eq!(r1.data, s1_cplx.data);
    assert_eq!(r1.imag, s1_cplx.imag);

    // 2. Multi-file 3D Z-plane series (`test%03d.ft3`)
    let s3 = Spectrum3D {
        name: "cube".into(),
        exp: "HNCACB".into(),
        axis_x: AxisPpm::from_bounds("1H", 16, 10.5, 6.0, 500.13),
        axis_y: AxisPpm::from_bounds("15N", 8, 132.0, 104.0, 50.68),
        axis_z: AxisPpm::from_bounds("13C", 5, 72.0, 14.0, 125.76),
        data: (0..(16 * 8 * 5))
            .map(|i| (i as f32) * 1.25 - 50.0)
            .collect(),
    };
    let tmpl = dir.join("planes").join("test%03d.ft3");
    assert_eq!(
        format_plane_template(&tmpl, 3).unwrap(),
        dir.join("planes").join("test003.ft3")
    );
    let written = write_ft3_plane_series(&tmpl, &s3).unwrap();
    assert_eq!(written.len(), 5);
    assert!(dir.join("planes").join("test001.ft3").is_file());
    assert!(dir.join("planes").join("test005.ft3").is_file());

    let r3 = read_ft3_plane_series(&tmpl, "HNCACB").unwrap();
    assert_eq!((r3.nx(), r3.ny(), r3.nz()), (16, 8, 5));
    assert_eq!(r3.axis_x.label, "1H");
    assert_eq!(r3.axis_y.label, "15N");
    assert_eq!(r3.axis_z.label, "13C");
    assert!((r3.axis_z.first_ppm - s3.axis_z.first_ppm).abs() < 1e-3);
    assert_eq!(r3.data, s3.data);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn synthetic_1d_and_indirect_half_dwell_autophasing() {
    // 1. Synthetic 1D amide spectrum with non-zero `(p0, p1)` = `(-118.0, 36.0)` to recover
    let n_td = 512usize;
    let n_fft = 1024usize;
    let sw_hz = 4000.0;
    let axis = AxisPpm {
        label: "1H".into(),
        size: n_fft,
        first_ppm: 11.5,
        step_ppm: -(sw_hz / 500.0) / (n_fft as f64),
        obs_mhz: 500.0,
    };
    let freqs = [-0.26, -0.19, -0.11, -0.04, 0.03]; // inside amide ROI (~6.3..9.8 ppm)
    let mut td = vec![[0.0, 0.0]; n_td];
    for (k, c) in td.iter_mut().enumerate() {
        let kf = k as f64;
        let env = (-kf / 140.0).exp();
        for (idx, &f) in freqs.iter().enumerate() {
            let amp = 1.0 + 0.25 * (idx as f64);
            let ang = 2.0 * PI * f * kf;
            c[0] += amp * env * ang.cos();
            c[1] += amp * env * ang.sin();
        }
    }
    td[0][0] *= 0.5;
    td[0][1] *= 0.5;
    let mut spec = td;
    spec.resize(n_fft, [0.0, 0.0]);
    fft_arbitrary_in_place(&mut spec, false);
    fftshift_in_place(&mut spec);

    // Corrupt by `(p0_err, p1_err) = (+118.0, -36.0)` so true correction is `(-118.0, +36.0)`!
    for (k, c) in spec.iter_mut().enumerate() {
        let phi = (118.0 + (-36.0) * (k as f64 / n_fft as f64)).to_radians();
        let (sp, cp) = phi.sin_cos();
        *c = [c[0] * cp - c[1] * sp, c[0] * sp + c[1] * cp];
    }

    let sol_1d = auto_phase_1d(&spec, &axis, &AutoPhaseConfig::default());
    // Evaluate phase accuracy at the amide centroid (`k / N ~ 0.38`)
    let true_mid = -118.0 + 36.0 * 0.38;
    let rec_mid = sol_1d.p0_deg + sol_1d.p1_deg * 0.38;
    assert!(
        phase_diff_deg(rec_mid, true_mid).abs() < 3.5 && (sol_1d.p1_deg - 36.0).abs() < 10.0,
        "1D autophase mismatch: got {sol_1d:?}"
    );

    // Real-only Hilbert-reconstructed 1D autophasing + direct N-D multi-trace autophasing!
    let real_only: Vec<f64> = spec.iter().map(|c| c[0]).collect();
    let sol_real = auto_phase_1d_real(&real_only, &axis, &AutoPhaseConfig::default());
    let rec_real_mid = sol_real.p0_deg + sol_real.p1_deg * 0.38;
    assert!(
        phase_diff_deg(rec_real_mid, true_mid).abs() < 4.0,
        "Hilbert 1D autophase mismatch: got {sol_real:?}"
    );

    let mut ht_trace: Vec<Complex64> = real_only.iter().map(|&r| [r, 0.0]).collect();
    hilbert_transform_in_place(&mut ht_trace);
    let neg_trace: Vec<Complex64> = ht_trace
        .iter()
        .map(|c| [-0.7 * c[0], -0.7 * c[1]])
        .collect();
    let cfg_biph = AutoPhaseConfig {
        polarity: PolarityMode::AllPositive,
        ..AutoPhaseConfig::default()
    };
    let sol_nd = auto_phase_direct_nd(&[ht_trace, neg_trace], &axis, &cfg_biph);
    let rec_nd_mid = sol_nd.p0_deg + sol_nd.p1_deg * 0.38;
    assert!(
        phase_diff_deg(rec_nd_mid, true_mid).abs() < 4.0,
        "Direct N-D autophase mismatch: got {sol_nd:?}"
    );

    // 2. Indirect half-dwell `(-90, +180)` blind recovery across 16 sparse 15N columns
    let n_ind = 128usize;
    let axis_n = AxisPpm::from_bounds("15N", n_ind, 134.0, 102.0, 60.8);
    let mut cols: Vec<Vec<Complex64>> = Vec::new();
    for m in 0..16 {
        let f_norm = -0.42 + 0.055 * (m as f64); // spans full indirect SW
        let mut c_td = vec![[0.0, 0.0]; 48];
        for (k, slot) in c_td.iter_mut().enumerate() {
            let kf = k as f64;
            let w = (PI * (0.5 + 0.5 * kf / 47.0)).sin().powi(2);
            let ang = 2.0 * PI * f_norm * kf;
            *slot = [w * ang.cos(), w * ang.sin()];
        }
        c_td[0][0] *= 0.5;
        c_td[0][1] *= 0.5;
        c_td.resize(n_ind, [0.0, 0.0]);
        fft_arbitrary_in_place(&mut c_td, false);
        fftshift_in_place(&mut c_td);
        // Apply inverse of `(-90, +180)` = `(+90, -180)` AND delete imaginary channel to test Hilbert!
        for (k, v) in c_td.iter_mut().enumerate() {
            let phi = (90.0 - 180.0 * (k as f64 / n_ind as f64)).to_radians();
            let (sp, cp) = phi.sin_cos();
            *v = [v[0] * cp - v[1] * sp, 0.0];
        }
        cols.push(c_td);
    }
    let sol_hd = auto_phase_indirect_nd(&cols, &axis_n, &AutoPhaseConfig::default());
    assert!(
        phase_diff_deg(sol_hd.p0_deg, -90.0).abs() < 5.0 && (sol_hd.p1_deg - 180.0).abs() < 8.0,
        "Half-dwell indirect autophase failed: got {sol_hd:?}, expected (-90, +180)"
    );
}

#[test]
fn blind_autophase_on_real_ubiquitin_hsqc_and_biphasic_hncacb_ser() {
    let Some(data_dir) = resolve_local_real_data_dir() else {
        eprintln!(
            "Skipping real Ubiquitin blind autophase test (local data/ubiquitin not present)"
        );
        return;
    };

    let exp2_dir = data_dir.join("ubiquitin/Ubiquitin_090309/2");
    let exp3_dir = data_dir.join("ubiquitin/Ubiquitin_090309/3");
    if !exp2_dir.join("ser").is_file() || !exp3_dir.join("ser").is_file() {
        eprintln!("Skipping real Ubiquitin blind autophase test (raw ser files absent)");
        return;
    }

    // =========================================================================
    // Part A: Exp 2 (`15N-HSQC`) — Blind 2D Autophasing vs `procs` (`PHC0 = -151.65`) & `2rr`
    // =========================================================================
    let mut fid2 = read_fid_2d(&exp2_dir).expect("read Ubiquitin exp 2 ser");
    let stored_p0_x2 = fid2.axis_x.stored_proc.p0_deg.unwrap_or(-151.65);
    let stored_p1_x2 = fid2.axis_x.stored_proc.p1_deg.unwrap_or(0.0);
    let stored_p0_y2 = fid2.axis_y.stored_proc.p0_deg.unwrap_or(0.0);
    let stored_p1_y2 = fid2.axis_y.stored_proc.p1_deg.unwrap_or(0.0);

    // Explicitly erase all stored phases before running blind autophaser!
    fid2.axis_x.stored_proc.p0_deg = None;
    fid2.axis_x.stored_proc.p1_deg = None;
    fid2.axis_y.stored_proc.p0_deg = None;
    fid2.axis_y.stored_proc.p1_deg = None;

    let cfg_hsqc = AutoPhaseConfig::default();
    let (sol_x2, sol_y2, blind_hsqc) =
        blind_autophase_fid_2d(&fid2, &cfg_hsqc).expect("blind autophase exp 2 HSQC");

    // Effective phase at the amide centroid (`~8.0 ppm` -> `frac ~ 0.30`)
    let f_amide2 = blind_hsqc.axis_x.ppm_to_index(8.0) / (blind_hsqc.nx() as f64);
    let eff_rec_x2 = sol_x2.p0_deg + sol_x2.p1_deg * f_amide2;
    let eff_ref_x2 = stored_p0_x2 + stored_p1_x2 * f_amide2;
    let err_eff_x2 = phase_diff_deg(eff_rec_x2, eff_ref_x2).abs();

    let eff_rec_y2 = sol_y2.p0_deg + 0.5 * sol_y2.p1_deg;
    let eff_ref_y2 = stored_p0_y2 + 0.5 * stored_p1_y2;
    let err_eff_y2 = phase_diff_deg(eff_rec_y2, eff_ref_y2).abs();

    let pdata2 = read_bruker_pdata_2d(exp2_dir.join("pdata/1")).expect("read exp 2 2rr");
    let pk_cfg = PickerConfig::default();
    let ts_peaks2 = pick_hsqc_2d(&pdata2, &pk_cfg);
    let blind_peaks2 = pick_hsqc_2d(&blind_hsqc, &pk_cfg);

    let mut matched2 = 0usize;
    for tp in &ts_peaks2 {
        if blind_peaks2
            .iter()
            .any(|bp| (bp.h - tp.h).abs() <= 0.03 && (bp.n - tp.n).abs() <= 0.25)
        {
            matched2 += 1;
        }
    }
    let recall2 = (matched2 as f64) / (ts_peaks2.len().max(1) as f64);

    eprintln!(
        "[Ubiquitin Exp 2 HSQC Blind Autophase] direct=({:.2} deg, {:.2} deg) vs procs=({:.2}, {:.2}) [eff_amide_err={:.2} deg]; \
         indirect=({:.2} deg, {:.2} deg) vs proc2s=({:.2}, {:.2}) [eff_err={:.2} deg]; \
         matched peaks={}/{} ({:.1}%)",
        sol_x2.p0_deg,
        sol_x2.p1_deg,
        stored_p0_x2,
        stored_p1_x2,
        err_eff_x2,
        sol_y2.p0_deg,
        sol_y2.p1_deg,
        stored_p0_y2,
        stored_p1_y2,
        err_eff_y2,
        matched2,
        ts_peaks2.len(),
        100.0 * recall2
    );

    assert!(
        err_eff_x2 < 12.0,
        "Exp 2 direct 1H blind phase deviated from procs (-151.65 deg): sol_x2={sol_x2:?}, eff_err={err_eff_x2:.2}"
    );
    assert!(
        err_eff_y2 < 10.0,
        "Exp 2 indirect 15N blind phase deviated from proc2s (0 deg): sol_y2={sol_y2:?}, eff_err={err_eff_y2:.2}"
    );
    assert!(
        recall2 >= 0.85,
        "Exp 2 blind-autophased HSQC peak recall too low: {matched2}/{} ({:.1}%)",
        ts_peaks2.len(),
        100.0 * recall2
    );

    // =========================================================================
    // Part B: Exp 3 (`3D HNCACB`) — Blind 3D `Biphasic` Autophasing vs `procs` (`0 / -180`) & `3rrr`
    // =========================================================================
    let mut fid3 = read_fid_3d(&exp3_dir).expect("read Ubiquitin exp 3 HNCACB ser");
    fid3.axis_x.stored_proc.p0_deg = None;
    fid3.axis_x.stored_proc.p1_deg = None;
    fid3.axis_y.stored_proc.p0_deg = None;
    fid3.axis_y.stored_proc.p1_deg = None;
    fid3.axis_z.stored_proc.p0_deg = None;
    fid3.axis_z.stored_proc.p1_deg = None;

    let cfg_biphasic = AutoPhaseConfig::biphasic();
    let (sol_x3, sol_y3, sol_z3, blind_hncacb) =
        blind_autophase_fid_3d(&fid3, "HNCACB", &cfg_biphasic)
            .expect("blind autophase exp 3 Biphasic HNCACB");

    let f_amide3 = blind_hncacb.axis_x.ppm_to_index(8.0) / (blind_hncacb.nx() as f64);
    let eff_x3 = sol_x3.p0_deg + sol_x3.p1_deg * f_amide3;
    let eff_y3 = sol_y3.p0_deg + 0.5 * sol_y3.p1_deg;
    let eff_z3 = sol_z3.p0_deg + 0.5 * sol_z3.p1_deg;

    // Modulo-180 absorption error on each axis + combined 3D global sign (`x + y + z == -180 deg mod 360`!)
    let mod180_x3 = phase_diff_mod180_deg(eff_x3, 0.0);
    let mod180_y3 = phase_diff_mod180_deg(eff_y3, -180.0);
    let mod180_z3 = phase_diff_mod180_deg(eff_z3, 0.0);
    let total_sum_err = phase_diff_deg(eff_x3 + eff_y3 + eff_z3, -180.0).abs();

    // Verify `CA` band (`48..66 ppm`) is strongly POSITIVE and aliphatic `CB` (`14..38 ppm`) is NEGATIVE!
    let mut ca_pos_sum = 0.0f64;
    let mut cb_neg_sum = 0.0f64;
    let (nx3, ny3, nz3) = (blind_hncacb.nx(), blind_hncacb.ny(), blind_hncacb.nz());
    let max_cube = blind_hncacb
        .data
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    let strong_cut = 0.12 * max_cube;
    for iz in 0..nz3 {
        let c_ppm = blind_hncacb.axis_z.index_to_ppm(iz as f64);
        let plane = &blind_hncacb.data[iz * ny3 * nx3..(iz + 1) * ny3 * nx3];
        if (48.0..=66.0).contains(&c_ppm) {
            for &v in plane {
                if v.abs() >= strong_cut {
                    ca_pos_sum += v as f64;
                }
            }
        } else if (14.0..=38.0).contains(&c_ppm) {
            for &v in plane {
                if v.abs() >= strong_cut {
                    cb_neg_sum += v as f64;
                }
            }
        }
    }

    let pdata3 = read_bruker_pdata_3d(exp3_dir.join("pdata/1"), "HNCACB").expect("read exp 3 3rrr");
    let ts_peaks3 = pick_triple_3d(&pdata3, &ts_peaks2, &pk_cfg);
    let blind_peaks3 = pick_triple_3d(&blind_hncacb, &blind_peaks2, &pk_cfg);
    let mut matched3 = 0usize;
    let mut sign_agree3 = 0usize;
    for tp in &ts_peaks3 {
        if let Some(bp) = blind_peaks3.iter().find(|bp| {
            (bp.h - tp.h).abs() <= 0.03
                && (bp.n - tp.n).abs() <= 0.35
                && (bp.c.unwrap_or(0.0) - tp.c.unwrap_or(999.0)).abs() <= 0.45
        }) {
            matched3 += 1;
            if bp.height.unwrap_or(1.0).signum() == tp.height.unwrap_or(1.0).signum() {
                sign_agree3 += 1;
            }
        }
    }
    let recall3 = (matched3 as f64) / (ts_peaks3.len().max(1) as f64);
    let sign_acc3 = (sign_agree3 as f64) / (matched3.max(1) as f64);

    eprintln!(
        "[Ubiquitin Exp 3 HNCACB Biphasic Blind Autophase] x=({:.2}, {:.2}), y=({:.2}, {:.2}), z=({:.2}, {:.2}) \
         [mod180 errs: x={:.2}, y={:.2}, z={:.2} deg; total_3d_phase_err={:.2} deg]; \
         CA_sum={:.3e} (>0), CB_sum={:.3e} (<0); matched_3d={}/{} ({:.1}%), biphasic_sign_agreement={}/{} ({:.1}%)",
        sol_x3.p0_deg,
        sol_x3.p1_deg,
        sol_y3.p0_deg,
        sol_y3.p1_deg,
        sol_z3.p0_deg,
        sol_z3.p1_deg,
        mod180_x3,
        mod180_y3,
        mod180_z3,
        total_sum_err,
        ca_pos_sum,
        cb_neg_sum,
        matched3,
        ts_peaks3.len(),
        100.0 * recall3,
        sign_agree3,
        matched3,
        100.0 * sign_acc3
    );

    assert!(
        mod180_x3 < 12.0 && mod180_y3 < 12.0 && mod180_z3 < 12.0 && total_sum_err < 18.0,
        "Exp 3 HNCACB blind phases deviated from procs (0 / -180 / 0): x={sol_x3:?}, y={sol_y3:?}, z={sol_z3:?}"
    );
    assert!(
        ca_pos_sum > 0.0 && cb_neg_sum < 0.0,
        "Exp 3 Biphasic polarity lock failed: CA_sum={ca_pos_sum}, CB_sum={cb_neg_sum}"
    );
    assert!(
        recall3 >= 0.68 && sign_acc3 >= 0.97,
        "Exp 3 HNCACB 3D peak/sign match too low: recall={:.1}%, sign_acc={:.1}%",
        100.0 * recall3,
        100.0 * sign_acc3
    );
}

#[test]
fn nus_1d_recovers_100_to_1_adjacent_weak_peak_and_rejects_unrecombined_trace() {
    let n = 128usize;
    // Strong peak (amp = 100.0) at normalized offset f1 = +0.125 (bin 64 - 16 = 48 -> 122.0 ppm)
    // Weak adjacent peak (amp = 1.0, 100:1 dynamic range!) just 8 bins away at f2 = +0.1875 (bin 40 -> 124.0 ppm)
    let dense: Vec<Complex64> = (0..n)
        .map(|k| {
            let t = k as f64;
            let env = (-0.008 * t).exp();
            let p_strong = 2.0 * PI * 0.125 * t;
            let p_weak = 2.0 * PI * 0.1875 * t;
            // Deterministic micro-noise (1e-3) so MAD is non-zero and realistic
            let noise = (((k * 97 + 13) % 53) as f64 - 26.0) * 4e-4;
            [
                env * (100.0 * p_strong.cos() + 1.0 * p_weak.cos()) + noise,
                -env * (100.0 * p_strong.sin() + 1.0 * p_weak.sin()) - noise,
            ]
        })
        .collect();

    // ~45% aperiodic pseudo-random NUS schedule (first 8 increments dense + Knuth multiplicative hash)
    let mask: Vec<bool> = (0..n)
        .map(|k| k < 8 || ((k.wrapping_mul(2_654_435_761) >> 16) % 100) < 42)
        .collect();
    let mut sparse = vec![[0.0, 0.0]; n];
    for k in 0..n {
        if mask[k] {
            sparse[k] = dense[k];
        }
    }

    let pipe = AxisPipeline::new(vec![
        StageOp::SineWindow {
            off: 0.5,
            end: 1.0,
            pow: 2.0,
            first_scale: 0.5,
        },
        StageOp::FourierTransform {
            inverse: false,
            negate_imag: true,
            sign_alt: false,
            real_only: false,
        },
        StageOp::DeleteImaginary,
    ]);
    let mut tr_dense = Trace1D::new_time_domain("15N", 1945.6, 60.8, 134.0, dense.clone());
    pipe.apply(&mut tr_dense).unwrap();
    let spec_dense = tr_dense.real_vec();
    // After `negate_imag` + forward FFT + `fftshift` (`rotate_right(64)`), `+0.125` (`m = 16`)
    // lands at `64 + 16 = 80` and `+0.1875` (`m = 24`) lands at `64 + 24 = 88`.
    let dense_weak_prom = spec_dense[88] - 0.5 * (spec_dense[84] + spec_dense[92]);
    assert!(
        dense_weak_prom > 30.0,
        "unexpected uniform weak prominence: {dense_weak_prom}"
    );

    for method in [NusMethod::Fista, NusMethod::Ist] {
        let cfg = NusConfig {
            method,
            max_iter: if method == NusMethod::Ist { 300 } else { 180 },
            stop_rel_thresh: 0.001,
            ..NusConfig::default()
        };
        let rec = reconstruct_nus_1d(&sparse, &mask, &cfg);
        let mut tr = Trace1D::new_time_domain("15N", 1945.6, 60.8, 134.0, rec);
        pipe.apply(&mut tr).unwrap();
        let spec = tr.real_vec();
        let strong_val = spec[80];
        let weak_prom = spec[88] - 0.5 * (spec[84] + spec[92]);
        let mut art: Vec<f64> = (12..56).map(|i| (spec[i] - spec_dense[i]).abs()).collect();
        art.sort_by(|a, b| a.total_cmp(b));
        let art_med = art[art.len() / 2].max(1e-9);
        assert!(
            weak_prom > 0.65 * dense_weak_prom && weak_prom > 12.0 * art_med,
            "{method:?} failed to recover 100:1 adjacent weak peak: strong={strong_val:.3}, weak_prom={weak_prom:.3} (dense={dense_weak_prom:.3}), art_med={art_med:.4}"
        );
    }

    // Verify `StageOp::NusReconstruct` rejects any trace not yet converted to States (`states_recombined == false`)
    let mut unrecombined = Trace1D::new_time_domain("15N", 1945.6, 60.8, 134.0, sparse);
    unrecombined.states_recombined = false;
    let err = AxisPipeline::new(vec![StageOp::NusReconstruct {
        mask,
        config: NusConfig::default(),
    }])
    .apply(&mut unrecombined)
    .expect_err("unrecombined Echo-Antiecho / States-TPPI trace must fail before IST/FISTA");
    assert!(
        err.contains("States"),
        "expected States precondition error, got: {err}"
    );
}

#[test]
fn biphasic_polarity_lock_resists_ser_thr_cb_and_intense_artifact_spike() {
    // 256-pt 13C indirect axis spanning 80.0 .. 10.0 ppm
    let n = 256usize;
    let axis = AxisPpm::from_bounds("13C", n, 80.0, 10.0, 150.9);
    let lorentz = |k: usize, ppm0: f64, amp: f64, hwhm_bins: f64| -> Complex64 {
        let k0 = axis.ppm_to_index(ppm0);
        let x = ((k as f64) - k0) / hwhm_bins;
        let denom = 1.0 + x * x;
        // Rotated by true phase +180 deg (inverted input: CA currently negative, CB positive!)
        [-amp / denom, -amp * x / denom]
    };

    // Build an inverted (`p0_true = 180 deg`) Ser/Thr-rich HNCACB column set where:
    // - 5 true CA(+) peaks sit in 52..59 ppm (amp = 10.0 each)
    // - 7 strong Ser/Thr CB(-) peaks sit in 61.5..65.5 ppm (amp = -14.0 each; previously inside 48..66!)
    // - 4 aliphatic CB(-) peaks sit in 24..35 ppm (amp = -9.0 each)
    // - 1 huge opposite-sign artifact spike (amp = -800.0) sits at 55.0 ppm!
    let mut cols: Vec<Vec<Complex64>> = Vec::new();
    let ca_ppms = [52.8, 54.4, 56.2, 57.8, 59.1];
    let ser_thr_cb_ppms = [61.6, 62.3, 63.1, 63.8, 64.4, 65.0, 65.6];
    let aliph_cb_ppms = [19.2, 27.8, 31.4, 35.6];

    for r in 0..7 {
        let mut col = vec![[0.0, 0.0]; n];
        for (k, slot) in col.iter_mut().enumerate() {
            let jitter = (((k * 31 + r * 17) % 23) as f64 - 11.0) * 0.02;
            slot[0] += jitter;
            if r < ca_ppms.len() {
                let c = lorentz(k, ca_ppms[r], 10.0, 1.4);
                slot[0] += c[0];
                slot[1] += c[1];
            }
            let c_st = lorentz(k, ser_thr_cb_ppms[r], -14.0, 1.4);
            slot[0] += c_st[0];
            slot[1] += c_st[1];
            if r < aliph_cb_ppms.len() {
                let c_cb = lorentz(k, aliph_cb_ppms[r], -9.0, 1.4);
                slot[0] += c_cb[0];
                slot[1] += c_cb[1];
            }
            if r == 0 {
                // Single 80x intense inverted artifact spike inside 48..60 ppm!
                let spike = lorentz(k, 55.0, -800.0, 1.1);
                slot[0] += spike[0];
                slot[1] += spike[1];
            }
        }
        cols.push(col);
    }

    let cfg = AutoPhaseConfig::biphasic();
    assert_eq!(cfg.biphasic_pos_band_ppm, (48.0, 60.0));
    let sol = auto_phase_indirect_nd(&cols, &axis, &cfg);

    // Applying `sol.p0_deg` to `cols[1]` (`CA` at 54.4 ppm, `Ser CB` at 62.3 ppm, `CB` at 27.8 ppm)
    // must make `CA` strictly positive and both `Ser CB` and aliphatic `CB` strictly negative!
    let eval_real_at = |col: &[Complex64], ppm: f64| -> f64 {
        let k = axis.ppm_to_index(ppm).round() as usize;
        let phi = (sol.p0_deg + sol.p1_deg * ((k as f64) / (n as f64))).to_radians();
        let (sp, cp) = phi.sin_cos();
        col[k][0] * cp - col[k][1] * sp
    };
    let ca_val = eval_real_at(&cols[1], 54.4);
    let ser_cb_val = eval_real_at(&cols[1], 62.3);
    let cb_val = eval_real_at(&cols[1], 27.8);
    assert!(
        ca_val > 5.0 && ser_cb_val < -5.0 && cb_val < -4.0,
        "Biphasic lock failed under Ser/Thr CB + intense spike: sol={sol:?}, ca={ca_val:.2}, ser_cb={ser_cb_val:.2}, cb={cb_val:.2}"
    );
}
