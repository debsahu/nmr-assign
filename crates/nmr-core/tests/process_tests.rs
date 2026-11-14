//! Unit, integration, and real-spectrometer validation tests for `nmr_core::process`
//! and 3D spectrum writers (`write_nmrpipe_3d`, `write_bruker_pdata_3d`).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use nmr_core::evaluate::{compare_shifts, score, to_reference};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::fid::{
    Complex64, Fid1D, Fid2D, Fid3D, FidAxisMeta, FidVendor, QuadMode, StoredDimProc,
};
use nmr_core::nmrstar::Reference;
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::process::{
    auto_phase_chen2002, fft_in_place, process_fid_1d_mut, process_fid_2d, process_fid_2d_mut,
    process_fid_3d, process_fid_dir_2d, process_fid_dir_3d, NusIstConfig, PhaseSource,
    ProcessingRecipe,
};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_3d, write_bruker_pdata_3d,
    write_nmrpipe_3d, AxisPpm,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

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
    if common_data.join("ubiquitin").is_dir() {
        Some(common_data)
    } else {
        None
    }
}

fn make_axis_meta(
    label: &str,
    complex_points: usize,
    sw_hz: f64,
    obs_mhz: f64,
    carrier_ppm: f64,
    conjugate_for_fft: bool,
    group_delay: f64,
) -> FidAxisMeta {
    let sw_ppm = sw_hz / obs_mhz;
    FidAxisMeta {
        label: label.to_string(),
        td_raw: complex_points * 2,
        complex_points,
        sw_hz,
        obs_mhz,
        carrier_mhz: obs_mhz * (1.0 + carrier_ppm * 1e-6),
        first_ppm: carrier_ppm + 0.5 * sw_ppm,
        quad_mode: QuadMode::Dqd,
        states_recombined: true,
        group_delay,
        conjugate_for_fft,
        stored_proc: StoredDimProc::default(),
    }
}

/// Parabolic 3-point sub-bin peak interpolation in ppm along an `AxisPpm`.
fn subbin_peak_ppm(axis: &AxisPpm, spec: &[f32]) -> (f64, usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in spec.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    if best_i > 0 && best_i + 1 < spec.len() {
        let yl = spec[best_i - 1] as f64;
        let yc = spec[best_i] as f64;
        let yr = spec[best_i + 1] as f64;
        let denom = yl - 2.0 * yc + yr;
        if denom.abs() > 1e-12 {
            let delta = (0.5 * (yl - yr) / denom).clamp(-0.5, 0.5);
            return (axis.index_to_ppm(best_i as f64 + delta), best_i, best_v);
        }
    }
    (axis.index_to_ppm(best_i as f64), best_i, best_v)
}

// ============================================================================
// 1. Clean-Room FFT, GRPDLY, Sign Discrimination, Auto-Phase, LP, NUS, Writers
// ============================================================================

#[test]
fn fft_radix2_and_bluestein_exact_roundtrip() {
    for &n in &[64usize, 128, 96, 150, 243] {
        let orig: Vec<Complex64> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                [
                    (2.0 * PI * 3.0 * t).cos() + 0.3 * (2.0 * PI * 11.0 * t).sin(),
                    -0.5 * (2.0 * PI * 7.0 * t).cos(),
                ]
            })
            .collect();
        let e_time: f64 = orig.iter().map(|c| c[0] * c[0] + c[1] * c[1]).sum();

        let mut buf = orig.clone();
        fft_in_place(&mut buf, false);
        let e_freq: f64 = buf.iter().map(|c| c[0] * c[0] + c[1] * c[1]).sum::<f64>() / (n as f64);
        assert!(
            (e_time - e_freq).abs() < 1e-10,
            "Parseval mismatch for n={n}: {e_time} vs {e_freq}"
        );

        fft_in_place(&mut buf, true);
        for (a, b) in orig.iter().zip(buf.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-11 && (a[1] - b[1]).abs() < 1e-11);
        }
    }
}

#[test]
fn exact_1d_peak_ppm_and_subsample_grpdly_alignment() {
    let n_td = 1024usize;
    let sw_hz = 11160.714;
    let obs_mhz = 800.13;
    let carrier_ppm = 4.70;
    let grpdly = 67.98427;
    let target_ppm = 8.42;

    for &bruker_conv in &[true, false] {
        let mut axis = make_axis_meta("1H", n_td, sw_hz, obs_mhz, carrier_ppm, bruker_conv, grpdly);
        axis.stored_proc.p0_deg = Some(0.0);
        axis.stored_proc.p1_deg = Some(0.0);

        let f_cyc = (target_ppm - carrier_ppm) * obs_mhz / sw_hz;
        let sign = if bruker_conv { 1.0 } else { -1.0 };
        let mut data = vec![[0.0, 0.0]; n_td];
        for (n, dst) in data.iter_mut().enumerate() {
            let t = (n as f64) - grpdly;
            let env = if t >= 0.0 {
                (-t / 350.0).exp()
            } else {
                (-t.abs() / 6.0).exp() * 0.05
            };
            let phi = sign * 2.0 * PI * f_cyc * t;
            *dst = [env * phi.cos(), env * phi.sin()];
        }
        let fid = Fid1D {
            name: "synth_grpdly".to_string(),
            vendor: if bruker_conv {
                FidVendor::Bruker
            } else {
                FidVendor::Varian
            },
            pulse_program: "zg".to_string(),
            axis_x: axis,
            data,
        };
        let mut recipe = ProcessingRecipe::from_fid_1d(&fid);
        recipe.dims[0].si = 4096;
        let (ax_out, spec) = process_fid_1d_mut(&fid, &mut recipe).expect("process 1D");
        let (meas_ppm, max_idx, max_val) = subbin_peak_ppm(&ax_out, &spec);
        assert!(
            (meas_ppm - target_ppm).abs() < 8e-4,
            "conv={bruker_conv}: expected {target_ppm:.4} ppm, got {meas_ppm:.4} ppm"
        );
        let min_wing = spec[max_idx.saturating_sub(25)..=(max_idx + 25).min(spec.len() - 1)]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_wing > -0.04 * max_val,
            "conv={bruker_conv}: dispersive negative lobe {min_wing} vs max {max_val}"
        );
    }
}

#[test]
fn chen2002_autophase_recovers_p0_p1_and_respects_priority() {
    let n_td = 1024usize;
    let sw_hz = 11200.0;
    let obs_mhz = 800.0;
    let carrier_ppm = 4.70;
    let peaks_ppm = [6.95, 7.62, 8.38, 9.41];
    let true_p0_deg: f64 = -134.0;
    let true_p1_deg: f64 = 28.0;

    let mut spec_clean = vec![[0.0, 0.0]; 2048];
    let axis = make_axis_meta("1H", n_td, sw_hz, obs_mhz, carrier_ppm, true, 0.0);
    let mut td_zero = vec![[0.0, 0.0]; n_td];
    for &ppm in &peaks_ppm {
        let f_cyc = (ppm - carrier_ppm) * obs_mhz / sw_hz;
        for (n, dst) in td_zero.iter_mut().enumerate() {
            let w_ssb = (PI * (0.5 + 0.5 * (n as f64) / (n_td as f64)))
                .sin()
                .powi(2);
            let env = (-(n as f64) / 300.0).exp() * w_ssb;
            let phi = 2.0 * PI * f_cyc * (n as f64);
            dst[0] += env * phi.cos();
            dst[1] += env * phi.sin();
        }
    }
    td_zero[0][0] *= 0.5;
    td_zero[0][1] *= 0.5;
    spec_clean[..n_td].copy_from_slice(&td_zero);
    fft_in_place(&mut spec_clean, false);
    let n_fft = spec_clean.len();
    let half = n_fft / 2;
    let mut ordered = vec![[0.0, 0.0]; n_fft];
    for j in 0..n_fft {
        ordered[j] = spec_clean[(half + n_fft - j) % n_fft];
    }
    let mut distorted = vec![[0.0, 0.0]; n_fft];
    for j in 0..n_fft {
        // Distort by exp(+i (p0 + p1 * j / N)) so `apply_phase_and_discard_imag` (which multiplies
        // by exp(-i (p0 + p1 * j / N))) recovers `ordered[j]` at `(true_p0_deg, true_p1_deg)`.
        let phi = (true_p0_deg + true_p1_deg * (j as f64) / (n_fft as f64)).to_radians();
        let (sp, cp) = phi.sin_cos();
        let c = ordered[j];
        distorted[j] = [c[0] * cp - c[1] * sp, c[0] * sp + c[1] * cp];
    }
    let (est_p0, est_p1) = auto_phase_chen2002(&distorted);
    let true_mid = true_p0_deg + 0.5 * true_p1_deg;
    let est_mid = est_p0 + 0.5 * est_p1;
    let diff_mid = ((est_mid - true_mid + 540.0) % 360.0) - 180.0;
    assert!(
        diff_mid.abs() < 3.0 && (est_p1 - true_p1_deg).abs() < 6.0,
        "Chen2002 auto-phase mismatch: true=({true_p0_deg}, {true_p1_deg}), est=({est_p0:.2}, {est_p1:.2})"
    );

    let fid = Fid1D {
        name: "phase_prio".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "zg".to_string(),
        axis_x: axis.clone(),
        data: td_zero.clone(),
    };
    let r_auto = ProcessingRecipe::from_fid_1d(&fid);
    assert_eq!(r_auto.dims[0].phase_source, PhaseSource::AutoChen2002);

    let mut axis_stored = axis;
    axis_stored.stored_proc.p0_deg = Some(-42.5);
    axis_stored.stored_proc.p1_deg = Some(10.0);
    let fid_stored = Fid1D {
        name: "phase_stored".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "zg".to_string(),
        axis_x: axis_stored,
        data: td_zero,
    };
    let mut r_stored = ProcessingRecipe::from_fid_1d(&fid_stored);
    assert_eq!(r_stored.dims[0].phase_source, PhaseSource::Stored);
    assert_eq!(r_stored.dims[0].p0_deg, Some(-42.5));
    r_stored.dims[0].set_explicit_phase(15.0, -5.0);
    assert_eq!(r_stored.dims[0].phase_source, PhaseSource::Explicit);
    assert_eq!(r_stored.dims[0].p0_deg, Some(15.0));
}

#[test]
fn states_tppi_reverse_and_milp_2d_tests() {
    let nx = 256usize;
    let ny = 48usize;
    let mut ax_x = make_axis_meta("1H", nx, 9600.0, 800.0, 4.70, true, 0.0);
    ax_x.stored_proc.p0_deg = Some(0.0);
    ax_x.stored_proc.p1_deg = Some(0.0);
    let mut ax_y = make_axis_meta("15N", ny, 2400.0, 81.0, 118.0, true, 0.0);
    ax_y.quad_mode = QuadMode::StatesTppi;
    ax_y.stored_proc.p0_deg = Some(0.0);
    ax_y.stored_proc.p1_deg = Some(0.0);
    ax_y.stored_proc.td_eff = Some(2 * ny);

    let true_h = 8.25_f64;
    let true_n = 123.4_f64;
    let fx = (true_h - 4.70) * 800.0 / 9600.0;
    let fy = (true_n - 118.0) * 81.0 / 2400.0;

    let mut data = vec![[0.0, 0.0]; ny * 2 * nx];
    for k1 in 0..ny {
        let env_y = (-(k1 as f64) / 36.0).exp();
        let py = 2.0 * PI * fy * (k1 as f64);
        let cy = env_y * py.cos();
        let sy = env_y * py.sin();
        for k0 in 0..nx {
            let env_x = (-(k0 as f64) / 90.0).exp();
            let px = 2.0 * PI * fx * (k0 as f64);
            let zx = [env_x * px.cos(), env_x * px.sin()];
            data[(k1 * 2) * nx + k0] = [cy * zx[0], cy * zx[1]];
            data[(k1 * 2 + 1) * nx + k0] = [sy * zx[0], sy * zx[1]];
        }
    }

    let fid = Fid2D {
        name: "synth_2d".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqcetfpf3gpsi".to_string(),
        axis_x: ax_x.clone(),
        axis_y: ax_y.clone(),
        nus_mask: None,
        truncated_traces: None,
        data,
    };

    let mut rec = ProcessingRecipe::from_fid_2d(&fid);
    assert!(!rec.dims[1].linear_prediction);
    assert!(
        rec.dims[1].td_eff.is_none(),
        "td_eff >= complex_points must be clamped when LP is disabled"
    );

    let spec = process_fid_2d(&fid, &rec).expect("process 2D");
    let peaks = pick_hsqc_2d(&spec, &PickerConfig::default());
    assert_eq!(
        peaks.len(),
        1,
        "single quadrant peak expected without mirror ghost"
    );
    assert!((peaks[0].h - true_h).abs() < 0.01);
    assert!((peaks[0].n - true_n).abs() < 0.08);

    // Echo-Antiecho sign discrimination (`fid.rs` recombines Rance-Kay pairs into standard States
    // `(cos, sin)` orientation with zero 90-deg offset)
    let mut ax_y_ea = ax_y;
    ax_y_ea.quad_mode = QuadMode::EchoAntiecho;
    let fid_ea = Fid2D {
        name: "synth_2d_ea".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqcetfpf3gpsi".to_string(),
        axis_x: ax_x,
        axis_y: ax_y_ea,
        nus_mask: None,
        truncated_traces: None,
        data: fid.data.clone(),
    };
    let spec_ea =
        process_fid_2d(&fid_ea, &ProcessingRecipe::from_fid_2d(&fid_ea)).expect("process 2D EA");
    let peaks_ea = pick_hsqc_2d(&spec_ea, &PickerConfig::default());
    assert_eq!(
        peaks_ea.len(),
        1,
        "Echo-Antiecho single quadrant peak expected"
    );
    assert!((peaks_ea[0].h - true_h).abs() < 0.01);
    assert!((peaks_ea[0].n - true_n).abs() < 0.08);

    let mut rec_rev = rec.clone();
    rec_rev.dims[1].reverse = true;
    let spec_rev = process_fid_2d(&fid, &rec_rev).expect("process 2D reverse");
    let peaks_rev = pick_hsqc_2d(&spec_rev, &PickerConfig::default());
    assert_eq!(peaks_rev.len(), 1);
    let expected_rev_n = 2.0 * 118.0 - true_n;
    assert!(
        (peaks_rev[0].n - expected_rev_n).abs() < 0.30,
        "REVERSE=yes expected ~{expected_rev_n:.2} ppm, got {:.2} ppm",
        peaks_rev[0].n
    );

    rec.set_linear_prediction(true);
    let spec_lp = process_fid_2d(&fid, &rec).expect("process 2D with MILP");
    let peaks_lp = pick_hsqc_2d(&spec_lp, &PickerConfig::default());
    assert_eq!(peaks_lp.len(), 1);
    assert!((peaks_lp[0].n - true_n).abs() < 0.08);
}

#[test]
fn synthetic_2d_and_3d_nus_ist_recovery_and_3d_writers_roundtrip() {
    let nx = 256usize;
    let ny = 64usize;
    let mut ax_x = make_axis_meta("1H", nx, 9600.0, 800.0, 4.70, true, 0.0);
    ax_x.stored_proc.p0_deg = Some(0.0);
    ax_x.stored_proc.p1_deg = Some(0.0);
    let mut ax_y = make_axis_meta("15N", ny, 2400.0, 81.0, 118.0, true, 0.0);
    ax_y.stored_proc.p0_deg = Some(0.0);
    ax_y.stored_proc.p1_deg = Some(0.0);

    let synth_peaks_2d = [(7.45, 114.2, 1.0), (8.30, 121.8, 0.85), (9.05, 126.5, 0.70)];
    let mut full_2d_data = vec![[0.0, 0.0]; ny * 2 * nx];
    for &(ph, pn, amp) in &synth_peaks_2d {
        let fx = (ph - 4.70) * 800.0 / 9600.0;
        let fy = (pn - 118.0) * 81.0 / 2400.0;
        for k1 in 0..ny {
            let ey = (-(k1 as f64) / 48.0).exp();
            let py = 2.0 * PI * fy * (k1 as f64);
            let (sy, cy) = (ey * py.sin(), ey * py.cos());
            for k0 in 0..nx {
                let ex = amp * (-(k0 as f64) / 96.0).exp();
                let px = 2.0 * PI * fx * (k0 as f64);
                let zx = [ex * px.cos(), ex * px.sin()];
                full_2d_data[(k1 * 2) * nx + k0][0] += cy * zx[0];
                full_2d_data[(k1 * 2) * nx + k0][1] += cy * zx[1];
                full_2d_data[(k1 * 2 + 1) * nx + k0][0] += sy * zx[0];
                full_2d_data[(k1 * 2 + 1) * nx + k0][1] += sy * zx[1];
            }
        }
    }

    let fid2_uniform = Fid2D {
        name: "uniform_2d".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqc".to_string(),
        axis_x: ax_x.clone(),
        axis_y: ax_y.clone(),
        nus_mask: None,
        truncated_traces: None,
        data: full_2d_data.clone(),
    };
    let spec2_ref =
        process_fid_2d(&fid2_uniform, &ProcessingRecipe::from_fid_2d(&fid2_uniform)).unwrap();

    let mut mask_2d = vec![false; ny];
    mask_2d[0] = true;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut count = 1usize;
    while count < 26 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = ((state >> 33) as usize) % ny;
        if !mask_2d[idx] {
            mask_2d[idx] = true;
            count += 1;
        }
    }
    let mut nus_2d_data = full_2d_data;
    for k1 in 0..ny {
        if !mask_2d[k1] {
            nus_2d_data[(k1 * 2) * nx..(k1 * 2 + 2) * nx].fill([0.0, 0.0]);
        }
    }
    let fid2_nus = Fid2D {
        name: "nus_2d".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqc".to_string(),
        axis_x: ax_x.clone(),
        axis_y: ax_y,
        nus_mask: Some(mask_2d),
        truncated_traces: None,
        data: nus_2d_data,
    };
    let mut rec2_nus = ProcessingRecipe::from_fid_2d(&fid2_nus);
    assert!(rec2_nus.nus.is_some());
    let json = rec2_nus.to_json_pretty().expect("serialize recipe");
    let parsed_rec = ProcessingRecipe::from_json_str(&json).expect("deserialize recipe");
    assert_eq!(rec2_nus, parsed_rec);

    let spec2_nus = process_fid_2d_mut(&fid2_nus, &mut rec2_nus).expect("process 2D NUS");
    let r_2d = pearson_r_f32(&spec2_ref.data, &spec2_nus.data);
    assert!(r_2d > 0.985, "2D NUS IST correlation r={r_2d:.4} < 0.985");
    let picked_2d = pick_hsqc_2d(&spec2_nus, &PickerConfig::default());
    assert_eq!(picked_2d.len(), 3);

    // 2. 3D HNCO with 35% 2D hypercomplex NUS mask (n1 = 24, n2 = 24 -> 202 / 576 points)
    let ny3 = 24usize;
    let nz3 = 24usize;
    let mut ax_y3 = make_axis_meta("15N", ny3, 2400.0, 81.0, 118.0, true, 0.0);
    ax_y3.stored_proc.p0_deg = Some(0.0);
    ax_y3.stored_proc.p1_deg = Some(0.0);
    let mut ax_z3 = make_axis_meta("13C", nz3, 2400.0, 201.0, 176.0, true, 0.0);
    ax_z3.stored_proc.p0_deg = Some(0.0);
    ax_z3.stored_proc.p1_deg = Some(0.0);

    let synth_peaks_3d = [(7.85, 116.5, 174.2, 1.0), (8.65, 122.4, 177.9, 0.85)];
    let mut full_3d_data = vec![[0.0, 0.0]; nz3 * 2 * ny3 * 2 * nx];
    for &(ph, pn, pc, amp) in &synth_peaks_3d {
        let fx = (ph - 4.70) * 800.0 / 9600.0;
        let fy = (pn - 118.0) * 81.0 / 2400.0;
        let fz = (pc - 176.0) * 201.0 / 2400.0;
        for k2 in 0..nz3 {
            let ez = (-(k2 as f64) / 20.0).exp();
            let pz = 2.0 * PI * fz * (k2 as f64);
            let (sz, cz) = (ez * pz.sin(), ez * pz.cos());
            for k1 in 0..ny3 {
                let ey = (-(k1 as f64) / 20.0).exp();
                let py = 2.0 * PI * fy * (k1 as f64);
                let (sy, cy) = (ey * py.sin(), ey * py.cos());
                let q = [cz * cy, cz * sy, sz * cy, sz * sy];
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        let w = q[c2 * 2 + c1];
                        let base = (((k2 * 2 + c2) * ny3 + k1) * 2 + c1) * nx;
                        for k0 in 0..nx {
                            let ex = amp * (-(k0 as f64) / 90.0).exp();
                            let px = 2.0 * PI * fx * (k0 as f64);
                            full_3d_data[base + k0][0] += w * ex * px.cos();
                            full_3d_data[base + k0][1] += w * ex * px.sin();
                        }
                    }
                }
            }
        }
    }

    let fid3_uniform = Fid3D {
        name: "uniform_3d".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hncogp3d".to_string(),
        axis_x: ax_x.clone(),
        axis_y: ax_y3.clone(),
        axis_z: ax_z3.clone(),
        nus_mask: None,
        truncated_traces: None,
        data: full_3d_data.clone(),
    };
    let spec3_ref = process_fid_3d(
        &fid3_uniform,
        "HNCO",
        &ProcessingRecipe::from_fid_3d(&fid3_uniform),
    )
    .unwrap();

    let mut mask_3d = vec![false; nz3 * ny3];
    mask_3d[0] = true;
    let mut kept = 1usize;
    while kept < 202 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = ((state >> 33) as usize) % (nz3 * ny3);
        if !mask_3d[idx] {
            mask_3d[idx] = true;
            kept += 1;
        }
    }
    let mut nus_3d_data = full_3d_data;
    for k2 in 0..nz3 {
        for k1 in 0..ny3 {
            if !mask_3d[k2 * ny3 + k1] {
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        let base = (((k2 * 2 + c2) * ny3 + k1) * 2 + c1) * nx;
                        nus_3d_data[base..base + nx].fill([0.0, 0.0]);
                    }
                }
            }
        }
    }
    let fid3_nus = Fid3D {
        name: "nus_3d".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hncogp3d".to_string(),
        axis_x: ax_x,
        axis_y: ax_y3,
        axis_z: ax_z3,
        nus_mask: Some(mask_3d),
        truncated_traces: None,
        data: nus_3d_data,
    };
    let mut rec3_nus = ProcessingRecipe::from_fid_3d(&fid3_nus);
    rec3_nus.nus = Some(NusIstConfig {
        iterations: 40,
        ..Default::default()
    });
    let spec3_nus = process_fid_3d(&fid3_nus, "HNCO", &rec3_nus).expect("process 3D NUS");
    let r_3d = pearson_r_f32(&spec3_ref.data, &spec3_nus.data);
    assert!(r_3d > 0.975, "3D NUS IST correlation r={r_3d:.4} < 0.975");

    // 3. Test lossless 3D NMRPipe and Bruker pdata writers round-trip
    let tmp_dir = std::env::temp_dir().join(format!("nmr_proc_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();

    let pipe_path = tmp_dir.join("test_hnco.ft3");
    write_nmrpipe_3d(&pipe_path, &spec3_nus).expect("write_nmrpipe_3d");
    let loaded_pipe = read_nmrpipe_3d(&pipe_path, "HNCO").expect("read_nmrpipe_3d");
    assert_eq!(loaded_pipe.axis_x.size, spec3_nus.axis_x.size);
    assert_eq!(loaded_pipe.axis_y.size, spec3_nus.axis_y.size);
    assert_eq!(loaded_pipe.axis_z.size, spec3_nus.axis_z.size);
    assert!((loaded_pipe.axis_x.first_ppm - spec3_nus.axis_x.first_ppm).abs() < 1e-4);
    assert!((loaded_pipe.axis_y.first_ppm - spec3_nus.axis_y.first_ppm).abs() < 1e-3);
    assert!((loaded_pipe.axis_z.first_ppm - spec3_nus.axis_z.first_ppm).abs() < 1e-3);
    assert_eq!(
        loaded_pipe.data, spec3_nus.data,
        "NMRPipe 3D f32 data must round-trip losslessly"
    );

    let pdata_dir = tmp_dir.join("pdata_out/pdata/1");
    write_bruker_pdata_3d(&pdata_dir, &spec3_nus).expect("write_bruker_pdata_3d");
    let loaded_pdata = read_bruker_pdata_3d(&pdata_dir, "HNCO").expect("read_bruker_pdata_3d");
    assert_eq!(loaded_pdata.axis_x.size, spec3_nus.axis_x.size);
    assert_eq!(loaded_pdata.axis_y.size, spec3_nus.axis_y.size);
    assert_eq!(loaded_pdata.axis_z.size, spec3_nus.axis_z.size);
    let r_pdata = pearson_r_f32(&loaded_pdata.data, &spec3_nus.data);
    assert!(r_pdata > 0.9999999, "Bruker 3rrr round-trip r={r_pdata}");

    let _ = fs::remove_dir_all(&tmp_dir);
}

// ============================================================================
// 2. Real TopSpin Ubiquitin Validation Gate (Part 3)
// ============================================================================

fn pearson_r_f32(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let ma = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut da = 0.0_f64;
    let mut db = 0.0_f64;
    for (&xa, &xb) in a.iter().zip(b.iter()) {
        let ua = (xa as f64) - ma;
        let ub = (xb as f64) - mb;
        num += ua * ub;
        da += ua * ua;
        db += ub * ub;
    }
    let den = (da * db).sqrt();
    if den <= 1e-30 {
        0.0
    } else {
        num / den
    }
}

fn pearson_r_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let ma = a[..n].iter().sum::<f64>() / nf;
    let mb = b[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0_f64;
    let mut da = 0.0_f64;
    let mut db = 0.0_f64;
    for i in 0..n {
        let ua = a[i] - ma;
        let ub = b[i] - mb;
        num += ua * ub;
        da += ua * ua;
        db += ub * ub;
    }
    let den = (da * db).sqrt();
    if den <= 1e-30 {
        0.0
    } else {
        num / den
    }
}

#[derive(Debug)]
struct PeakComparisonStats {
    exp: String,
    n_topspin: usize,
    n_fid: usize,
    n_matched: usize,
    recall: f64,
    med_dh: f64,
    max_dh: f64,
    med_dn: f64,
    max_dn: f64,
    med_dc: Option<f64>,
    max_dc: Option<f64>,
    height_pearson_r: f64,
}

fn median_sorted(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

#[inline]
fn signed_height(p: &Peak) -> f64 {
    let h = p.height.unwrap_or(1.0);
    if p.sign < 0 {
        -h
    } else {
        h
    }
}

fn compare_peak_lists(
    exp: &str,
    ref_peaks: &[Peak],
    fid_peaks: &[Peak],
    tol_h: f64,
    tol_n: f64,
    tol_c: f64,
) -> PeakComparisonStats {
    let is_3d = ref_peaks.first().and_then(|p| p.c).is_some();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (ir, pr) in ref_peaks.iter().enumerate() {
        for (ifd, pf) in fid_peaks.iter().enumerate() {
            if pr.sign != 0 && pf.sign != 0 && pr.sign != pf.sign {
                continue;
            }
            let dh = (pr.h - pf.h) / tol_h;
            let dn = (pr.n - pf.n) / tol_n;
            let d2 = if is_3d {
                let dc = (pr.c.unwrap_or(0.0) - pf.c.unwrap_or(0.0)) / tol_c;
                dh * dh + dn * dn + dc * dc
            } else {
                dh * dh + dn * dn
            };
            if d2 <= 1.0 {
                pairs.push((d2, ir, ifd));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut used_r = vec![false; ref_peaks.len()];
    let mut used_f = vec![false; fid_peaks.len()];
    let mut dhs = Vec::new();
    let mut dns = Vec::new();
    let mut dcs = Vec::new();
    let mut hr = Vec::new();
    let mut hf = Vec::new();

    let max_hr = ref_peaks
        .iter()
        .map(|p| signed_height(p).abs())
        .fold(1e-30_f64, f64::max);
    let max_hf = fid_peaks
        .iter()
        .map(|p| signed_height(p).abs())
        .fold(1e-30_f64, f64::max);

    for (_, ir, ifd) in pairs {
        if used_r[ir] || used_f[ifd] {
            continue;
        }
        used_r[ir] = true;
        used_f[ifd] = true;
        let pr = &ref_peaks[ir];
        let pf = &fid_peaks[ifd];
        dhs.push((pr.h - pf.h).abs());
        dns.push((pr.n - pf.n).abs());
        if is_3d {
            dcs.push((pr.c.unwrap_or(0.0) - pf.c.unwrap_or(0.0)).abs());
        }
        hr.push(signed_height(pr) / max_hr);
        hf.push(signed_height(pf) / max_hf);
    }

    let n_matched = dhs.len();
    let recall = if ref_peaks.is_empty() {
        1.0
    } else {
        (n_matched as f64) / (ref_peaks.len() as f64)
    };
    let max_dh = dhs.iter().copied().fold(0.0_f64, f64::max);
    let max_dn = dns.iter().copied().fold(0.0_f64, f64::max);
    let med_dh = median_sorted(&mut dhs);
    let med_dn = median_sorted(&mut dns);
    let (med_dc, max_dc) = if is_3d {
        let mx = dcs.iter().copied().fold(0.0_f64, f64::max);
        let md = median_sorted(&mut dcs);
        (Some(md), Some(mx))
    } else {
        (None, None)
    };
    let height_pearson_r = pearson_r_f64(&hr, &hf);

    PeakComparisonStats {
        exp: exp.to_string(),
        n_topspin: ref_peaks.len(),
        n_fid: fid_peaks.len(),
        n_matched,
        recall,
        med_dh,
        max_dh,
        med_dn,
        max_dn,
        med_dc,
        max_dc,
        height_pearson_r,
    }
}

fn parse_fasta_sequence(path: &Path) -> Vec<String> {
    let txt = fs::read_to_string(path).expect("read FASTA");
    let mut seq = Vec::new();
    for line in txt.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('>') {
            continue;
        }
        for ch in t.chars() {
            if ch.is_ascii_alphabetic() {
                seq.push(one_to_three(ch).to_string());
            }
        }
    }
    seq
}

fn parse_external_shift_table(path: &Path, sequence: &[String]) -> Reference {
    let txt = fs::read_to_string(path).expect("read external shift table");
    let seq_ids: Vec<i32> = (1..=sequence.len() as i32).collect();
    let mut reference = Reference::new(sequence.to_vec(), seq_ids);
    for line in txt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("Index") {
            continue;
        }
        let normalized = trimmed
            .replace("?1", "? 1")
            .replace("?2", "? 2")
            .replace("?3", "? 3")
            .replace("?4", "? 4")
            .replace("?5", "? 5")
            .replace("?6", "? 6")
            .replace("?7", "? 7")
            .replace("?8", "? 8")
            .replace("?9", "? 9")
            .replace(":H* ?", ":H*?");
        let cols: Vec<&str> = normalized.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let Ok(seq_num) = cols[0].parse::<usize>() else {
            continue;
        };
        if seq_num == 0 || seq_num > sequence.len() {
            continue;
        }
        let k = seq_num - 1;
        let parse_val = |tok: &str| -> Option<f64> {
            if tok == "-" || tok.contains('?') {
                return None;
            }
            tok.split(':').next().unwrap_or(tok).parse::<f64>().ok()
        };
        if let Some(v) = cols.get(2).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "H".to_string()), v);
        }
        if let Some(v) = cols.get(3).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "N".to_string()), v);
        }
        if let Some(v) = cols.get(4).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "C".to_string()), v);
        }
        if let Some(v) = cols.get(5).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "CA".to_string()), v);
        }
        if let Some(v) = cols
            .iter()
            .skip(6)
            .find(|t| t.ends_with(":Cb"))
            .and_then(|t| parse_val(t))
        {
            reference.shifts.insert((k, "CB".to_string()), v);
        }
    }
    reference
}

fn map_spin_systems_to_truth(
    systems: &[nmr_core::spinsystems::SpinSystem],
    truth_ref: &Reference,
) -> Vec<i32> {
    let mut out = vec![-1i32; systems.len()];
    for (s, ss) in systems.iter().enumerate() {
        let mut best_k = -1i32;
        let mut best_d = 1.0_f64;
        for k in 1..truth_ref.sequence.len() {
            let Some(&th) = truth_ref.shifts.get(&(k, "H".to_string())) else {
                continue;
            };
            let Some(&tn) = truth_ref.shifts.get(&(k, "N".to_string())) else {
                continue;
            };
            let d = ((ss.h - th) / 0.045).hypot((ss.n - tn) / 0.40);
            if d <= 1.0 && d < best_d {
                best_d = d;
                best_k = k as i32;
            }
        }
        out[s] = best_k;
    }
    out
}

#[test]
fn real_ubiquitin_fid_vs_topspin_pdata_and_5seed_gate() {
    let Some(data_dir) = resolve_local_real_data_dir() else {
        eprintln!("Skipping real Ubiquitin FID validation: local data/ubiquitin not present");
        return;
    };
    let ubq_dir = data_dir.join("ubiquitin");
    let fasta_path = ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt");
    let truth_path = data_dir.join("ubiquitin_backbone_assignment_191104.txt");
    if !fasta_path.is_file() || !truth_path.is_file() {
        eprintln!("Skipping real Ubiquitin FID validation: FASTA or shift table missing");
        return;
    }

    let pcfg = PickerConfig::default();

    // 1. TopSpin pdata reference peaks
    let ts_hsqc_spec =
        read_bruker_pdata_2d(ubq_dir.join("Ubiquitin_090309/2/pdata/1")).expect("read TS HSQC");
    let ts_hsqc_roots = pick_hsqc_2d(&ts_hsqc_spec, &pcfg);
    let mut ts_all_peaks = ts_hsqc_roots.clone();

    // 2. Process raw Bruker `ser` FIDs:
    //    - `no_crop = true` (`--no-crop` per Spec §2.1 step 9) for full-sweep peak comparison
    //      against TopSpin `pdata/1/{2rr,3rrr}` (`STSI == FTSIZE == 2048`, identical FIR stopband MAD).
    //    - `no_crop = false` (default `5.5..=11.5 ppm` amide ROI crop) for default end-to-end pipeline.
    let t_proc = Instant::now();
    let (fid_hsqc_full, _) = process_fid_dir_2d(ubq_dir.join("Ubiquitin_090309/2"), true, false)
        .expect("process Ubiquitin HSQC FID (full)");
    let (fid_hsqc_crop, fid_hsqc_rec) =
        process_fid_dir_2d(ubq_dir.join("Ubiquitin_090309/2"), false, false)
            .expect("process Ubiquitin HSQC FID (crop)");
    println!(
        "Ubiquitin HSQC FID processed: axis_x=({} pts, {:.3}..{:.3} ppm), axis_y=({} pts, {:.3}..{:.3} ppm), recipe p0={:?}",
        fid_hsqc_crop.axis_x.size,
        fid_hsqc_crop.axis_x.first_ppm,
        fid_hsqc_crop.axis_x.min_ppm(),
        fid_hsqc_crop.axis_y.size,
        fid_hsqc_crop.axis_y.first_ppm,
        fid_hsqc_crop.axis_y.min_ppm(),
        fid_hsqc_rec.dims[0].p0_deg
    );

    let fid_hsqc_roots_full = pick_hsqc_2d(&fid_hsqc_full, &pcfg);
    let fid_hsqc_roots_crop = pick_hsqc_2d(&fid_hsqc_crop, &pcfg);
    let mut stats_table = Vec::new();
    stats_table.push(compare_peak_lists(
        "HSQC",
        &ts_hsqc_roots,
        &fid_hsqc_roots_full,
        0.04,
        0.40,
        0.50,
    ));

    let mut fid_all_peaks_crop = fid_hsqc_roots_crop.clone();
    let mut fid_all_peaks_full = fid_hsqc_roots_full.clone();
    for (exp_dir, exp_name, tol_c) in [
        ("Ubiquitin_090309/3", "HNCACB", 0.50),
        ("Ubiquitin_090309/4", "CBCACONH", 0.50),
        ("Ubiquitin_111909/10", "HNCO", 0.35),
        ("Ubiquitin_111909/11", "HNCACO", 0.35),
    ] {
        let ts_spec3 = read_bruker_pdata_3d(ubq_dir.join(exp_dir).join("pdata/1"), exp_name)
            .expect("read TS 3D");
        let ts_p3 = pick_triple_3d(&ts_spec3, &ts_hsqc_roots, &pcfg);
        ts_all_peaks.extend(ts_p3.clone());

        let (fid_spec3_full, _) = process_fid_dir_3d(ubq_dir.join(exp_dir), exp_name, true, false)
            .expect("process 3D full");
        let (fid_spec3_crop, _) = process_fid_dir_3d(ubq_dir.join(exp_dir), exp_name, false, false)
            .expect("process 3D crop");

        let fid_p3_on_ts = pick_triple_3d(&fid_spec3_full, &ts_hsqc_roots, &pcfg);
        fid_all_peaks_full.extend(pick_triple_3d(&fid_spec3_full, &fid_hsqc_roots_full, &pcfg));
        fid_all_peaks_crop.extend(pick_triple_3d(&fid_spec3_crop, &fid_hsqc_roots_crop, &pcfg));

        stats_table.push(compare_peak_lists(
            exp_name,
            &ts_p3,
            &fid_p3_on_ts,
            0.04,
            0.40,
            tol_c,
        ));
    }
    let proc_secs = t_proc.elapsed().as_secs_f64();

    println!("\n=== Ubiquitin Raw FID (`--no-crop`) vs TopSpin pdata/1 Peak Comparison ({proc_secs:.2}s total) ===");
    println!(
        "{:<10} {:>7} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Exp",
        "TS_Pk",
        "FID_Pk",
        "Match",
        "Recall%",
        "Med_dH",
        "Max_dH",
        "Med_dN",
        "Max_dN",
        "Med_dC",
        "Max_dC",
        "Height_r"
    );
    for st in &stats_table {
        println!(
            "{:<10} {:>7} {:>7} {:>7} {:>7.1}% {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9} {:>9} {:>9.4}",
            st.exp,
            st.n_topspin,
            st.n_fid,
            st.n_matched,
            st.recall * 100.0,
            st.med_dh,
            st.max_dh,
            st.med_dn,
            st.max_dn,
            st.med_dc
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_string()),
            st.max_dc
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_string()),
            st.height_pearson_r
        );
    }

    // 3. End-to-end 5-seed `evo-graph` assignment on TopSpin pdata vs Raw FID peaks
    let sequence = parse_fasta_sequence(&fasta_path);
    let truth_ref = parse_external_shift_table(&truth_path, &sequence);
    let observable_amides = 70usize;
    let priors = Priors::provisional();

    let eval_5seed = |label: &str, peaks: &[Peak]| -> (f64, f64, f64, f64) {
        let mut accs = Vec::new();
        let mut recs = Vec::new();
        let mut hcps = Vec::new();
        let mut s_accs = Vec::new();
        for seed in 0..5u64 {
            let cfg = EvoGraphConfig {
                structure_torsions: None,
                seed,
                ..Default::default()
            };
            let (systems, res, rep) = assign_evo_graph_from_peaks(peaks, &sequence, &priors, &cfg);
            let sys_truth = map_spin_systems_to_truth(&systems, &truth_ref);
            let mut m = score(&res, &sys_truth, 0.8);
            m.assignable = observable_amides;

            let shifted_pri = rep.shifted_priors(&priors);
            let (pred_ref, _) = to_reference(
                &res,
                &systems,
                &sequence,
                &truth_ref.seq_ids,
                0.0,
                Some(&shifted_pri),
            );
            let atom_cmp: BTreeMap<String, (usize, usize)> = compare_shifts(&pred_ref, &truth_ref);
            let tot_ok: usize = atom_cmp.values().map(|x| x.0).sum();
            let tot_cmp: usize = atom_cmp.values().map(|x| x.1).sum();
            let shift_pct = 100.0 * (tot_ok as f64) / (tot_cmp.max(1) as f64);

            println!(
                "  [{label}] seed {seed}: acc={:.2}% ({}/{}), rec={:.2}%, HC_prec={:.2}% ({}/{}), shift_acc={:.2}%",
                m.accuracy() * 100.0,
                m.correct,
                m.assigned,
                m.recall() * 100.0,
                m.hc_precision() * 100.0,
                m.hc_correct,
                m.hc_assigned,
                shift_pct
            );
            accs.push(m.accuracy() * 100.0);
            recs.push(m.recall() * 100.0);
            hcps.push(m.hc_precision() * 100.0);
            s_accs.push(shift_pct);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);
        let res_tuple = (mean(&accs), mean(&recs), mean(&hcps), mean(&s_accs));
        println!(
            "  [{label}] 5-SEED MEAN: acc={:.2}%, rec={:.2}%, HC_prec={:.2}%, shift_acc={:.2}%",
            res_tuple.0, res_tuple.1, res_tuple.2, res_tuple.3
        );
        res_tuple
    };

    let (ts_acc, ts_rec, ts_hcp, _) = eval_5seed("TopSpin pdata", &ts_all_peaks);
    let (fid_acc, fid_rec, fid_hcp, _) =
        eval_5seed("Raw FID default (5.5..11.5)", &fid_all_peaks_crop);
    let (full_acc, full_rec, full_hcp, _) = eval_5seed("Raw FID --no-crop", &fid_all_peaks_full);

    for (lbl, acc, rec, hcp, min_hcp) in [
        ("Raw FID default", fid_acc, fid_rec, fid_hcp, ts_hcp - 2.0),
        ("Raw FID --no-crop", full_acc, full_rec, full_hcp, 95.0),
    ] {
        assert!(
            acc >= ts_acc - 2.0,
            "Part 3 gate failed ({lbl}): 5-seed accuracy {acc:.2}% < TopSpin {ts_acc:.2}% - 2.0"
        );
        assert!(
            rec >= ts_rec - 2.0,
            "Part 3 gate failed ({lbl}): 5-seed recall {rec:.2}% < TopSpin {ts_rec:.2}% - 2.0"
        );
        assert!(
            hcp >= min_hcp,
            "Part 3 gate failed ({lbl}): 5-seed HC precision {hcp:.2}% < {min_hcp:.2}%"
        );
    }
}

/// Fit global unlabelled `(dH, dN)` reference offset (`systems - truth_ref`) by 2D Gaussian
/// marginal log-likelihood over the unlabelled empirical `(H, N)` sets.
fn fit_unlabelled_hn_offset(
    systems: &[nmr_core::spinsystems::SpinSystem],
    truth_ref: &Reference,
) -> (f64, f64) {
    let mut t_hn = Vec::new();
    for k in 1..truth_ref.sequence.len() {
        if let (Some(&h), Some(&n)) = (
            truth_ref.shifts.get(&(k, "H".to_string())),
            truth_ref.shifts.get(&(k, "N".to_string())),
        ) {
            t_hn.push((h, n));
        }
    }
    if systems.len() < 5 || t_hn.len() < 5 {
        return (0.0, 0.0);
    }
    let eval_ll = |dh: f64, dn: f64| -> f64 {
        let mut ll = -0.5 * ((dh / 0.35).powi(2) + (dn / 2.50).powi(2));
        for ss in systems {
            let mut max_lp = -4.5_f64;
            for &(th, tn) in &t_hn {
                let lp =
                    -0.5 * (((ss.h - dh - th) / 0.045).powi(2) + ((ss.n - dn - tn) / 0.40).powi(2));
                if lp > max_lp {
                    max_lp = lp;
                }
            }
            let mut sum_exp = (-4.5 - max_lp).exp();
            for &(th, tn) in &t_hn {
                let lp =
                    -0.5 * (((ss.h - dh - th) / 0.045).powi(2) + ((ss.n - dn - tn) / 0.40).powi(2));
                sum_exp += (lp - max_lp).exp();
            }
            ll += max_lp + sum_exp.ln();
        }
        ll
    };
    let ll0 = eval_ll(0.0, 0.0);
    let mut best = (0.0_f64, 0.0_f64, ll0);
    for ih in -20..=20 {
        let dh = (ih as f64) * 0.025;
        for jn in -30..=30 {
            let dn = (jn as f64) * 0.10;
            let v = eval_ll(dh, dn);
            if v > best.2 {
                best = (dh, dn, v);
            }
        }
    }
    let (c_dh, c_dn, _) = best;
    for ih in -10..=10 {
        let dh = c_dh + (ih as f64) * 0.003;
        for jn in -10..=10 {
            let dn = c_dn + (jn as f64) * 0.015;
            let v = eval_ll(dh, dn);
            if v > best.2 {
                best = (dh, dn, v);
            }
        }
    }
    if best.2 - ll0 >= 4.0 && (best.0.abs() >= 0.05 || best.1.abs() >= 0.35) {
        (best.0, best.1)
    } else {
        (0.0, 0.0)
    }
}

#[test]
fn real_cisd3_fid_and_nus_vs_topspin_gate() {
    let Some(data_dir) = resolve_local_real_data_dir() else {
        eprintln!("Skipping CISD3 FID + 3D NUS validation: local data directory not present");
        return;
    };
    let cisd3_dir = data_dir.join("private/bmrb51439_cisd3");
    let fasta_path = cisd3_dir.join("cisd3.fasta");
    let star_path = cisd3_dir.join("bmr51439_3.str");
    if !fasta_path.is_file() || !star_path.is_file() {
        eprintln!("Skipping CISD3 FID + 3D NUS validation: private/bmrb51439_cisd3 not present");
        return;
    }

    let sequence = parse_fasta_sequence(&fasta_path);
    assert_eq!(sequence.len(), 93);
    let base_truth_ref =
        nmr_core::nmrstar::read_star(&star_path, None).expect("read BMRB 51439 STAR");
    let assignable_amides = (1..sequence.len())
        .filter(|&k| {
            sequence[k] != "PRO"
                && base_truth_ref.shifts.contains_key(&(k, "H".to_string()))
                && base_truth_ref.shifts.contains_key(&(k, "N".to_string()))
        })
        .count();
    assert_eq!(assignable_amides, 51);

    let pcfg = PickerConfig::default();

    // 1. TopSpin pdata/1 baseline
    let ts_hsqc = read_bruker_pdata_2d(cisd3_dir.join("CISD3_15N_HSQC_diamagnetic/pdata/1"))
        .expect("read CISD3 TS HSQC");
    let ts_roots = pick_hsqc_2d(&ts_hsqc, &pcfg);
    let mut ts_peaks = ts_roots.clone();

    // 2. Raw FID + 3D NUS (`CISD3_CBCACONH`: `FnTYPE = 2`, `NusTD = 40 x 80`, `REVERSE = yes`)
    let t_proc = Instant::now();
    let (fid_hsqc, _) =
        process_fid_dir_2d(cisd3_dir.join("CISD3_15N_HSQC_diamagnetic"), false, false)
            .expect("process CISD3 HSQC FID");
    let fid_roots = pick_hsqc_2d(&fid_hsqc, &pcfg);

    let mut stats_table = Vec::new();
    stats_table.push(compare_peak_lists(
        "HSQC", &ts_roots, &fid_roots, 0.04, 0.40, 0.50,
    ));

    let mut fid_peaks = fid_roots.clone();
    for (exp_dir, exp_name, tol_c, expect_nus, expect_rev_y) in [
        ("CISD3_HNCA", "HNCA", 0.50, false, true),
        ("CISD3_HNCOCA", "HNCOCA", 0.50, false, true),
        ("CISD3_CBCACONH", "CBCACONH", 0.50, true, true),
        ("CISD3_HNCO", "HNCO", 0.35, false, true),
    ] {
        let ts_spec3 = read_bruker_pdata_3d(cisd3_dir.join(exp_dir).join("pdata/1"), exp_name)
            .expect("read CISD3 TS 3D");
        let ts_p3 = pick_triple_3d(&ts_spec3, &ts_roots, &pcfg);
        ts_peaks.extend(ts_p3.clone());

        let t0 = Instant::now();
        let (fid_spec3, rec3) = process_fid_dir_3d(cisd3_dir.join(exp_dir), exp_name, false, false)
            .expect("process CISD3 3D FID");
        let dt_exp = t0.elapsed().as_secs_f64();

        assert_eq!(
            rec3.nus.is_some(),
            expect_nus,
            "{exp_name} NUS detection mismatch"
        );
        assert_eq!(
            rec3.dims[1].reverse, expect_rev_y,
            "{exp_name} proc2s REVERSE flag mismatch"
        );
        println!(
            "  CISD3 {exp_name:<9} (nus={expect_nus:<5}, rev_y={expect_rev_y}): processed in {dt_exp:.2}s, axes X=({},{:.2}..{:.2}) Y=({},{:.2}..{:.2}) Z=({},{:.2}..{:.2})",
            fid_spec3.axis_x.size, fid_spec3.axis_x.first_ppm, fid_spec3.axis_x.min_ppm(),
            fid_spec3.axis_y.size, fid_spec3.axis_y.first_ppm, fid_spec3.axis_y.min_ppm(),
            fid_spec3.axis_z.size, fid_spec3.axis_z.first_ppm, fid_spec3.axis_z.min_ppm(),
        );

        let fid_p3_on_ts = pick_triple_3d(&fid_spec3, &ts_roots, &pcfg);
        let fid_p3_e2e = pick_triple_3d(&fid_spec3, &fid_roots, &pcfg);
        fid_peaks.extend(fid_p3_e2e);

        stats_table.push(compare_peak_lists(
            exp_name,
            &ts_p3,
            &fid_p3_on_ts,
            0.04,
            0.40,
            tol_c,
        ));
    }
    let proc_secs = t_proc.elapsed().as_secs_f64();

    println!("\n=== CISD3 (BMRB 51439, incl. 3D NUS CBCACONH) Raw FID vs TopSpin pdata/1 ({proc_secs:.2}s total) ===");
    println!(
        "{:<10} {:>7} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Exp",
        "TS_Pk",
        "FID_Pk",
        "Match",
        "Recall%",
        "Med_dH",
        "Max_dH",
        "Med_dN",
        "Max_dN",
        "Med_dC",
        "Max_dC",
        "Height_r"
    );
    for st in &stats_table {
        println!(
            "{:<10} {:>7} {:>7} {:>7} {:>7.1}% {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9} {:>9} {:>9.4}",
            st.exp,
            st.n_topspin,
            st.n_fid,
            st.n_matched,
            st.recall * 100.0,
            st.med_dh,
            st.max_dh,
            st.med_dn,
            st.max_dn,
            st.med_dc
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_string()),
            st.max_dc
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_string()),
            st.height_pearson_r
        );
    }

    // Verify `REVERSE = yes` on `CISD3_CBCACONH` produces sub-0.05 ppm median 15N error!
    let cbca_st = stats_table
        .iter()
        .find(|s| s.exp == "CBCACONH")
        .expect("CBCACONH stats");
    assert!(
        cbca_st.n_matched >= 15 && cbca_st.med_dn <= 0.15,
        "3D NUS CBCACONH (`REVERSE=yes`) alignment/reconstruction check failed: matched={}, med_dn={:.4}",
        cbca_st.n_matched,
        cbca_st.med_dn
    );

    // 3. Evaluate 5-seed `standard` and `evo-graph` solvers on TopSpin pdata vs Raw FID + 3D NUS IST
    let priors = Priors::bmrb();
    let scfg = nmr_core::solver::ScoreConfig::default();

    let eval_cisd3_suite = |label: &str, raw_peaks: &[Peak]| -> (f64, f64, f64, f64) {
        let mut truth_ref = base_truth_ref.clone();
        let probe_systems = nmr_core::spinsystems::build_spin_systems(
            raw_peaks,
            &nmr_core::spinsystems::Tolerances::default(),
            3,
            false,
        )
        .expect("build CISD3 spin systems");
        let (dh, dn) = fit_unlabelled_hn_offset(&probe_systems, &truth_ref);
        for ((_, atom), val) in truth_ref.shifts.iter_mut() {
            if atom == "H" {
                *val += dh;
            } else if atom == "N" {
                *val += dn;
            }
        }

        let (eff_priors, _) =
            nmr_core::solver::prepare_scoring_priors(&probe_systems, &sequence, &priors, &scfg);
        let std_truth = map_spin_systems_to_truth(&probe_systems, &truth_ref);
        let mut std_accs = Vec::new();
        for seed in 0..5u64 {
            let acfg = nmr_core::solver::AnnealConfig {
                seed,
                ..Default::default()
            };
            let res =
                nmr_core::solver::assign(&probe_systems, &sequence, &eff_priors, &scfg, &acfg);
            let mut m = score(&res, &std_truth, 0.8);
            m.assignable = assignable_amides;
            std_accs.push(m.accuracy() * 100.0);
        }

        let mut evo_accs = Vec::new();
        let mut evo_recs = Vec::new();
        let mut evo_hcps = Vec::new();
        for seed in 0..5u64 {
            let cfg = EvoGraphConfig {
                structure_torsions: None,
                seed,
                ..Default::default()
            };
            let (systems, res, _) =
                assign_evo_graph_from_peaks(raw_peaks, &sequence, &priors, &cfg);
            let sys_truth = map_spin_systems_to_truth(&systems, &truth_ref);
            let mut m = score(&res, &sys_truth, 0.8);
            m.assignable = assignable_amides;
            evo_accs.push(m.accuracy() * 100.0);
            evo_recs.push(m.recall() * 100.0);
            evo_hcps.push(m.hc_precision() * 100.0);
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);
        let out = (
            mean(&std_accs),
            mean(&evo_accs),
            mean(&evo_recs),
            mean(&evo_hcps),
        );
        println!(
            "  [{label}] 5-SEED MEAN: standard_acc={:.2}%, evo_acc={:.2}%, evo_rec={:.2}%, evo_HC_prec={:.2}%",
            out.0, out.1, out.2, out.3
        );
        out
    };

    let _ts_res = eval_cisd3_suite("CISD3 TopSpin pdata", &ts_peaks);
    let (_std_fid, evo_fid_acc, _evo_fid_rec, evo_fid_hcp) =
        eval_cisd3_suite("CISD3 Raw FID + 3D NUS IST", &fid_peaks);

    assert!(
        evo_fid_acc >= 50.0,
        "CISD3 Raw FID + 3D NUS evo-graph 5-seed accuracy {evo_fid_acc:.2}% < 50.0%"
    );
    assert!(
        evo_fid_hcp >= 90.0,
        "CISD3 Raw FID + 3D NUS evo-graph 5-seed HC precision {evo_fid_hcp:.2}% < 90.0%"
    );
}

#[test]
fn blind_3d_hncacb_autophasing_locks_ca_positive_and_cb_negative() {
    let nx = 128usize;
    let ny = 24usize;
    let nz = 32usize;
    // Unphased axes (no stored procs phases -> `PhaseSource::Auto`)
    let ax_x = make_axis_meta("1H", nx, 9600.0, 800.0, 4.70, true, 0.0);
    let ax_y = make_axis_meta("15N", ny, 2400.0, 81.0, 118.0, true, 0.0);
    let ax_z = make_axis_meta("13C", nz, 12_800.0, 201.0, 44.0, true, 0.0);

    // Inject two HNCACB spin systems with CA > 0 and CB < 0, rotated by a blind 145 deg
    // direct-dimension phase (which would invert CA < 0 and CB > 0 without biphasic lock!).
    let peaks = [
        (8.25, 119.5, 57.5, 1.25),
        (8.25, 119.5, 30.5, -0.95),
        (7.72, 114.8, 54.0, 1.15),
        (7.72, 114.8, 39.2, -0.85),
    ];
    let phi_x = 145.0_f64.to_radians();
    let phi_y = (-28.0_f64).to_radians();
    let mut data = vec![[0.0, 0.0]; nz * 2 * ny * 2 * nx];
    for &(ph, pn, pc, amp) in &peaks {
        let fx = (ph - 4.70) * 800.0 / 9600.0;
        let fy = (pn - 118.0) * 81.0 / 2400.0;
        let fz = (pc - 44.0) * 201.0 / 12_800.0;
        for k2 in 0..nz {
            let ez = (-(k2 as f64) / 22.0).exp();
            let pz = 2.0 * PI * fz * (k2 as f64);
            let (sz, cz) = (ez * pz.sin(), ez * pz.cos());
            for k1 in 0..ny {
                let ey = (-(k1 as f64) / 18.0).exp();
                let py = 2.0 * PI * fy * (k1 as f64) + phi_y;
                let (sy, cy) = (ey * py.sin(), ey * py.cos());
                let q = [cz * cy, cz * sy, sz * cy, sz * sy];
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        let w = q[c2 * 2 + c1];
                        let base = (((k2 * 2 + c2) * ny + k1) * 2 + c1) * nx;
                        for k0 in 0..nx {
                            let ex = amp * (-(k0 as f64) / 72.0).exp();
                            let px = 2.0 * PI * fx * (k0 as f64) + phi_x;
                            data[base + k0][0] += w * ex * px.cos();
                            data[base + k0][1] += w * ex * px.sin();
                        }
                    }
                }
            }
        }
    }

    let fid3 = Fid3D {
        name: "blind_hncacb".to_string(),
        vendor: FidVendor::Varian,
        pulse_program: "ghncacb".to_string(),
        axis_x: ax_x,
        axis_y: ax_y,
        axis_z: ax_z,
        nus_mask: None,
        truncated_traces: None,
        data,
    };
    let recipe = ProcessingRecipe::from_fid_3d(&fid3);
    assert_eq!(recipe.dims[0].phase_source, PhaseSource::Auto);
    let spec3 = process_fid_3d(&fid3, "HNCACB", &recipe).expect("blind 3D HNCACB process");

    let sample_at = |h: f64, n: f64, c: f64| -> f32 {
        let ix = spec3.axis_x.ppm_to_index(h).round() as usize;
        let iy = spec3.axis_y.ppm_to_index(n).round() as usize;
        let iz = spec3.axis_z.ppm_to_index(c).round() as usize;
        let mut best: f32 = 0.0;
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let x = (ix as isize + dx).clamp(0, spec3.axis_x.size as isize - 1) as usize;
                    let y = (iy as isize + dy).clamp(0, spec3.axis_y.size as isize - 1) as usize;
                    let z = (iz as isize + dz).clamp(0, spec3.axis_z.size as isize - 1) as usize;
                    let v = spec3.data[(z * spec3.axis_y.size + y) * spec3.axis_x.size + x];
                    if v.abs() > best.abs() {
                        best = v;
                    }
                }
            }
        }
        best
    };

    let ca1 = sample_at(8.25, 119.5, 57.5);
    let cb1 = sample_at(8.25, 119.5, 30.5);
    let ca2 = sample_at(7.72, 114.8, 54.0);
    let cb2 = sample_at(7.72, 114.8, 39.2);
    assert!(
        ca1 > 0.0 && ca2 > 0.0,
        "expected positive CA peaks after blind HNCACB autophase, got ca1={ca1}, ca2={ca2}"
    );
    assert!(
        cb1 < 0.0 && cb2 < 0.0,
        "expected negative CB peaks after blind HNCACB autophase, got cb1={cb1}, cb2={cb2}"
    );
}

#[test]
fn nus_2d_and_3d_hypercomplex_recover_100_to_1_adjacent_weak_peak_and_reject_unrecombined_axes() {
    // 1. Precondition enforcement: `states_recombined == false` or `QuadMode::Qf` on NUS must fail
    let nx = 64usize;
    let ny = 64usize;
    let ax_x = make_axis_meta("1H", nx, 6400.0, 600.0, 4.70, true, 0.0);
    let mut ax_y_bad = make_axis_meta("15N", ny, 1920.0, 60.8, 118.0, true, 0.0);
    ax_y_bad.states_recombined = false;
    let mask_2d: Vec<bool> = (0..ny).map(|k| k < 8 || (k * 37 + 5) % 10 < 5).collect();
    let bad_fid2 = Fid2D {
        name: "bad_nus2".into(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqc".into(),
        axis_x: ax_x.clone(),
        axis_y: ax_y_bad,
        data: vec![[0.0, 0.0]; ny * 2 * nx],
        nus_mask: Some(mask_2d.clone()),
        truncated_traces: None,
    };
    let rec_bad2 = ProcessingRecipe::from_fid_2d(&bad_fid2);
    let err2 =
        process_fid_2d(&bad_fid2, &rec_bad2).expect_err("unrecombined 2D NUS axis must fail");
    assert!(err2.contains("States"), "unexpected error: {err2}");

    // 2. 2D hypercomplex NUS with 100:1 dynamic-range weak peak adjacent to strong peak at SAME 1H shift!
    let mut ax_x_ph = ax_x.clone();
    ax_x_ph.stored_proc.p0_deg = Some(0.0);
    ax_x_ph.stored_proc.p1_deg = Some(0.0);
    let mut ax_y_ok = make_axis_meta("15N", ny, 1920.0, 60.8, 118.0, true, 0.0);
    ax_y_ok.stored_proc.p0_deg = Some(0.0);
    ax_y_ok.stored_proc.p1_deg = Some(0.0);

    let mut data2 = vec![[0.0, 0.0]; ny * 2 * nx];
    let peaks_2d = [
        (8.30, 121.5, 100.0), // Strong peak
        (8.30, 124.2, 1.0),   // 100:1 weak peak in the exact same 1H column, 2.7 ppm away!
    ];
    for &(ph, pn, amp) in &peaks_2d {
        let fx = (ph - 4.70) * 600.0 / 6400.0;
        let fy = (pn - 118.0) * 60.8 / 1920.0;
        for (k1, &sampled) in mask_2d.iter().enumerate().take(ny) {
            if !sampled {
                continue;
            }
            let ey = (-(k1 as f64) / 48.0).exp();
            let py = 2.0 * PI * fy * (k1 as f64);
            let weights = [ey * py.cos(), ey * py.sin()];
            for (c1, &wy) in weights.iter().enumerate() {
                let base = (k1 * 2 + c1) * nx;
                for k0 in 0..nx {
                    let ex = amp * (-(k0 as f64) / 48.0).exp();
                    let px = 2.0 * PI * fx * (k0 as f64);
                    data2[base + k0][0] += wy * ex * px.cos();
                    data2[base + k0][1] += wy * ex * px.sin();
                }
            }
        }
    }
    let fid2 = Fid2D {
        name: "nus2_100_to_1".into(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqc".into(),
        axis_x: ax_x_ph.clone(),
        axis_y: ax_y_ok.clone(),
        data: data2,
        nus_mask: Some(mask_2d),
        truncated_traces: None,
    };
    let rec2 = ProcessingRecipe::from_fid_2d(&fid2);
    let spec2 = process_fid_2d(&fid2, &rec2).expect("process 2D NUS 100:1");
    let ix_col = spec2.axis_x.ppm_to_index(8.30).round() as usize;
    let iy_strong = spec2.axis_y.ppm_to_index(121.5).round() as usize;
    let iy_weak = spec2.axis_y.ppm_to_index(124.2).round() as usize;
    let strong_2d = (-1isize..=1)
        .map(|d| spec2.get(ix_col, (iy_strong as isize + d) as usize))
        .fold(0.0_f32, f32::max);
    let weak_2d = (-1isize..=1)
        .map(|d| spec2.get(ix_col, (iy_weak as isize + d) as usize))
        .fold(0.0_f32, f32::max);
    let bg_2d = spec2
        .get(ix_col, spec2.axis_y.ppm_to_index(110.0).round() as usize)
        .abs();
    assert!(
        weak_2d > 0.004 * strong_2d && weak_2d > 8.0 * bg_2d.max(1e-6),
        "2D NUS failed to recover 100:1 adjacent weak peak: strong={strong_2d}, weak={weak_2d}, bg={bg_2d}"
    );

    // 3. 3D hypercomplex NUS: precondition check + 100:1 dynamic-range weak peak recovery
    let n1 = 24usize;
    let n2 = 24usize;
    let mut ax_y3 = make_axis_meta("15N", n1, 1920.0, 60.8, 118.0, true, 0.0);
    let mut ax_z3 = make_axis_meta("13C", n2, 2400.0, 150.9, 176.0, true, 0.0);
    ax_y3.stored_proc.p0_deg = Some(0.0);
    ax_y3.stored_proc.p1_deg = Some(0.0);
    ax_z3.stored_proc.p0_deg = Some(0.0);
    ax_z3.stored_proc.p1_deg = Some(0.0);
    let mask_3d: Vec<bool> = (0..(n2 * n1))
        .map(|idx| {
            let (k2, k1) = (idx / n1, idx % n1);
            (k2 < 4 && k1 < 4) || ((k2 * 31 + k1 * 17 + 3) % 10 < 5)
        })
        .collect();

    let mut bad_z3 = ax_z3.clone();
    bad_z3.quad_mode = QuadMode::Qf;
    let bad_fid3 = Fid3D {
        name: "bad_nus3".into(),
        vendor: FidVendor::Bruker,
        pulse_program: "hnco".into(),
        axis_x: ax_x_ph.clone(),
        axis_y: ax_y3.clone(),
        axis_z: bad_z3,
        data: vec![[0.0, 0.0]; n2 * 2 * n1 * 2 * nx],
        nus_mask: Some(mask_3d.clone()),
        truncated_traces: None,
    };
    let rec_bad3 = ProcessingRecipe::from_fid_3d(&bad_fid3);
    assert!(process_fid_3d(&bad_fid3, "HNCO", &rec_bad3).is_err());

    let mut data3 = vec![[0.0, 0.0]; n2 * 2 * n1 * 2 * nx];
    let peaks_3d = [
        (8.30, 121.0, 177.2, 100.0), // Strong 3D peak
        (8.30, 121.0, 173.4, 1.0),   // 100:1 weak peak on the same (1H, 15N) strip!
    ];
    for &(ph, pn, pc, amp) in &peaks_3d {
        let fx = (ph - 4.70) * 600.0 / 6400.0;
        let fy = (pn - 118.0) * 60.8 / 1920.0;
        let fz = (pc - 176.0) * 150.9 / 2400.0;
        for k2 in 0..n2 {
            let ez = (-(k2 as f64) / 20.0).exp();
            let pz = 2.0 * PI * fz * (k2 as f64);
            let (sz, cz) = (ez * pz.sin(), ez * pz.cos());
            for k1 in 0..n1 {
                if !mask_3d[k2 * n1 + k1] {
                    continue;
                }
                let ey = (-(k1 as f64) / 20.0).exp();
                let py = 2.0 * PI * fy * (k1 as f64);
                let (sy, cy) = (ey * py.sin(), ey * py.cos());
                let q = [cz * cy, cz * sy, sz * cy, sz * sy];
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        let w = q[c2 * 2 + c1];
                        let base = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * nx;
                        for k0 in 0..nx {
                            let ex = amp * (-(k0 as f64) / 48.0).exp();
                            let px = 2.0 * PI * fx * (k0 as f64);
                            data3[base + k0][0] += w * ex * px.cos();
                            data3[base + k0][1] += w * ex * px.sin();
                        }
                    }
                }
            }
        }
    }
    let fid3 = Fid3D {
        name: "nus3_100_to_1".into(),
        vendor: FidVendor::Bruker,
        pulse_program: "hnco".into(),
        axis_x: ax_x_ph,
        axis_y: ax_y3,
        axis_z: ax_z3,
        data: data3,
        nus_mask: Some(mask_3d),
        truncated_traces: None,
    };
    let rec3 = ProcessingRecipe::from_fid_3d(&fid3);
    let spec3 = process_fid_3d(&fid3, "HNCO", &rec3).expect("process 3D NUS 100:1");
    let ix3 = spec3.axis_x.ppm_to_index(8.30).round() as usize;
    let iy3 = spec3.axis_y.ppm_to_index(121.0).round() as usize;
    let iz_strong = spec3.axis_z.ppm_to_index(177.2).round() as usize;
    let iz_weak = spec3.axis_z.ppm_to_index(173.4).round() as usize;
    let strong_3d = (-1isize..=1)
        .map(|d| spec3.get(ix3, iy3, (iz_strong as isize + d) as usize))
        .fold(0.0_f32, f32::max);
    let weak_3d = (-1isize..=1)
        .map(|d| spec3.get(ix3, iy3, (iz_weak as isize + d) as usize))
        .fold(0.0_f32, f32::max);
    let flank_lo = spec3.get(ix3, iy3, iz_weak.saturating_sub(3));
    let flank_hi = spec3.get(ix3, iy3, (iz_weak + 3).min(spec3.axis_z.size - 1));
    let mut plane_bg: Vec<f32> = Vec::new();
    for iy in 0..spec3.axis_y.size {
        if iy.abs_diff(iy3) <= 3 {
            continue;
        }
        for iz in 0..spec3.axis_z.size {
            plane_bg.push(spec3.get(ix3, iy, iz).abs());
        }
    }
    plane_bg.sort_by(|a, b| a.total_cmp(b));
    let bg_3d = plane_bg[plane_bg.len() / 2];
    let flank_mean = 0.5 * (flank_lo + flank_hi);
    let weak_prom = weak_3d - flank_mean;
    assert!(
        weak_prom > 0.006 * strong_3d
            && weak_3d > 2.2 * flank_mean
            && weak_prom > 50.0 * bg_3d.max(1e-6),
        "3D NUS failed to recover 100:1 adjacent weak peak: strong={strong_3d}, weak={weak_3d}, weak_prom={weak_prom}, flanks=({flank_lo}, {flank_hi}), bg={bg_3d}"
    );
}

#[test]
fn low_snr_3d_focused_autophase_mirrored_cacb_recovery_and_rigid_hn_registration() {
    use nmr_core::process::register_3d_hn_to_hsqc_2d_with_audit;

    // Synthesize a 16-spin-system Varian-layout 3D `CBCACONH` (`axis_y = 13C`, `axis_z = 15N`)
    // where:
    // 1. `axis_y` (`13C`, carrier = 46.0 ppm, SW = 62 ppm -> 77..15 ppm) is un-annotated and
    //    carrier-mirrored (`conjugate_for_fft = false` while time-domain signal uses `+i` phase),
    //    initially placing Ala CB (19.0 ppm) into the forbidden 73.0 ppm mirror zone and CA into 32..38 ppm!
    // 2. Direct `1H` has blind phase `-86 deg` and per-trace thermal noise so single raw traces have low SNR.
    // 3. Indirect `15N` has Rance-Kay + half-dwell phase `(+90 deg, -180 deg)`.
    let nx = 128usize;
    let nc = 32usize;
    let nn = 24usize;
    let ax_x = make_axis_meta("1H", nx, 8000.0, 500.0, 4.75, false, 0.0);
    let ax_c = make_axis_meta("13C", nc, 7800.0, 125.7, 46.0, false, 0.0); // intentionally inverted conjugate_for_fft!
    let ax_n = make_axis_meta("15N", nn, 1400.0, 50.66, 119.0, true, 0.0);

    let residues: &[(f64, f64, f64, f64)] = &[
        (8.42, 120.5, 53.2, 19.0),
        (8.18, 116.2, 56.4, 19.4),
        (7.94, 123.8, 52.9, 18.8),
        (8.65, 118.4, 58.6, 63.8),
        (7.68, 114.5, 61.4, 69.5),
        (8.05, 125.1, 55.2, 31.4),
        (8.81, 121.9, 57.1, 29.8),
        (7.52, 117.0, 54.6, 39.2),
        (8.29, 112.8, 59.8, 33.1),
        (7.83, 126.4, 53.5, 19.2),
        (8.51, 115.3, 60.5, 36.8),
        (7.41, 119.7, 56.9, 28.4),
    ];

    let phi_x_true = 86.0_f64.to_radians(); // Varian non-conjugated direct phase
    let phi_n0 = (-90.0_f64).to_radians();
    let mut data3 = vec![[0.0, 0.0]; nn * 2 * nc * 2 * nx];

    // Add deterministic pseudo-Gaussian noise across all traces (`sigma ~ 0.35`, comparable to single-peak trace amplitude!)
    let mut rng_state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next_noise = || -> f64 {
        let mut s = 0.0;
        for _ in 0..4 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            s += ((rng_state >> 32) as f64) / (u32::MAX as f64) - 0.5;
        }
        s * 0.30
    };
    for sample in &mut data3 {
        sample[0] = next_noise();
        sample[1] = next_noise();
    }

    for &(ph, pn, pca, pcb) in residues {
        // Non-conjugated direct `1H`: downfield (+offset) uses `-2*pi*fx*k0` so standard forward FFT places it at `j < N/2`
        let fx = -(ph - 4.75) * 500.0 / 8000.0;
        let fn_off = (pn - 119.0) * 50.66 / 1400.0;
        for &(pc, amp) in &[(pca, 1.4), (pcb, 1.0)] {
            // Generated with `+2*pi*fc*k1` (`conjugate_for_fft = true` physics), while `ax_c.conjugate_for_fft = false`!
            let fc = (pc - 46.0) * 125.7 / 7800.0;
            for k2 in 0..nn {
                let en = (-(k2 as f64) / 20.0).exp();
                let pn_ang = 2.0 * PI * fn_off * ((k2 as f64) + 0.5) + phi_n0;
                let (sn, cn) = (en * pn_ang.sin(), en * pn_ang.cos());
                for k1 in 0..nc {
                    let ec = (-(k1 as f64) / 24.0).exp();
                    let pc_ang = 2.0 * PI * fc * (k1 as f64);
                    let (sc, cc) = (ec * pc_ang.sin(), ec * pc_ang.cos());
                    let q = [cn * cc, cn * sc, sn * cc, sn * sc];
                    for c2 in 0..2 {
                        for c1 in 0..2 {
                            let w = q[c2 * 2 + c1];
                            let base = (((k2 * 2 + c2) * nc + k1) * 2 + c1) * nx;
                            for k0 in 0..nx {
                                let ex = amp * (-(k0 as f64) / 64.0).exp();
                                let px = 2.0 * PI * fx * (k0 as f64) + phi_x_true;
                                data3[base + k0][0] += w * ex * px.cos();
                                data3[base + k0][1] += w * ex * px.sin();
                            }
                        }
                    }
                }
            }
        }
    }

    let fid3 = Fid3D {
        name: "varian_unannotated_cbca".into(),
        vendor: FidVendor::Varian,
        pulse_program: "gcbca_co_nh".into(),
        axis_x: ax_x.clone(),
        axis_y: ax_c,
        axis_z: ax_n.clone(),
        data: data3,
        nus_mask: None,
        truncated_traces: None,
    };
    let mut rec3 = ProcessingRecipe::from_fid_3d(&fid3);
    let initial_conj_c = rec3.dims[1].conjugate_for_fft;
    assert!(
        initial_conj_c,
        "DimRecipe::from_axis_meta inverts ax_c.conjugate_for_fft (false -> true)"
    );
    let mut spec3 =
        nmr_core::process::process_fid_3d_mut(&fid3, "CBCACONH", &mut rec3).expect("process 3D");

    assert_eq!(
        rec3.dims[1].conjugate_for_fft, !initial_conj_c,
        "Un-annotated carrier-mirrored 13CA/CB axis must be automatically detected and un-mirrored"
    );
    // Verify true Ala CA (53.2 ppm) and Ala CB (19.0 ppm) are strong positive peaks on residue 0!
    let mut roots_true: Vec<Peak> = residues
        .iter()
        .map(|&(h, n, _, _)| Peak {
            exp: "HSQC".into(),
            h,
            n,
            c: None,
            sign: 1,
            height: Some(100.0),
            volume: None,
            h2: None,
        })
        .collect();
    let picked_3d = pick_triple_3d(&spec3, &roots_true, &PickerConfig::default());
    let has_ca0 = picked_3d.iter().any(|p| {
        (p.h - 8.42).abs() < 0.05
            && (p.n - 120.5).abs() < 0.5
            && (p.c.unwrap_or(0.0) - 53.2).abs() < 0.8
            && p.sign > 0
    });
    let has_ala_cb0 = picked_3d.iter().any(|p| {
        (p.h - 8.42).abs() < 0.05
            && (p.n - 120.5).abs() < 0.5
            && (p.c.unwrap_or(0.0) - 19.0).abs() < 0.8
            && p.sign > 0
    });
    assert!(
        has_ca0 && has_ala_cb0,
        "Expected un-mirrored positive CA (53.2) and Ala CB (19.0) peaks, picked {} peaks",
        picked_3d.len()
    );

    // 4. Test rigid `(dH, dN)` registration of `spec3` against a shifted 2D `HSQC` (`+0.035 ppm 1H, -0.32 ppm 15N`)
    let dh_shift = 0.035;
    let dn_shift = -0.32;
    let hsqc_ax_x = AxisPpm::from_bounds("1H", 256, 10.5, 6.0, 800.0);
    let hsqc_ax_y = AxisPpm::from_bounds("15N", 128, 132.0, 108.0, 81.0);
    let mut hsqc_data = vec![0.0_f32; 128 * 256];
    for &(h, n, _, _) in residues {
        let h_ref = h + dh_shift;
        let n_ref = n + dn_shift;
        for iy in 0..128 {
            let yn = (hsqc_ax_y.index_to_ppm(iy as f64) - n_ref) / 0.22;
            for ix in 0..256 {
                let xh = (hsqc_ax_x.index_to_ppm(ix as f64) - h_ref) / 0.025;
                hsqc_data[iy * 256 + ix] += (100.0 / ((1.0 + xh * xh) * (1.0 + yn * yn))) as f32;
            }
        }
    }
    let hsqc_2d = nmr_core::spectrum::Spectrum2D {
        name: "hsqc_ref".into(),
        axis_x: hsqc_ax_x,
        axis_y: hsqc_ax_y,
        data: hsqc_data,
    };
    let (applied_dh, applied_dn, reg_audit) =
        register_3d_hn_to_hsqc_2d_with_audit(&mut spec3, &hsqc_2d, 0.08, 0.8, true);
    assert!(reg_audit.is_some());
    assert!(
        (applied_dh - dh_shift).abs() < 0.008 && (applied_dn - dn_shift).abs() < 0.06,
        "Expected rigid shift ({dh_shift:+.4}, {dn_shift:+.3}), got ({applied_dh:+.4}, {applied_dn:+.3})"
    );
    for r in &mut roots_true {
        r.h += dh_shift;
        r.n += dn_shift;
    }
    let picked_reg = pick_triple_3d(&spec3, &roots_true, &PickerConfig::default());
    assert!(
        picked_reg.len() >= 18,
        "Expected >= 18 registered 3D peaks on shifted HSQC roots, got {}",
        picked_reg.len()
    );
}
