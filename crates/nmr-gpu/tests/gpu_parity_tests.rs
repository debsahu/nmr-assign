//! Parity, multi-axis 3D CNN tiling (`<= 64 MiB` cap), and `>256` / `>1024` GPU annealer tests
//! between `nmr-core` (`f64` reference) and `nmr-gpu` (`wgpu` WGSL compute shaders).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::evaluate::score;
use nmr_core::neural_picker::{
    generate_synthetic_spectrum_3d, normalize_spectrum_data_with_nx, NeuralPicker,
    NeuralPickerConfig, SyntheticGenConfig3D, CONV3D_OUT_CH,
};
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::rng::Rng;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, Matrix2D, ScoreConfig, FORBID,
};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d, AxisPpm, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_gpu::neural_gpu::{
    normalize_spectrum_3d_for_gpu, plan_3d_tile_shape, GpuNeuralPicker, GpuNeuralPickerConfig,
    CONV3D_GPU_HIDDEN_CH, MAX_CONV3D_BUFFER_BYTES, MAX_SLAB_VOXELS,
};
use nmr_gpu::GpuEngine;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Dynamically locate an optional local `data/ubiquitin` directory (via `NMR_REAL_UBIQUITIN_DIR`,
/// `<workspace>/data/ubiquitin`, or a git worktree's parent repository) without hardcoded user paths.
fn locate_optional_ubiquitin_dir() -> Option<PathBuf> {
    if let Some(env_dir) = std::env::var_os("NMR_REAL_UBIQUITIN_DIR") {
        let p = PathBuf::from(env_dir);
        if p.is_dir() {
            return Some(p);
        }
    }
    let ws = workspace_root();
    let direct = ws.join("data/ubiquitin");
    if direct.is_dir() {
        return Some(direct);
    }
    let dot_git = ws.join(".git");
    if dot_git.is_file() {
        if let Ok(txt) = fs::read_to_string(&dot_git) {
            if let Some(gitdir_str) = txt.trim().strip_prefix("gitdir:") {
                let gitdir = Path::new(gitdir_str.trim());
                // Standard git worktree layout: `<main_repo>/.git/worktrees/<name>`
                if let Some(main_repo) = gitdir
                    .parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.parent())
                {
                    let candidate = main_repo.join("data/ubiquitin");
                    if candidate.is_dir() {
                        return Some(candidate);
                    }
                }
            }
        }
    }
    None
}

fn exact_objective_from_position(
    position: &[i32],
    u: &Matrix2D,
    p: &Matrix2D,
) -> Result<f64, String> {
    let s_len = u.rows;
    let k_len = u.cols;
    if position.len() != s_len {
        return Err(format!(
            "position.len() {} != s_len {}",
            position.len(),
            s_len
        ));
    }
    let mut at = vec![-1i32; k_len];
    for (s, &k_pos) in position.iter().enumerate() {
        if k_pos >= 0 {
            let k = k_pos as usize;
            if k >= k_len {
                return Err(format!("invalid residue index {k} >= {k_len}"));
            }
            if at[k] >= 0 {
                return Err(format!("duplicate residue assignment at k={k}"));
            }
            if u.get(s, k) <= FORBID * 0.5 {
                return Err(format!("forbidden assignment (s={s}, k={k})"));
            }
            at[k] = s as i32;
        }
    }
    let mut total = 0.0f64;
    for (k, &s_i) in at.iter().enumerate() {
        if s_i >= 0 {
            total += u.get(s_i as usize, k);
        }
    }
    for k in 0..k_len.saturating_sub(1) {
        if at[k] >= 0 && at[k + 1] >= 0 {
            total += p.get(at[k] as usize, at[k + 1] as usize);
        }
    }
    Ok(total)
}

fn pad_with_decoy_spin_systems(
    base_systems: &[SpinSystem],
    base_truth: &[i32],
    target_s_len: usize,
    seed: u64,
) -> (Vec<SpinSystem>, Vec<i32>) {
    let mut systems = base_systems.to_vec();
    let mut truth = base_truth.to_vec();
    let mut rng = Rng::new(seed);
    while systems.len() < target_s_len {
        let h = 6.2 + rng.f64() * 3.8;
        let n = 104.0 + rng.f64() * 28.0;
        let mut ss = SpinSystem::new(h, n);
        // Realistic t1-noise / artifact spin systems: single-peak or discordant (i, i-1) carbon
        // candidates that remain admissible in `U` (`slots_packed` non-empty) so every MCMC move
        // genuinely samples all `S` spin systems without outscoring true 6-link sequential chains.
        let ca0 = if rng.f64() < 0.5 {
            39.5 + rng.f64() * 4.5
        } else {
            65.5 + rng.f64() * 5.5
        };
        let cb0 = 46.0 + rng.f64() * 14.0;
        ss.cands.insert(("CA".to_string(), 0), vec![(ca0, 1)]);
        ss.cands.insert(("CB".to_string(), 0), vec![(cb0, 1)]);
        ss.cands
            .insert(("CA".to_string(), -1), vec![(38.0 + rng.f64() * 3.0, 1)]);
        ss.cands
            .insert(("C".to_string(), -1), vec![(165.0 + rng.f64() * 2.5, 1)]);
        systems.push(ss);
        truth.push(-1);
    }
    (systems, truth)
}

#[test]
fn test_gpu_scoring_and_annealing_parity() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(star_path, None).unwrap();
    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::provisional();
    let scfg = ScoreConfig::default();

    let gpu = GpuEngine::new_headless().expect("headless wgpu adapter");
    for assign_cost in [0.0, 1.25] {
        let scfg_test = ScoreConfig {
            assign_cost,
            ..ScoreConfig::default()
        };
        let u_cpu = unary_matrix(&systems, &ref_data.sequence, &priors, &scfg_test);
        let p_cpu = pair_matrix(&systems, &scfg_test);

        let (u_gpu, p_gpu) = gpu
            .score_matrices_gpu(&systems, &ref_data.sequence, &priors, &scfg_test)
            .expect("GPU scoring matrices");

        assert_eq!(u_cpu.rows, u_gpu.rows);
        assert_eq!(u_cpu.cols, u_gpu.cols);
        for i in 0..u_cpu.data.len() {
            let a = u_cpu.data[i];
            let b = u_gpu.data[i];
            // f32 has ~1.5e-5 ppm ULP near 176 ppm (C'), giving ~3e-4 ULP sensitivity in (d / 0.12)^2
            if a < -1e8 {
                assert!(b < -1e8, "expected FORBID at {i}: cpu={a}, gpu={b}");
            } else {
                assert!(
                    (a - b).abs() < 1e-3,
                    "U mismatch (assign_cost={assign_cost}) at {i}: cpu={a}, gpu={b}"
                );
            }
        }

        for i in 0..p_cpu.data.len() {
            let a = p_cpu.data[i];
            let b = p_gpu.data[i];
            if a < -1e8 {
                assert!(b < -1e8);
            } else {
                assert!((a - b).abs() < 1e-3, "P mismatch at {i}: cpu={a}, gpu={b}");
            }
        }
    }

    let res_gpu = gpu
        .assign_gpu(
            &systems,
            &ref_data.sequence,
            &priors,
            &scfg,
            &AnnealConfig {
                runs: 32,
                steps_per_system: 1200,
                seed: 0,
                ..AnnealConfig::default()
            },
        )
        .expect("GPU annealing");
    let m = score(&res_gpu, &truth, 0.8);
    assert!(m.recall() >= 0.90, "{}", m.report());
    assert!(m.hc_precision() >= 0.95, "{}", m.report());
}

#[test]
fn test_conv3d_multi_axis_tiling_parity_and_64mib_cap() {
    let strict_gpu = match GpuEngine::new_headless_with_limits(wgpu::Limits::default()) {
        Ok(g) => g,
        Err(err) => {
            eprintln!("Skipping WebGPU test (no GPU adapter available): {err}");
            return;
        }
    };
    let cpu_picker = NeuralPicker::pretrained();
    let gpu_picker = GpuNeuralPicker::pretrained(&strict_gpu).expect("init strict GpuNeuralPicker");

    // 1. Verify multi-axis (X, Y, Z) tile planner strictly caps every 512 MiB (2048x128x512) buffer <= 64 MiB.
    for (ax_span, ay_span, az_span) in [
        (2048usize, 128usize, 512usize),
        (980, 128, 512),
        (2048, 512, 128),
        (4096, 256, 512),
    ] {
        let (tx, ty, tz) = plan_3d_tile_shape(
            0,
            ax_span,
            0,
            ay_span,
            0,
            az_span,
            ax_span,
            ay_span,
            az_span,
            7,
            MAX_SLAB_VOXELS,
        );
        let tile_vox = tx * ty * tz;
        let hidden_bytes = (tile_vox * CONV3D_GPU_HIDDEN_CH * 4) as u64;
        assert!(
            tile_vox <= MAX_SLAB_VOXELS && hidden_bytes <= MAX_CONV3D_BUFFER_BYTES,
            "tile ({tx}x{ty}x{tz}) = {hidden_bytes} B exceeded 64 MiB cap ({MAX_CONV3D_BUFFER_BYTES} B)"
        );
        assert!(tx >= 15 && ty >= 15 && tz >= 15);
    }

    // 2. Force multi-tile (X, Y, Z) decomposition (`max_tile_voxels = 8_192`) on a 52x36x48 HNCACB cube
    //    (89,856 voxels -> >15 overlapping 3D tiles) and verify <1e-4 parity with single-block GPU and CPU.
    let syn = generate_synthetic_spectrum_3d(&SyntheticGenConfig3D {
        nx: 52,
        ny: 36,
        nz: 48,
        exp: "HNCACB".to_string(),
        n_peaks: 18,
        allow_negative: true,
        min_snr: 4.5,
        max_snr: 26.0,
        ridge_prob: 1.0,
        ridge_rel_amp: 0.18,
        baseline_roll_sigma: 2.0,
        shoulder_prob: 0.40,
        j_split_prob: 0.25,
        seed: 20261112,
        ..SyntheticGenConfig3D::default()
    });
    let (nx, ny, nz) = (syn.spectrum.nx(), syn.spectrum.ny(), syn.spectrum.nz());
    let (norm_cpu, _, _) = normalize_spectrum_data_with_nx(&syn.spectrum.data, nx);
    let (norm_gpu, _, _) = normalize_spectrum_3d_for_gpu(&syn.spectrum.data, nx);

    let out_cpu = cpu_picker.net3d.forward_3d(&norm_cpu, nx, ny, nz);
    let out_tiled = gpu_picker
        .forward_3d_gpu_tiled(&norm_gpu, nx, ny, nz, 8_192)
        .expect("multi-tile forward_3d_gpu_tiled");
    assert!(
        gpu_picker.last_max_binding_bytes() <= (8_192 * CONV3D_GPU_HIDDEN_CH * 4) as u64,
        "expected <= 512 KiB hidden binding for 8192-voxel tile, got {} B",
        gpu_picker.last_max_binding_bytes()
    );
    assert_eq!(out_cpu.len(), out_tiled.len());
    let mut max_fwd_diff = 0.0f32;
    for (idx, (&c, &g)) in out_cpu.iter().zip(out_tiled.iter()).enumerate() {
        let d = (c - g).abs();
        if d > max_fwd_diff {
            max_fwd_diff = d;
        }
        assert!(
            d < 1e-4,
            "multi-tile forward mismatch at idx {idx} (ch {}): cpu={c} vs gpu={g}",
            idx % CONV3D_OUT_CH
        );
    }
    println!("Multi-tile (X,Y,Z) forward_3d max_abs_diff = {max_fwd_diff:.3e}");

    let cpu_cfg = NeuralPickerConfig {
        prob_threshold: 0.48,
        min_snr_norm: 0.08,
        allow_negative: true,
        ..NeuralPickerConfig::default()
    };
    let gpu_cfg = GpuNeuralPickerConfig {
        prob_threshold: 0.48,
        min_snr_norm: 0.08,
        allow_negative: true,
        ..GpuNeuralPickerConfig::default()
    };

    let cpu_full = cpu_picker.pick_3d(&syn.spectrum, "HNCACB", &cpu_cfg);
    let gpu_single = gpu_picker
        .pick_triple_3d_gpu(&syn.spectrum, "HNCACB", &[], &gpu_cfg)
        .expect("single-block GPU pick");
    let gpu_tiled = gpu_picker
        .pick_triple_3d_gpu_tiled(&syn.spectrum, "HNCACB", &[], &gpu_cfg, 8_192)
        .expect("multi-tile (X,Y,Z) GPU pick");

    assert!(!cpu_full.is_empty());
    assert_eq!(cpu_full.len(), gpu_single.len());
    assert_eq!(cpu_full.len(), gpu_tiled.len());
    for (i, ((cp, sp), tp)) in cpu_full
        .iter()
        .zip(gpu_single.iter())
        .zip(gpu_tiled.iter())
        .enumerate()
    {
        assert_eq!(cp.sign, tp.sign, "sign mismatch at peak {i}");
        assert_eq!(
            sp.sign, tp.sign,
            "single vs tiled sign mismatch at peak {i}"
        );
        assert!(
            (cp.h - tp.h).abs() < 1e-4,
            "H mismatch at {i}: {} vs {}",
            cp.h,
            tp.h
        );
        assert!(
            (cp.n - tp.n).abs() < 1e-4,
            "N mismatch at {i}: {} vs {}",
            cp.n,
            tp.n
        );
        assert!(
            (cp.c.unwrap() - tp.c.unwrap()).abs() < 1e-4,
            "C mismatch at {i}: {:?} vs {:?}",
            cp.c,
            tp.c
        );
        assert!((sp.h - tp.h).abs() < 1e-6);
        assert!((sp.n - tp.n).abs() < 1e-6);
        assert!((sp.c.unwrap() - tp.c.unwrap()).abs() < 1e-6);
    }

    // Verify HSQC-strip X/N bounding-box cropping + multi-tile coordinate map-back parity.
    let roots: Vec<Peak> = syn
        .peaks
        .iter()
        .take(6)
        .map(|p| Peak::hsqc(p.h, p.n))
        .collect();
    let cpu_roots_pks =
        cpu_picker.pick_3d(&syn.spectrum, "HNCACB", &cpu_cfg.clone().with_roots(&roots));
    let gpu_roots_pks = gpu_picker
        .pick_triple_3d_gpu_tiled(&syn.spectrum, "HNCACB", &roots, &gpu_cfg, 8_192)
        .expect("multi-tile strip-cropped GPU pick");
    assert_eq!(cpu_roots_pks.len(), gpu_roots_pks.len());
    for (cp, gp) in cpu_roots_pks.iter().zip(gpu_roots_pks.iter()) {
        assert_eq!(cp.sign, gp.sign);
        assert!((cp.h - gp.h).abs() < 1e-4);
        assert!((cp.n - gp.n).abs() < 1e-4);
        assert!((cp.c.unwrap() - gp.c.unwrap()).abs() < 1e-4);
    }

    // 3. Full 2048 x 128 x 512 (`512 MiB` = 134,217,728 f32 voxels) Bruker-scale cube on strict
    //    128 MiB baseline WebGPU limits, verifying `last_max_binding_bytes() <= 64 MiB`.
    let (big_nx, big_ny, big_nz) = (2048usize, 128usize, 512usize);
    let axis_x = AxisPpm::from_bounds("1H", big_nx, 11.5, 4.5, 800.13);
    let axis_y = AxisPpm::from_bounds("15N", big_ny, 134.0, 104.0, 81.08);
    let axis_z = AxisPpm::from_bounds("13C", big_nz, 78.0, 14.0, 201.19);
    let mut big_data = vec![0.0f32; big_nx * big_ny * big_nz];
    // Deterministic lightweight pseudo-thermal background across rows so Haar noise estimation is realistic.
    for iz in (0..big_nz).step_by(8) {
        let z_base = iz * big_ny * big_nx;
        for iy in (0..big_ny).step_by(4) {
            let row = z_base + iy * big_nx;
            let mut st = ((iz as u32) * 7919) ^ ((iy as u32) * 104_729) ^ 0x9e37_79b9;
            for ix in 0..big_nx {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = ((st >> 8) as f32) * (1.0 / 16_777_216.0) - 0.5;
                big_data[row + ix] = u * 2.0;
            }
        }
    }
    // Inject +CA and -CB peaks across widely separated (1H, 15N, 13C) coordinates so active X/Z spans multiple tiles.
    let injected = [
        (9.55, 127.8, 56.4, 28.0f32),
        (9.55, 127.8, 31.2, -24.0f32),
        (8.32, 118.4, 61.5, 30.0f32),
        (8.32, 118.4, 28.6, -25.0f32),
        (7.05, 109.2, 53.8, 29.0f32),
        (7.05, 109.2, 39.4, -23.0f32),
    ];
    for &(h_ppm, n_ppm, c_ppm, amp) in &injected {
        let cx = axis_x.ppm_to_index(h_ppm);
        let cy = axis_y.ppm_to_index(n_ppm);
        let cz = axis_z.ppm_to_index(c_ppm);
        let ix0 = (cx.floor() as isize - 6).max(0) as usize;
        let ix1 = (cx.ceil() as isize + 6).min(big_nx as isize - 1) as usize;
        let iy0 = (cy.floor() as isize - 5).max(0) as usize;
        let iy1 = (cy.ceil() as isize + 5).min(big_ny as isize - 1) as usize;
        let iz0 = (cz.floor() as isize - 6).max(0) as usize;
        let iz1 = (cz.ceil() as isize + 6).min(big_nz as isize - 1) as usize;
        for iz in iz0..=iz1 {
            let dz = (iz as f64 - cz) / 1.65;
            for iy in iy0..=iy1 {
                let dy = (iy as f64 - cy) / 1.45;
                let row = (iz * big_ny + iy) * big_nx;
                for ix in ix0..=ix1 {
                    let dx = (ix as f64 - cx) / 1.75;
                    let g = (-0.5 * (dx * dx + dy * dy + dz * dz)).exp() as f32;
                    big_data[row + ix] += amp * g;
                }
            }
        }
    }
    let big_spec = Spectrum3D {
        name: "synthetic_512mib_hncacb".to_string(),
        exp: "HNCACB".to_string(),
        axis_x,
        axis_y,
        axis_z,
        data: big_data,
    };
    let big_roots = vec![
        Peak::hsqc(9.55, 127.8),
        Peak::hsqc(8.32, 118.4),
        Peak::hsqc(7.05, 109.2),
    ];
    let big_gpu_peaks = gpu_picker
        .pick_triple_3d_gpu(&big_spec, "HNCACB", &big_roots, &gpu_cfg)
        .expect("512 MiB 2048x128x512 GPU pick on strict baseline limits");
    let big_max_bind = gpu_picker.last_max_binding_bytes();
    println!(
        "512 MiB (2048x128x512) synthetic HNCACB picked {} peaks, max GPU binding = {:.2} MiB ({big_max_bind} B)",
        big_gpu_peaks.len(),
        big_max_bind as f64 / (1024.0 * 1024.0)
    );
    assert!(
        big_max_bind <= MAX_CONV3D_BUFFER_BYTES,
        "GPU binding {big_max_bind} B exceeded 64 MiB cap ({MAX_CONV3D_BUFFER_BYTES} B)"
    );
    let big_cpu_peaks =
        cpu_picker.pick_3d(&big_spec, "HNCACB", &cpu_cfg.clone().with_roots(&big_roots));
    assert_eq!(big_cpu_peaks.len(), big_gpu_peaks.len());
    assert!(
        big_gpu_peaks.len() >= injected.len(),
        "expected all {} injected peaks, got {}",
        injected.len(),
        big_gpu_peaks.len()
    );
    for (cp, gp) in big_cpu_peaks.iter().zip(big_gpu_peaks.iter()) {
        assert_eq!(cp.sign, gp.sign);
        assert!((cp.h - gp.h).abs() < 1e-4);
        assert!((cp.n - gp.n).abs() < 1e-4);
        assert!((cp.c.unwrap() - gp.c.unwrap()).abs() < 1e-4);
    }
    drop(big_spec);

    // 4. Optional real Bruker Ubiquitin_090309 512 MiB HNCACB (`3/pdata/1`) verification if present on disk.
    if let Some(ubq_dir) = locate_optional_ubiquitin_dir() {
        let hsqc_pdata = ubq_dir.join("Ubiquitin_090309/2/pdata/1");
        let hncacb_pdata = ubq_dir.join("Ubiquitin_090309/3/pdata/1");
        if hsqc_pdata.is_dir() && hncacb_pdata.is_dir() {
            let hsqc = read_bruker_pdata_2d(&hsqc_pdata).expect("read real Bruker 2D HSQC");
            let hncacb =
                read_bruker_pdata_3d(&hncacb_pdata, "HNCACB").expect("read real Bruker 3D HNCACB");
            assert_eq!(
                (hncacb.nx(), hncacb.ny(), hncacb.nz()),
                (2048, 128, 512),
                "expected 2048x128x512 Bruker HNCACB cube"
            );
            let hsqc_roots = cpu_picker.pick_2d(&hsqc, &NeuralPickerConfig::default());
            let t0 = Instant::now();
            let real_peaks = gpu_picker
                .pick_triple_3d_gpu(
                    &hncacb,
                    "HNCACB",
                    &hsqc_roots,
                    &GpuNeuralPickerConfig {
                        allow_negative: true,
                        ..GpuNeuralPickerConfig::default()
                    },
                )
                .expect("GPU neural pick on real 512 MiB Bruker HNCACB");
            let real_bind = gpu_picker.last_max_binding_bytes();
            println!(
                "Real Bruker 512 MiB HNCACB: {} roots -> {} peaks (+CA={}, -CB={}) in {:.2}s, max binding = {:.2} MiB",
                hsqc_roots.len(),
                real_peaks.len(),
                real_peaks.iter().filter(|p| p.sign > 0).count(),
                real_peaks.iter().filter(|p| p.sign < 0).count(),
                t0.elapsed().as_secs_f64(),
                real_bind as f64 / (1024.0 * 1024.0)
            );
            assert!(
                real_bind <= MAX_CONV3D_BUFFER_BYTES,
                "real Bruker GPU binding {real_bind} B exceeded 64 MiB cap"
            );
            assert!(real_peaks.iter().any(|p| p.sign > 0), "expected +CA peaks");
            assert!(real_peaks.iter().any(|p| p.sign < 0), "expected -CB peaks");
        }
    }
}

#[test]
fn test_gpu_annealer_above_256_and_above_1024_spin_systems() {
    let strict_gpu = match GpuEngine::new_headless_with_limits(wgpu::Limits::default()) {
        Ok(g) => g,
        Err(err) => {
            eprintln!("Skipping WebGPU annealer test (no GPU adapter available): {err}");
            return;
        }
    };

    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(star_path, None).unwrap();
    let (peaks, base_truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let base_systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::provisional();
    let scfg = ScoreConfig::default();

    // 1. Test `S = 700, K = 76` (`anneal_chunk_1024` workgroup-SRAM pipeline within 16 KiB baseline limit).
    let (systems_700, truth_700) =
        pad_with_decoy_spin_systems(&base_systems, &base_truth, 700, 700);
    assert_eq!(systems_700.len(), 700);
    assert_eq!(ref_data.sequence.len(), 76);

    let (u_700, p_700) = strict_gpu
        .score_matrices_gpu(&systems_700, &ref_data.sequence, &priors, &scfg)
        .expect("GPU score_matrices_gpu for S=700, K=76");

    let cfg_700 = AnnealConfig {
        runs: 64,
        steps_per_system: 600,
        seed: 42,
        ..AnnealConfig::default()
    };
    let t0 = Instant::now();
    let res_gpu_700 = strict_gpu
        .anneal_from_matrices_gpu(&u_700, &p_700, &cfg_700)
        .expect("GPU anneal_from_matrices_gpu for S=700, K=76");
    let gpu_700_secs = t0.elapsed().as_secs_f64();

    let res_cpu_700 = assign_from_matrices(
        &u_700,
        &p_700,
        &AnnealConfig {
            runs: 16,
            steps_per_system: 600,
            seed: 42,
            ..AnnealConfig::default()
        },
    );

    let exact_obj_700 =
        exact_objective_from_position(&res_gpu_700.position, &u_700, &p_700).unwrap();
    assert!(
        (res_gpu_700.score - exact_obj_700).abs() < 1e-2,
        "S=700 score mismatch: reported={} vs exact={exact_obj_700}",
        res_gpu_700.score
    );
    assert_eq!(res_gpu_700.run_scores.len(), 64);

    let m_gpu_700 = score(&res_gpu_700, &truth_700, 0.8);
    let m_cpu_700 = score(&res_cpu_700, &truth_700, 0.8);
    println!(
        "S=700, K=76 GPU anneal: {:.3}s (64 replicas x 420k steps), obj={:.1} (CPU obj={:.1}), recall={:.1}% (CPU={:.1}%)",
        gpu_700_secs,
        res_gpu_700.score,
        res_cpu_700.score,
        m_gpu_700.recall() * 100.0,
        m_cpu_700.recall() * 100.0
    );
    assert!(
        m_gpu_700.recall() >= 0.88,
        "S=700 GPU recall regressed: {}",
        m_gpu_700.report()
    );
    // Check strong consensus agreement on the true ubiquitin spin systems between GPU and CPU annealers.
    let agree_true = (0..base_systems.len())
        .filter(|&s| truth_700[s] >= 0 && res_gpu_700.position[s] == res_cpu_700.position[s])
        .count();
    let total_true = base_truth.iter().filter(|&&t| t >= 0).count();
    assert!(
        agree_true * 100 >= total_true * 85,
        "S=700 GPU vs CPU consensus agreement too low: {agree_true}/{total_true}"
    );

    // 2. Test `S = 1200, K = 76` (`anneal_chunk_global` storage-buffer pipeline above 1024 slots).
    let (systems_1200, truth_1200) =
        pad_with_decoy_spin_systems(&base_systems, &base_truth, 1200, 1200);
    let (u_1200, p_1200) = strict_gpu
        .score_matrices_gpu(&systems_1200, &ref_data.sequence, &priors, &scfg)
        .expect("GPU score_matrices_gpu for S=1200, K=76");
    let t1 = Instant::now();
    let res_gpu_1200 = strict_gpu
        .anneal_from_matrices_gpu(
            &u_1200,
            &p_1200,
            &AnnealConfig {
                runs: 32,
                steps_per_system: 600,
                seed: 7,
                ..AnnealConfig::default()
            },
        )
        .expect("GPU anneal_from_matrices_gpu for S=1200, K=76");
    let gpu_1200_secs = t1.elapsed().as_secs_f64();
    let exact_obj_1200 =
        exact_objective_from_position(&res_gpu_1200.position, &u_1200, &p_1200).unwrap();
    assert!(
        (res_gpu_1200.score - exact_obj_1200).abs() < 1e-2,
        "S=1200 score mismatch: reported={} vs exact={exact_obj_1200}",
        res_gpu_1200.score
    );
    let unique_assigned: HashSet<i32> = res_gpu_1200
        .position
        .iter()
        .copied()
        .filter(|&k| k >= 0)
        .collect();
    assert!(
        unique_assigned.len() >= 64,
        "expected >= 64 assigned residues out of 68 assignable, got {}",
        unique_assigned.len()
    );
    let m_gpu_1200 = score(&res_gpu_1200, &truth_1200, 0.8);
    println!(
        "S=1200, K=76 GPU global anneal: {:.3}s (32 replicas x 720k steps), obj={:.1}, recall={:.1}%",
        gpu_1200_secs,
        res_gpu_1200.score,
        m_gpu_1200.recall() * 100.0
    );
    assert!(
        m_gpu_1200.recall() >= 0.85,
        "S=1200 GPU global recall regressed: {}",
        m_gpu_1200.report()
    );

    // 3. Verify loud `Err(...)` (never a quiet CPU fallback) when buffers exceed GPU binding limits.
    let (systems_6000, _) = pad_with_decoy_spin_systems(&base_systems, &base_truth, 6_000, 6_000);
    let over_limit_res =
        strict_gpu.score_matrices_gpu(&systems_6000, &ref_data.sequence, &priors, &scfg);
    assert!(
        over_limit_res.is_err(),
        "expected explicit GPU binding limit Err on S=6000 (144 MiB P > 128 MiB strict limit)"
    );
}
