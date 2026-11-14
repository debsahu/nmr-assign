//! Clean-room WebGPU (`wgpu` + WGSL) parity, TDR chunking equivalence, device buffer-limit
//! enforcement, and accuracy/latency verification suite for `nmr_gpu::bayes_sa_gpu`
//! (`docs/specs/19-bayesian-sa-solver.md`).
//!
//! # Provenance (100% Clean-Room)
//! 1. **Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J.** (2023).
//!    "Robust automated backbone triple resonance NMR assignments of proteins using
//!    Bayesian-based simulated annealing." *Nature Communications* **14**, 1556.
//!    DOI: `10.1038/s41467-023-37219-z`.
//! 2. **Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P.** (1983). "Optimization by Simulated
//!    Annealing." *Science* **220**(4598), 671–680.
//! 3. **Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.** (1995).
//!    *J. Biomol. NMR* **5**, 67–81 & **6**, 135–140 (primary-sequence covalent neighbor shifts).

use std::path::PathBuf;
use std::time::Instant;

use nmr_core::bayes_sa::{
    build_bayes_move_tables, build_initial_bayes_replicas, compute_bayes_energy_matrices,
    condition_peaks_preserving_hsqc_roots, evaluate_state_energy, BayesSaConfig, LockedAssignment,
    FORBID_ENERGY,
};
use nmr_core::evaluate::{score, Metrics};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::priors::{delta_covalent, Priors};
use nmr_core::rng::Rng;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::spinsystems::build_spin_systems;
use nmr_gpu::bayes_sa_gpu::{
    compute_bayes_energy_matrices_gpu, evaluate_states_energy_gpu,
    validate_bayes_gpu_buffer_budget, BayesSaGpuSolver, MAX_BAYES_GPU_CHUNK_STEPS,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_gpu_solver() -> Option<BayesSaGpuSolver> {
    match BayesSaGpuSolver::new_headless() {
        Ok(solver) => Some(solver),
        Err(err) => {
            let allow_no_gpu = std::env::var("NMR_ALLOW_NO_GPU")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if allow_no_gpu {
                eprintln!("Skipping WebGPU Bayesian SA test (NMR_ALLOW_NO_GPU=1): {err}");
                None
            } else {
                panic!(
                    "Hardware WebGPU adapter required for bayes_sa_gpu_tests (set NMR_ALLOW_NO_GPU=1 to skip on headless CI without GPU): {err}"
                );
            }
        }
    }
}

/// Build a clean-room synthetic protein `Reference` with realistic secondary structure
/// ($\alpha$-helix, $\beta$-strand, random coil), Wishart covalent neighbor shifts, and thermal jitter,
/// matching `bayes_sa_tests.rs`.
fn build_synthetic_protein_reference(length: usize, seed: u64) -> Reference {
    let priors = Priors::default();
    let mut rng = Rng::new(seed);

    let folded_alphabet = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];

    let mut sequence = Vec::with_capacity(length);
    let mut seq_ids = Vec::with_capacity(length);
    sequence.push("MET".to_string());
    seq_ids.push(1);

    for k in 1..length {
        let aa = folded_alphabet[rng.usize(folded_alphabet.len())];
        sequence.push(aa.to_string());
        seq_ids.push((k + 1) as i32);
    }

    // Assign secondary-structure segments: 0 = alpha, 1 = beta, 2 = coil
    let mut ss_state = vec![2usize; length];
    let mut pos = 3usize;
    let mut motif = 0usize;
    while pos + 8 < length {
        let seg_len = 7 + rng.usize(8);
        let end = (pos + seg_len).min(length - 2);
        let conf = if motif.is_multiple_of(2) { 0 } else { 1 };
        for slot in &mut ss_state[pos..end] {
            if *slot == 2 {
                *slot = conf;
            }
        }
        for k in pos..end {
            if sequence[k] == "PRO" && ss_state[k] == 0 {
                ss_state[k] = 2;
            }
        }
        pos = end + 3 + rng.usize(5);
        motif += 1;
    }

    let mut ref_data = Reference::new(sequence.clone(), seq_ids);
    for k in 0..length {
        let res = &sequence[k];
        let conf = ss_state[k];
        let (cov_n, cov_ca, cov_cb, cov_c) = delta_covalent(&sequence, k);

        let (d_ca, d_cb, d_c, d_n, d_h) = match (res.as_str(), conf) {
            ("GLY", 0) => (1.5, 0.0, 1.4, -1.2, -0.15),
            ("GLY", 1) => (-0.9, 0.0, -1.1, 1.0, 0.20),
            (_, 0) => (2.6, -0.4, 1.7, -1.5, -0.25),
            (_, 1) => (-1.4, 1.5, -1.3, 1.4, 0.32),
            _ => (-0.2, 0.0, -0.15, 0.0, 0.0),
        };

        if let Some((m, sd)) = priors.get(res, "CA") {
            let val = m + d_ca + cov_ca + rng.normal(0.0, (sd * 0.65).min(1.25));
            ref_data.shifts.insert((k, "CA".to_string()), val);
        }
        if res != "GLY" {
            if let Some((m, sd)) = priors.get(res, "CB") {
                let val = m + d_cb + cov_cb + rng.normal(0.0, (sd * 0.65).min(1.35));
                ref_data.shifts.insert((k, "CB".to_string()), val);
            }
        }
        if let Some((m, sd)) = priors.get(res, "C") {
            let val = m + d_c + cov_c + rng.normal(0.0, (sd * 0.65).min(1.20));
            ref_data.shifts.insert((k, "C".to_string()), val);
        }
        if k > 0 && res != "PRO" {
            if let Some((m, sd)) = priors.get(res, "N") {
                let val = m + d_n + cov_n + rng.normal(0.0, (sd * 0.80).min(3.2));
                ref_data.shifts.insert((k, "N".to_string()), val);
            }
            if let Some((m, sd)) = priors.get(res, "H") {
                let val = (m + d_h + rng.normal(0.0, (sd * 0.80).min(0.58))).clamp(5.9, 10.6);
                ref_data.shifts.insert((k, "H".to_string()), val);
            }
        }
    }

    ref_data
}

#[test]
fn cpu_vs_webgpu_pseudo_energy_parity_under_1e_4() {
    let Some(gpu) = require_gpu_solver() else {
        return;
    };
    let priors = Priors::default();

    // Part A: 76-aa Synthetic Ubiquitin (realistic Bruker 5-spectrum simulation + locked residue)
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ubq = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let (peaks_ubq, truth_ubq) = simulate(
        &ubq,
        &SimConfig {
            seed: 2,
            ..SimConfig::realistic_bruker()
        },
    );
    let cond_ubq =
        condition_peaks_preserving_hsqc_roots(&peaks_ubq, &BayesSaConfig::default().tolerances);
    let sys_ubq = build_spin_systems(&cond_ubq, &BayesSaConfig::default().tolerances, 8, false)
        .expect("build ubq spin systems");

    let (lock_s, lock_k) = truth_ubq
        .iter()
        .enumerate()
        .find_map(|(s, &t)| (t >= 3 && s < sys_ubq.len()).then_some((s, t as usize)))
        .expect("lockable residue");

    let cfg_ubq = BayesSaConfig {
        locked_assignments: vec![LockedAssignment::new(lock_s, lock_k)],
        ..BayesSaConfig::default_gpu()
    };

    let cpu_mats = compute_bayes_energy_matrices(&sys_ubq, &ubq.sequence, &priors, &cfg_ubq);
    let gpu_mats = compute_bayes_energy_matrices_gpu(&sys_ubq, &ubq.sequence, &priors, &cfg_ubq)
        .expect("gpu matrices");

    assert_eq!(cpu_mats.e_type.rows, gpu_mats.e_type.rows);
    assert_eq!(cpu_mats.e_type.cols, gpu_mats.e_type.cols);
    assert_eq!(cpu_mats.e_link.rows, gpu_mats.e_link.rows);

    let mut max_err_etype = 0.0_f64;
    let mut max_err_eunary = 0.0_f64;
    for s in 0..cpu_mats.e_type.rows {
        for k in 0..cpu_mats.e_type.cols {
            let ct = cpu_mats.e_type.get(s, k);
            let gt = gpu_mats.e_type.get(s, k);
            if ct < FORBID_ENERGY * 0.5 && gt < FORBID_ENERGY * 0.5 {
                max_err_etype = max_err_etype.max((ct - gt).abs());
            } else {
                assert!(
                    ct >= FORBID_ENERGY * 0.5 && gt >= FORBID_ENERGY * 0.5,
                    "Forbidden mask mismatch at E_type({s}, {k}): cpu={ct} gpu={gt}"
                );
            }

            let cu = cpu_mats.e_unary.get(s, k);
            let gu = gpu_mats.e_unary.get(s, k);
            if cu < FORBID_ENERGY * 0.5 && gu < FORBID_ENERGY * 0.5 {
                max_err_eunary = max_err_eunary.max((cu - gu).abs());
            } else {
                assert!(
                    cu >= FORBID_ENERGY * 0.5 && gu >= FORBID_ENERGY * 0.5,
                    "Forbidden mask mismatch at E_unary({s}, {k}): cpu={cu} gpu={gu}"
                );
            }
        }
    }

    let mut max_err_elink = 0.0_f64;
    for s in 0..cpu_mats.e_link.rows {
        for t in 0..cpu_mats.e_link.cols {
            let cl = cpu_mats.e_link.get(s, t);
            let gl = gpu_mats.e_link.get(s, t);
            if cl < FORBID_ENERGY * 0.5 && gl < FORBID_ENERGY * 0.5 {
                max_err_elink = max_err_elink.max((cl - gl).abs());
            } else {
                assert!(
                    cl >= FORBID_ENERGY * 0.5 && gl >= FORBID_ENERGY * 0.5,
                    "Forbidden mask mismatch at E_link({s}, {t}): cpu={cl} gpu={gl}"
                );
            }
        }
    }

    // Generate 64 diverse valid states (sparse, medium, dense, ground-truth-like) and verify
    // Kahan-compensated WebGPU state total energy parity vs f64 CPU evaluate_state_energy.
    let tables_ubq = build_bayes_move_tables(&cpu_mats, &cfg_ubq);
    let mut rng = Rng::new(2026);
    let (mut test_states, _, _, _) =
        build_initial_bayes_replicas(&cpu_mats, &tables_ubq, 32, &mut rng);

    for d in 0..32 {
        let mut at = vec![-1i32; ubq.sequence.len()];
        let mut used = vec![false; sys_ubq.len()];
        at[lock_k] = lock_s as i32;
        used[lock_s] = true;
        let target_occ = 20 + (d * 2).min(tables_ubq.free_k.len());
        for _ in 0..(target_occ * 3) {
            let s = tables_ubq.free_s[rng.usize(tables_ubq.free_s.len())] as usize;
            let k = tables_ubq.free_k[rng.usize(tables_ubq.free_k.len())] as usize;
            if !used[s] && at[k] < 0 && cpu_mats.e_unary.get(s, k) < FORBID_ENERGY * 0.5 {
                at[k] = s as i32;
                used[s] = true;
            }
        }
        test_states.push(at);
    }

    let gpu_state_es = evaluate_states_energy_gpu(&cpu_mats, &test_states).expect("gpu state eval");
    let gpu_mat_state_es =
        evaluate_states_energy_gpu(&gpu_mats, &test_states).expect("gpu-mat state eval");
    let mut max_err_state_ubq = 0.0_f64;
    let mut max_err_persite_ubq = 0.0_f64;
    for (idx, at) in test_states.iter().enumerate() {
        let cpu_e = evaluate_state_energy(at, &cpu_mats).expect("valid ubq state");
        let n_occ = at.iter().filter(|&&s| s >= 0).count().max(1) as f64;
        max_err_state_ubq = max_err_state_ubq.max((cpu_e - gpu_state_es[idx]).abs());
        max_err_persite_ubq =
            max_err_persite_ubq.max((cpu_e - gpu_mat_state_es[idx]).abs() / n_occ);
    }

    // Part B: 480-aa large protein matrix & TwoSum df64 state total energy parity
    let ref_480 = build_synthetic_protein_reference(480, 101);
    let (peaks_480, _) = simulate(
        &ref_480,
        &SimConfig {
            seed: 101,
            p_missing_root: 0.04,
            p_missing_peak: 0.05,
            spurious_rate: 0.03,
            ..SimConfig::default()
        },
    );
    let cfg_480 = BayesSaConfig::default_gpu();
    let cond_480 = condition_peaks_preserving_hsqc_roots(&peaks_480, &cfg_480.tolerances);
    let sys_480 = build_spin_systems(&cond_480, &cfg_480.tolerances, cfg_480.max_cands, false)
        .expect("build 480aa spin systems");

    let cpu_480 = compute_bayes_energy_matrices(&sys_480, &ref_480.sequence, &priors, &cfg_480);
    let gpu_480 = gpu
        .compute_energy_matrices(&sys_480, &ref_480.sequence, &priors, &cfg_480)
        .expect("gpu 480aa matrices");

    let mut max_err_480_mat = 0.0_f64;
    for s in 0..cpu_480.e_type.rows {
        for k in 0..cpu_480.e_type.cols {
            let ct = cpu_480.e_type.get(s, k);
            let gt = gpu_480.e_type.get(s, k);
            if ct < FORBID_ENERGY * 0.5 && gt < FORBID_ENERGY * 0.5 {
                max_err_480_mat = max_err_480_mat.max((ct - gt).abs());
            }
        }
        for t in 0..cpu_480.e_link.cols {
            let cl = cpu_480.e_link.get(s, t);
            let gl = gpu_480.e_link.get(s, t);
            if cl < FORBID_ENERGY * 0.5 && gl < FORBID_ENERGY * 0.5 {
                max_err_480_mat = max_err_480_mat.max((cl - gl).abs());
            }
        }
    }

    let tables_480 = build_bayes_move_tables(&cpu_480, &cfg_480);
    let (mut states_480, _, _, _) =
        build_initial_bayes_replicas(&cpu_480, &tables_480, 16, &mut rng);
    for _ in 0..16 {
        let mut at = vec![-1i32; ref_480.sequence.len()];
        let mut used = vec![false; sys_480.len()];
        for _ in 0..900 {
            let s = tables_480.free_s[rng.usize(tables_480.free_s.len())] as usize;
            let k = tables_480.free_k[rng.usize(tables_480.free_k.len())] as usize;
            if !used[s] && at[k] < 0 && cpu_480.e_unary.get(s, k) < FORBID_ENERGY * 0.5 {
                at[k] = s as i32;
                used[s] = true;
            }
        }
        states_480.push(at);
    }
    let gpu_480_es = gpu
        .evaluate_states_energy(&cpu_480, &states_480)
        .expect("gpu 480aa state eval");
    let gpu_480_mat_es = gpu
        .evaluate_states_energy(&gpu_480, &states_480)
        .expect("gpu 480aa end-to-end state eval");
    let mut max_err_state_480 = 0.0_f64;
    let mut max_err_persite_480 = 0.0_f64;
    for (idx, at) in states_480.iter().enumerate() {
        let cpu_e = evaluate_state_energy(at, &cpu_480).expect("valid 480aa state");
        let n_occ = at.iter().filter(|&&s| s >= 0).count().max(1) as f64;
        max_err_state_480 = max_err_state_480.max((cpu_e - gpu_480_es[idx]).abs());
        max_err_persite_480 = max_err_persite_480.max((cpu_e - gpu_480_mat_es[idx]).abs() / n_occ);
    }

    eprintln!(
        "[CPU vs WebGPU parity] ubq: max|dE_type|={max_err_etype:.3e} max|dE_unary|={max_err_eunary:.3e} max|dE_link|={max_err_elink:.3e} max|dE_state|={max_err_state_ubq:.3e} max|dE_site|={max_err_persite_ubq:.3e} | 480aa: max|dMat|={max_err_480_mat:.3e} max|dE_state|={max_err_state_480:.3e} max|dE_site|={max_err_persite_480:.3e}"
    );

    assert!(
        max_err_etype < 1e-4,
        "E_type parity {max_err_etype:.3e} exceeded 1e-4"
    );
    assert!(
        max_err_eunary < 1e-4,
        "E_unary parity {max_err_eunary:.3e} exceeded 1e-4"
    );
    assert!(
        max_err_elink < 1e-4,
        "E_link parity {max_err_elink:.3e} exceeded 1e-4"
    );
    assert!(
        max_err_state_ubq < 1e-4 && max_err_persite_ubq < 1e-4,
        "Ubiquitin state energy parity ({max_err_state_ubq:.3e}, per-site {max_err_persite_ubq:.3e}) exceeded 1e-4"
    );
    assert!(
        max_err_480_mat < 1e-4,
        "480aa matrix parity {max_err_480_mat:.3e} exceeded 1e-4"
    );
    assert!(
        max_err_state_480 < 1e-4 && max_err_persite_480 < 1e-4,
        "480aa state energy parity ({max_err_state_480:.3e}, per-site {max_err_persite_480:.3e}) exceeded 1e-4"
    );
}

#[test]
fn tdr_chunking_bit_equivalence_global_kernel_and_device_buffer_limits() {
    let Some(gpu) = require_gpu_solver() else {
        return;
    };
    const {
        assert!(MAX_BAYES_GPU_CHUNK_STEPS <= 4_096);
    };

    // 1. Verify device limit guard rejects impossible matrix allocations cleanly without panicking
    let ok_800 = validate_bayes_gpu_buffer_budget(&gpu.engine.device, 800, 800, 128);
    assert!(
        ok_800.is_ok(),
        "800 aa x 128 replicas must fit cleanly within device storage buffer limits: {ok_800:?}"
    );
    let too_huge = validate_bayes_gpu_buffer_budget(&gpu.engine.device, 250_000, 250_000, 128);
    assert!(
        too_huge.is_err(),
        "250,000 x 250,000 matrix (~500 GB) must return Err against device.limits().max_buffer_size / max_storage_buffer_binding_size"
    );

    // 2. Bit-for-bit TDR multi-chunk equivalence (512 steps/chunk x 10 vs 4,096 steps/chunk x 2 over 5,000 steps)
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ubq = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::default();
    let (peaks, _) = simulate(
        &ubq,
        &SimConfig {
            seed: 4,
            ..SimConfig::realistic_bruker()
        },
    );
    let cfg = BayesSaConfig {
        replicas: 32,
        seed: 99,
        ..BayesSaConfig::default_gpu()
    };
    let cond = condition_peaks_preserving_hsqc_roots(&peaks, &cfg.tolerances);
    let systems = build_spin_systems(&cond, &cfg.tolerances, cfg.max_cands, false).expect("sys");
    let matrices = compute_bayes_energy_matrices(&systems, &ubq.sequence, &priors, &cfg);
    let tables = build_bayes_move_tables(&matrices, &cfg);
    let mut master_rng = Rng::new(cfg.seed);
    let (init_ats, init_poss, init_es, sub_seeds) =
        build_initial_bayes_replicas(&matrices, &tables, 32, &mut master_rng);

    let traj_4096 = gpu
        .run_stage_raw_with_chunk_cap(
            &matrices,
            &tables,
            &init_ats,
            &init_poss,
            &init_es,
            &sub_seeds,
            2.4,
            5_000,
            cfg.max_fragment_len,
            4_096,
            false,
        )
        .expect("4096-chunk stage");
    let traj_512 = gpu
        .run_stage_raw_with_chunk_cap(
            &matrices,
            &tables,
            &init_ats,
            &init_poss,
            &init_es,
            &sub_seeds,
            2.4,
            5_000,
            cfg.max_fragment_len,
            512,
            false,
        )
        .expect("512-chunk stage");

    assert_eq!(
        traj_4096, traj_512,
        "TDR multi-chunk splitting (512 vs 4,096 steps/submit over 5,000 steps) must be bit-for-bit identical"
    );

    // 3. Verify `anneal_bayes_global` storage-buffer kernel (>1024-aa path) preserves exact bijective
    //    `at <-> pos` invariants and matches exact f64 state energy within < 1e-2 after 2,000 steps.
    let traj_global = gpu
        .run_stage_raw_with_chunk_cap(
            &matrices,
            &tables,
            &init_ats,
            &init_poss,
            &init_es,
            &sub_seeds,
            1.8,
            2_000,
            cfg.max_fragment_len,
            1_024,
            true,
        )
        .expect("global-buffer stage");

    for r in 0..32 {
        let at = &traj_global.cur_ats[r];
        let pos = &traj_global.cur_positions[r];
        for (k, &s_raw) in at.iter().enumerate() {
            if s_raw >= 0 {
                assert_eq!(pos[s_raw as usize], k as i32);
            }
        }
        for (s, &k_raw) in pos.iter().enumerate() {
            if k_raw >= 0 {
                assert_eq!(at[k_raw as usize], s as i32);
            }
        }
        let exact_e = evaluate_state_energy(at, &matrices).expect("valid global kernel state");
        let tracked_e = traj_global.stats_float[r][0] as f64;
        assert!(
            (exact_e - tracked_e).abs() < 1e-2,
            "Global kernel replica {r} tracked energy ({tracked_e:.4}) drifted from exact ({exact_e:.4})"
        );
    }
}

#[test]
fn ubiquitin_realistic_bruker_5_seeds_webgpu_accuracy_gate() {
    let Some(gpu) = require_gpu_solver() else {
        return;
    };
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::default();
    let base_sim = SimConfig::realistic_bruker();

    let mut total_assignable = 0usize;
    let mut total_assigned = 0usize;
    let mut total_correct = 0usize;
    let mut total_hc_assigned = 0usize;
    let mut total_hc_correct = 0usize;

    for seed in 0..5u64 {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                ..base_sim.clone()
            },
        );
        let cfg = BayesSaConfig {
            seed,
            ..BayesSaConfig::default_gpu()
        };
        let (_systems, res, rep) = gpu
            .solve_from_peaks(&peaks, &ref_data.sequence, &priors, &cfg)
            .expect("gpu ubq solve");

        for &p in &res.position {
            if p >= 0 {
                assert_ne!(ref_data.sequence[p as usize], "PRO");
            }
        }
        for step in &rep.cooling_trajectory {
            assert!(step.specific_heat_cv >= 0.0 && step.specific_heat_cv.is_finite());
        }

        let m = Metrics {
            assignable: truth.iter().filter(|&&t| t >= 0).count(),
            ..score(&res, &truth, 0.80)
        };
        eprintln!(
            "[WebGPU ubiquitin realistic seed={seed}] acc={:.2}% ({}/{}) rec={:.2}% ({}/{}) hc_prec={:.2}% ({}/{}) best_E={:.2} T_crit={:.2} stages={}",
            100.0 * m.accuracy(),
            m.correct,
            m.assigned,
            100.0 * m.recall(),
            m.correct,
            m.assignable,
            100.0 * m.hc_precision(),
            m.hc_correct,
            m.hc_assigned,
            rep.best_energy,
            rep.t_crit,
            rep.total_stages
        );
        total_assignable += m.assignable;
        total_assigned += m.assigned;
        total_correct += m.correct;
        total_hc_assigned += m.hc_assigned;
        total_hc_correct += m.hc_correct;
    }

    let overall_acc = (total_correct as f64) / (total_assigned.max(1) as f64);
    let overall_rec = (total_correct as f64) / (total_assignable.max(1) as f64);
    let overall_hc_prec = if total_hc_assigned > 0 {
        (total_hc_correct as f64) / (total_hc_assigned as f64)
    } else {
        1.0
    };
    eprintln!(
        "[WebGPU ubiquitin realistic 5-seed aggregate] accuracy={:.2}% ({total_correct}/{total_assigned}) recall={:.2}% ({total_correct}/{total_assignable}) hc_precision={:.2}% ({total_hc_correct}/{total_hc_assigned})",
        100.0 * overall_acc,
        100.0 * overall_rec,
        100.0 * overall_hc_prec
    );

    assert!(
        overall_acc >= 0.93,
        "WebGPU Ubiquitin 5-seed realistic accuracy {:.2}% fell below 93% gate",
        100.0 * overall_acc
    );
    assert!(
        overall_hc_prec >= 0.96,
        "WebGPU Ubiquitin 5-seed realistic HC precision {:.2}% fell below 96% gate",
        100.0 * overall_hc_prec
    );
}

#[test]
fn large_480aa_protein_webgpu_accuracy_and_under_5s_gate() {
    let Some(gpu) = require_gpu_solver() else {
        return;
    };
    let priors = Priors::default();
    let ref_480 = build_synthetic_protein_reference(480, 101);
    assert_eq!(ref_480.sequence.len(), 480);

    let (peaks, truth) = simulate(
        &ref_480,
        &SimConfig {
            seed: 101,
            p_missing_root: 0.04,
            p_missing_peak: 0.05,
            spurious_rate: 0.03,
            ..SimConfig::default()
        },
    );

    let cfg = BayesSaConfig {
        seed: 101,
        ..BayesSaConfig::default_gpu()
    };

    let t0 = Instant::now();
    let (_sys, res, rep) = gpu
        .solve_from_peaks(&peaks, &ref_480.sequence, &priors, &cfg)
        .expect("WebGPU 480aa solve");
    let elapsed_s = t0.elapsed().as_secs_f64();

    for &p in &res.position {
        if p >= 0 {
            assert_ne!(ref_480.sequence[p as usize], "PRO");
        }
    }
    let m = Metrics {
        assignable: truth.iter().filter(|&&t| t >= 0).count(),
        ..score(&res, &truth, 0.80)
    };
    eprintln!(
        "[WebGPU 480aa R={}] accuracy={:.2}% ({}/{}) recall={:.2}% ({}/{}) hc_prec={:.2}% ({}/{}) elapsed={elapsed_s:.3}s T_crit={:.2} max_Cv={:.2} stages={}",
        cfg.replicas,
        100.0 * m.accuracy(),
        m.correct,
        m.assigned,
        100.0 * m.recall(),
        m.correct,
        m.assignable,
        100.0 * m.hc_precision(),
        m.hc_correct,
        m.hc_assigned,
        rep.t_crit,
        rep.max_cv,
        rep.total_stages
    );

    assert!(
        m.accuracy() >= 0.88,
        "WebGPU 480aa accuracy {:.2}% fell below 88% gate",
        100.0 * m.accuracy()
    );
    assert!(
        elapsed_s < 5.0,
        "WebGPU 480aa runtime {elapsed_s:.3}s exceeded 5.0s gate"
    );
}
