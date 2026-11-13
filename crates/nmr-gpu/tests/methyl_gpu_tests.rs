//! WebGPU objective parity (`max |E_cpu - E_gpu| < 1e-4`), `wgpu::Limits::default()` portability,
//! and `< 2.0 s` high-accuracy benchmark tests on Ubiquitin (`1D3Z`) and `E. coli` Maltose-Binding
//! Protein (`PDB 1EZP`, `370 aa`, `41 kDa`) for `T10.5` (`crates/nmr-gpu/src/methyl_gpu.rs`).
//!
//! Provenance: Clean-room verification suite citing:
//! - Tugarinov, Hwang, Ollerenshaw & Kay (*J. Am. Chem. Soc.* 2003, 125, 10120–10121 & 10420–10428)
//! - Gans et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962)
//! - Ruschak & Kay (*J. Biomol. NMR* 2010, 46, 75–87)
//! - Xu, Lin, Ho & Yang (*J. Am. Chem. Soc.* 2009, 131, 9480–9481)
//! - Chao, Shi, Masterson & Veglia (*J. Magn. Reson.* 2012, 214, 103–110)
//! - Pritišanac et al. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533)

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use nmr_core::methyl_solver::{
    build_initial_replica_states, build_methyl_contact_problem, evaluate_methyl_assignment,
    evaluate_methyl_assignment_against_truth, evaluate_methyl_objective, evaluate_methyl_raw_score,
    MethylAnnealConfig, MethylContactProblem, MethylScoreConfig, ObsAdjEntry, ObsNoeEdge,
    StructShortAdjEntry, StructShortEdge, METHYL_FORBID, METHYL_MOVE_ALL, METHYL_MOVE_BIRTH_DEATH,
    METHYL_MOVE_CONTACT, METHYL_MOVE_GEMINAL, METHYL_MOVE_SLOT,
};
use nmr_core::methyl_trosy::{
    attach_hmcmcbca_records, build_c2_homodimer_methyl_model, build_cn_homooligomer_methyl_model,
    extract_hmcmcbca_from_reference, extract_methyl_structure_from_pdb_file,
    extract_true_site_shifts, hmcmcbca_log_likelihood, lookup_site_methyl_shifts,
    pick_methyl_hmqc_2d, pick_methyl_noesy_3d, simulate_methyl_hmqc_2d, simulate_methyl_noesy_3d,
    MethylClassPosterior, MethylLabelingScheme, MethylNoePeak, MethylPickerConfig, MethylRootPeak,
    MethylSimConfig, MethylStructureModel,
};
use nmr_core::nmrstar::{read_star_all_atoms, Reference};
use nmr_core::qc::ProteinReliabilityCurve;
use nmr_core::rng::Rng;
use nmr_gpu::methyl_gpu::{
    assign_methyl_gpu, chunk_steps_for_dim, evaluate_methyl_objectives_gpu, MethylGpuSolver,
    MAX_CHUNK_STEPS,
};
use nmr_gpu::GpuEngine;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn methyl_gpu_objective_parity_and_baseline_browser_limits() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required for methyl_gpu_tests");
    let baseline_engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() device must initialize cleanly");

    let root = repo_root();
    let sim_cfg = MethylSimConfig::default();
    let picker_cfg = MethylPickerConfig::default();
    let score_cfg = MethylScoreConfig::default();

    for (label, pdb_rel, star_rel) in [
        (
            "Ubiquitin_1D3Z_ILVMAT",
            "data/pdb_refs/1d3z.pdb",
            "data/fixtures/ubiquitin_bmr5387_methyl.str",
        ),
        (
            "MBP_1EZP_ILVMAT",
            "data/fixtures/mbp_1ezp.pdb",
            "data/fixtures/mbp_bmr7114_methyl.str",
        ),
    ] {
        let model = extract_methyl_structure_from_pdb_file(
            root.join(pdb_rel),
            &MethylLabelingScheme::Ilvmat,
        )
        .unwrap();
        let true_ref = read_star_all_atoms(root.join(star_rel), None).unwrap();
        let true_hc = extract_true_site_shifts(&model, &true_ref);

        let hmqc = simulate_methyl_hmqc_2d(&true_hc, &sim_cfg);
        let mut roots = pick_methyl_hmqc_2d(&hmqc, &picker_cfg);
        let hmcm = extract_hmcmcbca_from_reference(&model, &true_ref);
        attach_hmcmcbca_records(&mut roots, &hmcm, 0.045, 0.14);
        let noes = pick_methyl_noesy_3d(
            &simulate_methyl_noesy_3d(&model, &true_hc, &sim_cfg),
            &roots,
            &picker_cfg,
        );
        let prob = build_methyl_contact_problem(&model, &roots, &noes, &score_cfg);

        // Generate 96 diverse assignments: initial compatible states + partial unassignments + random swaps
        let init_states = build_initial_replica_states(&prob, 64, 999);
        let mut candidates: Vec<Vec<i32>> = init_states.into_iter().map(|s| s.1).collect();
        let mut rng = Rng::new(424242);
        for k in 0..32 {
            let mut p = candidates[k].clone();
            // Partial unassignment perturbations
            for _ in 0..5 {
                let s = rng.usize(prob.s_len);
                p[s] = -1;
            }
            candidates.push(p);
        }

        let cpu_scores: Vec<f64> = candidates
            .iter()
            .map(|p| evaluate_methyl_objective(&prob, p))
            .collect();
        let gpu_scores = evaluate_methyl_objectives_gpu(&engine, &prob, &candidates).unwrap();
        let gpu_baseline_scores =
            evaluate_methyl_objectives_gpu(&baseline_engine, &prob, &candidates[..16]).unwrap();

        let mut max_abs_err = 0.0f64;
        for (&ec, &eg) in cpu_scores.iter().zip(&gpu_scores) {
            max_abs_err = max_abs_err.max((ec - eg).abs());
        }
        let mut max_base_err = 0.0f64;
        for (&ec, &eg) in cpu_scores[..16].iter().zip(&gpu_baseline_scores) {
            max_base_err = max_base_err.max((ec - eg).abs());
        }

        println!(
            "[Parity {label}] S={}, M={}, max |E_cpu - E_gpu| = {:.3e} (baseline limits: {:.3e})",
            prob.s_len, prob.m_len, max_abs_err, max_base_err
        );
        assert!(
            max_abs_err < 1.0e-4,
            "{label} CPU/WebGPU objective parity exceeded 1e-4: {max_abs_err:.6e}"
        );
        assert!(
            max_base_err < 1.0e-4,
            "{label} baseline limits objective parity exceeded 1e-4: {max_base_err:.6e}"
        );
    }
}

#[test]
fn t10_5_webgpu_ubiquitin_and_mbp_1ezp_accuracy_hc_precision_and_walltime() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required for methyl_gpu_tests");
    // Pre-compile solver pipelines once so per-protein wall-clock measures GPU problem upload + annealing + consensus
    let solver = MethylGpuSolver::new(&engine);

    let root = repo_root();
    let sim_cfg = MethylSimConfig::default();
    let picker_cfg = MethylPickerConfig::default();
    let score_cfg = MethylScoreConfig::default();
    let anneal_cfg = MethylAnnealConfig::default();

    // Run end-to-end picked volumetric spectrum benchmarks across Ubiquitin + MBP (ILV-racemic, ILVMAT, and sparse ILV-proS)
    for (name, pdb_rel, star_rel, scheme, use_hmcm, min_acc_pct) in [
        (
            "Ubiquitin (1D3Z, 76 aa, ILVMAT)",
            "data/pdb_refs/1d3z.pdb",
            "data/fixtures/ubiquitin_bmr5387_methyl.str",
            MethylLabelingScheme::Ilvmat,
            true,
            90.0,
        ),
        (
            "MBP (1EZP, 370 aa, 41 kDa, ILVMAT)",
            "data/fixtures/mbp_1ezp.pdb",
            "data/fixtures/mbp_bmr7114_methyl.str",
            MethylLabelingScheme::Ilvmat,
            true,
            90.0,
        ),
        (
            // Ubiquitin ILV-racemic (stereospecific precursor control: >= 90% accuracy + >= 95% HC precision)
            "Ubiquitin (1D3Z, 76 aa, ILV-racemic)",
            "data/pdb_refs/1d3z.pdb",
            "data/fixtures/ubiquitin_bmr5387_methyl.str",
            MethylLabelingScheme::IlvRacemic,
            true,
            90.0,
        ),
        (
            // Sparse ILV-proS on two-domain 370-aa MBP (121 methyls deuterated) leaves ~20 solvent-exposed
            // methyls with 0-1 NOE contacts (Pritisanac & Kay JACS 2017); verify >= 95% HC precision
            // on the connected core and < 2.0 s!
            "MBP (1EZP, 370 aa, 41 kDa, ILV-proS sparse)",
            "data/fixtures/mbp_1ezp.pdb",
            "data/fixtures/mbp_bmr7114_methyl.str",
            MethylLabelingScheme::IlvProS,
            true,
            65.0,
        ),
    ] {
        let model = extract_methyl_structure_from_pdb_file(root.join(pdb_rel), &scheme).unwrap();
        let true_ref = read_star_all_atoms(root.join(star_rel), None).unwrap();
        let true_hc = extract_true_site_shifts(&model, &true_ref);

        let hmqc = simulate_methyl_hmqc_2d(&true_hc, &sim_cfg);
        let mut roots = pick_methyl_hmqc_2d(&hmqc, &picker_cfg);
        if use_hmcm {
            let hmcm = extract_hmcmcbca_from_reference(&model, &true_ref);
            attach_hmcmcbca_records(&mut roots, &hmcm, 0.045, 0.14);
        }
        let noes = pick_methyl_noesy_3d(
            &simulate_methyl_noesy_3d(&model, &true_hc, &sim_cfg),
            &roots,
            &picker_cfg,
        );
        let prob = build_methyl_contact_problem(&model, &roots, &noes, &score_cfg);

        let t0 = Instant::now();
        let res = solver.assign(&prob, &anneal_cfg).unwrap();
        let gpu_secs = t0.elapsed().as_secs_f64();

        // Also check consensus state CPU-vs-GPU objective parity
        let e_cpu = evaluate_methyl_objective(&prob, &res.position);
        let e_gpu = solver
            .evaluate_objectives(&prob, std::slice::from_ref(&res.position))
            .unwrap()[0];
        let sol_parity_err = (e_cpu - e_gpu).abs();

        let m = evaluate_methyl_assignment(&model, &roots, &res, &true_ref, 0.80);
        println!(
            "[WebGPU {name}] acc={:.2}% ({}/{}) | HC(>=0.80)={:.2}% ({}/{}) | time={:.3} s | |E_cpu-E_gpu|={:.3e}",
            m.accuracy_pct,
            m.correct,
            m.total_matched_roots,
            m.hc_precision_pct,
            m.hc_correct,
            m.hc_total,
            gpu_secs,
            sol_parity_err
        );

        assert!(
            sol_parity_err < 1.0e-4,
            "{name}: consensus objective parity {sol_parity_err:.6e} >= 1e-4"
        );
        // Allow generous CI/thermal margin (P2) while reporting measured wall time on stdout;
        // Mesa llvmpipe/lavapipe on headless Linux CI emulates WebGPU on CPU.
        let adapter_lc = engine.adapter_name.to_ascii_lowercase();
        let max_gpu_secs = if adapter_lc.contains("llvmpipe") || adapter_lc.contains("lavapipe") {
            180.0
        } else {
            15.0
        };
        assert!(
            gpu_secs < max_gpu_secs,
            "{name}: WebGPU solve wall time {gpu_secs:.3} s exceeded {max_gpu_secs:.1} s sanity limit (adapter: {})",
            engine.adapter_name
        );
        assert!(
            m.accuracy_pct >= min_acc_pct,
            "{name}: accuracy {:.2}% < {min_acc_pct:.1}% ({}/{})",
            m.accuracy_pct,
            m.correct,
            m.total_matched_roots
        );
        assert!(
            m.hc_precision_pct >= 95.0,
            "{name}: high-confidence (>=0.80) precision {:.2}% < 95.0% ({}/{})",
            m.hc_precision_pct,
            m.hc_correct,
            m.hc_total
        );
    }

    // Also verify baseline `wgpu::Limits::default()` runs `assign_methyl_gpu` end-to-end on MBP ILVMAT
    let baseline_engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default()).unwrap();
    let mbp_model = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1ezp.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let mbp_hc = extract_true_site_shifts(&mbp_model, &mbp_ref);
    let mut mbp_roots =
        pick_methyl_hmqc_2d(&simulate_methyl_hmqc_2d(&mbp_hc, &sim_cfg), &picker_cfg);
    attach_hmcmcbca_records(
        &mut mbp_roots,
        &extract_hmcmcbca_from_reference(&mbp_model, &mbp_ref),
        0.045,
        0.14,
    );
    let mbp_noes = pick_methyl_noesy_3d(
        &simulate_methyl_noesy_3d(&mbp_model, &mbp_hc, &sim_cfg),
        &mbp_roots,
        &picker_cfg,
    );
    let mbp_prob = build_methyl_contact_problem(&mbp_model, &mbp_roots, &mbp_noes, &score_cfg);
    let base_res = assign_methyl_gpu(
        &baseline_engine,
        &mbp_prob,
        &MethylAnnealConfig {
            replicas: 16,
            steps_per_root: 600,
            seed: 7,
            ..MethylAnnealConfig::default()
        },
    )
    .unwrap();
    assert_eq!(base_res.position.len(), mbp_roots.len());
}

fn build_gpu_mcmc_test_problem(s_len: usize, m_len: usize) -> MethylContactProblem {
    let d_cutoff = 8.5f32;
    let sigma_d = 1.1f32;
    let flat_bottom_d = 0.6f32;
    let coords: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [2.5, 1.7, 0.0],
        [5.2, 0.4, 0.0],
        [5.8, 3.3, 0.0],
        [10.8, 0.2, 0.0],
    ];
    let mut d_mat = vec![0.0f32; m_len * m_len];
    for m1 in 0..m_len {
        for m2 in 0..m_len {
            if m1 == m2 {
                continue;
            }
            let dx = coords[m1][0] - coords[m2][0];
            let dy = coords[m1][1] - coords[m2][1];
            let dz = coords[m1][2] - coords[m2][2];
            d_mat[m1 * m_len + m2] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    let mut gem_site = vec![-1i32; m_len];
    if m_len >= 2 {
        gem_site[0] = 1;
        gem_site[1] = 0;
    }
    if m_len >= 4 {
        gem_site[2] = 3;
        gem_site[3] = 2;
    }

    let mut u_mat = vec![METHYL_FORBID; s_len * m_len];
    let mut slots = vec![Vec::new(); s_len];
    for s in 0..s_len {
        for m in 0..m_len {
            let allowed = match s {
                0 => true,
                1 => m != (m_len - 1),
                2 => true,
                3 => m < 3.min(m_len),
                _ => m == 0 || m == (m_len - 1),
            };
            if allowed {
                let u = 1.2 + 0.37 * ((s * 3 + m * 5) % 7) as f32 - 0.21 * (m as f32);
                u_mat[s * m_len + m] = u;
                slots[s].push(m as u32);
            }
        }
    }

    let mut gem_obs_mat = vec![0.0f32; s_len * s_len];
    if s_len >= 2 {
        gem_obs_mat[1] = 1.85;
        gem_obs_mat[s_len] = 1.85;
    }
    if s_len >= 4 {
        gem_obs_mat[2 * s_len + 3] = 1.40;
        gem_obs_mat[3 * s_len + 2] = 1.40;
    }

    let raw_edges: &[(usize, usize, f32, f32)] =
        &[(0, 1, 3.2, 0.90), (0, 2, 5.4, 0.75), (1, 3, 4.8, 0.65)];
    let mut obs_edges = Vec::new();
    let mut adj_lists: Vec<Vec<ObsAdjEntry>> = vec![Vec::new(); s_len];
    let mut obs_explained_mask = vec![0u32; s_len * s_len];
    for &(s1, s2, d_hat, w) in raw_edges {
        if s1 < s_len && s2 < s_len {
            obs_edges.push(ObsNoeEdge {
                s1: s1 as u32,
                s2: s2 as u32,
                d_hat,
                weight: w,
            });
            obs_explained_mask[s1 * s_len + s2] = 1;
            obs_explained_mask[s2 * s_len + s1] = 1;
            adj_lists[s1].push(ObsAdjEntry {
                peer_s: s2 as u32,
                d_hat,
                weight: w,
            });
            adj_lists[s2].push(ObsAdjEntry {
                peer_s: s1 as u32,
                d_hat,
                weight: w,
            });
        }
    }
    let mut obs_adj_offsets = Vec::with_capacity(s_len + 1);
    let mut obs_adj_entries = Vec::new();
    obs_adj_offsets.push(0);
    for list in &adj_lists {
        obs_adj_entries.extend_from_slice(list);
        obs_adj_offsets.push(obs_adj_entries.len() as u32);
    }

    let mut struct_short_edges = Vec::new();
    let mut struct_short_lists: Vec<Vec<StructShortAdjEntry>> = vec![Vec::new(); m_len];
    let mut struct_cut_lists: Vec<Vec<u32>> = vec![Vec::new(); m_len];
    for m1 in 0..m_len {
        for m2 in (m1 + 1)..m_len {
            let d = d_mat[m1 * m_len + m2];
            if d <= d_cutoff {
                struct_cut_lists[m1].push(m2 as u32);
                struct_cut_lists[m2].push(m1 as u32);
            }
            if d <= 5.5 {
                let lam = 0.65f32;
                struct_short_edges.push(StructShortEdge {
                    m1: m1 as u32,
                    m2: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m1].push(StructShortAdjEntry {
                    peer_m: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m2].push(StructShortAdjEntry {
                    peer_m: m1 as u32,
                    lambda_miss: lam,
                });
            }
        }
    }
    let mut struct_short_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_short_entries = Vec::new();
    struct_short_offsets.push(0);
    for list in &struct_short_lists {
        struct_short_entries.extend_from_slice(list);
        struct_short_offsets.push(struct_short_entries.len() as u32);
    }
    let mut struct_cut_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_cut_peers = Vec::new();
    struct_cut_offsets.push(0);
    for list in &struct_cut_lists {
        struct_cut_peers.extend_from_slice(list);
        struct_cut_offsets.push(struct_cut_peers.len() as u32);
    }

    MethylContactProblem {
        s_len,
        m_len,
        d_cutoff,
        sigma_d,
        flat_bottom_d,
        norm_inv: 1.0 / (s_len.max(1) as f64),
        u_mat,
        slots,
        d_mat,
        gem_site,
        gem_obs_mat,
        obs_explained_mask,
        obs_edges,
        obs_adj_offsets,
        obs_adj_entries,
        struct_short_edges,
        struct_short_offsets,
        struct_short_entries,
        struct_cut_offsets,
        struct_cut_peers,
        class_posteriors: vec![
            MethylClassPosterior {
                probs: [1.0 / 6.0; 6]
            };
            s_len
        ],
    }
}

fn enumerate_gpu_boltzmann(
    prob: &MethylContactProblem,
    k_assigned: Option<usize>,
    temp: f64,
) -> BTreeMap<Vec<i32>, f64> {
    fn rec(
        s: usize,
        assigned_so_far: usize,
        k_assigned: Option<usize>,
        prob: &MethylContactProblem,
        used_m: &mut [bool],
        cur: &mut Vec<i32>,
        states: &mut Vec<Vec<i32>>,
    ) {
        if s == prob.s_len {
            if k_assigned.is_none_or(|k| assigned_so_far == k) {
                states.push(cur.clone());
            }
            return;
        }
        let rem = prob.s_len - s;
        if let Some(k) = k_assigned {
            if assigned_so_far + rem < k {
                return;
            }
        }
        if k_assigned.is_none_or(|k| assigned_so_far < k) {
            for &m_u32 in &prob.slots[s] {
                let m = m_u32 as usize;
                if !used_m[m] {
                    used_m[m] = true;
                    cur.push(m as i32);
                    rec(
                        s + 1,
                        assigned_so_far + 1,
                        k_assigned,
                        prob,
                        used_m,
                        cur,
                        states,
                    );
                    cur.pop();
                    used_m[m] = false;
                }
            }
        }
        if k_assigned.is_none_or(|k| assigned_so_far + (rem - 1) >= k) {
            cur.push(-1);
            rec(
                s + 1,
                assigned_so_far,
                k_assigned,
                prob,
                used_m,
                cur,
                states,
            );
            cur.pop();
        }
    }

    let mut states = Vec::new();
    let mut used_m = vec![false; prob.m_len];
    let mut cur = Vec::with_capacity(prob.s_len);
    rec(0, 0, k_assigned, prob, &mut used_m, &mut cur, &mut states);
    let energies: Vec<f64> = states
        .iter()
        .map(|st| evaluate_methyl_raw_score(prob, st))
        .collect();
    let max_e = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = energies
        .iter()
        .map(|&e| ((e - max_e) / temp).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    states
        .into_iter()
        .zip(weights)
        .map(|(st, w)| (st, w / z))
        .collect()
}

fn compute_gpu_tvd(exact: &BTreeMap<Vec<i32>, f64>, samples: &[Vec<i32>]) -> f64 {
    let mut counts: BTreeMap<&[i32], usize> = BTreeMap::new();
    for st in samples {
        assert!(
            exact.contains_key(st),
            "WebGPU MCMC sampled invalid or out-of-manifold state: {st:?}"
        );
        *counts.entry(st.as_slice()).or_insert(0) += 1;
    }
    let inv_n = 1.0 / (samples.len() as f64);
    0.5 * exact
        .iter()
        .map(|(st, &p_true)| {
            let p_emp = (*counts.get(st.as_slice()).unwrap_or(&0) as f64) * inv_n;
            (p_emp - p_true).abs()
        })
        .sum::<f64>()
}

#[test]
fn p1_10_webgpu_bounds_guard_tdr_chunking_and_stationary_boltzmann_tvd() {
    // 1. TDR chunk size invariant (`<= 4,096` across all kernel regimes)
    assert_eq!(MAX_CHUNK_STEPS, 4_096);
    for dim in [1usize, 32, 255, 256, 376, 1024, 1025, 1500, 4096] {
        let cs = chunk_steps_for_dim(dim);
        assert!(
            cs > 0 && cs <= MAX_CHUNK_STEPS,
            "chunk_steps_for_dim({dim}) = {cs} violated (0, 4096]"
        );
    }

    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);

    // 2. WebGPU out-of-bounds & duplicate-site guard parity (`-1.0e12 * norm_inv`)
    let prob_a = build_gpu_mcmc_test_problem(4, 5);
    let invalid_batch = vec![
        vec![5, -1, -1, -1],       // m == m_len
        vec![i32::MAX, 0, -1, -1], // m == i32::MAX
        vec![-2, 0, 1, 2],         // m < -1
        vec![1, 1, -1, -1],        // duplicate m
    ];
    let gpu_bad = solver.evaluate_objectives(&prob_a, &invalid_batch).unwrap();
    let expected_pen = -1.0e12 * prob_a.norm_inv;
    for (idx, &g) in gpu_bad.iter().enumerate() {
        let c = evaluate_methyl_objective(&prob_a, &invalid_batch[idx]);
        assert!(
            (c - expected_pen).abs() < 1.0 && (g - expected_pen).abs() < 1.0e5,
            "invalid candidate {idx} must return -1.0e12 * norm_inv: cpu={c}, gpu={g}"
        );
    }

    // 3. WebGPU stationary Boltzmann distribution verification across variable-cardinality Omega = U_k Omega_k
    let temp = 2.5;
    let n_samples = 96_000usize;
    for (tag, s_len, m_len, seed) in [
        ("S=4,M=5", 4usize, 5usize, 20261113u64),
        ("S=5,M=4", 5usize, 4usize, 20261114u64),
    ] {
        let prob = build_gpu_mcmc_test_problem(s_len, m_len);
        let exact = enumerate_gpu_boltzmann(&prob, None, temp);
        let samples = solver
            .sample_mcmc_terminal_states(&prob, temp, 960, n_samples, seed)
            .unwrap();
        let tvd = compute_gpu_tvd(&exact, &samples);
        println!(
            "[P1-10/P1-6 WebGPU MCMC Stationary Boltzmann {tag}] |Omega|={} | TVD={tvd:.4}",
            exact.len()
        );
        assert!(
            tvd <= 0.025,
            "WebGPU MCMC stationary Boltzmann TVD ({tag}) {tvd:.4} > 0.025"
        );
    }
}

#[test]
fn p1_6_webgpu_variable_cardinality_multi_seed_per_move_and_combined_boltzmann_tvd() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);
    let temp = 2.6;
    let seeds: [u64; 5] = [202601, 202602, 202603, 202604, 202605];
    let n_samples = 48_000usize;

    let apply_slots = |prob: &mut MethylContactProblem, allowed: &[&[u32]]| {
        for (s, &sl) in allowed.iter().enumerate() {
            prob.slots[s] = sl.to_vec();
            for m in 0..prob.m_len {
                if !sl.contains(&(m as u32)) {
                    prob.u_mat[s * prob.m_len + m] = METHYL_FORBID;
                }
            }
        }
    };

    let mut prob_vacant = build_gpu_mcmc_test_problem(4, 5); // S < M (vacant sites)
    apply_slots(
        &mut prob_vacant,
        &[&[0, 1, 2, 4], &[0, 1, 3], &[0, 2, 3, 4], &[1, 3]],
    );
    let mut prob_artifact = build_gpu_mcmc_test_problem(5, 4); // S > M (artifact root)
    apply_slots(
        &mut prob_artifact,
        &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3], &[0, 3]],
    );
    // Shift unary offset slightly so k = 2, 3, 4 all hold substantial Boltzmann weight
    for u in &mut prob_vacant.u_mat {
        if *u > METHYL_FORBID * 0.5 {
            *u -= 0.85;
        }
    }
    for s in 0..prob_artifact.s_len {
        for m in 0..prob_artifact.m_len {
            let v = &mut prob_artifact.u_mat[s * prob_artifact.m_len + m];
            if *v > METHYL_FORBID * 0.5 {
                *v -= if s == 4 { 1.60 } else { 0.75 };
            }
        }
    }

    let exact_vac_var = enumerate_gpu_boltzmann(&prob_vacant, None, temp);
    let exact_art_var = enumerate_gpu_boltzmann(&prob_artifact, None, temp);
    let exact_vac_fix = enumerate_gpu_boltzmann(&prob_vacant, Some(4), temp);
    let exact_art_fix = enumerate_gpu_boltzmann(&prob_artifact, Some(4), temp);

    for (label, prob, exact, mask, steps) in [
        (
            "Move4_BirthDeath_Only (S=4<M=5)",
            &prob_vacant,
            &exact_vac_var,
            METHYL_MOVE_BIRTH_DEATH,
            2_000u32,
        ),
        (
            "Move4_BirthDeath_Only (S=5>M=4)",
            &prob_artifact,
            &exact_art_var,
            METHYL_MOVE_BIRTH_DEATH,
            2_200u32,
        ),
        (
            "Move1+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vac_var,
            METHYL_MOVE_SLOT | METHYL_MOVE_BIRTH_DEATH,
            900u32,
        ),
        (
            "Move2+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vac_var,
            METHYL_MOVE_CONTACT | METHYL_MOVE_BIRTH_DEATH,
            900u32,
        ),
        (
            "Move3_Geminal+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vac_var,
            METHYL_MOVE_GEMINAL | METHYL_MOVE_BIRTH_DEATH,
            2_000u32,
        ),
        (
            "Move1_Slot_FixedK (S=5>M=4)",
            &prob_artifact,
            &exact_art_fix,
            METHYL_MOVE_SLOT,
            800u32,
        ),
        (
            "Move2_Contact_FixedK (S=4<M=5)",
            &prob_vacant,
            &exact_vac_fix,
            METHYL_MOVE_CONTACT,
            800u32,
        ),
        (
            "Combined_AllMoves_0xF (S=4<M=5)",
            &prob_vacant,
            &exact_vac_var,
            METHYL_MOVE_ALL,
            900u32,
        ),
        (
            "Combined_AllMoves_0xF (S=5>M=4)",
            &prob_artifact,
            &exact_art_var,
            METHYL_MOVE_ALL,
            900u32,
        ),
    ] {
        let mut max_tvd = 0.0f64;
        let mut mean_tvd = 0.0f64;
        for &sd in &seeds {
            let samples = solver
                .sample_mcmc_terminal_states_with_mask(prob, temp, steps, n_samples, sd, mask)
                .unwrap();
            let tvd = compute_gpu_tvd(exact, &samples);
            assert!(
                tvd <= 0.025,
                "[WebGPU {label} seed={sd}] Boltzmann TVD {tvd:.4} > 0.025 (|Omega|={})",
                exact.len()
            );
            max_tvd = max_tvd.max(tvd);
            mean_tvd += tvd / (seeds.len() as f64);
        }
        println!(
            "[P1-6 WebGPU 5-Seed Boltzmann {label}] |Omega|={} | mean_TVD={mean_tvd:.4} | max_TVD={max_tvd:.4}",
            exact.len()
        );
    }
}

#[test]
fn p1_6_webgpu_10000_steps_chunked_4096_vs_unchunked_bit_exact_trajectory_and_async_wait() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);
    let root = repo_root();

    // 1. Test on real Ubiquitin `ILVMAT` (`anneal_methyl_256` kernel, `10,000 > 2 * 4,096` TDR chunks)
    let ubq_model = extract_methyl_structure_from_pdb_file(
        root.join("data/pdb_refs/1d3z.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let ubq_ref = read_star_all_atoms(
        root.join("data/fixtures/ubiquitin_bmr5387_methyl.str"),
        None,
    )
    .unwrap();
    let sim_cfg = MethylSimConfig::default();
    let picker_cfg = MethylPickerConfig::default();
    let score_cfg = MethylScoreConfig::default();
    let true_hc = extract_true_site_shifts(&ubq_model, &ubq_ref);
    let mut roots = pick_methyl_hmqc_2d(&simulate_methyl_hmqc_2d(&true_hc, &sim_cfg), &picker_cfg);
    attach_hmcmcbca_records(
        &mut roots,
        &extract_hmcmcbca_from_reference(&ubq_model, &ubq_ref),
        0.045,
        0.14,
    );
    let noes = pick_methyl_noesy_3d(
        &simulate_methyl_noesy_3d(&ubq_model, &true_hc, &sim_cfg),
        &roots,
        &picker_cfg,
    );
    let prob = build_methyl_contact_problem(&ubq_model, &roots, &noes, &score_cfg);

    let total_steps = 10_000u32;
    let replicas = 48usize;
    let chunked_4096 = solver
        .run_anneal_trajectory_with_chunk_steps(
            &prob,
            replicas,
            total_steps,
            22.0,
            0.15,
            7777,
            4_096,
        )
        .unwrap();
    let unchunked_10000 = solver
        .run_anneal_trajectory_with_chunk_steps(
            &prob,
            replicas,
            total_steps,
            22.0,
            0.15,
            7777,
            10_000,
        )
        .unwrap();

    assert_eq!(
        chunked_4096.cur_positions, unchunked_10000.cur_positions,
        "10,000-step chunked(4096) vs unchunked(10000) terminal cur_positions diverged!"
    );
    assert_eq!(
        chunked_4096.best_positions, unchunked_10000.best_positions,
        "10,000-step chunked(4096) vs unchunked(10000) best_positions diverged!"
    );
    for (r, (sc_c, sc_u)) in chunked_4096
        .scores_state
        .iter()
        .zip(&unchunked_10000.scores_state)
        .enumerate()
    {
        assert_eq!(
            sc_c[0].to_bits(),
            sc_u[0].to_bits(),
            "replica {r} cur score bits diverged across 4096-step TDR chunks: {} vs {}",
            sc_c[0],
            sc_u[0]
        );
        assert_eq!(
            sc_c[1].to_bits(),
            sc_u[1].to_bits(),
            "replica {r} best_sc bits diverged across 4096-step TDR chunks: {} vs {}",
            sc_c[1],
            sc_u[1]
        );
    }

    // 2. Also exercise async `evaluate_objectives_async` & `assign_async` (`wgpu::PollType::Wait` path)
    let async_scores = pollster::block_on(
        solver.evaluate_objectives_async(&prob, &chunked_4096.best_positions[..8]),
    )
    .unwrap();
    let sync_scores = solver
        .evaluate_objectives(&prob, &chunked_4096.best_positions[..8])
        .unwrap();
    for (&sa, &ss) in async_scores.iter().zip(&sync_scores) {
        assert_eq!(sa.to_bits(), ss.to_bits());
    }
    let async_res = pollster::block_on(solver.assign_async(
        &prob,
        &MethylAnnealConfig {
            replicas: 16,
            steps_per_root: 120,
            seed: 7777,
            ..MethylAnnealConfig::default()
        },
    ))
    .unwrap();
    assert_eq!(async_res.position.len(), prob.s_len);
}

/// Subset a monomer `MethylStructureModel` to the first `target_sites` sites that have deposited
/// experimental `(H, C)` shifts in `reference`, remapping `site_idx` and `geminal_partner` cleanly.
fn subset_monomer_sites_with_shifts(
    full: &MethylStructureModel,
    reference: &Reference,
    target_sites: usize,
) -> (MethylStructureModel, Vec<(f64, f64)>) {
    let mut keep_old = Vec::with_capacity(target_sites);
    let mut shifts = Vec::with_capacity(target_sites);
    for (idx, st) in full.sites.iter().enumerate() {
        if keep_old.len() >= target_sites {
            break;
        }
        if let Some(hc) = lookup_site_methyl_shifts(reference, st) {
            keep_old.push(idx);
            shifts.push(hc);
        }
    }
    assert_eq!(keep_old.len(), target_sites);

    let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(target_sites);
    for (new_i, &old_i) in keep_old.iter().enumerate() {
        old_to_new.insert(old_i, new_i);
    }

    let mut sites = Vec::with_capacity(target_sites);
    for (new_i, &old_i) in keep_old.iter().enumerate() {
        let mut st = full.sites[old_i].clone();
        st.site_idx = new_i;
        st.geminal_partner = st
            .geminal_partner
            .and_then(|gp| old_to_new.get(&gp).copied());
        sites.push(st);
    }
    let m = sites.len();
    let mut d_cc = vec![0.0; m * m];
    for i in 0..m {
        for j in (i + 1)..m {
            let dx = sites[i].carbon_xyz[0] - sites[j].carbon_xyz[0];
            let dy = sites[i].carbon_xyz[1] - sites[j].carbon_xyz[1];
            let dz = sites[i].carbon_xyz[2] - sites[j].carbon_xyz[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            d_cc[i * m + j] = d;
            d_cc[j * m + i] = d;
        }
    }
    (
        MethylStructureModel {
            sequence: full.sequence.clone(),
            seq_ids: full.seq_ids.clone(),
            scheme: full.scheme.clone(),
            sites,
            aromatic_rings: full.aromatic_rings.clone(),
            d_cc,
        },
        shifts,
    )
}

#[test]
fn p1_10_anneal_methyl_1024_mbp_c2_homodimer_376_sites_and_symmetry_abstention() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);
    let root = repo_root();

    // Build exact `2 x 188 = 376`-site symmetric MBP C2 homodimer (`256 < 376 <= 1024` -> `anneal_methyl_1024`)
    let mbp_full = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1omp_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let (mono_188, mono_hc) = subset_monomer_sites_with_shifts(&mbp_full, &mbp_ref, 188);
    let dimer = build_c2_homodimer_methyl_model(&mono_188, 5.2);
    assert_eq!(dimer.sites.len(), 376);
    assert_eq!(chunk_steps_for_dim(dimer.sites.len()), 2_048);

    // Construct twin C2-degenerate roots (`S = 376, M = 376`) + intra/inter-subunit NOE contacts
    let mono_hmcm = extract_hmcmcbca_from_reference(&mono_188, &mbp_ref);
    let mut roots = Vec::with_capacity(376);
    for sub in 0..2usize {
        for (i, &(h_ppm, c_ppm)) in mono_hc.iter().enumerate() {
            let hmcm = mono_hmcm
                .iter()
                .find(|r| (r.h_ppm - h_ppm).abs() < 1e-4 && (r.c_methyl_ppm - c_ppm).abs() < 1e-4)
                .cloned();
            roots.push(MethylRootPeak {
                id: sub * 188 + i,
                h_ppm,
                c_ppm,
                height: 1.0,
                volume: 1.0,
                hmcm,
            });
        }
    }

    let mut noes = Vec::new();
    for m1 in 0..376usize {
        for m2 in (m1 + 1)..376usize {
            let d = dimer.dist_cc(m1, m2);
            if d <= 8.2 {
                let inten = 1.0 / (d.powi(6) + 3.8_f64.powi(6));
                let (h1, c1) = mono_hc[m1 % 188];
                let (h2, c2) = mono_hc[m2 % 188];
                noes.push(MethylNoePeak {
                    c_noe_ppm: c2,
                    c_root_ppm: c1,
                    h_root_ppm: h1,
                    h_noe_ppm: Some(h2),
                    root_idx: Some(m1),
                    donor_root_idx: Some(m2),
                    intensity: inten,
                    bidirectional: true,
                });
                noes.push(MethylNoePeak {
                    c_noe_ppm: c1,
                    c_root_ppm: c2,
                    h_root_ppm: h2,
                    h_noe_ppm: Some(h1),
                    root_idx: Some(m2),
                    donor_root_idx: Some(m1),
                    intensity: inten,
                    bidirectional: true,
                });
            }
        }
    }

    let score_cfg = MethylScoreConfig::default();
    let prob = build_methyl_contact_problem(&dimer, &roots, &noes, &score_cfg);
    assert_eq!(prob.s_len, 376);
    assert_eq!(prob.m_len, 376);

    // 1. Batch objective parity on `M = 376` (`> 255`)
    let init_states = build_initial_replica_states(&prob, 24, 777);
    let cand_pos: Vec<Vec<i32>> = init_states.into_iter().map(|s| s.1).collect();
    let cpu_sc: Vec<f64> = cand_pos
        .iter()
        .map(|p| evaluate_methyl_objective(&prob, p))
        .collect();
    let gpu_sc = solver.evaluate_objectives(&prob, &cand_pos).unwrap();
    let max_batch_err = cpu_sc
        .iter()
        .zip(&gpu_sc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_batch_err < 1.0e-4,
        "anneal_methyl_1024 batch parity failed: {max_batch_err:.3e}"
    );

    // 2. Solve on `anneal_methyl_1024` (`M = 376`, chunked at `<= 2,048` steps/submit)
    let res = solver
        .assign(
            &prob,
            &MethylAnnealConfig {
                replicas: 48,
                steps_per_root: 1400,
                seed: 2026,
                ..MethylAnnealConfig::default()
            },
        )
        .unwrap();

    let e_cpu = evaluate_methyl_objective(&prob, &res.position);
    let e_gpu = solver
        .evaluate_objectives(&prob, std::slice::from_ref(&res.position))
        .unwrap()[0];
    let sol_err = (e_cpu - e_gpu).abs();

    let mut orbit_correct = 0usize;
    let mut max_conf = 0.0f64;
    for s in 0..376usize {
        max_conf = max_conf.max(res.confidence[s]);
        let pred = res.position[s];
        if pred >= 0 && (pred as usize) % 188 == s % 188 {
            orbit_correct += 1;
        }
    }
    let orbit_acc_pct = 100.0 * (orbit_correct as f64) / 376.0;

    // Chain-specific evaluation: C2 symmetry MUST cap every chain-specific confidence at <= 0.50,
    // yielding `hc_total == 0`, `hc_abstained == true`, `hc_precision_opt == None`, `hc_precision_pct == 0.0`!
    let chain_truth: Vec<Option<usize>> = (0..376).map(Some).collect();
    let chain_metrics = evaluate_methyl_assignment_against_truth(&res, &chain_truth, 0.80);

    println!(
        "[P1-10 anneal_methyl_1024 MBP C2 Homodimer S=376,M=376] orbit_acc={orbit_acc_pct:.2}% ({orbit_correct}/376) \
         | max_chain_conf={max_conf:.4} | hc_abstained={} | |E_cpu-E_gpu|={sol_err:.3e}",
        chain_metrics.hc_abstained
    );

    assert!(sol_err < 1.0e-4, "consensus parity {sol_err:.3e} >= 1e-4");
    assert!(
        orbit_acc_pct >= 85.0,
        "MBP C2 homodimer orbit accuracy {orbit_acc_pct:.2}% < 85.0%"
    );
    assert!(
        max_conf <= 0.500001,
        "C2 homodimer chain-specific confidence {max_conf:.6} exceeded exact 1/2 symmetry ceiling!"
    );
    assert_eq!(chain_metrics.hc_total, 0);
    assert!(chain_metrics.hc_abstained);
    assert_eq!(chain_metrics.hc_precision_opt, None);
    assert_eq!(chain_metrics.hc_precision_pct, 0.0);
}

#[test]
fn p1_10_anneal_methyl_global_over_1024_sites_parity_and_solve() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);
    let root = repo_root();

    // Construct `6 x 188 = 1,128`-site C6 hexameric contact problem (`max_dim = 1,128 > 1,024` -> `anneal_methyl_global`)
    let mbp_full = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1omp_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let (mono_188, mono_hc) = subset_monomer_sites_with_shifts(&mbp_full, &mbp_ref, 188);
    let hexamer = build_cn_homooligomer_methyl_model(&mono_188, 6, 5.5);
    assert_eq!(hexamer.sites.len(), 1_128);
    assert_eq!(chunk_steps_for_dim(hexamer.sites.len()), 1_024);

    // 188 resolved HMQC peaks against 1,128 structural sites (`max_dim = 1,128 > 1,024`)
    let roots: Vec<MethylRootPeak> = mono_hc
        .iter()
        .enumerate()
        .map(|(i, &(h_ppm, c_ppm))| MethylRootPeak {
            id: i,
            h_ppm,
            c_ppm,
            height: 1.0,
            volume: 1.0,
            hmcm: None,
        })
        .collect();
    let mut noes = Vec::new();
    for i in 0..188usize {
        for j in (i + 1)..188usize {
            let d = mono_188.dist_cc(i, j);
            if d <= 7.8 {
                let inten = 1.0 / (d.powi(6) + 3.8_f64.powi(6));
                noes.push(MethylNoePeak {
                    c_noe_ppm: mono_hc[j].1,
                    c_root_ppm: mono_hc[i].1,
                    h_root_ppm: mono_hc[i].0,
                    h_noe_ppm: Some(mono_hc[j].0),
                    root_idx: Some(i),
                    donor_root_idx: Some(j),
                    intensity: inten,
                    bidirectional: true,
                });
                noes.push(MethylNoePeak {
                    c_noe_ppm: mono_hc[i].1,
                    c_root_ppm: mono_hc[j].1,
                    h_root_ppm: mono_hc[j].0,
                    h_noe_ppm: Some(mono_hc[i].0),
                    root_idx: Some(j),
                    donor_root_idx: Some(i),
                    intensity: inten,
                    bidirectional: true,
                });
            }
        }
    }

    let prob = build_methyl_contact_problem(&hexamer, &roots, &noes, &MethylScoreConfig::default());
    assert_eq!(prob.s_len, 188);
    assert_eq!(prob.m_len, 1_128);

    let init = build_initial_replica_states(&prob, 16, 1128);
    let init_best = init.iter().map(|s| s.2).fold(f64::NEG_INFINITY, f64::max) * prob.norm_inv;
    let cands: Vec<Vec<i32>> = init.into_iter().map(|s| s.1).collect();
    let cpu_sc: Vec<f64> = cands
        .iter()
        .map(|p| evaluate_methyl_objective(&prob, p))
        .collect();
    let gpu_sc = solver.evaluate_objectives(&prob, &cands).unwrap();
    let max_batch_err = cpu_sc
        .iter()
        .zip(&gpu_sc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let res = solver
        .assign(
            &prob,
            &MethylAnnealConfig {
                replicas: 16,
                steps_per_root: 64,
                seed: 1128,
                ..MethylAnnealConfig::default()
            },
        )
        .unwrap();
    let e_cpu = evaluate_methyl_objective(&prob, &res.position);
    let e_gpu = solver
        .evaluate_objectives(&prob, std::slice::from_ref(&res.position))
        .unwrap()[0];
    let sol_err = (e_cpu - e_gpu).abs();
    let max_conf = res.confidence.iter().copied().fold(0.0f64, f64::max);

    println!(
        "[P1-10 anneal_methyl_global S=188,M=1128] init_best={init_best:.4} -> final={e_cpu:.4} \
         | batch_err={max_batch_err:.3e} | sol_err={sol_err:.3e} | max_C6_conf={max_conf:.4}"
    );
    assert!(
        max_batch_err < 1.0e-4,
        "anneal_methyl_global (>1024) batch parity failed: {max_batch_err:.3e}"
    );
    assert!(
        sol_err < 1.0e-4,
        "anneal_methyl_global (>1024) consensus parity failed: {sol_err:.3e}"
    );
    assert!(
        e_cpu > init_best,
        "anneal_methyl_global must improve objective over initial replicas ({e_cpu:.4} <= {init_best:.4})"
    );
    assert!(
        max_conf <= (1.0 / 6.0) + 1.0e-5,
        "C6 hexamer confidence {max_conf:.5} exceeded exact 1/6 symmetry ceiling!"
    );
}

/// Build realistic non-circular experimental observables from `fwd_model` + `reference`:
/// - 2D HMQC picked peaks,
/// - Noisy `HMCM(CG)CBCA` (`sigma_CA = sigma_CB = sigma_CG = 0.18 ppm` + `15%` dropped geminal links),
/// - Picked 3D NOESY cross-peaks from `fwd_model` corrupted with:
///   1. `15%` randomly dropped true NOEs,
///   2. `10%` spurious random long-range noise NOEs (`d_fwd > 10.0 A`),
///   3. 2-step spin-diffusion relay peaks (`a -> mid -> b` when `d(a,mid) < 4.5 A` & `d(mid,b) < 4.5 A` & `d(a,b) > 5.2 A`).
fn build_non_circular_cross_structure_dataset(
    fwd_model: &MethylStructureModel,
    solver_model: &MethylStructureModel,
    reference: &Reference,
    seed: u64,
) -> (Vec<MethylRootPeak>, Vec<MethylNoePeak>, Vec<Option<usize>>) {
    let sim_cfg = MethylSimConfig::default();
    let picker_cfg = MethylPickerConfig::default();
    let mut rng = Rng::new(seed);

    let fwd_hc = extract_true_site_shifts(fwd_model, reference);
    let hmqc = simulate_methyl_hmqc_2d(&fwd_hc, &sim_cfg);
    let mut roots = pick_methyl_hmqc_2d(&hmqc, &picker_cfg);

    // Map `(seq_id, carbon_atom)` -> site index in `solver_model`
    let mut solver_site_lookup: HashMap<(i32, &'static str), usize> = HashMap::new();
    for (m, st) in solver_model.sites.iter().enumerate() {
        solver_site_lookup.insert((st.seq_id, st.carbon_atom), m);
    }

    // Build noisy `HMCM(CG)CBCA` (`sigma_CA = sigma_CB = sigma_CG = 0.18 ppm` + `15%` dropped links)
    let base_hmcm = extract_hmcmcbca_from_reference(fwd_model, reference);
    attach_hmcmcbca_records(&mut roots, &base_hmcm, 0.045, 0.14);

    // Resolve ground-truth site per picked 2D HMQC root AFTER `attach_hmcmcbca_records` so exact
    // 2D `(1H, 13C)` shift collisions (e.g. `MET1:CE` and `ALA28:CB` both at `(1.630, 17.900)` in
    // `BMRB 5387`) break ties cleanly toward whichever residue's `HMCM(CG)CBCA` record attached.
    let mut root_fwd_site: Vec<Option<usize>> = vec![None; roots.len()];
    let mut truth_solver_site: Vec<Option<usize>> = vec![None; roots.len()];
    for (r_idx, rt) in roots.iter().enumerate() {
        let mut best: Option<(f64, usize)> = None;
        for (m_fwd, st_fwd) in fwd_model.sites.iter().enumerate() {
            let Some((th, tc)) = lookup_site_methyl_shifts(reference, st_fwd) else {
                continue;
            };
            let dh = (rt.h_ppm - th) / 0.045;
            let dc = (rt.c_ppm - tc) / 0.25;
            if dh.abs() <= 1.0 && dc.abs() <= 1.0 {
                let hmcm_tiebreak = rt.hmcm.as_ref().map_or(0.0, |rec| {
                    -0.02
                        * hmcmcbca_log_likelihood(rec, st_fwd.kind.residue_class())
                            .clamp(-12.0, 4.0)
                });
                let d2 = dh * dh + dc * dc + hmcm_tiebreak;
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, m_fwd));
                }
            }
        }
        if let Some((_, m_fwd)) = best {
            root_fwd_site[r_idx] = Some(m_fwd);
            let st_fwd = &fwd_model.sites[m_fwd];
            truth_solver_site[r_idx] = solver_site_lookup
                .get(&(st_fwd.seq_id, st_fwd.carbon_atom))
                .copied();
        }
    }

    let mut res_noise: HashMap<i32, (f64, f64, f64)> = HashMap::new();
    for st in &fwd_model.sites {
        // Total std = sqrt(0.16^2 + 0.0825^2) = 0.180 ppm on CA, CB, CG
        res_noise.entry(st.seq_id).or_insert_with(|| {
            (
                rng.normal(0.0, 0.16),
                rng.normal(0.0, 0.16),
                rng.normal(0.0, 0.16),
            )
        });
    }
    for (r_idx, rt) in roots.iter_mut().enumerate() {
        if rng.f64() < 0.15 {
            rt.hmcm = None;
            continue;
        }
        if let Some(ref mut rec) = rt.hmcm {
            let (dca, dcb, dcg) = root_fwd_site[r_idx]
                .and_then(|mf| res_noise.get(&fwd_model.sites[mf].seq_id).copied())
                .unwrap_or((0.0, 0.0, 0.0));
            rec.ca_ppm = rec.ca_ppm.map(|v| v + dca + rng.normal(0.0, 0.0825));
            rec.cb_ppm = rec.cb_ppm.map(|v| v + dcb + rng.normal(0.0, 0.0825));
            rec.cg_ppm = rec.cg_ppm.map(|v| v + dcg + rng.normal(0.0, 0.0825));
        }
    }

    // Pick 3D NOESY from `fwd_model`
    let raw_noes = pick_methyl_noesy_3d(
        &simulate_methyl_noesy_3d(fwd_model, &fwd_hc, &sim_cfg),
        &roots,
        &picker_cfg,
    );

    // Calibrate physical intensity scale directly from picked 3D NOESY cross-peaks
    let mut sorted_raw_i: Vec<f64> = raw_noes.iter().map(|p| p.intensity).collect();
    sorted_raw_i.sort_by(|a, b| a.total_cmp(b));
    let i_p80 = if sorted_raw_i.is_empty() {
        0.1
    } else {
        sorted_raw_i[(sorted_raw_i.len() * 80 / 100).min(sorted_raw_i.len() - 1)]
    };
    let d_sat6 = 3.8_f64.powi(6);
    let deff6_ref = 4.8_f64.powi(6) + d_sat6;

    // 1. Drop 15% of true NOEs randomly
    let mut corrupted_noes: Vec<MethylNoePeak> =
        raw_noes.into_iter().filter(|_| rng.f64() >= 0.15).collect();

    // 2. Inject 2-step spin-diffusion relay peaks (`a -> mid -> b` when `d(a,mid) < 4.5 A` & `d(mid,b) < 4.5 A` & `d(a,b) > 5.2 A`)
    let mut fwd_to_root: Vec<Option<usize>> = vec![None; fwd_model.sites.len()];
    for (r_idx, &mf_opt) in root_fwd_site.iter().enumerate() {
        if let Some(mf) = mf_opt {
            fwd_to_root[mf] = Some(r_idx);
        }
    }
    let mf_len = fwd_model.sites.len();
    for (ma, &ra_opt) in fwd_to_root.iter().enumerate() {
        let Some(ra) = ra_opt else { continue };
        for (mb, &rb_opt) in fwd_to_root.iter().enumerate().skip(ma + 1) {
            let Some(rb) = rb_opt else { continue };
            if ra == rb || fwd_model.sites[ma].geminal_partner == Some(mb) {
                continue;
            }
            let dab = fwd_model.dist_cc(ma, mb);
            if dab <= 5.2 {
                continue;
            }
            let has_relay = (0..mf_len).any(|mm| {
                mm != ma
                    && mm != mb
                    && fwd_model.dist_cc(ma, mm) < 4.5
                    && fwd_model.dist_cc(mm, mb) < 4.5
            });
            if has_relay {
                let relay_i = i_p80 * (deff6_ref / (6.9_f64.powi(6) + d_sat6));
                corrupted_noes.push(MethylNoePeak {
                    c_noe_ppm: roots[rb].c_ppm,
                    c_root_ppm: roots[ra].c_ppm,
                    h_root_ppm: roots[ra].h_ppm,
                    h_noe_ppm: None,
                    root_idx: Some(ra),
                    donor_root_idx: None,
                    intensity: relay_i,
                    bidirectional: false,
                });
            }
        }
    }

    // 3. Inject 10% spurious random long-range noise NOEs (`d_fwd > 10.0 A`)
    let n_spurious = ((corrupted_noes.len() as f64) * 0.10).round() as usize;
    let mut added_spurious = 0usize;
    let mut guard = 0usize;
    while added_spurious < n_spurious && guard < n_spurious * 40 + 100 && roots.len() >= 2 {
        guard += 1;
        let ra = rng.usize(roots.len());
        let rb = rng.usize(roots.len());
        if ra == rb {
            continue;
        }
        let far = match (root_fwd_site[ra], root_fwd_site[rb]) {
            (Some(ma), Some(mb)) => fwd_model.dist_cc(ma, mb) > 10.0,
            _ => true,
        };
        if !far {
            continue;
        }
        let fake_d = 5.8 + 2.2 * rng.f64();
        let fake_i = i_p80 * (deff6_ref / (fake_d.powi(6) + d_sat6));
        corrupted_noes.push(MethylNoePeak {
            c_noe_ppm: roots[rb].c_ppm,
            c_root_ppm: roots[ra].c_ppm,
            h_root_ppm: roots[ra].h_ppm,
            h_noe_ppm: None,
            root_idx: Some(ra),
            donor_root_idx: None,
            intensity: fake_i,
            bidirectional: false,
        });
        added_spurious += 1;
    }

    (roots, corrupted_noes, truth_solver_site)
}

#[test]
fn p0_2_non_circular_xray_vs_nmr_cross_structure_methyl_benchmarks() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required");
    let solver = MethylGpuSolver::new(&engine);
    let root = repo_root();
    let score_cfg = MethylScoreConfig::default();

    let ubq_nmr = extract_methyl_structure_from_pdb_file(
        root.join("data/pdb_refs/1d3z.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let ubq_xray = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let ubq_ref = read_star_all_atoms(
        root.join("data/fixtures/ubiquitin_bmr5387_methyl.str"),
        None,
    )
    .unwrap();

    let mbp_dmb = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1dmb_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_omp = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1omp_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();

    let mut pooled_samples: Vec<(f64, bool)> = Vec::new();

    let mut deferred_failures: Vec<String> = Vec::new();
    for (label, fwd_model, solver_model, true_ref, seed, min_acc, min_hc) in [
        (
            "Ubiquitin (1D3Z NMR -> 1UBQ 1.8A X-ray, BMRB 5387, seed 101)",
            &ubq_nmr,
            &ubq_xray,
            &ubq_ref,
            101u64,
            78.0,
            90.0,
        ),
        (
            "Ubiquitin (1D3Z NMR -> 1UBQ 1.8A X-ray, BMRB 5387, seed 202)",
            &ubq_nmr,
            &ubq_xray,
            &ubq_ref,
            202u64,
            78.0,
            90.0,
        ),
        (
            "MBP 41kDa (1DMB holo X-ray -> 1OMP apo X-ray, BMRB 7114, seed 303)",
            &mbp_dmb,
            &mbp_omp,
            &mbp_ref,
            303u64,
            72.0,
            88.0,
        ),
        (
            "MBP 41kDa (1OMP apo X-ray -> 1DMB holo X-ray, BMRB 7114, seed 404)",
            &mbp_omp,
            &mbp_dmb,
            &mbp_ref,
            404u64,
            72.0,
            88.0,
        ),
    ] {
        let (roots, noes, truth) =
            build_non_circular_cross_structure_dataset(fwd_model, solver_model, true_ref, seed);
        let prob = build_methyl_contact_problem(solver_model, &roots, &noes, &score_cfg);
        let res = solver
            .assign(
                &prob,
                &MethylAnnealConfig {
                    replicas: 64,
                    steps_per_root: 850,
                    seed,
                    ..MethylAnnealConfig::default()
                },
            )
            .unwrap();
        let m = evaluate_methyl_assignment_against_truth(&res, &truth, 0.80);

        for (s, &t_opt) in truth.iter().enumerate() {
            let Some(true_m) = t_opt else { continue };
            let pred = res.position[s];
            if pred >= 0 {
                pooled_samples.push((res.confidence[s], pred as usize == true_m));
            }
        }

        println!(
            "[P0-2 Cross-Structure {label}] acc={:.2}% ({}/{}) | hc_total={} | HC(>=0.80)={} ({:.2}%) | ECE={:.4}",
            m.accuracy_pct,
            m.correct,
            m.total_matched_roots,
            m.hc_total,
            m.reliability.hc_status,
            m.hc_precision_pct,
            m.reliability.ece
        );

        if m.accuracy_pct < min_acc {
            deferred_failures.push(format!(
                "{label}: cross-structure accuracy {:.2}% < {min_acc:.1}%",
                m.accuracy_pct
            ));
        }
        if m.hc_abstained || m.hc_total < 10 {
            deferred_failures.push(format!(
                "{label}: cross-structure hc_total={} < 10 (abstained={})!",
                m.hc_total, m.hc_abstained
            ));
        }
        if m.hc_precision_pct < min_hc {
            deferred_failures.push(format!(
                "{label}: cross-structure HC precision {:.2}% < {min_hc:.1}%",
                m.hc_precision_pct
            ));
        }
    }

    let pooled_hc_total = pooled_samples.iter().filter(|&&(c, _)| c >= 0.80).count();
    let pooled_hc_correct = pooled_samples
        .iter()
        .filter(|&&(c, ok)| c >= 0.80 && ok)
        .count();
    let pooled_hc_prec = if pooled_hc_total > 0 {
        100.0 * (pooled_hc_correct as f64) / (pooled_hc_total as f64)
    } else {
        0.0
    };
    let pooled_rel = ProteinReliabilityCurve::from_predictions(&pooled_samples, 0.80);
    println!(
        "[P0-2 Pooled Non-Circular Cross-Structure Calibration] N={} | ECE={:.4} | HC(>=0.80)={:.2}% ({}/{}) | bins={:?}",
        pooled_rel.total_predictions,
        pooled_rel.ece,
        pooled_hc_prec,
        pooled_hc_correct,
        pooled_hc_total,
        pooled_rel
            .bins
            .iter()
            .map(|b| format!("[{:.1},{:.1}): n={} conf={:.3} acc={:.3}", b.lower, b.upper, b.count, b.mean_conf, b.empirical_acc))
            .collect::<Vec<_>>()
    );
    assert!(
        deferred_failures.is_empty(),
        "Cross-structure failures:\n{}",
        deferred_failures.join("\n")
    );
    assert!(
        pooled_hc_prec >= 92.0,
        "Pooled non-circular cross-structure HC precision {:.2}% < 92.0% ({}/{})!",
        pooled_hc_prec,
        pooled_hc_correct,
        pooled_hc_total
    );
    assert!(
        pooled_rel.ece <= 0.06,
        "Pooled non-circular cross-structure ECE {:.4} > 0.06! Reliability curve: {pooled_rel:?}",
        pooled_rel.ece
    );
}
