//! WebGPU `(U, P)` scoring matrix parity (`max |cpu - gpu| < 1e-4`), baseline browser
//! `wgpu::Limits::default()` portability, `anneal_idp_1024` large-chain SRAM dispatch, and
//! non-circular real BMRB IDP multi-replica GPU assignment benchmarks (`Phase 11`, `T11.2`).
//!
//! Provenance: Clean-room verification suite citing:
//! - Bermel, Bertini, Felli, Piccioli & Pierattelli (*Prog. Nucl. Magn. Reson. Spectrosc.*
//!   2006, 48, 25–45; *J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem. Soc.* 2009,
//!   131, 15339–15345 — `2D CON`, `CACO`, `3D CANCO`, `CBCACON`, `CCC(CO)N`, `NCACO`).
//! - Tamiola, Acar & Mulder (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP` library).
//! - Kjaergaard & Poulsen (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57).

use std::path::PathBuf;

use nmr_core::c13_detected::{
    C13DetectedExp, C13PickedDataset, C13Strip3DPeak, Caco2DPeak, CacoPeakKind, Con2DPeak,
};
use nmr_core::evaluate::score as evaluate_metrics;
use nmr_core::idp_solver::{
    idp_pair_matrix, idp_to_reference, idp_unary_matrix, register_and_build_con_dipeptide_systems,
    ConDipeptideSystem, IdpShiftOffsets, IdpSolverConfig,
};
use nmr_core::nmrstar::{read_star_all_atoms, Reference};
use nmr_core::qc::{DatasetQcVerdict, ProteinReliabilityCurve};
use nmr_core::rng::Rng;
use nmr_gpu::idp_gpu::{
    assign_idp_13c_from_picked_gpu, assign_idp_13c_gpu, score_idp_matrices_gpu,
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

fn simulate_mismatched_bmrb_picked_dataset(
    ref_data: &Reference,
    p_missing_3d: f64,
    spurious_rate: f64,
    seed: u64,
) -> (C13PickedDataset, Vec<i32>) {
    let mut rng = Rng::new(seed ^ 0xB42B_1D90_5EED_1234);
    let seq_len = ref_data.sequence.len();

    let glob_dn = -0.58;
    let glob_dc = 0.27;
    let glob_dca = -0.33;

    let off_con = (glob_dc + 0.09, glob_dn - 0.14);
    let off_caco = (glob_dc - 0.12, glob_dca + 0.15);
    let off_canco_ca = glob_dca - 0.10;
    let off_cbca_aliph = glob_dca + 0.13;
    let off_cccon_aliph = glob_dca - 0.16;
    let off_ncaco = (glob_dc + 0.18, glob_dn + 0.11, glob_dca + 0.09);

    let true_bonds: Vec<usize> = (1..seq_len)
        .filter(|&k| {
            ref_data.shifts.contains_key(&(k - 1, "C".to_string()))
                && ref_data.shifts.contains_key(&(k, "N".to_string()))
        })
        .collect();
    let n_spur_con = ((true_bonds.len() as f64) * 0.03).round().max(2.0) as usize;
    let mut root_specs: Vec<Option<usize>> = true_bonds.into_iter().map(Some).collect();
    root_specs.extend(std::iter::repeat_n(None, n_spur_con));

    for i in (1..root_specs.len()).rev() {
        let j = rng.usize(i + 1);
        root_specs.swap(i, j);
    }

    let mut con_roots = Vec::with_capacity(root_specs.len());
    let mut truth = Vec::with_capacity(root_specs.len());
    let mut strip_peaks = Vec::new();
    let sc_atoms = ["CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE"];

    for (r_idx, spec) in root_specs.iter().copied().enumerate() {
        let Some(k) = spec else {
            let cp_spur = 169.5 + rng.f64() * 9.5;
            let nc_spur = 107.5 + rng.f64() * 31.0;
            con_roots.push(Con2DPeak {
                c_prev_ppm: cp_spur,
                n_curr_ppm: nc_spur,
                height: 28.0 + rng.f64() * 6.0,
                volume: 28.0,
                is_proline_imino_region: nc_spur >= 131.5,
                deconvolved_from_cluster: false,
            });
            truth.push(-1);
            continue;
        };
        let cp_true = ref_data.shifts[&(k - 1, "C".to_string())];
        let nc_true = ref_data.shifts[&(k, "N".to_string())];
        let cp_obs = cp_true + off_con.0 + rng.normal(0.0, 0.012);
        let nc_obs = nc_true + off_con.1 + rng.normal(0.0, 0.025);

        con_roots.push(Con2DPeak {
            c_prev_ppm: cp_obs,
            n_curr_ppm: nc_obs,
            height: 100.0 * (1.0 + rng.normal(0.0, 0.08)).clamp(0.65, 1.35),
            volume: 100.0,
            is_proline_imino_region: nc_obs >= 131.5,
            deconvolved_from_cluster: false,
        });
        truth.push(k as i32);

        if let Some(&ca_k) = ref_data.shifts.get(&(k, "CA".to_string())) {
            if rng.f64() >= p_missing_3d {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: cp_obs,
                    n_ppm: nc_obs,
                    c_indirect_ppm: ca_k + off_canco_ca + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 95.0,
                    volume: 95.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }
        if let Some(&ca_km1) = ref_data.shifts.get(&(k - 1, "CA".to_string())) {
            if rng.f64() >= p_missing_3d {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: cp_obs,
                    n_ppm: nc_obs,
                    c_indirect_ppm: ca_km1 + off_canco_ca + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 65.0,
                    volume: 65.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
            if rng.f64() >= p_missing_3d {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: cp_obs,
                    n_ppm: nc_obs,
                    c_indirect_ppm: ca_km1 + off_cbca_aliph + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 90.0,
                    volume: 90.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }
        if !ref_data.sequence[k - 1].eq_ignore_ascii_case("GLY") {
            if let Some(&cb_km1) = ref_data.shifts.get(&(k - 1, "CB".to_string())) {
                if rng.f64() >= p_missing_3d {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cbcaco3D,
                        c_prime_ppm: cp_obs,
                        n_ppm: nc_obs,
                        c_indirect_ppm: cb_km1 + off_cbca_aliph + rng.normal(0.0, 0.03),
                        sign: -1,
                        height: -82.0,
                        volume: -82.0,
                        anchor_root_idx: Some(r_idx),
                    });
                }
            }
        }
        for &aname in &sc_atoms {
            if let Some(&sc_v) = ref_data.shifts.get(&(k - 1, aname.to_string())) {
                if (10.0..=56.0).contains(&sc_v) && rng.f64() >= p_missing_3d {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cccon3D,
                        c_prime_ppm: cp_obs,
                        n_ppm: nc_obs,
                        c_indirect_ppm: sc_v + off_cccon_aliph + rng.normal(0.0, 0.035),
                        sign: 1,
                        height: 60.0,
                        volume: 60.0,
                        anchor_root_idx: Some(r_idx),
                    });
                }
            }
        }
    }

    // Emit raw drifted unanchored 3D NCACO peaks (`anchor_root_idx: None`, zero oracle registration!)
    for k in 1..seq_len {
        let (Some(&nc), Some(&ca_k), Some(&c_k)) = (
            ref_data.shifts.get(&(k, "N".to_string())),
            ref_data.shifts.get(&(k, "CA".to_string())),
            ref_data.shifts.get(&(k, "C".to_string())),
        ) else {
            continue;
        };
        if rng.f64() >= p_missing_3d {
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Ncaco3D,
                c_prime_ppm: c_k + off_ncaco.0 + rng.normal(0.0, 0.015),
                n_ppm: nc + off_ncaco.1 + rng.normal(0.0, 0.025),
                c_indirect_ppm: ca_k + off_ncaco.2 + rng.normal(0.0, 0.025),
                sign: 1,
                height: 88.0,
                volume: 88.0,
                anchor_root_idx: None,
            });
        }
    }

    let mut caco_peaks = Vec::new();
    for i in 0..seq_len {
        let (Some(&c_i), Some(&ca_i)) = (
            ref_data.shifts.get(&(i, "C".to_string())),
            ref_data.shifts.get(&(i, "CA".to_string())),
        ) else {
            continue;
        };
        let is_gly = ref_data.sequence[i].eq_ignore_ascii_case("GLY");
        let sign = if is_gly { -1 } else { 1 };
        let ca_obs = ca_i + off_caco.1 + rng.normal(0.0, 0.02);
        let kind = if is_gly {
            CacoPeakKind::GlyOppositePhase
        } else if ca_obs >= 58.0 {
            CacoPeakKind::SerThrProDownfield
        } else {
            CacoPeakKind::GeneralAliphatic
        };
        caco_peaks.push(Caco2DPeak {
            c_curr_ppm: c_i + off_caco.0 + rng.normal(0.0, 0.014),
            ca_curr_ppm: ca_obs,
            sign,
            height: 95.0 * (sign as f64),
            volume: 95.0 * (sign as f64),
            kind,
        });
    }

    let n_spurious_3d = ((strip_peaks.len() as f64) * spurious_rate).round() as usize;
    let exps_3d = [
        C13DetectedExp::Canco3D,
        C13DetectedExp::Cbcaco3D,
        C13DetectedExp::Ncaco3D,
    ];
    for i in 0..n_spurious_3d {
        let r_idx = rng.usize(con_roots.len());
        let exp = exps_3d[i % exps_3d.len()];
        let (c_p, c_ind, sgn) = match exp {
            C13DetectedExp::Canco3D => (con_roots[r_idx].c_prev_ppm, 43.0 + rng.f64() * 23.0, 1),
            C13DetectedExp::Cbcaco3D => {
                let is_neg = rng.f64() < 0.5;
                (
                    con_roots[r_idx].c_prev_ppm,
                    if is_neg {
                        16.0 + rng.f64() * 52.0
                    } else {
                        43.0 + rng.f64() * 23.0
                    },
                    if is_neg { -1 } else { 1 },
                )
            }
            _ => (170.0 + rng.f64() * 8.5, 43.0 + rng.f64() * 23.0, 1),
        };
        let anchor_root_idx = if exp == C13DetectedExp::Ncaco3D {
            None
        } else {
            Some(r_idx)
        };
        let n_val = if exp == C13DetectedExp::Ncaco3D {
            108.0 + rng.f64() * 30.0
        } else {
            con_roots[r_idx].n_curr_ppm
        };
        strip_peaks.push(C13Strip3DPeak {
            exp,
            c_prime_ppm: c_p,
            n_ppm: n_val,
            c_indirect_ppm: c_ind,
            sign: sgn,
            height: 22.0 * (sgn as f64),
            volume: 22.0 * (sgn as f64),
            anchor_root_idx,
        });
    }

    (
        C13PickedDataset {
            con_roots,
            caco_peaks,
            strip_peaks,
        },
        truth,
    )
}

#[test]
fn test_idp_gpu_unary_and_pair_matrix_parity_and_browser_limits() {
    let engine = GpuEngine::new_headless().expect("WebGPU adapter required for idp_gpu_tests");
    let baseline_engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() device must initialize cleanly");

    let root = repo_root();
    for (rel_path, temp_k, ph) in [
        ("data/fixtures/bmr26639_idp_cbp_id4.str", 308.0, 6.5),
        ("data/benchmark/bmr6968.str", 288.0, 6.5),
    ] {
        let ref_data = read_star_all_atoms(root.join(rel_path), None).unwrap();
        let (picked, _) = simulate_mismatched_bmrb_picked_dataset(&ref_data, 0.08, 0.06, 123);
        let systems = register_and_build_con_dipeptide_systems(&picked);
        assert!(
            systems.iter().any(|s| !s.has_3d_support()),
            "parity test must include 3D-unsupported spurious 2D CON noise roots"
        );

        let cfg = IdpSolverConfig {
            temperature_k: temp_k,
            ph,
            ..IdpSolverConfig::default()
        };
        let offsets = IdpShiftOffsets {
            d_n: 0.16,
            d_c: -0.09,
            d_ca: 0.11,
            d_cb: -0.12,
        };

        let u_cpu = idp_unary_matrix(&systems, &ref_data.sequence, &cfg, &offsets);
        let p_cpu = idp_pair_matrix(&systems, &cfg);

        for (eng_label, eng) in [("native", &engine), ("baseline-limits", &baseline_engine)] {
            let (u_gpu, p_gpu) =
                score_idp_matrices_gpu(eng, &systems, &ref_data.sequence, &cfg, &offsets)
                    .expect("GPU IDP matrix scoring must succeed");
            assert_eq!(u_cpu.rows, u_gpu.rows);
            assert_eq!(u_cpu.cols, u_gpu.cols);
            assert_eq!(p_cpu.rows, p_gpu.rows);
            assert_eq!(p_cpu.cols, p_gpu.cols);

            let mut max_u_diff = 0.0f64;
            for (idx, (&a, &b)) in u_cpu.data.iter().zip(&u_gpu.data).enumerate() {
                let d = (a - b).abs();
                max_u_diff = max_u_diff.max(d);
                assert!(
                    d < 1.0e-4,
                    "[{rel_path} / {eng_label}] U parity violation at {idx}: cpu={a} gpu={b} (diff={d})"
                );
            }
            let mut max_p_diff = 0.0f64;
            for (idx, (&a, &b)) in p_cpu.data.iter().zip(&p_gpu.data).enumerate() {
                let d = (a - b).abs();
                max_p_diff = max_p_diff.max(d);
                assert!(
                    d < 1.0e-4,
                    "[{rel_path} / {eng_label}] P parity violation at {idx}: cpu={a} gpu={b} (diff={d})"
                );
            }
            println!(
                "[{rel_path} / {eng_label}] max |U_cpu - U_gpu| = {max_u_diff:.2e}, max |P_cpu - P_gpu| = {max_p_diff:.2e}"
            );
        }
    }
}

#[test]
fn test_idp_gpu_non_circular_real_bmrb_benchmarks_through_prolines() {
    let engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() device must initialize cleanly");
    let root = repo_root();

    let entries = [
        (
            root.join("data/benchmark/bmr6968.str"),
            "bmr6968 (alpha-synuclein, 140 aa)",
            288.15,
            6.5,
        ),
        (
            root.join("data/benchmark/bmr15176.str"),
            "bmr15176 (DARPP-32, 119 aa, 12.6% Pro)",
            298.15,
            6.5,
        ),
        (
            root.join("data/benchmark/bmr4375.str"),
            "bmr4375 (8M-urea Ubiquitin, 76 aa, pH 2.0)",
            298.15,
            2.0,
        ),
        (
            root.join("data/fixtures/bmr26639_idp_cbp_id4.str"),
            "bmr26639 (CBP-ID4, 207 aa, 21.7% Pro)",
            308.15,
            6.5,
        ),
        (
            root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
            "bmr51984 (p53-TAD-PRR, 102 aa, 21.6% Pro)",
            298.15,
            6.0,
        ),
    ];

    let mut pooled_preds: Vec<(f64, bool)> = Vec::new();
    for (idx, (path, label, temp_k, ph)) in entries.iter().enumerate() {
        let ref_data = read_star_all_atoms(path, None).unwrap();
        let (picked, truth) =
            simulate_mismatched_bmrb_picked_dataset(&ref_data, 0.10, 0.08, 700 + idx as u64);
        let cfg = IdpSolverConfig {
            temperature_k: *temp_k,
            ph: *ph,
            replicas: 48,
            islands: 4,
            generations: 3,
            steps_per_system: 2_600,
            seed: 42 + idx as u64,
            ..IdpSolverConfig::default()
        };

        let (systems, res) =
            assign_idp_13c_from_picked_gpu(&engine, &picked, &ref_data.sequence, &cfg)
                .expect("WebGPU IDP assignment must succeed");
        assert_eq!(res.qc_verdict, DatasetQcVerdict::Pass);

        let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
        let rel = res.reliability_curve(&truth, 0.80);
        for (s, &pos) in res.position.iter().enumerate() {
            if pos >= 0 {
                pooled_preds.push((res.confidence[s], pos == truth[s]));
            }
        }

        println!(
            "[GPU] {label}: acc={:.2}% ({}/{}), HC_prec={:.2}% ({}/{}), ECE={:.4}",
            m.accuracy() * 100.0,
            m.correct,
            m.assigned,
            m.hc_precision() * 100.0,
            m.hc_correct,
            m.hc_assigned,
            rel.ece
        );

        assert!(
            truth.iter().filter(|&&t| t == -1).count() >= 2,
            "[GPU] {label}: benchmark must include shuffled spurious 2D CON noise roots (truth == -1)"
        );
        for (s, &t) in truth.iter().enumerate() {
            if t == -1 && !systems[s].has_3d_support() {
                assert_eq!(
                    res.position[s], -1,
                    "[GPU] {label}: 3D-unsupported spurious 2D CON noise root s={s} falsely assigned to k={}",
                    res.position[s]
                );
            }
        }
        assert!(
            m.accuracy() >= 0.92,
            "[GPU] {label}: accuracy {:.2}% < 92.0%",
            m.accuracy() * 100.0
        );
        assert!(
            m.hc_precision() >= 0.96,
            "[GPU] {label}: HC precision {:.2}% < 96.0%",
            m.hc_precision() * 100.0
        );
        assert!(
            rel.ece <= 0.06,
            "[GPU] {label}: per-protein ECE {:.4} > 0.06",
            rel.ece
        );

        if label.contains("26639") {
            let (pred_ref, _) = idp_to_reference(
                &res,
                &systems,
                &ref_data.sequence,
                &ref_data.seq_ids,
                0.35,
                true,
            );
            let assigned_pro_n = ref_data
                .sequence
                .iter()
                .enumerate()
                .filter(|&(k, r)| {
                    k > 0
                        && r.eq_ignore_ascii_case("PRO")
                        && pred_ref.shifts.contains_key(&(k, "N".to_string()))
                })
                .count();
            let total_pro = ref_data
                .sequence
                .iter()
                .skip(1)
                .filter(|r| r.eq_ignore_ascii_case("PRO"))
                .count();
            let mut pro_roots = 0usize;
            let mut pro_correct = 0usize;
            for (s, &tk) in truth.iter().enumerate() {
                if tk >= 1 && ref_data.sequence[tk as usize].eq_ignore_ascii_case("PRO") {
                    pro_roots += 1;
                    if res.position[s] == tk {
                        pro_correct += 1;
                    }
                }
            }
            assert!(
                assigned_pro_n * 10 >= total_pro * 9 && pro_correct * 10 >= pro_roots * 9,
                "[GPU] {label}: Proline N shifts must be continuously & accurately assigned ({assigned_pro_n}/{total_pro} exported, {pro_correct}/{pro_roots} exact)"
            );
        }
    }

    let pooled = ProteinReliabilityCurve::from_predictions(&pooled_preds, 0.80);
    println!(
        "[GPU] Pooled 5-protein real BMRB IDP calibration: total={} HC_prec={:.2}% ECE={:.4}",
        pooled.total_predictions,
        pooled.hc_precision.unwrap_or(0.0) * 100.0,
        pooled.ece
    );
    assert!(
        pooled.ece <= 0.06,
        "[GPU] pooled ECE {:.4} > 0.06",
        pooled.ece
    );
}

#[test]
fn test_idp_gpu_1024_sram_kernel_dispatch() {
    let engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() device must initialize cleanly");
    let root = repo_root();
    let cbp =
        read_star_all_atoms(root.join("data/fixtures/bmr26639_idp_cbp_id4.str"), None).unwrap();
    let p53 = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();

    // Build a 309-residue proline-rich chimera (> 256 residues) to exercise `anneal_idp_1024`
    let mut seq = cbp.sequence.clone();
    seq.extend(p53.sequence.iter().cloned());
    assert!(seq.len() > 256 && seq.len() <= 1024);

    let (picked_cbp, _) = simulate_mismatched_bmrb_picked_dataset(&cbp, 0.05, 0.0, 901);
    let systems = register_and_build_con_dipeptide_systems(&picked_cbp);
    let cfg = IdpSolverConfig {
        replicas: 12,
        islands: 2,
        generations: 2,
        steps_per_system: 600,
        seed: 77,
        ..IdpSolverConfig::default()
    };
    let res = assign_idp_13c_gpu(&engine, &systems, &seq, &cfg)
        .expect("anneal_idp_1024 (>256 slots) must execute cleanly on WebGPU");
    assert!(res.score > 500.0);
    assert_eq!(res.position.len(), systems.len());
}

#[test]
fn test_idp_gpu_matrix_size_cap_and_multiplicity_parity_under_3x_noise() {
    let engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() device must initialize cleanly");

    // 1. Enforce `S * K <= 25,000,000` (`Err` on overflow) on WebGPU path too
    let big_systems: Vec<ConDipeptideSystem> = (0..5_001)
        .map(|i| ConDipeptideSystem::new(i, 174.5, 120.0, 90.0))
        .collect();
    let big_seq: Vec<String> = std::iter::repeat_n("ALA".to_string(), 5_001).collect();
    let cfg = IdpSolverConfig::default();
    let off0 = IdpShiftOffsets::default();
    assert!(score_idp_matrices_gpu(&engine, &big_systems, &big_seq, &cfg, &off0).is_err());
    assert!(assign_idp_13c_gpu(&engine, &big_systems, &big_seq, &cfg).is_err());

    // 2. Verify `< 1e-4` CPU/WebGPU `(U, P)` parity when multi-candidate multiplicity penalties are heavily active
    let root = repo_root();
    let p53 = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    let (mut picked, _) = simulate_mismatched_bmrb_picked_dataset(&p53, 0.05, 0.25, 3333);
    // Boost spurious peak heights so multiple candidates per channel survive and trigger `w_mult`
    for p in &mut picked.strip_peaks {
        if p.height.abs() < 50.0 {
            p.height = p.sign as f64 * 88.0;
        }
    }
    let systems = register_and_build_con_dipeptide_systems(&picked);
    let u_cpu = idp_unary_matrix(&systems, &p53.sequence, &cfg, &off0);
    let p_cpu = idp_pair_matrix(&systems, &cfg);
    let (u_gpu, p_gpu) = score_idp_matrices_gpu(&engine, &systems, &p53.sequence, &cfg, &off0)
        .expect("GPU scoring must succeed");

    let mut max_du = 0.0f64;
    let mut max_dp = 0.0f64;
    for s in 0..systems.len() {
        for k in 0..p53.sequence.len() {
            let cu = u_cpu.get(s, k);
            let gu = u_gpu.get(s, k);
            if cu > -1e15 || gu > -1e15 {
                max_du = max_du.max((cu - gu).abs());
            }
        }
        for t in 0..systems.len() {
            let cp = p_cpu.get(s, t);
            let gp = p_gpu.get(s, t);
            if cp > -1e15 || gp > -1e15 {
                max_dp = max_dp.max((cp - gp).abs());
            }
        }
    }
    println!("[GPU Multiplicity Stress Parity] max |U_cpu - U_gpu| = {max_du:.2e}, max |P_cpu - P_gpu| = {max_dp:.2e}");
    assert!(max_du < 1e-4, "max |U_cpu - U_gpu| = {max_du} >= 1e-4");
    assert!(max_dp < 1e-4, "max |P_cpu - P_gpu| = {max_dp} >= 1e-4");
}
