//! Integration & non-circular BMRB IDP benchmark tests for `Phase 11` `T11.2`:
//! Continuous through-Proline `13C`-detected IDP solver (`ConDipeptideSystem`, `ncIDP` temperature/pH
//! random-coil library, permutation-invariant unlabelled `(dN, dC', dCA, dCB)` offset estimation,
//! multi-island evolutionary + detailed-balance annealing, `ProteinReliabilityCurve` `ECE <= 0.06`,
//! explicit `ABSTAIN`, and model-mismatched real BMRB IDP benchmarks).

use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, pick_simulated_c13_dataset_with_diagnostics,
    simulate_idp_13c_cubes, C13DetectedExp, C13PickedDataset, C13PickerConfig, C13Strip3DPeak,
    Caco2DPeak, CacoPeakKind, Con2DPeak, IdpCubeSimConfig,
};
use nmr_core::evaluate::score as evaluate_metrics;
use nmr_core::idp_solver::{
    assign_idp_13c_cpu, assign_idp_13c_from_picked, estimate_idp_reference_offsets,
    evaluate_idp_lopo_calibration, idp_to_reference, predict_ncidp_shifts,
    register_and_build_con_dipeptide_systems, try_assign_idp_13c_cpu, try_idp_pair_matrix,
    try_idp_unary_matrix, ConDipeptideSystem, IdpShiftOffsets, IdpSolverConfig,
    IDP_CALIBRATION_KNOTS, MAX_IDP_CB_CANDS, MAX_IDP_MATRIX_CELLS,
};
use nmr_core::nmrstar::{read_star_all_atoms, Reference};
use nmr_core::qc::{DatasetQcVerdict, ProteinReliabilityCurve};
use nmr_core::rng::Rng;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Simulate realistic unlabelled `C13PickedDataset` peak lists directly from a real BMRB `Reference`
/// under model-mismatched conditions:
/// - Inter-experiment `0.08–0.20 ppm` referencing drift across `CON`, `CACO`, `CANCO`, `CBCACON`,
///   `CCC(CO)N`, and `NCACO` + global `(dN, dC', dCA, dCB)` referencing offset,
/// - Real experimental BMRB shifts (never `ncIDP`!) + measurement noise,
/// - `p_missing_3d` (`10%`) randomly dropped 3D relay peaks,
/// - `spurious_rate` (`8%`) spurious 3D noise artifact peaks + spurious `2D CON` noise roots,
/// - Shuffled root order so root indices carry zero sequence-order information!
fn simulate_mismatched_bmrb_picked_dataset(
    ref_data: &Reference,
    p_missing_3d: f64,
    spurious_rate: f64,
    seed: u64,
) -> (C13PickedDataset, Vec<i32>) {
    let mut rng = Rng::new(seed ^ 0xB42B_1D90_5EED_1234);
    let seq_len = ref_data.sequence.len();

    // Global spectrometer referencing offset + independent 0.08..0.20 ppm inter-experiment drifts
    let glob_dn = -0.58;
    let glob_dc = 0.27;
    let glob_dca = -0.33;

    let off_con = (glob_dc + 0.09, glob_dn - 0.14);
    let off_caco = (glob_dc - 0.12, glob_dca + 0.15);
    let off_canco_ca = glob_dca - 0.10;
    let off_cbca_aliph = glob_dca + 0.13;
    let off_cccon_aliph = glob_dca - 0.16;
    let off_ncaco = (glob_dc + 0.18, glob_dn + 0.11, glob_dca + 0.09);

    // Collect true peptide bonds `k in 1..seq_len` (`Some(k)`) plus ~3% spurious 2D CON noise roots (`None`)
    let true_bonds: Vec<usize> = (1..seq_len)
        .filter(|&k| {
            ref_data.shifts.contains_key(&(k - 1, "C".to_string()))
                && ref_data.shifts.contains_key(&(k, "N".to_string()))
        })
        .collect();
    let n_spur_con = ((true_bonds.len() as f64) * 0.03).round().max(2.0) as usize;
    let mut root_specs: Vec<Option<usize>> = true_bonds.into_iter().map(Some).collect();
    root_specs.extend(std::iter::repeat_n(None, n_spur_con));

    // Shuffle `root_specs` via Fisher-Yates so both true and spurious `con_roots` are completely scrambled!
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

        // 3D CANCO: CA(k) & CA(k-1) with 10% random dropout
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
        }

        // 3D CBCACON: +CA(k-1) and -CB(k-1) with 10% random dropout
        if let Some(&ca_km1) = ref_data.shifts.get(&(k - 1, "CA".to_string())) {
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

        // 3D CCC(CO)N: sidechain aliphatic carbons of (k-1) when deposited in BMRB
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

    // 3D NCACO: emit raw drifted unanchored 3D peaks (`anchor_root_idx: None`, zero oracle registration!)
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

    // 2D CACO: all residues `0..seq_len` with deposited `(C', CA)`
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

    // Inject 8% spurious 3D noise artifacts onto random roots!
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
fn test_ncidp_neighbor_ph_temperature_and_pre_proline_physics() {
    let seq: Vec<String> = ["ALA", "ASP", "GLU", "HIS", "SER", "PRO", "PRO", "GLY"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cfg_neutral = IdpSolverConfig {
        temperature_k: 298.15,
        ph: 7.0,
        ..IdpSolverConfig::default()
    };
    let cfg_acid_hot = IdpSolverConfig {
        temperature_k: 313.15,
        ph: 2.2,
        ..IdpSolverConfig::default()
    };

    let p_neut = predict_ncidp_shifts(&seq, &cfg_neutral);
    let p_acid = predict_ncidp_shifts(&seq, &cfg_acid_hot);

    // Pre-Proline Ser(4) preceding Pro(5) must be shifted ~ -1.95 ppm upfield on CA and C'
    assert!(
        p_neut[4].ca_ppm < 58.3 - 1.5,
        "Ser before Pro CA={}",
        p_neut[4].ca_ppm
    );
    assert!(
        p_neut[4].c_ppm < 174.6 - 1.5,
        "Ser before Pro C'={}",
        p_neut[4].c_ppm
    );
    // Pro(5) preceding Pro(6) has both Pro imino N (~136.5 ppm) AND pre-Proline CA/C' upfield shift!
    assert!(p_neut[5].n_ppm > 134.0 && p_neut[5].n_ppm < 140.0);
    assert!(p_neut[5].ca_ppm < 63.2 - 1.4);
    assert!(p_neut[6].ca_ppm > 62.5);

    // Acidic pH 2.2 protonates Asp(1), Glu(2), His(3), shifting CB & CA upfield
    assert!(p_acid[1].cb_ppm.unwrap() < p_neut[1].cb_ppm.unwrap() - 2.2);
    assert!(p_acid[2].cb_ppm.unwrap() < p_neut[2].cb_ppm.unwrap() - 1.2);
    // +15 K temperature shifts amide 15N upfield (~ -0.30 ppm)
    assert!(p_acid[4].n_ppm < p_neut[4].n_ppm - 0.20);
}

#[test]
fn test_idp_reference_offset_estimator_strict_permutation_invariance() {
    let root = workspace_root();
    let ref_data =
        read_star_all_atoms(root.join("data/fixtures/bmr26639_idp_cbp_id4.str"), None).unwrap();
    let (picked, _) = simulate_mismatched_bmrb_picked_dataset(&ref_data, 0.05, 0.04, 123);
    let systems = register_and_build_con_dipeptide_systems(&picked);

    let cfg = IdpSolverConfig {
        temperature_k: 308.15,
        ph: 6.5,
        ..IdpSolverConfig::default()
    };
    let off0 = estimate_idp_reference_offsets(&systems, &ref_data.sequence, &cfg);

    // Permute `systems` order AND reverse internal candidate vectors across 5 random permutations
    let mut rng = Rng::new(9999);
    for _ in 0..5 {
        let mut perm = systems.clone();
        for i in (1..perm.len()).rev() {
            let j = rng.usize(i + 1);
            perm.swap(i, j);
        }
        for s in &mut perm {
            s.ca_curr.reverse();
            s.ca_prev.reverse();
            s.cb_prev.reverse();
            s.c_curr.reverse();
        }
        let off_p = estimate_idp_reference_offsets(&perm, &ref_data.sequence, &cfg);
        assert_eq!(
            off0.d_n, off_p.d_n,
            "d_n must be bit-for-bit permutation invariant"
        );
        assert_eq!(
            off0.d_c, off_p.d_c,
            "d_c must be bit-for-bit permutation invariant"
        );
        assert_eq!(
            off0.d_ca, off_p.d_ca,
            "d_ca must be bit-for-bit permutation invariant"
        );
        assert_eq!(
            off0.d_cb, off_p.d_cb,
            "d_cb must be bit-for-bit permutation invariant"
        );
    }

    // Verify exact differential offset recovery under an arbitrary extra reference shift
    // (independent of any intrinsic referencing convention in the original BMRB deposition!)
    let (dn_inj, dc_inj, dca_inj, dcb_inj) = (-0.84, 0.41, -0.52, 0.47);
    let mut shifted_sys = systems.clone();
    for s in &mut shifted_sys {
        s.n_curr += dn_inj;
        s.c_prev += dc_inj;
        for v in &mut s.ca_curr {
            *v += dca_inj;
        }
        for v in &mut s.ca_prev {
            *v += dca_inj;
        }
        for v in &mut s.cb_prev {
            *v += dcb_inj;
        }
    }
    let off_sh = estimate_idp_reference_offsets(&shifted_sys, &ref_data.sequence, &cfg);
    assert!(
        ((off_sh.d_n - off0.d_n) - dn_inj).abs() < 0.04,
        "differential d_n recovery error: got {}, expected {}",
        off_sh.d_n - off0.d_n,
        dn_inj
    );
    assert!(
        ((off_sh.d_c - off0.d_c) - dc_inj).abs() < 0.03,
        "differential d_c recovery error: got {}, expected {}",
        off_sh.d_c - off0.d_c,
        dc_inj
    );
    assert!(
        ((off_sh.d_ca - off0.d_ca) - dca_inj).abs() < 0.04,
        "differential d_ca recovery error: got {}, expected {}",
        off_sh.d_ca - off0.d_ca,
        dca_inj
    );
    assert!(
        ((off_sh.d_cb - off0.d_cb) - dcb_inj).abs() < 0.05,
        "differential d_cb recovery error: got {}, expected {}",
        off_sh.d_cb - off0.d_cb,
        dcb_inj
    );
}

#[test]
fn test_explicit_abstain_when_3d_support_missing() {
    let seq: Vec<String> = [
        "MET", "ALA", "SER", "PRO", "GLY", "THR", "VAL", "PRO", "GLU", "LYS",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let systems: Vec<ConDipeptideSystem> = (0..9)
        .map(|i| {
            ConDipeptideSystem::new(i, 174.5 + 0.3 * (i as f64), 118.0 + 1.5 * (i as f64), 80.0)
        })
        .collect();
    let truth: Vec<i32> = (1..=9).collect();

    let res = assign_idp_13c_cpu(&systems, &seq, &IdpSolverConfig::default());
    assert_eq!(res.qc_verdict, DatasetQcVerdict::Abstain);
    assert!(res.confidence.iter().all(|&c| c <= 0.45));
    let rel = res.reliability_curve(&truth, 0.80);
    assert!(rel.is_abstained());
    assert_eq!(rel.hc_assigned, 0);
}

#[test]
fn test_non_circular_real_bmrb_idp_benchmarks_with_drift_missing_and_spurious_peaks() {
    let root = workspace_root();
    // 5 real public BMRB IDP depositions (3 in data/benchmark/, 2 >=20% Pro in data/fixtures/):
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
    let mut failures: Vec<String> = Vec::new();
    println!(
        "\n=== NON-CIRCULAR REAL BMRB 13C-IDP BENCHMARK (drift + 10% missing + 8% spurious) ==="
    );
    println!("| Entry | Roots | Assigned | Accuracy | HC (>=0.80) Precision | ECE | QC |");
    println!("|---|---:|---:|---:|---:|---:|---|");

    for (idx, (path, label, temp_k, ph)) in entries.iter().enumerate() {
        let ref_data = read_star_all_atoms(path, None).expect("valid BMRB STAR");
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
        let (systems, res) = assign_idp_13c_from_picked(&picked, &ref_data.sequence, &cfg);
        let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
        let rel = res.reliability_curve(&truth, 0.80);

        for (s, &pos) in res.position.iter().enumerate() {
            if pos >= 0 {
                pooled_preds.push((res.confidence[s], pos == truth[s]));
            }
        }

        println!(
            "| {} | {} | {} | {:.2}% ({}/{}) | {:.2}% ({}/{}) | {:.4} | {} |",
            label,
            m.assignable,
            m.assigned,
            m.accuracy() * 100.0,
            m.correct,
            m.assigned,
            m.hc_precision() * 100.0,
            m.hc_correct,
            m.hc_assigned,
            rel.ece,
            res.qc_verdict
        );
        assert_eq!(
            res.qc_verdict,
            DatasetQcVerdict::Pass,
            "{label}: 1x clean BMRB IDP must pass QC; got {:?} ({:?})",
            res.qc_verdict,
            res.qc_reasons
        );

        // Verify continuous through-Proline shift output on Proline-complete 13C-detected entries
        if label.contains("26639") || label.contains("6968") {
            let (pred_ref, _) = idp_to_reference(
                &res,
                &systems,
                &ref_data.sequence,
                &ref_data.seq_ids,
                0.50,
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
            assert!(
                assigned_pro_n * 10 >= total_pro * 9,
                "{label}: Proline N shifts must be continuously assigned ({assigned_pro_n}/{total_pro})"
            );
        }

        assert!(
            truth.iter().filter(|&&t| t == -1).count() >= 2,
            "{label}: benchmark must include shuffled spurious 2D CON noise roots (truth == -1)"
        );
        for (s, &t) in truth.iter().enumerate() {
            if t == -1 && !systems[s].has_3d_support() {
                assert_eq!(
                    res.position[s], -1,
                    "{label}: 3D-unsupported spurious 2D CON noise root s={s} falsely assigned to k={}",
                    res.position[s]
                );
            }
        }
        if m.accuracy() < 0.92 {
            failures.push(format!(
                "{label}: sequence-specific accuracy {:.2}% < 92.0% gate!",
                m.accuracy() * 100.0
            ));
        }
        if m.hc_precision() < 0.96 {
            failures.push(format!(
                "{label}: high-confidence precision {:.2}% < 96.0% gate!",
                m.hc_precision() * 100.0
            ));
        }
        if rel.ece > 0.06 {
            failures.push(format!(
                "{label}: per-protein ECE {:.4} > 0.06 gate!",
                rel.ece
            ));
        }
    }

    let pooled_curve = ProteinReliabilityCurve::from_predictions(&pooled_preds, 0.80);
    println!(
        "Pooled 5-protein real BMRB IDP calibration: total={} HC_prec={:.2}% ECE={:.4}",
        pooled_curve.total_predictions,
        pooled_curve.hc_precision.unwrap_or(0.0) * 100.0,
        pooled_curve.ece
    );
    assert!(
        failures.is_empty(),
        "Real BMRB IDP benchmark gate failures:\n{}",
        failures.join("\n")
    );
    assert!(
        pooled_curve.ece <= 0.06,
        "Pooled ECE {:.4} exceeds 0.06 gate!",
        pooled_curve.ece
    );
}

#[test]
fn test_end_to_end_volumetric_cubes_to_idp_solver_through_prolines() {
    let root = workspace_root();
    let full =
        read_star_all_atoms(root.join("data/fixtures/bmr26639_idp_cbp_id4.str"), None).unwrap();
    // 55-residue proline-dense segment of CBP-ID4 (residues 1..=55, containing multiple -Xaa-Pro- and -Pro-Pro-)
    let len = 55usize;
    let mut sub = Reference::new(full.sequence[..len].to_vec(), full.seq_ids[..len].to_vec());
    for ((i, atom), &val) in &full.shifts {
        if *i < len {
            sub.shifts.insert((*i, atom.clone()), val);
        }
    }

    let mut exp_offsets = BTreeMap::new();
    exp_offsets.insert(C13DetectedExp::Caco2D, (-0.10, 0.0, 0.12));
    exp_offsets.insert(C13DetectedExp::Canco3D, (0.11, -0.14, -0.09));
    exp_offsets.insert(C13DetectedExp::Cbcaco3D, (-0.09, 0.12, 0.11));
    exp_offsets.insert(C13DetectedExp::Ncaco3D, (0.08, 0.10, -0.08));

    let sim_cfg = IdpCubeSimConfig {
        proton_mhz: 800.0,
        temperature_k: 308.15,
        ph: 6.5,
        con_grid: (1024, 1024),
        caco_grid: (512, 512),
        cube_3d_grid: (160, 160, 160),
        exp_offsets,
        seed: 55,
        ..IdpCubeSimConfig::default()
    };
    let cubes = simulate_idp_13c_cubes(&sub, &sim_cfg);
    let pcfg = C13PickerConfig::default();
    let con_roots = pick_con_2d(&cubes.con_2d, &pcfg);
    let caco_peaks = pick_caco_2d(cubes.caco_2d.as_ref().unwrap(), &pcfg);
    let mut strip_peaks = Vec::new();
    for (exp, cube) in &cubes.cubes_3d {
        strip_peaks.extend(pick_con_anchored_3d(cube, *exp, &con_roots, &pcfg));
    }
    let picked = C13PickedDataset {
        con_roots,
        caco_peaks,
        strip_peaks,
    };

    // Map each picked CON root to ground-truth peptide bond index `k` if within (0.04 C', 0.16 N)
    let truth: Vec<i32> = picked
        .con_roots
        .iter()
        .map(|r| {
            cubes
                .ground_truth_con
                .iter()
                .find(|&&(_, cp, np, _, _)| {
                    (r.c_prev_ppm - cp).abs() <= 0.04 && (r.n_curr_ppm - np).abs() <= 0.16
                })
                .map(|&(k, _, _, _, _)| k as i32)
                .unwrap_or(-1)
        })
        .collect();

    let solver_cfg = IdpSolverConfig {
        temperature_k: 308.15,
        ph: 6.5,
        seed: 55,
        ..IdpSolverConfig::default()
    };
    let (_, res) = assign_idp_13c_from_picked(&picked, &sub.sequence, &solver_cfg);
    let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
    assert!(
        m.accuracy() >= 0.92,
        "End-to-end 3D volumetric CBP-ID4 accuracy {:.2}% < 92%!",
        m.accuracy() * 100.0
    );
    assert!(
        m.hc_precision() >= 0.96,
        "End-to-end 3D volumetric CBP-ID4 HC precision {:.2}% < 96%!",
        m.hc_precision() * 100.0
    );
}

#[test]
fn test_unsupervised_ncaco_drift_and_spurious_con_gap_rejection() {
    let root = workspace_root();
    // Use incomplete real BMRB deposition `bmr15176` (`DARPP-32`, 119 aa, 17 missing bond gaps -> `s_len + 1 < k_len`)
    let darpp = read_star_all_atoms(root.join("data/benchmark/bmr15176.str"), None).unwrap();
    let (mut picked, mut truth) = simulate_mismatched_bmrb_picked_dataset(&darpp, 0.05, 0.04, 909);

    // Inject 4 explicit high-risk spurious 2D CON noise roots (`truth = -1`, zero 3D relay peaks),
    // including Proline (`136.8 ppm`) and Glycine (`109.7 ppm`) where 2D `(C', N)` alone would match sequence gaps!
    for &(cp, np) in &[
        (175.40, 136.80),
        (174.10, 109.70),
        (173.20, 121.40),
        (176.10, 118.90),
    ] {
        picked.con_roots.push(Con2DPeak {
            c_prev_ppm: cp,
            n_curr_ppm: np,
            height: 30.0,
            volume: 30.0,
            is_proline_imino_region: np >= 131.5,
            deconvolved_from_cluster: false,
        });
        truth.push(-1);
    }

    // Verify unsupervised NCACO alignment on large `(+0.14 ppm C', -0.19 ppm N, +0.13 ppm CA)` unanchored drift:
    // replace all NCACO peaks with fresh heavily drifted unanchored peaks (`anchor_root_idx: None`)
    picked
        .strip_peaks
        .retain(|p| p.exp != C13DetectedExp::Ncaco3D);
    for k in 1..darpp.sequence.len() {
        let (Some(&nc), Some(&ca_k), Some(&c_k)) = (
            darpp.shifts.get(&(k, "N".to_string())),
            darpp.shifts.get(&(k, "CA".to_string())),
            darpp.shifts.get(&(k, "C".to_string())),
        ) else {
            continue;
        };
        picked.strip_peaks.push(C13Strip3DPeak {
            exp: C13DetectedExp::Ncaco3D,
            c_prime_ppm: c_k + 0.14,
            n_ppm: nc - 0.19,
            c_indirect_ppm: ca_k + 0.13,
            sign: 1,
            height: 88.0,
            volume: 88.0,
            anchor_root_idx: None,
        });
    }

    let systems = register_and_build_con_dipeptide_systems(&picked);
    // True roots with sequential partners must recover `c_curr` registered onto the `CON` `C'` frame (`+0.08 ppm` in `simulate_mismatched_bmrb_picked_dataset`)
    let with_c_curr = systems
        .iter()
        .zip(truth.iter())
        .filter(|&(s, &t)| t >= 1 && !s.c_curr.is_empty())
        .count();
    assert!(
        with_c_curr >= 80,
        "Unsupervised unanchored NCACO registration must recover >=80 c_curr shifts despite (+0.14 C', -0.19 N) drift, got {with_c_curr}"
    );

    let cfg = IdpSolverConfig {
        temperature_k: 298.15,
        ph: 6.5,
        seed: 909,
        ..IdpSolverConfig::default()
    };
    let res = assign_idp_13c_cpu(&systems, &darpp.sequence, &cfg);

    // Every single 3D-unsupported spurious 2D CON noise root (`truth == -1`, `n_3d_ch == 0`) MUST remain unassigned (`-1`),
    // even when `s_len + 1 < k_len` leaves empty sequence gaps!
    for (s, &t) in truth.iter().enumerate() {
        if t == -1 && !systems[s].has_3d_support() {
            assert_eq!(
                res.position[s], -1,
                "3D-unsupported spurious 2D CON root s={s} was falsely placed at k={} into an empty sequence gap!",
                res.position[s]
            );
        }
    }
    let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
    assert!(
        m.accuracy() >= 0.95,
        "DARPP-32 accuracy {:.2}% < 95%",
        m.accuracy() * 100.0
    );
    assert!(
        m.hc_precision() >= 0.98,
        "DARPP-32 HC precision {:.2}% < 98%",
        m.hc_precision() * 100.0
    );
}

/// Simulate a `3x` realistic-noise (`sigma_obs_ca = 0.055 ppm` -> `sigma_seq_ca ~ 0.078 ppm`,
/// `sigma_obs_c = 0.032 ppm`, `sigma_obs_n = 0.070 ppm`) + **equal-height** spurious 3D peak
/// (`height = 88.0..94.0` vs true `65..95`) dataset directly from real BMRB shifts to reproduce
/// and regression-gate the Phase 11 review `P1-7` / `Q3` stress test.
fn simulate_3x_realistic_noise_bmrb_dataset(
    ref_data: &Reference,
    p_missing_3d: f64,
    spurious_rate: f64,
    seed: u64,
) -> (C13PickedDataset, Vec<i32>) {
    let mut rng = Rng::new(seed ^ 0x3333_7777_AAAA_5555);
    let seq_len = ref_data.sequence.len();

    let glob_dn = -0.46;
    let glob_dc = 0.23;
    let glob_dca = -0.29;

    let off_con = (glob_dc + 0.08, glob_dn - 0.12);
    let off_caco = (glob_dc - 0.11, glob_dca + 0.14);
    let off_canco_ca = glob_dca - 0.09;
    let off_cbca_aliph = glob_dca + 0.12;
    let off_cccon_aliph = glob_dca - 0.14;
    let off_ncaco = (glob_dc + 0.16, glob_dn + 0.10, glob_dca + 0.08);

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
                height: 30.0,
                volume: 30.0,
                is_proline_imino_region: nc_spur >= 131.5,
                deconvolved_from_cluster: false,
            });
            truth.push(-1);
            continue;
        };
        let cp_true = ref_data.shifts[&(k - 1, "C".to_string())];
        let nc_true = ref_data.shifts[&(k, "N".to_string())];
        let cp_obs = cp_true + off_con.0 + rng.normal(0.0, 0.032);
        let nc_obs = nc_true + off_con.1 + rng.normal(0.0, 0.070);

        con_roots.push(Con2DPeak {
            c_prev_ppm: cp_obs,
            n_curr_ppm: nc_obs,
            height: 100.0 * (1.0 + rng.normal(0.0, 0.10)).clamp(0.65, 1.35),
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
                    c_indirect_ppm: ca_k + off_canco_ca + rng.normal(0.0, 0.055),
                    sign: 1,
                    height: 92.0,
                    volume: 92.0,
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
                    c_indirect_ppm: ca_km1 + off_canco_ca + rng.normal(0.0, 0.055),
                    sign: 1,
                    height: 70.0,
                    volume: 70.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }
        if let Some(&ca_km1) = ref_data.shifts.get(&(k - 1, "CA".to_string())) {
            if rng.f64() >= p_missing_3d {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: cp_obs,
                    n_ppm: nc_obs,
                    c_indirect_ppm: ca_km1 + off_cbca_aliph + rng.normal(0.0, 0.055),
                    sign: 1,
                    height: 88.0,
                    volume: 88.0,
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
                        c_indirect_ppm: cb_km1 + off_cbca_aliph + rng.normal(0.0, 0.065),
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
                        c_indirect_ppm: sc_v + off_cccon_aliph + rng.normal(0.0, 0.070),
                        sign: 1,
                        height: 60.0,
                        volume: 60.0,
                        anchor_root_idx: Some(r_idx),
                    });
                }
            }
        }
    }

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
                c_prime_ppm: c_k + off_ncaco.0 + rng.normal(0.0, 0.032),
                n_ppm: nc + off_ncaco.1 + rng.normal(0.0, 0.070),
                c_indirect_ppm: ca_k + off_ncaco.2 + rng.normal(0.0, 0.055),
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
        let ca_obs = ca_i + off_caco.1 + rng.normal(0.0, 0.050);
        let kind = if is_gly {
            CacoPeakKind::GlyOppositePhase
        } else if ca_obs >= 58.0 {
            CacoPeakKind::SerThrProDownfield
        } else {
            CacoPeakKind::GeneralAliphatic
        };
        caco_peaks.push(Caco2DPeak {
            c_curr_ppm: c_i + off_caco.0 + rng.normal(0.0, 0.030),
            ca_curr_ppm: ca_obs,
            sign,
            height: 95.0 * (sign as f64),
            volume: 95.0 * (sign as f64),
            kind,
        });
    }

    // Equal-height spurious 3D peaks (`height = 88.0` vs true `70..92`, surviving `0.35 * max` filters!)
    let n_spurious_3d = ((strip_peaks.len() as f64) * spurious_rate).round() as usize;
    let true_root_indices: Vec<usize> = truth
        .iter()
        .enumerate()
        .filter_map(|(idx, &t)| (t >= 1).then_some(idx))
        .collect();
    let exps_3d = [
        C13DetectedExp::Canco3D,
        C13DetectedExp::Cbcaco3D,
        C13DetectedExp::Ncaco3D,
    ];
    for i in 0..n_spurious_3d {
        let r_idx = true_root_indices[rng.usize(true_root_indices.len())];
        let exp = exps_3d[i % exps_3d.len()];
        let (c_p, c_ind, sgn) = match exp {
            C13DetectedExp::Canco3D => (con_roots[r_idx].c_prev_ppm, 50.0 + rng.f64() * 14.0, 1),
            C13DetectedExp::Cbcaco3D => {
                let is_neg = rng.f64() < 0.5;
                (
                    con_roots[r_idx].c_prev_ppm,
                    if is_neg {
                        24.0 + rng.f64() * 40.0
                    } else {
                        50.0 + rng.f64() * 14.0
                    },
                    if is_neg { -1 } else { 1 },
                )
            }
            _ => (171.5 + rng.f64() * 6.0, 50.0 + rng.f64() * 14.0, 1),
        };
        let anchor_root_idx = if exp == C13DetectedExp::Ncaco3D {
            None
        } else {
            Some(r_idx)
        };
        let n_val = if exp == C13DetectedExp::Ncaco3D {
            con_roots[r_idx].n_curr_ppm + rng.normal(0.0, 0.05)
        } else {
            con_roots[r_idx].n_curr_ppm
        };
        strip_peaks.push(C13Strip3DPeak {
            exp,
            c_prime_ppm: c_p,
            n_ppm: n_val,
            c_indirect_ppm: c_ind,
            sign: sgn,
            height: 88.0 * (sgn as f64),
            volume: 88.0 * (sgn as f64),
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
fn test_idp_5fold_leave_one_protein_out_calibration_across_low_and_3x_realistic_noise() {
    let root = workspace_root();
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

    let mut per_protein_raw: Vec<Vec<(f64, bool)>> = vec![Vec::new(); entries.len()];
    let mut noisy_pooled_cal: Vec<(f64, bool)> = Vec::new();
    let mut noisy_runs_warn = 0usize;
    let mut noisy_runs_total = 0usize;
    let mut correct_with_mult = 0usize;
    let mut assigned_with_mult = 0usize;
    let mut hc_correct_with_mult = 0usize;
    let mut hc_assigned_with_mult = 0usize;
    let mut correct_no_mult = 0usize;

    println!("\n=== P1-7 & Q3: 3x REALISTIC-NOISE + EQUAL-HEIGHT SPURIOUS IDP ABLATION (3 seeds x 5 BMRB IDPs = 15 runs) ===");
    for (idx, (path, label, temp_k, ph)) in entries.iter().enumerate() {
        let ref_data = read_star_all_atoms(path, None).unwrap();

        // Fold sample A: 1x regime
        let (picked_1x, truth_1x) =
            simulate_mismatched_bmrb_picked_dataset(&ref_data, 0.10, 0.08, 700 + idx as u64);
        let cfg_1x = IdpSolverConfig {
            temperature_k: *temp_k,
            ph: *ph,
            replicas: 36,
            islands: 4,
            generations: 3,
            steps_per_system: 2_200,
            seed: 42 + idx as u64,
            ..IdpSolverConfig::default()
        };
        let (_, res_1x) = assign_idp_13c_from_picked(&picked_1x, &ref_data.sequence, &cfg_1x);
        assert_eq!(res_1x.qc_verdict, DatasetQcVerdict::Pass);
        for (s, &pos) in res_1x.position.iter().enumerate() {
            if pos >= 0 {
                per_protein_raw[idx].push((res_1x.raw_posterior[s], pos == truth_1x[s]));
            }
        }

        // Fold sample B: 3 seeds of 3x realistic noise + equal-height spurious 3D peaks
        let mut prot_corr = 0usize;
        let mut prot_ass = 0usize;
        let mut prot_hc_c = 0usize;
        let mut prot_hc_a = 0usize;
        let mut prot_corr_nomult = 0usize;
        let mut prot_warn = 0usize;

        for sd in 0..3u64 {
            let (picked_3x, truth_3x) = simulate_3x_realistic_noise_bmrb_dataset(
                &ref_data,
                0.10,
                0.16,
                1_500 + (idx as u64) * 17 + sd,
            );
            let cfg_3x = IdpSolverConfig {
                temperature_k: *temp_k,
                ph: *ph,
                replicas: 36,
                islands: 4,
                generations: 3,
                steps_per_system: 2_200,
                seed: 100 + (idx as u64) * 13 + sd,
                ..IdpSolverConfig::default()
            };
            let (sys_3x, res_3x) =
                assign_idp_13c_from_picked(&picked_3x, &ref_data.sequence, &cfg_3x);
            noisy_runs_total += 1;
            if res_3x.qc_verdict != DatasetQcVerdict::Pass {
                noisy_runs_warn += 1;
                prot_warn += 1;
            }
            let m = evaluate_metrics(&res_3x.as_assign_result(), &truth_3x, 0.80);
            prot_corr += m.correct;
            prot_ass += m.assigned;
            prot_hc_c += m.hc_correct;
            prot_hc_a += m.hc_assigned;
            correct_with_mult += m.correct;
            assigned_with_mult += m.assigned;
            hc_correct_with_mult += m.hc_correct;
            hc_assigned_with_mult += m.hc_assigned;

            for (s, &pos) in res_3x.position.iter().enumerate() {
                if pos >= 0 {
                    let ok = pos == truth_3x[s];
                    noisy_pooled_cal.push((res_3x.confidence[s], ok));
                    per_protein_raw[idx].push((res_3x.raw_posterior[s], ok));
                }
            }

            // Ablation: identical systems & seed with `w_mult = 0.0` (unpenalized multiplicity)
            let cfg_nomult = IdpSolverConfig {
                w_mult: 0.0,
                ..cfg_3x
            };
            let res_nomult = assign_idp_13c_cpu(&sys_3x, &ref_data.sequence, &cfg_nomult);
            let m_nm = evaluate_metrics(&res_nomult.as_assign_result(), &truth_3x, 0.80);
            prot_corr_nomult += m_nm.correct;
            correct_no_mult += m_nm.correct;
        }

        println!(
            "| {label} | w_mult=0.35 Acc={:.2}% ({prot_corr}/{prot_ass}) | w_mult=0 Acc={:.2}% | HC_prec={:.2}% ({prot_hc_c}/{prot_hc_a}) | QC!=Pass: {prot_warn}/3 |",
            100.0 * (prot_corr as f64) / (prot_ass.max(1) as f64),
            100.0 * (prot_corr_nomult as f64) / (prot_ass.max(1) as f64),
            100.0 * (prot_hc_c as f64) / (prot_hc_a.max(1) as f64),
        );
    }

    let acc_mult = (correct_with_mult as f64) / (assigned_with_mult.max(1) as f64);
    let acc_nomult = (correct_no_mult as f64) / (assigned_with_mult.max(1) as f64);
    let hc_mult = (hc_correct_with_mult as f64) / (hc_assigned_with_mult.max(1) as f64);
    let noisy_curve = ProteinReliabilityCurve::from_predictions(&noisy_pooled_cal, 0.80);

    let knot_xs: Vec<f64> = IDP_CALIBRATION_KNOTS.iter().map(|&(x, _)| x).collect();
    let (fitted_knots, oof_ece, per_fold_ece) =
        evaluate_idp_lopo_calibration(&per_protein_raw, &knot_xs, 0.80);
    println!(
        "Pooled 15-run 3x realistic-noise IDP: Acc(w_mult=0.35)={:.2}% vs Acc(w_mult=0)={:.2}%, HC_prec={:.2}%, ECE={:.4}, QC_triggered={noisy_runs_warn}/{noisy_runs_total}",
        acc_mult * 100.0,
        acc_nomult * 100.0,
        hc_mult * 100.0,
        noisy_curve.ece,
    );
    println!(
        "5-Fold LOPO OOF ECE = {oof_ece:.4}, per-fold = {per_fold_ece:.4?}\nFitted LOPO knots = {fitted_knots:.4?}"
    );

    assert_eq!(
        noisy_runs_warn, noisy_runs_total,
        "All 15/15 3x realistic-noise + equal-height spurious runs must trigger QC Warn (got {noisy_runs_warn}/{noisy_runs_total})"
    );
    assert!(
        acc_mult >= acc_nomult,
        "Multiplicity penalty w_mult=0.35 ({:.2}%) must not underperform w_mult=0 ({:.2}%)",
        acc_mult * 100.0,
        acc_nomult * 100.0
    );
    assert!(
        hc_mult >= 0.94,
        "3x realistic-noise pooled HC precision {:.2}% < 94.0%",
        hc_mult * 100.0
    );
    assert!(
        noisy_curve.ece <= 0.06,
        "3x realistic-noise pooled ECE {:.4} > 0.06",
        noisy_curve.ece
    );
    assert!(
        oof_ece <= 0.055,
        "5-fold LOPO out-of-fold ECE {oof_ece:.4} > 0.055"
    );
    for (&(kx, ky), &(fx, fy)) in IDP_CALIBRATION_KNOTS.iter().zip(fitted_knots.iter()) {
        assert!((kx - fx).abs() < 1e-9);
        assert!(
            (ky - fy).abs() <= 0.005,
            "IDP_CALIBRATION_KNOTS at x={kx:.2} ({ky:.4}) drifted from reproducible LOPO knot ({fy:.4})!"
        );
    }
}

#[test]
fn test_end_to_end_volumetric_realistic_noise_benchmark_and_qc_triggers() {
    let root = workspace_root();
    let full = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    let len = 52usize;
    let mut sub = Reference::new(full.sequence[..len].to_vec(), full.seq_ids[..len].to_vec());
    for ((i, atom), &val) in &full.shifts {
        if *i < len {
            sub.shifts.insert((*i, atom.clone()), val);
        }
    }

    let mut exp_offsets = BTreeMap::new();
    exp_offsets.insert(C13DetectedExp::Caco2D, (-0.11, 0.0, 0.13));
    exp_offsets.insert(C13DetectedExp::Canco3D, (0.12, -0.15, -0.10));
    exp_offsets.insert(C13DetectedExp::Cbcaco3D, (-0.10, 0.13, 0.12));
    exp_offsets.insert(C13DetectedExp::Ncaco3D, (0.09, 0.11, -0.09));

    // Volumetric cubes with realistic post-registration matching noise (`sigma_seq_ca ~ 0.06–0.08 ppm`)
    // AND equal/comparable-height spurious 3D strip artifacts (`spurious_3d_rel_amp = 0.90`)
    let sim_cfg = IdpCubeSimConfig {
        proton_mhz: 700.0,
        temperature_k: 298.15,
        ph: 6.0,
        snr_2d: 36.0,
        snr_3d: 22.0,
        con_grid: (1024, 1024),
        caco_grid: (512, 512),
        cube_3d_grid: (128, 128, 144),
        exp_offsets,
        per_peak_jitter_ppm: Some((0.024, 0.058, 0.072)),
        spurious_con_rate: 0.04,
        spurious_3d_rel_amp: 0.90,
        spurious_strip_artifact_rate: 0.24,
        seed: 2027,
        ..IdpCubeSimConfig::default()
    };
    let cubes = simulate_idp_13c_cubes(&sub, &sim_cfg);
    let pcfg = C13PickerConfig::default();
    let (picked, diags) = pick_simulated_c13_dataset_with_diagnostics(&cubes, &pcfg);
    let picker_warnings: Vec<String> = diags.into_iter().filter_map(|d| d.warning).collect();

    let truth: Vec<i32> = picked
        .con_roots
        .iter()
        .map(|r| {
            cubes
                .ground_truth_con
                .iter()
                .find(|&&(_, cp, np, _, _)| {
                    (r.c_prev_ppm - cp).abs() <= 0.055 && (r.n_curr_ppm - np).abs() <= 0.22
                })
                .map(|&(k, _, _, _, _)| k as i32)
                .unwrap_or(-1)
        })
        .collect();

    let solver_cfg = IdpSolverConfig {
        temperature_k: 298.15,
        ph: 6.0,
        picker_warnings,
        seed: 2027,
        ..IdpSolverConfig::default()
    };
    let (_, res) = assign_idp_13c_from_picked(&picked, &sub.sequence, &solver_cfg);
    let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
    let rel = res.reliability_curve(&truth, 0.80);
    println!(
        "[Volumetric Realistic-Noise E2E p53-52aa] roots={} acc={:.2}% ({}/{}) HC_prec={:.2}% ({}/{}) ECE={:.4} QC={} reasons={:?}",
        picked.con_roots.len(),
        m.accuracy() * 100.0,
        m.correct,
        m.assigned,
        m.hc_precision() * 100.0,
        m.hc_correct,
        m.hc_assigned,
        rel.ece,
        res.qc_verdict,
        res.qc_reasons
    );

    assert_ne!(
        res.qc_verdict,
        DatasetQcVerdict::Pass,
        "Volumetric realistic-noise + equal-height spurious cubes must trigger QC Warn!"
    );
    assert!(
        m.accuracy() >= 0.84,
        "Volumetric realistic-noise E2E accuracy {:.2}% < 84%!",
        m.accuracy() * 100.0
    );
    assert!(
        m.hc_assigned == 0 || m.hc_precision() >= 0.95,
        "Volumetric realistic-noise E2E HC precision {:.2}% < 95%!",
        m.hc_precision() * 100.0
    );
}

#[test]
fn test_poly_q80_and_fg_repeat_volumetric_e2e_qc_triggers_and_hc_safety() {
    let pcfg = C13PickerConfig::default();

    // 1. Poly-Q80 (`M-K-Q80-G-S`, 84 aa, 83 expected peptide bonds):
    // Volumetric simulation with random-coil shifts + micro-dispersion (~0.015 ppm) merges overlapping
    // Gln CON roots (`n_roots << 0.70 * 83`) AND hits primary-sequence homopolymer QC!
    let mut q80_seq = vec!["MET".to_string(), "LYS".to_string()];
    q80_seq.extend(std::iter::repeat_n("GLN".to_string(), 80));
    q80_seq.push("GLY".to_string());
    q80_seq.push("SER".to_string());
    let q80_ids: Vec<i32> = (1..=(q80_seq.len() as i32)).collect();
    let mut q80_ref = Reference::new(q80_seq.clone(), q80_ids);
    let q80_preds = predict_ncidp_shifts(&q80_seq, &IdpSolverConfig::default());
    let mut rng = Rng::new(8080);
    for (i, pr) in q80_preds.iter().enumerate() {
        q80_ref
            .shifts
            .insert((i, "N".to_string()), pr.n_ppm + rng.normal(0.0, 0.04));
        q80_ref
            .shifts
            .insert((i, "C".to_string()), pr.c_ppm + rng.normal(0.0, 0.018));
        q80_ref
            .shifts
            .insert((i, "CA".to_string()), pr.ca_ppm + rng.normal(0.0, 0.025));
        if let Some(cb) = pr.cb_ppm {
            q80_ref
                .shifts
                .insert((i, "CB".to_string()), cb + rng.normal(0.0, 0.03));
        }
    }
    let q80_cubes = simulate_idp_13c_cubes(
        &q80_ref,
        &IdpCubeSimConfig {
            proton_mhz: 800.0,
            con_grid: (768, 768),
            caco_grid: (384, 384),
            cube_3d_grid: (112, 112, 128),
            seed: 8080,
            ..IdpCubeSimConfig::default()
        },
    );
    let (q80_picked, _) = pick_simulated_c13_dataset_with_diagnostics(&q80_cubes, &pcfg);
    let (_, q80_res) =
        assign_idp_13c_from_picked(&q80_picked, &q80_seq, &IdpSolverConfig::default());
    println!(
        "[Poly-Q80 Volumetric E2E] expected_bonds=83 picked_roots={} QC={} max_conf={:.3} reasons={:?}",
        q80_picked.con_roots.len(),
        q80_res.qc_verdict,
        q80_res.confidence.iter().copied().fold(0.0f64, f64::max),
        q80_res.qc_reasons
    );
    assert!(
        (q80_picked.con_roots.len() as f64) < 0.70 * 83.0,
        "Volumetric Q80 CON picker should exhibit severe root merging (<70% of 83 bonds), got {}",
        q80_picked.con_roots.len()
    );
    assert_ne!(
        q80_res.qc_verdict,
        DatasetQcVerdict::Pass,
        "Poly-Q80 must NEVER return QC Pass!"
    );
    assert!(
        q80_res.confidence.iter().all(|&c| c < 0.80),
        "Poly-Q80 must abstain from high-confidence (>=0.80) assignments!"
    );

    // 2. FG-repeat domain (Nup98-like 56-aa `GLFG`/`SLFG`/`GFGS`/`GFSG` repeat IDP):
    let fg_motifs = [
        ["GLY", "LEU", "PHE", "GLY"],
        ["SER", "LEU", "PHE", "GLY"],
        ["GLY", "PHE", "GLY", "SER"],
        ["THR", "GLY", "PHE", "GLY"],
    ];
    let mut fg_seq: Vec<String> = Vec::with_capacity(56);
    for blk in 0..14 {
        for &aa in &fg_motifs[blk % fg_motifs.len()] {
            fg_seq.push(aa.to_string());
        }
    }
    let fg_ids: Vec<i32> = (1..=(fg_seq.len() as i32)).collect();
    let mut fg_ref = Reference::new(fg_seq.clone(), fg_ids);
    let fg_preds = predict_ncidp_shifts(&fg_seq, &IdpSolverConfig::default());
    for (i, pr) in fg_preds.iter().enumerate() {
        fg_ref
            .shifts
            .insert((i, "N".to_string()), pr.n_ppm + rng.normal(0.0, 0.14));
        fg_ref
            .shifts
            .insert((i, "C".to_string()), pr.c_ppm + rng.normal(0.0, 0.05));
        fg_ref
            .shifts
            .insert((i, "CA".to_string()), pr.ca_ppm + rng.normal(0.0, 0.07));
        if let Some(cb) = pr.cb_ppm {
            fg_ref
                .shifts
                .insert((i, "CB".to_string()), cb + rng.normal(0.0, 0.08));
        }
    }
    let fg_cubes = simulate_idp_13c_cubes(
        &fg_ref,
        &IdpCubeSimConfig {
            proton_mhz: 800.0,
            con_grid: (768, 768),
            caco_grid: (384, 384),
            cube_3d_grid: (112, 112, 128),
            seed: 9898,
            ..IdpCubeSimConfig::default()
        },
    );
    let (fg_picked, _) = pick_simulated_c13_dataset_with_diagnostics(&fg_cubes, &pcfg);
    let fg_truth: Vec<i32> = fg_picked
        .con_roots
        .iter()
        .map(|r| {
            fg_cubes
                .ground_truth_con
                .iter()
                .find(|&&(_, cp, np, _, _)| {
                    (r.c_prev_ppm - cp).abs() <= 0.045 && (r.n_curr_ppm - np).abs() <= 0.18
                })
                .map(|&(k, _, _, _, _)| k as i32)
                .unwrap_or(-1)
        })
        .collect();
    let (_, fg_res) = assign_idp_13c_from_picked(&fg_picked, &fg_seq, &IdpSolverConfig::default());
    let fg_m = evaluate_metrics(&fg_res.as_assign_result(), &fg_truth, 0.80);
    println!(
        "[FG-Repeat Volumetric E2E] roots={} acc={:.2}% HC_assigned={} HC_prec={:.2}% QC={} reasons={:?}",
        fg_picked.con_roots.len(),
        fg_m.accuracy() * 100.0,
        fg_m.hc_assigned,
        fg_m.hc_precision() * 100.0,
        fg_res.qc_verdict,
        fg_res.qc_reasons
    );
    assert_ne!(
        fg_res.qc_verdict,
        DatasetQcVerdict::Pass,
        "FG-repeat IDP must trigger low-complexity / repeat QC (Warn or Abstain), never Pass!"
    );
    assert!(
        fg_m.hc_assigned == 0 || fg_m.hc_precision() >= 0.95,
        "FG-repeat HC precision must be >=95% or abstained (hc_assigned={}, hc_prec={:.2}%)",
        fg_m.hc_assigned,
        fg_m.hc_precision() * 100.0
    );
}

#[test]
fn test_idp_p2_cb_multi_candidate_size_cap_and_carrier_drift_clamp_warnings() {
    assert_eq!(MAX_IDP_CB_CANDS, 4);
    assert_eq!(MAX_IDP_MATRIX_CELLS, 25_000_000);

    // 1. CPU `S * K <= 25,000,000` size-cap enforcement (`5,001 * 5,001 = 25,010,001 > 25,000,000`)
    let big_systems: Vec<ConDipeptideSystem> = (0..5_001)
        .map(|i| ConDipeptideSystem::new(i, 174.5, 120.0, 90.0))
        .collect();
    let big_seq: Vec<String> = std::iter::repeat_n("ALA".to_string(), 5_001).collect();
    let cfg = IdpSolverConfig::default();

    assert!(
        try_idp_unary_matrix(&big_systems, &big_seq, &cfg, &IdpShiftOffsets::default()).is_err()
    );
    assert!(try_idp_pair_matrix(&big_systems, &cfg).is_err());
    assert!(try_assign_idp_13c_cpu(&big_systems, &big_seq, &cfg).is_err());
    let fallback = assign_idp_13c_cpu(&big_systems, &big_seq, &cfg);
    assert_eq!(fallback.qc_verdict, DatasetQcVerdict::Abstain);

    // 2. Multi-candidate `cb_prev` (`MAX_IDP_CB_CANDS = 4` vs legacy `first_cb` lowest-CB bug):
    // (a) Differential offset recovery on p53 when a stronger lower-ppm negative artifact (`26.5 ppm`)
    // is injected alongside every downfield `Ser`/`Thr` `CB` (`> 55 ppm`) under `+0.62 ppm` extra `CB` shift.
    let root = workspace_root();
    let p53 = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    let (mut picked, _) = simulate_mismatched_bmrb_picked_dataset(&p53, 0.0, 0.0, 515);
    let systems_base = register_and_build_con_dipeptide_systems(&picked);
    let off_base = estimate_idp_reference_offsets(&systems_base, &p53.sequence, &cfg);

    let dcb_extra = 1.42f64;
    let mut extra_artifacts = Vec::new();
    for p in &mut picked.strip_peaks {
        if p.exp == C13DetectedExp::Cbcaco3D && p.sign < 0 {
            p.c_indirect_ppm += dcb_extra;
            if p.c_indirect_ppm > 55.0 {
                extra_artifacts.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: p.c_prime_ppm,
                    n_ppm: p.n_ppm,
                    c_indirect_ppm: 26.5,
                    sign: -1,
                    height: -95.0,
                    volume: -95.0,
                    anchor_root_idx: p.anchor_root_idx,
                });
            }
        }
    }
    picked.strip_peaks.extend(extra_artifacts);
    let systems_cb = register_and_build_con_dipeptide_systems(&picked);
    assert!(
        systems_cb
            .iter()
            .any(|s| s.cb_prev.len() >= 2 && s.cb_prev.iter().any(|&v| v > 58.0)),
        "Systems with Ser/Thr + lower-ppm artifact must retain multiple CB candidates up to 4!"
    );
    let off_cb = estimate_idp_reference_offsets(&systems_cb, &p53.sequence, &cfg);
    assert!(
        ((off_cb.d_cb - off_base.d_cb) - dcb_extra).abs() <= 0.06,
        "Multi-CB offset estimator must recover +{dcb_extra:.2} ppm differential d_cb despite 26.5 ppm artifacts: got {:.3} (base {:.3}, shifted {:.3})",
        off_cb.d_cb - off_base.d_cb,
        off_base.d_cb,
        off_cb.d_cb
    );

    // (b) Pure Ser/Thr-dominated peptide where EVERY Ser/Thr system has `cb_prev = [25.0 (artifact), true_CB + 0.75]`:
    // legacy `first_cb` (`min CB = 25.0`) would fail completely (`d_cb <= 0`), whereas `MAX_IDP_CB_CANDS = 4` recovers `+0.75 ppm`!
    let st_seq: Vec<String> = ["SER", "THR", "SER", "THR", "SER", "THR", "SER"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let st_preds = predict_ncidp_shifts(&st_seq, &cfg);
    let st_systems: Vec<ConDipeptideSystem> = (1..st_seq.len())
        .map(|k| {
            let mut s =
                ConDipeptideSystem::new(k - 1, st_preds[k - 1].c_ppm, st_preds[k].n_ppm, 100.0);
            s.ca_curr.push(st_preds[k].ca_ppm);
            s.ca_prev.push(st_preds[k - 1].ca_ppm);
            s.cb_prev = vec![25.0, st_preds[k - 1].cb_ppm.unwrap() + 0.75];
            s
        })
        .collect();
    let st_off = estimate_idp_reference_offsets(&st_systems, &st_seq, &cfg);
    assert!(
        (st_off.d_cb - 0.75).abs() <= 0.06,
        "Ser/Thr multi-CB offset estimator must ignore 25.0 ppm lowest-CB artifact and recover +0.75 ppm, got {:.3}",
        st_off.d_cb
    );

    // 3. Solver inter-experiment drift clamp (`>= 0.65 ppm` hitting `±0.70 ppm` clamp) & picker clamp warning -> `Warn`
    let (mut picked_clamp, _) = simulate_mismatched_bmrb_picked_dataset(&p53, 0.05, 0.02, 616);
    for p in &mut picked_clamp.strip_peaks {
        if p.exp == C13DetectedExp::Cbcaco3D {
            p.c_indirect_ppm += 0.85; // pushes CBCACON_aliph drift against +0.70 ppm clamp
        }
    }
    let (_, res_clamp) = assign_idp_13c_from_picked(&picked_clamp, &p53.sequence, &cfg);
    assert_eq!(
        res_clamp.qc_verdict,
        DatasetQcVerdict::Warn,
        "Clamped solver inter-experiment drift (>=0.65 ppm) must trigger DatasetQcVerdict::Warn!"
    );
    assert!(
        res_clamp.qc_reasons.iter().any(|r| r.contains("clamp")),
        "QC reasons must report clamped carrier drift: {:?}",
        res_clamp.qc_reasons
    );
}

fn simulate_realistic_noise_bmrb_dataset_custom(
    ref_data: &Reference,
    sigma_obs_ca: f64,
    p_missing_3d: f64,
    spurious_rate: f64,
    spurious_height: f64,
    include_cccon: bool,
    seed: u64,
) -> (C13PickedDataset, Vec<i32>) {
    let mut rng = Rng::new(seed ^ 0x55AA_CC33_9911_77EE);
    let seq_len = ref_data.sequence.len();
    let sigma_obs_c = (sigma_obs_ca * 0.58).max(0.018);
    let sigma_obs_n = (sigma_obs_ca * 1.25).max(0.035);
    let sigma_obs_cb = (sigma_obs_ca * 1.18).max(0.030);

    let glob_dn = -0.42;
    let glob_dc = 0.21;
    let glob_dca = -0.27;

    let off_con = (glob_dc + 0.07, glob_dn - 0.11);
    let off_caco = (glob_dc - 0.10, glob_dca + 0.13);
    let off_canco_ca = glob_dca - 0.08;
    let off_cbca_aliph = glob_dca + 0.11;
    let off_cccon_aliph = glob_dca - 0.13;
    let off_ncaco = (glob_dc + 0.15, glob_dn + 0.09, glob_dca + 0.08);

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
                height: 30.0,
                volume: 30.0,
                is_proline_imino_region: nc_spur >= 131.5,
                deconvolved_from_cluster: false,
            });
            truth.push(-1);
            continue;
        };
        let cp_true = ref_data.shifts[&(k - 1, "C".to_string())];
        let nc_true = ref_data.shifts[&(k, "N".to_string())];
        let cp_obs = cp_true + off_con.0 + rng.normal(0.0, sigma_obs_c);
        let nc_obs = nc_true + off_con.1 + rng.normal(0.0, sigma_obs_n);

        con_roots.push(Con2DPeak {
            c_prev_ppm: cp_obs,
            n_curr_ppm: nc_obs,
            height: 100.0 * (1.0 + rng.normal(0.0, 0.10)).clamp(0.65, 1.35),
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
                    c_indirect_ppm: ca_k + off_canco_ca + rng.normal(0.0, sigma_obs_ca),
                    sign: 1,
                    height: 92.0,
                    volume: 92.0,
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
                    c_indirect_ppm: ca_km1 + off_canco_ca + rng.normal(0.0, sigma_obs_ca),
                    sign: 1,
                    height: 70.0,
                    volume: 70.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }
        if let Some(&ca_km1) = ref_data.shifts.get(&(k - 1, "CA".to_string())) {
            if rng.f64() >= p_missing_3d {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: cp_obs,
                    n_ppm: nc_obs,
                    c_indirect_ppm: ca_km1 + off_cbca_aliph + rng.normal(0.0, sigma_obs_ca),
                    sign: 1,
                    height: 88.0,
                    volume: 88.0,
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
                        c_indirect_ppm: cb_km1 + off_cbca_aliph + rng.normal(0.0, sigma_obs_cb),
                        sign: -1,
                        height: -82.0,
                        volume: -82.0,
                        anchor_root_idx: Some(r_idx),
                    });
                }
            }
        }
        if include_cccon {
            let sc_preds = predict_ncidp_shifts(&ref_data.sequence, &IdpSolverConfig::default());
            let mut emitted_any = false;
            for &aname in &sc_atoms {
                if let Some(&sc_v) = ref_data.shifts.get(&(k - 1, aname.to_string())) {
                    if (10.0..=56.0).contains(&sc_v) && rng.f64() >= p_missing_3d {
                        strip_peaks.push(C13Strip3DPeak {
                            exp: C13DetectedExp::Cccon3D,
                            c_prime_ppm: cp_obs,
                            n_ppm: nc_obs,
                            c_indirect_ppm: sc_v + off_cccon_aliph + rng.normal(0.0, sigma_obs_cb),
                            sign: 1,
                            height: 60.0,
                            volume: 60.0,
                            anchor_root_idx: Some(r_idx),
                        });
                        emitted_any = true;
                    }
                }
            }
            if !emitted_any {
                for &sc_mu in &sc_preds[k - 1].sidechain_c {
                    let sc_v = sc_mu + rng.normal(0.0, 0.35);
                    if (10.0..=56.0).contains(&sc_v) && rng.f64() >= p_missing_3d {
                        strip_peaks.push(C13Strip3DPeak {
                            exp: C13DetectedExp::Cccon3D,
                            c_prime_ppm: cp_obs,
                            n_ppm: nc_obs,
                            c_indirect_ppm: sc_v + off_cccon_aliph + rng.normal(0.0, sigma_obs_cb),
                            sign: 1,
                            height: 60.0,
                            volume: 60.0,
                            anchor_root_idx: Some(r_idx),
                        });
                    }
                }
            }
        }
    }

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
                c_prime_ppm: c_k + off_ncaco.0 + rng.normal(0.0, sigma_obs_c),
                n_ppm: nc + off_ncaco.1 + rng.normal(0.0, sigma_obs_n),
                c_indirect_ppm: ca_k + off_ncaco.2 + rng.normal(0.0, sigma_obs_ca),
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
        let ca_obs = ca_i + off_caco.1 + rng.normal(0.0, sigma_obs_ca * 0.90);
        let kind = if is_gly {
            CacoPeakKind::GlyOppositePhase
        } else if ca_obs >= 58.0 {
            CacoPeakKind::SerThrProDownfield
        } else {
            CacoPeakKind::GeneralAliphatic
        };
        caco_peaks.push(Caco2DPeak {
            c_curr_ppm: c_i + off_caco.0 + rng.normal(0.0, sigma_obs_c),
            ca_curr_ppm: ca_obs,
            sign,
            height: 95.0 * (sign as f64),
            volume: 95.0 * (sign as f64),
            kind,
        });
    }

    let n_spurious_3d = ((strip_peaks.len() as f64) * spurious_rate).round() as usize;
    let true_root_indices: Vec<usize> = truth
        .iter()
        .enumerate()
        .filter_map(|(idx, &t)| (t >= 1).then_some(idx))
        .collect();
    let exps_3d = [
        C13DetectedExp::Canco3D,
        C13DetectedExp::Cbcaco3D,
        C13DetectedExp::Ncaco3D,
    ];
    for i in 0..n_spurious_3d {
        let r_idx = true_root_indices[rng.usize(true_root_indices.len())];
        let exp = exps_3d[i % exps_3d.len()];
        let (c_p, c_ind, sgn) = match exp {
            C13DetectedExp::Canco3D => (con_roots[r_idx].c_prev_ppm, 50.0 + rng.f64() * 14.0, 1),
            C13DetectedExp::Cbcaco3D => {
                let is_neg = rng.f64() < 0.5;
                (
                    con_roots[r_idx].c_prev_ppm,
                    if is_neg {
                        24.0 + rng.f64() * 40.0
                    } else {
                        50.0 + rng.f64() * 14.0
                    },
                    if is_neg { -1 } else { 1 },
                )
            }
            _ => (171.5 + rng.f64() * 6.0, 50.0 + rng.f64() * 14.0, 1),
        };
        let anchor_root_idx = if exp == C13DetectedExp::Ncaco3D {
            None
        } else {
            Some(r_idx)
        };
        let n_val = if exp == C13DetectedExp::Ncaco3D {
            con_roots[r_idx].n_curr_ppm + rng.normal(0.0, 0.05)
        } else {
            con_roots[r_idx].n_curr_ppm
        };
        strip_peaks.push(C13Strip3DPeak {
            exp,
            c_prime_ppm: c_p,
            n_ppm: n_val,
            c_indirect_ppm: c_ind,
            sign: sgn,
            height: spurious_height * (sgn as f64),
            volume: spurious_height * (sgn as f64),
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
fn test_no_cccon_5spectrum_3x_noise_and_midband_0_054_ppm_qc_triggers() {
    let root = workspace_root();
    let entries = [
        (
            root.join("data/benchmark/bmr6968.str"),
            "bmr6968",
            288.15,
            6.5,
        ),
        (
            root.join("data/benchmark/bmr15176.str"),
            "bmr15176",
            298.15,
            6.5,
        ),
        (
            root.join("data/benchmark/bmr4375.str"),
            "bmr4375",
            298.15,
            2.0,
        ),
        (
            root.join("data/fixtures/bmr26639_idp_cbp_id4.str"),
            "bmr26639",
            308.15,
            6.5,
        ),
        (
            root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
            "bmr51984",
            298.15,
            6.0,
        ),
    ];

    // Part A1: 5-spectrum `{CON, CACO, CANCO, CBCACON, NCACO}` WITHOUT `3D CCC(CO)N` TOCSY
    // (`include_cccon = false` before 18% spurious injection) on fresh seeds (`99000+`)
    let mut noccc_corr = 0usize;
    let mut noccc_ass = 0usize;
    let mut noccc_hc_c = 0usize;
    let mut noccc_hc_a = 0usize;
    let mut noccc_warn = 0usize;
    let mut noccc_runs = 0usize;
    let mut noccc_cal: Vec<(f64, bool)> = Vec::new();

    // Part A2: Harder stripped-`CCC(CO)N` stress (`include_cccon = true` during 18% ht=95.0
    // spurious injection -> ~30% spurious backbone 3D density, then `Cccon3D` stripped!)
    let mut strip_corr = 0usize;
    let mut strip_ass = 0usize;
    let mut strip_hc_c = 0usize;
    let mut strip_hc_a = 0usize;
    let mut strip_warn = 0usize;
    let mut strip_cal: Vec<(f64, bool)> = Vec::new();

    // Part B: Lower-band `[0.05, 0.08] ppm` realistic matching noise (`sigma_obs_CA = 0.038 ppm`
    // -> `true sigma_seq_CA = 0.054 ppm`) + `15%` equal-height (`95.0`) spurious 3D peaks
    let mut mid_warn = 0usize;
    let mut mid_runs = 0usize;

    for (idx, (path, label, temp_k, ph)) in entries.iter().enumerate() {
        let ref_data = read_star_all_atoms(path, None).unwrap();
        for sd in 0..2u64 {
            let seed_a = 99_000 + (idx as u64) * 37 + sd;
            let (picked_no_cccon, truth_no_cccon) = simulate_realistic_noise_bmrb_dataset_custom(
                &ref_data, 0.055, 0.10, 0.18, 95.0, false, seed_a,
            );
            let cfg_a = IdpSolverConfig {
                temperature_k: *temp_k,
                ph: *ph,
                replicas: 36,
                islands: 4,
                generations: 3,
                steps_per_system: 2_200,
                seed: seed_a,
                ..IdpSolverConfig::default()
            };
            let (sys_a, res_a) =
                assign_idp_13c_from_picked(&picked_no_cccon, &ref_data.sequence, &cfg_a);
            assert!(
                sys_a.iter().all(|s| s.sidechain_c_prev.is_empty()),
                "no-CCCON simulation must have empty sidechain_c_prev"
            );
            noccc_runs += 1;
            if res_a.qc_verdict != DatasetQcVerdict::Pass {
                noccc_warn += 1;
            }
            let ma = evaluate_metrics(&res_a.as_assign_result(), &truth_no_cccon, 0.80);
            noccc_corr += ma.correct;
            noccc_ass += ma.assigned;
            noccc_hc_c += ma.hc_correct;
            noccc_hc_a += ma.hc_assigned;
            for (s, &pos) in res_a.position.iter().enumerate() {
                if pos >= 0 {
                    noccc_cal.push((res_a.confidence[s], pos == truth_no_cccon[s]));
                }
            }

            // Stripped-CCCON + noisy-CACO heavy stress (`Cccon3D` synthesized then stripped so
            // ~30% spurious backbone 3D density + 15% CACO dropout + 15% spurious CACO drives
            // accuracy down into the ~70-78% regime without CCC(CO)N!)
            let (mut picked_strip, truth_strip) = simulate_realistic_noise_bmrb_dataset_custom(
                &ref_data, 0.068, 0.14, 0.18, 95.0, true, seed_a,
            );
            assert!(
                picked_strip
                    .strip_peaks
                    .iter()
                    .any(|p| p.exp == C13DetectedExp::Cccon3D),
                "include_cccon=true must emit Cccon3D peaks"
            );
            // Also verify 6-spectrum WITH-CCCON stays calibrated on the same dataset!
            let (sys_with_ccc, res_with_ccc) =
                assign_idp_13c_from_picked(&picked_strip, &ref_data.sequence, &cfg_a);
            assert!(
                sys_with_ccc.iter().any(|s| !s.sidechain_c_prev.is_empty()),
                "with-CCCON systems must populate sidechain_c_prev"
            );
            assert_ne!(res_with_ccc.qc_verdict, DatasetQcVerdict::Pass);

            // Now strip Cccon3D and add 15% equal-height spurious 2D CACO peaks to stress ~70-78% acc!
            picked_strip
                .strip_peaks
                .retain(|p| p.exp != C13DetectedExp::Cccon3D);
            let mut rng_st = Rng::new(seed_a ^ 0xDEAD_BEEF);
            picked_strip.caco_peaks.retain(|_| rng_st.f64() >= 0.12);
            let n_spur_caco = ((picked_strip.caco_peaks.len() as f64) * 0.15).round() as usize;
            for _ in 0..n_spur_caco {
                picked_strip.caco_peaks.push(Caco2DPeak {
                    c_curr_ppm: 171.5 + rng_st.f64() * 6.0,
                    ca_curr_ppm: 50.0 + rng_st.f64() * 14.0,
                    sign: 1,
                    height: 95.0,
                    volume: 95.0,
                    kind: CacoPeakKind::GeneralAliphatic,
                });
            }
            let (sys_st, res_st) =
                assign_idp_13c_from_picked(&picked_strip, &ref_data.sequence, &cfg_a);
            assert!(sys_st.iter().all(|s| s.sidechain_c_prev.is_empty()));
            if res_st.qc_verdict != DatasetQcVerdict::Pass {
                strip_warn += 1;
            }
            let mst = evaluate_metrics(&res_st.as_assign_result(), &truth_strip, 0.80);
            strip_corr += mst.correct;
            strip_ass += mst.assigned;
            strip_hc_c += mst.hc_correct;
            strip_hc_a += mst.hc_assigned;
            for (s, &pos) in res_st.position.iter().enumerate() {
                if pos >= 0 {
                    strip_cal.push((res_st.confidence[s], pos == truth_strip[s]));
                }
            }

            let seed_b = 88_000 + (idx as u64) * 29 + sd;
            let (picked_mid, _) = simulate_realistic_noise_bmrb_dataset_custom(
                &ref_data, 0.038, 0.10, 0.15, 95.0, true, seed_b,
            );
            let cfg_b = IdpSolverConfig {
                temperature_k: *temp_k,
                ph: *ph,
                replicas: 36,
                islands: 4,
                generations: 3,
                steps_per_system: 2_200,
                seed: seed_b,
                ..IdpSolverConfig::default()
            };
            let (_, res_b) = assign_idp_13c_from_picked(&picked_mid, &ref_data.sequence, &cfg_b);
            mid_runs += 1;
            if res_b.qc_verdict != DatasetQcVerdict::Pass {
                mid_warn += 1;
            } else {
                println!(
                    "MISSED MID-BAND QC WARN on {label} seed={sd}: reasons={:?}",
                    res_b.qc_reasons
                );
            }
        }
    }

    let noccc_acc = (noccc_corr as f64) / (noccc_ass.max(1) as f64);
    let noccc_hc = if noccc_hc_a > 0 {
        (noccc_hc_c as f64) / (noccc_hc_a as f64)
    } else {
        1.0
    };
    let noccc_ece = ProteinReliabilityCurve::from_predictions(&noccc_cal, 0.80).ece;
    let strip_acc = (strip_corr as f64) / (strip_ass.max(1) as f64);
    let strip_hc = if strip_hc_a > 0 {
        (strip_hc_c as f64) / (strip_hc_a as f64)
    } else {
        1.0
    };
    let strip_ece = ProteinReliabilityCurve::from_predictions(&strip_cal, 0.80).ece;
    println!(
        "[No-CCCON 5-Spectrum 3x Noise (18% ht=95 spur)] Acc={:.2}% ({noccc_corr}/{noccc_ass}) HC_prec={:.2}% ({noccc_hc_c}/{noccc_hc_a}) ECE={noccc_ece:.4} QC_warn={noccc_warn}/{noccc_runs}",
        noccc_acc * 100.0,
        noccc_hc * 100.0,
    );
    println!(
        "[Stripped-CCCON Heavy-Spur 5-Spectrum 3x Noise] Acc={:.2}% ({strip_corr}/{strip_ass}) HC_prec={:.2}% ({strip_hc_c}/{strip_hc_a}) ECE={strip_ece:.4} QC_warn={strip_warn}/{noccc_runs}",
        strip_acc * 100.0,
        strip_hc * 100.0,
    );
    println!("[Mid-Band sigma_seq_CA=0.054 ppm + 15% ht=95 spur] QC_warn={mid_warn}/{mid_runs}");

    assert_eq!(
        noccc_warn, noccc_runs,
        "All {noccc_runs} no-CCCON 3x realistic-noise runs must trigger QC Warn!"
    );
    assert_eq!(
        strip_warn, noccc_runs,
        "All {noccc_runs} stripped-CCCON heavy-spurious runs must trigger QC Warn!"
    );
    assert!(
        noccc_hc >= 0.94,
        "No-CCCON 5-spectrum 3x realistic-noise HC precision {:.2}% < 94.0%!",
        noccc_hc * 100.0
    );
    assert!(
        noccc_ece <= 0.06,
        "No-CCCON 5-spectrum 3x realistic-noise ECE {noccc_ece:.4} > 0.06!"
    );
    assert!(
        strip_hc >= 0.94,
        "Stripped-CCCON heavy-spurious 3x realistic-noise HC precision {:.2}% < 94.0%!",
        strip_hc * 100.0
    );
    assert!(
        strip_ece <= 0.06,
        "Stripped-CCCON heavy-spurious 3x realistic-noise ECE {strip_ece:.4} > 0.06!"
    );
    assert_eq!(
        mid_warn, mid_runs,
        "All {mid_runs}/{mid_runs} sigma_seq_CA=0.054 ppm + 15% equal-height spurious runs must trigger QC Warn (got {mid_warn}/{mid_runs})!"
    );
}

#[test]
fn test_idp_offset_and_ncaco_look_elsewhere_null_fpr_and_dss_recovery() {
    let root = workspace_root();
    let p53 = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    let darpp = read_star_all_atoms(root.join("data/benchmark/bmr15176.str"), None).unwrap();

    // 1. Zero-offset null-hypothesis Monte Carlo (`250` noisy runs across two real IDP sequences):
    //    Generate synthetic picked datasets directly around `ncIDP` random-coil predictions with
    //    `d_true = (0, 0, 0, 0)` global offset AND `d_inter = 0` inter-experiment drift, plus
    //    realistic per-residue secondary-shift / conformational scatter (`0.16..0.35 ppm`),
    //    experimental peak jitter, 10% 3D dropout, and 8% spurious 3D peaks (including unanchored
    //    spurious NCACO peaks). Verify look-elsewhere correction keeps false non-zero offset shifts
    //    strictly `< 5%` (`< 12 / 250`) both per-axis and family-wise, with zero false clamp warnings.
    let n_null_trials = 250usize;
    let mut fp_n = 0usize;
    let mut fp_c = 0usize;
    let mut fp_ca = 0usize;
    let mut fp_cb = 0usize;
    let mut fp_any = 0usize;
    let mut fp_reg_warn = 0usize;

    for trial in 0..n_null_trials {
        let (ref_seq, temp_k, ph) = if trial % 2 == 0 {
            (&p53.sequence, 298.15, 6.0)
        } else {
            (&darpp.sequence, 298.15, 6.5)
        };
        let cfg = IdpSolverConfig {
            temperature_k: temp_k,
            ph,
            ..IdpSolverConfig::default()
        };
        let preds = predict_ncidp_shifts(ref_seq, &cfg);
        let mut rng = Rng::new(0x9110_0FF5_E700_u64 + trial as u64);

        // Per-residue zero-mean conformational scatter around ncIDP
        let seq_len = ref_seq.len();
        let n_true: Vec<f64> = (0..seq_len)
            .map(|k| preds[k].n_ppm + rng.normal(0.0, 0.34))
            .collect();
        let c_true: Vec<f64> = (0..seq_len)
            .map(|k| preds[k].c_ppm + rng.normal(0.0, 0.16))
            .collect();
        let ca_true: Vec<f64> = (0..seq_len)
            .map(|k| preds[k].ca_ppm + rng.normal(0.0, 0.20))
            .collect();
        let cb_true: Vec<Option<f64>> = (0..seq_len)
            .map(|k| preds[k].cb_ppm.map(|v| v + rng.normal(0.0, 0.22)))
            .collect();

        let mut con_roots = Vec::with_capacity(seq_len);
        let mut strip_peaks = Vec::new();
        let mut caco_peaks = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let is_gly = ref_seq[i].eq_ignore_ascii_case("GLY");
            let sign = if is_gly { -1 } else { 1 };
            let ca_obs = ca_true[i] + rng.normal(0.0, 0.022);
            caco_peaks.push(Caco2DPeak {
                c_curr_ppm: c_true[i] + rng.normal(0.0, 0.015),
                ca_curr_ppm: ca_obs,
                sign,
                height: 95.0 * (sign as f64),
                volume: 95.0 * (sign as f64),
                kind: if is_gly {
                    CacoPeakKind::GlyOppositePhase
                } else if ca_obs >= 58.0 {
                    CacoPeakKind::SerThrProDownfield
                } else {
                    CacoPeakKind::GeneralAliphatic
                },
            });
        }

        for k in 1..seq_len {
            let r_idx = con_roots.len();
            let cp = c_true[k - 1] + rng.normal(0.0, 0.015);
            let np = n_true[k] + rng.normal(0.0, 0.030);
            con_roots.push(Con2DPeak {
                c_prev_ppm: cp,
                n_curr_ppm: np,
                height: 100.0,
                volume: 100.0,
                is_proline_imino_region: np >= 131.5,
                deconvolved_from_cluster: false,
            });
            if rng.f64() >= 0.10 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: cp,
                    n_ppm: np,
                    c_indirect_ppm: ca_true[k] + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 92.0,
                    volume: 92.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
            if rng.f64() >= 0.10 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Canco3D,
                    c_prime_ppm: cp,
                    n_ppm: np,
                    c_indirect_ppm: ca_true[k - 1] + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 68.0,
                    volume: 68.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
            if rng.f64() >= 0.10 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: cp,
                    n_ppm: np,
                    c_indirect_ppm: ca_true[k - 1] + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 88.0,
                    volume: 88.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
            if let Some(cb_v) = cb_true[k - 1] {
                if rng.f64() >= 0.10 {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cbcaco3D,
                        c_prime_ppm: cp,
                        n_ppm: np,
                        c_indirect_ppm: cb_v + rng.normal(0.0, 0.030),
                        sign: -1,
                        height: -82.0,
                        volume: -82.0,
                        anchor_root_idx: Some(r_idx),
                    });
                }
            }
            if rng.f64() >= 0.10 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Ncaco3D,
                    c_prime_ppm: c_true[k] + rng.normal(0.0, 0.016),
                    n_ppm: n_true[k] + rng.normal(0.0, 0.030),
                    c_indirect_ppm: ca_true[k] + rng.normal(0.0, 0.025),
                    sign: 1,
                    height: 86.0,
                    volume: 86.0,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }

        // Inject 6 unanchored spurious NCACO noise peaks (mimicking volumetric background spikes
        // that previously triggered bogus -0.756 ppm NCACO_unanch_C' clamp drift!)
        for _ in 0..6 {
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Ncaco3D,
                c_prime_ppm: 170.5 + rng.f64() * 8.0,
                n_ppm: 112.0 + rng.f64() * 24.0,
                c_indirect_ppm: 46.0 + rng.f64() * 19.0,
                sign: 1,
                height: 35.0,
                volume: 35.0,
                anchor_root_idx: None,
            });
        }

        let picked = C13PickedDataset {
            con_roots,
            caco_peaks,
            strip_peaks,
        };
        let systems = register_and_build_con_dipeptide_systems(&picked);
        if systems.iter().any(|s| !s.registration_warnings.is_empty()) {
            fp_reg_warn += 1;
        }
        let off = estimate_idp_reference_offsets(&systems, ref_seq, &cfg);
        let nz_n = off.d_n.abs() > 0.0;
        let nz_c = off.d_c.abs() > 0.0;
        let nz_ca = off.d_ca.abs() > 0.0;
        let nz_cb = off.d_cb.abs() > 0.0;
        if nz_n {
            fp_n += 1;
        }
        if nz_c {
            fp_c += 1;
        }
        if nz_ca {
            fp_ca += 1;
        }
        if nz_cb {
            fp_cb += 1;
        }
        if nz_n || nz_c || nz_ca || nz_cb {
            fp_any += 1;
        }
    }

    let rate_any = (fp_any as f64) / (n_null_trials as f64);
    println!(
        "[Zero-Offset Null MC N={n_null_trials}] false non-zero rates: dN={fp_n}/{n_null_trials} ({:.2}%), dC'={fp_c}/{n_null_trials} ({:.2}%), dCA={fp_ca}/{n_null_trials} ({:.2}%), dCB={fp_cb}/{n_null_trials} ({:.2}%), family-wise={fp_any}/{n_null_trials} ({:.2}%), reg_clamp_FP={fp_reg_warn}/{n_null_trials}",
        100.0 * (fp_n as f64) / (n_null_trials as f64),
        100.0 * (fp_c as f64) / (n_null_trials as f64),
        100.0 * (fp_ca as f64) / (n_null_trials as f64),
        100.0 * (fp_cb as f64) / (n_null_trials as f64),
        100.0 * rate_any,
    );
    assert_eq!(
        fp_reg_warn, 0,
        "Unanchored NCACO background noise spikes must trigger 0 false carrier-drift clamp warnings!"
    );
    assert!(
        rate_any < 0.05,
        "Zero-offset null family-wise false non-zero shift rate {:.2}% ({fp_any}/{n_null_trials}) >= 5.0%!",
        rate_any * 100.0
    );

    // 2. Genuine DSS/TSP referencing offset (`+2.00 ppm 13C`, `+1.50 ppm 15N`): 100% recovery across 20 seeds
    let cfg_p53 = IdpSolverConfig {
        temperature_k: 298.15,
        ph: 6.0,
        ..IdpSolverConfig::default()
    };
    let preds_p53 = predict_ncidp_shifts(&p53.sequence, &cfg_p53);
    for sd in 0..20u64 {
        let mut rng = Rng::new(0xD55_7590_u64 + sd);
        let mut systems = Vec::new();
        for k in 1..p53.sequence.len() {
            let mut s = ConDipeptideSystem::new(
                k - 1,
                preds_p53[k - 1].c_ppm + 2.00 + rng.normal(0.0, 0.16),
                preds_p53[k].n_ppm + 1.50 + rng.normal(0.0, 0.34),
                100.0,
            );
            s.ca_curr
                .push(preds_p53[k].ca_ppm + 2.00 + rng.normal(0.0, 0.20));
            s.ca_prev
                .push(preds_p53[k - 1].ca_ppm + 2.00 + rng.normal(0.0, 0.20));
            if let Some(cb) = preds_p53[k - 1].cb_ppm {
                s.cb_prev.push(cb + 2.00 + rng.normal(0.0, 0.22));
            }
            systems.push(s);
        }
        let off = estimate_idp_reference_offsets(&systems, &p53.sequence, &cfg_p53);
        assert!(
            (off.d_n - 1.50).abs() <= 0.18
                && (off.d_c - 2.00).abs() <= 0.10
                && (off.d_ca - 2.00).abs() <= 0.10
                && (off.d_cb - 2.00).abs() <= 0.12,
            "Seed {sd}: genuine DSS/TSP (+2.0 13C, +1.5 15N) offset recovery failed: {off:?}"
        );
    }
}

#[test]
fn test_idp_qc_gate_d_clean_realistic_noise_pass_vs_3x_degraded_warn() {
    let root = workspace_root();
    let entries = [
        (
            root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
            "bmr51984 (p53-TAD-PRR)",
            298.15,
            6.0,
        ),
        (
            root.join("data/benchmark/bmr15176.str"),
            "bmr15176 (DARPP-32)",
            298.15,
            6.5,
        ),
    ];

    let mut clean_pass = 0usize;
    let mut degraded_warn = 0usize;
    let mut total = 0usize;

    for (idx, (path, label, temp_k, ph)) in entries.iter().enumerate() {
        let ref_data = read_star_all_atoms(path, None).unwrap();
        for sd in 0..5u64 {
            total += 1;
            let seed = 71_000 + (idx as u64) * 31 + sd;
            // Clean 1x realistic per-peak 13CA resolution noise (`sigma_peak = 0.036 ppm`, where
            // single-pair sqrt(2)*sigma = 0.051 ppm > legacy 0.034 ppm gate!) with low-amplitude
            // thermal clutter (`ht = 22.0`, filtered cleanly below `0.35 * max_height`).
            let (picked_clean, _) = simulate_realistic_noise_bmrb_dataset_custom(
                &ref_data, 0.036, 0.08, 0.06, 22.0, true, seed,
            );
            let cfg = IdpSolverConfig {
                temperature_k: *temp_k,
                ph: *ph,
                replicas: 28,
                islands: 4,
                generations: 2,
                steps_per_system: 1_800,
                seed,
                ..IdpSolverConfig::default()
            };
            let (_, res_clean) =
                assign_idp_13c_from_picked(&picked_clean, &ref_data.sequence, &cfg);
            if res_clean.qc_verdict == DatasetQcVerdict::Pass {
                clean_pass += 1;
            } else {
                println!(
                    "UNEXPECTED CLEAN WARN on {label} sd={sd}: reasons={:?}",
                    res_clean.qc_reasons
                );
            }

            // Pure 3x degraded per-peak broadening (`sigma_peak = 0.082 ppm`, `emp_rms_seq_ca ~ 0.085..0.105 ppm`,
            // even with 0% spurious peaks!) must reliably trigger `DatasetQcVerdict::Warn` via gate (d).
            let (picked_deg, _) = simulate_realistic_noise_bmrb_dataset_custom(
                &ref_data,
                0.082,
                0.08,
                0.0,
                22.0,
                true,
                seed + 500,
            );
            let (_, res_deg) = assign_idp_13c_from_picked(&picked_deg, &ref_data.sequence, &cfg);
            if res_deg.qc_verdict == DatasetQcVerdict::Warn {
                degraded_warn += 1;
            } else {
                println!(
                    "MISSED DEGRADED 3x WARN on {label} sd={sd}: verdict={} reasons={:?}",
                    res_deg.qc_verdict, res_deg.qc_reasons
                );
            }
        }
    }

    println!(
        "[IDP QC Gate (d) Noise-Floor Audit] clean sigma_peak=0.036 ppm Pass={clean_pass}/{total}, pure 3x sigma_peak=0.082 ppm Warn={degraded_warn}/{total}"
    );
    assert_eq!(
        clean_pass, total,
        "Clean 1x realistic-resolution (sigma_peak=0.036 ppm) runs must Pass QC ({clean_pass}/{total})"
    );
    assert_eq!(
        degraded_warn, total,
        "Degraded 3x broadening (sigma_peak=0.082 ppm) runs must Warn QC ({degraded_warn}/{total})"
    );
}

#[test]
fn test_3seed_multi_protein_volumetric_con_and_3d_e2e_calibration() {
    let root = workspace_root();
    let p53_full = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    let darpp_full = read_star_all_atoms(root.join("data/benchmark/bmr15176.str"), None).unwrap();

    // Two distinct real BMRB IDPs (`p53 TAD+PRR` 62-aa N-terminal domain with 14 Prolines, and
    // `DARPP-32` 65-aa disordered phosphatase-inhibitor domain) x 3 volumetric seeds = 6 full 2D+3D cubes
    let make_sub = |full: &Reference, len: usize| -> Reference {
        let n = len.min(full.sequence.len());
        let mut sub = Reference::new(full.sequence[..n].to_vec(), full.seq_ids[..n].to_vec());
        for ((i, atom), &val) in &full.shifts {
            if *i < n {
                sub.shifts.insert((*i, atom.clone()), val);
            }
        }
        sub
    };

    let proteins = [
        ("bmr51984 (p53-62aa)", make_sub(&p53_full, 62), 298.15, 6.0),
        (
            "bmr15176 (DARPP32-65aa)",
            make_sub(&darpp_full, 65),
            298.15,
            6.5,
        ),
    ];

    let mut exp_offsets = BTreeMap::new();
    exp_offsets.insert(C13DetectedExp::Caco2D, (-0.10, 0.0, 0.12));
    exp_offsets.insert(C13DetectedExp::Canco3D, (0.11, -0.14, -0.09));
    exp_offsets.insert(C13DetectedExp::Cbcaco3D, (-0.09, 0.12, 0.11));
    exp_offsets.insert(C13DetectedExp::Ncaco3D, (0.08, 0.10, -0.08));

    let pcfg = C13PickerConfig::default();
    let mut pooled_cal: Vec<(f64, bool)> = Vec::new();
    let mut tot_corr = 0usize;
    let mut tot_ass = 0usize;
    let mut tot_hc_c = 0usize;
    let mut tot_hc_a = 0usize;

    for (p_idx, (label, sub, temp_k, ph)) in proteins.iter().enumerate() {
        let mut p_corr = 0usize;
        let mut p_ass = 0usize;
        let mut p_hc_c = 0usize;
        let mut p_hc_a = 0usize;
        let mut p_cal: Vec<(f64, bool)> = Vec::new();

        for sd in 0..3u64 {
            let seed = 3100 + (p_idx as u64) * 97 + sd;
            let sim_cfg = IdpCubeSimConfig {
                proton_mhz: 700.0,
                temperature_k: *temp_k,
                ph: *ph,
                snr_2d: 40.0,
                snr_3d: 24.0,
                con_grid: (1024, 1024),
                caco_grid: (512, 512),
                cube_3d_grid: (128, 128, 144),
                exp_offsets: exp_offsets.clone(),
                per_peak_jitter_ppm: Some((0.020, 0.048, 0.055)),
                spurious_con_rate: 0.03,
                spurious_3d_rel_amp: 0.85,
                spurious_strip_artifact_rate: 0.18,
                seed,
                ..IdpCubeSimConfig::default()
            };
            let cubes = simulate_idp_13c_cubes(sub, &sim_cfg);
            let (picked, diags) = pick_simulated_c13_dataset_with_diagnostics(&cubes, &pcfg);
            let picker_warnings: Vec<String> =
                diags.into_iter().filter_map(|d| d.warning).collect();

            let truth: Vec<i32> = picked
                .con_roots
                .iter()
                .map(|r| {
                    cubes
                        .ground_truth_con
                        .iter()
                        .find(|&&(_, cp, np, _, _)| {
                            (r.c_prev_ppm - cp).abs() <= 0.055 && (r.n_curr_ppm - np).abs() <= 0.22
                        })
                        .map(|&(k, _, _, _, _)| k as i32)
                        .unwrap_or(-1)
                })
                .collect();

            let solver_cfg = IdpSolverConfig {
                temperature_k: *temp_k,
                ph: *ph,
                picker_warnings,
                seed,
                ..IdpSolverConfig::default()
            };
            let (_, res) = assign_idp_13c_from_picked(&picked, &sub.sequence, &solver_cfg);
            let m = evaluate_metrics(&res.as_assign_result(), &truth, 0.80);
            p_corr += m.correct;
            p_ass += m.assigned;
            p_hc_c += m.hc_correct;
            p_hc_a += m.hc_assigned;
            tot_corr += m.correct;
            tot_ass += m.assigned;
            tot_hc_c += m.hc_correct;
            tot_hc_a += m.hc_assigned;

            for (s, &pos) in res.position.iter().enumerate() {
                if pos >= 0 {
                    let ok = pos == truth[s];
                    p_cal.push((res.confidence[s], ok));
                    pooled_cal.push((res.confidence[s], ok));
                }
            }
        }
        let p_rel = ProteinReliabilityCurve::from_predictions(&p_cal, 0.80);
        println!(
            "[3-Seed Volumetric E2E {label}] Acc={:.2}% ({p_corr}/{p_ass}) HC_prec={:.2}% ({p_hc_c}/{p_hc_a}) ECE={:.4}",
            100.0 * (p_corr as f64) / (p_ass.max(1) as f64),
            100.0 * (p_hc_c as f64) / (p_hc_a.max(1) as f64),
            p_rel.ece,
        );
    }

    let pooled_acc = (tot_corr as f64) / (tot_ass.max(1) as f64);
    let pooled_hc = (tot_hc_c as f64) / (tot_hc_a.max(1) as f64);
    let pooled_rel = ProteinReliabilityCurve::from_predictions(&pooled_cal, 0.80);
    println!(
        "[Pooled 6-Run (3-Seed x 2-Protein) Volumetric 2D CON + 3D E2E] Acc={:.2}% ({tot_corr}/{tot_ass}) HC_prec={:.2}% ({tot_hc_c}/{tot_hc_a}) ECE={:.4}",
        pooled_acc * 100.0,
        pooled_hc * 100.0,
        pooled_rel.ece,
    );
    assert!(
        pooled_acc >= 0.90,
        "Pooled 3-seed volumetric E2E accuracy {:.2}% < 90.0%!",
        pooled_acc * 100.0
    );
    assert!(
        pooled_hc >= 0.95,
        "Pooled 3-seed volumetric E2E HC precision {:.2}% < 95.0%!",
        pooled_hc * 100.0
    );
    assert!(
        pooled_rel.ece <= 0.06,
        "Pooled 3-seed volumetric E2E ECE {:.4} > 0.06!",
        pooled_rel.ece
    );
}
