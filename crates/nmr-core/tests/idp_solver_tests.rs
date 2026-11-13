//! Integration & non-circular BMRB IDP benchmark tests for `Phase 11` `T11.2`:
//! Continuous through-Proline `13C`-detected IDP solver (`ConDipeptideSystem`, `ncIDP` temperature/pH
//! random-coil library, permutation-invariant unlabelled `(dN, dC', dCA, dCB)` offset estimation,
//! multi-island evolutionary + detailed-balance annealing, `ProteinReliabilityCurve` `ECE <= 0.06`,
//! explicit `ABSTAIN`, and model-mismatched real BMRB IDP benchmarks).

use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, simulate_idp_13c_cubes, C13DetectedExp,
    C13PickedDataset, C13PickerConfig, C13Strip3DPeak, Caco2DPeak, CacoPeakKind, Con2DPeak,
    IdpCubeSimConfig,
};
use nmr_core::evaluate::score as evaluate_metrics;
use nmr_core::idp_solver::{
    assign_idp_13c_cpu, assign_idp_13c_from_picked, estimate_idp_reference_offsets,
    idp_to_reference, predict_ncidp_shifts, register_and_build_con_dipeptide_systems,
    ConDipeptideSystem, IdpSolverConfig,
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
