pub use nmr_core::*;

use std::path::{Path, PathBuf};

use crate::dihedrals::{
    periodic_angle_abs_diff_deg, predict_dihedrals, validate_dihedrals_against_torsions,
    SecondaryStructure,
};
use crate::nmrstar::read_star_all_atoms;
use crate::noesy::{
    compute_chain_torsions, match_initial_adrs, parse_pdb_first_model, parse_pdb_str_first_model,
    pick_noesy_3d, render_noesy_3d_spectrum, simulate_noesy_peaks, summarize_adrs, v_dist,
    AdrConfig, NoesySimConfig,
};
use crate::picker::PickerConfig;
use crate::rng::Rng;
use crate::structure::{
    build_backbone_from_torsions, calculate_structure_iterative, evaluate_energy_and_gradients,
    format_pdb_ensemble, kabsch_rmsd, kabsch_transform, validate_ensemble, EnergyWeights,
    StructureCalcConfig, BOND_CA_C, BOND_C_N, BOND_C_O, BOND_N_CA, BOND_N_H,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_bmrb_reference(bmrb_file: &str) -> nmr_core::nmrstar::Reference {
    let bench = workspace_root().join("data/benchmark").join(bmrb_file);
    read_star_all_atoms(&bench, None).expect("failed to parse benchmark BMRB file")
}

#[test]
fn test_nerf_covalent_geometry_and_torsion_roundtrip() {
    let seq: Vec<String> = [
        "MET", "ALA", "GLY", "PRO", "VAL", "LEU", "TRP", "SER", "ILE", "LYS",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let seq_ids: Vec<i32> = (1..=10).collect();
    let phi = vec![
        -60.0, -63.5, 78.0, -62.0, -124.0, -65.0, -118.0, -75.0, -121.0, -64.0,
    ];
    let psi = vec![
        135.0, -42.0, 18.0, 144.0, 136.0, -41.0, 132.0, 145.0, 134.0, -40.0,
    ];
    let omega = vec![180.0; 10];

    let chain = build_backbone_from_torsions(&seq, &seq_ids, &phi, &psi, &omega);
    assert_eq!(chain.residues.len(), 10);

    for i in 0..10 {
        let r = &chain.residues[i];
        assert!((v_dist(r.n, r.ca) - BOND_N_CA).abs() < 1e-10);
        assert!((v_dist(r.ca, r.c) - BOND_CA_C).abs() < 1e-10);
        assert!((v_dist(r.c, r.o) - BOND_C_O).abs() < 1e-10);

        if r.res_name == "GLY" {
            assert!(r.cb.is_none());
        } else {
            assert!((v_dist(r.ca, r.cb.unwrap()) - 1.526).abs() < 1e-10);
        }

        if i > 0 {
            assert!((v_dist(chain.residues[i - 1].c, r.n) - BOND_C_N).abs() < 1e-10);
            if r.res_name == "PRO" {
                assert!(r.h.is_none());
            } else {
                assert!((v_dist(r.n, r.h.unwrap()) - BOND_N_H).abs() < 1e-10);
            }
        }
    }

    let recovered = compute_chain_torsions(&chain);
    for i in 0..10 {
        if i > 0 {
            let err_phi = periodic_angle_abs_diff_deg(recovered[i].2.unwrap(), phi[i]);
            assert!(err_phi < 1e-9, "phi mismatch at {i}: {err_phi}");
        }
        if i + 1 < 10 {
            let err_psi = periodic_angle_abs_diff_deg(recovered[i].3.unwrap(), psi[i]);
            assert!(err_psi < 1e-9, "psi mismatch at {i}: {err_psi}");
        }
    }
}

#[test]
fn test_analytical_torsion_gradients_match_finite_difference() {
    let seq: Vec<String> = [
        "ALA", "VAL", "GLY", "PRO", "LEU", "ILE", "PHE", "SER", "GLU", "LYS",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let seq_ids: Vec<i32> = (1..=10).collect();

    let mut rng = Rng::new(77);
    let phi_rad: Vec<f64> = (0..10).map(|_| rng.uniform(-2.6, 2.2)).collect();
    let psi_rad: Vec<f64> = (0..10).map(|_| rng.uniform(-2.6, 2.6)).collect();
    let omg_rad = vec![std::f64::consts::PI; 10];

    // Build synthetic ADR & dihedral restraints so every potential term is active
    let mut ref_data = nmr_core::nmrstar::Reference::new(seq.clone(), seq_ids.clone());
    for i in 0..10 {
        ref_data.shifts.insert((i, "CA".to_string()), 57.5);
        ref_data.shifts.insert((i, "CB".to_string()), 31.0);
        ref_data.shifts.insert((i, "C".to_string()), 177.2);
        ref_data
            .shifts
            .insert((i, "H".to_string()), 8.15 + 0.04 * (i as f64));
        ref_data
            .shifts
            .insert((i, "N".to_string()), 120.0 + 0.5 * (i as f64));
    }
    let dihs = predict_dihedrals(&ref_data);

    let adrs = vec![
        crate::noesy::AmbiguousDistanceRestraint {
            peak_index: 0,
            exp: "NOESY15N".to_string(),
            lower_bound: 1.8,
            upper_bound: 3.2,
            candidates: vec![
                crate::noesy::NoeCandidatePair {
                    res_a: 1,
                    atom_a: crate::noesy::NoeProton::AmideH,
                    res_b: 8,
                    atom_b: crate::noesy::NoeProton::AlphaHA,
                    shift_score: 0.8,
                    network_score: 1.0,
                    weight: 0.6,
                },
                crate::noesy::NoeCandidatePair {
                    res_a: 2,
                    atom_a: crate::noesy::NoeProton::AlphaHA,
                    res_b: 7,
                    atom_b: crate::noesy::NoeProton::BetaQB,
                    shift_score: 0.6,
                    network_score: 0.8,
                    weight: 0.4,
                },
            ],
            all_candidates: Vec::new(),
            is_unambiguous: false,
        },
        crate::noesy::AmbiguousDistanceRestraint {
            peak_index: 1,
            exp: "NOESY13C".to_string(),
            lower_bound: 1.8,
            upper_bound: 2.8,
            candidates: vec![crate::noesy::NoeCandidatePair {
                res_a: 0,
                atom_a: crate::noesy::NoeProton::BetaQB,
                res_b: 9,
                atom_b: crate::noesy::NoeProton::BetaQB,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 1.0,
            }],
            all_candidates: Vec::new(),
            is_unambiguous: true,
        },
    ];

    let weights = EnergyWeights {
        w_noe: 2.3,
        noe_soft_window: 2.5,
        w_dihedral: 1.4,
        w_vdw: 1.2,
        vdw_scale: 1.0,
        w_rama: 0.9,
        w_rg: 0.25,
    };

    let mut gp = vec![0.0; 10];
    let mut gs = vec![0.0; 10];
    evaluate_energy_and_gradients(
        &seq, &seq_ids, &phi_rad, &psi_rad, &omg_rad, &adrs, &dihs, &weights, &mut gp, &mut gs,
    );

    let eps = 1e-6;
    let mut dummy_p = vec![0.0; 10];
    let mut dummy_s = vec![0.0; 10];

    for i in 1..10 {
        let mut p_plus = phi_rad.clone();
        let mut p_minus = phi_rad.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;
        let (_, e_plus) = evaluate_energy_and_gradients(
            &seq,
            &seq_ids,
            &p_plus,
            &psi_rad,
            &omg_rad,
            &adrs,
            &dihs,
            &weights,
            &mut dummy_p,
            &mut dummy_s,
        );
        let (_, e_minus) = evaluate_energy_and_gradients(
            &seq,
            &seq_ids,
            &p_minus,
            &psi_rad,
            &omg_rad,
            &adrs,
            &dihs,
            &weights,
            &mut dummy_p,
            &mut dummy_s,
        );
        let fd = (e_plus.total - e_minus.total) / (2.0 * eps);
        let rel_err = (gp[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel_err < 1e-5,
            "dE/dphi[{i}] analytical={} vs fd={} (rel_err={})",
            gp[i],
            fd,
            rel_err
        );
    }

    for i in 0..9 {
        let mut s_plus = psi_rad.clone();
        let mut s_minus = psi_rad.clone();
        s_plus[i] += eps;
        s_minus[i] -= eps;
        let (_, e_plus) = evaluate_energy_and_gradients(
            &seq,
            &seq_ids,
            &phi_rad,
            &s_plus,
            &omg_rad,
            &adrs,
            &dihs,
            &weights,
            &mut dummy_p,
            &mut dummy_s,
        );
        let (_, e_minus) = evaluate_energy_and_gradients(
            &seq,
            &seq_ids,
            &phi_rad,
            &s_minus,
            &omg_rad,
            &adrs,
            &dihs,
            &weights,
            &mut dummy_p,
            &mut dummy_s,
        );
        let fd = (e_plus.total - e_minus.total) / (2.0 * eps);
        let rel_err = (gs[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel_err < 1e-5,
            "dE/dpsi[{i}] analytical={} vs fd={} (rel_err={})",
            gs[i],
            fd,
            rel_err
        );
    }
}

#[test]
fn test_kabsch_rmsd_rigid_invariance_and_proper_rotation() {
    let pdb_path = workspace_root().join("data/pdb_refs/1d3z.pdb");
    let pdb = parse_pdb_first_model(&pdb_path).expect("parse 1d3z.pdb");
    let orig: Vec<[f64; 3]> = pdb.residues.iter().flat_map(|r| [r.n, r.ca, r.c]).collect();

    // Apply an arbitrary proper 3D Euler rotation + large translation
    let (a, b, g) = (0.73f64, -1.12f64, 2.09f64);
    let (sa, ca) = a.sin_cos();
    let (sb, cb) = b.sin_cos();
    let (sg, cg) = g.sin_cos();
    let r_mat = [
        [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
        [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
        [-sb, cb * sg, cb * cg],
    ];
    let shift = [142.5, -87.3, 63.9];

    let transformed: Vec<[f64; 3]> = orig
        .iter()
        .map(|p| {
            [
                r_mat[0][0] * p[0] + r_mat[0][1] * p[1] + r_mat[0][2] * p[2] + shift[0],
                r_mat[1][0] * p[0] + r_mat[1][1] * p[1] + r_mat[1][2] * p[2] + shift[1],
                r_mat[2][0] * p[0] + r_mat[2][1] * p[1] + r_mat[2][2] * p[2] + shift[2],
            ]
        })
        .collect();

    let (rot_rec, _, _, rmsd) = kabsch_transform(&transformed, &orig);
    assert!(
        rmsd < 1e-11,
        "expected zero RMSD under rigid transform, got {rmsd}"
    );

    let det = rot_rec[0][0] * (rot_rec[1][1] * rot_rec[2][2] - rot_rec[1][2] * rot_rec[2][1])
        - rot_rec[0][1] * (rot_rec[1][0] * rot_rec[2][2] - rot_rec[1][2] * rot_rec[2][0])
        + rot_rec[0][2] * (rot_rec[1][0] * rot_rec[2][1] - rot_rec[1][1] * rot_rec[2][0]);
    assert!((det - 1.0).abs() < 1e-11, "expected det(R)=+1, got {det}");

    // Mirror reflection of chiral ubiquitin must NOT give zero RMSD under proper rotation!
    let mirrored: Vec<[f64; 3]> = orig.iter().map(|p| [-p[0], -p[1], -p[2]]).collect();
    let (rot_mir, _, _, rmsd_mir) = kabsch_transform(&mirrored, &orig);
    let det_mir = rot_mir[0][0] * (rot_mir[1][1] * rot_mir[2][2] - rot_mir[1][2] * rot_mir[2][1])
        - rot_mir[0][1] * (rot_mir[1][0] * rot_mir[2][2] - rot_mir[1][2] * rot_mir[2][0])
        + rot_mir[0][2] * (rot_mir[1][0] * rot_mir[2][1] - rot_mir[1][1] * rot_mir[2][0]);
    assert!(
        (det_mir - 1.0).abs() < 1e-11,
        "mirror alignment must keep det(R)=+1, got {det_mir}"
    );
    assert!(
        rmsd_mir > 4.0,
        "mirror ubiquitin should have large proper RMSD (>4 A), got {rmsd_mir}"
    );
    assert!((kabsch_rmsd(&transformed, &orig)).abs() < 1e-11);
}

#[test]
fn test_dihedral_predictions_against_experimental_pdbs() {
    let cases = [
        ("1D3Z", "1d3z.pdb", "bmr17769.str", 16.0, 16.0),
        ("1ENH", "1enh.pdb", "bmr15536.str", 12.0, 12.0),
        ("1MJC", "1mjc.pdb", "bmr4296.str", 24.0, 24.0),
    ];

    for (label, pdb_file, bmrb_file, max_phi_mae, max_psi_mae) in cases {
        let pdb = parse_pdb_first_model(workspace_root().join("data/pdb_refs").join(pdb_file))
            .expect("parse PDB");
        let shifts = load_bmrb_reference(bmrb_file);
        let preds = predict_dihedrals(&shifts);
        let exp_tors = compute_chain_torsions(&pdb);
        let metrics = validate_dihedrals_against_torsions(&preds, &exp_tors);

        let n_hel = preds
            .iter()
            .filter(|p| p.ss_class == SecondaryStructure::Helix)
            .count();
        let n_sht = preds
            .iter()
            .filter(|p| p.ss_class == SecondaryStructure::Sheet)
            .count();

        println!(
            "[DIHEDRALS {label}] n_all={} n_ord={} (H={}, E={}) | Ordered MAE: phi={:.2} deg, psi={:.2} deg | All MAE: phi={:.2} deg, psi={:.2} deg | Q3={:.1}%",
            metrics.n_compared_all,
            metrics.n_compared_ordered,
            n_hel,
            n_sht,
            metrics.phi_mae_ordered_deg,
            metrics.psi_mae_ordered_deg,
            metrics.phi_mae_all_deg,
            metrics.psi_mae_all_deg,
            metrics.q3_accuracy * 100.0
        );

        assert!(metrics.n_compared_ordered >= 20);
        assert!(
            metrics.phi_mae_ordered_deg <= max_phi_mae,
            "{label} ordered phi MAE too high: {:.2}",
            metrics.phi_mae_ordered_deg
        );
        assert!(
            metrics.psi_mae_ordered_deg <= max_psi_mae,
            "{label} ordered psi MAE too high: {:.2}",
            metrics.psi_mae_ordered_deg
        );
    }
}

#[test]
fn test_3d_noesy_volumetric_rendering_and_picking() {
    let pdb = parse_pdb_first_model(workspace_root().join("data/pdb_refs/1enh.pdb")).unwrap();
    let shifts = load_bmrb_reference("bmr15536.str");
    let sim_peaks = simulate_noesy_peaks(
        &pdb,
        &shifts,
        &NoesySimConfig {
            cutoff_angstrom: 4.5,
            include_15n_edited: true,
            include_13c_edited: false,
            dropout_rate: 0.0,
            ..Default::default()
        },
    );
    assert!(sim_peaks.len() >= 80);

    let spec = render_noesy_3d_spectrum(&sim_peaks[..25], "NOESY15N", 64, 48, 96, 2.0, 99);
    let picked = pick_noesy_3d(
        &spec,
        &[],
        &PickerConfig {
            snr_threshold: 6.0,
            ..Default::default()
        },
    );
    assert!(
        picked.len() >= 15,
        "expected volumetric 3D NOESY picker to recover crosspeaks, got {}",
        picked.len()
    );
    assert!(picked.iter().all(|p| p.exp == "NOESY15N" && p.h2.is_some()));
}

#[test]
fn test_full_iterative_adr_and_folding_1enh_and_1d3z() {
    let t0 = std::time::Instant::now();

    // 1. 1ENH (Engrailed homeodomain, 61 aa BMRB 15536 / 54 ordered X-ray residues, 3-helix bundle)
    let pdb_1enh = parse_pdb_first_model(workspace_root().join("data/pdb_refs/1enh.pdb")).unwrap();
    let shifts_1enh = load_bmrb_reference("bmr15536.str");
    let peaks_1enh = simulate_noesy_peaks(&pdb_1enh, &shifts_1enh, &NoesySimConfig::default());

    let cyc0_adrs_1enh = match_initial_adrs(&peaks_1enh, &shifts_1enh, &AdrConfig::default());
    let cyc0_sum_1enh = summarize_adrs(&cyc0_adrs_1enh);

    let cfg = StructureCalcConfig::default();
    let res_1enh = calculate_structure_iterative(&shifts_1enh, &peaks_1enh, &cfg);
    let final_sum_1enh = summarize_adrs(&res_1enh.final_adrs);
    let val_1enh = validate_ensemble(
        &res_1enh.conformers,
        &res_1enh.final_adrs,
        &res_1enh.dihedral_restraints,
        Some(&pdb_1enh),
    );

    println!(
        "[FOLD 1ENH] peaks={} | ADRs: total={} unambig {} -> {} (LR unambig {}) | Ordered RMSD: mean={:.2} A, lowE={:.2} A, best={:.2} A, all={:.2} A | Pairwise={:.2} A | Rama fav+allow={:.1}% (fav={:.1}%, out={:.1}%) | Viols >0.5A={:.2}, >5deg={:.2}",
        peaks_1enh.len(),
        final_sum_1enh.total_restraints,
        cyc0_sum_1enh.unambiguous_restraints,
        final_sum_1enh.unambiguous_restraints,
        final_sum_1enh.long_range_unambiguous,
        val_1enh.mean_backbone_rmsd_ordered_to_ref.unwrap(),
        val_1enh.lowest_energy_rmsd_ordered_to_ref.unwrap(),
        val_1enh.best_conformer_rmsd_ordered_to_ref.unwrap(),
        val_1enh.mean_backbone_rmsd_all_to_ref.unwrap(),
        val_1enh.ensemble_pairwise_rmsd_ordered,
        val_1enh.ramachandran_favored_pct + val_1enh.ramachandran_allowed_pct,
        val_1enh.ramachandran_favored_pct,
        val_1enh.ramachandran_outlier_pct,
        val_1enh.mean_distance_violations_gt_0_5a,
        val_1enh.mean_dihedral_violations_gt_5deg,
    );

    // 2. 1D3Z (Human ubiquitin, 76 aa BMRB 17769, alpha/beta 5-stranded mixed sheet + helix)
    let pdb_1d3z = parse_pdb_first_model(workspace_root().join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_1d3z = load_bmrb_reference("bmr17769.str");
    let peaks_1d3z = simulate_noesy_peaks(&pdb_1d3z, &shifts_1d3z, &NoesySimConfig::default());

    let cyc0_adrs_1d3z = match_initial_adrs(&peaks_1d3z, &shifts_1d3z, &AdrConfig::default());
    let cyc0_sum_1d3z = summarize_adrs(&cyc0_adrs_1d3z);

    let res_1d3z = calculate_structure_iterative(&shifts_1d3z, &peaks_1d3z, &cfg);
    let final_sum_1d3z = summarize_adrs(&res_1d3z.final_adrs);
    let val_1d3z = validate_ensemble(
        &res_1d3z.conformers,
        &res_1d3z.final_adrs,
        &res_1d3z.dihedral_restraints,
        Some(&pdb_1d3z),
    );

    println!(
        "[FOLD 1D3Z] peaks={} | ADRs: total={} unambig {} -> {} (LR unambig {}) | Ordered RMSD: mean={:.2} A, lowE={:.2} A, best={:.2} A, all={:.2} A | Pairwise={:.2} A | Rama fav+allow={:.1}% (fav={:.1}%, out={:.1}%) | Viols >0.5A={:.2}, >5deg={:.2}",
        peaks_1d3z.len(),
        final_sum_1d3z.total_restraints,
        cyc0_sum_1d3z.unambiguous_restraints,
        final_sum_1d3z.unambiguous_restraints,
        final_sum_1d3z.long_range_unambiguous,
        val_1d3z.mean_backbone_rmsd_ordered_to_ref.unwrap(),
        val_1d3z.lowest_energy_rmsd_ordered_to_ref.unwrap(),
        val_1d3z.best_conformer_rmsd_ordered_to_ref.unwrap(),
        val_1d3z.mean_backbone_rmsd_all_to_ref.unwrap(),
        val_1d3z.ensemble_pairwise_rmsd_ordered,
        val_1d3z.ramachandran_favored_pct + val_1d3z.ramachandran_allowed_pct,
        val_1d3z.ramachandran_favored_pct,
        val_1d3z.ramachandran_outlier_pct,
        val_1d3z.mean_distance_violations_gt_0_5a,
        val_1d3z.mean_dihedral_violations_gt_5deg,
    );

    println!(
        "Total folding benchmark elapsed: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    // Multi-model PDB writer round-trip verification
    let pdb_ensemble_text = format_pdb_ensemble(&res_1enh.conformers, "1ENH FOLDED ENSEMBLE");
    assert_eq!(pdb_ensemble_text.matches("MODEL ").count(), 10);
    assert_eq!(pdb_ensemble_text.matches("ENDMDL").count(), 10);
    let parsed_back = parse_pdb_str_first_model(&pdb_ensemble_text).expect("re-parse written PDB");
    assert_eq!(parsed_back.residues.len(), 61);

    // Quality assertions
    assert!(
        final_sum_1enh.unambiguous_restraints > cyc0_sum_1enh.unambiguous_restraints,
        "iterative ADR must increase unambiguous restraints on 1ENH"
    );
    assert!(
        final_sum_1d3z.unambiguous_restraints > cyc0_sum_1d3z.unambiguous_restraints,
        "iterative ADR must increase unambiguous restraints on 1D3Z"
    );
    assert!(
        val_1enh.ramachandran_favored_pct + val_1enh.ramachandran_allowed_pct >= 92.0,
        "1ENH Ramachandran favored+allowed below 92%"
    );
    assert!(
        val_1d3z.ramachandran_favored_pct + val_1d3z.ramachandran_allowed_pct >= 92.0,
        "1D3Z Ramachandran favored+allowed below 92%"
    );
    assert!(
        val_1enh.lowest_energy_rmsd_ordered_to_ref.unwrap() < 3.5,
        "1ENH lowest-energy ordered backbone RMSD too high: {:.2} A",
        val_1enh.lowest_energy_rmsd_ordered_to_ref.unwrap()
    );
    assert!(
        val_1d3z.lowest_energy_rmsd_ordered_to_ref.unwrap() < 3.5,
        "1D3Z lowest-energy ordered backbone RMSD too high: {:.2} A",
        val_1d3z.lowest_energy_rmsd_ordered_to_ref.unwrap()
    );
}

#[test]
fn test_5atom_benchmark_star_portability() {
    // Verify CI portability when only 5-atom (H, N, CA, CB, C) data/benchmark/*.str files are present
    for (label, pdb_file, star_file, max_mae) in [
        ("1D3Z", "1d3z.pdb", "bmr17769.str", 17.5),
        ("1ENH", "1enh.pdb", "bmr15536.str", 12.0),
        ("1MJC", "1mjc.pdb", "bmr4296.str", 24.0),
    ] {
        let pdb = parse_pdb_first_model(workspace_root().join("data/pdb_refs").join(pdb_file))
            .expect("parse PDB");
        let shifts = read_star_all_atoms(
            workspace_root().join("data/benchmark").join(star_file),
            None,
        )
        .expect("parse 5-atom benchmark STAR");
        let preds = predict_dihedrals(&shifts);
        let metrics = validate_dihedrals_against_torsions(&preds, &compute_chain_torsions(&pdb));
        assert!(
            metrics.phi_mae_ordered_deg <= max_mae && metrics.psi_mae_ordered_deg <= max_mae,
            "5-atom {label} ordered MAE exceeded bound: phi={:.2}, psi={:.2}",
            metrics.phi_mae_ordered_deg,
            metrics.psi_mae_ordered_deg
        );
    }

    // Also verify folding from 5-atom benchmark STAR (CI environment without scratch/bmrb_raw)
    let pdb_1enh = parse_pdb_first_model(workspace_root().join("data/pdb_refs/1enh.pdb")).unwrap();
    let shifts_5at =
        read_star_all_atoms(workspace_root().join("data/benchmark/bmr15536.str"), None).unwrap();
    let peaks_5at = simulate_noesy_peaks(&pdb_1enh, &shifts_5at, &NoesySimConfig::default());
    let res_5at =
        calculate_structure_iterative(&shifts_5at, &peaks_5at, &StructureCalcConfig::default());
    let val_5at = validate_ensemble(
        &res_5at.conformers,
        &res_5at.final_adrs,
        &res_5at.dihedral_restraints,
        Some(&pdb_1enh),
    );
    println!(
        "[FOLD 1ENH 5-ATOM CI] lowE={:.2} A, best={:.2} A, Rama={:.1}%",
        val_5at.lowest_energy_rmsd_ordered_to_ref.unwrap(),
        val_5at.best_conformer_rmsd_ordered_to_ref.unwrap(),
        val_5at.ramachandran_favored_pct + val_5at.ramachandran_allowed_pct
    );
    assert!(val_5at.lowest_energy_rmsd_ordered_to_ref.unwrap() < 3.5);
    assert!(val_5at.ramachandran_favored_pct + val_5at.ramachandran_allowed_pct >= 92.0);

    let pdb_1d3z = parse_pdb_first_model(workspace_root().join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_5at_ubq =
        read_star_all_atoms(workspace_root().join("data/benchmark/bmr17769.str"), None).unwrap();
    let peaks_5at_ubq =
        simulate_noesy_peaks(&pdb_1d3z, &shifts_5at_ubq, &NoesySimConfig::default());
    let res_5at_ubq = calculate_structure_iterative(
        &shifts_5at_ubq,
        &peaks_5at_ubq,
        &StructureCalcConfig::default(),
    );
    let val_5at_ubq = validate_ensemble(
        &res_5at_ubq.conformers,
        &res_5at_ubq.final_adrs,
        &res_5at_ubq.dihedral_restraints,
        Some(&pdb_1d3z),
    );
    println!(
        "[FOLD 1D3Z 5-ATOM CI] lowE={:.2} A, best={:.2} A, Rama={:.1}%",
        val_5at_ubq.lowest_energy_rmsd_ordered_to_ref.unwrap(),
        val_5at_ubq.best_conformer_rmsd_ordered_to_ref.unwrap(),
        val_5at_ubq.ramachandran_favored_pct + val_5at_ubq.ramachandran_allowed_pct
    );
    assert!(val_5at_ubq.lowest_energy_rmsd_ordered_to_ref.unwrap() < 3.5);
    assert!(val_5at_ubq.ramachandran_favored_pct + val_5at_ubq.ramachandran_allowed_pct >= 92.0);
}
