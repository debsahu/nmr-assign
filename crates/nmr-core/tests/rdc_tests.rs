use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use nmr_core::dihedrals::predict_dihedrals;
use nmr_core::nef::{read_nef_str, write_nef_str, NefDocument, NefRdcRestraintList};
use nmr_core::nmrstar::read_star_all_atoms;
use nmr_core::noesy::{
    parse_pdb_first_model, reconstruct_ca_substituents, simulate_noesy_peaks, v_dot, v_norm,
    v_scale, v_sub, NoesySimConfig, PdbChain, PdbResidue,
};
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, cornilescu_q, count_unresolved_rdc_restraints, evaluate_rdc_validation,
    fit_saupe_tensor_svd, format_rdc_tsv, normalize_rdc_sign_convention, parse_rdc_tsv_str,
    parse_rdc_tsv_str_with_convention, partition_rdc_free_set, partition_rdc_free_set_with_mode,
    rdc_peptide_plane_id, rdc_q_is_better, read_rdc_tsv, saupe_matrix_from_pas, RdcBondType,
    RdcFreeStratification, RdcMediumSet, RdcRestraint, RdcSignConvention, DMAX_CAHA_HZ, DMAX_NH_HZ,
    R_CAHA_EFF_A, R_CAHA_GEOM_A, R_NH_EFF_A, R_NH_GEOM_A,
};
use nmr_core::report::{generate_quality_report_html, ReportInput};
use nmr_core::rng::Rng;
use nmr_core::structure::{
    backbone_rmsd_seq_range, build_backbone_from_torsions,
    calculate_structure_with_rdcs_and_dihedrals, evaluate_energy_and_gradients_with_rdcs,
    evaluate_rdc_torsion_energy_and_gradients_fixed_saupe, fold_from_experimental_noes_and_rdcs,
    read_exp_noe_tsv, validate_ensemble, EnergyWeights, StructureCalcConfig, BOND_CA_HA,
};
use nmr_core::torsion_db::{predict_torsions_with_firewall, HomologFirewallQuery};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn build_diverse_test_chain() -> (Vec<String>, Vec<i32>, PdbChain) {
    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "ILE",
        "THR", "LEU", "GLU", "VAL", "GLU", "PRO", "SER",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let seq_ids: Vec<i32> = (1..=20).collect();
    let phi = vec![
        -60.0, -122.0, -131.0, -118.0, -125.0, -95.0, -82.0, 62.0, -78.0, 82.0, -92.0, -120.0,
        -128.0, -115.0, -124.0, -64.0, -61.0, -65.0, -62.0, -75.0,
    ];
    let psi = vec![
        135.0, 138.0, 152.0, 140.0, 128.0, 125.0, -16.0, 28.0, 145.0, 15.0, 136.0, 132.0, 144.0,
        138.0, 129.0, -42.0, -44.0, -39.0, 144.0, 130.0,
    ];
    let omega = vec![180.0; 20];
    let chain = build_backbone_from_torsions(&seq, &seq_ids, &phi, &psi, &omega);
    (seq, seq_ids, chain)
}

fn synthesize_all5_rdc_medium(
    chain: &PdbChain,
    medium_id: &str,
    s_true: &[[f64; 3]; 3],
) -> RdcMediumSet {
    let mut restraints = Vec::new();
    let l = chain.residues.len();
    for i in 0..l {
        let r = &chain.residues[i];
        // 1. 1D_NH
        if let Some(h) = r.h {
            let u = v_scale(v_sub(h, r.n), 1.0 / v_norm(v_sub(h, r.n)));
            let dmax = RdcBondType::NH.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "N".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "H".to_string(),
                bond_type: RdcBondType::NH,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_true),
                sigma_hz: 0.5,
                d_max_hz: dmax,
                is_free: i % 5 == 4,
            });
        }
        // 2. 1D_CAHA (explicit pro-R HA2 and pro-S HA3 on GLY; chiral HA on non-GLY)
        if r.res_name.eq_ignore_ascii_case("GLY") {
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            for (aname, pos_opt) in [("HA2", r.ha2), ("HA3", r.ha3)] {
                let ha_pos = pos_opt.expect("GLY must carry explicit HA2 and HA3 coordinates");
                let u = v_scale(v_sub(ha_pos, r.ca), 1.0 / v_norm(v_sub(ha_pos, r.ca)));
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: r.seq_id,
                    res_name_1: r.res_name.clone(),
                    atom_name_1: "CA".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: aname.to_string(),
                    bond_type: RdcBondType::CAHA,
                    d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_true),
                    sigma_hz: 1.0,
                    d_max_hz: dmax,
                    is_free: i % 6 == 5 && aname == "HA2",
                });
            }
        } else {
            let u = v_scale(v_sub(r.ha, r.ca), 1.0 / v_norm(v_sub(r.ha, r.ca)));
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "HA".to_string(),
                bond_type: RdcBondType::CAHA,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_true),
                sigma_hz: 1.0,
                d_max_hz: dmax,
                is_free: i % 6 == 5,
            });
        }
        // 3. 1D_CACO
        {
            let u = v_scale(v_sub(r.c, r.ca), 1.0 / v_norm(v_sub(r.c, r.ca)));
            let dmax = RdcBondType::CACO.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "C".to_string(),
                bond_type: RdcBondType::CACO,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_true),
                sigma_hz: 0.25,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        if i > 0 {
            let prev = &chain.residues[i - 1];
            // 4. 1D_NC'
            let u_nc = v_scale(v_sub(r.n, prev.c), 1.0 / v_norm(v_sub(r.n, prev.c)));
            let dmax_nc = RdcBondType::NCPrime.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: prev.seq_id,
                res_name_1: prev.res_name.clone(),
                atom_name_1: "C".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "N".to_string(),
                bond_type: RdcBondType::NCPrime,
                d_obs_hz: calc_rdc_from_unit_vec(u_nc, dmax_nc, s_true),
                sigma_hz: 0.20,
                d_max_hz: dmax_nc,
                is_free: false,
            });
            // 5. 1D_HNC'
            if let Some(h) = r.h {
                let u_hc = v_scale(v_sub(h, prev.c), 1.0 / v_norm(v_sub(h, prev.c)));
                let dmax_hc = RdcBondType::HNCPrime.default_d_max_hz();
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: prev.seq_id,
                    res_name_1: prev.res_name.clone(),
                    atom_name_1: "C".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: "H".to_string(),
                    bond_type: RdcBondType::HNCPrime,
                    d_obs_hz: calc_rdc_from_unit_vec(u_hc, dmax_hc, s_true),
                    sigma_hz: 0.35,
                    d_max_hz: dmax_hc,
                    is_free: false,
                });
            }
        }
    }
    RdcMediumSet {
        medium_id: medium_id.to_string(),
        restraints,
    }
}

#[test]
fn test_synthetic_saupe_tensor_svd_exact_recovery_all_5_bond_types() {
    let (_, _, chain) = build_diverse_test_chain();

    // Test multiple canonical alignment tensors (positive & negative Da, varied rhombicity & ZYZ Euler)
    let cases = [
        (14.5, 0.24, [35.0, 48.0, 22.0]),
        (-11.2, 0.51, [62.0, 37.0, 145.0]),
        (18.0, 0.08, [18.0, 71.0, 310.0]),
    ];

    for (da_true, r_true, euler_true) in cases {
        let s_true = saupe_matrix_from_pas(da_true, r_true, euler_true);
        let medium = synthesize_all5_rdc_medium(&chain, "syn_bicelle", &s_true);
        let fit = fit_saupe_tensor_svd(&chain, &medium).expect("exact Saupe SVD fit");

        // Canonical principal-axis invariants: |S_zz| >= |S_yy| >= |S_xx| and 0 <= R <= 2/3
        assert!(fit.principal_s[2].abs() + 1e-15 >= fit.principal_s[1].abs());
        assert!(fit.principal_s[1].abs() + 1e-15 >= fit.principal_s[0].abs());
        assert!(fit.rhombicity >= 0.0 && fit.rhombicity <= (2.0 / 3.0) + 1e-12);

        // < 1e-10 exact recovery of (D_a, R), S matrix, and ZYZ Euler angles
        assert!(
            (fit.da_nh_hz - da_true).abs() < 1e-10,
            "Da recovery error: got {}, expected {}",
            fit.da_nh_hz,
            da_true
        );
        assert!(
            (fit.rhombicity - r_true).abs() < 1e-10,
            "R recovery error: got {}, expected {}",
            fit.rhombicity,
            r_true
        );
        for (r, (row_fit, row_true)) in fit.s_matrix.iter().zip(s_true.iter()).enumerate() {
            for (c, (&v_fit, &v_true)) in row_fit.iter().zip(row_true.iter()).enumerate() {
                assert!((v_fit - v_true).abs() < 1e-12, "S[{r}][{c}] mismatch");
            }
        }
        // Reconstruct S from recovered (Da, R, Euler) and verify machine-precision identity
        let s_rebuilt = saupe_matrix_from_pas(fit.da_nh_hz, fit.rhombicity, fit.euler_zyz_deg);
        for (row_reb, row_true) in s_rebuilt.iter().zip(s_true.iter()) {
            for (&v_reb, &v_true) in row_reb.iter().zip(row_true.iter()) {
                assert!((v_reb - v_true).abs() < 1e-12);
            }
        }
        // Direct canonical Euler check (upper-hemisphere e_z and positive-x e_x)
        let (_, _, canon_euler) = nmr_core::rdc::decompose_saupe_matrix(s_true);
        for (k, (&got, &exp)) in fit.euler_zyz_deg.iter().zip(canon_euler.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "Euler[{k}] error: got {got}, expected {exp}",
            );
        }
        assert!(fit.q_work < 1e-13);
        assert!(fit.q_free < 1e-13);
        assert!(fit.rmsd_work_hz < 1e-11);
    }
}

#[test]
fn test_analytical_rdc_torsion_gradients_vs_finite_differences() {
    let (seq, seq_ids, ref_chain) = build_diverse_test_chain();
    let l = seq.len();

    let s1 = saupe_matrix_from_pas(15.0, 0.28, [41.0, 56.0, 118.0]);
    let s2 = saupe_matrix_from_pas(-12.5, 0.44, [125.0, 39.0, 240.0]);
    let m1 = synthesize_all5_rdc_medium(&ref_chain, "m1", &s1);
    let m2 = synthesize_all5_rdc_medium(&ref_chain, "m2", &s2);
    let media = vec![m1, m2];

    let mut rng = Rng::new(202612);
    let phi: Vec<f64> = (0..l).map(|_| rng.uniform(-2.4, 2.1)).collect();
    let psi: Vec<f64> = (0..l).map(|_| rng.uniform(-2.5, 2.5)).collect();
    let omega = vec![PI; l];

    // 1. Fixed-Saupe analytical torsion gradient vs central finite differences
    let mut gp_fix = vec![0.0; l];
    let mut gs_fix = vec![0.0; l];
    let fixed_mats = [s1, s2];
    let _ = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omega,
        &media,
        &fixed_mats,
        0.35,
        &mut gp_fix,
        &mut gs_fix,
    );

    let eps = 1e-6;
    let mut dummy_gp = vec![0.0; l];
    let mut dummy_gs = vec![0.0; l];
    for i in 0..l {
        if i > 0 {
            let mut p_plus = phi.clone();
            let mut p_minus = phi.clone();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let ep = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
                &seq,
                &seq_ids,
                &p_plus,
                &psi,
                &omega,
                &media,
                &fixed_mats,
                0.35,
                &mut dummy_gp,
                &mut dummy_gs,
            );
            let em = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
                &seq,
                &seq_ids,
                &p_minus,
                &psi,
                &omega,
                &media,
                &fixed_mats,
                0.35,
                &mut dummy_gp,
                &mut dummy_gs,
            );
            let fd = (ep - em) / (2.0 * eps);
            let rel = (gp_fix[i] - fd).abs() / fd.abs().max(1.0);
            assert!(
                rel < 1e-7,
                "fixed-Saupe grad_phi[{i}] mismatch: ana={} vs fd={} (rel={rel})",
                gp_fix[i],
                fd
            );
        }

        let mut s_plus = psi.clone();
        let mut s_minus = psi.clone();
        s_plus[i] += eps;
        s_minus[i] -= eps;
        let ep = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
            &seq,
            &seq_ids,
            &phi,
            &s_plus,
            &omega,
            &media,
            &fixed_mats,
            0.35,
            &mut dummy_gp,
            &mut dummy_gs,
        );
        let em = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
            &seq,
            &seq_ids,
            &phi,
            &s_minus,
            &omega,
            &media,
            &fixed_mats,
            0.35,
            &mut dummy_gp,
            &mut dummy_gs,
        );
        let fd = (ep - em) / (2.0 * eps);
        let rel = (gs_fix[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel < 1e-7,
            "fixed-Saupe grad_psi[{i}] mismatch: ana={} vs fd={} (rel={rel})",
            gs_fix[i],
            fd
        );
    }

    // 2. Variable-projection SVD (Envelope Theorem) combined physical + RDC gradient check
    let mut shifts = nmr_core::nmrstar::Reference::new(seq.clone(), seq_ids.clone());
    for i in 0..l {
        shifts.shifts.insert((i, "CA".to_string()), 56.5);
        shifts.shifts.insert((i, "CB".to_string()), 30.5);
        shifts.shifts.insert((i, "C".to_string()), 176.8);
    }
    let dihs = predict_dihedrals(&shifts);
    let weights = EnergyWeights::default();

    let mut gp_var = vec![0.0; l];
    let mut gs_var = vec![0.0; l];
    let _ = evaluate_energy_and_gradients_with_rdcs(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omega,
        &[],
        &dihs,
        &media,
        0.25,
        &weights,
        &mut gp_var,
        &mut gs_var,
    );

    for i in 1..l {
        let mut p_plus = phi.clone();
        let mut p_minus = phi.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;
        let (_, ebp) = evaluate_energy_and_gradients_with_rdcs(
            &seq,
            &seq_ids,
            &p_plus,
            &psi,
            &omega,
            &[],
            &dihs,
            &media,
            0.25,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
        );
        let (_, ebm) = evaluate_energy_and_gradients_with_rdcs(
            &seq,
            &seq_ids,
            &p_minus,
            &psi,
            &omega,
            &[],
            &dihs,
            &media,
            0.25,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
        );
        let fd = (ebp.total - ebm.total) / (2.0 * eps);
        let rel = (gp_var[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "Envelope-Theorem grad_phi[{i}] mismatch: ana={} vs fd={} (rel={rel})",
            gp_var[i],
            fd
        );
    }

    // 3. Exact analytical grad_omega (including vdW O_i separation + w_omega planarity) vs finite differences
    let omg_pert: Vec<f64> = (0..l).map(|_| PI + rng.uniform(-0.07, 0.07)).collect();
    let mut go_var = vec![0.0; l];
    let _ = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_and_omega(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omg_pert,
        &[],
        &dihs,
        &media,
        0.25,
        115.0,
        &weights,
        &mut gp_var,
        &mut gs_var,
        Some(&mut go_var),
    );
    for i in 0..(l - 1) {
        let mut o_plus = omg_pert.clone();
        let mut o_minus = omg_pert.clone();
        o_plus[i] += eps;
        o_minus[i] -= eps;
        let (_, ebp) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_and_omega(
            &seq,
            &seq_ids,
            &phi,
            &psi,
            &o_plus,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let (_, ebm) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_and_omega(
            &seq,
            &seq_ids,
            &phi,
            &psi,
            &o_minus,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let fd = (ebp.total - ebm.total) / (2.0 * eps);
        let rel = (go_var[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "Envelope-Theorem grad_omega[{i}] mismatch: ana={} vs fd={} (rel={rel})",
            go_var[i],
            fd
        );
    }
    assert_eq!(go_var[l - 1], 0.0);

    // 4. Exact analytical (grad_phi, grad_psi, grad_omega) for Kabsch-Sander 4-site H-bond electrostatics (`w_hbond = 1.75`)
    let mut gp_hb = vec![0.0; l];
    let mut gs_hb = vec![0.0; l];
    let mut go_hb = vec![0.0; l];
    let _ = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omg_pert,
        &[],
        &dihs,
        &media,
        0.25,
        115.0,
        1.75,
        &weights,
        &mut gp_hb,
        &mut gs_hb,
        Some(&mut go_hb),
    );
    for i in 1..l {
        let mut p_plus = phi.clone();
        let mut p_minus = phi.clone();
        p_plus[i] += eps;
        p_minus[i] -= eps;
        let (_, ebp) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &p_plus,
            &psi,
            &omg_pert,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let (_, ebm) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &p_minus,
            &psi,
            &omg_pert,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let fd = (ebp.total - ebm.total) / (2.0 * eps);
        let rel = (gp_hb[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "Kabsch-Sander H-bond grad_phi[{i}] mismatch: ana={} vs fd={} (rel={rel})",
            gp_hb[i],
            fd
        );
    }
    for i in 0..(l - 1) {
        let mut s_plus = psi.clone();
        let mut s_minus = psi.clone();
        s_plus[i] += eps;
        s_minus[i] -= eps;
        let (_, ebp) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &phi,
            &s_plus,
            &omg_pert,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let (_, ebm) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &phi,
            &s_minus,
            &omg_pert,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let fd = (ebp.total - ebm.total) / (2.0 * eps);
        let rel = (gs_hb[i] - fd).abs() / fd.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "Kabsch-Sander H-bond grad_psi[{i}] mismatch: ana={} vs fd={} (rel={rel})",
            gs_hb[i],
            fd
        );

        let mut o_plus = omg_pert.clone();
        let mut o_minus = omg_pert.clone();
        o_plus[i] += eps;
        o_minus[i] -= eps;
        let (_, ebop) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &phi,
            &psi,
            &o_plus,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let (_, ebom) = nmr_core::structure::evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            &seq,
            &seq_ids,
            &phi,
            &psi,
            &o_minus,
            &[],
            &dihs,
            &media,
            0.25,
            115.0,
            1.75,
            &weights,
            &mut dummy_gp,
            &mut dummy_gs,
            None,
        );
        let fdo = (ebop.total - ebom.total) / (2.0 * eps);
        let relo = (go_hb[i] - fdo).abs() / fdo.abs().max(1.0);
        assert!(
            relo < 1e-6,
            "Kabsch-Sander H-bond grad_omega[{i}] mismatch: ana={} vs fd={} (rel={relo})",
            go_hb[i],
            fdo
        );
    }
}

#[test]
fn test_adversarial_rdc_parsers_svd_rank_deficiency_and_nan_q_safety() {
    // 1. Parser rejections: non-ASCII, NaN, Inf, sigma <= 0
    assert!(parse_rdc_tsv_str("# non-ascii \u{03b1}\n").is_err());
    let header = "medium_id\tseq_id_1\tres_name_1\tatom_name_1\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\n";
    assert!(parse_rdc_tsv_str(&format!(
        "{header}m1\t2\tGLN\tN\t2\tGLN\tH\t1D_NH\tNaN\t0.5\n"
    ))
    .is_err());
    assert!(parse_rdc_tsv_str(&format!(
        "{header}m1\t2\tGLN\tN\t2\tGLN\tH\t1D_NH\tinf\t0.5\n"
    ))
    .is_err());
    assert!(parse_rdc_tsv_str(&format!(
        "{header}m1\t2\tGLN\tN\t2\tGLN\tH\t1D_NH\t5.0\t0.0\n"
    ))
    .is_err());
    assert!(parse_rdc_tsv_str(&format!(
        "{header}m1\t2\tGLN\tN\t2\tGLN\tH\t1D_NH\t5.0\t-0.2\n"
    ))
    .is_err());

    let (_, _, chain) = build_diverse_test_chain();
    let s_true = saupe_matrix_from_pas(12.0, 0.25, [20.0, 45.0, 80.0]);
    let mut m_full = synthesize_all5_rdc_medium(&chain, "m1", &s_true);

    // 2. Fewer than 6 working RDCs -> Err
    let m_five = RdcMediumSet {
        medium_id: "m_five".to_string(),
        restraints: m_full
            .restraints
            .iter()
            .filter(|r| !r.is_free)
            .take(5)
            .cloned()
            .collect(),
    };
    assert!(fit_saupe_tensor_svd(&chain, &m_five).is_err());

    // 3. Collinear & coplanar bond vectors -> singular design matrix (cond > 1e8) -> Err
    let mut planar_residues = Vec::new();
    for k in 0..10 {
        let ang = (k as f64) * 0.37;
        planar_residues.push(PdbResidue {
            seq_id: k + 1,
            res_name: "ALA".to_string(),
            chain_id: 'A',
            n: [0.0, 0.0, 0.0],
            // All N-H vectors strictly in the z = 0 plane -> S_xz and S_yz unobservable!
            h: Some([ang.cos(), ang.sin(), 0.0]),
            ca: [1.45, 0.0, 0.0],
            ha: [1.45, 1.09, 0.0],
            ha2: None,
            ha3: None,
            c: [2.5, 0.0, 0.0],
            o: [2.5, -1.2, 0.0],
            cb: Some([1.45, -1.5, 0.0]),
            qb: None,
        });
    }
    let planar_chain = PdbChain {
        residues: planar_residues,
    };
    let planar_med = RdcMediumSet {
        medium_id: "planar".to_string(),
        restraints: (1..=10)
            .map(|sid| RdcRestraint {
                medium_id: "planar".to_string(),
                seq_id_1: sid,
                res_name_1: "ALA".to_string(),
                atom_name_1: "N".to_string(),
                seq_id_2: sid,
                res_name_2: "ALA".to_string(),
                atom_name_2: "H".to_string(),
                bond_type: RdcBondType::NH,
                d_obs_hz: (sid as f64) * 1.1,
                sigma_hz: 0.5,
                d_max_hz: RdcBondType::NH.default_d_max_hz(),
                is_free: false,
            })
            .collect(),
    };
    let err_planar = fit_saupe_tensor_svd(&planar_chain, &planar_med)
        .expect_err("coplanar bond vectors must fail Saupe SVD condition check");
    assert!(
        err_planar.contains("degenerate")
            || err_planar.contains("rank-deficient")
            || err_planar.contains("ill-conditioned"),
        "unexpected error: {err_planar}"
    );

    // 4. All-zero D_obs -> q_work and q_free MUST be NaN (never 0.0000!), and rdc_q_is_better treats NaN as worst
    assert!(cornilescu_q(&[(0.0, 0.0), (0.0, 0.0)]).is_nan());
    for r in &mut m_full.restraints {
        r.d_obs_hz = 0.0;
    }
    let zero_fit = fit_saupe_tensor_svd(&chain, &m_full).unwrap();
    assert!(
        zero_fit.q_work.is_nan() && zero_fit.q_free.is_nan(),
        "zero D_obs must yield NaN Cornilescu Q, got work={} free={}",
        zero_fit.q_work,
        zero_fit.q_free
    );
    assert!(!rdc_q_is_better(f64::NAN, 0.95));
    assert!(rdc_q_is_better(0.95, f64::NAN));
    assert!(!rdc_q_is_better(f64::NAN, f64::NAN));
}

#[test]
fn test_nef_rdc_roundtrip_and_report_svg_xss_safety() {
    let (_, _, chain) = build_diverse_test_chain();
    let s_true = saupe_matrix_from_pas(13.5, 0.30, [25.0, 60.0, 110.0]);
    let mut medium =
        synthesize_all5_rdc_medium(&chain, "bicelle_<script>alert(1)</script>", &s_true);
    partition_rdc_free_set(std::slice::from_mut(&mut medium), 0.20, 42);

    // TSV round-trip
    let tsv = format_rdc_tsv(std::slice::from_ref(&medium));
    let parsed_tsv = parse_rdc_tsv_str(&tsv).unwrap();
    assert_eq!(parsed_tsv[0].restraints.len(), medium.restraints.len());

    // NEF v1.1 round-trip
    let mut doc = NefDocument::new("rdc_test");
    doc.rdc_restraint_lists
        .push(NefRdcRestraintList::from_rdc_medium(
            "bicelle_1",
            "A",
            &medium,
        ));
    let nef_text = write_nef_str(&doc);
    assert!(nef_text.contains("nef_rdc_restraint_list"));
    let doc_back = read_nef_str(&nef_text).unwrap();
    assert_eq!(doc_back.rdc_restraint_lists.len(), 1);
    let media_back = doc_back.rdc_media().unwrap();
    assert_eq!(media_back[0].restraints.len(), medium.restraints.len());

    // Adversarial NEF sigma <= 0 rejection
    let bad_nef = nef_text.replace("0.500", "0.000");
    assert!(read_nef_str(&bad_nef).is_err());

    // HTML report SVG count backward compatibility + XSS escaping check
    let ref_data =
        read_star_all_atoms(workspace_root().join("data/benchmark/bmr17769.str"), None).unwrap();
    let mut conf = std::collections::BTreeMap::new();
    conf.insert(1, 0.95);
    conf.insert(2, 0.92);
    let base_input = ReportInput::new("test", ref_data, conf);
    let base_html = generate_quality_report_html(&base_input);
    assert_eq!(
        base_html.matches("<svg").count(),
        5,
        "default ReportInput::new must preserve exact 5 <svg> panels"
    );

    let rdc_val = evaluate_rdc_validation(&chain, std::slice::from_ref(&medium)).unwrap();
    let rdc_html = generate_quality_report_html(&base_input.with_rdc_report(rdc_val));
    assert_eq!(rdc_html.matches("<svg").count(), 6);
    assert!(rdc_html.contains("id=\"section-rdc-validation\""));
    assert!(
        !rdc_html.contains("<script>alert(1)</script>"),
        "medium_id must be XML-escaped in HTML/SVG report!"
    );
    assert!(rdc_html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
}

#[test]
fn test_experimental_ubiquitin_1d3z_cross_validated_rdc_refinement() {
    let root = workspace_root();
    let pdb_1d3z = parse_pdb_first_model(root.join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_1d3z = read_star_all_atoms(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let mut media =
        read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).expect("read 1D3Z RDC TSV");
    assert_eq!(media.len(), 2);

    // 1. Verify experimental 1D3Z bicelle RDCs fit native 1D3Z structure cleanly across all 5 bond types
    let native_val = evaluate_rdc_validation(&pdb_1d3z, &media).unwrap();
    println!(
        "\n[1D3Z NATIVE EXPERIMENTAL RDC FIT] overall Q_work = {:.4} | m0 ({}, n={}): Da={:+.2} Hz, R={:.3}, Q={:.4} | m1 ({}, n={}): Da={:+.2} Hz, R={:.3}, Q={:.4}",
        native_val.overall_q_work,
        native_val.media[0].medium_id,
        native_val.media[0].n_work,
        native_val.media[0].da_nh_hz,
        native_val.media[0].rhombicity,
        native_val.media[0].q_work,
        native_val.media[1].medium_id,
        native_val.media[1].n_work,
        native_val.media[1].da_nh_hz,
        native_val.media[1].rhombicity,
        native_val.media[1].q_work,
    );
    assert!(native_val.overall_q_work < 0.13);

    // 2. Deterministically hold out 20% cross-validation free set BEFORE any refinement
    partition_rdc_free_set(&mut media, 0.20, 199807);
    let total_free: usize = media
        .iter()
        .map(|m| m.restraints.iter().filter(|r| r.is_free).count())
        .sum();
    let total_work: usize = media
        .iter()
        .map(|m| m.restraints.iter().filter(|r| !r.is_free).count())
        .sum();
    assert!(total_free >= 110 && total_work >= 480);

    // 3. Fold NOE + firewalled TorsionDb baseline ensemble (zero RDCs during folding)
    let peaks_1d3z = simulate_noesy_peaks(&pdb_1d3z, &shifts_1d3z, &NoesySimConfig::default());
    let cfg = StructureCalcConfig::default();

    // Enforce HomologFirewall on 1D3Z ubiquitin torsions too so zero ubiquitin templates participate!
    let fw_ubq = HomologFirewallQuery {
        query_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string(), "1G6J".to_string()],
        query_bmrb_ids: vec!["bmr17769".to_string(), "bmr5387".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: Some("3.10.20.90".to_string()),
        query_topology_id: Some("3.10.20".to_string()),
        query_ca_coords: Some(pdb_1d3z.residues.iter().map(|r| r.ca).collect()),
    };
    let fw_dihs = predict_torsions_with_firewall(&shifts_1d3z, &fw_ubq).to_dihedral_restraints();

    let (noe_base, none_rdc) = calculate_structure_with_rdcs_and_dihedrals(
        &shifts_1d3z,
        &peaks_1d3z,
        fw_dihs.clone(),
        &[],
        0.0,
        &cfg,
    )
    .unwrap();
    assert!(none_rdc.is_none());

    let before_rdc_val = evaluate_rdc_validation(&noe_base.conformers[0].chain, &media).unwrap();
    let before_struct_val = validate_ensemble(
        &noe_base.conformers,
        &noe_base.final_adrs,
        &fw_dihs,
        Some(&pdb_1d3z),
    );

    // 4. Joint NOE + firewalled TorsionDb + working (!is_free) experimental 1D3Z RDC refinement
    let (rdc_res, Some(after_rdc_val)) = calculate_structure_with_rdcs_and_dihedrals(
        &shifts_1d3z,
        &peaks_1d3z,
        fw_dihs.clone(),
        &media,
        0.75,
        &cfg,
    )
    .expect("1D3Z RDC ensemble calculation") else {
        panic!("expected Some(RdcValidationSummary)");
    };

    let after_struct_val = validate_ensemble(
        &rdc_res.conformers,
        &rdc_res.final_adrs,
        &fw_dihs,
        Some(&pdb_1d3z),
    );

    let rmsd_before = before_struct_val.lowest_energy_rmsd_ordered_to_ref.unwrap();
    let rmsd_after = after_struct_val.lowest_energy_rmsd_ordered_to_ref.unwrap();
    let mean_before = before_struct_val.mean_backbone_rmsd_ordered_to_ref.unwrap();
    let mean_after = after_struct_val.mean_backbone_rmsd_ordered_to_ref.unwrap();

    println!("\n=== T12.4 EXPERIMENTAL UBIQUITIN 1D3Z CROSS-VALIDATED RDC REFINEMENT ===");
    println!("  Working RDCs = {total_work}, Held-out Free RDCs = {total_free} (20% stratified)");
    println!(
        "  NOE-only baseline : Q_work = {:.4}, Q_free = {:.4} | Ordered BB RMSD lowE = {:.3} A, mean = {:.3} A | Rama fav+allow = {:.1}%",
        before_rdc_val.overall_q_work,
        before_rdc_val.overall_q_free,
        rmsd_before,
        mean_before,
        before_struct_val.ramachandran_favored_pct + before_struct_val.ramachandran_allowed_pct
    );
    println!(
        "  + 1D3Z RDC refined: Q_work = {:.4}, Q_free = {:.4} | Ordered BB RMSD lowE = {:.3} A, mean = {:.3} A | Rama fav+allow = {:.1}%",
        after_rdc_val.overall_q_work,
        after_rdc_val.overall_q_free,
        rmsd_after,
        mean_after,
        after_struct_val.ramachandran_favored_pct + after_struct_val.ramachandran_allowed_pct
    );
    for m in &after_rdc_val.media {
        println!(
            "    Medium {:<16}: n_work={:>3}, n_free={:>2}, Da={:+.2} Hz, R={:.3}, Q_work={:.4}, Q_free={:.4}",
            m.medium_id, m.n_work, m.n_free, m.da_nh_hz, m.rhombicity, m.q_work, m.q_free
        );
    }

    assert!(
        after_rdc_val.overall_q_work < 0.14 && after_rdc_val.overall_q_free < 0.34,
        "1D3Z ByPeptidePlane Q_work={:.4} (< 0.14) or Q_free={:.4} (< 0.34) out of bounds!",
        after_rdc_val.overall_q_work,
        after_rdc_val.overall_q_free
    );
    assert!(
        after_rdc_val.overall_q_free < before_rdc_val.overall_q_free,
        "RDC refinement must improve held-out Q_free ({:.4} -> {:.4})",
        before_rdc_val.overall_q_free,
        after_rdc_val.overall_q_free
    );
    assert!(
        rmsd_after < 0.80 && rmsd_after < rmsd_before && mean_after < mean_before,
        "RDC refinement must lower backbone RMSD to 1D3Z vs NOE-only folding (lowE: {rmsd_before:.3} -> {rmsd_after:.3} A; mean: {mean_before:.3} -> {mean_after:.3} A)"
    );
}

#[test]
fn test_p1_6_gly_explicit_ha2_ha3_tetrahedral_rdc_and_unresolved_counts() {
    let root = workspace_root();
    let pdb_1d3z = parse_pdb_first_model(root.join("data/pdb_refs/1d3z.pdb")).unwrap();
    let gly10 = pdb_1d3z
        .residues
        .iter()
        .find(|r| r.seq_id == 10 && r.res_name == "GLY")
        .expect("1D3Z GLY 10");
    let ha2_exp = gly10.ha2.expect("1D3Z GLY 10 HA2");
    let ha3_exp = gly10.ha3.expect("1D3Z GLY 10 HA3");

    // Verify reconstructed GLY HA2 (pro-R) and HA3 (pro-S) match IUPAC 1D3Z stereochemistry (< 0.03 A)
    let (_, _, rec_ha2, rec_ha3, _) =
        reconstruct_ca_substituents("GLY", gly10.n, gly10.ca, gly10.c);
    let d_ha2 = v_norm(v_sub(rec_ha2.unwrap(), ha2_exp));
    let d_ha3 = v_norm(v_sub(rec_ha3.unwrap(), ha3_exp));
    assert!(
        d_ha2 < 0.03 && d_ha3 < 0.03,
        "GLY 10 pro-R HA2 / pro-S HA3 reconstruction drifted: d_ha2={d_ha2:.4} A, d_ha3={d_ha3:.4} A"
    );

    // Verify tetrahedral bond lengths (~1.09 A) and H-C-H angle (~109.5 deg, NOT bisector midpoint!)
    let v2 = v_sub(rec_ha2.unwrap(), gly10.ca);
    let v3 = v_sub(rec_ha3.unwrap(), gly10.ca);
    let r2 = v_norm(v2);
    let r3 = v_norm(v3);
    let cos_ang = v_dot(v2, v3) / (r2 * r3);
    let ang_deg = cos_ang.clamp(-1.0, 1.0).acos().to_degrees();
    assert!((r2 - 1.090).abs() < 1e-6 && (r3 - 1.090).abs() < 1e-6);
    assert!(
        (107.5..=111.5).contains(&ang_deg),
        "GLY HA2-CA-HA3 angle must be tetrahedral (~109.5 deg), got {ang_deg:.2} deg"
    );

    // Verify RdcRestraint::validate rejects HA3/HA2 on ALA and ambiguous HA on GLY
    let (_, _, chain) = build_diverse_test_chain();
    let s_true = saupe_matrix_from_pas(14.0, 0.28, [30.0, 50.0, 105.0]);
    let med = synthesize_all5_rdc_medium(&chain, "m_gly", &s_true);
    let fit = fit_saupe_tensor_svd(&chain, &med).unwrap();
    assert!(
        fit.q_work < 1e-12,
        "exact synthetic Saupe SVD with GLY CA-HA2 & CA-HA3 must give Q_work < 1e-12, got {}",
        fit.q_work
    );
    assert_eq!(
        count_unresolved_rdc_restraints(&chain, std::slice::from_ref(&med)),
        0
    );

    let bad_ala_ha3 = RdcRestraint {
        medium_id: "m_gly".to_string(),
        seq_id_1: 1,
        res_name_1: "ALA".to_string(),
        atom_name_1: "CA".to_string(),
        seq_id_2: 1,
        res_name_2: "ALA".to_string(),
        atom_name_2: "HA3".to_string(),
        bond_type: RdcBondType::CAHA,
        d_obs_hz: 5.0,
        sigma_hz: 1.0,
        d_max_hz: RdcBondType::CAHA.default_d_max_hz(),
        is_free: false,
    };
    assert!(
        bad_ala_ha3.validate().is_err(),
        "HA3 on ALA must fail RdcRestraint::validate"
    );

    let bad_gly_ha = RdcRestraint {
        medium_id: "m_gly".to_string(),
        seq_id_1: 10,
        res_name_1: "GLY".to_string(),
        atom_name_1: "CA".to_string(),
        seq_id_2: 10,
        res_name_2: "GLY".to_string(),
        atom_name_2: "HA".to_string(),
        bond_type: RdcBondType::CAHA,
        d_obs_hz: 5.0,
        sigma_hz: 1.0,
        d_max_hz: RdcBondType::CAHA.default_d_max_hz(),
        is_free: false,
    };
    assert!(
        bad_gly_ha.validate().is_err(),
        "ambiguous HA on GLY must fail RdcRestraint::validate"
    );

    // Wildcard residue name "." with HA3 on non-GLY residue (seq_id=1 MET) + out-of-range seq_id=999
    // must NOT resolve and must be counted by `count_unresolved_rdc_restraints`.
    let mut med_with_drops = med;
    med_with_drops.restraints.push(RdcRestraint {
        medium_id: "m_gly".to_string(),
        seq_id_1: 1,
        res_name_1: ".".to_string(),
        atom_name_1: "CA".to_string(),
        seq_id_2: 1,
        res_name_2: ".".to_string(),
        atom_name_2: "HA3".to_string(),
        bond_type: RdcBondType::CAHA,
        d_obs_hz: 4.0,
        sigma_hz: 1.0,
        d_max_hz: RdcBondType::CAHA.default_d_max_hz(),
        is_free: false,
    });
    med_with_drops.restraints.push(RdcRestraint {
        medium_id: "m_gly".to_string(),
        seq_id_1: 999,
        res_name_1: "ILE".to_string(),
        atom_name_1: "N".to_string(),
        seq_id_2: 999,
        res_name_2: "ILE".to_string(),
        atom_name_2: "H".to_string(),
        bond_type: RdcBondType::NH,
        d_obs_hz: 7.0,
        sigma_hz: 0.5,
        d_max_hz: RdcBondType::NH.default_d_max_hz(),
        is_free: false,
    });
    assert_eq!(
        count_unresolved_rdc_restraints(&chain, &[med_with_drops]),
        2
    );
}

#[test]
fn test_rdc_free_stratification_by_peptide_plane_zero_plane_leakage() {
    let root = workspace_root();
    let media_raw =
        read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).expect("read 1D3Z RDC TSV");
    assert_eq!(
        RdcFreeStratification::default(),
        RdcFreeStratification::ByPeptidePlane
    );

    // 1. Default `partition_rdc_free_set` (ByPeptidePlane): ZERO peptide planes have mixed work/free
    //    couplings within or across alignment media!
    let mut media_plane = media_raw.clone();
    partition_rdc_free_set(&mut media_plane, 0.20, 199807);

    let mut plane_states: std::collections::BTreeMap<i32, (usize, usize)> =
        std::collections::BTreeMap::new();
    for m in &media_plane {
        let mut med_work = 0usize;
        for r in &m.restraints {
            if !r.is_free {
                med_work += 1;
            }
            let pid = rdc_peptide_plane_id(r);
            let entry = plane_states.entry(pid).or_insert((0, 0));
            if r.is_free {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        }
        assert!(
            med_work >= 6,
            "medium '{}' must retain >= 6 work RDCs",
            m.medium_id
        );
    }
    let split_planes_plane_mode = plane_states
        .iter()
        .filter(|(_, &(w, f))| w > 0 && f > 0)
        .count();
    let free_planes = plane_states.iter().filter(|(_, &(_, f))| f > 0).count();
    assert_eq!(
        split_planes_plane_mode, 0,
        "ByPeptidePlane must hold out whole peptide planes with zero mixed work/free planes"
    );
    assert!(
        free_planes >= 10,
        "expected >= 10 held-out peptide planes at 20% holdout, got {free_planes}"
    );

    // Verify specific coplanar group mapping: NH(i), NC'(i-1,i), HNC'(i-1,i), CACO(i-1), CAHA(i) -> plane_id = i
    for m in &media_plane {
        for r in &m.restraints {
            let pid = rdc_peptide_plane_id(r);
            match r.bond_type {
                RdcBondType::NH | RdcBondType::CAHA => assert_eq!(pid, r.seq_id_1.max(r.seq_id_2)),
                RdcBondType::NCPrime | RdcBondType::HNCPrime => {
                    assert_eq!(pid, r.seq_id_1.max(r.seq_id_2))
                }
                RdcBondType::CACO => assert_eq!(pid, r.seq_id_1.min(r.seq_id_2) + 1),
            }
        }
    }

    // 2. Legacy `ByCoupling` mode remains accessible and splits multi-coupling peptide planes
    let mut media_coupling = media_raw;
    partition_rdc_free_set_with_mode(
        &mut media_coupling,
        0.20,
        199807,
        RdcFreeStratification::ByCoupling,
    );
    let mut coupling_plane_states: std::collections::BTreeMap<i32, (usize, usize)> =
        std::collections::BTreeMap::new();
    for m in &media_coupling {
        for r in &m.restraints {
            let entry = coupling_plane_states
                .entry(rdc_peptide_plane_id(r))
                .or_insert((0, 0));
            if r.is_free {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        }
    }
    let split_planes_coupling_mode = coupling_plane_states
        .iter()
        .filter(|(_, &(w, f))| w > 0 && f > 0)
        .count();
    assert!(
        split_planes_coupling_mode > 0,
        "ByCoupling mode should split peptide planes across bond types, got 0 split planes"
    );
}

#[test]
fn test_rdc_signed_dmax_and_physical_gamma_sign_convention() {
    // Verify harmonized covalent geometry vs libration-averaged effective bond lengths and IUPAC labels
    assert!((R_NH_GEOM_A - 1.010).abs() < 1e-12);
    assert!((R_NH_EFF_A - 1.040).abs() < 1e-12);
    assert!((R_CAHA_GEOM_A - 1.090).abs() < 1e-12);
    assert!((BOND_CA_HA - R_CAHA_GEOM_A).abs() < 1e-12);
    assert!((R_CAHA_EFF_A - 1.105).abs() < 1e-12);
    assert!((DMAX_CAHA_HZ / DMAX_NH_HZ - 2.0679).abs() < 1e-3);
    assert_eq!(RdcBondType::HNCPrime.canonical_label(), "1D_HNC'");
    assert_eq!(RdcBondType::HNCPrime.iupac_coupling_label(), "2D_HNC'");
    assert_eq!(
        RdcBondType::parse_label("2D_HNC'").unwrap(),
        RdcBondType::HNCPrime
    );
    assert_eq!(RdcBondType::NH.physical_gamma_sign(), -1.0);
    assert_eq!(RdcBondType::NCPrime.physical_gamma_sign(), -1.0);
    assert_eq!(RdcBondType::CAHA.physical_gamma_sign(), 1.0);
    assert_eq!(RdcBondType::CACO.physical_gamma_sign(), 1.0);
    assert_eq!(RdcBondType::HNCPrime.physical_gamma_sign(), 1.0);

    let (seq, seq_ids, chain) = build_diverse_test_chain();
    let s_true = saupe_matrix_from_pas(14.2, 0.31, [33.0, 52.0, 128.0]);
    let mut med_pos = synthesize_all5_rdc_medium(&chain, "bicelle_sign", &s_true);
    partition_rdc_free_set(std::slice::from_mut(&mut med_pos), 0.20, 77);

    // Build equivalent physical-gamma-sign dataset (`gamma_15N < 0` => negative `d_obs_hz` & negative `d_max_hz`
    // on `1D_NH` and `1D_NC'`, using `"2D_HNC'"` label in TSV with `# sign_convention: physical_gamma`)
    let mut tsv_phys = String::from(
        "# sign_convention: physical_gamma\n\
         medium_id\tseq_id_1\tres_name_1\tatom_name_1\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\td_max_hz\tis_free\n",
    );
    for r in &med_pos.restraints {
        let g_sgn = r.bond_type.physical_gamma_sign();
        let d_obs_phys = r.d_obs_hz * g_sgn;
        tsv_phys.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.15e}\t{:.6}\t.\t{}\n",
            r.medium_id,
            r.seq_id_1,
            r.res_name_1,
            r.atom_name_1,
            r.seq_id_2,
            r.res_name_2,
            r.atom_name_2,
            r.bond_type.iupac_coupling_label(),
            d_obs_phys,
            r.sigma_hz,
            if r.is_free { 1 } else { 0 }
        ));
    }

    let media_phys = parse_rdc_tsv_str(&tsv_phys).expect("parse physical_gamma RDC TSV");
    assert_eq!(media_phys.len(), 1);
    assert!(media_phys[0]
        .restraints
        .iter()
        .filter(|r| matches!(r.bond_type, RdcBondType::NH | RdcBondType::NCPrime))
        .all(|r| r.d_max_hz < 0.0));
    assert!(media_phys[0]
        .restraints
        .iter()
        .filter(|r| !matches!(r.bond_type, RdcBondType::NH | RdcBondType::NCPrime))
        .all(|r| r.d_max_hz > 0.0));

    // Also test explicit negative `d_max_hz` column without header directive
    let mut med_explicit_neg = med_pos.clone();
    for r in &mut med_explicit_neg.restraints {
        if r.bond_type.physical_gamma_sign() < 0.0 {
            r.d_obs_hz = -r.d_obs_hz;
            r.d_max_hz = -r.d_max_hz;
        }
        r.validate()
            .expect("negative d_max_hz must validate cleanly");
    }
    let parsed_explicit = parse_rdc_tsv_str_with_convention(
        &format_rdc_tsv(std::slice::from_ref(&med_explicit_neg)),
        RdcSignConvention::SignNormalized,
    )
    .unwrap();
    assert!(parsed_explicit[0]
        .restraints
        .iter()
        .any(|r| r.d_max_hz < 0.0));

    // Zero `d_max_hz` must still fail validation
    let mut bad_zero_dmax = med_pos.restraints[0].clone();
    bad_zero_dmax.d_max_hz = 0.0;
    assert!(bad_zero_dmax.validate().is_err());

    // Machine-precision invariance of Saupe SVD fit (`S`, `Da`, `R`, `Euler`, `Q_work`, `Q_free`)
    let fit_pos = fit_saupe_tensor_svd(&chain, &med_pos).unwrap();
    let fit_phys = fit_saupe_tensor_svd(&chain, &media_phys[0]).unwrap();
    assert!((fit_pos.da_nh_hz - fit_phys.da_nh_hz).abs() < 1e-9);
    assert!((fit_pos.rhombicity - fit_phys.rhombicity).abs() < 1e-10);
    assert!((fit_pos.q_work - fit_phys.q_work).abs() < 1e-12);
    assert!((fit_pos.q_free - fit_phys.q_free).abs() < 1e-12);
    for r in 0..3 {
        for c in 0..3 {
            assert!((fit_pos.s_matrix[r][c] - fit_phys.s_matrix[r][c]).abs() < 1e-12);
        }
    }

    // Machine-precision invariance of analytical variable-projection torsion energy & gradients
    let mut rng = Rng::new(909);
    let l = seq.len();
    let phi: Vec<f64> = (0..l).map(|_| rng.uniform(-2.2, 2.0)).collect();
    let psi: Vec<f64> = (0..l).map(|_| rng.uniform(-2.4, 2.4)).collect();
    let omega = vec![PI; l];
    let weights = EnergyWeights::default();
    let mut gp_pos = vec![0.0; l];
    let mut gs_pos = vec![0.0; l];
    let mut gp_phys = vec![0.0; l];
    let mut gs_phys = vec![0.0; l];
    let (_, eb_pos) = evaluate_energy_and_gradients_with_rdcs(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omega,
        &[],
        &[],
        std::slice::from_ref(&med_pos),
        0.5,
        &weights,
        &mut gp_pos,
        &mut gs_pos,
    );
    let (_, eb_phys) = evaluate_energy_and_gradients_with_rdcs(
        &seq,
        &seq_ids,
        &phi,
        &psi,
        &omega,
        &[],
        &[],
        &media_phys,
        0.5,
        &weights,
        &mut gp_phys,
        &mut gs_phys,
    );
    assert!((eb_pos.total - eb_phys.total).abs() < 1e-9);
    for i in 0..l {
        assert!((gp_pos[i] - gp_phys[i]).abs() < 1e-9);
        assert!((gs_pos[i] - gs_phys[i]).abs() < 1e-9);
    }

    // `normalize_rdc_sign_convention` converts `media_phys` back to positive `d_max_hz > 0`
    let mut norm_media = media_phys;
    normalize_rdc_sign_convention(&mut norm_media, RdcSignConvention::PhysicalGammaSign);
    for (ra, rb) in norm_media[0]
        .restraints
        .iter()
        .zip(med_pos.restraints.iter())
    {
        assert!(ra.d_max_hz > 0.0);
        assert!((ra.d_obs_hz - rb.d_obs_hz).abs() < 1e-8);
        assert!((ra.d_max_hz - rb.d_max_hz).abs() < 1e-8);
    }
}

#[test]
fn test_noncircular_ubiquitin_exp_noe_fold_rdc_refine_vs_1ubq_xray() {
    let root = workspace_root();
    // Independent 1987 1.8 A X-ray crystal reference structure (1UBQ; Vijay-Kumar et al., JMB 1987),
    // solved 11 years BEFORE 1D3Z bicelle RDCs existed — strictly non-circular external reference!
    let xray_1ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("read 1UBQ");
    assert_eq!(xray_1ubq.residues.len(), 76);

    let seq: Vec<String> = xray_1ubq
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let seq_ids: Vec<i32> = xray_1ubq.residues.iter().map(|r| r.seq_id).collect();

    // Pure experimental proton-proton NOE distance bounds from wwPDB 1D3Z (`1d3z.mr`, zero simulated NOEs!)
    let exp_adrs = read_exp_noe_tsv(
        root.join("data/fixtures/ubiquitin_1d3z_exp_noe.tsv"),
        &seq,
        &seq_ids,
    )
    .expect("read experimental 1D3Z NOEs");
    assert!(exp_adrs.len() >= 2500);

    // Experimental bicelle RDCs partitioned with 20% ByPeptidePlane RdcFree holdout
    let mut media =
        read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).expect("read 1D3Z RDCs");
    partition_rdc_free_set_with_mode(
        &mut media,
        0.20,
        199807,
        RdcFreeStratification::ByPeptidePlane,
    );

    // 1. Validate experimental 1D3Z bicelle RDCs directly on independent 1.8 A X-ray 1UBQ
    let xray_rdc_val = evaluate_rdc_validation(&xray_1ubq, &media).unwrap();
    println!(
        "\n[NON-CIRCULAR UBIQUITIN -> 1UBQ 1.8A X-RAY] Direct 1D3Z RDC fit on unrefined 1UBQ crystal: Q_work = {:.4}, plane Q_free = {:.4}",
        xray_rdc_val.overall_q_work, xray_rdc_val.overall_q_free
    );

    // 2. Fold de novo from extended/random coordinates using ONLY experimental NOEs (`dihedrals = &[]`, zero RDCs)
    let cfg = StructureCalcConfig {
        seed: 20260401,
        ..StructureCalcConfig::default()
    };
    let (noe_only, none_rdc) =
        fold_from_experimental_noes_and_rdcs(&seq, &seq_ids, &exp_adrs, &[], &[], 0.0, &cfg)
            .expect("pure experimental NOE fold");
    assert!(none_rdc.is_none());

    let pre_rdc_val = evaluate_rdc_validation(&noe_only.conformers[0].chain, &media).unwrap();
    let pre_rmsd_low = backbone_rmsd_seq_range(&noe_only.conformers[0].chain, &xray_1ubq, 1, 71);
    let pre_rmsd_mean: f64 = noe_only
        .conformers
        .iter()
        .map(|c| backbone_rmsd_seq_range(&c.chain, &xray_1ubq, 1, 71))
        .sum::<f64>()
        / (noe_only.conformers.len() as f64);

    // 3. Multi-cycle pure-experimental-NOE + working (!is_free) bicelle RDC refinement from the same Cycle 0 pool
    let (rdc_ens, post_rdc_opt) =
        fold_from_experimental_noes_and_rdcs(&seq, &seq_ids, &exp_adrs, &[], &media, 0.75, &cfg)
            .expect("RDC refinement on pure-NOE ubiquitin ensemble");
    let post_rdc_val = post_rdc_opt.expect("RDC validation summary");
    let refined_conformers = rdc_ens.conformers;

    let post_rmsd_low = backbone_rmsd_seq_range(&refined_conformers[0].chain, &xray_1ubq, 1, 71);
    let post_rmsd_mean: f64 = refined_conformers
        .iter()
        .map(|c| backbone_rmsd_seq_range(&c.chain, &xray_1ubq, 1, 71))
        .sum::<f64>()
        / (refined_conformers.len() as f64);

    println!("=== P1-9 NON-CIRCULAR UBIQUITIN (EXP 1D3Z NOE -> RDC -> INDEPENDENT 1UBQ X-RAY) ===");
    println!(
        "  Exp NOE-only  : Q_work = {:.4}, plane Q_free = {:.4} | 1UBQ (1..71) BB RMSD lowE = {:.3} A, mean = {:.3} A",
        pre_rdc_val.overall_q_work, pre_rdc_val.overall_q_free, pre_rmsd_low, pre_rmsd_mean
    );
    println!(
        "  + Bicelle RDC : Q_work = {:.4}, plane Q_free = {:.4} | 1UBQ (1..71) BB RMSD lowE = {:.3} A, mean = {:.3} A",
        post_rdc_val.overall_q_work, post_rdc_val.overall_q_free, post_rmsd_low, post_rmsd_mean
    );

    assert!(
        post_rdc_val.overall_q_work < 0.22
            && post_rdc_val.overall_q_work < pre_rdc_val.overall_q_work,
        "working Q must improve ({:.4} -> {:.4})",
        pre_rdc_val.overall_q_work,
        post_rdc_val.overall_q_work
    );
    assert!(
        post_rdc_val.overall_q_free < pre_rdc_val.overall_q_free,
        "peptide-plane Q_free must improve ({:.4} -> {:.4})",
        pre_rdc_val.overall_q_free,
        post_rdc_val.overall_q_free
    );
    assert!(
        post_rdc_val.overall_q_free < 0.65,
        "non-circular pure-NOE ubiquitin peptide-plane Q_free {:.4} must be < 0.65",
        post_rdc_val.overall_q_free
    );
    assert!(
        post_rmsd_low < 1.80 && post_rmsd_mean < 2.35 && post_rmsd_low < pre_rmsd_low,
        "non-circular 1UBQ (1..71) BB RMSD out of range: lowE={pre_rmsd_low:.3} -> {post_rmsd_low:.3} A, mean={pre_rmsd_mean:.3} -> {post_rmsd_mean:.3} A"
    );
    assert!(
        post_rmsd_mean < pre_rmsd_mean,
        "RDC refinement must improve independent 1UBQ (1..71) mean BB RMSD ({pre_rmsd_mean:.3} -> {post_rmsd_mean:.3} A)"
    );
}

#[test]
fn test_noncircular_gb1_3gb1_exp_noe_fold_rdc_refine_vs_1pga_xray() {
    let root = workspace_root();
    // Independent 1994 2.07 A X-ray crystal structure of Streptococcal Protein G B1 domain (1PGA;
    // Gallagher et al., Biochemistry 1994) — solved 4 years BEFORE 3GB1 TMV/bicelle RDCs!
    let xray_1pga = parse_pdb_first_model(root.join("data/pdb_refs/1pga.pdb")).expect("read 1PGA");
    assert_eq!(xray_1pga.residues.len(), 56);

    let seq: Vec<String> = xray_1pga
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let seq_ids: Vec<i32> = xray_1pga.residues.iter().map(|r| r.seq_id).collect();

    // Pure experimental proton-proton NOEs from wwPDB 3GB1 (`3gb1.mr` Section A, zero H-bonds, zero simulated NOEs!)
    let exp_adrs = read_exp_noe_tsv(
        root.join("data/fixtures/gb1_3gb1_exp_noe.tsv"),
        &seq,
        &seq_ids,
    )
    .expect("read experimental 3GB1 NOEs");
    assert!(exp_adrs.len() >= 600);

    // Experimental two-medium peptide-plane RDCs (`tmv_3gb1` + `bicelle_3gb1`, 300 couplings) with 20% ByPeptidePlane holdout
    let mut media =
        read_rdc_tsv(root.join("data/fixtures/gb1_3gb1_rdc.tsv")).expect("read 3GB1 RDCs");
    assert_eq!(media.len(), 2);
    partition_rdc_free_set_with_mode(
        &mut media,
        0.20,
        2026,
        RdcFreeStratification::ByPeptidePlane,
    );

    // 1. Direct Saupe SVD validation of experimental 3GB1 RDCs on unrefined 1994 X-ray 1PGA
    let xray_rdc_val = evaluate_rdc_validation(&xray_1pga, &media).expect("1PGA Saupe SVD");
    let nh_only_media: Vec<RdcMediumSet> = media
        .iter()
        .map(|m| RdcMediumSet {
            medium_id: m.medium_id.clone(),
            restraints: m
                .restraints
                .iter()
                .filter(|r| r.bond_type == RdcBondType::NH)
                .cloned()
                .collect(),
        })
        .collect();
    let xray_nh_val = evaluate_rdc_validation(&xray_1pga, &nh_only_media).expect("1PGA NH SVD");
    println!(
        "\n[NON-CIRCULAR GB1 -> 1PGA 2.07A X-RAY] Direct 3GB1 RDC fit on unrefined 1994 1PGA crystal: 1D_NH Q = {:.4}, multi-bond Q_work = {:.4}, plane Q_free = {:.4} | m0 ({}, Da={:+.2} Hz, R={:.3}, Q={:.4}) | m1 ({}, Da={:+.2} Hz, R={:.3}, Q={:.4})",
        xray_nh_val.overall_q_work,
        xray_rdc_val.overall_q_work,
        xray_rdc_val.overall_q_free,
        xray_rdc_val.media[0].medium_id,
        xray_rdc_val.media[0].da_nh_hz,
        xray_rdc_val.media[0].rhombicity,
        xray_rdc_val.media[0].q_work,
        xray_rdc_val.media[1].medium_id,
        xray_rdc_val.media[1].da_nh_hz,
        xray_rdc_val.media[1].rhombicity,
        xray_rdc_val.media[1].q_work,
    );
    assert!(
        xray_nh_val.overall_q_work < 0.20 && xray_rdc_val.overall_q_work < 0.36,
        "experimental 3GB1 RDCs must agree with independent 1994 1PGA X-ray structure (got NH Q={:.4}, multi-bond Q_work={:.4})",
        xray_nh_val.overall_q_work,
        xray_rdc_val.overall_q_work
    );

    // 2. Predict non-homologous (phi, psi) torsion restraints from public CC0 solution-NMR BMRB 7280
    //    chemical shifts under the 4-layer HomologFirewall (excluding all GB1 / CATH 3.10.20 beta-grasp
    //    templates, with `query_ca_coords: None` so ZERO 1PGA/3GB1 coordinates ever enter folding!).
    let shifts_gb1 = read_star_all_atoms(root.join("data/fixtures/gb1_bmr7280_shifts.str"), None)
        .expect("read BMRB 7280 shifts");
    let fw_gb1 = HomologFirewallQuery {
        query_pdb_ids: vec![
            "1PGA".to_string(),
            "3GB1".to_string(),
            "1GB1".to_string(),
            "2GB1".to_string(),
            "2QMT".to_string(),
        ],
        query_bmrb_ids: vec!["bmr7280".to_string(), "bmr15156".to_string()],
        query_uniprot_id: Some("P06654".to_string()),
        query_family_id: Some("3.10.20.10".to_string()),
        query_topology_id: Some("3.10.20".to_string()),
        query_ca_coords: None,
    };
    let gb1_pred = predict_torsions_with_firewall(&shifts_gb1, &fw_gb1);
    assert!(gb1_pred.max_retained_seq_identity <= 0.25);
    let fw_dihs_gb1 = gb1_pred.to_dihedral_restraints();

    // 3. Fold GB1 de novo from extended coordinates using experimental 3GB1 NOEs + firewalled BMRB 7280 shifts (zero RDCs)
    let cfg = StructureCalcConfig {
        seed: 20260402,
        ..StructureCalcConfig::default()
    };
    let (noe_only, none_rdc) = fold_from_experimental_noes_and_rdcs(
        &seq,
        &seq_ids,
        &exp_adrs,
        &fw_dihs_gb1,
        &[],
        0.0,
        &cfg,
    )
    .expect("experimental GB1 NOE + firewall shift fold");
    assert!(none_rdc.is_none());

    let pre_rdc_val = evaluate_rdc_validation(&noe_only.conformers[0].chain, &media).unwrap();
    let pre_rmsd_low = backbone_rmsd_seq_range(&noe_only.conformers[0].chain, &xray_1pga, 1, 56);
    let pre_rmsd_mean: f64 = noe_only
        .conformers
        .iter()
        .map(|c| backbone_rmsd_seq_range(&c.chain, &xray_1pga, 1, 56))
        .sum::<f64>()
        / (noe_only.conformers.len() as f64);

    // 4. Multi-cycle experimental 3GB1 NOE + firewalled BMRB 7280 shifts + working (!is_free) 3GB1 TMV & bicelle RDC refinement
    let (rdc_ens, post_rdc_opt) = fold_from_experimental_noes_and_rdcs(
        &seq,
        &seq_ids,
        &exp_adrs,
        &fw_dihs_gb1,
        &media,
        0.75,
        &cfg,
    )
    .expect("RDC refinement on experimental GB1 ensemble");
    let post_rdc_val = post_rdc_opt.expect("GB1 RDC validation summary");
    let refined_conformers = rdc_ens.conformers;

    let post_rmsd_low = backbone_rmsd_seq_range(&refined_conformers[0].chain, &xray_1pga, 1, 56);
    let post_rmsd_mean: f64 = refined_conformers
        .iter()
        .map(|c| backbone_rmsd_seq_range(&c.chain, &xray_1pga, 1, 56))
        .sum::<f64>()
        / (refined_conformers.len() as f64);

    let post_nh_val =
        evaluate_rdc_validation(&refined_conformers[0].chain, &nh_only_media).expect("GB1 NH SVD");

    println!("=== P1-9 NON-CIRCULAR GB1 (EXP 3GB1 NOE + BMRB 7280 FIREWALL -> RDC -> INDEPENDENT 1994 1PGA X-RAY) ===");
    println!(
        "  Exp NOE+Shift : Q_work = {:.4}, plane Q_free = {:.4} | 1PGA (1..56) BB RMSD lowE = {:.3} A, mean = {:.3} A",
        pre_rdc_val.overall_q_work, pre_rdc_val.overall_q_free, pre_rmsd_low, pre_rmsd_mean
    );
    println!(
        "  + TMV+Bicelle : Q_work = {:.4}, plane Q_free = {:.4} (1D_NH Q_work = {:.4}, 1D_NH plane Q_free = {:.4}) | 1PGA (1..56) BB RMSD lowE = {:.3} A, mean = {:.3} A",
        post_rdc_val.overall_q_work,
        post_rdc_val.overall_q_free,
        post_nh_val.overall_q_work,
        post_nh_val.overall_q_free,
        post_rmsd_low,
        post_rmsd_mean
    );

    assert!(
        post_rdc_val.overall_q_work < 0.25
            && post_rdc_val.overall_q_work < pre_rdc_val.overall_q_work,
        "GB1 working Q must improve ({:.4} -> {:.4})",
        pre_rdc_val.overall_q_work,
        post_rdc_val.overall_q_work
    );
    assert!(
        post_rdc_val.overall_q_free < pre_rdc_val.overall_q_free,
        "GB1 peptide-plane Q_free must improve ({:.4} -> {:.4})",
        pre_rdc_val.overall_q_free,
        post_rdc_val.overall_q_free
    );
    assert!(
        post_rdc_val.overall_q_free < 0.55 && post_nh_val.overall_q_free < 0.35,
        "GB1 non-circular peptide-plane Q_free out of range: multi-bond={:.4} (<0.55), 1D_NH={:.4} (<0.35)",
        post_rdc_val.overall_q_free,
        post_nh_val.overall_q_free
    );
    assert!(
        post_rmsd_low < 1.50 && post_rmsd_mean < 2.00 && post_rmsd_low < pre_rmsd_low,
        "non-circular 1PGA (1..56) BB RMSD out of range: lowE={pre_rmsd_low:.3} -> {post_rmsd_low:.3} A, mean={pre_rmsd_mean:.3} -> {post_rmsd_mean:.3} A"
    );
    assert!(
        post_rmsd_mean < pre_rmsd_mean,
        "RDC refinement must improve independent 1PGA (1..56) mean BB RMSD ({pre_rmsd_mean:.3} -> {post_rmsd_mean:.3} A)"
    );
}
