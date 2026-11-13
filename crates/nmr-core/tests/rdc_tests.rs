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
    fit_saupe_tensor_svd, format_rdc_tsv, parse_rdc_tsv_str, partition_rdc_free_set,
    rdc_q_is_better, read_rdc_tsv, saupe_matrix_from_pas, RdcBondType, RdcMediumSet, RdcRestraint,
};
use nmr_core::report::{generate_quality_report_html, ReportInput};
use nmr_core::rng::Rng;
use nmr_core::structure::{
    build_backbone_from_torsions, calculate_structure_with_rdcs_and_dihedrals,
    evaluate_energy_and_gradients_with_rdcs, evaluate_rdc_torsion_energy_and_gradients_fixed_saupe,
    validate_ensemble, EnergyWeights, StructureCalcConfig,
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
        after_rdc_val.overall_q_free < 0.25,
        "held-out 1D3Z Q_free {:.4} must be < 0.25!",
        after_rdc_val.overall_q_free
    );
    assert!(
        after_rdc_val.overall_q_free < before_rdc_val.overall_q_free,
        "RDC refinement must improve held-out Q_free ({:.4} -> {:.4})",
        before_rdc_val.overall_q_free,
        after_rdc_val.overall_q_free
    );
    assert!(
        rmsd_after < rmsd_before && mean_after < mean_before,
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
