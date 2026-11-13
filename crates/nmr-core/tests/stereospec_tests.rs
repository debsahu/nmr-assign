use std::collections::BTreeMap;
use std::path::Path;

use nmr_core::dihedrals::{predict_dihedrals, wrap_deg, SecondaryStructure};
use nmr_core::nef::{
    classify_nef_atom_name, read_nef_str, write_nef_str, NefAtomWildcardKind, NefDocument,
    NefMolecularSystem,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms, read_star_str_all_atoms, Reference};
use nmr_core::noesy::{calc_dihedral_deg, v_add, v_dist, v_scale};
use nmr_core::rng::Rng;
use nmr_core::stereospec::{
    apply_stereospec_to_reference, assign_stereospecific_dataset,
    backbone_conditioning_from_all_atom_pdb, build_local_peptide_frame, format_stereospec_nmrstar,
    infer_prochiral_pair, infer_protein_stereospec, infer_residue_stereospec,
    iupac_cb_proton_dihedrals_deg, iupac_gly_ha_dihedrals_deg, iupac_leu_cd_dihedrals_deg,
    iupac_val_cg_dihedrals_deg, karplus_3j_ccg_hz, karplus_3j_hahb_hz, karplus_3j_ncg_hz,
    karplus_3j_nhb_mag_hz, parse_pdb_all_atoms_file, prochiral_pairs_for_residue,
    read_j_couplings_tsv, synthesize_ideal_observables_from_pdb, to_nef_shift_list_with_stereospec,
    to_star_chem_shift_rows, write_j_couplings_tsv, AllAtomPdbResidue, AnchorProton,
    BackboneConditioning, JCouplingTsvRow, NoePairObs, ProchiralObservables, ProchiralPairSpec,
    ProchiralSiteKind, RotamerState, StereospecConfig, StereospecDatasetInput, SwapHypothesis,
};

fn repo_file(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn perturb_and_scramble_observables(
    ideal: &ProchiralObservables,
    rng: &mut Rng,
    j_sigma_hz: f64,
    noe_sigma_ln: f64,
) -> (ProchiralObservables, SwapHypothesis) {
    let swap = rng.f64() < 0.5;
    let truth = if swap {
        SwapHypothesis::Swapped
    } else {
        SwapHypothesis::Direct
    };
    let perm = |(v1, v2): (f64, f64)| -> (f64, f64) {
        if swap {
            (v2, v1)
        } else {
            (v1, v2)
        }
    };

    let mut noisy = ProchiralObservables {
        proton_shifts_ppm: ideal.proton_shifts_ppm.map(perm),
        carbon_shifts_ppm: ideal.carbon_shifts_ppm.map(perm),
        shared_heavy_shift_ppm: ideal.shared_heavy_shift_ppm,
        j_hh_vicinal_hz: ideal.j_hh_vicinal_hz.map(|(j1, j2)| {
            perm((
                (j1 + rng.normal(0.0, j_sigma_hz)).max(0.0),
                (j2 + rng.normal(0.0, j_sigma_hz)).max(0.0),
            ))
        }),
        j_nhb_hz: ideal.j_nhb_hz.map(|(j1, j2)| {
            perm((
                (j1 + rng.normal(0.0, j_sigma_hz)).abs(),
                (j2 + rng.normal(0.0, j_sigma_hz)).abs(),
            ))
        }),
        j_ncg_hz: ideal.j_ncg_hz.map(|(j1, j2)| {
            perm((
                (j1 + rng.normal(0.0, j_sigma_hz)).max(0.0),
                (j2 + rng.normal(0.0, j_sigma_hz)).max(0.0),
            ))
        }),
        j_cc_vicinal_hz: ideal.j_cc_vicinal_hz.map(|(j1, j2)| {
            perm((
                (j1 + rng.normal(0.0, j_sigma_hz)).max(0.0),
                (j2 + rng.normal(0.0, j_sigma_hz)).max(0.0),
            ))
        }),
        j_methine_hahb_hz: ideal
            .j_methine_hahb_hz
            .map(|jm| (jm + rng.normal(0.0, j_sigma_hz)).max(0.0)),
        j_ccg_single_hz: ideal
            .j_ccg_single_hz
            .map(|jc| (jc + rng.normal(0.0, j_sigma_hz)).max(0.0)),
        noe_pairs: Vec::with_capacity(ideal.noe_pairs.len()),
        next_is_pro: ideal.next_is_pro,
        upstream_chi1_posterior: None,
    };

    for np in &ideal.noe_pairs {
        let i1 = np.intensity_a * rng.normal(0.0, noe_sigma_ln).exp();
        let i2 = np.intensity_b * rng.normal(0.0, noe_sigma_ln).exp();
        let (ia, ib) = perm((i1, i2));
        noisy.noe_pairs.push(NoePairObs::new(np.anchor, ia, ib));
    }

    (noisy, truth)
}

#[test]
fn test_prochiral_pair_catalog_covers_all_20_amino_acids() {
    assert!(prochiral_pairs_for_residue("ALA").is_empty());
    assert!(prochiral_pairs_for_residue("THR").is_empty());

    let gly = prochiral_pairs_for_residue("GLY");
    assert_eq!(gly.len(), 1);
    assert_eq!(gly[0].iupac_branch_1, "HA2");
    assert_eq!(gly[0].iupac_branch_2, "HA3");
    assert_eq!(gly[0].nef_nonstereo_protons, ("HAx", "HAy"));

    let val = prochiral_pairs_for_residue("VAL");
    assert_eq!(val.len(), 1);
    assert_eq!(val[0].iupac_carbons, Some(("CG1", "CG2")));
    assert_eq!(val[0].nef_nonstereo_carbons, Some(("CGx", "CGy")));
    assert_eq!(val[0].nef_stereo_protons, ("HG1%", "HG2%"));
    assert_eq!(val[0].nef_nonstereo_protons, ("HGx%", "HGy%"));

    let ile = prochiral_pairs_for_residue("ILE");
    assert_eq!(ile.len(), 1);
    assert_eq!(ile[0].iupac_branch_1, "HG12");
    assert_eq!(ile[0].iupac_branch_2, "HG13");
    assert_eq!(ile[0].nef_nonstereo_protons, ("HG1x", "HG1y"));

    let leu = prochiral_pairs_for_residue("LEU");
    assert_eq!(leu.len(), 2);
    assert_eq!(leu[0].iupac_branch_1, "HB2");
    assert_eq!(leu[1].iupac_carbons, Some(("CD1", "CD2")));
    assert_eq!(leu[1].nef_nonstereo_carbons, Some(("CDx", "CDy")));
    assert_eq!(leu[1].nef_nonstereo_protons, ("HDx%", "HDy%"));

    assert_eq!(prochiral_pairs_for_residue("ASN").len(), 2);
    assert_eq!(prochiral_pairs_for_residue("GLN").len(), 3);
    assert_eq!(prochiral_pairs_for_residue("GLU").len(), 2);
    assert_eq!(prochiral_pairs_for_residue("MET").len(), 2);
    assert_eq!(prochiral_pairs_for_residue("ARG").len(), 3);
    assert_eq!(prochiral_pairs_for_residue("PRO").len(), 3);
    assert_eq!(prochiral_pairs_for_residue("LYS").len(), 4);

    for single_cb in ["ASP", "CYS", "HIS", "PHE", "SER", "TRP", "TYR"] {
        let p = prochiral_pairs_for_residue(single_cb);
        assert_eq!(p.len(), 1, "expected single CB methylene on {single_cb}");
        assert_eq!(p[0].nef_nonstereo_protons, ("HBx", "HBy"));
    }
}

#[test]
fn test_1d3z_ubiquitin_exact_iupac_dihedral_offsets_zero_inversion() {
    let pdb = parse_pdb_all_atoms_file(repo_file("data/pdb_refs/1d3z.pdb"))
        .expect("parse 1d3z all atoms");
    assert_eq!(pdb.len(), 76);

    let mut cb_checked = 0usize;
    let mut val_checked = 0usize;
    let mut leu_checked = 0usize;
    let mut gly_checked = 0usize;
    let mut ile_checked = 0usize;
    let mut carboxamide_checked = 0usize;

    let mut dist_err_h_sum = 0.0;
    let mut dist_err_h_n = 0usize;
    let mut dist_err_ha_sum = 0.0;
    let mut dist_err_ha_n = 0usize;
    let mut dist_err_hn1_sum = 0.0;
    let mut dist_err_hn1_n = 0usize;

    for (idx, r) in pdb.iter().enumerate() {
        // 1. Verify all 15 CB-methylene residue types against IUPAC HB2/HB3 dihedral formulas
        let specs = prochiral_pairs_for_residue(&r.res_name);
        if specs
            .iter()
            .any(|s| s.kind == ProchiralSiteKind::BetaMethylene)
        {
            let (n, ca, cb, ha, hb2, hb3) = (
                r.coord("N").unwrap(),
                r.coord("CA").unwrap(),
                r.coord("CB").unwrap(),
                r.coord("HA").unwrap(),
                r.coord("HB2").unwrap(),
                r.coord("HB3").unwrap(),
            );
            let xg_name = ["CG", "SG", "OG"]
                .into_iter()
                .find(|name| r.coord(name).is_some())
                .expect("gamma heavy atom");
            let xg = r.coord(xg_name).unwrap();
            let chi1 = calc_dihedral_deg(n, ca, cb, xg);
            let (pred_ha_hb2, pred_ha_hb3, pred_n_hb2, pred_n_hb3) =
                iupac_cb_proton_dihedrals_deg(chi1);

            let obs_ha_hb2 = calc_dihedral_deg(ha, ca, cb, hb2);
            let obs_ha_hb3 = calc_dihedral_deg(ha, ca, cb, hb3);
            let obs_n_hb2 = calc_dihedral_deg(n, ca, cb, hb2);
            let obs_n_hb3 = calc_dihedral_deg(n, ca, cb, hb3);

            assert!(
                wrap_deg(obs_ha_hb2 - pred_ha_hb2).abs() < 6.5,
                "{} {}: HA-HB2 obs={obs_ha_hb2:.1} pred={pred_ha_hb2:.1}",
                r.res_name,
                r.seq_id
            );
            assert!(
                wrap_deg(obs_ha_hb3 - pred_ha_hb3).abs() < 6.5,
                "{} {}: HA-HB3 obs={obs_ha_hb3:.1} pred={pred_ha_hb3:.1}",
                r.res_name,
                r.seq_id
            );
            assert!(
                wrap_deg(obs_n_hb2 - pred_n_hb2).abs() < 6.5,
                "{} {}: N-HB2 obs={obs_n_hb2:.1} pred={pred_n_hb2:.1}",
                r.res_name,
                r.seq_id
            );
            assert!(
                wrap_deg(obs_n_hb3 - pred_n_hb3).abs() < 6.5,
                "{} {}: N-HB3 obs={obs_n_hb3:.1} pred={pred_n_hb3:.1}",
                r.res_name,
                r.seq_id
            );
            cb_checked += 1;

            // Validate analytical NeRF local peptide frame proton-proton distances
            if let BackboneConditioning::ExactTorsions { phi_deg, psi_deg } =
                backbone_conditioning_from_all_atom_pdb(&pdb, idx)
            {
                let next_pro = pdb
                    .get(idx + 1)
                    .map(|nx| nx.res_name == "PRO")
                    .unwrap_or(false);
                let frame = build_local_peptide_frame(&r.res_name, phi_deg, psi_deg, next_pro);
                let cb_f = frame.cb.unwrap();
                let hb2_f = nmr_core::structure::nerf_place_atom(
                    frame.n,
                    frame.ca,
                    cb_f,
                    1.090,
                    109.5_f64.to_radians(),
                    (chi1 + 120.0).to_radians(),
                );
                let hb3_f = nmr_core::structure::nerf_place_atom(
                    frame.n,
                    frame.ca,
                    cb_f,
                    1.090,
                    109.5_f64.to_radians(),
                    (chi1 - 120.0).to_radians(),
                );
                dist_err_ha_sum += (v_dist(frame.ha, hb2_f) - v_dist(ha, hb2)).abs()
                    + (v_dist(frame.ha, hb3_f) - v_dist(ha, hb3)).abs();
                dist_err_ha_n += 2;
                if let (Some(hf), Some(hp)) = (frame.h, r.coord("H")) {
                    dist_err_h_sum += (v_dist(hf, hb2_f) - v_dist(hp, hb2)).abs()
                        + (v_dist(hf, hb3_f) - v_dist(hp, hb3)).abs();
                    dist_err_h_n += 2;
                }
                if let (Some(hn1f), Some(hn1p)) =
                    (frame.h_next, pdb.get(idx + 1).and_then(|nx| nx.coord("H")))
                {
                    dist_err_hn1_sum += (v_dist(hn1f, hb2_f) - v_dist(hn1p, hb2)).abs()
                        + (v_dist(hn1f, hb3_f) - v_dist(hn1p, hb3)).abs();
                    dist_err_hn1_n += 2;
                }
            }
        }

        // 2. Verify Val CG1/CG2 IUPAC offsets
        if r.res_name == "VAL" {
            let (n, ca, c, cb, ha, hb, cg1, cg2) = (
                r.coord("N").unwrap(),
                r.coord("CA").unwrap(),
                r.coord("C").unwrap(),
                r.coord("CB").unwrap(),
                r.coord("HA").unwrap(),
                r.coord("HB").unwrap(),
                r.coord("CG1").unwrap(),
                r.coord("CG2").unwrap(),
            );
            let chi1 = calc_dihedral_deg(n, ca, cb, cg1);
            let (pn1, pn2, pc1, pc2, pha_hb) = iupac_val_cg_dihedrals_deg(chi1);
            assert!(wrap_deg(calc_dihedral_deg(n, ca, cb, cg1) - pn1).abs() < 1e-6);
            assert!(wrap_deg(calc_dihedral_deg(n, ca, cb, cg2) - pn2).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(c, ca, cb, cg1) - pc1).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(c, ca, cb, cg2) - pc2).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(ha, ca, cb, hb) - pha_hb).abs() < 5.0);
            val_checked += 1;
        }

        // 3. Verify Leu CD1/CD2 IUPAC offsets
        if r.res_name == "LEU" {
            let (ca, cb, cg, hg, cd1, cd2) = (
                r.coord("CA").unwrap(),
                r.coord("CB").unwrap(),
                r.coord("CG").unwrap(),
                r.coord("HG").unwrap(),
                r.coord("CD1").unwrap(),
                r.coord("CD2").unwrap(),
            );
            let chi2 = calc_dihedral_deg(ca, cb, cg, cd1);
            let (pcd1, pcd2, phg) = iupac_leu_cd_dihedrals_deg(chi2);
            assert!(wrap_deg(calc_dihedral_deg(ca, cb, cg, cd1) - pcd1).abs() < 1e-6);
            assert!(wrap_deg(calc_dihedral_deg(ca, cb, cg, cd2) - pcd2).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(ca, cb, cg, hg) - phg).abs() < 5.0);
            leu_checked += 1;
        }

        // 4. Verify Gly HA2/HA3 IUPAC offsets
        if r.res_name == "GLY" && idx > 0 && idx + 1 < pdb.len() {
            let c_prev = pdb[idx - 1].coord("C").unwrap();
            let (n, h, ca, c, ha2, ha3) = (
                r.coord("N").unwrap(),
                r.coord("H").unwrap(),
                r.coord("CA").unwrap(),
                r.coord("C").unwrap(),
                r.coord("HA2").unwrap(),
                r.coord("HA3").unwrap(),
            );
            let phi = calc_dihedral_deg(c_prev, n, ca, c);
            let (pc_ha2, pc_ha3, ph_ha2, ph_ha3) = iupac_gly_ha_dihedrals_deg(phi);
            assert!(wrap_deg(calc_dihedral_deg(c_prev, n, ca, ha2) - pc_ha2).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(c_prev, n, ca, ha3) - pc_ha3).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(h, n, ca, ha2) - ph_ha2).abs() < 6.0);
            assert!(wrap_deg(calc_dihedral_deg(h, n, ca, ha3) - ph_ha3).abs() < 6.0);
            gly_checked += 1;
        }

        // 5. Verify (2S,3S)-Ile CB/HB and CG1/HG12/HG13 IUPAC offsets
        if r.res_name == "ILE" {
            let (n, ca, cb, ha, hb, cg1, cd1, hg12, hg13) = (
                r.coord("N").unwrap(),
                r.coord("CA").unwrap(),
                r.coord("CB").unwrap(),
                r.coord("HA").unwrap(),
                r.coord("HB").unwrap(),
                r.coord("CG1").unwrap(),
                r.coord("CD1").unwrap(),
                r.coord("HG12").unwrap(),
                r.coord("HG13").unwrap(),
            );
            let chi1 = calc_dihedral_deg(n, ca, cb, cg1);
            let chi2 = calc_dihedral_deg(ca, cb, cg1, cd1);
            assert!(wrap_deg(calc_dihedral_deg(n, ca, cb, hb) - (chi1 + 120.0)).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(ha, ca, cb, hb) - (chi1 - 120.0)).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(ca, cb, cg1, hg12) - (chi2 + 120.0)).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(ca, cb, cg1, hg13) - (chi2 - 120.0)).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(hb, cb, cg1, hg12) - chi2).abs() < 5.0);
            assert!(wrap_deg(calc_dihedral_deg(hb, cb, cg1, hg13) - (chi2 + 120.0)).abs() < 5.0);
            ile_checked += 1;
        }

        // 6. Verify Asn HD21/HD22 & Gln HE21/HE22 planar syn/anti orientation
        if r.res_name == "ASN" {
            let (cg, od1, nd2, hb2, hb3, hd21, hd22) = (
                r.coord("CG").unwrap(),
                r.coord("OD1").unwrap(),
                r.coord("ND2").unwrap(),
                r.coord("HB2").unwrap(),
                r.coord("HB3").unwrap(),
                r.coord("HD21").unwrap(),
                r.coord("HD22").unwrap(),
            );
            assert!(wrap_deg(calc_dihedral_deg(od1, cg, nd2, hd21) - 180.0).abs() < 5.0);
            assert!(calc_dihedral_deg(od1, cg, nd2, hd22).abs() < 5.0);
            let qb = v_scale(v_add(hb2, hb3), 0.5);
            assert!(v_dist(qb, hd21) < v_dist(qb, hd22));
            carboxamide_checked += 1;
        }
        if r.res_name == "GLN" {
            let (cd, oe1, ne2, hg2, hg3, he21, he22) = (
                r.coord("CD").unwrap(),
                r.coord("OE1").unwrap(),
                r.coord("NE2").unwrap(),
                r.coord("HG2").unwrap(),
                r.coord("HG3").unwrap(),
                r.coord("HE21").unwrap(),
                r.coord("HE22").unwrap(),
            );
            assert!(wrap_deg(calc_dihedral_deg(oe1, cd, ne2, he21) - 180.0).abs() < 5.0);
            assert!(calc_dihedral_deg(oe1, cd, ne2, he22).abs() < 5.0);
            let qg = v_scale(v_add(hg2, hg3), 0.5);
            assert!(v_dist(qg, he21) < v_dist(qg, he22));
            carboxamide_checked += 1;
        }
    }

    assert_eq!(cb_checked, 50);
    assert_eq!(val_checked, 4);
    assert_eq!(leu_checked, 9);
    assert_eq!(gly_checked, 5);
    assert_eq!(ile_checked, 7);
    assert_eq!(carboxamide_checked, 8);

    let mae_h = dist_err_h_sum / (dist_err_h_n as f64);
    let mae_ha = dist_err_ha_sum / (dist_err_ha_n as f64);
    let mae_hn1 = dist_err_hn1_sum / (dist_err_hn1_n as f64);
    assert!(mae_h < 0.05, "NeRF H_i-HB MAE = {mae_h:.4} A");
    assert!(mae_ha < 0.05, "NeRF HA_i-HB MAE = {mae_ha:.4} A");
    assert!(mae_hn1 < 0.05, "NeRF H_{{i+1}}-HB MAE = {mae_hn1:.4} A");
}

#[test]
fn test_1d3z_ubiquitin_10_seed_noisy_scrambled_precision_gate() {
    let pdb = parse_pdb_all_atoms_file(repo_file("data/pdb_refs/1d3z.pdb"))
        .expect("parse 1d3z all atoms");
    let star_ref = read_star(repo_file("data/fixtures/synthetic_ubiquitin.str"), None)
        .expect("read synthetic_ubiquitin.str");
    assert_eq!(pdb.len(), star_ref.sequence.len());
    for (p, s) in pdb.iter().zip(star_ref.sequence.iter()) {
        assert_eq!(p.res_name, s.to_ascii_uppercase());
    }

    let pdb_backbones: Vec<BackboneConditioning> = (0..pdb.len())
        .map(|i| backbone_conditioning_from_all_atom_pdb(&pdb, i))
        .collect();
    let csi_restraints = predict_dihedrals(&star_ref);
    let csi_backbones: Vec<BackboneConditioning> = csi_restraints
        .iter()
        .map(BackboneConditioning::from_dihedral_restraint)
        .collect();

    let cfg = StereospecConfig::default();
    assert!((cfg.commit_posterior_threshold - 0.90).abs() < 1e-12);

    // Realistic experimental noise: +-0.70 Hz Gaussian J-coupling noise and +-25% log-normal NOE intensity noise
    let j_sigma_hz = 0.70;
    let noe_sigma_ln = 0.25;

    for (mode_name, backbones) in [
        ("1D3Z_structure_backbone", &pdb_backbones),
        ("CSI_chemical_shift_backbone", &csi_backbones),
    ] {
        let mut methylene_total = 0usize;
        let mut methylene_committed = 0usize;
        let mut methylene_correct = 0usize;

        let mut isopropyl_total = 0usize;
        let mut isopropyl_committed = 0usize;
        let mut isopropyl_correct = 0usize;

        let mut _all_total = 0usize;
        let mut all_committed = 0usize;
        let mut all_correct = 0usize;

        for seed in 0..10u64 {
            let mut rng = Rng::new(0x57E2_E050_EC10_0000 ^ seed);
            let mut per_res_obs: Vec<BTreeMap<String, ProchiralObservables>> =
                vec![BTreeMap::new(); pdb.len()];
            let mut ground_truth: BTreeMap<(usize, &'static str), SwapHypothesis> = BTreeMap::new();

            for (res_idx, res) in pdb.iter().enumerate() {
                for spec in prochiral_pairs_for_residue(&res.res_name) {
                    if let Some(mut ideal) =
                        synthesize_ideal_observables_from_pdb(&pdb, res_idx, spec)
                    {
                        // Attach realistic chemical-shift ordering for carboxamides (HD21/HE21 downfield)
                        if matches!(
                            spec.kind,
                            ProchiralSiteKind::AsnCarboxamide | ProchiralSiteKind::GlnCarboxamide
                        ) {
                            ideal.proton_shifts_ppm =
                                Some((7.45 + rng.normal(0.0, 0.12), 6.85 + rng.normal(0.0, 0.12)));
                        }
                        let (noisy, truth) = perturb_and_scramble_observables(
                            &ideal,
                            &mut rng,
                            j_sigma_hz,
                            noe_sigma_ln,
                        );
                        per_res_obs[res_idx].insert(spec.iupac_branch_1.to_string(), noisy);
                        ground_truth.insert((res_idx, spec.iupac_branch_1), truth);
                    }
                }
            }

            let results = infer_protein_stereospec(
                &star_ref.sequence,
                &star_ref.seq_ids,
                backbones,
                &per_res_obs,
                &cfg,
            );

            for r in &results {
                let Some(&truth) = ground_truth.get(&(r.residue_idx, r.spec.iupac_branch_1)) else {
                    // Distal CD/CE pairs without synthesized observables MUST stay uncommitted!
                    assert!(
                        !r.committed && r.nmr_star_ambiguity_code == 2,
                        "unobserved distal site {} {} falsely committed",
                        r.res_name,
                        r.seq_id
                    );
                    continue;
                };

                _all_total += 1;
                if r.committed {
                    assert_eq!(r.nmr_star_ambiguity_code, 1);
                    assert!(r.winning_posterior >= 0.90);
                    all_committed += 1;
                    if r.winning_hypothesis == truth {
                        all_correct += 1;
                    }
                } else {
                    assert_eq!(r.nmr_star_ambiguity_code, 2);
                }

                match r.spec.kind {
                    ProchiralSiteKind::BetaMethylene
                    | ProchiralSiteKind::GammaMethylene
                    | ProchiralSiteKind::GlyAlphaMethylene => {
                        methylene_total += 1;
                        if r.committed {
                            methylene_committed += 1;
                            if r.winning_hypothesis == truth {
                                methylene_correct += 1;
                            }
                        }
                    }
                    ProchiralSiteKind::ValIsopropyl | ProchiralSiteKind::LeuIsopropyl => {
                        isopropyl_total += 1;
                        if r.committed {
                            isopropyl_committed += 1;
                            if r.winning_hypothesis == truth {
                                isopropyl_correct += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let meth_iso_committed = methylene_committed + isopropyl_committed;
        let meth_iso_correct = methylene_correct + isopropyl_correct;
        let meth_iso_prec = 100.0 * (meth_iso_correct as f64) / (meth_iso_committed as f64);
        let meth_prec = 100.0 * (methylene_correct as f64) / (methylene_committed as f64);
        let iso_prec = 100.0 * (isopropyl_correct as f64) / (isopropyl_committed as f64);
        let all_prec = 100.0 * (all_correct as f64) / (all_committed as f64);

        println!(
            "[{mode_name}] 10-seed Ubiquitin 1D3Z: meth+iso committed {meth_iso_committed}/{} ({:.1}% cov), precision = {meth_iso_prec:.2}% (methylene {methylene_correct}/{methylene_committed} = {meth_prec:.2}%, Val/Leu {isopropyl_correct}/{isopropyl_committed} = {iso_prec:.2}%, all sites {all_correct}/{all_committed} = {all_prec:.2}%)",
            methylene_total + isopropyl_total,
            100.0 * (meth_iso_committed as f64) / ((methylene_total + isopropyl_total) as f64),
        );

        assert!(
            meth_iso_committed >= (methylene_total + isopropyl_total) / 2,
            "[{mode_name}] too few committed prochiral pairs: {meth_iso_committed}"
        );
        assert!(
            meth_iso_prec >= 95.0,
            "[{mode_name}] committed methylene + Val/Leu precision {meth_iso_prec:.2}% < 95.0% gate"
        );
        assert!(
            meth_prec >= 95.0,
            "[{mode_name}] committed methylene precision {meth_prec:.2}% < 95.0%"
        );
        assert!(
            iso_prec >= 95.0,
            "[{mode_name}] committed Val/Leu precision {iso_prec:.2}% < 95.0%"
        );
        assert!(
            all_prec >= 95.0,
            "[{mode_name}] overall committed precision {all_prec:.2}% < 95.0%"
        );
    }
}

#[test]
fn test_calibration_zero_false_swaps_under_absent_isotropic_or_symmetric_inputs() {
    let cfg = StereospecConfig::default();
    let star_ref = read_star(repo_file("data/fixtures/synthetic_ubiquitin.str"), None)
        .expect("read synthetic_ubiquitin.str");
    let csi_restraints = predict_dihedrals(&star_ref);
    let backbones: Vec<BackboneConditioning> = csi_restraints
        .iter()
        .map(BackboneConditioning::from_dihedral_restraint)
        .collect();

    // 1. Absent couplings/NOEs across entire 76-residue Ubiquitin sequence -> 0 committed!
    let empty_obs = vec![BTreeMap::new(); star_ref.sequence.len()];
    let absent_results = infer_protein_stereospec(
        &star_ref.sequence,
        &star_ref.seq_ids,
        &backbones,
        &empty_obs,
        &cfg,
    );
    assert!(!absent_results.is_empty());
    for r in &absent_results {
        assert!(
            (r.posterior_direct - 0.50).abs() < 1e-12,
            "absent data gave posterior {} on {} {}",
            r.posterior_direct,
            r.res_name,
            r.seq_id
        );
        assert!(!r.committed);
        assert_eq!(r.nmr_star_ambiguity_code, 2);
        let (p1, p2) = r.nef_proton_labels_for_ab();
        assert_eq!(
            classify_nef_atom_name(p1),
            NefAtomWildcardKind::NonStereoProchiral
        );
        assert_eq!(
            classify_nef_atom_name(p2),
            NefAtomWildcardKind::NonStereoProchiral
        );
        if let Some((c1, c2)) = r.nef_carbon_labels_for_ab() {
            assert_eq!(
                classify_nef_atom_name(c1),
                NefAtomWildcardKind::NonStereoProchiral
            );
            assert_eq!(
                classify_nef_atom_name(c2),
                NefAtomWildcardKind::NonStereoProchiral
            );
        }
    }

    // 2. Exact (1/3, 1/3, 1/3) rotamer-averaged couplings and symmetric NOEs -> posterior == 0.50!
    let j_iso_hahb =
        (karplus_3j_hahb_hz(-60.0) + karplus_3j_hahb_hz(60.0) + karplus_3j_hahb_hz(180.0)) / 3.0;
    let j_iso_nhb =
        (karplus_3j_nhb_mag_hz(-60.0) + karplus_3j_nhb_mag_hz(60.0) + karplus_3j_nhb_mag_hz(180.0))
            / 3.0;
    let j_iso_ncg = (karplus_3j_ncg_hz("VAL", -60.0)
        + karplus_3j_ncg_hz("VAL", 60.0)
        + karplus_3j_ncg_hz("VAL", 180.0))
        / 3.0;
    let j_iso_ccg = (karplus_3j_ccg_hz("VAL", -60.0)
        + karplus_3j_ccg_hz("VAL", 60.0)
        + karplus_3j_ccg_hz("VAL", 180.0))
        / 3.0;

    let asp_cb = prochiral_pairs_for_residue("ASP")[0];
    let val_cg = prochiral_pairs_for_residue("VAL")[0];
    let exact_bb = BackboneConditioning::ExactTorsions {
        phi_deg: -64.0,
        psi_deg: -41.0,
    };

    let iso_cb_res = infer_prochiral_pair(
        0,
        1,
        asp_cb,
        &exact_bb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((2.85, 2.52)),
            j_hh_vicinal_hz: Some((j_iso_hahb, j_iso_hahb)),
            j_nhb_hz: Some((j_iso_nhb, j_iso_nhb)),
            noe_pairs: vec![
                NoePairObs::new(AnchorProton::IntraAmideH, 1.0, 1.0),
                NoePairObs::new(AnchorProton::IntraAlphaHA, 1.0, 1.0),
                NoePairObs::new(AnchorProton::SeqNextAmideH, 1.0, 1.0),
            ],
            ..Default::default()
        },
        &cfg,
    );
    assert!((iso_cb_res.winning_posterior - 0.50).abs() < 1e-12);
    assert!(!iso_cb_res.committed);
    assert_eq!(iso_cb_res.nmr_star_ambiguity_code, 2);
    assert!(iso_cb_res.dynamic_mixture_posterior > 0.70);

    let iso_val_res = infer_prochiral_pair(
        0,
        2,
        val_cg,
        &exact_bb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((0.96, 0.84)),
            carbon_shifts_ppm: Some((21.4, 19.8)),
            j_ncg_hz: Some((j_iso_ncg, j_iso_ncg)),
            j_cc_vicinal_hz: Some((j_iso_ccg, j_iso_ccg)),
            j_methine_hahb_hz: Some(j_iso_hahb),
            noe_pairs: vec![
                NoePairObs::new(AnchorProton::IntraAmideH, 1.0, 1.0),
                NoePairObs::new(AnchorProton::IntraAlphaHA, 1.0, 1.0),
            ],
            ..Default::default()
        },
        &cfg,
    );
    assert!((iso_val_res.winning_posterior - 0.50).abs() < 1e-12);
    assert!(!iso_val_res.committed);
    assert_eq!(iso_val_res.nmr_star_ambiguity_code, 2);

    // 3. Noisy (1/3, 1/3, 1/3) rotamer-averaged couplings (+-0.7 Hz) & NOEs (+-25%) across 15 seeds:
    //    Prove ZERO uncalibrated false swaps exported as Ambiguity_code = 1!
    let mut false_commits = 0usize;
    for seed in 0..15u64 {
        let mut rng = Rng::new(0xCA11_B2A7_0000_0000 ^ seed);
        for (res_idx, res_name) in star_ref.sequence.iter().enumerate() {
            for spec in prochiral_pairs_for_residue(res_name) {
                let obs = match spec.kind {
                    ProchiralSiteKind::BetaMethylene => ProchiralObservables {
                        j_hh_vicinal_hz: Some((
                            (j_iso_hahb + rng.normal(0.0, 0.70)).max(0.0),
                            (j_iso_hahb + rng.normal(0.0, 0.70)).max(0.0),
                        )),
                        j_nhb_hz: Some((
                            (j_iso_nhb + rng.normal(0.0, 0.70)).abs(),
                            (j_iso_nhb + rng.normal(0.0, 0.70)).abs(),
                        )),
                        noe_pairs: vec![
                            NoePairObs::new(
                                AnchorProton::IntraAmideH,
                                rng.normal(0.0, 0.25).exp(),
                                rng.normal(0.0, 0.25).exp(),
                            ),
                            NoePairObs::new(
                                AnchorProton::IntraAlphaHA,
                                rng.normal(0.0, 0.25).exp(),
                                rng.normal(0.0, 0.25).exp(),
                            ),
                        ],
                        ..Default::default()
                    },
                    ProchiralSiteKind::ValIsopropyl => ProchiralObservables {
                        j_ncg_hz: Some((
                            (j_iso_ncg + rng.normal(0.0, 0.70)).max(0.0),
                            (j_iso_ncg + rng.normal(0.0, 0.70)).max(0.0),
                        )),
                        j_cc_vicinal_hz: Some((
                            (j_iso_ccg + rng.normal(0.0, 0.70)).max(0.0),
                            (j_iso_ccg + rng.normal(0.0, 0.70)).max(0.0),
                        )),
                        j_methine_hahb_hz: Some((j_iso_hahb + rng.normal(0.0, 0.70)).max(0.0)),
                        noe_pairs: vec![NoePairObs::new(
                            AnchorProton::IntraAmideH,
                            rng.normal(0.0, 0.25).exp(),
                            rng.normal(0.0, 0.25).exp(),
                        )],
                        ..Default::default()
                    },
                    _ => ProchiralObservables {
                        noe_pairs: vec![NoePairObs::new(
                            AnchorProton::IntraAmideH,
                            rng.normal(0.0, 0.25).exp(),
                            rng.normal(0.0, 0.25).exp(),
                        )],
                        ..Default::default()
                    },
                };
                let r = infer_prochiral_pair(
                    res_idx,
                    star_ref.seq_ids[res_idx],
                    spec,
                    &backbones[res_idx],
                    &obs,
                    &cfg,
                );
                // If noise happens to push Direct < 0.50 (Swapped) or >= 0.50 (Direct), it must NEVER commit >= 0.90!
                if r.committed {
                    false_commits += 1;
                }
            }
        }
    }
    assert_eq!(
        false_commits, 0,
        "expected 0 uncalibrated false swaps on rotamer-averaged inputs, got {false_commits}"
    );

    // 4. Classic single-pathway 3J_HAHB-only degeneracy: (g-, Direct) <-> (t, Swapped) MUST stay < 0.90!
    let single_j_res = infer_prochiral_pair(
        0,
        10,
        asp_cb,
        &BackboneConditioning::SecondaryStructurePrior(SecondaryStructure::Sheet),
        &ProchiralObservables {
            j_hh_vicinal_hz: Some((3.1, 10.9)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(
        !single_j_res.committed && single_j_res.nmr_star_ambiguity_code == 2,
        "3J_HAHB alone must not commit across (g-, Direct) <-> (t, Swapped) degeneracy (got posterior {:.3})",
        single_j_res.winning_posterior
    );

    // 5. Standalone Asn/Gln chemical shifts without NOEs stay < 0.90, whereas adding QB NOE commits!
    let asn_nh2 = prochiral_pairs_for_residue("ASN")[1];
    let shift_only_asn = infer_prochiral_pair(
        0,
        25,
        asn_nh2,
        &BackboneConditioning::Marginalized,
        &ProchiralObservables {
            proton_shifts_ppm: Some((7.55, 6.82)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(shift_only_asn.posterior_direct > 0.55 && shift_only_asn.winning_posterior < 0.90);
    assert!(!shift_only_asn.committed);

    let noe_plus_shift_asn = infer_prochiral_pair(
        0,
        25,
        asn_nh2,
        &BackboneConditioning::Marginalized,
        &ProchiralObservables {
            proton_shifts_ppm: Some((7.55, 6.82)),
            noe_pairs: vec![NoePairObs::new(AnchorProton::IntraBetaHB, 8.5, 1.0)],
            ..Default::default()
        },
        &cfg,
    );
    assert!(noe_plus_shift_asn.committed);
    assert_eq!(
        noe_plus_shift_asn.winning_hypothesis,
        SwapHypothesis::Direct
    );
    assert_eq!(noe_plus_shift_asn.nmr_star_ambiguity_code, 1);
}

#[test]
fn test_simplex_rotamer_mixture_population_inference() {
    let cfg = StereospecConfig::default();
    let phe_cb = prochiral_pairs_for_residue("PHE")[0];
    let phi = -120.0;
    let psi = 135.0;
    let bb = BackboneConditioning::ExactTorsions {
        phi_deg: phi,
        psi_deg: psi,
    };
    let frame = build_local_peptide_frame("PHE", phi, psi, false);
    let cb = frame.cb.unwrap();
    let h = frame.h.unwrap();
    let ha = frame.ha;
    let hn1 = frame.h_next.unwrap();

    // Synthesize a 55% g- (-64 deg) + 45% t (178 deg) 2-state rotamer mixture in Swapped (a=HB3, b=HB2) order
    let w_true = [0.55, 0.45, 0.00];
    let chis = [-64.0_f64, 178.0, 62.0];
    let mut j_hb2 = 0.0;
    let mut j_hb3 = 0.0;
    let mut jn_hb2 = 0.0;
    let mut jn_hb3 = 0.0;
    let mut r6_h = (0.0, 0.0);
    let mut r6_ha = (0.0, 0.0);
    let mut r6_hn1 = (0.0, 0.0);
    let floor = 5.5_f64.powi(-6);

    for (&wt, &chi) in w_true.iter().zip(&chis) {
        j_hb2 += wt * karplus_3j_hahb_hz(chi - 120.0);
        j_hb3 += wt * karplus_3j_hahb_hz(chi);
        jn_hb2 += wt * karplus_3j_nhb_mag_hz(chi + 120.0);
        jn_hb3 += wt * karplus_3j_nhb_mag_hz(chi - 120.0);
        let hb2 = nmr_core::structure::nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            1.090,
            109.5_f64.to_radians(),
            (chi + 120.0).to_radians(),
        );
        let hb3 = nmr_core::structure::nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            1.090,
            109.5_f64.to_radians(),
            (chi - 120.0).to_radians(),
        );
        r6_h.0 += wt / v_dist(h, hb2).powi(6);
        r6_h.1 += wt / v_dist(h, hb3).powi(6);
        r6_ha.0 += wt / v_dist(ha, hb2).powi(6);
        r6_ha.1 += wt / v_dist(ha, hb3).powi(6);
        r6_hn1.0 += wt / v_dist(hn1, hb2).powi(6);
        r6_hn1.1 += wt / v_dist(hn1, hb3).powi(6);
    }

    // Pass in Swapped order: a = HB3, b = HB2
    let obs = ProchiralObservables {
        j_hh_vicinal_hz: Some((j_hb3, j_hb2)),
        j_nhb_hz: Some((jn_hb3, jn_hb2)),
        noe_pairs: vec![
            NoePairObs::new(AnchorProton::IntraAmideH, r6_h.1 + floor, r6_h.0 + floor),
            NoePairObs::new(AnchorProton::IntraAlphaHA, r6_ha.1 + floor, r6_ha.0 + floor),
            NoePairObs::new(
                AnchorProton::SeqNextAmideH,
                r6_hn1.1 + floor,
                r6_hn1.0 + floor,
            ),
        ],
        ..Default::default()
    };

    let res = infer_prochiral_pair(0, 4, phe_cb, &bb, &obs, &cfg);
    assert_eq!(res.winning_hypothesis, SwapHypothesis::Swapped);
    assert!(res.committed);
    assert!(
        res.rotamer_marginal[RotamerState::GaucheMinus.index()] > 0.30
            && res.rotamer_marginal[RotamerState::Trans.index()] > 0.30
            && res.rotamer_marginal[RotamerState::GauchePlus.index()] < 0.15,
        "expected mixed (g-, t) marginal, got {:?}",
        res.rotamer_marginal
    );
    assert!(
        res.dynamic_mixture_posterior > 0.50,
        "expected interior simplex mixture posterior > 0.50, got {:.3}",
        res.dynamic_mixture_posterior
    );
}

#[test]
fn test_nmrstar_ambiguity_code_and_nef_v1_1_x_y_roundtrip_export() {
    let pdb = parse_pdb_all_atoms_file(repo_file("data/pdb_refs/1d3z.pdb"))
        .expect("parse 1d3z all atoms");
    let cfg = StereospecConfig::default();

    // Build a 2-residue test system from 1D3Z VAL 5 (committed Swapped) and LEU 8 (CB committed Direct, CD uncommitted)
    let base = Reference {
        sequence: vec!["VAL".to_string(), "LEU".to_string()],
        seq_ids: vec![1, 2],
        shifts: BTreeMap::from([
            ((0, "H".to_string()), 9.30),
            ((0, "N".to_string()), 121.2),
            ((0, "CA".to_string()), 60.2),
            ((0, "CB".to_string()), 33.8),
            ((1, "H".to_string()), 9.10),
            ((1, "N".to_string()), 121.5),
            ((1, "CA".to_string()), 57.5),
        ]),
        ..Default::default()
    };

    // Synthesize VAL 5 in Swapped order: a = CG2/HG2, b = CG1/HG1
    let val_spec = prochiral_pairs_for_residue("VAL")[0];
    let val_ideal = synthesize_ideal_observables_from_pdb(&pdb, 4, val_spec).unwrap();
    let val_swapped = ProchiralObservables {
        proton_shifts_ppm: Some((0.74, 0.89)), // a=0.74 (CG2/HG2), b=0.89 (CG1/HG1)
        carbon_shifts_ppm: Some((19.6, 21.8)), // a=19.6 (CG2), b=21.8 (CG1)
        j_ncg_hz: val_ideal.j_ncg_hz.map(|(x, y)| (y, x)),
        j_cc_vicinal_hz: val_ideal.j_cc_vicinal_hz.map(|(x, y)| (y, x)),
        j_methine_hahb_hz: val_ideal.j_methine_hahb_hz,
        noe_pairs: val_ideal
            .noe_pairs
            .iter()
            .map(|np| NoePairObs::new(np.anchor, np.intensity_b, np.intensity_a))
            .collect(),
        ..Default::default()
    };
    let val_bb = backbone_conditioning_from_all_atom_pdb(&pdb, 4);
    let val_res = infer_prochiral_pair(0, 1, val_spec, &val_bb, &val_swapped, &cfg);
    assert!(val_res.committed);
    assert_eq!(val_res.winning_hypothesis, SwapHypothesis::Swapped);
    assert_eq!(val_res.nmr_star_ambiguity_code, 1);
    assert_eq!(val_res.nef_carbon_labels_for_ab(), Some(("CG2", "CG1")));
    assert_eq!(val_res.nef_proton_labels_for_ab(), ("HG2%", "HG1%"));

    // LEU 8: give CB (HB2/HB3) ideal observables, leave CD (CD1/CD2) with shifts ONLY (no J/NOEs -> uncommitted!)
    let leu_cb_spec = prochiral_pairs_for_residue("LEU")[0];
    let mut leu_cb_obs = synthesize_ideal_observables_from_pdb(&pdb, 7, leu_cb_spec).unwrap();
    leu_cb_obs.proton_shifts_ppm = Some((1.72, 1.48));
    leu_cb_obs.shared_heavy_shift_ppm = Some(41.9);

    let leu_obs_map = BTreeMap::from([
        ("HB2".to_string(), leu_cb_obs),
        (
            "CD1".to_string(),
            ProchiralObservables {
                proton_shifts_ppm: Some((0.81, 0.94)),
                carbon_shifts_ppm: Some((22.4, 24.9)),
                ..Default::default()
            },
        ),
    ]);
    let leu_bb = backbone_conditioning_from_all_atom_pdb(&pdb, 7);
    let leu_results = infer_residue_stereospec(1, 2, "LEU", &leu_bb, &leu_obs_map, &cfg);
    assert_eq!(leu_results.len(), 2);
    assert!(leu_results[0].committed); // HB2/HB3 committed
    assert_eq!(leu_results[0].nmr_star_ambiguity_code, 1);
    assert!(!leu_results[1].committed); // CD1/CD2 uncommitted -> Ambiguity_code = 2, CDx/CDy!
    assert_eq!(leu_results[1].nmr_star_ambiguity_code, 2);
    assert_eq!(
        leu_results[1].nef_carbon_labels_for_ab(),
        Some(("CDx", "CDy"))
    );
    assert_eq!(leu_results[1].nef_proton_labels_for_ab(), ("HDx%", "HDy%"));

    let mut all_res = vec![val_res];
    all_res.extend(leu_results);

    // Verify Reference + Ambiguity_code map + NMR-STAR v3 text formatting & round-trip
    let (merged_ref, amb_map) = apply_stereospec_to_reference(&base, &all_res);
    assert!((merged_ref.shifts[&(0, "CG1".to_string())] - 21.8).abs() < 1e-9);
    assert!((merged_ref.shifts[&(0, "CG2".to_string())] - 19.6).abs() < 1e-9);
    assert_eq!(amb_map[&(0, "CG1".to_string())], 1);
    assert_eq!(amb_map[&(1, "HB2".to_string())], 1);
    assert_eq!(amb_map[&(1, "CD1".to_string())], 2);
    assert_eq!(amb_map[&(1, "CD2".to_string())], 2);

    let rows = to_star_chem_shift_rows(&base, &all_res);
    assert!(rows
        .iter()
        .any(|r| r.seq_id == 1 && r.atom_id == "CG1" && r.ambiguity_code == 1));
    assert!(rows
        .iter()
        .any(|r| r.seq_id == 2 && r.atom_id == "HB2" && r.ambiguity_code == 1));
    assert!(rows
        .iter()
        .any(|r| r.seq_id == 2 && r.atom_id == "CD1" && r.ambiguity_code == 2));

    let star_text = format_stereospec_nmrstar(&base, &all_res, Some("Test stereospec NMR-STAR"));
    let parsed_star = read_star_str_all_atoms(&star_text, None).expect("roundtrip NMR-STAR");
    assert_eq!(parsed_star.sequence, base.sequence);
    assert!((parsed_star.shifts[&(0, "CG1".to_string())] - 21.8).abs() < 1e-3);
    assert!((parsed_star.shifts[&(1, "HB2".to_string())] - 1.72).abs() < 1e-3);

    // Verify NEF v1.1 shift list formatting & round-trip
    let nef_list = to_nef_shift_list_with_stereospec(&base, &all_res, None, "A");
    assert!(nef_list.shifts.iter().any(|s| s.sequence_code == Some(1)
        && s.atom_name == "CG1"
        && (s.value - 21.8).abs() < 1e-6));
    assert!(nef_list.shifts.iter().any(|s| s.sequence_code == Some(1)
        && s.atom_name == "HG1%"
        && (s.value - 0.89).abs() < 1e-6));
    assert!(nef_list.shifts.iter().any(|s| s.sequence_code == Some(2)
        && s.atom_name == "HB2"
        && (s.value - 1.72).abs() < 1e-6));
    assert!(nef_list.shifts.iter().any(|s| {
        s.sequence_code == Some(2)
            && s.atom_name == "CDx"
            && classify_nef_atom_name(&s.atom_name) == NefAtomWildcardKind::NonStereoProchiral
    }));
    assert!(nef_list.shifts.iter().any(|s| {
        s.sequence_code == Some(2)
            && s.atom_name == "HDy%"
            && classify_nef_atom_name(&s.atom_name) == NefAtomWildcardKind::NonStereoProchiral
    }));

    let mut nef_doc = NefDocument::new("stereospec_test");
    nef_doc.molecular_system = NefMolecularSystem::from_reference(&base, "A");
    nef_doc.shift_lists.push(nef_list);
    let nef_str = write_nef_str(&nef_doc);
    let parsed_nef = read_nef_str(&nef_str).expect("parse NEF roundtrip");
    assert_eq!(parsed_nef.shift_lists.len(), 1);
    assert!(parsed_nef.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.atom_name == "CDx"));
    assert!(parsed_nef.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.atom_name == "CG1"));
}

#[test]
fn test_p1_1_gly_phi_sign_mirror_degeneracy_cap() {
    let cfg = StereospecConfig::default();
    let gly_spec = prochiral_pairs_for_residue("GLY")[0];

    // Exact probe: true phi = +100 deg gives 3J(HN, HA2/HA3) = (4.1, 9.0) Hz,
    // evaluated under CSI-predicted phi = -100 deg (DihedralEstimate, no 3D PDB).
    // Under phi -> -phi, Gly 3J(HN, HA) and intraresidue HN-HA NOEs are mirror-degenerate
    // with Direct <-> Swapped, so this MUST stay uncommitted (< 0.90, Ambiguity_code = 2).
    let csi_bb = BackboneConditioning::DihedralEstimate {
        phi_deg: -100.0,
        psi_deg: 0.0,
        phi_err_deg: 20.0,
        psi_err_deg: 20.0,
        ss_class: SecondaryStructure::Coil,
    };
    let obs_j_only = ProchiralObservables {
        proton_shifts_ppm: Some((3.82, 4.18)),
        j_hh_vicinal_hz: Some((4.1, 9.0)),
        ..Default::default()
    };
    let res_j = infer_prochiral_pair(0, 1, gly_spec, &csi_bb, &obs_j_only, &cfg);
    assert!(
        !res_j.committed,
        "Gly without PDB or sequential NOE must remain uncommitted, got {:?}",
        res_j
    );
    assert_eq!(res_j.nmr_star_ambiguity_code, 2);
    assert!(
        res_j.winning_posterior <= 0.85 + 1e-9,
        "expected winning_posterior <= 0.85, got {:.4}",
        res_j.winning_posterior
    );

    // Even adding intraresidue HN-HA NOEs under CSI-predicted phi = -100 deg is still
    // mirror-degenerate under phi -> -phi without a 3D PDB or sequential HA(i)-HN(i+1) NOE.
    let mut obs_with_intra = obs_j_only.clone();
    obs_with_intra
        .noe_pairs
        .push(NoePairObs::new(AnchorProton::IntraAmideH, 0.008, 0.028));
    let res_intra_csi = infer_prochiral_pair(0, 1, gly_spec, &csi_bb, &obs_with_intra, &cfg);
    assert!(!res_intra_csi.committed);
    assert_eq!(res_intra_csi.nmr_star_ambiguity_code, 2);
    assert!(res_intra_csi.winning_posterior < 0.90);

    // When phi = +100 deg comes from an explicit 3D PDB structure (ExactTorsions) AND
    // multi-pathway evidence (3J + intraresidue NOE) is present, Gly resolves cleanly.
    let pdb_bb = BackboneConditioning::ExactTorsions {
        phi_deg: 100.0,
        psi_deg: 0.0,
    };
    let res_pdb = infer_prochiral_pair(0, 1, gly_spec, &pdb_bb, &obs_j_only, &cfg);
    // Single J pathway alone is still capped by P1-2 (< 0.90):
    assert!(!res_pdb.committed);
    assert!(res_pdb.winning_posterior < 0.90);

    let frame = build_local_peptide_frame("GLY", 100.0, 0.0, false);
    let h_in = frame.h.unwrap();
    let (gly_ha2, gly_ha3) = frame.gly_ha2_ha3.unwrap();
    let r6_2 = v_dist(h_in, gly_ha2).powi(-6);
    let r6_3 = v_dist(h_in, gly_ha3).powi(-6);
    let obs_pdb_multi = ProchiralObservables {
        proton_shifts_ppm: Some((3.82, 4.18)),
        j_hh_vicinal_hz: Some((4.1, 9.0)),
        noe_pairs: vec![NoePairObs::new(AnchorProton::IntraAmideH, r6_2, r6_3)],
        ..Default::default()
    };
    let res_pdb_multi = infer_prochiral_pair(0, 1, gly_spec, &pdb_bb, &obs_pdb_multi, &cfg);
    assert!(res_pdb_multi.committed);
    assert_eq!(res_pdb_multi.winning_hypothesis, SwapHypothesis::Direct);
    assert_eq!(res_pdb_multi.nmr_star_ambiguity_code, 1);

    // P1-1R exact probe: Gly's local unit is achiral under the full mirror (phi, psi) -> (-phi, -psi)
    // with HA2 <-> HA3, so BOTH intraresidue HN(i)-HA and sequential HA(i)-HN(i+1) NOE ratios are
    // identical for the truth (+80, +170) and the mirror (-80, -170). Without an explicit 3D PDB
    // (`ExactTorsions`), Gly MUST stay uncommitted (`Ambiguity_code = 2`, `posterior <= 0.85 < 0.90`).
    let frame_mirror = build_local_peptide_frame("GLY", 80.0, 170.0, false);
    let h_curr = frame_mirror.h.unwrap();
    let (m_ha2, m_ha3) = frame_mirror.gly_ha2_ha3.unwrap();
    let h_next = frame_mirror.h_next.unwrap();
    let (_, _, th_h_2, th_h_3) = nmr_core::stereospec::iupac_gly_ha_dihedrals_deg(80.0);
    let j_ha2 = nmr_core::stereospec::karplus_3j_hnha_hz(th_h_2);
    let j_ha3 = nmr_core::stereospec::karplus_3j_hnha_hz(th_h_3);
    let obs_full_local = ProchiralObservables {
        proton_shifts_ppm: Some((3.82, 4.18)),
        j_hh_vicinal_hz: Some((j_ha2, j_ha3)),
        noe_pairs: vec![
            NoePairObs::new(
                AnchorProton::IntraAmideH,
                v_dist(h_curr, m_ha2).powi(-6),
                v_dist(h_curr, m_ha3).powi(-6),
            ),
            NoePairObs::new(
                AnchorProton::SeqNextAmideH,
                v_dist(h_next, m_ha2).powi(-6),
                v_dist(h_next, m_ha3).powi(-6),
            ),
        ],
        ..Default::default()
    };
    let mirrored_csi = BackboneConditioning::DihedralEstimate {
        phi_deg: -80.0,
        psi_deg: -170.0,
        phi_err_deg: 25.0,
        psi_err_deg: 25.0,
        ss_class: SecondaryStructure::Coil,
    };
    let res_mirrored_csi =
        infer_prochiral_pair(0, 1, gly_spec, &mirrored_csi, &obs_full_local, &cfg);
    assert!(
        !res_mirrored_csi.committed,
        "Gly under mirrored CSI (-80, -170) without PDB must NOT commit (got P={:.4})",
        res_mirrored_csi.winning_posterior
    );
    assert_eq!(res_mirrored_csi.nmr_star_ambiguity_code, 2);
    assert!(res_mirrored_csi.winning_posterior <= 0.85 + 1e-9);

    let phi_flipped_only = BackboneConditioning::DihedralEstimate {
        phi_deg: -80.0,
        psi_deg: 170.0,
        phi_err_deg: 25.0,
        psi_err_deg: 25.0,
        ss_class: SecondaryStructure::Coil,
    };
    let res_phi_flipped =
        infer_prochiral_pair(0, 1, gly_spec, &phi_flipped_only, &obs_full_local, &cfg);
    assert!(!res_phi_flipped.committed);
    assert_eq!(res_phi_flipped.nmr_star_ambiguity_code, 2);
    assert!(res_phi_flipped.winning_posterior <= 0.85 + 1e-9);
}

#[test]
fn test_p1_2_ile_hg12_hg13_and_general_single_pathway_degeneracy_cap() {
    let cfg = StereospecConfig::default();

    // Exact probe: Ile in helix with single 3J(HB, HG12/HG13) = (11.0, 3.0) Hz MUST stay < 0.90.
    let ile_spec = prochiral_pairs_for_residue("ILE")[0];
    let helix_ss = BackboneConditioning::SecondaryStructurePrior(SecondaryStructure::Helix);
    let helix_pdb = BackboneConditioning::ExactTorsions {
        phi_deg: -63.0,
        psi_deg: -42.0,
    };
    let ile_obs = ProchiralObservables {
        proton_shifts_ppm: Some((1.15, 1.48)),
        j_hh_vicinal_hz: Some((11.0, 3.0)),
        ..Default::default()
    };
    for bb in [&helix_ss, &helix_pdb] {
        let res = infer_prochiral_pair(0, 1, ile_spec, bb, &ile_obs, &cfg);
        assert!(
            !res.committed,
            "Ile HG12/HG13 with single 3J(HB, HG) must not commit: {:?}",
            res
        );
        assert_eq!(res.nmr_star_ambiguity_code, 2);
        assert!(
            res.winning_posterior <= 0.85 + 1e-9,
            "Ile single-pathway posterior must be <= 0.85, got {:.4}",
            res.winning_posterior
        );
    }

    // Even if single methine 3J(HA, HB) is also given on Ile without a second paired HG12/HG13 pathway
    // or discriminating NOE, HG12/HG13 remains single-pathway degenerate on chi2!
    let mut ile_obs_with_methine = ile_obs.clone();
    ile_obs_with_methine.j_methine_hahb_hz = Some(10.5);
    let res_ile_methine =
        infer_prochiral_pair(0, 1, ile_spec, &helix_pdb, &ile_obs_with_methine, &cfg);
    assert!(!res_ile_methine.committed);
    assert_eq!(res_ile_methine.nmr_star_ambiguity_code, 2);
    assert!(res_ile_methine.winning_posterior < 0.90);

    // General single-pathway cap across Asp HB2/HB3, Val CG1/CG2, Leu CD1/CD2, and Lys CG distal methylene:
    let asp_spec = prochiral_pairs_for_residue("ASP")[0];
    let res_asp = infer_prochiral_pair(
        0,
        2,
        asp_spec,
        &helix_pdb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((2.65, 2.91)),
            j_hh_vicinal_hz: Some((11.2, 3.4)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_asp.committed);
    assert_eq!(res_asp.nmr_star_ambiguity_code, 2);
    assert!(res_asp.winning_posterior < 0.90);

    let val_spec = prochiral_pairs_for_residue("VAL")[0];
    let res_val = infer_prochiral_pair(
        0,
        3,
        val_spec,
        &helix_pdb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((0.82, 0.95)),
            carbon_shifts_ppm: Some((19.5, 21.6)),
            j_ncg_hz: Some((2.1, 0.4)),
            j_methine_hahb_hz: Some(10.8),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_val.committed);
    assert_eq!(res_val.nmr_star_ambiguity_code, 2);
    assert!(res_val.winning_posterior < 0.90);

    let leu_cd_spec = prochiral_pairs_for_residue("LEU")[1];
    let res_leu = infer_prochiral_pair(
        0,
        4,
        leu_cd_spec,
        &helix_pdb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((0.81, 0.92)),
            carbon_shifts_ppm: Some((22.1, 24.8)),
            j_cc_vicinal_hz: Some((3.4, 1.1)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_leu.committed);
    assert_eq!(res_leu.nmr_star_ambiguity_code, 2);
    assert!(res_leu.winning_posterior < 0.90);

    let lys_cg_spec = prochiral_pairs_for_residue("LYS")[1];
    let res_lys = infer_prochiral_pair(
        0,
        5,
        lys_cg_spec,
        &helix_pdb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((1.38, 1.51)),
            j_hh_vicinal_hz: Some((10.8, 3.6)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_lys.committed);
    assert_eq!(res_lys.nmr_star_ambiguity_code, 2);
    assert!(res_lys.winning_posterior < 0.90);
}

#[test]
fn test_p1_3_outlier_model_error_and_nan_inf_guards() {
    let cfg = StereospecConfig::default();
    let asp_spec = prochiral_pairs_for_residue("ASP")[0];
    let bb = BackboneConditioning::ExactTorsions {
        phi_deg: -63.0,
        psi_deg: -42.0,
    };

    // (1) Unphysical Asp J = (40.0, 39.0) Hz
    let res_40 = infer_prochiral_pair(
        0,
        1,
        asp_spec,
        &bb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((2.60, 2.90)),
            j_hh_vicinal_hz: Some((40.0, 39.0)),
            j_nhb_hz: Some((3.1, 0.5)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_40.committed);
    assert_eq!(res_40.nmr_star_ambiguity_code, 2);
    assert!(!res_40.winning_posterior.is_nan());
    assert!(res_40.winning_posterior <= 0.50);

    // (2) Unphysical Asp J = (2.0, 25.0) Hz
    let res_25 = infer_prochiral_pair(
        0,
        1,
        asp_spec,
        &bb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((2.60, 2.90)),
            j_hh_vicinal_hz: Some((2.0, 25.0)),
            j_nhb_hz: Some((3.1, 0.5)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_25.committed);
    assert_eq!(res_25.nmr_star_ambiguity_code, 2);
    assert!(!res_25.winning_posterior.is_nan());
    assert!(res_25.winning_posterior <= 0.50);

    // (3) In-bounds [0..16 Hz] couplings that violate Karplus geometry severely (reduced chi^2 > 9.0):
    // e.g. both 3J(HA, HB2) and 3J(HA, HB3) = 15.5 Hz simultaneously AND 3J(N, HB) = (15.0, 14.5) Hz!
    let res_misfit = infer_prochiral_pair(
        0,
        1,
        asp_spec,
        &bb,
        &ProchiralObservables {
            proton_shifts_ppm: Some((2.60, 2.90)),
            j_hh_vicinal_hz: Some((15.5, 15.2)),
            j_nhb_hz: Some((15.0, 14.5)),
            ..Default::default()
        },
        &cfg,
    );
    assert!(!res_misfit.committed);
    assert_eq!(res_misfit.nmr_star_ambiguity_code, 2);
    assert!(res_misfit.winning_posterior.is_finite());
    assert!(res_misfit.winning_posterior <= 0.50);

    // (4) NaN and +/-INFINITY in J and NOE inputs, and non-positive NOE intensities
    let bad_inputs = [
        ProchiralObservables {
            j_hh_vicinal_hz: Some((f64::NAN, 4.0)),
            j_nhb_hz: Some((3.0, 0.5)),
            ..Default::default()
        },
        ProchiralObservables {
            j_hh_vicinal_hz: Some((10.5, f64::INFINITY)),
            j_nhb_hz: Some((3.0, 0.5)),
            ..Default::default()
        },
        ProchiralObservables {
            j_hh_vicinal_hz: Some((10.5, 3.2)),
            noe_pairs: vec![NoePairObs::new(AnchorProton::IntraAmideH, f64::NAN, 0.02)],
            ..Default::default()
        },
        ProchiralObservables {
            j_hh_vicinal_hz: Some((10.5, 3.2)),
            noe_pairs: vec![NoePairObs::new(
                AnchorProton::IntraAmideH,
                0.01,
                f64::INFINITY,
            )],
            ..Default::default()
        },
        ProchiralObservables {
            j_hh_vicinal_hz: Some((10.5, 3.2)),
            noe_pairs: vec![NoePairObs::new(AnchorProton::IntraAmideH, 0.0, 0.02)],
            ..Default::default()
        },
        ProchiralObservables {
            j_hh_vicinal_hz: Some((10.5, 3.2)),
            noe_pairs: vec![NoePairObs::new(AnchorProton::IntraAmideH, -0.01, 0.02)],
            ..Default::default()
        },
    ];
    for (idx, bad) in bad_inputs.iter().enumerate() {
        let res = infer_prochiral_pair(0, 1, asp_spec, &bb, bad, &cfg);
        assert!(!res.committed, "bad input #{idx} must not commit");
        assert_eq!(res.nmr_star_ambiguity_code, 2, "bad input #{idx} code != 2");
        assert!(
            res.winning_posterior.is_finite() && res.winning_posterior <= 0.50,
            "bad input #{idx} posterior must be finite <= 0.50, got {}",
            res.winning_posterior
        );
    }
}

#[test]
fn test_p1_4_ambiguity_code_seeding_wildcard_dedup_and_atomic_methyl_swaps() {
    let pdb = parse_pdb_all_atoms_file(repo_file("data/pdb_refs/1d3z.pdb"))
        .expect("parse 1d3z all atoms");
    let cfg = StereospecConfig::default();

    // (a) Half-present HB2 (HB3 missing) and unprocessed prochiral atoms -> Ambiguity_code = 2;
    //     symmetric Phe CD1/CD2/HD1/HD2/CE1/CE2/HE1/HE2 -> 3, Phe CZ/HZ -> 1;
    //     existing Reference.ambiguity_codes preserved when complete & untouched.
    let mut base_a = Reference {
        sequence: vec![
            "ASP".to_string(),
            "PHE".to_string(),
            "SER".to_string(),
            "ASN".to_string(),
        ],
        seq_ids: vec![1, 2, 3, 4],
        shifts: BTreeMap::from([
            // ASP 1: unique HA + half-present HB2 (HB3 missing!)
            ((0, "HA".to_string()), 4.62),
            ((0, "HB2".to_string()), 2.74),
            // PHE 2: symmetric ring CD1/HD1/CE1/HE1 + unique CZ/HZ + unprocessed HB2/HB3
            ((1, "HB2".to_string()), 3.12),
            ((1, "HB3".to_string()), 2.95),
            ((1, "CD1".to_string()), 131.4),
            ((1, "CD2".to_string()), 131.4),
            ((1, "HD1".to_string()), 7.22),
            ((1, "HD2".to_string()), 7.22),
            ((1, "CZ".to_string()), 129.1),
            ((1, "HZ".to_string()), 7.15),
            // SER 3: previously stereo-assigned HB2/HB3 with Ambiguity_code = 1 in input STAR
            ((2, "HB2".to_string()), 3.88),
            ((2, "HB3".to_string()), 3.96),
            // ASN 4: half-present HB2 annotated with bogus code 1 in input -> must still become 2!
            ((3, "HB2".to_string()), 2.81),
        ]),
        ambiguity_codes: BTreeMap::from([
            ((2, "HB2".to_string()), 1),
            ((2, "HB3".to_string()), 1),
            ((3, "HB2".to_string()), 1),
        ]),
    };
    let (merged_a, amb_a) = apply_stereospec_to_reference(&base_a, &[]);
    assert_eq!(amb_a[&(0, "HA".to_string())], 1);
    assert_eq!(
        amb_a[&(0, "HB2".to_string())],
        2,
        "half-present HB2 must get Ambiguity_code = 2"
    );
    assert_eq!(
        amb_a[&(1, "HB2".to_string())],
        2,
        "unprocessed prochiral HB2/HB3 must default to Ambiguity_code = 2"
    );
    assert_eq!(amb_a[&(1, "HB3".to_string())], 2);
    assert_eq!(amb_a[&(1, "CD1".to_string())], 3);
    assert_eq!(amb_a[&(1, "HD1".to_string())], 3);
    assert_eq!(amb_a[&(1, "CZ".to_string())], 1);
    assert_eq!(amb_a[&(1, "HZ".to_string())], 1);
    assert_eq!(
        amb_a[&(2, "HB2".to_string())],
        1,
        "pre-existing input Ambiguity_code = 1 on complete untouched pair must be preserved"
    );
    assert_eq!(amb_a[&(2, "HB3".to_string())], 1);
    assert_eq!(
        amb_a[&(3, "HB2".to_string())],
        2,
        "half-present HB2 must be forced to Ambiguity_code = 2 even if input had 1"
    );
    assert_eq!(merged_a.shifts.len(), base_a.shifts.len());
    base_a.ambiguity_codes = amb_a;

    // (b) NEF wildcard inputs (HBx/HBy, CGx/CGy, HGx%/HGy%, HE2x/HE2y, both mixed-case and
    //     uppercased as read_star_str_all_atoms produces) -> zero duplicate wildcard + canonical rows!
    let base_b = Reference {
        sequence: vec!["ASP".to_string(), "VAL".to_string(), "GLN".to_string()],
        seq_ids: vec![1, 2, 3],
        shifts: BTreeMap::from([
            ((0, "HBX".to_string()), 2.65),
            ((0, "HBY".to_string()), 2.92),
            ((1, "CGx".to_string()), 19.6),
            ((1, "CGy".to_string()), 21.8),
            ((1, "HGx%".to_string()), 0.74),
            ((1, "HGy%".to_string()), 0.89),
            ((2, "HE2X".to_string()), 6.85),
            ((2, "HE2Y".to_string()), 7.52),
        ]),
        ..Default::default()
    };
    let asp_spec = prochiral_pairs_for_residue("ASP")[0];
    let val_spec = prochiral_pairs_for_residue("VAL")[0];
    let gln_spec = prochiral_pairs_for_residue("GLN")[2];
    let b_results = vec![
        infer_prochiral_pair(
            0,
            1,
            asp_spec,
            &BackboneConditioning::Marginalized,
            &ProchiralObservables {
                proton_shifts_ppm: Some((2.65, 2.92)),
                ..Default::default()
            },
            &cfg,
        ),
        infer_prochiral_pair(
            1,
            2,
            val_spec,
            &BackboneConditioning::Marginalized,
            &ProchiralObservables {
                proton_shifts_ppm: Some((0.74, 0.89)),
                carbon_shifts_ppm: Some((19.6, 21.8)),
                ..Default::default()
            },
            &cfg,
        ),
        infer_prochiral_pair(
            2,
            3,
            gln_spec,
            &BackboneConditioning::Marginalized,
            &ProchiralObservables {
                proton_shifts_ppm: Some((6.85, 7.52)),
                ..Default::default()
            },
            &cfg,
        ),
    ];
    let (merged_b, _amb_b) = apply_stereospec_to_reference(&base_b, &b_results);
    for (k, atom) in merged_b.shifts.keys() {
        let up = atom.to_ascii_uppercase();
        assert!(
            !up.contains('X') && !up.contains('Y'),
            "wildcard alias ({k}, {atom}) leaked alongside canonical row: {:?}",
            merged_b.shifts
        );
    }
    assert_eq!(merged_b.shifts.len(), 12);
    assert!((merged_b.shifts[&(0, "HB2".to_string())] - 2.65).abs() < 1e-9);
    assert!((merged_b.shifts[&(0, "HB3".to_string())] - 2.92).abs() < 1e-9);
    // Uncommitted Val CG1/CG2 + HG11..13/HG21..23 MUST preserve input (a, b) correspondence, NOT sort C and H independently!
    assert!((merged_b.shifts[&(1, "CG1".to_string())] - 19.6).abs() < 1e-9);
    assert!((merged_b.shifts[&(1, "CG2".to_string())] - 21.8).abs() < 1e-9);
    assert!((merged_b.shifts[&(1, "HG11".to_string())] - 0.74).abs() < 1e-9);
    assert!((merged_b.shifts[&(1, "HG21".to_string())] - 0.89).abs() < 1e-9);
    assert!((merged_b.shifts[&(2, "HE21".to_string())] - 6.85).abs() < 1e-9);
    assert!((merged_b.shifts[&(2, "HE22".to_string())] - 7.52).abs() < 1e-9);

    // (c1) Val with CG1/CG2 + QG1/QG2 + HG11/12/13 & HG21/22/23 committed Swapped -> ALL members swap atomically!
    let val_ideal = synthesize_ideal_observables_from_pdb(&pdb, 4, val_spec).unwrap();
    let val_swapped_obs = ProchiralObservables {
        proton_shifts_ppm: Some((0.74, 0.89)),
        carbon_shifts_ppm: Some((19.6, 21.8)),
        j_ncg_hz: val_ideal.j_ncg_hz.map(|(x, y)| (y, x)),
        j_cc_vicinal_hz: val_ideal.j_cc_vicinal_hz.map(|(x, y)| (y, x)),
        j_methine_hahb_hz: val_ideal.j_methine_hahb_hz,
        noe_pairs: val_ideal
            .noe_pairs
            .iter()
            .map(|np| NoePairObs::new(np.anchor, np.intensity_b, np.intensity_a))
            .collect(),
        ..Default::default()
    };
    let val_bb = backbone_conditioning_from_all_atom_pdb(&pdb, 4);
    let val_res = infer_prochiral_pair(0, 1, val_spec, &val_bb, &val_swapped_obs, &cfg);
    assert!(val_res.committed);
    assert_eq!(val_res.winning_hypothesis, SwapHypothesis::Swapped);

    let base_val_qg = Reference {
        sequence: vec!["VAL".to_string()],
        seq_ids: vec![1],
        shifts: BTreeMap::from([
            ((0, "CG1".to_string()), 19.6),
            ((0, "CG2".to_string()), 21.8),
            ((0, "QG1".to_string()), 0.74),
            ((0, "QG2".to_string()), 0.89),
            ((0, "HG11".to_string()), 0.74),
            ((0, "HG12".to_string()), 0.74),
            ((0, "HG13".to_string()), 0.74),
            ((0, "HG21".to_string()), 0.89),
            ((0, "HG22".to_string()), 0.89),
            ((0, "HG23".to_string()), 0.89),
        ]),
        ..Default::default()
    };
    let (merged_val_qg, amb_val_qg) =
        apply_stereospec_to_reference(&base_val_qg, std::slice::from_ref(&val_res));
    assert!((merged_val_qg.shifts[&(0, "CG1".to_string())] - 21.8).abs() < 1e-9);
    assert!((merged_val_qg.shifts[&(0, "CG2".to_string())] - 19.6).abs() < 1e-9);
    assert!((merged_val_qg.shifts[&(0, "QG1".to_string())] - 0.89).abs() < 1e-9);
    assert!((merged_val_qg.shifts[&(0, "QG2".to_string())] - 0.74).abs() < 1e-9);
    assert!((merged_val_qg.shifts[&(0, "HG11".to_string())] - 0.89).abs() < 1e-9);
    assert!((merged_val_qg.shifts[&(0, "HG21".to_string())] - 0.74).abs() < 1e-9);
    for atom in ["CG1", "CG2", "QG1", "QG2", "HG11", "HG21"] {
        assert_eq!(amb_val_qg[&(0, atom.to_string())], 1);
    }

    // (c2) Val with CG1/CG2 + lone QG1 (missing QG2) -> refuse to commit! Keep ALL members unswapped at code 2!
    let base_val_half = Reference {
        sequence: vec!["VAL".to_string()],
        seq_ids: vec![1],
        shifts: BTreeMap::from([
            ((0, "CG1".to_string()), 19.6),
            ((0, "CG2".to_string()), 21.8),
            ((0, "QG1".to_string()), 0.74),
        ]),
        ..Default::default()
    };
    let mut val_res_no_obs_h = val_res.clone();
    val_res_no_obs_h.proton_shifts_ppm = None;
    let (merged_half, amb_half) =
        apply_stereospec_to_reference(&base_val_half, std::slice::from_ref(&val_res_no_obs_h));
    assert!(
        (merged_half.shifts[&(0, "CG1".to_string())] - 19.6).abs() < 1e-9,
        "CG1 must NOT swap when QG2 is missing"
    );
    assert!((merged_half.shifts[&(0, "CG2".to_string())] - 21.8).abs() < 1e-9);
    assert!((merged_half.shifts[&(0, "QG1".to_string())] - 0.74).abs() < 1e-9);
    assert_eq!(amb_half[&(0, "CG1".to_string())], 2);
    assert_eq!(amb_half[&(0, "CG2".to_string())], 2);
    assert_eq!(amb_half[&(0, "QG1".to_string())], 2);

    // (c3) Half-present VAL methyl wildcard (`CG1/CG2` + lone `HGx%` without `HGy%`) -> must NOT
    //      swap `CG1/CG2` AND must preserve `HGx%` normalized to `HG11..13` at `Ambiguity_code = 2`
    //      (both when `results = &[]` and when a committed Swapped `StereospecResult` is supplied).
    let base_val_half_wild = Reference {
        sequence: vec!["VAL".to_string()],
        seq_ids: vec![1],
        shifts: BTreeMap::from([
            ((0, "CG1".to_string()), 19.6),
            ((0, "CG2".to_string()), 21.8),
            ((0, "HGx%".to_string()), 0.74),
        ]),
        ..Default::default()
    };
    for half_res_slice in [&[][..], std::slice::from_ref(&val_res_no_obs_h)] {
        let (m_hw, a_hw) = apply_stereospec_to_reference(&base_val_half_wild, half_res_slice);
        assert!((m_hw.shifts[&(0, "CG1".to_string())] - 19.6).abs() < 1e-9);
        assert!((m_hw.shifts[&(0, "CG2".to_string())] - 21.8).abs() < 1e-9);
        assert_eq!(a_hw[&(0, "CG1".to_string())], 2);
        assert_eq!(a_hw[&(0, "CG2".to_string())], 2);
        for h1 in ["HG11", "HG12", "HG13"] {
            assert!(
                (m_hw.shifts[&(0, h1.to_string())] - 0.74).abs() < 1e-9,
                "half-present HGx% must be preserved as {h1}, got {:?}",
                m_hw.shifts
            );
            assert_eq!(a_hw[&(0, h1.to_string())], 2);
        }
        assert!(!m_hw.shifts.contains_key(&(0, "HGx%".to_string())));
        assert!(!m_hw.shifts.contains_key(&(0, "HG21".to_string())));
    }

    // (b2) Residue-aware NEF wildcard normalization on VAL (`HGx/HGy`), LEU (`HDx/HDy` & `HDx%/HDy%`),
    //      ILE (`HGx/HGy`), and PHE/TYR (`CDx/CDy`, `CEx/CEy`, `HDx/HDy`, `HEx/HEy`):
    //      - NEVER fabricates invalid methylene names (`HG2/HG3` on VAL/ILE, `HD2/HD3` on LEU/PHE, `HE3` on PHE/TYR),
    //      - assigns `Ambiguity_code = 3` (NOT 2!) to PHE/TYR symmetric ring wildcards,
    //      - normalizes VAL/LEU `%` and shorthand `HGx/HGy` / `HDx/HDy` even when `results = &[]`.
    let base_res_aware = Reference {
        sequence: vec![
            "VAL".to_string(),
            "LEU".to_string(),
            "ILE".to_string(),
            "PHE".to_string(),
            "LEU".to_string(),
        ],
        seq_ids: vec![1, 2, 3, 4, 5],
        shifts: BTreeMap::from([
            // res 0: VAL with shorthand HGx/HGy + CGx/CGy
            ((0, "CGx".to_string()), 19.6),
            ((0, "CGy".to_string()), 21.8),
            ((0, "HGx".to_string()), 0.74),
            ((0, "HGy".to_string()), 0.89),
            // res 1: LEU with shorthand HDx/HDy + CDx/CDy
            ((1, "CDx".to_string()), 22.4),
            ((1, "CDy".to_string()), 24.9),
            ((1, "HDx".to_string()), 0.81),
            ((1, "HDy".to_string()), 0.94),
            // res 2: ILE with shorthand HGx/HGy
            ((2, "HGx".to_string()), 1.18),
            ((2, "HGy".to_string()), 1.46),
            // res 3: PHE with symmetric aromatic wildcards CDx/CDy, CEx/CEy, HDx/HDy, HEx/HEy + HBx/HBy
            ((3, "HBx".to_string()), 2.95),
            ((3, "HBy".to_string()), 3.12),
            ((3, "CDx".to_string()), 131.4),
            ((3, "CDy".to_string()), 131.4),
            ((3, "CEx".to_string()), 130.2),
            ((3, "CEy".to_string()), 130.2),
            ((3, "HDx".to_string()), 7.22),
            ((3, "HDy".to_string()), 7.22),
            ((3, "HEx".to_string()), 7.31),
            ((3, "HEy".to_string()), 7.31),
            // res 4: LEU with HDx%/HDy% (unprocessed)
            ((4, "HDx%".to_string()), 0.77),
            ((4, "HDy%".to_string()), 0.91),
        ]),
        ..Default::default()
    };

    // First test with `results = &[]`
    let (m_empty, a_empty) = apply_stereospec_to_reference(&base_res_aware, &[]);
    for (k, atom) in m_empty.shifts.keys() {
        let up = atom.to_ascii_uppercase();
        assert!(
            !up.contains('X') && !up.contains('Y'),
            "un-normalized wildcard ({k}, {atom}) remained with results=&[]: {:?}",
            m_empty.shifts
        );
    }
    // VAL 0: HG11..13 & HG21..23, NO HG2/HG3!
    assert!(!m_empty.shifts.contains_key(&(0, "HG2".to_string())));
    assert!(!m_empty.shifts.contains_key(&(0, "HG3".to_string())));
    assert!((m_empty.shifts[&(0, "HG11".to_string())] - 0.74).abs() < 1e-9);
    assert!((m_empty.shifts[&(0, "HG21".to_string())] - 0.89).abs() < 1e-9);
    assert_eq!(a_empty[&(0, "HG11".to_string())], 2);
    // LEU 1 & LEU 4: HD11..13 & HD21..23, NO HD2/HD3!
    for lk in [1usize, 4usize] {
        assert!(!m_empty.shifts.contains_key(&(lk, "HD2".to_string())));
        assert!(!m_empty.shifts.contains_key(&(lk, "HD3".to_string())));
        assert!(m_empty.shifts.contains_key(&(lk, "HD11".to_string())));
        assert!(m_empty.shifts.contains_key(&(lk, "HD21".to_string())));
        assert_eq!(a_empty[&(lk, "HD11".to_string())], 2);
    }
    // ILE 2: HG12 & HG13, NO HG2/HG3!
    assert!(!m_empty.shifts.contains_key(&(2, "HG2".to_string())));
    assert!(!m_empty.shifts.contains_key(&(2, "HG3".to_string())));
    assert!((m_empty.shifts[&(2, "HG12".to_string())] - 1.18).abs() < 1e-9);
    assert!((m_empty.shifts[&(2, "HG13".to_string())] - 1.46).abs() < 1e-9);
    assert_eq!(a_empty[&(2, "HG12".to_string())], 2);
    // PHE 3: CD1/CD2/CE1/CE2/HD1/HD2/HE1/HE2 at Ambiguity_code = 3, HB2/HB3 at 2, NO HD3/HE3!
    assert!(
        !m_empty.shifts.contains_key(&(3, "HD2".to_string()))
            || a_empty[&(3, "HD2".to_string())] == 3
    );
    assert!(!m_empty.shifts.contains_key(&(3, "HD3".to_string())));
    assert!(!m_empty.shifts.contains_key(&(3, "HE3".to_string())));
    for sym in ["CD1", "CD2", "CE1", "CE2", "HD1", "HD2", "HE1", "HE2"] {
        assert_eq!(
            a_empty[&(3, sym.to_string())],
            3,
            "PHE {sym} must have Ambiguity_code = 3"
        );
    }
    assert_eq!(a_empty[&(3, "HB2".to_string())], 2);
    assert_eq!(a_empty[&(3, "HB3".to_string())], 2);

    // Also test with committed Swapped VAL 0 (`val_res`) + uncommitted ILE 2: still zero bogus HG2/HG3!
    let ile_spec = prochiral_pairs_for_residue("ILE")[0];
    let ile_uncomm = infer_prochiral_pair(
        2,
        3,
        ile_spec,
        &BackboneConditioning::Marginalized,
        &ProchiralObservables::default(),
        &cfg,
    );
    let (m_with_res, a_with_res) =
        apply_stereospec_to_reference(&base_res_aware, &[val_res, ile_uncomm]);
    assert!(!m_with_res.shifts.contains_key(&(0, "HG2".to_string())));
    assert!(!m_with_res.shifts.contains_key(&(0, "HG3".to_string())));
    assert!((m_with_res.shifts[&(0, "CG1".to_string())] - 21.8).abs() < 1e-9);
    assert!((m_with_res.shifts[&(0, "HG11".to_string())] - 0.89).abs() < 1e-9);
    assert!((m_with_res.shifts[&(0, "HG21".to_string())] - 0.74).abs() < 1e-9);
    assert_eq!(a_with_res[&(0, "HG11".to_string())], 1);
    assert!(!m_with_res.shifts.contains_key(&(2, "HG2".to_string())));
    assert!((m_with_res.shifts[&(2, "HG12".to_string())] - 1.18).abs() < 1e-9);
    assert!((m_with_res.shifts[&(2, "HG13".to_string())] - 1.46).abs() < 1e-9);
    assert_eq!(a_with_res[&(2, "HG12".to_string())], 2);

    // 7. `MG1`/`MG2` pseudo-atoms swap atomically with `CG1`/`CG2`, while an unknown methyl proton
    //    alias (`HG9`) keeps the entire VAL group uncommitted at `Ambiguity_code = 2`.
    let mut mg_base = Reference {
        sequence: vec!["VAL".to_string()],
        seq_ids: vec![1],
        ..Default::default()
    };
    mg_base.shifts.insert((0, "CG1".to_string()), 19.2);
    mg_base.shifts.insert((0, "CG2".to_string()), 21.8);
    mg_base.shifts.insert((0, "MG1".to_string()), 0.74);
    mg_base.shifts.insert((0, "MG2".to_string()), 0.89);
    let val_swap = infer_prochiral_pair(0, 1, val_spec, &val_bb, &val_swapped_obs, &cfg);
    assert!(val_swap.committed);
    let (mg_swapped, mg_amb) =
        apply_stereospec_to_reference(&mg_base, std::slice::from_ref(&val_swap));
    assert!((mg_swapped.shifts[&(0, "CG1".to_string())] - 21.8).abs() < 1e-9);
    assert!((mg_swapped.shifts[&(0, "MG1".to_string())] - 0.89).abs() < 1e-9);
    assert!((mg_swapped.shifts[&(0, "MG2".to_string())] - 0.74).abs() < 1e-9);
    assert_eq!(mg_amb[&(0, "MG1".to_string())], 1);

    let mut unk_base = mg_base;
    unk_base.shifts.insert((0, "HG9".to_string()), 0.81);
    let (unk_out, unk_amb) = apply_stereospec_to_reference(&unk_base, &[val_swap]);
    assert!((unk_out.shifts[&(0, "CG1".to_string())] - 19.2).abs() < 1e-9);
    assert!((unk_out.shifts[&(0, "MG1".to_string())] - 0.74).abs() < 1e-9);
    assert_eq!(unk_amb[&(0, "CG1".to_string())], 2);
    assert_eq!(unk_amb[&(0, "MG1".to_string())], 2);
    assert_eq!(unk_amb[&(0, "HG9".to_string())], 2);
}

fn swap_shift_atoms_in_ref(ref_data: &mut Reference, res_idx: usize, a1: &str, a2: &str) {
    let v1 = ref_data.shifts.get(&(res_idx, a1.to_string())).copied();
    let v2 = ref_data.shifts.get(&(res_idx, a2.to_string())).copied();
    if let (Some(x1), Some(x2)) = (v1, v2) {
        ref_data.shifts.insert((res_idx, a1.to_string()), x2);
        ref_data.shifts.insert((res_idx, a2.to_string()), x1);
    }
}

fn scramble_reference_prochiral_site(ref_data: &mut Reference, res_idx: usize, site: &str) {
    match site {
        "HB2" => swap_shift_atoms_in_ref(ref_data, res_idx, "HB2", "HB3"),
        "CG1" => {
            swap_shift_atoms_in_ref(ref_data, res_idx, "CG1", "CG2");
            for (p1, p2) in [
                ("HG1", "HG2"),
                ("HG11", "HG21"),
                ("HG12", "HG22"),
                ("HG13", "HG23"),
                ("MG1", "MG2"),
            ] {
                swap_shift_atoms_in_ref(ref_data, res_idx, p1, p2);
            }
        }
        _ => {}
    }
}

#[test]
fn test_p0_2_experimental_ubiquitin_3j_couplings() {
    let star_path = repo_file("data/fixtures/ubiquitin_bmr5387_methyl.str");
    let exp_j_path = repo_file("data/fixtures/ubiquitin_exp_3j.tsv");
    let pdb_path = repo_file("data/pdb_refs/1d3z.pdb");

    let base_ref =
        read_star_all_atoms(&star_path, None).expect("read ubiquitin_bmr5387_methyl.str");
    let base_j_rows = read_j_couplings_tsv(&exp_j_path).expect("read ubiquitin_exp_3j.tsv");
    assert!(
        base_j_rows.len() >= 90,
        "expected >= 90 experimental Ubiquitin 3J rows, got {}",
        base_j_rows.len()
    );

    // Exact literature-verified stereospecific ordering of BMRB 5387 (`ubiquitin_bmr5387_methyl.str`)
    // against BMRB 17769 / PDB 1D3Z IUPAC stereospecific assignments (Cornilescu et al., JACS 1998):
    // 13 sites in BMRB 5387 are swapped relative to 1D3Z IUPAC (`Swapped`), 26 are `Direct`.
    let bmrb17769_truth: BTreeMap<(i32, &'static str), SwapHypothesis> = BTreeMap::from([
        ((4, "HB2"), SwapHypothesis::Direct),
        ((5, "CG1"), SwapHypothesis::Swapped),
        ((6, "HB2"), SwapHypothesis::Direct),
        ((8, "HB2"), SwapHypothesis::Direct),
        ((11, "HB2"), SwapHypothesis::Direct),
        ((15, "HB2"), SwapHypothesis::Direct),
        ((16, "HB2"), SwapHypothesis::Swapped),
        ((17, "CG1"), SwapHypothesis::Direct),
        ((18, "HB2"), SwapHypothesis::Swapped),
        ((19, "HB2"), SwapHypothesis::Direct),
        ((20, "HB2"), SwapHypothesis::Direct),
        ((25, "HB2"), SwapHypothesis::Direct),
        ((26, "CG1"), SwapHypothesis::Swapped),
        ((27, "HB2"), SwapHypothesis::Direct),
        ((29, "HB2"), SwapHypothesis::Direct),
        ((31, "HB2"), SwapHypothesis::Swapped),
        ((32, "HB2"), SwapHypothesis::Direct),
        ((33, "HB2"), SwapHypothesis::Swapped),
        ((34, "HB2"), SwapHypothesis::Direct),
        ((37, "HB2"), SwapHypothesis::Direct),
        ((38, "HB2"), SwapHypothesis::Swapped),
        ((40, "HB2"), SwapHypothesis::Swapped),
        ((42, "HB2"), SwapHypothesis::Direct),
        ((43, "HB2"), SwapHypothesis::Direct),
        ((45, "HB2"), SwapHypothesis::Direct),
        ((50, "HB2"), SwapHypothesis::Direct),
        ((51, "HB2"), SwapHypothesis::Swapped),
        ((54, "HB2"), SwapHypothesis::Direct),
        ((56, "HB2"), SwapHypothesis::Direct),
        ((58, "HB2"), SwapHypothesis::Direct),
        ((59, "HB2"), SwapHypothesis::Swapped),
        ((60, "HB2"), SwapHypothesis::Direct),
        ((62, "HB2"), SwapHypothesis::Swapped),
        ((63, "HB2"), SwapHypothesis::Direct),
        ((64, "HB2"), SwapHypothesis::Swapped),
        ((65, "HB2"), SwapHypothesis::Direct),
        ((69, "HB2"), SwapHypothesis::Swapped),
        ((72, "HB2"), SwapHypothesis::Direct),
        ((73, "HB2"), SwapHypothesis::Direct),
    ]);

    // Dynamically averaged / g+ / proline / single-pathway surface sidechains that MUST abstain
    // (`!committed`, `Ambiguity_code = 2`) under genuine experimental 3J couplings:
    let must_abstain_sites: &[(i32, &str)] = &[
        (6, "HB2"),
        (11, "HB2"),
        (19, "HB2"),
        (20, "HB2"),
        (37, "HB2"),
        (38, "HB2"),
        (39, "HB2"),
        (52, "HB2"),
        (60, "HB2"),
        (63, "HB2"),
        (70, "CG1"),
        (72, "HB2"),
    ];

    // Rigid hydrophobic-core / H-bonded sidechains with multi-pathway experimental 3J couplings
    // (spanning Direct & Swapped, g- & t rotamers, and BetaMethylene + ValIsopropyl) that MUST commit:
    let must_commit_sites: &[(i32, &str)] = &[
        (4, "HB2"),
        (5, "CG1"),
        (17, "CG1"),
        (26, "CG1"),
        (29, "HB2"),
        (31, "HB2"),
        (40, "HB2"),
        (43, "HB2"),
        (54, "HB2"),
        (56, "HB2"),
        (58, "HB2"),
        (59, "HB2"),
        (64, "HB2"),
    ];

    let tmp_dir =
        std::env::temp_dir().join(format!("nmr_ubq_exp_3j_p0_2_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    for (mode_name, opt_pdb) in [
        ("exp_Ubq_3J_1D3Z_backbone", Some(pdb_path.as_path())),
        ("exp_Ubq_3J_CSI_only", None),
    ] {
        // 1. Direct unscrambled run on `ubiquitin_bmr5387_methyl.str` + `ubiquitin_exp_3j.tsv`
        let direct_out = assign_stereospecific_dataset(&StereospecDatasetInput {
            reference: &base_ref,
            j_couplings_tsv: Some(&exp_j_path),
            intra_noes_tsv: None,
            pdb_path: opt_pdb,
            config: StereospecConfig::default(),
        })
        .expect("run experimental Ubiquitin stereospec");

        let mut unscrambled_committed = 0usize;
        let mut unscrambled_correct = 0usize;
        for r in &direct_out.pair_results {
            if let Some(&truth) = bmrb17769_truth.get(&(r.seq_id, r.spec.iupac_branch_1)) {
                if r.committed {
                    unscrambled_committed += 1;
                    if r.winning_hypothesis == truth {
                        unscrambled_correct += 1;
                    } else {
                        println!(
                            "  [{mode_name}] MISMATCH on {} {} ({}): got {:?} (p={:.3}), expected {:?}",
                            r.res_name,
                            r.seq_id,
                            r.spec.iupac_branch_1,
                            r.winning_hypothesis,
                            r.winning_posterior,
                            truth
                        );
                    }
                }
            } else {
                // Degenerate or single-pathway sites outside the 39 non-degenerate paired entries
                // must never commit under experimental 3J-only data!
                assert!(
                    !r.committed && r.nmr_star_ambiguity_code == 2,
                    "[{mode_name}] non-paired/degenerate site {} {} ({}) unexpectedly committed",
                    r.res_name,
                    r.seq_id,
                    r.spec.iupac_branch_1
                );
            }
        }
        let unscrambled_prec =
            100.0 * (unscrambled_correct as f64) / (unscrambled_committed.max(1) as f64);
        println!(
            "[{mode_name}] unscrambled BMRB 5387 -> BMRB 17769: committed {unscrambled_committed}/39 ({:.1}% cov), precision = {unscrambled_correct}/{unscrambled_committed} ({unscrambled_prec:.2}%)",
            100.0 * (unscrambled_committed as f64) / 39.0
        );
        assert!(
            unscrambled_committed >= 14,
            "[{mode_name}] expected >= 14 committed experimental Ubiquitin sites, got {unscrambled_committed}"
        );
        assert!(
            unscrambled_prec >= 95.0,
            "[{mode_name}] experimental Ubiquitin precision {unscrambled_prec:.2}% < 95.0%"
        );

        // 2. 10-seed random 50% `(a, b)` scrambling of BOTH `Reference` shifts and paired `3J` TSV rows
        let mut total_eval = 0usize;
        let mut total_committed = 0usize;
        let mut total_correct = 0usize;

        for seed in 0..10u64 {
            let mut rng = Rng::new(0x0B10_3D32_E800_0000 ^ seed);
            let mut scr_ref = base_ref.clone();
            let mut flipped_sites: BTreeMap<(i32, &'static str), bool> = BTreeMap::new();

            for &(seq_id, site) in bmrb17769_truth.keys() {
                let flip = rng.f64() < 0.5;
                flipped_sites.insert((seq_id, site), flip);
                if flip {
                    if let Some(res_idx) = scr_ref.seq_ids.iter().position(|&s| s == seq_id) {
                        scramble_reference_prochiral_site(&mut scr_ref, res_idx, site);
                    }
                }
            }

            let scr_rows: Vec<JCouplingTsvRow> = base_j_rows
                .iter()
                .map(|row| {
                    let mut out_row = row.clone();
                    if let Some(vb) = row.val_b_hz {
                        let flip = flipped_sites
                            .get(&(row.seq_id, row.site.as_str()))
                            .copied()
                            .unwrap_or(false);
                        if flip {
                            out_row.val_a_hz = vb;
                            out_row.val_b_hz = Some(row.val_a_hz);
                        }
                    }
                    out_row
                })
                .collect();

            let scr_tsv = tmp_dir.join(format!("ubq_exp_3j_seed_{seed}.tsv"));
            write_j_couplings_tsv(&scr_tsv, &scr_rows).expect("write scrambled exp 3J TSV");

            let out = assign_stereospecific_dataset(&StereospecDatasetInput {
                reference: &scr_ref,
                j_couplings_tsv: Some(&scr_tsv),
                intra_noes_tsv: None,
                pdb_path: opt_pdb,
                config: StereospecConfig::default(),
            })
            .expect("run scrambled experimental Ubiquitin stereospec");

            let by_site: BTreeMap<(i32, &str), _> = out
                .pair_results
                .iter()
                .map(|r| ((r.seq_id, r.spec.iupac_branch_1), r))
                .collect();

            for &(sid, site) in must_abstain_sites {
                if let Some(r) = by_site.get(&(sid, site)) {
                    assert!(
                        !r.committed && r.nmr_star_ambiguity_code == 2,
                        "[{mode_name} seed {seed}] dynamically averaged / single-pathway site {sid} {site} falsely committed (p={:.3})",
                        r.winning_posterior
                    );
                }
            }
            for &(sid, site) in must_commit_sites {
                let r = by_site
                    .get(&(sid, site))
                    .unwrap_or_else(|| panic!("missing core site {sid} {site}"));
                assert!(
                    r.committed && r.nmr_star_ambiguity_code == 1,
                    "[{mode_name} seed {seed}] rigid core site {} {sid} {site} failed to commit (p={:.3})",
                    r.res_name,
                    r.winning_posterior
                );
            }

            for (&(sid, site), &base_truth) in &bmrb17769_truth {
                let r = by_site
                    .get(&(sid, site))
                    .unwrap_or_else(|| panic!("missing truth site {sid} {site}"));
                let flip = flipped_sites[&(sid, site)];
                let expected = if flip {
                    match base_truth {
                        SwapHypothesis::Direct => SwapHypothesis::Swapped,
                        SwapHypothesis::Swapped => SwapHypothesis::Direct,
                    }
                } else {
                    base_truth
                };
                total_eval += 1;
                if r.committed {
                    total_committed += 1;
                    if r.winning_hypothesis == expected {
                        total_correct += 1;
                        // Verify the output Reference chemical shifts match true BMRB 17769 IUPAC order!
                        let k = r.residue_idx;
                        let atom_check = if site == "CG1" { "CG1" } else { "HB2" };
                        let partner = if site == "CG1" { "CG2" } else { "HB3" };
                        let true_iupac_b1 = match base_truth {
                            SwapHypothesis::Direct => base_ref.shifts[&(k, atom_check.to_string())],
                            SwapHypothesis::Swapped => base_ref.shifts[&(k, partner.to_string())],
                        };
                        assert!(
                            (out.reference.shifts[&(k, atom_check.to_string())] - true_iupac_b1)
                                .abs()
                                < 1e-9,
                            "[{mode_name} seed {seed}] output shift mismatch on {sid} {atom_check}"
                        );
                        assert_eq!(out.ambiguity_codes[&(k, atom_check.to_string())], 1);
                    }
                }
            }
        }

        let scr_prec = 100.0 * (total_correct as f64) / (total_committed.max(1) as f64);
        let scr_cov = 100.0 * (total_committed as f64) / (total_eval as f64);
        println!(
            "[{mode_name}] 10-seed scrambled experimental Ubiquitin 3J: committed {total_committed}/{total_eval} ({scr_cov:.1}% cov), stereospecific precision = {total_correct}/{total_committed} ({scr_prec:.2}%)"
        );
        assert!(
            scr_prec >= 95.0,
            "[{mode_name}] 10-seed scrambled experimental Ubiquitin precision {scr_prec:.2}% < 95.0%"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

fn perturbed_karplus_eval(
    a: f64,
    b: f64,
    c: f64,
    eps_a: f64,
    eps_b: f64,
    phase_shift_deg: f64,
    theta_deg: f64,
) -> f64 {
    let rad = (theta_deg + phase_shift_deg).to_radians();
    let cs = rad.cos();
    let a_pert = a * (1.0 + eps_a);
    let b_pert = b * (1.0 + eps_b);
    (a_pert * cs * cs + b_pert * cs + c).max(0.0)
}

fn synthesize_perturbed_karplus_observables_from_pdb(
    pdb: &[AllAtomPdbResidue],
    res_idx: usize,
    spec: ProchiralPairSpec,
    eps_a: f64,
    eps_b: f64,
    phase_shift_deg: f64,
) -> Option<ProchiralObservables> {
    let mut obs = synthesize_ideal_observables_from_pdb(pdb, res_idx, spec)?;
    let cur = pdb.get(res_idx)?;

    let j_hahb =
        |th: f64| perturbed_karplus_eval(9.5, -1.6, 1.8, eps_a, eps_b, phase_shift_deg, th);
    let j_hnha =
        |th: f64| perturbed_karplus_eval(7.09, -1.42, 1.55, eps_a, eps_b, phase_shift_deg, th);
    let j_nhb = |th: f64| {
        let rad = (th + phase_shift_deg).to_radians();
        let cs = rad.cos();
        (-4.4 * (1.0 + eps_a) * cs * cs + 1.2 * (1.0 + eps_b) * cs + 0.1).abs()
    };
    let j_ncg_val =
        |th: f64| perturbed_karplus_eval(2.64, -0.26, 0.05, eps_a, eps_b, phase_shift_deg, th);
    let j_ccg_val =
        |th: f64| perturbed_karplus_eval(4.26, -1.00, 0.27, eps_a, eps_b, phase_shift_deg, th);
    let j_ccg_leu =
        |th: f64| perturbed_karplus_eval(4.02, -1.16, 0.24, eps_a, eps_b, phase_shift_deg, th);
    let j_ccg_cb = |rn: &str, th: f64| {
        let (a, b, c) = match rn {
            "PHE" | "TYR" | "TRP" | "HIS" => (2.31, -0.87, 0.49),
            "ASP" | "ASN" => (3.69, -1.16, 0.55),
            _ => (2.31, -0.87, 0.29),
        };
        perturbed_karplus_eval(a, b, c, eps_a, eps_b, phase_shift_deg, th)
    };

    match spec.kind {
        ProchiralSiteKind::BetaMethylene => {
            let (n, ca, cb, ha, hb2, hb3) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("CB")?,
                cur.coord("HA")?,
                cur.coord("HB2")?,
                cur.coord("HB3")?,
            );
            obs.j_hh_vicinal_hz = Some((
                j_hahb(calc_dihedral_deg(ha, ca, cb, hb2)),
                j_hahb(calc_dihedral_deg(ha, ca, cb, hb3)),
            ));
            obs.j_nhb_hz = Some((
                j_nhb(calc_dihedral_deg(n, ca, cb, hb2)),
                j_nhb(calc_dihedral_deg(n, ca, cb, hb3)),
            ));
            if !matches!(cur.res_name.as_str(), "PRO" | "SER" | "CYS" | "ALA" | "GLY") {
                if let (Some(c_prime), Some(cg)) = (cur.coord("C"), cur.coord("CG")) {
                    obs.j_ccg_single_hz = Some(j_ccg_cb(
                        &cur.res_name,
                        calc_dihedral_deg(c_prime, ca, cb, cg),
                    ));
                }
            }
        }
        ProchiralSiteKind::ValIsopropyl => {
            let (n, ca, c, cb, ha, hb, cg1, cg2) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("C")?,
                cur.coord("CB")?,
                cur.coord("HA")?,
                cur.coord("HB")?,
                cur.coord("CG1")?,
                cur.coord("CG2")?,
            );
            obs.j_ncg_hz = Some((
                j_ncg_val(calc_dihedral_deg(n, ca, cb, cg1)),
                j_ncg_val(calc_dihedral_deg(n, ca, cb, cg2)),
            ));
            obs.j_cc_vicinal_hz = Some((
                j_ccg_val(calc_dihedral_deg(c, ca, cb, cg1)),
                j_ccg_val(calc_dihedral_deg(c, ca, cb, cg2)),
            ));
            obs.j_methine_hahb_hz = Some(j_hahb(calc_dihedral_deg(ha, ca, cb, hb)));
        }
        ProchiralSiteKind::LeuIsopropyl => {
            let (ca, cb, cg, cd1, cd2) = (
                cur.coord("CA")?,
                cur.coord("CB")?,
                cur.coord("CG")?,
                cur.coord("CD1")?,
                cur.coord("CD2")?,
            );
            obs.j_cc_vicinal_hz = Some((
                j_ccg_leu(calc_dihedral_deg(ca, cb, cg, cd1)),
                j_ccg_leu(calc_dihedral_deg(ca, cb, cg, cd2)),
            ));
        }
        ProchiralSiteKind::GlyAlphaMethylene => {
            let (n, ca, ha2, ha3) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("HA2")?,
                cur.coord("HA3")?,
            );
            if let Some(h) = cur.coord("H") {
                obs.j_hh_vicinal_hz = Some((
                    j_hnha(calc_dihedral_deg(h, n, ca, ha2)),
                    j_hnha(calc_dihedral_deg(h, n, ca, ha3)),
                ));
            }
        }
        ProchiralSiteKind::GammaMethylene => {
            let cb = cur.coord("CB")?;
            let ha = cur.coord("HA")?;
            if cur.res_name == "ILE" {
                let (ca, cg1, hb, hg12, hg13) = (
                    cur.coord("CA")?,
                    cur.coord("CG1")?,
                    cur.coord("HB")?,
                    cur.coord("HG12")?,
                    cur.coord("HG13")?,
                );
                obs.j_hh_vicinal_hz = Some((
                    j_hahb(calc_dihedral_deg(hb, cb, cg1, hg12)),
                    j_hahb(calc_dihedral_deg(hb, cb, cg1, hg13)),
                ));
                obs.j_methine_hahb_hz = Some(j_hahb(calc_dihedral_deg(ha, ca, cb, hb)));
            } else {
                let (cg, hg2, hg3, hb2, hb3) = (
                    cur.coord("CG")?,
                    cur.coord("HG2")?,
                    cur.coord("HG3")?,
                    cur.coord("HB2")?,
                    cur.coord("HB3")?,
                );
                let j2 = 0.5
                    * (j_hahb(calc_dihedral_deg(hb2, cb, cg, hg2))
                        + j_hahb(calc_dihedral_deg(hb3, cb, cg, hg2)));
                let j3 = 0.5
                    * (j_hahb(calc_dihedral_deg(hb2, cb, cg, hg3))
                        + j_hahb(calc_dihedral_deg(hb3, cb, cg, hg3)));
                obs.j_hh_vicinal_hz = Some((j2, j3));
            }
        }
        _ => {}
    }

    Some(obs)
}

#[test]
fn test_p0_2_perturbed_karplus_parameter_mismatch() {
    let pdb_path = repo_file("data/pdb_refs/1d3z.pdb");
    let star_path = repo_file("data/fixtures/synthetic_ubiquitin.str");
    let pdb = parse_pdb_all_atoms_file(&pdb_path).expect("parse 1d3z.pdb");
    let star_ref = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    assert_eq!(pdb.len(), 76);

    let pdb_backbones: Vec<BackboneConditioning> = (0..pdb.len())
        .map(|i| backbone_conditioning_from_all_atom_pdb(&pdb, i))
        .collect();
    let csi_restraints = predict_dihedrals(&star_ref);
    let csi_backbones: Vec<BackboneConditioning> = csi_restraints
        .iter()
        .map(BackboneConditioning::from_dihedral_restraint)
        .collect();

    let cfg = StereospecConfig::default();
    let j_sigma_hz = 0.35;
    let noe_sigma_ln = 0.18;

    // 10 deterministic Karplus misspecification regimes spanning extremal ±15% (A, B) scaling
    // and ±8 deg dihedral phase shifts simultaneously across all coupling pathways:
    let mismatch_regimes: [(f64, f64, f64); 10] = [
        (0.15, 0.15, 8.0),
        (-0.15, -0.15, -8.0),
        (0.15, -0.15, 8.0),
        (-0.15, 0.15, -8.0),
        (0.15, -0.10, -8.0),
        (-0.15, 0.10, 8.0),
        (0.12, 0.14, -6.5),
        (-0.13, -0.11, 7.0),
        (0.10, -0.15, 5.5),
        (-0.11, 0.15, -7.5),
    ];

    for (mode_name, backbones) in [
        ("perturbed_Karplus_1D3Z_backbone", &pdb_backbones),
        ("perturbed_Karplus_CSI_backbone", &csi_backbones),
    ] {
        let mut meth_iso_total = 0usize;
        let mut meth_iso_committed = 0usize;
        let mut meth_iso_correct = 0usize;
        let mut all_committed = 0usize;
        let mut all_correct = 0usize;

        for (seed, &(eps_a, eps_b, phase_deg)) in mismatch_regimes.iter().enumerate() {
            let mut rng = Rng::new(0xCA20_1050_0000_0000 ^ (seed as u64));
            let mut per_res_obs: Vec<BTreeMap<String, ProchiralObservables>> =
                vec![BTreeMap::new(); pdb.len()];
            let mut ground_truth: BTreeMap<(usize, &'static str), SwapHypothesis> = BTreeMap::new();

            for (res_idx, res) in pdb.iter().enumerate() {
                for spec in prochiral_pairs_for_residue(&res.res_name) {
                    if let Some(mut pert_obs) = synthesize_perturbed_karplus_observables_from_pdb(
                        &pdb, res_idx, spec, eps_a, eps_b, phase_deg,
                    ) {
                        if matches!(
                            spec.kind,
                            ProchiralSiteKind::AsnCarboxamide | ProchiralSiteKind::GlnCarboxamide
                        ) {
                            pert_obs.proton_shifts_ppm =
                                Some((7.45 + rng.normal(0.0, 0.12), 6.85 + rng.normal(0.0, 0.12)));
                        }
                        let (noisy, truth) = perturb_and_scramble_observables(
                            &pert_obs,
                            &mut rng,
                            j_sigma_hz,
                            noe_sigma_ln,
                        );
                        per_res_obs[res_idx].insert(spec.iupac_branch_1.to_string(), noisy);
                        ground_truth.insert((res_idx, spec.iupac_branch_1), truth);
                    }
                }
            }

            let results = infer_protein_stereospec(
                &star_ref.sequence,
                &star_ref.seq_ids,
                backbones,
                &per_res_obs,
                &cfg,
            );

            for r in &results {
                let Some(&truth) = ground_truth.get(&(r.residue_idx, r.spec.iupac_branch_1)) else {
                    continue;
                };
                if r.committed {
                    all_committed += 1;
                    if r.winning_hypothesis == truth {
                        all_correct += 1;
                    }
                }
                if matches!(
                    r.spec.kind,
                    ProchiralSiteKind::BetaMethylene
                        | ProchiralSiteKind::GammaMethylene
                        | ProchiralSiteKind::GlyAlphaMethylene
                        | ProchiralSiteKind::ValIsopropyl
                        | ProchiralSiteKind::LeuIsopropyl
                ) {
                    meth_iso_total += 1;
                    if r.committed {
                        meth_iso_committed += 1;
                        if r.winning_hypothesis == truth {
                            meth_iso_correct += 1;
                        }
                    }
                }
            }
        }

        let meth_iso_cov = 100.0 * (meth_iso_committed as f64) / (meth_iso_total as f64);
        let meth_iso_prec = 100.0 * (meth_iso_correct as f64) / (meth_iso_committed as f64);
        let all_prec = 100.0 * (all_correct as f64) / (all_committed as f64);
        println!(
            "[{mode_name}] 10-regime (±15% A,B & ±8° phase) Ubiquitin 1D3Z: meth+iso committed {meth_iso_committed}/{meth_iso_total} ({meth_iso_cov:.1}% cov), precision = {meth_iso_prec:.2}% (all sites {all_correct}/{all_committed} = {all_prec:.2}%)"
        );

        assert!(
            meth_iso_cov >= 75.0,
            "[{mode_name}] coverage collapsed under ±15%/±8° Karplus mismatch: {meth_iso_cov:.1}% < 75.0%"
        );
        assert!(
            meth_iso_prec >= 95.0,
            "[{mode_name}] committed meth+iso precision {meth_iso_prec:.2}% < 95.0% under ±15%/±8° Karplus mismatch"
        );
        assert!(
            all_prec >= 95.0,
            "[{mode_name}] overall committed precision {all_prec:.2}% < 95.0% under ±15%/±8° Karplus mismatch"
        );
    }

    // Rare-Lovell-rotamer immunity under simultaneous +15% amplitude & +8° phase-shift mismatch:
    // In alpha-helical backbone (`phi = -62°, psi = -42°`), raw Lovell 2000 prior strongly disfavors
    // `g+` (`0.03` on VAL vs `0.90` for `t`; `0.02` on PHE vs `0.56` for `g-`). Multi-pathway `g+`
    // couplings (scrambled to `Swapped`) MUST still commit cleanly to `Swapped` without prior override!
    let helix_bb = BackboneConditioning::ExactTorsions {
        phi_deg: -62.0,
        psi_deg: -42.0,
    };
    let val_spec = prochiral_pairs_for_residue("VAL")[0];
    let (th_n1, th_n2, th_c1, th_c2, th_hahb) = iupac_val_cg_dihedrals_deg(63.0);
    let val_gplus_swapped = ProchiralObservables {
        j_ncg_hz: Some((
            perturbed_karplus_eval(2.64, -0.26, 0.05, 0.15, -0.15, 8.0, th_n2),
            perturbed_karplus_eval(2.64, -0.26, 0.05, 0.15, -0.15, 8.0, th_n1),
        )),
        j_cc_vicinal_hz: Some((
            perturbed_karplus_eval(4.26, -1.00, 0.27, 0.15, -0.15, 8.0, th_c2),
            perturbed_karplus_eval(4.26, -1.00, 0.27, 0.15, -0.15, 8.0, th_c1),
        )),
        j_methine_hahb_hz: Some(perturbed_karplus_eval(
            9.5, -1.6, 1.8, 0.15, -0.15, 8.0, th_hahb,
        )),
        ..Default::default()
    };
    let val_rare_res = infer_prochiral_pair(0, 1, val_spec, &helix_bb, &val_gplus_swapped, &cfg);
    assert!(
        val_rare_res.committed && val_rare_res.winning_hypothesis == SwapHypothesis::Swapped,
        "alpha-helical VAL in rare g+ rotamer failed under perturbed Karplus: {:?}",
        val_rare_res
    );
    assert_eq!(
        val_rare_res.dominant_rotamer,
        Some(RotamerState::GauchePlus)
    );

    // Also test rare alpha-helical PHE in `t` / `g-` with perturbed-Karplus `3J_HAHB` + single-CG `3J_CCG`
    let phe_spec = prochiral_pairs_for_residue("PHE")[0];
    let (th_hb2_t, th_hb3_t, _, _) = iupac_cb_proton_dihedrals_deg(177.0);
    let phe_t_swapped = ProchiralObservables {
        j_hh_vicinal_hz: Some((
            perturbed_karplus_eval(9.5, -1.6, 1.8, 0.15, -0.15, 8.0, th_hb3_t),
            perturbed_karplus_eval(9.5, -1.6, 1.8, 0.15, -0.15, 8.0, th_hb2_t),
        )),
        j_ccg_single_hz: Some(perturbed_karplus_eval(
            2.31, -0.87, 0.49, 0.15, -0.15, 8.0, 57.0,
        )),
        ..Default::default()
    };
    let phe_res = infer_prochiral_pair(0, 1, phe_spec, &helix_bb, &phe_t_swapped, &cfg);
    assert!(
        phe_res.committed && phe_res.winning_hypothesis == SwapHypothesis::Swapped,
        "PHE with perturbed-Karplus 3J_HAHB + Hu&Bax 3J_CCG failed to commit to Swapped: {:?}",
        phe_res
    );
}
