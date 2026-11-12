use std::collections::BTreeMap;
use std::path::Path;

use nmr_core::dihedrals::{predict_dihedrals, wrap_deg, SecondaryStructure};
use nmr_core::nef::{
    classify_nef_atom_name, read_nef_str, write_nef_str, NefAtomWildcardKind, NefDocument,
    NefMolecularSystem,
};
use nmr_core::nmrstar::{read_star, read_star_str_all_atoms, Reference};
use nmr_core::noesy::{calc_dihedral_deg, v_add, v_dist, v_scale};
use nmr_core::rng::Rng;
use nmr_core::stereospec::{
    apply_stereospec_to_reference, backbone_conditioning_from_all_atom_pdb,
    build_local_peptide_frame, format_stereospec_nmrstar, infer_prochiral_pair,
    infer_protein_stereospec, infer_residue_stereospec, iupac_cb_proton_dihedrals_deg,
    iupac_gly_ha_dihedrals_deg, iupac_leu_cd_dihedrals_deg, iupac_val_cg_dihedrals_deg,
    karplus_3j_ccg_hz, karplus_3j_hahb_hz, karplus_3j_ncg_hz, karplus_3j_nhb_mag_hz,
    parse_pdb_all_atoms_file, prochiral_pairs_for_residue, synthesize_ideal_observables_from_pdb,
    to_nef_shift_list_with_stereospec, to_star_chem_shift_rows, AnchorProton, BackboneConditioning,
    NoePairObs, ProchiralObservables, ProchiralSiteKind, RotamerState, StereospecConfig,
    SwapHypothesis,
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
