//! Quantitative unit and benchmark tests for `T10.4` & `T10.5` (`crates/nmr-core/src/methyl_trosy.rs`
//! and `crates/nmr-core/src/methyl_solver.rs`).
//!
//! Provenance: Clean-room verification suite citing:
//! - Tugarinov, Hwang, Ollerenshaw & Kay (*J. Am. Chem. Soc.* 2003, 125, 10120–10121 & 10420–10428)
//! - Gans et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962)
//! - Ruschak & Kay (*J. Biomol. NMR* 2010, 46, 75–87)
//! - Xu, Lin, Ho & Yang (*J. Am. Chem. Soc.* 2009, 131, 9480–9481)
//! - Chao, Shi, Masterson & Veglia (*J. Magn. Reson.* 2012, 214, 103–110)
//! - Pritišanac et al. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533)

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use nmr_core::methyl_solver::{
    assign_methyl_cpu, build_methyl_contact_problem, evaluate_methyl_assignment,
    evaluate_methyl_assignment_against_truth, evaluate_methyl_objective, evaluate_methyl_raw_score,
    sample_methyl_mcmc_terminal_cpu, MethylAnnealConfig, MethylAssignResult, MethylContactProblem,
    MethylScoreConfig, ObsAdjEntry, ObsNoeEdge, StructShortAdjEntry, StructShortEdge,
    METHYL_FORBID,
};
use nmr_core::methyl_trosy::{
    attach_hmcmcbca_records, build_c2_homodimer_methyl_model, build_cn_homooligomer_methyl_model,
    classify_methyl_peak, extract_hmcmcbca_from_reference, extract_methyl_structure_from_pdb_file,
    extract_multichain_methyl_structure_from_pdb_str, extract_true_site_shifts,
    link_hmcmcbca_geminal_pairs, pick_methyl_hmqc_2d, pick_methyl_noesy_3d, pick_methyl_noesy_4d,
    simulate_methyl_hmqc_2d, simulate_methyl_noesy_3d, simulate_methyl_noesy_4d, HmcmcbcaRecord,
    MethylClassPosterior, MethylLabelingScheme, MethylNoePeak, MethylPickerConfig,
    MethylResidueClass, MethylRootPeak, MethylSimConfig, MethylStructureModel,
};
use nmr_core::nmrstar::read_star_all_atoms;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Evaluate 2D HMQC root recovery against resolvable physical peak clusters (coalescing exact
/// superpositions within `0.35 * FWHM` where two methyls form a single indistinguishable maximum)
/// and artifact rate against true methyl sites.
fn evaluate_hmqc_picker_quality(
    true_hc: &[(f64, f64)],
    picked: &[MethylRootPeak],
    sim_cfg: &MethylSimConfig,
) -> (f64, f64) {
    let fwhm_h = sim_cfg.fwhm_h_ppm();
    let fwhm_c = sim_cfg.fwhm_c_ppm();

    // Cluster true methyls that sit closer than 0.42 * FWHM simultaneously in 1H and 13C
    let mut cluster_centers: Vec<(f64, f64)> = Vec::new();
    for &(th, tc) in true_hc {
        let merged = cluster_centers
            .iter()
            .any(|&(ch, cc)| (th - ch).abs() < 0.42 * fwhm_h && (tc - cc).abs() < 0.42 * fwhm_c);
        if !merged {
            cluster_centers.push((th, tc));
        }
    }

    let tol_h = 0.032; // ~0.85 * FWHM_H
    let tol_c = 0.105; // ~1.15 * FWHM_C
    let recovered = cluster_centers
        .iter()
        .filter(|&&(th, tc)| {
            picked
                .iter()
                .any(|p| (p.h_ppm - th).abs() <= tol_h && (p.c_ppm - tc).abs() <= tol_c)
        })
        .count();

    let artifacts = picked
        .iter()
        .filter(|p| {
            !true_hc.iter().any(|&(th, tc)| {
                (p.h_ppm - th).abs() <= tol_h * 1.25 && (p.c_ppm - tc).abs() <= tol_c * 1.25
            })
        })
        .count();

    let rec_pct = 100.0 * (recovered as f64) / (cluster_centers.len().max(1) as f64);
    let art_pct = 100.0 * (artifacts as f64) / (picked.len().max(1) as f64);
    (rec_pct, art_pct)
}

/// Evaluate 3D C-C-H or 4D C-H-C-H NOESY cross-peak recovery for non-diagonal-masked `d_CC <= 8.5 A`
/// methyl pairs and false-peak artifact rate against simulated `d_CC <= r_max_noe_angstrom` contacts.
fn evaluate_noesy_picker_quality(
    model: &MethylStructureModel,
    true_hc: &[(f64, f64)],
    picked_noes: &[MethylNoePeak],
    sim_cfg: &MethylSimConfig,
    picker_cfg: &MethylPickerConfig,
    is_4d: bool,
) -> (f64, f64, usize, usize) {
    let m_len = model.sites.len();
    let mut target_pairs_85: BTreeSet<(usize, usize)> = BTreeSet::new();
    for m1 in 0..m_len {
        for m2 in (m1 + 1)..m_len {
            let d = model.dist_cc(m1, m2);
            if d <= 8.5 {
                let (h1, c1) = true_hc[m1];
                let (h2, c2) = true_hc[m2];
                let diag_masked = if is_4d {
                    (c1 - c2).abs() <= picker_cfg.diag_tol_c_ppm
                        && (h1 - h2).abs() <= picker_cfg.diag_tol_h_ppm
                } else {
                    (c1 - c2).abs() <= picker_cfg.diag_tol_c_ppm * 1.15
                };
                if !diag_masked {
                    target_pairs_85.insert((m1, m2));
                }
            }
        }
    }

    let tol_h = 0.042;
    let tol_c = 0.135;
    let mut recovered_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut artifacts = 0usize;

    for pk in picked_noes {
        let mut matched_any_true = false;
        for (m_rec, &(hr, cr)) in true_hc.iter().enumerate().take(m_len) {
            if (pk.h_root_ppm - hr).abs() > tol_h || (pk.c_root_ppm - cr).abs() > tol_c {
                continue;
            }
            for (m_don, &(hd, cd)) in true_hc.iter().enumerate().take(m_len) {
                if m_don == m_rec {
                    continue;
                }
                if (pk.c_noe_ppm - cd).abs() > tol_c {
                    continue;
                }
                if let Some(hn) = pk.h_noe_ppm {
                    if (hn - hd).abs() > tol_h * 1.25 {
                        continue;
                    }
                }
                if model.dist_cc(m_rec, m_don) <= sim_cfg.r_max_noe_angstrom + 0.15 {
                    matched_any_true = true;
                    let pair = (m_rec.min(m_don), m_rec.max(m_don));
                    if target_pairs_85.contains(&pair) {
                        recovered_pairs.insert(pair);
                    }
                }
            }
        }
        if !matched_any_true {
            artifacts += 1;
        }
    }

    let rec_pct = 100.0 * (recovered_pairs.len() as f64) / (target_pairs_85.len().max(1) as f64);
    let art_pct = 100.0 * (artifacts as f64) / (picked_noes.len().max(1) as f64);
    (
        rec_pct,
        art_pct,
        recovered_pairs.len(),
        target_pairs_85.len(),
    )
}

#[test]
fn methyl_labeling_schemes_and_pdb_aromatic_extraction() {
    let root = repo_root();
    let ubq_pdb = root.join("data/pdb_refs/1d3z.pdb");
    let mbp_pdb = root.join("data/fixtures/mbp_1ezp.pdb");

    // 1. CLI token parsing for all canonical & custom schemes
    assert_eq!(
        MethylLabelingScheme::parse("ILV-proS").unwrap(),
        MethylLabelingScheme::IlvProS
    );
    assert_eq!(
        MethylLabelingScheme::parse("ILV-proR").unwrap(),
        MethylLabelingScheme::IlvProR
    );
    assert_eq!(
        MethylLabelingScheme::parse("ILV-racemic").unwrap(),
        MethylLabelingScheme::IlvRacemic
    );
    assert_eq!(
        MethylLabelingScheme::parse("ILVMAT").unwrap(),
        MethylLabelingScheme::Ilvmat
    );
    let custom = MethylLabelingScheme::parse("ILE:CD1,ALA:CB,MET:CE").unwrap();
    let ubq_custom = extract_methyl_structure_from_pdb_file(&ubq_pdb, &custom).unwrap();
    // Ubiquitin (76 aa): 7 Ile + 2 Ala + 1 Met = 10 sites
    assert_eq!(ubq_custom.sites.len(), 10);

    // 2. Ubiquitin (1D3Z, 76 aa): 7 Ile, 9 Leu, 4 Val, 1 Met, 2 Ala, 7 Thr
    let ubq_pros =
        extract_methyl_structure_from_pdb_file(&ubq_pdb, &MethylLabelingScheme::IlvProS).unwrap();
    let ubq_pror =
        extract_methyl_structure_from_pdb_file(&ubq_pdb, &MethylLabelingScheme::IlvProR).unwrap();
    let ubq_rac =
        extract_methyl_structure_from_pdb_file(&ubq_pdb, &MethylLabelingScheme::IlvRacemic)
            .unwrap();
    let ubq_ilvmat =
        extract_methyl_structure_from_pdb_file(&ubq_pdb, &MethylLabelingScheme::Ilvmat).unwrap();
    assert_eq!(ubq_pros.sites.len(), 20); // 7 Ile + 9 Leu-CD2 + 4 Val-CG2
    assert_eq!(ubq_pror.sites.len(), 20); // 7 Ile + 9 Leu-CD1 + 4 Val-CG1
    assert_eq!(ubq_rac.sites.len(), 33); // 7 Ile + 18 Leu + 8 Val
    assert_eq!(ubq_ilvmat.sites.len(), 43); // 33 + 1 Met + 2 Ala + 7 Thr
    assert!(ubq_pros.sites.iter().all(|s| s.geminal_partner.is_none()));
    assert_eq!(
        ubq_ilvmat
            .sites
            .iter()
            .filter(|s| s.geminal_partner.is_some())
            .count(),
        26 // 9*2 Leu + 4*2 Val
    );

    // 3. E. coli Maltose-Binding Protein (PDB 1EZP, 370 aa, 41 kDa):
    // 22 Ile, 30 Leu, 20 Val, 6 Met, 44 Ala, 21 Thr
    let mbp_pros =
        extract_methyl_structure_from_pdb_file(&mbp_pdb, &MethylLabelingScheme::IlvProS).unwrap();
    let mbp_rac =
        extract_methyl_structure_from_pdb_file(&mbp_pdb, &MethylLabelingScheme::IlvRacemic)
            .unwrap();
    let mbp_ilvmat =
        extract_methyl_structure_from_pdb_file(&mbp_pdb, &MethylLabelingScheme::Ilvmat).unwrap();
    assert_eq!(mbp_ilvmat.sequence.len(), 370);
    assert_eq!(mbp_pros.sites.len(), 72); // 22 Ile + 30 Leu-CD2 + 20 Val-CG2
    assert_eq!(mbp_rac.sites.len(), 122); // 22 Ile + 60 Leu + 40 Val
    assert_eq!(mbp_ilvmat.sites.len(), 193); // 122 + 6 Met + 44 Ala + 21 Thr
    assert!(
        mbp_ilvmat.aromatic_rings.len() >= 35,
        "expected >= 35 aromatic rings in MBP 1EZP, got {}",
        mbp_ilvmat.aromatic_rings.len()
    );

    // Verify ring-current proximity variance inflation on aromatic-packed MBP Leu160-CD1 (-1.02 ppm in BMRB 7114)
    let leu160_cd1 = mbp_ilvmat
        .sites
        .iter()
        .find(|s| s.seq_id == 160 && s.carbon_atom == "CD1")
        .expect("LEU 160 CD1 present in 1EZP");
    assert!(
        leu160_cd1.ring_current_h_extra_sd_ppm > 0.25 && leu160_cd1.ring_current_h_offset_ppm < -0.10,
        "expected upfield aromatic perturbation at LEU160 CD1, got offset={:.2} ppm, extra_sd={:.2} ppm",
        leu160_cd1.ring_current_h_offset_ppm,
        leu160_cd1.ring_current_h_extra_sd_ppm
    );
}

#[test]
fn bayesian_methyl_classifier_and_hmcmcbca_geminal_linking() {
    // Canonical 2D (1H, 13C) classification under ILVMAT
    let p_ile = classify_methyl_peak(0.72, 12.1, None, &MethylLabelingScheme::Ilvmat);
    assert_eq!(p_ile.map_class().0, MethylResidueClass::Ile);
    assert!(p_ile.prob(MethylResidueClass::Ile) > 0.90);

    let p_met = classify_methyl_peak(2.05, 16.5, None, &MethylLabelingScheme::Ilvmat);
    assert_eq!(p_met.map_class().0, MethylResidueClass::Met);
    assert!(p_met.prob(MethylResidueClass::Met) > 0.85);

    let p_ala = classify_methyl_peak(1.36, 18.2, None, &MethylLabelingScheme::Ilvmat);
    assert_eq!(p_ala.map_class().0, MethylResidueClass::Ala);
    assert!(p_ala.prob(MethylResidueClass::Ala) > 0.70);

    let p_thr = classify_methyl_peak(1.22, 21.3, None, &MethylLabelingScheme::Ilvmat);
    assert_eq!(p_thr.map_class().0, MethylResidueClass::Thr);
    assert!(p_thr.prob(MethylResidueClass::Thr) > 0.60);

    // Labeling mask strictly zeroes inactive classes (Met/Ala/Thr == 0 under IlvProS)
    let p_masked = classify_methyl_peak(1.36, 18.2, None, &MethylLabelingScheme::IlvProS);
    assert_eq!(p_masked.prob(MethylResidueClass::Ala), 0.0);
    assert_eq!(p_masked.prob(MethylResidueClass::Met), 0.0);
    assert_eq!(p_masked.prob(MethylResidueClass::Thr), 0.0);

    // HMCM(CG)CBCA cleanly separates overlapping Leu vs Val at (0.88 ppm, 23.4 ppm)
    let hmcm_leu = HmcmcbcaRecord {
        h_ppm: 0.88,
        c_methyl_ppm: 23.4,
        cg_ppm: Some(26.7),
        cb_ppm: Some(42.4),
        ca_ppm: Some(55.2),
    };
    let hmcm_val = HmcmcbcaRecord {
        h_ppm: 0.88,
        c_methyl_ppm: 21.8,
        cg_ppm: None,
        cb_ppm: Some(32.8),
        ca_ppm: Some(62.4),
    };
    let post_leu = classify_methyl_peak(
        0.88,
        23.4,
        Some(&hmcm_leu),
        &MethylLabelingScheme::IlvRacemic,
    );
    let post_val = classify_methyl_peak(
        0.88,
        21.8,
        Some(&hmcm_val),
        &MethylLabelingScheme::IlvRacemic,
    );
    assert!(post_leu.prob(MethylResidueClass::Leu) > 0.99);
    assert!(post_val.prob(MethylResidueClass::Val) > 0.99);

    // Real BMRB 7114 HMCM(CG)CBCA geminal pairing on MBP (30 Leu + 20 Val = 50 geminal pairs)
    let root = repo_root();
    let mbp_model = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1ezp.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let true_hc = extract_true_site_shifts(&mbp_model, &mbp_ref);
    let mut roots: Vec<MethylRootPeak> = true_hc
        .iter()
        .enumerate()
        .map(|(id, &(h_ppm, c_ppm))| MethylRootPeak {
            id,
            h_ppm,
            c_ppm,
            height: 1.0,
            volume: 1.0,
            hmcm: None,
        })
        .collect();
    let hmcm_recs = extract_hmcmcbca_from_reference(&mbp_model, &mbp_ref);
    attach_hmcmcbca_records(&mut roots, &hmcm_recs, 0.03, 0.10);
    let linked = link_hmcmcbca_geminal_pairs(&roots, 0.18);
    assert!(
        linked.len() >= 48,
        "expected >= 48/50 Leu/Val geminal pairs linked in MBP, got {}",
        linked.len()
    );
}

#[test]
fn t10_4_high_mw_hmqc_and_3d_4d_noesy_picker_quantitative_gate() {
    let root = repo_root();
    let sim_cfg = MethylSimConfig::default(); // 800 MHz, 25.0 Hz 1H LW, 18.0 Hz 13C LW (40-80 kDa)
    let picker_cfg = MethylPickerConfig::default();
    assert_eq!(sim_cfg.lw_h_hz, 25.0);
    assert_eq!(sim_cfg.lw_c_hz, 18.0);

    // 1. Ubiquitin (1D3Z + BMRB 5387, ILVMAT, 43 methyls) at 40-80 kDa linewidths: 2D + 3D + 4D
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
    let ubq_hc = extract_true_site_shifts(&ubq_model, &ubq_ref);

    let ubq_hmqc = simulate_methyl_hmqc_2d(&ubq_hc, &sim_cfg);
    let ubq_roots = pick_methyl_hmqc_2d(&ubq_hmqc, &picker_cfg);
    let (ubq_hmqc_rec, ubq_hmqc_art) = evaluate_hmqc_picker_quality(&ubq_hc, &ubq_roots, &sim_cfg);

    let ubq_noesy_3d = simulate_methyl_noesy_3d(&ubq_model, &ubq_hc, &sim_cfg);
    let ubq_noes_3d = pick_methyl_noesy_3d(&ubq_noesy_3d, &ubq_roots, &picker_cfg);
    let (ubq_noe3_rec, ubq_noe3_art, ubq_n3_got, ubq_n3_tot) = evaluate_noesy_picker_quality(
        &ubq_model,
        &ubq_hc,
        &ubq_noes_3d,
        &sim_cfg,
        &picker_cfg,
        false,
    );

    let ubq_noesy_4d = simulate_methyl_noesy_4d(&ubq_model, &ubq_hc, &sim_cfg, 84, 108);
    let ubq_noes_4d = pick_methyl_noesy_4d(&ubq_noesy_4d, &ubq_roots, &picker_cfg);
    let (ubq_noe4_rec, ubq_noe4_art, ubq_n4_got, ubq_n4_tot) = evaluate_noesy_picker_quality(
        &ubq_model,
        &ubq_hc,
        &ubq_noes_4d,
        &sim_cfg,
        &picker_cfg,
        true,
    );

    println!(
        "[T10.4 Ubiquitin ILVMAT 25/18 Hz] 2D HMQC: rec={ubq_hmqc_rec:.2}% art={ubq_hmqc_art:.2}% | \
         3D C-C-H NOESY (r<=8.5A): rec={ubq_noe3_rec:.2}% ({ubq_n3_got}/{ubq_n3_tot}) art={ubq_noe3_art:.2}% | \
         4D C-H-C-H NOESY: rec={ubq_noe4_rec:.2}% ({ubq_n4_got}/{ubq_n4_tot}) art={ubq_noe4_art:.2}%"
    );

    assert!(
        ubq_hmqc_rec >= 96.0 && ubq_hmqc_art < 3.0,
        "Ubiquitin 2D HMQC gate failed: rec={ubq_hmqc_rec:.2}%, art={ubq_hmqc_art:.2}%"
    );
    assert!(
        ubq_noe3_rec >= 96.0 && ubq_noe3_art < 3.0,
        "Ubiquitin 3D C-C-H NOESY gate failed: rec={ubq_noe3_rec:.2}%, art={ubq_noe3_art:.2}%"
    );
    assert!(
        ubq_noe4_rec >= 96.0 && ubq_noe4_art < 3.0,
        "Ubiquitin 4D C-H-C-H NOESY gate failed: rec={ubq_noe4_rec:.2}%, art={ubq_noe4_art:.2}%"
    );

    // 2. MBP (PDB 1EZP + BMRB 7114, 370 aa, 41 kDa, full ILVMAT 193 methyls) at 25 Hz / 18 Hz
    let mbp_model = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1ezp.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let mbp_hc = extract_true_site_shifts(&mbp_model, &mbp_ref);

    let mbp_hmqc = simulate_methyl_hmqc_2d(&mbp_hc, &sim_cfg);
    let mbp_roots = pick_methyl_hmqc_2d(&mbp_hmqc, &picker_cfg);
    let (mbp_hmqc_rec, mbp_hmqc_art) = evaluate_hmqc_picker_quality(&mbp_hc, &mbp_roots, &sim_cfg);

    let mbp_noesy_3d = simulate_methyl_noesy_3d(&mbp_model, &mbp_hc, &sim_cfg);
    let mbp_noes_3d = pick_methyl_noesy_3d(&mbp_noesy_3d, &mbp_roots, &picker_cfg);
    let (mbp_noe3_rec, mbp_noe3_art, mbp_n3_got, mbp_n3_tot) = evaluate_noesy_picker_quality(
        &mbp_model,
        &mbp_hc,
        &mbp_noes_3d,
        &sim_cfg,
        &picker_cfg,
        false,
    );

    println!(
        "[T10.4 MBP 1EZP ILVMAT 25/18 Hz] 2D HMQC: rec={mbp_hmqc_rec:.2}% art={mbp_hmqc_art:.2}% | \
         3D C-C-H NOESY (r<=8.5A): rec={mbp_noe3_rec:.2}% ({mbp_n3_got}/{mbp_n3_tot}) art={mbp_noe3_art:.2}%"
    );

    assert!(
        mbp_hmqc_rec >= 96.0 && mbp_hmqc_art < 3.0,
        "MBP 2D HMQC gate failed: rec={mbp_hmqc_rec:.2}%, art={mbp_hmqc_art:.2}%"
    );
    assert!(
        mbp_noe3_rec >= 96.0 && mbp_noe3_art < 3.0,
        "MBP 3D C-C-H NOESY gate failed: rec={mbp_noe3_rec:.2}%, art={mbp_noe3_art:.2}%"
    );
}

#[test]
fn t10_5_cpu_solver_ubiquitin_and_mbp_1ezp_benchmarks() {
    let root = repo_root();
    let sim_cfg = MethylSimConfig::default();
    let picker_cfg = MethylPickerConfig::default();
    let score_cfg = MethylScoreConfig::default();
    let anneal_cfg = MethylAnnealConfig {
        replicas: 48,
        steps_per_root: 2600,
        seed: 2026,
        ..MethylAnnealConfig::default()
    };

    // 1. Ubiquitin (1D3Z + BMRB 5387, ILVMAT, 43 methyls)
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
    let ubq_hc = extract_true_site_shifts(&ubq_model, &ubq_ref);
    let ubq_hmqc = simulate_methyl_hmqc_2d(&ubq_hc, &sim_cfg);
    let mut ubq_roots = pick_methyl_hmqc_2d(&ubq_hmqc, &picker_cfg);
    let ubq_hmcm = extract_hmcmcbca_from_reference(&ubq_model, &ubq_ref);
    attach_hmcmcbca_records(&mut ubq_roots, &ubq_hmcm, 0.045, 0.14);
    let ubq_noes = pick_methyl_noesy_3d(
        &simulate_methyl_noesy_3d(&ubq_model, &ubq_hc, &sim_cfg),
        &ubq_roots,
        &picker_cfg,
    );

    let ubq_prob = build_methyl_contact_problem(&ubq_model, &ubq_roots, &ubq_noes, &score_cfg);
    let ubq_res = assign_methyl_cpu(&ubq_prob, &anneal_cfg);
    let ubq_metrics = evaluate_methyl_assignment(&ubq_model, &ubq_roots, &ubq_res, &ubq_ref, 0.80);
    println!(
        "[T10.5 CPU Ubiquitin ILVMAT] acc={:.2}% ({}/{}) | HC(>=0.80)={:.2}% ({}/{}) | score={:.4}",
        ubq_metrics.accuracy_pct,
        ubq_metrics.correct,
        ubq_metrics.total_matched_roots,
        ubq_metrics.hc_precision_pct,
        ubq_metrics.hc_correct,
        ubq_metrics.hc_total,
        ubq_res.score
    );
    assert!(
        ubq_metrics.accuracy_pct >= 90.0 && ubq_metrics.hc_precision_pct >= 95.0,
        "CPU Ubiquitin benchmark failed: {ubq_metrics:?}"
    );

    // 2. MBP (1EZP + BMRB 7114, 370 aa, 41 kDa, ILVMAT, 193 methyls)
    let mbp_model = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/mbp_1ezp.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    let mbp_ref =
        read_star_all_atoms(root.join("data/fixtures/mbp_bmr7114_methyl.str"), None).unwrap();
    let mbp_hc = extract_true_site_shifts(&mbp_model, &mbp_ref);
    let mbp_hmqc = simulate_methyl_hmqc_2d(&mbp_hc, &sim_cfg);
    let mut mbp_roots = pick_methyl_hmqc_2d(&mbp_hmqc, &picker_cfg);
    let mbp_hmcm = extract_hmcmcbca_from_reference(&mbp_model, &mbp_ref);
    attach_hmcmcbca_records(&mut mbp_roots, &mbp_hmcm, 0.045, 0.14);
    let mbp_noes = pick_methyl_noesy_3d(
        &simulate_methyl_noesy_3d(&mbp_model, &mbp_hc, &sim_cfg),
        &mbp_roots,
        &picker_cfg,
    );

    let mbp_prob = build_methyl_contact_problem(&mbp_model, &mbp_roots, &mbp_noes, &score_cfg);
    let mbp_res = assign_methyl_cpu(&mbp_prob, &anneal_cfg);
    let mbp_metrics = evaluate_methyl_assignment(&mbp_model, &mbp_roots, &mbp_res, &mbp_ref, 0.80);
    println!(
        "[T10.5 CPU MBP 1EZP ILVMAT] acc={:.2}% ({}/{}) | HC(>=0.80)={:.2}% ({}/{}) | score={:.4}",
        mbp_metrics.accuracy_pct,
        mbp_metrics.correct,
        mbp_metrics.total_matched_roots,
        mbp_metrics.hc_precision_pct,
        mbp_metrics.hc_correct,
        mbp_metrics.hc_total,
        mbp_res.score
    );
    assert!(
        mbp_metrics.accuracy_pct >= 90.0 && mbp_metrics.hc_precision_pct >= 95.0,
        "CPU MBP 1EZP benchmark failed: {mbp_metrics:?}"
    );
}

#[test]
fn p1_10_methyl_bounds_guard_and_zero_hc_abstention_contract() {
    let root = repo_root();
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
    let ubq_hc = extract_true_site_shifts(&ubq_model, &ubq_ref);
    let ubq_roots = pick_methyl_hmqc_2d(
        &simulate_methyl_hmqc_2d(&ubq_hc, &MethylSimConfig::default()),
        &MethylPickerConfig::default(),
    );
    let prob =
        build_methyl_contact_problem(&ubq_model, &ubq_roots, &[], &MethylScoreConfig::default());

    // 1. Out-of-bounds (`m >= m_len`, `m == i32::MAX`, `m < -1`), wrong length, and duplicate site
    let mut oob_hi = vec![-1i32; prob.s_len];
    oob_hi[0] = prob.m_len as i32;
    assert_eq!(evaluate_methyl_raw_score(&prob, &oob_hi), -1.0e12);
    assert!((evaluate_methyl_objective(&prob, &oob_hi) - (-1.0e12 * prob.norm_inv)).abs() < 1e-3);

    let mut oob_max = vec![-1i32; prob.s_len];
    oob_max[prob.s_len - 1] = i32::MAX;
    assert_eq!(evaluate_methyl_raw_score(&prob, &oob_max), -1.0e12);

    let mut oob_neg = vec![-1i32; prob.s_len];
    oob_neg[0] = -2;
    assert_eq!(evaluate_methyl_raw_score(&prob, &oob_neg), -1.0e12);

    let wrong_len = vec![-1i32; prob.s_len - 1];
    assert_eq!(evaluate_methyl_raw_score(&prob, &wrong_len), -1.0e12);

    let mut dup = vec![-1i32; prob.s_len];
    dup[0] = 3;
    dup[1] = 3;
    assert_eq!(evaluate_methyl_raw_score(&prob, &dup), -1.0e12);

    // 2. Zero high-confidence predictions MUST report `hc_abstained = true`, `hc_precision_opt = None`,
    //    `hc_precision_pct = 0.0` (NEVER claiming `100.0%` on `0/0`!).
    let low_conf_res = MethylAssignResult {
        position: (0..prob.s_len as i32).collect(),
        confidence: vec![0.49; prob.s_len],
        class_posteriors: vec![
            MethylClassPosterior {
                probs: [1.0 / 6.0; 6]
            };
            prob.s_len
        ],
        raw_score: 0.0,
        score: 0.0,
        run_scores: vec![0.0],
    };
    let truth: Vec<Option<usize>> = (0..prob.s_len).map(Some).collect();
    let m_truth = evaluate_methyl_assignment_against_truth(&low_conf_res, &truth, 0.80);
    assert_eq!(m_truth.hc_total, 0);
    assert_eq!(m_truth.hc_correct, 0);
    assert!(m_truth.hc_abstained);
    assert_eq!(m_truth.hc_precision_opt, None);
    assert_eq!(m_truth.hc_precision_pct, 0.0);
    assert!(m_truth.reliability.is_abstained());

    let m_ref = evaluate_methyl_assignment(&ubq_model, &ubq_roots, &low_conf_res, &ubq_ref, 0.80);
    assert_eq!(m_ref.hc_total, 0);
    assert!(m_ref.hc_abstained);
    assert_eq!(m_ref.hc_precision_opt, None);
    assert_eq!(m_ref.hc_precision_pct, 0.0);
}

fn build_synthetic_mcmc_test_problem(s_len: usize, m_len: usize) -> MethylContactProblem {
    let d_cutoff = 8.5f32;
    let sigma_d = 1.1f32;
    let flat_bottom_d = 0.6f32;

    // Coordinates for up to 5 methyl sites: two geminal pairs `(0,1)` & `(2,3)` + singleton `4`
    let coords: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [2.5, 1.7, 0.0],  // d(0,1) = 3.02 A (geminal pair 0)
        [5.2, 0.4, 0.0],  // d(0,2) = 5.22 A, d(1,2) = 3.00 A
        [5.8, 3.3, 0.0],  // d(2,3) = 2.96 A (geminal pair 1), d(0,3) = 6.67 A
        [10.8, 0.2, 0.0], // d(2,4) = 5.60 A <= 8.5, d(0,4) = 10.80 A > 8.5 (asymmetric cut degree!)
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

    // Strongly unequal slot masks across roots to test Move 1E & Move 2 Hastings ratios
    let mut u_mat = vec![METHYL_FORBID; s_len * m_len];
    let mut slots = vec![Vec::new(); s_len];
    for s in 0..s_len {
        for m in 0..m_len {
            let allowed = match s {
                0 => true,                       // all M sites
                1 => m != (m_len - 1),           // M - 1 sites
                2 => true,                       // all M sites
                3 => m < 3.min(m_len),           // 3 sites
                _ => m == 0 || m == (m_len - 1), // 2 sites (ratio up to 2.5x!)
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

    // Asymmetric observed NOE graph (`deg_obs` differs across roots)
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

fn enumerate_boltzmann_distribution(
    prob: &MethylContactProblem,
    k_assigned: usize,
    temp: f64,
) -> BTreeMap<Vec<i32>, f64> {
    fn rec(
        s: usize,
        assigned_so_far: usize,
        k_assigned: usize,
        prob: &MethylContactProblem,
        used_m: &mut [bool],
        cur: &mut Vec<i32>,
        states: &mut Vec<Vec<i32>>,
    ) {
        if s == prob.s_len {
            if assigned_so_far == k_assigned {
                states.push(cur.clone());
            }
            return;
        }
        let rem = prob.s_len - s;
        if assigned_so_far + rem < k_assigned {
            return;
        }
        if assigned_so_far < k_assigned {
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
        if assigned_so_far + (rem - 1) >= k_assigned {
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
    assert!(!states.is_empty(), "valid state space must be non-empty");

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

#[test]
fn p1_10_cpu_mcmc_exact_boltzmann_stationary_distribution_all_four_moves() {
    let temp = 2.5;

    // Case A (`S = 4 <= M = 5`): exercises Move 1B/1C, Move 2 asymmetric contact extension,
    // Move 3 4-body geminal involution, and Move 4 prochiral 2-flip involution.
    let prob_a = build_synthetic_mcmc_test_problem(4, 5);
    let exact_a = enumerate_boltzmann_distribution(&prob_a, 4, temp);
    let n_samples = 24_000usize;
    let samples_a = sample_methyl_mcmc_terminal_cpu(&prob_a, temp, 520, n_samples, 20261113);
    let mut counts_a: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
    for st in samples_a {
        assert!(
            exact_a.contains_key(&st),
            "sampled invalid state outside Omega_4: {st:?}"
        );
        *counts_a.entry(st).or_insert(0) += 1;
    }
    let tvd_a: f64 = 0.5
        * exact_a
            .iter()
            .map(|(st, &p_true)| {
                let p_emp = (*counts_a.get(st).unwrap_or(&0) as f64) / (n_samples as f64);
                (p_emp - p_true).abs()
            })
            .sum::<f64>();
    println!(
        "[P1-10 CPU MCMC Stationary Boltzmann S=4,M=5] |Omega|={} | TVD={tvd_a:.4}",
        exact_a.len()
    );
    assert!(
        tvd_a < 0.025,
        "CPU MCMC stationary Boltzmann TVD (S=4,M=5) {tvd_a:.4} >= 0.025 (detailed balance violation!)"
    );

    // Case B (`S = 5 > M = 4`): exercises Move 1E (`ms < 0, t >= 0` displace-occupied with unequal
    // `|slots[s]| != |slots[t]|` up to 2x ratio) alongside Moves 1C, 2, 3, 4.
    let prob_b = build_synthetic_mcmc_test_problem(5, 4);
    let exact_b = enumerate_boltzmann_distribution(&prob_b, 4, temp);
    let samples_b = sample_methyl_mcmc_terminal_cpu(&prob_b, temp, 520, n_samples, 20261114);
    let mut counts_b: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
    for st in samples_b {
        assert!(
            exact_b.contains_key(&st),
            "sampled invalid state outside Omega_4 (S=5,M=4): {st:?}"
        );
        *counts_b.entry(st).or_insert(0) += 1;
    }
    let tvd_b: f64 = 0.5
        * exact_b
            .iter()
            .map(|(st, &p_true)| {
                let p_emp = (*counts_b.get(st).unwrap_or(&0) as f64) / (n_samples as f64);
                (p_emp - p_true).abs()
            })
            .sum::<f64>();
    println!(
        "[P1-10 CPU MCMC Stationary Boltzmann S=5,M=4] |Omega|={} | TVD={tvd_b:.4}",
        exact_b.len()
    );
    assert!(
        tvd_b < 0.025,
        "CPU MCMC stationary Boltzmann TVD (S=5,M=4) {tvd_b:.4} >= 0.025 (Move 1E Hastings violation!)"
    );
}

#[test]
fn p1_10_multichain_pdb_and_cn_homooligomer_isometry_invariants() {
    let root = repo_root();
    let ubq_mono = extract_methyl_structure_from_pdb_file(
        root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"),
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    assert!(ubq_mono.sites.len() >= 42);

    let c2 = build_c2_homodimer_methyl_model(&ubq_mono, 5.2);
    let m0 = ubq_mono.sites.len();
    assert_eq!(c2.sites.len(), 2 * m0);
    for a in 0..m0 {
        for b in (a + 1)..m0 {
            let d_aa = c2.dist_cc(a, b);
            let d_bb = c2.dist_cc(a + m0, b + m0);
            let d_ab = c2.dist_cc(a, b + m0);
            let d_ba = c2.dist_cc(a + m0, b);
            assert!((d_aa - d_bb).abs() < 1e-9);
            assert!((d_ab - d_ba).abs() < 1e-9);
        }
    }

    let c3 = build_cn_homooligomer_methyl_model(&ubq_mono, 3, 5.5);
    assert_eq!(c3.sites.len(), 3 * m0);

    // Multi-chain PDB parser preserves both chains A and B with identical res_seq
    let two_chain_pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C
ATOM      4  CB  ALA A   1       1.988  -0.760  -1.216  1.00 10.00           C
TER
ATOM      5  N   ALA B   1      10.000   0.000   0.000  1.00 10.00           N
ATOM      6  CA  ALA B   1      11.458   0.000   0.000  1.00 10.00           C
ATOM      7  C   ALA B   1      12.009   1.420   0.000  1.00 10.00           C
ATOM      8  CB  ALA B   1      11.988  -0.760  -1.216  1.00 10.00           C
END
";
    let mc = extract_multichain_methyl_structure_from_pdb_str(
        two_chain_pdb,
        &MethylLabelingScheme::Ilvmat,
    )
    .unwrap();
    assert_eq!(mc.sites.len(), 2);
    assert!((mc.dist_cc(0, 1) - 10.0).abs() < 1e-6);
}
