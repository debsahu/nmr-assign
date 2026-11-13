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

use std::collections::BTreeSet;
use std::path::PathBuf;

use nmr_core::methyl_solver::{
    assign_methyl_cpu, build_methyl_contact_problem, evaluate_methyl_assignment,
    MethylAnnealConfig, MethylScoreConfig,
};
use nmr_core::methyl_trosy::{
    attach_hmcmcbca_records, classify_methyl_peak, extract_hmcmcbca_from_reference,
    extract_methyl_structure_from_pdb_file, extract_true_site_shifts, link_hmcmcbca_geminal_pairs,
    pick_methyl_hmqc_2d, pick_methyl_noesy_3d, pick_methyl_noesy_4d, simulate_methyl_hmqc_2d,
    simulate_methyl_noesy_3d, simulate_methyl_noesy_4d, HmcmcbcaRecord, MethylLabelingScheme,
    MethylNoePeak, MethylPickerConfig, MethylResidueClass, MethylRootPeak, MethylSimConfig,
    MethylStructureModel,
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
