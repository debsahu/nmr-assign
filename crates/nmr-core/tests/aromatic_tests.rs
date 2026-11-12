use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::aromatic::{
    assign_aromatic_rings, classify_his_tautomer, cluster_aromatic_rings,
    evaluate_aromatic_assignment, hungarian_max_weight, pick_all_aromatic_spectra,
    pick_aromatic_2d, pick_aromatic_hcch_3d, simulate_aromatic_peaks, simulate_aromatic_spectra,
    synthesize_aromatic_reference, AromaticExperiment, AromaticMetrics, AromaticPickerConfig,
    AromaticRingKind, AromaticSimConfig, AromaticSolverConfig, HisTautomer, HisTautomerEvidence,
    AROMATIC_EXPERIMENTS,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms, Reference};
use nmr_core::peaks::Peak;
use nmr_core::spectrum::{AxisPpm, Spectrum2D};
use nmr_core::topology::SidechainPriors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn test_aromatic_experiment_parsing_and_catalog() {
    assert_eq!(AROMATIC_EXPERIMENTS.len(), 6);
    assert_eq!(
        AromaticExperiment::parse_exp("(HB)CB(CGCD)HD"),
        Some(AromaticExperiment::Hbcbcgcdhd)
    );
    assert_eq!(
        AromaticExperiment::parse_exp("(HB)CB(CGCDCE)HE"),
        Some(AromaticExperiment::Hbcbcgcdcehe)
    );
    assert_eq!(
        AromaticExperiment::parse_exp("AROM-C13-HSQC"),
        Some(AromaticExperiment::AromC13Hsqc)
    );
    assert_eq!(
        AromaticExperiment::parse_exp("AROM_HCCH_TOCSY"),
        Some(AromaticExperiment::AromHcchTocsy)
    );
    assert_eq!(
        AromaticExperiment::parse_exp("HIS_LR_N15HMQC"),
        Some(AromaticExperiment::HisLrN15Hmqc)
    );
    assert_eq!(
        AromaticExperiment::parse_exp("TRP_IND_N15HSQC"),
        Some(AromaticExperiment::TrpIndN15Hsqc)
    );
}

#[test]
fn test_pelton_his_tautomer_canonical_and_jittered_regimes_100_percent() {
    // Canonical Pelton, Torchia, Meadow & Roseman (Protein Sci. 1993, 2, 543-558) shift regimes:
    // 1. Neutral Nε2-H (tau tautomer): ND1 ~ 245-255 ppm (unprotonated pyridine-like), NE2 ~ 165-175 ppm (pyrrole-like), CD2 ~ 117.8 ppm, CB ~ 30.2 ppm
    // 2. Neutral Nδ1-H (pi tautomer):  ND1 ~ 165-178 ppm (pyrrole-like), NE2 ~ 245-255 ppm (pyridine-like), CD2 ~ 125.4 ppm, CB ~ 28.1 ppm
    // 3. DoublyProtonated (+1 imidazolium): ND1 ~ 173-182 ppm, NE2 ~ 170-180 ppm, CD2 < 122 ppm (~118.2 ppm), HE1 ~ 8.58 ppm, CB ~ 29.0 ppm
    let canonical_cases = [
        (
            HisTautomer::Ne2H,
            HisTautomerEvidence {
                nd1_ppm: Some(249.5),
                ne2_ppm: Some(169.5),
                cd2_ppm: Some(117.8),
                hd2_ppm: Some(7.02),
                ce1_ppm: Some(137.4),
                he1_ppm: Some(7.82),
                cb_ppm: Some(30.2),
            },
        ),
        (
            HisTautomer::Ne2H,
            HisTautomerEvidence {
                nd1_ppm: Some(246.2),
                ne2_ppm: Some(173.8),
                cd2_ppm: Some(119.5),
                hd2_ppm: Some(7.15),
                ce1_ppm: Some(136.5),
                he1_ppm: Some(7.72),
                cb_ppm: Some(31.1),
            },
        ),
        (
            HisTautomer::Nd1H,
            HisTautomerEvidence {
                nd1_ppm: Some(171.0),
                ne2_ppm: Some(249.2),
                cd2_ppm: Some(125.4),
                hd2_ppm: Some(7.14),
                ce1_ppm: Some(137.1),
                he1_ppm: Some(7.76),
                cb_ppm: Some(28.1),
            },
        ),
        (
            HisTautomer::Nd1H,
            HisTautomerEvidence {
                nd1_ppm: Some(176.4),
                ne2_ppm: Some(246.0),
                cd2_ppm: Some(124.2),
                hd2_ppm: Some(7.22),
                ce1_ppm: Some(138.0),
                he1_ppm: Some(7.85),
                cb_ppm: Some(27.4),
            },
        ),
        (
            HisTautomer::DoublyProtonated,
            HisTautomerEvidence {
                nd1_ppm: Some(177.2),
                ne2_ppm: Some(174.4),
                cd2_ppm: Some(118.2),
                hd2_ppm: Some(7.34),
                ce1_ppm: Some(135.2),
                he1_ppm: Some(8.58),
                cb_ppm: Some(29.0),
            },
        ),
        (
            HisTautomer::DoublyProtonated,
            HisTautomerEvidence {
                nd1_ppm: Some(180.5),
                ne2_ppm: Some(171.8),
                cd2_ppm: Some(119.8),
                hd2_ppm: Some(7.42),
                ce1_ppm: Some(134.5),
                he1_ppm: Some(8.68),
                cb_ppm: Some(28.6),
            },
        ),
        // Backup classification when one 15N cross-peak is missing: CD2 < 122 ppm + HE1/CB diagnostic!
        (
            HisTautomer::DoublyProtonated,
            HisTautomerEvidence {
                nd1_ppm: None,
                ne2_ppm: Some(174.5),
                cd2_ppm: Some(118.0),
                hd2_ppm: Some(7.36),
                ce1_ppm: Some(135.0),
                he1_ppm: Some(8.62),
                cb_ppm: Some(28.9),
            },
        ),
        (
            HisTautomer::Nd1H,
            HisTautomerEvidence {
                nd1_ppm: Some(171.5),
                ne2_ppm: None,
                cd2_ppm: Some(125.9),
                hd2_ppm: Some(7.14),
                ce1_ppm: Some(137.2),
                he1_ppm: Some(7.74),
                cb_ppm: Some(27.9),
            },
        ),
    ];

    for (idx, (expected, ev)) in canonical_cases.iter().enumerate() {
        let (pred, conf) = classify_his_tautomer(ev);
        assert_eq!(
            pred, *expected,
            "case #{idx} misclassified: got {pred:?}, expected {expected:?}"
        );
        assert!(
            conf >= 0.85,
            "case #{idx} expected high confidence >= 0.85, got {conf:.4}"
        );
    }

    // Test 60 randomized synthetic draws across the 3 Pelton shift regimes (20 each)
    let mut truth_taut = BTreeMap::new();
    let mut base_ref = Reference {
        sequence: Vec::new(),
        seq_ids: Vec::new(),
        shifts: BTreeMap::new(),
    };
    let states = [
        HisTautomer::Ne2H,
        HisTautomer::Nd1H,
        HisTautomer::DoublyProtonated,
    ];
    for i in 0..60usize {
        base_ref.sequence.push("HIS".to_string());
        base_ref.seq_ids.push((i + 1) as i32);
        truth_taut.insert(i, states[i % 3]);
    }
    let priors = SidechainPriors::empirical_bmrb();
    let full = synthesize_aromatic_reference(&base_ref, &priors, &truth_taut, 20261112);

    let mut correct = 0usize;
    for i in 0..60usize {
        let ev = HisTautomerEvidence {
            nd1_ppm: full.shifts.get(&(i, "ND1".to_string())).copied(),
            ne2_ppm: full.shifts.get(&(i, "NE2".to_string())).copied(),
            cd2_ppm: full.shifts.get(&(i, "CD2".to_string())).copied(),
            hd2_ppm: full.shifts.get(&(i, "HD2".to_string())).copied(),
            ce1_ppm: full.shifts.get(&(i, "CE1".to_string())).copied(),
            he1_ppm: full.shifts.get(&(i, "HE1".to_string())).copied(),
            cb_ppm: full.shifts.get(&(i, "CB".to_string())).copied(),
        };
        let (pred, conf) = classify_his_tautomer(&ev);
        if pred == states[i % 3] {
            correct += 1;
        }
        assert!(
            conf >= 0.90,
            "expected Pelton posterior >= 0.90, got {conf}"
        );
    }
    assert_eq!(
        correct, 60,
        "expected 100% accuracy across all 60 Pelton draws"
    );
}

#[test]
fn test_pick_aromatic_2d_sidelobe_and_ridge_artifact_suppression() {
    // Construct a synthetic 2D AROM_C13HSQC plane containing one intense genuine cross-peak
    // surrounded by +4.5% FT truncation sinc sidelobes within +/-4 F1 bins and a 1.2% vertical t1 ridge.
    let nx = 128usize;
    let ny = 96usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 9.0, 5.8, 800.0);
    let axis_y = AxisPpm::from_bounds("13C", ny, 142.0, 108.0, 201.2);
    let mut data = vec![0.0f32; nx * ny];

    let ix0 = 64usize;
    let iy0 = 48usize;
    let amp = 2500.0f64;

    for iy in 0..ny {
        let dy = (iy as f64) - (iy0 as f64);
        for ix in 0..nx {
            let dx = (ix as f64) - (ix0 as f64);
            let main = amp * (-0.5 * (dx * dx / 1.8 + dy * dy / 1.8)).exp();
            // Secondary positive FT truncation sinc sidelobes 4 bins away in F1 (4.5% of parent height < sidelobe_ratio)
            let side_y =
                0.045 * amp * (-0.5 * (dx * dx / 1.8 + (dy.abs() - 4.0).powi(2) / 0.8)).exp();
            // Weak t1 noise ridge at ix == ix0 with a local wiggle at iy = 16 (1.2% of column max < 4% col_sinc_frac)
            let ridge = if ix == ix0 {
                30.0 * (-0.5 * ((iy as f64 - 16.0).powi(2) / 1.1)).exp()
            } else {
                0.0
            };
            // Deterministic pseudo-noise background (sigma ~ 1.0)
            let noise = (((ix * 37 + iy * 91) % 19) as f64 - 9.0) * 0.18;
            data[iy * nx + ix] = (main + side_y + ridge + noise) as f32;
        }
    }

    let spec = Spectrum2D {
        name: "Synthetic AROM_C13HSQC".to_string(),
        axis_x: axis_x.clone(),
        axis_y: axis_y.clone(),
        data,
    };

    let picked = pick_aromatic_2d(
        &spec,
        AromaticExperiment::AromC13Hsqc,
        &AromaticPickerConfig::default(),
    );
    assert_eq!(
        picked.len(),
        1,
        "sidelobes and t1-ridge wiggles must be suppressed; got {:?}",
        picked
    );
    assert!((picked[0].h - axis_x.index_to_ppm(ix0 as f64)).abs() < 0.01);
    assert!((picked[0].c.unwrap() - axis_y.index_to_ppm(iy0 as f64)).abs() < 0.08);
}

#[test]
fn test_volumetric_simulation_fast_flip_degeneracy_trp_and_his_tautomers() {
    // Mini peptide containing all 4 aromatic types + 3 distinct His tautomers:
    // ALA - PHE - TYR - TRP - HIS(Ne2H) - GLY - HIS(Nd1H) - ALA - HIS(DoublyProtonated)
    let seq = vec![
        "ALA", "PHE", "TYR", "TRP", "HIS", "GLY", "HIS", "ALA", "HIS",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    let seq_ids = (1..=9).collect::<Vec<i32>>();
    let mut bb_shifts = BTreeMap::new();
    for (k, cb) in [
        (0, 19.1),
        (1, 39.8),
        (2, 38.4),
        (3, 29.6),
        (4, 31.8),
        (6, 26.8),
        (7, 18.8),
        (8, 29.1),
    ] {
        bb_shifts.insert((k, "CB".to_string()), cb);
        bb_shifts.insert((k, "CA".to_string()), 56.0 + (k as f64) * 0.4);
        bb_shifts.insert((k, "H".to_string()), 8.0 + (k as f64) * 0.08);
        bb_shifts.insert((k, "N".to_string()), 118.0 + (k as f64) * 0.9);
    }

    let bb_ref = Reference {
        sequence: seq,
        seq_ids,
        shifts: bb_shifts,
    };
    let mut his_map = BTreeMap::new();
    his_map.insert(4, HisTautomer::Ne2H);
    his_map.insert(6, HisTautomer::Nd1H);
    his_map.insert(8, HisTautomer::DoublyProtonated);

    let priors = SidechainPriors::empirical_bmrb();
    let truth = synthesize_aromatic_reference(&bb_ref, &priors, &his_map, 777);

    // Verify fast-flip symmetric degeneracy on PHE (idx 1) and TYR (idx 2): CD1 == CD2, CE1 == CE2
    assert_eq!(
        truth.shifts.get(&(1, "CD1".to_string())),
        truth.shifts.get(&(1, "CD2".to_string()))
    );
    assert_eq!(
        truth.shifts.get(&(1, "HD1".to_string())),
        truth.shifts.get(&(1, "HD2".to_string()))
    );
    assert_eq!(
        truth.shifts.get(&(2, "CE1".to_string())),
        truth.shifts.get(&(2, "CE2".to_string()))
    );

    let sim_cfg = AromaticSimConfig {
        p_missing_peak: 0.0,
        seed: 777,
        ..AromaticSimConfig::default()
    };
    let spectra = simulate_aromatic_spectra(&truth, &his_map, &sim_cfg);
    assert!(spectra.arom_c13_hsqc.is_some());
    assert!(spectra.hbcbcgcdhd.is_some());
    assert!(spectra.hbcbcgcdcehe.is_some());
    assert!(spectra.arom_hcch_tocsy.is_some());
    assert!(spectra.his_lr_n15_hmqc.is_some());
    assert!(spectra.trp_ind_n15_hsqc.is_some());

    // Pick 3D AROM_HCCH_TOCSY directly and check relay peaks exist
    let tocsy_peaks = pick_aromatic_hcch_3d(
        spectra.arom_hcch_tocsy.as_ref().unwrap(),
        None,
        &AromaticPickerConfig::default(),
    );
    assert!(
        tocsy_peaks.len() >= 16,
        "expected rich 3D intra-ring TOCSY relays for PHE/TYR/TRP/HIS, got {}",
        tocsy_peaks.len()
    );

    let all_peaks = pick_all_aromatic_spectra(&spectra, &AromaticPickerConfig::default());
    let solver_cfg = AromaticSolverConfig::default();
    let clusters = cluster_aromatic_rings(&all_peaks, Some(&bb_ref), &priors, &solver_cfg);
    assert!(
        clusters.iter().any(|c| c.kind == AromaticRingKind::Phe),
        "missing PHE cluster"
    );
    assert!(
        clusters.iter().any(|c| c.kind == AromaticRingKind::Tyr),
        "missing TYR cluster"
    );
    assert!(
        clusters.iter().any(|c| c.kind == AromaticRingKind::Trp),
        "missing TRP cluster"
    );

    let res = assign_aromatic_rings(&bb_ref, None, &all_peaks, &priors, &solver_cfg);
    let metrics = evaluate_aromatic_assignment(&res, &truth, &his_map, 0.05, 0.45, 0.70);
    assert_eq!(
        metrics.accuracy(),
        1.0,
        "expected 100% accuracy on clean 6-ring peptide, got {:.3}",
        metrics.accuracy()
    );
    assert_eq!(
        metrics.his_tautomer_accuracy(),
        1.0,
        "expected 100% HisTautomer accuracy across Ne2H/Nd1H/DoublyProtonated"
    );
}

#[test]
fn test_hungarian_exact_optimality_and_degenerate_cb_confidence_calibration() {
    // 1. Verify Hungarian solver matches brute-force maximum over a tricky 4x5 weight matrix
    let w = vec![
        vec![12.4, 11.9, -4.0, 2.1, -2.5],
        vec![12.2, 8.1, 3.0, 1.5, -2.5],
        vec![-5.0, 4.2, 14.0, 13.8, -2.5],
        vec![1.0, 2.0, 13.9, 9.5, -2.5],
    ];
    let (assign, total) = hungarian_max_weight(&w);
    // Optimal assignment: row 0 -> col 1 (11.9), row 1 -> col 0 (12.2), row 2 -> col 3 (13.8), row 3 -> col 2 (13.9) = 51.8
    assert!((total - 51.8).abs() < 1e-9, "got total {total}");
    assert_eq!(assign, vec![Some(1), Some(0), Some(3), Some(2)]);

    // 2. Two PHE residues with IDENTICAL CB = 39.309 ppm (like urea-unfolded ubiquitin bmr4375 PHE 4 & PHE 45):
    //    - Without HB2/HB3 NOEs: leave-one-out margin ΔS == 0 -> calibrated ring_confidence MUST be <= 0.55 (< 0.80)
    //    - With HB2/HB3 -> HD* NOE cross-peaks: ΔS >> 0 -> calibrated ring_confidence MUST rise >= 0.85 with 100% swap resolution!
    let seq = vec!["PHE".to_string(), "GLY".to_string(), "PHE".to_string()];
    let mut shifts = BTreeMap::new();
    shifts.insert((0, "CB".to_string()), 39.309);
    shifts.insert((2, "CB".to_string()), 39.309);
    shifts.insert((0, "HB2".to_string()), 2.82);
    shifts.insert((2, "HB2".to_string()), 3.31);

    let bb_no_hb = Reference {
        sequence: seq.clone(),
        seq_ids: vec![4, 25, 45],
        shifts: {
            let mut m = BTreeMap::new();
            m.insert((0, "CB".to_string()), 39.309);
            m.insert((2, "CB".to_string()), 39.309);
            m
        },
    };
    let bb_with_hb = Reference {
        sequence: seq,
        seq_ids: vec![4, 25, 45],
        shifts,
    };

    let priors = SidechainPriors::empirical_bmrb();
    let empty_his = BTreeMap::new();
    let mut truth = synthesize_aromatic_reference(&bb_with_hb, &priors, &empty_his, 4375);
    // Ensure distinct HD1 shifts on the two PHE rings so HB2->HD1 NOEs discriminate cleanly
    truth.shifts.insert((0, "HD1".to_string()), 7.14);
    truth.shifts.insert((0, "HD2".to_string()), 7.14);
    truth.shifts.insert((2, "HD1".to_string()), 7.36);
    truth.shifts.insert((2, "HD2".to_string()), 7.36);

    let sim_cfg = AromaticSimConfig {
        p_missing_peak: 0.0,
        jitter_h_ppm: 0.002,
        jitter_c_ppm: 0.015,
        seed: 4375,
        ..AromaticSimConfig::default()
    };
    let (mut peaks, _) = simulate_aromatic_peaks(&truth, &empty_his, &sim_cfg);

    let res_ambig = assign_aromatic_rings(
        &bb_no_hb,
        None,
        &peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );
    for r in &res_ambig.rings {
        assert!(
            r.ring_confidence <= 0.52,
            "identical-CB PHE pair without HB NOEs must have calibrated conf <= 0.52, got {:.4}",
            r.ring_confidence
        );
    }

    // Add HB2 -> HD* NOE tie-breakers (`AROM_HB_NOE`)
    peaks.push(Peak {
        exp: "AROM_HB_NOE".to_string(),
        h: 7.14,
        n: 0.0,
        c: truth.shifts.get(&(0, "CD1".to_string())).copied(),
        h2: Some(2.82),
        height: Some(250.0),
        volume: None,
        sign: 1,
    });
    peaks.push(Peak {
        exp: "AROM_HB_NOE".to_string(),
        h: 7.36,
        n: 0.0,
        c: truth.shifts.get(&(2, "CD1".to_string())).copied(),
        h2: Some(3.31),
        height: Some(250.0),
        volume: None,
        sign: 1,
    });

    let res_disambig = assign_aromatic_rings(
        &bb_with_hb,
        None,
        &peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );
    let m = evaluate_aromatic_assignment(&res_disambig, &truth, &empty_his, 0.04, 0.35, 0.50);
    assert_eq!(
        m.accuracy(),
        1.0,
        "HB2 NOE tie-breakers must achieve 100% accuracy"
    );
    for r in &res_disambig.rings {
        assert!(
            r.ring_confidence >= 0.85,
            "HB2 NOE disambiguated ring confidence must exceed 0.85, got {:.4}",
            r.ring_confidence
        );
    }
}

#[test]
fn test_missing_backbone_cb_fallback_and_trp_indole_tiebreaker() {
    // Two TRP residues with identical CB = 28.90 ppm disambiguated via pre-assigned (NE1, HE1) on backbone_ref,
    // plus one HIS residue lacking backbone CB altogether (recovered from Yamazaki CB + BMRB CB prior).
    let seq = vec!["TRP", "ALA", "TRP", "HIS"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut truth_base = Reference {
        sequence: seq.clone(),
        seq_ids: vec![10, 11, 12, 13],
        shifts: BTreeMap::new(),
    };
    truth_base.shifts.insert((0, "CB".to_string()), 28.90);
    truth_base.shifts.insert((2, "CB".to_string()), 28.90);
    truth_base.shifts.insert((0, "NE1".to_string()), 127.8);
    truth_base.shifts.insert((0, "HE1".to_string()), 9.85);
    truth_base.shifts.insert((2, "NE1".to_string()), 131.2);
    truth_base.shifts.insert((2, "HE1".to_string()), 10.48);

    let mut his_map = BTreeMap::new();
    his_map.insert(3, HisTautomer::Nd1H);

    let priors = SidechainPriors::empirical_bmrb();
    let truth = synthesize_aromatic_reference(&truth_base, &priors, &his_map, 9191);

    // Backbone input omits HIS CB at residue 3 completely, but provides TRP NE1/HE1 anchors
    let mut bb_ref = truth_base.clone();
    bb_ref.shifts.remove(&(3, "CB".to_string()));

    let sim_cfg = AromaticSimConfig {
        p_missing_peak: 0.0,
        jitter_h_ppm: 0.002,
        jitter_c_ppm: 0.015,
        seed: 9191,
        ..AromaticSimConfig::default()
    };
    let (peaks, relays) = simulate_aromatic_peaks(&truth, &his_map, &sim_cfg);
    let mut all_peaks = peaks;
    all_peaks.extend(relays);

    let res = assign_aromatic_rings(
        &bb_ref,
        None,
        &all_peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );
    let m = evaluate_aromatic_assignment(&res, &truth, &his_map, 0.05, 0.45, 0.70);
    assert_eq!(
        m.accuracy(),
        1.0,
        "expected 100% accuracy on TRP NE1/HE1 + missing-CB HIS"
    );
    assert_eq!(m.his_tautomer_accuracy(), 1.0);
    // Verify missing CB on HIS (idx 3) was back-populated from Yamazaki CB!
    let recovered_cb = res.reference.shifts.get(&(3, "CB".to_string())).copied();
    assert!(
        recovered_cb.is_some_and(|v| (v - truth.shifts[&(3, "CB".to_string())]).abs() < 0.15),
        "expected Yamazaki CB back-population for HIS without backbone CB, got {recovered_cb:?}"
    );
}

#[test]
fn test_12_protein_bmrb_aromatic_benchmark_gates() {
    let root = repo_root();
    let manifest_path = root.join("data/benchmark/manifest.tsv");
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));

    let priors = SidechainPriors::empirical_bmrb();
    let picker_cfg = AromaticPickerConfig::default();
    let solver_cfg = AromaticSolverConfig::default();

    let his_states = [
        HisTautomer::Ne2H,
        HisTautomer::Nd1H,
        HisTautomer::DoublyProtonated,
    ];
    let mut global_his_counter = 0usize;

    let mut agg = AromaticMetrics::default();
    let mut n_proteins = 0usize;

    println!(
        "\n{:<10} {:<6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "BMRB", "PDB", "Arom", "Acc(%)", "Recall(%)", "HC_Prec(%)", "HisTaut(%)"
    );
    println!("{}", "-".repeat(68));

    for (line_idx, line) in manifest.lines().skip(1).enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(cols.len() >= 6, "malformed manifest row: {line}");
        let bmrb_id = cols[0];
        let pdb_id = cols[1];
        let rel_path = cols[5];
        let star_path = root.join(rel_path);

        // Backbone input (`H, N, CA, CB, C` only) vs full deposited STAR shifts (if any sidechain shifts exist)
        let backbone_ref = read_star(&star_path, None)
            .unwrap_or_else(|e| panic!("failed to read backbone {}: {e}", star_path.display()));
        let deposited_all = read_star_all_atoms(&star_path, None)
            .unwrap_or_else(|e| panic!("failed to read all-atom {}: {e}", star_path.display()));

        // Deterministically cycle all 3 HisTautomer states across HIS environments so Ne2H, Nd1H, and
        // DoublyProtonated are all exercised end-to-end through volumetric 2D HIS_LR_N15HMQC + AROM_C13HSQC
        // + Yamazaki picking, while HIS residues in the same protein with near-degenerate CB (< 0.08 ppm,
        // e.g. disordered IDP HIS pairs) share a consistent local protonation state.
        let mut his_tautomers: BTreeMap<usize, HisTautomer> = BTreeMap::new();
        let mut n_arom_res = 0usize;
        for (k, aa) in backbone_ref.sequence.iter().enumerate() {
            if matches!(aa.as_str(), "PHE" | "TYR" | "TRP" | "HIS") {
                n_arom_res += 1;
            }
            if aa == "HIS" {
                let cb_k = backbone_ref.shifts.get(&(k, "CB".to_string())).copied();
                let existing_same_cb = cb_k.and_then(|cb| {
                    his_tautomers.iter().find_map(|(&prev_k, &prev_st)| {
                        backbone_ref
                            .shifts
                            .get(&(prev_k, "CB".to_string()))
                            .filter(|&&prev_cb| (prev_cb - cb).abs() < 0.08)
                            .map(|_| prev_st)
                    })
                });
                let st = existing_same_cb.unwrap_or_else(|| {
                    let s = his_states[global_his_counter % 3];
                    global_his_counter += 1;
                    s
                });
                his_tautomers.insert(k, st);
            }
        }

        let seed = 10_200_u64 + (line_idx as u64) * 137;
        let truth_ref =
            synthesize_aromatic_reference(&deposited_all, &priors, &his_tautomers, seed);

        let sim_cfg = AromaticSimConfig {
            jitter_h_ppm: 0.005,
            jitter_c_ppm: 0.035,
            jitter_n_ppm: 0.08,
            p_missing_peak: 0.025,
            seed: seed + 1,
            ..AromaticSimConfig::default()
        };
        let spectra = simulate_aromatic_spectra(&truth_ref, &his_tautomers, &sim_cfg);
        let picked_peaks = pick_all_aromatic_spectra(&spectra, &picker_cfg);
        let assigned =
            assign_aromatic_rings(&backbone_ref, None, &picked_peaks, &priors, &solver_cfg);

        let pm =
            evaluate_aromatic_assignment(&assigned, &truth_ref, &his_tautomers, 0.05, 0.45, 0.70);
        println!(
            "{:<10} {:<6} {:>6} {:>9.2}% {:>9.2}% {:>9.2}% {:>9.2}%  (assigned={}/{}, hc={}/{}, his={}/{})",
            bmrb_id,
            pdb_id,
            n_arom_res,
            pm.accuracy() * 100.0,
            pm.recall() * 100.0,
            pm.high_confidence_precision() * 100.0,
            pm.his_tautomer_accuracy() * 100.0,
            pm.correct_atoms,
            pm.assigned_atoms,
            pm.hc_correct_atoms,
            pm.hc_assigned_atoms,
            pm.his_correct,
            pm.his_total,
        );

        agg.accumulate(&pm);
        n_proteins += 1;
    }

    assert_eq!(n_proteins, 12, "expected all 12 benchmark proteins");
    println!("{}", "-".repeat(68));
    println!(
        "AGGREGATE (12 BMRB proteins): Acc = {:.2}% ({}/{}), Recall = {:.2}% ({}/{}), HC_Prec = {:.2}% ({}/{}), HisTaut = {:.2}% ({}/{})\n",
        agg.accuracy() * 100.0,
        agg.correct_atoms,
        agg.assigned_atoms,
        agg.recall() * 100.0,
        agg.correct_atoms,
        agg.assignable_atoms,
        agg.high_confidence_precision() * 100.0,
        agg.hc_correct_atoms,
        agg.hc_assigned_atoms,
        agg.his_tautomer_accuracy() * 100.0,
        agg.his_correct,
        agg.his_classified,
    );

    assert!(
        agg.accuracy() >= 0.920,
        "12-protein aromatic accuracy {:.2}% fell below 92.0% gate",
        agg.accuracy() * 100.0
    );
    assert!(
        agg.high_confidence_precision() >= 0.960,
        "12-protein high-confidence (>=0.80) precision {:.2}% fell below 96.0% gate",
        agg.high_confidence_precision() * 100.0
    );
    assert_eq!(
        agg.his_tautomer_accuracy(),
        1.0,
        "HisTautomer classification accuracy {:.2}% must be 100.0%",
        agg.his_tautomer_accuracy() * 100.0
    );
    assert_eq!(
        agg.his_classified, agg.his_total,
        "all {} HIS residues across 12 proteins must be classified",
        agg.his_total
    );
}
