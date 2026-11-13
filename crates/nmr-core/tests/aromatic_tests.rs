use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::aromatic::{
    assign_aromatic_rings, classify_his_state, classify_his_tautomer, cluster_aromatic_rings,
    estimate_unlabelled_yamazaki_cb_offset, evaluate_aromatic_assignment, hungarian_max_weight,
    merge_aromatic_assignments_preserving_existing, pick_all_aromatic_spectra, pick_aromatic_2d,
    pick_aromatic_hcch_3d, simulate_aromatic_peaks, simulate_aromatic_spectra,
    synthesize_aromatic_reference, AromaticExperiment, AromaticMetrics, AromaticPickerConfig,
    AromaticRingKind, AromaticSimConfig, AromaticSolverConfig, HisStateClassification, HisTautomer,
    HisTautomerEvidence, AROMATIC_EXPERIMENTS,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms, Reference};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::spectrum::{AxisPpm, Spectrum2D};
use nmr_core::topology::{compute_atom_bmrb_zscore, ResidueVariant, SidechainPriors};

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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
    let mut base_ref = Reference::default();
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
            ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
    };
    let bb_with_hb = Reference {
        sequence: seq,
        seq_ids: vec![4, 25, 45],
        shifts,
        ..Default::default()
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
        ..Default::default()
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

#[test]
fn test_p1_11_pelton_1993_hmqc_coupling_intensities_and_quaternary_cg() {
    // 1. Verify simulate_aromatic_peaks obeys Pelton et al. (Protein Sci. 1993, 2, 543-558, Table 1):
    //    - Ne2H (tau, =ND1 unprotonated ~249.5, >NE2-H protonated ~169.5):
    //      3J(HD2, ND1) ~ 8.8 Hz > 2J(HD2, NE2H) ~ 5.2 Hz  => I(HD2, ND1) > I(HD2, NE2)
    //      2J(HE1, ND1) ~ 10.6 Hz > 2J(HE1, NE2H) ~ 5.2 Hz => I(HE1, ND1) > I(HE1, NE2)
    //    - Nd1H (pi, >ND1-H protonated ~170.0, =NE2 unprotonated ~249.5):
    //      2J(HD2, NE2) ~ 9.8 Hz >> 3J(HD2, ND1H) ~ 1.4 Hz => I(HD2, NE2) > 2 * I(HD2, ND1)
    //      2J(HE1, NE2) ~ 10.6 Hz > 2J(HE1, ND1H) ~ 5.2 Hz => I(HE1, NE2) > I(HE1, ND1)
    //    - DoublyProtonated (+1, both >NH protonated):
    //      2J(HD2, NE2H) ~ 5.4 Hz > 3J(HD2, ND1H) ~ 1.4 Hz => I(HD2, NE2) > 1.5 * I(HD2, ND1)
    let priors = SidechainPriors::empirical_bmrb();
    for (idx, st) in [
        HisTautomer::Ne2H,
        HisTautomer::Nd1H,
        HisTautomer::DoublyProtonated,
    ]
    .into_iter()
    .enumerate()
    {
        let mut base = Reference {
            sequence: vec!["HIS".to_string()],
            seq_ids: vec![1],
            ..Default::default()
        };
        base.shifts.insert((0, "CB".to_string()), 29.4);
        let mut tmap = BTreeMap::new();
        tmap.insert(0, st);
        let full = synthesize_aromatic_reference(&base, &priors, &tmap, 500 + idx as u64);
        let sim_cfg = AromaticSimConfig {
            p_missing_peak: 0.0,
            jitter_h_ppm: 0.0,
            jitter_c_ppm: 0.0,
            jitter_n_ppm: 0.0,
            seed: 900 + idx as u64,
            ..AromaticSimConfig::default()
        };
        let (peaks, _) = simulate_aromatic_peaks(&full, &tmap, &sim_cfg);
        let nd1 = full.shifts[&(0, "ND1".to_string())];
        let ne2 = full.shifts[&(0, "NE2".to_string())];
        let hd2 = full.shifts[&(0, "HD2".to_string())];
        let he1 = full.shifts[&(0, "HE1".to_string())];

        let find_hmqc_ht = |h_t: f64, n_t: f64| -> f64 {
            peaks
                .iter()
                .find(|p| {
                    p.exp == "HIS_LR_N15HMQC"
                        && (p.h - h_t).abs() < 0.02
                        && (p.n - n_t).abs() < 0.25
                })
                .and_then(|p| p.height)
                .unwrap_or(0.0)
        };
        let ht_hd_nd1 = find_hmqc_ht(hd2, nd1);
        let ht_hd_ne2 = find_hmqc_ht(hd2, ne2);
        let ht_he_nd1 = find_hmqc_ht(he1, nd1);
        let ht_he_ne2 = find_hmqc_ht(he1, ne2);

        match st {
            HisTautomer::Ne2H => {
                assert!(
                    ht_hd_nd1 > 1.25 * ht_hd_ne2,
                    "Ne2H must have stronger 3J(HD2, =ND1) than 2J(HD2, >NE2H): {ht_hd_nd1:.1} vs {ht_hd_ne2:.1}"
                );
                assert!(
                    ht_he_nd1 > 1.25 * ht_he_ne2,
                    "Ne2H must have stronger 2J(HE1, =ND1) than 2J(HE1, >NE2H): {ht_he_nd1:.1} vs {ht_he_ne2:.1}"
                );
            }
            HisTautomer::Nd1H => {
                assert!(
                    ht_hd_ne2 > 2.2 * ht_hd_nd1,
                    "Nd1H must have 2J(HD2, =NE2) >> 3J(HD2, >ND1H): {ht_hd_ne2:.1} vs {ht_hd_nd1:.1}"
                );
                assert!(
                    ht_he_ne2 > 1.25 * ht_he_nd1,
                    "Nd1H must have stronger 2J(HE1, =NE2) than 2J(HE1, >ND1H): {ht_he_ne2:.1} vs {ht_he_nd1:.1}"
                );
            }
            HisTautomer::DoublyProtonated => {
                assert!(
                    ht_hd_ne2 > 1.6 * ht_hd_nd1,
                    "DoublyProtonated must have 2J(HD2, >NE2H) > 3J(HD2, >ND1H): {ht_hd_ne2:.1} vs {ht_hd_nd1:.1}"
                );
            }
        }
    }

    // 2. Verify solver uses Pelton HD2 coupling intensity (2J(HD2, NE2H) ~ 5.4 Hz vs 3J(HD2, ND1H) ~ 1.4 Hz)
    //    to orient closely spaced +1 DoublyProtonated 15N shifts (175.6 vs 175.6-delta -> 174.9 vs 176.1 ppm).
    let mut bb_his = Reference {
        sequence: vec!["HIS".to_string()],
        seq_ids: vec![10],
        ..Default::default()
    };
    bb_his.shifts.insert((0, "CB".to_string()), 28.9);

    for &(strong_n, weak_n) in &[(174.8_f64, 176.2_f64), (176.2_f64, 174.8_f64)] {
        let peaks = vec![
            Peak {
                exp: "AROM_C13HSQC".to_string(),
                h: 7.36,
                n: 0.0,
                c: Some(118.4),
                sign: 1,
                height: Some(110.0),
                volume: Some(500.0),
                h2: None,
            },
            Peak {
                exp: "AROM_C13HSQC".to_string(),
                h: 8.62,
                n: 0.0,
                c: Some(135.2),
                sign: 1,
                height: Some(110.0),
                volume: Some(500.0),
                h2: None,
            },
            Peak {
                exp: "HBCBCGCDHD".to_string(),
                h: 7.36,
                n: 0.0,
                c: Some(28.9),
                sign: 1,
                height: Some(95.0),
                volume: Some(420.0),
                h2: None,
            },
            Peak {
                exp: "HIS_LR_N15HMQC".to_string(),
                h: 8.62,
                n: strong_n,
                c: None,
                sign: 1,
                height: Some(82.0),
                volume: Some(360.0),
                h2: None,
            },
            Peak {
                exp: "HIS_LR_N15HMQC".to_string(),
                h: 8.62,
                n: weak_n,
                c: None,
                sign: 1,
                height: Some(82.0),
                volume: Some(360.0),
                h2: None,
            },
            // Stronger HD2 cross-peak identifies NE2 (2J ~ 5.4 Hz) over ND1 (3J ~ 1.4 Hz)
            Peak {
                exp: "HIS_LR_N15HMQC".to_string(),
                h: 7.36,
                n: strong_n,
                c: None,
                sign: 1,
                height: Some(76.0),
                volume: Some(340.0),
                h2: None,
            },
            Peak {
                exp: "HIS_LR_N15HMQC".to_string(),
                h: 7.36,
                n: weak_n,
                c: None,
                sign: 1,
                height: Some(18.0),
                volume: Some(80.0),
                h2: None,
            },
        ];
        let res = assign_aromatic_rings(
            &bb_his,
            None,
            &peaks,
            &priors,
            &AromaticSolverConfig::default(),
        );
        let pred_ne2 = res.reference.shifts[&(0, "NE2".to_string())];
        let pred_nd1 = res.reference.shifts[&(0, "ND1".to_string())];
        assert!(
            (pred_ne2 - strong_n).abs() < 0.05 && (pred_nd1 - weak_n).abs() < 0.05,
            "Pelton HD2 coupling intensity must map stronger cross-peak ({strong_n}) to NE2 and weaker ({weak_n}) to ND1; got ND1={pred_nd1}, NE2={pred_ne2}"
        );
        assert_eq!(
            res.rings[0].his_tautomer.map(|x| x.0),
            Some(HisTautomer::DoublyProtonated)
        );
    }

    // 3. Quaternary 13CG shift evidence (`~135.5` Ne2H vs `~127.2` Nd1H vs `~129.8` DoublyProtonated)
    let (st_ne2, conf_ne2) = classify_his_state(&HisTautomerEvidence {
        cg_ppm: Some(135.6),
        cd2_ppm: Some(118.1),
        he1_ppm: Some(7.78),
        cb_ppm: Some(30.4),
        ..Default::default()
    });
    assert_eq!(st_ne2, HisStateClassification::Resolved(HisTautomer::Ne2H));
    assert!(conf_ne2 >= 0.85, "got {conf_ne2}");

    let (st_nd1, conf_nd1) = classify_his_state(&HisTautomerEvidence {
        cg_ppm: Some(127.1),
        cd2_ppm: Some(125.1),
        he1_ppm: Some(7.80),
        cb_ppm: Some(28.0),
        ..Default::default()
    });
    assert_eq!(st_nd1, HisStateClassification::Resolved(HisTautomer::Nd1H));
    assert!(conf_nd1 >= 0.85, "got {conf_nd1}");

    let (st_dp, conf_dp) = classify_his_state(&HisTautomerEvidence {
        cg_ppm: Some(129.8),
        cd2_ppm: Some(118.4),
        he1_ppm: Some(8.60),
        cb_ppm: Some(28.8),
        ..Default::default()
    });
    assert_eq!(
        st_dp,
        HisStateClassification::Resolved(HisTautomer::DoublyProtonated)
    );
    assert!(conf_dp >= 0.85, "got {conf_dp}");
}

#[test]
fn test_p1_11_his_fast_exchange_and_zn_metal_coordinated_abstention() {
    let priors = SidechainPriors::empirical_bmrb();
    let mut bb = Reference {
        sequence: vec!["HIS".to_string()],
        seq_ids: vec![52],
        ..Default::default()
    };
    bb.shifts.insert((0, "CB".to_string()), 29.75);

    // Case A: Fast-exchange neutral tautomeric averaging (both 15N in 190..230 ppm coalesced ~208/214 ppm)
    let ev_fast = HisTautomerEvidence {
        nd1_ppm: Some(213.5),
        ne2_ppm: Some(207.0),
        cd2_ppm: Some(121.4),
        hd2_ppm: Some(7.08),
        ce1_ppm: Some(136.8),
        he1_ppm: Some(7.80),
        cb_ppm: Some(29.4),
        ..Default::default()
    };
    let (cls_fast, conf_fast) = classify_his_state(&ev_fast);
    assert_eq!(cls_fast, HisStateClassification::FastExchangeUncommitted);
    assert!(
        conf_fast <= 0.55,
        "fast-exchange confidence must be <= 0.55, got {conf_fast}"
    );

    let fast_peaks = vec![
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 7.08,
            n: 0.0,
            c: Some(121.4),
            sign: 1,
            height: Some(105.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 7.80,
            n: 0.0,
            c: Some(136.8),
            sign: 1,
            height: Some(105.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HBCBCGCDHD".to_string(),
            h: 7.08,
            n: 0.0,
            c: Some(29.75),
            sign: 1,
            height: Some(95.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 7.80,
            n: 213.5,
            c: None,
            sign: 1,
            height: Some(85.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 7.80,
            n: 207.0,
            c: None,
            sign: 1,
            height: Some(82.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 7.08,
            n: 213.5,
            c: None,
            sign: 1,
            height: Some(68.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 7.08,
            n: 207.0,
            c: None,
            sign: 1,
            height: Some(62.0),
            volume: None,
            h2: None,
        },
    ];
    let res_fast = assign_aromatic_rings(
        &bb,
        None,
        &fast_peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );
    assert!(
        res_fast.rings[0].his_tautomer.is_none(),
        "FastExchangeUncommitted must abstain from static his_tautomer call"
    );
    assert_eq!(
        res_fast.rings[0].his_state.map(|x| x.0),
        Some(HisStateClassification::FastExchangeUncommitted)
    );
    assert!(res_fast.rings[0].his_state.unwrap().1 <= 0.55);

    // Case B: Real Zn(II)-coordinated His52 from BMRB 16649 (ND1 = 200.88 ppm Zn-bound, NE2 = 178.27 ppm >NH)
    let zn_peaks = vec![
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 6.949,
            n: 0.0,
            c: Some(120.106),
            sign: 1,
            height: Some(108.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 8.246,
            n: 0.0,
            c: Some(137.789),
            sign: 1,
            height: Some(108.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HBCBCGCDHD".to_string(),
            h: 6.949,
            n: 0.0,
            c: Some(29.75),
            sign: 1,
            height: Some(95.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 8.246,
            n: 200.878,
            c: None,
            sign: 1,
            height: Some(98.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 8.246,
            n: 178.265,
            c: None,
            sign: 1,
            height: Some(62.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 6.949,
            n: 200.878,
            c: None,
            sign: 1,
            height: Some(84.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HIS_LR_N15HMQC".to_string(),
            h: 6.949,
            n: 178.265,
            c: None,
            sign: 1,
            height: Some(48.0),
            volume: None,
            h2: None,
        },
    ];
    let res_zn = assign_aromatic_rings(
        &bb,
        None,
        &zn_peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );
    assert!(
        res_zn.rings[0].his_tautomer.is_none(),
        "Zn(II)-coordinated His must NOT mis-commit to DoublyProtonated or static neutral tautomer"
    );
    assert_eq!(
        res_zn.rings[0].his_state.map(|x| x.0),
        Some(HisStateClassification::MetalCoordinatedCandidate)
    );
    assert!(
        res_zn.rings[0].his_state.unwrap().1 < 0.80,
        "metal-coordinated commitment confidence must be < 0.80"
    );
    assert!((res_zn.reference.shifts[&(0, "ND1".to_string())] - 200.878).abs() < 0.1);
    assert!((res_zn.reference.shifts[&(0, "NE2".to_string())] - 178.265).abs() < 0.1);

    // Case C: Upfield 175.5/172.8 ppm at basic pH 7.8 with neutral HE1 = 7.74 ppm (or explicit metal hint)
    let (cls_ph, conf_ph) = classify_his_state(&HisTautomerEvidence {
        nd1_ppm: Some(175.5),
        ne2_ppm: Some(172.8),
        cd2_ppm: Some(118.9),
        he1_ppm: Some(7.74),
        sample_ph: Some(7.8),
        ..Default::default()
    });
    assert_eq!(cls_ph, HisStateClassification::MetalCoordinatedCandidate);
    assert!(conf_ph < 0.80);

    let (cls_hint, conf_hint) = classify_his_state(&HisTautomerEvidence {
        nd1_ppm: Some(247.0),
        ne2_ppm: Some(170.0),
        cd2_ppm: Some(118.0),
        he1_ppm: Some(7.80),
        metal_coordinated_hint: true,
        ..Default::default()
    });
    assert_eq!(cls_hint, HisStateClassification::MetalCoordinatedCandidate);
    assert!(conf_hint < 0.80);
}

#[test]
fn test_p1_12_unified_his_variant_priors_between_topology_and_aromatic() {
    let bb_priors = Priors::bmrb();
    let sc_priors = SidechainPriors::empirical_bmrb();
    let pairs = [
        (HisTautomer::Ne2H, ResidueVariant::HisNe2H),
        (HisTautomer::Nd1H, ResidueVariant::HisNd1H),
        (
            HisTautomer::DoublyProtonated,
            ResidueVariant::HisDoublyProtonated,
        ),
    ];
    for (taut, var) in pairs {
        let (nd1, ne2, cd2, hd2, ce1, he1, cb) = taut.shift_priors();
        for (atom, expected) in [
            ("ND1", nd1),
            ("NE2", ne2),
            ("CD2", cd2),
            ("HD2", hd2),
            ("CE1", ce1),
            ("HE1", he1),
            ("CB", cb),
        ] {
            let (mu, sd, z) =
                compute_atom_bmrb_zscore("HIS", var, atom, expected.0, &bb_priors, &sc_priors)
                    .unwrap_or_else(|| panic!("missing unified topology prior for {var:?} {atom}"));
            assert_eq!(
                (mu, sd),
                expected,
                "divergent His prior on {var:?} {atom}: topology=({mu}, {sd}) vs aromatic={expected:?}"
            );
            assert!(
                z.abs() < 1e-12,
                "z-score at unified prior mean must be 0.0, got {z}"
            );
        }
    }
}

#[test]
fn test_p1_12_multi_candidate_boltzmann_ring_confidence_ties_and_unique() {
    let priors = SidechainPriors::empirical_bmrb();
    let empty_his = BTreeMap::new();

    // 1. Symmetric K-way ties (K = 3 and K = 4 identical-CB PHE residues):
    //    calibrated posterior confidence must satisfy <= 1 / K (<= 0.34 for K=3, <= 0.25 for K=4).
    for (k_way, max_conf) in [(3usize, 0.34_f64), (4usize, 0.25_f64)] {
        let mut bb = Reference {
            sequence: vec!["PHE".to_string(); k_way],
            seq_ids: (1..=k_way as i32).collect(),
            ..Default::default()
        };
        for i in 0..k_way {
            bb.shifts.insert((i, "CB".to_string()), 39.40);
        }
        let mut truth = bb.clone();
        for i in 0..k_way {
            let step = (i as f64) * 0.14;
            truth.shifts.insert((i, "CD1".to_string()), 131.6 + step);
            truth.shifts.insert((i, "CD2".to_string()), 131.6 + step);
            truth.shifts.insert((i, "HD1".to_string()), 7.12 + step);
            truth.shifts.insert((i, "HD2".to_string()), 7.12 + step);
            truth.shifts.insert((i, "CE1".to_string()), 130.8 + step);
            truth.shifts.insert((i, "CE2".to_string()), 130.8 + step);
            truth.shifts.insert((i, "HE1".to_string()), 7.30 + step);
            truth.shifts.insert((i, "HE2".to_string()), 7.30 + step);
        }
        let sim_cfg = AromaticSimConfig {
            p_missing_peak: 0.0,
            jitter_h_ppm: 0.0,
            jitter_c_ppm: 0.0,
            seed: 3000 + k_way as u64,
            ..AromaticSimConfig::default()
        };
        let (peaks, _) = simulate_aromatic_peaks(&truth, &empty_his, &sim_cfg);
        let res =
            assign_aromatic_rings(&bb, None, &peaks, &priors, &AromaticSolverConfig::default());
        assert_eq!(res.rings.len(), k_way);
        for r in &res.rings {
            assert!(
                r.ring_confidence <= max_conf,
                "{k_way}-way identical-CB PHE tie must have ring_confidence <= {max_conf:.3}, got {:.4}",
                r.ring_confidence
            );
        }
    }

    // 2. Well-separated 3-PHE system (CB = 36.8, 40.2, 43.6 ppm): unique match must achieve >= 0.95!
    let mut bb_unique = Reference {
        sequence: vec!["PHE".to_string(), "PHE".to_string(), "PHE".to_string()],
        seq_ids: vec![1, 2, 3],
        ..Default::default()
    };
    for (i, cb) in [(0, 36.8), (1, 40.2), (2, 43.6)] {
        bb_unique.shifts.insert((i, "CB".to_string()), cb);
    }
    let truth_u = synthesize_aromatic_reference(&bb_unique, &priors, &empty_his, 4242);
    let (peaks_u, _) = simulate_aromatic_peaks(
        &truth_u,
        &empty_his,
        &AromaticSimConfig {
            p_missing_peak: 0.0,
            jitter_h_ppm: 0.002,
            jitter_c_ppm: 0.015,
            seed: 4242,
            ..AromaticSimConfig::default()
        },
    );
    let res_u = assign_aromatic_rings(
        &bb_unique,
        None,
        &peaks_u,
        &priors,
        &AromaticSolverConfig::default(),
    );
    for r in &res_u.rings {
        assert!(
            r.ring_confidence >= 0.95,
            "unique well-separated PHE ring must achieve ring_confidence >= 0.95, got {:.4}",
            r.ring_confidence
        );
    }
}

#[test]
fn test_p1_12_merge_preserving_existing_shifts_and_ambiguity_codes() {
    let priors = SidechainPriors::empirical_bmrb();
    let mut existing = Reference {
        sequence: vec!["PHE".to_string(), "PHE".to_string()],
        seq_ids: vec![10, 20],
        ..Default::default()
    };
    existing.shifts.insert((0, "CB".to_string()), 38.2);
    existing.shifts.insert((1, "CB".to_string()), 42.6);

    // Pre-existing user/curated aromatic shifts on PHE #0 (whose experimental peaks dropped out)
    existing.shifts.insert((0, "CD1".to_string()), 131.40);
    existing.shifts.insert((0, "CD2".to_string()), 131.40);
    existing.shifts.insert((0, "HD1".to_string()), 7.11);
    existing.shifts.insert((0, "HD2".to_string()), 7.11);
    existing.ambiguity_codes.insert((0, "CD1".to_string()), 1);
    existing.ambiguity_codes.insert((0, "HD1".to_string()), 1);

    // Pre-existing user CZ/HZ + stereospecific HB2/HB3 (ambiguity_code = 1) on PHE #1
    existing.shifts.insert((1, "HB2".to_string()), 2.94);
    existing.shifts.insert((1, "HB3".to_string()), 3.19);
    existing.ambiguity_codes.insert((1, "HB2".to_string()), 1);
    existing.ambiguity_codes.insert((1, "HB3".to_string()), 1);
    existing.shifts.insert((1, "CZ".to_string()), 129.45);
    existing.shifts.insert((1, "HZ".to_string()), 7.28);
    existing.ambiguity_codes.insert((1, "CZ".to_string()), 1);
    existing.ambiguity_codes.insert((1, "HZ".to_string()), 1);

    // Experimental peaks observe ONLY CD*/HD* and CE*/HE* of PHE #1 (CB = 42.6 ppm; no CZ/HZ peak, no PHE #0 peaks)
    let peaks = vec![
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 7.34,
            n: 0.0,
            c: Some(132.20),
            sign: 1,
            height: Some(210.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "AROM_C13HSQC".to_string(),
            h: 7.51,
            n: 0.0,
            c: Some(131.85),
            sign: 1,
            height: Some(205.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HBCBCGCDHD".to_string(),
            h: 7.34,
            n: 0.0,
            c: Some(42.6),
            sign: 1,
            height: Some(170.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HBCBCGCDCEHE".to_string(),
            h: 7.51,
            n: 0.0,
            c: Some(42.6),
            sign: 1,
            height: Some(150.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "AROM_HCCH_TOCSY".to_string(),
            h: 7.34,
            n: 0.0,
            c: Some(132.20),
            sign: 1,
            height: Some(140.0),
            volume: None,
            h2: Some(7.51),
        },
        Peak {
            exp: "AROM_HCCH_TOCSY".to_string(),
            h: 7.51,
            n: 0.0,
            c: Some(131.85),
            sign: 1,
            height: Some(140.0),
            volume: None,
            h2: Some(7.34),
        },
    ];

    let res = assign_aromatic_rings(
        &existing,
        None,
        &peaks,
        &priors,
        &AromaticSolverConfig::default(),
    );

    // Unassigned PHE #0 shifts and ambiguity codes must be untouched
    assert_eq!(
        res.reference.shifts.get(&(0, "CD1".to_string())),
        Some(&131.40)
    );
    assert_eq!(
        res.reference.shifts.get(&(0, "HD1".to_string())),
        Some(&7.11)
    );
    assert_eq!(
        res.reference.ambiguity_codes.get(&(0, "CD1".to_string())),
        Some(&1)
    );
    assert!(!res.atom_confidence.contains_key(&(0, "CD1".to_string())));

    // PHE #1 stereospecific HB2/HB3 and pre-existing CZ/HZ must be preserved with ambiguity_code = 1
    assert_eq!(
        res.reference.shifts.get(&(1, "HB2".to_string())),
        Some(&2.94)
    );
    assert_eq!(
        res.reference.ambiguity_codes.get(&(1, "HB2".to_string())),
        Some(&1)
    );
    assert_eq!(
        res.reference.ambiguity_codes.get(&(1, "HB3".to_string())),
        Some(&1)
    );
    assert_eq!(
        res.reference.shifts.get(&(1, "CZ".to_string())),
        Some(&129.45)
    );
    assert_eq!(
        res.reference.shifts.get(&(1, "HZ".to_string())),
        Some(&7.28)
    );
    assert_eq!(
        res.reference.ambiguity_codes.get(&(1, "CZ".to_string())),
        Some(&1)
    );
    assert!(!res.atom_confidence.contains_key(&(1, "HB2".to_string())));
    assert!(!res.atom_confidence.contains_key(&(1, "CZ".to_string())));

    // Newly assigned CD1/CE1 on PHE #1 are written with fast-flip ambiguity_code = 3 and recorded in atom_confidence
    assert!((res.reference.shifts[&(1, "CD1".to_string())] - 132.20).abs() < 0.05);
    assert!((res.reference.shifts[&(1, "CE1".to_string())] - 131.85).abs() < 0.05);
    assert_eq!(
        res.reference.ambiguity_codes.get(&(1, "CD1".to_string())),
        Some(&3)
    );
    assert!(res.atom_confidence.contains_key(&(1, "CD1".to_string())));

    // Also test direct merge_aromatic_assignments_preserving_existing helper explicitly
    let merged = merge_aromatic_assignments_preserving_existing(
        &existing,
        &res.reference,
        &res.atom_confidence,
    );
    assert_eq!(merged.shifts, res.reference.shifts);
    assert_eq!(merged.ambiguity_codes, res.reference.ambiguity_codes);
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        let u1 = self.f64().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.f64();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        mean + sd * z
    }
}

/// Build experimental aromatic peak lists directly from REAL deposited BMRB shifts (`deposited_all`)
/// WITHOUT calling `synthesize_aromatic_reference` or filling missing atoms from priors, then perturb
/// with systematic `13CB` referencing offset (`cb_offset_ppm` on Yamazaki `HBCBCGCDHD`/`HBCBCGCDCEHE`),
/// `p_dropout` random peak dropout, and `noise_frac` uniform spurious noise peaks.
fn build_real_bmrb_aromatic_peaks_with_drift_dropout_and_noise(
    deposited_all: &Reference,
    cb_offset_ppm: f64,
    p_dropout: f64,
    noise_frac: f64,
    seed: u64,
) -> Vec<Peak> {
    let mut rng = SplitMix64(seed);
    let jitter_h = 0.005;
    let jitter_c = 0.035;
    let jitter_n = 0.08;

    let mut true_peaks: Vec<Peak> = Vec::new();
    let mut has_his_hmqc = false;
    let mut has_trp_ind = false;

    for (k, res) in deposited_all.sequence.iter().enumerate() {
        if !matches!(res.as_str(), "PHE" | "TYR" | "TRP" | "HIS") {
            continue;
        }
        let get = |a: &str| deposited_all.shifts.get(&(k, a.to_string())).copied();
        let get_sym = |a1: &str, a2: &str| get(a1).or_else(|| get(a2));

        // Collect real deposited (carbon_slot, c_ppm, h_ppm, equiv_protons)
        let mut chs: Vec<(&str, f64, f64, usize)> = Vec::new();
        match res.as_str() {
            "PHE" => {
                if let (Some(c), Some(h)) = (get_sym("CD1", "CD2"), get_sym("HD1", "HD2")) {
                    chs.push(("CD1", c, h, 2));
                }
                if let (Some(c), Some(h)) = (get_sym("CE1", "CE2"), get_sym("HE1", "HE2")) {
                    chs.push(("CE1", c, h, 2));
                }
                if let (Some(c), Some(h)) = (get("CZ"), get("HZ")) {
                    chs.push(("CZ", c, h, 1));
                }
            }
            "TYR" => {
                if let (Some(c), Some(h)) = (get_sym("CD1", "CD2"), get_sym("HD1", "HD2")) {
                    chs.push(("CD1", c, h, 2));
                }
                if let (Some(c), Some(h)) = (get_sym("CE1", "CE2"), get_sym("HE1", "HE2")) {
                    chs.push(("CE1", c, h, 2));
                }
            }
            "TRP" => {
                for &(ca, ha) in &[
                    ("CD1", "HD1"),
                    ("CE3", "HE3"),
                    ("CZ3", "HZ3"),
                    ("CH2", "HH2"),
                    ("CZ2", "HZ2"),
                ] {
                    if let (Some(c), Some(h)) = (get(ca), get(ha)) {
                        chs.push((ca, c, h, 1));
                    }
                }
            }
            "HIS" => {
                for &(ca, ha) in &[("CD2", "HD2"), ("CE1", "HE1")] {
                    if let (Some(c), Some(h)) = (get(ca), get(ha)) {
                        chs.push((ca, c, h, 1));
                    }
                }
            }
            _ => {}
        }

        for &(_, c_v, h_v, eq) in &chs {
            let mult = if eq == 2 { 1.95 } else { 1.0 };
            let ht = 110.0 * mult * rng.normal(0.0, 0.08).exp();
            true_peaks.push(Peak {
                exp: "AROM_C13HSQC".to_string(),
                h: h_v + rng.normal(0.0, jitter_h),
                n: 0.0,
                c: Some(c_v + rng.normal(0.0, jitter_c)),
                sign: 1,
                height: Some(ht),
                volume: Some(ht * 4.8),
                h2: None,
            });
        }

        // Yamazaki HBCBCGCDHD and HBCBCGCDCEHE with systematic +cb_offset_ppm drift!
        if let Some(cb) = get("CB") {
            let hd_opt = match res.as_str() {
                "PHE" | "TYR" => get_sym("HD1", "HD2"),
                "TRP" => get("HD1"),
                "HIS" => get("HD2"),
                _ => None,
            };
            if let Some(hd) = hd_opt {
                let mult = if matches!(res.as_str(), "PHE" | "TYR") {
                    1.85
                } else {
                    1.0
                };
                let ht = 95.0 * mult * rng.normal(0.0, 0.10).exp();
                true_peaks.push(Peak {
                    exp: "HBCBCGCDHD".to_string(),
                    h: hd + rng.normal(0.0, jitter_h),
                    n: 0.0,
                    c: Some(cb + cb_offset_ppm + rng.normal(0.0, jitter_c)),
                    sign: 1,
                    height: Some(ht),
                    volume: Some(ht * 4.5),
                    h2: None,
                });
            }
            let he_opt = match res.as_str() {
                "PHE" | "TYR" => get_sym("HE1", "HE2"),
                "TRP" => get("HE3"),
                _ => None,
            };
            if let Some(he) = he_opt {
                let mult = if matches!(res.as_str(), "PHE" | "TYR") {
                    1.70
                } else {
                    0.80
                };
                let ht = 90.0 * mult * rng.normal(0.0, 0.10).exp();
                true_peaks.push(Peak {
                    exp: "HBCBCGCDCEHE".to_string(),
                    h: he + rng.normal(0.0, jitter_h),
                    n: 0.0,
                    c: Some(cb + cb_offset_ppm + rng.normal(0.0, jitter_c)),
                    sign: 1,
                    height: Some(ht),
                    volume: Some(ht * 4.5),
                    h2: None,
                });
            }
        }

        // 3D AROM_HCCH_TOCSY intra-ring relays between real deposited (C, H) spins
        let steps_fn = |ca: &str, cb_s: &str| -> Option<u8> {
            if ca == cb_s {
                return Some(0);
            }
            match res.as_str() {
                "PHE" => match (ca, cb_s) {
                    ("CD1", "CE1") | ("CE1", "CD1") | ("CE1", "CZ") | ("CZ", "CE1") => Some(1),
                    ("CD1", "CZ") | ("CZ", "CD1") => Some(2),
                    _ => None,
                },
                "TYR" => match (ca, cb_s) {
                    ("CD1", "CE1") | ("CE1", "CD1") => Some(1),
                    _ => None,
                },
                "TRP" => {
                    let pos = |x: &str| match x {
                        "CE3" => Some(0_i32),
                        "CZ3" => Some(1),
                        "CH2" => Some(2),
                        "CZ2" => Some(3),
                        _ => None,
                    };
                    Some((pos(ca)? - pos(cb_s)?).unsigned_abs() as u8)
                }
                _ => None,
            }
        };
        for &(ca, uc, uh, eq_u) in &chs {
            for &(cb_s, _, vh, eq_v) in &chs {
                let Some(st) = steps_fn(ca, cb_s) else {
                    continue;
                };
                let atten = match st {
                    0 => 1.25,
                    1 => 0.88,
                    2 => 0.58,
                    _ => 0.40,
                };
                let mult = ((eq_u * eq_v) as f64).sqrt();
                let ht = 85.0 * atten * mult * rng.normal(0.0, 0.10).exp();
                true_peaks.push(Peak {
                    exp: "AROM_HCCH_TOCSY".to_string(),
                    h: uh + rng.normal(0.0, jitter_h),
                    n: 0.0,
                    c: Some(uc + rng.normal(0.0, jitter_c)),
                    sign: 1,
                    height: Some(ht),
                    volume: Some(ht * 5.0),
                    h2: Some(vh + rng.normal(0.0, jitter_h)),
                });
            }
        }

        // HIS_LR_N15HMQC only if real ND1/NE2 shifts are deposited (e.g. BMRB 16649 Zn-His52)
        if res == "HIS" {
            if let (Some(nd1), Some(ne2)) = (get("ND1"), get("NE2")) {
                has_his_hmqc = true;
                // Unprotonated / metal-ligated downfield nitrogen has stronger J_NH than >NH
                let nd1_downfield = nd1 > ne2 + 8.0;
                if let Some(he1) = get("HE1") {
                    let (a_d1, a_e2) = if nd1_downfield {
                        (1.20, 0.65)
                    } else if ne2 > nd1 + 8.0 {
                        (0.65, 1.20)
                    } else {
                        (0.85, 0.85)
                    };
                    for &(nt, amp) in &[(nd1, a_d1), (ne2, a_e2)] {
                        true_peaks.push(Peak {
                            exp: "HIS_LR_N15HMQC".to_string(),
                            h: he1 + rng.normal(0.0, jitter_h),
                            n: nt + rng.normal(0.0, jitter_n),
                            c: None,
                            sign: 1,
                            height: Some(95.0 * amp),
                            volume: Some(420.0 * amp),
                            h2: None,
                        });
                    }
                }
                if let Some(hd2) = get("HD2") {
                    let (a_d1, a_e2) = if nd1_downfield {
                        (1.00, 0.55)
                    } else {
                        (0.24, 1.05)
                    };
                    for &(nt, amp) in &[(nd1, a_d1), (ne2, a_e2)] {
                        true_peaks.push(Peak {
                            exp: "HIS_LR_N15HMQC".to_string(),
                            h: hd2 + rng.normal(0.0, jitter_h),
                            n: nt + rng.normal(0.0, jitter_n),
                            c: None,
                            sign: 1,
                            height: Some(90.0 * amp),
                            volume: Some(400.0 * amp),
                            h2: None,
                        });
                    }
                }
            }
        }

        // TRP_IND_N15HSQC + NOESY15N relays only if real NE1 & HE1 are deposited (e.g. BMRB 16649 Trp6)
        if res == "TRP" {
            if let (Some(ne1), Some(he1)) = (get("NE1"), get("HE1")) {
                has_trp_ind = true;
                let j_he1 = he1 + rng.normal(0.0, jitter_h);
                let j_ne1 = ne1 + rng.normal(0.0, jitter_n);
                true_peaks.push(Peak {
                    exp: "TRP_IND_N15HSQC".to_string(),
                    h: j_he1,
                    n: j_ne1,
                    c: None,
                    sign: 1,
                    height: Some(115.0),
                    volume: Some(520.0),
                    h2: None,
                });
                for &(ha, amp) in &[("HD1", 1.0), ("HZ2", 0.85)] {
                    if let Some(th) = get(ha) {
                        true_peaks.push(Peak {
                            exp: "NOESY15N".to_string(),
                            h: j_he1,
                            n: j_ne1,
                            c: None,
                            sign: 1,
                            height: Some(80.0 * amp),
                            volume: Some(360.0 * amp),
                            h2: Some(th + rng.normal(0.0, jitter_h)),
                        });
                    }
                }
            }
        }
    }

    // Apply random 10% peak dropout across experimental peaks
    let mut observed: Vec<Peak> = true_peaks
        .into_iter()
        .filter(|_| rng.f64() >= p_dropout)
        .collect();

    // Inject 8-10% spurious noise peaks uniformly across active experiment spectral windows
    let n_noise = ((observed.len() as f64) * noise_frac).round().max(2.0) as usize;
    let mut active_exps = vec![
        AromaticExperiment::AromC13Hsqc,
        AromaticExperiment::Hbcbcgcdhd,
        AromaticExperiment::Hbcbcgcdcehe,
        AromaticExperiment::AromHcchTocsy,
    ];
    if has_his_hmqc {
        active_exps.push(AromaticExperiment::HisLrN15Hmqc);
    }
    if has_trp_ind {
        active_exps.push(AromaticExperiment::TrpIndN15Hsqc);
    }

    for i in 0..n_noise {
        let exp = active_exps[i % active_exps.len()];
        let ((h_lo, h_hi), (y_lo, y_hi)) = exp.spectral_window();
        let h = rng.uniform(h_lo + 0.1, h_hi - 0.1);
        let y = rng.uniform(y_lo + 0.5, y_hi - 0.5);
        let ht = rng.uniform(35.0, 95.0);
        let peak = match exp {
            AromaticExperiment::HisLrN15Hmqc | AromaticExperiment::TrpIndN15Hsqc => Peak {
                exp: exp.as_str().to_string(),
                h,
                n: y,
                c: None,
                sign: 1,
                height: Some(ht),
                volume: Some(ht * 4.0),
                h2: None,
            },
            AromaticExperiment::AromHcchTocsy => Peak {
                exp: exp.as_str().to_string(),
                h,
                n: 0.0,
                c: Some(y),
                sign: 1,
                height: Some(ht),
                volume: Some(ht * 4.2),
                h2: Some(rng.uniform(h_lo + 0.1, h_hi - 0.1)),
            },
            _ => Peak {
                exp: exp.as_str().to_string(),
                h,
                n: 0.0,
                c: Some(y),
                sign: 1,
                height: Some(ht),
                volume: Some(ht * 4.0),
                h2: None,
            },
        };
        observed.push(peak);
    }

    observed
}

#[test]
fn test_p0_2_non_circular_real_bmrb_aromatic_benchmark_cb_drift_dropout_and_noise() {
    // Non-circular real-data BMRB aromatic benchmark (P0-2):
    // - NEVER calls `synthesize_aromatic_reference`!
    // - Uses real deposited BMRB shifts from public CC0 all-atom entries:
    //   1. BMRB 17769 (PDB 1D3Z, human ubiquitin, 76 aa: Phe4, Phe45, Tyr59, His68)
    //   2. BMRB 4562  (PDB 1E8L, hen lysozyme, 129 aa: 3 Phe, 3 Tyr, 6 Trp, 1 His; ring-current Trp28 HZ3 = 6.32 ppm)
    //   3. BMRB 16649 (PDB 2KRU, C. tepidum CtR69A, 63 aa: Trp6 HE1=12.08/NE1=137.53, 3 Phe, Tyr32, Zn(II)-His52 ND1=200.88/NE2=178.27)
    // - Also verifies perdeuterated ILV-methyl MBP fixture (`mbp_bmr7114_methyl.str`) loads cleanly without false assignments.
    let root = repo_root();

    // Verify perdeuterated MBP fixture (`mbp_bmr7114_methyl.str`) has 0 complete (1H, 13C/15N) aromatic pairs and yields 0 false positives on pure noise
    {
        let mbp_path = root.join("data/fixtures/mbp_bmr7114_methyl.str");
        let mbp_bb = read_star(&mbp_path, None).expect("read MBP backbone");
        let mbp_all = read_star_all_atoms(&mbp_path, None).expect("read MBP all atoms");
        let mbp_peaks = build_real_bmrb_aromatic_peaks_with_drift_dropout_and_noise(
            &mbp_all, 0.22, 0.10, 0.09, 7114,
        );
        // Perdeuterated MBP has 0 protonated (1H, 13C) HSQC pairs (only partial HE*/CB entries + noise)
        assert!(!mbp_peaks.is_empty());
        let mbp_res = assign_aromatic_rings(
            &mbp_bb,
            None,
            &mbp_peaks,
            &SidechainPriors::empirical_bmrb(),
            &AromaticSolverConfig::default(),
        );
        assert_eq!(
            mbp_res.atom_confidence.len(),
            0,
            "perdeuterated MBP without (1H, 13C) HSQC cross-peaks must produce 0 false aromatic assignments"
        );
    }

    let fixtures = [
        (
            "bmr17769",
            "1D3Z",
            "data/fixtures/bmr17769_ubiquitin_allatom.str",
        ),
        (
            "bmr4562",
            "1E8L",
            "data/fixtures/bmr4562_lysozyme_aromatic.str",
        ),
        (
            "bmr16649",
            "2KRU",
            "data/fixtures/bmr16649_pcp_aromatic.str",
        ),
    ];
    let cb_offsets = [0.16_f64, -0.21, 0.24, -0.27, 0.28];
    let p_dropout = 0.10;
    let noise_frac = 0.09;

    let priors = SidechainPriors::empirical_bmrb();
    let solver_cfg = AromaticSolverConfig::default();

    let mut total_agg = AromaticMetrics::default();
    let mut ece_records: Vec<(f64, bool)> = Vec::new();
    let mut max_offset_err = 0.0_f64;
    let mut zn_his52_checked = 0usize;

    println!(
        "\n=== NON-CIRCULAR REAL BMRB AROMATIC BENCHMARK (5 seeds, |ΔCB|=0.16..0.28 ppm, 10% dropout, 9% noise) ==="
    );
    println!(
        "{:<10} {:<6} {:>9} {:>9} {:>10} {:>12}",
        "BMRB", "PDB", "Acc(%)", "Recall(%)", "HC_Prec(%)", "Mean|δCB_err|"
    );
    println!("{}", "-".repeat(64));

    for (f_idx, &(bmrb_id, pdb_id, rel_path)) in fixtures.iter().enumerate() {
        let path = root.join(rel_path);
        let backbone_ref = read_star(&path, None).expect("read backbone_ref");
        let deposited_all = read_star_all_atoms(&path, None).expect("read deposited_all");

        let mut prot_agg = AromaticMetrics::default();
        let mut offset_err_sum = 0.0_f64;

        for (s_idx, &cb_off) in cb_offsets.iter().enumerate() {
            let seed = 88_000_u64 + (f_idx as u64) * 1_009 + (s_idx as u64) * 131;
            let peaks = build_real_bmrb_aromatic_peaks_with_drift_dropout_and_noise(
                &deposited_all,
                cb_off,
                p_dropout,
                noise_frac,
                seed,
            );

            let est_hd = estimate_unlabelled_yamazaki_cb_offset(
                &backbone_ref,
                None,
                &peaks,
                Some("HBCBCGCDHD"),
                0.5,
            );
            let err = (est_hd - cb_off).abs();
            offset_err_sum += err;
            max_offset_err = max_offset_err.max(err);

            let assigned = assign_aromatic_rings(&backbone_ref, None, &peaks, &priors, &solver_cfg);
            let empty_his = BTreeMap::new();
            let pm = evaluate_aromatic_assignment(
                &assigned,
                &deposited_all,
                &empty_his,
                0.05,
                0.45,
                0.70,
            );
            prot_agg.accumulate(&pm);
            total_agg.accumulate(&pm);

            // Collect atom-level (confidence, is_correct) pairs on atoms deposited in BMRB for 10-bin ECE
            for (&(k, ref atom), &conf) in &assigned.atom_confidence {
                if atom == "CB" {
                    continue;
                }
                let Some(&truth_v) = deposited_all.shifts.get(&(k, atom.clone())) else {
                    continue;
                };
                let pred_v = assigned.reference.shifts[&(k, atom.clone())];
                let tol = if atom.starts_with('H') {
                    0.05
                } else if atom.starts_with('C') {
                    0.45
                } else {
                    0.70
                };
                ece_records.push((conf, (pred_v - truth_v).abs() <= tol));
            }

            // Verify real Zn(II)-coordinated His52 in BMRB 16649 abstains from static HisTautomer
            if bmrb_id == "bmr16649" {
                for r in &assigned.rings {
                    if r.seq_id == 52
                        && assigned
                            .reference
                            .shifts
                            .contains_key(&(r.residue_idx, "ND1".to_string()))
                        && assigned
                            .reference
                            .shifts
                            .contains_key(&(r.residue_idx, "NE2".to_string()))
                    {
                        assert!(
                            r.his_tautomer.is_none(),
                            "BMRB 16649 Zn(II)-His52 must abstain (`his_tautomer == None`), got {:?}",
                            r.his_tautomer
                        );
                        assert_eq!(
                            r.his_state.map(|x| x.0),
                            Some(HisStateClassification::MetalCoordinatedCandidate)
                        );
                        zn_his52_checked += 1;
                    }
                }
            }
        }

        println!(
            "{:<10} {:<6} {:>8.2}% {:>8.2}% {:>9.2}% {:>10.4} ppm  (assigned={}/{}, hc={}/{})",
            bmrb_id,
            pdb_id,
            prot_agg.accuracy() * 100.0,
            prot_agg.recall() * 100.0,
            prot_agg.high_confidence_precision() * 100.0,
            offset_err_sum / (cb_offsets.len() as f64),
            prot_agg.correct_atoms,
            prot_agg.assigned_atoms,
            prot_agg.hc_correct_atoms,
            prot_agg.hc_assigned_atoms,
        );
    }

    // Compute 10-bin Expected Calibration Error (ECE)
    let mut ece = 0.0_f64;
    let n_ece = ece_records.len();
    for b in 0..10usize {
        let lo = (b as f64) * 0.1;
        let hi = if b == 9 {
            1.000_001
        } else {
            (b + 1) as f64 * 0.1
        };
        let bin: Vec<(f64, bool)> = ece_records
            .iter()
            .copied()
            .filter(|&(c, _)| c >= lo && c < hi)
            .collect();
        if !bin.is_empty() {
            let mean_conf = bin.iter().map(|&(c, _)| c).sum::<f64>() / (bin.len() as f64);
            let emp_acc = bin.iter().filter(|&&(_, ok)| ok).count() as f64 / (bin.len() as f64);
            ece += (bin.len() as f64 / n_ece as f64) * (mean_conf - emp_acc).abs();
        }
    }

    println!("{}", "-".repeat(64));
    println!(
        "AGGREGATE (3 real BMRB proteins x 5 seeds): Acc = {:.2}% ({}/{}), Recall = {:.2}% ({}/{}), HC_Prec = {:.2}% ({}/{}), ECE = {:.4}, max|δCB_err| = {:.4} ppm, Zn-His52 checks = {}\n",
        total_agg.accuracy() * 100.0,
        total_agg.correct_atoms,
        total_agg.assigned_atoms,
        total_agg.recall() * 100.0,
        total_agg.correct_atoms,
        total_agg.assignable_atoms,
        total_agg.high_confidence_precision() * 100.0,
        total_agg.hc_correct_atoms,
        total_agg.hc_assigned_atoms,
        ece,
        max_offset_err,
        zn_his52_checked,
    );

    assert!(
        zn_his52_checked >= 3,
        "expected BMRB 16649 Zn(II)-His52 15N pair to be tested on >=3/5 dropout seeds, got {zn_his52_checked}"
    );
    assert!(
        max_offset_err <= 0.08,
        "expected unlabelled Yamazaki CB offset recovery error <= 0.08 ppm across all proteins/seeds, got {max_offset_err:.4}"
    );
    assert!(
        total_agg.accuracy() >= 0.88,
        "non-circular real-BMRB aromatic accuracy {:.2}% fell below 88.0% gate",
        total_agg.accuracy() * 100.0
    );
    assert!(
        total_agg.recall() >= 0.80,
        "non-circular real-BMRB aromatic recall {:.2}% fell below 80.0% gate",
        total_agg.recall() * 100.0
    );
    assert!(
        total_agg.high_confidence_precision() >= 0.92,
        "non-circular real-BMRB HC-precision {:.2}% fell below 92.0% gate",
        total_agg.high_confidence_precision() * 100.0
    );
    assert!(
        ece <= 0.12,
        "non-circular real-BMRB Expected Calibration Error {ece:.4} exceeded 0.12 gate"
    );
}
