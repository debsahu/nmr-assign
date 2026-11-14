//! Unit and integration tests for per-dataset quality-control (`DatasetQcVerdict`,
//! `DatasetQcReport`, `evaluate_dataset_qc`, `apply_dataset_qc_to_assignments`),
//! per-protein reliability curves (`ProteinReliabilityCurve`), and detailed chemical-shift
//! reference offset diagnostics (`OffsetFitStatus`, `NucleusOffsetFit`, `DetailedReferenceOffsets`).
//!
//! Provenance: Clean-room verification of `nmr_core::qc` and `nmr_core::priors` (`P1-1`, `P1-3`,
//! `P1-7`, `P1-8`, `Q4`).

use std::path::PathBuf;

use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::priors::{
    fit_unlabelled_global_offsets, fit_unlabelled_global_offsets_detailed,
    DetailedReferenceOffsets, NucleusOffsetFit, OffsetFitStatus, Priors,
};
use nmr_core::qc::{
    apply_dataset_qc_to_assignments, apply_evo_graph_calibration_map,
    apply_standard_calibration_map, apply_standard_gpu_calibration_map, check_offset_consistency,
    compute_debiased_grid_ece, compute_grid_ece, compute_grid_ece_bernoulli_floor,
    eval_monotone_calibration_map, evaluate_dataset_qc, evaluate_seed_predictions_with_gate,
    fit_isotonic_monotone_knots, fit_isotonic_monotone_knots_beta, fit_protein_hc_gate_config,
    has_weak_likelihood_large_offset, is_unconverged_solution_margin,
    select_gpu_shrinkage_weight_nested_lopo, DatasetQcVerdict, ProteinHcTrainingSeed,
    ProteinReliabilityCurve, CANDIDATE_GPU_SHRINKAGE_WEIGHTS, CANONICAL_CALIBRATION_GRID_12,
    CANONICAL_CALIBRATION_LAPLACE_PRIOR, CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
    CANONICAL_PROTEIN_HC_GATE, EVO_GRAPH_CALIBRATION_KNOTS, QC_CROWDED_MIN_SOLUTION_MARGIN,
    QC_CROWDED_MULTIPLICITY_RATIO, QC_MIN_ASSIGNED_FOR_MARGIN_GATE, QC_MIN_SOLUTION_MARGIN,
    QC_PROTEIN_HC_CONFIDENCE_CAP, QC_STRONG_BELIEF_THRESHOLD, STANDARD_CALIBRATION_KNOTS,
    STANDARD_CPU_PRIOR_ALPHA, STANDARD_CPU_PRIOR_BETA, STANDARD_GPU_CALIBRATION_KNOTS,
    UNIFORM_CALIBRATION_GRID_10,
};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{AssignResult, ScoreConfig};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn detailed_reference_offsets_distinguish_fitted_not_significant_and_too_few_values() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");

    // Case 1: Shift CA & C by -2.77 ppm (TMS-like carbon referencing), leave H & N unshifted (DSS),
    // and retain only 2 CB observations so CB must report TooFewValues (`n<5(2)`).
    let mut pred = Reference {
        sequence: truth.sequence.clone(),
        seq_ids: truth.seq_ids.clone(),
        ..Default::default()
    };
    let mut cb_kept = 0usize;
    for (&(k, ref atom), &val) in &truth.shifts {
        match atom.as_str() {
            "CA" | "C" => {
                pred.shifts.insert((k, atom.clone()), val - 2.77);
            }
            "CB" => {
                if cb_kept < 2 {
                    pred.shifts.insert((k, atom.clone()), val - 2.77);
                    cb_kept += 1;
                }
            }
            "H" | "N" => {
                pred.shifts.insert((k, atom.clone()), val);
            }
            _ => {}
        }
    }

    let det = fit_unlabelled_global_offsets_detailed(&pred, &truth, None);
    assert_eq!(det.ca.status, OffsetFitStatus::Fitted);
    assert!(
        (det.ca.applied_ppm + 2.77).abs() < 0.08,
        "expected dCA ~ -2.77 ppm, got {:?}",
        det.ca
    );
    assert!(det.ca.log_likelihood_gain > 15.0);
    assert_eq!(det.c.status, OffsetFitStatus::Fitted);
    assert!((det.c.applied_ppm + 2.77).abs() < 0.08);

    assert_eq!(det.h.status, OffsetFitStatus::NotSignificant);
    assert_eq!(det.h.applied_ppm, 0.0);
    assert_eq!(det.n.status, OffsetFitStatus::NotSignificant);
    assert_eq!(det.n.applied_ppm, 0.0);

    assert_eq!(det.cb.status, OffsetFitStatus::TooFewValues);
    assert_eq!(det.cb.n_values, 2);
    assert_eq!(det.cb.applied_ppm, 0.0);

    // Compact formatting distinguishes all three states clearly
    let ca_tok = det.ca.format_compact("dCA");
    let n_tok = det.n.format_compact("dN");
    let cb_tok = det.cb.format_compact("dCB");
    assert!(
        ca_tok.starts_with("dCA=-2.7") && ca_tok.contains("ΔLL="),
        "unexpected fitted token: {ca_tok}"
    );
    assert!(
        n_tok.starts_with("dN=ns(ΔLL="),
        "unexpected not-significant token: {n_tok}"
    );
    assert_eq!(cb_tok, "dCB=n<5(2)");

    // Backwards compatibility with `fit_unlabelled_global_offsets`
    let compat = fit_unlabelled_global_offsets(&pred, &truth, None);
    assert_eq!(compat, det.to_shift_offsets());

    // Strict permutation invariance: cyclically permuting residue indices `k` produces identical fits
    let n_seq = pred.sequence.len().max(2);
    let mut permuted = Reference {
        sequence: pred.sequence.clone(),
        seq_ids: pred.seq_ids.clone(),
        ..Default::default()
    };
    for (&(k, ref atom), &val) in &pred.shifts {
        let k_perm = if k == 0 {
            0
        } else {
            1 + ((k + 17) % (n_seq - 1))
        };
        permuted.shifts.insert((k_perm, atom.clone()), val);
    }
    let det_perm = fit_unlabelled_global_offsets_detailed(&permuted, &truth, None);
    assert_eq!(
        det, det_perm,
        "unlabelled offset fitting must be strictly invariant to residue-index permutations"
    );
}

#[test]
fn check_offset_consistency_flags_divergent_aliphatic_offsets_and_allows_missing_cb() {
    let mk_fit =
        |status: OffsetFitStatus, applied: f64, cand: f64, gain: f64, n: usize| NucleusOffsetFit {
            status,
            applied_ppm: applied,
            candidate_ppm: cand,
            log_likelihood_gain: gain,
            n_values: n,
        };

    // CISD3-like case: CA & C shifted by ~-2.67 ppm (TMS), CB has TooFewValues (0 CB experiments) -> OK
    let cisd3_like = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.01, 0.4, 38),
        n: mk_fit(OffsetFitStatus::NotSignificant, 0.0, -0.05, 0.6, 38),
        ca: mk_fit(OffsetFitStatus::Fitted, -2.67, -2.67, 42.0, 38),
        cb: mk_fit(OffsetFitStatus::TooFewValues, 0.0, 0.0, 0.0, 0),
        c: mk_fit(OffsetFitStatus::Fitted, -2.66, -2.66, 39.5, 38),
    };
    let (ok_cisd3, issues_cisd3) = check_offset_consistency(&cisd3_like);
    assert!(
        ok_cisd3,
        "missing CB (`TooFewValues`) must not trigger aliphatic mismatch: {issues_cisd3:?}"
    );

    // P1-4 rule: `NotSignificant` means unknown / below deadband, NEVER evidence of `0.0 ppm`!
    // One channel Fitted and the other NotSignificant must NOT trigger an aliphatic mismatch.
    let ca_fitted_cb_ns = DetailedReferenceOffsets {
        cb: mk_fit(OffsetFitStatus::NotSignificant, 0.0, -0.10, 0.5, 45),
        ..cisd3_like
    };
    let (ok_a, issues_a) = check_offset_consistency(&ca_fitted_cb_ns);
    assert!(
        ok_a && issues_a.is_empty(),
        "Fitted vs NotSignificant must not trigger aliphatic mismatch under P1-4: {issues_a:?}"
    );

    let cb_fitted_ca_ns = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.0, 0.1, 60),
        n: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.0, 0.1, 60),
        ca: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.12, 0.7, 60),
        cb: mk_fit(OffsetFitStatus::Fitted, -2.72, -2.72, 24.0, 55),
        c: mk_fit(OffsetFitStatus::NotSignificant, 0.0, -0.05, 0.3, 60),
    };
    let (ok_b, issues_b) = check_offset_consistency(&cb_fitted_ca_ns);
    assert!(
        ok_b && issues_b.is_empty(),
        "NotSignificant vs Fitted must not trigger aliphatic mismatch under P1-4: {issues_b:?}"
    );

    // Coherent both-Fitted TMS offsets (-2.78 vs -2.74 ppm) -> OK
    let tms_both_ok = DetailedReferenceOffsets {
        cb: mk_fit(OffsetFitStatus::Fitted, -2.74, -2.74, 38.0, 52),
        ca: mk_fit(OffsetFitStatus::Fitted, -2.78, -2.78, 41.0, 55),
        ..cisd3_like
    };
    let (ok_tms, issues_tms) = check_offset_consistency(&tms_both_ok);
    assert!(ok_tms, "coherent TMS offsets must pass: {issues_tms:?}");

    // Genuine both-Fitted divergent aliphatic offsets (> max(3 * sigma_joint, 0.85) ppm) -> flagged!
    let both_divergent = DetailedReferenceOffsets {
        ca: mk_fit(OffsetFitStatus::Fitted, -2.80, -2.80, 40.0, 60),
        cb: mk_fit(OffsetFitStatus::Fitted, 2.20, 2.20, 35.0, 60),
        ..cisd3_like
    };
    let (ok_div, issues_div) = check_offset_consistency(&both_divergent);
    assert!(!ok_div && !issues_div.is_empty());

    // Suspect case C: extreme |dH| > 0.35 ppm or |dN| > 2.5 ppm
    let extreme_hn = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::Fitted, 0.48, 0.48, 18.0, 50),
        n: mk_fit(OffsetFitStatus::Fitted, -3.10, -3.10, 22.0, 50),
        ..cisd3_like
    };
    let (ok_c, issues_c) = check_offset_consistency(&extreme_hn);
    assert!(!ok_c && issues_c.len() >= 2);
}

#[test]
fn dataset_qc_pass_warn_abstain_and_multiday_lock_shift_behavior() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let score_cfg = ScoreConfig::default();

    let (clean_peaks, _truth_map) = simulate(&truth, &SimConfig::default());
    let (systems_clean, res_clean, rep_clean) = assign_evo_graph_from_peaks(
        &clean_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );
    assert_eq!(
        rep_clean.qc_report.verdict,
        DatasetQcVerdict::Pass,
        "clean synthetic Ubiquitin must earn PASS, got {:?}",
        rep_clean.qc_report
    );
    assert_eq!(rep_clean.qc_report.confidence_scale, 1.0);
    assert!(rep_clean.qc_report.already_applied);

    // Routine multi-day / multi-magnet acquisition: uniform (+0.025 H, -0.24 N) ppm carrier shift on HNCA
    // must NOT be penalized as drift (`#105`).
    let mut multiday_peaks = clean_peaks.clone();
    for p in &mut multiday_peaks {
        if p.exp == "HNCA" {
            p.h += 0.025;
            p.n -= 0.24;
        }
    }
    let (_sys_md, _res_md, rep_md) = assign_evo_graph_from_peaks(
        &multiday_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );
    assert_eq!(
        rep_md.qc_report.verdict,
        DatasetQcVerdict::Pass,
        "uniform multi-day per-experiment lock shift must still earn PASS: {:?}",
        rep_md.qc_report
    );

    // WARN case: slow-exchange minor-conformer crowding (`> 1.45x` amide HSQC multiplicity)
    let mut crowded_peaks = clean_peaks.clone();
    let base_hsqc: Vec<Peak> = clean_peaks
        .iter()
        .filter(|p| p.exp == "HSQC")
        .cloned()
        .collect();
    for (idx, mut p) in base_hsqc.into_iter().enumerate() {
        p.h += 0.09 + 0.002 * ((idx % 7) as f64);
        p.n += 0.85 + 0.03 * ((idx % 5) as f64);
        p.height = p.height.map(|h| 0.55 * h);
        // Give the minor conformer roots weak 3D support so 3D support stays above Abstain
        crowded_peaks.push(Peak {
            exp: "HNCA".to_string(),
            h: p.h,
            n: p.n,
            c: Some(54.0 + (idx % 9) as f64),
            sign: 1,
            height: Some(800.0),
            volume: Some(4000.0),
            h2: None,
        });
        crowded_peaks.push(p);
    }
    let qc_warn = evaluate_dataset_qc(
        Some(&crowded_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        None,
    );
    assert_eq!(
        qc_warn.verdict,
        DatasetQcVerdict::Warn,
        "2x minor-conformer HSQC multiplicity must trigger WARN, got {qc_warn:?}"
    );
    assert!(qc_warn.hsqc_multiplicity_ratio > 1.45);
    assert!(qc_warn.confidence_scale < 1.0 && qc_warn.confidence_scale >= 0.65);

    let mut damped = AssignResult {
        position: vec![1, 2, 3],
        confidence: vec![0.84, 0.95, 0.50],
        score: 42.0,
        run_scores: Vec::new(),
    };
    apply_dataset_qc_to_assignments(&mut damped, &qc_warn);
    assert!(
        damped.confidence[0] < 0.80,
        "borderline 0.84 confidence should be damped below 0.80 under WARN, got {}",
        damped.confidence[0]
    );
    let after_once = damped.confidence.clone();
    let mut marked_qc = qc_warn.clone();
    marked_qc.already_applied = true;
    apply_dataset_qc_to_assignments(&mut damped, &marked_qc);
    assert_eq!(
        damped.confidence, after_once,
        "`already_applied = true` must prevent double-scaling"
    );

    // ABSTAIN case 1: catastrophic 3D strip mismatch (< 25% supported HSQC roots)
    let mut broken_3d_peaks = clean_peaks.clone();
    for p in &mut broken_3d_peaks {
        if p.exp != "HSQC" {
            p.n += 8.5; // Completely misregistered 15N carrier on all 3D cubes
        }
    }
    let qc_abstain_3d = evaluate_dataset_qc(
        Some(&broken_3d_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        None,
    );
    assert_eq!(qc_abstain_3d.verdict, DatasetQcVerdict::Abstain);
    assert!(qc_abstain_3d.hsqc_3d_support_frac < 0.25);

    let mut abstained_res = AssignResult {
        position: vec![1, 2, 3, 4],
        confidence: vec![0.99, 0.91, 0.82, 0.60],
        score: 10.0,
        run_scores: Vec::new(),
    };
    apply_dataset_qc_to_assignments(&mut abstained_res, &qc_abstain_3d);
    assert!(
        abstained_res.confidence.iter().all(|&c| c <= 0.45),
        "ABSTAIN must cap every confidence at <= 0.45 (< 0.50), got {:?}",
        abstained_res.confidence
    );
    assert_eq!(
        abstained_res
            .confidence
            .iter()
            .filter(|&&c| c >= 0.80)
            .count(),
        0,
        "ABSTAIN must guarantee zero high-confidence calls"
    );

    // ABSTAIN case 2: sequential connectivity collapse (< 15% linked assigned residues)
    let mut scrambled_pos = vec![-1i32; systems_clean.len()];
    for (s, slot) in scrambled_pos.iter_mut().enumerate().take(12) {
        // Place isolated spin systems far apart with non-matching carbon links
        let target_k = (1 + s * 6).min(truth.sequence.len() - 1);
        *slot = target_k as i32;
    }
    let disconnected_res = AssignResult {
        position: scrambled_pos,
        confidence: vec![0.92; systems_clean.len()],
        score: 1.0,
        run_scores: Vec::new(),
    };
    let qc_abstain_conn = evaluate_dataset_qc(
        Some(&clean_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &disconnected_res,
        None,
    );
    assert_eq!(qc_abstain_conn.verdict, DatasetQcVerdict::Abstain);
    assert!(qc_abstain_conn.connectivity_frac < 0.15);
}

#[test]
fn protein_reliability_curve_bins_ece_and_abstention_status() {
    // 4 bins: [0.0..0.5), [0.5..0.8), [0.8..0.9), [0.9..1.0]
    let preds = vec![
        (0.20, false),
        (0.40, true), // bin 0: n=2, mean_conf=0.30, acc=0.50 -> |0.30 - 0.50| = 0.20
        (0.60, true), // bin 1: n=1, mean_conf=0.60, acc=1.00 -> |0.60 - 1.00| = 0.40
        (0.85, true), // bin 2: n=1, mean_conf=0.85, acc=1.00 -> |0.85 - 1.00| = 0.15
        (0.95, true),
        (1.00, false), // bin 3: n=2, mean_conf=0.975, acc=0.50 -> |0.975 - 0.50| = 0.475
    ];
    let curve = ProteinReliabilityCurve::from_predictions(&preds, 0.80);
    assert_eq!(curve.total_predictions, 6);
    assert_eq!(curve.bins[0].count, 2);
    assert_eq!(curve.bins[1].count, 1);
    assert_eq!(curve.bins[2].count, 1);
    assert_eq!(curve.bins[3].count, 2);
    assert!((curve.bins[0].empirical_acc - 0.50).abs() < 1e-9);
    assert!((curve.bins[3].mean_conf - 0.975).abs() < 1e-9);

    let expected_ece = (2.0 * 0.20 + 1.0 * 0.40 + 1.0 * 0.15 + 2.0 * 0.475) / 6.0;
    assert!(
        (curve.ece - expected_ece).abs() < 1e-9,
        "expected ECE {expected_ece:.6}, got {:.6}",
        curve.ece
    );
    assert_eq!(curve.hc_assigned, 3);
    assert_eq!(curve.hc_correct, 2);
    assert!(!curve.is_abstained());
    assert_eq!(curve.hc_status, "66.7% (2/3)");

    // Distinguish explicit abstention (`0 HC calls`) from wrong HC calls (`0.0% (0/2)`)
    let abstained = ProteinReliabilityCurve::from_predictions(&[(0.45, false), (0.35, true)], 0.80);
    assert!(abstained.is_abstained());
    assert_eq!(abstained.hc_precision, None);
    assert_eq!(abstained.hc_status, "Abstained (0 HC calls)");

    let wrong_hc = ProteinReliabilityCurve::from_predictions(&[(0.91, false), (0.84, false)], 0.80);
    assert!(!wrong_hc.is_abstained());
    assert_eq!(wrong_hc.hc_precision, Some(0.0));
    assert_eq!(wrong_hc.hc_status, "0.0% (0/2)");
}

#[test]
fn tms_shifted_all_helix_and_all_sheet_proteins_pass_joint_offset_qc() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let base_truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();

    // Build synthetic 100% alpha-helix and 100% beta-sheet chemical-shift references around BMRB
    // prior means (Wishart & Sykes 1994 secondary shifts relative to BMRB globular means:
    // alpha-helix: CA +1.65, CB -0.95, C +1.25 ppm; beta-sheet: CA -1.45, CB +1.35, C -1.10 ppm),
    // PLUS a uniform -2.70 ppm TMS 13C reference shift on all carbon nuclei!
    for (label, sec_ca, sec_cb, sec_c) in [
        ("all_helix_tms", 1.65_f64, -0.95_f64, 1.25_f64),
        ("all_sheet_tms", -1.45_f64, 1.35_f64, -1.10_f64),
    ] {
        let mut synth = base_truth.clone();
        for k in 0..synth.sequence.len() {
            let rname = &synth.sequence[k];
            let phi_conj = 0.5 * (1.0 + 5.0_f64.sqrt());
            let jitter = ((k as f64) * phi_conj).sin() * 0.35;
            if let Some((mu_ca, _)) = priors.get(rname, "CA") {
                synth
                    .shifts
                    .insert((k, "CA".to_string()), mu_ca + sec_ca + jitter - 2.70);
            }
            if rname != "GLY" {
                if let Some((mu_cb, _)) = priors.get(rname, "CB") {
                    synth
                        .shifts
                        .insert((k, "CB".to_string()), mu_cb + sec_cb - 0.5 * jitter - 2.70);
                }
            }
            if let Some((mu_c, _)) = priors.get(rname, "C") {
                synth
                    .shifts
                    .insert((k, "C".to_string()), mu_c + sec_c + 0.4 * jitter - 2.70);
            }
        }

        let (peaks, _) = simulate(
            &synth,
            &SimConfig {
                seed: 7,
                ..SimConfig::default()
            },
        );
        let (_systems, _res, rep) = assign_evo_graph_from_peaks(
            &peaks,
            &synth.sequence,
            &priors,
            &EvoGraphConfig::default(),
        );
        assert_eq!(
            rep.qc_report.verdict,
            DatasetQcVerdict::Pass,
            "{label} (-2.70 ppm TMS) must earn QC=PASS under unified joint (CA, CB, C') offset estimator, got {:?}",
            rep.qc_report
        );
        assert!(
            rep.qc_report.offset_consistency_ok,
            "{label} must have offset_consistency_ok=true: {:?}",
            rep.qc_report.detailed_offsets
        );
        assert_eq!(
            rep.qc_report.detailed_offsets.ca.status,
            OffsetFitStatus::Fitted
        );
        assert_eq!(
            rep.qc_report.detailed_offsets.cb.status,
            OffsetFitStatus::Fitted
        );
        assert!(
            (rep.qc_report.detailed_offsets.ca.applied_ppm - (-2.70)).abs() <= 0.75,
            "{label} expected dCA near -2.70 ppm, got {}",
            rep.qc_report.detailed_offsets.ca.applied_ppm
        );
        assert!(
            (rep.qc_report.detailed_offsets.cb.applied_ppm - (-2.70)).abs() <= 0.75,
            "{label} expected dCB near -2.70 ppm, got {}",
            rep.qc_report.detailed_offsets.cb.applied_ppm
        );
    }
}

#[test]
fn monotone_calibration_maps_isotonic_pav_and_unconverged_margin_abstention() {
    // 1. Canonical grid (`g12`) and symmetric Beta(0.25, 0.25) prior (`laplace_prior = 0.5`)
    assert_eq!(CANONICAL_CALIBRATION_GRID_12.len(), 13);
    assert_eq!(CANONICAL_CALIBRATION_GRID_12.first().copied(), Some(0.0));
    assert_eq!(CANONICAL_CALIBRATION_GRID_12.last().copied(), Some(1.0));
    assert!((CANONICAL_CALIBRATION_LAPLACE_PRIOR - 0.5).abs() < 1e-12);
    // Empty sample set under symmetric Beta(0.25, 0.25) (`0.25 / 0.50 = 0.50`) must shrink every bin to 0.50
    let neutral_knots = fit_isotonic_monotone_knots(
        &[],
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    assert_eq!(neutral_knots.len(), 15);
    assert_eq!(neutral_knots[0], (0.0, 0.0));
    assert!((neutral_knots[1].1 - 0.25).abs() < 1e-12);
    for &(_, y) in &neutral_knots[2..] {
        assert!(
            (y - 0.5).abs() < 1e-12,
            "expected neutral Beta(0.25, 0.25) mean 0.5, got {y}"
        );
    }

    // Fixed convergence specification constants `(0.65, 0.07, 1.60, 0.16)` with `>= 12` assigned gate
    assert!((QC_STRONG_BELIEF_THRESHOLD - 0.65).abs() < 1e-12);
    assert_eq!(QC_MIN_ASSIGNED_FOR_MARGIN_GATE, 12);
    assert!((QC_MIN_SOLUTION_MARGIN - 0.07).abs() < 1e-12);
    assert!((QC_CROWDED_MULTIPLICITY_RATIO - 1.60).abs() < 1e-12);
    assert!((QC_CROWDED_MIN_SOLUTION_MARGIN - 0.16).abs() < 1e-12);
    assert!(!is_unconverged_solution_margin(11, 0.0, 2.0));
    assert!(is_unconverged_solution_margin(12, 0.069, 1.0));
    assert!(!is_unconverged_solution_margin(12, 0.071, 1.60));
    assert!(is_unconverged_solution_margin(12, 0.159, 1.61));
    assert!(!is_unconverged_solution_margin(12, 0.161, 1.61));

    // All three committed LOPO knot tables must be strictly non-decreasing in both x and y,
    // anchored at (0.0, 0.0) and ending at x = 1.0 with y >= 0.90.
    for (name, knots) in [
        ("evo_graph", EVO_GRAPH_CALIBRATION_KNOTS),
        ("standard_cpu", STANDARD_CALIBRATION_KNOTS),
        ("standard_gpu", STANDARD_GPU_CALIBRATION_KNOTS),
    ] {
        assert_eq!(knots.len(), 15, "{name} expected 15 g12 knots");
        assert_eq!(
            knots.first().copied(),
            Some((0.0, 0.0)),
            "{name} start knot"
        );
        let (x_end, y_end) = knots.last().copied().expect("non-empty knots");
        assert!((x_end - 1.0).abs() < 1e-12, "{name} end x");
        assert!((0.90..=1.0).contains(&y_end), "{name} end y={y_end}");
        for w in knots.windows(2) {
            assert!(w[1].0 > w[0].0, "{name} x must strictly increase: {w:?}");
            assert!(
                w[1].1 + 1e-12 >= w[0].1,
                "{name} y must be monotone non-decreasing: {w:?}"
            );
        }
    }

    // 2. Piecewise-linear evaluation & isotonic PAV regression preserve monotonicity
    let mut prev_evo = 0.0;
    let mut prev_std = 0.0;
    let mut prev_gpu = 0.0;
    for step in 0..=200 {
        let p = (step as f64) / 200.0;
        let ce = apply_evo_graph_calibration_map(p);
        let cs = apply_standard_calibration_map(p);
        let cg = apply_standard_gpu_calibration_map(p);
        assert!(ce + 1e-12 >= prev_evo && (0.0..=1.0).contains(&ce));
        assert!(cs + 1e-12 >= prev_std && (0.0..=1.0).contains(&cs));
        assert!(cg + 1e-12 >= prev_gpu && (0.0..=1.0).contains(&cg));
        prev_evo = ce;
        prev_std = cs;
        prev_gpu = cg;
    }

    let synth_samples = vec![
        (0.10, false),
        (0.15, false),
        (0.35, true),
        (0.45, false), // non-monotone inversion that PAV must pool
        (0.75, true),
        (0.85, true),
        (0.95, true),
    ];
    let grid = [0.0, 0.25, 0.50, 0.80, 1.0];
    let fitted = fit_isotonic_monotone_knots(&synth_samples, &grid, 1.0);
    for w in fitted.windows(2) {
        assert!(w[1].1 + 1e-12 >= w[0].1, "PAV violation: {w:?}");
    }
    assert_eq!(eval_monotone_calibration_map(0.0, &fitted), 0.0);

    // 3. Unconverged ensemble (`solution_margin < 0.07` with `>= 12` placed residues) must ABSTAIN
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let score_cfg = ScoreConfig::default();
    let (clean_peaks, _) = simulate(&truth, &SimConfig::default());
    let (systems_clean, mut res_unconv, _) = assign_evo_graph_from_peaks(
        &clean_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );
    for c in &mut res_unconv.confidence {
        if *c > 0.0 {
            *c = 0.42; // 0% of placed residues >= 0.65 -> solution_margin = 0.0 < 0.07
        }
    }
    let qc_unconv = evaluate_dataset_qc(
        Some(&clean_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_unconv,
        None,
    );
    assert_eq!(
        qc_unconv.verdict,
        DatasetQcVerdict::Abstain,
        "collapsed solution_margin must trigger Abstain: {qc_unconv:?}"
    );
    assert!(qc_unconv.solution_margin < 0.07);
}

#[test]
fn p1_7_hnco_absent_complementary_4exp_earns_pass_even_with_partial_hncaco_dropout() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();

    // Simulate HNCO-absent / HNCA-absent complementary 4-experiment set:
    // HSQC + HNCACB + CBCACONH + HNCOCA + HNCACO (providing full CA(i/i-1), CB(i/i-1), C'(i/i-1)),
    // with Asn/Gln NH2 doublets, a multi-day (+0.024 H, -0.22 N) ppm lock shift on HNCOCA,
    // and 35% HNCACO sensitivity dropout so roots missing their own HNCACO peak sit near neighbors.
    let (mut peaks, _) = simulate(
        &truth,
        &SimConfig {
            experiments: vec![
                "HSQC".to_string(),
                "HNCACB".to_string(),
                "CBCACONH".to_string(),
                "HNCOCA".to_string(),
                "HNCACO".to_string(),
            ],
            sidechain_nh2: true,
            nh2_sidechain_3d: true,
            seed: 13,
            ..SimConfig::default()
        },
    );
    let mut hncaco_idx = 0usize;
    peaks.retain_mut(|p| {
        if p.exp == "HNCOCA" {
            p.h += 0.024;
            p.n -= 0.22;
        }
        if p.exp == "HNCACO" {
            hncaco_idx += 1;
            if hncaco_idx.is_multiple_of(3) {
                return false;
            }
        }
        true
    });

    let (_systems, res, rep) =
        assign_evo_graph_from_peaks(&peaks, &truth.sequence, &priors, &EvoGraphConfig::default());
    assert_eq!(
        rep.qc_report.verdict,
        DatasetQcVerdict::Pass,
        "HNCO-absent complementary 4-exp set with complete CA+CB+C' connectivity must earn PASS (`P1-7a`), got {:?}",
        rep.qc_report
    );
    assert!(!rep.qc_report.warn_sub_hc_uncalibrated);
    assert!(!rep.qc_report.protein_hc_gate_triggered);
    assert!(
        rep.qc_report.hsqc_3d_rms_dn_ppm < 0.26,
        "Voronoi-nearest HNCACO root matching must prevent artificial dN inflation: {}",
        rep.qc_report.hsqc_3d_rms_dn_ppm
    );
    assert!(rep.qc_report.aliphatic_support_frac >= 0.85);
    assert!(rep.qc_report.corroborated_connectivity_frac >= 0.70);
    assert!(
        res.confidence.iter().filter(|&&c| c >= 0.80).count() >= 30,
        "clean complementary 4-exp set must retain strong HC coverage"
    );
}

#[test]
fn p1_7_degraded_regimes_trigger_aliphatic_abstain_warn_and_protein_hc_cap() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let score_cfg = ScoreConfig::default();

    // 1. HNCO-only dataset (`aliphatic_support_frac < 0.20`) must ABSTAIN even when HSQC 3D support is ~100%.
    let (hnco_only_peaks, _) = simulate(
        &truth,
        &SimConfig {
            experiments: vec!["HSQC".to_string(), "HNCO".to_string()],
            seed: 5,
            ..SimConfig::default()
        },
    );
    let (_sys_hnco, res_hnco, rep_hnco) = assign_evo_graph_from_peaks(
        &hnco_only_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );
    assert_eq!(
        rep_hnco.qc_report.verdict,
        DatasetQcVerdict::Abstain,
        "HNCO-only dataset lacking CA/CB support must ABSTAIN (`P1-7b`), got {:?}",
        rep_hnco.qc_report
    );
    assert!(rep_hnco.qc_report.aliphatic_support_frac < 0.20);
    assert!((rep_hnco.qc_report.confidence_cap - 0.45).abs() < 1e-12);
    assert!(res_hnco.confidence.iter().all(|&c| c <= 0.45));

    // Baseline clean Ubiquitin spin systems for controlled degraded-regime probes
    let (clean_peaks, _) = simulate(&truth, &SimConfig::default());
    let (systems_clean, res_clean, _) = assign_evo_graph_from_peaks(
        &clean_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );

    // 2. Sparse aliphatic support (`0.20 <= aliphatic_support_frac < 0.50`) must trigger WARN + Q4 flag.
    let mut sparse_aliph_sys = systems_clean.clone();
    for (idx, s) in sparse_aliph_sys.iter_mut().enumerate() {
        if idx * 10 >= 4 * systems_clean.len() {
            s.cands.retain(|(nuc, _), _| nuc != "CA" && nuc != "CB");
            s.cands
                .entry(("C".to_string(), -1))
                .or_insert_with(|| vec![(175.5, 1)]);
        }
    }
    let qc_sparse_aliph = evaluate_dataset_qc(
        Some(&clean_peaks),
        &sparse_aliph_sys,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        None,
    );
    assert_eq!(qc_sparse_aliph.verdict, DatasetQcVerdict::Warn);
    assert!(
        (0.20..0.50).contains(&qc_sparse_aliph.aliphatic_support_frac),
        "expected aliphatic_support_frac in [0.20, 0.50), got {}",
        qc_sparse_aliph.aliphatic_support_frac
    );
    assert!(qc_sparse_aliph.warn_sub_hc_uncalibrated);

    // 3. Low corroborated connectivity (`P >= 3.5` < 0.70 while weak `P >= 1.0` stays >= 0.15):
    // strip CB and sequential CA(i-1) and perturb single-channel C'(i-1) by +0.26 ppm so single-channel
    // C' links sit in [1.0, 3.5) nats, triggering protein HC cap <= 0.79.
    let mut weak_link_sys = systems_clean.clone();
    for s in &mut weak_link_sys {
        s.cands
            .retain(|(nuc, off), _| nuc != "CB" && !(nuc == "CA" && *off != 0));
        if let Some(c_prev) = s.cands.get_mut(&("C".to_string(), -1)) {
            for cand in c_prev.iter_mut() {
                cand.0 += 0.26;
            }
        }
    }
    let qc_weak_corr = evaluate_dataset_qc(
        Some(&clean_peaks),
        &weak_link_sys,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        None,
    );
    assert_eq!(qc_weak_corr.verdict, DatasetQcVerdict::Warn);
    assert!(qc_weak_corr.corroborated_connectivity_frac < 0.66);
    assert!(qc_weak_corr.protein_hc_gate_triggered);
    assert!((qc_weak_corr.confidence_cap - QC_PROTEIN_HC_CONFIDENCE_CAP).abs() < 1e-12);
    let mut capped_weak = res_clean.clone();
    apply_dataset_qc_to_assignments(&mut capped_weak, &qc_weak_corr);
    assert!(
        capped_weak.confidence.iter().all(|&c| c < 0.80),
        "triggered protein-level HC gate must suppress all >= 0.80 commitments"
    );

    // 4. Isolated sparse HC tail (`1 / 40 = 2.5% < 8%` HC call fraction) must cap at <= 0.79.
    let mut isolated_tail_res = res_clean.clone();
    let mut placed_seen = 0usize;
    for (s, &pos) in isolated_tail_res.position.iter().enumerate() {
        if pos >= 0 {
            placed_seen += 1;
            isolated_tail_res.confidence[s] = if placed_seen == 1 {
                0.93
            } else if placed_seen <= 12 {
                0.70
            } else {
                0.48
            };
        }
    }
    let qc_sparse_hc = evaluate_dataset_qc(
        Some(&clean_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &isolated_tail_res,
        None,
    );
    assert_eq!(qc_sparse_hc.verdict, DatasetQcVerdict::Warn);
    assert!(qc_sparse_hc.protein_hc_gate_triggered);
    assert!((qc_sparse_hc.confidence_cap - QC_PROTEIN_HC_CONFIDENCE_CAP).abs() < 1e-12);
    apply_dataset_qc_to_assignments(&mut isolated_tail_res, &qc_sparse_hc);
    assert_eq!(
        isolated_tail_res
            .confidence
            .iter()
            .filter(|&&c| c >= 0.80)
            .count(),
        0,
        "isolated 1-residue HC tail spike must be capped below 0.80"
    );

    // 5. Weakly supported large carbon registration offset (`|dCA| = 3.80 > 3.20 ppm` with
    // `log_likelihood_gain / n_values = 2.0 / 50 = 0.04 < 0.15` and `solution_margin < 0.50`)
    // must flag `has_weak_likelihood_large_offset` and trigger protein-level HC cap.
    let mk_fit = |status, applied: f64, gain: f64, n: usize| NucleusOffsetFit {
        status,
        applied_ppm: applied,
        candidate_ppm: applied,
        log_likelihood_gain: gain,
        n_values: n,
    };
    let weak_large_off = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 50),
        n: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 50),
        ca: mk_fit(OffsetFitStatus::Fitted, -3.80, 2.0, 50),
        cb: mk_fit(OffsetFitStatus::Fitted, -3.75, 2.0, 48),
        c: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 50),
    };
    assert!(has_weak_likelihood_large_offset(&weak_large_off, 0.15).is_some());
    let mut low_margin_res = res_clean.clone();
    let mut k_seen = 0usize;
    for (s, &pos) in low_margin_res.position.iter().enumerate() {
        if pos >= 0 {
            k_seen += 1;
            low_margin_res.confidence[s] = if k_seen <= 15 { 0.94 } else { 0.52 };
        }
    }
    let qc_weak_off = evaluate_dataset_qc(
        Some(&clean_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &low_margin_res,
        Some(&weak_large_off),
    );
    assert_eq!(qc_weak_off.verdict, DatasetQcVerdict::Warn);
    assert!(qc_weak_off.protein_hc_gate_triggered);
    assert!((qc_weak_off.confidence_cap - QC_PROTEIN_HC_CONFIDENCE_CAP).abs() < 1e-12);
}

#[test]
fn q4_warn_marks_sub_hc_calls_uncalibrated_and_preserves_pass_calibration() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let truth = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let score_cfg = ScoreConfig::default();

    let (clean_peaks, _) = simulate(&truth, &SimConfig::default());
    let (systems_clean, res_clean, rep_clean) = assign_evo_graph_from_peaks(
        &clean_peaks,
        &truth.sequence,
        &priors,
        &EvoGraphConfig::default(),
    );

    // Under PASS: `warn_sub_hc_uncalibrated == false`; both sub-HC and HC calls are calibrated.
    assert_eq!(rep_clean.qc_report.verdict, DatasetQcVerdict::Pass);
    assert!(!rep_clean.qc_report.warn_sub_hc_uncalibrated);
    assert!(rep_clean.qc_report.is_call_calibrated(0.45));
    assert!(rep_clean.qc_report.is_call_calibrated(0.85));

    // Under WARN (e.g. aliphatic offset mismatch with high solution_margin):
    // `warn_sub_hc_uncalibrated == true`; sub-HC calls (`< 0.80`) are uncalibrated (`Q4`),
    // while surviving `>= 0.80` HC calls remain certified when `confidence_calibrated == true`.
    let mk_fit = |status, applied: f64, gain: f64, n: usize| NucleusOffsetFit {
        status,
        applied_ppm: applied,
        candidate_ppm: applied,
        log_likelihood_gain: gain,
        n_values: n,
    };
    let mismatch_off = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 60),
        n: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 60),
        ca: mk_fit(OffsetFitStatus::Fitted, -2.60, 42.0, 60),
        cb: mk_fit(OffsetFitStatus::Fitted, 1.40, 38.0, 55),
        c: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.1, 60),
    };
    let qc_warn = evaluate_dataset_qc(
        Some(&clean_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        Some(&mismatch_off),
    );
    assert_eq!(qc_warn.verdict, DatasetQcVerdict::Warn);
    assert!(qc_warn.warn_sub_hc_uncalibrated);
    assert!(!qc_warn.is_call_calibrated(0.799));
    assert!(!qc_warn.is_call_calibrated(0.50));
    assert!(qc_warn.is_call_calibrated(0.80));
    assert!(qc_warn.is_call_calibrated(0.92));

    // Under ABSTAIN: `warn_sub_hc_uncalibrated == false`, and `is_call_calibrated` is false everywhere.
    let mut broken_peaks = clean_peaks.clone();
    for p in &mut broken_peaks {
        if p.exp != "HSQC" {
            p.n += 9.0;
        }
    }
    let qc_abs = evaluate_dataset_qc(
        Some(&broken_peaks),
        &systems_clean,
        &truth.sequence,
        &priors,
        &score_cfg,
        &res_clean,
        None,
    );
    assert_eq!(qc_abs.verdict, DatasetQcVerdict::Abstain);
    assert!(!qc_abs.warn_sub_hc_uncalibrated);
    assert!(!qc_abs.is_call_calibrated(0.40));
    assert!(!qc_abs.is_call_calibrated(0.85));
}

#[test]
fn p1_8_debiased_ece_bernoulli_floor_beta_pav_and_nested_lopo_shrinkage() {
    // 1. Small-N Bernoulli oracle floor vs Ferro-Fricker (2012) debiased ECE_10:
    // Generate exact-oracle predictions where empirical rate deviates only by finite-sample quantization
    // across 10 bins of N = 40 total residues (`4` per bin).
    let mut small_oracle = Vec::new();
    for b in 0..10 {
        let p = 0.05 + 0.10 * (b as f64);
        let k_true = (4.0 * p).round() as usize;
        for j in 0..4 {
            small_oracle.push((p, j < k_true));
        }
    }
    let raw_ece_10 = compute_grid_ece(&small_oracle, UNIFORM_CALIBRATION_GRID_10);
    let floor_10 =
        compute_grid_ece_bernoulli_floor(&small_oracle, UNIFORM_CALIBRATION_GRID_10, 1.0);
    let deb_ece_10 = compute_debiased_grid_ece(&small_oracle, UNIFORM_CALIBRATION_GRID_10, 1.0);
    assert!(
        floor_10 > 0.06,
        "analytical 10-bin Bernoulli floor at N=40 must exceed 0.06, got {floor_10}"
    );
    assert!(
        deb_ece_10 < raw_ece_10 && deb_ece_10 < 0.02,
        "debiased ECE_10 ({deb_ece_10}) must eliminate small-sample plug-in bias ({raw_ece_10})"
    );
    let curve = ProteinReliabilityCurve::from_predictions(&small_oracle, 0.80);
    assert!((curve.ece_10 - raw_ece_10).abs() < 1e-12);
    assert!((curve.debiased_ece_10 - deb_ece_10).abs() < 1e-12);
    assert!((curve.ece_10_bernoulli_floor - floor_10).abs() < 1e-12);

    // Large-N asymptotic consistency: as N -> 4000, floor -> ~0 and debiased_ece_10 -> raw_ece_10.
    let mut large_miscal = Vec::new();
    for i in 0..4000 {
        // Consistently overconfident by +0.15 (`p = 0.85`, true accuracy `0.70`)
        large_miscal.push((0.85, (i % 10) < 7));
    }
    let raw_large = compute_grid_ece(&large_miscal, UNIFORM_CALIBRATION_GRID_10);
    let deb_large = compute_debiased_grid_ece(&large_miscal, UNIFORM_CALIBRATION_GRID_10, 1.0);
    let floor_large =
        compute_grid_ece_bernoulli_floor(&large_miscal, UNIFORM_CALIBRATION_GRID_10, 1.0);
    assert!(floor_large < 0.01);
    assert!((raw_large - 0.15).abs() < 1e-3);
    assert!((deb_large - raw_large).abs() < 2e-3);

    // 2. Conservative conjugate Beta(0.75, 2.25) PAV prior for CPU `standard` (`P1-8`) must shrink
    // sparse upper-intermediate bins strictly below symmetric Beta(0.25, 0.25).
    let sparse_samples = vec![(0.80, true), (0.81, true), (0.98, true)];
    let sym_knots = fit_isotonic_monotone_knots(
        &sparse_samples,
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    let cons_knots = fit_isotonic_monotone_knots_beta(
        &sparse_samples,
        CANONICAL_CALIBRATION_GRID_12,
        STANDARD_CPU_PRIOR_ALPHA,
        STANDARD_CPU_PRIOR_BETA,
    );
    let y_sym = eval_monotone_calibration_map(0.80, &sym_knots);
    let y_cons = eval_monotone_calibration_map(0.80, &cons_knots);
    assert!(
        y_cons + 0.10 < y_sym,
        "Beta(0.75, 2.25) prior ({y_cons}) must curb sparse-bin overconfidence relative to Beta(0.25, 0.25) ({y_sym})"
    );

    // 3. Nested LOPO GPU shrinkage selector (`select_gpu_shrinkage_weight_nested_lopo`) and
    // `fit_protein_hc_gate_config` must run cleanly with zero target leakage.
    assert_eq!(CANDIDATE_GPU_SHRINKAGE_WEIGHTS.len(), 7);
    assert!((CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT - 0.50).abs() < 1e-12);
    let mut gpu_proteins = Vec::new();
    let mut cpu_by_protein = Vec::new();
    for p_idx in 0..6 {
        let mut gpu_raw = Vec::new();
        let mut cpu_raw = Vec::new();
        for r in 0..30 {
            if r < 10 {
                // Sparse noisy GPU bin around 0.85 where true rate is 0.60, while CPU has dense observations
                gpu_raw.push((0.85, ((r + p_idx) % 5) < 3));
                for rep in 0..3 {
                    cpu_raw.push((0.85, ((r + rep) % 5) < 3));
                }
            } else {
                gpu_raw.push((0.995, (r % 20) != 0));
                cpu_raw.push((0.995, (r % 20) != 0));
            }
        }
        gpu_proteins.push(vec![ProteinHcTrainingSeed {
            n_assignable_amides: 32,
            hsqc_3d_support_frac: 0.95,
            supported_roots_per_assignable: 0.95,
            aliphatic_support_frac: 0.95,
            hsqc_multiplicity_ratio: 1.0,
            hsqc_3d_rms_dh_ppm: 0.01,
            hsqc_3d_rms_dn_ppm: 0.10,
            offset_consistency_ok: true,
            weak_large_offset: false,
            connectivity_frac: 0.92,
            corroborated_connectivity_frac: 0.88,
            raw_predictions: gpu_raw,
        }]);
        cpu_by_protein.push(cpu_raw);
    }
    let trainable = vec![true; 6];
    let (sel_w, stats) = select_gpu_shrinkage_weight_nested_lopo(
        &gpu_proteins,
        &cpu_by_protein,
        &trainable,
        &trainable,
        &CANONICAL_PROTEIN_HC_GATE,
    );
    assert_eq!(stats.len(), CANDIDATE_GPU_SHRINKAGE_WEIGHTS.len());
    assert!(
        sel_w > 0.0,
        "expected positive CPU anchor weight when sparse GPU bins overcommit, got sel_w={sel_w}, stats={stats:?}"
    );
    let fitted_gate = fit_protein_hc_gate_config(&gpu_proteins, STANDARD_GPU_CALIBRATION_KNOTS);
    let eval_out = evaluate_seed_predictions_with_gate(
        &gpu_proteins[0][0],
        STANDARD_GPU_CALIBRATION_KNOTS,
        &fitted_gate,
    );
    assert_eq!(eval_out.len(), 30);
}

#[test]
fn p1_7_real_cisd3_pdata_5seed_retains_pass_and_hc_calls() {
    let Ok(real_dir) = std::env::var("NMR_REAL_DATA_DIR") else {
        return;
    };
    let cisd3_dir = PathBuf::from(real_dir).join("private/bmrb51439_cisd3");
    if !cisd3_dir.exists() {
        return;
    }
    let fasta_txt = std::fs::read_to_string(cisd3_dir.join("cisd3.fasta")).unwrap();
    let sequence: Vec<String> = fasta_txt
        .lines()
        .filter(|l| !l.trim().starts_with('>'))
        .flat_map(|l| l.trim().chars())
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| nmr_core::priors::one_to_three(c).to_string())
        .collect();
    let pcfg = nmr_core::picker::PickerConfig::default();
    let hsqc = nmr_core::spectrum::read_bruker_pdata_2d(
        cisd3_dir.join("CISD3_15N_HSQC_diamagnetic/pdata/1"),
    )
    .unwrap();
    let hsqc_roots = nmr_core::picker::pick_hsqc_2d(&hsqc, &pcfg);
    let mut raw_peaks = hsqc_roots.clone();
    for (rel_path, exp_name) in [
        ("CISD3_HNCA/pdata/1", "HNCA"),
        ("CISD3_HNCOCA/pdata/1", "HNCOCA"),
        ("CISD3_CBCACONH/pdata/1", "CBCACONH"),
        ("CISD3_HNCO/pdata/1", "HNCO"),
    ] {
        let spec3d =
            nmr_core::spectrum::read_bruker_pdata_3d(cisd3_dir.join(rel_path), exp_name).unwrap();
        raw_peaks.extend(nmr_core::picker::pick_triple_3d(
            &spec3d,
            &hsqc_roots,
            &pcfg,
        ));
    }
    let priors = Priors::bmrb();
    for seed in 0..5u64 {
        let cfg = EvoGraphConfig {
            structure_torsions: None,
            seed,
            ..Default::default()
        };
        let (_sys, res, rep) = assign_evo_graph_from_peaks(&raw_peaks, &sequence, &priors, &cfg);
        let n_hc = res.confidence.iter().filter(|&&c| c >= 0.80).count();
        assert_eq!(
            rep.qc_report.verdict,
            DatasetQcVerdict::Pass,
            "CISD3 pdata seed {seed} expected PASS, got {:?}",
            rep.qc_report
        );
        assert!(
            n_hc >= 5,
            "CISD3 pdata seed {seed} expected >= 5 HC calls, got {n_hc}"
        );
    }
}
