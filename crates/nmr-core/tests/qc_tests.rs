//! Unit and integration tests for per-dataset quality-control (`DatasetQcVerdict`,
//! `DatasetQcReport`, `evaluate_dataset_qc`, `apply_dataset_qc_to_assignments`),
//! per-protein reliability curves (`ProteinReliabilityCurve`), and detailed chemical-shift
//! reference offset diagnostics (`OffsetFitStatus`, `NucleusOffsetFit`, `DetailedReferenceOffsets`).
//!
//! Provenance: Clean-room verification of `nmr_core::qc` and `nmr_core::priors` (`P1-1`, `P1-3`).

use std::path::PathBuf;

use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::priors::{
    fit_unlabelled_global_offsets, fit_unlabelled_global_offsets_detailed,
    DetailedReferenceOffsets, NucleusOffsetFit, OffsetFitStatus, Priors,
};
use nmr_core::qc::{
    apply_dataset_qc_to_assignments, check_offset_consistency, evaluate_dataset_qc,
    DatasetQcVerdict, ProteinReliabilityCurve,
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
        shifts: Default::default(),
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
        shifts: Default::default(),
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

    // Suspect case A: dCA Fitted at -2.70 ppm while dCB has 45 observations yet is NotSignificant near 0.0
    let ca_only_anom = DetailedReferenceOffsets {
        cb: mk_fit(OffsetFitStatus::NotSignificant, 0.0, -0.10, 0.5, 45),
        ..cisd3_like
    };
    let (ok_a, issues_a) = check_offset_consistency(&ca_only_anom);
    assert!(!ok_a && !issues_a.is_empty());

    // Suspect case B (e.g., BMRB 53280-like): dCB Fitted at +2.93 ppm while dCA is NotSignificant near 0.0
    let cb_only_anom = DetailedReferenceOffsets {
        h: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.0, 0.1, 60),
        n: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.0, 0.1, 60),
        ca: mk_fit(OffsetFitStatus::NotSignificant, 0.0, 0.12, 0.7, 60),
        cb: mk_fit(OffsetFitStatus::Fitted, 2.93, 2.93, 24.0, 55),
        c: mk_fit(OffsetFitStatus::NotSignificant, 0.0, -0.05, 0.3, 60),
    };
    let (ok_b, issues_b) = check_offset_consistency(&cb_only_anom);
    assert!(!ok_b && !issues_b.is_empty());

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
