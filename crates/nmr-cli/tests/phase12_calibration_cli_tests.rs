//! Phase 12 calibration hygiene, CLI parameter bounds, and replica-mismatch warning tests
//! (`P1-8`, `Q2`, and `P2`).
//!
//! Provenance: Clean-room verification of `nmr_core::qc` isotonic calibration knot properties,
//! multi-grid `ECE_4` / `ECE_10` / `ECE_12` reliability curves, empirical-Bayes shrinkage toward
//! an anchor sample set, and `nmrassign` CLI `--runs` / `--steps` / `--r-bleach` bounds and
//! uncalibrated replica-count warnings.

use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use nmr_core::qc::{
    eval_monotone_calibration_map, fit_isotonic_monotone_knots, fit_isotonic_monotone_knots_beta,
    fit_isotonic_monotone_knots_shrunk_toward, DatasetQcReport, DatasetQcVerdict,
    ProteinReliabilityCurve, CANDIDATE_GPU_SHRINKAGE_WEIGHTS,
    CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT, CANONICAL_CALIBRATION_GRID_12,
    CANONICAL_CALIBRATION_LAPLACE_PRIOR, CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
    CANONICAL_PROTEIN_HC_GATE, EVO_GRAPH_CALIBRATED_RUNS, QC_MIN_LARGE_OFFSET_GAIN_PER_OBS,
    QC_PROTEIN_HC_CONFIDENCE_CAP, STANDARD_CPU_CALIBRATED_RUNS, STANDARD_CPU_PRIOR_ALPHA,
    STANDARD_CPU_PRIOR_BETA, STANDARD_GPU_CALIBRATED_RUNS, STANDARD_GPU_PRIOR_ALPHA,
    STANDARD_GPU_PRIOR_BETA, UNIFORM_CALIBRATION_GRID_10,
};

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

fn fixture_star_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

#[test]
fn test_p2_x_left_knot_is_half_first_bin_mean_not_clamped_to_1e_4() {
    // All training samples sit in [0.10, 0.20) around x = 0.15, so x_knots[0] > 0.10.
    // Prior to P2, `(0.5 * x_knots[0]).min(1e-4)` forced x_left = 1e-4 unconditionally,
    // creating a near-vertical step at p_raw ~ 0 instead of smooth linear interpolation
    // across [0, x_knots[0]].
    let samples: Vec<(f64, bool)> = (0..40).map(|i| (0.15, i % 5 == 0)).collect();
    let knots = fit_isotonic_monotone_knots(
        &samples,
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    assert!(knots.len() >= 4);
    assert_eq!(knots[0], (0.0, 0.0));
    let x_first_bin = knots[2].0;
    let x_left = knots[1].0;
    assert!(
        x_first_bin > 0.02,
        "expected first interior bin center > 0.02, got {x_first_bin}"
    );
    assert!(
        (x_left - 0.5 * x_first_bin).abs() < 1e-12,
        "expected x_left == 0.5 * x_knots[0] ({:.6}), got {x_left:.6} (must NOT be clamped to 1e-4)",
        0.5 * x_first_bin
    );
    assert!(x_left > 1e-3, "x_left={x_left} should exceed 1e-3");

    // Verify monotonicity across [0, 1]
    let mut prev = 0.0;
    for step in 0..=200 {
        let x = (step as f64) / 200.0;
        let y = eval_monotone_calibration_map(x, &knots);
        assert!(
            y + 1e-12 >= prev,
            "monotone map decreased at x={x}: {y} < {prev}"
        );
        prev = y;
    }
}

#[test]
fn test_p2_reliability_curve_computes_ece_4_ece_10_and_ece_12() {
    // Construct predictions where 4-bin coarse pooling cancels within [0.0, 0.5),
    // while finer 10-bin and 12-bin grids resolve intra-bin miscalibration:
    // - 50 predictions at p = 0.05 with 45% empirical accuracy (+0.40 underconfident)
    // - 50 predictions at p = 0.45 with 5% empirical accuracy (-0.40 overconfident)
    // Coarse [0.0, 0.5) bin has mean_conf = 0.25 and empirical_acc = 0.25 -> ECE_4 == 0.0!
    // Finer 10-bin and 12-bin grids separate 0.05 and 0.45 -> ECE_10 > 0.35 and ECE_12 > 0.35!
    let mut preds = Vec::new();
    for i in 0..50 {
        preds.push((0.05, i < 22)); // 22/50 = 0.44
    }
    for i in 0..50 {
        preds.push((0.45, i < 3)); // 3/50 = 0.06
    }
    let rel = ProteinReliabilityCurve::from_predictions(&preds, 0.80);
    assert!(
        rel.ece < 1e-12,
        "expected coarse 4-bin ECE_4 == 0 due to cancellation, got {}",
        rel.ece
    );
    assert!(
        (rel.ece_10 - 0.39).abs() < 1e-9,
        "expected 10-bin ECE_10 == 0.39, got {}",
        rel.ece_10
    );
    assert!(
        (rel.ece_12 - 0.39).abs() < 1e-9,
        "expected 12-bin ECE_12 == 0.39, got {}",
        rel.ece_12
    );
}

#[test]
fn test_q2_shrunk_isotonic_knots_interpolate_smoothly_toward_anchor() {
    let target: Vec<(f64, bool)> = vec![(0.85, true); 4];
    let anchor: Vec<(f64, bool)> = (0..100).map(|i| (0.85, i < 50)).collect();

    let knots_unshrunk = fit_isotonic_monotone_knots_shrunk_toward(
        &target,
        &anchor,
        0.0,
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    let knots_shrunk = fit_isotonic_monotone_knots_shrunk_toward(
        &target,
        &anchor,
        0.5,
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    let y_unshrunk = eval_monotone_calibration_map(0.85, &knots_unshrunk);
    let y_shrunk = eval_monotone_calibration_map(0.85, &knots_shrunk);
    assert!(
        y_shrunk < y_unshrunk,
        "anchor shrinkage must pull small-N optimistic bin toward anchor empirical rate: shrunk={y_shrunk:.4} vs unshrunk={y_unshrunk:.4}"
    );
    assert!(y_shrunk > 0.50);
}

#[test]
fn test_p2_cli_bounds_reject_out_of_range_runs_steps_and_r_bleach() {
    let bin = nmrassign_bin();
    let star = fixture_star_path();

    let bad_cases: &[(&[&str], &str)] = &[
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                "/tmp/unused_p12",
                "--runs",
                "0",
            ],
            "out of allowed bounds (1..=4096)",
        ),
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                "/tmp/unused_p12",
                "--runs",
                "4097",
            ],
            "out of allowed bounds (1..=4096)",
        ),
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                "/tmp/unused_p12",
                "--steps",
                "99",
            ],
            "out of allowed bounds (100..=50000000)",
        ),
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                "/tmp/unused_p12",
                "--steps",
                "50000001",
            ],
            "out of allowed bounds (100..=50000000)",
        ),
        (
            &["benchmark", "--star", star.to_str().unwrap(), "--runs", "0"],
            "out of allowed bounds (1..=4096)",
        ),
        (
            &["benchmark-suite", "--steps", "50"],
            "out of allowed bounds (100..=50000000)",
        ),
        (
            &["real-benchmark", "--split", "dev", "--runs", "5000"],
            "out of allowed bounds (1..=4096)",
        ),
        (
            &[
                "sidechain",
                "--star",
                star.to_str().unwrap(),
                "--steps",
                "99",
            ],
            "out of allowed bounds (100..=50000000)",
        ),
        (
            &[
                "pipeline",
                "--star",
                star.to_str().unwrap(),
                "--out-dir",
                "/tmp/unused_p12_dir",
                "--runs",
                "0",
            ],
            "out of allowed bounds (1..=4096)",
        ),
        (
            &[
                "assign",
                "--regime",
                "paramagnetic",
                "--star",
                star.to_str().unwrap(),
                "--out",
                "/tmp/unused_p12",
                "--r-bleach",
                "0.5",
            ],
            "--r-bleach",
        ),
    ];

    for (args, expected_substr) in bad_cases {
        let out = Command::new(&bin)
            .args(*args)
            .output()
            .expect("spawn nmrassign");
        assert!(
            !out.status.success(),
            "expected non-zero exit for args {args:?}, stdout={}, stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(expected_substr),
            "expected {expected_substr:?} in stderr for {args:?}, got: {stderr}"
        );
    }
}

#[test]
fn test_p1_8_default_cpu_assign_is_calibrated_and_mismatched_runs_warns_and_marks_uncalibrated() {
    assert_eq!(STANDARD_CPU_CALIBRATED_RUNS, 16);
    assert_eq!(STANDARD_GPU_CALIBRATED_RUNS, 64);
    assert_eq!(EVO_GRAPH_CALIBRATED_RUNS, 64);

    let mut qc = DatasetQcReport::default();
    assert!(qc.confidence_calibrated);
    assert!(qc.calibration_warning.is_none());
    qc.mark_uncalibrated_runs("standard (CPU)", 4, STANDARD_CPU_CALIBRATED_RUNS);
    assert!(!qc.confidence_calibrated);
    assert!(qc
        .calibration_warning
        .as_deref()
        .unwrap()
        .contains("confidence_calibrated=false"));

    let dir = std::env::temp_dir().join(format!("nmr_p12_cal_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let bin = nmrassign_bin();
    let star = fixture_star_path();
    let peaks_tsv = dir.join("ubq_peaks.tsv");

    // Simulate peaks once
    let sim_out = Command::new(&bin)
        .args([
            "simulate",
            "--star",
            star.to_str().unwrap(),
            "--out",
            peaks_tsv.to_str().unwrap(),
            "--seed",
            "0",
        ])
        .output()
        .expect("simulate");
    assert!(sim_out.status.success());

    // 1. Default `assign --solver standard --engine cpu` (omitting --runs -> defaults to 16)
    // must be calibrated out-of-the-box (`confidence_calibrated: true`, zero stderr warning).
    let out_cal_prefix = dir.join("cal_default");
    let cal_run = Command::new(&bin)
        .args([
            "assign",
            "--peaks",
            peaks_tsv.to_str().unwrap(),
            "--star",
            star.to_str().unwrap(),
            "--solver",
            "standard",
            "--engine",
            "cpu",
            "--steps",
            "300",
            "--out",
            out_cal_prefix.to_str().unwrap(),
        ])
        .output()
        .expect("assign calibrated default");
    assert!(
        cal_run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&cal_run.stderr)
    );
    let cal_stderr = String::from_utf8_lossy(&cal_run.stderr);
    assert!(
        !cal_stderr.contains("confidence_calibrated=false"),
        "default CPU assign must NOT emit uncalibrated warning; stderr={cal_stderr}"
    );
    let cal_str = fs::read_to_string(dir.join("cal_default.str")).unwrap();
    assert!(
        !cal_str.contains("confidence_calibrated=false")
            && !cal_str.contains("calibration_warning="),
        "default calibrated .str output must not contain uncalibrated warning:\n{cal_str}"
    );

    // 2. Explicit `--runs 4` (`!= 16`) must emit a loud stderr warning AND mark
    // `confidence_calibrated=false` + `calibration_warning=` in the `.str` output header!
    let out_uncal_prefix = dir.join("uncal_r4");
    let uncal_run = Command::new(&bin)
        .args([
            "assign",
            "--peaks",
            peaks_tsv.to_str().unwrap(),
            "--star",
            star.to_str().unwrap(),
            "--solver",
            "standard",
            "--engine",
            "cpu",
            "--runs",
            "4",
            "--steps",
            "300",
            "--out",
            out_uncal_prefix.to_str().unwrap(),
        ])
        .output()
        .expect("assign uncalibrated r4");
    assert!(uncal_run.status.success());
    let uncal_stderr = String::from_utf8_lossy(&uncal_run.stderr);
    assert!(
        uncal_stderr.contains("Warning:") && uncal_stderr.contains("confidence_calibrated=false"),
        "expected loud stderr calibration warning for --runs 4, got: {uncal_stderr}"
    );
    let uncal_str = fs::read_to_string(dir.join("uncal_r4.str")).unwrap();
    assert!(
        uncal_str.contains("confidence_calibrated=false"),
        "expected `confidence_calibrated=false` in .str header:\n{uncal_str}"
    );
    assert!(
        uncal_str.contains("calibration_warning="),
        "expected `calibration_warning=` in .str header:\n{uncal_str}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_p1_1_cli_pick_and_pipeline_stderr_warning_report_and_no_axis_rescue() {
    use nmr_core::nmrstar::read_star;
    use nmr_core::pulprog::{apply_indirect_axis_transform_3d, AxisTransformOp};
    use nmr_core::simulate::{
        render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
    };
    use nmr_core::spectrum::{write_ucsf_2d, write_ucsf_3d};

    let dir = std::env::temp_dir().join(format!("nmr_p13_p1_1_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let ref_data = read_star(fixture_star_path(), None).unwrap();
    let (sim_peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            p_missing_peak: 0.0,
            spurious_rate: 0.0,
            seed: 13,
            ..SimConfig::default()
        },
    );
    let rcfg = SpectrumRenderConfig {
        nx_h: 128,
        ny_n: 64,
        nz_c: 64,
        ..SpectrumRenderConfig::default()
    };
    let hsqc = render_hsqc_2d(&sim_peaks, &rcfg);
    let mut hnco_corrupted = render_triple_3d("HNCO", &sim_peaks, &rcfg);
    apply_indirect_axis_transform_3d(
        &mut hnco_corrupted,
        AxisTransformOp::HalfSwShift,
        AxisTransformOp::Canonical,
        Some("HNCO"),
        false,
    );

    let hsqc_path = dir.join("HSQC.ucsf");
    let hnco_path = dir.join("HNCO_halfsw.ucsf");
    write_ucsf_2d(&hsqc_path, &hsqc).unwrap();
    write_ucsf_3d(&hnco_path, &hnco_corrupted).unwrap();

    // 1. Default `nmrassign pick`: emits `warning: [axis-rescue]` on `stderr`!
    let out_on = dir.join("peaks_on.tsv");
    let res_on = Command::new(nmrassign_bin())
        .args([
            "pick",
            "--hsqc",
            hsqc_path.to_str().unwrap(),
            "--triple",
            &format!("HNCO={}", hnco_path.display()),
            "--method",
            "classical",
            "--out",
            out_on.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign pick (rescue on)");
    assert!(
        res_on.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_on.stderr)
    );
    let stderr_on = String::from_utf8_lossy(&res_on.stderr);
    assert!(
        stderr_on.contains("warning: [axis-rescue]"),
        "expected `warning: [axis-rescue]` on stderr, got:\n{stderr_on}"
    );
    assert!(
        stderr_on.contains("HalfSwShiftN"),
        "expected `HalfSwShiftN` in stderr warning, got:\n{stderr_on}"
    );

    // 2. Opt-out `nmrassign pick --no-axis-rescue`: NO `[axis-rescue]` warning on `stderr`!
    let out_off = dir.join("peaks_off.tsv");
    let res_off = Command::new(nmrassign_bin())
        .args([
            "pick",
            "--hsqc",
            hsqc_path.to_str().unwrap(),
            "--triple",
            &format!("HNCO={}", hnco_path.display()),
            "--method",
            "classical",
            "--no-axis-rescue",
            "--out",
            out_off.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign pick (--no-axis-rescue)");
    assert!(
        res_off.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_off.stderr)
    );
    let stderr_off = String::from_utf8_lossy(&res_off.stderr);
    assert!(
        !stderr_off.contains("[axis-rescue]"),
        "expected zero axis-rescue warnings with --no-axis-rescue, got:\n{stderr_off}"
    );

    // 3. `nmrassign pipeline` records the applied rescue inside `report.html` (`id="axis-rescue-provenance"`)!
    let pipe_dir = dir.join("pipe_out");
    let res_pipe = Command::new(nmrassign_bin())
        .args([
            "pipeline",
            "--star",
            fixture_star_path().to_str().unwrap(),
            "--hsqc",
            hsqc_path.to_str().unwrap(),
            "--triple",
            &format!("HNCO={}", hnco_path.display()),
            "--method",
            "classical",
            "--solver",
            "standard",
            "--engine",
            "cpu",
            "--runs",
            "1",
            "--steps",
            "100",
            "--out-dir",
            pipe_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign pipeline");
    assert!(
        res_pipe.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&res_pipe.stderr)
    );
    let report_html = fs::read_to_string(pipe_dir.join("report.html")).expect("read report.html");
    assert!(report_html.contains("id=\"axis-rescue-provenance\""));
    assert!(report_html.contains("HalfSwShiftN"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_phase13_p1_7_p1_8_q4_cli_json_and_calibration_contracts() {
    // 1. Primary 10-bin calibration grid (`ECE_10`), nested LOPO shrinkage candidates, and protein HC gate constants
    assert_eq!(UNIFORM_CALIBRATION_GRID_10.len(), 11);
    assert_eq!(UNIFORM_CALIBRATION_GRID_10.first().copied(), Some(0.0));
    assert_eq!(UNIFORM_CALIBRATION_GRID_10.last().copied(), Some(1.0));
    assert_eq!(
        CANDIDATE_GPU_SHRINKAGE_WEIGHTS,
        &[0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
    );
    assert!((CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT - 0.00).abs() < 1e-12);
    assert!((CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT - 0.20).abs() < 1e-12);
    assert!((QC_PROTEIN_HC_CONFIDENCE_CAP - 0.79).abs() < 1e-12);
    assert!((QC_MIN_LARGE_OFFSET_GAIN_PER_OBS - 0.15).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_corroborated_connectivity - 0.45).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_hc_call_fraction - 0.0).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_crowded_solution_margin - 0.55).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.multiplicity_penalty_floor - 0.0).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_chain_corroborated_frac - 0.0).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_unary_specificity - 0.0).abs() < 1e-12);
    assert!((CANONICAL_PROTEIN_HC_GATE.min_bidir_aliphatic_frac - 0.58).abs() < 1e-12);
    assert!((STANDARD_GPU_PRIOR_ALPHA - 0.35).abs() < 1e-12);
    assert!((STANDARD_GPU_PRIOR_BETA - 0.65).abs() < 1e-12);

    // 2. Q4 `warn_sub_hc_uncalibrated` & `is_call_calibrated` invariants
    let mut warn_qc = DatasetQcReport {
        verdict: DatasetQcVerdict::Warn,
        warn_sub_hc_uncalibrated: true,
        ..DatasetQcReport::default()
    };
    assert!(!warn_qc.is_call_calibrated(0.79));
    assert!(warn_qc.is_call_calibrated(0.80));
    warn_qc.mark_uncalibrated_runs("standard (CPU)", 8, STANDARD_CPU_CALIBRATED_RUNS);
    assert!(!warn_qc.is_call_calibrated(0.95));

    // 3. Conservative Beta(0.75, 2.25) PAV prior for CPU `standard` (`P1-8`)
    let sparse = vec![(0.81, true), (0.82, true), (0.99, true)];
    let sym = fit_isotonic_monotone_knots(
        &sparse,
        CANONICAL_CALIBRATION_GRID_12,
        CANONICAL_CALIBRATION_LAPLACE_PRIOR,
    );
    let cons = fit_isotonic_monotone_knots_beta(
        &sparse,
        CANONICAL_CALIBRATION_GRID_12,
        STANDARD_CPU_PRIOR_ALPHA,
        STANDARD_CPU_PRIOR_BETA,
    );
    assert!(
        eval_monotone_calibration_map(0.81, &cons) < eval_monotone_calibration_map(0.81, &sym),
        "Beta(0.75, 2.25) must be strictly more conservative than symmetric Beta(0.25, 0.25) on sparse bins"
    );

    // 4. Verify `real_manifest.tsv` spent-cohort relabels (`cold_p12` -> `seen_p12`, `cold_p13` -> `seen_p13`; `P1-7` & `F4`)
    let manifest_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark/real_manifest.tsv");
    let manifest_txt = fs::read_to_string(&manifest_path).expect("read real_manifest.tsv");
    let mut n_seen_p12 = 0usize;
    let mut n_seen_p13 = 0usize;
    for line in manifest_txt.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("bmrb_id") {
            continue;
        }
        assert!(
            !t.ends_with("\tcold_p12") && !t.ends_with("\tcold_p13"),
            "spent cohorts `cold_p12` / `cold_p13` must be relabeled `seen_*`: {t}"
        );
        if t.ends_with("\tseen_p12") {
            n_seen_p12 += 1;
        }
        if t.ends_with("\tseen_p13") {
            n_seen_p13 += 1;
        }
    }
    assert_eq!(n_seen_p12, 7, "expected 7 relabeled `seen_p12` rows");
    assert_eq!(n_seen_p13, 7, "expected 7 relabeled `seen_p13` rows");

    // 5. CLI `real-benchmark --json-out` schema check for Phase 13 ECE_10 / debiased_ece_10 / Bernoulli floor fields
    let dir = std::env::temp_dir().join(format!("nmr_p13_cal_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mini_manifest = dir.join("mini_manifest.tsv");
    fs::write(
        &mini_manifest,
        "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n\
         99999\tDummyDev\t60\t600\tBruker\tHSQC,HNCO,HNCACB,CBCACONH\tdev\tdev\n",
    )
    .unwrap();
    let json_out = dir.join("bench_out.json");
    let empty_root = dir.join("empty_root");
    let out = Command::new(nmrassign_bin())
        .args([
            "real-benchmark",
            "--manifest",
            mini_manifest.to_str().unwrap(),
            "--split",
            "dev",
            "--data-root",
            empty_root.to_str().unwrap(),
            "--json-out",
            json_out.to_str().unwrap(),
        ])
        .output()
        .expect("run real-benchmark");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json_txt = fs::read_to_string(&json_out).expect("read bench_out.json");
    for key in [
        "\"mean_protein_ece\":",
        "\"mean_protein_ece_10\":",
        "\"mean_protein_debiased_ece_10\":",
        "\"mean_protein_ece_10_bernoulli_floor\":",
        "\"mean_protein_ece_12\":",
        "\"debiased_ece_10\":",
        "\"ece_10_bernoulli_floor\":",
    ] {
        assert!(
            json_txt.contains(key),
            "CLI real-benchmark JSON missing {key}:\n{json_txt}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}
