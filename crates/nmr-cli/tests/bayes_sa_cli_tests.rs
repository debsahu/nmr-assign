//! CLI integration tests for the Bayesian specific-heat adaptive simulated annealing solver
//! (`--solver bayes-sa` / `bayes_sa`, `T13.0b`).
//!
//! Provenance: Clean-room verification of `nmrassign assign`, `nmrassign pipeline`, and
//! `nmrassign benchmark --realistic` across CPU (`nmr_core::bayes_sa`) and WebGPU
//! (`nmr_gpu::bayes_sa_gpu`), including hard locks (`--locks`), `--no-auto-ref`, tolerances,
//! and telemetry propagation (`t_crit`, `max_cv`, `mu_null`, `qc_report`, `applied_offsets`,
//! `detailed_offsets`) into NMR-STAR, JSON, NEF v1.1, and HTML quality reports.

use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use nmr_core::nef::read_nef_str;
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::write_peaks;
use nmr_core::simulate::{simulate, SimConfig};

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

fn fixture_star_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn temp_test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nmrassign_bayes_sa_{}_{}", tag, process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\":");
    let idx = json.find(&needle)?;
    let rest = json[idx + needle.len()..].trim_start();
    let end = rest.find([',', '\n', '}', ']']).unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

fn assert_valid_json_syntax(json: &str) {
    let trimmed = json.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}'),
        "expected JSON object, got:\n{json}"
    );
    if let Ok(out) = Command::new("python3")
        .args(["-c", "import json,sys; json.loads(sys.stdin.read())"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                let _ = stdin.write_all(json.as_bytes());
            }
            child.wait_with_output()
        })
    {
        assert!(
            out.status.success(),
            "invalid RFC 8259 JSON:\n{}\n{}",
            String::from_utf8_lossy(&out.stderr),
            json
        );
    }
}

#[test]
fn test_cli_assign_bayes_sa_cpu_locks_json_nef_and_html_report() {
    let dir = temp_test_dir("assign_cpu");
    let star_path = fixture_star_path();
    let ref_data = read_star(&star_path, None).expect("read ubiquitin fixture");

    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            p_missing_peak: 0.0,
            spurious_rate: 0.0,
            seed: 42,
            ..SimConfig::default()
        },
    );
    let peaks_path = dir.join("peaks.tsv");
    write_peaks(&peaks_path, &peaks).expect("write peaks.tsv");

    // Pick a known true (spin_system_idx, 1-based seq_id) pair to exercise `--locks`
    let (lock_ss, lock_seq_idx) = truth
        .iter()
        .enumerate()
        .find_map(|(ss, &k)| (k > 0).then_some((ss, k as usize)))
        .expect("at least one assignable root");
    let lock_seq_id = ref_data.seq_ids[lock_seq_idx];
    let lock_arg = format!("{lock_ss}:{lock_seq_id}");

    let out_star = dir.join("assigned_cpu.str");
    let out_json = dir.join("assigned_cpu.json");
    let out_nef = dir.join("assigned_cpu.nef");
    let out_html = dir.join("assigned_cpu.html");

    let output = Command::new(nmrassign_bin())
        .args([
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--solver",
            "bayes-sa",
            "--engine",
            "cpu",
            "--runs",
            "8",
            "--steps",
            "250",
            "--seed",
            "42",
            "--locks",
            &lock_arg,
            "--tol-ca",
            "0.22",
            "--tol-cb",
            "0.28",
            "--out",
            out_star.to_str().unwrap(),
            "--json-out",
            out_json.to_str().unwrap(),
            "--nef-out",
            out_nef.to_str().unwrap(),
            "--report-out",
            out_html.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign assign --solver bayes-sa --engine cpu");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "assign cpu failed:\n{stderr}");
    assert!(
        stderr.contains("bayes-sa") && stderr.contains("T_crit=") && stderr.contains("max_Cv="),
        "expected Bayesian SA telemetry in stderr, got:\n{stderr}"
    );

    // 1. Verify NMR-STAR output metadata and assigned shifts
    let star_txt = fs::read_to_string(&out_star).expect("read assigned_cpu.str");
    assert!(star_txt.contains("bayes-sa"));
    assert!(star_txt.contains("bayes_sa(t_crit="));
    assert!(star_txt.contains("max_cv="));
    assert!(star_txt.contains("mu_null="));
    let solved_ref = read_star(&out_star, None).expect("parse solved NMR-STAR");
    assert!(
        solved_ref
            .shifts
            .contains_key(&(lock_seq_idx, "H".to_string())),
        "locked residue index {lock_seq_idx} (seq_id {lock_seq_id}) must be assigned"
    );

    // 2. Verify JSON telemetry output
    let json_txt = fs::read_to_string(&out_json).expect("read assigned_cpu.json");
    assert_valid_json_syntax(&json_txt);
    assert!(json_txt.contains("\"solver\": \"bayes-sa\""));
    assert!(json_txt.contains("\"qc_report\": {"));
    assert!(json_txt.contains("\"verdict\": \""));
    assert!(json_txt.contains("\"applied_offsets\": {"));
    assert!(json_txt.contains("\"detailed_offsets\": {"));
    assert!(json_txt.contains("\"cooling_trajectory\": ["));
    assert!(json_txt.contains("\"stage\": 0"));
    assert!(extract_json_f64(&json_txt, "t_crit").is_some_and(|v| v > 0.0));
    assert!(extract_json_f64(&json_txt, "max_cv").is_some_and(|v| v >= 0.0));
    assert!(extract_json_f64(&json_txt, "mu_null").is_some_and(f64::is_finite));

    // 3. Verify NEF v1.1 telemetry header and clean round-trip via `read_nef_str` + `nef-import`
    let nef_txt = fs::read_to_string(&out_nef).expect("read assigned_cpu.nef");
    assert!(
        nef_txt.contains("bayes_sa(t_crit=") && nef_txt.contains("mu_null="),
        "expected Bayesian SA header comment in NEF output, got:\n{nef_txt}"
    );
    let nef_doc = read_nef_str(&nef_txt).expect("parse NEF v1.1 output");
    let nef_ref = nef_doc.to_reference().expect("convert NEF to Reference");
    assert_eq!(nef_ref.sequence.len(), ref_data.sequence.len());

    let roundtrip_star = dir.join("roundtrip.str");
    let nef_imp = Command::new(nmrassign_bin())
        .args([
            "nef-import",
            "--nef",
            out_nef.to_str().unwrap(),
            "--out-star",
            roundtrip_star.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign nef-import");
    assert!(
        nef_imp.status.success(),
        "nef-import failed: {}",
        String::from_utf8_lossy(&nef_imp.stderr)
    );

    // 4. Verify HTML quality report contains Bayesian SA engine label & telemetry
    let html_txt = fs::read_to_string(&out_html).expect("read assigned_cpu.html");
    assert!(html_txt.contains("bayes-sa"));
    assert!(html_txt.contains("T_crit="));
    assert!(html_txt.contains("max_Cv="));
    assert!(html_txt.contains("mu_null="));

    // 5. Verify `bayes_sa` underscore alias + `--no-auto-ref` zeroes applied offsets
    let out_no_ref_json = dir.join("assigned_no_ref.json");
    let out_no_ref_star = dir.join("assigned_no_ref.str");
    let res_no_ref = Command::new(nmrassign_bin())
        .args([
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--solver",
            "bayes_sa",
            "--engine",
            "cpu",
            "--no-auto-ref",
            "--runs",
            "4",
            "--steps",
            "150",
            "--seed",
            "7",
            "--out",
            out_no_ref_star.to_str().unwrap(),
            "--json-out",
            out_no_ref_json.to_str().unwrap(),
        ])
        .output()
        .expect("run assign --solver bayes_sa --no-auto-ref");
    assert!(
        res_no_ref.status.success(),
        "bayes_sa --no-auto-ref failed: {}",
        String::from_utf8_lossy(&res_no_ref.stderr)
    );
    let no_ref_txt = fs::read_to_string(&out_no_ref_json).unwrap();
    assert_valid_json_syntax(&no_ref_txt);
    assert!(
        no_ref_txt.contains(
            "\"applied_offsets\": { \"h\": 0.000000, \"n\": 0.000000, \"ca\": 0.000000, \"cb\": 0.000000, \"c\": 0.000000 }"
        ),
        "expected zero applied_offsets under --no-auto-ref, got:\n{no_ref_txt}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_assign_and_pipeline_bayes_sa_gpu_and_cpu() {
    let dir = temp_test_dir("assign_pipeline_gpu");
    let star_path = fixture_star_path();
    let ref_data = read_star(&star_path, None).expect("read ubiquitin fixture");

    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            p_missing_peak: 0.0,
            spurious_rate: 0.0,
            seed: 42,
            ..SimConfig::default()
        },
    );
    let peaks_path = dir.join("peaks.tsv");
    write_peaks(&peaks_path, &peaks).expect("write peaks.tsv");

    // 1. Test `nmrassign assign --solver bayes-sa --engine gpu`
    let out_gpu_star = dir.join("assigned_gpu.str");
    let out_gpu_json = dir.join("assigned_gpu.json");
    let out_gpu = Command::new(nmrassign_bin())
        .args([
            "assign",
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--star",
            star_path.to_str().unwrap(),
            "--solver",
            "bayes-sa",
            "--engine",
            "gpu",
            "--runs",
            "16",
            "--seed",
            "42",
            "--out",
            out_gpu_star.to_str().unwrap(),
            "--json-out",
            out_gpu_json.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign assign --solver bayes-sa --engine gpu");

    let gpu_stderr = String::from_utf8_lossy(&out_gpu.stderr);
    if !out_gpu.status.success() && gpu_stderr.contains("no suitable WebGPU adapter") {
        eprintln!("Skipping WebGPU CLI assertion (no suitable WebGPU adapter in environment)");
    } else {
        assert!(
            out_gpu.status.success(),
            "assign --solver bayes-sa --engine gpu failed:\n{gpu_stderr}"
        );
        let gpu_json_txt = fs::read_to_string(&out_gpu_json).unwrap();
        assert_valid_json_syntax(&gpu_json_txt);
        assert!(gpu_json_txt.contains("\"solver\": \"bayes-sa\""));
        assert!(extract_json_f64(&gpu_json_txt, "t_crit").is_some_and(|v| v > 0.0));
    }

    // 2. Test `nmrassign pipeline --solver bayes-sa --engine cpu`
    let pipe_dir = dir.join("pipe_cpu");
    let out_pipe = Command::new(nmrassign_bin())
        .args([
            "pipeline",
            "--star",
            star_path.to_str().unwrap(),
            "--peaks",
            peaks_path.to_str().unwrap(),
            "--solver",
            "bayes-sa",
            "--engine",
            "cpu",
            "--runs",
            "8",
            "--steps",
            "200",
            "--seed",
            "42",
            "--out-dir",
            pipe_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign pipeline --solver bayes-sa --engine cpu");
    assert!(
        out_pipe.status.success(),
        "pipeline --solver bayes-sa failed:\n{}",
        String::from_utf8_lossy(&out_pipe.stderr)
    );
    assert!(pipe_dir.join("shifts.str").exists());
    assert!(pipe_dir.join("result.nef").exists());
    assert!(pipe_dir.join("qc_report.json").exists());
    assert!(pipe_dir.join("bayes_sa_report.json").exists());
    assert!(pipe_dir.join("report.html").exists());

    let pipe_bayes_txt = fs::read_to_string(pipe_dir.join("bayes_sa_report.json")).unwrap();
    assert_valid_json_syntax(&pipe_bayes_txt);
    assert!(pipe_bayes_txt.contains("\"solver\": \"bayes-sa\""));
    assert!(extract_json_f64(&pipe_bayes_txt, "t_crit").is_some_and(|v| v > 0.0));
    let pipe_html = fs::read_to_string(pipe_dir.join("report.html")).unwrap();
    assert!(pipe_html.contains("bayes-sa") && pipe_html.contains("T_crit="));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_benchmark_realistic_two_seeds_bayes_sa_cpu_and_gpu() {
    let star_path = fixture_star_path();

    for engine in ["cpu", "gpu"] {
        let out = Command::new(nmrassign_bin())
            .args([
                "benchmark",
                "--star",
                star_path.to_str().unwrap(),
                "--realistic",
                "--seeds",
                "2",
                "--solver",
                "bayes-sa",
                "--engine",
                engine,
                "--runs",
                "8",
            ])
            .output()
            .expect("run nmrassign benchmark --realistic --seeds 2 --solver bayes-sa");

        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !out.status.success() && engine == "gpu" && stderr.contains("no suitable WebGPU adapter")
        {
            continue;
        }
        assert!(
            out.status.success(),
            "benchmark --solver bayes-sa --engine {engine} failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("(sim=realistic_bruker, seeds=2)")
                && stdout.contains("bayes-sa)")
                && stdout.contains("bayes-sa: mean T_crit=")
                && stdout.contains("accuracy (correct/assigned)")
                && stdout.contains("recall   (correct/assignable)")
                && stdout.contains("high-confidence (>=0.8) precision"),
            "unexpected benchmark output for engine={engine}:\n{stdout}"
        );
    }
}
