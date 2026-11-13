//! End-to-end integration tests for `nmrassign methyl` (`crates/nmr-cli/src/methyl_cmd.rs`
//! and top-level `nmrassign` binary subcommand dispatch).
//!
//! Provenance: Clean-room integration test suite citing:
//! - Tugarinov, Hwang, Ollerenshaw & Kay (*J. Am. Chem. Soc.* 2003, 125, 10120–10121 & 10420–10428)
//! - Gans et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962)
//! - Ruschak & Kay (*J. Biomol. NMR* 2010, 46, 75–87)
//! - Xu, Lin, Ho & Yang (*J. Am. Chem. Soc.* 2009, 131, 9480–9481)
//! - Chao, Shi, Masterson & Veglia (*J. Magn. Reson.* 2012, 214, 103–110)
//! - Pritišanac et al. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533)

#[path = "../src/methyl_cmd.rs"]
mod methyl_cmd;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use nmr_core::methyl_trosy::{
    extract_hmcmcbca_from_reference, extract_methyl_structure_from_pdb_file,
    extract_true_site_shifts, pick_methyl_hmqc_2d, pick_methyl_noesy_3d, simulate_methyl_hmqc_2d,
    simulate_methyl_noesy_3d, write_hmcmcbca_tsv, write_methyl_hmqc_tsv, write_methyl_noesy_tsv,
    MethylLabelingScheme, MethylPickerConfig, MethylSimConfig,
};
use nmr_core::nmrstar::read_star_all_atoms;
use nmr_core::spectrum::{write_ucsf_2d, write_ucsf_3d};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn methyl_cli_help_and_validation_errors() {
    methyl_cmd::run_methyl_cli(&["--help".to_string()]).expect("help should succeed");
    assert!(methyl_cmd::run_methyl_cli(&[]).is_err());
    assert!(methyl_cmd::run_methyl_cli(&["--unknown-flag".to_string()]).is_err());
}

#[test]
fn methyl_cli_end_to_end_tsv_and_ucsf_spectra() {
    let root = repo_root();
    let pdb_path = root.join("data/pdb_refs/1d3z.pdb");
    let star_ref_path = root.join("data/fixtures/ubiquitin_bmr5387_methyl.str");
    let true_ref = read_star_all_atoms(&star_ref_path, None).expect("read ubiquitin methyl star");

    let dir = env::temp_dir().join(format!("nmr_methyl_cli_test_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let model = extract_methyl_structure_from_pdb_file(&pdb_path, &MethylLabelingScheme::Ilvmat)
        .expect("extract ubiquitin structure");
    let true_hc = extract_true_site_shifts(&model, &true_ref);

    let sim_cfg = MethylSimConfig {
        hmqc_nx: 384,
        hmqc_ny: 256,
        noesy_nx: 192,
        noesy_ny: 192,
        noesy_nz: 192,
        ..MethylSimConfig::default()
    };

    let hmqc_spec = simulate_methyl_hmqc_2d(&true_hc, &sim_cfg);
    let noesy_spec = simulate_methyl_noesy_3d(&model, &true_hc, &sim_cfg);

    // 1. Test direct UCSF 2D/3D volumetric spectrum auto-picking + GPU assignment
    let hmqc_ucsf = dir.join("hmqc.ucsf");
    let noesy_ucsf = dir.join("noesy_cch.ucsf");
    let hmcm_tsv = dir.join("hmcmcbca.tsv");
    let out_gpu_star = dir.join("ubq_gpu_out.str");

    write_ucsf_2d(&hmqc_ucsf, &hmqc_spec).expect("write 2D ucsf");
    write_ucsf_3d(&noesy_ucsf, &noesy_spec).expect("write 3D ucsf");
    let hmcm_rows = extract_hmcmcbca_from_reference(&model, &true_ref);
    write_hmcmcbca_tsv(&hmcm_tsv, &hmcm_rows).expect("write hmcm tsv");

    let engine_flag = if nmr_gpu::GpuEngine::new_headless().is_ok() {
        "gpu"
    } else {
        "cpu"
    };

    let cli_args_ucsf = vec![
        "--pdb".to_string(),
        pdb_path.to_string_lossy().to_string(),
        "--hmqc".to_string(),
        hmqc_ucsf.to_string_lossy().to_string(),
        "--noesy".to_string(),
        noesy_ucsf.to_string_lossy().to_string(),
        "--labeling".to_string(),
        "ILVMAT".to_string(),
        "--hmcmcbca".to_string(),
        hmcm_tsv.to_string_lossy().to_string(),
        "--engine".to_string(),
        engine_flag.to_string(),
        "--seeds".to_string(),
        "1".to_string(),
        "-o".to_string(),
        out_gpu_star.to_string_lossy().to_string(),
    ];
    methyl_cmd::run_methyl_cli(&cli_args_ucsf).expect("methyl CLI on UCSF spectra");

    let assigned_ref =
        read_star_all_atoms(&out_gpu_star, None).expect("parse written methyl NMR-STAR v3");
    assert_eq!(assigned_ref.sequence.len(), 76);
    assert!(
        assigned_ref.shifts.len() >= 120,
        "expected >= 120 assigned methyl C+H atom rows in STAR output, got {}",
        assigned_ref.shifts.len()
    );

    // 2. Test TSV peak list input + CPU engine
    let picker_cfg = MethylPickerConfig::default();
    let picked_roots = pick_methyl_hmqc_2d(&hmqc_spec, &picker_cfg);
    let picked_noes = pick_methyl_noesy_3d(&noesy_spec, &picked_roots, &picker_cfg);
    let hmqc_tsv = dir.join("hmqc_peaks.tsv");
    let noesy_tsv = dir.join("noesy_peaks.tsv");
    let out_cpu_star = dir.join("ubq_cpu_out.str");
    write_methyl_hmqc_tsv(&hmqc_tsv, &picked_roots).expect("write hmqc tsv");
    write_methyl_noesy_tsv(&noesy_tsv, &picked_noes).expect("write noesy tsv");

    let cli_args_tsv = vec![
        "--pdb".to_string(),
        pdb_path.to_string_lossy().to_string(),
        "--hmqc".to_string(),
        hmqc_tsv.to_string_lossy().to_string(),
        "--noesy".to_string(),
        noesy_tsv.to_string_lossy().to_string(),
        "--labeling".to_string(),
        "ILVMAT".to_string(),
        "--hmcmcbca".to_string(),
        hmcm_tsv.to_string_lossy().to_string(),
        "--engine".to_string(),
        "cpu".to_string(),
        "--seeds".to_string(),
        "1".to_string(),
        "-o".to_string(),
        out_cpu_star.to_string_lossy().to_string(),
    ];
    methyl_cmd::run_methyl_cli(&cli_args_tsv).expect("methyl CLI on TSV peak lists");
    assert!(out_cpu_star.exists());

    // 3. Exercise the compiled `nmrassign methyl` subcommand binary end-to-end
    let bin = env!("CARGO_BIN_EXE_nmrassign");
    let help_out = process::Command::new(bin)
        .args(["methyl", "--help"])
        .output()
        .expect("spawn nmrassign methyl --help");
    assert!(
        help_out.status.success(),
        "nmrassign methyl --help failed: {}",
        String::from_utf8_lossy(&help_out.stderr)
    );
    let help_stdout = String::from_utf8_lossy(&help_out.stdout);
    assert!(help_stdout.contains("Usage: nmrassign methyl"));

    let out_bin_star = dir.join("ubq_bin_out.str");
    let bin_run = process::Command::new(bin)
        .args([
            "methyl",
            "--pdb",
            &pdb_path.to_string_lossy(),
            "--hmqc",
            &hmqc_tsv.to_string_lossy(),
            "--noesy",
            &noesy_tsv.to_string_lossy(),
            "--labeling",
            "ILVMAT",
            "--hmcmcbca",
            &hmcm_tsv.to_string_lossy(),
            "--engine",
            "cpu",
            "--seeds",
            "1",
            "-o",
            &out_bin_star.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign methyl end-to-end");
    assert!(
        bin_run.status.success(),
        "nmrassign methyl failed: {}",
        String::from_utf8_lossy(&bin_run.stderr)
    );
    let bin_ref = read_star_all_atoms(&out_bin_star, None).expect("parse binary methyl STAR");
    assert_eq!(bin_ref.sequence.len(), 76);
    assert!(bin_ref.shifts.len() >= 120);

    // 4. Test P1-6 headless CPU fallback when `GpuEngine::new_headless()` fails with `Err("no adapter")`
    //    for both default (`--engine auto` / omitted) and explicit `--engine gpu`.
    for explicit_engine in [None, Some("auto"), Some("gpu")] {
        let out_fb_star = dir.join(format!(
            "ubq_fallback_{}.str",
            explicit_engine.unwrap_or("default")
        ));
        let mut fb_args = vec![
            "--pdb".to_string(),
            pdb_path.to_string_lossy().to_string(),
            "--hmqc".to_string(),
            hmqc_tsv.to_string_lossy().to_string(),
            "--noesy".to_string(),
            noesy_tsv.to_string_lossy().to_string(),
            "--labeling".to_string(),
            "ILVMAT".to_string(),
            "--hmcmcbca".to_string(),
            hmcm_tsv.to_string_lossy().to_string(),
            "--seeds".to_string(),
            "1".to_string(),
            "-o".to_string(),
            out_fb_star.to_string_lossy().to_string(),
        ];
        if let Some(eng) = explicit_engine {
            fb_args.push("--engine".to_string());
            fb_args.push(eng.to_string());
        }
        let warn_opt =
            methyl_cmd::run_methyl_cli_with_gpu_init(&fb_args, || Err("no adapter".to_string()))
                .expect(
                    "headless CPU fallback must succeed when WebGPU returns Err(\"no adapter\")",
                );
        let warn = warn_opt.expect("expected warning string on simulated WebGPU failure");
        assert!(
            warn.starts_with("warning:") && warn.contains("no adapter") && warn.contains("CPU"),
            "unexpected fallback warning: {warn}"
        );
        let fb_ref =
            read_star_all_atoms(&out_fb_star, None).expect("parse fallback methyl NMR-STAR v3");
        assert_eq!(fb_ref.sequence.len(), 76);
        assert!(fb_ref.shifts.len() >= 120);
    }

    let _ = fs::remove_dir_all(&dir);
}
