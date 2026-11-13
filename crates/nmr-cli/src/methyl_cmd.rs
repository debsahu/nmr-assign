//! `nmrassign methyl` command implementation for supramolecular `13C`-methyl TROSY (`ILVMAT`)
//! peak picking and WebGPU/CPU structure-based NOE contact-graph assignment
//! (`docs/specs/13-methyl-trosy.md` §3.4).
//!
//! Provenance: Clean-room CLI orchestration built on `nmr_core::{methyl_trosy, methyl_solver}`
//! and `nmr_gpu::methyl_gpu`, citing:
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125,
//!   10120–10121 & 10420–10428).
//! - Gans, P. et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962).
//! - Ruschak, A. M. & Kay, L. E. (*J. Biomol. NMR* 2010, 46, 75–87).
//! - Xu, Y., Lin, Z., Ho, C. & Yang, D. (*J. Am. Chem. Soc.* 2009, 131, 9480–9481).
//! - Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G. (*J. Magn. Reson.* 2012, 214, 103–110).
//! - Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J.
//!   & Kay, L. E. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::methyl_solver::{
    assign_methyl_cpu, build_methyl_contact_problem, write_methyl_star_assignments,
    MethylAnnealConfig, MethylAssignResult, MethylContactProblem, MethylScoreConfig,
};
use nmr_core::methyl_trosy::{
    attach_hmcmcbca_records, extract_methyl_structure_from_pdb_file, pick_methyl_hmqc_2d,
    pick_methyl_noesy_3d, read_hmcmcbca_tsv, read_methyl_hmqc_tsv, read_methyl_noesy_tsv,
    MethylLabelingScheme, MethylNoePeak, MethylPickerConfig, MethylRootPeak,
};
use nmr_core::spectrum::{
    is_ucsf_file, read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d,
    read_ucsf_2d, read_ucsf_3d, Spectrum2D, Spectrum3D,
};
use nmr_gpu::methyl_gpu::assign_methyl_gpu;
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComputeBackend {
    Gpu,
    Cpu,
}

fn print_methyl_usage() {
    println!(
        "Usage: nmrassign methyl --pdb <model.pdb> --hmqc <2d_or_tsv> --noesy <3d_or_tsv> \\\n  \
         [--labeling ILV-proS|ILV-proR|ILV-racemic|ILVMAT|<RES:ATOM,...>] \\\n  \
         [--hmcmcbca <hmcm.tsv>] [--engine gpu|cpu] [--seeds <N>] [-o <out.str>]"
    );
}

fn is_probably_text_tsv(path: &Path) -> bool {
    if path.is_dir() || is_ucsf_file(path) {
        return false;
    }
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let low = ext.to_ascii_lowercase();
        if matches!(low.as_str(), "tsv" | "txt" | "list" | "pk") {
            return true;
        }
        if matches!(low.as_str(), "ucsf" | "ft2" | "ft3" | "pipe" | "dat") {
            return false;
        }
    }
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let head = &bytes[..bytes.len().min(512)];
    !head.contains(&0) && std::str::from_utf8(head).is_ok()
}

fn load_spectrum_2d_auto(path: &Path) -> Result<Spectrum2D, String> {
    if path.is_dir() {
        return read_bruker_pdata_2d(path);
    }
    if path.file_name().and_then(|s| s.to_str()) == Some("2rr") {
        if let Some(parent) = path.parent() {
            return read_bruker_pdata_2d(parent);
        }
    }
    if is_ucsf_file(path) {
        return read_ucsf_2d(path);
    }
    read_nmrpipe_2d(path)
}

fn load_spectrum_3d_auto(path: &Path) -> Result<Spectrum3D, String> {
    if path.is_dir() {
        return read_bruker_pdata_3d(path, "METHYL_NOESY_3D");
    }
    if path.file_name().and_then(|s| s.to_str()) == Some("3rrr") {
        if let Some(parent) = path.parent() {
            return read_bruker_pdata_3d(parent, "METHYL_NOESY_3D");
        }
    }
    if is_ucsf_file(path) {
        return read_ucsf_3d(path, "METHYL_NOESY_3D");
    }
    read_nmrpipe_3d(path, "METHYL_NOESY_3D")
}

fn load_or_pick_hmqc_roots(
    path: &Path,
    picker_cfg: &MethylPickerConfig,
) -> Result<Vec<MethylRootPeak>, String> {
    if is_probably_text_tsv(path) {
        read_methyl_hmqc_tsv(path)
    } else {
        let spec = load_spectrum_2d_auto(path)?;
        Ok(pick_methyl_hmqc_2d(&spec, picker_cfg))
    }
}

fn load_or_pick_noesy_peaks(
    path: &Path,
    roots: &[MethylRootPeak],
    picker_cfg: &MethylPickerConfig,
) -> Result<Vec<MethylNoePeak>, String> {
    if is_probably_text_tsv(path) {
        read_methyl_noesy_tsv(path)
    } else {
        let cube = load_spectrum_3d_auto(path)?;
        Ok(pick_methyl_noesy_3d(&cube, roots, picker_cfg))
    }
}

/// Format a concise summary report of a completed methyl contact-graph assignment.
pub fn format_methyl_summary(
    pdb_path: &Path,
    scheme: &MethylLabelingScheme,
    engine_label: &str,
    prob: &MethylContactProblem,
    elapsed_ms: f64,
    result: &MethylAssignResult,
) -> String {
    let n_assigned = result.position.iter().filter(|&&m| m >= 0).count();
    let n_hc = result
        .position
        .iter()
        .zip(&result.confidence)
        .filter(|&(&m, &c)| m >= 0 && c >= 0.80)
        .count();
    format!(
        "Methyl-TROSY Contact-Graph Assignment\n\
         -------------------------------------\n\
         Structure: {}  |  Labeling: {}  |  Engine: {}\n\
         Active structural methyl sites: {}  |  Observed HMQC roots: {}  |  NOE graph edges: {}\n\
         Assigned roots: {}/{}  |  High-confidence (>=0.80): {}  |  Score: {:.4} nats/root  ({:.1} ms)\n",
        pdb_path.display(),
        scheme.label(),
        engine_label,
        prob.m_len,
        prob.s_len,
        prob.obs_edges.len(),
        n_assigned,
        prob.s_len,
        n_hc,
        result.score,
        elapsed_ms
    )
}

/// Execute `nmrassign methyl` from command-line arguments (excluding `argv[0]` and subcommand token).
pub fn run_methyl_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_methyl_usage();
        return Ok(());
    }

    let mut pdb_path: Option<PathBuf> = None;
    let mut hmqc_path: Option<PathBuf> = None;
    let mut noesy_path: Option<PathBuf> = None;
    let mut hmcm_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut labeling = MethylLabelingScheme::Ilvmat;
    let mut backend = ComputeBackend::Gpu;
    let mut seeds: usize = 1;
    let mut seed_base: u64 = 2026;

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: usize, name: &str| -> Result<&str, String> {
            args.get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("missing argument for `{name}`"))
        };
        match flag {
            "--pdb" => {
                pdb_path = Some(PathBuf::from(next_val(i, "--pdb")?));
                i += 2;
            }
            "--hmqc" => {
                hmqc_path = Some(PathBuf::from(next_val(i, "--hmqc")?));
                i += 2;
            }
            "--noesy" => {
                noesy_path = Some(PathBuf::from(next_val(i, "--noesy")?));
                i += 2;
            }
            "--hmcmcbca" => {
                hmcm_path = Some(PathBuf::from(next_val(i, "--hmcmcbca")?));
                i += 2;
            }
            "--labeling" => {
                labeling = MethylLabelingScheme::parse(next_val(i, "--labeling")?)?;
                i += 2;
            }
            "--engine" => {
                let v = next_val(i, "--engine")?;
                backend = match v.to_ascii_lowercase().as_str() {
                    "gpu" | "webgpu" => ComputeBackend::Gpu,
                    "cpu" => ComputeBackend::Cpu,
                    other => {
                        return Err(format!(
                            "unknown `--engine {other}` (expected `gpu` or `cpu`)"
                        ))
                    }
                };
                i += 2;
            }
            "--seeds" => {
                let v = next_val(i, "--seeds")?;
                seeds = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid `--seeds {v}`: {e}"))?
                    .max(1);
                i += 2;
            }
            "--seed" => {
                let v = next_val(i, "--seed")?;
                seed_base = v
                    .parse::<u64>()
                    .map_err(|e| format!("invalid `--seed {v}`: {e}"))?;
                i += 2;
            }
            "-o" | "--out" | "--output" => {
                out_path = Some(PathBuf::from(next_val(i, flag)?));
                i += 2;
            }
            other => {
                return Err(format!(
                    "unrecognized `nmrassign methyl` flag `{other}` (use `--help` for usage)"
                ));
            }
        }
    }

    let pdb = pdb_path.ok_or_else(|| "missing required `--pdb <model.pdb>`".to_string())?;
    let hmqc = hmqc_path.ok_or_else(|| "missing required `--hmqc <2d_or_tsv>`".to_string())?;
    let noesy = noesy_path.ok_or_else(|| "missing required `--noesy <3d_or_tsv>`".to_string())?;

    let t0 = Instant::now();
    let model = extract_methyl_structure_from_pdb_file(&pdb, &labeling)?;
    let picker_cfg = MethylPickerConfig::default();
    let mut roots = load_or_pick_hmqc_roots(&hmqc, &picker_cfg)?;

    if let Some(ref hmcm_file) = hmcm_path {
        let hmcm_rows = read_hmcmcbca_tsv(hmcm_file)?;
        attach_hmcmcbca_records(&mut roots, &hmcm_rows, 0.045, 0.14);
    }

    let noe_peaks = load_or_pick_noesy_peaks(&noesy, &roots, &picker_cfg)?;
    let score_cfg = MethylScoreConfig::default();
    let prob = build_methyl_contact_problem(&model, &roots, &noe_peaks, &score_cfg);

    let anneal_cfg = MethylAnnealConfig {
        replicas: 64usize.saturating_mul(seeds),
        seed: seed_base,
        ..MethylAnnealConfig::default()
    };

    let (result, engine_label) = match backend {
        ComputeBackend::Gpu => {
            let gpu = GpuEngine::new_headless()?;
            let label = format!("WebGPU ({} / {})", gpu.adapter_name, gpu.backend_name);
            let res = assign_methyl_gpu(&gpu, &prob, &anneal_cfg)?;
            (res, label)
        }
        ComputeBackend::Cpu => {
            let res = assign_methyl_cpu(&prob, &anneal_cfg);
            (res, "CPU (multi-threaded)".to_string())
        }
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    print!(
        "{}",
        format_methyl_summary(
            &pdb,
            &model.scheme,
            &engine_label,
            &prob,
            elapsed_ms,
            &result,
        )
    );

    if let Some(ref out_star) = out_path {
        write_methyl_star_assignments(out_star, &model, &roots, &result)?;
        println!(
            "Wrote assigned methyl NMR-STAR v3 shifts to {}",
            out_star.display()
        );
    }

    Ok(())
}
