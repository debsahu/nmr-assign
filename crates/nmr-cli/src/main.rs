//! Command-line interface (`nmrassign`) for GPU/CPU NMR backbone & sidechain resonance assignment,
//! 2D/3D classical & neural CNN peak picking, multi-protein benchmarking, 3D NOESY + torsion-angle
//! structure folding, CCPN NEF v1.1 interchange, HTML quality reporting, self-hosted HTTP service,
//! and Octet-Draw 2D chemical structure / IUPAC / 1D NMR analytics.
//!
//! Provenance: Original CLI implementing `docs/specs/01-data-io.md` through `docs/specs/08-service.md`
//! and Octet-Draw CLI workflows.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use nmr_core::dihedrals::{predict_dihedrals, validate_dihedrals_against_torsions};
use nmr_core::evaluate::{compare_shifts, score, to_reference};
use nmr_core::nef::{read_nef_file, write_nef_file, NefDocument};
use nmr_core::neural_picker::{
    run_e2e_blind_benchmark, E2eBenchmarkConfig, NeuralPicker, NeuralPickerConfig,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms, write_star};
use nmr_core::noesy::{
    compute_chain_torsions, parse_pdb_first_model, simulate_noesy_peaks, summarize_adrs,
    NoesySimConfig,
};
use nmr_core::octet_analyt::iupac_naming::{iupac_name_to_mol, mol_to_iupac_name};
use nmr_core::octet_analyt::jcamp_spec1d::{
    synthesize_1d_nmr, write_jcamp_nmr, JcampCompression, Synthesize1dOptions,
};
use nmr_core::octet_analyt::mass_props::compute_phys_chem_props;
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{
    build_peptide_2d, clean_2d, expand_abbreviations, kekulize, parse_condensed_formula,
    parse_inchi, parse_smiles, perceive_cip, read_cdx, read_cdxml, read_mol, read_oct, read_sdf,
    write_canonical_smiles, write_cdx, write_cdxml, write_inchi, write_inchikey, write_mol_v2000,
    write_oct, write_sdf, ChemDocument, Molecule, SdfRecord,
};
use nmr_core::octet_render::{
    render_scene_pdf, render_scene_png, render_scene_svg, DocumentStyle, JournalStyle, RenderScene,
};
use nmr_core::peaks::{read_peaks, write_peaks, Peak, EXPERIMENT_NAMES};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{compute_priors, one_to_three, Priors};
use nmr_core::report::{
    predict_csi_dihedrals, write_quality_report_html, ReportInput, StructureReportSummary,
};
use nmr_core::service::start_server;
use nmr_core::sidechain::{
    evaluate_sidechain_shifts, simulate_sidechain_peaks, solve_joint_backbone_sidechain,
    synthesize_all_atom_reference, SidechainSimConfig, SidechainSolverConfig,
};
use nmr_core::simulate::{render_hsqc_2d, simulate, SimConfig, SpectrumRenderConfig};
use nmr_core::solver::{assign, AnnealConfig, AssignResult, ScoreConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d,
    write_bruker_pdata_2d, write_nmrpipe_2d, Spectrum2D, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::structure::{
    calculate_structure_iterative, validate_ensemble, write_pdb_ensemble, StructureCalcConfig,
};
use nmr_core::topology::SidechainPriors;
use nmr_gpu::neural_gpu::{GpuNeuralPicker, GpuNeuralPickerConfig, PRETRAINED_WEIGHTS_3D_GPU};
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineKind {
    Gpu,
    Cpu,
}

fn usage() -> &'static str {
    "nmrassign (Rust + WebGPU)\n\
     Usage:\n\
       nmrassign benchmark       --star <ref.str> [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--missing 0.06] [--spurious 0.04] [--experiments HSQC,HNCACB,...] [--priors priors.json] [--seed 0] [-v]\n\
       nmrassign benchmark-suite [--manifest data/benchmark/manifest.tsv] [--engine gpu|cpu] [--runs 32] [--steps 1500] [--missing 0.06] [--spurious 0.04] [--seed 0]\n\
       nmrassign e2e-benchmark   [--star data/fixtures/synthetic_ubiquitin.str] [--noise 1.0] [--seed 42]\n\
       nmrassign assign          --peaks <peaks.tsv> (--seq <SEQ|fasta> | --star <ref.str>) --out <prefix> [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--priors priors.json]\n\
       nmrassign simulate        --star <ref.str> --out <peaks.tsv> [--spectra-dir <dir>] [--experiments ...] [--seed 0]\n\
       nmrassign pick            --hsqc <path> [--triple EXP=path ...] --out <peaks.tsv> [--method classical|neural] [--engine cpu|gpu] [--snr 5.0] [--prob 0.45]\n\
       nmrassign priors          --stars <dir_or_file> --out <priors.json>\n\
       nmrassign sidechain       --star <ref.str> [--peaks <peaks.tsv>] [--out <all_atom.str>] [--runs 16] [--steps 2500] [--seed 2026]\n\
       nmrassign structure       --star <shifts.str> [--noesy-peaks <noesy.tsv>] [--pdb-ref <ref.pdb>] [--out-pdb <ensemble.pdb>] [--conformers 10] [--seed 42] [--dihedrals-only]\n\
       nmrassign nef-import      --in <project.nef> [--out-star <shifts.str>] [--out-peaks <peaks.tsv>]\n\
       nmrassign nef-export      --star <shifts.str> [--peaks <peaks.tsv>] --out <project.nef> [--name <entry>]\n\
       nmrassign report          --star <shifts.str> [--peaks <peaks.tsv>] [--pdb <ensemble.pdb>] --out <report.html> [--title <name>]\n\
       nmrassign pipeline        (--star <entry.str> | --nef <project.nef>) --out-dir <dir> [--engine gpu|cpu] [--fold] [--pdb-ref <ref.pdb>]\n\
       nmrassign serve           [--bind 127.0.0.1:8080] [--storage ./jobs]\n\
       nmrassign chem [info]     (--smiles <smi> | --name <name> | --inchi <inchi> | --peptide <seq> | --input <file>) [--out <file.{cdxml,cdx,oct,mol,sdf,smi,svg,pdf,png,jdx}>] [--solvent CDCl3] [--mhz 400]\n"
}

fn run_solver(
    engine: EngineKind,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    cfg: &AnnealConfig,
) -> Result<(AssignResult, String), String> {
    match engine {
        EngineKind::Gpu => {
            let gpu = GpuEngine::new_headless()?;
            let label = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
            let res = gpu.assign_gpu(systems, sequence, priors, score_cfg, cfg)?;
            Ok((res, label))
        }
        EngineKind::Cpu => {
            let res = assign(systems, sequence, priors, score_cfg, cfg);
            Ok((res, "CPU (f64)".to_string()))
        }
    }
}

fn parse_engine(val: &str) -> Result<EngineKind, String> {
    match val.to_ascii_lowercase().as_str() {
        "gpu" | "wgpu" | "webgpu" => Ok(EngineKind::Gpu),
        "cpu" => Ok(EngineKind::Cpu),
        other => Err(format!("unknown --engine {other:?}; expected gpu or cpu")),
    }
}

fn load_priors(path: Option<&str>) -> Result<Priors, String> {
    match path {
        Some(p) => Priors::from_json(p),
        None => Ok(Priors::provisional()),
    }
}

fn cmd_benchmark(args: &[String]) -> Result<(), String> {
    let mut star: Option<String> = None;
    let mut experiments: Option<String> = None;
    let mut priors_path: Option<String> = None;
    let mut missing = 0.06f64;
    let mut spurious = 0.04f64;
    let mut engine = EngineKind::Gpu;
    let mut runs: Option<usize> = None;
    let mut steps = 1500usize;
    let mut assign_cost = 0.0f64;
    let mut seed = 0u64;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star = args.get(i).cloned();
            }
            "--experiments" => {
                i += 1;
                experiments = args.get(i).cloned();
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--missing" => {
                i += 1;
                missing = args
                    .get(i)
                    .ok_or("missing value for --missing")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--spurious" => {
                i += 1;
                spurious = args
                    .get(i)
                    .ok_or("missing value for --spurious")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing value for --engine")?)?;
            }
            "--runs" => {
                i += 1;
                runs = Some(
                    args.get(i)
                        .ok_or("missing value for --runs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .ok_or("missing value for --steps")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args
                    .get(i)
                    .ok_or("missing value for --assign-cost")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing value for --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown benchmark flag {other:?}")),
        }
        i += 1;
    }

    let star_path = star.ok_or("benchmark requires --star <ref.str>")?;
    let ref_data = read_star(&star_path, None)?;
    let exps: Vec<String> = match experiments {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().to_ascii_uppercase().replace(['(', ')'], ""))
            .collect(),
        None => EXPERIMENT_NAMES.iter().map(|s| (*s).to_string()).collect(),
    };

    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            experiments: exps.clone(),
            p_missing_peak: missing,
            spurious_rate: spurious,
            seed,
            ..SimConfig::default()
        },
    );

    let pri = load_priors(priors_path.as_deref())?;
    let default_runs = if engine == EngineKind::Gpu { 64 } else { 12 };
    let anneal_cfg = AnnealConfig {
        runs: runs.unwrap_or(default_runs),
        steps_per_system: steps,
        seed,
        ..AnnealConfig::default()
    };

    let score_cfg = ScoreConfig {
        assign_cost,
        ..ScoreConfig::default()
    };
    let t0 = Instant::now();
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
    let (res, engine_label) = run_solver(
        engine,
        &systems,
        &ref_data.sequence,
        &pri,
        &score_cfg,
        &anneal_cfg,
    )?;
    let elapsed = t0.elapsed().as_secs_f64();

    let m = score(&res, &truth, 0.8);
    println!(
        "protein: {} residues, {} peaks, {} spin systems",
        ref_data.sequence.len(),
        peaks.len(),
        systems.len()
    );
    println!(
        "engine: {} ({} replicas, {} steps/system)",
        engine_label, anneal_cfg.runs, anneal_cfg.steps_per_system
    );
    println!("experiments: {}", exps.join(", "));
    println!("{}", m.report());

    let (pred, _) = to_reference(
        &res,
        &systems,
        &ref_data.sequence,
        &ref_data.seq_ids,
        0.0,
        Some(&pri),
    );
    println!("shift agreement with reference (within tolerance / predicted):");
    for atom in ["H", "N", "CA", "CB", "C"] {
        if let Some(&(ok, n)) = compare_shifts(&pred, &ref_data).get(atom) {
            if n > 0 {
                println!("  {atom:2} {ok}/{n}");
            }
        }
    }
    println!("solver time: {:.3} s", elapsed);

    if verbose {
        let mut wrong = Vec::new();
        for (s, &k) in res.position.iter().enumerate() {
            if k >= 0 && k != truth[s] {
                wrong.push((s, k as usize, truth[s]));
            }
        }
        if !wrong.is_empty() {
            println!("misassigned spin systems (placed, true, confidence):");
            for (s, k, t) in wrong {
                let tl = if t >= 0 {
                    format!(
                        "{}{}",
                        ref_data.sequence[t as usize], ref_data.seq_ids[t as usize]
                    )
                } else {
                    "none".to_string()
                };
                println!(
                    "  {}{:<5} {:<8} {:.2}",
                    ref_data.sequence[k], ref_data.seq_ids[k], tl, res.confidence[s]
                );
            }
        }
    }
    Ok(())
}

fn cmd_simulate(args: &[String]) -> Result<(), String> {
    let mut star: Option<String> = None;
    let mut out: Option<String> = None;
    let mut experiments: Option<String> = None;
    let mut spectra_dir: Option<String> = None;
    let mut seed = 0u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star = args.get(i).cloned();
            }
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            "--experiments" => {
                i += 1;
                experiments = args.get(i).cloned();
            }
            "--spectra-dir" => {
                i += 1;
                spectra_dir = args.get(i).cloned();
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown simulate flag {other:?}")),
        }
        i += 1;
    }

    let star_path = star.ok_or("simulate requires --star <ref.str>")?;
    let out_path = PathBuf::from(out.ok_or("simulate requires --out <peaks.tsv>")?);
    let ref_data = read_star(&star_path, None)?;
    let exps: Vec<String> = match experiments {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().to_ascii_uppercase().replace(['(', ')'], ""))
            .collect(),
        None => EXPERIMENT_NAMES.iter().map(|s| (*s).to_string()).collect(),
    };

    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            experiments: exps,
            seed,
            ..SimConfig::default()
        },
    );
    write_peaks(&out_path, &peaks)?;

    let mut truth_tsv = String::from("root\tseq_id\n");
    for (idx, &t) in truth.iter().enumerate() {
        if t >= 0 {
            truth_tsv.push_str(&format!("{idx}\t{}\n", ref_data.seq_ids[t as usize]));
        } else {
            truth_tsv.push_str(&format!("{idx}\t.\n"));
        }
    }
    let truth_path = out_path.with_extension("truth.tsv");
    fs::write(&truth_path, truth_tsv).map_err(|e| format!("failed to write truth TSV: {e}"))?;
    println!("wrote {} peaks to {}", peaks.len(), out_path.display());

    if let Some(dir_str) = spectra_dir {
        let dir = Path::new(&dir_str);
        fs::create_dir_all(dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
        let rcfg = SpectrumRenderConfig {
            seed,
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&peaks, &rcfg);
        write_nmrpipe_2d(dir.join("hsqc.ft2"), &hsqc)?;
        write_bruker_pdata_2d(dir.join("bruker_hsqc_pdata"), &hsqc)?;
        println!(
            "rendered synthetic 2D HSQC to {} and {}",
            dir.join("hsqc.ft2").display(),
            dir.join("bruker_hsqc_pdata").display()
        );
    }
    Ok(())
}

fn cmd_assign(args: &[String]) -> Result<(), String> {
    let mut peaks_path: Option<String> = None;
    let mut seq_arg: Option<String> = None;
    let mut star_arg: Option<String> = None;
    let mut out_arg: Option<String> = None;
    let mut priors_path: Option<String> = None;
    let mut first_seq_id = 1i32;
    let mut engine = EngineKind::Gpu;
    let mut runs: Option<usize> = None;
    let mut steps = 1500usize;
    let mut assign_cost = 0.0f64;
    let mut seed = 0u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).cloned();
            }
            "--seq" => {
                i += 1;
                seq_arg = args.get(i).cloned();
            }
            "--star" => {
                i += 1;
                star_arg = args.get(i).cloned();
            }
            "--out" => {
                i += 1;
                out_arg = args.get(i).cloned();
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--first-seq-id" => {
                i += 1;
                first_seq_id = args
                    .get(i)
                    .ok_or("missing --first-seq-id")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing --engine")?)?;
            }
            "--runs" => {
                i += 1;
                runs = Some(
                    args.get(i)
                        .ok_or("missing --runs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .ok_or("missing --steps")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args
                    .get(i)
                    .ok_or("missing --assign-cost")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown assign flag {other:?}")),
        }
        i += 1;
    }

    let p_path = peaks_path.ok_or("assign requires --peaks <peaks.tsv>")?;
    let out_prefix = PathBuf::from(out_arg.ok_or("assign requires --out <prefix>")?);
    let (sequence, seq_ids) = if let Some(st) = star_arg {
        let r = read_star(st, None)?;
        (r.sequence, r.seq_ids)
    } else if let Some(sq) = seq_arg {
        let raw = if Path::new(&sq).exists() {
            let txt = fs::read_to_string(&sq).map_err(|e| format!("{e}"))?;
            txt.lines()
                .filter(|l| !l.trim_start().starts_with('>'))
                .map(|l| l.trim())
                .collect::<String>()
        } else {
            sq
        };
        let seq: Vec<String> = raw
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| one_to_three(c).to_string())
            .collect();
        let ids: Vec<i32> = (first_seq_id..first_seq_id + (seq.len() as i32)).collect();
        (seq, ids)
    } else {
        return Err("assign needs --seq or --star".to_string());
    };

    let peaks = read_peaks(p_path)?;
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
    let pri = load_priors(priors_path.as_deref())?;
    let default_runs = if engine == EngineKind::Gpu { 64 } else { 12 };
    let cfg = AnnealConfig {
        runs: runs.unwrap_or(default_runs),
        steps_per_system: steps,
        seed,
        ..AnnealConfig::default()
    };

    let score_cfg = ScoreConfig {
        assign_cost,
        ..ScoreConfig::default()
    };
    let (res, engine_label) = run_solver(engine, &systems, &sequence, &pri, &score_cfg, &cfg)?;
    let (pred, conf) = to_reference(&res, &systems, &sequence, &seq_ids, 0.0, Some(&pri));

    let str_out = out_prefix.with_extension("str");
    let tsv_out = out_prefix.with_extension("tsv");
    write_star(&str_out, &pred, Some(&conf))?;

    let mut tsv_lines = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
    for (&(k, ref atom), &v) in &pred.shifts {
        let c = conf.get(&k).copied().unwrap_or(0.0);
        tsv_lines.push(format!(
            "{}\t{}\t{}\t{:.3}\t{:.2}",
            seq_ids[k], sequence[k], atom, v, c
        ));
    }
    fs::write(&tsv_out, tsv_lines.join("\n") + "\n").map_err(|e| format!("{e}"))?;

    let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
    println!(
        "[{}] {} spin systems; {} residues assigned, {} with confidence >= 0.8",
        engine_label,
        systems.len(),
        conf.len(),
        n_hc
    );
    println!("wrote {} and {}", str_out.display(), tsv_out.display());
    Ok(())
}

fn load_2d_auto(path: &Path) -> Result<Spectrum2D, String> {
    if path.is_dir() {
        read_bruker_pdata_2d(path)
    } else if path.file_name().and_then(|s| s.to_str()) == Some("2rr") {
        read_bruker_pdata_2d(path.parent().unwrap_or(path))
    } else {
        read_nmrpipe_2d(path)
    }
}

fn load_3d_auto(path: &Path, exp: &str) -> Result<Spectrum3D, String> {
    if path.is_dir() {
        read_bruker_pdata_3d(path, exp)
    } else if path.file_name().and_then(|s| s.to_str()) == Some("3rrr") {
        read_bruker_pdata_3d(path.parent().unwrap_or(path), exp)
    } else {
        read_nmrpipe_3d(path, exp)
    }
}

fn cmd_pick(args: &[String]) -> Result<(), String> {
    let mut hsqc_path: Option<PathBuf> = None;
    let mut triples: Vec<(String, PathBuf)> = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut snr = 5.0f32;
    let mut prob = 0.45f32;
    let mut method = "classical".to_string();
    let mut engine = EngineKind::Cpu;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--hsqc" => {
                i += 1;
                hsqc_path = args.get(i).map(PathBuf::from);
            }
            "--triple" => {
                i += 1;
                let spec = args.get(i).ok_or("missing EXP=path for --triple")?;
                let (exp, p) = spec
                    .split_once('=')
                    .ok_or("expected EXP=path format for --triple")?;
                triples.push((
                    exp.trim().to_ascii_uppercase().replace(['(', ')'], ""),
                    PathBuf::from(p),
                ));
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--snr" => {
                i += 1;
                snr = args
                    .get(i)
                    .ok_or("missing --snr")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--prob" => {
                i += 1;
                prob = args
                    .get(i)
                    .ok_or("missing --prob")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--method" => {
                i += 1;
                method = args
                    .get(i)
                    .ok_or("missing value for --method")?
                    .to_ascii_lowercase();
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing value for --engine")?)?;
            }
            other => return Err(format!("unknown pick flag {other:?}")),
        }
        i += 1;
    }

    let hsqc_p = hsqc_path.ok_or("pick requires --hsqc <pdata_dir_or_ft2>")?;
    let out_p = out_path.ok_or("pick requires --out <peaks.tsv>")?;
    let hsqc_spec = load_2d_auto(&hsqc_p)?;

    let (roots, all_peaks, method_label) = match method.as_str() {
        "classical" => {
            let cfg = PickerConfig {
                snr_threshold: snr,
                ..PickerConfig::default()
            };
            let r = pick_hsqc_2d(&hsqc_spec, &cfg);
            let mut all = r.clone();
            for (exp, path) in &triples {
                let s3 = load_3d_auto(path, exp)?;
                all.extend(pick_triple_3d(&s3, &r, &cfg));
            }
            (r, all, "classical".to_string())
        }
        "neural" | "cnn" => {
            let model = NeuralPicker::pretrained();
            let ncfg = NeuralPickerConfig {
                prob_threshold: prob,
                ..NeuralPickerConfig::default()
            };
            let r = model.pick_2d(&hsqc_spec, &ncfg);
            let mut all = r.clone();
            match engine {
                EngineKind::Gpu if !triples.is_empty() => {
                    let gpu_picker = GpuNeuralPicker::new_headless(&PRETRAINED_WEIGHTS_3D_GPU)?;
                    let gcfg = GpuNeuralPickerConfig {
                        prob_threshold: prob,
                        ..GpuNeuralPickerConfig::default()
                    };
                    for (exp, path) in &triples {
                        let s3 = load_3d_auto(path, exp)?;
                        all.extend(gpu_picker.pick_triple_3d_gpu(&s3, exp, &r, &gcfg)?);
                    }
                    (
                        r,
                        all,
                        format!(
                            "neural (WebGPU: {}, {})",
                            gpu_picker.adapter_name, gpu_picker.backend_name
                        ),
                    )
                }
                _ => {
                    let cfg_3d = NeuralPickerConfig {
                        prob_threshold: prob,
                        hsqc_roots: r.clone(),
                        ..NeuralPickerConfig::default()
                    };
                    for (exp, path) in &triples {
                        let s3 = load_3d_auto(path, exp)?;
                        all.extend(model.pick_3d(&s3, exp, &cfg_3d));
                    }
                    (r, all, "neural (CPU)".to_string())
                }
            }
        }
        other => {
            return Err(format!(
                "unknown --method {other:?}; expected classical or neural"
            ))
        }
    };

    write_peaks(&out_p, &all_peaks)?;
    println!(
        "[{method_label}] picked {} HSQC roots and {} total peaks -> {}",
        roots.len(),
        all_peaks.len(),
        out_p.display()
    );
    Ok(())
}

fn cmd_priors(args: &[String]) -> Result<(), String> {
    let mut stars: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stars" => {
                i += 1;
                stars = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown priors flag {other:?}")),
        }
        i += 1;
    }

    let stars_p = stars.ok_or("priors requires --stars <dir_or_file>")?;
    let out_p = out.ok_or("priors requires --out <priors.json>")?;

    let mut paths: Vec<PathBuf> = Vec::new();
    if stars_p.is_dir() {
        for entry in fs::read_dir(&stars_p).map_err(|e| format!("{e}"))? {
            let entry = entry.map_err(|e| format!("{e}"))?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("str") {
                paths.push(p);
            }
        }
        paths.sort();
    } else {
        paths.push(stars_p);
    }

    let pri = compute_priors(&paths, 20);
    pri.to_json(&out_p)?;
    println!(
        "priors from {} files written to {}",
        paths.len(),
        out_p.display()
    );
    Ok(())
}

fn cmd_benchmark_suite(args: &[String]) -> Result<(), String> {
    let mut manifest = PathBuf::from("data/benchmark/manifest.tsv");
    let mut engine = EngineKind::Gpu;
    let mut runs: Option<usize> = None;
    let mut steps = 1500usize;
    let mut missing = 0.06f64;
    let mut spurious = 0.04f64;
    let mut seed = 0u64;
    let mut priors_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest = PathBuf::from(args.get(i).ok_or("missing --manifest")?);
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing --engine")?)?;
            }
            "--runs" => {
                i += 1;
                runs = Some(
                    args.get(i)
                        .ok_or("missing --runs")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .ok_or("missing --steps")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--missing" => {
                i += 1;
                missing = args
                    .get(i)
                    .ok_or("missing --missing")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--spurious" => {
                i += 1;
                spurious = args
                    .get(i)
                    .ok_or("missing --spurious")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            other => return Err(format!("unknown benchmark-suite flag {other:?}")),
        }
        i += 1;
    }

    let text = fs::read_to_string(&manifest)
        .map_err(|e| format!("failed to read {}: {e}", manifest.display()))?;
    let pri = load_priors(priors_path.as_deref())?;
    let default_runs = match engine {
        EngineKind::Gpu => 64,
        EngineKind::Cpu => 32,
    };
    let n_runs = runs.unwrap_or(default_runs);
    let maybe_gpu = if engine == EngineKind::Gpu {
        Some(GpuEngine::new_headless()?)
    } else {
        None
    };
    let engine_banner = match &maybe_gpu {
        Some(g) => format!("WebGPU ({}, {})", g.adapter_name, g.backend_name),
        None => "CPU (f64)".to_string(),
    };

    println!(
        "=== BMRB Multi-Protein Benchmark Suite ({}) ===",
        engine_banner
    );
    println!(
        "{:<10} {:<6} {:<10} {:>5} | {:>8} {:>8} {:>10} {:>8}",
        "BMRB", "PDB", "Fold", "Len", "Accuracy", "Recall", "HC-Prec", "Time(s)"
    );
    println!("{}", "-".repeat(78));

    let mut n_entries = 0usize;
    let mut sum_acc = 0.0f64;
    let mut sum_rec = 0.0f64;
    let mut sum_hc = 0.0f64;
    let mut sum_time = 0.0f64;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || (line_idx == 0 && line.starts_with("bmrb_id"))
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 {
            continue;
        }
        let (bmrb_id, pdb_id, fold_class, star_rel) = (cols[0], cols[1], cols[3], cols[5]);
        let ref_data = read_star(star_rel, None)
            .map_err(|e| format!("failed to read {star_rel} ({bmrb_id}): {e}"))?;
        let sim_cfg = SimConfig {
            p_missing_peak: missing,
            spurious_rate: spurious,
            seed: seed.wrapping_add(n_entries as u64),
            ..SimConfig::default()
        };
        let (peaks, truth) = simulate(&ref_data, &sim_cfg);
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
        let acfg = AnnealConfig {
            runs: n_runs,
            steps_per_system: steps,
            seed,
            ..AnnealConfig::default()
        };
        let scfg = ScoreConfig::default();

        let t0 = Instant::now();
        let res = match &maybe_gpu {
            Some(g) => g.assign_gpu(&systems, &ref_data.sequence, &pri, &scfg, &acfg)?,
            None => assign(&systems, &ref_data.sequence, &pri, &scfg, &acfg),
        };
        let dt = t0.elapsed().as_secs_f64();
        let m = score(&res, &truth, 0.8);

        println!(
            "{:<10} {:<6} {:<10} {:>5} | {:>7.1}% {:>7.1}% {:>9.1}% {:>7.2}s",
            bmrb_id,
            pdb_id,
            fold_class,
            ref_data.sequence.len(),
            m.accuracy() * 100.0,
            m.recall() * 100.0,
            m.hc_precision() * 100.0,
            dt
        );
        n_entries += 1;
        sum_acc += m.accuracy();
        sum_rec += m.recall();
        sum_hc += m.hc_precision();
        sum_time += dt;
    }

    if n_entries == 0 {
        return Err(format!("no valid entries found in {}", manifest.display()));
    }
    println!("{}", "-".repeat(78));
    println!(
        "MEAN ({n_entries} proteins)                  | {:>7.1}% {:>7.1}% {:>9.1}% {:>7.2}s total",
        (sum_acc / n_entries as f64) * 100.0,
        (sum_rec / n_entries as f64) * 100.0,
        (sum_hc / n_entries as f64) * 100.0,
        sum_time
    );
    Ok(())
}

fn cmd_e2e_benchmark(args: &[String]) -> Result<(), String> {
    let mut star = PathBuf::from("data/fixtures/synthetic_ubiquitin.str");
    let mut cfg = E2eBenchmarkConfig::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star = PathBuf::from(args.get(i).ok_or("missing --star")?);
            }
            "--noise" => {
                i += 1;
                let n: f32 = args
                    .get(i)
                    .ok_or("missing --noise")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                cfg.thermal_noise_2d = n;
                cfg.thermal_noise_3d = n;
            }
            "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown e2e-benchmark flag {other:?}")),
        }
        i += 1;
    }

    let ref_data = read_star(&star, None)?;
    let cmp = run_e2e_blind_benchmark(&ref_data, &cfg)?;
    println!("{}", cmp.format_table());
    println!("\nPer-experiment Peak F1 / Precision / Recall:");
    for m in &cmp.classical.per_experiment {
        if let Some(nm) = cmp
            .neural
            .per_experiment
            .iter()
            .find(|x| x.experiment == m.experiment)
        {
            println!(
                "  {:<10} | Classical F1={:.3} (P={:.1}%, R={:.1}%) | Neural F1={:.3} (P={:.1}%, R={:.1}%)",
                m.experiment,
                m.f1,
                m.precision * 100.0,
                m.recall * 100.0,
                nm.f1,
                nm.precision * 100.0,
                nm.recall * 100.0
            );
        }
    }
    Ok(())
}

fn cmd_sidechain(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut runs = 16usize;
    let mut steps = 2500usize;
    let mut seed = 2026u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--runs" => {
                i += 1;
                runs = args
                    .get(i)
                    .ok_or("missing --runs")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .ok_or("missing --steps")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown sidechain flag {other:?}")),
        }
        i += 1;
    }

    let star_p = star_path.ok_or("sidechain requires --star <entry.str>")?;
    let raw_ref = read_star_all_atoms(&star_p, None)?;
    let sc_priors = SidechainPriors::empirical_bmrb();
    let bb_priors = Priors::default();

    // If the input STAR only has backbone atoms, synthesize realistic all-atom reference shifts for benchmarking
    let has_sidechains = raw_ref
        .shifts
        .keys()
        .any(|(_, a)| !["H", "N", "CA", "CB", "C"].contains(&a.as_str()));
    let truth_all = if has_sidechains {
        raw_ref
    } else {
        synthesize_all_atom_reference(&raw_ref, &sc_priors, seed)
    };

    let (bb_peaks, sc_peaks, bb_truth_opt) = if let Some(pk_p) = &peaks_path {
        let loaded = read_peaks(pk_p)?;
        let (bb, sc): (Vec<Peak>, Vec<Peak>) = loaded
            .into_iter()
            .partition(|p| EXPERIMENT_NAMES.contains(&p.exp.as_str()));
        (bb, sc, None)
    } else {
        let (bb, bb_truth) = simulate(
            &truth_all,
            &SimConfig {
                p_missing_peak: 0.04,
                spurious_rate: 0.03,
                seed,
                ..SimConfig::default()
            },
        );
        let sc = simulate_sidechain_peaks(
            &truth_all,
            &SidechainSimConfig {
                include_aromatics: true,
                p_missing_peak: 0.04,
                spurious_rate: 0.02,
                seed,
                ..SidechainSimConfig::default()
            },
        );
        (bb, sc, Some(bb_truth))
    };

    let systems = build_spin_systems(&bb_peaks, &Tolerances::default(), 3, false)?;
    let t0 = Instant::now();
    let joint = solve_joint_backbone_sidechain(
        &systems,
        &truth_all.sequence,
        &truth_all.seq_ids,
        &bb_priors,
        &sc_priors,
        &sc_peaks,
        &ScoreConfig::default(),
        &AnnealConfig {
            runs,
            steps_per_system: steps,
            seed,
            ..AnnealConfig::default()
        },
        &SidechainSolverConfig::default(),
    );
    let dt = t0.elapsed().as_secs_f64();

    if let Some(bb_truth) = &bb_truth_opt {
        let bbm = score(&joint.backbone_result, bb_truth, 0.8);
        println!(
            "Backbone: accuracy={:.1}% ({}/{}), recall={:.1}%, hc_prec={:.1}%",
            bbm.accuracy() * 100.0,
            bbm.correct,
            bbm.assigned,
            bbm.recall() * 100.0,
            bbm.hc_precision() * 100.0
        );
    }
    if peaks_path.is_none() || has_sidechains {
        let scm =
            evaluate_sidechain_shifts(&joint.sidechain_result.reference, &truth_all, 0.40, 0.06);
        println!("{}", scm.summary());
    }
    println!(
        "Total assigned chemical shifts (backbone + sidechain): {} in {:.2}s",
        joint.sidechain_result.reference.shifts.len(),
        dt
    );

    if let Some(out_p) = out_path {
        let (_, conf) = to_reference(
            &joint.backbone_result,
            &systems,
            &truth_all.sequence,
            &truth_all.seq_ids,
            0.0,
            Some(&bb_priors),
        );
        write_star(&out_p, &joint.sidechain_result.reference, Some(&conf))?;
        println!("Wrote all-atom NMR-STAR shifts to {}", out_p.display());
    }
    Ok(())
}

fn cmd_structure(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut noesy_path: Option<PathBuf> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut out_pdb_path: Option<PathBuf> = None;
    let mut n_conformers = 10usize;
    let mut seed = 42u64;
    let mut dihedrals_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--noesy-peaks" => {
                i += 1;
                noesy_path = args.get(i).map(PathBuf::from);
            }
            "--pdb-ref" => {
                i += 1;
                pdb_ref_path = args.get(i).map(PathBuf::from);
            }
            "--out-pdb" => {
                i += 1;
                out_pdb_path = args.get(i).map(PathBuf::from);
            }
            "--conformers" => {
                i += 1;
                n_conformers = args
                    .get(i)
                    .ok_or("missing --conformers")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--dihedrals-only" => dihedrals_only = true,
            other => return Err(format!("unknown structure flag {other:?}")),
        }
        i += 1;
    }

    let star_p = star_path.ok_or("structure requires --star <shifts.str>")?;
    let shifts = read_star_all_atoms(&star_p, None)?;
    let dih_preds = predict_dihedrals(&shifts);
    let n_confident = dih_preds.iter().filter(|d| d.confidence >= 0.5).count();
    println!(
        "Predicted backbone (phi, psi) dihedrals for {} residues ({} confident):",
        shifts.sequence.len(),
        n_confident
    );

    let pdb_ref = match &pdb_ref_path {
        Some(p) => Some(parse_pdb_first_model(p)?),
        None => None,
    };
    if let Some(ref_chain) = &pdb_ref {
        let dm =
            validate_dihedrals_against_torsions(&dih_preds, &compute_chain_torsions(ref_chain));
        println!(
            "  vs Reference PDB: ordered MAE phi={:.1} deg, psi={:.1} deg (SS 3-state accuracy={:.1}%)",
            dm.phi_mae_ordered_deg,
            dm.psi_mae_ordered_deg,
            dm.q3_accuracy * 100.0
        );
    }

    if dihedrals_only {
        return Ok(());
    }

    let noesy_peaks: Vec<Peak> = if let Some(np) = &noesy_path {
        if np.extension().and_then(|s| s.to_str()) == Some("nef") {
            let doc = read_nef_file(np)?;
            doc.to_peaks()
                .into_iter()
                .filter(|p| p.exp.starts_with("NOESY"))
                .collect()
        } else {
            read_peaks(np)?
                .into_iter()
                .filter(|p| p.exp.starts_with("NOESY"))
                .collect()
        }
    } else if let Some(ref_chain) = &pdb_ref {
        simulate_noesy_peaks(
            ref_chain,
            &shifts,
            &NoesySimConfig {
                seed,
                ..NoesySimConfig::default()
            },
        )
    } else {
        Vec::new()
    };

    let mut cfg = StructureCalcConfig::default();
    cfg.n_select = n_conformers.max(1);
    cfg.n_replicas = cfg.n_replicas.max(cfg.n_select);
    cfg.seed = seed;

    let t0 = Instant::now();
    let ens = calculate_structure_iterative(&shifts, &noesy_peaks, &cfg);
    let dt = t0.elapsed().as_secs_f64();
    let adr_sum = summarize_adrs(&ens.final_adrs);
    let val = validate_ensemble(
        &ens.conformers,
        &ens.final_adrs,
        &ens.dihedral_restraints,
        pdb_ref.as_ref(),
    );

    println!(
        "Folded {}-conformer torsion-angle ensemble in {:.2}s ({} NOESY peaks -> {} ADRs, {} unambiguous, {} long-range):",
        ens.conformers.len(),
        dt,
        noesy_peaks.len(),
        adr_sum.total_restraints,
        adr_sum.unambiguous_restraints,
        adr_sum.long_range_unambiguous
    );
    println!(
        "  Pairwise ordered backbone RMSD: {:.2} A | Ramachandran favored+allowed: {:.1}% (favored {:.1}%, outlier {:.1}%)",
        val.ensemble_pairwise_rmsd_ordered,
        val.ramachandran_favored_pct + val.ramachandran_allowed_pct,
        val.ramachandran_favored_pct,
        val.ramachandran_outlier_pct
    );
    println!(
        "  Mean violations: {:.2} NOE > 0.5 A, {:.2} dihedral > 5 deg",
        val.mean_distance_violations_gt_0_5a, val.mean_dihedral_violations_gt_5deg
    );
    if let (Some(low_e), Some(best_r), Some(mean_r)) = (
        val.lowest_energy_rmsd_ordered_to_ref,
        val.best_conformer_rmsd_ordered_to_ref,
        val.mean_backbone_rmsd_ordered_to_ref,
    ) {
        println!(
            "  Backbone RMSD vs Reference PDB (ordered): lowest-E={:.2} A, best={:.2} A, ensemble-mean={:.2} A",
            low_e, best_r, mean_r
        );
    }

    if let Some(out_p) = out_pdb_path {
        write_pdb_ensemble(&out_p, &ens.conformers, "NMRASSIGN TORSION ENSEMBLE")?;
        println!(
            "Wrote {}-model PDB ensemble to {}",
            ens.conformers.len(),
            out_p.display()
        );
    }
    Ok(())
}

fn cmd_nef_import(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_star: Option<PathBuf> = None;
    let mut out_peaks: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "--nef" => {
                i += 1;
                in_path = args.get(i).map(PathBuf::from);
            }
            "--out-star" => {
                i += 1;
                out_star = args.get(i).map(PathBuf::from);
            }
            "--out-peaks" => {
                i += 1;
                out_peaks = args.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown nef-import flag {other:?}")),
        }
        i += 1;
    }

    let nef_p = in_path.ok_or("nef-import requires --in <project.nef>")?;
    let doc = read_nef_file(&nef_p)?;
    let ref_data = doc.to_reference()?;
    let peaks = doc.to_peaks();

    println!(
        "Imported CCPN NEF v1.1 '{}' ({} residues, {} shifts, {} spectrum lists / {} peaks, {} distance restraint lists, {} dihedral lists)",
        doc.project_name,
        ref_data.sequence.len(),
        ref_data.shifts.len(),
        doc.peak_lists.len(),
        peaks.len(),
        doc.distance_restraint_lists.len(),
        doc.dihedral_restraint_lists.len()
    );

    if let Some(sp) = out_star {
        write_star(&sp, &ref_data, None)?;
        println!("  Wrote NMR-STAR v3 shifts -> {}", sp.display());
    }
    if let Some(pp) = out_peaks {
        write_peaks(&pp, &peaks)?;
        println!(
            "  Wrote TSV peaks ({} rows) -> {}",
            peaks.len(),
            pp.display()
        );
    }
    Ok(())
}

fn cmd_nef_export(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut entry_name = "nmr_assign_project".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--name" => {
                i += 1;
                entry_name = args.get(i).ok_or("missing --name")?.clone();
            }
            other => return Err(format!("unknown nef-export flag {other:?}")),
        }
        i += 1;
    }

    let star_p = star_path.ok_or("nef-export requires --star <shifts.str>")?;
    let out_p = out_path.ok_or("nef-export requires --out <project.nef>")?;
    let ref_data = read_star_all_atoms(&star_p, None)?;
    let peaks = match peaks_path {
        Some(pp) => read_peaks(&pp)?,
        None => Vec::new(),
    };

    let doc = NefDocument::from_reference_and_peaks(&entry_name, &ref_data, None, &peaks);
    write_nef_file(&out_p, &doc)?;
    println!(
        "Exported CCPN NEF v1.1 '{}' ({} residues, {} shifts, {} peaks) -> {}",
        entry_name,
        ref_data.sequence.len(),
        ref_data.shifts.len(),
        peaks.len(),
        out_p.display()
    );
    Ok(())
}

fn cmd_report(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut pdb_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut title: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--pdb" => {
                i += 1;
                pdb_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--title" => {
                i += 1;
                title = args.get(i).cloned();
            }
            other => return Err(format!("unknown report flag {other:?}")),
        }
        i += 1;
    }

    let star_p = star_path.ok_or("report requires --star <shifts.str>")?;
    let out_p = out_path.ok_or("report requires --out <report.html>")?;
    let ref_data = read_star_all_atoms(&star_p, None)?;

    let mut conf = BTreeMap::new();
    for k in 0..ref_data.sequence.len() {
        if ref_data.shifts.contains_key(&(k, "H".to_string()))
            && ref_data.shifts.contains_key(&(k, "N".to_string()))
        {
            conf.insert(k, 0.95);
        }
    }

    let proj_name = title.unwrap_or_else(|| {
        star_p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("nmr_project")
            .to_string()
    });
    let mut input = ReportInput::new(proj_name, ref_data.clone(), conf);
    if let Some(pp) = peaks_path {
        let pks = read_peaks(&pp)?;
        input.n_peaks = Some(pks.len());
        if let Ok(sys) = build_spin_systems(&pks, &Tolerances::default(), 3, false) {
            input.n_spin_systems = Some(sys.len());
        }
    }
    if let Some(pdb_p) = pdb_path {
        let pdb_txt = fs::read_to_string(&pdb_p)
            .map_err(|e| format!("failed to read {}: {e}", pdb_p.display()))?;
        input.pdb_coordinates = Some(pdb_txt);
    } else {
        input.dihedrals = predict_csi_dihedrals(&ref_data);
    }

    write_quality_report_html(&out_p, &input)?;
    println!(
        "Wrote self-contained HTML + SVG quality report -> {}",
        out_p.display()
    );
    Ok(())
}

fn cmd_pipeline(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut nef_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut engine = EngineKind::Gpu;
    let mut fold = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--star" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--nef" => {
                i += 1;
                nef_path = args.get(i).map(PathBuf::from);
            }
            "--out-dir" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            "--pdb-ref" => {
                i += 1;
                pdb_ref_path = args.get(i).map(PathBuf::from);
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing --engine")?)?;
            }
            "--fold" => fold = true,
            other => return Err(format!("unknown pipeline flag {other:?}")),
        }
        i += 1;
    }

    let dir = out_dir.ok_or("pipeline requires --out-dir <dir>")?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let (base_ref, peaks, project_name) = if let Some(np) = &nef_path {
        let doc = read_nef_file(np)?;
        let r = doc.to_reference()?;
        let pks = doc.to_peaks();
        let pks = if pks.is_empty() {
            simulate(&r, &SimConfig::default()).0
        } else {
            pks
        };
        (r, pks, doc.project_name)
    } else if let Some(sp) = &star_path {
        let r = read_star_all_atoms(sp, None)?;
        let (pks, _) = simulate(&r, &SimConfig::default());
        let name = sp
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pipeline_project")
            .to_string();
        (r, pks, name)
    } else {
        return Err("pipeline requires --star <entry.str> or --nef <project.nef>".to_string());
    };

    let pri = Priors::default();
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
    let acfg = AnnealConfig {
        runs: if engine == EngineKind::Gpu { 64 } else { 32 },
        steps_per_system: 1500,
        seed: 0,
        ..AnnealConfig::default()
    };
    let t0 = Instant::now();
    let (res, engine_label) = run_solver(
        engine,
        &systems,
        &base_ref.sequence,
        &pri,
        &ScoreConfig::default(),
        &acfg,
    )
    .or_else(|_| {
        run_solver(
            EngineKind::Cpu,
            &systems,
            &base_ref.sequence,
            &pri,
            &ScoreConfig::default(),
            &acfg,
        )
    })?;
    let dt = t0.elapsed().as_secs_f64();

    let (pred_ref, conf) = to_reference(
        &res,
        &systems,
        &base_ref.sequence,
        &base_ref.seq_ids,
        0.0,
        Some(&pri),
    );
    let star_out = dir.join("shifts.str");
    let nef_out = dir.join("result.nef");
    let html_out = dir.join("report.html");

    write_star(&star_out, &pred_ref, Some(&conf))?;
    let nef_doc =
        NefDocument::from_reference_and_peaks(&project_name, &pred_ref, Some(&conf), &peaks);
    write_nef_file(&nef_out, &nef_doc)?;

    let mut rep = ReportInput::new(&project_name, pred_ref.clone(), conf.clone());
    rep.n_peaks = Some(peaks.len());
    rep.n_spin_systems = Some(systems.len());
    rep.solver_score = Some(res.score);
    rep.elapsed_secs = Some(dt);
    rep.engine_label = Some(engine_label.clone());
    rep.systems_and_assignment = Some((systems, res));
    rep.dihedrals = predict_csi_dihedrals(&pred_ref);

    if fold {
        let pdb_ref = match &pdb_ref_path {
            Some(p) => Some(parse_pdb_first_model(p)?),
            None => None,
        };
        let noesy_pks = match &pdb_ref {
            Some(rc) => simulate_noesy_peaks(rc, &pred_ref, &NoesySimConfig::default()),
            None => Vec::new(),
        };
        let ens =
            calculate_structure_iterative(&pred_ref, &noesy_pks, &StructureCalcConfig::default());
        let val = validate_ensemble(
            &ens.conformers,
            &ens.final_adrs,
            &ens.dihedral_restraints,
            pdb_ref.as_ref(),
        );
        let pdb_out = dir.join("ensemble.pdb");
        write_pdb_ensemble(&pdb_out, &ens.conformers, &project_name)?;
        rep.pdb_coordinates = fs::read_to_string(&pdb_out).ok();
        rep.structure_summary = Some(StructureReportSummary {
            n_conformers: ens.conformers.len(),
            backbone_rmsd_angstrom: val
                .lowest_energy_rmsd_ordered_to_ref
                .unwrap_or(val.ensemble_pairwise_rmsd_ordered),
            heavy_atom_rmsd_angstrom: None,
            n_distance_restraints: ens.final_adrs.len(),
            n_ambiguous_restraints: ens.final_adrs.iter().filter(|a| !a.is_unambiguous).count(),
            n_dihedral_restraints: ens.dihedral_restraints.len(),
            distance_violations_gt_05a: val.mean_distance_violations_gt_0_5a.round() as usize,
            dihedral_violations_gt_5deg: val.mean_dihedral_violations_gt_5deg.round() as usize,
            ramachandran_favored_pct: val.ramachandran_favored_pct,
        });
    }

    write_quality_report_html(&html_out, &rep)?;
    println!(
        "[{engine_label}] Pipeline '{}' completed in {:.2}s -> {}, {}, {}",
        project_name,
        dt,
        star_out.display(),
        nef_out.display(),
        html_out.display()
    );
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let mut bind_addr = "127.0.0.1:8080".to_string();
    let mut storage = PathBuf::from("./jobs");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind_addr = args.get(i).ok_or("missing --bind")?.clone();
            }
            "--storage" => {
                i += 1;
                storage = PathBuf::from(args.get(i).ok_or("missing --storage")?);
            }
            other => return Err(format!("unknown serve flag {other:?}")),
        }
        i += 1;
    }

    println!(
        "Starting nmrassign HTTP service & browser dashboard at http://{bind_addr} (storage: {})",
        storage.display()
    );
    start_server(&bind_addr, storage)
}

fn cmd_chem(args: &[String]) -> Result<(), String> {
    let slice = if args.first().map(String::as_str) == Some("info") {
        &args[1..]
    } else {
        args
    };

    let mut smiles_in: Option<String> = None;
    let mut name_in: Option<String> = None;
    let mut inchi_in: Option<String> = None;
    let mut peptide_in: Option<String> = None;
    let mut input_file: Option<PathBuf> = None;
    let mut out_file: Option<PathBuf> = None;
    let mut solvent = "CDCl3".to_string();
    let mut mhz = 400.0f64;

    let mut i = 0;
    while i < slice.len() {
        match slice[i].as_str() {
            "--smiles" => {
                i += 1;
                smiles_in = slice.get(i).cloned();
            }
            "--name" => {
                i += 1;
                name_in = slice.get(i).cloned();
            }
            "--inchi" => {
                i += 1;
                inchi_in = slice.get(i).cloned();
            }
            "--peptide" => {
                i += 1;
                peptide_in = slice.get(i).cloned();
            }
            "--input" | "--in" => {
                i += 1;
                input_file = slice.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_file = slice.get(i).map(PathBuf::from);
            }
            "--solvent" => {
                i += 1;
                solvent = slice.get(i).ok_or("missing --solvent")?.clone();
            }
            "--mhz" => {
                i += 1;
                mhz = slice
                    .get(i)
                    .ok_or("missing --mhz")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown chem flag {other:?}")),
        }
        i += 1;
    }

    let mut doc_opt: Option<ChemDocument> = None;
    let mut mol: Molecule = if let Some(smi) = smiles_in {
        let mut m = parse_smiles(&smi).map_err(|e| format!("SMILES error: {e}"))?;
        clean_2d(&mut m);
        m
    } else if let Some(nm) = name_in {
        if let Ok(amol) = iupac_name_to_mol(&nm) {
            let mut m = parse_smiles(&amol.to_smiles())
                .map_err(|e| format!("SMILES conversion error: {e}"))?;
            clean_2d(&mut m);
            m
        } else {
            let mut m = parse_condensed_formula(&nm)
                .map_err(|e| format!("unrecognized chemical name '{nm}': {e}"))?;
            clean_2d(&mut m);
            m
        }
    } else if let Some(inc) = inchi_in {
        let mut m = parse_inchi(&inc).map_err(|e| format!("InChI error: {e}"))?;
        clean_2d(&mut m);
        m
    } else if let Some(pep) = peptide_in {
        build_peptide_2d(&pep).map_err(|e| format!("peptide error: {e}"))?
    } else if let Some(inp) = &input_file {
        let ext = inp
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "cdx" => {
                let bytes = fs::read(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let doc = read_cdx(&bytes).map_err(|e| format!("CDX error: {e}"))?;
                let first = doc.molecules.first().cloned().unwrap_or_default();
                doc_opt = Some(doc);
                first
            }
            "cdxml" => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let doc = read_cdxml(&txt).map_err(|e| format!("CDXML error: {e}"))?;
                let first = doc.molecules.first().cloned().unwrap_or_default();
                doc_opt = Some(doc);
                first
            }
            "oct" => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let doc = read_oct(&txt).map_err(|e| format!("OCT error: {e}"))?;
                let first = doc.molecules.first().cloned().unwrap_or_default();
                doc_opt = Some(doc);
                first
            }
            "mol" => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                read_mol(&txt).map_err(|e| format!("MOL error: {e}"))?
            }
            "sdf" => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let recs = read_sdf(&txt).map_err(|e| format!("SDF error: {e}"))?;
                recs.into_iter()
                    .next()
                    .ok_or("SDF file has no records")?
                    .molecule
            }
            "inchi" => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let mut m = parse_inchi(txt.trim()).map_err(|e| format!("InChI error: {e}"))?;
                clean_2d(&mut m);
                m
            }
            _ => {
                let txt =
                    fs::read_to_string(inp).map_err(|e| format!("read {}: {e}", inp.display()))?;
                let tok = txt.split_whitespace().next().unwrap_or("");
                let mut m = parse_smiles(tok).map_err(|e| format!("SMILES error: {e}"))?;
                clean_2d(&mut m);
                m
            }
        }
    } else {
        return Err(
            "chem requires --smiles <smi>, --name <name>, --inchi <inchi>, --peptide <seq>, or --input <file>"
                .to_string(),
        );
    };

    let _ = expand_abbreviations(&mut mol);
    let _ = kekulize(&mut mol);
    let amol = AnalytMol::from(&mol);
    let props = compute_phys_chem_props(&amol)?;
    let pin =
        mol_to_iupac_name(&amol).unwrap_or_else(|_| "(multi-component / non-PIN)".to_string());
    let can_smi = write_canonical_smiles(&mol);
    let inchi = write_inchi(&mol);
    let inchikey = write_inchikey(&mol);
    let cip = perceive_cip(&mol);
    let h1 = predict_nmr(&amol, NmrNucleus::H1, Some(&solvent), None);
    let c13 = predict_nmr(&amol, NmrNucleus::C13, Some(&solvent), None);

    println!("=== Octet Chemical Structure & 1D NMR Report ===");
    println!("IUPAC Name:       {pin}");
    println!("Canonical SMILES: {can_smi}");
    println!("InChI:            {inchi}");
    println!("InChIKey:         {inchikey}");
    println!(
        "Formula:          {}  (MW: {:.2} g/mol, Exact Mass: {:.5} Da, RDBE: {:.1})",
        props.formula, props.molecular_weight, props.monoisotopic_mass, props.rdbe
    );
    println!("Composition:      {}", props.elemental_analysis_text);
    println!(
        "PhysChem:         cLogP={:.2}, TPSA={:.1} A^2, HBD={}, HBA={}, RotBonds={}, Lipinski Ro5={}",
        props.clogp,
        props.tpsa,
        props.h_bond_donors,
        props.h_bond_acceptors,
        props.rotatable_bonds,
        if props.lipinski_ro5_pass { "PASS" } else { "FAIL" }
    );
    if !cip.atoms.is_empty() || !cip.bonds.is_empty() {
        let mut tags = Vec::new();
        for (&idx, lbl) in &cip.atoms {
            tags.push(format!("atom {}={}", idx + 1, lbl.as_str()));
        }
        for (&idx, lbl) in &cip.bonds {
            tags.push(format!("bond {}={}", idx + 1, lbl.as_str()));
        }
        if !tags.is_empty() {
            println!("CIP Stereocenters: {}", tags.join(", "));
        }
    }
    println!("\n{}", h1.acs_text(mhz));
    println!("{}", c13.acs_text(mhz));

    if let Some(out_p) = out_file {
        let ext = out_p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let doc = doc_opt.unwrap_or_else(|| {
            let mut d = ChemDocument::new();
            d.molecules.push(mol.clone());
            d
        });
        match ext.as_str() {
            "cdxml" => fs::write(&out_p, write_cdxml(&doc)).map_err(|e| format!("{e}"))?,
            "cdx" => fs::write(&out_p, write_cdx(&doc)).map_err(|e| format!("{e}"))?,
            "oct" => {
                let s = write_oct(&doc).map_err(|e| format!("{e}"))?;
                fs::write(&out_p, s).map_err(|e| format!("{e}"))?;
            }
            "mol" => {
                let s = write_mol_v2000(&mol).map_err(|e| format!("{e}"))?;
                fs::write(&out_p, s).map_err(|e| format!("{e}"))?;
            }
            "sdf" => {
                let mut pmap = BTreeMap::new();
                pmap.insert("IUPAC_NAME".to_string(), pin);
                pmap.insert("INCHIKEY".to_string(), inchikey);
                let rec = SdfRecord {
                    molecule: mol,
                    properties: pmap,
                };
                fs::write(&out_p, write_sdf(&[rec])).map_err(|e| format!("{e}"))?;
            }
            "smi" => {
                fs::write(&out_p, format!("{can_smi}\n")).map_err(|e| format!("{e}"))?;
            }
            "inchi" => {
                fs::write(&out_p, format!("{inchi}\n")).map_err(|e| format!("{e}"))?;
            }
            "jdx" | "dx" => {
                let syn_opts = Synthesize1dOptions {
                    mhz_1h: mhz,
                    ..Synthesize1dOptions::default()
                };
                let spec = synthesize_1d_nmr(&h1, &syn_opts);
                fs::write(&out_p, write_jcamp_nmr(&spec, JcampCompression::Affn))
                    .map_err(|e| format!("{e}"))?;
            }
            "svg" => {
                let sc = RenderScene::from(&doc);
                let svg = render_scene_svg(&sc, &DocumentStyle::from_preset(JournalStyle::Acs1996));
                fs::write(&out_p, svg).map_err(|e| format!("{e}"))?;
            }
            "pdf" => {
                let sc = RenderScene::from(&doc);
                let pdf = render_scene_pdf(&sc, &DocumentStyle::from_preset(JournalStyle::Acs1996));
                fs::write(&out_p, pdf).map_err(|e| format!("{e}"))?;
            }
            "png" => {
                let sc = RenderScene::from(&doc);
                let png =
                    render_scene_png(&sc, &DocumentStyle::from_preset(JournalStyle::Acs1996), 2.0);
                fs::write(&out_p, png).map_err(|e| format!("{e}"))?;
            }
            other => {
                return Err(format!(
                    "unsupported --out extension .{other} (expected .cdxml, .cdx, .oct, .mol, .sdf, .smi, .inchi, .jdx, .svg, .pdf, .png)"
                ))
            }
        }
        println!("\nExported -> {}", out_p.display());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{}", usage());
        return;
    }
    let res = match args[0].as_str() {
        "benchmark" => cmd_benchmark(&args[1..]),
        "benchmark-suite" => cmd_benchmark_suite(&args[1..]),
        "e2e-benchmark" => cmd_e2e_benchmark(&args[1..]),
        "simulate" => cmd_simulate(&args[1..]),
        "assign" => cmd_assign(&args[1..]),
        "pick" => cmd_pick(&args[1..]),
        "priors" => cmd_priors(&args[1..]),
        "sidechain" => cmd_sidechain(&args[1..]),
        "structure" => cmd_structure(&args[1..]),
        "nef-import" => cmd_nef_import(&args[1..]),
        "nef-export" => cmd_nef_export(&args[1..]),
        "report" => cmd_report(&args[1..]),
        "pipeline" => cmd_pipeline(&args[1..]),
        "serve" => cmd_serve(&args[1..]),
        "chem" => cmd_chem(&args[1..]),
        other => Err(format!("unknown subcommand {other:?}\n\n{}", usage())),
    };
    if let Err(err) = res {
        eprintln!("error: {err}");
        process::exit(1);
    }
}
