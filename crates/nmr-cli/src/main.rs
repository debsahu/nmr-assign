//! Command-line interface for GPU/CPU NMR backbone resonance assignment, simulation,
//! classical spectrum peak picking, and BMRB prior computation.
//!
//! Provenance: Original CLI implementing `docs/specs/01-data-io.md`, `docs/specs/02-peak-picking.md`,
//! `docs/specs/03-assignment.md`, and `docs/specs/05-benchmark.md`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use nmr_core::evaluate::{compare_shifts, score, to_reference};
use nmr_core::nmrstar::{read_star, write_star};
use nmr_core::peaks::{read_peaks, write_peaks, EXPERIMENT_NAMES};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{compute_priors, one_to_three, Priors};
use nmr_core::simulate::{render_hsqc_2d, simulate, SimConfig, SpectrumRenderConfig};
use nmr_core::solver::{assign, AnnealConfig, AssignResult, ScoreConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d,
    write_bruker_pdata_2d, write_nmrpipe_2d, Spectrum2D, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineKind {
    Gpu,
    Cpu,
}

fn usage() -> &'static str {
    "nmrassign (Rust + WebGPU)\n\
     Usage:\n\
       nmrassign benchmark --star <ref.str> [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--missing 0.06] [--spurious 0.04] [--experiments HSQC,HNCACB,...] [--priors priors.json] [--seed 0] [-v]\n\
       nmrassign assign    --peaks <peaks.tsv> (--seq <SEQ|fasta> | --star <ref.str>) --out <prefix> [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--priors priors.json]\n\
       nmrassign simulate  --star <ref.str> --out <peaks.tsv> [--spectra-dir <dir>] [--experiments ...] [--seed 0]\n\
       nmrassign pick      --hsqc <path> [--triple EXP=path ...] --out <peaks.tsv> [--snr 5.0]\n\
       nmrassign priors    --stars <dir_or_file> --out <priors.json>\n"
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
                missing = args.get(i).ok_or("missing value for --missing")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--spurious" => {
                i += 1;
                spurious = args.get(i).ok_or("missing value for --spurious")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing value for --engine")?)?;
            }
            "--runs" => {
                i += 1;
                runs = Some(args.get(i).ok_or("missing value for --runs")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--steps" => {
                i += 1;
                steps = args.get(i).ok_or("missing value for --steps")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args.get(i).ok_or("missing value for --assign-cost")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).ok_or("missing value for --seed")?.parse().map_err(|e| format!("{e}"))?;
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
    let (res, engine_label) = run_solver(engine, &systems, &ref_data.sequence, &pri, &score_cfg, &anneal_cfg)?;
    let elapsed = t0.elapsed().as_secs_f64();

    let m = score(&res, &truth, 0.8);
    println!(
        "protein: {} residues, {} peaks, {} spin systems",
        ref_data.sequence.len(),
        peaks.len(),
        systems.len()
    );
    println!("engine: {} ({} replicas, {} steps/system)", engine_label, anneal_cfg.runs, anneal_cfg.steps_per_system);
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
                    format!("{}{}", ref_data.sequence[t as usize], ref_data.seq_ids[t as usize])
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
                seed = args.get(i).ok_or("missing --seed")?.parse().map_err(|e| format!("{e}"))?;
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
                first_seq_id = args.get(i).ok_or("missing --first-seq-id")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--engine" => {
                i += 1;
                engine = parse_engine(args.get(i).ok_or("missing --engine")?)?;
            }
            "--runs" => {
                i += 1;
                runs = Some(args.get(i).ok_or("missing --runs")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--steps" => {
                i += 1;
                steps = args.get(i).ok_or("missing --steps")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args.get(i).ok_or("missing --assign-cost")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).ok_or("missing --seed")?.parse().map_err(|e| format!("{e}"))?;
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
        tsv_lines.push(format!("{}\t{}\t{}\t{:.3}\t{:.2}", seq_ids[k], sequence[k], atom, v, c));
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
                snr = args.get(i).ok_or("missing --snr")?.parse().map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown pick flag {other:?}")),
        }
        i += 1;
    }

    let hsqc_p = hsqc_path.ok_or("pick requires --hsqc <pdata_dir_or_ft2>")?;
    let out_p = out_path.ok_or("pick requires --out <peaks.tsv>")?;
    let cfg = PickerConfig {
        snr_threshold: snr,
        ..PickerConfig::default()
    };

    let hsqc_spec = load_2d_auto(&hsqc_p)?;
    let roots = pick_hsqc_2d(&hsqc_spec, &cfg);
    let mut all_peaks = roots.clone();

    for (exp, path) in &triples {
        let s3 = load_3d_auto(path, exp)?;
        let picked = pick_triple_3d(&s3, &roots, &cfg);
        all_peaks.extend(picked);
    }

    write_peaks(&out_p, &all_peaks)?;
    println!(
        "picked {} HSQC roots and {} total peaks -> {}",
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
    println!("priors from {} files written to {}", paths.len(), out_p.display());
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
        "simulate" => cmd_simulate(&args[1..]),
        "assign" => cmd_assign(&args[1..]),
        "pick" => cmd_pick(&args[1..]),
        "priors" => cmd_priors(&args[1..]),
        other => Err(format!("unknown subcommand {other:?}\n\n{}", usage())),
    };
    if let Err(err) = res {
        eprintln!("error: {err}");
        process::exit(1);
    }
}
