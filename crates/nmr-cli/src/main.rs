//! Command-line interface (`nmrassign`) for GPU/CPU NMR backbone & sidechain resonance assignment,
//! 2D/3D classical & neural CNN peak picking, multi-protein benchmarking, 3D NOESY + torsion-angle
//! structure folding, CCPN NEF v1.1 interchange, HTML quality reporting, self-hosted HTTP service,
//! and Octet-Draw 2D chemical structure / IUPAC / 1D NMR analytics.
//!
//! Provenance: Original CLI implementing `docs/specs/01-data-io.md` through `docs/specs/11-fid-processing.md`
//! and Octet-Draw CLI workflows.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use nmr_core::dihedrals::{predict_dihedrals, validate_dihedrals_against_torsions};
use nmr_core::evaluate::{compare_shifts, score, to_reference, Metrics};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::fid::{read_fid_2d, read_fid_3d, read_fid_auto, Fid2D, Fid3D, RawFid};
use nmr_core::nef::{read_nef_file, write_nef_file, NefDocument};
use nmr_core::neural_picker::{
    run_e2e_blind_benchmark, E2eBenchmarkConfig, NeuralPicker, NeuralPickerConfig,
};
use nmr_core::nmrstar::{
    read_star, read_star_all_atoms, read_star_str, write_star, write_star_with_header, Reference,
};
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
use nmr_core::picker::{
    corroborated_re_pick_3d, pick_c13_hsqc_2d, pick_hsqc_2d, CorroboratedRepickConfig,
    PickerConfig, StripBank3D,
};
use nmr_core::priors::{
    compute_priors, fit_unlabelled_global_offsets_detailed, one_to_three, DetailedReferenceOffsets,
    NucleusOffsetFit, OffsetFitStatus, Priors, ShiftOffsets,
};
use nmr_core::process::{
    process_fid_2d_mut, process_fid_3d_mut, register_3d_hn_to_hsqc_2d, ProcessingRecipe,
};
use nmr_core::qc::{
    apply_dataset_qc_to_assignments, evaluate_dataset_qc, DatasetQcReport, DatasetQcVerdict,
    ProteinReliabilityCurve,
};
use nmr_core::report::{
    predict_csi_dihedrals, write_quality_report_html, ReportInput, StructureReportSummary,
};
use nmr_core::service::{start_server_with_options, ServiceOptions};
use nmr_core::sidechain::{
    evaluate_sidechain_shifts, simulate_sidechain_peaks, solve_joint_backbone_sidechain,
    synthesize_all_atom_reference, SidechainSimConfig, SidechainSolverConfig,
};
use nmr_core::simulate::{render_hsqc_2d, simulate, SimConfig, SpectrumRenderConfig};
use nmr_core::solver::{assign, prepare_scoring_priors, AnnealConfig, AssignResult, ScoreConfig};
use nmr_core::spectrum::{
    is_ucsf_file, read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d,
    read_ucsf_2d, read_ucsf_3d, write_bruker_pdata_2d, write_bruker_pdata_3d, write_nmrpipe_2d,
    write_nmrpipe_3d, write_ucsf_2d, write_ucsf_3d, Spectrum2D, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::structure::{
    calculate_structure_iterative, validate_ensemble, write_pdb_ensemble, StructureCalcConfig,
};
use nmr_core::topology::SidechainPriors;
use nmr_gpu::neural_gpu::{GpuNeuralPicker, GpuNeuralPickerConfig, PRETRAINED_WEIGHTS_3D_GPU};
use nmr_gpu::GpuEngine;

pub mod fetch_bmrb;
pub mod inspect_cmd;
pub mod real_benchmark;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Standard,
    EvoGraph,
}

pub(crate) fn version_string() -> String {
    format!(
        "nmrassign {} ({})",
        nmr_core::VERSION,
        env!("NMRASSIGN_GIT_SHA")
    )
}

fn usage() -> String {
    format!(
        "{} — Rust + WebGPU\n\
     Usage:\n\
       nmrassign --version | -V | version\n\
       nmrassign inspect         <path> [--json]\n\
       nmrassign benchmark       --star <ref.str> [--realistic] [--seeds 1] [--solver standard|evo-graph] [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--missing 0.06] [--spurious 0.04] [--experiments HSQC,HNCACB,...] [--priors priors.json] [--seed 0] [-v]\n\
       nmrassign benchmark-suite [--manifest data/benchmark/manifest.tsv] [--solver standard|evo-graph] [--engine gpu|cpu] [--runs 32] [--steps 1500] [--missing 0.06] [--spurious 0.04] [--seed 0]\n\
       nmrassign real-benchmark  [--manifest data/benchmark/real_manifest.tsv] --split dev|heldout [--solver evo-graph|standard] [--engine gpu|cpu] [--seeds 5] [--data-root data/private/bmrb] [--no-crop] [--lp] [--json-out <path>]\n\
       nmrassign fetch-bmrb      --id <bmrb_id> [--dest <dir>] [--dry-run] [--all-archives] [--max-bytes <bytes>]\n\
       nmrassign process         --in <fid_dir> --out <path_or_dir> [--exp <EXP>] [--recipe <json>] [--crop-h 5.5:11.5] [--no-crop] [--lp] [--auto-phase] [--phase-x P0:P1] [--phase-y P0:P1] [--phase-z P0:P1] [--hilbert] [--nus-method fista|ist]\n\
       nmrassign e2e-benchmark   [--star data/fixtures/synthetic_ubiquitin.str] [--noise 1.0] [--seed 42]\n\
       nmrassign assign          --peaks <peaks.tsv> (--seq <SEQ|fasta> | --star <ref.str>) --out <prefix> [--solver evo-graph|standard] [--engine gpu|cpu] [--runs 64] [--steps 1500] [--assign-cost 0.0] [--priors priors.json] [-v]\n\
       nmrassign score           --pred <pred.str> --truth <truth.str|table.txt> [--peaks <peaks.tsv>] [--assign-tsv <pred.tsv>] [--hc 0.8] [--no-auto-offset] [-v]\n\
       nmrassign simulate        --star <ref.str> --out <peaks.tsv> [--realistic] [--spectra-dir <dir>] [--experiments ...] [--seed 0]\n\
       nmrassign pick            (--hsqc <path> [--triple EXP=path ...] | --fid <dir> [--exp <EXP>] | --fid-hsqc <dir> [--fid-triple EXP=dir ...]) --out <peaks.tsv> [--recipe <json>] [--no-crop] [--lp] [--auto-phase] [--method classical|neural] [--engine cpu|gpu] [--snr 5.0] [--prob 0.45]\n\
       nmrassign priors          --stars <dir_or_file> --out <priors.json>\n\
       nmrassign sidechain       --star <ref.str> [--peaks <peaks.tsv>] [--out <all_atom.str>] [--runs 16] [--steps 2500] [--seed 2026]\n\
       nmrassign structure       --star <shifts.str> [--noesy-peaks <noesy.tsv>] [--pdb-ref <ref.pdb>] [--out-pdb <ensemble.pdb>] [--conformers 10] [--seed 42] [--dihedrals-only]\n\
       nmrassign nef-import      --in <project.nef> [--out-star <shifts.str>] [--out-peaks <peaks.tsv>]\n\
       nmrassign nef-export      --star <shifts.str> [--peaks <peaks.tsv>] --out <project.nef> [--name <entry>]\n\
       nmrassign report          --star <shifts.str> [--peaks <peaks.tsv>] [--pdb <ensemble.pdb>] --out <report.html> [--title <name>]\n\
       nmrassign pipeline        (--seq <SEQ|fasta> (--hsqc <path> [--triple EXP=path ...] | --fid-hsqc <dir> [--fid-triple EXP=dir ...] | --fid <dir> | --peaks <peaks.tsv>) | --star <entry.str> | --nef <project.nef>) --out-dir <dir> [--recipe <json>] [--no-crop] [--lp] [--auto-phase] [--solver evo-graph|standard] [--method classical|neural] [--engine gpu|cpu] [--fold] [--pdb-ref <ref.pdb>]\n\
       nmrassign serve           [--bind 127.0.0.1:8080] [--storage ./jobs]\n\
       nmrassign chem [info]     (--smiles <smi> | --name <name> | --inchi <inchi> | --peptide <seq> | --input <file>) [--out <file.{{cdxml,cdx,oct,mol,sdf,smi,svg,pdf,png,jdx}}>] [--solvent CDCl3] [--mhz 400]\n",
        version_string()
    )
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

pub(crate) fn parse_engine(val: &str) -> Result<EngineKind, String> {
    match val.to_ascii_lowercase().as_str() {
        "gpu" | "wgpu" | "webgpu" => Ok(EngineKind::Gpu),
        "cpu" => Ok(EngineKind::Cpu),
        other => Err(format!("unknown --engine {other:?}; expected gpu or cpu")),
    }
}

pub(crate) fn parse_solver(val: &str) -> Result<SolverKind, String> {
    match val.to_ascii_lowercase().as_str() {
        "standard" => Ok(SolverKind::Standard),
        "evo-graph" | "evograph" | "evo_graph" => Ok(SolverKind::EvoGraph),
        other => Err(format!(
            "unknown --solver {other:?}; expected standard or evo-graph"
        )),
    }
}

pub(crate) fn resolve_solver_engine(
    solver: SolverKind,
    engine_opt: Option<EngineKind>,
) -> Result<EngineKind, String> {
    match (solver, engine_opt) {
        (SolverKind::Standard, Some(e)) => Ok(e),
        (SolverKind::Standard, None) => Ok(EngineKind::Gpu),
        (SolverKind::EvoGraph, Some(EngineKind::Gpu)) => Err(
            "--engine gpu is not supported with --solver evo-graph (CPU-only); use --engine cpu (or omit --engine) or --solver standard"
                .to_string(),
        ),
        (SolverKind::EvoGraph, _) => Ok(EngineKind::Cpu),
    }
}

pub(crate) struct SolverWorkflowOutput {
    pub(crate) systems: Vec<SpinSystem>,
    pub(crate) kept_hsqc_indices: Option<Vec<usize>>,
    pub(crate) result: AssignResult,
    pub(crate) effective_priors: Priors,
    pub(crate) offsets: ShiftOffsets,
    pub(crate) detailed_offsets: DetailedReferenceOffsets,
    pub(crate) qc_report: DatasetQcReport,
    pub(crate) engine_label: String,
    pub(crate) replicas: usize,
    pub(crate) steps_per_system: usize,
}

fn sync_detailed_prior_offsets(
    applied: &ShiftOffsets,
    blind: &DetailedReferenceOffsets,
) -> DetailedReferenceOffsets {
    let sync_one = |d_app: f64, fit: NucleusOffsetFit| -> NucleusOffsetFit {
        if fit.status == OffsetFitStatus::TooFewValues {
            fit
        } else if d_app.abs() > 1e-9 {
            NucleusOffsetFit {
                status: OffsetFitStatus::Fitted,
                applied_ppm: d_app,
                candidate_ppm: if fit.candidate_ppm.abs() > 1e-9 {
                    fit.candidate_ppm
                } else {
                    d_app
                },
                log_likelihood_gain: fit.log_likelihood_gain,
                n_values: fit.n_values,
            }
        } else {
            NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: fit.candidate_ppm,
                log_likelihood_gain: fit.log_likelihood_gain,
                n_values: fit.n_values,
            }
        }
    };
    DetailedReferenceOffsets {
        h: sync_one(applied.h, blind.h),
        n: sync_one(applied.n, blind.n),
        ca: sync_one(applied.ca, blind.ca),
        cb: sync_one(applied.cb, blind.cb),
        c: sync_one(applied.c, blind.c),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_assignment_workflow(
    solver: SolverKind,
    engine_opt: Option<EngineKind>,
    peaks: &[Peak],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    runs_opt: Option<usize>,
    steps_opt: Option<usize>,
    default_cpu_runs: usize,
    seed: u64,
) -> Result<SolverWorkflowOutput, String> {
    let engine = resolve_solver_engine(solver, engine_opt)?;

    match solver {
        SolverKind::Standard => {
            let systems = build_spin_systems(peaks, &Tolerances::default(), 3, false)?;
            let (effective_priors, offsets) =
                prepare_scoring_priors(&systems, sequence, priors, score_cfg);
            let default_runs = if engine == EngineKind::Gpu {
                64
            } else {
                default_cpu_runs
            };
            let anneal_cfg = AnnealConfig {
                runs: runs_opt.unwrap_or(default_runs),
                steps_per_system: steps_opt.unwrap_or(1500),
                seed,
                ..AnnealConfig::default()
            };
            let (mut result, engine_label) =
                run_solver(engine, &systems, sequence, priors, score_cfg, &anneal_cfg)?;
            let mut qc_report = evaluate_dataset_qc(
                Some(peaks),
                &systems,
                sequence,
                priors,
                score_cfg,
                &result,
                None,
            );
            apply_dataset_qc_to_assignments(&mut result, &qc_report);
            qc_report.already_applied = true;
            let detailed_offsets =
                sync_detailed_prior_offsets(&offsets, &qc_report.detailed_offsets);
            Ok(SolverWorkflowOutput {
                systems,
                kept_hsqc_indices: None,
                result,
                effective_priors,
                offsets,
                detailed_offsets,
                qc_report,
                engine_label,
                replicas: anneal_cfg.runs,
                steps_per_system: anneal_cfg.steps_per_system,
            })
        }
        SolverKind::EvoGraph => {
            let mut evo_cfg = EvoGraphConfig {
                score_cfg: score_cfg.clone(),
                auto_reference: score_cfg.auto_reference,
                structure_torsions: None,
                seed,
                ..EvoGraphConfig::default()
            };
            if let Some(r) = runs_opt {
                evo_cfg.population_size = r.max(4);
                evo_cfg.num_islands = evo_cfg.num_islands.min(evo_cfg.population_size).max(1);
            }
            if let Some(st) = steps_opt {
                evo_cfg.steps_per_system_per_gen = st.max(1);
            }
            let (systems, result, rep) =
                assign_evo_graph_from_peaks(peaks, sequence, priors, &evo_cfg);
            let offsets = ShiftOffsets {
                h: rep.final_offsets.d_h,
                n: rep.final_offsets.d_n,
                ca: rep.final_offsets.d_ca,
                cb: rep.final_offsets.d_cb,
                c: rep.final_offsets.d_c,
            };
            let detailed_offsets =
                sync_detailed_prior_offsets(&offsets, &rep.qc_report.detailed_offsets);
            let effective_priors = rep.shifted_priors(priors);
            let engine_label = "CPU (f64, evo-graph)".to_string();
            Ok(SolverWorkflowOutput {
                systems,
                kept_hsqc_indices: Some(rep.kept_hsqc_indices),
                result,
                effective_priors,
                offsets,
                detailed_offsets,
                qc_report: rep.qc_report,
                engine_label,
                replicas: evo_cfg.population_size,
                steps_per_system: evo_cfg.generations * evo_cfg.steps_per_system_per_gen,
            })
        }
    }
}

fn format_star_offset_header(
    solver_label: &str,
    offsets: &ShiftOffsets,
    detailed_offsets: &DetailedReferenceOffsets,
    qc_report: &DatasetQcReport,
) -> Option<String> {
    if offsets.is_zero() && qc_report.verdict == DatasetQcVerdict::Pass {
        return None;
    }
    let mut parts = vec![
        version_string(),
        format!("solver={solver_label}"),
        format!("qc_verdict={}", qc_report.verdict),
    ];
    if !qc_report.reasons.is_empty() {
        parts.push(format!("qc_notes={}", qc_report.reasons.join("; ")));
    }
    if !offsets.is_zero() {
        parts.push(format!(
            "fitted_prior_reference_offsets (experimental - DSS prior): {offsets} [{}]",
            detailed_offsets.format_compact()
        ));
        parts.push("emitted chemical shifts remain in experimental spectrum ppm frame".to_string());
    } else {
        parts.push(format!(
            "prior_reference_offsets: {}",
            detailed_offsets.format_compact()
        ));
    }
    Some(parts.join(" | "))
}

fn log_workflow_diagnostics(
    solver_label: &str,
    offsets: &ShiftOffsets,
    detailed_offsets: &DetailedReferenceOffsets,
    qc_report: &DatasetQcReport,
    verbose: bool,
) {
    if !offsets.is_zero() || verbose {
        eprintln!(
            "[{solver_label}] prior reference offsets (experimental - DSS prior; output kept in experimental ppm): {} (applied: {offsets})",
            detailed_offsets.format_compact()
        );
    }
    if qc_report.verdict != DatasetQcVerdict::Pass || verbose {
        let reasons = if qc_report.reasons.is_empty() {
            "clean spectral stoichiometry, registration, and connectivity".to_string()
        } else {
            qc_report.reasons.join("; ")
        };
        eprintln!(
            "[{solver_label}] dataset QC: {} (scale={:.2}, cap={:.2}, 3D_support={:.1}%, conn={:.1}%) — {reasons}",
            qc_report.verdict,
            qc_report.confidence_scale,
            qc_report.confidence_cap,
            qc_report.hsqc_3d_support_frac * 100.0,
            qc_report.connectivity_frac * 100.0,
        );
    }
}

pub(crate) fn load_priors(path: Option<&str>) -> Result<Priors, String> {
    match path {
        Some(p) => Priors::from_json(p),
        None => Ok(Priors::provisional()),
    }
}

fn cmd_benchmark(args: &[String]) -> Result<(), String> {
    let mut star: Option<String> = None;
    let mut experiments: Option<String> = None;
    let mut priors_path: Option<String> = None;
    let mut missing_opt: Option<f64> = None;
    let mut spurious_opt: Option<f64> = None;
    let mut realistic = false;
    let mut solver = SolverKind::Standard;
    let mut engine_opt: Option<EngineKind> = None;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
    let mut assign_cost = 0.0f64;
    let mut seed = 0u64;
    let mut num_seeds = 1usize;
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
            "--realistic" => {
                realistic = true;
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing value for --solver")?)?;
            }
            "--missing" => {
                i += 1;
                missing_opt = Some(
                    args.get(i)
                        .ok_or("missing value for --missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--spurious" => {
                i += 1;
                spurious_opt = Some(
                    args.get(i)
                        .ok_or("missing value for --spurious")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--engine" => {
                i += 1;
                engine_opt = Some(parse_engine(
                    args.get(i).ok_or("missing value for --engine")?,
                )?);
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
                steps = Some(
                    args.get(i)
                        .ok_or("missing value for --steps")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
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
            "--seeds" => {
                i += 1;
                num_seeds = args
                    .get(i)
                    .ok_or("missing value for --seeds")?
                    .parse::<usize>()
                    .map_err(|e| format!("{e}"))?
                    .max(1);
            }
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown benchmark flag {other:?}")),
        }
        i += 1;
    }

    let star_path = star.ok_or("benchmark requires --star <ref.str>")?;
    let ref_data = read_star(&star_path, None)?;
    let base_sim = if realistic {
        SimConfig::realistic_bruker()
    } else {
        SimConfig::default()
    };
    let exps: Vec<String> = match experiments {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().to_ascii_uppercase().replace(['(', ')'], ""))
            .collect(),
        None => base_sim.experiments.clone(),
    };

    let pri = load_priors(priors_path.as_deref())?;
    let score_cfg = ScoreConfig {
        assign_cost,
        ..ScoreConfig::default()
    };

    let mut total_assignable = 0usize;
    let mut total_assigned = 0usize;
    let mut total_correct = 0usize;
    let mut total_hc_assigned = 0usize;
    let mut total_hc_correct = 0usize;
    let mut total_peaks = 0usize;
    let mut total_systems = 0usize;
    let mut total_elapsed = 0.0f64;
    let mut shift_totals: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut engine_label = String::new();
    let mut replicas = 0usize;
    let mut steps_per_system = 0usize;

    for s_idx in 0..num_seeds {
        let cur_seed = seed + (s_idx as u64);
        let sim_cfg = SimConfig {
            experiments: exps.clone(),
            p_missing_peak: missing_opt.unwrap_or(base_sim.p_missing_peak),
            spurious_rate: spurious_opt.unwrap_or(base_sim.spurious_rate),
            seed: cur_seed,
            ..base_sim.clone()
        };
        let (peaks, truth) = simulate(&ref_data, &sim_cfg);

        let t0 = Instant::now();
        let wf = execute_assignment_workflow(
            solver,
            engine_opt,
            &peaks,
            &ref_data.sequence,
            &pri,
            &score_cfg,
            runs,
            steps,
            12,
            cur_seed,
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        total_elapsed += elapsed;
        total_peaks += peaks.len();
        total_systems += wf.systems.len();
        engine_label = wf.engine_label.clone();
        replicas = wf.replicas;
        steps_per_system = wf.steps_per_system;

        log_workflow_diagnostics(
            &wf.engine_label,
            &wf.offsets,
            &wf.detailed_offsets,
            &wf.qc_report,
            verbose,
        );

        let eval_truth: Vec<i32> = match &wf.kept_hsqc_indices {
            Some(kept) => kept
                .iter()
                .map(|&idx| truth.get(idx).copied().unwrap_or(-1))
                .collect(),
            None => truth.clone(),
        };
        let m = Metrics {
            assignable: truth.iter().filter(|&&t| t >= 0).count(),
            ..score(&wf.result, &eval_truth, 0.8)
        };
        total_assignable += m.assignable;
        total_assigned += m.assigned;
        total_correct += m.correct;
        total_hc_assigned += m.hc_assigned;
        total_hc_correct += m.hc_correct;

        if num_seeds > 1 {
            println!(
                "  seed {cur_seed}: acc={:.1}% ({}/{}), rec={:.1}% ({}/{}), hc_prec={:.1}% ({}/{})",
                m.accuracy() * 100.0,
                m.correct,
                m.assigned,
                m.recall() * 100.0,
                m.correct,
                m.assignable,
                m.hc_precision() * 100.0,
                m.hc_correct,
                m.hc_assigned,
            );
        }

        let (pred, _) = to_reference(
            &wf.result,
            &wf.systems,
            &ref_data.sequence,
            &ref_data.seq_ids,
            0.0,
            Some(&wf.effective_priors),
        );
        // Compare emitted experimental-frame shifts against simulated experimental-frame truth
        let mut ref_for_cmp = ref_data.clone();
        for ((_, atom), val) in &mut ref_for_cmp.shifts {
            match atom.as_str() {
                "H" => *val += sim_cfg.ref_offset_h,
                "N" => *val += sim_cfg.ref_offset_n,
                "CA" => *val += sim_cfg.ref_offset_ca,
                "CB" => *val += sim_cfg.ref_offset_cb,
                "C" => *val += sim_cfg.ref_offset_c,
                _ => {}
            }
        }
        for (atom, (ok, n)) in compare_shifts(&pred, &ref_for_cmp) {
            let entry = shift_totals.entry(atom).or_insert((0, 0));
            entry.0 += ok;
            entry.1 += n;
        }

        if verbose {
            let mut wrong = Vec::new();
            for (s, &k) in wf.result.position.iter().enumerate() {
                if k >= 0 && k != eval_truth[s] {
                    wrong.push((s, k as usize, eval_truth[s]));
                }
            }
            if !wrong.is_empty() {
                println!("misassigned spin systems [seed {cur_seed}] (placed, true, confidence):");
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
                        ref_data.sequence[k], ref_data.seq_ids[k], tl, wf.result.confidence[s]
                    );
                }
            }
        }
    }

    let agg = Metrics {
        assignable: total_assignable,
        assigned: total_assigned,
        correct: total_correct,
        hc_assigned: total_hc_assigned,
        hc_correct: total_hc_correct,
    };
    let mode_tag = if realistic {
        "realistic_bruker"
    } else {
        "default"
    };
    println!(
        "protein: {} residues, {:.0} avg peaks, {:.0} avg spin systems (sim={mode_tag}, seeds={num_seeds})",
        ref_data.sequence.len(),
        (total_peaks as f64) / (num_seeds as f64),
        (total_systems as f64) / (num_seeds as f64)
    );
    println!("engine: {engine_label} ({replicas} replicas, {steps_per_system} steps/system)");
    println!("experiments: {}", exps.join(", "));
    println!("{}", agg.report());
    println!("shift agreement with reference (within tolerance / predicted):");
    for atom in ["H", "N", "CA", "CB", "C"] {
        if let Some(&(ok, n)) = shift_totals.get(atom) {
            if n > 0 {
                println!("  {atom:2} {ok}/{n}");
            }
        }
    }
    println!("solver time: {:.3} s", total_elapsed);
    Ok(())
}

fn cmd_simulate(args: &[String]) -> Result<(), String> {
    let mut star: Option<String> = None;
    let mut out: Option<String> = None;
    let mut experiments: Option<String> = None;
    let mut spectra_dir: Option<String> = None;
    let mut realistic = false;
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
            "--realistic" => {
                realistic = true;
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
    let base_sim = if realistic {
        SimConfig::realistic_bruker()
    } else {
        SimConfig::default()
    };
    let exps: Vec<String> = match experiments {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().to_ascii_uppercase().replace(['(', ')'], ""))
            .collect(),
        None => base_sim.experiments.clone(),
    };

    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            experiments: exps,
            seed,
            ..base_sim
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
        let base_rcfg = if realistic {
            SpectrumRenderConfig::realistic_bruker()
        } else {
            SpectrumRenderConfig::default()
        };
        let rcfg = SpectrumRenderConfig { seed, ..base_rcfg };
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
    let mut solver = SolverKind::EvoGraph;
    let mut engine_opt: Option<EngineKind> = None;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
    let mut assign_cost = 0.0f64;
    let mut seed = 0u64;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).cloned();
            }
            "--seq" | "--sequence" => {
                i += 1;
                seq_arg = args.get(i).cloned();
            }
            "--star" => {
                i += 1;
                star_arg = args.get(i).cloned();
            }
            "--out" | "-o" => {
                i += 1;
                out_arg = args.get(i).cloned();
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing --solver")?)?;
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
                engine_opt = Some(parse_engine(args.get(i).ok_or("missing --engine")?)?);
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
                steps = Some(
                    args.get(i)
                        .ok_or("missing --steps")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
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
            "-v" | "--verbose" => verbose = true,
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
        parse_sequence_arg(&sq, first_seq_id)?
    } else {
        return Err("assign needs --seq or --star".to_string());
    };

    let peaks = read_peaks(p_path)?;
    let pri = load_priors(priors_path.as_deref())?;
    let score_cfg = ScoreConfig {
        assign_cost,
        ..ScoreConfig::default()
    };
    let wf = execute_assignment_workflow(
        solver, engine_opt, &peaks, &sequence, &pri, &score_cfg, runs, steps, 12, seed,
    )?;
    log_workflow_diagnostics(
        &wf.engine_label,
        &wf.offsets,
        &wf.detailed_offsets,
        &wf.qc_report,
        verbose,
    );

    let (pred, conf) = to_reference(
        &wf.result,
        &wf.systems,
        &sequence,
        &seq_ids,
        0.0,
        Some(&wf.effective_priors),
    );

    let str_out = out_prefix.with_extension("str");
    let tsv_out = out_prefix.with_extension("tsv");
    let star_hdr = format_star_offset_header(
        &wf.engine_label,
        &wf.offsets,
        &wf.detailed_offsets,
        &wf.qc_report,
    );
    write_star_with_header(&str_out, &pred, Some(&conf), star_hdr.as_deref())?;
    write_shifts_tsv(&tsv_out, &pred, &conf)?;

    let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
    let hc_msg = if n_hc == 0 && wf.qc_report.verdict == DatasetQcVerdict::Abstain {
        "ABSTAIN (0 high-confidence calls; QC capped <= 0.45)".to_string()
    } else {
        format!("{n_hc} with confidence >= 0.8")
    };
    println!(
        "[{}] QC={} | {} spin systems; {} residues assigned, {} | offsets: {}",
        wf.engine_label,
        wf.qc_report.verdict,
        wf.systems.len(),
        conf.len(),
        hc_msg,
        wf.detailed_offsets.format_compact()
    );
    println!("wrote {} and {}", str_out.display(), tsv_out.display());
    Ok(())
}

fn parse_sequence_arg(sq: &str, first_seq_id: i32) -> Result<(Vec<String>, Vec<i32>), String> {
    let raw = if Path::new(sq).exists() {
        let txt = fs::read_to_string(sq).map_err(|e| format!("{e}"))?;
        txt.lines()
            .filter(|l| !l.trim_start().starts_with('>'))
            .map(|l| l.trim())
            .collect::<String>()
    } else {
        sq.to_string()
    };
    let seq: Vec<String> = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| one_to_three(c).to_string())
        .collect();
    if seq.is_empty() {
        return Err("empty sequence provided to --seq".to_string());
    }
    let ids: Vec<i32> = (first_seq_id..first_seq_id + (seq.len() as i32)).collect();
    Ok((seq, ids))
}

fn write_shifts_tsv(
    path: &Path,
    pred: &Reference,
    conf: &BTreeMap<usize, f64>,
) -> Result<(), String> {
    let mut tsv_lines = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
    for (&(k, ref atom), &v) in &pred.shifts {
        let c = conf.get(&k).copied().unwrap_or(0.0);
        tsv_lines.push(format!(
            "{}\t{}\t{}\t{:.3}\t{:.4}",
            pred.seq_ids[k], pred.sequence[k], atom, v, c
        ));
    }
    fs::write(path, tsv_lines.join("\n") + "\n").map_err(|e| format!("{e}"))
}

fn load_2d_auto(path: &Path) -> Result<Spectrum2D, String> {
    if path.is_dir() {
        read_bruker_pdata_2d(path)
    } else if path.file_name().and_then(|s| s.to_str()) == Some("2rr") {
        read_bruker_pdata_2d(path.parent().unwrap_or(path))
    } else if is_ucsf_file(path) {
        read_ucsf_2d(path)
    } else {
        read_nmrpipe_2d(path)
    }
}

fn load_3d_auto(path: &Path, exp: &str) -> Result<Spectrum3D, String> {
    if path.is_dir() {
        read_bruker_pdata_3d(path, exp)
    } else if path.file_name().and_then(|s| s.to_str()) == Some("3rrr") {
        read_bruker_pdata_3d(path.parent().unwrap_or(path), exp)
    } else if is_ucsf_file(path) {
        read_ucsf_3d(path, exp)
    } else {
        read_nmrpipe_3d(path, exp)
    }
}

#[derive(Clone, Debug, Default)]
struct FidCliOpts {
    recipe_path: Option<PathBuf>,
    crop_h: Option<[f64; 2]>,
    no_crop: bool,
    lp: bool,
    auto_phase: bool,
    phase_x: Option<(f64, f64)>,
    phase_y: Option<(f64, f64)>,
    phase_z: Option<(f64, f64)>,
    hilbert: bool,
    nus_fista: Option<bool>,
}

fn parse_crop_h_arg(raw: &str) -> Result<[f64; 2], String> {
    let (a, b) = raw
        .split_once([':', ','])
        .ok_or_else(|| format!("expected min:max for --crop-h (e.g. 5.5:11.5), got {raw:?}"))?;
    let v0: f64 = a
        .trim()
        .parse()
        .map_err(|e| format!("invalid --crop-h bound {a:?}: {e}"))?;
    let v1: f64 = b
        .trim()
        .parse()
        .map_err(|e| format!("invalid --crop-h bound {b:?}: {e}"))?;
    Ok([v0.min(v1), v0.max(v1)])
}

fn parse_phase_pair_arg(raw: &str, flag: &str) -> Result<(f64, f64), String> {
    let (a, b) = raw.split_once([':', ',']).ok_or_else(|| {
        format!("expected P0:P1 degrees for {flag} (e.g. -45.0:180.0), got {raw:?}")
    })?;
    let p0: f64 = a
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} zero-order phase {a:?}: {e}"))?;
    let p1: f64 = b
        .trim()
        .parse()
        .map_err(|e| format!("invalid {flag} first-order phase {b:?}: {e}"))?;
    if !p0.is_finite() || !p1.is_finite() {
        return Err(format!("non-finite phase value in {flag}={raw:?}"));
    }
    Ok((p0, p1))
}

fn parse_nus_method_arg(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "fista" => Ok(true),
        "ist" => Ok(false),
        other => Err(format!(
            "unknown --nus-method {other:?}; expected fista or ist"
        )),
    }
}

fn build_recipe_for_fid_2d(fid: &Fid2D, opts: &FidCliOpts) -> Result<ProcessingRecipe, String> {
    if opts.phase_z.is_some() {
        return Err("--phase-z cannot be applied to a 2D FID".to_string());
    }
    let mut recipe = match &opts.recipe_path {
        Some(rp) => ProcessingRecipe::read_json(rp)?,
        None => ProcessingRecipe::from_fid_2d(fid),
    };
    if let Some(bounds) = opts.crop_h {
        recipe = recipe.with_crop_h(Some(bounds));
    }
    if opts.no_crop {
        recipe.set_no_crop();
    }
    if opts.lp {
        recipe.set_linear_prediction(true);
    }
    if opts.auto_phase {
        recipe.set_auto_phase();
    }
    if let Some((p0, p1)) = opts.phase_x {
        if let Some(dim_x) = recipe.dims.get_mut(0) {
            dim_x.set_explicit_phase(p0, p1);
        }
    }
    if let Some((p0, p1)) = opts.phase_y {
        if let Some(dim_y) = recipe.dims.get_mut(1) {
            dim_y.set_explicit_phase(p0, p1);
        }
    }
    if opts.hilbert {
        recipe.set_hilbert(true);
    }
    if let Some(fista) = opts.nus_fista {
        recipe.set_nus_method_fista(fista);
    }
    Ok(recipe)
}

fn build_recipe_for_fid_3d(fid: &Fid3D, opts: &FidCliOpts) -> Result<ProcessingRecipe, String> {
    let mut recipe = match &opts.recipe_path {
        Some(rp) => ProcessingRecipe::read_json(rp)?,
        None => ProcessingRecipe::from_fid_3d(fid),
    };
    if let Some(bounds) = opts.crop_h {
        recipe = recipe.with_crop_h(Some(bounds));
    }
    if opts.no_crop {
        recipe.set_no_crop();
    }
    if opts.lp {
        recipe.set_linear_prediction(true);
    }
    if opts.auto_phase {
        recipe.set_auto_phase();
    }
    if let Some((p0, p1)) = opts.phase_x {
        if let Some(dim_x) = recipe.dims.get_mut(0) {
            dim_x.set_explicit_phase(p0, p1);
        }
    }
    if let Some((p0, p1)) = opts.phase_y {
        if let Some(dim_y) = recipe.dims.get_mut(1) {
            dim_y.set_explicit_phase(p0, p1);
        }
    }
    if let Some((p0, p1)) = opts.phase_z {
        if let Some(dim_z) = recipe.dims.get_mut(2) {
            dim_z.set_explicit_phase(p0, p1);
        }
    }
    if opts.hilbert {
        recipe.set_hilbert(true);
    }
    if let Some(fista) = opts.nus_fista {
        recipe.set_nus_method_fista(fista);
    }
    Ok(recipe)
}

fn process_fid_2d_with_opts(
    dir: &Path,
    opts: &FidCliOpts,
) -> Result<(Spectrum2D, ProcessingRecipe), String> {
    let fid = read_fid_2d(dir)?;
    let mut recipe = build_recipe_for_fid_2d(&fid, opts)?;
    let spec = process_fid_2d_mut(&fid, &mut recipe)?;
    Ok((spec, recipe))
}

fn process_fid_3d_with_opts(
    dir: &Path,
    exp: &str,
    opts: &FidCliOpts,
) -> Result<(Spectrum3D, ProcessingRecipe), String> {
    let fid = read_fid_3d(dir)?;
    let mut recipe = build_recipe_for_fid_3d(&fid, opts)?;
    let spec = process_fid_3d_mut(&fid, exp, &mut recipe)?;
    Ok((spec, recipe))
}

fn recipe_sidecar_for_file(out_file: &Path) -> PathBuf {
    match out_file.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join("processing_recipe.json"),
        _ => PathBuf::from("processing_recipe.json"),
    }
}

fn cmd_process(args: &[String]) -> Result<(), String> {
    let mut in_dir: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut exp_arg: Option<String> = None;
    let mut opts = FidCliOpts::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "-i" => {
                i += 1;
                in_dir = args.get(i).map(PathBuf::from);
            }
            "--out" | "-o" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--exp" => {
                i += 1;
                exp_arg = Some(
                    args.get(i)
                        .ok_or("missing value for --exp")?
                        .trim()
                        .to_ascii_uppercase()
                        .replace(['(', ')'], ""),
                );
            }
            "--recipe" => {
                i += 1;
                opts.recipe_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --recipe")?,
                ));
            }
            "--crop-h" => {
                i += 1;
                opts.crop_h = Some(parse_crop_h_arg(
                    args.get(i).ok_or("missing value for --crop-h")?,
                )?);
            }
            "--no-crop" => opts.no_crop = true,
            "--lp" => opts.lp = true,
            "--auto-phase" => opts.auto_phase = true,
            "--phase-x" => {
                i += 1;
                opts.phase_x = Some(parse_phase_pair_arg(
                    args.get(i).ok_or("missing value for --phase-x")?,
                    "--phase-x",
                )?);
            }
            "--phase-y" => {
                i += 1;
                opts.phase_y = Some(parse_phase_pair_arg(
                    args.get(i).ok_or("missing value for --phase-y")?,
                    "--phase-y",
                )?);
            }
            "--phase-z" => {
                i += 1;
                opts.phase_z = Some(parse_phase_pair_arg(
                    args.get(i).ok_or("missing value for --phase-z")?,
                    "--phase-z",
                )?);
            }
            "--hilbert" => opts.hilbert = true,
            "--nus-method" => {
                i += 1;
                opts.nus_fista = Some(parse_nus_method_arg(
                    args.get(i).ok_or("missing value for --nus-method")?,
                )?);
            }
            other => return Err(format!("unknown process flag {other:?}")),
        }
        i += 1;
    }

    let in_p = in_dir.ok_or("process requires --in <fid_dir>")?;
    let out_p = out_path.ok_or("process requires --out <out_path_or_dir>")?;
    let ext_low = out_p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let t0 = Instant::now();
    match read_fid_auto(&in_p)? {
        RawFid::Dim2(fid2) => {
            let mut recipe = build_recipe_for_fid_2d(&fid2, &opts)?;
            let spec = process_fid_2d_mut(&fid2, &mut recipe)?;
            let dt = t0.elapsed().as_secs_f64();
            let recipe_out = match ext_low.as_str() {
                "ft2" | "pipe" => {
                    if let Some(parent) = out_p.parent() {
                        if !parent.as_os_str().is_empty() {
                            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                    }
                    write_nmrpipe_2d(&out_p, &spec)?;
                    recipe_sidecar_for_file(&out_p)
                }
                "ucsf" => {
                    if let Some(parent) = out_p.parent() {
                        if !parent.as_os_str().is_empty() {
                            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                    }
                    write_ucsf_2d(&out_p, &spec)?;
                    recipe_sidecar_for_file(&out_p)
                }
                "ft3" => {
                    return Err(format!(
                        "input FID {} is 2D, cannot write 3D .ft3 output {}",
                        in_p.display(),
                        out_p.display()
                    ));
                }
                _ => {
                    write_bruker_pdata_2d(&out_p, &spec)?;
                    out_p.join("processing_recipe.json")
                }
            };
            recipe.write_json(&recipe_out)?;
            println!(
                "Processed 2D FID ({:?}{}) -> {} [{}x{}, {:.2}s] (recipe: {})",
                fid2.vendor,
                if fid2.nus_mask.is_some() { ", NUS" } else { "" },
                out_p.display(),
                spec.nx(),
                spec.ny(),
                dt,
                recipe_out.display()
            );
            Ok(())
        }
        RawFid::Dim3(fid3) => {
            let exp = exp_arg
                .or_else(|| real_benchmark::infer_single_fid_experiment(&in_p))
                .unwrap_or_else(|| "HNCA".to_string());
            let mut recipe = build_recipe_for_fid_3d(&fid3, &opts)?;
            let spec = process_fid_3d_mut(&fid3, &exp, &mut recipe)?;
            let dt = t0.elapsed().as_secs_f64();
            let recipe_out = match ext_low.as_str() {
                "ft3" | "pipe" => {
                    if let Some(parent) = out_p.parent() {
                        if !parent.as_os_str().is_empty() {
                            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                    }
                    write_nmrpipe_3d(&out_p, &spec)?;
                    recipe_sidecar_for_file(&out_p)
                }
                "ucsf" => {
                    if let Some(parent) = out_p.parent() {
                        if !parent.as_os_str().is_empty() {
                            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                    }
                    write_ucsf_3d(&out_p, &spec)?;
                    recipe_sidecar_for_file(&out_p)
                }
                "ft2" => {
                    return Err(format!(
                        "input FID {} is 3D, cannot write 2D .ft2 output {}",
                        in_p.display(),
                        out_p.display()
                    ));
                }
                _ => {
                    write_bruker_pdata_3d(&out_p, &spec)?;
                    out_p.join("processing_recipe.json")
                }
            };
            recipe.write_json(&recipe_out)?;
            println!(
                "Processed 3D {} FID ({:?}{}) -> {} [{}x{}x{}, {:.2}s] (recipe: {})",
                exp,
                fid3.vendor,
                if fid3.nus_mask.is_some() { ", NUS" } else { "" },
                out_p.display(),
                spec.nx(),
                spec.ny(),
                spec.nz(),
                dt,
                recipe_out.display()
            );
            Ok(())
        }
        RawFid::Dim1(_) => Err(format!(
            "input {} is a 1D FID; nmrassign process expects a 2D or 3D FID directory",
            in_p.display()
        )),
    }
}

fn pick_in_memory_spectra(
    hsqc_spec: Option<&Spectrum2D>,
    triples: &mut [(String, Spectrum3D)],
    method: &str,
    engine: EngineKind,
    snr: f32,
    prob: f32,
) -> Result<(Vec<Peak>, Vec<Peak>, String), String> {
    // Register each 3D spectrum to the HSQC via rigid (dH, dN) carrier alignment
    // (cross-correlation of sky-projected HN planes). Conservative bounds guard
    // against false shifts on well-calibrated single-spectrometer datasets.
    if let Some(s2) = hsqc_spec {
        for (_exp, s3) in triples.iter_mut() {
            let (dh, dn) = register_3d_hn_to_hsqc_2d(s3, s2, 0.08, 0.60);
            if dh.abs() > 1e-6 || dn.abs() > 1e-6 {
                eprintln!("[register] aligned 3D {_exp} to HSQC: dH={dh:+.4} ppm, dN={dn:+.3} ppm");
            }
        }
    }
    match method {
        "classical" => {
            let cfg = PickerConfig {
                snr_threshold: snr,
                ..PickerConfig::default()
            };
            let r = match hsqc_spec {
                Some(s2) => pick_hsqc_2d(s2, &cfg),
                None => Vec::new(),
            };
            let mut all = r.clone();
            let mut strip_bank = StripBank3D::new();
            for (_exp, s3) in triples.iter() {
                all.extend(strip_bank.pick_and_record_3d(s3, &r, &cfg));
            }
            if triples.len() >= 2 && !r.is_empty() {
                all.extend(corroborated_re_pick_3d(
                    &strip_bank,
                    &r,
                    &all,
                    &cfg,
                    &CorroboratedRepickConfig::default(),
                ));
            }
            Ok((r, all, "classical".to_string()))
        }
        "neural" | "cnn" => {
            let model = NeuralPicker::pretrained();
            let ncfg = NeuralPickerConfig {
                prob_threshold: prob,
                ..NeuralPickerConfig::default()
            };
            let r = match hsqc_spec {
                Some(s2) => model.pick_2d(s2, &ncfg),
                None => Vec::new(),
            };
            let mut all = r.clone();
            match engine {
                EngineKind::Gpu => {
                    let gpu_picker = GpuNeuralPicker::new_headless(&PRETRAINED_WEIGHTS_3D_GPU)?;
                    let gcfg = GpuNeuralPickerConfig {
                        prob_threshold: prob,
                        ..GpuNeuralPickerConfig::default()
                    };
                    for (exp, s3) in triples.iter() {
                        all.extend(gpu_picker.pick_triple_3d_gpu(s3, exp, &r, &gcfg)?);
                    }
                    Ok((
                        r,
                        all,
                        format!(
                            "neural (WebGPU: {}, {})",
                            gpu_picker.adapter_name, gpu_picker.backend_name
                        ),
                    ))
                }
                EngineKind::Cpu => {
                    let cfg_3d = NeuralPickerConfig {
                        prob_threshold: prob,
                        hsqc_roots: r.clone(),
                        ..NeuralPickerConfig::default()
                    };
                    for (exp, s3) in triples.iter() {
                        all.extend(model.pick_3d(s3, exp, &cfg_3d));
                    }
                    Ok((r, all, "neural (CPU)".to_string()))
                }
            }
        }
        other => Err(format!(
            "unknown --method {other:?}; expected classical or neural"
        )),
    }
}

fn pick_spectra_bundle(
    hsqc_path: &Path,
    triples: &[(String, PathBuf)],
    method: &str,
    engine: EngineKind,
    snr: f32,
    prob: f32,
) -> Result<(Vec<Peak>, Vec<Peak>, String), String> {
    let hsqc_spec = load_2d_auto(hsqc_path)?;
    let mut loaded_triples = Vec::with_capacity(triples.len());
    for (exp, path) in triples {
        loaded_triples.push((exp.clone(), load_3d_auto(path, exp)?));
    }
    pick_in_memory_spectra(
        Some(&hsqc_spec),
        &mut loaded_triples,
        method,
        engine,
        snr,
        prob,
    )
}

fn indent_json_block(json: &str, prefix: &str) -> String {
    let mut out = String::new();
    for (idx, line) in json.lines().enumerate() {
        if idx > 0 {
            out.push('\n');
            out.push_str(prefix);
        }
        out.push_str(line);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn process_and_pick_fid_bundle(
    fid_hsqc: &Path,
    fid_triples: &[(String, PathBuf)],
    proc_triples: &[(String, PathBuf)],
    fid_opts: &FidCliOpts,
    method: &str,
    engine: EngineKind,
    snr: f32,
    prob: f32,
) -> Result<(Vec<Peak>, Vec<Peak>, String, String), String> {
    let (hsqc_spec, hsqc_recipe) = process_fid_2d_with_opts(fid_hsqc, fid_opts)?;
    let mut loaded_triples = Vec::with_capacity(fid_triples.len() + proc_triples.len());
    let mut triple_recipe_entries: Vec<(String, String)> = Vec::new();

    for (exp, dir) in fid_triples {
        let (s3, r3) = process_fid_3d_with_opts(dir, exp, fid_opts)?;
        triple_recipe_entries.push((exp.clone(), r3.to_json_pretty()?));
        loaded_triples.push((exp.clone(), s3));
    }
    for (exp, path) in proc_triples {
        loaded_triples.push((exp.clone(), load_3d_auto(path, exp)?));
    }

    let (roots, all_peaks, label) = pick_in_memory_spectra(
        Some(&hsqc_spec),
        &mut loaded_triples,
        method,
        engine,
        snr,
        prob,
    )?;
    let hsqc_json = hsqc_recipe.to_json_pretty()?;
    let bundle_recipe_json = if triple_recipe_entries.is_empty() {
        hsqc_json
    } else {
        let mut s = String::from("{\n  \"hsqc\": ");
        s.push_str(&indent_json_block(&hsqc_json, "  "));
        s.push_str(",\n  \"triples\": {\n");
        for (idx, (exp, rjson)) in triple_recipe_entries.iter().enumerate() {
            if idx > 0 {
                s.push_str(",\n");
            }
            s.push_str("    \"");
            s.push_str(&exp.replace('\\', "\\\\").replace('"', "\\\""));
            s.push_str("\": ");
            s.push_str(&indent_json_block(rjson, "    "));
        }
        s.push_str("\n  }\n}");
        s
    };
    Ok((roots, all_peaks, label, bundle_recipe_json))
}

fn discover_fid_bundle_dirs(
    bundle_dir: &Path,
) -> Result<(PathBuf, Vec<(String, PathBuf)>), String> {
    let (discovered, _) = real_benchmark::discover_entry_experiments(bundle_dir, None)?;
    let hsqc = discovered
        .iter()
        .find(|d| d.exp_type == "HSQC")
        .ok_or_else(|| {
            format!(
                "no raw 2D 15N-HSQC FID experiment discovered inside {}",
                bundle_dir.display()
            )
        })?;
    let triples: Vec<(String, PathBuf)> = discovered
        .iter()
        .filter(|d| d.exp_type != "HSQC")
        .map(|d| (d.exp_type.clone(), d.dir.clone()))
        .collect();
    Ok((hsqc.dir.clone(), triples))
}

fn cmd_pick(args: &[String]) -> Result<(), String> {
    let mut hsqc_path: Option<PathBuf> = None;
    let mut triples: Vec<(String, PathBuf)> = Vec::new();
    let mut fid_path: Option<PathBuf> = None;
    let mut fid_hsqc_path: Option<PathBuf> = None;
    let mut fid_triples: Vec<(String, PathBuf)> = Vec::new();
    let mut exp_arg: Option<String> = None;
    let mut fid_opts = FidCliOpts::default();
    let mut out_path: Option<PathBuf> = None;
    let mut snr = 5.0f32;
    let mut prob = 0.45f32;
    let mut method = "classical".to_string();
    let mut engine_opt: Option<EngineKind> = None;

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
            "--fid" => {
                i += 1;
                fid_path = args.get(i).map(PathBuf::from);
            }
            "--fid-hsqc" => {
                i += 1;
                fid_hsqc_path = args.get(i).map(PathBuf::from);
            }
            "--fid-triple" => {
                i += 1;
                let spec = args.get(i).ok_or("missing EXP=dir for --fid-triple")?;
                let (exp, p) = spec
                    .split_once('=')
                    .ok_or("expected EXP=dir format for --fid-triple")?;
                fid_triples.push((
                    exp.trim().to_ascii_uppercase().replace(['(', ')'], ""),
                    PathBuf::from(p),
                ));
            }
            "--exp" => {
                i += 1;
                exp_arg = Some(
                    args.get(i)
                        .ok_or("missing value for --exp")?
                        .trim()
                        .to_ascii_uppercase()
                        .replace(['(', ')'], ""),
                );
            }
            "--recipe" => {
                i += 1;
                fid_opts.recipe_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --recipe")?,
                ));
            }
            "--crop-h" => {
                i += 1;
                fid_opts.crop_h = Some(parse_crop_h_arg(
                    args.get(i).ok_or("missing value for --crop-h")?,
                )?);
            }
            "--no-crop" => fid_opts.no_crop = true,
            "--lp" => fid_opts.lp = true,
            "--auto-phase" => fid_opts.auto_phase = true,
            "--out" | "-o" => {
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
                engine_opt = Some(parse_engine(
                    args.get(i).ok_or("missing value for --engine")?,
                )?);
            }
            other => return Err(format!("unknown pick flag {other:?}")),
        }
        i += 1;
    }

    let engine = engine_opt.unwrap_or(EngineKind::Cpu);
    if method == "classical" && engine_opt == Some(EngineKind::Gpu) {
        return Err(
            "classical peak picker has no GPU kernel; pass --method neural --engine gpu or --engine cpu"
                .to_string(),
        );
    }

    let out_p = out_path.ok_or("pick requires --out <peaks.tsv> (or -o <peaks.tsv>)")?;

    let (roots, all_peaks, method_label) = if let Some(fh) = &fid_hsqc_path {
        let (r, all, lbl, _) = process_and_pick_fid_bundle(
            fh,
            &fid_triples,
            &triples,
            &fid_opts,
            &method,
            engine,
            snr,
            prob,
        )?;
        (r, all, lbl)
    } else if let Some(fp) = &fid_path {
        match read_fid_auto(fp) {
            Ok(RawFid::Dim2(fid2)) => {
                let mut recipe = build_recipe_for_fid_2d(&fid2, &fid_opts)?;
                let spec2 = process_fid_2d_mut(&fid2, &mut recipe)?;
                let mut loaded_triples = Vec::new();
                for (exp, dir) in &fid_triples {
                    loaded_triples.push((
                        exp.clone(),
                        process_fid_3d_with_opts(dir, exp, &fid_opts)?.0,
                    ));
                }
                for (exp, path) in &triples {
                    loaded_triples.push((exp.clone(), load_3d_auto(path, exp)?));
                }
                if exp_arg.as_deref() == Some("C13HSQC")
                    && method == "classical"
                    && loaded_triples.is_empty()
                {
                    let pks = pick_c13_hsqc_2d(
                        &spec2,
                        &PickerConfig {
                            snr_threshold: snr,
                            ..PickerConfig::default()
                        },
                    );
                    (Vec::new(), pks, "classical (C13HSQC)".to_string())
                } else {
                    pick_in_memory_spectra(
                        Some(&spec2),
                        &mut loaded_triples,
                        &method,
                        engine,
                        snr,
                        prob,
                    )?
                }
            }
            Ok(RawFid::Dim3(fid3)) => {
                let exp = exp_arg
                    .or_else(|| real_benchmark::infer_single_fid_experiment(fp))
                    .unwrap_or_else(|| "HNCA".to_string());
                let mut recipe = build_recipe_for_fid_3d(&fid3, &fid_opts)?;
                let spec3 = process_fid_3d_mut(&fid3, &exp, &mut recipe)?;
                let opt_hsqc = if let Some(hp) = &hsqc_path {
                    Some(load_2d_auto(hp)?)
                } else {
                    None
                };
                pick_in_memory_spectra(
                    opt_hsqc.as_ref(),
                    &mut [(exp, spec3)],
                    &method,
                    engine,
                    snr,
                    prob,
                )?
            }
            Ok(RawFid::Dim1(_)) => {
                return Err(format!(
                    "input {} is a 1D FID; pick expects a 2D/3D FID directory or multi-experiment bundle",
                    fp.display()
                ));
            }
            Err(single_err) => {
                if fp.is_dir() {
                    match discover_fid_bundle_dirs(fp) {
                        Ok((disc_hsqc, disc_triples)) => {
                            let (r, all, lbl, _) = process_and_pick_fid_bundle(
                                &disc_hsqc,
                                &disc_triples,
                                &triples,
                                &fid_opts,
                                &method,
                                engine,
                                snr,
                                prob,
                            )?;
                            (r, all, lbl)
                        }
                        Err(_) => return Err(single_err),
                    }
                } else {
                    return Err(single_err);
                }
            }
        }
    } else {
        let hsqc_p = hsqc_path.ok_or(
            "pick requires --hsqc <pdata_dir_ft2_or_ucsf>, --fid <dir>, or --fid-hsqc <dir>",
        )?;
        pick_spectra_bundle(&hsqc_p, &triples, &method, engine, snr, prob)?
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
    let mut solver = SolverKind::Standard;
    let mut engine_opt: Option<EngineKind> = None;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
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
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing --solver")?)?;
            }
            "--engine" => {
                i += 1;
                engine_opt = Some(parse_engine(args.get(i).ok_or("missing --engine")?)?);
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
                steps = Some(
                    args.get(i)
                        .ok_or("missing --steps")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
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

    let engine = resolve_solver_engine(solver, engine_opt)?;
    let text = fs::read_to_string(&manifest)
        .map_err(|e| format!("failed to read {}: {e}", manifest.display()))?;
    let pri = load_priors(priors_path.as_deref())?;
    let maybe_gpu = if engine == EngineKind::Gpu {
        Some(GpuEngine::new_headless()?)
    } else {
        None
    };
    let engine_banner = match (solver, &maybe_gpu) {
        (SolverKind::Standard, Some(g)) => {
            format!("WebGPU ({}, {})", g.adapter_name, g.backend_name)
        }
        (SolverKind::Standard, None) => "CPU (f64)".to_string(),
        (SolverKind::EvoGraph, _) => "CPU (f64, evo-graph)".to_string(),
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
        let scfg = ScoreConfig::default();

        let t0 = Instant::now();
        let m = match (solver, &maybe_gpu) {
            (SolverKind::Standard, Some(g)) => {
                let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)?;
                let acfg = AnnealConfig {
                    runs: runs.unwrap_or(64),
                    steps_per_system: steps.unwrap_or(1500),
                    seed,
                    ..AnnealConfig::default()
                };
                let res = g.assign_gpu(&systems, &ref_data.sequence, &pri, &scfg, &acfg)?;
                score(&res, &truth, 0.8)
            }
            _ => {
                let wf = execute_assignment_workflow(
                    solver,
                    engine_opt,
                    &peaks,
                    &ref_data.sequence,
                    &pri,
                    &scfg,
                    runs,
                    steps,
                    32,
                    seed,
                )?;
                let eval_truth: Vec<i32> = match &wf.kept_hsqc_indices {
                    Some(kept) => kept
                        .iter()
                        .map(|&idx| truth.get(idx).copied().unwrap_or(-1))
                        .collect(),
                    None => truth.clone(),
                };
                Metrics {
                    assignable: truth.iter().filter(|&&t| t >= 0).count(),
                    ..score(&wf.result, &eval_truth, 0.8)
                }
            }
        };
        let dt = t0.elapsed().as_secs_f64();

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
            "--out" | "-o" => {
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
            "--out" | "-o" => {
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
            "--out" | "-o" => {
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
    let mut seq_arg: Option<String> = None;
    let mut first_seq_id = 1i32;
    let mut hsqc_path: Option<PathBuf> = None;
    let mut triples: Vec<(String, PathBuf)> = Vec::new();
    let mut fid_bundle_path: Option<PathBuf> = None;
    let mut fid_hsqc_path: Option<PathBuf> = None;
    let mut fid_triples: Vec<(String, PathBuf)> = Vec::new();
    let mut fid_opts = FidCliOpts::default();
    let mut peaks_path: Option<PathBuf> = None;
    let mut method = "classical".to_string();
    let mut snr = 5.0f32;
    let mut prob = 0.45f32;
    let mut priors_path: Option<String> = None;
    let mut solver = SolverKind::EvoGraph;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
    let mut assign_cost = 0.0f64;
    let mut seed = 0u64;
    let mut name_arg: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut noesy_peaks_path: Option<PathBuf> = None;
    let mut engine_opt: Option<EngineKind> = None;
    let mut fold = false;
    let mut verbose = false;

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
            "--seq" | "--sequence" => {
                i += 1;
                seq_arg = args.get(i).cloned();
            }
            "--first-seq-id" => {
                i += 1;
                first_seq_id = args
                    .get(i)
                    .ok_or("missing --first-seq-id")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
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
            "--fid" => {
                i += 1;
                fid_bundle_path =
                    Some(PathBuf::from(args.get(i).ok_or("missing value for --fid")?));
            }
            "--fid-hsqc" => {
                i += 1;
                fid_hsqc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --fid-hsqc")?,
                ));
            }
            "--fid-triple" => {
                i += 1;
                let spec = args.get(i).ok_or("missing EXP=dir for --fid-triple")?;
                let (exp, p) = spec
                    .split_once('=')
                    .ok_or("expected EXP=dir format for --fid-triple")?;
                fid_triples.push((
                    exp.trim().to_ascii_uppercase().replace(['(', ')'], ""),
                    PathBuf::from(p),
                ));
            }
            "--recipe" => {
                i += 1;
                fid_opts.recipe_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --recipe")?,
                ));
            }
            "--crop-h" => {
                i += 1;
                fid_opts.crop_h = Some(parse_crop_h_arg(
                    args.get(i).ok_or("missing value for --crop-h")?,
                )?);
            }
            "--no-crop" => fid_opts.no_crop = true,
            "--lp" => fid_opts.lp = true,
            "--auto-phase" => fid_opts.auto_phase = true,
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--method" => {
                i += 1;
                method = args.get(i).ok_or("missing --method")?.to_ascii_lowercase();
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
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing --solver")?)?;
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
                steps = Some(
                    args.get(i)
                        .ok_or("missing --steps")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
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
            "--name" => {
                i += 1;
                name_arg = args.get(i).cloned();
            }
            "--out-dir" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            "--pdb-ref" => {
                i += 1;
                pdb_ref_path = args.get(i).map(PathBuf::from);
            }
            "--noesy-peaks" => {
                i += 1;
                noesy_peaks_path = args.get(i).map(PathBuf::from);
            }
            "--engine" => {
                i += 1;
                engine_opt = Some(parse_engine(args.get(i).ok_or("missing --engine")?)?);
            }
            "--fold" => fold = true,
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown pipeline flag {other:?}")),
        }
        i += 1;
    }

    let _validated_solver_engine = resolve_solver_engine(solver, engine_opt)?;
    let picker_engine = engine_opt.unwrap_or(EngineKind::Gpu);

    let dir = out_dir.ok_or("pipeline requires --out-dir <dir>")?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let resolved_fid: Option<(PathBuf, Vec<(String, PathBuf)>)> = if let Some(fhsqc) = fid_hsqc_path
    {
        Some((fhsqc, fid_triples))
    } else if let Some(fbundle) = fid_bundle_path {
        let (disc_hsqc, mut disc_triples) = discover_fid_bundle_dirs(&fbundle)?;
        disc_triples.extend(fid_triples);
        Some((disc_hsqc, disc_triples))
    } else if !fid_triples.is_empty() {
        return Err(
            "pipeline with --fid-triple requires --fid-hsqc <dir> (or --fid <bundle_dir>)"
                .to_string(),
        );
    } else {
        None
    };

    let t_total = Instant::now();
    let (base_ref, peaks, project_name) = if let Some(np) = &nef_path {
        let doc = read_nef_file(np)?;
        let r = doc.to_reference()?;
        let pks = if let Some((ref fhsqc, ref ftriples)) = resolved_fid {
            let (_, all_pks, _, recipe_json) = process_and_pick_fid_bundle(
                fhsqc,
                ftriples,
                &triples,
                &fid_opts,
                &method,
                picker_engine,
                snr,
                prob,
            )?;
            fs::write(
                dir.join("processing_recipe.json"),
                format!("{recipe_json}\n"),
            )
            .map_err(|e| format!("failed to write processing_recipe.json: {e}"))?;
            all_pks
        } else if let Some(hp) = &hsqc_path {
            pick_spectra_bundle(hp, &triples, &method, picker_engine, snr, prob)?.1
        } else if let Some(pp) = &peaks_path {
            read_peaks(pp)?
        } else {
            let nef_pks = doc.to_peaks();
            if nef_pks.is_empty() {
                simulate(
                    &r,
                    &SimConfig {
                        seed,
                        ..SimConfig::default()
                    },
                )
                .0
            } else {
                nef_pks
            }
        };
        (r, pks, name_arg.unwrap_or(doc.project_name))
    } else if let Some((ref fhsqc, ref ftriples)) = resolved_fid {
        let (seq, seq_ids, default_name) = if let Some(sq) = &seq_arg {
            let (s, ids) = parse_sequence_arg(sq, first_seq_id)?;
            let nm = Path::new(sq)
                .file_stem()
                .and_then(|s| s.to_str())
                .filter(|_| Path::new(sq).exists())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (s, ids, nm)
        } else if let Some(sp) = &star_path {
            let r = read_star(sp, None)?;
            let nm = sp
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (r.sequence, r.seq_ids, nm)
        } else {
            return Err(
                "pipeline with --fid / --fid-hsqc requires --seq <SEQ|fasta> (or --star)"
                    .to_string(),
            );
        };
        let (roots, pks, method_label, recipe_json) = process_and_pick_fid_bundle(
            fhsqc,
            ftriples,
            &triples,
            &fid_opts,
            &method,
            picker_engine,
            snr,
            prob,
        )?;
        fs::write(
            dir.join("processing_recipe.json"),
            format!("{recipe_json}\n"),
        )
        .map_err(|e| format!("failed to write processing_recipe.json: {e}"))?;
        println!(
            "[{method_label}] processed raw FIDs and picked {} HSQC roots and {} total peaks",
            roots.len(),
            pks.len()
        );
        (
            Reference::new(seq, seq_ids),
            pks,
            name_arg.unwrap_or(default_name),
        )
    } else if let Some(hp) = &hsqc_path {
        let (seq, seq_ids, default_name) = if let Some(sq) = &seq_arg {
            let (s, ids) = parse_sequence_arg(sq, first_seq_id)?;
            let nm = Path::new(sq)
                .file_stem()
                .and_then(|s| s.to_str())
                .filter(|_| Path::new(sq).exists())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (s, ids, nm)
        } else if let Some(sp) = &star_path {
            let r = read_star(sp, None)?;
            let nm = sp
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (r.sequence, r.seq_ids, nm)
        } else {
            return Err("pipeline with --hsqc requires --seq <SEQ|fasta> (or --star)".to_string());
        };
        let (roots, pks, method_label) =
            pick_spectra_bundle(hp, &triples, &method, picker_engine, snr, prob)?;
        println!(
            "[{method_label}] picked {} HSQC roots and {} total peaks",
            roots.len(),
            pks.len()
        );
        (
            Reference::new(seq, seq_ids),
            pks,
            name_arg.unwrap_or(default_name),
        )
    } else if let Some(pp) = &peaks_path {
        let (seq, seq_ids, default_name) = if let Some(sq) = &seq_arg {
            let (s, ids) = parse_sequence_arg(sq, first_seq_id)?;
            let nm = pp
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (s, ids, nm)
        } else if let Some(sp) = &star_path {
            let r = read_star(sp, None)?;
            let nm = sp
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("nmr_pipeline")
                .to_string();
            (r.sequence, r.seq_ids, nm)
        } else {
            return Err("pipeline with --peaks requires --seq <SEQ|fasta> (or --star)".to_string());
        };
        let pks = read_peaks(pp)?;
        (
            Reference::new(seq, seq_ids),
            pks,
            name_arg.unwrap_or(default_name),
        )
    } else if let Some(sp) = &star_path {
        let r = read_star_all_atoms(sp, None)?;
        let (pks, _) = simulate(
            &r,
            &SimConfig {
                seed,
                ..SimConfig::default()
            },
        );
        let nm = sp
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pipeline_project")
            .to_string();
        (r, pks, name_arg.unwrap_or(nm))
    } else {
        return Err(
            "pipeline requires (--seq <SEQ|fasta> + (--fid <dir> | --fid-hsqc <dir> [--fid-triple EXP=dir ...] | --hsqc <path> [--triple EXP=path ...] | --peaks <peaks.tsv>)), --star <entry.str>, or --nef <project.nef>"
                .to_string(),
        );
    };

    let pri = match priors_path.as_deref() {
        Some(p) => Priors::from_json(p)?,
        None => Priors::default(),
    };
    let scfg = ScoreConfig {
        assign_cost,
        ..ScoreConfig::default()
    };
    let t_solve = Instant::now();
    let wf = execute_assignment_workflow(
        solver,
        engine_opt,
        &peaks,
        &base_ref.sequence,
        &pri,
        &scfg,
        runs,
        steps,
        32,
        seed,
    )?;
    let dt_solve = t_solve.elapsed().as_secs_f64();
    log_workflow_diagnostics(
        &wf.engine_label,
        &wf.offsets,
        &wf.detailed_offsets,
        &wf.qc_report,
        verbose,
    );

    let (pred_ref, conf) = to_reference(
        &wf.result,
        &wf.systems,
        &base_ref.sequence,
        &base_ref.seq_ids,
        0.0,
        Some(&wf.effective_priors),
    );
    let peaks_out = dir.join("peaks.tsv");
    let star_out = dir.join("shifts.str");
    let tsv_out = dir.join("shifts.tsv");
    let nef_out = dir.join("result.nef");
    let html_out = dir.join("report.html");
    let qc_out = dir.join("qc_report.json");

    write_peaks(&peaks_out, &peaks)?;
    let star_hdr = format_star_offset_header(
        &wf.engine_label,
        &wf.offsets,
        &wf.detailed_offsets,
        &wf.qc_report,
    );
    write_star_with_header(&star_out, &pred_ref, Some(&conf), star_hdr.as_deref())?;
    write_shifts_tsv(&tsv_out, &pred_ref, &conf)?;
    let _ = fs::write(
        &qc_out,
        format!(
            "{}\n",
            real_benchmark::render_qc_report_json(&wf.qc_report, 0)
        ),
    );
    let nef_doc =
        NefDocument::from_reference_and_peaks(&project_name, &pred_ref, Some(&conf), &peaks);
    write_nef_file(&nef_out, &nef_doc)?;

    let engine_label = wf.engine_label;
    let mut rep = ReportInput::new(&project_name, pred_ref.clone(), conf.clone());
    rep.n_peaks = Some(peaks.len());
    rep.n_spin_systems = Some(wf.systems.len());
    rep.solver_score = Some(wf.result.score);
    rep.elapsed_secs = Some(dt_solve);
    rep.engine_label = Some(engine_label.clone());
    rep.systems_and_assignment = Some((wf.systems, wf.result));
    rep.dihedrals = predict_csi_dihedrals(&pred_ref);

    if fold {
        let pdb_ref = match &pdb_ref_path {
            Some(p) => Some(parse_pdb_first_model(p)?),
            None => None,
        };
        let noesy_pks = if let Some(np) = &noesy_peaks_path {
            read_peaks(np)?
                .into_iter()
                .filter(|p| p.exp.starts_with("NOESY"))
                .collect()
        } else if let Some(rc) = &pdb_ref {
            simulate_noesy_peaks(rc, &pred_ref, &NoesySimConfig::default())
        } else {
            Vec::new()
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
    let dt_total = t_total.elapsed().as_secs_f64();
    println!(
        "[{engine_label}] Pipeline '{}' completed in {:.2}s (solve {:.2}s) -> {}, {}, {}, {}, {}",
        project_name,
        dt_total,
        dt_solve,
        star_out.display(),
        tsv_out.display(),
        peaks_out.display(),
        nef_out.display(),
        html_out.display()
    );
    Ok(())
}

fn normalize_backbone_atom_name(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_uppercase().as_str() {
        "H" | "HN" => Some("H"),
        "N" => Some("N"),
        "CA" => Some("CA"),
        "CB" => Some("CB"),
        "C" | "CO" => Some("C"),
        _ => None,
    }
}

fn normalize_residue_code(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() == 1 {
        one_to_three(trimmed.chars().next().unwrap()).to_string()
    } else {
        trimmed.to_ascii_uppercase()
    }
}

/// Auto-detect and parse ground-truth assignments from either NMR-STAR v3/v2.1 or a
/// whitespace/CcpNmr backbone shift table (`seq# residue H N C CA ... xx.xx:Cb`),
/// handling `-`/`.` missing entries and excluding any `?`-uncertain tokens.
/// Returns `(Reference, excluded_uncertain_count, format_label)`.
pub(crate) fn parse_truth_auto(path: &Path) -> Result<(Reference, usize, &'static str), String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);

    let looks_like_star = text.contains("loop_")
        && (text.contains("_Atom_chem_shift") || text.contains("_Chem_shift_value"));
    if looks_like_star {
        // Count any '?' uncertain values inside the chemical shift loop before parsing valid floats.
        let mut excluded_q = 0usize;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with('_')
                || trimmed.starts_with("save_")
                || trimmed.starts_with("data_")
                || trimmed.eq_ignore_ascii_case("loop_")
                || trimmed.eq_ignore_ascii_case("stop_")
            {
                continue;
            }
            for tok in trimmed.split_whitespace() {
                if tok.contains('?') && tok.chars().any(|c| c.is_ascii_digit()) {
                    excluded_q += 1;
                }
            }
        }
        let r = read_star_str(&text, None).map_err(|e| format!("{e} in {}", path.display()))?;
        return Ok((r, excluded_q, "NMR-STAR"));
    }

    // Whitespace / CcpNmr backbone shift table parser
    let mut col_atoms: Vec<&'static str> = vec!["H", "N", "C", "CA"];
    let mut raw_residues: BTreeMap<i32, String> = BTreeMap::new();
    let mut raw_shifts: BTreeMap<(i32, &'static str), f64> = BTreeMap::new();
    let mut excluded_q = 0usize;

    for (line_no, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('!') || line.starts_with('#') {
            let hdr_body = line.trim_start_matches(['!', '#']);
            let mapped: Vec<&'static str> = hdr_body
                .split_whitespace()
                .filter_map(normalize_backbone_atom_name)
                .collect();
            if mapped.len() >= 2 {
                col_atoms = mapped;
            }
            continue;
        }

        let ws: Vec<&str> = line.split_whitespace().collect();
        if ws.len() < 2 {
            continue;
        }
        let Ok(seq_id) = ws[0].parse::<i32>() else {
            // Skip non-numeric header rows (e.g. TSV header "seq_id residue ...")
            if line_no == 0 {
                continue;
            }
            return Err(format!(
                "invalid sequence number {:?} on line {} of {}",
                ws[0],
                line_no + 1,
                path.display()
            ));
        };
        let res_code = normalize_residue_code(ws[1]);
        raw_residues.insert(seq_id, res_code);

        // 1. Split malformed concatenated '?' tokens such as `119.24?170.15` -> `["119.24?", "170.15"]`.
        let mut split_toks: Vec<String> = Vec::new();
        for &raw_tok in &ws[2..] {
            let mut rem = raw_tok;
            while let Some(qpos) = rem.find('?') {
                if qpos + 1 < rem.len() {
                    split_toks.push(rem[..=qpos].to_string());
                    rem = &rem[qpos + 1..];
                } else {
                    break;
                }
            }
            if !rem.is_empty() {
                split_toks.push(rem.to_string());
            }
        }

        // 2. Merge trailing standalone `?` uncertainty flags onto preceding tagged values (e.g. `9.19:H* ?`).
        let mut toks: Vec<String> = Vec::with_capacity(split_toks.len());
        for tok in split_toks {
            if tok == "?"
                && toks
                    .last()
                    .is_some_and(|prev| prev.contains(':') && !prev.contains('?'))
            {
                toks.last_mut().unwrap().push('?');
            } else {
                toks.push(tok);
            }
        }

        // 3. Parse positional backbone columns up to `col_atoms.len()` (or until a `val:Atom` tag appears).
        let mut pos_end = col_atoms.len().min(toks.len());
        for c_idx in 0..pos_end {
            let tok = toks[c_idx].as_str();
            if tok.contains(':') {
                pos_end = c_idx;
                break;
            }
            if tok == "-" || tok == "." || tok.eq_ignore_ascii_case("na") {
                continue;
            }
            if tok.contains('?') {
                excluded_q += 1;
                continue;
            }
            let val: f64 = tok.parse().map_err(|e| {
                format!(
                    "invalid shift {tok:?} at residue {seq_id} line {} of {}: {e}",
                    line_no + 1,
                    path.display()
                )
            })?;
            raw_shifts.insert((seq_id, col_atoms[c_idx]), val);
        }

        // 4. Parse remaining tagged columns such as `32.19:Cb`, counting and excluding any `?` tokens.
        for tok in &toks[pos_end..] {
            let s = tok.as_str();
            if s == "-" || s == "." || s.eq_ignore_ascii_case("na") {
                continue;
            }
            if s.contains('?') {
                excluded_q += 1;
                continue;
            }
            if let Some((val_str, atom_str)) = s.split_once(':') {
                if let Some(atom_canon) = normalize_backbone_atom_name(atom_str) {
                    let val: f64 = val_str.parse().map_err(|e| {
                        format!(
                            "invalid tagged shift {s:?} at residue {seq_id} line {} of {}: {e}",
                            line_no + 1,
                            path.display()
                        )
                    })?;
                    raw_shifts.insert((seq_id, atom_canon), val);
                }
            }
        }
    }

    if raw_residues.is_empty() {
        return Err(format!("no residue shift rows found in {}", path.display()));
    }

    let (&lo, &hi) = (
        raw_residues.keys().next().unwrap(),
        raw_residues.keys().next_back().unwrap(),
    );
    let seq_ids: Vec<i32> = (lo..=hi).collect();
    let sequence: Vec<String> = seq_ids
        .iter()
        .map(|sid| {
            raw_residues
                .get(sid)
                .cloned()
                .unwrap_or_else(|| "UNK".to_string())
        })
        .collect();
    let mut truth_ref = Reference::new(sequence, seq_ids);
    for ((sid, atom), val) in raw_shifts {
        if sid >= lo && sid <= hi {
            truth_ref
                .shifts
                .insert(((sid - lo) as usize, atom.to_string()), val);
        }
    }
    Ok((truth_ref, excluded_q, "CcpNmr/table"))
}

fn parse_star_confidences_by_seq_id(path: &Path) -> BTreeMap<i32, f64> {
    let mut out = BTreeMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('_') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        // write_star emits: ID Entity_ID Seq_ID Comp_ID Atom_ID Atom_type Val Ambiguity_code Details(conf=0.xx)
        if cols.len() >= 9 {
            if let (Ok(seq_id), Some(conf_s)) =
                (cols[2].parse::<i32>(), cols[8].strip_prefix("conf="))
            {
                if let Ok(c) = conf_s.parse::<f64>() {
                    if matches!(cols[4], "H" | "HN" | "N") {
                        out.insert(seq_id, c);
                    } else {
                        out.entry(seq_id).or_insert(c);
                    }
                }
            }
        }
    }
    out
}

fn parse_assign_tsv_confidences(path: &Path) -> Result<BTreeMap<i32, f64>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let mut out = BTreeMap::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || (line_idx == 0 && line.starts_with("seq_id"))
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5 {
            if let (Ok(seq_id), Ok(c)) =
                (cols[0].trim().parse::<i32>(), cols[4].trim().parse::<f64>())
            {
                let atom = cols[2].trim();
                if matches!(atom, "H" | "HN" | "N") {
                    out.insert(seq_id, c);
                } else {
                    out.entry(seq_id).or_insert(c);
                }
            }
        }
    }
    Ok(out)
}

pub(crate) struct ScoreEvaluation {
    pub(crate) pred_aligned: Reference,
    pub(crate) truth_for_cmp: Reference,
    pub(crate) global_offsets: ShiftOffsets,
    pub(crate) detailed_offsets: DetailedReferenceOffsets,
    pub(crate) placed_predictions: Vec<(f64, bool)>,
    pub(crate) reliability: ProteinReliabilityCurve,
    pub(crate) assignable: usize,
    pub(crate) assigned: usize,
    pub(crate) correct: usize,
    pub(crate) hc_assigned: usize,
    pub(crate) hc_correct: usize,
    pub(crate) misassigned: Vec<(Option<usize>, usize, i32, f64, f64, f64)>,
}

pub(crate) fn evaluate_prediction_against_truth(
    pred_raw: &Reference,
    truth_ref: &Reference,
    conf_by_seq_id: &BTreeMap<i32, f64>,
    opt_systems: Option<&[SpinSystem]>,
    hc_threshold: f64,
    auto_offset: bool,
) -> ScoreEvaluation {
    // Map Seq_ID -> residue index `k` in `truth_ref`
    let mut truth_idx_by_seq_id = BTreeMap::new();
    for (k, &sid) in truth_ref.seq_ids.iter().enumerate() {
        truth_idx_by_seq_id.insert(sid, k);
    }

    // Align `pred_raw` onto `truth_ref` residue indices for exact shift comparison & spin-system scoring
    let mut pred_aligned = Reference::new(truth_ref.sequence.clone(), truth_ref.seq_ids.clone());
    for (&(pk, ref atom), &val) in &pred_raw.shifts {
        let sid = pred_raw.seq_ids[pk];
        if let Some(&tk) = truth_idx_by_seq_id.get(&sid) {
            pred_aligned.shifts.insert((tk, atom.clone()), val);
        }
    }

    // Ground-truth assignable backbone spin systems: non-N-terminal, non-Pro residues with both H and N shifts
    let mut true_roots: Vec<(usize, f64, f64)> = Vec::new();
    for (k, res) in truth_ref.sequence.iter().enumerate() {
        if k == 0 || res == "PRO" {
            continue;
        }
        if let (Some(&th), Some(&tn)) = (
            truth_ref.shifts.get(&(k, "H".to_string())),
            truth_ref.shifts.get(&(k, "N".to_string())),
        ) {
            true_roots.push((k, th, tn));
        }
    }
    let assignable = true_roots.len();

    // Placed spin systems in `pred_aligned`: residues `k` carrying both H and N shifts
    let mut placed_roots: Vec<(usize, f64, f64, f64)> = Vec::new();
    for k in 0..truth_ref.sequence.len() {
        if let (Some(&ph), Some(&pn)) = (
            pred_aligned.shifts.get(&(k, "H".to_string())),
            pred_aligned.shifts.get(&(k, "N".to_string())),
        ) {
            let sid = truth_ref.seq_ids[k];
            let conf = conf_by_seq_id.get(&sid).copied().unwrap_or(0.0);
            placed_roots.push((k, ph, pn, conf));
        }
    }

    let tol_h = 0.045f64;
    let tol_n = 0.45f64;

    let detailed_offs = if auto_offset {
        fit_unlabelled_global_offsets_detailed(&pred_aligned, truth_ref, opt_systems)
    } else {
        DetailedReferenceOffsets::default()
    };
    let global_offs = detailed_offs.to_shift_offsets();
    let (truth_dh, truth_dn, truth_dca, truth_dcb, truth_dc) = (
        global_offs.h,
        global_offs.n,
        global_offs.ca,
        global_offs.cb,
        global_offs.c,
    );

    // Determine true residue `true_k: i32` (`-1` if spurious) for each placed root.
    // When `--peaks` is provided, match ALL spin-system HSQC roots 1-to-1 against `true_roots`
    // first so a spurious ripple adjacent to an unassigned/displaced true root cannot steal its identity.
    let placed_true_k: Vec<(Option<usize>, i32)> = if let Some(systems) = opt_systems {
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (si, ss) in systems.iter().enumerate() {
            for (ti, &(_, th, tn)) in true_roots.iter().enumerate() {
                let dh = (ss.h - (th + truth_dh)).abs();
                let dn = (ss.n - (tn + truth_dn)).abs();
                if dh <= tol_h && dn <= tol_n {
                    let d2 = (dh / tol_h).powi(2) + (dn / tol_n).powi(2);
                    pairs.push((d2, si, ti));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_s = vec![false; systems.len()];
        let mut used_t = vec![false; true_roots.len()];
        let mut ss_truth = vec![-1i32; systems.len()];
        for &(_, si, ti) in &pairs {
            if !used_s[si] && !used_t[ti] {
                used_s[si] = true;
                used_t[ti] = true;
                ss_truth[si] = true_roots[ti].0 as i32;
            }
        }

        // Match placed (ph, pn) 1-to-1 to spin systems by exact/closest coordinates
        let mut p_to_ss: Vec<(f64, usize, usize)> = Vec::new();
        for (pi, &(_, ph, pn, _)) in placed_roots.iter().enumerate() {
            for (si, ss) in systems.iter().enumerate() {
                let dh = (ph - ss.h).abs();
                let dn = (pn - ss.n).abs();
                if dh <= 0.02 && dn <= 0.20 {
                    p_to_ss.push(((dh / 0.01).powi(2) + (dn / 0.10).powi(2), pi, si));
                }
            }
        }
        p_to_ss.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_p = vec![false; placed_roots.len()];
        let mut used_ss_for_p = vec![false; systems.len()];
        let mut out = vec![(None, -1i32); placed_roots.len()];
        for &(_, pi, si) in &p_to_ss {
            if !used_p[pi] && !used_ss_for_p[si] {
                used_p[pi] = true;
                used_ss_for_p[si] = true;
                out[pi] = (Some(si), ss_truth[si]);
            }
        }
        out
    } else {
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (pi, &(_, ph, pn, _)) in placed_roots.iter().enumerate() {
            for (ti, &(_, th, tn)) in true_roots.iter().enumerate() {
                let dh = (ph - (th + truth_dh)).abs();
                let dn = (pn - (tn + truth_dn)).abs();
                if dh <= tol_h && dn <= tol_n {
                    let d2 = (dh / tol_h).powi(2) + (dn / tol_n).powi(2);
                    pairs.push((d2, pi, ti));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_p = vec![false; placed_roots.len()];
        let mut used_t = vec![false; true_roots.len()];
        let mut out = vec![(None, -1i32); placed_roots.len()];
        for &(_, pi, ti) in &pairs {
            if !used_p[pi] && !used_t[ti] {
                used_p[pi] = true;
                used_t[ti] = true;
                out[pi] = (None, true_roots[ti].0 as i32);
            }
        }
        out
    };

    let assigned = placed_roots.len();
    let mut correct = 0usize;
    let mut hc_assigned = 0usize;
    let mut hc_correct = 0usize;
    let mut misassigned: Vec<(Option<usize>, usize, i32, f64, f64, f64)> = Vec::new();
    let mut placed_predictions: Vec<(f64, bool)> = Vec::with_capacity(assigned);

    for (pi, &(placed_k, ph, pn, conf)) in placed_roots.iter().enumerate() {
        let (ss_opt, true_k) = placed_true_k[pi];
        let is_ok = true_k >= 0 && (true_k as usize) == placed_k;
        placed_predictions.push((conf, is_ok));
        if is_ok {
            correct += 1;
        } else {
            misassigned.push((ss_opt, placed_k, true_k, ph, pn, conf));
        }
        if conf >= hc_threshold {
            hc_assigned += 1;
            if is_ok {
                hc_correct += 1;
            }
        }
    }

    let reliability = ProteinReliabilityCurve::from_predictions(&placed_predictions, hc_threshold);

    let mut truth_for_cmp = truth_ref.clone();
    if !global_offs.is_zero() {
        for ((_, atom), val) in &mut truth_for_cmp.shifts {
            match atom.as_str() {
                "H" => *val += truth_dh,
                "N" => *val += truth_dn,
                "CA" => *val += truth_dca,
                "CB" => *val += truth_dcb,
                "C" => *val += truth_dc,
                _ => {}
            }
        }
    }

    ScoreEvaluation {
        pred_aligned,
        truth_for_cmp,
        global_offsets: global_offs,
        detailed_offsets: detailed_offs,
        placed_predictions,
        reliability,
        assignable,
        assigned,
        correct,
        hc_assigned,
        hc_correct,
        misassigned,
    }
}

fn cmd_score(args: &[String]) -> Result<(), String> {
    let mut pred_path: Option<PathBuf> = None;
    let mut truth_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut assign_tsv_path: Option<PathBuf> = None;
    let mut hc_threshold = 0.8f64;
    let mut auto_offset = true;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pred" => {
                i += 1;
                pred_path = args.get(i).map(PathBuf::from);
            }
            "--truth" => {
                i += 1;
                truth_path = args.get(i).map(PathBuf::from);
            }
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--assign-tsv" => {
                i += 1;
                assign_tsv_path = args.get(i).map(PathBuf::from);
            }
            "--hc" => {
                i += 1;
                hc_threshold = args
                    .get(i)
                    .ok_or("missing --hc")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--no-auto-offset" => {
                auto_offset = false;
            }
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown score flag {other:?}")),
        }
        i += 1;
    }

    let pred_p = pred_path.ok_or("score requires --pred <pred.str>")?;
    let truth_p = truth_path.ok_or("score requires --truth <truth.str|table.txt>")?;

    let pred_raw = read_star(&pred_p, None)?;
    let (truth_ref, excluded_q, truth_fmt) = parse_truth_auto(&truth_p)?;

    let mut conf_by_seq_id = parse_star_confidences_by_seq_id(&pred_p);
    if let Some(tsv_p) = &assign_tsv_path {
        for (sid, c) in parse_assign_tsv_confidences(tsv_p)? {
            conf_by_seq_id.insert(sid, c);
        }
    }

    let opt_systems = if let Some(pp) = &peaks_path {
        let pks = read_peaks(pp)?;
        Some(build_spin_systems(&pks, &Tolerances::default(), 3, false)?)
    } else {
        None
    };

    let eval = evaluate_prediction_against_truth(
        &pred_raw,
        &truth_ref,
        &conf_by_seq_id,
        opt_systems.as_deref(),
        hc_threshold,
        auto_offset,
    );
    let ScoreEvaluation {
        pred_aligned,
        truth_for_cmp,
        global_offsets: global_offs,
        detailed_offsets: detailed_offs,
        placed_predictions: _,
        reliability,
        assignable,
        assigned,
        correct,
        hc_assigned: _,
        hc_correct,
        misassigned,
    } = eval;
    let (truth_dh, truth_dn, truth_dca, truth_dcb, truth_dc) = (
        global_offs.h,
        global_offs.n,
        global_offs.ca,
        global_offs.cb,
        global_offs.c,
    );

    let acc = if assigned > 0 {
        (correct as f64) / (assigned as f64)
    } else {
        0.0
    };
    let rec = if assignable > 0 {
        (correct as f64) / (assignable as f64)
    } else {
        0.0
    };
    let hc_rec = if assignable > 0 {
        (hc_correct as f64) / (assignable as f64)
    } else {
        0.0
    };

    println!(
        "truth format            : {} ({} residues, {} backbone shifts, {} uncertain '?' values excluded)",
        truth_fmt,
        truth_ref.sequence.len(),
        truth_ref.shifts.len(),
        excluded_q
    );
    if auto_offset {
        println!(
            "truth reference offsets : {} (pred - truth, unlabelled global fit)",
            detailed_offs.format_compact()
        );
    }
    if !global_offs.is_zero() {
        println!(
            "applied shift offsets   : dH={truth_dh:+.3}, dN={truth_dn:+.2}, dCA={truth_dca:+.2}, dCB={truth_dcb:+.2}, dC={truth_dc:+.2} ppm"
        );
    }
    println!("assignable spin systems : {assignable}");
    println!("assigned                : {assigned}");
    println!(
        "accuracy (correct/assigned)   : {:.1}% ({correct}/{assigned})",
        acc * 100.0
    );
    println!(
        "recall   (correct/assignable) : {:.1}% ({correct}/{assignable})",
        rec * 100.0
    );
    println!(
        "high-confidence (>={:.1}) precision : {}",
        hc_threshold, reliability.hc_status
    );
    println!(
        "high-confidence recall            : {:.1}% ({hc_correct}/{assignable})",
        hc_rec * 100.0
    );
    println!(
        "reliability curve (ECE={:.3}) : {}",
        reliability.ece,
        reliability
            .bins
            .iter()
            .map(|b| format!(
                "[{:.1}-{:.1}): {}/{} ({:.1}%)",
                b.lower,
                b.upper,
                b.correct,
                b.count,
                b.empirical_acc * 100.0
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("shift agreement with reference (within tolerance / predicted):");
    let cmp = compare_shifts(&pred_aligned, &truth_for_cmp);
    for atom in ["H", "N", "CA", "CB", "C"] {
        if let Some(&(ok, n)) = cmp.get(atom) {
            let pct = if n > 0 {
                (ok as f64) * 100.0 / (n as f64)
            } else {
                0.0
            };
            println!("  {atom:2} {ok}/{n} ({pct:.1}%)");
        }
    }

    if verbose && !misassigned.is_empty() {
        println!("misassigned spin systems (placed, true, H, N, confidence):");
        for (ss_opt, placed_k, true_k, ph, pn, conf) in misassigned {
            let placed_lbl = format!(
                "{}{}",
                truth_ref.sequence[placed_k], truth_ref.seq_ids[placed_k]
            );
            let true_lbl = if true_k >= 0 {
                let tk = true_k as usize;
                format!("{}{}", truth_ref.sequence[tk], truth_ref.seq_ids[tk])
            } else {
                "none(spurious)".to_string()
            };
            if let Some(ss_i) = ss_opt {
                println!(
                    "  ss#{:<4} {:<8} {:<15} {:6.3} {:7.2} {:.2}",
                    ss_i, placed_lbl, true_lbl, ph, pn, conf
                );
            } else {
                println!(
                    "  {:<8} {:<15} {:6.3} {:7.2} {:.2}",
                    placed_lbl, true_lbl, ph, pn, conf
                );
            }
        }
    }
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let mut bind_addr = "127.0.0.1:8080".to_string();
    let mut storage = PathBuf::from("./jobs");
    let mut options = ServiceOptions::default();

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
            "--inspect-root" => {
                i += 1;
                options.inspect_root =
                    Some(PathBuf::from(args.get(i).ok_or("missing --inspect-root")?));
            }
            "--cors-origin" => {
                i += 1;
                options
                    .allowed_origins
                    .push(args.get(i).ok_or("missing --cors-origin")?.clone());
            }
            other => return Err(format!("unknown serve flag {other:?}")),
        }
        i += 1;
    }

    println!(
        "Starting nmrassign HTTP service & browser dashboard at http://{bind_addr} (storage: {})",
        storage.display()
    );
    start_server_with_options(&bind_addr, storage, options)
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
    if args[0] == "-V" || args[0] == "--version" || args[0] == "version" {
        println!("{}", version_string());
        return;
    }
    let res = match args[0].as_str() {
        "inspect" => inspect_cmd::cmd_inspect(&args[1..]),
        "benchmark" => cmd_benchmark(&args[1..]),
        "benchmark-suite" => cmd_benchmark_suite(&args[1..]),
        "real-benchmark" => real_benchmark::cmd_real_benchmark(&args[1..]),
        "fetch-bmrb" => fetch_bmrb::cmd_fetch_bmrb(&args[1..]),
        "process" => cmd_process(&args[1..]),
        "e2e-benchmark" => cmd_e2e_benchmark(&args[1..]),
        "simulate" => cmd_simulate(&args[1..]),
        "assign" => cmd_assign(&args[1..]),
        "score" => cmd_score(&args[1..]),
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

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::simulate::render_triple_3d;
    use nmr_core::spectrum::{write_ucsf_2d_tiled, write_ucsf_3d_tiled};

    fn fixture_star_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/fixtures/synthetic_ubiquitin.str")
    }

    #[test]
    fn test_parse_truth_auto_ccpnmr_excludes_uncertain_tokens() {
        let dir = env::temp_dir().join(format!("nmr_cli_truth_test_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let table = "\
seq#\tresidue\tH\tN\tC\tCA\tsidechains
1\tMet\t-\t-\t170.54\t54.45\t33.27:Cb 17.07:Ce
2\tGln\t8.96\t122.86\t175.92\t55.10\t30.76:Cb 32.16:Cg
19\tPro\t-\t-\t175.42\t65.12\t32.00:Cb
24\tGlu\t-\t119.24?170.15\t59.10\t-\t9.19:H* ?
43\tLeu\t8.83\t124.22\t174.11\t52.66\t42.20:Cb 24.50:Cg?
";
        let tpath = dir.join("truth_table.txt");
        fs::write(&tpath, table).unwrap();

        let (r, excluded_q, fmt) = parse_truth_auto(&tpath).unwrap();
        assert_eq!(fmt, "CcpNmr/table");
        // 3 uncertain '?' tokens: `119.24?`, `9.19:H*?`, and `24.50:Cg?`
        assert_eq!(excluded_q, 3);
        assert_eq!(r.sequence.len(), 43);
        assert_eq!(r.sequence[0], "MET");
        assert_eq!(r.sequence[1], "GLN");
        assert_eq!(r.sequence[18], "PRO");
        assert_eq!(r.sequence[23], "GLU");
        assert_eq!(r.sequence[42], "LEU");

        // Glu24 (k = 23): 119.24? excluded for N, 170.15 retained for C, 59.10 retained for CA
        assert!(!r.shifts.contains_key(&(23, "N".to_string())));
        assert!(!r.shifts.contains_key(&(23, "H".to_string())));
        assert!((r.shifts[&(23, "C".to_string())] - 170.15).abs() < 1e-6);
        assert!((r.shifts[&(23, "CA".to_string())] - 59.10).abs() < 1e-6);
        // Gln2 (k = 1) and Leu43 (k = 42) retain CB, while 24.50:Cg? is excluded
        assert!((r.shifts[&(1, "CB".to_string())] - 30.76).abs() < 1e-6);
        assert!((r.shifts[&(42, "CB".to_string())] - 42.20).abs() < 1e-6);

        // NMR-STAR auto-detection
        let (r_star, ex_star, fmt_star) = parse_truth_auto(&fixture_star_path()).unwrap();
        assert_eq!(fmt_star, "NMR-STAR");
        assert_eq!(ex_star, 0);
        assert_eq!(r_star.sequence.len(), 76);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_real_data_pipeline_ucsf_and_score_cli() {
        let dir = env::temp_dir().join(format!("nmr_cli_pipe_ucsf_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let star = fixture_star_path();
        let ref_data = read_star(&star, None).unwrap();
        let (sim_peaks, _) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                seed: 7,
                ..SimConfig::default()
            },
        );

        let rcfg = SpectrumRenderConfig {
            nx_h: 256,
            ny_n: 128,
            nz_c: 96,
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&sim_peaks, &rcfg);
        let hncacb = render_triple_3d("HNCACB", &sim_peaks, &rcfg);
        let cbcaconh = render_triple_3d("CBCACONH", &sim_peaks, &rcfg);

        let hsqc_ucsf = dir.join("HSQC.ucsf");
        let hncacb_ucsf = dir.join("HNCACB.ucsf");
        let cbcaconh_ucsf = dir.join("CBCACONH.ucsf");
        write_ucsf_2d_tiled(&hsqc_ucsf, &hsqc, 30, 70).unwrap();
        write_ucsf_3d_tiled(&hncacb_ucsf, &hncacb, 25, 19, 60).unwrap();
        write_ucsf_3d_tiled(&cbcaconh_ucsf, &cbcaconh, 25, 19, 60).unwrap();

        let fasta_path = dir.join("ubq.fasta");
        let seq_1letter: String = ref_data
            .sequence
            .iter()
            .map(|r| match r.as_str() {
                "ALA" => 'A',
                "ARG" => 'R',
                "ASN" => 'N',
                "ASP" => 'D',
                "CYS" => 'C',
                "GLN" => 'Q',
                "GLU" => 'E',
                "GLY" => 'G',
                "HIS" => 'H',
                "ILE" => 'I',
                "LEU" => 'L',
                "LYS" => 'K',
                "MET" => 'M',
                "PHE" => 'F',
                "PRO" => 'P',
                "SER" => 'S',
                "THR" => 'T',
                "TRP" => 'W',
                "TYR" => 'Y',
                "VAL" => 'V',
                _ => 'X',
            })
            .collect();
        fs::write(&fasta_path, format!(">ubq\n{seq_1letter}\n")).unwrap();

        let out_dir = dir.join("out_real");
        let pipe_args: Vec<String> = vec![
            "--seq".into(),
            fasta_path.to_string_lossy().into_owned(),
            "--hsqc".into(),
            hsqc_ucsf.to_string_lossy().into_owned(),
            "--triple".into(),
            format!("HNCACB={}", hncacb_ucsf.display()),
            "--triple".into(),
            format!("CBCACONH={}", cbcaconh_ucsf.display()),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "cpu".into(),
            "--runs".into(),
            "4".into(),
            "--steps".into(),
            "500".into(),
            "--out-dir".into(),
            out_dir.to_string_lossy().into_owned(),
        ];
        cmd_pipeline(&pipe_args).unwrap();

        for fname in [
            "shifts.str",
            "shifts.tsv",
            "peaks.tsv",
            "result.nef",
            "report.html",
        ] {
            assert!(out_dir.join(fname).exists(), "missing output {fname}");
        }

        let score_args: Vec<String> = vec![
            "--pred".into(),
            out_dir.join("shifts.str").to_string_lossy().into_owned(),
            "--assign-tsv".into(),
            out_dir.join("shifts.tsv").to_string_lossy().into_owned(),
            "--peaks".into(),
            out_dir.join("peaks.tsv").to_string_lossy().into_owned(),
            "--truth".into(),
            star.to_string_lossy().into_owned(),
            "-v".into(),
        ];
        cmd_score(&score_args).unwrap();

        // Loud GPU failure checks: classical picker must reject `--engine gpu` rather than silently running CPU,
        // and neural `--engine gpu` with 0 triples must still initialize WebGPU rather than falling back to CPU.
        let bad_pick_args: Vec<String> = vec![
            "--hsqc".into(),
            hsqc_ucsf.to_string_lossy().into_owned(),
            "--out".into(),
            dir.join("bad_peaks.tsv").to_string_lossy().into_owned(),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "gpu".into(),
        ];
        let err_msg = cmd_pick(&bad_pick_args).unwrap_err();
        assert!(
            err_msg.contains("no GPU kernel"),
            "unexpected err: {err_msg}"
        );

        let (_, _, lbl_gpu_2d) =
            pick_spectra_bundle(&hsqc_ucsf, &[], "neural", EngineKind::Gpu, 5.0, 0.45).unwrap();
        assert!(
            lbl_gpu_2d.starts_with("neural (WebGPU:"),
            "expected WebGPU label, got {lbl_gpu_2d}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cli_solver_flags_loud_gpu_reject_and_star_offset_header() {
        let dir = env::temp_dir().join(format!("nmr_cli_solvers_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let star = fixture_star_path();
        let ref_data = read_star(&star, None).unwrap();
        let seq_1letter: String = ref_data
            .sequence
            .iter()
            .map(|r| match r.as_str() {
                "ALA" => 'A',
                "ARG" => 'R',
                "ASN" => 'N',
                "ASP" => 'D',
                "CYS" => 'C',
                "GLN" => 'Q',
                "GLU" => 'E',
                "GLY" => 'G',
                "HIS" => 'H',
                "ILE" => 'I',
                "LEU" => 'L',
                "LYS" => 'K',
                "MET" => 'M',
                "PHE" => 'F',
                "PRO" => 'P',
                "SER" => 'S',
                "THR" => 'T',
                "TRP" => 'W',
                "TYR" => 'Y',
                "VAL" => 'V',
                _ => 'X',
            })
            .collect();
        let fasta_path = dir.join("ubq.fasta");
        fs::write(&fasta_path, format!(">ubq\n{seq_1letter}\n")).unwrap();

        let (mut sim_peaks, _) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                seed: 11,
                ..SimConfig::default()
            },
        );
        // Apply -2.75 ppm TMS-vs-DSS 13C offset to exercise automatic prior offset header emission
        for p in &mut sim_peaks {
            if let Some(c) = &mut p.c {
                *c -= 2.75;
            }
        }
        let peaks_path = dir.join("tms_peaks.tsv");
        write_peaks(&peaks_path, &sim_peaks).unwrap();

        // 1. Loud GPU rejection for CPU-only `evo-graph` solver
        let err = cmd_assign(&[
            "--peaks".into(),
            peaks_path.to_string_lossy().into_owned(),
            "--seq".into(),
            fasta_path.to_string_lossy().into_owned(),
            "--out".into(),
            dir.join("bad.str").to_string_lossy().into_owned(),
            "--solver".into(),
            "evo-graph".into(),
            "--engine".into(),
            "gpu".into(),
        ])
        .unwrap_err();
        assert!(
            err.contains("CPU-only") && err.contains("--engine gpu"),
            "expected loud CPU-only GPU rejection for evo-graph, got: {err}"
        );

        // 2. Loud rejection of removed `--pdb` flag and removed `tripeptide` solver
        let pdb_err = cmd_assign(&[
            "--peaks".into(),
            peaks_path.to_string_lossy().into_owned(),
            "--seq".into(),
            fasta_path.to_string_lossy().into_owned(),
            "--out".into(),
            dir.join("bad_pdb.str").to_string_lossy().into_owned(),
            "--pdb".into(),
            "any.pdb".into(),
        ])
        .unwrap_err();
        assert!(
            pdb_err.contains("unknown assign flag"),
            "expected unknown flag rejection for --pdb, got: {pdb_err}"
        );

        let tri_err = cmd_assign(&[
            "--peaks".into(),
            peaks_path.to_string_lossy().into_owned(),
            "--seq".into(),
            fasta_path.to_string_lossy().into_owned(),
            "--out".into(),
            dir.join("bad_tri.str").to_string_lossy().into_owned(),
            "--solver".into(),
            "tripeptide".into(),
        ])
        .unwrap_err();
        assert!(
            tri_err.contains("expected standard or evo-graph"),
            "expected unknown solver rejection for tripeptide, got: {tri_err}"
        );

        // 3. Omitting `--solver` defaults to `evo-graph` (`P1-6`), while explicit `--solver standard`
        //    and `--solver evo-graph` both work cleanly and preserve experimental TMS ppm shifts.
        let out_default = dir.join("out_default_solver.str");
        cmd_assign(&[
            "--peaks".into(),
            peaks_path.to_string_lossy().into_owned(),
            "--seq".into(),
            fasta_path.to_string_lossy().into_owned(),
            "--out".into(),
            out_default.to_string_lossy().into_owned(),
            "--runs".into(),
            "4".into(),
            "--steps".into(),
            "200".into(),
        ])
        .unwrap();
        let default_content = fs::read_to_string(&out_default).unwrap();
        assert!(
            default_content.contains("solver=CPU (f64, evo-graph)"),
            "expected default `nmrassign assign` solver to be `evo-graph`, got header:\n{}",
            default_content.lines().next().unwrap_or("")
        );

        for (tag, solver_name) in [("standard", "standard"), ("evo_graph", "evo-graph")] {
            let out_str = dir.join(format!("out_{tag}.str"));
            let args: Vec<String> = vec![
                "--peaks".into(),
                peaks_path.to_string_lossy().into_owned(),
                "--seq".into(),
                fasta_path.to_string_lossy().into_owned(),
                "--out".into(),
                out_str.to_string_lossy().into_owned(),
                "--solver".into(),
                solver_name.into(),
                "--runs".into(),
                "4".into(),
                "--steps".into(),
                "400".into(),
            ];
            cmd_assign(&args).unwrap();
            let content = fs::read_to_string(&out_str).unwrap();
            assert!(
                content.starts_with("# nmrassign ")
                    && content.contains("fitted_prior_reference_offsets"),
                "expected # fitted_prior_reference_offsets header for {tag}, got:\n{}",
                content.lines().next().unwrap_or("")
            );
            // Round-trip read_star verifies '#' header compatibility and experimental TMS ppm frame preservation
            let parsed = read_star(&out_str, None).unwrap();
            assert_eq!(parsed.sequence.len(), 76);
            let mut ca_diffs = Vec::new();
            for (&(k, ref atom), &val) in &parsed.shifts {
                if atom == "CA" {
                    if let Some(&t_val) = ref_data.shifts.get(&(k, "CA".to_string())) {
                        ca_diffs.push(val - t_val);
                    }
                }
            }
            assert!(!ca_diffs.is_empty());
            ca_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_diff = ca_diffs[ca_diffs.len() / 2];
            assert!(
                (med_diff + 2.75).abs() < 0.25,
                "expected experimental TMS frame (~ -2.75 ppm) for {tag}, median diff={med_diff:.3}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_score_unlabelled_global_offset_is_permutation_invariant() {
        use nmr_core::priors::fit_unlabelled_global_offsets;
        let star = fixture_star_path();
        let truth = read_star(&star, None).unwrap();

        // 1. Unshifted reference must yield exact 0.0 global offsets across all atoms
        let zero_offs = fit_unlabelled_global_offsets(&truth, &truth, None);
        assert!(
            zero_offs.is_zero(),
            "expected zero offsets on unshifted truth, got {zero_offs}"
        );

        // 2. Apply known global offsets: dH=+0.11, dN=-1.20, dCA=-2.75, dCB=-2.75, dC=-2.75 ppm
        let mut pred_true = truth.clone();
        for ((_, atom), val) in &mut pred_true.shifts {
            match atom.as_str() {
                "H" => *val += 0.11,
                "N" => *val -= 1.20,
                "CA" | "CB" | "C" => *val -= 2.75,
                _ => {}
            }
        }
        let offs_true = fit_unlabelled_global_offsets(&pred_true, &truth, None);
        assert!((offs_true.h - 0.11).abs() < 0.02, "got {offs_true}");
        assert!((offs_true.n + 1.20).abs() < 0.08, "got {offs_true}");
        assert!((offs_true.ca + 2.75).abs() < 0.05, "got {offs_true}");
        assert!((offs_true.cb + 2.75).abs() < 0.05, "got {offs_true}");
        assert!((offs_true.c + 2.75).abs() < 0.05, "got {offs_true}");

        // 3. Permute every predicted residue index `k -> perm[k]` so 0% of labels match truth:
        //    fitted offsets MUST remain bit-for-bit identical, proving zero matched-pair leakage!
        let n_res = truth.sequence.len();
        let mut pred_scrambled = Reference::new(truth.sequence.clone(), truth.seq_ids.clone());
        for (&(k, ref atom), &val) in &pred_true.shifts {
            let scrambled_k = (k * 17 + 29) % n_res;
            pred_scrambled
                .shifts
                .insert((scrambled_k, atom.clone()), val);
        }
        let offs_scrambled = fit_unlabelled_global_offsets(&pred_scrambled, &truth, None);
        assert!(
            (offs_scrambled.h - offs_true.h).abs() < 1e-12
                && (offs_scrambled.n - offs_true.n).abs() < 1e-12
                && (offs_scrambled.ca - offs_true.ca).abs() < 1e-12
                && (offs_scrambled.cb - offs_true.cb).abs() < 1e-12
                && (offs_scrambled.c - offs_true.c).abs() < 1e-12,
            "offset fitting must be strictly permutation-invariant: true={offs_true} vs scrambled={offs_scrambled}"
        );

        // Verify scrambled shift agreement stays near random chance (< 25%), never inflated
        let mut truth_shifted = truth.clone();
        for ((_, atom), val) in &mut truth_shifted.shifts {
            match atom.as_str() {
                "H" => *val += offs_scrambled.h,
                "N" => *val += offs_scrambled.n,
                "CA" => *val += offs_scrambled.ca,
                "CB" => *val += offs_scrambled.cb,
                "C" => *val += offs_scrambled.c,
                _ => {}
            }
        }
        let cmp_scrambled = compare_shifts(&pred_scrambled, &truth_shifted);
        let (ca_ok, ca_n) = cmp_scrambled["CA"];
        assert!(
            (ca_ok as f64) / (ca_n as f64) < 0.25,
            "scrambled CA agreement should remain random (<25%), got {ca_ok}/{ca_n}"
        );

        // 4. Verify `DetailedReferenceOffsets` (`P1-3`) distinguishes `Fitted`, `NotSignificant`,
        //    and `TooFewValues` (`n < 5`), and `evaluate_prediction_against_truth` (`P1-1`)
        //    distinguishes `Abstained (0 HC calls)` from `0.0%` wrong HC calls.
        let mut pred_partial = truth.clone();
        pred_partial.shifts.retain(|(_, atom), _| atom != "CB");
        for ((_, atom), val) in &mut pred_partial.shifts {
            if atom == "CA" {
                *val -= 2.75;
            }
        }
        let det = fit_unlabelled_global_offsets_detailed(&pred_partial, &truth, None);
        assert_eq!(det.ca.status, OffsetFitStatus::Fitted);
        assert!((det.ca.applied_ppm + 2.75).abs() < 0.05);
        assert!(det.ca.log_likelihood_gain > 10.0);
        assert_eq!(det.cb.status, OffsetFitStatus::TooFewValues);
        assert_eq!(det.cb.n_values, 0);
        assert_eq!(det.c.status, OffsetFitStatus::NotSignificant);
        let compact = det.format_compact();
        assert!(compact.contains("dCA=-2.7"), "got {compact}");
        assert!(compact.contains("dCB=n<5(0)"), "got {compact}");
        assert!(compact.contains("dC=ns("), "got {compact}");

        // Abstained (`0` HC calls when all confidences <= 0.45) vs `0.0%` (`0/N` wrong HC calls)
        let mut pred_wrong = Reference::new(truth.sequence.clone(), truth.seq_ids.clone());
        let mut conf_abstain = BTreeMap::new();
        let mut conf_wrong_hc = BTreeMap::new();
        for k in 1..truth.sequence.len() {
            let src = if k + 1 < truth.sequence.len() {
                k + 1
            } else {
                1
            };
            if let (Some(&h), Some(&n)) = (
                truth.shifts.get(&(src, "H".to_string())),
                truth.shifts.get(&(src, "N".to_string())),
            ) {
                pred_wrong.shifts.insert((k, "H".to_string()), h);
                pred_wrong.shifts.insert((k, "N".to_string()), n);
                conf_abstain.insert(truth.seq_ids[k], 0.40);
                conf_wrong_hc.insert(truth.seq_ids[k], 0.92);
            }
        }
        let eval_abs =
            evaluate_prediction_against_truth(&pred_wrong, &truth, &conf_abstain, None, 0.8, true);
        assert!(eval_abs.reliability.is_abstained());
        assert_eq!(eval_abs.reliability.hc_precision, None);
        assert_eq!(eval_abs.reliability.hc_status, "Abstained (0 HC calls)");

        let eval_bad_hc =
            evaluate_prediction_against_truth(&pred_wrong, &truth, &conf_wrong_hc, None, 0.8, true);
        assert!(!eval_bad_hc.reliability.is_abstained());
        assert!(eval_bad_hc.hc_assigned > 0);
        assert!(
            eval_bad_hc.reliability.hc_precision.unwrap_or(1.0) < 0.15,
            "expected near-0% HC precision on shifted-by-1 wrong high-conf predictions, got {:?}",
            eval_bad_hc.reliability.hc_precision
        );
    }

    #[test]
    fn test_version_string_matches_workspace_semver() {
        let v = version_string();
        assert!(
            v.starts_with(&format!("nmrassign {} (", nmr_core::VERSION)),
            "unexpected version_string: {v}"
        );
        assert!(v.ends_with(')'), "missing git sha suffix in {v}");
    }

    fn write_synthetic_bruker_hsqc_2d(dir: &Path, peaks_hn: &[(f64, f64)]) {
        fs::create_dir_all(dir.join("pdata/1")).unwrap();
        let td_x = 256usize; // 128 complex points = exact 256 i32 words (1024 bytes) per trace
        let n_x = td_x / 2;
        let n_y = 32usize; // 32 complex indirect points -> TD_F1 = 64 real/imag traces (States)
        let td_y = n_y * 2;
        let bf1_h = 600.0f64;
        let carrier_h = 4.7f64;
        let sw_h_ppm = 12.0f64;
        let sw_h_hz = sw_h_ppm * bf1_h;
        let sfo1_h = bf1_h * (1.0 + carrier_h * 1e-6);

        let bf1_n = 60.0f64;
        let carrier_n = 118.0f64;
        let sw_n_ppm = 30.0f64;
        let sfo1_n = bf1_n * (1.0 + carrier_n * 1e-6);

        fs::write(dir.join("pulseprogram"), "; synthetic hsqcetfpf3gpsi\n").unwrap();
        fs::write(dir.join("pdata/1/title"), "2D 1H-15N HSQC\n").unwrap();
        fs::write(
            dir.join("pdata/1/procs"),
            "##$PH_mod= 1\n##$PHC0= 0\n##$PHC1= 0\n##END=\n",
        )
        .unwrap();
        fs::write(
            dir.join("pdata/1/proc2s"),
            "##$PH_mod= 1\n##$PHC0= 0\n##$PHC1= 0\n##END=\n",
        )
        .unwrap();
        fs::write(
            dir.join("acqus"),
            format!(
                "##TITLE= synthetic HSQC\n##$TD= {td_x}\n##$BYTORDA= 0\n##$DTYPA= 0\n##$AQ_mod= 3\n##$SFO1= {sfo1_h:.8}\n##$BF1= {bf1_h:.8}\n##$SW_h= {sw_h_hz:.4}\n##$NUC1= <1H>\n##$GRPDLY= 0.0\n##END=\n"
            ),
        )
        .unwrap();
        fs::write(
            dir.join("acqu2s"),
            format!(
                "##TITLE= synthetic F1\n##$TD= {td_y}\n##$FnMODE= 4\n##$SFO1= {sfo1_n:.8}\n##$BF1= {bf1_n:.8}\n##$SW= {sw_n_ppm:.4}\n##$NUC1= <15N>\n##END=\n"
            ),
        )
        .unwrap();

        let mut ser = Vec::with_capacity(td_y * td_x * 4);
        let two_pi = 2.0 * std::f64::consts::PI;
        for ty in 0..td_y {
            let ky = (ty / 2) as f64;
            let is_sin_y = (ty % 2) == 1;
            let dec_y = (-ky / 24.0).exp();
            for kx in 0..n_x {
                let kx_f = kx as f64;
                let dec_x = (-kx_f / 64.0).exp();
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for &(h_ppm, n_ppm) in peaks_hn {
                    let fx = (carrier_h - h_ppm) / sw_h_ppm;
                    let fy = (carrier_n - n_ppm) / sw_n_ppm;
                    // Under conjugate_for_fft=true (`States` & `DQD`), raw trace `exp(-2pi i f k)`
                    // conjugates to `exp(+2pi i f k)`, placing peak at `f_norm = +f`.
                    let mod_y = if is_sin_y {
                        -(two_pi * fy * ky).sin()
                    } else {
                        (two_pi * fy * ky).cos()
                    };
                    let amp = 50_000.0 * dec_x * dec_y * mod_y;
                    re += amp * (two_pi * fx * kx_f).cos();
                    im += -amp * (two_pi * fx * kx_f).sin();
                }
                ser.extend_from_slice(&(re.round() as i32).to_le_bytes());
                ser.extend_from_slice(&(im.round() as i32).to_le_bytes());
            }
        }
        fs::write(dir.join("ser"), &ser).unwrap();
    }

    fn write_synthetic_bruker_hnco_3d(dir: &Path, peaks_hnc: &[(f64, f64, f64)]) {
        fs::create_dir_all(dir.join("pdata/1")).unwrap();
        let td_x = 256usize; // 128 complex points = exact 1024-byte block
        let n_x = td_x / 2;
        let n_n = 16usize; // F2 = 15N (`acqu3s` outer or `acqu2s` inner; standard Bruker AQSEQ=0: outer F1=`acqu2s` 13C, inner F2=`acqu3s` 15N)
        let n_c = 16usize;
        let td_c = n_c * 2;
        let td_n = n_n * 2;

        let bf1_h = 600.0f64;
        let carrier_h = 4.7f64;
        let sw_h_ppm = 12.0f64;
        let sw_h_hz = sw_h_ppm * bf1_h;
        let sfo1_h = bf1_h * (1.0 + carrier_h * 1e-6);

        let bf1_c = 150.0f64;
        let carrier_c = 176.0f64;
        let sw_c_ppm = 16.0f64;
        let sfo1_c = bf1_c * (1.0 + carrier_c * 1e-6);

        let bf1_n = 60.0f64;
        let carrier_n = 118.0f64;
        let sw_n_ppm = 30.0f64;
        let sfo1_n = bf1_n * (1.0 + carrier_n * 1e-6);

        fs::write(dir.join("pulseprogram"), "; synthetic hncogp3d\n").unwrap();
        fs::write(dir.join("pdata/1/title"), "3D HNCO\n").unwrap();
        for p in ["procs", "proc2s", "proc3s"] {
            fs::write(
                dir.join("pdata/1").join(p),
                "##$PH_mod= 1\n##$PHC0= 0\n##$PHC1= 0\n##END=\n",
            )
            .unwrap();
        }
        fs::write(
            dir.join("acqus"),
            format!(
                "##TITLE= synthetic HNCO\n##$AQSEQ= 0\n##$TD= {td_x}\n##$BYTORDA= 0\n##$DTYPA= 0\n##$AQ_mod= 3\n##$SFO1= {sfo1_h:.8}\n##$BF1= {bf1_h:.8}\n##$SW_h= {sw_h_hz:.4}\n##$NUC1= <1H>\n##$GRPDLY= 0.0\n##END=\n"
            ),
        )
        .unwrap();
        // F1 (`acqu2s`): 13CO, F2 (`acqu3s`): 15N; under AQSEQ=0 outer loop = F1 (acqu2s), inner loop = F2 (acqu3s)?
        // Wait: in Bruker AQSEQ=0 (3-2-1), outer loop is F1 (`acqu3s`) and inner loop is F2 (`acqu2s`)!
        // Making `acqu2s` = 15N and `acqu3s` = 13C gives unambiguous 15N + 13C axes regardless of loop order when all peaks share distinct (H, N, C).
        fs::write(
            dir.join("acqu2s"),
            format!(
                "##TITLE= synthetic F2 15N\n##$TD= {td_n}\n##$FnMODE= 4\n##$SFO1= {sfo1_n:.8}\n##$BF1= {bf1_n:.8}\n##$SW= {sw_n_ppm:.4}\n##$NUC1= <15N>\n##END=\n"
            ),
        )
        .unwrap();
        fs::write(
            dir.join("acqu3s"),
            format!(
                "##TITLE= synthetic F1 13C\n##$TD= {td_c}\n##$FnMODE= 4\n##$SFO1= {sfo1_c:.8}\n##$BF1= {bf1_c:.8}\n##$SW= {sw_c_ppm:.4}\n##$NUC1= <13C>\n##END=\n"
            ),
        )
        .unwrap();

        // Standard Bruker AQSEQ=0 (3-2-1): outer loop = F1 (`acqu3s`, C), inner loop = F2 (`acqu2s`, N)
        let mut ser = Vec::with_capacity(td_c * td_n * td_x * 4);
        let two_pi = 2.0 * std::f64::consts::PI;
        for tc in 0..td_c {
            let kc = (tc / 2) as f64;
            let is_sin_c = (tc % 2) == 1;
            let dec_c = (-kc / 14.0).exp();
            for tn in 0..td_n {
                let kn = (tn / 2) as f64;
                let is_sin_n = (tn % 2) == 1;
                let dec_n = (-kn / 14.0).exp();
                for kx in 0..n_x {
                    let kx_f = kx as f64;
                    let dec_x = (-kx_f / 64.0).exp();
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    for &(h_ppm, n_ppm, c_ppm) in peaks_hnc {
                        let fx = (carrier_h - h_ppm) / sw_h_ppm;
                        let fn_ = (carrier_n - n_ppm) / sw_n_ppm;
                        let fc = (carrier_c - c_ppm) / sw_c_ppm;
                        let mod_c = if is_sin_c {
                            -(two_pi * fc * kc).sin()
                        } else {
                            (two_pi * fc * kc).cos()
                        };
                        let mod_n = if is_sin_n {
                            -(two_pi * fn_ * kn).sin()
                        } else {
                            (two_pi * fn_ * kn).cos()
                        };
                        let amp = 40_000.0 * dec_x * dec_n * dec_c * mod_n * mod_c;
                        re += amp * (two_pi * fx * kx_f).cos();
                        im += -amp * (two_pi * fx * kx_f).sin();
                    }
                    ser.extend_from_slice(&(re.round() as i32).to_le_bytes());
                    ser.extend_from_slice(&(im.round() as i32).to_le_bytes());
                }
            }
        }
        fs::write(dir.join("ser"), &ser).unwrap();
    }

    #[test]
    fn test_raw_fid_process_pick_and_pipeline_cli() {
        let dir = env::temp_dir().join(format!("nmr_cli_raw_fid_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let bundle = dir.join("bundle");
        let hsqc_fid = bundle.join("10_hsqc");
        let hnco_fid = bundle.join("20_hnco");
        let peaks_hn = [(8.42, 120.5), (7.91, 115.2), (8.76, 124.8), (7.58, 111.4)];
        let peaks_hnc = [
            (8.42, 120.5, 175.2),
            (7.91, 115.2, 177.6),
            (8.76, 124.8, 173.4),
            (7.58, 111.4, 178.9),
        ];
        write_synthetic_bruker_hsqc_2d(&hsqc_fid, &peaks_hn);
        write_synthetic_bruker_hnco_3d(&hnco_fid, &peaks_hnc);

        // 1. `nmrassign process` 2D -> `.ft2` + `processing_recipe.json` + `--recipe` replay into `pdata`
        let out_2d_dir = dir.join("proc_2d");
        fs::create_dir_all(&out_2d_dir).unwrap();
        let ft2_path = out_2d_dir.join("hsqc.ft2");
        cmd_process(&[
            "--in".into(),
            hsqc_fid.to_string_lossy().into_owned(),
            "--out".into(),
            ft2_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(ft2_path.exists());
        let recipe_2d_path = out_2d_dir.join("processing_recipe.json");
        assert!(recipe_2d_path.exists());

        let pdata_2d_dir = dir.join("proc_2d_pdata");
        cmd_process(&[
            "--in".into(),
            hsqc_fid.to_string_lossy().into_owned(),
            "--out".into(),
            pdata_2d_dir.to_string_lossy().into_owned(),
            "--recipe".into(),
            recipe_2d_path.to_string_lossy().into_owned(),
            "--no-crop".into(),
        ])
        .unwrap();
        assert!(pdata_2d_dir.join("2rr").exists());
        assert!(pdata_2d_dir.join("processing_recipe.json").exists());

        // 2. `nmrassign process` 3D -> `.ft3` + `processing_recipe.json`
        let out_3d_dir = dir.join("proc_3d");
        fs::create_dir_all(&out_3d_dir).unwrap();
        let ft3_path = out_3d_dir.join("hnco.ft3");
        cmd_process(&[
            "--in".into(),
            hnco_fid.to_string_lossy().into_owned(),
            "--out".into(),
            ft3_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(ft3_path.exists());
        assert!(out_3d_dir.join("processing_recipe.json").exists());

        // 3. `nmrassign pick --fid <single_2d_dir>` and `nmrassign pick --fid <bundle_dir>`
        let picked_2d_tsv = dir.join("picked_2d.tsv");
        cmd_pick(&[
            "--fid".into(),
            hsqc_fid.to_string_lossy().into_owned(),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "cpu".into(),
            "--snr".into(),
            "4.0".into(),
            "-o".into(),
            picked_2d_tsv.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let pks_2d = read_peaks(&picked_2d_tsv).unwrap();
        assert!(
            pks_2d.len() >= 4,
            "expected >=4 picked 2D HSQC peaks from raw FID, got {}",
            pks_2d.len()
        );

        let picked_bundle_tsv = dir.join("picked_bundle.tsv");
        cmd_pick(&[
            "--fid".into(),
            bundle.to_string_lossy().into_owned(),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "cpu".into(),
            "--snr".into(),
            "4.0".into(),
            "-o".into(),
            picked_bundle_tsv.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let pks_bundle = read_peaks(&picked_bundle_tsv).unwrap();
        assert!(pks_bundle.iter().any(|p| p.exp == "HSQC"));
        assert!(pks_bundle.iter().any(|p| p.exp == "HNCO"));

        // 4. `nmrassign pipeline --fid-hsqc ... --fid-triple HNCO=...` AND `nmrassign pipeline --fid <bundle>`
        let out_pipe_explicit = dir.join("pipe_explicit");
        cmd_pipeline(&[
            "--seq".into(),
            "MQIFV".into(),
            "--fid-hsqc".into(),
            hsqc_fid.to_string_lossy().into_owned(),
            "--fid-triple".into(),
            format!("HNCO={}", hnco_fid.display()),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "cpu".into(),
            "--runs".into(),
            "2".into(),
            "--steps".into(),
            "200".into(),
            "--out-dir".into(),
            out_pipe_explicit.to_string_lossy().into_owned(),
        ])
        .unwrap();
        for fname in [
            "processing_recipe.json",
            "peaks.tsv",
            "shifts.str",
            "shifts.tsv",
            "result.nef",
            "report.html",
        ] {
            assert!(
                out_pipe_explicit.join(fname).exists(),
                "missing {fname} in --fid-hsqc pipeline out-dir"
            );
        }

        let out_pipe_bundle = dir.join("pipe_bundle");
        cmd_pipeline(&[
            "--sequence".into(),
            "MQIFV".into(),
            "--fid".into(),
            bundle.to_string_lossy().into_owned(),
            "--method".into(),
            "classical".into(),
            "--engine".into(),
            "cpu".into(),
            "--runs".into(),
            "2".into(),
            "--steps".into(),
            "200".into(),
            "--out-dir".into(),
            out_pipe_bundle.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert!(out_pipe_bundle.join("processing_recipe.json").exists());
        assert!(out_pipe_bundle.join("shifts.str").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    fn write_synthetic_unphased_bruker_hsqc_2d(
        dir: &Path,
        peaks_hn: &[(f64, f64)],
        phi0_x_deg: f64,
        phi0_y_deg: f64,
    ) {
        fs::create_dir_all(dir).unwrap();
        let td_x = 512usize;
        let n_x = td_x / 2;
        let n_y = 64usize;
        let td_y = n_y * 2;

        let bf1_h = 600.0f64;
        let carrier_h = 4.7f64;
        let sw_h_ppm = 12.0f64;
        let sw_h_hz = sw_h_ppm * bf1_h;
        let sfo1_h = bf1_h * (1.0 + carrier_h * 1e-6);

        let bf1_n = 60.0f64;
        let carrier_n = 118.0f64;
        let sw_n_ppm = 30.0f64;
        let sfo1_n = bf1_n * (1.0 + carrier_n * 1e-6);

        // Deliberately omit `pdata/1/procs` and `pdata/1/proc2s` to mimic raw BMRB archives.
        fs::write(dir.join("pulseprogram"), "; synthetic hsqcetfpf3gpsi\n").unwrap();
        fs::write(
            dir.join("acqus"),
            format!(
                "##TITLE= unphased HSQC\n##$TD= {td_x}\n##$BYTORDA= 0\n##$DTYPA= 0\n##$AQ_mod= 3\n##$SFO1= {sfo1_h:.8}\n##$BF1= {bf1_h:.8}\n##$SW_h= {sw_h_hz:.4}\n##$NUC1= <1H>\n##$GRPDLY= 0.0\n##END=\n"
            ),
        )
        .unwrap();
        fs::write(
            dir.join("acqu2s"),
            format!(
                "##TITLE= unphased F1\n##$TD= {td_y}\n##$FnMODE= 4\n##$SFO1= {sfo1_n:.8}\n##$BF1= {bf1_n:.8}\n##$SW= {sw_n_ppm:.4}\n##$NUC1= <15N>\n##END=\n"
            ),
        )
        .unwrap();

        let mut ser = Vec::with_capacity(td_y * td_x * 4);
        let two_pi = 2.0 * std::f64::consts::PI;
        let phi_x = phi0_x_deg.to_radians();
        let phi_y = phi0_y_deg.to_radians();
        for ty in 0..td_y {
            let ky = (ty / 2) as f64;
            let is_sin_y = (ty % 2) == 1;
            let dec_y = (-ky / 24.0).exp();
            for kx in 0..n_x {
                let kx_f = kx as f64;
                let dec_x = (-kx_f / 64.0).exp();
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for &(h_ppm, n_ppm) in peaks_hn {
                    let fx = (carrier_h - h_ppm) / sw_h_ppm;
                    let fy = (carrier_n - n_ppm) / sw_n_ppm;
                    let mod_y = if is_sin_y {
                        -(two_pi * fy * ky + phi_y).sin()
                    } else {
                        (two_pi * fy * ky + phi_y).cos()
                    };
                    let amp = 50_000.0 * dec_x * dec_y * mod_y;
                    re += amp * (two_pi * fx * kx_f + phi_x).cos();
                    im += -amp * (two_pi * fx * kx_f + phi_x).sin();
                }
                ser.extend_from_slice(&(re.round() as i32).to_le_bytes());
                ser.extend_from_slice(&(im.round() as i32).to_le_bytes());
            }
        }
        fs::write(dir.join("ser"), &ser).unwrap();
    }

    fn make_cli_test_cube_3d(
        exp: &str,
        c_hi: f64,
        c_lo: f64,
        gaussians_sigma: &[(f64, f64, f64, f32)],
    ) -> Spectrum3D {
        use nmr_core::picker::estimate_noise_mad;
        use nmr_core::spectrum::AxisPpm;

        let (nx, ny, nz) = (160usize, 64usize, 384usize);
        let axis_x = AxisPpm::from_bounds("1H", nx, 10.2, 5.8, 600.0);
        let axis_y = AxisPpm::from_bounds("15N", ny, 130.0, 106.0, 60.0);
        let axis_z = AxisPpm::from_bounds("13C", nz, c_hi, c_lo, 150.0);
        let mut data = vec![0.0f32; nx * ny * nz];
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (exp.len() as u64 * 0x517C_C1B7_2722_0A95);
        for v in &mut data {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (((state >> 40) & 0xFFFF) as f32) / 65536.0 - 0.5;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (((state >> 40) & 0xFFFF) as f32) / 65536.0 - 0.5;
            *v = u1 + u2;
        }
        let sigma = estimate_noise_mad(&data);
        for &(h0, n0, c0, snr_mult) in gaussians_sigma {
            let amp = snr_mult * sigma;
            let cx = axis_x.ppm_to_index(h0);
            let cy = axis_y.ppm_to_index(n0);
            let cz = axis_z.ppm_to_index(c0);
            for iz in (cz.floor() as isize - 4).max(0) as usize
                ..=(cz.ceil() as isize + 4).min(nz as isize - 1) as usize
            {
                for iy in (cy.floor() as isize - 3).max(0) as usize
                    ..=(cy.ceil() as isize + 3).min(ny as isize - 1) as usize
                {
                    for ix in (cx.floor() as isize - 3).max(0) as usize
                        ..=(cx.ceil() as isize + 3).min(nx as isize - 1) as usize
                    {
                        if (ix as f64 - cx).abs() <= 1.3
                            && (iy as f64 - cy).abs() <= 1.3
                            && (iz as f64 - cz).abs() <= 1.3
                        {
                            data[(iz * ny + iy) * nx + ix] = 0.0;
                        }
                        let dx = (ix as f64 - cx) / 1.45;
                        let dy = (iy as f64 - cy) / 1.45;
                        let dz = (iz as f64 - cz) / 1.55;
                        let g = (-0.5 * (dx * dx + dy * dy + dz * dz)) as f32;
                        data[(iz * ny + iy) * nx + ix] += amp * g.exp();
                    }
                }
            }
        }
        Spectrum3D {
            name: exp.to_string(),
            exp: exp.to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        }
    }

    #[test]
    fn test_process_auto_phase_dsp_flags_and_two_pass_corroborated_3d_cli() {
        let dir = env::temp_dir().join(format!("nmr_cli_phase8_int_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let peaks_hn = [(8.42, 120.5), (7.91, 115.2), (8.76, 124.8), (7.58, 111.4)];
        let unphased_hsqc = dir.join("unphased_hsqc");
        write_synthetic_unphased_bruker_hsqc_2d(&unphased_hsqc, &peaks_hn, 68.0, -34.0);

        // 1. `nmrassign process --auto-phase --hilbert --nus-method fista`
        let auto_dir = dir.join("proc_auto");
        fs::create_dir_all(&auto_dir).unwrap();
        let auto_ft2 = auto_dir.join("hsqc_auto.ft2");
        cmd_process(&[
            "--in".into(),
            unphased_hsqc.to_string_lossy().into_owned(),
            "--out".into(),
            auto_ft2.to_string_lossy().into_owned(),
            "--auto-phase".into(),
            "--hilbert".into(),
            "--nus-method".into(),
            "fista".into(),
        ])
        .unwrap();
        let auto_recipe =
            ProcessingRecipe::read_json(auto_dir.join("processing_recipe.json")).unwrap();
        assert_eq!(
            auto_recipe.dims[0].phase_source,
            nmr_core::process::PhaseSource::Auto
        );
        assert!(auto_recipe.dims[0].hilbert);
        assert_eq!(
            auto_recipe.nus.as_ref().map(|n| n.fista_momentum),
            Some(true)
        );
        let auto_spec = read_nmrpipe_2d(&auto_ft2).unwrap();
        let auto_pks = pick_hsqc_2d(&auto_spec, &PickerConfig::default());
        for &(h0, n0) in &peaks_hn {
            assert!(
                auto_pks
                    .iter()
                    .any(|p| (p.h - h0).abs() < 0.12 && (p.n - n0).abs() < 0.75),
                "blind 2D auto-phased HSQC missed ({h0}, {n0}); picked={auto_pks:?}"
            );
        }

        // 2. Explicit `--phase-x`, `--phase-y`, `--nus-method ist`, and 2D `--phase-z` rejection
        let exp_dir = dir.join("proc_explicit");
        fs::create_dir_all(&exp_dir).unwrap();
        let exp_ft2 = exp_dir.join("hsqc_exp.ft2");
        cmd_process(&[
            "--in".into(),
            unphased_hsqc.to_string_lossy().into_owned(),
            "--out".into(),
            exp_ft2.to_string_lossy().into_owned(),
            "--phase-x".into(),
            "-68.0:0.0".into(),
            "--phase-y".into(),
            "34.0:0.0".into(),
            "--nus-method".into(),
            "ist".into(),
        ])
        .unwrap();
        let exp_recipe =
            ProcessingRecipe::read_json(exp_dir.join("processing_recipe.json")).unwrap();
        assert_eq!(
            exp_recipe.dims[0].phase_source,
            nmr_core::process::PhaseSource::Explicit
        );
        assert_eq!(
            exp_recipe.nus.as_ref().map(|n| n.fista_momentum),
            Some(false)
        );
        assert!(
            cmd_process(&[
                "--in".into(),
                unphased_hsqc.to_string_lossy().into_owned(),
                "--out".into(),
                exp_ft2.to_string_lossy().into_owned(),
                "--phase-z".into(),
                "0.0:0.0".into(),
            ])
            .is_err(),
            "expected --phase-z on a 2D FID to return an error"
        );

        // 3. End-to-end two-pass corroborated 3D re-picking via `nmrassign pick --auto-phase`:
        //    First pick 2D roots via `cmd_pick --fid <unphased_hsqc> --auto-phase` so the 3D cubes
        //    share the exact root coordinates produced by `cmd_pick --auto-phase`.
        let roots_tsv = dir.join("roots_2d.tsv");
        cmd_pick(&[
            "--fid".into(),
            unphased_hsqc.to_string_lossy().into_owned(),
            "--auto-phase".into(),
            "--method".into(),
            "classical".into(),
            "--snr".into(),
            "5.0".into(),
            "-o".into(),
            roots_tsv.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let roots_2d = read_peaks(&roots_tsv).unwrap();
        let r0 = roots_2d
            .iter()
            .find(|p| (p.h - 8.42).abs() < 0.12 && (p.n - 120.5).abs() < 0.75)
            .unwrap();
        let hnca_ft3 = dir.join("hnca.ft3");
        let cbcaconh_ft3 = dir.join("cbcaconh.ft3");
        let hnca_cube = make_cli_test_cube_3d(
            "HNCA",
            68.0,
            44.0,
            &[(r0.h, r0.n, 56.2, 4.8), (r0.h, r0.n, 63.8, 4.9)],
        );
        let cbcaconh_cube = make_cli_test_cube_3d(
            "CBCACONH",
            72.0,
            16.0,
            &[(r0.h, r0.n, 56.2, 16.0), (r0.h, r0.n, 31.4, 15.0)],
        );
        write_nmrpipe_3d(&hnca_ft3, &hnca_cube).unwrap();
        write_nmrpipe_3d(&cbcaconh_ft3, &cbcaconh_cube).unwrap();

        let single_tsv = dir.join("single_3d.tsv");
        cmd_pick(&[
            "--fid-hsqc".into(),
            unphased_hsqc.to_string_lossy().into_owned(),
            "--triple".into(),
            format!("HNCA={}", hnca_ft3.display()),
            "--auto-phase".into(),
            "--method".into(),
            "classical".into(),
            "--snr".into(),
            "5.5".into(),
            "-o".into(),
            single_tsv.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let single_pks = read_peaks(&single_tsv).unwrap();
        assert_eq!(
            single_pks.iter().filter(|p| p.exp == "HNCA").count(),
            0,
            "single-experiment pick at --snr 5.5 must not pick 4.8-sigma peaks without corroboration"
        );

        let bundle_tsv = dir.join("corroborated_bundle.tsv");
        cmd_pick(&[
            "--fid-hsqc".into(),
            unphased_hsqc.to_string_lossy().into_owned(),
            "--triple".into(),
            format!("HNCA={}", hnca_ft3.display()),
            "--triple".into(),
            format!("CBCACONH={}", cbcaconh_ft3.display()),
            "--auto-phase".into(),
            "--method".into(),
            "classical".into(),
            "--snr".into(),
            "5.5".into(),
            "-o".into(),
            bundle_tsv.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let bundle_pks = read_peaks(&bundle_tsv).unwrap();
        let hnca_pks: Vec<&Peak> = bundle_pks.iter().filter(|p| p.exp == "HNCA").collect();
        assert!(
            hnca_pks
                .iter()
                .any(|p| p.c.is_some_and(|c| (c - 56.2).abs() < 0.6)),
            "two-pass corroborated 3D picker failed to rescue 4.1-sigma HNCA CA peak at 56.2 ppm; got {hnca_pks:?}"
        );
        assert!(
            !hnca_pks
                .iter()
                .any(|p| p.c.is_some_and(|c| (c - 63.8).abs() < 0.8)),
            "two-pass corroborated 3D picker falsely rescued uncorroborated 63.8 ppm decoy: {hnca_pks:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
