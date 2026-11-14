//! Command-line interface (`nmrassign`) for GPU/CPU NMR backbone & sidechain resonance assignment,
//! 2D/3D classical & neural CNN peak picking, multi-protein benchmarking, 3D NOESY + torsion-angle
//! structure folding, CCPN NEF v1.1 interchange, HTML quality reporting, self-hosted HTTP service,
//! and Octet-Draw 2D chemical structure / IUPAC / 1D NMR analytics.
//!
//! Provenance: Original CLI implementing `docs/specs/01-data-io.md` through `docs/specs/15-structure-viewer.md`
//! and Octet-Draw CLI workflows.

pub mod methyl_cmd;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, simulate_idp_13c_cubes, C13DetectedExp,
    C13PickedDataset, C13PickerConfig, C13Strip3DPeak, Caco2DPeak, CacoPeakKind, Con2DPeak,
    IdpCubeSimConfig,
};
use nmr_core::dihedrals::{
    predict_dihedrals, validate_dihedrals_against_torsions, DihedralRestraint,
};
use nmr_core::evaluate::{compare_shifts, score, to_reference, Metrics};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::fid::{read_fid_2d, read_fid_3d, read_fid_auto, Fid2D, Fid3D, RawFid};
use nmr_core::idp_propensity::{
    compute_idp_propensity, format_propensity_json, format_propensity_tsv, PropensityConfig,
};
use nmr_core::idp_solver::{
    assign_idp_13c_cpu, idp_to_reference, register_and_build_con_dipeptide_systems,
    ConDipeptideSystem, IdpAssignResult, IdpSolverConfig,
};
use nmr_core::nef::{read_nef_file, write_nef_file, NefDocument};
use nmr_core::neural_picker::{
    run_e2e_blind_benchmark, E2eBenchmarkConfig, NeuralPicker, NeuralPickerConfig,
};
use nmr_core::nmrstar::{
    read_star, read_star_all_atoms, read_star_str, write_star, write_star_with_header, Reference,
};
use nmr_core::noesy::{
    compute_chain_torsions, parse_pdb_first_model, parse_pdb_str_first_model, simulate_noesy_peaks,
    summarize_adrs, NoesySimConfig, Vec3,
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
use nmr_core::paramagnetic::{
    assign_paramagnetic_backbone, assign_paramagnetic_from_peaks, fit_pcs_tensor_svd,
    parse_metal_ligands, refine_metal_position_and_tensor, DeltaChiTensor,
    ParamagneticAssignConfig, ParamagneticAssignReport, ParamagneticSolverMode, PcsObservation,
};
use nmr_core::peaks::{read_peaks, write_peaks, Peak, EXPERIMENT_NAMES};
use nmr_core::picker::{
    corroborated_re_pick_3d, pick_c13_hsqc_2d, pick_hsqc_2d, CorroboratedRepickConfig,
    PickerConfig, StripBank3D,
};
use nmr_core::priors::{
    compute_priors, estimate_reference_offsets_detailed, fit_unlabelled_global_offsets_detailed,
    one_to_three, sync_detailed_reference_offsets, DetailedReferenceOffsets, Priors, ShiftOffsets,
};
use nmr_core::process::{
    process_fid_2d_mut, process_fid_3d_mut, register_3d_hn_to_hsqc_2d_with_audit,
    AxisRescueAuditRecord, ProcessingRecipe,
};
use nmr_core::qc::{
    apply_dataset_qc_to_assignments, evaluate_dataset_qc, DatasetQcReport, DatasetQcVerdict,
    ProteinReliabilityCurve, EVO_GRAPH_CALIBRATED_RUNS, STANDARD_CPU_CALIBRATED_RUNS,
    STANDARD_GPU_CALIBRATED_RUNS,
};
use nmr_core::rdc::{
    count_unresolved_rdc_restraints, evaluate_rdc_validation, partition_rdc_free_set, read_rdc_tsv,
    RdcMediumSet, RdcValidationSummary,
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
    build_backbone_from_torsions, calculate_structure_iterative,
    calculate_structure_with_rdcs_and_dihedrals, validate_ensemble, write_pdb_ensemble,
    StructureCalcConfig, StructureEnsembleResult,
};
use nmr_core::topology::SidechainPriors;
use nmr_core::torsion_db::{
    format_torsions_json_with_options, predict_torsions, predict_torsions_with_firewall,
    write_torsions_tsv_with_options, HomologFirewallQuery, TorsionExportOptions,
    TorsionPredictionClass,
};
use nmr_gpu::neural_gpu::{GpuNeuralPicker, GpuNeuralPickerConfig, PRETRAINED_WEIGHTS_3D_GPU};
use nmr_gpu::rdc_gpu::refine_ensemble_with_rdcs_gpu;
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
pub enum EngineRequest {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Standard,
    EvoGraph,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegimeKind {
    #[default]
    Globular,
    Idp13C,
    Paramagnetic,
}

#[derive(Clone, Debug)]
pub struct ReportOptions {
    pub include_idp_propensity_track: bool,
    pub temp_k: f64,
    pub ph: Option<f64>,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            include_idp_propensity_track: false,
            temp_k: 298.15,
            ph: None,
        }
    }
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
       nmrassign benchmark       --star <ref.str> [--realistic] [--seeds 1] [--solver standard|evo-graph] [--engine gpu|cpu] [--runs 16] [--steps 1500] [--assign-cost 0.0] [--missing 0.06] [--spurious 0.04] [--experiments HSQC,HNCACB,...] [--priors priors.json] [--seed 0] [-v]\n\
       nmrassign benchmark-suite [--manifest data/benchmark/manifest.tsv] [--solver standard|evo-graph] [--engine gpu|cpu] [--runs 16] [--steps 1500] [--missing 0.06] [--spurious 0.04] [--seed 0]\n\
       nmrassign real-benchmark  [--manifest data/benchmark/real_manifest.tsv] --split dev|heldout [--cohort <name>] [--solver evo-graph|standard] [--engine gpu|cpu] [--seeds 5] [--data-root data/private/bmrb] [--no-crop] [--lp] [--json-out <path>]\n\
       nmrassign fetch-bmrb      --id <bmrb_id> [--dest <dir>] [--dry-run] [--all-archives] [--max-bytes <bytes>]\n\
       nmrassign process         --in <fid_dir> --out <path_or_dir> [--exp <EXP>] [--recipe <json>] [--crop-h 5.5:11.5] [--no-crop] [--lp] [--auto-phase] [--phase-x P0:P1] [--phase-y P0:P1] [--phase-z P0:P1] [--hilbert] [--nus-method fista|ist]\n\
       nmrassign e2e-benchmark   [--star data/fixtures/synthetic_ubiquitin.str] [--noise 1.0] [--seed 42]\n\
       nmrassign assign          [--peaks <peaks.tsv>] (--seq <SEQ|fasta> | --star <ref.str>) --out <prefix> [--regime globular|idp-13c|paramagnetic] [--metal-ligands <SPEC>] [--r-bleach 11.0] [--pdb <model.pdb>] [--pcs <pcs.tsv>] [--temp-k 298.15] [--ph <f64>] [--con <path>] [--caco <path>] [--canco <path>] [--cbcacon <path>] [--cccon <path>] [--ncaco <path>] [--solver evo-graph|standard] [--engine auto|gpu|cpu] [--runs 16] [--steps 1500] [--assign-cost 0.0] [--priors priors.json] [-v]\n\
       nmrassign propensity      --shifts <assigned.str|assigned.nef> [--temp-k 298.15] [--ph <f64>] [--format tsv|json] [-o <out_path>]\n\
       nmrassign torsions        --shifts <assigned.str|assigned.nef> -o <out.tsv|out.nef|out.json> [--include-chi1] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [--exclude-topology <id>]\n\
       nmrassign score           --pred <pred.str> --truth <truth.str|table.txt> [--peaks <peaks.tsv>] [--assign-tsv <pred.tsv>] [--hc 0.8] [--no-auto-offset] [-v]\n\
       nmrassign simulate        --star <ref.str> --out <peaks.tsv> [--realistic] [--spectra-dir <dir>] [--experiments ...] [--seed 0]\n\
       nmrassign pick            (--hsqc <path> [--triple EXP=path ...] | --fid <dir> [--exp <EXP>] | --fid-hsqc <dir> [--fid-triple EXP=dir ...]) --out <peaks.tsv> [--recipe <json>] [--no-crop] [--lp] [--auto-phase] [--method classical|neural] [--engine cpu|gpu] [--snr 5.0] [--prob 0.45]\n\
       nmrassign priors          --stars <dir_or_file> --out <priors.json>\n\
       nmrassign sidechain       --star <ref.str> [--peaks <peaks.tsv>] [--aromatic] [--ph <f64>] [--arom-c13hsqc <path>] [--hbcbgcdhd <path>] [--hbcbgcdce <path>] [--aromatic-tocsy <path>] [--aromatic-noesy <path>] [--his-hmqc <path>] [--trp-hsqc <path>] [--stereospec] [--j-couplings <j.tsv>] [--intra-noes <noes.tsv>] [--pdb <model.pdb>] [--out <all_atom.str>] [--runs 16] [--steps 2500] [--seed 2026]\n\
       nmrassign methyl          --pdb <model.pdb> --hmqc <2d_or_tsv> --noesy <3d_or_tsv> [--labeling ILV-proS|ILV-proR|ILV-racemic|ILVMAT|<RES:ATOM,...>] [--hmcmcbca <hmcm.tsv>] [--engine gpu|cpu] [--seeds <N>] [-o <out.str>]\n\
       nmrassign structure       --star <shifts.str> [--noesy-peaks <noesy.tsv>] [--rdc <rdcs.tsv|rdcs.nef>] [--torsion-db] [--include-chi1] [--rdc-weight 0.75] [--rdc-free-frac 0.20] [--engine auto|gpu|cpu] [--pdb-ref <ref.pdb>] [--out-pdb <ensemble.pdb>] [--conformers 10] [--seed 42] [--dihedrals-only]\n\
       nmrassign nef-import      --in <project.nef> [--out-star <shifts.str>] [--out-peaks <peaks.tsv>]\n\
       nmrassign nef-export      --star <shifts.str> [--peaks <peaks.tsv>] --out <project.nef> [--name <entry>]\n\
       nmrassign report          --star|--shifts <shifts.str|shifts.nef> [--peaks <peaks.tsv>] [--pdb <ensemble.pdb>] [--rdc <rdcs.tsv|rdcs.nef>] [--idp-propensity] [--temp-k 298.15] [--ph <f64>] --out <report.html> [--title <name>]\n\
       nmrassign pipeline        (--seq <SEQ|fasta> (--hsqc <path> [--triple EXP=path ...] | --fid-hsqc <dir> [--fid-triple EXP=dir ...] | --fid <dir> | --peaks <peaks.tsv>) | --star <entry.str> | --nef <project.nef>) --out-dir <dir> [--regime globular|idp-13c|paramagnetic] [--metal-ligands <SPEC>] [--pdb <model.pdb>] [--pcs <pcs.tsv>] [--temp-k 298.15] [--ph <f64>] [--idp-propensity] [--recipe <json>] [--no-crop] [--lp] [--auto-phase] [--solver evo-graph|standard] [--method classical|neural] [--engine auto|gpu|cpu] [--fold] [--pdb-ref <ref.pdb>]\n\
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

pub(crate) fn parse_engine_request(val: &str) -> Result<EngineRequest, String> {
    match val.to_ascii_lowercase().as_str() {
        "auto" => Ok(EngineRequest::Auto),
        "gpu" | "wgpu" | "webgpu" => Ok(EngineRequest::Gpu),
        "cpu" => Ok(EngineRequest::Cpu),
        other => Err(format!(
            "unknown --engine {other:?}; expected auto, gpu, or cpu"
        )),
    }
}

pub(crate) fn parse_regime(val: &str) -> Result<RegimeKind, String> {
    match val.trim().to_ascii_lowercase().as_str() {
        "globular" => Ok(RegimeKind::Globular),
        "idp-13c" | "idp_13c" | "idp13c" => Ok(RegimeKind::Idp13C),
        "paramagnetic" => Ok(RegimeKind::Paramagnetic),
        other => Err(format!(
            "unknown --regime {other:?}; expected globular, idp-13c, or paramagnetic"
        )),
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

/// Minimum allowed `--runs` replica / population count across CLI subcommands (`P2`).
pub(crate) const CLI_MIN_RUNS: usize = 1;
/// Maximum allowed `--runs` replica / population count across CLI subcommands (`P2`).
pub(crate) const CLI_MAX_RUNS: usize = 4096;
/// Minimum allowed `--steps` annealing step count across CLI subcommands (`P2`).
pub(crate) const CLI_MIN_STEPS: usize = 100;
/// Maximum allowed `--steps` annealing step count across CLI subcommands (`P2`).
pub(crate) const CLI_MAX_STEPS: usize = 50_000_000;

pub(crate) fn parse_bounded_runs(raw: &str) -> Result<usize, String> {
    let val: usize = raw
        .parse()
        .map_err(|e| format!("invalid --runs value {raw:?}: {e} (expected integer in {CLI_MIN_RUNS}..={CLI_MAX_RUNS})"))?;
    if !(CLI_MIN_RUNS..=CLI_MAX_RUNS).contains(&val) {
        return Err(format!(
            "--runs {val} out of allowed bounds ({CLI_MIN_RUNS}..={CLI_MAX_RUNS})"
        ));
    }
    Ok(val)
}

pub(crate) fn parse_bounded_steps(raw: &str) -> Result<usize, String> {
    let val: usize = raw
        .parse()
        .map_err(|e| format!("invalid --steps value {raw:?}: {e} (expected integer in {CLI_MIN_STEPS}..={CLI_MAX_STEPS})"))?;
    if !(CLI_MIN_STEPS..=CLI_MAX_STEPS).contains(&val) {
        return Err(format!(
            "--steps {val} out of allowed bounds ({CLI_MIN_STEPS}..={CLI_MAX_STEPS})"
        ));
    }
    Ok(val)
}

pub(crate) fn parse_bounded_r_bleach(raw: &str) -> Result<f64, String> {
    let val: f64 = raw.parse().map_err(|e| {
        format!("invalid --r-bleach value {raw:?}: {e} (expected finite radius in 1.0..=50.0 Å)")
    })?;
    if !val.is_finite() || !(1.0..=50.0).contains(&val) {
        return Err(format!(
            "--r-bleach {val} out of allowed bounds (expected finite radius in 1.0..=50.0 Å)"
        ));
    }
    Ok(val)
}

/// Two-stage rigid `(dH, dN)` alignment of a 3D `(1H, 15N, 13C)` spectrum to a 2D `15N`-HSQC:
/// - Stage 1 runs empirical `{sign, fold}` indirect-axis likelihood verification (`enable_axis_rescue`)
///   plus conservative `(±0.08 ppm H, ±0.60 ppm N)` alignment so well-calibrated
///   single-spectrometer suites remain bit-for-bit unchanged.
/// - Stage 2 triggers strictly when Stage 1 rails against its search boundary (`|dH| >= 0.0795`
///   or `|dN| >= 0.595`), allowing cross-spectrometer / cross-temperature `O1` water shifts
///   (e.g. `5..45 °C` H2O carrier differences up to `±0.24 ppm H`, `±1.20 ppm N`) to converge
///   while preserving the Stage-1 [`AxisRescueAuditRecord`] (`13.3:P1-1`).
pub(crate) fn register_3d_hn_to_hsqc_adaptive_with_audit(
    spec_3d: &mut Spectrum3D,
    hsqc_2d: &Spectrum2D,
    enable_axis_rescue: bool,
) -> (f64, f64, Option<AxisRescueAuditRecord>) {
    let (dh1, dn1, audit1) =
        register_3d_hn_to_hsqc_2d_with_audit(spec_3d, hsqc_2d, 0.08, 0.60, enable_axis_rescue);
    if dh1.abs() >= 0.0795 || dn1.abs() >= 0.595 {
        let (dh2, dn2, _) =
            register_3d_hn_to_hsqc_2d_with_audit(spec_3d, hsqc_2d, 0.16, 0.60, false);
        (dh1 + dh2, dn1 + dn2, audit1)
    } else {
        (dh1, dn1, audit1)
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
            let base_detailed = estimate_reference_offsets_detailed(&systems, sequence, priors);
            let (effective_priors, offsets) =
                prepare_scoring_priors(&systems, sequence, priors, score_cfg);
            let detailed_offsets = sync_detailed_reference_offsets(&offsets, &base_detailed);
            let fit_runs = if engine == EngineKind::Gpu {
                STANDARD_GPU_CALIBRATED_RUNS
            } else {
                STANDARD_CPU_CALIBRATED_RUNS
            };
            let default_runs = if engine == EngineKind::Gpu {
                STANDARD_GPU_CALIBRATED_RUNS
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
                Some(&detailed_offsets),
            );
            if anneal_cfg.runs != fit_runs {
                qc_report.mark_uncalibrated_runs(&engine_label, anneal_cfg.runs, fit_runs);
                if let Some(ref w) = qc_report.calibration_warning {
                    eprintln!("Warning: [{engine_label}] confidence_calibrated=false — {w}");
                }
            }
            apply_dataset_qc_to_assignments(&mut result, &qc_report);
            qc_report.already_applied = true;
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
            let (systems, result, mut rep) =
                assign_evo_graph_from_peaks(peaks, sequence, priors, &evo_cfg);
            let engine_label = "CPU (f64, evo-graph)".to_string();
            // Also guard callers passing `--runs 1..=3` (clamped to 4 != 64) or any `r != 64`
            let requested_runs = runs_opt.unwrap_or(evo_cfg.population_size);
            if requested_runs != EVO_GRAPH_CALIBRATED_RUNS {
                rep.qc_report.mark_uncalibrated_runs(
                    &engine_label,
                    requested_runs,
                    EVO_GRAPH_CALIBRATED_RUNS,
                );
            }
            if let Some(ref w) = rep.qc_report.calibration_warning {
                eprintln!("Warning: [{engine_label}] confidence_calibrated=false — {w}");
            }
            let detailed_offsets = rep.qc_report.detailed_offsets;
            let offsets = ShiftOffsets {
                h: rep.final_offsets.d_h,
                n: rep.final_offsets.d_n,
                ca: rep.final_offsets.d_ca,
                cb: rep.final_offsets.d_cb,
                c: rep.final_offsets.d_c,
            };
            let effective_priors = rep.shifted_priors(priors);
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
    if offsets.is_zero()
        && qc_report.verdict == DatasetQcVerdict::Pass
        && qc_report.confidence_calibrated
    {
        return None;
    }
    let mut parts = vec![
        version_string(),
        format!("solver={solver_label}"),
        format!("qc_verdict={}", qc_report.verdict),
        format!("confidence_calibrated={}", qc_report.confidence_calibrated),
    ];
    if let Some(ref cw) = qc_report.calibration_warning {
        parts.push(format!("calibration_warning={cw}"));
    }
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
                runs = Some(parse_bounded_runs(
                    args.get(i).ok_or("missing value for --runs")?,
                )?);
            }
            "--steps" => {
                i += 1;
                steps = Some(parse_bounded_steps(
                    args.get(i).ok_or("missing value for --steps")?,
                )?);
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
            STANDARD_CPU_CALIBRATED_RUNS,
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

fn classify_caco_peak_kind(ca_ppm: f64, sign: i32) -> CacoPeakKind {
    if sign < 0 {
        CacoPeakKind::GlyOppositePhase
    } else if ca_ppm <= 47.8 {
        CacoPeakKind::GlyRegion
    } else if ca_ppm >= 58.0 {
        CacoPeakKind::SerThrProDownfield
    } else {
        CacoPeakKind::GeneralAliphatic
    }
}

pub(crate) fn parse_c13_detected_tsv_str(
    text: &str,
    default_exp: Option<C13DetectedExp>,
) -> Result<C13PickedDataset, String> {
    let mut ds = C13PickedDataset::default();
    for (line_no, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        let cols: Vec<&str> = if cols.len() <= 1 {
            line.split_whitespace().collect()
        } else {
            cols
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "experiment" | "exp" | "spectrum" | "peak_id" | "id"
        ) {
            continue;
        }

        let (exp, start_col) = if let Some(parsed_exp) = C13DetectedExp::from_name(cols[0]) {
            (parsed_exp, 1usize)
        } else if let Some(def) = default_exp {
            if cols[0].parse::<f64>().is_ok() {
                (def, 0usize)
            } else {
                return Err(format!(
                    "line {}: unknown 13C-detected experiment {:?}",
                    line_no + 1,
                    cols[0]
                ));
            }
        } else {
            return Err(format!(
                "line {}: unknown 13C-detected experiment {:?} (expected CON, CACO, CANCO, CBCACON, CCCON, or NCACO)",
                line_no + 1,
                cols[0]
            ));
        };

        let rem = &cols[start_col..];
        let parse_opt_f64 = |tok: &str| -> Result<Option<f64>, String> {
            let t = tok.trim();
            if t.is_empty() || t == "." || t == "-" || t.eq_ignore_ascii_case("nan") {
                Ok(None)
            } else {
                t.parse::<f64>()
                    .map(Some)
                    .map_err(|e| format!("line {}: invalid float {t:?}: {e}", line_no + 1))
            }
        };

        match exp {
            C13DetectedExp::Con2D => {
                if rem.len() < 2 {
                    return Err(format!(
                        "line {}: CON row needs at least C_prev and N_curr columns",
                        line_no + 1
                    ));
                }
                let c_prev_ppm = parse_opt_f64(rem[0])?
                    .ok_or_else(|| format!("line {}: missing CON C_prev ppm", line_no + 1))?;
                let n_curr_ppm = parse_opt_f64(rem[1])?
                    .ok_or_else(|| format!("line {}: missing CON N_curr ppm", line_no + 1))?;
                let has_dummy_col2 = rem.len() >= 6
                    && (parse_opt_f64(rem[2])?.is_none()
                        || (!matches!(rem[2], "1" | "-1" | "+1")
                            && matches!(rem[3], "1" | "-1" | "+1")));
                let meta_offset = if has_dummy_col2 { 3usize } else { 2usize };
                let height = rem
                    .get(meta_offset + 1)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0e5);
                let volume = rem
                    .get(meta_offset + 2)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(height);
                ds.con_roots.push(Con2DPeak {
                    c_prev_ppm,
                    n_curr_ppm,
                    height,
                    volume,
                    is_proline_imino_region: n_curr_ppm >= 131.5,
                    deconvolved_from_cluster: false,
                });
            }
            C13DetectedExp::Caco2D => {
                if rem.len() < 2 {
                    return Err(format!(
                        "line {}: CACO row needs at least C_curr and CA_curr columns",
                        line_no + 1
                    ));
                }
                let f0 = parse_opt_f64(rem[0])?
                    .ok_or_else(|| format!("line {}: missing CACO C_curr ppm", line_no + 1))?;
                let (d1, d2, meta_offset) = if rem.len() >= 3 && parse_opt_f64(rem[1])?.is_none() {
                    let ca = parse_opt_f64(rem[2])?
                        .ok_or_else(|| format!("line {}: missing CACO CA_curr ppm", line_no + 1))?;
                    (f0, ca, 3usize)
                } else {
                    let ca = parse_opt_f64(rem[1])?
                        .ok_or_else(|| format!("line {}: missing CACO CA_curr ppm", line_no + 1))?;
                    let off = if rem.len() >= 6 && parse_opt_f64(rem[2])?.is_none() {
                        3usize
                    } else {
                        2usize
                    };
                    (f0, ca, off)
                };
                let (c_curr_ppm, ca_curr_ppm) = if d1 < 100.0 && d2 > 140.0 {
                    (d2, d1)
                } else {
                    (d1, d2)
                };
                let sign = rem
                    .get(meta_offset)
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(1);
                let height = rem
                    .get(meta_offset + 1)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0e5);
                let volume = rem
                    .get(meta_offset + 2)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(height);
                ds.caco_peaks.push(Caco2DPeak {
                    c_curr_ppm,
                    ca_curr_ppm,
                    sign,
                    height,
                    volume,
                    kind: classify_caco_peak_kind(ca_curr_ppm, sign),
                });
            }
            C13DetectedExp::Canco3D
            | C13DetectedExp::Cbcaco3D
            | C13DetectedExp::Cccon3D
            | C13DetectedExp::Ncaco3D => {
                if rem.len() < 3 {
                    return Err(format!(
                        "line {}: {} row needs C_prime, N, and C_indirect columns",
                        line_no + 1,
                        exp.as_str()
                    ));
                }
                let c_prime_ppm = parse_opt_f64(rem[0])?.ok_or_else(|| {
                    format!("line {}: missing {} C_prime ppm", line_no + 1, exp.as_str())
                })?;
                let n_ppm = parse_opt_f64(rem[1])?.ok_or_else(|| {
                    format!("line {}: missing {} N ppm", line_no + 1, exp.as_str())
                })?;
                let c_indirect_ppm = parse_opt_f64(rem[2])?.ok_or_else(|| {
                    format!(
                        "line {}: missing {} C_indirect ppm",
                        line_no + 1,
                        exp.as_str()
                    )
                })?;
                let sign = rem.get(3).and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
                let height = rem
                    .get(4)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0e5);
                let volume = rem
                    .get(5)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(height);
                let anchor_root_idx = rem.get(6).and_then(|s| {
                    let t = s.trim();
                    if t.is_empty() || t == "." || t == "-" {
                        None
                    } else {
                        t.parse::<usize>().ok()
                    }
                });
                ds.strip_peaks.push(C13Strip3DPeak {
                    exp,
                    c_prime_ppm,
                    n_ppm,
                    c_indirect_ppm,
                    sign,
                    height,
                    volume,
                    anchor_root_idx,
                });
            }
        }
    }
    Ok(ds)
}

pub(crate) fn write_c13_detected_tsv(path: &Path, ds: &C13PickedDataset) -> Result<(), String> {
    let mut lines = vec![
        "experiment\tc_prime\tn15\tc_aliph\tsign\theight\tvolume\tanchor_root_idx".to_string(),
    ];
    for r in &ds.con_roots {
        lines.push(format!(
            "CON\t{:.4}\t{:.4}\t.\t1\t{:.1}\t{:.1}\t.",
            r.c_prev_ppm, r.n_curr_ppm, r.height, r.volume
        ));
    }
    for p in &ds.caco_peaks {
        lines.push(format!(
            "CACO\t{:.4}\t.\t{:.4}\t{}\t{:.1}\t{:.1}\t.",
            p.c_curr_ppm, p.ca_curr_ppm, p.sign, p.height, p.volume
        ));
    }
    for p in &ds.strip_peaks {
        let anc = p
            .anchor_root_idx
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| ".".to_string());
        lines.push(format!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.1}\t{:.1}\t{}",
            p.exp.as_str(),
            p.c_prime_ppm,
            p.n_ppm,
            p.c_indirect_ppm,
            p.sign,
            p.height,
            p.volume,
            anc
        ));
    }
    fs::write(path, lines.join("\n") + "\n").map_err(|e| format!("{e}"))
}

fn merge_c13_datasets(dst: &mut C13PickedDataset, src: C13PickedDataset) {
    let root_offset = dst.con_roots.len();
    dst.con_roots.extend(src.con_roots);
    dst.caco_peaks.extend(src.caco_peaks);
    for mut p in src.strip_peaks {
        if root_offset > 0 {
            if let Some(idx) = p.anchor_root_idx.as_mut() {
                *idx += root_offset;
            }
        }
        dst.strip_peaks.push(p);
    }
}

pub(crate) fn anchor_unanchored_c13_strip_peaks(ds: &mut C13PickedDataset) {
    if ds.con_roots.is_empty() {
        return;
    }
    for exp in [
        C13DetectedExp::Canco3D,
        C13DetectedExp::Cbcaco3D,
        C13DetectedExp::Cccon3D,
    ] {
        let unanchored_indices: Vec<usize> = ds
            .strip_peaks
            .iter()
            .enumerate()
            .filter(|(_, p)| p.exp == exp && p.anchor_root_idx.is_none())
            .map(|(idx, _)| idx)
            .collect();
        if unanchored_indices.is_empty() {
            continue;
        }
        let mut dcs = Vec::new();
        let mut dns = Vec::new();
        for &idx in &unanchored_indices {
            let p = &ds.strip_peaks[idx];
            let mut best: Option<(f64, f64, f64)> = None;
            for r in &ds.con_roots {
                let dc = r.c_prev_ppm - p.c_prime_ppm;
                let dn = r.n_curr_ppm - p.n_ppm;
                if dc.abs() <= 0.65 && dn.abs() <= 0.95 {
                    let d2 = (dc / 0.08).powi(2) + (dn / 0.30).powi(2);
                    if best.is_none_or(|b| d2 < b.0) {
                        best = Some((d2, dc, dn));
                    }
                }
            }
            if let Some((_, dc, dn)) = best {
                dcs.push(dc);
                dns.push(dn);
            }
        }
        let median_of = |mut v: Vec<f64>| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        let dc_est = median_of(dcs);
        let dn_est = median_of(dns);

        for idx in unanchored_indices {
            let p = &mut ds.strip_peaks[idx];
            let c_corr = p.c_prime_ppm + dc_est;
            let n_corr = p.n_ppm + dn_est;
            let mut best_root: Option<(usize, f64)> = None;
            for (r_idx, r) in ds.con_roots.iter().enumerate() {
                let dc = (c_corr - r.c_prev_ppm).abs();
                let dn = (n_corr - r.n_curr_ppm).abs();
                if dc <= 0.18 && dn <= 0.58 {
                    let d2 = (dc / 0.055).powi(2) + (dn / 0.24).powi(2);
                    if best_root.is_none_or(|(_, b)| d2 < b) {
                        best_root = Some((r_idx, d2));
                    }
                }
            }
            p.anchor_root_idx = best_root.map(|(r_idx, _)| r_idx);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_c13_input_bundle(
    peaks_paths: &[PathBuf],
    con_path: Option<&Path>,
    caco_path: Option<&Path>,
    canco_path: Option<&Path>,
    cbcacon_path: Option<&Path>,
    cccon_path: Option<&Path>,
    ncaco_path: Option<&Path>,
    triples: &[(String, PathBuf)],
    snr: f32,
) -> Result<C13PickedDataset, String> {
    let picker_cfg = C13PickerConfig {
        snr_2d: snr,
        snr_3d: (snr * 0.90).max(3.5),
        ..C13PickerConfig::default()
    };
    let mut ds = C13PickedDataset::default();

    let load_one_2d = |path: &Path,
                       exp: C13DetectedExp,
                       out: &mut C13PickedDataset|
     -> Result<(), String> {
        if is_probable_peak_tsv(path) {
            let txt = fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let sub = parse_c13_detected_tsv_str(&txt, Some(exp))?;
            merge_c13_datasets(out, sub);
        } else {
            let spec2 = load_2d_spectrum_any(path)?;
            match exp {
                C13DetectedExp::Con2D => out.con_roots.extend(pick_con_2d(&spec2, &picker_cfg)),
                C13DetectedExp::Caco2D => out.caco_peaks.extend(pick_caco_2d(&spec2, &picker_cfg)),
                _ => unreachable!(),
            }
        }
        Ok(())
    };

    if let Some(p) = con_path {
        load_one_2d(p, C13DetectedExp::Con2D, &mut ds)?;
    }
    if let Some(p) = caco_path {
        load_one_2d(p, C13DetectedExp::Caco2D, &mut ds)?;
    }

    for raw_p in peaks_paths {
        let s = raw_p.to_string_lossy();
        for part in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let p = Path::new(part);
            let txt = fs::read_to_string(p)
                .map_err(|e| format!("failed to read {}: {e}", p.display()))?;
            let sub = parse_c13_detected_tsv_str(&txt, None)?;
            merge_c13_datasets(&mut ds, sub);
        }
    }

    for (exp_name, path) in triples {
        if let Some(exp) = C13DetectedExp::from_name(exp_name) {
            if exp.is_2d() {
                load_one_2d(path, exp, &mut ds)?;
            }
        }
    }

    let mut specs_3d: Vec<(C13DetectedExp, &Path)> = Vec::new();
    if let Some(p) = canco_path {
        specs_3d.push((C13DetectedExp::Canco3D, p));
    }
    if let Some(p) = cbcacon_path {
        specs_3d.push((C13DetectedExp::Cbcaco3D, p));
    }
    if let Some(p) = cccon_path {
        specs_3d.push((C13DetectedExp::Cccon3D, p));
    }
    if let Some(p) = ncaco_path {
        specs_3d.push((C13DetectedExp::Ncaco3D, p));
    }
    for (exp_name, path) in triples {
        let exp = C13DetectedExp::from_name(exp_name).ok_or_else(|| {
            format!(
                "unknown 13C-detected --triple experiment {exp_name:?} (expected CON, CACO, CANCO, CBCACON, CCCON, or NCACO)"
            )
        })?;
        if exp.is_3d() {
            specs_3d.push((exp, path.as_path()));
        }
    }

    for (exp, path) in specs_3d {
        if is_probable_peak_tsv(path) {
            let txt = fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let sub = parse_c13_detected_tsv_str(&txt, Some(exp))?;
            merge_c13_datasets(&mut ds, sub);
        } else {
            if ds.con_roots.is_empty() && exp != C13DetectedExp::Ncaco3D {
                return Err(format!(
                    "picking 3D {} spectrum {} requires 2D CON roots (--con <path> or CON peaks in --peaks)",
                    exp.as_str(),
                    path.display()
                ));
            }
            let spec3 = load_3d_spectrum_any(path, exp.as_str())?;
            let strips = pick_con_anchored_3d(&spec3, exp, &ds.con_roots, &picker_cfg);
            ds.strip_peaks.extend(strips);
        }
    }

    anchor_unanchored_c13_strip_peaks(&mut ds);
    if ds.con_roots.is_empty() {
        return Err(
            "assign --regime idp-13c requires 2D CON roots (via --peaks <c13_peaks.tsv> or --con <spectrum>)"
                .to_string(),
        );
    }
    Ok(ds)
}

pub(crate) struct IdpWorkflowOutput {
    pub(crate) systems: Vec<ConDipeptideSystem>,
    pub(crate) result: IdpAssignResult,
    pub(crate) pred: Reference,
    pub(crate) conf: BTreeMap<usize, f64>,
    pub(crate) engine_label: String,
    pub(crate) star_header: String,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_idp_13c_workflow(
    dataset: &C13PickedDataset,
    sequence: &[String],
    seq_ids: &[i32],
    engine_req: EngineRequest,
    temp_k: f64,
    ph_opt: Option<f64>,
    runs: Option<usize>,
    steps: Option<usize>,
    seed: u64,
) -> Result<IdpWorkflowOutput, String> {
    if !temp_k.is_finite() || !(250.0..=360.0).contains(&temp_k) {
        return Err(format!(
            "invalid --temp-k {temp_k}: expected finite temperature in 250..=360 K"
        ));
    }
    if let Some(ph) = ph_opt {
        if !ph.is_finite() || !(1.0..=12.0).contains(&ph) {
            return Err(format!(
                "invalid --ph {ph}: expected finite pH in 1.0..=12.0"
            ));
        }
    }

    let systems = register_and_build_con_dipeptide_systems(dataset);
    if systems.is_empty() {
        return Err(
            "no CON dipeptide spin systems could be built from 13C-detected input".to_string(),
        );
    }

    let mut idp_cfg = IdpSolverConfig {
        temperature_k: temp_k,
        ph: ph_opt.unwrap_or(6.5),
        seed,
        ..IdpSolverConfig::default()
    };
    if let Some(r) = runs {
        idp_cfg.replicas = r.max(4);
    }
    if let Some(st) = steps {
        idp_cfg.steps_per_system = st.max(200);
    }

    let (result, engine_label) = match engine_req {
        EngineRequest::Gpu => {
            let gpu = GpuEngine::new_headless()?;
            let label = format!(
                "idp-13c WebGPU ({}, {})",
                gpu.adapter_name, gpu.backend_name
            );
            let res = nmr_gpu::idp_gpu::assign_idp_13c_gpu(&gpu, &systems, sequence, &idp_cfg)?;
            (res, label)
        }
        EngineRequest::Auto => match GpuEngine::new_headless() {
            Ok(gpu) => {
                let label = format!(
                    "idp-13c WebGPU ({}, {})",
                    gpu.adapter_name, gpu.backend_name
                );
                match nmr_gpu::idp_gpu::assign_idp_13c_gpu(&gpu, &systems, sequence, &idp_cfg) {
                    Ok(res) => (res, label),
                    Err(err) => {
                        eprintln!(
                            "Warning: WebGPU idp-13c solver failed ({err}); falling back to CPU (f64) solver."
                        );
                        (
                            assign_idp_13c_cpu(&systems, sequence, &idp_cfg),
                            "idp-13c CPU (f64, auto-fallback)".to_string(),
                        )
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "Warning: WebGPU unavailable for --regime idp-13c ({err}); falling back to CPU (f64) solver."
                );
                (
                    assign_idp_13c_cpu(&systems, sequence, &idp_cfg),
                    "idp-13c CPU (f64, auto-fallback)".to_string(),
                )
            }
        },
        EngineRequest::Cpu => (
            assign_idp_13c_cpu(&systems, sequence, &idp_cfg),
            "idp-13c CPU (f64)".to_string(),
        ),
    };

    let (pred, conf) = idp_to_reference(&result, &systems, sequence, seq_ids, 0.0, false);
    let n_pro_total = (1..sequence.len())
        .filter(|&k| sequence[k] == "PRO" || sequence[k - 1] == "PRO")
        .count();
    let n_pro_assigned = result
        .position
        .iter()
        .filter_map(|&p| usize::try_from(p).ok())
        .filter(|&k| {
            k > 0 && k < sequence.len() && (sequence[k] == "PRO" || sequence[k - 1] == "PRO")
        })
        .count();
    let star_header = format!(
        "# nmrassign solver: {engine_label}\n\
         # regime: idp-13c (temp_k={temp_k:.2}, ph={:.2}, pro_transitions={n_pro_assigned}/{n_pro_total})\n\
         # qc_verdict: {}\n\
         # fitted_offsets (peak_ppm - BMRB_prior_ppm): dCA={:+.3}, dCB={:+.3}, dC={:+.3}, dN={:+.3} ppm",
        idp_cfg.ph,
        result.qc_verdict,
        result.fitted_offsets.d_ca,
        result.fitted_offsets.d_cb,
        result.fitted_offsets.d_c,
        result.fitted_offsets.d_n
    );

    Ok(IdpWorkflowOutput {
        systems,
        result,
        pred,
        conf,
        engine_label,
        star_header,
    })
}

type PdbAtomCoordsBundle = (BTreeMap<(i32, String), Vec3>, Vec<Vec3>, Vec<Vec3>);

fn cluster_metal_atoms_single_linkage(atoms: &[Vec3], cutoff_angstrom: f64) -> Vec<Vec3> {
    let mut unique: Vec<Vec3> = Vec::new();
    for &c in atoms {
        if !unique.iter().any(|&u| {
            let dx = u[0] - c[0];
            let dy = u[1] - c[1];
            let dz = u[2] - c[2];
            dx * dx + dy * dy + dz * dz < 1e-10
        }) {
            unique.push(c);
        }
    }
    let m = unique.len();
    if m == 0 {
        return Vec::new();
    }
    let cut2 = cutoff_angstrom * cutoff_angstrom;
    let mut visited = vec![false; m];
    let mut centroids = Vec::new();
    for start in 0..m {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = vec![start];
        let mut head = 0usize;
        let mut sum = [0.0_f64; 3];
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            let cc = unique[cur];
            sum[0] += cc[0];
            sum[1] += cc[1];
            sum[2] += cc[2];
            for nxt in 0..m {
                if !visited[nxt] {
                    let nc = unique[nxt];
                    let d2 =
                        (cc[0] - nc[0]).powi(2) + (cc[1] - nc[1]).powi(2) + (cc[2] - nc[2]).powi(2);
                    if d2 <= cut2 {
                        visited[nxt] = true;
                        queue.push(nxt);
                    }
                }
            }
        }
        let inv = 1.0 / (queue.len() as f64);
        centroids.push([sum[0] * inv, sum[1] * inv, sum[2] * inv]);
    }
    centroids
}

fn parse_pdb_atom_coords_first_model(pdb_text: &str) -> PdbAtomCoordsBundle {
    let mut atom_map: BTreeMap<(i32, String), Vec3> = BTreeMap::new();
    let mut raw_metal_atoms: Vec<Vec3> = Vec::new();
    let mut ca_coords: Vec<Vec3> = Vec::new();
    let mut active_chain: Option<char> = None;

    // Open-shell paramagnetic transition-metal and lanthanide elements only
    // (excludes diamagnetic closed-shell CA, MG, ZN, LA, LU and non-metals S, C, N, O, SE).
    const PARAMAGNETIC_METAL_ELEMENTS: &[&str] = &[
        "FE", "CO", "CU", "NI", "MN", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
        "TM", "YB",
    ];
    const METAL_COFACTOR_RESIDUES: &[&str] = &[
        "FE", "FE2", "FES", "SF4", "F3S", "HEM", "HEC", "CO", "CU", "NI", "MN", "CE", "PR", "ND",
        "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    ];

    #[inline]
    fn pdb_ascii_col(bytes: &[u8], start: usize, end: usize) -> Option<&str> {
        let slice = bytes.get(start..end)?;
        if !slice.is_ascii() {
            return None;
        }
        std::str::from_utf8(slice).ok()
    }

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        let bytes = line.as_bytes();
        let is_atom = line.starts_with("ATOM  ");
        let is_het = line.starts_with("HETATM");
        if (!is_atom && !is_het) || bytes.len() < 54 {
            continue;
        }
        let alt = bytes[16];
        if alt != b' ' && alt != b'A' && alt != b'1' {
            continue;
        }
        let (Some(atom_col), Some(res_col), Some(x_col), Some(y_col), Some(z_col)) = (
            pdb_ascii_col(bytes, 12, 16),
            pdb_ascii_col(bytes, 17, 20),
            pdb_ascii_col(bytes, 30, 38),
            pdb_ascii_col(bytes, 38, 46),
            pdb_ascii_col(bytes, 46, 54),
        ) else {
            continue;
        };
        let Ok(x) = x_col.trim().parse::<f64>() else {
            continue;
        };
        let Ok(y) = y_col.trim().parse::<f64>() else {
            continue;
        };
        let Ok(z) = z_col.trim().parse::<f64>() else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let pos = [x, y, z];

        if is_het {
            let res_name = res_col.trim().to_ascii_uppercase();
            let atom_name = atom_col.trim().to_ascii_uppercase();
            let elem_col: String = if bytes.len() > 76 {
                let end = bytes.len().min(78);
                let Some(elem_raw) = pdb_ascii_col(bytes, 76, end) else {
                    continue;
                };
                elem_raw
                    .bytes()
                    .filter(|b| b.is_ascii_alphabetic())
                    .map(|b| (b as char).to_ascii_uppercase())
                    .collect()
            } else {
                String::new()
            };
            let inferred_elem = if !elem_col.is_empty() {
                Some(elem_col)
            } else if METAL_COFACTOR_RESIDUES.contains(&res_name.as_str()) {
                let lead: String = atom_name
                    .chars()
                    .take_while(|c| c.is_ascii_alphabetic())
                    .collect();
                (!lead.is_empty()).then_some(lead)
            } else {
                None
            };
            if inferred_elem
                .as_deref()
                .is_some_and(|el| PARAMAGNETIC_METAL_ELEMENTS.contains(&el))
            {
                raw_metal_atoms.push(pos);
            }
            continue;
        }

        let chain_byte = bytes[21];
        if !chain_byte.is_ascii() {
            continue;
        }
        let ch = chain_byte as char;
        if let Some(c0) = active_chain {
            if ch != c0 {
                continue;
            }
        } else {
            active_chain = Some(ch);
        }
        let Some(seq_col) = pdb_ascii_col(bytes, 22, 26) else {
            continue;
        };
        let Ok(seq_id) = seq_col.trim().parse::<i32>() else {
            continue;
        };
        let mut atom_name = atom_col.trim().to_ascii_uppercase();
        if atom_name == "HN" {
            atom_name = "H".to_string();
        }
        atom_map.entry((seq_id, atom_name.clone())).or_insert(pos);
        if atom_name == "CA" {
            ca_coords.push(pos);
        }
    }

    if let Ok(chain) = parse_pdb_str_first_model(pdb_text) {
        for r in &chain.residues {
            for (aname, opt_coord) in [
                ("H", r.h),
                ("N", Some(r.n)),
                ("CA", Some(r.ca)),
                ("CB", r.cb),
                ("C", Some(r.c)),
            ] {
                if let Some(coord) = opt_coord {
                    atom_map
                        .entry((r.seq_id, aname.to_string()))
                        .or_insert(coord);
                }
            }
        }
    }

    // Cluster multi-iron cofactors (e.g. FE1+FE2 of one [2Fe-2S] FES ~2.7 Å apart) into 1 centroid
    // while keeping distinct clusters (e.g. tandem [2Fe-2S] sites ~16 Å apart) separate.
    let metal_centroids = cluster_metal_atoms_single_linkage(&raw_metal_atoms, 4.5);

    (atom_map, metal_centroids, ca_coords)
}

fn fit_and_report_pcs_tensor_cli(
    pcs_path: &Path,
    pdb_text: &str,
    para_report: &ParamagneticAssignReport,
) -> Result<DeltaChiTensor, String> {
    let raw_pcs = fs::read_to_string(pcs_path)
        .map_err(|e| format!("failed to read PCS file {}: {e}", pcs_path.display()))?;
    let (atom_map, pdb_metals, ca_coords) = parse_pdb_atom_coords_first_model(pdb_text);
    if atom_map.is_empty() {
        return Err("no ATOM records parsed from --pdb for PCS tensor SVD fitting".to_string());
    }

    let mut observations = Vec::new();
    for (line_no, raw_line) in raw_pcs.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(['\t', ',']).map(str::trim).collect();
        let cols: Vec<&str> = if cols.len() <= 1 {
            line.split_whitespace().collect()
        } else {
            cols
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(low0.as_str(), "seq_id" | "resid" | "residue" | "seq") {
            continue;
        }
        if cols.len() < 2 {
            return Err(format!(
                "PCS line {}: expected at least seq_id and pcs_ppm columns",
                line_no + 1
            ));
        }

        let seq_id: i32 = cols[0].parse().map_err(|e| {
            format!(
                "PCS line {}: invalid seq_id {:?}: {e}",
                line_no + 1,
                cols[0]
            )
        })?;

        let parse_pcs_val = |raw: &str| -> Result<f64, String> {
            let v: f64 = raw
                .parse()
                .map_err(|e| format!("PCS line {}: invalid pcs_ppm {raw:?}: {e}", line_no + 1))?;
            if !v.is_finite() {
                return Err(format!(
                    "PCS line {}: non-finite pcs_ppm {raw:?}",
                    line_no + 1
                ));
            }
            Ok(v)
        };
        let parse_sigma_opt = |opt_raw: Option<&&str>| -> Result<f64, String> {
            let Some(&raw) = opt_raw else {
                return Ok(0.05);
            };
            if raw.is_empty() {
                return Ok(0.05);
            }
            let s: f64 = raw
                .parse()
                .map_err(|e| format!("PCS line {}: invalid sigma_ppm {raw:?}: {e}", line_no + 1))?;
            if !s.is_finite() || s <= 0.0 {
                return Err(format!(
                    "PCS line {}: sigma_ppm {raw:?} must be finite and > 0",
                    line_no + 1
                ));
            }
            Ok(s)
        };

        let (atom_name, pcs_ppm, sigma_ppm) = if cols[1].parse::<f64>().is_ok() {
            let v = parse_pcs_val(cols[1])?;
            let sig = parse_sigma_opt(cols.get(2))?;
            ("H".to_string(), v, sig)
        } else if cols.len() >= 3 && cols[2].parse::<f64>().is_ok() {
            let mut at = cols[1].to_ascii_uppercase();
            if at == "HN" {
                at = "H".to_string();
            }
            let v = parse_pcs_val(cols[2])?;
            let sig = parse_sigma_opt(cols.get(3))?;
            (at, v, sig)
        } else if cols.len() >= 4 {
            let mut at = cols[2].to_ascii_uppercase();
            if at == "HN" {
                at = "H".to_string();
            }
            let v = parse_pcs_val(cols[3])?;
            let sig = parse_sigma_opt(cols.get(4))?;
            (at, v, sig)
        } else {
            return Err(format!(
                "PCS line {}: unable to parse row {:?}",
                line_no + 1,
                raw_line
            ));
        };

        let coord = atom_map
            .get(&(seq_id, atom_name.clone()))
            .or_else(|| {
                if atom_name == "H" {
                    atom_map.get(&(seq_id, "N".to_string()))
                } else {
                    None
                }
            })
            .copied();
        if let Some(pos) = coord {
            observations
                .push(PcsObservation::new(seq_id, pos, pcs_ppm).with_uncertainty(sigma_ppm));
        }
    }

    if observations.len() < 6 {
        return Err(format!(
            "PCS 5-parameter SVD fit requires at least 6 matched PDB nuclear coordinates (>= 1 DoF), found {}",
            observations.len()
        ));
    }

    let mut candidate_metals: Vec<Vec3> = pdb_metals;
    if candidate_metals.is_empty() {
        candidate_metals.extend(
            para_report
                .detected_clusters
                .iter()
                .filter_map(|c| c.metal_coord),
        );
    }

    let tensor = if !candidate_metals.is_empty() {
        let mut best_tensor: Option<DeltaChiTensor> = None;
        let mut last_err: Option<String> = None;
        for mpos in candidate_metals {
            match fit_pcs_tensor_svd(&observations, mpos) {
                Ok(t) => {
                    if best_tensor.as_ref().is_none_or(|b| {
                        nmr_core::paramagnetic::pcs_q_is_better(t.q_factor, b.q_factor)
                    }) {
                        best_tensor = Some(t);
                    }
                }
                Err(e) => last_err = Some(e),
            }
        }
        match best_tensor {
            Some(t) => t,
            None => return Err(last_err.unwrap_or_else(|| "PCS SVD fit failed".to_string())),
        }
    } else {
        let ca_centroid = if ca_coords.is_empty() {
            [0.0, 0.0, 0.0]
        } else {
            let inv = 1.0 / (ca_coords.len() as f64);
            let sx: f64 = ca_coords.iter().map(|c| c[0]).sum();
            let sy: f64 = ca_coords.iter().map(|c| c[1]).sum();
            let sz: f64 = ca_coords.iter().map(|c| c[2]).sum();
            [sx * inv, sy * inv, sz * inv]
        };
        if observations.len() >= 8 {
            refine_metal_position_and_tensor(&observations, ca_centroid, 8.0, 2.0)?
        } else {
            fit_pcs_tensor_svd(&observations, ca_centroid)?
        }
    };

    let deg = |r: f64| r.to_degrees();
    println!(
        "PCS 5-param Δχ SVD fit ({} obs, metal=[{:.2}, {:.2}, {:.2}] Å): Δχ_ax={:+.3}, Δχ_rh={:+.3} (10^-32 m^3, rhombicity={:.3}) | PCS Q-factor={:.4} (RMSD={:.4} ppm) | Euler ZYZ=({:.1}°, {:.1}°, {:.1}°)",
        observations.len(),
        tensor.metal_pos_angstrom[0],
        tensor.metal_pos_angstrom[1],
        tensor.metal_pos_angstrom[2],
        tensor.dchi_ax,
        tensor.dchi_rh,
        tensor.rhombicity,
        tensor.q_factor,
        tensor.rmsd_ppm,
        deg(tensor.euler_zyz_rad[0]),
        deg(tensor.euler_zyz_rad[1]),
        deg(tensor.euler_zyz_rad[2]),
    );
    Ok(tensor)
}

pub(crate) type ParamagneticWorkflowOutput = (
    Vec<SpinSystem>,
    AssignResult,
    Priors,
    ParamagneticAssignReport,
    Option<DeltaChiTensor>,
    String,
);

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_paramagnetic_workflow(
    solver: SolverKind,
    engine_req: EngineRequest,
    peaks: &[Peak],
    sequence: &[String],
    seq_ids: &[i32],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    metal_ligands_spec: Option<&str>,
    r_bleach_opt: Option<f64>,
    pdb_path: Option<&Path>,
    pcs_path: Option<&Path>,
    runs: Option<usize>,
    steps: Option<usize>,
    seed: u64,
) -> Result<ParamagneticWorkflowOutput, String> {
    if engine_req == EngineRequest::Gpu {
        return Err(
            "--engine gpu is not supported with --regime paramagnetic (CPU-only); use --engine cpu or --engine auto"
                .to_string(),
        );
    }
    if pcs_path.is_some() && pdb_path.is_none() {
        return Err(
            "--pcs requires --pdb <model.pdb> for 5-parameter Δχ PCS tensor SVD fitting"
                .to_string(),
        );
    }

    let explicit_ligand_indices = if let Some(spec) = metal_ligands_spec {
        parse_metal_ligands(spec, sequence, seq_ids)?
    } else {
        Vec::new()
    };

    let pdb_coordinates = if let Some(p) = pdb_path {
        Some(
            fs::read_to_string(p)
                .map_err(|e| format!("failed to read PDB {}: {e}", p.display()))?,
        )
    } else {
        None
    };

    let explicit_metal_coords = if let Some(ref txt) = pdb_coordinates {
        parse_pdb_atom_coords_first_model(txt).1
    } else {
        Vec::new()
    };

    let mode = match solver {
        SolverKind::Standard => ParamagneticSolverMode::Standard,
        SolverKind::EvoGraph => ParamagneticSolverMode::EvoGraph,
    };
    let mut anneal_cfg = AnnealConfig {
        seed,
        ..AnnealConfig::default()
    };
    if let Some(r) = runs {
        anneal_cfg.runs = r.max(4);
    }
    if let Some(st) = steps {
        anneal_cfg.steps_per_system = st.max(200);
    }

    let mut para_cfg = ParamagneticAssignConfig {
        mode,
        explicit_ligand_indices: explicit_ligand_indices.clone(),
        auto_detect_motifs: explicit_ligand_indices.is_empty(),
        pdb_coordinates: pdb_coordinates.clone(),
        explicit_metal_coords,
        score_cfg: score_cfg.clone(),
        anneal_cfg,
        seed,
        ..ParamagneticAssignConfig::default()
    };
    para_cfg.r_bleach_angstrom = r_bleach_opt;

    let (systems, result, para_report, eff_priors, engine_label) = match solver {
        SolverKind::Standard => {
            let sys = build_spin_systems(peaks, &Tolerances::default(), 3, false)?;
            if sys.is_empty() {
                return Err("no spin systems built from peaks".to_string());
            }
            let (eff, _) = prepare_scoring_priors(&sys, sequence, priors, score_cfg);
            let (res, rep) = assign_paramagnetic_backbone(&sys, sequence, priors, &para_cfg);
            (
                sys,
                res,
                rep,
                eff,
                "paramagnetic-standard CPU (f64)".to_string(),
            )
        }
        SolverKind::EvoGraph => {
            let (sys, res, rep) =
                assign_paramagnetic_from_peaks(peaks, sequence, priors, &para_cfg);
            if sys.is_empty() {
                return Err("no spin systems built from peaks".to_string());
            }
            let eff = rep
                .evo_report
                .as_ref()
                .map(|er| er.shifted_priors(priors))
                .unwrap_or_else(|| prepare_scoring_priors(&sys, sequence, priors, score_cfg).0);
            (
                sys,
                res,
                rep,
                eff,
                "paramagnetic-evo-graph CPU (f64)".to_string(),
            )
        }
    };

    let fitted_tensor = if let (Some(pcs_p), Some(pdb_txt)) = (pcs_path, pdb_coordinates.as_deref())
    {
        Some(fit_and_report_pcs_tensor_cli(pcs_p, pdb_txt, &para_report)?)
    } else {
        None
    };

    Ok((
        systems,
        result,
        eff_priors,
        para_report,
        fitted_tensor,
        engine_label,
    ))
}

fn cmd_assign(args: &[String]) -> Result<(), String> {
    let mut peaks_paths: Vec<PathBuf> = Vec::new();
    let mut seq_arg: Option<String> = None;
    let mut star_arg: Option<String> = None;
    let mut out_arg: Option<String> = None;
    let mut priors_path: Option<String> = None;
    let mut first_seq_id = 1i32;
    let mut regime = RegimeKind::Globular;
    let mut metal_ligands_spec: Option<String> = None;
    let mut r_bleach_opt: Option<f64> = None;
    let mut pdb_path: Option<PathBuf> = None;
    let mut pcs_path: Option<PathBuf> = None;
    let mut temp_k = 298.15f64;
    let mut temp_k_explicit = false;
    let mut ph_opt: Option<f64> = None;
    let mut con_path: Option<PathBuf> = None;
    let mut caco_path: Option<PathBuf> = None;
    let mut canco_path: Option<PathBuf> = None;
    let mut cbcacon_path: Option<PathBuf> = None;
    let mut cccon_path: Option<PathBuf> = None;
    let mut ncaco_path: Option<PathBuf> = None;
    let mut c13_triples: Vec<(String, PathBuf)> = Vec::new();
    let mut snr = 5.0f32;
    let mut solver = SolverKind::EvoGraph;
    let mut solver_explicit = false;
    let mut engine_req_opt: Option<EngineRequest> = None;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
    let mut assign_cost = 0.0f64;
    let mut assign_cost_explicit = false;
    let mut seed = 0u64;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            "--peaks" => {
                i += 1;
                let p = args.get(i).ok_or("missing --peaks")?;
                peaks_paths.push(PathBuf::from(p));
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
            "--regime" => {
                i += 1;
                regime = parse_regime(args.get(i).ok_or("missing --regime")?)?;
            }
            "--metal-ligands" => {
                i += 1;
                metal_ligands_spec = Some(args.get(i).ok_or("missing --metal-ligands")?.clone());
            }
            "--r-bleach" => {
                i += 1;
                r_bleach_opt = Some(parse_bounded_r_bleach(
                    args.get(i).ok_or("missing --r-bleach")?,
                )?);
            }
            "--pdb" => {
                i += 1;
                pdb_path = Some(PathBuf::from(args.get(i).ok_or("missing --pdb")?));
            }
            "--pcs" => {
                i += 1;
                pcs_path = Some(PathBuf::from(args.get(i).ok_or("missing --pcs")?));
            }
            "--temp-k" | "--temperature-k" => {
                i += 1;
                temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                temp_k_explicit = true;
            }
            "--ph" => {
                i += 1;
                ph_opt = Some(
                    args.get(i)
                        .ok_or("missing --ph")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--con" => {
                i += 1;
                con_path = Some(PathBuf::from(args.get(i).ok_or("missing --con")?));
            }
            "--caco" => {
                i += 1;
                caco_path = Some(PathBuf::from(args.get(i).ok_or("missing --caco")?));
            }
            "--canco" => {
                i += 1;
                canco_path = Some(PathBuf::from(args.get(i).ok_or("missing --canco")?));
            }
            "--cbcacon" | "--cbcaco" => {
                i += 1;
                cbcacon_path = Some(PathBuf::from(args.get(i).ok_or("missing --cbcacon")?));
            }
            "--cccon" => {
                i += 1;
                cccon_path = Some(PathBuf::from(args.get(i).ok_or("missing --cccon")?));
            }
            "--ncaco" => {
                i += 1;
                ncaco_path = Some(PathBuf::from(args.get(i).ok_or("missing --ncaco")?));
            }
            "--triple" => {
                i += 1;
                let spec = args.get(i).ok_or("missing EXP=path for --triple")?;
                let (exp, p) = spec
                    .split_once('=')
                    .ok_or("expected EXP=path format for --triple")?;
                c13_triples.push((
                    exp.trim().to_ascii_uppercase().replace(['(', ')'], ""),
                    PathBuf::from(p),
                ));
            }
            "--snr" => {
                i += 1;
                snr = args
                    .get(i)
                    .ok_or("missing --snr")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing --solver")?)?;
                solver_explicit = true;
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
                engine_req_opt = Some(parse_engine_request(
                    args.get(i).ok_or("missing --engine")?,
                )?);
            }
            "--runs" => {
                i += 1;
                runs = Some(parse_bounded_runs(args.get(i).ok_or("missing --runs")?)?);
            }
            "--steps" => {
                i += 1;
                steps = Some(parse_bounded_steps(args.get(i).ok_or("missing --steps")?)?);
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args
                    .get(i)
                    .ok_or("missing --assign-cost")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                assign_cost_explicit = true;
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
            "--rdc" | "--torsion-db" | "--include-chi1" | "--rdc-weight" | "--rdc-free-frac" => {
                return Err(format!(
                    "{} is a post-assignment structure/report flag (backbone assignment is strictly structure-free; use `nmrassign structure`, `nmrassign torsions`, or `nmrassign report`)",
                    args[i]
                ))
            }
            other => return Err(format!("unknown assign flag {other:?}")),
        }
        i += 1;
    }

    if regime != RegimeKind::Paramagnetic {
        if pdb_path.is_some() {
            return Err(
                "--pdb is only supported with --regime paramagnetic (globular and idp-13c backbone assignment are strictly structure-free)"
                    .to_string(),
            );
        }
        if pcs_path.is_some() {
            return Err("--pcs is only supported with --regime paramagnetic".to_string());
        }
        if metal_ligands_spec.is_some() {
            return Err("--metal-ligands is only supported with --regime paramagnetic".to_string());
        }
        if r_bleach_opt.is_some() {
            return Err("--r-bleach is only supported with --regime paramagnetic".to_string());
        }
    }
    if regime != RegimeKind::Idp13C {
        let has_c13_spectra = con_path.is_some()
            || caco_path.is_some()
            || canco_path.is_some()
            || cbcacon_path.is_some()
            || cccon_path.is_some()
            || ncaco_path.is_some()
            || !c13_triples.is_empty();
        if has_c13_spectra {
            return Err(
                "13C-detected spectrum flags (--con, --caco, --canco, --cbcacon, --cccon, --ncaco, --triple) are only supported with --regime idp-13c"
                    .to_string(),
            );
        }
        if temp_k_explicit {
            return Err(
                "--temp-k is only supported with --regime idp-13c in `nmrassign assign`"
                    .to_string(),
            );
        }
        if ph_opt.is_some() {
            return Err(
                "--ph is only supported with --regime idp-13c in `nmrassign assign`".to_string(),
            );
        }
    } else {
        if solver_explicit {
            return Err(
                "--solver is not supported with --regime idp-13c (idp-13c uses its dedicated CON-dipeptide multi-replica annealer)"
                    .to_string(),
            );
        }
        if priors_path.is_some() {
            return Err(
                "--priors is not supported with --regime idp-13c (idp-13c uses sequence-specific ncIDP random-coil priors)"
                    .to_string(),
            );
        }
        if assign_cost_explicit {
            return Err("--assign-cost is not supported with --regime idp-13c".to_string());
        }
    }

    let out_prefix = PathBuf::from(out_arg.ok_or("assign requires --out <prefix>")?);
    let (sequence, seq_ids) = if let Some(st) = star_arg {
        let r = read_star(st, None)?;
        (r.sequence, r.seq_ids)
    } else if let Some(sq) = seq_arg {
        parse_sequence_arg(&sq, first_seq_id)?
    } else {
        return Err("assign needs --seq or --star".to_string());
    };

    let str_out = out_prefix.with_extension("str");
    let tsv_out = out_prefix.with_extension("tsv");

    match regime {
        RegimeKind::Idp13C => {
            let ds = load_c13_input_bundle(
                &peaks_paths,
                con_path.as_deref(),
                caco_path.as_deref(),
                canco_path.as_deref(),
                cbcacon_path.as_deref(),
                cccon_path.as_deref(),
                ncaco_path.as_deref(),
                &c13_triples,
                snr,
            )?;
            let engine_req = engine_req_opt.unwrap_or(EngineRequest::Auto);
            let wf = execute_idp_13c_workflow(
                &ds, &sequence, &seq_ids, engine_req, temp_k, ph_opt, runs, steps, seed,
            )?;
            write_star_with_header(&str_out, &wf.pred, Some(&wf.conf), Some(&wf.star_header))?;
            write_shifts_tsv(&tsv_out, &wf.pred, &wf.conf)?;
            let n_hc = wf.conf.values().filter(|&&c| c >= 0.8).count();
            let n_pro_res = sequence.iter().filter(|r| r.as_str() == "PRO").count();
            let n_pro_with_shifts = sequence
                .iter()
                .enumerate()
                .filter(|&(k, r)| {
                    r == "PRO"
                        && (wf.pred.shifts.contains_key(&(k, "N".to_string()))
                            || wf.pred.shifts.contains_key(&(k, "CA".to_string())))
                })
                .count();
            println!(
                "[{}] regime=idp-13c QC={} | {} CON dipeptide systems; {} residues with shifts ({n_hc} with confidence >= 0.8; {n_pro_with_shifts}/{n_pro_res} Pro assigned) | offsets: dCA={:+.3}, dCB={:+.3}, dC={:+.3}, dN={:+.3} ppm",
                wf.engine_label,
                wf.result.qc_verdict,
                wf.systems.len(),
                wf.conf.len(),
                wf.result.fitted_offsets.d_ca,
                wf.result.fitted_offsets.d_cb,
                wf.result.fitted_offsets.d_c,
                wf.result.fitted_offsets.d_n,
            );
            println!("wrote {} and {}", str_out.display(), tsv_out.display());
            Ok(())
        }
        RegimeKind::Paramagnetic => {
            let p_path = peaks_paths
                .first()
                .ok_or("assign --regime paramagnetic requires --peaks <peaks.tsv>")?;
            let peaks = read_peaks(p_path)?;
            let pri = load_priors(priors_path.as_deref())?;
            let score_cfg = ScoreConfig {
                assign_cost,
                ..ScoreConfig::default()
            };
            let engine_req = engine_req_opt.unwrap_or(EngineRequest::Auto);
            let (systems, result, eff_priors, para_rep, pcs_tensor_opt, engine_label) =
                execute_paramagnetic_workflow(
                    solver,
                    engine_req,
                    &peaks,
                    &sequence,
                    &seq_ids,
                    &pri,
                    &score_cfg,
                    metal_ligands_spec.as_deref(),
                    r_bleach_opt,
                    pdb_path.as_deref(),
                    pcs_path.as_deref(),
                    runs,
                    steps,
                    seed,
                )?;
            let (pred, conf) = to_reference(
                &result,
                &systems,
                &sequence,
                &seq_ids,
                0.0,
                Some(&eff_priors),
            );
            let mut hdr = format!(
                "# nmrassign solver: {engine_label}\n\
                 # regime: paramagnetic (clusters={}, blind_zone_residues={}, abstentions={})",
                para_rep.detected_clusters.len(),
                para_rep.blind_zone_residues.len(),
                para_rep.n_blind_zone_abstentions,
            );
            if let Some(ref t) = pcs_tensor_opt {
                hdr.push_str(&format!(
                    "\n# pcs_delta_chi: dchi_ax={:+.3}, dchi_rh={:+.3} (10^-32 m^3), q_factor={:.4}, euler_zyz_deg=({:.1},{:.1},{:.1})",
                    t.dchi_ax,
                    t.dchi_rh,
                    t.q_factor,
                    t.euler_zyz_rad[0].to_degrees(),
                    t.euler_zyz_rad[1].to_degrees(),
                    t.euler_zyz_rad[2].to_degrees(),
                ));
            }
            write_star_with_header(&str_out, &pred, Some(&conf), Some(&hdr))?;
            write_shifts_tsv(&tsv_out, &pred, &conf)?;
            let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
            println!(
                "[{}] regime=paramagnetic | {} clusters, {} blind-zone residues ({} abstained) | {} spin systems; {} residues assigned ({n_hc} with confidence >= 0.8)",
                engine_label,
                para_rep.detected_clusters.len(),
                para_rep.blind_zone_residues.len(),
                para_rep.n_blind_zone_abstentions,
                systems.len(),
                conf.len(),
            );
            println!("wrote {} and {}", str_out.display(), tsv_out.display());
            Ok(())
        }
        RegimeKind::Globular => {
            let p_path = peaks_paths
                .first()
                .ok_or("assign requires --peaks <peaks.tsv>")?;
            let peaks = read_peaks(p_path)?;
            let pri = load_priors(priors_path.as_deref())?;
            let score_cfg = ScoreConfig {
                assign_cost,
                ..ScoreConfig::default()
            };
            let engine_opt = match engine_req_opt {
                None | Some(EngineRequest::Auto) => None,
                Some(EngineRequest::Gpu) => Some(EngineKind::Gpu),
                Some(EngineRequest::Cpu) => Some(EngineKind::Cpu),
            };
            let wf = execute_assignment_workflow(
                solver,
                engine_opt,
                &peaks,
                &sequence,
                &pri,
                &score_cfg,
                runs,
                steps,
                STANDARD_CPU_CALIBRATED_RUNS,
                seed,
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
    }
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
    no_axis_rescue: bool,
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
    if opts.no_axis_rescue {
        recipe.set_axis_rescue(false);
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
    if opts.no_axis_rescue {
        recipe.set_axis_rescue(false);
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
    let mut hsqc_ref_path: Option<PathBuf> = None;
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
            "--hsqc" => {
                i += 1;
                hsqc_ref_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --hsqc")?,
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
            "--no-axis-rescue" => opts.no_axis_rescue = true,
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
            let mut spec = process_fid_3d_mut(&fid3, &exp, &mut recipe)?;
            if let Some(ref hp) = hsqc_ref_path {
                let hsqc_2d = match read_fid_auto(hp) {
                    Ok(RawFid::Dim2(f2)) => {
                        let mut r2 = build_recipe_for_fid_2d(&f2, &opts)?;
                        process_fid_2d_mut(&f2, &mut r2)?
                    }
                    _ => load_2d_auto(hp)?,
                };
                let (dh, dn, audit) = register_3d_hn_to_hsqc_adaptive_with_audit(
                    &mut spec,
                    &hsqc_2d,
                    recipe.enable_axis_rescue,
                );
                recipe.axis_rescue_audit = audit;
                if dh.abs() > 1e-6 || dn.abs() > 1e-6 {
                    eprintln!(
                        "[register] aligned 3D {exp} to HSQC: dH={dh:+.4} ppm, dN={dn:+.3} ppm"
                    );
                }
            }
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

type PickedInMemoryWithAudit = (
    Vec<Peak>,
    Vec<Peak>,
    String,
    Vec<Option<AxisRescueAuditRecord>>,
);

fn pick_in_memory_spectra_with_audit(
    hsqc_spec: Option<&Spectrum2D>,
    triples: &mut [(String, Spectrum3D)],
    method: &str,
    engine: EngineKind,
    snr: f32,
    prob: f32,
    enable_axis_rescue: bool,
) -> Result<PickedInMemoryWithAudit, String> {
    // Register each 3D spectrum to the HSQC via rigid (dH, dN) carrier alignment
    // (cross-correlation of sky-projected HN planes) + empirical `{sign, fold}` verification (`13.3:P1-1`).
    let mut audits: Vec<Option<AxisRescueAuditRecord>> = Vec::with_capacity(triples.len());
    if let Some(s2) = hsqc_spec {
        for (_exp, s3) in triples.iter_mut() {
            let (dh, dn, audit) =
                register_3d_hn_to_hsqc_adaptive_with_audit(s3, s2, enable_axis_rescue);
            if dh.abs() > 1e-6 || dn.abs() > 1e-6 {
                eprintln!("[register] aligned 3D {_exp} to HSQC: dH={dh:+.4} ppm, dN={dn:+.3} ppm");
            }
            audits.push(audit);
        }
    } else {
        audits.resize(triples.len(), None);
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
            Ok((r, all, "classical".to_string(), audits))
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
                        audits,
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
                    Ok((r, all, "neural (CPU)".to_string(), audits))
                }
            }
        }
        other => Err(format!(
            "unknown --method {other:?}; expected classical or neural"
        )),
    }
}

type PickedBundleWithAudit = (Vec<Peak>, Vec<Peak>, String, Vec<AxisRescueAuditRecord>);
type ProcessedFidBundleWithAudit = (
    Vec<Peak>,
    Vec<Peak>,
    String,
    String,
    Vec<AxisRescueAuditRecord>,
);

fn pick_spectra_bundle_with_audit(
    hsqc_path: &Path,
    triples: &[(String, PathBuf)],
    method: &str,
    engine: EngineKind,
    snr: f32,
    prob: f32,
    enable_axis_rescue: bool,
) -> Result<PickedBundleWithAudit, String> {
    let hsqc_spec = load_2d_auto(hsqc_path)?;
    let mut loaded_triples = Vec::with_capacity(triples.len());
    for (exp, path) in triples {
        loaded_triples.push((exp.clone(), load_3d_auto(path, exp)?));
    }
    let (r, all, lbl, raw_audits) = pick_in_memory_spectra_with_audit(
        Some(&hsqc_spec),
        &mut loaded_triples,
        method,
        engine,
        snr,
        prob,
        enable_axis_rescue,
    )?;
    Ok((r, all, lbl, raw_audits.into_iter().flatten().collect()))
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
) -> Result<ProcessedFidBundleWithAudit, String> {
    let (hsqc_spec, hsqc_recipe) = process_fid_2d_with_opts(fid_hsqc, fid_opts)?;
    let mut loaded_triples = Vec::with_capacity(fid_triples.len() + proc_triples.len());
    let mut triple_recipes: Vec<(String, ProcessingRecipe)> = Vec::new();

    for (exp, dir) in fid_triples {
        let (s3, r3) = process_fid_3d_with_opts(dir, exp, fid_opts)?;
        triple_recipes.push((exp.clone(), r3));
        loaded_triples.push((exp.clone(), s3));
    }
    for (exp, path) in proc_triples {
        loaded_triples.push((exp.clone(), load_3d_auto(path, exp)?));
    }

    let enable_axis_rescue = !fid_opts.no_axis_rescue
        && hsqc_recipe.enable_axis_rescue
        && triple_recipes.iter().all(|(_, r)| r.enable_axis_rescue);

    let (roots, all_peaks, label, raw_audits) = pick_in_memory_spectra_with_audit(
        Some(&hsqc_spec),
        &mut loaded_triples,
        method,
        engine,
        snr,
        prob,
        enable_axis_rescue,
    )?;

    for ((_, r3), maybe_audit) in triple_recipes.iter_mut().zip(raw_audits.iter()) {
        r3.axis_rescue_audit = maybe_audit.clone();
    }

    let hsqc_json = hsqc_recipe.to_json_pretty()?;
    let bundle_recipe_json = if triple_recipes.is_empty() {
        hsqc_json
    } else {
        let mut s = String::from("{\n  \"hsqc\": ");
        s.push_str(&indent_json_block(&hsqc_json, "  "));
        s.push_str(",\n  \"triples\": {\n");
        for (idx, (exp, r3)) in triple_recipes.iter().enumerate() {
            let rjson = r3.to_json_pretty()?;
            if idx > 0 {
                s.push_str(",\n");
            }
            s.push_str("    \"");
            s.push_str(&exp.replace('\\', "\\\\").replace('"', "\\\""));
            s.push_str("\": ");
            s.push_str(&indent_json_block(&rjson, "    "));
        }
        s.push_str("\n  }\n}");
        s
    };
    let collected_audits: Vec<AxisRescueAuditRecord> = raw_audits.into_iter().flatten().collect();
    Ok((
        roots,
        all_peaks,
        label,
        bundle_recipe_json,
        collected_audits,
    ))
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
            "--no-axis-rescue" => fid_opts.no_axis_rescue = true,
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
        let (r, all, lbl, _, _) = process_and_pick_fid_bundle(
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
                let mut rescue_ok = !fid_opts.no_axis_rescue && recipe.enable_axis_rescue;
                for (exp, dir) in &fid_triples {
                    let (s3, r3) = process_fid_3d_with_opts(dir, exp, &fid_opts)?;
                    rescue_ok &= r3.enable_axis_rescue;
                    loaded_triples.push((exp.clone(), s3));
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
                    let (r, all, lbl, _) = pick_in_memory_spectra_with_audit(
                        Some(&spec2),
                        &mut loaded_triples,
                        &method,
                        engine,
                        snr,
                        prob,
                        rescue_ok,
                    )?;
                    (r, all, lbl)
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
                let (r, all, lbl, _) = pick_in_memory_spectra_with_audit(
                    opt_hsqc.as_ref(),
                    &mut [(exp, spec3)],
                    &method,
                    engine,
                    snr,
                    prob,
                    !fid_opts.no_axis_rescue && recipe.enable_axis_rescue,
                )?;
                (r, all, lbl)
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
                            let (r, all, lbl, _, _) = process_and_pick_fid_bundle(
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
        let (r, all, lbl, _) = pick_spectra_bundle_with_audit(
            &hsqc_p,
            &triples,
            &method,
            engine,
            snr,
            prob,
            !fid_opts.no_axis_rescue,
        )?;
        (r, all, lbl)
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
                runs = Some(parse_bounded_runs(args.get(i).ok_or("missing --runs")?)?);
            }
            "--steps" => {
                i += 1;
                steps = Some(parse_bounded_steps(args.get(i).ok_or("missing --steps")?)?);
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
                    STANDARD_CPU_CALIBRATED_RUNS,
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

fn is_probable_peak_tsv(path: &Path) -> bool {
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

fn load_2d_spectrum_any(path: &Path) -> Result<Spectrum2D, String> {
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

fn load_3d_spectrum_any(path: &Path, label: &str) -> Result<Spectrum3D, String> {
    if path.is_dir() {
        return read_bruker_pdata_3d(path, label);
    }
    if path.file_name().and_then(|s| s.to_str()) == Some("3rrr") {
        if let Some(parent) = path.parent() {
            return read_bruker_pdata_3d(parent, label);
        }
    }
    if is_ucsf_file(path) {
        return read_ucsf_3d(path, label);
    }
    read_nmrpipe_3d(path, label)
}

fn load_aromatic_input_peaks(
    path: &Path,
    default_exp: &str,
    c13_roots: Option<&[Peak]>,
) -> Result<Vec<Peak>, String> {
    use nmr_core::aromatic::{
        pick_aromatic_2d, pick_aromatic_hcch_3d, AromaticExperiment, AromaticPickerConfig,
        AROMATIC_EXPERIMENTS,
    };

    if is_probable_peak_tsv(path) {
        let mut pks = read_peaks(path)?;
        for p in &mut pks {
            let trimmed = p.exp.trim();
            if trimmed.is_empty()
                || (!AROMATIC_EXPERIMENTS.contains(&trimmed)
                    && trimmed != "NOESY13C"
                    && trimmed != "AROM_HB_NOE")
            {
                p.exp = default_exp.to_string();
            }
        }
        return Ok(pks);
    }

    let picker_cfg = AromaticPickerConfig::default();
    if default_exp == "AROM_HCCH_TOCSY" {
        let spec3d = load_3d_spectrum_any(path, "AROM_HCCH_TOCSY")?;
        return Ok(pick_aromatic_hcch_3d(&spec3d, c13_roots, &picker_cfg));
    }
    if default_exp == "NOESY13C" || default_exp == "AROM_HB_NOE" {
        let spec3d = load_3d_spectrum_any(path, default_exp)?;
        let mut raw = pick_aromatic_hcch_3d(&spec3d, c13_roots, &picker_cfg);
        for p in &mut raw {
            p.exp = default_exp.to_string();
        }
        return Ok(raw);
    }
    let exp = AromaticExperiment::parse_exp(default_exp)
        .ok_or_else(|| format!("unsupported aromatic experiment tag {default_exp:?}"))?;
    let spec2d = load_2d_spectrum_any(path)?;
    Ok(pick_aromatic_2d(&spec2d, exp, &picker_cfg))
}

fn run_sidechain(args: &[String]) -> Result<(), String> {
    use nmr_core::aromatic::{
        assign_aromatic_rings, default_his_tautomer_map,
        merge_aromatic_assignments_preserving_existing, simulate_aromatic_peaks,
        synthesize_aromatic_reference, AromaticSimConfig, AromaticSolverConfig,
        AROMATIC_EXPERIMENTS,
    };
    use nmr_core::sidechain::assign_sidechains;
    use nmr_core::stereospec::{
        assign_stereospecific_dataset, format_nmrstar_with_ambiguity, StereospecConfig,
        StereospecDatasetInput,
    };

    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut runs = 16usize;
    let mut steps = 2500usize;
    let mut seed = 2026u64;

    // Phase 10 (`T10.2`) optional aromatic ring assignment flags
    let mut aromatic_flag = false;
    let mut arom_c13hsqc_path: Option<PathBuf> = None;
    let mut hbcbgcdhd_path: Option<PathBuf> = None;
    let mut hbcbgcdce_path: Option<PathBuf> = None;
    let mut arom_tocsy_path: Option<PathBuf> = None;
    let mut arom_noesy_path: Option<PathBuf> = None;
    let mut his_hmqc_path: Option<PathBuf> = None;
    let mut trp_hsqc_path: Option<PathBuf> = None;

    // Phase 10 (`T10.3`) optional stereospecific prochiral assignment flags
    let mut stereospec_flag = false;
    let mut j_couplings_path: Option<PathBuf> = None;
    let mut intra_noes_path: Option<PathBuf> = None;
    let mut pdb_path: Option<PathBuf> = None;
    let mut stereo_threshold = 0.90f64;
    let mut simulate_flag = false;
    let mut sample_ph_opt: Option<f64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign sidechain --star <ref.str> [--peaks <peaks.tsv>] [--simulate] \\\n  \
                     [--aromatic] [--ph <f64>] [--arom-c13hsqc <path>] [--hbcbgcdhd <path>] [--hbcbgcdce <path>] \\\n  \
                     [--aromatic-tocsy <path>] [--aromatic-noesy <path>] [--his-hmqc <path>] [--trp-hsqc <path>] \\\n  \
                     [--stereospec] [--j-couplings <j.tsv>] [--intra-noes <noes.tsv>] [--pdb <model.pdb>] \\\n  \
                     [--stereo-threshold <0.90>] [--out <all_atom.str>] [--runs 16] [--steps 2500] [--seed 2026]"
                );
                return Ok(());
            }
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
                runs = parse_bounded_runs(args.get(i).ok_or("missing --runs")?)?;
            }
            "--steps" => {
                i += 1;
                steps = parse_bounded_steps(args.get(i).ok_or("missing --steps")?)?;
            }
            "--seed" => {
                i += 1;
                seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--aromatic" => {
                aromatic_flag = true;
            }
            "--arom-c13hsqc" | "--aromatic-hsqc" | "--aromatic-c13hsqc" => {
                i += 1;
                arom_c13hsqc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --arom-c13hsqc <path>")?,
                ));
            }
            "--hbcbgcdhd" => {
                i += 1;
                hbcbgcdhd_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --hbcbgcdhd <path>")?,
                ));
            }
            "--hbcbgcdce" | "--hbcbgcdcehe" => {
                i += 1;
                hbcbgcdce_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --hbcbgcdce <path>")?,
                ));
            }
            "--aromatic-tocsy" | "--arom-hcch-tocsy" => {
                i += 1;
                arom_tocsy_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --aromatic-tocsy <path>")?,
                ));
            }
            "--aromatic-noesy" | "--arom-noesy" => {
                i += 1;
                arom_noesy_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --aromatic-noesy <path>")?,
                ));
            }
            "--his-hmqc" => {
                i += 1;
                his_hmqc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --his-hmqc <path>")?,
                ));
            }
            "--trp-hsqc" => {
                i += 1;
                trp_hsqc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --trp-hsqc <path>")?,
                ));
            }
            "--stereospec" => {
                stereospec_flag = true;
            }
            "--j-couplings" => {
                i += 1;
                j_couplings_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --j-couplings <j.tsv>")?,
                ));
            }
            "--intra-noes" => {
                i += 1;
                intra_noes_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --intra-noes <noes.tsv>")?,
                ));
            }
            "--pdb" => {
                i += 1;
                pdb_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --pdb <model.pdb>")?,
                ));
            }
            "--stereo-threshold" => {
                i += 1;
                stereo_threshold = args
                    .get(i)
                    .ok_or("missing --stereo-threshold")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--simulate" | "--benchmark" => {
                simulate_flag = true;
            }
            "--ph" => {
                i += 1;
                let ph: f64 = args
                    .get(i)
                    .ok_or("missing --ph <0.0..=14.0>")?
                    .parse()
                    .map_err(|e| format!("invalid --ph: {e}"))?;
                if !ph.is_finite() || !(0.0..=14.0).contains(&ph) {
                    return Err(format!(
                        "invalid --ph {ph}: expected finite pH in 0.0..=14.0"
                    ));
                }
                sample_ph_opt = Some(ph);
            }
            other => return Err(format!("unknown sidechain flag {other:?}")),
        }
        i += 1;
    }

    if !stereo_threshold.is_finite() || stereo_threshold <= 0.5 || stereo_threshold > 1.0 {
        return Err(format!(
            "invalid --stereo-threshold {stereo_threshold}: must be finite and in (0.5, 1.0]"
        ));
    }

    let explicit_arom_inputs = arom_c13hsqc_path.is_some()
        || hbcbgcdhd_path.is_some()
        || hbcbgcdce_path.is_some()
        || arom_tocsy_path.is_some()
        || arom_noesy_path.is_some()
        || his_hmqc_path.is_some()
        || trp_hsqc_path.is_some();
    let do_aromatic_opt = aromatic_flag || explicit_arom_inputs;
    let do_stereospec_opt = stereospec_flag
        || j_couplings_path.is_some()
        || intra_noes_path.is_some()
        || pdb_path.is_some();

    let star_p = star_path.ok_or("sidechain requires --star <entry.str>")?;
    if !simulate_flag {
        if peaks_path.is_none() && aromatic_flag && !explicit_arom_inputs {
            return Err(
                "sidechain --aromatic without --peaks requires at least one aromatic spectrum or peak input (--arom-c13hsqc, --hbcbgcdhd, --hbcbgcdce, --aromatic-tocsy, --aromatic-noesy, --his-hmqc, or --trp-hsqc); pass --simulate only for synthetic benchmarking"
                    .to_string(),
            );
        }
        if peaks_path.is_none() && !explicit_arom_inputs && !do_stereospec_opt {
            return Err(
                "sidechain requires --peaks, explicit aromatic spectrum/peak inputs, or --stereospec inputs (pass --simulate only for synthetic benchmarking)"
                    .to_string(),
            );
        }
    }

    let raw_ref = read_star_all_atoms(&star_p, None)?;
    let sc_priors = SidechainPriors::empirical_bmrb();
    let bb_priors = Priors::default();

    let has_sidechains = raw_ref
        .shifts
        .keys()
        .any(|(_, a)| !["H", "N", "CA", "CB", "C"].contains(&a.as_str()));
    let synthetic_benchmark_mode = simulate_flag;

    let mut truth_all = if synthetic_benchmark_mode && !has_sidechains {
        synthesize_all_atom_reference(&raw_ref, &sc_priors, seed)
    } else {
        raw_ref.clone()
    };
    if synthetic_benchmark_mode && do_aromatic_opt {
        let his_map = default_his_tautomer_map(&truth_all.sequence, seed);
        truth_all = synthesize_aromatic_reference(&truth_all, &sc_priors, &his_map, seed);
    }

    let (bb_peaks, mut sc_peaks, mut bundled_arom_peaks, bb_truth_opt) =
        if let Some(pk_p) = &peaks_path {
            let loaded = read_peaks(pk_p)?;
            let mut bb = Vec::new();
            let mut sc = Vec::new();
            let mut ar = Vec::new();
            for p in loaded {
                if EXPERIMENT_NAMES.contains(&p.exp.as_str()) {
                    bb.push(p);
                } else if AROMATIC_EXPERIMENTS.contains(&p.exp.as_str()) || p.exp == "AROM_HB_NOE" {
                    ar.push(p);
                } else {
                    sc.push(p);
                }
            }
            (bb, sc, ar, None)
        } else if synthetic_benchmark_mode {
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
                    include_aromatics: !do_aromatic_opt,
                    p_missing_peak: 0.04,
                    spurious_rate: 0.02,
                    seed,
                    ..SidechainSimConfig::default()
                },
            );
            (bb, sc, Vec::new(), Some(bb_truth))
        } else {
            (Vec::new(), Vec::new(), Vec::new(), None)
        };

    // Load any explicit aromatic spectra or peak TSVs (`--hbcbgcdce`, `--aromatic-tocsy`, `--aromatic-noesy`, ...)
    let mut c13_arom_roots: Vec<Peak> = bundled_arom_peaks
        .iter()
        .filter(|p| p.exp == "AROM_C13HSQC")
        .cloned()
        .collect();
    if let Some(p) = &arom_c13hsqc_path {
        let pks = load_aromatic_input_peaks(p, "AROM_C13HSQC", None)?;
        c13_arom_roots.extend(pks.iter().cloned());
        bundled_arom_peaks.extend(pks);
    }
    if let Some(p) = &hbcbgcdhd_path {
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "HBCBCGCDHD", None)?);
    }
    if let Some(p) = &hbcbgcdce_path {
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "HBCBCGCDCEHE", None)?);
    }
    if let Some(p) = &arom_tocsy_path {
        let roots_opt = (!c13_arom_roots.is_empty()).then_some(c13_arom_roots.as_slice());
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "AROM_HCCH_TOCSY", roots_opt)?);
    }
    if let Some(p) = &arom_noesy_path {
        let roots_opt = (!c13_arom_roots.is_empty()).then_some(c13_arom_roots.as_slice());
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "AROM_HB_NOE", roots_opt)?);
    }
    if let Some(p) = &his_hmqc_path {
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "HIS_LR_N15HMQC", None)?);
    }
    if let Some(p) = &trp_hsqc_path {
        bundled_arom_peaks.extend(load_aromatic_input_peaks(p, "TRP_IND_N15HSQC", None)?);
    }
    if synthetic_benchmark_mode && aromatic_flag && bundled_arom_peaks.is_empty() {
        let his_map = default_his_tautomer_map(&truth_all.sequence, seed);
        let (sim_ar, sim_trp) = simulate_aromatic_peaks(
            &truth_all,
            &his_map,
            &AromaticSimConfig {
                seed,
                ..AromaticSimConfig::default()
            },
        );
        bundled_arom_peaks.extend(sim_ar);
        bundled_arom_peaks.extend(sim_trp);
    }
    if !bundled_arom_peaks.is_empty() {
        // Ensure aromatic NOESY peaks aren't duplicated in aliphatic sidechain peaks
        sc_peaks.retain(|p| !AROMATIC_EXPERIMENTS.contains(&p.exp.as_str()));
    }

    let t0 = Instant::now();
    let (mut working_ref, bb_conf, bb_eval_opt) =
        if peaks_path.is_none() && !synthetic_benchmark_mode {
            let mut conf = BTreeMap::new();
            for k in 0..raw_ref.sequence.len() {
                conf.insert(k, 1.0);
            }
            (raw_ref.clone(), conf, None)
        } else if !bb_peaks.is_empty() {
            let systems = build_spin_systems(&bb_peaks, &Tolerances::default(), 3, false)?;
            let joint = solve_joint_backbone_sidechain(
                &systems,
                &raw_ref.sequence,
                &raw_ref.seq_ids,
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
            let (_, conf) = to_reference(
                &joint.backbone_result,
                &systems,
                &raw_ref.sequence,
                &raw_ref.seq_ids,
                0.0,
                Some(&bb_priors),
            );
            let eval_pair = bb_truth_opt.map(|tr| (tr, joint.backbone_result));
            (joint.sidechain_result.reference, conf, eval_pair)
        } else {
            let mut conf = BTreeMap::new();
            for k in 0..raw_ref.sequence.len() {
                conf.insert(k, 1.0);
            }
            let assigned_ref = if !sc_peaks.is_empty() {
                assign_sidechains(
                    &raw_ref,
                    None,
                    &sc_peaks,
                    &sc_priors,
                    &SidechainSolverConfig::default(),
                )
                .reference
            } else {
                raw_ref.clone()
            };
            (assigned_ref, conf, None)
        };

    let mut ambiguity_codes: BTreeMap<(usize, String), u8> = if !synthetic_benchmark_mode {
        let mut base_amb = raw_ref.ambiguity_codes.clone();
        base_amb.extend(working_ref.ambiguity_codes.clone());
        base_amb
    } else {
        working_ref.ambiguity_codes.clone()
    };
    let mut stereo_posteriors: BTreeMap<(usize, String), f64> = BTreeMap::new();
    let run_arom_stage = !bundled_arom_peaks.is_empty();

    if run_arom_stage {
        // Clear heuristic ring-atom placeholders synthesized or placed by `assign_sidechains`,
        // while preserving 100% of pre-existing ring shifts & ambiguity codes from the user's input STAR (`P1-12`)!
        const RING_ATOMS: &[&str] = &[
            "CD1", "CD2", "HD1", "HD2", "CE1", "CE2", "HE1", "HE2", "CZ", "HZ", "CE3", "HE3",
            "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2", "ND1", "NE1", "NE2",
        ];
        let is_arom_ring_key = |seq_len_ref: &Reference, k: usize, atom: &str| -> bool {
            matches!(
                seq_len_ref.sequence.get(k).map(String::as_str),
                Some("PHE" | "TYR" | "TRP" | "HIS")
            ) && RING_ATOMS.contains(&atom)
        };

        let mut backbone_for_arom = working_ref.clone();
        backbone_for_arom
            .shifts
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&working_ref, k, atom));
        backbone_for_arom
            .ambiguity_codes
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&working_ref, k, atom));

        let mut pre_arom_user_ref = backbone_for_arom.clone();
        if !synthetic_benchmark_mode {
            for (&(k, ref atom), &v) in &raw_ref.shifts {
                if is_arom_ring_key(&raw_ref, k, atom) {
                    pre_arom_user_ref.shifts.insert((k, atom.clone()), v);
                }
            }
            for (&(k, ref atom), &code) in &raw_ref.ambiguity_codes {
                if is_arom_ring_key(&raw_ref, k, atom) {
                    pre_arom_user_ref
                        .ambiguity_codes
                        .insert((k, atom.clone()), code);
                }
            }
        }

        let arom_res = assign_aromatic_rings(
            &backbone_for_arom,
            Some(&bb_conf),
            &bundled_arom_peaks,
            &sc_priors,
            &AromaticSolverConfig {
                sample_ph: sample_ph_opt,
                ..AromaticSolverConfig::default()
            },
        );
        let n_arom_total = arom_res.rings.len();
        let n_arom_assigned = arom_res
            .rings
            .iter()
            .filter(|r| r.cluster_idx.is_some())
            .count();
        let n_his_taut = arom_res
            .rings
            .iter()
            .filter(|r| r.his_tautomer.is_some())
            .count();
        working_ref = merge_aromatic_assignments_preserving_existing(
            &pre_arom_user_ref,
            &arom_res.reference,
            &arom_res.atom_confidence,
        );
        for (k, atom) in arom_res.atom_confidence.keys() {
            let res_name = working_ref
                .sequence
                .get(*k)
                .map(String::as_str)
                .unwrap_or("");
            let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
            working_ref.ambiguity_codes.insert((*k, atom.clone()), code);
        }
        ambiguity_codes = working_ref.ambiguity_codes.clone();
        println!(
            "Aromatic rings: assigned {n_arom_assigned}/{n_arom_total} aromatic residues ({n_his_taut} His tautomers classified)"
        );
    }

    if do_stereospec_opt {
        let stereo_out = assign_stereospecific_dataset(&StereospecDatasetInput {
            reference: &working_ref,
            j_couplings_tsv: j_couplings_path.as_deref(),
            intra_noes_tsv: intra_noes_path.as_deref(),
            pdb_path: pdb_path.as_deref(),
            config: StereospecConfig {
                commit_posterior_threshold: stereo_threshold,
                ..StereospecConfig::default()
            },
        })?;
        let total_pairs = stereo_out.pair_results.len();
        println!(
            "Stereospecific prochiral: committed {}/{total_pairs} pairs (Ambiguity_code=1, posterior>={stereo_threshold:.2}), {} ambiguous (Ambiguity_code=2)",
            stereo_out.committed_pairs, stereo_out.ambiguous_pairs
        );
        working_ref = stereo_out.reference;
        ambiguity_codes = stereo_out.ambiguity_codes;
        stereo_posteriors = stereo_out.atom_posteriors;
    }

    let dt = t0.elapsed().as_secs_f64();

    if let Some((bb_truth, bb_res)) = &bb_eval_opt {
        let bbm = score(bb_res, bb_truth, 0.8);
        println!(
            "Backbone: accuracy={:.1}% ({}/{}), recall={:.1}%, hc_prec={:.1}%",
            bbm.accuracy() * 100.0,
            bbm.correct,
            bbm.assigned,
            bbm.recall() * 100.0,
            bbm.hc_precision() * 100.0
        );
    }
    if synthetic_benchmark_mode || (peaks_path.is_some() && has_sidechains && !bb_peaks.is_empty())
    {
        let scm = evaluate_sidechain_shifts(&working_ref, &truth_all, 0.40, 0.06);
        println!("{}", scm.summary());
    }
    println!(
        "Total assigned chemical shifts (backbone + sidechain): {} in {:.2}s",
        working_ref.shifts.len(),
        dt
    );

    if let Some(out_p) = out_path {
        if run_arom_stage || do_stereospec_opt {
            let star_txt = format_nmrstar_with_ambiguity(
                &working_ref,
                Some(&bb_conf),
                &ambiguity_codes,
                (!stereo_posteriors.is_empty()).then_some(&stereo_posteriors),
                None,
            );
            fs::write(&out_p, star_txt)
                .map_err(|e| format!("failed to write {}: {e}", out_p.display()))?;
        } else {
            write_star(&out_p, &working_ref, Some(&bb_conf))?;
        }
        println!("Wrote all-atom NMR-STAR shifts to {}", out_p.display());
    }
    Ok(())
}

fn cmd_sidechain(args: &[String]) -> Result<(), String> {
    run_sidechain(args)
}

fn load_shifts_str_or_nef(path: &Path) -> Result<Reference, String> {
    let is_nef = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nef"));
    if is_nef {
        read_nef_file(path)?.to_reference()
    } else {
        read_star_all_atoms(path, None)
    }
}

fn load_rdc_media_auto(path: &Path) -> Result<Vec<RdcMediumSet>, String> {
    let is_nef = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nef"));
    let media = if is_nef {
        read_nef_file(path)?.rdc_media()?
    } else {
        read_rdc_tsv(path)?
    };
    if media.is_empty() {
        return Err(format!(
            "RDC input '{}' contains zero alignment media / restraints",
            path.display()
        ));
    }
    Ok(media)
}

fn run_gpu_rdc_structure_workflow(
    gpu: &GpuEngine,
    shifts: &Reference,
    noesy_peaks: &[Peak],
    dihedral_restraints: Vec<DihedralRestraint>,
    rdc_media: &[RdcMediumSet],
    w_rdc: f64,
    cfg: &StructureCalcConfig,
) -> Result<(StructureEnsembleResult, RdcValidationSummary), String> {
    let mut init_cfg = cfg.clone();
    init_cfg.n_select = cfg.n_select.max(8).min(cfg.n_replicas.max(1));
    let (base_ens, _) = calculate_structure_with_rdcs_and_dihedrals(
        shifts,
        noesy_peaks,
        dihedral_restraints,
        &[],
        0.0,
        &init_cfg,
    )?;
    let (mut refined, rdc_summary) = refine_ensemble_with_rdcs_gpu(
        gpu,
        &base_ens.conformers,
        &shifts.sequence,
        &shifts.seq_ids,
        &base_ens.final_adrs,
        &base_ens.dihedral_restraints,
        rdc_media,
        w_rdc,
        cfg.lbfgs_iters.max(110),
    )?;
    refined.truncate(cfg.n_select.max(1).min(refined.len()));
    Ok((
        StructureEnsembleResult {
            conformers: refined,
            dihedral_restraints: base_ens.dihedral_restraints,
            initial_unambiguous_adrs: base_ens.initial_unambiguous_adrs,
            final_adrs: base_ens.final_adrs,
        },
        rdc_summary,
    ))
}

fn cmd_structure(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut noesy_path: Option<PathBuf> = None;
    let mut rdc_path: Option<PathBuf> = None;
    let mut use_torsion_db = false;
    let mut include_chi1 = false;
    let mut rdc_weight_opt: Option<f64> = None;
    let mut rdc_free_frac_opt: Option<f64> = None;
    let mut engine_req_opt: Option<EngineRequest> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut out_pdb_path: Option<PathBuf> = None;
    let mut n_conformers = 10usize;
    let mut seed = 42u64;
    let mut dihedrals_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign structure --star <shifts.str|shifts.nef> [--noesy-peaks <noesy.tsv>] [--rdc <rdcs.tsv|rdcs.nef>] [--torsion-db] [--include-chi1] [--rdc-weight 0.75] [--rdc-free-frac 0.20] [--engine auto|gpu|cpu] [--pdb-ref <ref.pdb>] [--out-pdb <ensemble.pdb>] [--conformers 10] [--seed 42] [--dihedrals-only]"
                );
                return Ok(());
            }
            "--star" | "--shifts" => {
                i += 1;
                star_path = args.get(i).map(PathBuf::from);
            }
            "--noesy-peaks" => {
                i += 1;
                noesy_path = args.get(i).map(PathBuf::from);
            }
            "--rdc" => {
                i += 1;
                rdc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --rdc <rdcs.tsv|rdcs.nef>")?,
                ));
            }
            "--torsion-db" => {
                use_torsion_db = true;
            }
            "--include-chi1" => {
                include_chi1 = true;
            }
            "--rdc-weight" => {
                i += 1;
                let w: f64 = args
                    .get(i)
                    .ok_or("missing --rdc-weight")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                if !w.is_finite() || !(1e-6..=1000.0).contains(&w) {
                    return Err(format!(
                        "invalid --rdc-weight {w}: expected finite positive weight in 1e-6..=1000.0"
                    ));
                }
                rdc_weight_opt = Some(w);
            }
            "--rdc-free-frac" => {
                i += 1;
                let frac: f64 = args
                    .get(i)
                    .ok_or("missing --rdc-free-frac")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                if !frac.is_finite() || !(0.0..=0.50).contains(&frac) {
                    return Err(format!(
                        "invalid --rdc-free-frac {frac}: expected finite fraction in 0.0..=0.50"
                    ));
                }
                rdc_free_frac_opt = Some(frac);
            }
            "--engine" => {
                i += 1;
                engine_req_opt = Some(parse_engine_request(
                    args.get(i).ok_or("missing --engine")?,
                )?);
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
                if !(1..=512).contains(&n_conformers) {
                    return Err(format!(
                        "invalid --conformers {n_conformers}: out of allowed bounds (1..=512)"
                    ));
                }
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

    if rdc_path.is_none() {
        if rdc_weight_opt.is_some() {
            return Err(
                "--rdc-weight requires --rdc <rdcs.tsv|rdcs.nef> in `nmrassign structure`"
                    .to_string(),
            );
        }
        if rdc_free_frac_opt.is_some() {
            return Err(
                "--rdc-free-frac requires --rdc <rdcs.tsv|rdcs.nef> in `nmrassign structure`"
                    .to_string(),
            );
        }
        if engine_req_opt == Some(EngineRequest::Gpu) {
            return Err(
                "--engine gpu in `nmrassign structure` requires --rdc <rdcs.tsv|rdcs.nef> (NOE-only torsion folding is CPU-only; use --engine cpu or --engine auto)"
                    .to_string(),
            );
        }
    }
    if include_chi1 && !use_torsion_db {
        return Err("--include-chi1 requires --torsion-db in `nmrassign structure`".to_string());
    }
    if dihedrals_only && (rdc_path.is_some() || engine_req_opt.is_some()) {
        return Err(
            "--dihedrals-only cannot be combined with --rdc or --engine in `nmrassign structure`"
                .to_string(),
        );
    }

    let star_p = star_path.ok_or("structure requires --star <shifts.str>")?;
    let shifts = load_shifts_str_or_nef(&star_p)?;
    let dih_preds = if use_torsion_db {
        predict_torsions(&shifts)
            .to_dihedral_restraints_with_options(TorsionExportOptions { include_chi1 })
    } else {
        predict_dihedrals(&shifts)
    };
    let n_confident = dih_preds.iter().filter(|d| d.confidence >= 0.5).count();
    let dih_src = if use_torsion_db {
        "empirical 3/9-mer TorsionDb"
    } else {
        "CSI/RCI"
    };
    println!(
        "Predicted backbone (phi, psi) dihedrals ({dih_src}) for {} residues ({} confident):",
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
        if np
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("nef"))
        {
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
    let (ens, rdc_val_opt, engine_label): (
        StructureEnsembleResult,
        Option<RdcValidationSummary>,
        String,
    ) = if let Some(ref rdc_p) = rdc_path {
        let mut rdc_media = load_rdc_media_auto(rdc_p)?;
        if !shifts.sequence.is_empty() {
            let l = shifts.sequence.len();
            let probe_chain = build_backbone_from_torsions(
                &shifts.sequence,
                &shifts.seq_ids,
                &vec![-60.0; l],
                &vec![-45.0; l],
                &vec![180.0; l],
            );
            let dropped_rdc = count_unresolved_rdc_restraints(&probe_chain, &rdc_media);
            if dropped_rdc > 0 {
                eprintln!(
                    "warning: dropped {dropped_rdc} RDC restraint(s) with unresolved residue or atom coordinates"
                );
            }
        }
        if let Some(frac) = rdc_free_frac_opt {
            partition_rdc_free_set(&mut rdc_media, frac, seed);
        } else if !rdc_media
            .iter()
            .any(|m| m.restraints.iter().any(|r| r.is_free))
        {
            partition_rdc_free_set(&mut rdc_media, 0.20, seed);
        }
        let w_rdc = rdc_weight_opt.unwrap_or(0.75);
        let engine_req = engine_req_opt.unwrap_or(EngineRequest::Auto);
        match engine_req {
            EngineRequest::Cpu => {
                let (e, rv) = calculate_structure_with_rdcs_and_dihedrals(
                    &shifts,
                    &noesy_peaks,
                    dih_preds.clone(),
                    &rdc_media,
                    w_rdc,
                    &cfg,
                )?;
                (e, rv, "CPU (f64)".to_string())
            }
            EngineRequest::Gpu => {
                let gpu = GpuEngine::new_headless()?;
                let lbl = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
                let (e, rv) = run_gpu_rdc_structure_workflow(
                    &gpu,
                    &shifts,
                    &noesy_peaks,
                    dih_preds.clone(),
                    &rdc_media,
                    w_rdc,
                    &cfg,
                )?;
                (e, Some(rv), lbl)
            }
            EngineRequest::Auto => match GpuEngine::new_headless() {
                Ok(gpu) => {
                    let lbl = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
                    match run_gpu_rdc_structure_workflow(
                        &gpu,
                        &shifts,
                        &noesy_peaks,
                        dih_preds.clone(),
                        &rdc_media,
                        w_rdc,
                        &cfg,
                    ) {
                        Ok((e, rv)) => (e, Some(rv), lbl),
                        Err(err) => {
                            eprintln!(
                                "Warning: WebGPU RDC structure refinement failed ({err}); falling back to CPU (f64) RDC refinement."
                            );
                            let (e, rv) = calculate_structure_with_rdcs_and_dihedrals(
                                &shifts,
                                &noesy_peaks,
                                dih_preds.clone(),
                                &rdc_media,
                                w_rdc,
                                &cfg,
                            )?;
                            (e, rv, "CPU (f64, auto-fallback)".to_string())
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "Warning: WebGPU unavailable for RDC structure refinement ({err}); falling back to CPU (f64) RDC refinement."
                    );
                    let (e, rv) = calculate_structure_with_rdcs_and_dihedrals(
                        &shifts,
                        &noesy_peaks,
                        dih_preds.clone(),
                        &rdc_media,
                        w_rdc,
                        &cfg,
                    )?;
                    (e, rv, "CPU (f64, auto-fallback)".to_string())
                }
            },
        }
    } else if use_torsion_db {
        let (e, _) = calculate_structure_with_rdcs_and_dihedrals(
            &shifts,
            &noesy_peaks,
            dih_preds.clone(),
            &[],
            0.0,
            &cfg,
        )?;
        (e, None, "CPU (f64)".to_string())
    } else {
        let e = calculate_structure_iterative(&shifts, &noesy_peaks, &cfg);
        (e, None, "CPU (f64)".to_string())
    };
    if noesy_path.is_some() && noesy_peaks.len() > ens.final_adrs.len() {
        let dropped_noe = noesy_peaks.len() - ens.final_adrs.len();
        eprintln!(
            "warning: dropped {dropped_noe} NOESY crosspeak(s) with no matching chemical-shift proton candidates"
        );
    }
    let dt = t0.elapsed().as_secs_f64();
    let adr_sum = summarize_adrs(&ens.final_adrs);
    let val = validate_ensemble(
        &ens.conformers,
        &ens.final_adrs,
        &ens.dihedral_restraints,
        pdb_ref.as_ref(),
    );

    println!(
        "[{}] Folded {}-conformer torsion-angle ensemble in {:.2}s ({} NOESY peaks -> {} ADRs, {} unambiguous, {} long-range):",
        engine_label,
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
    if let Some(ref rv) = rdc_val_opt {
        let q_free_str = if rv.overall_q_free.is_finite() {
            format!("{:.4}", rv.overall_q_free)
        } else {
            "n/a (0 free RDCs)".to_string()
        };
        println!(
            "  RDC validation ({} media, {} couplings): Cornilescu Q_work={:.4}, Q_free={}",
            rv.media.len(),
            rv.points.len(),
            rv.overall_q_work,
            q_free_str
        );
    }
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
        if let Some(parent) = out_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
            }
        }
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

fn build_propensity_config(temp_k: f64, ph_opt: Option<f64>) -> Result<PropensityConfig, String> {
    if !temp_k.is_finite() || !(250.0..=360.0).contains(&temp_k) {
        return Err(format!(
            "invalid --temp-k {temp_k}: expected finite temperature in 250..=360 K"
        ));
    }
    let mut cfg = PropensityConfig::default().with_temperature_kelvin(temp_k);
    if let Some(ph) = ph_opt {
        if !ph.is_finite() || !(1.0..=12.0).contains(&ph) {
            return Err(format!(
                "invalid --ph {ph}: expected finite pH in 1.0..=12.0"
            ));
        }
        cfg = cfg.with_sample_ph(ph);
    }
    Ok(cfg)
}

fn cmd_propensity(args: &[String]) -> Result<(), String> {
    let mut shifts_path: Option<PathBuf> = None;
    let mut temp_k = 298.15f64;
    let mut ph_opt: Option<f64> = None;
    let mut format = "tsv".to_string();
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign propensity --shifts <shifts.str|shifts.nef> [--temp-k 298.15] [--ph <f64>] [--format tsv|json] [-o <out_path>]"
                );
                return Ok(());
            }
            "--shifts" | "--star" => {
                i += 1;
                shifts_path = args.get(i).map(PathBuf::from);
            }
            "--temp-k" | "--temperature-k" => {
                i += 1;
                temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--ph" => {
                i += 1;
                ph_opt = Some(
                    args.get(i)
                        .ok_or("missing --ph")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--format" => {
                i += 1;
                format = args
                    .get(i)
                    .ok_or("missing --format")?
                    .trim()
                    .to_ascii_lowercase();
            }
            "-o" | "--out" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            other => return Err(format!("unknown propensity flag {other:?}")),
        }
        i += 1;
    }

    let shifts_p = shifts_path.ok_or("propensity requires --shifts <shifts.str|shifts.nef>")?;
    let prop_cfg = build_propensity_config(temp_k, ph_opt)?;
    let is_nef = shifts_p
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nef"));
    let ref_data = if is_nef {
        read_nef_file(&shifts_p)?.to_reference()?
    } else {
        read_star_all_atoms(&shifts_p, None)?
    };
    let report = compute_idp_propensity(&ref_data, &prop_cfg);

    let rendered = match format.as_str() {
        "tsv" => format_propensity_tsv(&report),
        "json" => format_propensity_json(&report)?,
        other => return Err(format!("unknown --format {other:?}; expected tsv or json")),
    };

    if let Some(out_p) = out_path {
        if let Some(parent) = out_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
            }
        }
        let body = if rendered.ends_with('\n') {
            rendered
        } else {
            format!("{rendered}\n")
        };
        fs::write(&out_p, body).map_err(|e| format!("failed to write {}: {e}", out_p.display()))?;
    } else if rendered.ends_with('\n') {
        print!("{rendered}");
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn parse_comma_ids_non_empty(raw: &str, flag: &str) -> Result<Vec<String>, String> {
    let vals: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if vals.is_empty() {
        return Err(format!("invalid empty {flag} list"));
    }
    Ok(vals)
}

fn cmd_torsions(args: &[String]) -> Result<(), String> {
    let mut shifts_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut include_chi1 = false;
    let mut exclude_pdb: Vec<String> = Vec::new();
    let mut exclude_bmrb: Vec<String> = Vec::new();
    let mut exclude_topology: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign torsions --shifts <assigned.str|assigned.nef> -o <out.tsv|out.nef|out.json> [--include-chi1] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [--exclude-topology <id>]"
                );
                return Ok(());
            }
            "--include-chi1" => {
                include_chi1 = true;
            }
            "--shifts" | "--star" => {
                i += 1;
                shifts_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing --shifts <assigned.str|assigned.nef>")?,
                ));
            }
            "-o" | "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing -o <out.tsv|out.nef|out.json>")?,
                ));
            }
            "--exclude-pdb" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --exclude-pdb <ids>")?;
                exclude_pdb.extend(parse_comma_ids_non_empty(raw, "--exclude-pdb")?);
            }
            "--exclude-bmrb" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --exclude-bmrb <ids>")?;
                for tok in parse_comma_ids_non_empty(raw, "--exclude-bmrb")? {
                    let norm = tok
                        .to_ascii_lowercase()
                        .trim_start_matches("bmr")
                        .to_string();
                    if norm.is_empty() {
                        return Err(format!("invalid --exclude-bmrb token {tok:?}"));
                    }
                    exclude_bmrb.push(norm);
                }
            }
            "--exclude-topology" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("missing --exclude-topology <id>")?
                    .trim()
                    .to_string();
                if raw.is_empty() {
                    return Err("invalid empty --exclude-topology value".to_string());
                }
                exclude_topology = Some(raw);
            }
            other => return Err(format!("unknown torsions flag {other:?}")),
        }
        i += 1;
    }

    let shifts_p = shifts_path.ok_or("torsions requires --shifts <assigned.str|assigned.nef>")?;
    let out_p = out_path.ok_or("torsions requires -o <out.tsv|out.nef|out.json>")?;
    let ext = out_p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(ext.as_str(), "tsv" | "nef" | "json") {
        return Err(format!(
            "unsupported torsions output extension {:?} on '{}' (expected .tsv, .nef, or .json)",
            ext,
            out_p.display()
        ));
    }

    let shifts = load_shifts_str_or_nef(&shifts_p)?;
    let has_firewall_overrides =
        !exclude_pdb.is_empty() || !exclude_bmrb.is_empty() || exclude_topology.is_some();
    let report = if has_firewall_overrides {
        let fw = HomologFirewallQuery {
            query_pdb_ids: exclude_pdb,
            query_bmrb_ids: exclude_bmrb,
            query_topology_id: exclude_topology,
            ..HomologFirewallQuery::default()
        };
        predict_torsions_with_firewall(&shifts, &fw)
    } else {
        predict_torsions(&shifts)
    };

    if let Some(parent) = out_p.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
        }
    }

    let export_opts = TorsionExportOptions { include_chi1 };
    match ext.as_str() {
        "tsv" => write_torsions_tsv_with_options(&out_p, &report, export_opts)?,
        "nef" => {
            let entry_name = shifts_p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("torsion_db_project");
            let mut doc = NefDocument::from_reference_and_peaks(entry_name, &shifts, None, &[]);
            doc.dihedral_restraint_lists
                .push(report.to_nef_dihedral_restraint_list_with_options(
                    "A",
                    "torsion_db_restraints",
                    export_opts,
                ));
            write_nef_file(&out_p, &doc)?;
        }
        "json" => {
            let json = format_torsions_json_with_options(&report, export_opts)?;
            fs::write(&out_p, format!("{json}\n"))
                .map_err(|e| format!("failed to write {}: {e}", out_p.display()))?;
        }
        _ => unreachable!(),
    }

    let mut n_good = 0usize;
    let mut n_dynamic = 0usize;
    let mut n_ambiguous = 0usize;
    let mut n_unassigned = 0usize;
    for r in &report.residues {
        match r.class {
            TorsionPredictionClass::Good => n_good += 1,
            TorsionPredictionClass::Dynamic => n_dynamic += 1,
            TorsionPredictionClass::Ambiguous => n_ambiguous += 1,
            TorsionPredictionClass::Unassigned => n_unassigned += 1,
        }
    }
    println!(
        "Predicted empirical 3/9-mer torsions for {} residues: Good={n_good}, Dynamic={n_dynamic}, Ambiguous={n_ambiguous}, Unassigned={n_unassigned} (excluded_templates={}, max_retained_seq_id={:.1}%, exclusion_list_sha256={}) -> {}",
        report.residues.len(),
        report.excluded_template_ids.len(),
        report.max_retained_seq_identity * 100.0,
        report.exclusion_list_sha256,
        out_p.display()
    );
    Ok(())
}

fn cmd_report(args: &[String]) -> Result<(), String> {
    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut pdb_path: Option<PathBuf> = None;
    let mut rdc_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut title: Option<String> = None;
    let mut report_opts = ReportOptions::default();
    let mut temp_k_explicit = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign report --star|--shifts <shifts.str|shifts.nef> [--peaks <peaks.tsv>] [--pdb <ensemble.pdb>] [--rdc <rdcs.tsv|rdcs.nef>] [--idp-propensity] [--temp-k 298.15] [--ph <f64>] --out <report.html> [--title <name>]"
                );
                return Ok(());
            }
            "--star" | "--shifts" => {
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
            "--rdc" => {
                i += 1;
                rdc_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --rdc <rdcs.tsv|rdcs.nef>")?,
                ));
            }
            "--out" | "-o" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "--title" => {
                i += 1;
                title = args.get(i).cloned();
            }
            "--idp-propensity" | "--propensity" => {
                report_opts.include_idp_propensity_track = true;
            }
            "--temp-k" | "--temperature-k" => {
                i += 1;
                report_opts.temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                temp_k_explicit = true;
            }
            "--ph" => {
                i += 1;
                report_opts.ph = Some(
                    args.get(i)
                        .ok_or("missing --ph")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            other => return Err(format!("unknown report flag {other:?}")),
        }
        i += 1;
    }

    if !report_opts.include_idp_propensity_track {
        if temp_k_explicit {
            return Err(
                "--temp-k requires --idp-propensity (or --propensity) in `nmrassign report`"
                    .to_string(),
            );
        }
        if report_opts.ph.is_some() {
            return Err(
                "--ph requires --idp-propensity (or --propensity) in `nmrassign report`"
                    .to_string(),
            );
        }
    }
    if rdc_path.is_some() && pdb_path.is_none() {
        return Err("--rdc requires --pdb <ensemble.pdb> in `nmrassign report`".to_string());
    }

    let star_p = star_path.ok_or("report requires --star <shifts.str>")?;
    let out_p = out_path.ok_or("report requires --out <report.html>")?;
    let ref_data = load_shifts_str_or_nef(&star_p)?;

    let mut conf = BTreeMap::new();
    for k in 0..ref_data.sequence.len() {
        let has_hn = ref_data.shifts.contains_key(&(k, "H".to_string()))
            && ref_data.shifts.contains_key(&(k, "N".to_string()));
        let has_c13_bb = ref_data.shifts.contains_key(&(k, "N".to_string()))
            && (ref_data.shifts.contains_key(&(k, "CA".to_string()))
                || ref_data.shifts.contains_key(&(k, "C".to_string())));
        if has_hn || has_c13_bb {
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
        if let Some(rdc_p) = rdc_path {
            let chain = parse_pdb_str_first_model(&pdb_txt)?;
            let rdc_media = load_rdc_media_auto(&rdc_p)?;
            let dropped_rdc = count_unresolved_rdc_restraints(&chain, &rdc_media);
            if dropped_rdc > 0 {
                eprintln!(
                    "warning: dropped {dropped_rdc} RDC restraint(s) with unresolved residue or atom coordinates"
                );
            }
            let rdc_summary = evaluate_rdc_validation(&chain, &rdc_media)?;
            input = input.with_rdc_report(rdc_summary);
        }
        input.pdb_coordinates = Some(pdb_txt);
    } else {
        input.dihedrals = predict_csi_dihedrals(&ref_data);
    }
    if report_opts.include_idp_propensity_track {
        let prop_cfg = build_propensity_config(report_opts.temp_k, report_opts.ph)?;
        let prop_rep = compute_idp_propensity(&ref_data, &prop_cfg);
        input = input.with_propensity_report(prop_rep);
    }

    if let Some(parent) = out_p.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
        }
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
    let mut con_path: Option<PathBuf> = None;
    let mut caco_path: Option<PathBuf> = None;
    let mut canco_path: Option<PathBuf> = None;
    let mut cbcacon_path: Option<PathBuf> = None;
    let mut cccon_path: Option<PathBuf> = None;
    let mut ncaco_path: Option<PathBuf> = None;
    let mut regime = RegimeKind::Globular;
    let mut metal_ligands_spec: Option<String> = None;
    let mut r_bleach_opt: Option<f64> = None;
    let mut pdb_para_path: Option<PathBuf> = None;
    let mut pcs_path: Option<PathBuf> = None;
    let mut report_opts = ReportOptions::default();
    let mut temp_k_explicit = false;
    let mut method = "classical".to_string();
    let mut method_explicit = false;
    let mut snr = 5.0f32;
    let mut prob = 0.45f32;
    let mut prob_explicit = false;
    let mut priors_path: Option<String> = None;
    let mut solver = SolverKind::EvoGraph;
    let mut solver_explicit = false;
    let mut runs: Option<usize> = None;
    let mut steps: Option<usize> = None;
    let mut assign_cost = 0.0f64;
    let mut assign_cost_explicit = false;
    let mut seed = 0u64;
    let mut name_arg: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut noesy_peaks_path: Option<PathBuf> = None;
    let mut engine_req_opt: Option<EngineRequest> = None;
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
            "--regime" => {
                i += 1;
                regime = parse_regime(args.get(i).ok_or("missing --regime")?)?;
            }
            "--metal-ligands" => {
                i += 1;
                metal_ligands_spec = Some(args.get(i).ok_or("missing --metal-ligands")?.clone());
            }
            "--r-bleach" => {
                i += 1;
                r_bleach_opt = Some(parse_bounded_r_bleach(
                    args.get(i).ok_or("missing --r-bleach")?,
                )?);
            }
            "--pdb" => {
                i += 1;
                pdb_para_path = Some(PathBuf::from(args.get(i).ok_or("missing --pdb")?));
            }
            "--pcs" => {
                i += 1;
                pcs_path = Some(PathBuf::from(args.get(i).ok_or("missing --pcs")?));
            }
            "--temp-k" | "--temperature-k" => {
                i += 1;
                report_opts.temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                temp_k_explicit = true;
            }
            "--ph" => {
                i += 1;
                report_opts.ph = Some(
                    args.get(i)
                        .ok_or("missing --ph")?
                        .parse()
                        .map_err(|e| format!("{e}"))?,
                );
            }
            "--idp-propensity" | "--propensity" => {
                report_opts.include_idp_propensity_track = true;
            }
            "--con" => {
                i += 1;
                con_path = Some(PathBuf::from(args.get(i).ok_or("missing --con")?));
            }
            "--caco" => {
                i += 1;
                caco_path = Some(PathBuf::from(args.get(i).ok_or("missing --caco")?));
            }
            "--canco" => {
                i += 1;
                canco_path = Some(PathBuf::from(args.get(i).ok_or("missing --canco")?));
            }
            "--cbcacon" | "--cbcaco" => {
                i += 1;
                cbcacon_path = Some(PathBuf::from(args.get(i).ok_or("missing --cbcacon")?));
            }
            "--cccon" => {
                i += 1;
                cccon_path = Some(PathBuf::from(args.get(i).ok_or("missing --cccon")?));
            }
            "--ncaco" => {
                i += 1;
                ncaco_path = Some(PathBuf::from(args.get(i).ok_or("missing --ncaco")?));
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
            "--no-axis-rescue" => fid_opts.no_axis_rescue = true,
            "--peaks" => {
                i += 1;
                peaks_path = args.get(i).map(PathBuf::from);
            }
            "--method" => {
                i += 1;
                method = args.get(i).ok_or("missing --method")?.to_ascii_lowercase();
                method_explicit = true;
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
                prob_explicit = true;
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing --solver")?)?;
                solver_explicit = true;
            }
            "--runs" => {
                i += 1;
                runs = Some(parse_bounded_runs(args.get(i).ok_or("missing --runs")?)?);
            }
            "--steps" => {
                i += 1;
                steps = Some(parse_bounded_steps(args.get(i).ok_or("missing --steps")?)?);
            }
            "--assign-cost" => {
                i += 1;
                assign_cost = args
                    .get(i)
                    .ok_or("missing --assign-cost")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                assign_cost_explicit = true;
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
                engine_req_opt = Some(parse_engine_request(
                    args.get(i).ok_or("missing --engine")?,
                )?);
            }
            "--fold" => fold = true,
            "-v" | "--verbose" => verbose = true,
            "--rdc" | "--torsion-db" | "--include-chi1" | "--rdc-weight" | "--rdc-free-frac" => {
                return Err(format!(
                    "{} is not supported in `nmrassign pipeline` (use post-assignment `nmrassign structure --rdc ...` or `nmrassign report --pdb ... --rdc ...`)",
                    args[i]
                ))
            }
            other => return Err(format!("unknown pipeline flag {other:?}")),
        }
        i += 1;
    }

    if regime == RegimeKind::Idp13C {
        report_opts.include_idp_propensity_track = true;
    }
    if regime != RegimeKind::Paramagnetic {
        if pdb_para_path.is_some() {
            return Err("--pdb is only supported with --regime paramagnetic (use --pdb-ref for globular --fold)".to_string());
        }
        if pcs_path.is_some() {
            return Err("--pcs is only supported with --regime paramagnetic".to_string());
        }
        if metal_ligands_spec.is_some() {
            return Err("--metal-ligands is only supported with --regime paramagnetic".to_string());
        }
        if r_bleach_opt.is_some() {
            return Err("--r-bleach is only supported with --regime paramagnetic".to_string());
        }
    }
    if regime != RegimeKind::Idp13C {
        let has_c13_flag = con_path.is_some()
            || caco_path.is_some()
            || canco_path.is_some()
            || cbcacon_path.is_some()
            || cccon_path.is_some()
            || ncaco_path.is_some()
            || triples
                .iter()
                .any(|(exp, _)| C13DetectedExp::from_name(exp).is_some());
        if has_c13_flag {
            return Err(
                "13C-detected spectrum flags (--con, --caco, --canco, --cbcacon, --cccon, --ncaco) are only supported with --regime idp-13c"
                    .to_string(),
            );
        }
        if !report_opts.include_idp_propensity_track {
            if temp_k_explicit {
                return Err(
                    "--temp-k in `nmrassign pipeline` requires --regime idp-13c or --idp-propensity"
                        .to_string(),
                );
            }
            if report_opts.ph.is_some() {
                return Err(
                    "--ph in `nmrassign pipeline` requires --regime idp-13c or --idp-propensity"
                        .to_string(),
                );
            }
        }
    } else {
        if solver_explicit {
            return Err("--solver is not supported with --regime idp-13c".to_string());
        }
        if priors_path.is_some() {
            return Err("--priors is not supported with --regime idp-13c".to_string());
        }
        if assign_cost_explicit {
            return Err("--assign-cost is not supported with --regime idp-13c".to_string());
        }
        if hsqc_path.is_some()
            || fid_bundle_path.is_some()
            || fid_hsqc_path.is_some()
            || !fid_triples.is_empty()
        {
            return Err(
                "1H-detected --hsqc / --fid / --fid-hsqc / --fid-triple flags are not supported with --regime idp-13c"
                    .to_string(),
            );
        }
        if fold || pdb_ref_path.is_some() || noesy_peaks_path.is_some() {
            return Err(
                "--fold / --pdb-ref / --noesy-peaks globular NOESY folding flags are not supported with --regime idp-13c"
                    .to_string(),
            );
        }
        if nef_path.is_some() {
            return Err(
                "--nef input is not supported with --regime idp-13c in `nmrassign pipeline` (use --seq or --star)"
                    .to_string(),
            );
        }
        if method_explicit || prob_explicit {
            return Err(
                "--method / --prob 1H neural-picker flags are not supported with --regime idp-13c"
                    .to_string(),
            );
        }
    }

    let dir = out_dir.ok_or("pipeline requires --out-dir <dir>")?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    if regime == RegimeKind::Idp13C {
        let t_total = Instant::now();
        let (sequence, seq_ids, project_name, star_ref_opt) = if let Some(sq) = &seq_arg {
            let (s, ids) = parse_sequence_arg(sq, first_seq_id)?;
            let nm = name_arg.unwrap_or_else(|| "idp_13c_pipeline".to_string());
            (s, ids, nm, None)
        } else if let Some(sp) = &star_path {
            let r = read_star_all_atoms(sp, None)?;
            let nm = name_arg.unwrap_or_else(|| {
                sp.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("idp_13c_pipeline")
                    .to_string()
            });
            let seq = r.sequence.clone();
            let ids = r.seq_ids.clone();
            (seq, ids, nm, Some(r))
        } else {
            return Err("pipeline --regime idp-13c requires --seq or --star".to_string());
        };

        let has_explicit_c13 = peaks_path.is_some()
            || con_path.is_some()
            || caco_path.is_some()
            || canco_path.is_some()
            || cbcacon_path.is_some()
            || cccon_path.is_some()
            || ncaco_path.is_some()
            || !triples.is_empty();

        let ds = if has_explicit_c13 {
            let pp_vec: Vec<PathBuf> = peaks_path.into_iter().collect();
            load_c13_input_bundle(
                &pp_vec,
                con_path.as_deref(),
                caco_path.as_deref(),
                canco_path.as_deref(),
                cbcacon_path.as_deref(),
                cccon_path.as_deref(),
                ncaco_path.as_deref(),
                &triples,
                snr,
            )?
        } else if let Some(ref r) = star_ref_opt {
            let cubes = simulate_idp_13c_cubes(
                r,
                &IdpCubeSimConfig {
                    seed,
                    ..IdpCubeSimConfig::default()
                },
            );
            let pcfg = C13PickerConfig::default();
            let con_roots = pick_con_2d(&cubes.con_2d, &pcfg);
            let caco_peaks = cubes
                .caco_2d
                .as_ref()
                .map(|sp| pick_caco_2d(sp, &pcfg))
                .unwrap_or_default();
            let mut strip_peaks = Vec::new();
            for (exp, ref sp3) in &cubes.cubes_3d {
                strip_peaks.extend(pick_con_anchored_3d(sp3, *exp, &con_roots, &pcfg));
            }
            C13PickedDataset {
                con_roots,
                caco_peaks,
                strip_peaks,
            }
        } else {
            return Err(
                "pipeline --regime idp-13c requires 13C-detected peaks/spectra (or --star)"
                    .to_string(),
            );
        };

        let t_solve = Instant::now();
        let wf = execute_idp_13c_workflow(
            &ds,
            &sequence,
            &seq_ids,
            engine_req_opt.unwrap_or(EngineRequest::Auto),
            report_opts.temp_k,
            report_opts.ph,
            runs,
            steps,
            seed,
        )?;
        let dt_solve = t_solve.elapsed().as_secs_f64();

        let peaks_out = dir.join("peaks.tsv");
        let star_out = dir.join("shifts.str");
        let tsv_out = dir.join("shifts.tsv");
        let nef_out = dir.join("result.nef");
        let html_out = dir.join("report.html");
        let prop_out = dir.join("propensity.tsv");

        write_c13_detected_tsv(&peaks_out, &ds)?;
        write_star_with_header(&star_out, &wf.pred, Some(&wf.conf), Some(&wf.star_header))?;
        write_shifts_tsv(&tsv_out, &wf.pred, &wf.conf)?;
        let nef_doc =
            NefDocument::from_reference_and_peaks(&project_name, &wf.pred, Some(&wf.conf), &[]);
        write_nef_file(&nef_out, &nef_doc)?;

        let prop_cfg = build_propensity_config(report_opts.temp_k, report_opts.ph)?;
        let prop_rep = compute_idp_propensity(&wf.pred, &prop_cfg);
        fs::write(&prop_out, format!("{}\n", format_propensity_tsv(&prop_rep)))
            .map_err(|e| format!("{e}"))?;

        let mut rep = ReportInput::new(&project_name, wf.pred.clone(), wf.conf.clone());
        rep.n_peaks = Some(ds.con_roots.len() + ds.caco_peaks.len() + ds.strip_peaks.len());
        rep.n_spin_systems = Some(wf.systems.len());
        rep.solver_score = Some(wf.result.score);
        rep.elapsed_secs = Some(dt_solve);
        rep.engine_label = Some(wf.engine_label.clone());
        rep.dihedrals = predict_csi_dihedrals(&wf.pred);
        rep = rep.with_propensity_report(prop_rep);
        write_quality_report_html(&html_out, &rep)?;

        let dt_total = t_total.elapsed().as_secs_f64();
        println!(
            "[{}] Pipeline '{}' (idp-13c) completed in {:.2}s (solve {:.2}s) -> {}, {}, {}, {}, {}, {}",
            wf.engine_label,
            project_name,
            dt_total,
            dt_solve,
            star_out.display(),
            tsv_out.display(),
            peaks_out.display(),
            prop_out.display(),
            nef_out.display(),
            html_out.display()
        );
        return Ok(());
    }

    let engine_opt = match engine_req_opt {
        None | Some(EngineRequest::Auto) => None,
        Some(EngineRequest::Gpu) => Some(EngineKind::Gpu),
        Some(EngineRequest::Cpu) => Some(EngineKind::Cpu),
    };
    if regime == RegimeKind::Paramagnetic && engine_req_opt == Some(EngineRequest::Gpu) {
        return Err(
            "--engine gpu is not supported with --regime paramagnetic (CPU-only); use --engine cpu or --engine auto"
                .to_string(),
        );
    }
    if regime == RegimeKind::Globular {
        let _validated_solver_engine = resolve_solver_engine(solver, engine_opt)?;
    }
    let picker_engine = engine_opt.unwrap_or(EngineKind::Gpu);

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
    let mut pipeline_axis_audits: Vec<AxisRescueAuditRecord> = Vec::new();
    let (base_ref, peaks, project_name) = if let Some(np) = &nef_path {
        let doc = read_nef_file(np)?;
        let r = doc.to_reference()?;
        let pks = if let Some((ref fhsqc, ref ftriples)) = resolved_fid {
            let (_, all_pks, _, recipe_json, audits) = process_and_pick_fid_bundle(
                fhsqc,
                ftriples,
                &triples,
                &fid_opts,
                &method,
                picker_engine,
                snr,
                prob,
            )?;
            pipeline_axis_audits = audits;
            fs::write(
                dir.join("processing_recipe.json"),
                format!("{recipe_json}\n"),
            )
            .map_err(|e| format!("failed to write processing_recipe.json: {e}"))?;
            all_pks
        } else if let Some(hp) = &hsqc_path {
            let (_, all_pks, _, audits) = pick_spectra_bundle_with_audit(
                hp,
                &triples,
                &method,
                picker_engine,
                snr,
                prob,
                !fid_opts.no_axis_rescue,
            )?;
            pipeline_axis_audits = audits;
            all_pks
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
        let (roots, pks, method_label, recipe_json, audits) = process_and_pick_fid_bundle(
            fhsqc,
            ftriples,
            &triples,
            &fid_opts,
            &method,
            picker_engine,
            snr,
            prob,
        )?;
        pipeline_axis_audits = audits;
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
        let (roots, pks, method_label, audits) = pick_spectra_bundle_with_audit(
            hp,
            &triples,
            &method,
            picker_engine,
            snr,
            prob,
            !fid_opts.no_axis_rescue,
        )?;
        pipeline_axis_audits = audits;
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
    let peaks_out = dir.join("peaks.tsv");
    let star_out = dir.join("shifts.str");
    let tsv_out = dir.join("shifts.tsv");
    let nef_out = dir.join("result.nef");
    let html_out = dir.join("report.html");
    let qc_out = dir.join("qc_report.json");

    let (pred_ref, conf, systems, result, engine_label, dt_solve, star_hdr, qc_opt) = if regime
        == RegimeKind::Paramagnetic
    {
        let (sys, res, eff_priors, para_rep, pcs_tensor_opt, elbl) = execute_paramagnetic_workflow(
            solver,
            engine_req_opt.unwrap_or(EngineRequest::Auto),
            &peaks,
            &base_ref.sequence,
            &base_ref.seq_ids,
            &pri,
            &scfg,
            metal_ligands_spec.as_deref(),
            r_bleach_opt,
            pdb_para_path.as_deref(),
            pcs_path.as_deref(),
            runs,
            steps,
            seed,
        )?;
        let dt = t_solve.elapsed().as_secs_f64();
        let (pr, cf) = to_reference(
            &res,
            &sys,
            &base_ref.sequence,
            &base_ref.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        let mut hdr = format!(
                "# nmrassign solver: {elbl}\n# regime: paramagnetic (clusters={}, blind_zone_residues={}, abstentions={})",
                para_rep.detected_clusters.len(),
                para_rep.blind_zone_residues.len(),
                para_rep.n_blind_zone_abstentions,
            );
        if let Some(ref t) = pcs_tensor_opt {
            hdr.push_str(&format!(
                "\n# pcs_delta_chi: dchi_ax={:+.3}, dchi_rh={:+.3} (10^-32 m^3), q_factor={:.4}",
                t.dchi_ax, t.dchi_rh, t.q_factor,
            ));
        }
        (pr, cf, sys, res, elbl, dt, Some(hdr), None)
    } else {
        let wf = execute_assignment_workflow(
            solver,
            engine_opt,
            &peaks,
            &base_ref.sequence,
            &pri,
            &scfg,
            runs,
            steps,
            STANDARD_CPU_CALIBRATED_RUNS,
            seed,
        )?;
        let dt = t_solve.elapsed().as_secs_f64();
        log_workflow_diagnostics(
            &wf.engine_label,
            &wf.offsets,
            &wf.detailed_offsets,
            &wf.qc_report,
            verbose,
        );
        let (pr, cf) = to_reference(
            &wf.result,
            &wf.systems,
            &base_ref.sequence,
            &base_ref.seq_ids,
            0.0,
            Some(&wf.effective_priors),
        );
        let hdr = format_star_offset_header(
            &wf.engine_label,
            &wf.offsets,
            &wf.detailed_offsets,
            &wf.qc_report,
        );
        (
            pr,
            cf,
            wf.systems,
            wf.result,
            wf.engine_label,
            dt,
            hdr,
            Some(wf.qc_report),
        )
    };

    write_peaks(&peaks_out, &peaks)?;
    write_star_with_header(&star_out, &pred_ref, Some(&conf), star_hdr.as_deref())?;
    write_shifts_tsv(&tsv_out, &pred_ref, &conf)?;
    if let Some(ref qcr) = qc_opt {
        let _ = fs::write(
            &qc_out,
            format!("{}\n", real_benchmark::render_qc_report_json(qcr, 0)),
        );
    }
    let nef_doc =
        NefDocument::from_reference_and_peaks(&project_name, &pred_ref, Some(&conf), &peaks);
    write_nef_file(&nef_out, &nef_doc)?;

    let mut rep = ReportInput::new(&project_name, pred_ref.clone(), conf.clone());
    rep.n_peaks = Some(peaks.len());
    rep.n_spin_systems = Some(systems.len());
    rep.solver_score = Some(result.score);
    rep.elapsed_secs = Some(dt_solve);
    rep.engine_label = Some(engine_label.clone());
    rep.systems_and_assignment = Some((systems, result));
    rep.dihedrals = predict_csi_dihedrals(&pred_ref);
    rep.axis_rescue_records = pipeline_axis_audits;
    if report_opts.include_idp_propensity_track {
        let prop_cfg = build_propensity_config(report_opts.temp_k, report_opts.ph)?;
        let prop_rep = compute_idp_propensity(&pred_ref, &prop_cfg);
        rep = rep.with_propensity_report(prop_rep);
    }

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
        "propensity" => cmd_propensity(&args[1..]),
        "torsions" => cmd_torsions(&args[1..]),
        "score" => cmd_score(&args[1..]),
        "pick" => cmd_pick(&args[1..]),
        "priors" => cmd_priors(&args[1..]),
        "sidechain" => cmd_sidechain(&args[1..]),
        "methyl" => methyl_cmd::run_methyl_cli(&args[1..]),
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

        let (_, _, lbl_gpu_2d, _) = pick_spectra_bundle_with_audit(
            &hsqc_ucsf,
            &[],
            "neural",
            EngineKind::Gpu,
            5.0,
            0.45,
            true,
        )
        .unwrap();
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

        // 2. Loud rejection of `--pdb` without `--regime paramagnetic` and removed `tripeptide` solver
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
            pdb_err.contains("--regime paramagnetic"),
            "expected structure-free rejection for --pdb without --regime paramagnetic, got: {pdb_err}"
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
        use nmr_core::priors::OffsetFitStatus;
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
