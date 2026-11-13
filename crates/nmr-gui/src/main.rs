//! Interactive WebGPU + `eframe`/`egui` NMR spectrum and backbone resonance assignment workbench.
//!
//! Provenance: Original scientific GUI built on `eframe`, `wgpu`, `nmr-core`, and `nmr-gpu`
//! (native Metal/Vulkan/DX12 and browser `wasm32-unknown-unknown` + WebGPU).

#[allow(dead_code)]
mod octet_canvas;

use std::collections::{BTreeMap, BTreeSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use eframe::{egui, egui_wgpu};
use nmr_core::aromatic::{
    assign_aromatic_rings, merge_aromatic_assignments_preserving_existing, AromaticRingCluster,
    AromaticSolverConfig, AROMATIC_EXPERIMENTS,
};
use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, simulate_idp_13c_cubes, C13DetectedExp,
    C13PickedDataset, C13PickerConfig, C13Strip3DPeak, Caco2DPeak, CacoPeakKind, Con2DPeak,
    IdpCubeSimConfig,
};
use nmr_core::classify::DataState;
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::classify::{ExperimentDimensionality, ExperimentVendor};
use nmr_core::evaluate::{score, to_reference, Metrics};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig, PropensityReport};
use nmr_core::idp_solver::{
    assign_idp_13c_cpu, idp_to_reference, register_and_build_con_dipeptide_systems, IdpSolverConfig,
};
use nmr_core::nef::read_nef_str;
use nmr_core::neural_picker::{pick_hsqc_2d_neural, NeuralPickerConfig};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::nmrstar::{read_star, read_star_all_atoms, write_star};
use nmr_core::nmrstar::{read_star_str, read_star_str_all_atoms, Reference};
use nmr_core::octet_analyt::iupac_naming::{iupac_name_to_mol, mol_to_iupac_name};
use nmr_core::octet_analyt::jcamp_spec1d::{
    pick_1d_multiplets, read_jcamp, synthesize_1d_nmr, write_jcamp_nmr, JcampCompression,
    Synthesize1dOptions,
};
use nmr_core::octet_analyt::mass_props::compute_phys_chem_props;
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{
    build_peptide_2d, clean_2d, expand_abbreviations, kekulize, parse_biopolymer_sequence,
    parse_condensed_formula, parse_inchi, parse_reaction_smiles, parse_smiles, perceive_sssr,
    read_cdx, read_cdxml, read_mol, read_oct, read_sdf, write_canonical_smiles, write_cdx,
    write_cdxml, write_inchi, write_inchikey, write_mol_v2000, write_oct, write_reaction_smiles,
    write_sdf, ChemDocument, Molecule, SdfRecord,
};
use nmr_core::octet_render::{
    render_scene_svg, DocumentStyle, RenderScene, SceneSpectrum1D, SpectrumSignal1D,
};
use nmr_core::paramagnetic::{
    assign_paramagnetic_backbone, assign_paramagnetic_from_peaks, parse_metal_ligands,
    ParamagneticAssignConfig, ParamagneticSolverMode,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::peaks::read_peaks;
use nmr_core::peaks::{read_peaks_str, Peak, EXPERIMENT_NAMES};
use nmr_core::picker::{estimate_noise_mad, pick_hsqc_2d, PickerConfig};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::picker::{pick_c13_hsqc_2d, pick_triple_3d_two_pass, CorroboratedRepickConfig};
use nmr_core::priors::{one_to_three, three_to_one, Priors, ATOMS};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::process::{process_fid_dir_2d, process_fid_dir_3d, register_3d_hn_to_hsqc_2d};
use nmr_core::qc::{evaluate_dataset_qc, DatasetQcReport};
use nmr_core::report::{generate_quality_report_html, predict_csi_dihedrals, ReportInput};
use nmr_core::rng::Rng;
use nmr_core::session::{
    ResidueAssignmentState, SessionContourParams, WorkbenchCommand, WorkbenchSessionState,
};
use nmr_core::sidechain::{
    assign_sidechains, solve_joint_backbone_sidechain, SidechainSolverConfig,
};
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, AssignResult, Matrix2D,
    ScoreConfig, FORBID,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d, read_ucsf_2d,
    read_ucsf_3d,
};
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::stereospec::{
    assign_stereospecific_dataset, StereospecConfig, StereospecDatasetInput,
};
use nmr_core::topology::SidechainPriors;
#[cfg(not(target_arch = "wasm32"))]
use nmr_gpu::GpuEngine;
#[cfg(not(target_arch = "wasm32"))]
use nmr_gui::ingest_panel::ingest_dropped_files_native;
#[cfg(target_arch = "wasm32")]
use nmr_gui::ingest_panel::ingest_dropped_files_wasm;
#[cfg(target_arch = "wasm32")]
use nmr_gui::session_io::{default_session_nef_filename, trigger_browser_nef_download};
use nmr_gui::{
    command_palette::CommandPalette,
    contour_painter::{
        default_overlay_palette, ContourPipelineResource, MultiContourCallback, OverlayLayerSpec,
        SyncCrosshair,
    },
    ingest_panel::{IngestAction, IngestDropOutcome, IngestExecutionPlan, IngestPanelState},
    keymap::{poll_global_modifier_shortcuts, ChordDispatcher, CONTOUR_FLOOR_STEP_RATIO},
    mol3d_overlay::Mol3dSyncAction,
    mol3d_viewer::{Mol3dPipelineResource, Mol3dViewerPanel},
    session_io::WorkbenchSessionController,
    sidechain_widget::{
        AssignedAtomInfo, SequenceRibbonState, SidechainInspectorPanel, SidechainWorkspaceInput,
    },
    strip_tiler::{StripSpectra3D, StripTilerAction, StripTilerContext, StripTilerPanel},
    workflow_bar::{draw_workbench_chrome, WorkbenchAction, WorkspaceReadiness},
};
use octet_canvas::{IoFormat, MolecularPropertiesSummary, OctetBackendAdapter, OctetDrawPanel};

const UBIQUITIN_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

const DEMO_UBIQUITIN_1UBQ_PDB: &str =
    include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

struct PerfTimer {
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
    #[cfg(target_arch = "wasm32")]
    start_ms: f64,
}

impl PerfTimer {
    fn start() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                start: Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let start_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0);
            Self { start_ms }
        }
    }

    fn elapsed_ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            let now_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(self.start_ms);
            (now_ms - self.start_ms).max(0.0)
        }
    }
}

fn build_synthetic_ubiquitin() -> Reference {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(r) = read_star("data/fixtures/synthetic_ubiquitin.str", None) {
        return r;
    }
    if let Ok(r) = read_star_str(SYNTHETIC_UBIQUITIN_STR, None) {
        return r;
    }
    let pri = Priors::provisional();
    let mut rng = Rng::new(1);
    let seq: Vec<String> = UBIQUITIN_SEQ
        .chars()
        .map(|c| one_to_three(c).to_string())
        .collect();
    let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
    let mut r = Reference::new(seq.clone(), seq_ids);
    for (k, res) in seq.iter().enumerate() {
        if let Some(atoms) = pri.table.get(res) {
            for (atom, &(mu, sd)) in atoms {
                if atom == "H" && (k == 0 || res == "PRO") {
                    continue;
                }
                r.shifts.insert((k, atom.clone()), rng.normal(mu, sd));
            }
        }
    }
    r
}

fn format_star_in_memory(
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
) -> String {
    let mut lines = vec![
        "data_nmr_assign_output".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
    ];
    for t in [
        "ID",
        "Entity_ID",
        "Seq_ID",
        "Comp_ID",
        "Atom_ID",
        "Atom_type",
        "Val",
        "Ambiguity_code",
        "Details",
    ] {
        lines.push(format!("      _Atom_chem_shift.{t}"));
    }
    lines.push(String::new());

    let mut entries: Vec<(usize, &str, f64)> = ref_data
        .shifts
        .iter()
        .map(|(&(idx, ref atom), &v)| (idx, atom.as_str(), v))
        .collect();
    entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
        let oa = ATOMS.iter().position(|&a| a == aa).unwrap_or(99);
        let ob = ATOMS.iter().position(|&a| a == ab).unwrap_or(99);
        (ia, oa, aa).cmp(&(ib, ob, ab))
    });

    for (i, (idx, atom, v)) in entries.into_iter().enumerate() {
        let det = confidence
            .and_then(|c| c.get(&idx))
            .map(|&c| format!("conf={c:.2}"))
            .unwrap_or_else(|| ".".to_string());
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_string())
            .unwrap_or_else(|| "X".to_string());
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} 1 {}",
            i + 1,
            ref_data.seq_ids[idx],
            ref_data.sequence[idx],
            atom,
            atom_type,
            v,
            det
        ));
    }
    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());
    lines.join("\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceMode {
    ProteinNmr,
    OctetChem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GuiSolverKind {
    #[default]
    EvoGraph,
    Standard,
}

impl GuiSolverKind {
    fn label(self) -> &'static str {
        match self {
            Self::EvoGraph => "Evo-Graph",
            Self::Standard => "Standard",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GuiRegimeKind {
    #[default]
    Globular,
    Idp13C,
    Paramagnetic,
}

impl GuiRegimeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Globular => "Globular",
            Self::Idp13C => "13C-IDP",
            Self::Paramagnetic => "Paramagnetic",
        }
    }
}

fn parse_gui_c13_tsv_str(text: &str) -> Option<C13PickedDataset> {
    let mut ds = C13PickedDataset::default();
    let mut saw_c13_row = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        let cols: Vec<&str> = if cols.len() <= 1 {
            line.split_whitespace().collect()
        } else {
            cols
        };
        if cols.len() < 3 {
            continue;
        }
        if matches!(
            cols[0].to_ascii_lowercase().as_str(),
            "experiment" | "exp" | "spectrum" | "peak_id"
        ) {
            continue;
        }
        let exp = C13DetectedExp::from_name(cols[0])?;
        saw_c13_row = true;
        let rem = &cols[1..];
        let opt_f64 = |s: &str| -> Option<f64> {
            let t = s.trim();
            if t.is_empty() || t == "." || t == "-" {
                None
            } else {
                t.parse::<f64>().ok()
            }
        };
        match exp {
            C13DetectedExp::Con2D => {
                let c_prev = opt_f64(rem.first().copied()?)?;
                let n_curr = opt_f64(rem.get(1).copied()?)?;
                ds.con_roots.push(Con2DPeak {
                    c_prev_ppm: c_prev,
                    n_curr_ppm: n_curr,
                    height: 1.0e5,
                    volume: 1.0e5,
                    is_proline_imino_region: n_curr >= 131.5,
                    deconvolved_from_cluster: false,
                });
            }
            C13DetectedExp::Caco2D => {
                let f0 = opt_f64(rem.first().copied()?)?;
                let ca = opt_f64(rem.get(1).copied()?).or_else(|| opt_f64(rem.get(2).copied()?))?;
                let (c_curr_ppm, ca_curr_ppm) = if f0 < 100.0 && ca > 140.0 {
                    (ca, f0)
                } else {
                    (f0, ca)
                };
                let kind = if ca_curr_ppm <= 47.8 {
                    CacoPeakKind::GlyRegion
                } else if ca_curr_ppm >= 58.0 {
                    CacoPeakKind::SerThrProDownfield
                } else {
                    CacoPeakKind::GeneralAliphatic
                };
                ds.caco_peaks.push(Caco2DPeak {
                    c_curr_ppm,
                    ca_curr_ppm,
                    sign: 1,
                    height: 1.0e5,
                    volume: 1.0e5,
                    kind,
                });
            }
            _ => {
                let c_prime_ppm = opt_f64(rem.first().copied()?)?;
                let n_ppm = opt_f64(rem.get(1).copied()?)?;
                let c_indirect_ppm = opt_f64(rem.get(2).copied()?)?;
                let sign = rem.get(3).and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
                let anchor_root_idx = rem.get(6).and_then(|s| s.parse::<usize>().ok());
                ds.strip_peaks.push(C13Strip3DPeak {
                    exp,
                    c_prime_ppm,
                    n_ppm,
                    c_indirect_ppm,
                    sign,
                    height: 1.0e5,
                    volume: 1.0e5,
                    anchor_root_idx,
                });
            }
        }
    }
    if !saw_c13_row || ds.con_roots.is_empty() {
        return None;
    }
    // Anchor any unanchored CANCO/CBCACON/CCCON peaks onto nearest CON root
    for p in &mut ds.strip_peaks {
        if p.anchor_root_idx.is_none() && p.exp != C13DetectedExp::Ncaco3D {
            let mut best: Option<(usize, f64)> = None;
            for (r_idx, r) in ds.con_roots.iter().enumerate() {
                let dc = (p.c_prime_ppm - r.c_prev_ppm).abs();
                let dn = (p.n_ppm - r.n_curr_ppm).abs();
                if dc <= 0.25 && dn <= 0.65 {
                    let d2 = (dc / 0.06).powi(2) + (dn / 0.25).powi(2);
                    if best.is_none_or(|(_, b)| d2 < b) {
                        best = Some((r_idx, d2));
                    }
                }
            }
            p.anchor_root_idx = best.map(|(i, _)| i);
        }
    }
    Some(ds)
}

#[derive(Clone, Copy, Debug, Default)]
struct ClusterCuratedEval {
    total_cur_obs: usize,
    total_clu_obs: usize,
    shared_obs: usize,
    conflict_obs: usize,
    cb_contradicts: bool,
    norm_dist_sq: f64,
}

fn eval_cluster_vs_curated_residue(
    clu: &AromaticRingCluster,
    k_cur: usize,
    curated_ref: &Reference,
    cfg: &AromaticSolverConfig,
) -> ClusterCuratedEval {
    let res_name = curated_ref
        .sequence
        .get(k_cur)
        .map(String::as_str)
        .unwrap_or("");
    if clu.kind.res_name() != res_name {
        return ClusterCuratedEval {
            conflict_obs: 1,
            ..ClusterCuratedEval::default()
        };
    }

    let tol_h = (cfg.tol_h * 2.0).max(0.055);
    let tol_c = (cfg.tol_c * 1.6).max(0.45);
    let tol_n = (cfg.tol_n * 1.6).max(0.65);

    let logical_schema: &[(&str, Option<&str>, f64)] = match res_name {
        "PHE" => &[
            ("CD1", Some("CD2"), tol_c),
            ("HD1", Some("HD2"), tol_h),
            ("CE1", Some("CE2"), tol_c),
            ("HE1", Some("HE2"), tol_h),
            ("CZ", None, tol_c),
            ("HZ", None, tol_h),
        ],
        "TYR" => &[
            ("CD1", Some("CD2"), tol_c),
            ("HD1", Some("HD2"), tol_h),
            ("CE1", Some("CE2"), tol_c),
            ("HE1", Some("HE2"), tol_h),
        ],
        "TRP" => &[
            ("CD1", None, tol_c),
            ("HD1", None, tol_h),
            ("CE3", None, tol_c),
            ("HE3", None, tol_h),
            ("CZ3", None, tol_c),
            ("HZ3", None, tol_h),
            ("CH2", None, tol_c),
            ("HH2", None, tol_h),
            ("CZ2", None, tol_c),
            ("HZ2", None, tol_h),
            ("NE1", None, tol_n),
            ("HE1", None, tol_h),
        ],
        "HIS" => &[
            ("CD2", None, tol_c),
            ("HD2", None, tol_h),
            ("CE1", None, tol_c),
            ("HE1", None, tol_h),
            ("ND1", None, tol_n),
            ("NE2", None, tol_n),
        ],
        _ => &[],
    };

    let clu_val_for = |primary: &str| -> Option<f64> {
        match (res_name, primary) {
            ("PHE" | "TYR", "CD1") => clu.ch_slots.get("CD1").map(|s| s.c_ppm),
            ("PHE" | "TYR", "HD1") => clu.ch_slots.get("CD1").map(|s| s.h_ppm),
            ("PHE" | "TYR", "CE1") => clu.ch_slots.get("CE1").map(|s| s.c_ppm),
            ("PHE" | "TYR", "HE1") => clu.ch_slots.get("CE1").map(|s| s.h_ppm),
            ("PHE", "CZ") => clu.ch_slots.get("CZ").map(|s| s.c_ppm),
            ("PHE", "HZ") => clu.ch_slots.get("CZ").map(|s| s.h_ppm),
            ("TRP", "CD1") => clu.ch_slots.get("CD1").map(|s| s.c_ppm),
            ("TRP", "HD1") => clu.ch_slots.get("CD1").map(|s| s.h_ppm),
            ("TRP", "CE3") => clu.ch_slots.get("CE3").map(|s| s.c_ppm),
            ("TRP", "HE3") => clu.ch_slots.get("CE3").map(|s| s.h_ppm),
            ("TRP", "CZ3") => clu.ch_slots.get("CZ3").map(|s| s.c_ppm),
            ("TRP", "HZ3") => clu.ch_slots.get("CZ3").map(|s| s.h_ppm),
            ("TRP", "CH2") => clu.ch_slots.get("CH2").map(|s| s.c_ppm),
            ("TRP", "HH2") => clu.ch_slots.get("CH2").map(|s| s.h_ppm),
            ("TRP", "CZ2") => clu.ch_slots.get("CZ2").map(|s| s.c_ppm),
            ("TRP", "HZ2") => clu.ch_slots.get("CZ2").map(|s| s.h_ppm),
            ("TRP", "NE1") => clu.trp_indole.map(|(n, _, _)| n),
            ("TRP", "HE1") => clu.trp_indole.map(|(_, h, _)| h),
            ("HIS", "CD2") => clu.ch_slots.get("CD2").map(|s| s.c_ppm),
            ("HIS", "HD2") => clu.ch_slots.get("CD2").map(|s| s.h_ppm),
            ("HIS", "CE1") => clu.ch_slots.get("CE1").map(|s| s.c_ppm),
            ("HIS", "HE1") => clu.ch_slots.get("CE1").map(|s| s.h_ppm),
            ("HIS", "ND1") => clu.nd1_ppm.map(|(n, _)| n),
            ("HIS", "NE2") => clu.ne2_ppm.map(|(n, _)| n),
            _ => None,
        }
    };

    let mut out = ClusterCuratedEval::default();
    for &(a1, a2_opt, tol) in logical_schema {
        let cur1 = curated_ref.shifts.get(&(k_cur, a1.to_string())).copied();
        let cur2 = a2_opt.and_then(|a2| curated_ref.shifts.get(&(k_cur, a2.to_string())).copied());
        let has_cur = cur1.is_some() || cur2.is_some();
        if has_cur {
            out.total_cur_obs += 1;
        }
        let Some(cv) = clu_val_for(a1) else {
            continue;
        };
        out.total_clu_obs += 1;
        if !has_cur {
            continue;
        }
        let best_diff = [cur1, cur2]
            .into_iter()
            .flatten()
            .map(|v| (cv - v).abs())
            .fold(f64::INFINITY, f64::min);
        if best_diff <= tol {
            out.shared_obs += 1;
            out.norm_dist_sq += (best_diff / tol).powi(2);
        } else {
            out.conflict_obs += 1;
        }
    }

    if let Some(&cur_cb) = curated_ref.shifts.get(&(k_cur, "CB".to_string())) {
        let cbs: Vec<f64> = [clu.cb_hd, clu.cb_he].into_iter().flatten().collect();
        if !cbs.is_empty()
            && cbs
                .iter()
                .all(|&cb| (cb - cur_cb).abs() > cfg.tol_cb * 1.20)
        {
            out.cb_contradicts = true;
        }
    }

    out
}

fn peak_matches_aromatic_cluster(
    p: &Peak,
    clu: &AromaticRingCluster,
    cfg: &AromaticSolverConfig,
) -> bool {
    let tol_h = (cfg.tol_h * 1.5).max(0.045);
    let tol_c = (cfg.tol_c * 1.5).max(0.42);
    let tol_n = (cfg.tol_n * 1.5).max(0.55);
    match p.exp.as_str() {
        "C13HSQC" => clu.ch_slots.values().any(|s| {
            (p.h - s.h_ppm).abs() <= tol_h && p.c.is_some_and(|c| (c - s.c_ppm).abs() <= tol_c)
        }),
        "AROM_HCCH_TOCSY" | "HCCHTOCSY" => {
            clu.ch_slots
                .values()
                .any(|s| (p.h - s.h_ppm).abs() <= tol_h)
                && clu
                    .ch_slots
                    .values()
                    .any(|s| p.c.is_some_and(|c| (c - s.c_ppm).abs() <= tol_c))
        }
        "HBCBCGCDHD" => {
            clu.ch_slots
                .get("CD1")
                .or_else(|| clu.ch_slots.get("CD2"))
                .is_some_and(|s| (p.h - s.h_ppm).abs() <= tol_h * 1.25)
                && clu
                    .cb_hd
                    .is_some_and(|cb| p.c.is_some_and(|c| (c - cb).abs() <= cfg.tol_cb * 1.15))
        }
        "HBCBCGCDCEHE" => {
            clu.ch_slots
                .get("CE1")
                .or_else(|| clu.ch_slots.get("CE3"))
                .is_some_and(|s| (p.h - s.h_ppm).abs() <= tol_h * 1.25)
                && clu
                    .cb_he
                    .is_some_and(|cb| p.c.is_some_and(|c| (c - cb).abs() <= cfg.tol_cb * 1.15))
        }
        "TRP_IND_N15HSQC" => clu
            .trp_indole
            .is_some_and(|(ne1, he1, _)| (p.h - he1).abs() <= tol_h && (p.n - ne1).abs() <= tol_n),
        "HIS_LR_N15HMQC" => {
            clu.ch_slots
                .values()
                .any(|s| (p.h - s.h_ppm).abs() <= tol_h)
                && (clu.nd1_ppm.is_some_and(|(n, _)| (p.n - n).abs() <= tol_n)
                    || clu.ne2_ppm.is_some_and(|(n, _)| (p.n - n).abs() <= tol_n))
        }
        _ => false,
    }
}

fn put_ring_atom_if_missing(
    merged: &mut Reference,
    k_cur: usize,
    res_name: &str,
    atom: &str,
    val: f64,
) {
    let key = (k_cur, atom.to_string());
    if !merged.shifts.contains_key(&key) {
        merged.shifts.insert(key.clone(), val);
        let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
        merged.ambiguity_codes.entry(key).or_insert(code);
    }
}

fn insert_corroborated_missing_ring_atoms(
    merged: &mut Reference,
    k_cur: usize,
    clu: &AromaticRingCluster,
) {
    let res_name = merged.sequence.get(k_cur).cloned().unwrap_or_default();
    let has_any = |r: &Reference, atoms: &[&str]| -> bool {
        atoms
            .iter()
            .any(|a| r.shifts.contains_key(&(k_cur, (*a).to_string())))
    };

    for (&c_atom, slot) in &clu.ch_slots {
        match (res_name.as_str(), c_atom) {
            ("PHE" | "TYR", "CD1") => {
                if !has_any(merged, &["CD1", "CD2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CD1", slot.c_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CD2", slot.c_ppm);
                }
                if !has_any(merged, &["HD1", "HD2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HD1", slot.h_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HD2", slot.h_ppm);
                }
            }
            ("PHE" | "TYR", "CE1") => {
                if !has_any(merged, &["CE1", "CE2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CE1", slot.c_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CE2", slot.c_ppm);
                }
                if !has_any(merged, &["HE1", "HE2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HE1", slot.h_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HE2", slot.h_ppm);
                }
            }
            ("PHE", "CZ") => {
                put_ring_atom_if_missing(merged, k_cur, &res_name, "CZ", slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, "HZ", slot.h_ppm);
            }
            ("TRP", ca) => {
                let pa = match ca {
                    "CD1" => "HD1",
                    "CE3" => "HE3",
                    "CZ3" => "HZ3",
                    "CH2" => "HH2",
                    "CZ2" => "HZ2",
                    _ => continue,
                };
                put_ring_atom_if_missing(merged, k_cur, &res_name, ca, slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, pa, slot.h_ppm);
            }
            ("HIS", ca) => {
                let pa = if ca == "CD2" { "HD2" } else { "HE1" };
                put_ring_atom_if_missing(merged, k_cur, &res_name, ca, slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, pa, slot.h_ppm);
            }
            _ => {}
        }
    }
    if res_name == "TRP" {
        if let Some((ne1, he1, _)) = clu.trp_indole {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "NE1", ne1);
            put_ring_atom_if_missing(merged, k_cur, &res_name, "HE1", he1);
        }
    }
    if res_name == "HIS" {
        if let Some((nd1, _)) = clu.nd1_ppm {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "ND1", nd1);
        }
        if let Some((ne2, _)) = clu.ne2_ppm {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "NE2", ne2);
        }
    }
    if !merged.shifts.contains_key(&(k_cur, "CB".to_string())) {
        if let Some(cb_y) = clu.cb_hd.or(clu.cb_he) {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "CB", cb_y);
        }
    }
}

#[allow(dead_code)]
fn merge_gui_aromatic_preserving_user_rings(
    working_ref: &Reference,
    prior_user_ref: Option<&Reference>,
    prior_amb: &BTreeMap<(usize, String), u8>,
    pred_conf: Option<&BTreeMap<usize, f64>>,
    arom_peaks: &[Peak],
    sc_priors: &SidechainPriors,
) -> Reference {
    merge_gui_aromatic_with_options(
        working_ref,
        prior_user_ref,
        prior_amb,
        pred_conf,
        arom_peaks,
        sc_priors,
        false,
    )
}

fn merge_gui_aromatic_with_options(
    working_ref: &Reference,
    prior_user_ref: Option<&Reference>,
    prior_amb: &BTreeMap<(usize, String), u8>,
    pred_conf: Option<&BTreeMap<usize, f64>>,
    arom_peaks: &[Peak],
    sc_priors: &SidechainPriors,
    overwrite_curated_rings: bool,
) -> Reference {
    const RING_ATOMS: &[&str] = &[
        "CG", "CD1", "CD2", "HD1", "HD2", "CE1", "CE2", "HE1", "HE2", "CZ", "HZ", "CE3", "HE3",
        "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2", "ND1", "NE1", "NE2",
    ];
    let is_arom_ring_key = |r: &Reference, k: usize, atom: &str| -> bool {
        matches!(
            r.sequence.get(k).map(String::as_str),
            Some("PHE" | "TYR" | "TRP" | "HIS")
        ) && RING_ATOMS.contains(&atom)
    };

    // Assemble full reference retaining all user-curated ring shifts & ambiguity codes
    let mut curated_ref = working_ref.clone();
    if let Some(user_ref) = prior_user_ref {
        for (&(k, ref atom), &v) in &user_ref.shifts {
            if is_arom_ring_key(user_ref, k, atom) {
                curated_ref.shifts.insert((k, atom.clone()), v);
            }
        }
        for (&(k, ref atom), &code) in &user_ref.ambiguity_codes {
            if is_arom_ring_key(user_ref, k, atom) {
                curated_ref.ambiguity_codes.insert((k, atom.clone()), code);
            }
        }
    }
    for (&(k, ref atom), &code) in prior_amb {
        if is_arom_ring_key(&curated_ref, k, atom) {
            curated_ref.ambiguity_codes.insert((k, atom.clone()), code);
        }
    }

    let cfg = AromaticSolverConfig::default();

    if overwrite_curated_rings {
        let mut backbone_for_arom = curated_ref.clone();
        backbone_for_arom
            .shifts
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&curated_ref, k, atom));
        backbone_for_arom
            .ambiguity_codes
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&curated_ref, k, atom));

        let arom_res =
            assign_aromatic_rings(&backbone_for_arom, pred_conf, arom_peaks, sc_priors, &cfg);
        let assigned_ring_residues: BTreeSet<usize> = arom_res
            .rings
            .iter()
            .filter_map(|r| r.cluster_idx.map(|_| r.residue_idx))
            .collect();
        let mut base_for_overwrite = curated_ref.clone();
        base_for_overwrite.shifts.retain(|&(k, ref atom), _| {
            !(assigned_ring_residues.contains(&k) && is_arom_ring_key(&curated_ref, k, atom))
        });
        base_for_overwrite
            .ambiguity_codes
            .retain(|&(k, ref atom), _| {
                !(assigned_ring_residues.contains(&k) && is_arom_ring_key(&curated_ref, k, atom))
            });
        let mut merged = merge_aromatic_assignments_preserving_existing(
            &base_for_overwrite,
            &arom_res.reference,
            &arom_res.atom_confidence,
        );
        for (k, atom) in arom_res.atom_confidence.keys() {
            let res_name = merged.sequence.get(*k).map(String::as_str).unwrap_or("");
            let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
            merged.ambiguity_codes.insert((*k, atom.clone()), code);
        }
        return merged;
    }

    // Default (`overwrite_curated_rings == false`):
    // 1. Do NOT strip user-curated ring shifts before conflict/corroboration checking.
    // 2. Identify residues with >= 1 user-curated aromatic ring shift.
    let mut curated_residues: BTreeSet<usize> = BTreeSet::new();
    for &(k, ref atom) in curated_ref.shifts.keys() {
        if is_arom_ring_key(&curated_ref, k, atom) {
            curated_residues.insert(k);
        }
    }

    let probe = assign_aromatic_rings(&curated_ref, pred_conf, arom_peaks, sc_priors, &cfg);
    if curated_residues.is_empty() {
        let mut merged = merge_aromatic_assignments_preserving_existing(
            &curated_ref,
            &probe.reference,
            &probe.atom_confidence,
        );
        for (k, atom) in probe.atom_confidence.keys() {
            let res_name = merged.sequence.get(*k).map(String::as_str).unwrap_or("");
            let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
            merged
                .ambiguity_codes
                .entry((*k, atom.clone()))
                .or_insert(code);
        }
        return merged;
    }

    // 3. Match each curated residue against compatible experimental clusters so:
    //    - partial disjoint clusters (e.g. CD-only vs curated CE/CZ) are never mixed into curated rings,
    //    - clusters whose shifts belong to a curated residue are claimed and cannot jump onto another residue.
    let mut claimed_clusters: BTreeMap<usize, (usize, ClusterCuratedEval)> = BTreeMap::new();
    for &k_cur in &curated_residues {
        let mut best: Option<(usize, ClusterCuratedEval)> = None;
        for (ci, clu) in probe.clusters.iter().enumerate() {
            if claimed_clusters.contains_key(&ci) {
                continue;
            }
            let ev = eval_cluster_vs_curated_residue(clu, k_cur, &curated_ref, &cfg);
            if ev.conflict_obs > 0 || ev.cb_contradicts {
                continue;
            }
            let min_corroborate = ev.total_cur_obs.clamp(1, 2);
            let corroborates = ev.shared_obs >= min_corroborate;
            let exact_subset_of_curated = ev.total_clu_obs > 0 && ev.shared_obs == ev.total_clu_obs;
            if !corroborates && !exact_subset_of_curated {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, b_ev)) => {
                    ev.shared_obs > b_ev.shared_obs
                        || (ev.shared_obs == b_ev.shared_obs && ev.norm_dist_sq < b_ev.norm_dist_sq)
                }
            };
            if better {
                best = Some((ci, ev));
            }
        }
        if let Some((ci, ev)) = best {
            claimed_clusters.insert(ci, (k_cur, ev));
        }
    }

    // 4. Solve uncurated residues on peaks excluding peaks owned solely by claimed curated clusters.
    let mut sub_ref = curated_ref.clone();
    for &k_cur in &curated_residues {
        if let Some(slot) = sub_ref.sequence.get_mut(k_cur) {
            *slot = "ALA".to_string();
        }
    }
    let sub_peaks: Vec<Peak> = arom_peaks
        .iter()
        .filter(|p| {
            let matches_claimed = claimed_clusters
                .keys()
                .any(|&ci| peak_matches_aromatic_cluster(p, &probe.clusters[ci], &cfg));
            if !matches_claimed {
                return true;
            }
            probe.clusters.iter().enumerate().any(|(cj, clu)| {
                !claimed_clusters.contains_key(&cj) && peak_matches_aromatic_cluster(p, clu, &cfg)
            })
        })
        .cloned()
        .collect();

    let sub_res = assign_aromatic_rings(&sub_ref, pred_conf, &sub_peaks, sc_priors, &cfg);

    let mut merged = curated_ref.clone();

    // Corroborated missing-atom completion on curated residues (never overwriting curated keys or mixing disjoint CD-only into CE/CZ)
    for (&ci, &(k_cur, ev)) in &claimed_clusters {
        let min_corroborate = ev.total_cur_obs.clamp(1, 2);
        if ev.conflict_obs == 0 && !ev.cb_contradicts && ev.shared_obs >= min_corroborate {
            insert_corroborated_missing_ring_atoms(&mut merged, k_cur, &probe.clusters[ci]);
        }
    }

    // Merge assignments onto uncurated residues while blocking duplicate curated ring clusters
    for r in &sub_res.rings {
        let k_other = r.residue_idx;
        if curated_residues.contains(&k_other) {
            continue;
        }
        let Some(sub_ci) = r.cluster_idx else {
            continue;
        };
        let sub_clu = &sub_res.clusters[sub_ci];

        let is_duplicate_of_curated = curated_residues.iter().any(|&k_cur| {
            if curated_ref.sequence.get(k_cur) != curated_ref.sequence.get(k_other) {
                return false;
            }
            let dup_ev = eval_cluster_vs_curated_residue(sub_clu, k_cur, &curated_ref, &cfg);
            if !(dup_ev.conflict_obs == 0
                && dup_ev.total_clu_obs > 0
                && dup_ev.shared_obs == dup_ev.total_clu_obs)
            {
                return false;
            }
            // Allow only if >= 2 distinct clusters in probe genuinely shared these degenerate shifts,
            // or if sub_clu carries a Yamazaki CB matching k_other's CB while contradicting k_cur's CB.
            let n_probe_matches = probe
                .clusters
                .iter()
                .filter(|c| {
                    let e = eval_cluster_vs_curated_residue(c, k_cur, &curated_ref, &cfg);
                    e.conflict_obs == 0 && e.total_clu_obs > 0 && e.shared_obs == e.total_clu_obs
                })
                .count();
            if n_probe_matches >= 2 {
                return false;
            }
            if let (Some(&cb_other), Some(&cb_cur)) = (
                curated_ref.shifts.get(&(k_other, "CB".to_string())),
                curated_ref.shifts.get(&(k_cur, "CB".to_string())),
            ) {
                let cbs: Vec<f64> = [sub_clu.cb_hd, sub_clu.cb_he]
                    .into_iter()
                    .flatten()
                    .collect();
                if !cbs.is_empty()
                    && cbs.iter().any(|&cb| (cb - cb_other).abs() <= cfg.tol_cb)
                    && cbs.iter().all(|&cb| (cb - cb_cur).abs() > cfg.tol_cb)
                {
                    return false;
                }
            }
            true
        });

        if is_duplicate_of_curated {
            continue;
        }

        for key in sub_res.atom_confidence.keys() {
            if key.0 != k_other {
                continue;
            }
            if let Some(&val) = sub_res.reference.shifts.get(key) {
                merged.shifts.entry(key.clone()).or_insert(val);
            }
            let res_name = merged
                .sequence
                .get(k_other)
                .map(String::as_str)
                .unwrap_or("");
            let default_code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, &key.1);
            let code = sub_res
                .reference
                .ambiguity_codes
                .get(key)
                .copied()
                .unwrap_or(default_code);
            merged.ambiguity_codes.entry(key.clone()).or_insert(code);
        }
    }

    merged
}

struct CoreOctetBackendAdapter;

impl CoreOctetBackendAdapter {
    fn encode_hex_pretty(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 3 + 64);
        out.push_str("# Binary ChemDraw CDX (VjCD0100) hex stream\n");
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 {
                if i % 24 == 0 {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            }
            out.push_str(&format!("{b:02X}"));
        }
        out
    }

    fn decode_hex_bytes(input: &str) -> Result<Vec<u8>, String> {
        let mut hex_chars = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }
            for ch in trimmed.chars() {
                if ch.is_ascii_hexdigit() {
                    hex_chars.push(ch);
                } else if !ch.is_whitespace() && ch != ',' && ch != ':' && ch != '-' {
                    return Err(format!(
                        "Unexpected non-hex character '{ch}' in CDX hex buffer"
                    ));
                }
            }
        }
        if hex_chars.len() % 2 != 0 {
            return Err("CDX hex stream must have an even number of hex digits".to_string());
        }
        let mut bytes = Vec::with_capacity(hex_chars.len() / 2);
        for pair in hex_chars.chunks_exact(2) {
            let s: String = pair.iter().collect();
            let v =
                u8::from_str_radix(&s, 16).map_err(|e| format!("Invalid hex byte '{s}': {e}"))?;
            bytes.push(v);
        }
        Ok(bytes)
    }

    fn mol_with_cip_to_scene(mut mol: Molecule) -> RenderScene {
        let _ = kekulize(&mut mol);
        let mut scene = RenderScene::from(&mol);
        let [cx, cy] = scene.atom_centroid();
        scene.translate(-cx, -cy);
        scene
    }
}

impl OctetBackendAdapter for CoreOctetBackendAdapter {
    fn clean_2d(&self, scene: &RenderScene) -> Option<RenderScene> {
        if scene.atoms.is_empty() {
            return Some(scene.clone());
        }
        let orig_centroid = scene.atom_centroid();
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        clean_2d(&mut mol);
        let mut cleaned = Self::mol_with_cip_to_scene(mol);
        cleaned.translate(orig_centroid[0], orig_centroid[1]);
        // Preserve non-molecular canvas annotations (arrows, brackets, orbitals, TLC, captions, embedded spectra)
        cleaned.reaction_arrows = scene.reaction_arrows.clone();
        cleaned.curved_arrows = scene.curved_arrows.clone();
        cleaned.brackets = scene.brackets.clone();
        cleaned.orbitals = scene.orbitals.clone();
        cleaned.tlc_plates = scene.tlc_plates.clone();
        cleaned.spectra_1d = scene.spectra_1d.clone();
        cleaned.texts = scene.texts.clone();
        Some(cleaned)
    }

    fn name_or_smiles_to_scene(&self, query: &str) -> Option<Result<RenderScene, String>> {
        let q = query.trim();
        if q.is_empty() {
            return Some(Err(
                "Enter a chemical name, SMILES, InChI, or peptide sequence".to_string(),
            ));
        }

        // 1. Reaction SMILES (`A.B>reag>C`)
        if q.contains(">>") || (q.matches('>').count() == 2 && !q.contains('<')) {
            if let Ok(mut doc) = parse_reaction_smiles(q) {
                let mut x_cursor = 0.0_f64;
                for m in &mut doc.molecules {
                    clean_2d(m);
                    for a in &mut m.atoms {
                        a.x += x_cursor * 1.54;
                    }
                    x_cursor += 5.5;
                }
                let mut scene = RenderScene::from(&doc);
                let [cx, cy] = scene.atom_centroid();
                scene.translate(-cx, -cy);
                return Some(Ok(scene));
            }
        }

        // 2. Standard InChI (`InChI=1S/...`)
        if q.starts_with("InChI=") {
            return Some(
                parse_inchi(q)
                    .map(|mut m| {
                        clean_2d(&mut m);
                        Self::mol_with_cip_to_scene(m)
                    })
                    .map_err(|e| format!("InChI parse error: {e}")),
            );
        }

        // 3. Biopolymer HELM / peptide 3-letter notation (`PEPTIDE1{...}` or `H-Ala-Pro-OH`)
        if q.starts_with("PEPTIDE") || q.starts_with("RNA") || q.starts_with("DNA") {
            if let Ok(mut m) = parse_biopolymer_sequence(q) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 4. Offline IUPAC / common chemical name (`aspirin`, `ibuprofen`, `bicyclo[2.2.1]hept-2-ene`, `(2S)-2-aminopropanoic acid`, ...)
        if let Ok(amol) = iupac_name_to_mol(q) {
            let smi = amol.to_smiles();
            if let Ok(mut m) = parse_smiles(&smi) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 5. SMILES (`CC(=O)Oc1ccccc1C(=O)O`)
        if let Ok(mut m) = parse_smiles(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }

        // 6. Condensed structural formula (`CH3CH2OH`, `(CH3)3COH`, `PhCOOH`) or peptide (`Ala-Pro-Tyr`)
        if let Ok(mut m) = parse_condensed_formula(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }
        if q.contains('-') {
            if let Ok(m) = build_peptide_2d(q) {
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        None
    }

    fn scene_to_name_and_smiles(&self, scene: &RenderScene) -> Option<(String, String)> {
        if scene.atoms.is_empty() {
            return Some((String::new(), String::new()));
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let smiles = if !scene.reaction_arrows.is_empty() {
            let doc = ChemDocument::from(scene);
            let rxn = write_reaction_smiles(&doc);
            if rxn.is_empty() {
                write_canonical_smiles(&mol)
            } else {
                rxn
            }
        } else {
            write_canonical_smiles(&mol)
        };
        let amol = AnalytMol::from(&mol);
        let name = mol_to_iupac_name(&amol).unwrap_or_else(|_| {
            compute_phys_chem_props(&amol)
                .map(|p| p.formula)
                .unwrap_or_else(|_| smiles.clone())
        });
        Some((name, smiles))
    }

    fn compute_properties(&self, scene: &RenderScene) -> Option<MolecularPropertiesSummary> {
        if scene.atoms.is_empty() {
            return Some(MolecularPropertiesSummary::default());
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let rings = perceive_sssr(&mol).len();
        let amol = AnalytMol::from(&mol);
        let props = compute_phys_chem_props(&amol).ok()?;
        let heavy_atoms = amol.atoms.iter().filter(|a| a.element != "H").count();
        Some(MolecularPropertiesSummary {
            formula_hill: props.formula,
            molecular_weight: props.molecular_weight,
            exact_mass: props.monoisotopic_mass,
            elemental_analysis: props.elemental_analysis_text,
            heavy_atoms,
            rings,
            h_bond_donors: props.h_bond_donors,
            h_bond_acceptors: props.h_bond_acceptors,
        })
    }

    fn predict_nmr_1h_13c(
        &self,
        scene: &RenderScene,
        _freq_mhz: f64,
    ) -> Option<(Vec<SpectrumSignal1D>, Vec<SpectrumSignal1D>)> {
        if scene.atoms.is_empty() {
            return Some((Vec::new(), Vec::new()));
        }
        let raw_mol = Molecule::from(scene);
        let has_abbrevs = raw_mol.atoms.iter().any(|a| a.superatom_label.is_some());
        let mol = if has_abbrevs {
            let mut m = raw_mol;
            let _ = expand_abbreviations(&mut m);
            m
        } else {
            raw_mol
        };
        let amol = AnalytMol::from(&mol);

        let pred_h = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
        let pred_c = predict_nmr(&amol, NmrNucleus::C13, Some("CDCl3"), None);

        let n_scene_atoms = scene.atoms.len();
        let sigs_1h = pred_h
            .signals
            .into_iter()
            .map(|s| {
                let mult = s.multiplicity();
                let mut couplings_hz = Vec::new();
                for c in &s.couplings {
                    for _ in 0..c.count.min(12) {
                        couplings_hz.push(c.j_hz);
                    }
                }
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: mult,
                    couplings_hz,
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}H", s.nuclei)),
                }
            })
            .collect();

        let sigs_13c = pred_c
            .signals
            .into_iter()
            .map(|s| {
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: "s".to_string(),
                    couplings_hz: Vec::new(),
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}C", s.nuclei)),
                }
            })
            .collect();

        Some((sigs_1h, sigs_13c))
    }

    fn export_custom(
        &self,
        scene: &RenderScene,
        style: &DocumentStyle,
        format: IoFormat,
    ) -> Option<Result<String, String>> {
        match format {
            IoFormat::Cdxml => {
                let doc = ChemDocument::from(scene);
                Some(Ok(write_cdxml(&doc)))
            }
            IoFormat::CdxHex => {
                let doc = ChemDocument::from(scene);
                let bytes = write_cdx(&doc);
                Some(Ok(Self::encode_hex_pretty(&bytes)))
            }
            IoFormat::OctJson => {
                let doc = ChemDocument::from(scene);
                Some(write_oct(&doc).map_err(|e| format!("OCT JSON export error: {e}")))
            }
            IoFormat::MolV2000 => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                Some(write_mol_v2000(&mol).map_err(|e| format!("MOL V2000 export error: {e}")))
            }
            IoFormat::Sdf => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let mut props = BTreeMap::new();
                if let Ok(pin) = mol_to_iupac_name(&amol) {
                    props.insert("IUPAC_NAME".to_string(), pin);
                }
                props.insert("SMILES".to_string(), write_canonical_smiles(&mol));
                props.insert("INCHIKEY".to_string(), write_inchikey(&mol));
                if let Ok(pc) = compute_phys_chem_props(&amol) {
                    props.insert("FORMULA".to_string(), pc.formula);
                    props.insert(
                        "EXACT_MASS".to_string(),
                        format!("{:.5}", pc.monoisotopic_mass),
                    );
                }
                Some(Ok(write_sdf(&[SdfRecord {
                    molecule: mol,
                    properties: props,
                }])))
            }
            IoFormat::Smiles => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                if !scene.reaction_arrows.is_empty() {
                    let doc = ChemDocument::from(scene);
                    let rxn = write_reaction_smiles(&doc);
                    if !rxn.is_empty() {
                        return Some(Ok(rxn));
                    }
                }
                Some(Ok(write_canonical_smiles(&mol)))
            }
            IoFormat::InChI => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let inchi = write_inchi(&mol);
                let key = write_inchikey(&mol);
                Some(Ok(format!("{inchi}\nInChIKey={key}")))
            }
            IoFormat::JcampDx => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let pred = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
                let spec = synthesize_1d_nmr(&pred, &Synthesize1dOptions::default());
                Some(Ok(write_jcamp_nmr(&spec, JcampCompression::Affn)))
            }
            IoFormat::Svg => Some(Ok(render_scene_svg(scene, style))),
            IoFormat::Pdf | IoFormat::Png => None,
        }
    }

    fn import_custom(&self, format: IoFormat, input: &str) -> Option<Result<RenderScene, String>> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Some(Err("Input buffer is empty".to_string()));
        }
        match format {
            IoFormat::Cdxml => Some(
                read_cdxml(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("CDXML import error: {e}")),
            ),
            IoFormat::CdxHex => Some(
                Self::decode_hex_bytes(trimmed)
                    .and_then(|bytes| {
                        read_cdx(&bytes).map_err(|e| format!("CDX binary error: {e}"))
                    })
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    }),
            ),
            IoFormat::OctJson => Some(
                read_oct(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("OCT JSON import error: {e}")),
            ),
            IoFormat::MolV2000 => Some(
                read_mol(trimmed)
                    .map(Self::mol_with_cip_to_scene)
                    .map_err(|e| format!("MOL import error: {e}")),
            ),
            IoFormat::Sdf => Some(
                read_sdf(trimmed)
                    .map_err(|e| format!("SDF import error: {e}"))
                    .and_then(|recs| {
                        recs.into_iter()
                            .next()
                            .map(|r| Self::mol_with_cip_to_scene(r.molecule))
                            .ok_or_else(|| "SDF contains no molecule records".to_string())
                    }),
            ),
            IoFormat::Smiles => self.name_or_smiles_to_scene(
                trimmed
                    .lines()
                    .next()
                    .unwrap_or(trimmed)
                    .split_whitespace()
                    .next()
                    .unwrap_or(trimmed),
            ),
            IoFormat::InChI => {
                let inchi_line = trimmed
                    .lines()
                    .map(str::trim)
                    .find(|l| l.starts_with("InChI="))
                    .unwrap_or(trimmed);
                Some(
                    parse_inchi(inchi_line)
                        .map(|mut m| {
                            clean_2d(&mut m);
                            Self::mol_with_cip_to_scene(m)
                        })
                        .map_err(|e| format!("InChI import error: {e}")),
                )
            }
            IoFormat::JcampDx => {
                let blocks = match read_jcamp(trimmed) {
                    Ok(b) if !b.is_empty() => b,
                    Ok(_) => return Some(Err("No spectrum blocks found in JCAMP-DX".to_string())),
                    Err(e) => return Some(Err(format!("JCAMP-DX import error: {e}"))),
                };
                let sp = &blocks[0];
                let picked = pick_1d_multiplets(sp, 0.04, 22.0, None);
                let signals: Vec<SpectrumSignal1D> = picked
                    .into_iter()
                    .map(|m| SpectrumSignal1D {
                        ppm: m.center_ppm,
                        multiplicity: m.estimated_multiplicity,
                        couplings_hz: m.estimated_j_hz.into_iter().collect(),
                        integral: m.normalized_integral.max(0.5),
                        atom_indices: Vec::new(),
                        assignment_label: None,
                    })
                    .collect();
                let n_pts = sp.y.len();
                let mut custom_trace = Vec::with_capacity(n_pts.min(2048));
                let max_y =
                    sp.y.iter()
                        .copied()
                        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
                        .max(1e-9);
                let step = (n_pts / 2048).max(1);
                for idx in (0..n_pts).step_by(step) {
                    custom_trace.push([sp.x_at(idx), sp.y[idx] / max_y]);
                }
                let ppm_min = sp.first_x.min(sp.last_x);
                let ppm_max = sp.first_x.max(sp.last_x);
                let mut scene = RenderScene::new();
                scene.spectra_1d.push(SceneSpectrum1D {
                    origin: [-5.5, -2.0],
                    width_mol: 11.0,
                    height_mol: 4.0,
                    nucleus: sp.nucleus.clone().unwrap_or_else(|| "1H".to_string()),
                    spectrometer_mhz: sp.mhz.unwrap_or(400.0).max(60.0),
                    ppm_min: if ppm_min.is_finite() { ppm_min } else { -0.5 },
                    ppm_max: if ppm_max.is_finite() { ppm_max } else { 11.0 },
                    linewidth_hz: 1.4,
                    show_integrals: !signals.is_empty(),
                    show_peak_labels: true,
                    title: Some(if sp.title.is_empty() {
                        "Imported JCAMP-DX 1D NMR".to_string()
                    } else {
                        sp.title.clone()
                    }),
                    signals,
                    custom_trace,
                });
                Some(Ok(scene))
            }
            _ => None,
        }
    }
}

struct NmrWorkbenchApp {
    workspace_mode: WorkspaceMode,
    octet_app: OctetDrawPanel,
    #[cfg(not(target_arch = "wasm32"))]
    gpu_engine: Option<GpuEngine>,
    adapter_label: Option<String>,
    solver_kind: GuiSolverKind,
    regime_kind: GuiRegimeKind,
    metal_ligands_input: String,
    idp_temp_k: f64,
    idp_ph_input: String,
    c13_picked_dataset: Option<C13PickedDataset>,
    propensity_report: Option<PropensityReport>,
    show_propensity_panel: bool,
    #[cfg(not(target_arch = "wasm32"))]
    use_gpu: bool,
    use_neural_picker: bool,
    status_msg: String,
    last_solve_ms: Option<f64>,

    // Active dataset
    priors: Priors,
    sidechain_priors: SidechainPriors,
    reference: Option<Reference>,
    sequence: Vec<String>,
    seq_ids: Vec<i32>,
    peaks: Vec<Peak>,
    truth: Option<Vec<i32>>,
    systems: Vec<SpinSystem>,
    u_mat: Option<Matrix2D>,
    p_mat: Option<Matrix2D>,
    assign_res: Option<AssignResult>,
    pred_ref: Option<Reference>,
    pred_conf: BTreeMap<usize, f64>,
    pred_ambiguity_codes: BTreeMap<(usize, String), u8>,
    metrics: Option<Metrics>,
    sidechain_ribbon_state: SequenceRibbonState,
    show_sidechain_inspector: bool,
    overwrite_curated_rings: bool,
    mol3d_panel: Mol3dViewerPanel,
    show_mol3d_viewport: bool,

    // Spectra for GPU contour & 3D strip visualization
    hsqc_spec: Option<Spectrum2D>,
    hsqc_arc: Arc<Vec<f32>>,
    hsqc_gen: u64,
    hncacb_3d: Option<Spectrum3D>,
    hnco_3d: Option<Spectrum3D>,
    extra_3d_count: usize,
    overlay_layers: Vec<(String, OverlayLayerSpec)>,
    primary_ppm_nudge: [f32; 2],
    next_layer_id: u64,
    active_overlay_idx: usize,

    // Phase 9 UX, Session, Overlays & Match-Finder State
    ingest_panel: IngestPanelState,
    command_palette: CommandPalette,
    chord_dispatcher: ChordDispatcher,
    session: WorkbenchSessionController,
    crosshair: SyncCrosshair,
    strip_tiler: StripTilerPanel,
    use_match_finder_strips: bool,
    qc_report: Option<DatasetQcReport>,
    raw_fids_total: usize,
    unprocessed_fids_remaining: usize,
    has_structure_or_dynamics: bool,
    exported_in_session: bool,
    show_contour_modal: bool,
    show_overlay_manager: bool,
    show_assign_popover: bool,
    show_qc_modal: bool,
    show_about_modal: bool,
    assign_target_res: usize,
    assign_lock_check: bool,

    // 2D HSQC viewport & contour settings
    view_h_left: f32,
    view_h_right: f32,
    view_n_top: f32,
    view_n_bottom: f32,
    contour_base: f32,
    contour_factor: f32,
    contour_levels: u32,
    show_negative: bool,
    show_peak_labels: bool,

    // Solver & simulator controls
    replicas: usize,
    steps_per_sys: usize,
    assign_cost: f64,
    sim_missing: f64,
    sim_spurious: f64,
    sim_seed: u64,
    picker_snr: f32,

    // Selection & File / In-Memory Text I/O
    selected_ss: Option<usize>,
    selected_res: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    path_input: String,
    #[cfg(not(target_arch = "wasm32"))]
    export_prefix: String,
    show_text_io_modal: bool,
    paste_text_buf: String,
    export_text_buf: String,
    #[cfg(target_arch = "wasm32")]
    pending_drop_outcome: Arc<std::sync::Mutex<Option<Result<IngestDropOutcome, String>>>>,
}

impl NmrWorkbenchApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = egui::Color32::from_rgb(16, 20, 28);
        style.visuals.window_fill = egui::Color32::from_rgb(20, 25, 35);
        cc.egui_ctx.set_style_of(egui::Theme::Dark, style);

        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_engine = None;
        let mut adapter_label = None;

        if let Some(rs) = &cc.wgpu_render_state {
            let contour_res = ContourPipelineResource::new(&rs.device, rs.target_format);
            let mol3d_res = Mol3dPipelineResource::new(&rs.device, Some(rs.target_format));
            {
                let mut rend = rs.renderer.write();
                rend.callback_resources.insert(contour_res);
                rend.callback_resources.insert(mol3d_res);
            }

            let info = rs.adapter.get_info();
            let name = if info.name.trim().is_empty() {
                "WebGPU Adapter".to_string()
            } else {
                info.name.clone()
            };
            adapter_label = Some(format!("{} ({:?})", name, info.backend));

            #[cfg(not(target_arch = "wasm32"))]
            {
                gpu_engine = Some(GpuEngine::from_device(
                    rs.device.clone(),
                    rs.queue.clone(),
                    info.name.clone(),
                    format!("{:?}", info.backend),
                ));
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let default_replicas = 64;
        #[cfg(target_arch = "wasm32")]
        let default_replicas = 16;

        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));

        let ingest_panel = IngestPanelState {
            open: false,
            ..Default::default()
        };

        let mut app = Self {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            solver_kind: GuiSolverKind::default(),
            regime_kind: GuiRegimeKind::default(),
            metal_ligands_input: String::new(),
            idp_temp_k: 298.15,
            idp_ph_input: "6.5".to_string(),
            c13_picked_dataset: None,
            propensity_report: None,
            show_propensity_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: gpu_engine.is_some(),
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine,
            use_neural_picker: true,
            adapter_label,
            status_msg: String::new(),
            last_solve_ms: None,
            priors: Priors::provisional(),
            sidechain_priors: SidechainPriors::empirical_bmrb(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            pred_ambiguity_codes: BTreeMap::new(),
            metrics: None,
            sidechain_ribbon_state: SequenceRibbonState::new(),
            show_sidechain_inspector: false,
            overwrite_curated_rings: false,
            mol3d_panel: Mol3dViewerPanel::new(),
            show_mol3d_viewport: true,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            hnco_3d: None,
            extra_3d_count: 0,
            overlay_layers: Vec::new(),
            primary_ppm_nudge: [0.0, 0.0],
            next_layer_id: 2,
            active_overlay_idx: 0,
            ingest_panel,
            command_palette: CommandPalette::new(),
            chord_dispatcher: ChordDispatcher::new(),
            session: WorkbenchSessionController::default(),
            crosshair: SyncCrosshair::default(),
            strip_tiler: StripTilerPanel::new(),
            use_match_finder_strips: true,
            qc_report: None,
            raw_fids_total: 0,
            unprocessed_fids_remaining: 0,
            has_structure_or_dynamics: false,
            exported_in_session: false,
            show_contour_modal: false,
            show_overlay_manager: false,
            show_assign_popover: false,
            show_qc_modal: false,
            show_about_modal: false,
            assign_target_res: 0,
            assign_lock_check: true,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: default_replicas,
            steps_per_sys: 1500,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            #[cfg(target_arch = "wasm32")]
            pending_drop_outcome: Arc::new(std::sync::Mutex::new(None)),
        };

        app.load_demo_and_solve();
        app
    }

    fn reset_view_to_spec(&mut self) {
        if let Some(spec) = &self.hsqc_spec {
            self.view_h_left = spec.axis_x.max_ppm() as f32;
            self.view_h_right = spec.axis_x.min_ppm() as f32;
            self.view_n_top = spec.axis_y.min_ppm() as f32;
            self.view_n_bottom = spec.axis_y.max_ppm() as f32;
            let noise = estimate_noise_mad(&spec.data);
            self.contour_base = (noise * 4.8).max(1.0);
        }
    }

    fn compute_matrices(&self, sys: &[SpinSystem], scfg: &ScoreConfig) -> (Matrix2D, Matrix2D) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu {
            if let Some(gpu) = &self.gpu_engine {
                if let Ok(pair) = gpu.score_matrices_gpu(sys, &self.sequence, &self.priors, scfg) {
                    return pair;
                }
            }
        }
        (
            unary_matrix(sys, &self.sequence, &self.priors, scfg),
            pair_matrix(sys, scfg),
        )
    }

    fn rebuild_systems_and_matrices(&mut self) {
        match build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
            Ok(sys) => {
                let locked: Vec<(usize, usize)> = self
                    .session
                    .state
                    .locked_assignments_vec()
                    .into_iter()
                    .filter(|&(s, k)| s < sys.len() && k < self.sequence.len())
                    .collect();
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: locked,
                    ..ScoreConfig::default()
                };
                let (u, p) = self.compute_matrices(&sys, &scfg);
                self.u_mat = Some(u);
                self.p_mat = Some(p);
                self.systems = sys;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
            }
            Err(e) => {
                self.status_msg = format!("Spin system build error: {e}");
            }
        }
    }

    fn regenerate_synthetic_spectra(&mut self) {
        let rcfg = SpectrumRenderConfig {
            nx_h: 512,
            ny_n: 256,
            nz_c: 96,
            seed: self.sim_seed.wrapping_add(7),
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&self.peaks, &rcfg);
        self.hsqc_arc = Arc::new(hsqc.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.hsqc_spec = Some(hsqc);
        self.hncacb_3d = Some(render_triple_3d("HNCACB", &self.peaks, &rcfg));
        self.hnco_3d = Some(render_triple_3d("HNCO", &self.peaks, &rcfg));
        self.extra_3d_count = 0;
        self.overlay_layers.clear();
        self.primary_ppm_nudge = [0.0, 0.0];
        self.reset_view_to_spec();
    }

    fn sync_session_from_app_fresh(&mut self, protein_name: &str) {
        let mut st = WorkbenchSessionState::new(protein_name, self.sequence.clone());
        if self.seq_ids.len() == self.sequence.len() {
            st.seq_ids = self.seq_ids.clone();
        }
        st.peaks = self.peaks.clone();
        let mut hsqc_layer = SessionContourParams::new("HSQC");
        hsqc_layer.base_level = self.contour_base;
        hsqc_layer.level_factor = self.contour_factor;
        hsqc_layer.num_levels = self.contour_levels;
        hsqc_layer.show_negative = self.show_negative;
        hsqc_layer.ppm_nudge = self.primary_ppm_nudge;
        st.contour_layers = vec![hsqc_layer];
        self.session = WorkbenchSessionController::new(st);
    }

    fn sync_session_assignments_from_solver(&mut self) {
        self.session.state.sequence = self.sequence.clone();
        if self.seq_ids.len() == self.sequence.len() {
            self.session.state.seq_ids = self.seq_ids.clone();
        }
        self.session.state.peaks = self.peaks.clone();
        self.session.state.qc_verdict = self.qc_report.as_ref().map(|q| q.verdict);

        let mut new_assign: BTreeMap<usize, ResidueAssignmentState> = BTreeMap::new();
        if let Some(pred) = &self.pred_ref {
            for (&(res_idx, ref atom), &val) in &pred.shifts {
                if res_idx < self.sequence.len() {
                    let entry =
                        new_assign
                            .entry(res_idx)
                            .or_insert_with(|| ResidueAssignmentState {
                                spin_system: None,
                                shifts: BTreeMap::new(),
                                confidence: self.pred_conf.get(&res_idx).copied().unwrap_or(0.0),
                            });
                    entry.shifts.insert(atom.clone(), val);
                }
            }
        }
        if let Some(res) = &self.assign_res {
            for (ss, &pos) in res.position.iter().enumerate() {
                if pos >= 0 && (pos as usize) < self.sequence.len() {
                    let k = pos as usize;
                    let c = res.confidence.get(ss).copied().unwrap_or(0.0);
                    let entry = new_assign
                        .entry(k)
                        .or_insert_with(|| ResidueAssignmentState {
                            spin_system: Some(ss),
                            shifts: BTreeMap::new(),
                            confidence: c,
                        });
                    entry.spin_system = Some(ss);
                    entry.confidence = c;
                }
            }
        }
        // Keep lock-assignment coherence invariant satisfied
        self.session.state.locked_residues.retain(|&ss, k| {
            new_assign
                .get(k)
                .is_some_and(|st| st.spin_system == Some(ss))
        });
        self.session.state.assignments = new_assign;
        self.sync_mol3d_from_app();
    }

    fn sync_app_from_session(&mut self) {
        self.sequence = self.session.state.sequence.clone();
        self.seq_ids = if self.session.state.seq_ids.len() == self.sequence.len() {
            self.session.state.seq_ids.clone()
        } else {
            (1..=(self.sequence.len() as i32)).collect()
        };
        self.octet_app
            .set_loaded_protein_sequence(self.sequence.clone());

        if self.peaks != self.session.state.peaks {
            self.peaks = self.session.state.peaks.clone();
            self.rebuild_systems_and_matrices();
        } else if !self.sequence.is_empty() && !self.systems.is_empty() {
            let locked = self.session.state.locked_assignments_vec();
            let scfg = ScoreConfig {
                assign_cost: self.assign_cost,
                locked_assignments: locked,
                ..ScoreConfig::default()
            };
            self.u_mat = Some(unary_matrix(
                &self.systems,
                &self.sequence,
                &self.priors,
                &scfg,
            ));
        }

        if let Some(hsqc_layer) = self
            .session
            .state
            .contour_layers
            .iter()
            .find(|l| l.layer_id == "HSQC")
        {
            self.contour_base = hsqc_layer.base_level.max(1e-4);
            self.contour_factor = hsqc_layer.level_factor.max(1.05);
            self.contour_levels = hsqc_layer.num_levels.clamp(1, 24);
            self.show_negative = hsqc_layer.show_negative;
            self.primary_ppm_nudge = hsqc_layer.ppm_nudge;
        }

        let n_sys = self.systems.len();
        let mut position = vec![-1i32; n_sys];
        let mut confidence = vec![0.0f64; n_sys];
        for (&res_idx, st) in &self.session.state.assignments {
            if let Some(ss) = st.spin_system {
                if ss < n_sys && res_idx < self.sequence.len() {
                    position[ss] = res_idx as i32;
                    confidence[ss] = st.confidence;
                }
            }
        }
        let (mut pred, mut conf) = self.session.state.to_reference_and_confidences();
        if n_sys > 0 {
            let synth_res = AssignResult {
                position: position.clone(),
                score: self.assign_res.as_ref().map_or(0.0, |r| r.score),
                confidence: confidence.clone(),
                run_scores: Vec::new(),
            };
            let (derived_ref, derived_conf) = to_reference(
                &synth_res,
                &self.systems,
                &self.sequence,
                &self.seq_ids,
                0.0,
                Some(&self.priors),
            );
            for (key, val) in derived_ref.shifts {
                pred.shifts.entry(key).or_insert(val);
            }
            for (k, c) in derived_conf {
                conf.entry(k).or_insert(c);
            }
            self.assign_res = Some(synth_res);
        } else {
            self.assign_res = None;
        }
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.refresh_propensity_report();
        self.sync_mol3d_from_app();
    }

    fn sync_mol3d_from_app(&mut self) {
        let src_ref = self.pred_ref.as_ref().or(self.reference.as_ref());
        self.mol3d_panel
            .sync_nmr_tracks(&self.pred_conf, src_ref, Some(&self.priors));
        self.mol3d_panel
            .highlight_sequence_residue(self.selected_res, false);
        self.mol3d_panel.flush_pending_updates();
    }

    fn load_structure_3d_in_memory(&mut self, text: &str, source_label: &str) {
        match self
            .mol3d_panel
            .load_structure_text(text, &self.sequence, &self.seq_ids)
        {
            Ok(()) => {
                if self.sequence.is_empty() {
                    if let Some(ens) = self.mol3d_panel.ensemble.as_ref() {
                        if !ens.primary_sequence.is_empty() {
                            self.sequence = ens.primary_sequence.clone();
                            self.seq_ids = ens.primary_seq_ids.clone();
                            self.octet_app
                                .set_loaded_protein_sequence(self.sequence.clone());
                            self.session.state.sequence = self.sequence.clone();
                            self.session.state.seq_ids = self.seq_ids.clone();
                        }
                    }
                }
                self.show_mol3d_viewport = true;
                self.has_structure_or_dynamics = true;
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.sync_mol3d_from_app();
                self.status_msg = format!(
                    "Loaded 3D structure from {source_label}: {}",
                    self.mol3d_panel.status_banner
                );
            }
            Err(e) => {
                self.status_msg = format!("3D structure load error ({source_label}): {e}");
            }
        }
    }

    fn apply_mol3d_sync_action(&mut self, sync: Mol3dSyncAction) {
        if sync.residue_idx < self.sequence.len() {
            self.select_residue_and_sync_strip(sync.residue_idx);
        }
        if sync.open_sidechain_graph {
            self.show_sidechain_inspector = true;
        }
        sync.apply_to_crosshair(&mut self.crosshair);
        let h_ppm = sync.atom_jump.as_ref().and_then(|j| j.backbone_h_ppm);
        let n_ppm = sync.atom_jump.as_ref().and_then(|j| j.backbone_n_ppm);
        if let (Some(h), Some(n)) = (h_ppm, n_ppm) {
            let h_f = h as f32;
            let n_f = n as f32;
            let h_min = self.view_h_left.min(self.view_h_right);
            let h_max = self.view_h_left.max(self.view_h_right);
            let n_min = self.view_n_top.min(self.view_n_bottom);
            let n_max = self.view_n_top.max(self.view_n_bottom);
            if h_f < h_min || h_f > h_max || n_f < n_min || n_f > n_max {
                let half_h = 0.5 * (self.view_h_left - self.view_h_right).abs().max(0.8);
                let half_n = 0.5 * (self.view_n_bottom - self.view_n_top).abs().max(6.0);
                self.view_h_left = h_f + half_h;
                self.view_h_right = h_f - half_h;
                self.view_n_top = n_f - half_n;
                self.view_n_bottom = n_f + half_n;
            }
        }
        let shift_summary = if let Some(ref j) = sync.atom_jump {
            match (j.backbone_h_ppm, j.backbone_n_ppm) {
                (Some(h), Some(n)) => {
                    format!(" ({} {:.3} ppm; ¹H {h:.3}, ¹⁵N {n:.2})", j.atom_name, j.ppm)
                }
                _ => format!(" ({} {:.3} ppm)", j.atom_name, j.ppm),
            }
        } else {
            String::new()
        };
        let atom_lbl = if sync.clicked_atom.is_empty() {
            "CA"
        } else {
            sync.clicked_atom.as_str()
        };
        self.status_msg = format!(
            "3D Pick → {} {} {}{}",
            sync.residue_name, sync.seq_id, atom_lbl, shift_summary
        );
    }

    fn refresh_propensity_report(&mut self) {
        let Some(src_ref) = self.pred_ref.as_ref().or(self.reference.as_ref()) else {
            self.propensity_report = None;
            return;
        };
        if src_ref.sequence.is_empty() {
            self.propensity_report = None;
            return;
        }
        let temp_k = self.idp_temp_k.clamp(250.0, 360.0);
        let mut cfg = PropensityConfig::default().with_temperature_kelvin(temp_k);
        if let Ok(ph) = self.idp_ph_input.trim().parse::<f64>() {
            if (1.0..=12.0).contains(&ph) {
                cfg = cfg.with_sample_ph(ph);
            }
        }
        self.propensity_report = Some(compute_idp_propensity(src_ref, &cfg));
    }

    fn raw_hsqc_locks_for_peaks(&self) -> Vec<(usize, usize)> {
        if self.session.state.locked_residues.is_empty() {
            return Vec::new();
        }
        let hsqc_roots: Vec<(f64, f64)> = self
            .peaks
            .iter()
            .filter(|p| p.exp == "HSQC")
            .map(|p| (p.h, p.n))
            .collect();
        let mut out = Vec::new();
        for (&ss_idx, &res_idx) in &self.session.state.locked_residues {
            if let Some(ss) = self.systems.get(ss_idx) {
                if let Some((raw_i, _)) = hsqc_roots
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(rh, rn))| {
                        (rh - ss.h).abs() <= 0.045 && (rn - ss.n).abs() <= 0.45
                    })
                    .min_by(|&(_, &(ah, an)), &(_, &(bh, bn))| {
                        let da = ((ah - ss.h) / 0.03).powi(2) + ((an - ss.n) / 0.3).powi(2);
                        let db = ((bh - ss.h) / 0.03).powi(2) + ((bn - ss.n) / 0.3).powi(2);
                        da.total_cmp(&db)
                    })
                {
                    out.push((raw_i, res_idx));
                }
            } else if ss_idx < hsqc_roots.len() {
                out.push((ss_idx, res_idx));
            }
        }
        out
    }

    fn load_demo_and_solve(&mut self) {
        let ref_data = build_synthetic_ubiquitin();
        self.load_from_reference(ref_data);
        let _ = self.mol3d_panel.load_structure_text(
            DEMO_UBIQUITIN_1UBQ_PDB,
            &self.sequence,
            &self.seq_ids,
        );
        self.has_structure_or_dynamics = true;
        self.run_assignment();
    }

    fn load_from_reference(&mut self, ref_data: Reference) {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: self.sim_missing,
                spurious_rate: self.sim_spurious,
                seed: self.sim_seed,
                ..SimConfig::default()
            },
        );
        self.sequence = ref_data.sequence.clone();
        self.seq_ids = ref_data.seq_ids.clone();
        self.octet_app
            .set_loaded_protein_sequence(self.sequence.clone());
        self.sidechain_ribbon_state = SequenceRibbonState::new();
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&ref_data);
        self.reference = Some(ref_data);
        self.c13_picked_dataset = None;
        self.peaks = peaks;
        self.truth = Some(truth);
        self.assign_res = None;
        self.pred_ref = None;
        self.pred_conf.clear();
        self.pred_ambiguity_codes.clear();
        self.metrics = None;
        self.qc_report = None;
        self.raw_fids_total = 0;
        self.unprocessed_fids_remaining = 0;
        self.has_structure_or_dynamics = false;
        self.exported_in_session = false;
        self.regenerate_synthetic_spectra();
        self.sync_session_from_app_fresh("Ubiquitin");
        self.rebuild_systems_and_matrices();
        self.refresh_propensity_report();
        self.status_msg = format!(
            "Simulated {} peaks & {} spin systems ({} aa) + 2D HSQC / 3D HNCACB & HNCO spectra",
            self.peaks.len(),
            self.systems.len(),
            self.sequence.len()
        );
    }

    fn run_idp_13c_assignment(&mut self) {
        if self.sequence.is_empty() {
            self.status_msg =
                "Load an IDP sequence or NMR-STAR entry before 13C-IDP assignment".to_string();
            return;
        }
        if self.c13_picked_dataset.is_none() {
            if let Some(ref r) = self.reference {
                let ph_val = self
                    .idp_ph_input
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|p| (1.0..=12.0).contains(p))
                    .unwrap_or(6.5);
                let cubes = simulate_idp_13c_cubes(
                    r,
                    &IdpCubeSimConfig {
                        temperature_k: self.idp_temp_k.clamp(250.0, 360.0),
                        ph: ph_val,
                        con_grid: (512, 512),
                        caco_grid: (256, 256),
                        cube_3d_grid: (128, 128, 128),
                        seed: self.sim_seed.wrapping_add(42),
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
                self.hsqc_arc = Arc::new(cubes.con_2d.data.clone());
                self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                self.hsqc_spec = Some(cubes.con_2d);
                self.reset_view_to_spec();
                self.c13_picked_dataset = Some(C13PickedDataset {
                    con_roots,
                    caco_peaks,
                    strip_peaks,
                });
            }
        }

        let Some(ref ds) = self.c13_picked_dataset else {
            self.status_msg =
                "Load a 13C-detected peak TSV (CON/CACO/CANCO/CBCACON) or NMR-STAR entry first"
                    .to_string();
            return;
        };
        let systems = register_and_build_con_dipeptide_systems(ds);
        if systems.is_empty() {
            self.status_msg =
                "No CON dipeptide spin systems could be built from 13C-detected peaks".to_string();
            return;
        }

        let timer = PerfTimer::start();
        let ph_val = self
            .idp_ph_input
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|p| (1.0..=12.0).contains(p))
            .unwrap_or(6.5);
        #[allow(unused_mut)]
        let mut idp_cfg = IdpSolverConfig {
            temperature_k: self.idp_temp_k.clamp(250.0, 360.0),
            ph: ph_val,
            replicas: self.replicas.max(8),
            steps_per_system: self.steps_per_sys.max(400),
            seed: self.sim_seed,
            ..IdpSolverConfig::default()
        };
        #[cfg(target_arch = "wasm32")]
        {
            idp_cfg.replicas = idp_cfg.replicas.min(16);
            idp_cfg.generations = 2;
            idp_cfg.steps_per_system = idp_cfg.steps_per_system.min(1000);
        }

        #[allow(unused_mut)]
        let mut gpu_out: Option<(nmr_core::idp_solver::IdpAssignResult, &'static str)> = None;
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu {
            if let Some(ref gpu) = self.gpu_engine {
                match nmr_gpu::idp_gpu::assign_idp_13c_gpu(gpu, &systems, &self.sequence, &idp_cfg)
                {
                    Ok(res) => gpu_out = Some((res, "13C-IDP WebGPU")),
                    Err(e) => {
                        self.status_msg =
                            format!("13C-IDP WebGPU error ({e}), falling back to CPU");
                    }
                }
            }
        }

        let default_tag = if cfg!(target_arch = "wasm32") {
            "13C-IDP WASM"
        } else {
            "13C-IDP CPU"
        };
        let (idp_res, tag) = gpu_out.unwrap_or_else(|| {
            (
                assign_idp_13c_cpu(&systems, &self.sequence, &idp_cfg),
                default_tag,
            )
        });
        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = idp_to_reference(
            &idp_res,
            &systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            false,
        );
        let n_pro = self
            .sequence
            .iter()
            .enumerate()
            .filter(|&(k, r)| {
                r == "PRO"
                    && (pred.shifts.contains_key(&(k, "N".to_string()))
                        || pred.shifts.contains_key(&(k, "CA".to_string())))
            })
            .count();
        let n_pro_tot = self.sequence.iter().filter(|r| r.as_str() == "PRO").count();
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();

        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(idp_res.as_assign_result());
        self.metrics = None;
        self.refresh_propensity_report();
        self.show_propensity_panel = true;
        self.sync_session_assignments_from_solver();
        self.status_msg = format!(
            "[{tag}] Assigned {}/{} residues ({n_hc} >=0.8, {n_pro}/{n_pro_tot} Pro) from {} CON systems in {dt_ms:.0} ms",
            self.pred_conf.len(),
            self.sequence.len(),
            systems.len()
        );
    }

    fn run_paramagnetic_assignment(&mut self) {
        if self.sequence.is_empty()
            || (self.solver_kind == GuiSolverKind::EvoGraph && self.peaks.is_empty())
            || (self.solver_kind == GuiSolverKind::Standard && self.systems.is_empty())
        {
            self.status_msg =
                "Need peaks/spin systems and sequence to run paramagnetic assignment".to_string();
            return;
        }
        let explicit_ligands = if self.metal_ligands_input.trim().is_empty() {
            Vec::new()
        } else {
            match parse_metal_ligands(
                self.metal_ligands_input.trim(),
                &self.sequence,
                &self.seq_ids,
            ) {
                Ok(idxs) => idxs,
                Err(e) => {
                    self.status_msg = format!("Invalid metal ligands: {e}");
                    return;
                }
            }
        };

        let timer = PerfTimer::start();
        let scfg = ScoreConfig {
            assign_cost: self.assign_cost,
            locked_assignments: self.session.state.locked_assignments_vec(),
            ..ScoreConfig::default()
        };
        let acfg = AnnealConfig {
            runs: self.replicas.max(4),
            steps_per_system: self.steps_per_sys.max(300),
            seed: self.sim_seed,
            ..AnnealConfig::default()
        };
        let mode = match self.solver_kind {
            GuiSolverKind::Standard => ParamagneticSolverMode::Standard,
            GuiSolverKind::EvoGraph => ParamagneticSolverMode::EvoGraph,
        };
        let para_cfg = ParamagneticAssignConfig {
            mode,
            explicit_ligand_indices: explicit_ligands.clone(),
            auto_detect_motifs: explicit_ligands.is_empty(),
            score_cfg: scfg.clone(),
            anneal_cfg: acfg,
            seed: self.sim_seed,
            ..ParamagneticAssignConfig::default()
        };

        let (res, para_rep, eff_priors, tag) = match self.solver_kind {
            GuiSolverKind::Standard => {
                if let Ok(sys) = build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
                    if !sys.is_empty() {
                        self.systems = sys;
                    }
                }
                let (r, rep) = assign_paramagnetic_backbone(
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &para_cfg,
                );
                let lbl = if cfg!(target_arch = "wasm32") {
                    "Paramagnetic WASM-SA"
                } else {
                    "Paramagnetic CPU-SA"
                };
                (r, rep, self.priors.clone(), lbl)
            }
            GuiSolverKind::EvoGraph => {
                let (sys, r, rep) = assign_paramagnetic_from_peaks(
                    &self.peaks,
                    &self.sequence,
                    &self.priors,
                    &para_cfg,
                );
                let eff = rep
                    .evo_report
                    .as_ref()
                    .map(|er| er.shifted_priors(&self.priors))
                    .unwrap_or_else(|| self.priors.clone());
                self.systems = sys;
                let lbl = if cfg!(target_arch = "wasm32") {
                    "Paramagnetic WASM-Evo"
                } else {
                    "Paramagnetic CPU-Evo"
                };
                (r, rep, eff, lbl)
            }
        };

        self.u_mat = Some(unary_matrix(
            &self.systems,
            &self.sequence,
            &eff_priors,
            &scfg,
        ));
        self.p_mat = Some(pair_matrix(&self.systems, &scfg));
        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        if let Some(truth) = &self.truth {
            if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
        self.status_msg = format!(
            "[{tag}] Assigned {}/{} residues ({n_hc} >=0.8 | {} metal cluster(s), {} blind-zone aa, {} abstained) in {dt_ms:.0} ms",
            self.pred_conf.len(),
            self.sequence.len(),
            para_rep.detected_clusters.len(),
            para_rep.blind_zone_residues.len(),
            para_rep.n_blind_zone_abstentions,
        );
    }

    fn run_assignment(&mut self) {
        match self.regime_kind {
            GuiRegimeKind::Idp13C => {
                self.run_idp_13c_assignment();
                return;
            }
            GuiRegimeKind::Paramagnetic => {
                self.run_paramagnetic_assignment();
                return;
            }
            GuiRegimeKind::Globular => {}
        }
        if self.sequence.is_empty()
            || (self.solver_kind == GuiSolverKind::EvoGraph && self.peaks.is_empty())
            || (self.solver_kind == GuiSolverKind::Standard && self.systems.is_empty())
        {
            self.status_msg = "Need peaks/spin systems and sequence to run assignment".to_string();
            return;
        }
        // Snapshot coordinates of user-locked spin systems so locks remap across filtered/rebuilt SpinSystems
        let locked_coords: Vec<(f64, f64, usize)> = self
            .session
            .state
            .locked_residues
            .iter()
            .filter_map(|(&ss_idx, &res_idx)| {
                self.systems.get(ss_idx).map(|ss| (ss.h, ss.n, res_idx))
            })
            .collect();

        let timer = PerfTimer::start();
        let (res, engine_tag, eff_priors, kept_hsqc_opt) = match self.solver_kind {
            GuiSolverKind::EvoGraph => {
                let raw_locks = self.raw_hsqc_locks_for_peaks();
                let scfg_base = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                let mut evo_cfg = EvoGraphConfig {
                    score_cfg: scfg_base.clone(),
                    auto_reference: scfg_base.auto_reference,
                    structure_torsions: None,
                    locked_assignments: raw_locks,
                    seed: self.sim_seed,
                    ..EvoGraphConfig::default()
                };
                evo_cfg.population_size = self.replicas.max(4);
                evo_cfg.num_islands = evo_cfg.num_islands.min(evo_cfg.population_size).max(1);
                evo_cfg.steps_per_system_per_gen = (self.steps_per_sys * 3 / 10).max(100);
                #[cfg(target_arch = "wasm32")]
                {
                    evo_cfg.generations = 4;
                    evo_cfg.steps_per_system_per_gen = evo_cfg.steps_per_system_per_gen.min(250);
                }
                let (evo_systems, r, rep) = assign_evo_graph_from_peaks(
                    &self.peaks,
                    &self.sequence,
                    &self.priors,
                    &evo_cfg,
                );
                let mut remapped_locks: BTreeMap<usize, usize> = BTreeMap::new();
                for &(lh, ln, lres) in &locked_coords {
                    if let Some((new_ss, _)) = evo_systems
                        .iter()
                        .enumerate()
                        .find(|&(_, s)| (s.h - lh).abs() <= 0.045 && (s.n - ln).abs() <= 0.45)
                    {
                        remapped_locks.insert(new_ss, lres);
                    }
                }
                self.session.state.locked_residues = remapped_locks;

                let scfg_u = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: self.session.state.locked_assignments_vec(),
                    ..ScoreConfig::default()
                };
                let shifted_pri = rep.shifted_priors(&self.priors);
                self.u_mat = Some(unary_matrix(
                    &evo_systems,
                    &self.sequence,
                    &shifted_pri,
                    &scfg_u,
                ));
                self.p_mat = Some(pair_matrix(&evo_systems, &scfg_u));
                self.qc_report = Some(rep.qc_report.clone());
                self.systems = evo_systems;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
                let tag = if cfg!(target_arch = "wasm32") {
                    "WASM Evo-Graph"
                } else {
                    "CPU Evo-Graph"
                };
                (r, tag, shifted_pri, Some(rep.kept_hsqc_indices))
            }
            GuiSolverKind::Standard => {
                // Rebuild unfiltered spin systems when switching back from Evo-Graph filtered systems
                if let Ok(sys) = build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
                    if sys.len() != self.systems.len() {
                        let mut remapped_locks: BTreeMap<usize, usize> = BTreeMap::new();
                        for &(lh, ln, lres) in &locked_coords {
                            if let Some((new_ss, _)) = sys.iter().enumerate().find(|&(_, s)| {
                                (s.h - lh).abs() <= 0.045 && (s.n - ln).abs() <= 0.45
                            }) {
                                remapped_locks.insert(new_ss, lres);
                            }
                        }
                        self.session.state.locked_residues = remapped_locks;
                        self.systems = sys;
                        self.selected_ss = (!self.systems.is_empty()).then_some(0);
                    }
                }
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: self.session.state.locked_assignments_vec(),
                    ..ScoreConfig::default()
                };
                self.p_mat = Some(pair_matrix(&self.systems, &scfg));
                let acfg = AnnealConfig {
                    runs: self.replicas,
                    steps_per_system: self.steps_per_sys,
                    seed: self.sim_seed,
                    ..AnnealConfig::default()
                };
                #[allow(unused_mut)]
                let mut res_opt: Option<(AssignResult, &'static str)> = None;

                #[cfg(not(target_arch = "wasm32"))]
                if self.use_gpu {
                    if let Some(gpu) = &self.gpu_engine {
                        match gpu.assign_gpu(
                            &self.systems,
                            &self.sequence,
                            &self.priors,
                            &scfg,
                            &acfg,
                        ) {
                            Ok(r) => {
                                self.u_mat = Some(unary_matrix(
                                    &self.systems,
                                    &self.sequence,
                                    &self.priors,
                                    &scfg,
                                ));
                                res_opt = Some((r, "WebGPU"));
                            }
                            Err(e) => {
                                self.status_msg =
                                    format!("GPU solver error ({e}), falling back to CPU");
                            }
                        }
                    }
                }

                let default_tag = if cfg!(target_arch = "wasm32") {
                    "WASM"
                } else {
                    "CPU-MT"
                };
                let (r, tag) = res_opt.unwrap_or_else(|| {
                    let u = unary_matrix(&self.systems, &self.sequence, &self.priors, &scfg);
                    let p = self
                        .p_mat
                        .clone()
                        .unwrap_or_else(|| pair_matrix(&self.systems, &scfg));
                    self.u_mat = Some(u.clone());
                    (assign_from_matrices(&u, &p, &acfg), default_tag)
                });
                let qc = evaluate_dataset_qc(
                    Some(&self.peaks),
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &scfg,
                    &r,
                    None,
                );
                self.qc_report = Some(qc);
                (r, tag, self.priors.clone(), None)
            }
        };

        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        if let Some(truth) = &self.truth {
            if let Some(kept) = &kept_hsqc_opt {
                let eval_truth: Vec<i32> = kept
                    .iter()
                    .map(|&idx| truth.get(idx).copied().unwrap_or(-1))
                    .collect();
                if eval_truth.len() == res.position.len() {
                    self.metrics = Some(Metrics {
                        assignable: truth.iter().filter(|&&t| t >= 0).count(),
                        ..score(&res, &eval_truth, 0.8)
                    });
                } else {
                    self.metrics = None;
                }
            } else if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        let n_locks = self.session.state.locked_residues.len();
        let lock_note = if n_locks > 0 {
            format!(", {n_locks} locked")
        } else {
            String::new()
        };
        self.status_msg = format!(
            "[{engine_tag}] Assigned {}/{} residues ({} high-conf >=0.8{lock_note}) in {:.0} ms",
            conf.len(),
            self.sequence.len(),
            n_hc,
            dt_ms
        );
        if let Some(ss) = self.selected_ss {
            if let Some(&k) = res.position.get(ss) {
                if k >= 0 {
                    self.selected_res = Some(k as usize);
                }
            }
        }
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
    }

    fn load_text_in_memory(&mut self, text: &str, source_label: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            self.status_msg =
                "Paste NMR-STAR (.str), NEF (.nef), Peak TSV (.tsv), CDXML, MOL, or SMILES first"
                    .to_string();
            return;
        }
        let lower_label = source_label.to_ascii_lowercase();

        // Check 3D biomolecular structure formats (`.pdb`, `.ent`, `.cif`, `.mmcif`)
        if lower_label.ends_with(".pdb")
            || lower_label.ends_with(".ent")
            || lower_label.ends_with(".cif")
            || lower_label.ends_with(".mmcif")
            || trimmed.contains("_atom_site.")
            || trimmed
                .lines()
                .take(48)
                .any(|ln| ln.starts_with("ATOM  ") || ln.starts_with("HETATM"))
        {
            self.load_structure_3d_in_memory(trimmed, source_label);
            return;
        }

        // Check chemical structure formats first (`CDXML`, `OCT`, `MOL`/`SDF`, `InChI`, `.smi`)
        if trimmed.contains("<CDXML")
            || lower_label.ends_with(".cdxml")
            || lower_label.ends_with(".oct")
            || lower_label.ends_with(".mol")
            || lower_label.ends_with(".sdf")
            || lower_label.ends_with(".smi")
            || trimmed.contains("V2000")
            || trimmed.contains("V3000")
            || trimmed.starts_with("InChI=")
        {
            let fmt = if trimmed.contains("<CDXML") || lower_label.ends_with(".cdxml") {
                IoFormat::Cdxml
            } else if lower_label.ends_with(".oct") || trimmed.starts_with('{') {
                IoFormat::OctJson
            } else if lower_label.ends_with(".sdf") || trimmed.contains("$$$$") {
                IoFormat::Sdf
            } else if trimmed.starts_with("InChI=") {
                IoFormat::InChI
            } else if trimmed.contains("V2000")
                || trimmed.contains("V3000")
                || lower_label.ends_with(".mol")
            {
                IoFormat::MolV2000
            } else {
                IoFormat::Smiles
            };
            if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, trimmed) {
                match res {
                    Ok(sc) => {
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded {} from {source_label}", fmt.label());
                        self.status_msg = format!("Loaded {} into Octet Studio", fmt.label());
                    }
                    Err(e) => self.status_msg = format!("Chemical file parse error: {e}"),
                }
                return;
            }
        }

        // Check NEF v1.1 (`data_nef_...`, `save_nef_molecular_system`, or `_nmrassign_workbench`)
        if lower_label.ends_with(".nef")
            || trimmed.contains("save_nef_molecular_system")
            || trimmed.starts_with("data_nef_")
        {
            if trimmed.contains("_nmrassign_workbench") {
                match self
                    .session
                    .load_from_nef_str(trimmed, Some(source_label.to_string()))
                {
                    Ok(summary) => {
                        self.truth = None;
                        self.metrics = None;
                        self.peaks = self.session.state.peaks.clone();
                        self.sequence = self.session.state.sequence.clone();
                        self.seq_ids = self.session.state.seq_ids.clone();
                        if !self.peaks.is_empty() {
                            self.regenerate_synthetic_spectra();
                        }
                        self.rebuild_systems_and_matrices();
                        self.sync_app_from_session();
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = summary;
                        return;
                    }
                    Err(e) => {
                        self.status_msg = format!("NEF session restore error: {e}");
                        return;
                    }
                }
            }

            match read_nef_str(trimmed) {
                Ok(doc) => match doc.to_reference() {
                    Ok(r) => {
                        let nef_pks = doc.to_peaks();
                        self.sequence = r.sequence.clone();
                        self.seq_ids = r.seq_ids.clone();
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.reference = Some(r.clone());
                        if !nef_pks.is_empty() {
                            self.peaks = nef_pks;
                            self.truth = None;
                            self.metrics = None;
                            self.regenerate_synthetic_spectra();
                            self.sync_session_from_app_fresh(&doc.project_name);
                            self.rebuild_systems_and_matrices();
                            self.run_assignment();
                        } else {
                            self.load_from_reference(r);
                            self.run_assignment();
                        }
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                    }
                    Err(e) => self.status_msg = format!("NEF molecular system error: {e}"),
                },
                Err(e) => self.status_msg = format!("NEF parse error ({source_label}): {e}"),
            }
            return;
        }

        let looks_like_star = trimmed.contains("loop_")
            || trimmed.contains("_Atom_chem_shift")
            || trimmed.contains("_Chem_shift_value")
            || trimmed.starts_with("data_");

        if looks_like_star {
            match read_star_str_all_atoms(trimmed, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.run_assignment();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR parse error ({source_label}): {e}"),
            }
        } else {
            match read_peaks_str(trimmed) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.sync_session_from_app_fresh("in_memory_peaks");
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!(
                        "Loaded {} peaks in memory from {source_label}",
                        self.peaks.len()
                    );
                }
                Err(e) => {
                    if let Some(ds) = parse_gui_c13_tsv_str(trimmed) {
                        let n_con = ds.con_roots.len();
                        let n_strip = ds.strip_peaks.len();
                        self.c13_picked_dataset = Some(ds);
                        self.regime_kind = GuiRegimeKind::Idp13C;
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        if !self.sequence.is_empty() {
                            self.run_assignment();
                        } else {
                            self.status_msg = format!(
                                "Loaded 13C-IDP TSV ({n_con} CON roots, {n_strip} 3D peaks) — load sequence/.str to assign"
                            );
                        }
                    } else {
                        self.status_msg = format!("Peak TSV parse error ({source_label}): {e}");
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_path_auto(&mut self) {
        let path_str = self.path_input.trim().to_string();
        let p = Path::new(&path_str);
        if !p.exists() {
            self.status_msg = format!("Path not found: {}", p.display());
            return;
        }
        let lower_full = path_str.to_ascii_lowercase();
        if p.is_dir()
            || lower_full.ends_with(".tar")
            || lower_full.ends_with(".tar.gz")
            || lower_full.ends_with(".tgz")
        {
            if p.is_dir() && p.join("2rr").exists() {
                match read_bruker_pdata_2d(p) {
                    Ok(spec) => {
                        self.hsqc_arc = Arc::new(spec.data.clone());
                        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                        self.hsqc_spec = Some(spec);
                        self.reset_view_to_spec();
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = format!("Loaded Bruker 2D pdata from {}", p.display());
                    }
                    Err(e) => self.status_msg = format!("Bruker load error: {e}"),
                }
                return;
            }
            match IngestPanelState::from_native_path(p) {
                Ok(panel) => {
                    let n_exp = panel.inventory.experiments.len();
                    self.ingest_panel = panel;
                    self.ingest_panel.open = true;
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg =
                        format!("Classified {n_exp} experiment(s) in {}", p.display());
                }
                Err(e) => self.status_msg = format!("Workspace classification error: {e}"),
            }
            return;
        }

        if p.file_name().and_then(|s| s.to_str()) == Some("2rr") {
            let dir = p.parent().unwrap_or(p);
            match read_bruker_pdata_2d(dir) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded Bruker 2D pdata from {}", dir.display());
                }
                Err(e) => self.status_msg = format!("Bruker load error: {e}"),
            }
            return;
        }

        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "pdb" | "ent" | "cif" | "mmcif" => match std::fs::read_to_string(p) {
                Ok(txt) => self.load_structure_3d_in_memory(&txt, &p.display().to_string()),
                Err(e) => self.status_msg = format!("3D structure read error: {e}"),
            },
            "str" | "star" => match read_star_all_atoms(p, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR error: {e}"),
            },
            "nef" => match std::fs::read_to_string(p) {
                Ok(txt) => self.load_text_in_memory(&txt, &p.display().to_string()),
                Err(e) => self.status_msg = format!("NEF read error: {e}"),
            },
            "fasta" | "fa" | "faa" | "seq" => match std::fs::read_to_string(p) {
                Ok(txt) => {
                    let parsed = nmr_gui::ingest_panel::parse_sequence_three_letter(&txt);
                    if parsed.is_empty() {
                        self.status_msg =
                            format!("No amino-acid sequence found in {}", p.display());
                    } else {
                        self.sequence = parsed;
                        self.seq_ids = (1..=(self.sequence.len() as i32)).collect();
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.session.state.sequence = self.sequence.clone();
                        self.session.state.seq_ids = self.seq_ids.clone();
                        self.rebuild_systems_and_matrices();
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = format!(
                            "Loaded {} residue sequence from {}",
                            self.sequence.len(),
                            p.display()
                        );
                    }
                }
                Err(e) => self.status_msg = format!("FASTA read error: {e}"),
            },
            "tsv" => match read_peaks(p) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.sync_session_from_app_fresh("tsv_peaks");
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg =
                        format!("Loaded {} peaks from {}", self.peaks.len(), p.display());
                }
                Err(e) => {
                    if let Ok(raw_txt) = std::fs::read_to_string(p) {
                        if let Some(ds) = parse_gui_c13_tsv_str(&raw_txt) {
                            let n_con = ds.con_roots.len();
                            let n_strip = ds.strip_peaks.len();
                            self.c13_picked_dataset = Some(ds);
                            self.regime_kind = GuiRegimeKind::Idp13C;
                            self.workspace_mode = WorkspaceMode::ProteinNmr;
                            self.status_msg = format!(
                                "Loaded 13C-IDP TSV from {} ({n_con} CON roots, {n_strip} 3D peaks)",
                                p.display()
                            );
                            return;
                        }
                    }
                    self.status_msg = format!("Peak TSV error: {e}");
                }
            },
            "ucsf" => match read_ucsf_2d(p) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded UCSF 2D spectrum {}", p.display());
                }
                Err(_) => match read_ucsf_3d(p, "HNCACB") {
                    Ok(spec3d) => {
                        self.hncacb_3d = Some(spec3d);
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = format!("Loaded UCSF 3D spectrum {}", p.display());
                    }
                    Err(e) => self.status_msg = format!("UCSF load error: {e}"),
                },
            },
            "ft2" | "dat" | "pipe" => match read_nmrpipe_2d(p) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded NMRPipe 2D spectrum {}", p.display());
                }
                Err(e) => self.status_msg = format!("NMRPipe load error: {e}"),
            },
            "ft3" => match read_nmrpipe_3d(p, "HNCACB") {
                Ok(spec3d) => {
                    self.hncacb_3d = Some(spec3d);
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded NMRPipe 3D spectrum {}", p.display());
                }
                Err(e) => self.status_msg = format!("NMRPipe 3D load error: {e}"),
            },
            "cdx" => match std::fs::read(p) {
                Ok(bytes) => match read_cdx(&bytes) {
                    Ok(doc) => {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded binary ChemDraw CDX {}", p.display());
                        self.status_msg = format!("Loaded binary CDX {}", p.display());
                    }
                    Err(e) => self.status_msg = format!("CDX parse error: {e}"),
                },
                Err(e) => self.status_msg = format!("CDX read error: {e}"),
            },
            "cdxml" | "oct" | "mol" | "sdf" | "smi" | "jdx" | "dx" => {
                match std::fs::read_to_string(p) {
                    Ok(txt) => {
                        let fmt = match ext.as_str() {
                            "cdxml" => IoFormat::Cdxml,
                            "oct" => IoFormat::OctJson,
                            "mol" => IoFormat::MolV2000,
                            "sdf" => IoFormat::Sdf,
                            "jdx" | "dx" => IoFormat::JcampDx,
                            _ => IoFormat::Smiles,
                        };
                        if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, &txt) {
                            match res {
                                Ok(sc) => {
                                    self.octet_app.set_scene(sc);
                                    self.workspace_mode = WorkspaceMode::OctetChem;
                                    self.octet_app.status_msg =
                                        format!("Loaded {} from {}", fmt.label(), p.display());
                                    self.status_msg = format!("Loaded {}", p.display());
                                }
                                Err(e) => self.status_msg = format!("Import error: {e}"),
                            }
                        }
                    }
                    Err(e) => self.status_msg = format!("File read error: {e}"),
                }
            }
            _ => {
                self.status_msg = format!(
                    "Unrecognized file extension .{ext} (expected folder, .tar.gz, .ucsf, .str, .nef, .fasta, .tsv, .ft2, .ft3)"
                );
            }
        }
    }

    fn pick_2d_roots(&mut self) {
        let Some(spec) = &self.hsqc_spec else {
            self.status_msg = "No 2D HSQC spectrum loaded".to_string();
            return;
        };
        let (picked, method_label) = if self.use_neural_picker {
            (
                pick_hsqc_2d_neural(
                    spec,
                    &NeuralPickerConfig {
                        prob_threshold: (0.25 + 0.035 * (self.picker_snr - 3.0)).clamp(0.20, 0.85),
                        ..NeuralPickerConfig::default()
                    },
                ),
                "Neural 2D CNN picker",
            )
        } else {
            (
                pick_hsqc_2d(
                    spec,
                    &PickerConfig {
                        snr_threshold: self.picker_snr,
                        ..PickerConfig::default()
                    },
                ),
                "Classical picker",
            )
        };
        let n_picked = picked.len();
        // Keep existing triple-resonance peaks while replacing HSQC roots
        let mut combined = picked;
        combined.extend(self.peaks.iter().filter(|p| p.exp != "HSQC").cloned());
        self.peaks = combined;
        self.session.state.peaks = self.peaks.clone();
        self.truth = None;
        self.metrics = None;
        self.rebuild_systems_and_matrices();
        self.status_msg = format!(
            "{method_label} found {n_picked} HSQC roots -> rebuilt {} spin systems",
            self.systems.len()
        );
    }

    fn format_shifts_tsv(&self, pred: &Reference) -> String {
        let mut rows = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
        for (&(k, ref atom), &v) in &pred.shifts {
            let c = self.pred_conf.get(&k).copied().unwrap_or(0.0);
            rows.push(format!(
                "{}\t{}\t{}\t{:.3}\t{:.2}",
                self.seq_ids[k], self.sequence[k], atom, v, c
            ));
        }
        rows.join("\n") + "\n"
    }

    fn export_results(&mut self, ctx: &egui::Context) {
        let Some(pred) = &self.pred_ref else {
            self.status_msg = "Run assignment first before exporting".to_string();
            return;
        };
        let tsv_text = self.format_shifts_tsv(pred);
        let star_text = format_star_in_memory(pred, Some(&self.pred_conf));
        self.export_text_buf =
            format!("# --- NMR-STAR v3 ---\n{star_text}\n# --- Shifts TSV ---\n{tsv_text}");
        self.exported_in_session = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            let prefix = Path::new(self.export_prefix.trim());
            let str_p = prefix.with_extension("str");
            let tsv_p = prefix.with_extension("tsv");
            if let Err(e) = write_star(&str_p, pred, Some(&self.pred_conf)) {
                self.status_msg = format!("Export failed: {e}");
                return;
            }
            let _ = std::fs::write(&tsv_p, &tsv_text);
            self.status_msg = format!("Exported {} and {}", str_p.display(), tsv_p.display());
        }

        #[cfg(target_arch = "wasm32")]
        {
            ctx.copy_text(star_text);
            self.show_text_io_modal = true;
            self.status_msg =
                "Exported NMR-STAR v3 + TSV in memory (NMR-STAR copied to clipboard)".to_string();
        }
    }

    fn apply_drop_outcome(&mut self, outcome: IngestDropOutcome) {
        match outcome {
            IngestDropOutcome::NefSession {
                source_name,
                nef_bytes,
            } => match String::from_utf8(nef_bytes) {
                Ok(txt) => self.load_text_in_memory(&txt, &source_name),
                Err(e) => self.status_msg = format!("Dropped NEF is not valid UTF-8: {e}"),
            },
            IngestDropOutcome::TextTable { source_name, text } => {
                self.load_text_in_memory(&text, &source_name);
            }
            IngestDropOutcome::Workspace(panel) => {
                let n_exp = panel.inventory.experiments.len();
                let root = panel.inventory.root_name.clone();
                self.ingest_panel = *panel;
                self.ingest_panel.open = true;
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.status_msg = format!(
                    "Classified '{root}' ({n_exp} experiment(s)) — review selection matrix"
                );
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        {
            let ready = self
                .pending_drop_outcome
                .lock()
                .ok()
                .and_then(|mut g| g.take());
            if let Some(res) = ready {
                match res {
                    Ok(outcome) => self.apply_drop_outcome(outcome),
                    Err(e) => self.status_msg = format!("Drop ingest error: {e}"),
                }
            }
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            // Intercept 3D biomolecular structure files (`.pdb`, `.ent`, `.cif`, `.mmcif`) directly
            let mut struct_files = Vec::new();
            let mut remaining = Vec::with_capacity(dropped.len());
            for df in dropped {
                let fname = df
                    .path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| df.path().display().to_string());
                let low = fname.to_ascii_lowercase();
                if low.ends_with(".pdb")
                    || low.ends_with(".ent")
                    || low.ends_with(".cif")
                    || low.ends_with(".mmcif")
                {
                    struct_files.push((fname, df));
                } else {
                    remaining.push(df);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _ = ctx;
                for (fname, df) in struct_files {
                    let loaded_txt = df
                        .bytes()
                        .or_else(|_| std::fs::read(df.path()).map_err(|e| e.to_string()))
                        .and_then(|b| {
                            String::from_utf8(b).map_err(|e| format!("UTF-8 error: {e}"))
                        });
                    match loaded_txt {
                        Ok(txt) => self.load_structure_3d_in_memory(&txt, &fname),
                        Err(e) => self.status_msg = format!("Structure drop error: {e}"),
                    }
                }
                if !remaining.is_empty() {
                    match ingest_dropped_files_native(&remaining) {
                        Ok(outcome) => self.apply_drop_outcome(outcome),
                        Err(e) => self.status_msg = format!("Drop ingest error: {e}"),
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some((fname, df)) = struct_files.into_iter().next() {
                    let slot = Arc::clone(&self.pending_drop_outcome);
                    let egui_ctx = ctx.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let res = df
                            .bytes_async()
                            .await
                            .and_then(|b| {
                                String::from_utf8(b).map_err(|e| format!("UTF-8 error: {e}"))
                            })
                            .map(|text| IngestDropOutcome::TextTable {
                                source_name: fname,
                                text,
                            });
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(res);
                        }
                        egui_ctx.request_repaint();
                    });
                } else if !remaining.is_empty() {
                    let slot = Arc::clone(&self.pending_drop_outcome);
                    let egui_ctx = ctx.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let res = ingest_dropped_files_wasm(remaining).await;
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(res);
                        }
                        egui_ctx.request_repaint();
                    });
                }
            }
        }
    }

    fn draw_text_io_window(&mut self, ctx: &egui::Context) {
        if !self.show_text_io_modal {
            return;
        }
        let mut open = self.show_text_io_modal;
        egui::Window::new("Paste / In-Memory NMR-STAR & Peak TSV")
            .open(&mut open)
            .default_width(560.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Paste an NMR-STAR (.str) entry or TSV peak list below (or drag & drop a .str/.tsv file onto the canvas):",
                    )
                    .small(),
                );
                ui.horizontal(|ui| {
                    if ui.small_button("Fill Synthetic Ubiquitin .str").clicked() {
                        self.paste_text_buf = SYNTHETIC_UBIQUITIN_STR.to_string();
                    }
                    if ui.small_button("Load Pasted Text & Solve").clicked() {
                        let buf = self.paste_text_buf.clone();
                        self.load_text_in_memory(&buf, "pasted text");
                    }
                    if ui.small_button("Clear").clicked() {
                        self.paste_text_buf.clear();
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("paste_text_scroll")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.paste_text_buf)
                                .font(egui::TextStyle::Monospace)
                                .desired_rows(8)
                                .desired_width(f32::INFINITY),
                        );
                    });

                if !self.export_text_buf.is_empty() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Exported NMR-STAR / TSV:").small().strong());
                        if ui.small_button("Copy All to Clipboard").clicked() {
                            ui.ctx().copy_text(self.export_text_buf.clone());
                        }
                    });
                    egui::ScrollArea::vertical()
                        .id_salt("export_text_scroll")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            let mut preview = self.export_text_buf.as_str();
                            ui.add(
                                egui::TextEdit::multiline(&mut preview)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_rows(8)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                }
            });
        self.show_text_io_modal = open;
    }

    fn draw_workspace_mode_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(
                egui::RichText::new("nmr-assign")
                    .strong()
                    .color(egui::Color32::from_rgb(110, 220, 255)),
            );
            if let Some(label) = &self.adapter_label {
                ui.label(
                    egui::RichText::new(format!("⚡ {label}"))
                        .small()
                        .color(egui::Color32::from_rgb(130, 230, 175)),
                );
            }
            ui.separator();

            ui.selectable_value(
                &mut self.workspace_mode,
                WorkspaceMode::ProteinNmr,
                egui::RichText::new("🧬 Protein NMR & 3D Strips").strong(),
            );
            ui.selectable_value(
                &mut self.workspace_mode,
                WorkspaceMode::OctetChem,
                egui::RichText::new("⚗️ Octet Chemical Draw & 1D NMR").strong(),
            );

            if self.workspace_mode == WorkspaceMode::OctetChem {
                ui.separator();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.label("Path:");
                    ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(185.0));
                    if ui.button("📂 Load (.cdxml/.cdx/.mol/.sdf/.jdx)").clicked() {
                        self.load_path_auto();
                    }
                }
                ui.label(
                    egui::RichText::new(&self.octet_app.status_msg)
                        .small()
                        .color(egui::Color32::from_rgb(190, 210, 235)),
                );
            }
        });
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("🧬 Ubiquitin Demo").clicked() {
                self.load_demo_and_solve();
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label("Path:");
                ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(200.0));
                if ui
                    .button("📂 Load (.str / .nef / .tsv / pdata / .ft2)")
                    .clicked()
                {
                    self.load_path_auto();
                }
            }

            if ui.button("📋 Paste .str / .nef / .tsv").clicked() {
                self.show_text_io_modal = !self.show_text_io_modal;
            }

            ui.separator();
            egui::ComboBox::from_id_salt("gui_regime_kind")
                .selected_text(format!("Regime: {}", self.regime_kind.label()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.regime_kind,
                        GuiRegimeKind::Globular,
                        "Globular (¹H–¹⁵N HSQC + 3D)",
                    );
                    ui.selectable_value(
                        &mut self.regime_kind,
                        GuiRegimeKind::Idp13C,
                        "13C-IDP (CON / CACO / CANCO + Pro)",
                    );
                    ui.selectable_value(
                        &mut self.regime_kind,
                        GuiRegimeKind::Paramagnetic,
                        "Paramagnetic (PRE Blind-Zone + Metal)",
                    );
                });
            if self.regime_kind == GuiRegimeKind::Paramagnetic {
                ui.label(egui::RichText::new("Ligands:").small());
                ui.add(
                    egui::TextEdit::singleline(&mut self.metal_ligands_input)
                        .hint_text("auto motif or CYS26,HIS41")
                        .desired_width(135.0),
                );
            } else if self.regime_kind == GuiRegimeKind::Idp13C {
                if ui
                    .add(
                        egui::DragValue::new(&mut self.idp_temp_k)
                            .speed(0.5)
                            .range(265.0..=335.0)
                            .prefix("T ")
                            .suffix(" K"),
                    )
                    .changed()
                {
                    self.refresh_propensity_report();
                }
                ui.label(egui::RichText::new("pH:").small());
                if ui
                    .add(egui::TextEdit::singleline(&mut self.idp_ph_input).desired_width(38.0))
                    .changed()
                {
                    self.refresh_propensity_report();
                }
            }
            if self.regime_kind != GuiRegimeKind::Idp13C {
                egui::ComboBox::from_id_salt("gui_solver_kind")
                    .selected_text(format!("Solver: {}", self.solver_kind.label()))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.solver_kind,
                            GuiSolverKind::EvoGraph,
                            "Evo-Graph (CPU f64, default)",
                        );
                        ui.selectable_value(
                            &mut self.solver_kind,
                            GuiSolverKind::Standard,
                            "Standard SA (GPU / CPU)",
                        );
                    });
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let gpu_supported = (self.regime_kind == GuiRegimeKind::Globular
                    && self.solver_kind == GuiSolverKind::Standard)
                    || self.regime_kind == GuiRegimeKind::Idp13C;
                ui.add_enabled(
                    gpu_supported,
                    egui::Checkbox::new(&mut self.use_gpu, "WebGPU Compute"),
                )
                .on_disabled_hover_text(
                    "WebGPU Compute accelerates Standard SA and 13C-IDP multi-replica annealing; Evo-Graph and Paramagnetic run in f64 on CPU",
                );
            }
            ui.add(egui::Slider::new(&mut self.replicas, 8..=256).text("Replicas"));
            ui.add(egui::Slider::new(&mut self.steps_per_sys, 400..=4000).text("Steps/SS"));
            if ui
                .add(
                    egui::Slider::new(&mut self.assign_cost, 0.0..=4.0)
                        .text("Assign cost")
                        .fixed_decimals(2),
                )
                .changed()
                && !self.systems.is_empty()
                && !self.sequence.is_empty()
            {
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                self.u_mat = Some(unary_matrix(
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &scfg,
                ));
            }

            let solve_btn = egui::Button::new(
                egui::RichText::new("▶ Run Assignment")
                    .strong()
                    .color(egui::Color32::from_rgb(15, 25, 20)),
            )
            .fill(egui::Color32::from_rgb(82, 224, 152));
            if ui.add(solve_btn).clicked() {
                self.run_assignment();
            }

            if ui.button("💾 Export").clicked() {
                let ctx = ui.ctx().clone();
                self.export_results(&ctx);
            }
        });

        ui.add_space(3.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Simulator / Picker:").small().weak());
            let mut changed_sim = false;
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_missing, 0.0..=0.25)
                        .text("Missing")
                        .fixed_decimals(2),
                )
                .changed();
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_spurious, 0.0..=0.20)
                        .text("Spurious")
                        .fixed_decimals(2),
                )
                .changed();
            let _ = changed_sim;
            if ui.small_button("🎲 Re-simulate & Solve").clicked() {
                self.sim_seed = self.sim_seed.wrapping_add(1);
                if let Some(r) = self.reference.clone() {
                    self.load_from_reference(r);
                    self.run_assignment();
                }
            }
            ui.separator();
            ui.checkbox(&mut self.use_neural_picker, "Neural CNN")
                .on_hover_text(
                    "Toggle 2D separable residual CNN peak picker vs classical MAD picker",
                );
            ui.add(egui::Slider::new(&mut self.picker_snr, 3.0..=12.0).text("Pick SNR"));
            if ui.small_button("🔍 Pick 2D HSQC Peaks").clicked() {
                self.pick_2d_roots();
            }
            ui.separator();
            ui.label(
                egui::RichText::new(&self.status_msg)
                    .small()
                    .color(egui::Color32::from_rgb(190, 210, 235)),
            );
        });
    }

    fn draw_sequence_ribbon(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            return;
        }
        let mut res_to_ss: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
        if let Some(res) = &self.assign_res {
            for (s, &k) in res.position.iter().enumerate() {
                if k >= 0 {
                    res_to_ss.insert(k as usize, (s, res.confidence[s]));
                }
            }
        }

        egui::ScrollArea::horizontal()
            .id_salt("seq_ribbon_scroll")
            .max_height(52.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    let toggle_btn = egui::Button::new(
                        egui::RichText::new("🔬 2D Graph")
                            .size(11.0)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if self.show_sidechain_inspector {
                        egui::Color32::from_rgb(20, 108, 130)
                    } else {
                        egui::Color32::from_rgb(32, 42, 58)
                    })
                    .stroke(egui::Stroke::new(
                        1.2,
                        if self.show_sidechain_inspector {
                            egui::Color32::from_rgb(120, 235, 255)
                        } else {
                            egui::Color32::from_rgb(76, 92, 118)
                        },
                    ))
                    .min_size(egui::vec2(68.0, 28.0));
                    if ui
                        .add(toggle_btn)
                        .on_hover_text(
                            "Toggle live 2D IUPAC covalent sidechain Z-score inspector & atom crosshair jump",
                        )
                        .clicked()
                    {
                        self.show_sidechain_inspector = !self.show_sidechain_inspector;
                        if self.show_sidechain_inspector && self.selected_res.is_none() {
                            self.selected_res = Some(0);
                        }
                    }
                    let prop_btn = egui::Button::new(
                        egui::RichText::new("📊 SSP / δ2D")
                            .size(11.0)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if self.show_propensity_panel {
                        egui::Color32::from_rgb(104, 48, 136)
                    } else {
                        egui::Color32::from_rgb(32, 42, 58)
                    })
                    .stroke(egui::Stroke::new(
                        1.2,
                        if self.show_propensity_panel {
                            egui::Color32::from_rgb(214, 165, 255)
                        } else {
                            egui::Color32::from_rgb(76, 92, 118)
                        },
                    ))
                    .min_size(egui::vec2(78.0, 28.0));
                    if ui
                        .add(prop_btn)
                        .on_hover_text(
                            "Toggle per-residue Marsh SSP & Camilloni 4-state δ2D (α / β / PPII / coil) ensemble track",
                        )
                        .clicked()
                    {
                        self.show_propensity_panel = !self.show_propensity_panel;
                        if self.show_propensity_panel {
                            self.refresh_propensity_report();
                            if self.selected_res.is_none() {
                                self.selected_res = Some(0);
                            }
                        }
                    }
                    let mol3d_btn = egui::Button::new(
                        egui::RichText::new("🧬 3D Structure")
                            .size(11.0)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if self.show_mol3d_viewport {
                        egui::Color32::from_rgb(24, 96, 118)
                    } else {
                        egui::Color32::from_rgb(32, 42, 58)
                    })
                    .stroke(egui::Stroke::new(
                        1.2,
                        if self.show_mol3d_viewport {
                            egui::Color32::from_rgb(120, 235, 255)
                        } else {
                            egui::Color32::from_rgb(76, 92, 118)
                        },
                    ))
                    .min_size(egui::vec2(86.0, 28.0));
                    if ui
                        .add(mol3d_btn)
                        .on_hover_text(
                            "Toggle dockable WebGPU 3D biomolecular structure, NMR putty & NOE/PRE restraint viewport",
                        )
                        .clicked()
                    {
                        self.show_mol3d_viewport = !self.show_mol3d_viewport;
                    }
                    ui.add_space(4.0);
                    for (k, three) in self.sequence.iter().enumerate() {
                        let one = three_to_one(three);
                        let seq_id = self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                        let is_pro_or_nterm = k == 0 || three == "PRO";
                        let assigned = res_to_ss.get(&k).copied();
                        let direct_conf = assigned
                            .map(|(_, c)| c)
                            .or_else(|| self.pred_conf.get(&k).copied());
                        let is_selected = self.selected_res == Some(k);

                        let fill = if let Some(conf) = direct_conf {
                            if conf >= 0.80 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if conf >= 0.50 {
                                egui::Color32::from_rgb(148, 112, 28)
                            } else {
                                egui::Color32::from_rgb(145, 68, 42)
                            }
                        } else if is_pro_or_nterm {
                            egui::Color32::from_rgb(38, 43, 54)
                        } else {
                            egui::Color32::from_rgb(48, 56, 74)
                        };

                        let is_locked = self.session.state.is_residue_locked(k);
                        let stroke = if is_selected {
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 235, 255))
                        } else if is_locked {
                            egui::Stroke::new(1.8, egui::Color32::from_rgb(255, 215, 90))
                        } else {
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 76, 98))
                        };

                        let label = if is_locked {
                            format!("🔒{one}{seq_id}")
                        } else {
                            format!("{one}{seq_id}")
                        };
                        let btn = egui::Button::new(
                            egui::RichText::new(label)
                                .monospace()
                                .size(11.5)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(fill)
                        .stroke(stroke)
                        .min_size(egui::vec2(34.0, 28.0));

                        let resp = ui.add(btn);
                        if resp.clicked() {
                            self.selected_res = Some(k);
                            self.sidechain_ribbon_state.selected_residue = k;
                            self.assign_target_res = k;
                            if let Some((ss_idx, _)) = assigned {
                                self.selected_ss = Some(ss_idx);
                                if let Some(ss) = self.systems.get(ss_idx) {
                                    self.crosshair.snap_to(ss.h, ss.n, None);
                                }
                            }
                            self.mol3d_panel.highlight_sequence_residue(Some(k), true);
                        }
                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new(format!("{three} {seq_id}")).strong());
                            if let Some((ss_idx, conf)) = assigned {
                                ui.label(format!(
                                    "Spin system #{ss_idx} (conf {:.0}%{})",
                                    conf * 100.0,
                                    if is_locked { ", LOCKED" } else { "" }
                                ));
                            } else if let Some(conf) = direct_conf {
                                ui.label(format!(
                                    "Assigned ({}) — conf {:.0}%{}",
                                    self.regime_kind.label(),
                                    conf * 100.0,
                                    if is_locked { ", LOCKED" } else { "" }
                                ));
                            } else if is_pro_or_nterm {
                                ui.label("No amide H root (N-term / Proline)");
                            } else {
                                ui.label("Unassigned");
                            }
                            if let Some(pred) = &self.pred_ref {
                                for atom in ["H", "N", "CA", "CB", "C"] {
                                    if let Some(&v) = pred.shifts.get(&(k, atom.to_string())) {
                                        ui.monospace(format!("  {atom:2}: {v:7.3} ppm"));
                                    }
                                }
                            }
                        });
                    }
                });
            });
    }

    fn draw_hsqc_canvas(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("2D ¹H–¹⁵N HSQC (WebGPU Multi-Layer)").strong());
            ui.add(
                egui::Slider::new(&mut self.contour_base, 0.5..=60.0)
                    .logarithmic(true)
                    .text("Base"),
            );
            ui.add(egui::Slider::new(&mut self.contour_factor, 1.15..=2.0).text("Step"));
            ui.add(egui::Slider::new(&mut self.contour_levels, 4..=16).text("Levels"));
            ui.checkbox(&mut self.show_negative, "Neg");
            ui.checkbox(&mut self.show_peak_labels, "Labels");
            if ui
                .small_button(format!("⧉ Overlays ({})", 1 + self.overlay_layers.len()))
                .clicked()
            {
                self.show_overlay_manager = !self.show_overlay_manager;
            }
            let lock_txt = if self.crosshair.locked {
                "🔒 Crosshair"
            } else {
                "┼ Crosshair"
            };
            if ui.small_button(lock_txt).clicked() {
                self.crosshair.toggle_lock();
            }
            if ui.small_button("Reset Zoom").clicked() {
                self.reset_view_to_spec();
            }
        });

        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(200.0), (avail.y - 4.0).max(200.0));
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

        // Pan with drag
        if response.dragged() {
            let delta = response.drag_delta();
            let span_h = self.view_h_right - self.view_h_left;
            let span_n = self.view_n_bottom - self.view_n_top;
            let dh = -delta.x / rect.width().max(1.0) * span_h;
            let dn = -delta.y / rect.height().max(1.0) * span_n;
            self.view_h_left += dh;
            self.view_h_right += dh;
            self.view_n_top += dn;
            self.view_n_bottom += dn;
        }
        if response.double_clicked() {
            self.reset_view_to_spec();
        }

        // Scroll-wheel zoom around pointer
        if let Some(hover_pos) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let zoom = (1.0 - scroll * 0.0025).clamp(0.75, 1.35);
                let u = ((hover_pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let v = ((hover_pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                let anchor_h = self.view_h_left + u * (self.view_h_right - self.view_h_left);
                let anchor_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
                self.view_h_left = anchor_h + (self.view_h_left - anchor_h) * zoom;
                self.view_h_right = anchor_h + (self.view_h_right - anchor_h) * zoom;
                self.view_n_top = anchor_n + (self.view_n_top - anchor_n) * zoom;
                self.view_n_bottom = anchor_n + (self.view_n_bottom - anchor_n) * zoom;
            }
        }

        let view_ppm = [
            self.view_h_left,
            self.view_h_right,
            self.view_n_top,
            self.view_n_bottom,
        ];

        let painter = ui.painter_at(rect);
        let mut gpu_layers: Vec<OverlayLayerSpec> = Vec::new();
        if let Some(spec) = &self.hsqc_spec {
            let mut base_layer = OverlayLayerSpec::from_spectrum_2d(
                1,
                self.hsqc_gen,
                spec,
                Arc::clone(&self.hsqc_arc),
                self.contour_base,
                0,
            );
            base_layer.level_factor = self.contour_factor;
            base_layer.num_levels = self.contour_levels;
            base_layer.show_negative = self.show_negative;
            base_layer.ppm_nudge = self.primary_ppm_nudge;
            gpu_layers.push(base_layer);
        }
        for (_, ov) in &self.overlay_layers {
            if ov.visible {
                gpu_layers.push(ov.clone());
            }
        }

        if !gpu_layers.is_empty() {
            let cb = egui_wgpu::Callback::new_paint_callback(
                rect,
                MultiContourCallback::new(view_ppm, [rect.width(), rect.height()], gpu_layers),
            );
            painter.add(cb);
        } else {
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 25));
        }

        let ppm_to_screen = |h: f64, n: f64| -> egui::Pos2 {
            SyncCrosshair::ppm_to_screen_2d(view_ppm, rect, h, n)
        };

        // Click to select nearest spin system
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                let mut best: Option<(usize, f32)> = None;
                for (s, ss) in self.systems.iter().enumerate() {
                    let pt = ppm_to_screen(ss.h, ss.n);
                    let dist = pt.distance(click_pos);
                    if dist <= 18.0 && best.is_none_or(|(_, bd)| dist < bd) {
                        best = Some((s, dist));
                    }
                }
                if let Some((s, _)) = best {
                    self.selected_ss = Some(s);
                    let ss = &self.systems[s];
                    self.crosshair.snap_to(ss.h, ss.n, None);
                    if let Some(res) = &self.assign_res {
                        if res.position.get(s).copied().unwrap_or(-1) >= 0 {
                            let rk = res.position[s] as usize;
                            self.selected_res = Some(rk);
                            self.assign_target_res = rk;
                            self.mol3d_panel.highlight_sequence_residue(Some(rk), true);
                        }
                    }
                }
            }
        }

        // Draw spin system peak markers and residue labels
        for (s, ss) in self.systems.iter().enumerate() {
            let pt = ppm_to_screen(ss.h, ss.n);
            if !rect.contains(pt) {
                continue;
            }
            let is_sel = self.selected_ss == Some(s);
            let is_locked = self.session.state.is_spin_system_locked(s);
            let (assigned_k, conf) = self
                .assign_res
                .as_ref()
                .and_then(|r| Some((*r.position.get(s)?, *r.confidence.get(s)?)))
                .unwrap_or((-1, 0.0));

            let marker_col = if is_locked {
                egui::Color32::from_rgb(255, 222, 92)
            } else if assigned_k >= 0 {
                if conf >= 0.80 {
                    egui::Color32::from_rgb(95, 245, 165)
                } else {
                    egui::Color32::from_rgb(255, 205, 95)
                }
            } else {
                egui::Color32::from_rgb(155, 175, 205)
            };

            let sz = if is_sel { 6.5 } else { 3.8 };
            painter.line_segment(
                [egui::pos2(pt.x - sz, pt.y), egui::pos2(pt.x + sz, pt.y)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            painter.line_segment(
                [egui::pos2(pt.x, pt.y - sz), egui::pos2(pt.x, pt.y + sz)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            if is_sel {
                painter.circle_stroke(
                    pt,
                    10.5,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 240, 120)),
                );
            }

            if self.show_peak_labels {
                let text = if assigned_k >= 0 && (assigned_k as usize) < self.sequence.len() {
                    let k = assigned_k as usize;
                    if is_locked {
                        format!("🔒{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                    } else {
                        format!("{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                    }
                } else {
                    format!("#{s}")
                };
                painter.text(
                    egui::pos2(pt.x + 6.0, pt.y - 6.0),
                    egui::Align2::LEFT_BOTTOM,
                    text,
                    egui::FontId::monospace(10.5),
                    marker_col,
                );
            }
        }

        // Update & draw synchronized crosshair
        if let Some(hp) = response.hover_pos() {
            let u = (hp.x - rect.left()) / rect.width().max(1.0);
            let v = (hp.y - rect.top()) / rect.height().max(1.0);
            let cur_h = (self.view_h_left + u * (self.view_h_right - self.view_h_left)) as f64;
            let cur_n = (self.view_n_top + v * (self.view_n_bottom - self.view_n_top)) as f64;
            self.crosshair.update_from_2d_hover(cur_h, cur_n);
        }
        let ch_pt = ppm_to_screen(self.crosshair.h_ppm, self.crosshair.n_ppm);
        if rect.contains(ch_pt) {
            let ch_stroke = if self.crosshair.locked {
                egui::Stroke::new(
                    1.1,
                    egui::Color32::from_rgba_unmultiplied(255, 215, 90, 175),
                )
            } else {
                egui::Stroke::new(
                    0.8,
                    egui::Color32::from_rgba_unmultiplied(110, 215, 255, 105),
                )
            };
            painter.line_segment(
                [
                    egui::pos2(rect.left(), ch_pt.y),
                    egui::pos2(rect.right(), ch_pt.y),
                ],
                ch_stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(ch_pt.x, rect.top()),
                    egui::pos2(ch_pt.x, rect.bottom()),
                ],
                ch_stroke,
            );
        }

        // Draw axis ticks and live cursor ppm readout
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 80, 105)),
            egui::StrokeKind::Inside,
        );
        for i in 0..=5 {
            let t = (i as f32) / 5.0;
            let h_val = self.view_h_left + t * (self.view_h_right - self.view_h_left);
            let x = rect.left() + t * rect.width();
            painter.text(
                egui::pos2(x, rect.bottom() - 3.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{h_val:.2} ¹H"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );

            let n_val = self.view_n_top + t * (self.view_n_bottom - self.view_n_top);
            let y = rect.top() + t * rect.height();
            painter.text(
                egui::pos2(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{n_val:.1}"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );
        }

        let lock_tag = if self.crosshair.locked {
            " [LOCKED]"
        } else {
            ""
        };
        painter.text(
            egui::pos2(rect.right() - 8.0, rect.top() + 6.0),
            egui::Align2::RIGHT_TOP,
            format!(
                "¹H {:.3} ppm | ¹⁵N {:.2} ppm{lock_tag}",
                self.crosshair.h_ppm, self.crosshair.n_ppm
            ),
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(220, 235, 255),
        );
    }

    fn select_residue_and_sync_strip(&mut self, k: usize) {
        if k >= self.sequence.len() {
            return;
        }
        self.selected_res = Some(k);
        self.sidechain_ribbon_state.selected_residue = k;
        self.assign_target_res = k;
        self.mol3d_panel.highlight_sequence_residue(Some(k), true);
        if let Some(res) = &self.assign_res {
            for (s, &pos) in res.position.iter().enumerate() {
                if pos >= 0 && (pos as usize) == k {
                    self.selected_ss = Some(s);
                    if let Some(ss) = self.systems.get(s) {
                        self.crosshair.snap_to(ss.h, ss.n, self.crosshair.c_ppm);
                    }
                    break;
                }
            }
        }
    }

    fn run_joint_sidechain_in_gui(&mut self) {
        if self.sequence.is_empty() {
            self.status_msg =
                "Load sequence and peaks/spin systems before sidechain assignment".to_string();
            return;
        }
        let timer = PerfTimer::start();
        let seed = self.sim_seed.wrapping_add(2026);
        let prior_user_ref = self.pred_ref.clone().or_else(|| self.reference.clone());
        let prior_amb = self.pred_ambiguity_codes.clone();
        let mut loaded_sc_peaks = Vec::new();
        let mut loaded_arom_peaks = Vec::new();
        for p in &self.peaks {
            if EXPERIMENT_NAMES.contains(&p.exp.as_str()) {
                continue;
            }
            if AROMATIC_EXPERIMENTS.contains(&p.exp.as_str()) || p.exp == "AROM_HB_NOE" {
                loaded_arom_peaks.push(p.clone());
            } else {
                loaded_sc_peaks.push(p.clone());
            }
        }

        let mut working_ref = if !loaded_sc_peaks.is_empty() {
            if !self.systems.is_empty() {
                let locked_pairs: Vec<(usize, usize)> = self
                    .session
                    .state
                    .locked_residues
                    .iter()
                    .map(|(&s, &k)| (s, k))
                    .collect();
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: locked_pairs,
                    ..ScoreConfig::default()
                };
                let acfg = AnnealConfig {
                    runs: self.replicas.clamp(4, 16),
                    steps_per_system: self.steps_per_sys,
                    seed,
                    ..AnnealConfig::default()
                };
                let joint = solve_joint_backbone_sidechain(
                    &self.systems,
                    &self.sequence,
                    &self.seq_ids,
                    &self.priors,
                    &self.sidechain_priors,
                    &loaded_sc_peaks,
                    &scfg,
                    &acfg,
                    &SidechainSolverConfig::default(),
                );
                let (_, bb_conf) = to_reference(
                    &joint.backbone_result,
                    &self.systems,
                    &self.sequence,
                    &self.seq_ids,
                    0.0,
                    Some(&self.priors),
                );
                if let Some(truth) = &self.truth {
                    self.metrics = Some(score(&joint.backbone_result, truth, 0.8));
                }
                self.assign_res = Some(joint.backbone_result);
                self.pred_conf = bb_conf;
                joint.sidechain_result.reference
            } else if let Some(base_ref) = prior_user_ref.clone() {
                let sc_res = assign_sidechains(
                    &base_ref,
                    (!self.pred_conf.is_empty()).then_some(&self.pred_conf),
                    &loaded_sc_peaks,
                    &self.sidechain_priors,
                    &SidechainSolverConfig::default(),
                );
                sc_res.reference
            } else {
                self.status_msg =
                    "Load backbone shifts or spin systems before sidechain assignment".to_string();
                return;
            }
        } else {
            if self.pred_ref.is_none() && !self.systems.is_empty() {
                self.run_assignment();
            }
            if let Some(pred) = &self.pred_ref {
                pred.clone()
            } else if self.systems.is_empty() {
                if let Some(r) = &self.reference {
                    r.clone()
                } else {
                    Reference::new(self.sequence.clone(), self.seq_ids.clone())
                }
            } else {
                Reference::new(self.sequence.clone(), self.seq_ids.clone())
            }
        };

        if !loaded_arom_peaks.is_empty() {
            working_ref = merge_gui_aromatic_with_options(
                &working_ref,
                prior_user_ref.as_ref(),
                &prior_amb,
                (!self.pred_conf.is_empty()).then_some(&self.pred_conf),
                &loaded_arom_peaks,
                &self.sidechain_priors,
                self.overwrite_curated_rings,
            );
        }

        let mut amb_codes = working_ref.ambiguity_codes.clone();
        if let Ok(stereo_out) = assign_stereospecific_dataset(&StereospecDatasetInput {
            reference: &working_ref,
            j_couplings_tsv: None,
            intra_noes_tsv: None,
            pdb_path: None,
            config: StereospecConfig::default(),
        }) {
            let preserved_amb = working_ref.ambiguity_codes.clone();
            working_ref = stereo_out.reference;
            amb_codes = stereo_out.ambiguity_codes;
            for (key, code) in preserved_amb {
                amb_codes.entry(key).or_insert(code);
            }
            working_ref.ambiguity_codes = amb_codes.clone();
        }

        // P2: Sync auto-inferred Cys redox & His Pelton tautomer variants cleanly without
        // clobbering user_variant_overrides or leaving stale auto-inferred states.
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&working_ref);

        let n_shifts = working_ref.shifts.len();
        let dt_ms = timer.elapsed_ms();
        self.pred_ref = Some(working_ref);
        self.pred_ambiguity_codes = amb_codes;
        if !loaded_sc_peaks.is_empty() || !loaded_arom_peaks.is_empty() {
            self.has_structure_or_dynamics = true;
        }
        self.show_sidechain_inspector = true;
        if self.selected_res.is_none() && !self.sequence.is_empty() {
            self.select_residue_and_sync_strip(0);
        }
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
        self.status_msg = if !loaded_sc_peaks.is_empty() && !loaded_arom_peaks.is_empty() {
            format!("Assigned {n_shifts} backbone + aliphatic + aromatic shifts in {dt_ms:.0} ms")
        } else if !loaded_sc_peaks.is_empty() {
            format!("Assigned {n_shifts} backbone + aliphatic sidechain shifts in {dt_ms:.0} ms")
        } else if !loaded_arom_peaks.is_empty() {
            format!("Assigned aromatic rings ({n_shifts} total shifts) in {dt_ms:.0} ms")
        } else {
            format!(
                "No sidechain or aromatic peak lists loaded — inspecting {n_shifts} existing shifts & BMRB Z-scores"
            )
        };
    }

    fn draw_sidechain_inspector_drawer(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            ui.label("Load a sequence or NMR-STAR file to inspect 2D covalent sidechains.");
            return;
        }
        let cur_k = self
            .selected_res
            .unwrap_or(self.sidechain_ribbon_state.selected_residue)
            .min(self.sequence.len() - 1);
        self.selected_res = Some(cur_k);
        self.sidechain_ribbon_state.selected_residue = cur_k;

        let mut trigger_sidechain_solve = false;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(cur_k > 0, egui::Button::new("◀").small())
                .on_hover_text("Previous residue")
                .clicked()
            {
                self.select_residue_and_sync_strip(cur_k - 1);
            }
            if ui
                .add_enabled(
                    cur_k + 1 < self.sequence.len(),
                    egui::Button::new("▶").small(),
                )
                .on_hover_text("Next residue")
                .clicked()
            {
                self.select_residue_and_sync_strip(cur_k + 1);
            }
            if ui
                .small_button("▶ Assign Sidechains")
                .on_hover_text(
                    "Run joint aliphatic, aromatic (Pelton His tautomer), and stereospecific assignment",
                )
                .clicked()
            {
                trigger_sidechain_solve = true;
            }
            ui.checkbox(
                &mut self.overwrite_curated_rings,
                egui::RichText::new("Overwrite curated rings").size(10.5),
            )
            .on_hover_text(
                "When unchecked (default), user-curated aromatic ring shifts (CD*, CE*, CZ*, HD*, HE*, HZ*) are protected from overwrite, partial mixing, and duplicate re-assignment",
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("✕")
                    .on_hover_text("Hide 2D covalent sidechain inspector")
                    .clicked()
                {
                    self.show_sidechain_inspector = false;
                }
            });
        });

        if trigger_sidechain_solve {
            self.run_joint_sidechain_in_gui();
        }
        if !self.show_sidechain_inspector {
            return;
        }

        let mut assigned_map: BTreeMap<(usize, String), AssignedAtomInfo> = BTreeMap::new();
        let display_ref = self.pred_ref.as_ref().or({
            if self.systems.is_empty() {
                self.reference.as_ref()
            } else {
                None
            }
        });
        if let Some(pred) = display_ref {
            for (&(rk, ref atom), &ppm) in &pred.shifts {
                let amb = self.pred_ambiguity_codes.get(&(rk, atom.clone())).copied();
                assigned_map.insert(
                    (rk, atom.clone()),
                    AssignedAtomInfo::with_metadata(ppm, None, amb),
                );
            }
        }
        let locked_set = self
            .session
            .state
            .locked_residues
            .values()
            .copied()
            .collect();

        let out = {
            let sc_input = SidechainWorkspaceInput::from_assigned_map(
                &self.sequence,
                &self.seq_ids,
                assigned_map,
                &self.priors,
                &self.sidechain_priors,
            )
            .with_backbone_confidences(self.pred_conf.clone())
            .with_locked_residues(locked_set);
            SidechainInspectorPanel::show_inspector(ui, &mut self.sidechain_ribbon_state, &sc_input)
        };

        if out.selected_residue < self.sequence.len()
            && self.selected_res != Some(out.selected_residue)
        {
            self.select_residue_and_sync_strip(out.selected_residue);
        }
        if let Some(jump) = out.jump_to_atom {
            self.select_residue_and_sync_strip(jump.residue_idx);
            jump.apply_to_crosshair(&mut self.crosshair);
            self.status_msg = format!(
                "Jumped crosshair to {} {} {} ({:.3} ppm)",
                jump.residue_name, jump.seq_id, jump.atom_name, jump.ppm
            );
        }
    }

    fn draw_propensity_inspector_drawer(&mut self, ui: &mut egui::Ui) {
        if self.propensity_report.is_none() && (self.pred_ref.is_some() || self.reference.is_some())
        {
            self.refresh_propensity_report();
        }
        let Some(rep) = self.propensity_report.clone() else {
            ui.label(
                "Load an NMR-STAR shift file or run assignment to compute Marsh SSP & Camilloni 4-state δ2D populations.",
            );
            return;
        };
        if rep.residues.is_empty() {
            ui.label("No residues available for SSP / δ2D conformational analysis.");
            return;
        }

        let sel_k = self
            .selected_res
            .unwrap_or(0)
            .min(rep.residues.len().saturating_sub(1));

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(20, 26, 37))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("SSP & 4-State δ2D Ensemble")
                            .strong()
                            .color(egui::Color32::from_rgb(220, 195, 255)),
                    );
                    let badge = if rep.is_predominantly_disordered {
                        ("IDP / Disordered", egui::Color32::from_rgb(186, 132, 255))
                    } else {
                        ("Folded / Structured", egui::Color32::from_rgb(92, 225, 175))
                    };
                    ui.label(egui::RichText::new(badge.0).small().color(badge.1));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("✕")
                            .on_hover_text("Hide SSP / δ2D propensity drawer")
                            .clicked()
                        {
                            self.show_propensity_panel = false;
                        }
                        if ui
                            .small_button("↻")
                            .on_hover_text("Recompute SSP / δ2D with current T(K) & pH")
                            .clicked()
                        {
                            self.refresh_propensity_report();
                        }
                    });
                });

                ui.label(
                    egui::RichText::new(format!(
                        "T={:.1} K, pH={:.2} | α {:.0}% · β {:.0}% · PPII {:.0}% · Coil {:.0}% | <|SSP|> {:.2}",
                        rep.temperature_kelvin,
                        rep.effective_ph,
                        rep.mean_p_alpha * 100.0,
                        rep.mean_p_beta * 100.0,
                        rep.mean_p_ppii * 100.0,
                        rep.mean_p_coil * 100.0,
                        rep.mean_abs_ssp
                    ))
                    .monospace()
                    .size(10.5)
                    .color(egui::Color32::from_rgb(175, 192, 218)),
                );

                ui.add_space(4.0);
                let avail_w = ui.available_width().max(220.0);
                let n = rep.residues.len().max(1);
                let col_w = (avail_w / (n as f32)).clamp(2.0, 14.0);
                let plot_w = (col_w * (n as f32)).max(avail_w);

                egui::ScrollArea::horizontal()
                    .id_salt("ssp_d2d_scroll_canvas")
                    .show(ui, |ui| {
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(plot_w, 114.0),
                            egui::Sense::click_and_drag(),
                        );
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(14, 18, 26));

                        let ssp_top = rect.top() + 4.0;
                        let ssp_mid = rect.top() + 34.0;
                        let ssp_bot = rect.top() + 64.0;
                        let pop_top = rect.top() + 72.0;
                        let pop_bot = rect.bottom() - 4.0;
                        let pop_h = (pop_bot - pop_top).max(12.0);

                        painter.line_segment(
                            [egui::pos2(rect.left(), ssp_mid), egui::pos2(rect.right(), ssp_mid)],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(62, 74, 96)),
                        );

                        for (k, r) in rep.residues.iter().enumerate() {
                            let x0 = rect.left() + (k as f32) * col_w;
                            let x1 = (x0 + col_w - 0.8).max(x0 + 1.0);
                            if k == sel_k {
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x0, rect.top()),
                                        egui::pos2(x0 + col_w, rect.bottom()),
                                    ),
                                    0.0,
                                    egui::Color32::from_rgba_unmultiplied(120, 215, 255, 28),
                                );
                            }

                            if let Some(ssp) = r.ssp_score {
                                let clamped = (ssp as f32).clamp(-1.0, 1.0);
                                let half_h = (ssp_bot - ssp_mid).max(8.0);
                                let (y_min, y_max, col) = if clamped >= 0.0 {
                                    (
                                        ssp_mid - clamped * (ssp_mid - ssp_top),
                                        ssp_mid,
                                        egui::Color32::from_rgb(56, 189, 248),
                                    )
                                } else {
                                    (
                                        ssp_mid,
                                        ssp_mid + (-clamped) * half_h,
                                        egui::Color32::from_rgb(245, 158, 11),
                                    )
                                };
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x0, y_min),
                                        egui::pos2(x1, y_max.max(y_min + 1.0)),
                                    ),
                                    1.0,
                                    col,
                                );
                            }

                            // Stacked 4-state δ2D bar: Alpha (cyan), Beta (amber), PPII (violet), Coil (slate)
                            let mut y_cur = pop_bot;
                            for (frac, col) in [
                                (r.p_alpha as f32, egui::Color32::from_rgb(56, 189, 248)),
                                (r.p_beta as f32, egui::Color32::from_rgb(245, 158, 11)),
                                (r.p_ppii as f32, egui::Color32::from_rgb(168, 85, 247)),
                                (r.p_coil as f32, egui::Color32::from_rgb(71, 85, 105)),
                            ] {
                                let h = (frac.clamp(0.0, 1.0) * pop_h).max(0.0);
                                if h > 0.3 {
                                    painter.rect_filled(
                                        egui::Rect::from_min_max(
                                            egui::pos2(x0, (y_cur - h).max(pop_top)),
                                            egui::pos2(x1, y_cur),
                                        ),
                                        0.0,
                                        col,
                                    );
                                    y_cur = (y_cur - h).max(pop_top);
                                }
                            }
                        }

                        if resp.clicked() || resp.dragged() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                let idx = (((pos.x - rect.left()) / col_w).floor() as isize)
                                    .clamp(0, (n as isize) - 1) as usize;
                                self.select_residue_and_sync_strip(idx);
                            }
                        }
                    });

                if let Some(cur) = rep.residues.get(sel_k) {
                    let ssp_txt = cur
                        .ssp_score
                        .map(|v| format!("{v:+.3}"))
                        .unwrap_or_else(|| "n/a".to_string());
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {} ({}) | SSP {} | α {:.0}% β {:.0}% PPII {:.0}% coil {:.0}%",
                            cur.residue,
                            cur.seq_id,
                            cur.dominant_state.label(),
                            ssp_txt,
                            cur.p_alpha * 100.0,
                            cur.p_beta * 100.0,
                            cur.p_ppii * 100.0,
                            cur.p_coil * 100.0,
                        ))
                        .monospace()
                        .size(10.5)
                        .color(egui::Color32::from_rgb(215, 232, 252)),
                    );
                }
            });
    }

    fn draw_right_inspector(&mut self, ui: &mut egui::Ui) {
        // 1. Metrics summary card if reference truth is present
        if let Some(m) = &self.metrics {
            egui::Frame::group(ui.style())
                .fill(egui::Color32::from_rgb(22, 29, 40))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Benchmark vs Truth").strong());
                        if let Some(ms) = self.last_solve_ms {
                            ui.label(
                                egui::RichText::new(format!("{ms:.0} ms"))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(120, 225, 255)),
                            );
                        }
                    });
                    ui.columns(3, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Accuracy").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.accuracy() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Recall").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.recall() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[2].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("HC Prec (>=0.8)").small().weak());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.1}% ({}/{})",
                                    m.hc_precision() * 100.0,
                                    m.hc_correct,
                                    m.hc_assigned
                                ))
                                .strong()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(95, 240, 165)),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        }

        // 2. Strip View Mode Switch: Match-Finder (`st`), Sequential Walk, & Live 2D Sidechain Graph
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.use_match_finder_strips,
                true,
                "🎯 Match-Finder (i−1/i/i+1)",
            );
            ui.selectable_value(&mut self.use_match_finder_strips, false, "🧬 Walk & LLR");
            let sc_btn = egui::Button::new(
                egui::RichText::new("🔬 Sidechain 2D")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_sidechain_inspector {
                egui::Color32::from_rgb(20, 108, 130)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(sc_btn)
                .on_hover_text("Inspect 2D IUPAC covalent atom-bond BMRB Z-score graph")
                .clicked()
            {
                self.show_sidechain_inspector = !self.show_sidechain_inspector;
                if self.show_sidechain_inspector && self.selected_res.is_none() {
                    self.selected_res = Some(0);
                }
            }
            let prop_btn = egui::Button::new(
                egui::RichText::new("📊 SSP / δ2D")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_propensity_panel {
                egui::Color32::from_rgb(104, 48, 136)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(prop_btn)
                .on_hover_text("Inspect Marsh SSP & Camilloni 4-state δ2D populations")
                .clicked()
            {
                self.show_propensity_panel = !self.show_propensity_panel;
                if self.show_propensity_panel {
                    self.refresh_propensity_report();
                    if self.selected_res.is_none() {
                        self.selected_res = Some(0);
                    }
                }
            }
            if ui.small_button("📌 Assign/Lock (at)").clicked() {
                self.show_assign_popover = true;
            }
        });

        if self.show_propensity_panel {
            ui.add_space(4.0);
            self.draw_propensity_inspector_drawer(ui);
            ui.add_space(6.0);
            ui.separator();
        }

        if self.show_sidechain_inspector {
            ui.add_space(4.0);
            self.draw_sidechain_inspector_drawer(ui);
            ui.add_space(6.0);
            ui.separator();
        }

        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            ui.label("Click any HSQC peak or assigned residue to inspect 3D strips.");
            return;
        };

        if self.use_match_finder_strips {
            if let Some(p_mat) = self.p_mat.clone() {
                let u_clone = self.u_mat.clone();
                let assign_pos = self.assign_res.as_ref().map(|r| r.position.clone());
                let locks_snapshot = self.session.state.locked_residues.clone();
                let links_snapshot = self.session.state.confirmed_links.clone();
                let strip_actions = {
                    let tiler_ctx = StripTilerContext {
                        systems: &self.systems,
                        pair_mat: &p_mat,
                        u_mat: u_clone.as_ref(),
                        sequence: &self.sequence,
                        seq_ids: &self.seq_ids,
                        active_root: sel_s,
                        assignments: assign_pos.as_deref(),
                        locks: Some(&locks_snapshot),
                        confirmed_links: Some(&links_snapshot),
                        spectra_3d: StripSpectra3D {
                            aliphatic_3d: self.hncacb_3d.as_ref(),
                            carbonyl_3d: self.hnco_3d.as_ref(),
                        },
                    };
                    self.strip_tiler
                        .show(ui, &tiler_ctx, Some(&mut self.crosshair))
                };
                for act in strip_actions {
                    self.handle_strip_tiler_action(act);
                }
                ui.separator();
            }
        } else {
            ui.label(
                egui::RichText::new(
                    "Cyan = +Cα, Coral = −Cβ, Gold ◆ = (i−1), Dashed = sequential tie",
                )
                .small()
                .weak(),
            );
            let (pred_s, succ_s) = self.find_walk_neighbors(sel_s);
            let strip_list: Vec<(Option<usize>, &str)> = vec![
                (pred_s, "i − 1"),
                (Some(sel_s), "Selected (i)"),
                (succ_s, "i + 1"),
            ];
            let strip_h = 275.0f32;
            let (strip_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(240.0), strip_h),
                egui::Sense::hover(),
            );
            self.paint_sequential_strips(ui.painter_at(strip_rect), strip_rect, &strip_list);
            ui.add_space(6.0);
            ui.separator();
        }
        ui.add_space(8.0);
        ui.separator();

        // 3. Detailed Spin System Candidate & LLR / Connectivity Inspector
        egui::ScrollArea::vertical()
            .id_salt("ss_inspector_scroll")
            .show(ui, |ui| {
                let ss = &self.systems[sel_s];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Spin System #{sel_s}  (¹H {:.3}, ¹⁵N {:.2})",
                            ss.h, ss.n
                        ))
                        .strong(),
                    );
                    if let Some(res) = &self.assign_res {
                        let k = res.position[sel_s];
                        if k >= 0 && (k as usize) < self.sequence.len() {
                            let ku = k as usize;
                            ui.label(
                                egui::RichText::new(format!(
                                    "→ {}{} ({:.0}%)",
                                    self.sequence[ku],
                                    self.seq_ids[ku],
                                    res.confidence[sel_s] * 100.0
                                ))
                                .color(egui::Color32::from_rgb(95, 240, 165))
                                .strong(),
                            );
                        }
                    }
                });

                egui::Grid::new("cands_grid")
                    .striped(true)
                    .min_col_width(58.0)
                    .show(ui, |ui| {
                        ui.strong("Atom");
                        ui.strong("Offset");
                        ui.strong("Candidates (ppm × support)");
                        ui.end_row();

                        for atom in ["CA", "CB", "C"] {
                            for (off, label) in [(0, "i"), (-1, "i − 1")] {
                                if let Some(cands) = ss.cands.get(&(atom.to_string(), off)) {
                                    ui.monospace(atom);
                                    ui.monospace(label);
                                    let text = cands
                                        .iter()
                                        .map(|(v, sup)| format!("{v:.2} (×{sup})"))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    ui.monospace(text);
                                    ui.end_row();
                                }
                            }
                        }
                    });

                if let Some(u_mat) = &self.u_mat {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Top Residue Typing LLR U(s, k):")
                            .small()
                            .strong(),
                    );
                    let mut ranked: Vec<(usize, f64)> = (0..u_mat.cols)
                        .map(|k| (k, u_mat.get(sel_s, k)))
                        .filter(|&(_, v)| v > FORBID * 0.5)
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let top_txt = if ranked.is_empty() {
                        "0 observed carbons (excluded from backbone)".to_string()
                    } else {
                        ranked
                            .iter()
                            .take(5)
                            .map(|&(k, sc)| {
                                format!("{}{}: {:+.1}", self.sequence[k], self.seq_ids[k], sc)
                            })
                            .collect::<Vec<_>>()
                            .join("  |  ")
                    };
                    ui.monospace(top_txt);
                }

                if let Some(p_mat) = &self.p_mat {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Best Sequential Successors P(s → t):")
                            .small()
                            .strong(),
                    );
                    let mut succs: Vec<(usize, f64)> = (0..p_mat.cols)
                        .filter(|&t| t != sel_s)
                        .map(|t| (t, p_mat.get(sel_s, t)))
                        .filter(|&(_, v)| v > -20.0 && v.abs() > 1e-6)
                        .collect();
                    succs.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (t, sc) in succs.into_iter().take(4) {
                        if ui
                            .small_button(format!("Jump → SS #{t} (P = {sc:+.2})"))
                            .clicked()
                        {
                            self.selected_ss = Some(t);
                        }
                    }
                }
            });
    }

    fn find_walk_neighbors(&self, sel_s: usize) -> (Option<usize>, Option<usize>) {
        if let Some(res) = &self.assign_res {
            let k = res.position[sel_s];
            if k >= 0 {
                let pred = res.position.iter().position(|&pk| pk == k - 1);
                let succ = res.position.iter().position(|&pk| pk == k + 1);
                return (pred, succ);
            }
        }
        if let Some(p_mat) = &self.p_mat {
            let pred = (0..p_mat.rows)
                .filter(|&a| a != sel_s)
                .max_by(|&a, &b| p_mat.get(a, sel_s).total_cmp(&p_mat.get(b, sel_s)))
                .filter(|&a| p_mat.get(a, sel_s) > 0.0);
            let succ = (0..p_mat.cols)
                .filter(|&b| b != sel_s)
                .max_by(|&a, &b| p_mat.get(sel_s, a).total_cmp(&p_mat.get(sel_s, b)))
                .filter(|&b| p_mat.get(sel_s, b) > 0.0);
            return (pred, succ);
        }
        (None, None)
    }

    fn paint_sequential_strips(
        &self,
        painter: egui::Painter,
        rect: egui::Rect,
        strips: &[(Option<usize>, &str)],
    ) {
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(13, 17, 24));
        let c_top = 13.0f64;
        let c_bot = 74.0f64;
        let top_pad = 22.0f32;
        let bot_pad = 8.0f32;
        let left_axis_w = 34.0f32;

        let plot_top = rect.top() + top_pad;
        let plot_bot = rect.bottom() - bot_pad;
        let c_to_y = |c_ppm: f64| -> f32 {
            let t = ((c_ppm - c_top) / (c_bot - c_top)) as f32;
            plot_top + t.clamp(0.0, 1.0) * (plot_bot - plot_top)
        };

        // 13C ppm scale on left
        for c_tick in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 72.0] {
            let y = c_to_y(c_tick);
            painter.text(
                egui::pos2(rect.left() + 3.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{c_tick:.0}"),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(145, 165, 190),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.left() + left_axis_w, y),
                    egui::pos2(rect.right() - 4.0, y),
                ],
                egui::Stroke::new(0.5, egui::Color32::from_rgb(28, 35, 48)),
            );
        }

        let total_w = (rect.width() - left_axis_w - 6.0).max(90.0);
        let col_w = total_w / (strips.len() as f32);

        let mut col_centers = Vec::with_capacity(strips.len());

        for (idx, &(ss_opt, role)) in strips.iter().enumerate() {
            let x0 = rect.left() + left_axis_w + (idx as f32) * col_w + 3.0;
            let x1 = x0 + col_w - 6.0;
            let cx = 0.5 * (x0 + x1);
            col_centers.push(cx);

            let strip_box =
                egui::Rect::from_min_max(egui::pos2(x0, plot_top), egui::pos2(x1, plot_bot));
            painter.rect_stroke(
                strip_box,
                2.0,
                egui::Stroke::new(
                    if idx == 1 { 1.5 } else { 1.0 },
                    if idx == 1 {
                        egui::Color32::from_rgb(95, 200, 255)
                    } else {
                        egui::Color32::from_rgb(50, 62, 82)
                    },
                ),
                egui::StrokeKind::Inside,
            );

            let Some(s) = ss_opt else {
                painter.text(
                    egui::pos2(cx, rect.top() + 10.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{role}: —"),
                    egui::FontId::monospace(10.0),
                    egui::Color32::GRAY,
                );
                continue;
            };
            let ss = &self.systems[s];

            // Header title (assigned residue or SS#)
            let title = self
                .assign_res
                .as_ref()
                .and_then(|r| (r.position[s] >= 0).then(|| r.position[s] as usize))
                .map(|k| {
                    format!(
                        "{}{} (#{s})",
                        three_to_one(&self.sequence[k]),
                        self.seq_ids[k]
                    )
                })
                .unwrap_or_else(|| format!("SS #{s}"));
            painter.text(
                egui::pos2(cx, rect.top() + 10.0),
                egui::Align2::CENTER_CENTER,
                title,
                egui::FontId::monospace(10.5),
                egui::Color32::from_rgb(225, 238, 255),
            );

            // Render 3D HNCACB strip slice intensity heatmap if available
            if let Some(s3) = &self.hncacb_3d {
                let slice = s3.extract_strip(ss.h, ss.n, 0.04);
                let max_v = slice.max_abs().max(1.0);
                for ic in 0..slice.ny() {
                    let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                    if !(c_top..=c_bot).contains(&c_ppm) {
                        continue;
                    }
                    let y = c_to_y(c_ppm);
                    let mid_x = slice.nx() / 2;
                    let val = slice.get(mid_x, ic);
                    let norm = (val.abs() / max_v).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(
                                40,
                                195,
                                255,
                                (norm * 130.0) as u8,
                            )
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                95,
                                105,
                                (norm * 130.0) as u8,
                            )
                        };
                        let half_bar = norm * (strip_box.width() * 0.36);
                        painter.line_segment(
                            [egui::pos2(cx - half_bar, y), egui::pos2(cx + half_bar, y)],
                            egui::Stroke::new(2.2, col),
                        );
                    }
                }
            }

            // Plot resolved spin-system candidates in strip: (i) solid circles, (i-1) diamonds
            for atom in ["CA", "CB"] {
                let base_col = if atom == "CA" {
                    egui::Color32::from_rgb(80, 230, 255)
                } else {
                    egui::Color32::from_rgb(255, 115, 120)
                };
                if let Some(cands_i) = ss.cands.get(&(atom.to_string(), 0)) {
                    if let Some(&(ppm, sup)) = cands_i.first() {
                        if sup > 0 {
                            let y = c_to_y(ppm);
                            painter.circle_filled(egui::pos2(cx - 8.0, y), 4.2, base_col);
                            painter.text(
                                egui::pos2(cx - 14.0, y),
                                egui::Align2::RIGHT_CENTER,
                                format!("{ppm:.1}"),
                                egui::FontId::monospace(8.8),
                                base_col,
                            );
                        }
                    }
                }
                if let Some(cands_im1) = ss.cands.get(&(atom.to_string(), -1)) {
                    if let Some(&(ppm, _)) = cands_im1.first() {
                        let y = c_to_y(ppm);
                        let d_col = egui::Color32::from_rgb(250, 205, 90);
                        let pts = vec![
                            egui::pos2(cx + 8.0, y - 4.2),
                            egui::pos2(cx + 12.2, y),
                            egui::pos2(cx + 8.0, y + 4.2),
                            egui::pos2(cx + 3.8, y),
                        ];
                        painter.add(egui::Shape::convex_polygon(pts, d_col, egui::Stroke::NONE));
                    }
                }
            }
        }

        // Draw sequential connectivity tie-lines between strip[w] (offset 0) and strip[w+1] (offset -1)
        for w in 0..strips.len().saturating_sub(1) {
            let (Some(sa), Some(sb)) = (strips[w].0, strips[w + 1].0) else {
                continue;
            };
            for atom in ["CA", "CB"] {
                let Some(va) = self.systems[sa]
                    .cands
                    .get(&(atom.to_string(), 0))
                    .and_then(|c| c.first())
                    .filter(|c| c.1 > 0)
                    .map(|c| c.0)
                else {
                    continue;
                };
                let Some(vb) = self.systems[sb]
                    .cands
                    .get(&(atom.to_string(), -1))
                    .and_then(|c| c.first())
                    .map(|c| c.0)
                else {
                    continue;
                };
                if (va - vb).abs() <= 0.45 {
                    let ya = c_to_y(va);
                    let yb = c_to_y(vb);
                    let x_start = col_centers[w] - 4.0;
                    let x_end = col_centers[w + 1] + 4.0;
                    let col = if atom == "CA" {
                        egui::Color32::from_rgba_unmultiplied(110, 245, 255, 210)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 145, 145, 210)
                    };
                    // Dashed horizontal connector
                    let segs = 9;
                    for j in (0..segs).step_by(2) {
                        let t0 = (j as f32) / (segs as f32);
                        let t1 = ((j + 1) as f32) / (segs as f32);
                        painter.line_segment(
                            [
                                egui::pos2(x_start + t0 * (x_end - x_start), ya + t0 * (yb - ya)),
                                egui::pos2(x_start + t1 * (x_end - x_start), ya + t1 * (yb - ya)),
                            ],
                            egui::Stroke::new(1.5, col),
                        );
                    }
                }
            }
        }
    }
    fn compute_workspace_readiness(&self) -> WorkspaceReadiness {
        let spectra_2d_count = usize::from(self.hsqc_spec.is_some()) + self.overlay_layers.len();
        let spectra_3d_count = usize::from(self.hncacb_3d.is_some())
            + usize::from(self.hnco_3d.is_some())
            + self.extra_3d_count;
        let assigned_residues_count = self.pred_conf.len();
        let high_confidence_count = self.pred_conf.values().filter(|&&c| c >= 0.80).count();
        let qc_verdict = self
            .qc_report
            .as_ref()
            .map(|q| q.verdict)
            .or(self.session.state.qc_verdict);
        let qc_summary = self.qc_report.as_ref().map(|q| {
            format!(
                "{:.0}% 3D support, scale {:.2}",
                q.hsqc_3d_support_frac * 100.0,
                q.confidence_scale
            )
        });

        WorkspaceReadiness {
            sequence_loaded: !self.sequence.is_empty(),
            sequence_len: self.sequence.len(),
            raw_fids_count: self.raw_fids_total,
            unprocessed_fids_count: self.unprocessed_fids_remaining,
            spectra_2d_count,
            spectra_3d_count,
            picked_peaks_count: self.peaks.len(),
            spin_systems_count: self.systems.len(),
            assigned_residues_count,
            high_confidence_count,
            locked_residues_count: self.session.state.locked_residues.len(),
            has_structure_or_dynamics: self.has_structure_or_dynamics,
            exported_in_session: self.exported_in_session,
            qc_verdict,
            qc_summary,
        }
    }

    fn handle_strip_tiler_action(&mut self, act: StripTilerAction) {
        match act {
            StripTilerAction::SelectRoot { spin_system_idx } => {
                if let Some(ss) = self.systems.get(spin_system_idx) {
                    self.selected_ss = Some(spin_system_idx);
                    self.crosshair.snap_to(ss.h, ss.n, None);
                    if let Some(res) = &self.assign_res {
                        if res.position.get(spin_system_idx).copied().unwrap_or(-1) >= 0 {
                            let k = res.position[spin_system_idx] as usize;
                            self.selected_res = Some(k);
                            self.assign_target_res = k;
                            self.mol3d_panel.highlight_sequence_residue(Some(k), true);
                        }
                    }
                }
            }
            StripTilerAction::ConfirmSequentialLink { pred_idx, succ_idx } => {
                let pair = (pred_idx, succ_idx);
                if let Some(pos) = self
                    .session
                    .state
                    .confirmed_links
                    .iter()
                    .position(|&lk| lk == pair)
                {
                    self.session.state.confirmed_links.remove(pos);
                    self.session.dirty = true;
                    self.status_msg =
                        format!("Removed sequential link SS #{pred_idx} → SS #{succ_idx}");
                } else {
                    self.session.state.confirmed_links.push(pair);
                    self.session.dirty = true;
                    self.status_msg = format!(
                        "Confirmed sequential link SS #{pred_idx} (i−1) → SS #{succ_idx} (i)"
                    );
                }
            }
            StripTilerAction::LockToResidue {
                spin_system_idx,
                residue_seq_idx,
            } => {
                if residue_seq_idx < self.sequence.len() {
                    let already = self
                        .session
                        .state
                        .locked_residue_for_spin_system(spin_system_idx)
                        == Some(residue_seq_idx);
                    let msg = self.session.execute(WorkbenchCommand::LockResidue {
                        spin_system: spin_system_idx,
                        residue_seq_idx,
                        locked: !already,
                    });
                    self.sync_app_from_session();
                    self.status_msg = msg;
                }
            }
        }
    }

    fn snap_selected_peak_parabolic(&mut self) {
        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            self.status_msg = "Select a peak / spin system first (`pc`)".to_string();
            return;
        };
        let ss = &self.systems[sel_s];
        let Some((pk_idx, pk)) = self
            .peaks
            .iter()
            .enumerate()
            .filter(|(_, p)| p.exp == "HSQC")
            .min_by(|(_, a), (_, b)| {
                let da = ((a.h - ss.h) / 0.03).powi(2) + ((a.n - ss.n) / 0.3).powi(2);
                let db = ((b.h - ss.h) / 0.03).powi(2) + ((b.n - ss.n) / 0.3).powi(2);
                da.total_cmp(&db)
            })
            .map(|(i, p)| (i, p.clone()))
        else {
            self.status_msg = "No HSQC peak found near active spin system".to_string();
            return;
        };

        let (mut fitted_h, mut fitted_n, mut new_height) = (pk.h, pk.n, pk.height);
        if let Some(spec) = &self.hsqc_spec {
            let ix = spec
                .axis_x
                .ppm_to_index(pk.h)
                .round()
                .clamp(1.0, (spec.nx().saturating_sub(2)) as f64) as usize;
            let iy = spec
                .axis_y
                .ppm_to_index(pk.n)
                .round()
                .clamp(1.0, (spec.ny().saturating_sub(2)) as f64) as usize;
            // Local 3x3 extremum search + 1D parabolic sub-voxel interpolation
            let mut best_ix = ix;
            let mut best_iy = iy;
            let mut best_val = spec.get(ix, iy).abs();
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let cx = (ix as isize + dx).clamp(1, (spec.nx().saturating_sub(2)) as isize)
                        as usize;
                    let cy = (iy as isize + dy).clamp(1, (spec.ny().saturating_sub(2)) as isize)
                        as usize;
                    let v = spec.get(cx, cy).abs();
                    if v > best_val {
                        best_val = v;
                        best_ix = cx;
                        best_iy = cy;
                    }
                }
            }
            let xm = spec.get(best_ix - 1, best_iy) as f64;
            let x0 = spec.get(best_ix, best_iy) as f64;
            let xp = spec.get(best_ix + 1, best_iy) as f64;
            let denom_x = xm - 2.0 * x0 + xp;
            let dx = if denom_x.abs() > 1e-9 {
                (0.5 * (xm - xp) / denom_x).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let ym = spec.get(best_ix, best_iy - 1) as f64;
            let yp = spec.get(best_ix, best_iy + 1) as f64;
            let denom_y = ym - 2.0 * x0 + yp;
            let dy = if denom_y.abs() > 1e-9 {
                (0.5 * (ym - yp) / denom_y).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            fitted_h = spec.axis_x.index_to_ppm(best_ix as f64 + dx);
            fitted_n = spec.axis_y.index_to_ppm(best_iy as f64 + dy);
            new_height = Some(x0);
        }

        let msg = self.session.execute(WorkbenchCommand::RecenterPeak {
            index: pk_idx,
            fitted_h,
            fitted_n,
            fitted_c: None,
            new_height,
        });
        self.sync_app_from_session();
        self.crosshair.snap_to(fitted_h, fitted_n, None);
        self.status_msg = msg;
    }

    fn save_session_nef_action(&mut self, ctx: &egui::Context) {
        self.session.state.sequence = self.sequence.clone();
        self.session.state.seq_ids = self.seq_ids.clone();
        self.session.state.peaks = self.peaks.clone();
        self.session.state.view_bounds_ppm = Some([
            self.view_h_left as f64,
            self.view_h_right as f64,
            self.view_n_top as f64,
            self.view_n_bottom as f64,
        ]);
        let mut hsqc_params = SessionContourParams::new("HSQC");
        hsqc_params.base_level = self.contour_base;
        hsqc_params.level_factor = self.contour_factor;
        hsqc_params.num_levels = self.contour_levels;
        hsqc_params.show_negative = self.show_negative;
        hsqc_params.ppm_nudge = self.primary_ppm_nudge;
        self.session.state.contour_layers = vec![hsqc_params];

        let nef_text = self.session.save_to_nef_string();
        self.export_text_buf = nef_text.clone();
        self.exported_in_session = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            let p = Path::new(self.export_prefix.trim()).with_extension("nef");
            match self.session.save_to_path(&p) {
                Ok(msg) => self.status_msg = msg,
                Err(e) => self.status_msg = format!("NEF session save failed: {e}"),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let fname = default_session_nef_filename(&self.session.state.protein_name);
            let _ = trigger_browser_nef_download(&fname, nef_text.as_bytes());
            ctx.copy_text(nef_text);
            self.status_msg = format!("Downloaded lossless NEF session '{fname}'");
        }
    }

    fn execute_ingest_plan(&mut self, plan: IngestExecutionPlan, run_full_pipeline: bool) {
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        if !plan.sequence_three_letter.is_empty() {
            self.sequence = plan.sequence_three_letter.clone();
            self.seq_ids = (1..=(self.sequence.len() as i32)).collect();
            self.octet_app
                .set_loaded_protein_sequence(self.sequence.clone());
        }
        self.truth = None;
        self.metrics = None;
        self.raw_fids_total = plan
            .selected_experiments
            .iter()
            .filter(|s| s.experiment.data_state == DataState::RawFid)
            .count();
        self.unprocessed_fids_remaining = self.raw_fids_total;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut primary_2d: Option<Spectrum2D> = None;
            let mut overlays_2d: Vec<(String, Spectrum2D)> = Vec::new();
            let mut cubes_3d: Vec<Spectrum3D> = Vec::new();
            let mut processed_fids = 0usize;

            for item in &plan.selected_experiments {
                let Some(ref path) = item.resolved_path else {
                    continue;
                };
                let exp_token = item.solver_exp_name.unwrap_or("HNCACB");
                match item.experiment.dimensionality {
                    ExperimentDimensionality::Dim2D => {
                        let loaded_2d = if item.experiment.data_state == DataState::RawFid {
                            match process_fid_dir_2d(path, false, false) {
                                Ok((s2, recipe)) => {
                                    processed_fids += 1;
                                    self.session
                                        .state
                                        .processing_recipes
                                        .insert(item.experiment.display_name.clone(), recipe);
                                    Some(s2)
                                }
                                Err(_) => None,
                            }
                        } else {
                            match item.experiment.vendor {
                                ExperimentVendor::SparkyUcsf => read_ucsf_2d(path).ok(),
                                ExperimentVendor::NmrPipe => read_nmrpipe_2d(path).ok(),
                                ExperimentVendor::Bruker => {
                                    let pdata = path.join("pdata").join("1");
                                    read_bruker_pdata_2d(if pdata.exists() { &pdata } else { path })
                                        .ok()
                                }
                                _ => None,
                            }
                        };
                        if let Some(s2) = loaded_2d {
                            if primary_2d.is_none() && exp_token == "HSQC" {
                                primary_2d = Some(s2);
                            } else {
                                overlays_2d.push((item.experiment.display_name.clone(), s2));
                            }
                        }
                    }
                    ExperimentDimensionality::Dim3D => {
                        let loaded_3d = if item.experiment.data_state == DataState::RawFid {
                            match process_fid_dir_3d(path, exp_token, false, false) {
                                Ok((s3, recipe)) => {
                                    processed_fids += 1;
                                    self.session
                                        .state
                                        .processing_recipes
                                        .insert(item.experiment.display_name.clone(), recipe);
                                    Some(s3)
                                }
                                Err(_) => None,
                            }
                        } else {
                            match item.experiment.vendor {
                                ExperimentVendor::SparkyUcsf => read_ucsf_3d(path, exp_token).ok(),
                                ExperimentVendor::NmrPipe => read_nmrpipe_3d(path, exp_token).ok(),
                                ExperimentVendor::Bruker => {
                                    let pdata = path.join("pdata").join("1");
                                    read_bruker_pdata_3d(
                                        if pdata.exists() { &pdata } else { path },
                                        exp_token,
                                    )
                                    .ok()
                                }
                                _ => None,
                            }
                        };
                        if let Some(s3) = loaded_3d {
                            cubes_3d.push(s3);
                        }
                    }
                    _ => {}
                }
            }

            self.unprocessed_fids_remaining = self.raw_fids_total.saturating_sub(processed_fids);

            if primary_2d.is_none() && !overlays_2d.is_empty() {
                primary_2d = Some(overlays_2d.remove(0).1);
            }
            if let Some(hsqc) = primary_2d {
                for s3 in &mut cubes_3d {
                    let _ = register_3d_hn_to_hsqc_2d(s3, &hsqc, 0.08, 0.60);
                }
                self.hsqc_arc = Arc::new(hsqc.data.clone());
                self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                self.hsqc_spec = Some(hsqc);
                self.reset_view_to_spec();
            }

            self.overlay_layers.clear();
            for (idx, (lbl, s2)) in overlays_2d.into_iter().enumerate() {
                let lid = self.next_layer_id;
                self.next_layer_id += 1;
                let base = (estimate_noise_mad(&s2.data) * 4.8).max(1.0);
                let arc = Arc::new(s2.data.clone());
                let ov = OverlayLayerSpec::from_spectrum_2d(lid, 1, &s2, arc, base, idx + 1);
                self.overlay_layers.push((lbl, ov));
            }

            let n_3d_loaded = cubes_3d.len();
            if run_full_pipeline {
                let pcfg = PickerConfig {
                    snr_threshold: self.picker_snr,
                    ..PickerConfig::default()
                };
                let mut all_peaks = Vec::new();
                if let Some(ref hsqc) = self.hsqc_spec {
                    let roots = if hsqc.axis_y.max_ppm() < 90.0 {
                        pick_c13_hsqc_2d(hsqc, &pcfg)
                    } else if self.use_neural_picker {
                        pick_hsqc_2d_neural(hsqc, &NeuralPickerConfig::default())
                    } else {
                        pick_hsqc_2d(hsqc, &pcfg)
                    };
                    let c_refs: Vec<&Spectrum3D> = cubes_3d.iter().collect();
                    if !c_refs.is_empty() && !roots.is_empty() {
                        let p3 = pick_triple_3d_two_pass(
                            &c_refs,
                            &roots,
                            &pcfg,
                            &CorroboratedRepickConfig::default(),
                        );
                        all_peaks.extend(roots);
                        all_peaks.extend(p3);
                    } else {
                        all_peaks.extend(roots);
                    }
                }
                if !all_peaks.is_empty() {
                    self.peaks = all_peaks;
                }
            }

            self.hncacb_3d = None;
            self.hnco_3d = None;
            for s3 in cubes_3d {
                if s3.exp == "HNCO" || s3.exp == "HNCACO" {
                    if self.hnco_3d.is_none() {
                        self.hnco_3d = Some(s3);
                    }
                } else if self.hncacb_3d.is_none() {
                    self.hncacb_3d = Some(s3);
                }
            }
            let stored_3d =
                usize::from(self.hncacb_3d.is_some()) + usize::from(self.hnco_3d.is_some());
            self.extra_3d_count = n_3d_loaded.saturating_sub(stored_3d);
        }

        self.sync_session_from_app_fresh(&plan.root_name);
        self.rebuild_systems_and_matrices();
        if run_full_pipeline && !self.sequence.is_empty() && !self.peaks.is_empty() {
            self.run_assignment();
        } else {
            self.status_msg = format!(
                "Loaded '{}' ({} selected experiment(s), {} peaks, {} aa)",
                plan.root_name,
                plan.selected_experiments.len(),
                self.peaks.len(),
                self.sequence.len()
            );
        }
    }

    fn dispatch_workbench_action(&mut self, ctx: &egui::Context, action: WorkbenchAction) {
        match action {
            // --- 1. File & Ingest ---
            WorkbenchAction::OpenIngestMatrix | WorkbenchAction::LoadSequenceFasta => {
                self.ingest_panel.open = true;
            }
            WorkbenchAction::LoadPathOrSpectrum => {
                #[cfg(not(target_arch = "wasm32"))]
                self.load_path_auto();
                #[cfg(target_arch = "wasm32")]
                {
                    self.show_text_io_modal = true;
                }
            }
            WorkbenchAction::OpenPasteTextModal => {
                self.show_text_io_modal = true;
            }
            WorkbenchAction::LoadUbiquitinDemo => {
                self.load_demo_and_solve();
            }
            WorkbenchAction::SaveSessionNef | WorkbenchAction::SaveSessionNefAs => {
                self.save_session_nef_action(ctx);
            }
            WorkbenchAction::LoadSessionNef => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let p = Path::new(self.export_prefix.trim()).with_extension("nef");
                    if p.exists() {
                        match std::fs::read_to_string(&p) {
                            Ok(txt) => self.load_text_in_memory(&txt, &p.display().to_string()),
                            Err(e) => self.status_msg = format!("NEF load error: {e}"),
                        }
                    } else {
                        self.show_text_io_modal = true;
                        self.status_msg =
                            "Drop a .nef file onto the window or paste NEF session text"
                                .to_string();
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.show_text_io_modal = true;
                }
            }
            WorkbenchAction::ExportNmrStar | WorkbenchAction::ExportPeaksTsv => {
                self.export_results(ctx);
            }
            WorkbenchAction::ExportQualityReportHtml => {
                if let Some(pred) = self.pred_ref.clone().or_else(|| self.reference.clone()) {
                    if (self.regime_kind == GuiRegimeKind::Idp13C || self.show_propensity_panel)
                        && self.propensity_report.is_none()
                    {
                        self.refresh_propensity_report();
                    }
                    let mut inp = ReportInput::new(
                        self.session.state.protein_name.clone(),
                        pred.clone(),
                        self.pred_conf.clone(),
                    );
                    inp.n_peaks = Some(self.peaks.len());
                    inp.n_spin_systems = Some(self.systems.len());
                    inp.dihedrals = predict_csi_dihedrals(&pred);
                    if let Some(res) = &self.assign_res {
                        inp.solver_score = Some(res.score);
                        inp.systems_and_assignment = Some((self.systems.clone(), res.clone()));
                    }
                    if self.regime_kind == GuiRegimeKind::Idp13C || self.show_propensity_panel {
                        inp.propensity_report = self.propensity_report.clone();
                    }
                    let html = generate_quality_report_html(&inp);
                    self.export_text_buf = html.clone();
                    self.exported_in_session = true;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let html_p = Path::new(self.export_prefix.trim()).with_extension("html");
                        let _ = std::fs::write(&html_p, &html);
                        self.status_msg =
                            format!("Exported HTML quality report to {}", html_p.display());
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.show_text_io_modal = true;
                        self.status_msg = "Generated HTML quality report in memory".to_string();
                    }
                } else {
                    self.status_msg =
                        "Run assignment before exporting HTML quality report".to_string();
                }
            }
            WorkbenchAction::ResetWorkspace => {
                self.peaks.clear();
                self.systems.clear();
                self.sequence.clear();
                self.seq_ids.clear();
                self.assign_res = None;
                self.pred_ref = None;
                self.pred_conf.clear();
                self.pred_ambiguity_codes.clear();
                self.sidechain_ribbon_state = SequenceRibbonState::new();
                self.c13_picked_dataset = None;
                self.propensity_report = None;
                self.metrics = None;
                self.qc_report = None;
                self.overlay_layers.clear();
                self.hsqc_spec = None;
                self.hncacb_3d = None;
                self.hnco_3d = None;
                self.session = WorkbenchSessionController::default();
                self.status_msg = "Workspace reset".to_string();
            }

            // --- 2. FID Processing ---
            WorkbenchAction::ProcessAllRawFids | WorkbenchAction::ReconstructNusIstFista => {
                let plan = self.ingest_panel.build_execution_plan();
                if plan.selected_experiments.is_empty() {
                    self.ingest_panel.open = true;
                    self.status_msg =
                        "Open Ingest Matrix to select raw FIDs to process".to_string();
                } else {
                    self.execute_ingest_plan(plan, false);
                }
            }
            WorkbenchAction::Register3dToHsqc2d =>
            {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(ref hsqc) = self.hsqc_spec {
                    let mut n_reg = 0usize;
                    if let Some(ref mut s3) = self.hncacb_3d {
                        let _ = register_3d_hn_to_hsqc_2d(s3, hsqc, 0.08, 0.60);
                        n_reg += 1;
                    }
                    if let Some(ref mut s3) = self.hnco_3d {
                        let _ = register_3d_hn_to_hsqc_2d(s3, hsqc, 0.08, 0.60);
                        n_reg += 1;
                    }
                    self.status_msg = format!("Registered {n_reg} 3D cube(s) to 2D HSQC reference");
                }
            }
            WorkbenchAction::ExportProcessingRecipeJson => {
                if let Ok(js) = serde_json::to_string_pretty(&self.session.state.processing_recipes)
                {
                    self.export_text_buf = js;
                    self.show_text_io_modal = true;
                    self.status_msg = "Exported ProcessingRecipe JSON".to_string();
                }
            }
            WorkbenchAction::AutoPhaseDirect1D
            | WorkbenchAction::AutoPhaseIndirectBiphasic
            | WorkbenchAction::ToggleLinearPrediction
            | WorkbenchAction::ApplyMedianBaseline => {
                self.status_msg = format!("Configured DSP action: {}", action.metadata().title);
            }

            // --- 3. Spectrum View & Contours ---
            WorkbenchAction::OpenContourThresholdPopup => {
                self.show_contour_modal = !self.show_contour_modal;
            }
            WorkbenchAction::ZoomFullSweep => {
                self.reset_view_to_spec();
            }
            WorkbenchAction::ZoomIn | WorkbenchAction::ZoomOut => {
                let factor = if action == WorkbenchAction::ZoomIn {
                    0.75f32
                } else {
                    1.0 / 0.75f32
                };
                let ah = self.crosshair.h_ppm as f32;
                let an = self.crosshair.n_ppm as f32;
                self.view_h_left = ah + (self.view_h_left - ah) * factor;
                self.view_h_right = ah + (self.view_h_right - ah) * factor;
                self.view_n_top = an + (self.view_n_top - an) * factor;
                self.view_n_bottom = an + (self.view_n_bottom - an) * factor;
            }
            WorkbenchAction::RaiseContourFloor | WorkbenchAction::LowerContourFloor => {
                let new_base = if action == WorkbenchAction::RaiseContourFloor {
                    (self.contour_base * CONTOUR_FLOOR_STEP_RATIO).min(500.0)
                } else {
                    (self.contour_base / CONTOUR_FLOOR_STEP_RATIO).max(1e-3)
                };
                self.contour_base = new_base;
                let mut params = SessionContourParams::new("HSQC");
                params.base_level = self.contour_base;
                params.level_factor = self.contour_factor;
                params.num_levels = self.contour_levels;
                params.show_negative = self.show_negative;
                params.ppm_nudge = self.primary_ppm_nudge;
                let msg = self.session.execute(WorkbenchCommand::SetContourParams {
                    layer_id: "HSQC".to_string(),
                    params,
                });
                self.status_msg = msg;
            }
            WorkbenchAction::ToggleNegativeContours => {
                self.show_negative = !self.show_negative;
            }
            WorkbenchAction::ToggleCrosshairLock => {
                let locked = self.crosshair.toggle_lock();
                self.status_msg = if locked {
                    format!(
                        "Crosshair locked at ¹H {:.3}, ¹⁵N {:.2} ppm",
                        self.crosshair.h_ppm, self.crosshair.n_ppm
                    )
                } else {
                    "Crosshair unlocked".to_string()
                };
            }
            WorkbenchAction::OpenOverlayStackManager
            | WorkbenchAction::NudgeSpectrumReferenceOffset => {
                self.show_overlay_manager = !self.show_overlay_manager;
            }
            WorkbenchAction::CycleActiveOverlayLayer => {
                let total = 1 + self.overlay_layers.len();
                self.active_overlay_idx = (self.active_overlay_idx + 1) % total;
                self.status_msg = format!("Active contour layer: #{}", self.active_overlay_idx);
            }
            WorkbenchAction::SwitchToProteinNmrWorkspace => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
            }

            // --- 4. Peak Picking & Editing ---
            WorkbenchAction::PickHsqc2dRoots => {
                self.pick_2d_roots();
            }
            WorkbenchAction::PickAll3dSpectra | WorkbenchAction::RunCorroboratedWeakPeakRescue => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let roots: Vec<Peak> = self
                        .peaks
                        .iter()
                        .filter(|p| p.exp == "HSQC")
                        .cloned()
                        .collect();
                    let mut cubes: Vec<&Spectrum3D> = Vec::new();
                    if let Some(ref s3) = self.hncacb_3d {
                        cubes.push(s3);
                    }
                    if let Some(ref s3) = self.hnco_3d {
                        cubes.push(s3);
                    }
                    if !roots.is_empty() && !cubes.is_empty() {
                        let pcfg = PickerConfig {
                            snr_threshold: self.picker_snr,
                            ..PickerConfig::default()
                        };
                        let p3 = pick_triple_3d_two_pass(
                            &cubes,
                            &roots,
                            &pcfg,
                            &CorroboratedRepickConfig::default(),
                        );
                        let n3 = p3.len();
                        let mut combined = roots;
                        combined.extend(p3);
                        self.peaks = combined;
                        self.session.state.peaks = self.peaks.clone();
                        self.rebuild_systems_and_matrices();
                        self.status_msg = format!("Two-pass 3D picker extracted {n3} cross-peaks");
                    } else {
                        self.status_msg =
                            "Load 2D HSQC roots and 3D cubes before 3D strip picking".to_string();
                    }
                }
            }
            WorkbenchAction::ToggleNeuralVsClassicalPicker => {
                self.use_neural_picker = !self.use_neural_picker;
                self.status_msg = format!(
                    "2D Picker: {}",
                    if self.use_neural_picker {
                        "Neural CNN"
                    } else {
                        "Classical MAD"
                    }
                );
            }
            WorkbenchAction::AddPeakAtCrosshair => {
                let amp = self.hsqc_spec.as_ref().map(|spec| {
                    let ix = spec
                        .axis_x
                        .ppm_to_index(self.crosshair.h_ppm)
                        .round()
                        .clamp(0.0, (spec.nx().saturating_sub(1)) as f64)
                        as usize;
                    let iy = spec
                        .axis_y
                        .ppm_to_index(self.crosshair.n_ppm)
                        .round()
                        .clamp(0.0, (spec.ny().saturating_sub(1)) as f64)
                        as usize;
                    spec.get(ix, iy) as f64
                });
                let pk = Peak {
                    exp: "HSQC".to_string(),
                    h: self.crosshair.h_ppm,
                    n: self.crosshair.n_ppm,
                    c: None,
                    sign: if amp.is_some_and(|a| a < 0.0) { -1 } else { 1 },
                    height: amp,
                    volume: None,
                    h2: None,
                };
                let msg = self.session.execute(WorkbenchCommand::AddPeak(pk));
                self.sync_app_from_session();
                self.status_msg = msg;
            }
            WorkbenchAction::DeleteSelectedPeaks => {
                if let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) {
                    let ss = &self.systems[sel_s];
                    let idxs: Vec<usize> = self
                        .peaks
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| (p.h - ss.h).abs() <= 0.035 && (p.n - ss.n).abs() <= 0.35)
                        .map(|(i, _)| i)
                        .collect();
                    if !idxs.is_empty() {
                        let msg = self.session.execute(WorkbenchCommand::DeletePeaks(idxs));
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                } else {
                    self.status_msg = "Select a peak / spin system to delete (`pd`)".to_string();
                }
            }
            WorkbenchAction::SnapPeakParabolicCenter => {
                self.snap_selected_peak_parabolic();
            }
            WorkbenchAction::TogglePeakLabels => {
                self.show_peak_labels = !self.show_peak_labels;
            }
            WorkbenchAction::FilterArtifactAndFoldedPeaks => {
                self.rebuild_systems_and_matrices();
            }

            // --- 5. Spin-System, Assignment & Session Undo/Redo ---
            WorkbenchAction::BuildSpinSystems => {
                self.rebuild_systems_and_matrices();
            }
            WorkbenchAction::RunBackboneAssignment => {
                self.run_assignment();
            }
            WorkbenchAction::SelectSolverEvoGraph => {
                self.solver_kind = GuiSolverKind::EvoGraph;
            }
            WorkbenchAction::SelectSolverStandardSa => {
                self.solver_kind = GuiSolverKind::Standard;
            }
            WorkbenchAction::ToggleGpuComputeEngine => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.use_gpu = !self.use_gpu && self.gpu_engine.is_some();
                }
            }
            WorkbenchAction::TileSequentialStrips => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.use_match_finder_strips = true;
            }
            WorkbenchAction::OpenResidueAssignmentPopover => {
                self.show_assign_popover = true;
            }
            WorkbenchAction::LockSelectedResidueAssignment => {
                if let (Some(ss), Some(rk)) = (self.selected_ss, self.selected_res) {
                    if ss < self.systems.len()
                        && rk < self.sequence.len()
                        && self.sequence[rk] != "PRO"
                        && rk > 0
                    {
                        let msg = self.session.execute(WorkbenchCommand::LockResidue {
                            spin_system: ss,
                            residue_seq_idx: rk,
                            locked: true,
                        });
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                }
            }
            WorkbenchAction::UnlockSelectedResidueAssignment => {
                if let Some(ss) = self.selected_ss {
                    if let Some(rk) = self.session.state.locked_residue_for_spin_system(ss) {
                        let msg = self.session.execute(WorkbenchCommand::LockResidue {
                            spin_system: ss,
                            residue_seq_idx: rk,
                            locked: false,
                        });
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                }
            }
            WorkbenchAction::UnassignSelectedSpinSystem => {
                if let Some(rk) = self.selected_res {
                    let msg = self.session.execute(WorkbenchCommand::UnassignResidue {
                        residue_seq_idx: rk,
                    });
                    self.sync_app_from_session();
                    self.status_msg = msg;
                }
            }
            WorkbenchAction::EstimateGlobalReferenceOffsets => {
                self.run_assignment();
            }
            WorkbenchAction::InspectDatasetQcReport => {
                self.show_qc_modal = !self.show_qc_modal;
            }
            WorkbenchAction::UndoLastCommand => {
                if let Some(msg) = self.session.undo() {
                    self.sync_app_from_session();
                    self.status_msg = msg;
                } else {
                    self.status_msg = "Nothing to undo".to_string();
                }
            }
            WorkbenchAction::RedoLastCommand => {
                if let Some(msg) = self.session.redo() {
                    self.sync_app_from_session();
                    self.status_msg = msg;
                } else {
                    self.status_msg = "Nothing to redo".to_string();
                }
            }

            // --- 6 & 7. Structure, Sidechain & Dynamics ---
            WorkbenchAction::PredictBackboneDihedrals => {
                if let Some(pred) = &self.pred_ref {
                    let dih = predict_csi_dihedrals(pred);
                    self.has_structure_or_dynamics = !dih.is_empty();
                    self.status_msg = format!(
                        "Predicted CSI backbone (φ, ψ) dihedrals for {} residues",
                        dih.len()
                    );
                } else {
                    self.status_msg =
                        "Run backbone assignment before predicting dihedrals".to_string();
                }
            }
            WorkbenchAction::RunJointSidechainAssignment => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.run_joint_sidechain_in_gui();
            }
            WorkbenchAction::InspectSidechainCovalentGraph => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_sidechain_inspector = true;
                if self.selected_res.is_none() && !self.sequence.is_empty() {
                    self.select_residue_and_sync_strip(0);
                }
                self.status_msg =
                    "Opened live 2D IUPAC covalent sidechain Z-score inspector".to_string();
            }
            WorkbenchAction::AssignNoesyRestraints
            | WorkbenchAction::FoldStructureEnsemble
            | WorkbenchAction::ValidateEnsembleRmsdRamachandran
            | WorkbenchAction::FitRelaxationR1R2HetNoe
            | WorkbenchAction::RunModelFreeLipariSzabo
            | WorkbenchAction::AnalyzeCpmgDispersion
            | WorkbenchAction::AnalyzeCestExchange
            | WorkbenchAction::FitLigandTitrationCspKd => {
                self.has_structure_or_dynamics = true;
                self.status_msg = format!("Triggered {}", action.metadata().title);
            }

            // --- 8. Octet Chemical Studio ---
            WorkbenchAction::SwitchToOctetChemStudio => {
                self.workspace_mode = WorkspaceMode::OctetChem;
            }
            WorkbenchAction::ImportLoadedProteinToOctet => {
                self.octet_app
                    .set_loaded_protein_sequence(self.sequence.clone());
                self.workspace_mode = WorkspaceMode::OctetChem;
            }
            WorkbenchAction::Clean2dMolecularLayout
            | WorkbenchAction::Predict1dProtonCarbonNmr
            | WorkbenchAction::ExportChemicalPublicationSvg => {
                self.workspace_mode = WorkspaceMode::OctetChem;
            }

            // --- 9. Help & Diagnostics ---
            WorkbenchAction::OpenCommandPalette => {
                self.command_palette.open_palette();
            }
            WorkbenchAction::OpenKeyboardCheatSheet => {
                self.chord_dispatcher.open_cheat_sheet();
            }
            WorkbenchAction::ReSimulateSyntheticNoise => {
                self.sim_seed = self.sim_seed.wrapping_add(1);
                if let Some(r) = self.reference.clone() {
                    self.load_from_reference(r);
                    self.run_assignment();
                }
            }
            WorkbenchAction::ShowAboutAndProvenance => {
                self.show_about_modal = true;
            }
        }
    }

    fn draw_phase9_modals(&mut self, ctx: &egui::Context) {
        // 1. Ingest Selection Matrix Modal (`T09.2`)
        if let Some(ingest_act) = self.ingest_panel.show(ctx) {
            match ingest_act {
                IngestAction::ScanNativePath(path_str) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match IngestPanelState::from_native_path(Path::new(&path_str)) {
                            Ok(panel) => {
                                self.ingest_panel = panel;
                                self.ingest_panel.open = true;
                            }
                            Err(e) => {
                                self.ingest_panel.status_error = Some(e);
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = path_str;
                    }
                }
                IngestAction::LoadSelectedSpectra(plan) => {
                    self.execute_ingest_plan(plan, false);
                }
                IngestAction::ProcessPickAndRun(plan) => {
                    self.execute_ingest_plan(plan, true);
                }
            }
        }

        // 2. `Cmd/Ctrl+K` Fuzzy Command Palette (`T09.3`)
        if let Some(pal_act) = self.command_palette.show(ctx) {
            self.dispatch_workbench_action(ctx, pal_act);
        }

        // 3. `?` Keyboard Accelerator Cheat-Sheet (`T09.4`)
        if let Some(cs_act) = self.chord_dispatcher.show_cheat_sheet_window(ctx) {
            self.dispatch_workbench_action(ctx, cs_act);
        }

        // 4. `vt` Contour Threshold Popover
        if self.show_contour_modal {
            let mut open = self.show_contour_modal;
            egui::Window::new("🎚 Contour Threshold & Levels (vt)")
                .id(egui::Id::unique("nmr_contour_threshold_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(340.0)
                .show(ctx, |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.contour_base, 0.2..=120.0)
                            .logarithmic(true)
                            .text("Base Floor"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.contour_factor, 1.12..=2.0).text("Step Ratio"),
                    );
                    ui.add(egui::Slider::new(&mut self.contour_levels, 2..=20).text("Levels"));
                    ui.checkbox(&mut self.show_negative, "Show Negative Contours");
                    if ui.button("Commit to Undo Stack").clicked() {
                        let mut params = SessionContourParams::new("HSQC");
                        params.base_level = self.contour_base;
                        params.level_factor = self.contour_factor;
                        params.num_levels = self.contour_levels;
                        params.show_negative = self.show_negative;
                        params.ppm_nudge = self.primary_ppm_nudge;
                        let msg = self.session.execute(WorkbenchCommand::SetContourParams {
                            layer_id: "HSQC".to_string(),
                            params,
                        });
                        self.status_msg = msg;
                    }
                });
            self.show_contour_modal = open;
        }

        // 5. Multi-Spectrum Overlay Stack Manager (`T09.5`)
        if self.show_overlay_manager {
            let mut open = self.show_overlay_manager;
            egui::Window::new("⧉ Multi-Spectrum GPU Contour Overlays")
                .id(egui::Id::unique("nmr_overlay_stack_manager"))
                .open(&mut open)
                .collapsible(false)
                .default_width(520.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Primary 2D HSQC Nudge (ppm):");
                        ui.add(
                            egui::DragValue::new(&mut self.primary_ppm_nudge[0])
                                .speed(0.002)
                                .prefix("Δ¹H "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.primary_ppm_nudge[1])
                                .speed(0.02)
                                .prefix("Δ¹⁵N "),
                        );
                        if ui.small_button("Zero").clicked() {
                            self.primary_ppm_nudge = [0.0, 0.0];
                        }
                    });

                    ui.horizontal(|ui| {
                        if let Some(ref s3) = self.hncacb_3d {
                            if ui
                                .small_button("+ Add 3D HNCACB ¹H–¹⁵N Plane Slice")
                                .clicked()
                            {
                                let c_ppm = self.crosshair.c_ppm.unwrap_or(56.0);
                                let lid = self.next_layer_id;
                                self.next_layer_id += 1;
                                let pal_i = self.overlay_layers.len() + 1;
                                let base = (estimate_noise_mad(&s3.data) * 4.5).max(1.0);
                                let mut ov = OverlayLayerSpec::from_spectrum_3d_hn_plane(
                                    lid, 1, s3, c_ppm, base, pal_i,
                                );
                                let (pos, neg) = default_overlay_palette(pal_i);
                                ov.pos_rgba = pos;
                                ov.neg_rgba = neg;
                                self.overlay_layers
                                    .push((format!("HNCACB @ {c_ppm:.1} ppm"), ov));
                            }
                        }
                        if !self.overlay_layers.is_empty()
                            && ui.small_button("Clear Overlays").clicked()
                        {
                            self.overlay_layers.clear();
                        }
                    });
                    ui.separator();

                    for (idx, (lbl, ov)) in self.overlay_layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut ov.visible, "");
                            ui.monospace(format!("#{} {lbl}", idx + 1));
                            let mut alpha = ov.pos_rgba[3];
                            if ui
                                .add(egui::Slider::new(&mut alpha, 0.15..=1.0).text("α"))
                                .changed()
                            {
                                ov.pos_rgba[3] = alpha;
                                ov.neg_rgba[3] = alpha;
                            }
                            ui.add(
                                egui::DragValue::new(&mut ov.base_level)
                                    .speed(0.2)
                                    .range(0.1..=200.0)
                                    .prefix("Base "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ov.ppm_nudge[0])
                                    .speed(0.002)
                                    .prefix("ΔH "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ov.ppm_nudge[1])
                                    .speed(0.02)
                                    .prefix("ΔN "),
                            );
                        });
                    }
                });
            self.show_overlay_manager = open;
        }

        // 6. `at` Residue Assignment & Lock Popover (`T09.6` / `T09.7`)
        if self.show_assign_popover {
            let mut open = self.show_assign_popover;
            egui::Window::new("📌 Assign & Lock Spin System (at)")
                .id(egui::Id::unique("nmr_assign_lock_popover"))
                .open(&mut open)
                .collapsible(false)
                .default_width(360.0)
                .show(ctx, |ui| {
                    if let Some(ss) = self.selected_ss.filter(|&s| s < self.systems.len()) {
                        ui.label(format!(
                            "Active Spin System: #{ss} (¹H {:.3}, ¹⁵N {:.2})",
                            self.systems[ss].h, self.systems[ss].n
                        ));
                        if !self.sequence.is_empty() {
                            let max_k = self.sequence.len().saturating_sub(1);
                            self.assign_target_res =
                                self.assign_target_res.clamp(1.min(max_k), max_k);
                            egui::ComboBox::from_id_salt("assign_target_res_combo")
                                .selected_text(format!(
                                    "{}{}",
                                    self.sequence[self.assign_target_res],
                                    self.seq_ids[self.assign_target_res]
                                ))
                                .show_ui(ui, |ui| {
                                    for k in 1..self.sequence.len() {
                                        if self.sequence[k] == "PRO" {
                                            continue;
                                        }
                                        ui.selectable_value(
                                            &mut self.assign_target_res,
                                            k,
                                            format!("{}{}", self.sequence[k], self.seq_ids[k]),
                                        );
                                    }
                                });
                            ui.checkbox(
                                &mut self.assign_lock_check,
                                "Pin as hard solver lock (🔒)",
                            );
                            ui.horizontal(|ui| {
                                if ui.button("Apply Assignment").clicked() {
                                    let target_k = self.assign_target_res;
                                    if self.assign_lock_check {
                                        let msg =
                                            self.session.execute(WorkbenchCommand::LockResidue {
                                                spin_system: ss,
                                                residue_seq_idx: target_k,
                                                locked: true,
                                            });
                                        self.status_msg = msg;
                                    } else {
                                        let msg = self.session.execute(
                                            WorkbenchCommand::AssignSpinSystem {
                                                spin_system: ss,
                                                residue_seq_idx: target_k,
                                            },
                                        );
                                        self.status_msg = msg;
                                    }
                                    self.sync_app_from_session();
                                    self.selected_res = Some(target_k);
                                }
                                if ui.button("Re-Solve with Locks").clicked() {
                                    self.run_assignment();
                                }
                            });
                        }
                    } else {
                        ui.label("Select an HSQC root on the 2D canvas first.");
                    }
                });
            self.show_assign_popover = open;
        }

        // 7. Dataset QC Report Modal
        if self.show_qc_modal {
            let mut open = self.show_qc_modal;
            egui::Window::new("🛡 Calibrated Dataset QC Gate")
                .id(egui::Id::unique("nmr_qc_report_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(440.0)
                .show(ctx, |ui| {
                    if let Some(ref qc) = self.qc_report {
                        ui.strong(format!("Verdict: {}", qc.verdict));
                        ui.monospace(format!(
                            "3D HSQC Support: {:.1}% | Confidence Scale: {:.2}",
                            qc.hsqc_3d_support_frac * 100.0,
                            qc.confidence_scale
                        ));
                        for reason in &qc.reasons {
                            ui.label(format!("• {reason}"));
                        }
                    } else {
                        ui.label("Run backbone assignment to evaluate Dataset QC metrics.");
                    }
                });
            self.show_qc_modal = open;
        }

        // 8. About & Clean-Room Provenance Modal
        if self.show_about_modal {
            let mut open = self.show_about_modal;
            egui::Window::new("About nmr-assign")
                .id(egui::Id::unique("nmr_about_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(440.0)
                .show(ctx, |ui| {
                    ui.strong(format!("nmr-assign v{}", nmr_core::VERSION));
                    ui.label(
                        "Clean-room open-source WebGPU protein NMR resonance assignment, raw FID/NUS DSP, and Octet 2D chemical studio (MIT License).",
                    );
                });
            self.show_about_modal = open;
        }
    }

    fn draw_bottom_status_bar(&mut self, ui: &mut egui::Ui, now_ms: u64) {
        ui.horizontal(|ui| {
            self.chord_dispatcher.draw_status_bar_indicator(ui, now_ms);
            ui.label(
                egui::RichText::new(format!(
                    "Crosshair: ¹H {:.3} | ¹⁵N {:.2}{}",
                    self.crosshair.h_ppm,
                    self.crosshair.n_ppm,
                    if self.crosshair.locked { " 🔒" } else { "" }
                ))
                .small()
                .monospace(),
            );
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "Undo: {} | Redo: {} | Locks: {}",
                    self.session.history.undo_len(),
                    self.session.history.redo_len(),
                    self.session.state.locked_residues.len()
                ))
                .small()
                .monospace()
                .weak(),
            );
            ui.separator();
            ui.label(
                egui::RichText::new(&self.status_msg)
                    .small()
                    .color(egui::Color32::from_rgb(190, 212, 240)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new("Cmd/Ctrl+K Palette  |  ? Shortcuts")
                        .small()
                        .monospace()
                        .weak(),
                );
            });
        });
    }
}

impl eframe::App for NmrWorkbenchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now_ms = ctx.input(|i| (i.time.max(0.0) * 1000.0).round() as u64);

        // 1. Global Cmd/Ctrl+K Command Palette toggle
        let toggle_palette = ctx.input_mut(|i| {
            let cmd = i.modifiers.command || i.modifiers.ctrl;
            cmd && !i.modifiers.alt
                && !i.modifiers.shift
                && i.consume_key(i.modifiers, egui::Key::K)
        });
        if toggle_palette {
            self.command_palette.toggle();
        }

        // 2. Global modifier shortcuts (Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z, Cmd/Ctrl+S, Cmd/Ctrl+O, ...)
        let octet_active = self.workspace_mode == WorkspaceMode::OctetChem;
        let mod_actions = poll_global_modifier_shortcuts(&ctx, octet_active);
        for act in mod_actions {
            self.dispatch_workbench_action(&ctx, act);
        }

        // 3. Modal Two-Key NMR Chord Dispatcher (suppressed whenever ctx.egui_wants_keyboard_input() or Octet tab has focus)
        let suppress_chords = ctx.egui_wants_keyboard_input()
            || octet_active
            || self.command_palette.open
            || self.ingest_panel.open
            || self.show_text_io_modal
            || self.show_assign_popover
            || self.show_contour_modal;
        let chord_actions = self
            .chord_dispatcher
            .handle_egui_input(&ctx, suppress_chords);
        for act in chord_actions {
            self.dispatch_workbench_action(&ctx, act);
        }

        // 4. Drag-and-drop + Modals
        self.handle_dropped_files(&ctx);
        self.draw_text_io_window(&ctx);
        self.draw_phase9_modals(&ctx);

        // 5. Top Menu Bar + 6-Stage Workflow Stepper + QC Verdict Pill + "What Next?"
        let readiness = self.compute_workspace_readiness();
        let mut chrome_actions = Vec::new();
        egui::Panel::top("workbench_phase9_chrome").show(ui, |ui| {
            self.draw_workspace_mode_header(ui);
            ui.add_space(2.0);
            chrome_actions = draw_workbench_chrome(ui, &readiness);
        });
        for act in chrome_actions {
            self.dispatch_workbench_action(&ctx, act);
        }

        // 6. Bottom Status Bar (with pending `Chord: p_` badge)
        egui::Panel::bottom("workbench_status_bar").show(ui, |ui| {
            self.draw_bottom_status_bar(ui, now_ms);
        });

        match self.workspace_mode {
            WorkspaceMode::ProteinNmr => {
                egui::Panel::top("top_controls_panel").show(ui, |ui| {
                    self.draw_top_bar(ui);
                    ui.add_space(4.0);
                    self.draw_sequence_ribbon(ui);
                });

                egui::Panel::right("right_strip_inspector")
                    .default_size(400.0)
                    .min_size(320.0)
                    .show(ui, |ui| {
                        self.draw_right_inspector(ui);
                    });

                if self.show_mol3d_viewport {
                    let mut picked_3d: Option<Mol3dSyncAction> = None;
                    egui::Panel::left("mol3d_structure_dock")
                        .default_size(390.0)
                        .min_size(280.0)
                        .max_size(680.0)
                        .resizable(true)
                        .show(ui, |ui| {
                            let display_ref = self.pred_ref.as_ref().or(self.reference.as_ref());
                            picked_3d = self.mol3d_panel.ui(
                                ui,
                                _frame.wgpu_render_state(),
                                &self.sequence,
                                &self.seq_ids,
                                display_ref,
                            );
                        });
                    if let Some(sync) = picked_3d {
                        self.apply_mol3d_sync_action(sync);
                    }
                }

                egui::CentralPanel::default().show(ui, |ui| {
                    self.draw_hsqc_canvas(ui);
                });
            }
            WorkspaceMode::OctetChem => {
                self.octet_app.ui(ui);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!(
                "nmr-assign v{} — WebGPU Protein NMR & Octet Chemical Studio",
                nmr_core::VERSION
            ))
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([960.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "nmr-assign",
        native_options,
        Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no browser Window")
            .document()
            .expect("no browser Document");

        let canvas = document
            .get_element_by_id("nmr_canvas")
            .expect("missing <canvas id=\"nmr_canvas\"> element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("nmr_canvas is not an HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
            )
            .await;

        if let Some(loading) = document.get_element_by_id("nmr_loading") {
            match start_result {
                Ok(()) => loading.remove(),
                Err(e) => {
                    loading.set_inner_html(&format!(
                        "<p style=\"color:#ff7b7b;max-width:520px;text-align:center\">Failed to start WebGPU workbench: {:?}<br>\
                         Ensure WebGPU is enabled in your browser (Chrome/Edge 113+, Safari 18+, Firefox Nightly).</p>",
                        e
                    ));
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nmr_core::aromatic::{
        default_his_tautomer_map, simulate_aromatic_peaks, synthesize_aromatic_reference,
        AromaticSimConfig,
    };
    use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig};
    use nmr_core::nmrstar::Reference;
    use nmr_core::sidechain::synthesize_all_atom_reference;
    use nmr_core::topology::{ResidueVariant, SidechainPriors};
    use nmr_gui::sidechain_widget::SequenceRibbonState;

    use super::{
        build_synthetic_ubiquitin, merge_gui_aromatic_preserving_user_rings,
        merge_gui_aromatic_with_options, parse_gui_c13_tsv_str, GuiRegimeKind, GuiSolverKind,
    };

    #[test]
    fn test_gui_default_solver_is_evo_graph() {
        assert_eq!(GuiSolverKind::default(), GuiSolverKind::EvoGraph);
        assert_eq!(GuiSolverKind::default().label(), "Evo-Graph");
    }

    #[test]
    fn test_gui_p2_variant_state_separation_across_resolves() {
        let seq = vec!["CYS".to_string(), "HIS".to_string(), "ALA".to_string()];
        let seq_ids = vec![1, 2, 3];
        let mut r1 = Reference::new(seq.clone(), seq_ids.clone());
        r1.shifts.insert((0, "CB".to_string()), 41.2); // Oxidized Cys
        r1.shifts.insert((1, "ND1".to_string()), 249.5); // Ne2H His
        r1.shifts.insert((1, "NE2".to_string()), 167.8);

        let mut state = SequenceRibbonState::new();
        state.set_variant(0, ResidueVariant::CysReduced); // Explicit user override on Cys 1
        state.sync_inferred_variants_from_reference(&r1);

        // User override wins on Cys 1 while auto-inferred state is recorded cleanly underneath
        assert_eq!(
            state.auto_inferred_variants.get(&0),
            Some(&ResidueVariant::CysOxidized)
        );
        assert_eq!(state.variant_for(0), ResidueVariant::CysReduced);
        assert_eq!(state.variant_for(1), ResidueVariant::HisNe2H);

        // Re-solve where His 2 has no ring shifts anymore -> stale auto-inferred His variant clears!
        let mut r2 = Reference::new(seq, seq_ids);
        r2.shifts.insert((0, "CB".to_string()), 28.0);
        state.sync_inferred_variants_from_reference(&r2);
        assert_eq!(state.variant_for(1), ResidueVariant::Canonical);
        assert_eq!(state.variant_for(0), ResidueVariant::CysReduced);
    }

    #[test]
    fn test_gui_p1_12_aromatic_merge_preserves_unassigned_user_rings() {
        let seq = vec![
            "ALA".to_string(),
            "PHE".to_string(),
            "GLY".to_string(),
            "PHE".to_string(),
            "HIS".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4, 5];
        let mut base = Reference::new(seq.clone(), seq_ids);
        for (k, ca, cb) in [
            (0, 52.5, 19.1),
            (1, 57.8, 39.4),
            (2, 45.2, 0.0),
            (3, 56.1, 42.3),
            (4, 55.4, 29.8),
        ] {
            base.shifts.insert((k, "CA".to_string()), ca);
            if cb > 0.0 {
                base.shifts.insert((k, "CB".to_string()), cb);
            }
        }
        let sc_priors = SidechainPriors::empirical_bmrb();
        let all_ref = synthesize_all_atom_reference(&base, &sc_priors, 77);
        let his_map = default_his_tautomer_map(&seq, 77);
        let truth_ref = synthesize_aromatic_reference(&all_ref, &sc_priors, &his_map, 77);

        // Simulate peaks ONLY for PHE at index 1 (drop peaks belonging to PHE index 3 and HIS index 4)
        let mut solve_source = truth_ref.clone();
        solve_source.shifts.retain(|&(k, ref a), _| {
            !(k >= 3 && !matches!(a.as_str(), "CA" | "CB" | "HA" | "HB2" | "HB3"))
        });
        let (arom_peaks, _) = simulate_aromatic_peaks(
            &solve_source,
            &BTreeMap::new(),
            &AromaticSimConfig::default(),
        );

        // User-curated shifts on unassigned PHE (idx 3) and HIS (idx 4)
        let mut user_ref = base.clone();
        user_ref.shifts.insert(
            (1, "HB2".to_string()),
            truth_ref.shifts[&(1, "HB2".to_string())],
        );
        user_ref.shifts.insert(
            (1, "HB3".to_string()),
            truth_ref.shifts[&(1, "HB3".to_string())],
        );
        user_ref.shifts.insert((3, "CD1".to_string()), 131.42);
        user_ref.shifts.insert((3, "HD1".to_string()), 7.26);
        user_ref.shifts.insert((4, "CE1".to_string()), 137.85);
        user_ref.shifts.insert((4, "HE1".to_string()), 8.14);
        let mut prior_amb = BTreeMap::new();
        prior_amb.insert((3, "CD1".to_string()), 2u8);
        prior_amb.insert((4, "CE1".to_string()), 1u8);

        let merged = merge_gui_aromatic_preserving_user_rings(
            &user_ref,
            Some(&user_ref),
            &prior_amb,
            None,
            &arom_peaks,
            &sc_priors,
        );

        // Observed PHE 2 is solved, while unobserved PHE 4 and HIS 5 preserve user ring shifts & ambiguity codes!
        assert!(merged.shifts.contains_key(&(1, "CD1".to_string())));
        assert_eq!(
            merged.shifts.get(&(3, "CD1".to_string())).copied(),
            Some(131.42)
        );
        assert_eq!(
            merged.shifts.get(&(3, "HD1".to_string())).copied(),
            Some(7.26)
        );
        assert_eq!(
            merged.shifts.get(&(4, "CE1".to_string())).copied(),
            Some(137.85)
        );
        assert_eq!(
            merged.shifts.get(&(4, "HE1".to_string())).copied(),
            Some(8.14)
        );
        assert_eq!(
            merged.ambiguity_codes.get(&(3, "CD1".to_string())).copied(),
            Some(2)
        );
        assert_eq!(
            merged.ambiguity_codes.get(&(4, "CE1".to_string())).copied(),
            Some(1)
        );
    }

    #[test]
    fn test_gui_phase11_regimes_c13_tsv_and_propensity_track() {
        assert_eq!(GuiRegimeKind::default(), GuiRegimeKind::Globular);
        assert_eq!(GuiRegimeKind::Idp13C.label(), "13C-IDP");
        assert_eq!(GuiRegimeKind::Paramagnetic.label(), "Paramagnetic");

        let tsv = "\
# exp\tdim1\tdim2\tdim3\tsign\theight
CON\t175.40\t122.10\t.\t1\t1200.0
CACO\t175.42\t55.10\t.\t1\t950.0
CANCO\t175.40\t122.11\t55.08\t1\t600.0
CBCACON\t175.41\t122.09\t30.40\t1\t520.0
";
        let ds = parse_gui_c13_tsv_str(tsv).expect("valid 13C-detected TSV");
        assert_eq!(ds.con_roots.len(), 1);
        assert_eq!(ds.caco_peaks.len(), 1);
        assert_eq!(ds.strip_peaks.len(), 2);
        assert_eq!(ds.strip_peaks[0].anchor_root_idx, Some(0));
        assert_eq!(ds.strip_peaks[1].anchor_root_idx, Some(0));

        let ubiq = build_synthetic_ubiquitin();
        let rep = compute_idp_propensity(&ubiq, &PropensityConfig::default());
        assert_eq!(rep.residues.len(), ubiq.sequence.len());
        for r in &rep.residues {
            assert!((r.population_sum() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gui_p2_explicit_canonical_override_persists_across_resolves() {
        let seq = vec!["CYS".to_string(), "HIS".to_string()];
        let seq_ids = vec![1, 2];
        let mut ref_shifts = Reference::new(seq, seq_ids);
        ref_shifts.shifts.insert((0, "CB".to_string()), 41.6); // Oxidized Cys signal
        ref_shifts.shifts.insert((1, "ND1".to_string()), 249.4); // His Ne2H signal
        ref_shifts.shifts.insert((1, "NE2".to_string()), 167.5);

        let mut state = SequenceRibbonState::new();
        state.sync_inferred_variants_from_reference(&ref_shifts);
        assert_eq!(state.variant_for(0), ResidueVariant::CysOxidized);
        assert_eq!(state.variant_for(1), ResidueVariant::HisNe2H);

        // User explicitly selects Canonical on both residues: must record explicit Canonical override!
        state.set_variant(0, ResidueVariant::Canonical);
        state.set_variant(1, ResidueVariant::Canonical);
        assert_eq!(
            state.user_variant_overrides.get(&0),
            Some(&ResidueVariant::Canonical)
        );
        assert_eq!(
            state.user_variant_overrides.get(&1),
            Some(&ResidueVariant::Canonical)
        );

        // Subsequent auto-sync must NEVER silently override explicit Canonical choice!
        state.sync_inferred_variants_from_reference(&ref_shifts);
        assert_eq!(state.variant_for(0), ResidueVariant::Canonical);
        assert_eq!(state.variant_for(1), ResidueVariant::Canonical);
    }

    #[test]
    fn test_gui_p1_10_curated_ring_protection_partial_and_duplicate_guards() {
        use nmr_core::peaks::Peak;

        let sc_priors = SidechainPriors::empirical_bmrb();
        let seq = vec![
            "ALA".to_string(),
            "PHE".to_string(),
            "GLY".to_string(),
            "PHE".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4];

        // 1. Conflicting solver ring must NOT overwrite curated ring atoms unless overwrite_curated_rings == true
        let mut curated = Reference::new(seq.clone(), seq_ids.clone());
        curated.shifts.insert((1, "CA".to_string()), 57.5);
        curated.shifts.insert((1, "CB".to_string()), 39.2);
        curated.shifts.insert((1, "CD1".to_string()), 131.10);
        curated.shifts.insert((1, "CD2".to_string()), 131.10);
        curated.shifts.insert((1, "HD1".to_string()), 7.120);
        curated.shifts.insert((1, "HD2".to_string()), 7.120);
        curated.shifts.insert((1, "CE1".to_string()), 129.30);
        curated.shifts.insert((1, "CE2".to_string()), 129.30);
        curated.shifts.insert((1, "HE1".to_string()), 7.280);
        curated.shifts.insert((1, "HE2".to_string()), 7.280);
        curated.shifts.insert((1, "CZ".to_string()), 127.60);
        curated.shifts.insert((1, "HZ".to_string()), 7.210);
        let mut prior_amb = BTreeMap::new();
        prior_amb.insert((1, "CD1".to_string()), 1u8);

        let mk_peak = |exp: &str, h: f64, c: f64, height: f64| Peak {
            exp: exp.to_string(),
            h,
            n: 0.0,
            c: Some(c),
            h2: None,
            sign: 1,
            height: Some(height),
            volume: None,
        };

        // Experimental peaks with same CB (39.2) but shifted ring frequencies (7.45 / 7.56 / 7.39 ppm)
        let conflicting_peaks = vec![
            mk_peak("C13HSQC", 7.450, 132.40, 1000.0),
            mk_peak("C13HSQC", 7.560, 130.60, 1000.0),
            mk_peak("C13HSQC", 7.390, 128.90, 900.0),
            mk_peak("HBCBCGCDHD", 7.450, 39.20, 700.0),
            mk_peak("HBCBCGCDCEHE", 7.560, 39.20, 650.0),
        ];

        let kept = merge_gui_aromatic_with_options(
            &curated,
            Some(&curated),
            &prior_amb,
            None,
            &conflicting_peaks,
            &sc_priors,
            false,
        );
        assert_eq!(
            kept.shifts.get(&(1, "CD1".to_string())).copied(),
            Some(131.10)
        );
        assert_eq!(
            kept.shifts.get(&(1, "HD1".to_string())).copied(),
            Some(7.120)
        );
        assert_eq!(
            kept.shifts.get(&(1, "CE1".to_string())).copied(),
            Some(129.30)
        );
        assert_eq!(
            kept.shifts.get(&(1, "CZ".to_string())).copied(),
            Some(127.60)
        );
        assert_eq!(
            kept.ambiguity_codes.get(&(1, "CD1".to_string())).copied(),
            Some(1)
        );

        // Explicit overwrite_curated_rings = true replaces curated shifts with solved shifts
        let replaced = merge_gui_aromatic_with_options(
            &curated,
            Some(&curated),
            &prior_amb,
            None,
            &conflicting_peaks,
            &sc_priors,
            true,
        );
        assert!(
            (replaced.shifts[&(1, "HD1".to_string())] - 7.450).abs() < 0.03,
            "overwrite_curated_rings=true should replace curated HD1"
        );

        // 2. Partial CD-only cluster must NOT be mixed into curated CE/CZ shifts when overwrite_curated_rings == false
        let mut ce_cz_curated = Reference::new(seq.clone(), seq_ids.clone());
        ce_cz_curated.shifts.insert((1, "CA".to_string()), 57.5);
        ce_cz_curated.shifts.insert((1, "CB".to_string()), 39.2);
        ce_cz_curated.shifts.insert((1, "CE1".to_string()), 129.30);
        ce_cz_curated.shifts.insert((1, "CE2".to_string()), 129.30);
        ce_cz_curated.shifts.insert((1, "HE1".to_string()), 7.280);
        ce_cz_curated.shifts.insert((1, "HE2".to_string()), 7.280);
        ce_cz_curated.shifts.insert((1, "CZ".to_string()), 127.60);
        ce_cz_curated.shifts.insert((1, "HZ".to_string()), 7.210);
        let cd_only_peaks = vec![
            mk_peak("C13HSQC", 7.120, 131.10, 1000.0),
            mk_peak("HBCBCGCDHD", 7.120, 39.20, 800.0),
        ];
        let no_partial_mix = merge_gui_aromatic_with_options(
            &ce_cz_curated,
            Some(&ce_cz_curated),
            &BTreeMap::new(),
            None,
            &cd_only_peaks,
            &sc_priors,
            false,
        );
        assert!(
            !no_partial_mix.shifts.contains_key(&(1, "CD1".to_string()))
                && !no_partial_mix.shifts.contains_key(&(1, "HD1".to_string())),
            "Partial CD-only cluster must not be mixed into curated CE/CZ shifts"
        );
        assert_eq!(
            no_partial_mix.shifts.get(&(1, "CE1".to_string())).copied(),
            Some(129.30)
        );
        assert_eq!(
            no_partial_mix.shifts.get(&(1, "CZ".to_string())).copied(),
            Some(127.60)
        );

        // 3. Duplicate ring cluster matching curated PHE #1 (idx 1) must NOT be assigned onto uncurated PHE #2 (idx 3)
        let mut dup_ref = curated.clone();
        dup_ref.shifts.insert((3, "CA".to_string()), 56.8);
        dup_ref.shifts.insert((3, "CB".to_string()), 39.25); // Degenerate CB near PHE #1!
        let dup_peaks_matching_phe1 = vec![
            mk_peak("C13HSQC", 7.120, 131.10, 1000.0),
            mk_peak("C13HSQC", 7.280, 129.30, 1000.0),
            mk_peak("C13HSQC", 7.210, 127.60, 900.0),
            mk_peak("HBCBCGCDHD", 7.120, 39.22, 750.0),
            mk_peak("HBCBCGCDCEHE", 7.280, 39.22, 700.0),
        ];
        let dup_checked = merge_gui_aromatic_with_options(
            &dup_ref,
            Some(&dup_ref),
            &prior_amb,
            None,
            &dup_peaks_matching_phe1,
            &sc_priors,
            false,
        );
        assert_eq!(
            dup_checked.shifts.get(&(1, "HD1".to_string())).copied(),
            Some(7.120)
        );
        assert!(
            !dup_checked.shifts.contains_key(&(3, "CD1".to_string()))
                && !dup_checked.shifts.contains_key(&(3, "HD1".to_string())),
            "Cluster matching curated PHE #1 must not be assigned as duplicate onto PHE #2"
        );
    }
}
