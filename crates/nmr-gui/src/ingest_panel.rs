//! Zero-friction spectrometer folder & archive drag-and-drop loader and editable
//! Experiment Selection Matrix (`T09.2`, desktop + `wasm32` WebGPU).
//!
//! Provenance: Clean-room original implementation of `docs/specs/12-workbench-ux.md` §1,
//! grounded in published multidimensional NMR pulse-sequence physics and open interchange specs:
//! - Kay, L. E., Ikura, M., Tschudin, R., & Bax, A. (*J. Magn. Reson.* 1990, 89, 496–514) —
//!   triple-resonance `HNCO`, `HNCA`, and `HN(CO)CA` transfer pathways.
//! - Grzesiek, S., & Bax, A. (*J. Am. Chem. Soc.* 1992, 114, 6291–6293) — `CBCA(CO)NH` and
//!   `HNCACB` sequential `Cα`/`Cβ` correlation windows.
//! - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C., & Pierattelli, R.
//!   (*J. Am. Chem. Soc.* 2009, 131, 15339–15345) — protonless `13C`-detected `CON`, `CACO`,
//!   `CANCO`, and `CBCACO` experiments for intrinsically disordered and paramagnetic proteins.
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E., & Kay, L. E. (*J. Am. Chem. Soc.* 2003,
//!   125, 10120–10121) — `1H-13C` methyl-TROSY `HMQC` spectroscopy.
//! - Gutmanas, A., et al. (*Nat. Struct. Mol. Biol.* 2015, 22, 433–434) — NMR Exchange Format
//!   (`NEF v1.1`) session and shift/peak interchange.

use std::path::{Path, PathBuf};
#[cfg(target_arch = "wasm32")]
use std::pin::Pin;
use std::sync::Arc;

use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::classify::classify_nmr_workspace;
use nmr_core::classify::{
    classify_nmr_entries, extract_archive_virtual_entries, ClassifiedExperiment, DataState,
    ExperimentDimensionality, ProblemRegime, PulseSequenceClass, VirtualEntry, WorkspaceInventory,
};
use nmr_core::nef::read_nef_str;
use nmr_core::nmrstar::read_star_str;
use nmr_core::priors::{one_to_three, three_to_one};
use nmr_core::session::read_session_nef_str;

/// Ordered catalog of all 25 [`PulseSequenceClass`] options exposed in the per-row override dropdown.
pub const ALL_PULSE_CLASSES: &[PulseSequenceClass] = &[
    PulseSequenceClass::N15Hsqc,
    PulseSequenceClass::Hncacb,
    PulseSequenceClass::Cbcaconh,
    PulseSequenceClass::Hnca,
    PulseSequenceClass::Hncoca,
    PulseSequenceClass::Hnco,
    PulseSequenceClass::Hncaco,
    PulseSequenceClass::C13Hsqc,
    PulseSequenceClass::C13Hmqc,
    PulseSequenceClass::Ccconh,
    PulseSequenceClass::Hccconh,
    PulseSequenceClass::HcchTocsy,
    PulseSequenceClass::N15Noesy,
    PulseSequenceClass::C13Noesy,
    PulseSequenceClass::Con2D,
    PulseSequenceClass::Caco2D,
    PulseSequenceClass::Canco3D,
    PulseSequenceClass::Cbcaco3D,
    PulseSequenceClass::RelaxR1,
    PulseSequenceClass::RelaxR2,
    PulseSequenceClass::HetNoe,
    PulseSequenceClass::Cpmg,
    PulseSequenceClass::Cest,
    PulseSequenceClass::TitrationSeries,
    PulseSequenceClass::Unknown,
];

/// Human-readable spectroscopist label for a [`PulseSequenceClass`].
pub fn pulse_class_display_label(cls: PulseSequenceClass) -> &'static str {
    match cls {
        PulseSequenceClass::N15Hsqc => "2D 15N-HSQC / TROSY",
        PulseSequenceClass::C13Hsqc => "2D 13C-HSQC",
        PulseSequenceClass::C13Hmqc => "2D 13C-Methyl HMQC",
        PulseSequenceClass::Hnco => "3D HNCO",
        PulseSequenceClass::Hncaco => "3D HN(CA)CO",
        PulseSequenceClass::Hnca => "3D HNCA",
        PulseSequenceClass::Hncoca => "3D HN(CO)CA",
        PulseSequenceClass::Hncacb => "3D HNCACB",
        PulseSequenceClass::Cbcaconh => "3D CBCA(CO)NH",
        PulseSequenceClass::Ccconh => "3D CC(CO)NH",
        PulseSequenceClass::Hccconh => "3D H(CCO)NH",
        PulseSequenceClass::HcchTocsy => "3D HCCH-TOCSY",
        PulseSequenceClass::N15Noesy => "3D 15N-NOESY",
        PulseSequenceClass::C13Noesy => "3D 13C-NOESY",
        PulseSequenceClass::Con2D => "2D CON (13C-det)",
        PulseSequenceClass::Caco2D => "2D CACO (13C-det)",
        PulseSequenceClass::Canco3D => "3D CANCO (13C-det)",
        PulseSequenceClass::Cbcaco3D => "3D CBCACO (13C-det)",
        PulseSequenceClass::RelaxR1 => "Pseudo-3D R1 (T1)",
        PulseSequenceClass::RelaxR2 => "Pseudo-3D R2 (T2)",
        PulseSequenceClass::HetNoe => "{1H}-15N hetNOE",
        PulseSequenceClass::Cpmg => "CPMG Dispersion",
        PulseSequenceClass::Cest => "CEST Exchange",
        PulseSequenceClass::TitrationSeries => "Titration CSP Series",
        PulseSequenceClass::Unknown => "Unknown / 1D Decoy",
    }
}

/// Human-readable label for a [`ProblemRegime`].
pub fn problem_regime_display_label(regime: ProblemRegime) -> &'static str {
    match regime {
        ProblemRegime::GlobularBackbone => "Globular 1H/15N/13C Backbone",
        ProblemRegime::SidechainNoesy => "Backbone + Sidechain TOCSY / 3D NOESY",
        ProblemRegime::Idp13cDetected => "IDP Protonless 13C-Detected Suite",
        ProblemRegime::MethylTrosy => "High-MW 13C-Methyl TROSY",
        ProblemRegime::Paramagnetic => "Paramagnetic PCS / PRE Suite",
        ProblemRegime::RelaxationDynamics => "Spin Relaxation & Exchange Dynamics",
        ProblemRegime::TitrationCsp => "Ligand Titration CSP Series",
    }
}

/// Default selection rule: include 2D/3D/Pseudo3D/4D classified experiments while excluding
/// `1D` checks (`pulsecal`, `zgpr`) and `Unknown` decoys by default.
pub fn should_select_experiment_by_default(exp: &ClassifiedExperiment) -> bool {
    if exp.dimensionality == ExperimentDimensionality::Dim1D
        || exp.pulse_class == PulseSequenceClass::Unknown
    {
        return false;
    }
    exp.selected_by_default
}

/// Parse a sequence string (FASTA, raw 1-letter, or whitespace-separated 3-letter codes)
/// into canonical uppercase 3-letter amino-acid codes (`MET`, `GLN`, ...).
pub fn parse_sequence_three_letter(input: &str) -> Vec<String> {
    let mut body_lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') || trimmed.starts_with('#') {
            continue;
        }
        body_lines.push(trimmed);
    }
    if body_lines.is_empty() {
        return Vec::new();
    }
    let joined = body_lines.join(" ");
    let tokens: Vec<&str> = joined.split_whitespace().collect();
    let looks_three_letter = tokens.len() >= 2
        && tokens
            .iter()
            .all(|t| t.len() == 3 && t.chars().all(|c| c.is_ascii_alphabetic()));

    if looks_three_letter {
        tokens.into_iter().map(|t| t.to_ascii_uppercase()).collect()
    } else {
        let mut out = Vec::new();
        for ch in joined.chars() {
            if ch.is_ascii_alphabetic() {
                let tri = one_to_three(ch.to_ascii_uppercase());
                if tri != "UNK" || ch.eq_ignore_ascii_case(&'X') {
                    out.push(tri.to_string());
                }
            }
        }
        out
    }
}

/// Extract a 1-letter sequence string from a FASTA, NMR-STAR (`.str`), or NEF (`.nef`) text buffer.
pub fn extract_sequence_from_auxiliary_text(filename: &str, text: &str) -> Option<String> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".bmrb") {
        if let Ok(r) = read_star_str(text, None) {
            if !r.sequence.is_empty() {
                let one: String = r.sequence.iter().map(|s| three_to_one(s)).collect();
                return Some(one);
            }
        }
    }
    if lower.ends_with(".nef") || text.contains("data_nef_") {
        if let Ok(sess) = read_session_nef_str(text) {
            if !sess.sequence.is_empty() {
                let one: String = sess.sequence.iter().map(|s| three_to_one(s)).collect();
                return Some(one);
            }
        }
        if let Ok(doc) = read_nef_str(text) {
            if let Ok(r) = doc.to_reference() {
                if !r.sequence.is_empty() {
                    let one: String = r.sequence.iter().map(|s| three_to_one(s)).collect();
                    return Some(one);
                }
            }
        }
    }
    if lower.ends_with(".fasta")
        || lower.ends_with(".fa")
        || lower.ends_with(".faa")
        || lower.ends_with(".seq")
    {
        let parsed = parse_sequence_three_letter(text);
        if !parsed.is_empty() {
            let one: String = parsed.iter().map(|s| three_to_one(s)).collect();
            return Some(one);
        }
    }
    None
}

/// Cross-platform in-memory implementation of [`egui::DroppedFile`] usable in unit/integration tests
/// and browser/clipboard byte-buffer pipelines on both native desktop and `wasm32`.
#[derive(Clone, Debug)]
pub struct InMemoryDroppedFile {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl InMemoryDroppedFile {
    /// Construct an in-memory [`egui::DroppedFile`] with `name` and `bytes`.
    pub fn new(name: impl Into<PathBuf>, bytes: Vec<u8>) -> Self {
        Self {
            path: name.into(),
            bytes,
        }
    }

    /// Wrap as a shared [`egui::DroppedFileHandle`].
    pub fn into_handle(name: impl Into<PathBuf>, bytes: Vec<u8>) -> egui::DroppedFileHandle {
        Arc::new(Self::new(name, bytes))
    }
}

impl egui::DroppedFile for InMemoryDroppedFile {
    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self.bytes.clone())
    }

    #[cfg(target_arch = "wasm32")]
    fn bytes_async(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
        let out = self.bytes.clone();
        Box::pin(async move { Ok(out) })
    }
}

/// Cross-platform `(name, bytes)` payload produced from browser or in-memory [`egui::DroppedFile`] handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DroppedPayload {
    pub name: String,
    pub bytes: Vec<u8>,
}

impl DroppedPayload {
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytes,
        }
    }
}

/// Resolved experiment selection row emitted when the user triggers **"Load Selected Spectra"**
/// or **"Process, Pick & Run"**.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedExperimentSpec {
    pub row_index: usize,
    pub experiment: ClassifiedExperiment,
    pub effective_class: PulseSequenceClass,
    pub solver_exp_name: Option<&'static str>,
    pub resolved_path: Option<PathBuf>,
}

/// Actionable execution plan built from the current [`IngestPanelState`].
#[derive(Clone, Debug, PartialEq)]
pub struct IngestExecutionPlan {
    pub root_name: String,
    pub root_path: Option<PathBuf>,
    pub problem_regime: ProblemRegime,
    pub sequence_three_letter: Vec<String>,
    pub selected_experiments: Vec<SelectedExperimentSpec>,
}

/// User action emitted by [`IngestPanelState::show`].
#[derive(Clone, Debug, PartialEq)]
pub enum IngestAction {
    /// Re-scan a folder or `.tar` / `.tar.gz` path typed into the path bar (native desktop).
    ScanNativePath(String),
    /// Load selected spectra + sequence into the workbench without running automated solving.
    LoadSelectedSpectra(IngestExecutionPlan),
    /// One-click pipeline: load/process selected experiments, corroborated 2D/3D pick, QC, and solve.
    ProcessPickAndRun(IngestExecutionPlan),
}

/// Result of inspecting one or more dropped files/folders/archives.
#[derive(Clone, Debug)]
pub enum IngestDropOutcome {
    /// A `.nef` workbench session file was dropped and should be restored directly.
    NefSession {
        source_name: String,
        nef_bytes: Vec<u8>,
    },
    /// A single standalone `.str` or `.tsv` text table was dropped.
    TextTable { source_name: String, text: String },
    /// A spectrometer folder, `.tar`/`.tar.gz` archive, or spectrum set was classified for the Ingest Matrix.
    Workspace(Box<IngestPanelState>),
}

/// Stateful controller for the **Editable Experiment Selection Matrix** (`T09.2`).
#[derive(Clone, Debug)]
pub struct IngestPanelState {
    pub open: bool,
    pub workspace_path_input: String,
    pub root_path: Option<PathBuf>,
    pub inventory: WorkspaceInventory,
    /// Per-row include/exclude checkbox (`1D` and `Unknown` decoys default to `false`).
    pub selected: Vec<bool>,
    /// Per-row effective [`PulseSequenceClass`] (initialized from auto-classifier, user-overridable).
    pub effective_classes: Vec<PulseSequenceClass>,
    /// True for rows where the user explicitly overrode [`PulseSequenceClass`].
    pub user_overridden: Vec<bool>,
    /// Optional native filesystem path per row when classified from disk.
    pub row_resolved_paths: Vec<Option<PathBuf>>,
    /// Editable polymer sequence box (1-letter FASTA or 3-letter residue codes).
    pub sequence_text: String,
    /// Provenance note when `sequence_text` was auto-populated from `.fasta`/`.str`/`.nef`.
    pub sequence_source_label: Option<String>,
    /// Optional inline error message displayed inside the modal.
    pub status_error: Option<String>,
}

impl Default for IngestPanelState {
    fn default() -> Self {
        Self::from_inventory(WorkspaceInventory {
            root_name: "workspace".to_string(),
            experiments: Vec::new(),
            auxiliary_files: Vec::new(),
            problem_regime: ProblemRegime::GlobularBackbone,
            secondary_regimes: Vec::new(),
            missing_input_hints: Vec::new(),
            elapsed_ms: 0.0,
        })
    }
}

impl IngestPanelState {
    /// Initialize an editable Experiment Selection Matrix from a [`WorkspaceInventory`].
    pub fn from_inventory(inventory: WorkspaceInventory) -> Self {
        let n = inventory.experiments.len();
        let selected: Vec<bool> = inventory
            .experiments
            .iter()
            .map(should_select_experiment_by_default)
            .collect();
        let effective_classes: Vec<PulseSequenceClass> = inventory
            .experiments
            .iter()
            .map(|e| e.pulse_class)
            .collect();

        Self {
            open: true,
            workspace_path_input: String::new(),
            root_path: None,
            inventory,
            selected,
            effective_classes,
            user_overridden: vec![false; n],
            row_resolved_paths: vec![None; n],
            sequence_text: String::new(),
            sequence_source_label: None,
            status_error: None,
        }
    }

    /// Classify a native folder, single spectrum, or `.tar`/`.tar.gz` archive on disk and populate
    /// per-row paths plus any discovered `.fasta` / `.str` / `.nef` polymer sequence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_native_path(path: &Path) -> Result<Self, String> {
        let inv = classify_nmr_workspace(path)?;
        let mut state = Self::from_inventory(inv);
        state.workspace_path_input = path.display().to_string();
        state.root_path = Some(path.to_path_buf());

        let is_dir = path.is_dir();
        for (idx, exp) in state.inventory.experiments.iter().enumerate() {
            let candidate = if is_dir {
                if exp.relative_path == "." || exp.relative_path.is_empty() {
                    path.to_path_buf()
                } else {
                    path.join(&exp.relative_path)
                }
            } else if path.is_file()
                && !is_archive_name_lower(&path.to_string_lossy().to_ascii_lowercase())
            {
                path.to_path_buf()
            } else {
                PathBuf::new()
            };
            if !candidate.as_os_str().is_empty() && candidate.exists() {
                state.row_resolved_paths[idx] = Some(candidate);
            }
        }

        // Auto-populate sequence from discovered auxiliary files on disk (or inside `.tar`/`.tar.gz`)
        if is_dir {
            for aux in &state.inventory.auxiliary_files {
                let aux_path = path.join(&aux.relative_path);
                if let Ok(txt) = std::fs::read_to_string(&aux_path) {
                    if let Some(seq1) =
                        extract_sequence_from_auxiliary_text(&aux.relative_path, &txt)
                    {
                        state.sequence_source_label = Some(format!(
                            "{} ({} aa)",
                            aux.relative_path,
                            seq1.chars().count()
                        ));
                        state.sequence_text = seq1;
                        break;
                    }
                }
            }
        } else if path.is_file()
            && is_archive_name_lower(&path.to_string_lossy().to_ascii_lowercase())
        {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(entries) =
                    extract_archive_virtual_entries(&state.inventory.root_name, &bytes)
                {
                    state.populate_sequence_from_virtual_entries(&entries);
                }
            }
        }

        Ok(state)
    }

    /// Populate `sequence_text` from header/table bytes in a `VirtualEntry` slice when empty.
    pub fn populate_sequence_from_virtual_entries(&mut self, entries: &[VirtualEntry]) {
        if !self.sequence_text.trim().is_empty() {
            return;
        }
        for entry in entries {
            if entry.header_bytes.is_empty() {
                continue;
            }
            if let Ok(txt) = std::str::from_utf8(&entry.header_bytes) {
                if let Some(seq1) = extract_sequence_from_auxiliary_text(&entry.relative_path, txt)
                {
                    self.sequence_source_label = Some(format!(
                        "{} ({} aa)",
                        entry.relative_path,
                        seq1.chars().count()
                    ));
                    self.sequence_text = seq1;
                    break;
                }
            }
        }
    }

    /// Toggle row selection in the matrix.
    pub fn set_row_selected(&mut self, row: usize, include: bool) {
        if let Some(slot) = self.selected.get_mut(row) {
            *slot = include;
        }
    }

    /// User override of `PulseSequenceClass` for `row`, persisted across selection changes and plan builds.
    pub fn override_pulse_class(&mut self, row: usize, new_class: PulseSequenceClass) {
        if row >= self.effective_classes.len() {
            return;
        }
        let prev_class = self.effective_classes[row];
        self.effective_classes[row] = new_class;
        self.user_overridden[row] = new_class != self.inventory.experiments[row].pulse_class;

        if new_class == PulseSequenceClass::Unknown {
            self.selected[row] = false;
        } else if prev_class == PulseSequenceClass::Unknown
            && self.inventory.experiments[row].dimensionality != ExperimentDimensionality::Dim1D
        {
            self.selected[row] = true;
        }
    }

    /// Select all 2D/3D backbone & sidechain experiments (excluding `1D` and `Unknown`).
    pub fn select_all_2d_3d(&mut self) {
        for (i, exp) in self.inventory.experiments.iter().enumerate() {
            let cls = self.effective_classes[i];
            self.selected[i] = exp.dimensionality != ExperimentDimensionality::Dim1D
                && cls != PulseSequenceClass::Unknown;
        }
    }

    /// Uncheck every experiment row.
    pub fn deselect_all(&mut self) {
        self.selected.fill(false);
    }

    /// Count of currently checked rows.
    pub fn selected_count(&self) -> usize {
        self.selected.iter().filter(|&&b| b).count()
    }

    /// Parse the inline sequence box into uppercase 3-letter residue codes.
    pub fn parsed_sequence_three_letter(&self) -> Vec<String> {
        parse_sequence_three_letter(&self.sequence_text)
    }

    /// Compute live missing-input hints, clearing the sequence hint once `sequence_text` is populated,
    /// and never penalizing multi-day or multi-magnet acquisitions on stable proteins.
    pub fn dynamic_missing_hints(&self) -> Vec<String> {
        let has_seq = !self.parsed_sequence_three_letter().is_empty();
        let mut hints: Vec<String> = self
            .inventory
            .missing_input_hints
            .iter()
            .filter(|h| {
                let low = h.to_ascii_lowercase();
                if has_seq && (low.contains("sequence") || low.contains("fasta")) {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        if !has_seq
            && !hints
                .iter()
                .any(|h| h.to_ascii_lowercase().contains("sequence"))
        {
            hints.push(
                "No polymer sequence loaded yet — paste FASTA/1-letter sequence below before solving."
                    .to_string(),
            );
        }
        hints
    }

    /// Collect all checked rows with their effective (possibly user-overridden) [`PulseSequenceClass`].
    pub fn selected_experiments(&self) -> Vec<SelectedExperimentSpec> {
        self.inventory
            .experiments
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.selected.get(i).copied().unwrap_or(false))
            .map(|(i, exp)| {
                let cls = self.effective_classes[i];
                SelectedExperimentSpec {
                    row_index: i,
                    experiment: exp.clone(),
                    effective_class: cls,
                    solver_exp_name: cls.canonical_solver_exp(),
                    resolved_path: self.row_resolved_paths.get(i).and_then(|p| p.clone()),
                }
            })
            .collect()
    }

    /// Build a concrete [`IngestExecutionPlan`] from the current matrix state.
    pub fn build_execution_plan(&self) -> IngestExecutionPlan {
        IngestExecutionPlan {
            root_name: self.inventory.root_name.clone(),
            root_path: self.root_path.clone(),
            problem_regime: self.inventory.problem_regime,
            sequence_three_letter: self.parsed_sequence_three_letter(),
            selected_experiments: self.selected_experiments(),
        }
    }
}

fn is_archive_name_lower(lower: &str) -> bool {
    lower.ends_with(".tar") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz")
}

/// Check whether a filename or leading magic bytes represent a `.tar` / `.tar.gz` / `.tgz` archive.
pub fn is_archive_filename_or_magic(name: &str, bytes: &[u8]) -> bool {
    let lower = name.to_ascii_lowercase();
    if is_archive_name_lower(&lower) {
        return true;
    }
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        return true;
    }
    bytes.len() > 262 && &bytes[257..262] == b"ustar"
}

fn looks_like_nef_bytes(name: &str, bytes: &[u8]) -> bool {
    if name.trim().to_ascii_lowercase().ends_with(".nef") {
        return true;
    }
    let head = &bytes[..bytes.len().min(4096)];
    if let Ok(txt) = std::str::from_utf8(head) {
        let trimmed = txt.trim_start();
        trimmed.starts_with("data_nef_") || txt.contains("_nmrassign_workbench")
    } else {
        false
    }
}

/// Cross-platform (`native` + `wasm32`) byte-payload ingestion pipeline:
/// - Single `.nef` drop -> [`IngestDropOutcome::NefSession`]
/// - Single `.str` / `.tsv` text table -> [`IngestDropOutcome::TextTable`]
/// - `.tar` / `.tar.gz` / `.tgz` archive or multi-file spectrometer bundle ->
///   `extract_archive_virtual_entries` / `VirtualEntry` -> `classify_nmr_entries` -> [`IngestPanelState`].
pub fn ingest_dropped_payloads(payloads: &[DroppedPayload]) -> Result<IngestDropOutcome, String> {
    if payloads.is_empty() {
        return Err("No dropped files provided".to_string());
    }

    if payloads.len() == 1 {
        let p = &payloads[0];
        let lower = p.name.trim().to_ascii_lowercase();
        if looks_like_nef_bytes(&p.name, &p.bytes) {
            return Ok(IngestDropOutcome::NefSession {
                source_name: p.name.clone(),
                nef_bytes: p.bytes.clone(),
            });
        }
        if !is_archive_filename_or_magic(&p.name, &p.bytes)
            && (lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".tsv"))
        {
            if let Ok(text) = std::str::from_utf8(&p.bytes) {
                return Ok(IngestDropOutcome::TextTable {
                    source_name: p.name.clone(),
                    text: text.to_string(),
                });
            }
        }
    }

    let mut virtual_entries: Vec<VirtualEntry> = Vec::new();
    for p in payloads {
        if is_archive_filename_or_magic(&p.name, &p.bytes) {
            let extracted = extract_archive_virtual_entries(&p.name, &p.bytes)?;
            virtual_entries.extend(extracted);
        } else {
            let cap = p.bytes.len().min(256 * 1024);
            virtual_entries.push(VirtualEntry::header_only(
                p.name.replace('\\', "/"),
                p.bytes[..cap].to_vec(),
                p.bytes.len() as u64,
            ));
        }
    }

    let mut inv = classify_nmr_entries(&virtual_entries);
    if payloads.len() == 1 {
        let fname = Path::new(&payloads[0].name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&payloads[0].name);
        if !fname.is_empty() {
            inv.root_name = fname.to_string();
        }
    }

    let mut panel = IngestPanelState::from_inventory(inv);
    panel.populate_sequence_from_virtual_entries(&virtual_entries);
    Ok(IngestDropOutcome::Workspace(Box::new(panel)))
}

/// Inspect `egui::DroppedFileHandle` items on native desktop (`DroppedFile::path` + fallback `DroppedFile::bytes`).
#[cfg(not(target_arch = "wasm32"))]
pub fn ingest_dropped_files_native(
    files: &[egui::DroppedFileHandle],
) -> Result<IngestDropOutcome, String> {
    if files.is_empty() {
        return Err("No dropped files provided".to_string());
    }

    if files.len() == 1 {
        let f = &files[0];
        let p = f.path();
        let p_str = p.display().to_string();
        let lower = p_str.to_ascii_lowercase();

        if lower.ends_with(".nef") {
            let nef_bytes = if p.exists() {
                std::fs::read(p)
                    .map_err(|e| format!("failed to read dropped NEF file {}: {e}", p.display()))?
            } else {
                f.bytes()?
            };
            return Ok(IngestDropOutcome::NefSession {
                source_name: p_str,
                nef_bytes,
            });
        }

        if p.exists() {
            if p.is_file()
                && !is_archive_name_lower(&lower)
                && (lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".tsv"))
            {
                let text = std::fs::read_to_string(p)
                    .map_err(|e| format!("failed to read {}: {e}", p.display()))?;
                return Ok(IngestDropOutcome::TextTable {
                    source_name: p_str,
                    text,
                });
            }
            let panel = IngestPanelState::from_native_path(p)?;
            return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
        }
    }

    // Multi-file native drop on disk (e.g. multiple `.ucsf` + `.fasta`) or in-memory `DroppedFile::bytes`
    if files.iter().all(|f| f.path().exists()) {
        let mut combined_exps = Vec::new();
        let mut combined_aux = Vec::new();
        let mut row_paths = Vec::new();
        let mut seq_found: Option<(String, String)> = None;

        for f in files {
            let p = f.path();
            let inv = classify_nmr_workspace(p)?;
            for exp in inv.experiments {
                let rp = if p.is_dir() && exp.relative_path != "." && !exp.relative_path.is_empty()
                {
                    p.join(&exp.relative_path)
                } else {
                    p.to_path_buf()
                };
                row_paths.push(Some(rp));
                combined_exps.push(exp);
            }
            for aux in inv.auxiliary_files {
                if seq_found.is_none() {
                    let ap = if p.is_dir() {
                        p.join(&aux.relative_path)
                    } else {
                        p.to_path_buf()
                    };
                    if let Ok(txt) = std::fs::read_to_string(&ap) {
                        if let Some(seq1) =
                            extract_sequence_from_auxiliary_text(&aux.relative_path, &txt)
                        {
                            seq_found = Some((
                                format!("{} ({} aa)", aux.relative_path, seq1.chars().count()),
                                seq1,
                            ));
                        }
                    }
                }
                combined_aux.push(aux);
            }
        }

        let inv = WorkspaceInventory {
            root_name: format!("{} dropped items", files.len()),
            experiments: combined_exps,
            auxiliary_files: combined_aux,
            problem_regime: ProblemRegime::GlobularBackbone,
            secondary_regimes: Vec::new(),
            missing_input_hints: Vec::new(),
            elapsed_ms: 0.0,
        };
        let mut panel = IngestPanelState::from_inventory(inv);
        panel.row_resolved_paths = row_paths;
        if let Some((lbl, seq1)) = seq_found {
            panel.sequence_source_label = Some(lbl);
            panel.sequence_text = seq1;
        }
        return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
    }

    let mut payloads = Vec::with_capacity(files.len());
    for f in files {
        let name = f.path().display().to_string();
        let bytes = if f.path().exists() {
            std::fs::read(f.path())
                .map_err(|e| format!("failed to read {}: {e}", f.path().display()))?
        } else {
            f.bytes()?
        };
        payloads.push(DroppedPayload::new(name, bytes));
    }
    ingest_dropped_payloads(&payloads)
}

/// Asynchronously inspect `egui::DroppedFileHandle` items on `wasm32-unknown-unknown` via `DroppedFile::bytes_async`.
#[cfg(target_arch = "wasm32")]
pub async fn ingest_dropped_files_wasm(
    files: Vec<egui::DroppedFileHandle>,
) -> Result<IngestDropOutcome, String> {
    let mut payloads = Vec::with_capacity(files.len());
    for f in files {
        let name = f.path().display().to_string();
        let bytes = f.bytes_async().await?;
        payloads.push(DroppedPayload::new(name, bytes));
    }
    ingest_dropped_payloads(&payloads)
}

fn regime_badge_colors(regime: ProblemRegime) -> (egui::Color32, egui::Color32) {
    match regime {
        ProblemRegime::GlobularBackbone => (
            egui::Color32::from_rgb(20, 62, 54),
            egui::Color32::from_rgb(105, 240, 185),
        ),
        ProblemRegime::SidechainNoesy => (
            egui::Color32::from_rgb(24, 54, 78),
            egui::Color32::from_rgb(125, 215, 255),
        ),
        ProblemRegime::Idp13cDetected => (
            egui::Color32::from_rgb(56, 38, 82),
            egui::Color32::from_rgb(210, 175, 255),
        ),
        ProblemRegime::MethylTrosy => (
            egui::Color32::from_rgb(66, 52, 22),
            egui::Color32::from_rgb(255, 215, 115),
        ),
        ProblemRegime::Paramagnetic => (
            egui::Color32::from_rgb(72, 34, 38),
            egui::Color32::from_rgb(255, 155, 155),
        ),
        ProblemRegime::RelaxationDynamics | ProblemRegime::TitrationCsp => (
            egui::Color32::from_rgb(28, 56, 70),
            egui::Color32::from_rgb(135, 230, 240),
        ),
    }
}

fn format_axes_summary(exp: &ClassifiedExperiment) -> String {
    if exp.axes.is_empty() {
        return exp.quadrature_badge.clone();
    }
    let nucs: Vec<&str> = exp.axes.iter().map(|a| a.nucleus.as_str()).collect();
    format!("{} · {}", nucs.join("/"), exp.quadrature_badge)
}

impl IngestPanelState {
    /// Render the modal **Editable Experiment Selection Matrix** (`T09.2`) and return any triggered action.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<IngestAction> {
        if !self.open {
            return None;
        }

        let mut window_open = self.open;
        let mut emitted: Option<IngestAction> = None;

        egui::Window::new("📂 Spectrometer Folder & Archive Ingest Matrix")
            .id(egui::Id::unique("nmr_ingest_selection_matrix_modal"))
            .open(&mut window_open)
            .collapsible(false)
            .resizable(true)
            .default_width(980.0)
            .default_height(610.0)
            .show(ctx, |ui| {
                // 1. Native Path Bar + Scan button
                #[cfg(not(target_arch = "wasm32"))]
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Folder / Archive:").strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.workspace_path_input)
                            .hint_text("Drop a Bruker/Varian/UCSF/NMRPipe folder or .tar/.tar.gz archive, or enter path…")
                            .desired_width(560.0),
                    );
                    let enter_scan =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("🔍 Scan Folder / Archive").clicked() || enter_scan {
                        let trimmed = self.workspace_path_input.trim().to_string();
                        if !trimmed.is_empty() {
                            emitted = Some(IngestAction::ScanNativePath(trimmed));
                        }
                    }
                    if self.inventory.elapsed_ms > 0.0 {
                        ui.label(
                            egui::RichText::new(format!("({:.1} ms)", self.inventory.elapsed_ms))
                                .small()
                                .monospace()
                                .weak(),
                        );
                    }
                });

                if let Some(err) = &self.status_error {
                    ui.colored_label(egui::Color32::from_rgb(255, 130, 130), err);
                }

                // 2. Detected Primary / Secondary ProblemRegime Banner
                let (bg_col, fg_col) = regime_badge_colors(self.inventory.problem_regime);
                egui::Frame::group(ui.style())
                    .fill(bg_col)
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Detected Regime: {}",
                                    problem_regime_display_label(self.inventory.problem_regime)
                                ))
                                .strong()
                                .color(fg_col),
                            );
                            for &sec in &self.inventory.secondary_regimes {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "+ {}",
                                        problem_regime_display_label(sec)
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(210, 225, 245)),
                                );
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Root: {} ({} experiments)",
                                        self.inventory.root_name,
                                        self.inventory.experiments.len()
                                    ))
                                    .small()
                                    .monospace(),
                                );
                            });
                        });
                    });

                // 3. Missing-Input Hints Alert Strip
                let hints = self.dynamic_missing_hints();
                if !hints.is_empty() {
                    egui::Frame::group(ui.style())
                        .fill(egui::Color32::from_rgb(56, 43, 18))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(215, 165, 65)))
                        .inner_margin(egui::Margin::symmetric(8, 4))
                        .show(ui, |ui| {
                            for h in &hints {
                                ui.label(
                                    egui::RichText::new(format!("⚠ {h}"))
                                        .small()
                                        .color(egui::Color32::from_rgb(255, 218, 130)),
                                );
                            }
                        });
                }

                // 4. Inline Polymer Sequence Box
                let parsed_seq = self.parsed_sequence_three_letter();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Polymer Sequence ({} aa):",
                            parsed_seq.len()
                        ))
                        .strong(),
                    );
                    if let Some(src) = &self.sequence_source_label {
                        ui.label(
                            egui::RichText::new(format!("Auto-discovered from {src}"))
                                .small()
                                .color(egui::Color32::from_rgb(115, 235, 175)),
                        );
                    }
                    if ui.small_button("Clear Sequence").clicked() {
                        self.sequence_text.clear();
                        self.sequence_source_label = None;
                    }
                });
                ui.add(
                    egui::TextEdit::multiline(&mut self.sequence_text)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(2)
                        .desired_width(f32::INFINITY)
                        .hint_text("Paste FASTA, 1-letter (MQIFVKTL…), or 3-letter sequence here…"),
                );

                // 5. Matrix Selection Toolbar & Scrollable Editable Experiment Grid
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Selected: {} / {} experiments",
                            self.selected_count(),
                            self.inventory.experiments.len()
                        ))
                        .strong(),
                    );
                    if ui.small_button("Select All 2D/3D").clicked() {
                        self.select_all_2d_3d();
                    }
                    if ui.small_button("Clear Selection").clicked() {
                        self.deselect_all();
                    }
                    ui.label(
                        egui::RichText::new("1D checks & Unknown decoys are unchecked by default.")
                            .small()
                            .weak(),
                    );
                });

                egui::ScrollArea::both()
                    .id_salt("ingest_matrix_scroll")
                    .max_height(275.0)
                    .show(ui, |ui| {
                        egui::Grid::new("ingest_experiment_matrix_grid")
                            .striped(true)
                            .min_col_width(42.0)
                            .show(ui, |ui| {
                                ui.strong("Use");
                                ui.strong("Relative Path");
                                ui.strong("Pulse Sequence Class");
                                ui.strong("Vendor");
                                ui.strong("Dim");
                                ui.strong("State");
                                ui.strong("MHz");
                                ui.strong("Axes & Quadrature");
                                ui.strong("NUS");
                                ui.end_row();

                                let n_rows = self.inventory.experiments.len();
                                for i in 0..n_rows {
                                    let mut checked = self.selected[i];
                                    if ui.checkbox(&mut checked, "").changed() {
                                        self.set_row_selected(i, checked);
                                    }

                                    let (rel_path, pp_tip, vendor, dim, dstate, mhz, axes_txt, nus_is, nus_badge) = {
                                        let exp = &self.inventory.experiments[i];
                                        let tip = format!(
                                            "Display: {}\nPULPROG/seqfil: {}\nTitle: {}",
                                            exp.display_name,
                                            exp.pulse_program.as_deref().unwrap_or("-"),
                                            exp.title.as_deref().unwrap_or("-")
                                        );
                                        (
                                            exp.relative_path.clone(),
                                            tip,
                                            exp.vendor,
                                            exp.dimensionality,
                                            exp.data_state,
                                            exp.spectrometer_mhz,
                                            format_axes_summary(exp),
                                            exp.nus.is_nus,
                                            exp.nus.badge.clone(),
                                        )
                                    };

                                    ui.monospace(&rel_path).on_hover_text(pp_tip);

                                    let cur_cls = self.effective_classes[i];
                                    let overridden = self.user_overridden[i];
                                    let combo_lbl = if overridden {
                                        format!("✎ {}", pulse_class_display_label(cur_cls))
                                    } else {
                                        pulse_class_display_label(cur_cls).to_string()
                                    };
                                    egui::ComboBox::from_id_salt(("ingest_cls_override", i))
                                        .selected_text(combo_lbl)
                                        .width(170.0)
                                        .show_ui(ui, |ui| {
                                            for &cand in ALL_PULSE_CLASSES {
                                                if ui
                                                    .selectable_label(
                                                        cand == cur_cls,
                                                        pulse_class_display_label(cand),
                                                    )
                                                    .clicked()
                                                {
                                                    self.override_pulse_class(i, cand);
                                                }
                                            }
                                        });

                                    ui.monospace(vendor.as_str());
                                    ui.monospace(dim.as_str());

                                    let state_col = match dstate {
                                        DataState::Processed | DataState::Both => {
                                            egui::Color32::from_rgb(110, 230, 175)
                                        }
                                        DataState::RawFid => egui::Color32::from_rgb(255, 205, 110),
                                    };
                                    ui.colored_label(state_col, dstate.as_str());

                                    if let Some(f_mhz) = mhz {
                                        ui.monospace(format!("{f_mhz:.0}"));
                                    } else {
                                        ui.monospace("-");
                                    }

                                    ui.label(egui::RichText::new(axes_txt).small().monospace());

                                    let nus_col = if nus_is {
                                        egui::Color32::from_rgb(125, 215, 255)
                                    } else {
                                        egui::Color32::from_rgb(145, 160, 180)
                                    };
                                    ui.colored_label(nus_col, nus_badge);
                                    ui.end_row();
                                }
                            });
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    let has_sel = self.selected_count() > 0;
                    let run_btn = egui::Button::new(
                        egui::RichText::new("▶ Process, Pick & Run")
                            .strong()
                            .color(egui::Color32::from_rgb(15, 25, 20)),
                    )
                    .fill(egui::Color32::from_rgb(82, 224, 152));

                    if ui.add_enabled(has_sel, run_btn).clicked() {
                        emitted = Some(IngestAction::ProcessPickAndRun(
                            self.build_execution_plan(),
                        ));
                    }

                    if ui
                        .add_enabled(has_sel, egui::Button::new("📂 Load Selected Spectra"))
                        .clicked()
                    {
                        emitted = Some(IngestAction::LoadSelectedSpectra(
                            self.build_execution_plan(),
                        ));
                    }

                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            });

        if !window_open {
            self.open = false;
        }
        if matches!(
            emitted,
            Some(IngestAction::LoadSelectedSpectra(_) | IngestAction::ProcessPickAndRun(_))
        ) {
            self.open = false;
        }
        emitted
    }
}
