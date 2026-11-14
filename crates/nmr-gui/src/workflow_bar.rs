//! Guided 6-stage *"What Next?"* workflow stepper, structured top menu bar, workspace readiness
//! diagnostics, `DatasetQcVerdict` quality-control badge surfacing, and decoupled
//! [`WorkbenchAction`] dispatch catalog (`T09.3`).
//!
//! Provenance: Original clean-room human-computer interaction design for progressive biomolecular
//! NMR workflows (`docs/specs/12-workbench-ux.md` §2), grounded in peer-reviewed triple-resonance
//! sequential assignment and selective-prediction quality-control principles:
//! 1. Kay, L. E., Ikura, M., Tschudin, R. & Bax, A. (1990). "Three-dimensional triple-resonance
//!    NMR spectroscopy of isotopically enriched proteins." *J. Magn. Reson.* 89(3), 496–514.
//! 2. Grzesiek, S. & Bax, A. (1992). "Correlating backbone amide and side chain resonances in
//!    larger proteins by multiple relayed triple resonance NMR." *J. Am. Chem. Soc.* 114(16),
//!    6291–6293.
//! 3. Chow, C. K. (1970). "On Optimum Recognition Error and Reject Tradeoff." *IEEE Trans. Inf.
//!    Theory* 16(1), 41–46 — explicit abstention (`ABSTAIN`) surfaced in interactive scientific
//!    decision support when observable spectral evidence cannot support high-confidence calls.
//! 4. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). "On Calibration of Modern Neural
//!    Networks." *Proc. 34th ICML*, PMLR 70, 1321–1330 — calibrated confidence diagnostics.

use eframe::egui;
use nmr_core::qc::DatasetQcVerdict;

/// Top-level menu categories rendered in the structured workbench menu bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopMenu {
    File,
    Process,
    Spectrum,
    Peaks,
    Assignment,
    Structure,
    Dynamics,
    Chemistry,
    Help,
}

impl TopMenu {
    /// Ordered list of all 9 top-level workbench menus.
    pub const ALL: [Self; 9] = [
        Self::File,
        Self::Process,
        Self::Spectrum,
        Self::Peaks,
        Self::Assignment,
        Self::Structure,
        Self::Dynamics,
        Self::Chemistry,
        Self::Help,
    ];

    /// Display label for the menu bar button and command-palette category pill.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Process => "Process",
            Self::Spectrum => "Spectrum",
            Self::Peaks => "Peaks",
            Self::Assignment => "Assignment",
            Self::Structure => "Structure",
            Self::Dynamics => "Dynamics",
            Self::Chemistry => "Chemistry",
            Self::Help => "Help",
        }
    }
}

/// Decoupled action catalog emitted by the Top Menu Bar, Workflow Stepper, Command Palette,
/// and Two-Key Keyboard Accelerator engine for clean dispatch in the main GUI shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkbenchAction {
    // --- 1. File & Ingest (12) ---
    OpenIngestMatrix,
    LoadPathOrSpectrum,
    OpenPasteTextModal,
    LoadSequenceFasta,
    LoadUbiquitinDemo,
    SaveSessionNef,
    SaveSessionNefAs,
    LoadSessionNef,
    ExportNmrStar,
    ExportPeaksTsv,
    ExportQualityReportHtml,
    ResetWorkspace,

    // --- 2. FID Processing (8) ---
    ProcessAllRawFids,
    ReconstructNusIstFista,
    AutoPhaseDirect1D,
    AutoPhaseIndirectBiphasic,
    ToggleLinearPrediction,
    ApplyMedianBaseline,
    Register3dToHsqc2d,
    ExportProcessingRecipeJson,

    // --- 3. Spectrum View & Contours (12) ---
    OpenContourThresholdPopup,
    ZoomFullSweep,
    ZoomIn,
    ZoomOut,
    RaiseContourFloor,
    LowerContourFloor,
    ToggleNegativeContours,
    ToggleCrosshairLock,
    OpenOverlayStackManager,
    CycleActiveOverlayLayer,
    NudgeSpectrumReferenceOffset,
    SwitchToProteinNmrWorkspace,

    // --- 4. Peak Picking & Editing (9) ---
    PickHsqc2dRoots,
    PickAll3dSpectra,
    RunCorroboratedWeakPeakRescue,
    ToggleNeuralVsClassicalPicker,
    AddPeakAtCrosshair,
    DeleteSelectedPeaks,
    SnapPeakParabolicCenter,
    TogglePeakLabels,
    FilterArtifactAndFoldedPeaks,

    // --- 5. Spin-System, Backbone Assignment & Session Undo/Redo (15) ---
    BuildSpinSystems,
    RunBackboneAssignment,
    SelectSolverEvoGraph,
    SelectSolverStandardSa,
    SelectSolverBayesSa,
    ToggleGpuComputeEngine,
    TileSequentialStrips,
    OpenResidueAssignmentPopover,
    LockSelectedResidueAssignment,
    UnlockSelectedResidueAssignment,
    UnassignSelectedSpinSystem,
    EstimateGlobalReferenceOffsets,
    InspectDatasetQcReport,
    UndoLastCommand,
    RedoLastCommand,

    // --- 6. Sidechain & 3D Structure (6) ---
    RunJointSidechainAssignment,
    InspectSidechainCovalentGraph,
    PredictBackboneDihedrals,
    AssignNoesyRestraints,
    FoldStructureEnsemble,
    ValidateEnsembleRmsdRamachandran,

    // --- 7. Spin Relaxation & Chemical Exchange Dynamics (5) ---
    FitRelaxationR1R2HetNoe,
    RunModelFreeLipariSzabo,
    AnalyzeCpmgDispersion,
    AnalyzeCestExchange,
    FitLigandTitrationCspKd,

    // --- 8. Octet Chemical Studio (5) ---
    SwitchToOctetChemStudio,
    ImportLoadedProteinToOctet,
    Clean2dMolecularLayout,
    Predict1dProtonCarbonNmr,
    ExportChemicalPublicationSvg,

    // --- 9. Help & Diagnostics (4) ---
    OpenCommandPalette,
    OpenKeyboardCheatSheet,
    ReSimulateSyntheticNoise,
    ShowAboutAndProvenance,
}

/// Static metadata describing a registered [`WorkbenchAction`] across menus, palettes, and keymaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionMetadata {
    pub action: WorkbenchAction,
    pub menu: TopMenu,
    pub category: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub modifier_shortcut: Option<&'static str>,
    pub chord_badge: Option<&'static str>,
    /// True when a visual separator should appear *after* this item in its dropdown menu.
    pub menu_separator_after: bool,
}

impl WorkbenchAction {
    /// Complete deterministic registry of all 76 workbench commands (`>= 55` required).
    pub const ALL: &'static [Self] = &[
        // File (12)
        Self::OpenIngestMatrix,
        Self::LoadPathOrSpectrum,
        Self::OpenPasteTextModal,
        Self::LoadSequenceFasta,
        Self::LoadUbiquitinDemo,
        Self::SaveSessionNef,
        Self::SaveSessionNefAs,
        Self::LoadSessionNef,
        Self::ExportNmrStar,
        Self::ExportPeaksTsv,
        Self::ExportQualityReportHtml,
        Self::ResetWorkspace,
        // Process (8)
        Self::ProcessAllRawFids,
        Self::ReconstructNusIstFista,
        Self::AutoPhaseDirect1D,
        Self::AutoPhaseIndirectBiphasic,
        Self::ToggleLinearPrediction,
        Self::ApplyMedianBaseline,
        Self::Register3dToHsqc2d,
        Self::ExportProcessingRecipeJson,
        // Spectrum (12)
        Self::OpenContourThresholdPopup,
        Self::ZoomFullSweep,
        Self::ZoomIn,
        Self::ZoomOut,
        Self::RaiseContourFloor,
        Self::LowerContourFloor,
        Self::ToggleNegativeContours,
        Self::ToggleCrosshairLock,
        Self::OpenOverlayStackManager,
        Self::CycleActiveOverlayLayer,
        Self::NudgeSpectrumReferenceOffset,
        Self::SwitchToProteinNmrWorkspace,
        // Peaks (9)
        Self::PickHsqc2dRoots,
        Self::PickAll3dSpectra,
        Self::RunCorroboratedWeakPeakRescue,
        Self::ToggleNeuralVsClassicalPicker,
        Self::AddPeakAtCrosshair,
        Self::DeleteSelectedPeaks,
        Self::SnapPeakParabolicCenter,
        Self::TogglePeakLabels,
        Self::FilterArtifactAndFoldedPeaks,
        // Assignment & Undo/Redo (15)
        Self::BuildSpinSystems,
        Self::RunBackboneAssignment,
        Self::SelectSolverEvoGraph,
        Self::SelectSolverStandardSa,
        Self::SelectSolverBayesSa,
        Self::ToggleGpuComputeEngine,
        Self::TileSequentialStrips,
        Self::OpenResidueAssignmentPopover,
        Self::LockSelectedResidueAssignment,
        Self::UnlockSelectedResidueAssignment,
        Self::UnassignSelectedSpinSystem,
        Self::EstimateGlobalReferenceOffsets,
        Self::InspectDatasetQcReport,
        Self::UndoLastCommand,
        Self::RedoLastCommand,
        // Structure (6)
        Self::RunJointSidechainAssignment,
        Self::InspectSidechainCovalentGraph,
        Self::PredictBackboneDihedrals,
        Self::AssignNoesyRestraints,
        Self::FoldStructureEnsemble,
        Self::ValidateEnsembleRmsdRamachandran,
        // Dynamics (5)
        Self::FitRelaxationR1R2HetNoe,
        Self::RunModelFreeLipariSzabo,
        Self::AnalyzeCpmgDispersion,
        Self::AnalyzeCestExchange,
        Self::FitLigandTitrationCspKd,
        // Chemistry (5)
        Self::SwitchToOctetChemStudio,
        Self::ImportLoadedProteinToOctet,
        Self::Clean2dMolecularLayout,
        Self::Predict1dProtonCarbonNmr,
        Self::ExportChemicalPublicationSvg,
        // Help (4)
        Self::OpenCommandPalette,
        Self::OpenKeyboardCheatSheet,
        Self::ReSimulateSyntheticNoise,
        Self::ShowAboutAndProvenance,
    ];

    /// Retrieve canonical UI metadata, category, description, and shortcut/chord badges.
    pub const fn metadata(self) -> ActionMetadata {
        match self {
            // --- File ---
            Self::OpenIngestMatrix => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Open Folder / Archive Ingest Matrix",
                description: "Scan Bruker/Varian folders or .tar.gz/.zip bundles into the Experiment Selection Matrix",
                modifier_shortcut: Some("Cmd/Ctrl+O"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::LoadPathOrSpectrum => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Load Spectrum / STAR / NEF Path",
                description: "Load a local 2D/3D spectrum (pdata, .ft2, .ucsf), NMR-STAR (.str), or peak list (.tsv)",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::OpenPasteTextModal => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Paste In-Memory STAR / NEF / Peak TSV",
                description: "Open clipboard text import/export dialog for NMR-STAR, NEF, Peak TSV, or SMILES",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::LoadSequenceFasta => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Load Protein Sequence (.fasta / .str)",
                description: "Import primary amino-acid sequence for backbone and sidechain resonance assignment",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::LoadUbiquitinDemo => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Load Ubiquitin Demo Dataset",
                description: "Load synthetic 76-residue Ubiquitin 2D HSQC + 3D HNCACB benchmark workspace",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::SaveSessionNef => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "Session & NEF",
                title: "Save Session (.nef)",
                description: "Persist peaks, shifts, user residue locks, recipes, and contour state to lossless NEF v1.1",
                modifier_shortcut: Some("Cmd/Ctrl+S"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::SaveSessionNefAs => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "Session & NEF",
                title: "Save Session As (.nef)",
                description: "Export lossless NEF v1.1 workbench session to a new file path or browser download",
                modifier_shortcut: Some("Cmd/Ctrl+Shift+S"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::LoadSessionNef => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "Session & NEF",
                title: "Open Saved Session (.nef)",
                description: "Restore complete workbench state, peaks, locks, and view settings from a .nef file",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::ExportNmrStar => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Export Assigned Shifts (.str NMR-STAR v3)",
                description: "Write BMRB-compatible NMR-STAR v3 chemical shift loop with per-residue confidences",
                modifier_shortcut: Some("Cmd/Ctrl+E"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ExportPeaksTsv => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Export Picked Peak List (.tsv)",
                description: "Write picked 2D and 3D cross-peaks with heights and volumes to tab-separated format",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ExportQualityReportHtml => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Export Quality & Strip Report (.html)",
                description: "Generate self-contained HTML + SVG report with CSI secondary shifts, strip ladders, and QC",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::ResetWorkspace => ActionMetadata {
                action: self,
                menu: TopMenu::File,
                category: "File & Ingest",
                title: "Reset Active Workspace",
                description: "Clear loaded spectra, peaks, spin systems, and assignments to start a fresh project",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Process ---
            Self::ProcessAllRawFids => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Process All Raw FIDs",
                description: "Run apodization, zero-filling, quadrature recombination, FFT, and blind autophasing on FIDs",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ReconstructNusIstFista => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Reconstruct NUS Indirect Axes (IST / FISTA)",
                description: "Reconstruct non-uniformly sampled 2D/3D hypercomplex indirect planes with active debiasing",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::AutoPhaseDirect1D => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Blind Autophase Direct Dimension",
                description: "Optimize zero- and first-order phase corrections (P0, P1) on direct proton/carbon traces",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::AutoPhaseIndirectBiphasic => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Biphasic Autophase Indirect 13CA/CB Axis",
                description: "Lock positive CA (48–60 ppm) vs negative CB (14–38 ppm) sign polarity in HNCACB planes",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ToggleLinearPrediction => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Toggle Indirect Linear Prediction (LP)",
                description: "Enable or disable mirror-image / forward linear prediction on truncated indirect evolution axes",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ApplyMedianBaseline => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Apply Median Baseline Correction",
                description: "Subtract rolling median baseline pedestals across direct and indirect spectral traces",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::Register3dToHsqc2d => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Register 3D (1H, 15N) Axes to 2D HSQC",
                description: "Align inter-experiment 3D amide carrier offsets against the 2D 15N-HSQC root reference",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ExportProcessingRecipeJson => ActionMetadata {
                action: self,
                menu: TopMenu::Process,
                category: "FID Processing",
                title: "Export ProcessingRecipe (.json)",
                description: "Save deterministic DSP apodization, phase, zero-fill, and NUS parameters as JSON",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Spectrum ---
            Self::OpenContourThresholdPopup => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "View / Contour Threshold Popup",
                description: "Open interactive contour base threshold, geometric step factor, and level count popover",
                modifier_shortcut: None,
                chord_badge: Some("vt"),
                menu_separator_after: false,
            },
            Self::RaiseContourFloor => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Raise Contour Floor (×1.25)",
                description: "Multiply lowest positive/negative contour threshold by 1.25x to suppress thermal noise",
                modifier_shortcut: None,
                chord_badge: Some("ct"),
                menu_separator_after: false,
            },
            Self::LowerContourFloor => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Lower Contour Floor (÷1.25)",
                description: "Divide lowest positive/negative contour threshold by 1.25x to reveal weak cross-peaks",
                modifier_shortcut: None,
                chord_badge: Some("cb"),
                menu_separator_after: false,
            },
            Self::ToggleNegativeContours => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Toggle Negative Contours",
                description: "Show or hide negative-intensity contours (folded peaks and HNCACB -CB resonances)",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::ZoomFullSweep => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Zoom Out to Full Spectral Sweep",
                description: "Reset 2D viewport bounds to the full spectral width of the active spectrum",
                modifier_shortcut: None,
                chord_badge: Some("oz"),
                menu_separator_after: false,
            },
            Self::ZoomIn => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Zoom In Spectrum Viewport",
                description: "Zoom into the active 2D/strip spectral region around the cursor or viewport center",
                modifier_shortcut: None,
                chord_badge: Some("zi"),
                menu_separator_after: false,
            },
            Self::ZoomOut => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Zoom Out Spectrum Viewport",
                description: "Expand the active 2D/strip spectral region around the cursor or viewport center",
                modifier_shortcut: None,
                chord_badge: Some("zo"),
                menu_separator_after: false,
            },
            Self::ToggleCrosshairLock => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Toggle Synchronized Crosshair Lock",
                description: "Pin or unpin the synchronized (1H, 15N, 13C) frequency crosshair across 2D and 3D views",
                modifier_shortcut: None,
                chord_badge: Some("Space"),
                menu_separator_after: true,
            },
            Self::OpenOverlayStackManager => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Multi-Spectrum Overlay Stack Manager",
                description: "Configure overlaid 2D spectra / 3D slices, color palettes, visibility, and ppm nudges",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::CycleActiveOverlayLayer => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Cycle Active Overlay Layer",
                description: "Switch the primary target spectrum layer in the multi-spectrum GPU contour stack",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::NudgeSpectrumReferenceOffset => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Calibration / Reference Offset Nudge",
                description: "Adjust per-spectrum (Δ1H, Δ15N, Δ13C) chemical shift referencing calibration",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::SwitchToProteinNmrWorkspace => ActionMetadata {
                action: self,
                menu: TopMenu::Spectrum,
                category: "Spectrum View",
                title: "Switch to Protein NMR & 3D Strips View",
                description: "Focus the primary 2D WebGPU contour canvas, sequence ribbon, and 3D strip inspector",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Peaks ---
            Self::PickHsqc2dRoots => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Picking",
                title: "Pick 2D HSQC Root Peaks",
                description: "Detect 2D 1H–15N amide root cross-peaks using the active Neural CNN or Classical MAD picker",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::PickAll3dSpectra => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Picking",
                title: "Pick All 3D Triple-Resonance Spectra",
                description: "Run HSQC-restricted 3D extrema & Voigt/parabolic fitting across loaded 3D spectra",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::RunCorroboratedWeakPeakRescue => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Picking",
                title: "Run Pass-2 Corroborated 3D Weak-Peak Rescue",
                description: "Re-probe raw 3D strips at cross-experiment corroborated carbon frequencies with halo guards",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ToggleNeuralVsClassicalPicker => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Picking",
                title: "Toggle Neural CNN vs Classical MAD Picker",
                description: "Switch between separable residual CNN peak detection and local MAD extrema picking",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::AddPeakAtCrosshair => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Editing",
                title: "Add Peak at Crosshair",
                description: "Insert a manual cross-peak at the active (1H, 15N / 13C) cursor or locked crosshair position",
                modifier_shortcut: None,
                chord_badge: Some("pa"),
                menu_separator_after: false,
            },
            Self::DeleteSelectedPeaks => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Editing",
                title: "Delete Selected Peak(s)",
                description: "Remove the currently selected cross-peak(s) with full Cmd/Ctrl+Z undo history",
                modifier_shortcut: None,
                chord_badge: Some("pd"),
                menu_separator_after: false,
            },
            Self::SnapPeakParabolicCenter => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Editing",
                title: "Snap Peak to Parabolic Sub-Voxel Center",
                description: "Refine selected peak coordinates to the local 3-point log-parabolic intensity extremum",
                modifier_shortcut: None,
                chord_badge: Some("pc"),
                menu_separator_after: false,
            },
            Self::TogglePeakLabels => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Editing",
                title: "Toggle Peak & Residue Labels",
                description: "Show or hide spin-system IDs and assigned residue annotations on the spectrum canvas",
                modifier_shortcut: None,
                chord_badge: Some("pl"),
                menu_separator_after: true,
            },
            Self::FilterArtifactAndFoldedPeaks => ActionMetadata {
                action: self,
                menu: TopMenu::Peaks,
                category: "Peak Picking",
                title: "Filter Sinc Sidelobes, NH2 & Folded Roots",
                description: "Prune t1-noise ridges, sidechain NH2 doublets, and opposite-sign folded HSQC roots",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Assignment & Session Undo/Redo ---
            Self::RunBackboneAssignment => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Run Automated Backbone Assignment",
                description: "Solve global spin-system-to-sequence mapping with calibrated marginal confidences",
                modifier_shortcut: Some("Cmd/Ctrl+Enter"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::BuildSpinSystems => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Rebuild Spin Systems & U/P Matrices",
                description: "Cluster 3D cross-peaks onto 2D HSQC roots and recompute typing U(s,k) and pair P(s,t) scores",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::TileSequentialStrips => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Tile Sequential i-1 / i / i+1 Match Strips",
                description: "Rank and tile candidate predecessor (i-1) and successor (i+1) 3D strips by P(s,t) connectivity",
                modifier_shortcut: None,
                chord_badge: Some("st"),
                menu_separator_after: false,
            },
            Self::OpenResidueAssignmentPopover => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Open Residue Assignment Popover",
                description: "Inspect top U(s,k) residue candidates and assign or lock the active spin system interactively",
                modifier_shortcut: None,
                chord_badge: Some("at"),
                menu_separator_after: false,
            },
            Self::LockSelectedResidueAssignment => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Lock Selected Spin-System → Residue",
                description: "Pin the active spin system to its assigned sequence position as a hard solver constraint",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::UnlockSelectedResidueAssignment => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Unlock Selected Residue Constraint",
                description: "Release user hard lock on the selected residue so automated re-annealing may reassign it",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::UnassignSelectedSpinSystem => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Unassign Selected Spin System",
                description: "Clear sequence assignment on the active spin system and record undo entry",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::SelectSolverEvoGraph => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Select Evo-Graph Solver (Default)",
                description: "Use factor-graph belief propagation + island evolutionary solver with Pass-2 QC calibration",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::SelectSolverStandardSa => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Select Standard Simulated Annealing Solver",
                description: "Use multi-replica CPU/WebGPU Metropolis simulated annealing over U(s,k) and P(s,t)",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::SelectSolverBayesSa => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Select Bayesian SA (Cv Adaptive) Solver",
                description: "Use Bayesian shift/overlap pseudo-energy with specific-heat Cv(T) adaptive CPU/WebGPU annealing",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ToggleGpuComputeEngine => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Toggle WebGPU Compute Acceleration",
                description: "Enable or disable wgpu WGSL compute shaders for scoring matrices and multi-replica annealing",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::EstimateGlobalReferenceOffsets => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Estimate Global Shift Reference Offsets",
                description: "Fit unlabelled (Δ1H, Δ15N, Δ13CA, Δ13CB, Δ13C') referencing shifts with ΔLL significance",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::InspectDatasetQcReport => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Backbone Assignment",
                title: "Inspect Dataset QC Verdict & Reliability",
                description: "View unlabelled PASS/WARN/ABSTAIN diagnostics, 3D root support, and calibration status",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::UndoLastCommand => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Session & Undo",
                title: "Undo Last Edit",
                description: "Revert the most recent peak, assignment, lock, contour, or reference offset action",
                modifier_shortcut: Some("Cmd/Ctrl+Z"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::RedoLastCommand => ActionMetadata {
                action: self,
                menu: TopMenu::Assignment,
                category: "Session & Undo",
                title: "Redo Last Edit",
                description: "Re-apply the most recently undone workbench command from the 256-step history stack",
                modifier_shortcut: Some("Cmd/Ctrl+Shift+Z"),
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Structure ---
            Self::RunJointSidechainAssignment => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Run Joint Backbone + Sidechain Assignment",
                description: "Assign aliphatic & aromatic sidechain resonances across C13-HSQC, HCCH-TOCSY, and (H)CCCONH",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::InspectSidechainCovalentGraph => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Inspect 2D Covalent Amino-Acid Graph",
                description: "Display IUPAC covalent atom-bond topology colored by BMRB Z-score for the selected residue",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::PredictBackboneDihedrals => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Predict Backbone (phi, psi) Dihedral Restraints",
                description: "Derive secondary-shift CSI and torsion-angle basin restraints from assigned backbone shifts",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::AssignNoesyRestraints => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Assign 3D 15N/13C-NOESY Distance Restraints",
                description: "Run network-anchored 7-cycle ambiguous NOE distance restraint (r^-6) calibration",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::FoldStructureEnsemble => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Fold 3D Torsion-Angle Structure Ensemble",
                description: "Compute multi-conformer PDB ensemble via NeRF + analytical-gradient torsion annealing",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::ValidateEnsembleRmsdRamachandran => ActionMetadata {
                action: self,
                menu: TopMenu::Structure,
                category: "Sidechain & Structure",
                title: "Validate Ensemble RMSD & Ramachandran",
                description: "Evaluate Kabsch backbone/heavy-atom RMSD, NOE/dihedral violations, and Ramachandran statistics",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Dynamics ---
            Self::FitRelaxationR1R2HetNoe => ActionMetadata {
                action: self,
                menu: TopMenu::Dynamics,
                category: "Relaxation & Dynamics",
                title: "Fit R1, R2 & {1H}-15N hetNOE Rates",
                description: "Fit pseudo-3D relaxation decay planes with Monte Carlo uncertainties and R2/R1 τm estimate",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::RunModelFreeLipariSzabo => ActionMetadata {
                action: self,
                menu: TopMenu::Dynamics,
                category: "Relaxation & Dynamics",
                title: "Run Lipari–Szabo Model-Free (S², τe, Rex)",
                description: "Select per-residue M1–M5 internal motion models via AIC/BIC over multi-field relaxation data",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::AnalyzeCpmgDispersion => ActionMetadata {
                action: self,
                menu: TopMenu::Dynamics,
                category: "Relaxation & Dynamics",
                title: "Analyze CPMG / R1ρ Relaxation Dispersion",
                description: "Fit Carver–Richards / Baldwin millisecond-to-microsecond chemical exchange parameters",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::AnalyzeCestExchange => ActionMetadata {
                action: self,
                menu: TopMenu::Dynamics,
                category: "Relaxation & Dynamics",
                title: "Analyze CEST / DEST Dark-State Exchange",
                description: "Fit 2-/3-state Bloch–McConnell matrix-exponential saturation profiles for minor conformers",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::FitLigandTitrationCspKd => ActionMetadata {
                action: self,
                menu: TopMenu::Dynamics,
                category: "Relaxation & Dynamics",
                title: "Fit Ligand Titration CSP & Quadratic Kd",
                description: "Track 2D HSQC titration chemical shift perturbations (Δδ) and fit tight-binding Kd isotherms",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Chemistry (Octet Studio) ---
            Self::SwitchToOctetChemStudio => ActionMetadata {
                action: self,
                menu: TopMenu::Chemistry,
                category: "Octet Chemical Studio",
                title: "Open Octet 2D Chemical Draw & 1D NMR Studio",
                description: "Switch workspace to 2D molecular structure drawing, reaction schemes, and 1D NMR synthesis",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ImportLoadedProteinToOctet => ActionMetadata {
                action: self,
                menu: TopMenu::Chemistry,
                category: "Octet Chemical Studio",
                title: "Build Peptide in Octet from Protein Sequence",
                description: "Convert the loaded protein sequence or selected segment into an explicit 2D chemical graph",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: true,
            },
            Self::Clean2dMolecularLayout => ActionMetadata {
                action: self,
                menu: TopMenu::Chemistry,
                category: "Octet Chemical Studio",
                title: "Clean 2D Coordinates & CIP Stereochemistry",
                description: "Standardize bond lengths/angles and perceive (R)/(S) and (E)/(Z) descriptors in Octet Studio",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::Predict1dProtonCarbonNmr => ActionMetadata {
                action: self,
                menu: TopMenu::Chemistry,
                category: "Octet Chemical Studio",
                title: "Predict 1H & 13C 1D NMR Spectrum",
                description: "Synthesize HOSE-code + additivity 1D 1H/13C chemical shifts and first-order J multiplets",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ExportChemicalPublicationSvg => ActionMetadata {
                action: self,
                menu: TopMenu::Chemistry,
                category: "Octet Chemical Studio",
                title: "Export Chemical Structure (.svg / .cdxml)",
                description: "Render publication-grade ACS/RSC/Nature vector graphics or lossless CDXML document",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },

            // --- Help ---
            Self::OpenCommandPalette => ActionMetadata {
                action: self,
                menu: TopMenu::Help,
                category: "Help & Shortcuts",
                title: "Open Fuzzy Command Palette",
                description: "Search and execute any workbench command, workflow action, or two-key accelerator",
                modifier_shortcut: Some("Cmd/Ctrl+K"),
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::OpenKeyboardCheatSheet => ActionMetadata {
                action: self,
                menu: TopMenu::Help,
                category: "Help & Shortcuts",
                title: "Keyboard Accelerators Cheat-Sheet",
                description: "Open searchable reference modal of two-key NMR spectrum chords and modifier shortcuts",
                modifier_shortcut: None,
                chord_badge: Some("?"),
                menu_separator_after: true,
            },
            Self::ReSimulateSyntheticNoise => ActionMetadata {
                action: self,
                menu: TopMenu::Help,
                category: "Help & Shortcuts",
                title: "Re-Simulate Synthetic Benchmark Noise",
                description: "Advance RNG seed, regenerate synthetic 2D/3D spectra with missing/spurious peaks, and re-solve",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
            Self::ShowAboutAndProvenance => ActionMetadata {
                action: self,
                menu: TopMenu::Help,
                category: "Help & Shortcuts",
                title: "About nmr-assign & Clean-Room Provenance",
                description: "Display release version, WebGPU adapter status, and peer-reviewed algorithm citations",
                modifier_shortcut: None,
                chord_badge: None,
                menu_separator_after: false,
            },
        }
    }

    #[inline]
    pub const fn menu(self) -> TopMenu {
        self.metadata().menu
    }

    #[inline]
    pub const fn category(self) -> &'static str {
        self.metadata().category
    }

    #[inline]
    pub const fn title(self) -> &'static str {
        self.metadata().title
    }

    #[inline]
    pub const fn description(self) -> &'static str {
        self.metadata().description
    }

    #[inline]
    pub const fn modifier_shortcut(self) -> Option<&'static str> {
        self.metadata().modifier_shortcut
    }

    #[inline]
    pub const fn chord_badge(self) -> Option<&'static str> {
        self.metadata().chord_badge
    }

    /// Return all actions belonging to a specific [`TopMenu`], in menu display order.
    pub fn actions_for_menu(menu: TopMenu) -> Vec<Self> {
        Self::ALL
            .iter()
            .copied()
            .filter(|a| a.menu() == menu)
            .collect()
    }
}

/// Progressive 6-stage scientific NMR workflow stages (`T09.3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkflowStage {
    /// 1. Ingest sequence, raw FID folders/archives, processed spectra, or peak lists.
    Ingest,
    /// 2. Apodize, zero-fill, reconstruct NUS, FFT, and autophase raw time-domain FIDs.
    ProcessFids,
    /// 3. Pick 2D HSQC roots, 3D triple-resonance peaks, and corroborated weak peaks.
    PickPeaks,
    /// 4. Cluster spin systems, run `evo-graph` / `standard` assignment, and inspect QC.
    AssignSpins,
    /// 5. Sidechain assignment, 3D NOESY/torsion folding, or relaxation/exchange dynamics.
    FoldAndDynamics,
    /// 6. Save lossless `.nef` session and export NMR-STAR `.str` & HTML quality report.
    ExportAndReport,
}

impl WorkflowStage {
    /// Ordered sequence of all 6 workflow stages.
    pub const ALL: [Self; 6] = [
        Self::Ingest,
        Self::ProcessFids,
        Self::PickPeaks,
        Self::AssignSpins,
        Self::FoldAndDynamics,
        Self::ExportAndReport,
    ];

    /// 1-based stage ordinal (`1..=6`).
    #[inline]
    pub const fn step_number(self) -> u8 {
        match self {
            Self::Ingest => 1,
            Self::ProcessFids => 2,
            Self::PickPeaks => 3,
            Self::AssignSpins => 4,
            Self::FoldAndDynamics => 5,
            Self::ExportAndReport => 6,
        }
    }

    /// Concise stage title without ordinal prefix.
    #[inline]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Ingest => "Ingest",
            Self::ProcessFids => "Process FIDs",
            Self::PickPeaks => "Pick Peaks",
            Self::AssignSpins => "Assign Spins",
            Self::FoldAndDynamics => "Fold / Dynamics",
            Self::ExportAndReport => "Export & Report",
        }
    }

    /// Numbered stage label displayed on the persistent workflow stepper.
    #[inline]
    pub const fn numbered_label(self) -> &'static str {
        match self {
            Self::Ingest => "1. Ingest",
            Self::ProcessFids => "2. Process FIDs",
            Self::PickPeaks => "3. Pick Peaks",
            Self::AssignSpins => "4. Assign Spins",
            Self::FoldAndDynamics => "5. Fold / Dynamics",
            Self::ExportAndReport => "6. Export & Report",
        }
    }

    /// Representative action invoked if the user clicks directly on a stepper pill.
    #[inline]
    pub const fn default_action(self) -> WorkbenchAction {
        match self {
            Self::Ingest => WorkbenchAction::OpenIngestMatrix,
            Self::ProcessFids => WorkbenchAction::ProcessAllRawFids,
            Self::PickPeaks => WorkbenchAction::PickHsqc2dRoots,
            Self::AssignSpins => WorkbenchAction::RunBackboneAssignment,
            Self::FoldAndDynamics => WorkbenchAction::FoldStructureEnsemble,
            Self::ExportAndReport => WorkbenchAction::SaveSessionNef,
        }
    }
}

/// Readiness state of an individual [`WorkflowStage`] in the 6-stage stepper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageStatus {
    /// Stage prerequisites and outputs are satisfied.
    Completed,
    /// Current recommended active stage ready for immediate execution.
    Active,
    /// Stage cannot proceed (or requires curation) due to missing prerequisite input or `ABSTAIN` QC.
    Blocked,
    /// Downstream stage awaiting completion of earlier stages.
    Upcoming,
}

/// Visual & diagnostic specification for the prominent `DatasetQcVerdict` pill (`PASS` / `WARN` / `ABSTAIN`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QcBadgeSpec {
    pub verdict: DatasetQcVerdict,
    pub label: &'static str,
    pub fill_rgb: [u8; 3],
    pub stroke_rgb: [u8; 3],
    pub text_rgb: [u8; 3],
    pub summary: String,
}

/// Contextual single primary action recommendation surfaced by the *"What Next?"* button.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhatNextRecommendation {
    pub stage: WorkflowStage,
    pub button_label: &'static str,
    pub action: WorkbenchAction,
    pub rationale: String,
    pub blocked_reason: Option<String>,
}

/// Immutable snapshot of current workbench state consumed by the Top Menu Bar and Workflow Stepper.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkspaceReadiness {
    /// True when a primary amino-acid sequence (`>= 1` residue) is loaded.
    pub sequence_loaded: bool,
    /// Length of the loaded sequence in residues (`0` if none).
    pub sequence_len: usize,
    /// Total raw time-domain FID experiments (`fid` / `ser`) detected in the workspace.
    pub raw_fids_count: usize,
    /// Count of raw FIDs not yet Fourier-transformed / reconstructed into processed spectra.
    pub unprocessed_fids_count: usize,
    /// Number of processed 2D spectra loaded (e.g., `15N-HSQC`, `13C-HSQC`, `CON`).
    pub spectra_2d_count: usize,
    /// Number of processed 3D/4D spectra loaded (e.g., `HNCACB`, `HNCA`, `HNCO`).
    pub spectra_3d_count: usize,
    /// Total cross-peaks currently present in the peak table.
    pub picked_peaks_count: usize,
    /// Number of clustered candidate spin systems.
    pub spin_systems_count: usize,
    /// Number of sequence positions currently assigned to a spin system.
    pub assigned_residues_count: usize,
    /// Number of assigned residues with calibrated marginal confidence `>= 0.80`.
    pub high_confidence_count: usize,
    /// Number of user-pinned `(spin_system -> residue)` hard constraints.
    pub locked_residues_count: usize,
    /// True once a 3D structure ensemble or relaxation/exchange dynamics fit has been computed.
    pub has_structure_or_dynamics: bool,
    /// True once the current workspace has been saved to `.nef` or exported to `.str`/`.html`.
    pub exported_in_session: bool,
    /// Unlabelled dataset quality-control gate verdict (`PASS`, `WARN`, `ABSTAIN`), when evaluated.
    pub qc_verdict: Option<DatasetQcVerdict>,
    /// Optional concise human-readable QC note (e.g., `"93% 3D support"` or top warning trigger).
    pub qc_summary: Option<String>,
}

impl WorkspaceReadiness {
    /// Total processed 2D + 3D/4D spectra currently in memory.
    #[inline]
    pub const fn total_spectra_count(&self) -> usize {
        self.spectra_2d_count + self.spectra_3d_count
    }

    #[inline]
    fn needs_fid_processing(&self) -> bool {
        self.unprocessed_fids_count > 0
            || (self.raw_fids_count > 0
                && self.total_spectra_count() == 0
                && self.picked_peaks_count == 0)
    }

    #[inline]
    fn has_any_experimental_data(&self) -> bool {
        self.raw_fids_count > 0 || self.total_spectra_count() > 0 || self.picked_peaks_count > 0
    }

    /// Determine which of the 6 [`WorkflowStage`]s currently owns primary user focus.
    pub fn active_stage(&self) -> WorkflowStage {
        if !self.has_any_experimental_data() {
            return WorkflowStage::Ingest;
        }
        if self.needs_fid_processing() {
            return WorkflowStage::ProcessFids;
        }
        if self.picked_peaks_count == 0 {
            return WorkflowStage::PickPeaks;
        }
        if !self.sequence_loaded
            || self.assigned_residues_count == 0
            || self.qc_verdict == Some(DatasetQcVerdict::Abstain)
        {
            return WorkflowStage::AssignSpins;
        }
        if !self.has_structure_or_dynamics {
            return WorkflowStage::FoldAndDynamics;
        }
        WorkflowStage::ExportAndReport
    }

    /// Compute the [`StageStatus`] (`Completed`, `Active`, `Blocked`, `Upcoming`) of `stage`.
    pub fn stage_status(&self, stage: WorkflowStage) -> StageStatus {
        let active = self.active_stage();
        let is_abstain = self.qc_verdict == Some(DatasetQcVerdict::Abstain);

        match stage {
            WorkflowStage::Ingest => {
                if active == WorkflowStage::Ingest {
                    StageStatus::Active
                } else {
                    StageStatus::Completed
                }
            }
            WorkflowStage::ProcessFids => {
                if active == WorkflowStage::ProcessFids {
                    StageStatus::Active
                } else if active.step_number() > WorkflowStage::ProcessFids.step_number() {
                    StageStatus::Completed
                } else {
                    StageStatus::Upcoming
                }
            }
            WorkflowStage::PickPeaks => {
                if active == WorkflowStage::PickPeaks {
                    StageStatus::Active
                } else if self.picked_peaks_count > 0 {
                    StageStatus::Completed
                } else {
                    StageStatus::Upcoming
                }
            }
            WorkflowStage::AssignSpins => {
                if active == WorkflowStage::AssignSpins {
                    if !self.sequence_loaded || is_abstain {
                        StageStatus::Blocked
                    } else {
                        StageStatus::Active
                    }
                } else if self.assigned_residues_count > 0 && self.sequence_loaded && !is_abstain {
                    StageStatus::Completed
                } else if self.has_any_experimental_data() && !self.sequence_loaded {
                    StageStatus::Blocked
                } else {
                    StageStatus::Upcoming
                }
            }
            WorkflowStage::FoldAndDynamics => {
                if active == WorkflowStage::FoldAndDynamics {
                    StageStatus::Active
                } else if self.has_structure_or_dynamics && !is_abstain {
                    StageStatus::Completed
                } else if is_abstain {
                    StageStatus::Blocked
                } else {
                    StageStatus::Upcoming
                }
            }
            WorkflowStage::ExportAndReport => {
                if active == WorkflowStage::ExportAndReport {
                    if self.exported_in_session && self.assigned_residues_count > 0 && !is_abstain {
                        StageStatus::Completed
                    } else {
                        StageStatus::Active
                    }
                } else if is_abstain {
                    StageStatus::Blocked
                } else {
                    StageStatus::Upcoming
                }
            }
        }
    }

    /// Snapshot of all 6 stages paired with their current [`StageStatus`].
    pub fn all_stage_statuses(&self) -> [(WorkflowStage, StageStatus); 6] {
        WorkflowStage::ALL.map(|st| (st, self.stage_status(st)))
    }

    /// Actionable callouts describing missing prerequisite inputs or QC blocks.
    pub fn missing_inputs(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.sequence_loaded {
            missing.push("Missing sequence (.fasta / .str / .nef)".to_string());
        }
        if !self.has_any_experimental_data() {
            missing.push("Missing spectra, raw FIDs, or peak list (.tsv)".to_string());
        } else if !self.needs_fid_processing()
            && self.spectra_2d_count > 0
            && self.spectra_3d_count == 0
            && self.assigned_residues_count == 0
            && self.spin_systems_count == 0
        {
            missing.push("No 3D triple-resonance spectra loaded yet".to_string());
        }
        if self.qc_verdict == Some(DatasetQcVerdict::Abstain) {
            let detail = self
                .qc_summary
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("insufficient 3D root support or sequential connectivity");
            missing.push(format!("QC ABSTAIN ({detail})"));
        }
        missing
    }

    /// Build the prominent `PASS` (green) / `WARN` (amber) / `ABSTAIN` (red) QC badge spec.
    pub fn qc_badge(&self) -> Option<QcBadgeSpec> {
        let verdict = self.qc_verdict?;
        let note = self.qc_summary.as_deref().filter(|s| !s.trim().is_empty());

        match verdict {
            DatasetQcVerdict::Pass => Some(QcBadgeSpec {
                verdict,
                label: "QC: PASS",
                fill_rgb: [22, 112, 72],
                stroke_rgb: [82, 232, 160],
                text_rgb: [195, 255, 225],
                summary: match note {
                    Some(n) => format!("Dataset QC PASS — {n}. Unlabelled stoichiometry, 3D support, referencing, and sequential connectivity are self-consistent."),
                    None => "Dataset QC PASS — unlabelled stoichiometry, 3D support, referencing, and sequential connectivity are self-consistent.".to_string(),
                },
            }),
            DatasetQcVerdict::Warn => Some(QcBadgeSpec {
                verdict,
                label: "QC: WARN",
                fill_rgb: [128, 86, 18],
                stroke_rgb: [250, 198, 78],
                text_rgb: [255, 238, 185],
                summary: match note {
                    Some(n) => format!("Dataset QC WARN — {n}. Marginal confidences are conservatively damped; inspect offsets and weak 3D strips."),
                    None => "Dataset QC WARN — partial 3D support, crowding, or offset drift detected; confidences conservatively damped.".to_string(),
                },
            }),
            DatasetQcVerdict::Abstain => Some(QcBadgeSpec {
                verdict,
                label: "QC: ABSTAIN",
                fill_rgb: [138, 28, 38],
                stroke_rgb: [255, 110, 120],
                text_rgb: [255, 215, 220],
                summary: match note {
                    Some(n) => format!("Dataset QC ABSTAIN — {n}. All confidences capped <= 0.45 (0 high-confidence calls) to prevent false assignments."),
                    None => "Dataset QC ABSTAIN — severe 3D mismatch or sequential disconnection; all confidences capped <= 0.45 (0 HC calls).".to_string(),
                },
            }),
        }
    }

    /// Compute the contextual primary *"What Next?"* recommendation for the current workspace.
    pub fn what_next(&self) -> WhatNextRecommendation {
        let stage = self.active_stage();
        match stage {
            WorkflowStage::Ingest => {
                if self.sequence_loaded {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Load Spectra or FIDs",
                        action: WorkbenchAction::OpenIngestMatrix,
                        rationale: format!(
                            "Sequence loaded ({} aa); drop Bruker/Varian folders, spectra, or peak lists next.",
                            self.sequence_len
                        ),
                        blocked_reason: None,
                    }
                } else {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Open Folder / Ingest",
                        action: WorkbenchAction::OpenIngestMatrix,
                        rationale: "Drop a spectrometer folder, .tar.gz/.zip archive, .nef, or .str dataset to begin.".to_string(),
                        blocked_reason: None,
                    }
                }
            }
            WorkflowStage::ProcessFids => {
                let pending = self.unprocessed_fids_count.max(self.raw_fids_count);
                WhatNextRecommendation {
                    stage,
                    button_label: "What Next: Process Raw FIDs",
                    action: WorkbenchAction::ProcessAllRawFids,
                    rationale: format!(
                        "{pending} raw time-domain FID experiment(s) ready for apodization, NUS/FFT, and autophasing."
                    ),
                    blocked_reason: None,
                }
            }
            WorkflowStage::PickPeaks => {
                if self.spectra_2d_count > 0 {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Pick 2D HSQC Peaks",
                        action: WorkbenchAction::PickHsqc2dRoots,
                        rationale: format!(
                            "{} 2D and {} 3D spectra loaded; pick 2D amide roots and 3D strip peaks.",
                            self.spectra_2d_count, self.spectra_3d_count
                        ),
                        blocked_reason: None,
                    }
                } else {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Pick 3D Spectra",
                        action: WorkbenchAction::PickAll3dSpectra,
                        rationale: format!(
                            "{} 3D spectrum(a) loaded; run 3D peak picking to extract spin-system candidates.",
                            self.spectra_3d_count
                        ),
                        blocked_reason: None,
                    }
                }
            }
            WorkflowStage::AssignSpins => {
                if !self.sequence_loaded {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Load Sequence (.fasta)",
                        action: WorkbenchAction::LoadSequenceFasta,
                        rationale: format!(
                            "{} peaks ready, but primary amino-acid sequence is required for assignment.",
                            self.picked_peaks_count
                        ),
                        blocked_reason: Some("Blocked: primary protein sequence not loaded".to_string()),
                    }
                } else if self.qc_verdict == Some(DatasetQcVerdict::Abstain) {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Inspect QC Abstention",
                        action: WorkbenchAction::InspectDatasetQcReport,
                        rationale: "Dataset QC abstained from high-confidence calls; inspect 3D registration, sign polarity, or referencing.".to_string(),
                        blocked_reason: Some("Blocked: Dataset QC verdict is ABSTAIN".to_string()),
                    }
                } else {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Run Assignment",
                        action: WorkbenchAction::RunBackboneAssignment,
                        rationale: format!(
                            "{} peaks & {} aa sequence ready for automated Evo-Graph backbone assignment.",
                            self.picked_peaks_count, self.sequence_len
                        ),
                        blocked_reason: None,
                    }
                }
            }
            WorkflowStage::FoldAndDynamics => WhatNextRecommendation {
                stage,
                button_label: "What Next: Fold 3D Ensemble",
                action: WorkbenchAction::FoldStructureEnsemble,
                rationale: format!(
                    "{}/{} residues assigned ({} HC >= 0.80); proceed to sidechains, 3D folding, or relaxation dynamics.",
                    self.assigned_residues_count, self.sequence_len, self.high_confidence_count
                ),
                blocked_reason: None,
            },
            WorkflowStage::ExportAndReport => {
                if self.exported_in_session {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Export HTML Report",
                        action: WorkbenchAction::ExportQualityReportHtml,
                        rationale: "Session saved; export publication-ready HTML/SVG quality & strip ladder report.".to_string(),
                        blocked_reason: None,
                    }
                } else {
                    WhatNextRecommendation {
                        stage,
                        button_label: "What Next: Save Session (.nef)",
                        action: WorkbenchAction::SaveSessionNef,
                        rationale: "Save lossless NEF v1.1 session and export NMR-STAR v3 shifts + HTML quality report.".to_string(),
                        blocked_reason: None,
                    }
                }
            }
        }
    }
}

/// Render the structured 9-menu Top Menu Bar (`File`, `Process`, `Spectrum`, `Peaks`,
/// `Assignment`, `Structure`, `Dynamics`, `Chemistry`, `Help`) and return any clicked actions.
pub fn draw_top_menu_bar(
    ui: &mut egui::Ui,
    readiness: &WorkspaceReadiness,
) -> Vec<WorkbenchAction> {
    let mut emitted = Vec::new();

    egui::MenuBar::new().ui(ui, |ui| {
        for menu in TopMenu::ALL {
            ui.menu_button(menu.label(), |ui| {
                ui.set_min_width(290.0);
                for action in WorkbenchAction::ALL
                    .iter()
                    .copied()
                    .filter(|a| a.menu() == menu)
                {
                    let meta = action.metadata();
                    let clicked = draw_menu_item_row(ui, &meta);
                    if clicked {
                        emitted.push(action);
                        ui.close();
                    }
                    if meta.menu_separator_after {
                        ui.separator();
                    }
                }
            });
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let help_btn = egui::Button::new(
                egui::RichText::new("? Keys")
                    .monospace()
                    .size(11.5)
                    .color(egui::Color32::from_rgb(195, 215, 240)),
            )
            .fill(egui::Color32::from_rgb(28, 35, 48));
            if ui
                .add(help_btn)
                .on_hover_text("Open two-key NMR spectrum keyboard cheat-sheet (?)")
                .clicked()
            {
                emitted.push(WorkbenchAction::OpenKeyboardCheatSheet);
            }

            let palette_btn = egui::Button::new(
                egui::RichText::new("⌘/Ctrl+K Palette")
                    .monospace()
                    .size(11.5)
                    .color(egui::Color32::from_rgb(145, 225, 255)),
            )
            .fill(egui::Color32::from_rgb(24, 38, 56));
            if ui
                .add(palette_btn)
                .on_hover_text(
                    "Open fuzzy command palette across all workbench actions (Cmd/Ctrl+K)",
                )
                .clicked()
            {
                emitted.push(WorkbenchAction::OpenCommandPalette);
            }

            if readiness.assigned_residues_count > 0 {
                ui.label(
                    egui::RichText::new(format!(
                        "{}/{} assigned ({} HC)",
                        readiness.assigned_residues_count,
                        readiness.sequence_len,
                        readiness.high_confidence_count
                    ))
                    .small()
                    .monospace()
                    .color(egui::Color32::from_rgb(155, 195, 220)),
                );
            }
        });
    });

    emitted
}

fn draw_menu_item_row(ui: &mut egui::Ui, meta: &ActionMetadata) -> bool {
    let badge_str = match (meta.chord_badge, meta.modifier_shortcut) {
        (Some(ch), Some(sc)) => format!("[{ch}] {sc}"),
        (Some(ch), None) => format!("[{ch}]"),
        (None, Some(sc)) => sc.to_string(),
        (None, None) => String::new(),
    };

    let btn = if badge_str.is_empty() {
        egui::Button::new(meta.title)
    } else {
        egui::Button::new(meta.title).shortcut_text(
            egui::RichText::new(badge_str)
                .monospace()
                .size(11.0)
                .color(egui::Color32::from_rgb(135, 205, 240)),
        )
    };

    ui.add(btn).on_hover_text(meta.description).clicked()
}

/// Render the persistent 6-stage **Workflow Stepper**, `DatasetQcVerdict` pill (`PASS` / `WARN` /
/// `ABSTAIN`), missing-input callouts, and the contextual primary **"What Next?"** button.
pub fn draw_workflow_stepper(
    ui: &mut egui::Ui,
    readiness: &WorkspaceReadiness,
) -> Vec<WorkbenchAction> {
    let mut emitted = Vec::new();
    let statuses = readiness.all_stage_statuses();
    let rec = readiness.what_next();
    let missing = readiness.missing_inputs();

    egui::Frame::group(ui.style())
        .fill(egui::Color32::from_rgb(18, 24, 34))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 56, 78)))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;

                // 1. 6-Stage Stepper Pills
                for (idx, (stage, status)) in statuses.iter().copied().enumerate() {
                    if idx > 0 {
                        ui.label(
                            egui::RichText::new("›")
                                .strong()
                                .color(egui::Color32::from_rgb(95, 112, 138)),
                        );
                    }

                    let (prefix, fill, stroke, text_col) = match status {
                        StageStatus::Completed => (
                            "✓",
                            egui::Color32::from_rgb(22, 68, 52),
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(62, 175, 122)),
                            egui::Color32::from_rgb(185, 248, 215),
                        ),
                        StageStatus::Active => (
                            "●",
                            egui::Color32::from_rgb(26, 74, 118),
                            egui::Stroke::new(1.6, egui::Color32::from_rgb(105, 215, 255)),
                            egui::Color32::WHITE,
                        ),
                        StageStatus::Blocked => (
                            "⚠",
                            egui::Color32::from_rgb(96, 46, 28),
                            egui::Stroke::new(1.4, egui::Color32::from_rgb(245, 152, 76)),
                            egui::Color32::from_rgb(255, 222, 185),
                        ),
                        StageStatus::Upcoming => (
                            "○",
                            egui::Color32::from_rgb(26, 32, 44),
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(56, 68, 88)),
                            egui::Color32::from_rgb(148, 164, 188),
                        ),
                    };

                    let pill = egui::Button::new(
                        egui::RichText::new(format!("{prefix} {}", stage.numbered_label()))
                            .size(11.5)
                            .strong()
                            .color(text_col),
                    )
                    .fill(fill)
                    .stroke(stroke);

                    let hover = format!(
                        "{} ({status:?}) — click to trigger default stage action",
                        stage.numbered_label()
                    );
                    if ui.add(pill).on_hover_text(hover).clicked() {
                        let act = if stage == rec.stage {
                            rec.action
                        } else {
                            stage.default_action()
                        };
                        emitted.push(act);
                    }
                }

                // 2. Prominent DatasetQcVerdict Badge (PASS green / WARN amber / ABSTAIN red)
                if let Some(qc) = readiness.qc_badge() {
                    ui.separator();
                    let qc_btn = egui::Button::new(
                        egui::RichText::new(qc.label)
                            .monospace()
                            .size(11.5)
                            .strong()
                            .color(egui::Color32::from_rgb(
                                qc.text_rgb[0],
                                qc.text_rgb[1],
                                qc.text_rgb[2],
                            )),
                    )
                    .fill(egui::Color32::from_rgb(
                        qc.fill_rgb[0],
                        qc.fill_rgb[1],
                        qc.fill_rgb[2],
                    ))
                    .stroke(egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgb(
                            qc.stroke_rgb[0],
                            qc.stroke_rgb[1],
                            qc.stroke_rgb[2],
                        ),
                    ));

                    if ui.add(qc_btn).on_hover_text(&qc.summary).clicked() {
                        emitted.push(WorkbenchAction::InspectDatasetQcReport);
                    }
                }

                // 3. Missing-input / Blocked callout pill (compact)
                if let Some(first_missing) = missing.first() {
                    let extra = if missing.len() > 1 {
                        format!(" (+{})", missing.len() - 1)
                    } else {
                        String::new()
                    };
                    let all_missing_hover = missing.join("\n• ");
                    ui.label(
                        egui::RichText::new(format!("⚠ {first_missing}{extra}"))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(255, 198, 112)),
                    )
                    .on_hover_text(format!("Actionable prerequisites:\n• {all_missing_hover}"));
                }

                // 4. Contextual Primary "What Next?" Action Button
                let is_abstain_btn = readiness.qc_verdict == Some(DatasetQcVerdict::Abstain)
                    && rec.stage == WorkflowStage::AssignSpins;
                let (btn_fill, btn_text_col) = if is_abstain_btn {
                    (
                        egui::Color32::from_rgb(235, 92, 98),
                        egui::Color32::from_rgb(25, 10, 12),
                    )
                } else if rec.blocked_reason.is_some() {
                    (
                        egui::Color32::from_rgb(245, 182, 72),
                        egui::Color32::from_rgb(28, 20, 8),
                    )
                } else {
                    (
                        egui::Color32::from_rgb(76, 222, 152),
                        egui::Color32::from_rgb(12, 26, 18),
                    )
                };

                let next_btn = egui::Button::new(
                    egui::RichText::new(format!("▶ {}", rec.button_label))
                        .size(12.0)
                        .strong()
                        .color(btn_text_col),
                )
                .fill(btn_fill);

                if ui.add(next_btn).on_hover_text(&rec.rationale).clicked() {
                    emitted.push(rec.action);
                }
            });
        });

    emitted
}

/// Convenience helper drawing both the structured Top Menu Bar and the 6-stage Workflow Stepper.
pub fn draw_workbench_chrome(
    ui: &mut egui::Ui,
    readiness: &WorkspaceReadiness,
) -> Vec<WorkbenchAction> {
    let mut actions = draw_top_menu_bar(ui, readiness);
    ui.add_space(2.0);
    actions.extend(draw_workflow_stepper(ui, readiness));
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_action_catalog_exceeds_55_and_covers_all_9_menus() {
        assert!(
            WorkbenchAction::ALL.len() >= 55,
            "Expected >= 55 WorkbenchAction variants, got {}",
            WorkbenchAction::ALL.len()
        );

        let unique: HashSet<WorkbenchAction> = WorkbenchAction::ALL.iter().copied().collect();
        assert_eq!(unique.len(), WorkbenchAction::ALL.len());

        for menu in TopMenu::ALL {
            let items = WorkbenchAction::actions_for_menu(menu);
            assert!(
                !items.is_empty(),
                "TopMenu::{menu:?} must have at least one registered WorkbenchAction"
            );
            for act in items {
                let meta = act.metadata();
                assert!(!meta.title.is_empty());
                assert!(!meta.description.is_empty());
                assert!(!meta.category.is_empty());
            }
        }
    }

    #[test]
    fn test_all_6_workflow_stage_transitions_deterministic() {
        // Stage 1: Empty workspace -> Ingest
        let mut r = WorkspaceReadiness::default();
        assert_eq!(r.active_stage(), WorkflowStage::Ingest);
        assert_eq!(r.stage_status(WorkflowStage::Ingest), StageStatus::Active);
        assert_eq!(r.what_next().action, WorkbenchAction::OpenIngestMatrix);
        assert_eq!(r.missing_inputs().len(), 2);

        // Stage 2: Raw FIDs ingested -> ProcessFids (AssignSpins proactively flagged Blocked without sequence)
        r.raw_fids_count = 6;
        r.unprocessed_fids_count = 6;
        assert_eq!(r.active_stage(), WorkflowStage::ProcessFids);
        assert_eq!(
            r.stage_status(WorkflowStage::Ingest),
            StageStatus::Completed
        );
        assert_eq!(
            r.stage_status(WorkflowStage::ProcessFids),
            StageStatus::Active
        );
        assert_eq!(
            r.stage_status(WorkflowStage::AssignSpins),
            StageStatus::Blocked
        );
        assert_eq!(r.what_next().action, WorkbenchAction::ProcessAllRawFids);

        // Stage 3: FIDs processed into 2D + 3D spectra -> PickPeaks
        r.unprocessed_fids_count = 0;
        r.spectra_2d_count = 1;
        r.spectra_3d_count = 5;
        assert_eq!(r.active_stage(), WorkflowStage::PickPeaks);
        assert_eq!(
            r.stage_status(WorkflowStage::ProcessFids),
            StageStatus::Completed
        );
        assert_eq!(
            r.stage_status(WorkflowStage::PickPeaks),
            StageStatus::Active
        );
        assert_eq!(r.what_next().action, WorkbenchAction::PickHsqc2dRoots);

        // Stage 4a: Peaks picked, sequence still missing -> AssignSpins Blocked -> LoadSequenceFasta
        r.picked_peaks_count = 412;
        r.spin_systems_count = 68;
        assert_eq!(r.active_stage(), WorkflowStage::AssignSpins);
        assert_eq!(
            r.stage_status(WorkflowStage::PickPeaks),
            StageStatus::Completed
        );
        assert_eq!(
            r.stage_status(WorkflowStage::AssignSpins),
            StageStatus::Blocked
        );
        let rec_blocked = r.what_next();
        assert_eq!(rec_blocked.action, WorkbenchAction::LoadSequenceFasta);
        assert!(rec_blocked.blocked_reason.is_some());

        // Stage 4b: Sequence loaded -> AssignSpins Active -> RunBackboneAssignment
        r.sequence_loaded = true;
        r.sequence_len = 76;
        assert_eq!(r.active_stage(), WorkflowStage::AssignSpins);
        assert_eq!(
            r.stage_status(WorkflowStage::AssignSpins),
            StageStatus::Active
        );
        assert!(r.missing_inputs().is_empty());
        assert_eq!(r.what_next().action, WorkbenchAction::RunBackboneAssignment);

        // Stage 5: Backbone assigned with QC PASS -> FoldAndDynamics Active
        r.assigned_residues_count = 66;
        r.high_confidence_count = 62;
        r.qc_verdict = Some(DatasetQcVerdict::Pass);
        r.qc_summary = Some("95.5% 3D support".to_string());
        assert_eq!(r.active_stage(), WorkflowStage::FoldAndDynamics);
        assert_eq!(
            r.stage_status(WorkflowStage::AssignSpins),
            StageStatus::Completed
        );
        assert_eq!(
            r.stage_status(WorkflowStage::FoldAndDynamics),
            StageStatus::Active
        );
        assert_eq!(r.what_next().action, WorkbenchAction::FoldStructureEnsemble);

        // Stage 6a: Structure/dynamics computed -> ExportAndReport Active
        r.has_structure_or_dynamics = true;
        assert_eq!(r.active_stage(), WorkflowStage::ExportAndReport);
        assert_eq!(
            r.stage_status(WorkflowStage::FoldAndDynamics),
            StageStatus::Completed
        );
        assert_eq!(
            r.stage_status(WorkflowStage::ExportAndReport),
            StageStatus::Active
        );
        assert_eq!(r.what_next().action, WorkbenchAction::SaveSessionNef);

        // Stage 6b: Session saved -> all 6 stages Completed
        r.exported_in_session = true;
        assert_eq!(r.active_stage(), WorkflowStage::ExportAndReport);
        for (stage, st) in r.all_stage_statuses() {
            assert_eq!(
                st,
                StageStatus::Completed,
                "Expected {stage:?} to be Completed"
            );
        }
        assert_eq!(
            r.what_next().action,
            WorkbenchAction::ExportQualityReportHtml
        );
    }

    #[test]
    fn test_qc_verdict_badges_and_abstain_blocking() {
        let mut r = WorkspaceReadiness {
            sequence_loaded: true,
            sequence_len: 76,
            spectra_2d_count: 1,
            spectra_3d_count: 3,
            picked_peaks_count: 180,
            spin_systems_count: 64,
            assigned_residues_count: 30,
            high_confidence_count: 0,
            qc_verdict: Some(DatasetQcVerdict::Pass),
            qc_summary: Some("clean lock alignment".to_string()),
            ..WorkspaceReadiness::default()
        };

        let pass_badge = r.qc_badge().expect("PASS badge");
        assert_eq!(pass_badge.label, "QC: PASS");
        assert!(pass_badge.fill_rgb[1] > pass_badge.fill_rgb[0]); // green dominant

        r.qc_verdict = Some(DatasetQcVerdict::Warn);
        let warn_badge = r.qc_badge().expect("WARN badge");
        assert_eq!(warn_badge.label, "QC: WARN");
        assert!(warn_badge.stroke_rgb[0] > 200 && warn_badge.stroke_rgb[1] > 150); // amber

        r.qc_verdict = Some(DatasetQcVerdict::Abstain);
        r.qc_summary = Some("< 25% 3D root support".to_string());
        let abstain_badge = r.qc_badge().expect("ABSTAIN badge");
        assert_eq!(abstain_badge.label, "QC: ABSTAIN");
        assert!(abstain_badge.fill_rgb[0] > abstain_badge.fill_rgb[1]); // red dominant

        // ABSTAIN blocks AssignSpins & downstream stages and routes "What Next?" to QC inspection
        assert_eq!(r.active_stage(), WorkflowStage::AssignSpins);
        assert_eq!(
            r.stage_status(WorkflowStage::AssignSpins),
            StageStatus::Blocked
        );
        assert_eq!(
            r.stage_status(WorkflowStage::FoldAndDynamics),
            StageStatus::Blocked
        );
        assert_eq!(
            r.what_next().action,
            WorkbenchAction::InspectDatasetQcReport
        );
        assert!(r.missing_inputs().iter().any(|m| m.contains("QC ABSTAIN")));
    }
}
