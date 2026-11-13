//! Interactive sequence ribbon and live 2D IUPAC covalent amino-acid sidechain graph widget (`T10.1`).
//!
//! Provenance: Clean-room original implementation derived strictly from:
//! - Markley et al. (*J. Mol. Biol.* 1998, 280, 933–952 — IUPAC-IUBMB-IUPAB biochemical NMR atom
//!   nomenclature recommendations).
//! - wwPDB Chemical Component Dictionary (`CCD`, public domain / CC0 1.0) standard L-peptide
//!   monomer covalent topologies.
//! - Ulrich et al. (*Nucleic Acids Res.* 2008, 36, D402–D408 — Biological Magnetic Resonance Data
//!   Bank empirical chemical shift statistics).

use std::collections::{BTreeMap, BTreeSet};

use eframe::egui;
use nmr_core::aromatic::{
    classify_his_state, HisStateClassification, HisTautomer, HisTautomerEvidence,
};
use nmr_core::nmrstar::Reference;
use nmr_core::priors::{three_to_one, Priors};
use nmr_core::topology::{
    compute_atom_bmrb_zscore, residue_layout_2d, residue_topology, AtomNode2D, CovalentBond2D,
    CovalentBondOrder, Element, ProchiralGroup, ResidueLayout2D, ResidueTopology, ResidueVariant,
    RingPolygon2D, SidechainPriors,
};

use crate::contour_painter::SyncCrosshair;

/// Visual classification of an atom node in the 2D covalent sidechain inspector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomZScoreClass {
    /// Assigned shift within `|z| <= 1.5` of BMRB empirical mean (emerald green fill).
    NormalGreen,
    /// Assigned shift in `1.5 < |z| <= 3.0` warning band (amber warning fill).
    WarningAmber,
    /// Assigned shift in `|z| > 3.0` outlier regime (crimson outlier fill).
    OutlierCrimson,
    /// Assigned NMR-observable atom lacking a finite positive-sigma BMRB prior — calm slate neutral fill.
    NoPriorNeutral,
    /// Unassigned spin-1/2 NMR-active atom (`1H`, `13C`, `15N`) — hollow circle.
    UnassignedHollow,
    /// Non-observable heteroatom (`O`, `S`) or unprotonated sidechain carbonyl/quaternary carbon
    /// (`O`, `CG` on `Phe`/`Tyr`/`Trp`/`His`, etc.) — subtle muted node.
    MutedNonObservable,
}

impl AtomZScoreClass {
    /// Classify a signed or unsigned BMRB Z-score into one of the three empirical bands.
    #[inline]
    pub fn from_zscore(z_score: f64) -> Self {
        let az = z_score.abs();
        if az <= 1.5 {
            Self::NormalGreen
        } else if az <= 3.0 {
            Self::WarningAmber
        } else {
            Self::OutlierCrimson
        }
    }

    /// Short human-readable status label for tooltips and legends.
    pub const fn status_label(self) -> &'static str {
        match self {
            Self::NormalGreen => "Normal (|z| <= 1.5)",
            Self::WarningAmber => "Warning (1.5 < |z| <= 3.0)",
            Self::OutlierCrimson => "Outlier (|z| > 3.0)",
            Self::NoPriorNeutral => "No BMRB prior",
            Self::UnassignedHollow => "Unassigned (NMR-active)",
            Self::MutedNonObservable => "Non-observable / Quaternary",
        }
    }

    /// Node interior fill color on dark workbench canvases.
    pub const fn fill_color(self) -> egui::Color32 {
        match self {
            Self::NormalGreen => egui::Color32::from_rgb(22, 163, 74),
            Self::WarningAmber => egui::Color32::from_rgb(217, 119, 6),
            Self::OutlierCrimson => egui::Color32::from_rgb(220, 38, 38),
            Self::NoPriorNeutral => egui::Color32::from_rgb(59, 82, 114),
            // Hollow interior matches canvas background so covalent bond lines terminate cleanly at rim
            Self::UnassignedHollow => egui::Color32::from_rgb(18, 22, 31),
            Self::MutedNonObservable => egui::Color32::from_rgb(36, 42, 54),
        }
    }

    /// Node border stroke (`width, color`) for the given element.
    pub fn border_stroke(self, element: Element) -> egui::Stroke {
        match self {
            Self::NormalGreen => egui::Stroke::new(1.8, egui::Color32::from_rgb(134, 239, 172)),
            Self::WarningAmber => egui::Stroke::new(1.8, egui::Color32::from_rgb(253, 224, 71)),
            Self::OutlierCrimson => egui::Stroke::new(2.0, egui::Color32::from_rgb(252, 165, 165)),
            Self::NoPriorNeutral => egui::Stroke::new(1.8, egui::Color32::from_rgb(148, 184, 226)),
            Self::UnassignedHollow => {
                let ring_col = match element {
                    Element::C => egui::Color32::from_rgb(125, 211, 252),
                    Element::N => egui::Color32::from_rgb(129, 140, 248),
                    Element::H => egui::Color32::from_rgb(203, 213, 225),
                    Element::O | Element::S => egui::Color32::from_rgb(100, 116, 139),
                };
                egui::Stroke::new(1.9, ring_col)
            }
            Self::MutedNonObservable => {
                egui::Stroke::new(1.0, egui::Color32::from_rgb(76, 86, 106))
            }
        }
    }

    /// Foreground atom-label color inside the node circle.
    pub const fn label_color(self) -> egui::Color32 {
        match self {
            Self::NormalGreen
            | Self::WarningAmber
            | Self::OutlierCrimson
            | Self::NoPriorNeutral => egui::Color32::WHITE,
            Self::UnassignedHollow => egui::Color32::from_rgb(226, 232, 240),
            Self::MutedNonObservable => egui::Color32::from_rgb(138, 148, 168),
        }
    }
}

/// Chemical shift assignment metadata for a single `(residue_idx, atom_name)` spin.
#[derive(Clone, Debug, PartialEq)]
pub struct AssignedAtomInfo {
    /// Assigned chemical shift in ppm.
    pub ppm: f64,
    /// Optional 1-sigma chemical shift uncertainty in ppm.
    pub uncertainty_ppm: Option<f64>,
    /// Optional NMR-STAR `_Atom_chem_shift.Ambiguity_code` (`1`, `2`, `3`, ...).
    pub ambiguity_code: Option<u8>,
}

impl AssignedAtomInfo {
    /// Construct an assigned shift with no explicit uncertainty or ambiguity override.
    #[inline]
    pub const fn new(ppm: f64) -> Self {
        Self {
            ppm,
            uncertainty_ppm: None,
            ambiguity_code: None,
        }
    }

    /// Construct an assigned shift with explicit uncertainty and NMR-STAR ambiguity code.
    #[inline]
    pub const fn with_metadata(
        ppm: f64,
        uncertainty_ppm: Option<f64>,
        ambiguity_code: Option<u8>,
    ) -> Self {
        Self {
            ppm,
            uncertainty_ppm,
            ambiguity_code,
        }
    }
}

/// Structured rich hover tooltip strings for an atom node (`Atom`, `Assigned ppm`, `BMRB mu +- sd`,
/// `Z-score`, `Ambiguity_code`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomTooltipContent {
    pub atom: String,
    pub assigned_ppm: String,
    pub bmrb_prior: String,
    pub z_score: String,
    pub ambiguity_code: String,
}

/// Evaluated visual & statistical state of one atom in a residue's 2D covalent graph.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedAtomNode {
    pub atom_name: &'static str,
    pub element: Element,
    pub is_backbone: bool,
    pub is_nmr_observable: bool,
    pub assigned_ppm: Option<f64>,
    pub uncertainty_ppm: Option<f64>,
    pub ambiguity_code: Option<u8>,
    pub bmrb_mu_sd: Option<(f64, f64)>,
    pub z_score: Option<f64>,
    pub class: AtomZScoreClass,
}

impl EvaluatedAtomNode {
    /// Compact shift caption rendered adjacent to assigned atom nodes (`"53.35"` or `"53.35±0.05"`).
    pub fn shift_caption(&self) -> Option<String> {
        let ppm = self.assigned_ppm?;
        match self.uncertainty_ppm {
            Some(unc) if unc > 0.0 && unc.is_finite() => Some(format!("{ppm:.2}±{unc:.2}")),
            _ => Some(format!("{ppm:.2}")),
        }
    }

    /// Build rich hover tooltip fields (`Atom`, `Assigned ppm`, `BMRB mu +- sd`, `Z-score`, `Ambiguity_code`).
    pub fn build_tooltip(
        &self,
        res_three: &str,
        seq_id: i32,
        variant: ResidueVariant,
    ) -> AtomTooltipContent {
        let nuc_tag = match self.element {
            Element::H => "1H",
            Element::C => "13C",
            Element::N => "15N",
            Element::O => "16O",
            Element::S => "32S",
        };
        let variant_suffix = if variant == ResidueVariant::Canonical {
            String::new()
        } else {
            format!(" [{}]", variant.short_badge())
        };
        let atom = format!(
            "{res_three} {seq_id} — {} ({nuc_tag}){variant_suffix}",
            self.atom_name
        );

        let assigned_ppm = match (self.assigned_ppm, self.uncertainty_ppm) {
            (Some(v), Some(u)) if u > 0.0 && u.is_finite() => format!("{v:.3} +- {u:.3} ppm"),
            (Some(v), _) => format!("{v:.3} ppm"),
            (None, _) => "Unassigned".to_string(),
        };

        let bmrb_prior = match self.bmrb_mu_sd {
            Some((mu, sd)) => format!("{mu:.3} +- {sd:.3} ppm"),
            None if self.is_nmr_observable => "No BMRB prior".to_string(),
            None => "N/A (non-observable)".to_string(),
        };

        let z_score = match self.z_score {
            Some(z) => format!("{z:+.2} sigma ({})", self.class.status_label()),
            None => self.class.status_label().to_string(),
        };

        let ambiguity_code = match self.ambiguity_code {
            Some(1) => "1 (Unique / stereospecific)".to_string(),
            Some(2) => "2 (Geminal / prochiral pair)".to_string(),
            Some(3) => "3 (Aromatic ring-flip symmetric)".to_string(),
            Some(c) => format!("{c}"),
            None => "—".to_string(),
        };

        AtomTooltipContent {
            atom,
            assigned_ppm,
            bmrb_prior,
            z_score,
            ambiguity_code,
        }
    }
}

/// Evaluate BMRB Z-score and visual node class for a single [`AtomNode2D`].
pub fn evaluate_atom_node(
    res_name: &str,
    variant: ResidueVariant,
    node: &AtomNode2D,
    assigned: Option<&AssignedAtomInfo>,
    inferred_ambiguity: Option<u8>,
    priors: &Priors,
    sc_priors: &SidechainPriors,
) -> EvaluatedAtomNode {
    if let Some(info) = assigned {
        let z_tuple =
            compute_atom_bmrb_zscore(res_name, variant, node.name, info.ppm, priors, sc_priors);
        let (bmrb_mu_sd, z_score, class) = match z_tuple {
            Some((mu, sd, z)) => (Some((mu, sd)), Some(z), AtomZScoreClass::from_zscore(z)),
            None if node.is_nmr_observable => (None, None, AtomZScoreClass::NoPriorNeutral),
            None => (None, None, AtomZScoreClass::MutedNonObservable),
        };
        return EvaluatedAtomNode {
            atom_name: node.name,
            element: node.element,
            is_backbone: node.is_backbone,
            is_nmr_observable: node.is_nmr_observable,
            assigned_ppm: Some(info.ppm),
            uncertainty_ppm: info.uncertainty_ppm,
            ambiguity_code: info.ambiguity_code.or(inferred_ambiguity).or(Some(1)),
            bmrb_mu_sd,
            z_score,
            class,
        };
    }

    // Even when unassigned, pre-populate BMRB prior (mu, sd) so hover tooltip shows expected shift!
    let dummy_prior =
        compute_atom_bmrb_zscore(res_name, variant, node.name, 0.0, priors, sc_priors)
            .map(|(mu, sd, _)| (mu, sd));

    let class = if node.is_nmr_observable {
        AtomZScoreClass::UnassignedHollow
    } else {
        AtomZScoreClass::MutedNonObservable
    };

    EvaluatedAtomNode {
        atom_name: node.name,
        element: node.element,
        is_backbone: node.is_backbone,
        is_nmr_observable: node.is_nmr_observable,
        assigned_ppm: None,
        uncertainty_ppm: None,
        ambiguity_code: inferred_ambiguity,
        bmrb_mu_sd: dummy_prior,
        z_score: None,
        class,
    }
}

/// Frequency navigation payload emitted when the user clicks an assigned atom node in the
/// 2D covalent sidechain inspector.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomJumpTarget {
    /// `0`-based residue index in the polymer sequence.
    pub residue_idx: usize,
    /// Author / NMR-STAR sequence number (`Seq_ID`).
    pub seq_id: i32,
    /// `3`-letter IUPAC residue name (`"ALA"`..`"VAL"`).
    pub residue_name: String,
    /// Clicked IUPAC atom name (`"CA"`, `"CB"`, `"HB2"`, `"HD1"`, `"N"`, ...).
    pub atom_name: String,
    /// Chemical element of the clicked atom (`H`, `C`, `N`).
    pub element: Element,
    /// Assigned chemical shift (ppm) of the clicked atom.
    pub ppm: f64,
    /// Covalently bonded heavy atom name when `element == Element::H`.
    pub attached_heavy_atom: Option<String>,
    /// Assigned shift (ppm) of the covalently bonded heavy atom, if available.
    pub attached_heavy_ppm: Option<f64>,
    /// Assigned shift (ppm) of a directly bonded proton when `element` is `C` or `N`.
    pub attached_proton_ppm: Option<f64>,
    /// Residue's assigned amide `1H` root shift (ppm), if available (for 3D amide strip centering).
    pub backbone_h_ppm: Option<f64>,
    /// Residue's assigned amide `15N` root shift (ppm), if available (for 3D amide strip centering).
    pub backbone_n_ppm: Option<f64>,
}

impl AtomJumpTarget {
    /// Return `(h_ppm, c_ppm)` when jumping a `2D 1H-13C HSQC/HMQC` or `3D HCCH-TOCSY` plane.
    pub fn c13_plane_coords(&self) -> Option<(f64, f64)> {
        match self.element {
            Element::C => Some((self.attached_proton_ppm?, self.ppm)),
            Element::H => {
                let heavy_name = self.attached_heavy_atom.as_deref()?;
                if heavy_name.starts_with('C') {
                    Some((self.ppm, self.attached_heavy_ppm?))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Return `(h_ppm, n_ppm)` when jumping a `2D 1H-15N HSQC` plane.
    pub fn n15_plane_coords(&self) -> Option<(f64, f64)> {
        match self.element {
            Element::N => Some((self.attached_proton_ppm.or(self.backbone_h_ppm)?, self.ppm)),
            Element::H => {
                let heavy_name = self.attached_heavy_atom.as_deref()?;
                if heavy_name.starts_with('N') {
                    Some((self.ppm, self.attached_heavy_ppm?))
                } else {
                    Some((self.backbone_h_ppm?, self.backbone_n_ppm?))
                }
            }
            Element::C => Some((self.backbone_h_ppm?, self.backbone_n_ppm?)),
            _ => None,
        }
    }

    /// Snap a synchronized workbench [`SyncCrosshair`] directly to this atom's 2D/3D frequencies.
    pub fn apply_to_crosshair(&self, crosshair: &mut SyncCrosshair) {
        match self.element {
            Element::C => {
                let h = self
                    .backbone_h_ppm
                    .or(self.attached_proton_ppm)
                    .unwrap_or(crosshair.h_ppm);
                let n = self.backbone_n_ppm.unwrap_or(crosshair.n_ppm);
                crosshair.snap_to(h, n, Some(self.ppm));
            }
            Element::N => {
                let h = self
                    .attached_proton_ppm
                    .or(self.backbone_h_ppm)
                    .unwrap_or(crosshair.h_ppm);
                crosshair.snap_to(h, self.ppm, crosshair.c_ppm);
            }
            Element::H => {
                let is_nh = self
                    .attached_heavy_atom
                    .as_deref()
                    .is_some_and(|hv| hv.starts_with('N'));
                if is_nh {
                    let n = self
                        .attached_heavy_ppm
                        .or(self.backbone_n_ppm)
                        .unwrap_or(crosshair.n_ppm);
                    crosshair.snap_to(self.ppm, n, crosshair.c_ppm);
                } else {
                    let n = self.backbone_n_ppm.unwrap_or(crosshair.n_ppm);
                    let c = self.attached_heavy_ppm.or(crosshair.c_ppm);
                    crosshair.snap_to(self.ppm, n, c);
                }
            }
            Element::O | Element::S => {}
        }
    }
}

/// Per-frame output emitted by [`SidechainInspectorPanel`] for 1-call integration into `nmr-gui`.
#[derive(Clone, Debug, PartialEq)]
pub struct SidechainWidgetOutput {
    /// Active `0`-based selected residue index.
    pub selected_residue: usize,
    /// Populated when the user clicks an assigned atom node to jump 2D/3D spectrum strips.
    pub jump_to_atom: Option<AtomJumpTarget>,
    /// Populated when the user toggles a `Cys` redox or `His` protonation state variant.
    pub variant_override: Option<(usize, ResidueVariant)>,
}

/// Borrowed workspace view feeding [`SequenceRibbonState`] and [`SidechainInspectorPanel`].
pub struct SidechainWorkspaceInput<'a> {
    pub sequence: &'a [String],
    pub seq_ids: &'a [i32],
    pub assigned_atoms: BTreeMap<(usize, String), AssignedAtomInfo>,
    pub backbone_confidences: BTreeMap<usize, f64>,
    pub locked_residues: BTreeSet<usize>,
    pub priors: &'a Priors,
    pub sc_priors: &'a SidechainPriors,
}

impl<'a> SidechainWorkspaceInput<'a> {
    /// Build workspace input directly from an NMR-STAR / solver [`Reference`] shift table.
    pub fn from_reference(
        ref_data: &'a Reference,
        priors: &'a Priors,
        sc_priors: &'a SidechainPriors,
    ) -> Self {
        let mut assigned_atoms = BTreeMap::new();
        for (&(res_idx, ref atom), &ppm) in &ref_data.shifts {
            assigned_atoms.insert((res_idx, atom.clone()), AssignedAtomInfo::new(ppm));
        }
        Self {
            sequence: &ref_data.sequence,
            seq_ids: &ref_data.seq_ids,
            assigned_atoms,
            backbone_confidences: BTreeMap::new(),
            locked_residues: BTreeSet::new(),
            priors,
            sc_priors,
        }
    }

    /// Build workspace input from explicit `AssignedAtomInfo` entries (including uncertainties
    /// and NMR-STAR ambiguity codes).
    pub fn from_assigned_map(
        sequence: &'a [String],
        seq_ids: &'a [i32],
        assigned_atoms: BTreeMap<(usize, String), AssignedAtomInfo>,
        priors: &'a Priors,
        sc_priors: &'a SidechainPriors,
    ) -> Self {
        Self {
            sequence,
            seq_ids,
            assigned_atoms,
            backbone_confidences: BTreeMap::new(),
            locked_residues: BTreeSet::new(),
            priors,
            sc_priors,
        }
    }

    /// Attach per-residue backbone assignment confidences (`[0.0, 1.0]`) for ribbon cell coloring.
    pub fn with_backbone_confidences(mut self, confs: BTreeMap<usize, f64>) -> Self {
        self.backbone_confidences = confs;
        self
    }

    /// Attach user-locked residue indices (`0`-based) for gold lock badges on the sequence ribbon.
    pub fn with_locked_residues(mut self, locks: BTreeSet<usize>) -> Self {
        self.locked_residues = locks;
        self
    }

    /// Set uncertainty (`+- ppm`) on a specific `(residue_idx, atom_name)` entry.
    pub fn set_uncertainty(&mut self, residue_idx: usize, atom: &str, uncertainty_ppm: f64) {
        if let Some(entry) = self
            .assigned_atoms
            .get_mut(&(residue_idx, atom.to_string()))
        {
            entry.uncertainty_ppm = Some(uncertainty_ppm);
        }
    }

    /// Set `_Atom_chem_shift.Ambiguity_code` on a specific `(residue_idx, atom_name)` entry.
    pub fn set_ambiguity_code(&mut self, residue_idx: usize, atom: &str, code: u8) {
        if let Some(entry) = self
            .assigned_atoms
            .get_mut(&(residue_idx, atom.to_string()))
        {
            entry.ambiguity_code = Some(code);
        }
    }

    /// Look up [`AssignedAtomInfo`] for `(residue_idx, atom_name)`.
    pub fn assigned_info(&self, residue_idx: usize, atom: &str) -> Option<&AssignedAtomInfo> {
        self.assigned_atoms
            .get(&(residue_idx, atom.to_string()))
            .or_else(|| match atom {
                "H" => self.assigned_atoms.get(&(residue_idx, "HN".to_string())),
                "C" => self.assigned_atoms.get(&(residue_idx, "CO".to_string())),
                _ => None,
            })
    }
}

/// Persistent UI state for the interactive sequence ribbon and 2D covalent sidechain inspector.
#[derive(Clone, Debug)]
pub struct SequenceRibbonState {
    /// Currently selected `0`-based residue index.
    pub selected_residue: usize,
    /// Automatically inferred per-residue chemical state variants (`Cys` redox, `His` Pelton tautomer)
    /// recomputed from chemical shifts via [`Self::sync_inferred_variants_from_reference`].
    pub auto_inferred_variants: BTreeMap<usize, ResidueVariant>,
    /// Explicit user-selected per-residue chemical state overrides taking precedence over auto-inference.
    pub user_variant_overrides: BTreeMap<usize, ResidueVariant>,
    /// Legacy combined per-residue chemical state map kept synced for backward compatibility.
    pub variant_overrides: BTreeMap<usize, ResidueVariant>,
    /// Whether to display `ppm +- uncertainty` captions next to assigned nodes in the 2D diagram.
    pub show_ppm_captions: bool,
    /// Optional sample pH in `[0.0, 14.0]` forwarded to Pelton `His` tautomer/protonation state inference.
    pub sample_ph: Option<f64>,
    /// Remembered numeric pH draft (`0.0..=14.0`) for the UI checkbox + `DragValue` control.
    pub sample_ph_draft: f64,
    /// Screen-space hit rectangles of sequence ribbon cells from the most recent frame.
    pub last_ribbon_cell_rects: Vec<egui::Rect>,
    /// Screen-space hit rectangles of 2D atom nodes (`atom_name -> Rect`) from the most recent frame.
    pub last_atom_hit_rects: BTreeMap<String, egui::Rect>,
    /// Screen-space hit rectangles of `Cys`/`His` state variant buttons from the most recent frame.
    pub last_variant_button_rects: BTreeMap<ResidueVariant, egui::Rect>,
    /// Screen-space bounding rectangle of the 2D covalent diagram canvas from the most recent frame.
    pub last_canvas_rect: Option<egui::Rect>,
}

/// Alias for [`SequenceRibbonState`] representing the persistent state of the 2D sidechain inspector.
pub type SidechainInspectorState = SequenceRibbonState;

impl Default for SequenceRibbonState {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceRibbonState {
    /// Create a fresh [`SequenceRibbonState`] focused on residue `0`.
    pub fn new() -> Self {
        Self {
            selected_residue: 0,
            auto_inferred_variants: BTreeMap::new(),
            user_variant_overrides: BTreeMap::new(),
            variant_overrides: BTreeMap::new(),
            show_ppm_captions: true,
            sample_ph: None,
            sample_ph_draft: 6.5,
            last_ribbon_cell_rects: Vec::new(),
            last_atom_hit_rects: BTreeMap::new(),
            last_variant_button_rects: BTreeMap::new(),
            last_canvas_rect: None,
        }
    }

    fn rebuild_legacy_variant_overrides(&mut self) {
        for (&k, &v) in &self.auto_inferred_variants {
            if !self.user_variant_overrides.contains_key(&k) {
                self.variant_overrides.insert(k, v);
            }
        }
        for (&k, &v) in &self.user_variant_overrides {
            if v == ResidueVariant::Canonical {
                self.variant_overrides.remove(&k);
            } else {
                self.variant_overrides.insert(k, v);
            }
        }
    }

    /// Active [`ResidueVariant`] for `residue_idx`, checking `user_variant_overrides` first,
    /// then `auto_inferred_variants`, then legacy `variant_overrides`, then `Canonical`.
    pub fn variant_for(&self, residue_idx: usize) -> ResidueVariant {
        if let Some(&v) = self.user_variant_overrides.get(&residue_idx) {
            return v;
        }
        if let Some(&v) = self.auto_inferred_variants.get(&residue_idx) {
            return v;
        }
        self.variant_overrides
            .get(&residue_idx)
            .copied()
            .unwrap_or(ResidueVariant::Canonical)
    }

    /// Set an explicit user chemical state variant override for `residue_idx`.
    ///
    /// Selecting [`ResidueVariant::Canonical`] records an explicit `Canonical` entry in
    /// `user_variant_overrides` (rather than removing the key) so subsequent automatic shift
    /// inference via [`Self::sync_inferred_variants_from_reference`] never silently overrides
    /// the user's explicit `Canonical` choice. Call [`Self::clear_user_variant_override`] to
    /// return a residue to automatic inference.
    pub fn set_variant(&mut self, residue_idx: usize, variant: ResidueVariant) {
        self.user_variant_overrides.insert(residue_idx, variant);
        if variant == ResidueVariant::Canonical {
            self.variant_overrides.remove(&residue_idx);
        } else {
            self.variant_overrides.insert(residue_idx, variant);
        }
    }

    /// Clear any explicit user variant override on `residue_idx`, falling back to `auto_inferred_variants` or `Canonical`.
    pub fn clear_user_variant_override(&mut self, residue_idx: usize) {
        self.user_variant_overrides.remove(&residue_idx);
        if let Some(&auto_v) = self.auto_inferred_variants.get(&residue_idx) {
            self.variant_overrides.insert(residue_idx, auto_v);
        } else {
            self.variant_overrides.remove(&residue_idx);
        }
    }

    /// Configure optional sample pH (`0.0..=14.0`) used by [`Self::sync_inferred_variants_from_reference`].
    pub fn set_sample_ph(&mut self, ph: Option<f64>) {
        match ph.filter(|v| v.is_finite()) {
            Some(v) => {
                let clamped = v.clamp(0.0, 14.0);
                self.sample_ph = Some(clamped);
                self.sample_ph_draft = clamped;
            }
            None => {
                self.sample_ph = None;
            }
        }
    }

    /// Recompute `auto_inferred_variants` directly from a [`SidechainWorkspaceInput`] snapshot.
    pub fn sync_inferred_variants_from_workspace(&mut self, input: &SidechainWorkspaceInput<'_>) {
        let mut r = Reference::new(input.sequence.to_vec(), input.seq_ids.to_vec());
        for (&(k, ref atom), info) in &input.assigned_atoms {
            if info.ppm.is_finite() {
                r.shifts.insert((k, atom.clone()), info.ppm);
            }
        }
        self.sync_inferred_variants_from_reference(&r);
    }

    /// Recompute `auto_inferred_variants` from `reference` (`Cys` `13CB` redox regime + `His`
    /// Pelton `classify_his_state` `Resolved` tautomer), clearing stale auto-inferred entries
    /// whenever shifts change or abstain, WITHOUT ever overwriting `user_variant_overrides`.
    pub fn sync_inferred_variants_from_reference(&mut self, reference: &Reference) {
        let prev_auto_keys: Vec<usize> = self.auto_inferred_variants.keys().copied().collect();
        self.auto_inferred_variants.clear();
        for k in prev_auto_keys {
            if !self.user_variant_overrides.contains_key(&k) {
                self.variant_overrides.remove(&k);
            }
        }

        let valid_ph = self
            .sample_ph
            .filter(|v| v.is_finite() && (0.0..=14.0).contains(v));

        for (k, res_raw) in reference.sequence.iter().enumerate() {
            let Some(topo) = residue_topology(res_raw) else {
                continue;
            };
            match topo.three_letter {
                "CYS" => {
                    if let Some(&cb) = reference
                        .shifts
                        .get(&(k, "CB".to_string()))
                        .filter(|v| v.is_finite())
                    {
                        // Kornhaber et al. (2006) / Sharma & Rajarathnam (2000):
                        // reduced Cys SH CB ~ 28.0 +- 2.1 ppm; oxidized disulfide CB ~ 40.7 +- 3.4 ppm.
                        if cb <= 32.5 {
                            self.auto_inferred_variants
                                .insert(k, ResidueVariant::CysReduced);
                        } else if cb >= 35.5 {
                            self.auto_inferred_variants
                                .insert(k, ResidueVariant::CysOxidized);
                        }
                    }
                }
                "HIS" => {
                    let ev = HisTautomerEvidence {
                        nd1_ppm: reference.shifts.get(&(k, "ND1".to_string())).copied(),
                        ne2_ppm: reference.shifts.get(&(k, "NE2".to_string())).copied(),
                        cg_ppm: reference.shifts.get(&(k, "CG".to_string())).copied(),
                        cd2_ppm: reference.shifts.get(&(k, "CD2".to_string())).copied(),
                        hd2_ppm: reference.shifts.get(&(k, "HD2".to_string())).copied(),
                        ce1_ppm: reference.shifts.get(&(k, "CE1".to_string())).copied(),
                        he1_ppm: reference.shifts.get(&(k, "HE1".to_string())).copied(),
                        cb_ppm: reference.shifts.get(&(k, "CB".to_string())).copied(),
                        sample_ph: valid_ph,
                        metal_coordinated_hint: false,
                    };
                    // P1-2: Require >= 1 imidazole 15N observation (`ND1` or `NE2`) before auto-applying
                    // any static `His` tautomer/protonation state in the GUI inspector!
                    let has_his_15n = ev.nd1_ppm.is_some() || ev.ne2_ppm.is_some();
                    if has_his_15n {
                        let (state, conf) = classify_his_state(&ev);
                        if conf >= 0.75 {
                            if let HisStateClassification::Resolved(taut) = state {
                                let var = match taut {
                                    HisTautomer::Ne2H => ResidueVariant::HisNe2H,
                                    HisTautomer::Nd1H => ResidueVariant::HisNd1H,
                                    HisTautomer::DoublyProtonated => {
                                        ResidueVariant::HisDoublyProtonated
                                    }
                                };
                                self.auto_inferred_variants.insert(k, var);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        self.rebuild_legacy_variant_overrides();
    }
}

/// Isotropic 2D projection mapping normalized layout coordinates `(x, y)` onto an `egui::Rect`.
#[derive(Clone, Copy, Debug)]
pub struct CanvasProjection {
    pub rect: egui::Rect,
    pub mid_x: f64,
    pub mid_y: f64,
    pub scale: f32,
}

impl CanvasProjection {
    /// Compute a uniform isotropic projection fitting `layout` inside `rect` with label margins.
    pub fn fit_layout(layout: &ResidueLayout2D, rect: egui::Rect) -> Self {
        let (min_x, max_x, min_y, max_y) = layout.bounding_box();
        let pad_x = 1.15;
        let pad_y = 0.95;
        let span_x = ((max_x - min_x) + 2.0 * pad_x).max(2.5) as f32;
        let span_y = ((max_y - min_y) + 2.0 * pad_y).max(2.5) as f32;
        let scale = (rect.width() / span_x)
            .min(rect.height() / span_y)
            .clamp(14.0, 64.0);
        Self {
            rect,
            mid_x: 0.5 * (min_x + max_x),
            mid_y: 0.5 * (min_y + max_y),
            scale,
        }
    }

    /// Map normalized covalent coordinates `(x, y)` (`+y` up) to screen `egui::Pos2` (`+y` down).
    #[inline]
    pub fn to_screen(&self, x: f64, y: f64) -> egui::Pos2 {
        let c = self.rect.center();
        egui::pos2(
            c.x + ((x - self.mid_x) as f32) * self.scale,
            c.y - ((y - self.mid_y) as f32) * self.scale,
        )
    }

    /// Circle radius (in pixels) for an atom node of `element`.
    #[inline]
    pub fn node_radius(&self, element: Element) -> f32 {
        match element {
            Element::H => (0.20 * self.scale).clamp(7.5, 12.5),
            _ => (0.25 * self.scale).clamp(9.5, 15.5),
        }
    }
}

fn infer_default_ambiguity_codes(
    topo: &ResidueTopology,
    res_idx: usize,
    input: &SidechainWorkspaceInput<'_>,
) -> BTreeMap<&'static str, u8> {
    let mut codes = BTreeMap::new();
    for pg in topo.prochiral_groups {
        match pg {
            ProchiralGroup::Methylene { protons, .. } => {
                let p0 = input.assigned_info(res_idx, protons[0]).map(|i| i.ppm);
                let p1 = input.assigned_info(res_idx, protons[1]).map(|i| i.ppm);
                let code = match (p0, p1) {
                    (Some(a), Some(b)) if (a - b).abs() < 0.005 => 2,
                    (Some(_), None) | (None, Some(_)) => 2,
                    _ => 2,
                };
                codes.insert(protons[0], code);
                codes.insert(protons[1], code);
            }
            ProchiralGroup::IsopropylMethyls {
                carbon_a,
                protons_a,
                carbon_b,
                protons_b,
            } => {
                codes.insert(carbon_a, 2);
                codes.insert(carbon_b, 2);
                for &p in protons_a.iter().chain(protons_b.iter()) {
                    codes.insert(p, 2);
                }
            }
            ProchiralGroup::SymmetricAromatic {
                carbon_a,
                proton_a,
                carbon_b,
                proton_b,
            } => {
                codes.insert(carbon_a, 3);
                codes.insert(proton_a, 3);
                codes.insert(carbon_b, 3);
                codes.insert(proton_b, 3);
            }
        }
    }
    codes
}

fn draw_dashed_segment(
    painter: &egui::Painter,
    p0: egui::Pos2,
    p1: egui::Pos2,
    stroke: egui::Stroke,
    dash_px: f32,
    gap_px: f32,
) {
    let delta = p1 - p0;
    let total = delta.length();
    if total < 1.0 {
        return;
    }
    let dir = delta / total;
    let step = (dash_px + gap_px).max(2.0);
    let mut t = 0.0f32;
    while t < total {
        let t_end = (t + dash_px).min(total);
        painter.line_segment([p0 + dir * t, p0 + dir * t_end], stroke);
        t += step;
    }
}

fn find_ring_for_bond<'a>(
    rings: &'a [RingPolygon2D],
    a: &str,
    b: &str,
) -> Option<&'a RingPolygon2D> {
    rings
        .iter()
        .find(|r| r.atoms.contains(&a) && r.atoms.contains(&b))
}

fn paint_covalent_bond(
    painter: &egui::Painter,
    proj: &CanvasProjection,
    layout: &ResidueLayout2D,
    bond: &CovalentBond2D,
) {
    let (Some(na), Some(nb)) = (layout.atom(bond.atom_a), layout.atom(bond.atom_b)) else {
        return;
    };
    let pa = proj.to_screen(na.x, na.y);
    let pb = proj.to_screen(nb.x, nb.y);
    let vec_ab = pb - pa;
    let len = vec_ab.length();
    if len < 1.0 {
        return;
    }
    let dir = vec_ab / len;
    let normal = egui::vec2(-dir.y, dir.x);
    let base_col = egui::Color32::from_rgb(148, 163, 184);
    let accent_col = egui::Color32::from_rgb(186, 202, 224);

    match bond.order {
        CovalentBondOrder::Single => {
            painter.line_segment([pa, pb], egui::Stroke::new(1.8, base_col));
        }
        CovalentBondOrder::Double => {
            let off = normal * 2.5;
            let st = egui::Stroke::new(1.6, accent_col);
            painter.line_segment([pa + off, pb + off], st);
            painter.line_segment([pa - off, pb - off], st);
        }
        CovalentBondOrder::Resonance => {
            let off = normal * 2.5;
            painter.line_segment([pa + off, pb + off], egui::Stroke::new(1.7, accent_col));
            draw_dashed_segment(
                painter,
                pa - off,
                pb - off,
                egui::Stroke::new(1.4, base_col),
                4.0,
                3.0,
            );
        }
        CovalentBondOrder::Aromatic => {
            painter.line_segment([pa, pb], egui::Stroke::new(1.85, accent_col));
            let inward =
                if let Some(ring) = find_ring_for_bond(&layout.rings, bond.atom_a, bond.atom_b) {
                    let rc = proj.to_screen(ring.center_x, ring.center_y);
                    let mid = pa + vec_ab * 0.5;
                    if (rc - mid).dot(normal) >= 0.0 {
                        normal
                    } else {
                        -normal
                    }
                } else {
                    normal
                };
            let inset = inward * 4.2;
            let trim = (len * 0.18).min(8.0);
            if len > 2.0 * trim + 2.0 {
                let ia = pa + dir * trim + inset;
                let ib = pb - dir * trim + inset;
                draw_dashed_segment(
                    painter,
                    ia,
                    ib,
                    egui::Stroke::new(1.35, egui::Color32::from_rgb(125, 185, 235)),
                    4.5,
                    2.8,
                );
            }
        }
    }
}

fn outward_caption_direction(layout: &ResidueLayout2D, node: &AtomNode2D) -> egui::Vec2 {
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;
    let mut count = 0usize;
    for b in &layout.bonds {
        let nbr_name = if b.atom_a == node.name {
            Some(b.atom_b)
        } else if b.atom_b == node.name {
            Some(b.atom_a)
        } else {
            None
        };
        if let Some(nb_name) = nbr_name {
            if let Some(nb) = layout.atom(nb_name) {
                sum_dx += node.x - nb.x;
                sum_dy += node.y - nb.y;
                count += 1;
            }
        }
    }
    if count > 0 {
        let v = egui::vec2(sum_dx as f32, -(sum_dy as f32));
        if v.length_sq() > 1e-4 {
            return v.normalized();
        }
    }
    egui::vec2(0.0, 1.0)
}

fn build_jump_target(
    res_idx: usize,
    seq_id: i32,
    topo: &ResidueTopology,
    node: &AtomNode2D,
    ppm: f64,
    input: &SidechainWorkspaceInput<'_>,
) -> AtomJumpTarget {
    let attached_heavy_atom = node.attached_heavy.map(str::to_string);
    let attached_heavy_ppm = node
        .attached_heavy
        .and_then(|hv| input.assigned_info(res_idx, hv).map(|i| i.ppm));

    let attached_proton_ppm = if matches!(node.element, Element::C | Element::N) {
        topo.atoms
            .iter()
            .filter(|a| a.element == Element::H && a.attached_heavy == Some(node.name))
            .find_map(|h| input.assigned_info(res_idx, h.name).map(|i| i.ppm))
    } else {
        None
    };

    let backbone_h_ppm = input.assigned_info(res_idx, "H").map(|i| i.ppm);
    let backbone_n_ppm = input.assigned_info(res_idx, "N").map(|i| i.ppm);

    AtomJumpTarget {
        residue_idx: res_idx,
        seq_id,
        residue_name: topo.three_letter.to_string(),
        atom_name: node.name.to_string(),
        element: node.element,
        ppm,
        attached_heavy_atom,
        attached_heavy_ppm,
        attached_proton_ppm,
        backbone_h_ppm,
        backbone_n_ppm,
    }
}

/// Stateless renderer for the interactive sequence ribbon and 2D IUPAC covalent sidechain diagram.
pub struct SidechainInspectorPanel;

impl SidechainInspectorPanel {
    /// Render both the top scrollable sequence ribbon and the 2D IUPAC covalent sidechain diagram
    /// in a single 1-call integration helper.
    pub fn show(
        ui: &mut egui::Ui,
        state: &mut SequenceRibbonState,
        input: &SidechainWorkspaceInput<'_>,
    ) -> SidechainWidgetOutput {
        let _ = Self::show_ribbon(ui, state, input);
        ui.add_space(4.0);
        Self::show_inspector(ui, state, input)
    }

    /// Render the horizontal clickable sequence ribbon, updating `state.selected_residue` on click
    /// and returning `Some(clicked_residue_idx)` whenever a cell is clicked.
    pub fn show_ribbon(
        ui: &mut egui::Ui,
        state: &mut SequenceRibbonState,
        input: &SidechainWorkspaceInput<'_>,
    ) -> Option<usize> {
        state.last_ribbon_cell_rects.clear();
        if input.sequence.is_empty() {
            state.selected_residue = 0;
            return None;
        }
        if state.selected_residue >= input.sequence.len() {
            state.selected_residue = input.sequence.len() - 1;
        }

        let mut clicked_idx = None;

        egui::ScrollArea::horizontal()
            .id_salt("sidechain_seq_ribbon_scroll")
            .max_height(54.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for (k, three) in input.sequence.iter().enumerate() {
                        let one = three_to_one(three);
                        let seq_id = input.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                        let is_selected = state.selected_residue == k;
                        let is_locked = input.locked_residues.contains(&k);
                        let is_pro_or_nterm = k == 0 || three.eq_ignore_ascii_case("PRO");
                        let variant = state.variant_for(k);

                        // Count assigned NMR-observable atoms on residue k
                        let (assigned_cnt, observable_cnt) = if let Some(layout) =
                            residue_layout_2d(three, variant)
                        {
                            let obs = layout.atoms.iter().filter(|a| a.is_nmr_observable).count();
                            let asg = layout
                                .atoms
                                .iter()
                                .filter(|a| {
                                    a.is_nmr_observable && input.assigned_info(k, a.name).is_some()
                                })
                                .count();
                            (asg, obs)
                        } else {
                            (0, 0)
                        };

                        let fill = if let Some(&conf) = input.backbone_confidences.get(&k) {
                            if conf >= 0.80 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if conf >= 0.50 {
                                egui::Color32::from_rgb(148, 112, 28)
                            } else {
                                egui::Color32::from_rgb(145, 68, 42)
                            }
                        } else if assigned_cnt > 0 {
                            let frac = (assigned_cnt as f32) / (observable_cnt.max(1) as f32);
                            if frac >= 0.70 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if frac >= 0.30 {
                                egui::Color32::from_rgb(44, 98, 132)
                            } else {
                                egui::Color32::from_rgb(56, 68, 92)
                            }
                        } else if is_pro_or_nterm {
                            egui::Color32::from_rgb(38, 43, 54)
                        } else {
                            egui::Color32::from_rgb(48, 56, 74)
                        };

                        let stroke = if is_selected {
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 235, 255))
                        } else if is_locked {
                            egui::Stroke::new(1.8, egui::Color32::from_rgb(255, 215, 90))
                        } else {
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 76, 98))
                        };

                        let label = if is_locked {
                            format!("*{one}{seq_id}")
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
                        .min_size(egui::vec2(36.0, 28.0));

                        let resp = ui.add(btn);
                        state.last_ribbon_cell_rects.push(resp.rect);

                        if resp.clicked() {
                            state.selected_residue = k;
                            clicked_idx = Some(k);
                        }

                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new(format!("{three} {seq_id}")).strong());
                            ui.label(format!(
                                "NMR atoms assigned: {assigned_cnt} / {observable_cnt}"
                            ));
                            if variant != ResidueVariant::Canonical {
                                ui.label(format!("State: {}", variant.label()));
                            }
                            if let Some(&conf) = input.backbone_confidences.get(&k) {
                                ui.label(format!("Backbone confidence: {:.0}%", conf * 100.0));
                            }
                        });
                    }
                });
            });

        clicked_idx
    }

    /// Render the 2D IUPAC covalent sidechain inspector panel for `state.selected_residue`.
    pub fn show_inspector(
        ui: &mut egui::Ui,
        state: &mut SequenceRibbonState,
        input: &SidechainWorkspaceInput<'_>,
    ) -> SidechainWidgetOutput {
        state.last_atom_hit_rects.clear();
        state.last_variant_button_rects.clear();
        state.last_canvas_rect = None;

        if input.sequence.is_empty() {
            ui.label("Load a sequence or NMR-STAR file to inspect 2D covalent sidechain graphs.");
            return SidechainWidgetOutput {
                selected_residue: 0,
                jump_to_atom: None,
                variant_override: None,
            };
        }

        if state.selected_residue >= input.sequence.len() {
            state.selected_residue = input.sequence.len() - 1;
        }
        let res_idx = state.selected_residue;
        let res_raw = &input.sequence[res_idx];
        let seq_id = input
            .seq_ids
            .get(res_idx)
            .copied()
            .unwrap_or((res_idx + 1) as i32);

        let Some(topo) = residue_topology(res_raw) else {
            ui.label(format!(
                "Unsupported non-canonical residue: {res_raw} {seq_id}"
            ));
            return SidechainWidgetOutput {
                selected_residue: res_idx,
                jump_to_atom: None,
                variant_override: None,
            };
        };

        let mut variant_override: Option<(usize, ResidueVariant)> = None;
        let applicable_variants = ResidueVariant::applicable_for_residue(topo.three_letter);

        // Header bar: residue name, ppm toggle, optional sample pH control, and Cys/His variant selector
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{} {} ({})",
                    topo.three_letter, seq_id, topo.one_letter
                ))
                .strong()
                .size(14.0),
            );

            ui.separator();
            ui.checkbox(&mut state.show_ppm_captions, "Show ppm");

            ui.separator();
            let mut ph_enabled = state.sample_ph.is_some();
            let mut ph_changed = false;
            if ui
                .checkbox(&mut ph_enabled, "pH")
                .on_hover_text(
                    "Optional sample pH (0.0..=14.0) forwarded to Pelton His tautomer/protonation classification and aromatic ring assignment",
                )
                .changed()
            {
                if ph_enabled {
                    state.sample_ph = Some(state.sample_ph_draft.clamp(0.0, 14.0));
                } else {
                    state.sample_ph = None;
                }
                ph_changed = true;
            }
            if ph_enabled {
                let mut draft = state.sample_ph.unwrap_or(state.sample_ph_draft);
                if ui
                    .add(
                        egui::DragValue::new(&mut draft)
                            .range(0.0..=14.0)
                            .speed(0.05)
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    state.set_sample_ph(Some(draft));
                    ph_changed = true;
                }
            }
            if ph_changed {
                state.sync_inferred_variants_from_workspace(input);
            }

            let active_variant = state.variant_for(res_idx);
            if applicable_variants.len() > 1 {
                ui.separator();
                ui.label(egui::RichText::new("State:").weak());
                for &cand_var in applicable_variants {
                    let selected = active_variant == cand_var;
                    let v_resp = ui
                        .selectable_label(selected, cand_var.short_badge())
                        .on_hover_text(cand_var.label());
                    state
                        .last_variant_button_rects
                        .insert(cand_var, v_resp.rect);
                    if v_resp.clicked() && !selected {
                        state.set_variant(res_idx, cand_var);
                        variant_override = Some((res_idx, cand_var));
                    }
                }
            }
        });

        // Compact Z-score legend strip
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            for (cls, txt) in [
                (AtomZScoreClass::NormalGreen, "|z|<=1.5"),
                (AtomZScoreClass::WarningAmber, "1.5..3.0"),
                (AtomZScoreClass::OutlierCrimson, "|z|>3.0"),
                (AtomZScoreClass::NoPriorNeutral, "No BMRB prior"),
                (AtomZScoreClass::UnassignedHollow, "Unassigned"),
                (AtomZScoreClass::MutedNonObservable, "Quat/O/S"),
            ] {
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
                ui.painter().circle(
                    dot_rect.center(),
                    4.2,
                    cls.fill_color(),
                    cls.border_stroke(Element::C),
                );
                ui.label(egui::RichText::new(txt).size(10.5).weak());
            }
        });

        let effective_variant = state.variant_for(res_idx);
        let layout = topo.layout_2d(effective_variant);
        let inferred_amb = infer_default_ambiguity_codes(topo, res_idx, input);

        // Allocate 2D covalent diagram canvas
        let desired_w = ui.available_width().max(260.0);
        let desired_h = ui.available_height().clamp(240.0, 420.0);
        let (canvas_rect, _) =
            ui.allocate_exact_size(egui::vec2(desired_w, desired_h), egui::Sense::hover());
        state.last_canvas_rect = Some(canvas_rect);

        let painter = ui.painter_at(canvas_rect);
        painter.rect(
            canvas_rect,
            6.0,
            egui::Color32::from_rgb(16, 20, 28),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
            egui::StrokeKind::Inside,
        );

        let proj = CanvasProjection::fit_layout(&layout, canvas_rect);

        // 1. Paint all covalent bond edges underneath atom nodes
        for bond in &layout.bonds {
            paint_covalent_bond(&painter, &proj, &layout, bond);
        }

        // 2. Evaluate, paint, and attach interactive click + hover tooltips to every atom node
        let mut jump_to_atom: Option<AtomJumpTarget> = None;

        for node in &layout.atoms {
            let assigned = input.assigned_info(res_idx, node.name);
            let eval = evaluate_atom_node(
                topo.three_letter,
                effective_variant,
                node,
                assigned,
                inferred_amb.get(node.name).copied(),
                input.priors,
                input.sc_priors,
            );

            let center = proj.to_screen(node.x, node.y);
            let radius = proj.node_radius(node.element);
            let hit_rect =
                egui::Rect::from_center_size(center, egui::vec2(radius * 2.2, radius * 2.2));
            state
                .last_atom_hit_rects
                .insert(node.name.to_string(), hit_rect);

            let interact_id = ui.make_persistent_id(("sidechain_atom_node", res_idx, node.name));
            let resp = ui.interact(hit_rect, interact_id, egui::Sense::click());

            let stroke = if resp.hovered() {
                egui::Stroke::new(2.4, egui::Color32::from_rgb(125, 240, 255))
            } else {
                eval.class.border_stroke(node.element)
            };

            painter.circle(center, radius, eval.class.fill_color(), stroke);

            let font_sz = if node.name.len() >= 4 {
                (radius * 0.68).clamp(7.0, 9.5)
            } else if node.name.len() == 3 {
                (radius * 0.76).clamp(7.5, 10.5)
            } else {
                (radius * 0.88).clamp(8.5, 11.5)
            };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                node.name,
                egui::FontId::monospace(font_sz),
                eval.class.label_color(),
            );

            // Render assigned ppm (+- uncertainty) badge next to assigned nodes
            if state.show_ppm_captions {
                if let Some(caption) = eval.shift_caption() {
                    let out_dir = outward_caption_direction(&layout, node);
                    let cap_pos = center + out_dir * (radius + 9.5);
                    painter.text(
                        cap_pos,
                        egui::Align2::CENTER_CENTER,
                        caption,
                        egui::FontId::monospace(9.0),
                        egui::Color32::from_rgb(226, 238, 255),
                    );
                }
            }

            if resp.clicked() {
                if let Some(ppm) = eval.assigned_ppm {
                    jump_to_atom = Some(build_jump_target(res_idx, seq_id, topo, node, ppm, input));
                }
            }

            let tooltip = eval.build_tooltip(topo.three_letter, seq_id, effective_variant);
            resp.on_hover_ui(|ui| {
                ui.label(egui::RichText::new(&tooltip.atom).strong());
                ui.monospace(format!("Assigned ppm  : {}", tooltip.assigned_ppm));
                ui.monospace(format!("BMRB mu +- sd : {}", tooltip.bmrb_prior));
                ui.monospace(format!("Z-score       : {}", tooltip.z_score));
                ui.monospace(format!("Ambiguity_code: {}", tooltip.ambiguity_code));
                if eval.assigned_ppm.is_some() {
                    ui.label(
                        egui::RichText::new("Click to jump 2D/3D spectrum strips")
                            .italics()
                            .weak(),
                    );
                }
            });
        }

        SidechainWidgetOutput {
            selected_residue: state.selected_residue,
            jump_to_atom,
            variant_override,
        }
    }
}

#[cfg(test)]
mod tests {
    use nmr_core::nmrstar::Reference;
    use nmr_core::topology::ResidueVariant;

    use super::SequenceRibbonState;

    #[test]
    fn explicit_canonical_override_persists_across_auto_sync() {
        let seq = vec!["CYS".to_string(), "HIS".to_string()];
        let seq_ids = vec![1, 2];

        // 1. User selects Canonical BEFORE any shifts are synced (auto_inferred_variants is empty).
        let mut state = SequenceRibbonState::new();
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

        // 2. Subsequent auto-sync sees oxidized Cys CB and resolved Ne2H His shifts,
        //    but MUST NOT overwrite the user's explicit Canonical override!
        let mut r = Reference::new(seq.clone(), seq_ids.clone());
        r.shifts.insert((0, "CB".to_string()), 41.4);
        r.shifts.insert((1, "ND1".to_string()), 249.2);
        r.shifts.insert((1, "NE2".to_string()), 167.5);
        state.sync_inferred_variants_from_reference(&r);

        assert_eq!(
            state.auto_inferred_variants.get(&0),
            Some(&ResidueVariant::CysOxidized)
        );
        assert_eq!(
            state.auto_inferred_variants.get(&1),
            Some(&ResidueVariant::HisNe2H)
        );
        assert_eq!(state.variant_for(0), ResidueVariant::Canonical);
        assert_eq!(state.variant_for(1), ResidueVariant::Canonical);
        assert!(!state.variant_overrides.contains_key(&0));
        assert!(!state.variant_overrides.contains_key(&1));

        // 3. Switching from CysReduced -> Canonical after auto-sync expires also persists across future syncs.
        let empty_ref = Reference::new(seq, seq_ids);
        state.sync_inferred_variants_from_reference(&empty_ref);
        state.set_variant(0, ResidueVariant::CysReduced);
        state.set_variant(0, ResidueVariant::Canonical);
        state.sync_inferred_variants_from_reference(&r);
        assert_eq!(state.variant_for(0), ResidueVariant::Canonical);

        // 4. Clearing the explicit user override restores automatic inference cleanly.
        state.clear_user_variant_override(0);
        assert_eq!(state.variant_for(0), ResidueVariant::CysOxidized);
    }

    #[test]
    fn p1_2_his_auto_infer_requires_15n_and_honors_sample_ph() {
        let seq = vec!["HIS".to_string()];
        let seq_ids = vec![10];

        // 1. CH/CG-only His shifts (0 15N observations) MUST NOT auto-apply a static His variant!
        let mut r_ch_only = Reference::new(seq.clone(), seq_ids.clone());
        r_ch_only.shifts.insert((0, "CG".to_string()), 135.6);
        r_ch_only.shifts.insert((0, "CD2".to_string()), 118.1);
        r_ch_only.shifts.insert((0, "HE1".to_string()), 7.78);
        let mut state = SequenceRibbonState::new();
        state.sync_inferred_variants_from_reference(&r_ch_only);
        assert_eq!(state.variant_for(0), ResidueVariant::Canonical);
        assert!(state.auto_inferred_variants.is_empty());

        // 2. Discordant alkaline pH = 7.8 on uncorroborated 175.5/172.8 + HE1=7.74 abstains when sample_ph is wired!
        let mut r_disc = Reference::new(seq, seq_ids);
        r_disc.shifts.insert((0, "ND1".to_string()), 175.5);
        r_disc.shifts.insert((0, "NE2".to_string()), 172.8);
        r_disc.shifts.insert((0, "CD2".to_string()), 118.9);
        r_disc.shifts.insert((0, "HE1".to_string()), 7.74);

        state.set_sample_ph(Some(4.5));
        state.sync_inferred_variants_from_reference(&r_disc);
        assert_eq!(state.variant_for(0), ResidueVariant::HisDoublyProtonated);

        state.set_sample_ph(Some(7.8));
        state.sync_inferred_variants_from_reference(&r_disc);
        assert_eq!(
            state.variant_for(0),
            ResidueVariant::Canonical,
            "sample_ph=7.8 must abstain on discordant uncorroborated His shifts in GUI inspector"
        );
    }
}
