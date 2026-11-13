//! Bounded reversible workbench command history (`Cmd/Ctrl+Z`) and lossless NEF v1.1
//! session persistence (`save_nmrassign_workbench`).
//!
//! Provenance: Clean-room original implementation of `docs/roadmap/MASTER_ROADMAP.md` (`T09.6`):
//! - Gutmanas, A. et al., "NMR Exchange Format: a unified and open standard for
//!   representation of NMR restraint data", *Nature Structural & Molecular Biology*
//!   22, 433–434 (2015) — NEF v1.1 vendor-prefixed saveframe extensibility rules
//!   (`save_nmrassign_workbench`, `_nmrassign_workbench.*` tags, and `_nmrassign_*` loops).
//! - Original bounded `VecDeque` command-log architecture with exact state-delta undo records
//!   preserving bijective `(spin_system <-> residue_seq_idx)` assignment and lock invariants.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::nef::{
    compute_nef_shift_lists_crc32, normalize_nef_atom_name, read_nef_str, write_nef_str,
    NefDocument, NefMolecularSystem, NefShiftList, NefWorkbenchAssignmentRow,
    NefWorkbenchContourLayerRow, NefWorkbenchLinkRow, NefWorkbenchLockRow, NefWorkbenchPeakRow,
    NefWorkbenchRecipeRow, NefWorkbenchResidueRow, NefWorkbenchSaveframe, NefWorkbenchShiftRow,
};
use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::priors::one_to_three;
use crate::process::ProcessingRecipe;
use crate::qc::DatasetQcVerdict;
use crate::solver::AssignResult;

/// Default maximum undo/redo ring-buffer depth (`256` reversible steps).
pub const MAX_UNDO_STEPS: usize = 256;

// ============================================================================
// Serde shim for `Peak` (kept local so `crates/nmr-core/src/peaks.rs` stays untouched)
// ============================================================================

#[derive(Serialize, Deserialize)]
struct PeakSerdeRepr {
    exp: String,
    h: f64,
    n: f64,
    c: Option<f64>,
    sign: i32,
    height: Option<f64>,
    volume: Option<f64>,
    h2: Option<f64>,
}

impl From<&Peak> for PeakSerdeRepr {
    fn from(p: &Peak) -> Self {
        Self {
            exp: p.exp.clone(),
            h: p.h,
            n: p.n,
            c: p.c,
            sign: p.sign,
            height: p.height,
            volume: p.volume,
            h2: p.h2,
        }
    }
}

impl From<PeakSerdeRepr> for Peak {
    fn from(r: PeakSerdeRepr) -> Self {
        Self {
            exp: r.exp,
            h: r.h,
            n: r.n,
            c: r.c,
            sign: r.sign,
            height: r.height,
            volume: r.volume,
            h2: r.h2,
        }
    }
}

mod peak_serde {
    use super::{Deserializer, Peak, PeakSerdeRepr, Serialize, Serializer};
    use serde::Deserialize;

    pub fn serialize<S: Serializer>(peak: &Peak, serializer: S) -> Result<S::Ok, S::Error> {
        PeakSerdeRepr::from(peak).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Peak, D::Error> {
        PeakSerdeRepr::deserialize(deserializer).map(Peak::from)
    }
}

mod peak_vec_serde {
    use super::{Deserializer, Peak, PeakSerdeRepr, Serialize, Serializer};
    use serde::Deserialize;

    pub fn serialize<S: Serializer>(peaks: &[Peak], serializer: S) -> Result<S::Ok, S::Error> {
        let reprs: Vec<PeakSerdeRepr> = peaks.iter().map(PeakSerdeRepr::from).collect();
        reprs.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Peak>, D::Error> {
        let reprs = Vec::<PeakSerdeRepr>::deserialize(deserializer)?;
        Ok(reprs.into_iter().map(Peak::from).collect())
    }
}

// ============================================================================
// Session State Data Model & Bijection Invariants
// ============================================================================

/// Per-layer 2D / orthogonal-slice contour display parameters persisted in a workbench session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionContourParams {
    /// Stable spectrum / overlay layer identifier (e.g. `"HSQC"`, `"HNCACB_proj"`).
    pub layer_id: String,
    /// Whether this contour layer is currently visible in the overlay stack.
    pub visible: bool,
    /// Lowest contour threshold in intensity / noise-floor units.
    pub base_level: f32,
    /// Geometric spacing multiplier between successive contour levels (`> 1.0`, e.g. `1.4`).
    pub level_factor: f32,
    /// Number of geometric contour levels rendered above `base_level`.
    pub num_levels: u32,
    /// Whether negative-phase / inverted-sign (`CB` in `HNCACB`) contours are rendered.
    pub show_negative: bool,
    /// RGBA color in `[0.0, 1.0]` for positive contours.
    pub pos_rgba: [f32; 4],
    /// RGBA color in `[0.0, 1.0]` for negative contours.
    pub neg_rgba: [f32; 4],
    /// Interactive `[dx_ppm, dy_ppm]` visual alignment nudge for this overlay layer.
    pub ppm_nudge: [f32; 2],
}

impl SessionContourParams {
    /// Create default positive-cyan / negative-coral contour settings for `layer_id`.
    pub fn new(layer_id: impl Into<String>) -> Self {
        Self {
            layer_id: layer_id.into(),
            visible: true,
            base_level: 1.0,
            level_factor: 1.4,
            num_levels: 10,
            show_negative: true,
            pos_rgba: [0.18, 0.72, 0.96, 0.92],
            neg_rgba: [0.96, 0.36, 0.32, 0.88],
            ppm_nudge: [0.0, 0.0],
        }
    }

    fn bitwise_eq(&self, other: &Self) -> bool {
        self.layer_id == other.layer_id
            && self.visible == other.visible
            && self.base_level.to_bits() == other.base_level.to_bits()
            && self.level_factor.to_bits() == other.level_factor.to_bits()
            && self.num_levels == other.num_levels
            && self.show_negative == other.show_negative
            && f32_slice_bitwise_eq(&self.pos_rgba, &other.pos_rgba)
            && f32_slice_bitwise_eq(&self.neg_rgba, &other.neg_rgba)
            && f32_slice_bitwise_eq(&self.ppm_nudge, &other.ppm_nudge)
    }
}

impl Default for SessionContourParams {
    fn default() -> Self {
        Self::new("HSQC")
    }
}

/// Assignment state for a single polymer residue at 0-based `residue_seq_idx`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueAssignmentState {
    /// Assigned spin-system index (`Some(ss)` when linked to a spin system, `None` when imported shift-only).
    pub spin_system: Option<usize>,
    /// Assigned chemical shifts in ppm keyed by canonical atom name (`"H"`, `"N"`, `"CA"`, `"CB"`, `"C"`, etc.).
    pub shifts: BTreeMap<String, f64>,
    /// Posterior / marginal confidence in `[0.0, 1.0]` (`1.0` for user-curated or locked assignments).
    pub confidence: f64,
}

impl ResidueAssignmentState {
    /// Construct a residue assignment linked to `spin_system` with empty shift map and `confidence`.
    pub fn from_spin_system(spin_system: usize, confidence: f64) -> Self {
        Self {
            spin_system: Some(spin_system),
            shifts: BTreeMap::new(),
            confidence,
        }
    }

    /// Construct a residue assignment with explicit shifts and optional `spin_system`.
    pub fn with_shifts(
        spin_system: Option<usize>,
        shifts: BTreeMap<String, f64>,
        confidence: f64,
    ) -> Self {
        Self {
            spin_system,
            shifts,
            confidence,
        }
    }

    fn bitwise_eq(&self, other: &Self) -> bool {
        if self.spin_system != other.spin_system
            || self.confidence.to_bits() != other.confidence.to_bits()
            || self.shifts.len() != other.shifts.len()
        {
            return false;
        }
        self.shifts
            .iter()
            .zip(other.shifts.iter())
            .all(|((ka, va), (kb, vb))| ka == kb && va.to_bits() == vb.to_bits())
    }
}

/// Complete serializable & undoable interactive NMR workbench session state.
///
/// ### Bijective Invariants
/// 1. **`assignments: BTreeMap<usize, ResidueAssignmentState>`** (`residue_seq_idx -> state`):
///    At most one residue index may hold `spin_system == Some(s)` for any spin-system ID `s`.
/// 2. **`locked_residues: BTreeMap<usize, usize>`** (`spin_system_id -> residue_seq_idx`):
///    Strictly bijective: each `spin_system_id` maps to at most one `residue_seq_idx`, and each
///    `residue_seq_idx` appears as a target in `locked_residues` at most once.
/// 3. **Lock-Assignment Coherence**:
///    Whenever `(s -> k)` is present in `locked_residues`, `assignments.get(&k)` exists and has
///    `spin_system == Some(s)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchSessionState {
    /// Project / protein display name (e.g. `"Ubiquitin"`).
    pub protein_name: String,
    /// Polymer chain identifier for NEF / NMR-STAR interchange (default `"A"`).
    pub chain_code: String,
    /// Ordered 3-letter residue sequence (0-indexed by `residue_seq_idx`).
    pub sequence: Vec<String>,
    /// Author / PDB residue sequence numbers aligned 1-to-1 with `sequence` (defaults to `1..=N`).
    pub seq_ids: Vec<i32>,
    /// Ordered multi-experiment peak list (`HSQC`, `HNCA`, `HNCACB`, `HNCO`, `NOESY15N`, ...).
    #[serde(with = "peak_vec_serde")]
    pub peaks: Vec<Peak>,
    /// Per-residue assignment map keyed by 0-based `residue_seq_idx`.
    pub assignments: BTreeMap<usize, ResidueAssignmentState>,
    /// Bijective user-locked constraints mapping `spin_system_id -> residue_seq_idx`.
    pub locked_residues: BTreeMap<usize, usize>,
    /// User-confirmed sequential `(pred_spin_system_i_minus_1, succ_spin_system_i)` links.
    pub confirmed_links: Vec<(usize, usize)>,
    /// Ordered stack of 2D/slice contour overlay parameters.
    pub contour_layers: Vec<SessionContourParams>,
    /// Optional 2D viewport `[h_left_ppm, h_right_ppm, n_top_ppm, n_bottom_ppm]`.
    pub view_bounds_ppm: Option<[f64; 4]>,
    /// Global chemical-shift reference offsets `[dH, dN, dCA, dCB, dC]` in ppm.
    pub reference_offsets: [f64; 5],
    /// Deterministic time-domain `ProcessingRecipe` metadata keyed by spectrum identifier.
    pub processing_recipes: BTreeMap<String, ProcessingRecipe>,
    /// Optional unlabelled per-dataset quality-control verdict (`Pass`, `Warn`, `Abstain`).
    pub qc_verdict: Option<DatasetQcVerdict>,
    /// Non-fatal load warnings (e.g. when external edits to `nef_chemical_shift_list` are detected and adopted).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub load_warnings: Vec<String>,
}

impl Default for WorkbenchSessionState {
    fn default() -> Self {
        Self {
            protein_name: "untitled".to_string(),
            chain_code: "A".to_string(),
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            assignments: BTreeMap::new(),
            locked_residues: BTreeMap::new(),
            confirmed_links: Vec::new(),
            contour_layers: Vec::new(),
            view_bounds_ppm: None,
            reference_offsets: [0.0; 5],
            processing_recipes: BTreeMap::new(),
            qc_verdict: None,
            load_warnings: Vec::new(),
        }
    }
}

impl WorkbenchSessionState {
    /// Construct a new session state with canonicalized 3-letter sequence codes and `1..=N` residue IDs.
    pub fn new(protein_name: impl Into<String>, sequence: Vec<String>) -> Self {
        let norm_seq: Vec<String> = sequence
            .into_iter()
            .map(|r| {
                let t = r.trim();
                if t.len() == 1 {
                    one_to_three(t.chars().next().unwrap_or('X')).to_string()
                } else {
                    t.to_ascii_uppercase()
                }
            })
            .collect();
        let seq_ids: Vec<i32> = (1..=(norm_seq.len() as i32)).collect();
        Self {
            protein_name: protein_name.into(),
            chain_code: "A".to_string(),
            sequence: norm_seq,
            seq_ids,
            ..Self::default()
        }
    }

    /// Construct a new session state from a 1-letter FASTA sequence string (ignoring whitespace and `>` header lines).
    pub fn from_fasta_sequence(protein_name: impl Into<String>, fasta_or_raw: &str) -> Self {
        let mut seq = Vec::new();
        for line in fasta_or_raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('>') || trimmed.is_empty() {
                continue;
            }
            for ch in trimmed.chars() {
                if ch.is_ascii_alphabetic() {
                    seq.push(one_to_three(ch.to_ascii_uppercase()).to_string());
                }
            }
        }
        Self::new(protein_name, seq)
    }

    /// Construct a session state from a `Reference` shift table, optional solver `AssignResult`, and `Peak` slice.
    pub fn from_reference_and_peaks(
        protein_name: impl Into<String>,
        ref_data: &Reference,
        assign_res: Option<&AssignResult>,
        peaks: &[Peak],
    ) -> Self {
        let n_res = ref_data.sequence.len();
        let seq_ids = if ref_data.seq_ids.len() == n_res {
            ref_data.seq_ids.clone()
        } else {
            (1..=(n_res as i32)).collect()
        };

        let mut assignments: BTreeMap<usize, ResidueAssignmentState> = BTreeMap::new();
        for (&(res_idx, ref atom), &val) in &ref_data.shifts {
            if res_idx < n_res {
                let entry = assignments
                    .entry(res_idx)
                    .or_insert_with(|| ResidueAssignmentState {
                        spin_system: None,
                        shifts: BTreeMap::new(),
                        confidence: 1.0,
                    });
                entry.shifts.insert(atom.clone(), val);
            }
        }

        if let Some(res) = assign_res {
            for (ss, &pos) in res.position.iter().enumerate() {
                if pos >= 0 && (pos as usize) < n_res {
                    let k = pos as usize;
                    let conf = res.confidence.get(ss).copied().unwrap_or(0.0);
                    let entry = assignments
                        .entry(k)
                        .or_insert_with(|| ResidueAssignmentState {
                            spin_system: Some(ss),
                            shifts: BTreeMap::new(),
                            confidence: conf,
                        });
                    entry.spin_system = Some(ss);
                    entry.confidence = conf;
                }
            }
        }

        Self {
            protein_name: protein_name.into(),
            chain_code: "A".to_string(),
            sequence: ref_data.sequence.clone(),
            seq_ids,
            peaks: peaks.to_vec(),
            assignments,
            ..Self::default()
        }
    }

    /// Export sequence and assigned chemical shifts as `(Reference, residue_confidence_map)`.
    pub fn to_reference_and_confidences(&self) -> (Reference, BTreeMap<usize, f64>) {
        let seq_ids = if self.seq_ids.len() == self.sequence.len() {
            self.seq_ids.clone()
        } else {
            (1..=(self.sequence.len() as i32)).collect()
        };
        let mut out_ref = Reference::new(self.sequence.clone(), seq_ids);
        let mut conf = BTreeMap::new();
        for (&res_idx, astate) in &self.assignments {
            conf.insert(res_idx, astate.confidence);
            for (atom, &val) in &astate.shifts {
                out_ref.shifts.insert((res_idx, atom.clone()), val);
            }
        }
        (out_ref, conf)
    }

    /// Return the locked `residue_seq_idx` for `spin_system`, if locked.
    #[inline]
    pub fn locked_residue_for_spin_system(&self, spin_system: usize) -> Option<usize> {
        self.locked_residues.get(&spin_system).copied()
    }

    /// Return the locked `spin_system_id` for `residue_seq_idx`, if locked.
    pub fn locked_spin_system_for_residue(&self, residue_seq_idx: usize) -> Option<usize> {
        self.locked_residues
            .iter()
            .find_map(|(&ss, &res)| (res == residue_seq_idx).then_some(ss))
    }

    /// Return `true` when `residue_seq_idx` holds a user lock.
    #[inline]
    pub fn is_residue_locked(&self, residue_seq_idx: usize) -> bool {
        self.locked_spin_system_for_residue(residue_seq_idx)
            .is_some()
    }

    /// Return `true` when `spin_system` holds a user lock.
    #[inline]
    pub fn is_spin_system_locked(&self, spin_system: usize) -> bool {
        self.locked_residues.contains_key(&spin_system)
    }

    /// Return the currently assigned `residue_seq_idx` for `spin_system`, if any.
    pub fn assigned_residue_for_spin_system(&self, spin_system: usize) -> Option<usize> {
        self.assignments
            .iter()
            .find_map(|(&res_idx, st)| (st.spin_system == Some(spin_system)).then_some(res_idx))
    }

    /// Export `(spin_system_id, residue_seq_idx)` pairs sorted deterministically for solver constraints.
    pub fn locked_assignments_vec(&self) -> Vec<(usize, usize)> {
        self.locked_residues
            .iter()
            .map(|(&ss, &res)| (ss, res))
            .collect()
    }

    /// Verify all bijective `(spin_system <-> residue_seq_idx)` assignment and lock invariants.
    pub fn validate_invariants(&self) -> Result<(), String> {
        if !self.seq_ids.is_empty() && self.seq_ids.len() != self.sequence.len() {
            return Err(format!(
                "seq_ids length ({}) does not match sequence length ({})",
                self.seq_ids.len(),
                self.sequence.len()
            ));
        }

        let mut seen_assigned_ss: BTreeMap<usize, usize> = BTreeMap::new();
        for (&res_idx, st) in &self.assignments {
            if let Some(ss) = st.spin_system {
                if let Some(prev_res) = seen_assigned_ss.insert(ss, res_idx) {
                    return Err(format!(
                        "non-injective assignment: spin_system {ss} assigned to both residue {prev_res} and {res_idx}"
                    ));
                }
            }
        }

        let mut seen_locked_res: BTreeSet<usize> = BTreeSet::new();
        for (&ss, &res_idx) in &self.locked_residues {
            if !seen_locked_res.insert(res_idx) {
                return Err(format!(
                    "non-bijective lock: multiple spin systems locked to residue {res_idx}"
                ));
            }
            match self.assignments.get(&res_idx).and_then(|a| a.spin_system) {
                Some(assigned_ss) if assigned_ss == ss => {}
                other => {
                    return Err(format!(
                        "incoherent lock ({ss} -> {res_idx}): assignments[{res_idx}].spin_system is {other:?}"
                    ));
                }
            }
        }

        let mut seen_layers: BTreeSet<&str> = BTreeSet::new();
        for cl in &self.contour_layers {
            if !seen_layers.insert(cl.layer_id.as_str()) {
                return Err(format!(
                    "duplicate contour layer_id '{}' in session contour_layers",
                    cl.layer_id
                ));
            }
        }

        Ok(())
    }

    /// Strict IEEE-754 bit-exact comparison across every `f64`, `f32`, map, and vector in the session.
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        if self.protein_name != other.protein_name
            || self.chain_code != other.chain_code
            || self.sequence != other.sequence
            || self.seq_ids != other.seq_ids
            || self.peaks.len() != other.peaks.len()
            || self.assignments.len() != other.assignments.len()
            || self.locked_residues != other.locked_residues
            || self.confirmed_links != other.confirmed_links
            || self.contour_layers.len() != other.contour_layers.len()
            || !opt_f64_arr4_bitwise_eq(self.view_bounds_ppm, other.view_bounds_ppm)
            || !f64_slice_bitwise_eq(&self.reference_offsets, &other.reference_offsets)
            || self.processing_recipes.len() != other.processing_recipes.len()
            || self.qc_verdict != other.qc_verdict
        {
            return false;
        }

        if !self
            .peaks
            .iter()
            .zip(other.peaks.iter())
            .all(|(a, b)| peak_bitwise_eq(a, b))
        {
            return false;
        }

        if !self
            .assignments
            .iter()
            .zip(other.assignments.iter())
            .all(|((ka, va), (kb, vb))| ka == kb && va.bitwise_eq(vb))
        {
            return false;
        }

        if !self
            .contour_layers
            .iter()
            .zip(other.contour_layers.iter())
            .all(|(a, b)| a.bitwise_eq(b))
        {
            return false;
        }

        self.processing_recipes
            .iter()
            .zip(other.processing_recipes.iter())
            .all(|((ka, va), (kb, vb))| ka == kb && recipe_bitwise_eq(va, vb))
    }
}

fn opt_f64_bitwise_eq(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn f64_slice_bitwise_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn f32_slice_bitwise_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn opt_f64_arr4_bitwise_eq(a: Option<[f64; 4]>, b: Option<[f64; 4]>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => f64_slice_bitwise_eq(&x, &y),
        (None, None) => true,
        _ => false,
    }
}

fn peak_bitwise_eq(a: &Peak, b: &Peak) -> bool {
    a.exp == b.exp
        && a.h.to_bits() == b.h.to_bits()
        && a.n.to_bits() == b.n.to_bits()
        && opt_f64_bitwise_eq(a.c, b.c)
        && a.sign == b.sign
        && opt_f64_bitwise_eq(a.height, b.height)
        && opt_f64_bitwise_eq(a.volume, b.volume)
        && opt_f64_bitwise_eq(a.h2, b.h2)
}

fn recipe_bitwise_eq(a: &ProcessingRecipe, b: &ProcessingRecipe) -> bool {
    if a != b {
        return false;
    }
    // Also verify exact JSON serialization match to catch any signed-zero float edge cases
    serde_json::to_string(a).ok() == serde_json::to_string(b).ok()
}

// ============================================================================
// Reversible Workbench Commands & Bounded `256`-Step Ring Buffer
// ============================================================================

/// Reversible interactive curation / display command in the NMR workbench.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkbenchCommand {
    /// Append a newly picked or user-placed peak to `state.peaks`.
    AddPeak(#[serde(with = "peak_serde")] Peak),
    /// Delete one or more peaks by index (undo restores exact original indices and order).
    DeletePeaks(Vec<usize>),
    /// Move an existing peak to updated `(H, N, optional C)` coordinates in ppm.
    MovePeak {
        index: usize,
        new_h: f64,
        new_n: f64,
        new_c: Option<f64>,
    },
    /// Snap an existing peak to locally fitted sub-voxel coordinates and optional peak amplitude.
    RecenterPeak {
        index: usize,
        fitted_h: f64,
        fitted_n: f64,
        fitted_c: Option<f64>,
        new_height: Option<f64>,
    },
    /// Assign `spin_system` to 0-based `residue_seq_idx`, cleanly displacing any conflicting assignment/lock.
    AssignSpinSystem {
        spin_system: usize,
        residue_seq_idx: usize,
    },
    /// Lock (`locked = true`) or unlock (`locked = false`) `(spin_system -> residue_seq_idx)`.
    LockResidue {
        spin_system: usize,
        residue_seq_idx: usize,
        locked: bool,
    },
    /// Clear the assignment (and any associated lock) at 0-based `residue_seq_idx`.
    UnassignResidue { residue_seq_idx: usize },
    /// Update or insert contour display parameters for `layer_id`.
    SetContourParams {
        layer_id: String,
        params: SessionContourParams,
    },
    /// Set global chemical-shift reference offsets `[dH, dN, dCA, dCB, dC]` in ppm.
    SetReferenceOffset { offsets: [f64; 5] },
}

impl WorkbenchCommand {
    /// Human-readable summary suitable for GUI menu items and status-bar notifications.
    pub fn description(&self) -> String {
        match self {
            Self::AddPeak(p) => match p.c {
                Some(c) => format!("Add {} peak ({:.3}, {:.2}, {:.2} ppm)", p.exp, p.h, p.n, c),
                None => format!("Add {} peak ({:.3}, {:.2} ppm)", p.exp, p.h, p.n),
            },
            Self::DeletePeaks(idxs) => format!("Delete {} peak(s)", idxs.len()),
            Self::MovePeak {
                index,
                new_h,
                new_n,
                ..
            } => format!("Move peak #{index} to ({new_h:.3}, {new_n:.2} ppm)"),
            Self::RecenterPeak {
                index,
                fitted_h,
                fitted_n,
                ..
            } => format!("Recenter peak #{index} to ({fitted_h:.3}, {fitted_n:.2} ppm)"),
            Self::AssignSpinSystem {
                spin_system,
                residue_seq_idx,
            } => format!(
                "Assign spin system #{spin_system} to residue #{}",
                residue_seq_idx + 1
            ),
            Self::LockResidue {
                spin_system,
                residue_seq_idx,
                locked,
            } => {
                if *locked {
                    format!(
                        "Lock spin system #{spin_system} to residue #{}",
                        residue_seq_idx + 1
                    )
                } else {
                    format!(
                        "Unlock spin system #{spin_system} at residue #{}",
                        residue_seq_idx + 1
                    )
                }
            }
            Self::UnassignResidue { residue_seq_idx } => {
                format!("Unassign residue #{}", residue_seq_idx + 1)
            }
            Self::SetContourParams { layer_id, .. } => {
                format!("Update contour parameters ({layer_id})")
            }
            Self::SetReferenceOffset { offsets } => format!(
                "Set reference offsets (dH={:+.3}, dN={:+.2}, dCA={:+.2}, dCB={:+.2}, dC={:+.2})",
                offsets[0], offsets[1], offsets[2], offsets[3], offsets[4]
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MovedPeakSnapshot {
    index: usize,
    old_h: f64,
    old_n: f64,
    old_c: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct RecenteredPeakSnapshot {
    index: usize,
    old_h: f64,
    old_n: f64,
    old_c: Option<f64>,
    old_height: Option<f64>,
}

/// Exact state-delta snapshot allowing bit-exact `O(1)` / `O(k)` inverse replay.
#[derive(Clone, Debug)]
enum UndoRecord {
    AddedPeak {
        added_index: usize,
    },
    DeletedPeaks {
        /// Stored in ascending original-index order so forward re-insertion restores exact positions.
        deleted_ascending: Vec<(usize, Peak)>,
    },
    MovedPeak {
        moved: Option<MovedPeakSnapshot>,
    },
    RecenteredPeak {
        recentered: Option<RecenteredPeakSnapshot>,
    },
    AssignmentMutation {
        target_res_idx: usize,
        prev_at_target: Option<ResidueAssignmentState>,
        displaced_other_residue: Option<(usize, ResidueAssignmentState)>,
        inserted_lock: Option<usize>,
        removed_locks: Vec<(usize, usize)>,
    },
    UnassignedResidue {
        residue_seq_idx: usize,
        prev_assignment: Option<ResidueAssignmentState>,
        removed_locks: Vec<(usize, usize)>,
    },
    ContourMutation {
        prev_layer: Option<(usize, SessionContourParams)>,
        inserted_at: Option<usize>,
    },
    ReferenceOffset {
        prev_offsets: [f64; 5],
    },
}

#[derive(Clone, Debug)]
struct HistoryEntry {
    cmd: WorkbenchCommand,
    undo: UndoRecord,
    description: String,
}

fn apply_command(state: &mut WorkbenchSessionState, cmd: &WorkbenchCommand) -> UndoRecord {
    match cmd {
        WorkbenchCommand::AddPeak(peak) => {
            let added_index = state.peaks.len();
            state.peaks.push(peak.clone());
            UndoRecord::AddedPeak { added_index }
        }
        WorkbenchCommand::DeletePeaks(indices) => {
            let mut valid: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| i < state.peaks.len())
                .collect();
            valid.sort_unstable();
            valid.dedup();

            let mut deleted_ascending = Vec::with_capacity(valid.len());
            for &idx in &valid {
                deleted_ascending.push((idx, state.peaks[idx].clone()));
            }
            for &idx in valid.iter().rev() {
                state.peaks.remove(idx);
            }
            UndoRecord::DeletedPeaks { deleted_ascending }
        }
        WorkbenchCommand::MovePeak {
            index,
            new_h,
            new_n,
            new_c,
        } => {
            if let Some(p) = state.peaks.get_mut(*index) {
                let old = MovedPeakSnapshot {
                    index: *index,
                    old_h: p.h,
                    old_n: p.n,
                    old_c: p.c,
                };
                p.h = *new_h;
                p.n = *new_n;
                p.c = *new_c;
                UndoRecord::MovedPeak { moved: Some(old) }
            } else {
                UndoRecord::MovedPeak { moved: None }
            }
        }
        WorkbenchCommand::RecenterPeak {
            index,
            fitted_h,
            fitted_n,
            fitted_c,
            new_height,
        } => {
            if let Some(p) = state.peaks.get_mut(*index) {
                let old = RecenteredPeakSnapshot {
                    index: *index,
                    old_h: p.h,
                    old_n: p.n,
                    old_c: p.c,
                    old_height: p.height,
                };
                p.h = *fitted_h;
                p.n = *fitted_n;
                p.c = *fitted_c;
                p.height = *new_height;
                UndoRecord::RecenteredPeak {
                    recentered: Some(old),
                }
            } else {
                UndoRecord::RecenteredPeak { recentered: None }
            }
        }
        WorkbenchCommand::AssignSpinSystem {
            spin_system,
            residue_seq_idx,
        } => {
            let s = *spin_system;
            let k = *residue_seq_idx;
            let prev_at_target = state.assignments.get(&k).cloned();

            // Check whether `s` was previously assigned at a different residue `other_k != k`
            let existing_res_for_s = state
                .assignments
                .iter()
                .find_map(|(&rk, st)| (rk != k && st.spin_system == Some(s)).then_some(rk));
            let displaced_other_residue = existing_res_for_s
                .and_then(|rk| state.assignments.remove(&rk).map(|old| (rk, old)));

            // Remove any stale locks conflicting with moving `s` to `k` or displacing another spin system at `k`:
            // - if `s` was locked to `other_k != k`, remove that lock;
            // - if some `other_s != s` was locked to `k`, remove that lock.
            let stale_lock_keys: Vec<usize> = state
                .locked_residues
                .iter()
                .filter_map(|(&lk_s, &lk_k)| {
                    if (lk_s == s && lk_k != k) || (lk_k == k && lk_s != s) {
                        Some(lk_s)
                    } else {
                        None
                    }
                })
                .collect();
            let mut removed_locks = Vec::with_capacity(stale_lock_keys.len());
            for lk_s in stale_lock_keys {
                if let Some(lk_k) = state.locked_residues.remove(&lk_s) {
                    removed_locks.push((lk_s, lk_k));
                }
            }

            let carried_shifts = match &prev_at_target {
                Some(prev) if prev.spin_system == Some(s) => prev.shifts.clone(),
                _ => match &displaced_other_residue {
                    Some((_, other_old)) => other_old.shifts.clone(),
                    None => BTreeMap::new(),
                },
            };

            state.assignments.insert(
                k,
                ResidueAssignmentState {
                    spin_system: Some(s),
                    shifts: carried_shifts,
                    confidence: 1.0,
                },
            );

            UndoRecord::AssignmentMutation {
                target_res_idx: k,
                prev_at_target,
                displaced_other_residue,
                inserted_lock: None,
                removed_locks,
            }
        }
        WorkbenchCommand::LockResidue {
            spin_system,
            residue_seq_idx,
            locked,
        } => {
            let s = *spin_system;
            let k = *residue_seq_idx;
            let prev_at_target = state.assignments.get(&k).cloned();

            if *locked {
                let existing_res_for_s = state
                    .assignments
                    .iter()
                    .find_map(|(&rk, st)| (rk != k && st.spin_system == Some(s)).then_some(rk));
                let displaced_other_residue = existing_res_for_s
                    .and_then(|rk| state.assignments.remove(&rk).map(|old| (rk, old)));

                // Collect & remove any existing locks touching `s` or `k` so bijectivity is strictly preserved
                let lock_keys: Vec<usize> = state
                    .locked_residues
                    .iter()
                    .filter_map(|(&lk_s, &lk_k)| (lk_s == s || lk_k == k).then_some(lk_s))
                    .collect();
                let mut removed_locks = Vec::with_capacity(lock_keys.len());
                for lk_s in lock_keys {
                    if let Some(lk_k) = state.locked_residues.remove(&lk_s) {
                        removed_locks.push((lk_s, lk_k));
                    }
                }

                state.locked_residues.insert(s, k);

                let carried_shifts = match &prev_at_target {
                    Some(prev) if prev.spin_system == Some(s) => prev.shifts.clone(),
                    _ => match &displaced_other_residue {
                        Some((_, other_old)) => other_old.shifts.clone(),
                        None => BTreeMap::new(),
                    },
                };
                state.assignments.insert(
                    k,
                    ResidueAssignmentState {
                        spin_system: Some(s),
                        shifts: carried_shifts,
                        confidence: 1.0,
                    },
                );

                UndoRecord::AssignmentMutation {
                    target_res_idx: k,
                    prev_at_target,
                    displaced_other_residue,
                    inserted_lock: Some(s),
                    removed_locks,
                }
            } else {
                let mut removed_locks = Vec::new();
                if state.locked_residues.get(&s) == Some(&k) {
                    state.locked_residues.remove(&s);
                    removed_locks.push((s, k));
                }
                UndoRecord::AssignmentMutation {
                    target_res_idx: k,
                    prev_at_target,
                    displaced_other_residue: None,
                    inserted_lock: None,
                    removed_locks,
                }
            }
        }
        WorkbenchCommand::UnassignResidue { residue_seq_idx } => {
            let k = *residue_seq_idx;
            let prev_assignment = state.assignments.remove(&k);
            let lock_keys: Vec<usize> = state
                .locked_residues
                .iter()
                .filter_map(|(&lk_s, &lk_k)| (lk_k == k).then_some(lk_s))
                .collect();
            let mut removed_locks = Vec::with_capacity(lock_keys.len());
            for lk_s in lock_keys {
                if let Some(lk_k) = state.locked_residues.remove(&lk_s) {
                    removed_locks.push((lk_s, lk_k));
                }
            }
            UndoRecord::UnassignedResidue {
                residue_seq_idx: k,
                prev_assignment,
                removed_locks,
            }
        }
        WorkbenchCommand::SetContourParams { layer_id, params } => {
            let mut normalized = params.clone();
            normalized.layer_id = layer_id.clone();
            if let Some(pos) = state
                .contour_layers
                .iter()
                .position(|c| c.layer_id == *layer_id)
            {
                let old = std::mem::replace(&mut state.contour_layers[pos], normalized);
                UndoRecord::ContourMutation {
                    prev_layer: Some((pos, old)),
                    inserted_at: None,
                }
            } else {
                let idx = state.contour_layers.len();
                state.contour_layers.push(normalized);
                UndoRecord::ContourMutation {
                    prev_layer: None,
                    inserted_at: Some(idx),
                }
            }
        }
        WorkbenchCommand::SetReferenceOffset { offsets } => {
            let prev_offsets = state.reference_offsets;
            state.reference_offsets = *offsets;
            UndoRecord::ReferenceOffset { prev_offsets }
        }
    }
}

fn apply_undo(state: &mut WorkbenchSessionState, record: &UndoRecord) {
    match record {
        UndoRecord::AddedPeak { added_index } => {
            if *added_index < state.peaks.len() {
                state.peaks.remove(*added_index);
            }
        }
        UndoRecord::DeletedPeaks { deleted_ascending } => {
            for (idx, peak) in deleted_ascending {
                let insert_pos = (*idx).min(state.peaks.len());
                state.peaks.insert(insert_pos, peak.clone());
            }
        }
        UndoRecord::MovedPeak { moved } => {
            if let Some(snap) = *moved {
                if let Some(p) = state.peaks.get_mut(snap.index) {
                    p.h = snap.old_h;
                    p.n = snap.old_n;
                    p.c = snap.old_c;
                }
            }
        }
        UndoRecord::RecenteredPeak { recentered } => {
            if let Some(snap) = *recentered {
                if let Some(p) = state.peaks.get_mut(snap.index) {
                    p.h = snap.old_h;
                    p.n = snap.old_n;
                    p.c = snap.old_c;
                    p.height = snap.old_height;
                }
            }
        }
        UndoRecord::AssignmentMutation {
            target_res_idx,
            prev_at_target,
            displaced_other_residue,
            inserted_lock,
            removed_locks,
        } => {
            if let Some(s) = inserted_lock {
                state.locked_residues.remove(s);
            }
            for &(lk_s, lk_k) in removed_locks {
                state.locked_residues.insert(lk_s, lk_k);
            }
            match prev_at_target {
                Some(old_target) => {
                    state
                        .assignments
                        .insert(*target_res_idx, old_target.clone());
                }
                None => {
                    state.assignments.remove(target_res_idx);
                }
            }
            if let Some((other_k, other_old)) = displaced_other_residue {
                state.assignments.insert(*other_k, other_old.clone());
            }
        }
        UndoRecord::UnassignedResidue {
            residue_seq_idx,
            prev_assignment,
            removed_locks,
        } => {
            if let Some(old) = prev_assignment {
                state.assignments.insert(*residue_seq_idx, old.clone());
            }
            for &(lk_s, lk_k) in removed_locks {
                state.locked_residues.insert(lk_s, lk_k);
            }
        }
        UndoRecord::ContourMutation {
            prev_layer,
            inserted_at,
        } => {
            if let Some((pos, old)) = prev_layer {
                if *pos < state.contour_layers.len() {
                    state.contour_layers[*pos] = old.clone();
                }
            } else if let Some(idx) = inserted_at {
                if *idx < state.contour_layers.len() {
                    state.contour_layers.remove(*idx);
                }
            }
        }
        UndoRecord::ReferenceOffset { prev_offsets } => {
            state.reference_offsets = *prev_offsets;
        }
    }
}

/// Bounded reversible command history backed by a fixed-capacity `VecDeque` ring buffer (`256` steps).
#[derive(Clone, Debug)]
pub struct CommandHistory {
    max_steps: usize,
    undo_stack: VecDeque<HistoryEntry>,
    redo_stack: VecDeque<WorkbenchCommand>,
}

impl Default for CommandHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandHistory {
    /// Construct a new `CommandHistory` with the canonical `256`-step ring-buffer bound.
    pub fn new() -> Self {
        Self::with_capacity(MAX_UNDO_STEPS)
    }

    /// Construct a `CommandHistory` with an explicit ring-buffer step bound (`>= 1`).
    pub fn with_capacity(max_steps: usize) -> Self {
        let cap = max_steps.max(1);
        Self {
            max_steps: cap,
            undo_stack: VecDeque::with_capacity(cap),
            redo_stack: VecDeque::with_capacity(cap),
        }
    }

    /// Maximum number of reversible commands retained in the ring buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.max_steps
    }

    /// Return `true` if at least one command can be undone.
    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Return `true` if at least one undone command can be redone.
    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Current number of undoable steps in the ring buffer (`<= capacity()`).
    #[inline]
    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    /// Current number of redoable steps on the redo tail (`<= capacity()`).
    #[inline]
    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    /// Description of the next command that would be undone by `Cmd/Ctrl+Z`.
    pub fn next_undo_description(&self) -> Option<&str> {
        self.undo_stack.back().map(|e| e.description.as_str())
    }

    /// Description of the next command that would be redone by `Cmd/Ctrl+Shift+Z`.
    pub fn next_redo_description(&self) -> Option<&str> {
        self.redo_stack.back().map(|_| "Redo")
    }

    /// Clear both undo and redo stacks (e.g. after loading a fresh session from disk).
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Execute `cmd` against `state`, clearing any pending redo tail and evicting the oldest undo
    /// record if the `256`-step ring buffer is full.
    pub fn execute(&mut self, state: &mut WorkbenchSessionState, cmd: WorkbenchCommand) -> String {
        self.redo_stack.clear();
        let description = cmd.description();
        let undo = apply_command(state, &cmd);
        if self.undo_stack.len() == self.max_steps {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(HistoryEntry {
            cmd,
            undo,
            description: description.clone(),
        });
        description
    }

    /// Undo the most recent command on `state` (`Cmd/Ctrl+Z`), returning its description if available.
    pub fn undo(&mut self, state: &mut WorkbenchSessionState) -> Option<String> {
        let entry = self.undo_stack.pop_back()?;
        apply_undo(state, &entry.undo);
        if self.redo_stack.len() == self.max_steps {
            self.redo_stack.pop_front();
        }
        self.redo_stack.push_back(entry.cmd);
        Some(entry.description)
    }

    /// Redo the most recently undone command on `state` (`Cmd/Ctrl+Shift+Z`), returning its description.
    pub fn redo(&mut self, state: &mut WorkbenchSessionState) -> Option<String> {
        let cmd = self.redo_stack.pop_back()?;
        let description = cmd.description();
        let undo = apply_command(state, &cmd);
        if self.undo_stack.len() == self.max_steps {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(HistoryEntry {
            cmd,
            undo,
            description: description.clone(),
        });
        Some(description)
    }
}

// ============================================================================
// Lossless NEF v1.1 Session Persistence (`save_nmrassign_workbench`)
// ============================================================================

fn parse_qc_verdict_tag(s: &str) -> Result<DatasetQcVerdict, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "PASS" => Ok(DatasetQcVerdict::Pass),
        "WARN" => Ok(DatasetQcVerdict::Warn),
        "ABSTAIN" => Ok(DatasetQcVerdict::Abstain),
        other => Err(format!("unknown _nmrassign_workbench.qc_verdict '{other}'")),
    }
}

/// Serialize a [`WorkbenchSessionState`] into a standards-compliant NEF v1.1 document string
/// containing canonical `nef_nmr_meta_data`, `nef_molecular_system`, `nef_chemical_shift_list`,
/// `nef_nmr_spectrum` peak lists, plus the lossless `save_nmrassign_workbench` vendor saveframe.
pub fn write_session_nef_str(state: &WorkbenchSessionState) -> String {
    let chain = if state.chain_code.trim().is_empty() {
        "A"
    } else {
        state.chain_code.as_str()
    };
    let (ref_data, conf_map) = state.to_reference_and_confidences();

    let project_id = if state.protein_name.trim().is_empty() {
        "workbench_session"
    } else {
        state.protein_name.as_str()
    };
    let mut doc = NefDocument::new(project_id);
    if !ref_data.sequence.is_empty() {
        doc.molecular_system = NefMolecularSystem::from_reference(&ref_data, chain);
    }
    if !ref_data.shifts.is_empty() {
        doc.shift_lists.push(NefShiftList::from_reference(
            &ref_data,
            Some(&conf_map),
            chain,
        ));
    }
    doc.peak_lists = crate::nef::peaks_to_nef_lists(&state.peaks);

    let seq_ids = if state.seq_ids.len() == state.sequence.len() {
        state.seq_ids.clone()
    } else {
        (1..=(state.sequence.len() as i32)).collect()
    };
    let residues: Vec<NefWorkbenchResidueRow> = state
        .sequence
        .iter()
        .zip(seq_ids.iter())
        .map(|(res, &sid)| NefWorkbenchResidueRow {
            seq_id: sid,
            residue_name: res.clone(),
        })
        .collect();

    let peaks: Vec<NefWorkbenchPeakRow> = state
        .peaks
        .iter()
        .map(|p| NefWorkbenchPeakRow {
            experiment: p.exp.clone(),
            h: p.h,
            n: p.n,
            c: p.c,
            sign: p.sign,
            height: p.height,
            volume: p.volume,
            h2: p.h2,
        })
        .collect();

    let mut assignments = Vec::with_capacity(state.assignments.len());
    let mut assignment_shifts = Vec::new();
    for (&res_idx, astate) in &state.assignments {
        assignments.push(NefWorkbenchAssignmentRow {
            residue_seq_idx: res_idx,
            spin_system_id: astate.spin_system,
            confidence: astate.confidence,
        });
        for (atom, &val) in &astate.shifts {
            assignment_shifts.push(NefWorkbenchShiftRow {
                residue_seq_idx: res_idx,
                atom_name: atom.clone(),
                value: val,
            });
        }
    }

    let locks: Vec<NefWorkbenchLockRow> = state
        .locked_residues
        .iter()
        .map(|(&ss, &res_idx)| NefWorkbenchLockRow {
            spin_system_id: ss,
            residue_seq_idx: res_idx,
        })
        .collect();

    let confirmed_links: Vec<NefWorkbenchLinkRow> = state
        .confirmed_links
        .iter()
        .map(|&(pred, succ)| NefWorkbenchLinkRow {
            pred_spin_system: pred,
            succ_spin_system: succ,
        })
        .collect();

    let contour_layers: Vec<NefWorkbenchContourLayerRow> = state
        .contour_layers
        .iter()
        .map(|cl| NefWorkbenchContourLayerRow {
            layer_id: cl.layer_id.clone(),
            visible: cl.visible,
            base_level: cl.base_level,
            level_factor: cl.level_factor,
            num_levels: cl.num_levels,
            show_negative: cl.show_negative,
            pos_rgba: cl.pos_rgba,
            neg_rgba: cl.neg_rgba,
            ppm_nudge: cl.ppm_nudge,
        })
        .collect();

    let mut processing_recipes = Vec::with_capacity(state.processing_recipes.len());
    for (spec_id, recipe) in &state.processing_recipes {
        let recipe_json = serde_json::to_string(recipe).unwrap_or_else(|_| "{}".to_string());
        processing_recipes.push(NefWorkbenchRecipeRow {
            spectrum_id: spec_id.clone(),
            recipe_json,
        });
    }

    let shift_list_crc32 = Some(compute_nef_shift_lists_crc32(&doc.shift_lists));

    doc.workbench_session = Some(NefWorkbenchSaveframe {
        schema_version: "1.1".to_string(),
        protein_name: state.protein_name.clone(),
        chain_code: state.chain_code.clone(),
        reference_offsets: state.reference_offsets,
        qc_verdict: state.qc_verdict.map(|v| v.as_str().to_string()),
        view_bounds_ppm: state.view_bounds_ppm,
        shift_list_crc32,
        residues,
        peaks,
        assignments,
        assignment_shifts,
        locks,
        confirmed_links,
        contour_layers,
        processing_recipes,
    });

    write_nef_str(&doc)
}

/// Parse a [`WorkbenchSessionState`] from a NEF v1.1 string.
///
/// - When the vendor-prefixed `save_nmrassign_workbench` saveframe is present, reconstructs the
///   exact `WorkbenchSessionState` losslessly (bit-exact IEEE-754 floats, peak order, locks,
///   confirmed links, contour layers, reference offsets, QC verdict, and `ProcessingRecipe`s).
/// - When importing a plain third-party NEF v1.1 file lacking `save_nmrassign_workbench`,
///   gracefully populates sequence, shifts, and peaks from `nef_molecular_system`,
///   `nef_chemical_shift_list`, and `nef_nmr_spectrum`.
pub fn read_session_nef_str(text: &str) -> Result<WorkbenchSessionState, String> {
    let mut doc = read_nef_str(text)?;

    if let Some(wb) = doc.workbench_session.take() {
        let (sequence, seq_ids) =
            if !wb.residues.is_empty() || doc.molecular_system.residues.is_empty() {
                let seq = wb.residues.iter().map(|r| r.residue_name.clone()).collect();
                let sids = wb.residues.iter().map(|r| r.seq_id).collect();
                (seq, sids)
            } else {
                let ref_data = doc.to_reference()?;
                (ref_data.sequence, ref_data.seq_ids)
            };

        let peaks: Vec<Peak> = wb
            .peaks
            .into_iter()
            .map(|r| Peak {
                exp: r.experiment,
                h: r.h,
                n: r.n,
                c: r.c,
                sign: r.sign,
                height: r.height,
                volume: r.volume,
                h2: r.h2,
            })
            .collect();

        let mut assignments: BTreeMap<usize, ResidueAssignmentState> = BTreeMap::new();
        for a in wb.assignments {
            assignments.insert(
                a.residue_seq_idx,
                ResidueAssignmentState {
                    spin_system: a.spin_system_id,
                    shifts: BTreeMap::new(),
                    confidence: a.confidence,
                },
            );
        }
        for sh in wb.assignment_shifts {
            let entry =
                assignments
                    .entry(sh.residue_seq_idx)
                    .or_insert_with(|| ResidueAssignmentState {
                        spin_system: None,
                        shifts: BTreeMap::new(),
                        confidence: 1.0,
                    });
            entry.shifts.insert(sh.atom_name, sh.value);
        }

        let mut locked_residues: BTreeMap<usize, usize> = BTreeMap::new();
        for lk in wb.locks {
            if locked_residues
                .insert(lk.spin_system_id, lk.residue_seq_idx)
                .is_some()
            {
                return Err(format!(
                    "duplicate spin_system_id {} in _nmrassign_lock",
                    lk.spin_system_id
                ));
            }
        }

        let confirmed_links: Vec<(usize, usize)> = wb
            .confirmed_links
            .into_iter()
            .map(|l| (l.pred_spin_system, l.succ_spin_system))
            .collect();

        let contour_layers: Vec<SessionContourParams> = wb
            .contour_layers
            .into_iter()
            .map(|cl| SessionContourParams {
                layer_id: cl.layer_id,
                visible: cl.visible,
                base_level: cl.base_level,
                level_factor: cl.level_factor,
                num_levels: cl.num_levels,
                show_negative: cl.show_negative,
                pos_rgba: cl.pos_rgba,
                neg_rgba: cl.neg_rgba,
                ppm_nudge: cl.ppm_nudge,
            })
            .collect();

        let mut processing_recipes: BTreeMap<String, ProcessingRecipe> = BTreeMap::new();
        for pr in wb.processing_recipes {
            let recipe: ProcessingRecipe = serde_json::from_str(&pr.recipe_json).map_err(|e| {
                format!(
                    "failed to parse _nmrassign_processing_recipe JSON for '{}': {e}",
                    pr.spectrum_id
                )
            })?;
            processing_recipes.insert(pr.spectrum_id, recipe);
        }

        let qc_verdict = match wb.qc_verdict {
            Some(ref s) => Some(parse_qc_verdict_tag(s)?),
            None => None,
        };

        let mut load_warnings = Vec::new();
        if let Some(ref expected_crc) = wb.shift_list_crc32 {
            let actual_crc = compute_nef_shift_lists_crc32(&doc.shift_lists);
            if !actual_crc.eq_ignore_ascii_case(expected_crc) {
                reconcile_external_nef_shift_edits(
                    &mut assignments,
                    &sequence,
                    &seq_ids,
                    &wb.chain_code,
                    &doc.shift_lists,
                );
                load_warnings.push(format!(
                    "Detected external edits to nef_chemical_shift_list (CRC32 expected {expected_crc}, found {actual_crc}); adopted updated chemical shifts from standard NEF shift list."
                ));
            }
        }

        let state = WorkbenchSessionState {
            protein_name: wb.protein_name,
            chain_code: wb.chain_code,
            sequence,
            seq_ids,
            peaks,
            assignments,
            locked_residues,
            confirmed_links,
            contour_layers,
            view_bounds_ppm: wb.view_bounds_ppm,
            reference_offsets: wb.reference_offsets,
            processing_recipes,
            qc_verdict,
            load_warnings,
        };
        state.validate_invariants()?;
        Ok(state)
    } else {
        // Plain standard NEF v1.1 fallback import
        let peaks = doc.to_peaks();
        let chain_code = doc
            .molecular_system
            .residues
            .first()
            .map(|r| r.chain_code.clone())
            .unwrap_or_else(|| "A".to_string());
        if doc.molecular_system.residues.is_empty() {
            return Ok(WorkbenchSessionState {
                protein_name: doc.project_name,
                chain_code,
                peaks,
                ..WorkbenchSessionState::default()
            });
        }
        let ref_data = doc.to_reference()?;
        let mut state = WorkbenchSessionState::from_reference_and_peaks(
            &doc.project_name,
            &ref_data,
            None,
            &peaks,
        );
        state.chain_code = chain_code;
        Ok(state)
    }
}

/// Write a [`WorkbenchSessionState`] to a `.nef` file on disk.
pub fn write_session_nef_file(
    path: impl AsRef<Path>,
    state: &WorkbenchSessionState,
) -> Result<(), String> {
    let text = write_session_nef_str(state);
    fs::write(path.as_ref(), text).map_err(|e| {
        format!(
            "failed to write NEF session file {}: {e}",
            path.as_ref().display()
        )
    })
}

/// Read a [`WorkbenchSessionState`] from a `.nef` file on disk.
pub fn read_session_nef_file(path: impl AsRef<Path>) -> Result<WorkbenchSessionState, String> {
    let text = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read NEF session file {}: {e}",
            path.as_ref().display()
        )
    })?;
    read_session_nef_str(&text)
}

fn reconcile_external_nef_shift_edits(
    assignments: &mut BTreeMap<usize, ResidueAssignmentState>,
    sequence: &[String],
    seq_ids: &[i32],
    chain_code: &str,
    shift_lists: &[NefShiftList],
) {
    let primary_chain = if chain_code.trim().is_empty() {
        "A"
    } else {
        chain_code.trim()
    };
    let mut seq_map: BTreeMap<i32, usize> = BTreeMap::new();
    for (idx, &sid) in seq_ids.iter().enumerate() {
        seq_map.insert(sid, idx);
    }
    if seq_map.is_empty() {
        for idx in 0..sequence.len() {
            seq_map.insert((idx + 1) as i32, idx);
        }
    }

    let mut external_by_res: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
    for slist in shift_lists {
        for sh in &slist.shifts {
            if sh.chain_code != primary_chain && sh.chain_code != "." {
                continue;
            }
            let Some(sid) = sh.sequence_code else {
                continue;
            };
            let Some(&res_idx) = seq_map.get(&sid) else {
                continue;
            };
            let atom = match sh.atom_name.as_str() {
                "HN" => "H".to_string(),
                "CO" => "C".to_string(),
                other => other.to_string(),
            };
            external_by_res
                .entry(res_idx)
                .or_default()
                .insert(atom, sh.value);
        }
    }

    let fmt_4dp = |v: f64| -> String {
        let nv = if v == 0.0 { 0.0 } else { v };
        format!("{nv:.4}")
    };

    // Update or prune existing residue shift maps against the externally edited standard shifts
    let existing_keys: Vec<usize> = assignments.keys().copied().collect();
    for res_idx in existing_keys {
        let res_name = sequence.get(res_idx).map(String::as_str).unwrap_or("GLY");
        let ext_map = external_by_res.remove(&res_idx).unwrap_or_default();
        if let Some(entry) = assignments.get_mut(&res_idx) {
            let mut next_shifts: BTreeMap<String, f64> = BTreeMap::new();
            // Carry over unchanged atoms at full IEEE-754 precision; update changed atoms; drop deleted atoms
            for (raw_atom, &old_val) in &entry.shifts {
                let nef_atom = normalize_nef_atom_name(res_name, raw_atom);
                if let Some(&ext_val) = ext_map.get(raw_atom).or_else(|| ext_map.get(&nef_atom)) {
                    if fmt_4dp(old_val) == fmt_4dp(ext_val) {
                        next_shifts.insert(raw_atom.clone(), old_val);
                    } else {
                        next_shifts.insert(raw_atom.clone(), ext_val);
                    }
                }
            }
            // Insert newly added external atoms
            for (ext_atom, &ext_val) in &ext_map {
                let already_mapped = next_shifts.keys().any(|raw_atom| {
                    raw_atom == ext_atom || normalize_nef_atom_name(res_name, raw_atom) == *ext_atom
                });
                if !already_mapped {
                    next_shifts.insert(ext_atom.clone(), ext_val);
                }
            }
            entry.shifts = next_shifts;
        }
        if assignments
            .get(&res_idx)
            .is_some_and(|e| e.shifts.is_empty() && e.spin_system.is_none())
        {
            assignments.remove(&res_idx);
        }
    }

    // Insert brand-new residue assignments added externally to `nef_chemical_shift_list`
    for (res_idx, ext_map) in external_by_res {
        if !ext_map.is_empty() {
            assignments.insert(
                res_idx,
                ResidueAssignmentState {
                    spin_system: None,
                    shifts: ext_map,
                    confidence: 1.0,
                },
            );
        }
    }
}
