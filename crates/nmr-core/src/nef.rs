//! NMR Exchange Format (NEF) v1.1 reader, writer, and conversion utilities.
//!
//! Provenance: Clean-room original implementation of the open NEF v1.1 specification:
//! Gutmanas, A. et al., "NMR Exchange Format: a unified and open standard for
//! representation of NMR restraint data", *Nature Structural & Molecular Biology*
//! 22, 433–434 (2015), covering `nef_nmr_meta_data`, `nef_molecular_system`,
//! `nef_chemical_shift_list`, `nef_nmr_spectrum`, `nef_distance_restraint_list`,
//! `nef_dihedral_restraint_list`, and vendor-prefixed `save_nmrassign_workbench`
//! (`nmrassign_workbench_session`) saveframes per NEF v1.1 extensibility rules.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::peaks::Peak;
pub use crate::session::{
    read_session_nef_file, read_session_nef_str, write_session_nef_file, write_session_nef_str,
};

/// Alias to `nmr-core`'s `Reference` shift/sequence container for interoperability.
pub type StarData = Reference;

/// Alias for a single polymer residue entry in `nef_molecular_system`.
pub type SequenceResidue = NefSequenceResidue;

/// Alias for a single chemical shift entry in `nef_chemical_shift_list`.
pub type AssignedShift = NefAssignedShift;

/// Classification of NEF v1.1 stereospecific / degenerate / wildcard atom nomenclature
/// (Gutmanas et al., *Nat. Struct. Mol. Biol.* 2015).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NefAtomWildcardKind {
    /// Stereospecifically resolved single atom (e.g. `CA`, `H`, `HB2`, `CG1`).
    Exact,
    /// Equivalent or degenerate atom set denoted with `%` (e.g. `HB%`, `HD1%`, `HG%`).
    DegenerateSet,
    /// Non-stereospecifically assigned prochiral pair denoted with `x`/`y` (e.g. `HBx`, `HBy`, `CDx%`).
    NonStereoProchiral,
}

/// Classify a NEF atom name according to NEF v1.1 wildcard and prochiral conventions.
pub fn classify_nef_atom_name(atom_name: &str) -> NefAtomWildcardKind {
    let has_xy = atom_name.contains('x')
        || atom_name.contains('y')
        || atom_name.contains('X')
        || atom_name.contains('Y');
    if has_xy {
        NefAtomWildcardKind::NonStereoProchiral
    } else if atom_name.contains('%') || atom_name.contains('*') {
        NefAtomWildcardKind::DegenerateSet
    } else {
        NefAtomWildcardKind::Exact
    }
}

/// Normalize legacy `*` pseudo-atom notation or methyl groups into canonical NEF v1.1 `%` notation.
pub fn normalize_nef_atom_name(residue_name: &str, atom_name: &str) -> String {
    let trimmed = atom_name.trim();
    if trimmed.contains('*') {
        return trimmed.replace('*', "%");
    }
    let res = residue_name.trim().to_ascii_uppercase();
    let up = trimmed.to_ascii_uppercase();
    match (res.as_str(), up.as_str()) {
        ("ALA", "HB" | "QB") => "HB%".to_string(),
        ("VAL", "HG1" | "QG1") => "HG1%".to_string(),
        ("VAL", "HG2" | "QG2") => "HG2%".to_string(),
        ("LEU", "HD1" | "QD1") => "HD1%".to_string(),
        ("LEU", "HD2" | "QD2") => "HD2%".to_string(),
        ("ILE", "HG2" | "QG2") => "HG2%".to_string(),
        ("ILE", "HD1" | "QD1") => "HD1%".to_string(),
        ("THR", "HG2" | "QG2") => "HG2%".to_string(),
        ("MET", "HE" | "QE") => "HE%".to_string(),
        (_, "HN") => "H".to_string(),
        (_, "CO") => "C".to_string(),
        _ => trimmed.to_string(),
    }
}

/// Infer element symbol and standard NMR-active isotope number (`1H`, `13C`, `15N`, `31P`) from an atom name.
pub fn infer_element_and_isotope(atom_name: &str) -> (Option<String>, Option<u16>) {
    let first = atom_name
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase());
    match first {
        Some('H') | Some('Q') => (Some("H".to_string()), Some(1)),
        Some('C') => (Some("C".to_string()), Some(13)),
        Some('N') => (Some("N".to_string()), Some(15)),
        Some('P') => (Some("P".to_string()), Some(31)),
        Some('O') => (Some("O".to_string()), None),
        Some('S') => (Some("S".to_string()), None),
        _ => (None, None),
    }
}

/// `save_nef_nmr_meta_data` saveframe metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefMetaData {
    pub format_name: String,
    pub format_version: String,
    pub program_name: String,
    pub program_version: String,
    pub creation_date: String,
    pub uuid: String,
    pub coordinate_file_name: Option<String>,
}

impl Default for NefMetaData {
    fn default() -> Self {
        Self {
            format_name: "nmr_exchange_format".to_string(),
            format_version: "1.1".to_string(),
            program_name: "nmr_assign".to_string(),
            program_version: crate::VERSION.to_string(),
            creation_date: "2026-01-01T00:00:00Z".to_string(),
            uuid: "nmr_assign-1.1-0001".to_string(),
            coordinate_file_name: None,
        }
    }
}

/// Residue entry in `_nef_sequence` within `save_nef_molecular_system`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefSequenceResidue {
    pub chain_code: String,
    pub sequence_code: i32,
    pub residue_name: String,
    pub linking: String,
    pub residue_variant: Option<String>,
    pub cis_peptide: Option<bool>,
}

/// `save_nef_molecular_system` saveframe.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefMolecularSystem {
    pub residues: Vec<NefSequenceResidue>,
}

impl NefMolecularSystem {
    pub fn from_reference(ref_data: &Reference, chain_code: &str) -> Self {
        let n = ref_data.sequence.len();
        let mut residues = Vec::with_capacity(n);
        for (i, res) in ref_data.sequence.iter().enumerate() {
            let seq_code = ref_data.seq_ids.get(i).copied().unwrap_or((i + 1) as i32);
            let linking = if n == 1 {
                "single"
            } else if i == 0 {
                "start"
            } else if i + 1 == n {
                "end"
            } else {
                "middle"
            };
            residues.push(NefSequenceResidue {
                chain_code: chain_code.to_string(),
                sequence_code: seq_code,
                residue_name: res.to_ascii_uppercase(),
                linking: linking.to_string(),
                residue_variant: None,
                cis_peptide: if res.eq_ignore_ascii_case("PRO") {
                    Some(false)
                } else {
                    None
                },
            });
        }
        Self { residues }
    }
}

/// Single row in `_nef_chemical_shift`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefAssignedShift {
    pub chain_code: String,
    pub sequence_code: Option<i32>,
    pub residue_name: String,
    pub atom_name: String,
    pub element: Option<String>,
    pub isotope_number: Option<u16>,
    pub value: f64,
    pub value_uncertainty: Option<f64>,
}

/// `save_nef_chemical_shift_list_<name>` saveframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefShiftList {
    pub name: String,
    pub atom_chem_shift_units: String,
    pub shifts: Vec<NefAssignedShift>,
}

impl Default for NefShiftList {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            atom_chem_shift_units: "ppm".to_string(),
            shifts: Vec::new(),
        }
    }
}

impl NefShiftList {
    pub fn from_reference(
        ref_data: &Reference,
        confidence: Option<&BTreeMap<usize, f64>>,
        chain_code: &str,
    ) -> Self {
        let atom_order = ["H", "N", "CA", "CB", "C"];
        let mut entries: Vec<(usize, &str, f64)> = ref_data
            .shifts
            .iter()
            .map(|(&(idx, ref atom), &val)| (idx, atom.as_str(), val))
            .collect();
        entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
            let oa = atom_order.iter().position(|&x| x == aa).unwrap_or(99);
            let ob = atom_order.iter().position(|&x| x == ab).unwrap_or(99);
            (ia, oa, aa).cmp(&(ib, ob, ab))
        });

        let mut shifts = Vec::with_capacity(entries.len());
        for (idx, raw_atom, val) in entries {
            let Some(res_name) = ref_data.sequence.get(idx) else {
                continue;
            };
            let seq_code = ref_data
                .seq_ids
                .get(idx)
                .copied()
                .unwrap_or((idx + 1) as i32);
            let nef_atom = normalize_nef_atom_name(res_name, raw_atom);
            let (element, isotope_number) = infer_element_and_isotope(&nef_atom);
            // Encode uncertainty inversely to confidence when confidence is supplied, else standard default
            let value_uncertainty = confidence.and_then(|c| c.get(&idx)).map(|&conf| {
                let base = match element.as_deref() {
                    Some("H") => 0.02,
                    Some("N") => 0.20,
                    _ => 0.15,
                };
                ((base * (2.0 - conf.clamp(0.0, 1.0))) * 1000.0).round() / 1000.0
            });
            shifts.push(NefAssignedShift {
                chain_code: chain_code.to_string(),
                sequence_code: Some(seq_code),
                residue_name: res_name.to_ascii_uppercase(),
                atom_name: nef_atom,
                element,
                isotope_number,
                value: val,
                value_uncertainty,
            });
        }

        Self {
            name: "default".to_string(),
            atom_chem_shift_units: "ppm".to_string(),
            shifts,
        }
    }
}

/// Dimension descriptor in `_nef_spectrum_dimension`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefSpectrumDimension {
    pub dimension_id: usize,
    pub axis_unit: String,
    pub axis_code: String,
    pub spectrometer_frequency: Option<f64>,
    pub spectral_width: Option<f64>,
    pub value_first_point: Option<f64>,
    pub folding: String,
    pub absolute_peak_positions: bool,
    pub is_acquisition: bool,
}

impl NefSpectrumDimension {
    pub fn ppm_axis(dimension_id: usize, axis_code: &str, freq_mhz: f64, is_acq: bool) -> Self {
        Self {
            dimension_id,
            axis_unit: "ppm".to_string(),
            axis_code: axis_code.to_string(),
            spectrometer_frequency: Some(freq_mhz),
            spectral_width: None,
            value_first_point: None,
            folding: "circular".to_string(),
            absolute_peak_positions: true,
            is_acquisition: is_acq,
        }
    }
}

/// Peak entry in `_nef_peak`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefPeak {
    pub index: usize,
    pub peak_id: usize,
    pub positions: Vec<f64>,
    pub position_uncertainties: Vec<Option<f64>>,
    pub height: Option<f64>,
    pub height_uncertainty: Option<f64>,
    pub volume: Option<f64>,
    pub volume_uncertainty: Option<f64>,
    pub chain_codes: Vec<Option<String>>,
    pub sequence_codes: Vec<Option<i32>>,
    pub residue_names: Vec<Option<String>>,
    pub atom_names: Vec<Option<String>>,
    pub details: Option<String>,
}

impl NefPeak {
    pub fn unassigned(
        index: usize,
        peak_id: usize,
        positions: Vec<f64>,
        height: Option<f64>,
        volume: Option<f64>,
        details: Option<String>,
    ) -> Self {
        let ndim = positions.len();
        Self {
            index,
            peak_id,
            positions,
            position_uncertainties: vec![None; ndim],
            height,
            height_uncertainty: None,
            volume,
            volume_uncertainty: None,
            chain_codes: vec![None; ndim],
            sequence_codes: vec![None; ndim],
            residue_names: vec![None; ndim],
            atom_names: vec![None; ndim],
            details,
        }
    }
}

/// `save_nef_nmr_spectrum_<spectrum_name>` saveframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefPeakList {
    pub spectrum_name: String,
    pub experiment_classification: Option<String>,
    pub experiment_type: Option<String>,
    pub chemical_shift_list: String,
    pub dimensions: Vec<NefSpectrumDimension>,
    pub peaks: Vec<NefPeak>,
}

/// Single contribution row in `_nef_distance_restraint`.
///
/// Rows sharing the same `restraint_id` represent an ambiguous `OR` distance restraint
/// (`restraint_combination_id = None` renders `.`), while `Some(comb_id)` encodes combined
/// `AND`-within-`OR` groups per NEF v1.1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefDistanceRestraint {
    pub index: usize,
    pub restraint_id: usize,
    pub restraint_combination_id: Option<usize>,
    pub chain_code_1: String,
    pub sequence_code_1: i32,
    pub residue_name_1: String,
    pub atom_name_1: String,
    pub chain_code_2: String,
    pub sequence_code_2: i32,
    pub residue_name_2: String,
    pub atom_name_2: String,
    pub weight: f64,
    pub target_value: Option<f64>,
    pub target_value_uncertainty: Option<f64>,
    pub lower_linear_limit: Option<f64>,
    pub lower_limit: Option<f64>,
    pub upper_limit: f64,
    pub upper_linear_limit: Option<f64>,
}

impl NefDistanceRestraint {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: usize,
        restraint_id: usize,
        restraint_combination_id: Option<usize>,
        chain_1: &str,
        seq_1: i32,
        res_1: &str,
        atom_1: &str,
        chain_2: &str,
        seq_2: i32,
        res_2: &str,
        atom_2: &str,
        target_value: Option<f64>,
        lower_limit: Option<f64>,
        upper_limit: f64,
        weight: f64,
    ) -> Self {
        Self {
            index,
            restraint_id,
            restraint_combination_id,
            chain_code_1: chain_1.to_string(),
            sequence_code_1: seq_1,
            residue_name_1: res_1.to_string(),
            atom_name_1: atom_1.to_string(),
            chain_code_2: chain_2.to_string(),
            sequence_code_2: seq_2,
            residue_name_2: res_2.to_string(),
            atom_name_2: atom_2.to_string(),
            weight,
            target_value,
            target_value_uncertainty: None,
            lower_linear_limit: None,
            lower_limit,
            upper_limit,
            upper_linear_limit: None,
        }
    }
}

/// `save_nef_distance_restraint_list_<name>` saveframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefDistanceRestraintList {
    pub name: String,
    pub potential_type: String,
    pub restraints: Vec<NefDistanceRestraint>,
}

impl Default for NefDistanceRestraintList {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            potential_type: "upper-bound-parabolic".to_string(),
            restraints: Vec::new(),
        }
    }
}

/// Single torsion restraint row in `_nef_dihedral_restraint` (`PHI`, `PSI`, etc.).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefDihedralRestraint {
    pub index: usize,
    pub restraint_id: usize,
    pub restraint_combination_id: Option<usize>,
    pub chain_code_1: String,
    pub sequence_code_1: i32,
    pub residue_name_1: String,
    pub atom_name_1: String,
    pub chain_code_2: String,
    pub sequence_code_2: i32,
    pub residue_name_2: String,
    pub atom_name_2: String,
    pub chain_code_3: String,
    pub sequence_code_3: i32,
    pub residue_name_3: String,
    pub atom_name_3: String,
    pub chain_code_4: String,
    pub sequence_code_4: i32,
    pub residue_name_4: String,
    pub atom_name_4: String,
    pub weight: f64,
    pub target_value: f64,
    pub target_value_uncertainty: Option<f64>,
    pub lower_linear_limit: Option<f64>,
    pub lower_limit: f64,
    pub upper_limit: f64,
    pub upper_linear_limit: Option<f64>,
    pub name: String,
}

impl NefDihedralRestraint {
    /// Construct a canonical NEF `PHI` (`C(i-1) - N(i) - CA(i) - C(i)`) restraint.
    #[allow(clippy::too_many_arguments)]
    pub fn phi(
        index: usize,
        restraint_id: usize,
        chain_code: &str,
        prev_seq_code: i32,
        prev_res_name: &str,
        seq_code: i32,
        res_name: &str,
        target_deg: f64,
        half_width_deg: f64,
        weight: f64,
    ) -> Self {
        Self {
            index,
            restraint_id,
            restraint_combination_id: None,
            chain_code_1: chain_code.to_string(),
            sequence_code_1: prev_seq_code,
            residue_name_1: prev_res_name.to_string(),
            atom_name_1: "C".to_string(),
            chain_code_2: chain_code.to_string(),
            sequence_code_2: seq_code,
            residue_name_2: res_name.to_string(),
            atom_name_2: "N".to_string(),
            chain_code_3: chain_code.to_string(),
            sequence_code_3: seq_code,
            residue_name_3: res_name.to_string(),
            atom_name_3: "CA".to_string(),
            chain_code_4: chain_code.to_string(),
            sequence_code_4: seq_code,
            residue_name_4: res_name.to_string(),
            atom_name_4: "C".to_string(),
            weight,
            target_value: target_deg,
            target_value_uncertainty: Some(half_width_deg),
            lower_linear_limit: None,
            lower_limit: wrap_deg(target_deg - half_width_deg),
            upper_limit: wrap_deg(target_deg + half_width_deg),
            upper_linear_limit: None,
            name: "PHI".to_string(),
        }
    }

    /// Construct a canonical NEF `PSI` (`N(i) - CA(i) - C(i) - N(i+1)`) restraint.
    #[allow(clippy::too_many_arguments)]
    pub fn psi(
        index: usize,
        restraint_id: usize,
        chain_code: &str,
        seq_code: i32,
        res_name: &str,
        next_seq_code: i32,
        next_res_name: &str,
        target_deg: f64,
        half_width_deg: f64,
        weight: f64,
    ) -> Self {
        Self {
            index,
            restraint_id,
            restraint_combination_id: None,
            chain_code_1: chain_code.to_string(),
            sequence_code_1: seq_code,
            residue_name_1: res_name.to_string(),
            atom_name_1: "N".to_string(),
            chain_code_2: chain_code.to_string(),
            sequence_code_2: seq_code,
            residue_name_2: res_name.to_string(),
            atom_name_2: "CA".to_string(),
            chain_code_3: chain_code.to_string(),
            sequence_code_3: seq_code,
            residue_name_3: res_name.to_string(),
            atom_name_3: "C".to_string(),
            chain_code_4: chain_code.to_string(),
            sequence_code_4: next_seq_code,
            residue_name_4: next_res_name.to_string(),
            atom_name_4: "N".to_string(),
            weight,
            target_value: target_deg,
            target_value_uncertainty: Some(half_width_deg),
            lower_linear_limit: None,
            lower_limit: wrap_deg(target_deg - half_width_deg),
            upper_limit: wrap_deg(target_deg + half_width_deg),
            upper_linear_limit: None,
            name: "PSI".to_string(),
        }
    }
}

fn wrap_deg(angle: f64) -> f64 {
    let mut a = (angle + 180.0) % 360.0;
    if a < 0.0 {
        a += 360.0;
    }
    a - 180.0
}

/// `save_nef_dihedral_restraint_list_<name>` saveframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefDihedralRestraintList {
    pub name: String,
    pub potential_type: String,
    pub restraints: Vec<NefDihedralRestraint>,
}

impl Default for NefDihedralRestraintList {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            potential_type: "square-well-parabolic".to_string(),
            restraints: Vec::new(),
        }
    }
}

/// Lossless sequence residue row in `_nmrassign_residue` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefWorkbenchResidueRow {
    pub seq_id: i32,
    pub residue_name: String,
}

/// Lossless peak row in `_nmrassign_peak` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefWorkbenchPeakRow {
    pub experiment: String,
    pub h: f64,
    pub n: f64,
    pub c: Option<f64>,
    pub sign: i32,
    pub height: Option<f64>,
    pub volume: Option<f64>,
    pub h2: Option<f64>,
}

/// Lossless residue assignment row in `_nmrassign_assignment` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefWorkbenchAssignmentRow {
    pub residue_seq_idx: usize,
    pub spin_system_id: Option<usize>,
    pub confidence: f64,
}

/// Lossless assigned atom shift row in `_nmrassign_assignment_shift` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefWorkbenchShiftRow {
    pub residue_seq_idx: usize,
    pub atom_name: String,
    pub value: f64,
}

/// Lossless user lock row in `_nmrassign_lock` within `save_nmrassign_workbench`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefWorkbenchLockRow {
    pub spin_system_id: usize,
    pub residue_seq_idx: usize,
}

/// Lossless confirmed sequential link row in `_nmrassign_confirmed_link` within `save_nmrassign_workbench`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefWorkbenchLinkRow {
    pub pred_spin_system: usize,
    pub succ_spin_system: usize,
}

/// Lossless 2D/slice contour layer row in `_nmrassign_contour_layer` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefWorkbenchContourLayerRow {
    pub layer_id: String,
    pub visible: bool,
    pub base_level: f32,
    pub level_factor: f32,
    pub num_levels: u32,
    pub show_negative: bool,
    pub pos_rgba: [f32; 4],
    pub neg_rgba: [f32; 4],
    pub ppm_nudge: [f32; 2],
}

/// Lossless FID processing recipe row in `_nmrassign_processing_recipe` within `save_nmrassign_workbench`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NefWorkbenchRecipeRow {
    pub spectrum_id: String,
    pub recipe_json: String,
}

/// Vendor-prefixed `save_nmrassign_workbench` saveframe per NEF v1.1 extensibility rules
/// (Gutmanas et al., *Nat. Struct. Mol. Biol.* 2015).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefWorkbenchSaveframe {
    pub schema_version: String,
    pub protein_name: String,
    pub chain_code: String,
    pub reference_offsets: [f64; 5],
    pub qc_verdict: Option<String>,
    pub view_bounds_ppm: Option<[f64; 4]>,
    #[serde(default)]
    pub shift_list_crc32: Option<String>,
    pub residues: Vec<NefWorkbenchResidueRow>,
    pub peaks: Vec<NefWorkbenchPeakRow>,
    pub assignments: Vec<NefWorkbenchAssignmentRow>,
    pub assignment_shifts: Vec<NefWorkbenchShiftRow>,
    pub locks: Vec<NefWorkbenchLockRow>,
    pub confirmed_links: Vec<NefWorkbenchLinkRow>,
    pub contour_layers: Vec<NefWorkbenchContourLayerRow>,
    pub processing_recipes: Vec<NefWorkbenchRecipeRow>,
}

/// Complete in-memory NEF v1.1 data block (`data_nef_<project_name>`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NefDocument {
    pub project_name: String,
    pub meta_data: NefMetaData,
    pub molecular_system: NefMolecularSystem,
    pub shift_lists: Vec<NefShiftList>,
    pub peak_lists: Vec<NefPeakList>,
    pub distance_restraint_lists: Vec<NefDistanceRestraintList>,
    pub dihedral_restraint_lists: Vec<NefDihedralRestraintList>,
    /// Optional vendor-prefixed `save_nmrassign_workbench` session saveframe (`NEF v1.1` extensibility).
    #[serde(default)]
    pub workbench_session: Option<NefWorkbenchSaveframe>,
}

impl NefDocument {
    pub fn new(project_name: impl Into<String>) -> Self {
        Self {
            project_name: sanitize_identifier(&project_name.into()),
            meta_data: NefMetaData::default(),
            molecular_system: NefMolecularSystem::default(),
            shift_lists: Vec::new(),
            peak_lists: Vec::new(),
            distance_restraint_lists: Vec::new(),
            dihedral_restraint_lists: Vec::new(),
            workbench_session: None,
        }
    }

    /// Build a NEF v1.1 document from `nmr-core`'s `Reference` shifts and `Peak` lists.
    pub fn from_reference_and_peaks(
        project_name: &str,
        ref_data: &Reference,
        confidence: Option<&BTreeMap<usize, f64>>,
        peaks: &[Peak],
    ) -> Self {
        let mut doc = Self::new(project_name);
        doc.molecular_system = NefMolecularSystem::from_reference(ref_data, "A");
        if !ref_data.shifts.is_empty() {
            doc.shift_lists
                .push(NefShiftList::from_reference(ref_data, confidence, "A"));
        }
        doc.peak_lists = peaks_to_nef_lists(peaks);
        doc
    }

    /// Extract the polymer sequence and primary chemical shift list as `nmrstar::Reference`.
    pub fn to_reference(&self) -> Result<Reference, String> {
        if self.molecular_system.residues.is_empty() {
            return Err("NEF document contains an empty nef_molecular_system".to_string());
        }

        // Choose primary chain (first chain in molecular_system)
        let primary_chain = &self.molecular_system.residues[0].chain_code;
        let chain_res: Vec<&NefSequenceResidue> = self
            .molecular_system
            .residues
            .iter()
            .filter(|r| &r.chain_code == primary_chain)
            .collect();

        let sequence: Vec<String> = chain_res
            .iter()
            .map(|r| r.residue_name.to_ascii_uppercase())
            .collect();
        let seq_ids: Vec<i32> = chain_res.iter().map(|r| r.sequence_code).collect();

        let mut seq_map: BTreeMap<i32, usize> = BTreeMap::new();
        for (idx, &sid) in seq_ids.iter().enumerate() {
            seq_map.insert(sid, idx);
        }

        let mut out = Reference::new(sequence, seq_ids);
        if let Some(slist) = self.shift_lists.first() {
            for sh in &slist.shifts {
                if &sh.chain_code != primary_chain && sh.chain_code != "." {
                    continue;
                }
                let Some(sid) = sh.sequence_code else {
                    continue;
                };
                let Some(&idx) = seq_map.get(&sid) else {
                    continue;
                };
                let atom = match sh.atom_name.as_str() {
                    "HN" => "H".to_string(),
                    "CO" => "C".to_string(),
                    other => other.to_string(),
                };
                out.shifts.insert((idx, atom), sh.value);
            }
        }
        Ok(out)
    }

    /// Convert all `save_nef_nmr_spectrum_*` saveframes back into flat `nmr-core` `Peak` structs.
    pub fn to_peaks(&self) -> Vec<Peak> {
        nef_lists_to_peaks(&self.peak_lists)
    }
}

/// Convert flat `nmr-core` `Peak` slice into per-experiment `NefPeakList` saveframes.
pub fn peaks_to_nef_lists(peaks: &[Peak]) -> Vec<NefPeakList> {
    let mut grouped: BTreeMap<String, Vec<&Peak>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for p in peaks {
        if !grouped.contains_key(&p.exp) {
            order.push(p.exp.clone());
        }
        grouped.entry(p.exp.clone()).or_default().push(p);
    }

    let mut out = Vec::with_capacity(order.len());
    for exp in order {
        let Some(plist) = grouped.get(&exp) else {
            continue;
        };
        let (dims, classification, exp_type) = spectrum_dimensions_for_exp(&exp);
        let mut nef_peaks = Vec::with_capacity(plist.len());
        for (i, p) in plist.iter().enumerate() {
            let positions = match exp.as_str() {
                "HSQC" => vec![p.h, p.n],
                "C13HSQC" => vec![p.h, p.c.unwrap_or(0.0)],
                "NOESY15N" | "HCCCONH" => {
                    let third = p.h2.or(p.c).unwrap_or(0.0);
                    vec![p.h, p.n, third]
                }
                "NOESY13C" | "HCCHTOCSY" => {
                    let heavy = p.c.unwrap_or(p.n);
                    let indirect = p.h2.or(p.c).unwrap_or(0.0);
                    vec![p.h, heavy, indirect]
                }
                _ => vec![p.h, p.n, p.c.unwrap_or(0.0)],
            };
            let sign_f = if p.sign < 0 { -1.0 } else { 1.0 };
            let signed_height = p.height.map(|h| h.abs() * sign_f);
            let signed_volume = p.volume.map(|v| v.abs() * sign_f);
            let details = if p.height.is_none() && p.volume.is_none() && p.sign < 0 {
                Some("sign=-1".to_string())
            } else {
                None
            };
            nef_peaks.push(NefPeak::unassigned(
                i + 1,
                i + 1,
                positions,
                signed_height,
                signed_volume,
                details,
            ));
        }

        out.push(NefPeakList {
            spectrum_name: exp,
            experiment_classification: Some(classification.to_string()),
            experiment_type: Some(exp_type.to_string()),
            chemical_shift_list: "nef_chemical_shift_list_default".to_string(),
            dimensions: dims,
            peaks: nef_peaks,
        });
    }
    out
}

/// Convert NEF spectrum peak lists back into `nmr-core` `Peak` structs.
pub fn nef_lists_to_peaks(lists: &[NefPeakList]) -> Vec<Peak> {
    let mut out = Vec::new();
    for spec in lists {
        let exp = spec
            .spectrum_name
            .trim()
            .to_ascii_uppercase()
            .replace(['(', ')', '-'], "");
        for np in &spec.peaks {
            if np.positions.len() < 2 {
                continue;
            }
            let raw_sign = if let Some(h) = np.height {
                if h < 0.0 {
                    -1
                } else {
                    1
                }
            } else if let Some(v) = np.volume {
                if v < 0.0 {
                    -1
                } else {
                    1
                }
            } else if np.details.as_deref() == Some("sign=-1") {
                -1
            } else {
                1
            };
            let height = np.height.map(f64::abs);
            let volume = np.volume.map(f64::abs);

            let (h, n, c, h2) = match exp.as_str() {
                "HSQC" => (np.positions[0], np.positions[1], None, None),
                "C13HSQC" => (np.positions[0], 0.0, Some(np.positions[1]), None),
                "NOESY15N" => (
                    np.positions[0],
                    np.positions[1],
                    None,
                    np.positions.get(2).copied(),
                ),
                "HCCCONH" => (
                    np.positions[0],
                    np.positions[1],
                    np.positions.get(2).copied(),
                    np.positions.get(2).copied(),
                ),
                "NOESY13C" => (
                    np.positions[0],
                    0.0,
                    Some(np.positions[1]),
                    np.positions.get(2).copied(),
                ),
                "HCCHTOCSY" => (
                    np.positions[0],
                    np.positions[1],
                    Some(np.positions[1]),
                    np.positions.get(2).copied(),
                ),
                _ => (
                    np.positions[0],
                    np.positions[1],
                    np.positions.get(2).copied(),
                    None,
                ),
            };

            out.push(Peak {
                exp: exp.clone(),
                h,
                n,
                c,
                sign: raw_sign,
                height,
                volume,
                h2,
            });
        }
    }
    out
}

fn spectrum_dimensions_for_exp(
    exp: &str,
) -> (Vec<NefSpectrumDimension>, &'static str, &'static str) {
    match exp {
        "HSQC" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
            ],
            "H_N",
            "15N-HSQC",
        ),
        "C13HSQC" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "13C", 150.90, false),
            ],
            "H_C",
            "13C-HSQC",
        ),
        "NOESY15N" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "1H", 600.13, false),
            ],
            "H_N.through-space",
            "15N-NOESY-HSQC",
        ),
        "HCCCONH" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "1H", 600.13, false),
            ],
            "H_N_C_H",
            "H(CCCO)NH",
        ),
        "NOESY13C" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "13C", 150.90, false),
                NefSpectrumDimension::ppm_axis(3, "1H", 600.13, false),
            ],
            "H_C.through-space",
            "13C-NOESY-HSQC",
        ),
        "HCCHTOCSY" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "13C", 150.90, false),
                NefSpectrumDimension::ppm_axis(3, "1H", 600.13, false),
            ],
            "H_C_C_H",
            "HCCH-TOCSY",
        ),
        "HNCA" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CA",
            "HNCA",
        ),
        "HNCOCA" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CO_CA",
            "HN(CO)CA",
        ),
        "HNCACB" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CAB",
            "HNCACB",
        ),
        "CBCACONH" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CO_CAB",
            "CBCA(CO)NH",
        ),
        "HNCO" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CO",
            "HNCO",
        ),
        "HNCACO" => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_CA_CO",
            "HN(CA)CO",
        ),
        _ => (
            vec![
                NefSpectrumDimension::ppm_axis(1, "1H", 600.13, true),
                NefSpectrumDimension::ppm_axis(2, "15N", 60.81, false),
                NefSpectrumDimension::ppm_axis(3, "13C", 150.90, false),
            ],
            "H_N_C",
            "3D-triple-resonance",
        ),
    }
}

// ============================================================================
// STAR / NEF v1.1 Tokenizer & Saveframe Parser
// ============================================================================

#[derive(Clone, Debug)]
struct Saveframe {
    name: String,
    tags: BTreeMap<String, String>,
    loops: Vec<(Vec<String>, Vec<Vec<String>>)>,
}

fn tokenize_line(line: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            break;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut buf = String::new();
            while i < n {
                if chars[i] == quote
                    && (i + 1 == n || chars[i + 1].is_whitespace() || chars[i + 1] == '#')
                {
                    i += 1;
                    break;
                }
                buf.push(chars[i]);
                i += 1;
            }
            out.push(buf);
        } else {
            let mut buf = String::new();
            while i < n && !chars[i].is_whitespace() && chars[i] != '#' {
                buf.push(chars[i]);
                i += 1;
            }
            out.push(buf);
        }
    }
}

fn tokenize_star(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(rest) = line.strip_prefix(';') {
            let mut buf = Vec::new();
            if !rest.is_empty() {
                buf.push(rest);
            }
            i += 1;
            while i < lines.len() && !lines[i].starts_with(';') {
                buf.push(lines[i]);
                i += 1;
            }
            out.push(buf.join("\n"));
            i += 1;
            continue;
        }
        tokenize_line(line, &mut out);
        i += 1;
    }
    out
}

fn parse_nef_tokens(tokens: &[String]) -> Result<(String, Vec<Saveframe>), String> {
    let mut project = "default".to_string();
    let mut frames = Vec::new();
    let mut i = 0;
    let n = tokens.len();

    while i < n {
        let tok = &tokens[i];
        if let Some(rest) = tok.strip_prefix("data_") {
            project = rest.strip_prefix("nef_").unwrap_or(rest).to_string();
            i += 1;
            continue;
        }
        if tok.len() > 5 && tok[..5].eq_ignore_ascii_case("save_") {
            let sf_name = tok[5..].to_string();
            i += 1;
            let mut tags = BTreeMap::new();
            let mut loops = Vec::new();

            while i < n {
                if tokens[i].eq_ignore_ascii_case("save_") {
                    i += 1;
                    break;
                }
                if tokens[i].eq_ignore_ascii_case("loop_") {
                    i += 1;
                    let mut col_tags = Vec::new();
                    while i < n && tokens[i].starts_with('_') {
                        col_tags.push(tokens[i].to_ascii_lowercase());
                        i += 1;
                    }
                    let mut vals = Vec::new();
                    while i < n
                        && !tokens[i].eq_ignore_ascii_case("stop_")
                        && !tokens[i].eq_ignore_ascii_case("save_")
                        && !tokens[i].eq_ignore_ascii_case("loop_")
                        && !(tokens[i].len() > 5 && tokens[i][..5].eq_ignore_ascii_case("save_"))
                    {
                        vals.push(tokens[i].clone());
                        i += 1;
                    }
                    if i < n && tokens[i].eq_ignore_ascii_case("stop_") {
                        i += 1;
                    }
                    if !col_tags.is_empty() {
                        let ncols = col_tags.len();
                        if vals.len() % ncols != 0 {
                            return Err(format!(
                                "saveframe '{}': loop value count {} is not a multiple of column count {}",
                                sf_name,
                                vals.len(),
                                ncols
                            ));
                        }
                        let nrows = vals.len() / ncols;
                        let mut rows = Vec::with_capacity(nrows);
                        for r in 0..nrows {
                            rows.push(vals[r * ncols..(r + 1) * ncols].to_vec());
                        }
                        loops.push((col_tags, rows));
                    }
                } else if tokens[i].starts_with('_') {
                    let key = tokens[i].to_ascii_lowercase();
                    i += 1;
                    if i < n {
                        tags.insert(key, tokens[i].clone());
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }

            frames.push(Saveframe {
                name: sf_name,
                tags,
                loops,
            });
            continue;
        }
        i += 1;
    }

    Ok((project, frames))
}

fn col_idx(tags: &[String], suffix: &str) -> Option<usize> {
    let low_suf = format!(".{}", suffix.to_ascii_lowercase());
    tags.iter()
        .position(|t| t == suffix || t.ends_with(&low_suf))
}

fn opt_str(s: &str) -> Option<String> {
    if s.is_empty() || s == "." || s == "?" {
        None
    } else {
        Some(s.to_string())
    }
}

fn opt_f64(s: &str) -> Option<f64> {
    opt_str(s).and_then(|v| v.parse::<f64>().ok())
}

fn opt_i32(s: &str) -> Option<i32> {
    opt_str(s).and_then(|v| v.parse::<i32>().ok())
}

fn opt_usize(s: &str) -> Option<usize> {
    opt_str(s).and_then(|v| v.parse::<usize>().ok())
}

fn opt_bool(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => Some(true),
        "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

/// Read a NEF v1.1 document from a file on disk.
pub fn read_nef_file(path: impl AsRef<Path>) -> Result<NefDocument, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read NEF file {}: {e}", path.as_ref().display()))?;
    read_nef_str(&text)
}

/// Parse a NEF v1.1 document from a string.
pub fn read_nef_str(text: &str) -> Result<NefDocument, String> {
    let tokens = tokenize_star(text);
    if tokens.is_empty() {
        return Err("empty NEF input".to_string());
    }
    let (project, frames) = parse_nef_tokens(&tokens)?;
    if frames.is_empty() {
        return Err("no saveframes found in NEF input".to_string());
    }

    let mut doc = NefDocument::new(project);

    for sf in frames {
        let cat = sf
            .tags
            .iter()
            .find(|(k, _)| k.ends_with(".sf_category"))
            .map(|(_, v)| v.to_ascii_lowercase())
            .unwrap_or_else(|| sf.name.to_ascii_lowercase());

        if cat == "nef_nmr_meta_data" || sf.name.eq_ignore_ascii_case("nef_nmr_meta_data") {
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.format_name") {
                doc.meta_data.format_name = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.format_version") {
                doc.meta_data.format_version = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.program_name") {
                doc.meta_data.program_name = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.program_version") {
                doc.meta_data.program_version = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.creation_date") {
                doc.meta_data.creation_date = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.uuid") {
                doc.meta_data.uuid = v.clone();
            }
            if let Some(v) = sf.tags.get("_nef_nmr_meta_data.coordinate_file_name") {
                doc.meta_data.coordinate_file_name = opt_str(v);
            }
        } else if cat == "nef_molecular_system"
            || sf.name.eq_ignore_ascii_case("nef_molecular_system")
        {
            for (ltags, rows) in &sf.loops {
                let Some(c_chain) = col_idx(ltags, "chain_code") else {
                    continue;
                };
                let Some(c_seq) = col_idx(ltags, "sequence_code") else {
                    continue;
                };
                let Some(c_res) = col_idx(ltags, "residue_name") else {
                    continue;
                };
                let c_link = col_idx(ltags, "linking");
                let c_var = col_idx(ltags, "residue_variant");
                let c_cis = col_idx(ltags, "cis_peptide");

                for r in rows {
                    let seq_code = r[c_seq]
                        .parse::<i32>()
                        .map_err(|e| format!("invalid sequence_code '{}': {e}", r[c_seq]))?;
                    doc.molecular_system.residues.push(NefSequenceResidue {
                        chain_code: r[c_chain].clone(),
                        sequence_code: seq_code,
                        residue_name: r[c_res].clone(),
                        linking: c_link
                            .and_then(|c| opt_str(&r[c]))
                            .unwrap_or_else(|| "middle".to_string()),
                        residue_variant: c_var.and_then(|c| opt_str(&r[c])),
                        cis_peptide: c_cis.and_then(|c| opt_bool(&r[c])),
                    });
                }
            }
        } else if cat == "nef_chemical_shift_list"
            || sf
                .name
                .to_ascii_lowercase()
                .starts_with("nef_chemical_shift_list")
        {
            let name = sf
                .name
                .strip_prefix("nef_chemical_shift_list_")
                .unwrap_or(&sf.name)
                .to_string();
            let units = sf
                .tags
                .get("_nef_chemical_shift_list.atom_chem_shift_units")
                .cloned()
                .unwrap_or_else(|| "ppm".to_string());
            let mut shifts = Vec::new();
            for (ltags, rows) in &sf.loops {
                let Some(c_val) = col_idx(ltags, "value") else {
                    continue;
                };
                let Some(c_atom) = col_idx(ltags, "atom_name") else {
                    continue;
                };
                let c_chain = col_idx(ltags, "chain_code");
                let c_seq = col_idx(ltags, "sequence_code");
                let c_res = col_idx(ltags, "residue_name");
                let c_elem = col_idx(ltags, "element");
                let c_iso = col_idx(ltags, "isotope_number");
                let c_unc = col_idx(ltags, "value_uncertainty");

                for r in rows {
                    let Ok(val) = r[c_val].parse::<f64>() else {
                        continue;
                    };
                    if !val.is_finite() {
                        return Err(format!(
                            "non-finite chemical shift value {:?} in NEF shift list",
                            r[c_val]
                        ));
                    }
                    shifts.push(NefAssignedShift {
                        chain_code: c_chain
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "A".to_string()),
                        sequence_code: c_seq.and_then(|c| opt_i32(&r[c])),
                        residue_name: c_res
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| ".".to_string()),
                        atom_name: r[c_atom].clone(),
                        element: c_elem.and_then(|c| opt_str(&r[c])),
                        isotope_number: c_iso
                            .and_then(|c| opt_str(&r[c]))
                            .and_then(|s| s.parse::<u16>().ok()),
                        value: val,
                        value_uncertainty: c_unc.and_then(|c| opt_f64(&r[c])),
                    });
                }
            }
            doc.shift_lists.push(NefShiftList {
                name,
                atom_chem_shift_units: units,
                shifts,
            });
        } else if cat == "nef_nmr_spectrum"
            || sf.name.to_ascii_lowercase().starts_with("nef_nmr_spectrum")
        {
            let spec_name = sf
                .name
                .strip_prefix("nef_nmr_spectrum_")
                .unwrap_or(&sf.name)
                .to_string();
            let num_dim = sf
                .tags
                .get("_nef_nmr_spectrum.num_dimensions")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2);
            let cs_list = sf
                .tags
                .get("_nef_nmr_spectrum.chemical_shift_list")
                .cloned()
                .unwrap_or_else(|| "nef_chemical_shift_list_default".to_string());
            let exp_class = sf
                .tags
                .get("_nef_nmr_spectrum.experiment_classification")
                .and_then(|v| opt_str(v));
            let exp_type = sf
                .tags
                .get("_nef_nmr_spectrum.experiment_type")
                .and_then(|v| opt_str(v));

            let mut dims = Vec::new();
            let mut peaks = Vec::new();

            for (ltags, rows) in &sf.loops {
                if ltags
                    .iter()
                    .any(|t| t.starts_with("_nef_spectrum_dimension."))
                {
                    let c_id = col_idx(ltags, "dimension_id");
                    let c_unit = col_idx(ltags, "axis_unit");
                    let c_code = col_idx(ltags, "axis_code");
                    let c_freq = col_idx(ltags, "spectrometer_frequency");
                    let c_sw = col_idx(ltags, "spectral_width");
                    let c_vfp = col_idx(ltags, "value_first_point");
                    let c_fold = col_idx(ltags, "folding");
                    let c_abs = col_idx(ltags, "absolute_peak_positions");
                    let c_acq = col_idx(ltags, "is_acquisition");

                    for (idx, r) in rows.iter().enumerate() {
                        dims.push(NefSpectrumDimension {
                            dimension_id: c_id.and_then(|c| opt_usize(&r[c])).unwrap_or(idx + 1),
                            axis_unit: c_unit
                                .and_then(|c| opt_str(&r[c]))
                                .unwrap_or_else(|| "ppm".to_string()),
                            axis_code: c_code
                                .and_then(|c| opt_str(&r[c]))
                                .unwrap_or_else(|| "1H".to_string()),
                            spectrometer_frequency: c_freq.and_then(|c| opt_f64(&r[c])),
                            spectral_width: c_sw.and_then(|c| opt_f64(&r[c])),
                            value_first_point: c_vfp.and_then(|c| opt_f64(&r[c])),
                            folding: c_fold
                                .and_then(|c| opt_str(&r[c]))
                                .unwrap_or_else(|| "circular".to_string()),
                            absolute_peak_positions: c_abs
                                .and_then(|c| opt_bool(&r[c]))
                                .unwrap_or(true),
                            is_acquisition: c_acq.and_then(|c| opt_bool(&r[c])).unwrap_or(idx == 0),
                        });
                    }
                } else if ltags.iter().any(|t| t.starts_with("_nef_peak.")) {
                    let c_idx = col_idx(ltags, "index");
                    let c_pid = col_idx(ltags, "peak_id");
                    let c_ht = col_idx(ltags, "height");
                    let c_htu = col_idx(ltags, "height_uncertainty");
                    let c_vol = col_idx(ltags, "volume");
                    let c_volu = col_idx(ltags, "volume_uncertainty");
                    let c_det = col_idx(ltags, "details");

                    // Detect actual dimension count from position_d columns
                    let mut max_d = num_dim;
                    for d in 1..=8 {
                        if col_idx(ltags, &format!("position_{d}")).is_some() {
                            max_d = max_d.max(d);
                        }
                    }

                    for (row_i, r) in rows.iter().enumerate() {
                        let mut positions = Vec::with_capacity(max_d);
                        let mut pos_unc = Vec::with_capacity(max_d);
                        let mut chains = Vec::with_capacity(max_d);
                        let mut seqs = Vec::with_capacity(max_d);
                        let mut res_names = Vec::with_capacity(max_d);
                        let mut atom_names = Vec::with_capacity(max_d);

                        let mut valid_positions = true;
                        for d in 1..=max_d {
                            if let Some(cp) = col_idx(ltags, &format!("position_{d}")) {
                                if let Some(val) = opt_f64(&r[cp]) {
                                    positions.push(val);
                                } else {
                                    valid_positions = false;
                                    break;
                                }
                            } else {
                                break;
                            }
                            pos_unc.push(
                                col_idx(ltags, &format!("position_uncertainty_{d}"))
                                    .and_then(|c| opt_f64(&r[c])),
                            );
                            chains.push(
                                col_idx(ltags, &format!("chain_code_{d}"))
                                    .and_then(|c| opt_str(&r[c])),
                            );
                            seqs.push(
                                col_idx(ltags, &format!("sequence_code_{d}"))
                                    .and_then(|c| opt_i32(&r[c])),
                            );
                            res_names.push(
                                col_idx(ltags, &format!("residue_name_{d}"))
                                    .and_then(|c| opt_str(&r[c])),
                            );
                            atom_names.push(
                                col_idx(ltags, &format!("atom_name_{d}"))
                                    .and_then(|c| opt_str(&r[c])),
                            );
                        }
                        if !valid_positions || positions.is_empty() {
                            continue;
                        }

                        peaks.push(NefPeak {
                            index: c_idx.and_then(|c| opt_usize(&r[c])).unwrap_or(row_i + 1),
                            peak_id: c_pid.and_then(|c| opt_usize(&r[c])).unwrap_or(row_i + 1),
                            positions,
                            position_uncertainties: pos_unc,
                            height: c_ht.and_then(|c| opt_f64(&r[c])),
                            height_uncertainty: c_htu.and_then(|c| opt_f64(&r[c])),
                            volume: c_vol.and_then(|c| opt_f64(&r[c])),
                            volume_uncertainty: c_volu.and_then(|c| opt_f64(&r[c])),
                            chain_codes: chains,
                            sequence_codes: seqs,
                            residue_names: res_names,
                            atom_names,
                            details: c_det.and_then(|c| opt_str(&r[c])),
                        });
                    }
                }
            }

            doc.peak_lists.push(NefPeakList {
                spectrum_name: spec_name,
                experiment_classification: exp_class,
                experiment_type: exp_type,
                chemical_shift_list: cs_list,
                dimensions: dims,
                peaks,
            });
        } else if cat == "nef_distance_restraint_list"
            || sf
                .name
                .to_ascii_lowercase()
                .starts_with("nef_distance_restraint_list")
        {
            let name = sf
                .name
                .strip_prefix("nef_distance_restraint_list_")
                .unwrap_or(&sf.name)
                .to_string();
            let pot = sf
                .tags
                .get("_nef_distance_restraint_list.potential_type")
                .cloned()
                .unwrap_or_else(|| "upper-bound-parabolic".to_string());
            let mut restraints = Vec::new();

            for (ltags, rows) in &sf.loops {
                let Some(c_rid) = col_idx(ltags, "restraint_id") else {
                    continue;
                };
                let Some(c_up) = col_idx(ltags, "upper_limit") else {
                    continue;
                };
                let c_idx = col_idx(ltags, "index");
                let c_rcid = col_idx(ltags, "restraint_combination_id");
                let c_ch1 = col_idx(ltags, "chain_code_1");
                let c_sq1 = col_idx(ltags, "sequence_code_1");
                let c_rs1 = col_idx(ltags, "residue_name_1");
                let c_at1 = col_idx(ltags, "atom_name_1");
                let c_ch2 = col_idx(ltags, "chain_code_2");
                let c_sq2 = col_idx(ltags, "sequence_code_2");
                let c_rs2 = col_idx(ltags, "residue_name_2");
                let c_at2 = col_idx(ltags, "atom_name_2");
                let c_wt = col_idx(ltags, "weight");
                let c_tv = col_idx(ltags, "target_value");
                let c_tvu = col_idx(ltags, "target_value_uncertainty");
                let c_lll = col_idx(ltags, "lower_linear_limit");
                let c_ll = col_idx(ltags, "lower_limit");
                let c_ull = col_idx(ltags, "upper_linear_limit");

                for (row_i, r) in rows.iter().enumerate() {
                    let Some(rid) = opt_usize(&r[c_rid]) else {
                        continue;
                    };
                    let Some(upper) = opt_f64(&r[c_up]) else {
                        continue;
                    };
                    restraints.push(NefDistanceRestraint {
                        index: c_idx.and_then(|c| opt_usize(&r[c])).unwrap_or(row_i + 1),
                        restraint_id: rid,
                        restraint_combination_id: c_rcid.and_then(|c| opt_usize(&r[c])),
                        chain_code_1: c_ch1
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "A".to_string()),
                        sequence_code_1: c_sq1.and_then(|c| opt_i32(&r[c])).unwrap_or(1),
                        residue_name_1: c_rs1
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "UNK".to_string()),
                        atom_name_1: c_at1
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "H".to_string()),
                        chain_code_2: c_ch2
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "A".to_string()),
                        sequence_code_2: c_sq2.and_then(|c| opt_i32(&r[c])).unwrap_or(1),
                        residue_name_2: c_rs2
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "UNK".to_string()),
                        atom_name_2: c_at2
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "H".to_string()),
                        weight: c_wt.and_then(|c| opt_f64(&r[c])).unwrap_or(1.0),
                        target_value: c_tv.and_then(|c| opt_f64(&r[c])),
                        target_value_uncertainty: c_tvu.and_then(|c| opt_f64(&r[c])),
                        lower_linear_limit: c_lll.and_then(|c| opt_f64(&r[c])),
                        lower_limit: c_ll.and_then(|c| opt_f64(&r[c])),
                        upper_limit: upper,
                        upper_linear_limit: c_ull.and_then(|c| opt_f64(&r[c])),
                    });
                }
            }

            doc.distance_restraint_lists.push(NefDistanceRestraintList {
                name,
                potential_type: pot,
                restraints,
            });
        } else if cat == "nef_dihedral_restraint_list"
            || sf
                .name
                .to_ascii_lowercase()
                .starts_with("nef_dihedral_restraint_list")
        {
            let name = sf
                .name
                .strip_prefix("nef_dihedral_restraint_list_")
                .unwrap_or(&sf.name)
                .to_string();
            let pot = sf
                .tags
                .get("_nef_dihedral_restraint_list.potential_type")
                .cloned()
                .unwrap_or_else(|| "square-well-parabolic".to_string());
            let mut restraints = Vec::new();

            for (ltags, rows) in &sf.loops {
                let Some(c_rid) = col_idx(ltags, "restraint_id") else {
                    continue;
                };
                let c_idx = col_idx(ltags, "index");
                let c_rcid = col_idx(ltags, "restraint_combination_id");
                let c_wt = col_idx(ltags, "weight");
                let c_tv = col_idx(ltags, "target_value");
                let c_tvu = col_idx(ltags, "target_value_uncertainty");
                let c_lll = col_idx(ltags, "lower_linear_limit");
                let c_ll = col_idx(ltags, "lower_limit");
                let c_ul = col_idx(ltags, "upper_limit");
                let c_ull = col_idx(ltags, "upper_linear_limit");
                let c_name = col_idx(ltags, "name");

                for (row_i, r) in rows.iter().enumerate() {
                    let Some(rid) = opt_usize(&r[c_rid]) else {
                        continue;
                    };
                    let get_atom = |k: usize| -> (String, i32, String, String) {
                        let ch = col_idx(ltags, &format!("chain_code_{k}"))
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "A".to_string());
                        let sq = col_idx(ltags, &format!("sequence_code_{k}"))
                            .and_then(|c| opt_i32(&r[c]))
                            .unwrap_or(1);
                        let rs = col_idx(ltags, &format!("residue_name_{k}"))
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "UNK".to_string());
                        let at = col_idx(ltags, &format!("atom_name_{k}"))
                            .map(|c| r[c].clone())
                            .unwrap_or_else(|| "CA".to_string());
                        (ch, sq, rs, at)
                    };
                    let (ch1, sq1, rs1, at1) = get_atom(1);
                    let (ch2, sq2, rs2, at2) = get_atom(2);
                    let (ch3, sq3, rs3, at3) = get_atom(3);
                    let (ch4, sq4, rs4, at4) = get_atom(4);

                    let target = c_tv.and_then(|c| opt_f64(&r[c])).unwrap_or(0.0);
                    let lower = c_ll.and_then(|c| opt_f64(&r[c])).unwrap_or(target - 20.0);
                    let upper = c_ul.and_then(|c| opt_f64(&r[c])).unwrap_or(target + 20.0);

                    restraints.push(NefDihedralRestraint {
                        index: c_idx.and_then(|c| opt_usize(&r[c])).unwrap_or(row_i + 1),
                        restraint_id: rid,
                        restraint_combination_id: c_rcid.and_then(|c| opt_usize(&r[c])),
                        chain_code_1: ch1,
                        sequence_code_1: sq1,
                        residue_name_1: rs1,
                        atom_name_1: at1,
                        chain_code_2: ch2,
                        sequence_code_2: sq2,
                        residue_name_2: rs2,
                        atom_name_2: at2,
                        chain_code_3: ch3,
                        sequence_code_3: sq3,
                        residue_name_3: rs3,
                        atom_name_3: at3,
                        chain_code_4: ch4,
                        sequence_code_4: sq4,
                        residue_name_4: rs4,
                        atom_name_4: at4,
                        weight: c_wt.and_then(|c| opt_f64(&r[c])).unwrap_or(1.0),
                        target_value: target,
                        target_value_uncertainty: c_tvu.and_then(|c| opt_f64(&r[c])),
                        lower_linear_limit: c_lll.and_then(|c| opt_f64(&r[c])),
                        lower_limit: lower,
                        upper_limit: upper,
                        upper_linear_limit: c_ull.and_then(|c| opt_f64(&r[c])),
                        name: c_name
                            .and_then(|c| opt_str(&r[c]))
                            .unwrap_or_else(|| "PHI".to_string()),
                    });
                }
            }

            doc.dihedral_restraint_lists.push(NefDihedralRestraintList {
                name,
                potential_type: pot,
                restraints,
            });
        } else if cat == "nmrassign_workbench"
            || cat == "nmrassign_workbench_session"
            || sf.name.eq_ignore_ascii_case("nmrassign_workbench")
            || sf
                .name
                .to_ascii_lowercase()
                .starts_with("nmrassign_workbench")
        {
            doc.workbench_session = Some(parse_workbench_saveframe(&sf)?);
        }
    }

    Ok(doc)
}

fn parse_exact_f64_tag(
    tags: &BTreeMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64, String> {
    match tags.get(key) {
        Some(v) if v != "." && v != "?" && !v.is_empty() => v
            .parse::<f64>()
            .map_err(|e| format!("invalid float for tag '{key}' ('{v}'): {e}")),
        _ => Ok(default),
    }
}

fn parse_opt_exact_f64_cell(raw: &str, label: &str) -> Result<Option<f64>, String> {
    if raw.is_empty() || raw == "." || raw == "?" {
        Ok(None)
    } else {
        raw.parse::<f64>()
            .map(Some)
            .map_err(|e| format!("invalid float for {label} ('{raw}'): {e}"))
    }
}

fn parse_workbench_saveframe(sf: &Saveframe) -> Result<NefWorkbenchSaveframe, String> {
    let schema_version = sf
        .tags
        .get("_nmrassign_workbench.schema_version")
        .cloned()
        .unwrap_or_else(|| "1.1".to_string());
    let protein_name = match sf.tags.get("_nmrassign_workbench.protein_name") {
        Some(v) => decode_lossless_star_token(v)?,
        None => String::new(),
    };
    let chain_code = match sf.tags.get("_nmrassign_workbench.chain_code") {
        Some(v) => decode_lossless_star_token(v)?,
        None => "A".to_string(),
    };
    let reference_offsets = [
        parse_exact_f64_tag(&sf.tags, "_nmrassign_workbench.ref_offset_h", 0.0)?,
        parse_exact_f64_tag(&sf.tags, "_nmrassign_workbench.ref_offset_n", 0.0)?,
        parse_exact_f64_tag(&sf.tags, "_nmrassign_workbench.ref_offset_ca", 0.0)?,
        parse_exact_f64_tag(&sf.tags, "_nmrassign_workbench.ref_offset_cb", 0.0)?,
        parse_exact_f64_tag(&sf.tags, "_nmrassign_workbench.ref_offset_c", 0.0)?,
    ];
    let qc_verdict = match sf.tags.get("_nmrassign_workbench.qc_verdict") {
        Some(v) if v != "." && v != "?" && !v.is_empty() => Some(decode_lossless_star_token(v)?),
        _ => None,
    };
    let shift_list_crc32 = match sf.tags.get("_nmrassign_workbench.shift_list_crc32") {
        Some(v) if v != "." && v != "?" && !v.is_empty() => Some(decode_lossless_star_token(v)?),
        _ => None,
    };
    let view_bounds_ppm = match (
        sf.tags
            .get("_nmrassign_workbench.view_h_left")
            .and_then(|s| opt_str(s)),
        sf.tags
            .get("_nmrassign_workbench.view_h_right")
            .and_then(|s| opt_str(s)),
        sf.tags
            .get("_nmrassign_workbench.view_n_top")
            .and_then(|s| opt_str(s)),
        sf.tags
            .get("_nmrassign_workbench.view_n_bottom")
            .and_then(|s| opt_str(s)),
    ) {
        (Some(hl), Some(hr), Some(nt), Some(nb)) => Some([
            hl.parse::<f64>()
                .map_err(|e| format!("invalid view_h_left: {e}"))?,
            hr.parse::<f64>()
                .map_err(|e| format!("invalid view_h_right: {e}"))?,
            nt.parse::<f64>()
                .map_err(|e| format!("invalid view_n_top: {e}"))?,
            nb.parse::<f64>()
                .map_err(|e| format!("invalid view_n_bottom: {e}"))?,
        ]),
        _ => None,
    };

    let mut residues = Vec::new();
    let mut peaks = Vec::new();
    let mut assignments = Vec::new();
    let mut assignment_shifts = Vec::new();
    let mut locks = Vec::new();
    let mut confirmed_links = Vec::new();
    let mut contour_layers = Vec::new();
    let mut processing_recipes = Vec::new();

    for (ltags, rows) in &sf.loops {
        if ltags.iter().any(|t| t.starts_with("_nmrassign_residue.")) {
            let c_seq = col_idx(ltags, "seq_id")
                .ok_or_else(|| "missing _nmrassign_residue.seq_id".to_string())?;
            let c_res = col_idx(ltags, "residue_name")
                .ok_or_else(|| "missing _nmrassign_residue.residue_name".to_string())?;
            for r in rows {
                let seq_id = r[c_seq]
                    .parse::<i32>()
                    .map_err(|e| format!("invalid _nmrassign_residue.seq_id: {e}"))?;
                let residue_name = decode_lossless_star_token(&r[c_res])?;
                residues.push(NefWorkbenchResidueRow {
                    seq_id,
                    residue_name,
                });
            }
        } else if ltags.iter().any(|t| t.starts_with("_nmrassign_peak.")) {
            let c_exp = col_idx(ltags, "experiment")
                .ok_or_else(|| "missing _nmrassign_peak.experiment".to_string())?;
            let c_h = col_idx(ltags, "h").ok_or_else(|| "missing _nmrassign_peak.h".to_string())?;
            let c_n = col_idx(ltags, "n").ok_or_else(|| "missing _nmrassign_peak.n".to_string())?;
            let c_c = col_idx(ltags, "c");
            let c_sgn = col_idx(ltags, "sign");
            let c_ht = col_idx(ltags, "height");
            let c_vol = col_idx(ltags, "volume");
            let c_h2 = col_idx(ltags, "h2");
            for r in rows {
                let experiment = decode_lossless_star_token(&r[c_exp])?;
                let h = r[c_h]
                    .parse::<f64>()
                    .map_err(|e| format!("invalid _nmrassign_peak.h: {e}"))?;
                let n = r[c_n]
                    .parse::<f64>()
                    .map_err(|e| format!("invalid _nmrassign_peak.n: {e}"))?;
                let c = match c_c {
                    Some(col) => parse_opt_exact_f64_cell(&r[col], "_nmrassign_peak.c")?,
                    None => None,
                };
                let sign = match c_sgn {
                    Some(col) => r[col]
                        .parse::<i32>()
                        .map_err(|e| format!("invalid _nmrassign_peak.sign: {e}"))?,
                    None => 1,
                };
                let height = match c_ht {
                    Some(col) => parse_opt_exact_f64_cell(&r[col], "_nmrassign_peak.height")?,
                    None => None,
                };
                let volume = match c_vol {
                    Some(col) => parse_opt_exact_f64_cell(&r[col], "_nmrassign_peak.volume")?,
                    None => None,
                };
                let h2 = match c_h2 {
                    Some(col) => parse_opt_exact_f64_cell(&r[col], "_nmrassign_peak.h2")?,
                    None => None,
                };
                peaks.push(NefWorkbenchPeakRow {
                    experiment,
                    h,
                    n,
                    c,
                    sign,
                    height,
                    volume,
                    h2,
                });
            }
        } else if ltags
            .iter()
            .any(|t| t.starts_with("_nmrassign_assignment."))
        {
            let c_res = col_idx(ltags, "residue_seq_idx")
                .ok_or_else(|| "missing _nmrassign_assignment.residue_seq_idx".to_string())?;
            let c_ss = col_idx(ltags, "spin_system_id");
            let c_conf = col_idx(ltags, "confidence")
                .ok_or_else(|| "missing _nmrassign_assignment.confidence".to_string())?;
            for r in rows {
                let residue_seq_idx = r[c_res]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid _nmrassign_assignment.residue_seq_idx: {e}"))?;
                let spin_system_id = c_ss.and_then(|c| opt_usize(&r[c]));
                let confidence = r[c_conf]
                    .parse::<f64>()
                    .map_err(|e| format!("invalid _nmrassign_assignment.confidence: {e}"))?;
                assignments.push(NefWorkbenchAssignmentRow {
                    residue_seq_idx,
                    spin_system_id,
                    confidence,
                });
            }
        } else if ltags
            .iter()
            .any(|t| t.starts_with("_nmrassign_assignment_shift."))
        {
            let c_res = col_idx(ltags, "residue_seq_idx")
                .ok_or_else(|| "missing _nmrassign_assignment_shift.residue_seq_idx".to_string())?;
            let c_atom = col_idx(ltags, "atom_name")
                .ok_or_else(|| "missing _nmrassign_assignment_shift.atom_name".to_string())?;
            let c_val = col_idx(ltags, "value")
                .ok_or_else(|| "missing _nmrassign_assignment_shift.value".to_string())?;
            for r in rows {
                let residue_seq_idx = r[c_res].parse::<usize>().map_err(|e| {
                    format!("invalid _nmrassign_assignment_shift.residue_seq_idx: {e}")
                })?;
                let atom_name = decode_lossless_star_token(&r[c_atom])?;
                let value = r[c_val]
                    .parse::<f64>()
                    .map_err(|e| format!("invalid _nmrassign_assignment_shift.value: {e}"))?;
                assignment_shifts.push(NefWorkbenchShiftRow {
                    residue_seq_idx,
                    atom_name,
                    value,
                });
            }
        } else if ltags.iter().any(|t| t.starts_with("_nmrassign_lock.")) {
            let c_ss = col_idx(ltags, "spin_system_id")
                .ok_or_else(|| "missing _nmrassign_lock.spin_system_id".to_string())?;
            let c_res = col_idx(ltags, "residue_seq_idx")
                .ok_or_else(|| "missing _nmrassign_lock.residue_seq_idx".to_string())?;
            for r in rows {
                let spin_system_id = r[c_ss]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid _nmrassign_lock.spin_system_id: {e}"))?;
                let residue_seq_idx = r[c_res]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid _nmrassign_lock.residue_seq_idx: {e}"))?;
                locks.push(NefWorkbenchLockRow {
                    spin_system_id,
                    residue_seq_idx,
                });
            }
        } else if ltags
            .iter()
            .any(|t| t.starts_with("_nmrassign_confirmed_link."))
        {
            let c_pred = col_idx(ltags, "pred_spin_system")
                .ok_or_else(|| "missing _nmrassign_confirmed_link.pred_spin_system".to_string())?;
            let c_succ = col_idx(ltags, "succ_spin_system")
                .ok_or_else(|| "missing _nmrassign_confirmed_link.succ_spin_system".to_string())?;
            for r in rows {
                let pred_spin_system = r[c_pred].parse::<usize>().map_err(|e| {
                    format!("invalid _nmrassign_confirmed_link.pred_spin_system: {e}")
                })?;
                let succ_spin_system = r[c_succ].parse::<usize>().map_err(|e| {
                    format!("invalid _nmrassign_confirmed_link.succ_spin_system: {e}")
                })?;
                confirmed_links.push(NefWorkbenchLinkRow {
                    pred_spin_system,
                    succ_spin_system,
                });
            }
        } else if ltags
            .iter()
            .any(|t| t.starts_with("_nmrassign_contour_layer."))
        {
            let req = |name: &str| -> Result<usize, String> {
                col_idx(ltags, name)
                    .ok_or_else(|| format!("missing _nmrassign_contour_layer.{name}"))
            };
            let c_lid = req("layer_id")?;
            let c_vis = req("visible")?;
            let c_base = req("base_level")?;
            let c_fac = req("level_factor")?;
            let c_num = req("num_levels")?;
            let c_neg = req("show_negative")?;
            let c_pr = req("pos_r")?;
            let c_pg = req("pos_g")?;
            let c_pb = req("pos_b")?;
            let c_pa = req("pos_a")?;
            let c_nr = req("neg_r")?;
            let c_ng = req("neg_g")?;
            let c_nb = req("neg_b")?;
            let c_na = req("neg_a")?;
            let c_nx = req("ppm_nudge_x")?;
            let c_ny = req("ppm_nudge_y")?;

            let pf32 = |s: &str, name: &str| -> Result<f32, String> {
                s.parse::<f32>()
                    .map_err(|e| format!("invalid _nmrassign_contour_layer.{name} ('{s}'): {e}"))
            };
            for r in rows {
                contour_layers.push(NefWorkbenchContourLayerRow {
                    layer_id: decode_lossless_star_token(&r[c_lid])?,
                    visible: opt_bool(&r[c_vis]).unwrap_or(true),
                    base_level: pf32(&r[c_base], "base_level")?,
                    level_factor: pf32(&r[c_fac], "level_factor")?,
                    num_levels: r[c_num]
                        .parse::<u32>()
                        .map_err(|e| format!("invalid num_levels: {e}"))?,
                    show_negative: opt_bool(&r[c_neg]).unwrap_or(true),
                    pos_rgba: [
                        pf32(&r[c_pr], "pos_r")?,
                        pf32(&r[c_pg], "pos_g")?,
                        pf32(&r[c_pb], "pos_b")?,
                        pf32(&r[c_pa], "pos_a")?,
                    ],
                    neg_rgba: [
                        pf32(&r[c_nr], "neg_r")?,
                        pf32(&r[c_ng], "neg_g")?,
                        pf32(&r[c_nb], "neg_b")?,
                        pf32(&r[c_na], "neg_a")?,
                    ],
                    ppm_nudge: [
                        pf32(&r[c_nx], "ppm_nudge_x")?,
                        pf32(&r[c_ny], "ppm_nudge_y")?,
                    ],
                });
            }
        } else if ltags
            .iter()
            .any(|t| t.starts_with("_nmrassign_processing_recipe."))
        {
            let c_sid = col_idx(ltags, "spectrum_id")
                .ok_or_else(|| "missing _nmrassign_processing_recipe.spectrum_id".to_string())?;
            let c_json = col_idx(ltags, "recipe_json")
                .ok_or_else(|| "missing _nmrassign_processing_recipe.recipe_json".to_string())?;
            for r in rows {
                let spectrum_id = decode_lossless_star_token(&r[c_sid])?;
                let recipe_json = decode_lossless_star_token(&r[c_json])?;
                processing_recipes.push(NefWorkbenchRecipeRow {
                    spectrum_id,
                    recipe_json,
                });
            }
        }
    }

    Ok(NefWorkbenchSaveframe {
        schema_version,
        protein_name,
        chain_code,
        reference_offsets,
        qc_verdict,
        view_bounds_ppm,
        shift_list_crc32,
        residues,
        peaks,
        assignments,
        assignment_shifts,
        locks,
        confirmed_links,
        contour_layers,
        processing_recipes,
    })
}

/// Compute an 8-character lowercase hex IEEE 802.3 CRC-32 digest over the canonicalized rows
/// of `nef_chemical_shift_list` saveframes so external edits to standard NEF shift tables can
/// be detected and adopted when reloading a `save_nmrassign_workbench` session.
pub fn compute_nef_shift_lists_crc32(shift_lists: &[NefShiftList]) -> String {
    let mut crc = 0xFFFF_FFFFu32;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            crc ^= b as u32;
            for _ in 0..8 {
                if (crc & 1) != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
    };

    for slist in shift_lists {
        let sname = sanitize_identifier(&slist.name);
        feed(b"list:");
        feed(sname.as_bytes());
        feed(b"\n");
        for sh in &slist.shifts {
            let norm_val = if sh.value == 0.0 { 0.0 } else { sh.value };
            let seq_s = sh
                .sequence_code
                .map(|n| n.to_string())
                .unwrap_or_else(|| ".".to_string());
            let row = format!(
                "{}\t{}\t{}\t{}\t{norm_val:.4}\n",
                sh.chain_code, seq_s, sh.residue_name, sh.atom_name
            );
            feed(row.as_bytes());
        }
    }
    format!("{:08x}", !crc)
}

// ============================================================================
// Canonical NEF v1.1 Writer
// ============================================================================

fn sanitize_identifier(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = s.trim_matches('_');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

fn can_use_multiline_star_block(s: &str) -> bool {
    !s.is_empty()
        && !s.contains("\n;")
        && !s.contains('\r')
        && !s.chars().any(|c| c.is_control() && c != '\n' && c != '\t')
}

fn fmt_multiline_star(s: &str) -> String {
    if s.starts_with('\n') {
        format!("\n;\n{s}\n;\n")
    } else {
        format!("\n;{s}\n;\n")
    }
}

fn fmt_val(s: &str) -> String {
    if s.is_empty() {
        return ".".to_string();
    }
    let needs_multiline = (s.contains('\'') && s.contains('"')) || s.contains('\n') || s.len() > 80;
    if needs_multiline && can_use_multiline_star_block(s) {
        return fmt_multiline_star(s);
    }
    let needs_quotes = s.chars().any(char::is_whitespace)
        || s.contains('#')
        || s.contains('\'')
        || s.contains('"')
        || s.starts_with('_')
        || s.starts_with(';')
        || s.eq_ignore_ascii_case("loop_")
        || s.eq_ignore_ascii_case("stop_")
        || s.eq_ignore_ascii_case("save_")
        || s.to_ascii_lowercase().starts_with("data_")
        || s.to_ascii_lowercase().starts_with("save_");
    if needs_quotes {
        if !s.contains('\'') {
            format!("'{s}'")
        } else if !s.contains('"') {
            format!("\"{s}\"")
        } else {
            let sanitized = s.replace("\n;", "\n ;").replace('\r', "");
            fmt_multiline_star(&sanitized)
        }
    } else {
        s.to_string()
    }
}

fn fmt_opt_str(v: Option<&str>) -> String {
    match v {
        Some(s) if !s.is_empty() => fmt_val(s),
        _ => ".".to_string(),
    }
}

fn fmt_opt_f64(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(x) => format!("{x:.decimals$}"),
        None => ".".to_string(),
    }
}

fn fmt_opt_i32(v: Option<i32>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => ".".to_string(),
    }
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => ".".to_string(),
    }
}

fn fmt_opt_bool(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "true",
        Some(false) => "false",
        None => ".",
    }
}

/// Write a NEF v1.1 document to a file on disk.
pub fn write_nef_file(path: impl AsRef<Path>, doc: &NefDocument) -> Result<(), String> {
    let text = write_nef_str(doc);
    fs::write(path.as_ref(), text)
        .map_err(|e| format!("failed to write NEF file {}: {e}", path.as_ref().display()))
}

/// Serialize a `NefDocument` into canonical NEF v1.1 text.
pub fn write_nef_str(doc: &NefDocument) -> String {
    let mut out = String::with_capacity(8192);
    let proj = sanitize_identifier(&doc.project_name);
    out.push_str(&format!("data_nef_{proj}\n\n"));

    // 1. save_nef_nmr_meta_data
    out.push_str("save_nef_nmr_meta_data\n");
    out.push_str("   _nef_nmr_meta_data.sf_category           nef_nmr_meta_data\n");
    out.push_str("   _nef_nmr_meta_data.sf_framecode          nef_nmr_meta_data\n");
    out.push_str(&format!(
        "   _nef_nmr_meta_data.format_name           {}\n",
        fmt_val(&doc.meta_data.format_name)
    ));
    out.push_str(&format!(
        "   _nef_nmr_meta_data.format_version        {}\n",
        fmt_val(&doc.meta_data.format_version)
    ));
    out.push_str(&format!(
        "   _nef_nmr_meta_data.program_name          {}\n",
        fmt_val(&doc.meta_data.program_name)
    ));
    out.push_str(&format!(
        "   _nef_nmr_meta_data.program_version       {}\n",
        fmt_val(&doc.meta_data.program_version)
    ));
    out.push_str(&format!(
        "   _nef_nmr_meta_data.creation_date         {}\n",
        fmt_val(&doc.meta_data.creation_date)
    ));
    out.push_str(&format!(
        "   _nef_nmr_meta_data.uuid                  {}\n",
        fmt_val(&doc.meta_data.uuid)
    ));
    if let Some(ref coord) = doc.meta_data.coordinate_file_name {
        out.push_str(&format!(
            "   _nef_nmr_meta_data.coordinate_file_name  {}\n",
            fmt_val(coord)
        ));
    }
    out.push_str("save_\n\n");

    // 2. save_nef_molecular_system
    out.push_str("save_nef_molecular_system\n");
    out.push_str("   _nef_molecular_system.sf_category   nef_molecular_system\n");
    out.push_str("   _nef_molecular_system.sf_framecode  nef_molecular_system\n\n");
    if !doc.molecular_system.residues.is_empty() {
        out.push_str("   loop_\n");
        for tag in [
            "_nef_sequence.index",
            "_nef_sequence.chain_code",
            "_nef_sequence.sequence_code",
            "_nef_sequence.residue_name",
            "_nef_sequence.linking",
            "_nef_sequence.residue_variant",
            "_nef_sequence.cis_peptide",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, r) in doc.molecular_system.residues.iter().enumerate() {
            out.push_str(&format!(
                "      {:5} {:4} {:5} {:5} {:8} {:6} {}\n",
                i + 1,
                fmt_val(&r.chain_code),
                r.sequence_code,
                fmt_val(&r.residue_name),
                fmt_val(&r.linking),
                fmt_opt_str(r.residue_variant.as_deref()),
                fmt_opt_bool(r.cis_peptide)
            ));
        }
        out.push_str("   stop_\n");
    }
    out.push_str("save_\n\n");

    // 3. save_nef_chemical_shift_list_<name>
    for slist in &doc.shift_lists {
        let sname = sanitize_identifier(&slist.name);
        let frame = format!("nef_chemical_shift_list_{sname}");
        out.push_str(&format!("save_{frame}\n"));
        out.push_str(
            "   _nef_chemical_shift_list.sf_category            nef_chemical_shift_list\n",
        );
        out.push_str(&format!(
            "   _nef_chemical_shift_list.sf_framecode           {frame}\n"
        ));
        out.push_str(&format!(
            "   _nef_chemical_shift_list.atom_chem_shift_units  {}\n\n",
            fmt_val(&slist.atom_chem_shift_units)
        ));
        if !slist.shifts.is_empty() {
            out.push_str("   loop_\n");
            for tag in [
                "_nef_chemical_shift.chain_code",
                "_nef_chemical_shift.sequence_code",
                "_nef_chemical_shift.residue_name",
                "_nef_chemical_shift.atom_name",
                "_nef_chemical_shift.element",
                "_nef_chemical_shift.isotope_number",
                "_nef_chemical_shift.value",
                "_nef_chemical_shift.value_uncertainty",
            ] {
                out.push_str(&format!("      {tag}\n"));
            }
            out.push('\n');
            for sh in &slist.shifts {
                let iso_str = sh
                    .isotope_number
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| ".".to_string());
                out.push_str(&format!(
                    "      {:4} {:5} {:5} {:6} {:3} {:3} {:9.4} {}\n",
                    fmt_val(&sh.chain_code),
                    fmt_opt_i32(sh.sequence_code),
                    fmt_val(&sh.residue_name),
                    fmt_val(&sh.atom_name),
                    fmt_opt_str(sh.element.as_deref()),
                    iso_str,
                    sh.value,
                    fmt_opt_f64(sh.value_uncertainty, 4)
                ));
            }
            out.push_str("   stop_\n");
        }
        out.push_str("save_\n\n");
    }

    // 4. save_nef_nmr_spectrum_<exp>
    for spec in &doc.peak_lists {
        let exp_id = sanitize_identifier(&spec.spectrum_name);
        let frame = format!("nef_nmr_spectrum_{exp_id}");
        let ndim = spec
            .dimensions
            .len()
            .max(spec.peaks.first().map(|p| p.positions.len()).unwrap_or(2));
        out.push_str(&format!("save_{frame}\n"));
        out.push_str("   _nef_nmr_spectrum.sf_category                nef_nmr_spectrum\n");
        out.push_str(&format!(
            "   _nef_nmr_spectrum.sf_framecode               {frame}\n"
        ));
        out.push_str(&format!(
            "   _nef_nmr_spectrum.num_dimensions             {ndim}\n"
        ));
        out.push_str(&format!(
            "   _nef_nmr_spectrum.chemical_shift_list        {}\n",
            fmt_val(&spec.chemical_shift_list)
        ));
        out.push_str(&format!(
            "   _nef_nmr_spectrum.experiment_classification  {}\n",
            fmt_opt_str(spec.experiment_classification.as_deref())
        ));
        out.push_str(&format!(
            "   _nef_nmr_spectrum.experiment_type            {}\n\n",
            fmt_opt_str(spec.experiment_type.as_deref())
        ));

        if !spec.dimensions.is_empty() {
            out.push_str("   loop_\n");
            for tag in [
                "_nef_spectrum_dimension.dimension_id",
                "_nef_spectrum_dimension.axis_unit",
                "_nef_spectrum_dimension.axis_code",
                "_nef_spectrum_dimension.spectrometer_frequency",
                "_nef_spectrum_dimension.spectral_width",
                "_nef_spectrum_dimension.value_first_point",
                "_nef_spectrum_dimension.folding",
                "_nef_spectrum_dimension.absolute_peak_positions",
                "_nef_spectrum_dimension.is_acquisition",
            ] {
                out.push_str(&format!("      {tag}\n"));
            }
            out.push('\n');
            for d in &spec.dimensions {
                out.push_str(&format!(
                    "      {:2} {:5} {:5} {:8} {:8} {:8} {:9} {:5} {}\n",
                    d.dimension_id,
                    fmt_val(&d.axis_unit),
                    fmt_val(&d.axis_code),
                    fmt_opt_f64(d.spectrometer_frequency, 2),
                    fmt_opt_f64(d.spectral_width, 2),
                    fmt_opt_f64(d.value_first_point, 2),
                    fmt_val(&d.folding),
                    fmt_opt_bool(Some(d.absolute_peak_positions)),
                    fmt_opt_bool(Some(d.is_acquisition)),
                ));
            }
            out.push_str("   stop_\n\n");
        }

        if !spec.peaks.is_empty() {
            out.push_str("   loop_\n");
            for tag in [
                "_nef_peak.index",
                "_nef_peak.peak_id",
                "_nef_peak.volume",
                "_nef_peak.volume_uncertainty",
                "_nef_peak.height",
                "_nef_peak.height_uncertainty",
            ] {
                out.push_str(&format!("      {tag}\n"));
            }
            for d in 1..=ndim {
                out.push_str(&format!("      _nef_peak.position_{d}\n"));
                out.push_str(&format!("      _nef_peak.position_uncertainty_{d}\n"));
            }
            for d in 1..=ndim {
                out.push_str(&format!("      _nef_peak.chain_code_{d}\n"));
                out.push_str(&format!("      _nef_peak.sequence_code_{d}\n"));
                out.push_str(&format!("      _nef_peak.residue_name_{d}\n"));
                out.push_str(&format!("      _nef_peak.atom_name_{d}\n"));
            }
            out.push_str("      _nef_peak.details\n\n");

            for pk in &spec.peaks {
                let mut row = format!(
                    "      {:5} {:5} {:10} {:6} {:10} {:6}",
                    pk.index,
                    pk.peak_id,
                    fmt_opt_f64(pk.volume, 3),
                    fmt_opt_f64(pk.volume_uncertainty, 3),
                    fmt_opt_f64(pk.height, 3),
                    fmt_opt_f64(pk.height_uncertainty, 3),
                );
                for d in 0..ndim {
                    let pos = pk.positions.get(d).copied().unwrap_or(0.0);
                    let punc = pk.position_uncertainties.get(d).copied().flatten();
                    row.push_str(&format!(" {:9.4} {:6}", pos, fmt_opt_f64(punc, 4)));
                }
                for d in 0..ndim {
                    let ch = pk.chain_codes.get(d).and_then(|x| x.as_deref());
                    let sq = pk.sequence_codes.get(d).copied().flatten();
                    let rs = pk.residue_names.get(d).and_then(|x| x.as_deref());
                    let at = pk.atom_names.get(d).and_then(|x| x.as_deref());
                    row.push_str(&format!(
                        " {:3} {:4} {:4} {:5}",
                        fmt_opt_str(ch),
                        fmt_opt_i32(sq),
                        fmt_opt_str(rs),
                        fmt_opt_str(at),
                    ));
                }
                row.push_str(&format!(" {}\n", fmt_opt_str(pk.details.as_deref())));
                out.push_str(&row);
            }
            out.push_str("   stop_\n");
        }
        out.push_str("save_\n\n");
    }

    // 5. save_nef_distance_restraint_list_<name>
    for dlist in &doc.distance_restraint_lists {
        let dname = sanitize_identifier(&dlist.name);
        let frame = format!("nef_distance_restraint_list_{dname}");
        out.push_str(&format!("save_{frame}\n"));
        out.push_str(
            "   _nef_distance_restraint_list.sf_category     nef_distance_restraint_list\n",
        );
        out.push_str(&format!(
            "   _nef_distance_restraint_list.sf_framecode    {frame}\n"
        ));
        out.push_str(&format!(
            "   _nef_distance_restraint_list.potential_type  {}\n\n",
            fmt_val(&dlist.potential_type)
        ));
        if !dlist.restraints.is_empty() {
            out.push_str("   loop_\n");
            for tag in [
                "_nef_distance_restraint.index",
                "_nef_distance_restraint.restraint_id",
                "_nef_distance_restraint.restraint_combination_id",
                "_nef_distance_restraint.chain_code_1",
                "_nef_distance_restraint.sequence_code_1",
                "_nef_distance_restraint.residue_name_1",
                "_nef_distance_restraint.atom_name_1",
                "_nef_distance_restraint.chain_code_2",
                "_nef_distance_restraint.sequence_code_2",
                "_nef_distance_restraint.residue_name_2",
                "_nef_distance_restraint.atom_name_2",
                "_nef_distance_restraint.weight",
                "_nef_distance_restraint.target_value",
                "_nef_distance_restraint.target_value_uncertainty",
                "_nef_distance_restraint.lower_linear_limit",
                "_nef_distance_restraint.lower_limit",
                "_nef_distance_restraint.upper_limit",
                "_nef_distance_restraint.upper_linear_limit",
            ] {
                out.push_str(&format!("      {tag}\n"));
            }
            out.push('\n');
            for r in &dlist.restraints {
                out.push_str(&format!(
                    "      {:5} {:5} {:4} {:3} {:4} {:4} {:6} {:3} {:4} {:4} {:6} {:6.3} {:7} {:7} {:7} {:7} {:7.3} {}\n",
                    r.index,
                    r.restraint_id,
                    fmt_opt_usize(r.restraint_combination_id),
                    fmt_val(&r.chain_code_1),
                    r.sequence_code_1,
                    fmt_val(&r.residue_name_1),
                    fmt_val(&r.atom_name_1),
                    fmt_val(&r.chain_code_2),
                    r.sequence_code_2,
                    fmt_val(&r.residue_name_2),
                    fmt_val(&r.atom_name_2),
                    r.weight,
                    fmt_opt_f64(r.target_value, 3),
                    fmt_opt_f64(r.target_value_uncertainty, 3),
                    fmt_opt_f64(r.lower_linear_limit, 3),
                    fmt_opt_f64(r.lower_limit, 3),
                    r.upper_limit,
                    fmt_opt_f64(r.upper_linear_limit, 3),
                ));
            }
            out.push_str("   stop_\n");
        }
        out.push_str("save_\n\n");
    }

    // 6. save_nef_dihedral_restraint_list_<name>
    for tlist in &doc.dihedral_restraint_lists {
        let tname = sanitize_identifier(&tlist.name);
        let frame = format!("nef_dihedral_restraint_list_{tname}");
        out.push_str(&format!("save_{frame}\n"));
        out.push_str(
            "   _nef_dihedral_restraint_list.sf_category     nef_dihedral_restraint_list\n",
        );
        out.push_str(&format!(
            "   _nef_dihedral_restraint_list.sf_framecode    {frame}\n"
        ));
        out.push_str(&format!(
            "   _nef_dihedral_restraint_list.potential_type  {}\n\n",
            fmt_val(&tlist.potential_type)
        ));
        if !tlist.restraints.is_empty() {
            out.push_str("   loop_\n");
            for tag in [
                "_nef_dihedral_restraint.index",
                "_nef_dihedral_restraint.restraint_id",
                "_nef_dihedral_restraint.restraint_combination_id",
                "_nef_dihedral_restraint.chain_code_1",
                "_nef_dihedral_restraint.sequence_code_1",
                "_nef_dihedral_restraint.residue_name_1",
                "_nef_dihedral_restraint.atom_name_1",
                "_nef_dihedral_restraint.chain_code_2",
                "_nef_dihedral_restraint.sequence_code_2",
                "_nef_dihedral_restraint.residue_name_2",
                "_nef_dihedral_restraint.atom_name_2",
                "_nef_dihedral_restraint.chain_code_3",
                "_nef_dihedral_restraint.sequence_code_3",
                "_nef_dihedral_restraint.residue_name_3",
                "_nef_dihedral_restraint.atom_name_3",
                "_nef_dihedral_restraint.chain_code_4",
                "_nef_dihedral_restraint.sequence_code_4",
                "_nef_dihedral_restraint.residue_name_4",
                "_nef_dihedral_restraint.atom_name_4",
                "_nef_dihedral_restraint.weight",
                "_nef_dihedral_restraint.target_value",
                "_nef_dihedral_restraint.target_value_uncertainty",
                "_nef_dihedral_restraint.lower_linear_limit",
                "_nef_dihedral_restraint.lower_limit",
                "_nef_dihedral_restraint.upper_limit",
                "_nef_dihedral_restraint.upper_linear_limit",
                "_nef_dihedral_restraint.name",
            ] {
                out.push_str(&format!("      {tag}\n"));
            }
            out.push('\n');
            for r in &tlist.restraints {
                out.push_str(&format!(
                    "      {:4} {:4} {:3} {:2} {:4} {:4} {:4} {:2} {:4} {:4} {:4} {:2} {:4} {:4} {:4} {:2} {:4} {:4} {:4} {:5.2} {:8.2} {:6} {:6} {:8.2} {:8.2} {:6} {}\n",
                    r.index,
                    r.restraint_id,
                    fmt_opt_usize(r.restraint_combination_id),
                    fmt_val(&r.chain_code_1),
                    r.sequence_code_1,
                    fmt_val(&r.residue_name_1),
                    fmt_val(&r.atom_name_1),
                    fmt_val(&r.chain_code_2),
                    r.sequence_code_2,
                    fmt_val(&r.residue_name_2),
                    fmt_val(&r.atom_name_2),
                    fmt_val(&r.chain_code_3),
                    r.sequence_code_3,
                    fmt_val(&r.residue_name_3),
                    fmt_val(&r.atom_name_3),
                    fmt_val(&r.chain_code_4),
                    r.sequence_code_4,
                    fmt_val(&r.residue_name_4),
                    fmt_val(&r.atom_name_4),
                    r.weight,
                    r.target_value,
                    fmt_opt_f64(r.target_value_uncertainty, 2),
                    fmt_opt_f64(r.lower_linear_limit, 2),
                    r.lower_limit,
                    r.upper_limit,
                    fmt_opt_f64(r.upper_linear_limit, 2),
                    fmt_val(&r.name),
                ));
            }
            out.push_str("   stop_\n");
        }
        out.push_str("save_\n\n");
    }

    // 7. Optional vendor-prefixed `save_nmrassign_workbench` (`NEF v1.1` extensibility)
    if let Some(ref wb) = doc.workbench_session {
        write_workbench_saveframe(&mut out, wb);
    }

    out
}

pub(crate) fn fmt_exact_f64(v: f64) -> String {
    if v == 0.0 {
        if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        }
    } else {
        format!("{v}")
    }
}

pub(crate) fn fmt_exact_f32(v: f32) -> String {
    if v == 0.0 {
        if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        }
    } else {
        format!("{v}")
    }
}

pub(crate) fn fmt_opt_exact_f64(v: Option<f64>) -> String {
    match v {
        Some(x) => fmt_exact_f64(x),
        None => ".".to_string(),
    }
}

pub(crate) fn encode_lossless_star_token(s: &str) -> String {
    let low = s.to_ascii_lowercase();
    let is_star_keyword = low == "stop_"
        || low == "save_"
        || low == "loop_"
        || low.starts_with("save_")
        || low.starts_with("data_");
    let force_hex = s.is_empty()
        || s == "."
        || s == "?"
        || is_star_keyword
        || s.starts_with("~hex:")
        || s.contains("\n;")
        || s.contains('\r')
        || s.chars().any(|c| c.is_control() && c != '\n' && c != '\t');
    if force_hex {
        let mut out = String::with_capacity(5 + s.len() * 2);
        out.push_str("~hex:");
        for b in s.as_bytes() {
            out.push_str(&format!("{b:02x}"));
        }
        out
    } else if (s.contains('\'') && s.contains('"')) || s.contains('\n') || s.len() > 80 {
        fmt_multiline_star(s)
    } else {
        fmt_val(s)
    }
}

pub(crate) fn decode_lossless_star_token(raw: &str) -> Result<String, String> {
    if let Some(hex) = raw.strip_prefix("~hex:") {
        if hex.len() % 2 != 0 {
            return Err(format!("invalid ~hex token length in '{raw}'"));
        }
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        for chunk in hex.as_bytes().chunks_exact(2) {
            let pair = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
            let b = u8::from_str_radix(pair, 16)
                .map_err(|e| format!("invalid hex byte '{pair}' in ~hex token: {e}"))?;
            bytes.push(b);
        }
        String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 in ~hex token: {e}"))
    } else {
        Ok(raw.to_string())
    }
}

pub(crate) fn encode_json_star_token(json: &str) -> String {
    if !can_use_multiline_star_block(json) || json.starts_with("~hex:") {
        encode_lossless_star_token(json)
    } else {
        fmt_multiline_star(json)
    }
}

fn write_workbench_saveframe(out: &mut String, wb: &NefWorkbenchSaveframe) {
    out.push_str("save_nmrassign_workbench\n");
    out.push_str("   _nmrassign_workbench.sf_category     nmrassign_workbench\n");
    out.push_str("   _nmrassign_workbench.sf_framecode    nmrassign_workbench\n");
    out.push_str(&format!(
        "   _nmrassign_workbench.schema_version  {}\n",
        encode_lossless_star_token(&wb.schema_version)
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.protein_name    {}\n",
        encode_lossless_star_token(&wb.protein_name)
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.chain_code      {}\n",
        encode_lossless_star_token(&wb.chain_code)
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.ref_offset_h    {}\n",
        fmt_exact_f64(wb.reference_offsets[0])
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.ref_offset_n    {}\n",
        fmt_exact_f64(wb.reference_offsets[1])
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.ref_offset_ca   {}\n",
        fmt_exact_f64(wb.reference_offsets[2])
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.ref_offset_cb   {}\n",
        fmt_exact_f64(wb.reference_offsets[3])
    ));
    out.push_str(&format!(
        "   _nmrassign_workbench.ref_offset_c    {}\n",
        fmt_exact_f64(wb.reference_offsets[4])
    ));
    let qc_tok = match wb.qc_verdict.as_deref() {
        Some(v) => encode_lossless_star_token(v),
        None => ".".to_string(),
    };
    out.push_str(&format!(
        "   _nmrassign_workbench.qc_verdict      {qc_tok}\n"
    ));
    if let Some(ref crc) = wb.shift_list_crc32 {
        out.push_str(&format!(
            "   _nmrassign_workbench.shift_list_crc32 {}\n",
            encode_lossless_star_token(crc)
        ));
    }
    if let Some([hl, hr, nt, nb]) = wb.view_bounds_ppm {
        out.push_str(&format!(
            "   _nmrassign_workbench.view_h_left     {}\n",
            fmt_exact_f64(hl)
        ));
        out.push_str(&format!(
            "   _nmrassign_workbench.view_h_right    {}\n",
            fmt_exact_f64(hr)
        ));
        out.push_str(&format!(
            "   _nmrassign_workbench.view_n_top      {}\n",
            fmt_exact_f64(nt)
        ));
        out.push_str(&format!(
            "   _nmrassign_workbench.view_n_bottom   {}\n",
            fmt_exact_f64(nb)
        ));
    }

    if !wb.residues.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_residue.index",
            "_nmrassign_residue.seq_id",
            "_nmrassign_residue.residue_name",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, r) in wb.residues.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {}\n",
                i + 1,
                r.seq_id,
                encode_lossless_star_token(&r.residue_name)
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.peaks.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_peak.index",
            "_nmrassign_peak.experiment",
            "_nmrassign_peak.h",
            "_nmrassign_peak.n",
            "_nmrassign_peak.c",
            "_nmrassign_peak.sign",
            "_nmrassign_peak.height",
            "_nmrassign_peak.volume",
            "_nmrassign_peak.h2",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, p) in wb.peaks.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {} {} {} {} {} {} {}\n",
                i + 1,
                encode_lossless_star_token(&p.experiment),
                fmt_exact_f64(p.h),
                fmt_exact_f64(p.n),
                fmt_opt_exact_f64(p.c),
                p.sign,
                fmt_opt_exact_f64(p.height),
                fmt_opt_exact_f64(p.volume),
                fmt_opt_exact_f64(p.h2),
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.assignments.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_assignment.index",
            "_nmrassign_assignment.residue_seq_idx",
            "_nmrassign_assignment.spin_system_id",
            "_nmrassign_assignment.confidence",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, a) in wb.assignments.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {} {}\n",
                i + 1,
                a.residue_seq_idx,
                fmt_opt_usize(a.spin_system_id),
                fmt_exact_f64(a.confidence),
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.assignment_shifts.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_assignment_shift.index",
            "_nmrassign_assignment_shift.residue_seq_idx",
            "_nmrassign_assignment_shift.atom_name",
            "_nmrassign_assignment_shift.value",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, s) in wb.assignment_shifts.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {} {}\n",
                i + 1,
                s.residue_seq_idx,
                encode_lossless_star_token(&s.atom_name),
                fmt_exact_f64(s.value),
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.locks.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_lock.index",
            "_nmrassign_lock.spin_system_id",
            "_nmrassign_lock.residue_seq_idx",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, lk) in wb.locks.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {}\n",
                i + 1,
                lk.spin_system_id,
                lk.residue_seq_idx,
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.confirmed_links.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_confirmed_link.index",
            "_nmrassign_confirmed_link.pred_spin_system",
            "_nmrassign_confirmed_link.succ_spin_system",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, lnk) in wb.confirmed_links.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {}\n",
                i + 1,
                lnk.pred_spin_system,
                lnk.succ_spin_system,
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.contour_layers.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_contour_layer.index",
            "_nmrassign_contour_layer.layer_id",
            "_nmrassign_contour_layer.visible",
            "_nmrassign_contour_layer.base_level",
            "_nmrassign_contour_layer.level_factor",
            "_nmrassign_contour_layer.num_levels",
            "_nmrassign_contour_layer.show_negative",
            "_nmrassign_contour_layer.pos_r",
            "_nmrassign_contour_layer.pos_g",
            "_nmrassign_contour_layer.pos_b",
            "_nmrassign_contour_layer.pos_a",
            "_nmrassign_contour_layer.neg_r",
            "_nmrassign_contour_layer.neg_g",
            "_nmrassign_contour_layer.neg_b",
            "_nmrassign_contour_layer.neg_a",
            "_nmrassign_contour_layer.ppm_nudge_x",
            "_nmrassign_contour_layer.ppm_nudge_y",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, cl) in wb.contour_layers.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
                i + 1,
                encode_lossless_star_token(&cl.layer_id),
                fmt_opt_bool(Some(cl.visible)),
                fmt_exact_f32(cl.base_level),
                fmt_exact_f32(cl.level_factor),
                cl.num_levels,
                fmt_opt_bool(Some(cl.show_negative)),
                fmt_exact_f32(cl.pos_rgba[0]),
                fmt_exact_f32(cl.pos_rgba[1]),
                fmt_exact_f32(cl.pos_rgba[2]),
                fmt_exact_f32(cl.pos_rgba[3]),
                fmt_exact_f32(cl.neg_rgba[0]),
                fmt_exact_f32(cl.neg_rgba[1]),
                fmt_exact_f32(cl.neg_rgba[2]),
                fmt_exact_f32(cl.neg_rgba[3]),
                fmt_exact_f32(cl.ppm_nudge[0]),
                fmt_exact_f32(cl.ppm_nudge[1]),
            ));
        }
        out.push_str("   stop_\n");
    }

    if !wb.processing_recipes.is_empty() {
        out.push_str("\n   loop_\n");
        for tag in [
            "_nmrassign_processing_recipe.index",
            "_nmrassign_processing_recipe.spectrum_id",
            "_nmrassign_processing_recipe.recipe_json",
        ] {
            out.push_str(&format!("      {tag}\n"));
        }
        out.push('\n');
        for (i, pr) in wb.processing_recipes.iter().enumerate() {
            out.push_str(&format!(
                "      {} {} {}\n",
                i + 1,
                encode_lossless_star_token(&pr.spectrum_id),
                encode_json_star_token(&pr.recipe_json),
            ));
        }
        out.push_str("   stop_\n");
    }

    out.push_str("save_\n\n");
}
