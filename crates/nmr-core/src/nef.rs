//! NMR Exchange Format (NEF) v1.1 reader, writer, and conversion utilities.
//!
//! Provenance: Clean-room original implementation of the open NEF v1.1 specification:
//! Gutmanas, A. et al., "NMR Exchange Format: a unified and open standard for
//! representation of NMR restraint data", *Nature Structural & Molecular Biology*
//! 22, 433–434 (2015), covering `nef_nmr_meta_data`, `nef_molecular_system`,
//! `nef_chemical_shift_list`, `nef_nmr_spectrum`, `nef_distance_restraint_list`,
//! and `nef_dihedral_restraint_list`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::peaks::Peak;

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
            program_version: "0.2.0".to_string(),
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
            out.push(buf.join("\n").trim().to_string());
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
        }
    }

    Ok(doc)
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

fn fmt_val(s: &str) -> String {
    if s.is_empty() {
        return ".".to_string();
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
        } else {
            format!("\"{}\"", s.replace('"', "'"))
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

    out
}
