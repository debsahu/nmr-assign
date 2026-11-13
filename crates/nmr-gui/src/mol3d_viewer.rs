//! Interactive WebGPU (`wgpu`) 3D Biomolecular Structure Viewport (`T12.1` & `T12.2`).
//!
//! Provenance: Clean-room original implementation (`docs/specs/15-structure-viewer.md`)
//! grounded in peer-reviewed structural biology, computer graphics, and numerical geometry papers:
//! - Carson, M. & Bugg, C. E. (*J. Mol. Graph.* 1986, 4, 121–122 — smooth cubic B-spline / Hermite
//!   peptide-plane ribbon extrusion for alpha-helices, arrowed beta-strands, and coil/putty tubes).
//! - Kabsch, W. (*Acta Crystallogr. A* 1976, 32, 922–923) & Coutsias, E. A., Seok, C. & Dill, K. A.
//!   (*J. Comput. Chem.* 2004, 25, 1849–1857 — quaternion least-squares proper-rotation `det(R) = +1`
//!   multi-conformer NMR ensemble superposition and per-residue `CA` spatial RMSD).
//! - Kabsch, W. & Sander, C. (*Biopolymers* 1983, 22, 2577–2637) & Lovell, S. C. et al. (*Proteins*
//!   2003, 50, 437–450 — electrostatic backbone hydrogen-bond secondary-structure assignment with
//!   Ramachandran `(phi, psi)` basin fallback when header `HELIX`/`SHEET` records are absent).
//! - Shoemake, K. (*ACM SIGGRAPH Comput. Graph.* 1985, 19(3), 245–254 — quaternion arcball rotation).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu, wgpu};
use nmr_core::nmrstar::Reference;
use nmr_core::priors::Priors;
use wgpu::util::DeviceExt;

use crate::mol3d_overlay::{
    pick_scene_analytical, resolve_3d_pick_sync, AtomCoordLookup, ColormapKind, CylinderInstance,
    DistanceRestraint3d, Mol3dSyncAction, PickableRestraint3d, PickableRibbonSegment3d,
    PickableSphere3d, Picked3dHit, QuantitativeTrack, QuantitativeTrackData, RestraintFilterConfig,
    RestraintOverlayBuild,
};

/// Hardware MSAA sample count for crisp silhouette anti-aliasing in the offscreen 3D pass.
pub const MOL3D_MSAA_SAMPLES: u32 = 4;

const MOL3D_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MOL3D_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const DEFAULT_BG_RGBA: [f32; 4] = [0.055, 0.068, 0.095, 1.0];

// ============================================================================
// 1. XML Escaping & Safe ASCII/UTF-8 Parsing Utilities
// ============================================================================

/// Mandatory XML character-entity escaper applied to every string interpolated into exported SVG.
pub fn escape_xml(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() && c != '\t' && c != '\n' => {}
            c => out.push(c),
        }
    }
    out
}

/// Safe byte-bounded ASCII column extractor for fixed-width PDB records (`0`-based `[start, end)`).
///
/// Never slices `&str` by byte offsets: inspects `bytes.get(...)` and verifies `.is_ascii()` so
/// arbitrary multibyte UTF-8 input can never panic.
#[inline]
fn pdb_ascii_col(bytes: &[u8], start: usize, end: usize) -> Result<&str, String> {
    let n = bytes.len();
    if start >= n || start >= end {
        return Ok("");
    }
    let slice = bytes
        .get(start..end.min(n))
        .ok_or_else(|| "PDB column range out of bounds".to_string())?;
    if !slice.is_ascii() {
        return Err(format!(
            "Non-ASCII byte in fixed-width PDB column {}..{}",
            start + 1,
            end
        ));
    }
    std::str::from_utf8(slice).map_err(|e| format!("Invalid UTF-8 in PDB column: {e}"))
}

#[inline]
fn validate_finite_coord(val: f64, axis: char, context: &str) -> Result<f32, String> {
    if !val.is_finite() || val.abs() > 1.0e6 {
        return Err(format!(
            "Invalid non-finite or out-of-range {axis} coordinate ({val}) at {context}"
        ));
    }
    Ok(val as f32)
}

#[inline]
fn parse_finite_f64_token(tok: &str, label: &str, context: &str) -> Result<f64, String> {
    let trimmed = tok.trim();
    if trimmed.is_empty() {
        return Err(format!("Missing {label} value at {context}"));
    }
    let v: f64 = trimmed
        .parse()
        .map_err(|_| format!("Malformed {label} '{trimmed}' at {context}"))?;
    if !v.is_finite() {
        return Err(format!("Non-finite {label} '{trimmed}' at {context}"));
    }
    Ok(v)
}

// ============================================================================
// 2. Biomolecular Data Model (`.pdb` & `.cif`/`.mmcif` Ensemble Support)
// ============================================================================

/// Secondary-structure classification assigned by Kabsch–Sander H-bonds / Ramachandran basins or
/// explicit `HELIX` / `SHEET` annotations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SecondaryStructureKind {
    #[default]
    Coil,
    Helix,
    Sheet,
}

impl SecondaryStructureKind {
    pub const fn default_rgba(self) -> [f32; 4] {
        match self {
            Self::Helix => [0.16, 0.76, 0.94, 1.0],
            Self::Sheet => [0.96, 0.78, 0.22, 1.0],
            Self::Coil => [0.68, 0.75, 0.84, 1.0],
        }
    }
}

/// Primary 3D molecular representation mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RepresentationStyle {
    /// Smooth Carson–Bugg cartoon ribbons + metal cofactors.
    Cartoon,
    /// Smooth Carson–Bugg cartoon ribbons + heavy-atom sidechain sticks & metal cofactors.
    #[default]
    CartoonAndSidechains,
    /// All-atom covalent sticks + coordination bonds.
    LicoriceSticks,
    /// Ball-and-stick spheres + half-bond CPK cylinders.
    BallAndStick,
    /// Van der Waals space-filling CPK spheres.
    Spacefill,
}

impl RepresentationStyle {
    pub const ALL: [Self; 5] = [
        Self::Cartoon,
        Self::CartoonAndSidechains,
        Self::LicoriceSticks,
        Self::BallAndStick,
        Self::Spacefill,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cartoon => "Cartoon",
            Self::CartoonAndSidechains => "Cartoon + Sidechains",
            Self::LicoriceSticks => "Sticks",
            Self::BallAndStick => "Ball & Stick",
            Self::Spacefill => "VDW Spheres",
        }
    }
}

/// Atom entry normalized across PDB and mmCIF formats (`MSE` -> `MET`, altlocs resolved).
#[derive(Clone, Debug)]
pub struct StructureAtom3d {
    pub serial: u32,
    pub atom_name: String,
    pub element: String,
    pub res_name: String,
    pub chain_id: char,
    pub seq_id: i32,
    pub ins_code: char,
    pub pos: [f32; 3],
    pub occupancy: f32,
    pub b_factor: f32,
    pub is_hetatm: bool,
    pub is_metal: bool,
}

/// Polymer amino-acid residue within a single chain and conformer.
#[derive(Clone, Debug)]
pub struct PolymerResidue3d {
    pub chain_id: char,
    pub seq_id: i32,
    pub ins_code: char,
    pub res_name: String,
    /// Optional `0`-based index into the primary polymer sequence for NMR shift/track coupling.
    pub sequence_idx: Option<usize>,
    pub ss: SecondaryStructureKind,
    pub atoms: Vec<StructureAtom3d>,
}

impl PolymerResidue3d {
    #[inline]
    pub fn atom_pos(&self, name: &str) -> Option<[f32; 3]> {
        self.atoms
            .iter()
            .find(|a| a.atom_name == name)
            .map(|a| a.pos)
    }
}

/// Contiguous polypeptide chain within a single conformer.
#[derive(Clone, Debug)]
pub struct PolymerChain3d {
    pub chain_id: char,
    pub residues: Vec<PolymerResidue3d>,
}

/// Single 3D conformer (`MODEL`) after Kabsch superposition onto conformer `0`.
#[derive(Clone, Debug)]
pub struct StructureModel3d {
    pub model_number: usize,
    pub chains: Vec<PolymerChain3d>,
    pub het_atoms: Vec<StructureAtom3d>,
    /// Kabsch `CA` RMSD (`Å`) of this conformer relative to conformer `0` (`0.0` for conformer `0`).
    pub kabsch_rmsd_to_ref: f64,
}

impl StructureModel3d {
    pub fn total_atom_count(&self) -> usize {
        self.chains
            .iter()
            .map(|c| c.residues.iter().map(|r| r.atoms.len()).sum::<usize>())
            .sum::<usize>()
            + self.het_atoms.len()
    }
}

/// Multi-conformer biomolecular structure ensemble (`1..M` models) with Kabsch superposition,
/// per-residue `CA` spatial RMSD, sequence mapping, and pre-indexed coordinate lookup.
#[derive(Clone, Debug)]
pub struct StructureEnsemble3d {
    pub title: String,
    pub models: Vec<StructureModel3d>,
    /// Primary polymer chain `3`-letter residue names (`MET` normalized from `MSE`).
    pub primary_sequence: Vec<String>,
    /// Primary polymer chain author `seq_id`s matching `primary_sequence`.
    pub primary_seq_ids: Vec<i32>,
    /// Per-residue `CA` spatial RMSD (`Å`) across all conformers for `0..primary_sequence.len()`.
    pub per_residue_ca_rmsd: Vec<f64>,
    /// Mean ensemble `CA` RMSD (`Å`) across `per_residue_ca_rmsd`.
    pub mean_ensemble_ca_rmsd: f64,
    /// Bounding center (`Å`) of conformer `0`.
    pub center: [f64; 3],
    /// Bounding sphere radius (`Å`) around `center`.
    pub bounding_radius: f64,
    /// Fast coordinate index on conformer `0` for `< 0.5 ms` NOE/PRE restraint evaluation.
    pub coord_lookup: AtomCoordLookup,
}

impl StructureEnsemble3d {
    pub fn total_atom_count(&self) -> usize {
        self.models
            .iter()
            .map(StructureModel3d::total_atom_count)
            .sum()
    }

    pub fn residue_count_model0(&self) -> usize {
        self.models
            .first()
            .map(|m| m.chains.iter().map(|c| c.residues.len()).sum())
            .unwrap_or(0)
    }

    /// Re-align `sequence_idx` on polymer residues to an external NMR [`Reference`] sequence when
    /// loaded independently (matching by author `seq_id` offset or direct `0`-based index).
    pub fn bind_to_nmr_sequence(&mut self, nmr_seq: &[String], nmr_seq_ids: &[i32]) {
        if nmr_seq.is_empty() {
            return;
        }
        let map_res = |seq_id: i32, fallback_idx: usize, res_name: &str| -> Option<usize> {
            if !nmr_seq_ids.is_empty() {
                if let Some(pos) = nmr_seq_ids.iter().position(|&sid| sid == seq_id) {
                    if pos < nmr_seq.len() {
                        return Some(pos);
                    }
                }
            }
            if seq_id >= 1 && ((seq_id - 1) as usize) < nmr_seq.len() {
                let k = (seq_id - 1) as usize;
                if nmr_seq[k].eq_ignore_ascii_case(res_name) || nmr_seq.len() > fallback_idx {
                    return Some(k);
                }
            }
            (fallback_idx < nmr_seq.len()).then_some(fallback_idx)
        };

        for model in &mut self.models {
            for (c_idx, chain) in model.chains.iter_mut().enumerate() {
                if c_idx > 0 {
                    continue;
                }
                for (r_idx, res) in chain.residues.iter_mut().enumerate() {
                    res.sequence_idx = map_res(res.seq_id, r_idx, &res.res_name);
                }
            }
        }
        self.rebuild_coord_lookup();
    }

    fn rebuild_coord_lookup(&mut self) {
        if let Some(m0) = self.models.first() {
            let polymer_iter = m0.chains.iter().flat_map(|c| {
                c.residues.iter().flat_map(move |r| {
                    r.atoms.iter().map(move |a| {
                        (
                            c.chain_id,
                            r.seq_id,
                            r.sequence_idx,
                            r.res_name.as_str(),
                            a.atom_name.as_str(),
                            a.pos,
                        )
                    })
                })
            });
            let het_iter = m0.het_atoms.iter().map(|a| {
                (
                    a.chain_id,
                    a.seq_id,
                    None,
                    a.res_name.as_str(),
                    a.atom_name.as_str(),
                    a.pos,
                )
            });
            self.coord_lookup = AtomCoordLookup::from_atom_tuples(polymer_iter.chain(het_iter));
        }
    }
}

// ============================================================================
// 3. Amino-Acid, Metal Cofactor & Altloc Normalization Helpers
// ============================================================================

#[inline]
fn is_standard_amino_acid(res_name: &str) -> bool {
    matches!(
        res_name,
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "CYS"
            | "GLN"
            | "GLU"
            | "GLY"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "SEC"
            | "PYL"
    )
}

#[inline]
fn is_solvent_res(res_name: &str) -> bool {
    matches!(res_name, "HOH" | "WAT" | "DOD" | "D2O" | "SOL")
}

#[inline]
fn is_metal_element(elem: &str) -> bool {
    matches!(
        elem,
        "ZN" | "FE"
            | "CU"
            | "MN"
            | "CO"
            | "NI"
            | "MG"
            | "CA"
            | "NA"
            | "K"
            | "CD"
            | "HG"
            | "GD"
            | "TB"
            | "DY"
            | "TM"
            | "YB"
            | "LU"
            | "EU"
            | "LA"
            | "PT"
            | "MO"
            | "W"
            | "V"
    )
}

fn infer_clean_element(raw_elem: &str, atom_name: &str, res_name: &str, is_hetatm: bool) -> String {
    let trimmed = raw_elem.trim().to_ascii_uppercase();
    let cleaned: String = trimmed
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if !cleaned.is_empty() {
        if !is_hetatm && is_standard_amino_acid(res_name) && cleaned == "CA" && atom_name == "CA" {
            return "C".to_string();
        }
        return cleaned;
    }

    let alpha: String = atom_name
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase();

    if !is_hetatm && is_standard_amino_acid(res_name) {
        return alpha
            .chars()
            .next()
            .map(|c| {
                if c == 'D' {
                    "H".to_string()
                } else {
                    c.to_string()
                }
            })
            .unwrap_or_else(|| "C".to_string());
    }

    if is_metal_element(&alpha)
        && (res_name == alpha || matches!(res_name, "FES" | "SF4" | "F3S" | "HEM" | "HEC"))
    {
        return alpha;
    }
    alpha
        .chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "C".to_string())
}

#[derive(Clone, Debug)]
struct RawAtomRecord {
    model_num: usize,
    serial: u32,
    atom_name: String,
    alt_loc: char,
    res_name: String,
    chain_id: char,
    seq_id: i32,
    ins_code: char,
    pos: [f32; 3],
    occupancy: f32,
    b_factor: f32,
    element: String,
    is_hetatm: bool,
}

impl RawAtomRecord {
    /// Normalize `MSE` -> `MET` (`SE` -> `SD`, `S`) and promote selenomethionine to polymer atom.
    fn normalize_mse(&mut self) {
        if self.res_name == "MSE" {
            self.res_name = "MET".to_string();
            self.is_hetatm = false;
            if self.atom_name == "SE" {
                self.atom_name = "SD".to_string();
                self.element = "S".to_string();
            }
        }
    }

    /// Priority score for resolving alternate conformations (`' '` > higher occupancy > `'A'` > `'B'`).
    /// Ensures `B`-only altloc atoms are retained when `' '` and `'A'` are absent!
    #[inline]
    fn altloc_priority(&self) -> (u8, i32, i32) {
        let blank_rank = if matches!(self.alt_loc, ' ' | '.' | '?') {
            2u8
        } else {
            1u8
        };
        let occ_milli = (self.occupancy.clamp(0.0, 10.0) * 1000.0).round() as i32;
        let char_tiebreak = -(self.alt_loc.to_ascii_uppercase() as i32);
        (blank_rank, occ_milli, char_tiebreak)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SsSpan {
    chain_id: char,
    start_seq: i32,
    end_seq: i32,
    kind: SecondaryStructureKind,
}

// ============================================================================
// 4. Safe `.pdb` & `.cif`/`.mmcif` Parsers
// ============================================================================

/// Auto-detect `.cif`/`.mmcif` vs `.pdb` format from text content and parse into a superposed
/// [`StructureEnsemble3d`].
pub fn parse_structure_auto(text: &str) -> Result<StructureEnsemble3d, String> {
    let looks_like_cif = text.lines().any(|line| {
        let t = line.trim_start();
        t.starts_with("data_") || t.starts_with("_atom_site.")
    });
    if looks_like_cif {
        parse_mmcif_ensemble(text)
    } else {
        parse_pdb_ensemble(text)
    }
}

/// Parse a Protein Data Bank (`.pdb` / `.ent`) single structure or multi-`MODEL` NMR ensemble
/// using strictly byte-safe ASCII column access (zero `&str` byte-offset slicing).
pub fn parse_pdb_ensemble(text: &str) -> Result<StructureEnsemble3d, String> {
    let mut title_parts: Vec<String> = Vec::new();
    let mut header_fallback = String::new();
    let mut ss_spans: Vec<SsSpan> = Vec::new();
    let mut raw_atoms: Vec<RawAtomRecord> = Vec::new();
    let mut current_model: usize = 1;
    let mut saw_explicit_model = false;
    let mut auto_model_counter = 0usize;

    for (line_idx, line) in text.lines().enumerate() {
        let bytes = line.as_bytes();
        let rec_prefix = bytes.get(0..6).unwrap_or(bytes);

        if rec_prefix.starts_with(b"TITLE ") {
            let body: String = line.chars().skip(10).collect();
            let trimmed = body.trim();
            if !trimmed.is_empty() {
                title_parts.push(trimmed.to_string());
            }
            continue;
        }
        if rec_prefix.starts_with(b"HEADER") {
            let body: String = line.chars().skip(10).take(56).collect();
            let trimmed = body.trim();
            if !trimmed.is_empty() && header_fallback.is_empty() {
                header_fallback = trimmed.to_string();
            }
            continue;
        }
        if rec_prefix.starts_with(b"MODEL ") || rec_prefix == b"MODEL" {
            saw_explicit_model = true;
            auto_model_counter += 1;
            let rest: String = line.chars().skip(5).collect();
            current_model = rest
                .split_whitespace()
                .next()
                .and_then(|t| t.parse::<usize>().ok())
                .filter(|&m| m > 0)
                .unwrap_or(auto_model_counter);
            continue;
        }
        if rec_prefix.starts_with(b"ENDMDL") {
            if saw_explicit_model {
                current_model = current_model.saturating_add(1);
            }
            continue;
        }
        if rec_prefix.starts_with(b"HELIX ") {
            if let (Ok(ch_s), Ok(s0), Ok(s1)) = (
                pdb_ascii_col(bytes, 19, 20),
                pdb_ascii_col(bytes, 21, 25),
                pdb_ascii_col(bytes, 33, 37),
            ) {
                let chain_id = ch_s.chars().next().unwrap_or('A');
                if let (Ok(a), Ok(b)) = (s0.trim().parse::<i32>(), s1.trim().parse::<i32>()) {
                    ss_spans.push(SsSpan {
                        chain_id: if chain_id == ' ' { 'A' } else { chain_id },
                        start_seq: a.min(b),
                        end_seq: a.max(b),
                        kind: SecondaryStructureKind::Helix,
                    });
                }
            }
            continue;
        }
        if rec_prefix.starts_with(b"SHEET ") {
            if let (Ok(ch_s), Ok(s0), Ok(s1)) = (
                pdb_ascii_col(bytes, 21, 22),
                pdb_ascii_col(bytes, 22, 26),
                pdb_ascii_col(bytes, 33, 37),
            ) {
                let chain_id = ch_s.chars().next().unwrap_or('A');
                if let (Ok(a), Ok(b)) = (s0.trim().parse::<i32>(), s1.trim().parse::<i32>()) {
                    ss_spans.push(SsSpan {
                        chain_id: if chain_id == ' ' { 'A' } else { chain_id },
                        start_seq: a.min(b),
                        end_seq: a.max(b),
                        kind: SecondaryStructureKind::Sheet,
                    });
                }
            }
            continue;
        }

        let is_atom = rec_prefix == b"ATOM  ";
        let is_hetatm = rec_prefix == b"HETATM";
        if !is_atom && !is_hetatm {
            continue;
        }

        let ctx = format!("PDB line {}", line_idx + 1);
        if bytes.len() < 54 {
            return Err(format!(
                "Truncated ATOM/HETATM record (< 54 columns) at {ctx}"
            ));
        }

        let serial = pdb_ascii_col(bytes, 6, 11)?
            .trim()
            .parse::<u32>()
            .unwrap_or((raw_atoms.len() + 1) as u32);
        let atom_name = pdb_ascii_col(bytes, 12, 16)?.trim().to_ascii_uppercase();
        if atom_name.is_empty() {
            return Err(format!("Empty atom name at {ctx}"));
        }
        let alt_loc = pdb_ascii_col(bytes, 16, 17)?.chars().next().unwrap_or(' ');
        let res_name = pdb_ascii_col(bytes, 17, 20)?.trim().to_ascii_uppercase();
        if res_name.is_empty() {
            return Err(format!("Empty residue name at {ctx}"));
        }
        let raw_chain = pdb_ascii_col(bytes, 21, 22)?.chars().next().unwrap_or('A');
        let chain_id = if raw_chain == ' ' { 'A' } else { raw_chain };
        let seq_id_str = pdb_ascii_col(bytes, 22, 26)?.trim();
        let seq_id: i32 = seq_id_str
            .parse()
            .map_err(|_| format!("Invalid residue sequence number '{seq_id_str}' at {ctx}"))?;
        let ins_code = pdb_ascii_col(bytes, 26, 27)?.chars().next().unwrap_or(' ');

        let x_val = parse_finite_f64_token(pdb_ascii_col(bytes, 30, 38)?, "X", &ctx)?;
        let y_val = parse_finite_f64_token(pdb_ascii_col(bytes, 38, 46)?, "Y", &ctx)?;
        let z_val = parse_finite_f64_token(pdb_ascii_col(bytes, 46, 54)?, "Z", &ctx)?;
        let pos = [
            validate_finite_coord(x_val, 'X', &ctx)?,
            validate_finite_coord(y_val, 'Y', &ctx)?,
            validate_finite_coord(z_val, 'Z', &ctx)?,
        ];

        let occupancy = pdb_ascii_col(bytes, 54, 60)
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(1.0);
        let b_factor = pdb_ascii_col(bytes, 60, 66)
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let raw_elem = pdb_ascii_col(bytes, 76, 78).unwrap_or("");
        let element = infer_clean_element(raw_elem, &atom_name, &res_name, is_hetatm);

        let mut rec = RawAtomRecord {
            model_num: if saw_explicit_model { current_model } else { 1 },
            serial,
            atom_name,
            alt_loc,
            res_name,
            chain_id,
            seq_id,
            ins_code,
            pos,
            occupancy,
            b_factor,
            element,
            is_hetatm,
        };
        rec.normalize_mse();
        if !is_solvent_res(&rec.res_name) {
            raw_atoms.push(rec);
        }
    }

    if raw_atoms.is_empty() {
        return Err("No valid ATOM/HETATM coordinates found in PDB input".to_string());
    }

    let title = if !title_parts.is_empty() {
        title_parts.join(" ")
    } else if !header_fallback.is_empty() {
        header_fallback
    } else {
        "PDB Structure".to_string()
    };

    finalize_structure_ensemble(title, raw_atoms, &ss_spans)
}

/// Tokenize a single mmCIF line using Unicode `chars()` iteration (zero byte-offset slicing).
fn tokenize_cif_line(line: &str, out: &mut Vec<String>) {
    let mut chars = line.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '#' {
            break;
        }
        if ch == '\'' || ch == '"' {
            let quote = ch;
            chars.next();
            let mut buf = String::new();
            while let Some(c) = chars.next() {
                if c == quote && chars.peek().is_none_or(|n| n.is_whitespace()) {
                    break;
                }
                buf.push(c);
            }
            out.push(buf);
        } else {
            let mut buf = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                buf.push(c);
                chars.next();
            }
            out.push(buf);
        }
    }
}

/// Parse a Macromolecular Crystallographic Information File (`.cif` / `.mmcif`) `_atom_site` loop
/// into a superposed [`StructureEnsemble3d`] using safe Unicode tokenization.
pub fn parse_mmcif_ensemble(text: &str) -> Result<StructureEnsemble3d, String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut in_semicolon_block = false;
    let mut semi_buf = String::new();

    for line in text.lines() {
        if in_semicolon_block {
            if line.starts_with(';') {
                tokens.push(semi_buf.trim().to_string());
                semi_buf.clear();
                in_semicolon_block = false;
                let rest: String = line.chars().skip(1).collect();
                tokenize_cif_line(&rest, &mut tokens);
            } else {
                if !semi_buf.is_empty() {
                    semi_buf.push(' ');
                }
                semi_buf.push_str(line.trim());
            }
        } else if line.starts_with(';') {
            in_semicolon_block = true;
            let rest: String = line.chars().skip(1).collect();
            if !rest.trim().is_empty() {
                semi_buf.push_str(rest.trim());
            }
        } else {
            tokenize_cif_line(line, &mut tokens);
        }
    }

    let mut title = String::new();
    let mut raw_atoms: Vec<RawAtomRecord> = Vec::new();
    let mut idx = 0usize;

    while idx < tokens.len() {
        let tok = &tokens[idx];
        if title.is_empty() && tok.starts_with("data_") {
            title = tok.chars().skip(5).collect();
            idx += 1;
            continue;
        }
        if tok == "_struct.title" && idx + 1 < tokens.len() {
            let t = tokens[idx + 1].trim();
            if !t.is_empty() && t != "." && t != "?" {
                title = t.to_string();
            }
            idx += 2;
            continue;
        }
        if tok.eq_ignore_ascii_case("loop_") {
            idx += 1;
            let mut headers: Vec<String> = Vec::new();
            while idx < tokens.len() && tokens[idx].starts_with('_') {
                headers.push(tokens[idx].to_ascii_lowercase());
                idx += 1;
            }
            let ncols = headers.len();
            if ncols == 0 {
                continue;
            }
            let is_atom_site = headers.iter().any(|h| h.starts_with("_atom_site."));
            if !is_atom_site {
                while idx < tokens.len()
                    && !tokens[idx].starts_with('_')
                    && !tokens[idx].eq_ignore_ascii_case("loop_")
                    && !tokens[idx].starts_with("data_")
                {
                    idx += 1;
                }
                continue;
            }

            let col_of = |names: &[&str]| -> Option<usize> {
                names
                    .iter()
                    .find_map(|target| headers.iter().position(|h| h == target))
            };

            let c_group = col_of(&["_atom_site.group_pdb"]);
            let c_serial = col_of(&["_atom_site.id"]);
            let c_symbol = col_of(&["_atom_site.type_symbol"]);
            let c_atom = col_of(&["_atom_site.auth_atom_id", "_atom_site.label_atom_id"])
                .ok_or_else(|| "mmCIF _atom_site missing atom_id column".to_string())?;
            let c_alt = col_of(&["_atom_site.label_alt_id"]);
            let c_comp = col_of(&["_atom_site.auth_comp_id", "_atom_site.label_comp_id"])
                .ok_or_else(|| "mmCIF _atom_site missing comp_id column".to_string())?;
            let c_asym = col_of(&["_atom_site.auth_asym_id", "_atom_site.label_asym_id"]);
            let c_seq = col_of(&["_atom_site.auth_seq_id", "_atom_site.label_seq_id"])
                .ok_or_else(|| "mmCIF _atom_site missing seq_id column".to_string())?;
            let c_ins = col_of(&["_atom_site.pdbx_pdb_ins_code"]);
            let c_x = col_of(&["_atom_site.cartn_x"])
                .ok_or_else(|| "mmCIF _atom_site missing Cartn_x column".to_string())?;
            let c_y = col_of(&["_atom_site.cartn_y"])
                .ok_or_else(|| "mmCIF _atom_site missing Cartn_y column".to_string())?;
            let c_z = col_of(&["_atom_site.cartn_z"])
                .ok_or_else(|| "mmCIF _atom_site missing Cartn_z column".to_string())?;
            let c_occ = col_of(&["_atom_site.occupancy"]);
            let c_bfac = col_of(&["_atom_site.b_iso_or_equiv"]);
            let c_model = col_of(&["_atom_site.pdbx_pdb_model_num"]);

            let mut row_num = 0usize;
            while idx + ncols <= tokens.len() {
                let first = &tokens[idx];
                if first.starts_with('_')
                    || first.eq_ignore_ascii_case("loop_")
                    || first.starts_with("data_")
                {
                    break;
                }
                row_num += 1;
                let row = &tokens[idx..idx + ncols];
                idx += ncols;

                let ctx = format!("mmCIF _atom_site row {row_num}");
                let group = c_group.map(|c| row[c].as_str()).unwrap_or("ATOM");
                let is_hetatm = group.eq_ignore_ascii_case("HETATM");
                let serial = c_serial
                    .and_then(|c| row[c].parse::<u32>().ok())
                    .unwrap_or(row_num as u32);
                let atom_name = row[c_atom].trim().to_ascii_uppercase();
                let alt_loc = c_alt
                    .and_then(|c| row[c].chars().next())
                    .map(|ch| if ch == '.' || ch == '?' { ' ' } else { ch })
                    .unwrap_or(' ');
                let res_name = row[c_comp].trim().to_ascii_uppercase();
                let chain_id = c_asym
                    .and_then(|c| row[c].chars().next())
                    .filter(|&ch| ch != '.' && ch != '?')
                    .unwrap_or('A');
                let seq_id: i32 = row[c_seq].parse::<i32>().unwrap_or(row_num as i32);
                let ins_code = c_ins
                    .and_then(|c| row[c].chars().next())
                    .map(|ch| if ch == '.' || ch == '?' { ' ' } else { ch })
                    .unwrap_or(' ');

                let x_val = parse_finite_f64_token(&row[c_x], "Cartn_x", &ctx)?;
                let y_val = parse_finite_f64_token(&row[c_y], "Cartn_y", &ctx)?;
                let z_val = parse_finite_f64_token(&row[c_z], "Cartn_z", &ctx)?;
                let pos = [
                    validate_finite_coord(x_val, 'X', &ctx)?,
                    validate_finite_coord(y_val, 'Y', &ctx)?,
                    validate_finite_coord(z_val, 'Z', &ctx)?,
                ];

                let occupancy = c_occ
                    .and_then(|c| row[c].parse::<f32>().ok())
                    .filter(|v| v.is_finite())
                    .unwrap_or(1.0);
                let b_factor = c_bfac
                    .and_then(|c| row[c].parse::<f32>().ok())
                    .filter(|v| v.is_finite())
                    .unwrap_or(0.0);
                let model_num = c_model
                    .and_then(|c| row[c].parse::<usize>().ok())
                    .filter(|&m| m > 0)
                    .unwrap_or(1);
                let raw_sym = c_symbol
                    .map(|c| row[c].as_str())
                    .filter(|&s| s != "." && s != "?")
                    .unwrap_or("");
                let element = infer_clean_element(raw_sym, &atom_name, &res_name, is_hetatm);

                let mut rec = RawAtomRecord {
                    model_num,
                    serial,
                    atom_name,
                    alt_loc,
                    res_name,
                    chain_id,
                    seq_id,
                    ins_code,
                    pos,
                    occupancy,
                    b_factor,
                    element,
                    is_hetatm,
                };
                rec.normalize_mse();
                if !is_solvent_res(&rec.res_name) {
                    raw_atoms.push(rec);
                }
            }
            continue;
        }
        idx += 1;
    }

    if raw_atoms.is_empty() {
        return Err("No valid _atom_site coordinates found in mmCIF input".to_string());
    }
    if title.is_empty() {
        title = "mmCIF Structure".to_string();
    }

    finalize_structure_ensemble(title, raw_atoms, &[])
}

fn finalize_structure_ensemble(
    title: String,
    raw_atoms: Vec<RawAtomRecord>,
    explicit_ss: &[SsSpan],
) -> Result<StructureEnsemble3d, String> {
    // Group raw records by `model_num` preserving encounter order
    let mut model_order: Vec<usize> = Vec::new();
    let mut per_model_raw: BTreeMap<usize, Vec<RawAtomRecord>> = BTreeMap::new();
    for rec in raw_atoms {
        if !per_model_raw.contains_key(&rec.model_num) {
            model_order.push(rec.model_num);
        }
        per_model_raw.entry(rec.model_num).or_default().push(rec);
    }

    let mut models: Vec<StructureModel3d> = Vec::with_capacity(model_order.len());

    for m_num in model_order {
        let Some(records) = per_model_raw.remove(&m_num) else {
            continue;
        };

        // Resolve altlocs per `(is_het_bucket, chain_id, seq_id, ins_code, atom_name)` while
        // preserving chain and residue encounter order.
        let mut chosen: Vec<RawAtomRecord> = Vec::with_capacity(records.len());
        let mut slot_map: BTreeMap<(bool, char, i32, char, String), usize> = BTreeMap::new();

        for rec in records {
            let is_polymer = !rec.is_hetatm && is_standard_amino_acid(&rec.res_name);
            let key = (
                !is_polymer,
                rec.chain_id,
                rec.seq_id,
                rec.ins_code,
                rec.atom_name.clone(),
            );
            if let Some(&slot_idx) = slot_map.get(&key) {
                if rec.altloc_priority() > chosen[slot_idx].altloc_priority() {
                    chosen[slot_idx] = rec;
                }
            } else {
                let idx = chosen.len();
                slot_map.insert(key, idx);
                chosen.push(rec);
            }
        }

        let mut chains: Vec<PolymerChain3d> = Vec::new();
        let mut het_atoms: Vec<StructureAtom3d> = Vec::new();

        for rec in chosen {
            let is_polymer = !rec.is_hetatm && is_standard_amino_acid(&rec.res_name);
            let is_metal = !is_polymer && is_metal_element(&rec.element);
            let atom = StructureAtom3d {
                serial: rec.serial,
                atom_name: rec.atom_name,
                element: rec.element,
                res_name: rec.res_name.clone(),
                chain_id: rec.chain_id,
                seq_id: rec.seq_id,
                ins_code: rec.ins_code,
                pos: rec.pos,
                occupancy: rec.occupancy,
                b_factor: rec.b_factor,
                is_hetatm: !is_polymer,
                is_metal,
            };

            if is_polymer {
                let chain_pos = chains
                    .iter()
                    .position(|c| c.chain_id == rec.chain_id)
                    .unwrap_or_else(|| {
                        chains.push(PolymerChain3d {
                            chain_id: rec.chain_id,
                            residues: Vec::new(),
                        });
                        chains.len() - 1
                    });
                let chain = &mut chains[chain_pos];
                let res_pos = chain
                    .residues
                    .iter()
                    .position(|r| r.seq_id == rec.seq_id && r.ins_code == rec.ins_code)
                    .unwrap_or_else(|| {
                        chain.residues.push(PolymerResidue3d {
                            chain_id: rec.chain_id,
                            seq_id: rec.seq_id,
                            ins_code: rec.ins_code,
                            res_name: rec.res_name,
                            sequence_idx: None,
                            ss: SecondaryStructureKind::Coil,
                            atoms: Vec::new(),
                        });
                        chain.residues.len() - 1
                    });
                chain.residues[res_pos].atoms.push(atom);
            } else {
                het_atoms.push(atom);
            }
        }

        // Assign secondary structure: Kabsch–Sander H-bonds + Ramachandran basins first,
        // overridden by explicit PDB HELIX/SHEET spans when present.
        for chain in &mut chains {
            compute_kabsch_sander_secondary_structure(&mut chain.residues);
            if !explicit_ss.is_empty() {
                for res in &mut chain.residues {
                    for span in explicit_ss {
                        if span.chain_id == chain.chain_id
                            && res.seq_id >= span.start_seq
                            && res.seq_id <= span.end_seq
                        {
                            res.ss = span.kind;
                        }
                    }
                }
            }
        }

        models.push(StructureModel3d {
            model_number: m_num,
            chains,
            het_atoms,
            kabsch_rmsd_to_ref: 0.0,
        });
    }

    if models.is_empty() {
        return Err("No valid structure models decoded".to_string());
    }

    // Bind `sequence_idx` on primary chain (`chain 0`) across all conformers
    let mut primary_sequence = Vec::new();
    let mut primary_seq_ids = Vec::new();
    if let Some(c0) = models[0].chains.first() {
        for (k, r) in c0.residues.iter().enumerate() {
            primary_sequence.push(r.res_name.clone());
            primary_seq_ids.push(r.seq_id);
            let _ = k;
        }
    }
    let primary_chain_id = models[0].chains.first().map(|c| c.chain_id);
    for model in &mut models {
        for chain in &mut model.chains {
            if Some(chain.chain_id) == primary_chain_id {
                for res in &mut chain.residues {
                    res.sequence_idx = primary_seq_ids.iter().position(|&sid| sid == res.seq_id);
                }
            }
        }
    }

    // Kabsch-superpose conformers `1..M` onto conformer `0` and compute per-residue `CA` RMSD
    let (per_residue_ca_rmsd, mean_ensemble_ca_rmsd) =
        superpose_ensemble_kabsch(&mut models, primary_sequence.len());

    // Compute bounding center & sphere radius on conformer `0`
    let (center, bounding_radius) = compute_model_bounds(&models[0]);

    let mut ensemble = StructureEnsemble3d {
        title,
        models,
        primary_sequence,
        primary_seq_ids,
        per_residue_ca_rmsd,
        mean_ensemble_ca_rmsd,
        center,
        bounding_radius,
        coord_lookup: AtomCoordLookup::default(),
    };
    ensemble.rebuild_coord_lookup();
    Ok(ensemble)
}

fn compute_model_bounds(model: &StructureModel3d) -> ([f64; 3], f64) {
    let mut sum = [0.0f64; 3];
    let mut count = 0usize;
    for chain in &model.chains {
        for res in &chain.residues {
            for a in &res.atoms {
                sum[0] += a.pos[0] as f64;
                sum[1] += a.pos[1] as f64;
                sum[2] += a.pos[2] as f64;
                count += 1;
            }
        }
    }
    for a in &model.het_atoms {
        sum[0] += a.pos[0] as f64;
        sum[1] += a.pos[1] as f64;
        sum[2] += a.pos[2] as f64;
        count += 1;
    }
    if count == 0 {
        return ([0.0, 0.0, 0.0], 15.0);
    }
    let center = [
        sum[0] / (count as f64),
        sum[1] / (count as f64),
        sum[2] / (count as f64),
    ];
    let mut max_r2 = 4.0f64;
    for chain in &model.chains {
        for res in &chain.residues {
            for a in &res.atoms {
                let dx = (a.pos[0] as f64) - center[0];
                let dy = (a.pos[1] as f64) - center[1];
                let dz = (a.pos[2] as f64) - center[2];
                max_r2 = max_r2.max(dx * dx + dy * dy + dz * dz);
            }
        }
    }
    for a in &model.het_atoms {
        let dx = (a.pos[0] as f64) - center[0];
        let dy = (a.pos[1] as f64) - center[1];
        let dz = (a.pos[2] as f64) - center[2];
        max_r2 = max_r2.max(dx * dx + dy * dy + dz * dz);
    }
    (center, max_r2.sqrt().max(5.0))
}

// ============================================================================
// 5. Kabsch & Sander (1983) Secondary Structure Assignment
// ============================================================================

#[inline]
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        (a[0] - b[0]) as f64,
        (a[1] - b[1]) as f64,
        (a[2] - b[2]) as f64,
    ]
}

#[inline]
fn vec3_len(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn vec3_norm(v: [f64; 3]) -> [f64; 3] {
    let l = vec3_len(v);
    if l > 1e-12 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[inline]
fn vec3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dihedral_deg(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f64 {
    let b1 = vec3_sub(p1, p0);
    let b2 = vec3_norm(vec3_sub(p2, p1));
    let b3 = vec3_sub(p3, p2);
    let n1 = vec3_cross(b1, b2);
    let n2 = vec3_cross(b2, b3);
    let m1 = vec3_cross(n1, b2);
    let x = vec3_dot(n1, n2);
    let y = vec3_dot(m1, n2);
    y.atan2(x).to_degrees()
}

/// Assign secondary structure (`Helix`, `Sheet`, `Coil`) directly from atomic coordinates using
/// Kabsch & Sander (*Biopolymers* 1983, 22, 2577–2637) electrostatic backbone hydrogen-bond energy
/// (`E < -0.5 kcal/mol`) with Lovell et al. (*Proteins* 2003) Ramachandran `(phi, psi)` fallback
/// so models lacking `HELIX`/`SHEET` records always render full cartoons.
pub fn compute_kabsch_sander_secondary_structure(residues: &mut [PolymerResidue3d]) {
    let n = residues.len();
    if n < 4 {
        return;
    }

    let mut c_pos = vec![None::<[f64; 3]>; n];
    let mut o_pos = vec![None::<[f64; 3]>; n];
    let mut n_pos = vec![None::<[f64; 3]>; n];
    let mut h_pos = vec![None::<[f64; 3]>; n];
    let mut ca_pos = vec![None::<[f32; 3]>; n];

    for i in 0..n {
        ca_pos[i] = residues[i].atom_pos("CA");
        c_pos[i] = residues[i]
            .atom_pos("C")
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]);
        o_pos[i] = residues[i]
            .atom_pos("O")
            .or_else(|| residues[i].atom_pos("OXT"))
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]);
        n_pos[i] = residues[i]
            .atom_pos("N")
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]);
        if residues[i].res_name != "PRO" {
            h_pos[i] = residues[i]
                .atom_pos("H")
                .or_else(|| residues[i].atom_pos("HN"))
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]);
        }
    }

    // Reconstruct ideal amide proton positions `r_H = r_N + 1.00 * unit(r_C(i-1) - r_O(i-1))`
    // (Kabsch & Sander 1983, p. 2580) when hydrogens are absent (X-ray / heavy-atom models).
    for i in 1..n {
        if residues[i].res_name == "PRO" || h_pos[i].is_some() {
            continue;
        }
        if let (Some(rn), Some(rc_prev), Some(ro_prev)) = (n_pos[i], c_pos[i - 1], o_pos[i - 1]) {
            let co_dir = vec3_norm([
                rc_prev[0] - ro_prev[0],
                rc_prev[1] - ro_prev[1],
                rc_prev[2] - ro_prev[2],
            ]);
            // Guard against chain breaks (`d(C_{i-1}, N_i) <= 2.2 Å`)
            let d_cn = vec3_len([rn[0] - rc_prev[0], rn[1] - rc_prev[1], rn[2] - rc_prev[2]]);
            if d_cn <= 2.2 {
                h_pos[i] = Some([rn[0] + co_dir[0], rn[1] + co_dir[1], rn[2] + co_dir[2]]);
            }
        }
    }

    // Electrostatic H-bond predicate `hbond(acceptor_co = i, donor_nh = j)` (`E < -0.5 kcal/mol`)
    let has_hbond = |acc_i: usize, don_j: usize| -> bool {
        if acc_i >= n || don_j >= n || acc_i.abs_diff(don_j) < 2 {
            return false;
        }
        if acc_i + 1 == don_j {
            return false;
        }
        let (Some(rc), Some(ro), Some(rn), Some(rh)) =
            (c_pos[acc_i], o_pos[acc_i], n_pos[don_j], h_pos[don_j])
        else {
            return false;
        };
        let dx_on = ro[0] - rn[0];
        let dy_on = ro[1] - rn[1];
        let dz_on = ro[2] - rn[2];
        let r_on2 = dx_on * dx_on + dy_on * dy_on + dz_on * dz_on;
        if !(1.0..=36.0).contains(&r_on2) {
            return false;
        }
        let r_on = r_on2.sqrt();
        let r_ch = vec3_len([rc[0] - rh[0], rc[1] - rh[1], rc[2] - rh[2]]).max(0.5);
        let r_oh = vec3_len([ro[0] - rh[0], ro[1] - rh[1], ro[2] - rh[2]]).max(0.5);
        let r_cn = vec3_len([rc[0] - rn[0], rc[1] - rn[1], rc[2] - rn[2]]).max(0.5);
        // `0.42 * 0.20 * 332.0 = 27.888 kcal*Å/mol`
        let energy = 27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);
        energy < -0.5
    };

    let any_carbonyl = o_pos.iter().filter(|o| o.is_some()).count() >= n / 2;
    let mut assigned = vec![SecondaryStructureKind::Coil; n];

    if any_carbonyl {
        // 1. Beta-bridges (parallel and antiparallel, Kabsch & Sander 1983, p. 2584)
        let mut is_bridge = vec![false; n];
        for i in 1..n.saturating_sub(1) {
            for j in (i + 3)..n.saturating_sub(1) {
                let parallel = (has_hbond(i - 1, j) && has_hbond(j, i + 1))
                    || (has_hbond(j - 1, i) && has_hbond(i, j + 1));
                let antiparallel = (has_hbond(i, j) && has_hbond(j, i))
                    || (has_hbond(i - 1, j + 1) && has_hbond(j - 1, i + 1));
                if parallel || antiparallel {
                    is_bridge[i] = true;
                    is_bridge[j] = true;
                }
            }
        }
        for i in 0..n {
            if is_bridge[i] || (i > 0 && i + 1 < n && is_bridge[i - 1] && is_bridge[i + 1]) {
                assigned[i] = SecondaryStructureKind::Sheet;
            }
        }

        // 2. Helical `n`-turns (`n = 4` alpha, `n = 3` 3_10, `n = 5` pi)
        for pitch in [4usize, 3usize, 5usize] {
            if n <= pitch {
                continue;
            }
            let mut turn = vec![false; n];
            for (i, slot) in turn.iter_mut().enumerate().take(n - pitch) {
                if has_hbond(i, i + pitch) {
                    *slot = true;
                }
            }
            for i in 1..(n - pitch) {
                if turn[i - 1] && turn[i] {
                    for slot in assigned.iter_mut().take(i + pitch).skip(i) {
                        if pitch == 4 || *slot == SecondaryStructureKind::Coil {
                            *slot = SecondaryStructureKind::Helix;
                        }
                    }
                }
            }
        }
    }

    // 3. Ramachandran `(phi, psi)` + `CA(i)..CA(i+3)` fallback for backbone-only / unrefined models
    let non_coil = assigned
        .iter()
        .filter(|&&s| s != SecondaryStructureKind::Coil)
        .count();
    if non_coil == 0 {
        let mut raw_rama = vec![SecondaryStructureKind::Coil; n];
        for i in 1..n.saturating_sub(1) {
            if let (Some(c_prev), Some(n_cur), Some(ca_cur), Some(c_cur), Some(n_next)) = (
                residues[i - 1].atom_pos("C"),
                residues[i].atom_pos("N"),
                residues[i].atom_pos("CA"),
                residues[i].atom_pos("C"),
                residues[i + 1].atom_pos("N"),
            ) {
                let phi = dihedral_deg(c_prev, n_cur, ca_cur, c_cur);
                let psi = dihedral_deg(n_cur, ca_cur, c_cur, n_next);
                if (-105.0..=-32.0).contains(&phi) && (-75.0..=-10.0).contains(&psi) {
                    raw_rama[i] = SecondaryStructureKind::Helix;
                } else if (-175.0..=-75.0).contains(&phi)
                    && ((90.0..=180.0).contains(&psi) || (-180.0..=-145.0).contains(&psi))
                {
                    raw_rama[i] = SecondaryStructureKind::Sheet;
                }
            } else if i + 3 < n {
                if let (Some(ca0), Some(ca3)) = (ca_pos[i], ca_pos[i + 3]) {
                    let d = vec3_len(vec3_sub(ca3, ca0));
                    if (4.7..=5.7).contains(&d) {
                        raw_rama[i] = SecondaryStructureKind::Helix;
                    } else if d >= 9.2 {
                        raw_rama[i] = SecondaryStructureKind::Sheet;
                    }
                }
            }
        }
        // Require >= 3 consecutive helical or >= 2 consecutive beta residues to suppress noise
        for i in 0..n {
            if raw_rama[i] == SecondaryStructureKind::Helix {
                let run = (i >= 2
                    && raw_rama[i - 2] == SecondaryStructureKind::Helix
                    && raw_rama[i - 1] == SecondaryStructureKind::Helix)
                    || (i >= 1
                        && i + 1 < n
                        && raw_rama[i - 1] == SecondaryStructureKind::Helix
                        && raw_rama[i + 1] == SecondaryStructureKind::Helix)
                    || (i + 2 < n
                        && raw_rama[i + 1] == SecondaryStructureKind::Helix
                        && raw_rama[i + 2] == SecondaryStructureKind::Helix);
                if run {
                    assigned[i] = SecondaryStructureKind::Helix;
                }
            } else if raw_rama[i] == SecondaryStructureKind::Sheet {
                let run = (i >= 1 && raw_rama[i - 1] == SecondaryStructureKind::Sheet)
                    || (i + 1 < n && raw_rama[i + 1] == SecondaryStructureKind::Sheet);
                if run {
                    assigned[i] = SecondaryStructureKind::Sheet;
                }
            }
        }
    }

    for (res, ss) in residues.iter_mut().zip(assigned) {
        res.ss = ss;
    }
}

// ============================================================================
// 6. Kabsch (1976) / Coutsias–Seok–Dill (2004) Quaternion Superposition
// ============================================================================

/// Compute the optimal proper rotation `R` (`det(R) = +1`), centroids, and coordinate RMSD (`Å`)
/// aligning `moving` onto `reference` (`R * (p - c_mov) + c_ref`) via the `4x4` quaternion matrix.
pub fn kabsch_superpose_points(
    reference: &[[f64; 3]],
    moving: &[[f64; 3]],
) -> ([[f64; 3]; 3], [f64; 3], [f64; 3], f64) {
    let n = reference.len().min(moving.len());
    if n < 3 {
        return (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            [0.0; 3],
            0.0,
        );
    }

    let inv_n = 1.0 / (n as f64);
    let mut c_ref = [0.0f64; 3];
    let mut c_mov = [0.0f64; 3];
    for k in 0..n {
        for d in 0..3 {
            c_ref[d] += reference[k][d];
            c_mov[d] += moving[k][d];
        }
    }
    for d in 0..3 {
        c_ref[d] *= inv_n;
        c_mov[d] *= inv_n;
    }

    // Cross-covariance `H_{ij} = sum_k (mov_k[i] - c_mov[i]) * (ref_k[j] - c_ref[j])`
    let mut h = [[0.0f64; 3]; 3];
    for k in 0..n {
        let xm = [
            moving[k][0] - c_mov[0],
            moving[k][1] - c_mov[1],
            moving[k][2] - c_mov[2],
        ];
        let yr = [
            reference[k][0] - c_ref[0],
            reference[k][1] - c_ref[1],
            reference[k][2] - c_ref[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += xm[i] * yr[j];
            }
        }
    }

    // Symmetric 4x4 Davenport/Coutsias quaternion matrix `F` (maximizing `q^T F q`)
    let mut f = [
        [
            h[0][0] + h[1][1] + h[2][2],
            h[1][2] - h[2][1],
            h[2][0] - h[0][2],
            h[0][1] - h[1][0],
        ],
        [
            h[1][2] - h[2][1],
            h[0][0] - h[1][1] - h[2][2],
            h[0][1] + h[1][0],
            h[0][2] + h[2][0],
        ],
        [
            h[2][0] - h[0][2],
            h[0][1] + h[1][0],
            -h[0][0] + h[1][1] - h[2][2],
            h[1][2] + h[2][1],
        ],
        [
            h[0][1] - h[1][0],
            h[0][2] + h[2][0],
            h[1][2] + h[2][1],
            -h[0][0] - h[1][1] + h[2][2],
        ],
    ];
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Cyclic Jacobi eigensolver on 4x4 symmetric `F`
    for _sweep in 0..32 {
        let mut max_off = 0.0f64;
        for p in 0..4 {
            for q in (p + 1)..4 {
                max_off = max_off.max(f[p][q].abs());
                if f[p][q].abs() < 1e-15 {
                    continue;
                }
                let diff = f[q][q] - f[p][p];
                let t = if diff.abs() < 1e-15 {
                    1.0
                } else {
                    let phi = diff / (2.0 * f[p][q]);
                    let sign = if phi >= 0.0 { 1.0 } else { -1.0 };
                    sign / (phi.abs() + (1.0 + phi * phi).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                let fpq = f[p][q];
                f[p][q] = 0.0;
                f[q][p] = 0.0;
                f[p][p] -= t * fpq;
                f[q][q] += t * fpq;

                for r in 0..4 {
                    if r != p && r != q {
                        let frp = f[r][p];
                        let frq = f[r][q];
                        f[r][p] = frp - s * (frq + tau * frp);
                        f[p][r] = f[r][p];
                        f[r][q] = frq + s * (frp - tau * frq);
                        f[q][r] = f[r][q];
                    }
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = vrp - s * (vrq + tau * vrp);
                    v[r][q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
    }

    let mut best_col = 0usize;
    for col in 1..4 {
        if f[col][col] > f[best_col][best_col] {
            best_col = col;
        }
    }
    let mut q = [
        v[0][best_col],
        v[1][best_col],
        v[2][best_col],
        v[3][best_col],
    ];
    let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if qn > 1e-15 {
        for c in &mut q {
            *c /= qn;
        }
    } else {
        q = [1.0, 0.0, 0.0, 0.0];
    }

    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    let rot = [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ];

    let mut sq_err = 0.0f64;
    for k in 0..n {
        let xm = [
            moving[k][0] - c_mov[0],
            moving[k][1] - c_mov[1],
            moving[k][2] - c_mov[2],
        ];
        for r in 0..3 {
            let aligned = rot[r][0] * xm[0] + rot[r][1] * xm[1] + rot[r][2] * xm[2] + c_ref[r];
            let d = aligned - reference[k][r];
            sq_err += d * d;
        }
    }
    let rmsd = (sq_err * inv_n).sqrt();
    (rot, c_ref, c_mov, rmsd)
}

fn collect_ca_map(model: &StructureModel3d) -> BTreeMap<(char, i32, char), [f64; 3]> {
    let mut out = BTreeMap::new();
    for chain in &model.chains {
        for res in &chain.residues {
            if let Some(ca) = res.atom_pos("CA") {
                out.insert(
                    (chain.chain_id, res.seq_id, res.ins_code),
                    [ca[0] as f64, ca[1] as f64, ca[2] as f64],
                );
            }
        }
    }
    out
}

fn superpose_ensemble_kabsch(
    models: &mut [StructureModel3d],
    primary_len: usize,
) -> (Vec<f64>, f64) {
    if models.len() <= 1 {
        return (vec![0.0; primary_len], 0.0);
    }

    let ref_ca = collect_ca_map(&models[0]);
    for m in models.iter_mut().skip(1) {
        let mov_ca = collect_ca_map(m);
        let mut ref_pts = Vec::with_capacity(ref_ca.len());
        let mut mov_pts = Vec::with_capacity(ref_ca.len());
        for (key, &rp) in &ref_ca {
            if let Some(&mp) = mov_ca.get(key) {
                ref_pts.push(rp);
                mov_pts.push(mp);
            }
        }
        if ref_pts.len() < 3 {
            continue;
        }
        let (rot, c_ref, c_mov, rmsd) = kabsch_superpose_points(&ref_pts, &mov_pts);
        m.kabsch_rmsd_to_ref = rmsd;

        let apply_xform = |p: &mut [f32; 3]| {
            let xm = [
                (p[0] as f64) - c_mov[0],
                (p[1] as f64) - c_mov[1],
                (p[2] as f64) - c_mov[2],
            ];
            p[0] = (rot[0][0] * xm[0] + rot[0][1] * xm[1] + rot[0][2] * xm[2] + c_ref[0]) as f32;
            p[1] = (rot[1][0] * xm[0] + rot[1][1] * xm[1] + rot[1][2] * xm[2] + c_ref[1]) as f32;
            p[2] = (rot[2][0] * xm[0] + rot[2][1] * xm[1] + rot[2][2] * xm[2] + c_ref[2]) as f32;
        };

        for chain in &mut m.chains {
            for res in &mut chain.residues {
                for atom in &mut res.atoms {
                    apply_xform(&mut atom.pos);
                }
            }
        }
        for atom in &mut m.het_atoms {
            apply_xform(&mut atom.pos);
        }
    }

    // Compute per-residue `CA` spatial RMSD across all superposed conformers on primary chain
    let mut per_res_rmsd = vec![0.0f64; primary_len];
    let mut valid_res = 0usize;
    let mut rmsd_sum = 0.0f64;

    for (k, rmsd_slot) in per_res_rmsd.iter_mut().enumerate() {
        let mut pts: Vec<[f64; 3]> = Vec::with_capacity(models.len());
        for m in models.iter() {
            if let Some(c0) = m.chains.first() {
                if let Some(ca) = c0
                    .residues
                    .iter()
                    .find(|r| r.sequence_idx == Some(k))
                    .and_then(|r| r.atom_pos("CA"))
                {
                    pts.push([ca[0] as f64, ca[1] as f64, ca[2] as f64]);
                }
            }
        }
        if pts.len() >= 2 {
            let inv = 1.0 / (pts.len() as f64);
            let mean = [
                pts.iter().map(|p| p[0]).sum::<f64>() * inv,
                pts.iter().map(|p| p[1]).sum::<f64>() * inv,
                pts.iter().map(|p| p[2]).sum::<f64>() * inv,
            ];
            let var = pts
                .iter()
                .map(|p| {
                    let dx = p[0] - mean[0];
                    let dy = p[1] - mean[1];
                    let dz = p[2] - mean[2];
                    dx * dx + dy * dy + dz * dz
                })
                .sum::<f64>()
                * inv;
            let r = var.sqrt();
            *rmsd_slot = r;
            rmsd_sum += r;
            valid_res += 1;
        }
    }

    let mean_rmsd = if valid_res > 0 {
        rmsd_sum / (valid_res as f64)
    } else {
        0.0
    };
    (per_res_rmsd, mean_rmsd)
}

// ============================================================================
// 7. Quaternion Arcball Camera & Perspective / Orthographic Projection
// ============================================================================

/// Viewport projection model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ProjectionMode {
    #[default]
    Perspective,
    Orthographic,
}

impl ProjectionMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Perspective => "Perspective",
            Self::Orthographic => "Orthographic",
        }
    }
}

/// Quaternion arcball 3D camera with view-space XY pan, zoom, Z-slab clipping, and exact
/// analytical ray unprojection (`screen_to_world_ray`).
#[derive(Clone, Debug)]
pub struct ArcballCamera {
    pub target: [f64; 3],
    /// Unit quaternion `[x, y, z, w]` mapping world vectors into camera orientation.
    pub orientation: [f64; 4],
    pub distance: f64,
    pub pan_xy: [f64; 2],
    pub fov_y_rad: f64,
    pub projection: ProjectionMode,
    pub slab_near_offset: f64,
    pub slab_far_offset: f64,
    pub depth_fog_enabled: bool,
}

impl Default for ArcballCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            distance: 45.0,
            pan_xy: [0.0, 0.0],
            fov_y_rad: 38.0f64.to_radians(),
            projection: ProjectionMode::Perspective,
            slab_near_offset: -35.0,
            slab_far_offset: 35.0,
            depth_fog_enabled: true,
        }
    }
}

#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline]
fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1e-15 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[inline]
fn shoemake_arcball_point(nx: f64, ny: f64) -> [f64; 3] {
    let d2 = nx * nx + ny * ny;
    if d2 <= 1.0 {
        [nx, ny, (1.0 - d2).sqrt()]
    } else {
        let inv = 1.0 / d2.sqrt();
        [nx * inv, ny * inv, 0.0]
    }
}

impl ArcballCamera {
    /// Frame `ensemble` comfortably within the viewport and reset slab offsets around it.
    pub fn fit_ensemble(&mut self, ensemble: &StructureEnsemble3d) {
        self.target = ensemble.center;
        self.pan_xy = [0.0, 0.0];
        let half_tan = (0.5 * self.fov_y_rad).tan().max(0.15);
        self.distance = (ensemble.bounding_radius / half_tan * 1.18).max(8.0);
        let pad = (ensemble.bounding_radius * 1.85).max(18.0);
        self.slab_near_offset = -pad;
        self.slab_far_offset = pad;
    }

    /// Smoothly recenter `target` onto `pos` without disturbing orientation or zoom.
    pub fn focus_point(&mut self, pos: [f32; 3]) {
        self.target = [pos[0] as f64, pos[1] as f64, pos[2] as f64];
        self.pan_xy = [0.0, 0.0];
    }

    /// Rotate the camera using Shoemake (1985) virtual sphere drag from normalized `[-1, 1]`
    /// viewport coordinates `prev_ndc` to `curr_ndc`.
    pub fn drag_arcball(&mut self, prev_ndc: [f64; 2], curr_ndc: [f64; 2]) {
        let p0 = shoemake_arcball_point(prev_ndc[0], prev_ndc[1]);
        let p1 = shoemake_arcball_point(curr_ndc[0], curr_ndc[1]);
        let axis = vec3_cross(p0, p1);
        let dot = vec3_dot(p0, p1).clamp(-1.0, 1.0);
        let dq = quat_normalize([axis[0], axis[1], axis[2], 1.0 + dot]);
        self.orientation = quat_normalize(quat_mul(dq, self.orientation));
    }

    /// Rotate around world/view Y & X axes by `(yaw_rad, pitch_rad)` (useful in automated turntables).
    pub fn orbit_angles(&mut self, yaw_rad: f64, pitch_rad: f64) {
        let (sy, cy) = (0.5 * yaw_rad).sin_cos();
        let (sp, cp) = (0.5 * pitch_rad).sin_cos();
        let q_yaw = [0.0, sy, 0.0, cy];
        let q_pitch = [sp, 0.0, 0.0, cp];
        let dq = quat_mul(q_pitch, q_yaw);
        self.orientation = quat_normalize(quat_mul(dq, self.orientation));
    }

    /// Pan in view-space XY plane according to pixel drag `(dx_px, dy_px)`.
    pub fn drag_pan(&mut self, dx_px: f64, dy_px: f64, viewport_height_px: f64) {
        let scale = (2.0 * self.ortho_half_height()) / viewport_height_px.max(1.0);
        self.pan_xy[0] += dx_px * scale;
        self.pan_xy[1] -= dy_px * scale;
    }

    /// Multiplicative dolly/zoom factor (`factor < 1` zooms in, `factor > 1` zooms out).
    pub fn zoom_by(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(2.5, 2500.0);
    }

    #[inline]
    pub fn ortho_half_height(&self) -> f64 {
        (self.distance * (0.5 * self.fov_y_rad).tan()).max(0.5)
    }

    #[inline]
    pub fn slab_planes(&self) -> (f64, f64) {
        let z_near = (self.distance + self.slab_near_offset).max(0.25);
        let z_far = (self.distance + self.slab_far_offset).max(z_near + 0.50);
        (z_near, z_far)
    }

    /// `3x3` orthogonal rotation matrix `R(q)` (`row-major`).
    #[inline]
    pub fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        let [x, y, z, w] = quat_normalize(self.orientation);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Transform world-space point `p` into right-handed camera view space (`-Z` forward).
    #[inline]
    pub fn world_to_view(&self, p: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix();
        let d = [
            p[0] - self.target[0],
            p[1] - self.target[1],
            p[2] - self.target[2],
        ];
        [
            r[0][0] * d[0] + r[0][1] * d[1] + r[0][2] * d[2] + self.pan_xy[0],
            r[1][0] * d[0] + r[1][1] * d[1] + r[1][2] * d[2] + self.pan_xy[1],
            r[2][0] * d[0] + r[2][1] * d[1] + r[2][2] * d[2] - self.distance,
        ]
    }

    /// Transform view-space point `pv` back into world space (`exact inverse` of `world_to_view`).
    #[inline]
    pub fn view_to_world(&self, pv: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix();
        let s = [
            pv[0] - self.pan_xy[0],
            pv[1] - self.pan_xy[1],
            pv[2] + self.distance,
        ];
        [
            self.target[0] + r[0][0] * s[0] + r[1][0] * s[1] + r[2][0] * s[2],
            self.target[1] + r[0][1] * s[0] + r[1][1] * s[1] + r[2][1] * s[2],
            self.target[2] + r[0][2] * s[0] + r[1][2] * s[1] + r[2][2] * s[2],
        ]
    }

    /// Project world-space point `p` to viewport pixel coordinates `(px_x, px_y, view_depth)`
    /// (`None` if clipped outside the Z-slab).
    pub fn project_world_to_screen(
        &self,
        p: [f64; 3],
        vp_width: f64,
        vp_height: f64,
    ) -> Option<(f64, f64, f64)> {
        let pv = self.world_to_view(p);
        let depth = -pv[2];
        let (z_near, z_far) = self.slab_planes();
        if depth < z_near || depth > z_far {
            return None;
        }
        let aspect = (vp_width / vp_height.max(1.0)).max(1e-4);
        let (ndc_x, ndc_y) = match self.projection {
            ProjectionMode::Perspective => {
                let f = 1.0 / (0.5 * self.fov_y_rad).tan();
                ((pv[0] * f / aspect) / depth, (pv[1] * f) / depth)
            }
            ProjectionMode::Orthographic => {
                let top = self.ortho_half_height();
                let right = top * aspect;
                (pv[0] / right, pv[1] / top)
            }
        };
        let px_x = 0.5 * (ndc_x + 1.0) * vp_width;
        let px_y = 0.5 * (1.0 - ndc_y) * vp_height;
        Some((px_x, px_y, depth))
    }

    /// Unproject viewport pixel `(px_x, px_y)` into an exact world-space [` crate::mol3d_overlay::Ray3d `]
    /// bounded by the active front/back Z-slab planes (`[t_min, t_max]`).
    pub fn screen_to_world_ray(
        &self,
        px_x: f64,
        px_y: f64,
        vp_width: f64,
        vp_height: f64,
    ) -> crate::mol3d_overlay::Ray3d {
        let w = vp_width.max(1.0);
        let h = vp_height.max(1.0);
        let aspect = w / h;
        let ndc_x = (2.0 * px_x / w) - 1.0;
        let ndc_y = 1.0 - (2.0 * px_y / h);
        let (z_near, z_far) = self.slab_planes();
        let r = self.rotation_matrix();

        let rotate_inv = |dv: [f64; 3]| -> [f64; 3] {
            [
                r[0][0] * dv[0] + r[1][0] * dv[1] + r[2][0] * dv[2],
                r[0][1] * dv[0] + r[1][1] * dv[1] + r[2][1] * dv[2],
                r[0][2] * dv[0] + r[1][2] * dv[1] + r[2][2] * dv[2],
            ]
        };

        match self.projection {
            ProjectionMode::Perspective => {
                let f = 1.0 / (0.5 * self.fov_y_rad).tan();
                let raw_dv = [ndc_x * aspect / f, ndc_y / f, -1.0];
                let len = vec3_len(raw_dv).max(1e-12);
                let unit_dv = [raw_dv[0] / len, raw_dv[1] / len, raw_dv[2] / len];
                crate::mol3d_overlay::Ray3d {
                    origin: self.view_to_world([0.0, 0.0, 0.0]),
                    dir: vec3_norm(rotate_inv(unit_dv)),
                    t_min: z_near * len,
                    t_max: z_far * len,
                }
            }
            ProjectionMode::Orthographic => {
                let top = self.ortho_half_height();
                let right = top * aspect;
                let ov = [ndc_x * right, ndc_y * top, 0.0];
                crate::mol3d_overlay::Ray3d {
                    origin: self.view_to_world(ov),
                    dir: vec3_norm(rotate_inv([0.0, 0.0, -1.0])),
                    t_min: z_near,
                    t_max: z_far,
                }
            }
        }
    }

    /// Build column-major `view` and `proj` `4x4` matrices plus lighting/slab uniform block for
    /// `mol3d.wgsl` (`z_clip in [0, 1]`).
    pub fn to_gpu_uniforms(
        &self,
        vp_width: f32,
        vp_height: f32,
        bg_rgba: [f32; 4],
    ) -> Mol3dUniforms {
        let r = self.rotation_matrix();
        // Translation in view space: `t = -R * target + [pan_x, pan_y, -distance]`
        let tx = -(r[0][0] * self.target[0] + r[0][1] * self.target[1] + r[0][2] * self.target[2])
            + self.pan_xy[0];
        let ty = -(r[1][0] * self.target[0] + r[1][1] * self.target[1] + r[1][2] * self.target[2])
            + self.pan_xy[1];
        let tz = -(r[2][0] * self.target[0] + r[2][1] * self.target[1] + r[2][2] * self.target[2])
            - self.distance;

        // Column-major `mat4x4<f32>`
        let view = [
            [r[0][0] as f32, r[1][0] as f32, r[2][0] as f32, 0.0],
            [r[0][1] as f32, r[1][1] as f32, r[2][1] as f32, 0.0],
            [r[0][2] as f32, r[1][2] as f32, r[2][2] as f32, 0.0],
            [tx as f32, ty as f32, tz as f32, 1.0],
        ];

        let aspect = (vp_width.max(1.0) / vp_height.max(1.0)) as f64;
        let (z_near, z_far) = self.slab_planes();
        let proj = match self.projection {
            ProjectionMode::Perspective => {
                let f = 1.0 / (0.5 * self.fov_y_rad).tan();
                let z_scale = z_far / (z_near - z_far);
                let z_bias = (z_near * z_far) / (z_near - z_far);
                [
                    [(f / aspect) as f32, 0.0, 0.0, 0.0],
                    [0.0, f as f32, 0.0, 0.0],
                    [0.0, 0.0, z_scale as f32, -1.0],
                    [0.0, 0.0, z_bias as f32, 0.0],
                ]
            }
            ProjectionMode::Orthographic => {
                let top = self.ortho_half_height();
                let right = top * aspect;
                let span = (z_far - z_near).max(1e-3);
                [
                    [(1.0 / right) as f32, 0.0, 0.0, 0.0],
                    [0.0, (1.0 / top) as f32, 0.0, 0.0],
                    [0.0, 0.0, (-1.0 / span) as f32, 0.0],
                    [0.0, 0.0, (-z_near / span) as f32, 1.0],
                ]
            }
        };

        let (fog_start, fog_end) = if self.depth_fog_enabled {
            let span = z_far - z_near;
            (z_near + 0.38 * span, z_far + 0.15 * span)
        } else {
            (1.0e5, 1.0e5 + 100.0)
        };

        Mol3dUniforms {
            view,
            proj,
            bg_color: bg_rgba,
            slab_fog: [
                z_near as f32,
                z_far as f32,
                fog_start as f32,
                fog_end as f32,
            ],
            light_dir: [0.38, 0.52, 0.76, 36.0],
            params: [
                0.24,
                0.70,
                0.26,
                if self.projection == ProjectionMode::Orthographic {
                    1.0
                } else {
                    0.0
                },
            ],
        }
    }
}

// ============================================================================
// 8. Carson & Bugg (1986) Cubic Ribbon Mesh + Instanced Atoms/Bonds Builder
// ============================================================================

/// GPU uniform struct (`192` bytes, 16-byte aligned) matching `Mol3dUniforms` in `mol3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Mol3dUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub bg_color: [f32; 4],
    pub slab_fog: [f32; 4],
    pub light_dir: [f32; 4],
    pub params: [f32; 4],
}

/// Per-vertex ribbon attribute (`40` bytes) matching `vs_ribbon` in `mol3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RibbonVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Hardware-instanced sphere descriptor (`32` bytes) matching `vs_sphere` in `mol3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SphereInstance {
    pub center_radius: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TemplatePosVertex {
    pos: [f32; 3],
}

const RIBBON_RING_VERTS: usize = 8;
const RIBBON_SUBDIVS: usize = 4;

/// Cached spine cross-section frame along the Carson–Bugg cubic spline enabling `< 0.2 ms`
/// quantitative putty radius and colormap updates without re-running spline tangents!
#[derive(Clone, Debug)]
struct CachedRibbonRing {
    vbo_start: usize,
    center: [f32; 3],
    normal_u: [f32; 3],
    binormal_v: [f32; 3],
    seq_idx: Option<usize>,
    ss: SecondaryStructureKind,
    ensemble_rmsd: Option<f64>,
    /// `(half_width_helical_or_sheet, half_thickness)` when `putty_enabled == false`.
    cartoon_half_axes: [f32; 2],
    alpha: f32,
    is_highlighted: bool,
}

/// Complete GPU-ready geometry buffers (opaque + alpha-blended ensemble conformer bundles) and
/// analytical 3D picking primitives.
#[derive(Clone, Debug, Default)]
pub struct Mol3dSceneBuffers {
    pub scene_gen: u64,
    pub opaque_ribbon_verts: Vec<RibbonVertex>,
    pub opaque_ribbon_indices: Vec<u32>,
    pub blend_ribbon_verts: Vec<RibbonVertex>,
    pub blend_ribbon_indices: Vec<u32>,
    pub opaque_spheres: Vec<SphereInstance>,
    pub blend_spheres: Vec<SphereInstance>,
    pub opaque_cylinders: Vec<CylinderInstance>,
    pub blend_cylinders: Vec<CylinderInstance>,
    pub restraint_cylinders: Vec<CylinderInstance>,
    pub pick_spheres: Vec<PickableSphere3d>,
    pub pick_ribbons: Vec<PickableRibbonSegment3d>,
    pub pick_restraints: Vec<PickableRestraint3d>,
    opaque_rings: Vec<CachedRibbonRing>,
    blend_rings: Vec<CachedRibbonRing>,
}

#[inline]
fn cpk_element_rgba(elem: &str, is_metal: bool, alpha: f32) -> [f32; 4] {
    if is_metal {
        let rgb = match elem {
            "ZN" => [0.54, 0.62, 0.86],
            "FE" => [0.88, 0.44, 0.20],
            "CU" => [0.82, 0.52, 0.26],
            "MG" => [0.36, 0.86, 0.36],
            "CA" => [0.32, 0.90, 0.58],
            "MN" => [0.64, 0.48, 0.84],
            "CO" | "NI" => [0.38, 0.72, 0.78],
            _ => [0.76, 0.64, 0.92],
        };
        return [rgb[0], rgb[1], rgb[2], alpha];
    }
    let rgb = match elem {
        "C" => [0.74, 0.78, 0.84],
        "N" => [0.26, 0.54, 0.96],
        "O" => [0.94, 0.28, 0.30],
        "S" | "SE" => [0.94, 0.80, 0.22],
        "P" => [0.96, 0.56, 0.18],
        "H" | "D" => [0.92, 0.95, 0.98],
        _ => [0.68, 0.78, 0.74],
    };
    [rgb[0], rgb[1], rgb[2], alpha]
}

#[inline]
fn vdw_radius_angstrom(elem: &str, is_metal: bool) -> f32 {
    if is_metal {
        return 1.45;
    }
    match elem {
        "H" | "D" => 1.10,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        _ => 1.65,
    }
}

#[inline]
fn catmull_rom_pos_tan(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    t: f64,
) -> ([f64; 3], [f64; 3]) {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut pos = [0.0f64; 3];
    let mut tan = [0.0f64; 3];
    for d in 0..3 {
        let a0 = 2.0 * p1[d];
        let a1 = -p0[d] + p2[d];
        let a2 = 2.0 * p0[d] - 5.0 * p1[d] + 4.0 * p2[d] - p3[d];
        let a3 = -p0[d] + 3.0 * p1[d] - 3.0 * p2[d] + p3[d];
        pos[d] = 0.5 * (a0 + a1 * t + a2 * t2 + a3 * t3);
        tan[d] = 0.5 * (a1 + 2.0 * a2 * t + 3.0 * a3 * t2);
    }
    (pos, vec3_norm(tan))
}

#[inline]
fn write_ring_vertices(
    verts: &mut [RibbonVertex],
    ring: &CachedRibbonRing,
    track_data: &QuantitativeTrackData,
) {
    let style = track_data.evaluate_residue_style(
        ring.seq_idx,
        ring.ss.default_rgba(),
        0.28,
        ring.ensemble_rmsd,
        ring.alpha,
    );
    let rgba = if ring.is_highlighted {
        [1.0, 0.92, 0.32, 1.0]
    } else {
        style.rgba
    };
    let (hw, ht) = if track_data.putty_enabled {
        (style.radius_angstrom, style.radius_angstrom)
    } else {
        let scale = if ring.is_highlighted { 1.18 } else { 1.0 };
        (
            ring.cartoon_half_axes[0] * scale,
            ring.cartoon_half_axes[1] * scale,
        )
    };

    for k in 0..RIBBON_RING_VERTS {
        let theta = (k as f32) * std::f32::consts::TAU / (RIBBON_RING_VERTS as f32);
        let (st, ct) = theta.sin_cos();
        let px = ring.center[0] + hw * ct * ring.binormal_v[0] + ht * st * ring.normal_u[0];
        let py = ring.center[1] + hw * ct * ring.binormal_v[1] + ht * st * ring.normal_u[1];
        let pz = ring.center[2] + hw * ct * ring.binormal_v[2] + ht * st * ring.normal_u[2];

        // Elliptical outward normal `(ct / hw) * v + (st / ht) * u`
        let inv_w = 1.0 / hw.max(0.04);
        let inv_t = 1.0 / ht.max(0.04);
        let nx_raw = ct * inv_w * ring.binormal_v[0] + st * inv_t * ring.normal_u[0];
        let ny_raw = ct * inv_w * ring.binormal_v[1] + st * inv_t * ring.normal_u[1];
        let nz_raw = ct * inv_w * ring.binormal_v[2] + st * inv_t * ring.normal_u[2];
        let nl = (nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw)
            .sqrt()
            .max(1e-6);

        verts[ring.vbo_start + k] = RibbonVertex {
            pos: [px, py, pz],
            normal: [nx_raw / nl, ny_raw / nl, nz_raw / nl],
            color: rgba,
        };
    }
}

/// Configuration controlling how a [`StructureEnsemble3d`] is compiled into [`Mol3dSceneBuffers`].
#[derive(Clone, Debug)]
pub struct SceneBuildConfig {
    pub style: RepresentationStyle,
    pub show_all_ensemble_models: bool,
    pub active_model_idx: usize,
    pub ensemble_bundle_opacity: f32,
    pub show_hydrogens: bool,
    pub show_metal_cofactors: bool,
    pub highlighted_seq_idx: Option<usize>,
    pub highlighted_seq_id: Option<i32>,
}

impl Default for SceneBuildConfig {
    fn default() -> Self {
        Self {
            style: RepresentationStyle::CartoonAndSidechains,
            show_all_ensemble_models: true,
            active_model_idx: 0,
            ensemble_bundle_opacity: 0.28,
            show_hydrogens: false,
            show_metal_cofactors: true,
            highlighted_seq_idx: None,
            highlighted_seq_id: None,
        }
    }
}

impl Mol3dSceneBuffers {
    /// Update ribbon putty cross-section radii and colormap RGBA values in-place in `< 0.5 ms`
    /// using cached Carson–Bugg spine frames (`zero` spline/bond recomputation!).
    pub fn update_quantitative_putty(&mut self, track_data: &QuantitativeTrackData) {
        for ring in &self.opaque_rings {
            write_ring_vertices(&mut self.opaque_ribbon_verts, ring, track_data);
        }
        for ring in &self.blend_rings {
            write_ring_vertices(&mut self.blend_ribbon_verts, ring, track_data);
        }
        self.scene_gen = self.scene_gen.wrapping_add(1);
    }

    /// Replace 3D NOE/PRE restraint cylinders and pickables in `< 2.0 ms`.
    pub fn set_restraint_overlay(&mut self, overlay: &RestraintOverlayBuild) {
        self.restraint_cylinders.clone_from(&overlay.cylinders);
        self.pick_restraints.clone_from(&overlay.pickables);
        self.scene_gen = self.scene_gen.wrapping_add(1);
    }
}

/// Compile a [`StructureEnsemble3d`] into hardware-instanced [`Mol3dSceneBuffers`].
pub fn build_mol3d_scene(
    ensemble: &StructureEnsemble3d,
    cfg: &SceneBuildConfig,
    track_data: &QuantitativeTrackData,
    restraint_overlay: Option<&RestraintOverlayBuild>,
    next_gen: u64,
) -> Mol3dSceneBuffers {
    let mut out = Mol3dSceneBuffers {
        scene_gen: next_gen,
        ..Default::default()
    };

    if ensemble.models.is_empty() {
        return out;
    }

    let primary_idx = cfg.active_model_idx.min(ensemble.models.len() - 1);
    let model_indices: Vec<usize> = if cfg.show_all_ensemble_models && ensemble.models.len() > 1 {
        (0..ensemble.models.len()).collect()
    } else {
        vec![primary_idx]
    };

    let draw_cartoon = matches!(
        cfg.style,
        RepresentationStyle::Cartoon | RepresentationStyle::CartoonAndSidechains
    );
    let draw_sidechain_sticks = cfg.style == RepresentationStyle::CartoonAndSidechains;
    let draw_all_sticks = matches!(
        cfg.style,
        RepresentationStyle::LicoriceSticks | RepresentationStyle::BallAndStick
    );
    let draw_atom_spheres = matches!(
        cfg.style,
        RepresentationStyle::BallAndStick | RepresentationStyle::Spacefill
    );

    for &m_idx in &model_indices {
        let model = &ensemble.models[m_idx];
        let is_primary = m_idx == primary_idx;
        let alpha = if is_primary {
            1.0
        } else {
            cfg.ensemble_bundle_opacity.clamp(0.06, 0.92)
        };
        let is_translucent = alpha < 0.98;

        // 1. Carson & Bugg (1986) Cubic Ribbon Extrusion along each contiguous polypeptide segment
        if draw_cartoon {
            for chain in &model.chains {
                append_chain_cartoon_ribbon(
                    chain,
                    m_idx,
                    is_primary,
                    alpha,
                    is_translucent,
                    ensemble,
                    cfg,
                    track_data,
                    &mut out,
                );
            }
        }

        // 2. Covalent Atom Spheres & Half-Bond CPK Cylinders
        // Note: on 20-conformer ensembles in `CartoonAndSidechains` mode, render full cartoon bundle
        // across all conformers and detailed sidechain sticks on the primary conformer (or selected
        // residue across all conformers) unless an explicit all-atom mode is chosen, keeping visual
        // clarity high and draw buffers lean while still supporting full 48,000-atom rendering!
        let emit_sticks_for_model = draw_all_sticks
            || draw_atom_spheres
            || (draw_sidechain_sticks && (is_primary || ensemble.models.len() <= 5));

        for chain in &model.chains {
            let mut prev_c_atom: Option<(i32, [f32; 3], [f32; 4])> = None;
            for res in &chain.residues {
                let is_hi = (cfg.highlighted_seq_idx.is_some()
                    && res.sequence_idx == cfg.highlighted_seq_idx)
                    || (cfg.highlighted_seq_id.is_some()
                        && Some(res.seq_id) == cfg.highlighted_seq_id);

                if is_hi && is_primary {
                    if let Some(ca) = res.atom_pos("CA") {
                        out.blend_spheres.push(SphereInstance {
                            center_radius: [ca[0], ca[1], ca[2], 0.82],
                            color: [1.0, 0.88, 0.22, 0.42],
                        });
                    }
                }

                if !(emit_sticks_for_model || is_hi) {
                    continue;
                }

                let stick_radius = match cfg.style {
                    RepresentationStyle::CartoonAndSidechains => {
                        if is_hi {
                            0.19
                        } else {
                            0.14
                        }
                    }
                    RepresentationStyle::LicoriceSticks => 0.18,
                    RepresentationStyle::BallAndStick => 0.13,
                    _ => 0.14,
                };

                // Collect visible atoms for this residue
                let vis_atoms: Vec<&StructureAtom3d> = res
                    .atoms
                    .iter()
                    .filter(|a| {
                        if !cfg.show_hydrogens && (a.element == "H" || a.element == "D") {
                            return false;
                        }
                        if draw_sidechain_sticks && !draw_all_sticks && !draw_atom_spheres {
                            // Include `CA` anchor + sidechain atoms (plus `N` on `PRO` ring closure)
                            !matches!(a.atom_name.as_str(), "C" | "O" | "OXT" | "H" | "HN")
                                && (a.atom_name != "N" || res.res_name == "PRO")
                        } else {
                            true
                        }
                    })
                    .collect();

                for &a in &vis_atoms {
                    let mut cpk = cpk_element_rgba(&a.element, false, alpha);
                    if is_hi && a.element == "C" {
                        cpk = [0.98, 0.84, 0.26, alpha];
                    }

                    if draw_atom_spheres || draw_sidechain_sticks || draw_all_sticks {
                        let rad = match cfg.style {
                            RepresentationStyle::Spacefill => {
                                vdw_radius_angstrom(&a.element, false)
                            }
                            RepresentationStyle::BallAndStick => {
                                if a.element == "H" {
                                    0.24
                                } else {
                                    0.36
                                }
                            }
                            _ => stick_radius * 1.02,
                        };
                        let inst = SphereInstance {
                            center_radius: [a.pos[0], a.pos[1], a.pos[2], rad],
                            color: cpk,
                        };
                        if is_translucent {
                            out.blend_spheres.push(inst);
                        } else {
                            out.opaque_spheres.push(inst);
                        }

                        if is_primary {
                            out.pick_spheres.push(PickableSphere3d {
                                model_idx: m_idx,
                                chain_id: chain.chain_id,
                                seq_id: res.seq_id,
                                residue_idx: res.sequence_idx,
                                residue_name: res.res_name.clone(),
                                atom_name: a.atom_name.clone(),
                                is_metal: false,
                                center: [a.pos[0] as f64, a.pos[1] as f64, a.pos[2] as f64],
                                pick_radius: (rad as f64).max(0.32),
                            });
                        }
                    }
                }

                if cfg.style != RepresentationStyle::Spacefill {
                    // Intra-residue covalent bonds via interatomic distance threshold (`<= 1.98 Å` heavy, `<= 1.25 Å` H)
                    for i in 0..vis_atoms.len() {
                        for j in (i + 1)..vis_atoms.len() {
                            let ai = vis_atoms[i];
                            let aj = vis_atoms[j];
                            let max_d = if ai.element == "H" || aj.element == "H" {
                                1.25
                            } else if ai.element == "S" || aj.element == "S" {
                                2.12
                            } else {
                                1.92
                            };
                            let dx = ai.pos[0] - aj.pos[0];
                            let dy = ai.pos[1] - aj.pos[1];
                            let dz = ai.pos[2] - aj.pos[2];
                            let d2 = dx * dx + dy * dy + dz * dz;
                            if d2 > 0.16 && d2 <= max_d * max_d {
                                let mut ci = cpk_element_rgba(&ai.element, false, alpha);
                                let mut cj = cpk_element_rgba(&aj.element, false, alpha);
                                if is_hi {
                                    if ai.element == "C" {
                                        ci = [0.98, 0.84, 0.26, alpha];
                                    }
                                    if aj.element == "C" {
                                        cj = [0.98, 0.84, 0.26, alpha];
                                    }
                                }
                                let cyl = CylinderInstance::split_bond(
                                    ai.pos,
                                    aj.pos,
                                    stick_radius,
                                    ci,
                                    cj,
                                );
                                if is_translucent {
                                    out.blend_cylinders.push(cyl);
                                } else {
                                    out.opaque_cylinders.push(cyl);
                                }
                            }
                        }
                    }

                    // Inter-residue peptide bond `C(i-1) - N(i)` in all-atom stick modes
                    if draw_all_sticks {
                        if let Some(n_atom) = vis_atoms.iter().find(|a| a.atom_name == "N") {
                            if let Some((prev_sid, prev_c, prev_col)) = prev_c_atom {
                                let dx = prev_c[0] - n_atom.pos[0];
                                let dy = prev_c[1] - n_atom.pos[1];
                                let dz = prev_c[2] - n_atom.pos[2];
                                let d2 = dx * dx + dy * dy + dz * dz;
                                if res.seq_id >= prev_sid && d2 <= 4.0 {
                                    let cn = cpk_element_rgba("N", false, alpha);
                                    let cyl = CylinderInstance::split_bond(
                                        prev_c,
                                        n_atom.pos,
                                        stick_radius,
                                        prev_col,
                                        cn,
                                    );
                                    if is_translucent {
                                        out.blend_cylinders.push(cyl);
                                    } else {
                                        out.opaque_cylinders.push(cyl);
                                    }
                                }
                            }
                        }
                        prev_c_atom = vis_atoms
                            .iter()
                            .find(|a| a.atom_name == "C")
                            .map(|c| (res.seq_id, c.pos, cpk_element_rgba("C", false, alpha)));
                    }
                }
            }
        }

        // 3. Metal Cofactors, Ligands & Coordination Bonds (`<= 2.75 Å` to Cys/His/Asp/Glu donors)
        if cfg.show_metal_cofactors {
            for het in &model.het_atoms {
                let rad = if het.is_metal {
                    if cfg.style == RepresentationStyle::Spacefill {
                        1.55
                    } else {
                        0.92
                    }
                } else if cfg.style == RepresentationStyle::Spacefill {
                    vdw_radius_angstrom(&het.element, false)
                } else {
                    0.34
                };
                let col = cpk_element_rgba(&het.element, het.is_metal, alpha);
                let sp = SphereInstance {
                    center_radius: [het.pos[0], het.pos[1], het.pos[2], rad],
                    color: col,
                };
                if is_translucent {
                    out.blend_spheres.push(sp);
                } else {
                    out.opaque_spheres.push(sp);
                }

                if is_primary {
                    out.pick_spheres.push(PickableSphere3d {
                        model_idx: m_idx,
                        chain_id: het.chain_id,
                        seq_id: het.seq_id,
                        residue_idx: None,
                        residue_name: het.res_name.clone(),
                        atom_name: het.atom_name.clone(),
                        is_metal: het.is_metal,
                        center: [het.pos[0] as f64, het.pos[1] as f64, het.pos[2] as f64],
                        pick_radius: (rad as f64).max(0.45),
                    });
                }

                if het.is_metal && is_primary {
                    for chain in &model.chains {
                        for res in &chain.residues {
                            for donor in &res.atoms {
                                if !matches!(
                                    donor.atom_name.as_str(),
                                    "SG" | "SD"
                                        | "ND1"
                                        | "NE2"
                                        | "OD1"
                                        | "OD2"
                                        | "OE1"
                                        | "OE2"
                                        | "OG"
                                        | "OG1"
                                        | "O"
                                ) {
                                    continue;
                                }
                                let dx = het.pos[0] - donor.pos[0];
                                let dy = het.pos[1] - donor.pos[1];
                                let dz = het.pos[2] - donor.pos[2];
                                let d2 = dx * dx + dy * dy + dz * dz;
                                if d2 <= 2.75 * 2.75 {
                                    out.opaque_cylinders.push(CylinderInstance::uniform(
                                        het.pos, donor.pos, 0.09, col, 5.0,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(ov) = restraint_overlay {
        out.restraint_cylinders.clone_from(&ov.cylinders);
        out.pick_restraints.clone_from(&ov.pickables);
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn append_chain_cartoon_ribbon(
    chain: &PolymerChain3d,
    model_idx: usize,
    is_primary: bool,
    alpha: f32,
    is_translucent: bool,
    ensemble: &StructureEnsemble3d,
    cfg: &SceneBuildConfig,
    track_data: &QuantitativeTrackData,
    out: &mut Mol3dSceneBuffers,
) {
    // Split chain at missing `CA` or peptide breaks (`d(CA_i, CA_{i+1}) > 4.8 Å`)
    let mut seg_start = 0usize;
    while seg_start < chain.residues.len() {
        if chain.residues[seg_start].atom_pos("CA").is_none() {
            seg_start += 1;
            continue;
        }
        let mut seg_end = seg_start + 1;
        while seg_end < chain.residues.len() {
            let (Some(prev_ca), Some(cur_ca)) = (
                chain.residues[seg_end - 1].atom_pos("CA"),
                chain.residues[seg_end].atom_pos("CA"),
            ) else {
                break;
            };
            if vec3_len(vec3_sub(cur_ca, prev_ca)) > 4.8 {
                break;
            }
            seg_end += 1;
        }
        let slice = &chain.residues[seg_start..seg_end];
        if slice.len() >= 2 {
            extrude_carson_bugg_segment(
                slice,
                chain.chain_id,
                model_idx,
                is_primary,
                alpha,
                is_translucent,
                ensemble,
                cfg,
                track_data,
                out,
            );
        }
        seg_start = seg_end;
    }
}

#[allow(clippy::too_many_arguments)]
fn extrude_carson_bugg_segment(
    residues: &[PolymerResidue3d],
    chain_id: char,
    model_idx: usize,
    is_primary: bool,
    alpha: f32,
    is_translucent: bool,
    ensemble: &StructureEnsemble3d,
    cfg: &SceneBuildConfig,
    track_data: &QuantitativeTrackData,
    out: &mut Mol3dSceneBuffers,
) {
    let n = residues.len();
    let mut ca: Vec<[f64; 3]> = Vec::with_capacity(n);
    for r in residues {
        let p = r.atom_pos("CA").unwrap_or([0.0; 3]);
        ca.push([p[0] as f64, p[1] as f64, p[2] as f64]);
    }

    // Smooth beta-strand `CA` pleat zig-zag (`0.5*CA_i + 0.25*(CA_{i-1} + CA_{i+1})`) per Carson & Bugg (1986)
    let mut guide = ca.clone();
    for i in 1..(n - 1) {
        if residues[i].ss == SecondaryStructureKind::Sheet {
            for d in 0..3 {
                guide[i][d] = 0.50 * ca[i][d] + 0.25 * (ca[i - 1][d] + ca[i + 1][d]);
            }
        }
    }

    // Sign-consistent peptide-plane normal/width vectors `plane_v[i]` (`v_i . v_{i-1} >= 0`)
    let mut plane_v = vec![[0.0f64, 1.0, 0.0]; n];
    for i in 0..n {
        let ca_cur = ca[i];
        let next_ca = if i + 1 < n { ca[i + 1] } else { ca[i] };
        let prev_ca = if i > 0 { ca[i - 1] } else { ca[i] };
        let axis = vec3_norm([
            next_ca[0] - prev_ca[0],
            next_ca[1] - prev_ca[1],
            next_ca[2] - prev_ca[2],
        ]);
        let raw_b = if let Some(o) = residues[i].atom_pos("O") {
            let co = [
                (o[0] as f64) - ca_cur[0],
                (o[1] as f64) - ca_cur[1],
                (o[2] as f64) - ca_cur[2],
            ];
            let proj = vec3_dot(co, axis);
            let perp = [
                co[0] - proj * axis[0],
                co[1] - proj * axis[1],
                co[2] - proj * axis[2],
            ];
            if vec3_len(perp) > 1e-4 {
                vec3_norm(perp)
            } else {
                vec3_norm(vec3_cross(axis, [0.0, 1.0, 0.0]))
            }
        } else {
            let fallback = if axis[1].abs() > 0.85 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            vec3_norm(vec3_cross(axis, fallback))
        };

        plane_v[i] = if i > 0 && vec3_dot(raw_b, plane_v[i - 1]) < 0.0 {
            [-raw_b[0], -raw_b[1], -raw_b[2]]
        } else {
            raw_b
        };
    }

    let (verts_dst, indices_dst, rings_dst) = if is_translucent {
        (
            &mut out.blend_ribbon_verts,
            &mut out.blend_ribbon_indices,
            &mut out.blend_rings,
        )
    } else {
        (
            &mut out.opaque_ribbon_verts,
            &mut out.opaque_ribbon_indices,
            &mut out.opaque_rings,
        )
    };

    let total_rings = (n - 1) * RIBBON_SUBDIVS + 1;
    let first_ring_idx = verts_dst.len() / RIBBON_RING_VERTS;
    verts_dst.resize(
        verts_dst.len() + total_rings * RIBBON_RING_VERTS,
        RibbonVertex::zeroed(),
    );

    let mut prev_pick_pt: Option<[f64; 3]> = None;

    for seg in 0..(n - 1) {
        let p0 = if seg > 0 { guide[seg - 1] } else { guide[0] };
        let p1 = guide[seg];
        let p2 = guide[seg + 1];
        let p3 = if seg + 2 < n {
            guide[seg + 2]
        } else {
            guide[n - 1]
        };

        let sub_end = if seg + 2 == n {
            RIBBON_SUBDIVS
        } else {
            RIBBON_SUBDIVS - 1
        };

        for sub in 0..=sub_end {
            let t = (sub as f64) / (RIBBON_SUBDIVS as f64);
            let (pos, tan) = catmull_rom_pos_tan(p0, p1, p2, p3, t);

            let v_interp = [
                (1.0 - t) * plane_v[seg][0] + t * plane_v[seg + 1][0],
                (1.0 - t) * plane_v[seg][1] + t * plane_v[seg + 1][1],
                (1.0 - t) * plane_v[seg][2] + t * plane_v[seg + 1][2],
            ];
            let dot_vt = vec3_dot(v_interp, tan);
            let binormal = vec3_norm([
                v_interp[0] - dot_vt * tan[0],
                v_interp[1] - dot_vt * tan[1],
                v_interp[2] - dot_vt * tan[2],
            ]);
            let normal = vec3_norm(vec3_cross(tan, binormal));

            let owner_idx = if t < 0.5 { seg } else { seg + 1 };
            let res = &residues[owner_idx];
            let ss = res.ss;

            // Detect C-terminal end of a beta-strand (`Sheet` followed by non-`Sheet` or chain end)
            // to flare into a classic directional Carson–Bugg arrowhead!
            let is_sheet_cterm = ss == SecondaryStructureKind::Sheet
                && (owner_idx + 1 >= n
                    || residues[owner_idx + 1].ss != SecondaryStructureKind::Sheet);

            let cartoon_half_axes = match ss {
                SecondaryStructureKind::Helix => [1.12f32, 0.22f32],
                SecondaryStructureKind::Sheet => {
                    if is_sheet_cterm && owner_idx == seg + 1 {
                        let arrow_u = ((t - 0.35) / 0.65).clamp(0.0, 1.0) as f32;
                        let flare = 1.95 * (1.0 - arrow_u) + 0.14 * arrow_u;
                        [flare, 0.22]
                    } else if is_sheet_cterm && owner_idx == seg && seg + 2 == n {
                        let flare = 1.95 * (1.0 - (t as f32)) + 0.14 * (t as f32);
                        [flare, 0.22]
                    } else {
                        [1.18, 0.20]
                    }
                }
                SecondaryStructureKind::Coil => [0.28, 0.28],
            };

            let rmsd = res
                .sequence_idx
                .and_then(|k| ensemble.per_residue_ca_rmsd.get(k).copied());
            let is_hi = (cfg.highlighted_seq_idx.is_some()
                && res.sequence_idx == cfg.highlighted_seq_idx)
                || (cfg.highlighted_seq_id.is_some() && Some(res.seq_id) == cfg.highlighted_seq_id);

            let ring_serial = seg * RIBBON_SUBDIVS + sub;
            let vbo_start = (first_ring_idx + ring_serial) * RIBBON_RING_VERTS;

            let ring = CachedRibbonRing {
                vbo_start,
                center: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
                normal_u: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
                binormal_v: [binormal[0] as f32, binormal[1] as f32, binormal[2] as f32],
                seq_idx: res.sequence_idx,
                ss,
                ensemble_rmsd: rmsd,
                cartoon_half_axes,
                alpha,
                is_highlighted: is_hi,
            };
            write_ring_vertices(verts_dst, &ring, track_data);
            rings_dst.push(ring);

            if is_primary {
                if let Some(p_prev) = prev_pick_pt {
                    let pick_rad = (cartoon_half_axes[0].max(0.42)) as f64;
                    out.pick_ribbons.push(PickableRibbonSegment3d {
                        model_idx,
                        chain_id,
                        seq_id: res.seq_id,
                        residue_idx: res.sequence_idx,
                        residue_name: res.res_name.clone(),
                        p0: p_prev,
                        p1: pos,
                        pick_radius: pick_rad,
                    });
                }
                prev_pick_pt = Some(pos);
            }
        }
    }

    // Stitch consecutive profile rings into indexed triangle strips
    for r in 0..(total_rings - 1) {
        let base0 = ((first_ring_idx + r) * RIBBON_RING_VERTS) as u32;
        let base1 = ((first_ring_idx + r + 1) * RIBBON_RING_VERTS) as u32;
        for k in 0..(RIBBON_RING_VERTS as u32) {
            let kn = (k + 1) % (RIBBON_RING_VERTS as u32);
            indices_dst.extend_from_slice(&[
                base0 + k,
                base1 + k,
                base1 + kn,
                base0 + k,
                base1 + kn,
                base0 + kn,
            ]);
        }
    }
}

// ============================================================================
// 9. Unit Sphere & Half-Bond Cylinder Template Meshes + `wgpu` Pipeline
// ============================================================================

fn build_unit_icosphere_sub1() -> (Vec<TemplatePosVertex>, Vec<u16>) {
    let phi = 0.5 * (1.0 + 5.0f32.sqrt());
    let norm = |v: [f32; 3]| -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let mut verts: Vec<[f32; 3]> = vec![
        norm([-1.0, phi, 0.0]),
        norm([1.0, phi, 0.0]),
        norm([-1.0, -phi, 0.0]),
        norm([1.0, -phi, 0.0]),
        norm([0.0, -1.0, phi]),
        norm([0.0, 1.0, phi]),
        norm([0.0, -1.0, -phi]),
        norm([0.0, 1.0, -phi]),
        norm([phi, 0.0, -1.0]),
        norm([phi, 0.0, 1.0]),
        norm([-phi, 0.0, -1.0]),
        norm([-phi, 0.0, 1.0]),
    ];
    let base_tris: [[u16; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut mid_cache: BTreeMap<(u16, u16), u16> = BTreeMap::new();
    let mut sub_tris: Vec<u16> = Vec::with_capacity(240);
    for [a, b, c] in base_tris {
        let mut midpoint = |i: u16, j: u16| -> u16 {
            let key = if i < j { (i, j) } else { (j, i) };
            if let Some(&idx) = mid_cache.get(&key) {
                return idx;
            }
            let vi = verts[i as usize];
            let vj = verts[j as usize];
            let m = norm([
                0.5 * (vi[0] + vj[0]),
                0.5 * (vi[1] + vj[1]),
                0.5 * (vi[2] + vj[2]),
            ]);
            let idx = verts.len() as u16;
            verts.push(m);
            mid_cache.insert(key, idx);
            idx
        };
        let ab = midpoint(a, b);
        let bc = midpoint(b, c);
        let ca = midpoint(c, a);
        sub_tris.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
    }

    (
        verts
            .into_iter()
            .map(|pos| TemplatePosVertex { pos })
            .collect(),
        sub_tris,
    )
}

fn build_unit_split_cylinder() -> (Vec<TemplatePosVertex>, Vec<u16>) {
    const SIDES: usize = 10;
    let z_rings: [f32; 4] = [0.0, 0.50, 0.50005, 1.0];
    let mut verts = Vec::with_capacity(SIDES * 4);
    for &z in &z_rings {
        for s in 0..SIDES {
            let th = (s as f32) * std::f32::consts::TAU / (SIDES as f32);
            let (sin_t, cos_t) = th.sin_cos();
            verts.push(TemplatePosVertex {
                pos: [cos_t, sin_t, z],
            });
        }
    }
    let mut indices = Vec::with_capacity(SIDES * 12);
    for &(r0, r1) in &[(0usize, 1usize), (2usize, 3usize)] {
        let b0 = (r0 * SIDES) as u16;
        let b1 = (r1 * SIDES) as u16;
        for s in 0..(SIDES as u16) {
            let sn = (s + 1) % (SIDES as u16);
            indices.extend_from_slice(&[b0 + s, b1 + s, b1 + sn, b0 + s, b1 + sn, b0 + sn]);
        }
    }
    (verts, indices)
}

struct OffscreenTarget3d {
    width: u32,
    height: u32,
    msaa_view: wgpu::TextureView,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    resolve_tex: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
}

struct UploadedSceneBuffers {
    scene_gen: u64,
    opaque_ribbon_vbo: wgpu::Buffer,
    opaque_ribbon_ibo: wgpu::Buffer,
    opaque_ribbon_idx_count: u32,
    blend_ribbon_vbo: wgpu::Buffer,
    blend_ribbon_ibo: wgpu::Buffer,
    blend_ribbon_idx_count: u32,
    opaque_sphere_vbo: wgpu::Buffer,
    opaque_sphere_count: u32,
    blend_sphere_vbo: wgpu::Buffer,
    blend_sphere_count: u32,
    opaque_cyl_vbo: wgpu::Buffer,
    opaque_cyl_count: u32,
    blend_cyl_vbo: wgpu::Buffer,
    blend_cyl_count: u32,
    restraint_cyl_vbo: wgpu::Buffer,
    restraint_cyl_count: u32,
}

/// Persistent `wgpu` 3D rendering resource registered in `egui_wgpu::CallbackResources` (and usable
/// headlessly in native unit/benchmark tests and high-DPI PNG export).
pub struct Mol3dPipelineResource {
    uniform_buf: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    ribbon_opaque_pl: wgpu::RenderPipeline,
    ribbon_blend_pl: wgpu::RenderPipeline,
    sphere_opaque_pl: wgpu::RenderPipeline,
    sphere_blend_pl: wgpu::RenderPipeline,
    cyl_opaque_pl: wgpu::RenderPipeline,
    cyl_blend_pl: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    blit_pl: Option<wgpu::RenderPipeline>,
    sphere_template_vbo: wgpu::Buffer,
    sphere_template_ibo: wgpu::Buffer,
    sphere_template_indices: u32,
    cyl_template_vbo: wgpu::Buffer,
    cyl_template_ibo: wgpu::Buffer,
    cyl_template_indices: u32,
    target: Option<OffscreenTarget3d>,
    uploaded: Option<UploadedSceneBuffers>,
}

impl Mol3dPipelineResource {
    /// Create all 3D hardware pipelines (`4x MSAA` `Rgba8Unorm` + `Depth32Float`) and optional
    /// `egui` surface blit pipeline (`egui_target_format`).
    pub fn new(device: &wgpu::Device, egui_target_format: Option<wgpu::TextureFormat>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-mol3d-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mol3d.wgsl").into()),
        });

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-mol3d-scene-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-mol3d-uniforms"),
            contents: bytemuck::bytes_of(&Mol3dUniforms::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-mol3d-scene-bg"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-mol3d-scene-pl-layout"),
            bind_group_layouts: &[Some(&scene_bgl)],
            immediate_size: 0,
        });

        let ribbon_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RibbonVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let template_pos_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TemplatePosVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let sphere_inst_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SphereInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let cyl_inst_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CylinderInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let make_3d_pl = |label: &str,
                          vs_entry: &str,
                          fs_entry: &str,
                          buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                          cull_mode: Option<wgpu::Face>,
                          alpha_blend: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&scene_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs_entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: MOL3D_COLOR_FORMAT,
                        blend: Some(if alpha_blend {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: MOL3D_DEPTH_FORMAT,
                    depth_write_enabled: Some(!alpha_blend),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MOL3D_MSAA_SAMPLES,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let ribbon_vbls = [Some(ribbon_vbl)];
        let ribbon_opaque_pl = make_3d_pl(
            "mol3d-ribbon-opaque",
            "vs_ribbon",
            "fs_lit",
            &ribbon_vbls,
            None,
            false,
        );
        let ribbon_blend_pl = make_3d_pl(
            "mol3d-ribbon-blend",
            "vs_ribbon",
            "fs_lit",
            &ribbon_vbls,
            None,
            true,
        );
        let sphere_vbls = [Some(template_pos_vbl.clone()), Some(sphere_inst_vbl)];
        let sphere_opaque_pl = make_3d_pl(
            "mol3d-sphere-opaque",
            "vs_sphere",
            "fs_lit",
            &sphere_vbls,
            Some(wgpu::Face::Back),
            false,
        );
        let sphere_blend_pl = make_3d_pl(
            "mol3d-sphere-blend",
            "vs_sphere",
            "fs_lit",
            &sphere_vbls,
            Some(wgpu::Face::Back),
            true,
        );
        let cyl_vbls = [Some(template_pos_vbl), Some(cyl_inst_vbl)];
        let cyl_opaque_pl = make_3d_pl(
            "mol3d-cyl-opaque",
            "vs_cylinder",
            "fs_cylinder",
            &cyl_vbls,
            None,
            false,
        );
        let cyl_blend_pl = make_3d_pl(
            "mol3d-cyl-blend",
            "vs_cylinder",
            "fs_cylinder",
            &cyl_vbls,
            None,
            true,
        );

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-mol3d-blit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nmr-mol3d-blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blit_pl = egui_target_format.map(|fmt| {
            let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nmr-mol3d-blit-layout"),
                bind_group_layouts: &[Some(&blit_bgl)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("nmr-mol3d-blit-pl"),
                layout: Some(&blit_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_blit"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_blit"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });

        let (sph_v, sph_i) = build_unit_icosphere_sub1();
        let sphere_template_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mol3d-sphere-template-vbo"),
            contents: bytemuck::cast_slice(&sph_v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_template_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mol3d-sphere-template-ibo"),
            contents: bytemuck::cast_slice(&sph_i),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (cyl_v, cyl_i) = build_unit_split_cylinder();
        let cyl_template_vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mol3d-cyl-template-vbo"),
            contents: bytemuck::cast_slice(&cyl_v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cyl_template_ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mol3d-cyl-template-ibo"),
            contents: bytemuck::cast_slice(&cyl_i),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            uniform_buf,
            scene_bind_group,
            ribbon_opaque_pl,
            ribbon_blend_pl,
            sphere_opaque_pl,
            sphere_blend_pl,
            cyl_opaque_pl,
            cyl_blend_pl,
            blit_bgl,
            blit_sampler,
            blit_pl,
            sphere_template_vbo,
            sphere_template_ibo,
            sphere_template_indices: sph_i.len() as u32,
            cyl_template_vbo,
            cyl_template_ibo,
            cyl_template_indices: cyl_i.len() as u32,
            target: None,
            uploaded: None,
        }
    }

    fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.clamp(2, 4096);
        let h = height.clamp(2, 4096);
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.width == w && t.height == h)
        {
            return;
        }
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let msaa_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mol3d-msaa-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: MOL3D_MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: MOL3D_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mol3d-resolve-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MOL3D_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mol3d-msaa-depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: MOL3D_MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: MOL3D_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let msaa_view = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let resolve_view = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mol3d-blit-bg"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        self.target = Some(OffscreenTarget3d {
            width: w,
            height: h,
            msaa_view,
            resolve_tex,
            resolve_view,
            depth_view,
            blit_bind_group,
        });
    }

    fn ensure_scene_uploaded(&mut self, device: &wgpu::Device, scene: &Mol3dSceneBuffers) {
        if self
            .uploaded
            .as_ref()
            .is_some_and(|u| u.scene_gen == scene.scene_gen)
        {
            return;
        }

        let make_vbo = |label: &str, bytes: &[u8]| -> wgpu::Buffer {
            let fallback = [0u8; 64];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &fallback } else { bytes },
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let make_ibo = |label: &str, indices: &[u32]| -> wgpu::Buffer {
            let fallback = [0u32; 3];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if indices.is_empty() {
                    bytemuck::cast_slice(&fallback)
                } else {
                    bytemuck::cast_slice(indices)
                },
                usage: wgpu::BufferUsages::INDEX,
            })
        };

        self.uploaded = Some(UploadedSceneBuffers {
            scene_gen: scene.scene_gen,
            opaque_ribbon_vbo: make_vbo(
                "mol3d-op-rib-vbo",
                bytemuck::cast_slice(&scene.opaque_ribbon_verts),
            ),
            opaque_ribbon_ibo: make_ibo("mol3d-op-rib-ibo", &scene.opaque_ribbon_indices),
            opaque_ribbon_idx_count: scene.opaque_ribbon_indices.len() as u32,
            blend_ribbon_vbo: make_vbo(
                "mol3d-bl-rib-vbo",
                bytemuck::cast_slice(&scene.blend_ribbon_verts),
            ),
            blend_ribbon_ibo: make_ibo("mol3d-bl-rib-ibo", &scene.blend_ribbon_indices),
            blend_ribbon_idx_count: scene.blend_ribbon_indices.len() as u32,
            opaque_sphere_vbo: make_vbo(
                "mol3d-op-sph-vbo",
                bytemuck::cast_slice(&scene.opaque_spheres),
            ),
            opaque_sphere_count: scene.opaque_spheres.len() as u32,
            blend_sphere_vbo: make_vbo(
                "mol3d-bl-sph-vbo",
                bytemuck::cast_slice(&scene.blend_spheres),
            ),
            blend_sphere_count: scene.blend_spheres.len() as u32,
            opaque_cyl_vbo: make_vbo(
                "mol3d-op-cyl-vbo",
                bytemuck::cast_slice(&scene.opaque_cylinders),
            ),
            opaque_cyl_count: scene.opaque_cylinders.len() as u32,
            blend_cyl_vbo: make_vbo(
                "mol3d-bl-cyl-vbo",
                bytemuck::cast_slice(&scene.blend_cylinders),
            ),
            blend_cyl_count: scene.blend_cylinders.len() as u32,
            restraint_cyl_vbo: make_vbo(
                "mol3d-rst-cyl-vbo",
                bytemuck::cast_slice(&scene.restraint_cylinders),
            ),
            restraint_cyl_count: scene.restraint_cylinders.len() as u32,
        });
    }

    /// Encode the offscreen `4x MSAA` + `Depth32Float` 3D pass into `encoder` (retaining VRAM
    /// geometry buffers whenever `scene.scene_gen` is unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn record_offscreen_3d_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        uniforms: &Mol3dUniforms,
        scene: &Mol3dSceneBuffers,
    ) {
        self.ensure_target(device, width, height);
        self.ensure_scene_uploaded(device, scene);
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));

        let (Some(target), Some(up)) = (self.target.as_ref(), self.uploaded.as_ref()) else {
            return;
        };

        let clear = wgpu::Color {
            r: uniforms.bg_color[0] as f64,
            g: uniforms.bg_color[1] as f64,
            b: uniforms.bg_color[2] as f64,
            a: uniforms.bg_color[3] as f64,
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nmr-mol3d-offscreen-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.msaa_view,
                depth_slice: None,
                resolve_target: Some(&target.resolve_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_bind_group(0, &self.scene_bind_group, &[]);

        // 1. Opaque geometry (`depth_write_enabled = true`)
        if up.opaque_ribbon_idx_count > 0 {
            rpass.set_pipeline(&self.ribbon_opaque_pl);
            rpass.set_vertex_buffer(0, up.opaque_ribbon_vbo.slice(..));
            rpass.set_index_buffer(up.opaque_ribbon_ibo.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..up.opaque_ribbon_idx_count, 0, 0..1);
        }
        if up.opaque_sphere_count > 0 {
            rpass.set_pipeline(&self.sphere_opaque_pl);
            rpass.set_vertex_buffer(0, self.sphere_template_vbo.slice(..));
            rpass.set_vertex_buffer(1, up.opaque_sphere_vbo.slice(..));
            rpass.set_index_buffer(
                self.sphere_template_ibo.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            rpass.draw_indexed(
                0..self.sphere_template_indices,
                0,
                0..up.opaque_sphere_count,
            );
        }
        if up.opaque_cyl_count > 0 {
            rpass.set_pipeline(&self.cyl_opaque_pl);
            rpass.set_vertex_buffer(0, self.cyl_template_vbo.slice(..));
            rpass.set_vertex_buffer(1, up.opaque_cyl_vbo.slice(..));
            rpass.set_index_buffer(self.cyl_template_ibo.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.cyl_template_indices, 0, 0..up.opaque_cyl_count);
        }
        if up.restraint_cyl_count > 0 {
            rpass.set_pipeline(&self.cyl_opaque_pl);
            rpass.set_vertex_buffer(0, self.cyl_template_vbo.slice(..));
            rpass.set_vertex_buffer(1, up.restraint_cyl_vbo.slice(..));
            rpass.set_index_buffer(self.cyl_template_ibo.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.cyl_template_indices, 0, 0..up.restraint_cyl_count);
        }

        // 2. Translucent ensemble conformers & selection halos (`depth_write_enabled = false`)
        if up.blend_ribbon_idx_count > 0 {
            rpass.set_pipeline(&self.ribbon_blend_pl);
            rpass.set_vertex_buffer(0, up.blend_ribbon_vbo.slice(..));
            rpass.set_index_buffer(up.blend_ribbon_ibo.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..up.blend_ribbon_idx_count, 0, 0..1);
        }
        if up.blend_cyl_count > 0 {
            rpass.set_pipeline(&self.cyl_blend_pl);
            rpass.set_vertex_buffer(0, self.cyl_template_vbo.slice(..));
            rpass.set_vertex_buffer(1, up.blend_cyl_vbo.slice(..));
            rpass.set_index_buffer(self.cyl_template_ibo.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.cyl_template_indices, 0, 0..up.blend_cyl_count);
        }
        if up.blend_sphere_count > 0 {
            rpass.set_pipeline(&self.sphere_blend_pl);
            rpass.set_vertex_buffer(0, self.sphere_template_vbo.slice(..));
            rpass.set_vertex_buffer(1, up.blend_sphere_vbo.slice(..));
            rpass.set_index_buffer(
                self.sphere_template_ibo.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            rpass.draw_indexed(0..self.sphere_template_indices, 0, 0..up.blend_sphere_count);
        }
    }

    /// Blit the MSAA-resolved offscreen 3D texture into `egui`'s active UI `RenderPass`.
    pub fn paint_into_egui(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let (Some(pl), Some(target)) = (self.blit_pl.as_ref(), self.target.as_ref()) else {
            return;
        };
        render_pass.set_pipeline(pl);
        render_pass.set_bind_group(0, &target.blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Render an offscreen `width x height` RGBA8 image synchronously on native desktop / headless
    /// `wgpu` tests (suitable for high-DPI publication PNG export).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_offscreen_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        uniforms: &Mol3dUniforms,
        scene: &Mol3dSceneBuffers,
    ) -> Result<Vec<u8>, String> {
        let w = width.clamp(2, 4096);
        let h = height.clamp(2, 4096);
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mol3d-png-readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mol3d-offscreen-readback-enc"),
        });
        self.record_offscreen_3d_pass(device, queue, &mut encoder, w, h, uniforms, scene);

        let Some(target) = self.target.as_ref() else {
            return Err("Offscreen 3D target unavailable".to_string());
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.resolve_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("Failed mapping offscreen 3D readback buffer: {e}"))?;

        let mut tight = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let start = (row * bytes_per_row) as usize;
            let end = start + (w * 4) as usize;
            tight.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        readback.unmap();
        Ok(tight)
    }

    /// Render and encode a high-DPI PNG byte buffer using `nmr_core::octet_render::encode_rgba_png`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn export_offscreen_png(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        uniforms: &Mol3dUniforms,
        scene: &Mol3dSceneBuffers,
    ) -> Result<Vec<u8>, String> {
        let rgba = self.render_offscreen_rgba(device, queue, width, height, uniforms, scene)?;
        Ok(nmr_core::octet_render::encode_rgba_png(
            width.clamp(2, 4096) as usize,
            height.clamp(2, 4096) as usize,
            &rgba,
        ))
    }
}

/// `egui_wgpu` paint callback rendering `Mol3dSceneBuffers` offscreen (`prepare`) and blitting the
/// resolved `4x MSAA` texture into the dockable 3D Structure Viewport (`paint`).
pub struct Mol3dPaintCallback {
    pub width_px: u32,
    pub height_px: u32,
    pub uniforms: Mol3dUniforms,
    pub scene: Arc<Mol3dSceneBuffers>,
}

impl egui_wgpu::CallbackTrait for Mol3dPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<Mol3dPipelineResource>() {
            res.record_offscreen_3d_pass(
                device,
                queue,
                egui_encoder,
                self.width_px,
                self.height_px,
                &self.uniforms,
                &self.scene,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<Mol3dPipelineResource>() else {
            return;
        };
        let vp = info.viewport_in_pixels();
        if vp.width_px <= 0 || vp.height_px <= 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );
        res.paint_into_egui(render_pass);
    }
}

// ============================================================================
// 10. Deterministic Depth-Sorted & XML-Escaped Publication SVG Export
// ============================================================================

#[derive(Clone)]
enum SvgPrim3d {
    Line {
        depth: f64,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        stroke_px: f64,
        rgba: [f32; 4],
        dashed: bool,
        tooltip: String,
    },
    Circle {
        depth: f64,
        cx: f64,
        cy: f64,
        r_px: f64,
        rgba: [f32; 4],
        tooltip: String,
    },
}

impl SvgPrim3d {
    #[inline]
    fn depth(&self) -> f64 {
        match self {
            Self::Line { depth, .. } | Self::Circle { depth, .. } => *depth,
        }
    }
}

#[inline]
fn rgba_to_svg_hex(rgba: [f32; 4]) -> (String, f32) {
    let r = (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    (format!("#{r:02x}{g:02x}{b:02x}"), rgba[3].clamp(0.0, 1.0))
}

/// Export a deterministic, back-to-front depth-sorted vector `SVG` snapshot of the current 3D
/// view, applying [`escape_xml`] to EVERY interpolated metadata/title/chain/residue/atom/restraint
/// string so hostile characters (`<`, `>`, `&`, `"`, `'`) can never break XML well-formedness.
pub fn export_structure_svg(
    ensemble: &StructureEnsemble3d,
    camera: &ArcballCamera,
    cfg: &SceneBuildConfig,
    track_data: &QuantitativeTrackData,
    restraint_overlay: Option<&RestraintOverlayBuild>,
    width: u32,
    height: u32,
) -> String {
    let w = width.clamp(120, 8192) as f64;
    let h = height.clamp(120, 8192) as f64;
    let px_per_angstrom_ref = h / (2.0 * camera.ortho_half_height()).max(1.0);
    let mut prims: Vec<SvgPrim3d> = Vec::new();

    if let Some(m0) = ensemble.models.get(
        cfg.active_model_idx
            .min(ensemble.models.len().saturating_sub(1)),
    ) {
        for chain in &m0.chains {
            for w_pair in chain.residues.windows(2) {
                let (r0, r1) = (&w_pair[0], &w_pair[1]);
                let (Some(ca0), Some(ca1)) = (r0.atom_pos("CA"), r1.atom_pos("CA")) else {
                    continue;
                };
                if vec3_len(vec3_sub(ca1, ca0)) > 4.8 {
                    continue;
                }
                let p0 = [ca0[0] as f64, ca0[1] as f64, ca0[2] as f64];
                let p1 = [ca1[0] as f64, ca1[1] as f64, ca1[2] as f64];
                if let (Some((x0, y0, d0)), Some((x1, y1, d1))) = (
                    camera.project_world_to_screen(p0, w, h),
                    camera.project_world_to_screen(p1, w, h),
                ) {
                    let rmsd = r0
                        .sequence_idx
                        .and_then(|k| ensemble.per_residue_ca_rmsd.get(k).copied());
                    let style = track_data.evaluate_residue_style(
                        r0.sequence_idx,
                        r0.ss.default_rgba(),
                        0.32,
                        rmsd,
                        1.0,
                    );
                    let width_ang = if track_data.putty_enabled {
                        style.radius_angstrom * 2.0
                    } else {
                        match r0.ss {
                            SecondaryStructureKind::Helix => 1.05,
                            SecondaryStructureKind::Sheet => 1.15,
                            SecondaryStructureKind::Coil => 0.48,
                        }
                    };
                    prims.push(SvgPrim3d::Line {
                        depth: 0.5 * (d0 + d1),
                        x0,
                        y0,
                        x1,
                        y1,
                        stroke_px: ((width_ang as f64) * px_per_angstrom_ref).clamp(1.4, 22.0),
                        rgba: style.rgba,
                        dashed: false,
                        tooltip: format!("{}:{}{}", chain.chain_id, r0.res_name, r0.seq_id),
                    });
                }
            }
        }

        if cfg.show_metal_cofactors {
            for het in &m0.het_atoms {
                let p = [het.pos[0] as f64, het.pos[1] as f64, het.pos[2] as f64];
                if let Some((cx, cy, depth)) = camera.project_world_to_screen(p, w, h) {
                    let col = cpk_element_rgba(&het.element, het.is_metal, 1.0);
                    let r_ang = if het.is_metal { 0.92 } else { 0.36 };
                    prims.push(SvgPrim3d::Circle {
                        depth,
                        cx,
                        cy,
                        r_px: (r_ang * px_per_angstrom_ref).clamp(2.5, 28.0),
                        rgba: col,
                        tooltip: format!(
                            "{}:{}{} {}",
                            het.chain_id, het.res_name, het.seq_id, het.atom_name
                        ),
                    });
                }
            }
        }
    }

    if let Some(ov) = restraint_overlay {
        for (ev, cyl) in ov.evaluated.iter().zip(ov.cylinders.iter()) {
            let p0 = [ev.p0[0] as f64, ev.p0[1] as f64, ev.p0[2] as f64];
            let p1 = [ev.p1[0] as f64, ev.p1[1] as f64, ev.p1[2] as f64];
            if let (Some((x0, y0, d0)), Some((x1, y1, d1))) = (
                camera.project_world_to_screen(p0, w, h),
                camera.project_world_to_screen(p1, w, h),
            ) {
                prims.push(SvgPrim3d::Line {
                    depth: 0.5 * (d0 + d1),
                    x0,
                    y0,
                    x1,
                    y1,
                    stroke_px: ((cyl.p0_rad[3] as f64) * 2.0 * px_per_angstrom_ref).clamp(1.2, 8.0),
                    rgba: ev.status.rgba(),
                    dashed: cyl.p1_flags[3] >= 2.0,
                    tooltip: ev.label.clone(),
                });
            }
        }
    }

    // Painter's order: farthest (`max depth`) first
    prims.sort_by(|a, b| b.depth().total_cmp(&a.depth()));

    let mut svg = String::with_capacity(prims.len() * 140 + 1024);
    let _ = writeln!(
        svg,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width.clamp(120, 8192),
        height.clamp(120, 8192),
        width.clamp(120, 8192),
        height.clamp(120, 8192)
    );
    let safe_title = escape_xml(&ensemble.title);
    let safe_track = escape_xml(track_data.active_track.label());
    let _ = writeln!(svg, "  <title>{safe_title}</title>");
    let _ = writeln!(
        svg,
        r##"  <rect width="100%" height="100%" fill="#0e1118"/>"##
    );
    let _ = writeln!(
        svg,
        r#"  <g stroke-linecap="round" stroke-linejoin="round">"#
    );

    for prim in &prims {
        match prim {
            SvgPrim3d::Line {
                x0,
                y0,
                x1,
                y1,
                stroke_px,
                rgba,
                dashed,
                tooltip,
                ..
            } => {
                let (hex, op) = rgba_to_svg_hex(*rgba);
                let dash_attr = if *dashed {
                    r#" stroke-dasharray="5,4""#
                } else {
                    ""
                };
                let safe_tip = escape_xml(tooltip);
                let _ = writeln!(
                    svg,
                    r#"    <line x1="{x0:.2}" y1="{y0:.2}" x2="{x1:.2}" y2="{y1:.2}" stroke="{hex}" stroke-opacity="{op:.3}" stroke-width="{stroke_px:.2}"{dash_attr}><title>{safe_tip}</title></line>"#
                );
            }
            SvgPrim3d::Circle {
                cx,
                cy,
                r_px,
                rgba,
                tooltip,
                ..
            } => {
                let (hex, op) = rgba_to_svg_hex(*rgba);
                let safe_tip = escape_xml(tooltip);
                let _ = writeln!(
                    svg,
                    r##"    <circle cx="{cx:.2}" cy="{cy:.2}" r="{r_px:.2}" fill="{hex}" fill-opacity="{op:.3}" stroke="#18202c" stroke-width="1.0"><title>{safe_tip}</title></circle>"##
                );
            }
        }
    }

    let _ = writeln!(svg, "  </g>");
    let _ = writeln!(
        svg,
        r##"  <text x="16" y="26" fill="#dce6f5" font-family="monospace" font-size="13">{safe_title} — {safe_track}</text>"##
    );
    let _ = writeln!(svg, "</svg>");
    svg
}

// ============================================================================
// 11. Interactive Dockable `egui` 3D Structure Viewport Panel
// ============================================================================

/// Stateful controller for the dockable 3D Biomolecular Structure Viewport (`nmr-gui`).
pub struct Mol3dViewerPanel {
    pub ensemble: Option<StructureEnsemble3d>,
    pub camera: ArcballCamera,
    pub scene_cfg: SceneBuildConfig,
    pub track_data: QuantitativeTrackData,
    pub restraints: Vec<DistanceRestraint3d>,
    pub restraint_filter: RestraintFilterConfig,
    pub restraint_overlay: RestraintOverlayBuild,
    pub scene: Arc<Mol3dSceneBuffers>,
    pub last_picked_hit: Option<Picked3dHit>,
    pub status_banner: String,
    pub show_paste_modal: bool,
    pub paste_structure_buf: String,
    pub paste_restraints_buf: String,
    pub last_exported_svg: Option<String>,
    pub last_exported_png_bytes: Option<Vec<u8>>,
    next_scene_gen: u64,
    scene_dirty: bool,
    putty_only_dirty: bool,
    restraints_only_dirty: bool,
}

impl Default for Mol3dViewerPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Mol3dViewerPanel {
    pub fn new() -> Self {
        Self {
            ensemble: None,
            camera: ArcballCamera::default(),
            scene_cfg: SceneBuildConfig::default(),
            track_data: QuantitativeTrackData::default(),
            restraints: Vec::new(),
            restraint_filter: RestraintFilterConfig::default(),
            restraint_overlay: RestraintOverlayBuild::default(),
            scene: Arc::new(Mol3dSceneBuffers::default()),
            last_picked_hit: None,
            status_banner: "Drop or load a .pdb / .cif structure to inspect in 3D.".to_string(),
            show_paste_modal: false,
            paste_structure_buf: String::new(),
            paste_restraints_buf: String::new(),
            last_exported_svg: None,
            last_exported_png_bytes: None,
            next_scene_gen: 1,
            scene_dirty: false,
            putty_only_dirty: false,
            restraints_only_dirty: false,
        }
    }

    /// Parse `.pdb` or `.cif`/`.mmcif` text, Kabsch-superpose conformers, fit camera, and rebuild
    /// GPU scene buffers.
    pub fn load_structure_text(
        &mut self,
        text: &str,
        nmr_seq: &[String],
        nmr_seq_ids: &[i32],
    ) -> Result<(), String> {
        let mut ens = parse_structure_auto(text)?;
        ens.bind_to_nmr_sequence(nmr_seq, nmr_seq_ids);
        self.camera.fit_ensemble(&ens);
        self.status_banner = format!(
            "{} — {} conformer(s), {} residues, {} atoms (mean CA RMSD {:.2} Å)",
            ens.title,
            ens.models.len(),
            ens.residue_count_model0(),
            ens.total_atom_count(),
            ens.mean_ensemble_ca_rmsd
        );
        self.ensemble = Some(ens);
        self.recompute_restraints_now();
        self.rebuild_scene_now();
        Ok(())
    }

    /// Replace an already-parsed [`StructureEnsemble3d`] directly.
    pub fn set_ensemble(
        &mut self,
        mut ens: StructureEnsemble3d,
        nmr_seq: &[String],
        nmr_seq_ids: &[i32],
    ) {
        ens.bind_to_nmr_sequence(nmr_seq, nmr_seq_ids);
        self.camera.fit_ensemble(&ens);
        self.status_banner = format!(
            "{} — {} conformer(s), {} residues, {} atoms",
            ens.title,
            ens.models.len(),
            ens.residue_count_model0(),
            ens.total_atom_count()
        );
        self.ensemble = Some(ens);
        self.recompute_restraints_now();
        self.rebuild_scene_now();
    }

    /// Sync per-residue assignment confidence (`0`-based residue index -> `[0, 1]`) and `CSI`
    /// secondary-shift scores from the workbench solver state.
    pub fn sync_nmr_tracks(
        &mut self,
        confidence: &BTreeMap<usize, f64>,
        ref_data: Option<&Reference>,
        priors: Option<&Priors>,
    ) {
        let changed_conf = self.track_data.confidence != *confidence;
        if changed_conf {
            self.track_data.confidence.clone_from(confidence);
        }
        if let (Some(r), Some(p)) = (ref_data, priors) {
            self.track_data.populate_csi_from_reference(r, p);
        }
        if changed_conf
            && matches!(
                self.track_data.active_track,
                QuantitativeTrack::AssignmentConfidence | QuantitativeTrack::ChemicalShiftIndex
            )
        {
            self.putty_only_dirty = true;
        }
    }

    /// Highlight (and optionally focus camera onto) a residue selected from the 1D sequence ribbon
    /// or 3D NMR strip plot (`T12.2` reverse coupling).
    pub fn highlight_sequence_residue(&mut self, seq_idx: Option<usize>, recenter_camera: bool) {
        if self.scene_cfg.highlighted_seq_idx == seq_idx && !recenter_camera {
            return;
        }
        self.scene_cfg.highlighted_seq_idx = seq_idx;
        let seq_id_opt = seq_idx.and_then(|k| {
            self.ensemble
                .as_ref()
                .and_then(|e| e.primary_seq_ids.get(k).copied())
        });
        self.scene_cfg.highlighted_seq_id = seq_id_opt;
        self.restraint_filter.selected_seq_id = seq_id_opt;
        if self.restraint_filter.selected_residue_only {
            self.restraints_only_dirty = true;
        }
        if recenter_camera {
            if let (Some(k), Some(ens)) = (seq_idx, self.ensemble.as_ref()) {
                if let Some(ca) = ens
                    .models
                    .first()
                    .and_then(|m| m.chains.first())
                    .and_then(|c| c.residues.iter().find(|r| r.sequence_idx == Some(k)))
                    .and_then(|r| r.atom_pos("CA"))
                {
                    self.camera.focus_point(ca);
                }
            }
        }
        self.scene_dirty = true;
    }

    /// Load 3D NOE/PRE distance restraints and evaluate violations in `< 2.0 ms`.
    pub fn set_restraints(&mut self, restraints: Vec<DistanceRestraint3d>) {
        self.restraints = restraints;
        self.recompute_restraints_now();
        self.restraints_only_dirty = true;
    }

    /// Parse whitespace-delimited NOE/PRE distance restraints and update 3D violation cylinders.
    pub fn load_restraints_tsv(&mut self, text: &str) -> Result<usize, String> {
        let parsed = crate::mol3d_overlay::parse_distance_restraints_tsv(text)?;
        let count = parsed.len();
        self.set_restraints(parsed);
        self.flush_pending_updates();
        Ok(count)
    }

    fn recompute_restraints_now(&mut self) {
        if let Some(ens) = self.ensemble.as_ref() {
            self.restraint_overlay = ens
                .coord_lookup
                .evaluate_restraints(&self.restraints, &self.restraint_filter);
        } else {
            self.restraint_overlay = RestraintOverlayBuild::default();
        }
    }

    fn rebuild_scene_now(&mut self) {
        self.next_scene_gen = self.next_scene_gen.wrapping_add(1);
        if let Some(ens) = self.ensemble.as_ref() {
            let built = build_mol3d_scene(
                ens,
                &self.scene_cfg,
                &self.track_data,
                Some(&self.restraint_overlay),
                self.next_scene_gen,
            );
            self.scene = Arc::new(built);
        } else {
            self.scene = Arc::new(Mol3dSceneBuffers::default());
        }
        self.scene_dirty = false;
        self.putty_only_dirty = false;
        self.restraints_only_dirty = false;
    }

    /// Synchronously rebuild all 3D scene buffers (exposed for integration and picking tests).
    pub fn rebuild_for_test(&mut self) {
        self.rebuild_scene_now();
    }

    /// Apply any pending fast-path (`< 2 ms`) putty/colormap or restraint-cylinder updates.
    pub fn flush_pending_updates(&mut self) {
        if self.scene_dirty {
            if self.restraints_only_dirty {
                self.recompute_restraints_now();
            }
            self.rebuild_scene_now();
            return;
        }
        if self.putty_only_dirty || self.restraints_only_dirty {
            if self.restraints_only_dirty {
                self.recompute_restraints_now();
            }
            let mut updated = (*self.scene).clone();
            if self.putty_only_dirty {
                updated.update_quantitative_putty(&self.track_data);
            }
            if self.restraints_only_dirty {
                updated.set_restraint_overlay(&self.restraint_overlay);
            }
            self.next_scene_gen = updated.scene_gen.max(self.next_scene_gen + 1);
            updated.scene_gen = self.next_scene_gen;
            self.scene = Arc::new(updated);
            self.putty_only_dirty = false;
            self.restraints_only_dirty = false;
        }
    }

    /// Perform exact analytical 3D ray picking at viewport-local pixel `(local_x, local_y)` and
    /// resolve any hit into a [`Mol3dSyncAction`] (`T12.2`).
    #[allow(clippy::too_many_arguments)]
    pub fn pick_at_viewport_pixel(
        &mut self,
        local_x: f64,
        local_y: f64,
        vp_width: f64,
        vp_height: f64,
        nmr_seq: &[String],
        nmr_seq_ids: &[i32],
        ref_data: Option<&Reference>,
    ) -> Option<Mol3dSyncAction> {
        let ray = self
            .camera
            .screen_to_world_ray(local_x, local_y, vp_width, vp_height);
        let hit = pick_scene_analytical(
            &ray,
            &self.scene.pick_spheres,
            &self.scene.pick_ribbons,
            &self.scene.pick_restraints,
        )?;
        self.last_picked_hit = Some(hit.clone());

        let (eff_seq, eff_ids) = if !nmr_seq.is_empty() {
            (nmr_seq, nmr_seq_ids)
        } else {
            let ens = self.ensemble.as_ref()?;
            (
                ens.primary_sequence.as_slice(),
                ens.primary_seq_ids.as_slice(),
            )
        };

        let sync = resolve_3d_pick_sync(&hit, eff_seq, eff_ids, ref_data)?;
        self.highlight_sequence_residue(Some(sync.residue_idx), false);
        self.flush_pending_updates();
        Some(sync)
    }

    /// Render the dockable 3D Biomolecular Structure Viewport (`egui` toolbar + `wgpu` canvas) and
    /// return a [`Mol3dSyncAction`] whenever the user clicks a residue/atom/restraint in 3D.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] wgpu_render_state: Option<
            &egui_wgpu::RenderState,
        >,
        nmr_seq: &[String],
        nmr_seq_ids: &[i32],
        ref_data: Option<&Reference>,
    ) -> Option<Mol3dSyncAction> {
        let mut emitted_sync: Option<Mol3dSyncAction> = None;

        // Top compact control strip 1: Representation, Projection, Load/Paste, Fit, PNG/SVG export
        ui.horizontal_wrapped(|ui| {
            ui.strong("3D Structure");
            egui::ComboBox::from_id_salt("mol3d_repr_combo")
                .selected_text(self.scene_cfg.style.label())
                .width(138.0)
                .show_ui(ui, |ui| {
                    for style in RepresentationStyle::ALL {
                        if ui
                            .selectable_value(&mut self.scene_cfg.style, style, style.label())
                            .changed()
                        {
                            self.scene_dirty = true;
                        }
                    }
                });

            let proj_btn = match self.camera.projection {
                ProjectionMode::Perspective => "Persp",
                ProjectionMode::Orthographic => "Ortho",
            };
            if ui
                .button(proj_btn)
                .on_hover_text("Toggle Perspective / Orthographic camera projection")
                .clicked()
            {
                self.camera.projection = match self.camera.projection {
                    ProjectionMode::Perspective => ProjectionMode::Orthographic,
                    ProjectionMode::Orthographic => ProjectionMode::Perspective,
                };
            }

            if ui
                .button("Fit")
                .on_hover_text("Reset view & Z-slab around loaded structure")
                .clicked()
            {
                if let Some(ens) = self.ensemble.as_ref() {
                    self.camera.fit_ensemble(ens);
                }
            }

            if ui
                .button("Paste PDB/CIF")
                .on_hover_text("Paste .pdb / .cif coordinates or NOE/PRE restraints")
                .clicked()
            {
                self.show_paste_modal = !self.show_paste_modal;
            }

            #[cfg(not(target_arch = "wasm32"))]
            if ui
                .button("PNG")
                .on_hover_text("Export high-DPI offscreen 4x MSAA PNG snapshot")
                .clicked()
            {
                if let Some(rs) = wgpu_render_state {
                    let mut renderer = rs.renderer.write();
                    if let Some(res) = renderer
                        .callback_resources
                        .get_mut::<Mol3dPipelineResource>()
                    {
                        self.flush_pending_updates();
                        let uni = self.camera.to_gpu_uniforms(1600.0, 1200.0, DEFAULT_BG_RGBA);
                        if let Ok(png) = res.export_offscreen_png(
                            &rs.device,
                            &rs.queue,
                            1600,
                            1200,
                            &uni,
                            &self.scene,
                        ) {
                            let out_path = std::env::temp_dir().join("nmr_structure_3d.png");
                            let _ = std::fs::write(&out_path, &png);
                            self.last_exported_png_bytes = Some(png);
                            self.status_banner =
                                format!("Exported 1600x1200 PNG ({})", out_path.display());
                        }
                    }
                }
            }

            if ui
                .button("SVG")
                .on_hover_text("Export deterministic XML-escaped vector SVG")
                .clicked()
            {
                if let Some(ens) = self.ensemble.as_ref() {
                    let svg = export_structure_svg(
                        ens,
                        &self.camera,
                        &self.scene_cfg,
                        &self.track_data,
                        Some(&self.restraint_overlay),
                        1200,
                        900,
                    );
                    ui.ctx().copy_text(svg.clone());
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let out_path = std::env::temp_dir().join("nmr_structure_3d.svg");
                        let _ = std::fs::write(&out_path, &svg);
                    }
                    self.last_exported_svg = Some(svg);
                    self.status_banner = "Exported vector SVG (copied to clipboard).".to_string();
                }
            }
        });

        // Control strip 2: Quantitative Putty / Colormap Track + Ensemble & Z-Slab sliders
        ui.horizontal_wrapped(|ui| {
            ui.label("Color:");
            egui::ComboBox::from_id_salt("mol3d_track_combo")
                .selected_text(self.track_data.active_track.label())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for trk in QuantitativeTrack::ALL {
                        if ui
                            .selectable_value(&mut self.track_data.active_track, trk, trk.label())
                            .changed()
                        {
                            self.track_data.colormap = trk.default_colormap();
                            self.putty_only_dirty = true;
                        }
                    }
                });

            if self.track_data.active_track != QuantitativeTrack::SecondaryStructure {
                egui::ComboBox::from_id_salt("mol3d_cmap_combo")
                    .selected_text(self.track_data.colormap.label())
                    .width(112.0)
                    .show_ui(ui, |ui| {
                        for cm in ColormapKind::ALL {
                            if ui
                                .selectable_value(&mut self.track_data.colormap, cm, cm.label())
                                .changed()
                            {
                                self.putty_only_dirty = true;
                            }
                        }
                    });
            }

            if ui
                .checkbox(&mut self.track_data.putty_enabled, "Putty")
                .on_hover_text("Scale ribbon tube radius by the active quantitative NMR track")
                .changed()
            {
                self.putty_only_dirty = true;
            }

            if self.ensemble.as_ref().is_some_and(|e| e.models.len() > 1) {
                if ui
                    .checkbox(&mut self.scene_cfg.show_all_ensemble_models, "Bundle")
                    .changed()
                {
                    self.scene_dirty = true;
                }
                if self.scene_cfg.show_all_ensemble_models
                    && ui
                        .add(
                            egui::Slider::new(
                                &mut self.scene_cfg.ensemble_bundle_opacity,
                                0.08..=0.85,
                            )
                            .text("α"),
                        )
                        .changed()
                {
                    self.scene_dirty = true;
                }
            }

            ui.label("Slab:");
            let mut near_f = self.camera.slab_near_offset as f32;
            let mut far_f = self.camera.slab_far_offset as f32;
            if ui
                .add(egui::DragValue::new(&mut near_f).speed(0.5).prefix("N "))
                .changed()
            {
                self.camera.slab_near_offset =
                    (near_f as f64).min(self.camera.slab_far_offset - 1.0);
            }
            if ui
                .add(egui::DragValue::new(&mut far_f).speed(0.5).prefix("F "))
                .changed()
            {
                self.camera.slab_far_offset =
                    (far_f as f64).max(self.camera.slab_near_offset + 1.0);
            }
        });

        // Control strip 3: NOE/PRE Restraints filter bar when restraints are loaded
        if !self.restraints.is_empty() {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .checkbox(&mut self.restraint_filter.enabled, "NOE/PRE")
                    .changed()
                {
                    self.restraints_only_dirty = true;
                }
                let sat_lbl = format!("Sat ({})", self.restraint_overlay.satisfied_count);
                let min_lbl = format!("<0.5Å ({})", self.restraint_overlay.minor_violation_count);
                let maj_lbl = format!("≥0.5Å ({})", self.restraint_overlay.major_violation_count);
                if ui
                    .checkbox(&mut self.restraint_filter.show_satisfied, sat_lbl)
                    .changed()
                    || ui
                        .checkbox(&mut self.restraint_filter.show_minor, min_lbl)
                        .changed()
                    || ui
                        .checkbox(&mut self.restraint_filter.show_major, maj_lbl)
                        .changed()
                    || ui
                        .checkbox(&mut self.restraint_filter.selected_residue_only, "Sel Only")
                        .changed()
                {
                    self.restraints_only_dirty = true;
                }
                ui.small(format!(
                    "RMS {:.2} Å | Max {:.2} Å",
                    self.restraint_overlay.rms_violation_angstrom,
                    self.restraint_overlay.max_violation_angstrom
                ));
            });
        }

        if self.show_paste_modal {
            ui.group(|ui| {
                ui.label("Paste .pdb or .cif/.mmcif coordinates:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.paste_structure_buf)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    if ui.button("Load Coordinates").clicked() {
                        let raw = self.paste_structure_buf.clone();
                        match self.load_structure_text(&raw, nmr_seq, nmr_seq_ids) {
                            Ok(()) => self.show_paste_modal = false,
                            Err(e) => self.status_banner = format!("Structure load error: {e}"),
                        }
                    }
                    if ui.button("Close").clicked() {
                        self.show_paste_modal = false;
                    }
                });
                ui.label("Or paste NOE/PRE restraints (seq1 atom1 seq2 atom2 upper_A):");
                ui.add(
                    egui::TextEdit::multiline(&mut self.paste_restraints_buf)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                if ui.button("Load Restraints").clicked() {
                    let raw = self.paste_restraints_buf.clone();
                    match self.load_restraints_tsv(&raw) {
                        Ok(n) => {
                            self.status_banner = format!("Loaded {n} 3D NOE/PRE restraints.");
                            self.show_paste_modal = false;
                        }
                        Err(e) => self.status_banner = format!("Restraint parse error: {e}"),
                    }
                }
            });
        }

        self.flush_pending_updates();

        // Allocate remaining panel region to the interactive 3D wgpu canvas
        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(120.0), (avail.y - 20.0).max(160.0));
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

        // Arcball rotation (primary drag), XY pan (secondary / Shift+drag), scroll zoom
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            let shift = ui.input(|i| i.modifiers.shift);
            if shift {
                self.camera
                    .drag_pan(delta.x as f64, delta.y as f64, rect.height() as f64);
            } else if let Some(pos) = response.interact_pointer_pos() {
                let prev = pos - delta;
                let to_ndc = |p: egui::Pos2| -> [f64; 2] {
                    [
                        (2.0 * ((p.x - rect.left()) / rect.width().max(1.0)) - 1.0) as f64,
                        (1.0 - 2.0 * ((p.y - rect.top()) / rect.height().max(1.0))) as f64,
                    ]
                };
                self.camera.drag_arcball(to_ndc(prev), to_ndc(pos));
            }
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            self.camera
                .drag_pan(delta.x as f64, delta.y as f64, rect.height() as f64);
        }

        if response.hovered() {
            let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_y.abs() > 0.1 {
                let factor = (-0.0025 * (scroll_y as f64)).exp();
                self.camera.zoom_by(factor);
            }
        }

        // Analytical 3D Ray Picking on click -> Sequence Ribbon + 2D Covalent Graph + SyncCrosshair
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                let lx = (click_pos.x - rect.left()) as f64;
                let ly = (click_pos.y - rect.top()) as f64;
                if let Some(sync) = self.pick_at_viewport_pixel(
                    lx,
                    ly,
                    rect.width() as f64,
                    rect.height() as f64,
                    nmr_seq,
                    nmr_seq_ids,
                    ref_data,
                ) {
                    self.status_banner = format!(
                        "Picked {}{} : {} (synced 2D HSQC / 3D strips & covalent graph)",
                        sync.residue_name, sync.seq_id, sync.clicked_atom
                    );
                    emitted_sync = Some(sync);
                }
            }
        }

        let ppp = ui.ctx().pixels_per_point().max(1.0);
        let vp_w = ((rect.width() * ppp).round() as u32).clamp(2, 4096);
        let vp_h = ((rect.height() * ppp).round() as u32).clamp(2, 4096);
        let uniforms = self
            .camera
            .to_gpu_uniforms(vp_w as f32, vp_h as f32, DEFAULT_BG_RGBA);

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            Mol3dPaintCallback {
                width_px: vp_w,
                height_px: vp_h,
                uniforms,
                scene: Arc::clone(&self.scene),
            },
        ));

        ui.painter().rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 72)),
            egui::StrokeKind::Inside,
        );

        ui.small(&self.status_banner);
        emitted_sync
    }
}
