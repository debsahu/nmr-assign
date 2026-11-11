//! MDL Molfile V2000 & V3000 (`.mol`) and Multi-Record Structure-Data File (`.sdf`)
//! Reader and Writer.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0:
//! `molv3.ts`, `chem.ts`) and the public Accelrys/BIOVIA CTfile Formats specification.

use super::graph::{
    symbol_from_atomic_number, Atom, BondKind, ChemError, DoubleBondGeometry, Molecule, Radical,
};
use super::stereo::kekulize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single record in a multi-molecule Structure-Data File (`.sdf`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SdfRecord {
    pub molecule: Molecule,
    pub properties: BTreeMap<String, String>,
}

impl SdfRecord {
    pub fn new(mut molecule: Molecule, properties: BTreeMap<String, String>) -> Self {
        for (k, v) in &properties {
            molecule.properties.insert(k.clone(), v.clone());
        }
        Self {
            molecule,
            properties,
        }
    }
}

/// Auto-detect and parse either an MDL V2000 or V3000 `.mol` block into a `Molecule`.
pub fn read_mol(mol_text: &str) -> Result<Molecule, ChemError> {
    let normalized = mol_text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalized.lines().collect();
    if lines.len() < 4 {
        return Err(ChemError::new(
            "Invalid MOL file: fewer than 4 header/counts lines",
        ));
    }

    let is_v3000 = lines[3].contains("V3000")
        || lines
            .iter()
            .take(12)
            .any(|l| l.starts_with("M  V30 BEGIN CTAB"));

    if is_v3000 {
        read_mol_v3000_lines(&lines)
    } else {
        read_mol_v2000_lines(&lines)
    }
}

/// Write a `Molecule` in MDL Molfile V2000 format (or automatically V3000 via `write_mol_auto` when >999 atoms).
pub fn write_mol_v2000(mol: &Molecule) -> Result<String, ChemError> {
    if mol.atoms.len() > 999 || mol.bonds.len() > 999 {
        return Err(ChemError::new(
            "Molecule exceeds 999 atoms/bonds limit of V2000; use write_mol_v3000 instead",
        ));
    }

    let is_3d = mol.has_3d_coords();
    let dim_code = if is_3d { "3D" } else { "2D" };
    let mut out = String::new();

    // Line 1: Molecule name (single line)
    out.push_str(mol.name.lines().next().unwrap_or(""));
    out.push('\n');
    // Line 2: Program / timestamp / dimension code (`2D` or `3D` at cols 20..22)
    out.push_str(&format!("  OctetMol          {}\n", dim_code));
    // Line 3: Comment
    out.push_str(mol.comment.lines().next().unwrap_or(""));
    out.push('\n');

    // Line 4: Counts line `aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv`
    let has_chiral = mol.bonds.iter().any(|b| b.kind.is_stereo_display());
    let chiral_flag = if has_chiral { 1 } else { 0 };
    out.push_str(&format!(
        "{:>3}{:>3}  0  0{:>3}  0  0  0  0  0999 V2000\n",
        mol.atoms.len(),
        mol.bonds.len(),
        chiral_flag
    ));

    // Atom block
    let mut chg_entries: Vec<(usize, i8)> = Vec::new();
    let mut iso_entries: Vec<(usize, u16)> = Vec::new();
    let mut rad_entries: Vec<(usize, u8)> = Vec::new();
    let mut alias_entries: Vec<(usize, String)> = Vec::new();

    for (idx, a) in mol.atoms.iter().enumerate() {
        let z_val = a.z_coord.unwrap_or(0.0);
        let sym = if a.superatom_label.is_some() {
            alias_entries.push((idx + 1, a.superatom_label.clone().unwrap()));
            if a.z > 0 {
                symbol_from_atomic_number(a.z)
            } else {
                "*"
            }
        } else if a.z > 0 {
            symbol_from_atomic_number(a.z)
        } else {
            "*"
        };

        if a.formal_charge != 0 {
            chg_entries.push((idx + 1, a.formal_charge));
        }
        if let Some(iso) = a.isotope {
            iso_entries.push((idx + 1, iso));
        }
        match a.radical {
            Radical::None => {}
            Radical::Singlet => rad_entries.push((idx + 1, 1)),
            Radical::Doublet => rad_entries.push((idx + 1, 2)),
            Radical::Triplet => rad_entries.push((idx + 1, 3)),
        }

        let val_code: u8 = match a.explicit_valence {
            Some(0) => 15,
            Some(v) if v <= 14 => v,
            _ => 0,
        };
        let amap = a.atom_map.unwrap_or(0);

        out.push_str(&format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0{:>3}  0  0  0{:>3}  0  0\n",
            a.x, a.y, z_val, sym, val_code, amap
        ));
    }

    // Bond block
    for b in &mol.bonds {
        let (btype, bstereo) = match b.kind {
            BondKind::Single | BondKind::Bold | BondKind::Dashed => (1, 0),
            BondKind::WedgeUp | BondKind::HollowWedge => (1, 1),
            BondKind::HashDown => (1, 6),
            BondKind::Wavy => (1, 4),
            BondKind::Double => {
                let st = if b.double_geometry == DoubleBondGeometry::Either {
                    3
                } else {
                    0
                };
                (2, st)
            }
            BondKind::Triple => (3, 0),
            BondKind::Aromatic => (4, 0),
            BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic => (8, 0),
        };
        out.push_str(&format!(
            "{:>3}{:>3}{:>3}{:>3}  0  0  0\n",
            b.a + 1,
            b.b + 1,
            btype,
            bstereo
        ));
    }

    // Superatom aliases (`A  nnn\n<label>`)
    for (num1, label) in alias_entries {
        out.push_str(&format!("A  {:>3}\n{}\n", num1, label));
    }

    // M  CHG (up to 8 per line)
    for chunk in chg_entries.chunks(8) {
        out.push_str(&format!("M  CHG{:>3}", chunk.len()));
        for &(a_num, q) in chunk {
            out.push_str(&format!(" {:>3} {:>3}", a_num, q));
        }
        out.push('\n');
    }

    // M  ISO (up to 8 per line)
    for chunk in iso_entries.chunks(8) {
        out.push_str(&format!("M  ISO{:>3}", chunk.len()));
        for &(a_num, iso) in chunk {
            out.push_str(&format!(" {:>3} {:>3}", a_num, iso));
        }
        out.push('\n');
    }

    // M  RAD (up to 8 per line)
    for chunk in rad_entries.chunks(8) {
        out.push_str(&format!("M  RAD{:>3}", chunk.len()));
        for &(a_num, rcode) in chunk {
            out.push_str(&format!(" {:>3} {:>3}", a_num, rcode));
        }
        out.push('\n');
    }

    out.push_str("M  END\n");
    Ok(out)
}

/// Write a `Molecule` in MDL Extended Molfile V3000 format (supports arbitrary atom counts,
/// `CFG=` stereobonds, `CHG=`, `MASS=`, `RAD=`, `ENDPTS=` multi-center organometallic bonds, and `SUP` Sgroups).
pub fn write_mol_v3000(mol: &Molecule) -> String {
    let is_3d = mol.has_3d_coords();
    let dim_code = if is_3d { "3D" } else { "2D" };
    let mut out = String::new();

    out.push_str(mol.name.lines().next().unwrap_or(""));
    out.push('\n');
    out.push_str(&format!("  OctetMol          {}\n", dim_code));
    out.push_str(mol.comment.lines().next().unwrap_or(""));
    out.push('\n');
    out.push_str("  0  0  0     0  0            999 V3000\n");
    out.push_str("M  V30 BEGIN CTAB\n");

    let sgroup_atoms: Vec<(usize, &str)> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.superatom_label.as_deref().map(|lbl| (i, lbl)))
        .collect();
    let chiral_flag = if mol.bonds.iter().any(|b| b.kind.is_stereo_display()) {
        1
    } else {
        0
    };

    out.push_str(&format!(
        "M  V30 COUNTS {} {} {} 0 {}\n",
        mol.atoms.len(),
        mol.bonds.len(),
        sgroup_atoms.len(),
        chiral_flag
    ));

    out.push_str("M  V30 BEGIN ATOM\n");
    for (idx, a) in mol.atoms.iter().enumerate() {
        let sym = if a.z > 0 {
            symbol_from_atomic_number(a.z)
        } else {
            "*"
        };
        let z_val = a.z_coord.unwrap_or(0.0);
        let amap = a.atom_map.unwrap_or(0);
        let mut line = format!(
            "M  V30 {} {} {:.4} {:.4} {:.4} {}",
            idx + 1,
            sym,
            a.x,
            a.y,
            z_val,
            amap
        );
        if a.formal_charge != 0 {
            line.push_str(&format!(" CHG={}", a.formal_charge));
        }
        if let Some(iso) = a.isotope {
            line.push_str(&format!(" MASS={}", iso));
        }
        match a.radical {
            Radical::None => {}
            Radical::Singlet => line.push_str(" RAD=1"),
            Radical::Doublet => line.push_str(" RAD=2"),
            Radical::Triplet => line.push_str(" RAD=3"),
        }
        if let Some(ev) = a.explicit_valence {
            let v_val: i32 = if ev == 0 { -1 } else { ev as i32 };
            line.push_str(&format!(" VAL={}", v_val));
        }
        if let Some(ap) = a.attachment_point {
            line.push_str(&format!(" ATTCHPT={}", ap));
        }
        push_v30_wrapped(&mut out, &line);
    }
    out.push_str("M  V30 END ATOM\n");

    if !mol.bonds.is_empty() {
        out.push_str("M  V30 BEGIN BOND\n");
        for (bi, b) in mol.bonds.iter().enumerate() {
            let (btype, cfg) = match b.kind {
                BondKind::Single | BondKind::Bold | BondKind::Dashed => (1, 0),
                BondKind::WedgeUp | BondKind::HollowWedge => (1, 1),
                BondKind::Wavy => (1, 2),
                BondKind::HashDown => (1, 3),
                BondKind::Double => {
                    let c = if b.double_geometry == DoubleBondGeometry::Either {
                        2
                    } else {
                        0
                    };
                    (2, c)
                }
                BondKind::Triple => (3, 0),
                BondKind::Aromatic => (4, 0),
                BondKind::Dative => (9, 0),
                BondKind::Hydrogen => (10, 0),
                BondKind::Ionic => (8, 0),
            };
            let mut line = format!("M  V30 {} {} {} {}", bi + 1, btype, b.a + 1, b.b + 1);
            if cfg != 0 {
                line.push_str(&format!(" CFG={}", cfg));
            }
            // Multi-center η-bond metadata (`ENDPTS=(k m1 m2 ...)` + `ATTACH=ALL`)
            let mc = if !mol.atoms[b.b].multi_center_atoms.is_empty() {
                Some(&mol.atoms[b.b].multi_center_atoms)
            } else if !mol.atoms[b.a].multi_center_atoms.is_empty() {
                Some(&mol.atoms[b.a].multi_center_atoms)
            } else {
                None
            };
            if let Some(pts) = mc {
                let ids: Vec<String> = pts.iter().map(|&u| (u + 1).to_string()).collect();
                line.push_str(&format!(
                    " ATTACH=ALL ENDPTS=({} {})",
                    pts.len(),
                    ids.join(" ")
                ));
            }
            push_v30_wrapped(&mut out, &line);
        }
        out.push_str("M  V30 END BOND\n");
    }

    if !sgroup_atoms.is_empty() {
        out.push_str("M  V30 BEGIN SGROUP\n");
        for (sg_idx, &(atom_i, lbl)) in sgroup_atoms.iter().enumerate() {
            let escaped = lbl.replace('"', "\"\"");
            let line = format!(
                "M  V30 {} SUP 0 ATOMS=(1 {}) LABEL=\"{}\"",
                sg_idx + 1,
                atom_i + 1,
                escaped
            );
            push_v30_wrapped(&mut out, &line);
        }
        out.push_str("M  V30 END SGROUP\n");
    }

    out.push_str("M  V30 END CTAB\n");
    out.push_str("M  END\n");
    out
}

fn push_v30_wrapped(out: &mut String, line: &str) {
    if line.len() <= 78 {
        out.push_str(line);
        out.push('\n');
        return;
    }
    // Wrap long V3000 lines at spaces before column 77 using trailing `-` continuation
    let body = line.strip_prefix("M  V30 ").unwrap_or(line);
    let mut rem = body;
    while rem.len() > 70 {
        let split_at = rem[..70].rfind(' ').unwrap_or(70);
        out.push_str("M  V30 ");
        out.push_str(&rem[..split_at]);
        out.push_str("-\n");
        rem = rem[split_at..].trim_start();
    }
    out.push_str("M  V30 ");
    out.push_str(rem);
    out.push('\n');
}

// ============================================================================
// V2000 & V3000 Readers
// ============================================================================

fn read_mol_v2000_lines(lines: &[&str]) -> Result<Molecule, ChemError> {
    let mut mol = Molecule::with_name(lines[0].trim());
    let header_3d = lines[1].len() >= 22 && &lines[1][20..22] == "3D";
    mol.comment = lines[2].trim().to_string();

    let counts = lines[3];
    if counts.len() < 6 {
        return Err(ChemError::new("Malformed V2000 counts line"));
    }
    let n_atoms = counts[0..3]
        .trim()
        .parse::<usize>()
        .map_err(|_| ChemError::new("Invalid atom count in V2000 counts line"))?;
    let n_bonds = counts[3..6]
        .trim()
        .parse::<usize>()
        .map_err(|_| ChemError::new("Invalid bond count in V2000 counts line"))?;

    if lines.len() < 4 + n_atoms + n_bonds {
        return Err(ChemError::new(
            "Truncated V2000 MOL file: fewer lines than atom + bond counts",
        ));
    }

    let mut raw_z = Vec::with_capacity(n_atoms);
    let mut any_nonzero_z = header_3d;
    let mut has_m_chg = false;
    let mut legacy_charges = Vec::new();

    for i in 0..n_atoms {
        let l = lines[4 + i];
        let (x, y, z, sym, chg_code, val_code, amap) = parse_v2000_atom_line(l)?;
        if z.abs() > 1e-4 {
            any_nonzero_z = true;
        }
        raw_z.push(z);

        let mut atom = Atom::new(&sym);
        atom.x = x;
        atom.y = y;
        if val_code == 15 {
            atom.explicit_valence = Some(0);
            atom.explicit_hydrogens = Some(0);
        } else if (1..=14).contains(&val_code) {
            atom.explicit_valence = Some(val_code);
        }
        if amap > 0 {
            atom.atom_map = Some(amap);
        }
        if chg_code != 0 {
            legacy_charges.push((i, chg_code));
        }
        mol.add_atom(atom);
    }

    if any_nonzero_z {
        for (i, z) in raw_z.into_iter().enumerate() {
            mol.atoms[i].z_coord = Some(z);
        }
    }

    let mut has_arom_bond = false;
    for j in 0..n_bonds {
        let l = lines[4 + n_atoms + j];
        if l.len() < 9 {
            return Err(ChemError::new("Malformed V2000 bond line"));
        }
        let a1 = l[0..3]
            .trim()
            .parse::<usize>()
            .map_err(|_| ChemError::new("Invalid bond atom 1 in V2000"))?;
        let a2 = l[3..6]
            .trim()
            .parse::<usize>()
            .map_err(|_| ChemError::new("Invalid bond atom 2 in V2000"))?;
        let btype = l[6..9]
            .trim()
            .parse::<u8>()
            .map_err(|_| ChemError::new("Invalid bond type in V2000"))?;
        let bstereo = if l.len() >= 12 {
            l[9..12].trim().parse::<u8>().unwrap_or(0)
        } else {
            0
        };
        if a1 == 0 || a2 == 0 || a1 > n_atoms || a2 > n_atoms {
            return Err(ChemError::new("Bond atom index out of range in V2000"));
        }
        let kind = match (btype, bstereo) {
            (1, 1) => BondKind::WedgeUp,
            (1, 6) => BondKind::HashDown,
            (1, 4) => BondKind::Wavy,
            (1, _) => BondKind::Single,
            (2, _) => BondKind::Double,
            (3, _) => BondKind::Triple,
            (4, _) => {
                has_arom_bond = true;
                BondKind::Aromatic
            }
            (8 | 9, _) => BondKind::Dative,
            _ => BondKind::Single,
        };
        let bi = mol.add_bond(a1 - 1, a2 - 1, kind);
        if btype == 2 && bstereo == 3 {
            mol.bonds[bi].double_geometry = DoubleBondGeometry::Either;
        }
    }

    // Parse property block (`M  CHG`, `M  ISO`, `M  RAD`, `A  `, `M  END`)
    let mut line_idx = 4 + n_atoms + n_bonds;
    while line_idx < lines.len() {
        let l = lines[line_idx];
        if l.starts_with("M  END") {
            break;
        }
        if let Some(rest) = l.strip_prefix("A  ") {
            if let Ok(a_num) = rest.trim().parse::<usize>() {
                if line_idx + 1 < lines.len() && a_num >= 1 && a_num <= n_atoms {
                    let alias = lines[line_idx + 1].trim().to_string();
                    mol.atoms[a_num - 1].superatom_label = Some(alias.clone());
                    mol.atoms[a_num - 1].symbol = alias;
                    line_idx += 2;
                    continue;
                }
            }
        } else if let Some(rest) = l.strip_prefix("M  CHG") {
            has_m_chg = true;
            for (a_num, val) in parse_m_pairs(rest) {
                if a_num >= 1 && a_num <= n_atoms {
                    mol.atoms[a_num - 1].formal_charge = val as i8;
                }
            }
        } else if let Some(rest) = l.strip_prefix("M  ISO") {
            for (a_num, val) in parse_m_pairs(rest) {
                if a_num >= 1 && a_num <= n_atoms && val > 0 {
                    mol.atoms[a_num - 1].isotope = Some(val as u16);
                }
            }
        } else if let Some(rest) = l.strip_prefix("M  RAD") {
            for (a_num, val) in parse_m_pairs(rest) {
                if a_num >= 1 && a_num <= n_atoms {
                    mol.atoms[a_num - 1].radical = match val {
                        1 => Radical::Singlet,
                        2 => Radical::Doublet,
                        3 => Radical::Triplet,
                        _ => Radical::None,
                    };
                }
            }
        }
        line_idx += 1;
    }

    if !has_m_chg {
        for (idx, code) in legacy_charges {
            mol.atoms[idx].formal_charge = match code {
                1 => 3,
                2 => 2,
                3 => 1,
                5 => -1,
                6 => -2,
                7 => -3,
                _ => 0,
            };
        }
    }

    if has_arom_bond {
        kekulize(&mut mol)?;
    }

    Ok(mol)
}

type V2000AtomRow = (f64, f64, f64, String, u8, u8, u16);

fn parse_v2000_atom_line(l: &str) -> Result<V2000AtomRow, ChemError> {
    if l.len() >= 34 {
        let x = l[0..10]
            .trim()
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid X in V2000 atom line"))?;
        let y = l[10..20]
            .trim()
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid Y in V2000 atom line"))?;
        let z = l[20..30]
            .trim()
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid Z in V2000 atom line"))?;
        let sym_end = l.len().min(34);
        let sym = l[31..sym_end].trim().to_string();
        let chg_code = if l.len() >= 39 {
            l[36..39].trim().parse::<u8>().unwrap_or(0)
        } else {
            0
        };
        let val_code = if l.len() >= 51 {
            l[48..51].trim().parse::<u8>().unwrap_or(0)
        } else {
            0
        };
        let amap = if l.len() >= 63 {
            l[60..63].trim().parse::<u16>().unwrap_or(0)
        } else {
            0
        };
        Ok((x, y, z, sym, chg_code, val_code, amap))
    } else {
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(ChemError::new("Truncated V2000 atom line"));
        }
        let x = toks[0]
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid X"))?;
        let y = toks[1]
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid Y"))?;
        let z = toks[2]
            .parse::<f64>()
            .map_err(|_| ChemError::new("Invalid Z"))?;
        Ok((x, y, z, toks[3].to_string(), 0, 0, 0))
    }
}

fn parse_m_pairs(rest: &str) -> Vec<(usize, i32)> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    if toks.is_empty() {
        return Vec::new();
    }
    let cnt = toks[0].parse::<usize>().unwrap_or(0);
    let mut out = Vec::with_capacity(cnt);
    for k in 0..cnt {
        if 1 + 2 * k + 1 < toks.len() {
            if let (Ok(a), Ok(v)) = (
                toks[1 + 2 * k].parse::<usize>(),
                toks[1 + 2 * k + 1].parse::<i32>(),
            ) {
                out.push((a, v));
            }
        }
    }
    out
}

fn read_mol_v3000_lines(lines: &[&str]) -> Result<Molecule, ChemError> {
    let mut mol = Molecule::with_name(lines[0].trim());
    let header_3d = lines[1].len() >= 22 && &lines[1][20..22] == "3D";
    mol.comment = lines[2].trim().to_string();

    // Unwrap `M  V30 ...-` continuations into logical V3000 statements
    let mut stmts: Vec<String> = Vec::new();
    let mut accum = String::new();
    for &l in &lines[3..] {
        if l.starts_with("M  END") {
            break;
        }
        if let Some(body) = l.strip_prefix("M  V30 ") {
            if let Some(stripped) = body.strip_suffix('-') {
                accum.push_str(stripped);
            } else {
                accum.push_str(body);
                stmts.push(std::mem::take(&mut accum));
            }
        }
    }

    let mut section = "";
    let mut id_to_idx: BTreeMap<usize, usize> = BTreeMap::new();
    let mut raw_z: Vec<f64> = Vec::new();
    let mut any_3d = header_3d;
    let mut has_arom_bond = false;

    for stmt in &stmts {
        let trimmed = stmt.trim();
        match trimmed {
            "BEGIN ATOM" => {
                section = "ATOM";
                continue;
            }
            "END ATOM" => {
                section = "";
                continue;
            }
            "BEGIN BOND" => {
                section = "BOND";
                continue;
            }
            "END BOND" => {
                section = "";
                continue;
            }
            "BEGIN SGROUP" => {
                section = "SGROUP";
                continue;
            }
            "END SGROUP" => {
                section = "";
                continue;
            }
            _ => {}
        }

        if section == "ATOM" {
            let toks = tokenize_v30_line(trimmed);
            if toks.len() < 6 {
                return Err(ChemError::new("Malformed V3000 ATOM record"));
            }
            let atom_id = toks[0]
                .parse::<usize>()
                .map_err(|_| ChemError::new("Invalid V3000 atom ID"))?;
            let sym = &toks[1];
            let x = toks[2]
                .parse::<f64>()
                .map_err(|_| ChemError::new("Invalid V3000 X"))?;
            let y = toks[3]
                .parse::<f64>()
                .map_err(|_| ChemError::new("Invalid V3000 Y"))?;
            let z = toks[4]
                .parse::<f64>()
                .map_err(|_| ChemError::new("Invalid V3000 Z"))?;
            let amap = toks[5].parse::<u16>().unwrap_or(0);

            if z.abs() > 1e-4 {
                any_3d = true;
            }
            raw_z.push(z);

            let mut atom = Atom::new(sym);
            atom.x = x;
            atom.y = y;
            if amap > 0 {
                atom.atom_map = Some(amap);
            }

            for kv in &toks[6..] {
                if let Some((k, v)) = kv.split_once('=') {
                    match k {
                        "CHG" => {
                            atom.formal_charge = v.parse::<i8>().unwrap_or(0);
                        }
                        "MASS" => {
                            if let Ok(m) = v.parse::<u16>() {
                                atom.isotope = Some(m);
                            }
                        }
                        "RAD" => {
                            atom.radical = match v {
                                "1" => Radical::Singlet,
                                "2" => Radical::Doublet,
                                "3" => Radical::Triplet,
                                _ => Radical::None,
                            };
                        }
                        "VAL" => {
                            if let Ok(vv) = v.parse::<i32>() {
                                if vv == -1 {
                                    atom.explicit_valence = Some(0);
                                    atom.explicit_hydrogens = Some(0);
                                } else if vv > 0 {
                                    atom.explicit_valence = Some(vv as u8);
                                }
                            }
                        }
                        "ATTCHPT" => {
                            atom.attachment_point = v.parse::<u8>().ok();
                        }
                        _ => {}
                    }
                }
            }

            let idx = mol.add_atom(atom);
            id_to_idx.insert(atom_id, idx);
        } else if section == "BOND" {
            let toks = tokenize_v30_line(trimmed);
            if toks.len() < 4 {
                return Err(ChemError::new("Malformed V3000 BOND record"));
            }
            let btype = toks[1].parse::<u8>().unwrap_or(1);
            let id_a = toks[2]
                .parse::<usize>()
                .map_err(|_| ChemError::new("Invalid V3000 bond atom 1"))?;
            let id_b = toks[3]
                .parse::<usize>()
                .map_err(|_| ChemError::new("Invalid V3000 bond atom 2"))?;
            let &a = id_to_idx
                .get(&id_a)
                .ok_or_else(|| ChemError::new("Unknown V3000 atom ID in bond"))?;
            let &b = id_to_idx
                .get(&id_b)
                .ok_or_else(|| ChemError::new("Unknown V3000 atom ID in bond"))?;

            let mut cfg: u8 = 0;
            let mut endpts: Vec<usize> = Vec::new();
            for kv in &toks[4..] {
                if let Some((k, v)) = kv.split_once('=') {
                    match k {
                        "CFG" => cfg = v.parse::<u8>().unwrap_or(0),
                        "ENDPTS" => {
                            let inner = v.trim_start_matches('(').trim_end_matches(')');
                            let nums: Vec<usize> = inner
                                .split_whitespace()
                                .filter_map(|s| s.parse::<usize>().ok())
                                .collect();
                            if nums.len() >= 2 {
                                for &aid in &nums[1..] {
                                    if let Some(&mapped) = id_to_idx.get(&aid) {
                                        endpts.push(mapped);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            let kind = match (btype, cfg) {
                (1, 1) => BondKind::WedgeUp,
                (1, 2) => BondKind::Wavy,
                (1, 3) => BondKind::HashDown,
                (1, _) => BondKind::Single,
                (2, _) => BondKind::Double,
                (3, _) => BondKind::Triple,
                (4, _) => {
                    has_arom_bond = true;
                    BondKind::Aromatic
                }
                (9, _) => BondKind::Dative,
                (10, _) => BondKind::Hydrogen,
                (8, _) => BondKind::Ionic,
                _ => BondKind::Single,
            };
            let bi = mol.add_bond(a, b, kind);
            if btype == 2 && cfg == 2 {
                mol.bonds[bi].double_geometry = DoubleBondGeometry::Either;
            }
            if !endpts.is_empty() {
                if mol.atoms[b].z == 0 {
                    mol.atoms[b].multi_center_atoms = endpts;
                } else if mol.atoms[a].z == 0 {
                    mol.atoms[a].multi_center_atoms = endpts;
                }
            }
        } else if section == "SGROUP" {
            let toks = tokenize_v30_line(trimmed);
            if toks.len() >= 3 && toks[1] == "SUP" {
                let mut sg_atoms = Vec::new();
                let mut sg_label = None;
                for kv in &toks[3..] {
                    if let Some((k, v)) = kv.split_once('=') {
                        if k == "ATOMS" {
                            let inner = v.trim_start_matches('(').trim_end_matches(')');
                            let nums: Vec<usize> = inner
                                .split_whitespace()
                                .filter_map(|s| s.parse::<usize>().ok())
                                .collect();
                            if nums.len() >= 2 {
                                for &aid in &nums[1..] {
                                    if let Some(&mapped) = id_to_idx.get(&aid) {
                                        sg_atoms.push(mapped);
                                    }
                                }
                            }
                        } else if k == "LABEL" {
                            let unq = v
                                .strip_prefix('"')
                                .and_then(|s| s.strip_suffix('"'))
                                .unwrap_or(v)
                                .replace("\"\"", "\"");
                            sg_label = Some(unq);
                        }
                    }
                }
                if sg_atoms.len() == 1 {
                    if let Some(lbl) = sg_label {
                        mol.atoms[sg_atoms[0]].superatom_label = Some(lbl);
                    }
                }
            }
        }
    }

    if any_3d {
        for (i, z) in raw_z.into_iter().enumerate() {
            mol.atoms[i].z_coord = Some(z);
        }
    }

    if has_arom_bond {
        kekulize(&mut mol)?;
    }

    Ok(mol)
}

/// Tokenize a V3000 line keeping `KEY=(a b c)` and `KEY="quoted text"` together as single tokens!
fn tokenize_v30_line(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut paren_depth = 0i32;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            cur.push(c);
            if in_quotes && i + 1 < chars.len() && chars[i + 1] == '"' {
                cur.push('"');
                i += 2;
                continue;
            }
            in_quotes = !in_quotes;
        } else if !in_quotes && c == '(' {
            paren_depth += 1;
            cur.push(c);
        } else if !in_quotes && c == ')' {
            paren_depth = (paren_depth - 1).max(0);
            cur.push(c);
        } else if !in_quotes && paren_depth == 0 && c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
        i += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

// ============================================================================
// Multi-Record SDF Reader & Writer (`read_sdf` & `write_sdf`)
// ============================================================================

/// Read a multi-record `.sdf` file into `Vec<SdfRecord>`, parsing each record's V2000/V3000 structure
/// and all `> <FieldName>` data properties.
pub fn read_sdf(sdf_text: &str) -> Result<Vec<SdfRecord>, ChemError> {
    let normalized = sdf_text.replace("\r\n", "\n").replace('\r', "\n");
    let mut records = Vec::new();

    for raw_block in normalized.split("$$$$") {
        let trimmed_check = raw_block.trim();
        if trimmed_check.is_empty() {
            continue;
        }
        // Strip leading newline left over after previous `$$$$\n` without trimming line 1's leading molecule name!
        let block = raw_block.strip_prefix('\n').unwrap_or(raw_block);

        // Split at `M  END` into MOL CTAB block + SDF data items block
        let (mol_part, props_part) = if let Some(m_end_pos) = block.find("M  END") {
            let after_m_end = m_end_pos + "M  END".len();
            (&block[..after_m_end], &block[after_m_end..])
        } else {
            (block, "")
        };

        let mut molecule = read_mol(mol_part)?;
        let properties = parse_sdf_properties(props_part);
        for (k, v) in &properties {
            molecule.properties.insert(k.clone(), v.clone());
        }
        records.push(SdfRecord {
            molecule,
            properties,
        });
    }

    Ok(records)
}

fn parse_sdf_properties(props_part: &str) -> BTreeMap<String, String> {
    let mut props = BTreeMap::new();
    let lines: Vec<&str> = props_part.lines().collect();
    let mut i = 0usize;

    while i < lines.len() {
        let l = lines[i].trim();
        if l.starts_with('>') {
            // Extract `<FieldName>` from `> <FieldName>` or `>  <FieldName> (registry)`
            if let (Some(open), Some(close)) = (l.find('<'), l.rfind('>')) {
                if close > open + 1 {
                    let field_name = l[open + 1..close].trim().to_string();
                    i += 1;
                    let mut val_lines = Vec::new();
                    while i < lines.len() {
                        if lines[i].trim().is_empty() {
                            break;
                        }
                        if lines[i].trim_start().starts_with('>') {
                            break;
                        }
                        val_lines.push(lines[i]);
                        i += 1;
                    }
                    props.insert(field_name, val_lines.join("\n"));
                    continue;
                }
            }
        }
        i += 1;
    }

    props
}

/// Write a slice of `SdfRecord`s into a standard multi-record `.sdf` string (automatically using V2000,
/// or V3000 when a record exceeds 999 atoms/bonds).
pub fn write_sdf(records: &[SdfRecord]) -> String {
    let mut out = String::new();
    for rec in records {
        match write_mol_v2000(&rec.molecule) {
            Ok(m2) => out.push_str(&m2),
            Err(_) => out.push_str(&write_mol_v3000(&rec.molecule)),
        }

        // Merge `rec.molecule.properties` and `rec.properties` (`rec.properties` wins on collision)
        let mut merged: BTreeMap<&str, &str> = BTreeMap::new();
        for (k, v) in &rec.molecule.properties {
            merged.insert(k.as_str(), v.as_str());
        }
        for (k, v) in &rec.properties {
            merged.insert(k.as_str(), v.as_str());
        }

        for (field, val) in merged {
            out.push_str(&format!(">  <{}>\n{}\n\n", field, val));
        }
        out.push_str("$$$$\n");
    }
    out
}
