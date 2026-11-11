//! Superatom Abbreviation Dictionary (`Boc`, `Fmoc`, `Cbz`, `Ac`, `Bz`, `Ts`, `Ms`, `Tf`,
//! `TBS`, `TBDPS`, `TMS`, `TIPS`, `THP`, `MOM`, `Bn`, `PMB`, `Ph`, `Mes`, `tBu`, `iPr`,
//! `nBu`, `Et`, `Me`, `CF3`, `NO2`, `CN`, `N3`, `COOH`, `COOMe`, `COOEt`, `SO3H`, `PO3H2`),
//! Superatom Expand/Collapse, and Condensed Chemical Formula Parser.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0:
//! `abbreviations.json`, `labels.ts`, `condensed.ts`, `expand.ts`).

use super::graph::{atomic_number_from_symbol, Atom, BondKind, ChemError, Molecule};
use super::layout::clean_2d;
use super::smiles_inchi::parse_smiles;
use std::collections::BTreeSet;

/// Canonical superatom templates: `(canonical_label, aliases, smiles_with_attachment_star_at_atom_0)`.
/// In every template SMILES, atom `0` is `*` (the external attachment point) bonded to atom `1` (the root atom of the group).
pub const SUPERATOM_TEMPLATES: &[(&str, &[&str], &str)] = &[
    ("Boc", &["tBoc", "BOC"], "*C(=O)OC(C)(C)C"),
    ("Fmoc", &["FMOC"], "*C(=O)OCC1C2=CC=CC=C2C3=CC=CC=C13"),
    ("Cbz", &["Z"], "*C(=O)OCC1=CC=CC=C1"),
    ("Ac", &["COMe", "MeCO"], "*C(=O)C"),
    ("Bz", &["COPh", "PhCO"], "*C(=O)C1=CC=CC=C1"),
    ("Ts", &["Tos"], "*S(=O)(=O)C1=CC=C(C)C=C1"),
    ("Ms", &["Mesyl", "SO2Me"], "*S(=O)(=O)C"),
    ("Tf", &["SO2CF3"], "*S(=O)(=O)C(F)(F)F"),
    ("TBS", &["TBDMS"], "*[Si](C)(C)C(C)(C)C"),
    ("TBDPS", &[], "*[Si](C1=CC=CC=C1)(C2=CC=CC=C2)C(C)(C)C"),
    ("TMS", &["SiMe3"], "*[Si](C)(C)C"),
    ("TIPS", &["Si(iPr)3"], "*[Si](C(C)C)(C(C)C)C(C)C"),
    ("THP", &[], "*C1CCCCO1"),
    ("MOM", &["CH2OMe"], "*COC"),
    ("Bn", &["Bzl", "CH2Ph"], "*CC1=CC=CC=C1"),
    ("PMB", &["MPM"], "*CC1=CC=C(OC)C=C1"),
    ("Ph", &["C6H5"], "*C1=CC=CC=C1"),
    ("Mes", &[], "*C1=C(C)C=C(C)C=C1C"),
    ("tBu", &["t-Bu", "Bu-t", "CMe3"], "*C(C)(C)C"),
    ("iPr", &["i-Pr", "Pr-i"], "*C(C)C"),
    ("nBu", &["n-Bu", "Bu"], "*CCCC"),
    ("Et", &["C2H5"], "*CC"),
    ("Me", &[], "*C"),
    ("CF3", &["F3C"], "*C(F)(F)F"),
    ("NO2", &["O2N"], "*[N+](=O)[O-]"),
    ("CN", &["NC"], "*C#N"),
    ("N3", &[], "*N=[N+]=[N-]"),
    ("COOH", &["CO2H", "HOOC", "HO2C"], "*C(=O)O"),
    ("COOMe", &["CO2Me", "MeOOC", "MeO2C"], "*C(=O)OC"),
    ("COOEt", &["CO2Et", "EtOOC", "EtO2C"], "*C(=O)OCC"),
    ("SO3H", &["HO3S"], "*S(=O)(=O)O"),
    ("PO3H2", &["H2O3P"], "*P(=O)(O)O"),
    // Additional standard organic groups used in condensed formulas
    ("CHO", &["OHC"], "*C=O"),
    ("Piv", &["COtBu"], "*C(=O)C(C)(C)C"),
    ("Cy", &["C6H11"], "*C1CCCCC1"),
    ("Pr", &["nPr", "n-Pr"], "*CCC"),
    ("OMe", &["MeO"], "*OC"),
    ("OEt", &["EtO"], "*OCC"),
    ("OAc", &["AcO"], "*OC(=O)C"),
    ("OTf", &["TfO"], "*OS(=O)(=O)C(F)(F)F"),
    ("OTs", &["TsO"], "*OS(=O)(=O)C1=CC=C(C)C=C1"),
    ("OMs", &["MsO"], "*OS(=O)(=O)C"),
    ("OTBS", &["TBSO"], "*O[Si](C)(C)C(C)(C)C"),
    ("OTMS", &["TMSO"], "*O[Si](C)(C)C"),
    ("NHBoc", &["BocHN"], "*NC(=O)OC(C)(C)C"),
    ("NHFmoc", &["FmocHN"], "*NC(=O)OCC1C2=CC=CC=C2C3=CC=CC=C13"),
    ("NHCbz", &["CbzHN"], "*NC(=O)OCC1=CC=CC=C1"),
    ("NHAc", &["AcHN"], "*NC(=O)C"),
    ("CONH2", &["H2NCO"], "*C(=O)N"),
    ("OH", &["HO"], "*O"),
    ("SH", &["HS"], "*S"),
    ("NH2", &["H2N"], "*N"),
];

/// Look up the template SMILES (with `*` at atom `0` as the attachment point) for a superatom abbreviation.
pub fn lookup_abbreviation_smiles(label: &str) -> Option<&'static str> {
    let trimmed = label.trim().trim_start_matches('-');
    for &(canon, aliases, smi) in SUPERATOM_TEMPLATES {
        if canon == trimmed || aliases.contains(&trimmed) {
            return Some(smi);
        }
    }
    None
}

/// Return the canonical superatom name for `label` if recognized.
pub fn canonical_abbreviation_name(label: &str) -> Option<&'static str> {
    let trimmed = label.trim().trim_start_matches('-');
    for &(canon, aliases, _) in SUPERATOM_TEMPLATES {
        if canon == trimmed || aliases.contains(&trimmed) {
            return Some(canon);
        }
    }
    None
}

/// List all primary canonical superatom names supported out of the box.
pub fn supported_abbreviations() -> Vec<&'static str> {
    SUPERATOM_TEMPLATES.iter().map(|&(c, _, _)| c).collect()
}

/// Expand all superatom / abbreviated nodes (`atom.superatom_label.is_some()`) in `mol` into their
/// full all-atom chemical structures and clean up coordinates.
/// Returns the number of superatoms expanded.
pub fn expand_abbreviations(mol: &mut Molecule) -> Result<usize, ChemError> {
    let mut expanded_count = 0usize;

    loop {
        // Find the next superatom node to expand
        let next_idx = mol.atoms.iter().position(|a| a.superatom_label.is_some());
        let Some(sa_idx) = next_idx else {
            break;
        };

        let label = mol.atoms[sa_idx].superatom_label.clone().unwrap();
        let incident: Vec<(usize, usize)> = mol.neighbors(sa_idx);

        // Case 1: Superatom was previously collapsed with `inner_fragment` stored and 1 external bond
        if let Some(inner) = mol.atoms[sa_idx].inner_fragment.take() {
            if incident.len() <= 1 {
                // Find atom in `inner` with `attachment_point == Some(1)` (or atom 0)
                let attach_in_inner = inner
                    .atoms
                    .iter()
                    .position(|a| a.attachment_point == Some(1))
                    .unwrap_or(0);
                splice_group_replacing_atom(mol, sa_idx, &inner, attach_in_inner);
                expanded_count += 1;
                continue;
            }
        }

        // Case 2: Standard dictionary superatom (`*...` template)
        let template_smi = if let Some(smi) = lookup_abbreviation_smiles(&label) {
            smi.to_string()
        } else {
            // Try parsing `label` as a monovalent condensed group (e.g. `CH2CH2OH`, `CH2CF3`)!
            let sub_smi = condensed_substituent_smiles(&label).ok_or_else(|| {
                ChemError::new(format!("Unknown superatom abbreviation: '{}'", label))
            })?;
            sub_smi
        };

        if incident.len() > 1 {
            return Err(ChemError::new(format!(
                "Monovalent superatom '{}' at atom {} has {} bonds (expected <= 1)",
                label,
                sa_idx,
                incident.len()
            )));
        }

        let tmpl = parse_smiles(&template_smi)?;
        // In `*...`, atom 0 is `*` and atom 1 is the root atom of the group bonded to `*`!
        if tmpl.atoms.len() < 2 || tmpl.atoms[0].z != 0 {
            return Err(ChemError::new(format!(
                "Malformed superatom template for '{}'",
                label
            )));
        }
        let root_in_tmpl = tmpl
            .neighbors(0)
            .first()
            .map(|&(nb, _)| nb)
            .ok_or_else(|| ChemError::new("Disconnected superatom attachment star"))?;

        // Extract `tmpl` without atom `0` (`*`), adjusting `root_in_tmpl` index
        let keep_indices: Vec<usize> = (1..tmpl.atoms.len()).collect();
        let mut group_mol = tmpl.extract_subgraph(&keep_indices);
        let new_root = root_in_tmpl - 1;
        // Clear any explicit_hydrogens override on `new_root` if `*` affected it? `parse_smiles` only sets
        // `explicit_hydrogens` on bracketed atoms (like `[Si]`, `[N+]`), where `*` already stood in for the 1 external bond!
        group_mol.atoms[new_root].attachment_point = Some(1);

        splice_group_replacing_atom(mol, sa_idx, &group_mol, new_root);
        expanded_count += 1;
    }

    if expanded_count > 0 {
        clean_2d(mol);
    }

    Ok(expanded_count)
}

fn splice_group_replacing_atom(
    mol: &mut Molecule,
    sa_idx: usize,
    group_mol: &Molecule,
    attach_in_group: usize,
) {
    let anchor_x = mol.atoms[sa_idx].x;
    let anchor_y = mol.atoms[sa_idx].y;

    // Replace `mol.atoms[sa_idx]` in-place with `group_mol.atoms[attach_in_group]` so external bond indices & stereocenters pointing to `sa_idx` remain valid!
    let mut root_atom = group_mol.atoms[attach_in_group].clone();
    root_atom.superatom_label = None;
    root_atom.inner_fragment = None;
    root_atom.attachment_point = None;
    root_atom.x = anchor_x;
    root_atom.y = anchor_y;
    mol.atoms[sa_idx] = root_atom;

    // Append all other atoms of `group_mol` to `mol`
    let mut group_to_mol = vec![0usize; group_mol.atoms.len()];
    group_to_mol[attach_in_group] = sa_idx;
    for (gi, ga) in group_mol.atoms.iter().enumerate() {
        if gi == attach_in_group {
            continue;
        }
        let mut na = ga.clone();
        na.attachment_point = None;
        group_to_mol[gi] = mol.add_atom(na);
    }
    // Remap parities on newly added group atoms
    for gi in 0..group_mol.atoms.len() {
        let mi = group_to_mol[gi];
        if let Some(mut p) = group_mol.atoms[gi].parity {
            for nb in &mut p.neighbors {
                if *nb != usize::MAX {
                    *nb = group_to_mol[*nb];
                }
            }
            mol.atoms[mi].parity = Some(p);
        }
    }
    // Add all internal bonds of `group_mol`
    for gb in &group_mol.bonds {
        let ma = group_to_mol[gb.a];
        let mb = group_to_mol[gb.b];
        let bi = mol.add_bond(ma, mb, gb.kind);
        mol.bonds[bi].double_geometry = match gb.double_geometry {
            super::graph::DoubleBondGeometry::Cis { ref_a, ref_b } => {
                super::graph::DoubleBondGeometry::Cis {
                    ref_a: group_to_mol[ref_a],
                    ref_b: group_to_mol[ref_b],
                }
            }
            super::graph::DoubleBondGeometry::Trans { ref_a, ref_b } => {
                super::graph::DoubleBondGeometry::Trans {
                    ref_a: group_to_mol[ref_a],
                    ref_b: group_to_mol[ref_b],
                }
            }
            other => other,
        };
    }
}

/// Collapse a monovalent substituent (`group_atoms`) in `mol` into a single superatom node labeled `label`,
/// storing the original `group_atoms` subgraph inside `Atom::inner_fragment` for lossless re-expansion!
/// Returns the index of the newly created superatom in `mol.atoms`.
pub fn collapse_abbreviation(
    mol: &mut Molecule,
    group_atoms: &[usize],
    label: &str,
) -> Result<usize, ChemError> {
    if group_atoms.is_empty() {
        return Err(ChemError::new("Cannot collapse an empty atom set"));
    }
    let group_set: BTreeSet<usize> = group_atoms.iter().copied().collect();
    if group_set.iter().any(|&u| u >= mol.atoms.len()) {
        return Err(ChemError::new("Group atom index out of bounds"));
    }

    // Find crossing bonds between `group_set` and the rest of `mol`
    let mut crossing: Vec<(usize, usize, BondKind)> = Vec::new(); // (inside_atom, outside_atom, kind)
    for b in &mol.bonds {
        let in_a = group_set.contains(&b.a);
        let in_b = group_set.contains(&b.b);
        if in_a != in_b {
            let (inside, outside) = if in_a { (b.a, b.b) } else { (b.b, b.a) };
            // Preserve stereo direction if bond started at `outside`
            crossing.push((
                inside,
                outside,
                if b.a == outside {
                    b.kind
                } else {
                    BondKind::Single
                },
            ));
        }
    }
    if crossing.len() > 1 {
        return Err(ChemError::new(format!(
            "Cannot collapse '{}' with {} external bonds (expected <= 1)",
            label,
            crossing.len()
        )));
    }

    let ordered_group: Vec<usize> = group_set.iter().copied().collect();
    let mut inner = mol.extract_subgraph(&ordered_group);
    let anchor_old = crossing
        .first()
        .map(|&(ins, _, _)| ins)
        .unwrap_or(ordered_group[0]);
    if let Some(pos) = ordered_group.iter().position(|&u| u == anchor_old) {
        inner.atoms[pos].attachment_point = Some(1);
    }

    let ax = mol.atoms[anchor_old].x;
    let ay = mol.atoms[anchor_old].y;
    let az = mol.atoms[anchor_old].z_coord;

    // Build reduced molecule keeping all atoms NOT in `group_set`, plus 1 new superatom node!
    let kept_old: Vec<usize> = (0..mol.atoms.len())
        .filter(|u| !group_set.contains(u))
        .collect();
    let mut new_mol = mol.extract_subgraph(&kept_old);
    new_mol.name = mol.name.clone();
    new_mol.comment = mol.comment.clone();
    new_mol.properties = mol.properties.clone();

    let mut sa = Atom::new(label);
    // Represent superatom element as attachment atom's element (or Z=0 dummy) with `superatom_label` set
    sa.z = 0;
    sa.symbol = label.to_string();
    sa.superatom_label = Some(label.to_string());
    sa.inner_fragment = Some(Box::new(inner));
    sa.x = ax;
    sa.y = ay;
    sa.z_coord = az;
    let sa_new_idx = new_mol.add_atom(sa);

    if let Some(&(_inside_old, outside_old, bkind)) = crossing.first() {
        let outside_new = kept_old.iter().position(|&u| u == outside_old).unwrap();
        new_mol.add_bond(outside_new, sa_new_idx, bkind);
        // Update `outside_new`'s parity if it pointed to `anchor_old`!
        if let Some(mut p) = mol.atoms[outside_old].parity {
            let mut ok = true;
            for slot in &mut p.neighbors {
                if *slot == usize::MAX {
                    continue;
                } else if *slot == anchor_old {
                    *slot = sa_new_idx;
                } else if let Some(pos) = kept_old.iter().position(|&u| u == *slot) {
                    *slot = pos;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                new_mol.atoms[outside_new].parity = Some(p);
            }
        }
    }

    *mol = new_mol;
    Ok(sa_new_idx)
}

// ============================================================================
// Condensed Formula Parser (`CH3CH2OH`, `(CH3)3COH`, `PhB(OH)2`, `CH3COOEt`, etc.)
// ============================================================================

#[derive(Clone, Debug)]
enum CondToken {
    Atom { el: String, z: u8, h: u8 },
    Group { smiles: String },
    Bracket { inner: Vec<CondToken> },
    Bond { order: u8 },
    DoubleBranchAtom { el: String, z: u8, h: u8 },
}

fn condensed_allowed_valences(z: u8) -> &'static [u8] {
    match z {
        5 => &[3],
        6 => &[4],
        7 => &[3],
        8 => &[2],
        9 | 17 | 35 | 53 | 3 | 11 | 19 => &[1],
        12 | 30 | 34 => &[2],
        13 => &[3],
        14 | 32 | 50 => &[4],
        15 => &[5, 3],
        16 => &[6, 4, 2],
        _ => &[],
    }
}

/// Parse a condensed chemical formula (such as `"CH3CH2OH"`, `"(CH3)3COH"`, `"PhB(OH)2"`,
/// `"CH3COOEt"`, `"CF3COOH"`, `"Me3SiCl"`, `"CH3CN"`, `"BocNHCH2COOH"`) into a 2D `Molecule`.
pub fn parse_condensed_formula(formula: &str) -> Result<Molecule, ChemError> {
    let smi = condensed_formula_to_smiles(formula)?;
    let mut mol = parse_smiles(&smi)?;
    clean_2d(&mut mol);
    Ok(mol)
}

/// Convert a condensed chemical formula into a valid SMILES string.
pub fn condensed_formula_to_smiles(formula: &str) -> Result<String, ChemError> {
    let mut s = formula.trim().replace('≡', "#");
    if s.is_empty() {
        return Err(ChemError::new("Empty condensed formula"));
    }
    // Check if the entire string is a single superatom + H/halide (e.g. `BocOH`, `TsCl`, `BzCl`, `AcOH`)
    s = replace_formula_shorthands(&s);

    let tokens = tokenize_condensed(&s, 0).ok_or_else(|| {
        ChemError::new(format!(
            "Could not tokenize condensed formula '{}'",
            formula
        ))
    })?;
    if tokens.is_empty() {
        return Err(ChemError::new("Empty condensed formula tokens"));
    }

    let state = assemble_condensed(&tokens, 0).ok_or_else(|| {
        ChemError::new(format!(
            "Could not find unambiguous valence assignment for condensed formula '{}'",
            formula
        ))
    })?;

    state_to_smiles(&state).ok_or_else(|| {
        ChemError::new(format!(
            "Failed to serialize condensed formula '{}' to SMILES",
            formula
        ))
    })
}

fn condensed_substituent_smiles(label: &str) -> Option<String> {
    let s = replace_formula_shorthands(label.trim());
    let mut tokens = vec![CondToken::Group {
        smiles: "[*]".to_string(),
    }];
    tokens.extend(tokenize_condensed(&s, 0)?);
    let state = assemble_condensed(&tokens, 0)?;
    let smi = state_to_smiles(&state)?;
    // Ensure `[*]` is present so caller can splice on atom 0 (`*`)!
    // Parse and re-root at `*` if needed:
    if smi.starts_with("[*]") || smi.starts_with('*') {
        Some(smi.replacen("[*]", "*", 1))
    } else if smi.contains("([*])") {
        // Move `*` to front when attached to the first bracketed atom `[X...]([*])rest` -> `*[X...]rest`
        let idx = smi.find("([*])")?;
        let mut rebuilt = String::from("*");
        rebuilt.push_str(&smi[..idx]);
        rebuilt.push_str(&smi[idx + 5..]);
        Some(rebuilt)
    } else {
        None
    }
}

fn replace_formula_shorthands(s: &str) -> String {
    let mut w = s.to_string();
    for (pat, rep) in [("C6H5", "Ph"), ("C2H5", "Et"), ("C6H11", "Cy")] {
        w = replace_non_followed_by_digit(&w, pat, rep);
    }
    w
}

fn replace_non_followed_by_digit(s: &str, pat: &str, rep: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(pat) {
            let after = i + pat.len();
            let next_is_digit = s[after..]
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            if !next_is_digit {
                out.push_str(rep);
                i = after;
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Strip leading `*` from a superatom SMILES so it can be embedded as a branch in a condensed formula:
/// e.g. `*C(=O)OC(C)(C)C` -> `C(=O)OC(C)(C)C`, `*C1=CC=CC=C1` -> `C1=CC=CC=C1`.
fn group_branch_smiles(label: &str) -> Option<String> {
    let smi = lookup_abbreviation_smiles(label)?;
    Some(smi.strip_prefix('*').unwrap_or(smi).to_string())
}

fn tokenize_condensed(text: &str, depth: usize) -> Option<Vec<CondToken>> {
    if depth > 4 {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out: Vec<CondToken> = Vec::new();
    let mut i = 0usize;
    let mut lead_h: u8 = 0;

    let read_count = |pos: &mut usize| -> usize {
        let start = *pos;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() && *pos - start < 2 {
            *pos += 1;
        }
        if *pos > start {
            text[start..*pos].parse::<usize>().unwrap_or(1)
        } else {
            1
        }
    };

    // Sort multi-letter group names longest-first so `COOMe` matches before `CO` + `OMe`, `TBDPS` before `TBS`, `Ph` before `P`!
    let mut group_names: Vec<&str> = Vec::new();
    for &(canon, aliases, _) in SUPERATOM_TEMPLATES {
        // Exclude bare element symbols like `Ac`? Wait: in organic condensed formulas `Ac` IS acetyl (`AcOH`, `Ac2O`),
        // `Pr` IS propyl, `Me`, `Et`, `Ph`, `Bn`, `Boc`, `Fmoc`, `Cbz`, `Ts`, `Ms`, `Tf`, `TBS`, `TBDPS`, `TMS`, `TIPS`,
        // `THP`, `MOM`, `PMB`, `Mes`, `tBu`, `iPr`, `nBu`, `CF3`, `NO2`, `CN`, `N3`, `COOH`, `COOMe`, `COOEt`, `SO3H`, `PO3H2`!
        // Note: `OH`, `SH`, `NH2`, `CHO`, `CN` — let `CN` be handled by chain triple bond (so `HCN` & `CH3CN` both work!),
        // and let `OH`, `SH`, `NH2` be handled as element + H so `HOCH2CH2OH` and `NH2CH2COOH` work either way!
        if matches!(canon, "OH" | "SH" | "NH2" | "CN") {
            continue;
        }
        group_names.push(canon);
        for &al in aliases {
            if !matches!(al, "HO" | "HS" | "H2N" | "NC" | "Bu") && !al.contains('-') {
                group_names.push(al);
            }
        }
    }
    group_names.sort_by_key(|b| std::cmp::Reverse(b.len()));

    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '(' {
            let mut d = 0i32;
            let mut j = i;
            while j < bytes.len() {
                if bytes[j] == b'(' {
                    d += 1;
                } else if bytes[j] == b')' {
                    d -= 1;
                    if d == 0 {
                        break;
                    }
                }
                j += 1;
            }
            if j >= bytes.len() {
                return None;
            }
            let body = &text[i + 1..j];
            i = j + 1;
            let count = read_count(&mut i);
            if let Some(after_eq) = body.strip_prefix('=') {
                let inner = tokenize_condensed(after_eq, depth + 1)?;
                if inner.len() != 1 {
                    return None;
                }
                if let CondToken::Atom { el, z, h } = &inner[0] {
                    for _ in 0..count {
                        out.push(CondToken::DoubleBranchAtom {
                            el: el.clone(),
                            z: *z,
                            h: *h,
                        });
                    }
                    continue;
                }
                return None;
            }
            let inner = tokenize_condensed(body, depth + 1)?;
            if inner.is_empty() {
                return None;
            }
            for _ in 0..count {
                out.push(CondToken::Bracket {
                    inner: inner.clone(),
                });
            }
            continue;
        }
        if c == '-' || c == '=' || c == '#' {
            let order = match c {
                '-' => 1,
                '=' => 2,
                _ => 3,
            };
            out.push(CondToken::Bond { order });
            i += 1;
            continue;
        }

        // Try matching a known superatom group (provided the next char is not a lowercase letter that belongs to another symbol,
        // and for groups ending in digits like `CF3`/`NO2`/`N3`/`PO3H2`, not followed by another digit unless parenthesized):
        let mut matched_group = None;
        for &gname in &group_names {
            if text[i..].starts_with(gname) {
                let after = i + gname.len();
                let next_ch = text[after..].chars().next();
                let ends_with_digit = gname.as_bytes().last().unwrap().is_ascii_digit();
                let boundary_ok = match next_ch {
                    Some(nc) if nc.is_ascii_lowercase() => false,
                    Some(nc) if ends_with_digit && nc.is_ascii_digit() => false,
                    _ => true,
                };
                if boundary_ok && lead_h == 0 {
                    matched_group = Some(gname);
                    break;
                }
            }
        }
        if let Some(gname) = matched_group {
            i += gname.len();
            let count = if gname.as_bytes().last().unwrap().is_ascii_digit() {
                1
            } else {
                read_count(&mut i)
            };
            let smi = group_branch_smiles(gname)?;
            for _ in 0..count {
                out.push(CondToken::Group {
                    smiles: smi.clone(),
                });
            }
            continue;
        }

        // Otherwise read a chemical element symbol (`[A-Z][a-z]?`) + optional integer count
        if !c.is_ascii_uppercase() {
            return None;
        }
        let el_len = if i + 1 < bytes.len() && bytes[i + 1].is_ascii_lowercase() {
            let two = &text[i..i + 2];
            if atomic_number_from_symbol(two).is_some() {
                2
            } else {
                1
            }
        } else {
            1
        };
        let el_sym = &text[i..i + el_len];
        let z = atomic_number_from_symbol(el_sym)?;
        i += el_len;
        let n = read_count(&mut i);

        if z == 1 {
            // Hydrogen! Attach to previous Atom token if immediately preceding, or accumulate `lead_h` at formula start (`HOCH2CH2OH`, `HC#CH`)!
            if let Some(CondToken::Atom { h, .. }) = out.last_mut() {
                *h = h.saturating_add(n as u8);
            } else if out.iter().all(|t| matches!(t, CondToken::Bond { .. })) {
                lead_h = lead_h.saturating_add(n as u8);
            } else {
                return None;
            }
        } else {
            if condensed_allowed_valences(z).is_empty() {
                return None;
            }
            out.push(CondToken::Atom {
                el: el_sym.to_string(),
                z,
                h: lead_h,
            });
            lead_h = 0;
            for _ in 1..n {
                out.push(CondToken::Atom {
                    el: el_sym.to_string(),
                    z,
                    h: 0,
                });
            }
        }
    }

    if lead_h > 0 {
        // e.g. `BocOH` or `MeOH` wasn't caught or `PhH` (benzene!): if `out` has 1 Group and `lead_h == 1`!
        if out.len() == 1 {
            if let CondToken::Group { smiles } = &out[0] {
                return Some(vec![CondToken::Group {
                    smiles: smiles.clone(),
                }]);
            }
        }
        return None;
    }
    // Also handle `Group + H` like `PhH` or `BocH`? Above `z == 1` returned `None` when `out.last()` was `Group`.
    Some(out)
}

#[derive(Clone, Debug)]
struct CondNode {
    el: Option<String>,
    h: u8,
    smiles: Option<String>,
    val: u8,
    deg: u8,
}

#[derive(Clone, Debug)]
struct CondBond {
    a: usize,
    b: usize,
    order: u8,
    fixed: bool,
}

#[derive(Clone, Debug)]
struct CondState {
    nodes: Vec<CondNode>,
    bonds: Vec<CondBond>,
    cur: Option<usize>,
    pending: Vec<usize>,
    next_order: Option<u8>,
}

impl CondState {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            bonds: Vec::new(),
            cur: None,
            pending: Vec::new(),
            next_order: None,
        }
    }

    fn add_bond(&mut self, a: usize, b: usize, order: u8, fixed: bool) -> bool {
        self.bonds.push(CondBond { a, b, order, fixed });
        self.nodes[a].deg += order;
        self.nodes[b].deg += order;
        self.nodes[a].deg <= self.nodes[a].val && self.nodes[b].deg <= self.nodes[b].val
    }

    fn attach_chain(&mut self, idx: usize) -> bool {
        let mut ok = true;
        if let Some(c) = self.cur {
            let ord = self.next_order.unwrap_or(1);
            let fixed = self.next_order.is_some();
            ok = self.add_bond(c, idx, ord, fixed);
        }
        self.next_order = None;
        let pend = std::mem::take(&mut self.pending);
        for p in pend {
            ok = ok && self.add_bond(idx, p, 1, true);
        }
        self.cur = Some(idx);
        ok
    }

    fn attach_substituent(&mut self, idx: usize) -> bool {
        if let (Some(c), None) = (self.cur, self.next_order) {
            self.add_bond(c, idx, 1, true)
        } else {
            self.pending.push(idx);
            true
        }
    }

    fn spare(&self) -> u8 {
        self.nodes.iter().map(|n| n.val.saturating_sub(n.deg)).sum()
    }
}

fn assemble_condensed(tokens: &[CondToken], open: u8) -> Option<CondState> {
    let mut tries = 0usize;
    let init = CondState::new();
    let mut results = Vec::new();
    search_readings(tokens, 0, init, &mut tries, &mut results);

    for mut st in results {
        if open == 0 && st.cur.is_none() && st.pending.len() == 2 {
            let p0 = st.pending[0];
            let p1 = st.pending[1];
            st.pending.clear();
            if !st.add_bond(p0, p1, 1, true) {
                continue;
            }
        }
        if !st.pending.is_empty() || st.next_order.is_some() || st.nodes.is_empty() {
            continue;
        }
        if open > 0 {
            if st.spare() == open {
                return Some(st);
            }
            continue;
        }
        // Promote unsaturated adjacent chain bonds (`CH3CN`, `HC#CH`, `CH2=CH2`)
        for bi in 0..st.bonds.len() {
            if st.bonds[bi].fixed {
                continue;
            }
            let a = st.bonds[bi].a;
            let b = st.bonds[bi].b;
            let rem_a = st.nodes[a].val.saturating_sub(st.nodes[a].deg);
            let rem_b = st.nodes[b].val.saturating_sub(st.nodes[b].deg);
            let max_inc = 3u8.saturating_sub(st.bonds[bi].order);
            let add = rem_a.min(rem_b).min(max_inc);
            if add > 0 {
                st.bonds[bi].order += add;
                st.nodes[a].deg += add;
                st.nodes[b].deg += add;
            }
        }
        if st.nodes.iter().all(|n| n.deg == n.val) {
            return Some(st);
        }
    }
    None
}

fn search_readings(
    tokens: &[CondToken],
    k: usize,
    st: CondState,
    tries: &mut usize,
    out: &mut Vec<CondState>,
) {
    *tries += 1;
    if *tries > 15_000 || !out.is_empty() || st.nodes.len() > 120 {
        return;
    }
    if k == tokens.len() {
        // Check right here if valid or record candidate? Wait: only stop early if candidate passes final check,
        // so collect up to 64 candidates!
        if out.len() < 64 {
            out.push(st);
        }
        return;
    }
    // Wait: remove `!out.is_empty()` early exit above so all up-to-64 candidates are explored!
    match &tokens[k] {
        CondToken::Bond { order } => {
            if st.cur.is_none() || st.next_order.is_some() {
                return;
            }
            let mut nst = st;
            nst.next_order = Some(*order);
            search_readings_multi(tokens, k + 1, nst, tries, out);
        }
        CondToken::DoubleBranchAtom { el, z, h } => {
            let Some(cur) = st.cur else {
                return;
            };
            for &val in condensed_allowed_valences(*z) {
                if val.checked_sub(*h) != Some(2) {
                    continue;
                }
                let mut nst = st.clone();
                let idx = nst.nodes.len();
                nst.nodes.push(CondNode {
                    el: Some(el.clone()),
                    h: *h,
                    smiles: None,
                    val: 2,
                    deg: 0,
                });
                if nst.add_bond(cur, idx, 2, true) {
                    search_readings_multi(tokens, k + 1, nst, tries, out);
                }
            }
        }
        CondToken::Group { smiles } => {
            let mut nst = st;
            let idx = nst.nodes.len();
            nst.nodes.push(CondNode {
                el: None,
                h: 0,
                smiles: Some(smiles.clone()),
                val: 1,
                deg: 0,
            });
            if nst.attach_substituent(idx) {
                search_readings_multi(tokens, k + 1, nst, tries, out);
            }
        }
        CondToken::Bracket { inner } => {
            let Some((bst, ends)) = assemble_bracket(inner) else {
                return;
            };
            let mut nst = st;
            let base = nst.nodes.len();
            nst.nodes.extend(bst.nodes);
            for b in bst.bonds {
                nst.bonds.push(CondBond {
                    a: b.a + base,
                    b: b.b + base,
                    order: b.order,
                    fixed: true,
                });
            }
            if ends.len() == 1 {
                if nst.attach_substituent(ends[0] + base) {
                    search_readings_multi(tokens, k + 1, nst, tries, out);
                }
            } else if ends.len() == 2 && nst.attach_chain(ends[0] + base) {
                nst.cur = Some(ends[1] + base);
                search_readings_multi(tokens, k + 1, nst, tries, out);
            }
        }
        CondToken::Atom { el, z, h } => {
            for &val in condensed_allowed_valences(*z) {
                if val < *h {
                    continue;
                }
                let free = val - *h;
                // Try double-bonded =O / =S branch first (e.g. ketone `CH3COCH3`, ester `CH3COOEt`, sulfone `MeSO2Cl`)
                if free == 2
                    && st.cur.is_some()
                    && st.next_order.is_none()
                    && matches!(*z, 8 | 16 | 34)
                {
                    let mut bst = st.clone();
                    let cur = bst.cur.unwrap();
                    let idx = bst.nodes.len();
                    bst.nodes.push(CondNode {
                        el: Some(el.clone()),
                        h: *h,
                        smiles: None,
                        val: free,
                        deg: 0,
                    });
                    if bst.add_bond(cur, idx, 2, true) {
                        search_readings_multi(tokens, k + 1, bst, tries, out);
                    }
                }
                let mut cst = st.clone();
                let idx = cst.nodes.len();
                cst.nodes.push(CondNode {
                    el: Some(el.clone()),
                    h: *h,
                    smiles: None,
                    val: free,
                    deg: 0,
                });
                let ok = if free == 1 && cst.cur.is_some() && cst.next_order.is_none() {
                    cst.attach_substituent(idx)
                } else {
                    cst.attach_chain(idx)
                };
                if ok {
                    search_readings_multi(tokens, k + 1, cst, tries, out);
                }
            }
        }
    }
}

fn search_readings_multi(
    tokens: &[CondToken],
    k: usize,
    st: CondState,
    tries: &mut usize,
    out: &mut Vec<CondState>,
) {
    *tries += 1;
    if *tries > 15_000 || out.len() >= 128 || st.nodes.len() > 120 {
        return;
    }
    if k == tokens.len() {
        out.push(st);
        return;
    }
    // Delegate back without the `!out.is_empty()` guard
    match &tokens[k] {
        CondToken::Bond { order } => {
            if st.cur.is_none() || st.next_order.is_some() {
                return;
            }
            let mut nst = st;
            nst.next_order = Some(*order);
            search_readings_multi(tokens, k + 1, nst, tries, out);
        }
        CondToken::DoubleBranchAtom { el, z, h } => {
            let Some(cur) = st.cur else {
                return;
            };
            for &val in condensed_allowed_valences(*z) {
                if val.checked_sub(*h) != Some(2) {
                    continue;
                }
                let mut nst = st.clone();
                let idx = nst.nodes.len();
                nst.nodes.push(CondNode {
                    el: Some(el.clone()),
                    h: *h,
                    smiles: None,
                    val: 2,
                    deg: 0,
                });
                if nst.add_bond(cur, idx, 2, true) {
                    search_readings_multi(tokens, k + 1, nst, tries, out);
                }
            }
        }
        CondToken::Group { smiles } => {
            let mut nst = st;
            let idx = nst.nodes.len();
            nst.nodes.push(CondNode {
                el: None,
                h: 0,
                smiles: Some(smiles.clone()),
                val: 1,
                deg: 0,
            });
            if nst.attach_substituent(idx) {
                search_readings_multi(tokens, k + 1, nst, tries, out);
            }
        }
        CondToken::Bracket { inner } => {
            let Some((bst, ends)) = assemble_bracket(inner) else {
                return;
            };
            let mut nst = st;
            let base = nst.nodes.len();
            nst.nodes.extend(bst.nodes);
            for b in bst.bonds {
                nst.bonds.push(CondBond {
                    a: b.a + base,
                    b: b.b + base,
                    order: b.order,
                    fixed: true,
                });
            }
            if ends.len() == 1 {
                if nst.attach_substituent(ends[0] + base) {
                    search_readings_multi(tokens, k + 1, nst, tries, out);
                }
            } else if ends.len() == 2 && nst.attach_chain(ends[0] + base) {
                nst.cur = Some(ends[1] + base);
                search_readings_multi(tokens, k + 1, nst, tries, out);
            }
        }
        CondToken::Atom { el, z, h } => {
            for &val in condensed_allowed_valences(*z) {
                if val < *h {
                    continue;
                }
                let free = val - *h;
                if free == 2
                    && st.cur.is_some()
                    && st.next_order.is_none()
                    && matches!(*z, 8 | 16 | 34)
                {
                    let mut bst = st.clone();
                    let cur = bst.cur.unwrap();
                    let idx = bst.nodes.len();
                    bst.nodes.push(CondNode {
                        el: Some(el.clone()),
                        h: *h,
                        smiles: None,
                        val: free,
                        deg: 0,
                    });
                    if bst.add_bond(cur, idx, 2, true) {
                        search_readings_multi(tokens, k + 1, bst, tries, out);
                    }
                }
                let mut cst = st.clone();
                let idx = cst.nodes.len();
                cst.nodes.push(CondNode {
                    el: Some(el.clone()),
                    h: *h,
                    smiles: None,
                    val: free,
                    deg: 0,
                });
                let ok = if free == 1 && cst.cur.is_some() && cst.next_order.is_none() {
                    cst.attach_substituent(idx)
                } else {
                    cst.attach_chain(idx)
                };
                if ok {
                    search_readings_multi(tokens, k + 1, cst, tries, out);
                }
            }
        }
    }
}

fn assemble_bracket(tokens: &[CondToken]) -> Option<(CondState, Vec<usize>)> {
    if tokens.len() == 1 {
        if let CondToken::Group { smiles } = &tokens[0] {
            let mut st = CondState::new();
            st.nodes.push(CondNode {
                el: None,
                h: 0,
                smiles: Some(smiles.clone()),
                val: 1,
                deg: 0,
            });
            return Some((st, vec![0]));
        }
    }
    for open in [1u8, 2u8] {
        if let Some(st) = assemble_condensed(tokens, open) {
            let mut ends = Vec::new();
            for (i, n) in st.nodes.iter().enumerate() {
                for _ in 0..(n.val - n.deg) {
                    ends.push(i);
                }
            }
            return Some((st, ends));
        }
    }
    None
}

fn state_to_smiles(st: &CondState) -> Option<String> {
    if st.nodes.iter().all(|n| n.smiles.is_some()) {
        if st.nodes.len() == 1 {
            return st.nodes[0].smiles.clone();
        }
        if st.nodes.len() == 2 {
            // Splice group 0 as a branch right after the first atom of group 1!
            let s0 = st.nodes[0].smiles.as_ref()?;
            let s1 = st.nodes[1].smiles.as_ref()?;
            let head_len = first_smiles_atom_len(s1)?;
            return Some(format!("{}({}){}", &s1[..head_len], s0, &s1[head_len..]));
        }
        return None;
    }

    let root = st.nodes.iter().position(|n| n.el.is_some())?;
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); st.nodes.len()];
    for b in &st.bonds {
        adj[b.a].push((b.b, b.order));
        adj[b.b].push((b.a, b.order));
    }

    let mut seen = vec![false; st.nodes.len()];
    fn dfs_write(u: usize, st: &CondState, adj: &[Vec<(usize, u8)>], seen: &mut [bool]) -> String {
        seen[u] = true;
        let node = &st.nodes[u];
        if let Some(smi) = &node.smiles {
            return smi.clone();
        }
        let mut kids = Vec::new();
        for &(v, ord) in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                kids.push((v, ord));
            }
        }
        let parts: Vec<String> = kids
            .into_iter()
            .map(|(v, ord)| {
                let prefix = match ord {
                    2 => "=",
                    3 => "#",
                    _ => "",
                };
                format!("{}{}", prefix, dfs_write(v, st, adj, seen))
            })
            .collect();

        let el = node.el.as_deref().unwrap_or("C");
        let head = if node.h == 0 {
            format!("[{}]", el)
        } else if node.h == 1 {
            format!("[{}H]", el)
        } else {
            format!("[{}H{}]", el, node.h)
        };
        let mut out = head;
        for (idx, p) in parts.iter().enumerate() {
            if idx + 1 < parts.len() {
                out.push('(');
                out.push_str(p);
                out.push(')');
            } else {
                out.push_str(p);
            }
        }
        out
    }

    let smi = dfs_write(root, st, &adj, &mut seen);
    if seen.iter().all(|&x| x) {
        Some(smi)
    } else {
        None
    }
}

fn first_smiles_atom_len(smi: &str) -> Option<usize> {
    let b = smi.as_bytes();
    if b.is_empty() {
        return None;
    }
    if b[0] == b'[' {
        let close = smi.find(']')?;
        return Some(close + 1);
    }
    if smi.starts_with("Cl") || smi.starts_with("Br") {
        return Some(2);
    }
    Some(1)
}
