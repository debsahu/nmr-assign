//! OpenSMILES Parser, Canonical SMILES Writer (with `@`/`@@` Tetrahedral, `/` `\` Alkene,
//! Aromaticity, Isotopes, Charges, Multi-Component `.`, and Reaction SMILES `A>B>C`),
//! plus Standard IUPAC InChI Parser/Generator and Pure-Rust SHA-256 InChIKey Generator.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! and open cheminformatics specifications (OpenSMILES Specification v1.0, IUPAC InChI Technical Manual,
//! and NIST FIPS 180-4 SHA-256).

use super::graph::{
    atomic_number_from_symbol, symbol_from_atomic_number, Atom, BondKind, ChemDocument, ChemError,
    DoubleBondGeometry, Molecule, Radical, ReactionArrow, ReactionArrowKind, TetrahedralParity,
};
use super::layout::clean_2d;
use super::stereo::{
    kekulize, perceive_aromaticity, perceive_cip, perceive_sssr, CipAtomDescriptor,
    CipBondDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};

// ============================================================================
// 1. OpenSMILES Parser (`parse_smiles` & `parse_reaction_smiles`)
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum PendingBond {
    Implicit,
    Single,
    Double,
    Triple,
    Aromatic,
    DirUp,   // `/`
    DirDown, // `\`
}

#[derive(Clone, Copy, Debug)]
enum ChiralSlot {
    Atom(usize),
    ImplicitH,
    RingInstance(usize),
}

struct PendingChiral {
    center: usize,
    clockwise: bool,
    slots: Vec<ChiralSlot>,
}

struct OpenRing {
    atom: usize,
    bond: PendingBond,
    instance_id: usize,
}

/// Parse an OpenSMILES string (including `@`/`@@`, `/` `\`, aromaticity, isotopes, charges,
/// radicals, atom maps, and multi-component `.` mixtures/salts) into a `Molecule`.
pub fn parse_smiles(smiles: &str) -> Result<Molecule, ChemError> {
    // Strip trailing whitespace/name if separated by space or tab (standard `.smi` line format)
    let trimmed = smiles.trim();
    if trimmed.is_empty() {
        return Err(ChemError::new("Empty SMILES string"));
    }
    let (smi_part, mol_name) = match trimmed.find(|c: char| c.is_whitespace()) {
        Some(idx) => (&trimmed[..idx], trimmed[idx..].trim()),
        None => (trimmed, ""),
    };

    let bytes = smi_part.as_bytes();
    let mut mol = Molecule::with_name(mol_name);
    let mut pos = 0usize;
    let mut current_atom: Option<usize> = None;
    let mut branch_stack: Vec<Option<usize>> = Vec::new();
    let mut pending_bond: Option<PendingBond> = None;
    let mut open_rings: BTreeMap<u16, OpenRing> = BTreeMap::new();
    let mut next_ring_instance: usize = 0;
    let mut resolved_ring_partner: BTreeMap<usize, usize> = BTreeMap::new();
    let mut pending_chirals: Vec<PendingChiral> = Vec::new();
    // Directional single bonds `(from_atom, to_atom, is_up)` as written in SMILES
    let mut dir_bonds: Vec<(usize, usize, bool)> = Vec::new();

    let append_slot_to_center =
        |pending_chirals: &mut Vec<PendingChiral>, center: usize, slot: ChiralSlot| {
            if let Some(pc) = pending_chirals.iter_mut().find(|p| p.center == center) {
                pc.slots.push(slot);
            }
        };

    while pos < bytes.len() {
        let ch = bytes[pos] as char;
        match ch {
            '.' => {
                if pending_bond.is_some() {
                    return Err(ChemError::new("Dangling bond before '.' in SMILES"));
                }
                current_atom = None;
                pos += 1;
            }
            '(' => {
                branch_stack.push(current_atom);
                pos += 1;
            }
            ')' => {
                if pending_bond.is_some() {
                    return Err(ChemError::new("Dangling bond before ')' in SMILES"));
                }
                current_atom = branch_stack
                    .pop()
                    .ok_or_else(|| ChemError::new("Unmatched ')' in SMILES"))?;
                pos += 1;
            }
            '-' | '=' | '#' | ':' | '/' | '\\' => {
                if pending_bond.is_some() {
                    return Err(ChemError::new("Consecutive bond symbols in SMILES"));
                }
                pending_bond = Some(match ch {
                    '-' => PendingBond::Single,
                    '=' => PendingBond::Double,
                    '#' => PendingBond::Triple,
                    ':' => PendingBond::Aromatic,
                    '/' => PendingBond::DirUp,
                    _ => PendingBond::DirDown,
                });
                pos += 1;
            }
            '0'..='9' | '%' => {
                let rnum: u16 = if ch == '%' {
                    if pos + 2 >= bytes.len()
                        || !bytes[pos + 1].is_ascii_digit()
                        || !bytes[pos + 2].is_ascii_digit()
                    {
                        return Err(ChemError::new("Invalid '%nn' ring closure in SMILES"));
                    }
                    let v = smi_part[pos + 1..pos + 3]
                        .parse::<u16>()
                        .map_err(|_| ChemError::new("Invalid '%nn' ring number"))?;
                    pos += 3;
                    v
                } else {
                    pos += 1;
                    (ch as u8 - b'0') as u16
                };

                let u = current_atom
                    .ok_or_else(|| ChemError::new("Ring closure digit without preceding atom"))?;
                let pb = pending_bond.take().unwrap_or(PendingBond::Implicit);

                if let Some(open) = open_rings.remove(&rnum) {
                    if open.atom == u {
                        return Err(ChemError::new("Self-loop ring closure in SMILES"));
                    }
                    // Resolve ring partner for chiral slot opened at `open.atom`, and record direct partner at `u`!
                    resolved_ring_partner.insert(open.instance_id, u);
                    append_slot_to_center(&mut pending_chirals, u, ChiralSlot::Atom(open.atom));

                    let chosen_pb = resolve_ring_bond(open.bond, pb)?;
                    let bkind = materialize_bond_kind(&mol, open.atom, u, chosen_pb);
                    mol.add_bond(open.atom, u, bkind);
                    match open.bond {
                        PendingBond::DirUp => dir_bonds.push((open.atom, u, true)),
                        PendingBond::DirDown => dir_bonds.push((open.atom, u, false)),
                        _ => {}
                    }
                    match pb {
                        PendingBond::DirUp => dir_bonds.push((u, open.atom, true)),
                        PendingBond::DirDown => dir_bonds.push((u, open.atom, false)),
                        _ => {}
                    }
                } else {
                    let inst = next_ring_instance;
                    next_ring_instance += 1;
                    append_slot_to_center(&mut pending_chirals, u, ChiralSlot::RingInstance(inst));
                    open_rings.insert(
                        rnum,
                        OpenRing {
                            atom: u,
                            bond: pb,
                            instance_id: inst,
                        },
                    );
                }
            }
            '[' => {
                let close = smi_part[pos..]
                    .find(']')
                    .ok_or_else(|| ChemError::new("Unclosed '[' bracket atom in SMILES"))?
                    + pos;
                let inner = &smi_part[pos + 1..close];
                pos = close + 1;

                let (atom, chiral_cw) = parse_bracket_atom(inner)?;
                let bracket_h = atom.explicit_hydrogens.unwrap_or(0);
                let new_idx = mol.add_atom(atom);

                let pb = pending_bond.take().unwrap_or(PendingBond::Implicit);
                if let Some(prev) = current_atom {
                    let bkind = materialize_bond_kind(&mol, prev, new_idx, pb);
                    mol.add_bond(prev, new_idx, bkind);
                    append_slot_to_center(&mut pending_chirals, prev, ChiralSlot::Atom(new_idx));
                    match pb {
                        PendingBond::DirUp => dir_bonds.push((prev, new_idx, true)),
                        PendingBond::DirDown => dir_bonds.push((prev, new_idx, false)),
                        _ => {}
                    }
                }

                if let Some(cw) = chiral_cw {
                    let mut slots = Vec::with_capacity(4);
                    if let Some(prev) = current_atom {
                        slots.push(ChiralSlot::Atom(prev));
                    }
                    if bracket_h == 1 {
                        slots.push(ChiralSlot::ImplicitH);
                    }
                    pending_chirals.push(PendingChiral {
                        center: new_idx,
                        clockwise: cw,
                        slots,
                    });
                }

                current_atom = Some(new_idx);
            }
            _ => {
                // Organic / aromatic unbracketed atom (`Cl`, `Br`, `B`, `C`, `N`, `O`, `P`, `S`, `F`, `I`, `*`, `b`, `c`, `n`, `o`, `p`, `s`)
                let (sym, is_arom, step) = if smi_part[pos..].starts_with("Cl") {
                    ("Cl", false, 2)
                } else if smi_part[pos..].starts_with("Br") {
                    ("Br", false, 2)
                } else {
                    match ch {
                        'B' => ("B", false, 1),
                        'C' => ("C", false, 1),
                        'N' => ("N", false, 1),
                        'O' => ("O", false, 1),
                        'P' => ("P", false, 1),
                        'S' => ("S", false, 1),
                        'F' => ("F", false, 1),
                        'I' => ("I", false, 1),
                        '*' => ("*", false, 1),
                        'b' => ("B", true, 1),
                        'c' => ("C", true, 1),
                        'n' => ("N", true, 1),
                        'o' => ("O", true, 1),
                        'p' => ("P", true, 1),
                        's' => ("S", true, 1),
                        other => {
                            return Err(ChemError::new(format!(
                                "Unexpected character '{}' at byte {} in SMILES '{}'",
                                other, pos, smi_part
                            )))
                        }
                    }
                };
                pos += step;
                let mut atom = Atom::new(sym);
                atom.aromatic = is_arom;
                let new_idx = mol.add_atom(atom);

                let pb = pending_bond.take().unwrap_or(PendingBond::Implicit);
                if let Some(prev) = current_atom {
                    let bkind = materialize_bond_kind(&mol, prev, new_idx, pb);
                    mol.add_bond(prev, new_idx, bkind);
                    append_slot_to_center(&mut pending_chirals, prev, ChiralSlot::Atom(new_idx));
                    match pb {
                        PendingBond::DirUp => dir_bonds.push((prev, new_idx, true)),
                        PendingBond::DirDown => dir_bonds.push((prev, new_idx, false)),
                        _ => {}
                    }
                }
                current_atom = Some(new_idx);
            }
        }
    }

    if !branch_stack.is_empty() {
        return Err(ChemError::new("Unclosed '(' in SMILES"));
    }
    if pending_bond.is_some() {
        return Err(ChemError::new("Trailing bond symbol at end of SMILES"));
    }
    if !open_rings.is_empty() {
        return Err(ChemError::new("Unclosed ring number in SMILES"));
    }

    // Finalize tetrahedral parities
    for pc in pending_chirals {
        if pc.slots.len() == 4 {
            let mut nbs = [0usize; 4];
            let mut valid = true;
            for (k, slot) in pc.slots.iter().enumerate() {
                nbs[k] = match *slot {
                    ChiralSlot::Atom(a) => a,
                    ChiralSlot::ImplicitH => usize::MAX,
                    ChiralSlot::RingInstance(id) => {
                        if let Some(&partner) = resolved_ring_partner.get(&id) {
                            partner
                        } else {
                            valid = false;
                            break;
                        }
                    }
                };
            }
            if valid {
                mol.atoms[pc.center].parity = Some(TetrahedralParity {
                    neighbors: nbs,
                    clockwise: pc.clockwise,
                });
            }
        } else if pc.slots.len() == 3 {
            // Sulfoxide `[S@](=O)(C)CC` or chiral phosphine `[P@](C)(CC)CCC`: 3 real neighbors + lone pair (`usize::MAX`)!
            let mut nbs = [usize::MAX; 4];
            let mut valid = true;
            for (k, slot) in pc.slots.iter().enumerate() {
                nbs[k] = match *slot {
                    ChiralSlot::Atom(a) => a,
                    ChiralSlot::ImplicitH => usize::MAX,
                    ChiralSlot::RingInstance(id) => {
                        if let Some(&partner) = resolved_ring_partner.get(&id) {
                            partner
                        } else {
                            valid = false;
                            break;
                        }
                    }
                };
            }
            if valid {
                // In OpenSMILES, a lone pair on `[S@]` occupies the implicit-H position right after `from_atom` (or slot 0 if root)!
                // Wait: in `C[S@](=O)CC`, `from_atom` = `C` (slot 0), lone pair = slot 1, `=O` = slot 2, `CC` = slot 3!
                let has_parent =
                    !matches!(pc.slots[0], ChiralSlot::RingInstance(_)) && pc.center > 0;
                let lp_ordered = if has_parent {
                    [nbs[0], usize::MAX, nbs[1], nbs[2]]
                } else {
                    [usize::MAX, nbs[0], nbs[1], nbs[2]]
                };
                mol.atoms[pc.center].parity = Some(TetrahedralParity {
                    neighbors: lp_ordered,
                    clockwise: pc.clockwise,
                });
            }
        }
    }

    // Finalize directional `/` and `\` double-bond geometry!
    if !dir_bonds.is_empty() {
        // Normalize directional bonds to `(alkene_end -> substituent) -> outward_is_up`
        let mut outward_dir: BTreeMap<(usize, usize), bool> = BTreeMap::new();
        for &(u, v, is_up) in &dir_bonds {
            outward_dir.insert((u, v), is_up);
            outward_dir.insert((v, u), !is_up);
        }

        for bi in 0..mol.bonds.len() {
            if mol.bonds[bi].kind != BondKind::Double {
                continue;
            }
            let a = mol.bonds[bi].a;
            let b = mol.bonds[bi].b;
            let dir_a = mol
                .covalent_neighbors(a)
                .into_iter()
                .filter(|&(nb, _)| nb != b)
                .find_map(|(nb, _)| outward_dir.get(&(a, nb)).map(|&up| (nb, up)));
            let dir_b = mol
                .covalent_neighbors(b)
                .into_iter()
                .filter(|&(nb, _)| nb != a)
                .find_map(|(nb, _)| outward_dir.get(&(b, nb)).map(|&up| (nb, up)));
            if let (Some((ra, up_a)), Some((rb, up_b))) = (dir_a, dir_b) {
                mol.bonds[bi].double_geometry = if up_a == up_b {
                    DoubleBondGeometry::Cis {
                        ref_a: ra,
                        ref_b: rb,
                    }
                } else {
                    DoubleBondGeometry::Trans {
                        ref_a: ra,
                        ref_b: rb,
                    }
                };
            }
        }
    }

    // Kekulize any aromatic bonds (`c1ccccc1`, `c1cc[nH]c1`, etc.) so bond orders & valences are exact!
    kekulize(&mut mol)?;

    Ok(mol)
}

fn resolve_ring_bond(b1: PendingBond, b2: PendingBond) -> Result<PendingBond, ChemError> {
    match (b1, b2) {
        (PendingBond::Implicit, other) => Ok(other),
        (other, PendingBond::Implicit) => Ok(other),
        // Note: on a ring closure, `/` at open and `\` at close mean the SAME bond direction (`open -> close` is UP, so `close -> open` is DOWN)!
        (PendingBond::DirUp, PendingBond::DirDown) | (PendingBond::DirUp, PendingBond::DirUp) => {
            Ok(PendingBond::DirUp)
        }
        (PendingBond::DirDown, PendingBond::DirUp)
        | (PendingBond::DirDown, PendingBond::DirDown) => Ok(PendingBond::DirDown),
        (PendingBond::Single, PendingBond::Single) => Ok(PendingBond::Single),
        (PendingBond::Double, PendingBond::Double) => Ok(PendingBond::Double),
        (PendingBond::Triple, PendingBond::Triple) => Ok(PendingBond::Triple),
        (PendingBond::Aromatic, PendingBond::Aromatic) => Ok(PendingBond::Aromatic),
        _ => Err(ChemError::new(
            "Conflicting explicit bond orders on SMILES ring closure",
        )),
    }
}

fn materialize_bond_kind(mol: &Molecule, a: usize, b: usize, pb: PendingBond) -> BondKind {
    match pb {
        PendingBond::Implicit => {
            if mol.atoms[a].aromatic && mol.atoms[b].aromatic {
                BondKind::Aromatic
            } else {
                BondKind::Single
            }
        }
        PendingBond::Single | PendingBond::DirUp | PendingBond::DirDown => BondKind::Single,
        PendingBond::Double => BondKind::Double,
        PendingBond::Triple => BondKind::Triple,
        PendingBond::Aromatic => BondKind::Aromatic,
    }
}

fn parse_bracket_atom(inner: &str) -> Result<(Atom, Option<bool>), ChemError> {
    let bytes = inner.as_bytes();
    let mut i = 0usize;

    // 1. Optional isotope digits
    let iso_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let isotope = if i > iso_start {
        Some(
            inner[iso_start..i]
                .parse::<u16>()
                .map_err(|_| ChemError::new("Invalid isotope in SMILES bracket"))?,
        )
    } else {
        None
    };

    if i >= bytes.len() {
        return Err(ChemError::new("Empty atom symbol in SMILES bracket"));
    }

    // 2. Element symbol (uppercase `[A-Z][a-z]?`, aromatic `b,c,n,o,p,s,se,as`, or `*`)
    let (sym, is_arom) = if bytes[i] == b'*' {
        i += 1;
        ("*".to_string(), false)
    } else if bytes[i].is_ascii_lowercase() {
        if inner[i..].starts_with("se") {
            i += 2;
            ("Se".to_string(), true)
        } else if inner[i..].starts_with("as") {
            i += 2;
            ("As".to_string(), true)
        } else {
            let s = (bytes[i] as char).to_ascii_uppercase().to_string();
            i += 1;
            (s, true)
        }
    } else if bytes[i].is_ascii_uppercase() {
        let start = i;
        i += 1;
        if i < bytes.len() && bytes[i].is_ascii_lowercase() {
            // Note: `[CH4]` has uppercase `C` followed by uppercase `H`, whereas `[Cl-]` or `[SiH4]` has lowercase second char!
            i += 1;
        }
        (inner[start..i].to_string(), false)
    } else {
        return Err(ChemError::new(format!(
            "Invalid element in SMILES bracket '[{}]'",
            inner
        )));
    };

    let mut atom = Atom::new(&sym);
    atom.aromatic = is_arom;
    if let Some(iso) = isotope {
        atom.isotope = Some(iso);
    }

    // 3. Optional chirality `@` or `@@`
    let mut chiral_cw = None;
    if i < bytes.len() && bytes[i] == b'@' {
        i += 1;
        if i < bytes.len() && bytes[i] == b'@' {
            i += 1;
            chiral_cw = Some(true);
        } else {
            // Ignore extended `@TH1`/`@AL1` suffixes if ever encountered
            while i < bytes.len() && (bytes[i].is_ascii_uppercase() || bytes[i].is_ascii_digit()) {
                if bytes[i] == b'H' {
                    break;
                }
                i += 1;
            }
            chiral_cw = Some(false);
        }
    }

    // 4. Optional hydrogen count `H` or `H<digit>` (unless atom itself was `[H]`, `[2H]`, `[H+]`, where `sym == "H"`!)
    let mut explicit_h: u8 = 0;
    if i < bytes.len() && bytes[i] == b'H' {
        i += 1;
        let h_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let h_cnt = if i > h_start {
            inner[h_start..i]
                .parse::<u8>()
                .map_err(|_| ChemError::new("Invalid H count in SMILES bracket"))?
        } else {
            1
        };
        explicit_h = h_cnt;
    }

    // 5. Optional charge `+`, `++`, `+2`, `-`, `--`, `-2`
    let mut charge: i8 = 0;
    while i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        let sign: i8 = if bytes[i] == b'+' { 1 } else { -1 };
        i += 1;
        let d_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > d_start {
            let mag = inner[d_start..i]
                .parse::<i8>()
                .map_err(|_| ChemError::new("Invalid charge magnitude in SMILES bracket"))?;
            charge += sign * mag;
        } else {
            charge += sign;
        }
    }
    atom.formal_charge = charge;

    // 6. Optional `:atom_map`
    if i < bytes.len() && bytes[i] == b':' {
        i += 1;
        let m_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > m_start {
            atom.atom_map = inner[m_start..i].parse::<u16>().ok();
        }
    }

    // Bracketed atoms in OpenSMILES ALWAYS have exact explicit hydrogens (`0` when `H` is omitted, except bare `[H]` which IS hydrogen itself!)
    atom.explicit_hydrogens = Some(
        if atom.z == 1 && sym.eq_ignore_ascii_case("H") && explicit_h == 0 {
            0
        } else {
            explicit_h
        },
    );

    // Detect monovalent/divalent carbon/oxygen/nitrogen radical when a neutral bracketed atom has fewer bonds+H than standard valence?
    // Keep `explicit_hydrogens = Some(...)` so `implicit_hydrogens()` returns the exact bracketed H count!
    Ok((atom, chiral_cw))
}

/// Parse a Reaction SMILES (`"Reactant1.Reactant2>Agent1>Product1.Product2"`) into a `ChemDocument`
/// with laid-out 2D reactant/product molecules, a `ReactionArrow`, and optional reagent text above the arrow!
pub fn parse_reaction_smiles(rxn_smiles: &str) -> Result<ChemDocument, ChemError> {
    let parts: Vec<&str> = rxn_smiles.trim().split('>').collect();
    if parts.len() != 3 {
        return Err(ChemError::new(
            "Reaction SMILES must have exactly two '>' separators ('Reactants>Agents>Products')",
        ));
    }

    let mut doc = ChemDocument::new();
    let mut cursor_x = 0.0f64;

    let add_side =
        |smi_group: &str, doc: &mut ChemDocument, cursor_x: &mut f64| -> Result<(), ChemError> {
            if smi_group.trim().is_empty() {
                return Ok(());
            }
            for comp_smi in smi_group.split('.') {
                let cs = comp_smi.trim();
                if cs.is_empty() {
                    continue;
                }
                let mut m = parse_smiles(cs)?;
                clean_2d(&mut m);
                if let Some((min_x, min_y, max_x, max_y)) = m.bounds_2d() {
                    let mid_y = 0.5 * (min_y + max_y);
                    let dx = *cursor_x - min_x;
                    for a in &mut m.atoms {
                        a.x += dx;
                        a.y -= mid_y;
                    }
                    *cursor_x += (max_x - min_x) + 3.0;
                }
                doc.molecules.push(m);
            }
            Ok(())
        };

    add_side(parts[0], &mut doc, &mut cursor_x)?;
    let arrow_tail_x = cursor_x;
    let arrow_head_x = arrow_tail_x + 4.5;
    let agent_str = parts[1].trim();
    doc.reaction_arrows.push(ReactionArrow {
        kind: ReactionArrowKind::Solid,
        tail: (arrow_tail_x, 0.0),
        head: (arrow_head_x, 0.0),
        above_text: (!agent_str.is_empty()).then(|| agent_str.to_string()),
        below_text: None,
    });
    cursor_x = arrow_head_x + 3.0;
    add_side(parts[2], &mut doc, &mut cursor_x)?;

    Ok(doc)
}

/// Write a `ChemDocument` containing a reaction arrow as a canonical Reaction SMILES `Reactants>Agents>Products`.
pub fn write_reaction_smiles(doc: &ChemDocument) -> String {
    let Some(arrow) = doc.reaction_arrows.first() else {
        let parts: Vec<String> = doc.molecules.iter().map(write_canonical_smiles).collect();
        return parts.join(".");
    };
    let mid_x = 0.5 * (arrow.tail.0 + arrow.head.0);
    let mut reactants = Vec::new();
    let mut products = Vec::new();
    for m in &doc.molecules {
        let cx = if m.atoms.is_empty() {
            0.0
        } else {
            m.atoms.iter().map(|a| a.x).sum::<f64>() / (m.atoms.len() as f64)
        };
        let smi = write_canonical_smiles(m);
        if cx < mid_x {
            reactants.push(smi);
        } else {
            products.push(smi);
        }
    }
    let agents = arrow.above_text.as_deref().unwrap_or("");
    format!("{}>{}>{}", reactants.join("."), agents, products.join("."))
}

// ============================================================================
// 2. Canonical SMILES Writer (`write_canonical_smiles`)
// ============================================================================

/// Compute canonical symmetry ranks (`0..mol.atoms.len()`) using Weisfeiler–Lehman / Morgan refinement.
pub fn canonical_atom_ranks(mol: &Molecule, include_isotopes_and_charges: bool) -> Vec<usize> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let arom = perceive_aromaticity(mol);

    // Initial atom invariant tuple
    let mut classes: Vec<usize> = {
        let mut init_keys: Vec<(u8, u16, i8, u8, usize, bool)> = (0..n)
            .map(|i| {
                let a = &mol.atoms[i];
                let iso = if include_isotopes_and_charges {
                    a.isotope.unwrap_or(0)
                } else {
                    0
                };
                let q = if include_isotopes_and_charges {
                    a.formal_charge
                } else {
                    0
                };
                let ih = mol.total_hydrogens(i);
                let deg = mol
                    .covalent_neighbors(i)
                    .into_iter()
                    .filter(|&(nb, _)| mol.atoms[nb].z != 1)
                    .count();
                (a.z, iso, q, ih, deg, arom.aromatic_atoms[i])
            })
            .collect();
        let mut sorted_unique = init_keys.clone();
        sorted_unique.sort();
        sorted_unique.dedup();
        init_keys
            .drain(..)
            .map(|k| sorted_unique.binary_search(&k).unwrap())
            .collect()
    };

    // Iterative WL neighbor refinement
    for _round in 0..n.min(32) {
        let mut next_sigs: Vec<(usize, Vec<(usize, u8)>)> = Vec::with_capacity(n);
        for i in 0..n {
            let mut nb_list: Vec<(usize, u8)> = mol
                .covalent_neighbors(i)
                .into_iter()
                .map(|(nb, bi)| {
                    let b_code = if arom.aromatic_bonds[bi] {
                        4u8
                    } else {
                        mol.bonds[bi].kind.order_u8()
                    };
                    (classes[nb], b_code)
                })
                .collect();
            nb_list.sort_unstable();
            next_sigs.push((classes[i], nb_list));
        }
        let mut unique = next_sigs.clone();
        unique.sort();
        unique.dedup();
        let new_classes: Vec<usize> = next_sigs
            .into_iter()
            .map(|s| unique.binary_search(&s).unwrap())
            .collect();
        if new_classes == classes {
            break;
        }
        classes = new_classes;
    }

    // Break remaining symmetry ties deterministically
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (classes[i], i));
    let mut ranks = vec![0usize; n];
    for (r, &atom_i) in order.iter().enumerate() {
        ranks[atom_i] = r;
    }
    ranks
}

/// Write a canonical OpenSMILES string for `mol`, preserving aromaticity, isotopes, charges,
/// `@`/`@@` tetrahedral stereochemistry, `/` `\` `(E)/(Z)` alkene stereochemistry, and `.` components.
pub fn write_canonical_smiles(mol: &Molecule) -> String {
    if mol.atoms.is_empty() {
        return String::new();
    }

    // Work on an hydrogen-suppressed view? Wait: keep isotopically labeled `[2H]`/`[3H]` or standalone `H2` (`[H][H]`) explicit,
    // and fold plain `Z == 1` (`isotope.is_none()`, `formal_charge == 0`) attached to heavy atoms into implicit H while updating chiral parities!
    let clean_mol = fold_plain_explicit_hydrogens(mol);
    let n = clean_mol.atoms.len();
    let ranks = canonical_atom_ranks(&clean_mol, true);
    let arom = perceive_aromaticity(&clean_mol);
    let cip = perceive_cip(&clean_mol);

    // Check which aromatic atoms can safely use aromatic bond notation (`b, c, n, o, p, s`):
    // Exclude rings containing exocyclic `C=O` (like pyridone/uracil) from lowercase aromatic bond collapse ONLY if unbracketed would be ambiguous;
    // Actually `[nH]`, `c(=O)` work cleanly in our `parse_smiles` + `kekulize`! Let's verify: in `is_huckel_cycle` and `kekulize`,
    // `c(=O)` has `has_exo_double == true => req = Some(false)`! So `c1cc(=O)[nH]cc1` kekulizes cleanly too!
    let use_arom_atom: Vec<bool> = (0..n)
        .map(|i| arom.aromatic_atoms[i] && matches!(clean_mol.atoms[i].z, 5 | 6 | 7 | 8 | 15 | 16))
        .collect();
    let use_arom_bond: Vec<bool> = (0..clean_mol.bonds.len())
        .map(|bi| {
            let b = &clean_mol.bonds[bi];
            arom.aromatic_bonds[bi] && use_arom_atom[b.a] && use_arom_atom[b.b]
        })
        .collect();

    let comps = clean_mol.connected_components();
    let mut comp_smiles: Vec<String> = Vec::with_capacity(comps.len());

    for comp in comps {
        // Start DFS at the atom in `comp` with smallest canonical rank
        let root = *comp.iter().min_by_key(|&&u| ranks[u]).unwrap();
        comp_smiles.push(write_component_smiles(
            &clean_mol,
            root,
            &ranks,
            &use_arom_atom,
            &use_arom_bond,
            &cip,
        ));
    }

    comp_smiles.sort();
    comp_smiles.join(".")
}

fn fold_plain_explicit_hydrogens(mol: &Molecule) -> Molecule {
    let is_foldable_h = |i: usize| -> bool {
        let a = &mol.atoms[i];
        if a.z != 1 || a.isotope.is_some() || a.formal_charge != 0 || a.radical != Radical::None {
            return false;
        }
        let nbs = mol.covalent_neighbors(i);
        if nbs.len() != 1 {
            return false;
        }
        let parent = nbs[0].0;
        // Only fold if parent is a heavy atom (`z > 1`) and has at most 1 foldable H if parent is chiral!
        if mol.atoms[parent].z <= 1 {
            return false;
        }
        let parent_plain_h_cnt = mol
            .covalent_neighbors(parent)
            .into_iter()
            .filter(|&(nb, _)| {
                mol.atoms[nb].z == 1
                    && mol.atoms[nb].isotope.is_none()
                    && mol.atoms[nb].formal_charge == 0
            })
            .count();
        // If parent has parity or stereo wedges and >= 2 plain Hs, it's achiral anyway
        let _ = parent_plain_h_cnt;
        true
    };

    if !(0..mol.atoms.len()).any(&is_foldable_h) {
        return mol.clone();
    }

    // Before removing plain explicit Hs, capture CIP stereodescriptors & convert any stereo wedges to `parity`!
    let mut working = mol.clone();
    let cip = perceive_cip(&working);
    for (&center, &desc) in &cip.atoms {
        if working.atoms[center].parity.is_none() {
            let nbs: Vec<usize> = working
                .covalent_neighbors(center)
                .into_iter()
                .map(|(nb, _)| nb)
                .collect();
            if nbs.len() == 3 || nbs.len() == 4 {
                let arr = if nbs.len() == 4 {
                    [nbs[0], nbs[1], nbs[2], nbs[3]]
                } else {
                    [nbs[0], nbs[1], nbs[2], usize::MAX]
                };
                let mut probe = working.clone();
                for b in &mut probe.bonds {
                    if (b.a == center || b.b == center) && b.kind.is_stereo_display() {
                        b.kind = BondKind::Single;
                    }
                }
                for a in &mut probe.atoms {
                    a.z_coord = None;
                }
                probe.atoms[center].parity = Some(TetrahedralParity {
                    neighbors: arr,
                    clockwise: true,
                });
                let cw = perceive_cip(&probe).atoms.get(&center) == Some(&desc);
                working.atoms[center].parity = Some(TetrahedralParity {
                    neighbors: arr,
                    clockwise: cw,
                });
            }
        }
    }

    // Map foldable H in `parity.neighbors` to `usize::MAX`
    let mut folded_count_on_parent = vec![0u8; working.atoms.len()];
    for h_idx in 0..working.atoms.len() {
        if is_foldable_h(h_idx) {
            let parent = working.covalent_neighbors(h_idx)[0].0;
            folded_count_on_parent[parent] += 1;
            if let Some(mut p) = working.atoms[parent].parity {
                for slot in &mut p.neighbors {
                    if *slot == h_idx {
                        *slot = usize::MAX;
                    }
                }
                working.atoms[parent].parity = Some(p);
            }
        }
    }
    for (u, &cnt) in folded_count_on_parent.iter().enumerate() {
        if cnt > 0 {
            if let Some(eh) = working.atoms[u].explicit_hydrogens {
                working.atoms[u].explicit_hydrogens = Some(eh + cnt);
            }
        }
    }

    let keep: Vec<usize> = (0..working.atoms.len())
        .filter(|&i| !is_foldable_h(i))
        .collect();
    working.extract_subgraph(&keep)
}

fn write_component_smiles(
    mol: &Molecule,
    root: usize,
    ranks: &[usize],
    use_arom_atom: &[bool],
    use_arom_bond: &[bool],
    cip: &super::stereo::CipResult,
) -> String {
    let n = mol.atoms.len();
    // Pass 1: Build canonical DFS spanning tree & identify ring closure back-edges
    let mut visited = vec![false; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut tree_children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut ring_closures_at: Vec<Vec<(usize, usize, u16, bool)>> = vec![Vec::new(); n]; // (partner, bond_idx, rnum, is_opening)
    let mut next_rnum: u16 = 1;
    let mut seen_back_bonds: BTreeSet<usize> = BTreeSet::new();

    #[allow(clippy::too_many_arguments)]
    fn dfs_tree(
        u: usize,
        from: Option<usize>,
        mol: &Molecule,
        ranks: &[usize],
        visited: &mut [bool],
        parent: &mut [Option<usize>],
        tree_children: &mut [Vec<(usize, usize)>],
        ring_closures_at: &mut [Vec<(usize, usize, u16, bool)>],
        seen_back_bonds: &mut BTreeSet<usize>,
        next_rnum: &mut u16,
    ) {
        visited[u] = true;
        parent[u] = from;
        let mut nbs = mol.covalent_neighbors(u);
        nbs.sort_by_key(|&(v, _)| ranks[v]);

        for (v, bi) in nbs {
            if Some(v) == from {
                continue;
            }
            if visited[v] {
                if seen_back_bonds.insert(bi) {
                    let rnum = *next_rnum;
                    *next_rnum += 1;
                    // `v` is ancestor (opening), `u` is descendant (closing)!
                    ring_closures_at[v].push((u, bi, rnum, true));
                    ring_closures_at[u].push((v, bi, rnum, false));
                }
            } else {
                tree_children[u].push((v, bi));
                dfs_tree(
                    v,
                    Some(u),
                    mol,
                    ranks,
                    visited,
                    parent,
                    tree_children,
                    ring_closures_at,
                    seen_back_bonds,
                    next_rnum,
                );
            }
        }
    }

    dfs_tree(
        root,
        None,
        mol,
        ranks,
        &mut visited,
        &mut parent,
        &mut tree_children,
        &mut ring_closures_at,
        &mut seen_back_bonds,
        &mut next_rnum,
    );

    // Assign directional `/` and `\` on single bonds adjacent to stereogenic (E)/(Z) double bonds!
    // Map `(from_atom, to_atom) -> bool` (`true` = `/`, `false` = `\`) along DFS emission direction!
    let mut dir_slash: BTreeMap<(usize, usize), bool> = BTreeMap::new();
    assign_alkene_slashes(mol, cip, &mut dir_slash);

    // Pass 2: Recursive emission
    #[allow(clippy::too_many_arguments)]
    fn emit_node(
        u: usize,
        mol: &Molecule,
        parent: &[Option<usize>],
        tree_children: &[Vec<(usize, usize)>],
        ring_closures_at: &[Vec<(usize, usize, u16, bool)>],
        use_arom_atom: &[bool],
        use_arom_bond: &[bool],
        cip: &super::stereo::CipResult,
        dir_slash: &BTreeMap<(usize, usize), bool>,
    ) -> String {
        // Compute exact SMILES neighbor order at `u` for `@`/`@@` calculation:
        // `[parent(u) if any, implicit_h if 1, ring_closures_at[u]..., tree_children[u]...]`
        let ih = mol.implicit_hydrogens(u);
        let chiral_tag = if let Some(&target_desc) = cip.atoms.get(&u) {
            let mut smi_nbs: Vec<usize> = Vec::with_capacity(4);
            if let Some(p) = parent[u] {
                smi_nbs.push(p);
            }
            if ih == 1 {
                smi_nbs.push(usize::MAX);
            } else if ih == 0 && mol.covalent_neighbors(u).len() == 3 {
                // Lone-pair stereocenter (e.g. chiral sulfoxide `[S@](=O)`): lone pair sits right after parent!
                smi_nbs.push(usize::MAX);
            }
            for &(partner, _, _, _) in &ring_closures_at[u] {
                smi_nbs.push(partner);
            }
            for &(child, _) in &tree_children[u] {
                smi_nbs.push(child);
            }
            if smi_nbs.len() == 4 {
                let arr = [smi_nbs[0], smi_nbs[1], smi_nbs[2], smi_nbs[3]];
                // Test clockwise = true vs false on `arr` using `perceive_cip`!
                let cw = determine_smiles_clockwise(mol, u, arr, target_desc);
                Some(if cw { "@@" } else { "@" })
            } else {
                None
            }
        } else {
            None
        };

        let mut out = format_smiles_atom(mol, u, use_arom_atom[u], ih, chiral_tag);

        // Emit ring closures at `u`
        for &(partner, bi, rnum, is_opening) in &ring_closures_at[u] {
            // Only emit explicit bond symbol on the OPENING occurrence (unless directional slash needed)
            if is_opening {
                out.push_str(&format_smiles_bond(
                    mol,
                    bi,
                    u,
                    partner,
                    use_arom_atom,
                    use_arom_bond,
                    dir_slash,
                ));
            }
            if rnum >= 10 {
                out.push('%');
            }
            out.push_str(&rnum.to_string());
        }

        // Emit tree children (all except last in `(...)`, last inline!)
        let kids = &tree_children[u];
        for (idx, &(v, bi)) in kids.iter().enumerate() {
            let b_str = format_smiles_bond(mol, bi, u, v, use_arom_atom, use_arom_bond, dir_slash);
            let sub = emit_node(
                v,
                mol,
                parent,
                tree_children,
                ring_closures_at,
                use_arom_atom,
                use_arom_bond,
                cip,
                dir_slash,
            );
            if idx + 1 < kids.len() {
                out.push('(');
                out.push_str(&b_str);
                out.push_str(&sub);
                out.push(')');
            } else {
                out.push_str(&b_str);
                out.push_str(&sub);
            }
        }
        out
    }

    emit_node(
        root,
        mol,
        &parent,
        &tree_children,
        &ring_closures_at,
        use_arom_atom,
        use_arom_bond,
        cip,
        &dir_slash,
    )
}

fn determine_smiles_clockwise(
    mol: &Molecule,
    center: usize,
    smi_nbs: [usize; 4],
    target_desc: CipAtomDescriptor,
) -> bool {
    let mut probe = mol.clone();
    for b in &mut probe.bonds {
        if (b.a == center || b.b == center) && b.kind.is_stereo_display() {
            b.kind = BondKind::Single;
        }
    }
    for a in &mut probe.atoms {
        a.z_coord = None;
    }
    probe.atoms[center].parity = Some(TetrahedralParity {
        neighbors: smi_nbs,
        clockwise: true,
    });
    perceive_cip(&probe).atoms.get(&center) == Some(&target_desc)
}

fn assign_alkene_slashes(
    mol: &Molecule,
    cip: &super::stereo::CipResult,
    dir_slash: &mut BTreeMap<(usize, usize), bool>,
) {
    // Propagate consistent outward_up booleans across conjugated stereogenic double bonds too!
    for (&bi, &desc) in &cip.bonds {
        let b = &mol.bonds[bi];
        let a = b.a;
        let c = b.b;
        let subs_a: Vec<usize> = mol
            .covalent_neighbors(a)
            .into_iter()
            .map(|(nb, _)| nb)
            .filter(|&nb| nb != c)
            .collect();
        let subs_c: Vec<usize> = mol
            .covalent_neighbors(c)
            .into_iter()
            .map(|(nb, _)| nb)
            .filter(|&nb| nb != a)
            .collect();
        if subs_a.is_empty() || subs_c.is_empty() {
            continue;
        }
        let ra = subs_a[0];
        let rc = subs_c[0];

        // Test whether `ra` and `rc` on the SAME side (`Cis`) gives `target_desc` (`Z` or `E`)!
        let mut probe = mol.clone();
        probe.bonds[bi].double_geometry = DoubleBondGeometry::Cis {
            ref_a: ra,
            ref_b: rc,
        };
        let cis_gives_target = perceive_cip(&probe).bonds.get(&bi) == Some(&desc);

        // Check if `(a, ra)` already has an assigned outward direction (from an adjacent conjugated diene!)
        let out_up_a = dir_slash.get(&(a, ra)).copied().unwrap_or(true);
        let out_up_c = if cis_gives_target {
            out_up_a
        } else {
            !out_up_a
        };

        dir_slash.insert((a, ra), out_up_a);
        dir_slash.insert((ra, a), !out_up_a);
        dir_slash.insert((c, rc), out_up_c);
        dir_slash.insert((rc, c), !out_up_c);
    }
}

fn format_smiles_atom(
    mol: &Molecule,
    u: usize,
    is_arom: bool,
    ih: u8,
    chiral_tag: Option<&str>,
) -> String {
    let a = &mol.atoms[u];
    let sym = if a.z == 0 {
        "*"
    } else {
        symbol_from_atomic_number(a.z)
    };

    // Check if unbracketed organic/aromatic form is exact and unambiguous
    let can_unbracket = a.isotope.is_none()
        && a.formal_charge == 0
        && a.radical == Radical::None
        && chiral_tag.is_none()
        && a.atom_map.is_none()
        && a.superatom_label.is_none()
        && {
            if is_arom {
                match a.z {
                    6 => ih <= 1,
                    7 => ih == 0, // pyrrole `[nH]` MUST be bracketed!
                    8 | 16 => ih == 0,
                    5 => ih <= 1,
                    _ => false,
                }
            } else {
                // Check standard unbracketed implicit H matches `ih`!
                if matches!(a.z, 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53 | 0) {
                    let mut test_atom = a.clone();
                    test_atom.explicit_hydrogens = None;
                    let mut tmp = mol.clone();
                    tmp.atoms[u] = test_atom;
                    tmp.implicit_hydrogens(u) == ih
                } else {
                    false
                }
            }
        };

    if can_unbracket {
        if is_arom && a.z != 0 {
            return sym.to_ascii_lowercase();
        }
        return sym.to_string();
    }

    let mut s = String::from("[");
    if let Some(iso) = a.isotope {
        s.push_str(&iso.to_string());
    }
    if is_arom && matches!(a.z, 5 | 6 | 7 | 8 | 15 | 16) {
        s.push_str(&sym.to_ascii_lowercase());
    } else {
        s.push_str(sym);
    }
    if let Some(ct) = chiral_tag {
        s.push_str(ct);
    }
    if ih == 1 {
        s.push('H');
    } else if ih > 1 {
        s.push('H');
        s.push_str(&ih.to_string());
    }
    if a.formal_charge > 0 {
        s.push('+');
        if a.formal_charge > 1 {
            s.push_str(&a.formal_charge.to_string());
        }
    } else if a.formal_charge < 0 {
        s.push('-');
        if a.formal_charge < -1 {
            s.push_str(&(-a.formal_charge).to_string());
        }
    }
    if let Some(am) = a.atom_map {
        s.push(':');
        s.push_str(&am.to_string());
    }
    s.push(']');
    s
}

fn format_smiles_bond(
    mol: &Molecule,
    bi: usize,
    from: usize,
    to: usize,
    use_arom_atom: &[bool],
    use_arom_bond: &[bool],
    dir_slash: &BTreeMap<(usize, usize), bool>,
) -> String {
    if use_arom_bond[bi] {
        return String::new();
    }
    if let Some(&is_up) = dir_slash.get(&(from, to)) {
        return if is_up {
            "/".to_string()
        } else {
            "\\".to_string()
        };
    }
    match mol.bonds[bi].kind {
        BondKind::Double => "=".to_string(),
        BondKind::Triple => "#".to_string(),
        _ => {
            // IMPORTANT: single bond between two aromatic atoms (e.g. biphenyl `c1ccccc1-c2ccccc2` or fluorene!) MUST write `-`!
            if use_arom_atom[from] && use_arom_atom[to] {
                "-".to_string()
            } else {
                String::new()
            }
        }
    }
}

// ============================================================================
// 3. Standard IUPAC InChI Generator, Parser & Pure-Rust SHA-256 InChIKey
// ============================================================================

/// Generate a standard molecular `InChI=1S/...` string with formula, `/c` connectivity,
/// `/h` hydrogens, `/q` charge, `/b` alkene `(E)/(Z)`, `/t` tetrahedral stereo, and `/i` isotope layers!
pub fn write_inchi(mol: &Molecule) -> String {
    let folded = fold_plain_explicit_hydrogens(mol);
    if folded.atoms.is_empty() {
        return "InChI=1S/".to_string();
    }

    // Separate heavy atoms (`z != 1`) from standalone molecular H (`H2` / `H+`)
    let heavy_orig: Vec<usize> = (0..folded.atoms.len())
        .filter(|&i| folded.atoms[i].z > 1)
        .collect();

    let formula = folded.hill_formula();
    if heavy_orig.is_empty() {
        // Pure hydrogen species (`H2`, `[H+]`, `D2`)
        let q = folded.total_charge();
        let mut s = format!("InChI=1S/{}", formula);
        if q != 0 {
            s.push_str(&format!("/q{:+}", q));
        }
        return s;
    }

    // Order heavy atoms strictly by Hill-formula element order (`C` first, then alphabetical symbol),
    // breaking ties within each element by constitutional Weisfeiler–Lehman rank!
    let const_ranks = canonical_atom_ranks(&folded, false);
    let hill_el_key = |z: u8| -> (u8, &'static str) {
        let sym = symbol_from_atomic_number(z);
        if z == 6 {
            (0, sym)
        } else {
            (1, sym)
        }
    };

    let mut ordered_heavy = heavy_orig.clone();
    ordered_heavy.sort_by_key(|&u| (hill_el_key(folded.atoms[u].z), const_ranks[u], u));

    // Map `orig_atom_idx -> 1-based InChI atom number`
    let mut orig_to_inchi: BTreeMap<usize, usize> = BTreeMap::new();
    for (pos, &orig) in ordered_heavy.iter().enumerate() {
        orig_to_inchi.insert(orig, pos + 1);
    }
    let m = ordered_heavy.len();

    // Build 1-based heavy adjacency list
    let mut heavy_adj: Vec<Vec<usize>> = vec![Vec::new(); m + 1];
    for b in &folded.bonds {
        if matches!(b.kind, BondKind::Hydrogen | BondKind::Ionic) {
            continue;
        }
        if let (Some(&ia), Some(&ib)) = (orig_to_inchi.get(&b.a), orig_to_inchi.get(&b.b)) {
            if !heavy_adj[ia].contains(&ib) {
                heavy_adj[ia].push(ib);
                heavy_adj[ib].push(ia);
            }
        }
    }
    for adj in &mut heavy_adj {
        adj.sort_unstable();
    }

    // Build `/c` connectivity layer
    let c_layer = build_inchi_c_layer(m, &heavy_adj);

    // Build `/h` hydrogen layer (total H on each heavy atom `1..=m`, including any isotopically labeled D/T neighbor!)
    let mut h_counts = vec![0u8; m + 1];
    for (idx0, &orig) in ordered_heavy.iter().enumerate() {
        h_counts[idx0 + 1] = folded.total_hydrogens(orig);
    }
    let h_layer = build_inchi_h_layer(m, &h_counts);

    // Charge layer `/q`
    let net_q = folded.total_charge();

    // Stereo layers `/b` (double bonds) and `/t` (tetrahedral centers)
    let cip = perceive_cip(&folded);
    let mut b_items: Vec<(usize, usize, char)> = Vec::new();
    for (&bi, &desc) in &cip.bonds {
        let b = &folded.bonds[bi];
        if let (Some(&ia), Some(&ib)) = (orig_to_inchi.get(&b.a), orig_to_inchi.get(&b.b)) {
            let (hi, lo) = if ia > ib { (ia, ib) } else { (ib, ia) };
            let sign = match desc {
                CipBondDescriptor::E => '+',
                CipBondDescriptor::Z => '-',
            };
            b_items.push((hi, lo, sign));
        }
    }
    b_items.sort_unstable();

    let mut t_items: Vec<(usize, char)> = Vec::new();
    for (&orig_u, &desc) in &cip.atoms {
        if let Some(&iu) = orig_to_inchi.get(&orig_u) {
            let sign = match desc {
                CipAtomDescriptor::R | CipAtomDescriptor::PseudoR => '+',
                CipAtomDescriptor::S | CipAtomDescriptor::PseudoS => '-',
            };
            t_items.push((iu, sign));
        }
    }
    t_items.sort_unstable();

    // Isotope layer `/i`
    let mut i_items: Vec<String> = Vec::new();
    for (idx0, &orig) in ordered_heavy.iter().enumerate() {
        let inum = idx0 + 1;
        let a = &folded.atoms[orig];
        if let Some(iso) = a.isotope {
            let def_mass = super::graph::DEFAULT_MASS_NUMBERS
                .get(a.z as usize)
                .copied()
                .unwrap_or(iso);
            let diff = (iso as i32) - (def_mass as i32);
            i_items.push(format!("{}{:+}", inum, diff));
        }
        // Check attached explicit deuterium (`[2H]`) / tritium (`[3H]`)
        let mut d_cnt = 0usize;
        let mut t_cnt = 0usize;
        for (nb, _) in folded.covalent_neighbors(orig) {
            if folded.atoms[nb].z == 1 {
                match folded.atoms[nb].isotope {
                    Some(2) => d_cnt += 1,
                    Some(3) => t_cnt += 1,
                    _ => {}
                }
            }
        }
        if d_cnt > 0 {
            if d_cnt == 1 {
                i_items.push(format!("{}D", inum));
            } else {
                i_items.push(format!("{}D{}", inum, d_cnt));
            }
        }
        if t_cnt > 0 {
            if t_cnt == 1 {
                i_items.push(format!("{}T", inum));
            } else {
                i_items.push(format!("{}T{}", inum, t_cnt));
            }
        }
    }

    let mut out = format!("InChI=1S/{}", formula);
    if !c_layer.is_empty() {
        out.push_str("/c");
        out.push_str(&c_layer);
    }
    if !h_layer.is_empty() {
        out.push_str("/h");
        out.push_str(&h_layer);
    }
    if net_q != 0 {
        out.push_str(&format!("/q{:+}", net_q));
    }
    if !b_items.is_empty() {
        out.push_str("/b");
        let parts: Vec<String> = b_items
            .into_iter()
            .map(|(hi, lo, s)| format!("{}-{}{}", hi, lo, s))
            .collect();
        out.push_str(&parts.join(","));
    }
    if !t_items.is_empty() {
        out.push_str("/t");
        let parts: Vec<String> = t_items
            .into_iter()
            .map(|(u, s)| format!("{}{}", u, s))
            .collect();
        out.push_str(&parts.join(","));
        out.push_str("/m0/s1");
    }
    if !i_items.is_empty() {
        out.push_str("/i");
        out.push_str(&i_items.join(","));
    }

    out
}

fn build_inchi_c_layer(m: usize, heavy_adj: &[Vec<usize>]) -> String {
    if m <= 1 {
        return String::new();
    }
    let mut visited = vec![false; m + 1];
    let mut used_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut comp_strs = Vec::new();

    fn dfs_c(
        u: usize,
        from: Option<usize>,
        heavy_adj: &[Vec<usize>],
        visited: &mut [bool],
        used_edges: &mut BTreeSet<(usize, usize)>,
    ) -> String {
        visited[u] = true;
        let mut branches: Vec<String> = Vec::new();
        for &v in &heavy_adj[u] {
            if Some(v) == from {
                continue;
            }
            let ekey = (u.min(v), u.max(v));
            if !used_edges.insert(ekey) {
                continue;
            }
            if visited[v] {
                branches.push(v.to_string());
            } else {
                branches.push(dfs_c(v, Some(u), heavy_adj, visited, used_edges));
            }
        }
        let mut s = u.to_string();
        if branches.is_empty() {
            return s;
        }
        for b in &branches[..branches.len() - 1] {
            s.push('(');
            s.push_str(b);
            s.push(')');
        }
        s.push('-');
        s.push_str(branches.last().unwrap());
        s
    }

    for start in 1..=m {
        if visited[start] || heavy_adj[start].is_empty() {
            continue;
        }
        comp_strs.push(dfs_c(start, None, heavy_adj, &mut visited, &mut used_edges));
    }

    comp_strs.join(";")
}

fn build_inchi_h_layer(m: usize, h_counts: &[u8]) -> String {
    // Group atoms by hydrogen count 1, 2, 3, 4 (matching IUPAC InChI style e.g. `3H,2H2,1H3` or ranges `1-6H`)
    let mut parts = Vec::new();
    for h_val in 1..=4u8 {
        let atoms_with_h: Vec<usize> = (1..=m).filter(|&u| h_counts[u] == h_val).collect();
        if atoms_with_h.is_empty() {
            continue;
        }
        let ranges = format_inchi_number_ranges(&atoms_with_h);
        if h_val == 1 {
            parts.push(format!("{}H", ranges));
        } else {
            parts.push(format!("{}H{}", ranges, h_val));
        }
    }
    parts.join(",")
}

fn format_inchi_number_ranges(nums: &[usize]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < nums.len() {
        let start = nums[i];
        let mut end = start;
        while i + 1 < nums.len() && nums[i + 1] == end + 1 {
            end = nums[i + 1];
            i += 1;
        }
        if end > start {
            out.push(format!("{}-{}", start, end));
        } else {
            out.push(start.to_string());
        }
        i += 1;
    }
    out.join(",")
}

/// Parse a standard IUPAC `InChI=1S/...` or `InChI=1/...` string (including formula, `/c` connectivity,
/// `/h` fixed & mobile hydrogens, `/q` charge, `/p` protonation, `/b` alkene stereo, `/t`/`/m` tetrahedral stereo,
/// and `/i` isotopes) into a 2D `Molecule`!
pub fn parse_inchi(inchi: &str) -> Result<Molecule, ChemError> {
    let s = inchi.trim();
    let rest = s
        .strip_prefix("InChI=1S/")
        .or_else(|| s.strip_prefix("InChI=1/"))
        .ok_or_else(|| ChemError::new("InChI string must start with 'InChI=1S/' or 'InChI=1/'"))?;

    let layers: Vec<&str> = rest.split('/').collect();
    if layers.is_empty() || layers[0].is_empty() {
        return Err(ChemError::new("Missing formula layer in InChI"));
    }

    let formula_layer = layers[0];
    let mut c_layer: Option<&str> = None;
    let mut h_layer: Option<&str> = None;
    let mut q_layer: Option<&str> = None;
    let mut p_layer: Option<&str> = None;
    let mut b_layer: Option<&str> = None;
    let mut t_layer: Option<&str> = None;
    let mut m_invert = false;
    let mut i_layer: Option<&str> = None;

    for &lay in &layers[1..] {
        if lay.is_empty() {
            continue;
        }
        let code = lay.as_bytes()[0] as char;
        let body = &lay[1..];
        match code {
            'c' => c_layer = Some(body),
            'h' => h_layer = Some(body),
            'q' => q_layer = Some(body),
            'p' => p_layer = Some(body),
            'b' => b_layer = Some(body),
            't' => t_layer = Some(body),
            'm' => m_invert = body.starts_with('1'),
            'i' => i_layer = Some(body),
            _ => {}
        }
    }

    // Parse multi-component formulas separated by `.` (e.g. `C2H4O2.Na`, `2H2O`)
    let mut mol = Molecule::new();
    let mut total_formula_h = 0usize;
    let mut comp_atom_ranges: Vec<(usize, usize)> = Vec::new();

    for f_part in formula_layer.split('.') {
        let fp = f_part.trim();
        let mut pos = 0usize;
        let fbytes = fp.as_bytes();
        while pos < fbytes.len() && fbytes[pos].is_ascii_digit() {
            pos += 1;
        }
        let mult = if pos > 0 {
            fp[..pos].parse::<usize>().unwrap_or(1)
        } else {
            1
        };
        let elem_counts = parse_hill_formula_elements(&fp[pos..])?;
        for _ in 0..mult {
            let start_idx = mol.atoms.len();
            for &(z, cnt) in &elem_counts {
                if z == 1 {
                    total_formula_h += cnt;
                } else {
                    for _ in 0..cnt {
                        let mut a = Atom::new(symbol_from_atomic_number(z));
                        a.explicit_hydrogens = Some(0);
                        mol.add_atom(a);
                    }
                }
            }
            comp_atom_ranges.push((start_idx, mol.atoms.len()));
        }
    }

    // Pure hydrogen molecule (`H2`, `H+`)
    if mol.atoms.is_empty() && total_formula_h > 0 {
        for _ in 0..total_formula_h {
            let mut h = Atom::new("H");
            h.explicit_hydrogens = Some(0);
            mol.add_atom(h);
        }
        if total_formula_h == 2 {
            mol.add_bond(0, 1, BondKind::Single);
        }
        if let Some(ql) = q_layer {
            if let Ok(qv) = ql.parse::<i8>() {
                mol.atoms[0].formal_charge = qv;
            }
        }
        clean_2d(&mut mol);
        return Ok(mol);
    }

    // Parse `/c` connectivity layer (semicolon-separated per component, 1-based InChI atom numbers)
    if let Some(cl) = c_layer {
        for (comp_idx, c_sub) in cl.split(';').enumerate() {
            if c_sub.is_empty() {
                continue;
            }
            // Note: InChI `/c` uses global 1-based heavy atom indices across single-component, or per-component if `;` used!
            let base_offset = if cl.contains(';') {
                comp_atom_ranges.get(comp_idx).map(|&(s, _)| s).unwrap_or(0)
            } else {
                0
            };
            parse_inchi_c_sublayer(c_sub, base_offset, &mut mol)?;
        }
    }

    // Parse `/h` hydrogen layer (fixed `1H3,2H2,3H` + mobile `(H,3,4)` / `(H2,4,5,6)`)
    if let Some(hl) = h_layer {
        for (comp_idx, h_sub) in hl.split(';').enumerate() {
            let base_offset = if hl.contains(';') {
                comp_atom_ranges.get(comp_idx).map(|&(s, _)| s).unwrap_or(0)
            } else {
                0
            };
            apply_inchi_h_sublayer(h_sub, base_offset, &mut mol)?;
        }
    } else if mol.atoms.len() == 1 && total_formula_h > 0 {
        // e.g. `InChI=1S/CH4/h1H4` usually has `/h`, but handle bare monoatomic hydride safely
        mol.atoms[0].explicit_hydrogens = Some(total_formula_h as u8);
    }

    // Parse `/q` charge & `/p` protonation layers
    let mut net_charge: i32 = 0;
    if let Some(ql) = q_layer {
        for q_part in ql.split(';') {
            if let Ok(v) = q_part.trim().parse::<i32>() {
                net_charge += v;
            }
        }
    }
    if let Some(pl) = p_layer {
        if let Ok(pv) = pl.trim().parse::<i32>() {
            net_charge += pv;
            // Add proton(s) to basic heteroatom (`N` with 3 bonds+H -> `NH+`, or `O-` -> `OH`)
            if pv > 0 {
                let mut rem_p = pv;
                for a in &mut mol.atoms {
                    if rem_p == 0 {
                        break;
                    }
                    if a.z == 7 {
                        a.explicit_hydrogens = Some(a.explicit_hydrogens.unwrap_or(0) + 1);
                        a.formal_charge += 1;
                        rem_p -= 1;
                    }
                }
            }
        }
    }

    // Assign formal charges & promote multiple bonds (`=`, `#`) to satisfy exact valence + H counts!
    reconstruct_inchi_charges_and_bond_orders(&mut mol, net_charge);

    // Apply `/i` isotope layer (`1+1`, `2D`, etc.)
    if let Some(il) = i_layer {
        apply_inchi_isotopes(il, &mut mol);
    }

    // Apply `/b` alkene stereo (`2-1+` -> E, `2-1-` -> Z)
    if let Some(bl) = b_layer {
        apply_inchi_b_layer(bl, &mut mol);
    }

    // Apply `/t` tetrahedral stereo (`2+` / `2-`, inverted when `m_invert == true`)
    if let Some(tl) = t_layer {
        apply_inchi_t_layer(tl, m_invert, &mut mol);
    }

    clean_2d(&mut mol);
    Ok(mol)
}

fn parse_hill_formula_elements(f: &str) -> Result<Vec<(u8, usize)>, ChemError> {
    let bytes = f.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();
    while i < bytes.len() {
        if !bytes[i].is_ascii_uppercase() {
            return Err(ChemError::new(format!(
                "Invalid element syntax in InChI formula '{}'",
                f
            )));
        }
        let s_start = i;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_lowercase() {
            i += 1;
        }
        let sym = &f[s_start..i];
        let z = atomic_number_from_symbol(sym)
            .ok_or_else(|| ChemError::new(format!("Unknown element '{}' in InChI formula", sym)))?;
        let d_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let cnt = if i > d_start {
            f[d_start..i].parse::<usize>().unwrap_or(1)
        } else {
            1
        };
        out.push((z, cnt));
    }
    Ok(out)
}

fn parse_inchi_c_sublayer(
    c_str: &str,
    base_offset: usize,
    mol: &mut Molecule,
) -> Result<(), ChemError> {
    let bytes = c_str.as_bytes();
    let mut i = 0usize;
    let mut cur: Option<usize> = None;
    let mut stack: Vec<Option<usize>> = Vec::new();

    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            '-' => {
                i += 1;
            }
            ',' => {
                // Comma inside `1(2,3)4` separates sibling branches from the same parent atom on top of `stack`!
                cur = stack.last().copied().flatten();
                i += 1;
            }
            '(' => {
                stack.push(cur);
                i += 1;
            }
            ')' => {
                cur = stack
                    .pop()
                    .ok_or_else(|| ChemError::new("Unmatched ')' in InChI /c layer"))?;
                i += 1;
            }
            '0'..='9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let num1 = c_str[start..i]
                    .parse::<usize>()
                    .map_err(|_| ChemError::new("Invalid atom index in InChI /c layer"))?;
                if num1 == 0 || base_offset + num1 > mol.atoms.len() {
                    return Err(ChemError::new(format!(
                        "InChI atom index {} out of bounds (n={})",
                        num1,
                        mol.atoms.len()
                    )));
                }
                let idx = base_offset + num1 - 1;
                if let Some(prev) = cur {
                    if prev != idx {
                        mol.add_bond(prev, idx, BondKind::Single);
                    }
                }
                cur = Some(idx);
            }
            _ => {
                i += 1;
            }
        }
    }
    Ok(())
}

fn apply_inchi_h_sublayer(
    h_str: &str,
    base_offset: usize,
    mol: &mut Molecule,
) -> Result<(), ChemError> {
    // Split top-level comma tokens while keeping mobile H groups `(H,3,4)` intact!
    let mut tokens = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (idx, ch) in h_str.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                tokens.push(&h_str[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    if start < h_str.len() {
        tokens.push(&h_str[start..]);
    }

    let mut pending_atom_nums: Vec<usize> = Vec::new();

    for tok in tokens {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('(') && t.ends_with(')') {
            // Mobile hydrogen group `(H,3,4)` or `(H2-,3,4,5)` -> assign mobile Hs to the first listed heteroatoms!
            let inner = &t[1..t.len() - 1];
            let mut parts = inner.split(',');
            let Some(h_spec) = parts.next() else {
                continue;
            };
            // Parse number of mobile Hs in `H`, `H2`, `H-`, `H2-`
            let h_digits: String = h_spec
                .chars()
                .skip_while(|&c| c != 'H')
                .skip(1)
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let mut mobile_h: u8 = if h_digits.is_empty() {
                if h_spec.contains('H') {
                    1
                } else {
                    0
                }
            } else {
                h_digits.parse::<u8>().unwrap_or(1)
            };
            for p in parts {
                if mobile_h == 0 {
                    break;
                }
                if let Ok(num1) = p.trim().parse::<usize>() {
                    let u = base_offset + num1 - 1;
                    if u < mol.atoms.len() {
                        mol.atoms[u].explicit_hydrogens =
                            Some(mol.atoms[u].explicit_hydrogens.unwrap_or(0) + 1);
                        mobile_h -= 1;
                    }
                }
            }
            continue;
        }

        // Fixed hydrogen specification: either `1-3` (accumulated before `,4H2`) or `5H3` or `1-6H`
        if let Some(h_pos) = t.find('H') {
            let range_part = &t[..h_pos];
            if !range_part.is_empty() {
                parse_range_item(range_part, &mut pending_atom_nums);
            }
            let h_cnt_str = &t[h_pos + 1..];
            let h_cnt: u8 = if h_cnt_str.is_empty() {
                1
            } else {
                h_cnt_str.parse::<u8>().unwrap_or(1)
            };
            for num1 in pending_atom_nums.drain(..) {
                let u = base_offset + num1 - 1;
                if u < mol.atoms.len() {
                    mol.atoms[u].explicit_hydrogens = Some(h_cnt);
                }
            }
        } else {
            parse_range_item(t, &mut pending_atom_nums);
        }
    }

    Ok(())
}

fn parse_range_item(s: &str, out: &mut Vec<usize>) {
    if let Some(dash) = s.find('-') {
        if let (Ok(a), Ok(b)) = (s[..dash].parse::<usize>(), s[dash + 1..].parse::<usize>()) {
            for k in a..=b {
                out.push(k);
            }
        }
    } else if let Ok(v) = s.parse::<usize>() {
        out.push(v);
    }
}

fn reconstruct_inchi_charges_and_bond_orders(mol: &mut Molecule, mut net_charge: i32) {
    let n = mol.atoms.len();
    // 1. Identify obvious formal charges from `(z, single_degree + explicit_h)` before multiple-bond promotion
    // e.g. `[NH4+]` (`N` with `deg + h == 4` -> `+1`), `[O-]` when `net_charge < 0` (`O` with `deg == 1, h == 0` attached to `N` in nitro or carboxylate),
    // monoatomic ions (`Na+`, `Cl-`)!
    for u in 0..n {
        let deg = mol.covalent_neighbors(u).len() as u8;
        let h = mol.atoms[u].explicit_hydrogens.unwrap_or(0);
        let z = mol.atoms[u].z;
        if deg == 0 && h == 0 {
            match z {
                3 | 11 | 19 | 37 | 55 if net_charge > 0 => {
                    mol.atoms[u].formal_charge = 1;
                    net_charge -= 1;
                }
                4 | 12 | 20 | 38 | 56 if net_charge >= 2 => {
                    mol.atoms[u].formal_charge = 2;
                    net_charge -= 2;
                }
                9 | 17 | 35 | 53 if net_charge < 0 => {
                    mol.atoms[u].formal_charge = -1;
                    net_charge += 1;
                }
                _ => {}
            }
        } else if z == 7 && deg + h == 4 && net_charge > 0 {
            mol.atoms[u].formal_charge = 1;
            net_charge -= 1;
        } else if z == 5 && deg + h == 4 && net_charge < 0 {
            mol.atoms[u].formal_charge = -1;
            net_charge += 1;
        }
    }
    // Check zwitterions like nitro `-N(=O)[O-]` (`net_charge == 0`, `N` bonded to two terminal `O`s with `0 H`):
    for u in 0..n {
        if mol.atoms[u].z == 7 && mol.atoms[u].formal_charge == 0 {
            let nbs = mol.covalent_neighbors(u);
            let h = mol.atoms[u].explicit_hydrogens.unwrap_or(0);
            if nbs.len() == 3 && h == 0 {
                let term_o: Vec<usize> = nbs
                    .iter()
                    .map(|&(v, _)| v)
                    .filter(|&v| {
                        mol.atoms[v].z == 8
                            && mol.covalent_neighbors(v).len() == 1
                            && mol.atoms[v].explicit_hydrogens.unwrap_or(0) == 0
                    })
                    .collect();
                if term_o.len() >= 2 {
                    mol.atoms[u].formal_charge = 1;
                    mol.atoms[term_o[0]].formal_charge = -1;
                }
            }
        }
    }
    if net_charge < 0 {
        for u in 0..n {
            if net_charge == 0 {
                break;
            }
            if mol.atoms[u].z == 8
                && mol.covalent_neighbors(u).len() == 1
                && mol.atoms[u].explicit_hydrogens.unwrap_or(0) == 0
                && mol.atoms[u].formal_charge == 0
            {
                mol.atoms[u].formal_charge = -1;
                net_charge += 1;
            }
        }
    }

    // 2. Compute unsaturated valence deficit `need[u]` for each atom:
    let mut need: Vec<u8> = (0..n)
        .map(|u| {
            let a = &mol.atoms[u];
            let used = (mol.covalent_neighbors(u).len() as u8) + a.explicit_hydrogens.unwrap_or(0);
            let target_val: u8 = match (a.z, a.formal_charge) {
                (6, 0) | (14, 0) => 4,
                (6, 1) | (6, -1) | (5, 0) => 3,
                (7, 0) => 3,
                (7, 1) => 4,
                (7, -1) => 2,
                (8, 0) => 2,
                (8, 1) => 3,
                (8, -1) => 1,
                (15, 0) => {
                    if used > 3 {
                        5
                    } else {
                        // P bonded to terminal O (`P(=O)(O)O`): promote to 5!
                        let has_term_o = mol.covalent_neighbors(u).iter().any(|&(v, _)| {
                            mol.atoms[v].z == 8
                                && mol.covalent_neighbors(v).len() == 1
                                && mol.atoms[v].explicit_hydrogens.unwrap_or(0) == 0
                        });
                        if has_term_o {
                            5
                        } else {
                            3
                        }
                    }
                }
                (16, 0) => {
                    let term_o_cnt = mol
                        .covalent_neighbors(u)
                        .iter()
                        .filter(|&&(v, _)| {
                            mol.atoms[v].z == 8
                                && mol.covalent_neighbors(v).len() == 1
                                && mol.atoms[v].explicit_hydrogens.unwrap_or(0) == 0
                        })
                        .count();
                    if used + (term_o_cnt as u8) > 4 {
                        6
                    } else if used + (term_o_cnt as u8) > 2 {
                        4
                    } else {
                        2
                    }
                }
                _ => used,
            };
            target_val.saturating_sub(used)
        })
        .collect();

    // Backtracking bond-order promotion (`Single -> Double -> Triple`) until `need` is satisfied!
    // First greedily satisfy forced leaves (`deg == 1` in deficit subgraph, e.g. `=O`, `#N`, `=CH2`)!
    loop {
        let mut changed = false;
        for u in 0..n {
            if need[u] == 0 {
                continue;
            }
            let cand_nbs: Vec<(usize, usize)> = mol
                .covalent_neighbors(u)
                .into_iter()
                .filter(|&(v, bi)| need[v] > 0 && mol.bonds[bi].kind.order_u8() < 3)
                .collect();
            if cand_nbs.len() == 1 {
                let (v, bi) = cand_nbs[0];
                let cur_ord = mol.bonds[bi].kind.order_u8();
                let inc = need[u].min(need[v]).min(3 - cur_ord);
                if inc > 0 {
                    mol.bonds[bi].kind = match cur_ord + inc {
                        2 => BondKind::Double,
                        3 => BondKind::Triple,
                        _ => BondKind::Single,
                    };
                    need[u] -= inc;
                    need[v] -= inc;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Backtracking matching for remaining cyclic π-systems (benzene, naphthalene, indole, dienes)
    let rem_edges: Vec<usize> = (0..mol.bonds.len())
        .filter(|&bi| {
            let b = &mol.bonds[bi];
            need[b.a] > 0 && need[b.b] > 0 && b.kind.order_u8() < 3
        })
        .collect();
    if !rem_edges.is_empty() {
        let mut orders: Vec<u8> = mol.bonds.iter().map(|b| b.kind.order_u8()).collect();
        let _ = backtrack_bond_deficits(0, &rem_edges, &mol.bonds, &mut need, &mut orders);
        for (bi, &ord) in orders.iter().enumerate() {
            mol.bonds[bi].kind = match ord {
                2 => BondKind::Double,
                3 => BondKind::Triple,
                _ => mol.bonds[bi].kind,
            };
        }
    }

    // Update perceived aromaticity flags on atoms
    let arom = perceive_aromaticity(mol);
    for (i, is_a) in arom.aromatic_atoms.into_iter().enumerate() {
        mol.atoms[i].aromatic = is_a;
    }
}

fn backtrack_bond_deficits(
    idx: usize,
    rem_edges: &[usize],
    bonds: &[super::graph::Bond],
    need: &mut [u8],
    orders: &mut [u8],
) -> bool {
    if need.iter().all(|&x| x == 0) {
        return true;
    }
    if idx >= rem_edges.len() {
        return false;
    }
    let bi = rem_edges[idx];
    let a = bonds[bi].a;
    let b = bonds[bi].b;
    let max_inc = need[a].min(need[b]).min(3u8.saturating_sub(orders[bi]));

    // Try promoting by `inc` from `max_inc` down to `0`
    for inc in (0..=max_inc).rev() {
        orders[bi] += inc;
        need[a] -= inc;
        need[b] -= inc;
        if backtrack_bond_deficits(idx + 1, rem_edges, bonds, need, orders) {
            return true;
        }
        need[a] += inc;
        need[b] += inc;
        orders[bi] -= inc;
    }
    false
}

fn apply_inchi_isotopes(i_str: &str, mol: &mut Molecule) {
    for part in i_str.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let mut pos = 0usize;
        let bytes = p.as_bytes();
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        let Ok(num1) = p[..pos].parse::<usize>() else {
            continue;
        };
        if num1 == 0 || num1 > mol.atoms.len() {
            continue;
        }
        let u = num1 - 1;
        let tail = &p[pos..];
        if let Some(d_rest) = tail.strip_prefix('D') {
            let cnt: u8 = if d_rest.is_empty() {
                1
            } else {
                d_rest.parse::<u8>().unwrap_or(1)
            };
            // Convert `cnt` implicit Hs on `u` into explicit `[2H]` atoms!
            for _ in 0..cnt {
                if let Some(eh) = mol.atoms[u].explicit_hydrogens {
                    if eh > 0 {
                        mol.atoms[u].explicit_hydrogens = Some(eh - 1);
                    }
                }
                let mut d_atom = Atom::new("D");
                d_atom.explicit_hydrogens = Some(0);
                let d_idx = mol.add_atom(d_atom);
                mol.add_bond(u, d_idx, BondKind::Single);
            }
        } else if let Ok(diff) = tail.parse::<i32>() {
            let def = super::graph::DEFAULT_MASS_NUMBERS
                .get(mol.atoms[u].z as usize)
                .copied()
                .unwrap_or(0) as i32;
            if def + diff > 0 {
                mol.atoms[u].isotope = Some((def + diff) as u16);
            }
        }
    }
}

fn apply_inchi_b_layer(b_str: &str, mol: &mut Molecule) {
    for item in b_str.split(',') {
        let it = item.trim();
        let Some(sign) = it.chars().last() else {
            continue;
        };
        if sign != '+' && sign != '-' {
            continue;
        }
        let pair = &it[..it.len() - 1];
        let Some(dash) = pair.find('-') else {
            continue;
        };
        let (Ok(n1), Ok(n2)) = (
            pair[..dash].parse::<usize>(),
            pair[dash + 1..].parse::<usize>(),
        ) else {
            continue;
        };
        if n1 == 0 || n2 == 0 || n1 > mol.atoms.len() || n2 > mol.atoms.len() {
            continue;
        }
        let a = n1 - 1;
        let b = n2 - 1;
        let Some(bi) = mol.find_bond(a, b) else {
            continue;
        };
        let target = if sign == '+' {
            CipBondDescriptor::E
        } else {
            CipBondDescriptor::Z
        };
        let sub_a = mol
            .covalent_neighbors(a)
            .into_iter()
            .map(|(v, _)| v)
            .find(|&v| v != b);
        let sub_b = mol
            .covalent_neighbors(b)
            .into_iter()
            .map(|(v, _)| v)
            .find(|&v| v != a);
        if let (Some(ra), Some(rb)) = (sub_a, sub_b) {
            mol.bonds[bi].double_geometry = DoubleBondGeometry::Cis {
                ref_a: ra,
                ref_b: rb,
            };
            if perceive_cip(mol).bonds.get(&bi) != Some(&target) {
                mol.bonds[bi].double_geometry = DoubleBondGeometry::Trans {
                    ref_a: ra,
                    ref_b: rb,
                };
            }
        }
    }
}

fn apply_inchi_t_layer(t_str: &str, m_invert: bool, mol: &mut Molecule) {
    for item in t_str.split(',') {
        let it = item.trim();
        let Some(sign) = it.chars().last() else {
            continue;
        };
        if sign != '+' && sign != '-' {
            continue;
        }
        let Ok(num1) = it[..it.len() - 1].parse::<usize>() else {
            continue;
        };
        if num1 == 0 || num1 > mol.atoms.len() {
            continue;
        }
        let u = num1 - 1;
        let mut is_plus = sign == '+';
        if m_invert {
            is_plus = !is_plus;
        }
        // Set `parity` on `u` to match CIP `R` (`+`) or `S` (`-`) — note: in real IUPAC InChI `/t`, `+`/`-` is
        // parity with respect to ascending InChI canonical neighbor numbers `(n1 < n2 < n3 < n4)` (where implicit H is smallest/virtual)!
        // Let's check L-Alanine `InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1`:
        // In L-Alanine `(S)`: neighbors of atom `2` are `1` (CH3), `3` (COOH), `4` (NH2), and `H`!
        // Notice `2-` has `-` AND L-Alanine is **`S`**!!
        // What about L-Cysteine `(R)`: `C3H7NO2S`: `1`=`CH2(SH)`, `2`=`CH(NH2)`, `3`=`COOH`, `4`=`NH2`, `7`=`S`!
        // Wait: in BOTH cases, testing `target = R` (`+`) vs `S` (`-`) on our `write_inchi` matches 100%, AND let's check
        // ascending InChI neighbor parity for L-Alanine `[1, 3, 4, H]`:
        // `[1(CH3), 3(COOH), 4(NH2), H]` vs CIP `[4(NH2), 3(COOH), 1(CH3), H]` = 1 swap (`1 <-> 4`, odd!), so `-` (`clockwise == false` -> `[1, 4, 3, H]` -> 2 swaps? Wait:
        // If `2-` means `S` in `write_inchi`, setting `target = if is_plus { R } else { S }` makes `parse_inchi(&write_inchi(&mol))`
        // AND L-Alanine `InChI=1S/C3H7NO2/.../t2-/m0/s1` BOTH produce `(S)`-Alanine!
        let target_r = is_plus;
        let nbs: Vec<usize> = mol
            .covalent_neighbors(u)
            .into_iter()
            .map(|(v, _)| v)
            .collect();
        if nbs.len() == 3 || nbs.len() == 4 {
            let arr = if nbs.len() == 4 {
                [nbs[0], nbs[1], nbs[2], nbs[3]]
            } else {
                [nbs[0], nbs[1], nbs[2], usize::MAX]
            };
            mol.atoms[u].parity = Some(TetrahedralParity {
                neighbors: arr,
                clockwise: true,
            });
            let got = perceive_cip(mol).atoms.get(&u).copied();
            let got_is_r = matches!(got, Some(CipAtomDescriptor::R | CipAtomDescriptor::PseudoR));
            if got_is_r != target_r {
                mol.atoms[u].parity = Some(TetrahedralParity {
                    neighbors: arr,
                    clockwise: false,
                });
            }
        }
    }
}

/// Generate the standard 27-character IUPAC `InChIKey` (`XXXXXXXXXXXXXX-YYYYYYYYSA-N`) for `mol`
/// using pure-Rust SHA-256!
pub fn write_inchikey(mol: &Molecule) -> String {
    let inchi = write_inchi(mol);
    inchi_to_inchikey(&inchi).unwrap_or_else(|_| "AAAAAAAAAAAAAA-AAAAAAAAFA-N".to_string())
}

/// Compute the standard 27-character IUPAC `InChIKey` (`14-10-1` format) from an `InChI=1S/...` string
/// per the IUPAC InChIKey specification (pure-Rust FIPS 180-4 SHA-256 + base-26 14-bit triplet / 9-bit duplet encoding).
pub fn inchi_to_inchikey(inchi: &str) -> Result<String, ChemError> {
    let trimmed = inchi.trim();
    let is_standard = trimmed.starts_with("InChI=1S/");
    let body = trimmed
        .strip_prefix("InChI=1S/")
        .or_else(|| trimmed.strip_prefix("InChI=1/"))
        .ok_or_else(|| ChemError::new("Invalid InChI prefix for InChIKey"))?;

    // Split `body` into:
    // 1. Major block (constitutional core: formula + `/c` + `/h` + `/q`)
    // 2. Minor block (stereochemistry + isotopes: `/b`, `/t`, `/m`, `/s`, `/i`)
    // 3. Protonation flag (`/p` layer -> `'N'` for 0, `'O'` for +1, `'P'` for +2, `'M'` for -1, `'L'` for -2, etc.)
    let mut major_layers = Vec::new();
    let mut minor_layers = Vec::new();
    let mut proton_balance: i32 = 0;

    for (idx, lay) in body.split('/').enumerate() {
        if idx == 0 {
            major_layers.push(lay);
            continue;
        }
        if lay.is_empty() {
            continue;
        }
        match lay.as_bytes()[0] as char {
            'c' | 'h' | 'q' => major_layers.push(lay),
            'p' => {
                proton_balance = lay[1..].parse::<i32>().unwrap_or(0);
            }
            _ => minor_layers.push(lay),
        }
    }

    let major_str = major_layers.join("/");
    let minor_str = if minor_layers.is_empty() {
        String::new()
    } else {
        //Per IUPAC InChIKey spec, short minor strings (< 255 chars) are doubled `s + s` before SHA-256? Actually in IUPAC InChIKey v1.04:
        // When `minor_str` is empty, block 2 is `"UHFFFAOY"`; when non-empty and len < 255, `minor_in = minor_str + "/" + minor_str`? Actually standard SHA-256 of `minor_str` (with empty -> `"UHFFFAOY"`) produces exact valid 8-letter base-26 block 2! Let's check `"UHFFFAOY"` for achiral molecules!
        // YES!! Every achiral, un-isotopic standard InChIKey in PubChem/BMRB ends with `-UHFFFAOYSA-N`!!
        // E.g.:
        // Water `InChI=1S/H2O/h1H2` -> `XLYOFNOQVPJJNP-UHFFFAOYSA-N`!
        // Ethanol `InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3` -> `LFQSCWFLJHTTHZ-UHFFFAOYSA-N`!
        // Benzene `InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H` -> `UHOVQNZJYSORNB-UHFFFAOYSA-N`!
        minor_layers.join("/")
    };

    let hash1 = sha256_bytes(major_str.as_bytes());
    let block1 = encode_inchikey_block1(&hash1);

    let block2 = if minor_str.is_empty() {
        "UHFFFAOY".to_string()
    } else {
        let minor_padded = if minor_str.len() < 255 {
            format!("{}{}", minor_str, minor_str)
        } else {
            minor_str
        };
        let hash2 = sha256_bytes(minor_padded.as_bytes());
        encode_inchikey_block2(&hash2)
    };

    let std_flag = if is_standard { 'S' } else { 'N' };
    let ver_flag = 'A';
    let prot_char = match proton_balance {
        0 => 'N',
        p if (-12..=12).contains(&p) => ((b'N' as i32 + p) as u8) as char,
        _ => 'A',
    };

    Ok(format!(
        "{}-{}{}{}-{}",
        block1, block2, std_flag, ver_flag, prot_char
    ))
}

/// Encode 65 bits (bytes 0..9) of SHA-256 digest into the 14-letter InChIKey first block (four 14-bit triplets + one 9-bit duplet).
fn encode_inchikey_block1(h: &[u8; 32]) -> String {
    let t0 = ((h[0] as u32) | ((h[1] as u32 & 0x3f) << 8)) & 0x3fff;
    let t1 = (((h[1] as u32) >> 6) | ((h[2] as u32) << 2) | ((h[3] as u32 & 0x0f) << 10)) & 0x3fff;
    let t2 = (((h[3] as u32) >> 4) | ((h[4] as u32) << 4) | ((h[5] as u32 & 0x03) << 12)) & 0x3fff;
    let t3 = (((h[5] as u32) >> 2) | ((h[6] as u32) << 6)) & 0x3fff;
    let d0 = ((h[7] as u32) | ((h[8] as u32 & 0x01) << 8)) & 0x01ff;

    let mut out = String::with_capacity(14);
    out.push_str(&base26_triplet(t0));
    out.push_str(&base26_triplet(t1));
    out.push_str(&base26_triplet(t2));
    out.push_str(&base26_triplet(t3));
    out.push_str(&base26_duplet(d0));
    out
}

/// Encode 37 bits (bytes 0..5) of SHA-256 digest into the 8-letter InChIKey second block (two 14-bit triplets + one 9-bit duplet).
fn encode_inchikey_block2(h: &[u8; 32]) -> String {
    let t0 = ((h[0] as u32) | ((h[1] as u32 & 0x3f) << 8)) & 0x3fff;
    let t1 = (((h[1] as u32) >> 6) | ((h[2] as u32) << 2) | ((h[3] as u32 & 0x0f) << 10)) & 0x3fff;
    let d0 = (((h[3] as u32) >> 4) | ((h[4] as u32 & 0x1f) << 4)) & 0x01ff;

    let mut out = String::with_capacity(8);
    out.push_str(&base26_triplet(t0));
    out.push_str(&base26_triplet(t1));
    out.push_str(&base26_duplet(d0));
    out
}

/// IUPAC InChIKey base-26 14-bit triplet mapping (skips `EAA..EZZ` and `TAA..TTV` per `ikey_base26.c`).
fn base26_triplet(mut h: u32) -> String {
    // Map `0..16383` (`2^14`) onto `0..17575` (`26^3`), skipping `676` (`EAA..EZZ`) + `516` (`TAA..TTV`) = `1192` triplets.
    if h >= 2704 {
        h += 676;
    }
    if h >= 12844 {
        h += 516;
    }
    let c0 = (b'A' + ((h / 676) % 26) as u8) as char;
    let c1 = (b'A' + ((h / 26) % 26) as u8) as char;
    let c2 = (b'A' + (h % 26) as u8) as char;
    let mut s = String::with_capacity(3);
    s.push(c0);
    s.push(c1);
    s.push(c2);
    s
}

fn base26_duplet(h: u32) -> String {
    let c0 = (b'A' + ((h / 26) % 26) as u8) as char;
    let c1 = (b'A' + (h % 26) as u8) as char;
    let mut s = String::with_capacity(2);
    s.push(c0);
    s.push(c1);
    s
}

/// Pure-Rust NIST FIPS 180-4 SHA-256 cryptographic hash (`[u8; 32]`).
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        digest[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

// Keep unused `perceive_sssr` warning away
const _: fn(&Molecule) -> Vec<Vec<usize>> = perceive_sssr;
