//! Small-molecule analytical & cheminformatics engine for NMR, MS, stoichiometry, and IUPAC naming.
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules:
//! - Cahn, R. S.; Ingold, C.; Prelog, V. *Angew. Chem. Int. Ed.* **1966**, 5, 385–415 (CIP stereochemistry).
//! - Weininger, D. *J. Chem. Inf. Comput. Sci.* **1988**, 28, 31–36 (SMILES notation).
//! - Favre, H. A.; Powell, W. H. *Nomenclature of Organic Chemistry: IUPAC Recommendations and
//!   Preferred Names 2013* (Blue Book), Royal Society of Chemistry, 2014.

pub mod iupac_naming;
pub mod jcamp_spec1d;
pub mod mass_props;
pub mod nmr_predict;
pub mod stoichiometry;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Atomic stereochemical parity or absolute CIP descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AtomStereoTag {
    #[default]
    None,
    /// SMILES `@` (anticlockwise looking from first neighbor toward central atom).
    TetrahedralCcw,
    /// SMILES `@@` (clockwise looking from first neighbor toward central atom).
    TetrahedralCw,
    /// Explicit Cahn–Ingold–Prelog `(R)` configuration.
    R,
    /// Explicit Cahn–Ingold–Prelog `(S)` configuration.
    S,
}

/// Bond stereochemical wedge/hash or double-bond geometry tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BondStereoTag {
    #[default]
    None,
    /// Solid wedge pointing from atom `a` toward atom `b`.
    Up,
    /// Hashed wedge pointing from atom `a` toward atom `b`.
    Down,
    /// Wavy / unspecified stereobond.
    Either,
    /// Explicit *cis* / `(Z)` double-bond geometry.
    Cis,
    /// Explicit *trans* / `(E)` double-bond geometry.
    Trans,
    /// SMILES directional single bond `/` from `a` to `b`.
    Slash,
    /// SMILES directional single bond `\` from `a` to `b`.
    Backslash,
}

/// Lightweight atom node for small-molecule analytical and cheminformatics perception.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalytAtom {
    pub element: String,
    pub isotope: Option<u16>,
    pub formal_charge: i8,
    /// Optional override for implicit hydrogen count; `None` computes from standard valence.
    pub implicit_h: Option<u8>,
    pub x: f64,
    pub y: f64,
    pub stereo_tag: AtomStereoTag,
    /// Internal neighbor ordering from SMILES parsing (used for exact `@`/`@@` perception).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub smiles_neighbor_order: Vec<usize>,
}

impl AnalytAtom {
    pub fn new(element: &str) -> Self {
        Self {
            element: normalize_element_symbol(element),
            isotope: None,
            formal_charge: 0,
            implicit_h: None,
            x: 0.0,
            y: 0.0,
            stereo_tag: AtomStereoTag::None,
            smiles_neighbor_order: Vec::new(),
        }
    }

    pub fn with_coords(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn with_charge(mut self, charge: i8) -> Self {
        self.formal_charge = charge;
        self
    }

    pub fn with_isotope(mut self, isotope: u16) -> Self {
        self.isotope = Some(isotope);
        self
    }

    pub fn with_implicit_h(mut self, h: u8) -> Self {
        self.implicit_h = Some(h);
        self
    }

    pub fn with_stereo(mut self, stereo: AtomStereoTag) -> Self {
        self.stereo_tag = stereo;
        self
    }
}

/// Lightweight bond edge for small-molecule analytical and cheminformatics perception.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalytBond {
    pub a: usize,
    pub b: usize,
    /// Kekulé bond order (`1` = single, `2` = double, `3` = triple, `0` = dative/coordinate/hapto).
    pub order: u8,
    pub aromatic: bool,
    pub stereo: BondStereoTag,
}

impl AnalytBond {
    pub fn new(a: usize, b: usize, order: u8) -> Self {
        Self {
            a,
            b,
            order,
            aromatic: false,
            stereo: BondStereoTag::None,
        }
    }

    pub fn single(a: usize, b: usize) -> Self {
        Self::new(a, b, 1)
    }

    pub fn double(a: usize, b: usize) -> Self {
        Self::new(a, b, 2)
    }

    pub fn triple(a: usize, b: usize) -> Self {
        Self::new(a, b, 3)
    }

    pub fn aromatic_bond(a: usize, b: usize, kekule_order: u8) -> Self {
        Self {
            a,
            b,
            order: kekule_order,
            aromatic: true,
            stereo: BondStereoTag::None,
        }
    }

    pub fn with_stereo(mut self, stereo: BondStereoTag) -> Self {
        self.stereo = stereo;
        self
    }

    pub fn other(&self, idx: usize) -> Option<usize> {
        if self.a == idx {
            Some(self.b)
        } else if self.b == idx {
            Some(self.a)
        } else {
            None
        }
    }
}

/// Lightweight small-molecule chemical graph designed for zero-friction conversion
/// from `octet_mol::Molecule` or SMILES / IUPAC names.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnalytMol {
    pub atoms: Vec<AnalytAtom>,
    pub bonds: Vec<AnalytBond>,
    pub name: Option<String>,
}

/// Perceived CIP stereochemistry across all tetrahedral centers and stereogenic double bonds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceivedStereo {
    /// Map from atom index to `'R'` or `'S'`.
    pub atom_rs: HashMap<usize, char>,
    /// Map from bond index to `'E'` or `'Z'`.
    pub bond_ez: HashMap<usize, char>,
}

/// Sentinel index inside `smiles_neighbor_order` representing the bracket implicit hydrogen `[C@H]`.
pub const SMILES_BRACKET_H_SENTINEL: usize = usize::MAX;

/// Normalize element symbols (`"cl"` -> `"Cl"`, `"D"` -> `"D"` or `"H"` handled by caller).
pub fn normalize_element_symbol(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "C".to_string();
    }
    let mut chars = trimmed.chars();
    let first = chars.next().unwrap().to_ascii_uppercase();
    let rest: String = chars.map(|c| c.to_ascii_lowercase()).collect();
    format!("{first}{rest}")
}

/// Atomic number (`Z`) for elements `1..=92` (plus `D`/`T` mapped to `1`).
pub fn atomic_number(el: &str) -> u8 {
    match el {
        "H" | "D" | "T" => 1,
        "He" => 2,
        "Li" => 3,
        "Be" => 4,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Ne" => 10,
        "Na" => 11,
        "Mg" => 12,
        "Al" => 13,
        "Si" => 14,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Ar" => 18,
        "K" => 19,
        "Ca" => 20,
        "Sc" => 21,
        "Ti" => 22,
        "V" => 23,
        "Cr" => 24,
        "Mn" => 25,
        "Fe" => 26,
        "Co" => 27,
        "Ni" => 28,
        "Cu" => 29,
        "Zn" => 30,
        "Ga" => 31,
        "Ge" => 32,
        "As" => 33,
        "Se" => 34,
        "Br" => 35,
        "Kr" => 36,
        "Rb" => 37,
        "Sr" => 38,
        "Y" => 39,
        "Zr" => 40,
        "Nb" => 41,
        "Mo" => 42,
        "Tc" => 43,
        "Ru" => 44,
        "Rh" => 45,
        "Pd" => 46,
        "Ag" => 47,
        "Cd" => 48,
        "In" => 49,
        "Sn" => 50,
        "Sb" => 51,
        "Te" => 52,
        "I" => 53,
        "Xe" => 54,
        "Cs" => 55,
        "Ba" => 56,
        "La" => 57,
        "Ce" => 58,
        "Pr" => 59,
        "Nd" => 60,
        "Sm" => 62,
        "Eu" => 63,
        "Gd" => 64,
        "Tb" => 65,
        "Dy" => 66,
        "Ho" => 67,
        "Er" => 68,
        "Tm" => 69,
        "Yb" => 70,
        "Lu" => 71,
        "Hf" => 72,
        "Ta" => 73,
        "W" => 74,
        "Re" => 75,
        "Os" => 76,
        "Ir" => 77,
        "Pt" => 78,
        "Au" => 79,
        "Hg" => 80,
        "Tl" => 81,
        "Pb" => 82,
        "Bi" => 83,
        "Th" => 90,
        "Pa" => 91,
        "U" => 92,
        _ => 0,
    }
}

/// True when `el` is a metallic element (alkali, alkaline-earth, transition, lanthanide, actinide, post-transition).
pub fn is_metal_element(el: &str) -> bool {
    matches!(
        el,
        "Li" | "Be"
            | "Na"
            | "Mg"
            | "Al"
            | "K"
            | "Ca"
            | "Sc"
            | "Ti"
            | "V"
            | "Cr"
            | "Mn"
            | "Fe"
            | "Co"
            | "Ni"
            | "Cu"
            | "Zn"
            | "Ga"
            | "Rb"
            | "Sr"
            | "Y"
            | "Zr"
            | "Nb"
            | "Mo"
            | "Ru"
            | "Rh"
            | "Pd"
            | "Ag"
            | "Cd"
            | "In"
            | "Sn"
            | "Cs"
            | "Ba"
            | "La"
            | "Ce"
            | "Pr"
            | "Nd"
            | "Sm"
            | "Eu"
            | "Gd"
            | "Tb"
            | "Dy"
            | "Ho"
            | "Er"
            | "Tm"
            | "Yb"
            | "Lu"
            | "Hf"
            | "Ta"
            | "W"
            | "Re"
            | "Os"
            | "Ir"
            | "Pt"
            | "Au"
            | "Hg"
            | "Tl"
            | "Pb"
            | "Bi"
            | "Th"
            | "Pa"
            | "U"
    )
}

/// Standard neutral organic/main-group valences used to infer implicit hydrogens when `implicit_h == None`.
pub fn standard_valences(el: &str, charge: i8) -> &'static [u8] {
    match (el, charge) {
        ("H" | "D" | "T", 0) => &[1],
        ("H" | "D" | "T", _) => &[0],
        ("B", 0) => &[3],
        ("B", -1) => &[4],
        ("C", 0) => &[4],
        ("C", 1 | -1) => &[3],
        ("N", 0) => &[3, 5],
        ("N", 1) => &[4],
        ("N", -1) => &[2],
        ("O", 0) => &[2],
        ("O", 1) => &[3],
        ("O", -1) => &[1],
        ("F" | "Cl" | "Br" | "I", 0) => &[1],
        ("Cl" | "Br" | "I", _) => &[0],
        ("F", _) => &[0],
        ("Si", 0) => &[4],
        ("P", 0) => &[3, 5],
        ("P", 1) => &[4],
        ("S", 0) => &[2, 4, 6],
        ("S", 1) => &[3],
        ("S", -1) => &[1],
        ("Se", 0) => &[2, 4, 6],
        ("As", 0) => &[3, 5],
        _ => &[0],
    }
}

impl AnalytMol {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_atom(&mut self, atom: AnalytAtom) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(atom);
        idx
    }

    pub fn add_bond(&mut self, bond: AnalytBond) -> usize {
        let idx = self.bonds.len();
        self.bonds.push(bond);
        idx
    }

    /// Adjacency list returning `(neighbor_atom_idx, bond_idx)` for each atom.
    pub fn adjacency(&self) -> Vec<Vec<(usize, usize)>> {
        let mut adj = vec![Vec::new(); self.atoms.len()];
        for (bi, b) in self.bonds.iter().enumerate() {
            if b.a < self.atoms.len() && b.b < self.atoms.len() {
                adj[b.a].push((b.b, bi));
                adj[b.b].push((b.a, bi));
            }
        }
        adj
    }

    /// Find the bond index connecting `u` and `v`, if any.
    pub fn bond_between(&self, u: usize, v: usize) -> Option<usize> {
        self.bonds
            .iter()
            .position(|b| (b.a == u && b.b == v) || (b.a == v && b.b == u))
    }

    /// Sum of bond orders around atom `idx`.
    pub fn bond_order_sum(&self, idx: usize) -> u8 {
        let mut s: u8 = 0;
        for b in &self.bonds {
            if b.a == idx || b.b == idx {
                s = s.saturating_add(b.order);
            }
        }
        s
    }

    /// Implicit hydrogen count on atom `idx` (respecting `implicit_h` override when set).
    pub fn implicit_h_count(&self, idx: usize) -> u8 {
        let atom = &self.atoms[idx];
        if let Some(h) = atom.implicit_h {
            return h;
        }
        let bo = self.bond_order_sum(idx);
        for &v in standard_valences(&atom.element, atom.formal_charge) {
            if v >= bo {
                return v - bo;
            }
        }
        0
    }

    /// Count of explicit neutral `1H` neighbors directly attached to `idx`.
    pub fn explicit_protium_neighbors(&self, idx: usize) -> usize {
        self.bonds
            .iter()
            .filter_map(|b| b.other(idx))
            .filter(|&nb| {
                let a = &self.atoms[nb];
                a.element == "H" && a.isotope.unwrap_or(1) == 1 && a.formal_charge == 0
            })
            .count()
    }

    /// Total attached `1H` count (implicit + explicit neutral `1H` leaves) on atom `idx`.
    pub fn total_h_count(&self, idx: usize) -> u8 {
        self.implicit_h_count(idx)
            .saturating_add(self.explicit_protium_neighbors(idx) as u8)
    }

    /// Smallest Set of Smallest Rings (SSSR / fundamental cycle basis reduced to chordless or smallest rings),
    /// adapted from `rings.ts` in Octet-Draw.
    pub fn sssr_rings(&self) -> Vec<Vec<usize>> {
        let n = self.atoms.len();
        if n == 0 {
            return Vec::new();
        }
        let adj = self.adjacency();
        let mut rings: Vec<Vec<usize>> = Vec::new();
        let mut seen_sets: Vec<HashSet<usize>> = Vec::new();

        // For each bond (u, v), find the shortest path between u and v not using that bond.
        for (bi, bond) in self.bonds.iter().enumerate() {
            if bond.order == 0 || bond.a >= n || bond.b >= n {
                continue;
            }
            let u = bond.a;
            let v = bond.b;
            let mut dist = vec![usize::MAX; n];
            let mut parent = vec![usize::MAX; n];
            let mut q = VecDeque::new();
            dist[u] = 0;
            q.push_back(u);
            while let Some(cur) = q.pop_front() {
                if cur == v {
                    break;
                }
                if dist[cur] >= 14 {
                    continue;
                }
                for &(nxt, e_idx) in &adj[cur] {
                    if e_idx == bi || self.bonds[e_idx].order == 0 {
                        continue;
                    }
                    if dist[nxt] == usize::MAX {
                        dist[nxt] = dist[cur] + 1;
                        parent[nxt] = cur;
                        q.push_back(nxt);
                    }
                }
            }
            if dist[v] != usize::MAX {
                let mut cycle = Vec::with_capacity(dist[v] + 1);
                let mut cur = v;
                while cur != usize::MAX {
                    cycle.push(cur);
                    if cur == u {
                        break;
                    }
                    cur = parent[cur];
                }
                if cycle.len() >= 3 {
                    let set: HashSet<usize> = cycle.iter().copied().collect();
                    if !seen_sets.iter().any(|s| s == &set) {
                        seen_sets.push(set);
                        rings.push(cycle);
                    }
                }
            }
        }
        rings.sort_by_key(|r| r.len());
        rings
    }

    /// Hückel `4n + 2` aromaticity perception across small rings and fused ring systems (`rings.ts`).
    pub fn aromatic_atoms(&self) -> Vec<bool> {
        let n = self.atoms.len();
        let mut arom = vec![false; n];
        for b in &self.bonds {
            if b.aromatic && b.a < n && b.b < n {
                arom[b.a] = true;
                arom[b.b] = true;
            }
        }
        let rings = self.sssr_rings();
        if rings.is_empty() {
            return arom;
        }
        let adj = self.adjacency();

        // Check individual SSSR rings and fused pairs for Hückel 4n+2 pi electrons.
        let pi_electrons = |i: usize, ring_set: &HashSet<usize>| -> Option<usize> {
            let a = &self.atoms[i];
            let el = a.element.as_str();
            if !matches!(el, "C" | "N" | "O" | "S" | "Se" | "B" | "P" | "As") {
                return None;
            }
            let h = self.total_h_count(i);
            let mut in_ring_double = 0usize;
            let mut exo_double_hetero = false;
            let mut total_heavy_deg = 0usize;
            for &(nb, bi) in &adj[i] {
                let b = &self.bonds[bi];
                if b.order == 0 {
                    continue;
                }
                if self.atoms[nb].element == "H" && self.atoms[nb].isotope.unwrap_or(1) == 1 {
                    continue;
                }
                total_heavy_deg += 1;
                if ring_set.contains(&nb) {
                    if b.order == 2 || b.aromatic {
                        in_ring_double += 1;
                    }
                } else if b.order == 2 {
                    let nb_el = self.atoms[nb].element.as_str();
                    if matches!(nb_el, "O" | "S" | "N") {
                        exo_double_hetero = true;
                    } else {
                        return None;
                    }
                } else if b.order >= 3 {
                    return None;
                }
            }
            let total_deg = total_heavy_deg + h as usize;
            match (el, a.formal_charge) {
                ("C", 0) => {
                    if in_ring_double >= 1 && total_deg == 3 {
                        Some(1)
                    } else if exo_double_hetero && total_deg == 3 {
                        Some(0)
                    } else {
                        None
                    }
                }
                ("C", -1) => {
                    if in_ring_double == 0 && total_deg <= 3 {
                        Some(2)
                    } else {
                        None
                    }
                }
                ("C", 1) => {
                    if in_ring_double == 0 && total_deg == 3 {
                        Some(0)
                    } else {
                        None
                    }
                }
                ("N" | "P" | "As", 0) => {
                    if in_ring_double >= 1 && total_deg == 2 {
                        Some(1)
                    } else if in_ring_double == 0 && total_deg == 3 {
                        Some(2)
                    } else {
                        None
                    }
                }
                ("N", 1) => {
                    if in_ring_double >= 1 && total_deg == 3 {
                        Some(1)
                    } else {
                        None
                    }
                }
                ("N", -1) => {
                    if in_ring_double == 0 && total_deg == 2 {
                        Some(2)
                    } else {
                        None
                    }
                }
                ("O" | "S" | "Se", 0) => {
                    if in_ring_double == 0 && total_deg == 2 {
                        Some(2)
                    } else {
                        None
                    }
                }
                ("O" | "S", 1) => {
                    if in_ring_double >= 1 && total_deg == 2 {
                        Some(1)
                    } else {
                        None
                    }
                }
                ("B", 0) => {
                    if in_ring_double == 0 && total_deg == 3 {
                        Some(0)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        for r in &rings {
            if r.len() > 10 {
                continue;
            }
            let set: HashSet<usize> = r.iter().copied().collect();
            let mut sum = 0usize;
            let mut ok = true;
            for &v in r {
                if let Some(p) = pi_electrons(v, &set) {
                    sum += p;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok && sum >= 2 && (sum - 2).is_multiple_of(4) {
                for &v in r {
                    arom[v] = true;
                }
            }
        }

        // Propagate bond.aromatic knowledge if fused 10-pi or 14-pi perimeter holds.
        if rings.len() >= 2 {
            for i in 0..rings.len() {
                for j in (i + 1)..rings.len() {
                    let s1: HashSet<usize> = rings[i].iter().copied().collect();
                    let s2: HashSet<usize> = rings[j].iter().copied().collect();
                    let shared: Vec<usize> = s1.intersection(&s2).copied().collect();
                    if shared.len() == 2 {
                        let union_set: HashSet<usize> = s1.union(&s2).copied().collect();
                        if union_set.len() <= 14 {
                            let mut sum = 0usize;
                            let mut ok = true;
                            for &v in &union_set {
                                if let Some(p) = pi_electrons(v, &union_set) {
                                    sum += p;
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok && sum >= 2 && (sum - 2).is_multiple_of(4) {
                                for &v in &union_set {
                                    arom[v] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        arom
    }

    /// Update `bond.aromatic` flags from `aromatic_atoms()` across ring bonds.
    pub fn perceive_aromaticity(&mut self) {
        let arom = self.aromatic_atoms();
        let rings = self.sssr_rings();
        for r in &rings {
            if r.iter().all(|&v| arom[v]) {
                for k in 0..r.len() {
                    let u = r[k];
                    let v = r[(k + 1) % r.len()];
                    if let Some(bi) = self.bond_between(u, v) {
                        self.bonds[bi].aromatic = true;
                    }
                }
            }
        }
    }

    /// Kekulize aromatic subsystems where bonds were parsed from lowercase aromatic SMILES
    /// (assigning alternating single=1 and double=2 bond orders while preserving `bond.aromatic = true`).
    pub fn kekulize(&mut self, arom_atom_flags: &[bool]) -> Result<(), String> {
        let n = self.atoms.len();
        let adj = self.adjacency();

        // Collect aromatic bonds whose Kekulé order needs assignment.
        let mut arom_bonds: Vec<usize> = Vec::new();
        for (bi, b) in self.bonds.iter().enumerate() {
            if b.aromatic {
                arom_bonds.push(bi);
            }
        }
        if arom_bonds.is_empty() {
            return Ok(());
        }

        // Reset aromatic bonds to order 1 first, then determine which aromatic atoms require one pi-double bond.
        for &bi in &arom_bonds {
            self.bonds[bi].order = 1;
        }

        let mut needs_double = vec![false; n];
        let mut can_double = vec![false; n];
        for i in 0..n {
            if !arom_atom_flags.get(i).copied().unwrap_or(false) {
                continue;
            }
            let a = &self.atoms[i];
            let h = a.implicit_h.unwrap_or(0);
            let mut non_arom_bo = 0u8;
            let mut arom_deg = 0u8;
            let mut explicit_h = 0u8;
            for &(nb, bi) in &adj[i] {
                if self.bonds[bi].aromatic {
                    arom_deg += 1;
                } else {
                    non_arom_bo = non_arom_bo.saturating_add(self.bonds[bi].order);
                    if self.atoms[nb].element == "H" {
                        explicit_h += 1;
                    }
                }
            }
            // Current sigma valence including aromatic single bonds + non-aromatic bonds + implicit H
            let sigma_sum = arom_deg + non_arom_bo + h;
            let has_explicit_h_flag = a.implicit_h.is_some() || explicit_h > 0;

            match (a.element.as_str(), a.formal_charge) {
                ("C", 0) => {
                    if has_explicit_h_flag {
                        if sigma_sum == 3 {
                            needs_double[i] = true;
                            can_double[i] = true;
                        }
                    } else if sigma_sum <= 3 {
                        // e.g., bare `c` with 2 or 3 neighbors gets 1 double bond (and 1 implicit H if 2 neighbors)
                        needs_double[i] = true;
                        can_double[i] = true;
                    }
                }
                ("N" | "P" | "As", 0) => {
                    if has_explicit_h_flag {
                        // `[nH]` has sigma_sum == 3 -> pyrrole-like lone pair (no double bond); `[n]` with H=0 and 2 arom bonds -> pyridine-like
                        if sigma_sum == 2 {
                            needs_double[i] = true;
                            can_double[i] = true;
                        }
                    } else if sigma_sum == 2 {
                        needs_double[i] = true;
                        can_double[i] = true;
                    }
                    // If sigma_sum == 3 (e.g. N-methylpyrrole `cn(C)c`), needs_double = false.
                }
                ("N", 1) => {
                    if sigma_sum == 3 {
                        needs_double[i] = true;
                        can_double[i] = true;
                    }
                }
                ("O" | "S" | "Se", 0) => {
                    // furan/thiophene: 2 single bonds, lone pair donates 2 pi electrons
                    needs_double[i] = false;
                }
                ("O" | "S", 1) if sigma_sum == 2 => {
                    needs_double[i] = true;
                    can_double[i] = true;
                }
                _ => {}
            }
        }

        // Backtracking matching over `arom_bonds` for atoms with `needs_double == true`.
        let candidates: Vec<usize> = (0..n).filter(|&i| needs_double[i]).collect();
        let mut matched = vec![false; n];

        fn solve_matching(
            pos: usize,
            candidates: &[usize],
            matched: &mut [bool],
            can_double: &[bool],
            adj: &[Vec<(usize, usize)>],
            bonds: &mut [AnalytBond],
        ) -> bool {
            let mut idx = pos;
            while idx < candidates.len() && matched[candidates[idx]] {
                idx += 1;
            }
            if idx == candidates.len() {
                return true;
            }
            let u = candidates[idx];
            for &(v, bi) in &adj[u] {
                if bonds[bi].aromatic && !matched[v] && can_double[v] {
                    matched[u] = true;
                    matched[v] = true;
                    bonds[bi].order = 2;
                    if solve_matching(idx + 1, candidates, matched, can_double, adj, bonds) {
                        return true;
                    }
                    bonds[bi].order = 1;
                    matched[u] = false;
                    matched[v] = false;
                }
            }
            false
        }

        if !solve_matching(
            0,
            &candidates,
            &mut matched,
            &can_double,
            &adj,
            &mut self.bonds,
        ) {
            return Err("Failed to Kekulize aromatic ring system".to_string());
        }
        Ok(())
    }

    /// Parse a compact SMILES string into an `AnalytMol` with Kekulized bond orders,
    /// aromaticity flags, `@`/`@@` and `/` `\` stereochemistry, and deterministic 2D coordinates.
    pub fn from_smiles(smiles: &str) -> Result<Self, String> {
        let s = smiles.trim();
        if s.is_empty() {
            return Err("Empty SMILES string".to_string());
        }
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0usize;
        let mut mol = AnalytMol::new();
        let mut arom_atoms: Vec<bool> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        let mut prev_atom: Option<usize> = None;
        let mut pending_bond: Option<(u8, bool, BondStereoTag)> = None;
        // Ring closure map: ring_num -> (open_atom_idx, placeholder_slot_in_open_atom_smiles_neighbors, pending_bond_at_open)
        #[allow(clippy::type_complexity)]
        let mut rings: HashMap<u16, (usize, usize, Option<(u8, bool, BondStereoTag)>)> =
            HashMap::new();

        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                break;
            }
            match c {
                '(' => {
                    if let Some(p) = prev_atom {
                        stack.push(p);
                    } else {
                        return Err("Unexpected '(' at start of SMILES".to_string());
                    }
                    i += 1;
                }
                ')' => {
                    let p = stack
                        .pop()
                        .ok_or_else(|| "Unmatched ')' in SMILES".to_string())?;
                    prev_atom = Some(p);
                    pending_bond = None;
                    i += 1;
                }
                '.' => {
                    prev_atom = None;
                    pending_bond = None;
                    i += 1;
                }
                '-' => {
                    pending_bond = Some((1, false, BondStereoTag::None));
                    i += 1;
                }
                '=' => {
                    pending_bond = Some((2, false, BondStereoTag::None));
                    i += 1;
                }
                '#' => {
                    pending_bond = Some((3, false, BondStereoTag::None));
                    i += 1;
                }
                ':' => {
                    pending_bond = Some((1, true, BondStereoTag::None));
                    i += 1;
                }
                '/' => {
                    pending_bond = Some((1, false, BondStereoTag::Slash));
                    i += 1;
                }
                '\\' => {
                    pending_bond = Some((1, false, BondStereoTag::Backslash));
                    i += 1;
                }
                '%' | '0'..='9' => {
                    let rnum: u16 = if c == '%' {
                        if i + 2 >= chars.len() {
                            return Err("Truncated '%nn' ring closure in SMILES".to_string());
                        }
                        let d1 = chars[i + 1]
                            .to_digit(10)
                            .ok_or_else(|| "Invalid '%nn' ring digit".to_string())?;
                        let d2 = chars[i + 2]
                            .to_digit(10)
                            .ok_or_else(|| "Invalid '%nn' ring digit".to_string())?;
                        i += 3;
                        (d1 * 10 + d2) as u16
                    } else {
                        i += 1;
                        c.to_digit(10).unwrap() as u16
                    };
                    let cur = prev_atom
                        .ok_or_else(|| "Ring digit before any atom in SMILES".to_string())?;
                    if let Some((open_atom, slot_idx, open_bond)) = rings.remove(&rnum) {
                        if open_atom == cur {
                            return Err("Self-loop ring closure in SMILES".to_string());
                        }
                        let (order, arom, stereo, dir_from_open) = resolve_ring_bond(
                            open_bond,
                            pending_bond,
                            arom_atoms[open_atom] && arom_atoms[cur],
                        );
                        pending_bond = None;
                        mol.atoms[open_atom].smiles_neighbor_order[slot_idx] = cur;
                        mol.atoms[cur].smiles_neighbor_order.push(open_atom);
                        let (a, b) = if dir_from_open {
                            (open_atom, cur)
                        } else {
                            (cur, open_atom)
                        };
                        mol.add_bond(AnalytBond {
                            a,
                            b,
                            order,
                            aromatic: arom,
                            stereo,
                        });
                    } else {
                        let slot = mol.atoms[cur].smiles_neighbor_order.len();
                        mol.atoms[cur].smiles_neighbor_order.push(usize::MAX - 1);
                        rings.insert(rnum, (cur, slot, pending_bond.take()));
                    }
                }
                '[' => {
                    i += 1;
                    let start = i;
                    while i < chars.len() && chars[i] != ']' {
                        i += 1;
                    }
                    if i >= chars.len() {
                        return Err("Unclosed '[' in SMILES".to_string());
                    }
                    let content: String = chars[start..i].iter().collect();
                    i += 1; // consume ']'
                    let (mut atom, is_arom, has_bracket_h) = parse_bracket_atom(&content)?;
                    let cur_idx = mol.atoms.len();
                    if let Some(p) = prev_atom {
                        atom.smiles_neighbor_order.push(p);
                    }
                    if has_bracket_h && atom.implicit_h == Some(1) {
                        atom.smiles_neighbor_order.push(SMILES_BRACKET_H_SENTINEL);
                    }
                    mol.atoms.push(atom);
                    arom_atoms.push(is_arom);
                    if let Some(p) = prev_atom {
                        mol.atoms[p].smiles_neighbor_order.push(cur_idx);
                        let (order, arom, stereo) = pending_bond.take().unwrap_or_else(|| {
                            if arom_atoms[p] && is_arom {
                                (1, true, BondStereoTag::None)
                            } else {
                                (1, false, BondStereoTag::None)
                            }
                        });
                        mol.add_bond(AnalytBond {
                            a: p,
                            b: cur_idx,
                            order,
                            aromatic: arom,
                            stereo,
                        });
                    }
                    pending_bond = None;
                    prev_atom = Some(cur_idx);
                }
                _ => {
                    let (el, is_arom, consumed) = parse_organic_subset(&chars[i..])?;
                    i += consumed;
                    let mut atom = AnalytAtom::new(&el);
                    if el == "D" {
                        atom.element = "H".to_string();
                        atom.isotope = Some(2);
                    }
                    let cur_idx = mol.atoms.len();
                    if let Some(p) = prev_atom {
                        atom.smiles_neighbor_order.push(p);
                    }
                    mol.atoms.push(atom);
                    arom_atoms.push(is_arom);
                    if let Some(p) = prev_atom {
                        mol.atoms[p].smiles_neighbor_order.push(cur_idx);
                        let (order, arom, stereo) = pending_bond.take().unwrap_or_else(|| {
                            if arom_atoms[p] && is_arom {
                                (1, true, BondStereoTag::None)
                            } else {
                                (1, false, BondStereoTag::None)
                            }
                        });
                        mol.add_bond(AnalytBond {
                            a: p,
                            b: cur_idx,
                            order,
                            aromatic: arom,
                            stereo,
                        });
                    }
                    pending_bond = None;
                    prev_atom = Some(cur_idx);
                }
            }
        }

        if !rings.is_empty() {
            return Err("Unclosed ring number in SMILES".to_string());
        }
        if !stack.is_empty() {
            return Err("Unclosed '(' branch in SMILES".to_string());
        }

        // Normalize hypervalent nitro `-N(=O)=O` if present and Kekulize aromatic rings.
        mol.kekulize(&arom_atoms)?;
        mol.perceive_aromaticity();
        mol.generate_2d_coords_if_missing();
        Ok(mol)
    }

    /// Generate clean deterministic 2D coordinates when all atoms are at `(0, 0)`.
    pub fn generate_2d_coords_if_missing(&mut self) {
        if self.atoms.len() <= 1 {
            return;
        }
        let all_zero = self
            .atoms
            .iter()
            .all(|a| a.x.abs() < 1e-9 && a.y.abs() < 1e-9);
        if !all_zero {
            return;
        }
        let n = self.atoms.len();
        let adj = self.adjacency();
        let rings = self.sssr_rings();
        let mut placed = vec![false; n];
        let bond_len = 1.5;

        let mut comp_offset_x = 0.0;
        for start in 0..n {
            if placed[start] {
                continue;
            }
            // Place any ring containing `start` as a regular polygon first, else place `start` at origin.
            if let Some(r) = rings.iter().find(|r| r.contains(&start)) {
                let k = r.len();
                let radius = bond_len / (2.0 * (std::f64::consts::PI / k as f64).sin());
                for (idx_in_r, &v) in r.iter().enumerate() {
                    let ang = 2.0 * std::f64::consts::PI * (idx_in_r as f64) / (k as f64);
                    self.atoms[v].x = comp_offset_x + radius * ang.cos();
                    self.atoms[v].y = radius * ang.sin();
                    placed[v] = true;
                }
            } else {
                self.atoms[start].x = comp_offset_x;
                self.atoms[start].y = 0.0;
                placed[start] = true;
            }

            let mut q: VecDeque<(usize, f64)> = VecDeque::new();
            for (v, &is_placed) in placed.iter().enumerate().take(n) {
                if is_placed {
                    q.push_back((v, std::f64::consts::FRAC_PI_6)); // 30 deg default zig-zag
                }
            }
            while let Some((u, incoming_ang)) = q.pop_front() {
                // If u has an unplaced ring attached, place that fused/spiro ring first
                for r in &rings {
                    let placed_in_r: Vec<usize> =
                        r.iter().copied().filter(|&x| placed[x]).collect();
                    if !placed_in_r.is_empty() && placed_in_r.len() < r.len() && r.contains(&u) {
                        let k = r.len();
                        let radius = bond_len / (2.0 * (std::f64::consts::PI / k as f64).sin());
                        let anchor = placed_in_r[0];
                        let pos_a = r.iter().position(|&x| x == anchor).unwrap();
                        let base_ang = if placed_in_r.len() >= 2 {
                            let anchor2 = placed_in_r[1];
                            let mx = 0.5 * (self.atoms[anchor].x + self.atoms[anchor2].x);
                            let my = 0.5 * (self.atoms[anchor].y + self.atoms[anchor2].y);
                            let dx = self.atoms[anchor2].x - self.atoms[anchor].x;
                            let dy = self.atoms[anchor2].y - self.atoms[anchor].y;
                            let nx = -dy;
                            let ny = dx;
                            let norm = (nx * nx + ny * ny).sqrt().max(1e-6);
                            let h_dist = (radius * radius - 0.25 * bond_len * bond_len)
                                .max(0.0)
                                .sqrt();
                            let cx = mx + (nx / norm) * h_dist;
                            let cy = my + (ny / norm) * h_dist;
                            let a0 = (self.atoms[anchor].y - cy).atan2(self.atoms[anchor].x - cx);
                            let step = 2.0 * std::f64::consts::PI / (k as f64);
                            let next_v = r[(pos_a + 1) % k];
                            let sign = if next_v == anchor2 { -1.0 } else { 1.0 };
                            for (m, &rv) in r.iter().enumerate() {
                                if !placed[rv] {
                                    let delta = (m as isize - pos_a as isize) as f64;
                                    let ang = a0 + sign * delta * step;
                                    self.atoms[rv].x = cx + radius * ang.cos();
                                    self.atoms[rv].y = cy + radius * ang.sin();
                                    placed[rv] = true;
                                    q.push_back((rv, ang));
                                }
                            }
                            continue;
                        } else {
                            incoming_ang
                        };
                        let cx = self.atoms[anchor].x + radius * base_ang.cos();
                        let cy = self.atoms[anchor].y + radius * base_ang.sin();
                        let a0 = base_ang + std::f64::consts::PI;
                        let step = 2.0 * std::f64::consts::PI / (k as f64);
                        for (m, &rv) in r.iter().enumerate() {
                            if !placed[rv] {
                                let delta = (m as isize - pos_a as isize) as f64;
                                let ang = a0 + delta * step;
                                self.atoms[rv].x = cx + radius * ang.cos();
                                self.atoms[rv].y = cy + radius * ang.sin();
                                placed[rv] = true;
                                q.push_back((rv, ang));
                            }
                        }
                    }
                }

                let unplaced_nbs: Vec<(usize, usize)> = adj[u]
                    .iter()
                    .copied()
                    .filter(|&(v, _)| !placed[v])
                    .collect();
                if unplaced_nbs.is_empty() {
                    continue;
                }
                let placed_nbs: Vec<usize> = adj[u]
                    .iter()
                    .map(|&(v, _)| v)
                    .filter(|&v| placed[v])
                    .collect();
                let base_dir = if placed_nbs.is_empty() {
                    incoming_ang
                } else {
                    let mut sx = 0.0;
                    let mut sy = 0.0;
                    for pv in &placed_nbs {
                        sx += self.atoms[*pv].x - self.atoms[u].x;
                        sy += self.atoms[*pv].y - self.atoms[u].y;
                    }
                    if sx.abs() < 1e-6 && sy.abs() < 1e-6 {
                        incoming_ang + std::f64::consts::FRAC_PI_2
                    } else {
                        (-sy).atan2(-sx)
                    }
                };
                for (slot, &(v, bi)) in unplaced_nbs.iter().enumerate() {
                    let ang = if unplaced_nbs.len() == 1 {
                        if self.bonds[bi].order == 3 {
                            base_dir
                        } else if placed_nbs.len() == 1 {
                            // Alternate +/- 60 deg from incoming bond -> 120 deg zig-zag chain
                            let sign = if (u + v) % 2 == 0 { 1.0 } else { -1.0 };
                            base_dir + sign * (std::f64::consts::PI / 6.0)
                        } else {
                            base_dir
                        }
                    } else if unplaced_nbs.len() == 2 {
                        base_dir
                            + if slot == 0 {
                                std::f64::consts::PI / 3.0
                            } else {
                                -std::f64::consts::PI / 3.0
                            }
                    } else {
                        base_dir
                            + (slot as f64 - (unplaced_nbs.len() as f64 - 1.0) / 2.0)
                                * (std::f64::consts::PI / 2.0)
                    };
                    self.atoms[v].x = self.atoms[u].x + bond_len * ang.cos();
                    self.atoms[v].y = self.atoms[u].y + bond_len * ang.sin();
                    placed[v] = true;
                    q.push_back((v, ang));
                }
            }
            comp_offset_x += 8.0;
        }
    }

    /// Serialize this molecule to a clean, deterministic Kekulé SMILES string.
    pub fn to_smiles(&self) -> String {
        let n = self.atoms.len();
        if n == 0 {
            return String::new();
        }
        let adj = self.adjacency();
        // Fold explicit neutral 1H leaves when writing compact SMILES unless the H is the entire molecule.
        let mut hide_h = vec![false; n];
        for i in 0..n {
            let a = &self.atoms[i];
            if a.element == "H"
                && a.isotope.unwrap_or(1) == 1
                && a.formal_charge == 0
                && adj[i].len() == 1
            {
                let (nb, _) = adj[i][0];
                if self.atoms[nb].element != "H" {
                    hide_h[i] = true;
                }
            }
        }

        let mut visited = vec![false; n];
        let mut tree_edges: HashSet<usize> = HashSet::new();
        let mut back_edges: Vec<usize> = Vec::new();
        let mut components: Vec<usize> = Vec::new();

        fn dfs_span(
            u: usize,
            parent_bond: Option<usize>,
            adj: &[Vec<(usize, usize)>],
            hide_h: &[bool],
            visited: &mut [bool],
            tree_edges: &mut HashSet<usize>,
            back_edges: &mut Vec<usize>,
        ) {
            visited[u] = true;
            for &(v, bi) in &adj[u] {
                if hide_h[v] || Some(bi) == parent_bond {
                    continue;
                }
                if !visited[v] {
                    tree_edges.insert(bi);
                    dfs_span(v, Some(bi), adj, hide_h, visited, tree_edges, back_edges);
                } else if !tree_edges.contains(&bi) && !back_edges.contains(&bi) {
                    back_edges.push(bi);
                }
            }
        }

        for i in 0..n {
            if !hide_h[i] && !visited[i] {
                components.push(i);
                dfs_span(
                    i,
                    None,
                    &adj,
                    &hide_h,
                    &mut visited,
                    &mut tree_edges,
                    &mut back_edges,
                );
            }
        }

        let mut ring_num_for_edge: HashMap<usize, usize> = HashMap::new();
        for (k, &bi) in back_edges.iter().enumerate() {
            ring_num_for_edge.insert(bi, k + 1);
        }

        let mut emitted = vec![false; n];
        let mut opened_rings: HashSet<usize> = HashSet::new();

        #[allow(clippy::too_many_arguments)]
        fn write_sub(
            u: usize,
            incoming_bond: Option<usize>,
            mol: &AnalytMol,
            adj: &[Vec<(usize, usize)>],
            hide_h: &[bool],
            tree_edges: &HashSet<usize>,
            ring_num_for_edge: &HashMap<usize, usize>,
            emitted: &mut [bool],
            opened_rings: &mut HashSet<usize>,
            out: &mut String,
        ) {
            emitted[u] = true;
            if let Some(bi) = incoming_bond {
                out.push_str(bond_smiles_symbol(&mol.bonds[bi], u));
            }
            out.push_str(&format_smiles_atom(mol, u));

            // Ring closures on u
            for &(_, bi) in &adj[u] {
                if let Some(&rnum) = ring_num_for_edge.get(&bi) {
                    if !opened_rings.contains(&bi) {
                        opened_rings.insert(bi);
                        out.push_str(bond_smiles_symbol(
                            &mol.bonds[bi],
                            mol.bonds[bi].other(u).unwrap(),
                        ));
                    }
                    if rnum < 10 {
                        out.push(char::from_digit(rnum as u32, 10).unwrap());
                    } else {
                        out.push_str(&format!("%{rnum:02}"));
                    }
                }
            }

            let children: Vec<(usize, usize)> = adj[u]
                .iter()
                .copied()
                .filter(|&(v, bi)| !hide_h[v] && tree_edges.contains(&bi) && !emitted[v])
                .collect();
            if children.is_empty() {
                return;
            }
            for &(v, bi) in &children[..children.len() - 1] {
                out.push('(');
                write_sub(
                    v,
                    Some(bi),
                    mol,
                    adj,
                    hide_h,
                    tree_edges,
                    ring_num_for_edge,
                    emitted,
                    opened_rings,
                    out,
                );
                out.push(')');
            }
            let (last_v, last_bi) = children[children.len() - 1];
            write_sub(
                last_v,
                Some(last_bi),
                mol,
                adj,
                hide_h,
                tree_edges,
                ring_num_for_edge,
                emitted,
                opened_rings,
                out,
            );
        }

        let mut parts = Vec::with_capacity(components.len());
        for root in components {
            let mut part = String::new();
            write_sub(
                root,
                None,
                self,
                &adj,
                &hide_h,
                &tree_edges,
                &ring_num_for_edge,
                &mut emitted,
                &mut opened_rings,
                &mut part,
            );
            parts.push(part);
        }
        parts.join(".")
    }

    /// Perceive Cahn–Ingold–Prelog `(R)`/`(S)` tetrahedral stereocenters and `(E)`/`(Z)` alkene bonds.
    pub fn perceive_cip_stereo(&self) -> PerceivedStereo {
        let mut out = PerceivedStereo::default();
        let n = self.atoms.len();
        if n == 0 {
            return out;
        }
        let adj = self.adjacency();

        // 1. Tetrahedral (R)/(S) perception
        for i in 0..n {
            let a = &self.atoms[i];
            match a.stereo_tag {
                AtomStereoTag::R => {
                    out.atom_rs.insert(i, 'R');
                    continue;
                }
                AtomStereoTag::S => {
                    out.atom_rs.insert(i, 'S');
                    continue;
                }
                _ => {}
            }
            // Only sp3 / 4-substituent centers can be tetrahedral stereocenters
            if adj[i].iter().any(|&(_, bi)| self.bonds[bi].order != 1) {
                continue;
            }
            let h_implicit = self.implicit_h_count(i);
            let total_subs = adj[i].len() + h_implicit as usize;
            if total_subs != 4 || h_implicit > 1 {
                continue;
            }

            let has_smiles_parity = matches!(
                a.stereo_tag,
                AtomStereoTag::TetrahedralCcw | AtomStereoTag::TetrahedralCw
            );
            let has_wedge = adj[i].iter().any(|&(_, bi)| {
                self.bonds[bi].a == i
                    && matches!(
                        self.bonds[bi].stereo,
                        BondStereoTag::Up | BondStereoTag::Down
                    )
            });
            if !has_smiles_parity && !has_wedge {
                continue;
            }

            // Compute CIP priority signatures of the 4 substituents (smaller tuple = lower priority; larger = higher priority).
            // Substituents are either real neighbor atom indices or `SMILES_BRACKET_H_SENTINEL` for implicit 1H.
            let mut subs: Vec<usize> = adj[i].iter().map(|&(nb, _)| nb).collect();
            if h_implicit == 1 {
                subs.push(SMILES_BRACKET_H_SENTINEL);
            }
            let sigs: Vec<Vec<u32>> = subs
                .iter()
                .map(|&s| {
                    if s == SMILES_BRACKET_H_SENTINEL {
                        vec![1, 1]
                    } else {
                        cip_branch_signature(self, &adj, i, s, 7)
                    }
                })
                .collect();

            // All 4 CIP signatures must be pairwise distinct for a true stereocenter.
            let mut distinct = true;
            for p in 0..4 {
                for q in (p + 1)..4 {
                    if sigs[p] == sigs[q] {
                        distinct = false;
                    }
                }
            }
            if !distinct {
                continue;
            }

            // Order substituents from highest CIP priority (rank 0) to lowest CIP priority (rank 3).
            let mut ranked_indices: Vec<usize> = (0..4).collect();
            ranked_indices.sort_by(|&p, &q| sigs[q].cmp(&sigs[p]));
            let sub_by_rank: [usize; 4] = [
                subs[ranked_indices[0]],
                subs[ranked_indices[1]],
                subs[ranked_indices[2]],
                subs[ranked_indices[3]],
            ];

            // Build 3D vectors v[sub] around center (0,0,0) from either SMILES `@`/`@@` or 2D wedge/hash.
            let coords_3d: Option<HashMap<usize, [f64; 3]>> = if has_smiles_parity {
                let mut order = a.smiles_neighbor_order.clone();
                if order.len() != 4 {
                    order = subs.clone();
                }
                if order.len() == 4 {
                    let mut m = HashMap::with_capacity(4);
                    // Looking FROM order[0] at (0, 0, +1) TOWARD center (0, 0, 0):
                    // `@` (Ccw) places order[1], order[2], order[3] anticlockwise in (x, y) at z = -1/3.
                    // `@@` (Cw) places order[1], order[2], order[3] clockwise in (x, y) at z = -1/3.
                    m.insert(order[0], [0.0, 0.0, 1.0]);
                    let s3 = 3.0_f64.sqrt() * 0.5;
                    let sign = if a.stereo_tag == AtomStereoTag::TetrahedralCcw {
                        1.0
                    } else {
                        -1.0
                    };
                    m.insert(order[1], [1.0, 0.0, -1.0 / 3.0]);
                    m.insert(order[2], [-0.5, sign * s3, -1.0 / 3.0]);
                    m.insert(order[3], [-0.5, -sign * s3, -1.0 / 3.0]);
                    Some(m)
                } else {
                    None
                }
            } else {
                // 2D coordinates + Up/Down wedge from center `i`
                let mut m = HashMap::with_capacity(4);
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_z = 0.0;
                for &(nb, bi) in &adj[i] {
                    let dx = self.atoms[nb].x - a.x;
                    let dy = self.atoms[nb].y - a.y;
                    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
                    let z = if self.bonds[bi].a == i {
                        match self.bonds[bi].stereo {
                            BondStereoTag::Up => 1.0,
                            BondStereoTag::Down => -1.0,
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };
                    let ux = dx / len;
                    let uy = dy / len;
                    sum_x += ux;
                    sum_y += uy;
                    sum_z += z;
                    m.insert(nb, [ux, uy, z]);
                }
                if h_implicit == 1 {
                    let hz = if sum_z > 0.1 {
                        -1.0
                    } else if sum_z < -0.1 {
                        1.0
                    } else {
                        -1.0
                    };
                    m.insert(SMILES_BRACKET_H_SENTINEL, [-sum_x, -sum_y, hz]);
                }
                Some(m)
            };

            if let Some(c3) = coords_3d {
                if let (Some(&v1), Some(&v2), Some(&v3), Some(&v4)) = (
                    c3.get(&sub_by_rank[0]),
                    c3.get(&sub_by_rank[1]),
                    c3.get(&sub_by_rank[2]),
                    c3.get(&sub_by_rank[3]),
                ) {
                    let d1 = [v1[0] - v4[0], v1[1] - v4[1], v1[2] - v4[2]];
                    let d2 = [v2[0] - v4[0], v2[1] - v4[1], v2[2] - v4[2]];
                    let d3 = [v3[0] - v4[0], v3[1] - v4[1], v3[2] - v4[2]];
                    let det = d1[0] * (d2[1] * d3[2] - d2[2] * d3[1])
                        - d1[1] * (d2[0] * d3[2] - d2[2] * d3[0])
                        + d1[2] * (d2[0] * d3[1] - d2[1] * d3[0]);
                    if det > 1e-6 {
                        out.atom_rs.insert(i, 'S');
                    } else if det < -1e-6 {
                        out.atom_rs.insert(i, 'R');
                    }
                }
            }
        }

        // 2. Double-bond (E)/(Z) perception
        let rings = self.sssr_rings();
        for (bi, b) in self.bonds.iter().enumerate() {
            if b.order != 2 || b.aromatic {
                continue;
            }
            match b.stereo {
                BondStereoTag::Cis => {
                    out.bond_ez.insert(bi, 'Z');
                    continue;
                }
                BondStereoTag::Trans => {
                    out.bond_ez.insert(bi, 'E');
                    continue;
                }
                BondStereoTag::Either => continue,
                _ => {}
            }
            // Skip small rings (< 8 members) where E/Z is typically omitted by IUPAC unless explicit
            if rings
                .iter()
                .any(|r| r.len() < 8 && r.contains(&b.a) && r.contains(&b.b))
            {
                continue;
            }
            let u = b.a;
            let v = b.b;
            let u_subs: Vec<(usize, usize)> =
                adj[u].iter().copied().filter(|&(x, _)| x != v).collect();
            let v_subs: Vec<(usize, usize)> =
                adj[v].iter().copied().filter(|&(x, _)| x != u).collect();
            if u_subs.is_empty() || u_subs.len() > 2 || v_subs.is_empty() || v_subs.len() > 2 {
                continue;
            }

            // Pick highest-CIP substituent on u and v; verify the two substituents on each end are non-identical.
            #[allow(clippy::type_complexity)]
            let pick_top = |center: usize,
                            subs: &[(usize, usize)]|
             -> Option<(usize, usize, Option<(usize, usize)>)> {
                if subs.len() == 1 {
                    if self.implicit_h_count(center) == 1 {
                        let s0 = cip_branch_signature(self, &adj, center, subs[0].0, 6);
                        if s0 == vec![1, 1] {
                            return None;
                        }
                        Some((subs[0].0, subs[0].1, None))
                    } else {
                        None
                    }
                } else {
                    let s0 = cip_branch_signature(self, &adj, center, subs[0].0, 6);
                    let s1 = cip_branch_signature(self, &adj, center, subs[1].0, 6);
                    if s0 == s1 {
                        return None;
                    }
                    if s0 > s1 {
                        Some((subs[0].0, subs[0].1, Some(subs[1])))
                    } else {
                        Some((subs[1].0, subs[1].1, Some(subs[0])))
                    }
                }
            };

            let Some((top_u, bond_u, other_u)) = pick_top(u, &u_subs) else {
                continue;
            };
            let Some((top_v, bond_v, other_v)) = pick_top(v, &v_subs) else {
                continue;
            };

            // First check SMILES directional `/` and `\` bonds on u and v
            let dir_on =
                |center: usize, nb_bond: usize, alt: Option<(usize, usize)>| -> Option<i8> {
                    let side = |b_idx: usize| -> Option<i8> {
                        let sb = &self.bonds[b_idx];
                        match sb.stereo {
                            BondStereoTag::Slash => Some(if sb.a == center { 1 } else { -1 }),
                            BondStereoTag::Backslash => Some(if sb.a == center { -1 } else { 1 }),
                            _ => None,
                        }
                    };
                    if let Some(s) = side(nb_bond) {
                        Some(s)
                    } else if let Some((_, alt_bi)) = alt {
                        side(alt_bi).map(|s| -s)
                    } else {
                        None
                    }
                };

            if let (Some(su), Some(sv)) = (dir_on(u, bond_u, other_u), dir_on(v, bond_v, other_v)) {
                out.bond_ez.insert(bi, if su == sv { 'Z' } else { 'E' });
                continue;
            }

            // Fallback: 2D geometric cross-product ONLY if coordinates were explicitly drawn (not auto-generated from plain `=`)
            let has_drawn_2d = self.atoms[u].smiles_neighbor_order.is_empty()
                && ((self.atoms[u].x - self.atoms[v].x).hypot(self.atoms[u].y - self.atoms[v].y)
                    > 1e-4);
            if has_drawn_2d {
                let ax = self.atoms[v].x - self.atoms[u].x;
                let ay = self.atoms[v].y - self.atoms[u].y;
                let ux = self.atoms[top_u].x - self.atoms[u].x;
                let uy = self.atoms[top_u].y - self.atoms[u].y;
                let vx = self.atoms[top_v].x - self.atoms[v].x;
                let vy = self.atoms[top_v].y - self.atoms[v].y;
                let cu = ax * uy - ay * ux;
                let cv = ax * vy - ay * vx;
                if (cu * cv).abs() > 1e-6 {
                    out.bond_ez
                        .insert(bi, if cu * cv > 0.0 { 'Z' } else { 'E' });
                }
            }
        }

        out
    }
}

/// Hierarchical CIP branch priority signature `(Z_root, mass_root, sphere_1_sorted..., sphere_2_sorted...)`.
fn cip_branch_signature(
    mol: &AnalytMol,
    adj: &[Vec<(usize, usize)>],
    from: usize,
    start: usize,
    max_depth: usize,
) -> Vec<u32> {
    let atom_code = |idx: usize| -> u32 {
        let a = &mol.atoms[idx];
        let z = atomic_number(&a.element) as u32;
        let iso = a.isotope.unwrap_or(0) as u32;
        (z << 10) | iso
    };
    let ghost_code = |idx: usize| -> u32 {
        let a = &mol.atoms[idx];
        let z = atomic_number(&a.element) as u32;
        z << 10 // duplicate atom for multiple bond (mass 0 tie-break below real isotope if needed)
    };

    let mut sig = vec![atom_code(start)];
    // Frontier holds `(prev_atom, cur_atom, is_ghost, path_mask)`
    let mut frontier: Vec<(usize, usize, bool, Vec<usize>)> =
        vec![(from, start, false, vec![from, start])];

    for _depth in 0..max_depth {
        if frontier.is_empty() {
            break;
        }
        let mut next_frontier = Vec::new();
        let mut sphere_codes = Vec::new();

        for (prev, cur, is_ghost, path) in &frontier {
            if *is_ghost {
                continue;
            }
            // Implicit hydrogens on `cur`
            let h_imp = mol.implicit_h_count(*cur);
            for _ in 0..h_imp {
                sphere_codes.push(1u32 << 10);
            }
            for &(nb, bi) in &adj[*cur] {
                let bo = mol.bonds[bi].order.max(1);
                if nb == *prev {
                    // Add (bo - 1) ghost copies pointing back along the incoming multiple bond
                    for _ in 1..bo {
                        let gc = ghost_code(nb);
                        sphere_codes.push(gc);
                        next_frontier.push((*cur, nb, true, path.clone()));
                    }
                    continue;
                }
                if path.contains(&nb) {
                    // Ring closure -> ghost atom per CIP rule
                    for _ in 0..bo {
                        let gc = ghost_code(nb);
                        sphere_codes.push(gc);
                        next_frontier.push((*cur, nb, true, path.clone()));
                    }
                } else {
                    let mut p2 = path.clone();
                    p2.push(nb);
                    sphere_codes.push(atom_code(nb));
                    next_frontier.push((*cur, nb, false, p2));
                    for _ in 1..bo {
                        sphere_codes.push(ghost_code(nb));
                        next_frontier.push((*cur, nb, true, path.clone()));
                    }
                }
            }
        }
        sphere_codes.sort_unstable_by(|a, b| b.cmp(a));
        sig.push(0xFFFF_FFFE); // sphere separator
        sig.extend(sphere_codes);
        frontier = next_frontier;
    }
    sig
}

fn parse_organic_subset(chars: &[char]) -> Result<(String, bool, usize), String> {
    let c0 = chars[0];
    let c1 = chars.get(1).copied();
    match (c0, c1) {
        ('C', Some('l')) => Ok(("Cl".to_string(), false, 2)),
        ('B', Some('r')) => Ok(("Br".to_string(), false, 2)),
        ('B', _) => Ok(("B".to_string(), false, 1)),
        ('C', _) => Ok(("C".to_string(), false, 1)),
        ('N', _) => Ok(("N".to_string(), false, 1)),
        ('O', _) => Ok(("O".to_string(), false, 1)),
        ('P', _) => Ok(("P".to_string(), false, 1)),
        ('S', _) => Ok(("S".to_string(), false, 1)),
        ('F', _) => Ok(("F".to_string(), false, 1)),
        ('I', _) => Ok(("I".to_string(), false, 1)),
        ('H', _) => Ok(("H".to_string(), false, 1)),
        ('D', _) => Ok(("D".to_string(), false, 1)),
        ('b', _) => Ok(("B".to_string(), true, 1)),
        ('c', _) => Ok(("C".to_string(), true, 1)),
        ('n', _) => Ok(("N".to_string(), true, 1)),
        ('o', _) => Ok(("O".to_string(), true, 1)),
        ('p', _) => Ok(("P".to_string(), true, 1)),
        ('s', _) => Ok(("S".to_string(), true, 1)),
        _ => Err(format!("Unsupported unbracketed SMILES character '{c0}'")),
    }
}

fn parse_bracket_atom(content: &str) -> Result<(AnalytAtom, bool, bool), String> {
    let cs: Vec<char> = content.chars().collect();
    let mut k = 0usize;
    let mut isotope: Option<u16> = None;
    if k < cs.len() && cs[k].is_ascii_digit() {
        let s = k;
        while k < cs.len() && cs[k].is_ascii_digit() {
            k += 1;
        }
        let iso_str: String = cs[s..k].iter().collect();
        isotope = Some(
            iso_str
                .parse::<u16>()
                .map_err(|_| format!("Invalid isotope '{iso_str}'"))?,
        );
    }
    if k >= cs.len() {
        return Err("Missing element in SMILES bracket atom".to_string());
    }
    let first = cs[k];
    let is_arom = first.is_ascii_lowercase();
    k += 1;
    let mut el = String::new();
    el.push(first.to_ascii_uppercase());
    if k < cs.len() && cs[k].is_ascii_lowercase() && !(cs[k] == 'h' && is_arom) {
        // Note: `[nH]` has uppercase H so `cs[k].is_ascii_lowercase()` is false; `[se]` has lowercase 'e'.
        el.push(cs[k]);
        k += 1;
    }
    if el == "D" {
        el = "H".to_string();
        isotope = Some(2);
    } else if el == "T" {
        el = "H".to_string();
        isotope = Some(3);
    }

    let mut stereo_tag = AtomStereoTag::None;
    if k < cs.len() && cs[k] == '@' {
        k += 1;
        if k < cs.len() && cs[k] == '@' {
            k += 1;
            stereo_tag = AtomStereoTag::TetrahedralCw;
        } else {
            stereo_tag = AtomStereoTag::TetrahedralCcw;
        }
    }

    let mut h_count: u8 = 0;
    let mut has_bracket_h = false;
    if k < cs.len() && cs[k] == 'H' {
        has_bracket_h = true;
        k += 1;
        if k < cs.len() && cs[k].is_ascii_digit() {
            let s = k;
            while k < cs.len() && cs[k].is_ascii_digit() {
                k += 1;
            }
            let hs: String = cs[s..k].iter().collect();
            h_count = hs.parse::<u8>().unwrap_or(1);
        } else {
            h_count = 1;
        }
    }

    let mut charge: i8 = 0;
    while k < cs.len() {
        if cs[k] == '+' || cs[k] == '-' {
            let sign: i8 = if cs[k] == '+' { 1 } else { -1 };
            k += 1;
            if k < cs.len() && cs[k].is_ascii_digit() {
                let s = k;
                while k < cs.len() && cs[k].is_ascii_digit() {
                    k += 1;
                }
                let qs: String = cs[s..k].iter().collect();
                charge += sign * qs.parse::<i8>().unwrap_or(1);
            } else {
                charge += sign;
                while k < cs.len() && cs[k] == (if sign > 0 { '+' } else { '-' }) {
                    charge += sign;
                    k += 1;
                }
            }
        } else {
            // Skip atom-map `:1` or unknown qualifier
            k += 1;
        }
    }

    let mut atom = AnalytAtom::new(&el);
    atom.isotope = isotope;
    atom.formal_charge = charge;
    atom.implicit_h = Some(h_count);
    atom.stereo_tag = stereo_tag;
    Ok((atom, is_arom, has_bracket_h))
}

fn resolve_ring_bond(
    open_bond: Option<(u8, bool, BondStereoTag)>,
    close_bond: Option<(u8, bool, BondStereoTag)>,
    both_arom: bool,
) -> (u8, bool, BondStereoTag, bool) {
    match (open_bond, close_bond) {
        (Some((o, a, s)), _) => (o, a, s, true),
        (None, Some((o, a, s))) => (o, a, s, false),
        (None, None) => {
            if both_arom {
                (1, true, BondStereoTag::None, true)
            } else {
                (1, false, BondStereoTag::None, true)
            }
        }
    }
}

fn bond_smiles_symbol(b: &AnalytBond, target: usize) -> &'static str {
    match b.stereo {
        BondStereoTag::Slash => {
            if b.b == target {
                "/"
            } else {
                "\\"
            }
        }
        BondStereoTag::Backslash => {
            if b.b == target {
                "\\"
            } else {
                "/"
            }
        }
        _ => match b.order {
            2 => "=",
            3 => "#",
            _ => "",
        },
    }
}

fn format_smiles_atom(mol: &AnalytMol, idx: usize) -> String {
    let a = &mol.atoms[idx];
    let organic_default = matches!(
        a.element.as_str(),
        "B" | "C" | "N" | "O" | "P" | "S" | "F" | "Cl" | "Br" | "I"
    );
    let total_h = mol.total_h_count(idx);
    let bo = mol.bond_order_sum(idx);
    let mut default_h = 0u8;
    for &v in standard_valences(&a.element, 0) {
        if v >= bo {
            default_h = v - bo;
            break;
        }
    }
    let needs_bracket = !organic_default
        || a.isotope.is_some()
        || a.formal_charge != 0
        || a.stereo_tag != AtomStereoTag::None
        || total_h != default_h;

    if !needs_bracket {
        return a.element.clone();
    }

    let mut s = String::from("[");
    if let Some(iso) = a.isotope {
        s.push_str(&iso.to_string());
    }
    s.push_str(&a.element);
    match a.stereo_tag {
        AtomStereoTag::TetrahedralCcw => s.push('@'),
        AtomStereoTag::TetrahedralCw => s.push_str("@@"),
        _ => {}
    }
    if total_h == 1 {
        s.push('H');
    } else if total_h > 1 {
        s.push_str(&format!("H{total_h}"));
    }
    if a.formal_charge > 0 {
        if a.formal_charge == 1 {
            s.push('+');
        } else {
            s.push_str(&format!("+{}", a.formal_charge));
        }
    } else if a.formal_charge < 0 {
        if a.formal_charge == -1 {
            s.push('-');
        } else {
            s.push_str(&format!("{}", a.formal_charge));
        }
    }
    s.push(']');
    s
}
