//! Rich Chemical Document, Molecular Graph, Periodic Table, Valence Model, and Canvas Page Objects.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! and open cheminformatics specifications (IUPAC Periodic Table, OpenSMILES valence model,
//! and ChemDraw page object model).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Error type for molecular parsing, layout, stereochemistry, and file I/O operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChemError {
    pub message: String,
}

impl ChemError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for ChemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ChemError {}

/// Standard periodic table symbols indexed by atomic number Z (1..=118; index 0 is `*` dummy).
pub const ELEMENT_SYMBOLS: [&str; 119] = [
    "*", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Most abundant natural isotope mass numbers indexed by Z (0..=118), used for CIP Rule 2
/// and molecular weight / InChI isotopic shift calculations.
pub const DEFAULT_MASS_NUMBERS: [u16; 119] = [
    0, 1, 4, 7, 9, 11, 12, 14, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 40, 39, 40, 45, 48, 51, 52,
    55, 56, 59, 58, 63, 64, 69, 74, 75, 80, 79, 84, 85, 88, 89, 90, 93, 98, 98, 102, 103, 106, 107,
    114, 115, 120, 121, 130, 127, 132, 133, 138, 139, 140, 141, 142, 145, 152, 153, 158, 159, 164,
    165, 166, 169, 174, 175, 180, 181, 184, 187, 192, 193, 195, 197, 202, 205, 208, 209, 209, 210,
    222, 223, 226, 227, 232, 231, 238, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262, 267,
    268, 271, 270, 277, 278, 281, 282, 285, 286, 289, 290, 293, 294, 294,
];

/// Look up atomic number `Z` (1..=118, or 0 for wildcard/dummy) from element symbol.
/// Also handles deuterium `"D"` and tritium `"T"` as Z=1.
pub fn atomic_number_from_symbol(sym: &str) -> Option<u8> {
    let trimmed = sym.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "*" || trimmed == "R" || trimmed == "A" || trimmed == "Q" {
        return Some(0);
    }
    if trimmed.eq_ignore_ascii_case("D") || trimmed.eq_ignore_ascii_case("T") {
        return Some(1);
    }
    let mut chars = trimmed.chars();
    let first = chars.next()?.to_ascii_uppercase();
    let rest: String = chars.map(|c| c.to_ascii_lowercase()).collect();
    let norm = format!("{}{}", first, rest);
    ELEMENT_SYMBOLS
        .iter()
        .position(|&s| s == norm)
        .map(|idx| idx as u8)
}

/// Canonical element symbol for atomic number `Z` (0..=118).
pub fn symbol_from_atomic_number(z: u8) -> &'static str {
    ELEMENT_SYMBOLS.get(z as usize).copied().unwrap_or("*")
}

/// Single-bond covalent radius in Ångströms (Pyykkö & Atsumi 2009 / Alvarez 2008).
pub fn covalent_radius(z: u8) -> f64 {
    match z {
        1 => 0.31,
        2 => 0.28,
        3 => 1.28,
        4 => 0.96,
        5 => 0.84,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        11 => 1.66,
        12 => 1.41,
        13 => 1.21,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        19 => 2.03,
        20 => 1.76,
        21..=30 => 1.32, // 3d transition metals
        31 => 1.22,
        32 => 1.20,
        33 => 1.19,
        34 => 1.20,
        35 => 1.20,
        39..=48 => 1.45, // 4d transition metals
        49 => 1.42,
        50 => 1.39,
        51 => 1.39,
        52 => 1.38,
        53 => 1.39,
        72..=80 => 1.46, // 5d transition metals
        81..=83 => 1.46,
        _ => 1.50,
    }
}

/// Check if an atomic number `Z` is a metal (alkali, alkaline earth, transition, lanthanide, actinide, post-transition).
pub fn is_metal(z: u8) -> bool {
    matches!(
        z,
        3 | 4
            | 11..=13
            | 19..=31
            | 37..=50
            | 55..=84
            | 87..=116
    )
}

/// Radical state on an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Radical {
    #[default]
    None,
    Singlet,
    Doublet,
    Triplet,
}

impl Radical {
    /// Number of valence electrons occupied by unpaired radical character.
    pub fn valence_deficit(self) -> u8 {
        match self {
            Radical::None => 0,
            Radical::Doublet => 1,
            Radical::Singlet | Radical::Triplet => 2,
        }
    }
}

/// Stored tetrahedral stereochemical parity relative to atom neighbor ordering `(n0, n1, n2, n3)`
/// (where an implicit H is treated as a virtual 4th neighbor at index `usize::MAX`).
/// Follows OpenSMILES convention: looking from `n0` toward the chiral center, `Anticlockwise` (`@`)
/// means `(n1, n2, n3)` wind counter-clockwise, and `Clockwise` (`@@`) means clockwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TetrahedralParity {
    /// Ordered neighbor atom indices (`usize::MAX` denotes the single implicit/virtual H).
    pub neighbors: [usize; 4],
    /// `true` = clockwise (`@@`), `false` = anticlockwise (`@`).
    pub clockwise: bool,
}

/// Chemical bond type / stereo / coordination display style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BondKind {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
    /// Solid wedge pointing from `a` (narrow) toward `b` (wide, toward viewer).
    WedgeUp,
    /// Hashed wedge pointing from `a` (narrow) toward `b` (wide, away from viewer).
    HashDown,
    /// Hollow wedge pointing from `a` toward `b`.
    HollowWedge,
    /// Wavy / unknown or racemic stereobond (`Either`).
    Wavy,
    /// Heavy bold single bond (Haworth / chair front edge).
    Bold,
    /// Dashed single bond.
    Dashed,
    /// Dative / coordination bond (`a -> b`, donor `a` to acceptor `b`).
    Dative,
    /// Non-covalent hydrogen bond (`a ... b`).
    Hydrogen,
    /// Non-covalent ionic association.
    Ionic,
}

impl BondKind {
    /// Numeric bond order for valence calculations (`0` for non-covalent/dative-to-metal).
    pub fn valence_order(self) -> u8 {
        match self {
            BondKind::Single
            | BondKind::WedgeUp
            | BondKind::HashDown
            | BondKind::HollowWedge
            | BondKind::Wavy
            | BondKind::Bold
            | BondKind::Dashed => 1,
            BondKind::Double => 2,
            BondKind::Triple => 3,
            BondKind::Aromatic => 1, // aromatic fractional valence handled in Molecule::implicit_hydrogens
            BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic => 0,
        }
    }

    /// Integer bond order (`1`, `2`, `3`, `0`) suitable for standard Kekule / analytical consumers.
    pub fn order_u8(self) -> u8 {
        match self {
            BondKind::Double => 2,
            BondKind::Triple => 3,
            BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic => 0,
            _ => 1,
        }
    }

    /// True if this bond is a stereo-indicating single bond (`WedgeUp`, `HashDown`, `HollowWedge`, `Wavy`).
    pub fn is_stereo_display(self) -> bool {
        matches!(
            self,
            BondKind::WedgeUp | BondKind::HashDown | BondKind::HollowWedge | BondKind::Wavy
        )
    }
}

/// Double-bond geometric configuration (`Auto` = infer from 2D coordinates, `Cis`, `Trans`, `Either`/crossed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DoubleBondGeometry {
    #[default]
    Auto,
    /// Explicit cis (`Z`-like on reference substituents `[ref_a, ref_b]`).
    Cis { ref_a: usize, ref_b: usize },
    /// Explicit trans (`E`-like on reference substituents `[ref_a, ref_b]`).
    Trans { ref_a: usize, ref_b: usize },
    /// Crossed / unknown alkene stereochemistry (`CFG=2` / wavy).
    Either,
}

/// Atom in a `Molecule`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    /// Atomic number `Z` (1..=118, or 0 for wildcard `*`/superatom/multi-center dummy).
    pub z: u8,
    /// Canonical element or alias symbol (`"C"`, `"N"`, `"Fe"`, `"*"`).
    pub symbol: String,
    /// Explicit isotope mass number (e.g., `Some(13)` for `13C`, `Some(15)` for `15N`, `Some(2)` for `2H`/`D`).
    pub isotope: Option<u16>,
    /// Formal integer charge (e.g., `-1`, `0`, `+1`).
    pub formal_charge: i8,
    /// Radical state.
    pub radical: Radical,
    /// Optional explicit valence override.
    pub explicit_valence: Option<u8>,
    /// Optional explicit hydrogen count override (e.g., from SMILES bracket `[CH3]` or `[nH]`).
    pub explicit_hydrogens: Option<u8>,
    /// Flag indicating this atom was parsed as aromatic (e.g., lowercase SMILES `c`, `n`, `o`, `s`).
    pub aromatic: bool,
    /// 2D X coordinate (Ångströms; standard bond length = `1.54` Å).
    pub x: f64,
    /// 2D Y coordinate (Ångströms; standard bond length = `1.54` Å).
    pub y: f64,
    /// Optional 3D Z coordinate (Ångströms).
    pub z_coord: Option<f64>,
    /// Superatom / abbreviation label (e.g. `"Boc"`, `"Ph"`, `"COOH"`).
    pub superatom_label: Option<String>,
    /// Optional collapsed inner fragment preserved when an abbreviation is collapsed.
    pub inner_fragment: Option<Box<Molecule>>,
    /// Atom numbering annotation (e.g., `"1"`, `"2a"`, `"Cα"`).
    pub atom_number: Option<String>,
    /// Reaction atom-to-atom mapping number (`[C:1]`).
    pub atom_map: Option<u16>,
    /// Attachment point index (`1`, `2`, ...) for superatom/monomer templates.
    pub attachment_point: Option<u8>,
    /// Multi-center η-hapticity ring/ligand atom indices when `z == 0` is a centroid node (e.g. ferrocene η5-Cp).
    pub multi_center_atoms: Vec<usize>,
    /// Stored tetrahedral parity (from SMILES `@`/`@@`, InChI `/t`, or MOL parity).
    pub parity: Option<TetrahedralParity>,
}

impl Atom {
    /// Construct a standard atom from element symbol (e.g. `"C"`, `"N"`, `"O"`, `"D"`, `"Fe"`).
    pub fn new(symbol: &str) -> Self {
        let trimmed = symbol.trim();
        let (z, canon_sym, iso) = if trimmed.eq_ignore_ascii_case("D") {
            (1, "H".to_string(), Some(2))
        } else if trimmed.eq_ignore_ascii_case("T") {
            (1, "H".to_string(), Some(3))
        } else if let Some(z) = atomic_number_from_symbol(trimmed) {
            let s = if z == 0 {
                trimmed.to_string()
            } else {
                symbol_from_atomic_number(z).to_string()
            };
            (z, s, None)
        } else {
            (0, trimmed.to_string(), None)
        };
        Self {
            z,
            symbol: canon_sym,
            isotope: iso,
            formal_charge: 0,
            radical: Radical::None,
            explicit_valence: None,
            explicit_hydrogens: None,
            aromatic: false,
            x: 0.0,
            y: 0.0,
            z_coord: None,
            superatom_label: if z == 0 && trimmed != "*" {
                Some(trimmed.to_string())
            } else {
                None
            },
            inner_fragment: None,
            atom_number: None,
            atom_map: None,
            attachment_point: None,
            multi_center_atoms: Vec::new(),
            parity: None,
        }
    }

    /// Construct an explicit superatom / functional-group abbreviation node (e.g. `"Boc"`, `"Ac"`, `"Ts"`, `"Ph"`),
    /// disambiguating organic abbreviations like `"Ac"` (acetyl) and `"Ts"` (tosyl) from heavy elements (`Z=89`, `Z=117`).
    pub fn new_superatom(label: &str) -> Self {
        let trimmed = label.trim();
        Self {
            z: 0,
            symbol: trimmed.to_string(),
            isotope: None,
            formal_charge: 0,
            radical: Radical::None,
            explicit_valence: None,
            explicit_hydrogens: None,
            aromatic: false,
            x: 0.0,
            y: 0.0,
            z_coord: None,
            superatom_label: Some(trimmed.to_string()),
            inner_fragment: None,
            atom_number: None,
            atom_map: None,
            attachment_point: None,
            multi_center_atoms: Vec::new(),
            parity: None,
        }
    }

    /// Construct a standard atom from atomic number `Z` with 2D coordinates.
    pub fn from_z_xy(z: u8, x: f64, y: f64) -> Self {
        let mut atom = Self::new(symbol_from_atomic_number(z));
        atom.x = x;
        atom.y = y;
        atom
    }

    /// Effective mass number (explicit isotope if set, otherwise standard natural isotope mass number).
    pub fn mass_number(&self) -> u16 {
        self.isotope.unwrap_or_else(|| {
            DEFAULT_MASS_NUMBERS
                .get(self.z as usize)
                .copied()
                .unwrap_or(0)
        })
    }
}

/// Bond between two atoms `a` and `b` in a `Molecule`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bond {
    /// Start atom index (for `WedgeUp`/`HashDown`, the narrow chiral tip; for `Dative`, the electron donor).
    pub a: usize,
    /// End atom index (for `WedgeUp`/`HashDown`, the wide end; for `Dative`, the electron acceptor).
    pub b: usize,
    /// Bond order / stereo / coordination type.
    pub kind: BondKind,
    /// Explicit or automatic double-bond `cis`/`trans` geometry.
    pub double_geometry: DoubleBondGeometry,
}

impl Bond {
    pub fn new(a: usize, b: usize, kind: BondKind) -> Self {
        Self {
            a,
            b,
            kind,
            double_geometry: DoubleBondGeometry::Auto,
        }
    }

    /// Return the other endpoint of this bond if `atom` is one of its endpoints.
    pub fn other(&self, atom: usize) -> Option<usize> {
        if self.a == atom {
            Some(self.b)
        } else if self.b == atom {
            Some(self.a)
        } else {
            None
        }
    }

    /// True if this bond is aromatic (`BondKind::Aromatic`).
    pub fn is_aromatic(&self) -> bool {
        self.kind == BondKind::Aromatic
    }
}

/// Core Molecular Graph (`Molecule`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Molecule {
    pub name: String,
    pub comment: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Arbitrary key-value metadata (e.g. SDF properties or experimental notes).
    pub properties: BTreeMap<String, String>,
}

impl Molecule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Add an atom and return its index.
    pub fn add_atom(&mut self, atom: Atom) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(atom);
        idx
    }

    /// Add a bond between `a` and `b` (no-op returning existing index if already bonded) and return its index.
    pub fn add_bond(&mut self, a: usize, b: usize, kind: BondKind) -> usize {
        if let Some(existing) = self.find_bond(a, b) {
            self.bonds[existing].kind = kind;
            return existing;
        }
        let idx = self.bonds.len();
        self.bonds.push(Bond::new(a, b, kind));
        idx
    }

    /// Find bond index connecting `u` and `v` (in either direction).
    pub fn find_bond(&self, u: usize, v: usize) -> Option<usize> {
        self.bonds
            .iter()
            .position(|b| (b.a == u && b.b == v) || (b.a == v && b.b == u))
    }

    /// Return `(neighbor_atom_idx, bond_idx)` pairs incident on `atom` (covalent + aromatic + dative).
    pub fn neighbors(&self, atom: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (bi, b) in self.bonds.iter().enumerate() {
            if matches!(b.kind, BondKind::Hydrogen | BondKind::Ionic) {
                continue;
            }
            if let Some(nb) = b.other(atom) {
                out.push((nb, bi));
            }
        }
        out
    }

    /// Return covalent `(neighbor_atom_idx, bond_idx)` pairs (excluding dative, hydrogen, ionic, and multi-center dummies).
    pub fn covalent_neighbors(&self, atom: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (bi, b) in self.bonds.iter().enumerate() {
            if matches!(
                b.kind,
                BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic
            ) {
                continue;
            }
            if let Some(nb) = b.other(atom) {
                if self
                    .atoms
                    .get(nb)
                    .map(|a| !a.multi_center_atoms.is_empty())
                    .unwrap_or(false)
                {
                    continue;
                }
                out.push((nb, bi));
            }
        }
        out
    }

    /// Compute standard OpenSMILES / MDL implicit hydrogen count for atom `idx`.
    pub fn implicit_hydrogens(&self, idx: usize) -> u8 {
        let Some(atom) = self.atoms.get(idx) else {
            return 0;
        };
        if atom.z == 0 || atom.superatom_label.is_some() || !atom.multi_center_atoms.is_empty() {
            return atom.explicit_hydrogens.unwrap_or(0);
        }
        if let Some(eh) = atom.explicit_hydrogens {
            return eh;
        }

        // Sum covalent bond orders around `idx`.
        let mut bond_order_sum: u8 = 0;
        let mut arom_bonds: u8 = 0;
        for (bi, b) in self.bonds.iter().enumerate() {
            let _ = bi;
            if b.a != idx && b.b != idx {
                continue;
            }
            match b.kind {
                BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic => {}
                BondKind::Aromatic => {
                    arom_bonds += 1;
                }
                other => {
                    bond_order_sum = bond_order_sum.saturating_add(other.valence_order());
                }
            }
        }

        // In unkekulized aromatic rings, 2 aromatic bonds contribute 3 to valence (1 single + 1 double)
        // for neutral sp2 ring carbons/aza-nitrogens unless heteroatom lone-pair rules apply.
        let valence_used = if arom_bonds >= 2 {
            // e.g. benzene C (2 arom -> 3), naphthalene fusion C (3 arom -> 4)
            bond_order_sum + arom_bonds + 1
        } else if arom_bonds == 1 {
            bond_order_sum + 1
        } else {
            bond_order_sum
        };

        let used_with_rad = valence_used.saturating_add(atom.radical.valence_deficit());

        if let Some(ev) = atom.explicit_valence {
            return ev.saturating_sub(used_with_rad);
        }

        let q = atom.formal_charge;
        let allowed_valences: &[u8] = match (atom.z, q) {
            (1, 0) => &[1],
            (1, _) => &[0],
            (5, 0) => &[3],
            (5, -1) => &[4],
            (6, 0) => &[4],
            (6, 1) | (6, -1) => &[3],
            (7, 0) => {
                // Pyridine-like unkekulized aromatic N with 2 aromatic bonds and 0 single bonds has 0 implicit H!
                // Pyrrole-like [nH] always sets explicit_hydrogens = Some(1) in SMILES, or has Kekule single bonds.
                if atom.aromatic && arom_bonds == 2 && bond_order_sum == 0 {
                    &[3]
                } else {
                    &[3, 5]
                }
            }
            (7, 1) => &[4],
            (7, -1) => &[2],
            (8, 0) => {
                if atom.aromatic && arom_bonds >= 2 {
                    return 0;
                }
                &[2]
            }
            (8, 1) => &[3],
            (8, -1) => &[1],
            (9, 0) | (17, 0) | (35, 0) | (53, 0) => &[1],
            (14, 0) => &[4],
            (15, 0) => &[3, 5],
            (15, 1) => &[4],
            (16, 0) => {
                if atom.aromatic && arom_bonds >= 2 {
                    return 0;
                }
                &[2, 4, 6]
            }
            (16, 1) => &[3, 5],
            (16, -1) => &[1],
            _ => &[],
        };

        for &v in allowed_valences {
            if v >= used_with_rad {
                return v - used_with_rad;
            }
        }
        0
    }

    /// Total hydrogens attached to `idx` (implicit hydrogens + explicit `Z == 1` atom neighbors).
    pub fn total_hydrogens(&self, idx: usize) -> u8 {
        let explicit_h_neighbors = self
            .covalent_neighbors(idx)
            .into_iter()
            .filter(|&(nb, _)| self.atoms[nb].z == 1)
            .count() as u8;
        self.implicit_hydrogens(idx)
            .saturating_add(explicit_h_neighbors)
    }

    /// True if all atoms have zero 2D coordinates (or fewer than 2 non-overlapping atoms).
    pub fn needs_2d_coords(&self) -> bool {
        if self.atoms.len() <= 1 {
            return false;
        }
        let p0 = (self.atoms[0].x, self.atoms[0].y);
        self.atoms
            .iter()
            .skip(1)
            .all(|a| (a.x - p0.0).hypot(a.y - p0.1) < 1e-4)
    }

    /// True if all atoms carry `Some(z_coord)` with at least one non-zero Z or 3D layout.
    pub fn has_3d_coords(&self) -> bool {
        !self.atoms.is_empty() && self.atoms.iter().all(|a| a.z_coord.is_some())
    }

    /// Compute connected components as lists of atom indices.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.atoms.len();
        let mut visited = vec![false; n];
        let mut comps = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(u) = stack.pop() {
                comp.push(u);
                for (bi, b) in self.bonds.iter().enumerate() {
                    let _ = bi;
                    if let Some(v) = b.other(u) {
                        if !visited[v] {
                            visited[v] = true;
                            stack.push(v);
                        }
                    }
                }
                for &mc in &self.atoms[u].multi_center_atoms {
                    if mc < n && !visited[mc] {
                        visited[mc] = true;
                        stack.push(mc);
                    }
                }
            }
            comp.sort_unstable();
            comps.push(comp);
        }
        comps
    }

    /// Split a multi-fragment `Molecule` into separate `Molecule` objects (one per connected component).
    pub fn split_fragments(&self) -> Vec<Molecule> {
        let comps = self.connected_components();
        if comps.len() <= 1 {
            return vec![self.clone()];
        }
        comps
            .into_iter()
            .map(|comp| self.extract_subgraph(&comp))
            .collect()
    }

    /// Extract an induced subgraph for `atom_indices` (preserving atom order and remapping bonds/parities).
    pub fn extract_subgraph(&self, atom_indices: &[usize]) -> Molecule {
        let mut map = vec![None; self.atoms.len()];
        let mut sub = Molecule::new();
        for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
            map[old_idx] = Some(new_idx);
            let mut atom = self.atoms[old_idx].clone();
            atom.multi_center_atoms = atom
                .multi_center_atoms
                .into_iter()
                .filter_map(|m| map.get(m).and_then(|&x| x))
                .collect();
            sub.atoms.push(atom);
        }
        // Fix up multi_center_atoms and parities after full map is populated
        for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
            sub.atoms[new_idx].multi_center_atoms = self.atoms[old_idx]
                .multi_center_atoms
                .iter()
                .filter_map(|&m| map.get(m).and_then(|&x| x))
                .collect();
            if let Some(p) = self.atoms[old_idx].parity {
                let mut ok = true;
                let mut new_nbs = [0usize; 4];
                for (k, dst) in new_nbs.iter_mut().enumerate() {
                    if p.neighbors[k] == usize::MAX {
                        *dst = usize::MAX;
                    } else if let Some(mapped) = map.get(p.neighbors[k]).and_then(|&x| x) {
                        *dst = mapped;
                    } else {
                        ok = false;
                        break;
                    }
                }
                sub.atoms[new_idx].parity = if ok {
                    Some(TetrahedralParity {
                        neighbors: new_nbs,
                        clockwise: p.clockwise,
                    })
                } else {
                    None
                };
            }
        }
        for b in &self.bonds {
            if let (Some(na), Some(nb)) = (map[b.a], map[b.b]) {
                let mut nbond = b.clone();
                nbond.a = na;
                nbond.b = nb;
                nbond.double_geometry = match b.double_geometry {
                    DoubleBondGeometry::Cis { ref_a, ref_b } => {
                        if let (Some(ra), Some(rb)) = (
                            map.get(ref_a).and_then(|&x| x),
                            map.get(ref_b).and_then(|&x| x),
                        ) {
                            DoubleBondGeometry::Cis {
                                ref_a: ra,
                                ref_b: rb,
                            }
                        } else {
                            DoubleBondGeometry::Auto
                        }
                    }
                    DoubleBondGeometry::Trans { ref_a, ref_b } => {
                        if let (Some(ra), Some(rb)) = (
                            map.get(ref_a).and_then(|&x| x),
                            map.get(ref_b).and_then(|&x| x),
                        ) {
                            DoubleBondGeometry::Trans {
                                ref_a: ra,
                                ref_b: rb,
                            }
                        } else {
                            DoubleBondGeometry::Auto
                        }
                    }
                    other => other,
                };
                sub.bonds.push(nbond);
            }
        }
        sub
    }

    /// Merge another `Molecule` into `self`, returning the atom index offset applied to `other`.
    pub fn append_molecule(&mut self, other: &Molecule) -> usize {
        let offset = self.atoms.len();
        for a in &other.atoms {
            let mut na = a.clone();
            na.multi_center_atoms = na
                .multi_center_atoms
                .into_iter()
                .map(|m| m + offset)
                .collect();
            if let Some(mut p) = na.parity {
                for nb in &mut p.neighbors {
                    if *nb != usize::MAX {
                        *nb += offset;
                    }
                }
                na.parity = Some(p);
            }
            self.atoms.push(na);
        }
        for b in &other.bonds {
            let mut nb = b.clone();
            nb.a += offset;
            nb.b += offset;
            nb.double_geometry = match b.double_geometry {
                DoubleBondGeometry::Cis { ref_a, ref_b } => DoubleBondGeometry::Cis {
                    ref_a: ref_a + offset,
                    ref_b: ref_b + offset,
                },
                DoubleBondGeometry::Trans { ref_a, ref_b } => DoubleBondGeometry::Trans {
                    ref_a: ref_a + offset,
                    ref_b: ref_b + offset,
                },
                other_geom => other_geom,
            };
            self.bonds.push(nb);
        }
        offset
    }

    /// Hill system molecular formula string (e.g. `"C8H9NO2"`, `"H2O"`, `"C6H12O6"`).
    pub fn hill_formula(&self) -> String {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..self.atoms.len() {
            let a = &self.atoms[i];
            if a.z == 0 {
                continue;
            }
            *counts.entry(a.symbol.clone()).or_insert(0) += 1;
            let ih = self.implicit_hydrogens(i) as usize;
            if ih > 0 {
                *counts.entry("H".to_string()).or_insert(0) += ih;
            }
        }
        let mut out = String::new();
        let has_c = counts.contains_key("C");
        if has_c {
            if let Some(c) = counts.remove("C") {
                out.push('C');
                if c > 1 {
                    out.push_str(&c.to_string());
                }
            }
            if let Some(h) = counts.remove("H") {
                out.push('H');
                if h > 1 {
                    out.push_str(&h.to_string());
                }
            }
        }
        for (sym, cnt) in counts {
            out.push_str(&sym);
            if cnt > 1 {
                out.push_str(&cnt.to_string());
            }
        }
        out
    }

    /// Net formal charge across all atoms.
    pub fn total_charge(&self) -> i32 {
        self.atoms.iter().map(|a| a.formal_charge as i32).sum()
    }

    /// 2D bounding box `(min_x, min_y, max_x, max_y)` in Ångströms.
    pub fn bounds_2d(&self) -> Option<(f64, f64, f64, f64)> {
        if self.atoms.is_empty() {
            return None;
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for a in &self.atoms {
            min_x = min_x.min(a.x);
            min_y = min_y.min(a.y);
            max_x = max_x.max(a.x);
            max_y = max_y.max(a.y);
        }
        Some((min_x, min_y, max_x, max_y))
    }
}

// ============================================================================
// Canvas Page Objects for Rich Reaction Schemes & Figures (CDXML / CDX / OCT)
// ============================================================================

/// Reaction arrow style on a chemical drawing page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ReactionArrowKind {
    #[default]
    Solid,
    Equilibrium,
    UnbalancedEquilibrium,
    Resonance,
    Retrosynthetic,
    NoGoCross,
    NoGoHash,
    Dashed,
}

/// Straight or curved reaction scheme arrow (`->`, `<=>`, `<->`, `=>`, `-X->`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactionArrow {
    pub kind: ReactionArrowKind,
    /// Tail `(x, y)` in Ångström canvas coordinates.
    pub tail: (f64, f64),
    /// Head `(x, y)` in Ångström canvas coordinates.
    pub head: (f64, f64),
    /// Optional reagent / condition text above the arrow.
    pub above_text: Option<String>,
    /// Optional solvent / temperature / yield text below the arrow.
    pub below_text: Option<String>,
}

/// Anchor target for an electron-pushing `CurvedArrow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MechanismAnchor {
    /// Molecule index and atom index within `ChemDocument::molecules`.
    Atom { mol: usize, atom: usize },
    /// Molecule index and bond index within `ChemDocument::molecules`.
    Bond { mol: usize, bond: usize },
}

/// Electron-pushing curved arrow (2-electron full head or 1-electron radical fishhook).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvedArrow {
    /// `2` = full double-barbed arrowhead (electron pair), `1` = single-barbed fishhook (radical).
    pub electrons: u8,
    /// Start `(x, y)` in Ångström canvas coordinates.
    pub start: (f64, f64),
    /// Cubic/quadratic control point `(x, y)` defining the arc curvature.
    pub control: (f64, f64),
    /// End `(x, y)` in Ångström canvas coordinates.
    pub end: (f64, f64),
    /// Optional logical source anchor (lone pair / atom or π/σ bond).
    pub source_anchor: Option<MechanismAnchor>,
    /// Optional logical target anchor (atom or bond).
    pub target_anchor: Option<MechanismAnchor>,
}

/// Polymer / mixture / SRU bracket shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BracketStyle {
    #[default]
    Square,
    Round,
    Brace,
}

/// Semantic polymer / group bracket usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BracketUsage {
    #[default]
    Sru,
    MultipleGroup,
    Copolymer,
    CopolymerAlternating,
    CopolymerBlock,
    CopolymerRandom,
    Monomer,
    Mixture,
    Generic,
}

/// Paired bracket group (e.g. `[...]_n`, copolymer `( ... )_co`, mixture `{ ... }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bracket {
    pub style: BracketStyle,
    pub usage: BracketUsage,
    /// Bounding box `(min_x, min_y, max_x, max_y)` in Ångström canvas coordinates.
    pub bounds: (f64, f64, f64, f64),
    /// Subscript label (e.g. `"n"`, `"m"`, `"12"`, `"ran"`, `"block"`).
    pub subscript: String,
    /// Superscript / connectivity label (e.g. `"HT"`, `"HH"`, `"EU"`).
    pub superscript: Option<String>,
    /// Enclosed `(mol_idx, atom_idx)` pairs.
    pub enclosed_atoms: Vec<(usize, usize)>,
}

/// Atomic / hybrid / d-orbital symmetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OrbitalKind {
    #[default]
    S,
    P,
    Sp3,
    Dz2,
    Dxy,
}

/// Orbital lobe shading / sign phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OrbitalPhase {
    #[default]
    FilledPlus,
    ShadedMinus,
    Hollow,
}

/// Drawn atomic or hybrid orbital (`s`, `p`, `sp3`, `dz2`, `dxy`) on the canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Orbital {
    pub kind: OrbitalKind,
    pub phase: OrbitalPhase,
    /// Nodal / atomic center `(x, y)` in Ångström canvas coordinates.
    pub center: (f64, f64),
    /// Major lobe tip `(x, y)` defining orientation and size.
    pub axis_end: (f64, f64),
    /// Optional anchored `(mol_idx, atom_idx)`.
    pub atom_anchor: Option<(usize, usize)>,
}

/// Spot display style on a TLC plate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TlcSpotStyle {
    #[default]
    Solid,
    Hollow,
    Shaded,
    UvActive,
}

/// Single spot on a `TlcPlate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TlcSpot {
    /// 0-based lane index.
    pub lane: usize,
    /// Retention factor $R_f \in [0.0, 1.0]$.
    pub rf: f64,
    pub style: TlcSpotStyle,
    pub label: Option<String>,
}

/// Thin-layer chromatography (TLC) plate figure with origin line, solvent front, lanes, and $R_f$ spots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TlcPlate {
    /// Plate rectangle `(min_x, min_y, max_x, max_y)` in Ångström canvas coordinates.
    pub bounds: (f64, f64, f64, f64),
    /// Baseline origin fraction from bottom (`0.0..1.0`, typically `0.10`).
    pub origin_fraction: f64,
    /// Solvent front fraction from top (`0.0..1.0`, typically `0.10`, i.e. 90% up the plate).
    pub solvent_front_fraction: f64,
    /// Lane labels (length = number of lanes).
    pub lane_labels: Vec<String>,
    /// Spots across all lanes.
    pub spots: Vec<TlcSpot>,
}

/// Styled text run within a `RichTextBox`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSpan {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub subscript: bool,
    pub superscript: bool,
    pub font_size_pt: Option<f64>,
    pub color_rgb: Option<(u8, u8, u8)>,
}

impl TextSpan {
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            bold: false,
            italic: false,
            subscript: false,
            superscript: false,
            font_size_pt: None,
            color_rgb: None,
        }
    }
}

/// Free-floating rich text box / caption on a `ChemDocument` page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichTextBox {
    pub position: (f64, f64),
    pub spans: Vec<TextSpan>,
}

impl RichTextBox {
    pub fn plain(x: f64, y: f64, text: impl Into<String>) -> Self {
        Self {
            position: (x, y),
            spans: vec![TextSpan::plain(text)],
        }
    }

    pub fn plain_text(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }
}

/// Geometric shape kind on a chemical canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ShapeKind {
    #[default]
    Rectangle,
    RoundedRectangle,
    Oval,
    Circle,
    Line,
}

/// Geometric box / circle / divider shape on a `ChemDocument` page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanvasShape {
    pub kind: ShapeKind,
    pub bounds: (f64, f64, f64, f64),
    pub dashed: bool,
    pub filled: bool,
}

/// Complete Chemical Drawing Document (`ChemDocument`) capable of lossless round-tripping
/// across `.oct`, `.cdxml`, and binary `.cdx` (`VjCD0100`), including molecules, reaction schemes,
/// electron-pushing mechanisms, polymer brackets, orbitals, TLC plates, and rich text captions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ChemDocument {
    pub title: String,
    pub molecules: Vec<Molecule>,
    pub reaction_arrows: Vec<ReactionArrow>,
    pub curved_arrows: Vec<CurvedArrow>,
    pub brackets: Vec<Bracket>,
    pub orbitals: Vec<Orbital>,
    pub tlc_plates: Vec<TlcPlate>,
    pub text_boxes: Vec<RichTextBox>,
    pub shapes: Vec<CanvasShape>,
}

impl ChemDocument {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap a single `Molecule` into a `ChemDocument`.
    pub fn from_molecule(mol: Molecule) -> Self {
        Self {
            title: mol.name.clone(),
            molecules: vec![mol],
            ..Self::default()
        }
    }

    /// Combine all fragments in `self.molecules` into a single `Molecule`.
    pub fn merged_molecule(&self) -> Molecule {
        let mut out = Molecule::with_name(self.title.clone());
        for m in &self.molecules {
            out.append_molecule(m);
        }
        out
    }
}
