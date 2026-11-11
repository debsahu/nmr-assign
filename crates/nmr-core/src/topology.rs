//! Standard 20-amino-acid IUPAC covalent spin-system topologies and BMRB sidechain priors.
//!
//! Provenance: Clean-room original implementation derived strictly from:
//! - IUPAC-IUBMB 1998 NMR atom nomenclature recommendations: Markley et al.,
//!   *J. Biomol. NMR* 12, 1-23 (1998).
//! - RCSB PDB Chemical Component Dictionary (CCD) standard L-peptide monomer covalent
//!   bond graphs: Westbrook et al., *Bioinformatics* 31, 1274-1278 (2015) (CC0 1.0).
//! - Public Biological Magnetic Resonance Data Bank (BMRB) chemical shift statistics:
//!   Ulrich et al., *Nucleic Acids Res.* 36, D402-D408 (2008), computed with a robust
//!   3.5-MAD outlier trim across 30 public BMRB NMR-STAR entries strictly disjoint from
//!   the benchmark suite (`bmr4022`, `bmr4098`, `bmr4106`, `bmr4144`, `bmr4267`, `bmr4302`,
//!   `bmr4391`, `bmr4472`, `bmr4562`, `bmr4670`, `bmr4752`, `bmr4807`, `bmr5010`, `bmr5106`,
//!   `bmr5211`, `bmr5505`, `bmr5694`, `bmr5757`, `bmr6032`, `bmr6240`, `bmr6338`, `bmr6622`,
//!   `bmr6801`, `bmr7021`, `bmr7280`, `bmr15250`, `bmr15380`, `bmr16042`, `bmr16310`, `bmr18025`).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

/// Chemical element in a standard protein residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    S,
}

/// Proton multiplicity of a `13C-1H_n` spin pair observed in `13C-HSQC`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChMultiplicity {
    /// Methine `CH` (1 attached proton, e.g. `CA-HA`, `ILE CB-HB`, `LEU CG-HG`).
    Methine,
    /// Methylene `CH2` (2 diastereotopic/prochiral protons, e.g. `CB-HB2/HB3`, `GLY CA-HA2/HA3`).
    Methylene,
    /// Methyl `CH3` (3 magnetically equivalent protons under fast C3 rotation, e.g. `ALA CB-HB1..3`).
    Methyl,
}

impl ChMultiplicity {
    #[inline]
    pub const fn proton_count(self) -> usize {
        match self {
            Self::Methine => 1,
            Self::Methylene => 2,
            Self::Methyl => 3,
        }
    }

    /// Expected sign in a multiplicity-edited `13C-HSQC` (`CH`/`CH3` = `+1`, `CH2` = `-1`).
    #[inline]
    pub const fn edited_hsqc_sign(self) -> i32 {
        match self {
            Self::Methine | Self::Methyl => 1,
            Self::Methylene => -1,
        }
    }
}

/// Spin-network classification of a protonated carbon atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CarbonRegion {
    /// Part of the continuous `13C-13C` one-bond aliphatic TOCSY network rooted at `CA`.
    /// Observed in `C13HSQC`, `CCCONH` (`i-1`), `HCCCONH` (`i-1`), and aliphatic `HCCHTOCSY`.
    AliphaticCaConnected,
    /// Aliphatic methyl isolated from `CA` by a heteroatom (`MET CE` across thioether `SD`).
    /// Observed in `C13HSQC` and diagonal `HCCHTOCSY`, but NOT relayed to `CA`/`CO` in `CCCONH`.
    AliphaticIsolatedMethyl,
    /// Aromatic ring `CH` group (`PHE`, `TYR`, `HIS`, `TRP`).
    Aromatic,
}

impl CarbonRegion {
    #[inline]
    pub const fn is_aliphatic(self) -> bool {
        matches!(
            self,
            Self::AliphaticCaConnected | Self::AliphaticIsolatedMethyl
        )
    }
}

/// IUPAC atom specification within a standard amino acid residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomSpec {
    pub name: &'static str,
    pub element: Element,
    /// For `Element::H`, the covalently bonded heavy atom (`N`, `CA`, `CB`, ...).
    pub attached_heavy: Option<&'static str>,
    /// True if this atom participates in the `CA`-connected aliphatic `13C-13C` TOCSY network.
    pub in_aliphatic_tocsy: bool,
    /// True for backbone peptide atoms (`N`, `H`, `CA`, `C`, `O`).
    pub is_backbone: bool,
    /// True if routinely assigned in solution `1H`/`13C`/`15N` NMR (`C`, `H`, `N` excluding `O`/`S`).
    pub is_assignable: bool,
}

const fn heavy(
    name: &'static str,
    el: Element,
    aliph: bool,
    bb: bool,
    assignable: bool,
) -> AtomSpec {
    AtomSpec {
        name,
        element: el,
        attached_heavy: None,
        in_aliphatic_tocsy: aliph,
        is_backbone: bb,
        is_assignable: assignable,
    }
}

const fn proton(
    name: &'static str,
    parent: &'static str,
    aliph: bool,
    bb: bool,
    assignable: bool,
) -> AtomSpec {
    AtomSpec {
        name,
        element: Element::H,
        attached_heavy: Some(parent),
        in_aliphatic_tocsy: aliph,
        is_backbone: bb,
        is_assignable: assignable,
    }
}

/// Specification of a protonated carbon `CH` / `CH2` / `CH3` group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChGroupSpec {
    pub carbon: &'static str,
    pub protons: &'static [&'static str],
    pub multiplicity: ChMultiplicity,
    pub region: CarbonRegion,
    /// Adjacent protonated carbons within the same `13C-13C` scalar-coupled network.
    pub covalent_carbon_neighbors: &'static [&'static str],
    /// Number of `13C-13C` covalent bonds from `CA` (`Some(0)` for `CA`, `None` if disconnected).
    pub steps_from_ca: Option<u8>,
}

/// Magnetically equivalent 3-proton methyl group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethylGroup {
    pub carbon: &'static str,
    pub protons: [&'static str; 3],
}

/// Prochiral or ring-flip symmetry equivalence class (NMR-STAR `Ambiguity_code` 2 or 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProchiralGroup {
    /// Diastereotopic geminal methylene protons on a single carbon (`HB2`/`HB3`, `HA2`/`HA3`, etc.).
    Methylene {
        carbon: &'static str,
        protons: [&'static str; 2],
    },
    /// Prochiral isopropyl methyl branches (`VAL CG1/CG2`, `LEU CD1/CD2`).
    IsopropylMethyls {
        carbon_a: &'static str,
        protons_a: [&'static str; 3],
        carbon_b: &'static str,
        protons_b: [&'static str; 3],
    },
    /// Symmetric aromatic ortho/meta `CH` pair under fast 180-degree ring flips (`PHE`, `TYR`).
    SymmetricAromatic {
        carbon_a: &'static str,
        proton_a: &'static str,
        carbon_b: &'static str,
        proton_b: &'static str,
    },
}

/// Complete IUPAC covalent topology and TOCSY metadata for one standard amino acid.
#[derive(Clone, Copy, Debug)]
pub struct ResidueTopology {
    pub three_letter: &'static str,
    pub one_letter: char,
    pub atoms: &'static [AtomSpec],
    pub bonds: &'static [(&'static str, &'static str)],
    pub ch_groups: &'static [ChGroupSpec],
    pub methyl_groups: &'static [MethylGroup],
    pub prochiral_groups: &'static [ProchiralGroup],
}

impl ResidueTopology {
    /// Find an atom specification by IUPAC name.
    pub fn atom(&self, name: &str) -> Option<&'static AtomSpec> {
        self.atoms.iter().find(|a| a.name == name)
    }

    /// Find a protonated carbon group specification by carbon name.
    pub fn ch_group(&self, carbon: &str) -> Option<&'static ChGroupSpec> {
        self.ch_groups.iter().find(|g| g.carbon == carbon)
    }

    /// Return all `CA`-connected aliphatic `CH_n` groups (`CA`, `CB`, `CG`, `CD`, `CE`).
    pub fn aliphatic_ca_groups(&self) -> impl Iterator<Item = &'static ChGroupSpec> {
        self.ch_groups
            .iter()
            .filter(|g| g.region == CarbonRegion::AliphaticCaConnected)
    }

    /// Return all aliphatic `CH_n` groups (including isolated `MET CE`).
    pub fn all_aliphatic_groups(&self) -> impl Iterator<Item = &'static ChGroupSpec> {
        self.ch_groups.iter().filter(|g| g.region.is_aliphatic())
    }

    /// Return all aromatic `CH` groups (`PHE`, `TYR`, `HIS`, `TRP`).
    pub fn aromatic_groups(&self) -> impl Iterator<Item = &'static ChGroupSpec> {
        self.ch_groups
            .iter()
            .filter(|g| g.region == CarbonRegion::Aromatic)
    }

    /// True if `atom_a` and `atom_b` share a direct covalent bond in this residue.
    pub fn are_bonded(&self, atom_a: &str, atom_b: &str) -> bool {
        self.bonds
            .iter()
            .any(|&(u, v)| (u == atom_a && v == atom_b) || (u == atom_b && v == atom_a))
    }
}

// ============================================================================
// Static IUPAC / PDB CCD Topologies for All 20 Standard Amino Acids (ALA..VAL)
// ============================================================================

const TOPOLOGIES: [ResidueTopology; 20] = [
    // 1. ALA (A)
    ResidueTopology {
        three_letter: "ALA",
        one_letter: 'A',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB1", "CB", true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB1"),
            ("CB", "HB2"),
            ("CB", "HB3"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB1", "HB2", "HB3"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
        ],
        methyl_groups: &[MethylGroup {
            carbon: "CB",
            protons: ["HB1", "HB2", "HB3"],
        }],
        prochiral_groups: &[],
    },
    // 2. ARG (R)
    ResidueTopology {
        three_letter: "ARG",
        one_letter: 'R',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("CD", Element::C, true, false, true),
            proton("HD2", "CD", true, false, true),
            proton("HD3", "CD", true, false, true),
            heavy("NE", Element::N, false, false, true),
            proton("HE", "NE", false, false, true),
            heavy("CZ", Element::C, false, false, false),
            heavy("NH1", Element::N, false, false, false),
            proton("HH11", "NH1", false, false, false),
            proton("HH12", "NH1", false, false, false),
            heavy("NH2", Element::N, false, false, false),
            proton("HH21", "NH2", false, false, false),
            proton("HH22", "NH2", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "HD2"),
            ("CD", "HD3"),
            ("CD", "NE"),
            ("NE", "HE"),
            ("NE", "CZ"),
            ("CZ", "NH1"),
            ("NH1", "HH11"),
            ("NH1", "HH12"),
            ("CZ", "NH2"),
            ("NH2", "HH21"),
            ("NH2", "HH22"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB", "CD"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CD",
                protons: &["HD2", "HD3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG"],
                steps_from_ca: Some(3),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CD",
                protons: ["HD2", "HD3"],
            },
        ],
    },
    // 3. ASN (N)
    ResidueTopology {
        three_letter: "ASN",
        one_letter: 'N',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, true),
            heavy("OD1", Element::O, false, false, false),
            heavy("ND2", Element::N, false, false, true),
            proton("HD21", "ND2", false, false, true),
            proton("HD22", "ND2", false, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "OD1"),
            ("CG", "ND2"),
            ("ND2", "HD21"),
            ("ND2", "HD22"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 4. ASP (D)
    ResidueTopology {
        three_letter: "ASP",
        one_letter: 'D',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, true),
            heavy("OD1", Element::O, false, false, false),
            heavy("OD2", Element::O, false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "OD1"),
            ("CG", "OD2"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 5. CYS (C)
    ResidueTopology {
        three_letter: "CYS",
        one_letter: 'C',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("SG", Element::S, false, false, false),
            proton("HG", "SG", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "SG"),
            ("SG", "HG"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 6. GLN (Q)
    ResidueTopology {
        three_letter: "GLN",
        one_letter: 'Q',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("CD", Element::C, false, false, true),
            heavy("OE1", Element::O, false, false, false),
            heavy("NE2", Element::N, false, false, true),
            proton("HE21", "NE2", false, false, true),
            proton("HE22", "NE2", false, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "OE1"),
            ("CD", "NE2"),
            ("NE2", "HE21"),
            ("NE2", "HE22"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
        ],
    },
    // 7. GLU (E)
    ResidueTopology {
        three_letter: "GLU",
        one_letter: 'E',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("CD", Element::C, false, false, true),
            heavy("OE1", Element::O, false, false, false),
            heavy("OE2", Element::O, false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "OE1"),
            ("CD", "OE2"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
        ],
    },
    // 8. GLY (G)
    ResidueTopology {
        three_letter: "GLY",
        one_letter: 'G',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA2", "CA", true, false, true),
            proton("HA3", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA2"),
            ("CA", "HA3"),
            ("CA", "C"),
            ("C", "O"),
        ],
        ch_groups: &[ChGroupSpec {
            carbon: "CA",
            protons: &["HA2", "HA3"],
            multiplicity: ChMultiplicity::Methylene,
            region: CarbonRegion::AliphaticCaConnected,
            covalent_carbon_neighbors: &[],
            steps_from_ca: Some(0),
        }],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CA",
            protons: ["HA2", "HA3"],
        }],
    },
    // 9. HIS (H)
    ResidueTopology {
        three_letter: "HIS",
        one_letter: 'H',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, false),
            heavy("ND1", Element::N, false, false, true),
            proton("HD1", "ND1", false, false, false),
            heavy("CD2", Element::C, false, false, true),
            proton("HD2", "CD2", false, false, true),
            heavy("CE1", Element::C, false, false, true),
            proton("HE1", "CE1", false, false, true),
            heavy("NE2", Element::N, false, false, true),
            proton("HE2", "NE2", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "ND1"),
            ("ND1", "HD1"),
            ("ND1", "CE1"),
            ("CE1", "HE1"),
            ("CG", "CD2"),
            ("CD2", "HD2"),
            ("CD2", "NE2"),
            ("NE2", "HE2"),
            ("CE1", "NE2"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CD2",
                protons: &["HD2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &[],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE1",
                protons: &["HE1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &[],
                steps_from_ca: None,
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 10. ILE (I)
    ResidueTopology {
        three_letter: "ILE",
        one_letter: 'I',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB", "CB", true, false, true),
            heavy("CG1", Element::C, true, false, true),
            proton("HG12", "CG1", true, false, true),
            proton("HG13", "CG1", true, false, true),
            heavy("CG2", Element::C, true, false, true),
            proton("HG21", "CG2", true, false, true),
            proton("HG22", "CG2", true, false, true),
            proton("HG23", "CG2", true, false, true),
            heavy("CD1", Element::C, true, false, true),
            proton("HD11", "CD1", true, false, true),
            proton("HD12", "CD1", true, false, true),
            proton("HD13", "CD1", true, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB"),
            ("CB", "CG1"),
            ("CG1", "HG12"),
            ("CG1", "HG13"),
            ("CB", "CG2"),
            ("CG2", "HG21"),
            ("CG2", "HG22"),
            ("CG2", "HG23"),
            ("CG1", "CD1"),
            ("CD1", "HD11"),
            ("CD1", "HD12"),
            ("CD1", "HD13"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG1", "CG2"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG1",
                protons: &["HG12", "HG13"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB", "CD1"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CG2",
                protons: &["HG21", "HG22", "HG23"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CD1",
                protons: &["HD11", "HD12", "HD13"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG1"],
                steps_from_ca: Some(3),
            },
        ],
        methyl_groups: &[
            MethylGroup {
                carbon: "CG2",
                protons: ["HG21", "HG22", "HG23"],
            },
            MethylGroup {
                carbon: "CD1",
                protons: ["HD11", "HD12", "HD13"],
            },
        ],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CG1",
            protons: ["HG12", "HG13"],
        }],
    },
    // 11. LEU (L)
    ResidueTopology {
        three_letter: "LEU",
        one_letter: 'L',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG", "CG", true, false, true),
            heavy("CD1", Element::C, true, false, true),
            proton("HD11", "CD1", true, false, true),
            proton("HD12", "CD1", true, false, true),
            proton("HD13", "CD1", true, false, true),
            heavy("CD2", Element::C, true, false, true),
            proton("HD21", "CD2", true, false, true),
            proton("HD22", "CD2", true, false, true),
            proton("HD23", "CD2", true, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG"),
            ("CG", "CD1"),
            ("CD1", "HD11"),
            ("CD1", "HD12"),
            ("CD1", "HD13"),
            ("CG", "CD2"),
            ("CD2", "HD21"),
            ("CD2", "HD22"),
            ("CD2", "HD23"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB", "CD1", "CD2"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CD1",
                protons: &["HD11", "HD12", "HD13"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG"],
                steps_from_ca: Some(3),
            },
            ChGroupSpec {
                carbon: "CD2",
                protons: &["HD21", "HD22", "HD23"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG"],
                steps_from_ca: Some(3),
            },
        ],
        methyl_groups: &[
            MethylGroup {
                carbon: "CD1",
                protons: ["HD11", "HD12", "HD13"],
            },
            MethylGroup {
                carbon: "CD2",
                protons: ["HD21", "HD22", "HD23"],
            },
        ],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::IsopropylMethyls {
                carbon_a: "CD1",
                protons_a: ["HD11", "HD12", "HD13"],
                carbon_b: "CD2",
                protons_b: ["HD21", "HD22", "HD23"],
            },
        ],
    },
    // 12. LYS (K)
    ResidueTopology {
        three_letter: "LYS",
        one_letter: 'K',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("CD", Element::C, true, false, true),
            proton("HD2", "CD", true, false, true),
            proton("HD3", "CD", true, false, true),
            heavy("CE", Element::C, true, false, true),
            proton("HE2", "CE", true, false, true),
            proton("HE3", "CE", true, false, true),
            heavy("NZ", Element::N, false, false, true),
            proton("HZ1", "NZ", false, false, false),
            proton("HZ2", "NZ", false, false, false),
            proton("HZ3", "NZ", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "HD2"),
            ("CD", "HD3"),
            ("CD", "CE"),
            ("CE", "HE2"),
            ("CE", "HE3"),
            ("CE", "NZ"),
            ("NZ", "HZ1"),
            ("NZ", "HZ2"),
            ("NZ", "HZ3"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB", "CD"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CD",
                protons: &["HD2", "HD3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG", "CE"],
                steps_from_ca: Some(3),
            },
            ChGroupSpec {
                carbon: "CE",
                protons: &["HE2", "HE3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CD"],
                steps_from_ca: Some(4),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CD",
                protons: ["HD2", "HD3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CE",
                protons: ["HE2", "HE3"],
            },
        ],
    },
    // 13. MET (M)
    ResidueTopology {
        three_letter: "MET",
        one_letter: 'M',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("SD", Element::S, false, false, false),
            heavy("CE", Element::C, false, false, true),
            proton("HE1", "CE", false, false, true),
            proton("HE2", "CE", false, false, true),
            proton("HE3", "CE", false, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "SD"),
            ("SD", "CE"),
            ("CE", "HE1"),
            ("CE", "HE2"),
            ("CE", "HE3"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CE",
                protons: &["HE1", "HE2", "HE3"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticIsolatedMethyl,
                covalent_carbon_neighbors: &[],
                steps_from_ca: None,
            },
        ],
        methyl_groups: &[MethylGroup {
            carbon: "CE",
            protons: ["HE1", "HE2", "HE3"],
        }],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
        ],
    },
    // 14. PHE (F)
    ResidueTopology {
        three_letter: "PHE",
        one_letter: 'F',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, false),
            heavy("CD1", Element::C, false, false, true),
            proton("HD1", "CD1", false, false, true),
            heavy("CD2", Element::C, false, false, true),
            proton("HD2", "CD2", false, false, true),
            heavy("CE1", Element::C, false, false, true),
            proton("HE1", "CE1", false, false, true),
            heavy("CE2", Element::C, false, false, true),
            proton("HE2", "CE2", false, false, true),
            heavy("CZ", Element::C, false, false, true),
            proton("HZ", "CZ", false, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CD1", "HD1"),
            ("CD1", "CE1"),
            ("CE1", "HE1"),
            ("CE1", "CZ"),
            ("CG", "CD2"),
            ("CD2", "HD2"),
            ("CD2", "CE2"),
            ("CE2", "HE2"),
            ("CE2", "CZ"),
            ("CZ", "HZ"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CD1",
                protons: &["HD1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE1"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CD2",
                protons: &["HD2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE2"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE1",
                protons: &["HE1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CD1", "CZ"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE2",
                protons: &["HE2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CD2", "CZ"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CZ",
                protons: &["HZ"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE1", "CE2"],
                steps_from_ca: None,
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::SymmetricAromatic {
                carbon_a: "CD1",
                proton_a: "HD1",
                carbon_b: "CD2",
                proton_b: "HD2",
            },
            ProchiralGroup::SymmetricAromatic {
                carbon_a: "CE1",
                proton_a: "HE1",
                carbon_b: "CE2",
                proton_b: "HE2",
            },
        ],
    },
    // 15. PRO (P)
    ResidueTopology {
        three_letter: "PRO",
        one_letter: 'P',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, true, false, true),
            proton("HG2", "CG", true, false, true),
            proton("HG3", "CG", true, false, true),
            heavy("CD", Element::C, true, false, true),
            proton("HD2", "CD", true, false, true),
            proton("HD3", "CD", true, false, true),
        ],
        bonds: &[
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "HG2"),
            ("CG", "HG3"),
            ("CG", "CD"),
            ("CD", "HD2"),
            ("CD", "HD3"),
            ("CD", "N"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG",
                protons: &["HG2", "HG3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB", "CD"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CD",
                protons: &["HD2", "HD3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CG"],
                steps_from_ca: Some(3),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CG",
                protons: ["HG2", "HG3"],
            },
            ProchiralGroup::Methylene {
                carbon: "CD",
                protons: ["HD2", "HD3"],
            },
        ],
    },
    // 16. SER (S)
    ResidueTopology {
        three_letter: "SER",
        one_letter: 'S',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("OG", Element::O, false, false, false),
            proton("HG", "OG", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "OG"),
            ("OG", "HG"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 17. THR (T)
    ResidueTopology {
        three_letter: "THR",
        one_letter: 'T',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB", "CB", true, false, true),
            heavy("OG1", Element::O, false, false, false),
            proton("HG1", "OG1", false, false, false),
            heavy("CG2", Element::C, true, false, true),
            proton("HG21", "CG2", true, false, true),
            proton("HG22", "CG2", true, false, true),
            proton("HG23", "CG2", true, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB"),
            ("CB", "OG1"),
            ("OG1", "HG1"),
            ("CB", "CG2"),
            ("CG2", "HG21"),
            ("CG2", "HG22"),
            ("CG2", "HG23"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG2"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG2",
                protons: &["HG21", "HG22", "HG23"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
        ],
        methyl_groups: &[MethylGroup {
            carbon: "CG2",
            protons: ["HG21", "HG22", "HG23"],
        }],
        prochiral_groups: &[],
    },
    // 18. TRP (W)
    ResidueTopology {
        three_letter: "TRP",
        one_letter: 'W',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, false),
            heavy("CD1", Element::C, false, false, true),
            proton("HD1", "CD1", false, false, true),
            heavy("CD2", Element::C, false, false, false),
            heavy("NE1", Element::N, false, false, true),
            proton("HE1", "NE1", false, false, true),
            heavy("CE2", Element::C, false, false, false),
            heavy("CE3", Element::C, false, false, true),
            proton("HE3", "CE3", false, false, true),
            heavy("CZ2", Element::C, false, false, true),
            proton("HZ2", "CZ2", false, false, true),
            heavy("CZ3", Element::C, false, false, true),
            proton("HZ3", "CZ3", false, false, true),
            heavy("CH2", Element::C, false, false, true),
            proton("HH2", "CH2", false, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CD1", "HD1"),
            ("CD1", "NE1"),
            ("NE1", "HE1"),
            ("NE1", "CE2"),
            ("CG", "CD2"),
            ("CD2", "CE2"),
            ("CD2", "CE3"),
            ("CE3", "HE3"),
            ("CE3", "CZ3"),
            ("CZ3", "HZ3"),
            ("CE2", "CZ2"),
            ("CZ2", "HZ2"),
            ("CZ2", "CH2"),
            ("CZ3", "CH2"),
            ("CH2", "HH2"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CD1",
                protons: &["HD1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &[],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE3",
                protons: &["HE3"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CZ3"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CZ2",
                protons: &["HZ2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CH2"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CZ3",
                protons: &["HZ3"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE3", "CH2"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CH2",
                protons: &["HH2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CZ2", "CZ3"],
                steps_from_ca: None,
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[ProchiralGroup::Methylene {
            carbon: "CB",
            protons: ["HB2", "HB3"],
        }],
    },
    // 19. TYR (Y)
    ResidueTopology {
        three_letter: "TYR",
        one_letter: 'Y',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB2", "CB", true, false, true),
            proton("HB3", "CB", true, false, true),
            heavy("CG", Element::C, false, false, false),
            heavy("CD1", Element::C, false, false, true),
            proton("HD1", "CD1", false, false, true),
            heavy("CD2", Element::C, false, false, true),
            proton("HD2", "CD2", false, false, true),
            heavy("CE1", Element::C, false, false, true),
            proton("HE1", "CE1", false, false, true),
            heavy("CE2", Element::C, false, false, true),
            proton("HE2", "CE2", false, false, true),
            heavy("CZ", Element::C, false, false, false),
            heavy("OH", Element::O, false, false, false),
            proton("HH", "OH", false, false, false),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB2"),
            ("CB", "HB3"),
            ("CB", "CG"),
            ("CG", "CD1"),
            ("CD1", "HD1"),
            ("CD1", "CE1"),
            ("CE1", "HE1"),
            ("CE1", "CZ"),
            ("CG", "CD2"),
            ("CD2", "HD2"),
            ("CD2", "CE2"),
            ("CE2", "HE2"),
            ("CE2", "CZ"),
            ("CZ", "OH"),
            ("OH", "HH"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB2", "HB3"],
                multiplicity: ChMultiplicity::Methylene,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CD1",
                protons: &["HD1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE1"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CD2",
                protons: &["HD2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CE2"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE1",
                protons: &["HE1"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CD1"],
                steps_from_ca: None,
            },
            ChGroupSpec {
                carbon: "CE2",
                protons: &["HE2"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::Aromatic,
                covalent_carbon_neighbors: &["CD2"],
                steps_from_ca: None,
            },
        ],
        methyl_groups: &[],
        prochiral_groups: &[
            ProchiralGroup::Methylene {
                carbon: "CB",
                protons: ["HB2", "HB3"],
            },
            ProchiralGroup::SymmetricAromatic {
                carbon_a: "CD1",
                proton_a: "HD1",
                carbon_b: "CD2",
                proton_b: "HD2",
            },
            ProchiralGroup::SymmetricAromatic {
                carbon_a: "CE1",
                proton_a: "HE1",
                carbon_b: "CE2",
                proton_b: "HE2",
            },
        ],
    },
    // 20. VAL (V)
    ResidueTopology {
        three_letter: "VAL",
        one_letter: 'V',
        atoms: &[
            heavy("N", Element::N, false, true, true),
            proton("H", "N", false, true, true),
            heavy("CA", Element::C, true, true, true),
            proton("HA", "CA", true, false, true),
            heavy("C", Element::C, false, true, true),
            heavy("O", Element::O, false, true, false),
            heavy("CB", Element::C, true, false, true),
            proton("HB", "CB", true, false, true),
            heavy("CG1", Element::C, true, false, true),
            proton("HG11", "CG1", true, false, true),
            proton("HG12", "CG1", true, false, true),
            proton("HG13", "CG1", true, false, true),
            heavy("CG2", Element::C, true, false, true),
            proton("HG21", "CG2", true, false, true),
            proton("HG22", "CG2", true, false, true),
            proton("HG23", "CG2", true, false, true),
        ],
        bonds: &[
            ("N", "H"),
            ("N", "CA"),
            ("CA", "HA"),
            ("CA", "C"),
            ("C", "O"),
            ("CA", "CB"),
            ("CB", "HB"),
            ("CB", "CG1"),
            ("CG1", "HG11"),
            ("CG1", "HG12"),
            ("CG1", "HG13"),
            ("CB", "CG2"),
            ("CG2", "HG21"),
            ("CG2", "HG22"),
            ("CG2", "HG23"),
        ],
        ch_groups: &[
            ChGroupSpec {
                carbon: "CA",
                protons: &["HA"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(0),
            },
            ChGroupSpec {
                carbon: "CB",
                protons: &["HB"],
                multiplicity: ChMultiplicity::Methine,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CA", "CG1", "CG2"],
                steps_from_ca: Some(1),
            },
            ChGroupSpec {
                carbon: "CG1",
                protons: &["HG11", "HG12", "HG13"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
            ChGroupSpec {
                carbon: "CG2",
                protons: &["HG21", "HG22", "HG23"],
                multiplicity: ChMultiplicity::Methyl,
                region: CarbonRegion::AliphaticCaConnected,
                covalent_carbon_neighbors: &["CB"],
                steps_from_ca: Some(2),
            },
        ],
        methyl_groups: &[
            MethylGroup {
                carbon: "CG1",
                protons: ["HG11", "HG12", "HG13"],
            },
            MethylGroup {
                carbon: "CG2",
                protons: ["HG21", "HG22", "HG23"],
            },
        ],
        prochiral_groups: &[ProchiralGroup::IsopropylMethyls {
            carbon_a: "CG1",
            protons_a: ["HG11", "HG12", "HG13"],
            carbon_b: "CG2",
            protons_b: ["HG21", "HG22", "HG23"],
        }],
    },
];

/// Return the static table of all 20 standard amino-acid IUPAC topologies (`ALA..VAL`).
#[inline]
pub fn all_residue_topologies() -> &'static [ResidueTopology; 20] {
    &TOPOLOGIES
}

/// Look up a standard amino-acid topology by 3-letter (`"ILE"`) or 1-letter (`"I"`) code.
pub fn residue_topology(res: &str) -> Option<&'static ResidueTopology> {
    let trimmed = res.trim();
    if trimmed.len() == 1 {
        let ch = trimmed.chars().next()?.to_ascii_uppercase();
        return TOPOLOGIES.iter().find(|t| t.one_letter == ch);
    }
    let up = trimmed.to_ascii_uppercase();
    TOPOLOGIES.iter().find(|t| t.three_letter == up)
}

// ============================================================================
// Empirical BMRB All-Atom Chemical Shift Priors (30 Disjoint Public BMRB Entries)
// ============================================================================

#[allow(clippy::type_complexity)]
const RAW_BMRB_ALL_ATOM_PRIORS: &[(&str, &[(&str, (f64, f64))])] = &[
    (
        "ALA",
        &[
            ("H", (8.193, 0.624)),
            ("N", (123.418, 3.898)),
            ("C", (177.781, 2.299)),
            ("CA", (53.330, 2.022)),
            ("HA", (4.212, 0.365)),
            ("CB", (18.835, 1.201)),
            ("HB1", (1.396, 0.240)),
            ("HB2", (1.396, 0.240)),
            ("HB3", (1.396, 0.240)),
        ],
    ),
    (
        "ARG",
        &[
            ("H", (8.361, 0.725)),
            ("N", (121.561, 4.518)),
            ("C", (175.995, 2.011)),
            ("CA", (56.812, 2.455)),
            ("HA", (4.287, 0.479)),
            ("CB", (30.756, 1.685)),
            ("HB2", (1.773, 0.250)),
            ("HB3", (1.731, 0.260)),
            ("CG", (27.422, 1.050)),
            ("HG2", (1.553, 0.260)),
            ("HG3", (1.537, 0.274)),
            ("CD", (43.244, 0.850)),
            ("HD2", (3.152, 0.240)),
            ("HD3", (3.092, 0.240)),
            ("NE", (84.741, 1.959)),
            ("HE", (7.177, 0.366)),
        ],
    ),
    (
        "ASN",
        &[
            ("H", (8.439, 0.628)),
            ("N", (119.407, 4.257)),
            ("C", (175.213, 1.600)),
            ("CA", (53.504, 1.669)),
            ("HA", (4.719, 0.414)),
            ("CB", (38.799, 1.765)),
            ("HB2", (2.826, 0.279)),
            ("HB3", (2.725, 0.250)),
            ("CG", (176.996, 0.950)),
            ("ND2", (112.749, 2.637)),
            ("HD21", (7.434, 0.367)),
            ("HD22", (6.998, 0.372)),
        ],
    ),
    (
        "ASP",
        &[
            ("H", (8.359, 0.614)),
            ("N", (121.567, 4.366)),
            ("C", (176.423, 1.769)),
            ("CA", (54.394, 1.941)),
            ("HA", (4.641, 0.313)),
            ("CB", (41.030, 1.694)),
            ("HB2", (2.724, 0.250)),
            ("HB3", (2.655, 0.250)),
            ("CG", (179.712, 1.021)),
        ],
    ),
    (
        "CYS",
        &[
            ("H", (8.406, 0.833)),
            ("N", (122.023, 4.140)),
            ("C", (175.415, 2.373)),
            ("CA", (57.401, 3.707)),
            ("HA", (4.640, 0.724)),
            ("CB", (35.429, 7.115)),
            ("HB2", (2.953, 0.479)),
            ("HB3", (2.891, 0.420)),
        ],
    ),
    (
        "GLN",
        &[
            ("H", (8.073, 0.703)),
            ("N", (119.818, 4.011)),
            ("C", (176.577, 1.974)),
            ("CA", (56.361, 2.195)),
            ("HA", (4.281, 0.387)),
            ("CB", (29.652, 1.670)),
            ("HB2", (2.027, 0.260)),
            ("HB3", (2.016, 0.260)),
            ("CG", (33.971, 0.900)),
            ("HG2", (2.338, 0.250)),
            ("HG3", (2.319, 0.250)),
            ("CD", (179.909, 0.850)),
            ("NE2", (112.324, 1.591)),
            ("HE21", (7.320, 0.384)),
            ("HE22", (6.868, 0.277)),
        ],
    ),
    (
        "GLU",
        &[
            ("H", (8.424, 0.632)),
            ("N", (121.021, 4.118)),
            ("C", (176.890, 2.033)),
            ("CA", (57.321, 2.192)),
            ("HA", (4.251, 0.426)),
            ("CB", (30.099, 1.829)),
            ("HB2", (2.030, 0.240)),
            ("HB3", (1.991, 0.240)),
            ("CG", (36.377, 0.900)),
            ("HG2", (2.256, 0.240)),
            ("HG3", (2.238, 0.240)),
            ("CD", (182.951, 0.892)),
        ],
    ),
    (
        "GLY",
        &[
            ("H", (8.360, 0.707)),
            ("N", (109.893, 4.193)),
            ("C", (174.162, 1.752)),
            ("CA", (45.544, 1.243)),
            ("HA2", (3.964, 0.340)),
            ("HA3", (3.897, 0.308)),
        ],
    ),
    (
        "HIS",
        &[
            ("H", (8.327, 0.929)),
            ("N", (119.702, 5.183)),
            ("C", (175.017, 2.435)),
            ("CA", (56.051, 2.632)),
            ("HA", (4.775, 0.619)),
            ("CB", (31.085, 2.783)),
            ("HB2", (3.100, 0.369)),
            ("HB3", (3.002, 0.339)),
            ("CD2", (119.179, 1.844)),
            ("HD2", (7.023, 0.320)),
            ("CE1", (138.242, 1.400)),
            ("HE1", (7.849, 0.380)),
            ("ND1", (205.105, 26.609)),
            ("NE2", (171.143, 8.028)),
        ],
    ),
    (
        "ILE",
        &[
            ("H", (8.313, 0.724)),
            ("N", (122.199, 4.980)),
            ("C", (175.602, 1.884)),
            ("CA", (61.285, 2.629)),
            ("HA", (4.282, 0.629)),
            ("CB", (39.005, 1.969)),
            ("HB", (1.758, 0.295)),
            ("CG1", (27.777, 1.373)),
            ("HG12", (1.272, 0.428)),
            ("HG13", (1.103, 0.377)),
            ("CG2", (17.608, 0.980)),
            ("HG21", (0.773, 0.250)),
            ("HG22", (0.773, 0.250)),
            ("HG23", (0.773, 0.250)),
            ("CD1", (13.861, 1.593)),
            ("HD11", (0.719, 0.250)),
            ("HD12", (0.719, 0.250)),
            ("HD13", (0.719, 0.250)),
        ],
    ),
    (
        "LEU",
        &[
            ("H", (8.207, 0.714)),
            ("N", (122.185, 4.582)),
            ("C", (176.808, 2.091)),
            ("CA", (55.259, 2.095)),
            ("HA", (4.360, 0.507)),
            ("CB", (42.622, 1.880)),
            ("HB2", (1.596, 0.327)),
            ("HB3", (1.511, 0.335)),
            ("CG", (26.917, 0.950)),
            ("HG", (1.493, 0.294)),
            ("CD1", (24.861, 1.550)),
            ("HD11", (0.770, 0.250)),
            ("HD12", (0.770, 0.250)),
            ("HD13", (0.770, 0.250)),
            ("CD2", (24.253, 1.658)),
            ("HD21", (0.721, 0.260)),
            ("HD22", (0.721, 0.260)),
            ("HD23", (0.721, 0.260)),
        ],
    ),
    (
        "LYS",
        &[
            ("H", (8.246, 0.648)),
            ("N", (121.342, 4.295)),
            ("C", (176.459, 2.121)),
            ("CA", (56.678, 2.284)),
            ("HA", (4.367, 0.476)),
            ("CB", (33.023, 1.801)),
            ("HB2", (1.800, 0.240)),
            ("HB3", (1.770, 0.240)),
            ("CG", (25.068, 0.850)),
            ("HG2", (1.403, 0.240)),
            ("HG3", (1.386, 0.240)),
            ("CD", (29.221, 0.850)),
            ("HD2", (1.621, 0.240)),
            ("HD3", (1.601, 0.240)),
            ("CE", (42.046, 0.850)),
            ("HE2", (2.938, 0.220)),
            ("HE3", (2.909, 0.220)),
            ("NZ", (32.882, 1.200)),
        ],
    ),
    (
        "MET",
        &[
            ("H", (8.306, 0.718)),
            ("N", (120.616, 4.326)),
            ("C", (175.639, 2.349)),
            ("CA", (55.766, 2.167)),
            ("HA", (4.554, 0.569)),
            ("CB", (33.419, 1.743)),
            ("HB2", (1.979, 0.260)),
            ("HB3", (1.904, 0.343)),
            ("CG", (32.154, 1.050)),
            ("HG2", (2.413, 0.295)),
            ("HG3", (2.295, 0.350)),
            ("CE", (16.910, 1.100)),
            ("HE1", (1.880, 0.294)),
            ("HE2", (1.880, 0.294)),
            ("HE3", (1.880, 0.294)),
        ],
    ),
    (
        "PHE",
        &[
            ("H", (8.409, 0.802)),
            ("N", (121.046, 4.413)),
            ("C", (175.294, 1.939)),
            ("CA", (57.368, 2.401)),
            ("HA", (4.873, 0.643)),
            ("CB", (40.448, 2.126)),
            ("HB2", (3.021, 0.364)),
            ("HB3", (2.967, 0.335)),
            ("CD1", (131.350, 1.150)),
            ("HD1", (7.100, 0.280)),
            ("CD2", (131.350, 1.150)),
            ("HD2", (7.100, 0.280)),
            ("CE1", (130.630, 1.200)),
            ("HE1", (7.015, 0.330)),
            ("CE2", (130.630, 1.200)),
            ("HE2", (7.015, 0.330)),
            ("CZ", (129.310, 1.278)),
            ("HZ", (6.927, 0.380)),
        ],
    ),
    (
        "PRO",
        &[
            ("N", (134.500, 5.500)),
            ("C", (177.106, 2.007)),
            ("CA", (63.411, 1.583)),
            ("HA", (4.383, 0.301)),
            ("CB", (31.958, 0.950)),
            ("HB2", (2.141, 0.268)),
            ("HB3", (1.988, 0.325)),
            ("CG", (27.652, 0.950)),
            ("HG2", (1.959, 0.240)),
            ("HG3", (1.972, 0.240)),
            ("CD", (50.524, 0.900)),
            ("HD2", (3.756, 0.250)),
            ("HD3", (3.624, 0.298)),
        ],
    ),
    (
        "SER",
        &[
            ("H", (8.320, 0.610)),
            ("N", (116.672, 4.145)),
            ("C", (174.538, 1.765)),
            ("CA", (58.747, 2.246)),
            ("HA", (4.525, 0.458)),
            ("CB", (64.014, 1.348)),
            ("HB2", (3.842, 0.240)),
            ("HB3", (3.863, 0.240)),
        ],
    ),
    (
        "THR",
        &[
            ("H", (8.326, 0.661)),
            ("N", (116.124, 5.369)),
            ("C", (174.464, 1.616)),
            ("CA", (62.128, 2.483)),
            ("HA", (4.612, 0.539)),
            ("CB", (69.962, 1.449)),
            ("HB", (4.129, 0.319)),
            ("CG2", (21.624, 0.900)),
            ("HG21", (1.130, 0.220)),
            ("HG22", (1.130, 0.220)),
            ("HG23", (1.130, 0.220)),
        ],
    ),
    (
        "TRP",
        &[
            ("H", (8.382, 0.988)),
            ("N", (123.185, 4.657)),
            ("C", (176.304, 1.650)),
            ("CA", (57.378, 2.374)),
            ("HA", (4.795, 0.599)),
            ("CB", (30.530, 1.789)),
            ("HB2", (3.192, 0.279)),
            ("HB3", (3.166, 0.260)),
            ("CD1", (126.403, 1.651)),
            ("HD1", (7.086, 0.315)),
            ("NE1", (129.080, 1.854)),
            ("HE1", (10.048, 0.345)),
            ("CE3", (120.000, 1.100)),
            ("HE3", (7.297, 0.309)),
            ("CZ2", (114.375, 1.100)),
            ("HZ2", (7.244, 0.280)),
            ("CZ3", (121.560, 1.000)),
            ("HZ3", (6.817, 0.380)),
            ("CH2", (123.882, 1.199)),
            ("HH2", (6.928, 0.312)),
        ],
    ),
    (
        "TYR",
        &[
            ("H", (8.544, 0.737)),
            ("N", (121.298, 4.877)),
            ("C", (175.106, 1.941)),
            ("CA", (57.720, 2.390)),
            ("HA", (4.800, 0.598)),
            ("CB", (39.875, 2.011)),
            ("HB2", (2.965, 0.287)),
            ("HB3", (2.828, 0.261)),
            ("CD1", (132.770, 0.950)),
            ("HD1", (6.930, 0.280)),
            ("CD2", (132.770, 0.950)),
            ("HD2", (6.930, 0.280)),
            ("CE1", (117.930, 0.900)),
            ("HE1", (6.700, 0.260)),
            ("CE2", (117.930, 0.900)),
            ("HE2", (6.700, 0.260)),
        ],
    ),
    (
        "VAL",
        &[
            ("H", (8.321, 0.732)),
            ("N", (122.025, 5.364)),
            ("C", (175.382, 1.859)),
            ("CA", (61.927, 2.628)),
            ("HA", (4.312, 0.554)),
            ("CB", (33.051, 1.740)),
            ("HB", (1.947, 0.261)),
            ("CG1", (21.471, 1.300)),
            ("HG11", (0.796, 0.250)),
            ("HG12", (0.796, 0.250)),
            ("HG13", (0.796, 0.250)),
            ("CG2", (21.234, 1.322)),
            ("HG21", (0.774, 0.250)),
            ("HG22", (0.774, 0.250)),
            ("HG23", (0.774, 0.250)),
        ],
    ),
];

#[inline]
fn gaussian_logpdf(v: f64, mu: f64, sd: f64) -> f64 {
    let z = (v - mu) / sd;
    -0.5 * z * z - (sd * TAU.sqrt()).ln()
}

fn log_mean_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let m = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = vals.iter().map(|&x| (x - m).exp()).sum();
    m + (s / (vals.len() as f64)).ln()
}

/// Empirical BMRB chemical shift priors `(mean_ppm, std_ppm)` across all 20 amino acids
/// for backbone, aliphatic sidechain, and aromatic sidechain `13C`, `1H`, and `15N` atoms.
#[derive(Clone, Debug)]
pub struct SidechainPriors {
    pub table: BTreeMap<String, BTreeMap<String, (f64, f64)>>,
    aliphatic_c_components: Vec<(f64, f64)>,
    aliphatic_h_components: Vec<(f64, f64)>,
    aromatic_c_components: Vec<(f64, f64)>,
    aromatic_h_components: Vec<(f64, f64)>,
}

impl Default for SidechainPriors {
    fn default() -> Self {
        Self::empirical_bmrb()
    }
}

impl SidechainPriors {
    /// Construct empirical BMRB priors from the disjoint 30-entry public BMRB statistics.
    pub fn empirical_bmrb() -> Self {
        let mut table: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
        for &(res, atoms) in RAW_BMRB_ALL_ATOM_PRIORS {
            let mut m = BTreeMap::new();
            for &(atom, pair) in atoms {
                m.insert(atom.to_string(), pair);
            }
            table.insert(res.to_string(), m);
        }

        let mut aliphatic_c_components = Vec::new();
        let mut aliphatic_h_components = Vec::new();
        let mut aromatic_c_components = Vec::new();
        let mut aromatic_h_components = Vec::new();

        for topo in &TOPOLOGIES {
            if let Some(rmap) = table.get(topo.three_letter) {
                for chg in topo.ch_groups {
                    if let Some(&cp) = rmap.get(chg.carbon) {
                        if chg.region.is_aliphatic() {
                            aliphatic_c_components.push(cp);
                        } else {
                            aromatic_c_components.push(cp);
                        }
                    }
                    // Weight each CH_n group once in the proton background mixture so methyls
                    // with 3 identical proton names do not 3x-skew the background density.
                    if let Some(&first_h) = chg.protons.first() {
                        if let Some(&hp) = rmap.get(first_h) {
                            if chg.region.is_aliphatic() {
                                aliphatic_h_components.push(hp);
                            } else {
                                aromatic_h_components.push(hp);
                            }
                        }
                    }
                }
            }
        }

        Self {
            table,
            aliphatic_c_components,
            aliphatic_h_components,
            aromatic_c_components,
            aromatic_h_components,
        }
    }

    /// Look up `(mean_ppm, std_ppm)` for `(residue_3letter, iupac_atom)`.
    #[inline]
    pub fn prior(&self, res: &str, atom: &str) -> Option<(f64, f64)> {
        self.table.get(res)?.get(atom).copied()
    }

    /// Gaussian log-density `log N(v | mu, sd)` for `(res, atom)`.
    #[inline]
    pub fn logpdf(&self, res: &str, atom: &str, v: f64) -> Option<f64> {
        let (mu, sd) = self.prior(res, atom)?;
        Some(gaussian_logpdf(v, mu, sd))
    }

    /// Log background mixture density for a `13C` shift in the specified `CarbonRegion`.
    pub fn background_c13(&self, region: CarbonRegion, v: f64) -> f64 {
        let comps = if region.is_aliphatic() {
            &self.aliphatic_c_components
        } else {
            &self.aromatic_c_components
        };
        let vals: Vec<f64> = comps
            .iter()
            .map(|&(mu, sd)| gaussian_logpdf(v, mu, sd))
            .collect();
        log_mean_exp(&vals)
    }

    /// Log background mixture density for a `1H` shift in the specified `CarbonRegion`.
    pub fn background_h1(&self, region: CarbonRegion, v: f64) -> f64 {
        let comps = if region.is_aliphatic() {
            &self.aliphatic_h_components
        } else {
            &self.aromatic_h_components
        };
        let vals: Vec<f64> = comps
            .iter()
            .map(|&(mu, sd)| gaussian_logpdf(v, mu, sd))
            .collect();
        log_mean_exp(&vals)
    }

    /// Log-likelihood ratio `log N(v | mu, sd) - log B(v)` floored at `floor` (`-10.0`).
    pub fn llr_c13(&self, res: &str, carbon: &str, region: CarbonRegion, v: f64) -> f64 {
        let Some(lp) = self.logpdf(res, carbon, v) else {
            return -10.0;
        };
        (lp - self.background_c13(region, v)).max(-10.0)
    }

    /// Log-likelihood ratio `log N(v | mu, sd) - log B(v)` for an attached proton, floored at `-10.0`.
    pub fn llr_h1(&self, res: &str, proton: &str, region: CarbonRegion, v: f64) -> f64 {
        let Some(lp) = self.logpdf(res, proton, v) else {
            return -10.0;
        };
        (lp - self.background_h1(region, v)).max(-10.0)
    }
}
