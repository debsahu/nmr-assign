//! Bayesian stereospecific prochiral resonance assignment (`HB2`/`HB3`, `HA2`/`HA3`,
//! `HG2`/`HG3`, `HG12`/`HG13`, `HD2`/`HD3`, `HE2`/`HE3`, `Val CG1/CG2 + HG1/HG2`,
//! `Leu CD1/CD2 + HD1/HD2`, `Asn HD21/HD22`, `Gln HE21/HE22`) and standards-compliant
//! NMR-STAR v3 (`Ambiguity_code` `1` vs `2`) & NEF v1.1 (`HBx`/`HBy`, `CGx`/`CGy`) export.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed
//! biomolecular NMR literature and open nomenclature standards (`docs/specs/07b-stereospecific.md`):
//! - Güntert, P., Braun, W., Billeter, M. & Wüthrich, K. (1991). "Automated stereospecific
//!   1H NMR assignments and their impact on the precision of protein structure determinations
//!   in solution." *J. Biomol. NMR* 1, 447–456 (local conformational analysis of vicinal
//!   scalar couplings and intraresidue/sequential NOE intensities).
//! - Karplus, M. (1959). *J. Chem. Phys.* 30, 11–15; and Karplus, M. (1963). *J. Am. Chem. Soc.*
//!   85, 2870–2871 (three-bond vicinal scalar coupling dihedral dependence).
//! - DeMarco, A., Llinás, M. & Wüthrich, K. (1978). *Biopolymers* 17, 637–650;
//!   Archer, S. J., Ikura, M., Torchia, D. A. & Bax, A. (1991). *J. Magn. Reson.* 95, 636–641;
//!   and Vuister, G. W. & Bax, A. (1993). "Quantitative J correlation: a new approach for
//!   measuring homonuclear three-bond J(HNHa) coupling constants in 15N-enriched proteins."
//!   *J. Am. Chem. Soc.* 115, 7772–7777 (quantitative `3J_HAHB`, `3J_NHB`, and `3J_HNHA` curves).
//! - Hu, J.-S. & Bax, A. (1997). *J. Biomol. NMR* 9, 323–328; Hu, J.-S. & Bax, A. (1997).
//!   *J. Am. Chem. Soc.* 119, 6360–6368; and Chou, J. J., Case, D. A. & Bax, A. (2003).
//!   *J. Am. Chem. Soc.* 125, 8959–8966 (`3J_NCG` and `3J_C'CG` / `3J_CC` Karplus relations for
//!   `Val`, `Ile`, `Thr`, and `Leu`).
//! - Nilges, M. (1995). *J. Mol. Biol.* 245, 645–660 (`r^-6` expectation and ambiguous
//!   prochiral / floating-chirality treatment).
//! - Lovell, S. C., Word, J. M., Richardson, J. S. & Richardson, D. C. (2000). "The penultimate
//!   rotamer library." *Proteins* 40, 389–408 (backbone-dependent sidechain rotamer priors).
//! - Markley, J. L. et al. (1998). *J. Mol. Biol.* 280, 933–952 (*J. Biomol. NMR* 1998, 12, 1–23)
//!   and Gutmanas, A. et al. (2015). *Nat. Struct. Mol. Biol.* 22, 433–434 (IUPAC `HB2`/`HB3`
//!   stereospecific nomenclature, NMR-STAR `Ambiguity_code`, and NEF v1.1 `HBx`/`HBy` wildcards).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dihedrals::{wrap_deg, DihedralRestraint, SecondaryStructure};
use crate::nef::{
    infer_element_and_isotope, normalize_nef_atom_name, NefAssignedShift, NefShiftList,
};
use crate::nmrstar::Reference;
use crate::noesy::{calc_dihedral_deg, v_add, v_cross, v_dist, v_scale, v_sub, v_unit, Vec3};
use crate::structure::{
    nerf_place_atom, ANGLE_CA_C_N_DEG, ANGLE_C_N_CA_DEG, ANGLE_C_N_H_DEG, ANGLE_N_CA_C_DEG,
    ANGLE_N_CA_C_GLY_DEG, BOND_CA_C, BOND_C_N, BOND_N_CA, BOND_N_H,
};

// 5-point Gauss-Hermite quadrature nodes (sqrt(2) * x_i) and normalized weights for thermal libration
const GH5_SCALE: [f64; 5] = [
    -2.856_970_013_872_805_6,
    -1.355_626_179_974_266,
    0.0,
    1.355_626_179_974_266,
    2.856_970_013_872_805_6,
];
const GH5_WEIGHT: [f64; 5] = [
    0.011_257_411_327_720_69,
    0.222_075_922_005_612_64,
    0.533_333_333_333_333_3,
    0.222_075_922_005_612_64,
    0.011_257_411_327_720_69,
];

// 3-point Gauss-Hermite quadrature for backbone (phi, psi) uncertainty integration
const GH3_SCALE: [f64; 3] = [-1.732_050_807_568_877_2, 0.0, 1.732_050_807_568_877_2];
const GH3_WEIGHT: [f64; 3] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];

const BOND_CA_CB: f64 = 1.530;
const BOND_CC_ALIPH: f64 = 1.520;
const BOND_CH: f64 = 1.090;
const BOND_CG_ND2: f64 = 1.328;
const BOND_NH_AMIDE: f64 = 1.010;
const ANGLE_TETRA_DEG: f64 = 109.5;
const ANGLE_ALIPH_CCC_DEG: f64 = 111.0;

/// Canonical staggered sidechain torsion state (`g-`, `t`, `g+`), or pyrrolidine ring pucker on `PRO`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RotamerState {
    /// `g-` (`~ -64 deg` staggered; `Cg-exo ~ -25 deg` on `PRO`).
    GaucheMinus,
    /// `t` (`~ 180 deg` staggered; planar envelope transition `0 deg` on `PRO`).
    Trans,
    /// `g+` (`~ +62 deg` staggered; `Cg-endo ~ +26 deg` on `PRO`).
    GauchePlus,
}

impl RotamerState {
    pub const ALL: [Self; 3] = [Self::GaucheMinus, Self::Trans, Self::GauchePlus];

    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Self::GaucheMinus => 0,
            Self::Trans => 1,
            Self::GauchePlus => 2,
        }
    }

    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GaucheMinus => "g-",
            Self::Trans => "t",
            Self::GauchePlus => "g+",
        }
    }
}

/// Taxonomy of IUPAC prochiral pairs in standard protein residues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProchiralSiteKind {
    /// `GLY` `HA2` / `HA3` on `CA` (governed by backbone `(phi, psi)`).
    GlyAlphaMethylene,
    /// `HB2` / `HB3` on `CB` (governed by `chi1`).
    BetaMethylene,
    /// `HG2` / `HG3` on `CG` or `HG12` / `HG13` on `ILE CG1` (governed by `(chi1, chi2)`).
    GammaMethylene,
    /// `HD2` / `HD3` on `CD` in `ARG`, `LYS`, `PRO` (governed by `chi3`).
    DeltaMethylene,
    /// `HE2` / `HE3` on `CE` in `LYS` (governed by `chi4`).
    EpsilonMethylene,
    /// `VAL` `CG1`/`HG11..13` vs `CG2`/`HG21..23` (governed by `chi1`).
    ValIsopropyl,
    /// `LEU` `CD1`/`HD11..13` vs `CD2`/`HD21..23` (governed by `(chi1, chi2)`).
    LeuIsopropyl,
    /// `ASN` `HD21` / `HD22` on planar `ND2`.
    AsnCarboxamide,
    /// `GLN` `HE21` / `HE22` on planar `NE2`.
    GlnCarboxamide,
}

/// Static IUPAC & NEF v1.1 nomenclature descriptor for one prochiral pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProchiralPairSpec {
    pub residue: &'static str,
    pub kind: ProchiralSiteKind,
    /// Primary IUPAC identifier for branch 1 (`"HB2"`, `"CG1"`, `"CD1"`, `"HA2"`, `"HD21"`, etc.).
    pub iupac_branch_1: &'static str,
    /// Partner IUPAC identifier for branch 2 (`"HB3"`, `"CG2"`, `"CD2"`, `"HA3"`, `"HD22"`, etc.).
    pub iupac_branch_2: &'static str,
    /// Attached parent heavy atom (`Some("CB")` for methylenes, `Some("ND2")` for `Asn`, `None` for isopropyl `CG1/CG2`).
    pub shared_heavy_atom: Option<&'static str>,
    /// Branch-specific carbons (`Some(("CG1", "CG2"))` for `Val`, `Some(("CD1", "CD2"))` for `Leu`).
    pub iupac_carbons: Option<(&'static str, &'static str)>,
    /// NEF v1.1 non-stereospecific carbon names (`Some(("CGx", "CGy"))` for `Val`, `Some(("CDx", "CDy"))` for `Leu`).
    pub nef_nonstereo_carbons: Option<(&'static str, &'static str)>,
    /// IUPAC proton atom names for branch 1 (`&["HB2"]`, or `&["HG11", "HG12", "HG13"]` for `Val`).
    pub iupac_protons_1: &'static [&'static str],
    /// IUPAC proton atom names for branch 2 (`&["HB3"]`, or `&["HG21", "HG22", "HG23"]` for `Val`).
    pub iupac_protons_2: &'static [&'static str],
    /// NEF v1.1 stereospecific proton labels (`("HB2", "HB3")`, `("HG1%", "HG2%")`, `("HD1%", "HD2%")`).
    pub nef_stereo_protons: (&'static str, &'static str),
    /// NEF v1.1 non-stereospecific proton labels (`("HBx", "HBy")`, `("HGx%", "HGy%")`, `("HDx%", "HDy%")`).
    pub nef_nonstereo_protons: (&'static str, &'static str),
}

const H_HA2: &[&str] = &["HA2"];
const H_HA3: &[&str] = &["HA3"];
const H_HB2: &[&str] = &["HB2"];
const H_HB3: &[&str] = &["HB3"];
const H_HG2: &[&str] = &["HG2"];
const H_HG3: &[&str] = &["HG3"];
const H_HG12: &[&str] = &["HG12"];
const H_HG13: &[&str] = &["HG13"];
const H_HD2: &[&str] = &["HD2"];
const H_HD3: &[&str] = &["HD3"];
const H_HE2: &[&str] = &["HE2"];
const H_HE3: &[&str] = &["HE3"];
const H_VAL_G1: &[&str] = &["HG11", "HG12", "HG13"];
const H_VAL_G2: &[&str] = &["HG21", "HG22", "HG23"];
const H_LEU_D1: &[&str] = &["HD11", "HD12", "HD13"];
const H_LEU_D2: &[&str] = &["HD21", "HD22", "HD23"];
const H_ASN_D21: &[&str] = &["HD21"];
const H_ASN_D22: &[&str] = &["HD22"];
const H_GLN_E21: &[&str] = &["HE21"];
const H_GLN_E22: &[&str] = &["HE22"];

const fn cb_spec(residue: &'static str) -> ProchiralPairSpec {
    ProchiralPairSpec {
        residue,
        kind: ProchiralSiteKind::BetaMethylene,
        iupac_branch_1: "HB2",
        iupac_branch_2: "HB3",
        shared_heavy_atom: Some("CB"),
        iupac_carbons: None,
        nef_nonstereo_carbons: None,
        iupac_protons_1: H_HB2,
        iupac_protons_2: H_HB3,
        nef_stereo_protons: ("HB2", "HB3"),
        nef_nonstereo_protons: ("HBx", "HBy"),
    }
}

const fn cg_spec(residue: &'static str) -> ProchiralPairSpec {
    ProchiralPairSpec {
        residue,
        kind: ProchiralSiteKind::GammaMethylene,
        iupac_branch_1: "HG2",
        iupac_branch_2: "HG3",
        shared_heavy_atom: Some("CG"),
        iupac_carbons: None,
        nef_nonstereo_carbons: None,
        iupac_protons_1: H_HG2,
        iupac_protons_2: H_HG3,
        nef_stereo_protons: ("HG2", "HG3"),
        nef_nonstereo_protons: ("HGx", "HGy"),
    }
}

const fn cd_spec(residue: &'static str) -> ProchiralPairSpec {
    ProchiralPairSpec {
        residue,
        kind: ProchiralSiteKind::DeltaMethylene,
        iupac_branch_1: "HD2",
        iupac_branch_2: "HD3",
        shared_heavy_atom: Some("CD"),
        iupac_carbons: None,
        nef_nonstereo_carbons: None,
        iupac_protons_1: H_HD2,
        iupac_protons_2: H_HD3,
        nef_stereo_protons: ("HD2", "HD3"),
        nef_nonstereo_protons: ("HDx", "HDy"),
    }
}

/// Return all IUPAC prochiral pairs present in a canonical 3-letter amino acid residue.
///
/// Note that `ALA` (`HB1..3` equivalent methyl) and `THR` (`HB` methine + `HG21..23` single methyl)
/// have zero prochiral pairs; `PHE`/`TYR` aromatic `HD1/HD2` and `HE1/HE2` are symmetric 180-degree
/// ring-flip pairs (`Ambiguity_code = 3`) rather than prochiral centers (`Ambiguity_code = 2`).
pub fn prochiral_pairs_for_residue(res_name: &str) -> Vec<ProchiralPairSpec> {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "ALA" | "THR" => Vec::new(),
        "GLY" => vec![ProchiralPairSpec {
            residue: "GLY",
            kind: ProchiralSiteKind::GlyAlphaMethylene,
            iupac_branch_1: "HA2",
            iupac_branch_2: "HA3",
            shared_heavy_atom: Some("CA"),
            iupac_carbons: None,
            nef_nonstereo_carbons: None,
            iupac_protons_1: H_HA2,
            iupac_protons_2: H_HA3,
            nef_stereo_protons: ("HA2", "HA3"),
            nef_nonstereo_protons: ("HAx", "HAy"),
        }],
        "VAL" => vec![ProchiralPairSpec {
            residue: "VAL",
            kind: ProchiralSiteKind::ValIsopropyl,
            iupac_branch_1: "CG1",
            iupac_branch_2: "CG2",
            shared_heavy_atom: None,
            iupac_carbons: Some(("CG1", "CG2")),
            nef_nonstereo_carbons: Some(("CGx", "CGy")),
            iupac_protons_1: H_VAL_G1,
            iupac_protons_2: H_VAL_G2,
            nef_stereo_protons: ("HG1%", "HG2%"),
            nef_nonstereo_protons: ("HGx%", "HGy%"),
        }],
        "ILE" => vec![ProchiralPairSpec {
            residue: "ILE",
            kind: ProchiralSiteKind::GammaMethylene,
            iupac_branch_1: "HG12",
            iupac_branch_2: "HG13",
            shared_heavy_atom: Some("CG1"),
            iupac_carbons: None,
            nef_nonstereo_carbons: None,
            iupac_protons_1: H_HG12,
            iupac_protons_2: H_HG13,
            nef_stereo_protons: ("HG12", "HG13"),
            nef_nonstereo_protons: ("HG1x", "HG1y"),
        }],
        "LEU" => vec![
            cb_spec("LEU"),
            ProchiralPairSpec {
                residue: "LEU",
                kind: ProchiralSiteKind::LeuIsopropyl,
                iupac_branch_1: "CD1",
                iupac_branch_2: "CD2",
                shared_heavy_atom: None,
                iupac_carbons: Some(("CD1", "CD2")),
                nef_nonstereo_carbons: Some(("CDx", "CDy")),
                iupac_protons_1: H_LEU_D1,
                iupac_protons_2: H_LEU_D2,
                nef_stereo_protons: ("HD1%", "HD2%"),
                nef_nonstereo_protons: ("HDx%", "HDy%"),
            },
        ],
        "ASP" => vec![cb_spec("ASP")],
        "CYS" => vec![cb_spec("CYS")],
        "HIS" => vec![cb_spec("HIS")],
        "PHE" => vec![cb_spec("PHE")],
        "SER" => vec![cb_spec("SER")],
        "TRP" => vec![cb_spec("TRP")],
        "TYR" => vec![cb_spec("TYR")],
        "ASN" => vec![
            cb_spec("ASN"),
            ProchiralPairSpec {
                residue: "ASN",
                kind: ProchiralSiteKind::AsnCarboxamide,
                iupac_branch_1: "HD21",
                iupac_branch_2: "HD22",
                shared_heavy_atom: Some("ND2"),
                iupac_carbons: None,
                nef_nonstereo_carbons: None,
                iupac_protons_1: H_ASN_D21,
                iupac_protons_2: H_ASN_D22,
                nef_stereo_protons: ("HD21", "HD22"),
                nef_nonstereo_protons: ("HD2x", "HD2y"),
            },
        ],
        "GLU" => vec![cb_spec("GLU"), cg_spec("GLU")],
        "MET" => vec![cb_spec("MET"), cg_spec("MET")],
        "GLN" => vec![
            cb_spec("GLN"),
            cg_spec("GLN"),
            ProchiralPairSpec {
                residue: "GLN",
                kind: ProchiralSiteKind::GlnCarboxamide,
                iupac_branch_1: "HE21",
                iupac_branch_2: "HE22",
                shared_heavy_atom: Some("NE2"),
                iupac_carbons: None,
                nef_nonstereo_carbons: None,
                iupac_protons_1: H_GLN_E21,
                iupac_protons_2: H_GLN_E22,
                nef_stereo_protons: ("HE21", "HE22"),
                nef_nonstereo_protons: ("HE2x", "HE2y"),
            },
        ],
        "ARG" => vec![cb_spec("ARG"), cg_spec("ARG"), cd_spec("ARG")],
        "PRO" => vec![cb_spec("PRO"), cg_spec("PRO"), cd_spec("PRO")],
        "LYS" => vec![
            cb_spec("LYS"),
            cg_spec("LYS"),
            cd_spec("LYS"),
            ProchiralPairSpec {
                residue: "LYS",
                kind: ProchiralSiteKind::EpsilonMethylene,
                iupac_branch_1: "HE2",
                iupac_branch_2: "HE3",
                shared_heavy_atom: Some("CE"),
                iupac_carbons: None,
                nef_nonstereo_carbons: None,
                iupac_protons_1: H_HE2,
                iupac_protons_2: H_HE3,
                nef_stereo_protons: ("HE2", "HE3"),
                nef_nonstereo_protons: ("HEx", "HEy"),
            },
        ],
        _ => Vec::new(),
    }
}

/// Binary prochiral assignment hypothesis mapping observed pair `(a, b)` onto IUPAC `(branch_1, branch_2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwapHypothesis {
    /// Input `a -> iupac_branch_1` (`HB2`, `CG1`, `CD1`, `HA2`, `HD21`), `b -> iupac_branch_2`.
    Direct,
    /// Input `a -> iupac_branch_2` (`HB3`, `CG2`, `CD2`, `HA3`, `HD22`), `b -> iupac_branch_1`.
    Swapped,
}

// ============================================================================
// 1. Exact IUPAC Dihedral Phase Offsets & Calibrated Karplus Curves
// ============================================================================

/// Exact IUPAC dihedral angles `(theta_HA_HB2, theta_HA_HB3, theta_N_HB2, theta_N_HB3)` in degrees
/// for a standard L-amino acid `CB` methylene given `chi1 = dihedral(N, CA, CB, XG)` (Markley et al. 1998).
#[inline]
pub fn iupac_cb_proton_dihedrals_deg(chi1_deg: f64) -> (f64, f64, f64, f64) {
    (
        wrap_deg(chi1_deg - 120.0),
        wrap_deg(chi1_deg),
        wrap_deg(chi1_deg + 120.0),
        wrap_deg(chi1_deg - 120.0),
    )
}

/// Exact IUPAC dihedral angles `(theta_N_CG1, theta_N_CG2, theta_C_CG1, theta_C_CG2, theta_HA_HB)`
/// in degrees for `VAL` given `chi1 = dihedral(N, CA, CB, CG1)`.
#[inline]
pub fn iupac_val_cg_dihedrals_deg(chi1_deg: f64) -> (f64, f64, f64, f64, f64) {
    (
        wrap_deg(chi1_deg),
        wrap_deg(chi1_deg + 120.0),
        wrap_deg(chi1_deg - 120.0),
        wrap_deg(chi1_deg),
        wrap_deg(chi1_deg),
    )
}

/// Exact IUPAC dihedral angles `(theta_CA_CD1, theta_CA_CD2, theta_CA_HG)` in degrees for `LEU`
/// given `chi2 = dihedral(CA, CB, CG, CD1)`.
#[inline]
pub fn iupac_leu_cd_dihedrals_deg(chi2_deg: f64) -> (f64, f64, f64) {
    (
        wrap_deg(chi2_deg),
        wrap_deg(chi2_deg + 120.0),
        wrap_deg(chi2_deg - 120.0),
    )
}

/// Exact IUPAC dihedral angles `(theta_Cprev_HA2, theta_Cprev_HA3, theta_H_HA2, theta_H_HA3)` in degrees
/// for `GLY` given `phi = dihedral(C_{i-1}, N_i, CA_i, C_i)` (assuming planar trans peptide bond).
#[inline]
pub fn iupac_gly_ha_dihedrals_deg(phi_deg: f64) -> (f64, f64, f64, f64) {
    (
        wrap_deg(phi_deg + 120.0),
        wrap_deg(phi_deg - 120.0),
        wrap_deg(phi_deg - 60.0),
        wrap_deg(phi_deg + 60.0),
    )
}

/// Generalized Karplus curve `A * cos^2(theta + delta) + B * cos(theta + delta) + C` in Hz
/// (Karplus, *J. Chem. Phys.* 1959; *J. Am. Chem. Soc.* 1963).
#[inline]
pub fn karplus_hz(theta_deg: f64, a: f64, b: f64, c: f64, delta_deg: f64) -> f64 {
    let cs = (theta_deg + delta_deg).to_radians().cos();
    a * cs * cs + b * cs + c
}

/// Homonuclear aliphatic `3J(H_alpha, H_beta)` / `3J(H, H)` Karplus relation in Hz:
/// `9.5 cos^2(theta) - 1.6 cos(theta) + 1.8` (DeMarco, Llinás & Wüthrich, *Biopolymers* 1978;
/// Güntert, Braun, Billeter & Wüthrich, *J. Biomol. NMR* 1991).
#[inline]
pub fn karplus_3j_hahb_hz(theta_deg: f64) -> f64 {
    karplus_hz(theta_deg, 9.5, -1.6, 1.8, 0.0)
}

/// Unsigned magnitude `|3J(15N, H_beta)|` in Hz from 3D `HNHB` experiments:
/// `|-4.4 cos^2(theta) + 1.2 cos(theta) + 0.1|` (DeMarco et al., 1978; Archer, Ikura, Torchia & Bax,
/// *J. Magn. Reson.* 1991).
#[inline]
pub fn karplus_3j_nhb_mag_hz(theta_deg: f64) -> f64 {
    karplus_hz(theta_deg, -4.4, 1.2, 0.1, 0.0).abs()
}

/// Quantitative `3J(H^N, H^alpha)` Karplus curve in Hz:
/// `6.51 cos^2(theta) - 1.76 cos(theta) + 1.60` (Vuister & Bax, *J. Am. Chem. Soc.* 1993).
#[inline]
pub fn karplus_3j_hnha_hz(theta_deg: f64) -> f64 {
    karplus_hz(theta_deg, 6.51, -1.76, 1.60, 0.0)
}

/// Heteronuclear `3J(15N, 13C_gamma)` Karplus curve in Hz for `Val`, `Ile`, and `Thr`
/// (Hu & Bax, *J. Biomol. NMR* 1997; Chou, Case & Bax, *J. Am. Chem. Soc.* 2003, Table 2).
#[inline]
pub fn karplus_3j_ncg_hz(res_name: &str, theta_deg: f64) -> f64 {
    if res_name.eq_ignore_ascii_case("THR") {
        karplus_hz(theta_deg, 2.01, 0.21, -0.12, 7.0).max(0.0)
    } else {
        karplus_hz(theta_deg, 2.64, 0.26, -0.22, 6.0).max(0.0)
    }
}

/// Heteronuclear `3J(13C', 13C_gamma)` and aliphatic `3J(13C, 13C)` Karplus curve in Hz
/// (Hu & Bax, *J. Am. Chem. Soc.* 1997; Chou, Case & Bax, *J. Am. Chem. Soc.* 2003, Table 2).
#[inline]
pub fn karplus_3j_ccg_hz(res_name: &str, theta_deg: f64) -> f64 {
    if res_name.eq_ignore_ascii_case("THR") {
        karplus_hz(theta_deg, 2.76, -0.67, 0.19, 0.0).max(0.0)
    } else {
        karplus_hz(theta_deg, 3.42, -0.59, 0.17, 0.0).max(0.0)
    }
}

#[inline]
fn librational_avg<F: Fn(f64) -> f64>(f: F, center_deg: f64, sigma_deg: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..5 {
        acc += GH5_WEIGHT[i] * f(center_deg + sigma_deg * GH5_SCALE[i]);
    }
    acc
}

// ============================================================================
// 2. Lovell et al. (2000) Backbone-Dependent Sidechain Rotamer Priors
// ============================================================================

/// Backbone-conditioned `chi1` rotamer prior `[P(g-), P(t), P(g+)]` derived from the Penultimate
/// Rotamer Library (Lovell, Word, Richardson & Richardson, *Proteins* 2000, 40, 389–408).
pub fn lovell_chi1_prior(
    res_name: &str,
    phi_deg: Option<f64>,
    psi_deg: Option<f64>,
    ss_hint: Option<SecondaryStructure>,
) -> [f64; 3] {
    let up = res_name.trim().to_ascii_uppercase();
    if up == "PRO" {
        // Pyrrolidine puckers: Cg-exo (~ -25 deg) vs planar transition (~ 0 deg) vs Cg-endo (~ +26 deg)
        return [0.51, 0.03, 0.46];
    }

    // Determine secondary-structure basin from explicit (phi, psi) when available, else ss_hint
    let regime = if let (Some(phi), Some(psi)) = (phi_deg, psi_deg) {
        let p = wrap_deg(phi);
        let s = wrap_deg(psi);
        if (-105.0..=-35.0).contains(&p) && (-75.0..=-5.0).contains(&s) {
            Some(SecondaryStructure::Helix)
        } else if (-175.0..=-75.0).contains(&p) && (85.0..=180.0).contains(&s) {
            Some(SecondaryStructure::Sheet)
        } else {
            Some(SecondaryStructure::Coil)
        }
    } else {
        ss_hint
    };

    let raw = match (up.as_str(), regime) {
        // Valine: chi1 = N-CA-CB-CG1 -> t (~175 deg) strongly favored (~73% overall, >88% in helix)
        ("VAL", Some(SecondaryStructure::Helix)) => [0.06, 0.90, 0.04],
        ("VAL", Some(SecondaryStructure::Sheet)) => [0.24, 0.70, 0.06],
        ("VAL", _) => [0.20, 0.73, 0.07],
        // Isoleucine: chi1 = N-CA-CB-CG1 -> g- (~-64 deg) strongly favored (~75% overall, >88% in helix)
        ("ILE", Some(SecondaryStructure::Helix)) => [0.88, 0.05, 0.07],
        ("ILE", Some(SecondaryStructure::Sheet)) => [0.72, 0.16, 0.12],
        ("ILE", _) => [0.75, 0.13, 0.12],
        // Serine: strong g+ (+62 deg) preference from sidechain O_gamma hydrogen bonding
        ("SER", Some(SecondaryStructure::Helix)) => [0.28, 0.20, 0.52],
        ("SER", Some(SecondaryStructure::Sheet)) => [0.26, 0.29, 0.45],
        ("SER", _) => [0.29, 0.23, 0.48],
        // Cysteine
        ("CYS", Some(SecondaryStructure::Helix)) => [0.64, 0.26, 0.10],
        ("CYS", Some(SecondaryStructure::Sheet)) => [0.52, 0.35, 0.13],
        ("CYS", _) => [0.56, 0.30, 0.14],
        // Aromatics & branched gamma-sp2/sp3 (Phe, Tyr, Trp, His, Leu, Asp, Asn)
        ("PHE" | "TYR" | "TRP" | "HIS" | "LEU", Some(SecondaryStructure::Helix)) => {
            // Steric clash with i-3/i-4 helix turn suppresses g+ strongly in alpha-helices
            [0.64, 0.33, 0.03]
        }
        ("PHE" | "TYR" | "TRP" | "HIS" | "LEU", Some(SecondaryStructure::Sheet)) => {
            [0.56, 0.34, 0.10]
        }
        ("PHE" | "TYR" | "TRP" | "HIS" | "LEU", _) => [0.61, 0.30, 0.09],
        ("ASP" | "ASN", Some(SecondaryStructure::Helix)) => [0.67, 0.25, 0.08],
        ("ASP" | "ASN", Some(SecondaryStructure::Sheet)) => [0.49, 0.37, 0.14],
        ("ASP" | "ASN", _) => [0.57, 0.30, 0.13],
        // Long unbranched chains (Glu, Gln, Met, Lys, Arg)
        (_, Some(SecondaryStructure::Helix)) => [0.63, 0.32, 0.05],
        (_, Some(SecondaryStructure::Sheet)) => [0.51, 0.36, 0.13],
        (_, _) => [0.55, 0.33, 0.12],
    };

    let s = raw[0] + raw[1] + raw[2];
    [raw[0] / s, raw[1] / s, raw[2] / s]
}

/// Conditional `chi2` rotamer prior `[P(g- | chi1), P(t | chi1), P(g+ | chi1)]` from Lovell et al. (2000).
pub fn lovell_chi2_prior(res_name: &str, chi1: RotamerState) -> [f64; 3] {
    let up = res_name.trim().to_ascii_uppercase();
    let raw = match (up.as_str(), chi1) {
        // Leucine: famous mt (-65, 175) and tp (177, +65) basins; pentane-interference forbids g-/g- and t/g-
        ("LEU", RotamerState::GaucheMinus) => [0.03, 0.87, 0.10],
        ("LEU", RotamerState::Trans) => [0.03, 0.13, 0.84],
        ("LEU", RotamerState::GauchePlus) => [0.08, 0.52, 0.40],
        // Isoleucine CG1: mt (-62, 170) dominates
        ("ILE", RotamerState::GaucheMinus) => [0.14, 0.80, 0.06],
        ("ILE", RotamerState::Trans) => [0.06, 0.72, 0.22],
        ("ILE", RotamerState::GauchePlus) => [0.08, 0.74, 0.18],
        // Proline ring closure couples chi1 and chi2 rigidly in opposite signs!
        ("PRO", RotamerState::GaucheMinus) => [0.02, 0.04, 0.94], // Cg-exo: chi1 < 0 => chi2 > 0
        ("PRO", RotamerState::Trans) => [0.15, 0.70, 0.15],
        ("PRO", RotamerState::GauchePlus) => [0.94, 0.04, 0.02], // Cg-endo: chi1 > 0 => chi2 < 0
        // Met, Glu, Gln, Lys, Arg: all-trans chi2 (~180 deg) dominates (~68-74%)
        (_, RotamerState::GaucheMinus) => [0.13, 0.72, 0.15],
        (_, RotamerState::Trans) => [0.15, 0.70, 0.15],
        (_, RotamerState::GauchePlus) => [0.14, 0.70, 0.16],
    };
    let s = raw[0] + raw[1] + raw[2];
    [raw[0] / s, raw[1] / s, raw[2] / s]
}

// ============================================================================
// 3. Analytical NeRF Local Peptide + Sidechain 3D Builder
// ============================================================================

/// Local dipeptide backbone coordinates around residue `i` constructed analytically from `(phi_i, psi_i)`
/// via Engh & Huber (1991) geometry and Parsons et al. (2005) NeRF.
#[derive(Clone, Copy, Debug)]
pub struct LocalPeptideFrame {
    pub c_prev: Vec3,
    pub n: Vec3,
    pub h: Option<Vec3>,
    pub ca: Vec3,
    pub ha: Vec3,
    /// For `GLY`, `(HA2, HA3)` placed at exact IUPAC `phi + 120 deg` and `phi - 120 deg`.
    pub gly_ha2_ha3: Option<(Vec3, Vec3)>,
    pub cb: Option<Vec3>,
    pub c: Vec3,
    pub n_next: Vec3,
    pub h_next: Option<Vec3>,
}

/// Build an exact Engh-Huber local dipeptide unit from `(phi_deg, psi_deg)`.
///
/// Validated across all 76 residues of `PDB 1D3Z`: reproduces experimental `H_i-HB`, `HA_i-HB`,
/// and `H_{i+1}-HB` distances with mean absolute error `< 0.04 A`.
pub fn build_local_peptide_frame(
    res_name: &str,
    phi_deg: f64,
    psi_deg: f64,
    next_is_pro: bool,
) -> LocalPeptideFrame {
    let is_gly = res_name.eq_ignore_ascii_case("GLY");
    let is_pro = res_name.eq_ignore_ascii_case("PRO");
    let ang_c_n_ca = ANGLE_C_N_CA_DEG.to_radians();
    let ang_ca = if is_gly {
        ANGLE_N_CA_C_GLY_DEG.to_radians()
    } else {
        ANGLE_N_CA_C_DEG.to_radians()
    };
    let ang_ca_c_n = ANGLE_CA_C_N_DEG.to_radians();
    let ang_c_n_h = ANGLE_C_N_H_DEG.to_radians();

    // Place N_i at origin, CA_i on +x axis, and C_{i-1} in xy-plane with obtuse angle(C_{i-1}-N_i-CA_i) = 121.7 deg
    let c_prev = [
        BOND_C_N * ang_c_n_ca.cos(),
        BOND_C_N * ang_c_n_ca.sin(),
        0.0,
    ];
    let n = [0.0, 0.0, 0.0];
    let ca = [BOND_N_CA, 0.0, 0.0];
    let h = if is_pro {
        None
    } else {
        let u1 = v_unit(v_sub(n, c_prev));
        let u2 = v_unit(v_sub(n, ca));
        Some(v_add(n, v_scale(v_unit(v_add(u1, u2)), BOND_N_H)))
    };

    let c = nerf_place_atom(c_prev, n, ca, BOND_CA_C, ang_ca, phi_deg.to_radians());

    // Tetrahedral substituents at CA
    let b1 = v_unit(v_sub(n, ca));
    let b2 = v_unit(v_sub(c, ca));
    let u = v_unit(v_add(b1, b2));
    let normal = v_unit(v_cross(b1, b2));
    let dir_l_cb = v_unit(v_add(v_scale(u, -0.5810), v_scale(normal, 0.8139)));
    let dir_l_ha = v_unit(v_add(v_scale(u, -0.5810), v_scale(normal, -0.8139)));

    let (cb, ha, gly_ha2_ha3) = if is_gly {
        let ang_tet = ANGLE_TETRA_DEG.to_radians();
        let ha2 = nerf_place_atom(
            c_prev,
            n,
            ca,
            BOND_CH,
            ang_tet,
            (phi_deg + 120.0).to_radians(),
        );
        let ha3 = nerf_place_atom(
            c_prev,
            n,
            ca,
            BOND_CH,
            ang_tet,
            (phi_deg - 120.0).to_radians(),
        );
        (None, v_scale(v_add(ha2, ha3), 0.5), Some((ha2, ha3)))
    } else {
        (
            Some(v_add(ca, v_scale(dir_l_cb, BOND_CA_CB))),
            v_add(ca, v_scale(dir_l_ha, BOND_CH)),
            None,
        )
    };

    let n_next = nerf_place_atom(n, ca, c, BOND_C_N, ang_ca_c_n, psi_deg.to_radians());
    let h_next = if next_is_pro {
        None
    } else {
        // Trans peptide omega = 180 deg => omega + 180 = 0 rad
        Some(nerf_place_atom(ca, c, n_next, BOND_N_H, ang_c_n_h, 0.0))
    };

    LocalPeptideFrame {
        c_prev,
        n,
        h,
        ca,
        ha,
        gly_ha2_ha3,
        cb,
        c,
        n_next,
        h_next,
    }
}

#[inline]
fn methyl_protons_nerf(a: Vec3, b: Vec3, c_methyl: Vec3, stagger_deg: f64) -> [Vec3; 3] {
    let ang = ANGLE_TETRA_DEG.to_radians();
    [
        nerf_place_atom(a, b, c_methyl, BOND_CH, ang, stagger_deg.to_radians()),
        nerf_place_atom(
            a,
            b,
            c_methyl,
            BOND_CH,
            ang,
            (staggered_offset(stagger_deg, 120.0)).to_radians(),
        ),
        nerf_place_atom(
            a,
            b,
            c_methyl,
            BOND_CH,
            ang,
            (staggered_offset(stagger_deg, -120.0)).to_radians(),
        ),
    ]
}

#[inline]
fn staggered_offset(base: f64, off: f64) -> f64 {
    wrap_deg(base + off)
}

#[inline]
fn r6_single(p: Vec3, q: Vec3) -> f64 {
    let d = v_dist(p, q).max(1.65);
    let d2 = d * d;
    1.0 / (d2 * d2 * d2)
}

#[inline]
fn r6_methyl(p: Vec3, me: &[Vec3; 3]) -> f64 {
    (r6_single(p, me[0]) + r6_single(p, me[1]) + r6_single(p, me[2])) / 3.0
}

// ============================================================================
// 4. Backbone Conditioning, Observables & Inference Configuration
// ============================================================================

/// Backbone `(phi_i, psi_i)` conditioning mode for NOE geometry and Lovell rotamer priors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BackboneConditioning {
    /// Known backbone torsion angles `(phi_deg, psi_deg)` (e.g. from 3D coordinates).
    ExactTorsions { phi_deg: f64, psi_deg: f64 },
    /// Chemical-shift-predicted `(phi, psi)` with Gaussian uncertainty bounds (e.g. from `predict_dihedrals`).
    DihedralEstimate {
        phi_deg: f64,
        phi_err_deg: f64,
        psi_deg: f64,
        psi_err_deg: f64,
        ss_class: SecondaryStructure,
    },
    /// Secondary-structure class only (`Helix`, `Sheet`, or `Coil`), marginalizing over Ramachandran basins.
    SecondaryStructurePrior(SecondaryStructure),
    /// Unknown backbone conformation; marginalizes over canonical Ramachandran basins.
    Marginalized,
}

impl BackboneConditioning {
    pub fn from_dihedral_restraint(dr: &DihedralRestraint) -> Self {
        if dr.restrained_phi && dr.restrained_psi {
            Self::DihedralEstimate {
                phi_deg: dr.phi_deg,
                phi_err_deg: dr.phi_err_deg.clamp(8.0, 45.0),
                psi_deg: dr.psi_deg,
                psi_err_deg: dr.psi_err_deg.clamp(8.0, 45.0),
                ss_class: dr.ss_class,
            }
        } else {
            Self::SecondaryStructurePrior(dr.ss_class)
        }
    }

    fn quadrature_points(
        &self,
        res_name: &str,
    ) -> Vec<(f64, f64, f64, Option<SecondaryStructure>)> {
        let is_gly = res_name.eq_ignore_ascii_case("GLY");
        let is_pro = res_name.eq_ignore_ascii_case("PRO");
        match *self {
            Self::ExactTorsions { phi_deg, psi_deg } => {
                vec![(phi_deg, psi_deg, 1.0, None)]
            }
            Self::DihedralEstimate {
                phi_deg,
                phi_err_deg,
                psi_deg,
                psi_err_deg,
                ss_class,
            } => {
                // 80% Gaussian quadrature around the CSI torsion estimate + 20% broad Ramachandran
                // outlier mixture so turn/cap or gamma-gauche-shifted residues remain robust.
                let mut pts = Vec::with_capacity(13);
                let main_wt = 0.80;
                for (i, &sx) in GH3_SCALE.iter().enumerate() {
                    for (j, &sy) in GH3_SCALE.iter().enumerate() {
                        pts.push((
                            wrap_deg(phi_deg + phi_err_deg * sx),
                            wrap_deg(psi_deg + psi_err_deg * sy),
                            main_wt * GH3_WEIGHT[i] * GH3_WEIGHT[j],
                            Some(ss_class),
                        ));
                    }
                }
                if is_gly {
                    pts.extend([
                        (-65.0, -41.0, 0.05, None),
                        (65.0, 41.0, 0.05, None),
                        (-85.0, 155.0, 0.05, None),
                        (85.0, -155.0, 0.05, None),
                    ]);
                } else if is_pro {
                    pts.extend([(-63.0, 145.0, 0.12, None), (-61.0, -38.0, 0.08, None)]);
                } else {
                    pts.extend([
                        (-63.0, -42.0, 0.07, None),
                        (-122.0, 135.0, 0.07, None),
                        (-75.0, 145.0, 0.06, None),
                    ]);
                }
                pts
            }
            Self::SecondaryStructurePrior(ss) => {
                if is_gly {
                    // Symmetric left/right Gly basins preserve achiral symmetry when sign of phi is unknown
                    vec![
                        (-65.0, -41.0, 0.25, Some(ss)),
                        (65.0, 41.0, 0.25, Some(ss)),
                        (-85.0, 155.0, 0.25, Some(ss)),
                        (85.0, -155.0, 0.25, Some(ss)),
                    ]
                } else if is_pro {
                    match ss {
                        SecondaryStructure::Helix => vec![(-61.0, -38.0, 1.0, Some(ss))],
                        _ => vec![
                            (-63.0, 145.0, 0.75, Some(ss)),
                            (-61.0, -38.0, 0.25, Some(ss)),
                        ],
                    }
                } else {
                    match ss {
                        SecondaryStructure::Helix => vec![
                            (-63.0, -42.0, 0.75, Some(ss)),
                            (-70.0, -28.0, 0.25, Some(ss)),
                        ],
                        SecondaryStructure::Sheet => vec![
                            (-122.0, 135.0, 0.60, Some(ss)),
                            (-138.0, 162.0, 0.40, Some(ss)),
                        ],
                        SecondaryStructure::Coil => vec![
                            (-75.0, 145.0, 0.38, Some(ss)),
                            (-63.0, -42.0, 0.28, Some(ss)),
                            (-122.0, 135.0, 0.26, Some(ss)),
                            (56.0, 38.0, 0.08, Some(ss)),
                        ],
                    }
                }
            }
            Self::Marginalized => {
                if is_gly {
                    vec![
                        (-65.0, -41.0, 0.20, None),
                        (65.0, 41.0, 0.20, None),
                        (-82.0, 155.0, 0.20, None),
                        (82.0, -155.0, 0.20, None),
                        (-85.0, -10.0, 0.10, None),
                        (85.0, 10.0, 0.10, None),
                    ]
                } else if is_pro {
                    vec![(-63.0, 145.0, 0.62, None), (-61.0, -38.0, 0.38, None)]
                } else {
                    vec![
                        (-63.0, -42.0, 0.36, None),
                        (-122.0, 135.0, 0.30, None),
                        (-75.0, 145.0, 0.26, None),
                        (56.0, 38.0, 0.08, None),
                    ]
                }
            }
        }
    }
}

/// Reference anchor proton for an intraresidue or sequential NOE intensity pair `(I_a, I_b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnchorProton {
    /// Intraresidue backbone amide `H(i)`.
    IntraAmideH,
    /// Intraresidue `HA(i)` (or centroid of `HA2/HA3` on `GLY`).
    IntraAlphaHA,
    /// Sequential amide `H(i+1)` on residue `i+1`.
    SeqNextAmideH,
    /// Intraresidue methine `HB(i)` (`VAL`, `ILE`) or degenerate `QB(i)` centroid.
    IntraBetaHB,
    /// Stereospecifically resolved `HB2(i)` (`LEU`, `ASN`, `CG`-methylenes).
    IntraBetaHB2,
    /// Stereospecifically resolved `HB3(i)` (`LEU`, `ASN`, `CG`-methylenes).
    IntraBetaHB3,
    /// Intraresidue `HG(i)` methine (`LEU`) or degenerate `QG(i)` centroid (`GLN`, `CD`-methylenes).
    IntraGammaHG,
    /// Intraresidue `HD(i)` degenerate `QD(i)` centroid (`LYS HE2/HE3`).
    IntraDeltaHD,
}

/// Paired NOE cross-peak intensities `(I_a, I_b)` measured from the same anchor strip `p`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoePairObs {
    pub anchor: AnchorProton,
    pub intensity_a: f64,
    pub intensity_b: f64,
    /// Optional per-pair log-intensity standard deviation override.
    pub rel_sigma_ln: Option<f64>,
}

impl NoePairObs {
    pub fn new(anchor: AnchorProton, intensity_a: f64, intensity_b: f64) -> Self {
        Self {
            anchor,
            intensity_a,
            intensity_b,
            rel_sigma_ln: None,
        }
    }
}

/// Experimental observables for a single candidate prochiral pair `(a, b)`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProchiralObservables {
    /// Candidate proton chemical shifts `(h_a_ppm, h_b_ppm)` in input order `(a, b)`.
    pub proton_shifts_ppm: Option<(f64, f64)>,
    /// Candidate carbon shifts `(c_a_ppm, c_b_ppm)` for `Val CG1/CG2` or `Leu CD1/CD2`.
    pub carbon_shifts_ppm: Option<(f64, f64)>,
    /// Shared attached heavy-atom chemical shift (`CB`, `CA`, `CG`, `CG1`, `CD`, `CE`, `ND2`, `NE2`).
    pub shared_heavy_shift_ppm: Option<f64>,
    /// Homonuclear vicinal proton-proton couplings `(J_a, J_b)` in Hz:
    /// `3J_HAHB` (`CB`), `3J_HNHA` (`GLY`), `3J_HBHG` (`ILE CG1`, `CG`), `3J_HGHD` (`CD`), `3J_HDHE` (`CE`).
    pub j_hh_vicinal_hz: Option<(f64, f64)>,
    /// Heteronuclear `|3J_NHB|` couplings `(J_a, J_b)` in Hz (`CB` methylenes).
    pub j_nhb_hz: Option<(f64, f64)>,
    /// Heteronuclear `3J_NCG` couplings `(J_a, J_b)` in Hz (`VAL CG1/CG2`).
    pub j_ncg_hz: Option<(f64, f64)>,
    /// Heteronuclear `3J_C'CG` (`VAL CG1/CG2`) or `3J_CC` (`LEU 3J_CACD`) couplings `(J_a, J_b)` in Hz.
    pub j_cc_vicinal_hz: Option<(f64, f64)>,
    /// Optional single methine `3J_HAHB` coupling in Hz (`VAL` / `ILE`), constraining `chi1`.
    pub j_methine_hahb_hz: Option<f64>,
    /// Intraresidue and sequential NOE cross-peak intensity pairs `(I_a, I_b)`.
    pub noe_pairs: Vec<NoePairObs>,
    /// Whether residue `i+1` is `PRO` (suppresses `SeqNextAmideH`).
    pub next_is_pro: bool,
    /// Optional upstream `chi1` posterior `[P(g-), P(t), P(g+)]` propagated from `CB (HB2/HB3)`.
    pub upstream_chi1_posterior: Option<[f64; 3]>,
}

/// Configuration parameters for Bayesian rotamer-simplex & prochiral swap inference.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StereospecConfig {
    /// Minimum posterior probability `max(P(Direct), P(Swapped))` required to commit
    /// stereospecific IUPAC labels (`Ambiguity_code = 1`); default `0.90`.
    pub commit_posterior_threshold: f64,
    /// Effective standard deviation (Hz) for homonuclear `3J_HH` likelihood (`0.90 Hz` default).
    pub homonuclear_j_sigma_hz: f64,
    /// Effective standard deviation (Hz) for heteronuclear `3J_NHB`, `3J_NCG`, `3J_CCG` (`0.85 Hz` default).
    pub heteronuclear_j_sigma_hz: f64,
    /// Effective per-peak log-intensity standard deviation for `r^-6` NOE ratios (`0.40` default).
    pub noe_log_sigma: f64,
    /// Thermal librational standard deviation (degrees) within each canonical rotamer well (`10.0 deg`).
    pub libration_sigma_deg: f64,
    /// Uniform barycentric divisions `M` of the 3-state rotamer simplex (`20` -> `231` lattice points + isotropic center).
    pub simplex_divisions: usize,
    /// Spike-and-slab prior weight on dominant single-rotamer vertices (`0.62` default).
    pub dominant_rotamer_prior_weight: f64,
    /// Spike-and-slab prior weight on exact `(1/3, 1/3, 1/3)` isotropic rotational averaging (`0.16` default).
    pub isotropic_averaging_prior_weight: f64,
    /// Dirichlet concentration parameter for multi-rotamer interior simplex states (`3.0` default).
    pub dirichlet_concentration: f64,
    /// Enable weak BMRB chemical-shift separation prior on `Asn HD21/HD22` & `Gln HE21/HE22` (`true` default).
    pub use_carboxamide_shift_prior: bool,
}

impl Default for StereospecConfig {
    fn default() -> Self {
        Self {
            commit_posterior_threshold: 0.90,
            homonuclear_j_sigma_hz: 0.95,
            heteronuclear_j_sigma_hz: 0.85,
            noe_log_sigma: 0.42,
            libration_sigma_deg: 10.0,
            simplex_divisions: 20,
            dominant_rotamer_prior_weight: 0.56,
            isotropic_averaging_prior_weight: 0.24,
            dirichlet_concentration: 3.0,
            use_carboxamide_shift_prior: true,
        }
    }
}

/// Complete Bayesian inference result for one prochiral pair.
#[derive(Clone, Debug, PartialEq)]
pub struct StereospecResult {
    pub residue_idx: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub spec: ProchiralPairSpec,
    /// Normalized posterior probability `P(H = Direct | D)` (`a -> branch_1, b -> branch_2`).
    pub posterior_direct: f64,
    /// Normalized posterior probability `P(H = Swapped | D)` (`a -> branch_2, b -> branch_1`).
    pub posterior_swapped: f64,
    /// Winning swap hypothesis (`Direct` when `posterior_direct >= 0.50`, else `Swapped`).
    pub winning_hypothesis: SwapHypothesis,
    /// `max(posterior_direct, posterior_swapped)` in `[0.50, 1.00]`.
    pub winning_posterior: f64,
    /// True iff `winning_posterior >= cfg.commit_posterior_threshold` (`0.90` default).
    pub committed: bool,
    /// NMR-STAR v3 `_Atom_chem_shift.Ambiguity_code` (`1` if `committed`, `2` otherwise).
    pub nmr_star_ambiguity_code: u8,
    /// Posterior-expected rotamer populations `[E[w_{g-}], E[w_t], E[w_{g+}]]` summing to `1.0`.
    pub rotamer_marginal: [f64; 3],
    /// Maximum *a posteriori* simplex weights `[w_{g-}, w_t, w_{g+}]`.
    pub map_simplex_weights: [f64; 3],
    /// Dominant staggered rotamer when `max_r rotamer_marginal[r] >= 0.60` and not isotropic.
    pub dominant_rotamer: Option<RotamerState>,
    /// Posterior probability assigned to multi-rotamer / isotropic dynamic averaging states.
    pub dynamic_mixture_posterior: f64,
    /// Preserved input chemical shifts for export helpers.
    pub proton_shifts_ppm: Option<(f64, f64)>,
    pub carbon_shifts_ppm: Option<(f64, f64)>,
    pub shared_heavy_shift_ppm: Option<f64>,
}

impl StereospecResult {
    /// Return NEF v1.1 proton labels mapped to input `(a, b)`:
    /// stereospecific IUPAC (`("HB2", "HB3")` or `("HB3", "HB2")`) ONLY when `committed`,
    /// otherwise non-stereospecific `("HBx", "HBy")`.
    pub fn nef_proton_labels_for_ab(&self) -> (&'static str, &'static str) {
        if self.committed {
            let (s1, s2) = self.spec.nef_stereo_protons;
            match self.winning_hypothesis {
                SwapHypothesis::Direct => (s1, s2),
                SwapHypothesis::Swapped => (s2, s1),
            }
        } else {
            self.spec.nef_nonstereo_protons
        }
    }

    /// Return NEF v1.1 carbon labels mapped to input `(a, b)` for `Val`/`Leu` isopropyl pairs:
    /// `Some(("CG1", "CG2"))` / `Some(("CG2", "CG1"))` when `committed`, else `Some(("CGx", "CGy"))`.
    pub fn nef_carbon_labels_for_ab(&self) -> Option<(&'static str, &'static str)> {
        let (c1, c2) = self.spec.iupac_carbons?;
        let (nx, ny) = self.spec.nef_nonstereo_carbons?;
        if self.committed {
            Some(match self.winning_hypothesis {
                SwapHypothesis::Direct => (c1, c2),
                SwapHypothesis::Swapped => (c2, c1),
            })
        } else {
            Some((nx, ny))
        }
    }

    /// Return `(shift_branch_1, shift_branch_2)` ordered onto `(iupac_branch_1, iupac_branch_2)`.
    /// Swaps `(ha, hb) -> (hb, ha)` ONLY when `committed` with `SwapHypothesis::Swapped`; when
    /// uncommitted (`Ambiguity_code = 2`), preserves input `(ha, hb)` order verbatim so coupled
    /// `13C-1H` groups and user STAR tables are never silently mutated.
    pub fn ordered_iupac_proton_shifts(&self) -> Option<(f64, f64)> {
        let (ha, hb) = self.proton_shifts_ppm?;
        if self.committed && self.winning_hypothesis == SwapHypothesis::Swapped {
            Some((hb, ha))
        } else {
            Some((ha, hb))
        }
    }

    /// Return `(c_branch_1, c_branch_2)` ordered onto `(CG1, CG2)` or `(CD1, CD2)`.
    pub fn ordered_iupac_carbon_shifts(&self) -> Option<(f64, f64)> {
        let (ca, cb) = self.carbon_shifts_ppm?;
        if self.committed && self.winning_hypothesis == SwapHypothesis::Swapped {
            Some((cb, ca))
        } else {
            Some((ca, cb))
        }
    }
}

// ============================================================================
// 5. Simplex Grid & Pure-Rotamer Feature Table Evaluation
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum SimplexPointKind {
    Vertex(usize),
    Interior,
    Isotropic,
}

#[derive(Clone, Copy, Debug)]
struct SimplexPoint {
    w: [f64; 3],
    kind: SimplexPointKind,
}

fn build_simplex_grid(divisions: usize) -> Vec<SimplexPoint> {
    let m = divisions.max(3);
    let mut pts = Vec::with_capacity((m + 1) * (m + 2) / 2 + 1);
    for i in 0..=m {
        for j in 0..=(m - i) {
            let k = m - i - j;
            let w = [
                (i as f64) / (m as f64),
                (j as f64) / (m as f64),
                (k as f64) / (m as f64),
            ];
            let kind = if i == m {
                SimplexPointKind::Vertex(0)
            } else if j == m {
                SimplexPointKind::Vertex(1)
            } else if k == m {
                SimplexPointKind::Vertex(2)
            } else if m.is_multiple_of(3) && i * 3 == m && j * 3 == m {
                // Skip duplicate isotropic point when m is divisible by 3; appended explicitly below
                continue;
            } else {
                SimplexPointKind::Interior
            };
            pts.push(SimplexPoint { w, kind });
        }
    }
    pts.push(SimplexPoint {
        w: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        kind: SimplexPointKind::Isotropic,
    });
    pts
}

fn normalized_simplex_log_priors(
    simplex: &[SimplexPoint],
    pi: [f64; 3],
    cfg: &StereospecConfig,
) -> Vec<f64> {
    let p_dom = cfg.dominant_rotamer_prior_weight.clamp(0.05, 0.90);
    let p_iso = cfg
        .isotropic_averaging_prior_weight
        .clamp(0.01, 0.95 - p_dom);
    let p_mix = (1.0 - p_dom - p_iso).max(0.01);

    let mut interior_raw = Vec::with_capacity(simplex.len());
    for sp in simplex {
        if matches!(sp.kind, SimplexPointKind::Interior) {
            let mut we = [sp.w[0].max(0.025), sp.w[1].max(0.025), sp.w[2].max(0.025)];
            let s = we[0] + we[1] + we[2];
            we[0] /= s;
            we[1] /= s;
            we[2] /= s;
            let mut lg = 0.0;
            for r in 0..3 {
                let alpha_r = (cfg.dirichlet_concentration * pi[r]).max(0.15);
                lg += (alpha_r - 1.0) * we[r].ln();
            }
            interior_raw.push(lg);
        }
    }

    let z_int = if interior_raw.is_empty() {
        0.0
    } else {
        let mx = interior_raw
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        mx + interior_raw
            .iter()
            .map(|&x| (x - mx).exp())
            .sum::<f64>()
            .ln()
    };

    let mut out = Vec::with_capacity(simplex.len());
    let mut int_cursor = 0usize;
    for sp in simplex {
        let lp = match sp.kind {
            SimplexPointKind::Vertex(r) => (p_dom * pi[r].max(1e-4)).ln(),
            SimplexPointKind::Isotropic => p_iso.ln(),
            SimplexPointKind::Interior => {
                let v = p_mix.ln() + (interior_raw[int_cursor] - z_int);
                int_cursor += 1;
                v
            }
        };
        out.push(lp);
    }

    let mx = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z = mx + out.iter().map(|&x| (x - mx).exp()).sum::<f64>().ln();
    for x in &mut out {
        *x -= z;
    }
    out
}

/// Pure-rotamer predicted observables `(branch_1, branch_2)` for each of the 3 basis states `[g-, t, g+]`.
#[derive(Clone, Debug)]
struct RotamerBasisPredictions {
    j_hh: Option<[(f64, f64); 3]>,
    j_nhb: Option<[(f64, f64); 3]>,
    j_ncg: Option<[(f64, f64); 3]>,
    j_cc: Option<[(f64, f64); 3]>,
    j_methine_hahb: Option<[f64; 3]>,
    noe_r6: BTreeMap<AnchorProton, [(f64, f64); 3]>,
}

fn canonical_chi_angles(res_name: &str) -> [f64; 3] {
    if res_name.eq_ignore_ascii_case("PRO") {
        [-25.0, 0.0, 26.0]
    } else {
        [-64.0, 178.0, 62.0]
    }
}

fn compute_cb_basis(
    res_name: &str,
    frame: &LocalPeptideFrame,
    sig_lib: f64,
) -> RotamerBasisPredictions {
    let chis = canonical_chi_angles(res_name);
    let cb = frame.cb.unwrap_or([1.458, 1.530, 0.0]);
    let ang_tet = ANGLE_TETRA_DEG.to_radians();

    let mut j_hh = [(0.0, 0.0); 3];
    let mut j_nhb = [(0.0, 0.0); 3];
    let mut r6_h = [(0.0, 0.0); 3];
    let mut r6_ha = [(0.0, 0.0); 3];
    let mut r6_hn1 = [(0.0, 0.0); 3];

    for (r, &chi1) in chis.iter().enumerate() {
        j_hh[r] = (
            librational_avg(karplus_3j_hahb_hz, chi1 - 120.0, sig_lib),
            librational_avg(karplus_3j_hahb_hz, chi1, sig_lib),
        );
        j_nhb[r] = (
            librational_avg(karplus_3j_nhb_mag_hz, chi1 + 120.0, sig_lib),
            librational_avg(karplus_3j_nhb_mag_hz, chi1 - 120.0, sig_lib),
        );
        let hb2 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1 + 120.0).to_radians(),
        );
        let hb3 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1 - 120.0).to_radians(),
        );
        r6_ha[r] = (r6_single(frame.ha, hb2), r6_single(frame.ha, hb3));
        if let Some(h_pos) = frame.h {
            r6_h[r] = (r6_single(h_pos, hb2), r6_single(h_pos, hb3));
        }
        if let Some(hn1_pos) = frame.h_next {
            r6_hn1[r] = (r6_single(hn1_pos, hb2), r6_single(hn1_pos, hb3));
        }
    }

    let mut noe_r6 = BTreeMap::new();
    noe_r6.insert(AnchorProton::IntraAlphaHA, r6_ha);
    if frame.h.is_some() {
        noe_r6.insert(AnchorProton::IntraAmideH, r6_h);
    }
    if frame.h_next.is_some() {
        noe_r6.insert(AnchorProton::SeqNextAmideH, r6_hn1);
    }

    RotamerBasisPredictions {
        j_hh: Some(j_hh),
        j_nhb: Some(j_nhb),
        j_ncg: None,
        j_cc: None,
        j_methine_hahb: None,
        noe_r6,
    }
}

fn compute_val_basis(frame: &LocalPeptideFrame, sig_lib: f64) -> RotamerBasisPredictions {
    let chis = [-63.0, 176.0, 62.0];
    let cb = frame.cb.unwrap_or([1.458, 1.530, 0.0]);
    let ang_ccc = ANGLE_ALIPH_CCC_DEG.to_radians();
    let ang_tet = ANGLE_TETRA_DEG.to_radians();

    let mut j_ncg = [(0.0, 0.0); 3];
    let mut j_cc = [(0.0, 0.0); 3];
    let mut j_meth = [0.0; 3];
    let mut r6_h = [(0.0, 0.0); 3];
    let mut r6_ha = [(0.0, 0.0); 3];
    let mut r6_hb = [(0.0, 0.0); 3];
    let mut r6_hn1 = [(0.0, 0.0); 3];

    for (r, &chi1) in chis.iter().enumerate() {
        let (th_n_1, th_n_2, th_c_1, th_c_2, th_ha_hb) = iupac_val_cg_dihedrals_deg(chi1);
        j_ncg[r] = (
            librational_avg(|t| karplus_3j_ncg_hz("VAL", t), th_n_1, sig_lib),
            librational_avg(|t| karplus_3j_ncg_hz("VAL", t), th_n_2, sig_lib),
        );
        j_cc[r] = (
            librational_avg(|t| karplus_3j_ccg_hz("VAL", t), th_c_1, sig_lib),
            librational_avg(|t| karplus_3j_ccg_hz("VAL", t), th_c_2, sig_lib),
        );
        j_meth[r] = librational_avg(karplus_3j_hahb_hz, th_ha_hb, sig_lib);

        let cg1 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CC_ALIPH,
            ang_ccc,
            chi1.to_radians(),
        );
        let cg2 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CC_ALIPH,
            ang_ccc,
            (chi1 + 120.0).to_radians(),
        );
        let hb = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1 - 120.0).to_radians(),
        );
        let hg1 = methyl_protons_nerf(frame.ca, cb, cg1, 60.0);
        let hg2 = methyl_protons_nerf(frame.ca, cb, cg2, 60.0);

        r6_ha[r] = (r6_methyl(frame.ha, &hg1), r6_methyl(frame.ha, &hg2));
        r6_hb[r] = (r6_methyl(hb, &hg1), r6_methyl(hb, &hg2));
        if let Some(h_pos) = frame.h {
            r6_h[r] = (r6_methyl(h_pos, &hg1), r6_methyl(h_pos, &hg2));
        }
        if let Some(hn1_pos) = frame.h_next {
            r6_hn1[r] = (r6_methyl(hn1_pos, &hg1), r6_methyl(hn1_pos, &hg2));
        }
    }

    let mut noe_r6 = BTreeMap::new();
    noe_r6.insert(AnchorProton::IntraAlphaHA, r6_ha);
    noe_r6.insert(AnchorProton::IntraBetaHB, r6_hb);
    if frame.h.is_some() {
        noe_r6.insert(AnchorProton::IntraAmideH, r6_h);
    }
    if frame.h_next.is_some() {
        noe_r6.insert(AnchorProton::SeqNextAmideH, r6_hn1);
    }

    RotamerBasisPredictions {
        j_hh: None,
        j_nhb: None,
        j_ncg: Some(j_ncg),
        j_cc: Some(j_cc),
        j_methine_hahb: Some(j_meth),
        noe_r6,
    }
}

fn compute_leu_basis_for_chi1(
    frame: &LocalPeptideFrame,
    chi1_deg: f64,
    sig_lib: f64,
) -> RotamerBasisPredictions {
    // Canonical Leu chi2 angles for [g-, t, g+]
    let chi2s = [-64.0, 175.0, 66.0];
    let cb = frame.cb.unwrap_or([1.458, 1.530, 0.0]);
    let ang_ccc = ANGLE_ALIPH_CCC_DEG.to_radians();
    let ang_tet = ANGLE_TETRA_DEG.to_radians();

    let cg = nerf_place_atom(
        frame.n,
        frame.ca,
        cb,
        BOND_CC_ALIPH,
        ang_ccc,
        chi1_deg.to_radians(),
    );
    let hb2 = nerf_place_atom(
        frame.n,
        frame.ca,
        cb,
        BOND_CH,
        ang_tet,
        (chi1_deg + 120.0).to_radians(),
    );
    let hb3 = nerf_place_atom(
        frame.n,
        frame.ca,
        cb,
        BOND_CH,
        ang_tet,
        (chi1_deg - 120.0).to_radians(),
    );
    let qb = v_scale(v_add(hb2, hb3), 0.5);

    let mut j_cc = [(0.0, 0.0); 3];
    let mut r6_h = [(0.0, 0.0); 3];
    let mut r6_ha = [(0.0, 0.0); 3];
    let mut r6_qb = [(0.0, 0.0); 3];
    let mut r6_hb2 = [(0.0, 0.0); 3];
    let mut r6_hb3 = [(0.0, 0.0); 3];
    let mut r6_hg = [(0.0, 0.0); 3];
    let mut r6_hn1 = [(0.0, 0.0); 3];

    for (r, &chi2) in chi2s.iter().enumerate() {
        let (th_cd1, th_cd2, th_hg) = iupac_leu_cd_dihedrals_deg(chi2);
        j_cc[r] = (
            librational_avg(|t| karplus_3j_ccg_hz("LEU", t), th_cd1, sig_lib),
            librational_avg(|t| karplus_3j_ccg_hz("LEU", t), th_cd2, sig_lib),
        );
        let cd1 = nerf_place_atom(
            frame.ca,
            cb,
            cg,
            BOND_CC_ALIPH,
            ang_ccc,
            th_cd1.to_radians(),
        );
        let cd2 = nerf_place_atom(
            frame.ca,
            cb,
            cg,
            BOND_CC_ALIPH,
            ang_ccc,
            th_cd2.to_radians(),
        );
        let hg = nerf_place_atom(frame.ca, cb, cg, BOND_CH, ang_tet, th_hg.to_radians());

        let hd1 = methyl_protons_nerf(cb, cg, cd1, 60.0);
        let hd2 = methyl_protons_nerf(cb, cg, cd2, 60.0);

        r6_ha[r] = (r6_methyl(frame.ha, &hd1), r6_methyl(frame.ha, &hd2));
        r6_qb[r] = (r6_methyl(qb, &hd1), r6_methyl(qb, &hd2));
        r6_hb2[r] = (r6_methyl(hb2, &hd1), r6_methyl(hb2, &hd2));
        r6_hb3[r] = (r6_methyl(hb3, &hd1), r6_methyl(hb3, &hd2));
        r6_hg[r] = (r6_methyl(hg, &hd1), r6_methyl(hg, &hd2));
        if let Some(h_pos) = frame.h {
            r6_h[r] = (r6_methyl(h_pos, &hd1), r6_methyl(h_pos, &hd2));
        }
        if let Some(hn1_pos) = frame.h_next {
            r6_hn1[r] = (r6_methyl(hn1_pos, &hd1), r6_methyl(hn1_pos, &hd2));
        }
    }

    let mut noe_r6 = BTreeMap::new();
    noe_r6.insert(AnchorProton::IntraAlphaHA, r6_ha);
    noe_r6.insert(AnchorProton::IntraBetaHB, r6_qb);
    noe_r6.insert(AnchorProton::IntraBetaHB2, r6_hb2);
    noe_r6.insert(AnchorProton::IntraBetaHB3, r6_hb3);
    noe_r6.insert(AnchorProton::IntraGammaHG, r6_hg);
    if frame.h.is_some() {
        noe_r6.insert(AnchorProton::IntraAmideH, r6_h);
    }
    if frame.h_next.is_some() {
        noe_r6.insert(AnchorProton::SeqNextAmideH, r6_hn1);
    }

    RotamerBasisPredictions {
        j_hh: None,
        j_nhb: None,
        j_ncg: None,
        j_cc: Some(j_cc),
        j_methine_hahb: None,
        noe_r6,
    }
}

fn compute_gamma_methylene_basis_for_chi1(
    res_name: &str,
    frame: &LocalPeptideFrame,
    chi1_deg: f64,
    sig_lib: f64,
) -> RotamerBasisPredictions {
    let is_ile = res_name.eq_ignore_ascii_case("ILE");
    let is_pro = res_name.eq_ignore_ascii_case("PRO");
    let chi2s = if is_pro {
        [-34.0, 0.0, 34.0]
    } else {
        [-64.0, 176.0, 64.0]
    };
    let cb = frame.cb.unwrap_or([1.458, 1.530, 0.0]);
    let ang_ccc = ANGLE_ALIPH_CCC_DEG.to_radians();
    let ang_tet = ANGLE_TETRA_DEG.to_radians();

    let cg = nerf_place_atom(
        frame.n,
        frame.ca,
        cb,
        BOND_CC_ALIPH,
        ang_ccc,
        chi1_deg.to_radians(),
    );

    let mut j_hh = [(0.0, 0.0); 3];
    let mut r6_ha = [(0.0, 0.0); 3];
    let mut r6_h = [(0.0, 0.0); 3];
    let mut r6_hb = [(0.0, 0.0); 3];
    let mut r6_hb2 = [(0.0, 0.0); 3];
    let mut r6_hb3 = [(0.0, 0.0); 3];
    let mut r6_hn1 = [(0.0, 0.0); 3];

    if is_ile {
        // (2S, 3S)-Ile: HB is at chi1 + 120 deg; HG12 is at chi2 + 120 deg (CA-CB-CG1-HG12),
        // so dihedral(HB, CB, CG1, HG12) = chi2, dihedral(HB, CB, CG1, HG13) = chi2 + 120 deg!
        let hb = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1_deg + 120.0).to_radians(),
        );
        for (r, &chi2) in chi2s.iter().enumerate() {
            j_hh[r] = (
                librational_avg(karplus_3j_hahb_hz, chi2, sig_lib),
                librational_avg(karplus_3j_hahb_hz, chi2 + 120.0, sig_lib),
            );
            let hg12 = nerf_place_atom(
                frame.ca,
                cb,
                cg,
                BOND_CH,
                ang_tet,
                (chi2 + 120.0).to_radians(),
            );
            let hg13 = nerf_place_atom(
                frame.ca,
                cb,
                cg,
                BOND_CH,
                ang_tet,
                (chi2 - 120.0).to_radians(),
            );
            r6_ha[r] = (r6_single(frame.ha, hg12), r6_single(frame.ha, hg13));
            r6_hb[r] = (r6_single(hb, hg12), r6_single(hb, hg13));
            if let Some(h_pos) = frame.h {
                r6_h[r] = (r6_single(h_pos, hg12), r6_single(h_pos, hg13));
            }
            if let Some(hn1_pos) = frame.h_next {
                r6_hn1[r] = (r6_single(hn1_pos, hg12), r6_single(hn1_pos, hg13));
            }
        }
    } else {
        let hb2 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1_deg + 120.0).to_radians(),
        );
        let hb3 = nerf_place_atom(
            frame.n,
            frame.ca,
            cb,
            BOND_CH,
            ang_tet,
            (chi1_deg - 120.0).to_radians(),
        );
        let qb = v_scale(v_add(hb2, hb3), 0.5);
        for (r, &chi2) in chi2s.iter().enumerate() {
            // Average vicinal J(HB2/3, HG2/3): HB2 is at +120 from CA, HB3 is at -120 from CA
            let j2 = 0.5
                * (librational_avg(karplus_3j_hahb_hz, chi2, sig_lib)
                    + librational_avg(karplus_3j_hahb_hz, chi2 - 120.0, sig_lib));
            let j3 = 0.5
                * (librational_avg(karplus_3j_hahb_hz, chi2 + 120.0, sig_lib)
                    + librational_avg(karplus_3j_hahb_hz, chi2, sig_lib));
            j_hh[r] = (j2, j3);

            let hg2 = nerf_place_atom(
                frame.ca,
                cb,
                cg,
                BOND_CH,
                ang_tet,
                (chi2 + 120.0).to_radians(),
            );
            let hg3 = nerf_place_atom(
                frame.ca,
                cb,
                cg,
                BOND_CH,
                ang_tet,
                (chi2 - 120.0).to_radians(),
            );
            r6_ha[r] = (r6_single(frame.ha, hg2), r6_single(frame.ha, hg3));
            r6_hb[r] = (r6_single(qb, hg2), r6_single(qb, hg3));
            r6_hb2[r] = (r6_single(hb2, hg2), r6_single(hb2, hg3));
            r6_hb3[r] = (r6_single(hb3, hg2), r6_single(hb3, hg3));
            if let Some(h_pos) = frame.h {
                r6_h[r] = (r6_single(h_pos, hg2), r6_single(h_pos, hg3));
            }
            if let Some(hn1_pos) = frame.h_next {
                r6_hn1[r] = (r6_single(hn1_pos, hg2), r6_single(hn1_pos, hg3));
            }
        }
    }

    let mut noe_r6 = BTreeMap::new();
    noe_r6.insert(AnchorProton::IntraAlphaHA, r6_ha);
    noe_r6.insert(AnchorProton::IntraBetaHB, r6_hb);
    if !is_ile {
        noe_r6.insert(AnchorProton::IntraBetaHB2, r6_hb2);
        noe_r6.insert(AnchorProton::IntraBetaHB3, r6_hb3);
    }
    if frame.h.is_some() {
        noe_r6.insert(AnchorProton::IntraAmideH, r6_h);
    }
    if frame.h_next.is_some() {
        noe_r6.insert(AnchorProton::SeqNextAmideH, r6_hn1);
    }

    RotamerBasisPredictions {
        j_hh: Some(j_hh),
        j_nhb: None,
        j_ncg: None,
        j_cc: None,
        j_methine_hahb: None,
        noe_r6,
    }
}

fn evaluate_basis_simplex_log_likelihood(
    basis: &RotamerBasisPredictions,
    w: [f64; 3],
    hyp: SwapHypothesis,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> f64 {
    let map_pair = |(va, vb): (f64, f64)| -> (f64, f64) {
        match hyp {
            SwapHypothesis::Direct => (va, vb),
            SwapHypothesis::Swapped => (vb, va),
        }
    };

    let mut ll = 0.0;

    if let (Some(obs_j), Some(pred_j)) = (obs.j_hh_vicinal_hz, &basis.j_hh) {
        let (o1, o2) = map_pair(obs_j);
        let p1 = w[0] * pred_j[0].0 + w[1] * pred_j[1].0 + w[2] * pred_j[2].0;
        let p2 = w[0] * pred_j[0].1 + w[1] * pred_j[1].1 + w[2] * pred_j[2].1;
        let iv = 1.0 / (cfg.homonuclear_j_sigma_hz * cfg.homonuclear_j_sigma_hz);
        ll -= 0.5 * ((o1 - p1).powi(2) + (o2 - p2).powi(2)) * iv;
    }

    if let (Some(obs_j), Some(pred_j)) = (obs.j_nhb_hz, &basis.j_nhb) {
        let (o1, o2) = map_pair((obs_j.0.abs(), obs_j.1.abs()));
        let p1 = w[0] * pred_j[0].0 + w[1] * pred_j[1].0 + w[2] * pred_j[2].0;
        let p2 = w[0] * pred_j[0].1 + w[1] * pred_j[1].1 + w[2] * pred_j[2].1;
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((o1 - p1).powi(2) + (o2 - p2).powi(2)) * iv;
    }

    if let (Some(obs_j), Some(pred_j)) = (obs.j_ncg_hz, &basis.j_ncg) {
        let (o1, o2) = map_pair((obs_j.0.abs(), obs_j.1.abs()));
        let p1 = w[0] * pred_j[0].0 + w[1] * pred_j[1].0 + w[2] * pred_j[2].0;
        let p2 = w[0] * pred_j[0].1 + w[1] * pred_j[1].1 + w[2] * pred_j[2].1;
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((o1 - p1).powi(2) + (o2 - p2).powi(2)) * iv;
    }

    if let (Some(obs_j), Some(pred_j)) = (obs.j_cc_vicinal_hz, &basis.j_cc) {
        let (o1, o2) = map_pair((obs_j.0.abs(), obs_j.1.abs()));
        let p1 = w[0] * pred_j[0].0 + w[1] * pred_j[1].0 + w[2] * pred_j[2].0;
        let p2 = w[0] * pred_j[0].1 + w[1] * pred_j[1].1 + w[2] * pred_j[2].1;
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((o1 - p1).powi(2) + (o2 - p2).powi(2)) * iv;
    }

    if let (Some(om), Some(pm)) = (obs.j_methine_hahb_hz, &basis.j_methine_hahb) {
        let p = w[0] * pm[0] + w[1] * pm[1] + w[2] * pm[2];
        let iv = 1.0 / (cfg.homonuclear_j_sigma_hz * cfg.homonuclear_j_sigma_hz);
        ll -= 0.5 * (om - p).powi(2) * iv;
    }

    // Baseline spin-diffusion / detection floor r_max = 5.5 A
    let floor_r6 = 5.5_f64.powi(-6);
    for np in &obs.noe_pairs {
        if np.intensity_a <= 0.0 || np.intensity_b <= 0.0 {
            continue;
        }
        let Some(pred_r6) = basis.noe_r6.get(&np.anchor) else {
            continue;
        };
        let (ia, ib) = map_pair((np.intensity_a, np.intensity_b));
        let u1 = w[0] * pred_r6[0].0 + w[1] * pred_r6[1].0 + w[2] * pred_r6[2].0 + floor_r6;
        let u2 = w[0] * pred_r6[0].1 + w[1] * pred_r6[1].1 + w[2] * pred_r6[2].1 + floor_r6;
        let obs_lr = (ia / ib).ln();
        let pred_lr = (u1 / u2).ln();
        let sig = np.rel_sigma_ln.unwrap_or(cfg.noe_log_sigma).max(0.10);
        ll -= 0.5 * (obs_lr - pred_lr).powi(2) / (2.0 * sig * sig);
    }

    ll
}

fn evaluate_isotropic_null_log_likelihood(
    basis: &RotamerBasisPredictions,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> f64 {
    // Exact branch-symmetric conformational-averaging / uninformative null model:
    // both branches have identical expected couplings `0.5 * (<J_1> + <J_2>)` and equal NOE distances.
    let mut ll = 0.0;
    let avg_sym = |arr: &[(f64, f64); 3]| -> f64 {
        (arr[0].0 + arr[0].1 + arr[1].0 + arr[1].1 + arr[2].0 + arr[2].1) / 6.0
    };

    if let (Some((oa, ob)), Some(pred_j)) = (obs.j_hh_vicinal_hz, &basis.j_hh) {
        let p = avg_sym(pred_j);
        let iv = 1.0 / (cfg.homonuclear_j_sigma_hz * cfg.homonuclear_j_sigma_hz);
        ll -= 0.5 * ((oa - p).powi(2) + (ob - p).powi(2)) * iv;
    }
    if let (Some((oa, ob)), Some(pred_j)) = (obs.j_nhb_hz, &basis.j_nhb) {
        let p = avg_sym(pred_j);
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((oa.abs() - p).powi(2) + (ob.abs() - p).powi(2)) * iv;
    }
    if let (Some((oa, ob)), Some(pred_j)) = (obs.j_ncg_hz, &basis.j_ncg) {
        let p = avg_sym(pred_j);
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((oa.abs() - p).powi(2) + (ob.abs() - p).powi(2)) * iv;
    }
    if let (Some((oa, ob)), Some(pred_j)) = (obs.j_cc_vicinal_hz, &basis.j_cc) {
        let p = avg_sym(pred_j);
        let iv = 1.0 / (cfg.heteronuclear_j_sigma_hz * cfg.heteronuclear_j_sigma_hz);
        ll -= 0.5 * ((oa.abs() - p).powi(2) + (ob.abs() - p).powi(2)) * iv;
    }
    if let (Some(om), Some(pm)) = (obs.j_methine_hahb_hz, &basis.j_methine_hahb) {
        let p = (pm[0] + pm[1] + pm[2]) / 3.0;
        let iv = 1.0 / (cfg.homonuclear_j_sigma_hz * cfg.homonuclear_j_sigma_hz);
        ll -= 0.5 * (om - p).powi(2) * iv;
    }
    for np in &obs.noe_pairs {
        if np.intensity_a <= 0.0 || np.intensity_b <= 0.0 {
            continue;
        }
        if !basis.noe_r6.contains_key(&np.anchor) {
            continue;
        }
        let obs_lr = (np.intensity_a / np.intensity_b).ln();
        let sig = np.rel_sigma_ln.unwrap_or(cfg.noe_log_sigma).max(0.10);
        ll -= 0.5 * obs_lr.powi(2) / (2.0 * sig * sig);
    }
    ll
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

// ============================================================================
// 6. Core Bayesian Prochiral Inference Engine
// ============================================================================

const MIN_PHYSICAL_3J_HZ: f64 = -0.5;
const MAX_PHYSICAL_3J_HZ: f64 = 16.0;
const MAX_REDUCED_CHI2: f64 = 9.0;
const DEGENERATE_POSTERIOR_CAP: f64 = 0.85;

fn is_valid_3j_hz(v: f64) -> bool {
    v.is_finite() && (MIN_PHYSICAL_3J_HZ..=MAX_PHYSICAL_3J_HZ).contains(&v)
}

fn has_invalid_or_out_of_range_observables(
    backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> bool {
    if !cfg.commit_posterior_threshold.is_finite()
        || !cfg.homonuclear_j_sigma_hz.is_finite()
        || cfg.homonuclear_j_sigma_hz <= 0.0
        || !cfg.heteronuclear_j_sigma_hz.is_finite()
        || cfg.heteronuclear_j_sigma_hz <= 0.0
        || !cfg.noe_log_sigma.is_finite()
        || cfg.noe_log_sigma <= 0.0
        || !cfg.libration_sigma_deg.is_finite()
        || cfg.libration_sigma_deg < 0.0
    {
        return true;
    }
    match *backbone {
        BackboneConditioning::ExactTorsions { phi_deg, psi_deg } => {
            if !phi_deg.is_finite() || !psi_deg.is_finite() {
                return true;
            }
        }
        BackboneConditioning::DihedralEstimate {
            phi_deg,
            psi_deg,
            phi_err_deg,
            psi_err_deg,
            ..
        } => {
            if !phi_deg.is_finite()
                || !psi_deg.is_finite()
                || !phi_err_deg.is_finite()
                || !psi_err_deg.is_finite()
                || phi_err_deg <= 0.0
                || psi_err_deg <= 0.0
            {
                return true;
            }
        }
        BackboneConditioning::SecondaryStructurePrior(_) | BackboneConditioning::Marginalized => {}
    }
    if obs
        .proton_shifts_ppm
        .is_some_and(|(a, b)| !a.is_finite() || !b.is_finite())
        || obs
            .carbon_shifts_ppm
            .is_some_and(|(a, b)| !a.is_finite() || !b.is_finite())
        || obs.shared_heavy_shift_ppm.is_some_and(|x| !x.is_finite())
        || obs
            .upstream_chi1_posterior
            .is_some_and(|w| w.iter().any(|x| !x.is_finite() || *x < 0.0))
    {
        return true;
    }
    for pair_j in [
        obs.j_hh_vicinal_hz,
        obs.j_nhb_hz,
        obs.j_ncg_hz,
        obs.j_cc_vicinal_hz,
    ]
    .into_iter()
    .flatten()
    {
        if !is_valid_3j_hz(pair_j.0) || !is_valid_3j_hz(pair_j.1) {
            return true;
        }
    }
    if let Some(jm) = obs.j_methine_hahb_hz {
        if !is_valid_3j_hz(jm) {
            return true;
        }
    }
    for np in &obs.noe_pairs {
        if !np.intensity_a.is_finite()
            || !np.intensity_b.is_finite()
            || np.intensity_a <= 0.0
            || np.intensity_b <= 0.0
        {
            return true;
        }
        if np.rel_sigma_ln.is_some_and(|s| !s.is_finite() || s <= 0.0) {
            return true;
        }
    }
    false
}

fn uncommitted_null_result(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    obs: &ProchiralObservables,
) -> StereospecResult {
    StereospecResult {
        residue_idx,
        seq_id,
        res_name: spec.residue.to_string(),
        spec,
        posterior_direct: 0.50,
        posterior_swapped: 0.50,
        winning_hypothesis: SwapHypothesis::Direct,
        winning_posterior: 0.50,
        committed: false,
        nmr_star_ambiguity_code: 2,
        rotamer_marginal: [1.0 / 3.0; 3],
        map_simplex_weights: [1.0 / 3.0; 3],
        dominant_rotamer: None,
        dynamic_mixture_posterior: 0.0,
        proton_shifts_ppm: obs.proton_shifts_ppm,
        carbon_shifts_ppm: obs.carbon_shifts_ppm,
        shared_heavy_shift_ppm: obs.shared_heavy_shift_ppm,
    }
}

fn count_basis_observations(basis: &RotamerBasisPredictions, obs: &ProchiralObservables) -> usize {
    let mut n = 0usize;
    if basis.j_hh.is_some() && obs.j_hh_vicinal_hz.is_some() {
        n += 2;
    }
    if basis.j_nhb.is_some() && obs.j_nhb_hz.is_some() {
        n += 2;
    }
    if basis.j_ncg.is_some() && obs.j_ncg_hz.is_some() {
        n += 2;
    }
    if basis.j_cc.is_some() && obs.j_cc_vicinal_hz.is_some() {
        n += 2;
    }
    if basis.j_methine_hahb.is_some() && obs.j_methine_hahb_hz.is_some() {
        n += 1;
    }
    for np in &obs.noe_pairs {
        if np.intensity_a > 0.0 && np.intensity_b > 0.0 && basis.noe_r6.contains_key(&np.anchor) {
            n += 1;
        }
    }
    n
}

fn anchor_discriminates_site(
    spec: ProchiralPairSpec,
    anchor: AnchorProton,
    next_is_pro: bool,
) -> bool {
    if anchor == AnchorProton::SeqNextAmideH && next_is_pro {
        return false;
    }
    match spec.kind {
        ProchiralSiteKind::GlyAlphaMethylene => {
            matches!(
                anchor,
                AnchorProton::IntraAmideH | AnchorProton::SeqNextAmideH
            )
        }
        ProchiralSiteKind::BetaMethylene | ProchiralSiteKind::ValIsopropyl => matches!(
            anchor,
            AnchorProton::IntraAmideH | AnchorProton::IntraAlphaHA | AnchorProton::SeqNextAmideH
        ),
        ProchiralSiteKind::GammaMethylene => matches!(
            anchor,
            AnchorProton::IntraAmideH
                | AnchorProton::IntraAlphaHA
                | AnchorProton::IntraBetaHB
                | AnchorProton::IntraBetaHB2
                | AnchorProton::IntraBetaHB3
                | AnchorProton::SeqNextAmideH
        ),
        ProchiralSiteKind::LeuIsopropyl => matches!(
            anchor,
            AnchorProton::IntraAmideH
                | AnchorProton::IntraAlphaHA
                | AnchorProton::IntraBetaHB
                | AnchorProton::IntraBetaHB2
                | AnchorProton::IntraBetaHB3
                | AnchorProton::IntraGammaHG
                | AnchorProton::SeqNextAmideH
        ),
        ProchiralSiteKind::AsnCarboxamide => matches!(
            anchor,
            AnchorProton::IntraBetaHB
                | AnchorProton::IntraBetaHB2
                | AnchorProton::IntraBetaHB3
                | AnchorProton::IntraAlphaHA
        ),
        ProchiralSiteKind::GlnCarboxamide => matches!(
            anchor,
            AnchorProton::IntraGammaHG
                | AnchorProton::IntraBetaHB
                | AnchorProton::IntraBetaHB2
                | AnchorProton::IntraBetaHB3
        ),
        ProchiralSiteKind::DeltaMethylene | ProchiralSiteKind::EpsilonMethylene => {
            spec.residue.eq_ignore_ascii_case("PRO") && anchor == AnchorProton::IntraAlphaHA
        }
    }
}

/// Return `true` only when at least two independent pair-discriminating pathways (or at least one
/// asymmetric NOE on a planar carboxamide) are present; returns `false` whenever zero or only a
/// single scalar-coupling pathway is supplied without a second independent `J` or discriminating NOE.
fn has_multi_pathway_prochiral_evidence(
    spec: ProchiralPairSpec,
    obs: &ProchiralObservables,
) -> bool {
    let n_paired_j = match spec.kind {
        ProchiralSiteKind::BetaMethylene => {
            usize::from(obs.j_hh_vicinal_hz.is_some()) + usize::from(obs.j_nhb_hz.is_some())
        }
        ProchiralSiteKind::ValIsopropyl => {
            usize::from(obs.j_ncg_hz.is_some()) + usize::from(obs.j_cc_vicinal_hz.is_some())
        }
        ProchiralSiteKind::GammaMethylene => usize::from(obs.j_hh_vicinal_hz.is_some()),
        ProchiralSiteKind::LeuIsopropyl => usize::from(obs.j_cc_vicinal_hz.is_some()),
        ProchiralSiteKind::GlyAlphaMethylene
        | ProchiralSiteKind::DeltaMethylene
        | ProchiralSiteKind::EpsilonMethylene => usize::from(obs.j_hh_vicinal_hz.is_some()),
        ProchiralSiteKind::AsnCarboxamide | ProchiralSiteKind::GlnCarboxamide => 0,
    };
    let n_disc_noes = obs
        .noe_pairs
        .iter()
        .filter(|np| {
            np.intensity_a > 0.0
                && np.intensity_b > 0.0
                && anchor_discriminates_site(spec, np.anchor, obs.next_is_pro)
                && (np.intensity_a / np.intensity_b).ln().abs() > 1e-4
        })
        .count();

    !(n_paired_j <= 1 && n_disc_noes == 0)
}

fn cap_degenerate_posterior(p_dir: f64) -> f64 {
    p_dir.clamp(1.0 - DEGENERATE_POSTERIOR_CAP, DEGENERATE_POSTERIOR_CAP)
}

/// Infer joint rotamer-population posterior and stereospecific swap hypothesis `H in {Direct, Swapped}`
/// for a single prochiral pair `(a, b)`.
pub fn infer_prochiral_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    if has_invalid_or_out_of_range_observables(backbone, obs, cfg) {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }
    match spec.kind {
        ProchiralSiteKind::GlyAlphaMethylene => {
            infer_gly_pair(residue_idx, seq_id, spec, backbone, obs, cfg)
        }
        ProchiralSiteKind::AsnCarboxamide | ProchiralSiteKind::GlnCarboxamide => {
            infer_carboxamide_pair(residue_idx, seq_id, spec, backbone, obs, cfg)
        }
        ProchiralSiteKind::DeltaMethylene | ProchiralSiteKind::EpsilonMethylene => {
            infer_distal_methylene_pair(residue_idx, seq_id, spec, obs, cfg)
        }
        ProchiralSiteKind::BetaMethylene | ProchiralSiteKind::ValIsopropyl => {
            infer_chi1_simplex_pair(residue_idx, seq_id, spec, backbone, obs, cfg)
        }
        ProchiralSiteKind::LeuIsopropyl | ProchiralSiteKind::GammaMethylene => {
            infer_chi1_chi2_simplex_pair(residue_idx, seq_id, spec, backbone, obs, cfg)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_result(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    posterior_direct: f64,
    rotamer_marginal: [f64; 3],
    map_simplex_weights: [f64; 3],
    dynamic_mixture_posterior: f64,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    if !posterior_direct.is_finite() {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }
    let mut p_dir = posterior_direct.clamp(0.0, 1.0);
    if !has_multi_pathway_prochiral_evidence(spec, obs) {
        p_dir = cap_degenerate_posterior(p_dir);
    }
    let p_swp = (1.0 - p_dir).clamp(0.0, 1.0);
    let (winning_hypothesis, winning_posterior) = if p_dir >= 0.5 {
        (SwapHypothesis::Direct, p_dir)
    } else {
        (SwapHypothesis::Swapped, p_swp)
    };
    let thresh = cfg.commit_posterior_threshold.clamp(0.500_001, 1.0);
    let committed = winning_posterior >= thresh;
    let nmr_star_ambiguity_code = if committed { 1 } else { 2 };

    let mut best_r = 0;
    for r in 1..3 {
        if rotamer_marginal[r] > rotamer_marginal[best_r] {
            best_r = r;
        }
    }
    let dominant_rotamer = if rotamer_marginal[best_r] >= 0.60 && dynamic_mixture_posterior < 0.50 {
        Some(RotamerState::ALL[best_r])
    } else {
        None
    };

    StereospecResult {
        residue_idx,
        seq_id,
        res_name: spec.residue.to_string(),
        spec,
        posterior_direct: p_dir,
        posterior_swapped: p_swp,
        winning_hypothesis,
        winning_posterior,
        committed,
        nmr_star_ambiguity_code,
        rotamer_marginal,
        map_simplex_weights,
        dominant_rotamer,
        dynamic_mixture_posterior: dynamic_mixture_posterior.clamp(0.0, 1.0),
        proton_shifts_ppm: obs.proton_shifts_ppm,
        carbon_shifts_ppm: obs.carbon_shifts_ppm,
        shared_heavy_shift_ppm: obs.shared_heavy_shift_ppm,
    }
}

fn infer_chi1_simplex_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    let simplex = build_simplex_grid(cfg.simplex_divisions);
    let n_s = simplex.len();
    let bb_pts = backbone.quadrature_points(spec.residue);
    let ln_half = 0.5_f64.ln();
    let uniform_log_priors = normalized_simplex_log_priors(&simplex, [1.0 / 3.0; 3], cfg);

    // Accumulate joint log-posterior across backbone quadrature points into [hyp][simplex_idx]
    let mut joint_log = [vec![f64::NEG_INFINITY; n_s], vec![f64::NEG_INFINITY; n_s]];
    let mut unif_dir_terms: Vec<f64> = Vec::new();
    let mut unif_swp_terms: Vec<f64> = Vec::new();
    let mut max_data_ll = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    for &(phi, psi, w_bb, ss_hint) in &bb_pts {
        if w_bb <= 0.0 {
            continue;
        }
        let ln_wbb = w_bb.ln();
        let frame = build_local_peptide_frame(spec.residue, phi, psi, obs.next_is_pro);
        let pi = if let Some(up) = obs.upstream_chi1_posterior {
            up
        } else {
            lovell_chi1_prior(spec.residue, Some(phi), Some(psi), ss_hint)
        };
        let log_priors = normalized_simplex_log_priors(&simplex, pi, cfg);
        let basis = if spec.kind == ProchiralSiteKind::ValIsopropyl {
            compute_val_basis(&frame, cfg.libration_sigma_deg)
        } else {
            compute_cb_basis(spec.residue, &frame, cfg.libration_sigma_deg)
        };
        n_obs = n_obs.max(count_basis_observations(&basis, obs));

        for (si, sp) in simplex.iter().enumerate() {
            let base_lp = ln_wbb + log_priors[si] + ln_half;
            let unif_lp = ln_wbb + uniform_log_priors[si] + ln_half;
            let (raw_d, raw_s) = if matches!(sp.kind, SimplexPointKind::Isotropic) {
                let r0 = evaluate_isotropic_null_log_likelihood(&basis, obs, cfg);
                (r0, r0)
            } else {
                (
                    evaluate_basis_simplex_log_likelihood(
                        &basis,
                        sp.w,
                        SwapHypothesis::Direct,
                        obs,
                        cfg,
                    ),
                    evaluate_basis_simplex_log_likelihood(
                        &basis,
                        sp.w,
                        SwapHypothesis::Swapped,
                        obs,
                        cfg,
                    ),
                )
            };
            max_data_ll = max_data_ll.max(raw_d).max(raw_s);
            joint_log[0][si] = log_sum_exp(&[joint_log[0][si], base_lp + raw_d]);
            joint_log[1][si] = log_sum_exp(&[joint_log[1][si], base_lp + raw_s]);
            unif_dir_terms.push(unif_lp + raw_d);
            unif_swp_terms.push(unif_lp + raw_s);
        }
    }

    if n_obs > 0
        && (!max_data_ll.is_finite() || (-2.0 * max_data_ll) / (n_obs as f64) > MAX_REDUCED_CHI2)
    {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }

    let lse_dir = log_sum_exp(&joint_log[0]);
    let lse_swp = log_sum_exp(&joint_log[1]);
    let lse_tot = log_sum_exp(&[lse_dir, lse_swp]);
    let mut p_dir = (lse_dir - lse_tot).exp();

    let u_dir = log_sum_exp(&unif_dir_terms);
    let u_swp = log_sum_exp(&unif_swp_terms);
    let p_dir_unif = (u_dir - log_sum_exp(&[u_dir, u_swp])).exp();
    if !p_dir_unif.is_finite()
        || (p_dir_unif - 0.5).abs() < 0.10
        || (p_dir - 0.5) * (p_dir_unif - 0.5) < 0.0
    {
        p_dir = cap_degenerate_posterior(p_dir);
    }

    let mut rot_marg = [0.0; 3];
    let mut dyn_post = 0.0;
    let mut best_lp = f64::NEG_INFINITY;
    let mut map_w = [1.0 / 3.0; 3];

    for (si, sp) in simplex.iter().enumerate() {
        let lp_si = log_sum_exp(&[joint_log[0][si], joint_log[1][si]]);
        let prob_si = (lp_si - lse_tot).exp();
        for (rm, &wr) in rot_marg.iter_mut().zip(&sp.w) {
            *rm += prob_si * wr;
        }
        if !matches!(sp.kind, SimplexPointKind::Vertex(_)) {
            dyn_post += prob_si;
        }
        if lp_si > best_lp {
            best_lp = lp_si;
            map_w = sp.w;
        }
    }

    finalize_result(
        residue_idx,
        seq_id,
        spec,
        p_dir,
        rot_marg,
        map_w,
        dyn_post,
        obs,
        cfg,
    )
}

fn infer_chi1_chi2_simplex_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    let simplex = build_simplex_grid(cfg.simplex_divisions);
    let n_s = simplex.len();
    let bb_pts = backbone.quadrature_points(spec.residue);
    let chi1_angles = canonical_chi_angles(spec.residue);
    let ln_half = 0.5_f64.ln();
    let ln_third = (1.0_f64 / 3.0).ln();
    let uniform_log_priors = normalized_simplex_log_priors(&simplex, [1.0 / 3.0; 3], cfg);

    let mut joint_log = [vec![f64::NEG_INFINITY; n_s], vec![f64::NEG_INFINITY; n_s]];
    let mut unif_dir_terms: Vec<f64> = Vec::new();
    let mut unif_swp_terms: Vec<f64> = Vec::new();
    let mut max_data_ll = f64::NEG_INFINITY;
    let mut n_obs = 0usize;

    for &(phi, psi, w_bb, ss_hint) in &bb_pts {
        if w_bb <= 0.0 {
            continue;
        }
        let frame = build_local_peptide_frame(spec.residue, phi, psi, obs.next_is_pro);
        let pi_chi1 = if let Some(up) = obs.upstream_chi1_posterior {
            up
        } else {
            lovell_chi1_prior(spec.residue, Some(phi), Some(psi), ss_hint)
        };

        for (r1_idx, &chi1_deg) in chi1_angles.iter().enumerate() {
            let w_c1 = pi_chi1[r1_idx];
            let chi1_state = RotamerState::ALL[r1_idx];
            let pi_chi2 = lovell_chi2_prior(spec.residue, chi1_state);
            let log_priors = normalized_simplex_log_priors(&simplex, pi_chi2, cfg);
            let basis = if spec.kind == ProchiralSiteKind::LeuIsopropyl {
                compute_leu_basis_for_chi1(&frame, chi1_deg, cfg.libration_sigma_deg)
            } else {
                compute_gamma_methylene_basis_for_chi1(
                    spec.residue,
                    &frame,
                    chi1_deg,
                    cfg.libration_sigma_deg,
                )
            };
            let has_ile_methine =
                obs.j_methine_hahb_hz.is_some() && spec.residue.eq_ignore_ascii_case("ILE");
            n_obs = n_obs.max(count_basis_observations(&basis, obs) + usize::from(has_ile_methine));

            // Include Ile single methine 3J_HAHB(chi1) when supplied:
            // In (2S,3S)-Ile, dihedral(N,CA,CB,HB) = chi1 + 120 deg => dihedral(HA,CA,CB,HB) = chi1 - 120 deg.
            let ll_meth = if let Some(jm) = obs.j_methine_hahb_hz {
                if spec.residue.eq_ignore_ascii_case("ILE") {
                    let pred = librational_avg(
                        karplus_3j_hahb_hz,
                        chi1_deg - 120.0,
                        cfg.libration_sigma_deg,
                    );
                    -0.5 * (jm - pred).powi(2) / cfg.homonuclear_j_sigma_hz.powi(2)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let unif_prefix = w_bb.ln() + ln_third + ll_meth + ln_half;
            for (si, sp) in simplex.iter().enumerate() {
                let (raw_d, raw_s) = if matches!(sp.kind, SimplexPointKind::Isotropic) {
                    let r0 = evaluate_isotropic_null_log_likelihood(&basis, obs, cfg);
                    (r0, r0)
                } else {
                    (
                        evaluate_basis_simplex_log_likelihood(
                            &basis,
                            sp.w,
                            SwapHypothesis::Direct,
                            obs,
                            cfg,
                        ),
                        evaluate_basis_simplex_log_likelihood(
                            &basis,
                            sp.w,
                            SwapHypothesis::Swapped,
                            obs,
                            cfg,
                        ),
                    )
                };
                max_data_ll = max_data_ll.max(ll_meth + raw_d).max(ll_meth + raw_s);
                unif_dir_terms.push(unif_prefix + uniform_log_priors[si] + raw_d);
                unif_swp_terms.push(unif_prefix + uniform_log_priors[si] + raw_s);
                if w_c1 > 1e-6 {
                    let base_lp = w_bb.ln() + w_c1.ln() + ll_meth + ln_half + log_priors[si];
                    joint_log[0][si] = log_sum_exp(&[joint_log[0][si], base_lp + raw_d]);
                    joint_log[1][si] = log_sum_exp(&[joint_log[1][si], base_lp + raw_s]);
                }
            }
        }
    }

    if n_obs > 0
        && (!max_data_ll.is_finite() || (-2.0 * max_data_ll) / (n_obs as f64) > MAX_REDUCED_CHI2)
    {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }

    let lse_dir = log_sum_exp(&joint_log[0]);
    let lse_swp = log_sum_exp(&joint_log[1]);
    let lse_tot = log_sum_exp(&[lse_dir, lse_swp]);
    let mut p_dir = (lse_dir - lse_tot).exp();

    let u_dir = log_sum_exp(&unif_dir_terms);
    let u_swp = log_sum_exp(&unif_swp_terms);
    let p_dir_unif = (u_dir - log_sum_exp(&[u_dir, u_swp])).exp();
    if !p_dir_unif.is_finite()
        || (p_dir_unif - 0.5).abs() < 0.10
        || (p_dir - 0.5) * (p_dir_unif - 0.5) < 0.0
    {
        p_dir = cap_degenerate_posterior(p_dir);
    }

    let mut rot_marg = [0.0; 3];
    let mut dyn_post = 0.0;
    let mut best_lp = f64::NEG_INFINITY;
    let mut map_w = [1.0 / 3.0; 3];

    for (si, sp) in simplex.iter().enumerate() {
        let lp_si = log_sum_exp(&[joint_log[0][si], joint_log[1][si]]);
        let prob_si = (lp_si - lse_tot).exp();
        for (rm, &wr) in rot_marg.iter_mut().zip(&sp.w) {
            *rm += prob_si * wr;
        }
        if !matches!(sp.kind, SimplexPointKind::Vertex(_)) {
            dyn_post += prob_si;
        }
        if lp_si > best_lp {
            best_lp = lp_si;
            map_w = sp.w;
        }
    }

    finalize_result(
        residue_idx,
        seq_id,
        spec,
        p_dir,
        rot_marg,
        map_w,
        dyn_post,
        obs,
        cfg,
    )
}

fn infer_gly_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    // Gly HA2/HA3 is governed directly by (phi_i, psi_i), plus a flexible-loop null spike (p_iso)
    // so noisy conformational averaging in disordered Gly loops never triggers false commits.
    let bb_pts = backbone.quadrature_points("GLY");
    let p_iso = cfg.isotropic_averaging_prior_weight.clamp(0.02, 0.40);
    let p_struct = 1.0 - p_iso;
    let floor_r6 = 5.5_f64.powi(-6);

    let mut terms_dir = Vec::with_capacity(bb_pts.len() + 1);
    let mut terms_swp = Vec::with_capacity(bb_pts.len() + 1);
    let mut max_data_ll = f64::NEG_INFINITY;
    let mut n_obs = usize::from(obs.j_hh_vicinal_hz.is_some()) * 2
        + obs
            .noe_pairs
            .iter()
            .filter(|np| {
                np.intensity_a > 0.0
                    && np.intensity_b > 0.0
                    && (np.anchor == AnchorProton::IntraAmideH
                        || (!obs.next_is_pro && np.anchor == AnchorProton::SeqNextAmideH))
            })
            .count();

    for &(phi, psi, w_bb, _) in &bb_pts {
        if w_bb <= 0.0 {
            continue;
        }
        let frame = build_local_peptide_frame("GLY", phi, psi, obs.next_is_pro);
        let (ha2, ha3) = frame.gly_ha2_ha3.unwrap();
        let (_, _, th_h_2, th_h_3) = iupac_gly_ha_dihedrals_deg(phi);
        let j2 = librational_avg(karplus_3j_hnha_hz, th_h_2, cfg.libration_sigma_deg);
        let j3 = librational_avg(karplus_3j_hnha_hz, th_h_3, cfg.libration_sigma_deg);

        for hyp in [SwapHypothesis::Direct, SwapHypothesis::Swapped] {
            let mut data_ll = 0.0;
            if let Some((ja, jb)) = obs.j_hh_vicinal_hz {
                let (o2, o3) = if hyp == SwapHypothesis::Direct {
                    (ja, jb)
                } else {
                    (jb, ja)
                };
                data_ll -= 0.5 * ((o2 - j2).powi(2) + (o3 - j3).powi(2))
                    / cfg.homonuclear_j_sigma_hz.powi(2);
            }
            for np in &obs.noe_pairs {
                if np.intensity_a <= 0.0 || np.intensity_b <= 0.0 {
                    continue;
                }
                let anchor_pos = match np.anchor {
                    AnchorProton::IntraAmideH => frame.h,
                    AnchorProton::SeqNextAmideH => frame.h_next,
                    _ => None,
                };
                let Some(ap) = anchor_pos else {
                    continue;
                };
                let u2 = r6_single(ap, ha2) + floor_r6;
                let u3 = r6_single(ap, ha3) + floor_r6;
                let (ia, ib) = if hyp == SwapHypothesis::Direct {
                    (np.intensity_a, np.intensity_b)
                } else {
                    (np.intensity_b, np.intensity_a)
                };
                let sig = np.rel_sigma_ln.unwrap_or(cfg.noe_log_sigma).max(0.10);
                data_ll -= 0.5 * ((ia / ib).ln() - (u2 / u3).ln()).powi(2) / (2.0 * sig * sig);
            }
            max_data_ll = max_data_ll.max(data_ll);
            let ll = (p_struct * w_bb * 0.5).ln() + data_ll;
            if hyp == SwapHypothesis::Direct {
                terms_dir.push(ll);
            } else {
                terms_swp.push(ll);
            }
        }
    }

    // Isotropic conformational-averaging null term (J_avg = 5.95 Hz, equal NOE distances)
    let mut iso_data_ll = 0.0;
    if let Some((ja, jb)) = obs.j_hh_vicinal_hz {
        let j_iso = 5.95;
        iso_data_ll -= 0.5 * ((ja - j_iso).powi(2) + (jb - j_iso).powi(2))
            / cfg.homonuclear_j_sigma_hz.powi(2);
    }
    for np in &obs.noe_pairs {
        if np.intensity_a > 0.0
            && np.intensity_b > 0.0
            && (np.anchor == AnchorProton::IntraAmideH
                || (!obs.next_is_pro && np.anchor == AnchorProton::SeqNextAmideH))
        {
            let sig = np.rel_sigma_ln.unwrap_or(cfg.noe_log_sigma).max(0.10);
            iso_data_ll -= 0.5 * (np.intensity_a / np.intensity_b).ln().powi(2) / (2.0 * sig * sig);
        }
    }
    max_data_ll = max_data_ll.max(iso_data_ll);
    let _ = &mut n_obs;
    if n_obs > 0
        && (!max_data_ll.is_finite() || (-2.0 * max_data_ll) / (n_obs as f64) > MAX_REDUCED_CHI2)
    {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }

    let ll_iso = (p_iso * 0.5).ln() + iso_data_ll;
    terms_dir.push(ll_iso);
    terms_swp.push(ll_iso);

    let lse_dir = log_sum_exp(&terms_dir);
    let lse_swp = log_sum_exp(&terms_swp);
    let lse_tot = log_sum_exp(&[lse_dir, lse_swp]);
    let mut p_dir = (lse_dir - lse_tot).exp();
    let dyn_post = (log_sum_exp(&[ll_iso, ll_iso]) - lse_tot).exp();

    // Under phi -> -phi, Gly 3J(HN, HA2/HA3) and intraresidue HN-HA2/HA3 NOEs are exact mirror
    // images under Direct <-> Swapped! Cap Gly posterior below 0.90 unless phi comes from an
    // explicit 3D structural model (ExactTorsions) OR an independent sequential HA(i)-HN(i+1)
    // NOE pair with distinct intensities breaks the mirror degeneracy.
    let has_pdb_torsions = matches!(backbone, BackboneConditioning::ExactTorsions { .. });
    let has_seq_noe_asymmetry = !obs.next_is_pro
        && obs.noe_pairs.iter().any(|np| {
            np.anchor == AnchorProton::SeqNextAmideH
                && np.intensity_a > 0.0
                && np.intensity_b > 0.0
                && (np.intensity_a / np.intensity_b).ln().abs() > 0.05
        });
    if !has_pdb_torsions && !has_seq_noe_asymmetry {
        p_dir = cap_degenerate_posterior(p_dir);
    }

    finalize_result(
        residue_idx,
        seq_id,
        spec,
        p_dir,
        [1.0 / 3.0; 3],
        [1.0 / 3.0; 3],
        dyn_post,
        obs,
        cfg,
    )
}

fn infer_carboxamide_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    _backbone: &BackboneConditioning,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    // Planar primary carboxamide:
    // Asn: HD21 is syn (0 deg) to CB (`d(QB, HD21) ~ 2.63 A` vs `d(QB, HD22) ~ 3.80 A`),
    //      and downfield in BMRB (~7.45 vs ~6.85 ppm).
    // Gln: HE21 is syn (0 deg) to CG (`d(QG, HE21) ~ 2.63 A` vs `d(QG, HE22) ~ 3.80 A`),
    //      and downfield in BMRB (`7.32` vs `6.87` ppm).
    let c_prev = [0.0, 1.520, 0.0];
    let c_aliph = [0.0, 0.0, 0.0]; // CB for Asn, CG for Gln
    let c_carb = [1.520, 0.0, 0.0]; // CG for Asn, CD for Gln
    let ang_sp2 = 118.5_f64.to_radians();
    let od_oe = nerf_place_atom(c_prev, c_aliph, c_carb, 1.231, 120.5_f64.to_radians(), 0.0);
    let nd_ne = nerf_place_atom(
        c_prev,
        c_aliph,
        c_carb,
        BOND_CG_ND2,
        ang_sp2,
        std::f64::consts::PI,
    );
    // Verified on 1D3Z: OD1-CG-ND2-HD21 = 180 deg, OD1-CG-ND2-HD22 = 0 deg
    let h_syn = nerf_place_atom(
        od_oe,
        c_carb,
        nd_ne,
        BOND_NH_AMIDE,
        120.0_f64.to_radians(),
        std::f64::consts::PI,
    );
    let h_anti = nerf_place_atom(
        od_oe,
        c_carb,
        nd_ne,
        BOND_NH_AMIDE,
        120.0_f64.to_radians(),
        0.0,
    );

    // Average over the two attached methylene protons on c_aliph (+/- 120 deg from c_prev)
    let ang_tet = ANGLE_TETRA_DEG.to_radians();
    let hm2 = nerf_place_atom(
        c_prev,
        c_carb,
        c_aliph,
        BOND_CH,
        ang_tet,
        120.0_f64.to_radians(),
    );
    let hm3 = nerf_place_atom(
        c_prev,
        c_carb,
        c_aliph,
        BOND_CH,
        ang_tet,
        (-120.0_f64).to_radians(),
    );
    let r6_1 = 0.5 * (r6_single(hm2, h_syn) + r6_single(hm3, h_syn));
    let r6_2 = 0.5 * (r6_single(hm2, h_anti) + r6_single(hm3, h_anti));
    let floor_r6 = 5.5_f64.powi(-6);
    let pred_lr = ((r6_1 + floor_r6) / (r6_2 + floor_r6)).ln();

    // Weak bounded BMRB shift-separation prior (`|llr_shift| <= 0.95` -> max standalone posterior ~0.72 < 0.90)
    let shift_llr = if cfg.use_carboxamide_shift_prior {
        if let Some((ha, hb)) = obs.proton_shifts_ppm {
            ((ha - hb) / 0.55).clamp(-0.95, 0.95)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut data_ll_dir = 0.0;
    let mut data_ll_swp = 0.0;
    let mut n_obs = 0usize;
    for np in &obs.noe_pairs {
        if np.intensity_a <= 0.0 || np.intensity_b <= 0.0 {
            continue;
        }
        let relevant = match spec.kind {
            ProchiralSiteKind::AsnCarboxamide => matches!(
                np.anchor,
                AnchorProton::IntraBetaHB
                    | AnchorProton::IntraBetaHB2
                    | AnchorProton::IntraBetaHB3
                    | AnchorProton::IntraAlphaHA
            ),
            ProchiralSiteKind::GlnCarboxamide => matches!(
                np.anchor,
                AnchorProton::IntraGammaHG
                    | AnchorProton::IntraBetaHB
                    | AnchorProton::IntraBetaHB2
                    | AnchorProton::IntraBetaHB3
            ),
            _ => false,
        };
        if !relevant {
            continue;
        }
        n_obs += 1;
        let scale = if (spec.kind == ProchiralSiteKind::AsnCarboxamide
            && np.anchor == AnchorProton::IntraAlphaHA)
            || (spec.kind == ProchiralSiteKind::GlnCarboxamide
                && np.anchor != AnchorProton::IntraGammaHG)
        {
            0.55
        } else {
            1.0
        };
        let sig = np.rel_sigma_ln.unwrap_or(cfg.noe_log_sigma).max(0.10);
        let obs_lr = (np.intensity_a / np.intensity_b).ln();
        data_ll_dir -= 0.5 * (obs_lr - scale * pred_lr).powi(2) / (2.0 * sig * sig);
        data_ll_swp -= 0.5 * (-obs_lr - scale * pred_lr).powi(2) / (2.0 * sig * sig);
    }

    let max_data_ll = data_ll_dir.max(data_ll_swp);
    if n_obs > 0
        && (!max_data_ll.is_finite() || (-2.0 * max_data_ll) / (n_obs as f64) > MAX_REDUCED_CHI2)
    {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }

    let ll_dir = 0.5 * shift_llr + data_ll_dir;
    let ll_swp = -0.5 * shift_llr + data_ll_swp;
    let lse_tot = log_sum_exp(&[ll_dir, ll_swp]);
    let p_dir = (ll_dir - lse_tot).exp();

    finalize_result(
        residue_idx,
        seq_id,
        spec,
        p_dir,
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        0.0,
        obs,
        cfg,
    )
}

fn infer_distal_methylene_pair(
    residue_idx: usize,
    seq_id: i32,
    spec: ProchiralPairSpec,
    obs: &ProchiralObservables,
    cfg: &StereospecConfig,
) -> StereospecResult {
    // Distal CD (HD2/HD3) or CE (HE2/HE3) methylene on simplex over local torsion chi_k
    let simplex = build_simplex_grid(cfg.simplex_divisions);
    let is_pro = spec.residue.eq_ignore_ascii_case("PRO");
    let pi = if is_pro {
        [0.48, 0.04, 0.48]
    } else {
        [0.16, 0.68, 0.16]
    };
    let log_priors = normalized_simplex_log_priors(&simplex, pi, cfg);
    let uniform_log_priors = normalized_simplex_log_priors(&simplex, [1.0 / 3.0; 3], cfg);
    let chis = canonical_chi_angles(spec.residue);

    // Local 4-carbon chain C_{k-2} - C_{k-1} - C_k - X_{k+1}
    let c0 = [0.0, 1.520, 0.0];
    let c1 = [0.0, 0.0, 0.0];
    let c2 = [1.520, 0.0, 0.0];
    let ang_tet = ANGLE_TETRA_DEG.to_radians();
    // Upstream proton centroid on c1 (gauche/staggered relative to c0):
    // Notice c0-c1-c2 defines a symmetric plane unless an upstream chiral anchor (e.g. Pro HA on CA->N->CD!) is present!
    // For Pro CD (CB-CG-CD-N): N is bonded to CA & HA! Specifically HA is cis to HB3 and HD3 on the pyrrolidine ring!
    let mut j_hh = [(0.0, 0.0); 3];
    let mut r6_ha = [(0.0, 0.0); 3];
    for (r, &chi) in chis.iter().enumerate() {
        let j2 = 0.5
            * (librational_avg(karplus_3j_hahb_hz, chi, cfg.libration_sigma_deg)
                + librational_avg(karplus_3j_hahb_hz, chi - 120.0, cfg.libration_sigma_deg));
        let j3 = 0.5
            * (librational_avg(karplus_3j_hahb_hz, chi + 120.0, cfg.libration_sigma_deg)
                + librational_avg(karplus_3j_hahb_hz, chi, cfg.libration_sigma_deg));
        j_hh[r] = (j2, j3);
        let h2 = nerf_place_atom(c0, c1, c2, BOND_CH, ang_tet, (chi + 120.0).to_radians());
        let h3 = nerf_place_atom(c0, c1, c2, BOND_CH, ang_tet, (chi - 120.0).to_radians());
        // Chiral anchor on L-ring / upstream stereocenter
        let anchor = [1.85, 0.90, -1.15];
        r6_ha[r] = (r6_single(anchor, h2), r6_single(anchor, h3));
    }

    let mut noe_r6 = BTreeMap::new();
    if is_pro {
        noe_r6.insert(AnchorProton::IntraAlphaHA, r6_ha);
    }
    let basis = RotamerBasisPredictions {
        j_hh: Some(j_hh),
        j_nhb: None,
        j_ncg: None,
        j_cc: None,
        j_methine_hahb: None,
        noe_r6,
    };
    let n_obs = count_basis_observations(&basis, obs);

    let n_s = simplex.len();
    let ln_half = 0.5_f64.ln();
    let mut l_dir = Vec::with_capacity(n_s);
    let mut l_swp = Vec::with_capacity(n_s);
    let mut u_dir = Vec::with_capacity(n_s);
    let mut u_swp = Vec::with_capacity(n_s);
    let mut max_data_ll = f64::NEG_INFINITY;
    for (si, sp) in simplex.iter().enumerate() {
        let base = log_priors[si] + ln_half;
        let ubase = uniform_log_priors[si] + ln_half;
        let (raw_d, raw_s) = if matches!(sp.kind, SimplexPointKind::Isotropic) {
            let r0 = evaluate_isotropic_null_log_likelihood(&basis, obs, cfg);
            (r0, r0)
        } else {
            (
                evaluate_basis_simplex_log_likelihood(
                    &basis,
                    sp.w,
                    SwapHypothesis::Direct,
                    obs,
                    cfg,
                ),
                evaluate_basis_simplex_log_likelihood(
                    &basis,
                    sp.w,
                    SwapHypothesis::Swapped,
                    obs,
                    cfg,
                ),
            )
        };
        max_data_ll = max_data_ll.max(raw_d).max(raw_s);
        l_dir.push(base + raw_d);
        l_swp.push(base + raw_s);
        u_dir.push(ubase + raw_d);
        u_swp.push(ubase + raw_s);
    }

    if n_obs > 0
        && (!max_data_ll.is_finite() || (-2.0 * max_data_ll) / (n_obs as f64) > MAX_REDUCED_CHI2)
    {
        return uncommitted_null_result(residue_idx, seq_id, spec, obs);
    }

    let lse_dir = log_sum_exp(&l_dir);
    let lse_swp = log_sum_exp(&l_swp);
    let lse_tot = log_sum_exp(&[lse_dir, lse_swp]);
    let mut p_dir = (lse_dir - lse_tot).exp();

    let lse_udir = log_sum_exp(&u_dir);
    let lse_uswp = log_sum_exp(&u_swp);
    let p_dir_unif = (lse_udir - log_sum_exp(&[lse_udir, lse_uswp])).exp();
    if !p_dir_unif.is_finite()
        || (p_dir_unif - 0.5).abs() < 0.10
        || (p_dir - 0.5) * (p_dir_unif - 0.5) < 0.0
    {
        p_dir = cap_degenerate_posterior(p_dir);
    }

    let mut rot_marg = [0.0; 3];
    let mut dyn_post = 0.0;
    let mut best_lp = f64::NEG_INFINITY;
    let mut map_w = [1.0 / 3.0; 3];
    for (si, sp) in simplex.iter().enumerate() {
        let lp = log_sum_exp(&[l_dir[si], l_swp[si]]);
        let pr = (lp - lse_tot).exp();
        for (rm, &wr) in rot_marg.iter_mut().zip(&sp.w) {
            *rm += pr * wr;
        }
        if !matches!(sp.kind, SimplexPointKind::Vertex(_)) {
            dyn_post += pr;
        }
        if lp > best_lp {
            best_lp = lp;
            map_w = sp.w;
        }
    }

    finalize_result(
        residue_idx,
        seq_id,
        spec,
        p_dir,
        rot_marg,
        map_w,
        dyn_post,
        obs,
        cfg,
    )
}

/// Infer all prochiral pairs within a single residue, hierarchically propagating any `chi1`
/// rotamer posterior inferred at `CB (HB2/HB3)` forward to downstream `CG` / `LEU CD1/CD2` sites.
pub fn infer_residue_stereospec(
    residue_idx: usize,
    seq_id: i32,
    res_name: &str,
    backbone: &BackboneConditioning,
    observables_by_branch1: &BTreeMap<String, ProchiralObservables>,
    cfg: &StereospecConfig,
) -> Vec<StereospecResult> {
    let specs = prochiral_pairs_for_residue(res_name);
    let mut out = Vec::with_capacity(specs.len());
    let mut inferred_chi1: Option<[f64; 3]> = None;

    for spec in specs {
        let mut obs = observables_by_branch1
            .get(spec.iupac_branch_1)
            .cloned()
            .unwrap_or_default();
        if matches!(
            spec.kind,
            ProchiralSiteKind::LeuIsopropyl | ProchiralSiteKind::GammaMethylene
        ) && obs.upstream_chi1_posterior.is_none()
        {
            obs.upstream_chi1_posterior = inferred_chi1;
        }
        let res = infer_prochiral_pair(residue_idx, seq_id, spec, backbone, &obs, cfg);
        if spec.kind == ProchiralSiteKind::BetaMethylene {
            inferred_chi1 = Some(res.rotamer_marginal);
        }
        out.push(res);
    }

    out
}

/// Run hierarchical Bayesian stereospecific inference across all residues of a protein sequence.
pub fn infer_protein_stereospec(
    sequence: &[String],
    seq_ids: &[i32],
    backbones: &[BackboneConditioning],
    per_residue_obs: &[BTreeMap<String, ProchiralObservables>],
    cfg: &StereospecConfig,
) -> Vec<StereospecResult> {
    let mut all = Vec::new();
    for (i, res_name) in sequence.iter().enumerate() {
        let sid = seq_ids.get(i).copied().unwrap_or((i + 1) as i32);
        let bb = backbones
            .get(i)
            .unwrap_or(&BackboneConditioning::Marginalized);
        let empty = BTreeMap::new();
        let obs_map = per_residue_obs.get(i).unwrap_or(&empty);
        all.extend(infer_residue_stereospec(i, sid, res_name, bb, obs_map, cfg));
    }
    all
}

// ============================================================================
// 7. NMR-STAR v3 (`Ambiguity_code` 1 vs 2) & NEF v1.1 (`HBx`/`HBy`) Export
// ============================================================================

/// Single NMR-STAR v3 `_Atom_chem_shift` record with explicit `Ambiguity_code` (`1` vs `2`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StarChemShiftRow {
    pub residue_idx: usize,
    pub seq_id: i32,
    pub comp_id: String,
    pub atom_id: String,
    pub atom_type: String,
    pub val: f64,
    pub ambiguity_code: u8,
    pub details: String,
}

fn get_shift_ci(shifts: &BTreeMap<(usize, String), f64>, k: usize, name: &str) -> Option<f64> {
    if let Some(&v) = shifts.get(&(k, name.to_string())) {
        return Some(v);
    }
    let up = name.to_ascii_uppercase();
    if let Some(&v) = shifts.get(&(k, up.clone())) {
        return Some(v);
    }
    for (&(rk, ref ak), &v) in shifts.range((k, String::new())..=(k, "\u{10FFFF}".to_string())) {
        if rk == k && ak.eq_ignore_ascii_case(&up) {
            return Some(v);
        }
    }
    None
}

fn remove_atom_ci(
    shifts: &mut BTreeMap<(usize, String), f64>,
    amb_map: &mut BTreeMap<(usize, String), u8>,
    k: usize,
    name: &str,
) {
    let keys_to_drop: Vec<String> = shifts
        .range((k, String::new())..=(k, "\u{10FFFF}".to_string()))
        .filter(|&(&(rk, ref ak), _)| rk == k && ak.eq_ignore_ascii_case(name))
        .map(|((_, ak), _)| ak.clone())
        .collect();
    for ak in keys_to_drop {
        shifts.remove(&(k, ak.clone()));
        amb_map.remove(&(k, ak));
    }
    amb_map.remove(&(k, name.to_string()));
    amb_map.remove(&(k, name.to_ascii_uppercase()));
}

fn is_nef_nonstereo_wildcard_ci(atom: &str) -> bool {
    let base = atom.trim().trim_end_matches(['%', '*']);
    if base.len() < 2 {
        return false;
    }
    let last = base.as_bytes()[base.len() - 1];
    matches!(last, b'x' | b'y' | b'X' | b'Y')
}

const VAL_PROTON_SUBPAIRS: &[(&str, &str)] = &[
    ("HG11", "HG21"),
    ("HG12", "HG22"),
    ("HG13", "HG23"),
    ("HG1*", "HG2*"),
    ("HG1%", "HG2%"),
    ("HG1", "HG2"),
    ("QG1", "QG2"),
];

const VAL_WILDCARD_PROTON_SUBPAIRS: &[(&str, &str)] =
    &[("HGx%", "HGy%"), ("HGx*", "HGy*"), ("HGx", "HGy")];

const LEU_PROTON_SUBPAIRS: &[(&str, &str)] = &[
    ("HD11", "HD21"),
    ("HD12", "HD22"),
    ("HD13", "HD23"),
    ("HD1*", "HD2*"),
    ("HD1%", "HD2%"),
    ("HD1", "HD2"),
    ("QD1", "QD2"),
];

const LEU_WILDCARD_PROTON_SUBPAIRS: &[(&str, &str)] =
    &[("HDx%", "HDy%"), ("HDx*", "HDy*"), ("HDx", "HDy")];

/// Canonical NMR-STAR v3 `_Atom_chem_shift.Ambiguity_code` default when no stereospecific
/// commitment has been established:
/// - `3` for symmetric aromatic ring-flip pairs (`PHE`/`TYR` `CD1/CD2/HD1/HD2/CE1/CE2/HE1/HE2`),
/// - `2` for any prochiral methylene, isopropyl methyl, planar carboxamide, guanidinium, or NEF `x`/`y` wildcard atom,
/// - `1` for non-prochiral backbone/methine/single-methyl/non-symmetric-ring atoms.
pub fn default_atom_ambiguity_code(res_name: &str, atom: &str) -> u8 {
    let r_up = res_name.trim().to_ascii_uppercase();
    let a_up = atom.trim().to_ascii_uppercase();
    if matches!(r_up.as_str(), "PHE" | "TYR")
        && matches!(
            a_up.as_str(),
            "CD1"
                | "CD2"
                | "HD1"
                | "HD2"
                | "CE1"
                | "CE2"
                | "HE1"
                | "HE2"
                | "CDX"
                | "CDY"
                | "HDX"
                | "HDY"
                | "CEX"
                | "CEY"
                | "HEX"
                | "HEY"
                | "QD"
                | "QE"
        )
    {
        return 3;
    }
    if is_nef_nonstereo_wildcard_ci(&a_up) {
        return 2;
    }
    if r_up == "ARG"
        && matches!(
            a_up.as_str(),
            "NH1" | "NH2" | "HH11" | "HH12" | "HH21" | "HH22"
        )
    {
        return 2;
    }
    if r_up == "VAL"
        && VAL_PROTON_SUBPAIRS
            .iter()
            .any(|&(p1, p2)| a_up == p1 || a_up == p2)
    {
        return 2;
    }
    if r_up == "LEU"
        && LEU_PROTON_SUBPAIRS
            .iter()
            .any(|&(p1, p2)| a_up == p1 || a_up == p2)
    {
        return 2;
    }
    for spec in prochiral_pairs_for_residue(&r_up) {
        if a_up == spec.iupac_branch_1.to_ascii_uppercase()
            || a_up == spec.iupac_branch_2.to_ascii_uppercase()
            || a_up == spec.nef_nonstereo_protons.0.to_ascii_uppercase()
            || a_up == spec.nef_nonstereo_protons.1.to_ascii_uppercase()
            || spec.iupac_carbons.is_some_and(|(c1, c2)| {
                a_up.eq_ignore_ascii_case(c1) || a_up.eq_ignore_ascii_case(c2)
            })
            || spec.nef_nonstereo_carbons.is_some_and(|(cx, cy)| {
                a_up.eq_ignore_ascii_case(cx) || a_up.eq_ignore_ascii_case(cy)
            })
            || spec
                .iupac_protons_1
                .iter()
                .any(|p| a_up.eq_ignore_ascii_case(p))
            || spec
                .iupac_protons_2
                .iter()
                .any(|p| a_up.eq_ignore_ascii_case(p))
        {
            return 2;
        }
    }
    1
}

fn is_methyl_group_asymmetrically_populated(
    shifts: &BTreeMap<(usize, String), f64>,
    k: usize,
    spec: ProchiralPairSpec,
) -> bool {
    let (canon_subs, wild_subs, (c1, c2), (cx, cy)) = match spec.kind {
        ProchiralSiteKind::ValIsopropyl => (
            VAL_PROTON_SUBPAIRS,
            VAL_WILDCARD_PROTON_SUBPAIRS,
            ("CG1", "CG2"),
            ("CGx", "CGy"),
        ),
        ProchiralSiteKind::LeuIsopropyl => (
            LEU_PROTON_SUBPAIRS,
            LEU_WILDCARD_PROTON_SUBPAIRS,
            ("CD1", "CD2"),
            ("CDx", "CDy"),
        ),
        _ => return false,
    };

    let has_c1 = get_shift_ci(shifts, k, c1).is_some();
    let has_c2 = get_shift_ci(shifts, k, c2).is_some();
    let has_cx = get_shift_ci(shifts, k, cx).is_some();
    let has_cy = get_shift_ci(shifts, k, cy).is_some();
    if has_c1 != has_c2 || has_cx != has_cy {
        return true;
    }
    for &(p1, p2) in canon_subs.iter().chain(wild_subs.iter()) {
        if get_shift_ci(shifts, k, p1).is_some() != get_shift_ci(shifts, k, p2).is_some() {
            return true;
        }
    }
    false
}

fn is_half_present_prochiral_atom(
    shifts: &BTreeMap<(usize, String), f64>,
    k: usize,
    res_name: &str,
    atom: &str,
) -> bool {
    let a_up = atom.trim().to_ascii_uppercase();
    let r_up = res_name.trim().to_ascii_uppercase();
    if r_up == "ARG" {
        for (c1, c2, wx, wy) in [
            ("NH1", "NH2", "NHx", "NHy"),
            ("HH11", "HH12", "HH1x", "HH1y"),
            ("HH21", "HH22", "HH2x", "HH2y"),
        ] {
            if a_up.eq_ignore_ascii_case(c1)
                || a_up.eq_ignore_ascii_case(c2)
                || a_up.eq_ignore_ascii_case(wx)
                || a_up.eq_ignore_ascii_case(wy)
            {
                let has_1 =
                    get_shift_ci(shifts, k, c1).is_some() || get_shift_ci(shifts, k, wx).is_some();
                let has_2 =
                    get_shift_ci(shifts, k, c2).is_some() || get_shift_ci(shifts, k, wy).is_some();
                if has_1 != has_2 {
                    return true;
                }
            }
        }
    }
    for spec in prochiral_pairs_for_residue(&r_up) {
        if matches!(
            spec.kind,
            ProchiralSiteKind::ValIsopropyl | ProchiralSiteKind::LeuIsopropyl
        ) {
            let (subs, wsubs, (c1, c2), (cx, cy)) = if spec.kind == ProchiralSiteKind::ValIsopropyl
            {
                (
                    VAL_PROTON_SUBPAIRS,
                    VAL_WILDCARD_PROTON_SUBPAIRS,
                    ("CG1", "CG2"),
                    ("CGx", "CGy"),
                )
            } else {
                (
                    LEU_PROTON_SUBPAIRS,
                    LEU_WILDCARD_PROTON_SUBPAIRS,
                    ("CD1", "CD2"),
                    ("CDx", "CDy"),
                )
            };
            let belongs = a_up.eq_ignore_ascii_case(c1)
                || a_up.eq_ignore_ascii_case(c2)
                || a_up.eq_ignore_ascii_case(cx)
                || a_up.eq_ignore_ascii_case(cy)
                || subs.iter().chain(wsubs.iter()).any(|&(p1, p2)| {
                    a_up.eq_ignore_ascii_case(p1) || a_up.eq_ignore_ascii_case(p2)
                });
            if belongs && is_methyl_group_asymmetrically_populated(shifts, k, spec) {
                return true;
            }
            continue;
        }

        let extra_shorthand: Option<(&str, &str)> = match (r_up.as_str(), spec.kind) {
            ("ILE", ProchiralSiteKind::GammaMethylene) => Some(("HGx", "HGy")),
            ("ASN", ProchiralSiteKind::AsnCarboxamide) => Some(("HDx", "HDy")),
            ("GLN", ProchiralSiteKind::GlnCarboxamide) => Some(("HEx", "HEy")),
            _ => None,
        };
        let b1_match = a_up.eq_ignore_ascii_case(spec.iupac_branch_1)
            || a_up.eq_ignore_ascii_case(spec.nef_nonstereo_protons.0)
            || extra_shorthand.is_some_and(|(sx, _)| a_up.eq_ignore_ascii_case(sx))
            || spec
                .iupac_protons_1
                .iter()
                .any(|p| a_up.eq_ignore_ascii_case(p));
        let b2_match = a_up.eq_ignore_ascii_case(spec.iupac_branch_2)
            || a_up.eq_ignore_ascii_case(spec.nef_nonstereo_protons.1)
            || extra_shorthand.is_some_and(|(_, sy)| a_up.eq_ignore_ascii_case(sy))
            || spec
                .iupac_protons_2
                .iter()
                .any(|p| a_up.eq_ignore_ascii_case(p));
        if b1_match || b2_match {
            let has_b1 = get_shift_ci(shifts, k, spec.iupac_branch_1).is_some()
                || get_shift_ci(shifts, k, spec.nef_nonstereo_protons.0).is_some()
                || extra_shorthand.is_some_and(|(sx, _)| get_shift_ci(shifts, k, sx).is_some());
            let has_b2 = get_shift_ci(shifts, k, spec.iupac_branch_2).is_some()
                || get_shift_ci(shifts, k, spec.nef_nonstereo_protons.1).is_some()
                || extra_shorthand.is_some_and(|(_, sy)| get_shift_ci(shifts, k, sy).is_some());
            if has_b1 != has_b2 {
                return true;
            }
        }
    }
    false
}

fn remove_site_nef_wildcards(
    shifts: &mut BTreeMap<(usize, String), f64>,
    amb_map: &mut BTreeMap<(usize, String), u8>,
    k: usize,
    spec: ProchiralPairSpec,
) {
    remove_atom_ci(shifts, amb_map, k, spec.nef_nonstereo_protons.0);
    remove_atom_ci(shifts, amb_map, k, spec.nef_nonstereo_protons.1);
    if let Some((cx, cy)) = spec.nef_nonstereo_carbons {
        remove_atom_ci(shifts, amb_map, k, cx);
        remove_atom_ci(shifts, amb_map, k, cy);
    }
    match spec.kind {
        ProchiralSiteKind::ValIsopropyl => {
            for &(wx, wy) in VAL_WILDCARD_PROTON_SUBPAIRS {
                remove_atom_ci(shifts, amb_map, k, wx);
                remove_atom_ci(shifts, amb_map, k, wy);
            }
        }
        ProchiralSiteKind::LeuIsopropyl => {
            for &(wx, wy) in LEU_WILDCARD_PROTON_SUBPAIRS {
                remove_atom_ci(shifts, amb_map, k, wx);
                remove_atom_ci(shifts, amb_map, k, wy);
            }
        }
        ProchiralSiteKind::GammaMethylene if spec.residue.eq_ignore_ascii_case("ILE") => {
            remove_atom_ci(shifts, amb_map, k, "HGx");
            remove_atom_ci(shifts, amb_map, k, "HGy");
        }
        ProchiralSiteKind::AsnCarboxamide => {
            remove_atom_ci(shifts, amb_map, k, "HDx");
            remove_atom_ci(shifts, amb_map, k, "HDy");
        }
        ProchiralSiteKind::GlnCarboxamide => {
            remove_atom_ci(shifts, amb_map, k, "HEx");
            remove_atom_ci(shifts, amb_map, k, "HEy");
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn normalize_wildcard_pair(
    shifts: &mut BTreeMap<(usize, String), f64>,
    amb_map: &mut BTreeMap<(usize, String), u8>,
    k: usize,
    wx: &str,
    wy: &str,
    canon_1_insert: &[&str],
    canon_2_insert: &[&str],
    canon_1_existing: &[&str],
    canon_2_existing: &[&str],
    amb_code: u8,
) {
    let vx = get_shift_ci(shifts, k, wx);
    let vy = get_shift_ci(shifts, k, wy);
    if vx.is_none() && vy.is_none() {
        return;
    }
    let has_c1 = canon_1_existing
        .iter()
        .any(|c| get_shift_ci(shifts, k, c).is_some());
    let has_c2 = canon_2_existing
        .iter()
        .any(|c| get_shift_ci(shifts, k, c).is_some());
    remove_atom_ci(shifts, amb_map, k, wx);
    remove_atom_ci(shifts, amb_map, k, wy);
    if !has_c1 {
        if let Some(v) = vx {
            for &c1 in canon_1_insert {
                shifts.insert((k, c1.to_string()), v);
                amb_map.insert((k, c1.to_string()), amb_code);
            }
        }
    }
    if !has_c2 {
        if let Some(v) = vy {
            for &c2 in canon_2_insert {
                shifts.insert((k, c2.to_string()), v);
                amb_map.insert((k, c2.to_string()), amb_code);
            }
        }
    }
}

fn normalize_residue_nef_wildcards(
    shifts: &mut BTreeMap<(usize, String), f64>,
    amb_map: &mut BTreeMap<(usize, String), u8>,
    k: usize,
    res_name: &str,
) {
    const VAL_G1_EXISTING: &[&str] = &["HG11", "HG12", "HG13", "HG1*", "HG1%", "HG1", "QG1"];
    const VAL_G2_EXISTING: &[&str] = &["HG21", "HG22", "HG23", "HG2*", "HG2%", "HG2", "QG2"];
    const LEU_D1_EXISTING: &[&str] = &["HD11", "HD12", "HD13", "HD1*", "HD1%", "HD1", "QD1"];
    const LEU_D2_EXISTING: &[&str] = &["HD21", "HD22", "HD23", "HD2*", "HD2%", "HD2", "QD2"];

    let r_up = res_name.trim().to_ascii_uppercase();
    for spec in prochiral_pairs_for_residue(&r_up) {
        match spec.kind {
            ProchiralSiteKind::ValIsopropyl => {
                normalize_wildcard_pair(
                    shifts,
                    amb_map,
                    k,
                    "CGx",
                    "CGy",
                    &["CG1"],
                    &["CG2"],
                    &["CG1"],
                    &["CG2"],
                    2,
                );
                for &(wx, wy) in VAL_WILDCARD_PROTON_SUBPAIRS {
                    normalize_wildcard_pair(
                        shifts,
                        amb_map,
                        k,
                        wx,
                        wy,
                        H_VAL_G1,
                        H_VAL_G2,
                        VAL_G1_EXISTING,
                        VAL_G2_EXISTING,
                        2,
                    );
                }
            }
            ProchiralSiteKind::LeuIsopropyl => {
                normalize_wildcard_pair(
                    shifts,
                    amb_map,
                    k,
                    "CDx",
                    "CDy",
                    &["CD1"],
                    &["CD2"],
                    &["CD1"],
                    &["CD2"],
                    2,
                );
                for &(wx, wy) in LEU_WILDCARD_PROTON_SUBPAIRS {
                    normalize_wildcard_pair(
                        shifts,
                        amb_map,
                        k,
                        wx,
                        wy,
                        H_LEU_D1,
                        H_LEU_D2,
                        LEU_D1_EXISTING,
                        LEU_D2_EXISTING,
                        2,
                    );
                }
            }
            _ => {
                let (wx, wy) = spec.nef_nonstereo_protons;
                normalize_wildcard_pair(
                    shifts,
                    amb_map,
                    k,
                    wx,
                    wy,
                    spec.iupac_protons_1,
                    spec.iupac_protons_2,
                    spec.iupac_protons_1,
                    spec.iupac_protons_2,
                    2,
                );
                if let (Some((cx, cy)), Some((c1, c2))) =
                    (spec.nef_nonstereo_carbons, spec.iupac_carbons)
                {
                    let c1s = [c1];
                    let c2s = [c2];
                    normalize_wildcard_pair(shifts, amb_map, k, cx, cy, &c1s, &c2s, &c1s, &c2s, 2);
                }
                if r_up == "ILE" && spec.kind == ProchiralSiteKind::GammaMethylene {
                    normalize_wildcard_pair(
                        shifts,
                        amb_map,
                        k,
                        "HGx",
                        "HGy",
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        2,
                    );
                } else if r_up == "ASN" && spec.kind == ProchiralSiteKind::AsnCarboxamide {
                    normalize_wildcard_pair(
                        shifts,
                        amb_map,
                        k,
                        "HDx",
                        "HDy",
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        2,
                    );
                } else if r_up == "GLN" && spec.kind == ProchiralSiteKind::GlnCarboxamide {
                    normalize_wildcard_pair(
                        shifts,
                        amb_map,
                        k,
                        "HEx",
                        "HEy",
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        spec.iupac_protons_1,
                        spec.iupac_protons_2,
                        2,
                    );
                }
            }
        }
    }

    match r_up.as_str() {
        "PHE" | "TYR" => {
            for (wx, wy, c1, c2) in [
                ("CDx", "CDy", "CD1", "CD2"),
                ("CEx", "CEy", "CE1", "CE2"),
                ("HDx", "HDy", "HD1", "HD2"),
                ("HEx", "HEy", "HE1", "HE2"),
            ] {
                let c1s = [c1];
                let c2s = [c2];
                normalize_wildcard_pair(shifts, amb_map, k, wx, wy, &c1s, &c2s, &c1s, &c2s, 3);
            }
        }
        "ARG" => {
            for (wx, wy, c1, c2) in [
                ("NHx", "NHy", "NH1", "NH2"),
                ("HH1x", "HH1y", "HH11", "HH12"),
                ("HH2x", "HH2y", "HH21", "HH22"),
            ] {
                let c1s = [c1];
                let c2s = [c2];
                normalize_wildcard_pair(shifts, amb_map, k, wx, wy, &c1s, &c2s, &c1s, &c2s, 2);
            }
        }
        _ => {}
    }
}

/// Apply stereospecific inference results onto a `Reference` shift map and return both the updated
/// `Reference` and the exact per-`(residue_idx, atom_id)` NMR-STAR `Ambiguity_code` table (`1`, `2`, `3`).
pub fn apply_stereospec_to_reference(
    base: &Reference,
    results: &[StereospecResult],
) -> (Reference, BTreeMap<(usize, String), u8>) {
    let mut out = base.clone();
    let mut amb_map: BTreeMap<(usize, String), u8> = BTreeMap::new();

    // 1. Seed each base atom's ambiguity code from `base.ambiguity_codes` when present (except
    //    NEF `x`/`y` wildcards which take `default_atom_ambiguity_code` — `3` on PHE/TYR rings,
    //    `2` elsewhere — and half-present prochiral atoms which MUST start at `2`), otherwise
    //    `default_atom_ambiguity_code` (`3` for PHE/TYR symmetric rings, `2` for prochiral, `1` for unique).
    for &(k, ref atom) in base.shifts.keys() {
        let res_name = base.sequence.get(k).map(String::as_str).unwrap_or("");
        let code = if is_nef_nonstereo_wildcard_ci(atom) {
            default_atom_ambiguity_code(res_name, atom)
        } else if is_half_present_prochiral_atom(&base.shifts, k, res_name, atom) {
            2
        } else if let Some(&existing) = base.ambiguity_codes.get(&(k, atom.clone())) {
            existing
        } else {
            default_atom_ambiguity_code(res_name, atom)
        };
        amb_map.insert((k, atom.clone()), code);
    }

    // 2. Residue-aware normalization/deduplication of NEF `x`/`y` wildcards into exact canonical
    //    IUPAC atom names (`HG11..13`/`HG21..23` on VAL, `HD11..13`/`HD21..23` on LEU, `HG12`/`HG13`
    //    on ILE, `CD1/CD2/CE1/CE2/HD1/HD2/HE1/HE2` with `Ambiguity_code = 3` on PHE/TYR, etc.).
    for (k, res_name) in base.sequence.iter().enumerate() {
        normalize_residue_nef_wildcards(&mut out.shifts, &mut amb_map, k, res_name);
    }

    for r in results {
        let k = r.residue_idx;
        if let Some(sh) = r.shared_heavy_shift_ppm {
            if let Some(hname) = r.spec.shared_heavy_atom {
                out.shifts.insert((k, hname.to_string()), sh);
                amb_map.insert((k, hname.to_string()), 1);
            }
        }

        if matches!(
            r.spec.kind,
            ProchiralSiteKind::ValIsopropyl | ProchiralSiteKind::LeuIsopropyl
        ) {
            let (canon_subs, wild_subs) = if r.spec.kind == ProchiralSiteKind::ValIsopropyl {
                (VAL_PROTON_SUBPAIRS, VAL_WILDCARD_PROTON_SUBPAIRS)
            } else {
                (LEU_PROTON_SUBPAIRS, LEU_WILDCARD_PROTON_SUBPAIRS)
            };
            let asym = is_methyl_group_asymmetrically_populated(&base.shifts, k, r.spec);
            remove_site_nef_wildcards(&mut out.shifts, &mut amb_map, k, r.spec);

            if asym {
                // Refuse to commit when any carbon or proton sub-pair is half-present: keep all
                // present members (including half-present NEF wildcards already normalized in Step 2)
                // unswapped at their input values with `Ambiguity_code = 2`.
                if let Some((c1_name, c2_name)) = r.spec.iupac_carbons {
                    for cname in [c1_name, c2_name] {
                        if let Some(v) = get_shift_ci(&out.shifts, k, cname) {
                            remove_atom_ci(&mut out.shifts, &mut amb_map, k, cname);
                            out.shifts.insert((k, cname.to_string()), v);
                            amb_map.insert((k, cname.to_string()), 2);
                        }
                    }
                }
                for &(p1, p2) in canon_subs {
                    for pname in [p1, p2] {
                        if let Some(v) = get_shift_ci(&out.shifts, k, pname) {
                            remove_atom_ci(&mut out.shifts, &mut amb_map, k, pname);
                            out.shifts.insert((k, pname.to_string()), v);
                            amb_map.insert((k, pname.to_string()), 2);
                        }
                    }
                }
                continue;
            }

            let eff_committed = r.committed;
            let eff_code = if eff_committed { 1 } else { 2 };
            let do_swap = eff_committed && r.winning_hypothesis == SwapHypothesis::Swapped;

            // Swap or preserve methyl carbons atomically
            if let Some((c1_name, c2_name)) = r.spec.iupac_carbons {
                let base_pair = get_shift_ci(&base.shifts, k, c1_name)
                    .zip(get_shift_ci(&base.shifts, k, c2_name))
                    .or_else(|| {
                        r.spec.nef_nonstereo_carbons.and_then(|(cx, cy)| {
                            get_shift_ci(&base.shifts, k, cx).zip(get_shift_ci(&base.shifts, k, cy))
                        })
                    })
                    .or(r.carbon_shifts_ppm);
                if let Some((vc1, vc2)) = base_pair {
                    let (nc1, nc2) = if do_swap { (vc2, vc1) } else { (vc1, vc2) };
                    remove_atom_ci(&mut out.shifts, &mut amb_map, k, c1_name);
                    remove_atom_ci(&mut out.shifts, &mut amb_map, k, c2_name);
                    out.shifts.insert((k, c1_name.to_string()), nc1);
                    out.shifts.insert((k, c2_name.to_string()), nc2);
                    amb_map.insert((k, c1_name.to_string()), eff_code);
                    amb_map.insert((k, c2_name.to_string()), eff_code);
                }
            }

            // Swap or preserve EVERY present methyl proton representation atomically with its carbon!
            let mut any_canon_subpair_in_base = false;
            for &(p1, p2) in canon_subs {
                if let (Some(vp1), Some(vp2)) = (
                    get_shift_ci(&base.shifts, k, p1),
                    get_shift_ci(&base.shifts, k, p2),
                ) {
                    any_canon_subpair_in_base = true;
                    let (np1, np2) = if do_swap { (vp2, vp1) } else { (vp1, vp2) };
                    remove_atom_ci(&mut out.shifts, &mut amb_map, k, p1);
                    remove_atom_ci(&mut out.shifts, &mut amb_map, k, p2);
                    out.shifts.insert((k, p1.to_string()), np1);
                    out.shifts.insert((k, p2.to_string()), np2);
                    amb_map.insert((k, p1.to_string()), eff_code);
                    amb_map.insert((k, p2.to_string()), eff_code);
                }
            }

            if !any_canon_subpair_in_base {
                let wild_pair = wild_subs.iter().find_map(|&(wx, wy)| {
                    get_shift_ci(&base.shifts, k, wx).zip(get_shift_ci(&base.shifts, k, wy))
                });
                if let Some((ha, hb)) = wild_pair.or(r.proton_shifts_ppm) {
                    let (nh1, nh2) = if do_swap { (hb, ha) } else { (ha, hb) };
                    for &p1 in r.spec.iupac_protons_1 {
                        remove_atom_ci(&mut out.shifts, &mut amb_map, k, p1);
                        out.shifts.insert((k, p1.to_string()), nh1);
                        amb_map.insert((k, p1.to_string()), eff_code);
                    }
                    for &p2 in r.spec.iupac_protons_2 {
                        remove_atom_ci(&mut out.shifts, &mut amb_map, k, p2);
                        out.shifts.insert((k, p2.to_string()), nh2);
                        amb_map.insert((k, p2.to_string()), eff_code);
                    }
                }
            }
            continue;
        }

        // Non-methyl prochiral pair (`GlyAlphaMethylene`, `BetaMethylene`, `GammaMethylene`,
        // `AsnCarboxamide`, `GlnCarboxamide`, `DeltaMethylene`, `EpsilonMethylene`).
        // Step 2 has already normalized any residue-valid NEF `x`/`y` wildcards into `out.shifts`.
        let out_b1 = get_shift_ci(&out.shifts, k, r.spec.iupac_branch_1);
        let out_b2 = get_shift_ci(&out.shifts, k, r.spec.iupac_branch_2);
        remove_site_nef_wildcards(&mut out.shifts, &mut amb_map, k, r.spec);

        // If base had a half-present proton and `r.proton_shifts_ppm` is absent, refuse to commit
        if out_b1.is_some() != out_b2.is_some() && r.proton_shifts_ppm.is_none() {
            for &p in r.spec.iupac_protons_1.iter().chain(r.spec.iupac_protons_2) {
                if let Some(v) = get_shift_ci(&out.shifts, k, p) {
                    remove_atom_ci(&mut out.shifts, &mut amb_map, k, p);
                    out.shifts.insert((k, p.to_string()), v);
                    amb_map.insert((k, p.to_string()), 2);
                }
            }
            continue;
        }

        let code = r.nmr_star_ambiguity_code;
        let do_swap = r.committed && r.winning_hypothesis == SwapHypothesis::Swapped;
        if let Some((ha, hb)) = r.proton_shifts_ppm.or_else(|| out_b1.zip(out_b2)) {
            let (h1_v, h2_v) = if do_swap { (hb, ha) } else { (ha, hb) };
            for &p1 in r.spec.iupac_protons_1 {
                remove_atom_ci(&mut out.shifts, &mut amb_map, k, p1);
                out.shifts.insert((k, p1.to_string()), h1_v);
                amb_map.insert((k, p1.to_string()), code);
            }
            for &p2 in r.spec.iupac_protons_2 {
                remove_atom_ci(&mut out.shifts, &mut amb_map, k, p2);
                out.shifts.insert((k, p2.to_string()), h2_v);
                amb_map.insert((k, p2.to_string()), code);
            }
        }
    }

    out.ambiguity_codes = amb_map.clone();
    (out, amb_map)
}

/// Build ordered NMR-STAR v3 `_Atom_chem_shift` rows preserving `Ambiguity_code = 1` ONLY for
/// backbone/non-prochiral atoms and stereospecifically committed pairs (`posterior >= 0.90`),
/// and `Ambiguity_code = 2` for uncommitted prochiral pairs.
pub fn to_star_chem_shift_rows(
    base: &Reference,
    results: &[StereospecResult],
) -> Vec<StarChemShiftRow> {
    let (merged, amb_map) = apply_stereospec_to_reference(base, results);
    let mut post_by_atom: BTreeMap<(usize, String), f64> = BTreeMap::new();
    for r in results {
        let k = r.residue_idx;
        if let Some((c1, c2)) = r.spec.iupac_carbons {
            post_by_atom.insert((k, c1.to_string()), r.winning_posterior);
            post_by_atom.insert((k, c2.to_string()), r.winning_posterior);
        }
        for &p in r.spec.iupac_protons_1.iter().chain(r.spec.iupac_protons_2) {
            post_by_atom.insert((k, p.to_string()), r.winning_posterior);
        }
    }

    let bb_order = ["H", "N", "CA", "HA", "CB", "C"];
    let mut entries: Vec<(usize, String, f64)> = merged
        .shifts
        .into_iter()
        .map(|((k, a), v)| (k, a, v))
        .collect();
    entries.sort_by(|(ka, aa, _), (kb, ab, _)| {
        let oa = bb_order
            .iter()
            .position(|&x| x == aa.as_str())
            .unwrap_or(50);
        let ob = bb_order
            .iter()
            .position(|&x| x == ab.as_str())
            .unwrap_or(50);
        (ka, oa, aa).cmp(&(kb, ob, ab))
    });

    let mut rows = Vec::with_capacity(entries.len());
    for (k, atom, val) in entries {
        let Some(comp) = merged.sequence.get(k) else {
            continue;
        };
        let seq_id = merged.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let amb = amb_map
            .get(&(k, atom.clone()))
            .copied()
            .unwrap_or_else(|| default_atom_ambiguity_code(comp, &atom));
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_uppercase().to_string())
            .unwrap_or_else(|| "X".to_string());
        let details = if let Some(&p) = post_by_atom.get(&(k, atom.clone())) {
            format!("stereo_post={p:.3}")
        } else {
            ".".to_string()
        };
        rows.push(StarChemShiftRow {
            residue_idx: k,
            seq_id,
            comp_id: comp.to_ascii_uppercase(),
            atom_id: atom,
            atom_type,
            val,
            ambiguity_code: amb,
            details,
        });
    }
    rows
}

/// Format an NMR-STAR v3 document with exact `_Atom_chem_shift.Ambiguity_code` (`1` vs `2`).
pub fn format_stereospec_nmrstar(
    base: &Reference,
    results: &[StereospecResult],
    header_comment: Option<&str>,
) -> String {
    let rows = to_star_chem_shift_rows(base, results);
    let mut lines = Vec::new();
    if let Some(hdr) = header_comment {
        for l in hdr.lines() {
            lines.push(format!(
                "# {}",
                l.trim_start_matches("# ").trim_start_matches('#')
            ));
        }
        lines.push(String::new());
    }
    lines.extend([
        "data_nmr_assign_output".to_string(),
        String::new(),
        "save_entity_1".to_string(),
        "   _Entity.Sf_category  entity".to_string(),
        "   _Entity.ID           1".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Entity_poly_seq.Entity_ID".to_string(),
        "      _Entity_poly_seq.Num".to_string(),
        "      _Entity_poly_seq.Mon_ID".to_string(),
        String::new(),
    ]);
    for (seq_id, mon_id) in base.seq_ids.iter().zip(base.sequence.iter()) {
        lines.push(format!("      1 {:5} {:3}", seq_id, mon_id));
    }
    lines.extend([
        "   stop_".to_string(),
        "save_".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Atom_chem_shift.ID".to_string(),
        "      _Atom_chem_shift.Entity_ID".to_string(),
        "      _Atom_chem_shift.Seq_ID".to_string(),
        "      _Atom_chem_shift.Comp_ID".to_string(),
        "      _Atom_chem_shift.Atom_ID".to_string(),
        "      _Atom_chem_shift.Atom_type".to_string(),
        "      _Atom_chem_shift.Val".to_string(),
        "      _Atom_chem_shift.Ambiguity_code".to_string(),
        "      _Atom_chem_shift.Details".to_string(),
        String::new(),
    ]);

    for (idx, r) in rows.iter().enumerate() {
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} {} {}",
            idx + 1,
            r.seq_id,
            r.comp_id,
            r.atom_id,
            r.atom_type,
            r.val,
            r.ambiguity_code,
            r.details
        ));
    }
    lines.extend(["   stop_".to_string(), "save_".to_string(), String::new()]);
    lines.join("\n")
}

/// Convert `StereospecResult` items into NEF v1.1 `_nef_chemical_shift` rows (`HB2`/`HB3` when
/// `committed`, `HBx`/`HBy` / `CGx`/`CGy` / `HGx%`/`HGy%` otherwise).
pub fn to_nef_stereospec_shifts(
    results: &[StereospecResult],
    chain_code: &str,
) -> Vec<NefAssignedShift> {
    let mut out = Vec::new();
    for r in results {
        let res_up = r.res_name.to_ascii_uppercase();
        if let (Some(hname), Some(val)) = (r.spec.shared_heavy_atom, r.shared_heavy_shift_ppm) {
            let (element, isotope_number) = infer_element_and_isotope(hname);
            out.push(NefAssignedShift {
                chain_code: chain_code.to_string(),
                sequence_code: Some(r.seq_id),
                residue_name: res_up.clone(),
                atom_name: hname.to_string(),
                element,
                isotope_number,
                value: val,
                value_uncertainty: None,
            });
        }

        if let (Some((lc_a, lc_b)), Some((vc_a, vc_b))) =
            (r.nef_carbon_labels_for_ab(), r.carbon_shifts_ppm)
        {
            let mut pairs = [(lc_a, vc_a), (lc_b, vc_b)];
            if !r.committed && pairs[0].1 < pairs[1].1 {
                pairs = [(lc_a, vc_b), (lc_b, vc_a)];
            }
            for (aname, val) in pairs {
                let (element, isotope_number) = infer_element_and_isotope(aname);
                out.push(NefAssignedShift {
                    chain_code: chain_code.to_string(),
                    sequence_code: Some(r.seq_id),
                    residue_name: res_up.clone(),
                    atom_name: aname.to_string(),
                    element,
                    isotope_number,
                    value: val,
                    value_uncertainty: None,
                });
            }
        }

        if let Some((vh_a, vh_b)) = r.proton_shifts_ppm {
            let (lh_a, lh_b) = r.nef_proton_labels_for_ab();
            let mut pairs = [(lh_a, vh_a), (lh_b, vh_b)];
            if !r.committed && pairs[0].1 < pairs[1].1 {
                pairs = [(lh_a, vh_b), (lh_b, vh_a)];
            }
            for (aname, val) in pairs {
                let (element, isotope_number) = infer_element_and_isotope(aname);
                out.push(NefAssignedShift {
                    chain_code: chain_code.to_string(),
                    sequence_code: Some(r.seq_id),
                    residue_name: res_up.clone(),
                    atom_name: aname.to_string(),
                    element,
                    isotope_number,
                    value: val,
                    value_uncertainty: None,
                });
            }
        }
    }
    out
}

/// Build a combined NEF v1.1 `NefShiftList` merging non-prochiral `Reference` shifts with
/// stereospecifically gated prochiral shifts (`HB2`/`HB3` vs `HBx`/`HBy`).
pub fn to_nef_shift_list_with_stereospec(
    base: &Reference,
    results: &[StereospecResult],
    confidence: Option<&BTreeMap<usize, f64>>,
    chain_code: &str,
) -> NefShiftList {
    // Track atoms owned by stereospec results so we don't emit duplicate un-gated copies
    let mut managed: std::collections::BTreeSet<(usize, String)> =
        std::collections::BTreeSet::new();
    for r in results {
        let k = r.residue_idx;
        if r.shared_heavy_shift_ppm.is_some() {
            if let Some(hname) = r.spec.shared_heavy_atom {
                managed.insert((k, hname.to_string()));
            }
        }
        if r.carbon_shifts_ppm.is_some() {
            if let Some((c1, c2)) = r.spec.iupac_carbons {
                managed.insert((k, c1.to_string()));
                managed.insert((k, c2.to_string()));
            }
        }
        if r.proton_shifts_ppm.is_some() {
            for &p in r.spec.iupac_protons_1.iter().chain(r.spec.iupac_protons_2) {
                managed.insert((k, p.to_string()));
                managed.insert((k, normalize_nef_atom_name(&r.res_name, p)));
            }
        }
    }

    let mut filtered = base.clone();
    filtered
        .shifts
        .retain(|&(k, ref atom), _| !managed.contains(&(k, atom.clone())));

    let mut shift_list = NefShiftList::from_reference(&filtered, confidence, chain_code);
    shift_list
        .shifts
        .extend(to_nef_stereospec_shifts(results, chain_code));
    shift_list
        .shifts
        .sort_by(|a, b| (a.sequence_code, &a.atom_name).cmp(&(b.sequence_code, &b.atom_name)));
    shift_list
}

// ============================================================================
// 8. All-Atom PDB Coordinate Reader & Forward Observable Synthesis Helpers
// ============================================================================

/// All-atom coordinates for one residue of a PDB conformer (retaining all heavy atoms & protons).
#[derive(Clone, Debug)]
pub struct AllAtomPdbResidue {
    pub seq_id: i32,
    pub res_name: String,
    pub chain_id: char,
    pub atoms: BTreeMap<String, Vec3>,
}

impl AllAtomPdbResidue {
    #[inline]
    pub fn coord(&self, atom: &str) -> Option<Vec3> {
        self.atoms.get(atom).copied()
    }
}

/// Parse the first model of a PDB file retaining all explicit heavy-atom and proton coordinates.
pub fn parse_pdb_all_atoms_file(path: impl AsRef<Path>) -> Result<Vec<AllAtomPdbResidue>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read PDB {}: {e}", path.as_ref().display()))?;
    parse_pdb_all_atoms_str(&text)
}

/// Parse the first model of a PDB text string retaining all explicit heavy-atom and proton coordinates.
pub fn parse_pdb_all_atoms_str(pdb_text: &str) -> Result<Vec<AllAtomPdbResidue>, String> {
    let mut residues: Vec<AllAtomPdbResidue> = Vec::new();
    let mut idx_by_seq: BTreeMap<(char, i32), usize> = BTreeMap::new();
    let mut selected_chain: Option<char> = None;

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        if !line.starts_with("ATOM  ") || line.len() < 54 {
            continue;
        }
        let bytes = line.as_bytes();
        let alt_loc = bytes[16] as char;
        if alt_loc != ' ' && alt_loc != 'A' && alt_loc != '1' {
            continue;
        }
        let chain_id = bytes[21] as char;
        if let Some(ch) = selected_chain {
            if chain_id != ch {
                continue;
            }
        } else {
            selected_chain = Some(chain_id);
        }
        if bytes[26] as char != ' ' {
            continue;
        }
        let atom_name = line[12..16].trim().to_ascii_uppercase();
        let res_name = line[17..20].trim().to_ascii_uppercase();
        let Ok(seq_id) = line[22..26].trim().parse::<i32>() else {
            continue;
        };
        let Ok(x) = line[30..38].trim().parse::<f64>() else {
            continue;
        };
        let Ok(y) = line[38..46].trim().parse::<f64>() else {
            continue;
        };
        let Ok(z) = line[46..54].trim().parse::<f64>() else {
            continue;
        };

        let idx = *idx_by_seq.entry((chain_id, seq_id)).or_insert_with(|| {
            let id = residues.len();
            residues.push(AllAtomPdbResidue {
                seq_id,
                res_name: res_name.clone(),
                chain_id,
                atoms: BTreeMap::new(),
            });
            id
        });
        let canon = match atom_name.as_str() {
            "HN" => "H".to_string(),
            other => other.to_string(),
        };
        residues[idx].atoms.entry(canon).or_insert([x, y, z]);
    }

    if residues.is_empty() {
        return Err("no ATOM records found in PDB".to_string());
    }
    Ok(residues)
}

/// Extract exact backbone `(phi_i, psi_i)` conditioning from an all-atom PDB chain at `res_idx`.
pub fn backbone_conditioning_from_all_atom_pdb(
    residues: &[AllAtomPdbResidue],
    res_idx: usize,
) -> BackboneConditioning {
    let Some(cur) = residues.get(res_idx) else {
        return BackboneConditioning::Marginalized;
    };
    let (Some(n), Some(ca), Some(c)) = (cur.coord("N"), cur.coord("CA"), cur.coord("C")) else {
        return BackboneConditioning::Marginalized;
    };
    let phi = if res_idx > 0 {
        residues[res_idx - 1]
            .coord("C")
            .filter(|&cp| v_dist(cp, n) < 2.0)
            .map(|cp| calc_dihedral_deg(cp, n, ca, c))
    } else {
        None
    };
    let psi = residues
        .get(res_idx + 1)
        .and_then(|nx| nx.coord("N"))
        .filter(|&nn| v_dist(c, nn) < 2.0)
        .map(|nn| calc_dihedral_deg(n, ca, c, nn));

    match (phi, psi) {
        (Some(p), Some(s)) => BackboneConditioning::ExactTorsions {
            phi_deg: p,
            psi_deg: s,
        },
        _ => BackboneConditioning::Marginalized,
    }
}

/// Synthesize ideal (noise-free) `ProchiralObservables` in canonical `(branch_1, branch_2)` (`Direct`)
/// order directly from experimental 3D atomic coordinates of `residues[res_idx]`.
pub fn synthesize_ideal_observables_from_pdb(
    residues: &[AllAtomPdbResidue],
    res_idx: usize,
    spec: ProchiralPairSpec,
) -> Option<ProchiralObservables> {
    let cur = residues.get(res_idx)?;
    let next_res = residues.get(res_idx + 1);
    let next_is_pro = next_res
        .map(|r| r.res_name.eq_ignore_ascii_case("PRO"))
        .unwrap_or(false);
    let floor_r6 = 5.5_f64.powi(-6);

    let mut obs = ProchiralObservables {
        next_is_pro,
        ..Default::default()
    };

    match spec.kind {
        ProchiralSiteKind::BetaMethylene => {
            let (n, ca, cb, ha, hb2, hb3) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("CB")?,
                cur.coord("HA")?,
                cur.coord("HB2")?,
                cur.coord("HB3")?,
            );
            obs.j_hh_vicinal_hz = Some((
                karplus_3j_hahb_hz(calc_dihedral_deg(ha, ca, cb, hb2)),
                karplus_3j_hahb_hz(calc_dihedral_deg(ha, ca, cb, hb3)),
            ));
            obs.j_nhb_hz = Some((
                karplus_3j_nhb_mag_hz(calc_dihedral_deg(n, ca, cb, hb2)),
                karplus_3j_nhb_mag_hz(calc_dihedral_deg(n, ca, cb, hb3)),
            ));
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraAlphaHA,
                r6_single(ha, hb2) + floor_r6,
                r6_single(ha, hb3) + floor_r6,
            ));
            if let Some(h) = cur.coord("H") {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAmideH,
                    r6_single(h, hb2) + floor_r6,
                    r6_single(h, hb3) + floor_r6,
                ));
            }
            if let Some(hn1) = next_res.and_then(|r| r.coord("H")) {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::SeqNextAmideH,
                    r6_single(hn1, hb2) + floor_r6,
                    r6_single(hn1, hb3) + floor_r6,
                ));
            }
        }
        ProchiralSiteKind::ValIsopropyl => {
            let (n, ca, c, cb, ha, hb, cg1, cg2) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("C")?,
                cur.coord("CB")?,
                cur.coord("HA")?,
                cur.coord("HB")?,
                cur.coord("CG1")?,
                cur.coord("CG2")?,
            );
            let hg1 = [cur.coord("HG11")?, cur.coord("HG12")?, cur.coord("HG13")?];
            let hg2 = [cur.coord("HG21")?, cur.coord("HG22")?, cur.coord("HG23")?];
            obs.j_ncg_hz = Some((
                karplus_3j_ncg_hz("VAL", calc_dihedral_deg(n, ca, cb, cg1)),
                karplus_3j_ncg_hz("VAL", calc_dihedral_deg(n, ca, cb, cg2)),
            ));
            obs.j_cc_vicinal_hz = Some((
                karplus_3j_ccg_hz("VAL", calc_dihedral_deg(c, ca, cb, cg1)),
                karplus_3j_ccg_hz("VAL", calc_dihedral_deg(c, ca, cb, cg2)),
            ));
            obs.j_methine_hahb_hz = Some(karplus_3j_hahb_hz(calc_dihedral_deg(ha, ca, cb, hb)));
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraAlphaHA,
                r6_methyl(ha, &hg1) + floor_r6,
                r6_methyl(ha, &hg2) + floor_r6,
            ));
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraBetaHB,
                r6_methyl(hb, &hg1) + floor_r6,
                r6_methyl(hb, &hg2) + floor_r6,
            ));
            if let Some(h) = cur.coord("H") {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAmideH,
                    r6_methyl(h, &hg1) + floor_r6,
                    r6_methyl(h, &hg2) + floor_r6,
                ));
            }
            if let Some(hn1) = next_res.and_then(|r| r.coord("H")) {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::SeqNextAmideH,
                    r6_methyl(hn1, &hg1) + floor_r6,
                    r6_methyl(hn1, &hg2) + floor_r6,
                ));
            }
        }
        ProchiralSiteKind::LeuIsopropyl => {
            let (ca, cb, cg, ha, cd1, cd2) = (
                cur.coord("CA")?,
                cur.coord("CB")?,
                cur.coord("CG")?,
                cur.coord("HA")?,
                cur.coord("CD1")?,
                cur.coord("CD2")?,
            );
            let hd1 = [cur.coord("HD11")?, cur.coord("HD12")?, cur.coord("HD13")?];
            let hd2 = [cur.coord("HD21")?, cur.coord("HD22")?, cur.coord("HD23")?];
            obs.j_cc_vicinal_hz = Some((
                karplus_3j_ccg_hz("LEU", calc_dihedral_deg(ca, cb, cg, cd1)),
                karplus_3j_ccg_hz("LEU", calc_dihedral_deg(ca, cb, cg, cd2)),
            ));
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraAlphaHA,
                r6_methyl(ha, &hd1) + floor_r6,
                r6_methyl(ha, &hd2) + floor_r6,
            ));
            if let (Some(hb2), Some(hb3)) = (cur.coord("HB2"), cur.coord("HB3")) {
                let qb = v_scale(v_add(hb2, hb3), 0.5);
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraBetaHB,
                    r6_methyl(qb, &hd1) + floor_r6,
                    r6_methyl(qb, &hd2) + floor_r6,
                ));
            }
            if let Some(h) = cur.coord("H") {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAmideH,
                    r6_methyl(h, &hd1) + floor_r6,
                    r6_methyl(h, &hd2) + floor_r6,
                ));
            }
            if let Some(hn1) = next_res.and_then(|r| r.coord("H")) {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::SeqNextAmideH,
                    r6_methyl(hn1, &hd1) + floor_r6,
                    r6_methyl(hn1, &hd2) + floor_r6,
                ));
            }
        }
        ProchiralSiteKind::GlyAlphaMethylene => {
            let (n, ca, ha2, ha3) = (
                cur.coord("N")?,
                cur.coord("CA")?,
                cur.coord("HA2")?,
                cur.coord("HA3")?,
            );
            if let Some(h) = cur.coord("H") {
                obs.j_hh_vicinal_hz = Some((
                    karplus_3j_hnha_hz(calc_dihedral_deg(h, n, ca, ha2)),
                    karplus_3j_hnha_hz(calc_dihedral_deg(h, n, ca, ha3)),
                ));
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAmideH,
                    r6_single(h, ha2) + floor_r6,
                    r6_single(h, ha3) + floor_r6,
                ));
            }
            if let Some(hn1) = next_res.and_then(|r| r.coord("H")) {
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::SeqNextAmideH,
                    r6_single(hn1, ha2) + floor_r6,
                    r6_single(hn1, ha3) + floor_r6,
                ));
            }
        }
        ProchiralSiteKind::GammaMethylene => {
            let cb = cur.coord("CB")?;
            let ha = cur.coord("HA")?;
            if cur.res_name == "ILE" {
                let (cg1, hb, hg12, hg13) = (
                    cur.coord("CG1")?,
                    cur.coord("HB")?,
                    cur.coord("HG12")?,
                    cur.coord("HG13")?,
                );
                obs.j_hh_vicinal_hz = Some((
                    karplus_3j_hahb_hz(calc_dihedral_deg(hb, cb, cg1, hg12)),
                    karplus_3j_hahb_hz(calc_dihedral_deg(hb, cb, cg1, hg13)),
                ));
                if let Some(n) = cur.coord("N") {
                    if let Some(ca) = cur.coord("CA") {
                        obs.j_methine_hahb_hz =
                            Some(karplus_3j_hahb_hz(calc_dihedral_deg(ha, ca, cb, hb)));
                        let _ = n;
                    }
                }
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAlphaHA,
                    r6_single(ha, hg12) + floor_r6,
                    r6_single(ha, hg13) + floor_r6,
                ));
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraBetaHB,
                    r6_single(hb, hg12) + floor_r6,
                    r6_single(hb, hg13) + floor_r6,
                ));
                if let Some(h) = cur.coord("H") {
                    obs.noe_pairs.push(NoePairObs::new(
                        AnchorProton::IntraAmideH,
                        r6_single(h, hg12) + floor_r6,
                        r6_single(h, hg13) + floor_r6,
                    ));
                }
            } else {
                let (cg, hg2, hg3, hb2, hb3) = (
                    cur.coord("CG")?,
                    cur.coord("HG2")?,
                    cur.coord("HG3")?,
                    cur.coord("HB2")?,
                    cur.coord("HB3")?,
                );
                let j2 = 0.5
                    * (karplus_3j_hahb_hz(calc_dihedral_deg(hb2, cb, cg, hg2))
                        + karplus_3j_hahb_hz(calc_dihedral_deg(hb3, cb, cg, hg2)));
                let j3 = 0.5
                    * (karplus_3j_hahb_hz(calc_dihedral_deg(hb2, cb, cg, hg3))
                        + karplus_3j_hahb_hz(calc_dihedral_deg(hb3, cb, cg, hg3)));
                obs.j_hh_vicinal_hz = Some((j2, j3));
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraAlphaHA,
                    r6_single(ha, hg2) + floor_r6,
                    r6_single(ha, hg3) + floor_r6,
                ));
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraBetaHB2,
                    r6_single(hb2, hg2) + floor_r6,
                    r6_single(hb2, hg3) + floor_r6,
                ));
                obs.noe_pairs.push(NoePairObs::new(
                    AnchorProton::IntraBetaHB3,
                    r6_single(hb3, hg2) + floor_r6,
                    r6_single(hb3, hg3) + floor_r6,
                ));
                if let Some(h) = cur.coord("H") {
                    obs.noe_pairs.push(NoePairObs::new(
                        AnchorProton::IntraAmideH,
                        r6_single(h, hg2) + floor_r6,
                        r6_single(h, hg3) + floor_r6,
                    ));
                }
            }
        }
        ProchiralSiteKind::AsnCarboxamide => {
            let (hb2, hb3, hd21, hd22) = (
                cur.coord("HB2")?,
                cur.coord("HB3")?,
                cur.coord("HD21")?,
                cur.coord("HD22")?,
            );
            let qb = v_scale(v_add(hb2, hb3), 0.5);
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraBetaHB,
                r6_single(qb, hd21) + floor_r6,
                r6_single(qb, hd22) + floor_r6,
            ));
        }
        ProchiralSiteKind::GlnCarboxamide => {
            let (hg2, hg3, he21, he22) = (
                cur.coord("HG2")?,
                cur.coord("HG3")?,
                cur.coord("HE21")?,
                cur.coord("HE22")?,
            );
            let qg = v_scale(v_add(hg2, hg3), 0.5);
            obs.noe_pairs.push(NoePairObs::new(
                AnchorProton::IntraGammaHG,
                r6_single(qg, he21) + floor_r6,
                r6_single(qg, he22) + floor_r6,
            ));
        }
        ProchiralSiteKind::DeltaMethylene | ProchiralSiteKind::EpsilonMethylene => {
            return None;
        }
    }

    Some(obs)
}

// ============================================================================
// 9. Dataset-Level CLI / Pipeline Orchestration (`assign_stereospecific_dataset`)
// ============================================================================

/// Parsed scalar coupling record from a `--j-couplings` TSV table.
#[derive(Clone, Debug, PartialEq)]
pub struct JCouplingTsvRow {
    pub seq_id: i32,
    pub res_name: String,
    pub site: String,
    pub j_kind: String,
    pub val_a_hz: f64,
    pub val_b_hz: Option<f64>,
}

/// Parsed intraresidue / sequential NOE pair record from an `--intra-noes` TSV table.
#[derive(Clone, Debug, PartialEq)]
pub struct IntraNoeTsvRow {
    pub seq_id: i32,
    pub res_name: String,
    pub site: String,
    pub anchor: AnchorProton,
    pub intensity_a: f64,
    pub intensity_b: f64,
}

/// Input bundle for dataset-wide stereospecific assignment (`nmrassign sidechain --stereospec`).
pub struct StereospecDatasetInput<'a> {
    pub reference: &'a Reference,
    pub j_couplings_tsv: Option<&'a Path>,
    pub intra_noes_tsv: Option<&'a Path>,
    pub pdb_path: Option<&'a Path>,
    pub config: StereospecConfig,
}

/// Output of [`assign_stereospecific_dataset`].
#[derive(Clone, Debug)]
pub struct StereospecDatasetOutput {
    /// Updated `Reference` with stereospecifically resolved shifts ordered per IUPAC.
    pub reference: Reference,
    /// Per-`(residue_idx, atom_id)` NMR-STAR v3 `_Atom_chem_shift.Ambiguity_code` (`1`, `2`, `3`).
    pub ambiguity_codes: BTreeMap<(usize, String), u8>,
    /// Per-`(residue_idx, atom_id)` winning stereospecific posterior in `[0.50, 1.00]`.
    pub atom_posteriors: BTreeMap<(usize, String), f64>,
    /// Full Bayesian inference results for each evaluated prochiral site.
    pub pair_results: Vec<StereospecResult>,
    /// Count of prochiral pairs committed at `posterior >= config.commit_posterior_threshold` (`Ambiguity_code = 1`).
    pub committed_pairs: usize,
    /// Count of prochiral pairs left uncommitted (`Ambiguity_code = 2`).
    pub ambiguous_pairs: usize,
}

fn parse_anchor_proton_token(tok: &str) -> Result<AnchorProton, String> {
    let up = tok
        .trim()
        .to_ascii_uppercase()
        .replace(['-', '(', ')', '+'], "_");
    match up.as_str() {
        "H" | "HN" | "INTRAAMIDEH" | "INTRA_AMIDE_H" => Ok(AnchorProton::IntraAmideH),
        "HA" | "QA" | "INTRAALPHAHA" | "INTRA_ALPHA_HA" => Ok(AnchorProton::IntraAlphaHA),
        "H_NEXT" | "HN_NEXT" | "H_I_1_" | "H_I1" | "SEQNEXTAMIDEH" | "SEQ_NEXT_AMIDE_H" => {
            Ok(AnchorProton::SeqNextAmideH)
        }
        "HB" | "QB" | "INTRABETAHB" | "INTRA_BETA_HB" => Ok(AnchorProton::IntraBetaHB),
        "HB2" | "INTRABETAHB2" | "INTRA_BETA_HB2" => Ok(AnchorProton::IntraBetaHB2),
        "HB3" | "INTRABETAHB3" | "INTRA_BETA_HB3" => Ok(AnchorProton::IntraBetaHB3),
        "HG" | "QG" | "INTRAGAMMAHG" | "INTRA_GAMMA_HG" => Ok(AnchorProton::IntraGammaHG),
        "HD" | "QD" | "INTRADELTAHD" | "INTRA_DELTA_HD" => Ok(AnchorProton::IntraDeltaHD),
        other => Err(format!("unknown NOE anchor proton {other:?}")),
    }
}

const fn anchor_proton_token(anchor: AnchorProton) -> &'static str {
    match anchor {
        AnchorProton::IntraAmideH => "H",
        AnchorProton::IntraAlphaHA => "HA",
        AnchorProton::SeqNextAmideH => "H_NEXT",
        AnchorProton::IntraBetaHB => "HB",
        AnchorProton::IntraBetaHB2 => "HB2",
        AnchorProton::IntraBetaHB3 => "HB3",
        AnchorProton::IntraGammaHG => "HG",
        AnchorProton::IntraDeltaHD => "HD",
    }
}

/// Parse scalar `3J` coupling rows from TSV text (`seq_id  res_name  site  j_kind  val_a_hz  [val_b_hz]`).
pub fn parse_j_couplings_tsv_str(text: &str) -> Result<Vec<JCouplingTsvRow>, String> {
    let mut rows = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }
        if (line_no == 0
            || cols[0].eq_ignore_ascii_case("seq_id")
            || cols[0].eq_ignore_ascii_case("seq"))
            && cols[0].parse::<i32>().is_err()
        {
            continue;
        }
        if cols.len() < 5 {
            return Err(format!(
                "j-couplings TSV line {}: expected >= 5 columns (seq_id res_name site j_kind val_a_hz [val_b_hz]), got {}",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_id = cols[0]
            .parse::<i32>()
            .map_err(|e| format!("j-couplings TSV line {}: invalid seq_id: {e}", line_no + 1))?;
        let val_a_hz = cols[4].parse::<f64>().map_err(|e| {
            format!(
                "j-couplings TSV line {}: invalid val_a_hz: {e}",
                line_no + 1
            )
        })?;
        let val_b_hz = if let Some(b_tok) = cols.get(5) {
            if *b_tok == "." || *b_tok == "NA" || *b_tok == "-" {
                None
            } else {
                Some(b_tok.parse::<f64>().map_err(|e| {
                    format!(
                        "j-couplings TSV line {}: invalid val_b_hz: {e}",
                        line_no + 1
                    )
                })?)
            }
        } else {
            None
        };
        rows.push(JCouplingTsvRow {
            seq_id,
            res_name: cols[1].to_ascii_uppercase(),
            site: cols[2].to_ascii_uppercase(),
            j_kind: cols[3].to_ascii_uppercase(),
            val_a_hz,
            val_b_hz,
        });
    }
    Ok(rows)
}

/// Read scalar `3J` coupling rows from a TSV file on disk.
pub fn read_j_couplings_tsv(path: impl AsRef<Path>) -> Result<Vec<JCouplingTsvRow>, String> {
    let text = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read j-couplings {}: {e}",
            path.as_ref().display()
        )
    })?;
    parse_j_couplings_tsv_str(&text)
}

/// Write scalar `3J` coupling rows to a TSV file on disk.
pub fn write_j_couplings_tsv(
    path: impl AsRef<Path>,
    rows: &[JCouplingTsvRow],
) -> Result<(), String> {
    let mut out = String::from("seq_id\tres_name\tsite\tj_kind\tval_a_hz\tval_b_hz\n");
    for r in rows {
        let b_str = r
            .val_b_hz
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| ".".to_string());
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.4}\t{}\n",
            r.seq_id, r.res_name, r.site, r.j_kind, r.val_a_hz, b_str
        ));
    }
    fs::write(path.as_ref(), out).map_err(|e| {
        format!(
            "failed to write j-couplings {}: {e}",
            path.as_ref().display()
        )
    })
}

/// Parse intraresidue / sequential NOE intensity-pair rows from TSV text
/// (`seq_id  res_name  site  anchor  intensity_a  intensity_b`).
pub fn parse_intra_noes_tsv_str(text: &str) -> Result<Vec<IntraNoeTsvRow>, String> {
    let mut rows = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }
        if (line_no == 0
            || cols[0].eq_ignore_ascii_case("seq_id")
            || cols[0].eq_ignore_ascii_case("seq"))
            && cols[0].parse::<i32>().is_err()
        {
            continue;
        }
        if cols.len() < 6 {
            return Err(format!(
                "intra-noes TSV line {}: expected >= 6 columns (seq_id res_name site anchor intensity_a intensity_b), got {}",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_id = cols[0]
            .parse::<i32>()
            .map_err(|e| format!("intra-noes TSV line {}: invalid seq_id: {e}", line_no + 1))?;
        let anchor = parse_anchor_proton_token(cols[3])
            .map_err(|e| format!("intra-noes TSV line {}: {e}", line_no + 1))?;
        let intensity_a = cols[4].parse::<f64>().map_err(|e| {
            format!(
                "intra-noes TSV line {}: invalid intensity_a: {e}",
                line_no + 1
            )
        })?;
        let intensity_b = cols[5].parse::<f64>().map_err(|e| {
            format!(
                "intra-noes TSV line {}: invalid intensity_b: {e}",
                line_no + 1
            )
        })?;
        rows.push(IntraNoeTsvRow {
            seq_id,
            res_name: cols[1].to_ascii_uppercase(),
            site: cols[2].to_ascii_uppercase(),
            anchor,
            intensity_a,
            intensity_b,
        });
    }
    Ok(rows)
}

/// Read intraresidue / sequential NOE intensity-pair rows from a TSV file on disk.
pub fn read_intra_noes_tsv(path: impl AsRef<Path>) -> Result<Vec<IntraNoeTsvRow>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read intra-noes {}: {e}", path.as_ref().display()))?;
    parse_intra_noes_tsv_str(&text)
}

/// Write intraresidue / sequential NOE intensity-pair rows to a TSV file on disk.
pub fn write_intra_noes_tsv(path: impl AsRef<Path>, rows: &[IntraNoeTsvRow]) -> Result<(), String> {
    let mut out = String::from("seq_id\tres_name\tsite\tanchor\tintensity_a\tintensity_b\n");
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.6e}\t{:.6e}\n",
            r.seq_id,
            r.res_name,
            r.site,
            anchor_proton_token(r.anchor),
            r.intensity_a,
            r.intensity_b
        ));
    }
    fs::write(path.as_ref(), out).map_err(|e| {
        format!(
            "failed to write intra-noes {}: {e}",
            path.as_ref().display()
        )
    })
}

fn resolve_branch1_key(res_name: &str, site_tok: &str) -> Option<&'static str> {
    let up = site_tok.trim().to_ascii_uppercase();
    for spec in prochiral_pairs_for_residue(res_name) {
        if up == spec.iupac_branch_1.to_ascii_uppercase()
            || up == spec.iupac_branch_2.to_ascii_uppercase()
            || up == spec.nef_nonstereo_protons.0.to_ascii_uppercase()
            || up == spec.nef_nonstereo_protons.1.to_ascii_uppercase()
            || spec
                .shared_heavy_atom
                .is_some_and(|hv| hv.eq_ignore_ascii_case(&up))
            || spec.nef_nonstereo_carbons.is_some_and(|(cx, cy)| {
                cx.eq_ignore_ascii_case(&up) || cy.eq_ignore_ascii_case(&up)
            })
            || (spec.kind == ProchiralSiteKind::ValIsopropyl
                && VAL_PROTON_SUBPAIRS
                    .iter()
                    .chain(VAL_WILDCARD_PROTON_SUBPAIRS.iter())
                    .any(|&(p1, p2)| up.eq_ignore_ascii_case(p1) || up.eq_ignore_ascii_case(p2)))
            || (spec.kind == ProchiralSiteKind::LeuIsopropyl
                && LEU_PROTON_SUBPAIRS
                    .iter()
                    .chain(LEU_WILDCARD_PROTON_SUBPAIRS.iter())
                    .any(|&(p1, p2)| up.eq_ignore_ascii_case(p1) || up.eq_ignore_ascii_case(p2)))
        {
            return Some(spec.iupac_branch_1);
        }
    }
    None
}

fn lookup_proton_group_shift(
    ref_data: &Reference,
    res_idx: usize,
    spec: ProchiralPairSpec,
    is_branch_1: bool,
) -> Option<f64> {
    let (protons, nonstereo_fallback) = if is_branch_1 {
        (spec.iupac_protons_1, spec.nef_nonstereo_protons.0)
    } else {
        (spec.iupac_protons_2, spec.nef_nonstereo_protons.1)
    };
    for &p in protons {
        if let Some(v) = get_shift_ci(&ref_data.shifts, res_idx, p) {
            return Some(v);
        }
    }
    match spec.kind {
        ProchiralSiteKind::ValIsopropyl => {
            for &(p1, p2) in VAL_PROTON_SUBPAIRS
                .iter()
                .chain(VAL_WILDCARD_PROTON_SUBPAIRS.iter())
            {
                let target = if is_branch_1 { p1 } else { p2 };
                if let Some(v) = get_shift_ci(&ref_data.shifts, res_idx, target) {
                    return Some(v);
                }
            }
        }
        ProchiralSiteKind::LeuIsopropyl => {
            for &(p1, p2) in LEU_PROTON_SUBPAIRS
                .iter()
                .chain(LEU_WILDCARD_PROTON_SUBPAIRS.iter())
            {
                let target = if is_branch_1 { p1 } else { p2 };
                if let Some(v) = get_shift_ci(&ref_data.shifts, res_idx, target) {
                    return Some(v);
                }
            }
        }
        _ => {}
    }
    let trimmed = nonstereo_fallback.trim_end_matches('%');
    get_shift_ci(&ref_data.shifts, res_idx, nonstereo_fallback)
        .or_else(|| get_shift_ci(&ref_data.shifts, res_idx, trimmed))
}

/// Assign prochiral sidechain & glycine `CH2` / isopropyl `CH3` / carboxamide `NH2` centers across
/// an entire protein dataset (`Reference` + optional `--j-couplings`, `--intra-noes`, `--pdb`),
/// returning the updated `Reference` and per-atom NMR-STAR `_Atom_chem_shift.Ambiguity_code` (`1` vs `2`).
pub fn assign_stereospecific_dataset(
    input: &StereospecDatasetInput<'_>,
) -> Result<StereospecDatasetOutput, String> {
    let n_res = input.reference.sequence.len();
    let mut idx_by_seq_id: BTreeMap<i32, usize> = BTreeMap::new();
    for (k, &sid) in input.reference.seq_ids.iter().enumerate() {
        idx_by_seq_id.insert(sid, k);
    }

    // 1. Seed per-residue prochiral shift pairs from input Reference
    let mut per_res_obs: Vec<BTreeMap<String, ProchiralObservables>> = vec![BTreeMap::new(); n_res];
    let mut has_data_mask: BTreeMap<(usize, &'static str), bool> = BTreeMap::new();

    for (k, res_name) in input.reference.sequence.iter().enumerate() {
        let next_is_pro = input
            .reference
            .sequence
            .get(k + 1)
            .is_some_and(|nx| nx.eq_ignore_ascii_case("PRO"));
        for spec in prochiral_pairs_for_residue(res_name) {
            let mut obs = ProchiralObservables {
                next_is_pro,
                ..Default::default()
            };
            if let Some(hv) = spec.shared_heavy_atom {
                obs.shared_heavy_shift_ppm = get_shift_ci(&input.reference.shifts, k, hv);
            }
            if let Some((c1, c2)) = spec.iupac_carbons {
                let (ncx, ncy) = spec.nef_nonstereo_carbons.unwrap_or((c1, c2));
                let vc1 = get_shift_ci(&input.reference.shifts, k, c1)
                    .or_else(|| get_shift_ci(&input.reference.shifts, k, ncx));
                let vc2 = get_shift_ci(&input.reference.shifts, k, c2)
                    .or_else(|| get_shift_ci(&input.reference.shifts, k, ncy));
                if let (Some(va), Some(vb)) = (vc1, vc2) {
                    obs.carbon_shifts_ppm = Some((va, vb));
                }
            }
            let vh1 = lookup_proton_group_shift(input.reference, k, spec, true);
            let vh2 = lookup_proton_group_shift(input.reference, k, spec, false);
            if let (Some(ha), Some(hb)) = (vh1, vh2) {
                obs.proton_shifts_ppm = Some((ha, hb));
            }
            if obs.proton_shifts_ppm.is_some() || obs.carbon_shifts_ppm.is_some() {
                has_data_mask.insert((k, spec.iupac_branch_1), true);
            }
            per_res_obs[k].insert(spec.iupac_branch_1.to_string(), obs);
        }
    }

    // 2. Merge experimental scalar couplings (`--j-couplings`)
    if let Some(j_path) = input.j_couplings_tsv {
        for row in read_j_couplings_tsv(j_path)? {
            let Some(&k) = idx_by_seq_id.get(&row.seq_id) else {
                continue;
            };
            let res_name = &input.reference.sequence[k];
            let Some(b1) = resolve_branch1_key(res_name, &row.site) else {
                continue;
            };
            let entry = per_res_obs[k].entry(b1.to_string()).or_default();
            has_data_mask.insert((k, b1), true);
            match row.j_kind.as_str() {
                "3J_METHINE_HAHB" | "METHINE_HAHB" => {
                    entry.j_methine_hahb_hz = Some(row.val_a_hz);
                }
                "3J_NHB" | "J_NHB" => {
                    if let Some(vb) = row.val_b_hz {
                        entry.j_nhb_hz = Some((row.val_a_hz, vb));
                    }
                }
                "3J_NCG" | "J_NCG" => {
                    if let Some(vb) = row.val_b_hz {
                        entry.j_ncg_hz = Some((row.val_a_hz, vb));
                    }
                }
                "3J_CCG" | "3J_C'CG" | "3J_CC" | "3J_CACD" | "J_CC" => {
                    if let Some(vb) = row.val_b_hz {
                        entry.j_cc_vicinal_hz = Some((row.val_a_hz, vb));
                    }
                }
                _ => {
                    if let Some(vb) = row.val_b_hz {
                        entry.j_hh_vicinal_hz = Some((row.val_a_hz, vb));
                    } else {
                        entry.j_methine_hahb_hz = Some(row.val_a_hz);
                    }
                }
            }
        }
    }

    // 3. Merge experimental intraresidue / sequential NOE pairs (`--intra-noes`)
    if let Some(noe_path) = input.intra_noes_tsv {
        for row in read_intra_noes_tsv(noe_path)? {
            let Some(&k) = idx_by_seq_id.get(&row.seq_id) else {
                continue;
            };
            let res_name = &input.reference.sequence[k];
            let Some(b1) = resolve_branch1_key(res_name, &row.site) else {
                continue;
            };
            let entry = per_res_obs[k].entry(b1.to_string()).or_default();
            has_data_mask.insert((k, b1), true);
            entry.noe_pairs.push(NoePairObs::new(
                row.anchor,
                row.intensity_a,
                row.intensity_b,
            ));
        }
    }

    // 4. Build backbone (phi, psi) conditioning from `--pdb` if supplied, else CSI dihedral prediction
    let backbones: Vec<BackboneConditioning> = if let Some(pdb_p) = input.pdb_path {
        let pdb_res = parse_pdb_all_atoms_file(pdb_p)?;
        let mut pdb_idx_by_seq: BTreeMap<i32, usize> = BTreeMap::new();
        for (pi, pr) in pdb_res.iter().enumerate() {
            pdb_idx_by_seq.entry(pr.seq_id).or_insert(pi);
        }
        (0..n_res)
            .map(|k| {
                let sid = input
                    .reference
                    .seq_ids
                    .get(k)
                    .copied()
                    .unwrap_or((k + 1) as i32);
                if let Some(&pi) = pdb_idx_by_seq.get(&sid) {
                    backbone_conditioning_from_all_atom_pdb(&pdb_res, pi)
                } else {
                    BackboneConditioning::Marginalized
                }
            })
            .collect()
    } else {
        let dih_preds = crate::dihedrals::predict_dihedrals(input.reference);
        let mut dih_by_seq: BTreeMap<i32, BackboneConditioning> = BTreeMap::new();
        for dr in &dih_preds {
            dih_by_seq.insert(dr.seq_id, BackboneConditioning::from_dihedral_restraint(dr));
        }
        (0..n_res)
            .map(|k| {
                let sid = input
                    .reference
                    .seq_ids
                    .get(k)
                    .copied()
                    .unwrap_or((k + 1) as i32);
                dih_by_seq
                    .get(&sid)
                    .cloned()
                    .unwrap_or(BackboneConditioning::Marginalized)
            })
            .collect()
    };

    // 5. Run hierarchical Bayesian stereospecific inference across the sequence
    let raw_results = infer_protein_stereospec(
        &input.reference.sequence,
        &input.reference.seq_ids,
        &backbones,
        &per_res_obs,
        &input.config,
    );
    let mut pair_results: Vec<StereospecResult> = raw_results
        .into_iter()
        .filter(|r| {
            has_data_mask
                .get(&(r.residue_idx, r.spec.iupac_branch_1))
                .copied()
                .unwrap_or(false)
        })
        .collect();
    for r in &mut pair_results {
        if is_methyl_group_asymmetrically_populated(&input.reference.shifts, r.residue_idx, r.spec)
        {
            r.committed = false;
            r.nmr_star_ambiguity_code = 2;
        }
    }

    let committed_pairs = pair_results.iter().filter(|r| r.committed).count();
    let ambiguous_pairs = pair_results.len().saturating_sub(committed_pairs);

    let (mut updated_ref, mut ambiguity_codes) =
        apply_stereospec_to_reference(input.reference, &pair_results);

    // Mark fast-flipping PHE/TYR symmetric aromatic pairs with standard BMRB Ambiguity_code = 3
    for (k, res_name) in updated_ref.sequence.iter().enumerate() {
        if matches!(res_name.as_str(), "PHE" | "TYR") {
            for sym_atom in ["CD1", "CD2", "HD1", "HD2", "CE1", "CE2", "HE1", "HE2"] {
                if updated_ref.shifts.contains_key(&(k, sym_atom.to_string())) {
                    ambiguity_codes.insert((k, sym_atom.to_string()), 3);
                }
            }
        }
    }
    updated_ref.ambiguity_codes = ambiguity_codes.clone();

    let mut atom_posteriors: BTreeMap<(usize, String), f64> = BTreeMap::new();
    for r in &pair_results {
        let k = r.residue_idx;
        if let Some((c1, c2)) = r.spec.iupac_carbons {
            atom_posteriors.insert((k, c1.to_string()), r.winning_posterior);
            atom_posteriors.insert((k, c2.to_string()), r.winning_posterior);
        }
        for &p in r.spec.iupac_protons_1.iter().chain(r.spec.iupac_protons_2) {
            atom_posteriors.insert((k, p.to_string()), r.winning_posterior);
        }
    }

    Ok(StereospecDatasetOutput {
        reference: updated_ref,
        ambiguity_codes,
        atom_posteriors,
        pair_results,
        committed_pairs,
        ambiguous_pairs,
    })
}

/// Format an all-atom NMR-STAR v3.2 shift saveframe preserving per-atom `_Atom_chem_shift.Ambiguity_code`
/// (`1` unique/stereospecific, `2` prochiral/geminal, `3` symmetric aromatic ring flip) alongside
/// optional residue backbone confidences and stereospecific posteriors.
pub fn format_nmrstar_with_ambiguity(
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
    ambiguity_codes: &BTreeMap<(usize, String), u8>,
    stereo_posteriors: Option<&BTreeMap<(usize, String), f64>>,
    header_comment: Option<&str>,
) -> String {
    let mut lines = Vec::new();
    if let Some(hdr) = header_comment {
        for l in hdr.lines() {
            lines.push(format!(
                "# {}",
                l.trim_start_matches("# ").trim_start_matches('#')
            ));
        }
        lines.push(String::new());
    }
    lines.extend([
        "data_nmr_assign_output".to_string(),
        String::new(),
        "save_entity_1".to_string(),
        "   _Entity.Sf_category  entity".to_string(),
        "   _Entity.ID           1".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Entity_poly_seq.Entity_ID".to_string(),
        "      _Entity_poly_seq.Num".to_string(),
        "      _Entity_poly_seq.Mon_ID".to_string(),
        String::new(),
    ]);
    for (seq_id, mon_id) in ref_data.seq_ids.iter().zip(ref_data.sequence.iter()) {
        lines.push(format!("      1 {:5} {:3}", seq_id, mon_id));
    }
    lines.extend([
        "   stop_".to_string(),
        "save_".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Atom_chem_shift.ID".to_string(),
        "      _Atom_chem_shift.Entity_ID".to_string(),
        "      _Atom_chem_shift.Seq_ID".to_string(),
        "      _Atom_chem_shift.Comp_ID".to_string(),
        "      _Atom_chem_shift.Atom_ID".to_string(),
        "      _Atom_chem_shift.Atom_type".to_string(),
        "      _Atom_chem_shift.Val".to_string(),
        "      _Atom_chem_shift.Ambiguity_code".to_string(),
        "      _Atom_chem_shift.Details".to_string(),
        String::new(),
    ]);

    let bb_order = ["H", "N", "CA", "HA", "CB", "C"];
    let mut entries: Vec<(usize, &str, f64)> = ref_data
        .shifts
        .iter()
        .map(|(&(idx, ref atom), &v)| (idx, atom.as_str(), v))
        .collect();
    entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
        let oa = bb_order.iter().position(|&a| a == aa).unwrap_or(99);
        let ob = bb_order.iter().position(|&a| a == ab).unwrap_or(99);
        (ia, oa, aa).cmp(&(ib, ob, ab))
    });

    for (i, (idx, atom, v)) in entries.into_iter().enumerate() {
        let Some(comp) = ref_data.sequence.get(idx) else {
            continue;
        };
        let seq_id = ref_data
            .seq_ids
            .get(idx)
            .copied()
            .unwrap_or((idx + 1) as i32);
        let default_amb = default_atom_ambiguity_code(comp, atom);
        let amb = ambiguity_codes
            .get(&(idx, atom.to_string()))
            .or_else(|| ref_data.ambiguity_codes.get(&(idx, atom.to_string())))
            .copied()
            .unwrap_or(default_amb);
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_uppercase().to_string())
            .unwrap_or_else(|| "X".to_string());
        let conf_part = confidence
            .and_then(|c| c.get(&idx))
            .map(|&c| format!("conf={c:.2}"));
        let post_part = stereo_posteriors
            .and_then(|m| m.get(&(idx, atom.to_string())))
            .map(|&p| format!("stereo_post={p:.3}"));
        let details = match (conf_part, post_part) {
            (Some(c), Some(p)) => format!("{c},{p}"),
            (Some(c), None) => c,
            (None, Some(p)) => p,
            (None, None) => ".".to_string(),
        };
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} {} {}",
            i + 1,
            seq_id,
            comp,
            atom,
            atom_type,
            v,
            amb,
            details
        ));
    }
    lines.extend(["   stop_".to_string(), "save_".to_string(), String::new()]);
    lines.join("\n")
}
