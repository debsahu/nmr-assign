//! Parallel tripeptide `(i-1, i, i+1)` fragment & cooperative secondary-shift backbone assignment solver.
//!
//! Provenance: Clean-room implementation from peer-reviewed NMR literature and public CC0 BMRB data:
//! 1. Shen, Y., Lange, O., Delaglio, F., Rossi, P., Aramini, J. M., Liu, G., Eletsky, A., Wu, Y.,
//!    Singarapu, K. K., Lemak, A., Ignatchenko, A., Arrowsmith, C. H., Szyperski, T., Montelione, G. T.,
//!    Baker, D. & Bax, A. (2008). "Consistent blind protein structure generation from NMR chemical shift
//!    data." *Proc. Natl. Acad. Sci. USA* 105(12), 4685–4690 (chemical-shift chi-squared + BLOSUM62
//!    sequence-similarity fragment scoring, Eqs. 1–3, with sequence-homolog exclusion).
//! 2. Shen, Y. & Bax, A. (2013). "Protein backbone and sidechain torsion angles predicted from NMR
//!    chemical shifts using artificial neural networks." *J. Biomol. NMR* 56(3), 227–241 (tripeptide
//!    `(i-1, i, i+1)` secondary chemical shifts and sequence-context conformational propensities).
//! 3. Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (1995). "1H, 13C and 15N
//!    random coil NMR chemical shifts of the common amino acids. I. Investigations of nearest-neighbor
//!    effects." *J. Biomol. NMR* 5(1), 67–81; and Schwarzinger, S., Kroon, G. J. A., Foss, T. R.,
//!    Chung, J., Wright, P. E. & Dyson, H. J. (2001). "Sequence-dependent correction of random coil
//!    NMR chemical shifts." *J. Am. Chem. Soc.* 123(13), 2970–2978 (`i +/- 1` neighbor corrections).
//! 4. Spera, S. & Bax, A. (1991). *J. Am. Chem. Soc.* 113(14), 5490–5492 (`13CA`/`13CB` secondary shifts);
//!    Henikoff, S. & Henikoff, J. G. (1992). *Proc. Natl. Acad. Sci. USA* 89(22), 10915–10919 (BLOSUM62);
//!    Zhang, H., Neal, S. & Wishart, D. S. (2003). *J. Biomol. NMR* 25(3), 173–195 (reference alignment).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::nmrstar::{read_star, Reference};
use crate::priors::{logpdf, Priors};
use crate::rng::Rng;
use crate::solver::{
    consensus_from_runs, AnnealConfig, AssignResult, Matrix2D, ScoreConfig, FORBID,
};
use crate::spinsystems::SpinSystem;

/// Four canonical Ramachandran conformational basins for tripeptide secondary-shift modeling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConformationalBasin {
    Helix,
    Sheet,
    Ppii,
    Turn,
}

/// Estimated global chemical-shift reference offsets (`observed_ppm - dss_prior_ppm`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReferenceOffsets {
    pub dca: f64,
    pub dcb: f64,
    pub dc: f64,
    pub dh: f64,
    pub dn: f64,
}

impl ReferenceOffsets {
    pub const fn zero() -> Self {
        Self {
            dca: 0.0,
            dcb: 0.0,
            dc: 0.0,
            dh: 0.0,
            dn: 0.0,
        }
    }

    #[inline]
    pub fn for_atom(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.dca,
            "CB" => self.dcb,
            "C" => self.dc,
            "H" => self.dh,
            "N" => self.dn,
            _ => 0.0,
        }
    }

    /// Shift `Priors` means into the experimental spectrum's chemical-shift frame (`mu + offset`).
    pub fn apply_to_priors(&self, priors: &Priors) -> Priors {
        let mut table = BTreeMap::new();
        for (res, atoms) in &priors.table {
            let mut m = BTreeMap::new();
            for (atom, &(mu, sd)) in atoms {
                m.insert(atom.clone(), (mu + self.for_atom(atom), sd));
            }
            table.insert(res.clone(), m);
        }
        Priors { table }
    }
}

/// Configuration for the parallel tripeptide fragment & cooperative secondary-shift solver.
#[derive(Clone, Debug)]
pub struct TripeptideConfig {
    pub score: ScoreConfig,
    pub anneal: AnnealConfig,
    /// Automatically estimate global `(dCA, dCB, dC, dH, dN)` reference offsets (e.g. TMS vs DSS).
    pub auto_reference: bool,
    /// Exclude harvested BMRB entries homologous (`>= max_homolog_identity` or 7-mer match) to query.
    pub exclude_homologs: bool,
    pub max_homolog_identity: f64,
    /// Weight of Wishart/Schwarzinger sequence-neighbor-adjusted component in unary typing (`[0, 1)`).
    pub neighbor_rc_weight: f64,
    /// Weight of `(k-1, k)` empirical dipeptide / conformational coherence in `U_tri(s, k)`.
    pub dipeptide_weight: f64,
    /// Weight of `(k-1, k, k+1)` cooperative Shen–Baker–Bax tripeptide score in `P_tri(s, t, k)`.
    pub tripeptide_weight: f64,
    /// Bayesian mixture weight `lambda_coop` on cooperative vs independent secondary shifts (`[0, 1)`).
    pub coop_mix: f64,
    /// Tie-breaking log-penalty per candidate rank index (preserving intensity & `(H, N)` ordering).
    pub cand_rank_penalty: f64,
    /// Offer soft degenerate `i == i-1` connectivity links when `support(i-1) >= 2` and `i` is absent.
    pub allow_soft_degenerate: bool,
    /// Seed half of annealing replicas with stochastic high-margin tripeptide chain growth.
    pub seed_growth: bool,
    /// Gate amide `1H` window and 0-`CA` noise strips when `systems.len()` vastly exceeds assignable slots.
    pub spurious_root_guard: bool,
}

impl Default for TripeptideConfig {
    fn default() -> Self {
        Self {
            score: ScoreConfig::default(),
            anneal: AnnealConfig {
                runs: 20,
                steps_per_system: 2500,
                t_start: 10.0,
                t_end: 0.05,
                max_block: 8,
                seed: 0,
            },
            auto_reference: true,
            exclude_homologs: true,
            max_homolog_identity: 0.35,
            neighbor_rc_weight: 0.35,
            dipeptide_weight: 0.25,
            tripeptide_weight: 0.65,
            coop_mix: 0.55,
            cand_rank_penalty: 0.03,
            allow_soft_degenerate: true,
            seed_growth: false,
            spurious_root_guard: true,
        }
    }
}

impl TripeptideConfig {
    pub fn from_configs(score: &ScoreConfig, anneal: &AnnealConfig) -> Self {
        Self {
            score: score.clone(),
            anneal: anneal.clone(),
            ..Self::default()
        }
    }
}

// ============================================================================
// Amino-Acid Indexing (`0..20` in `STANDARD_RESIDUES` order) & BLOSUM62
// ============================================================================

const AA_UNK: u8 = 255;
const AA_ALA: u8 = 0;
const AA_ARG: u8 = 1;
const AA_ASN: u8 = 2;
const AA_ASP: u8 = 3;
const AA_CYS: u8 = 4;
const AA_GLN: u8 = 5;
const AA_GLU: u8 = 6;
const AA_GLY: u8 = 7;
const AA_HIS: u8 = 8;
const AA_ILE: u8 = 9;
const AA_LEU: u8 = 10;
const AA_LYS: u8 = 11;
const AA_MET: u8 = 12;
const AA_PHE: u8 = 13;
const AA_PRO: u8 = 14;
const AA_SER: u8 = 15;
const AA_THR: u8 = 16;
const AA_TRP: u8 = 17;
const AA_TYR: u8 = 18;
const AA_VAL: u8 = 19;

#[inline]
fn aa_index(res: &str) -> u8 {
    match res.trim().to_ascii_uppercase().as_str() {
        "ALA" | "A" => AA_ALA,
        "ARG" | "R" => AA_ARG,
        "ASN" | "N" => AA_ASN,
        "ASP" | "D" => AA_ASP,
        "CYS" | "C" => AA_CYS,
        "GLN" | "Q" => AA_GLN,
        "GLU" | "E" => AA_GLU,
        "GLY" | "G" => AA_GLY,
        "HIS" | "H" => AA_HIS,
        "ILE" | "I" => AA_ILE,
        "LEU" | "L" => AA_LEU,
        "LYS" | "K" => AA_LYS,
        "MET" | "M" => AA_MET,
        "PHE" | "F" => AA_PHE,
        "PRO" | "P" => AA_PRO,
        "SER" | "S" => AA_SER,
        "THR" | "T" => AA_THR,
        "TRP" | "W" => AA_TRP,
        "TYR" | "Y" => AA_TYR,
        "VAL" | "V" => AA_VAL,
        _ => AA_UNK,
    }
}

#[inline]
fn atom_index(atom: &str) -> Option<usize> {
    match atom {
        "CA" => Some(0),
        "CB" => Some(1),
        "C" => Some(2),
        "N" => Some(3),
        "H" => Some(4),
        _ => None,
    }
}

/// Standard symmetric BLOSUM62 log-odds substitution matrix (Henikoff & Henikoff, *PNAS* 1992)
/// ordered as `A R N D C Q E G H I L K M F P S T W Y V`.
const BLOSUM62: [[i8; 20]; 20] = [
    // A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [
        4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0,
    ], // A
    [
        -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3,
    ], // R
    [
        -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3,
    ], // N
    [
        -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3,
    ], // D
    [
        0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,
    ], // C
    [
        -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2,
    ], // Q
    [
        -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2,
    ], // E
    [
        0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3,
    ], // G
    [
        -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3,
    ], // H
    [
        -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3,
    ], // I
    [
        -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1,
    ], // L
    [
        -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2,
    ], // K
    [
        -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1,
    ], // M
    [
        -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1,
    ], // F
    [
        -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2,
    ], // P
    [
        1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2,
    ], // S
    [
        0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0,
    ], // T
    [
        -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3,
    ], // W
    [
        -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1,
    ], // Y
    [
        0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4,
    ], // V
];

#[inline]
pub fn blosum62_score(a: &str, b: &str) -> i32 {
    let ia = aa_index(a);
    let ib = aa_index(b);
    if ia < 20 && ib < 20 {
        BLOSUM62[ia as usize][ib as usize] as i32
    } else {
        0
    }
}

// ============================================================================
// Neighbor-Corrected Random-Coil Shifts (Wishart 1995 + Schwarzinger 2001)
// ============================================================================

/// Base DSS random-coil shifts `[CA, CB, C, N, H]` in `GGXGG` peptides (Wishart et al., 1995).
const BASE_RC: [[f64; 5]; 20] = [
    [52.5, 19.1, 177.8, 123.8, 8.24], // ALA
    [56.0, 30.9, 176.3, 121.3, 8.23], // ARG
    [53.1, 38.9, 175.2, 118.7, 8.40], // ASN
    [54.2, 41.1, 176.3, 120.4, 8.34], // ASP
    [58.2, 28.0, 174.6, 118.8, 8.32], // CYS (reduced; oxidized handled dynamically)
    [55.7, 29.4, 176.0, 119.8, 8.32], // GLN
    [56.6, 29.9, 176.6, 120.2, 8.42], // GLU
    [45.1, 0.00, 173.9, 108.8, 8.33], // GLY (no CB)
    [55.0, 29.0, 174.1, 118.2, 8.42], // HIS
    [61.1, 38.8, 176.4, 119.9, 8.00], // ILE
    [55.1, 42.4, 177.6, 121.8, 8.16], // LEU
    [56.2, 33.1, 176.6, 120.4, 8.29], // LYS
    [55.4, 32.9, 176.3, 119.6, 8.28], // MET
    [57.7, 39.6, 175.8, 120.3, 8.30], // PHE
    [63.3, 32.1, 177.3, 135.0, 0.00], // PRO (no H)
    [58.3, 63.8, 174.6, 115.7, 8.31], // SER
    [61.8, 69.8, 174.7, 113.6, 8.15], // THR
    [57.3, 29.6, 176.1, 121.3, 8.25], // TRP
    [57.9, 38.8, 175.9, 120.3, 8.12], // TYR
    [62.2, 32.9, 176.3, 119.2, 8.03], // VAL
];

/// Preceding-residue (`aa_{k-1}`) substituent effect `[CA, CB, C, N, H]` on residue `k`
/// relative to `GLY` (`GGXGG`), per Schwarzinger et al. (*JACS* 2001) & Wishart et al. (1995).
const PREV_CORR: [[f64; 5]; 20] = [
    [-0.05, 0.00, -0.06, 0.45, 0.00],   // ALA
    [-0.08, 0.00, -0.10, 1.15, 0.02],   // ARG
    [-0.06, 0.00, -0.08, 0.70, 0.01],   // ASN
    [-0.05, 0.00, -0.06, 0.65, 0.00],   // ASP
    [-0.08, 0.00, -0.10, 0.95, 0.02],   // CYS
    [-0.08, 0.00, -0.10, 1.10, 0.02],   // GLN
    [-0.07, 0.00, -0.10, 1.05, 0.02],   // GLU
    [0.00, 0.00, 0.00, 0.00, 0.00],     // GLY (host reference)
    [-0.18, -0.05, -0.20, 1.25, -0.03], // HIS
    [-0.22, 0.04, -0.30, 2.55, 0.06],   // ILE (beta-branched)
    [-0.09, 0.00, -0.12, 1.05, 0.01],   // LEU
    [-0.08, 0.00, -0.10, 1.10, 0.02],   // LYS
    [-0.09, 0.00, -0.12, 1.10, 0.01],   // MET
    [-0.25, -0.08, -0.25, 1.70, -0.08], // PHE (aromatic)
    [-0.15, 0.10, -0.35, 1.30, 0.12],   // PRO (post-Proline)
    [-0.05, 0.00, -0.08, 0.35, 0.02],   // SER
    [-0.12, 0.02, -0.18, 1.35, 0.04],   // THR (beta-branched)
    [-0.28, -0.10, -0.28, 1.90, -0.10], // TRP (aromatic)
    [-0.25, -0.08, -0.25, 1.75, -0.08], // TYR (aromatic)
    [-0.20, 0.04, -0.28, 2.35, 0.06],   // VAL (beta-branched)
];

/// Following-residue (`aa_{k+1}`) substituent effect `[CA, CB, C, N, H]` on non-Gly residue `k`
/// relative to `GLY` (`GGXGG`), per Wishart et al. (1995, Table 3) & Schwarzinger et al. (2001).
const NEXT_CORR: [[f64; 5]; 20] = [
    [-0.03, 0.00, -0.05, -0.05, 0.00],  // ALA
    [-0.06, 0.00, -0.08, 0.05, 0.01],   // ARG
    [-0.05, 0.00, -0.08, 0.00, 0.01],   // ASN
    [-0.04, 0.00, -0.06, 0.00, 0.01],   // ASP
    [-0.06, 0.00, -0.08, 0.05, 0.01],   // CYS
    [-0.06, 0.00, -0.08, 0.05, 0.01],   // GLN
    [-0.05, 0.00, -0.08, 0.05, 0.01],   // GLU
    [0.00, 0.00, 0.00, 0.00, 0.00],     // GLY (host reference)
    [-0.18, 0.05, -0.22, -0.18, -0.05], // HIS
    [-0.15, 0.05, -0.22, 0.20, 0.03],   // ILE
    [-0.06, 0.00, -0.08, 0.04, 0.00],   // LEU
    [-0.06, 0.00, -0.08, 0.05, 0.01],   // LYS
    [-0.06, 0.00, -0.08, 0.04, 0.00],   // MET
    [-0.22, 0.08, -0.30, -0.25, -0.07], // PHE
    [-1.95, 0.25, -2.00, 0.55, -0.08],  // PRO (pre-Proline!)
    [-0.04, 0.00, -0.06, -0.05, 0.01],  // SER
    [-0.12, 0.04, -0.18, 0.12, 0.02],   // THR
    [-0.28, 0.10, -0.32, -0.28, -0.10], // TRP
    [-0.22, 0.08, -0.30, -0.25, -0.07], // TYR
    [-0.15, 0.05, -0.22, 0.18, 0.03],   // VAL
];

#[inline]
fn rc_shift_indexed(
    prev_aa: u8,
    aa: u8,
    next_aa: u8,
    atom_idx: usize,
    cys_oxidized: bool,
) -> Option<f64> {
    if aa >= 20 || atom_idx >= 5 {
        return None;
    }
    if (aa == AA_GLY && atom_idx == 1) || (aa == AA_PRO && atom_idx == 4) {
        return None;
    }
    let mut base = BASE_RC[aa as usize][atom_idx];
    if aa == AA_CYS && cys_oxidized {
        if atom_idx == 0 {
            base = 55.6;
        } else if atom_idx == 1 {
            base = 41.1;
        }
    }
    let prev = if prev_aa < 20 {
        PREV_CORR[prev_aa as usize][atom_idx]
    } else {
        0.0
    };
    let next = if next_aa < 20 {
        if aa == AA_GLY && next_aa == AA_PRO {
            match atom_idx {
                0 => -1.00,
                2 => -1.50,
                3 => 0.40,
                4 => -0.05,
                _ => 0.0,
            }
        } else {
            NEXT_CORR[next_aa as usize][atom_idx]
        }
    } else {
        0.0
    };
    Some(base + prev + next)
}

/// Zero-centered neighbor shift adjustment relative to the uniform 20-residue average,
/// suitable for shifting empirical `Priors::bmrb()` means without global offset drift.
#[inline]
fn zero_mean_neighbor_delta(prev_aa: u8, aa: u8, next_aa: u8, atom_idx: usize) -> f64 {
    if aa >= 20 || atom_idx >= 5 {
        return 0.0;
    }
    let mean_prev = match atom_idx {
        0 => -0.12,
        1 => 0.00,
        2 => -0.15,
        3 => 1.15,
        4 => 0.01,
        _ => 0.0,
    };
    // Exclude Pro (~5% natural frequency) from dominating average next shift
    let mean_next = match atom_idx {
        0 => -0.18,
        1 => 0.03,
        2 => -0.22,
        3 => 0.00,
        4 => -0.01,
        _ => 0.0,
    };
    let prev = if prev_aa < 20 {
        PREV_CORR[prev_aa as usize][atom_idx] - mean_prev
    } else {
        0.0
    };
    let next = if next_aa < 20 {
        let raw_next = if aa == AA_GLY && next_aa == AA_PRO {
            match atom_idx {
                0 => -1.00,
                2 => -1.50,
                3 => 0.40,
                4 => -0.05,
                _ => 0.0,
            }
        } else {
            NEXT_CORR[next_aa as usize][atom_idx]
        };
        raw_next - mean_next
    } else {
        0.0
    };
    prev + next
}

/// Sequence-neighbor-corrected DSS random-coil shift (ppm) for `(prev_aa, aa, next_aa, atom)`.
pub fn neighbor_corrected_random_coil(
    prev_aa: Option<&str>,
    aa: &str,
    next_aa: Option<&str>,
    atom: &str,
    cys_oxidized: bool,
) -> Option<f64> {
    let p = prev_aa.map(aa_index).unwrap_or(AA_UNK);
    let c = aa_index(aa);
    let n = next_aa.map(aa_index).unwrap_or(AA_UNK);
    let ai = atom_index(atom)?;
    rc_shift_indexed(p, c, n, ai, cys_oxidized)
}

/// Compute neighbor-corrected secondary chemical shift:
/// `(obs_ppm - ref_offset_ppm) - (mu_RC(aa, X) + d_prev(prev, X) + d_next(next, X))`.
pub fn secondary_shift(
    obs_ppm: f64,
    ref_offset_ppm: f64,
    prev_aa: Option<&str>,
    aa: &str,
    next_aa: Option<&str>,
    atom: &str,
    cys_oxidized: bool,
) -> Option<f64> {
    let rc = neighbor_corrected_random_coil(prev_aa, aa, next_aa, atom, cys_oxidized)?;
    Some((obs_ppm - ref_offset_ppm) - rc)
}

// ============================================================================
// Automatic Global Chemical-Shift Reference Offset Estimation
// ============================================================================

fn mixture_sum_loglike(obs: &[f64], comps: &[(f64, f64)], offset: f64) -> f64 {
    if obs.is_empty() || comps.is_empty() {
        return 0.0;
    }
    let inv_k = 1.0 / (comps.len() as f64);
    let outlier_floor = 1e-4_f64;
    let mut total = 0.0;
    for &v in obs {
        let x = v - offset;
        let mut max_lp = f64::NEG_INFINITY;
        for &(mu, sd) in comps {
            let lp = logpdf(x, mu, sd);
            if lp > max_lp {
                max_lp = lp;
            }
        }
        let mut sum_exp = 0.0;
        for &(mu, sd) in comps {
            sum_exp += (logpdf(x, mu, sd) - max_lp).exp();
        }
        let density = (max_lp.exp() * sum_exp * inv_k).max(outlier_floor);
        total += density.ln();
    }
    total
}

/// Estimate global `(dCA, dCB, dC, dH, dN)` chemical-shift reference offsets (`obs - DSS`)
/// by maximizing robust marginal sequence-mixture log-likelihood with aliphatic `dCA ≈ dCB` coupling.
pub fn estimate_reference_offsets(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
) -> ReferenceOffsets {
    if systems.is_empty() || sequence.is_empty() {
        return ReferenceOffsets::zero();
    }
    let assignable = sequence
        .iter()
        .enumerate()
        .filter(|&(k, r)| k > 0 && !r.eq_ignore_ascii_case("PRO"))
        .count()
        .max(1);

    let amide_roots_count = systems
        .iter()
        .filter(|ss| ss.has_observed_carbons() && (5.2..=11.8).contains(&ss.h))
        .count();
    let enforce_amide_h = amide_roots_count >= assignable / 2;

    // Rank spin systems by carbon completeness & support so spurious 1-peak noise roots do not bias referencing
    let mut ranked: Vec<(usize, usize, usize)> = systems
        .iter()
        .enumerate()
        .filter(|(_, ss)| {
            ss.has_observed_carbons() && (!enforce_amide_h || (5.2..=11.8).contains(&ss.h))
        })
        .map(|(idx, ss)| {
            let mut distinct_slots = 0usize;
            let mut sum_sup = 0usize;
            for list in ss.cands.values() {
                if let Some(&(_, sup)) = list.first() {
                    if sup > 0 {
                        distinct_slots += 1;
                        sum_sup += sup;
                    }
                }
            }
            (idx, distinct_slots, sum_sup)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    let keep = (assignable + assignable / 8).max(12).min(ranked.len());
    let chosen = &ranked[..keep];

    let mut obs_ca = Vec::new();
    let mut obs_cb = Vec::new();
    let mut obs_c = Vec::new();
    let mut obs_n = Vec::new();
    let mut obs_h = Vec::new();

    for &(idx, slots_n, _) in chosen {
        // When well-populated spin systems exist, ignore 1-slot noise fragments
        if keep >= 16 && slots_n < 2 && ranked[0].1 >= 3 {
            continue;
        }
        let ss = &systems[idx];
        if (5.0..=12.2).contains(&ss.h) {
            obs_h.push(ss.h);
        }
        if (95.0..=142.0).contains(&ss.n) {
            obs_n.push(ss.n);
        }
        for ((atom, _off), list) in &ss.cands {
            if let Some(&(v, sup)) = list.first() {
                if sup == 0 {
                    continue;
                }
                match atom.as_str() {
                    "CA" if (36.0..=74.0).contains(&v) => obs_ca.push(v),
                    "CB" if (10.0..=78.0).contains(&v) => obs_cb.push(v),
                    "C" if (164.0..=187.0).contains(&v) => obs_c.push(v),
                    _ => {}
                }
            }
        }
    }

    let build_comps = |atom: &str, pad_sd: f64| -> Vec<(f64, f64)> {
        let ai = atom_index(atom).unwrap_or(0);
        let mut out = Vec::with_capacity(sequence.len());
        for k in 0..sequence.len() {
            let res = sequence[k].to_ascii_uppercase();
            if (atom == "H" || atom == "N") && (k == 0 || res == "PRO") {
                continue;
            }
            if let Some((mu, sd)) = priors.get(&res, atom) {
                let prev_aa = if k > 0 {
                    aa_index(&sequence[k - 1])
                } else {
                    AA_UNK
                };
                let curr_aa = aa_index(&res);
                let next_aa = if k + 1 < sequence.len() {
                    aa_index(&sequence[k + 1])
                } else {
                    AA_UNK
                };
                let nb = zero_mean_neighbor_delta(prev_aa, curr_aa, next_aa, ai);
                out.push((mu + nb, (sd * sd + pad_sd * pad_sd).sqrt()));
            }
        }
        out
    };

    let comps_ca = build_comps("CA", 0.50);
    let comps_cb = build_comps("CB", 0.50);
    let comps_c = build_comps("C", 0.50);
    let comps_n = build_comps("N", 0.75);
    let comps_h = build_comps("H", 0.12);

    // 1. Joint aliphatic 13C scan over total log-likelihood:
    //    secondary-structure shifts on CA and CB have opposite signs and cancel!
    let mut best_aliph = 0.0;
    if obs_ca.len() >= 8 && !comps_ca.is_empty() {
        let mut best_obj = f64::NEG_INFINITY;
        for step in -120..=120 {
            let d = (step as f64) * 0.05;
            let l_ca = mixture_sum_loglike(&obs_ca, &comps_ca, d);
            let l_cb = if obs_cb.len() >= 6 && !comps_cb.is_empty() {
                0.35 * mixture_sum_loglike(&obs_cb, &comps_cb, d)
            } else {
                0.0
            };
            // Weak dataset-level Gaussian prior (sigma = 6.0 ppm) avoids boundary artifacts without shrinking real offsets
            let reg = -0.5 * (d / 6.0).powi(2);
            let obj = l_ca + l_cb + reg;
            if obj > best_obj {
                best_obj = obj;
                best_aliph = d;
            }
        }
    }

    let refine_atom = |obs: &[f64],
                       comps: &[(f64, f64)],
                       center: f64,
                       span: f64,
                       step_sz: f64,
                       coupling_sd: f64|
     -> f64 {
        if obs.len() < 8 || comps.is_empty() {
            return 0.0;
        }
        let n_steps = ((span / step_sz).round() as i32).max(1);
        let mut best_d = center;
        let mut best_v = f64::NEG_INFINITY;
        for i in -n_steps..=n_steps {
            let d = center + (i as f64) * step_sz;
            let ll = mixture_sum_loglike(obs, comps, d);
            let prior = -0.5 * ((d - center) / coupling_sd).powi(2) - 0.5 * (d / 6.0).powi(2);
            let val = ll + prior;
            if val > best_v {
                best_v = val;
                best_d = d;
            }
        }
        best_d
    };

    // Aliphatic `CA` and `CB` share a single `13C` indirect axis in HNCACB / CBCA(CO)NH, so their
    // reference offset (`obs - DSS`) is identical (`best_aliph` refined at 0.01 ppm precision).
    let mut refined_aliph = best_aliph;
    if obs_ca.len() >= 8 && !comps_ca.is_empty() {
        let mut best_v = f64::NEG_INFINITY;
        for i in -35..=35 {
            let d = best_aliph + (i as f64) * 0.01;
            let l_ca = mixture_sum_loglike(&obs_ca, &comps_ca, d);
            let l_cb = if obs_cb.len() >= 6 && !comps_cb.is_empty() {
                0.35 * mixture_sum_loglike(&obs_cb, &comps_cb, d)
            } else {
                0.0
            };
            let val = l_ca + l_cb - 0.5 * (d / 6.0).powi(2);
            if val > best_v {
                best_v = val;
                refined_aliph = d;
            }
        }
    }
    let raw_dca = refined_aliph;
    let raw_dcb = refined_aliph;
    // Carbonyl `13CO` shares the same spectrometer 13C chemical-shift reference standard (DSS vs TMS)
    // unless a separate carrier miscalibration (> 1.6 ppm discrepancy) is present; tight coupling prevents
    // secondary-structure composition (e.g. beta-sheet rich proteins) from biasing `dc` by ~0.4 ppm.
    let free_dc = refine_atom(&obs_c, &comps_c, refined_aliph, 4.50, 0.05, 3.00);
    let raw_dc = if (free_dc - refined_aliph).abs() <= 1.60 {
        refine_atom(&obs_c, &comps_c, refined_aliph, 0.25, 0.01, 0.14)
    } else {
        free_dc
    };
    let raw_dn = refine_atom(&obs_n, &comps_n, 0.0, 4.00, 0.05, 3.00);
    let raw_dh = refine_atom(&obs_h, &comps_h, 0.0, 0.80, 0.01, 0.50);

    // Deadband keeps DSS-referenced datasets (including all-alpha/all-beta proteins) strictly at 0.0 ppm
    let aliph_shifted = best_aliph.abs() >= 1.15;
    let (dca, dcb) = if aliph_shifted {
        (raw_dca, raw_dcb)
    } else {
        (0.0, 0.0)
    };
    let dc = if (aliph_shifted && raw_dc.signum() == best_aliph.signum() && raw_dc.abs() >= 1.00)
        || raw_dc.abs() >= 1.85
    {
        raw_dc
    } else {
        0.0
    };
    let dn = if raw_dn.abs() >= 1.35 { raw_dn } else { 0.0 };
    let dh = if raw_dh.abs() >= 0.20 { raw_dh } else { 0.0 };

    ReferenceOffsets {
        dca,
        dcb,
        dc,
        dh,
        dn,
    }
}

// ============================================================================
// Tripeptide Fragment Library & 4-State Conformational Basin Templates
// ============================================================================

/// Empirical or canonical tripeptide `(i-1, i, i+1)` secondary-shift fragment.
#[derive(Clone, Debug)]
pub struct TripeptideFragment {
    /// Indexed standard amino acids `[aa_{i-1}, aa_i, aa_{i+1}]` (`0..20`).
    pub aa: [u8; 3],
    pub dca: [Option<f32>; 3],
    pub dcb: [Option<f32>; 3],
    pub dc: [Option<f32>; 3],
    pub dn: [Option<f32>; 3],
    pub dh: [Option<f32>; 3],
    pub basin: ConformationalBasin,
    /// `u16::MAX` for embedded standalone archetypes; otherwise index into `TripeptideFragmentDb::sources`.
    pub source_id: u16,
}

#[derive(Clone, Debug)]
pub struct FragmentSource {
    pub id: String,
    pub sequence: Vec<u8>,
}

/// Library of harvested BMRB tripeptides + standalone 4-state conformational basin templates.
#[derive(Clone, Debug)]
pub struct TripeptideFragmentDb {
    pub fragments: Vec<TripeptideFragment>,
    pub sources: Vec<FragmentSource>,
}

static DEFAULT_FRAGMENT_DB: OnceLock<TripeptideFragmentDb> = OnceLock::new();

/// Multi-residue conformational transition templates across `(m = -1, 0, +1)` so standalone/WASM
/// builds capture cooperative helices, sheets, caps, turns, and random-coil/PPII segments.
const TRANSITION_TEMPLATES: [([ConformationalBasin; 3], f64); 8] = [
    (
        [
            ConformationalBasin::Helix,
            ConformationalBasin::Helix,
            ConformationalBasin::Helix,
        ],
        0.24,
    ),
    (
        [
            ConformationalBasin::Sheet,
            ConformationalBasin::Sheet,
            ConformationalBasin::Sheet,
        ],
        0.20,
    ),
    (
        [
            ConformationalBasin::Ppii,
            ConformationalBasin::Ppii,
            ConformationalBasin::Ppii,
        ],
        0.22,
    ),
    (
        [
            ConformationalBasin::Ppii,
            ConformationalBasin::Helix,
            ConformationalBasin::Helix,
        ],
        0.07,
    ),
    (
        [
            ConformationalBasin::Helix,
            ConformationalBasin::Helix,
            ConformationalBasin::Turn,
        ],
        0.07,
    ),
    (
        [
            ConformationalBasin::Ppii,
            ConformationalBasin::Sheet,
            ConformationalBasin::Sheet,
        ],
        0.07,
    ),
    (
        [
            ConformationalBasin::Sheet,
            ConformationalBasin::Sheet,
            ConformationalBasin::Turn,
        ],
        0.07,
    ),
    (
        [
            ConformationalBasin::Turn,
            ConformationalBasin::Turn,
            ConformationalBasin::Ppii,
        ],
        0.06,
    ),
];

/// Expected secondary shift `(mean_ppm, sd_ppm)` for `(aa, basin, atom_idx)`.
#[inline]
fn basin_atom_profile(aa: u8, basin: ConformationalBasin, atom_idx: usize) -> (f64, f64) {
    let is_gly = aa == AA_GLY;
    let is_pro = aa == AA_PRO;
    let is_beta_br = matches!(aa, AA_VAL | AA_ILE | AA_THR);

    match basin {
        ConformationalBasin::Helix => match atom_idx {
            0 => {
                if is_gly {
                    (1.90, 1.05)
                } else if is_pro {
                    (2.10, 1.05)
                } else {
                    (3.15, 1.15)
                }
            }
            1 => (-0.55, 1.00),
            2 => {
                if is_gly {
                    (1.80, 1.00)
                } else {
                    (2.15, 1.05)
                }
            }
            3 => (-1.55, 2.75),
            _ => (-0.28, 0.48),
        },
        ConformationalBasin::Sheet => match atom_idx {
            0 => {
                if is_gly {
                    (-0.95, 0.95)
                } else if is_beta_br {
                    (-1.85, 1.15)
                } else {
                    (-1.50, 1.10)
                }
            }
            1 => {
                if is_beta_br {
                    (2.15, 1.20)
                } else {
                    (1.80, 1.20)
                }
            }
            2 => (-1.45, 1.05),
            3 => (1.35, 2.85),
            _ => (0.42, 0.50),
        },
        ConformationalBasin::Ppii => match atom_idx {
            0 => (0.05, 0.85),
            1 => (0.05, 0.85),
            2 => (-0.08, 0.85),
            3 => (0.00, 2.15),
            _ => (0.02, 0.42),
        },
        ConformationalBasin::Turn => match atom_idx {
            0 => (0.35, 1.35),
            1 => (0.20, 1.25),
            2 => (0.25, 1.25),
            3 => (-0.65, 2.85),
            _ => (-0.05, 0.52),
        },
    }
}

/// Relative amino-acid conformational propensity multiplier for `(aa, basin)`
/// from peer-reviewed secondary-structure & coil conformational statistics (Shen & Bax, 2013).
#[inline]
fn residue_basin_propensity(aa: u8, basin: ConformationalBasin, pos_in_tri: i32) -> f64 {
    match basin {
        ConformationalBasin::Helix => match aa {
            AA_ALA | AA_GLU | AA_LEU | AA_MET | AA_GLN | AA_LYS | AA_ARG => 1.38,
            AA_PRO => {
                if pos_in_tri <= 0 {
                    0.55
                } else {
                    0.15
                }
            }
            AA_GLY => 0.48,
            AA_VAL | AA_ILE | AA_THR | AA_SER | AA_ASN | AA_ASP => 0.82,
            _ => 1.0,
        },
        ConformationalBasin::Sheet => match aa {
            AA_VAL | AA_ILE | AA_TYR | AA_PHE | AA_TRP | AA_THR | AA_CYS => 1.45,
            AA_LEU | AA_MET => 1.12,
            AA_PRO | AA_GLY => 0.38,
            AA_ASP | AA_ASN | AA_GLU | AA_SER => 0.78,
            _ => 0.95,
        },
        ConformationalBasin::Ppii => match aa {
            AA_PRO => 1.75,
            AA_ALA | AA_SER | AA_LYS | AA_GLN | AA_GLU | AA_ARG | AA_ASP => 1.12,
            AA_GLY => 1.15,
            _ => 0.95,
        },
        ConformationalBasin::Turn => match aa {
            AA_GLY => 1.95,
            AA_ASN | AA_ASP | AA_SER | AA_PRO => 1.55,
            AA_HIS | AA_LYS | AA_THR => 1.10,
            AA_VAL | AA_ILE => 0.65,
            _ => 0.90,
        },
    }
}

fn classify_basin(dca: Option<f32>, dcb: Option<f32>, dc: Option<f32>) -> ConformationalBasin {
    let ca = dca.unwrap_or(0.0) as f64;
    let cb = dcb.unwrap_or(0.0) as f64;
    let c = dc.unwrap_or(0.0) as f64;
    let csi = 0.55 * (ca - cb) + 0.45 * c;
    if csi >= 1.05 {
        ConformationalBasin::Helix
    } else if csi <= -0.95 {
        ConformationalBasin::Sheet
    } else if ca.abs() <= 0.85 && cb.abs() <= 0.85 && c.abs() <= 0.85 {
        ConformationalBasin::Ppii
    } else {
        ConformationalBasin::Turn
    }
}

fn detect_star_c13_offset(ref_data: &Reference) -> f64 {
    let n = ref_data.sequence.len();
    let mut dca = Vec::new();
    let mut dcb = Vec::new();
    for k in 0..n {
        let prev = if k > 0 {
            aa_index(&ref_data.sequence[k - 1])
        } else {
            AA_UNK
        };
        let curr = aa_index(&ref_data.sequence[k]);
        let next = if k + 1 < n {
            aa_index(&ref_data.sequence[k + 1])
        } else {
            AA_UNK
        };
        let cb_raw = ref_data.shifts.get(&(k, "CB".to_string())).copied();
        let cys_ox = curr == AA_CYS && cb_raw.unwrap_or(28.0) > 35.0;
        if let (Some(&ca), Some(rc_ca)) = (
            ref_data.shifts.get(&(k, "CA".to_string())),
            rc_shift_indexed(prev, curr, next, 0, cys_ox),
        ) {
            dca.push(ca - rc_ca);
        }
        if let (Some(cb), Some(rc_cb)) = (cb_raw, rc_shift_indexed(prev, curr, next, 1, cys_ox)) {
            dcb.push(cb - rc_cb);
        }
    }
    if dca.len() < 8 || dcb.len() < 8 {
        return 0.0;
    }
    dca.sort_by(|a, b| a.total_cmp(b));
    dcb.sort_by(|a, b| a.total_cmp(b));
    let m_ca = dca[dca.len() / 2];
    let m_cb = dcb[dcb.len() / 2];
    if m_ca.signum() == m_cb.signum() && m_ca.abs() > 1.2 && m_cb.abs() > 1.2 {
        0.5 * (m_ca + m_cb)
    } else {
        0.0
    }
}

impl Default for TripeptideFragmentDb {
    fn default() -> Self {
        Self::default_cached().clone()
    }
}

impl TripeptideFragmentDb {
    /// Cached process-wide library combining embedded 4-state archetypes + `data/benchmark/*.str` when present.
    pub fn default_cached() -> &'static TripeptideFragmentDb {
        DEFAULT_FRAGMENT_DB.get_or_init(|| {
            #[cfg(target_arch = "wasm32")]
            {
                Self::embedded()
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut db = Self::embedded();
                let mut candidates = Vec::new();
                let manifest_bench =
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark");
                if manifest_bench.is_dir() {
                    candidates.push(manifest_bench);
                } else if Path::new("data/benchmark").is_dir() {
                    candidates.push(PathBuf::from("data/benchmark"));
                }
                for dir in candidates {
                    if let Ok(rd) = std::fs::read_dir(&dir) {
                        let mut star_paths: Vec<PathBuf> = rd
                            .filter_map(|e| e.ok().map(|x| x.path()))
                            .filter(|p| p.extension().is_some_and(|ext| ext == "str"))
                            .collect();
                        star_paths.sort();
                        db.harvest_star_paths(&star_paths);
                    }
                }
                db
            }
        })
    }

    /// Standalone embedded tripeptide secondary-shift corpus + 4-state basin archetypes (zero disk I/O, WASM-ready).
    pub fn embedded() -> Self {
        let mut fragments = Vec::with_capacity(72);
        // Representative sequence-diverse tripeptide motifs across Helix, Sheet, PPII/Coil, and Turn
        let motifs: &[([u8; 3], [ConformationalBasin; 3])] = &[
            ([AA_ALA, AA_GLU, AA_LEU], [ConformationalBasin::Helix; 3]),
            ([AA_LEU, AA_LYS, AA_ALA], [ConformationalBasin::Helix; 3]),
            ([AA_GLU, AA_GLN, AA_ARG], [ConformationalBasin::Helix; 3]),
            ([AA_LYS, AA_ILE, AA_GLN], [ConformationalBasin::Helix; 3]),
            ([AA_ALA, AA_ALA, AA_LYS], [ConformationalBasin::Helix; 3]),
            ([AA_MET, AA_GLU, AA_ALA], [ConformationalBasin::Helix; 3]),
            ([AA_VAL, AA_ILE, AA_PHE], [ConformationalBasin::Sheet; 3]),
            ([AA_THR, AA_LEU, AA_THR], [ConformationalBasin::Sheet; 3]),
            ([AA_ILE, AA_PHE, AA_VAL], [ConformationalBasin::Sheet; 3]),
            ([AA_TYR, AA_LYS, AA_VAL], [ConformationalBasin::Sheet; 3]),
            ([AA_LEU, AA_HIS, AA_LEU], [ConformationalBasin::Sheet; 3]),
            ([AA_PHE, AA_THR, AA_ILE], [ConformationalBasin::Sheet; 3]),
            (
                [AA_LEU, AA_LYS, AA_GLY],
                [
                    ConformationalBasin::Helix,
                    ConformationalBasin::Helix,
                    ConformationalBasin::Turn,
                ],
            ),
            (
                [AA_SER, AA_ASP, AA_THR],
                [
                    ConformationalBasin::Ppii,
                    ConformationalBasin::Helix,
                    ConformationalBasin::Helix,
                ],
            ),
            (
                [AA_PRO, AA_ASP, AA_GLN],
                [
                    ConformationalBasin::Ppii,
                    ConformationalBasin::Turn,
                    ConformationalBasin::Helix,
                ],
            ),
            (
                [AA_ALA, AA_GLY, AA_LYS],
                [
                    ConformationalBasin::Sheet,
                    ConformationalBasin::Turn,
                    ConformationalBasin::Sheet,
                ],
            ),
            (
                [AA_GLU, AA_ASP, AA_GLY],
                [
                    ConformationalBasin::Ppii,
                    ConformationalBasin::Turn,
                    ConformationalBasin::Turn,
                ],
            ),
            (
                [AA_ILE, AA_PRO, AA_ASP],
                [
                    ConformationalBasin::Ppii,
                    ConformationalBasin::Ppii,
                    ConformationalBasin::Turn,
                ],
            ),
            ([AA_LYS, AA_THR, AA_LYS], [ConformationalBasin::Ppii; 3]),
            ([AA_GLY, AA_VAL, AA_ALA], [ConformationalBasin::Ppii; 3]),
        ];

        for &(aa, states) in motifs {
            let mut dca = [None; 3];
            let mut dcb = [None; 3];
            let mut dc = [None; 3];
            let mut dn = [None; 3];
            let mut dh = [None; 3];
            for m in 0..3 {
                dca[m] = Some(basin_atom_profile(aa[m], states[m], 0).0 as f32);
                if aa[m] != AA_GLY {
                    dcb[m] = Some(basin_atom_profile(aa[m], states[m], 1).0 as f32);
                }
                dc[m] = Some(basin_atom_profile(aa[m], states[m], 2).0 as f32);
                dn[m] = Some(basin_atom_profile(aa[m], states[m], 3).0 as f32);
                if aa[m] != AA_PRO {
                    dh[m] = Some(basin_atom_profile(aa[m], states[m], 4).0 as f32);
                }
            }
            fragments.push(TripeptideFragment {
                aa,
                dca,
                dcb,
                dc,
                dn,
                dh,
                basin: states[1],
                source_id: u16::MAX,
            });
        }

        Self {
            fragments,
            sources: Vec::new(),
        }
    }

    /// Build a library from embedded archetypes plus explicit NMR-STAR paths.
    pub fn from_star_paths<P: AsRef<Path>>(paths: &[P]) -> Self {
        let mut db = Self::embedded();
        db.harvest_star_paths(paths);
        db
    }

    /// Build a library from embedded archetypes plus all `.str` files in `dir`.
    pub fn from_star_dir(dir: impl AsRef<Path>) -> Self {
        let mut db = Self::embedded();
        if let Ok(rd) = std::fs::read_dir(dir.as_ref()) {
            let mut paths: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|x| x.path()))
                .filter(|p| p.extension().is_some_and(|ext| ext == "str"))
                .collect();
            paths.sort();
            db.harvest_star_paths(&paths);
        }
        db
    }

    /// Harvest empirical contiguous `(i-1, i, i+1)` secondary-shift fragments from NMR-STAR files.
    pub fn harvest_star_paths<P: AsRef<Path>>(&mut self, paths: &[P]) {
        for p in paths {
            let Ok(ref_data) = read_star(p.as_ref(), None) else {
                continue;
            };
            let n = ref_data.sequence.len();
            if n < 3 {
                continue;
            }
            let seq_u8: Vec<u8> = ref_data.sequence.iter().map(|r| aa_index(r)).collect();
            let c13_off = detect_star_c13_offset(&ref_data);
            let src_id = self.sources.len() as u16;
            let src_name = p
                .as_ref()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("star")
                .to_string();
            self.sources.push(FragmentSource {
                id: src_name,
                sequence: seq_u8.clone(),
            });

            // Precompute neighbor-corrected secondary shifts across the chain
            let mut sec = vec![[None::<f32>; 5]; n];
            for k in 0..n {
                let prev = if k > 0 { seq_u8[k - 1] } else { AA_UNK };
                let curr = seq_u8[k];
                let next = if k + 1 < n { seq_u8[k + 1] } else { AA_UNK };
                if curr >= 20 {
                    continue;
                }
                let cb_obs = ref_data
                    .shifts
                    .get(&(k, "CB".to_string()))
                    .map(|&v| v - c13_off);
                let cys_ox = curr == AA_CYS && cb_obs.unwrap_or(28.0) > 35.0;
                for (ai, aname) in ["CA", "CB", "C", "N", "H"].iter().enumerate() {
                    let Some(&raw_v) = ref_data.shifts.get(&(k, (*aname).to_string())) else {
                        continue;
                    };
                    let off = if ai <= 2 { c13_off } else { 0.0 };
                    if let Some(rc) = rc_shift_indexed(prev, curr, next, ai, cys_ox) {
                        let dv = (raw_v - off) - rc;
                        let max_plausible = if ai == 3 {
                            14.0
                        } else if ai == 4 {
                            2.8
                        } else {
                            9.5
                        };
                        if dv.abs() <= max_plausible {
                            sec[k][ai] = Some(dv as f32);
                        }
                    }
                }
            }

            for k in 1..(n - 1) {
                if seq_u8[k - 1] >= 20 || seq_u8[k] >= 20 || seq_u8[k + 1] >= 20 {
                    continue;
                }
                let n_heavy = (0..3)
                    .map(|m| {
                        let r = k + m - 1;
                        usize::from(sec[r][0].is_some())
                            + usize::from(sec[r][1].is_some())
                            + usize::from(sec[r][2].is_some())
                    })
                    .sum::<usize>();
                if n_heavy < 5 {
                    continue;
                }
                let dca = [sec[k - 1][0], sec[k][0], sec[k + 1][0]];
                let dcb = [sec[k - 1][1], sec[k][1], sec[k + 1][1]];
                let dc = [sec[k - 1][2], sec[k][2], sec[k + 1][2]];
                let dn = [sec[k - 1][3], sec[k][3], sec[k + 1][3]];
                let dh = [sec[k - 1][4], sec[k][4], sec[k + 1][4]];
                self.fragments.push(TripeptideFragment {
                    aa: [seq_u8[k - 1], seq_u8[k], seq_u8[k + 1]],
                    dca,
                    dcb,
                    dc,
                    dn,
                    dh,
                    basin: classify_basin(dca[1], dcb[1], dc[1]),
                    source_id: src_id,
                });
            }
        }
    }

    /// Boolean mask over `sources` indicating which harvested entries are non-homologous to `sequence`
    /// (best sliding-window sequence identity `< max_identity` and zero exact 7-mer peptide overlap).
    pub fn non_homologous_mask(&self, sequence: &[String], max_identity: f64) -> Vec<bool> {
        let q: Vec<u8> = sequence.iter().map(|r| aa_index(r)).collect();
        self.sources
            .iter()
            .map(|src| !is_homologous(&q, &src.sequence, max_identity))
            .collect()
    }
}

fn is_homologous(query: &[u8], target: &[u8], max_identity: f64) -> bool {
    if query.is_empty() || target.is_empty() {
        return false;
    }
    // 1. Exact 7-residue peptide match check
    if query.len() >= 7 && target.len() >= 7 {
        for qw in query.windows(7) {
            if qw.iter().all(|&x| x < 20) && target.windows(7).any(|tw| tw == qw) {
                return true;
            }
        }
    }
    // 2. Best ungapped sliding-alignment global identity
    let min_len = query.len().min(target.len()).max(1);
    let qn = query.len() as isize;
    let tn = target.len() as isize;
    let mut best_matches = 0usize;
    for shift in -(tn - 1)..qn {
        let mut matches = 0usize;
        let mut overlap = 0usize;
        for qi in 0..qn {
            let ti = qi - shift;
            if ti >= 0 && ti < tn {
                overlap += 1;
                let a = query[qi as usize];
                let b = target[ti as usize];
                if a < 20 && a == b {
                    matches += 1;
                }
            }
        }
        if overlap >= min_len / 2 && matches > best_matches {
            best_matches = matches;
        }
    }
    (best_matches as f64) / (min_len as f64) >= max_identity
}

// ============================================================================
// Precomputed Sequence-Window Conformational Modes & Tripeptide Scoring
// ============================================================================

const MAX_MODES: usize = 14;
type ModeProfile = [[(f64, f64); 5]; 3];

#[derive(Clone, Copy, Debug)]
struct TripeptideMode {
    log_weight: f64,
    /// `[m in 0..3][atom in 0..5] -> (mean_sec_shift, sd_sec_shift)`
    prof: ModeProfile,
}

#[derive(Clone, Debug)]
struct WindowModel {
    modes: Vec<TripeptideMode>,
    /// Uncorrelated single-residue marginal `(mean, sd)` across modes for each `[m][atom]`.
    marg: ModeProfile,
}

fn build_window_models(
    seq_u8: &[u8],
    db: &TripeptideFragmentDb,
    allowed_sources: &[bool],
) -> Vec<WindowModel> {
    let k_len = seq_u8.len();
    let mut out = Vec::with_capacity(k_len);

    for k in 0..k_len {
        let w_aa = [
            if k > 0 { seq_u8[k - 1] } else { AA_ALA },
            seq_u8[k].min(19),
            if k + 1 < k_len { seq_u8[k + 1] } else { AA_ALA },
        ];

        let mut raw_modes: Vec<(f64, ModeProfile)> = Vec::with_capacity(MAX_MODES);

        // 1. Sequence-propensity-weighted 8 multi-residue conformational basin/transition templates
        for &(states, base_prior) in &TRANSITION_TEMPLATES {
            let p0 = residue_basin_propensity(w_aa[0], states[0], -1);
            let p1 = residue_basin_propensity(w_aa[1], states[1], 0);
            let p2 = residue_basin_propensity(w_aa[2], states[2], 1);
            let w = (base_prior * p0 * p1 * p2).max(1e-4);
            let prof: ModeProfile = std::array::from_fn(|m| {
                std::array::from_fn(|ai| basin_atom_profile(w_aa[m], states[m], ai))
            });
            raw_modes.push((w, prof));
        }

        // 2. Top-6 BLOSUM62 sequence-matched empirical BMRB tripeptide fragments (Shen et al. PNAS 2008 Eqs. 2–3)
        let mut top_frags: Vec<(f64, usize)> = Vec::new();
        for (fi, frag) in db.fragments.iter().enumerate() {
            if frag.source_id != u16::MAX {
                let sid = frag.source_id as usize;
                if sid < allowed_sources.len() && !allowed_sources[sid] {
                    continue;
                }
            }
            // Enforce strict Gly/Pro compatibility at central & flanking positions (Shen & Bax 2013)
            let mut gp_pen = 0.0;
            for (m, (&qa, &fa)) in w_aa.iter().zip(frag.aa.iter()).enumerate() {
                let q_gp = qa == AA_GLY || qa == AA_PRO;
                let f_gp = fa == AA_GLY || fa == AA_PRO;
                if q_gp != f_gp {
                    gp_pen += if m == 1 { 6.0 } else { 3.5 };
                } else if q_gp && qa != fa {
                    gp_pen += 6.0;
                }
            }
            if gp_pen > 7.0 {
                continue;
            }
            let b0 = BLOSUM62[w_aa[0] as usize][frag.aa[0] as usize] as f64;
            let b1 = BLOSUM62[w_aa[1] as usize][frag.aa[1] as usize] as f64;
            let b2 = BLOSUM62[w_aa[2] as usize][frag.aa[2] as usize] as f64;
            let seq_sim = 0.65 * b0 + 1.15 * b1 + 0.65 * b2 - gp_pen;
            if seq_sim >= 2.5 {
                top_frags.push((seq_sim, fi));
            }
        }
        top_frags.sort_by(|a, b| b.0.total_cmp(&a.0));
        top_frags.truncate(MAX_MODES - 8);

        if !top_frags.is_empty() {
            let max_sim = top_frags[0].0;
            let mut frag_weights = Vec::with_capacity(top_frags.len());
            let mut sum_fw = 0.0;
            for &(sim, fi) in &top_frags {
                let fw = (0.32 * (sim - max_sim)).exp();
                sum_fw += fw;
                frag_weights.push((fw, fi));
            }
            // Empirical fragment pool receives up to 0.42 total prior mass when strong sequence matches exist
            let emp_mass = (0.42 * (max_sim / 9.0).clamp(0.25, 1.0)) / sum_fw.max(1e-9);
            for (fw, fi) in frag_weights {
                let frag = &db.fragments[fi];
                let mut prof = [[(0.0, 1.0); 5]; 3];
                for m in 0..3 {
                    let fallback = [
                        basin_atom_profile(w_aa[m], frag.basin, 0),
                        basin_atom_profile(w_aa[m], frag.basin, 1),
                        basin_atom_profile(w_aa[m], frag.basin, 2),
                        basin_atom_profile(w_aa[m], frag.basin, 3),
                        basin_atom_profile(w_aa[m], frag.basin, 4),
                    ];
                    let f_vals = [frag.dca[m], frag.dcb[m], frag.dc[m], frag.dn[m], frag.dh[m]];
                    let f_sds = [1.00, 1.10, 1.00, 2.55, 0.48];
                    for ai in 0..5 {
                        prof[m][ai] = match f_vals[ai] {
                            Some(v) => (v as f64, f_sds[ai]),
                            None => fallback[ai],
                        };
                    }
                }
                raw_modes.push((fw * emp_mass, prof));
            }
        }

        let total_w: f64 = raw_modes.iter().map(|x| x.0).sum();
        let modes: Vec<TripeptideMode> = raw_modes
            .iter()
            .map(|&(w, prof)| TripeptideMode {
                log_weight: (w / total_w).ln(),
                prof,
            })
            .collect();

        // Compute marginal `(mean, sd)` per `(m, atom)` under the same mixture so log-Bayes-factors
        // measure pure multi-atom & multi-residue cooperativity without double-counting single-atom priors!
        let mut marg = [[(0.0, 1.0); 5]; 3];
        for m in 0..3 {
            for ai in 0..5 {
                let mut mu = 0.0;
                for &(w, ref prof) in &raw_modes {
                    mu += (w / total_w) * prof[m][ai].0;
                }
                let mut var = 0.0;
                for &(w, ref prof) in &raw_modes {
                    let (pmu, psd) = prof[m][ai];
                    let diff = pmu - mu;
                    var += (w / total_w) * (psd * psd + diff * diff);
                }
                marg[m][ai] = (mu, var.sqrt().max(0.35));
            }
        }

        out.push(WindowModel { modes, marg });
    }
    out
}

/// Atom weights `[CA, CB, C, N, H]` in Shen–Baker–Bax secondary-shift chi-squared scoring:
/// backbone carbons carry primary `(phi, psi)` information; `N`/`H` carry softer H-bond/ring-current weight.
const SEC_ATOM_WEIGHTS: [f64; 5] = [1.0, 0.90, 0.90, 0.30, 0.30];

#[inline]
fn eval_window_coop_log_bf(
    wm: &WindowModel,
    obs: &[[Option<f64>; 5]; 3],
    pos_weights: [f64; 3],
    coop_mix: f64,
) -> f64 {
    let mut n_terms = 0usize;
    let mut indep_ll = 0.0;
    for m in 0..3 {
        if pos_weights[m] <= 0.0 {
            continue;
        }
        for ai in 0..5 {
            if let Some(dv) = obs[m][ai] {
                let w = pos_weights[m] * SEC_ATOM_WEIGHTS[ai];
                let (mmu, msd) = wm.marg[m][ai];
                indep_ll += w * logpdf(dv, mmu, msd);
                n_terms += 1;
            }
        }
    }
    if n_terms < 2 {
        return 0.0;
    }

    let mut mode_lls = [0.0f64; MAX_MODES];
    let mut max_ll = f64::NEG_INFINITY;
    for (ci, mode) in wm.modes.iter().enumerate() {
        let mut ll = mode.log_weight;
        for m in 0..3 {
            if pos_weights[m] <= 0.0 {
                continue;
            }
            for ai in 0..5 {
                if let Some(dv) = obs[m][ai] {
                    let w = pos_weights[m] * SEC_ATOM_WEIGHTS[ai];
                    let (pmu, psd) = mode.prof[m][ai];
                    ll += w * logpdf(dv, pmu, psd);
                }
            }
        }
        mode_lls[ci] = ll;
        if ll > max_ll {
            max_ll = ll;
        }
    }
    let mut sum_exp = 0.0;
    for &ll in &mode_lls[..wm.modes.len()] {
        sum_exp += (ll - max_ll).exp();
    }
    let coop_ll = max_ll + sum_exp.ln();
    let log_ratio = (coop_ll - indep_ll).clamp(-8.0, 8.0);
    let mix = coop_mix.clamp(0.0, 0.95);
    ((1.0 - mix) + mix * log_ratio.exp()).ln()
}

// ============================================================================
// Position-Specific Unary & Tripeptide Pair Scoring
// ============================================================================

#[inline]
fn neighbor_adjusted_llr(
    priors: &Priors,
    res: &str,
    ctx_aa: [u8; 3],
    atom: &str,
    v: f64,
    nb_weight: f64,
) -> f64 {
    if !priors.table.contains_key(res) {
        return 0.0;
    }
    if !priors.has(res, atom) {
        return 0.02_f64.ln();
    }
    let (mu, sd) = priors.table[res][atom];
    let lp_base = logpdf(v, mu, sd);
    let lp_mix = if nb_weight > 0.0 {
        let ai = atom_index(atom).unwrap_or(0);
        let delta_nb = zero_mean_neighbor_delta(ctx_aa[0], ctx_aa[1], ctx_aa[2], ai);
        let lp_nb = logpdf(v, mu + delta_nb, sd);
        let m = lp_base.max(lp_nb);
        m + ((1.0 - nb_weight) * (lp_base - m).exp() + nb_weight * (lp_nb - m).exp()).ln()
    } else {
        lp_base
    };
    (lp_mix - priors.background(atom, v)).max(-12.0)
}

#[derive(Clone, Debug, Default)]
struct SlotCandCache {
    /// Per `[off_idx: 0 => off=0, 1 => off=-1][atom_idx: 0..3 (CA, CB, C)]`:
    /// `(best_ppm, best_cand_score, per_cand_scores_without_ln_n)`
    best_ppm: [[Option<f64>; 3]; 2],
    best_score: [[f64; 3]; 2],
    cand_scores: [[Vec<f64>; 3]; 2],
    /// Dipeptide `(k-1, k)` bonus already included in `U_tri(s, k)` so `P_tri` avoids double-counting.
    dip_bonus: f64,
}

/// Position-specific pair score tensor `P_tri(s, t, k)` for placing `s` at `k` and `t` at `k + 1`.
#[derive(Clone, Debug)]
pub struct TripeptidePairTensor {
    pub s_len: usize,
    pub k_len: usize,
    active_map: Vec<i32>,
    n_active: usize,
    /// Dense `(k_len - 1) * n_active * n_active` table (`f32` saves 2x L3 cache bandwidth in SA!).
    dense: Vec<f32>,
}

impl TripeptidePairTensor {
    #[inline]
    pub fn get(&self, s: usize, t: usize, k: usize) -> f64 {
        if s == t || k + 1 >= self.k_len {
            return FORBID;
        }
        let ia = self.active_map[s];
        let ib = self.active_map[t];
        if ia < 0 || ib < 0 {
            return FORBID;
        }
        let idx = (k * self.n_active + (ia as usize)) * self.n_active + (ib as usize);
        self.dense[idx] as f64
    }
}

/// Stateful tripeptide fragment & cooperative secondary-shift solver.
#[derive(Clone, Debug, Default)]
pub struct TripeptideSolver {
    pub cfg: TripeptideConfig,
    pub db: TripeptideFragmentDb,
}

impl TripeptideSolver {
    pub fn new(cfg: TripeptideConfig) -> Self {
        Self {
            cfg,
            db: TripeptideFragmentDb::default(),
        }
    }

    pub fn with_db(cfg: TripeptideConfig, db: TripeptideFragmentDb) -> Self {
        Self { cfg, db }
    }

    /// Evaluate observed secondary shifts `[m in -1..=1][CA, CB, C, N, H]` at sequence position `k`.
    pub fn score_window_secondary_shifts(
        &self,
        sequence: &[String],
        k: usize,
        sec_shifts: &[[Option<f64>; 5]; 3],
    ) -> f64 {
        if sequence.is_empty() || k >= sequence.len() {
            return 0.0;
        }
        let seq_u8: Vec<u8> = sequence.iter().map(|r| aa_index(r)).collect();
        let mask = if self.cfg.exclude_homologs {
            self.db
                .non_homologous_mask(sequence, self.cfg.max_homolog_identity)
        } else {
            vec![true; self.db.sources.len()]
        };
        let models = build_window_models(&seq_u8, &self.db, &mask);
        eval_window_coop_log_bf(&models[k], sec_shifts, [0.75, 1.0, 0.75], self.cfg.coop_mix)
    }

    /// Build `(U_tri, P_tri, estimated_reference_offsets)` for `systems` and `sequence`.
    pub fn score_tables(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
    ) -> (Matrix2D, TripeptidePairTensor, ReferenceOffsets) {
        let s_len = systems.len();
        let k_len = sequence.len();
        let offsets = if self.cfg.auto_reference {
            estimate_reference_offsets(systems, sequence, priors)
        } else {
            ReferenceOffsets::zero()
        };
        let shifted_priors = offsets.apply_to_priors(priors);
        let seq_u8: Vec<u8> = sequence.iter().map(|r| aa_index(r)).collect();

        let allowed_sources = if self.cfg.exclude_homologs {
            self.db
                .non_homologous_mask(sequence, self.cfg.max_homolog_identity)
        } else {
            vec![true; self.db.sources.len()]
        };
        let win_models = build_window_models(&seq_u8, &self.db, &allowed_sources);

        let assignable = sequence
            .iter()
            .enumerate()
            .filter(|&(k, r)| k > 0 && !r.eq_ignore_ascii_case("PRO"))
            .count()
            .max(1);
        let carb_roots = systems
            .iter()
            .filter(|ss| ss.has_observed_carbons())
            .count();
        let heavy_spurious_regime =
            self.cfg.spurious_root_guard && carb_roots > assignable * 2 + 20;
        let auto_spurious_cost = if heavy_spurious_regime && self.cfg.score.assign_cost == 0.0 {
            0.55 * ((carb_roots as f64) / (assignable as f64)).ln()
        } else {
            0.0
        };

        // Helper to compute secondary shift at sequence index `r_idx`
        let sec_at =
            |r_idx: usize, atom_i: usize, obs_ppm: f64, cb_hint: Option<f64>| -> Option<f64> {
                let prev = if r_idx > 0 { seq_u8[r_idx - 1] } else { AA_UNK };
                let curr = seq_u8[r_idx];
                let next = if r_idx + 1 < k_len {
                    seq_u8[r_idx + 1]
                } else {
                    AA_UNK
                };
                let off_ppm = match atom_i {
                    0 => offsets.dca,
                    1 => offsets.dcb,
                    2 => offsets.dc,
                    3 => offsets.dn,
                    _ => offsets.dh,
                };
                let cys_ox =
                    curr == AA_CYS && cb_hint.map(|v| v - offsets.dcb).unwrap_or(28.0) > 35.0;
                let rc = rc_shift_indexed(prev, curr, next, atom_i, cys_ox)?;
                Some((obs_ppm - off_ppm) - rc)
            };

        // Detect spin systems that overlap another HSQC root within the 2D (H, N) attachment cylinder:
        // on overlapped roots, `mean_dist` candidate ordering reflects 2D peak jitter rather than true ownership.
        let overlapped_root: Vec<bool> = (0..s_len)
            .map(|i| {
                (0..s_len).any(|j| {
                    i != j
                        && ((systems[i].h - systems[j].h) / 0.03)
                            .hypot((systems[i].n - systems[j].n) / 0.30)
                            <= 1.0
                })
            })
            .collect();

        let mut u = Matrix2D::filled(s_len, k_len, FORBID);
        let mut cache: Vec<Vec<SlotCandCache>> = vec![vec![SlotCandCache::default(); k_len]; s_len];

        for k in 1..k_len {
            let res = &sequence[k];
            if res == "PRO" {
                continue;
            }
            let prev_res = &sequence[k - 1];
            let k_prev_aa = seq_u8[k - 1];
            let k_curr_aa = seq_u8[k];
            let k_next_aa = if k + 1 < k_len { seq_u8[k + 1] } else { AA_UNK };
            let km1_prev_aa = if k >= 2 { seq_u8[k - 2] } else { AA_UNK };

            for (s, ss) in systems.iter().enumerate() {
                if self.cfg.score.forbid_zero_carbons && !ss.has_observed_carbons() {
                    continue;
                }
                if heavy_spurious_regime {
                    if !(5.2..=11.8).contains(&ss.h) {
                        continue;
                    }
                    let has_any_ca = ss
                        .cands
                        .get(&("CA".to_string(), 0))
                        .is_some_and(|l| l.iter().any(|c| c.1 > 0))
                        || ss
                            .cands
                            .get(&("CA".to_string(), -1))
                            .is_some_and(|l| l.iter().any(|c| c.1 > 0));
                    if !has_any_ca && res != "GLY" && prev_res != "PRO" {
                        continue;
                    }
                }

                let mut sc = 0.0;
                let mut slot_c = SlotCandCache::default();

                for (ai, atom) in ["CA", "CB", "C"].iter().enumerate() {
                    for (oi, off) in [0i32, -1i32].iter().enumerate() {
                        let Some(raw_cands) = ss.cands.get(&(atom.to_string(), *off)) else {
                            continue;
                        };
                        let mut c_scores = Vec::with_capacity(raw_cands.len());
                        let mut n_pos = 0usize;
                        let mut best = f64::NEG_INFINITY;
                        let mut best_ppm = None;

                        for (rank_i, &(v, sup)) in raw_cands.iter().enumerate() {
                            if sup == 0 {
                                c_scores.push(0.0);
                                continue;
                            }
                            n_pos += 1;
                            let llr = if *off == 0 {
                                neighbor_adjusted_llr(
                                    &shifted_priors,
                                    res,
                                    [k_prev_aa, k_curr_aa, k_next_aa],
                                    atom,
                                    v,
                                    self.cfg.neighbor_rc_weight,
                                )
                            } else {
                                neighbor_adjusted_llr(
                                    &shifted_priors,
                                    prev_res,
                                    [km1_prev_aa, k_prev_aa, k_curr_aa],
                                    atom,
                                    v,
                                    self.cfg.neighbor_rc_weight,
                                )
                            };
                            let rank_pen = if overlapped_root[s] {
                                0.0
                            } else {
                                self.cfg.cand_rank_penalty * (rank_i as f64)
                            };
                            let c_sc = llr + self.cfg.score.support_bonus * ((sup as f64) - 1.0)
                                - rank_pen;
                            c_scores.push(c_sc);
                            if c_sc > best {
                                best = c_sc;
                                best_ppm = Some(v);
                            }
                        }
                        if n_pos > 0 && best.is_finite() {
                            sc += best - (n_pos as f64).ln();
                            slot_c.best_ppm[oi][ai] = best_ppm;
                            slot_c.best_score[oi][ai] = best;
                        }
                        slot_c.cand_scores[oi][ai] = c_scores;
                    }
                }

                if self.cfg.score.use_hn {
                    sc += neighbor_adjusted_llr(
                        &shifted_priors,
                        res,
                        [k_prev_aa, k_curr_aa, k_next_aa],
                        "N",
                        ss.n,
                        self.cfg.neighbor_rc_weight,
                    );
                    sc += neighbor_adjusted_llr(
                        &shifted_priors,
                        res,
                        [k_prev_aa, k_curr_aa, k_next_aa],
                        "H",
                        ss.h,
                        self.cfg.neighbor_rc_weight,
                    );
                }

                // Dipeptide `(k-1, k)` secondary-shift coherence within `s`
                if self.cfg.dipeptide_weight > 0.0 {
                    let mut obs_dip = [[None; 5]; 3];
                    let cb_km1 = slot_c.best_ppm[1][1];
                    let cb_k = slot_c.best_ppm[0][1];
                    for (ai, (&v_km1, &v_k)) in slot_c.best_ppm[1]
                        .iter()
                        .zip(slot_c.best_ppm[0].iter())
                        .enumerate()
                    {
                        if let Some(v) = v_km1 {
                            obs_dip[0][ai] = sec_at(k - 1, ai, v, cb_km1);
                        }
                        if let Some(v) = v_k {
                            obs_dip[1][ai] = sec_at(k, ai, v, cb_k);
                        }
                    }
                    if self.cfg.score.use_hn {
                        obs_dip[1][3] = sec_at(k, 3, ss.n, cb_k);
                        obs_dip[1][4] = sec_at(k, 4, ss.h, cb_k);
                    }
                    let dip_bf = eval_window_coop_log_bf(
                        &win_models[k],
                        &obs_dip,
                        [0.75, 1.0, 0.0],
                        self.cfg.coop_mix,
                    );
                    let dip_term = self.cfg.dipeptide_weight * dip_bf;
                    sc += dip_term;
                    slot_c.dip_bonus = dip_bf;
                }

                sc -= self.cfg.score.assign_cost + auto_spurious_cost;
                u.set(s, k, sc);
                cache[s][k] = slot_c;
            }
        }

        // Map spin systems with >= 1 valid sequence slot (and cap to top active candidates if S is huge)
        let mut active_list: Vec<(usize, f64)> = (0..s_len)
            .filter_map(|s| {
                let best_u = u.row(s).iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (best_u > FORBID * 0.5).then_some((s, best_u))
            })
            .collect();
        let max_active = (assignable * 3 + 48).max(200);
        if active_list.len() > max_active {
            active_list.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(dropped_s, _) in &active_list[max_active..] {
                for k in 0..k_len {
                    u.set(dropped_s, k, FORBID);
                }
            }
            active_list.truncate(max_active);
            active_list.sort_by_key(|x| x.0);
        }

        let mut active_map = vec![-1i32; s_len];
        for (compact_i, &(orig_s, _)) in active_list.iter().enumerate() {
            active_map[orig_s] = compact_i as i32;
        }
        let n_active = active_list.len();
        let n_edges = k_len.saturating_sub(1);
        let mut dense = vec![FORBID as f32; n_edges * n_active * n_active];

        let max_n = systems
            .iter()
            .flat_map(|ss| ss.cands.values().map(|c| c.len()))
            .max()
            .unwrap_or(1)
            .max(1);
        let log_range = self.cfg.score.range_ppm.ln();
        let sigs = [
            self.cfg.score.sigma_ca,
            self.cfg.score.sigma_cb,
            self.cfg.score.sigma_c,
        ];
        let const_terms = [
            -(sigs[0] * TAU.sqrt()).ln() + log_range,
            -(sigs[1] * TAU.sqrt()).ln() + log_range,
            -(sigs[2] * TAU.sqrt()).ln() + log_range,
        ];
        let keys_i = [
            ("CA".to_string(), 0),
            ("CB".to_string(), 0),
            ("C".to_string(), 0),
        ];
        let keys_im1 = [
            ("CA".to_string(), -1),
            ("CB".to_string(), -1),
            ("C".to_string(), -1),
        ];

        // Detect whether multiple intra-residue CA experiments (e.g. HNCA + HNCACB) were acquired:
        // when dual CA(i) support exists, genuine degenerate i/i-1 CA clusters carry support >= 3.
        let dual_ca_i_count = systems
            .iter()
            .filter(|ss| {
                ss.cands
                    .get(&("CA".to_string(), 0))
                    .is_some_and(|l| l.iter().any(|&(_, sup)| sup >= 2))
            })
            .count();
        let has_dual_ca_i = dual_ca_i_count >= assignable / 4;

        // Build position-specific `P_tri(s, t, k)` across each edge `(k, k + 1)`
        for k in 1..n_edges {
            if sequence[k] == "PRO" || sequence[k + 1] == "PRO" {
                continue;
            }
            let k_has_cb = seq_u8[k] != AA_GLY;

            for (ia, &(s, _)) in active_list.iter().enumerate() {
                if u.get(s, k) <= FORBID * 0.5 {
                    continue;
                }
                let ss_s = &systems[s];
                let cache_sk = &cache[s][k];

                for (ib, &(t, _)) in active_list.iter().enumerate() {
                    if s == t || u.get(t, k + 1) <= FORBID * 0.5 {
                        continue;
                    }
                    let ss_t = &systems[t];
                    let cache_tk1 = &cache[t][k + 1];

                    let mut p_val = 0.0;
                    let mut cons_k_ppm = [None::<f64>; 3];

                    for ai in 0..3 {
                        let a_opt = ss_s.cands.get(&keys_i[ai]);
                        let b_opt = ss_t.cands.get(&keys_im1[ai]);

                        // Soft degenerate fallback when `s` has no `i` candidate, `t` has `i-1`,
                        // and `s(i-1)` has support >= 2 (coincident `i + i-1` peak when `degenerate=false`).
                        let mut soft_deg_buf: [(f64, usize); 1] = [(0.0, 0)];
                        let (a_slice, is_soft_deg): (&[(f64, usize)], bool) = match a_opt {
                            Some(v) if !v.is_empty() => (&v[..v.len().min(max_n)], false),
                            _ if self.cfg.allow_soft_degenerate && (ai != 1 || k_has_cb) => {
                                let req_im1_sup = if ai == 0 && has_dual_ca_i { 3 } else { 2 };
                                if let Some(im1_list) = ss_s.cands.get(&keys_im1[ai]) {
                                    if let Some(&(v_im1, sup_im1)) = im1_list.first() {
                                        if sup_im1 >= req_im1_sup {
                                            soft_deg_buf[0] = (v_im1, 0);
                                            (&soft_deg_buf[..], true)
                                        } else {
                                            (&[], false)
                                        }
                                    } else {
                                        (&[], false)
                                    }
                                } else {
                                    (&[], false)
                                }
                            }
                            _ => (&[], false),
                        };

                        let b_slice: &[(f64, usize)] = match b_opt {
                            Some(v) if !v.is_empty() => &v[..v.len().min(max_n)],
                            _ => &[],
                        };

                        if !a_slice.is_empty() && !b_slice.is_empty() {
                            let sig = sigs[ai];
                            let ct = const_terms[ai];
                            let mut best_pair = f64::NEG_INFINITY;
                            let mut best_consensus = 0.5 * (a_slice[0].0 + b_slice[0].0);

                            for (ca_i, &(va, sup_a)) in a_slice.iter().enumerate() {
                                let cons_a =
                                    if sup_a > 0 && ca_i < cache_sk.cand_scores[0][ai].len() {
                                        0.20 * (cache_sk.cand_scores[0][ai][ca_i]
                                            - cache_sk.best_score[0][ai])
                                            .max(-3.0)
                                    } else if is_soft_deg {
                                        -2.10
                                    } else {
                                        0.0
                                    };

                                for (cb_i, &(vb, sup_b)) in b_slice.iter().enumerate() {
                                    let cons_b =
                                        if sup_b > 0 && cb_i < cache_tk1.cand_scores[1][ai].len() {
                                            0.20 * (cache_tk1.cand_scores[1][ai][cb_i]
                                                - cache_tk1.best_score[1][ai])
                                                .max(-3.0)
                                        } else {
                                            0.0
                                        };
                                    let d = (va - vb) / sig;
                                    let seq_term =
                                        (-0.5 * d * d + ct).max(self.cfg.score.pair_floor);
                                    let joint = seq_term + cons_a + cons_b;
                                    if joint > best_pair {
                                        best_pair = joint;
                                        best_consensus = 0.5 * (va + vb);
                                    }
                                }
                            }

                            if best_pair.is_finite() {
                                let mult_pen = ((a_slice.len() * b_slice.len()) as f64).ln();
                                let net = best_pair - mult_pen;
                                // Soft degenerate hypothesis is only adopted when it genuinely matches `t(i-1)`
                                if !is_soft_deg || net > 0.0 {
                                    p_val += net;
                                    cons_k_ppm[ai] = Some(best_consensus);
                                }
                            }
                        } else {
                            // Even when sequential link is missing on one side (e.g. missing HNCACO C(i)),
                            // `t(i-1)` (from HNCO!) or `s(i)` still observes residue `k` for tripeptide scoring!
                            cons_k_ppm[ai] = cache_sk.best_ppm[0][ai].or(cache_tk1.best_ppm[1][ai]);
                        }
                    }

                    // Shen–Baker–Bax cooperative tripeptide `(k-1, k, k+1)` secondary-shift score
                    if self.cfg.tripeptide_weight > 0.0 {
                        let mut obs_tri = [[None; 5]; 3];
                        let cb_km1 = cache_sk.best_ppm[1][1];
                        let cb_k = cons_k_ppm[1];
                        let cb_kp1 = cache_tk1.best_ppm[0][1];

                        for ai in 0..3 {
                            if let Some(v) = cache_sk.best_ppm[1][ai] {
                                obs_tri[0][ai] = sec_at(k - 1, ai, v, cb_km1);
                            }
                            if let Some(v) = cons_k_ppm[ai] {
                                obs_tri[1][ai] = sec_at(k, ai, v, cb_k);
                            }
                            if let Some(v) = cache_tk1.best_ppm[0][ai] {
                                obs_tri[2][ai] = sec_at(k + 1, ai, v, cb_kp1);
                            }
                        }
                        if self.cfg.score.use_hn {
                            obs_tri[1][3] = sec_at(k, 3, ss_s.n, cb_k);
                            obs_tri[1][4] = sec_at(k, 4, ss_s.h, cb_k);
                            obs_tri[2][3] = sec_at(k + 1, 3, ss_t.n, cb_kp1);
                            obs_tri[2][4] = sec_at(k + 1, 4, ss_t.h, cb_kp1);
                        }

                        let tri_bf = eval_window_coop_log_bf(
                            &win_models[k],
                            &obs_tri,
                            [0.75, 1.0, 0.80],
                            self.cfg.coop_mix,
                        );
                        // Subtract overlapping dipeptide bonus on `(k-1, k)` only when `k+1` adds heavy-atom data
                        let kp1_heavy = usize::from(obs_tri[2][0].is_some())
                            + usize::from(obs_tri[2][1].is_some())
                            + usize::from(obs_tri[2][2].is_some());
                        let incremental_coop = if kp1_heavy > 0 {
                            self.cfg.tripeptide_weight * tri_bf
                                - self.cfg.dipeptide_weight * cache_sk.dip_bonus
                        } else {
                            0.0
                        };
                        p_val += incremental_coop;
                    }

                    let idx = (k * n_active + ia) * n_active + ib;
                    dense[idx] = p_val as f32;
                }
            }
        }

        let tensor = TripeptidePairTensor {
            s_len,
            k_len,
            active_map,
            n_active,
            dense,
        };
        (u, tensor, offsets)
    }

    pub fn assign(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
    ) -> AssignResult {
        if systems.is_empty() || sequence.is_empty() {
            return AssignResult {
                position: vec![-1; systems.len()],
                confidence: vec![0.0; systems.len()],
                score: 0.0,
                run_scores: Vec::new(),
            };
        }
        let (u, p_tri, _) = self.score_tables(systems, sequence, priors);
        let nav = NeighborTables::build(&u, &p_tri);
        let acfg = &self.cfg.anneal;
        let mut master_rng = Rng::new(acfg.seed);
        let steps = acfg.steps_per_system * u.rows.max(sequence.len());
        let seeds: Vec<(usize, u64)> = (0..acfg.runs).map(|r| (r, master_rng.next_u64())).collect();

        let n_threads = if cfg!(target_arch = "wasm32") {
            1
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(acfg.runs.max(1))
        };

        let run_one = |rep_idx: usize, sub_seed: u64| -> (Vec<i32>, f64) {
            let mut ann = TripeptideAnnealer::new(&u, &p_tri, &nav, Rng::new(sub_seed));
            if self.cfg.seed_growth && rep_idx.is_multiple_of(4) {
                ann.seed_from_tripeptide_fragments();
            }
            let sc = ann.run(steps, acfg.t_start, acfg.t_end, acfg.max_block);
            (ann.pos, sc)
        };

        if n_threads <= 1 || acfg.runs <= 1 {
            let mut positions = Vec::with_capacity(acfg.runs);
            let mut scores = Vec::with_capacity(acfg.runs);
            for (rep_idx, sub_seed) in seeds {
                let (pos, sc) = run_one(rep_idx, sub_seed);
                positions.push(pos);
                scores.push(sc);
            }
            return consensus_from_runs(&positions, scores);
        }

        let chunk_len = acfg.runs.div_ceil(n_threads);
        let mut chunk_results: Vec<Vec<(Vec<i32>, f64)>> = Vec::with_capacity(n_threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for seed_chunk in seeds.chunks(chunk_len) {
                handles.push(scope.spawn(move || {
                    let mut local = Vec::with_capacity(seed_chunk.len());
                    for &(rep_idx, sub_seed) in seed_chunk {
                        local.push(run_one(rep_idx, sub_seed));
                    }
                    local
                }));
            }
            for h in handles {
                if let Ok(part) = h.join() {
                    chunk_results.push(part);
                }
            }
        });

        let mut positions = Vec::with_capacity(acfg.runs);
        let mut scores = Vec::with_capacity(acfg.runs);
        for part in chunk_results {
            for (pos, sc) in part {
                positions.push(pos);
                scores.push(sc);
            }
        }
        consensus_from_runs(&positions, scores)
    }
}

/// Top sequential partners per `(k, s)` for chain-growth seeding and guided annealing proposals.
struct NeighborTables {
    /// `next_partners[k][ia]` -> up to 6 spin-system indices `t` maximizing `P_tri(s, t, k) + U_tri(t, k+1)`
    next_partners: Vec<Vec<Vec<usize>>>,
    /// `prev_partners[k][ib]` -> up to 6 spin-system indices `s` maximizing `P_tri(s, t, k-1) + U_tri(s, k-1)`
    prev_partners: Vec<Vec<Vec<usize>>>,
    /// High-margin anchor edges `(edge_score, margin, k, s, t)`
    seed_edges: Vec<(f64, f64, usize, usize, usize)>,
}

impl NeighborTables {
    fn build(u: &Matrix2D, p: &TripeptidePairTensor) -> Self {
        let k_len = u.cols;
        let n_act = p.n_active;
        let mut orig_of_active = vec![0usize; n_act];
        for (s, &ia) in p.active_map.iter().enumerate() {
            if ia >= 0 {
                orig_of_active[ia as usize] = s;
            }
        }

        let mut next_partners = vec![vec![Vec::new(); n_act]; k_len];
        let mut prev_partners = vec![vec![Vec::new(); n_act]; k_len];
        let mut seed_edges = Vec::new();

        for k in 1..k_len.saturating_sub(1) {
            let mut edge_cands: Vec<(f64, usize, usize)> = Vec::new();
            for (ia, &s) in orig_of_active.iter().enumerate() {
                let usk = u.get(s, k);
                if usk <= FORBID * 0.5 {
                    continue;
                }
                let mut row_best: Vec<(f64, usize)> = Vec::new();
                for &t in &orig_of_active {
                    if s == t {
                        continue;
                    }
                    let utk1 = u.get(t, k + 1);
                    if utk1 <= FORBID * 0.5 {
                        continue;
                    }
                    let pst = p.get(s, t, k);
                    if pst > -2.5 {
                        row_best.push((pst + 0.5 * utk1, t));
                        edge_cands.push((usk + utk1 + pst, s, t));
                    }
                }
                row_best.sort_by(|a, b| b.0.total_cmp(&a.0));
                row_best.truncate(6);
                next_partners[k][ia] = row_best.into_iter().map(|x| x.1).collect();
            }

            for (ib, &t) in orig_of_active.iter().enumerate() {
                let utk1 = u.get(t, k + 1);
                if utk1 <= FORBID * 0.5 {
                    continue;
                }
                let mut col_best: Vec<(f64, usize)> = Vec::new();
                for &s in &orig_of_active {
                    if s == t {
                        continue;
                    }
                    let usk = u.get(s, k);
                    if usk <= FORBID * 0.5 {
                        continue;
                    }
                    let pst = p.get(s, t, k);
                    if pst > -2.5 {
                        col_best.push((pst + 0.5 * usk, s));
                    }
                }
                col_best.sort_by(|a, b| b.0.total_cmp(&a.0));
                col_best.truncate(6);
                prev_partners[k + 1][ib] = col_best.into_iter().map(|x| x.1).collect();
            }

            if !edge_cands.is_empty() {
                edge_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
                let best_sc = edge_cands[0].0;
                let second_sc = edge_cands.get(1).map(|x| x.0).unwrap_or(0.0);
                let margin = best_sc - second_sc;
                if best_sc > 8.0 && margin > 2.0 {
                    seed_edges.push((best_sc, margin, k, edge_cands[0].1, edge_cands[0].2));
                }
            }
        }
        seed_edges.sort_by(|a, b| b.1.total_cmp(&a.1));
        Self {
            next_partners,
            prev_partners,
            seed_edges,
        }
    }
}

struct TripeptideAnnealer<'a> {
    u: &'a Matrix2D,
    p: &'a TripeptidePairTensor,
    nav: &'a NeighborTables,
    rng: Rng,
    s_len: usize,
    k_len: usize,
    slots: Vec<Vec<usize>>,
    active_systems: Vec<usize>,
    at: Vec<i32>,
    pos: Vec<i32>,
    edge_marks: Vec<u32>,
    mark_gen: u32,
}

impl<'a> TripeptideAnnealer<'a> {
    fn new(
        u: &'a Matrix2D,
        p: &'a TripeptidePairTensor,
        nav: &'a NeighborTables,
        rng: Rng,
    ) -> Self {
        let s_len = u.rows;
        let k_len = u.cols;
        let slots: Vec<Vec<usize>> = (0..s_len)
            .map(|s| (0..k_len).filter(|&k| u.get(s, k) > FORBID * 0.5).collect())
            .collect();
        let active_systems: Vec<usize> = (0..s_len).filter(|&s| !slots[s].is_empty()).collect();
        Self {
            u,
            p,
            nav,
            rng,
            s_len,
            k_len,
            slots,
            active_systems,
            at: vec![-1; k_len],
            pos: vec![-1; s_len],
            edge_marks: vec![0; k_len],
            mark_gen: 1,
        }
    }

    /// Stochastic high-margin tripeptide seed & bidirectional fragment chain growth.
    fn seed_from_tripeptide_fragments(&mut self) {
        for &(sc, _margin, k, s, t) in &self.nav.seed_edges {
            if sc < 10.0 || self.rng.f64() < 0.25 {
                continue;
            }
            if self.at[k] >= 0 || self.at[k + 1] >= 0 || self.pos[s] >= 0 || self.pos[t] >= 0 {
                continue;
            }
            self.at[k] = s as i32;
            self.pos[s] = k as i32;
            self.at[k + 1] = t as i32;
            self.pos[t] = (k + 1) as i32;

            // Bidirectional greedy chain extension when the partner link is strong (`P_tri > 3.5`)
            let mut cur_k = k + 1;
            while cur_k + 1 < self.k_len && self.at[cur_k + 1] < 0 {
                let cur_s = self.at[cur_k] as usize;
                let ia = self.p.active_map[cur_s];
                if ia < 0 {
                    break;
                }
                let cands = &self.nav.next_partners[cur_k][ia as usize];
                let Some(&nxt_s) = cands.first() else {
                    break;
                };
                if self.pos[nxt_s] >= 0
                    || self.p.get(cur_s, nxt_s, cur_k) < 3.5
                    || self.u.get(nxt_s, cur_k + 1) < 0.0
                {
                    break;
                }
                self.at[cur_k + 1] = nxt_s as i32;
                self.pos[nxt_s] = (cur_k + 1) as i32;
                cur_k += 1;
            }

            let mut back_k = k;
            while back_k >= 2 && self.at[back_k - 1] < 0 {
                let cur_t = self.at[back_k] as usize;
                let ib = self.p.active_map[cur_t];
                if ib < 0 {
                    break;
                }
                let cands = &self.nav.prev_partners[back_k][ib as usize];
                let Some(&prv_s) = cands.first() else {
                    break;
                };
                if self.pos[prv_s] >= 0
                    || self.p.get(prv_s, cur_t, back_k - 1) < 3.5
                    || self.u.get(prv_s, back_k - 1) < 0.0
                {
                    break;
                }
                self.at[back_k - 1] = prv_s as i32;
                self.pos[prv_s] = (back_k - 1) as i32;
                back_k -= 1;
            }
        }
    }

    fn local(&mut self, changed_residues: &[usize]) -> f64 {
        if self.k_len == 0 {
            return 0.0;
        }
        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.edge_marks.fill(0);
            self.mark_gen = 1;
        }
        let gen = self.mark_gen;
        let mut sc = 0.0;
        let mut edges = [0usize; 64];
        let mut n_edges = 0usize;

        for &k in changed_residues {
            let s = self.at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
            if k > 0 && k - 1 < self.k_len - 1 && self.edge_marks[k - 1] != gen {
                self.edge_marks[k - 1] = gen;
                edges[n_edges] = k - 1;
                n_edges += 1;
            }
            if k < self.k_len - 1 && self.edge_marks[k] != gen {
                self.edge_marks[k] = gen;
                edges[n_edges] = k;
                n_edges += 1;
            }
        }
        for &e in &edges[..n_edges] {
            let a = self.at[e];
            let b = self.at[e + 1];
            if a >= 0 && b >= 0 {
                sc += self.p.get(a as usize, b as usize, e);
            }
        }
        sc
    }

    fn total(&self) -> f64 {
        let mut sc = 0.0;
        for k in 0..self.k_len {
            let s = self.at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
        }
        for e in 0..self.k_len.saturating_sub(1) {
            let a = self.at[e];
            let b = self.at[e + 1];
            if a >= 0 && b >= 0 {
                sc += self.p.get(a as usize, b as usize, e);
            }
        }
        sc
    }

    fn apply(&mut self, changes: &[(usize, i32)], undo: &mut Vec<(usize, i32)>) {
        undo.clear();
        for &(k, _) in changes {
            undo.push((k, self.at[k]));
        }
        for &(k, _) in changes {
            let prev_s = self.at[k];
            if prev_s >= 0 {
                self.pos[prev_s as usize] = -1;
            }
        }
        for &(k, s) in changes {
            self.at[k] = s;
        }
        for &(k, s) in changes {
            if s >= 0 {
                self.pos[s as usize] = k as i32;
            }
        }
    }

    fn propose_place(&self, s: usize, k: usize, changes: &mut Vec<(usize, i32)>) {
        let ks = self.pos[s];
        let t = self.at[k];
        if ks == k as i32 {
            changes.push((k, -1));
        } else {
            changes.push((k, s as i32));
            if ks >= 0 {
                let ks_u = ks as usize;
                let repl = if t >= 0 && self.u.get(t as usize, ks_u) > FORBID * 0.5 {
                    t
                } else {
                    -1
                };
                changes.push((ks_u, repl));
            }
        }
    }

    fn run(&mut self, steps: usize, t0: f64, t1: f64, max_block: usize) -> f64 {
        if self.active_systems.is_empty() || self.k_len == 0 {
            return 0.0;
        }
        let mut cur = self.total();
        let mut best = cur;
        let mut best_at = self.at.clone();

        let mut changes: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let mut affected: Vec<usize> = Vec::with_capacity(max_block * 2 + 4);
        let mut undo: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let mut dummy: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 4);
        let denom = (steps.saturating_sub(1)).max(1) as f64;
        let ratio = t1 / t0;
        let n_act = self.active_systems.len();

        for step in 0..steps {
            let temp = t0 * ratio.powf((step as f64) / denom);
            changes.clear();

            let r = self.rng.f64();
            if r < 0.65 {
                // Symmetric single / displace-swap move over active spin systems (preserves detailed balance)
                let s = self.active_systems[self.rng.usize(n_act)];
                let k = self.slots[s][self.rng.usize(self.slots[s].len())];
                self.propose_place(s, k, &mut changes);
            } else if r < 0.95 {
                // Block-swap move
                let max_l = max_block.min(self.k_len / 2);
                if max_l == 0 {
                    continue;
                }
                let l = self.rng.range_usize(1, max_l + 1);
                let k1 = self.rng.usize(self.k_len - l + 1);
                let k2 = self.rng.usize(self.k_len - l + 1);
                if k1.abs_diff(k2) < l {
                    continue;
                }
                let mut valid = true;
                for i in 0..l {
                    let a = self.at[k1 + i];
                    let b = self.at[k2 + i];
                    if (b >= 0 && self.u.get(b as usize, k1 + i) < FORBID * 0.5)
                        || (a >= 0 && self.u.get(a as usize, k2 + i) < FORBID * 0.5)
                    {
                        valid = false;
                        break;
                    }
                    changes.push((k1 + i, b));
                    changes.push((k2 + i, a));
                }
                if !valid {
                    continue;
                }
            } else {
                // Unassign move
                let s = self.active_systems[self.rng.usize(n_act)];
                if self.pos[s] < 0 {
                    continue;
                }
                changes.push((self.pos[s] as usize, -1));
            }

            if changes.is_empty() {
                continue;
            }
            affected.clear();
            for &(k, _) in &changes {
                affected.push(k);
            }
            let before = self.local(&affected);
            self.apply(&changes, &mut undo);
            let delta = self.local(&affected) - before;
            if delta >= 0.0 || self.rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best {
                    best = cur;
                    best_at.copy_from_slice(&self.at);
                }
            } else {
                self.apply(&undo, &mut dummy);
            }
        }

        // Restore best trajectory state (preserving Boltzmann diversity across replicas so ambiguous
        // IDP / overlapped-HSQC placements keep calibrated replica consensus confidence)
        self.at = best_at;
        self.pos.fill(-1);
        for (k, &s) in self.at.iter().enumerate() {
            if s >= 0 {
                self.pos[s as usize] = k as i32;
            }
        }

        let _ = self.s_len;
        best
    }
}

/// Parallel tripeptide `(k-1, k, k+1)` fragment & cooperative secondary-shift assignment entry point,
/// returning an `AssignResult` drop-in compatible with `solver::assign` and `evaluate::score`.
pub fn assign_tripeptide(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &TripeptideConfig,
) -> AssignResult {
    TripeptideSolver::new(cfg.clone()).assign(systems, sequence, priors)
}
