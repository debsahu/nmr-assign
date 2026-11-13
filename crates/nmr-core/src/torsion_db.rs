//! Strictly post-assignment backbone & sidechain `(phi, psi, chi1)` torsion-angle predictor
//! from secondary chemical shifts (`HN, N, CA, CB, C', HA`) and sequence similarity, with
//! unlabelled global chemical-shift offset estimation, Berjanskii-Wishart `RCI` / `S^2`
//! flexibility classification, and the 5-layer `HomologFirewall` (`Q5`).
//!
//! Provenance: Clean-room implementation from peer-reviewed NMR and structural bioinformatics literature:
//! - Shen, Y., Vernon, R., Baker, D. & Bax, A. (2008). "Consistent blind protein structure
//!   generation from NMR chemical shift data." *Proc. Natl. Acad. Sci. U.S.A.* 105(12), 4685-4690.
//! - Shen, Y. & Bax, A. (2013). "Protein backbone and sidechain torsion angles predicted from
//!   NMR chemical shifts using artificial neural networks." *J. Biomol. NMR* 56(3), 227-241.
//! - Berjanskii, M. V. & Wishart, D. S. (2005). "A simple method to predict protein flexibility
//!   using secondary chemical shifts." *J. Am. Chem. Soc.* 127(43), 14970-14971 (`RCI` & `S^2`).
//! - Wang, L. & Wishart, D. S. (2005). "A simple method to check chemically mis-referenced
//!   protein NMR chemical shifts." *J. Biomol. NMR* 31(2), 143-148 (unlabelled shift offset check).
//! - Lovell, S. C., Word, J. M., Richardson, J. S. & Richardson, D. C. (2000). "The penultimate
//!   rotamer library." *Proteins* 40(3), 389-408; and Lovell, S. C. et al. (2003). *Proteins* 50(3),
//!   437-450 (Ramachandran favored/allowed basins and `chi1` rotamers).
//! - Zhang, Y. & Skolnick, J. (2004). "Scoring function for automated assessment of protein
//!   structure template quality." *Proteins* 57(4), 702-710 (`TM-score` with `d_0(L)`).
//! - Henikoff, S. & Henikoff, J. G. (1992). *Proc. Natl. Acad. Sci. U.S.A.* 89(22), 10915-10919
//!   (`BLOSUM62` amino-acid substitution matrix).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::dihedrals::{
    periodic_angle_abs_diff_deg, random_coil_shifts, wrap_deg, DihedralRestraint,
    SecondaryStructure,
};
use crate::nef::{NefDihedralRestraint, NefDihedralRestraintList};
use crate::nmrstar::Reference;
use crate::noesy::{v_add, v_dist_sq, v_sub, Vec3};
use crate::priors::three_to_one;
use crate::structure::kabsch_transform;

const EMBEDDED_TORSION_CORPUS_TSV: &str =
    include_str!("../../../data/torsion_db/torsion_corpus.tsv");

const BLOSUM62_ALPHABET: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";
const BLOSUM62_MATRIX: [[i8; 20]; 20] = [
    [
        4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0,
    ],
    [
        -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3,
    ],
    [
        -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3,
    ],
    [
        -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3,
    ],
    [
        0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,
    ],
    [
        -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2,
    ],
    [
        -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2,
    ],
    [
        0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3,
    ],
    [
        -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3,
    ],
    [
        -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3,
    ],
    [
        -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1,
    ],
    [
        -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2,
    ],
    [
        -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1,
    ],
    [
        -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1,
    ],
    [
        -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2,
    ],
    [
        1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2,
    ],
    [
        0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0,
    ],
    [
        -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3,
    ],
    [
        -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1,
    ],
    [
        0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4,
    ],
];

#[inline]
fn aa_idx(aa1: u8) -> usize {
    BLOSUM62_ALPHABET
        .iter()
        .position(|&c| c == aa1.to_ascii_uppercase())
        .unwrap_or(0)
}

#[inline]
pub fn blosum62_score(a: u8, b: u8) -> i32 {
    BLOSUM62_MATRIX[aa_idx(a)][aa_idx(b)] as i32
}

/// Pure-Rust FIPS 180-4 SHA-256 hex digest (zero external crates).
pub fn sha256_hex(data: &[u8]) -> String {
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
    let mut h: [u32; 8] = [
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

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut hex = String::with_capacity(64);
    for word in h {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

/// Per-residue torsion prediction confidence category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TorsionPredictionClass {
    /// Tight consensus cluster (`>= 7/10` within `40 deg`, circular std `<= 25 deg`) and `RCI S^2 >= 0.65`.
    Good,
    /// Backbone flexibility indicated by Berjanskii-Wishart `RCI S^2 < 0.65`.
    Dynamic,
    /// Structured shifts (`RCI S^2 >= 0.65`), but fragment database matches split across multiple basins.
    Ambiguous,
    /// Terminal residue or fewer than 2 assigned chemical shifts at residue `i`.
    Unassigned,
}

impl TorsionPredictionClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Good => "Good",
            Self::Dynamic => "Dynamic",
            Self::Ambiguous => "Ambiguous",
            Self::Unassigned => "Unassigned",
        }
    }
}

/// Unlabelled global chemical-shift reference offset estimates (ppm, `obs - DSS_expected`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GlobalShiftOffsets {
    pub dh: f64,
    pub dn: f64,
    pub dca: f64,
    pub dcb: f64,
    pub dc: f64,
    pub dha: f64,
}

impl GlobalShiftOffsets {
    pub fn as_array(self) -> [f64; 6] {
        [self.dh, self.dn, self.dca, self.dcb, self.dc, self.dha]
    }
}

/// 5-Layer `HomologFirewall` (`Q5`) query descriptor for strict non-homologous evaluation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HomologFirewallQuery {
    /// PDB IDs of the query protein, mutants, or complexes (`Layer 1`).
    pub query_pdb_ids: Vec<String>,
    /// BMRB IDs of the query protein (`Layer 1`, matches with or without `"bmr"` prefix).
    pub query_bmrb_ids: Vec<String>,
    /// UniProt primary accession (`Layer 1`, e.g. `"P0CG48"`).
    pub query_uniprot_id: Option<String>,
    /// Protein superfamily / family identifier (`Layer 1`).
    pub query_family_id: Option<String>,
    /// ECOD H-group or CATH `C.A.T` topology fold group (`Layer 2`, e.g. `"3.10.20"`).
    pub query_topology_id: Option<String>,
    /// Optional reference `C_alpha` coordinates (`Layer 4`: excludes `TM-score >= 0.50`).
    pub query_ca_coords: Option<Vec<[f64; 3]>>,
}

/// Detailed audit record for a template chain excluded by `HomologFirewall`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcludedTemplateRecord {
    pub chain_key: String,
    pub pdb_id: String,
    pub bmrb_id: String,
    pub reasons: Vec<String>,
}

/// Per-residue `(phi, psi, chi1)` prediction and flexibility summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueTorsionPrediction {
    pub seq_id: i32,
    pub res_name: String,
    pub class: TorsionPredictionClass,
    pub phi_deg: Option<f64>,
    pub phi_std_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub psi_std_deg: Option<f64>,
    pub chi1_deg: Option<f64>,
    pub chi1_std_deg: Option<f64>,
    pub rci: f64,
    pub rci_s2: f64,
    pub consensus_fraction: f64,
    pub ss_class: SecondaryStructure,
}

/// Complete torsion prediction output with `HomologFirewall` SHA-256 audit trail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TorsionPredictionReport {
    pub residues: Vec<ResidueTorsionPrediction>,
    pub shift_offsets: GlobalShiftOffsets,
    pub excluded_template_ids: Vec<String>,
    pub excluded_details: Vec<ExcludedTemplateRecord>,
    pub exclusion_list_sha256: String,
    pub max_retained_seq_identity: f64,
    pub nearest_template_seq_identities: Vec<f64>,
}

impl TorsionPredictionReport {
    /// Convert predicted torsions into `DihedralRestraint` rows for `nmr_core::structure` folding.
    /// Only `Good` residues actively restrain `(phi, psi)`; half-widths are `max(2 * std, 15 deg)`.
    pub fn to_dihedral_restraints(&self) -> Vec<DihedralRestraint> {
        let mut out = Vec::with_capacity(self.residues.len());
        for r in &self.residues {
            let is_good = r.class == TorsionPredictionClass::Good
                && r.phi_deg.is_some()
                && r.psi_deg.is_some();
            let phi_deg = r.phi_deg.unwrap_or(-65.0);
            let psi_deg = r.psi_deg.unwrap_or(130.0);
            let phi_err = r
                .phi_std_deg
                .map(|s| (2.0 * s).clamp(20.0, 35.0))
                .unwrap_or(25.0);
            let psi_err = r
                .psi_std_deg
                .map(|s| (2.0 * s).clamp(20.0, 35.0))
                .unwrap_or(25.0);
            let conf = if is_good {
                (r.consensus_fraction * r.rci_s2).clamp(0.55, 0.99)
            } else {
                0.0
            };
            out.push(DihedralRestraint {
                seq_id: r.seq_id,
                res_name: r.res_name.clone(),
                phi_deg,
                phi_err_deg: phi_err,
                psi_deg,
                psi_err_deg: psi_err,
                ss_class: r.ss_class,
                confidence: conf,
                secondary_shift_score: match r.ss_class {
                    SecondaryStructure::Helix => 1.5,
                    SecondaryStructure::Sheet => -1.5,
                    SecondaryStructure::Coil => 0.0,
                },
                restrained_phi: is_good,
                restrained_psi: is_good,
            });
        }
        out
    }

    /// Export `Good` `(phi, psi, chi1)` torsion restraints to a canonical NEF v1.1 `NefDihedralRestraintList`.
    pub fn to_nef_dihedral_restraint_list(
        &self,
        chain_code: &str,
        list_name: &str,
    ) -> NefDihedralRestraintList {
        let mut rows = Vec::new();
        let n = self.residues.len();
        for i in 0..n {
            let r = &self.residues[i];
            if r.class != TorsionPredictionClass::Good {
                continue;
            }
            if i > 0 && self.residues[i - 1].seq_id + 1 == r.seq_id {
                if let Some(phi) = r.phi_deg {
                    let hw = r
                        .phi_std_deg
                        .map(|s| (2.0 * s).clamp(15.0, 35.0))
                        .unwrap_or(20.0);
                    let idx = rows.len() + 1;
                    rows.push(NefDihedralRestraint::phi(
                        idx,
                        idx,
                        chain_code,
                        self.residues[i - 1].seq_id,
                        &self.residues[i - 1].res_name,
                        r.seq_id,
                        &r.res_name,
                        phi,
                        hw,
                        1.0,
                    ));
                }
            }
            if i + 1 < n && self.residues[i + 1].seq_id == r.seq_id + 1 {
                if let Some(psi) = r.psi_deg {
                    let hw = r
                        .psi_std_deg
                        .map(|s| (2.0 * s).clamp(15.0, 35.0))
                        .unwrap_or(20.0);
                    let idx = rows.len() + 1;
                    rows.push(NefDihedralRestraint::psi(
                        idx,
                        idx,
                        chain_code,
                        r.seq_id,
                        &r.res_name,
                        self.residues[i + 1].seq_id,
                        &self.residues[i + 1].res_name,
                        psi,
                        hw,
                        1.0,
                    ));
                }
            }
            if let Some(chi1) = r.chi1_deg {
                let hw = r
                    .chi1_std_deg
                    .map(|s| (2.0 * s).clamp(18.0, 35.0))
                    .unwrap_or(24.0);
                let idx = rows.len() + 1;
                if let Some(cr) = NefDihedralRestraint::chi1(
                    idx,
                    idx,
                    chain_code,
                    r.seq_id,
                    &r.res_name,
                    chi1,
                    hw,
                    1.0,
                ) {
                    rows.push(cr);
                }
            }
        }
        NefDihedralRestraintList {
            name: list_name.to_string(),
            potential_type: "square-well-parabolic".to_string(),
            restraints: rows,
        }
    }
}

/// One residue row inside a `TorsionDb` template chain.
#[derive(Clone, Debug, PartialEq)]
pub struct TorsionCorpusResidue {
    pub seq_id: i32,
    pub res_name: String,
    pub aa1: u8,
    /// Secondary shifts `[dH, dN, dCA, dCB, dC', dHA]` (ppm).
    pub sec_shifts: [Option<f64>; 6],
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub chi1_deg: Option<f64>,
    pub ca_coord: [f64; 3],
}

/// One non-redundant protein chain in `TorsionDb`.
#[derive(Clone, Debug, PartialEq)]
pub struct TorsionCorpusChain {
    pub chain_key: String,
    pub pdb_id: String,
    pub chain_id: String,
    pub bmrb_id: String,
    pub uniprot_id: String,
    pub family_id: String,
    pub topology_id: String,
    pub seq1: Vec<u8>,
    pub residues: Vec<TorsionCorpusResidue>,
}

/// Non-redundant experimental BMRB + wwPDB + CATH torsion fragment corpus.
#[derive(Clone, Debug, Default)]
pub struct TorsionDb {
    pub chains: Vec<TorsionCorpusChain>,
}

static DEFAULT_TORSION_DB: OnceLock<TorsionDb> = OnceLock::new();

impl TorsionDb {
    /// Return a shared reference to the embedded CC0 `data/torsion_db/torsion_corpus.tsv` database.
    pub fn embedded() -> &'static TorsionDb {
        DEFAULT_TORSION_DB.get_or_init(|| {
            Self::from_tsv_str(EMBEDDED_TORSION_CORPUS_TSV)
                .expect("embedded data/torsion_db/torsion_corpus.tsv must parse cleanly")
        })
    }

    /// FIPS 180-4 SHA-256 hex digest of the embedded `data/torsion_db/torsion_corpus.tsv`.
    pub fn embedded_sha256() -> String {
        sha256_hex(EMBEDDED_TORSION_CORPUS_TSV.as_bytes())
    }

    pub fn load_default() -> Result<Self, String> {
        Self::from_tsv_str(EMBEDDED_TORSION_CORPUS_TSV)
    }

    pub fn from_tsv_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path.as_ref()).map_err(|e| {
            format!(
                "failed to read torsion corpus '{}': {e}",
                path.as_ref().display()
            )
        })?;
        if !bytes.is_ascii() {
            return Err(format!(
                "torsion corpus '{}' contains non-ASCII bytes",
                path.as_ref().display()
            ));
        }
        let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
        Self::from_tsv_str(text)
    }

    /// ASCII-safe parser for `torsion_corpus.tsv`, rejecting non-finite numbers.
    pub fn from_tsv_str(text: &str) -> Result<Self, String> {
        if !text.as_bytes().is_ascii() {
            return Err("torsion corpus TSV contains non-ASCII bytes".to_string());
        }
        let mut chain_map: BTreeMap<String, usize> = BTreeMap::new();
        let mut chains: Vec<TorsionCorpusChain> = Vec::new();

        let parse_opt_finite =
            |s: &str, line_no: usize, name: &str| -> Result<Option<f64>, String> {
                let t = s.trim();
                if t.is_empty() || t == "." || t == "?" {
                    return Ok(None);
                }
                let v: f64 = t
                    .parse()
                    .map_err(|e| format!("line {line_no}: invalid {name} '{t}': {e}"))?;
                if !v.is_finite() {
                    return Err(format!("line {line_no}: non-finite {name} '{t}'"));
                }
                Ok(Some(v))
            };
        let parse_req_finite = |s: &str, line_no: usize, name: &str| -> Result<f64, String> {
            let t = s.trim();
            let v: f64 = t
                .parse()
                .map_err(|e| format!("line {line_no}: invalid {name} '{t}': {e}"))?;
            if !v.is_finite() {
                return Err(format!("line {line_no}: non-finite {name} '{t}'"));
            }
            Ok(v)
        };

        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            if raw.as_bytes().get(0..1).is_none() {
                continue;
            }
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let p: Vec<&str> = trimmed.split('\t').collect();
            if p[0] == "@CHAIN" {
                if p.len() < 8 {
                    return Err(format!("line {line_no}: malformed @CHAIN header"));
                }
                let ck = p[1].trim().to_string();
                let c_idx = chains.len();
                chain_map.insert(ck.clone(), c_idx);
                chains.push(TorsionCorpusChain {
                    chain_key: ck,
                    pdb_id: p[2].trim().to_ascii_uppercase(),
                    chain_id: p[3].trim().to_string(),
                    bmrb_id: p[4].trim().trim_start_matches("bmr").to_string(),
                    uniprot_id: p[5].trim().to_ascii_uppercase(),
                    family_id: p[6].trim().to_ascii_uppercase(),
                    topology_id: p[7].trim().to_ascii_uppercase(),
                    seq1: Vec::new(),
                    residues: Vec::new(),
                });
            } else if p[0] == "RES" {
                if p.len() < 16 {
                    return Err(format!("line {line_no}: malformed RES row"));
                }
                let ck = p[1].trim();
                let &c_idx = chain_map.get(ck).ok_or_else(|| {
                    format!("line {line_no}: RES references unknown chain '{ck}'")
                })?;
                let seq_id: i32 = p[2]
                    .trim()
                    .parse()
                    .map_err(|e| format!("line {line_no}: invalid seq_id: {e}"))?;
                let res_name = p[3].trim().to_ascii_uppercase();
                let aa1 = three_to_one(&res_name) as u8;
                let sec_shifts = [
                    parse_opt_finite(p[4], line_no, "dh")?,
                    parse_opt_finite(p[5], line_no, "dn")?,
                    parse_opt_finite(p[6], line_no, "dca")?,
                    parse_opt_finite(p[7], line_no, "dcb")?,
                    parse_opt_finite(p[8], line_no, "dc")?,
                    parse_opt_finite(p[9], line_no, "dha")?,
                ];
                let phi_deg = parse_opt_finite(p[10], line_no, "phi")?;
                let psi_deg = parse_opt_finite(p[11], line_no, "psi")?;
                let chi1_deg = parse_opt_finite(p[12], line_no, "chi1")?;
                let ca_coord = [
                    parse_req_finite(p[13], line_no, "ca_x")?,
                    parse_req_finite(p[14], line_no, "ca_y")?,
                    parse_req_finite(p[15], line_no, "ca_z")?,
                ];
                let ch = &mut chains[c_idx];
                ch.seq1.push(aa1);
                ch.residues.push(TorsionCorpusResidue {
                    seq_id,
                    res_name,
                    aa1,
                    sec_shifts,
                    phi_deg,
                    psi_deg,
                    chi1_deg,
                    ca_coord,
                });
            }
        }

        if chains.is_empty() {
            return Err("torsion corpus contains zero chains".to_string());
        }
        Ok(Self { chains })
    }
}

/// Alignment summary `(identities, aligned_pair_count, alignment_span_cols)` for sequence firewalling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqAlignmentStats {
    pub identities: usize,
    pub aligned_pairs: usize,
    pub alignment_cols: usize,
}

impl SeqAlignmentStats {
    /// Canonical sequence identity over the aligned column span (`identities / (matches + mismatches + internal_gaps)`).
    #[inline]
    pub fn identity_over_pairs(self) -> f64 {
        if self.alignment_cols == 0 {
            0.0
        } else {
            (self.identities as f64) / (self.alignment_cols as f64)
        }
    }

    #[inline]
    pub fn exceeds_firewall(self, query_len: usize, template_len: usize) -> bool {
        if query_len == 0
            || template_len == 0
            || self.aligned_pairs == 0
            || self.alignment_cols == 0
        {
            return false;
        }
        let q_cov = (self.aligned_pairs as f64) / (query_len as f64);
        let t_cov = (self.aligned_pairs as f64) / (template_len as f64);
        (q_cov >= 0.50 || t_cov >= 0.50) && self.identity_over_pairs() > 0.25
    }
}

/// Gotoh affine-gap alignment spanning the shorter sequence end-to-end (`alignment_cols >= min(len_a, len_b)`).
///
/// When `seq_a.len() == seq_b.len()`, performs true end-to-end global alignment `(0, 0) -> (n, m)` with
/// penalized terminal gaps on both sequences (preventing 1-2 residue corner-overhang escapes).
/// When lengths differ, allows free N/C overhangs ONLY on the longer sequence so an embedded homologous
/// domain is aligned end-to-end without dilution from unrelated flanking domains.
fn gotoh_shorter_global_stats<F>(
    seq_a: &[u8],
    seq_b: &[u8],
    gap_open: i32,
    gap_ext: i32,
    mut score_fn: F,
) -> SeqAlignmentStats
where
    F: FnMut(u8, u8) -> i32,
{
    const NEG_INF: i32 = -1_000_000_000;

    if seq_a.is_empty() || seq_b.is_empty() {
        return SeqAlignmentStats {
            identities: 0,
            aligned_pairs: 0,
            alignment_cols: 0,
        };
    }

    let (s_short, s_long) = if seq_a.len() <= seq_b.len() {
        (seq_a, seq_b)
    } else {
        (seq_b, seq_a)
    };
    let n = s_short.len();
    let m = s_long.len();
    let equal_len = n == m;

    let stride = m + 1;
    let total = (n + 1) * stride;
    let mut mat = vec![NEG_INF; total];
    let mut ix = vec![NEG_INF; total]; // gap in s_long (up move, consumes s_short)
    let mut iy = vec![NEG_INF; total]; // gap in s_short (left move, consumes s_long)
                                       // 6-bit packed traceback per cell:
                                       // bits 0..2: prev state for M  (0=M, 1=Ix, 2=Iy)
                                       // bits 2..4: prev state for Ix (0=M, 1=Ix, 2=Iy)
                                       // bits 4..6: prev state for Iy (0=M, 1=Ix, 2=Iy)
    let mut tb = vec![0u8; total];

    mat[0] = 0;
    // Leading gaps consuming s_short (col 0) are ALWAYS penalized
    for i in 1..=n {
        ix[i * stride] = gap_open + ((i - 1) as i32) * gap_ext;
        let ix_prev = if i == 1 { 0u8 } else { 1u8 };
        tb[i * stride] = ix_prev << 2;
    }
    if equal_len {
        // Equal lengths: true global (0,0) -> (n,m), leading gaps on s_long also penalized
        for j in 1..=m {
            iy[j] = gap_open + ((j - 1) as i32) * gap_ext;
            let iy_prev = if j == 1 { 0u8 } else { 2u8 };
            tb[j] = iy_prev << 4;
        }
    } else {
        // Strictly shorter query/template: free N-terminal overhang ONLY on s_long
        mat[..=m].fill(0);
    }

    for i in 1..=n {
        let ca = s_short[i - 1].to_ascii_uppercase();
        for j in 1..=m {
            let cb = s_long[j - 1].to_ascii_uppercase();
            let idx = i * stride + j;

            // Ix[i, j]: gap in s_long (from i-1, j)
            let up_idx = (i - 1) * stride + j;
            let cx_m = mat[up_idx] + gap_open;
            let cx_x = ix[up_idx] + gap_ext;
            let cx_y = iy[up_idx] + gap_open;
            let (best_ix, ix_prev) = if cx_x >= cx_m && cx_x >= cx_y {
                (cx_x, 1u8)
            } else if cx_m >= cx_y {
                (cx_m, 0u8)
            } else {
                (cx_y, 2u8)
            };
            ix[idx] = best_ix;

            // Iy[i, j]: gap in s_short (from i, j-1)
            let left_idx = i * stride + (j - 1);
            let cy_m = mat[left_idx] + gap_open;
            let cy_y = iy[left_idx] + gap_ext;
            let cy_x = ix[left_idx] + gap_open;
            let (best_iy, iy_prev) = if cy_y >= cy_m && cy_y >= cy_x {
                (cy_y, 2u8)
            } else if cy_m >= cy_x {
                (cy_m, 0u8)
            } else {
                (cy_x, 1u8)
            };
            iy[idx] = best_iy;

            // M[i, j]: align s_short[i-1] with s_long[j-1]
            let sub = score_fn(ca, cb);
            let diag_idx = (i - 1) * stride + (j - 1);
            let s_m = mat[diag_idx];
            let s_x = ix[diag_idx];
            let s_y = iy[diag_idx];
            let (best_prev, m_prev) = if s_m >= s_x && s_m >= s_y {
                (s_m, 0u8)
            } else if s_x >= s_y {
                (s_x, 1u8)
            } else {
                (s_y, 2u8)
            };
            mat[idx] = best_prev + sub;
            tb[idx] = m_prev | (ix_prev << 2) | (iy_prev << 4);
        }
    }

    let (mut i, mut j, mut state) = if equal_len {
        let end_idx = n * stride + m;
        let sm = mat[end_idx];
        let sx = ix[end_idx];
        let sy = iy[end_idx];
        let st = if sm >= sx && sm >= sy {
            0u8
        } else if sx >= sy {
            1u8
        } else {
            2u8
        };
        (n, m, st)
    } else {
        // Free C-terminal overhang ONLY on s_long (must finish at i == n)
        let mut best_score = ix[n * stride];
        let mut best_j = 0usize;
        let mut best_st = 1u8;
        for j_cand in 1..=m {
            let cell = n * stride + j_cand;
            for (sc, st) in [(mat[cell], 0u8), (ix[cell], 1u8), (iy[cell], 2u8)] {
                if sc > best_score {
                    best_score = sc;
                    best_j = j_cand;
                    best_st = st;
                }
            }
        }
        (n, best_j, best_st)
    };

    let mut identities = 0usize;
    let mut aligned_pairs = 0usize;
    let mut cols = 0usize;

    while i > 0 || (equal_len && j > 0) {
        let code = tb[i * stride + j];
        match state {
            0 => {
                cols += 1;
                aligned_pairs += 1;
                if s_short[i - 1].eq_ignore_ascii_case(&s_long[j - 1]) {
                    identities += 1;
                }
                state = code & 0x03;
                i -= 1;
                j -= 1;
            }
            1 => {
                cols += 1;
                state = (code >> 2) & 0x03;
                i -= 1;
            }
            _ => {
                cols += 1;
                state = (code >> 4) & 0x03;
                j -= 1;
            }
        }
    }

    SeqAlignmentStats {
        identities,
        aligned_pairs,
        alignment_cols: cols,
    }
}

/// Global/shorter-spanning Gotoh Needleman-Wunsch alignment (`BLOSUM62`, affine `gap_open = -11, gap_extend = -1`).
pub fn needleman_wunsch_stats(seq_a: &[u8], seq_b: &[u8]) -> SeqAlignmentStats {
    gotoh_shorter_global_stats(seq_a, seq_b, -11, -1, blosum62_score)
}

/// Global/shorter-spanning Gotoh Needleman-Wunsch exact-identity alignment (`match = +3, mismatch = -1, gap_open = -11, gap_extend = -2`),
/// guaranteeing positive alignment score for every `> 25%` sequence-identity homolog regardless of non-conservative substitutions
/// while preventing spurious gap-hopping onto random single-residue coincidences.
pub fn needleman_wunsch_identity_stats(seq_a: &[u8], seq_b: &[u8]) -> SeqAlignmentStats {
    gotoh_shorter_global_stats(seq_a, seq_b, -11, -2, |a, b| if a == b { 3 } else { -1 })
}

fn gotoh_local_stats<F>(
    seq_a: &[u8],
    seq_b: &[u8],
    gap_open: i32,
    gap_ext: i32,
    mut score_fn: F,
) -> SeqAlignmentStats
where
    F: FnMut(u8, u8) -> i32,
{
    const NEG_INF: i32 = -1_000_000_000;

    let n = seq_a.len();
    let m = seq_b.len();
    if n == 0 || m == 0 {
        return SeqAlignmentStats {
            identities: 0,
            aligned_pairs: 0,
            alignment_cols: 0,
        };
    }
    let stride = m + 1;
    let total = (n + 1) * stride;
    let mut mat = vec![0i32; total];
    let mut ix = vec![NEG_INF; total];
    let mut iy = vec![NEG_INF; total];
    let mut tb = vec![3u8; total];

    let mut best_score = 0i32;
    let mut best_ij = (0usize, 0usize);

    for i in 1..=n {
        let ca = seq_a[i - 1].to_ascii_uppercase();
        for j in 1..=m {
            let cb = seq_b[j - 1].to_ascii_uppercase();
            let idx = i * stride + j;

            let up_idx = (i - 1) * stride + j;
            let open_x = mat[up_idx] + gap_open;
            let ext_x = ix[up_idx] + gap_ext;
            let ix_from_ix = if ext_x > open_x {
                ix[idx] = ext_x;
                1u8
            } else {
                ix[idx] = open_x;
                0u8
            };

            let left_idx = i * stride + (j - 1);
            let open_y = mat[left_idx] + gap_open;
            let ext_y = iy[left_idx] + gap_ext;
            let iy_from_iy = if ext_y > open_y {
                iy[idx] = ext_y;
                1u8
            } else {
                iy[idx] = open_y;
                0u8
            };

            let sub = score_fn(ca, cb);
            let diag_idx = (i - 1) * stride + (j - 1);
            let s_m = mat[diag_idx];
            let s_x = ix[diag_idx];
            let s_y = iy[diag_idx];
            let (best_prev, prev_st) = if s_m >= s_x && s_m >= s_y {
                (
                    if s_m <= 0 { 0 } else { s_m },
                    if s_m <= 0 { 3u8 } else { 0u8 },
                )
            } else if s_x >= s_y {
                (s_x, 1u8)
            } else {
                (s_y, 2u8)
            };

            let cand = best_prev + sub;
            if cand <= 0 {
                mat[idx] = 0;
                tb[idx] = 3 | (ix_from_ix << 3) | (iy_from_iy << 4);
            } else {
                mat[idx] = cand;
                tb[idx] = prev_st | (ix_from_ix << 3) | (iy_from_iy << 4);
                if cand > best_score {
                    best_score = cand;
                    best_ij = (i, j);
                }
            }
        }
    }

    let (mut i, mut j) = best_ij;
    let mut state = 0u8;
    let mut identities = 0usize;
    let mut aligned_pairs = 0usize;
    let mut cols = 0usize;

    while i > 0 && j > 0 {
        let idx = i * stride + j;
        if state == 0 && mat[idx] <= 0 {
            break;
        }
        let code = tb[idx];
        match state {
            0 => {
                cols += 1;
                aligned_pairs += 1;
                if seq_a[i - 1].eq_ignore_ascii_case(&seq_b[j - 1]) {
                    identities += 1;
                }
                state = code & 0x07;
                i -= 1;
                j -= 1;
                if state == 3 {
                    break;
                }
            }
            1 => {
                cols += 1;
                state = if ((code >> 3) & 1) == 1 { 1 } else { 0 };
                i -= 1;
            }
            _ => {
                cols += 1;
                state = if ((code >> 4) & 1) == 1 { 2 } else { 0 };
                j -= 1;
            }
        }
    }

    SeqAlignmentStats {
        identities,
        aligned_pairs,
        alignment_cols: cols,
    }
}

/// Local Gotoh Smith-Waterman alignment (`BLOSUM62`, affine `gap_open = -11, gap_extend = -1`).
pub fn smith_waterman_stats(seq_a: &[u8], seq_b: &[u8]) -> SeqAlignmentStats {
    gotoh_local_stats(seq_a, seq_b, -11, -1, blosum62_score)
}

/// Local Gotoh Smith-Waterman exact-identity alignment (`match = +3, mismatch = -1, gap_open = -11, gap_extend = -2`).
pub fn smith_waterman_identity_stats(seq_a: &[u8], seq_b: &[u8]) -> SeqAlignmentStats {
    gotoh_local_stats(seq_a, seq_b, -11, -2, |a, b| if a == b { 3 } else { -1 })
}

/// Maximum sequence identity over `>= 50%` query OR template coverage across all four
/// `BLOSUM62` + exact-identity global/local alignments.
fn effective_seq_identity_50cov(seq_a: &[u8], seq_b: &[u8]) -> f64 {
    if seq_a.is_empty() || seq_b.is_empty() {
        return 0.0;
    }
    let qlen = seq_a.len() as f64;
    let tlen = seq_b.len() as f64;
    let mut best = 0.0f64;
    for st in [
        needleman_wunsch_stats(seq_a, seq_b),
        smith_waterman_stats(seq_a, seq_b),
        needleman_wunsch_identity_stats(seq_a, seq_b),
        smith_waterman_identity_stats(seq_a, seq_b),
    ] {
        let q_cov = (st.aligned_pairs as f64) / qlen;
        let t_cov = (st.aligned_pairs as f64) / tlen;
        if q_cov >= 0.50 || t_cov >= 0.50 {
            best = best.max(st.identity_over_pairs());
        }
    }
    best
}

/// Returns `true` if `seq_a` and `seq_b` share any 7-mer with `<= 1` mismatch (`>= 6/7` identical).
pub fn shares_7mer_within_1_mismatch(seq_a: &[u8], seq_b: &[u8]) -> bool {
    if seq_a.len() < 7 || seq_b.len() < 7 {
        return false;
    }
    for wa in seq_a.windows(7) {
        for wb in seq_b.windows(7) {
            let mut m = 0usize;
            for k in 0..7 {
                if wa[k].eq_ignore_ascii_case(&wb[k]) {
                    m += 1;
                }
            }
            if m >= 6 {
                return true;
            }
        }
    }
    false
}

/// Clean-room Zhang & Skolnick (*Proteins* 2004, 57, 702-710) `TM-score` on `C_alpha` coordinates,
/// normalized by `min(query_ca.len(), template_ca.len())`, with dense `shift_step = 1` diagonal +
/// sliding-window Kabsch seeding and iterative dynamic-programming equivalence refinement.
pub fn tm_score_ca(query_ca: &[[f64; 3]], template_ca: &[[f64; 3]]) -> f64 {
    let lq = query_ca.len();
    let lt = template_ca.len();
    if lq < 4 || lt < 4 {
        return 0.0;
    }

    let l_norm = lq.min(lt);
    let d0 = if l_norm > 15 {
        (1.24 * ((l_norm as f64) - 15.0).cbrt() - 1.8).max(0.5)
    } else {
        0.5
    };
    let d0_sq = d0 * d0;
    let inv_lnorm = 1.0 / (l_norm as f64);

    let apply_xform = |p: Vec3, rot: &[[f64; 3]; 3], c_mov: Vec3, c_ref: Vec3| -> Vec3 {
        let u = v_sub(p, c_mov);
        let r = [
            rot[0][0] * u[0] + rot[0][1] * u[1] + rot[0][2] * u[2],
            rot[1][0] * u[0] + rot[1][1] * u[1] + rot[1][2] * u[2],
            rot[2][0] * u[0] + rot[2][1] * u[1] + rot[2][2] * u[2],
        ];
        v_add(r, c_ref)
    };

    // Given an initial pairing `pairs` of `(q_idx, t_idx)`, iteratively refine Kabsch superposition
    // over contiguous sub-fragments and full equivalence to maximize Zhang-Skolnick TM sum.
    let refine_pairs = |pairs: &[(usize, usize)]| -> (f64, [[f64; 3]; 3], Vec3, Vec3) {
        let np = pairs.len();
        if np < 4 {
            return (
                0.0,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
                [0.0; 3],
            );
        }
        let mut best_tm = 0.0f64;
        let mut best_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut best_cm = [0.0; 3];
        let mut best_cr = [0.0; 3];

        let mut frag_lens = vec![np, (np / 2).max(4), (np / 4).max(4)];
        frag_lens.sort_unstable();
        frag_lens.dedup();

        let mut mov_buf = Vec::with_capacity(np);
        let mut ref_buf = Vec::with_capacity(np);

        for &flen in &frag_lens {
            let step = ((np - flen) / 4).max(1);
            let mut start = 0usize;
            while start + flen <= np {
                mov_buf.clear();
                ref_buf.clear();
                for &(qi, ti) in &pairs[start..(start + flen)] {
                    mov_buf.push(template_ca[ti]);
                    ref_buf.push(query_ca[qi]);
                }
                let (mut rot, mut cm, mut cr, _) = kabsch_transform(&mov_buf, &ref_buf);

                for cut_mult in [3.5, 2.0, 1.2] {
                    let cut_sq = (cut_mult * d0) * (cut_mult * d0);
                    let mut sum_tm = 0.0f64;
                    mov_buf.clear();
                    ref_buf.clear();
                    for &(qi, ti) in pairs {
                        let xt = apply_xform(template_ca[ti], &rot, cm, cr);
                        let d2 = v_dist_sq(query_ca[qi], xt);
                        sum_tm += 1.0 / (1.0 + d2 / d0_sq);
                        if d2 <= cut_sq {
                            mov_buf.push(template_ca[ti]);
                            ref_buf.push(query_ca[qi]);
                        }
                    }
                    let tm = sum_tm * inv_lnorm;
                    if tm > best_tm {
                        best_tm = tm;
                        best_rot = rot;
                        best_cm = cm;
                        best_cr = cr;
                    }
                    if mov_buf.len() >= 4 {
                        let (r2, cm2, cr2, _) = kabsch_transform(&mov_buf, &ref_buf);
                        rot = r2;
                        cm = cm2;
                        cr = cr2;
                    } else {
                        break;
                    }
                }
                let sum_final: f64 = pairs
                    .iter()
                    .map(|&(qi, ti)| {
                        let xt = apply_xform(template_ca[ti], &rot, cm, cr);
                        1.0 / (1.0 + v_dist_sq(query_ca[qi], xt) / d0_sq)
                    })
                    .sum();
                let tm_final = sum_final * inv_lnorm;
                if tm_final > best_tm {
                    best_tm = tm_final;
                    best_rot = rot;
                    best_cm = cm;
                    best_cr = cr;
                }
                if start + flen == np {
                    break;
                }
                start = (start + step).min(np - flen);
            }
        }
        (best_tm, best_rot, best_cm, best_cr)
    };

    // Semi-global DP equivalence search given current superposition `(rot, cm, cr)`,
    // allowing free N/C terminal overhangs so embedded domains & cloning tags are never forced onto random tails.
    let dp_structural_pairs = |rot: &[[f64; 3]; 3], cm: Vec3, cr: Vec3| -> Vec<(usize, usize)> {
        let stride = lt + 1;
        let mut dp = vec![0.0f64; (lq + 1) * stride];
        let mut tb = vec![3u8; (lq + 1) * stride];
        let gap_pen = -0.20f64;

        let xformed_t: Vec<Vec3> = template_ca
            .iter()
            .map(|&p| apply_xform(p, rot, cm, cr))
            .collect();

        for i in 1..=lq {
            let pq = query_ca[i - 1];
            for j in 1..=lt {
                let s_ij = 1.0 / (1.0 + v_dist_sq(pq, xformed_t[j - 1]) / d0_sq);
                let s_diag = dp[(i - 1) * stride + (j - 1)] + s_ij;
                let s_up = dp[(i - 1) * stride + j] + gap_pen;
                let s_left = dp[i * stride + (j - 1)] + gap_pen;
                let idx = i * stride + j;
                if s_diag >= s_up && s_diag >= s_left {
                    dp[idx] = s_diag;
                    tb[idx] = 0;
                } else if s_up >= s_left {
                    dp[idx] = s_up;
                    tb[idx] = 1;
                } else {
                    dp[idx] = s_left;
                    tb[idx] = 2;
                }
            }
        }
        let mut best_end_score = f64::NEG_INFINITY;
        let mut best_end = (lq, lt);
        for j in 1..=lt {
            let sc = dp[lq * stride + j];
            if sc > best_end_score {
                best_end_score = sc;
                best_end = (lq, j);
            }
        }
        for i in 1..=lq {
            let sc = dp[i * stride + lt];
            if sc > best_end_score {
                best_end_score = sc;
                best_end = (i, lt);
            }
        }

        let mut pairs = Vec::with_capacity(l_norm);
        let (mut i, mut j) = best_end;
        while i > 0 && j > 0 {
            match tb[i * stride + j] {
                0 => {
                    pairs.push((i - 1, j - 1));
                    i -= 1;
                    j -= 1;
                }
                1 => i -= 1,
                2 => j -= 1,
                _ => break,
            }
        }
        pairs.reverse();
        pairs
    };

    let mut best_overall_tm = 0.0f64;

    // Dense `shift_step = 1` gapless diagonal + sliding-window seed scan (zero skipped offsets)
    let min_ov = (l_norm / 2).max(4);
    let shift_min = -((lt as isize) - (min_ov as isize));
    let shift_max = (lq as isize) - (min_ov as isize);
    let mut seeds: Vec<(f64, Vec<(usize, usize)>)> =
        Vec::with_capacity((shift_max - shift_min + 1).max(0) as usize);

    let mut mov_tmp = Vec::with_capacity(l_norm);
    let mut ref_tmp = Vec::with_capacity(l_norm);
    for sh in shift_min..=shift_max {
        let q_start = sh.max(0) as usize;
        let t_start = (-sh).max(0) as usize;
        let len = (lq - q_start).min(lt - t_start);
        if len < 4 {
            continue;
        }
        let mut pairs = Vec::with_capacity(len);
        for k in 0..len {
            pairs.push((q_start + k, t_start + k));
        }

        let mut win_specs: Vec<(usize, usize)> = vec![(0, len)];
        let w_len = (l_norm / 2).clamp(8, len);
        if w_len < len {
            win_specs.push((0, w_len));
            win_specs.push(((len - w_len) / 2, w_len));
            win_specs.push((len - w_len, w_len));
            win_specs.sort_unstable();
            win_specs.dedup();
        }

        let mut best_sh_tm = 0.0f64;
        for (ws, wl) in win_specs {
            mov_tmp.clear();
            ref_tmp.clear();
            for &(qi, ti) in &pairs[ws..(ws + wl)] {
                mov_tmp.push(template_ca[ti]);
                ref_tmp.push(query_ca[qi]);
            }
            let (rot, cm, cr, _) = kabsch_transform(&mov_tmp, &ref_tmp);
            let tm_cand: f64 = pairs
                .iter()
                .map(|&(qi, ti)| {
                    let xt = apply_xform(template_ca[ti], &rot, cm, cr);
                    1.0 / (1.0 + v_dist_sq(query_ca[qi], xt) / d0_sq)
                })
                .sum::<f64>()
                * inv_lnorm;
            if tm_cand > best_sh_tm {
                best_sh_tm = tm_cand;
            }
        }
        seeds.push((best_sh_tm, pairs));
    }
    seeds.sort_by(|a, b| b.0.total_cmp(&a.0));
    seeds.truncate(8);

    for (_, mut pairs) in seeds {
        for _iter in 0..2 {
            let (tm, rot, cm, cr) = refine_pairs(&pairs);
            if tm > best_overall_tm {
                best_overall_tm = tm;
            }
            if best_overall_tm >= 0.999_999 {
                return best_overall_tm.min(1.0);
            }
            pairs = dp_structural_pairs(&rot, cm, cr);
        }
        let (tm_last, _, _, _) = refine_pairs(&pairs);
        if tm_last > best_overall_tm {
            best_overall_tm = tm_last;
        }
    }

    best_overall_tm.clamp(0.0, 1.0)
}

/// Apply all 5 layers of `HomologFirewall` (`Q5`), returning `(retained_chains, excluded_records, exclusion_list_sha256, max_retained_seq_id)`.
pub fn apply_homolog_firewall<'a>(
    db: &'a TorsionDb,
    query_seq1: &[u8],
    firewall: &HomologFirewallQuery,
) -> (
    Vec<&'a TorsionCorpusChain>,
    Vec<ExcludedTemplateRecord>,
    String,
    f64,
) {
    let q_pdbs: Vec<String> = firewall
        .query_pdb_ids
        .iter()
        .map(|s| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty() && s != "NONE")
        .collect();
    let q_bmrbs: Vec<String> = firewall
        .query_bmrb_ids
        .iter()
        .map(|s| s.trim().trim_start_matches("bmr").to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let q_unp = firewall
        .query_uniprot_id
        .as_deref()
        .map(|s| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty() && s != "NONE");
    let q_fam = firewall
        .query_family_id
        .as_deref()
        .map(|s| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty() && s != "NONE");
    let q_top = firewall
        .query_topology_id
        .as_deref()
        .map(|s| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty() && s != "NONE");

    let mut retained = Vec::with_capacity(db.chains.len());
    let mut excluded = Vec::new();
    let mut max_retained_seq_id = 0.0f64;

    for ch in &db.chains {
        let mut reasons = Vec::new();

        // Layer 1: Same PDB / BMRB / UniProt / protein family
        if q_pdbs.iter().any(|p| p == &ch.pdb_id) {
            reasons.push("L1:same_pdb".to_string());
        }
        if q_bmrbs.iter().any(|b| b == &ch.bmrb_id) {
            reasons.push("L1:same_bmrb".to_string());
        }
        if let Some(ref u) = q_unp {
            if !ch.uniprot_id.is_empty() && &ch.uniprot_id == u {
                reasons.push("L1:same_uniprot".to_string());
            }
        }
        if let Some(ref f) = q_fam {
            if !ch.family_id.is_empty() && &ch.family_id == f {
                reasons.push("L1:same_family".to_string());
            }
        }

        // Layer 2: Same ECOD H-group / CATH C.A.T topology fold group
        if let Some(ref top) = q_top {
            if !ch.topology_id.is_empty()
                && (&ch.topology_id == top
                    || ch
                        .family_id
                        .strip_prefix(top.as_str())
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with('.')))
            {
                reasons.push("L2:same_topology_fold".to_string());
            }
        }

        // Layer 3: Global/shorter-spanning NW + Local SW (BLOSUM62 AND exact-identity scoring,
        // > 25% seq identity over >= 50% query OR template coverage) + 7-mer <= 1 mismatch
        let qlen = query_seq1.len();
        let tlen = ch.seq1.len();
        let nw = needleman_wunsch_stats(query_seq1, &ch.seq1);
        let nw_id = needleman_wunsch_identity_stats(query_seq1, &ch.seq1);
        let best_nw_sid = [nw, nw_id]
            .into_iter()
            .filter(|st| st.exceeds_firewall(qlen, tlen))
            .map(|st| st.identity_over_pairs())
            .fold(0.0f64, f64::max);
        if best_nw_sid > 0.25 {
            reasons.push(format!("L3:nw_seq_id({best_nw_sid:.3})"));
        }

        let sw = smith_waterman_stats(query_seq1, &ch.seq1);
        let sw_id = smith_waterman_identity_stats(query_seq1, &ch.seq1);
        let best_sw_sid = [sw, sw_id]
            .into_iter()
            .filter(|st| st.exceeds_firewall(qlen, tlen))
            .map(|st| st.identity_over_pairs())
            .fold(0.0f64, f64::max);
        if best_sw_sid > 0.25 {
            reasons.push(format!("L3:sw_seq_id({best_sw_sid:.3})"));
        }

        if shares_7mer_within_1_mismatch(query_seq1, &ch.seq1) {
            reasons.push("L3:7mer_6of7".to_string());
        }

        // Layer 4: 3D TM-score >= 0.50 whenever query reference C_alpha coordinates exist
        if let Some(ref q_ca) = firewall.query_ca_coords {
            if q_ca.len() >= 4 {
                let t_ca: Vec<[f64; 3]> = ch.residues.iter().map(|r| r.ca_coord).collect();
                let tm = tm_score_ca(q_ca, &t_ca);
                if tm >= 0.50 {
                    reasons.push(format!("L4:tm_score({tm:.4})"));
                }
            }
        }

        if reasons.is_empty() {
            // Track max retained sequence identity (strictly gated on >= 50% query/template coverage)
            let qlen_f = qlen.max(1) as f64;
            let tlen_f = tlen.max(1) as f64;
            for st in [nw, sw, nw_id, sw_id] {
                if (st.aligned_pairs as f64) / qlen_f >= 0.50
                    || (st.aligned_pairs as f64) / tlen_f >= 0.50
                {
                    max_retained_seq_id = max_retained_seq_id.max(st.identity_over_pairs());
                }
            }
            retained.push(ch);
        } else {
            excluded.push(ExcludedTemplateRecord {
                chain_key: ch.chain_key.clone(),
                pdb_id: ch.pdb_id.clone(),
                bmrb_id: ch.bmrb_id.clone(),
                reasons,
            });
        }
    }

    // Layer 5: Deterministic SHA-256 digest of sorted excluded template manifest
    excluded.sort_by(|a, b| a.chain_key.cmp(&b.chain_key));
    let mut manifest = String::new();
    for ex in &excluded {
        manifest.push_str(&ex.chain_key);
        manifest.push(':');
        manifest.push_str(&ex.reasons.join(","));
        manifest.push('\n');
    }
    let sha = sha256_hex(manifest.as_bytes());

    (retained, excluded, sha, max_retained_seq_id)
}

/// Extract raw Wishart-1995 secondary shifts `[dH, dN, dCA, dCB, dC', dHA]` from `Reference`.
#[allow(clippy::needless_range_loop)]
fn extract_raw_secondary_shifts(shifts: &Reference) -> Vec<[Option<f64>; 6]> {
    let n = shifts.sequence.len();
    let mut out = vec![[None; 6]; n];
    for i in 0..n {
        let res = shifts.sequence[i].trim().to_ascii_uppercase();
        let next_pro = i + 1 < n && shifts.sequence[i + 1].trim().eq_ignore_ascii_case("PRO");
        let cb_obs = shifts
            .shifts
            .get(&(i, "CB".to_string()))
            .copied()
            .filter(|v| v.is_finite());
        let cys_ox = res == "CYS" && cb_obs.is_some_and(|v| v > 35.0);
        let rc = random_coil_shifts(&res, cys_ox, next_pro);

        let get_fin = |atom: &str| -> Option<f64> {
            shifts
                .shifts
                .get(&(i, atom.to_string()))
                .copied()
                .filter(|v| v.is_finite())
        };
        let h_obs = get_fin("H").or_else(|| get_fin("HN"));
        let n_obs = get_fin("N");
        let ca_obs = get_fin("CA");
        let c_obs = get_fin("C").or_else(|| get_fin("CO"));
        let ha_obs = get_fin("HA").or_else(|| {
            if res == "GLY" {
                match (get_fin("HA2"), get_fin("HA3")) {
                    (Some(a), Some(b)) => Some(0.5 * (a + b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            } else {
                None
            }
        });

        out[i] = [
            if res != "PRO" {
                h_obs.map(|v| v - rc.h)
            } else {
                None
            },
            if res != "PRO" {
                n_obs.map(|v| v - rc.n)
            } else {
                None
            },
            ca_obs.map(|v| v - rc.ca),
            if res != "GLY" {
                cb_obs.map(|v| v - rc.cb)
            } else {
                None
            },
            c_obs.map(|v| v - rc.c),
            ha_obs.map(|v| v - rc.ha),
        ];
    }
    out
}

/// Unlabelled 3-state `(Coil, Helix, Sheet)` EM + soft-thresholded coil-origin chemical-shift
/// reference offset estimator (Wang & Wishart, *J. Biomol. NMR* 2005, 31, 143-148).
///
/// Uses zero ground-truth secondary-structure or PDB labels: separates secondary-structure
/// polarization (`dCA` and `dCB` move in opposite directions in helices vs strands) from
/// uniform spectrometer referencing offsets (`dCA` and `dCB` shift in the same direction).
pub fn estimate_global_shift_offsets(shifts: &Reference) -> GlobalShiftOffsets {
    let raw = extract_raw_secondary_shifts(shifts);
    estimate_offsets_from_sec_shifts(&raw)
}

fn estimate_offsets_from_sec_shifts(raw: &[[Option<f64>; 6]]) -> GlobalShiftOffsets {
    const MU_C: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const MU_H: [f64; 6] = [-0.26, -1.50, 3.10, -0.40, 2.05, -0.38];
    const MU_E: [f64; 6] = [0.40, 1.20, -1.45, 2.05, -1.55, 0.42];
    const SIG: [f64; 6] = [0.42, 3.20, 1.35, 1.45, 1.25, 0.28];
    const PRIORS: [f64; 3] = [0.40, 0.32, 0.28];

    let mut offs = [0.0f64; 6];
    let mus = [MU_C, MU_H, MU_E];

    for _iter in 0..12 {
        let mut num = [0.0f64; 6];
        let mut cnt = [0.0f64; 6];

        for sh in raw {
            let mut lls = [0.0f64; 3];
            for s in 0..3 {
                let mut ll = PRIORS[s].ln();
                for k in 0..6 {
                    if let Some(v) = sh[k] {
                        let z = (v - offs[k] - mus[s][k]) / SIG[k];
                        ll -= 0.5 * (z * z).min(14.0);
                    }
                }
                lls[s] = ll;
            }
            let max_ll = lls[0].max(lls[1]).max(lls[2]);
            let p0 = (lls[0] - max_ll).exp();
            let p1 = (lls[1] - max_ll).exp();
            let p2 = (lls[2] - max_ll).exp();
            let p_sum = p0 + p1 + p2;
            let ph = p1 / p_sum;
            let pe = p2 / p_sum;

            for k in 0..6 {
                if let Some(v) = sh[k] {
                    let expected = ph * MU_H[k] + pe * MU_E[k];
                    let resid = v - expected;
                    if (resid - offs[k]).abs() <= 4.0 * SIG[k] {
                        num[k] += resid;
                        cnt[k] += 1.0;
                    }
                }
            }
        }

        for k in 0..6 {
            if cnt[k] >= 6.0 {
                let est = num[k] / cnt[k];
                // Soft deadzone (`0.32 * sigma_k`) keeps well-referenced BMRB entries untouched
                // while cleanly tracking systematic >0.5 ppm 13C / >1.0 ppm 15N / >0.15 ppm 1H offsets.
                let dead = 0.32 * SIG[k];
                offs[k] = if est.abs() > dead {
                    est - dead.copysign(est) * 0.35
                } else {
                    0.10 * est
                };
            }
        }
    }

    GlobalShiftOffsets {
        dh: offs[0],
        dn: offs[1],
        dca: offs[2],
        dcb: offs[3],
        dc: offs[4],
        dha: offs[5],
    }
}

/// Compute Berjanskii & Wishart (*J. Am. Chem. Soc.* 2005, 127, 14970-14971) Random Coil Index
/// (`RCI`) and model-free order parameter `S^2_RCI` from offset-corrected secondary shifts.
pub fn compute_rci_and_s2(sec_shifts: &[[Option<f64>; 6]]) -> (Vec<f64>, Vec<f64>) {
    const SCALES: [f64; 6] = [0.45, 2.60, 1.80, 1.80, 1.70, 0.38];
    const WEIGHTS: [f64; 6] = [1.0, 1.0, 2.5, 1.5, 2.0, 2.0];
    let n = sec_shifts.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut raw_rci = vec![0.20f64; n];
    for (i, rci_slot) in raw_rci.iter_mut().enumerate() {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for k in 0..6 {
            let mut sm_sum = 0.0f64;
            let mut sm_w = 0.0f64;
            for (d, wt) in [(-1isize, 0.25), (0, 0.50), (1, 0.25)] {
                let j = i as isize + d;
                if j >= 0 && (j as usize) < n {
                    if let Some(v) = sec_shifts[j as usize][k] {
                        sm_sum += wt * v.abs();
                        sm_w += wt;
                    }
                }
            }
            if sm_w > 0.0 {
                let sm = sm_sum / sm_w;
                num += WEIGHTS[k] * (sm / SCALES[k]);
                den += WEIGHTS[k];
            }
        }
        let avg_z = if den > 0.0 { num / den } else { 0.0 };
        *rci_slot = 0.22 / (1.0 + 3.2 * avg_z);
    }

    let mut rci_out = Vec::with_capacity(n);
    let mut s2_out = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = 0.0f64;
        let mut w = 0.0f64;
        for (d, wt) in [(-1isize, 0.25), (0, 0.50), (1, 0.25)] {
            let j = i as isize + d;
            if j >= 0 && (j as usize) < n {
                s += wt * raw_rci[j as usize];
                w += wt;
            }
        }
        let rci_v = s / w.max(1e-9);
        let s2_v = (1.0 - 0.4 * (1.0 + 17.8 * rci_v).ln()).clamp(0.0, 1.0);
        rci_out.push(rci_v);
        s2_out.push(s2_v);
    }
    (rci_out, s2_out)
}

/// Check whether `(phi_deg, psi_deg)` is a Lovell et al. (*Proteins* 2003) Ramachandran basin outlier.
pub fn is_ramachandran_outlier(res_name: &str, phi_deg: f64, psi_deg: f64) -> bool {
    let up = res_name.trim().to_ascii_uppercase();
    if up == "GLY" {
        return false;
    }
    if up == "PRO" {
        let allowed_pro = (-115.0..=-30.0).contains(&phi_deg)
            && ((-85.0..=45.0).contains(&psi_deg)
                || (90.0..=180.0).contains(&psi_deg)
                || (-180.0..=-135.0).contains(&psi_deg));
        return !allowed_pro;
    }
    let in_alpha = (-160.0..=-20.0).contains(&phi_deg) && (-90.0..=50.0).contains(&psi_deg);
    let in_beta = (-180.0..=-35.0).contains(&phi_deg)
        && ((50.0..=180.0).contains(&psi_deg) || (-180.0..=-115.0).contains(&psi_deg));
    let in_alpha_l = (25.0..=105.0).contains(&phi_deg) && (-35.0..=95.0).contains(&psi_deg);
    !(in_alpha || in_beta || in_alpha_l)
}

fn classify_torsion_ss(phi_deg: f64, psi_deg: f64) -> SecondaryStructure {
    if (-120.0..=-35.0).contains(&phi_deg) && (-75.0..=-5.0).contains(&psi_deg) {
        SecondaryStructure::Helix
    } else if (-180.0..=-60.0).contains(&phi_deg)
        && ((85.0..=180.0).contains(&psi_deg) || (-180.0..=-135.0).contains(&psi_deg))
    {
        SecondaryStructure::Sheet
    } else {
        SecondaryStructure::Coil
    }
}

/// Predict `(phi, psi, chi1)` torsions from assigned chemical shifts using the embedded `TorsionDb`
/// without benchmark firewall exclusions (production mode; Layer 5 audit SHA-256 still recorded).
pub fn predict_torsions(shifts: &Reference) -> TorsionPredictionReport {
    predict_torsions_with_db(
        TorsionDb::embedded(),
        shifts,
        &HomologFirewallQuery::default(),
    )
}

/// Predict `(phi, psi, chi1)` torsions from assigned chemical shifts enforcing all 5 layers
/// of `HomologFirewall` (`Q5`).
pub fn predict_torsions_with_firewall(
    shifts: &Reference,
    firewall: &HomologFirewallQuery,
) -> TorsionPredictionReport {
    predict_torsions_with_db(TorsionDb::embedded(), shifts, firewall)
}

/// Predict `(phi, psi, chi1)` torsions against an explicit `TorsionDb` instance and `HomologFirewallQuery`.
pub fn predict_torsions_with_db(
    db: &TorsionDb,
    shifts: &Reference,
    firewall: &HomologFirewallQuery,
) -> TorsionPredictionReport {
    const INV_VAR: [f64; 6] = [
        1.0 / (0.35 * 0.35),
        1.0 / (2.10 * 2.10),
        1.0 / (0.90 * 0.90),
        1.0 / (1.05 * 1.05),
        1.0 / (0.90 * 0.90),
        1.0 / (0.24 * 0.24),
    ];
    const POS_NUC_W: [[f64; 6]; 3] = [
        [0.6, 0.8, 1.4, 1.1, 2.0, 1.3], // i - 1: C'_{i-1} couples directly across peptide bond to phi_i
        [1.3, 1.6, 2.4, 1.9, 2.1, 2.2], // i: all 6 nuclei
        [1.4, 1.8, 1.4, 1.1, 1.2, 1.3], // i + 1: N_{i+1}, H_{i+1} couple directly to psi_i
    ];

    let n = shifts.sequence.len();
    let q_seq1: Vec<u8> = shifts
        .sequence
        .iter()
        .map(|r| three_to_one(r) as u8)
        .collect();
    let (kept_chains, excluded_details, exclusion_list_sha256, max_retained_seq_identity) =
        apply_homolog_firewall(db, &q_seq1, firewall);
    let excluded_template_ids: Vec<String> = excluded_details
        .iter()
        .map(|e| e.chain_key.clone())
        .collect();

    let mut sec = extract_raw_secondary_shifts(shifts);
    let shift_offsets = estimate_offsets_from_sec_shifts(&sec);
    let offs_arr = shift_offsets.as_array();
    for row in &mut sec {
        for k in 0..6 {
            if let Some(v) = row[k] {
                row[k] = Some(v - offs_arr[k]);
            }
        }
    }

    let (rci_vec, s2_vec) = compute_rci_and_s2(&sec);
    let kept_seq_ids: Vec<f64> = kept_chains
        .iter()
        .map(|ch| effective_seq_identity_50cov(&q_seq1, &ch.seq1))
        .collect();
    let mut predictions = Vec::with_capacity(n);
    let mut nearest_template_seq_identities = vec![0.0f64; n];

    for i in 0..n {
        let seq_id = shifts.seq_ids.get(i).copied().unwrap_or((i + 1) as i32);
        let res_name = shifts.sequence[i].trim().to_ascii_uppercase();
        let obs_count = sec[i].iter().filter(|x| x.is_some()).count();

        // Terminal residues or < 2 assigned shifts -> Unassigned
        if i == 0 || i + 1 >= n || obs_count < 2 {
            predictions.push(ResidueTorsionPrediction {
                seq_id,
                res_name,
                class: TorsionPredictionClass::Unassigned,
                phi_deg: None,
                phi_std_deg: None,
                psi_deg: None,
                psi_std_deg: None,
                chi1_deg: None,
                chi1_std_deg: None,
                rci: rci_vec[i],
                rci_s2: s2_vec[i],
                consensus_fraction: 0.0,
                ss_class: SecondaryStructure::Coil,
            });
            continue;
        }

        // Flexible residues (`RCI S^2 < 0.65`) -> Dynamic
        if s2_vec[i] < 0.65 {
            predictions.push(ResidueTorsionPrediction {
                seq_id,
                res_name,
                class: TorsionPredictionClass::Dynamic,
                phi_deg: None,
                phi_std_deg: None,
                psi_deg: None,
                psi_std_deg: None,
                chi1_deg: None,
                chi1_std_deg: None,
                rci: rci_vec[i],
                rci_s2: s2_vec[i],
                consensus_fraction: 0.0,
                ss_class: SecondaryStructure::Coil,
            });
            continue;
        }

        let q_gly = res_name == "GLY";
        let q_pro = res_name == "PRO";
        let q_next_pro = shifts.sequence[i + 1].trim().eq_ignore_ascii_case("PRO");

        // Score all 3-mer + 9-mer candidates across firewall-retained corpus chains
        let mut cands: Vec<(f64, f64, f64, Option<f64>, f64)> = Vec::with_capacity(256);
        for (ch, &ch_seq_id) in kept_chains.iter().zip(kept_seq_ids.iter()) {
            let cres = &ch.residues;
            let clen = cres.len();
            if clen < 3 {
                continue;
            }
            for j in 1..(clen - 1) {
                let rj = &cres[j];
                let (Some(t_phi), Some(t_psi)) = (rj.phi_deg, rj.psi_deg) else {
                    continue;
                };
                if q_gly != (rj.res_name == "GLY")
                    || q_pro != (rj.res_name == "PRO")
                    || q_next_pro != (cres[j + 1].res_name == "PRO")
                {
                    continue;
                }

                // 3-mer core distance (`i-1, i, i+1`)
                let mut dist = 0.0f64;
                let mut sh_matches_center = 0usize;
                for (d_idx, (d, pos_w)) in [(-1isize, 1.15f64), (0, 2.50), (1, 1.15)]
                    .into_iter()
                    .enumerate()
                {
                    let qi = (i as isize + d) as usize;
                    let tj = (j as isize + d) as usize;
                    let qa1 = q_seq1[qi];
                    let ta1 = cres[tj].aa1;
                    let b_diff =
                        (blosum62_score(qa1, qa1) - blosum62_score(qa1, ta1)).max(0) as f64;
                    dist += pos_w * 0.13 * b_diff;

                    let mut sh_sum = 0.0f64;
                    let mut sh_w = 0.0f64;
                    let nw = &POS_NUC_W[d_idx];
                    for k in 0..6 {
                        if let (Some(qv), Some(tv)) = (sec[qi][k], cres[tj].sec_shifts[k]) {
                            let diff = (qv - tv).clamp(-3.8, 3.8);
                            sh_sum += nw[k] * diff * diff * INV_VAR[k];
                            sh_w += nw[k];
                            if d == 0 {
                                sh_matches_center += 1;
                            }
                        }
                    }
                    if let (Some(qca), Some(qcb), Some(tca), Some(tcb)) = (
                        sec[qi][2],
                        sec[qi][3],
                        cres[tj].sec_shifts[2],
                        cres[tj].sec_shifts[3],
                    ) {
                        let dd = ((qca - qcb) - (tca - tcb)).clamp(-5.0, 5.0);
                        sh_sum += 1.4 * (dd * dd) / (1.45 * 1.45);
                        sh_w += 1.4;
                    }
                    dist += pos_w
                        * if sh_w > 0.0 {
                            (sh_sum / sh_w) * 3.2
                        } else {
                            5.5
                        };
                }
                if sh_matches_center < 2 {
                    continue;
                }

                // 9-mer outer flanking context (`-4..-2`, `+2..+4`)
                let mut ctx_sum = 0.0f64;
                let mut ctx_w = 0.0f64;
                for (d, cw) in [
                    (-4isize, 0.4f64),
                    (-3, 0.6),
                    (-2, 1.0),
                    (2, 1.0),
                    (3, 0.6),
                    (4, 0.4),
                ] {
                    let qi = i as isize + d;
                    let tj = j as isize + d;
                    if qi >= 0 && (qi as usize) < n && tj >= 0 && (tj as usize) < clen {
                        let qu = qi as usize;
                        let tu = tj as usize;
                        let mut s_d = 0.0f64;
                        let mut s_c = 0.0f64;
                        for k in 0..6 {
                            if let (Some(qv), Some(tv)) = (sec[qu][k], cres[tu].sec_shifts[k]) {
                                let diff = (qv - tv).clamp(-4.0, 4.0);
                                s_d += diff * diff * INV_VAR[k];
                                s_c += 1.0;
                            }
                        }
                        if s_c > 0.0 {
                            ctx_sum += cw * (s_d / s_c);
                            ctx_w += cw;
                        }
                        let b_diff = (blosum62_score(q_seq1[qu], q_seq1[qu])
                            - blosum62_score(q_seq1[qu], cres[tu].aa1))
                        .max(0) as f64;
                        ctx_sum += cw * 0.06 * b_diff;
                    }
                }
                if ctx_w > 0.0 {
                    dist += 1.15 * (ctx_sum / ctx_w);
                }

                cands.push((dist, t_phi, t_psi, rj.chi1_deg, ch_seq_id));
            }
        }

        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let top_k = cands.len().min(10);
        if top_k < 7 {
            predictions.push(ResidueTorsionPrediction {
                seq_id,
                res_name,
                class: TorsionPredictionClass::Ambiguous,
                phi_deg: None,
                phi_std_deg: None,
                psi_deg: None,
                psi_std_deg: None,
                chi1_deg: None,
                chi1_std_deg: None,
                rci: rci_vec[i],
                rci_s2: s2_vec[i],
                consensus_fraction: 0.0,
                ss_class: SecondaryStructure::Coil,
            });
            continue;
        }
        let top = &cands[..top_k];
        nearest_template_seq_identities[i] = top[0].4;

        // Find tightest consensus cluster in `(phi, psi)` within 40.0 deg radius
        let mut best_cluster: Vec<(f64, f64, f64, Option<f64>, f64)> = Vec::new();
        for &(_, p0, s0, _, _) in top {
            let cl: Vec<_> = top
                .iter()
                .copied()
                .filter(|&(_, p, s, _, _)| {
                    periodic_angle_abs_diff_deg(p, p0).hypot(periodic_angle_abs_diff_deg(s, s0))
                        <= 40.0
                })
                .collect();
            if cl.len() > best_cluster.len() {
                best_cluster = cl;
            }
        }

        let cons_frac = (best_cluster.len() as f64) / (top_k as f64);
        if best_cluster.len() < 7 {
            predictions.push(ResidueTorsionPrediction {
                seq_id,
                res_name,
                class: TorsionPredictionClass::Ambiguous,
                phi_deg: None,
                phi_std_deg: None,
                psi_deg: None,
                psi_std_deg: None,
                chi1_deg: None,
                chi1_std_deg: None,
                rci: rci_vec[i],
                rci_s2: s2_vec[i],
                consensus_fraction: cons_frac,
                ss_class: SecondaryStructure::Coil,
            });
            continue;
        }
        nearest_template_seq_identities[i] = best_cluster[0].4;

        // Boltzmann-weighted circular mean and spread
        let d_min = best_cluster[0].0;
        let mut w_tot = 0.0f64;
        let mut sin_phi = 0.0f64;
        let mut cos_phi = 0.0f64;
        let mut sin_psi = 0.0f64;
        let mut cos_psi = 0.0f64;
        let mut ws = Vec::with_capacity(best_cluster.len());
        for &(d, p, s, _, _) in &best_cluster {
            let w = (-0.45 * (d - d_min)).exp();
            ws.push(w);
            w_tot += w;
            let (sp, cp) = p.to_radians().sin_cos();
            let (ss, cs) = s.to_radians().sin_cos();
            sin_phi += w * sp;
            cos_phi += w * cp;
            sin_psi += w * ss;
            cos_psi += w * cs;
        }
        let pred_phi = wrap_deg((sin_phi / w_tot).atan2(cos_phi / w_tot).to_degrees());
        let pred_psi = wrap_deg((sin_psi / w_tot).atan2(cos_psi / w_tot).to_degrees());

        let mut var_phi = 0.0f64;
        let mut var_psi = 0.0f64;
        for (&w, &(_, p, s, _, _)) in ws.iter().zip(best_cluster.iter()) {
            let dp = periodic_angle_abs_diff_deg(p, pred_phi);
            let ds = periodic_angle_abs_diff_deg(s, pred_psi);
            var_phi += w * dp * dp;
            var_psi += w * ds * ds;
        }
        let std_phi = (var_phi / w_tot).sqrt().max(5.0);
        let std_psi = (var_psi / w_tot).sqrt().max(5.0);
        let comb_std = std_phi.hypot(std_psi);

        if comb_std > 25.0 || is_ramachandran_outlier(&res_name, pred_phi, pred_psi) {
            predictions.push(ResidueTorsionPrediction {
                seq_id,
                res_name,
                class: TorsionPredictionClass::Ambiguous,
                phi_deg: None,
                phi_std_deg: None,
                psi_deg: None,
                psi_std_deg: None,
                chi1_deg: None,
                chi1_std_deg: None,
                rci: rci_vec[i],
                rci_s2: s2_vec[i],
                consensus_fraction: cons_frac,
                ss_class: SecondaryStructure::Coil,
            });
            continue;
        }

        // Sidechain chi1 rotamer consensus (for non-Gly/Ala residues when >= 7 cluster members agree within 36 deg)
        let (chi1_deg, chi1_std_deg) = if res_name != "GLY" && res_name != "ALA" {
            let chi_vals: Vec<(f64, f64)> = ws
                .iter()
                .zip(best_cluster.iter())
                .filter_map(|(&w, &(_, _, _, c, _))| c.map(|ang| (w, ang)))
                .collect();
            let mut best_chi_cl: Vec<(f64, f64)> = Vec::new();
            for &(_, c0) in &chi_vals {
                let cl: Vec<_> = chi_vals
                    .iter()
                    .copied()
                    .filter(|&(_, c)| periodic_angle_abs_diff_deg(c, c0) <= 36.0)
                    .collect();
                if cl.len() > best_chi_cl.len() {
                    best_chi_cl = cl;
                }
            }
            if best_chi_cl.len() >= 7 {
                let cw: f64 = best_chi_cl.iter().map(|x| x.0).sum();
                let sc: f64 = best_chi_cl
                    .iter()
                    .map(|&(w, c)| w * c.to_radians().sin())
                    .sum();
                let cc: f64 = best_chi_cl
                    .iter()
                    .map(|&(w, c)| w * c.to_radians().cos())
                    .sum();
                let m_chi = wrap_deg((sc / cw).atan2(cc / cw).to_degrees());
                let v_chi: f64 = best_chi_cl
                    .iter()
                    .map(|&(w, c)| {
                        let d = periodic_angle_abs_diff_deg(c, m_chi);
                        w * d * d
                    })
                    .sum::<f64>()
                    / cw;
                (Some(m_chi), Some(v_chi.sqrt().max(8.0)))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let ss_class = classify_torsion_ss(pred_phi, pred_psi);
        predictions.push(ResidueTorsionPrediction {
            seq_id,
            res_name,
            class: TorsionPredictionClass::Good,
            phi_deg: Some(pred_phi),
            phi_std_deg: Some(std_phi),
            psi_deg: Some(pred_psi),
            psi_std_deg: Some(std_psi),
            chi1_deg,
            chi1_std_deg,
            rci: rci_vec[i],
            rci_s2: s2_vec[i],
            consensus_fraction: cons_frac,
            ss_class,
        });
    }

    TorsionPredictionReport {
        residues: predictions,
        shift_offsets,
        excluded_template_ids,
        excluded_details,
        exclusion_list_sha256,
        max_retained_seq_identity,
        nearest_template_seq_identities,
    }
}

/// Format a `TorsionPredictionReport` as canonical TSV text including provenance & firewall SHA-256 header.
pub fn format_torsions_tsv(report: &TorsionPredictionReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# exclusion_list_sha256: {}\n# excluded_templates: {}\n# max_retained_seq_identity: {:.4}\n",
        report.exclusion_list_sha256,
        report.excluded_template_ids.len(),
        report.max_retained_seq_identity
    ));
    out.push_str(&format!(
        "# global_shift_offsets_ppm: dH={:+.3} dN={:+.2} dCA={:+.2} dCB={:+.2} dC={:+.2} dHA={:+.3}\n",
        report.shift_offsets.dh,
        report.shift_offsets.dn,
        report.shift_offsets.dca,
        report.shift_offsets.dcb,
        report.shift_offsets.dc,
        report.shift_offsets.dha
    ));
    out.push_str(
        "seq_id\tres_name\tclass\tphi_deg\tphi_std_deg\tpsi_deg\tpsi_std_deg\tchi1_deg\tchi1_std_deg\trci\trci_s2\tconsensus\tss\n",
    );
    let fmt_opt = |v: Option<f64>| -> String {
        v.map(|x| format!("{x:.2}"))
            .unwrap_or_else(|| ".".to_string())
    };
    for r in &report.residues {
        let ss = match r.ss_class {
            SecondaryStructure::Helix => "H",
            SecondaryStructure::Sheet => "E",
            SecondaryStructure::Coil => "C",
        };
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.2}\t{}\n",
            r.seq_id,
            r.res_name,
            r.class.as_str(),
            fmt_opt(r.phi_deg),
            fmt_opt(r.phi_std_deg),
            fmt_opt(r.psi_deg),
            fmt_opt(r.psi_std_deg),
            fmt_opt(r.chi1_deg),
            fmt_opt(r.chi1_std_deg),
            r.rci,
            r.rci_s2,
            r.consensus_fraction,
            ss
        ));
    }
    out
}

pub fn write_torsions_tsv(
    path: impl AsRef<Path>,
    report: &TorsionPredictionReport,
) -> Result<(), String> {
    fs::write(path.as_ref(), format_torsions_tsv(report)).map_err(|e| {
        format!(
            "failed to write torsions TSV '{}': {e}",
            path.as_ref().display()
        )
    })
}

pub fn format_torsions_json(report: &TorsionPredictionReport) -> Result<String, String> {
    serde_json::to_string_pretty(report).map_err(|e| format!("JSON serialization error: {e}"))
}
