//! Empirical BMRB chemical shift priors per residue type and atom, plus automatic reference alignment
//! and deterministic covalent primary-sequence neighbor corrections.
//!
//! Provenance: Clean-room original implementation of `docs/specs/03-assignment.md` section 3,
//! shipping empirical `(mean, sd)` priors in ppm computed from 21 DSS-referenced public
//! BioMagResBank (BMRB) NMR-STAR depositions (`data/priors/bmrb_priors.json` &
//! `data/priors/PROVENANCE.md`; Ulrich et al., *Nucleic Acids Res.* 2008, 36, D402–D408;
//! referencing QC and marginal chemical-shift offset alignment per Zhang, Neal & Wishart,
//! *J. Biomol. NMR* 2003, 25, 173–195; Wang, Wishart et al., *J. Biomol. NMR* 2005, 31, 143–148),
//! strictly disjoint from the benchmark suite (`data/benchmark/manifest.tsv`), with literature
//! fallback strictly when `N < 5` (Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81).
//! Primary-sequence covalent neighbor chemical shift adjustments (`delta_covalent`) are derived
//! strictly from peer-reviewed random-coil host-guest peptide and proline conformational studies:
//! Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81; Wishart et al., *J. Biomol. NMR* 1995, 6,
//! 135–140; Braun, Wider & Wüthrich, *J. Am. Chem. Soc.* 1994, 116, 8466–8469; Schwarzinger et
//! al., *J. Am. Chem. Soc.* 2001, 123, 2970–2978; and MacArthur & Thornton, *J. Mol. Biol.* 1991,
//! 218, 397–412.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::nmrstar::{read_star, read_star_all_atoms};
use crate::spinsystems::SpinSystem;

pub const ATOMS: [&str; 5] = ["H", "N", "CA", "CB", "C"];

/// Exact 21 DSS-referenced BMRB training entry IDs used to build `data/priors/bmrb_priors.json`
/// and `data/priors/bmrb_all_atom_priors.json` (strictly disjoint from `data/benchmark/`).
pub const TRAINING_BMRB_IDS: [&str; 21] = [
    "bmr4022", "bmr4267", "bmr4302", "bmr4472", "bmr4562", "bmr4670", "bmr4752", "bmr5010",
    "bmr5106", "bmr5211", "bmr5505", "bmr5757", "bmr6032", "bmr6240", "bmr6338", "bmr6622",
    "bmr7021", "bmr7280", "bmr15250", "bmr15380", "bmr18025",
];

const EMBEDDED_BMRB_PRIORS_JSON: &str = include_str!("../../../data/priors/bmrb_priors.json");
const EMBEDDED_BMRB_ALL_ATOM_PRIORS_JSON: &str =
    include_str!("../../../data/priors/bmrb_all_atom_priors.json");

static BMRB_BACKBONE_PRIORS: OnceLock<Priors> = OnceLock::new();
static BMRB_ALL_ATOM_PRIORS: OnceLock<Priors> = OnceLock::new();

pub const STANDARD_RESIDUES: [(&str, char); 20] = [
    ("ALA", 'A'),
    ("ARG", 'R'),
    ("ASN", 'N'),
    ("ASP", 'D'),
    ("CYS", 'C'),
    ("GLN", 'Q'),
    ("GLU", 'E'),
    ("GLY", 'G'),
    ("HIS", 'H'),
    ("ILE", 'I'),
    ("LEU", 'L'),
    ("LYS", 'K'),
    ("MET", 'M'),
    ("PHE", 'F'),
    ("PRO", 'P'),
    ("SER", 'S'),
    ("THR", 'T'),
    ("TRP", 'W'),
    ("TYR", 'Y'),
    ("VAL", 'V'),
];

pub fn one_to_three(c: char) -> &'static str {
    let up = c.to_ascii_uppercase();
    for &(three, one) in &STANDARD_RESIDUES {
        if one == up {
            return three;
        }
    }
    "UNK"
}

pub fn three_to_one(res: &str) -> char {
    let up = res.to_ascii_uppercase();
    for &(three, one) in &STANDARD_RESIDUES {
        if three == up {
            return one;
        }
    }
    'X'
}

pub fn is_standard_residue(res: &str) -> bool {
    STANDARD_RESIDUES.iter().any(|&(three, _)| three == res)
}

#[inline]
pub fn logpdf(v: f64, mu: f64, sd: f64) -> f64 {
    let z = (v - mu) / sd;
    -0.5 * z * z - (sd * TAU.sqrt()).ln()
}

/// Look up empirical BMRB backbone `(mean_ppm, std_ppm)` prior for `(residue, atom)`.
pub fn aa_prior(res: &str, atom: &str) -> Option<(f64, f64)> {
    let up_res = res.to_ascii_uppercase();
    let up_atom = atom.to_ascii_uppercase();
    let pri = BMRB_BACKBONE_PRIORS.get_or_init(|| {
        Priors::from_json_str(EMBEDDED_BMRB_PRIORS_JSON).expect("valid embedded BMRB priors")
    });
    pri.table
        .get(&up_res)
        .and_then(|m| m.get(&up_atom))
        .copied()
}

/// Look up empirical BMRB all-atom `(mean_ppm, std_ppm)` prior (backbone + sidechain heavy/H atoms).
pub fn aa_all_atom_prior(res: &str, atom: &str) -> Option<(f64, f64)> {
    let up_res = res.to_ascii_uppercase();
    let up_atom = atom.to_ascii_uppercase();
    let pri = BMRB_ALL_ATOM_PRIORS.get_or_init(|| {
        Priors::from_json_str(EMBEDDED_BMRB_ALL_ATOM_PRIORS_JSON)
            .expect("valid embedded all-atom BMRB priors")
    });
    pri.table
        .get(&up_res)
        .and_then(|m| m.get(&up_atom))
        .copied()
}

#[derive(Clone, Debug)]
pub struct Priors {
    pub table: BTreeMap<String, BTreeMap<String, (f64, f64)>>,
}

impl Default for Priors {
    fn default() -> Self {
        Self::bmrb()
    }
}

impl Priors {
    /// Empirical BMRB backbone priors (`CA, CB, C, N, H`) trained on 21 disjoint BMRB entries.
    pub fn bmrb() -> Self {
        BMRB_BACKBONE_PRIORS
            .get_or_init(|| {
                Self::from_json_str(EMBEDDED_BMRB_PRIORS_JSON)
                    .expect("valid embedded data/priors/bmrb_priors.json")
            })
            .clone()
    }

    /// Empirical BMRB all-atom priors (backbone + sidechain `13C`/`15N`/`1H`) trained on 21 disjoint BMRB entries.
    pub fn all_atoms() -> Self {
        BMRB_ALL_ATOM_PRIORS
            .get_or_init(|| {
                Self::from_json_str(EMBEDDED_BMRB_ALL_ATOM_PRIORS_JSON)
                    .expect("valid embedded data/priors/bmrb_all_atom_priors.json")
            })
            .clone()
    }

    /// Default backbone priors entry point (backed by empirical BMRB statistics from `bmrb_priors.json`).
    pub fn provisional() -> Self {
        Self::bmrb()
    }

    /// Literature fallback statistics (Wishart et al., *J. Biomol. NMR* 1995), used only when `N < min_count`.
    pub fn literature_fallback() -> Self {
        type AtomStat<'a> = (&'a str, (f64, f64));
        let raw: &[(&str, &[AtomStat<'_>])] = &[
            (
                "ALA",
                &[
                    ("CA", (53.1, 2.0)),
                    ("CB", (19.0, 1.8)),
                    ("C", (177.8, 2.1)),
                    ("N", (123.2, 3.5)),
                    ("H", (8.19, 0.6)),
                ],
            ),
            (
                "ARG",
                &[
                    ("CA", (56.8, 2.3)),
                    ("CB", (30.7, 1.8)),
                    ("C", (176.4, 2.0)),
                    ("N", (120.8, 3.6)),
                    ("H", (8.24, 0.6)),
                ],
            ),
            (
                "ASN",
                &[
                    ("CA", (53.5, 1.9)),
                    ("CB", (38.7, 1.7)),
                    ("C", (175.3, 1.8)),
                    ("N", (118.9, 4.0)),
                    ("H", (8.33, 0.6)),
                ],
            ),
            (
                "ASP",
                &[
                    ("CA", (54.7, 2.0)),
                    ("CB", (40.9, 1.6)),
                    ("C", (176.4, 1.8)),
                    ("N", (120.6, 3.9)),
                    ("H", (8.30, 0.6)),
                ],
            ),
            (
                "CYS",
                &[
                    ("CA", (58.0, 3.4)),
                    ("CB", (32.0, 5.5)),
                    ("C", (174.9, 2.1)),
                    ("N", (120.0, 4.4)),
                    ("H", (8.39, 0.7)),
                ],
            ),
            (
                "GLN",
                &[
                    ("CA", (56.6, 2.1)),
                    ("CB", (29.2, 1.8)),
                    ("C", (176.3, 2.0)),
                    ("N", (119.9, 3.5)),
                    ("H", (8.21, 0.6)),
                ],
            ),
            (
                "GLU",
                &[
                    ("CA", (57.3, 2.1)),
                    ("CB", (30.0, 1.7)),
                    ("C", (176.9, 2.0)),
                    ("N", (120.7, 3.5)),
                    ("H", (8.33, 0.6)),
                ],
            ),
            (
                "GLY",
                &[
                    ("CA", (45.4, 1.3)),
                    ("C", (173.9, 1.9)),
                    ("N", (109.6, 3.8)),
                    ("H", (8.33, 0.6)),
                ],
            ),
            (
                "HIS",
                &[
                    ("CA", (56.5, 2.3)),
                    ("CB", (30.2, 2.1)),
                    ("C", (175.2, 2.0)),
                    ("N", (119.7, 4.0)),
                    ("H", (8.25, 0.7)),
                ],
            ),
            (
                "ILE",
                &[
                    ("CA", (61.6, 2.7)),
                    ("CB", (38.6, 2.0)),
                    ("C", (175.8, 1.9)),
                    ("N", (121.4, 4.2)),
                    ("H", (8.28, 0.7)),
                ],
            ),
            (
                "LEU",
                &[
                    ("CA", (55.6, 2.1)),
                    ("CB", (42.3, 1.9)),
                    ("C", (177.0, 2.0)),
                    ("N", (121.8, 3.9)),
                    ("H", (8.22, 0.6)),
                ],
            ),
            (
                "LYS",
                &[
                    ("CA", (56.9, 2.2)),
                    ("CB", (32.8, 1.8)),
                    ("C", (176.6, 2.0)),
                    ("N", (121.0, 3.7)),
                    ("H", (8.18, 0.6)),
                ],
            ),
            (
                "MET",
                &[
                    ("CA", (56.1, 2.2)),
                    ("CB", (32.9, 2.2)),
                    ("C", (176.2, 2.1)),
                    ("N", (120.1, 3.5)),
                    ("H", (8.26, 0.6)),
                ],
            ),
            (
                "PHE",
                &[
                    ("CA", (58.1, 2.6)),
                    ("CB", (40.0, 2.0)),
                    ("C", (175.5, 2.0)),
                    ("N", (120.4, 4.1)),
                    ("H", (8.35, 0.7)),
                ],
            ),
            (
                "PRO",
                &[
                    ("CA", (63.3, 1.5)),
                    ("CB", (31.9, 1.2)),
                    ("C", (176.7, 1.6)),
                    ("N", (134.0, 6.0)),
                ],
            ),
            (
                "SER",
                &[
                    ("CA", (58.7, 2.1)),
                    ("CB", (63.8, 1.5)),
                    ("C", (174.6, 1.8)),
                    ("N", (116.3, 3.5)),
                    ("H", (8.28, 0.6)),
                ],
            ),
            (
                "THR",
                &[
                    ("CA", (62.2, 2.6)),
                    ("CB", (69.7, 1.7)),
                    ("C", (174.5, 1.8)),
                    ("N", (115.4, 4.7)),
                    ("H", (8.24, 0.6)),
                ],
            ),
            (
                "TRP",
                &[
                    ("CA", (57.7, 2.6)),
                    ("CB", (30.1, 2.0)),
                    ("C", (176.1, 2.0)),
                    ("N", (121.6, 4.1)),
                    ("H", (8.29, 0.8)),
                ],
            ),
            (
                "TYR",
                &[
                    ("CA", (58.1, 2.6)),
                    ("CB", (39.3, 2.2)),
                    ("C", (175.4, 2.0)),
                    ("N", (120.5, 4.1)),
                    ("H", (8.30, 0.7)),
                ],
            ),
            (
                "VAL",
                &[
                    ("CA", (62.5, 2.9)),
                    ("CB", (32.7, 1.8)),
                    ("C", (175.7, 1.9)),
                    ("N", (121.1, 4.5)),
                    ("H", (8.29, 0.7)),
                ],
            ),
        ];
        let mut table = BTreeMap::new();
        for &(res, atoms) in raw {
            let mut m = BTreeMap::new();
            for &(atom, (mu, sd)) in atoms {
                m.insert(atom.to_string(), (mu, sd));
            }
            table.insert(res.to_string(), m);
        }
        Self { table }
    }

    pub fn from_json_str(content: &str) -> Result<Self, String> {
        let raw: BTreeMap<String, BTreeMap<String, [f64; 2]>> =
            serde_json::from_str(content).map_err(|e| format!("invalid priors JSON: {e}"))?;
        let table = raw
            .into_iter()
            .map(|(r, d)| {
                (
                    r,
                    d.into_iter()
                        .map(|(a, pair)| (a, (pair[0], pair[1])))
                        .collect(),
                )
            })
            .collect();
        Ok(Self { table })
    }

    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("failed to read priors JSON: {e}"))?;
        Self::from_json_str(&content)
    }

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let raw: BTreeMap<&String, BTreeMap<&String, [f64; 2]>> = self
            .table
            .iter()
            .map(|(r, d)| (r, d.iter().map(|(a, &(mu, sd))| (a, [mu, sd])).collect()))
            .collect();
        let text = serde_json::to_string_pretty(&raw)
            .map_err(|e| format!("failed to serialize priors: {e}"))?;
        fs::write(path.as_ref(), format!("{text}\n"))
            .map_err(|e| format!("failed to write priors JSON: {e}"))
    }

    #[inline]
    pub fn get(&self, res: &str, atom: &str) -> Option<(f64, f64)> {
        self.table.get(res).and_then(|m| m.get(atom)).copied()
    }

    #[inline]
    pub fn has(&self, res: &str, atom: &str) -> bool {
        self.table
            .get(res)
            .is_some_and(|atoms| atoms.contains_key(atom))
    }

    #[inline]
    pub fn logpdf(&self, res: &str, atom: &str, v: f64) -> f64 {
        let (mu, sd) = self.table[res][atom];
        logpdf(v, mu, sd)
    }

    /// Log of the residue-averaged density (Pro excluded for H, Gly excluded for CB).
    pub fn background(&self, atom: &str, v: f64) -> f64 {
        let mut vals = Vec::with_capacity(20);
        for &(res, _) in &STANDARD_RESIDUES {
            if let Some(&(mu, sd)) = self.table.get(res).and_then(|d| d.get(atom)) {
                vals.push(logpdf(v, mu, sd));
            }
        }
        if vals.is_empty() {
            for d in self.table.values() {
                if let Some(&(mu, sd)) = d.get(atom) {
                    vals.push(logpdf(v, mu, sd));
                }
            }
        }
        let m = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = vals.iter().map(|&x| (x - m).exp()).sum();
        m + (sum / (vals.len() as f64)).ln()
    }

    pub fn llr(&self, res: &str, atom: &str, v: f64) -> f64 {
        self.llr_with(res, atom, v, -12.0, 0.02_f64.ln())
    }

    pub fn llr_with(&self, res: &str, atom: &str, v: f64, floor: f64, gly_cb: f64) -> f64 {
        if !self.table.contains_key(res) {
            return 0.0;
        }
        if !self.has(res, atom) {
            return gly_cb;
        }
        (self.logpdf(res, atom, v) - self.background(atom, v)).max(floor)
    }

    /// Log-likelihood ratio with deterministic zero-mean primary-sequence covalent shift `d_mu`
    /// (`delta_covalent`), modeled on pre-Proline positions as a two-state conformational mixture
    /// (`0.85 * N(v | mu_non + raw, sd) + 0.15 * N(v | mu_non, sd)`, where `mu_non` removes the
    /// `91 / 2508` BMRB training pre-Proline background) and on non-pre-Proline positions as
    /// `N(v | mu_non, sd)`, while keeping `background(atom, v)` slot-independent.
    pub fn llr_shifted(&self, res: &str, atom: &str, v: f64, d_mu: f64) -> f64 {
        self.llr_shifted_with(res, atom, v, d_mu, -12.0, 0.02_f64.ln())
    }

    pub fn llr_shifted_with(
        &self,
        res: &str,
        atom: &str,
        v: f64,
        d_mu: f64,
        floor: f64,
        gly_cb: f64,
    ) -> f64 {
        if !self.table.contains_key(res) {
            return 0.0;
        }
        if !self.has(res, atom) {
            return gly_cb;
        }
        if d_mu.abs() < 1e-12 {
            return (self.logpdf(res, atom, v) - self.background(atom, v)).max(floor);
        }
        let (mu, sd) = self.table[res][atom];
        // Pre-Proline expected shifts have |d_mu| >= 0.81 ppm (CA/C); non-pre-Proline background
        // centering has |d_mu| <= 0.061 ppm and evaluates a single background-debiased Gaussian.
        let lp = if d_mu.abs() >= 0.09 {
            let raw = d_mu / (COVALENT_COIL_WEIGHT * (1.0 - NATURAL_PRO_FRACTION));
            let d_non = -COVALENT_COIL_WEIGHT * NATURAL_PRO_FRACTION * raw;
            let a = COVALENT_COIL_WEIGHT.ln() + logpdf(v, mu + d_non + raw, sd);
            let b = (1.0 - COVALENT_COIL_WEIGHT).ln() + logpdf(v, mu + d_non, sd);
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        } else {
            logpdf(v, mu + d_mu, sd)
        };
        (lp - self.background(atom, v)).max(floor)
    }

    /// Return a copy of `self` with backbone prior means shifted by `offsets` (`mu_shifted = mu + offset`),
    /// leaving standard deviations untouched so scoring matches the experimental spectrum's referencing frame.
    pub fn with_offsets(&self, offsets: &ShiftOffsets) -> Self {
        if offsets.is_zero() {
            return self.clone();
        }
        let mut table = self.table.clone();
        for atoms in table.values_mut() {
            for &atom in &ATOMS {
                let d = offsets.get(atom);
                if d.abs() > 1e-12 {
                    if let Some(entry) = atoms.get_mut(atom) {
                        entry.0 += d;
                    }
                }
            }
        }
        Self { table }
    }
}

/// Global chemical-shift referencing offsets `(experimental_ppm - dss_prior_ppm)` per backbone nucleus.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShiftOffsets {
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub cb: f64,
    pub c: f64,
}

impl ShiftOffsets {
    #[inline]
    pub fn get(&self, atom: &str) -> f64 {
        match atom {
            "H" => self.h,
            "N" => self.n,
            "CA" => self.ca,
            "CB" => self.cb,
            "C" => self.c,
            _ => 0.0,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.h.abs() < 1e-9
            && self.n.abs() < 1e-9
            && self.ca.abs() < 1e-9
            && self.cb.abs() < 1e-9
            && self.c.abs() < 1e-9
    }
}

impl fmt::Display for ShiftOffsets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dH={:+.3}, dN={:+.2}, dCA={:+.2}, dCB={:+.2}, dC={:+.2} ppm",
            self.h, self.n, self.ca, self.cb, self.c
        )
    }
}

/// Empirical BMRB training-set Proline occurrence fraction (`91 / 2508 ≈ 3.63%`, from
/// `data/priors/PROVENANCE.md`) already absorbed into pooled BMRB prior means.
const NATURAL_PRO_FRACTION: f64 = 91.0 / 2508.0;

/// Mixture weight (`0.85`) for the sterically upfield coil/turn/PPII pre-Proline state in
/// [`Priors::llr_shifted`], paired with `0.15` weight on the unshifted globular state (consistent
/// with the ~85% non-helical vs ~15% kinked alpha-helical distribution of proline-preceding
/// residues in globular proteins per MacArthur & Thornton, *J. Mol. Biol.* 1991, 218, 397–412) so
/// kinked alpha-helical pre-Proline residues do not suffer unbounded quadratic tail penalties.
pub const COVALENT_COIL_WEIGHT: f64 = 0.85;

/// Deterministic zero-mean primary-sequence covalent tripeptide `(seq[k-1], seq[k], seq[k+1])`
/// expected shift adjustments `(d_n, d_ca, d_cb, d_c)` in ppm for residue index `k`:
/// - **Pre-Proline (`seq[k+1] == "PRO"`) vs non-pre-Proline (`seq[k+1] != "PRO"`)**: steric and
///   tertiary-imide electronic upfield shifts on `CA(k)` and `C'(k)` (`(-1.95, -1.80) ppm` for
///   standard `X-PRO`, `(-1.00, -1.50) ppm` for `GLY-PRO`, `(-1.75, -1.70) ppm` for `PRO-PRO` per
///   Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81 & 6, 135–140; Schwarzinger et al., *J. Am.
///   Chem. Soc.* 2001, 123, 2970–2978), weighted by `COVALENT_COIL_WEIGHT = 0.85` (MacArthur &
///   Thornton, *J. Mol. Biol.* 1991, 218, 397–412) and centered relative to the empirical BMRB
///   training Proline occurrence `p = 91 / 2508` (`scale = 0.85 * (1 - p)` when `k+1 == PRO`,
///   `scale = -0.85 * p` otherwise) so `p * d_pre + (1 - p) * d_non == 0.0` identically and global
///   `13C` reference offset estimation remains unbiased.
/// - **`CB(k)` (`d_cb = 0.0`) & `15N(k)` (`d_n = 0.0`)**: unfolded pentapeptide `CB` pre-Proline
///   (`+0.25 ppm`) and `i-1` `15N` substituent offsets (Braun, Wider & Wüthrich, *J. Am. Chem.
///   Soc.* 1994, 116, 8466–8469; Schwarzinger et al., 2001) are dominated and inverted by folded
///   backbone `(phi, psi)` and hydrogen-bonding dispersion, and are kept at `0.0` per empirical
///   ablation.
#[inline]
pub fn delta_covalent(sequence: &[String], k: usize) -> (f64, f64, f64, f64) {
    if k + 1 >= sequence.len() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let res = sequence[k].as_str();
    if !STANDARD_RESIDUES.iter().any(|&(r, _)| r == res) {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let scale = if sequence[k + 1] == "PRO" {
        COVALENT_COIL_WEIGHT * (1.0 - NATURAL_PRO_FRACTION)
    } else {
        -COVALENT_COIL_WEIGHT * NATURAL_PRO_FRACTION
    };
    let (raw_ca, raw_c) = match res {
        "GLY" => (-1.00, -1.50),
        "PRO" => (-1.75, -1.70),
        _ => (-1.95, -1.80),
    };
    (0.0, scale * raw_ca, 0.0, scale * raw_c)
}

/// Extract the covalent shift adjustment in ppm for a named backbone atom (`"N"`, `"CA"`, `"CB"`, `"C"`).
#[inline]
pub fn covalent_atom_offset(cov: (f64, f64, f64, f64), atom: &str) -> f64 {
    match atom {
        "N" => cov.0,
        "CA" => cov.1,
        "CB" => cov.2,
        "C" => cov.3,
        _ => 0.0,
    }
}

/// Expected mean shift of the two-state pre-Proline mixture for a named backbone atom
/// (`delta_covalent` already returns the mixture expectation `COVALENT_COIL_WEIGHT * scale * raw`).
#[inline]
pub fn expected_covalent_atom_offset(cov: (f64, f64, f64, f64), atom: &str) -> f64 {
    covalent_atom_offset(cov, atom)
}

fn build_sequence_mixture(
    sequence: &[String],
    priors: &Priors,
    atom: &str,
    is_im1: bool,
) -> Vec<(f64, f64, f64)> {
    let mut counts: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for k in 1..sequence.len() {
        if sequence[k] == "PRO" {
            continue;
        }
        let target_idx = if is_im1 { k - 1 } else { k };
        let target = &sequence[target_idx];
        if priors.has(target, atom) {
            let d_cov = covalent_atom_offset(delta_covalent(sequence, target_idx), atom);
            let entry = counts.entry(target.as_str()).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += d_cov;
        }
    }
    if counts.is_empty() {
        for &(res, _) in &STANDARD_RESIDUES {
            if priors.has(res, atom) {
                counts.insert(res, (1, 0.0));
            }
        }
    }
    let total = counts.values().map(|&(c, _)| c).sum::<usize>().max(1) as f64;
    counts
        .into_iter()
        .filter_map(|(res, (cnt, cov_sum))| {
            priors.get(res, atom).map(|(mu, sd)| {
                let mean_cov = cov_sum / (cnt.max(1) as f64);
                (mu + mean_cov, sd, (cnt as f64) / total)
            })
        })
        .collect()
}

fn outlier_robust_mixture_ll(
    obs: &[(f64, bool)],
    mix0: &[(f64, f64, f64)],
    mixm1: &[(f64, f64, f64)],
    range_ppm: f64,
    offset: f64,
) -> f64 {
    const EPS: f64 = 0.05;
    let bg = EPS / range_ppm;
    let fg = 1.0 - EPS;
    let mut total = 0.0;
    for &(v, is_im1) in obs {
        let mix = if is_im1 { mixm1 } else { mix0 };
        let mut sum = 0.0;
        for &(mu, sd, w) in mix {
            let z = (v - (mu + offset)) / sd;
            sum += w * (-0.5 * z * z).exp() / (sd * TAU.sqrt());
        }
        total += (fg * sum + bg).max(1e-30).ln();
    }
    total
}

#[inline]
fn refine_grid_peak(scores: &[f64], best_idx: usize, min_d: f64, step: f64) -> f64 {
    let d0 = min_d + (best_idx as f64) * step;
    if best_idx == 0 || best_idx + 1 >= scores.len() {
        return d0;
    }
    let ym1 = scores[best_idx - 1];
    let y0 = scores[best_idx];
    let yp1 = scores[best_idx + 1];
    let denom = ym1 - 2.0 * y0 + yp1;
    if denom < -1e-12 {
        let delta = (0.5 * (ym1 - yp1) / denom).clamp(-0.5, 0.5);
        d0 + delta * step
    } else {
        d0
    }
}

/// Estimate global chemical-shift reference offsets `(dH, dN, dCA, dCB, dC)` maximizing the
/// outlier-robust marginal mixture log-likelihood of observed spin-system shifts against
/// sequence-weighted `Priors` (with joint aliphatic `dCA ≈ dCB` regularization and a deadband/BIC
/// gate so DSS-referenced spectra remain at exact `0.0 ppm`).
pub fn estimate_reference_offsets(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
) -> ShiftOffsets {
    if systems.is_empty() {
        return ShiftOffsets::default();
    }

    let slot_and_support = |ss: &SpinSystem| -> (usize, usize) {
        let mut slots = 0usize;
        let mut sup_sum = 0usize;
        for cands in ss.cands.values() {
            if let Some(&(_, sup)) = cands.iter().find(|&&(_, sup)| sup > 0) {
                slots += 1;
                sup_sum += sup;
            }
        }
        (sup_sum, slots)
    };
    let assignable = sequence
        .iter()
        .enumerate()
        .filter(|&(k, r)| k > 0 && r != "PRO")
        .count()
        .max(sequence.len());
    let max_selected = ((assignable * 4) / 3).max(24);
    let mut multi_slot: Vec<&SpinSystem> = systems
        .iter()
        .filter(|ss| slot_and_support(ss).1 >= 2)
        .collect();
    if multi_slot.len() > max_selected {
        multi_slot.sort_by_key(|b| std::cmp::Reverse(slot_and_support(b)));
        multi_slot.truncate(max_selected);
    }
    let mut selected: Vec<&SpinSystem> = if multi_slot.len() >= 8 {
        multi_slot
    } else {
        systems
            .iter()
            .filter(|ss| ss.has_observed_carbons())
            .collect()
    };
    if selected.len() > max_selected {
        selected.sort_by_key(|b| std::cmp::Reverse(slot_and_support(b)));
        selected.truncate(max_selected);
    }
    if selected.len() < 6 {
        return ShiftOffsets::default();
    }

    let collect_carbon = |atom: &str, lo: f64, hi: f64| -> (Vec<(f64, bool)>, f64) {
        let mut obs = Vec::new();
        let mut n0 = 0usize;
        let mut nm1 = 0usize;
        for ss in &selected {
            for (off, is_im1) in [(0, false), (-1, true)] {
                if let Some(list) = ss.cands.get(&(atom.to_string(), off)) {
                    if let Some(&(v, _)) = list.iter().find(|&&(_, sup)| sup > 0) {
                        if (lo..=hi).contains(&v) {
                            obs.push((v, is_im1));
                            if is_im1 {
                                nm1 += 1;
                            } else {
                                n0 += 1;
                            }
                        }
                    }
                }
            }
        }
        let scale = if obs.is_empty() {
            1.0
        } else {
            (n0.max(nm1).max(1) as f64) / (obs.len() as f64)
        };
        (obs, scale)
    };

    let (obs_ca, scale_ca) = collect_carbon("CA", 34.0, 76.0);
    let (obs_cb, scale_cb) = collect_carbon("CB", 8.0, 78.0);
    let (obs_c, scale_c) = collect_carbon("C", 162.0, 188.0);

    let obs_h: Vec<(f64, bool)> = selected
        .iter()
        .filter(|ss| (5.0..=12.5).contains(&ss.h))
        .map(|ss| (ss.h, false))
        .collect();
    let obs_n: Vec<(f64, bool)> = selected
        .iter()
        .filter(|ss| (95.0..=145.0).contains(&ss.n))
        .map(|ss| (ss.n, false))
        .collect();

    // 13C grid: [-6.0, +6.0] ppm in 0.05 ppm steps (index 120 == 0.0 ppm)
    const N_GRID_C: usize = 241;
    const ZERO_C: usize = 120;
    const MIN_C: f64 = -6.0;
    const STEP_C: f64 = 0.05;

    let eval_c_curve = |atom: &str, obs: &[(f64, bool)], scale: f64, range_ppm: f64| -> Vec<f64> {
        let mix0 = build_sequence_mixture(sequence, priors, atom, false);
        let mixm1 = build_sequence_mixture(sequence, priors, atom, true);
        (0..N_GRID_C)
            .map(|i| {
                let d = MIN_C + (i as f64) * STEP_C;
                scale * outlier_robust_mixture_ll(obs, &mix0, &mixm1, range_ppm, d)
                    - 0.5 * (d / 4.5).powi(2)
            })
            .collect()
    };

    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;

    // 1. Joint aliphatic (dCA, dCB) estimation with carrier-tether regularization dCA ≈ dCB.
    // Because alpha-helix vs beta-sheet secondary structure shifts CA and CB in OPPOSITE directions
    // whereas chemical-shift referencing shifts CA and CB in the SAME direction, coupling dCA ≈ dCB
    // cancels secondary-structure bias on DSS-referenced proteins while capturing TMS/carrier offsets.
    let (ca_off, cb_off) = if obs_ca.len() >= 8 && obs_cb.len() >= 8 {
        let l_ca = eval_c_curve("CA", &obs_ca, scale_ca, 45.0);
        let l_cb = eval_c_curve("CB", &obs_cb, scale_cb, 65.0);
        let null_score = l_ca[ZERO_C] + l_cb[ZERO_C];
        let mut best_score = f64::NEG_INFINITY;
        let mut best_i = ZERO_C;
        let mut best_j = ZERO_C;
        for (i, &sca) in l_ca.iter().enumerate() {
            let di = MIN_C + (i as f64) * STEP_C;
            for (j, &scb) in l_cb.iter().enumerate() {
                let dj = MIN_C + (j as f64) * STEP_C;
                let tether = -0.5 * ((di - dj) / 0.25).powi(2);
                let sc = sca + scb + tether;
                if sc > best_score {
                    best_score = sc;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        // Refine 1D slices along best_j / best_i
        let slice_ca: Vec<f64> = (0..N_GRID_C)
            .map(|i| {
                let di = MIN_C + (i as f64) * STEP_C;
                let dj = MIN_C + (best_j as f64) * STEP_C;
                l_ca[i] + l_cb[best_j] - 0.5 * ((di - dj) / 0.25).powi(2)
            })
            .collect();
        let slice_cb: Vec<f64> = (0..N_GRID_C)
            .map(|j| {
                let di = MIN_C + (best_i as f64) * STEP_C;
                let dj = MIN_C + (j as f64) * STEP_C;
                l_ca[best_i] + l_cb[j] - 0.5 * ((di - dj) / 0.25).powi(2)
            })
            .collect();
        let d_ca = refine_grid_peak(&slice_ca, best_i, MIN_C, STEP_C);
        let d_cb = refine_grid_peak(&slice_cb, best_j, MIN_C, STEP_C);
        if d_ca.abs().max(d_cb.abs()) >= 0.85 && (best_score - null_score) >= 12.0 {
            (round3(d_ca), round3(d_cb))
        } else {
            (0.0, 0.0)
        }
    } else if obs_ca.len() >= 8 {
        let l_ca = eval_c_curve("CA", &obs_ca, scale_ca, 45.0);
        let (best_i, &best_sc) = l_ca
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((ZERO_C, &0.0));
        let d_ca = refine_grid_peak(&l_ca, best_i, MIN_C, STEP_C);
        // Without CB to cancel secondary-structure shift propensities, require a clear offset
        if d_ca.abs() >= 1.15 && (best_sc - l_ca[ZERO_C]) >= 12.0 {
            let r = round3(d_ca);
            (r, r)
        } else {
            (0.0, 0.0)
        }
    } else if obs_cb.len() >= 8 {
        let l_cb = eval_c_curve("CB", &obs_cb, scale_cb, 65.0);
        let (best_j, &best_sc) = l_cb
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((ZERO_C, &0.0));
        let d_cb = refine_grid_peak(&l_cb, best_j, MIN_C, STEP_C);
        if d_cb.abs() >= 0.85 && (best_sc - l_cb[ZERO_C]) >= 12.0 {
            let r = round3(d_cb);
            (r, r)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    // 2. Carbonyl dC estimation
    let c_off = if obs_c.len() >= 8 {
        let l_c = eval_c_curve("C", &obs_c, scale_c, 25.0);
        let (best_i, &best_sc) = l_c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((ZERO_C, &0.0));
        let d_c = refine_grid_peak(&l_c, best_i, MIN_C, STEP_C);
        if d_c.abs() >= 0.45 && (best_sc - l_c[ZERO_C]) >= 5.0 {
            round3(d_c)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 3. Amide dH and dN estimation (conservative deadband so fold-dependent dispersion stays 0.0)
    let h_off = if obs_h.len() >= 12 {
        const N_GRID_H: usize = 301;
        const ZERO_H: usize = 150;
        const MIN_H: f64 = -1.5;
        const STEP_H: f64 = 0.01;
        let mix_h = build_sequence_mixture(sequence, priors, "H", false);
        let l_h: Vec<f64> = (0..N_GRID_H)
            .map(|i| {
                let d = MIN_H + (i as f64) * STEP_H;
                outlier_robust_mixture_ll(&obs_h, &mix_h, &mix_h, 6.0, d) - 0.5 * (d / 0.8).powi(2)
            })
            .collect();
        let (best_i, &best_sc) = l_h
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((ZERO_H, &0.0));
        let d_h = refine_grid_peak(&l_h, best_i, MIN_H, STEP_H);
        if d_h.abs() >= 0.22 && (best_sc - l_h[ZERO_H]) >= 6.0 {
            round3(d_h)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let n_off = if obs_n.len() >= 12 {
        let mix_n = build_sequence_mixture(sequence, priors, "N", false);
        let l_n: Vec<f64> = (0..N_GRID_C)
            .map(|i| {
                let d = MIN_C + (i as f64) * STEP_C;
                outlier_robust_mixture_ll(&obs_n, &mix_n, &mix_n, 40.0, d) - 0.5 * (d / 4.0).powi(2)
            })
            .collect();
        let (best_i, &best_sc) = l_n
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((ZERO_C, &0.0));
        let d_n = refine_grid_peak(&l_n, best_i, MIN_C, STEP_C);
        if d_n.abs() >= 1.25 && (best_sc - l_n[ZERO_C]) >= 6.0 {
            round3(d_n)
        } else {
            0.0
        }
    } else {
        0.0
    };

    ShiftOffsets {
        h: h_off,
        n: n_off,
        ca: ca_off,
        cb: cb_off,
        c: c_off,
    }
}

fn bio_plausible_bounds(atom: &str) -> Option<(f64, f64)> {
    match atom {
        "CA" => Some((38.0, 72.0)),
        "CB" => Some((12.0, 78.0)),
        "C" => Some((165.0, 186.0)),
        "N" => Some((95.0, 145.0)),
        "H" => Some((5.0, 12.5)),
        "NE" => Some((75.0, 96.0)),
        "NZ" => Some((25.0, 45.0)),
        _ if atom.starts_with('C') => Some((5.0, 186.0)),
        _ if atom.starts_with('N') => Some((65.0, 185.0)),
        _ if atom.starts_with('H') => Some((-2.5, 15.0)),
        _ => None,
    }
}

fn mean_and_pop_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Recompute backbone `(mean, sd)` per standard residue/atom from NMR-STAR files
/// using bio-plausible shift bounds + 3-sigma outlier trimming (with pooled bimodal `CYS CB`
/// and literature fallback strictly when `N < min_count`).
pub fn compute_priors<P: AsRef<Path>>(star_paths: &[P], min_count: usize) -> Priors {
    let base = Priors::literature_fallback();
    let mut pool: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for p in star_paths {
        let Ok(ref_data) = read_star(p.as_ref(), None) else {
            continue;
        };
        for (&(idx, ref atom), &v) in &ref_data.shifts {
            let res = &ref_data.sequence[idx];
            if base.has(res, atom) {
                if let Some((lo, hi)) = bio_plausible_bounds(atom) {
                    if v >= lo && v <= hi {
                        pool.entry((res.clone(), atom.clone())).or_default().push(v);
                    }
                }
            }
        }
    }

    let mut table = base.table.clone();
    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
    for ((res, atom), vals) in pool {
        if vals.len() < min_count {
            continue;
        }
        let trimmed: Vec<f64> = if res == "CYS" && atom == "CB" {
            // Pool reduced thiol (~28 ppm) and disulfide-oxidized (~41 ppm) CYS CB
            vals
        } else {
            let (mu0, sd0) = mean_and_pop_std(&vals);
            if sd0 > 0.0 {
                vals.into_iter()
                    .filter(|&x| (x - mu0).abs() <= 3.0 * sd0)
                    .collect()
            } else {
                vals
            }
        };
        if !trimmed.is_empty() {
            let (mu, sd) = mean_and_pop_std(&trimmed);
            if let Some(res_map) = table.get_mut(&res) {
                res_map.insert(atom, (round3(mu), round3(sd.max(0.05))));
            }
        }
    }

    Priors { table }
}

/// Recompute all-atom `(mean, sd)` (backbone + sidechains) from NMR-STAR files.
pub fn compute_all_atom_priors<P: AsRef<Path>>(star_paths: &[P], min_count: usize) -> Priors {
    let mut out = compute_priors(star_paths, min_count);
    let mut pool: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for p in star_paths {
        let Ok(ref_data) = read_star_all_atoms(p.as_ref(), None) else {
            continue;
        };
        for (&(idx, ref atom), &v) in &ref_data.shifts {
            let res = &ref_data.sequence[idx];
            if is_standard_residue(res) && !ATOMS.contains(&atom.as_str()) {
                if let Some((lo, hi)) = bio_plausible_bounds(atom) {
                    if v >= lo && v <= hi {
                        pool.entry((res.clone(), atom.clone())).or_default().push(v);
                    }
                }
            }
        }
    }
    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
    for ((res, atom), vals) in pool {
        if vals.len() < min_count {
            continue;
        }
        let (mu0, sd0) = mean_and_pop_std(&vals);
        let trimmed: Vec<f64> = if sd0 > 0.0 {
            vals.into_iter()
                .filter(|&x| (x - mu0).abs() <= 3.0 * sd0)
                .collect()
        } else {
            vals
        };
        if !trimmed.is_empty() {
            let (mu, sd) = mean_and_pop_std(&trimmed);
            if let Some(res_map) = out.table.get_mut(&res) {
                res_map.insert(atom, (round3(mu), round3(sd.max(0.05))));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::{simulate, SimConfig};
    use crate::spinsystems::{build_spin_systems, Tolerances};

    #[test]
    fn test_reference_offsets_dss_and_shifted() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let priors = Priors::bmrb();
        let rd = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None).unwrap();
        let (pks, _) = simulate(
            &rd,
            &SimConfig {
                seed: 0,
                ..SimConfig::default()
            },
        );
        let sys = build_spin_systems(&pks, &Tolerances::default(), 3, true).unwrap();
        let off_dss = estimate_reference_offsets(&sys, &rd.sequence, &priors);
        assert!(
            off_dss.is_zero(),
            "expected 0.0 ppm on DSS ubiquitin, got {off_dss}"
        );

        let mut shifted_pks = pks;
        for p in &mut shifted_pks {
            if let Some(c) = p.c.as_mut() {
                *c -= 2.75;
            }
        }
        let sys_tms = build_spin_systems(&shifted_pks, &Tolerances::default(), 3, true).unwrap();
        let off_tms = estimate_reference_offsets(&sys_tms, &rd.sequence, &priors);
        assert!((off_tms.ca - (-2.75)).abs() <= 0.55, "dCA={}", off_tms.ca);
        assert!((off_tms.cb - (-2.75)).abs() <= 0.55, "dCB={}", off_tms.cb);
        assert!((off_tms.c - (-2.75)).abs() <= 0.55, "dC={}", off_tms.c);
    }

    #[test]
    fn test_delta_covalent_pre_proline_and_two_state_mixture() {
        let seq: Vec<String> = ["MET", "GLU", "PRO", "GLY", "PRO", "PRO", "ASP", "VAL"]
            .into_iter()
            .map(String::from)
            .collect();

        // Standard residue before non-Pro (MET before GLU) vs before Pro (GLU before PRO)
        let non_pro = delta_covalent(&seq, 0);
        let pre_pro = delta_covalent(&seq, 1);
        let p = NATURAL_PRO_FRACTION;
        let a = COVALENT_COIL_WEIGHT;
        assert_eq!(non_pro.0, 0.0);
        assert_eq!(non_pro.2, 0.0);
        assert!((non_pro.1 - (-a * p * (-1.95))).abs() < 1e-12);
        assert!((non_pro.3 - (-a * p * (-1.80))).abs() < 1e-12);
        assert_eq!(pre_pro.0, 0.0);
        assert!((pre_pro.1 - (a * (1.0 - p) * (-1.95))).abs() < 1e-12);
        assert_eq!(pre_pro.2, 0.0);
        assert!((pre_pro.3 - (a * (1.0 - p) * (-1.80))).abs() < 1e-12);
        // Exact zero-mean centering across natural BMRB proline occurrence p = 91 / 2508
        assert!((p * pre_pro.1 + (1.0 - p) * non_pro.1).abs() < 1e-12);
        assert!((p * pre_pro.3 + (1.0 - p) * non_pro.3).abs() < 1e-12);

        // GLY before PRO (k = 3) and PRO before PRO (k = 4)
        let gly_pro = delta_covalent(&seq, 3);
        assert!((gly_pro.1 - (-a * (1.0 - p))).abs() < 1e-12);
        assert_eq!(gly_pro.2, 0.0);
        assert!((gly_pro.3 - (a * (1.0 - p) * (-1.50))).abs() < 1e-12);

        let pro_pro = delta_covalent(&seq, 4);
        assert!((pro_pro.1 - (a * (1.0 - p) * (-1.75))).abs() < 1e-12);
        assert_eq!(pro_pro.2, 0.0);
        assert!((pro_pro.3 - (a * (1.0 - p) * (-1.70))).abs() < 1e-12);

        // C-terminal residue (k = 7) and out-of-bounds return exact zeros
        assert_eq!(delta_covalent(&seq, 7), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(delta_covalent(&seq, 99), (0.0, 0.0, 0.0, 0.0));

        // Priors::llr_shifted preserves exact background cancellation across competing slots
        // and protects helical pre-Proline outliers (e.g. Calmodulin Phe65 CA = 63.59 ppm)
        let priors = Priors::bmrb();
        assert!(
            (priors.llr_shifted("GLU", "CA", 55.4, 0.0) - priors.llr("GLU", "CA", 55.4)).abs()
                < 1e-12
        );
        assert_eq!(
            priors.llr_shifted("GLY", "CB", 30.0, -1.0),
            priors.llr("GLY", "CB", 30.0)
        );
        // Upfield pre-Proline CA (54.2 ppm on GLU) receives a strong LLR boost vs non-pre-Proline GLU
        assert!(
            priors.llr_shifted("GLU", "CA", 54.2, pre_pro.1)
                > priors.llr_shifted("GLU", "CA", 54.2, non_pro.1) + 0.65
        );
        // Strongly helical pre-Proline CA (63.59 ppm on PHE) has penalty bounded by ~ -ln(0.15) ≈ 1.90 nats
        // instead of ~5.0 nats under a rigid single-Gaussian pre-Proline shift
        let phe_pro_ca = a * (1.0 - p) * (-1.95);
        let hel_drop =
            priors.llr("PHE", "CA", 63.59) - priors.llr_shifted("PHE", "CA", 63.59, phe_pro_ca);
        assert!(hel_drop < 1.95, "hel_drop={hel_drop}");
    }
}
