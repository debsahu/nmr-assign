//! Empirical BMRB chemical shift priors per residue type and atom.
//!
//! Provenance: Clean-room original implementation of `docs/specs/03-assignment.md` section 3,
//! shipping empirical `(mean, sd)` priors in ppm computed from 21 DSS-referenced public
//! BioMagResBank (BMRB) NMR-STAR depositions (`data/priors/bmrb_priors.json` &
//! `data/priors/PROVENANCE.md`; Ulrich et al., *Nucleic Acids Res.* 2008, 36, D402–D408;
//! referencing QC per Zhang, Neal & Wishart, *J. Biomol. NMR* 2003, 25, 173–195), strictly
//! disjoint from the benchmark suite (`data/benchmark/manifest.tsv`), with literature fallback
//! strictly when `N < 5` (Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::nmrstar::{read_star, read_star_all_atoms};

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
