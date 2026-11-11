//! Chemical shift priors per residue type and backbone atom.
//!
//! Provenance: Original implementation of `docs/specs/03-assignment.md` section 3.
//! Ships provisional values approximating published BMRB-wide statistics (mean, sd in ppm);
//! `compute_priors` recomputes them from public BMRB NMR-STAR files with a robust 3-MAD trim.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

use crate::nmrstar::read_star;

pub const ATOMS: [&str; 5] = ["H", "N", "CA", "CB", "C"];

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

#[derive(Clone, Debug)]
pub struct Priors {
    pub table: BTreeMap<String, BTreeMap<String, (f64, f64)>>,
}

impl Default for Priors {
    fn default() -> Self {
        Self::provisional()
    }
}

impl Priors {
    /// Provisional BMRB-wide statistics (mean, sd in ppm).
    pub fn provisional() -> Self {
        let raw: &[(&str, &[(&str, (f64, f64))])] = &[
            ("ALA", &[("CA", (53.1, 2.0)), ("CB", (19.0, 1.8)), ("C", (177.8, 2.1)), ("N", (123.2, 3.5)), ("H", (8.19, 0.6))]),
            ("ARG", &[("CA", (56.8, 2.3)), ("CB", (30.7, 1.8)), ("C", (176.4, 2.0)), ("N", (120.8, 3.6)), ("H", (8.24, 0.6))]),
            ("ASN", &[("CA", (53.5, 1.9)), ("CB", (38.7, 1.7)), ("C", (175.3, 1.8)), ("N", (118.9, 4.0)), ("H", (8.33, 0.6))]),
            ("ASP", &[("CA", (54.7, 2.0)), ("CB", (40.9, 1.6)), ("C", (176.4, 1.8)), ("N", (120.6, 3.9)), ("H", (8.30, 0.6))]),
            ("CYS", &[("CA", (58.0, 3.4)), ("CB", (32.0, 5.5)), ("C", (174.9, 2.1)), ("N", (120.0, 4.4)), ("H", (8.39, 0.7))]),
            ("GLN", &[("CA", (56.6, 2.1)), ("CB", (29.2, 1.8)), ("C", (176.3, 2.0)), ("N", (119.9, 3.5)), ("H", (8.21, 0.6))]),
            ("GLU", &[("CA", (57.3, 2.1)), ("CB", (30.0, 1.7)), ("C", (176.9, 2.0)), ("N", (120.7, 3.5)), ("H", (8.33, 0.6))]),
            ("GLY", &[("CA", (45.4, 1.3)), ("C", (173.9, 1.9)), ("N", (109.6, 3.8)), ("H", (8.33, 0.6))]),
            ("HIS", &[("CA", (56.5, 2.3)), ("CB", (30.2, 2.1)), ("C", (175.2, 2.0)), ("N", (119.7, 4.0)), ("H", (8.25, 0.7))]),
            ("ILE", &[("CA", (61.6, 2.7)), ("CB", (38.6, 2.0)), ("C", (175.8, 2.0 - 0.1)), ("N", (121.4, 4.2)), ("H", (8.28, 0.7))]),
            ("LEU", &[("CA", (55.6, 2.1)), ("CB", (42.3, 1.9)), ("C", (177.0, 2.0)), ("N", (121.8, 3.9)), ("H", (8.22, 0.6))]),
            ("LYS", &[("CA", (56.9, 2.2)), ("CB", (32.8, 1.8)), ("C", (176.6, 2.0)), ("N", (121.0, 3.7)), ("H", (8.18, 0.6))]),
            ("MET", &[("CA", (56.1, 2.2)), ("CB", (32.9, 2.2)), ("C", (176.2, 2.1)), ("N", (120.1, 3.5)), ("H", (8.26, 0.6))]),
            ("PHE", &[("CA", (58.1, 2.6)), ("CB", (40.0, 2.0)), ("C", (175.5, 2.0)), ("N", (120.4, 4.1)), ("H", (8.35, 0.7))]),
            ("PRO", &[("CA", (63.3, 1.5)), ("CB", (31.9, 1.2)), ("C", (176.7, 1.6)), ("N", (134.0, 6.0))]),
            ("SER", &[("CA", (58.7, 2.1)), ("CB", (63.8, 1.5)), ("C", (174.6, 1.8)), ("N", (116.3, 3.5)), ("H", (8.28, 0.6))]),
            ("THR", &[("CA", (62.2, 2.6)), ("CB", (69.7, 1.7)), ("C", (174.5, 1.8)), ("N", (115.4, 4.7)), ("H", (8.24, 0.6))]),
            ("TRP", &[("CA", (57.7, 2.6)), ("CB", (30.1, 2.0)), ("C", (176.1, 2.0)), ("N", (121.6, 4.1)), ("H", (8.29, 0.8))]),
            ("TYR", &[("CA", (58.1, 2.6)), ("CB", (39.3, 2.2)), ("C", (175.4, 2.0)), ("N", (120.5, 4.1)), ("H", (8.30, 0.7))]),
            ("VAL", &[("CA", (62.5, 2.9)), ("CB", (32.7, 1.8)), ("C", (175.7, 1.9)), ("N", (121.1, 4.5)), ("H", (8.29, 0.7))]),
        ];
        let mut table = BTreeMap::new();
        for &(res, atoms) in raw {
            let mut m = BTreeMap::new();
            for &(atom, (mu, sd)) in atoms {
                // Keep exact f64 literal parity (e.g. 1.9 for ILE C)
                let sd_clean = if res == "ILE" && atom == "C" { 1.9 } else { sd };
                m.insert(atom.to_string(), (mu, sd_clean));
            }
            table.insert(res.to_string(), m);
        }
        Self { table }
    }

    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("failed to read priors JSON: {e}"))?;
        let raw: BTreeMap<String, BTreeMap<String, [f64; 2]>> =
            serde_json::from_str(&content).map_err(|e| format!("invalid priors JSON: {e}"))?;
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

    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let raw: BTreeMap<&String, BTreeMap<&String, [f64; 2]>> = self
            .table
            .iter()
            .map(|(r, d)| {
                (
                    r,
                    d.iter().map(|(a, &(mu, sd))| (a, [mu, sd])).collect(),
                )
            })
            .collect();
        let text = serde_json::to_string_pretty(&raw)
            .map_err(|e| format!("failed to serialize priors: {e}"))?;
        fs::write(path.as_ref(), text).map_err(|e| format!("failed to write priors JSON: {e}"))
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
        for d in self.table.values() {
            if let Some(&(mu, sd)) = d.get(atom) {
                vals.push(logpdf(v, mu, sd));
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

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn mean_and_pop_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Recompute `(mean, sd)` per standard residue/atom from NMR-STAR files using a 3-MAD trim.
pub fn compute_priors<P: AsRef<Path>>(star_paths: &[P], min_count: usize) -> Priors {
    let base = Priors::provisional();
    let mut pool: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for p in star_paths {
        let Ok(ref_data) = read_star(p.as_ref(), None) else {
            continue;
        };
        for (&(idx, ref atom), &v) in &ref_data.shifts {
            let res = &ref_data.sequence[idx];
            // Only collect valid standard residue/atom combinations present in the topology
            if base.has(res, atom) {
                pool.entry((res.clone(), atom.clone())).or_default().push(v);
            }
        }
    }

    let mut table = base.table.clone();
    for ((res, atom), mut vals) in pool {
        if vals.len() < min_count {
            continue;
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        let med = median(&vals);
        let mut abs_dev: Vec<f64> = vals.iter().map(|&x| (x - med).abs()).collect();
        abs_dev.sort_by(|a, b| a.total_cmp(b));
        let mut mad = 1.4826 * median(&abs_dev);
        if mad == 0.0 {
            mad = mean_and_pop_std(&vals).1;
        }
        let trimmed: Vec<f64> = vals
            .into_iter()
            .filter(|&x| (x - med).abs() <= 3.0 * mad)
            .collect();
        if !trimmed.is_empty() {
            let (mu, sd) = mean_and_pop_std(&trimmed);
            let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
            if let Some(res_map) = table.get_mut(&res) {
                res_map.insert(atom, (round3(mu), round3(sd.max(0.05))));
            }
        }
    }

    Priors { table }
}
