//! Exact mass, NIST/IUPAC isotopic fine distribution envelope, ESI/MALDI adduct calculator,
//! MS bond-cleavage fragmentation, and physicochemical property suite (Crippen cLogP, Ertl TPSA,
//! rotatable bonds, ESOL logS, Joback thermodynamics, and pKa sites).
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules:
//! - Coursey, J. S. et al. *NIST Standard Reference Database 144: Atomic Weights and Isotopic Compositions* (IUPAC CIAAW).
//! - Wildman, S. A.; Crippen, G. M. *J. Chem. Inf. Comput. Sci.* **1999**, 39, 868–873 (cLogP & MR).
//! - Ertl, P.; Rohde, B.; Selzer, P. *J. Med. Chem.* **2000**, 43, 3714–3717 (TPSA).
//! - Delaney, J. S. *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000–1005 (ESOL logS).
//! - Joback, K. G.; Reid, R. C. *Chem. Eng. Commun.* **1987**, 57, 233–243 (group contribution thermo).
//! - Ropp, P. J. et al. *J. Cheminform.* **2019**, 11, 14 (Dimorphite-DL pKa site table, Apache-2.0).

use crate::octet_analyt::{atomic_number, AnalytMol};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// CODATA 2022 electron rest mass in unified atomic mass units (`u`).
pub const ELECTRON_MASS_U: f64 = 5.485_799_090_441e-4;
const TRITIUM_MASS_U: f64 = 3.016_049_281_32;

/// Natural isotopes `(mass_u, abundance_percent)` from NIST SRD 144 / IUPAC CIAAW (`isotopes.ts`).
pub fn natural_isotopes(symbol: &str) -> Option<&'static [(f64, f64)]> {
    match symbol {
        "H" => Some(&[(1.00782503223, 99.9885), (2.01410177812, 0.0115)]),
        "He" => Some(&[(3.0160293201, 0.000134), (4.00260325413, 99.999866)]),
        "Li" => Some(&[(6.0151228874, 7.59), (7.0160034366, 92.41)]),
        "Be" => Some(&[(9.012183065, 100.0)]),
        "B" => Some(&[(10.01293695, 19.9), (11.00930536, 80.1)]),
        "C" => Some(&[(12.0000000, 98.93), (13.00335483507, 1.07)]),
        "N" => Some(&[(14.00307400443, 99.636), (15.00010889888, 0.364)]),
        "O" => Some(&[
            (15.99491461957, 99.757),
            (16.99913175650, 0.038),
            (17.99915961286, 0.205),
        ]),
        "F" => Some(&[(18.99840316273, 100.0)]),
        "Ne" => Some(&[
            (19.9924401762, 90.48),
            (20.993846685, 0.27),
            (21.991385114, 9.25),
        ]),
        "Na" => Some(&[(22.9897692820, 100.0)]),
        "Mg" => Some(&[
            (23.985041697, 78.99),
            (24.985836976, 10.0),
            (25.982592968, 11.01),
        ]),
        "Al" => Some(&[(26.98153853, 100.0)]),
        "Si" => Some(&[
            (27.97692653465, 92.223),
            (28.97649466490, 4.685),
            (29.973770136, 3.092),
        ]),
        "P" => Some(&[(30.97376199842, 100.0)]),
        "S" => Some(&[
            (31.9720711744, 94.99),
            (32.9714589098, 0.75),
            (33.967867004, 4.25),
            (35.96708071, 0.01),
        ]),
        "Cl" => Some(&[(34.968852682, 75.76), (36.965902602, 24.24)]),
        "Ar" => Some(&[
            (35.967545105, 0.3336),
            (37.96273211, 0.0629),
            (39.9623831237, 99.6035),
        ]),
        "K" => Some(&[
            (38.9637064864, 93.2581),
            (39.963998166, 0.0117),
            (40.9618252579, 6.7302),
        ]),
        "Ca" => Some(&[
            (39.962590863, 96.941),
            (41.95861783, 0.647),
            (42.95876644, 0.135),
            (43.95548156, 2.086),
            (45.9536890, 0.004),
            (47.95252276, 0.187),
        ]),
        "Sc" => Some(&[(44.95590828, 100.0)]),
        "Ti" => Some(&[
            (45.95262772, 8.25),
            (46.95175879, 7.44),
            (47.94794198, 73.72),
            (48.94786568, 5.41),
            (49.94478689, 5.18),
        ]),
        "V" => Some(&[(49.94715601, 0.25), (50.94395704, 99.75)]),
        "Cr" => Some(&[
            (49.94604183, 4.345),
            (51.94050623, 83.789),
            (52.94064815, 9.501),
            (53.93887916, 2.365),
        ]),
        "Mn" => Some(&[(54.93804391, 100.0)]),
        "Fe" => Some(&[
            (53.93960899, 5.845),
            (55.93493633, 91.754),
            (56.93539284, 2.119),
            (57.93327443, 0.282),
        ]),
        "Co" => Some(&[(58.93319429, 100.0)]),
        "Ni" => Some(&[
            (57.93534241, 68.077),
            (59.93078588, 26.223),
            (60.93105557, 1.1399),
            (61.92834537, 3.6346),
            (63.92796682, 0.9255),
        ]),
        "Cu" => Some(&[(62.92959772, 69.15), (64.92778970, 30.85)]),
        "Zn" => Some(&[
            (63.92914201, 49.17),
            (65.92603381, 27.73),
            (66.92712775, 4.04),
            (67.92484455, 18.45),
            (69.9253192, 0.61),
        ]),
        "Ga" => Some(&[(68.9255735, 60.108), (70.92470258, 39.892)]),
        "Ge" => Some(&[
            (69.92424875, 20.57),
            (71.922075826, 27.45),
            (72.923458956, 7.75),
            (73.921177761, 36.5),
            (75.921402726, 7.73),
        ]),
        "As" => Some(&[(74.92159457, 100.0)]),
        "Se" => Some(&[
            (73.922475934, 0.89),
            (75.919213704, 9.37),
            (76.919914154, 7.63),
            (77.91730928, 23.77),
            (79.9165218, 49.61),
            (81.9166995, 8.73),
        ]),
        "Br" => Some(&[(78.9183376, 50.69), (80.9162897, 49.31)]),
        "Kr" => Some(&[
            (77.92036494, 0.355),
            (79.91637808, 2.286),
            (81.91348273, 11.593),
            (82.91412716, 11.5),
            (83.9114977282, 56.987),
            (85.9106106269, 17.279),
        ]),
        "Rb" => Some(&[(84.9117897379, 72.17), (86.9091805310, 27.83)]),
        "Sr" => Some(&[
            (83.9134191, 0.56),
            (85.9092606, 9.86),
            (86.9088775, 7.0),
            (87.9056125, 82.58),
        ]),
        "Y" => Some(&[(88.9058403, 100.0)]),
        "Zr" => Some(&[
            (89.9046977, 51.45),
            (90.9056396, 11.22),
            (91.9050347, 17.15),
            (93.9063108, 17.38),
            (95.9082714, 2.8),
        ]),
        "Nb" => Some(&[(92.9063730, 100.0)]),
        "Mo" => Some(&[
            (91.90680796, 14.53),
            (93.90508490, 9.15),
            (94.90583877, 15.84),
            (95.90467612, 16.67),
            (96.90601812, 9.6),
            (97.90540482, 24.39),
            (99.9074718, 9.82),
        ]),
        "Ru" => Some(&[
            (95.90759025, 5.54),
            (97.9052868, 1.87),
            (98.9059341, 12.76),
            (99.9042143, 12.6),
            (100.9055769, 17.06),
            (101.9043441, 31.55),
            (103.9054275, 18.62),
        ]),
        "Rh" => Some(&[(102.9054980, 100.0)]),
        "Pd" => Some(&[
            (101.9056022, 1.02),
            (103.9040305, 11.14),
            (104.9050796, 22.33),
            (105.9034804, 27.33),
            (107.9038916, 26.46),
            (109.90517220, 11.72),
        ]),
        "Ag" => Some(&[(106.9050916, 51.839), (108.9047553, 48.161)]),
        "Cd" => Some(&[
            (105.9064599, 1.25),
            (107.9041834, 0.89),
            (109.90300661, 12.49),
            (110.90418287, 12.8),
            (111.90276287, 24.13),
            (112.90440813, 12.22),
            (113.90336509, 28.73),
            (115.90476315, 7.49),
        ]),
        "In" => Some(&[(112.90406184, 4.29), (114.903878776, 95.71)]),
        "Sn" => Some(&[
            (111.90482387, 0.97),
            (113.9027827, 0.66),
            (114.903344699, 0.34),
            (115.90174280, 14.54),
            (116.90295398, 7.68),
            (117.90160657, 24.22),
            (118.90331117, 8.59),
            (119.90220163, 32.58),
            (121.9034438, 4.63),
            (123.9052766, 5.79),
        ]),
        "Sb" => Some(&[(120.9038120, 57.21), (122.9042132, 42.79)]),
        "Te" => Some(&[
            (119.9040593, 0.09),
            (121.9030435, 2.55),
            (122.9042698, 0.89),
            (123.9028171, 4.74),
            (124.9044299, 7.07),
            (125.9033109, 18.84),
            (127.90446128, 31.74),
            (129.906222748, 34.08),
        ]),
        "I" => Some(&[(126.9044719, 100.0)]),
        "Xe" => Some(&[
            (123.9058920, 0.0952),
            (125.9042983, 0.089),
            (127.9035310, 1.9102),
            (128.9047808611, 26.4006),
            (129.903509349, 4.071),
            (130.90508406, 21.2324),
            (131.9041550856, 26.9086),
            (133.90539466, 10.4357),
            (135.907214484, 8.8573),
        ]),
        "Cs" => Some(&[(132.9054519610, 100.0)]),
        "Ba" => Some(&[
            (129.9063207, 0.106),
            (131.9050611, 0.101),
            (133.90450818, 2.417),
            (134.90568838, 6.592),
            (135.90457573, 7.854),
            (136.90582714, 11.232),
            (137.90524700, 71.698),
        ]),
        "La" => Some(&[(137.9071149, 0.08881), (138.9063563, 99.91119)]),
        "Pt" => Some(&[
            (189.9599297, 0.012),
            (191.9610387, 0.782),
            (193.9626809, 32.86),
            (194.9647917, 33.78),
            (195.96495209, 25.21),
            (197.9678949, 7.356),
        ]),
        "Au" => Some(&[(196.96656879, 100.0)]),
        "Hg" => Some(&[
            (195.9658326, 0.15),
            (197.96676860, 9.97),
            (198.96828064, 16.87),
            (199.96832659, 23.1),
            (200.97030284, 13.18),
            (201.97064340, 29.86),
            (203.97349398, 6.87),
        ]),
        "Tl" => Some(&[(202.9723446, 29.52), (204.9744278, 70.48)]),
        "Pb" => Some(&[
            (203.9730440, 1.4),
            (205.9744657, 24.1),
            (206.9758973, 22.1),
            (207.9766525, 52.4),
        ]),
        "Bi" => Some(&[(208.9803991, 100.0)]),
        "Th" => Some(&[(232.0380558, 100.0)]),
        "Pa" => Some(&[(231.0358842, 100.0)]),
        "U" => Some(&[
            (234.0409523, 0.0054),
            (235.0439301, 0.7204),
            (238.0507884, 99.2742),
        ]),
        _ => None,
    }
}

/// Resolve isotopes for an element key (`"C"`, `"Cl"`, `"D"`, `"T"`, or `"[13C]"`).
pub fn isotopes_of_key(key: &str) -> Option<Vec<(f64, f64)>> {
    if let Some(iso) = natural_isotopes(key) {
        return Some(iso.to_vec());
    }
    if key == "D" {
        return Some(vec![(2.01410177812, 100.0)]);
    }
    if key == "T" {
        return Some(vec![(TRITIUM_MASS_U, 100.0)]);
    }
    if let Some(inner) = key.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let split_at = inner.find(|c: char| c.is_ascii_alphabetic())?;
        let mass_num = inner[..split_at].parse::<u16>().ok()?;
        let el = &inner[split_at..];
        if el == "H" && mass_num == 3 {
            return Some(vec![(TRITIUM_MASS_U, 100.0)]);
        }
        let table = natural_isotopes(el)?;
        let &(m, _) = table
            .iter()
            .find(|&&(m, _)| (m.round() as u16) == mass_num)?;
        return Some(vec![(m, 100.0)]);
    }
    None
}

/// Abundance-weighted average atomic weight of an element or pure isotope key.
pub fn average_atomic_mass(key: &str) -> Option<f64> {
    let iso = isotopes_of_key(key)?;
    let num: f64 = iso.iter().map(|&(m, a)| m * a).sum();
    let den: f64 = iso.iter().map(|&(_, a)| a).sum();
    if den > 0.0 {
        Some(num / den)
    } else {
        None
    }
}

/// Monoisotopic mass of an element or pure isotope key (mass of its most abundant natural isotope).
pub fn monoisotopic_atomic_mass(key: &str) -> Option<f64> {
    let iso = isotopes_of_key(key)?;
    iso.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(m, _)| m)
}

fn base_element_of_key(key: &str) -> &str {
    if key == "D" || key == "T" {
        return "H";
    }
    if let Some(inner) = key.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        if let Some(pos) = inner.find(|c: char| c.is_ascii_alphabetic()) {
            return &inner[pos..];
        }
    }
    key
}

/// Element/isotope count map (`"C" -> 13`, `"H" -> 18`, `"[13C]" -> 1`, etc.).
pub type ElementCounts = BTreeMap<String, usize>;

/// Extract complete elemental/isotopic counts and net charge from `mol`.
pub fn element_counts_of_mol(mol: &AnalytMol) -> (ElementCounts, i32) {
    let mut counts: ElementCounts = BTreeMap::new();
    let mut net_charge: i32 = 0;

    for (i, a) in mol.atoms.iter().enumerate() {
        if a.element == "*" {
            continue;
        }
        net_charge += a.formal_charge as i32;
        let key = if a.element == "H" && a.isotope == Some(2) {
            "D".to_string()
        } else if a.element == "H" && a.isotope == Some(3) {
            "T".to_string()
        } else if let Some(iso) = a.isotope {
            if a.element == "H" && iso == 1 {
                "H".to_string()
            } else {
                format!("[{iso}{}]", a.element)
            }
        } else {
            a.element.clone()
        };
        *counts.entry(key).or_insert(0) += 1;

        let imp_h = mol.implicit_h_count(i) as usize;
        if imp_h > 0 {
            *counts.entry("H".to_string()).or_insert(0) += imp_h;
        }
    }
    (counts, net_charge)
}

/// Parse a plain molecular formula string (`"C8H9NO2"`, `"C2H5ClBr"`, `"[13C]CH5O"`) into `ElementCounts`.
pub fn parse_formula_counts(formula: &str) -> Result<ElementCounts, String> {
    let s: String = formula.chars().filter(|c| !c.is_whitespace()).collect();
    if s.is_empty() {
        return Err("Empty molecular formula".to_string());
    }
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut out = ElementCounts::new();

    while i < bytes.len() {
        let key = if bytes[i] == b'[' {
            let start = i;
            while i < bytes.len() && bytes[i] != b']' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err("Unclosed '[' in molecular formula".to_string());
            }
            i += 1;
            s[start..i].to_string()
        } else if bytes[i].is_ascii_uppercase() {
            let start = i;
            i += 1;
            if i < bytes.len() && bytes[i].is_ascii_lowercase() {
                i += 1;
            }
            s[start..i].to_string()
        } else {
            return Err(format!(
                "Unexpected character '{}' in formula '{formula}'",
                bytes[i] as char
            ));
        };

        if isotopes_of_key(&key).is_none() {
            return Err(format!("Unknown element or isotope '{key}' in formula"));
        }

        let num_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let n = if i > num_start {
            s[num_start..i]
                .parse::<usize>()
                .map_err(|_| "Invalid count in formula".to_string())?
        } else {
            1
        };
        if n > 0 {
            *out.entry(key).or_insert(0) += n;
        }
    }
    Ok(out)
}

/// Format `ElementCounts` in IUPAC Hill order (`C` first, then `H`, then remaining alphabetically;
/// purely alphabetical when no carbon is present), matching `hill` in `msanalysis.ts`.
pub fn hill_formula(counts: &ElementCounts) -> String {
    let keys: Vec<&String> = counts
        .iter()
        .filter(|&(_, &n)| n > 0)
        .map(|(k, _)| k)
        .collect();
    let has_c = keys.iter().any(|k| base_element_of_key(k) == "C");

    let rank = |k: &str| -> (u8, String, u8) {
        let el = base_element_of_key(k);
        let group = if has_c {
            if el == "C" {
                0
            } else if el == "H" && k == "H" {
                1
            } else {
                2
            }
        } else {
            2
        };
        let sort_sym = if k == "D" || k == "T" {
            k.to_string()
        } else {
            el.to_string()
        };
        let is_bracket = u8::from(k != el && k != "D" && k != "T");
        (group, sort_sym, is_bracket)
    };

    let mut sorted = keys;
    sorted.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.cmp(b)));

    let mut out = String::new();
    for k in sorted {
        let n = counts[k];
        out.push_str(k);
        if n > 1 {
            out.push_str(&n.to_string());
        }
    }
    out
}

/// Exact monoisotopic neutral mass of `counts`.
pub fn monoisotopic_mass(counts: &ElementCounts) -> Option<f64> {
    let mut sum = 0.0;
    for (k, &n) in counts {
        sum += (n as f64) * monoisotopic_atomic_mass(k)?;
    }
    Some(sum)
}

/// IUPAC natural-abundance average molecular weight of `counts`.
pub fn molecular_weight(counts: &ElementCounts) -> Option<f64> {
    let mut sum = 0.0;
    for (k, &n) in counts {
        sum += (n as f64) * average_atomic_mass(k)?;
    }
    Some(sum)
}

/// Elemental mass composition (`%C`, `%H`, `%N`, `%O`, ...) grouped by base element.
pub fn elemental_composition(counts: &ElementCounts) -> Option<BTreeMap<String, f64>> {
    let mw = molecular_weight(counts)?;
    if mw <= 0.0 {
        return None;
    }
    let mut by_el: BTreeMap<String, f64> = BTreeMap::new();
    for (k, &n) in counts {
        let m = (n as f64) * average_atomic_mass(k)?;
        let el = base_element_of_key(k).to_string();
        *by_el.entry(el).or_insert(0.0) += (m / mw) * 100.0;
    }
    Some(by_el)
}

/// Format elemental analysis string in standard ACS format (`"C, 63.56; H, 6.00; N, 9.27; O, 21.17"`).
pub fn format_elemental_analysis(comp: &BTreeMap<String, f64>) -> String {
    let has_c = comp.contains_key("C");
    let mut keys: Vec<&String> = comp.keys().collect();
    keys.sort_by(|a, b| {
        let ra = if has_c {
            if a.as_str() == "C" {
                0
            } else if a.as_str() == "H" {
                1
            } else {
                2
            }
        } else {
            2
        };
        let rb = if has_c {
            if b.as_str() == "C" {
                0
            } else if b.as_str() == "H" {
                1
            } else {
                2
            }
        } else {
            2
        };
        ra.cmp(&rb).then_with(|| a.cmp(b))
    });
    keys.into_iter()
        .map(|k| format!("{k}, {:.2}", comp[k]))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Degree of unsaturation (Rings + Double Bonds Equivalent, RDBE / IHD) from elemental counts & net charge:
/// `RDBE = 1 + 0.5 * (2*C + 2*Si + N + P - H - F - Cl - Br - I + charge)`.
pub fn rdbe_from_counts(counts: &ElementCounts, net_charge: i32) -> f64 {
    let mut sum2: isize = 2 + (net_charge as isize);
    for (k, &n) in counts {
        let n_i = n as isize;
        match base_element_of_key(k) {
            "C" | "Si" | "Ge" | "Sn" | "Pb" => sum2 += 2 * n_i,
            "N" | "P" | "As" | "Sb" | "B" => sum2 += n_i,
            "H" | "F" | "Cl" | "Br" | "I" | "Li" | "Na" | "K" | "Ag" => sum2 -= n_i,
            _ => {}
        }
    }
    0.5 * (sum2 as f64)
}

// ============================================================================
// Polynomial Convolution Isotopic Fine Distribution Envelope (`mass.ts`)
// ============================================================================

/// Single peak in an isotopic fine distribution envelope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsotopePeak {
    /// Abundance-weighted `m/z` of this nominal isotopologue cluster.
    pub mz: f64,
    /// Relative abundance scaled to `100.0` for the base peak (`[M]` or `[M+2]` etc.).
    pub rel_abundance: f64,
    /// Absolute natural probability fraction (`0.0..=1.0`, summing to `~1.0`).
    pub probability: f64,
}

/// Compute the polynomial-convolved isotopic envelope for `counts`, pruning clusters below `min_rel_percent`
/// (`0.1` or `0.5` % of the base peak), matching `isotopePattern` in `mass.ts`.
pub fn isotope_envelope(counts: &ElementCounts, min_rel_percent: f64) -> Option<Vec<IsotopePeak>> {
    let total_atoms: usize = counts.values().sum();
    if total_atoms == 0 || total_atoms > 2000 {
        return None;
    }

    // Convolve atom-by-atom at 1 mDa resolution (`round(mass * 1000)`).
    let mut peaks: Vec<(f64, f64)> = vec![(0.0, 1.0)];
    for (el_key, &cnt) in counts {
        let raw_iso = isotopes_of_key(el_key)?;
        let iso: Vec<(f64, f64)> = raw_iso.into_iter().map(|(m, a)| (m, a / 100.0)).collect();
        for _ in 0..cnt {
            let mut next_map: HashMap<i64, (f64, f64)> =
                HashMap::with_capacity(peaks.len() * iso.len());
            for &(m, p) in &peaks {
                for &(im, ip) in &iso {
                    let mass = m + im;
                    let prob = p * ip;
                    let key = (mass * 1000.0).round() as i64;
                    next_map
                        .entry(key)
                        .and_modify(|was| {
                            let tot = was.1 + prob;
                            was.0 = (was.0 * was.1 + mass * prob) / tot;
                            was.1 = tot;
                        })
                        .or_insert((mass, prob));
                }
            }
            let top = next_map
                .values()
                .map(|x| x.1)
                .fold(0.0f64, |a, b| if b > a { b } else { a });
            let cutoff = top * 1e-6;
            peaks = next_map
                .into_values()
                .filter(|&(_, prob)| prob > cutoff)
                .collect();
        }
    }

    if peaks.is_empty() {
        return None;
    }

    // Merge fine structure in each nominal 1 u step relative to the lightest isotopologue
    let lightest = peaks
        .iter()
        .map(|x| x.0)
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    let mut merged: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for (m, p) in peaks {
        let step = (m - lightest).round() as i64;
        merged
            .entry(step)
            .and_modify(|was| {
                let tot = was.1 + p;
                was.0 = (was.0 * was.1 + m * p) / tot;
                was.1 = tot;
            })
            .or_insert((m, p));
    }

    let top_prob = merged
        .values()
        .map(|x| x.1)
        .fold(0.0f64, |a, b| if b > a { b } else { a });
    if top_prob <= 0.0 {
        return None;
    }

    let mut out: Vec<IsotopePeak> = merged
        .into_values()
        .map(|(mz, prob)| IsotopePeak {
            mz,
            rel_abundance: (prob / top_prob) * 100.0,
            probability: prob,
        })
        .filter(|pk| pk.rel_abundance >= min_rel_percent)
        .collect();
    out.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(std::cmp::Ordering::Equal));
    Some(out)
}

/// Compute the neutral `[M]` isotopic envelope directly for `mol`.
pub fn isotope_envelope_for_mol(mol: &AnalytMol, min_rel_percent: f64) -> Option<Vec<IsotopePeak>> {
    let (counts, _) = element_counts_of_mol(mol);
    isotope_envelope(&counts, min_rel_percent)
}

// ============================================================================
// ESI / MALDI / EI Adducts, Charge States & HRMS Formatting (`msanalysis.ts`)
// ============================================================================

/// Standard MS adduct / ionization species (`msanalysis.ts` + `[2M+Na]+` / `[2M+H]+` / `[2M-H]-` dimers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MsIonKind {
    /// `[M+H]+` (or `[M+zH]z+` for `z > 1`)
    MPlusH,
    /// `[M+Na]+`
    MPlusNa,
    /// `[M+K]+`
    MPlusK,
    /// `[M+NH4]+`
    MPlusNH4,
    /// `[M+Li]+`
    MPlusLi,
    /// `[M+Ag]+`
    MPlusAg,
    /// `[2M+H]+`
    TwoMPlusH,
    /// `[2M+Na]+`
    TwoMPlusNa,
    /// `[2M+K]+`
    TwoMPlusK,
    /// `[M-Cl]+`
    MMinusClPos,
    /// `[M]+•` (radical cation, EI)
    MRadicalPos,
    /// `[M]` with molecule's own drawn formal charge
    MIntrinsic,
    /// `[M-H]−` (or `[M-zH]z−` for `z > 1`)
    MMinusH,
    /// `[M+Cl]−`
    MPlusClNeg,
    /// `[M+HCOO]−` (formate adduct)
    MPlusFormate,
    /// `[M+CH3COO]−` (acetate adduct)
    MPlusAcetate,
    /// `[2M-H]−`
    TwoMMinusH,
    /// `[M]−•` (radical anion)
    MRadicalNeg,
}

/// Calculated MS ion adduct with exact `m/z`, Hill formula, radical status, and isotopic envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MsIonAdduct {
    pub kind: MsIonKind,
    /// Formatted ion label (`"[M+H]+"`, `"[2M+Na]+"`, `"[M−H]−"`, `"[M]+•"`).
    pub label: String,
    /// Hill formula of the ion (`"C13H19O2"`).
    pub formula: String,
    pub counts: ElementCounts,
    pub charge: i32,
    pub radical: bool,
    /// Monoisotopic `m/z` corrected for electron mass (`(mono - q * m_e) / |q|`).
    pub monoisotopic_mz: f64,
    /// `m/z` of the base (100% relative abundance) peak in the isotopic envelope.
    pub base_peak_mz: f64,
    /// Full `m/z` isotopic envelope of this ion.
    pub pattern: Vec<IsotopePeak>,
}

fn add_counts(a: &ElementCounts, b: &ElementCounts) -> ElementCounts {
    let mut out = a.clone();
    for (k, &n) in b {
        if n > 0 {
            *out.entry(k.clone()).or_insert(0) += n;
        }
    }
    out
}

fn scale_counts(a: &ElementCounts, mult: usize) -> ElementCounts {
    let mut out = ElementCounts::new();
    if mult == 0 {
        return out;
    }
    for (k, &n) in a {
        if n > 0 {
            out.insert(k.clone(), n * mult);
        }
    }
    out
}

fn sub_counts(a: &ElementCounts, b: &ElementCounts) -> Option<ElementCounts> {
    let mut out = a.clone();
    for (k, &n) in b {
        let cur = out.get(k).copied().unwrap_or(0);
        if cur < n {
            return None;
        }
        if cur == n {
            out.remove(k);
        } else {
            out.insert(k.clone(), cur - n);
        }
    }
    Some(out)
}

fn counts_from_slice(items: &[(&str, usize)]) -> ElementCounts {
    let mut m = ElementCounts::new();
    for &(k, v) in items {
        m.insert(k.to_string(), v);
    }
    m
}

fn is_odd_electron_ion(counts: &ElementCounts, charge: i32) -> bool {
    let mut electrons: i64 = -(charge as i64);
    for (k, &n) in counts {
        let z = atomic_number(base_element_of_key(k)) as i64;
        electrons += (n as i64) * z;
    }
    electrons.unsigned_abs() % 2 == 1
}

fn format_charge_suffix(charge: i32, radical: bool) -> String {
    let mag = charge.unsigned_abs();
    let mag_str = if mag > 1 {
        mag.to_string()
    } else {
        String::new()
    };
    let sign = if charge > 0 { "+" } else { "−" };
    let dot = if radical { "•" } else { "" };
    format!("{mag_str}{sign}{dot}")
}

/// Calculate a specific MS adduct / ion species at charge state `z` (`1..=4`) for `(base_counts, base_charge)`.
pub fn calculate_ms_ion_from_counts(
    base_counts: &ElementCounts,
    base_charge: i32,
    kind: MsIonKind,
    z: usize,
) -> Option<MsIonAdduct> {
    let mult_z = if kind == MsIonKind::MIntrinsic {
        0
    } else {
        z.clamp(1, 4)
    };

    let (m_mult, ion_sign, add_unit, lose_unit, adduct_text): (
        usize,
        i32,
        ElementCounts,
        ElementCounts,
        &str,
    ) = match kind {
        MsIonKind::MPlusH => (
            1,
            1,
            counts_from_slice(&[("H", 1)]),
            ElementCounts::new(),
            "H",
        ),
        MsIonKind::MPlusNa => (
            1,
            1,
            counts_from_slice(&[("Na", 1)]),
            ElementCounts::new(),
            "Na",
        ),
        MsIonKind::MPlusK => (
            1,
            1,
            counts_from_slice(&[("K", 1)]),
            ElementCounts::new(),
            "K",
        ),
        MsIonKind::MPlusNH4 => (
            1,
            1,
            counts_from_slice(&[("N", 1), ("H", 4)]),
            ElementCounts::new(),
            "NH4",
        ),
        MsIonKind::MPlusLi => (
            1,
            1,
            counts_from_slice(&[("Li", 1)]),
            ElementCounts::new(),
            "Li",
        ),
        MsIonKind::MPlusAg => (
            1,
            1,
            counts_from_slice(&[("Ag", 1)]),
            ElementCounts::new(),
            "Ag",
        ),
        MsIonKind::TwoMPlusH => (
            2,
            1,
            counts_from_slice(&[("H", 1)]),
            ElementCounts::new(),
            "H",
        ),
        MsIonKind::TwoMPlusNa => (
            2,
            1,
            counts_from_slice(&[("Na", 1)]),
            ElementCounts::new(),
            "Na",
        ),
        MsIonKind::TwoMPlusK => (
            2,
            1,
            counts_from_slice(&[("K", 1)]),
            ElementCounts::new(),
            "K",
        ),
        MsIonKind::MMinusClPos => (
            1,
            1,
            ElementCounts::new(),
            counts_from_slice(&[("Cl", 1)]),
            "Cl",
        ),
        MsIonKind::MRadicalPos => (1, 1, ElementCounts::new(), ElementCounts::new(), ""),
        MsIonKind::MIntrinsic => (1, 1, ElementCounts::new(), ElementCounts::new(), ""),
        MsIonKind::MMinusH => (
            1,
            -1,
            ElementCounts::new(),
            counts_from_slice(&[("H", 1)]),
            "H",
        ),
        MsIonKind::MPlusClNeg => (
            1,
            -1,
            counts_from_slice(&[("Cl", 1)]),
            ElementCounts::new(),
            "Cl",
        ),
        MsIonKind::MPlusFormate => (
            1,
            -1,
            counts_from_slice(&[("C", 1), ("H", 1), ("O", 2)]),
            ElementCounts::new(),
            "HCOO",
        ),
        MsIonKind::MPlusAcetate => (
            1,
            -1,
            counts_from_slice(&[("C", 2), ("H", 3), ("O", 2)]),
            ElementCounts::new(),
            "CH3COO",
        ),
        MsIonKind::TwoMMinusH => (
            2,
            -1,
            ElementCounts::new(),
            counts_from_slice(&[("H", 1)]),
            "H",
        ),
        MsIonKind::MRadicalNeg => (1, -1, ElementCounts::new(), ElementCounts::new(), ""),
    };

    let m_scaled = scale_counts(base_counts, m_mult);
    let with_add = add_counts(&m_scaled, &scale_counts(&add_unit, mult_z));
    let ion_counts = sub_counts(&with_add, &scale_counts(&lose_unit, mult_z))?;

    let charge = (base_charge * (m_mult as i32))
        + if kind == MsIonKind::MIntrinsic {
            0
        } else {
            ion_sign * (mult_z as i32)
        };
    if charge == 0 || (kind != MsIonKind::MIntrinsic && charge * ion_sign < 0) {
        return None;
    }

    let radical = is_odd_electron_ion(&ion_counts, charge);
    let m_prefix = if m_mult == 2 { "2M" } else { "M" };
    let adduct_part = if matches!(
        kind,
        MsIonKind::MIntrinsic | MsIonKind::MRadicalPos | MsIonKind::MRadicalNeg
    ) {
        String::new()
    } else {
        let op = if !add_unit.is_empty() { "+" } else { "−" };
        let z_str = if mult_z > 1 {
            mult_z.to_string()
        } else {
            String::new()
        };
        format!("{op}{z_str}{adduct_text}")
    };
    let label = format!(
        "[{m_prefix}{adduct_part}]{}",
        format_charge_suffix(charge, radical)
    );

    let q_abs = (charge.unsigned_abs()) as f64;
    let mono_neutral = monoisotopic_mass(&ion_counts)?;
    let monoisotopic_mz = (mono_neutral - (charge as f64) * ELECTRON_MASS_U) / q_abs;

    let raw_env = isotope_envelope(&ion_counts, 0.1)?;
    let pattern: Vec<IsotopePeak> = raw_env
        .into_iter()
        .map(|p| IsotopePeak {
            mz: (p.mz - (charge as f64) * ELECTRON_MASS_U) / q_abs,
            rel_abundance: p.rel_abundance,
            probability: p.probability,
        })
        .collect();
    let base_peak_mz = pattern
        .iter()
        .max_by(|a, b| {
            a.rel_abundance
                .partial_cmp(&b.rel_abundance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.mz)
        .unwrap_or(monoisotopic_mz);

    Some(MsIonAdduct {
        kind,
        label,
        formula: hill_formula(&ion_counts),
        counts: ion_counts,
        charge,
        radical,
        monoisotopic_mz,
        base_peak_mz,
        pattern,
    })
}

/// Calculate standard ESI/MALDI positive & negative adducts (`[M+H]+`, `[M+Na]+`, `[M+K]+`,
/// `[M+NH4]+`, `[2M+Na]+`, `[M-H]−`, `[M+Cl]−`, `[M+HCOO]−`, `[M]+•`) for `mol`.
pub fn calculate_standard_ms_adducts(mol: &AnalytMol) -> Vec<MsIonAdduct> {
    let (counts, q) = element_counts_of_mol(mol);
    let kinds = [
        MsIonKind::MPlusH,
        MsIonKind::MPlusNa,
        MsIonKind::MPlusK,
        MsIonKind::MPlusNH4,
        MsIonKind::TwoMPlusH,
        MsIonKind::TwoMPlusNa,
        MsIonKind::MRadicalPos,
        MsIonKind::MMinusH,
        MsIonKind::MPlusClNeg,
        MsIonKind::MPlusFormate,
        MsIonKind::TwoMMinusH,
    ];
    kinds
        .into_iter()
        .filter_map(|k| calculate_ms_ion_from_counts(&counts, q, k, 1))
        .collect()
}

/// Format an ACS HRMS report line (`"HRMS (ESI) m/z: [M+H]+ calcd for C13H19O2 207.1380; found 207.1382."`).
pub fn hrms_report_line(
    source: &str,
    adduct: &MsIonAdduct,
    found_mz: Option<f64>,
    decimals: usize,
) -> String {
    let calcd_s = format!("{:.prec$}", adduct.monoisotopic_mz, prec = decimals);
    match found_mz {
        Some(f) if f.is_finite() => {
            let found_s = format!("{:.prec$}", f, prec = decimals);
            format!(
                "HRMS ({source}) m/z: {} calcd for {} {calcd_s}; found {found_s}.",
                adduct.label, adduct.formula
            )
        }
        _ => format!(
            "HRMS ({source}) m/z: {} calcd for {} {calcd_s}.",
            adduct.label, adduct.formula
        ),
    }
}

/// Mass accuracy error in parts per million (`ppm`).
pub fn mass_error_ppm(found_mz: f64, calcd_mz: f64) -> f64 {
    ((found_mz - calcd_mz) / calcd_mz) * 1e6
}

// ============================================================================
// Single-Bond & Cut-Path MS Cleavage Fragmentation (`masscut.ts` & `msanalysis.ts`)
// ============================================================================

/// Multi-line MS fragmentation mode (`combinatorial`, `sequential`, `simultaneous`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentationMode {
    Combinatorial,
    Sequential,
    Simultaneous,
}

/// Predicted MS cleavage fragment piece.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MsFragment {
    /// Atom indices in `AnalytMol` belonging to this fragment.
    pub atoms: Vec<usize>,
    /// Hill formula of the homolytic radical/neutral fragment.
    pub formula: String,
    pub counts: ElementCounts,
    /// Exact monoisotopic mass of the fragment (`u`) and `[Frag]+` ion `m/z` (`exact - m_e`).
    pub exact_mass: f64,
    pub cation_mz: f64,
    /// Indices of cleaved bonds or cut lines that produced this fragment.
    pub cleaved_indices: Vec<usize>,
    /// Mechanistic cleavage description (`"alpha-carbonyl"`, `"benzylic"`, `"allylic"`, `"heteroatom-C"`, `"single-bond"`).
    pub cleavage_kind: String,
}

fn per_atom_counts(mol: &AnalytMol) -> Vec<ElementCounts> {
    mol.atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut c = ElementCounts::new();
            if a.element == "*" {
                return c;
            }
            let key = if a.element == "H" && a.isotope == Some(2) {
                "D".to_string()
            } else if a.element == "H" && a.isotope == Some(3) {
                "T".to_string()
            } else if let Some(iso) = a.isotope {
                if a.element == "H" && iso == 1 {
                    "H".to_string()
                } else {
                    format!("[{iso}{}]", a.element)
                }
            } else {
                a.element.clone()
            };
            *c.entry(key).or_insert(0) += 1;
            let h = mol.implicit_h_count(i) as usize;
            if h > 0 {
                *c.entry("H".to_string()).or_insert(0) += h;
            }
            c
        })
        .collect()
}

/// Predict all unique single-bond MS cleavage fragments (acylium α-cleavage, benzylic, allylic,
/// C–heteroatom, and general acyclic single bonds), sorted from heaviest to lightest.
pub fn predict_ms_fragments(mol: &AnalytMol) -> Vec<MsFragment> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let atom_cts = per_atom_counts(mol);
    let arom = mol.aromatic_atoms();
    let rings = mol.sssr_rings();
    let adj = mol.adjacency();

    let is_carbonyl = |idx: usize| -> bool {
        mol.atoms[idx].element == "C"
            && !arom[idx]
            && adj[idx]
                .iter()
                .any(|&(nb, bi)| mol.bonds[bi].order == 2 && mol.atoms[nb].element == "O")
    };
    let is_alkenyl = |idx: usize| -> bool {
        mol.atoms[idx].element == "C"
            && !arom[idx]
            && adj[idx]
                .iter()
                .any(|&(nb, bi)| mol.bonds[bi].order == 2 && mol.atoms[nb].element == "C")
    };

    let mut seen_atom_sets: HashSet<Vec<usize>> = HashSet::new();
    let mut out: Vec<MsFragment> = Vec::new();

    for (bi, b) in mol.bonds.iter().enumerate() {
        if b.order != 1 || b.aromatic {
            continue;
        }
        // Skip explicit 1H leaf bonds unless desired; heavy-heavy single bonds give diagnostic MS fragments
        if mol.atoms[b.a].element == "H" || mol.atoms[b.b].element == "H" {
            continue;
        }
        // Skip ring bonds (cutting a single ring bond leaves 1 connected component)
        if rings.iter().any(|r| r.contains(&b.a) && r.contains(&b.b)) {
            continue;
        }

        let cleavage_kind = if is_carbonyl(b.a) || is_carbonyl(b.b) {
            "alpha-carbonyl"
        } else if (arom[b.a] && !arom[b.b])
            || (arom[b.b] && !arom[b.a])
            || adj[b.a].iter().any(|&(x, _)| x != b.b && arom[x])
            || adj[b.b].iter().any(|&(x, _)| x != b.a && arom[x])
        {
            "benzylic"
        } else if is_alkenyl(b.a)
            || is_alkenyl(b.b)
            || adj[b.a].iter().any(|&(x, _)| x != b.b && is_alkenyl(x))
            || adj[b.b].iter().any(|&(x, _)| x != b.a && is_alkenyl(x))
        {
            "allylic"
        } else if mol.atoms[b.a].element != "C" || mol.atoms[b.b].element != "C" {
            "heteroatom-C"
        } else {
            "single-bond"
        };

        // Find connected components after removing bond `bi`
        let mut visited = vec![false; n];
        for &seed in &[b.a, b.b] {
            if visited[seed] {
                continue;
            }
            let mut comp = Vec::new();
            let mut st = vec![seed];
            visited[seed] = true;
            while let Some(u) = st.pop() {
                comp.push(u);
                for &(v, e_idx) in &adj[u] {
                    if e_idx == bi || visited[v] {
                        continue;
                    }
                    visited[v] = true;
                    st.push(v);
                }
            }
            comp.sort_unstable();
            if comp.len() == n || !seen_atom_sets.insert(comp.clone()) {
                continue;
            }
            let mut cts = ElementCounts::new();
            for &idx in &comp {
                cts = add_counts(&cts, &atom_cts[idx]);
            }
            if let Some(exact) = monoisotopic_mass(&cts) {
                out.push(MsFragment {
                    atoms: comp,
                    formula: hill_formula(&cts),
                    counts: cts,
                    exact_mass: exact,
                    cation_mz: exact - ELECTRON_MASS_U,
                    cleaved_indices: vec![bi],
                    cleavage_kind: cleavage_kind.to_string(),
                });
            }
        }
    }

    out.sort_by(|a, b| {
        b.exact_mass
            .partial_cmp(&a.exact_mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn cross_2d(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn segment_crossing(a: (f64, f64), b: (f64, f64), p: (f64, f64), q: (f64, f64)) -> i8 {
    let sp = cross_2d(a, b, p);
    let sq = cross_2d(a, b, q);
    if sp * sq < 0.0 && cross_2d(p, q, a) * cross_2d(p, q, b) < 0.0 {
        if sp > 0.0 {
            1
        } else {
            -1
        }
    } else {
        0
    }
}

/// Find bonds crossed by a 2D fragmentation polyline `path` (`crossedBonds` in `masscut.ts`).
pub fn crossed_bonds_along_path(mol: &AnalytMol, path: &[(f64, f64)]) -> Vec<(usize, i8)> {
    let mut out = Vec::new();
    for (bi, b) in mol.bonds.iter().enumerate() {
        let pa = (mol.atoms[b.a].x, mol.atoms[b.a].y);
        let pb = (mol.atoms[b.b].x, mol.atoms[b.b].y);
        for s in 0..path.len().saturating_sub(1) {
            let side = segment_crossing(path[s], path[s + 1], pa, pb);
            if side != 0 {
                out.push((bi, side));
                break;
            }
        }
    }
    out
}

/// Evaluate multiple MS cut lines (each represented by the bond indices it crosses) in
/// `Combinatorial`, `Sequential`, or `Simultaneous` mode (`fragmentsOf` in `msanalysis.ts`).
pub fn fragments_from_bond_cuts(
    mol: &AnalytMol,
    cut_lines_bonds: &[Vec<usize>],
    mode: FragmentationMode,
) -> Vec<MsFragment> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let atom_cts = per_atom_counts(mol);
    let active_lines: Vec<(usize, &Vec<usize>)> = cut_lines_bonds
        .iter()
        .take(12)
        .enumerate()
        .filter(|(_, l)| !l.is_empty())
        .collect();

    let mut sets: Vec<Vec<usize>> = match mode {
        FragmentationMode::Simultaneous => {
            if active_lines.is_empty() {
                Vec::new()
            } else {
                vec![active_lines.iter().map(|&(k, _)| k).collect()]
            }
        }
        FragmentationMode::Sequential => active_lines.iter().map(|&(k, _)| vec![k]).collect(),
        FragmentationMode::Combinatorial => {
            let m = active_lines.len();
            let mut v: Vec<Vec<usize>> = (1..(1usize << m))
                .map(|mask| {
                    active_lines
                        .iter()
                        .enumerate()
                        .filter(|&(bit, _)| ((mask >> bit) & 1) == 1)
                        .map(|(_, &(k, _))| k)
                        .collect()
                })
                .collect();
            v.sort_by_key(|s| s.len());
            v
        }
    };

    let all_atoms: Vec<usize> = (0..n).collect();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    seen.insert(all_atoms);

    let adj = mol.adjacency();
    let mut out = Vec::new();

    for set in sets.drain(..) {
        let cut_bonds: HashSet<usize> = set
            .iter()
            .flat_map(|&line_idx| cut_lines_bonds[line_idx].iter().copied())
            .collect();
        let mut visited = vec![false; n];
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(u) = stack.pop() {
                comp.push(u);
                for &(v, bi) in &adj[u] {
                    if cut_bonds.contains(&bi) || visited[v] {
                        continue;
                    }
                    visited[v] = true;
                    stack.push(v);
                }
            }
            comp.sort_unstable();
            if seen.insert(comp.clone()) {
                let mut cts = ElementCounts::new();
                for &idx in &comp {
                    cts = add_counts(&cts, &atom_cts[idx]);
                }
                if let Some(exact) = monoisotopic_mass(&cts) {
                    out.push(MsFragment {
                        atoms: comp,
                        formula: hill_formula(&cts),
                        counts: cts,
                        exact_mass: exact,
                        cation_mz: exact - ELECTRON_MASS_U,
                        cleaved_indices: set.clone(),
                        cleavage_kind: "cut-line".to_string(),
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| {
        b.exact_mass
            .partial_cmp(&a.exact_mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

// ============================================================================
// Physicochemical Properties: Crippen cLogP & MR, Ertl TPSA, Rotatable Bonds,
// Lipinski HBD/HBA, Delaney ESOL logS, Joback Thermo, and Typical pKa Sites
// ============================================================================

/// Joback & Reid (1987) ideal-gas / critical thermodynamic group contribution estimates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JobackThermo {
    pub boiling_point_k: f64,
    pub melting_point_k: f64,
    pub critical_temp_k: f64,
    pub critical_pressure_bar: f64,
    pub critical_volume_cm3_mol: f64,
    pub enthalpy_formation_kj_mol: f64,
    pub gibbs_energy_kj_mol: f64,
}

/// Ionizable site with typical pKa mean and standard deviation (Dimorphite-DL table).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkaSite {
    pub atom_index: usize,
    pub kind: String,
    pub pka: f64,
    pub sd: f64,
}

/// Comprehensive small-molecule physicochemical property report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysChemProps {
    pub formula: String,
    pub molecular_weight: f64,
    pub monoisotopic_mass: f64,
    pub elemental_percent: BTreeMap<String, f64>,
    pub elemental_analysis_text: String,
    /// Rings + Double Bonds Equivalent (IHD / RDBE).
    pub rdbe: f64,
    /// Wildman & Crippen (1999) octanol-water partition coefficient (`cLogP`).
    pub clogp: f64,
    /// Wildman & Crippen (1999) molar refractivity (`cm³/mol`).
    pub molar_refractivity: f64,
    /// Ertl, Rohde & Selzer (2000) topological polar surface area (`Å²`).
    pub tpsa: f64,
    /// Lipinski hydrogen-bond donors (`N` and `O` atoms bearing ≥ 1 hydrogen).
    pub h_bond_donors: usize,
    /// Lipinski hydrogen-bond acceptors (`N` and `O` atoms).
    pub h_bond_acceptors: usize,
    /// Strict rotatable single bonds (`props.ts` / RDKit definition).
    pub rotatable_bonds: usize,
    /// Fraction of heavy atoms in aromatic rings.
    pub aromatic_proportion: f64,
    /// Delaney (2004) ESOL aqueous solubility (`log10(mol/L)`).
    pub esol_logs: f64,
    /// Passes Lipinski's Rule of Five (`MW <= 500`, `cLogP <= 5`, `HBD <= 5`, `HBA <= 10`).
    pub lipinski_ro5_pass: bool,
    /// Joback & Reid (1987) thermodynamic estimates when all groups are covered.
    pub joback: Option<JobackThermo>,
    /// Typical ionizable `pKa` sites.
    pub pka_sites: Vec<PkaSite>,
}

#[derive(Debug, Clone)]
struct HeavyEdge {
    j: usize,
    k: usize,
    order: u8,
}

#[derive(Debug, Clone)]
struct PropCtx {
    el: Vec<String>,
    q: Vec<i8>,
    h: Vec<u8>,
    of_mol: Vec<usize>,
    nb: Vec<Vec<HeavyEdge>>,
    rings: Vec<Vec<usize>>,
    ring_atom: Vec<bool>,
    ring_bond: Vec<bool>,
    arom: Vec<bool>,
    arom_bond: Vec<bool>,
}

fn build_prop_ctx(mol: &AnalytMol) -> PropCtx {
    let n = mol.atoms.len();
    let adj_full = mol.adjacency();
    let keep: Vec<bool> = (0..n)
        .map(|i| {
            let a = &mol.atoms[i];
            !(a.element == "H" && a.formal_charge == 0 && adj_full[i].len() == 1)
        })
        .collect();
    let mut to_ctx = vec![usize::MAX; n];
    let mut of_mol = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k {
            to_ctx[i] = of_mol.len();
            of_mol.push(i);
        }
    }

    let m = of_mol.len();
    let mut el: Vec<String> = of_mol
        .iter()
        .map(|&i| mol.atoms[i].element.clone())
        .collect();
    let q: Vec<i8> = of_mol.iter().map(|&i| mol.atoms[i].formal_charge).collect();
    let mut h: Vec<u8> = of_mol.iter().map(|&i| mol.implicit_h_count(i)).collect();
    for item in el.iter_mut().take(m) {
        if item == "D" || item == "T" {
            *item = "H".to_string();
        }
    }

    let mut bonds: Vec<(usize, usize, u8)> = Vec::new();
    for b in &mol.bonds {
        if b.order == 0 {
            continue;
        }
        if keep[b.a] && keep[b.b] {
            bonds.push((to_ctx[b.a], to_ctx[b.b], b.order));
        } else if keep[b.a] {
            h[to_ctx[b.a]] = h[to_ctx[b.a]].saturating_add(1);
        } else if keep[b.b] {
            h[to_ctx[b.b]] = h[to_ctx[b.b]].saturating_add(1);
        }
    }

    let mut nb = vec![Vec::new(); m];
    for (k, &(a, b, order)) in bonds.iter().enumerate() {
        nb[a].push(HeavyEdge { j: b, k, order });
        nb[b].push(HeavyEdge { j: a, k, order });
    }

    let mol_arom = mol.aromatic_atoms();
    let arom: Vec<bool> = of_mol.iter().map(|&i| mol_arom[i]).collect();

    // Project SSSR rings onto HeavyCtx
    let raw_rings = mol.sssr_rings();
    let mut rings: Vec<Vec<usize>> = Vec::new();
    for r in raw_rings {
        if r.iter().all(|&orig| keep[orig]) {
            rings.push(r.into_iter().map(|orig| to_ctx[orig]).collect());
        }
    }

    let mut ring_atom = vec![false; m];
    let mut ring_bond = vec![false; bonds.len()];
    let mut arom_bond = vec![false; bonds.len()];

    for r in &rings {
        let all_arom = r.iter().all(|&v| arom[v]);
        for idx in 0..r.len() {
            let a = r[idx];
            let b = r[(idx + 1) % r.len()];
            ring_atom[a] = true;
            if let Some(edge) = nb[a].iter().find(|x| x.j == b) {
                ring_bond[edge.k] = true;
                if all_arom {
                    arom_bond[edge.k] = true;
                }
            }
        }
    }

    PropCtx {
        el,
        q,
        h,
        of_mol,
        nb,
        rings,
        ring_atom,
        ring_bond,
        arom,
        arom_bond,
    }
}

/// Ertl, Rohde & Selzer (2000) TPSA in Å² (`tpsa` in `props.ts`).
fn compute_ertl_tpsa(c: &PropCtx) -> f64 {
    let mut sum = 0.0;
    for i in 0..c.el.len() {
        let el = c.el[i].as_str();
        if el != "N" && el != "O" {
            continue;
        }
        let n = c.nb[i].len();
        let h = c.h[i];
        let q = c.q[i];
        let mut t = [0usize; 4]; // single, double, triple, aromatic
        for x in &c.nb[i] {
            let slot = if c.arom_bond[x.k] {
                3
            } else {
                (x.order.saturating_sub(1) as usize).min(2)
            };
            t[slot] += 1;
        }
        let [s, d, tr, ar] = t;
        let r3 = c.rings.iter().any(|r| r.len() == 3 && r.contains(&i));
        let is = |nh: u8, ch: i8, ns: usize, nd: usize, nt: usize, na: usize| {
            h == nh && q == ch && s == ns && d == nd && tr == nt && ar == na
        };

        let val: Option<f64> = if el == "N" {
            match n {
                1 => {
                    if is(0, 0, 0, 0, 1, 0) {
                        Some(23.79)
                    } else if is(1, 0, 0, 1, 0, 0) {
                        Some(23.85)
                    } else if is(2, 0, 1, 0, 0, 0) {
                        Some(26.02)
                    } else if is(2, 1, 0, 1, 0, 0) {
                        Some(25.59)
                    } else if is(3, 1, 1, 0, 0, 0) {
                        Some(27.64)
                    } else {
                        None
                    }
                }
                2 => {
                    if is(0, 0, 1, 1, 0, 0) {
                        Some(12.36)
                    } else if is(0, 0, 0, 1, 1, 0) {
                        Some(13.60)
                    } else if is(1, 0, 2, 0, 0, 0) {
                        Some(if r3 { 21.94 } else { 12.03 })
                    } else if is(0, 1, 1, 0, 1, 0) {
                        Some(4.36)
                    } else if is(1, 1, 1, 1, 0, 0) {
                        Some(13.97)
                    } else if is(2, 1, 2, 0, 0, 0) {
                        Some(16.61)
                    } else if is(0, 0, 0, 0, 0, 2) {
                        Some(12.89)
                    } else if is(1, 0, 0, 0, 0, 2) {
                        Some(15.79)
                    } else if is(1, 1, 0, 0, 0, 2) {
                        Some(14.14)
                    } else {
                        None
                    }
                }
                3 => {
                    if is(0, 0, 3, 0, 0, 0) {
                        Some(if r3 { 3.01 } else { 3.24 })
                    } else if is(0, 0, 1, 2, 0, 0) {
                        Some(11.68)
                    } else if is(0, 1, 2, 1, 0, 0) {
                        Some(3.01)
                    } else if is(1, 1, 3, 0, 0, 0) {
                        Some(4.44)
                    } else if is(0, 0, 0, 0, 0, 3) {
                        Some(4.41)
                    } else if is(0, 0, 1, 0, 0, 2) {
                        Some(4.93)
                    } else if is(0, 0, 0, 1, 0, 2) {
                        Some(8.39)
                    } else if is(0, 1, 0, 0, 0, 3) {
                        Some(4.10)
                    } else if is(0, 1, 1, 0, 0, 2) {
                        Some(3.88)
                    } else {
                        None
                    }
                }
                4 => {
                    if is(0, 1, 4, 0, 0, 0) {
                        Some(0.0)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            // Oxygen
            match n {
                1 => {
                    if is(0, 0, 0, 1, 0, 0) {
                        Some(17.07)
                    } else if is(1, 0, 1, 0, 0, 0) {
                        Some(20.23)
                    } else if is(0, -1, 1, 0, 0, 0) {
                        Some(23.06)
                    } else {
                        None
                    }
                }
                2 => {
                    if is(0, 0, 2, 0, 0, 0) {
                        Some(if r3 { 12.53 } else { 9.23 })
                    } else if is(0, 0, 0, 0, 0, 2) {
                        Some(13.14)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        sum += val.unwrap_or_else(|| {
            if el == "N" {
                (30.5 - (n as f64) * 8.2 + (h as f64) * 1.5).max(0.0)
            } else {
                (28.5 - (n as f64) * 8.6 + (h as f64) * 1.5).max(0.0)
            }
        });
    }
    sum
}

/// Strict RDKit / Octet-Draw rotatable bond count (`rotatable` in `props.ts`).
fn compute_rotatable_bonds(c: &PropCtx) -> usize {
    let al = |i: usize, els: &[&str]| !c.arom[i] && els.contains(&c.el[i].as_str());
    let tail = |i: usize| -> bool {
        c.nb[i].iter().any(|x| x.order == 3)
            || c.nb[i].len() < 2
            || (al(i, &["C"])
                && (["F", "Cl", "Br"]
                    .iter()
                    .any(|&hal| c.nb[i].iter().filter(|x| c.el[x.j] == hal).count() >= 3)
                    || c.nb[i]
                        .iter()
                        .filter(|x| al(x.j, &["C"]) && c.h[x.j] == 3)
                        .count()
                        >= 3))
    };
    let acyl = |i: usize, plus: bool| -> bool {
        al(i, &["C"])
            && c.nb[i].len() == 3
            && c.nb[i].iter().any(|x| {
                x.order == 2
                    && !c.arom_bond[x.k]
                    && if plus {
                        al(x.j, &["N"]) && c.q[x.j] == 1
                    } else {
                        al(x.j, &["N", "O", "S"])
                    }
            })
    };
    let out_single = |i: usize| -> Vec<usize> {
        c.nb[i]
            .iter()
            .filter(|x| x.order == 1 && !c.ring_bond[x.k])
            .map(|x| x.j)
            .collect()
    };
    let hetero = |j: usize| -> bool {
        c.el[j] == "N" || al(j, &["O"]) || (al(j, &["S"]) && c.nb[j].len() != 1)
    };
    let amide_end = |i: usize| -> bool {
        (acyl(i, false) && out_single(i).into_iter().any(&hetero))
            || (hetero(i) && out_single(i).into_iter().any(|j| acyl(j, false)))
            || (acyl(i, true)
                && out_single(i)
                    .into_iter()
                    .any(|j| c.el[j] == "N" && c.nb[j].len() != 1))
            || (c.el[i] == "N"
                && c.nb[i].len() != 1
                && out_single(i).into_iter().any(|j| acyl(j, true)))
    };

    let mut count = 0usize;
    for a in 0..c.nb.len() {
        for e in &c.nb[a] {
            let b = e.j;
            if a < b
                && e.order == 1
                && !c.ring_bond[e.k]
                && !tail(a)
                && !tail(b)
                && (!amide_end(a) || !amide_end(b))
            {
                count += 1;
            }
        }
    }
    count
}

/// Wildman & Crippen (*J. Chem. Inf. Comput. Sci.* **1999**, 39, 868–873) atom-based `cLogP`
/// and molar refractivity (`MR`, cm³/mol).
fn compute_crippen_clogp_mr(c: &PropCtx) -> (f64, f64) {
    let mut logp = 0.0f64;
    let mut mr = 0.0f64;

    let is_hetero = |j: usize| !matches!(c.el[j].as_str(), "C" | "H");
    let has_pi = |j: usize| c.arom[j] || c.nb[j].iter().any(|e| e.order >= 2 && !c.arom_bond[e.k]);

    for i in 0..c.el.len() {
        let el = c.el[i].as_str();
        let h = c.h[i] as usize;
        let q = c.q[i];

        match el {
            "C" => {
                // Hydrogen contributions attached to C:
                // H1 (alkane/aryl/alkene C-H: +0.1230 logP, 1.057 MR; alpha to hetero/pi H4: +0.2980 / +0.1125)
                let het_nbs = c.nb[i].iter().filter(|e| is_hetero(e.j)).count();
                let pi_nbs = c.nb[i]
                    .iter()
                    .filter(|e| !is_hetero(e.j) && has_pi(e.j))
                    .count();

                let (c_lp, c_mr) = if c.arom[i] {
                    if h >= 1 {
                        // C18: aromatic CH
                        (0.1581, 3.350)
                    } else if het_nbs >= 1 {
                        // C20/C21: aromatic C bonded to heteroatom
                        (0.1260, 3.509)
                    } else {
                        // C21: alkyl/aryl-substituted aromatic C (`c(-C)(:c):c`)
                        (0.2966, 3.509)
                    }
                } else {
                    let max_bo = c.nb[i].iter().map(|e| e.order).max().unwrap_or(1);
                    if max_bo == 3 {
                        // C15/C16: sp carbon (alkyne / nitrile)
                        if het_nbs >= 1 {
                            (0.1388, 3.000) // nitrile C
                        } else {
                            (0.2000, 3.800)
                        }
                    } else if max_bo == 2 {
                        // Carbonyl vs alkene
                        let dbl_het = c.nb[i].iter().any(|e| e.order == 2 && is_hetero(e.j));
                        if dbl_het {
                            // C25 / C26: R-(C=X)-R or R-(C=X)-X (Wildman-Crippen Table 1)
                            if het_nbs >= 2 {
                                (0.1539, 2.753) // carboxylic acid / ester / amide C
                            } else {
                                (-0.0625, 2.753) // aldehyde / ketone C
                            }
                        } else {
                            // Alkene sp2 C (C13 / C14 / C17)
                            match h {
                                2 => (0.1581, 3.350),
                                1 => (0.1950, 3.450),
                                _ => (0.2650, 3.650),
                            }
                        }
                    } else {
                        // sp3 carbon: Wildman-Crippen C1..C12
                        if q != 0 {
                            (-0.80, 2.50)
                        } else if het_nbs == 0 {
                            if pi_nbs == 0 {
                                // Pure aliphatic C1 (CH4/CH3/CH2), C2 (CH/C)
                                match h {
                                    2..=4 => (0.1441, 2.503),
                                    _ => (0.0000, 2.433),
                                }
                            } else {
                                // C8/C9/C10/C11: benzylic / allylic / alpha-carbonyl sp3 C
                                match h {
                                    3 => (0.0845, 2.753),
                                    2 => (-0.0516, 2.550),
                                    _ => (0.1193, 2.480),
                                }
                            }
                        } else {
                            // C3/C4: sp3 C bonded to O, N, S, halogen
                            match h {
                                3 | 2 => (-0.2035, 2.753),
                                _ => (-0.2051, 2.731),
                            }
                        }
                    }
                };
                logp += c_lp + (h as f64) * 0.1230;
                mr += c_mr + (h as f64) * 1.057;
            }
            "O" => {
                // O1..O12 + H2 (alcohol/acid OH hydrogen = -0.2677 logP, 0.885 MR)
                let max_bo = c.nb[i].iter().map(|e| e.order).max().unwrap_or(1);
                let has_arom = c.nb[i].iter().any(|e| c.arom[e.j]);
                let has_carbonyl = c.nb[i].iter().any(|e| {
                    c.el[e.j] == "C"
                        && c.nb[e.j]
                            .iter()
                            .any(|x| x.j != i && x.order == 2 && c.el[x.j] == "O")
                });
                let is_acid_or_ester_oxo = max_bo == 2
                    && c.nb[i].iter().any(|e| {
                        c.el[e.j] == "C"
                            && (c.arom[e.j] || c.nb[e.j].iter().any(|x| x.j != i && is_hetero(x.j)))
                    });
                let (o_lp, o_mr) = if q < 0 {
                    (-1.189, 1.60)
                } else if c.arom[i] {
                    (0.5482, 1.65) // furan O
                } else if max_bo == 2 {
                    // O9/O10/O11: carbonyl =O / nitro =O / S=O (Wildman-Crippen O9 = +0.0352, O10 = +0.1795)
                    if c.nb[i].iter().any(|e| c.el[e.j] == "N") {
                        (-0.1500, 1.70)
                    } else if is_acid_or_ester_oxo {
                        (0.1795, 1.795) // O10: acid/ester/amide/aryl C=O oxygen
                    } else if has_carbonyl || c.nb[i].iter().any(|e| c.el[e.j] == "C") {
                        (0.0352, 1.795) // O9: aliphatic aldehyde/ketone C=O oxygen
                    } else {
                        (-0.4200, 1.70)
                    }
                } else if h >= 1 {
                    if has_carbonyl {
                        (-0.2893, 1.80) // carboxylic -OH
                    } else if has_arom {
                        (0.1129, 1.80) // phenolic -OH
                    } else {
                        (-0.2893, 1.562) // aliphatic -OH
                    }
                } else if has_carbonyl {
                    (0.0150, 1.65) // ester -O-
                } else if has_arom {
                    (0.4195, 1.75) // aryl ether
                } else {
                    (-0.0684, 1.60) // dialkyl ether
                };
                logp += o_lp + (h as f64) * (-0.2677);
                mr += o_mr + (h as f64) * 0.885;
            }
            "N" => {
                // N1..N14 + H3 (amine/amide NH hydrogen = +0.2142 logP, 0.962 MR)
                let is_nitro = q > 0
                    && c.nb[i]
                        .iter()
                        .filter(|e| c.el[e.j] == "O" && c.nb[e.j].len() == 1)
                        .count()
                        >= 2;
                let max_bo = c.nb[i].iter().map(|e| e.order).max().unwrap_or(1);
                let has_arom = c.nb[i].iter().any(|e| c.arom[e.j]);
                let has_carbonyl = c.nb[i].iter().any(|e| {
                    c.el[e.j] == "C"
                        && c.nb[e.j]
                            .iter()
                            .any(|x| x.order == 2 && matches!(c.el[x.j].as_str(), "O" | "S"))
                });
                let (n_lp, n_mr) = if is_nitro {
                    (0.1800, 2.60)
                } else if q > 0 {
                    (-1.6500, 2.20)
                } else if c.arom[i] {
                    if h >= 1 {
                        (-0.5400, 2.50) // pyrrole NH
                    } else {
                        (-0.3239, 2.50) // pyridine N
                    }
                } else if max_bo == 3 {
                    (0.0660, 2.45) // nitrile N
                } else if max_bo == 2 {
                    (-0.1000, 2.50) // imine N
                } else if has_carbonyl {
                    (-0.6460, 2.40) // amide N
                } else if has_arom {
                    (-0.4930, 2.60) // aniline N
                } else {
                    match h {
                        2 => (-1.0190, 2.26),
                        1 => (-0.7096, 2.30),
                        _ => (-0.3000, 2.45),
                    }
                };
                logp += n_lp + (h as f64) * 0.2142;
                mr += n_mr + (h as f64) * 0.962;
            }
            "F" => {
                logp += 0.4202;
                mr += 1.108;
            }
            "Cl" => {
                logp += 0.6895;
                mr += 5.853;
            }
            "Br" => {
                logp += 0.8456;
                mr += 8.927;
            }
            "I" => {
                logp += 0.8857;
                mr += 14.02;
            }
            "S" => {
                let ox = c.nb[i]
                    .iter()
                    .filter(|e| e.order == 2 && c.el[e.j] == "O")
                    .count();
                if ox >= 2 {
                    logp += -0.6000;
                    mr += 6.80;
                } else if ox == 1 {
                    logp += -0.5500;
                    mr += 7.20;
                } else if c.arom[i] {
                    logp += 0.6237;
                    mr += 7.365;
                } else {
                    logp += 0.6482 + (h as f64) * 0.10;
                    mr += 7.591 + (h as f64) * 1.00;
                }
            }
            "P" => {
                logp += -0.2500;
                mr += 6.90;
            }
            "Si" => {
                logp += 0.6500 + (h as f64) * 0.12;
                mr += 7.30 + (h as f64) * 1.05;
            }
            "B" => {
                logp += -0.1500;
                mr += 3.50;
            }
            _ => {
                // Metal / ion fallback
                logp += -0.50;
                mr += 3.00;
            }
        }
    }

    (logp, mr)
}

fn joback_contrib(group: &str) -> Option<[f64; 7]> {
    // [Tc, Pc, Vc, Tb, Tm, dHf, dGf] from Joback & Reid (1987) (`props.ts`)
    match group {
        "-CH3" => Some([0.0141, -0.0012, 65.0, 23.58, -5.1, -76.45, -43.96]),
        "-CH2-" => Some([0.0189, 0.0, 56.0, 22.88, 11.27, -20.64, 8.42]),
        ">CH-" => Some([0.0164, 0.002, 41.0, 21.74, 12.64, 29.89, 58.36]),
        ">C<" => Some([0.0067, 0.0043, 27.0, 18.25, 46.43, 82.23, 116.02]),
        "=CH2" => Some([0.0113, -0.0028, 56.0, 18.18, -4.32, -9.63, 3.77]),
        "=CH-" => Some([0.0129, -0.0006, 46.0, 24.96, 8.73, 37.97, 48.53]),
        "=C<" => Some([0.0117, 0.0011, 38.0, 24.14, 11.14, 83.99, 92.36]),
        "=C=" => Some([0.0026, 0.0028, 36.0, 26.15, 17.78, 142.14, 136.7]),
        "≡CH" => Some([0.0027, -0.0008, 46.0, 9.2, -11.18, 79.3, 77.71]),
        "≡C-" => Some([0.002, 0.0016, 37.0, 27.38, 64.32, 115.51, 109.82]),
        "ring -CH2-" => Some([0.01, 0.0025, 48.0, 27.15, 7.75, -26.8, -3.68]),
        "ring >CH-" => Some([0.0122, 0.0004, 38.0, 21.78, 19.88, 8.67, 40.99]),
        "ring >C<" => Some([0.0042, 0.0061, 27.0, 21.32, 60.15, 79.72, 87.88]),
        "ring =CH-" => Some([0.0082, 0.0011, 41.0, 26.73, 8.13, 2.09, 11.3]),
        "ring =C<" => Some([0.0143, 0.0008, 32.0, 31.01, 37.02, 46.43, 54.05]),
        "-F" => Some([0.0111, -0.0057, 27.0, -0.03, -15.78, -251.92, -247.19]),
        "-Cl" => Some([0.0105, -0.0049, 58.0, 38.13, 13.55, -71.55, -64.31]),
        "-Br" => Some([0.0133, 0.0057, 71.0, 66.86, 43.43, -29.48, -38.06]),
        "-I" => Some([0.0068, -0.0034, 97.0, 93.84, 41.69, 21.06, 5.74]),
        "-OH" => Some([0.0741, 0.0112, 28.0, 92.88, 44.45, -208.04, -189.2]),
        "phenol -OH" => Some([0.024, 0.0184, -25.0, 76.34, 82.83, -221.65, -197.37]),
        "-O-" => Some([0.0168, 0.0015, 18.0, 22.42, 22.23, -132.22, -105.0]),
        "ring -O-" => Some([0.0098, 0.0048, 13.0, 31.22, 23.05, -138.16, -98.22]),
        ">C=O" => Some([0.038, 0.0031, 62.0, 76.75, 61.2, -133.22, -120.5]),
        "ring >C=O" => Some([0.0284, 0.0028, 55.0, 94.97, 75.97, -164.5, -126.27]),
        "O=CH-" => Some([0.0379, 0.003, 82.0, 72.24, 36.9, -162.03, -143.48]),
        "-COOH" => Some([0.0791, 0.0077, 89.0, 169.09, 155.5, -426.72, -387.87]),
        "-COO-" => Some([0.0481, 0.0005, 82.0, 81.1, 53.6, -337.92, -301.95]),
        "=O" => Some([0.0143, 0.0101, 36.0, -10.5, 2.08, -247.61, -250.83]),
        "-NH2" => Some([0.0243, 0.0109, 38.0, 73.23, 66.89, -22.02, 14.07]),
        ">NH" => Some([0.0295, 0.0077, 35.0, 50.17, 52.66, 53.47, 89.39]),
        "ring >NH" => Some([0.013, 0.0114, 29.0, 52.82, 101.51, 31.65, 75.61]),
        ">N-" => Some([0.0169, 0.0074, 9.0, 11.74, 48.84, 123.34, 163.16]),
        "ring -N=" => Some([0.0085, 0.0076, 34.0, 57.55, 68.4, 55.52, 79.93]),
        "-CN" => Some([0.0496, -0.0101, 91.0, 125.66, 59.89, 88.43, 89.22]),
        "-NO2" => Some([0.0437, 0.0064, 91.0, 152.54, 127.24, -66.57, -16.83]),
        "-SH" => Some([0.0031, 0.0084, 63.0, 63.56, 20.09, -17.33, -22.99]),
        "-S-" => Some([0.0119, 0.0049, 54.0, 68.78, 34.4, 41.87, 33.12]),
        "ring -S-" => Some([0.0019, 0.0051, 38.0, 52.1, 79.93, 39.1, 27.76]),
        _ => None,
    }
}

fn compute_joback(c: &PropCtx) -> Option<JobackThermo> {
    let n = c.el.len();
    if n == 0 {
        return None;
    }
    let mut owner = vec![false; n];
    let mut groups: Vec<&'static str> = Vec::new();
    let x_deg = |i: usize| c.nb[i].len() + (c.h[i] as usize);

    let mut claim = |g: &'static str, atoms: &[usize]| -> bool {
        if atoms.iter().any(|&a| owner[a]) {
            return false;
        }
        for &a in atoms {
            owner[a] = true;
        }
        groups.push(g);
        true
    };

    let terminal_o = |i: usize, order: u8| -> Vec<usize> {
        c.nb[i]
            .iter()
            .filter(|b| {
                b.order == order && c.el[b.j] == "O" && c.nb[b.j].len() == 1 && c.h[b.j] == 0
            })
            .map(|b| b.j)
            .collect()
    };
    let is_carbonyl = |i: usize| c.el[i] == "C" && !terminal_o(i, 2).is_empty();

    for i in 0..n {
        let el = c.el[i].as_str();
        let ox2 = terminal_o(i, 2);
        let ox1: Vec<usize> = c.nb[i]
            .iter()
            .filter(|b| b.order == 1 && c.el[b.j] == "O" && x_deg(b.j) == 2)
            .map(|b| b.j)
            .collect();
        if el == "N" && c.nb[i].len() == 3 && !ox2.is_empty() {
            let mut all_term = terminal_o(i, 1);
            all_term.extend(ox2.iter().copied());
            if all_term.len() == 2 && c.nb[i].iter().any(|b| c.el[b.j] != "O") {
                let mut grp_atoms = vec![i];
                grp_atoms.extend(all_term);
                if !claim("-NO2", &grp_atoms) {
                    return None;
                }
            }
        } else if el == "C" && x_deg(i) == 2 {
            if let Some(t) = c.nb[i]
                .iter()
                .find(|b| b.order == 3 && c.el[b.j] == "N" && c.nb[b.j].len() == 1 && c.h[b.j] == 0)
            {
                if !claim("-CN", &[i, t.j]) {
                    return None;
                }
            }
        } else if el == "C" && !ox2.is_empty() && x_deg(i) == 3 {
            let o = ox2[0];
            let oh = ox1.iter().copied().find(|&j| c.h[j] == 1);
            let ok = if let Some(oh_idx) = oh {
                !c.arom[i] && claim("-COOH", &[i, o, oh_idx])
            } else if c.h[i] == 0 && ox1.len() == 1 && c.h[ox1[0]] == 0 {
                claim("-COO-", &[i, o, ox1[0]])
            } else if !ox1.is_empty() {
                false
            } else if c.h[i] == 1 && c.nb[i].iter().any(|b| c.el[b.j] == "C") {
                claim("O=CH-", &[i, o])
            } else if c.h[i] == 0 {
                claim(if c.ring_atom[i] { "ring >C=O" } else { ">C=O" }, &[i, o])
            } else {
                false
            };
            if !ok {
                return None;
            }
        }
    }

    for (i, &is_owned) in owner.iter().enumerate().take(n) {
        if is_owned {
            continue;
        }
        if c.q[i] != 0 {
            return None;
        }
        let el = c.el[i].as_str();
        let h = c.h[i];
        let multi: Vec<&HeavyEdge> = c.nb[i].iter().filter(|b| b.order > 1).collect();
        let in_ring = c.ring_atom[i];

        let g: Option<&'static str> = match el {
            "C" if x_deg(i) == 4 && multi.is_empty() => match (in_ring, h) {
                (_, 3) => Some("-CH3"),
                (false, 2) => Some("-CH2-"),
                (false, 1) => Some(">CH-"),
                (false, 0) => Some(">C<"),
                (true, 2) => Some("ring -CH2-"),
                (true, 1) => Some("ring >CH-"),
                (true, 0) => Some("ring >C<"),
                _ => None,
            },
            "C" if x_deg(i) == 3 && multi.len() == 1 && multi[0].order == 2 => match (in_ring, h) {
                (_, 2) => Some("=CH2"),
                (false, 1) => Some("=CH-"),
                (false, 0) => Some("=C<"),
                (true, 1) => Some("ring =CH-"),
                (true, 0) => Some("ring =C<"),
                _ => None,
            },
            "C" if x_deg(i) == 2 => {
                if multi.len() == 2 {
                    Some("=C=")
                } else if multi
                    .first()
                    .is_some_and(|m| m.order == 3 && c.el[m.j] != "N")
                {
                    Some(if h >= 1 { "≡CH" } else { "≡C-" })
                } else {
                    None
                }
            }
            "N" if x_deg(i) == 3 => match (in_ring, h) {
                (_, 2) => Some("-NH2"),
                (false, 1) => Some(">NH"),
                (true, 1) => Some("ring >NH"),
                (_, 0) => Some(">N-"),
                _ => None,
            },
            "N" if x_deg(i) == 2 && h == 0 && in_ring => Some("ring -N="),
            "O" if x_deg(i) == 2 && !c.nb[i].iter().any(|b| is_carbonyl(b.j)) => {
                if h == 1 {
                    Some(if c.nb[i].iter().any(|b| c.arom[b.j]) {
                        "phenol -OH"
                    } else {
                        "-OH"
                    })
                } else if h == 0 {
                    Some(if in_ring { "ring -O-" } else { "-O-" })
                } else {
                    None
                }
            }
            "O" if x_deg(i) == 1 && !multi.is_empty() && c.el[multi[0].j] != "C" => Some("=O"),
            "S" if x_deg(i) == 2 && multi.is_empty() => {
                if h == 1 {
                    Some("-SH")
                } else if h == 0 {
                    Some(if in_ring { "ring -S-" } else { "-S-" })
                } else {
                    None
                }
            }
            "F" if x_deg(i) == 1 => Some("-F"),
            "Cl" if x_deg(i) == 1 => Some("-Cl"),
            "Br" if x_deg(i) == 1 => Some("-Br"),
            "I" if x_deg(i) == 1 => Some("-I"),
            _ => None,
        };
        groups.push(g?);
    }

    let mut s = [0.0f64; 7];
    for g in groups {
        let row = joback_contrib(g)?;
        for k in 0..7 {
            s[k] += row[k];
        }
    }
    let total_atoms = n + c.h.iter().map(|&x| x as usize).sum::<usize>();
    let tb = 198.2 + s[3];
    let tc_denom = 0.584 + 0.965 * s[0] - s[0] * s[0];
    let pc_base = 0.113 + 0.0032 * (total_atoms as f64) - s[1];
    if tc_denom <= 0.0 || pc_base <= 0.0 {
        return None;
    }

    Some(JobackThermo {
        boiling_point_k: tb,
        melting_point_k: 122.5 + s[4],
        critical_temp_k: tb / tc_denom,
        critical_pressure_bar: pc_base.powi(-2),
        critical_volume_cm3_mol: 17.5 + s[2],
        enthalpy_formation_kj_mol: 68.29 + s[5],
        gibbs_energy_kj_mol: 53.88 + s[6],
    })
}

/// Detect typical ionizable pKa sites (carboxylic acids, sulfonic acids, phenols, thiols,
/// sulfonamides, anilines, pyridines, and primary/secondary/tertiary amines).
fn compute_pka_sites(c: &PropCtx) -> Vec<PkaSite> {
    let n = c.el.len();
    let mut h = c.h.clone();
    let mut q = c.q.clone();
    let mut taken = vec![false; n];

    // Normalize simple deprotonated O-/S- and protonated NH+ to neutral form (excluding nitro N+O-)
    for i in 0..n {
        if (c.el[i] == "O" || c.el[i] == "S") && q[i] == -1 && c.nb[i].len() == 1 {
            let nb = c.nb[i][0].j;
            if c.el[nb] == "N" && c.q[nb] > 0 {
                taken[i] = true;
                taken[nb] = true;
            } else {
                h[i] += 1;
                q[i] = 0;
            }
        } else if c.el[i] == "N" && q[i] == 1 && h[i] > 0 {
            h[i] -= 1;
            q[i] = 0;
        } else if c.el[i] == "N"
            && c.nb[i]
                .iter()
                .any(|b| b.order == 2 && c.el[b.j] == "O" && !c.arom_bond[b.k])
        {
            taken[i] = true;
        }
    }

    let mut out = Vec::new();

    for i in 0..n {
        if taken[i] {
            continue;
        }
        let el = c.el[i].as_str();
        if el == "O" && !c.arom[i] && h[i] > 0 && q[i] == 0 {
            // Check sulfonic acid, carboxylic acid, phenol, alcohol
            if let Some(s_nb) = c.nb[i].iter().find(|b| {
                b.order == 1
                    && c.el[b.j] == "S"
                    && c.nb[b.j]
                        .iter()
                        .filter(|x| x.order == 2 && c.el[x.j] == "O")
                        .count()
                        == 2
            }) {
                taken[i] = true;
                taken[s_nb.j] = true;
                out.push(PkaSite {
                    atom_index: c.of_mol[i],
                    kind: "sulfonic acid".to_string(),
                    pka: -1.82,
                    sd: 1.41,
                });
                continue;
            }
            if let Some(c_nb) = c.nb[i].iter().find(|b| {
                b.order == 1
                    && c.el[b.j] == "C"
                    && !c.arom[b.j]
                    && c.nb[b.j].iter().any(|x| x.order == 2 && c.el[x.j] == "O")
            }) {
                taken[i] = true;
                for x in &c.nb[c_nb.j] {
                    if c.el[x.j] == "O" {
                        taken[x.j] = true;
                    }
                }
                out.push(PkaSite {
                    atom_index: c.of_mol[i],
                    kind: "carboxylic acid".to_string(),
                    pka: 3.46,
                    sd: 1.29,
                });
                continue;
            }
            if c.nb[i].iter().any(|b| b.order == 1 && c.arom[b.j]) {
                taken[i] = true;
                out.push(PkaSite {
                    atom_index: c.of_mol[i],
                    kind: "phenol".to_string(),
                    pka: 9.95,
                    sd: 0.85,
                });
                continue;
            }
        } else if el == "S" && !c.arom[i] && h[i] > 0 && q[i] == 0 {
            taken[i] = true;
            let aryl = c.nb[i].iter().any(|b| c.arom[b.j]);
            out.push(PkaSite {
                atom_index: c.of_mol[i],
                kind: if aryl { "thiophenol" } else { "thiol" }.to_string(),
                pka: if aryl { 6.50 } else { 10.40 },
                sd: 0.80,
            });
        } else if el == "N" && q[i] == 0 {
            let x_deg = c.nb[i].len() + (h[i] as usize);
            if c.arom[i] && h[i] == 0 && x_deg == 2 {
                taken[i] = true;
                out.push(PkaSite {
                    atom_index: c.of_mol[i],
                    kind: "aromatic N".to_string(),
                    pka: 4.35,
                    sd: 2.07,
                });
            } else if !c.arom[i] && x_deg == 3 {
                let bonded_to_carbonyl = c.nb[i].iter().any(|b| {
                    c.el[b.j] == "C"
                        && !c.arom[b.j]
                        && c.nb[b.j]
                            .iter()
                            .any(|x| x.order == 2 && matches!(c.el[x.j].as_str(), "O" | "S" | "N"))
                });
                let bonded_to_so2 = c.nb[i].iter().any(|b| {
                    c.el[b.j] == "S"
                        && c.nb[b.j]
                            .iter()
                            .filter(|x| x.order == 2 && c.el[x.j] == "O")
                            .count()
                            >= 2
                });
                if bonded_to_so2 && h[i] >= 1 {
                    taken[i] = true;
                    out.push(PkaSite {
                        atom_index: c.of_mol[i],
                        kind: "sulfonamide".to_string(),
                        pka: 10.16,
                        sd: 0.55,
                    });
                } else if !bonded_to_carbonyl && !bonded_to_so2 {
                    taken[i] = true;
                    if c.nb[i].iter().any(|b| c.arom[b.j]) {
                        out.push(PkaSite {
                            atom_index: c.of_mol[i],
                            kind: "aniline".to_string(),
                            pka: 4.30,
                            sd: 1.80,
                        });
                    } else {
                        let (kind_s, pka_v) = match h[i] {
                            2 => ("primary amine", 10.55),
                            1 => ("secondary amine", 11.02),
                            _ => ("tertiary amine", 10.28),
                        };
                        out.push(PkaSite {
                            atom_index: c.of_mol[i],
                            kind: kind_s.to_string(),
                            pka: pka_v,
                            sd: 0.35,
                        });
                    }
                }
            }
        }
    }

    out
}

/// Compute all physicochemical & mass-spectrometric molecular properties on `mol`.
pub fn compute_phys_chem_props(mol: &AnalytMol) -> Result<PhysChemProps, String> {
    let (counts, net_charge) = element_counts_of_mol(mol);
    if counts.is_empty() {
        return Err("Molecule contains no atoms".to_string());
    }
    let formula = hill_formula(&counts);
    let mw = molecular_weight(&counts)
        .ok_or_else(|| "Unsupported isotope/element in molecular weight calculation".to_string())?;
    let mono = monoisotopic_mass(&counts).ok_or_else(|| {
        "Unsupported isotope/element in monoisotopic mass calculation".to_string()
    })?;
    let elem_pct = elemental_composition(&counts)
        .ok_or_else(|| "Failed to compute elemental composition".to_string())?;
    let elem_txt = format_elemental_analysis(&elem_pct);
    let rdbe = rdbe_from_counts(&counts, net_charge);

    let ctx = build_prop_ctx(mol);
    let tpsa = compute_ertl_tpsa(&ctx);
    let rotatable_bonds = compute_rotatable_bonds(&ctx);
    let (clogp, molar_refractivity) = compute_crippen_clogp_mr(&ctx);

    let heavy_count = ctx.el.iter().filter(|e| e.as_str() != "H").count();
    let arom_heavy = ctx
        .el
        .iter()
        .enumerate()
        .filter(|&(i, e)| e.as_str() != "H" && ctx.arom[i])
        .count();
    let aromatic_proportion = if heavy_count > 0 {
        (arom_heavy as f64) / (heavy_count as f64)
    } else {
        0.0
    };

    // Delaney ESOL logS (PatWalters Crippen-refit coefficients matching `props.ts`)
    let esol_logs = 0.261_210_661_378_016_96
        - 0.741_673_952_340_899_5 * clogp
        - 0.006_613_884_773_866_712_5 * mw
        + 0.003_451_545_565_957_996 * (rotatable_bonds as f64)
        - 0.426_248_404_413_169_75 * aromatic_proportion;

    let h_bond_donors = ctx
        .el
        .iter()
        .enumerate()
        .filter(|&(i, e)| matches!(e.as_str(), "N" | "O") && ctx.h[i] > 0)
        .count();
    let h_bond_acceptors = ctx
        .el
        .iter()
        .enumerate()
        .filter(|&(i, e)| {
            (e == "O") || (e == "N" && !(ctx.q[i] > 0 && ctx.nb[i].len() + ctx.h[i] as usize >= 4))
        })
        .count();

    let lipinski_ro5_pass =
        mw <= 500.0 && clogp <= 5.0 && h_bond_donors <= 5 && h_bond_acceptors <= 10;

    let joback = compute_joback(&ctx);
    let pka_sites = compute_pka_sites(&ctx);

    Ok(PhysChemProps {
        formula,
        molecular_weight: mw,
        monoisotopic_mass: mono,
        elemental_percent: elem_pct,
        elemental_analysis_text: elem_txt,
        rdbe,
        clogp,
        molar_refractivity,
        tpsa,
        h_bond_donors,
        h_bond_acceptors,
        rotatable_bonds,
        aromatic_proportion,
        esol_logs,
        lipinski_ro5_pass,
        joback,
        pka_sites,
    })
}
