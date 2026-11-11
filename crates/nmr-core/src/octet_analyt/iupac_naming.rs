//! Offline bidirectional IUPAC Structure ↔ Preferred Name engine (`mol_to_iupac_name` and
//! `iupac_name_to_mol` / `iupac_name_to_smiles`).
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules:
//! - Favre, H. A.; Powell, W. H. *Nomenclature of Organic Chemistry: IUPAC Recommendations and
//!   Preferred Names 2013* (Blue Book), Royal Society of Chemistry, 2014 (P-14, P-15.3, P-22,
//!   P-23/von Baeyer, P-24/spiro, P-25/fused, P-44 seniority, P-51, P-61..P-68, P-91..P-93).
//! - Moss, G. P. *Pure Appl. Chem.* **1999**, 71, 513–529 (von Baeyer polycyclic nomenclature).
//! - Connelly, N. G. et al. *Nomenclature of Inorganic Chemistry: IUPAC Recommendations 2005* (Red Book).

use crate::octet_analyt::{
    AnalytAtom, AnalytBond, AnalytMol, AtomStereoTag, BondStereoTag, PerceivedStereo,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

// ============================================================================
// Part 1: Stems, Multipliers, Alphabetical Assembly & Von Baeyer / Spiro Rings
// ============================================================================

/// IUPAC hydrocarbon chain stem for `n` carbons (`1 -> "meth"`, `2 -> "eth"`, ..., `20 -> "icos"`).
pub fn alkane_stem(n: usize) -> String {
    const STEMS: [&str; 22] = [
        "", "meth", "eth", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec", "undec",
        "dodec", "tridec", "tetradec", "pentadec", "hexadec", "heptadec", "octadec", "nonadec",
        "icos", "henicos",
    ];
    if n < STEMS.len() {
        return STEMS[n].to_string();
    }
    let units = [
        "", "hen", "do", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona",
    ];
    let tens = [
        "",
        "",
        "cos",
        "triacont",
        "tetracont",
        "pentacont",
        "hexacont",
    ];
    let u = n % 10;
    let t = n / 10;
    if t < tens.len() {
        format!("{}{}", units[u], tens[t])
    } else {
        format!("alk{n}")
    }
}

/// Parse an IUPAC chain stem (`"meth"` -> `1`, `"but"` -> `4`, `"hept"` -> `7`, etc.).
pub fn stem_to_length(stem: &str) -> Option<usize> {
    let s = stem.trim().to_ascii_lowercase();
    let s = s.strip_suffix('a').unwrap_or(&s);
    (1..=30).find(|&n| alkane_stem(n) == s)
}

fn simple_mult_prefix(count: usize) -> &'static str {
    match count {
        0 | 1 => "",
        2 => "di",
        3 => "tri",
        4 => "tetra",
        5 => "penta",
        6 => "hexa",
        7 => "hepta",
        8 => "octa",
        9 => "nona",
        10 => "deca",
        _ => "poly",
    }
}

fn complex_mult_prefix(count: usize) -> &'static str {
    match count {
        0 | 1 => "",
        2 => "bis",
        3 => "tris",
        4 => "tetrakis",
        5 => "pentakis",
        6 => "hexakis",
        _ => "polykis",
    }
}

fn to_superscript_digits(n: usize) -> String {
    const SUP: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string()
        .bytes()
        .map(|b| SUP[(b - b'0') as usize])
        .collect()
}

/// Alphabetical sort key per IUPAC P-14.5 (stripping leading locants and punctuation; multiplicative
/// prefixes on simple substituents are added only *after* sorting).
fn iupac_alpha_key(sub_name: &str) -> String {
    sub_name
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[derive(Debug, Clone)]
struct SubPrefixItem {
    locant: String,
    base_name: String,
    is_compound: bool,
}

fn locant_sort_tuple(loc: &str) -> (u8, usize, String) {
    if let Ok(num) = loc.trim_end_matches('\'').parse::<usize>() {
        let primes = loc.chars().filter(|&c| c == '\'').count();
        (1, num * 10 + primes, loc.to_string())
    } else {
        // Letter locants like "N" come before numeric locants in IUPAC amide/amine prefixes (`N,2-dimethyl...`)
        (0, 0, loc.to_string())
    }
}

/// Assemble substituent prefixes alphabetically (`"4-amino"`, `"3-bromo-4-chloro"`, `"2,2-dimethyl"`,
/// `"2-[4-(2-methylpropyl)phenyl]"`), eliding trailing `'a'` of multiplier Only when appropriate (`P-16.3`).
fn assemble_prefixes(items: &[SubPrefixItem], drop_terminal_e_check: bool) -> String {
    if items.is_empty() {
        return String::new();
    }
    let mut grouped: BTreeMap<String, (Vec<String>, bool, String)> = BTreeMap::new();
    for it in items {
        let key = format!("{}|{}", iupac_alpha_key(&it.base_name), it.base_name);
        let entry = grouped
            .entry(key)
            .or_insert_with(|| (Vec::new(), it.is_compound, it.base_name.clone()));
        entry.0.push(it.locant.clone());
        entry.1 |= it.is_compound;
    }

    let mut sorted_groups: Vec<(Vec<String>, bool, String)> = grouped.into_values().collect();
    sorted_groups.sort_by(|a, b| {
        iupac_alpha_key(&a.2)
            .cmp(&iupac_alpha_key(&b.2))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut parts: Vec<String> = Vec::with_capacity(sorted_groups.len());
    for (mut locs, is_comp, base) in sorted_groups {
        locs.sort_by_key(|x| locant_sort_tuple(x));
        let cnt = locs.len();
        let loc_str = locs
            .into_iter()
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(",");
        let loc_prefix = if loc_str.is_empty() {
            String::new()
        } else {
            format!("{loc_str}-")
        };

        let body = if cnt == 1 {
            if is_comp {
                if base.contains('(') {
                    format!("[{base}]")
                } else {
                    format!("({base})")
                }
            } else {
                base
            }
        } else if is_comp
            || base.starts_with("dec")
            || base.starts_with("cos")
            || base.contains('-')
            || base.starts_with("oxy")
        {
            let mp = complex_mult_prefix(cnt);
            if base.contains('(') {
                format!("{mp}[{base}]")
            } else {
                format!("{mp}({base})")
            }
        } else {
            let mp = simple_mult_prefix(cnt);
            format!("{mp}{base}")
        };
        let _ = drop_terminal_e_check;
        parts.push(format!("{loc_prefix}{body}"));
    }
    parts.join("-")
}

// --- Principal Functional Group Seniority (IUPAC P-41 / P-44.1) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PrincipalGroup {
    CarboxylicAcid = 1,
    SulfonicAcid = 2,
    Ester = 3,
    AcidHalide = 4,
    Amide = 5,
    Nitrile = 6,
    Aldehyde = 7,
    Ketone = 8,
    Alcohol = 9,
    Thiol = 10,
    Amine = 11,
    None = 99,
}

#[derive(Debug, Clone)]
struct FunctionalHit {
    kind: PrincipalGroup,
    /// Atom that carries the group (`C` for `-COOH`/`-COOR`/`-CONH2`/`-CN`/`-CHO`/`>C=O`; `O`/`S`/`N` for `-OH`/`-SH`/`-NH2`; `S` for `-SO3H`).
    group_atom: usize,
    /// Ring/chain skeleton atom to which the group attaches (equals `group_atom` for chain `-COOH`/`-CN`/`-CHO`/`>C=O`,
    /// or the attached carbon for `-OH`/`-SH`/`-NH2` and ring-attached `-COOH`).
    anchor_carbon: usize,
    /// All non-skeleton atoms consumed by this functional group (e.g. `=O` and `-OH` of `-COOH`, so they aren't re-named as prefixes).
    group_atoms: Vec<usize>,
    /// Extra detail (e.g. halide element `"Cl"` for acid halide, or ester oxygen index).
    extra_atom: Option<usize>,
}

// --- Small Inorganic & Main-Group Hydride Namer (`namer-maingroup.ts`) ---

pub(crate) fn try_inorganic_or_hydride_name(mol: &AnalytMol) -> Option<String> {
    let adj = mol.adjacency();
    // Fold explicit neutral 1H atoms into heavy counts
    let keep: Vec<usize> = (0..mol.atoms.len())
        .filter(|&i| {
            !(mol.atoms[i].element == "H"
                && mol.atoms[i].isotope.unwrap_or(1) == 1
                && mol.atoms[i].formal_charge == 0
                && adj[i].len() == 1
                && mol.atoms[adj[i][0].0].element != "H")
        })
        .collect();
    if keep.is_empty() {
        return None;
    }

    // Single heavy-atom hydrides & simple ions
    if keep.len() == 1 {
        let idx = keep[0];
        let a = &mol.atoms[idx];
        let h = mol.total_h_count(idx);
        return match (a.element.as_str(), h, a.formal_charge) {
            ("O", 2, 0) => Some("oxidane".to_string()),
            ("O", 3, 1) => Some("oxidanium".to_string()),
            ("O", 1, -1) => Some("hydroxide".to_string()),
            ("N", 3, 0) => Some("azane".to_string()),
            ("N", 4, 1) => Some("ammonium".to_string()),
            ("N", 2, -1) => Some("azanide".to_string()),
            ("P", 3, 0) => Some("phosphane".to_string()),
            ("P", 4, 1) => Some("phosphanium".to_string()),
            ("S", 2, 0) => Some("sulfane".to_string()),
            ("Si", 4, 0) => Some("silane".to_string()),
            ("B", 3, 0) => Some("borane".to_string()),
            ("B", 4, -1) => Some("boranuide".to_string()),
            ("C", 4, 0) => Some("methane".to_string()),
            ("F", 1, 0) => Some("fluorane".to_string()),
            ("Cl", 1, 0) => Some("chlorane".to_string()),
            ("Br", 1, 0) => Some("bromane".to_string()),
            ("I", 1, 0) => Some("iodane".to_string()),
            ("Na", 0, 1) => Some("sodium(1+)".to_string()),
            ("K", 0, 1) => Some("potassium(1+)".to_string()),
            ("Li", 0, 1) => Some("lithium(1+)".to_string()),
            ("Cl", 0, -1) => Some("chloride".to_string()),
            ("Br", 0, -1) => Some("bromide".to_string()),
            ("I", 0, -1) => Some("iodide".to_string()),
            ("F", 0, -1) => Some("fluoride".to_string()),
            _ => None,
        };
    }

    // Homogeneous main-group catenated hydrides (diazane/hydrazine, disilane, diphosphane, hydrogen peroxide/dioxidane)
    if keep
        .iter()
        .all(|&i| mol.atoms[i].element != "C" && mol.atoms[i].formal_charge == 0)
    {
        let first_el = mol.atoms[keep[0]].element.as_str();
        if keep.iter().all(|&i| mol.atoms[i].element == first_el) && keep.len() == 2 {
            if let Some(bi) = mol.bond_between(keep[0], keep[1]) {
                let bo = mol.bonds[bi].order;
                match (first_el, bo) {
                    ("N", 1)
                        if mol.total_h_count(keep[0]) == 2 && mol.total_h_count(keep[1]) == 2 =>
                    {
                        return Some("hydrazine".to_string());
                    }
                    ("N", 2)
                        if mol.total_h_count(keep[0]) == 1 && mol.total_h_count(keep[1]) == 1 =>
                    {
                        return Some("diazene".to_string());
                    }
                    ("O", 1)
                        if mol.total_h_count(keep[0]) == 1 && mol.total_h_count(keep[1]) == 1 =>
                    {
                        return Some("hydrogen peroxide".to_string());
                    }
                    ("Si", 1) => return Some("disilane".to_string()),
                    ("P", 1) => return Some("diphosphane".to_string()),
                    ("S", 1) => return Some("disulfane".to_string()),
                    _ => {}
                }
            }
        }
        // Common inorganic oxoacids / oxides (`O=C=O` handled below, `H2SO4`, `HNO3`, `SO2`)
        if keep.len() == 3 {
            let els: BTreeMap<&str, usize> = keep.iter().fold(BTreeMap::new(), |mut m, &i| {
                *m.entry(mol.atoms[i].element.as_str()).or_insert(0) += 1;
                m
            });
            if els.get("S") == Some(&1) && els.get("O") == Some(&2) {
                let s_idx = *keep.iter().find(|&&i| mol.atoms[i].element == "S").unwrap();
                if mol.bond_order_sum(s_idx) == 4 && keep.iter().all(|&i| mol.total_h_count(i) == 0)
                {
                    return Some("sulfur dioxide".to_string());
                }
            }
        }
        if keep.len() == 4 {
            let n_cnt = keep
                .iter()
                .filter(|&&i| mol.atoms[i].element == "N")
                .count();
            let o_cnt = keep
                .iter()
                .filter(|&&i| mol.atoms[i].element == "O")
                .count();
            let tot_h: u8 = keep.iter().map(|&i| mol.total_h_count(i)).sum();
            if n_cnt == 1 && o_cnt == 3 && tot_h == 1 {
                return Some("nitric acid".to_string());
            }
        }
        if keep.len() == 5 {
            let s_cnt = keep
                .iter()
                .filter(|&&i| mol.atoms[i].element == "S")
                .count();
            let o_cnt = keep
                .iter()
                .filter(|&&i| mol.atoms[i].element == "O")
                .count();
            let tot_h: u8 = keep.iter().map(|&i| mol.total_h_count(i)).sum();
            if s_cnt == 1 && o_cnt == 4 && tot_h == 2 {
                return Some("sulfuric acid".to_string());
            }
            let p_cnt = keep
                .iter()
                .filter(|&&i| mol.atoms[i].element == "P")
                .count();
            if p_cnt == 1 && o_cnt == 4 && tot_h == 3 {
                return Some("phosphoric acid".to_string());
            }
        }
    }

    // Carbon dioxide (`O=C=O`)
    if keep.len() == 3 {
        let c_cnt = keep
            .iter()
            .filter(|&&i| mol.atoms[i].element == "C")
            .count();
        let o_cnt = keep
            .iter()
            .filter(|&&i| mol.atoms[i].element == "O")
            .count();
        if c_cnt == 1 && o_cnt == 2 && mol.bonds.iter().filter(|b| b.order == 2).count() == 2 {
            return Some("carbon dioxide".to_string());
        }
    }

    None
}

// --- Von Baeyer Bridged Bicyclo / Polycyclo & Monospiro Ring Systems (`vonbaeyer.ts`) ---

#[derive(Debug, Clone)]
pub(crate) struct VonBaeyerNumbering {
    /// Prefix descriptor such as `"bicyclo[2.2.1]"`, `"tricyclo[3.3.1.1³,⁷]"`, or `"spiro[4.5]"`.
    pub descriptor: String,
    /// Candidate atom-index orderings (`order[locant - 1] = atom_index`), allowing lowest-locant selection for unsaturations/groups.
    pub candidate_orders: Vec<Vec<usize>>,
}

fn skeletal_aza_oxa_prefix(mol: &AnalytMol, numbering: &[usize]) -> String {
    // Cite oxa > thia > selena > aza > phospha > sila > bora in IUPAC P-15.4 order
    const ORDER: [(&str, &str); 7] = [
        ("O", "oxa"),
        ("S", "thia"),
        ("Se", "selena"),
        ("N", "aza"),
        ("P", "phospha"),
        ("Si", "sila"),
        ("B", "bora"),
    ];
    let mut parts = Vec::new();
    for &(el, term) in &ORDER {
        let locs: Vec<String> = numbering
            .iter()
            .enumerate()
            .filter(|&(_, &atom_idx)| mol.atoms[atom_idx].element == el)
            .map(|(pos, _)| (pos + 1).to_string())
            .collect();
        if !locs.is_empty() {
            let mp = simple_mult_prefix(locs.len());
            parts.push(format!("{}-{mp}{term}", locs.join(",")));
        }
    }
    parts.join("-")
}

/// Analyze a connected ring system (`ring_atoms`) for monospiro `[a.b]`, bicyclo `[a.b.c]`,
/// or polycyclo `[a.b.c.d^{x,y}]` von Baeyer topology (`vonbaeyer.ts`).
pub(crate) fn analyze_von_baeyer_or_spiro(
    mol: &AnalytMol,
    ring_atoms: &[usize],
) -> Option<VonBaeyerNumbering> {
    let n = ring_atoms.len();
    if !(4..=40).contains(&n) {
        return None;
    }
    let to_local: HashMap<usize, usize> = ring_atoms
        .iter()
        .copied()
        .enumerate()
        .map(|(i, a)| (a, i))
        .collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for b in &mol.bonds {
        if b.order == 0 {
            continue;
        }
        if let (Some(&la), Some(&lb)) = (to_local.get(&b.a), to_local.get(&b.b)) {
            adj[la].push(lb);
            adj[lb].push(la);
            let (u, v) = if la < lb { (la, lb) } else { (lb, la) };
            if !edges.contains(&(u, v)) {
                edges.push((u, v));
            }
        }
    }
    let cyclomatic = (edges.len() as isize) - (n as isize) + 1;
    if !(2..=8).contains(&cyclomatic) {
        return None;
    }

    // 1. Check monospiro (`cyclomatic == 2`, single shared spiro atom of ring-degree 4, all others degree 2)
    if cyclomatic == 2 {
        let deg4: Vec<usize> = (0..n).filter(|&i| adj[i].len() == 4).collect();
        if deg4.len() == 1 && (0..n).all(|i| i == deg4[0] || adj[i].len() == 2) {
            let sp = deg4[0];
            // Trace the two loops starting from `sp`
            let mut used_nb = HashSet::new();
            let mut loops: Vec<Vec<usize>> = Vec::new();
            for &nb in &adj[sp] {
                if used_nb.contains(&nb) {
                    continue;
                }
                let mut chain = vec![nb];
                let mut prev = sp;
                let mut cur = nb;
                while cur != sp {
                    let nxt = *adj[cur].iter().find(|&&x| x != prev)?;
                    if nxt == sp {
                        used_nb.insert(cur);
                        break;
                    }
                    chain.push(nxt);
                    prev = cur;
                    cur = nxt;
                }
                used_nb.insert(nb);
                loops.push(chain);
            }
            if loops.len() == 2 {
                // IUPAC P-24.2.1: smaller ring first `[a.b]` (`a <= b`), numbered around smaller ring starting adjacent to spiro atom,
                // then spiro atom itself, then larger ring!
                loops.sort_by_key(|l| l.len());
                let (small, large) = (&loops[0], &loops[1]);
                let a_len = small.len();
                let b_len = large.len();
                let mut orders = Vec::new();
                let small_dirs = [small.clone(), small.iter().copied().rev().collect()];
                let large_dirs = [large.clone(), large.iter().copied().rev().collect()];
                let both_ways = if a_len == b_len {
                    vec![(&small_dirs, &large_dirs), (&large_dirs, &small_dirs)]
                } else {
                    vec![(&small_dirs, &large_dirs)]
                };
                for (sd, ld) in both_ways {
                    for s_seq in sd {
                        for l_seq in ld {
                            let mut ord = Vec::with_capacity(n);
                            ord.extend(s_seq.iter().map(|&x| ring_atoms[x]));
                            ord.push(ring_atoms[sp]);
                            ord.extend(l_seq.iter().map(|&x| ring_atoms[x]));
                            orders.push(ord);
                        }
                    }
                }
                return Some(VonBaeyerNumbering {
                    descriptor: format!("spiro[{a_len}.{b_len}]"),
                    candidate_orders: orders,
                });
            }
        }

        // 2. Bicyclo `[a.b.c]` (`cyclomatic == 2`, two bridgehead atoms of ring-degree 3, all others degree 2)
        let deg3: Vec<usize> = (0..n).filter(|&i| adj[i].len() == 3).collect();
        if deg3.len() == 2 && (0..n).all(|i| deg3.contains(&i) || adj[i].len() == 2) {
            let u = deg3[0];
            let v = deg3[1];
            let mut paths: Vec<Vec<usize>> = Vec::new(); // internal atoms from `u` to `v`
            for &nb in &adj[u] {
                let mut inner = Vec::new();
                let mut prev = u;
                let mut cur = nb;
                while cur != v {
                    inner.push(cur);
                    let nxt = *adj[cur].iter().find(|&&x| x != prev)?;
                    prev = cur;
                    cur = nxt;
                }
                paths.push(inner);
            }
            if paths.len() == 3 {
                // Sort bridges descending by length: `[a.b.c]` with `a >= b >= c`
                let mut idx_paths: Vec<usize> = vec![0, 1, 2];
                idx_paths.sort_by(|&i, &j| paths[j].len().cmp(&paths[i].len()));
                let a_len = paths[idx_paths[0]].len();
                let b_len = paths[idx_paths[1]].len();
                let c_len = paths[idx_paths[2]].len();

                // Generate all valid IUPAC P-23.2.3 numberings:
                // Start at either bridgehead (h1=u, h2=v or h1=v, h2=u), walk longest bridge (any tie-breaking permutation),
                // then h2, then second-longest bridge back toward h1, then shortest bridge from end nearer h1!
                let perms: [[usize; 3]; 6] = [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ];
                let mut orders = Vec::new();
                for perm in &perms {
                    if paths[perm[0]].len() != a_len
                        || paths[perm[1]].len() != b_len
                        || paths[perm[2]].len() != c_len
                    {
                        continue;
                    }
                    for &(h1, h2, rev_u_to_v) in &[(u, v, false), (v, u, true)] {
                        let p_long: Vec<usize> = if rev_u_to_v {
                            paths[perm[0]].iter().copied().rev().collect()
                        } else {
                            paths[perm[0]].clone()
                        };
                        // Second bridge runs from h2 back to h1!
                        let p_mid: Vec<usize> = if rev_u_to_v {
                            paths[perm[1]].clone()
                        } else {
                            paths[perm[1]].iter().copied().rev().collect()
                        };
                        // Third bridge runs from h1 toward h2!
                        let p_short: Vec<usize> = if rev_u_to_v {
                            paths[perm[2]].iter().copied().rev().collect()
                        } else {
                            paths[perm[2]].clone()
                        };

                        let mut ord = Vec::with_capacity(n);
                        ord.push(ring_atoms[h1]);
                        ord.extend(p_long.into_iter().map(|x| ring_atoms[x]));
                        ord.push(ring_atoms[h2]);
                        ord.extend(p_mid.into_iter().map(|x| ring_atoms[x]));
                        ord.extend(p_short.into_iter().map(|x| ring_atoms[x]));
                        if !orders.contains(&ord) {
                            orders.push(ord);
                        }
                    }
                }
                return Some(VonBaeyerNumbering {
                    descriptor: format!("bicyclo[{a_len}.{b_len}.{c_len}]"),
                    candidate_orders: orders,
                });
            }
        }
        return None;
    }

    // 3. Polycyclo (`cyclomatic >= 3`, e.g. adamantane `tricyclo[3.3.1.1³,⁷]`, cubane `pentacyclo[...]`)
    let rings = cyclomatic as usize;
    let mut parent = vec![usize::MAX; n];
    let mut depth = vec![usize::MAX; n];
    let mut q = VecDeque::new();
    depth[0] = 0;
    q.push_back(0);
    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if depth[v] == usize::MAX {
                depth[v] = depth[u] + 1;
                parent[v] = u;
                q.push_back(v);
            }
        }
    }
    if depth.contains(&usize::MAX) {
        return None;
    }

    let edge_idx = |i: usize, j: usize| -> Option<usize> {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        edges.iter().position(|&e| e == (a, b))
    };

    let mut basis: Vec<u64> = Vec::new();
    for (ei, &(i, j)) in edges.iter().enumerate() {
        if parent[i] == j || parent[j] == i {
            continue;
        }
        if ei >= 62 {
            return None;
        }
        let mut mask = 1u64 << ei;
        let mut a = i;
        let mut b = j;
        while a != b {
            if depth[a] < depth[b] {
                std::mem::swap(&mut a, &mut b);
            }
            let pe = edge_idx(a, parent[a])?;
            mask ^= 1u64 << pe;
            a = parent[a];
        }
        basis.push(mask);
    }
    if basis.len() != rings || rings > 10 {
        return None;
    }

    let mut longest_cycles: Vec<Vec<usize>> = Vec::new();
    let mut best_len = 0usize;
    let mut mask = 0u64;
    for s in 1usize..(1usize << rings) {
        let tz = s.trailing_zeros() as usize;
        mask ^= basis[tz];
        let e_cnt = mask.count_ones() as usize;
        if e_cnt < best_len || e_cnt < 3 {
            continue;
        }
        let mut sub_nb: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut first_v = 0usize;
        for (ei, &(u, v)) in edges.iter().enumerate() {
            if ((mask >> ei) & 1) == 1 {
                first_v = u;
                sub_nb.entry(u).or_default().push(v);
                sub_nb.entry(v).or_default().push(u);
            }
        }
        if sub_nb.values().any(|l| l.len() != 2) {
            continue;
        }
        let mut cyc = vec![first_v];
        let mut prev = usize::MAX;
        let mut cur = first_v;
        loop {
            let nxt = *sub_nb[&cur].iter().find(|&&x| x != prev)?;
            if nxt == first_v {
                break;
            }
            cyc.push(nxt);
            prev = cur;
            cur = nxt;
        }
        if cyc.len() != e_cnt {
            continue;
        }
        if e_cnt > best_len {
            best_len = e_cnt;
            longest_cycles.clear();
        }
        longest_cycles.push(cyc);
    }

    #[derive(Clone)]
    struct PolyCand {
        order: Vec<usize>,
        sups: Vec<usize>,
        cited: Vec<usize>,
        prefix: String,
    }

    let mut picks: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut best_score = (-1isize, -1isize);
    let mut steps = 0usize;

    for cyc in &longest_cycles {
        let l_cyc = cyc.len();
        let pos_map: HashMap<usize, usize> = cyc
            .iter()
            .copied()
            .enumerate()
            .map(|(k, a)| (a, k))
            .collect();
        let in_cyc_edge = |i: usize, j: usize| -> bool {
            if let (Some(&pi), Some(&pj)) = (pos_map.get(&i), pos_map.get(&j)) {
                (pi + l_cyc - pj) % l_cyc == 1 || (pj + l_cyc - pi) % l_cyc == 1
            } else {
                false
            }
        };

        #[allow(clippy::too_many_arguments)]
        fn walk_bridge(
            path: &mut Vec<usize>,
            cyc: &[usize],
            pos_map: &HashMap<usize, usize>,
            in_cyc_edge: &dyn Fn(usize, usize) -> bool,
            adj: &[Vec<usize>],
            steps: &mut usize,
            best_score: &mut (isize, isize),
            picks: &mut Vec<(Vec<usize>, Vec<usize>)>,
        ) {
            *steps += 1;
            if *steps > 50_000 {
                return;
            }
            let l_cyc = cyc.len();
            let last = *path.last().unwrap();
            for &y in &adj[last] {
                if path.contains(&y) && y != path[0] {
                    continue;
                }
                if path.len() == 1 && in_cyc_edge(last, y) {
                    continue;
                }
                if let Some(&py) = pos_map.get(&y) {
                    if y == path[0] || (path.len() == 1 && y < last) {
                        continue;
                    }
                    let mut p = path.clone();
                    p.push(y);
                    let arc = (py + l_cyc - pos_map[&path[0]]) % l_cyc - 1;
                    let sym_arc = arc.min(l_cyc - 2 - arc) as isize;
                    let sc = ((p.len() as isize) - 2, sym_arc);
                    if sc > *best_score {
                        *best_score = sc;
                        picks.clear();
                    }
                    if sc >= *best_score {
                        picks.push((cyc.to_vec(), p));
                    }
                } else {
                    path.push(y);
                    walk_bridge(
                        path,
                        cyc,
                        pos_map,
                        in_cyc_edge,
                        adj,
                        steps,
                        best_score,
                        picks,
                    );
                    path.pop();
                }
            }
        }

        for &u in cyc {
            let mut p = vec![u];
            walk_bridge(
                &mut p,
                cyc,
                &pos_map,
                &in_cyc_edge,
                &adj,
                &mut steps,
                &mut best_score,
                &mut picks,
            );
        }
    }

    let mut cands: Vec<PolyCand> = Vec::new();
    for (cyc, path) in &picks {
        let l_cyc = cyc.len();
        let u = path[0];
        let v = *path.last().unwrap();
        let inner = &path[1..path.len() - 1];
        let mut built: HashSet<usize> = cyc.iter().copied().collect();
        built.extend(inner.iter().copied());
        let mut used_edges: HashSet<(usize, usize)> = HashSet::new();
        let norm_e = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
        for k in 0..l_cyc {
            used_edges.insert(norm_e(cyc[k], cyc[(k + 1) % l_cyc]));
        }
        for k in 1..path.len() {
            used_edges.insert(norm_e(path[k - 1], path[k]));
        }

        let mut sec_bridges: Vec<((usize, usize), Vec<usize>)> = Vec::new();
        for &(i, j) in &edges {
            if !used_edges.contains(&(i, j)) && built.contains(&i) && built.contains(&j) {
                sec_bridges.push(((i, j), Vec::new()));
            }
        }
        let mut seen = HashSet::new();
        let mut ok = true;
        for s in 0..n {
            if built.contains(&s) || seen.contains(&s) {
                continue;
            }
            if adj[s].len() != 2 {
                ok = false;
                break;
            }
            seen.insert(s);
            let mut ends = Vec::new();
            let mut left_part = Vec::new();
            let mut right_part = Vec::new();
            for (d_idx, &dir) in adj[s].iter().enumerate() {
                let mut part = Vec::new();
                let mut prev = s;
                let mut cur = dir;
                while !built.contains(&cur) {
                    if adj[cur].len() != 2 || !seen.insert(cur) {
                        ok = false;
                        break;
                    }
                    part.push(cur);
                    if let Some(&nxt) = adj[cur].iter().find(|&&x| x != prev) {
                        prev = cur;
                        cur = nxt;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
                ends.push(cur);
                if d_idx == 0 {
                    part.reverse();
                    left_part = part;
                } else {
                    right_part = part;
                }
            }
            if !ok || ends.len() != 2 || ends[0] == ends[1] {
                ok = false;
                break;
            }
            let mut chain = left_part;
            chain.push(s);
            chain.extend(right_part);
            sec_bridges.push(((ends[0], ends[1]), chain));
        }
        if !ok || sec_bridges.len() != rings - 2 {
            continue;
        }

        let arcs_from = |h: usize, target: usize| -> (Vec<usize>, Vec<usize>) {
            let k = cyc.iter().position(|&x| x == h).unwrap();
            let mut fwd = Vec::new();
            let mut idx = 1usize;
            while cyc[(k + idx) % l_cyc] != target {
                fwd.push(cyc[(k + idx) % l_cyc]);
                idx += 1;
            }
            let mut back = Vec::new();
            idx = 1;
            while cyc[(k + l_cyc - idx) % l_cyc] != target {
                back.push(cyc[(k + l_cyc - idx) % l_cyc]);
                idx += 1;
            }
            (fwd, back)
        };

        let mult_word = match rings {
            3 => "tri",
            4 => "tetra",
            5 => "penta",
            6 => "hexa",
            7 => "hepta",
            8 => "octa",
            _ => continue,
        };

        for &(h1, h2, rev_bridge) in &[(u, v, false), (v, u, true)] {
            let (p_arc, q_arc) = arcs_from(h1, h2);
            let main_bridge: Vec<usize> = if rev_bridge {
                inner.iter().copied().rev().collect()
            } else {
                inner.to_vec()
            };
            for (long_arc, short_arc) in [(&p_arc, &q_arc), (&q_arc, &p_arc)] {
                if long_arc.len() < short_arc.len() {
                    continue;
                }
                let mut base_ord = Vec::new();
                base_ord.push(h1);
                base_ord.extend(long_arc.iter().copied());
                base_ord.push(h2);
                base_ord.extend(short_arc.iter().copied().rev());
                base_ord.extend(main_bridge.iter().copied());

                let loc_of = |a: usize| base_ord.iter().position(|&x| x == a).unwrap() + 1;
                let mut sec_info: Vec<(usize, usize, usize, Vec<usize>)> = sec_bridges
                    .iter()
                    .map(|&((e0, e1), ref ch)| {
                        let x = loc_of(e0);
                        let y = loc_of(e1);
                        let (lo, hi, atoms_seq) = if x < y {
                            (x, y, ch.iter().copied().rev().collect())
                        } else {
                            (y, x, ch.clone())
                        };
                        (lo, hi, ch.len(), atoms_seq)
                    })
                    .collect();

                let mut numbered = sec_info.clone();
                numbered.sort_by(|a, b| {
                    b.1.cmp(&a.1)
                        .then_with(|| b.0.cmp(&a.0))
                        .then_with(|| b.2.cmp(&a.2))
                });
                sec_info.sort_by(|a, b| {
                    b.2.cmp(&a.2)
                        .then_with(|| a.0.cmp(&b.0))
                        .then_with(|| a.1.cmp(&b.1))
                });

                let mut sups: Vec<usize> = sec_info.iter().flat_map(|b| [b.0, b.1]).collect();
                sups.sort_unstable();
                let cited: Vec<usize> = sec_info.iter().flat_map(|b| [b.0, b.1]).collect();
                let sec_desc: String = sec_info
                    .iter()
                    .map(|b| {
                        format!(
                            ".{}{},{}",
                            b.2,
                            to_superscript_digits(b.0),
                            to_superscript_digits(b.1)
                        )
                    })
                    .collect();
                let mut full_ord = base_ord;
                for nb_b in numbered {
                    full_ord.extend(nb_b.3);
                }
                cands.push(PolyCand {
                    order: full_ord.into_iter().map(|i| ring_atoms[i]).collect(),
                    sups,
                    cited,
                    prefix: format!(
                        "{mult_word}cyclo[{}.{}.{}{sec_desc}]",
                        long_arc.len(),
                        short_arc.len(),
                        main_bridge.len()
                    ),
                });
            }
        }
    }

    if cands.is_empty() {
        return None;
    }
    cands.sort_by(|a, b| a.sups.cmp(&b.sups).then_with(|| a.cited.cmp(&b.cited)));
    let best_sups = cands[0].sups.clone();
    let best_cited = cands[0].cited.clone();
    let best_prefix = cands[0].prefix.clone();
    let mut orders = Vec::new();
    for c in cands {
        if c.sups == best_sups
            && c.cited == best_cited
            && c.prefix == best_prefix
            && !orders.contains(&c.order)
        {
            orders.push(c.order);
        }
    }
    Some(VonBaeyerNumbering {
        descriptor: best_prefix,
        candidate_orders: orders,
    })
}

// ============================================================================
// Part 2A: Functional Group Perception, Substituent Radicals, Esters & Multiplicative
// ============================================================================

#[derive(Debug, Clone)]
struct NormalizedMol {
    mol: AnalytMol,
    stereo: PerceivedStereo,
}

/// Fold explicit neutral protium (`¹H`) leaves into implicit hydrogens while preserving perceived CIP `(R)/(S)` and `(E)/(Z)`.
fn normalize_for_naming(raw: &AnalytMol) -> NormalizedMol {
    let raw_stereo = raw.perceive_cip_stereo();
    let adj = raw.adjacency();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    let mut mol = AnalytMol::new();

    for (i, (a, nbrs)) in raw.atoms.iter().zip(adj.iter()).enumerate() {
        let is_protium_leaf = a.element == "H"
            && a.isotope.unwrap_or(1) == 1
            && a.formal_charge == 0
            && nbrs.len() == 1
            && raw.atoms[nbrs[0].0].element != "H";
        if is_protium_leaf {
            continue;
        }
        let mut na = a.clone();
        na.implicit_h = Some(raw.total_h_count(i));
        na.smiles_neighbor_order.clear();
        if let Some(&rs) = raw_stereo.atom_rs.get(&i) {
            na.stereo_tag = if rs == 'R' {
                AtomStereoTag::R
            } else {
                AtomStereoTag::S
            };
        }
        let new_idx = mol.add_atom(na);
        old_to_new.insert(i, new_idx);
    }

    for (bi, b) in raw.bonds.iter().enumerate() {
        if let (Some(&na), Some(&nb)) = (old_to_new.get(&b.a), old_to_new.get(&b.b)) {
            let mut nbond = AnalytBond {
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                stereo: b.stereo,
            };
            if let Some(&ez) = raw_stereo.bond_ez.get(&bi) {
                nbond.stereo = if ez == 'Z' {
                    BondStereoTag::Cis
                } else {
                    BondStereoTag::Trans
                };
            }
            mol.add_bond(nbond);
        }
    }

    mol.perceive_aromaticity();
    let stereo = mol.perceive_cip_stereo();
    NormalizedMol { mol, stereo }
}

fn is_nitro_group(
    mol: &AnalytMol,
    n_idx: usize,
    parent_atom: usize,
    adj: &[Vec<(usize, usize)>],
) -> bool {
    let a = &mol.atoms[n_idx];
    if a.element != "N" || adj[n_idx].len() != 3 {
        return false;
    }
    let others: Vec<(usize, usize)> = adj[n_idx]
        .iter()
        .copied()
        .filter(|&(x, _)| x != parent_atom)
        .collect();
    if others.len() != 2 {
        return false;
    }
    others.iter().all(|&(ox, _)| {
        mol.atoms[ox].element == "O" && adj[ox].len() == 1 && mol.total_h_count(ox) == 0
    })
}

fn is_azido_group(
    mol: &AnalytMol,
    n_idx: usize,
    parent_atom: usize,
    adj: &[Vec<(usize, usize)>],
) -> bool {
    if mol.atoms[n_idx].element != "N" || adj[n_idx].len() != 2 {
        return false;
    }
    let Some(&(mid, _)) = adj[n_idx].iter().find(|&&(x, _)| x != parent_atom) else {
        return false;
    };
    if mol.atoms[mid].element != "N" || adj[mid].len() != 2 {
        return false;
    }
    let Some(&(term, _)) = adj[mid].iter().find(|&&(x, _)| x != n_idx) else {
        return false;
    };
    mol.atoms[term].element == "N" && adj[term].len() == 1 && mol.total_h_count(term) == 0
}

/// Perceive candidate principal functional groups across `mol`.
fn perceive_functional_hits(mol: &AnalytMol) -> Vec<FunctionalHit> {
    let adj = mol.adjacency();
    let rings = mol.sssr_rings();
    let arom = mol.aromatic_atoms();
    let mut consumed: HashSet<usize> = HashSet::new();
    let mut hits: Vec<FunctionalHit> = Vec::new();

    let in_same_ring =
        |u: usize, v: usize| -> bool { rings.iter().any(|r| r.contains(&u) && r.contains(&v)) };

    // Pass 1: Carboxylic acids, Sulfonic acids, Esters (acyclic), Acid halides, Amides (acyclic), Nitriles, Aldehydes, Ketones
    for i in 0..mol.atoms.len() {
        let el = mol.atoms[i].element.as_str();
        if el == "S" && mol.atoms[i].formal_charge == 0 {
            let mut dbl_o = Vec::new();
            let mut oh = Vec::new();
            let mut c_nb = Vec::new();
            for &(nb, bi) in &adj[i] {
                let b_ord = mol.bonds[bi].order;
                if mol.atoms[nb].element == "O" && adj[nb].len() == 1 {
                    if b_ord == 2 && mol.total_h_count(nb) == 0 && mol.atoms[nb].formal_charge == 0
                    {
                        dbl_o.push(nb);
                    } else if b_ord == 1
                        && mol.total_h_count(nb) == 1
                        && mol.atoms[nb].formal_charge == 0
                    {
                        oh.push(nb);
                    }
                } else if mol.atoms[nb].element == "C" && b_ord == 1 {
                    c_nb.push(nb);
                }
            }
            if dbl_o.len() == 2 && oh.len() == 1 && c_nb.len() == 1 {
                let mut ga = vec![i, dbl_o[0], dbl_o[1], oh[0]];
                for &x in &ga {
                    consumed.insert(x);
                }
                ga.shrink_to_fit();
                hits.push(FunctionalHit {
                    kind: PrincipalGroup::SulfonicAcid,
                    group_atom: i,
                    anchor_carbon: c_nb[0],
                    group_atoms: ga,
                    extra_atom: None,
                });
            }
            continue;
        }

        if el != "C" || arom[i] || mol.atoms[i].formal_charge != 0 {
            continue;
        }

        // Check nitrile `-C#N`
        if let Some(&(n_nb, _)) = adj[i].iter().find(|&&(nb, bi)| {
            mol.bonds[bi].order == 3
                && mol.atoms[nb].element == "N"
                && adj[nb].len() == 1
                && mol.atoms[nb].formal_charge == 0
        }) {
            consumed.insert(n_nb);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::Nitrile,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![n_nb],
                extra_atom: None,
            });
            continue;
        }

        // Check carbonyl `C=O`
        let dbl_os: Vec<usize> = adj[i]
            .iter()
            .filter_map(|&(nb, bi)| {
                if mol.bonds[bi].order == 2
                    && mol.atoms[nb].element == "O"
                    && adj[nb].len() == 1
                    && mol.atoms[nb].formal_charge == 0
                    && mol.total_h_count(nb) == 0
                {
                    Some(nb)
                } else {
                    None
                }
            })
            .collect();
        if dbl_os.len() != 1 {
            continue;
        }
        let o_dbl = dbl_os[0];
        let other_nbs: Vec<(usize, u8)> = adj[i]
            .iter()
            .filter_map(|&(nb, bi)| {
                if nb == o_dbl {
                    None
                } else {
                    Some((nb, mol.bonds[bi].order))
                }
            })
            .collect();

        // Carboxylic acid: one `-OH` leaf and at most one carbon/H neighbor (not carbonate/carbamate)
        let oh_nbs: Vec<usize> = other_nbs
            .iter()
            .filter_map(|&(nb, ord)| {
                if ord == 1
                    && mol.atoms[nb].element == "O"
                    && adj[nb].len() == 1
                    && mol.total_h_count(nb) == 1
                    && mol.atoms[nb].formal_charge == 0
                {
                    Some(nb)
                } else {
                    None
                }
            })
            .collect();
        let non_c_other = other_nbs
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].element != "C")
            .count();

        if oh_nbs.len() == 1 && non_c_other == 1 {
            consumed.insert(o_dbl);
            consumed.insert(oh_nbs[0]);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::CarboxylicAcid,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl, oh_nbs[0]],
                extra_atom: None,
            });
            continue;
        }

        // Acyclic Ester: `R-C(=O)-O-R'` where `O` has degree 2 (`i` and carbon `R'`), not in same ring
        let ester_os: Vec<usize> = other_nbs
            .iter()
            .filter_map(|&(nb, ord)| {
                if ord == 1
                    && mol.atoms[nb].element == "O"
                    && mol.atoms[nb].formal_charge == 0
                    && adj[nb].len() == 2
                    && !in_same_ring(i, nb)
                {
                    let r_prime = adj[nb].iter().find(|&&(x, _)| x != i)?.0;
                    if mol.atoms[r_prime].element == "C" {
                        return Some(nb);
                    }
                }
                None
            })
            .collect();
        if ester_os.len() == 1 && non_c_other == 1 {
            consumed.insert(o_dbl);
            consumed.insert(ester_os[0]);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::Ester,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl, ester_os[0]],
                extra_atom: Some(ester_os[0]),
            });
            continue;
        }

        // Acid halide: `R-C(=O)-X`
        let hal_nbs: Vec<usize> = other_nbs
            .iter()
            .filter_map(|&(nb, ord)| {
                if ord == 1
                    && matches!(mol.atoms[nb].element.as_str(), "F" | "Cl" | "Br" | "I")
                    && adj[nb].len() == 1
                {
                    Some(nb)
                } else {
                    None
                }
            })
            .collect();
        if hal_nbs.len() == 1 && non_c_other == 1 {
            consumed.insert(o_dbl);
            consumed.insert(hal_nbs[0]);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::AcidHalide,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl, hal_nbs[0]],
                extra_atom: Some(hal_nbs[0]),
            });
            continue;
        }

        // Acyclic Amide: `R-C(=O)-NR'R''` where `N` is neutral, single-bonded, not in ring with `i`
        let amide_ns: Vec<usize> = other_nbs
            .iter()
            .filter_map(|&(nb, ord)| {
                if ord == 1
                    && mol.atoms[nb].element == "N"
                    && mol.atoms[nb].formal_charge == 0
                    && !arom[nb]
                    && !in_same_ring(i, nb)
                    && adj[nb].iter().all(|&(x, bi)| {
                        mol.bonds[bi].order == 1 && (x == i || mol.atoms[x].element == "C")
                    })
                {
                    Some(nb)
                } else {
                    None
                }
            })
            .collect();
        if amide_ns.len() == 1 && non_c_other == 1 {
            consumed.insert(o_dbl);
            consumed.insert(amide_ns[0]);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::Amide,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl, amide_ns[0]],
                extra_atom: Some(amide_ns[0]),
            });
            continue;
        }

        // Aldehyde (`-CH=O` or `H2C=O`) vs Ketone (`>C=O`)
        if mol.total_h_count(i) >= 1 && non_c_other == 0 {
            consumed.insert(o_dbl);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::Aldehyde,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl],
                extra_atom: None,
            });
        } else if mol.total_h_count(i) == 0 {
            consumed.insert(o_dbl);
            hits.push(FunctionalHit {
                kind: PrincipalGroup::Ketone,
                group_atom: i,
                anchor_carbon: i,
                group_atoms: vec![o_dbl],
                extra_atom: None,
            });
        }
    }

    // Pass 2: Alcohols, Thiols, Amines
    for i in 0..mol.atoms.len() {
        if consumed.contains(&i) || arom[i] || mol.atoms[i].formal_charge != 0 {
            continue;
        }
        let el = mol.atoms[i].element.as_str();
        match el {
            "O" if mol.total_h_count(i) == 1 && adj[i].len() == 1 => {
                let (c_nb, bi) = adj[i][0];
                if mol.bonds[bi].order == 1 && mol.atoms[c_nb].element == "C" {
                    consumed.insert(i);
                    hits.push(FunctionalHit {
                        kind: PrincipalGroup::Alcohol,
                        group_atom: i,
                        anchor_carbon: c_nb,
                        group_atoms: vec![i],
                        extra_atom: None,
                    });
                }
            }
            "S" if mol.total_h_count(i) == 1 && adj[i].len() == 1 => {
                let (c_nb, bi) = adj[i][0];
                if mol.bonds[bi].order == 1 && mol.atoms[c_nb].element == "C" {
                    consumed.insert(i);
                    hits.push(FunctionalHit {
                        kind: PrincipalGroup::Thiol,
                        group_atom: i,
                        anchor_carbon: c_nb,
                        group_atoms: vec![i],
                        extra_atom: None,
                    });
                }
            }
            "N" if !rings.iter().any(|r| r.contains(&i))
                // Neutral non-ring amine where all neighbors are single-bonded carbons (not attached to C=O / S=O / nitro)
                && !adj[i].is_empty()
                    && adj[i].iter().all(|&(nb, bi)| {
                        mol.bonds[bi].order == 1
                            && mol.atoms[nb].element == "C"
                            && !adj[nb].iter().any(|&(x, xbi)| {
                                x != i
                                    && mol.bonds[xbi].order >= 2
                                    && matches!(mol.atoms[x].element.as_str(), "O" | "S" | "N")
                            })
                    }) =>
            {
                // Pick the largest / most functionalized carbon neighbor as `anchor_carbon`
                let mut best_c = adj[i][0].0;
                let mut best_reach = 0usize;
                for &(c_nb, _) in &adj[i] {
                    let reach = component_size_excluding(mol, c_nb, i);
                    if reach > best_reach {
                        best_reach = reach;
                        best_c = c_nb;
                    }
                }
                consumed.insert(i);
                hits.push(FunctionalHit {
                    kind: PrincipalGroup::Amine,
                    group_atom: i,
                    anchor_carbon: best_c,
                    group_atoms: vec![i],
                    extra_atom: None,
                });
            }
            _ => {}
        }
    }

    hits
}

fn component_size_excluding(mol: &AnalytMol, start: usize, blocked_atom: usize) -> usize {
    let adj = mol.adjacency();
    let mut vis = HashSet::new();
    vis.insert(blocked_atom);
    vis.insert(start);
    let mut q = VecDeque::new();
    q.push_back(start);
    let mut cnt = 0usize;
    while let Some(u) = q.pop_front() {
        cnt += 1;
        for &(v, _) in &adj[u] {
            if vis.insert(v) {
                q.push_back(v);
            }
        }
    }
    cnt
}

fn collect_subfragment(mol: &AnalytMol, start: usize, blocked: &HashSet<usize>) -> HashSet<usize> {
    let adj = mol.adjacency();
    let mut sub = HashSet::new();
    if blocked.contains(&start) {
        return sub;
    }
    let mut q = VecDeque::new();
    sub.insert(start);
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        for &(v, _) in &adj[u] {
            if !blocked.contains(&v) && sub.insert(v) {
                q.push_back(v);
            }
        }
    }
    sub
}

/// Name a monovalent or divalent-attachment substituent starting at `start` attached to `parent_atom`
/// (`(substituent_name, is_compound)`).
fn name_substituent(
    mol: &AnalytMol,
    start: usize,
    parent_atom: usize,
    attach_order: u8,
    blocked: &HashSet<usize>,
) -> (String, bool) {
    let adj = mol.adjacency();
    let el = mol.atoms[start].element.as_str();

    if attach_order == 2 {
        return match el {
            "O" => ("oxo".to_string(), false),
            "S" => ("sulfanylidene".to_string(), false),
            "N" if mol.total_h_count(start) == 1 => ("imino".to_string(), false),
            "C" if mol.total_h_count(start) == 2 => ("methylidene".to_string(), false),
            _ => (format!("{}ylidene", el.to_ascii_lowercase()), true),
        };
    }

    // Halides
    match el {
        "F" => return ("fluoro".to_string(), false),
        "Cl" => return ("chloro".to_string(), false),
        "Br" => return ("bromo".to_string(), false),
        "I" => return ("iodo".to_string(), false),
        _ => {}
    }

    // Nitro / Nitroso / Azido
    if is_nitro_group(mol, start, parent_atom, &adj) {
        return ("nitro".to_string(), false);
    }
    if is_azido_group(mol, start, parent_atom, &adj) {
        return ("azido".to_string(), false);
    }
    if el == "N" && adj[start].len() == 2 && mol.total_h_count(start) == 0 {
        if let Some(&(ox, bi)) = adj[start].iter().find(|&&(x, _)| x != parent_atom) {
            if mol.atoms[ox].element == "O" && mol.bonds[bi].order == 2 && adj[ox].len() == 1 {
                return ("nitroso".to_string(), false);
            }
        }
    }

    // Oxygen substituents (`-OH`, `-OR`, `-O-C(=O)R`)
    if el == "O" {
        if mol.total_h_count(start) == 1 && adj[start].len() == 1 {
            return ("hydroxy".to_string(), false);
        }
        if let Some(&(other, bi)) = adj[start].iter().find(|&&(x, _)| x != parent_atom) {
            if mol.bonds[bi].order == 1 {
                let mut blk = blocked.clone();
                blk.insert(start);
                // Check acyloxy `-O-C(=O)CH3` -> `acetyloxy`
                if mol.atoms[other].element == "C" {
                    let has_oxo = adj[other].iter().any(|&(z, zbi)| {
                        mol.bonds[zbi].order == 2
                            && mol.atoms[z].element == "O"
                            && adj[z].len() == 1
                    });
                    if has_oxo {
                        let (acyl, _) = name_substituent(mol, other, start, 1, &blk);
                        return (format!("{acyl}oxy"), true);
                    }
                }
                let (rad, is_comp) = name_substituent(mol, other, start, 1, &blk);
                let short = match rad.as_str() {
                    "methyl" => Some("methoxy"),
                    "ethyl" => Some("ethoxy"),
                    "propyl" => Some("propoxy"),
                    "butyl" => Some("butoxy"),
                    "phenyl" => Some("phenoxy"),
                    _ => None,
                };
                if let Some(s) = short {
                    return (s.to_string(), false);
                }
                if is_comp {
                    return (format!("({rad})oxy"), true);
                }
                return (format!("{rad}oxy"), true);
            }
        }
    }

    // Sulfur substituents (`-SH`, `-SR`, `-SO2R`, `-SO3H`)
    if el == "S" {
        if mol.total_h_count(start) == 1 && adj[start].len() == 1 {
            return ("sulfanyl".to_string(), false);
        }
        let dbl_os: Vec<usize> = adj[start]
            .iter()
            .filter_map(|&(x, bi)| {
                if x != parent_atom && mol.bonds[bi].order == 2 && mol.atoms[x].element == "O" {
                    Some(x)
                } else {
                    None
                }
            })
            .collect();
        if dbl_os.len() == 2 {
            if let Some(&(other, _)) = adj[start]
                .iter()
                .find(|&&(x, _)| x != parent_atom && !dbl_os.contains(&x))
            {
                if mol.atoms[other].element == "O" && mol.total_h_count(other) == 1 {
                    return ("sulfo".to_string(), false);
                }
                let mut blk = blocked.clone();
                blk.insert(start);
                let (rad, is_comp) = name_substituent(mol, other, start, 1, &blk);
                if is_comp {
                    return (format!("({rad})sulfonyl"), true);
                }
                return (format!("{rad}sulfonyl"), true);
            }
        }
        if dbl_os.is_empty() {
            if let Some(&(other, _)) = adj[start].iter().find(|&&(x, _)| x != parent_atom) {
                let mut blk = blocked.clone();
                blk.insert(start);
                let (rad, is_comp) = name_substituent(mol, other, start, 1, &blk);
                if is_comp {
                    return (format!("({rad})sulfanyl"), true);
                }
                return (format!("{rad}sulfanyl"), true);
            }
        }
    }

    // Nitrogen substituents (`-NH2`, `-NHR`, `-NR2`)
    if el == "N" {
        let others: Vec<(usize, u8)> = adj[start]
            .iter()
            .filter_map(|&(x, bi)| {
                if x == parent_atom {
                    None
                } else {
                    Some((x, mol.bonds[bi].order))
                }
            })
            .collect();
        if others.is_empty() && mol.total_h_count(start) == 2 {
            return ("amino".to_string(), false);
        }
        let mut blk = blocked.clone();
        blk.insert(start);
        if others.len() == 1 && others[0].1 == 1 {
            let (rad, _) = name_substituent(mol, others[0].0, start, 1, &blk);
            return (format!("{rad}amino"), true);
        }
        if others.len() == 2 && others.iter().all(|&(_, ord)| ord == 1) {
            let (r1, c1) = name_substituent(mol, others[0].0, start, 1, &blk);
            let (r2, c2) = name_substituent(mol, others[1].0, start, 1, &blk);
            if r1 == r2 && !c1 {
                return (format!("di{r1}amino"), true);
            }
            let mut pair = [r1, r2];
            pair.sort_by_key(|a| iupac_alpha_key(a));
            let _ = c2;
            return (format!("{}({}amino)", pair[0], pair[1]), true);
        }
    }

    // Carbon-centered functional substituents (`-CN`, `-CHO`, `-COOH`, `-COOR`, `-CONH2`, `-COR`)
    if el == "C" {
        let sub_nbs: Vec<(usize, u8)> = adj[start]
            .iter()
            .filter_map(|&(x, bi)| {
                if x == parent_atom || blocked.contains(&x) {
                    None
                } else {
                    Some((x, mol.bonds[bi].order))
                }
            })
            .collect();
        if sub_nbs.len() == 1
            && sub_nbs[0].1 == 3
            && mol.atoms[sub_nbs[0].0].element == "N"
            && adj[sub_nbs[0].0].len() == 1
        {
            return ("cyano".to_string(), false);
        }
        let oxo_nb = sub_nbs
            .iter()
            .find(|&&(x, ord)| ord == 2 && mol.atoms[x].element == "O" && adj[x].len() == 1);
        if let Some(&(o_idx, _)) = oxo_nb {
            let rest: Vec<(usize, u8)> = sub_nbs
                .iter()
                .copied()
                .filter(|&(x, _)| x != o_idx)
                .collect();
            if rest.is_empty() && mol.total_h_count(start) == 1 {
                return ("formyl".to_string(), false);
            }
            if rest.len() == 1 && rest[0].1 == 1 {
                let r_nb = rest[0].0;
                if mol.atoms[r_nb].element == "O"
                    && mol.total_h_count(r_nb) == 1
                    && adj[r_nb].len() == 1
                {
                    return ("carboxy".to_string(), false);
                }
                if mol.atoms[r_nb].element == "O" && adj[r_nb].len() == 2 {
                    if let Some(&(alk, _)) = adj[r_nb].iter().find(|&&(z, _)| z != start) {
                        let mut blk = blocked.clone();
                        blk.insert(start);
                        blk.insert(r_nb);
                        let (rad, _) = name_substituent(mol, alk, r_nb, 1, &blk);
                        return (format!("{rad}oxycarbonyl"), true);
                    }
                }
                if mol.atoms[r_nb].element == "N"
                    && mol.total_h_count(r_nb) == 2
                    && adj[r_nb].len() == 1
                {
                    return ("carbamoyl".to_string(), false);
                }
                if mol.atoms[r_nb].element == "C" {
                    let mut blk = blocked.clone();
                    blk.insert(start);
                    let sub_set = collect_subfragment(mol, r_nb, &blk);
                    if sub_set.len() == 1 && mol.total_h_count(r_nb) == 3 {
                        return ("acetyl".to_string(), false);
                    }
                    let (rad, _) = name_substituent(mol, r_nb, start, 1, &blk);
                    if rad == "phenyl" {
                        return ("benzoyl".to_string(), false);
                    }
                    return (format!("{rad}carbonyl"), true);
                }
            }
        }

        // Cyclic carbon substituents (`phenyl`, substituted `phenyl`, `cycloalkyl`)
        let sub_atoms = collect_subfragment(mol, start, blocked);
        let arom = mol.aromatic_atoms();
        let rings = mol.sssr_rings();
        let sub_rings: Vec<Vec<usize>> = rings
            .into_iter()
            .filter(|r| r.iter().all(|a| sub_atoms.contains(a)))
            .collect();

        if let Some(r) = sub_rings.iter().find(|r| r.contains(&start)) {
            if r.len() == 6 && r.iter().all(|&a| mol.atoms[a].element == "C" && arom[a]) {
                return name_phenyl_substituent(mol, start, r, blocked);
            }
            if sub_rings.len() == 1 && r.iter().all(|&a| mol.atoms[a].element == "C" && !arom[a]) {
                let stem = alkane_stem(r.len());
                if sub_atoms.len() == r.len()
                    && r.iter().all(|&u| {
                        adj[u]
                            .iter()
                            .filter(|&&(v, _)| r.contains(&v))
                            .all(|&(_, bi)| mol.bonds[bi].order == 1)
                    })
                {
                    return (format!("cyclo{stem}yl"), false);
                }
            }
        }

        // Acyclic carbon radical starting at `start`
        return name_acyclic_carbon_radical(mol, start, &sub_atoms, blocked);
    }

    if el == "Si" && adj[start].len() == 4 {
        let mut blk = blocked.clone();
        blk.insert(start);
        let mut subs = Vec::new();
        for &(nb, _) in &adj[start] {
            if nb != parent_atom {
                subs.push(name_substituent(mol, nb, start, 1, &blk).0);
            }
        }
        if subs.len() == 3 && subs.iter().all(|s| s == &subs[0]) {
            return (format!("tri{}silyl", subs[0]), true);
        }
    }

    (format!("{}yl", el.to_ascii_lowercase()), false)
}

fn name_phenyl_substituent(
    mol: &AnalytMol,
    attach: usize,
    ring: &[usize],
    outer_blocked: &HashSet<usize>,
) -> (String, bool) {
    let adj = mol.adjacency();
    let k = ring.len();
    let pos0 = ring.iter().position(|&x| x == attach).unwrap();
    let cw: Vec<usize> = (0..k).map(|d| ring[(pos0 + d) % k]).collect();
    let mut ccw: Vec<usize> = Vec::with_capacity(k);
    ccw.push(attach);
    for d in 1..k {
        ccw.push(ring[(pos0 + k - d) % k]);
    }

    let mut ring_blk = outer_blocked.clone();
    for &ra in ring {
        ring_blk.insert(ra);
    }

    // Check if unsubstituted phenyl
    let has_exo = ring
        .iter()
        .any(|&ra| adj[ra].iter().any(|&(nb, _)| !ring_blk.contains(&nb)));
    if !has_exo {
        return ("phenyl".to_string(), false);
    }

    let eval_dir = |ord: &[usize]| -> (Vec<usize>, Vec<String>, String) {
        let mut locs = Vec::new();
        let mut items = Vec::new();
        for (idx, &ra) in ord.iter().enumerate() {
            let loc = idx + 1;
            for &(nb, bi) in &adj[ra] {
                if !ring_blk.contains(&nb) {
                    let (sname, is_comp) =
                        name_substituent(mol, nb, ra, mol.bonds[bi].order, &ring_blk);
                    locs.push(loc);
                    items.push(SubPrefixItem {
                        locant: loc.to_string(),
                        base_name: sname,
                        is_compound: is_comp,
                    });
                }
            }
        }
        locs.sort_unstable();
        let mut alpha_locs: Vec<(String, usize)> = items
            .iter()
            .map(|it| {
                (
                    iupac_alpha_key(&it.base_name),
                    it.locant.parse::<usize>().unwrap_or(99),
                )
            })
            .collect();
        alpha_locs.sort();
        let alpha_tie: Vec<String> = alpha_locs
            .into_iter()
            .map(|(k, l)| format!("{k}:{l}"))
            .collect();
        let pref = assemble_prefixes(&items, false);
        (locs, alpha_tie, format!("{pref}phenyl"))
    };

    let c1 = eval_dir(&cw);
    let c2 = eval_dir(&ccw);
    let best = if (c1.0.as_slice(), c1.1.as_slice()) <= (c2.0.as_slice(), c2.1.as_slice()) {
        c1.2
    } else {
        c2.2
    };
    (best, true)
}

fn name_acyclic_carbon_radical(
    mol: &AnalytMol,
    start: usize,
    sub_atoms: &HashSet<usize>,
    outer_blocked: &HashSet<usize>,
) -> (String, bool) {
    let adj = mol.adjacency();
    let sub_ring_atoms: HashSet<usize> = mol
        .sssr_rings()
        .into_iter()
        .flatten()
        .filter(|a| sub_atoms.contains(a))
        .collect();
    let all_sp3_carbon = sub_ring_atoms.is_empty()
        && sub_atoms.iter().all(|&a| {
            mol.atoms[a].element == "C"
                && adj[a]
                    .iter()
                    .filter(|&&(nb, _)| sub_atoms.contains(&nb))
                    .all(|&(_, bi)| mol.bonds[bi].order == 1)
        });

    // Special IUPAC 2013 P-29.3.2.2 check for unbranched internal attachment (`propan-2-yl`, `butan-2-yl`)
    // and retained `tert-butyl` (`-C(CH3)3`).
    if all_sp3_carbon {
        let start_c_nbs: Vec<usize> = adj[start]
            .iter()
            .map(|&(nb, _)| nb)
            .filter(|nb| sub_atoms.contains(nb))
            .collect();
        if sub_atoms.len() == 4
            && start_c_nbs.len() == 3
            && start_c_nbs.iter().all(|&x| mol.total_h_count(x) == 3)
        {
            return ("tert-butyl".to_string(), false);
        }
        // Check if `sub_atoms` as an induced graph is a single unbranched path with `start` at an internal node!
        let degs_ok = sub_atoms.iter().all(|&a| {
            adj[a]
                .iter()
                .filter(|&&(nb, _)| sub_atoms.contains(&nb))
                .count()
                <= 2
        });
        if degs_ok && start_c_nbs.len() == 2 {
            // Trace whole path from one end to the other
            let ends: Vec<usize> = sub_atoms
                .iter()
                .copied()
                .filter(|&a| {
                    adj[a]
                        .iter()
                        .filter(|&&(nb, _)| sub_atoms.contains(&nb))
                        .count()
                        == 1
                })
                .collect();
            if ends.len() == 2 {
                let mut path = vec![ends[0]];
                let mut prev = usize::MAX;
                let mut cur = ends[0];
                while cur != ends[1] {
                    let nxt = adj[cur]
                        .iter()
                        .map(|&(nb, _)| nb)
                        .find(|&nb| sub_atoms.contains(&nb) && nb != prev)
                        .unwrap();
                    path.push(nxt);
                    prev = cur;
                    cur = nxt;
                }
                let pos1 = path.iter().position(|&x| x == start).unwrap() + 1;
                let pos2 = path.len() + 1 - pos1;
                let loc = pos1.min(pos2);
                let stem = alkane_stem(path.len());
                return (format!("{stem}an-{loc}-yl"), true);
            }
        }
    }

    // Find longest acyclic carbon chain starting at `start` (`C1 = start`), stopping before any ring atoms
    fn longest_c_chain_from(
        u: usize,
        visited: &mut HashSet<usize>,
        mol: &AnalytMol,
        adj: &[Vec<(usize, usize)>],
        sub_atoms: &HashSet<usize>,
        sub_ring_atoms: &HashSet<usize>,
    ) -> Vec<usize> {
        let mut best: Vec<usize> = Vec::new();
        for &(v, _) in &adj[u] {
            if sub_atoms.contains(&v)
                && !sub_ring_atoms.contains(&v)
                && mol.atoms[v].element == "C"
                && visited.insert(v)
            {
                let child = longest_c_chain_from(v, visited, mol, adj, sub_atoms, sub_ring_atoms);
                visited.remove(&v);
                if child.len() > best.len() {
                    best = child;
                }
            }
        }
        let mut out = vec![u];
        out.extend(best);
        out
    }

    let mut vis_chain = HashSet::new();
    vis_chain.insert(start);
    let chain = longest_c_chain_from(start, &mut vis_chain, mol, &adj, sub_atoms, &sub_ring_atoms);
    let len = chain.len();
    let stem = alkane_stem(len);

    // Unsaturations along `chain`
    let mut dbl_locs = Vec::new();
    let mut trp_locs = Vec::new();
    for k in 0..len.saturating_sub(1) {
        if let Some(bi) = mol.bond_between(chain[k], chain[k + 1]) {
            match mol.bonds[bi].order {
                2 => dbl_locs.push(k + 1),
                3 => trp_locs.push(k + 1),
                _ => {}
            }
        }
    }

    let mut chain_blk = outer_blocked.clone();
    for &ca in &chain {
        chain_blk.insert(ca);
    }

    let mut items = Vec::new();
    for (idx, &ca) in chain.iter().enumerate() {
        let loc = idx + 1;
        for &(nb, bi) in &adj[ca] {
            if !chain_blk.contains(&nb) {
                let (sname, is_comp) =
                    name_substituent(mol, nb, ca, mol.bonds[bi].order, &chain_blk);
                let loc_s = if len == 1 {
                    String::new()
                } else {
                    loc.to_string()
                };
                items.push(SubPrefixItem {
                    locant: loc_s,
                    base_name: sname,
                    is_compound: is_comp,
                });
            }
        }
    }

    // Check benzyl (`phenylmethyl`)
    let pref = assemble_prefixes(&items, false);
    if dbl_locs.is_empty() && trp_locs.is_empty() {
        if items.is_empty() {
            return (format!("{stem}yl"), false);
        }
        return (format!("{pref}{stem}yl"), true);
    }

    // Unsaturated radical (`ethenyl`, `ethynyl`, `prop-2-en-1-yl`)
    if items.is_empty() && len == 2 && dbl_locs == [1] && trp_locs.is_empty() {
        return ("ethenyl".to_string(), false);
    }
    if items.is_empty() && len == 2 && trp_locs == [1] && dbl_locs.is_empty() {
        return ("ethynyl".to_string(), false);
    }
    let unsat = format_unsat_infix(&stem, &dbl_locs, &trp_locs, false);
    if pref.is_empty() {
        (format!("{unsat}-1-yl"), true)
    } else {
        (format!("{pref}-{unsat}-1-yl"), true)
    }
}

fn format_unsat_infix(
    stem: &str,
    dbl_locs: &[usize],
    trp_locs: &[usize],
    terminal_e: bool,
) -> String {
    if dbl_locs.is_empty() && trp_locs.is_empty() {
        return if terminal_e {
            format!("{stem}ane")
        } else {
            format!("{stem}an")
        };
    }
    let mut out = stem.to_string();
    if !dbl_locs.is_empty() {
        if dbl_locs.len() >= 2 && !out.ends_with('a') {
            out.push('a');
        }
        let mp = simple_mult_prefix(dbl_locs.len());
        let locs = dbl_locs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if stem == "eth" && dbl_locs.len() == 1 && trp_locs.is_empty() {
            out.push_str("en");
        } else {
            out.push_str(&format!("-{locs}-{mp}en"));
        }
        if trp_locs.is_empty() && terminal_e {
            out.push('e');
        }
    }
    if !trp_locs.is_empty() {
        if dbl_locs.is_empty() && trp_locs.len() >= 2 && !out.ends_with('a') {
            out.push('a');
        }
        let mp = simple_mult_prefix(trp_locs.len());
        let locs = trp_locs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        if stem == "eth" && trp_locs.len() == 1 && dbl_locs.is_empty() {
            out.push_str("yn");
        } else {
            out.push_str(&format!("-{locs}-{mp}yn"));
        }
        if terminal_e {
            out.push('e');
        }
    }
    out
}

// ============================================================================
// Part 2B: Multiplicative, Ring Parents, Acyclic P-44 Chains & `mol_to_iupac_name`
// ============================================================================

fn format_stereo_prefix(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    atom_to_loc: &HashMap<usize, usize>,
    parent_size: usize,
) -> String {
    let mut descriptors: Vec<(usize, String)> = Vec::new();
    for (&atom_idx, &rs) in &stereo.atom_rs {
        if let Some(&loc) = atom_to_loc.get(&atom_idx) {
            if parent_size <= 1 {
                descriptors.push((0, rs.to_string()));
            } else {
                descriptors.push((loc, format!("{loc}{rs}")));
            }
        }
    }
    for (&bi, &ez) in &stereo.bond_ez {
        let b = &mol.bonds[bi];
        if let (Some(&lu), Some(&lv)) = (atom_to_loc.get(&b.a), atom_to_loc.get(&b.b)) {
            let lo = lu.min(lv);
            let hi = lu.max(lv);
            if hi == lo + 1 {
                descriptors.push((lo, format!("{lo}{ez}")));
            }
        }
    }
    if descriptors.is_empty() {
        return String::new();
    }
    descriptors.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let joined = descriptors
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>()
        .join(",");
    format!("({joined})-")
}

/// Induce a submolecule from `keep_atoms`, returning `(sub_mol, sub_to_orig, orig_to_sub)`.
fn induce_submolecule(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    keep: &HashSet<usize>,
) -> (AnalytMol, Vec<usize>, HashMap<usize, usize>) {
    let mut sorted_keep: Vec<usize> = keep.iter().copied().collect();
    sorted_keep.sort_unstable();
    let mut orig_to_sub = HashMap::with_capacity(sorted_keep.len());
    let mut sub = AnalytMol::new();

    // First copy atoms with temporary implicit_h = None, then restore exact H count accounting for severed bonds
    for (new_idx, &old_idx) in sorted_keep.iter().enumerate() {
        orig_to_sub.insert(old_idx, new_idx);
        let mut a = mol.atoms[old_idx].clone();
        if let Some(&rs) = stereo.atom_rs.get(&old_idx) {
            a.stereo_tag = if rs == 'R' {
                AtomStereoTag::R
            } else {
                AtomStereoTag::S
            };
        }
        sub.add_atom(a);
    }
    for (bi, b) in mol.bonds.iter().enumerate() {
        if let (Some(&na), Some(&nb)) = (orig_to_sub.get(&b.a), orig_to_sub.get(&b.b)) {
            let mut nbond = AnalytBond {
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                stereo: b.stereo,
            };
            if let Some(&ez) = stereo.bond_ez.get(&bi) {
                nbond.stereo = if ez == 'Z' {
                    BondStereoTag::Cis
                } else {
                    BondStereoTag::Trans
                };
            }
            sub.add_bond(nbond);
        }
    }
    // Add 1 implicit H for each severed single bond so severed attachment points behave as intact parent hydrides
    for (new_idx, &old_idx) in sorted_keep.iter().enumerate() {
        let old_bo = mol.bond_order_sum(old_idx);
        let new_bo = sub.bond_order_sum(new_idx);
        if old_bo > new_bo {
            let cur_h = mol.total_h_count(old_idx);
            sub.atoms[new_idx].implicit_h = Some(cur_h.saturating_add(old_bo - new_bo));
        }
    }
    sub.perceive_aromaticity();
    (sub, sorted_keep, orig_to_sub)
}

/// Detect symmetric divalent-linked multiplicative assemblies (`namer-multiplicative.ts`),
/// e.g. `4,4'-oxydibenzoic acid`, `1,1'-sulfonylbis(4-chlorobenzene)`.
fn try_multiplicative_name(mol: &AnalytMol, stereo: &PerceivedStereo) -> Option<String> {
    let adj = mol.adjacency();
    let rings = mol.sssr_rings();
    if rings.len() < 2 {
        return None;
    }

    for center in 0..mol.atoms.len() {
        if rings.iter().any(|r| r.contains(&center)) {
            continue;
        }
        let el = mol.atoms[center].element.as_str();
        let mut linker_atoms: HashSet<usize> = HashSet::new();
        linker_atoms.insert(center);
        let mut core_links: Vec<usize> = Vec::new();

        let linker_name = match el {
            "O" if adj[center].len() == 2 && mol.total_h_count(center) == 0 => {
                if adj[center].iter().all(|&(_, bi)| mol.bonds[bi].order == 1) {
                    core_links.push(adj[center][0].0);
                    core_links.push(adj[center][1].0);
                    Some("oxy")
                } else {
                    None
                }
            }
            "S" => {
                let dbl_os: Vec<usize> = adj[center]
                    .iter()
                    .filter_map(|&(nb, bi)| {
                        if mol.bonds[bi].order == 2
                            && mol.atoms[nb].element == "O"
                            && adj[nb].len() == 1
                        {
                            Some(nb)
                        } else {
                            None
                        }
                    })
                    .collect();
                let single_cs: Vec<usize> = adj[center]
                    .iter()
                    .filter_map(|&(nb, bi)| {
                        if mol.bonds[bi].order == 1 && mol.atoms[nb].element == "C" {
                            Some(nb)
                        } else {
                            None
                        }
                    })
                    .collect();
                if dbl_os.len() == 2 && single_cs.len() == 2 {
                    for &ox in &dbl_os {
                        linker_atoms.insert(ox);
                    }
                    core_links = single_cs;
                    Some("sulfonyl")
                } else if dbl_os.is_empty() && single_cs.len() == 2 && adj[center].len() == 2 {
                    core_links = single_cs;
                    Some("sulfanediyl")
                } else {
                    None
                }
            }
            "N" if adj[center].len() == 2 && mol.total_h_count(center) == 1 => {
                if adj[center].iter().all(|&(_, bi)| mol.bonds[bi].order == 1) {
                    core_links.push(adj[center][0].0);
                    core_links.push(adj[center][1].0);
                    Some("azanediyl")
                } else {
                    None
                }
            }
            "C" if mol.total_h_count(center) == 2 && adj[center].len() == 2 => {
                if adj[center].iter().all(|&(nb, bi)| {
                    mol.bonds[bi].order == 1 && rings.iter().any(|r| r.contains(&nb))
                }) {
                    core_links.push(adj[center][0].0);
                    core_links.push(adj[center][1].0);
                    Some("methylene")
                } else {
                    None
                }
            }
            _ => None,
        };

        let Some(lname) = linker_name else {
            continue;
        };
        if core_links.len() != 2 {
            continue;
        }
        // Both sides must be in rings and disconnected when `linker_atoms` is removed
        if !core_links
            .iter()
            .all(|&c| rings.iter().any(|r| r.contains(&c)))
        {
            continue;
        }
        let frag1 = collect_subfragment(mol, core_links[0], &linker_atoms);
        if frag1.contains(&core_links[1]) {
            continue;
        }
        let frag2 = collect_subfragment(mol, core_links[1], &linker_atoms);
        if frag1.len() != frag2.len()
            || frag1.len() + frag2.len() + linker_atoms.len() != mol.atoms.len()
        {
            continue;
        }

        let (sub1, _, map1) = induce_submolecule(mol, stereo, &frag1);
        let (sub2, _, map2) = induce_submolecule(mol, stereo, &frag2);
        let st1 = sub1.perceive_cip_stereo();
        let st2 = sub2.perceive_cip_stereo();
        let Some((name1, loc_map1, has_sub1)) =
            name_single_with_locants(&sub1, &st1, Some(map1[&core_links[0]]))
        else {
            continue;
        };
        let Some((name2, loc_map2, _)) =
            name_single_with_locants(&sub2, &st2, Some(map2[&core_links[1]]))
        else {
            continue;
        };
        if name1 != name2 {
            continue;
        }
        let (Some(&l1), Some(&l2)) = (
            loc_map1.get(&map1[&core_links[0]]),
            loc_map2.get(&map2[&core_links[1]]),
        ) else {
            continue;
        };
        if l1 != l2 {
            continue;
        }
        // Only prefer multiplicative nomenclature when the parent has a functional suffix or substituent
        let is_functional_or_sub = has_sub1 || name1 != "benzene";
        if !is_functional_or_sub {
            continue;
        }
        if has_sub1 {
            return Some(format!("{l1},{l2}'-{lname}bis({name1})"));
        } else {
            return Some(format!("{l1},{l2}'-{lname}di{name1}"));
        }
    }

    None
}

/// Name an acyclic carboxylic ester `R-C(=O)-O-R'` as `<alkyl-yl> <acyl-oate / carboxylate>`.
fn name_ester_molecule(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    hit: &FunctionalHit,
) -> Option<String> {
    let c_acyl = hit.group_atom;
    let o_ester = hit.extra_atom?;
    let o_dbl = *hit.group_atoms.iter().find(|&&x| x != o_ester)?;
    let adj = mol.adjacency();
    let r_prime = adj[o_ester].iter().find(|&&(x, _)| x != c_acyl)?.0;

    let mut alk_blocked = HashSet::new();
    alk_blocked.insert(o_ester);
    let (alkyl_part, _) = name_substituent(mol, r_prime, o_ester, 1, &alk_blocked);

    // Replace `-O-R'` by `-OH` to get the parent carboxylic acid name, then convert `-ic acid` -> `-ate`!
    let alk_atoms = collect_subfragment(mol, r_prime, &alk_blocked);
    let mut acid_mol = AnalytMol::new();
    let mut old_to_new = HashMap::new();
    for i in 0..mol.atoms.len() {
        if alk_atoms.contains(&i) {
            continue;
        }
        let mut a = mol.atoms[i].clone();
        if i == o_ester {
            a.implicit_h = Some(1);
        } else if let Some(&rs) = stereo.atom_rs.get(&i) {
            a.stereo_tag = if rs == 'R' {
                AtomStereoTag::R
            } else {
                AtomStereoTag::S
            };
        }
        let ni = acid_mol.add_atom(a);
        old_to_new.insert(i, ni);
    }
    for (bi, b) in mol.bonds.iter().enumerate() {
        if let (Some(&na), Some(&nb)) = (old_to_new.get(&b.a), old_to_new.get(&b.b)) {
            let mut nbond = AnalytBond {
                a: na,
                b: nb,
                order: b.order,
                aromatic: b.aromatic,
                stereo: b.stereo,
            };
            if let Some(&ez) = stereo.bond_ez.get(&bi) {
                nbond.stereo = if ez == 'Z' {
                    BondStereoTag::Cis
                } else {
                    BondStereoTag::Trans
                };
            }
            acid_mol.add_bond(nbond);
        }
    }
    let _ = o_dbl;
    acid_mol.perceive_aromaticity();
    let acid_st = acid_mol.perceive_cip_stereo();
    let (acid_name, _, _) = name_single_with_locants(&acid_mol, &acid_st, None)?;
    let ate_name = {
        let stem = acid_name.strip_suffix("ic acid")?;
        format!("{stem}ate")
    };
    Some(format!("{alkyl_part} {ate_name}"))
}

// --- Retained Heterocycles, Hantzsch-Widman & Fused Ring Parents (`namer-hetero.ts`) ---

#[derive(Debug, Clone)]
struct RingParentSpec {
    name: String,
    /// Candidate atom-index orderings (`order[locant - 1] = atom_idx`).
    orders: Vec<Vec<usize>>,
    is_aromatic_parent: bool,
}

fn identify_monocycle_parent(
    mol: &AnalytMol,
    ring: &[usize],
    arom: &[bool],
) -> Option<RingParentSpec> {
    let k = ring.len();
    let adj = mol.adjacency();
    let is_arom = ring.iter().all(|&a| arom[a]);
    let ring_set: HashSet<usize> = ring.iter().copied().collect();

    // Generate all cyclic walk orderings (`cw` and `ccw` from every ring vertex)
    let all_cyclic_orders = |start_filter: &dyn Fn(usize) -> bool| -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        for pos in 0..k {
            let u = ring[pos];
            if !start_filter(u) {
                continue;
            }
            let cw: Vec<usize> = (0..k).map(|d| ring[(pos + d) % k]).collect();
            let mut ccw = Vec::with_capacity(k);
            ccw.push(u);
            for d in 1..k {
                ccw.push(ring[(pos + k - d) % k]);
            }
            out.push(cw);
            out.push(ccw);
        }
        out
    };

    let hetero: Vec<usize> = ring
        .iter()
        .copied()
        .filter(|&a| mol.atoms[a].element != "C")
        .collect();

    if hetero.is_empty() {
        if k == 6 && is_arom {
            return Some(RingParentSpec {
                name: "benzene".to_string(),
                orders: all_cyclic_orders(&|_| true),
                is_aromatic_parent: true,
            });
        }
        return Some(RingParentSpec {
            name: format!("cyclo{}", alkane_stem(k)),
            orders: all_cyclic_orders(&|_| true),
            is_aromatic_parent: false,
        });
    }

    // Single-heteroatom monocycles
    if hetero.len() == 1 {
        let h_atom = hetero[0];
        let hel = mol.atoms[h_atom].element.as_str();
        let orders = all_cyclic_orders(&|u| u == h_atom);
        let all_ring_single = ring.iter().all(|&u| {
            adj[u]
                .iter()
                .filter(|&&(v, _)| ring_set.contains(&v))
                .all(|&(_, bi)| mol.bonds[bi].order == 1)
        });

        let name = match (k, hel, is_arom, all_ring_single) {
            (3, "O", false, true) => Some("oxirane"),
            (3, "N", false, true) => Some("aziridine"),
            (3, "S", false, true) => Some("thiirane"),
            (4, "O", false, true) => Some("oxetane"),
            (4, "N", false, true) => Some("azetidine"),
            (5, "O", true, _) => Some("furan"),
            (5, "S", true, _) => Some("thiophene"),
            (5, "N", true, _) => Some("1H-pyrrole"),
            (5, "O", false, true) => Some("oxolane"),
            (5, "S", false, true) => Some("thiolane"),
            (5, "N", false, true) => Some("pyrrolidine"),
            (6, "N", true, _) => Some("pyridine"),
            (6, "O", false, true) => Some("oxane"),
            (6, "N", false, true) => Some("piperidine"),
            (7, "O", false, true) => Some("oxepane"),
            (7, "N", false, true) => Some("azepane"),
            _ => None,
        };
        if let Some(nm) = name {
            return Some(RingParentSpec {
                name: nm.to_string(),
                orders,
                is_aromatic_parent: is_arom,
            });
        }
    }

    // Two-heteroatom 5- and 6-membered retained/PIN heterocycles
    if hetero.len() == 2 {
        let n_cnt = hetero
            .iter()
            .filter(|&&a| mol.atoms[a].element == "N")
            .count();
        let o_cnt = hetero
            .iter()
            .filter(|&&a| mol.atoms[a].element == "O")
            .count();
        let s_cnt = hetero
            .iter()
            .filter(|&&a| mol.atoms[a].element == "S")
            .count();

        if k == 5 && is_arom {
            if n_cnt == 2 {
                // Start at NH-type pyrrolic N (`total_h >= 1` or 3 single sigma bonds) and number toward pyridine-like N!
                let nh_atoms: Vec<usize> = hetero
                    .iter()
                    .copied()
                    .filter(|&a| {
                        mol.total_h_count(a) >= 1
                            || adj[a]
                                .iter()
                                .filter(|&&(_, bi)| !mol.bonds[bi].aromatic)
                                .count()
                                >= 1
                    })
                    .collect();
                let start_set: Vec<usize> = if nh_atoms.is_empty() {
                    hetero.clone()
                } else {
                    nh_atoms
                };
                let mut valid_orders = Vec::new();
                let mut second_loc = 99usize;
                for ord in all_cyclic_orders(&|u| start_set.contains(&u)) {
                    let loc2 = ord
                        .iter()
                        .position(|&x| x != ord[0] && hetero.contains(&x))
                        .unwrap()
                        + 1;
                    if loc2 < second_loc {
                        second_loc = loc2;
                        valid_orders.clear();
                    }
                    if loc2 == second_loc {
                        valid_orders.push(ord);
                    }
                }
                let nm = if second_loc == 2 {
                    "1H-pyrazole"
                } else {
                    "1H-imidazole"
                };
                return Some(RingParentSpec {
                    name: nm.to_string(),
                    orders: valid_orders,
                    is_aromatic_parent: true,
                });
            }
            if (o_cnt == 1 || s_cnt == 1) && n_cnt == 1 {
                let chalc = *hetero
                    .iter()
                    .find(|&&a| mol.atoms[a].element != "N")
                    .unwrap();
                let mut valid_orders = Vec::new();
                let mut n_loc = 99usize;
                for ord in all_cyclic_orders(&|u| u == chalc) {
                    let loc2 = ord
                        .iter()
                        .position(|&x| mol.atoms[x].element == "N")
                        .unwrap()
                        + 1;
                    if loc2 < n_loc {
                        n_loc = loc2;
                        valid_orders.clear();
                    }
                    if loc2 == n_loc {
                        valid_orders.push(ord);
                    }
                }
                let base = if o_cnt == 1 { "oxazole" } else { "thiazole" };
                return Some(RingParentSpec {
                    name: format!("1,{n_loc}-{base}"),
                    orders: valid_orders,
                    is_aromatic_parent: true,
                });
            }
        }

        if k == 6 {
            if is_arom && n_cnt == 2 {
                let mut valid_orders = Vec::new();
                let mut loc2_best = 99usize;
                for ord in all_cyclic_orders(&|u| hetero.contains(&u)) {
                    let loc2 = ord
                        .iter()
                        .position(|&x| x != ord[0] && hetero.contains(&x))
                        .unwrap()
                        + 1;
                    if loc2 < loc2_best {
                        loc2_best = loc2;
                        valid_orders.clear();
                    }
                    if loc2 == loc2_best {
                        valid_orders.push(ord);
                    }
                }
                let nm = match loc2_best {
                    2 => "pyridazine",
                    3 => "pyrimidine",
                    _ => "pyrazine",
                };
                return Some(RingParentSpec {
                    name: nm.to_string(),
                    orders: valid_orders,
                    is_aromatic_parent: true,
                });
            }
            if !is_arom && n_cnt == 2 {
                let mut valid_orders = Vec::new();
                for ord in all_cyclic_orders(&|u| hetero.contains(&u)) {
                    if ord.get(3).is_some_and(|x| hetero.contains(x)) {
                        valid_orders.push(ord);
                    }
                }
                if !valid_orders.is_empty() {
                    return Some(RingParentSpec {
                        name: "piperazine".to_string(),
                        orders: valid_orders,
                        is_aromatic_parent: false,
                    });
                }
            }
            if !is_arom && o_cnt == 1 && n_cnt == 1 {
                let o_atom = *hetero
                    .iter()
                    .find(|&&a| mol.atoms[a].element == "O")
                    .unwrap();
                let mut valid_orders = Vec::new();
                for ord in all_cyclic_orders(&|u| u == o_atom) {
                    if ord.get(3).is_some_and(|&x| mol.atoms[x].element == "N") {
                        valid_orders.push(ord);
                    }
                }
                if !valid_orders.is_empty() {
                    return Some(RingParentSpec {
                        name: "morpholine".to_string(),
                        orders: valid_orders,
                        is_aromatic_parent: false,
                    });
                }
            }
        }
    }

    // General skeletal replacement fallback (`oxa`/`aza`/`thia` + `cycloalkane`)
    let mut best_locs: Option<Vec<usize>> = None;
    let mut best_orders = Vec::new();
    for ord in all_cyclic_orders(&|u| hetero.contains(&u)) {
        let mut hlocs: Vec<usize> = ord
            .iter()
            .enumerate()
            .filter(|&(_, &a)| mol.atoms[a].element != "C")
            .map(|(p, _)| p + 1)
            .collect();
        hlocs.sort_unstable();
        if best_locs.as_ref().is_none_or(|b| hlocs < *b) {
            best_locs = Some(hlocs);
            best_orders.clear();
            best_orders.push(ord);
        } else if best_locs.as_ref() == Some(&hlocs) {
            best_orders.push(ord);
        }
    }
    if let Some(first_ord) = best_orders.first() {
        let aza_oxa = skeletal_aza_oxa_prefix(mol, first_ord);
        return Some(RingParentSpec {
            name: format!("{aza_oxa}cyclo{}", alkane_stem(k)),
            orders: best_orders,
            is_aromatic_parent: false,
        });
    }

    None
}

/// Identify common ortho-fused bicyclic aromatic systems (`naphthalene`, `quinoline`, `isoquinoline`,
/// `1H-indole`, `1-benzofuran`, `1-benzothiophene`, `1H-benzimidazole`, `9H-purine`).
fn identify_fused_bicyclic_parent(
    mol: &AnalytMol,
    sys_rings: &[Vec<usize>],
    arom: &[bool],
) -> Option<RingParentSpec> {
    if sys_rings.len() != 2 {
        return None;
    }
    let mut r_pair = sys_rings.to_vec();
    r_pair.sort_by_key(|r| r.len());
    let r_small = &r_pair[0];
    let r_large = &r_pair[1];
    let shared: Vec<usize> = r_small
        .iter()
        .copied()
        .filter(|a| r_large.contains(a))
        .collect();
    if shared.len() != 2 {
        return None;
    }
    let mut union_atoms: Vec<usize> = r_small.clone();
    for &a in r_large {
        if !union_atoms.contains(&a) {
            union_atoms.push(a);
        }
    }
    if !union_atoms.iter().all(|&a| arom[a]) {
        return None;
    }

    // Helper: walk perimeter of ring `r` starting adjacent to fusion atom `f_start` (moving AWAY from `f_other`)
    let perimeter_arc = |r: &[usize], f_start: usize, f_other: usize| -> Vec<usize> {
        let m = r.len();
        let p = r.iter().position(|&x| x == f_start).unwrap();
        let fwd_nb = r[(p + 1) % m];
        let step: isize = if fwd_nb == f_other { -1 } else { 1 };
        let mut arc = Vec::new();
        for d in 1..(m - 1) {
            let idx = ((p as isize + step * (d as isize)).rem_euclid(m as isize)) as usize;
            arc.push(r[idx]);
        }
        arc
    };

    // 6+6 fused (`naphthalene`, `quinoline`, `isoquinoline`)
    if r_small.len() == 6 && r_large.len() == 6 {
        let hetero: Vec<usize> = union_atoms
            .iter()
            .copied()
            .filter(|&a| mol.atoms[a].element != "C")
            .collect();
        // Standard IUPAC peripheral numbering of 6,6-fused system (positions 1..8 on non-fusion atoms, 9..10 on fusion)
        let mut perim_orders = Vec::new();
        for &(r_first, r_second) in &[(r_small, r_large), (r_large, r_small)] {
            for &(f1, f2) in &[(shared[0], shared[1]), (shared[1], shared[0])] {
                let arc1 = perimeter_arc(r_first, f1, f2); // 4 atoms: pos 1,2,3,4
                let arc2 = perimeter_arc(r_second, f2, f1); // 4 atoms: pos 5,6,7,8
                let mut ord = Vec::with_capacity(10);
                ord.extend(arc1);
                ord.extend(arc2);
                ord.push(f1); // 4a (9)
                ord.push(f2); // 8a (10)
                perim_orders.push(ord);
            }
        }
        if hetero.is_empty() {
            return Some(RingParentSpec {
                name: "naphthalene".to_string(),
                orders: perim_orders,
                is_aromatic_parent: true,
            });
        }
        if hetero.len() == 1 && mol.atoms[hetero[0]].element == "N" && !shared.contains(&hetero[0])
        {
            let n_atom = hetero[0];
            let pos1_orders: Vec<Vec<usize>> = perim_orders
                .iter()
                .filter(|ord| ord[0] == n_atom)
                .cloned()
                .collect();
            if !pos1_orders.is_empty() {
                return Some(RingParentSpec {
                    name: "quinoline".to_string(),
                    orders: pos1_orders,
                    is_aromatic_parent: true,
                });
            }
            let pos2_orders: Vec<Vec<usize>> = perim_orders
                .iter()
                .filter(|ord| ord[1] == n_atom)
                .cloned()
                .collect();
            if !pos2_orders.is_empty() {
                return Some(RingParentSpec {
                    name: "isoquinoline".to_string(),
                    orders: pos2_orders,
                    is_aromatic_parent: true,
                });
            }
        }
    }

    // 5+6 fused (`1H-indole`, `1-benzofuran`, `1-benzothiophene`, `1H-benzimidazole`, `9H-purine`)
    if r_small.len() == 5 && r_large.len() == 6 {
        let h_small: Vec<usize> = r_small
            .iter()
            .copied()
            .filter(|&a| mol.atoms[a].element != "C" && !shared.contains(&a))
            .collect();
        let h_large: Vec<usize> = r_large
            .iter()
            .copied()
            .filter(|&a| mol.atoms[a].element != "C" && !shared.contains(&a))
            .collect();

        // Benzo-fused 5-ring (`h_large` empty, `shared` all C)
        if h_large.is_empty() && shared.iter().all(|&a| mol.atoms[a].element == "C") {
            let mut orders = Vec::new();
            for &(f1, f2) in &[(shared[0], shared[1]), (shared[1], shared[0])] {
                let arc5 = perimeter_arc(r_small, f1, f2); // 3 atoms: pos 1,2,3
                let arc6 = perimeter_arc(r_large, f2, f1); // 4 atoms: pos 4,5,6,7
                let mut ord = Vec::with_capacity(9);
                ord.extend(arc5);
                ord.extend(arc6);
                ord.push(f2);
                ord.push(f1);
                orders.push(ord);
            }
            if h_small.len() == 1 {
                let ha = h_small[0];
                let hel = mol.atoms[ha].element.as_str();
                let pos1: Vec<Vec<usize>> = orders.iter().filter(|o| o[0] == ha).cloned().collect();
                if !pos1.is_empty() {
                    let nm = match hel {
                        "N" => "1H-indole",
                        "O" => "1-benzofuran",
                        "S" => "1-benzothiophene",
                        _ => return None,
                    };
                    return Some(RingParentSpec {
                        name: nm.to_string(),
                        orders: pos1,
                        is_aromatic_parent: true,
                    });
                }
            }
            if h_small.len() == 2 && h_small.iter().all(|&a| mol.atoms[a].element == "N") {
                let pos13: Vec<Vec<usize>> = orders
                    .into_iter()
                    .filter(|o| h_small.contains(&o[0]) && h_small.contains(&o[2]))
                    .collect();
                if !pos13.is_empty() {
                    return Some(RingParentSpec {
                        name: "1H-benzimidazole".to_string(),
                        orders: pos13,
                        is_aromatic_parent: true,
                    });
                }
            }
        }
        // Purine (`2 N` in 6-ring + `2 N` in 5-ring)
        if h_small.len() == 2
            && h_large.len() == 2
            && h_small
                .iter()
                .chain(h_large.iter())
                .all(|&a| mol.atoms[a].element == "N")
        {
            return Some(RingParentSpec {
                name: "9H-purine".to_string(),
                orders: vec![union_atoms],
                is_aromatic_parent: true,
            });
        }
    }

    None
}

/// Core single-component IUPAC namer returning `(preferred_name, atom_to_locant_map, has_substituents)`.
fn name_single_with_locants(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    tie_break_atom: Option<usize>,
) -> Option<(String, HashMap<usize, usize>, bool)> {
    if let Some(inorg) = try_inorganic_or_hydride_name(mol) {
        let mut m = HashMap::new();
        if !mol.atoms.is_empty() {
            m.insert(0, 1);
        }
        return Some((inorg, m, false));
    }

    let hits = perceive_functional_hits(mol);
    let senior_kind = hits
        .iter()
        .map(|h| h.kind)
        .min()
        .unwrap_or(PrincipalGroup::None);

    // Acyclic ester `R-C(=O)-O-R'`
    if senior_kind == PrincipalGroup::Ester {
        let ester_hit = hits.iter().find(|h| h.kind == PrincipalGroup::Ester)?;
        if let Some(nm) = name_ester_molecule(mol, stereo, ester_hit) {
            return Some((nm, HashMap::new(), false));
        }
    }

    // Check symmetric multiplicative assembly (`4,4'-oxydibenzoic acid`) when caller isn't already inside one
    if tie_break_atom.is_none() {
        if let Some(mult) = try_multiplicative_name(mol, stereo) {
            return Some((mult, HashMap::new(), true));
        }
    }

    let adj = mol.adjacency();
    let sssr = mol.sssr_rings();
    let arom = mol.aromatic_atoms();

    // Group SSSR rings into connected ring systems (sharing >= 1 atom)
    let mut ring_systems: Vec<(Vec<usize>, Vec<Vec<usize>>)> = Vec::new();
    let mut used_ring = vec![false; sssr.len()];
    for i in 0..sssr.len() {
        if used_ring[i] {
            continue;
        }
        used_ring[i] = true;
        let mut sys_atoms: BTreeSet<usize> = sssr[i].iter().copied().collect();
        let mut sys_rings = vec![sssr[i].clone()];
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..sssr.len() {
                if !used_ring[j] && sssr[j].iter().any(|a| sys_atoms.contains(a)) {
                    used_ring[j] = true;
                    sys_atoms.extend(sssr[j].iter().copied());
                    sys_rings.push(sssr[j].clone());
                    changed = true;
                }
            }
        }
        ring_systems.push((sys_atoms.into_iter().collect(), sys_rings));
    }

    // Determine whether the principal parent is a RING system or an ACYCLIC chain (IUPAC P-44.1 / P-44.4):
    // 1. If `senior_kind != None`: pick the system (ring or acyclic) carrying the maximum count of `senior_kind` groups!
    //    Note: For `-COOH`, `-CN`, `-CHO`, `-CONH2`, `-COCl`, if the group carbon is directly attached to a ring atom `r`
    //    with NO intervening chain carbons (e.g., `benzoic acid`, `cyclohexanecarboxylic acid`), that group belongs to the ring!
    //    Whereas in `2-[4-(2-methylpropyl)phenyl]propanoic acid` or `1-(4-bromophenyl)ethan-1-one`, the `-COOH` or `>C=O` is on a >=2-carbon chain, so the acyclic chain wins!
    let ring_atom_set: HashSet<usize> = ring_systems
        .iter()
        .flat_map(|(atoms, _)| atoms.iter().copied())
        .collect();

    let hit_ring_anchor = |h: &FunctionalHit| -> Option<usize> {
        if ring_atom_set.contains(&h.anchor_carbon) {
            return Some(h.anchor_carbon);
        }
        if matches!(
            h.kind,
            PrincipalGroup::CarboxylicAcid
                | PrincipalGroup::AcidHalide
                | PrincipalGroup::Amide
                | PrincipalGroup::Nitrile
                | PrincipalGroup::Aldehyde
        ) {
            // Exocyclic 1-carbon suffix (`-COOH`, `-CN`, `-CHO`, `-CONH2`, `-COX`) directly bonded to a ring atom
            for &(nb, bi) in &adj[h.group_atom] {
                if !h.group_atoms.contains(&nb)
                    && mol.bonds[bi].order == 1
                    && ring_atom_set.contains(&nb)
                {
                    return Some(nb);
                }
            }
        }
        None
    };

    let senior_hits: Vec<&FunctionalHit> = hits.iter().filter(|h| h.kind == senior_kind).collect();

    if senior_kind != PrincipalGroup::None {
        // Check if any ring system carries >= senior hits than any acyclic chain
        let mut best_ring_idx: Option<usize> = None;
        let mut best_ring_cnt = 0usize;
        for (ri, (ratoms, _)) in ring_systems.iter().enumerate() {
            let rset: HashSet<usize> = ratoms.iter().copied().collect();
            let cnt = senior_hits
                .iter()
                .filter(|h| hit_ring_anchor(h).is_some_and(|ra| rset.contains(&ra)))
                .count();
            if cnt > best_ring_cnt {
                best_ring_cnt = cnt;
                best_ring_idx = Some(ri);
            }
        }
        let acyclic_cnt = senior_hits
            .iter()
            .filter(|h| hit_ring_anchor(h).is_none())
            .count();

        if let Some(ri) = best_ring_idx {
            if best_ring_cnt >= acyclic_cnt && best_ring_cnt > 0 {
                return name_ring_system_parent(
                    mol,
                    stereo,
                    &ring_systems[ri].0,
                    &ring_systems[ri].1,
                    senior_kind,
                    &senior_hits,
                    &arom,
                    tie_break_atom,
                );
            }
        }
        return name_acyclic_parent(mol, stereo, senior_kind, &senior_hits);
    }

    // No principal functional suffix (`senior_kind == None`):
    // IUPAC 2013 P-44.1.2.2: Rings ALWAYS have seniority over chains when neither has a principal functional suffix!
    if !ring_systems.is_empty() {
        // Pick most senior ring system (most heteroatoms / largest number of rings / largest ring atom count)
        let mut best_ri = 0usize;
        let mut best_key = (0usize, 0usize, 0usize);
        for (ri, (ratoms, srings)) in ring_systems.iter().enumerate() {
            let h_cnt = ratoms
                .iter()
                .filter(|&&a| mol.atoms[a].element != "C")
                .count();
            let key = (h_cnt, srings.len(), ratoms.len());
            if key > best_key {
                best_key = key;
                best_ri = ri;
            }
        }
        return name_ring_system_parent(
            mol,
            stereo,
            &ring_systems[best_ri].0,
            &ring_systems[best_ri].1,
            PrincipalGroup::None,
            &[],
            &arom,
            tie_break_atom,
        );
    }

    name_acyclic_parent(mol, stereo, PrincipalGroup::None, &[])
}

#[allow(clippy::too_many_arguments)]
fn name_ring_system_parent(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    sys_atoms: &[usize],
    sys_rings: &[Vec<usize>],
    senior_kind: PrincipalGroup,
    senior_hits: &[&FunctionalHit],
    arom: &[bool],
    tie_break_atom: Option<usize>,
) -> Option<(String, HashMap<usize, usize>, bool)> {
    let adj = mol.adjacency();
    let sys_set: HashSet<usize> = sys_atoms.iter().copied().collect();

    // Identify parent ring descriptor & valid candidate numberings
    let parent_spec: RingParentSpec = if sys_rings.len() == 1 {
        identify_monocycle_parent(mol, &sys_rings[0], arom)?
    } else if let Some(fused) = identify_fused_bicyclic_parent(mol, sys_rings, arom) {
        fused
    } else {
        let vb = analyze_von_baeyer_or_spiro(mol, sys_atoms)?;
        RingParentSpec {
            name: vb.descriptor,
            orders: vb.candidate_orders,
            is_aromatic_parent: false,
        }
    };

    // Collect senior functional hits anchored on `sys_set`
    let mut ring_hits: Vec<(&FunctionalHit, usize)> = Vec::new();
    let mut consumed_exo: HashSet<usize> = HashSet::new();
    for &h in senior_hits {
        if sys_set.contains(&h.anchor_carbon) {
            for &ga in &h.group_atoms {
                consumed_exo.insert(ga);
            }
            ring_hits.push((h, h.anchor_carbon));
        } else {
            for &(nb, bi) in &adj[h.group_atom] {
                if !h.group_atoms.contains(&nb) && mol.bonds[bi].order == 1 && sys_set.contains(&nb)
                {
                    consumed_exo.insert(h.group_atom);
                    for &ga in &h.group_atoms {
                        consumed_exo.insert(ga);
                    }
                    ring_hits.push((h, nb));
                    break;
                }
            }
        }
    }

    let is_benzene = parent_spec.name == "benzene";
    let is_von_baeyer_or_cyclo = parent_spec.name.starts_with("bicyclo[")
        || parent_spec.name.starts_with("tricyclo[")
        || parent_spec.name.starts_with("tetracyclo[")
        || parent_spec.name.starts_with("pentacyclo[")
        || parent_spec.name.starts_with("spiro[")
        || parent_spec.name.starts_with("cyclo")
        || parent_spec.name.contains("oxacyclo")
        || parent_spec.name.contains("azacyclo");

    let mut blocked = sys_set.clone();
    for &x in &consumed_exo {
        blocked.insert(x);
    }

    // Evaluate every candidate numbering order, choosing lowest locants for:
    // (1) skeletal heteroatoms (for von Baeyer), (2) principal functional groups, (3) double/triple bonds,
    // (4) all substituents, (5) alphabetical first substituent!
    type CandKey = (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<String>,
        usize,
    );
    let mut best_cand: Option<(CandKey, String, HashMap<usize, usize>, bool)> = None;

    for ord in &parent_spec.orders {
        let mut atom_to_loc = HashMap::with_capacity(ord.len());
        for (idx, &a) in ord.iter().enumerate() {
            atom_to_loc.insert(a, idx + 1);
        }

        let mut het_locs: Vec<usize> = ord
            .iter()
            .enumerate()
            .filter(|&(_, &a)| mol.atoms[a].element != "C")
            .map(|(p, _)| p + 1)
            .collect();
        het_locs.sort_unstable();

        let mut fg_locs: Vec<usize> = ring_hits
            .iter()
            .filter_map(|&(_, anchor)| atom_to_loc.get(&anchor).copied())
            .collect();
        fg_locs.sort_unstable();

        // Ring unsaturations (only for non-aromatic parents)
        let mut dbl_locs = Vec::new();
        if !parent_spec.is_aromatic_parent {
            for b in &mol.bonds {
                if b.order == 2 && !b.aromatic && sys_set.contains(&b.a) && sys_set.contains(&b.b) {
                    if let (Some(&la), Some(&lb)) = (atom_to_loc.get(&b.a), atom_to_loc.get(&b.b)) {
                        let lo = la.min(lb);
                        dbl_locs.push(lo);
                    }
                }
            }
            dbl_locs.sort_unstable();
        }

        // Collect substituent prefixes on ring atoms ( plus `N`-substituents on principal amide/amine!)
        let mut sub_items: Vec<SubPrefixItem> = Vec::new();
        let mut sub_locs: Vec<usize> = Vec::new();

        for (idx, &ra) in ord.iter().enumerate() {
            let loc = idx + 1;
            for &(nb, bi) in &adj[ra] {
                if !blocked.contains(&nb) {
                    let (sname, is_comp) =
                        name_substituent(mol, nb, ra, mol.bonds[bi].order, &blocked);
                    sub_locs.push(loc);
                    sub_items.push(SubPrefixItem {
                        locant: loc.to_string(),
                        base_name: sname,
                        is_compound: is_comp,
                    });
                }
            }
        }
        // Check N-substituents on principal ring amide (`N-methylbenzamide`) or aniline (`N,N-dimethylaniline`)
        for &(h, _) in &ring_hits {
            if h.kind == PrincipalGroup::Amide {
                if let Some(n_idx) = h.extra_atom {
                    for &(nb, bi) in &adj[n_idx] {
                        if nb != h.group_atom {
                            let mut n_blk = blocked.clone();
                            n_blk.insert(n_idx);
                            let (sname, is_comp) =
                                name_substituent(mol, nb, n_idx, mol.bonds[bi].order, &n_blk);
                            sub_items.push(SubPrefixItem {
                                locant: "N".to_string(),
                                base_name: sname,
                                is_compound: is_comp,
                            });
                        }
                    }
                }
            } else if h.kind == PrincipalGroup::Amine {
                let n_idx = h.group_atom;
                for &(nb, bi) in &adj[n_idx] {
                    if nb != h.anchor_carbon {
                        let mut n_blk = blocked.clone();
                        n_blk.insert(n_idx);
                        let (sname, is_comp) =
                            name_substituent(mol, nb, n_idx, mol.bonds[bi].order, &n_blk);
                        sub_items.push(SubPrefixItem {
                            locant: "N".to_string(),
                            base_name: sname,
                            is_compound: is_comp,
                        });
                    }
                }
            }
        }

        sub_locs.sort_unstable();
        let mut alpha_pairs: Vec<(String, usize)> = sub_items
            .iter()
            .map(|it| {
                (
                    iupac_alpha_key(&it.base_name),
                    it.locant.parse::<usize>().unwrap_or(0),
                )
            })
            .collect();
        alpha_pairs.sort();
        let alpha_key: Vec<String> = alpha_pairs
            .into_iter()
            .map(|(k, l)| format!("{k}:{l}"))
            .collect();
        let tie_loc = tie_break_atom
            .and_then(|ta| atom_to_loc.get(&ta).copied())
            .unwrap_or(999);

        let has_subs = !sub_items.is_empty();
        // Omit redundant `"1-"` on monosubstituted benzene/cycloalkane with no functional group (`bromobenzene`, `methylcyclohexane`)
        if fg_locs.is_empty()
            && dbl_locs.is_empty()
            && sub_items.len() == 1
            && (is_benzene || (sys_rings.len() == 1 && het_locs.is_empty()))
            && sub_items[0].locant == "1"
        {
            sub_items[0].locant.clear();
        }

        let sub_prefix = assemble_prefixes(&sub_items, false);

        // Build core stem + unsaturation + principal suffix
        let core_and_suffix = build_ring_core_and_suffix(
            mol,
            &parent_spec,
            ord,
            is_von_baeyer_or_cyclo,
            &dbl_locs,
            senior_kind,
            &ring_hits,
            &fg_locs,
        );

        let full_body = if sub_prefix.is_empty() {
            core_and_suffix
        } else if sub_prefix.ends_with(')')
            || sub_prefix.ends_with(']')
            || !sub_prefix.chars().any(|c| c.is_ascii_digit())
            || core_and_suffix.starts_with(|c: char| c.is_ascii_digit())
        {
            if core_and_suffix.starts_with(|c: char| c.is_ascii_digit()) {
                format!("{sub_prefix}-{core_and_suffix}")
            } else {
                format!("{sub_prefix}{core_and_suffix}")
            }
        } else {
            format!("{sub_prefix}{core_and_suffix}")
        };

        let st_pref = format_stereo_prefix(mol, stereo, &atom_to_loc, ord.len());
        let final_name = format!("{st_pref}{full_body}");
        let key: CandKey = (het_locs, fg_locs, dbl_locs, sub_locs, alpha_key, tie_loc);

        if best_cand.as_ref().is_none_or(|b| key < b.0) {
            best_cand = Some((key, final_name, atom_to_loc, has_subs));
        }
    }

    best_cand.map(|(_, nm, map, hs)| (nm, map, hs))
}

#[allow(clippy::too_many_arguments)]
fn build_ring_core_and_suffix(
    mol: &AnalytMol,
    parent_spec: &RingParentSpec,
    ord: &[usize],
    is_von_baeyer_or_cyclo: bool,
    dbl_locs: &[usize],
    senior_kind: PrincipalGroup,
    ring_hits: &[(&FunctionalHit, usize)],
    fg_locs: &[usize],
) -> String {
    // Retained single-group benzene parent names (IUPAC 2013 P-65.1.1.1 / P-63.1.1 / P-62.2.1.1):
    // `benzoic acid`, `benzenesulfonic acid`, `benzoyl chloride`, `benzamide`, `benzonitrile`, `benzaldehyde`, `phenol`, `aniline`!
    if parent_spec.name == "benzene" && fg_locs == [1] && ring_hits.len() == 1 {
        let retained = match senior_kind {
            PrincipalGroup::CarboxylicAcid => Some("benzoic acid".to_string()),
            PrincipalGroup::SulfonicAcid => Some("benzenesulfonic acid".to_string()),
            PrincipalGroup::AcidHalide => {
                let hal = ring_hits[0]
                    .0
                    .extra_atom
                    .map(|x| mol.atoms[x].element.as_str())
                    .unwrap_or("Cl");
                let hword = match hal {
                    "F" => "fluoride",
                    "Br" => "bromide",
                    "I" => "iodide",
                    _ => "chloride",
                };
                Some(format!("benzoyl {hword}"))
            }
            PrincipalGroup::Amide => Some("benzamide".to_string()),
            PrincipalGroup::Nitrile => Some("benzonitrile".to_string()),
            PrincipalGroup::Aldehyde => Some("benzaldehyde".to_string()),
            PrincipalGroup::Alcohol => Some("phenol".to_string()),
            PrincipalGroup::Amine => Some("aniline".to_string()),
            _ => None,
        };
        if let Some(ret) = retained {
            return ret;
        }
    }

    // Build ring hydride stem
    let need_drop_e = !fg_locs.is_empty()
        && matches!(
            senior_kind,
            PrincipalGroup::Ketone | PrincipalGroup::Alcohol | PrincipalGroup::Amine
        )
        && fg_locs.len() == 1;

    let base_hydride = if parent_spec.name.starts_with("bicyclo[")
        || parent_spec.name.starts_with("tricyclo[")
        || parent_spec.name.starts_with("tetracyclo[")
        || parent_spec.name.starts_with("pentacyclo[")
        || parent_spec.name.starts_with("spiro[")
    {
        let aza_oxa = skeletal_aza_oxa_prefix(mol, ord);
        let stem = alkane_stem(ord.len());
        let unsat = format_unsat_infix(&stem, dbl_locs, &[], !need_drop_e);
        if aza_oxa.is_empty() {
            format!("{}{unsat}", parent_spec.name)
        } else {
            format!("{aza_oxa}{}{unsat}", parent_spec.name)
        }
    } else if is_von_baeyer_or_cyclo {
        // `cycloalk` or `oxacycloalk`
        if dbl_locs.is_empty() {
            if need_drop_e {
                format!("{}an", parent_spec.name)
            } else {
                format!("{}ane", parent_spec.name)
            }
        } else if dbl_locs == [1] && fg_locs.is_empty() {
            format!("{}ene", parent_spec.name)
        } else {
            format_unsat_infix(&parent_spec.name, dbl_locs, &[], !need_drop_e)
        }
    } else if need_drop_e && parent_spec.name.ends_with('e') {
        parent_spec.name[..parent_spec.name.len() - 1].to_string()
    } else {
        parent_spec.name.clone()
    };

    if fg_locs.is_empty() {
        return base_hydride;
    }

    let cnt = fg_locs.len();
    let mp = simple_mult_prefix(cnt);
    let loc_s = fg_locs
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let omit_loc_1 =
        cnt == 1 && fg_locs == [1] && dbl_locs.is_empty() && parent_spec.name.starts_with("cyclo");

    let suf = match senior_kind {
        PrincipalGroup::CarboxylicAcid => format!("{mp}carboxylic acid"),
        PrincipalGroup::SulfonicAcid => format!("{mp}sulfonic acid"),
        PrincipalGroup::AcidHalide => {
            let hal = ring_hits[0]
                .0
                .extra_atom
                .map(|x| mol.atoms[x].element.as_str())
                .unwrap_or("Cl");
            let hword = match hal {
                "F" => "fluoride",
                "Br" => "bromide",
                "I" => "iodide",
                _ => "chloride",
            };
            format!("{mp}carbonyl {hword}")
        }
        PrincipalGroup::Amide => format!("{mp}carboxamide"),
        PrincipalGroup::Nitrile => format!("{mp}carbonitrile"),
        PrincipalGroup::Aldehyde => format!("{mp}carbaldehyde"),
        PrincipalGroup::Ketone => format!("{mp}one"),
        PrincipalGroup::Alcohol => format!("{mp}ol"),
        PrincipalGroup::Thiol => format!("{mp}thiol"),
        PrincipalGroup::Amine => format!("{mp}amine"),
        PrincipalGroup::Ester | PrincipalGroup::None => String::new(),
    };

    if omit_loc_1 {
        if suf.starts_with(['o', 'a', 'i', 'e']) && base_hydride.ends_with('e') {
            format!("{}{suf}", &base_hydride[..base_hydride.len() - 1])
        } else {
            format!("{base_hydride}{suf}")
        }
    } else {
        format!("{base_hydride}-{loc_s}-{suf}")
    }
}

/// Name an acyclic parent molecule using IUPAC 2013 P-44.3 / P-44.4 chain selection & lowest-locant numbering.
fn name_acyclic_parent(
    mol: &AnalytMol,
    stereo: &PerceivedStereo,
    senior_kind: PrincipalGroup,
    senior_hits: &[&FunctionalHit],
) -> Option<(String, HashMap<usize, usize>, bool)> {
    let adj = mol.adjacency();
    let rings = mol.sssr_rings();
    let ring_atoms: HashSet<usize> = rings.into_iter().flatten().collect();

    // Acyclic carbon atoms eligible for the principal chain
    let chain_carbons: Vec<usize> = (0..mol.atoms.len())
        .filter(|&i| mol.atoms[i].element == "C" && !ring_atoms.contains(&i))
        .collect();
    if chain_carbons.is_empty() {
        return None;
    }
    let cc_set: HashSet<usize> = chain_carbons.iter().copied().collect();

    // Enumerate all maximal simple paths across `cc_set` (tree/forest of acyclic carbons)
    let mut paths: Vec<Vec<usize>> = Vec::new();
    let endpoints: Vec<usize> = chain_carbons
        .iter()
        .copied()
        .filter(|&u| adj[u].iter().filter(|&&(v, _)| cc_set.contains(&v)).count() <= 1)
        .collect();

    fn dfs_paths(
        u: usize,
        prev: usize,
        cur: &mut Vec<usize>,
        adj: &[Vec<(usize, usize)>],
        cc_set: &HashSet<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        let nexts: Vec<usize> = adj[u]
            .iter()
            .map(|&(v, _)| v)
            .filter(|&v| v != prev && cc_set.contains(&v) && !cur.contains(&v))
            .collect();
        if nexts.is_empty() {
            out.push(cur.clone());
            return;
        }
        for v in nexts {
            cur.push(v);
            dfs_paths(v, u, cur, adj, cc_set, out);
            cur.pop();
        }
    }

    for &ep in &endpoints {
        let mut cur = vec![ep];
        dfs_paths(ep, usize::MAX, &mut cur, &adj, &cc_set, &mut paths);
    }
    if paths.is_empty() {
        paths.push(vec![chain_carbons[0]]);
    }

    // Score each directed chain candidate per IUPAC 2013 P-44.3 / P-44.4 + P-14.4 numbering:
    // Maximize: (1) count of senior principal groups on chain, (2) chain length (P-44.3: length BEFORE unsaturation in IUPAC 2013!),
    // (3) total multiple bonds, (4) double bonds, (5) number of substituents;
    // Then Minimize: (a) principal group locants, (b) multiple bond locants, (c) double bond locants,
    // (d) substituent locants, (e) alphabetical first substituent locants!
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct ChainScore {
        neg_fg_cnt: isize,
        neg_len: isize,
        neg_mult_bonds: isize,
        neg_dbl_bonds: isize,
        fg_locs: Vec<usize>,
        mult_locs: Vec<usize>,
        dbl_locs: Vec<usize>,
        neg_sub_cnt: isize,
        sub_locs: Vec<usize>,
        alpha_locs: Vec<String>,
    }

    let mut best: Option<(ChainScore, String, HashMap<usize, usize>, bool)> = None;

    for chain in &paths {
        let len = chain.len();
        let chain_set: HashSet<usize> = chain.iter().copied().collect();

        // Which senior hits lie on `chain`?
        // Note: Terminal groups (`CarboxylicAcid`, `AcidHalide`, `Amide`, `Nitrile`, `Aldehyde`) MUST have `group_atom` as an endpoint of `chain` to be named as `-oic acid`/`-al`/`-nitrile`!
        let chain_hits: Vec<&FunctionalHit> = senior_hits
            .iter()
            .copied()
            .filter(|h| {
                if matches!(
                    h.kind,
                    PrincipalGroup::CarboxylicAcid
                        | PrincipalGroup::AcidHalide
                        | PrincipalGroup::Amide
                        | PrincipalGroup::Nitrile
                        | PrincipalGroup::Aldehyde
                ) {
                    chain.first() == Some(&h.group_atom) || chain.last() == Some(&h.group_atom)
                } else {
                    chain_set.contains(&h.anchor_carbon)
                }
            })
            .collect();

        let mut atom_to_loc = HashMap::with_capacity(len);
        for (idx, &ca) in chain.iter().enumerate() {
            atom_to_loc.insert(ca, idx + 1);
        }

        let mut fg_locs: Vec<usize> = chain_hits
            .iter()
            .filter_map(|h| atom_to_loc.get(&h.anchor_carbon).copied())
            .collect();
        fg_locs.sort_unstable();

        let mut dbl_locs = Vec::new();
        let mut trp_locs = Vec::new();
        let mut mult_locs = Vec::new();
        for k in 0..len.saturating_sub(1) {
            if let Some(bi) = mol.bond_between(chain[k], chain[k + 1]) {
                match mol.bonds[bi].order {
                    2 => {
                        dbl_locs.push(k + 1);
                        mult_locs.push(k + 1);
                    }
                    3 => {
                        trp_locs.push(k + 1);
                        mult_locs.push(k + 1);
                    }
                    _ => {}
                }
            }
        }

        let mut blocked = chain_set.clone();
        for h in &chain_hits {
            for &ga in &h.group_atoms {
                blocked.insert(ga);
            }
        }

        let mut sub_items = Vec::new();
        let mut sub_locs = Vec::new();
        for (idx, &ca) in chain.iter().enumerate() {
            let loc = idx + 1;
            for &(nb, bi) in &adj[ca] {
                if !blocked.contains(&nb) {
                    let (sname, is_comp) =
                        name_substituent(mol, nb, ca, mol.bonds[bi].order, &blocked);
                    sub_locs.push(loc);
                    let loc_s = if len == 1
                        || (len == 2
                            && chain_hits.is_empty()
                            && dbl_locs.is_empty()
                            && trp_locs.is_empty()
                            && adj[chain[0]].len() + adj[chain[1]].len() == 3)
                    {
                        String::new()
                    } else {
                        loc.to_string()
                    };
                    sub_items.push(SubPrefixItem {
                        locant: loc_s,
                        base_name: sname,
                        is_compound: is_comp,
                    });
                }
            }
        }

        // Add N-substituents for acyclic amides / amines (`N,N-dimethylformamide`, `N-methylpropan-1-amine`)
        for h in &chain_hits {
            if h.kind == PrincipalGroup::Amide {
                if let Some(n_idx) = h.extra_atom {
                    for &(nb, bi) in &adj[n_idx] {
                        if nb != h.group_atom {
                            let mut n_blk = blocked.clone();
                            n_blk.insert(n_idx);
                            let (sname, is_comp) =
                                name_substituent(mol, nb, n_idx, mol.bonds[bi].order, &n_blk);
                            sub_items.push(SubPrefixItem {
                                locant: "N".to_string(),
                                base_name: sname,
                                is_compound: is_comp,
                            });
                        }
                    }
                }
            } else if h.kind == PrincipalGroup::Amine {
                let n_idx = h.group_atom;
                for &(nb, bi) in &adj[n_idx] {
                    if nb != h.anchor_carbon {
                        let mut n_blk = blocked.clone();
                        n_blk.insert(n_idx);
                        let (sname, is_comp) =
                            name_substituent(mol, nb, n_idx, mol.bonds[bi].order, &n_blk);
                        sub_items.push(SubPrefixItem {
                            locant: "N".to_string(),
                            base_name: sname,
                            is_compound: is_comp,
                        });
                    }
                }
            }
        }

        sub_locs.sort_unstable();
        let mut alpha_pairs: Vec<(String, usize)> = sub_items
            .iter()
            .map(|it| {
                (
                    iupac_alpha_key(&it.base_name),
                    it.locant.parse::<usize>().unwrap_or(0),
                )
            })
            .collect();
        alpha_pairs.sort();
        let alpha_locs: Vec<String> = alpha_pairs
            .into_iter()
            .map(|(k, l)| format!("{k}:{l}"))
            .collect();

        let has_subs = !sub_items.is_empty();
        let sub_pref = assemble_prefixes(&sub_items, false);
        let stem = alkane_stem(len);
        let core_suf = build_acyclic_core_and_suffix(
            mol,
            &stem,
            len,
            &dbl_locs,
            &trp_locs,
            senior_kind,
            &chain_hits,
            &fg_locs,
            has_subs,
        );

        let body = if sub_pref.is_empty() {
            core_suf
        } else {
            format!("{sub_pref}{core_suf}")
        };
        let st_pref = format_stereo_prefix(mol, stereo, &atom_to_loc, len);
        let full_name = format!("{st_pref}{body}");

        let sc = ChainScore {
            neg_fg_cnt: -(chain_hits.len() as isize),
            neg_len: -(len as isize),
            neg_mult_bonds: -(mult_locs.len() as isize),
            neg_dbl_bonds: -(dbl_locs.len() as isize),
            fg_locs,
            mult_locs,
            dbl_locs,
            neg_sub_cnt: -(sub_items.len() as isize),
            sub_locs,
            alpha_locs,
        };

        if best.as_ref().is_none_or(|b| sc < b.0) {
            best = Some((sc, full_name, atom_to_loc, has_subs));
        }
    }

    best.map(|(_, nm, map, hs)| (nm, map, hs))
}

#[allow(clippy::too_many_arguments)]
fn build_acyclic_core_and_suffix(
    mol: &AnalytMol,
    stem: &str,
    len: usize,
    dbl_locs: &[usize],
    trp_locs: &[usize],
    senior_kind: PrincipalGroup,
    chain_hits: &[&FunctionalHit],
    fg_locs: &[usize],
    has_subs: bool,
) -> String {
    if fg_locs.is_empty() {
        return format_unsat_infix(stem, dbl_locs, trp_locs, true);
    }

    let cnt = fg_locs.len();
    let mp = simple_mult_prefix(cnt);
    let loc_s = fg_locs
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(",");

    // Terminal chain-ending suffixes (`-oic acid`, `-dioic acid`, `-oyl chloride`, `-amide`, `-nitrile`, `-al`)
    // omit locant `1` (and `1,n` for `dioic acid` / `dial` / `dinitrile`) per IUPAC P-14.3.4!
    match senior_kind {
        PrincipalGroup::CarboxylicAcid => {
            let keep_e = cnt >= 2; // `butanedioic acid` keeps `'e'`, `propanoic acid` drops `'e'`
            let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, keep_e);
            return format!("{unsat}{mp}oic acid");
        }
        PrincipalGroup::AcidHalide => {
            let hal = chain_hits[0]
                .extra_atom
                .map(|x| mol.atoms[x].element.as_str())
                .unwrap_or("Cl");
            let hword = match hal {
                "F" => "fluoride",
                "Br" => "bromide",
                "I" => "iodide",
                _ => "chloride",
            };
            let keep_e = cnt >= 2;
            let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, keep_e);
            return format!("{unsat}{mp}oyl {hword}");
        }
        PrincipalGroup::Amide => {
            let keep_e = cnt >= 2;
            let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, keep_e);
            return format!("{unsat}{mp}amide");
        }
        PrincipalGroup::Nitrile => {
            // `ethanenitrile` / `butanedinitrile` BOTH keep `'e'` before `'n'`/`'d'`!
            let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, true);
            return format!("{unsat}{mp}nitrile");
        }
        PrincipalGroup::Aldehyde => {
            let keep_e = cnt >= 2; // `ethanal` drops `'e'`, `ethanedial` keeps `'e'`
            let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, keep_e);
            return format!("{unsat}{mp}al");
        }
        _ => {}
    }

    // Non-terminal / heteroatom suffixes (`-one`, `-ol`, `-thiol`, `-amine`, `-sulfonic acid`)
    let suf = match senior_kind {
        PrincipalGroup::SulfonicAcid => format!("{mp}sulfonic acid"),
        PrincipalGroup::Ketone => format!("{mp}one"),
        PrincipalGroup::Alcohol => format!("{mp}ol"),
        PrincipalGroup::Thiol => format!("{mp}thiol"),
        PrincipalGroup::Amine => format!("{mp}amine"),
        _ => String::new(),
    };

    let starts_with_vowel = suf.starts_with(['a', 'e', 'i', 'o', 'u', 'y']);
    let unsat = format_unsat_infix(stem, dbl_locs, trp_locs, !starts_with_vowel);

    // IUPAC P-14.3.4.2: omit locant `"1"` ONLY for unsubstituted C1/C2 (`methanol`, `ethanol`, `ethanamine`, `ethanethiol`)
    // or `propan-2-one`? In IUPAC 2013 PIN, `propan-2-one` and `1-(4-bromophenyl)ethan-1-one` KEEP locants,
    // while `methanol`, `ethanol`, `ethanethiol`, `ethanamine`, `methanamine` omit `"1"` when `len <= 2 && !has_subs && dbl_locs.is_empty() && trp_locs.is_empty() && cnt == 1`!
    let omit_loc = cnt == 1
        && fg_locs == [1]
        && len <= 2
        && !has_subs
        && dbl_locs.is_empty()
        && trp_locs.is_empty()
        && matches!(
            senior_kind,
            PrincipalGroup::Alcohol | PrincipalGroup::Thiol | PrincipalGroup::Amine
        );

    if omit_loc {
        format!("{unsat}{suf}")
    } else {
        format!("{unsat}-{loc_s}-{suf}")
    }
}

/// Convert a chemical graph (`AnalytMol`) offline into its Preferred IUPAC Name (PIN).
pub fn mol_to_iupac_name(raw_mol: &AnalytMol) -> Result<String, String> {
    if raw_mol.atoms.is_empty() {
        return Err("Cannot generate IUPAC name for an empty molecule".to_string());
    }
    let norm = normalize_for_naming(raw_mol);
    let mol = &norm.mol;
    let stereo = &norm.stereo;

    // Split disconnected components (e.g. salts, hydrates)
    let adj = mol.adjacency();
    let mut vis = vec![false; mol.atoms.len()];
    let mut comps: Vec<HashSet<usize>> = Vec::new();
    for i in 0..mol.atoms.len() {
        if vis[i] {
            continue;
        }
        vis[i] = true;
        let mut cset = HashSet::new();
        cset.insert(i);
        let mut q = VecDeque::new();
        q.push_back(i);
        while let Some(u) = q.pop_front() {
            for &(v, _) in &adj[u] {
                if !vis[v] {
                    vis[v] = true;
                    cset.insert(v);
                    q.push_back(v);
                }
            }
        }
        comps.push(cset);
    }

    if comps.len() == 1 {
        return name_single_with_locants(mol, stereo, None)
            .map(|(nm, _, _)| nm)
            .ok_or_else(|| "Unsupported molecular topology for offline IUPAC naming".to_string());
    }

    // Multi-component naming (carboxylate salts `sodium ethanoate`, `... hydrochloride`, hydrates)
    let mut cations = Vec::new();
    let mut organics = Vec::new();
    let mut anions = Vec::new();
    let mut hydrates = 0usize;

    for cset in &comps {
        if cset.len() == 1 {
            let idx = *cset.iter().next().unwrap();
            let a = &mol.atoms[idx];
            match (a.element.as_str(), mol.total_h_count(idx), a.formal_charge) {
                ("O", 2, 0) => {
                    hydrates += 1;
                    continue;
                }
                ("Na", 0, 1) => {
                    cations.push("sodium".to_string());
                    continue;
                }
                ("K", 0, 1) => {
                    cations.push("potassium".to_string());
                    continue;
                }
                ("Li", 0, 1) => {
                    cations.push("lithium".to_string());
                    continue;
                }
                ("N", 4, 1) => {
                    cations.push("ammonium".to_string());
                    continue;
                }
                ("Cl", 1, 0) => {
                    anions.push("hydrochloride".to_string());
                    continue;
                }
                ("Br", 1, 0) => {
                    anions.push("hydrobromide".to_string());
                    continue;
                }
                ("Cl", 0, -1) => {
                    anions.push("chloride".to_string());
                    continue;
                }
                ("Br", 0, -1) => {
                    anions.push("bromide".to_string());
                    continue;
                }
                ("I", 0, -1) => {
                    anions.push("iodide".to_string());
                    continue;
                }
                _ => {}
            }
        }
        let (mut sub, _, _) = induce_submolecule(mol, stereo, cset);
        // Check carboxylate anion `-C(=O)[O-]` -> neutralize `[O-]` to `-OH` and convert `-ic acid` -> `-ate`!
        let mut had_carboxylate = false;
        for a in &mut sub.atoms {
            if a.element == "O" && a.formal_charge == -1 {
                a.formal_charge = 0;
                a.implicit_h = Some(1);
                had_carboxylate = true;
            }
        }
        let sub_st = sub.perceive_cip_stereo();
        if let Some((nm, _, _)) = name_single_with_locants(&sub, &sub_st, None) {
            if had_carboxylate {
                if let Some(stem) = nm.strip_suffix("ic acid") {
                    anions.push(format!("{stem}ate"));
                } else {
                    anions.push(nm);
                }
            } else {
                organics.push(nm);
            }
        }
    }

    let mut parts = Vec::new();
    parts.extend(cations);
    parts.extend(organics);
    parts.extend(anions);
    if hydrates > 0 {
        let hp = match hydrates {
            1 => "hydrate".to_string(),
            _ => format!("{}hydrate", simple_mult_prefix(hydrates)),
        };
        parts.push(hp);
    }
    if parts.is_empty() {
        Err("Unable to name multi-component structure".to_string())
    } else {
        Ok(parts.join(" "))
    }
}

// ============================================================================
// Part 3A: Built-in Reagent/Drug/Amino-Acid Dictionary & Substituent Graph Builder
// ============================================================================

/// Lookup SMILES in the offline dictionary of common reagents, solvents, catalysts,
/// amino acids, nucleobases, and benchmark pharmaceuticals (`names.ts`, `dictionary.ts`, `drugs.ts`).
pub fn lookup_common_or_drug_smiles(raw: &str) -> Option<&'static str> {
    let key: String = raw
        .trim()
        .chars()
        .map(|c| c.to_ascii_lowercase())
        .collect::<String>()
        .replace(['–', '—'], "-");
    let k = key.as_str();

    match k {
        // Solvents & everyday organics
        "water" | "oxidane" | "h2o" => Some("O"),
        "heavy water" | "d2o" | "deuterium oxide" => Some("[2H]O[2H]"),
        "ammonia" | "azane" | "nh3" => Some("N"),
        "ammonium" => Some("[NH4+]"),
        "phosphane" | "phosphine" => Some("P"),
        "silane" => Some("[SiH4]"),
        "disilane" => Some("[SiH3][SiH3]"),
        "borane" => Some("B"),
        "sulfane" | "hydrogen sulfide" | "h2s" => Some("S"),
        "disulfane" => Some("SS"),
        "diphosphane" => Some("PP"),
        "hydrazine" | "diazane" => Some("NN"),
        "diazene" => Some("N=N"),
        "hydrogen peroxide" | "dioxidane" | "h2o2" => Some("OO"),
        "carbon dioxide" | "co2" => Some("O=C=O"),
        "sulfur dioxide" | "so2" => Some("O=S=O"),
        "sulfuric acid" | "h2so4" => Some("OS(=O)(=O)O"),
        "nitric acid" | "hno3" => Some("O[N+](=O)[O-]"),
        "phosphoric acid" | "h3po4" => Some("OP(=O)(O)O"),
        "hydrochloric acid" | "hydrogen chloride" | "chlorane" | "hcl" => Some("Cl"),
        "hydrobromic acid" | "hydrogen bromide" | "bromane" | "hbr" => Some("Br"),
        "hydroiodic acid" | "hydrogen iodide" | "iodane" | "hi" => Some("I"),
        "hydrofluoric acid" | "hydrogen fluoride" | "fluorane" | "hf" => Some("F"),
        "methanol" | "meoh" | "methyl alcohol" => Some("CO"),
        "ethanol" | "etoh" | "ethyl alcohol" => Some("CCO"),
        "isopropanol" | "ipa" | "2-propanol" | "isopropyl alcohol" => Some("CC(C)O"),
        "tert-butanol" | "t-buoh" | "tbuoh" => Some("CC(C)(C)O"),
        "acetone" | "propan-2-one" | "2-propanone" => Some("CC(=O)C"),
        "butanone" | "mek" | "2-butanone" => Some("CCC(=O)C"),
        "acetonitrile" | "mecn" | "acn" | "ethanenitrile" => Some("CC#N"),
        "dimethyl sulfoxide" | "dmso" => Some("CS(=O)C"),
        "dimethylformamide" | "n,n-dimethylformamide" | "dmf" => Some("CN(C)C=O"),
        "dimethylacetamide" | "n,n-dimethylacetamide" | "dmac" => Some("CC(=O)N(C)C"),
        "n-methyl-2-pyrrolidone" | "nmp" => Some("CN1CCCC1=O"),
        "tetrahydrofuran" | "thf" | "oxolane" => Some("C1CCOC1"),
        "1,4-dioxane" | "dioxane" => Some("C1COCCO1"),
        "diethyl ether" | "et2o" | "ether" | "ethoxyethane" => Some("CCOCC"),
        "methyl tert-butyl ether" | "mtbe" | "tbme" => Some("COC(C)(C)C"),
        "dichloromethane" | "dcm" | "methylene chloride" => Some("ClCCl"),
        "chloroform" | "trichloromethane" | "chcl3" => Some("ClC(Cl)Cl"),
        "carbon tetrachloride" | "tetrachloromethane" | "ccl4" => Some("ClC(Cl)(Cl)Cl"),
        "ethyl acetate" | "etoac" | "ea" => Some("CC(=O)OCC"),
        "methyl acetate" | "meoac" => Some("CC(=O)OC"),
        "acetic acid" | "hoac" | "acoh" | "ethanoic acid" => Some("CC(=O)O"),
        "formic acid" | "methanoic acid" => Some("O=CO"),
        "trifluoroacetic acid" | "tfa" => Some("OC(=O)C(F)(F)F"),
        "acetic anhydride" | "ac2o" => Some("CC(=O)OC(=O)C"),
        "boc anhydride" | "di-tert-butyl dicarbonate" | "boc2o" => {
            Some("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
        }
        "triethylamine" | "et3n" | "tea" => Some("CCN(CC)CC"),
        "diisopropylethylamine" | "dipea" | "hunig's base" => Some("CCN(C(C)C)C(C)C"),
        "4-dimethylaminopyridine" | "dmap" => Some("CN(C)c1ccncc1"),
        "dbu" => Some("C1CCN2CCCCC2=NC1"),
        "sodium borohydride" | "nabh4" => Some("[Na+].[BH4-]"),
        "lithium aluminium hydride" | "lialh4" | "lah" => Some("[Li+].[AlH4-]"),
        "sodium hydroxide" | "naoh" => Some("[Na+].[OH-]"),
        "potassium hydroxide" | "koh" => Some("[K+].[OH-]"),
        "sodium acetate" | "naoac" => Some("[Na+].CC(=O)[O-]"),
        // Aromatics & common building blocks
        "benzene" => Some("c1ccccc1"),
        "toluene" | "methylbenzene" => Some("Cc1ccccc1"),
        "o-xylene" | "1,2-dimethylbenzene" => Some("Cc1ccccc1C"),
        "m-xylene" | "1,3-dimethylbenzene" => Some("Cc1cccc(C)c1"),
        "p-xylene" | "1,4-dimethylbenzene" => Some("Cc1ccc(C)cc1"),
        "styrene" | "ethenylbenzene" | "vinylbenzene" => Some("C=Cc1ccccc1"),
        "phenol" => Some("Oc1ccccc1"),
        "anisole" | "methoxybenzene" => Some("COc1ccccc1"),
        "aniline" => Some("Nc1ccccc1"),
        "benzaldehyde" => Some("O=Cc1ccccc1"),
        "benzoic acid" => Some("O=C(O)c1ccccc1"),
        "benzonitrile" => Some("N#Cc1ccccc1"),
        "benzamide" => Some("NC(=O)c1ccccc1"),
        "acetophenone" | "1-phenylethan-1-one" => Some("CC(=O)c1ccccc1"),
        "4-bromoacetophenone" | "p-bromoacetophenone" => Some("CC(=O)c1ccc(Br)cc1"),
        "benzophenone" | "diphenylmethanone" => Some("O=C(c1ccccc1)c2ccccc2"),
        "nitrobenzene" => Some("O=[N+]([O-])c1ccccc1"),
        "ferrocene" => Some("[Fe]123456789(C%10C1C2C3C4%10)C%11C5C6C7C89%11"),
        // Bridged / cage / spiro benchmarks
        "norbornane" => Some("C1CC2CCC1C2"),
        "norbornene" => Some("C1=CC2CCC1C2"),
        "adamantane" | "tricyclo[3.3.1.1³,⁷]decane" | "tricyclo[3.3.1.13,7]decane" => {
            Some("C1C2CC3CC1CC(C2)C3")
        }
        "cubane" => Some("C12C3C4C1C5C2C3C45"),
        // Benchmark pharmaceuticals & natural products
        "ibuprofen" => Some("CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        "naproxen" => Some("COc1ccc2cc([C@H](C)C(=O)O)ccc2c1"),
        "aspirin" | "acetylsalicylic acid" => Some("CC(=O)Oc1ccccc1C(=O)O"),
        "paracetamol" | "acetaminophen" => Some("CC(=O)Nc1ccc(O)cc1"),
        "caffeine" => Some("Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        "nicotine" => Some("CN1CCC[C@H]1c2cccnc2"),
        "dopamine" => Some("NCCc1ccc(O)c(O)c1"),
        "serotonin" => Some("NCCc1c[nH]c2ccc(O)cc12"),
        "cholesterol" => {
            Some("CC(C)CCCC(C)[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]3[C@H]2CC=C4[C@@]3(C)CC[C@@H](O)C4")
        }
        // 20 Standard L-amino acids
        "glycine" | "gly" => Some("NCC(=O)O"),
        "l-alanine" | "alanine" | "ala" => Some("N[C@@H](C)C(=O)O"),
        "d-alanine" => Some("N[C@H](C)C(=O)O"),
        "l-valine" | "valine" | "val" => Some("CC(C)[C@H](N)C(=O)O"),
        "l-leucine" | "leucine" | "leu" => Some("CC(C)C[C@H](N)C(=O)O"),
        "l-isoleucine" | "isoleucine" | "ile" => Some("CC[C@H](C)[C@H](N)C(=O)O"),
        "l-proline" | "proline" | "pro" => Some("O=C(O)[C@@H]1CCCN1"),
        "l-phenylalanine" | "phenylalanine" | "phe" => Some("N[C@@H](Cc1ccccc1)C(=O)O"),
        "l-tyrosine" | "tyrosine" | "tyr" => Some("N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        "l-tryptophan" | "tryptophan" | "trp" => Some("N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
        "l-serine" | "serine" | "ser" => Some("N[C@@H](CO)C(=O)O"),
        "l-threonine" | "threonine" | "thr" => Some("C[C@@H](O)[C@H](N)C(=O)O"),
        "l-cysteine" | "cysteine" | "cys" => Some("N[C@@H](CS)C(=O)O"),
        "l-methionine" | "methionine" | "met" => Some("CSCC[C@H](N)C(=O)O"),
        "l-aspartic acid" | "aspartic acid" | "asp" => Some("N[C@@H](CC(=O)O)C(=O)O"),
        "l-glutamic acid" | "glutamic acid" | "glu" => Some("N[C@@H](CCC(=O)O)C(=O)O"),
        "l-asparagine" | "asparagine" | "asn" => Some("NC(=O)C[C@H](N)C(=O)O"),
        "l-glutamine" | "glutamine" | "gln" => Some("NC(=O)CC[C@H](N)C(=O)O"),
        "l-lysine" | "lysine" | "lys" => Some("NCCCC[C@H](N)C(=O)O"),
        "l-arginine" | "arginine" | "arg" => Some("NC(=N)NCCC[C@H](N)C(=O)O"),
        "l-histidine" | "histidine" | "his" => Some("N[C@@H](Cc1c[nH]cn1)C(=O)O"),
        _ => None,
    }
}

/// Attach a parsed substituent radical (`"methyl"`, `"4-bromophenyl"`, `"2-methylpropyl"`, `"amino"`,
/// `"hydroxy"`, `"oxo"`, `"nitro"`, `"methoxy"`, etc.) onto `anchor_atom` of `mol`.
fn attach_substituent_by_name(
    mol: &mut AnalytMol,
    anchor_atom: usize,
    raw_sub: &str,
) -> Result<(), String> {
    let sub = strip_outer_brackets(raw_sub.trim());
    let s = sub.to_ascii_lowercase();

    match s.as_str() {
        "fluoro" => {
            let f = mol.add_atom(AnalytAtom::new("F"));
            mol.add_bond(AnalytBond::single(anchor_atom, f));
            return Ok(());
        }
        "chloro" => {
            let cl = mol.add_atom(AnalytAtom::new("Cl"));
            mol.add_bond(AnalytBond::single(anchor_atom, cl));
            return Ok(());
        }
        "bromo" => {
            let br = mol.add_atom(AnalytAtom::new("Br"));
            mol.add_bond(AnalytBond::single(anchor_atom, br));
            return Ok(());
        }
        "iodo" => {
            let io = mol.add_atom(AnalytAtom::new("I"));
            mol.add_bond(AnalytBond::single(anchor_atom, io));
            return Ok(());
        }
        "hydroxy" | "hydroxyl" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            return Ok(());
        }
        "oxo" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::double(anchor_atom, o));
            return Ok(());
        }
        "sulfanyl" | "mercapto" => {
            let sv = mol.add_atom(AnalytAtom::new("S"));
            mol.add_bond(AnalytBond::single(anchor_atom, sv));
            return Ok(());
        }
        "sulfanylidene" | "thioxo" => {
            let sv = mol.add_atom(AnalytAtom::new("S"));
            mol.add_bond(AnalytBond::double(anchor_atom, sv));
            return Ok(());
        }
        "amino" => {
            let n = mol.add_atom(AnalytAtom::new("N"));
            mol.add_bond(AnalytBond::single(anchor_atom, n));
            return Ok(());
        }
        "imino" => {
            let n = mol.add_atom(AnalytAtom::new("N"));
            mol.add_bond(AnalytBond::double(anchor_atom, n));
            return Ok(());
        }
        "cyano" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let n = mol.add_atom(AnalytAtom::new("N"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::triple(c, n));
            return Ok(());
        }
        "formyl" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::double(c, o));
            return Ok(());
        }
        "carboxy" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let o1 = mol.add_atom(AnalytAtom::new("O"));
            let o2 = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::double(c, o1));
            mol.add_bond(AnalytBond::single(c, o2));
            return Ok(());
        }
        "carbamoyl" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let o = mol.add_atom(AnalytAtom::new("O"));
            let n = mol.add_atom(AnalytAtom::new("N"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::double(c, o));
            mol.add_bond(AnalytBond::single(c, n));
            return Ok(());
        }
        "acetyl" | "ethanoyl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let o = mol.add_atom(AnalytAtom::new("O"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::double(c1, o));
            mol.add_bond(AnalytBond::single(c1, c2));
            return Ok(());
        }
        "benzoyl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::double(c1, o));
            return attach_substituent_by_name(mol, c1, "phenyl");
        }
        "nitro" => {
            let n = mol.add_atom(AnalytAtom::new("N").with_charge(1));
            let o1 = mol.add_atom(AnalytAtom::new("O"));
            let o2 = mol.add_atom(AnalytAtom::new("O").with_charge(-1));
            mol.add_bond(AnalytBond::single(anchor_atom, n));
            mol.add_bond(AnalytBond::double(n, o1));
            mol.add_bond(AnalytBond::single(n, o2));
            return Ok(());
        }
        "nitroso" => {
            let n = mol.add_atom(AnalytAtom::new("N"));
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, n));
            mol.add_bond(AnalytBond::double(n, o));
            return Ok(());
        }
        "azido" => {
            let n1 = mol.add_atom(AnalytAtom::new("N"));
            let n2 = mol.add_atom(AnalytAtom::new("N").with_charge(1));
            let n3 = mol.add_atom(AnalytAtom::new("N").with_charge(-1));
            mol.add_bond(AnalytBond::single(anchor_atom, n1));
            mol.add_bond(AnalytBond::double(n1, n2));
            mol.add_bond(AnalytBond::double(n2, n3));
            return Ok(());
        }
        "sulfo" => {
            let sv = mol.add_atom(AnalytAtom::new("S"));
            let o1 = mol.add_atom(AnalytAtom::new("O"));
            let o2 = mol.add_atom(AnalytAtom::new("O"));
            let oh = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, sv));
            mol.add_bond(AnalytBond::double(sv, o1));
            mol.add_bond(AnalytBond::double(sv, o2));
            mol.add_bond(AnalytBond::single(sv, oh));
            return Ok(());
        }
        "methoxy" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            let c = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            mol.add_bond(AnalytBond::single(o, c));
            return Ok(());
        }
        "ethoxy" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            mol.add_bond(AnalytBond::single(o, c1));
            mol.add_bond(AnalytBond::single(c1, c2));
            return Ok(());
        }
        "propoxy" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            return attach_substituent_by_name(mol, o, "propyl");
        }
        "butoxy" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            return attach_substituent_by_name(mol, o, "butyl");
        }
        "phenoxy" => {
            let o = mol.add_atom(AnalytAtom::new("O"));
            mol.add_bond(AnalytBond::single(anchor_atom, o));
            return attach_substituent_by_name(mol, o, "phenyl");
        }
        "isopropyl" | "propan-2-yl" | "1-methylethyl" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let m1 = mol.add_atom(AnalytAtom::new("C"));
            let m2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::single(c, m1));
            mol.add_bond(AnalytBond::single(c, m2));
            return Ok(());
        }
        "isobutyl" | "2-methylpropyl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let c3 = mol.add_atom(AnalytAtom::new("C"));
            let m = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::single(c1, c2));
            mol.add_bond(AnalytBond::single(c2, c3));
            mol.add_bond(AnalytBond::single(c2, m));
            return Ok(());
        }
        "sec-butyl" | "butan-2-yl" | "1-methylpropyl" => {
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c3 = mol.add_atom(AnalytAtom::new("C"));
            let c4 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c2));
            mol.add_bond(AnalytBond::single(c2, c1));
            mol.add_bond(AnalytBond::single(c2, c3));
            mol.add_bond(AnalytBond::single(c3, c4));
            return Ok(());
        }
        "tert-butyl" | "t-butyl" | "2-methylpropan-2-yl" | "1,1-dimethylethyl" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            let m1 = mol.add_atom(AnalytAtom::new("C"));
            let m2 = mol.add_atom(AnalytAtom::new("C"));
            let m3 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c));
            mol.add_bond(AnalytBond::single(c, m1));
            mol.add_bond(AnalytBond::single(c, m2));
            mol.add_bond(AnalytBond::single(c, m3));
            return Ok(());
        }
        "vinyl" | "ethenyl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::double(c1, c2));
            return Ok(());
        }
        "ethynyl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::triple(c1, c2));
            return Ok(());
        }
        "allyl" | "prop-2-en-1-yl" => {
            let c1 = mol.add_atom(AnalytAtom::new("C"));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let c3 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, c1));
            mol.add_bond(AnalytBond::single(c1, c2));
            mol.add_bond(AnalytBond::double(c2, c3));
            return Ok(());
        }
        "methylidene" => {
            let c = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::double(anchor_atom, c));
            return Ok(());
        }
        "benzyl" | "phenylmethyl" => {
            let ch2 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::single(anchor_atom, ch2));
            return attach_substituent_by_name(mol, ch2, "phenyl");
        }
        "trimethylsilyl" => {
            let si = mol.add_atom(AnalytAtom::new("Si"));
            mol.add_bond(AnalytBond::single(anchor_atom, si));
            for _ in 0..3 {
                let c = mol.add_atom(AnalytAtom::new("C"));
                mol.add_bond(AnalytBond::single(si, c));
            }
            return Ok(());
        }
        _ => {}
    }

    // Compound `-oxy`, `-oxycarbonyl`, `-sulfonyl`, `-sulfanyl`, `-amino`
    if let Some(rad) = s.strip_suffix("oxycarbonyl") {
        let c = mol.add_atom(AnalytAtom::new("C"));
        let o_dbl = mol.add_atom(AnalytAtom::new("O"));
        let o_est = mol.add_atom(AnalytAtom::new("O"));
        mol.add_bond(AnalytBond::single(anchor_atom, c));
        mol.add_bond(AnalytBond::double(c, o_dbl));
        mol.add_bond(AnalytBond::single(c, o_est));
        return attach_substituent_by_name(mol, o_est, rad);
    }
    if let Some(rad) = s.strip_suffix("oxy") {
        let o = mol.add_atom(AnalytAtom::new("O"));
        mol.add_bond(AnalytBond::single(anchor_atom, o));
        return attach_substituent_by_name(mol, o, rad);
    }
    if let Some(rad) = s.strip_suffix("sulfonyl") {
        let sv = mol.add_atom(AnalytAtom::new("S"));
        let o1 = mol.add_atom(AnalytAtom::new("O"));
        let o2 = mol.add_atom(AnalytAtom::new("O"));
        mol.add_bond(AnalytBond::single(anchor_atom, sv));
        mol.add_bond(AnalytBond::double(sv, o1));
        mol.add_bond(AnalytBond::double(sv, o2));
        return attach_substituent_by_name(mol, sv, rad);
    }
    if let Some(rad) = s.strip_suffix("sulfanyl") {
        let sv = mol.add_atom(AnalytAtom::new("S"));
        mol.add_bond(AnalytBond::single(anchor_atom, sv));
        return attach_substituent_by_name(mol, sv, rad);
    }
    if let Some(rad) = s.strip_suffix("amino") {
        let n = mol.add_atom(AnalytAtom::new("N"));
        mol.add_bond(AnalytBond::single(anchor_atom, n));
        if let Some(di_rad) = rad.strip_prefix("di") {
            if stem_to_length(di_rad.strip_suffix("yl").unwrap_or("")).is_some()
                || di_rad == "phenyl"
            {
                attach_substituent_by_name(mol, n, di_rad)?;
                attach_substituent_by_name(mol, n, di_rad)?;
                return Ok(());
            }
        }
        return attach_substituent_by_name(mol, n, rad);
    }

    // Substituted or unsubstituted `phenyl` (`"phenyl"`, `"4-bromophenyl"`, `"4-(2-methylpropyl)phenyl"`)
    if let Some(pref) = s.strip_suffix("phenyl") {
        let mut ring = Vec::with_capacity(6);
        for _ in 0..6 {
            ring.push(mol.add_atom(AnalytAtom::new("C")));
        }
        for k in 0..6 {
            let ord = if k % 2 == 0 { 2 } else { 1 };
            mol.add_bond(AnalytBond::aromatic_bond(ring[k], ring[(k + 1) % 6], ord));
        }
        mol.add_bond(AnalytBond::single(anchor_atom, ring[0]));
        if !pref.is_empty() {
            let mut loc_map = HashMap::new();
            for (idx, &ra) in ring.iter().enumerate() {
                loc_map.insert((idx + 1).to_string(), ra);
            }
            apply_prefix_substituents(mol, pref, &loc_map)?;
        }
        return Ok(());
    }

    // Unsubstituted `cycloalkyl` (`cyclopropyl`..`cyclooctyl`)
    if let Some(rest) = s.strip_prefix("cyclo") {
        if let Some(stem) = rest.strip_suffix("yl") {
            if let Some(k) = stem_to_length(stem) {
                if k >= 3 {
                    let mut ring = Vec::with_capacity(k);
                    for _ in 0..k {
                        ring.push(mol.add_atom(AnalytAtom::new("C")));
                    }
                    for m in 0..k {
                        mol.add_bond(AnalytBond::single(ring[m], ring[(m + 1) % k]));
                    }
                    mol.add_bond(AnalytBond::single(anchor_atom, ring[0]));
                    return Ok(());
                }
            }
        }
    }

    // Internal-attachment `alkan-k-yl` (`propan-2-yl`, `pentan-3-yl`)
    if let Some(stem_and_loc) = s.strip_suffix("-yl") {
        if let Some(dash_pos) = stem_and_loc.rfind('-') {
            let left = &stem_and_loc[..dash_pos];
            let loc_str = &stem_and_loc[dash_pos + 1..];
            if let (Some(stem), Ok(loc_k)) = (left.strip_suffix("an"), loc_str.parse::<usize>()) {
                if let Some(len) = stem_to_length(stem) {
                    if loc_k >= 1 && loc_k <= len {
                        let mut ch = Vec::with_capacity(len);
                        for _ in 0..len {
                            ch.push(mol.add_atom(AnalytAtom::new("C")));
                        }
                        for m in 0..len.saturating_sub(1) {
                            mol.add_bond(AnalytBond::single(ch[m], ch[m + 1]));
                        }
                        mol.add_bond(AnalytBond::single(anchor_atom, ch[loc_k - 1]));
                        return Ok(());
                    }
                }
            }
        }
    }

    // General linear or substituted `<prefixes><stem>yl` (`methyl`, `ethyl`, `2-methylpropyl`, `trifluoromethyl`)
    if let Some(before_yl) = s.strip_suffix("yl") {
        // Sort candidate stems by descending string length so `"meth"` (len 4) is checked before `"eth"` (len 3)!
        let mut stem_cands: Vec<(usize, String)> = (1..=20).map(|n| (n, alkane_stem(n))).collect();
        stem_cands.sort_by_key(|b| std::cmp::Reverse(b.1.len()));
        for (n, st) in stem_cands {
            if let Some(pref) = before_yl.strip_suffix(&st) {
                let mut ch = Vec::with_capacity(n);
                for _ in 0..n {
                    ch.push(mol.add_atom(AnalytAtom::new("C")));
                }
                for m in 0..n.saturating_sub(1) {
                    mol.add_bond(AnalytBond::single(ch[m], ch[m + 1]));
                }
                mol.add_bond(AnalytBond::single(anchor_atom, ch[0]));
                if !pref.is_empty() {
                    let mut loc_map = HashMap::new();
                    for (idx, &ca) in ch.iter().enumerate() {
                        loc_map.insert((idx + 1).to_string(), ca);
                    }
                    loc_map.insert(String::new(), ch[0]);
                    apply_prefix_substituents(mol, pref, &loc_map)?;
                }
                return Ok(());
            }
        }
    }

    Err(format!(
        "Unrecognized IUPAC substituent radical '{raw_sub}'"
    ))
}

fn strip_outer_brackets(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let open = bytes[0] as char;
        let close = bytes[bytes.len() - 1] as char;
        if (open == '(' && close == ')') || (open == '[' && close == ']') {
            let mut depth = 0i32;
            let mut wraps_all = true;
            for (i, ch) in s.char_indices() {
                if ch == '(' || ch == '[' {
                    depth += 1;
                } else if ch == ')' || ch == ']' {
                    depth -= 1;
                    if depth == 0 && i + ch.len_utf8() < s.len() {
                        wraps_all = false;
                        break;
                    }
                }
            }
            if wraps_all && depth == 0 {
                return &s[1..s.len() - 1];
            }
        }
    }
    s
}

// ============================================================================
// Part 3B: Systematic Compositional IUPAC Name -> Structure Parser
// ============================================================================

fn split_multiplier(body: &str) -> (usize, &str) {
    // Check complex multipliers followed by bracket (`bis(...)`, `tris[...]`, etc.)
    for &(mp, cnt) in &[
        ("hexakis", 6usize),
        ("pentakis", 5),
        ("tetrakis", 4),
        ("tris", 3),
        ("bis", 2),
    ] {
        if let Some(rest) = body.strip_prefix(mp) {
            if rest.starts_with('(') || rest.starts_with('[') {
                return (cnt, rest);
            }
        }
    }
    // Simple multipliers (`di`, `tri`, `tetra`, ...), avoiding false positives like `"phenyl"` or `"propyl"`
    for &(mp, cnt) in &[
        ("deca", 10usize),
        ("nona", 9),
        ("octa", 8),
        ("hepta", 7),
        ("hexa", 6),
        ("penta", 5),
        ("tetra", 4),
        ("tri", 3),
        ("di", 2),
    ] {
        if let Some(rest) = body.strip_prefix(mp) {
            if !rest.is_empty() && rest != "azo" && !rest.starts_with("methylamino") {
                return (cnt, rest);
            }
        }
    }
    (1, body)
}

fn looks_like_locant_list(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.split(',').all(|tok| {
        let t = tok.trim_end_matches('\'');
        t.eq_ignore_ascii_case("N")
            || t.eq_ignore_ascii_case("O")
            || t.eq_ignore_ascii_case("S")
            || (!t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
    })
}

/// Parse and attach a sequence of IUPAC substituent prefixes (`"2-methyl"`, `"3-bromo-4-chloro"`,
/// `"2-[4-(2-methylpropyl)phenyl]"`, `"N,N-dimethyl"`) onto `mol` using `loc_map`.
fn apply_prefix_substituents(
    mol: &mut AnalytMol,
    raw_pref: &str,
    loc_map: &HashMap<String, usize>,
) -> Result<(), String> {
    let pref = raw_pref.trim().trim_end_matches('-');
    if pref.is_empty() {
        return Ok(());
    }

    // Split `pref` at depth-0 hyphens that immediately precede `<locant_list>-`
    let bytes = pref.as_bytes();
    let mut segments: Vec<&str> = Vec::new();
    let mut seg_start = 0usize;
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] as char {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '-' if depth == 0 && i > seg_start => {
                // Check if `pref[i+1..]` begins with `<locant_list>-`
                let after = &pref[i + 1..];
                if let Some(next_dash) = after.find('-') {
                    let cand_locs = &after[..next_dash];
                    // Make sure we are not splitting inside `propan-2-yl` when unbracketed!
                    let after_dash = &after[next_dash + 1..];
                    let is_internal_yl = after_dash == "yl" || after_dash.starts_with("yl-");
                    if looks_like_locant_list(cand_locs) && !is_internal_yl {
                        segments.push(&pref[seg_start..i]);
                        seg_start = i + 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    segments.push(&pref[seg_start..]);

    for seg in segments {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        // Extract optional `<locant_list>-` at the start of `seg` (depth 0)
        let (locs, raw_body): (Vec<String>, &str) = if let Some(dpos) = seg.find('-') {
            let left = &seg[..dpos];
            if looks_like_locant_list(left) {
                let list = left
                    .split(',')
                    .map(|x| {
                        if x.eq_ignore_ascii_case("n") {
                            "N".to_string()
                        } else {
                            x.to_string()
                        }
                    })
                    .collect();
                (list, &seg[dpos + 1..])
            } else {
                (Vec::new(), seg)
            }
        } else {
            (Vec::new(), seg)
        };

        let (mult_cnt, sub_name) = split_multiplier(raw_body);
        if locs.is_empty() {
            let anchor = loc_map
                .get("")
                .or_else(|| loc_map.get("1"))
                .copied()
                .ok_or_else(|| format!("Missing default locant for prefix '{seg}'"))?;
            for _ in 0..mult_cnt {
                attach_substituent_by_name(mol, anchor, sub_name)?;
            }
        } else {
            for loc in &locs {
                let anchor = loc_map
                    .get(loc)
                    .or_else(|| if loc == "N" { loc_map.get("n") } else { None })
                    .copied()
                    .ok_or_else(|| format!("Unknown locant '{loc}' in prefix '{seg}'"))?;
                attach_substituent_by_name(mol, anchor, sub_name)?;
            }
        }
    }

    Ok(())
}

fn strip_leading_stereoprefix(input: &str) -> (&str, Vec<(Option<usize>, char)>) {
    let s = input.trim();
    if !s.starts_with('(') {
        return (s, Vec::new());
    }
    let Some(close_idx) = s.find(")-") else {
        return (s, Vec::new());
    };
    let inside = &s[1..close_idx];
    let mut specs = Vec::new();
    for tok in inside.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            return (s, Vec::new());
        }
        let last = t.chars().last().unwrap().to_ascii_uppercase();
        if !matches!(last, 'R' | 'S' | 'E' | 'Z') {
            return (s, Vec::new());
        }
        let num_part = &t[..t.len() - 1].trim_end_matches('\'');
        if num_part.is_empty() {
            specs.push((None, last));
        } else if let Ok(loc) = num_part.parse::<usize>() {
            specs.push((Some(loc), last));
        } else {
            return (s, Vec::new());
        }
    }
    (&s[close_idx + 2..], specs)
}

fn apply_stereodescriptors(
    mol: &mut AnalytMol,
    loc_map: &HashMap<String, usize>,
    specs: &[(Option<usize>, char)],
) {
    for &(opt_loc, kind) in specs {
        match kind {
            'R' | 'S' => {
                let tag = if kind == 'R' {
                    AtomStereoTag::R
                } else {
                    AtomStereoTag::S
                };
                if let Some(loc) = opt_loc {
                    if let Some(&a) = loc_map.get(&loc.to_string()) {
                        mol.atoms[a].stereo_tag = tag;
                    }
                } else {
                    // Find unique sp3 carbon with 4 distinct-looking attachments or C2/C1
                    for idx in 0..mol.atoms.len() {
                        if mol.atoms[idx].element == "C" && mol.total_h_count(idx) <= 1 {
                            mol.atoms[idx].stereo_tag = tag;
                            break;
                        }
                    }
                }
            }
            'E' | 'Z' => {
                let btag = if kind == 'E' {
                    BondStereoTag::Trans
                } else {
                    BondStereoTag::Cis
                };
                if let Some(loc) = opt_loc {
                    if let (Some(&u), Some(&v)) = (
                        loc_map.get(&loc.to_string()),
                        loc_map.get(&(loc + 1).to_string()),
                    ) {
                        if let Some(bi) = mol.bond_between(u, v) {
                            mol.bonds[bi].stereo = btag;
                        }
                    }
                } else if let Some(b) = mol.bonds.iter_mut().find(|b| b.order == 2 && !b.aromatic) {
                    b.stereo = btag;
                }
            }
            _ => {}
        }
    }
}

/// Apply chain/ring unsaturations (`-an-`, `-2-en-`, `-1,3-dien-`, `-2-yn-`) and principal suffixes
/// (`-oic acid`, `-dioic acid`, `-carboxylic acid`, `-sulfonic acid`, `-oyl chloride`, `-amide`, `-carboxamide`,
/// `-nitrile`, `-carbonitrile`, `-al`, `-carbaldehyde`, `-2-one`, `-2-ol`, `-1,2-diol`, `-thiol`, `-amine`).
fn apply_unsat_and_suffix(
    mol: &mut AnalytMol,
    tail: &str,
    skeleton: &[usize],
    loc_map: &mut HashMap<String, usize>,
    is_ring: bool,
) -> Result<(), String> {
    let mut rest = tail.trim().to_ascii_lowercase();
    if rest.is_empty() || rest == "e" || rest == "ane" {
        return Ok(());
    }

    // Extract principal suffix at the right end of `rest`
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SufKind {
        None,
        OicAcid,
        CarboxylicAcid,
        SulfonicAcid,
        OylHalide(&'static str),
        Amide,
        Carboxamide,
        Nitrile,
        Carbonitrile,
        Al,
        Carbaldehyde,
        One,
        Ol,
        Thiol,
        Amine,
    }

    let suffix_table: &[(&str, SufKind)] = &[
        ("carboxylic acid", SufKind::CarboxylicAcid),
        ("sulfonic acid", SufKind::SulfonicAcid),
        ("oic acid", SufKind::OicAcid),
        ("carbonyl chloride", SufKind::OylHalide("Cl")),
        ("carbonyl bromide", SufKind::OylHalide("Br")),
        ("carbonyl fluoride", SufKind::OylHalide("F")),
        ("oyl chloride", SufKind::OylHalide("Cl")),
        ("oyl bromide", SufKind::OylHalide("Br")),
        ("oyl fluoride", SufKind::OylHalide("F")),
        ("carboxamide", SufKind::Carboxamide),
        ("amide", SufKind::Amide),
        ("carbonitrile", SufKind::Carbonitrile),
        ("nitrile", SufKind::Nitrile),
        ("carbaldehyde", SufKind::Carbaldehyde),
        ("al", SufKind::Al),
        ("one", SufKind::One),
        ("ol", SufKind::Ol),
        ("thiol", SufKind::Thiol),
        ("amine", SufKind::Amine),
    ];

    let mut chosen_suf = SufKind::None;
    let mut suf_locs: Vec<usize> = Vec::new();
    let mut suf_mult = 1usize;

    for &(pat, kind) in suffix_table {
        if let Some(before_suf) = rest.strip_suffix(pat) {
            // Avoid matching `"thiol"` inside `"thiolane"` (already handled if `rest == "ane"`)
            let mut left = before_suf;
            // Check optional multiplier right before `pat` (`di`, `tri`, `tetra`)
            for &(mp, cnt) in &[("tetra", 4usize), ("tri", 3), ("di", 2)] {
                if let Some(stripped) = left.strip_suffix(mp) {
                    suf_mult = cnt;
                    left = stripped;
                    break;
                }
            }
            // Check optional `-<locs>-` right before multiplier/suffix
            let left_trimmed = left.trim_end_matches('-');
            if let Some(dash_idx) = left_trimmed.rfind('-') {
                let cand = &left_trimmed[dash_idx + 1..];
                if cand
                    .split(',')
                    .all(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
                {
                    suf_locs = cand
                        .split(',')
                        .filter_map(|t| t.parse::<usize>().ok())
                        .collect();
                    rest = left_trimmed[..dash_idx].to_string();
                } else {
                    rest = left_trimmed.to_string();
                }
            } else if !left_trimmed.is_empty()
                && left_trimmed
                    .split(',')
                    .all(|t| t.chars().all(|c| c.is_ascii_digit()))
            {
                suf_locs = left_trimmed
                    .split(',')
                    .filter_map(|t| t.parse::<usize>().ok())
                    .collect();
                rest.clear();
            } else {
                rest = left_trimmed.to_string();
            }
            chosen_suf = kind;
            break;
        }
    }

    // Parse unsaturations in `rest` (`an`, `ane`, `a`, `-2-en`, `en`, `-1,3-dien`, `-2-yn`, `yn`)
    let mut u_str = rest.trim_matches('-').to_string();
    if u_str.ends_with('e') {
        u_str.pop();
    }
    if u_str == "an" || u_str == "a" {
        u_str.clear();
    } else if let Some(after_a) = u_str.strip_prefix('a') {
        if after_a.starts_with('-') || after_a.starts_with(|c: char| c.is_ascii_digit()) {
            u_str = after_a.trim_start_matches('-').to_string();
        }
    }

    // Extract triple bonds (`yn`) and double bonds (`en`)
    for &(tag, order) in &[("yn", 3u8), ("en", 2u8)] {
        if let Some(pos) = u_str.rfind(tag) {
            if pos + tag.len() == u_str.len() {
                let mut before = &u_str[..pos];
                for mp in ["tetra", "tri", "di"] {
                    if let Some(s) = before.strip_suffix(mp) {
                        before = s;
                        break;
                    }
                }
                let before = before.trim_end_matches('-');
                let mut locs = Vec::new();
                if before.is_empty() {
                    locs.push(1usize);
                    u_str.clear();
                } else if let Some(dpos) = before.rfind('-') {
                    let cand = &before[dpos + 1..];
                    if cand
                        .split(',')
                        .all(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
                    {
                        locs = cand.split(',').filter_map(|t| t.parse().ok()).collect();
                        u_str = before[..dpos].trim_matches('-').to_string();
                    }
                } else if before
                    .split(',')
                    .all(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
                {
                    locs = before.split(',').filter_map(|t| t.parse().ok()).collect();
                    u_str.clear();
                }
                for l in locs {
                    let u_atom = *skeleton
                        .get(l - 1)
                        .ok_or_else(|| format!("Invalid unsaturation locant {l}"))?;
                    let v_atom = if l < skeleton.len() {
                        skeleton[l]
                    } else if is_ring && l == skeleton.len() {
                        skeleton[0]
                    } else {
                        return Err(format!("Out-of-range unsaturation locant {l}"));
                    };
                    if let Some(bi) = mol.bond_between(u_atom, v_atom) {
                        mol.bonds[bi].order = order;
                        mol.bonds[bi].aromatic = false;
                    }
                }
            }
        }
    }

    // Determine default locants for suffix when omitted (`propanoic acid` -> `[1]`, `butanedioic acid` -> `[1, len]`, `ethanol` -> `[1]`)
    if chosen_suf != SufKind::None && suf_locs.is_empty() {
        if suf_mult == 2 && !is_ring && skeleton.len() >= 2 {
            suf_locs = vec![1, skeleton.len()];
        } else {
            suf_locs = vec![1];
        }
    }

    for l in suf_locs {
        let c_idx = *loc_map
            .get(&l.to_string())
            .or_else(|| skeleton.get(l.saturating_sub(1)))
            .ok_or_else(|| format!("Invalid suffix locant {l}"))?;

        match chosen_suf {
            SufKind::None => {}
            SufKind::OicAcid => {
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                let oh = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::double(c_idx, o_dbl));
                mol.add_bond(AnalytBond::single(c_idx, oh));
            }
            SufKind::CarboxylicAcid => {
                let c = mol.add_atom(AnalytAtom::new("C"));
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                let oh = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::single(c_idx, c));
                mol.add_bond(AnalytBond::double(c, o_dbl));
                mol.add_bond(AnalytBond::single(c, oh));
            }
            SufKind::SulfonicAcid => {
                attach_substituent_by_name(mol, c_idx, "sulfo")?;
            }
            SufKind::OylHalide(hal) => {
                let target_c = if is_ring {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    mol.add_bond(AnalytBond::single(c_idx, c));
                    c
                } else {
                    c_idx
                };
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                let x = mol.add_atom(AnalytAtom::new(hal));
                mol.add_bond(AnalytBond::double(target_c, o_dbl));
                mol.add_bond(AnalytBond::single(target_c, x));
            }
            SufKind::Amide => {
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                let n = mol.add_atom(AnalytAtom::new("N"));
                mol.add_bond(AnalytBond::double(c_idx, o_dbl));
                mol.add_bond(AnalytBond::single(c_idx, n));
                loc_map.entry("N".to_string()).or_insert(n);
            }
            SufKind::Carboxamide => {
                let c = mol.add_atom(AnalytAtom::new("C"));
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                let n = mol.add_atom(AnalytAtom::new("N"));
                mol.add_bond(AnalytBond::single(c_idx, c));
                mol.add_bond(AnalytBond::double(c, o_dbl));
                mol.add_bond(AnalytBond::single(c, n));
                loc_map.entry("N".to_string()).or_insert(n);
            }
            SufKind::Nitrile => {
                let n = mol.add_atom(AnalytAtom::new("N"));
                mol.add_bond(AnalytBond::triple(c_idx, n));
            }
            SufKind::Carbonitrile => {
                let c = mol.add_atom(AnalytAtom::new("C"));
                let n = mol.add_atom(AnalytAtom::new("N"));
                mol.add_bond(AnalytBond::single(c_idx, c));
                mol.add_bond(AnalytBond::triple(c, n));
            }
            SufKind::Al => {
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::double(c_idx, o_dbl));
            }
            SufKind::Carbaldehyde => {
                let c = mol.add_atom(AnalytAtom::new("C"));
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::single(c_idx, c));
                mol.add_bond(AnalytBond::double(c, o_dbl));
            }
            SufKind::One => {
                let o_dbl = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::double(c_idx, o_dbl));
            }
            SufKind::Ol => {
                let oh = mol.add_atom(AnalytAtom::new("O"));
                mol.add_bond(AnalytBond::single(c_idx, oh));
            }
            SufKind::Thiol => {
                let sh = mol.add_atom(AnalytAtom::new("S"));
                mol.add_bond(AnalytBond::single(c_idx, sh));
            }
            SufKind::Amine => {
                let n = mol.add_atom(AnalytAtom::new("N"));
                mol.add_bond(AnalytBond::single(c_idx, n));
                loc_map.entry("N".to_string()).or_insert(n);
            }
        }
    }

    Ok(())
}

/// Apply skeletal replacement prefixes (`7-oxa`, `1,4-diaza`, `2-thia`) to `mol` using `loc_map`,
/// returning the remaining ordinary substituent prefix string.
fn apply_skeletal_oxa_aza_prefixes(
    mol: &mut AnalytMol,
    raw_pref: &str,
    loc_map: &HashMap<String, usize>,
) -> String {
    let pref = raw_pref.trim().trim_end_matches('-');
    if pref.is_empty() {
        return String::new();
    }
    let terms: &[(&str, &str)] = &[
        ("oxa", "O"),
        ("thia", "S"),
        ("selena", "Se"),
        ("aza", "N"),
        ("phospha", "P"),
        ("sila", "Si"),
        ("bora", "B"),
    ];
    let mut keep_parts: Vec<&str> = Vec::new();
    // Split top-level `-<locs>-` segments
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let bytes = pref.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] as char {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '-' if depth == 0 && i > start => {
                let after = &pref[i + 1..];
                if let Some(nd) = after.find('-') {
                    if looks_like_locant_list(&after[..nd]) {
                        parts.push(&pref[start..i]);
                        start = i + 1;
                    }
                }
            }
            _ => {}
        }
    }
    parts.push(&pref[start..]);

    for p in parts {
        let mut matched_skel = false;
        if let Some(dpos) = p.find('-') {
            let lpart = &p[..dpos];
            let body = &p[dpos + 1..];
            if looks_like_locant_list(lpart) {
                let (_, core) = split_multiplier(body);
                if let Some(&(_, el)) = terms.iter().find(|&&(t, _)| t == core) {
                    for l in lpart.split(',') {
                        if let Some(&atom_idx) = loc_map.get(l) {
                            mol.atoms[atom_idx].element = el.to_string();
                        }
                    }
                    matched_skel = true;
                }
            }
        } else {
            let (_, core) = split_multiplier(p);
            if let Some(&(_, el)) = terms.iter().find(|&&(t, _)| t == core) {
                if let Some(&atom_idx) = loc_map.get("1") {
                    mol.atoms[atom_idx].element = el.to_string();
                    matched_skel = true;
                }
            }
        }
        if !matched_skel {
            keep_parts.push(p);
        }
    }
    keep_parts.join("-")
}

/// Build retained aromatic / heterocyclic parent ring along with standard IUPAC locant map `1..=k` (and `"N"` when applicable).
fn build_retained_ring_parent(
    parent_key: &str,
) -> Option<(AnalytMol, Vec<usize>, HashMap<String, usize>)> {
    let mut mol = AnalytMol::new();
    let mut skel = Vec::new();
    let mut loc_map = HashMap::new();

    let add_six_arom = |m: &mut AnalytMol, elements: [&str; 6]| -> Vec<usize> {
        let mut r = Vec::with_capacity(6);
        for el in elements {
            r.push(m.add_atom(AnalytAtom::new(el)));
        }
        for k in 0..6 {
            let ord = if k % 2 == 0 { 2 } else { 1 };
            m.add_bond(AnalytBond::aromatic_bond(r[k], r[(k + 1) % 6], ord));
        }
        r
    };

    match parent_key {
        "benzene"
        | "benzoic acid"
        | "benzenesulfonic acid"
        | "benzoyl chloride"
        | "benzoyl bromide"
        | "benzamide"
        | "benzonitrile"
        | "benzaldehyde"
        | "phenol"
        | "aniline"
        | "toluene"
        | "acetophenone" => {
            skel = add_six_arom(&mut mol, ["C", "C", "C", "C", "C", "C"]);
            for (i, &a) in skel.iter().enumerate() {
                loc_map.insert((i + 1).to_string(), a);
            }
            loc_map.insert(String::new(), skel[0]);
            match parent_key {
                "benzoic acid" => {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    let o1 = mol.add_atom(AnalytAtom::new("O"));
                    let o2 = mol.add_atom(AnalytAtom::new("O"));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                    mol.add_bond(AnalytBond::double(c, o1));
                    mol.add_bond(AnalytBond::single(c, o2));
                }
                "benzenesulfonic acid" => {
                    let _ = attach_substituent_by_name(&mut mol, skel[0], "sulfo");
                }
                "benzoyl chloride" | "benzoyl bromide" => {
                    let hal = if parent_key.ends_with("bromide") {
                        "Br"
                    } else {
                        "Cl"
                    };
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    let o1 = mol.add_atom(AnalytAtom::new("O"));
                    let x = mol.add_atom(AnalytAtom::new(hal));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                    mol.add_bond(AnalytBond::double(c, o1));
                    mol.add_bond(AnalytBond::single(c, x));
                }
                "benzamide" => {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    let o1 = mol.add_atom(AnalytAtom::new("O"));
                    let n = mol.add_atom(AnalytAtom::new("N"));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                    mol.add_bond(AnalytBond::double(c, o1));
                    mol.add_bond(AnalytBond::single(c, n));
                    loc_map.insert("N".to_string(), n);
                }
                "benzonitrile" => {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    let n = mol.add_atom(AnalytAtom::new("N"));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                    mol.add_bond(AnalytBond::triple(c, n));
                }
                "benzaldehyde" => {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    let o1 = mol.add_atom(AnalytAtom::new("O"));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                    mol.add_bond(AnalytBond::double(c, o1));
                }
                "phenol" => {
                    let o = mol.add_atom(AnalytAtom::new("O"));
                    mol.add_bond(AnalytBond::single(skel[0], o));
                }
                "aniline" => {
                    let n = mol.add_atom(AnalytAtom::new("N"));
                    mol.add_bond(AnalytBond::single(skel[0], n));
                    loc_map.insert("N".to_string(), n);
                }
                "toluene" => {
                    let c = mol.add_atom(AnalytAtom::new("C"));
                    mol.add_bond(AnalytBond::single(skel[0], c));
                }
                "acetophenone" => {
                    let _ = attach_substituent_by_name(&mut mol, skel[0], "acetyl");
                }
                _ => {}
            }
            return Some((mol, skel, loc_map));
        }
        "pyridine" => skel = add_six_arom(&mut mol, ["N", "C", "C", "C", "C", "C"]),
        "pyridazine" => skel = add_six_arom(&mut mol, ["N", "N", "C", "C", "C", "C"]),
        "pyrimidine" => skel = add_six_arom(&mut mol, ["N", "C", "N", "C", "C", "C"]),
        "pyrazine" => skel = add_six_arom(&mut mol, ["N", "C", "C", "N", "C", "C"]),
        "furan" | "thiophene" | "1h-pyrrole" | "pyrrole" => {
            let hel = match parent_key {
                "furan" => "O",
                "thiophene" => "S",
                _ => "N",
            };
            let h = mol.add_atom(if hel == "N" {
                AnalytAtom::new("N").with_implicit_h(1)
            } else {
                AnalytAtom::new(hel)
            });
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let c3 = mol.add_atom(AnalytAtom::new("C"));
            let c4 = mol.add_atom(AnalytAtom::new("C"));
            let c5 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::aromatic_bond(h, c2, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c2, c3, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c3, c4, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c4, c5, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c5, h, 1));
            skel = vec![h, c2, c3, c4, c5];
        }
        "1h-imidazole" | "imidazole" => {
            let n1 = mol.add_atom(AnalytAtom::new("N").with_implicit_h(1));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let n3 = mol.add_atom(AnalytAtom::new("N"));
            let c4 = mol.add_atom(AnalytAtom::new("C"));
            let c5 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::aromatic_bond(n1, c2, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c2, n3, 2));
            mol.add_bond(AnalytBond::aromatic_bond(n3, c4, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c4, c5, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c5, n1, 1));
            skel = vec![n1, c2, n3, c4, c5];
        }
        "1h-pyrazole" | "pyrazole" => {
            let n1 = mol.add_atom(AnalytAtom::new("N").with_implicit_h(1));
            let n2 = mol.add_atom(AnalytAtom::new("N"));
            let c3 = mol.add_atom(AnalytAtom::new("C"));
            let c4 = mol.add_atom(AnalytAtom::new("C"));
            let c5 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::aromatic_bond(n1, n2, 1));
            mol.add_bond(AnalytBond::aromatic_bond(n2, c3, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c3, c4, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c4, c5, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c5, n1, 1));
            skel = vec![n1, n2, c3, c4, c5];
        }
        "1,3-oxazole" | "oxazole" | "1,3-thiazole" | "thiazole" => {
            let hel = if parent_key.contains("thia") {
                "S"
            } else {
                "O"
            };
            let x1 = mol.add_atom(AnalytAtom::new(hel));
            let c2 = mol.add_atom(AnalytAtom::new("C"));
            let n3 = mol.add_atom(AnalytAtom::new("N"));
            let c4 = mol.add_atom(AnalytAtom::new("C"));
            let c5 = mol.add_atom(AnalytAtom::new("C"));
            mol.add_bond(AnalytBond::aromatic_bond(x1, c2, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c2, n3, 2));
            mol.add_bond(AnalytBond::aromatic_bond(n3, c4, 1));
            mol.add_bond(AnalytBond::aromatic_bond(c4, c5, 2));
            mol.add_bond(AnalytBond::aromatic_bond(c5, x1, 1));
            skel = vec![x1, c2, n3, c4, c5];
        }
        "oxirane" | "aziridine" | "thiirane" => {
            let hel = match parent_key {
                "oxirane" => "O",
                "aziridine" => "N",
                _ => "S",
            };
            skel = vec![
                mol.add_atom(AnalytAtom::new(hel)),
                mol.add_atom(AnalytAtom::new("C")),
                mol.add_atom(AnalytAtom::new("C")),
            ];
            for k in 0..3 {
                mol.add_bond(AnalytBond::single(skel[k], skel[(k + 1) % 3]));
            }
        }
        "oxolane" | "pyrrolidine" | "thiolane" => {
            let hel = match parent_key {
                "oxolane" => "O",
                "pyrrolidine" => "N",
                _ => "S",
            };
            skel.push(mol.add_atom(AnalytAtom::new(hel)));
            for _ in 0..4 {
                skel.push(mol.add_atom(AnalytAtom::new("C")));
            }
            for k in 0..5 {
                mol.add_bond(AnalytBond::single(skel[k], skel[(k + 1) % 5]));
            }
        }
        "oxane" | "piperidine" | "piperazine" | "morpholine" => {
            let els: [&str; 6] = match parent_key {
                "oxane" => ["O", "C", "C", "C", "C", "C"],
                "piperidine" => ["N", "C", "C", "C", "C", "C"],
                "piperazine" => ["N", "C", "C", "N", "C", "C"],
                _ => ["O", "C", "C", "N", "C", "C"],
            };
            for el in els {
                skel.push(mol.add_atom(AnalytAtom::new(el)));
            }
            for k in 0..6 {
                mol.add_bond(AnalytBond::single(skel[k], skel[(k + 1) % 6]));
            }
        }
        "naphthalene" | "quinoline" | "isoquinoline" => {
            let smi = match parent_key {
                "naphthalene" => "c1ccc2ccccc2c1",
                "quinoline" => "n1cccc2ccccc12",
                _ => "c1nccc2ccccc12",
            };
            let parsed = AnalytMol::from_smiles(smi).ok()?;
            let arom = parsed.aromatic_atoms();
            let spec = identify_fused_bicyclic_parent(&parsed, &parsed.sssr_rings(), &arom)?;
            skel = spec.orders[0].clone();
            mol = parsed;
        }
        "1h-indole" | "indole" | "1-benzofuran" | "1-benzothiophene" | "1h-benzimidazole"
        | "9h-purine" | "purine" => {
            let smi = match parent_key {
                "1h-indole" | "indole" => "[nH]1ccc2ccccc12",
                "1-benzofuran" => "o1ccc2ccccc12",
                "1-benzothiophene" => "s1ccc2ccccc12",
                "1h-benzimidazole" => "[nH]1cnc2ccccc12",
                _ => "n1cnc2ncnc2c1",
            };
            let parsed = AnalytMol::from_smiles(smi).ok()?;
            let arom = parsed.aromatic_atoms();
            if let Some(spec) = identify_fused_bicyclic_parent(&parsed, &parsed.sssr_rings(), &arom)
            {
                skel = spec.orders[0].clone();
            } else {
                skel = (0..parsed.atoms.len()).collect();
            }
            mol = parsed;
        }
        _ => return None,
    }

    for (i, &a) in skel.iter().enumerate() {
        loc_map.insert((i + 1).to_string(), a);
    }
    if let Some(&first) = skel.first() {
        loc_map.insert(String::new(), first);
    }
    mol.perceive_aromaticity();
    Some((mol, skel, loc_map))
}

/// Parse a systematic IUPAC name (plus stereodescriptors, multiplicative assemblies, esters,
/// bridged von Baeyer / spiro rings, heterocycles, and branched chains) into `(AnalytMol, loc_map)`.
fn parse_systematic_with_locants(
    raw_name: &str,
) -> Result<(AnalytMol, HashMap<String, usize>), String> {
    let trimmed = raw_name.trim();
    let (after_stereo, stereo_specs) = strip_leading_stereoprefix(trimmed);
    let low = after_stereo.trim().to_ascii_lowercase();

    // 1. Multiplicative assembly: `<l1>,<l2>'-<linker>(di|bis)<parent>`
    //    e.g. `4,4'-oxydibenzoic acid`, `1,1'-sulfonylbis(4-chlorobenzene)`
    if let Some(dash_idx) = low.find('-') {
        let loc_pair = &low[..dash_idx];
        if let Some((l1_s, l2_raw)) = loc_pair.split_once(',') {
            if let Some(l2_s) = l2_raw.strip_suffix('\'') {
                if let (Ok(l1), Ok(l2)) = (l1_s.parse::<usize>(), l2_s.parse::<usize>()) {
                    let rest = &after_stereo.trim()[dash_idx + 1..];
                    let rest_low = rest.to_ascii_lowercase();
                    for &(linker, kind) in &[
                        ("sulfonyl", "SO2"),
                        ("sulfanediyl", "S"),
                        ("azanediyl", "NH"),
                        ("methylene", "CH2"),
                        ("oxy", "O"),
                    ] {
                        if let Some(after_link) = rest_low.strip_prefix(linker) {
                            let orig_after = &rest[linker.len()..];
                            let parent_str = if after_link.starts_with("bis(")
                                || after_link.starts_with("bis[")
                            {
                                strip_outer_brackets(&orig_after[3..])
                            } else if after_link.starts_with("di") {
                                &orig_after[2..]
                            } else {
                                continue;
                            };
                            let (mut m1, map1) = parse_systematic_with_locants(parent_str)?;
                            let (m2, map2) = parse_systematic_with_locants(parent_str)?;
                            let a1 = *map1
                                .get(&l1.to_string())
                                .ok_or_else(|| format!("Unknown multiplicative locant {l1}"))?;
                            let offset = m1.atoms.len();
                            for a in m2.atoms {
                                m1.add_atom(a);
                            }
                            for mut b in m2.bonds {
                                b.a += offset;
                                b.b += offset;
                                m1.add_bond(b);
                            }
                            let a2 = *map2
                                .get(&l2.to_string())
                                .ok_or_else(|| format!("Unknown multiplicative locant {l2}'"))?
                                + offset;
                            // Clear explicit implicit_h overrides on a1 and a2 so the new linker bond reduces implicit H by 1
                            m1.atoms[a1].implicit_h = None;
                            m1.atoms[a2].implicit_h = None;
                            match kind {
                                "O" => {
                                    let o = m1.add_atom(AnalytAtom::new("O"));
                                    m1.add_bond(AnalytBond::single(a1, o));
                                    m1.add_bond(AnalytBond::single(o, a2));
                                }
                                "S" => {
                                    let sv = m1.add_atom(AnalytAtom::new("S"));
                                    m1.add_bond(AnalytBond::single(a1, sv));
                                    m1.add_bond(AnalytBond::single(sv, a2));
                                }
                                "SO2" => {
                                    let sv = m1.add_atom(AnalytAtom::new("S"));
                                    let o1 = m1.add_atom(AnalytAtom::new("O"));
                                    let o2 = m1.add_atom(AnalytAtom::new("O"));
                                    m1.add_bond(AnalytBond::single(a1, sv));
                                    m1.add_bond(AnalytBond::single(sv, a2));
                                    m1.add_bond(AnalytBond::double(sv, o1));
                                    m1.add_bond(AnalytBond::double(sv, o2));
                                }
                                "NH" => {
                                    let n = m1.add_atom(AnalytAtom::new("N"));
                                    m1.add_bond(AnalytBond::single(a1, n));
                                    m1.add_bond(AnalytBond::single(n, a2));
                                }
                                _ => {
                                    let c = m1.add_atom(AnalytAtom::new("C"));
                                    m1.add_bond(AnalytBond::single(a1, c));
                                    m1.add_bond(AnalytBond::single(c, a2));
                                }
                            }
                            m1.perceive_aromaticity();
                            m1.generate_2d_coords_if_missing();
                            return Ok((m1, map1));
                        }
                    }
                }
            }
        }
    }

    // 2. Acyclic ester / carboxylate salt: `<alkyl_yl_or_cation> <acyl_ate>`
    //    e.g. `ethyl ethanoate`, `methyl 4-aminobenzoate`, `sodium ethanoate`
    if let Some(sp_idx) = find_depth0_space(&low) {
        let first_word = low[..sp_idx].trim();
        let second_word = after_stereo.trim()[sp_idx + 1..].trim();
        let sec_low = second_word.to_ascii_lowercase();
        let is_salt_cation = matches!(first_word, "sodium" | "potassium" | "lithium" | "ammonium");
        if (first_word.ends_with("yl") || is_salt_cation) && sec_low.ends_with("ate") {
            let acid_name = match sec_low.as_str() {
                "acetate" => "ethanoic acid".to_string(),
                "formate" => "methanoic acid".to_string(),
                "propionate" => "propanoic acid".to_string(),
                "butyrate" => "butanoic acid".to_string(),
                _ => {
                    let base = &second_word[..second_word.len() - 3];
                    format!("{base}ic acid")
                }
            };
            let (mut mol, loc_map) = parse_systematic_with_locants(&acid_name)?;
            // Locate carboxylic `-OH` oxygen (degree-1 single-bonded neutral O attached to C=O)
            let adj = mol.adjacency();
            let oh_idx = (0..mol.atoms.len())
                .find(|&i| {
                    mol.atoms[i].element == "O"
                        && adj[i].len() == 1
                        && mol.bonds[adj[i][0].1].order == 1
                        && mol.atoms[adj[i][0].0].element == "C"
                        && adj[adj[i][0].0].iter().any(|&(nb, bi)| {
                            nb != i && mol.bonds[bi].order == 2 && mol.atoms[nb].element == "O"
                        })
                })
                .ok_or_else(|| format!("Could not locate carboxylate oxygen in '{acid_name}'"))?;
            if is_salt_cation {
                mol.atoms[oh_idx].formal_charge = -1;
                mol.atoms[oh_idx].implicit_h = Some(0);
                let cat_el = match first_word {
                    "sodium" => ("Na", 0u8),
                    "potassium" => ("K", 0),
                    "lithium" => ("Li", 0),
                    _ => ("N", 4),
                };
                mol.add_atom(
                    AnalytAtom::new(cat_el.0)
                        .with_charge(1)
                        .with_implicit_h(cat_el.1),
                );
            } else {
                mol.atoms[oh_idx].implicit_h = Some(0);
                attach_substituent_by_name(&mut mol, oh_idx, first_word)?;
            }
            mol.generate_2d_coords_if_missing();
            return Ok((mol, loc_map));
        }
    }

    // 3. Von Baeyer `bicyclo[a.b.c]<stem><tail>` and `spiro[a.b]<stem><tail>`
    if let Some(vb_res) = try_parse_von_baeyer_or_spiro(&low, &stereo_specs)? {
        return Ok(vb_res);
    }

    // 4. Retained aromatic / heterocyclic ring parents (`4-aminobenzoic acid`, `2-chloropyridine`, `oxolan-2-one`, etc.)
    let retained_parents = [
        "benzenesulfonic acid",
        "benzoyl chloride",
        "benzoyl bromide",
        "benzoic acid",
        "benzonitrile",
        "benzaldehyde",
        "acetophenone",
        "benzamide",
        "1-benzothiophene",
        "1h-benzimidazole",
        "1-benzofuran",
        "isoquinoline",
        "naphthalene",
        "9h-purine",
        "1h-imidazole",
        "1h-pyrazole",
        "1,3-thiazole",
        "1,3-oxazole",
        "1h-pyrrole",
        "1h-indole",
        "pyrimidine",
        "pyridazine",
        "piperidine",
        "piperazine",
        "morpholine",
        "pyrrolidine",
        "quinoline",
        "thiophene",
        "pyridine",
        "pyrazine",
        "imidazole",
        "pyrazole",
        "thiazole",
        "oxazole",
        "pyrrole",
        "indole",
        "purine",
        "oxolane",
        "thiolane",
        "aziridine",
        "oxirane",
        "thiirane",
        "benzene",
        "toluene",
        "aniline",
        "phenol",
        "furan",
        "oxane",
    ];
    for pat in retained_parents {
        if let Some(pos) = find_depth0_substring(&low, pat) {
            let pref = &low[..pos];
            let tail = &low[pos + pat.len()..];
            // Ensure `tail` is either empty or starts with `-` or is a dropped-`'e'` suffix (`ol`, `one`)
            let pat_no_e = pat.strip_suffix('e').unwrap_or(pat);
            let valid_tail = tail.is_empty() || tail.starts_with('-');
            if valid_tail {
                if let Some((mut mol, skel, mut loc_map)) = build_retained_ring_parent(pat) {
                    apply_unsat_and_suffix(&mut mol, tail, &skel, &mut loc_map, true)?;
                    apply_prefix_substituents(&mut mol, pref, &loc_map)?;
                    for &ra in &skel {
                        mol.atoms[ra].implicit_h = None;
                    }
                    mol.perceive_aromaticity();
                    apply_stereodescriptors(&mut mol, &loc_map, &stereo_specs);
                    mol.generate_2d_coords_if_missing();
                    return Ok((mol, loc_map));
                }
            } else if pat.ends_with('e') && pat_no_e.len() >= 4 {
                if let Some(pos2) = find_depth0_substring(&low, pat_no_e) {
                    let tail2 = &low[pos2 + pat_no_e.len()..];
                    if tail2.starts_with('-') {
                        let pref2 = &low[..pos2];
                        if let Some((mut mol, skel, mut loc_map)) = build_retained_ring_parent(pat)
                        {
                            apply_unsat_and_suffix(&mut mol, tail2, &skel, &mut loc_map, true)?;
                            apply_prefix_substituents(&mut mol, pref2, &loc_map)?;
                            for &ra in &skel {
                                mol.atoms[ra].implicit_h = None;
                            }
                            mol.perceive_aromaticity();
                            apply_stereodescriptors(&mut mol, &loc_map, &stereo_specs);
                            mol.generate_2d_coords_if_missing();
                            return Ok((mol, loc_map));
                        }
                    }
                }
            }
        } else if let Some(pat_no_e) = pat.strip_suffix('e') {
            if pat_no_e.len() >= 4 {
                if let Some(pos2) = find_depth0_substring(&low, pat_no_e) {
                    let tail2 = &low[pos2 + pat_no_e.len()..];
                    if tail2.starts_with('-') {
                        let pref2 = &low[..pos2];
                        if let Some((mut mol, skel, mut loc_map)) = build_retained_ring_parent(pat)
                        {
                            apply_unsat_and_suffix(&mut mol, tail2, &skel, &mut loc_map, true)?;
                            apply_prefix_substituents(&mut mol, pref2, &loc_map)?;
                            for &ra in &skel {
                                mol.atoms[ra].implicit_h = None;
                            }
                            mol.perceive_aromaticity();
                            apply_stereodescriptors(&mut mol, &loc_map, &stereo_specs);
                            mol.generate_2d_coords_if_missing();
                            return Ok((mol, loc_map));
                        }
                    }
                }
            }
        }
    }

    // 5. Monocyclic `cyclo<stem><tail>` (including skeletal `oxacyclo...` / `azacyclo...`)
    if let Some(cpos) = find_depth0_substring(&low, "cyclo") {
        let pref = &low[..cpos];
        let after_cyclo = &low[cpos + 5..];
        for n in (3..=20).rev() {
            let st = alkane_stem(n);
            if let Some(tail) = after_cyclo.strip_prefix(&st) {
                let mut mol = AnalytMol::new();
                let mut skel = Vec::with_capacity(n);
                let mut loc_map = HashMap::new();
                for k in 0..n {
                    let a = mol.add_atom(AnalytAtom::new("C"));
                    skel.push(a);
                    loc_map.insert((k + 1).to_string(), a);
                }
                loc_map.insert(String::new(), skel[0]);
                for k in 0..n {
                    mol.add_bond(AnalytBond::single(skel[k], skel[(k + 1) % n]));
                }
                let rem_pref = apply_skeletal_oxa_aza_prefixes(&mut mol, pref, &loc_map);
                apply_unsat_and_suffix(&mut mol, tail, &skel, &mut loc_map, true)?;
                apply_prefix_substituents(&mut mol, &rem_pref, &loc_map)?;
                apply_stereodescriptors(&mut mol, &loc_map, &stereo_specs);
                mol.generate_2d_coords_if_missing();
                return Ok((mol, loc_map));
            }
        }
    }

    // 6. Acyclic chain `<prefixes><stem><tail>` (`2-methylpropan-2-ol`, `2-[4-(2-methylpropyl)phenyl]propanoic acid`, `(2E)-but-2-ene`)
    // Match the rightmost depth-0 `<stem>` whose remaining `tail` begins with a valid chain continuation (`an`, `en`, `yn`, `a-`, `-`)
    let mut best_chain_match: Option<(usize, usize, &str)> = None;
    for n in 1..=22 {
        let st = alkane_stem(n);
        if let Some(pos) = rfind_depth0_substring(&low, &st) {
            let tail = &low[pos + st.len()..];
            let valid = tail.starts_with("an")
                || tail.starts_with("en")
                || tail.starts_with("yn")
                || tail.starts_with("a-")
                || tail.starts_with("adi")
                || tail.starts_with("atri")
                || tail.starts_with('-');
            if valid {
                // Prefer rightmost occurrence, and on tie, longest stem (`undec` over `dec`!)
                if best_chain_match.as_ref().is_none_or(|&(bp, bl, _)| {
                    pos + st.len() > bp + bl || (pos + st.len() == bp + bl && st.len() > bl)
                }) {
                    best_chain_match = Some((pos, st.len(), tail));
                }
            }
        }
    }

    if let Some((pos, st_len, tail)) = best_chain_match {
        let stem_str = &low[pos..pos + st_len];
        let n =
            stem_to_length(stem_str).ok_or_else(|| format!("Invalid chain stem '{stem_str}'"))?;
        let pref = &low[..pos];
        let mut mol = AnalytMol::new();
        let mut skel = Vec::with_capacity(n);
        let mut loc_map = HashMap::new();
        for k in 0..n {
            let a = mol.add_atom(AnalytAtom::new("C"));
            skel.push(a);
            loc_map.insert((k + 1).to_string(), a);
        }
        loc_map.insert(String::new(), skel[0]);
        for k in 0..n.saturating_sub(1) {
            mol.add_bond(AnalytBond::single(skel[k], skel[k + 1]));
        }
        let rem_pref = apply_skeletal_oxa_aza_prefixes(&mut mol, pref, &loc_map);
        apply_unsat_and_suffix(&mut mol, tail, &skel, &mut loc_map, false)?;
        apply_prefix_substituents(&mut mol, &rem_pref, &loc_map)?;
        apply_stereodescriptors(&mut mol, &loc_map, &stereo_specs);
        mol.generate_2d_coords_if_missing();
        return Ok((mol, loc_map));
    }

    if let Some(smi) = lookup_common_or_drug_smiles(trimmed) {
        let mut m = AnalytMol::from_smiles(smi)?;
        m.generate_2d_coords_if_missing();
        return Ok((m, HashMap::new()));
    }

    Err(format!(
        "Unable to parse IUPAC or common chemical name '{raw_name}'"
    ))
}

#[allow(clippy::type_complexity)]
fn try_parse_von_baeyer_or_spiro(
    low: &str,
    stereo_specs: &[(Option<usize>, char)],
) -> Result<Option<(AnalytMol, HashMap<String, usize>)>, String> {
    if let Some(bpos) = find_depth0_substring(low, "bicyclo[") {
        let pref = &low[..bpos];
        let rest = &low[bpos + 8..];
        let close = rest
            .find(']')
            .ok_or_else(|| "Unclosed 'bicyclo[' bracket".to_string())?;
        let nums: Vec<usize> = rest[..close]
            .split('.')
            .map(|t| {
                t.parse::<usize>()
                    .map_err(|_| "Invalid bicyclo bridge count".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if nums.len() != 3 {
            return Err("bicyclo[a.b.c] requires exactly 3 bridge lengths".to_string());
        }
        let (a, b, c) = (nums[0], nums[1], nums[2]);
        let total_n = a + b + c + 2;
        let after_br = &rest[close + 1..];
        let st = alkane_stem(total_n);
        let tail = after_br
            .strip_prefix(&st)
            .ok_or_else(|| format!("Expected stem '{st}' for bicyclo[{a}.{b}.{c}]"))?;

        let mut mol = AnalytMol::new();
        let mut skel = Vec::with_capacity(total_n);
        let mut loc_map = HashMap::new();
        for k in 0..total_n {
            let idx = mol.add_atom(AnalytAtom::new("C"));
            skel.push(idx);
            loc_map.insert((k + 1).to_string(), idx);
        }
        let h1 = skel[0];
        let h2 = skel[a + 1];
        // Bridge 1 (`a` atoms): h1 - skel[1] - ... - skel[a] - h2
        let mut prev = h1;
        for &v in &skel[1..=a] {
            mol.add_bond(AnalytBond::single(prev, v));
            prev = v;
        }
        mol.add_bond(AnalytBond::single(prev, h2));
        // Bridge 2 (`b` atoms): h2 - skel[a+2] - ... - skel[a+1+b] - h1
        if b > 0 {
            prev = h2;
            for &v in &skel[a + 2..=a + 1 + b] {
                mol.add_bond(AnalytBond::single(prev, v));
                prev = v;
            }
            mol.add_bond(AnalytBond::single(prev, h1));
        }
        // Bridge 3 (`c` atoms): h1 - skel[a+2+b] - ... - h2 (only add direct bond if b > 0 to avoid duplicate edge when b==0 && c==0)
        if c > 0 {
            prev = h1;
            for &v in &skel[a + 2 + b..total_n] {
                mol.add_bond(AnalytBond::single(prev, v));
                prev = v;
            }
            mol.add_bond(AnalytBond::single(prev, h2));
        }

        let rem_pref = apply_skeletal_oxa_aza_prefixes(&mut mol, pref, &loc_map);
        apply_unsat_and_suffix(&mut mol, tail, &skel, &mut loc_map, false)?;
        apply_prefix_substituents(&mut mol, &rem_pref, &loc_map)?;
        apply_stereodescriptors(&mut mol, &loc_map, stereo_specs);
        mol.generate_2d_coords_if_missing();
        return Ok(Some((mol, loc_map)));
    }

    if let Some(spos) = find_depth0_substring(low, "spiro[") {
        let pref = &low[..spos];
        let rest = &low[spos + 6..];
        let close = rest
            .find(']')
            .ok_or_else(|| "Unclosed 'spiro[' bracket".to_string())?;
        let nums: Vec<usize> = rest[..close]
            .split('.')
            .map(|t| {
                t.parse::<usize>()
                    .map_err(|_| "Invalid spiro loop count".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if nums.len() != 2 || nums[0] < 2 || nums[1] < 2 {
            return Err("spiro[a.b] requires 2 ring loop lengths >= 2".to_string());
        }
        let (a, b) = (nums[0], nums[1]);
        let total_n = a + b + 1;
        let after_br = &rest[close + 1..];
        let st = alkane_stem(total_n);
        let tail = after_br
            .strip_prefix(&st)
            .ok_or_else(|| format!("Expected stem '{st}' for spiro[{a}.{b}]"))?;

        let mut mol = AnalytMol::new();
        let mut skel = Vec::with_capacity(total_n);
        let mut loc_map = HashMap::new();
        for k in 0..total_n {
            let idx = mol.add_atom(AnalytAtom::new("C"));
            skel.push(idx);
            loc_map.insert((k + 1).to_string(), idx);
        }
        let sp = skel[a]; // locant `a + 1`
                          // Small loop: sp - skel[0] - ... - skel[a-1] - sp
        mol.add_bond(AnalytBond::single(sp, skel[0]));
        for m in 0..a - 1 {
            mol.add_bond(AnalytBond::single(skel[m], skel[m + 1]));
        }
        mol.add_bond(AnalytBond::single(skel[a - 1], sp));
        // Large loop: sp - skel[a+1] - ... - skel[a+b] - sp
        mol.add_bond(AnalytBond::single(sp, skel[a + 1]));
        for m in (a + 1)..(total_n - 1) {
            mol.add_bond(AnalytBond::single(skel[m], skel[m + 1]));
        }
        mol.add_bond(AnalytBond::single(skel[total_n - 1], sp));

        let rem_pref = apply_skeletal_oxa_aza_prefixes(&mut mol, pref, &loc_map);
        apply_unsat_and_suffix(&mut mol, tail, &skel, &mut loc_map, false)?;
        apply_prefix_substituents(&mut mol, &rem_pref, &loc_map)?;
        apply_stereodescriptors(&mut mol, &loc_map, stereo_specs);
        mol.generate_2d_coords_if_missing();
        return Ok(Some((mol, loc_map)));
    }

    Ok(None)
}

fn find_depth0_space(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ' ' if depth == 0 => {
                let right = s[i + 1..].trim();
                // Do not split before `"acid"`, `"chloride"`, `"bromide"`, `"fluoride"`, `"iodide"`
                if !matches!(
                    right,
                    "acid" | "chloride" | "bromide" | "fluoride" | "iodide"
                ) {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_depth0_substring(hay: &str, needle: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in hay.char_indices() {
        if depth == 0 && hay[i..].starts_with(needle) {
            return Some(i);
        }
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    None
}

fn rfind_depth0_substring(hay: &str, needle: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut last = None;
    for (i, ch) in hay.char_indices() {
        if depth == 0 && hay[i..].starts_with(needle) {
            last = Some(i);
        }
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    last
}

/// Offline Name -> Structure parser converting systematic IUPAC names, common lab reagents/solvents,
/// amino acids, and benchmark pharmaceuticals into an `AnalytMol` with generated 2D coordinates.
pub fn iupac_name_to_mol(name: &str) -> Result<AnalytMol, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Empty chemical name".to_string());
    }

    // Handle trailing salt / hydrate modifiers (`... hydrochloride`, `... hydrate`)
    let low = trimmed.to_ascii_lowercase();
    for &(suf, smi) in &[
        (" hydrochloride", "Cl"),
        (" hydrobromide", "Br"),
        (" monohydrate", "O"),
        (" hydrate", "O"),
    ] {
        if low.ends_with(suf) {
            let base = &trimmed[..trimmed.len() - suf.len()];
            let (mut mol, _) = parse_systematic_with_locants(base)?;
            let extra = AnalytMol::from_smiles(smi)?;
            let off = mol.atoms.len();
            for mut a in extra.atoms {
                a.x += 6.0;
                mol.add_atom(a);
            }
            for mut b in extra.bonds {
                b.a += off;
                b.b += off;
                mol.add_bond(b);
            }
            mol.name = Some(trimmed.to_string());
            return Ok(mol);
        }
    }

    let (mut mol, _) = parse_systematic_with_locants(trimmed)?;
    mol.name = Some(trimmed.to_string());
    Ok(mol)
}

/// Offline Name -> SMILES helper wrapping [`iupac_name_to_mol`].
pub fn iupac_name_to_smiles(name: &str) -> Result<String, String> {
    let mol = iupac_name_to_mol(name)?;
    Ok(mol.to_smiles())
}
