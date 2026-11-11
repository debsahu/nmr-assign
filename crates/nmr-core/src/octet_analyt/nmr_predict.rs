//! Small-molecule ¹H and ¹³C NMR chemical shift, topological equivalence, J-coupling,
//! multiplet splitting tree, and ACS characterization string predictor.
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules:
//! - Bremser, W. *Anal. Chim. Acta* **1978**, 103, 355–365 (HOSE 6-sphere topological codes).
//! - Morgan, H. L. *J. Chem. Doc.* **1965**, 5, 107–113 (extended connectivity symmetry classes).
//! - Grant, D. M.; Paul, E. G. *J. Am. Chem. Soc.* **1964**, 86, 2984–2990 (¹³C alkane additivity).
//! - Pascual, C.; Meier, J.; Simon, W. *Helv. Chim. Acta* **1966**, 49, 164–168 (vinylic ¹H additivity).
//! - Pretsch, E.; Bühlmann, P.; Badertscher, M. *Structure Determination of Organic Compounds:
//!   Tables of Spectral Data*, 4th ed., Springer, 2009 (¹H/¹³C substituent increments & J-couplings).

use crate::octet_analyt::{is_metal_element, AnalytBond, AnalytMol, BondStereoTag};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

/// Number of HOSE code spheres generated per heavy center.
pub const HOSE_LEVELS: usize = 6;

/// Target NMR nucleus (`1H` or `13C`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NmrNucleus {
    H1,
    C13,
}

impl NmrNucleus {
    pub fn as_str(&self) -> &'static str {
        match self {
            NmrNucleus::H1 => "1H",
            NmrNucleus::C13 => "13C",
        }
    }
}

/// First-order scalar spin-spin coupling entry (`J` in Hz to `n` equivalent protons).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpinCoupling {
    /// Coupling constant `J` in Hz.
    pub j_hz: f64,
    /// Number of equivalent coupled protons (`n` gives `n + 1` lines with binomial intensities).
    pub count: usize,
    /// Indices of coupled heavy atoms in the original `AnalytMol`.
    pub coupled_atoms: Vec<usize>,
    /// Classification (`"3J_alkyl"`, `"3J_trans"`, `"3J_cis"`, `"2J_gem"`, `"3J_ortho"`, `"4J_meta"`, `"3J_ald"`, `"3J_hapto"`).
    pub kind: String,
}

/// Individual sub-peak in an exact first-order multiplet splitting tree.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultipletLine {
    /// Offset from multiplet center in Hz.
    pub offset_hz: f64,
    /// Relative intensity normalized so the sum of `rel_intensity` across the multiplet equals `nuclei`.
    pub intensity: f64,
}

/// Confidence tier of a predicted shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShiftQuality {
    Good,
    Medium,
    Rough,
}

impl ShiftQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            ShiftQuality::Good => "good",
            ShiftQuality::Medium => "medium",
            ShiftQuality::Rough => "rough",
        }
    }
}

/// Predicted ¹H or ¹³C NMR resonance group (chemically equivalent nuclei).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NmrSignal {
    /// Indices of symmetry-equivalent heavy atoms in `AnalytMol` (for ¹H, the heavy atoms bearing the protons).
    pub atoms: Vec<usize>,
    /// Predicted chemical shift `δ` in ppm.
    pub shift_ppm: f64,
    /// Integral count (number of ¹H protons or ¹³C carbons).
    pub nuclei: usize,
    /// Scalar couplings sorted from largest `J` to smallest `J`.
    pub couplings: Vec<SpinCoupling>,
    /// Exact first-order multiplet splitting tree lines.
    pub multiplet_lines: Vec<MultipletLine>,
    /// True for labile exchangeable `OH` / `NH` / `SH` resonances.
    pub broad: bool,
    /// Estimated uncertainty (standard deviation in ppm).
    pub sd_ppm: f64,
    /// Effective sphere match depth (`0..=6`).
    pub sphere_level: usize,
    /// Human-readable group label (e.g. `"CH3"`, `"2 × CH"`, `"OH"`).
    pub group_label: String,
    /// Base-36 FNV-1a hash of the 6-sphere HOSE code (canonical equivalence class identifier).
    pub hose_hash: String,
    /// Full 6-sphere HOSE code string.
    pub hose_code: String,
    /// Solvent name if predicted for a specific deuterated solvent.
    pub solvent: Option<String>,
}

impl NmrSignal {
    /// Standard chemist's multiplicity abbreviation (`s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `dt`, `td`, `ddd`, `m`, `br s`).
    pub fn multiplicity(&self) -> String {
        multiplicity_string(self)
    }

    /// Prediction quality tier (`good`, `medium`, `rough`).
    pub fn quality(&self, nucleus: NmrNucleus) -> ShiftQuality {
        signal_quality(self, nucleus)
    }

    /// Realized sub-peak `(ppm, intensity)` pairs at a given ¹H spectrometer frequency (`mhz_1h`).
    pub fn lines_at_mhz(&self, nucleus: NmrNucleus, mhz_1h: f64) -> Vec<(f64, f64)> {
        let obs_mhz = match nucleus {
            NmrNucleus::H1 => mhz_1h.max(1.0),
            NmrNucleus::C13 => (mhz_1h * 0.25145).max(1.0),
        };
        if self.multiplet_lines.is_empty() {
            return vec![(self.shift_ppm, self.nuclei as f64)];
        }
        self.multiplet_lines
            .iter()
            .map(|l| (self.shift_ppm + l.offset_hz / obs_mhz, l.intensity))
            .collect()
    }
}

/// Complete 1D ¹H or ¹³C NMR prediction result for a molecule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NmrPrediction {
    pub nucleus: NmrNucleus,
    /// Signals ordered from downfield (high δ) to upfield (low δ).
    pub signals: Vec<NmrSignal>,
    pub solvent: String,
}

impl NmrPrediction {
    /// Format an ACS journal characterization report line:
    /// - `¹H NMR (400 MHz, CDCl₃) δ 7.83 (d, J = 8.0 Hz, 2H), ...`
    /// - `¹³C NMR (101 MHz, CDCl₃) δ 197.0, 135.8, ...`
    pub fn acs_text(&self, mhz_1h: f64) -> String {
        acs_report_text(self, mhz_1h, &self.solvent)
    }

    /// Tab-separated summary table matching Octet-Draw's `detailsTable`.
    pub fn details_tsv(&self) -> String {
        details_table_tsv(self)
    }
}

/// Optional binary HOSE lookup table (`ONMR` v1/v2 format from Octet-Draw).
#[derive(Debug, Clone, Default)]
pub struct OnmrShiftTable {
    pub hashes: Vec<u32>,
    pub shifts_centi_ppm: Vec<i16>,
    pub sds_centi_ppm: Vec<u16>,
    pub counts: Vec<u16>,
}

impl OnmrShiftTable {
    pub fn lookup(&self, hash: u32) -> Option<(f64, f64, usize)> {
        let idx = self.hashes.binary_search(&hash).ok()?;
        Some((
            self.shifts_centi_ppm[idx] as f64 / 100.0,
            self.sds_centi_ppm[idx] as f64 / 100.0,
            self.counts[idx] as usize,
        ))
    }
}

/// Pair of ¹³C and ¹H `ONMR` binary shift tables.
#[derive(Debug, Clone, Default)]
pub struct OnmrTables {
    pub c13: OnmrShiftTable,
    pub h1: OnmrShiftTable,
    pub solvent_c13: Option<OnmrShiftTable>,
    pub solvent_h1: Option<OnmrShiftTable>,
}

impl OnmrTables {
    /// Parse an `ONMR` v1 or v2 binary buffer (`readTables` in `nmr.ts`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[0..4] != b"ONMR" {
            return Err("Not an ONMR shift table".to_string());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 && version != 2 {
            return Err(format!("Unsupported ONMR table version {version}"));
        }
        let parts = if version == 2 { 4usize } else { 2usize };
        let head = 8 + 4 * parts;
        if bytes.len() < head {
            return Err("ONMR shift table header is truncated".to_string());
        }
        let mut counts = Vec::with_capacity(parts);
        for k in 0..parts {
            let off = 8 + 4 * k;
            counts.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
        }
        let mut at = head;
        let mut parsed = Vec::with_capacity(parts);
        for &count in &counts {
            if at + count * 10 > bytes.len() {
                return Err("ONMR shift table body is truncated".to_string());
            }
            let mut hashes = Vec::with_capacity(count);
            for _ in 0..count {
                hashes.push(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
                at += 4;
            }
            let mut shifts = Vec::with_capacity(count);
            for _ in 0..count {
                shifts.push(i16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
                at += 2;
            }
            let mut sds = Vec::with_capacity(count);
            for _ in 0..count {
                sds.push(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
                at += 2;
            }
            let mut ns = Vec::with_capacity(count);
            for _ in 0..count {
                ns.push(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
                at += 2;
            }
            parsed.push(OnmrShiftTable {
                hashes,
                shifts_centi_ppm: shifts,
                sds_centi_ppm: sds,
                counts: ns,
            });
        }
        Ok(Self {
            c13: parsed[0].clone(),
            h1: parsed[1].clone(),
            solvent_c13: if parts == 4 {
                Some(parsed[2].clone())
            } else {
                None
            },
            solvent_h1: if parts == 4 {
                Some(parsed[3].clone())
            } else {
                None
            },
        })
    }

    /// Serialize to `ONMR` v1/v2 binary format (`writeTables` in `nmr.ts`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let has_solv = self.solvent_c13.is_some() && self.solvent_h1.is_some();
        let mut parts = vec![&self.c13, &self.h1];
        if has_solv {
            parts.push(self.solvent_c13.as_ref().unwrap());
            parts.push(self.solvent_h1.as_ref().unwrap());
        }
        let head = 8 + 4 * parts.len();
        let body: usize = parts.iter().map(|p| p.hashes.len() * 10).sum();
        let mut out = Vec::with_capacity(head + body);
        out.extend_from_slice(b"ONMR");
        out.extend_from_slice(&(if has_solv { 2u32 } else { 1u32 }).to_le_bytes());
        for p in &parts {
            out.extend_from_slice(&(p.hashes.len() as u32).to_le_bytes());
        }
        for p in &parts {
            for &h in &p.hashes {
                out.extend_from_slice(&h.to_le_bytes());
            }
            for &s in &p.shifts_centi_ppm {
                out.extend_from_slice(&s.to_le_bytes());
            }
            for &sd in &p.sds_centi_ppm {
                out.extend_from_slice(&sd.to_le_bytes());
            }
            for &n in &p.counts {
                out.extend_from_slice(&n.to_le_bytes());
            }
        }
        out
    }
}

// --- Heavy-atom normalization, η-hapto perception, Morgan symmetry & HOSE codes ---

#[derive(Debug, Clone)]
struct HeavyAtom {
    el: String,
    charge: i8,
    h: u8,
}

#[derive(Debug, Clone)]
struct HeavyGraph {
    atoms: Vec<HeavyAtom>,
    bonds: Vec<AnalytBond>,
    from_mol: Vec<usize>,
}

#[derive(Debug, Clone)]
struct EnvEdge {
    to: usize,
    bond_sym: &'static str,
    bond_idx: usize,
}

/// Topological environment of a heavy-atom molecular graph (`environment` in `nmr.ts`).
#[derive(Debug, Clone)]
pub struct TopologicalEnv {
    atoms: Vec<HeavyAtom>,
    bonds: Vec<AnalytBond>,
    edges: Vec<Vec<EnvEdge>>,
    labels: Vec<String>,
    /// Canonical symmetry equivalence class per heavy atom (`0..num_classes`).
    pub symmetry_class: Vec<usize>,
    pub aromatic: Vec<bool>,
    /// Index of η-coordinated metal atom in `HeavyGraph` if hapto-bonded, else `None`.
    pub eta_metal: Vec<Option<usize>>,
    /// Map from heavy-atom index back to atom index in the original `AnalytMol`.
    pub from_mol: Vec<usize>,
}

fn is_metal_like(el: &str) -> bool {
    el == "*" || is_metal_element(el)
}

fn detect_hapto(atoms: &[HeavyAtom], bonds: &[AnalytBond]) -> Vec<Option<usize>> {
    let n = atoms.len();
    let mut out = vec![None; n];
    let mut metal_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for b in bonds {
        for &(m, x) in &[(b.a, b.b), (b.b, b.a)] {
            if is_metal_like(&atoms[m].el) && !is_metal_like(&atoms[x].el) && atoms[x].el != "H" {
                metal_of.entry(x).or_default().push(m);
            }
        }
    }
    if metal_of.is_empty() {
        return out;
    }
    for b in bonds {
        if let (Some(ma), Some(mb)) = (metal_of.get(&b.a), metal_of.get(&b.b)) {
            if let Some(&shared) = ma.iter().find(|m| mb.contains(m)) {
                out[b.a] = Some(shared);
                out[b.b] = Some(shared);
            }
        }
    }
    out
}

fn build_heavy_graph(mol: &AnalytMol) -> HeavyGraph {
    let n = mol.atoms.len();
    let mut keep = vec![false; n];
    let mut to_heavy = vec![usize::MAX; n];
    let mut from_mol = Vec::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        let is_protium = a.element == "H" && a.isotope.unwrap_or(1) == 1;
        if !is_protium {
            to_heavy[i] = from_mol.len();
            from_mol.push(i);
            keep[i] = true;
        }
    }
    let mut atoms: Vec<HeavyAtom> = from_mol
        .iter()
        .map(|&orig| HeavyAtom {
            el: mol.atoms[orig].element.clone(),
            charge: mol.atoms[orig].formal_charge,
            h: mol.total_h_count(orig),
        })
        .collect();

    let mut bonds = Vec::new();
    for b in &mol.bonds {
        if b.a < n && b.b < n && keep[b.a] && keep[b.b] {
            let mut nb = b.clone();
            nb.a = to_heavy[b.a];
            nb.b = to_heavy[b.b];
            bonds.push(nb);
        }
    }

    let eta = detect_hapto(&atoms, &bonds);
    if eta.iter().any(|m| m.is_some()) {
        for i in 0..atoms.len() {
            let Some(metal_idx) = eta[i] else {
                continue;
            };
            atoms[i].charge = 0;
            if atoms[i].el != "C" {
                continue;
            }
            let mut used: u8 = 1;
            for b in &bonds {
                if let Some(o) = b.other(i) {
                    if !is_metal_like(&atoms[o].el) {
                        used = used.saturating_add(if eta[o] == Some(metal_idx) {
                            1
                        } else {
                            b.order
                        });
                    }
                }
            }
            atoms[i].h = 4u8.saturating_sub(used);
        }
    }

    HeavyGraph {
        atoms,
        bonds,
        from_mol,
    }
}

/// Non-recursive Tarjan bridge detection (`bridges` in `nmr.ts`).
fn tarjan_bridges(n: usize, adj: &[Vec<(usize, usize)>]) -> HashSet<usize> {
    let mut order = vec![-1isize; n];
    let mut low = vec![0isize; n];
    let mut out = HashSet::new();
    let mut t = 0isize;

    for s in 0..n {
        if order[s] >= 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize, usize)> = vec![(s, usize::MAX, 0)];
        order[s] = t;
        low[s] = t;
        t += 1;
        while let Some((v, via, idx)) = stack.last_mut() {
            let v_cur = *v;
            let via_cur = *via;
            if *idx < adj[v_cur].len() {
                let (to, k) = adj[v_cur][*idx];
                *idx += 1;
                if k == via_cur {
                    continue;
                }
                if order[to] < 0 {
                    order[to] = t;
                    low[to] = t;
                    t += 1;
                    stack.push((to, k, 0));
                } else if order[to] < low[v_cur] {
                    low[v_cur] = order[to];
                }
            } else {
                stack.pop();
                if let Some((up_v, _, _)) = stack.last() {
                    let up = *up_v;
                    if low[v_cur] < low[up] {
                        low[up] = low[v_cur];
                    }
                    if low[v_cur] > order[up] {
                        out.insert(via_cur);
                    }
                }
            }
        }
    }
    out
}

/// Compute refined Morgan topological symmetry classes and bond environments on `mol`.
pub fn perceive_topological_env(mol: &AnalytMol) -> TopologicalEnv {
    let hg = build_heavy_graph(mol);
    let n = hg.atoms.len();
    let mut adj = vec![Vec::new(); n];
    for (k, b) in hg.bonds.iter().enumerate() {
        adj[b.a].push((b.b, k));
        adj[b.b].push((b.a, k));
    }

    let eta = detect_hapto(&hg.atoms, &hg.bonds);

    // Perceive aromaticity on the original molecule and project to non-hapto heavy atoms.
    let mol_arom = mol.aromatic_atoms();
    let aromatic: Vec<bool> = (0..n)
        .map(|i| mol_arom[hg.from_mol[i]] && eta[i].is_none())
        .collect();

    let bridge_set = tarjan_bridges(n, &adj);

    // Perceive E/Z double-bond stereochemistry so trans vs cis substituents on unsymmetrical alkenes
    // (e.g. RCH=CH2 or R1R2C=CHR3) split properly in symmetry classes when geometrically distinct.
    let stereo = mol.perceive_cip_stereo();

    let sym = |k: usize| -> &'static str {
        let b = &hg.bonds[k];
        if eta[b.a].is_some() && eta[b.a] == eta[b.b] {
            return "~";
        }
        if eta[b.a] == Some(b.b) || eta[b.b] == Some(b.a) {
            return "^";
        }
        if aromatic[b.a] && aromatic[b.b] && !bridge_set.contains(&k) {
            "*"
        } else if b.order == 2 {
            "="
        } else if b.order == 3 {
            "%"
        } else {
            ""
        }
    };

    let edges: Vec<Vec<EnvEdge>> = (0..n)
        .map(|i| {
            adj[i]
                .iter()
                .map(|&(to, k)| EnvEdge {
                    to,
                    bond_sym: sym(k),
                    bond_idx: k,
                })
                .collect()
        })
        .collect();

    let labels: Vec<String> = hg
        .atoms
        .iter()
        .map(|a| {
            let q = if a.charge > 0 {
                "+"
            } else if a.charge < 0 {
                "-"
            } else {
                ""
            };
            format!("{}{q}", a.el)
        })
        .collect();

    let rank_keys = |keys: &[String]| -> (Vec<usize>, usize) {
        let unique: BTreeSet<&String> = keys.iter().collect();
        let map: HashMap<&String, usize> = unique
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, i))
            .collect();
        let classes = keys.iter().map(|k| map[k]).collect();
        (classes, map.len())
    };

    // Initial atom invariants (note: neutral resonating oxygens on -COO- or -NO2 normalize symmetrically via Morgan if charges match,
    // and we also symmetrize resonance-equivalent nitro/carboxylate oxygens).
    let init_keys: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "{}|{}|{}|{}",
                labels[i],
                hg.atoms[i].h,
                usize::from(aromatic[i]),
                edges[i].len()
            )
        })
        .collect();

    let (mut cls, mut count) = rank_keys(&init_keys);
    for _ in 0..n {
        let next_keys: Vec<String> = (0..n)
            .map(|i| {
                let mut nb_tags: Vec<String> = edges[i]
                    .iter()
                    .map(|e| {
                        // Include E/Z bond descriptor on double bonds if present
                        let ez_tag = if e.bond_sym == "=" {
                            let orig_b = mol
                                .bond_between(hg.from_mol[i], hg.from_mol[e.to])
                                .and_then(|bi| stereo.bond_ez.get(&bi).copied());
                            match orig_b {
                                Some('E') => "E",
                                Some('Z') => "Z",
                                _ => "",
                            }
                        } else {
                            ""
                        };
                        format!("{}{ez_tag}{}", e.bond_sym, cls[e.to])
                    })
                    .collect();
                nb_tags.sort();
                format!("{}|{}", cls[i], nb_tags.join(","))
            })
            .collect();
        let (next_cls, next_count) = rank_keys(&next_keys);
        cls = next_cls;
        if next_count == count {
            break;
        }
        count = next_count;
    }

    TopologicalEnv {
        atoms: hg.atoms,
        bonds: hg.bonds,
        edges,
        labels,
        symmetry_class: cls,
        aromatic,
        eta_metal: eta,
        from_mol: hg.from_mol,
    }
}

/// 32-bit FNV-1a hash of `"{nucleus}:{code}"` matching Octet-Draw's `codeHash`.
pub fn fnv1a_code_hash(nucleus: NmrNucleus, code: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in nucleus
        .as_str()
        .bytes()
        .chain(std::iter::once(b':'))
        .chain(code.bytes())
    {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn short_hex(s: &str) -> String {
    format!("{:08x}", fnv1a_code_hash(NmrNucleus::H1, s))
}

/// Generate canonical Bremser-style 1..=`levels` sphere HOSE codes (`hose` in `nmr.ts`).
pub fn hose_codes(env: &TopologicalEnv, center: usize, levels: usize) -> Vec<String> {
    let mut dist: HashMap<usize, usize> = HashMap::new();
    dist.insert(center, 0);
    let head = format!("{}H{}", env.labels[center], env.atoms[center].h);
    let mut desc: HashMap<usize, String> = HashMap::new();
    desc.insert(center, head.clone());

    let mut spheres: Vec<String> = Vec::with_capacity(levels);
    let mut out: Vec<String> = Vec::with_capacity(levels);

    for level in 1..=levels {
        let mut reached: HashMap<usize, Vec<String>> = HashMap::new();
        for (&p, d) in &desc {
            for e in &env.edges[p] {
                if let Some(&at) = dist.get(&e.to) {
                    if at != level {
                        continue;
                    }
                } else {
                    dist.insert(e.to, level);
                }
                reached
                    .entry(e.to)
                    .or_default()
                    .push(format!("{}{d}", e.bond_sym));
            }
        }
        let mut next: HashMap<usize, String> = HashMap::new();
        for (v, mut from) in reached {
            from.sort();
            let mut ring_closures: Vec<String> = env.edges[v]
                .iter()
                .filter(|e| dist.get(&e.to) == Some(&level))
                .map(|e| format!("{}&", e.bond_sym))
                .collect();
            ring_closures.sort();
            let raw = format!(
                "{}({}){}",
                env.labels[v],
                from.join(","),
                ring_closures.join("")
            );
            next.insert(v, short_hex(&raw));
        }
        let mut sphere_vals: Vec<String> = next.values().cloned().collect();
        sphere_vals.sort();
        spheres.push(sphere_vals.join(","));
        out.push(format!("{head}({})", spheres.join("/")));
        desc = next;
    }
    out
}

/// Level `0..=HOSE_LEVELS` codes for `center` (`codes` in `nmr.ts`).
pub fn all_hose_codes(env: &TopologicalEnv, center: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(HOSE_LEVELS + 1);
    let hat = if env.eta_metal[center].is_some() {
        "^"
    } else {
        ""
    };
    v.push(format!(
        "{}H{}{hat}",
        env.labels[center], env.atoms[center].h
    ));
    v.extend(hose_codes(env, center, HOSE_LEVELS));
    v
}

// --- Functional-group & Substituent Additivity Chemical Shift Engine ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CarbonHybrid {
    Sp3,
    Sp2Alkene,
    Sp2Aromatic,
    Sp2Carbonyl,
    SpAlkyne,
    SpNitrile,
    SpAllene,
    HaptoCp,
    HaptoArene,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubstituentKind {
    Alkyl,
    Aryl,
    Alkenyl,
    Alkynyl,
    CarbonylKetoneAld,
    CarbonylAcidEsterAmide,
    Nitrile,
    Nitro,
    Sulfonyl,
    Sulfinyl,
    Thio,
    Hydroxy,
    Alkoxy,
    Acyloxy,
    Amino,
    Ammonium,
    AmidoN,
    Fluoro,
    Chloro,
    Bromo,
    Iodo,
    Silyl,
    Phosphoryl,
    Boron,
    Other,
}

fn classify_carbon_hybrid(env: &TopologicalEnv, i: usize) -> CarbonHybrid {
    if let Some(m) = env.eta_metal[i] {
        // Count hapto ring size
        let mut ring = HashSet::new();
        let mut q = VecDeque::new();
        ring.insert(i);
        q.push_back(i);
        while let Some(u) = q.pop_front() {
            for e in &env.edges[u] {
                if env.eta_metal[e.to] == Some(m) && ring.insert(e.to) {
                    q.push_back(e.to);
                }
            }
        }
        return if ring.len() == 5 {
            CarbonHybrid::HaptoCp
        } else {
            CarbonHybrid::HaptoArene
        };
    }
    if env.aromatic[i] {
        return CarbonHybrid::Sp2Aromatic;
    }
    let mut n_double = 0usize;
    let mut n_triple = 0usize;
    let mut double_to_hetero_o_s_n = false;
    let mut triple_to_n = false;
    for e in &env.edges[i] {
        let b = &env.bonds[e.bond_idx];
        if b.order == 2 {
            n_double += 1;
            if matches!(env.atoms[e.to].el.as_str(), "O" | "S") {
                double_to_hetero_o_s_n = true;
            }
        } else if b.order == 3 {
            n_triple += 1;
            if env.atoms[e.to].el == "N" {
                triple_to_n = true;
            }
        }
    }
    if n_triple >= 1 {
        if triple_to_n {
            CarbonHybrid::SpNitrile
        } else {
            CarbonHybrid::SpAlkyne
        }
    } else if n_double >= 2 {
        CarbonHybrid::SpAllene
    } else if n_double == 1 {
        if double_to_hetero_o_s_n {
            CarbonHybrid::Sp2Carbonyl
        } else {
            CarbonHybrid::Sp2Alkene
        }
    } else {
        CarbonHybrid::Sp3
    }
}

fn is_carbonyl_carbon(env: &TopologicalEnv, idx: usize) -> bool {
    env.atoms[idx].el == "C"
        && !env.aromatic[idx]
        && env.edges[idx].iter().any(|e| {
            env.bonds[e.bond_idx].order == 2
                && matches!(env.atoms[e.to].el.as_str(), "O" | "S")
                && env.edges[e.to].len() == 1
        })
}

fn is_acid_ester_amide_carbonyl(env: &TopologicalEnv, idx: usize) -> bool {
    if !is_carbonyl_carbon(env, idx) {
        return false;
    }
    env.edges[idx].iter().any(|e| {
        env.bonds[e.bond_idx].order == 1
            && matches!(env.atoms[e.to].el.as_str(), "O" | "N" | "F" | "Cl" | "Br")
    })
}

fn is_nitro_group(env: &TopologicalEnv, idx: usize) -> bool {
    if env.atoms[idx].el != "N" {
        return false;
    }
    let ox_count = env.edges[idx]
        .iter()
        .filter(|e| env.atoms[e.to].el == "O" && env.edges[e.to].len() == 1)
        .count();
    ox_count >= 2
}

/// Classify substituent `sub` attached to `center` by a single bond.
fn classify_substituent(env: &TopologicalEnv, center: usize, sub: usize) -> SubstituentKind {
    let el = env.atoms[sub].el.as_str();
    match el {
        "F" => SubstituentKind::Fluoro,
        "Cl" => SubstituentKind::Chloro,
        "Br" => SubstituentKind::Bromo,
        "I" => SubstituentKind::Iodo,
        "Si" => SubstituentKind::Silyl,
        "B" => SubstituentKind::Boron,
        "P" => SubstituentKind::Phosphoryl,
        "O" => {
            if env.atoms[sub].h >= 1 || env.atoms[sub].charge < 0 {
                SubstituentKind::Hydroxy
            } else if env.edges[sub]
                .iter()
                .any(|e| e.to != center && is_carbonyl_carbon(env, e.to))
            {
                SubstituentKind::Acyloxy
            } else {
                SubstituentKind::Alkoxy
            }
        }
        "N" => {
            if is_nitro_group(env, sub) {
                SubstituentKind::Nitro
            } else if env.atoms[sub].charge > 0
                && env.edges[sub].len() + env.atoms[sub].h as usize >= 4
            {
                SubstituentKind::Ammonium
            } else if env.edges[sub]
                .iter()
                .any(|e| e.to != center && is_carbonyl_carbon(env, e.to))
            {
                SubstituentKind::AmidoN
            } else {
                SubstituentKind::Amino
            }
        }
        "S" => {
            let ox = env.edges[sub]
                .iter()
                .filter(|e| env.atoms[e.to].el == "O" && env.edges[e.to].len() == 1)
                .count();
            if ox >= 2 {
                SubstituentKind::Sulfonyl
            } else if ox == 1 {
                SubstituentKind::Sulfinyl
            } else {
                SubstituentKind::Thio
            }
        }
        "C" => match classify_carbon_hybrid(env, sub) {
            CarbonHybrid::Sp2Aromatic => SubstituentKind::Aryl,
            CarbonHybrid::Sp2Alkene | CarbonHybrid::SpAllene => SubstituentKind::Alkenyl,
            CarbonHybrid::SpAlkyne => SubstituentKind::Alkynyl,
            CarbonHybrid::SpNitrile => SubstituentKind::Nitrile,
            CarbonHybrid::Sp2Carbonyl => {
                if is_acid_ester_amide_carbonyl(env, sub) {
                    SubstituentKind::CarbonylAcidEsterAmide
                } else {
                    SubstituentKind::CarbonylKetoneAld
                }
            }
            _ => SubstituentKind::Alkyl,
        },
        _ => SubstituentKind::Other,
    }
}

/// Aromatic substituent increments `(d_ipso, d_ortho, d_meta, d_para)` for ¹³C (base 128.5 ppm)
/// and `(h_ortho, h_meta, h_para)` for ¹H (base 7.27 ppm), from Pretsch et al. (2009).
fn benzene_substituent_increments(
    env: &TopologicalEnv,
    ring_c: usize,
    sub: usize,
) -> ([f64; 4], [f64; 3]) {
    match classify_substituent(env, ring_c, sub) {
        SubstituentKind::Alkyl => {
            let c_deg = env.edges[sub].len();
            match c_deg {
                1 => ([9.3, 0.7, -0.1, -2.9], [-0.15, -0.10, -0.18]), // -CH3
                2 => ([15.6, -0.4, -0.1, -2.6], [-0.14, -0.06, -0.17]), // -CH2R (e.g. -CH2iPr)
                3 => ([20.1, -2.0, -0.0, -2.5], [-0.10, -0.04, -0.14]), // -CHR2 (e.g. -CH(Me)COOH ~12.5 if COOH on alpha)
                _ => ([22.4, -3.1, -0.2, -2.9], [0.02, -0.08, -0.16]),  // -CR3
            }
        }
        SubstituentKind::Aryl => ([13.1, -1.1, 0.4, -1.2], [0.30, 0.12, 0.10]),
        SubstituentKind::Alkenyl => ([9.1, -2.3, 0.2, -0.8], [0.13, 0.04, -0.05]),
        SubstituentKind::Alkynyl => ([-6.2, 3.6, -0.4, -0.3], [0.19, 0.02, 0.05]),
        SubstituentKind::CarbonylKetoneAld => {
            if env.atoms[sub].h >= 1 {
                ([8.2, 1.2, 0.6, 5.8], [0.58, 0.21, 0.27]) // -CHO
            } else {
                ([8.9, 0.1, -0.1, 4.4], [0.62, 0.14, 0.21]) // -COR (acetophenone)
            }
        }
        SubstituentKind::CarbonylAcidEsterAmide => {
            let has_oh = env.edges[sub]
                .iter()
                .any(|e| env.atoms[e.to].el == "O" && env.atoms[e.to].h >= 1);
            let has_n = env.edges[sub]
                .iter()
                .any(|e| e.to != ring_c && env.atoms[e.to].el == "N");
            if has_oh {
                ([2.1, 1.6, -0.1, 5.2], [0.80, 0.16, 0.25]) // -COOH
            } else if has_n {
                ([5.0, -1.2, 0.1, 3.4], [0.55, 0.12, 0.18]) // -CONR2
            } else {
                ([2.0, 1.2, -0.1, 4.3], [0.74, 0.14, 0.20]) // -COOR
            }
        }
        SubstituentKind::Nitrile => ([-15.7, 3.6, 0.7, 4.3], [0.36, 0.18, 0.28]),
        SubstituentKind::Nitro => ([19.9, -4.9, 0.9, 6.1], [0.95, 0.26, 0.38]),
        SubstituentKind::Sulfonyl => ([12.3, -1.4, 0.8, 5.1], [0.68, 0.25, 0.32]),
        SubstituentKind::Sulfinyl => ([17.6, -5.0, 1.1, 2.4], [0.40, 0.16, 0.18]),
        SubstituentKind::Thio => ([10.1, -1.6, 0.2, -3.5], [0.05, -0.06, -0.12]),
        SubstituentKind::Hydroxy => ([26.9, -12.8, 1.4, -7.4], [-0.53, -0.14, -0.42]),
        SubstituentKind::Alkoxy => ([31.4, -14.4, 1.0, -7.7], [-0.45, -0.09, -0.37]),
        SubstituentKind::Acyloxy => ([22.4, -7.1, 0.4, -3.2], [-0.21, 0.08, -0.08]),
        SubstituentKind::Amino => {
            if env.atoms[sub].h == 2 {
                ([18.2, -13.4, 0.8, -10.0], [-0.65, -0.18, -0.58]) // -NH2
            } else {
                ([21.5, -16.0, 0.7, -11.8], [-0.66, -0.16, -0.62]) // -NHR / -NR2
            }
        }
        SubstituentKind::Ammonium => ([0.1, -5.8, 2.2, 2.0], [0.40, 0.26, 0.25]),
        SubstituentKind::AmidoN => ([9.7, -8.1, 0.2, -4.4], [0.24, 0.02, -0.18]),
        SubstituentKind::Fluoro => ([34.8, -12.9, 1.4, -4.5], [-0.30, -0.02, -0.22]),
        SubstituentKind::Chloro => ([6.3, 0.4, 1.4, -1.9], [0.03, -0.02, -0.09]),
        SubstituentKind::Bromo => ([-5.8, 3.2, 1.6, -1.6], [0.21, -0.08, -0.04]),
        SubstituentKind::Iodo => ([-34.1, 8.9, 1.6, -1.1], [0.40, -0.25, -0.03]),
        SubstituentKind::Silyl => ([11.7, 4.9, -0.7, 0.4], [0.22, 0.06, 0.04]),
        SubstituentKind::Phosphoryl => ([4.0, 3.5, 0.2, 3.2], [0.45, 0.15, 0.20]),
        SubstituentKind::Boron => ([3.0, 6.4, -0.6, 2.4], [0.52, 0.11, 0.16]),
        SubstituentKind::Other => ([2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    }
}

/// Predict ¹³C chemical shift `(ppm, sd_ppm, level)` for carbon `c` using substituent additivity.
fn predict_c13_additivity(env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    match classify_carbon_hybrid(env, c) {
        CarbonHybrid::HaptoCp => {
            let sub_count = env.edges[c]
                .iter()
                .filter(|e| e.bond_sym != "~" && e.bond_sym != "^")
                .count();
            let shift = 68.5 + (sub_count as f64) * 10.5;
            (shift, 1.2, 5)
        }
        CarbonHybrid::HaptoArene => (86.0, 1.8, 4),
        CarbonHybrid::SpNitrile => {
            let mut shift = 118.2;
            for e in &env.edges[c] {
                if env.bonds[e.bond_idx].order == 1 && env.aromatic[e.to] {
                    shift = 118.8;
                }
            }
            (shift, 1.2, 5)
        }
        CarbonHybrid::SpAlkyne => {
            let mut shift = if env.atoms[c].h >= 1 { 71.5 } else { 80.5 };
            for e in &env.edges[c] {
                if env.bonds[e.bond_idx].order == 1 {
                    match classify_substituent(env, c, e.to) {
                        SubstituentKind::Alkyl => shift += 9.5,
                        SubstituentKind::Aryl => shift += 12.0,
                        SubstituentKind::CarbonylKetoneAld
                        | SubstituentKind::CarbonylAcidEsterAmide => shift += 4.5,
                        SubstituentKind::Silyl => shift += 18.0,
                        _ => shift += 5.0,
                    }
                }
            }
            (shift, 1.5, 5)
        }
        CarbonHybrid::SpAllene => {
            let is_central = env.edges[c]
                .iter()
                .filter(|e| env.bonds[e.bond_idx].order == 2)
                .count()
                >= 2;
            if is_central {
                (208.0, 2.0, 4)
            } else {
                (76.0, 2.0, 4)
            }
        }
        CarbonHybrid::Sp2Carbonyl => {
            // Check hetero/aryl/alkyl neighbors on the carbonyl carbon
            let mut has_thio = false;
            let mut n_oh = 0;
            let mut n_or = 0;
            let mut n_nr2 = 0;
            let mut n_hal = 0;
            let mut n_aryl = 0;
            let mut n_alkenyl = 0;
            let mut n_alkyl = 0;
            let mut alpha_amino = false;
            for e in &env.edges[c] {
                let b = &env.bonds[e.bond_idx];
                if b.order == 2 {
                    if env.atoms[e.to].el == "S" {
                        has_thio = true;
                    }
                    continue;
                }
                match classify_substituent(env, c, e.to) {
                    SubstituentKind::Hydroxy => n_oh += 1,
                    SubstituentKind::Alkoxy | SubstituentKind::Acyloxy => n_or += 1,
                    SubstituentKind::Amino | SubstituentKind::AmidoN => n_nr2 += 1,
                    SubstituentKind::Fluoro
                    | SubstituentKind::Chloro
                    | SubstituentKind::Bromo
                    | SubstituentKind::Iodo => n_hal += 1,
                    SubstituentKind::Aryl => n_aryl += 1,
                    SubstituentKind::Alkenyl | SubstituentKind::Alkynyl => n_alkenyl += 1,
                    SubstituentKind::Alkyl => {
                        n_alkyl += 1;
                        if env.edges[e.to].iter().any(|x| {
                            x.to != c
                                && matches!(
                                    classify_substituent(env, e.to, x.to),
                                    SubstituentKind::Amino | SubstituentKind::Ammonium
                                )
                        }) {
                            alpha_amino = true;
                        }
                    }
                    _ => {}
                }
            }
            let mut shift = if n_or + n_oh >= 2 {
                155.0 // carbonate / carbonic acid
            } else if (n_or + n_oh == 1) && n_nr2 == 1 {
                156.5 // carbamate
            } else if n_nr2 >= 2 {
                158.5 // urea
            } else if n_oh == 1 {
                if alpha_amino {
                    176.5 // alpha-amino acid COOH / COO-
                } else if n_aryl >= 1 {
                    172.0 // benzoic acid
                } else if env.atoms[c].h >= 1 {
                    166.3 // formic acid
                } else {
                    178.5 // aliphatic carboxylic acid (e.g. ibuprofen ~180.5 with branching)
                }
            } else if n_or == 1 {
                if n_aryl >= 1 {
                    166.5 // benzoate ester
                } else if env.atoms[c].h >= 1 {
                    161.0 // formate ester
                } else {
                    171.2 // aliphatic ester (ethyl acetate = 171.2)
                }
            } else if n_nr2 == 1 {
                if n_aryl >= 1 {
                    168.0
                } else if env.atoms[c].h >= 1 {
                    162.5
                } else {
                    172.5
                }
            } else if n_hal >= 1 {
                170.5 // acyl halide
            } else if env.atoms[c].h >= 1 {
                if n_aryl >= 1 {
                    192.0 // benzaldehyde
                } else if n_alkenyl >= 1 {
                    194.0
                } else {
                    201.5 // aliphatic aldehyde
                }
            } else {
                // Ketone
                if n_aryl >= 2 {
                    196.5 // benzophenone
                } else if n_aryl == 1 {
                    197.4 // acetophenone / 4-bromoacetophenone ~197.0
                } else if n_alkenyl >= 1 {
                    198.5 // enone
                } else {
                    207.5 // dialkyl ketone
                }
            };
            // Slight branching adjustment on aliphatic acids/ketones
            if (n_oh == 1 || (n_oh == 0 && n_or == 0 && n_nr2 == 0)) && n_alkyl >= 1 && !alpha_amino
            {
                for e in &env.edges[c] {
                    if env.bonds[e.bond_idx].order == 1
                        && env.atoms[e.to].el == "C"
                        && !env.aromatic[e.to]
                    {
                        let branch = env.edges[e.to].len().saturating_sub(1);
                        if branch >= 2 {
                            shift += 2.0;
                        }
                    }
                }
            }
            if has_thio {
                shift += 25.0;
            }
            (shift, 1.0, 5)
        }
        CarbonHybrid::Sp2Alkene => {
            // Pascual-Meier-Simon / Pretsch alkene 13C additivity (base 123.3 ppm)
            let mut shift = 123.3;
            let dbl = env.edges[c]
                .iter()
                .find(|e| env.bonds[e.bond_idx].order == 2)
                .map(|e| e.to);
            for e in &env.edges[c] {
                if Some(e.to) == dbl {
                    continue;
                }
                shift += match classify_substituent(env, c, e.to) {
                    SubstituentKind::Alkyl => {
                        let extra_c = env.edges[e.to]
                            .iter()
                            .filter(|x| x.to != c && env.atoms[x.to].el == "C")
                            .count();
                        10.6 + (extra_c as f64) * 4.5
                    }
                    SubstituentKind::Aryl => 12.5,
                    SubstituentKind::Alkenyl => 13.6,
                    SubstituentKind::CarbonylKetoneAld => 15.0,
                    SubstituentKind::CarbonylAcidEsterAmide => 6.0,
                    SubstituentKind::Nitrile => -15.0,
                    SubstituentKind::Hydroxy | SubstituentKind::Alkoxy => 29.0,
                    SubstituentKind::Acyloxy => 18.0,
                    SubstituentKind::Amino => 28.0,
                    SubstituentKind::Chloro => 2.8,
                    SubstituentKind::Bromo => -8.0,
                    SubstituentKind::Iodo => -38.0,
                    SubstituentKind::Fluoro => 24.9,
                    SubstituentKind::Nitro => 22.0,
                    _ => 5.0,
                };
            }
            if let Some(other_c) = dbl {
                for e in &env.edges[other_c] {
                    if e.to == c {
                        continue;
                    }
                    shift += match classify_substituent(env, other_c, e.to) {
                        SubstituentKind::Alkyl => -7.9,
                        SubstituentKind::Aryl => -11.0,
                        SubstituentKind::Alkenyl => -7.0,
                        SubstituentKind::CarbonylKetoneAld => 13.5,
                        SubstituentKind::CarbonylAcidEsterAmide => 9.5,
                        SubstituentKind::Nitrile => 15.0,
                        SubstituentKind::Hydroxy | SubstituentKind::Alkoxy => -38.0,
                        SubstituentKind::Acyloxy => -27.0,
                        SubstituentKind::Amino => -32.0,
                        SubstituentKind::Chloro => -6.0,
                        SubstituentKind::Bromo => -1.5,
                        SubstituentKind::Iodo => 7.0,
                        SubstituentKind::Fluoro => -34.0,
                        SubstituentKind::Nitro => -5.0,
                        _ => -2.0,
                    };
                }
            }
            (shift, 1.5, 5)
        }
        CarbonHybrid::Sp2Aromatic => predict_aromatic_c13(env, c),
        CarbonHybrid::Sp3 => predict_sp3_c13(env, c),
    }
}

fn find_aromatic_6ring_containing(env: &TopologicalEnv, c: usize) -> Option<Vec<usize>> {
    // DFS for a 6-membered all-aromatic cycle containing `c`
    let mut path = vec![c];
    fn dfs(env: &TopologicalEnv, start: usize, cur: usize, path: &mut Vec<usize>) -> bool {
        if path.len() == 6 {
            return env.edges[cur]
                .iter()
                .any(|e| e.to == start && e.bond_sym == "*");
        }
        for e in &env.edges[cur] {
            if e.bond_sym == "*" && env.aromatic[e.to] && !path.contains(&e.to) {
                path.push(e.to);
                if dfs(env, start, e.to, path) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    if dfs(env, c, c, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn find_aromatic_5ring_containing(env: &TopologicalEnv, c: usize) -> Option<Vec<usize>> {
    let mut path = vec![c];
    fn dfs(env: &TopologicalEnv, start: usize, cur: usize, path: &mut Vec<usize>) -> bool {
        if path.len() == 5 {
            return env.edges[cur]
                .iter()
                .any(|e| e.to == start && e.bond_sym == "*");
        }
        for e in &env.edges[cur] {
            if e.bond_sym == "*" && env.aromatic[e.to] && !path.contains(&e.to) {
                path.push(e.to);
                if dfs(env, start, e.to, path) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    if dfs(env, c, c, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn predict_aromatic_c13(env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    if let Some(ring6) = find_aromatic_6ring_containing(env, c) {
        let mut shift = 128.5;
        // Pyridine / diazine ring-nitrogen increments relative to benzene (128.5):
        // alpha (+21.4 -> 149.9), beta (-4.7 -> 123.8), gamma (+7.4 -> 135.9)
        for (pos, &rv) in ring6.iter().enumerate() {
            let d_steps = pos.min(6 - pos); // 0=ipso, 1=ortho, 2=meta, 3=para
            if env.atoms[rv].el == "N" {
                shift += match d_steps {
                    1 => 21.4,
                    2 => -4.7,
                    3 => 7.4,
                    _ => 0.0,
                };
                continue;
            }
            // Exocyclic substituents on `rv`
            for e in &env.edges[rv] {
                if !ring6.contains(&e.to) {
                    // Check if e.to has an alpha-polar group on benzylic carbon (e.g. ibuprofen -CH(Me)COOH)
                    let (mut inc_c, _) = benzene_substituent_increments(env, rv, e.to);
                    if env.atoms[e.to].el == "C" && !env.aromatic[e.to] {
                        let has_alpha_ewg = env.edges[e.to].iter().any(|x| {
                            x.to != rv
                                && matches!(
                                    classify_substituent(env, e.to, x.to),
                                    SubstituentKind::CarbonylAcidEsterAmide
                                        | SubstituentKind::CarbonylKetoneAld
                                        | SubstituentKind::Nitrile
                                        | SubstituentKind::Hydroxy
                                )
                        });
                        if has_alpha_ewg {
                            inc_c[0] -= 7.8; // ipso ~140.8 -> ~137.2 for Ar-CH(Me)COOH
                            inc_c[1] += 0.8;
                        }
                    }
                    shift += inc_c[d_steps];
                }
            }
        }
        return (shift, 1.0, 5);
    }

    if let Some(ring5) = find_aromatic_5ring_containing(env, c) {
        // Pyrrole / furan / thiophene / azole
        let mut base = 125.0;
        for (pos, &rv) in ring5.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let d = pos.min(5 - pos); // 1 = alpha to heteroatom, 2 = beta to heteroatom
            match env.atoms[rv].el.as_str() {
                "O" => {
                    base = if d == 1 { 142.8 } else { 109.6 };
                }
                "S" => {
                    base = if d == 1 { 125.4 } else { 127.2 };
                }
                "N" => {
                    if env.atoms[rv].h >= 1 || env.edges[rv].len() == 3 {
                        base = if d == 1 { 118.2 } else { 107.9 };
                    } else {
                        base += if d == 1 { 18.0 } else { -2.0 };
                    }
                }
                _ => {}
            }
        }
        for e in &env.edges[c] {
            if !ring5.contains(&e.to) {
                let (inc_c, _) = benzene_substituent_increments(env, c, e.to);
                base += inc_c[0] * 0.75;
            }
        }
        return (base, 1.8, 4);
    }

    (128.5, 2.0, 3)
}

fn predict_sp3_c13(env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    // Base chemical shift + Grant-Paul & Pretsch alpha/beta/gamma substituent increments
    let mut shift = -2.3;
    let mut n_alpha_c = 0usize;
    let mut n_beta_c = 0usize;
    let mut n_gamma_c = 0usize;

    let mut visited_beta: HashSet<usize> = HashSet::new();
    visited_beta.insert(c);

    for e1 in &env.edges[c] {
        let a = e1.to;
        visited_beta.insert(a);
        match classify_substituent(env, c, a) {
            SubstituentKind::Alkyl => {
                n_alpha_c += 1;
                shift += 9.1;
            }
            SubstituentKind::Aryl => {
                n_alpha_c += 1;
                shift += 22.8;
            }
            SubstituentKind::Alkenyl => {
                n_alpha_c += 1;
                shift += 19.5;
            }
            SubstituentKind::Alkynyl => {
                n_alpha_c += 1;
                shift += 4.5;
            }
            SubstituentKind::CarbonylKetoneAld => {
                shift += 30.0;
            }
            SubstituentKind::CarbonylAcidEsterAmide => {
                shift += 22.5;
            }
            SubstituentKind::Nitrile => {
                shift += 3.2;
            }
            SubstituentKind::Hydroxy => {
                shift += 50.5;
            }
            SubstituentKind::Alkoxy => {
                shift += 51.5;
            }
            SubstituentKind::Acyloxy => {
                shift += 53.5;
            }
            SubstituentKind::Amino | SubstituentKind::Ammonium => {
                shift += 30.0;
            }
            SubstituentKind::AmidoN => {
                shift += 31.5;
            }
            SubstituentKind::Nitro => {
                shift += 63.5;
            }
            SubstituentKind::Sulfonyl => {
                shift += 41.0;
            }
            SubstituentKind::Sulfinyl => {
                shift += 38.0;
            }
            SubstituentKind::Thio => {
                shift += 11.5;
            }
            SubstituentKind::Fluoro => {
                shift += 70.0;
            }
            SubstituentKind::Chloro => {
                shift += 31.0;
            }
            SubstituentKind::Bromo => {
                shift += 19.5;
            }
            SubstituentKind::Iodo => {
                shift += -6.0;
            }
            SubstituentKind::Silyl => {
                shift += -2.5;
            }
            _ => {
                shift += 8.0;
            }
        }
    }

    // Beta & gamma effects along sp3 / functional chains
    for e1 in &env.edges[c] {
        let a = e1.to;
        let a_kind = classify_substituent(env, c, a);
        if matches!(
            a_kind,
            SubstituentKind::Aryl
                | SubstituentKind::CarbonylKetoneAld
                | SubstituentKind::CarbonylAcidEsterAmide
                | SubstituentKind::Nitro
                | SubstituentKind::Nitrile
        ) {
            // Do not walk through conjugated pi systems as simple alkane beta/gamma carbons;
            // apply dedicated beta effect if `a` is carbonyl/ester oxygen etc.
            continue;
        }
        for e2 in &env.edges[a] {
            let b = e2.to;
            if b == c {
                continue;
            }
            match classify_substituent(env, a, b) {
                SubstituentKind::Alkyl => {
                    n_beta_c += 1;
                    shift += 9.4;
                }
                SubstituentKind::Aryl => {
                    n_beta_c += 1;
                    shift += 9.5;
                }
                SubstituentKind::Alkenyl => {
                    n_beta_c += 1;
                    shift += 7.0;
                }
                SubstituentKind::CarbonylKetoneAld => shift += 2.0,
                SubstituentKind::CarbonylAcidEsterAmide => shift += 2.2,
                SubstituentKind::Hydroxy | SubstituentKind::Alkoxy => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 9.8;
                    } else if a_kind == SubstituentKind::Alkoxy
                        || a_kind == SubstituentKind::Acyloxy
                    {
                        shift -= 2.0; // e.g. O-CH2-CH3 vs O-CH3
                    }
                }
                SubstituentKind::Acyloxy => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 6.5;
                    }
                }
                SubstituentKind::Amino | SubstituentKind::Ammonium | SubstituentKind::AmidoN => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 8.0;
                    }
                }
                SubstituentKind::Chloro | SubstituentKind::Bromo => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 10.0;
                    }
                }
                SubstituentKind::Fluoro => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 7.8;
                    }
                }
                SubstituentKind::Nitro => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 3.0;
                    }
                }
                _ => {
                    if a_kind == SubstituentKind::Alkyl {
                        shift += 5.0;
                    }
                }
            }

            // Gamma steric gauche effect (-2.5 ppm per gamma heavy atom along alkyl chain)
            if a_kind == SubstituentKind::Alkyl && env.atoms[b].el == "C" && !env.aromatic[b] {
                for e3 in &env.edges[b] {
                    if e3.to != a && e3.to != c {
                        n_gamma_c += 1;
                    }
                }
            }
        }
    }

    shift -= (n_gamma_c as f64) * 2.5;

    // Grant-Paul branching steric crowding correction
    if n_alpha_c >= 2 && n_beta_c >= 1 {
        let corr = match (n_alpha_c, n_beta_c) {
            (2, 1) => -1.1,
            (2, 2) => -2.5,
            (2, _) => -3.5,
            (3, 1) => -3.7,
            (3, _) => -5.5,
            (4, _) => -7.5,
            _ => 0.0,
        };
        shift += corr;
    }

    // Special well-known methyl-ketone / acetate methyl calibration:
    // CH3-C(=O)O-R (~21.0 ppm), CH3-C(=O)-Ar (~26.5 ppm), CH3-C(=O)-R (~29.5 ppm)
    if env.edges[c].len() == 1 {
        let nb = env.edges[c][0].to;
        if is_carbonyl_carbon(env, nb) {
            if is_acid_ester_amide_carbonyl(env, nb) {
                shift = 21.0;
            } else if env.edges[nb]
                .iter()
                .any(|x| x.to != c && env.aromatic[x.to])
            {
                shift = 26.5;
            } else if env.atoms[nb].h >= 1 {
                shift = 30.8;
            } else {
                shift = 29.5;
            }
        }
    }

    // Small-ring strain corrections (cyclopropane ~ -2.8 ppm)
    let in_3ring = env.edges[c].iter().any(|e1| {
        env.edges[e1.to]
            .iter()
            .any(|e2| env.edges[c].iter().any(|e3| e3.to == e2.to))
    });
    if in_3ring {
        shift -= 18.0;
    }

    (shift.max(-5.0), 1.2, 5)
}

/// Predict ¹H chemical shift `(ppm, sd_ppm, level)` for protons attached to heavy atom `c`.
fn predict_h1_additivity(
    mol: &AnalytMol,
    env: &TopologicalEnv,
    c: usize,
    solvent: Option<&str>,
) -> (f64, f64, usize) {
    let el = env.atoms[c].el.as_str();
    let is_dmso = solvent.is_some_and(|s| s.to_ascii_uppercase().contains("DMSO"));

    match el {
        "O" => {
            // Carboxylic acid, phenol, or aliphatic alcohol
            let nb = env.edges[c].first().map(|e| e.to);
            if let Some(n) = nb {
                if is_carbonyl_carbon(env, n) {
                    return (if is_dmso { 12.20 } else { 11.40 }, 0.35, 5);
                }
                if env.aromatic[n] {
                    return (if is_dmso { 9.35 } else { 5.35 }, 0.30, 5);
                }
            }
            return (if is_dmso { 4.55 } else { 2.45 }, 0.25, 4);
        }
        "N" => {
            if env.atoms[c].charge > 0 {
                return (7.85, 0.30, 4);
            }
            if env.aromatic[c] {
                // Pyrrole / indole NH
                return (if is_dmso { 10.80 } else { 8.40 }, 0.25, 5);
            }
            if env.edges[c].iter().any(|e| is_carbonyl_carbon(env, e.to)) {
                // Amide NH
                return (if is_dmso { 7.90 } else { 6.35 }, 0.25, 5);
            }
            if env.edges[c].iter().any(|e| env.aromatic[e.to]) {
                // Aniline NH2
                return (if is_dmso { 5.05 } else { 3.65 }, 0.20, 5);
            }
            return (1.65, 0.25, 4);
        }
        "S" => {
            let aryl = env.edges[c].iter().any(|e| env.aromatic[e.to]);
            return (if aryl { 3.45 } else { 1.55 }, 0.20, 4);
        }
        "C" => {}
        _ => return (2.00, 0.40, 2),
    }

    match classify_carbon_hybrid(env, c) {
        CarbonHybrid::HaptoCp => (4.16, 0.08, 6), // Ferrocene / Cp ~4.15 ppm
        CarbonHybrid::HaptoArene => (5.30, 0.15, 4),
        CarbonHybrid::SpAlkyne => {
            let mut shift = 1.95;
            for e in &env.edges[c] {
                if env.bonds[e.bond_idx].order == 3 {
                    let other_c = e.to;
                    for e2 in &env.edges[other_c] {
                        if e2.to != c {
                            match classify_substituent(env, other_c, e2.to) {
                                SubstituentKind::Aryl => shift = 3.07,
                                SubstituentKind::CarbonylKetoneAld
                                | SubstituentKind::CarbonylAcidEsterAmide => shift = 3.20,
                                _ => {}
                            }
                        }
                    }
                }
            }
            (shift, 0.10, 5)
        }
        CarbonHybrid::SpAllene => (4.85, 0.15, 4),
        CarbonHybrid::SpNitrile => (2.0, 0.20, 3),
        CarbonHybrid::Sp2Carbonyl => {
            // Aldehyde or formyl H
            let mut shift = 9.76;
            for e in &env.edges[c] {
                if env.bonds[e.bond_idx].order == 1 {
                    match classify_substituent(env, c, e.to) {
                        SubstituentKind::Aryl => shift = 10.00,
                        SubstituentKind::Alkenyl => shift = 9.55,
                        SubstituentKind::Alkoxy => shift = 8.08, // formate
                        SubstituentKind::Amino | SubstituentKind::AmidoN => shift = 8.02, // formamide
                        SubstituentKind::Hydroxy => shift = 8.06, // formic acid
                        _ => {}
                    }
                }
            }
            (shift, 0.08, 6)
        }
        CarbonHybrid::Sp2Alkene => predict_alkene_h1(mol, env, c),
        CarbonHybrid::Sp2Aromatic => predict_aromatic_h1(env, c),
        CarbonHybrid::Sp3 => predict_sp3_h1(env, c),
    }
}

fn predict_alkene_h1(mol: &AnalytMol, env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    // Pascual-Meier-Simon rule: delta = 5.25 + Z_gem + Z_cis + Z_trans
    let mut shift = 5.25;
    let dbl_nb = env.edges[c]
        .iter()
        .find(|e| env.bonds[e.bond_idx].order == 2)
        .map(|e| e.to);

    let pms_inc = |kind: SubstituentKind| -> (f64, f64, f64) {
        // (Z_gem, Z_cis, Z_trans)
        match kind {
            SubstituentKind::Alkyl => (0.45, -0.22, -0.28),
            SubstituentKind::Aryl => (1.38, 0.36, -0.07),
            SubstituentKind::Alkenyl => (1.00, -0.09, -0.23),
            SubstituentKind::Alkynyl => (0.47, 0.38, 0.12),
            SubstituentKind::CarbonylKetoneAld => (1.10, 0.98, 0.32),
            SubstituentKind::CarbonylAcidEsterAmide => (0.80, 0.98, 0.32),
            SubstituentKind::Nitrile => (0.27, 0.75, 0.55),
            SubstituentKind::Hydroxy | SubstituentKind::Alkoxy => (1.22, -1.07, -1.21),
            SubstituentKind::Acyloxy => (2.11, -0.35, -0.64),
            SubstituentKind::Amino | SubstituentKind::AmidoN => (0.80, -1.26, -1.21),
            SubstituentKind::Chloro => (1.08, 0.18, 0.13),
            SubstituentKind::Bromo => (1.07, 0.45, 0.55),
            SubstituentKind::Fluoro => (1.54, -0.40, -1.02),
            SubstituentKind::Iodo => (1.14, 0.81, 0.88),
            SubstituentKind::Nitro => (1.87, 1.30, 0.62),
            _ => (0.30, 0.0, 0.0),
        }
    };

    for e in &env.edges[c] {
        if Some(e.to) == dbl_nb {
            continue;
        }
        let (zg, _, _) = pms_inc(classify_substituent(env, c, e.to));
        shift += zg;
    }

    if let Some(other) = dbl_nb {
        let other_subs: Vec<usize> = env.edges[other]
            .iter()
            .filter(|e| e.to != c)
            .map(|e| e.to)
            .collect();
        if other_subs.len() == 2 {
            // One is cis, one is trans -> sum of cis + trans if unassigned, or oriented
            let (_, zc0, zt0) = pms_inc(classify_substituent(env, other, other_subs[0]));
            let (_, zc1, zt1) = pms_inc(classify_substituent(env, other, other_subs[1]));
            shift += 0.5 * (zc0 + zt0 + zc1 + zt1);
        } else if other_subs.len() == 1 {
            let sub = other_subs[0];
            let (_, z_cis, z_trans) = pms_inc(classify_substituent(env, other, sub));
            // Check if H on `c` and `sub` on `other` are cis or trans!
            // If `c` also has 1 heavy sub `sub_c`, then if `sub_c` and `sub` are trans across `c=other`,
            // H on `c` is CIS to `sub`! And if `sub_c` and `sub` are cis, H on `c` is TRANS to `sub`!
            let c_subs: Vec<usize> = env.edges[c]
                .iter()
                .filter(|e| e.to != other)
                .map(|e| e.to)
                .collect();
            let rel = if c_subs.len() == 1 {
                alkene_heavy_subs_trans(
                    mol,
                    env.from_mol[c],
                    env.from_mol[other],
                    env.from_mol[c_subs[0]],
                    env.from_mol[sub],
                )
            } else {
                None
            };
            match rel {
                Some(true) => shift += z_cis, // heavy subs trans => H(c) is cis to sub(other)
                Some(false) => shift += z_trans, // heavy subs cis => H(c) is trans to sub(other)
                None => shift += 0.5 * (z_cis + z_trans),
            }
        }
    }

    (shift, 0.10, 5)
}

fn predict_aromatic_h1(env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    if let Some(ring6) = find_aromatic_6ring_containing(env, c) {
        let mut shift = 7.27;
        for (pos, &rv) in ring6.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let d = pos.min(6 - pos); // 1=ortho, 2=meta, 3=para
            if env.atoms[rv].el == "N" {
                shift += match d {
                    1 => 1.33, // pyridine H2 ~ 8.60
                    2 => 0.01, // pyridine H3 ~ 7.28
                    3 => 0.41, // pyridine H4 ~ 7.68
                    _ => 0.0,
                };
                continue;
            }
            for e in &env.edges[rv] {
                if !ring6.contains(&e.to) {
                    let (_, mut inc_h) = benzene_substituent_increments(env, rv, e.to);
                    if env.atoms[e.to].el == "C" && !env.aromatic[e.to] {
                        let has_alpha_ewg = env.edges[e.to].iter().any(|x| {
                            x.to != rv
                                && matches!(
                                    classify_substituent(env, e.to, x.to),
                                    SubstituentKind::CarbonylAcidEsterAmide
                                        | SubstituentKind::CarbonylKetoneAld
                                        | SubstituentKind::Nitrile
                                        | SubstituentKind::Hydroxy
                                )
                        });
                        if has_alpha_ewg {
                            inc_h[0] += 0.12; // e.g. ibuprofen ortho to -CH(Me)COOH ~7.22 vs ~7.10
                            inc_h[1] += 0.03;
                        }
                    }
                    shift += inc_h[d - 1];
                }
            }
        }
        return (shift, 0.08, 5);
    }

    if let Some(ring5) = find_aromatic_5ring_containing(env, c) {
        let mut shift = 7.10;
        for (pos, &rv) in ring5.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let d = pos.min(5 - pos);
            match env.atoms[rv].el.as_str() {
                "O" => shift = if d == 1 { 7.42 } else { 6.38 },
                "S" => shift = if d == 1 { 7.31 } else { 7.10 },
                "N" => {
                    if env.atoms[rv].h >= 1 || env.edges[rv].len() == 3 {
                        shift = if d == 1 { 6.80 } else { 6.22 };
                    } else {
                        shift += if d == 1 { 0.65 } else { 0.15 };
                    }
                }
                _ => {}
            }
        }
        return (shift, 0.12, 5);
    }

    (7.27, 0.15, 4)
}

fn predict_sp3_h1(env: &TopologicalEnv, c: usize) -> (f64, f64, usize) {
    let h = env.atoms[c].h;
    // Base sp3 shift: CH3 = 0.88, CH2 = 1.25, CH = 1.55 (methane = 0.23)
    let mut shift = match h {
        4 => 0.23,
        3 => 0.88,
        2 => 1.25,
        _ => 1.55,
    };

    // Alpha substituent increments (Curzon / Shoolery / Pretsch)
    let mut non_alkyl_alpha = 0usize;
    for e in &env.edges[c] {
        let a = e.to;
        let kind = classify_substituent(env, c, a);
        if kind != SubstituentKind::Alkyl {
            non_alkyl_alpha += 1;
        }
        let inc = match kind {
            SubstituentKind::Alkyl => 0.0,
            SubstituentKind::Aryl => {
                // Check if aryl bears electron-withdrawing para/ortho groups
                1.38
            }
            SubstituentKind::Alkenyl => 0.78,
            SubstituentKind::Alkynyl => 0.95,
            SubstituentKind::CarbonylKetoneAld => {
                if env.edges[a].iter().any(|x| x.to != c && env.aromatic[x.to]) {
                    1.70 // ArC(=O)-CH3 -> 0.88 + 1.70 = 2.58 ppm!
                } else {
                    1.24 // RC(=O)-CH3 -> 2.12 ppm
                }
            }
            SubstituentKind::CarbonylAcidEsterAmide => {
                let has_or_oh = env.edges[a]
                    .iter()
                    .any(|x| x.to != c && env.atoms[x.to].el == "O");
                if has_or_oh {
                    1.17 // CH3C(=O)OR -> 0.88 + 1.17 = 2.05 ppm!
                } else {
                    1.10 // amide
                }
            }
            SubstituentKind::Nitrile => 1.25,
            SubstituentKind::Hydroxy => 2.42, // CH3CH2OH -> 1.25 + 2.42 = 3.67 ppm!
            SubstituentKind::Alkoxy => 2.28,
            SubstituentKind::Acyloxy => 2.87, // CH3COOCH2CH3 -> 1.25 + 2.87 = 4.12 ppm!
            SubstituentKind::Amino => 1.55,
            SubstituentKind::Ammonium => 2.10,
            SubstituentKind::AmidoN => 1.95,
            SubstituentKind::Nitro => 3.38,
            SubstituentKind::Sulfonyl => 1.95,
            SubstituentKind::Sulfinyl => 1.55,
            SubstituentKind::Thio => 1.25,
            SubstituentKind::Fluoro => 3.35,
            SubstituentKind::Chloro => 2.20,
            SubstituentKind::Bromo => 2.05,
            SubstituentKind::Iodo => 1.85,
            SubstituentKind::Silyl => -0.85,
            _ => 0.40,
        };
        shift += inc;
    }

    // Damping factor when 2+ strong polar/pi groups share the same sp3 carbon (e.g., alpha-amino acid CH(NH2)COOH:
    // 1.55 + (1.55 + 1.17)*0.82 = 3.78 ppm; ibuprofen Ar-CH(Me)COOH: 1.55 + (1.38 + 1.17)*0.85 = 3.72 ppm!)
    if non_alkyl_alpha >= 2 {
        let base = match h {
            3 => 0.88,
            2 => 1.25,
            _ => 1.55,
        };
        let excess = shift - base;
        shift = base + excess * 0.84;
    }

    // Beta substituent increments (attached to an sp3 carbon neighbor `a` of `c`)
    for e in &env.edges[c] {
        let a = e.to;
        if classify_substituent(env, c, a) != SubstituentKind::Alkyl {
            continue;
        }
        for e2 in &env.edges[a] {
            if e2.to == c {
                continue;
            }
            shift += match classify_substituent(env, a, e2.to) {
                SubstituentKind::Alkyl => 0.02,
                SubstituentKind::Aryl => 0.32,
                SubstituentKind::Alkenyl | SubstituentKind::Alkynyl => 0.12,
                SubstituentKind::CarbonylKetoneAld => 0.22,
                SubstituentKind::CarbonylAcidEsterAmide => 0.28,
                SubstituentKind::Nitrile => 0.35,
                SubstituentKind::Hydroxy | SubstituentKind::Alkoxy => 0.34, // CH3-CH2-OH -> 0.88 + 0.34 = 1.22 ppm!
                SubstituentKind::Acyloxy => 0.38, // CH3-CH2-OAc -> 0.88 + 0.38 = 1.26 ppm!
                SubstituentKind::Amino => 0.25,
                SubstituentKind::Ammonium => 0.45,
                SubstituentKind::AmidoN => 0.28,
                SubstituentKind::Nitro => 0.62,
                SubstituentKind::Sulfonyl => 0.42,
                SubstituentKind::Fluoro => 0.42,
                SubstituentKind::Chloro => 0.52,
                SubstituentKind::Bromo => 0.62,
                SubstituentKind::Iodo => 0.68,
                _ => 0.05,
            };
        }
    }

    // Cyclopropane ring current upfield shift
    let in_3ring = env.edges[c].iter().any(|e1| {
        env.edges[e1.to]
            .iter()
            .any(|e2| env.edges[c].iter().any(|e3| e3.to == e2.to))
    });
    if in_3ring {
        shift -= 0.70;
    }

    (shift, 0.08, 5)
}

/// Determine whether two heavy substituents `sub_u` on `u` and `sub_v` on `v` across double bond `u=v`
/// are *trans* (`Some(true)`), *cis* (`Some(false)`), or geometrically unspecified (`None`).
fn alkene_heavy_subs_trans(
    mol: &AnalytMol,
    u: usize,
    v: usize,
    sub_u: usize,
    sub_v: usize,
) -> Option<bool> {
    let dbl_idx = mol.bond_between(u, v)?;
    let dbl = &mol.bonds[dbl_idx];
    match dbl.stereo {
        BondStereoTag::Trans => return Some(true),
        BondStereoTag::Cis => return Some(false),
        BondStereoTag::Either => return None,
        _ => {}
    }

    let bu = mol.bond_between(u, sub_u)?;
    let bv = mol.bond_between(v, sub_v)?;

    let side_on = |center: usize, b_idx: usize| -> Option<i8> {
        let sb = &mol.bonds[b_idx];
        match sb.stereo {
            BondStereoTag::Slash => Some(if sb.a == center { 1 } else { -1 }),
            BondStereoTag::Backslash => Some(if sb.a == center { -1 } else { 1 }),
            _ => None,
        }
    };

    if let (Some(su), Some(sv)) = (side_on(u, bu), side_on(v, bv)) {
        return Some(su != sv);
    }

    // Check 2D coordinates if explicitly provided (not auto-generated from SMILES without `/` `\`)
    if mol.atoms[u].smiles_neighbor_order.is_empty() {
        let ax = mol.atoms[v].x - mol.atoms[u].x;
        let ay = mol.atoms[v].y - mol.atoms[u].y;
        let ux = mol.atoms[sub_u].x - mol.atoms[u].x;
        let uy = mol.atoms[sub_u].y - mol.atoms[u].y;
        let vx = mol.atoms[sub_v].x - mol.atoms[v].x;
        let vy = mol.atoms[sub_v].y - mol.atoms[v].y;
        let cu = ax * uy - ay * ux;
        let cv = ax * vy - ay * vx;
        if (cu * cv).abs() > 1e-6 {
            return Some(cu * cv < 0.0);
        }
    }

    None
}

/// Estimate vicinal `³J_HH` (in Hz) between protons on adjacent heavy atoms `a` and `b`.
fn estimate_vicinal_j(
    mol: &AnalytMol,
    env: &TopologicalEnv,
    a: usize,
    b: usize,
) -> (f64, &'static str) {
    let Some(edge) = env.edges[a].iter().find(|e| e.to == b) else {
        return (7.0, "3J_alkyl");
    };
    if edge.bond_sym == "~" {
        // Hapto ring (Cp = 2.5 Hz, η6-arene = 6.5 Hz)
        let mut ring = HashSet::new();
        let mut q = VecDeque::new();
        ring.insert(a);
        q.push_back(a);
        while let Some(u) = q.pop_front() {
            for x in &env.edges[u] {
                if x.bond_sym == "~" && ring.insert(x.to) {
                    q.push_back(x.to);
                }
            }
        }
        return if ring.len() == 5 {
            (2.5, "3J_hapto")
        } else {
            (6.5, "3J_hapto")
        };
    }
    if edge.bond_sym == "*" {
        return (8.0, "3J_ortho");
    }
    if edge.bond_sym == "=" {
        // Alkene 3J_HH: check cis (~10.5 Hz) vs trans (~15.5 Hz)
        let subs_a: Vec<usize> = env.edges[a]
            .iter()
            .filter(|x| x.to != b)
            .map(|x| x.to)
            .collect();
        let subs_b: Vec<usize> = env.edges[b]
            .iter()
            .filter(|x| x.to != a)
            .map(|x| x.to)
            .collect();
        if subs_a.len() == 1 && subs_b.len() == 1 {
            match alkene_heavy_subs_trans(
                mol,
                env.from_mol[a],
                env.from_mol[b],
                env.from_mol[subs_a[0]],
                env.from_mol[subs_b[0]],
            ) {
                Some(true) => return (15.5, "3J_trans"),
                Some(false) => return (10.5, "3J_cis"),
                None => {}
            }
        }
        return (12.0, "3J_alkene");
    }

    let is_aldehyde = |i: usize| {
        env.edges[i]
            .iter()
            .any(|x| x.bond_sym == "=" && env.labels[x.to] == "O")
            && env.atoms[i].h == 1
    };
    if is_aldehyde(a) || is_aldehyde(b) {
        return (2.5, "3J_ald");
    }

    let is_sp2 = |i: usize| {
        env.edges[i]
            .iter()
            .any(|x| x.bond_sym == "=" || x.bond_sym == "*")
    };
    if is_sp2(a) || is_sp2(b) {
        (7.2, "3J_allylic")
    } else {
        (7.0, "3J_alkyl")
    }
}

/// Compute exact first-order multiplet splitting tree lines normalized to `total_nuclei`.
pub fn compute_multiplet_lines(
    couplings: &[SpinCoupling],
    total_nuclei: usize,
) -> Vec<MultipletLine> {
    let mut tree: BTreeMap<i64, f64> = BTreeMap::new(); // key = offset in 0.001 Hz
    tree.insert(0, total_nuclei as f64);

    for c in couplings {
        if c.count == 0 || c.j_hz <= 0.0 {
            continue;
        }
        let n = c.count;
        // Binomial coefficients C(n, k) / 2^n
        let mut binom = vec![1.0f64; n + 1];
        for k in 1..=n {
            binom[k] = binom[k - 1] * ((n - k + 1) as f64) / (k as f64);
        }
        let norm = 2.0_f64.powi(n as i32);
        let mut next: BTreeMap<i64, f64> = BTreeMap::new();
        for (&base_milli_hz, &amp) in &tree {
            for (k, &bk) in binom.iter().enumerate() {
                // Shifts from -n/2 * J to +n/2 * J in steps of J
                let shift_hz = (k as f64 - 0.5 * (n as f64)) * c.j_hz;
                let key = base_milli_hz + (shift_hz * 1000.0).round() as i64;
                *next.entry(key).or_insert(0.0) += amp * (bk / norm);
            }
        }
        tree = next;
    }

    // Merge lines within 0.05 Hz (50 milli-Hz)
    let mut merged: Vec<MultipletLine> = Vec::with_capacity(tree.len());
    for (k, intensity) in tree {
        let hz = k as f64 / 1000.0;
        if let Some(last) = merged.last_mut() {
            if (hz - last.offset_hz).abs() < 0.05 {
                let total = last.intensity + intensity;
                if total > 0.0 {
                    last.offset_hz = (last.offset_hz * last.intensity + hz * intensity) / total;
                }
                last.intensity = total;
                continue;
            }
        }
        merged.push(MultipletLine {
            offset_hz: hz,
            intensity,
        });
    }
    merged
}

/// Solvents whose deuterium exchanges out labile `OH`, `NH`, and `SH` protons in ¹H NMR.
pub fn is_exchanging_solvent(solvent: &str) -> bool {
    let s = solvent.trim().to_ascii_uppercase();
    matches!(
        s.as_str(),
        "D2O" | "D₂O" | "CD3OD" | "CD₃OD" | "MEOD" | "METHANOL-D4" | "DEUTERIUM OXIDE"
    )
}

/// Predict 1D ¹H or ¹³C NMR shifts, equivalence groups, J-couplings, and multiplet splitting trees.
///
/// If `custom_tables` is provided, HOSE spheres `6..=0` are queried first in `custom_tables` before
/// falling back to the built-in substituent additivity + ring-current engine.
pub fn predict_nmr(
    mol: &AnalytMol,
    nucleus: NmrNucleus,
    solvent: Option<&str>,
    custom_tables: Option<&OnmrTables>,
) -> NmrPrediction {
    let solv_display = solvent.unwrap_or("CDCl₃").to_string();
    let env = perceive_topological_env(mol);
    let gone = nucleus == NmrNucleus::H1 && solvent.is_some_and(is_exchanging_solvent);
    let is_labile = |i: usize| matches!(env.atoms[i].el.as_str(), "O" | "N" | "S");

    let centers: Vec<usize> = (0..env.atoms.len())
        .filter(|&i| match nucleus {
            NmrNucleus::C13 => env.atoms[i].el == "C",
            NmrNucleus::H1 => env.atoms[i].h > 0 && !(gone && is_labile(i)),
        })
        .collect();

    let mut done_classes: HashSet<usize> = HashSet::new();
    let mut signals: Vec<NmrSignal> = Vec::new();

    for &c in &centers {
        let cls = env.symmetry_class[c];
        if !done_classes.insert(cls) {
            continue;
        }

        let codes = all_hose_codes(&env, c);
        let full_hose = codes.last().cloned().unwrap_or_default();

        // Check optional ONMR table first, then built-in clean-room additivity predictor
        let table_hit = custom_tables.and_then(|tables| {
            let main_t = match nucleus {
                NmrNucleus::C13 => &tables.c13,
                NmrNucleus::H1 => &tables.h1,
            };
            let solv_t = match nucleus {
                NmrNucleus::C13 => tables.solvent_c13.as_ref(),
                NmrNucleus::H1 => tables.solvent_h1.as_ref(),
            };
            for level in (0..codes.len()).rev() {
                let h = fnv1a_code_hash(nucleus, &codes[level]);
                if let Some((sh, sd, _cnt)) = main_t.lookup(h) {
                    if let (Some(st), Some(sv)) = (solv_t, solvent) {
                        let sh_key = format!("{sv}|{}", codes[level]);
                        if let Some((s_sh, s_sd, _)) = st.lookup(fnv1a_code_hash(nucleus, &sh_key))
                        {
                            return Some((s_sh, s_sd, level, Some(sv.to_string())));
                        }
                    }
                    return Some((sh, sd, level, None));
                }
            }
            None
        });

        let (shift_ppm, sd_ppm, sphere_level, matched_solv) = match table_hit {
            Some(hit) => hit,
            None => match nucleus {
                NmrNucleus::C13 => {
                    let (sh, sd, lv) = predict_c13_additivity(&env, c);
                    (sh, sd, lv, solvent.map(|s| s.to_string()))
                }
                NmrNucleus::H1 => {
                    let (sh, sd, lv) = predict_h1_additivity(mol, &env, c, solvent);
                    (sh, sd, lv, solvent.map(|s| s.to_string()))
                }
            },
        };

        let members: Vec<usize> = centers
            .iter()
            .copied()
            .filter(|&i| env.symmetry_class[i] == cls)
            .collect();

        let nuclei: usize = match nucleus {
            NmrNucleus::C13 => members.len(),
            NmrNucleus::H1 => members.iter().map(|&i| env.atoms[i].h as usize).sum(),
        };

        let mut couplings: Vec<SpinCoupling> = Vec::new();
        let broad = nucleus == NmrNucleus::H1 && is_labile(c);

        if nucleus == NmrNucleus::H1 && !broad {
            let mut add_coupling = |j: f64, n_h: usize, to_heavy: usize, kind: &str| {
                let orig_to = env.from_mol[to_heavy];
                if let Some(same) = couplings.iter_mut().find(|x| (x.j_hz - j).abs() < 0.6) {
                    same.count += n_h;
                    if !same.coupled_atoms.contains(&orig_to) {
                        same.coupled_atoms.push(orig_to);
                    }
                } else {
                    couplings.push(SpinCoupling {
                        j_hz: j,
                        count: n_h,
                        coupled_atoms: vec![orig_to],
                        kind: kind.to_string(),
                    });
                }
            };

            // 3J vicinal couplings across bonds to non-equivalent CHx neighbors
            for e in &env.edges[c] {
                let nb = e.to;
                if env.atoms[nb].h > 0 && !is_labile(nb) && env.symmetry_class[nb] != cls {
                    let (j, kind) = estimate_vicinal_j(mol, &env, c, nb);
                    add_coupling(j, env.atoms[nb].h as usize, nb, kind);
                }
            }

            // 4J meta-aromatic couplings (~2.0 Hz) on aromatic rings to non-equivalent ring CH
            if env.aromatic[c] {
                let mut meta_set: BTreeSet<usize> = BTreeSet::new();
                for e in &env.edges[c] {
                    if e.bond_sym == "*" {
                        for f in &env.edges[e.to] {
                            if f.bond_sym == "*"
                                && f.to != c
                                && env.edges[c].iter().all(|x| x.to != f.to)
                            {
                                meta_set.insert(f.to);
                            }
                        }
                    }
                }
                for m in meta_set {
                    if env.atoms[m].h > 0 && !is_labile(m) && env.symmetry_class[m] != cls {
                        add_coupling(2.0, env.atoms[m].h as usize, m, "4J_meta");
                    }
                }
            }
        }

        couplings.sort_by(|a, b| {
            b.j_hz
                .partial_cmp(&a.j_hz)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let multiplet_lines = if broad || couplings.is_empty() {
            vec![MultipletLine {
                offset_hz: 0.0,
                intensity: nuclei as f64,
            }]
        } else {
            compute_multiplet_lines(&couplings, nuclei)
        };

        let h_on_c = env.atoms[c].h;
        let h_suffix = if h_on_c == 0 {
            String::new()
        } else if h_on_c == 1 {
            "H".to_string()
        } else {
            format!("H{h_on_c}")
        };
        let mult_prefix = if members.len() > 1 {
            format!("{} × ", members.len())
        } else {
            String::new()
        };
        let group_label = format!("{mult_prefix}{}{h_suffix}", env.atoms[c].el);

        let hash_val = fnv1a_code_hash(nucleus, &full_hose);
        let hose_hash = to_base36(hash_val);

        signals.push(NmrSignal {
            atoms: members.iter().map(|&i| env.from_mol[i]).collect(),
            shift_ppm: (shift_ppm * 100.0).round() / 100.0,
            nuclei,
            couplings,
            multiplet_lines,
            broad,
            sd_ppm,
            sphere_level,
            group_label,
            hose_hash,
            hose_code: full_hose,
            solvent: matched_solv,
        });
    }

    signals.sort_by(|a, b| {
        b.shift_ppm
            .partial_cmp(&a.shift_ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    NmrPrediction {
        nucleus,
        signals,
        solvent: solv_display,
    }
}

fn to_base36(mut v: u32) -> String {
    if v == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::new();
    while v > 0 {
        buf.push(DIGITS[(v % 36) as usize] as char);
        v /= 36;
    }
    buf.iter().rev().collect()
}

/// Chemist's multiplicity label (`s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `dt`, `td`, `ddd`, `m`, `br s`),
/// matching `multiplicity` in `nmr.ts`.
pub fn multiplicity_string(s: &NmrSignal) -> String {
    if s.broad {
        return "br s".to_string();
    }
    const NAMES: [&str; 7] = ["s", "d", "t", "q", "quint", "sext", "sept"];
    let mut c = s.couplings.clone();
    c.sort_by(|a, b| {
        b.j_hz
            .partial_cmp(&a.j_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if c.len() <= 1 {
        let n = c.first().map(|x| x.count).unwrap_or(0);
        return if n > 6 {
            "m".to_string()
        } else {
            NAMES[n].to_string()
        };
    }
    if c.len() > 3 || c.iter().any(|x| x.count > 3) {
        "m".to_string()
    } else {
        c.iter()
            .map(|x| NAMES[x.count])
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Confidence quality rating (`good`, `medium`, `rough`) matching `quality` in `nmr.ts`.
pub fn signal_quality(s: &NmrSignal, nucleus: NmrNucleus) -> ShiftQuality {
    let tight = match nucleus {
        NmrNucleus::H1 => 0.15,
        NmrNucleus::C13 => 1.5,
    };
    if s.sphere_level <= 2 || s.sd_ppm > 2.0 * tight {
        return ShiftQuality::Rough;
    }
    if s.sphere_level >= 4 && s.sd_ppm <= tight {
        ShiftQuality::Good
    } else {
        ShiftQuality::Medium
    }
}

/// ACS style ¹H or ¹³C NMR characterization string (`acsText` in `nmr.ts`).
pub fn acs_report_text(p: &NmrPrediction, mhz_1h: f64, solvent: &str) -> String {
    let mut signals = p.signals.clone();
    signals.sort_by(|a, b| {
        b.shift_ppm
            .partial_cmp(&a.shift_ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if p.nucleus == NmrNucleus::C13 {
        let c_mhz = (mhz_1h * 0.25145).round() as usize;
        let list = signals
            .iter()
            .map(|s| format!("{:.1}", s.shift_ppm))
            .collect::<Vec<_>>()
            .join(", ");
        return format!("¹³C NMR ({c_mhz} MHz, {solvent}) δ {list}.");
    }
    let mhz_int = mhz_1h.round() as usize;
    let items: Vec<String> = signals
        .iter()
        .map(|s| {
            let m = multiplicity_string(s);
            let j_part = if !s.couplings.is_empty() && m != "m" && !s.broad {
                let mut cs = s.couplings.clone();
                cs.sort_by(|a, b| {
                    b.j_hz
                        .partial_cmp(&a.j_hz)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let js = cs
                    .iter()
                    .map(|c| format!("{:.1}", c.j_hz))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(", J = {js} Hz")
            } else {
                String::new()
            };
            format!("{:.2} ({m}{j_part}, {}H)", s.shift_ppm, s.nuclei)
        })
        .collect();
    format!("¹H NMR ({mhz_int} MHz, {solvent}) δ {}.", items.join(", "))
}

/// Tab-separated details table (`detailsTable` in `nmr.ts`).
pub fn details_table_tsv(p: &NmrPrediction) -> String {
    let mut signals = p.signals.clone();
    signals.sort_by(|a, b| {
        b.shift_ppm
            .partial_cmp(&a.shift_ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut lines: Vec<String> = Vec::with_capacity(signals.len() + 1);
    if p.nucleus == NmrNucleus::H1 {
        lines.push(
            [
                "Group",
                "δ (ppm)",
                "Multiplicity",
                "J (Hz)",
                "Integral",
                "Spheres matched",
                "Measured in",
                "Spread (ppm)",
                "Quality",
            ]
            .join("\t"),
        );
        for s in &signals {
            let js = s
                .couplings
                .iter()
                .map(|c| format!("{:.1}", c.j_hz))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(
                [
                    s.group_label.clone(),
                    format!("{:.2}", s.shift_ppm),
                    multiplicity_string(s),
                    js,
                    format!("{}H", s.nuclei),
                    s.sphere_level.to_string(),
                    s.solvent
                        .clone()
                        .unwrap_or_else(|| "any solvent".to_string()),
                    format!("{:.2}", s.sd_ppm),
                    signal_quality(s, p.nucleus).as_str().to_string(),
                ]
                .join("\t"),
            );
        }
    } else {
        lines.push(
            [
                "Group",
                "δ (ppm)",
                "Carbons",
                "Spheres matched",
                "Measured in",
                "Spread (ppm)",
                "Quality",
            ]
            .join("\t"),
        );
        for s in &signals {
            lines.push(
                [
                    s.group_label.clone(),
                    format!("{:.1}", s.shift_ppm),
                    s.nuclei.to_string(),
                    s.sphere_level.to_string(),
                    s.solvent
                        .clone()
                        .unwrap_or_else(|| "any solvent".to_string()),
                    format!("{:.1}", s.sd_ppm),
                    signal_quality(s, p.nucleus).as_str().to_string(),
                ]
                .join("\t"),
            );
        }
    }
    lines.join("\n")
}
