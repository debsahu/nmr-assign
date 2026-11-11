//! Group peaks into backbone spin systems and resolve `i` / `i-1` offsets.
//!
//! Provenance: Original method implementing `docs/specs/03-assignment.md` sections 2 and 4.

use std::collections::BTreeMap;

use crate::peaks::Peak;

#[derive(Clone, Debug)]
pub struct Tolerances {
    pub h: f64,
    pub n: f64,
    pub match_ca: f64,
    pub match_cb: f64,
    pub match_c: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            h: 0.03,
            n: 0.30,
            match_ca: 0.25,
            match_cb: 0.35,
            match_c: 0.25,
        }
    }
}

impl Tolerances {
    #[inline]
    pub fn carbon_tol(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.match_ca,
            "CB" => self.match_cb,
            "C" => self.match_c,
            _ => 0.25,
        }
    }
}

/// Backbone spin system rooted on one HSQC peak.
///
/// `cands` holds competing hypotheses per `(atom, offset)` as `(ppm, support)` sorted by
/// descending support, where `support` is the number of peaks agreeing within tolerance
/// (`0` denotes a degenerate `i`/`i-1` hypothesis used only for connectivity).
#[derive(Clone, Debug)]
pub struct SpinSystem {
    pub h: f64,
    pub n: f64,
    pub cands: BTreeMap<(String, i32), Vec<(f64, usize)>>,
}

impl SpinSystem {
    pub fn new(h: f64, n: f64) -> Self {
        Self {
            h,
            n,
            cands: BTreeMap::new(),
        }
    }

    /// Best-supported value per `(atom, offset)`.
    pub fn shifts(&self) -> BTreeMap<(String, i32), f64> {
        let mut out = BTreeMap::new();
        for (k, v) in &self.cands {
            if let Some(&(ppm, _)) = v.first() {
                out.insert(k.clone(), ppm);
            }
        }
        out
    }

    /// True if at least one observed carbon candidate (`support > 0`) is attached to this root.
    pub fn has_observed_carbons(&self) -> bool {
        self.cands
            .values()
            .any(|list| list.iter().any(|&(_, sup)| sup > 0))
    }
}

fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / (vals.len() as f64)
}

fn mean_val(pairs: &[(f64, f64)]) -> f64 {
    pairs.iter().map(|p| p.0).sum::<f64>() / (pairs.len() as f64)
}

fn mean_dist(pairs: &[(f64, f64)]) -> f64 {
    pairs.iter().map(|p| p.1).sum::<f64>() / (pairs.len() as f64)
}

fn cluster_pairs(pairs: &[(f64, f64)], tol: f64) -> Vec<Vec<(f64, f64)>> {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    for p in sorted {
        if let Some(last) = out.last_mut() {
            if p.0 - mean_val(last) <= tol {
                last.push(p);
                continue;
            }
        }
        out.push(vec![p]);
    }
    out
}

#[derive(Clone, Debug)]
struct Entry {
    atom: Option<&'static str>,
    offset: Option<i32>,
    val: f64,
    hn_dist: f64,
    both_offsets: bool,
}

pub fn build_spin_systems(
    peaks: &[Peak],
    tol: &Tolerances,
    max_cands: usize,
    degenerate: bool,
) -> Result<Vec<SpinSystem>, String> {
    let roots: Vec<&Peak> = peaks.iter().filter(|p| p.exp == "HSQC").collect();
    if roots.is_empty() {
        return Err("an HSQC peak list is required to define spin system roots".to_string());
    }

    let mut attached: Vec<Vec<&Peak>> = vec![Vec::new(); roots.len()];
    for p in peaks {
        if p.exp == "HSQC" || p.c.is_none() {
            continue;
        }
        for (j, r) in roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
            if d <= 1.0 {
                attached[j].push(p);
            }
        }
    }

    let systems = roots
        .into_iter()
        .zip(attached)
        .map(|(r, plist)| resolve_root(r, &plist, tol, max_cands, degenerate))
        .collect();
    Ok(systems)
}

fn resolve_root(
    root: &Peak,
    plist: &[&Peak],
    tol: &Tolerances,
    max_cands: usize,
    degenerate: bool,
) -> SpinSystem {
    let hn_dist = |p: &Peak| ((root.h - p.h) / tol.h).hypot((root.n - p.n) / tol.n);

    let mut hncacb: Vec<Entry> = plist
        .iter()
        .filter(|p| p.exp == "HNCACB")
        .filter_map(|p| {
            p.c.map(|c| Entry {
                atom: Some(if p.sign >= 0 { "CA" } else { "CB" }),
                offset: None,
                val: c,
                hn_dist: hn_dist(p),
                both_offsets: true,
            })
        })
        .collect();

    let mut entries: Vec<Entry> = Vec::new();
    for p in plist {
        let Some(c) = p.c else { continue };
        let hd = hn_dist(p);
        match p.exp.as_str() {
            "HNCA" => entries.push(Entry {
                atom: Some("CA"),
                offset: None,
                val: c,
                hn_dist: hd,
                both_offsets: true,
            }),
            "HNCOCA" => entries.push(Entry {
                atom: Some("CA"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                both_offsets: false,
            }),
            "HNCO" => entries.push(Entry {
                atom: Some("C"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                both_offsets: false,
            }),
            "HNCACO" => entries.push(Entry {
                atom: Some("C"),
                offset: None,
                val: c,
                hn_dist: hd,
                both_offsets: true,
            }),
            _ => {}
        }
    }

    // CBCA(CO)NH peaks take atom type from the matching HNCACB peak, and that HNCACB peak becomes i-1.
    for p in plist {
        if p.exp != "CBCACONH" {
            continue;
        }
        let Some(c) = p.c else { continue };
        let hd = hn_dist(p);
        let mut best_idx: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        for (idx, e) in hncacb.iter().enumerate() {
            let atom = e.atom.unwrap_or("CA");
            let d = (e.val - c).abs();
            if d <= tol.carbon_tol(atom) && d < best_d {
                best_d = d;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            let matched_atom = hncacb[idx].atom;
            hncacb[idx].offset = Some(-1);
            entries.push(Entry {
                atom: matched_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                both_offsets: false,
            });
        } else {
            entries.push(Entry {
                atom: None,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                both_offsets: false,
            });
        }
    }
    entries.extend(hncacb);

    // Type untyped CBCA(CO)NH peaks by matching other i-1 references.
    for i in 0..entries.len() {
        if entries[i].atom.is_none() {
            let v = entries[i].val;
            for atom in ["CA", "CB"] {
                let m = tol.carbon_tol(atom);
                let matched = entries
                    .iter()
                    .any(|x| x.atom == Some(atom) && x.offset == Some(-1) && (x.val - v).abs() <= m);
                if matched {
                    entries[i].atom = Some(atom);
                    break;
                }
            }
        }
    }

    // Resolve offsets of "i and i-1" peaks against i-1 references.
    for i in 0..entries.len() {
        if let (Some(atom), None) = (entries[i].atom, entries[i].offset) {
            let v = entries[i].val;
            let m = tol.carbon_tol(atom);
            let mut has_refs = false;
            let mut matches_ref = false;
            for (j, x) in entries.iter().enumerate() {
                if j != i && x.atom == Some(atom) && x.offset == Some(-1) {
                    has_refs = true;
                    if (x.val - v).abs() <= m {
                        matches_ref = true;
                    }
                }
            }
            if has_refs {
                entries[i].offset = Some(if matches_ref { -1 } else { 0 });
            }
        }
    }

    let mut ss = SpinSystem::new(root.h, root.n);
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        for off in [0, -1] {
            let pairs: Vec<(f64, f64)> = entries
                .iter()
                .filter(|e| {
                    e.atom == Some(atom) && (e.offset == Some(off) || e.offset.is_none())
                })
                .map(|e| (e.val, e.hn_dist))
                .collect();
            if !pairs.is_empty() {
                let mut clusters = cluster_pairs(&pairs, m);
                clusters.sort_by(|a, b| {
                    b.len()
                        .cmp(&a.len())
                        .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                });
                clusters.truncate(max_cands);
                let cands: Vec<(f64, usize)> =
                    clusters.into_iter().map(|c| (mean_val(&c), c.len())).collect();
                ss.cands.insert((atom.to_string(), off), cands);
            } else if off == 0 && degenerate {
                let vals_deg: Vec<f64> = entries
                    .iter()
                    .filter(|e| e.atom == Some(atom) && e.offset == Some(-1) && e.both_offsets)
                    .map(|e| e.val)
                    .collect();
                if !vals_deg.is_empty() {
                    ss.cands
                        .insert((atom.to_string(), off), vec![(mean(&vals_deg), 0)]);
                }
            }
        }
    }
    ss
}
