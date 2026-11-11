//! Group peaks into backbone spin systems and resolve `i` / `i-1` offsets.
//!
//! Provenance: Clean-room original method implementing `docs/specs/03-assignment.md` sections 2 and 4,
//! including `1J_NCa` vs `2J_NCa` intensity-ratio `i` / `i-1` disambiguation (Kay, Ikura, Tschudin &
//! Bax, *J. Magn. Reson.* 1990, 89, 496–514) and intensity-gated coincident `i == i-1` degenerate
//! candidate detection.

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

#[derive(Clone, Copy, Debug)]
struct ClusterPoint {
    val: f64,
    hn_dist: f64,
    height: f64,
}

fn mean_val(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.val).sum::<f64>() / (pts.len() as f64)
}

fn mean_dist(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.hn_dist).sum::<f64>() / (pts.len() as f64)
}

fn sum_height(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.height).sum::<f64>()
}

fn cluster_points(points: &[ClusterPoint], tol: f64) -> Vec<Vec<ClusterPoint>> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.val.total_cmp(&b.val));
    let mut out: Vec<Vec<ClusterPoint>> = Vec::new();
    for p in sorted {
        if let Some(last) = out.last_mut() {
            if p.val - mean_val(last) <= tol {
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
    exp: &'static str,
    atom: Option<&'static str>,
    offset: Option<i32>,
    val: f64,
    hn_dist: f64,
    height: Option<f64>,
    both_offsets: bool,
}

fn median_f64(mut vals: Vec<f64>) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    Some(vals[vals.len() / 2])
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

    // Estimate global median peak height per "both-offsets" experiment for degenerate gating
    let mut exp_med_height: BTreeMap<&'static str, f64> = BTreeMap::new();
    for &ename in &["HNCA", "HNCACB", "HNCACO"] {
        let hs: Vec<f64> = peaks
            .iter()
            .filter(|p| p.exp == ename)
            .filter_map(|p| p.height)
            .collect();
        if let Some(m) = median_f64(hs) {
            exp_med_height.insert(ename, m);
        }
    }

    let mut attached: Vec<Vec<&Peak>> = vec![Vec::new(); roots.len()];
    for p in peaks {
        if p.exp == "HSQC" || p.c.is_none() {
            continue;
        }
        if !matches!(
            p.exp.as_str(),
            "HNCA" | "HNCOCA" | "HNCACB" | "CBCACONH" | "HNCO" | "HNCACO"
        ) {
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
        .map(|(r, plist)| resolve_root(r, &plist, tol, max_cands, degenerate, &exp_med_height))
        .collect();
    Ok(systems)
}

fn resolve_root(
    root: &Peak,
    plist: &[&Peak],
    tol: &Tolerances,
    max_cands: usize,
    degenerate: bool,
    exp_med_height: &BTreeMap<&'static str, f64>,
) -> SpinSystem {
    let hn_dist = |p: &Peak| ((root.h - p.h) / tol.h).hypot((root.n - p.n) / tol.n);

    let mut hncacb: Vec<Entry> = plist
        .iter()
        .filter(|p| p.exp == "HNCACB")
        .filter_map(|p| {
            p.c.map(|c| Entry {
                exp: "HNCACB",
                atom: Some(if p.sign >= 0 { "CA" } else { "CB" }),
                offset: None,
                val: c,
                hn_dist: hn_dist(p),
                height: p.height,
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
                exp: "HNCA",
                atom: Some("CA"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                both_offsets: true,
            }),
            "HNCOCA" => entries.push(Entry {
                exp: "HNCOCA",
                atom: Some("CA"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                both_offsets: false,
            }),
            "HNCO" => entries.push(Entry {
                exp: "HNCO",
                atom: Some("C"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                both_offsets: false,
            }),
            "HNCACO" => entries.push(Entry {
                exp: "HNCACO",
                atom: Some("C"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
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
                exp: "CBCACONH",
                atom: matched_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                both_offsets: false,
            });
        } else {
            entries.push(Entry {
                exp: "CBCACONH",
                atom: None,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
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
                let matched = entries.iter().any(|x| {
                    x.atom == Some(atom) && x.offset == Some(-1) && (x.val - v).abs() <= m
                });
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

    // Intensity-ratio i / i-1 disambiguation when NO i-1-only reference exists for `atom`:
    // Within each "both_offsets" experiment (`HNCA`, `HNCACB`, `HNCACO`), if exactly 2 well-centered
    // peaks exist for `atom` (`hn_dist <= 0.72`) and one is >= 1.6x stronger (`1J` vs `2J`),
    // assign the stronger to `i` (`0`) and the weaker to `i-1` (`-1`).
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        let any_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if any_im1 {
            continue;
        }
        for &ename in &["HNCA", "HNCACB", "HNCACO"] {
            let idxs: Vec<usize> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.exp == ename
                        && e.atom == Some(atom)
                        && e.offset.is_none()
                        && e.hn_dist <= 0.72
                        && e.height.is_some()
                })
                .map(|(idx, _)| idx)
                .collect();
            if idxs.len() == 2 {
                let (i0, i1) = (idxs[0], idxs[1]);
                if (entries[i0].val - entries[i1].val).abs() > m {
                    let h0 = entries[i0].height.unwrap_or(0.0);
                    let h1 = entries[i1].height.unwrap_or(0.0);
                    let (strong, weak, hi, lo) = if h0 >= h1 {
                        (i0, i1, h0, h1)
                    } else {
                        (i1, i0, h1, h0)
                    };
                    if lo > 0.0 && hi / lo >= 1.60 {
                        entries[strong].offset = Some(0);
                        entries[weak].offset = Some(-1);
                    }
                }
            }
        }
        // Propagate newly resolved i-1 offset to remaining unresolved entries of the same atom
        let newly_resolved_im1 = entries
            .iter()
            .any(|e| e.atom == Some(atom) && e.offset == Some(-1));
        if newly_resolved_im1 {
            for i in 0..entries.len() {
                if entries[i].atom == Some(atom) && entries[i].offset.is_none() {
                    let v = entries[i].val;
                    let matches = entries.iter().enumerate().any(|(j, x)| {
                        j != i
                            && x.atom == Some(atom)
                            && x.offset == Some(-1)
                            && (x.val - v).abs() <= m
                    });
                    entries[i].offset = Some(if matches { -1 } else { 0 });
                }
            }
        }
    }

    let mut ss = SpinSystem::new(root.h, root.n);
    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        for off in [0, -1] {
            let pts: Vec<ClusterPoint> = entries
                .iter()
                .filter(|e| e.atom == Some(atom) && (e.offset == Some(off) || e.offset.is_none()))
                .map(|e| ClusterPoint {
                    val: e.val,
                    hn_dist: e.hn_dist,
                    height: e.height.unwrap_or(0.0),
                })
                .collect();
            if !pts.is_empty() {
                let mut clusters = cluster_points(&pts, m);
                clusters.sort_by(|a, b| {
                    b.len()
                        .cmp(&a.len())
                        .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                        .then_with(|| sum_height(b).total_cmp(&sum_height(a)))
                });
                clusters.truncate(max_cands);
                let cands: Vec<(f64, usize)> = clusters
                    .into_iter()
                    .map(|c| (mean_val(&c), c.len()))
                    .collect();
                ss.cands.insert((atom.to_string(), off), cands);
            } else if off == 0 && degenerate {
                // Intensity-gated degenerate i == i-1 hypothesis:
                // When heights are present, only offer the i-1 value as residue i if an experiment
                // observing both offsets (`HNCA`, `HNCACB`, `HNCACO`) has total superposed amplitude
                // at that carbon shift >= 1.65x the experiment's median single-peak height
                // (discriminating a coincident i + i-1 peak from a missing i peak).
                let im1_both: Vec<&Entry> = entries
                    .iter()
                    .filter(|e| e.atom == Some(atom) && e.offset == Some(-1) && e.both_offsets)
                    .collect();
                let mut vals_deg: Vec<f64> = Vec::new();
                for &e in &im1_both {
                    let keep = match exp_med_height.get(e.exp) {
                        Some(&med) if med > 0.0 && e.height.is_some() => {
                            let summed_h: f64 = im1_both
                                .iter()
                                .filter(|o| o.exp == e.exp && (o.val - e.val).abs() <= m)
                                .filter_map(|o| o.height)
                                .sum();
                            summed_h >= 1.65 * med
                        }
                        _ => true,
                    };
                    if keep {
                        vals_deg.push(e.val);
                    }
                }
                if !vals_deg.is_empty() {
                    let avg = vals_deg.iter().sum::<f64>() / (vals_deg.len() as f64);
                    ss.cands.insert((atom.to_string(), off), vec![(avg, 0)]);
                }
            }
        }
    }
    ss
}
