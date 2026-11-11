//! Group peaks into backbone spin systems and resolve `i` / `i-1` offsets.
//!
//! Provenance: Clean-room original method implementing `docs/specs/03-assignment.md` sections 2 and 4,
//! including `1J_NCa` vs `2J_NCa` intensity-ratio `i` / `i-1` disambiguation (Kay, Ikura, Tschudin &
//! Bax, *J. Magn. Reson.* 1990, 89, 496–514), distinct-experiment support counting, per-experiment
//! intensity-weighted peak capping, automatic intensity-gated coincident `i == i-1` degenerate
//! candidate detection, and Asn/Gln sidechain `NH2` doublet rejection.

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
/// descending distinct-experiment support (and normalized intensity when peak heights exist),
/// where `support` is the count of distinct experiments agreeing within tolerance
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExpSignKey {
    Hnca,
    Hncoca,
    HncacbPos,
    HncacbNeg,
    Cbcaconh,
    Hnco,
    Hncaco,
}

impl ExpSignKey {
    const ALL: [Self; 7] = [
        Self::Hnca,
        Self::Hncoca,
        Self::HncacbPos,
        Self::HncacbNeg,
        Self::Cbcaconh,
        Self::Hnco,
        Self::Hncaco,
    ];

    #[inline]
    fn from_peak(p: &Peak) -> Option<Self> {
        match p.exp.as_str() {
            "HNCA" => Some(Self::Hnca),
            "HNCOCA" => Some(Self::Hncoca),
            "HNCACB" => Some(if p.sign >= 0 {
                Self::HncacbPos
            } else {
                Self::HncacbNeg
            }),
            "CBCACONH" => Some(Self::Cbcaconh),
            "HNCO" => Some(Self::Hnco),
            "HNCACO" => Some(Self::Hncaco),
            _ => None,
        }
    }

    /// Physically expected maximum peaks per `(root, experiment, sign_class)`, allowing up to 2x
    /// the nominal single-root multiplicity (`1` for `i-1`-only `HNCO`/`HNCOCA`, `2` otherwise)
    /// so overlapped HSQC roots retain both spin systems' peaks while suppressing ripple tails.
    #[inline]
    fn max_peaks(self) -> usize {
        match self {
            Self::Hnco | Self::Hncoca => 2,
            Self::Hnca | Self::Hncaco | Self::Cbcaconh | Self::HncacbPos | Self::HncacbNeg => 4,
        }
    }

    /// Distinct-experiment bitmask (`HncacbPos` and `HncacbNeg` share the `HNCACB` bit).
    #[inline]
    fn exp_bit(self) -> u8 {
        match self {
            Self::Hnca => 1 << 0,
            Self::Hncoca => 1 << 1,
            Self::HncacbPos | Self::HncacbNeg => 1 << 2,
            Self::Cbcaconh => 1 << 3,
            Self::Hnco => 1 << 4,
            Self::Hncaco => 1 << 5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ClusterPoint {
    val: f64,
    hn_dist: f64,
    eff_norm_height: f64,
    rank_score: f64,
    exp_bit: u8,
}

fn mean_val(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.val).sum::<f64>() / (pts.len() as f64)
}

fn cluster_val(pts: &[ClusterPoint], has_heights: bool) -> f64 {
    if !has_heights || pts.len() <= 1 {
        return mean_val(pts);
    }
    let mut w_sum = 0.0;
    let mut v_sum = 0.0;
    for p in pts {
        let w = p.eff_norm_height.clamp(0.2, 5.0);
        w_sum += w;
        v_sum += w * p.val;
    }
    if w_sum > 0.0 {
        v_sum / w_sum
    } else {
        mean_val(pts)
    }
}

fn mean_dist(pts: &[ClusterPoint]) -> f64 {
    pts.iter().map(|p| p.hn_dist).sum::<f64>() / (pts.len() as f64)
}

#[inline]
fn distinct_support(pts: &[ClusterPoint]) -> usize {
    pts.iter().fold(0u8, |acc, p| acc | p.exp_bit).count_ones() as usize
}

#[inline]
fn max_norm_height(pts: &[ClusterPoint]) -> f64 {
    pts.iter().fold(0.0f64, |m, p| m.max(p.eff_norm_height))
}

#[inline]
fn max_rank_score(pts: &[ClusterPoint]) -> f64 {
    pts.iter().fold(0.0f64, |m, p| m.max(p.rank_score))
}

/// Strictly transitive total order over same-support candidate clusters:
/// - When an `i-1`-only experiment exists (`has_im1_spectrum == true`), rank by `max_rank_score`
///   (`norm_height * exp(-1.8 * hn_dist^2)`) so genuine peaks (`~1.0x`) crush sinc/ridge artifacts
///   (`<0.15x`) and `1J` (`~1.4x`) beats `2J` (`~0.6x`), while on-center own peaks (`hd ~ 0.2`)
///   beat overlapped outer-ellipse neighbor `1J` peaks (`hd ~ 0.85`).
/// - When no `i-1`-only experiment exists (`HNCA`-only), keep both `1J` and `2J` signal clusters
///   (`eff_norm_height >= 0.30`) in the credible signal tier and order within that tier by `(H, N)`
///   centering (`mean_dist`) so the weaker `2J` `CA(i-1)` peak is never dropped by neighbor bleed.
#[inline]
fn cmp_cluster_height_then_dist(
    a: &[ClusterPoint],
    b: &[ClusterPoint],
    has_im1_spectrum: bool,
) -> std::cmp::Ordering {
    if has_im1_spectrum {
        max_rank_score(b)
            .total_cmp(&max_rank_score(a))
            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
    } else {
        let ta = max_norm_height(a) >= 0.30;
        let tb = max_norm_height(b) >= 0.30;
        tb.cmp(&ta)
            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
            .then_with(|| max_rank_score(b).total_cmp(&max_rank_score(a)))
    }
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
    key: ExpSignKey,
    atom: Option<&'static str>,
    offset: Option<i32>,
    val: f64,
    hn_dist: f64,
    height: Option<f64>,
    eff_norm_height: f64,
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

    let has_heights = peaks
        .iter()
        .any(|p| p.exp != "HSQC" && p.c.is_some() && p.height.is_some());

    let mut attached: Vec<Vec<&Peak>> = vec![Vec::new(); roots.len()];
    for p in peaks {
        if p.exp == "HSQC" || p.c.is_none() || ExpSignKey::from_peak(p).is_none() {
            continue;
        }
        for (j, r) in roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
            if d <= 1.0 {
                attached[j].push(p);
            }
        }
    }

    // Only cap when an experiment group actually exceeds M_exp due to ripple/noise proliferation;
    // rank by height * exp(-2.0 * hd^2) so on-center 2J peaks beat off-center neighbor 1J peaks.
    if has_heights {
        for (j, r) in roots.iter().enumerate() {
            let plist = &mut attached[j];
            if plist.len() <= 2 {
                continue;
            }
            let mut capped: Vec<&Peak> = Vec::with_capacity(plist.len());
            for key in ExpSignKey::ALL {
                let mut grp: Vec<(&Peak, f64)> = plist
                    .iter()
                    .copied()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .map(|p| {
                        let hd = ((r.h - p.h) / tol.h).hypot((r.n - p.n) / tol.n);
                        let score = p.height.unwrap_or(0.0) * (-0.5 * hd * hd).exp();
                        (p, score)
                    })
                    .collect();
                if grp.is_empty() {
                    continue;
                }
                grp.sort_by(|a, b| b.1.total_cmp(&a.1));
                grp.truncate(key.max_peaks());
                capped.extend(grp.into_iter().map(|(p, _)| p));
            }
            *plist = capped;
        }
    }

    // Compute single-peak median height per (experiment, sign_class) across the capped attachments
    // (falling back to all peaks of that class if needed) so normalized heights are O(1) across spectra.
    let mut med_by_key: BTreeMap<ExpSignKey, f64> = BTreeMap::new();
    if has_heights {
        for key in ExpSignKey::ALL {
            let mut hs: Vec<f64> = attached
                .iter()
                .flat_map(|plist| plist.iter().copied())
                .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                .filter_map(|p| p.height.filter(|&h| h > 0.0))
                .collect();
            if hs.is_empty() {
                hs = peaks
                    .iter()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .filter_map(|p| p.height.filter(|&h| h > 0.0))
                    .collect();
            }
            if let Some(m) = median_f64(hs) {
                med_by_key.insert(key, m);
            }
        }

        // Prune residual <5% strip-bleed remnants inside intense strips so weak artifacts in
        // i-1-only spectra cannot mis-tag strong intra-residue peaks.
        for plist in &mut attached {
            let mut filtered = Vec::with_capacity(plist.len());
            for key in ExpSignKey::ALL {
                let med = med_by_key.get(&key).copied().unwrap_or(0.0);
                let grp: Vec<&Peak> = plist
                    .iter()
                    .copied()
                    .filter(|p| ExpSignKey::from_peak(p) == Some(key))
                    .collect();
                let h_max = grp.iter().filter_map(|p| p.height).fold(0.0f64, f64::max);
                for p in grp {
                    if med > 0.0 && h_max > 2.5 * med {
                        if let Some(h) = p.height {
                            if h < 0.05 * h_max {
                                continue;
                            }
                        }
                    }
                    filtered.push(p);
                }
            }
            *plist = filtered;
        }
    }

    let global_im1_exp = (
        peaks
            .iter()
            .any(|p| p.c.is_some() && (p.exp == "HNCOCA" || p.exp == "CBCACONH")),
        peaks.iter().any(|p| p.c.is_some() && p.exp == "CBCACONH"),
        peaks.iter().any(|p| p.c.is_some() && p.exp == "HNCO"),
    );

    let use_degenerate = degenerate || has_heights;
    let mut systems = Vec::with_capacity(roots.len());
    let mut lacks_ca_support = vec![false; roots.len()];

    for (idx, (r, plist)) in roots.iter().copied().zip(attached).enumerate() {
        let (ss, has_credible_ca) = resolve_root(
            r,
            &plist,
            tol,
            max_cands,
            use_degenerate,
            has_heights,
            &med_by_key,
            global_im1_exp,
        );
        lacks_ca_support[idx] = !has_credible_ca;
        systems.push(ss);
    }

    // Reject Asn/Gln sidechain NH2 paired 1H doublets sharing a common 15N shift (`|dN| <= 0.12 ppm`,
    // `0.30 <= |dH| <= 1.20 ppm`) when both prongs lack credible intra-residue CA(i) support.
    // Clear candidates in-place rather than removing roots so 1-to-1 HSQC root indexing is preserved.
    let mut reject_nh2 = vec![false; systems.len()];
    for i in 0..systems.len() {
        if !lacks_ca_support[i] {
            continue;
        }
        for j in (i + 1)..systems.len() {
            if !lacks_ca_support[j] {
                continue;
            }
            let dn = (systems[i].n - systems[j].n).abs();
            let dh = (systems[i].h - systems[j].h).abs();
            if dn <= 0.12 && (0.30..=1.20).contains(&dh) {
                reject_nh2[i] = true;
                reject_nh2[j] = true;
            }
        }
    }
    for (ss, rej) in systems.iter_mut().zip(reject_nh2) {
        if rej {
            ss.cands.clear();
        }
    }

    Ok(systems)
}

#[allow(clippy::too_many_arguments)]
fn resolve_root(
    root: &Peak,
    plist: &[&Peak],
    tol: &Tolerances,
    max_cands: usize,
    use_degenerate: bool,
    has_heights: bool,
    med_by_key: &BTreeMap<ExpSignKey, f64>,
    global_im1_exp: (bool, bool, bool),
) -> (SpinSystem, bool) {
    let hn_dist = |p: &Peak| ((root.h - p.h) / tol.h).hypot((root.n - p.n) / tol.n);
    let raw_norm = |height: Option<f64>, key: ExpSignKey| -> f64 {
        let Some(h) = height else {
            return 0.0;
        };
        match med_by_key.get(&key) {
            Some(&m) if m > 0.0 => h / m,
            _ => h,
        }
    };
    let eff_norm = |p: &Peak, key: ExpSignKey, hd: f64| -> f64 {
        raw_norm(p.height, key) * (-0.5 * hd * hd).exp()
    };

    let mut hncacb: Vec<Entry> = plist
        .iter()
        .filter(|p| p.exp == "HNCACB")
        .filter_map(|p| {
            let c = p.c?;
            let key = ExpSignKey::from_peak(p)?;
            let hd = hn_dist(p);
            Some(Entry {
                exp: "HNCACB",
                key,
                atom: Some(if p.sign >= 0 { "CA" } else { "CB" }),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: eff_norm(p, key, hd),
                both_offsets: true,
            })
        })
        .collect();

    let mut entries: Vec<Entry> = Vec::new();
    for p in plist {
        let Some(c) = p.c else { continue };
        let Some(key) = ExpSignKey::from_peak(p) else {
            continue;
        };
        let hd = hn_dist(p);
        let enh = eff_norm(p, key, hd);
        match p.exp.as_str() {
            "HNCA" => entries.push(Entry {
                exp: "HNCA",
                key,
                atom: Some("CA"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: true,
            }),
            "HNCOCA" => entries.push(Entry {
                exp: "HNCOCA",
                key,
                atom: Some("CA"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            }),
            "HNCO" => entries.push(Entry {
                exp: "HNCO",
                key,
                atom: Some("C"),
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            }),
            "HNCACO" => entries.push(Entry {
                exp: "HNCACO",
                key,
                atom: Some("C"),
                offset: None,
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
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
        let enh = eff_norm(p, ExpSignKey::Cbcaconh, hd);
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
                key: ExpSignKey::Cbcaconh,
                atom: matched_atom,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
                both_offsets: false,
            });
        } else {
            entries.push(Entry {
                exp: "CBCACONH",
                key: ExpSignKey::Cbcaconh,
                atom: None,
                offset: Some(-1),
                val: c,
                hn_dist: hd,
                height: p.height,
                eff_norm_height: enh,
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
        let has_im1_spectrum = match atom {
            "CA" => global_im1_exp.0,
            "CB" => global_im1_exp.1,
            "C" => global_im1_exp.2,
            _ => true,
        };
        for &ename in &["HNCA", "HNCACB", "HNCACO"] {
            let mut idxs: Vec<usize> = entries
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
            let mut used_fallback = false;
            if !has_im1_spectrum && idxs.len() != 2 {
                // Isolated-strip fallback (only when no i-1-only spectrum was recorded for `atom`):
                // if the entire root ellipse has only 2 peaks within hn_dist <= 0.88 and the stronger
                // peak is not much farther off-center than the weaker peak, disambiguate 1J vs 2J.
                let all_for_exp: Vec<usize> = entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.exp == ename && e.atom == Some(atom) && e.offset.is_none())
                    .map(|(idx, _)| idx)
                    .collect();
                if all_for_exp.len() == 2
                    && all_for_exp
                        .iter()
                        .all(|&idx| entries[idx].hn_dist <= 0.88 && entries[idx].height.is_some())
                {
                    idxs = all_for_exp;
                    used_fallback = true;
                }
            }
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
                    let min_ratio = if !has_im1_spectrum {
                        if used_fallback {
                            1.65
                        } else {
                            1.55
                        }
                    } else {
                        1.60
                    };
                    let geom_ok =
                        !used_fallback || entries[strong].hn_dist <= entries[weak].hn_dist + 0.20;
                    if lo > 0.0 && hi / lo >= min_ratio && geom_ok {
                        entries[strong].offset = Some(0);
                        entries[weak].offset = Some(-1);
                        if !has_im1_spectrum {
                            for e in entries.iter_mut() {
                                if e.exp == ename
                                    && e.atom == Some(atom)
                                    && e.offset.is_none()
                                    && e.hn_dist > 0.72
                                {
                                    e.atom = None;
                                }
                            }
                        }
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
    let mut max_ca_i_sup = 0usize;
    let mut max_ca_i_norm = 0.0f64;

    for atom in ["CA", "CB", "C"] {
        let m = tol.carbon_tol(atom);
        let has_im1_spectrum = match atom {
            "CA" => global_im1_exp.0,
            "CB" => global_im1_exp.1,
            "C" => global_im1_exp.2,
            _ => true,
        };
        for off in [0, -1] {
            let pts: Vec<ClusterPoint> = entries
                .iter()
                .filter(|e| e.atom == Some(atom) && (e.offset == Some(off) || e.offset.is_none()))
                .map(|e| ClusterPoint {
                    val: e.val,
                    hn_dist: e.hn_dist,
                    eff_norm_height: e.eff_norm_height,
                    rank_score: e.eff_norm_height * (-1.3 * e.hn_dist * e.hn_dist).exp(),
                    exp_bit: e.key.exp_bit(),
                })
                .collect();
            if !pts.is_empty() {
                let mut clusters = cluster_points(&pts, m);
                if has_heights {
                    if has_im1_spectrum {
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| max_norm_height(b).total_cmp(&max_norm_height(a)))
                                .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                        });
                        clusters.truncate(max_cands);
                        // Among surviving multi-experiment corroborated (`support >= 2`) full-strength
                        // (`max_norm_height >= 0.65`) clusters well inside the root core (`mean_dist <= 0.50`),
                        // prefer the core cluster over a peripheral neighbor leak (`mean_dist > 0.50`) while
                        // preserving `max_norm_height` ordering everywhere else (including `CA(i)` vs `CA(i-1)`).
                        let best_sup2_h = clusters
                            .iter()
                            .filter(|c| distinct_support(c) >= 2)
                            .map(|c| max_norm_height(c))
                            .fold(0.0_f64, f64::max);
                        let min_comparable_h = (0.55 * best_sup2_h).max(0.65);
                        let on_root_corroborated = |c: &[ClusterPoint]| -> bool {
                            distinct_support(c) >= 2
                                && max_norm_height(c) >= min_comparable_h
                                && mean_dist(c) <= 0.28
                        };
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| on_root_corroborated(b).cmp(&on_root_corroborated(a)))
                                .then_with(|| max_norm_height(b).total_cmp(&max_norm_height(a)))
                                .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                        });
                    } else {
                        clusters.sort_by(|a, b| {
                            distinct_support(b)
                                .cmp(&distinct_support(a))
                                .then_with(|| cmp_cluster_height_then_dist(a, b, false))
                        });
                        clusters.truncate(max_cands);
                    }
                } else {
                    clusters.sort_by(|a, b| {
                        distinct_support(b)
                            .cmp(&distinct_support(a))
                            .then_with(|| mean_dist(a).total_cmp(&mean_dist(b)))
                    });
                    clusters.truncate(max_cands);
                }
                if atom == "CA" && off == 0 {
                    for c in &clusters {
                        max_ca_i_sup = max_ca_i_sup.max(distinct_support(c));
                        max_ca_i_norm = max_ca_i_norm.max(max_norm_height(c));
                    }
                }
                let cands: Vec<(f64, usize)> = clusters
                    .into_iter()
                    .map(|c| (cluster_val(&c, has_heights), distinct_support(&c)))
                    .collect();
                ss.cands.insert((atom.to_string(), off), cands);
            } else if off == 0 && use_degenerate {
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
                    let keep = if has_heights {
                        match med_by_key.get(&e.key) {
                            Some(&med) if med > 0.0 && e.height.is_some() => {
                                let summed_h: f64 = im1_both
                                    .iter()
                                    .filter(|o| o.key == e.key && (o.val - e.val).abs() <= m)
                                    .filter_map(|o| o.height)
                                    .sum();
                                summed_h >= 1.65 * med
                            }
                            _ => false,
                        }
                    } else {
                        true
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
    let has_credible_ca =
        max_ca_i_sup >= 2 || (max_ca_i_sup == 1 && (!has_heights || max_ca_i_norm >= 0.30));
    (ss, has_credible_ca)
}
