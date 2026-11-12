//! Interactive `i - 1 / i / i + 1` Candidate Match-Finder Strip Tiler for 3D triple-resonance
//! backbone resonance assignment.
//!
//! Provenance: Clean-room implementation of sequential triple-resonance strip walk matching from
//! published peer-reviewed methodology:
//! - Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990, **89**, 496–514 (`HNCA` / `HNCO`
//!   heteronuclear 3D correlation principles).
//! - Grzesiek & Bax, *J. Am. Chem. Soc.* 1992, **114**, 6291–6293 & *J. Magn. Reson.* 1992,
//!   **96**, 432–440 (`CBCA(CO)NH` and `HNCACB` sequential `CA`/`CB`/`C'` strip matching).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use eframe::egui;
use nmr_core::priors::three_to_one;
use nmr_core::solver::{Matrix2D, ScoreConfig, FORBID};
use nmr_core::spectrum::Spectrum3D;
use nmr_core::spinsystems::SpinSystem;

use crate::contour_painter::SyncCrosshair;

/// Direction of a candidate sequential strip relative to the active query root `i`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StripMatchDirection {
    /// Predecessor `p` at `i - 1`, ranked by `P(p, i)`.
    Predecessor,
    /// Successor `q` at `i + 1`, ranked by `P(i, q)`.
    Successor,
}

/// Configuration for [`CandidateMatchFinder`].
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MatchFinderConfig {
    /// Maximum number of ranked predecessor (`i - 1`) and successor (`i + 1`) strips to return.
    pub top_k: usize,
    /// Scoring tolerances (`sigma_ca`, `sigma_cb`, `sigma_c`, `range_ppm`) matching `pair_matrix`.
    pub score_cfg: ScoreConfig,
    /// When `true`, filters out candidates whose user-locked sequence index conflicts with the
    /// active root's locked sequence index.
    pub exclude_conflicting_locks: bool,
}

impl Default for MatchFinderConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            score_cfg: ScoreConfig::default(),
            exclude_conflicting_locks: false,
        }
    }
}

/// Ranked predecessor (`i - 1`) or successor (`i + 1`) candidate spin system around an active root.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct StripMatchCandidate {
    pub spin_system_idx: usize,
    pub direction: StripMatchDirection,
    /// Exact sequential connectivity log-likelihood ratio `P(p, s)` or `P(s, q)`.
    pub pair_score: f64,
    /// Posterior relative probability in `[0.0, 1.0]` normalized over all candidates plus an
    /// unlinked/gap (`P = 0.0`) null hypothesis.
    pub rel_prob: f64,
    /// Signed `CA` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `CA`.
    pub delta_ca_ppm: Option<f64>,
    /// Signed `CB` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `CB`.
    pub delta_cb_ppm: Option<f64>,
    /// Signed `C'` residual `ppm(left, 0) - ppm(right, -1)` when both sides carry `C`.
    pub delta_c_ppm: Option<f64>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `CA`.
    pub ca_pair_ppm: Option<(f64, f64)>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `CB`.
    pub cb_pair_ppm: Option<(f64, f64)>,
    /// Matched `(ppm_left_i, ppm_right_im1)` pair for `C'`.
    pub c_pair_ppm: Option<(f64, f64)>,
    /// Number of nuclei (`0..=3`) contributing positive Gaussian log-likelihood support.
    pub matched_nuclei: usize,
    /// Number of nuclei (`1..=3`) present on both `left(0)` and `right(-1)`.
    pub compared_nuclei: usize,
    /// Optional current solver sequence placement (`0`-based residue index).
    pub assigned_residue: Option<usize>,
    /// Optional hard user lock (`0`-based residue index).
    pub locked_residue: Option<usize>,
    /// Whether this directed sequential edge has been explicitly confirmed by the user.
    pub confirmed_link: bool,
}

#[allow(dead_code)]
impl StripMatchCandidate {
    /// Root-mean-square chemical-shift residual (ppm) across all compared nuclei.
    pub fn rms_residual_ppm(&self) -> f64 {
        let mut sum_sq = 0.0;
        let mut n = 0usize;
        for d in [self.delta_ca_ppm, self.delta_cb_ppm, self.delta_c_ppm]
            .into_iter()
            .flatten()
        {
            sum_sq += d * d;
            n += 1;
        }
        if n > 0 {
            (sum_sq / (n as f64)).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

/// Top-$K$ predecessor (`i - 1`) and successor (`i + 1`) candidate sets for an active spin system.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct StripMatchResult {
    pub root_idx: usize,
    pub predecessors: Vec<StripMatchCandidate>,
    pub successors: Vec<StripMatchCandidate>,
}

/// Stateless ranking engine computing ranked `i - 1` and `i + 1` candidate strips from `SpinSystem`
/// carbon hypotheses and `nmr_core::solver::pair_matrix`.
#[allow(dead_code)]
pub struct CandidateMatchFinder;

#[allow(dead_code)]
impl CandidateMatchFinder {
    /// Convenience entry point ranking top-$K$ predecessors and successors with default settings.
    pub fn rank_top_k(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        spin_system_idx: usize,
        top_k: usize,
    ) -> StripMatchResult {
        let cfg = MatchFinderConfig {
            top_k,
            ..MatchFinderConfig::default()
        };
        Self::rank_neighbors(systems, pair_mat, spin_system_idx, None, None, None, &cfg)
    }

    /// Rank top-$K$ predecessor (`i - 1`) candidates `p` by `P(p, s)` and successor (`i + 1`)
    /// candidates `q` by `P(s, q)` around `spin_system_idx`.
    pub fn rank_neighbors(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        spin_system_idx: usize,
        assignments: Option<&[i32]>,
        locks: Option<&BTreeMap<usize, usize>>,
        confirmed_links: Option<&[(usize, usize)]>,
        cfg: &MatchFinderConfig,
    ) -> StripMatchResult {
        let n_sys = systems.len().min(pair_mat.rows).min(pair_mat.cols);
        if spin_system_idx >= n_sys || cfg.top_k == 0 {
            return StripMatchResult {
                root_idx: spin_system_idx,
                predecessors: Vec::new(),
                successors: Vec::new(),
            };
        }

        let predecessors = Self::rank_direction(
            systems,
            pair_mat,
            n_sys,
            spin_system_idx,
            StripMatchDirection::Predecessor,
            assignments,
            locks,
            confirmed_links,
            cfg,
        );
        let successors = Self::rank_direction(
            systems,
            pair_mat,
            n_sys,
            spin_system_idx,
            StripMatchDirection::Successor,
            assignments,
            locks,
            confirmed_links,
            cfg,
        );

        StripMatchResult {
            root_idx: spin_system_idx,
            predecessors,
            successors,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_direction(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        n_sys: usize,
        s: usize,
        direction: StripMatchDirection,
        assignments: Option<&[i32]>,
        locks: Option<&BTreeMap<usize, usize>>,
        confirmed_links: Option<&[(usize, usize)]>,
        cfg: &MatchFinderConfig,
    ) -> Vec<StripMatchCandidate> {
        let root_lock = locks.and_then(|m| m.get(&s).copied());
        let log_range = cfg.score_cfg.range_ppm.ln();

        let mut pool: Vec<StripMatchCandidate> = Vec::with_capacity(n_sys);
        for cand_idx in 0..n_sys {
            if cand_idx == s {
                continue;
            }
            let (left_idx, right_idx) = match direction {
                StripMatchDirection::Predecessor => (cand_idx, s),
                StripMatchDirection::Successor => (s, cand_idx),
            };

            let pair_score = pair_mat.get(left_idx, right_idx);
            if !pair_score.is_finite() || pair_score <= FORBID * 0.5 {
                continue;
            }

            let cand_lock = locks.and_then(|m| m.get(&cand_idx).copied());
            if cfg.exclude_conflicting_locks {
                if let Some(ks) = root_lock {
                    match direction {
                        StripMatchDirection::Predecessor => {
                            if ks == 0 {
                                continue;
                            }
                            if let Some(kc) = cand_lock {
                                if kc + 1 != ks {
                                    continue;
                                }
                            } else if locks.is_some_and(|m| {
                                m.iter().any(|(&os, &ok)| os != cand_idx && ok == ks - 1)
                            }) {
                                continue;
                            }
                        }
                        StripMatchDirection::Successor => {
                            if let Some(kc) = cand_lock {
                                if ks + 1 != kc {
                                    continue;
                                }
                            } else if locks.is_some_and(|m| {
                                m.iter().any(|(&os, &ok)| os != cand_idx && ok == ks + 1)
                            }) {
                                continue;
                            }
                        }
                    }
                } else if let Some(kc) = cand_lock {
                    // Candidate is locked to kc; exclude if the adjacent slot needed by s is locked to someone else
                    match direction {
                        StripMatchDirection::Predecessor => {
                            if locks
                                .is_some_and(|m| m.iter().any(|(&os, &ok)| os != s && ok == kc + 1))
                            {
                                continue;
                            }
                        }
                        StripMatchDirection::Successor => {
                            if kc == 0
                                || locks.is_some_and(|m| {
                                    m.iter().any(|(&os, &ok)| os != s && ok == kc - 1)
                                })
                            {
                                continue;
                            }
                        }
                    }
                }
            }

            let (ca_pair, delta_ca, ca_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "CA",
                cfg.score_cfg.sigma_ca,
                log_range,
            );
            let (cb_pair, delta_cb, cb_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "CB",
                cfg.score_cfg.sigma_cb,
                log_range,
            );
            let (c_pair, delta_c, c_pos) = Self::best_atom_match(
                &systems[left_idx],
                &systems[right_idx],
                "C",
                cfg.score_cfg.sigma_c,
                log_range,
            );

            let compared_nuclei = usize::from(delta_ca.is_some())
                + usize::from(delta_cb.is_some())
                + usize::from(delta_c.is_some());
            if compared_nuclei == 0 {
                continue;
            }
            let matched_nuclei = usize::from(ca_pos) + usize::from(cb_pos) + usize::from(c_pos);

            let assigned_residue = assignments
                .and_then(|pos| pos.get(cand_idx).copied())
                .and_then(|k| (k >= 0).then_some(k as usize));
            let confirmed_link =
                confirmed_links.is_some_and(|links| links.contains(&(left_idx, right_idx)));

            pool.push(StripMatchCandidate {
                spin_system_idx: cand_idx,
                direction,
                pair_score,
                rel_prob: 0.0,
                delta_ca_ppm: delta_ca,
                delta_cb_ppm: delta_cb,
                delta_c_ppm: delta_c,
                ca_pair_ppm: ca_pair,
                cb_pair_ppm: cb_pair,
                c_pair_ppm: c_pair,
                matched_nuclei,
                compared_nuclei,
                assigned_residue,
                locked_residue: cand_lock,
                confirmed_link,
            });
        }

        if pool.is_empty() {
            return pool;
        }

        // Numerically stable log-sum-exp normalization including null gap hypothesis P_null = 0.0
        let max_lp = pool.iter().fold(0.0_f64, |m, c| m.max(c.pair_score));
        let z = (0.0 - max_lp).exp()
            + pool
                .iter()
                .map(|c| (c.pair_score - max_lp).exp())
                .sum::<f64>();
        for c in &mut pool {
            c.rel_prob = ((c.pair_score - max_lp).exp() / z.max(1e-30)).clamp(0.0, 1.0);
        }

        pool.sort_by(|a, b| {
            b.confirmed_link
                .cmp(&a.confirmed_link)
                .then_with(|| b.pair_score.total_cmp(&a.pair_score))
                .then_with(|| b.matched_nuclei.cmp(&a.matched_nuclei))
                .then_with(|| a.rms_residual_ppm().total_cmp(&b.rms_residual_ppm()))
                .then_with(|| a.spin_system_idx.cmp(&b.spin_system_idx))
        });
        pool.truncate(cfg.top_k);
        pool
    }

    fn best_atom_match(
        left_ss: &SpinSystem,
        right_ss: &SpinSystem,
        atom: &str,
        sigma: f64,
        log_range: f64,
    ) -> (Option<(f64, f64)>, Option<f64>, bool) {
        let Some(a_list) = left_ss.cands.get(&(atom.to_string(), 0)) else {
            return (None, None, false);
        };
        let Some(b_list) = right_ss.cands.get(&(atom.to_string(), -1)) else {
            return (None, None, false);
        };
        if a_list.is_empty() || b_list.is_empty() {
            return (None, None, false);
        }

        let mut best_pair = (a_list[0].0, b_list[0].0);
        let mut best_abs = (best_pair.0 - best_pair.1).abs();
        for &(va, _) in a_list {
            for &(vb, _) in b_list {
                let d = (va - vb).abs();
                if d < best_abs {
                    best_abs = d;
                    best_pair = (va, vb);
                }
            }
        }
        let delta = best_pair.0 - best_pair.1;
        let const_term = -(sigma * TAU.sqrt()).ln() + log_range;
        let llr = -0.5 * (delta / sigma.max(1e-6)).powi(2) + const_term;
        (Some(best_pair), Some(delta), llr > 0.0)
    }
}

/// Evaluate a synthetic 1D Gaussian strip intensity profile at `c_ppm` from a `SpinSystem`'s
/// resolved carbon candidates when volumetric 3D cubes are absent (`peak-only` mode).
///
/// Positive lobes represent `CA` and `C'` peaks; `CB(i)` emits a negative lobe (matching standard
/// `HNCACB` sign convention) while `CB(i - 1)` emits a positive lobe (`CBCA(CO)NH` convention).
#[allow(dead_code)]
pub fn synthetic_strip_intensity_at_ppm(ss: &SpinSystem, c_ppm: f64) -> f32 {
    let mut acc = 0.0f64;
    for ((atom, off), cands) in &ss.cands {
        let lw = match atom.as_str() {
            "CA" => 0.32,
            "CB" => 0.38,
            "C" => 0.24,
            _ => 0.30,
        };
        for (rank, &(ppm, sup)) in cands.iter().enumerate() {
            if sup == 0 && rank > 0 {
                continue;
            }
            let d = (c_ppm - ppm) / lw;
            if d.abs() > 4.2 {
                continue;
            }
            let amp_scale = if *off == 0 { 1.0 } else { 0.68 }
                * (1.0 + 0.18 * (sup.saturating_sub(1) as f64))
                / ((rank + 1) as f64);
            let sign = if atom == "CB" && *off == 0 { -1.0 } else { 1.0 };
            acc += sign * amp_scale * (-0.5 * d * d).exp();
        }
    }
    acc.clamp(-1.5, 1.5) as f32
}

/// Optional volumetric 3D cubes used by [`StripTilerPanel`] when experimental 3D data are loaded.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
pub struct StripSpectra3D<'a> {
    /// Aliphatic `13C` 3D cube (`HNCACB`, `CBCA(CO)NH`, `HNCA`, `HN(CO)CA`).
    pub aliphatic_3d: Option<&'a Spectrum3D>,
    /// Carbonyl `13C'` 3D cube (`HNCO`, `HN(CA)CO`).
    pub carbonyl_3d: Option<&'a Spectrum3D>,
}

/// Read-only data context passed to [`StripTilerPanel::show`].
#[allow(dead_code)]
pub struct StripTilerContext<'a> {
    pub systems: &'a [SpinSystem],
    pub pair_mat: &'a Matrix2D,
    pub u_mat: Option<&'a Matrix2D>,
    pub sequence: &'a [String],
    pub seq_ids: &'a [i32],
    pub active_root: usize,
    pub assignments: Option<&'a [i32]>,
    pub locks: Option<&'a BTreeMap<usize, usize>>,
    pub confirmed_links: Option<&'a [(usize, usize)]>,
    pub spectra_3d: StripSpectra3D<'a>,
}

/// User actions emitted by 1-click controls inside [`StripTilerPanel`].
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StripTilerAction {
    /// Make `spin_system_idx` the active query strip `i` (`1`-click sequential strip walk).
    SelectRoot { spin_system_idx: usize },
    /// Confirm directed sequential connectivity `pred_idx (i - 1) -> succ_idx (i)`.
    ConfirmSequentialLink { pred_idx: usize, succ_idx: usize },
    /// Lock `spin_system_idx` onto `0`-based sequence slot `residue_seq_idx`.
    LockToResidue {
        spin_system_idx: usize,
        residue_seq_idx: usize,
    },
}

/// Interactive `egui` `Predecessors (i - 1) | Active Root (i) | Successors (i + 1)` strip tiler.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct StripTilerPanel {
    pub cfg: MatchFinderConfig,
    pub show_ca_guides: bool,
    pub show_cb_guides: bool,
    pub show_c_guides: bool,
    pub c_aliphatic_range: (f64, f64),
    pub c_carbonyl_range: (f64, f64),
    pub strip_half_width_h_ppm: f64,
}

impl Default for StripTilerPanel {
    fn default() -> Self {
        Self {
            cfg: MatchFinderConfig::default(),
            show_ca_guides: true,
            show_cb_guides: true,
            show_c_guides: true,
            c_aliphatic_range: (13.0, 74.0),
            c_carbonyl_range: (167.5, 182.5),
            strip_half_width_h_ppm: 0.04,
        }
    }
}

#[derive(Clone)]
enum ColumnRole {
    Predecessor {
        rank_1based: usize,
        cand: StripMatchCandidate,
    },
    ActiveRoot {
        ss_idx: usize,
    },
    Successor {
        rank_1based: usize,
        cand: StripMatchCandidate,
    },
}

impl ColumnRole {
    fn ss_idx(&self) -> usize {
        match self {
            Self::Predecessor { cand, .. } | Self::Successor { cand, .. } => cand.spin_system_idx,
            Self::ActiveRoot { ss_idx } => *ss_idx,
        }
    }
}

#[allow(dead_code)]
impl StripTilerPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Infer the best candidate sequence residue index (`0`-based) for a 1-click `LockToResidue`
    /// button on `col` given the active root's assignment/lock or unary matrix `U`.
    fn suggest_lock_residue(&self, ctx: &StripTilerContext<'_>, col: &ColumnRole) -> Option<usize> {
        let root_k = ctx
            .locks
            .and_then(|m| m.get(&ctx.active_root).copied())
            .or_else(|| {
                ctx.assignments
                    .and_then(|a| a.get(ctx.active_root).copied())
                    .and_then(|k| (k >= 0).then_some(k as usize))
            });

        let ss_idx = col.ss_idx();
        if let Some(existing_lock) = ctx.locks.and_then(|m| m.get(&ss_idx).copied()) {
            return Some(existing_lock);
        }

        match col {
            ColumnRole::Predecessor { .. } => {
                if let Some(rk) = root_k {
                    if rk > 1 && ctx.sequence.get(rk - 1).is_some_and(|r| r != "PRO") {
                        return Some(rk - 1);
                    }
                }
            }
            ColumnRole::Successor { .. } => {
                if let Some(rk) = root_k {
                    if rk + 1 < ctx.sequence.len()
                        && ctx.sequence.get(rk + 1).is_some_and(|r| r != "PRO")
                    {
                        return Some(rk + 1);
                    }
                }
            }
            ColumnRole::ActiveRoot { .. } => {
                if let Some(rk) = root_k {
                    return Some(rk);
                }
            }
        }

        if let Some(assigned_k) = ctx
            .assignments
            .and_then(|a| a.get(ss_idx).copied())
            .and_then(|k| (k >= 0).then_some(k as usize))
        {
            return Some(assigned_k);
        }

        if let Some(u_mat) = ctx.u_mat {
            if ss_idx < u_mat.rows {
                return (1..u_mat.cols.min(ctx.sequence.len()))
                    .filter(|&k| ctx.sequence[k] != "PRO" && u_mat.get(ss_idx, k) > FORBID * 0.5)
                    .max_by(|&ka, &kb| u_mat.get(ss_idx, ka).total_cmp(&u_mat.get(ss_idx, kb)));
            }
        }
        None
    }

    /// Render the interactive `Predecessors (i-1) | Active Root (i) | Successors (i+1)` strip
    /// tiler and return all 1-click [`StripTilerAction`]s triggered during this frame.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &StripTilerContext<'_>,
        mut crosshair: Option<&mut SyncCrosshair>,
    ) -> Vec<StripTilerAction> {
        let mut actions = Vec::new();
        if ctx.active_root >= ctx.systems.len() {
            ui.label("Select an HSQC root to inspect ranked i − 1 / i / i + 1 candidate strips.");
            return actions;
        }

        let matches = CandidateMatchFinder::rank_neighbors(
            ctx.systems,
            ctx.pair_mat,
            ctx.active_root,
            ctx.assignments,
            ctx.locks,
            ctx.confirmed_links,
            &self.cfg,
        );

        // Top controls bar: Top-K selector + alignment guide toggles
        ui.horizontal_wrapped(|ui| {
            let root_ss = &ctx.systems[ctx.active_root];
            ui.label(
                egui::RichText::new(format!(
                    "Strip Match-Finder  •  Root #{} (¹H {:.3}, ¹⁵N {:.2})",
                    ctx.active_root, root_ss.h, root_ss.n
                ))
                .strong(),
            );
            ui.separator();
            ui.label(egui::RichText::new("Top-K:").small());
            for k_opt in [3usize, 5, 8] {
                if ui
                    .selectable_label(self.cfg.top_k == k_opt, format!("{k_opt}"))
                    .clicked()
                {
                    self.cfg.top_k = k_opt;
                }
            }
            ui.separator();
            ui.checkbox(
                &mut self.show_ca_guides,
                egui::RichText::new("Cα")
                    .color(egui::Color32::from_rgb(75, 215, 255))
                    .small(),
            );
            ui.checkbox(
                &mut self.show_cb_guides,
                egui::RichText::new("Cβ")
                    .color(egui::Color32::from_rgb(255, 125, 80))
                    .small(),
            );
            ui.checkbox(
                &mut self.show_c_guides,
                egui::RichText::new("C'")
                    .color(egui::Color32::from_rgb(90, 235, 135))
                    .small(),
            );
            ui.checkbox(
                &mut self.cfg.exclude_conflicting_locks,
                egui::RichText::new("Filter Locks").small(),
            );
        });

        // Build ordered column list: [p_m ... p_2, p_1, Active(i), q_1, q_2 ... q_n]
        // so #1 predecessor and #1 successor sit immediately adjacent to Active Root (i).
        let mut columns: Vec<ColumnRole> =
            Vec::with_capacity(matches.predecessors.len() + 1 + matches.successors.len());
        for (idx, cand) in matches.predecessors.iter().cloned().enumerate().rev() {
            columns.push(ColumnRole::Predecessor {
                rank_1based: idx + 1,
                cand,
            });
        }
        let center_col_idx = columns.len();
        columns.push(ColumnRole::ActiveRoot {
            ss_idx: ctx.active_root,
        });
        for (idx, cand) in matches.successors.iter().cloned().enumerate() {
            columns.push(ColumnRole::Successor {
                rank_1based: idx + 1,
                cand,
            });
        }

        let left_axis_w = 38.0f32;
        let col_w = 96.0f32;
        let strip_h = 286.0f32;
        let desired_w = (left_axis_w + (columns.len() as f32) * col_w + 8.0)
            .max(ui.available_width().max(260.0));

        egui::ScrollArea::horizontal()
            .id_salt("strip_tiler_hscroll")
            .show(ui, |ui| {
                // 1. Paint synchronized split-band (C' + CA/CB) strip columns & dashed guides
                let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
                    egui::vec2(desired_w, strip_h),
                    egui::Sense::click_and_drag(),
                );
                let painter = ui.painter_at(canvas_rect);
                painter.rect_filled(canvas_rect, 4.0, egui::Color32::from_rgb(13, 17, 24));

                let header_h = 34.0f32;
                let co_top = canvas_rect.top() + header_h;
                let co_bot = co_top + 54.0;
                let ali_top = co_bot + 10.0;
                let ali_bot = canvas_rect.bottom() - 8.0;

                let eff_col_w =
                    ((canvas_rect.width() - left_axis_w - 6.0) / (columns.len() as f32)).max(col_w);

                let co_to_y = |ppm: f64| -> f32 {
                    SyncCrosshair::ppm_to_strip_y(
                        self.c_carbonyl_range.0,
                        self.c_carbonyl_range.1,
                        co_top,
                        co_bot,
                        ppm,
                    )
                };
                let ali_to_y = |ppm: f64| -> f32 {
                    SyncCrosshair::ppm_to_strip_y(
                        self.c_aliphatic_range.0,
                        self.c_aliphatic_range.1,
                        ali_top,
                        ali_bot,
                        ppm,
                    )
                };

                // Draw 13C' and 13CA/CB axis ticks
                for c_tick in [170.0, 175.0, 180.0] {
                    let y = co_to_y(c_tick);
                    painter.text(
                        egui::pos2(canvas_rect.left() + 3.0, y),
                        egui::Align2::LEFT_CENTER,
                        format!("{c_tick:.0}"),
                        egui::FontId::monospace(9.0),
                        egui::Color32::from_rgb(135, 190, 150),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(canvas_rect.left() + left_axis_w, y),
                            egui::pos2(canvas_rect.right() - 4.0, y),
                        ],
                        egui::Stroke::new(0.5, egui::Color32::from_rgb(26, 36, 46)),
                    );
                }
                for c_tick in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0] {
                    let y = ali_to_y(c_tick);
                    painter.text(
                        egui::pos2(canvas_rect.left() + 3.0, y),
                        egui::Align2::LEFT_CENTER,
                        format!("{c_tick:.0}"),
                        egui::FontId::monospace(9.0),
                        egui::Color32::from_rgb(145, 165, 190),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(canvas_rect.left() + left_axis_w, y),
                            egui::pos2(canvas_rect.right() - 4.0, y),
                        ],
                        egui::Stroke::new(0.5, egui::Color32::from_rgb(26, 35, 48)),
                    );
                }

                // Draw horizontal dashed CA (cyan), CB (orange-red), and C' (green) alignment guides
                // Left half (Predecessors -> Active Root): anchored at Active Root's (i - 1) shifts
                // Right half (Active Root -> Successors): anchored at Active Root's (i) shifts
                let x_pred_left = canvas_rect.left() + left_axis_w + 2.0;
                let x_center_mid =
                    canvas_rect.left() + left_axis_w + (center_col_idx as f32 + 0.5) * eff_col_w;
                let x_succ_right =
                    canvas_rect.left() + left_axis_w + (columns.len() as f32) * eff_col_w - 2.0;

                let root_ss = &ctx.systems[ctx.active_root];
                let guide_specs = [
                    (
                        "CA",
                        self.show_ca_guides,
                        egui::Color32::from_rgba_unmultiplied(75, 215, 255, 195),
                        false,
                    ),
                    (
                        "CB",
                        self.show_cb_guides,
                        egui::Color32::from_rgba_unmultiplied(255, 120, 75, 195),
                        false,
                    ),
                    (
                        "C",
                        self.show_c_guides,
                        egui::Color32::from_rgba_unmultiplied(90, 235, 135, 195),
                        true,
                    ),
                ];

                for (atom, enabled, color, is_carbonyl) in guide_specs {
                    if !enabled {
                        continue;
                    }
                    if center_col_idx > 0 {
                        if let Some(&(ppm_im1, _)) = root_ss
                            .cands
                            .get(&(atom.to_string(), -1))
                            .and_then(|c| c.first())
                        {
                            let y = if is_carbonyl {
                                co_to_y(ppm_im1)
                            } else {
                                ali_to_y(ppm_im1)
                            };
                            draw_dashed_hline(&painter, x_pred_left, x_center_mid, y, color);
                        }
                    }
                    if center_col_idx + 1 < columns.len() {
                        if let Some(&(ppm_i, sup)) = root_ss
                            .cands
                            .get(&(atom.to_string(), 0))
                            .and_then(|c| c.first())
                        {
                            if sup > 0 {
                                let y = if is_carbonyl {
                                    co_to_y(ppm_i)
                                } else {
                                    ali_to_y(ppm_i)
                                };
                                draw_dashed_hline(&painter, x_center_mid, x_succ_right, y, color);
                            }
                        }
                    }
                }

                // Render each strip column (header, 3D slice or synthetic Gaussian + sticks, click-to-select)
                for (c_idx, col) in columns.iter().enumerate() {
                    let x0 = canvas_rect.left() + left_axis_w + (c_idx as f32) * eff_col_w + 2.5;
                    let x1 = x0 + eff_col_w - 5.0;
                    let cx = 0.5 * (x0 + x1);
                    let ss_idx = col.ss_idx();
                    let ss = &ctx.systems[ss_idx];

                    let is_center = matches!(col, ColumnRole::ActiveRoot { .. });
                    let border_col = match col {
                        ColumnRole::ActiveRoot { .. } => egui::Color32::from_rgb(95, 210, 255),
                        ColumnRole::Predecessor { cand, .. }
                        | ColumnRole::Successor { cand, .. }
                            if cand.confirmed_link =>
                        {
                            egui::Color32::from_rgb(95, 240, 155)
                        }
                        _ => egui::Color32::from_rgb(50, 62, 82),
                    };

                    let co_box =
                        egui::Rect::from_min_max(egui::pos2(x0, co_top), egui::pos2(x1, co_bot));
                    let ali_box =
                        egui::Rect::from_min_max(egui::pos2(x0, ali_top), egui::pos2(x1, ali_bot));
                    for sub_box in [co_box, ali_box] {
                        painter.rect_stroke(
                            sub_box,
                            2.0,
                            egui::Stroke::new(if is_center { 1.6 } else { 1.0 }, border_col),
                            egui::StrokeKind::Inside,
                        );
                    }

                    // Column header & score subtitle
                    let res_badge = ctx
                        .locks
                        .and_then(|m| m.get(&ss_idx).copied())
                        .map(|k| format!("🔒{}{}", three_to_one(&ctx.sequence[k]), ctx.seq_ids[k]))
                        .or_else(|| {
                            ctx.assignments
                                .and_then(|a| a.get(ss_idx).copied())
                                .filter(|&k| k >= 0 && (k as usize) < ctx.sequence.len())
                                .map(|k| {
                                    let ku = k as usize;
                                    format!(
                                        "{}{}",
                                        three_to_one(&ctx.sequence[ku]),
                                        ctx.seq_ids[ku]
                                    )
                                })
                        })
                        .unwrap_or_else(|| format!("SS#{ss_idx}"));

                    let (top_label, sub_label, sub_col) = match col {
                        ColumnRole::Predecessor { rank_1based, cand } => (
                            format!("i−1 #{rank_1based} {res_badge}"),
                            format!(
                                "P{:+.1} {:>2.0}% ({}n)",
                                cand.pair_score,
                                cand.rel_prob * 100.0,
                                cand.matched_nuclei
                            ),
                            egui::Color32::from_rgb(185, 215, 250),
                        ),
                        ColumnRole::ActiveRoot { .. } => (
                            format!("Root(i) {res_badge}"),
                            format!("¹H {:.2} N {:.1}", ss.h, ss.n),
                            egui::Color32::from_rgb(110, 230, 255),
                        ),
                        ColumnRole::Successor { rank_1based, cand } => (
                            format!("i+1 #{rank_1based} {res_badge}"),
                            format!(
                                "P{:+.1} {:>2.0}% ({}n)",
                                cand.pair_score,
                                cand.rel_prob * 100.0,
                                cand.matched_nuclei
                            ),
                            egui::Color32::from_rgb(185, 215, 250),
                        ),
                    };

                    painter.text(
                        egui::pos2(cx, canvas_rect.top() + 9.0),
                        egui::Align2::CENTER_CENTER,
                        top_label,
                        egui::FontId::monospace(9.5),
                        egui::Color32::from_rgb(232, 242, 255),
                    );
                    painter.text(
                        egui::pos2(cx, canvas_rect.top() + 22.0),
                        egui::Align2::CENTER_CENTER,
                        sub_label,
                        egui::FontId::monospace(8.8),
                        sub_col,
                    );

                    // Paint volumetric 3D slice OR synthetic Gaussian strip profile
                    self.paint_band_profile(
                        &painter,
                        co_box,
                        ss,
                        ctx.spectra_3d.carbonyl_3d,
                        self.c_carbonyl_range,
                        &co_to_y,
                    );
                    self.paint_band_profile(
                        &painter,
                        ali_box,
                        ss,
                        ctx.spectra_3d.aliphatic_3d,
                        self.c_aliphatic_range,
                        &ali_to_y,
                    );

                    // Paint resolved candidate markers: (i) filled circles on left, (i-1) diamonds on right
                    self.paint_spin_system_markers(&painter, cx, ss, &co_to_y, &ali_to_y);

                    // Pointer hover -> update synchronized crosshair; header click -> SelectRoot
                    if let Some(hp) = canvas_resp.hover_pos() {
                        if (x0..=x1).contains(&hp.x) {
                            if co_box.contains(hp) {
                                let c_ppm = SyncCrosshair::strip_y_to_ppm(
                                    self.c_carbonyl_range.0,
                                    self.c_carbonyl_range.1,
                                    co_top,
                                    co_bot,
                                    hp.y,
                                );
                                if let Some(xh) = crosshair.as_deref_mut() {
                                    xh.update_from_strip_hover(Some(ss.h), Some(ss.n), c_ppm);
                                }
                            } else if ali_box.contains(hp) {
                                let c_ppm = SyncCrosshair::strip_y_to_ppm(
                                    self.c_aliphatic_range.0,
                                    self.c_aliphatic_range.1,
                                    ali_top,
                                    ali_bot,
                                    hp.y,
                                );
                                if let Some(xh) = crosshair.as_deref_mut() {
                                    xh.update_from_strip_hover(Some(ss.h), Some(ss.n), c_ppm);
                                }
                            }
                        }
                    }

                    if canvas_resp.clicked() {
                        if let Some(cp) = canvas_resp.interact_pointer_pos() {
                            let hdr_rect = egui::Rect::from_min_max(
                                egui::pos2(x0, canvas_rect.top()),
                                egui::pos2(x1, co_top),
                            );
                            if hdr_rect.contains(cp) && !is_center {
                                actions.push(StripTilerAction::SelectRoot {
                                    spin_system_idx: ss_idx,
                                });
                            }
                        }
                    }
                }

                // Draw synchronized 13C crosshair line across all strips when c_ppm is active
                if let Some(xh) = crosshair.as_deref() {
                    let line_col = if xh.locked {
                        egui::Color32::from_rgba_unmultiplied(255, 215, 90, 190)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(185, 225, 255, 125)
                    };
                    if let Some(y_co) = xh.project_c_to_strip_y(
                        self.c_carbonyl_range.0,
                        self.c_carbonyl_range.1,
                        co_top,
                        co_bot,
                    ) {
                        painter.line_segment(
                            [
                                egui::pos2(x_pred_left, y_co),
                                egui::pos2(x_succ_right, y_co),
                            ],
                            egui::Stroke::new(1.0, line_col),
                        );
                    }
                    if let Some(y_ali) = xh.project_c_to_strip_y(
                        self.c_aliphatic_range.0,
                        self.c_aliphatic_range.1,
                        ali_top,
                        ali_bot,
                    ) {
                        painter.line_segment(
                            [
                                egui::pos2(x_pred_left, y_ali),
                                egui::pos2(x_succ_right, y_ali),
                            ],
                            egui::Stroke::new(1.0, line_col),
                        );
                    }
                }

                // 2. Per-column residual summary & 1-click curation action buttons
                ui.horizontal(|ui| {
                    ui.add_space(left_axis_w);
                    for col in &columns {
                        let ss_idx = col.ss_idx();
                        ui.allocate_ui(egui::vec2(eff_col_w - 4.0, 56.0), |ui| {
                            ui.vertical_centered(|ui| {
                                match col {
                                    ColumnRole::Predecessor { cand, .. }
                                    | ColumnRole::Successor { cand, .. } => {
                                        let resid_txt = format_residuals_compact(cand);
                                        ui.label(
                                            egui::RichText::new(resid_txt).monospace().size(8.6),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui.small_button("Walk").clicked() {
                                                actions.push(StripTilerAction::SelectRoot {
                                                    spin_system_idx: ss_idx,
                                                });
                                            }
                                            let (pred_idx, succ_idx) = match col {
                                                ColumnRole::Predecessor { .. } => {
                                                    (ss_idx, ctx.active_root)
                                                }
                                                _ => (ctx.active_root, ss_idx),
                                            };
                                            if ui.small_button("Link").clicked() {
                                                actions.push(
                                                    StripTilerAction::ConfirmSequentialLink {
                                                        pred_idx,
                                                        succ_idx,
                                                    },
                                                );
                                            }
                                        });
                                    }
                                    ColumnRole::ActiveRoot { .. } => {
                                        ui.label(
                                            egui::RichText::new("Active (i)")
                                                .monospace()
                                                .size(8.8)
                                                .color(egui::Color32::from_rgb(110, 230, 255)),
                                        );
                                    }
                                }
                                if let Some(target_k) = self.suggest_lock_residue(ctx, col) {
                                    let rlabel = format!(
                                        "Lock {}{}",
                                        three_to_one(&ctx.sequence[target_k]),
                                        ctx.seq_ids[target_k]
                                    );
                                    if ui.small_button(rlabel).clicked() {
                                        actions.push(StripTilerAction::LockToResidue {
                                            spin_system_idx: ss_idx,
                                            residue_seq_idx: target_k,
                                        });
                                    }
                                }
                            });
                        });
                    }
                });
            });

        actions
    }

    fn paint_band_profile(
        &self,
        painter: &egui::Painter,
        band_box: egui::Rect,
        ss: &SpinSystem,
        spec3d: Option<&Spectrum3D>,
        c_range: (f64, f64),
        c_to_y: &dyn Fn(f64) -> f32,
    ) {
        let cx = band_box.center().x;
        let lo = c_range.0.min(c_range.1);
        let hi = c_range.0.max(c_range.1);

        if let Some(s3) = spec3d {
            let slice = s3.extract_strip(ss.h, ss.n, self.strip_half_width_h_ppm);
            let max_v = slice.max_abs().max(1.0);
            let nx = slice.nx().max(1);
            let col_dx = (band_box.width() * 0.80) / (nx as f32);
            for ic in 0..slice.ny() {
                let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                if !(lo..=hi).contains(&c_ppm) {
                    continue;
                }
                let y = c_to_y(c_ppm);
                for ix in 0..nx {
                    let val = slice.get(ix, ic);
                    let norm = (val.abs() / max_v).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(
                                45,
                                200,
                                255,
                                (norm * 145.0) as u8,
                            )
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                105,
                                95,
                                (norm * 145.0) as u8,
                            )
                        };
                        let px0 = band_box.left() + band_box.width() * 0.10 + (ix as f32) * col_dx;
                        painter.line_segment(
                            [egui::pos2(px0, y), egui::pos2(px0 + col_dx.max(1.5), y)],
                            egui::Stroke::new(2.0, col),
                        );
                    }
                }
            }
        } else {
            // Peak-only synthetic Gaussian strip profile
            let n_samples = (band_box.height().round() as usize).clamp(24, 180);
            for s_idx in 0..=n_samples {
                let t = (s_idx as f64) / (n_samples as f64);
                let c_ppm = c_range.0 + t * (c_range.1 - c_range.0);
                let amp = synthetic_strip_intensity_at_ppm(ss, c_ppm);
                let norm = amp.abs().clamp(0.0, 1.0);
                if norm > 0.08 {
                    let y = c_to_y(c_ppm);
                    let col = if amp >= 0.0 {
                        egui::Color32::from_rgba_unmultiplied(55, 200, 255, (norm * 115.0) as u8)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 115, 85, (norm * 115.0) as u8)
                    };
                    let half_w = norm * (band_box.width() * 0.34);
                    painter.line_segment(
                        [egui::pos2(cx - half_w, y), egui::pos2(cx + half_w, y)],
                        egui::Stroke::new(1.8, col),
                    );
                }
            }
        }
    }

    fn paint_spin_system_markers(
        &self,
        painter: &egui::Painter,
        cx: f32,
        ss: &SpinSystem,
        co_to_y: &dyn Fn(f64) -> f32,
        ali_to_y: &dyn Fn(f64) -> f32,
    ) {
        for (atom, base_col, is_co) in [
            ("CA", egui::Color32::from_rgb(75, 225, 255), false),
            ("CB", egui::Color32::from_rgb(255, 120, 80), false),
            ("C", egui::Color32::from_rgb(90, 235, 135), true),
        ] {
            if let Some(cands_i) = ss.cands.get(&(atom.to_string(), 0)) {
                if let Some(&(ppm, sup)) = cands_i.first() {
                    if sup > 0 {
                        let y = if is_co { co_to_y(ppm) } else { ali_to_y(ppm) };
                        painter.circle_filled(egui::pos2(cx - 7.0, y), 3.6, base_col);
                    }
                }
            }
            if let Some(cands_im1) = ss.cands.get(&(atom.to_string(), -1)) {
                if let Some(&(ppm, _)) = cands_im1.first() {
                    let y = if is_co { co_to_y(ppm) } else { ali_to_y(ppm) };
                    let pts = vec![
                        egui::pos2(cx + 7.0, y - 3.8),
                        egui::pos2(cx + 10.8, y),
                        egui::pos2(cx + 7.0, y + 3.8),
                        egui::pos2(cx + 3.2, y),
                    ];
                    painter.add(egui::Shape::convex_polygon(
                        pts,
                        base_col,
                        egui::Stroke::NONE,
                    ));
                }
            }
        }
    }
}

fn draw_dashed_hline(
    painter: &egui::Painter,
    x_start: f32,
    x_end: f32,
    y: f32,
    color: egui::Color32,
) {
    let span = (x_end - x_start).abs();
    if span < 2.0 {
        return;
    }
    let dash_px = 6.0f32;
    let gap_px = 4.0f32;
    let step = dash_px + gap_px;
    let x0 = x_start.min(x_end);
    let x1 = x_start.max(x_end);
    let mut x = x0;
    while x < x1 {
        let seg_end = (x + dash_px).min(x1);
        painter.line_segment(
            [egui::pos2(x, y), egui::pos2(seg_end, y)],
            egui::Stroke::new(1.2, color),
        );
        x += step;
    }
}

fn format_residuals_compact(cand: &StripMatchCandidate) -> String {
    let mut parts = Vec::with_capacity(3);
    if let Some(d) = cand.delta_ca_ppm {
        parts.push(format!("α{d:+.2}"));
    }
    if let Some(d) = cand.delta_cb_ppm {
        parts.push(format!("β{d:+.2}"));
    }
    if let Some(d) = cand.delta_c_ppm {
        parts.push(format!("C'{d:+.2}"));
    }
    if parts.is_empty() {
        "—".to_string()
    } else {
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    use nmr_core::nmrstar::{read_star_str, Reference};
    use nmr_core::picker::{
        corroborated_re_pick_3d, pick_hsqc_2d, CorroboratedRepickConfig, PickerConfig, StripBank3D,
    };
    use nmr_core::priors::one_to_three;
    use nmr_core::simulate::{simulate, SimConfig};
    use nmr_core::solver::{pair_matrix, unary_matrix};
    use nmr_core::spectrum::{read_ucsf_2d, read_ucsf_3d};
    use nmr_core::spinsystems::{build_spin_systems, Tolerances};

    fn make_ss(h: f64, n: f64, entries: &[(&str, i32, f64, usize)]) -> SpinSystem {
        let mut ss = SpinSystem::new(h, n);
        for &(atom, off, ppm, sup) in entries {
            ss.cands
                .entry((atom.to_string(), off))
                .or_default()
                .push((ppm, sup));
        }
        ss
    }

    #[test]
    fn test_candidate_match_finder_residuals_probabilities_and_locks() {
        // Build a 4-spin-system chain: SS0 -> SS1 -> SS2, plus SS3 (close competitor for SS1 -> ?)
        // and SS4 (carbon-less noise root that must NEVER appear in candidate lists).
        let systems = vec![
            make_ss(
                8.30,
                120.0,
                &[
                    ("CA", 0, 56.20, 2),
                    ("CB", 0, 31.40, 2),
                    ("C", 0, 175.10, 2),
                ],
            ),
            make_ss(
                8.10,
                118.4,
                &[
                    ("CA", -1, 56.24, 2),
                    ("CB", -1, 31.35, 2),
                    ("C", -1, 175.08, 2),
                    ("CA", 0, 61.10, 2),
                    ("CB", 0, 68.50, 2),
                    ("C", 0, 173.80, 2),
                ],
            ),
            make_ss(
                8.65,
                123.2,
                &[
                    ("CA", -1, 61.08, 2),
                    ("CB", -1, 68.54, 2),
                    ("C", -1, 173.82, 2),
                    ("CA", 0, 45.10, 2),
                    ("C", 0, 171.90, 2),
                ],
            ),
            make_ss(
                7.85,
                116.0,
                &[
                    ("CA", -1, 61.22, 1),
                    ("CB", -1, 68.75, 1),
                    ("C", -1, 174.05, 1),
                    ("CA", 0, 52.00, 1),
                ],
            ),
            SpinSystem::new(7.40, 112.1), // zero carbons
        ];

        let scfg = ScoreConfig::default();
        let p_mat = pair_matrix(&systems, &scfg);
        let res = CandidateMatchFinder::rank_top_k(&systems, &p_mat, 1, 5);

        // Carbon-less SS4 must be excluded
        assert!(res.predecessors.iter().all(|c| c.spin_system_idx != 4));
        assert!(res.successors.iter().all(|c| c.spin_system_idx != 4));

        // Top predecessor of SS1 is SS0 with 3 matched nuclei and tiny residuals
        assert_eq!(res.predecessors[0].spin_system_idx, 0);
        assert_eq!(res.predecessors[0].matched_nuclei, 3);
        assert!((res.predecessors[0].delta_ca_ppm.unwrap() - (-0.04)).abs() < 1e-9);
        assert!((res.predecessors[0].delta_cb_ppm.unwrap() - 0.05).abs() < 1e-9);
        assert!((res.predecessors[0].delta_c_ppm.unwrap() - 0.02).abs() < 1e-9);
        assert!(res.predecessors[0].rel_prob > 0.99);

        // Top successor of SS1 is SS2, followed by competitor SS3
        assert_eq!(res.successors[0].spin_system_idx, 2);
        assert_eq!(res.successors[1].spin_system_idx, 3);
        assert!(res.successors[0].rel_prob > res.successors[1].rel_prob);
        let prob_sum: f64 = res.successors.iter().map(|c| c.rel_prob).sum();
        assert!(prob_sum > 0.99 && prob_sum <= 1.0 + 1e-9);

        // Lock SS1 to k=5 and SS2 to conflicting k=9: with exclude_conflicting_locks=true, SS2 drops out
        let mut locks = BTreeMap::new();
        locks.insert(1usize, 5usize);
        locks.insert(2usize, 9usize);
        let locked_res = CandidateMatchFinder::rank_neighbors(
            &systems,
            &p_mat,
            1,
            None,
            Some(&locks),
            None,
            &MatchFinderConfig {
                top_k: 5,
                score_cfg: scfg,
                exclude_conflicting_locks: true,
            },
        );
        assert_eq!(locked_res.successors[0].spin_system_idx, 3);

        // Synthetic Gaussian strip profile polarity check on SS1: CA(0)=61.10 positive, CB(0)=68.50 negative
        assert!(synthetic_strip_intensity_at_ppm(&systems[1], 61.10) > 0.7);
        assert!(synthetic_strip_intensity_at_ppm(&systems[1], 68.50) < -0.7);
    }

    #[test]
    fn test_strip_tiler_panel_headless_egui_render() {
        let systems = vec![
            make_ss(8.30, 120.0, &[("CA", 0, 56.2, 2), ("C", 0, 175.1, 2)]),
            make_ss(
                8.10,
                118.4,
                &[
                    ("CA", -1, 56.2, 2),
                    ("C", -1, 175.1, 2),
                    ("CA", 0, 61.1, 2),
                    ("CB", 0, 68.5, 2),
                ],
            ),
            make_ss(
                8.65,
                123.2,
                &[("CA", -1, 61.1, 2), ("CB", -1, 68.5, 2), ("CA", 0, 45.1, 2)],
            ),
        ];
        let seq = vec![
            "MET".to_string(),
            "GLN".to_string(),
            "ILE".to_string(),
            "THR".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4];
        let scfg = ScoreConfig::default();
        let p_mat = pair_matrix(&systems, &scfg);
        let u_mat = unary_matrix(
            &systems,
            &seq,
            &nmr_core::priors::Priors::provisional(),
            &scfg,
        );
        let assignments = vec![1, 2, 3];

        let mut panel = StripTilerPanel::new();
        let mut xh = SyncCrosshair::new(8.10, 118.4, Some(61.1));
        let egui_ctx = egui::Context::default();
        let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: Some(&u_mat),
                sequence: &seq,
                seq_ids: &seq_ids,
                active_root: 1,
                assignments: Some(&assignments),
                locks: None,
                confirmed_links: None,
                spectra_3d: StripSpectra3D::default(),
            };
            assert_eq!(
                panel.suggest_lock_residue(&tctx, &ColumnRole::ActiveRoot { ss_idx: 1 }),
                Some(2)
            );
            let matches = CandidateMatchFinder::rank_top_k(&systems, &p_mat, 1, 3);
            assert_eq!(
                panel.suggest_lock_residue(
                    &tctx,
                    &ColumnRole::Predecessor {
                        rank_1based: 1,
                        cand: matches.predecessors[0].clone(),
                    }
                ),
                Some(1)
            );
            assert_eq!(
                panel.suggest_lock_residue(
                    &tctx,
                    &ColumnRole::Successor {
                        rank_1based: 1,
                        cand: matches.successors[0].clone(),
                    }
                ),
                Some(3)
            );
            let actions = panel.show(ui, &tctx, Some(&mut xh));
            assert!(actions.is_empty());
        });

        // Also verify volumetric 3D slice path (aliphatic + carbonyl cubes) + confirmed_links
        let hncacb = Spectrum3D {
            name: "HNCACB".to_string(),
            exp: "HNCACB".to_string(),
            axis_x: nmr_core::spectrum::AxisPpm::from_bounds("1H", 16, 9.5, 7.5, 600.0),
            axis_y: nmr_core::spectrum::AxisPpm::from_bounds("15N", 12, 128.0, 112.0, 60.8),
            axis_z: nmr_core::spectrum::AxisPpm::from_bounds("13C", 32, 74.0, 14.0, 150.9),
            data: vec![5.0f32; 16 * 12 * 32],
        };
        let hnco = Spectrum3D {
            name: "HNCO".to_string(),
            exp: "HNCO".to_string(),
            axis_x: nmr_core::spectrum::AxisPpm::from_bounds("1H", 16, 9.5, 7.5, 600.0),
            axis_y: nmr_core::spectrum::AxisPpm::from_bounds("15N", 12, 128.0, 112.0, 60.8),
            axis_z: nmr_core::spectrum::AxisPpm::from_bounds("13C", 16, 182.0, 168.0, 150.9),
            data: vec![8.0f32; 16 * 12 * 16],
        };
        let confirmed = vec![(0usize, 1usize)];
        let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: Some(&u_mat),
                sequence: &seq,
                seq_ids: &seq_ids,
                active_root: 1,
                assignments: Some(&assignments),
                locks: None,
                confirmed_links: Some(&confirmed),
                spectra_3d: StripSpectra3D {
                    aliphatic_3d: Some(&hncacb),
                    carbonyl_3d: Some(&hnco),
                },
            };
            let actions = panel.show(ui, &tctx, Some(&mut xh));
            assert!(actions.is_empty());
        });
    }

    fn evaluate_top3_sequential_neighbor_recall(
        systems: &[SpinSystem],
        pair_mat: &Matrix2D,
        truth_sys_for_res: &[Option<usize>],
    ) -> (usize, usize, f64) {
        let mut hits = 0usize;
        let mut total = 0usize;
        for k in 1..truth_sys_for_res.len() {
            let (Some(s_prev), Some(s_curr)) = (truth_sys_for_res[k - 1], truth_sys_for_res[k])
            else {
                continue;
            };
            if s_prev == s_curr {
                continue;
            }
            // 1. Successor (i + 1) query from s_prev must recover s_curr in top-3
            let res_prev = CandidateMatchFinder::rank_top_k(systems, pair_mat, s_prev, 3);
            total += 1;
            if res_prev
                .successors
                .iter()
                .any(|c| c.spin_system_idx == s_curr)
            {
                hits += 1;
            }

            // 2. Predecessor (i - 1) query from s_curr must recover s_prev in top-3
            let res_curr = CandidateMatchFinder::rank_top_k(systems, pair_mat, s_curr, 3);
            total += 1;
            if res_curr
                .predecessors
                .iter()
                .any(|c| c.spin_system_idx == s_prev)
            {
                hits += 1;
            }
        }
        let recall = if total > 0 {
            (hits as f64) / (total as f64)
        } else {
            0.0
        };
        (hits, total, recall)
    }

    #[test]
    fn test_synthetic_ubiquitin_top3_neighbor_recovery() {
        let star_txt = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
        let u_ref = read_star_str(star_txt, None).expect("parse synthetic_ubiquitin.str");
        let (peaks, truth) = simulate(
            &u_ref,
            &SimConfig {
                seed: 0,
                ..SimConfig::default()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let p_mat = pair_matrix(&systems, &ScoreConfig::default());

        let mut truth_sys_for_res = vec![None; u_ref.sequence.len()];
        for (s, &k) in truth.iter().enumerate() {
            if k >= 0 && (k as usize) < truth_sys_for_res.len() {
                truth_sys_for_res[k as usize] = Some(s);
            }
        }

        let (hits, total, recall) =
            evaluate_top3_sequential_neighbor_recall(&systems, &p_mat, &truth_sys_for_res);
        assert!(
            total >= 110 && recall >= 0.97,
            "Expected >= 97% top-3 sequential neighbor recall on synthetic Ubiquitin, got {hits}/{total} ({:.2}%)",
            recall * 100.0
        );
    }

    fn find_local_real_data_dir() -> Option<PathBuf> {
        if let Some(env_dir) = std::env::var_os("NMR_REAL_DATA_DIR") {
            let p = PathBuf::from(env_dir);
            return p.exists().then_some(p);
        }
        if let Some(home) = std::env::var_os("HOME") {
            let p = PathBuf::from(home).join("Workspace/nmr_software/data");
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    fn parse_fasta_three(path: &Path) -> Vec<String> {
        let txt = fs::read_to_string(path).expect("read FASTA");
        let mut seq = Vec::new();
        for line in txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('>') {
                continue;
            }
            for ch in t.chars() {
                if ch.is_ascii_alphabetic() {
                    seq.push(one_to_three(ch).to_string());
                }
            }
        }
        seq
    }

    fn parse_ubiquitin_truth(table_path: &Path, seq: Vec<String>) -> Reference {
        let txt = fs::read_to_string(table_path).expect("read Ubiquitin reference table");
        let n_res = seq.len();
        let mut truth = Reference::new(seq, (1..=n_res as i32).collect());
        for line in txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("Index") || t.starts_with('!') || t.starts_with('#') {
                continue;
            }
            let norm = t.replace("?1", "? 1").replace(":H* ?", ":H*?");
            let cols: Vec<&str> = norm.split_whitespace().collect();
            let Some(Ok(sn)) = cols.first().map(|s| s.parse::<usize>()) else {
                continue;
            };
            if sn == 0 || sn > n_res {
                continue;
            }
            let k = sn - 1;
            let pv = |s: &str| {
                (!s.contains('?') && s != "-")
                    .then(|| s.split(':').next().unwrap().parse::<f64>().ok())
                    .flatten()
            };
            if let Some(v) = cols.get(2).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "H".into()), v);
            }
            if let Some(v) = cols.get(3).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "N".into()), v);
            }
            if let Some(v) = cols.get(4).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "C".into()), v);
            }
            if let Some(v) = cols.get(5).and_then(|s| pv(s)) {
                truth.shifts.insert((k, "CA".into()), v);
            }
            if let Some(v) = cols
                .iter()
                .skip(6)
                .find(|s| s.ends_with(":Cb"))
                .and_then(|s| pv(s))
            {
                truth.shifts.insert((k, "CB".into()), v);
            }
        }
        truth
    }

    #[test]
    fn test_real_ubiquitin_ucsf_top3_neighbor_recovery_gate() {
        let Some(droot) = find_local_real_data_dir() else {
            eprintln!("Skipping real Ubiquitin .ucsf top-3 strip match-finder gate: NMR_REAL_DATA_DIR absent");
            return;
        };
        let ucsf_dir = droot.join("ubiquitin/sparky_files/ubiquitin_data");
        let truth_path = droot.join("ubiquitin_backbone_assignment_191104.txt");
        let fasta_path = ucsf_dir.join("1ubq.fasta.txt");
        if !ucsf_dir.join("HSQC.ucsf").exists() || !truth_path.exists() || !fasta_path.exists() {
            eprintln!("Skipping real Ubiquitin .ucsf top-3 gate: local .ucsf files absent");
            return;
        }

        let seq = parse_fasta_three(&fasta_path);
        let truth_ref = parse_ubiquitin_truth(&truth_path, seq.clone());

        let pcfg = PickerConfig::default();
        let repick_cfg = CorroboratedRepickConfig::default();
        let hsqc = read_ucsf_2d(ucsf_dir.join("HSQC.ucsf")).expect("read HSQC.ucsf");
        let roots = pick_hsqc_2d(&hsqc, &pcfg);
        let mut peaks = roots.clone();
        let mut bank = StripBank3D::new();
        for (fname, exp) in [
            ("HNCACB.ucsf", "HNCACB"),
            ("CBCACONH.ucsf", "CBCACONH"),
            ("HNCO.ucsf", "HNCO"),
            ("HNCACO.ucsf", "HNCACO"),
        ] {
            let s3 = read_ucsf_3d(ucsf_dir.join(fname), exp).expect("read 3D .ucsf");
            peaks.extend(bank.pick_and_record_3d(&s3, &roots, &pcfg));
        }
        let rescued = corroborated_re_pick_3d(&bank, &roots, &peaks, &pcfg, &repick_cfg);
        peaks.extend(rescued);

        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)
            .expect("build Ubiquitin .ucsf spin systems");
        let p_mat = pair_matrix(&systems, &ScoreConfig::default());

        // Map each reference amide residue 1-to-1 onto its picked spin system
        let mut used_sys = vec![false; systems.len()];
        let mut truth_sys_for_res = vec![None; seq.len()];
        for (k, aa) in seq.iter().enumerate().skip(1) {
            if aa == "PRO" {
                continue;
            }
            let (Some(&th), Some(&tn)) = (
                truth_ref.shifts.get(&(k, "H".to_string())),
                truth_ref.shifts.get(&(k, "N".to_string())),
            ) else {
                continue;
            };
            let mut best: Option<(usize, f64)> = None;
            for (s, ss) in systems.iter().enumerate() {
                if used_sys[s] || !ss.has_observed_carbons() {
                    continue;
                }
                let d = ((ss.h - th) / 0.08).hypot((ss.n - tn) / 0.65);
                if d <= 1.0 && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((s, d));
                }
            }
            if let Some((s, _)) = best {
                used_sys[s] = true;
                truth_sys_for_res[k] = Some(s);
            }
        }

        let (hits, total, recall) =
            evaluate_top3_sequential_neighbor_recall(&systems, &p_mat, &truth_sys_for_res);
        println!(
            "[Ubiquitin .ucsf Strip Match-Finder] Top-3 sequential neighbor recovery: {hits}/{total} ({:.2}%)",
            recall * 100.0
        );
        assert!(
            total >= 110 && recall >= 0.97,
            "Ubiquitin .ucsf top-3 sequential neighbor recovery {:.2}% ({hits}/{total}) below 97.0% gate",
            recall * 100.0
        );
    }
}
