//! Score spin system placements and optimize the backbone assignment.
//!
//! Provenance: Original implementation of `docs/specs/03-assignment.md` section 3:
//! log-likelihood-ratio typing scores with automatic marginal reference offset alignment and
//! primary-sequence covalent neighbor corrections (Wishart et al., *J. Biomol. NMR* 1995, 5,
//! 67–81; Wishart et al., *J. Biomol. NMR* 1995, 6, 135–140; Braun, Wider & Wüthrich,
//! *J. Am. Chem. Soc.* 1994, 116, 8466–8469; Schwarzinger et al., *J. Am. Chem. Soc.* 2001, 123,
//! 2970–2978; MacArthur & Thornton, *J. Mol. Biol.* 1991, 218, 397–412), excess-root zero-`CA` /
//! 1-peak noise placement protection, user-locked `(spin_system, residue)` hard constraints,
//! Gaussian sequential connectivity scores, Metropolis-Hastings simulated annealing (Metropolis et
//! al., *J. Chem. Phys.* 1953, 21, 1087–1092; Hastings, *Biometrika* 1970, 57, 97–109) with
//! directed sequential chain-extension and contiguous dipeptide/tripeptide segment shift/swap
//! proposals (cf. Schmidt & Güntert, *J. Am. Chem. Soc.* 2012, 134, 12817–12829), and multi-run
//! consensus.

use std::f64::consts::TAU;

pub use crate::priors::{
    covalent_atom_offset, delta_covalent, estimate_reference_offsets,
    expected_covalent_atom_offset, Priors, ShiftOffsets, COVALENT_COIL_WEIGHT,
};
use crate::qc::apply_standard_calibration_map;
use crate::rng::Rng;
use crate::spinsystems::SpinSystem;

pub const FORBID: f64 = -1e9;

#[derive(Clone, Debug)]
pub struct ScoreConfig {
    pub sigma_ca: f64,
    pub sigma_cb: f64,
    pub sigma_c: f64,
    pub range_ppm: f64,
    pub pair_floor: f64,
    pub support_bonus: f64,
    pub assign_cost: f64,
    pub forbid_zero_carbons: bool,
    pub use_hn: bool,
    /// Automatically estimate and apply global chemical-shift reference offsets to `Priors`
    /// during unary scoring while keeping output shifts in the experimental ppm frame.
    pub auto_reference: bool,
    /// Protect exchange-broadened sequence slots against spurious roots lacking `CA(i)`/`CA(i-1)`
    /// or having only a single uncorroborated carbon peak when `systems.len() > assignable`.
    pub forbid_ca_less_excess_roots: bool,
    /// Optional user-locked `(spin_system_idx, seq_idx)` hard assignment constraints.
    /// When empty (default), `U(s, k)` and `P(s, t)` are bit-for-bit unchanged.
    pub locked_assignments: Vec<(usize, usize)>,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            sigma_ca: 0.12,
            sigma_cb: 0.20,
            sigma_c: 0.12,
            range_ppm: 30.0,
            pair_floor: -8.0,
            support_bonus: 0.7,
            assign_cost: 0.0,
            forbid_zero_carbons: true,
            use_hn: true,
            auto_reference: true,
            forbid_ca_less_excess_roots: true,
            locked_assignments: Vec::new(),
        }
    }
}

impl ScoreConfig {
    #[inline]
    pub fn sigma(&self, atom: &str) -> f64 {
        match atom {
            "CA" => self.sigma_ca,
            "CB" => self.sigma_cb,
            "C" => self.sigma_c,
            _ => 0.15,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnnealConfig {
    pub runs: usize,
    pub steps_per_system: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub max_block: usize,
    pub seed: u64,
}

impl Default for AnnealConfig {
    fn default() -> Self {
        Self {
            runs: 16,
            steps_per_system: 2500,
            t_start: 8.0,
            t_end: 0.05,
            max_block: 8,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Matrix2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix2D {
    pub fn filled(rows: usize, cols: usize, val: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![val; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Best candidate's LLR plus support bonus, minus `ln(n)` penalty for choosing among `n` hypotheses.
/// Returns `(score, index_of_chosen_candidate)`.
pub fn candidate_llr(
    priors: &Priors,
    res: &str,
    atom: &str,
    cands: &[(f64, usize)],
    support_bonus: f64,
) -> (f64, usize) {
    candidate_llr_shifted(priors, res, atom, cands, support_bonus, 0.0)
}

/// Same as [`candidate_llr`], shifting the residue expectation mean by `d_mu` ppm while keeping
/// the residue-averaged background density fixed.
pub fn candidate_llr_shifted(
    priors: &Priors,
    res: &str,
    atom: &str,
    cands: &[(f64, usize)],
    support_bonus: f64,
    d_mu: f64,
) -> (f64, usize) {
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &(v, sup)) in cands.iter().enumerate() {
        let sc = priors.llr_shifted(res, atom, v, d_mu) + support_bonus * ((sup as f64) - 1.0);
        if sc > best {
            best = sc;
            best_i = i;
        }
    }
    (best - (cands.len() as f64).ln(), best_i)
}

/// Prepare effective scoring `Priors` (applying `estimate_reference_offsets` when
/// `cfg.auto_reference` is enabled) and return `(effective_priors, applied_offsets)`.
///
/// Exposed publicly so GPU (`nmr-gpu`) and CLI (`nmr-cli`) engines share identical reference
/// alignment while `SpinSystem` candidates and output shifts remain in the experimental ppm frame.
pub fn prepare_scoring_priors(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &ScoreConfig,
) -> (Priors, ShiftOffsets) {
    if cfg.auto_reference {
        let offsets = estimate_reference_offsets(systems, sequence, priors);
        (priors.with_offsets(&offsets), offsets)
    } else {
        (priors.clone(), ShiftOffsets::default())
    }
}

pub fn unary_matrix(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &ScoreConfig,
) -> Matrix2D {
    let s_len = systems.len();
    let k_len = sequence.len();
    let mut u = Matrix2D::filled(s_len, k_len, FORBID);
    if s_len == 0 || k_len == 0 {
        return u;
    }

    let (eff_priors, _) = prepare_scoring_priors(systems, sequence, priors, cfg);
    let assignable = sequence
        .iter()
        .enumerate()
        .filter(|&(k, r)| k > 0 && r != "PRO")
        .count();
    let has_excess_roots = cfg.forbid_ca_less_excess_roots && s_len > assignable;

    let mut has_obs_ca = vec![true; s_len];
    let mut obs_slots = vec![0usize; s_len];
    let mut max_sup = vec![0usize; s_len];
    let mut multi_slot_count = 0usize;
    if has_excess_roots {
        for (s, ss) in systems.iter().enumerate() {
            let ca0 = ss
                .cands
                .get(&("CA".to_string(), 0))
                .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0));
            let cam1 = ss
                .cands
                .get(&("CA".to_string(), -1))
                .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0));
            has_obs_ca[s] = ca0 || cam1;
            let mut n_slots = 0usize;
            let mut m_sup = 0usize;
            for raw in ss.cands.values() {
                let mut slot_pos = false;
                for &(_, sup) in raw {
                    if sup > 0 {
                        slot_pos = true;
                        m_sup = m_sup.max(sup);
                    }
                }
                if slot_pos {
                    n_slots += 1;
                }
            }
            obs_slots[s] = n_slots;
            max_sup[s] = m_sup;
            if n_slots >= 3 {
                multi_slot_count += 1;
            }
        }
    }
    let multi_slot_dataset = multi_slot_count >= 8;

    for k in 1..k_len {
        let res = &sequence[k];
        let prev = &sequence[k - 1];
        if res == "PRO" {
            continue;
        }
        let cov_k = delta_covalent(sequence, k);
        let cov_km1 = delta_covalent(sequence, k - 1);
        for (s, ss) in systems.iter().enumerate() {
            if cfg.forbid_zero_carbons && !ss.has_observed_carbons() {
                continue;
            }
            if has_excess_roots && !has_obs_ca[s] && res != "GLY" && prev != "PRO" {
                continue;
            }
            let mut sc = 0.0;
            for ((atom, off), raw_cands) in &ss.cands {
                let cands: Vec<(f64, usize)> =
                    raw_cands.iter().copied().filter(|c| c.1 > 0).collect();
                if cands.is_empty() {
                    continue;
                }
                let (target_res, cov) = if *off == 0 {
                    (res, cov_k)
                } else {
                    (prev, cov_km1)
                };
                let d_cov = covalent_atom_offset(cov, atom);
                sc += candidate_llr_shifted(
                    &eff_priors,
                    target_res,
                    atom,
                    &cands,
                    cfg.support_bonus,
                    d_cov,
                )
                .0;
            }
            if cfg.use_hn {
                sc += eff_priors.llr_shifted(res, "N", ss.n, cov_k.0)
                    + eff_priors.llr(res, "H", ss.h);
            }
            if has_excess_roots
                && multi_slot_dataset
                && (!has_obs_ca[s] || (obs_slots[s] <= 1 && max_sup[s] <= 1))
            {
                sc -= 4.5;
            }
            sc -= cfg.assign_cost;
            u.set(s, k, sc);
        }
    }
    if !cfg.locked_assignments.is_empty() {
        apply_locked_assignments_to_unary(&mut u, systems, sequence, &eff_priors, cfg);
    }
    u
}

/// Filter user-supplied `(spin_system_idx, seq_idx)` locks to injective, assignable backbone
/// amide positions (`s < s_len`, `0 < k < sequence.len()`, `sequence[k] != "PRO"`).
pub fn resolve_locked_assignments(
    s_len: usize,
    sequence: &[String],
    locks: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    if locks.is_empty() || s_len == 0 || sequence.len() <= 1 {
        return Vec::new();
    }
    let mut used_s = vec![false; s_len];
    let mut used_k = vec![false; sequence.len()];
    let mut valid = Vec::with_capacity(locks.len());
    for &(s, k) in locks {
        if s >= s_len || k == 0 || k >= sequence.len() {
            continue;
        }
        let r = sequence[k].trim();
        if r.eq_ignore_ascii_case("PRO") || r.eq_ignore_ascii_case("P") {
            continue;
        }
        if used_s[s] || used_k[k] {
            continue;
        }
        used_s[s] = true;
        used_k[k] = true;
        valid.push((s, k));
    }
    valid
}

/// Enforce `cfg.locked_assignments` on a unary matrix `u` by masking `U(s_lock, k != k_lock) = FORBID`
/// and `U(s != s_lock, k_lock) = FORBID` while ensuring `U(s_lock, k_lock)` is finite.
///
/// No-op when `cfg.locked_assignments` is empty.
pub fn apply_locked_assignments_to_unary(
    u: &mut Matrix2D,
    systems: &[SpinSystem],
    sequence: &[String],
    eff_priors: &Priors,
    cfg: &ScoreConfig,
) {
    if cfg.locked_assignments.is_empty() {
        return;
    }
    let valid = resolve_locked_assignments(systems.len(), sequence, &cfg.locked_assignments);
    if valid.is_empty() {
        return;
    }
    let s_len = u.rows;
    let k_len = u.cols;
    for &(s_lock, k_lock) in &valid {
        let mut lock_sc = u.get(s_lock, k_lock);
        if lock_sc <= FORBID * 0.5 || !lock_sc.is_finite() {
            let ss = &systems[s_lock];
            let res = &sequence[k_lock];
            let prev = &sequence[k_lock - 1];
            let cov_k = delta_covalent(sequence, k_lock);
            let cov_km1 = delta_covalent(sequence, k_lock - 1);
            let mut sc = 0.0;
            for ((atom, off), raw_cands) in &ss.cands {
                let cands: Vec<(f64, usize)> =
                    raw_cands.iter().copied().filter(|c| c.1 > 0).collect();
                if cands.is_empty() {
                    continue;
                }
                let (target_res, cov) = if *off == 0 {
                    (res, cov_k)
                } else {
                    (prev, cov_km1)
                };
                let d_cov = covalent_atom_offset(cov, atom);
                sc += candidate_llr_shifted(
                    eff_priors,
                    target_res,
                    atom,
                    &cands,
                    cfg.support_bonus,
                    d_cov,
                )
                .0;
            }
            if cfg.use_hn {
                sc += eff_priors.llr_shifted(res, "N", ss.n, cov_k.0)
                    + eff_priors.llr(res, "H", ss.h);
            }
            sc -= cfg.assign_cost;
            lock_sc = if sc.is_finite() { sc.max(-500.0) } else { 0.0 };
        }
        for k in 0..k_len {
            u.set(s_lock, k, FORBID);
        }
        for s in 0..s_len {
            u.set(s, k_lock, FORBID);
        }
        u.set(s_lock, k_lock, lock_sc);
    }
}

pub fn pair_matrix(systems: &[SpinSystem], cfg: &ScoreConfig) -> Matrix2D {
    let s_len = systems.len();
    let max_n = systems
        .iter()
        .flat_map(|ss| ss.cands.values().map(|c| c.len()))
        .max()
        .unwrap_or(1)
        .max(1);

    let mut p = Matrix2D::filled(s_len, s_len, 0.0);
    let log_range = cfg.range_ppm.ln();

    for atom in ["CA", "CB", "C"] {
        let sig = cfg.sigma(atom);
        let const_term = -(sig * TAU.sqrt()).ln() + log_range;
        let key_i = (atom.to_string(), 0);
        let key_im1 = (atom.to_string(), -1);

        for s in 0..s_len {
            let Some(a_raw) = systems[s].cands.get(&key_i) else {
                continue;
            };
            let a_slice = &a_raw[..a_raw.len().min(max_n)];
            if a_slice.is_empty() {
                continue;
            }
            let na = a_slice.len();

            for (t, sys_t) in systems.iter().enumerate().take(s_len) {
                if s == t {
                    continue;
                }
                let Some(b_raw) = sys_t.cands.get(&key_im1) else {
                    continue;
                };
                let b_slice = &b_raw[..b_raw.len().min(max_n)];
                if b_slice.is_empty() {
                    continue;
                }
                let nb = b_slice.len();

                let mut best = f64::NEG_INFINITY;
                for &(va, _) in a_slice {
                    for &(vb, _) in b_slice {
                        let d = (va - vb) / sig;
                        let term = (-0.5 * d * d + const_term).max(cfg.pair_floor);
                        if term > best {
                            best = term;
                        }
                    }
                }
                if best.is_finite() {
                    let pen = ((na * nb) as f64).ln();
                    let prev_val = p.get(s, t);
                    p.set(s, t, prev_val + best - pen);
                }
            }
        }
    }
    for s in 0..s_len {
        p.set(s, s, FORBID);
    }
    p
}

const MAX_SEQ_PARTNERS: usize = 8;
const P_RESTORE_EVICT: f64 = 0.15;

struct Annealer<'a> {
    u: &'a Matrix2D,
    p: &'a Matrix2D,
    rng: Rng,
    s_len: usize,
    k_len: usize,
    slots: Vec<Vec<usize>>,
    free_spins: Vec<usize>,
    locked_s: Vec<bool>,
    locked_k: Vec<bool>,
    succ_partners: Vec<Vec<usize>>,
    pred_partners: Vec<Vec<usize>>,
    w_single: f64,
    w_chain_ext: f64,
    w_unassign: f64,
    w_local_shift: f64,
    w_blk_unif: f64,
    w_blk_ext: f64,
    p_short_blk: f64,
    sparse_alpha: f64,
    spin_prob: f64,
    at: Vec<i32>,
    pos: Vec<i32>,
    edge_marks: Vec<u32>,
    edges_buf: Vec<usize>,
    mark_gen: u32,
}

impl<'a> Annealer<'a> {
    fn new(u: &'a Matrix2D, p: &'a Matrix2D, rng: Rng, locks: &[(usize, usize)]) -> Self {
        let s_len = u.rows;
        let k_len = u.cols;
        let slots: Vec<Vec<usize>> = (0..s_len)
            .map(|s| {
                (0..k_len)
                    .filter(|&k| u.get(s, k) > FORBID * 0.5)
                    .collect::<Vec<usize>>()
            })
            .collect();
        let mut locked_s = vec![false; s_len];
        let mut locked_k = vec![false; k_len];
        let mut at = vec![-1; k_len];
        let mut pos = vec![-1; s_len];
        for &(s_lock, k_lock) in locks {
            if s_lock < s_len && k_lock < k_len && !locked_s[s_lock] && !locked_k[k_lock] {
                locked_s[s_lock] = true;
                locked_k[k_lock] = true;
                at[k_lock] = s_lock as i32;
                pos[s_lock] = k_lock as i32;
            }
        }
        let free_spins: Vec<usize> = (0..s_len)
            .filter(|&s| !locked_s[s] && !slots[s].is_empty())
            .collect();

        let mut succ_partners = vec![Vec::new(); s_len];
        let mut pred_partners = vec![Vec::new(); s_len];
        let mut best_edges: Vec<f64> = Vec::with_capacity(free_spins.len());
        for s in 0..s_len {
            let mut succ: Vec<(usize, f64)> = free_spins
                .iter()
                .copied()
                .filter(|&t| t != s && p.get(s, t) > 0.0)
                .map(|t| (t, p.get(s, t)))
                .collect();
            succ.sort_by(|a, b| b.1.total_cmp(&a.1));
            if !locked_s[s] && !slots[s].is_empty() {
                if let Some(&(_, top_p)) = succ.first() {
                    best_edges.push(top_p);
                }
            }
            succ.truncate(MAX_SEQ_PARTNERS);
            succ_partners[s] = succ.into_iter().map(|(t, _)| t).collect();

            let mut pred: Vec<(usize, f64)> = free_spins
                .iter()
                .copied()
                .filter(|&t| t != s && p.get(t, s) > 0.0)
                .map(|t| (t, p.get(t, s)))
                .collect();
            pred.sort_by(|a, b| b.1.total_cmp(&a.1));
            pred.truncate(MAX_SEQ_PARTNERS);
            pred_partners[s] = pred.into_iter().map(|(t, _)| t).collect();
        }

        // Static connectivity-sparsity factor in [0, 1] determined once from `max_{s != t} P(s, t)`:
        // In `pair_matrix`, single-nucleus CA-only has mathematical ceiling <= 4.39 nats and 2-nucleus
        // HNCACB-absent (CA+C) has ceiling <= 8.77 nats (`sparse_alpha = 1.0`), whereas 3-nucleus
        // (CA+CB+C) datasets reach ~11.0–12.6 nats (`sparse_alpha = 0.0`).
        let p_max = best_edges.iter().copied().fold(0.0_f64, f64::max);
        let sparse_alpha = if best_edges.is_empty() {
            0.0
        } else {
            ((10.3 - p_max) / 1.4).clamp(0.0, 1.0)
        };
        let lerp = |dense: f64, sparse: f64| dense + sparse_alpha * (sparse - dense);
        let w_single = lerp(0.545, 0.385);
        let w_chain_ext = lerp(0.125, 0.26);
        let w_unassign = lerp(0.04, 0.015);
        let w_local_shift = lerp(0.04, 0.12);
        let w_blk_unif = lerp(0.22, 0.12);
        let w_blk_ext = lerp(0.03, 0.10);
        let p_short_blk = lerp(0.20, 0.50);
        let spin_prob = if free_spins.is_empty() {
            0.0
        } else {
            1.0 / (free_spins.len() as f64)
        };

        Self {
            u,
            p,
            rng,
            s_len,
            k_len,
            slots,
            free_spins,
            locked_s,
            locked_k,
            succ_partners,
            pred_partners,
            w_single,
            w_chain_ext,
            w_unassign,
            w_local_shift,
            w_blk_unif,
            w_blk_ext,
            p_short_blk,
            sparse_alpha,
            spin_prob,
            at,
            pos,
            edge_marks: vec![0; k_len],
            edges_buf: Vec::with_capacity(k_len),
            mark_gen: 1,
        }
    }

    #[inline]
    fn sample_spin(&mut self) -> Option<usize> {
        if self.free_spins.is_empty() {
            None
        } else {
            Some(self.free_spins[self.rng.usize(self.free_spins.len())])
        }
    }

    #[inline]
    fn count_restore_targets(&self, u_spin: usize, occ: usize, k: usize, cur_at: &[i32]) -> usize {
        self.slots[occ]
            .iter()
            .filter(|&&ks_cand| {
                ks_cand != k
                    && !self.locked_k[ks_cand]
                    && cur_at[ks_cand] < 0
                    && self.u.get(u_spin, ks_cand) <= FORBID * 0.5
            })
            .count()
    }

    #[inline]
    fn nth_restore_target(
        &self,
        u_spin: usize,
        occ: usize,
        k: usize,
        cur_at: &[i32],
        nth: usize,
    ) -> Option<usize> {
        self.slots[occ]
            .iter()
            .copied()
            .filter(|&ks_cand| {
                ks_cand != k
                    && !self.locked_k[ks_cand]
                    && cur_at[ks_cand] < 0
                    && self.u.get(u_spin, ks_cand) <= FORBID * 0.5
            })
            .nth(nth)
    }

    #[inline]
    fn reservoir_weight(&self, s: usize) -> f64 {
        1.0 + (1.0 - self.sparse_alpha)
            * (self.w_unassign / self.w_single)
            * (self.slots[s].len() as f64)
    }

    #[inline]
    fn unassign_prob(&self, s: usize) -> f64 {
        self.spin_prob * (self.w_single / (self.slots[s].len() as f64) + self.w_unassign)
    }

    #[inline]
    fn place_prob(&self, s: usize, k: usize, cur_at: &[i32]) -> f64 {
        let mut q = self.spin_prob * self.w_single / (self.slots[s].len() as f64);
        if self.k_len > 1 {
            let base_ext = self.w_chain_ext / (2.0 * ((self.k_len - 1) as f64));
            if k > 1 && cur_at[k - 1] >= 0 {
                let sa = cur_at[k - 1] as usize;
                let succ = &self.succ_partners[sa];
                if succ.contains(&s) {
                    q += base_ext / (succ.len() as f64);
                }
            }
            if k + 1 < self.k_len && cur_at[k + 1] >= 0 {
                let sb = cur_at[k + 1] as usize;
                let pred = &self.pred_partners[sb];
                if pred.contains(&s) {
                    q += base_ext / (pred.len() as f64);
                }
            }
        }
        q
    }

    #[inline]
    fn nonoverlap_swap_prob(
        &self,
        l: usize,
        a: usize,
        b: usize,
        max_l: usize,
        cur_at: &[i32],
    ) -> f64 {
        let mw = (self.k_len - l) as f64;
        let short_max = max_l.min(3);
        let p_l_unif = if max_l >= 2 {
            let base = (1.0 - self.p_short_blk) / (max_l as f64);
            let short = if (2..=short_max).contains(&l) {
                self.p_short_blk / ((short_max - 1) as f64)
            } else {
                0.0
            };
            base + short
        } else {
            1.0
        };
        let mut q = self.w_blk_unif * p_l_unif * (2.0 / (mw * mw));

        let ext_eligible = if max_l >= 2 {
            (2..=short_max).contains(&l)
        } else {
            l == 1
        };
        if ext_eligible && self.k_len > 1 {
            let n_ext_l = if max_l >= 2 {
                (short_max - 1) as f64
            } else {
                1.0
            };
            let base_ext = self.w_blk_ext / (n_ext_l * 2.0 * ((self.k_len - 1) as f64));
            if a > 1 && cur_at[a - 1] >= 0 && cur_at[b] >= 0 {
                let succ = &self.succ_partners[cur_at[a - 1] as usize];
                let sb = cur_at[b] as usize;
                if succ.contains(&sb) {
                    q += base_ext / (succ.len() as f64);
                }
            }
            if b > 1 && cur_at[b - 1] >= 0 && cur_at[a] >= 0 {
                let succ = &self.succ_partners[cur_at[b - 1] as usize];
                let sa = cur_at[a] as usize;
                if succ.contains(&sa) {
                    q += base_ext / (succ.len() as f64);
                }
            }
            if a + l < self.k_len && cur_at[a + l] >= 0 && cur_at[b + l - 1] >= 0 {
                let pred = &self.pred_partners[cur_at[a + l] as usize];
                let sb_end = cur_at[b + l - 1] as usize;
                if pred.contains(&sb_end) {
                    q += base_ext / (pred.len() as f64);
                }
            }
            if b + l < self.k_len && cur_at[b + l] >= 0 && cur_at[a + l - 1] >= 0 {
                let pred = &self.pred_partners[cur_at[b + l] as usize];
                let sa_end = cur_at[a + l - 1] as usize;
                if pred.contains(&sa_end) {
                    q += base_ext / (pred.len() as f64);
                }
            }
        }
        q
    }

    fn local(&mut self, changed_residues: &[usize]) -> f64 {
        if self.k_len == 0 {
            return 0.0;
        }
        self.mark_gen = self.mark_gen.wrapping_add(1);
        if self.mark_gen == 0 {
            self.edge_marks.fill(0);
            self.mark_gen = 1;
        }
        let gen = self.mark_gen;
        let mut sc = 0.0;
        self.edges_buf.clear();

        for &k in changed_residues {
            let s = self.at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
            if k > 0 && k - 1 < self.k_len - 1 && self.edge_marks[k - 1] != gen {
                self.edge_marks[k - 1] = gen;
                self.edges_buf.push(k - 1);
            }
            if k < self.k_len - 1 && self.edge_marks[k] != gen {
                self.edge_marks[k] = gen;
                self.edges_buf.push(k);
            }
        }
        for &e in &self.edges_buf {
            let a = self.at[e];
            let b = self.at[e + 1];
            if a >= 0 && b >= 0 {
                sc += self.p.get(a as usize, b as usize);
            }
        }
        sc
    }

    fn total(&self) -> f64 {
        let mut sc = 0.0;
        for k in 0..self.k_len {
            let s = self.at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
        }
        for e in 0..self.k_len.saturating_sub(1) {
            let a = self.at[e];
            let b = self.at[e + 1];
            if a >= 0 && b >= 0 {
                sc += self.p.get(a as usize, b as usize);
            }
        }
        sc
    }

    fn apply(&mut self, changes: &[(usize, i32)], undo: &mut Vec<(usize, i32)>) {
        undo.clear();
        for &(k, _) in changes {
            undo.push((k, self.at[k]));
        }
        for &(k, _) in changes {
            let prev_s = self.at[k];
            if prev_s >= 0 {
                self.pos[prev_s as usize] = -1;
            }
        }
        for &(k, s) in changes {
            self.at[k] = s;
        }
        for &(k, s) in changes {
            if s >= 0 {
                self.pos[s as usize] = k as i32;
            }
        }
    }

    fn run(&mut self, steps: usize, t0: f64, t1: f64, max_block: usize) -> f64 {
        if self.s_len == 0 || self.k_len == 0 {
            return 0.0;
        }
        let mut cur = self.total();
        let mut best = cur;
        let mut best_at = self.at.clone();
        if self.free_spins.is_empty() || self.k_len <= 1 || steps == 0 {
            return best;
        }

        #[derive(Clone, Copy)]
        enum MoveHastings {
            Symmetric,
            Unassign {
                s: usize,
                k: usize,
            },
            Place {
                s: usize,
                k: usize,
                ks: i32,
                t: i32,
                q_fwd: f64,
            },
            Evict {
                s: usize,
                k: usize,
                evicted: usize,
                q_fwd: f64,
            },
            RestoreEvict {
                s: usize,
                occ: usize,
                k: usize,
                q_fwd: f64,
            },
            BlockSwap {
                l: usize,
                a: usize,
                b: usize,
                max_l: usize,
                q_fwd: f64,
            },
        }

        let mut changes: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 8);
        let mut affected: Vec<usize> = Vec::with_capacity(max_block * 2 + 8);
        let mut undo: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 8);
        let mut dummy: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 8);
        let denom = (steps.saturating_sub(1)).max(1) as f64;
        let ratio = t1 / t0;

        let c1 = self.w_single;
        let c2 = c1 + self.w_chain_ext;
        let c3 = c2 + self.w_unassign;
        let c4 = c3 + self.w_local_shift;
        let c5 = c4 + self.w_blk_unif;

        for step in 0..steps {
            let temp = t0 * ratio.powf((step as f64) / denom);
            changes.clear();
            let mh: MoveHastings;

            let r = self.rng.f64();
            if r < c3 {
                if r < c2 {
                    let mut prop_sk: Option<(usize, usize)> = None;
                    let mut prop_unassign: Option<(usize, usize)> = None;
                    if r < c1 {
                        let Some(s) = self.sample_spin() else {
                            continue;
                        };
                        let k = self.slots[s][self.rng.usize(self.slots[s].len())];
                        if self.locked_k[k] {
                            continue;
                        }
                        if self.pos[s] == k as i32 {
                            prop_unassign = Some((s, k));
                        } else {
                            prop_sk = Some((s, k));
                        }
                    } else {
                        // Directed sequential chain-extension proposal (s_anc at k_anc -> top-P partner at k_anc ± 1)
                        let k_anc = self.rng.range_usize(1, self.k_len);
                        let forward = self.rng.usize(2) == 0;
                        let k = if forward {
                            if k_anc + 1 >= self.k_len {
                                continue;
                            }
                            k_anc + 1
                        } else {
                            if k_anc <= 1 {
                                continue;
                            }
                            k_anc - 1
                        };
                        if self.locked_k[k] || self.at[k_anc] < 0 {
                            continue;
                        }
                        let s_anc = self.at[k_anc] as usize;
                        let partners = if forward {
                            &self.succ_partners[s_anc]
                        } else {
                            &self.pred_partners[s_anc]
                        };
                        if partners.is_empty() {
                            continue;
                        }
                        let s = partners[self.rng.usize(partners.len())];
                        if self.pos[s] == k as i32 || self.u.get(s, k) <= FORBID * 0.5 {
                            continue;
                        }
                        prop_sk = Some((s, k));
                    }

                    if let Some((s, k)) = prop_unassign {
                        changes.push((k, -1));
                        mh = MoveHastings::Unassign { s, k };
                    } else if let Some((s, k)) = prop_sk {
                        let ks = self.pos[s];
                        let t = self.at[k];
                        if t >= 0 && self.locked_s[t as usize] {
                            continue;
                        }
                        if ks < 0 && t >= 0 {
                            let t_u = t as usize;
                            let n_elig = self.count_restore_targets(s, t_u, k, &self.at);
                            if n_elig > 0 && self.rng.f64() < P_RESTORE_EVICT {
                                let pick_idx = self.rng.usize(n_elig);
                                let Some(ks_cand) =
                                    self.nth_restore_target(s, t_u, k, &self.at, pick_idx)
                                else {
                                    continue;
                                };
                                let q_fwd = self.place_prob(s, k, &self.at) * P_RESTORE_EVICT
                                    / (n_elig as f64);
                                changes.push((k, s as i32));
                                changes.push((ks_cand, t));
                                mh = MoveHastings::RestoreEvict {
                                    s,
                                    occ: t_u,
                                    k,
                                    q_fwd,
                                };
                            } else {
                                let disp_factor = if n_elig > 0 {
                                    1.0 - P_RESTORE_EVICT
                                } else {
                                    1.0
                                };
                                let q_fwd = self.place_prob(s, k, &self.at) * disp_factor;
                                changes.push((k, s as i32));
                                mh = MoveHastings::Place { s, k, ks, t, q_fwd };
                            }
                        } else if ks >= 0 && t >= 0 {
                            let ks_u = ks as usize;
                            let t_u = t as usize;
                            if self.u.get(t_u, ks_u) > FORBID * 0.5 {
                                let q_fwd = self.place_prob(s, k, &self.at)
                                    + self.place_prob(t_u, ks_u, &self.at);
                                changes.push((k, s as i32));
                                changes.push((ks_u, t));
                                mh = MoveHastings::Place { s, k, ks, t, q_fwd };
                            } else {
                                let q_fwd = self.place_prob(s, k, &self.at);
                                changes.push((k, s as i32));
                                changes.push((ks_u, -1));
                                mh = MoveHastings::Evict {
                                    s,
                                    k,
                                    evicted: t_u,
                                    q_fwd,
                                };
                            }
                        } else {
                            let q_fwd = self.place_prob(s, k, &self.at);
                            changes.push((k, s as i32));
                            if ks >= 0 {
                                changes.push((ks as usize, -1));
                            }
                            mh = MoveHastings::Place { s, k, ks, t, q_fwd };
                        }
                    } else {
                        continue;
                    }
                } else {
                    let Some(s) = self.sample_spin() else {
                        continue;
                    };
                    if self.pos[s] < 0 {
                        continue;
                    }
                    let k = self.pos[s] as usize;
                    changes.push((k, -1));
                    mh = MoveHastings::Unassign { s, k };
                }
            } else if r < c4 {
                // Symmetric contiguous segment register shift (±1..±3, L=1..4) via interval rotation
                let max_shift_l = 4usize.min(self.k_len.saturating_sub(1));
                if max_shift_l == 0 {
                    continue;
                }
                let l = self.rng.range_usize(1, max_shift_l + 1);
                let k1 = self.rng.range_usize(1, self.k_len - l + 1);
                const SHIFTS: [isize; 6] = [-3, -2, -1, 1, 2, 3];
                let d = SHIFTS[self.rng.usize(SHIFTS.len())];
                let k2_i = (k1 as isize) + d;
                if k2_i < 1 || (k2_i as usize) + l > self.k_len {
                    continue;
                }
                let k2 = k2_i as usize;
                let lo = k1.min(k2);
                let span_end = k1.max(k2) + l;
                let span = span_end - lo;
                if (lo..span_end).any(|idx| self.locked_k[idx]) {
                    continue;
                }
                let shift_amt = k1.abs_diff(k2);
                let mut valid = true;
                for idx in 0..span {
                    let src_idx = if k2 > k1 {
                        lo + (idx + span - shift_amt) % span
                    } else {
                        lo + (idx + shift_amt) % span
                    };
                    let dest_k = lo + idx;
                    let spin = self.at[src_idx];
                    if spin >= 0 && self.u.get(spin as usize, dest_k) <= FORBID * 0.5 {
                        valid = false;
                        break;
                    }
                    changes.push((dest_k, spin));
                }
                if !valid {
                    continue;
                }
                mh = MoveHastings::Symmetric;
            } else {
                // Non-overlapping contiguous block swap (uniform + partner-guided dipeptide/tripeptide splice)
                let max_l = max_block.min((self.k_len.saturating_sub(1)) / 2);
                if max_l == 0 {
                    continue;
                }
                let short_max = max_l.min(3);
                let (l, a, b) = if r < c5 {
                    let l = if max_l >= 2
                        && self.p_short_blk > 0.0
                        && self.rng.f64() < self.p_short_blk
                    {
                        self.rng.range_usize(2, short_max + 1)
                    } else {
                        self.rng.range_usize(1, max_l + 1)
                    };
                    let k1 = self.rng.range_usize(1, self.k_len - l + 1);
                    let k2 = self.rng.range_usize(1, self.k_len - l + 1);
                    if k1.abs_diff(k2) < l {
                        continue;
                    }
                    (l, k1.min(k2), k1.max(k2))
                } else {
                    let l = if max_l >= 2 {
                        self.rng.range_usize(2, short_max + 1)
                    } else {
                        1
                    };
                    let k_anc = self.rng.range_usize(1, self.k_len);
                    let forward = self.rng.usize(2) == 0;
                    if self.at[k_anc] < 0 {
                        continue;
                    }
                    let s_anc = self.at[k_anc] as usize;
                    let (w_dst, w_src) = if forward {
                        let dst = k_anc + 1;
                        if dst + l > self.k_len {
                            continue;
                        }
                        let succ = &self.succ_partners[s_anc];
                        if succ.is_empty() {
                            continue;
                        }
                        let sp = succ[self.rng.usize(succ.len())];
                        if self.pos[sp] < 1 {
                            continue;
                        }
                        let src = self.pos[sp] as usize;
                        if src + l > self.k_len {
                            continue;
                        }
                        (dst, src)
                    } else {
                        if k_anc <= l {
                            continue;
                        }
                        let dst = k_anc - l;
                        let pred = &self.pred_partners[s_anc];
                        if pred.is_empty() {
                            continue;
                        }
                        let sp = pred[self.rng.usize(pred.len())];
                        if self.pos[sp] < l as i32 {
                            continue;
                        }
                        let src = (self.pos[sp] as usize) + 1 - l;
                        (dst, src)
                    };
                    if w_src.abs_diff(w_dst) < l {
                        continue;
                    }
                    (l, w_src.min(w_dst), w_src.max(w_dst))
                };

                let mut valid = true;
                for i in 0..l {
                    if self.locked_k[a + i] || self.locked_k[b + i] {
                        valid = false;
                        break;
                    }
                    let sa = self.at[a + i];
                    let sb = self.at[b + i];
                    if (sb >= 0 && self.u.get(sb as usize, a + i) <= FORBID * 0.5)
                        || (sa >= 0 && self.u.get(sa as usize, b + i) <= FORBID * 0.5)
                    {
                        valid = false;
                        break;
                    }
                    changes.push((a + i, sb));
                    changes.push((b + i, sa));
                }
                if !valid {
                    continue;
                }
                let q_fwd = self.nonoverlap_swap_prob(l, a, b, max_l, &self.at);
                mh = MoveHastings::BlockSwap {
                    l,
                    a,
                    b,
                    max_l,
                    q_fwd,
                };
            }

            if changes.is_empty() {
                continue;
            }
            affected.clear();
            for &(k, _) in &changes {
                affected.push(k);
            }
            let before = self.local(&affected);
            self.apply(&changes, &mut undo);
            let delta = self.local(&affected) - before;

            let log_q_ratio = match mh {
                MoveHastings::Symmetric => 0.0,
                MoveHastings::Unassign { s, k } => {
                    let q_fwd = self.unassign_prob(s);
                    let q_rev = self.place_prob(s, k, &self.at);
                    (self.reservoir_weight(s) * q_rev / q_fwd).ln()
                }
                MoveHastings::Place { s, k, ks, t, q_fwd } => {
                    let (q_rev, g_ratio) = if ks < 0 && t < 0 {
                        (self.unassign_prob(s), 1.0 / self.reservoir_weight(s))
                    } else if ks >= 0 && t < 0 {
                        (self.place_prob(s, ks as usize, &self.at), 1.0)
                    } else if ks < 0 && t >= 0 {
                        let t_u = t as usize;
                        let rev_elig = self.count_restore_targets(t_u, s, k, &self.at);
                        let rev_factor = if rev_elig > 0 {
                            1.0 - P_RESTORE_EVICT
                        } else {
                            1.0
                        };
                        (
                            self.place_prob(t_u, k, &self.at) * rev_factor,
                            self.reservoir_weight(t_u) / self.reservoir_weight(s),
                        )
                    } else {
                        (
                            self.place_prob(s, ks as usize, &self.at)
                                + self.place_prob(t as usize, k, &self.at),
                            1.0,
                        )
                    };
                    (g_ratio * q_rev / q_fwd).ln()
                }
                MoveHastings::Evict {
                    s,
                    k,
                    evicted,
                    q_fwd,
                } => {
                    let rev_elig = self.count_restore_targets(evicted, s, k, &self.at).max(1);
                    let q_rev =
                        self.place_prob(evicted, k, &self.at) * P_RESTORE_EVICT / (rev_elig as f64);
                    (self.reservoir_weight(evicted) * q_rev / q_fwd).ln()
                }
                MoveHastings::RestoreEvict { s, occ, k, q_fwd } => {
                    let q_rev = self.place_prob(occ, k, &self.at);
                    ((q_rev / q_fwd) / self.reservoir_weight(s)).ln()
                }
                MoveHastings::BlockSwap {
                    l,
                    a,
                    b,
                    max_l,
                    q_fwd,
                } => {
                    let q_rev = self.nonoverlap_swap_prob(l, a, b, max_l, &self.at);
                    (q_rev / q_fwd).ln()
                }
            };

            let log_accept = delta / temp + log_q_ratio;
            if log_accept >= 0.0 || self.rng.f64() < log_accept.exp() {
                cur += delta;
                if cur > best {
                    best = cur;
                    best_at.copy_from_slice(&self.at);
                }
            } else {
                self.apply(&undo, &mut dummy);
            }
        }

        self.at = best_at;
        self.pos.fill(-1);
        for (k, &s) in self.at.iter().enumerate() {
            if s >= 0 {
                self.pos[s as usize] = k as i32;
            }
        }
        best
    }
}

#[derive(Clone, Debug)]
pub struct AssignResult {
    /// Spin system -> residue index (`-1` if unassigned).
    pub position: Vec<i32>,
    /// Spin system -> fraction of runs agreeing with the highest-scoring run.
    pub confidence: Vec<f64>,
    pub score: f64,
    pub run_scores: Vec<f64>,
}

/// Aggregate multi-replica trajectories (`positions` + `scores`) into a consensus `AssignResult`.
pub fn consensus_from_runs(positions: &[Vec<i32>], scores: Vec<f64>) -> AssignResult {
    if positions.is_empty() {
        return AssignResult {
            position: Vec::new(),
            confidence: Vec::new(),
            score: 0.0,
            run_scores: Vec::new(),
        };
    }
    let best_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let ref_pos = positions[best_idx].clone();
    let s_len = ref_pos.len();
    let n_runs = positions.len() as f64;
    let mut confidence = vec![0.0; s_len];
    for s in 0..s_len {
        if ref_pos[s] >= 0 {
            let agree = positions.iter().filter(|p| p[s] == ref_pos[s]).count() as f64;
            confidence[s] = agree / n_runs;
        }
    }
    AssignResult {
        position: ref_pos,
        confidence,
        score: scores[best_idx],
        run_scores: scores,
    }
}

/// Calibrate `standard` multi-replica consensus (`agree / n_runs`) via parameter-free physical
/// (`T = 1.0` nat) conditional Boltzmann + score-weighted elite-basin replica consensus +
/// full connected-stretch register-translation marginal damping, followed by the 10-fold
/// LOPO-validated monotonic calibration map (`P1-6`).
pub fn calibrate_standard_confidence(
    result: &mut AssignResult,
    positions: Option<&[Vec<i32>]>,
    u: &Matrix2D,
    p: &Matrix2D,
) {
    calibrate_standard_confidence_impl(result, positions, u, p, true);
}

fn calibrate_standard_confidence_impl(
    result: &mut AssignResult,
    positions: Option<&[Vec<i32>]>,
    u: &Matrix2D,
    p: &Matrix2D,
    apply_calibration_map: bool,
) {
    let s_len = u.rows;
    let k_len = u.cols;
    if s_len == 0 || k_len == 0 || result.position.len() != s_len {
        return;
    }
    let c0_raw = result.confidence.clone();
    let mut at = vec![-1i32; k_len];
    let mut n_assigned = 0usize;
    let mut sum_c0_raw = 0.0_f64;
    for (s, &k) in result.position.iter().enumerate() {
        if k >= 0 && (k as usize) < k_len {
            at[k as usize] = s as i32;
            n_assigned += 1;
            sum_c0_raw += c0_raw[s];
        }
    }
    if n_assigned == 0 {
        return;
    }
    let mut col_assignable = vec![false; k_len];
    for (k, slot) in col_assignable.iter_mut().enumerate().skip(1) {
        *slot = (0..s_len).any(|sp| u.get(sp, k) > FORBID * 0.5);
    }

    // Count 3D carbon-supported spin systems directly from non-zero off-diagonal `P` rows/cols
    // (disordered NMR-invisible FASTA tails produce zero carbon peaks and therefore do NOT lower
    // `placed_carbon_frac`, whereas unplaced real spin systems in the reservoir DO lower it).
    let mut n_carbon_spins = 0usize;
    let mut top_edges: Vec<f64> = Vec::with_capacity(s_len);
    for s in 0..s_len {
        let mut has_c = false;
        let mut best_p = 0.0_f64;
        for t in 0..s_len {
            if s != t {
                let pv = p.get(s, t);
                let qv = p.get(t, s);
                if pv.abs() > 1e-12 || qv.abs() > 1e-12 {
                    has_c = true;
                }
                if pv > best_p {
                    best_p = pv;
                }
            }
        }
        if has_c {
            n_carbon_spins += 1;
        }
        if best_p > 0.0 {
            top_edges.push(best_p);
        }
    }
    let placed_carbon_frac = if n_carbon_spins == 0 {
        1.0
    } else {
        ((n_assigned as f64) / (n_carbon_spins as f64)).clamp(0.0, 1.0)
    };
    top_edges.sort_by(|a, b| b.total_cmp(a));
    let p_max = top_edges.first().copied().unwrap_or(0.0);
    // Without `C(0)` (`HNCACO` absent), `CA + CB` has strict ceiling `5.584 + 5.073 = 10.657` nats
    // (`p_max ~ 8.7` nats in practice), whereas 3-channel `(CA, CB, C)` reaches `12.3..13.3` nats.
    let two_channel_alpha = ((11.2 - p_max) / 2.2).clamp(0.0, 1.0);

    let mut n_linked_edges = 0usize;
    for k in 1..(k_len.saturating_sub(1)) {
        if at[k] >= 0 && at[k + 1] >= 0 && p.get(at[k] as usize, at[k + 1] as usize) > 0.0 {
            n_linked_edges += 1;
        }
    }
    let linked_edge_frac = (n_linked_edges as f64) / (n_assigned as f64);

    let mean_c0 = sum_c0_raw / (n_assigned as f64);
    let cov_gate = ((linked_edge_frac - 0.42) / 0.25).clamp(0.0, 1.0);
    let coherence = (((mean_c0 - 0.28) / 0.35).clamp(0.0, 1.0) * cov_gate).clamp(0.0, 1.0);

    // Score-weighted elite-basin replica consensus: kinetically trapped SA restarts far below
    // `result.score` receive exponentially small weight ONLY when both ensemble coherence and
    // backbone assignment coverage are high, with a 2-replica Laplace pseudocount so a single
    // outlier restart cannot inflate `c0` on multi-modal or incomplete landscapes.
    // Conversely, on sub-default restart ensembles (`n_runs < 16`, e.g. 12-replica CLI presets vs
    // the 16-replica default ensemble), raw empirical agreement `k / N` has elevated binomial
    // variance (`10/12 = 0.833` despite 2 dissenting trajectories), so apply conservative Laplace-1
    // shrinkage `k / (N + 1)` before mapping through the calibration curve.
    let mut c0 = c0_raw.clone();
    let n_runs_known = positions
        .map(|p| p.len())
        .unwrap_or(result.run_scores.len());
    if (1..16).contains(&n_runs_known) {
        let shrink = (n_runs_known as f64) / ((n_runs_known as f64) + 1.0);
        for s in 0..s_len {
            if result.position[s] >= 0 {
                c0[s] = (c0_raw[s] * shrink).clamp(0.0, 1.0);
            }
        }
    } else if let Some(pos_list) = positions {
        if pos_list.len() >= 24 && pos_list.len() == result.run_scores.len() && coherence > 0.0 {
            let tau_ens = (0.20 * (n_assigned as f64)).max(9.0);
            let mut weights = Vec::with_capacity(pos_list.len());
            let mut sum_w = 0.0_f64;
            for &sc in &result.run_scores {
                let w = ((sc - result.score) / tau_ens).clamp(-40.0, 0.0).exp();
                weights.push(w);
                sum_w += w;
            }
            if sum_w >= 2.5 {
                for s in 0..s_len {
                    let k_ref = result.position[s];
                    if k_ref < 0 {
                        continue;
                    }
                    let mut agree_w = 0.0_f64;
                    for (pos_r, &w) in pos_list.iter().zip(weights.iter()) {
                        if pos_r.get(s).copied() == Some(k_ref) {
                            agree_w += w;
                        }
                    }
                    let c_elite = (agree_w / (sum_w + 2.0)).clamp(0.0, 1.0);
                    c0[s] = ((1.0 - coherence) * c0_raw[s] + coherence * c_elite.max(c0_raw[s]))
                        .clamp(0.0, 1.0);
                }
            }
        }
    }

    let site_energy = |spin: usize, slot: usize, cur_at: &[i32]| -> f64 {
        let mut sc = u.get(spin, slot);
        if slot > 0 && cur_at[slot - 1] >= 0 {
            sc += p.get(cur_at[slot - 1] as usize, spin);
        }
        if slot + 1 < k_len && cur_at[slot + 1] >= 0 {
            sc += p.get(spin, cur_at[slot + 1] as usize);
        }
        sc
    };

    // Pass A: Unit-temperature (`T = 1.0` nat) single-spin conditional Boltzmann beliefs `b1`
    let mut b1 = vec![0.0_f64; s_len];
    for s in 0..s_len {
        let k_raw = result.position[s];
        if k_raw < 0 || c0[s] <= 0.0 {
            continue;
        }
        let k = k_raw as usize;
        let e_cur = site_energy(s, k, &at);
        if e_cur <= FORBID * 0.5 || !e_cur.is_finite() {
            continue;
        }
        let mut alt_part = (-e_cur).clamp(-40.0, 20.0).exp();

        for s_alt in 0..s_len {
            if s_alt != s && result.position[s_alt] < 0 && u.get(s_alt, k) > FORBID * 0.5 {
                let d = site_energy(s_alt, k, &at) - e_cur;
                alt_part += d.clamp(-40.0, 20.0).exp();
            }
        }

        for k_alt in 0..k_len {
            if k_alt == k || u.get(s, k_alt) <= FORBID * 0.5 {
                continue;
            }
            let t_raw = at[k_alt];
            if t_raw < 0 {
                let mut e_alt = u.get(s, k_alt);
                if k_alt > 0 && k_alt - 1 != k && at[k_alt - 1] >= 0 {
                    e_alt += p.get(at[k_alt - 1] as usize, s);
                }
                if k_alt + 1 < k_len && k_alt + 1 != k && at[k_alt + 1] >= 0 {
                    e_alt += p.get(s, at[k_alt + 1] as usize);
                }
                alt_part += (e_alt - e_cur).clamp(-40.0, 20.0).exp();
            } else {
                let t = t_raw as usize;
                if u.get(t, k) <= FORBID * 0.5 {
                    continue;
                }
                let (a, sa, b, sb) = if k < k_alt {
                    (k, s, k_alt, t)
                } else {
                    (k_alt, t, k, s)
                };
                let mut before_2 = u.get(sa, a) + u.get(sb, b);
                let mut after_2 = u.get(sb, a) + u.get(sa, b);
                if a > 0 && at[a - 1] >= 0 {
                    let left = at[a - 1] as usize;
                    before_2 += p.get(left, sa);
                    after_2 += p.get(left, sb);
                }
                if b + 1 < k_len && at[b + 1] >= 0 {
                    let right = at[b + 1] as usize;
                    before_2 += p.get(sb, right);
                    after_2 += p.get(sa, right);
                }
                if a + 1 == b {
                    before_2 += p.get(sa, sb);
                    after_2 += p.get(sb, sa);
                } else {
                    if a + 1 < k_len && at[a + 1] >= 0 {
                        let mid_a = at[a + 1] as usize;
                        before_2 += p.get(sa, mid_a);
                        after_2 += p.get(sb, mid_a);
                    }
                    if b > 0 && at[b - 1] >= 0 {
                        let mid_b = at[b - 1] as usize;
                        before_2 += p.get(mid_b, sb);
                        after_2 += p.get(mid_b, sa);
                    }
                }
                alt_part += (after_2 - before_2).clamp(-40.0, 20.0).exp();
            }
        }
        let p_cond = 1.0 / (1.0 + alt_part);
        b1[s] = (c0[s] * p_cond).clamp(0.0, 1.0);
    }

    if k_len <= 1 {
        for s in 0..s_len {
            if result.position[s] >= 0 && c0[s] > 0.0 {
                result.confidence[s] = apply_standard_calibration_map(b1[s]);
            }
        }
        return;
    }

    // Identify connected sequential stretches along the consensus backbone
    let mut chain_len = vec![1usize; s_len];
    let mut chain_p_sum = vec![0.0_f64; s_len];
    let mut maximal_stretches: Vec<(usize, usize)> = Vec::new();
    {
        let mut k = 1usize;
        while k < k_len {
            if at[k] < 0 {
                k += 1;
                continue;
            }
            let start = k;
            let mut p_acc = 0.0_f64;
            while k + 1 < k_len && at[k + 1] >= 0 {
                let pv = p.get(at[k] as usize, at[k + 1] as usize);
                if pv >= 1.5 {
                    p_acc += pv;
                    k += 1;
                } else {
                    break;
                }
            }
            let end = k;
            let clen = end - start + 1;
            maximal_stretches.push((start, end));
            for &sp_raw in &at[start..=end] {
                let sp = sp_raw as usize;
                chain_len[sp] = clen;
                chain_p_sum[sp] = p_acc;
            }
            k += 1;
        }
    }

    // Pass B: Belief-weighted neighbor damping + full connected-stretch & subwindow (`L = 1..=10`) partition
    let clamp_w = |spin_idx: usize| -> f64 {
        let r = (b1[spin_idx] / 0.85).clamp(0.0, 1.0);
        r * r
    };
    let weighted_site = |spin: usize, slot: usize| -> f64 {
        let mut sc = u.get(spin, slot);
        if slot > 0 && at[slot - 1] >= 0 {
            let sl = at[slot - 1] as usize;
            sc += clamp_w(sl) * p.get(sl, spin);
        }
        if slot + 1 < k_len && at[slot + 1] >= 0 {
            let sr = at[slot + 1] as usize;
            sc += clamp_w(sr) * p.get(spin, sr);
        }
        sc
    };

    let mut bp_unassign = vec![f64::NEG_INFINITY; s_len];
    let mut bp_frag_dest = vec![vec![f64::NEG_INFINITY; k_len]; s_len];

    let mut eval_stretch = |a: usize, b: usize| {
        let len = b - a + 1;
        let mut u_self_sum = 0.0;
        let mut internal_p = 0.0;
        for off in 0..len {
            u_self_sum += u.get(at[a + off] as usize, a + off);
            if off + 1 < len {
                internal_p += p.get(at[a + off] as usize, at[a + off + 1] as usize);
            }
        }
        let mut bnd_self = 0.0;
        if a > 0 && at[a - 1] >= 0 {
            let sl = at[a - 1] as usize;
            bnd_self += clamp_w(sl) * p.get(sl, at[a] as usize);
        }
        if b + 1 < k_len && at[b + 1] >= 0 {
            let sr = at[b + 1] as usize;
            bnd_self += clamp_w(sr) * p.get(at[b] as usize, sr);
        }
        let e_frag_self = u_self_sum + bnd_self;
        let delta_un = -(e_frag_self + internal_p);
        for &spin_raw in &at[a..=b] {
            let sj = spin_raw as usize;
            if delta_un > bp_unassign[sj] {
                bp_unassign[sj] = delta_un;
            }
        }

        if len >= k_len {
            return;
        }
        for k2 in 1..=(k_len - len) {
            if k2 == a {
                continue;
            }
            if !(0..len).all(|off| u.get(at[a + off] as usize, k2 + off) > FORBID * 0.5) {
                continue;
            }
            let mut u_alt_sum = 0.0;
            let mut occ_cost = 0.0;
            let in_src = |idx: usize| idx >= a && idx <= b;
            let in_dst = |idx: usize| idx >= k2 && idx < k2 + len;
            for off in 0..len {
                let q = k2 + off;
                u_alt_sum += u.get(at[a + off] as usize, q);
                if !in_src(q) && at[q] >= 0 {
                    let occ = at[q] as usize;
                    let mut e_occ = u.get(occ, q);
                    if q > 0 && !in_src(q - 1) && !in_dst(q - 1) && at[q - 1] >= 0 {
                        let sl = at[q - 1] as usize;
                        e_occ += clamp_w(sl) * p.get(sl, occ);
                    }
                    if q + 1 < k_len && !in_src(q + 1) && at[q + 1] >= 0 {
                        let sr = at[q + 1] as usize;
                        let edge_w = if in_dst(q + 1) {
                            clamp_w(occ).min(clamp_w(sr))
                        } else {
                            clamp_w(occ) * clamp_w(sr)
                        };
                        occ_cost += edge_w * p.get(occ, sr).max(0.0);
                    }
                    occ_cost += clamp_w(occ) * e_occ.max(0.0);
                }
            }
            let mut bnd_alt = 0.0;
            if k2 > 0 && (k2 - 1 < a || k2 - 1 > b) && at[k2 - 1] >= 0 {
                let sl = at[k2 - 1] as usize;
                bnd_alt += clamp_w(sl) * p.get(sl, at[a] as usize);
            }
            if k2 + len < k_len && (k2 + len < a || k2 + len > b) && at[k2 + len] >= 0 {
                let sr = at[k2 + len] as usize;
                bnd_alt += clamp_w(sr) * p.get(at[b] as usize, sr);
            }
            let delta_frag = (u_alt_sum + bnd_alt - occ_cost) - e_frag_self;
            for off in 0..len {
                let sj = at[a + off] as usize;
                let k_dest = k2 + off;
                if delta_frag > bp_frag_dest[sj][k_dest] {
                    bp_frag_dest[sj][k_dest] = delta_frag;
                }
            }
        }
    };

    // 1. Evaluate every maximal connected stretch `[start..=end]` regardless of length (closes `L > 5` blind spot)
    for &(start, end) in &maximal_stretches {
        if end > start {
            eval_stretch(start, end);
        }
    }
    // 2. Also evaluate unanchored subwindows `L = 2..=8`
    let max_island_len = 8usize.min(k_len.saturating_sub(1));
    for a in 1..k_len {
        if at[a] < 0 {
            continue;
        }
        let left_anchored = a > 0
            && at[a - 1] >= 0
            && b1[at[a - 1] as usize] >= 0.85
            && p.get(at[a - 1] as usize, at[a] as usize) >= 2.5;
        if left_anchored {
            continue;
        }
        for len in 2..=max_island_len {
            let b = a + len - 1;
            if b >= k_len || at[b] < 0 {
                break;
            }
            let right_anchored = b + 1 < k_len
                && at[b + 1] >= 0
                && b1[at[b + 1] as usize] >= 0.85
                && p.get(at[b] as usize, at[b + 1] as usize) >= 2.5;
            if right_anchored {
                continue;
            }
            eval_stretch(a, b);
        }
    }

    // 3. Unassigned-dimer cavity competition: evaluate replacing adjacent occupied `(at[k], at[k+1])`
    // with unassigned linked pairs `(t, tr)` (`pos[t] < 0, pos[tr] < 0, P(t, tr) > 0`) from the reservoir.
    let mut unassigned_dimer_alt = vec![0.0_f64; s_len];
    {
        let unassigned_spins: Vec<usize> =
            (0..s_len).filter(|&sp| result.position[sp] < 0).collect();
        let mut un_dimers: Vec<(usize, usize, f64)> = Vec::new();
        for &t in &unassigned_spins {
            let mut succs: Vec<(usize, f64)> = unassigned_spins
                .iter()
                .copied()
                .filter(|&tr| tr != t && p.get(t, tr) > 0.5)
                .map(|tr| (tr, p.get(t, tr)))
                .collect();
            succs.sort_by(|a, b| b.1.total_cmp(&a.1));
            succs.truncate(6);
            for (tr, pv) in succs {
                un_dimers.push((t, tr, pv));
            }
        }
        if !un_dimers.is_empty() {
            for k in 1..k_len.saturating_sub(1) {
                if at[k] < 0 || at[k + 1] < 0 {
                    continue;
                }
                let sa = at[k] as usize;
                let sb = at[k + 1] as usize;
                let mut e_cur_pair = u.get(sa, k) + u.get(sb, k + 1) + p.get(sa, sb);
                if k > 1 && at[k - 1] >= 0 {
                    let sl = at[k - 1] as usize;
                    e_cur_pair += clamp_w(sl) * p.get(sl, sa);
                }
                if k + 2 < k_len && at[k + 2] >= 0 {
                    let sr = at[k + 2] as usize;
                    e_cur_pair += clamp_w(sr) * p.get(sb, sr);
                }
                let mut sum_exp = 0.0_f64;
                for &(t, tr, p_t_tr) in &un_dimers {
                    if u.get(t, k) <= FORBID * 0.5 || u.get(tr, k + 1) <= FORBID * 0.5 {
                        continue;
                    }
                    let mut e_alt_pair = u.get(t, k) + u.get(tr, k + 1) + p_t_tr;
                    if k > 1 && at[k - 1] >= 0 {
                        let sl = at[k - 1] as usize;
                        e_alt_pair += clamp_w(sl) * p.get(sl, t);
                    }
                    if k + 2 < k_len && at[k + 2] >= 0 {
                        let sr = at[k + 2] as usize;
                        e_alt_pair += clamp_w(sr) * p.get(tr, sr);
                    }
                    sum_exp += (e_alt_pair - e_cur_pair).clamp(-40.0, 20.0).exp();
                }
                unassigned_dimer_alt[sa] += sum_exp;
                unassigned_dimer_alt[sb] += sum_exp;
            }
        }
    }

    for s in 0..s_len {
        let k_raw = result.position[s];
        if k_raw < 0 || b1[s] <= 0.0 {
            if k_raw >= 0 {
                result.confidence[s] = 0.0;
            }
            continue;
        }
        let k = k_raw as usize;
        let e_self_w = weighted_site(s, k);
        let un_w = (-e_self_w).max(bp_unassign[s]);
        let mut alt_w = un_w.clamp(-40.0, 20.0).exp() + unassigned_dimer_alt[s];

        // Uncorroborated singleton & low-coherence / incomplete-backbone exclusion uncertainty
        let can_link = (k > 1 && col_assignable[k - 1]) || (k + 1 < k_len && col_assignable[k + 1]);
        if can_link && chain_len[s] == 1 {
            alt_w += 0.18 / coherence.max(0.22);
        }
        let soup_gap = (0.58 - mean_c0).max(0.0);
        let missing_gap = (0.52 - placed_carbon_frac).max(0.0) * (1.0 + 0.50 * two_channel_alpha);
        if soup_gap > 0.0 || missing_gap > 0.0 {
            let p_scale = if two_channel_alpha > 0.5 { 0.085 } else { 0.14 };
            let chain_shield = (-0.36 * ((chain_len[s].saturating_sub(1)) as f64)
                - p_scale * chain_p_sum[s].min(20.0))
            .exp();
            alt_w += (soup_gap * 4.2 + missing_gap * 3.0) * chain_shield;
        }
        // Global backbone-walk fragmentation uncertainty (`Wüthrich 1986`): when `< 52%` of placed
        // residues have any sequential `P(at[k], at[k+1]) > 0` link (average fragment length `< 2`
        // residues), combinatorial placement of disconnected 1-mer/2-mer islands across open sequence
        // slots remains globally degenerate even if SA restarts agree on the best local packing.
        let frag_gap = (0.52 - linked_edge_frac).max(0.0);
        if frag_gap > 0.0 {
            let frag_pen = (frag_gap / 0.11).powi(2)
                * 1.05
                * (-0.08 * ((chain_len[s].saturating_sub(1)) as f64)).exp();
            alt_w += frag_pen;
        }

        for t in 0..s_len {
            if t != s && result.position[t] < 0 && u.get(t, k) > FORBID * 0.5 {
                let delta = weighted_site(t, k) - e_self_w;
                alt_w += delta.clamp(-40.0, 20.0).exp();
            }
        }
        for k2 in 1..k_len {
            if k2 == k {
                continue;
            }
            let mut best_k2 = bp_frag_dest[s][k2];
            if u.get(s, k2) > FORBID * 0.5 {
                let mut e_k2 = u.get(s, k2);
                if k2 > 0 && k2 - 1 != k && at[k2 - 1] >= 0 {
                    let sl = at[k2 - 1] as usize;
                    e_k2 += clamp_w(sl) * p.get(sl, s);
                }
                if k2 + 1 < k_len && k2 + 1 != k && at[k2 + 1] >= 0 {
                    let sr = at[k2 + 1] as usize;
                    e_k2 += clamp_w(sr) * p.get(s, sr);
                }
                let occ_cost = if at[k2] >= 0 {
                    let t = at[k2] as usize;
                    clamp_w(t) * weighted_site(t, k2).max(0.0)
                } else {
                    0.0
                };
                best_k2 = best_k2.max(e_k2 - occ_cost - e_self_w);

                // Belief-weighted pairwise 2-swap `(s @ k) <-> (t @ k2)` when `k2` is occupied
                if at[k2] >= 0 {
                    let t = at[k2] as usize;
                    if u.get(t, k) > FORBID * 0.5 {
                        let (a, sa, b, sb) = if k < k2 { (k, s, k2, t) } else { (k2, t, k, s) };
                        let mut before_2 = u.get(sa, a) + u.get(sb, b);
                        let mut after_2 = u.get(sb, a) + u.get(sa, b);
                        if a > 0 && at[a - 1] >= 0 {
                            let sl = at[a - 1] as usize;
                            let w = clamp_w(sl);
                            before_2 += w * p.get(sl, sa);
                            after_2 += w * p.get(sl, sb);
                        }
                        if b + 1 < k_len && at[b + 1] >= 0 {
                            let sr = at[b + 1] as usize;
                            let w = clamp_w(sr);
                            before_2 += w * p.get(sb, sr);
                            after_2 += w * p.get(sa, sr);
                        }
                        if a + 1 == b {
                            before_2 += p.get(sa, sb);
                            after_2 += p.get(sb, sa);
                        } else {
                            if a + 1 < k_len && at[a + 1] >= 0 {
                                let ma = at[a + 1] as usize;
                                let w = clamp_w(ma);
                                before_2 += w * p.get(sa, ma);
                                after_2 += w * p.get(sb, ma);
                            }
                            if b > 0 && at[b - 1] >= 0 {
                                let mb = at[b - 1] as usize;
                                let w = clamp_w(mb);
                                before_2 += w * p.get(mb, sb);
                                after_2 += w * p.get(mb, sa);
                            }
                        }
                        best_k2 = best_k2.max(after_2 - before_2);
                    }
                }
            }
            if best_k2 > f64::NEG_INFINITY * 0.5 {
                alt_w += best_k2.clamp(-40.0, 20.0).exp();
            }
        }
        let p_bp = 1.0 / (1.0 + alt_w);
        let p_prod = b1[s].min(c0[s] * p_bp).clamp(0.0, 1.0);
        let p_boltz1 = if c0[s] > 1e-9 {
            (b1[s] / c0[s]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p_upper = c0[s].min(p_bp).min(p_boltz1);
        let p_raw = (0.5 * (p_prod + p_upper)).clamp(0.0, 1.0);
        result.confidence[s] = if apply_calibration_map {
            apply_standard_calibration_map(p_raw)
        } else {
            p_raw
        };
    }
}

/// Run `assign` returning raw physical (`T = 1.0` nat) marginal posteriors BEFORE applying
/// `STANDARD_CALIBRATION_KNOTS`, used by Leave-One-Protein-Out (LOPO) isotonic cross-validation.
pub fn assign_uncalibrated(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    cfg: &AnnealConfig,
) -> AssignResult {
    let u = unary_matrix(systems, sequence, priors, score_cfg);
    let p = pair_matrix(systems, score_cfg);
    let locks = resolve_locked_assignments(systems.len(), sequence, &score_cfg.locked_assignments);
    assign_from_matrices_with_locks_opts(&u, &p, cfg, &locks, false)
}

pub fn assign(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    cfg: &AnnealConfig,
) -> AssignResult {
    let u = unary_matrix(systems, sequence, priors, score_cfg);
    let p = pair_matrix(systems, score_cfg);
    let locks = resolve_locked_assignments(systems.len(), sequence, &score_cfg.locked_assignments);
    assign_from_matrices_with_locks(&u, &p, cfg, &locks)
}

pub fn assign_from_matrices(u: &Matrix2D, p: &Matrix2D, cfg: &AnnealConfig) -> AssignResult {
    assign_from_matrices_with_locks(u, p, cfg, &[])
}

pub fn assign_from_matrices_with_locks(
    u: &Matrix2D,
    p: &Matrix2D,
    cfg: &AnnealConfig,
    locks: &[(usize, usize)],
) -> AssignResult {
    assign_from_matrices_with_locks_opts(u, p, cfg, locks, true)
}

fn assign_from_matrices_with_locks_opts(
    u: &Matrix2D,
    p: &Matrix2D,
    cfg: &AnnealConfig,
    locks: &[(usize, usize)],
    apply_calibration_map: bool,
) -> AssignResult {
    let mut master_rng = Rng::new(cfg.seed);
    let steps = cfg.steps_per_system * u.rows;
    let seeds: Vec<u64> = (0..cfg.runs).map(|_| master_rng.next_u64()).collect();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(cfg.runs.max(1));

    let (positions, scores) = if n_threads <= 1 || cfg.runs <= 1 {
        let mut positions = Vec::with_capacity(cfg.runs);
        let mut scores = Vec::with_capacity(cfg.runs);
        for sub_seed in seeds {
            let mut ann = Annealer::new(u, p, Rng::new(sub_seed), locks);
            let sc = ann.run(steps, cfg.t_start, cfg.t_end, cfg.max_block);
            scores.push(sc);
            positions.push(ann.pos);
        }
        (positions, scores)
    } else {
        let chunk_len = cfg.runs.div_ceil(n_threads);
        let mut chunk_results: Vec<Vec<(Vec<i32>, f64)>> = Vec::with_capacity(n_threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for seed_chunk in seeds.chunks(chunk_len) {
                handles.push(scope.spawn(move || {
                    let mut local = Vec::with_capacity(seed_chunk.len());
                    for &sub_seed in seed_chunk {
                        let mut ann = Annealer::new(u, p, Rng::new(sub_seed), locks);
                        let sc = ann.run(steps, cfg.t_start, cfg.t_end, cfg.max_block);
                        local.push((ann.pos, sc));
                    }
                    local
                }));
            }
            for h in handles {
                if let Ok(part) = h.join() {
                    chunk_results.push(part);
                }
            }
        });

        let mut positions = Vec::with_capacity(cfg.runs);
        let mut scores = Vec::with_capacity(cfg.runs);
        for part in chunk_results {
            for (pos, sc) in part {
                positions.push(pos);
                scores.push(sc);
            }
        }
        (positions, scores)
    };
    let mut result = consensus_from_runs(&positions, scores);

    calibrate_standard_confidence_impl(&mut result, Some(&positions), u, p, apply_calibration_map);

    for &(s_lock, k_lock) in locks {
        if s_lock < result.position.len() && k_lock < u.cols {
            result.position[s_lock] = k_lock as i32;
            result.confidence[s_lock] = 1.0;
        }
    }
    result
}
