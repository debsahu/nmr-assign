//! Score spin system placements and optimize the backbone assignment.
//!
//! Provenance: Original implementation of `docs/specs/03-assignment.md` section 3:
//! log-likelihood-ratio typing scores, Gaussian sequential connectivity scores,
//! simulated annealing with single, swap, and block moves, and multi-run consensus.

use std::f64::consts::TAU;

use crate::priors::Priors;
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
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &(v, sup)) in cands.iter().enumerate() {
        let sc = priors.llr(res, atom, v) + support_bonus * ((sup as f64) - 1.0);
        if sc > best {
            best = sc;
            best_i = i;
        }
    }
    (best - (cands.len() as f64).ln(), best_i)
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
    for k in 1..k_len {
        let res = &sequence[k];
        let prev = &sequence[k - 1];
        if res == "PRO" {
            continue;
        }
        for (s, ss) in systems.iter().enumerate() {
            if cfg.forbid_zero_carbons && !ss.has_observed_carbons() {
                continue;
            }
            let mut sc = 0.0;
            for ((atom, off), raw_cands) in &ss.cands {
                let cands: Vec<(f64, usize)> =
                    raw_cands.iter().copied().filter(|c| c.1 > 0).collect();
                if cands.is_empty() {
                    continue;
                }
                let target_res = if *off == 0 { res } else { prev };
                sc += candidate_llr(priors, target_res, atom, &cands, cfg.support_bonus).0;
            }
            if cfg.use_hn {
                sc += priors.llr(res, "N", ss.n) + priors.llr(res, "H", ss.h);
            }
            sc -= cfg.assign_cost;
            u.set(s, k, sc);
        }
    }
    u
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
            let Some(a_raw) = systems[s].cands.get(&key_i) else { continue };
            let a_slice = &a_raw[..a_raw.len().min(max_n)];
            if a_slice.is_empty() {
                continue;
            }
            let na = a_slice.len();

            for t in 0..s_len {
                if s == t {
                    continue;
                }
                let Some(b_raw) = systems[t].cands.get(&key_im1) else { continue };
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

struct Annealer<'a> {
    u: &'a Matrix2D,
    p: &'a Matrix2D,
    rng: Rng,
    s_len: usize,
    k_len: usize,
    slots: Vec<Vec<usize>>,
    at: Vec<i32>,
    pos: Vec<i32>,
    edge_marks: Vec<u32>,
    mark_gen: u32,
}

impl<'a> Annealer<'a> {
    fn new(u: &'a Matrix2D, p: &'a Matrix2D, rng: Rng) -> Self {
        let s_len = u.rows;
        let k_len = u.cols;
        let slots = (0..s_len)
            .map(|s| {
                (0..k_len)
                    .filter(|&k| u.get(s, k) > FORBID * 0.5)
                    .collect::<Vec<usize>>()
            })
            .collect();
        Self {
            u,
            p,
            rng,
            s_len,
            k_len,
            slots,
            at: vec![-1; k_len],
            pos: vec![-1; s_len],
            edge_marks: vec![0; k_len],
            mark_gen: 1,
        }
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
        let mut edges = [0usize; 64];
        let mut n_edges = 0usize;

        for &k in changed_residues {
            let s = self.at[k];
            if s >= 0 {
                sc += self.u.get(s as usize, k);
            }
            if k > 0 && k - 1 < self.k_len - 1 && self.edge_marks[k - 1] != gen {
                self.edge_marks[k - 1] = gen;
                edges[n_edges] = k - 1;
                n_edges += 1;
            }
            if k < self.k_len - 1 && self.edge_marks[k] != gen {
                self.edge_marks[k] = gen;
                edges[n_edges] = k;
                n_edges += 1;
            }
        }
        for &e in &edges[..n_edges] {
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

        let mut changes: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 2);
        let mut affected: Vec<usize> = Vec::with_capacity(max_block * 2 + 2);
        let mut undo: Vec<(usize, i32)> = Vec::with_capacity(max_block * 2 + 2);
        let denom = (steps.saturating_sub(1)).max(1) as f64;
        let ratio = t1 / t0;

        for step in 0..steps {
            let temp = t0 * ratio.powf((step as f64) / denom);
            changes.clear();

            let r = self.rng.f64();
            if r < 0.65 {
                let s = self.rng.usize(self.s_len);
                if self.slots[s].is_empty() {
                    continue;
                }
                let k = self.slots[s][self.rng.usize(self.slots[s].len())];
                let ks = self.pos[s];
                let t = self.at[k];
                if ks == k as i32 {
                    changes.push((k, -1));
                } else {
                    changes.push((k, s as i32));
                    if ks >= 0 {
                        let ks_u = ks as usize;
                        let repl = if t >= 0 && self.u.get(t as usize, ks_u) > FORBID * 0.5 {
                            t
                        } else {
                            -1
                        };
                        changes.push((ks_u, repl));
                    }
                }
            } else if r < 0.95 {
                let max_l = max_block.min(self.k_len / 2);
                if max_l == 0 {
                    continue;
                }
                let l = self.rng.range_usize(1, max_l + 1);
                let k1 = self.rng.usize(self.k_len - l + 1);
                let k2 = self.rng.usize(self.k_len - l + 1);
                if k1.abs_diff(k2) < l {
                    continue;
                }
                let mut valid = true;
                for i in 0..l {
                    let a = self.at[k1 + i];
                    let b = self.at[k2 + i];
                    if (b >= 0 && self.u.get(b as usize, k1 + i) < FORBID * 0.5)
                        || (a >= 0 && self.u.get(a as usize, k2 + i) < FORBID * 0.5)
                    {
                        valid = false;
                        break;
                    }
                    changes.push((k1 + i, b));
                    changes.push((k2 + i, a));
                }
                if !valid {
                    continue;
                }
            } else {
                let s = self.rng.usize(self.s_len);
                if self.pos[s] < 0 {
                    continue;
                }
                changes.push((self.pos[s] as usize, -1));
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
            if delta >= 0.0 || self.rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best {
                    best = cur;
                    best_at.copy_from_slice(&self.at);
                }
            } else {
                let mut dummy = Vec::new();
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

pub fn assign(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    cfg: &AnnealConfig,
) -> AssignResult {
    let u = unary_matrix(systems, sequence, priors, score_cfg);
    let p = pair_matrix(systems, score_cfg);
    assign_from_matrices(&u, &p, cfg)
}

pub fn assign_from_matrices(u: &Matrix2D, p: &Matrix2D, cfg: &AnnealConfig) -> AssignResult {
    let mut master_rng = Rng::new(cfg.seed);
    let steps = cfg.steps_per_system * u.rows;
    let seeds: Vec<u64> = (0..cfg.runs).map(|_| master_rng.next_u64()).collect();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(cfg.runs.max(1));

    if n_threads <= 1 || cfg.runs <= 1 {
        let mut positions = Vec::with_capacity(cfg.runs);
        let mut scores = Vec::with_capacity(cfg.runs);
        for sub_seed in seeds {
            let mut ann = Annealer::new(u, p, Rng::new(sub_seed));
            let sc = ann.run(steps, cfg.t_start, cfg.t_end, cfg.max_block);
            scores.push(sc);
            positions.push(ann.pos);
        }
        return consensus_from_runs(&positions, scores);
    }

    let chunk_len = cfg.runs.div_ceil(n_threads);
    let mut chunk_results: Vec<Vec<(Vec<i32>, f64)>> = Vec::with_capacity(n_threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for seed_chunk in seeds.chunks(chunk_len) {
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(seed_chunk.len());
                for &sub_seed in seed_chunk {
                    let mut ann = Annealer::new(u, p, Rng::new(sub_seed));
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
    consensus_from_runs(&positions, scores)
}
