"""Score spin system placements and optimize the assignment.

Original implementation of docs/specs/03-assignment.md section 3: log-likelihood-ratio
typing scores, Gaussian connectivity scores, simulated annealing with
single, swap and block moves, and multi-run consensus for confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .priors import Priors
from .spinsystems import SpinSystem

FORBID = -1e9


@dataclass
class ScoreConfig:
    sigma: dict = field(default_factory=lambda: {"CA": 0.12, "CB": 0.20, "C": 0.12})
    range_ppm: float = 30.0
    pair_floor: float = -8.0
    support_bonus: float = 0.7
    use_HN: bool = True


@dataclass
class AnnealConfig:
    runs: int = 12
    steps_per_system: int = 1500
    t_start: float = 8.0
    t_end: float = 0.05
    max_block: int = 8
    seed: int = 0


def candidate_llr(priors: Priors, res: str, atom: str, cands, support_bonus: float) -> tuple[float, int]:
    """Best candidate's LLR plus support bonus, minus a log(n) penalty for
    choosing among n hypotheses. Returns (score, index of chosen candidate)."""
    best, bi = -math.inf, 0
    for i, (v, sup) in enumerate(cands):
        sc = priors.llr(res, atom, v) + support_bonus * (sup - 1)
        if sc > best:
            best, bi = sc, i
    return best - math.log(len(cands)), bi


def unary_matrix(systems: list[SpinSystem], sequence: list[str], priors: Priors,
                 cfg: ScoreConfig = ScoreConfig()) -> np.ndarray:
    S, K = len(systems), len(sequence)
    U = np.full((S, K), FORBID)
    for k in range(1, K):
        res, prev = sequence[k], sequence[k - 1]
        if res == "PRO":
            continue
        for s, ss in enumerate(systems):
            sc = 0.0
            for (atom, off), cands in ss.cands.items():
                cands = [c for c in cands if c[1] > 0]      # support 0: degenerate hypothesis
                if not cands:
                    continue
                sc += candidate_llr(priors, res if off == 0 else prev, atom, cands, cfg.support_bonus)[0]
            if cfg.use_HN:
                sc += priors.llr(res, "N", ss.N) + priors.llr(res, "H", ss.H)
            U[s, k] = sc
    return U


def _cand_array(systems, key, n):
    out = np.full((len(systems), n), np.nan)
    for s, ss in enumerate(systems):
        for i, (v, _) in enumerate(ss.cands.get(key, [])[:n]):
            out[s, i] = v
    return out


def pair_matrix(systems: list[SpinSystem], cfg: ScoreConfig = ScoreConfig()) -> np.ndarray:
    """P[s, t]: evidence that t follows s, from s(i) vs t(i-1) carbons.
    Takes the best-matching candidate pair, minus log(n_s * n_t)."""
    S = len(systems)
    n = max([len(c) for ss in systems for c in ss.cands.values()] + [1])
    P = np.zeros((S, S))
    for atom, sig in cfg.sigma.items():
        a = _cand_array(systems, (atom, 0), n)
        b = _cand_array(systems, (atom, -1), n)
        d = a[:, None, :, None] - b[None, :, None, :]
        term = -0.5 * (d / sig) ** 2 - math.log(sig * math.sqrt(2 * math.pi)) + math.log(cfg.range_ppm)
        term = np.where(np.isnan(term), -np.inf, np.maximum(term, cfg.pair_floor))
        best = term.max(axis=(2, 3))
        na = (~np.isnan(a)).sum(1)
        nb = (~np.isnan(b)).sum(1)
        pen = np.log(np.maximum(na, 1)[:, None] * np.maximum(nb, 1)[None, :])
        P += np.where(np.isfinite(best), best - pen, 0.0)
    np.fill_diagonal(P, FORBID)
    return P


class Annealer:
    def __init__(self, U: np.ndarray, P: np.ndarray, rng: np.random.Generator):
        self.U, self.P, self.rng = U, P, rng
        self.S, self.K = U.shape
        self.slots = [np.flatnonzero(U[s] > FORBID / 2) for s in range(self.S)]
        self.at = -np.ones(self.K, dtype=int)     # residue -> spin system
        self.pos = -np.ones(self.S, dtype=int)    # spin system -> residue

    def local(self, A) -> float:
        U, P, at = self.U, self.P, self.at
        sc = 0.0
        edges = set()
        for k in A:
            if k < 0:
                continue
            s = at[k]
            if s >= 0:
                sc += U[s, k]
            edges.add(k - 1)
            edges.add(k)
        for e in edges:
            if 0 <= e < self.K - 1:
                a, b = at[e], at[e + 1]
                if a >= 0 and b >= 0:
                    sc += P[a, b]
        return sc

    def total(self) -> float:
        return self.local(range(self.K))

    def _apply(self, changes):
        """changes: list of (k, new_s). Returns undo list."""
        undo = [(k, self.at[k]) for k, _ in changes]
        for k, _ in changes:
            if self.at[k] >= 0:
                self.pos[self.at[k]] = -1
        for k, s in changes:
            self.at[k] = s
        for k, s in changes:
            if s >= 0:
                self.pos[s] = k
        return undo

    def _propose(self):
        r = self.rng.random()
        if r < 0.65:                                   # place or swap one system
            s = int(self.rng.integers(self.S))
            if len(self.slots[s]) == 0:
                return None
            k = int(self.rng.choice(self.slots[s]))
            ks, t = self.pos[s], self.at[k]
            if ks == k:
                return [(k, -1)]                        # unassign
            ch = [(k, s)]
            if ks >= 0:
                ch.append((ks, t if (t >= 0 and self.U[t, ks] > FORBID / 2) else -1))
            return ch
        if r < 0.95:                                   # swap two blocks
            L = int(self.rng.integers(1, self.max_block + 1))
            k1 = int(self.rng.integers(0, self.K - L + 1))
            k2 = int(self.rng.integers(0, self.K - L + 1))
            if abs(k1 - k2) < L:
                return None
            ch = []
            for i in range(L):
                a, b = self.at[k1 + i], self.at[k2 + i]
                if (b >= 0 and self.U[b, k1 + i] < FORBID / 2) or (a >= 0 and self.U[a, k2 + i] < FORBID / 2):
                    return None
                ch += [(k1 + i, b), (k2 + i, a)]
            return ch
        s = int(self.rng.integers(self.S))             # unassign
        return [(self.pos[s], -1)] if self.pos[s] >= 0 else None

    def run(self, steps: int, t0: float, t1: float, max_block: int) -> float:
        self.max_block = max_block
        cur = self.total()
        best, best_at = cur, self.at.copy()
        for i in range(steps):
            T = t0 * (t1 / t0) ** (i / max(steps - 1, 1))
            ch = self._propose()
            if not ch:
                continue
            A = [k for k, _ in ch]
            before = self.local(A)
            undo = self._apply(ch)
            delta = self.local(A) - before
            if delta >= 0 or self.rng.random() < math.exp(delta / T):
                cur += delta
                if cur > best:
                    best, best_at = cur, self.at.copy()
            else:
                self._apply(undo)
        self.at = best_at
        self.pos[:] = -1
        for k, s in enumerate(self.at):
            if s >= 0:
                self.pos[s] = k
        return best


@dataclass
class Result:
    position: np.ndarray        # spin system -> residue index (-1 unassigned)
    confidence: np.ndarray      # spin system -> fraction of runs agreeing
    score: float
    run_scores: list


def assign(systems: list[SpinSystem], sequence: list[str], priors: Priors,
           score_cfg: ScoreConfig = ScoreConfig(), cfg: AnnealConfig = AnnealConfig()) -> Result:
    U = unary_matrix(systems, sequence, priors, score_cfg)
    P = pair_matrix(systems, score_cfg)
    rng = np.random.default_rng(cfg.seed)
    steps = cfg.steps_per_system * len(systems)
    positions, scores = [], []
    for _ in range(cfg.runs):
        ann = Annealer(U, P, np.random.default_rng(rng.integers(1 << 32)))
        scores.append(ann.run(steps, cfg.t_start, cfg.t_end, cfg.max_block))
        positions.append(ann.pos.copy())
    best = int(np.argmax(scores))
    ref = positions[best]
    agree = np.mean([p == ref for p in positions], axis=0)
    conf = np.where(ref >= 0, agree, 0.0)
    return Result(ref, conf, scores[best], scores)
