"""Turn a solver result into per-residue shifts and score it against truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nmrstar import Reference
from .priors import Priors
from .solver import Result, candidate_llr
from .spinsystems import SpinSystem


def to_reference(result: Result, systems: list[SpinSystem], sequence: list[str],
                 seq_ids: list[int], min_conf: float = 0.0,
                 priors: Priors | None = None) -> tuple[Reference, dict[int, float]]:
    """Per-residue shifts. i-1 carbons fill the preceding residue where it has no
    spin system of its own (for example the residue before a proline)."""
    priors = priors or Priors()
    out = Reference(list(sequence), list(seq_ids))
    conf: dict[int, float] = {}
    at = {int(k): s for s, k in enumerate(result.position) if k >= 0}

    def pick(cands, res, atom, partner):
        # prefer the candidate confirmed by the neighbouring spin system
        if partner:
            return min(cands, key=lambda c: min(abs(c[0] - p[0]) for p in partner))[0]
        return cands[candidate_llr(priors, res, atom, cands, 0.7)[1]][0]

    for s, k in enumerate(result.position):
        if k < 0 or result.confidence[s] < min_conf:
            continue
        ss = systems[s]
        conf[k] = float(result.confidence[s])
        out.shifts[(k, "H")] = ss.H
        out.shifts[(k, "N")] = ss.N
        nxt = systems[at[k + 1]] if (k + 1) in at else None
        for (atom, off), cands in ss.cands.items():
            if off == 0:
                partner = nxt.cands.get((atom, -1)) if nxt else None
                out.shifts[(k, atom)] = pick(cands, sequence[k], atom, partner)
    for s, k in enumerate(result.position):
        if k < 1 or result.confidence[s] < min_conf:
            continue
        for (atom, off), cands in systems[s].cands.items():
            if off == -1 and (k - 1, atom) not in out.shifts:
                out.shifts[(k - 1, atom)] = pick(cands, sequence[k - 1], atom, None)
                conf.setdefault(k - 1, float(result.confidence[s]))
    return out, conf


@dataclass
class Metrics:
    assignable: int
    assigned: int
    correct: int
    hc_assigned: int
    hc_correct: int

    @property
    def accuracy(self): return self.correct / self.assigned if self.assigned else 0.0
    @property
    def recall(self): return self.correct / self.assignable if self.assignable else 0.0
    @property
    def hc_precision(self): return self.hc_correct / self.hc_assigned if self.hc_assigned else 0.0
    @property
    def hc_recall(self): return self.hc_correct / self.assignable if self.assignable else 0.0

    def report(self) -> str:
        return (f"assignable spin systems : {self.assignable}\n"
                f"assigned                : {self.assigned}\n"
                f"accuracy (correct/assigned)   : {self.accuracy:.1%}\n"
                f"recall   (correct/assignable) : {self.recall:.1%}\n"
                f"high-confidence (>=0.8) precision : {self.hc_precision:.1%}"
                f" ({self.hc_correct}/{self.hc_assigned})\n"
                f"high-confidence recall            : {self.hc_recall:.1%}")


def score(result: Result, truth: list[int], hc: float = 0.8) -> Metrics:
    pos = result.position
    t = np.asarray(truth)
    assigned = pos >= 0
    correct = assigned & (pos == t)
    high = assigned & (result.confidence >= hc)
    return Metrics(int((t >= 0).sum()), int(assigned.sum()), int(correct.sum()),
                   int(high.sum()), int((high & correct).sum()))


def compare_shifts(pred: Reference, ref: Reference, tol: dict | None = None) -> dict[str, tuple[int, int]]:
    """Per atom type: (matching within tol, total predicted). For real data,
    where no spin-system truth exists."""
    tol = tol or {"H": 0.05, "N": 0.5, "CA": 0.3, "CB": 0.4, "C": 0.3}
    out: dict[str, list[int]] = {a: [0, 0] for a in tol}
    for (k, atom), v in pred.shifts.items():
        r = ref.shifts.get((k, atom))
        if r is None:
            continue
        out[atom][1] += 1
        out[atom][0] += abs(r - v) <= tol[atom]
    return {a: tuple(v) for a, v in out.items()}
