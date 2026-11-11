"""Group peaks into backbone spin systems and resolve i / i-1 offsets.

Original method implementing docs/specs/03-assignment.md sections 2 and 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .peaks import I_MINUS_1_ONLY, Peak


@dataclass
class Tolerances:
    H: float = 0.03
    N: float = 0.30
    match: dict = field(default_factory=lambda: {"CA": 0.25, "CB": 0.35, "C": 0.25})


@dataclass
class SpinSystem:
    """Backbone spin system rooted on one HSQC peak.

    cands holds competing hypotheses per (atom, offset): a list of
    (ppm, support) sorted by support, where support is the number of peaks
    agreeing within tolerance. The solver chooses among them; ambiguity is
    never resolved here by fiat.
    """
    H: float
    N: float
    cands: dict = field(default_factory=dict)    # (atom, offset) -> [(ppm, support), ...]

    @property
    def shifts(self) -> dict:
        """Best-supported value per (atom, offset)."""
        return {k: v[0][0] for k, v in self.cands.items()}


def _cluster(values: list[float], tol: float) -> list[list[float]]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - np.mean(out[-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def build_spin_systems(peaks: list[Peak], tol: Tolerances = Tolerances(),
                       max_cands: int = 3, degenerate: bool = False) -> list[SpinSystem]:
    roots = [p for p in peaks if p.exp == "HSQC"]
    if not roots:
        raise ValueError("an HSQC peak list is required to define spin system roots")
    rH = np.array([p.H for p in roots])
    rN = np.array([p.N for p in roots])
    attached: list[list[Peak]] = [[] for _ in roots]
    for p in peaks:
        if p.exp == "HSQC" or p.C is None:
            continue
        d = np.hypot((rH - p.H) / tol.H, (rN - p.N) / tol.N)
        j = int(np.argmin(d))
        if d[j] <= 1.0:
            attached[j].append(p)

    systems = []
    for r, plist in zip(roots, attached):
        systems.append(_resolve(r, plist, tol, max_cands, degenerate))
    return systems


def _resolve(root: Peak, plist: list[Peak], tol: Tolerances, max_cands: int,
             degenerate: bool) -> SpinSystem:
    m = tol.match
    # entries: [atom or None, offset or None, value, from an i-and-i-1 experiment]
    hncacb = [["CA" if p.sign >= 0 else "CB", None, p.C, True] for p in plist if p.exp == "HNCACB"]
    entries: list[list] = []
    for p in plist:
        if p.exp == "HNCA":
            entries.append(["CA", None, p.C, True])
        elif p.exp == "HNCOCA":
            entries.append(["CA", -1, p.C, False])
        elif p.exp == "HNCO":
            entries.append(["C", -1, p.C, False])
        elif p.exp == "HNCACO":
            entries.append(["C", None, p.C, True])
    # CBCA(CO)NH peaks take atom type from the matching HNCACB peak,
    # and that HNCACB peak becomes i-1.
    for p in plist:
        if p.exp != "CBCACONH":
            continue
        best, bd = None, None
        for e in hncacb:
            d = abs(e[2] - p.C)
            if d <= m[e[0]] and (bd is None or d < bd):
                best, bd = e, d
        if best is not None:
            best[1] = -1
            entries.append([best[0], -1, p.C, False])
        else:
            entries.append([None, -1, p.C, False])          # untyped i-1 carbon
    entries += hncacb

    # type untyped CBCA(CO)NH peaks by matching other i-1 references
    for e in entries:
        if e[0] is None:
            for atom in ("CA", "CB"):
                refs = [x[2] for x in entries if x[0] == atom and x[1] == -1]
                if any(abs(r - e[2]) <= m[atom] for r in refs):
                    e[0] = atom
                    break

    # resolve offsets of "i and i-1" peaks against i-1 references
    for e in entries:
        if e[0] is not None and e[1] is None:
            refs = [x[2] for x in entries if x[0] == e[0] and x[1] == -1 and x is not e]
            if refs:
                e[1] = -1 if any(abs(r - e[2]) <= m[e[0]] for r in refs) else 0

    ss = SpinSystem(root.H, root.N)
    for atom in ("CA", "CB", "C"):
        for off in (0, -1):
            # offset still unknown (no i-1 reference observed): candidate for both
            vals = [e[2] for e in entries if e[0] == atom and e[1] in (off, None)]
            if vals:
                clusters = sorted(_cluster(vals, m[atom]), key=len, reverse=True)[:max_cands]
                ss.cands[(atom, off)] = [(float(np.mean(c)), len(c)) for c in clusters]
            elif off == 0 and degenerate:
                # Degenerate i / i-1 shifts: an i-and-i-1 experiment showed only the
                # peak matching i-1, so it may also be residue i. Stored with
                # support 0: used for connectivity, never for residue typing.
                vals = [e[2] for e in entries if e[0] == atom and e[1] == -1 and e[3]]
                if vals:
                    ss.cands[(atom, off)] = [(float(np.mean(vals)), 0)]
    return ss
