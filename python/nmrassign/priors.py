"""Chemical shift priors per residue type and backbone atom.

Provenance: PROVISIONAL values approximating published BMRB-wide
statistics (mean, sd in ppm). Replace with `cli priors`, which recomputes
them from NMR-STAR files, before any release.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ATOMS = ("H", "N", "CA", "CB", "C")

THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
ONE = {v: k for k, v in THREE.items()}

# (mean, sd) in ppm. PROVISIONAL.
_P = {
    "ALA": {"CA": (53.1, 2.0), "CB": (19.0, 1.8), "C": (177.8, 2.1), "N": (123.2, 3.5), "H": (8.19, 0.6)},
    "ARG": {"CA": (56.8, 2.3), "CB": (30.7, 1.8), "C": (176.4, 2.0), "N": (120.8, 3.6), "H": (8.24, 0.6)},
    "ASN": {"CA": (53.5, 1.9), "CB": (38.7, 1.7), "C": (175.3, 1.8), "N": (118.9, 4.0), "H": (8.33, 0.6)},
    "ASP": {"CA": (54.7, 2.0), "CB": (40.9, 1.6), "C": (176.4, 1.8), "N": (120.6, 3.9), "H": (8.30, 0.6)},
    "CYS": {"CA": (58.0, 3.4), "CB": (32.0, 5.5), "C": (174.9, 2.1), "N": (120.0, 4.4), "H": (8.39, 0.7)},
    "GLN": {"CA": (56.6, 2.1), "CB": (29.2, 1.8), "C": (176.3, 2.0), "N": (119.9, 3.5), "H": (8.21, 0.6)},
    "GLU": {"CA": (57.3, 2.1), "CB": (30.0, 1.7), "C": (176.9, 2.0), "N": (120.7, 3.5), "H": (8.33, 0.6)},
    "GLY": {"CA": (45.4, 1.3), "C": (173.9, 1.9), "N": (109.6, 3.8), "H": (8.33, 0.6)},
    "HIS": {"CA": (56.5, 2.3), "CB": (30.2, 2.1), "C": (175.2, 2.0), "N": (119.7, 4.0), "H": (8.25, 0.7)},
    "ILE": {"CA": (61.6, 2.7), "CB": (38.6, 2.0), "C": (175.8, 1.9), "N": (121.4, 4.2), "H": (8.28, 0.7)},
    "LEU": {"CA": (55.6, 2.1), "CB": (42.3, 1.9), "C": (177.0, 2.0), "N": (121.8, 3.9), "H": (8.22, 0.6)},
    "LYS": {"CA": (56.9, 2.2), "CB": (32.8, 1.8), "C": (176.6, 2.0), "N": (121.0, 3.7), "H": (8.18, 0.6)},
    "MET": {"CA": (56.1, 2.2), "CB": (32.9, 2.2), "C": (176.2, 2.1), "N": (120.1, 3.5), "H": (8.26, 0.6)},
    "PHE": {"CA": (58.1, 2.6), "CB": (40.0, 2.0), "C": (175.5, 2.0), "N": (120.4, 4.1), "H": (8.35, 0.7)},
    "PRO": {"CA": (63.3, 1.5), "CB": (31.9, 1.2), "C": (176.7, 1.6), "N": (134.0, 6.0)},
    "SER": {"CA": (58.7, 2.1), "CB": (63.8, 1.5), "C": (174.6, 1.8), "N": (116.3, 3.5), "H": (8.28, 0.6)},
    "THR": {"CA": (62.2, 2.6), "CB": (69.7, 1.7), "C": (174.5, 1.8), "N": (115.4, 4.7), "H": (8.24, 0.6)},
    "TRP": {"CA": (57.7, 2.6), "CB": (30.1, 2.0), "C": (176.1, 2.0), "N": (121.6, 4.1), "H": (8.29, 0.8)},
    "TYR": {"CA": (58.1, 2.6), "CB": (39.3, 2.2), "C": (175.4, 2.0), "N": (120.5, 4.1), "H": (8.30, 0.7)},
    "VAL": {"CA": (62.5, 2.9), "CB": (32.7, 1.8), "C": (175.7, 1.9), "N": (121.1, 4.5), "H": (8.29, 0.7)},
}


def _logpdf(v: float, mu: float, sd: float) -> float:
    z = (v - mu) / sd
    return -0.5 * z * z - math.log(sd * math.sqrt(2 * math.pi))


class Priors:
    def __init__(self, table: dict | None = None):
        self.table = table or _P

    @classmethod
    def from_json(cls, path: str | Path) -> "Priors":
        raw = json.loads(Path(path).read_text())
        return cls({r: {a: tuple(v) for a, v in d.items()} for r, d in raw.items()})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=1))

    def has(self, res: str, atom: str) -> bool:
        return atom in self.table.get(res, {})

    def logpdf(self, res: str, atom: str, v: float) -> float:
        mu, sd = self.table[res][atom]
        return _logpdf(v, mu, sd)

    def background(self, atom: str, v: float) -> float:
        """Log of the residue-averaged density (Pro excluded for H)."""
        vals = [self.logpdf(r, atom, v) for r in self.table if atom in self.table[r]]
        m = max(vals)
        return m + math.log(sum(math.exp(x - m) for x in vals) / len(vals))

    def llr(self, res: str, atom: str, v: float, floor: float = -12.0,
            gly_cb: float = math.log(0.02)) -> float:
        if res not in self.table:          # unknown residue: uninformative
            return 0.0
        if not self.has(res, atom):        # e.g. a CB observed at glycine
            return gly_cb
        return max(self.logpdf(res, atom, v) - self.background(atom, v), floor)


def compute_priors(star_paths: list[str | Path], min_count: int = 20) -> Priors:
    """Recompute (mean, sd) per residue/atom from NMR-STAR files.

    Uses a robust 3-sigma trim to drop misreferenced or mistyped values.
    """
    from .nmrstar import read_star
    import numpy as np

    pool: dict[tuple[str, str], list[float]] = {}
    for p in star_paths:
        try:
            ref = read_star(p)
        except Exception:
            continue
        for (idx, atom), v in ref.shifts.items():
            pool.setdefault((ref.sequence[idx], atom), []).append(v)
    table: dict = {}
    for (res, atom), vals in pool.items():
        a = np.asarray(vals)
        if len(a) < min_count:
            continue
        med = np.median(a)
        mad = 1.4826 * np.median(np.abs(a - med)) or a.std()
        a = a[np.abs(a - med) <= 3 * mad]
        table.setdefault(res, {})[atom] = (round(float(a.mean()), 3), round(float(a.std()), 3))
    base = Priors().table
    for res, d in base.items():            # keep provisional values for gaps
        for atom, v in d.items():
            table.setdefault(res, {}).setdefault(atom, v)
    return Priors(table)
