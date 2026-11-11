"""Peak data model, experiment definitions and TSV I/O (format in docs/specs/03-assignment.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# experiment -> list of (atom, offset, sign); offset None means "i and i-1"
EXPERIMENTS: dict[str, list[tuple[str, int, int]]] = {
    "HSQC": [],
    "HNCA": [("CA", 0, 1), ("CA", -1, 1)],
    "HNCOCA": [("CA", -1, 1)],
    "HNCACB": [("CA", 0, 1), ("CB", 0, -1), ("CA", -1, 1), ("CB", -1, -1)],
    "CBCACONH": [("CA", -1, 1), ("CB", -1, 1)],
    "HNCO": [("C", -1, 1)],
    "HNCACO": [("C", 0, 1), ("C", -1, 1)],
}
I_MINUS_1_ONLY = {"HNCOCA", "HNCO", "CBCACONH"}


@dataclass
class Peak:
    exp: str
    H: float
    N: float
    C: float | None = None
    sign: int = 1


def write_peaks(path: str | Path, peaks: list[Peak]) -> None:
    rows = ["experiment\tH\tN\tC\tsign"]
    for p in peaks:
        c = "" if p.C is None else f"{p.C:.3f}"
        rows.append(f"{p.exp}\t{p.H:.4f}\t{p.N:.3f}\t{c}\t{p.sign}")
    Path(path).write_text("\n".join(rows) + "\n")


def read_peaks(path: str | Path) -> list[Peak]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith(("experiment", "#")):
            continue
        f = line.split("\t")
        f += [""] * (5 - len(f))
        exp = f[0].strip().upper().replace("(", "").replace(")", "")
        if exp not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {f[0]!r}; expected one of {list(EXPERIMENTS)}")
        out.append(Peak(exp, float(f[1]), float(f[2]),
                        float(f[3]) if f[3].strip() else None,
                        int(f[4]) if f[4].strip() else 1))
    return out
