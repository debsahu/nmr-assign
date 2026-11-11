"""Write a synthetic NMR-STAR reference for testing.

Shifts are sampled from the provisional priors, so this fixture contains no
real or third-party data. Sequence: human ubiquitin (76 aa), used only as a
realistic composition including prolines and glycines.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nmrassign.nmrstar import Reference, write_star  # noqa: E402
from nmrassign.priors import THREE, Priors  # noqa: E402

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def make(path, sequence=UBIQUITIN, seed=1):
    rng = np.random.default_rng(seed)
    pri = Priors()
    seq = [THREE[c] for c in sequence]
    ref = Reference(seq, list(range(1, len(seq) + 1)))
    for k, res in enumerate(seq):
        for atom, (mu, sd) in pri.table[res].items():
            if atom == "H" and (k == 0 or res == "PRO"):
                continue
            ref.shifts[(k, atom)] = float(rng.normal(mu, sd))
    write_star(path, ref)
    return ref


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_ubiquitin.str"
    make(out)
    print("wrote", out)
