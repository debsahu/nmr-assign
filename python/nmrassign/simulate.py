"""Simulate triple-resonance peak lists from a reference assignment.

Original method. Produces realistic imperfections: per-peak noise, missing
peaks, missing roots, Asn/Gln sidechain NH2 roots, and spurious peaks.
The ground truth (root index -> residue index) is returned separately and
never passed to the solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nmrstar import Reference
from .peaks import EXPERIMENTS, Peak

DEFAULT_EXPS = ("HSQC", "HNCA", "HNCOCA", "HNCACB", "CBCACONH", "HNCO", "HNCACO")
_C_RANGE = {"CA": (42, 68), "CB": (15, 72), "C": (170, 181)}


@dataclass
class SimConfig:
    experiments: tuple[str, ...] = DEFAULT_EXPS
    noise_H: float = 0.008
    noise_N: float = 0.06
    noise_C: float = 0.04
    p_missing_peak: float = 0.06
    p_missing_root: float = 0.03
    spurious_rate: float = 0.04       # spurious peaks per real peak, per experiment
    sidechain_nh2: bool = True
    seed: int = 0


def simulate(ref: Reference, cfg: SimConfig = SimConfig()) -> tuple[list[Peak], list[int]]:
    rng = np.random.default_rng(cfg.seed)
    peaks: list[Peak] = []
    truth: list[int] = []                        # residue index per HSQC root, -1 if none
    roots: list[tuple[float, float]] = []

    def jitter(h, n):
        return h + rng.normal(0, cfg.noise_H), n + rng.normal(0, cfg.noise_N)

    for k, res in enumerate(ref.sequence):
        h, n = ref.shifts.get((k, "H")), ref.shifts.get((k, "N"))
        if k == 0 or res == "PRO" or h is None or n is None:
            continue
        if rng.random() < cfg.p_missing_root:
            continue
        roots.append((h, n))
        peaks.append(Peak("HSQC", *jitter(h, n)))
        truth.append(k)
        for exp in cfg.experiments:
            for atom, off, sign in EXPERIMENTS[exp]:
                c = ref.shifts.get((k + off, atom))
                if c is None or rng.random() < cfg.p_missing_peak:
                    continue
                peaks.append(Peak(exp, *jitter(h, n), c + rng.normal(0, cfg.noise_C), sign))

    if cfg.sidechain_nh2:
        for k, res in enumerate(ref.sequence):
            if res in ("ASN", "GLN"):
                n = rng.normal(112.5, 1.0)
                for h in (rng.normal(6.85, 0.1), rng.normal(7.55, 0.1)):
                    peaks.append(Peak("HSQC", h, n))
                    truth.append(-1)

    for exp in cfg.experiments:
        if exp == "HSQC" or not roots:
            continue
        n_real = sum(1 for p in peaks if p.exp == exp)
        for _ in range(rng.poisson(cfg.spurious_rate * n_real)):
            h, n = roots[rng.integers(len(roots))]
            atom, _, sign = EXPERIMENTS[exp][rng.integers(len(EXPERIMENTS[exp]))]
            lo, hi = _C_RANGE[atom]
            peaks.append(Peak(exp, *jitter(h, n), rng.uniform(lo, hi), sign))

    # shuffle HSQC order so root index carries no sequence information
    hsqc = [i for i, p in enumerate(peaks) if p.exp == "HSQC"]
    perm = rng.permutation(len(hsqc))
    hsqc_peaks = [peaks[hsqc[j]] for j in perm]
    truth = [truth[j] for j in perm]
    others = [p for p in peaks if p.exp != "HSQC"]
    rng.shuffle(others)
    return hsqc_peaks + others, truth
