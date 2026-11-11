"""Regression tests. Run: python -m pytest tests  (or python tests/test_pipeline.py)"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fixture import make  # noqa: E402
from nmrassign.evaluate import score  # noqa: E402
from nmrassign.nmrstar import read_star  # noqa: E402
from nmrassign.peaks import read_peaks, write_peaks  # noqa: E402
from nmrassign.priors import Priors  # noqa: E402
from nmrassign.simulate import SimConfig, simulate  # noqa: E402
from nmrassign.solver import AnnealConfig, assign  # noqa: E402
from nmrassign.spinsystems import build_spin_systems  # noqa: E402

FAST = AnnealConfig(runs=4, steps_per_system=800)


def _ref(tmp):
    p = Path(tmp) / "ref.str"
    make(p)
    return read_star(p)


def test_star_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ref = _ref(tmp)
        assert len(ref.sequence) == 76 and ref.sequence[0] == "MET"
        assert (9, "CA") in ref.shifts                        # Gly10 CA
        assert (9, "CB") not in ref.shifts                    # Gly has no CB


def test_peak_io_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ref = _ref(tmp)
        peaks, _ = simulate(ref, SimConfig(seed=1))
        f = Path(tmp) / "p.tsv"
        write_peaks(f, peaks)
        back = read_peaks(f)
        assert len(back) == len(peaks) and back[0].exp == peaks[0].exp


def test_clean_data_spin_systems_exact():
    with tempfile.TemporaryDirectory() as tmp:
        ref = _ref(tmp)
        cfg = SimConfig(p_missing_peak=0, p_missing_root=0, spurious_rate=0,
                        sidechain_nh2=False, seed=2)
        peaks, truth = simulate(ref, cfg)
        top_wrong = total = lost = 0
        for ss, k in zip(build_spin_systems(peaks), truth):
            for atom in ("CA", "CB", "C"):
                for off in (0, -1):
                    true = ref.shifts.get((k + off, atom))
                    if true is None:
                        continue
                    cands = ss.cands.get((atom, off), [])
                    total += 1
                    if not any(abs(v - true) < 0.2 for v, _ in cands):
                        lost += 1
                    elif abs(cands[0][0] - true) >= 0.2:
                        top_wrong += 1
        # Regression guard, not an ideal. Measured baseline (seed 2): 22/421 = 5.2%
        # lost, from overlapping HSQC roots and coincident i / i-1 shifts (the
        # degenerate hypothesis is off by default; see docs/specs/03-assignment.md).
        assert lost / total < 0.06, (lost, total)
        assert top_wrong / total < 0.03, (top_wrong, total)


def test_full_set_benchmark_floor():
    with tempfile.TemporaryDirectory() as tmp:
        ref = _ref(tmp)
        peaks, truth = simulate(ref, SimConfig(seed=0))
        res = assign(build_spin_systems(peaks), ref.sequence, Priors(), cfg=FAST)
        m = score(res, truth)
        assert m.recall >= 0.85, m.report()
        assert m.hc_precision >= 0.95, m.report()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
