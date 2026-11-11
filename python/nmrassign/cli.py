"""Command line entry point.

  python -m nmrassign.cli benchmark --star ref.str          # blind test vs a known assignment
  python -m nmrassign.cli simulate  --star ref.str --out peaks.tsv
  python -m nmrassign.cli assign    --seq MQIFV... --peaks peaks.tsv --out result
  python -m nmrassign.cli priors    --stars bmrb_dir/ --out priors.json
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

from .evaluate import compare_shifts, score, to_reference
from .nmrstar import Reference, read_star, write_star
from .peaks import read_peaks, write_peaks
from .priors import THREE, Priors, compute_priors
from .simulate import DEFAULT_EXPS, SimConfig, simulate
from .solver import AnnealConfig, assign
from .spinsystems import build_spin_systems


def _priors(args) -> Priors:
    return Priors.from_json(args.priors) if getattr(args, "priors", None) else Priors()


def _sequence(args) -> tuple[list[str], list[int]]:
    if args.star:
        ref = read_star(args.star)
        return ref.sequence, ref.seq_ids
    s = args.seq
    if Path(s).exists():
        s = "".join(l.strip() for l in Path(s).read_text().splitlines() if not l.startswith(">"))
    start = args.first_seq_id
    return [THREE.get(c.upper(), "UNK") for c in s], list(range(start, start + len(s)))


def _anneal(args) -> AnnealConfig:
    return AnnealConfig(runs=args.runs, steps_per_system=args.steps, seed=args.seed)


def cmd_benchmark(args):
    ref = read_star(args.star)
    exps = tuple(args.experiments.split(",")) if args.experiments else DEFAULT_EXPS
    peaks, truth = simulate(ref, SimConfig(experiments=exps, seed=args.seed,
                                           p_missing_peak=args.missing, spurious_rate=args.spurious))
    t0 = time.time()
    systems = build_spin_systems(peaks)
    res = assign(systems, ref.sequence, _priors(args), cfg=_anneal(args))
    m = score(res, truth)
    print(f"protein: {len(ref.sequence)} residues, {len(peaks)} peaks, {len(systems)} spin systems")
    print(f"experiments: {', '.join(exps)}")
    print(m.report())
    pred, _ = to_reference(res, systems, ref.sequence, ref.seq_ids)
    print("shift agreement with reference (within tolerance / predicted):")
    for atom, (ok, n) in compare_shifts(pred, ref).items():
        if n:
            print(f"  {atom:2s} {ok}/{n}")
    print(f"solver time: {time.time() - t0:.1f} s")
    wrong = [(s, k, truth[s]) for s, k in enumerate(res.position) if k >= 0 and k != truth[s]]
    if wrong and args.verbose:
        print("misassigned spin systems (placed, true, confidence):")
        for s, k, t in wrong:
            tl = f"{ref.sequence[t]}{ref.seq_ids[t]}" if t >= 0 else "none"
            print(f"  {ref.sequence[k]}{ref.seq_ids[k]:<5d} {tl:<8s} {res.confidence[s]:.2f}")
    return m


def cmd_simulate(args):
    ref = read_star(args.star)
    exps = tuple(args.experiments.split(",")) if args.experiments else DEFAULT_EXPS
    peaks, truth = simulate(ref, SimConfig(experiments=exps, seed=args.seed))
    write_peaks(args.out, peaks)
    Path(args.out).with_suffix(".truth.tsv").write_text(
        "root\tseq_id\n" + "".join(f"{i}\t{ref.seq_ids[t] if t >= 0 else '.'}\n"
                                   for i, t in enumerate(truth)))
    print(f"wrote {len(peaks)} peaks to {args.out}")


def cmd_assign(args):
    sequence, seq_ids = _sequence(args)
    systems = build_spin_systems(read_peaks(args.peaks))
    res = assign(systems, sequence, _priors(args), cfg=_anneal(args))
    pred, conf = to_reference(res, systems, sequence, seq_ids)
    out = Path(args.out)
    write_star(out.with_suffix(".str"), pred, conf)
    rows = ["seq_id\tresidue\tatom\tshift\tconfidence"]
    for (k, atom), v in sorted(pred.shifts.items()):
        rows.append(f"{seq_ids[k]}\t{sequence[k]}\t{atom}\t{v:.3f}\t{conf.get(k, 0):.2f}")
    out.with_suffix(".tsv").write_text("\n".join(rows) + "\n")
    n_hc = sum(1 for c in conf.values() if c >= 0.8)
    print(f"{len(systems)} spin systems; {len(conf)} residues assigned, {n_hc} with confidence >= 0.8")
    print(f"wrote {out.with_suffix('.str')} and {out.with_suffix('.tsv')}")


def cmd_priors(args):
    paths = sorted(glob.glob(str(Path(args.stars) / "*.str"))) if Path(args.stars).is_dir() else [args.stars]
    pri = compute_priors(paths)
    pri.to_json(args.out)
    print(f"priors from {len(paths)} files written to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nmrassign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--priors", help="priors JSON (default: provisional built-in)")
        p.add_argument("--runs", type=int, default=12)
        p.add_argument("--steps", type=int, default=1500, help="annealing steps per spin system")
        p.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("benchmark", help="simulate peaks from a reference and assign blind")
    b.add_argument("--star", required=True)
    b.add_argument("--experiments")
    b.add_argument("--missing", type=float, default=0.06)
    b.add_argument("--spurious", type=float, default=0.04)
    b.add_argument("-v", "--verbose", action="store_true")
    common(b)
    b.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("simulate")
    s.add_argument("--star", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--experiments")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_simulate)

    a = sub.add_parser("assign")
    a.add_argument("--peaks", required=True)
    a.add_argument("--seq", help="one-letter sequence or FASTA path")
    a.add_argument("--star", help="take the sequence from this NMR-STAR file")
    a.add_argument("--first-seq-id", type=int, default=1)
    a.add_argument("--out", required=True, help="output prefix")
    common(a)
    a.set_defaults(fn=cmd_assign)

    p = sub.add_parser("priors")
    p.add_argument("--stars", required=True, help="NMR-STAR file or directory of .str files")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_priors)

    args = ap.parse_args(argv)
    if args.cmd == "assign" and not (args.seq or args.star):
        ap.error("assign needs --seq or --star")
    return args.fn(args)


if __name__ == "__main__":
    main()
