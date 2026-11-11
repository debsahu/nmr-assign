# nmr-assign

Open source, automated protein NMR backbone resonance assignment.
Peak lists in, backbone chemical shifts (H, N, CA, CB, C) out, with a
confidence for every residue. Built clean-room from published methods and
open data; no commercial or academic-licence tools are used or bundled.

Status: v0.1 research prototype. Python reference implementation; Julia
performance core planned (see `julia/README.md`).

## Quick start

Requires Python 3.10+ and NumPy.

```bash
cd python

# 1. Blind benchmark against a known assignment (NMR-STAR v3 or v2.1)
python -m nmrassign.cli benchmark --star my_assignment.str -v

# 2. Assign your own peak lists
python -m nmrassign.cli assign --star my_assignment.str --peaks peaks.tsv --out result
#    or give the sequence directly: --seq MQIFVKTLTG...

# 3. Rebuild shift priors from BMRB entries (recommended before real use)
python -m nmrassign.cli priors --stars bmrb_entries/ --out priors.json
python -m nmrassign.cli benchmark --star my_assignment.str --priors priors.json
```

Specifications live in `docs/specs/`; open work is tracked in `TODO.md`.

## How it works

1. Each HSQC peak roots a spin system; triple-resonance peaks attach to the
   nearest root.
2. Carbons are labelled residue i or i-1 using the i-1-only experiments.
   Ambiguities (spurious peaks, overlapping roots) are kept as competing
   candidates rather than resolved by fiat.
3. Every spin system / residue pairing gets a log-likelihood-ratio typing
   score; every spin system pair gets a sequential connectivity score.
4. Simulated annealing (single, swap and block moves) maximizes the total,
   repeated over independent runs; agreement across runs is the confidence.

## Benchmark (synthetic 76-residue protein, v0.1)

| Experiments | Peak quality | Accuracy | Recall | High-confidence precision |
|---|---|---|---|---|
| Full set (7 experiments) | 6% missing, 4% spurious | 97.2% | 97.2% | 98.6% |
| HSQC, HNCACB, CBCA(CO)NH | 6% missing, 4% spurious | 98.5% | 94.3% | 100% |
| Full set | 15% missing, 10% spurious | 77.8% | 70.0% | 100% |
| Full set, 4x annealing | 15% missing, 10% spurious | | 90.0% | |
| HSQC, HNCA, HN(CO)CA only | 6% missing, 4% spurious | 20% | 20% | 100% (2/2) |

Reading the table:
- High-confidence calls are reliable across every scenario; the tool
  reports ambiguity rather than guessing.
- The degraded case is limited by search budget: recall rises with more
  annealing, which is the motivation for the Julia core.
- CA-only data is information-limited: the wrong assignment scores as well
  as the truth, so no optimizer can fix it. Add CB or CO connectivity.
- The synthetic shifts are sampled from the priors themselves, so these
  numbers are optimistic for typing. Real-data validation is the next step.

## Repository layout

```
AGENTS.md            rules for AI agents (clean-room, licences, definition of done)
CLAUDE.md, GEMINI.md pointers to AGENTS.md for tools that look for them
TODO.md              roadmap and open tasks, by phase and module
docs/specs/          one specification per module (start at 00-architecture.md)
python/nmrassign/    reference implementation
python/tests/        regression tests (python -m pytest tests)
data/fixtures/       synthetic test data only
data/private/        your unpublished data (git-ignored)
julia/               performance core (planned)
```

## Licence

MIT. See LICENSE and THIRD_PARTY.md.
