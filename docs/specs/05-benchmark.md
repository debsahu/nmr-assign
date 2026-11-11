# 05. Benchmark and simulation

## Simulator (`python/nmrassign/simulate.py`)
From a reference NMR-STAR assignment, generates peak lists with noise
(H 0.008, N 0.06, C 0.04 ppm), missing peaks (6%), missing roots (3%),
Asn/Gln NH2 roots, and spurious peaks (4%). Ground truth is returned
separately and never reaches the solver. Planned: render full spectra for
peak picking (02).

## Command
`python -m nmrassign.cli benchmark --star ref.str [-v] [--experiments ...]`

## Metrics
Accuracy, recall, high-confidence (>= 0.8) precision and recall, per-atom
shift agreement.

## Baseline (v0.1, synthetic 76-residue protein)
| Experiments | Peaks | Accuracy | Recall | HC precision |
|---|---|---|---|---|
| Full set | 6% missing, 4% spurious | 97.2% | 97.2% | 98.6% |
| HNCACB + CBCACONH | 6% / 4% | 98.5% | 94.3% | 100% |
| Full set | 15% / 10% | 77.8% | 70.0% | 100% |
| HNCA + HNCOCA | 6% / 4% | 20% | 20% | 100% (2/2) |

## Benchmark suite (TODO)
1. Maintainer's completed assignment (private, `data/private/`).
2. 10 to 20 public BMRB entries spanning size, fold, and IDPs.
3. Synthetic stress set (noise, overlap, deuteration).

Regression rule: no merged change may lower accuracy or HC precision on the
suite (CLAUDE.md).
