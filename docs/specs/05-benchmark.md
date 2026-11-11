# 05. Benchmark and simulation

## Simulator (`crates/nmr-core/src/simulate.rs`)
From a reference NMR-STAR assignment:
1. `simulate`: generates blind peak lists with jitter (`H` 0.008, `N` 0.06, `C` 0.04 ppm), missing peaks (6%), missing roots (3%), Asn/Gln NH2 sidechain roots, and spurious peaks (4%). Ground truth is returned separately and never reaches the solver.
2. `render_hsqc_2d` & `render_triple_3d`: renders full 2D `1H-15N` HSQC and 3D triple-resonance spectra with mixed Lorentzian-Gaussian lineshapes and thermal noise for GUI visualization and peak-picker benchmarking.

## Command
`cargo run --release -p nmr-cli -- benchmark --star ref.str [--engine gpu|cpu] [-v] [--experiments ...]`

## Metrics
Accuracy, recall, high-confidence (`>= 0.8`) precision and recall, per-atom shift agreement.

## Baseline (synthetic 76-residue ubiquitin)
| Scenario | Engine | Accuracy | Recall | HC precision |
|---|---|---|---|---|
| Reference v0.1 full set (6% missing, 4% spurious) | Rust CPU (`f64`) | 97.2% | 97.2% | 98.3% |
| Full set (6% missing, 4% spurious, 256 replicas) | WebGPU (Metal) | 95.7% | 97.1% | 98.1% |
| Degraded full set (15% missing, 10% spurious, 256 replicas) | WebGPU (Metal) | 96.9% | 91.2% | 96.7% |

## Benchmark suite (TODO)
1. Maintainer's completed assignment (private, `data/private/`).
2. 10 to 20 public BMRB entries spanning size, fold, and IDPs.
3. Synthetic stress set (noise, overlap, deuteration).

Regression rule: no merged change may lower accuracy or HC precision on the suite (`AGENTS.md`).
