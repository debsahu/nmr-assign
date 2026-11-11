# 05. Benchmark and simulation

## Simulator (`crates/nmr-core/src/simulate.rs`)
From a reference NMR-STAR assignment:
1. `simulate` (`SimConfig::default()`): generates deterministic blind peak lists across all 7 experiments with jitter (`H` 0.008, `N` 0.06, `C` 0.04 ppm), missing peaks (6%), missing roots (3%), Asn/Gln NH2 sidechain roots, and uniform spurious peaks (4%). Ground truth is returned separately and never reaches the solver.
2. `simulate` (`SimConfig::realistic_bruker()`): models realistic 5-spectrum Bruker cryoprobe acquisitions (`HSQC`, `HNCACB`, `CBCACONH`, `HNCO`, `HNCACO` — omitting `HNCA`/`HNCOCA`), systematic spectrometer referencing offsets (`-2.75 ppm` TMS-vs-DSS `13C`, `-0.02 ppm` `1H`, `+0.25 ppm` `15N`), wide log-normal conformational-exchange intensity dispersion (`exchange_sigma = 0.60`) + `4%` exchange-broadened missing amides, paired Asn/Gln `NH2` sidechain doublets with `HNCO`/`CBCACONH` sidechain correlations, F1/F2 truncation `sinc` sidelobes, indirect `t1`-ridge spurious peaks, and residual water/aliphatic roots.
3. `render_hsqc_2d` & `render_triple_3d`:
   - `SpectrumRenderConfig::default()`: cropped amide/aliphatic windows (`10.6..5.8 ppm` `1H`, `136..102 ppm` `15N`, `75..14 ppm` `13C`), mixed Lorentzian-Gaussian lineshapes, and uniform thermal noise (`SNR ~ 67`).
   - `SpectrumRenderConfig::realistic_bruker()` (and `E2eBenchmarkConfig::realistic_bruker()`): full uncropped Bruker sweep widths (`11.75..-2.25 ppm` `1H`, `137.5..100.2 ppm` `15N`, `76.5..1.63 ppm` aliphatic `13C`, `187.0..165.0 ppm` carbonyl `13C`), high dynamic range (`peak_amp / thermal_noise ~ 2,500–4,000`), finite-`TD` apodized `sinc` truncation wiggles (`2–6%` positive/negative sidelobes), multiplicative `t1` indirect ridge noise along strong columns/strips, `~4.77 ppm` residual water suppression stripe, upfield `< 4.3 ppm` aliphatic bleed, and gentle baseline roll.

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
