# nmr-assign

**Open-source, GPU-accelerated protein NMR backbone resonance assignment and interactive WebGPU spectrum workbench — written in pure Rust.**

`nmr-assign` takes processed 2D/3D NMR spectra (Bruker TopSpin `pdata`, NMRPipe `.ft2`/`.ft3`) or peak lists and produces sequence-specific backbone chemical shift assignments (`H`, `N`, `CA`, `CB`, `C`) with calibrated per-residue confidences, alongside an interactive desktop workbench for 2D contour inspection and 3D sequential strip walks.

Built clean-room under the MIT licence from peer-reviewed methods and public BMRB statistics, with zero proprietary or copyleft dependencies.

---

## Highlights

- **Cross-platform WebGPU compute (`wgpu` + WGSL)**: Evaluates residue-typing log-likelihood-ratio matrices $U(s, k)$, Gaussian sequential connectivity matrices $P(s, t)$, and hundreds of parallel Metropolis-Hastings simulated annealing replicas directly in on-chip GPU workgroup SRAM (Apple Metal, Vulkan, DX12, and WebGPU), with a deterministic multi-threaded `f64` CPU fallback (`--engine cpu`).
- **Interactive spectrum & strip workbench (`eframe` / `egui` + `wgpu`)**:
  - Real-time GPU 2D positive/negative contour shader (`contour_shader.wgsl`) with smooth pan/zoom and dynamic threshold controls.
  - Clickable sequence & consensus-confidence ribbon (`>= 0.8` high-confidence highlighting).
  - 3D `HNCACB` / `CBCA(CO)NH` sequential walk strip plots (`i - 1`, `i`, `i + 1`) with phase-colored `CA`/`CB` bars and connectivity tie-lines.
- **Native spectrum I/O & classical peak picking**:
  - Reads Bruker TopSpin `pdata` (`procs`/`proc2s`/`proc3s` + submatrix-untiled `2rr`/`3rrr`) and NMRPipe 512-float header `.ft2`/`.ft3` cubes with full `ppm` axis referencing.
  - Two-pass outlier-trimmed MAD noise estimation, local-extrema detection, strip-restricted 3D picking under 2D $^{15}\text{N}$-HSQC roots, and 3-point parabolic/log-Gaussian sub-voxel peak refinement.
- **Ambiguity-preserving spin-system builder**:
  - Attaches 3D triple-resonance peaks across overlapping $^{1}\text{H}$–$^{15}\text{N}$ roots (`d <= 1`), resolves `i` vs `i - 1` offsets, retains competing `(ppm, support)` cluster hypotheses per `(atom, offset)`, and automatically excludes 0-carbon artifacts such as sidechain $\text{NH}_2$ roots.
- **Standards-compliant I/O**: Reads and writes NMR-STAR v3.2 (`_Atom_chem_shift`, with v2.1 fallback) and plain TSV peak/shift tables.

## Supported Experiments (v0.2)

| Experiment | Observed Carbon(s) | Residue Offset | Phase / Sign |
| :--- | :--- | :--- | :--- |
| `HSQC` ($^{15}\text{N}$-HSQC) | — (defines $^{1}\text{H}^{\text{N}}, {}^{15}\text{N}$ roots) | $i$ | — |
| `HNCA` | $\text{C}^\alpha$ | $i$ and $i - 1$ | $+$ |
| `HN(CO)CA` (`HNCOCA`) | $\text{C}^\alpha$ | $i - 1$ only | $+$ |
| `HNCACB` | $\text{C}^\alpha$, $\text{C}^\beta$ | $i$ and $i - 1$ | $+\text{C}^\alpha$ / $-\text{C}^\beta$ |
| `CBCA(CO)NH` (`CBCACONH`) | $\text{C}^\alpha$, $\text{C}^\beta$ | $i - 1$ only | $+$ |
| `HNCO` | $\text{C}'$ | $i - 1$ only | $+$ |
| `HN(CA)CO` (`HNCACO`) | $\text{C}'$ | $i$ and $i - 1$ | $+$ |

 Subset combinations (for example `HSQC + HNCACB + CBCA(CO)NH` or `HSQC + HNCA + HN(CO)CA + HNCO`) are supported out of the box.

---

## Quick Start

Requires **Rust 1.85+** (`cargo`). Automatically selects **Metal** on macOS, **Vulkan / DX12** on Linux & Windows, or multi-threaded CPU (`--engine cpu`).

```bash
# 1. Launch the interactive WebGPU workbench (auto-loads a 76-aa ubiquitin demo)
cargo run --release -p nmr-gui

# 2. Run a blind assignment benchmark from a reference NMR-STAR entry
cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --engine gpu -v
cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --engine cpu -v

# 3. Pick peaks directly from Bruker pdata or NMRPipe spectra, then assign
cargo run --release -p nmr-cli -- pick \
  --hsqc path/to/bruker/10/pdata/1 \
  --triple HNCACB=path/to/hncacb.ft3 \
  --triple CBCACONH=path/to/cbcaconh.ft3 \
  --out peaks.tsv --snr 5.5

cargo run --release -p nmr-cli -- assign \
  --peaks peaks.tsv \
  --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
  --out ubiquitin_assigned --engine gpu

# 4. Recompute amino-acid chemical shift priors from public BMRB NMR-STAR files
cargo run --release -p nmr-cli -- priors --stars path/to/bmrb_stars/ --out priors.json

# 5. Run the full test suite (I/O round-trips, picker, CPU benchmark floor, and GPU parity)
cargo test --release --workspace
```

---

## Benchmark (76-residue Ubiquitin, Blind Simulated Peak Lists)

Simulated with realistic thermal jitter ($0.008\text{ ppm }{}^{1}\text{H}$, $0.06\text{ ppm }{}^{15}\text{N}$, $0.04\text{ ppm }{}^{13}\text{C}$), $6\%$ missing triple-resonance peaks, $3\%$ missing HSQC roots, $4\%$ spurious 3D noise peaks, and 18 Asn/Gln sidechain $\text{NH}_2$ decoy roots:

| Scenario | Engine | Replicas | Accuracy | Recall | High-Conf (`>= 0.8`) Precision | Shift Agreement (`H` / `N` / `CA` / `CB`) | Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard 7-exp set (6% missing, 4% spurious) | WebGPU (Apple M5 Pro, Metal) | 64 | **97.1%** | **97.1%** | **100.0%** (51/51) | `68/68`, `68/68`, `74/75`, `68/69` | 3.7 s |
| Standard 7-exp set (6% missing, 4% spurious) | Multi-core CPU (`f64`) | 12 | **97.1%** | **97.1%** | **98.0%** (48/49) | `68/68`, `68/68`, `74/75`, `68/69` | **0.03 s** |

`nmr-gpu` (`f32` WGSL compute shaders) matches the `nmr-core` `f64` reference scoring matrices $U$ and $P$ within `< 1e-3` across all entries (1 `f32` ULP near $176\text{ ppm}$ carbonyl shifts).

---

## Workspace Architecture

```text
crates/
  nmr-core/     f64 reference algorithms: NMR-STAR & TSV I/O, Bruker pdata & NMRPipe
                readers, MAD/parabolic 2D & 3D peak picker, spin-system builder,
                LLR + Gaussian scoring, multi-threaded annealer, spectrum simulator
  nmr-gpu/      Headless & shared-device WebGPU (wgpu + WGSL) compute engine for
                scoring matrices and workgroup-SRAM parallel simulated annealing
  nmr-gui/      Interactive eframe/egui + wgpu workbench (GPU 2D contour shader,
                sequence confidence ribbon, 3D sequential strip inspector)
  nmr-cli/      `nmrassign` command-line tool (benchmark, assign, pick, simulate, priors)
docs/specs/     Module specifications & scoring equations (see 00-architecture.md)
data/fixtures/  Synthetic & public-BMRB test fixtures
```

---

## Roadmap

- [x] NMR-STAR v3 / TSV I/O, multi-hypothesis spin-system builder, LLR + Gaussian connectivity solver
- [x] Pure Rust + WebGPU (`wgpu` / WGSL) compute engine & interactive `egui` 2D/3D strip workbench
- [x] Bruker `pdata` (`2rr`/`3rrr`) & NMRPipe (`.ft2`/`.ft3`) spectrum readers + classical sub-voxel peak picker
- [ ] Recompute production shift priors & multi-protein benchmark suite from 10–20 public BMRB entries
- [ ] Neural 3D CNN peak picker (`wgpu` inference) for overlapped/low-SNR regions
- [ ] Aliphatic & aromatic sidechain assignment (`HCCH-TOCSY`, `HCCCONH`)
- [ ] 3D $^{15}\text{N}$/$^{13}\text{C}$-NOESY restraint assignment & distance-geometry / simulated-annealing fold calculation

See [`TODO.md`](TODO.md) and [`docs/specs/`](docs/specs/) for details, and [`AGENTS.md`](AGENTS.md) for clean-room and contribution rules.

## Licence

Released under the [MIT License](LICENSE). Third-party crate licences are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
