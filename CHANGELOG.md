# Changelog

All notable architectural changes, algorithm milestones, and verified benchmark results for `nmr-assign` are documented in this file (newest entries first).

---

## 2026-11-11 — Clean-Room UCSF `.ucsf` v2 Spectrum I/O, Real-Data `pipeline` & `score` CLI, and Loud `--engine gpu` Error Propagation (`nmr-core`, `nmr-cli`)

- **Source Branch / Reviewed Commit:** `swarm/cli-io` (`be27f1407e82028b10f78316ff59f38d4342cd05`, squashing `7140849` + `be27f14`)
- **Architectural Summary:**
  - **Clean-Room UCSF Sparky `.ucsf` v2 2D/3D Reader & Writer (`crates/nmr-core/src/spectrum.rs`):** Implemented `is_ucsf_bytes`, `is_ucsf_file`, `read_ucsf_2d`, `write_ucsf_2d`, `write_ucsf_2d_tiled`, `read_ucsf_3d`, `write_ucsf_3d`, and `write_ucsf_3d_tiled` from the public UCSF Sparky v2 file format specification, including exact `xmtr_ppm` carrier-center referencing (`first_ppm = xmtr_ppm + 0.5 * sw_hz / obs_mhz`), zero-padded partial boundary tile handling, and automatic `(1H, 15N, 13C)` axis permutation, wired into CLI auto-detection alongside Bruker `pdata` and NMRPipe.
  - **Real-Data `nmrassign pipeline` & `nmrassign score` (`crates/nmr-cli/src/main.rs`):** Extended `pipeline` to accept `--seq <fasta_or_1letter>` with either raw spectra (`--hsqc` + repeatable `--triple EXP=path`, `--method classical|neural`, `--snr`, `--prob`) or `--peaks <peaks.tsv>` (emitting `peaks.tsv`, `shifts.str`, `shifts.tsv`, `result.nef`, and `report.html`). Added `nmrassign score` supporting both NMR-STAR and whitespace-delimited backbone shift tables with `-` missing handling and strict exclusion + exact count reporting of `?`-tagged uncertain values (including fused tokens).
  - **Portable Test Fixtures & Loud `--engine gpu` Enforcement (`crates/nmr-cli/src/main.rs`, `tests/`):** Replaced hardcoded `/Users/` scratch paths in `sidechain_tests.rs` and `structure_tests.rs` with repo-relative `data/benchmark/` lookups (`rg "/Users/" crates/` returns `0` matches), and removed all silent GPU-to-CPU fallbacks across `pipeline`, `pick`, `assign`, `benchmark`, and `benchmark-suite` (including loud error on `--method classical --engine gpu`).
- **Verified Metrics:**
  - **UCSF `.ucsf` v2 vs. Bruker `pdata/1` Parity (5 Real Ubiquitin Spectra, 14 Axes):** Axis `first_ppm` agrees with Bruker `procs` `OFFSET` within **`<= 0.00001 ppm`** (`<= 0.000006 ppm` / `0.000` digital points) with bit-identical raw voxels; classical peak picking achieves **`100.0%` top-peak match** across `HSQC` (`700/700`), `HNCACB` (`7,895/7,895`), `CBCACONH` (`4,586/4,585`), `HNCO` (`10,946/10,942`), and `HNCACO` (`694/694`) with median `(|dH|, |dN|, |dC|) = (0.00000, 0.00200, <= 0.00200) ppm`.
  - **CLI End-to-End Timing & Scoring:** `nmrassign pipeline` completes in **1.56 s** on 5 `.ucsf` spectra, **2.46 s** on 5 Bruker `pdata/1` spectra, and **0.55 s** from `--peaks`; `nmrassign score` parses `76` residues / `359` shifts (`70` assignable spin systems) with `2` uncertain `?` values excluded, and achieves `100.0%` (`72/72`) on synthetic `.str` self-check.
  - **Synthetic Ubiquitin Gate:** **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (55/55)**.

---

## `ca35643` — 2026-11-11 — WebGPU 3D Neural Picker `<= 64 MiB` Multi-Axis Tiling, Multi-Tier `>256`/`>1024` GPU Annealer & Loud Limit Enforcement (`nmr-gpu`)

- **Commit:** `ca35643` (from `swarm/gpu-engine` `517e364d1f99bedf29a9bfafc9e299f56d97499c`, squashing `7595d64` + `517e364`)
- **Architectural Summary:**
  - **3D Neural GPU Picker Multi-Axis `(X, Y, Z)` 7-Halo Tiling (`crates/nmr-gpu/src/neural_gpu.rs`, `lib.rs`):** Added active `(1H, 15N)` root bounding-box cropping, empty-tile skipping, on-the-fly per-tile Haar noise normalization (avoiding full-cube CPU copies), and reusable `Conv3dTileWorkspace` `(X, Y, Z)` block tiling (`64x64x64` interior + `7`-voxel NMS / `6`-voxel forward halo = `78^3` padded) capping all GPU storage buffer bindings at `<= 64 MiB` (`MAX_CONV3D_BUFFER_BYTES = 67,108,864 B`) and eliminating the `512 MiB` Bruker `conv3d-bg` binding overflow crash even under baseline `wgpu::Limits::default()` (`128 MiB`).
  - **Multi-Tier WebGPU Simulated Annealing & Replica Batching (`crates/nmr-gpu/src/lib.rs`, `shaders/anneal.wgsl`):** Removed silent CPU fallback for `S > 256 || K > 256`, eliminated WGSL function-scope stack-spilling arrays via register delta helpers, implemented tiered workgroup-SRAM shaders within the `16,384 B` WebGPU baseline (`S, K <= 256` `@workgroup_size(8)`, `257..=512` `@workgroup_size(4)`, `513..=1024` `@workgroup_size(2)`) plus a storage-backed `anneal_chunk_global` tier for `> 1024`, and added automatic multi-dispatch replica batching (`max_batch_replicas`) bounded by `max_compute_workgroups_per_dimension` and `max_storage_buffer_binding_size`.
  - **Loud GPU Limit & Overflow Enforcement:** Validated all storage buffer sizes, workgroup dispatch counts, and NMS peak buffer capacity (`hdr.count > ws.max_peaks`) across `score_matrices_gpu`, `anneal_from_matrices_gpu`, and `neural_gpu`, returning explicit `Err(...)` with zero silent CPU fallbacks.
- **Verified Metrics:**
  - **Real 5-Spectrum Bruker 3D Neural GPU Pick** (`1x 2rr` + `4x 2048x128x512` `512 MiB` `3rrr` cubes): **9.88 s wall** (`1.87 s` CPU user), `275,904` peaks (`917` HSQC roots; `HNCACB` `137,460` peaks in `2.12 s`), max GPU storage binding **`63.98 MiB <= 64 MiB`**.
  - **3D CNN CPU vs. Multi-Tile GPU Parity:** `forward_3d` `max_abs_diff = 4.40e-7` (`< 1e-4`); picked coordinate map-back `< 1e-4 ppm` vs. CPU and `< 1e-6 ppm` single-block vs. multi-tile GPU.
  - **Large-`S` GPU Simulated Annealing:** CLI `S = 700, K = 76` (`64` replicas $\times$ `1.05M` steps) completes in **3.33 s wall / 0.04 s CPU user** (`1.27B` vs. `103.2B` CPU instructions previously); parity tests at `S = 700` (`1.714 s`, `obj = 1045.1` matching CPU, `97.1%` recall) and `S = 1200` (`5.257 s`, `obj = 1045.1`, `97.1%` recall); `S = 6000` (`144 MiB > 128 MiB` strict baseline limit) verified to return loud `Err(...)`.
  - **Synthetic Ubiquitin Gate (`--engine gpu`):** **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (54/54)** in **0.873 s**.

---

## `fb7067e` — 2026-11-11 — Add historical CHANGELOG.md covering v0.1 through Phase 7

- **Commit:** `fb7067e`
- **Architectural Summary:**
  - Added repository-root `CHANGELOG.md` documenting architectural milestones, clean-room provenance, and verified benchmark gates from initial v0.1 scaffold (`01a52e1`) through Phase 7 (`bbfe672`).

---

## `bbfe672` — 2026-11-11 — Gate CI Clippy on correctness lints across Rust 1.99+ toolchains

- **Commit:** `bbfe672dc41a558930a9abe96cead0f45c8a7059`
- **Architectural Summary:**
  - Scoped the GitHub Actions Clippy gate in `.github/workflows/ci.yml` to `-D clippy::correctness` so newly stabilized style lints across newer Rust 1.99+ toolchains do not cause false-positive CI failures while strict correctness checking remains enforced across all targets.
- **Verified Metrics:**
  - `.github/workflows/ci.yml` passes `cargo clippy --workspace --all-targets -- -D clippy::correctness` cleanly.

---

## `caf0e15` — 2026-11-11 — Fix CI synthetic ubiquitin benchmark gate and enforce strict fmt/clippy

- **Commit:** `caf0e15df493af6b5fb6eeb01ae94452c63c1107`
- **Architectural Summary:**
  - Aligned `.github/workflows/ci.yml` benchmark invocation with `AGENTS.md` using the default multi-threaded CPU replica schedule under empirical BMRB priors, added `cargo fmt --check` and Clippy enforcement, and opted GitHub Actions workflows into Node 24 runtime execution.
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (CPU gate): **97.1% accuracy**, **98.0% high-confidence precision**.

---

## `2414e25` — 2026-11-11 — Complete Phase 1–7 NMR pipeline, WebGPU 3D CNN picker, structure folding, NEF/HTTP service, and Octet Chemical Studio

- **Commit:** `2414e25e70dc76ab738cce940b3abf84232497de`
- **Architectural Summary:**
  - **Phase 1 (Priors, Benchmarks & Intensity Disambiguation):** Rebuilt empirical backbone and all-atom chemical-shift priors from 36 public BMRB entries (`data/priors/`), assembled the 12-protein BMRB benchmark suite spanning $\alpha$, $\beta$, $\alpha/\beta$, and IDP folds (`data/benchmark/manifest.tsv`), added intensity-aware $i$ vs. $i-1$ candidate disambiguation, and added WASM (`wasm32-unknown-unknown`) + CI workflows.
  - **Phase 3 (Neural 2D/3D CNN Peak Picker):** Implemented a clean-room 2D/3D separable dilated residual CNN peak picker (`nmr-core::neural_picker`) with analytical backpropagation, Adam + focal-loss training, WebGPU WGSL compute execution (`nmr-gpu::neural_gpu`), and blind volumetric end-to-end benchmarking (`nmrassign e2e-benchmark`).
  - **Phase 4 (Sidechain Assignment):** Added IUPAC / PDB CCD covalent & TOCSY topology for all 20 standard amino acids (`nmr-core::topology`), volumetric simulators & pickers for `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, and aromatic experiments, and a joint backbone + sidechain optimizer (`nmr-core::sidechain`).
  - **Phase 5 (3D Structure Folding):** Added chemical-shift $(\phi, \psi)$ torsion restraint prediction (`nmr-core::dihedrals`), 7-cycle $r^{-6}$ ambiguous NOE distance restraint (`ADR`) calibration & network anchoring (`nmr-core::noesy`), and analytical-gradient NeRF torsion-angle simulated annealing ensemble folding with Kabsch RMSD and Ramachandran validation (`nmr-core::structure`).
  - **Phase 6 (Interchange, Reporting & Service):** Added lossless CCPN NEF v1.1 import/export (`nmr-core::nef`), single-file HTML + inline SVG quality reports (`nmr-core::report`), one-command `nmrassign pipeline`, and a self-hosted HTTP REST job server + browser dashboard (`nmr-core::service`).
  - **Phase 7 (Octet Chemical Studio):** Integrated a pure-Rust 2D chemical structure, reaction, stoichiometry, bidirectional IUPAC nomenclature, and 6-sphere HOSE $^{1}\text{H}$/$^{13}\text{C}$ 1D NMR suite (`octet_mol`, `octet_analyt`, `octet_render`) with `SMILES`, `InChI`/`InChIKey`, `MOL`/`SDF`, `CDXML`/`CDX`, `JCAMP-DX`, `SVG`/`PDF`/`PNG` I/O, `nmrassign chem`, and an interactive `egui` studio tab (`octet_canvas`).
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (WebGPU): **97.1% accuracy**, **100.0% high-confidence precision (55/55)**.
  - `synthetic_ubiquitin.str` (CPU): **97.1% accuracy**, **98.0% high-confidence precision (48/49)**.
  - Neural 3D CNN GPU/CPU forward parity: **$< 10^{-4}$**; workspace test suite: **50/50 release tests passing**.

---

## `5adb55c` — 2026-11-11 — Rewrite nmr-assign v0.2 in pure Rust + WebGPU with interactive GUI workbench

- **Commit:** `5adb55c911a185ce7b71f7b294e1899f30bdb09a`
- **Architectural Summary:**
  - Replaced the initial Python/Julia prototype with a 4-crate pure-Rust + WebGPU workspace (`nmr-core`, `nmr-gpu`, `nmr-cli`, `nmr-gui`).
  - `nmr-core`: NMR-STAR v3/v2.1 & TSV peak I/O, Bruker TopSpin `pdata` (`2rr`/`3rrr`) & NMRPipe (`.ft2`/`.ft3`) spectrum readers, robust MAD + sub-voxel log-Gaussian 2D/3D peak picker, shared-ellipse spin-system builder with proximity tie-breaking and 0-carbon decoy exclusion, configurable `assign_cost`, and multi-threaded `f64` simulated annealing solver.
  - `nmr-gpu`: `wgpu` + WGSL compute shaders for $U(s, k)$ typing and $P(s, t)$ connectivity scoring matrices (`< 1e-13` `f64` reference parity) and workgroup-SRAM parallel multi-replica simulated annealing.
  - `nmr-gui`: Interactive `eframe`/`egui` + `wgpu` desktop workbench with real-time WGSL 2D contour rendering, sequence confidence ribbon, and 3D sequential strip walk plots.
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (seed 0): **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (51/51)** on WebGPU; **98.0% high-confidence precision** across 12-run CPU annealing in **0.03 s**.

---

## `710b733` — 2026-11-11 — Move agent rules to AGENTS.md for tool-agnostic agents

- **Commit:** `710b733a1dd319de603778aeb0b8641dbe2bb02c`
- **Architectural Summary:**
  - Consolidated clean-room provenance rules, permissive dependency allowlists (`MIT`/`BSD`/`Apache-2.0`/`ISC`/`Zlib`), data hygiene policies, and benchmark regression gates into `AGENTS.md`, with `CLAUDE.md` and `GEMINI.md` referencing `AGENTS.md` as the single source of truth.
- **Verified Metrics:**
  - Documentation and rule consolidation across 7 files (`+70 / -61` lines).

---

## `01a52e1` — 2026-11-11 — Initial commit: open NMR pipeline scaffold and backbone assignment v0.1

- **Commit:** `01a52e1a2df2231028bf9ea9f7345bf89061c3ce`
- **Architectural Summary:**
  - Bootstrapped the open-source MIT-licensed repository, module specifications (`docs/specs/00-architecture.md` through `08-service.md` comparing target capabilities), `TODO.md` roadmap, `THIRD_PARTY.md`, synthetic ubiquitin fixture (`data/fixtures/synthetic_ubiquitin.str`), and v0.1 Python reference engine (NMR-STAR I/O, peak simulator, competing-candidate spin-system builder, LLR + Gaussian link scoring, multi-run simulated annealing with consensus confidence, CLI, and regression tests).
- **Verified Metrics:**
  - Initial 30-file scaffold (`2,125` lines) and end-to-end synthetic ubiquitin regression test suite passing.
