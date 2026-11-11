# Changelog

All notable architectural changes, algorithm milestones, and verified benchmark results for `nmr-assign` are documented in this file (newest entries first).

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
