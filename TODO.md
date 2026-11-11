# TODO

Target: a complete open equivalent of NMRtist (spectra in, peak lists,
assignments, structures, and interactive GUI/web workbench out), plus an
integrated 2D Chemical Structure, Reaction & 1D NMR Studio (`Octet-Draw`).
Parity table: `docs/specs/00-architecture.md`.

Status key: `[x]` done, `[ ]` open. Owner: `(M)` maintainer, `(A)` AI agents.
Specs: `docs/specs/`. Agent rules: `AGENTS.md`.

## Phase 1: Backbone assignment on real data
- [x] (A) NMR-STAR reader/writer, peak TSV format (`crates/nmr-core`)
- [x] (A) Spin system builder with competing candidates (`crates/nmr-core`)
- [x] (A) Scoring (typing LLR + connectivity) and simulated annealing (`crates/nmr-core`)
- [x] (A) Simulator, benchmark command, regression tests (`crates/nmr-core`, `crates/nmr-cli`)
- [ ] (M) Run benchmark on the completed assignment (`data/private/`)
- [ ] (M) Review and set PROVISIONAL constants in `specs/03` sections 3 and 4
- [x] (A) Rebuild priors from 36 public BMRB entries (`nmrassign priors`, `data/priors/bmrb_priors.json`, `data/priors/PROVENANCE.md`)
- [x] (A) Assemble public benchmark set: 12 BMRB entries across alpha, beta, alpha/beta & IDP folds (`data/benchmark/manifest.tsv`, `nmrassign benchmark-suite`)
- [ ] (M) Decide: secondary-structure-dependent priors?
- [x] (A) Intensity columns (`height`, `volume`) in peak TSV; intensity-aware `i` / `i-1` candidate disambiguation
- [x] (A) Re-test degenerate `i` / `i-1` handling once intensities exist

## Phase 2: Rust + WebGPU performance & interactive GUI (`specs/06`)
- [x] (A) Port full reference pipeline to pure Rust (`crates/nmr-core`) with `< 1e-13` `U`/`P` parity
- [x] (A) WebGPU (`wgpu`/WGSL) scoring matrices and multi-replica GPU annealer (`--engine gpu`)
- [x] (A) Interactive `eframe` + `wgpu` GUI workbench: GPU 2D HSQC contour shader, sequence ribbon, 3D sequential walk strips (`crates/nmr-gui`)
- [x] (A) CI running CPU, WebGPU, clippy, and WASM checks on the benchmark suite (`.github/workflows/ci.yml`)
- [x] (A) Browser WASM + WebGPU build target for `crates/nmr-gui` (`wasm32-unknown-unknown`, `crates/nmr-gui/index.html`)

## Phase 3: Peak picking (`specs/01`, `specs/02`)
- [ ] (M) Identify processed spectra behind the completed assignment
- [x] (A) Spectrum reader (Bruker `pdata` `2rr`/`3rrr`, NMRPipe `.ft2`/`.ft3`) with ppm axes
- [x] (A) Classical picker baseline: robust MAD noise, local extrema, sub-voxel log-Gaussian fit, HSQC strip restriction
- [x] (A) Synthetic 2D/3D spectrum renderer in simulator (`render_hsqc_2d`, `render_triple_3d`)
- [x] (A) Neural picker (2D & 3D separable residual CNN in `nmr-core::neural_picker` + WebGPU WGSL compute `nmr-gpu::neural_gpu`) trained on synthetic Voigt/ridge/baseline spectra (`nmrassign pick --method neural --engine gpu`)
- [x] (A) End-to-end benchmark on volumetric spectra: `spectra -> peaks -> assignment` (`nmrassign e2e-benchmark`)

## Phase 4: Sidechain assignment (`specs/07`)
- [x] (A) Residue topology tables for all 20 standard amino acids from IUPAC / PDB CCD (`nmr-core::topology`) + empirical BMRB sidechain shift priors
- [x] (A) Expected peaks & volumetric simulation for `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, and aromatic experiments (`nmr-core::sidechain`)
- [x] (A) Joint backbone + sidechain optimization (`solve_joint_backbone_sidechain`, `nmrassign sidechain`, pre-proline & prochiral handling)
- [x] (A) Peak picking for 13C-edited and 3D TOCSY spectra (`pick_c13_hsqc`, `pick_hcch_tocsy_3d`, `pick_ccconh_3d`, `pick_hccconh_3d`)

## Phase 5: Structure (`specs/04`)
- [x] (A) Shift-based backbone `(phi, psi)` dihedral restraint predictor (`nmr-core::dihedrals`)
- [x] (A) 3D `15N`-/`13C`-NOESY peak picking, $r^{-6}$ calibration, and volumetric simulator (`nmr-core::noesy`)
- [x] (A) Ambiguous NOE distance restraints ($d_{\text{eff}} = (\sum r^{-6})^{-1/6}$), network anchoring, and 7-cycle iterative ADR assignment (`nmr-core::noesy`, `nmr-core::structure`)
- [x] (A) Torsion-angle NeRF + analytical-gradient simulated annealing & multi-conformer PDB ensemble folding (`nmr-core::structure`, `nmrassign structure`)
- [x] (A) Kabsch superposition RMSD, Ramachandran basin, and restraint-violation validation against deposited PDB structures (`data/pdb_refs/`)

## Phase 6: Service (`specs/08`)
- [x] (A) One-command local pipeline: `spectra/peaks + sequence -> shifts.str + result.nef + report.html` (`nmrassign pipeline`)
- [x] (A) Self-hosted HTTP REST job API and background solver workers (`nmr-core::service`, `nmrassign serve`)
- [x] (A) Single-page browser upload & live job results dashboard (`GET /` on `nmrassign serve`) + `wasm32` WebGPU workbench
- [x] (A) Self-contained HTML + inline SVG quality report with CSI secondary shifts, Ramachandran plot, strip ladders & confidence summaries (`nmr-core::report`, `nmrassign report`)

## Phase 7: Integrated Octet-Draw Chemical Structure, Reaction & 1D NMR Suite
- [x] (A) Core 2D chemical graph, SSSR rings, Hückel aromaticity, CIP `(R)/(S)` & `(E)/(Z)` stereochemistry, Clean 2D layout, peptide/HELM builder, and multi-format I/O (`SMILES`, Reaction SMILES, `InChI`/`InChIKey`, `MOL` V2000/V3000, `SDF`, ChemDraw `CDXML` & binary `CDX`, `.oct`) (`nmr-core::octet_mol`)
- [x] (A) Analytical chemistry engine: 6-sphere HOSE + additivity ¹H/¹³C NMR shift & first-order multiplet predictor, Lorentzian 1D synthesis & `JCAMP-DX` I/O, IUPAC 2021 exact mass / isotopic envelopes / cLogP / TPSA, reaction stoichiometry solver, and offline bidirectional `IUPAC Name ↔ Structure` (`nmr-core::octet_analyt`, `nmrassign chem`)
- [x] (A) Publication-grade 2D chemical renderer (`ACS 1996`, `RSC`, `JOC`, `Angewandte`, `Nature`, `Dark`), 50+ template library, and pure-Rust `SVG`, `PDF 1.4`, and antialiased `PNG` exporters (`nmr-core::octet_render`)
- [x] (A) Interactive `egui` Octet Chemical Studio tab (`⚗️ Octet Chemical Draw & 1D NMR`) wired into `nmr-gui` (native desktop + `wasm32`) with live 1D NMR peak-to-atom highlighting and 1-click import from loaded protein sequences

## Housekeeping
- [x] (A) CCPN NEF v1.1 lossless import/export (`nmr-core::nef`, `nmrassign nef-import` / `nef-export`)
- [ ] (M) Choose a public project name (avoid commercial product names)
- [ ] (M) Patent search on deep-learning peak picking and automated assignment
- [x] (A) Add `CONTRIBUTING.md`, `THIRD_PARTY.md`, author metadata, and CI workflow
