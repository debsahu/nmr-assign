# TODO

Target: a complete open equivalent of NMRtist (spectra in, peak lists,
assignments, structures, and interactive GUI/web workbench out).
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
- [ ] (A) Rebuild priors from BMRB entries (`nmrassign priors`); record entry IDs
- [ ] (A) Assemble public benchmark set: 10 to 20 BMRB entries (`specs/05`)
- [ ] (M) Decide: secondary-structure-dependent priors?
- [ ] (A) Intensity columns in peak TSV; intensity-aware `i` / `i-1` scoring
- [ ] (A) Re-test degenerate `i` / `i-1` handling once intensities exist

## Phase 2: Rust + WebGPU performance & interactive GUI (`specs/06`)
- [x] (A) Port full reference pipeline to pure Rust (`crates/nmr-core`) with `< 1e-13` `U`/`P` parity
- [x] (A) WebGPU (`wgpu`/WGSL) scoring matrices and multi-replica GPU annealer (`--engine gpu`)
- [x] (A) Interactive `eframe` + `wgpu` GUI workbench: GPU 2D HSQC contour shader, sequence ribbon, 3D sequential walk strips (`crates/nmr-gui`)
- [ ] (A) CI running CPU and WebGPU tests on the benchmark suite
- [ ] (A) Browser WASM + WebGPU build target for `crates/nmr-gui`

## Phase 3: Peak picking (`specs/01`, `specs/02`)
- [ ] (M) Identify processed spectra behind the completed assignment
- [x] (A) Spectrum reader (Bruker `pdata` `2rr`/`3rrr`, NMRPipe `.ft2`/`.ft3`) with ppm axes
- [x] (A) Classical picker baseline: robust MAD noise, local extrema, sub-voxel log-Gaussian fit, HSQC strip restriction
- [x] (A) Synthetic 2D/3D spectrum renderer in simulator (`render_hsqc_2d`, `render_triple_3d`)
- [ ] (A) Neural picker (3D CNN on `wgpu`) trained on synthetic spectra
- [ ] (A) End-to-end benchmark on real spectra: `spectra -> peaks -> assignment`

## Phase 4: Sidechain assignment (`specs/07`)
- [ ] (A) Residue topology tables from the PDB Chemical Component Dictionary
- [ ] (A) Expected peaks for HCCH-TOCSY, (H)CC(CO)NH, 13C-HSQC
- [ ] (A) Joint backbone + sidechain optimization
- [ ] (A) Peak picking for 13C-edited and TOCSY spectra

## Phase 5: Structure (`specs/04`)
- [ ] (A) Shift-based dihedral restraint predictor
- [ ] (A) NOESY peak picking (hardest picking tier)
- [ ] (A) Ambiguous NOE restraints and iterative assignment
- [ ] (A) Structure annealing and ensemble selection
- [ ] (A) Validation against deposited PDB structures

## Phase 6: Service (`specs/08`)
- [ ] (A) One-command local pipeline: `spectra + sequence -> report`
- [ ] (A) Job API and GPU workers
- [ ] (A) Web upload and WebGPU results UI
- [ ] (A) Report with strip plots and confidence summaries

## Housekeeping
- [ ] (A) NEF import/export
- [ ] (M) Choose a public project name (avoid commercial product names)
- [ ] (M) Patent search on deep-learning peak picking and automated assignment
- [ ] (A) Publish repository; add `CONTRIBUTING.md`
