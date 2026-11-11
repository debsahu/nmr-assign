# TODO

Target: a complete open equivalent of NMRtist (spectra in, peak lists,
assignments, and structures out). Parity table: docs/specs/00-architecture.md.

Status key: [x] done, [ ] open. Owner: (M) maintainer, (A) AI agents.
Specs: docs/specs/. Agent rules: AGENTS.md.

## Phase 1: Backbone assignment on real data
- [x] (A) NMR-STAR reader/writer, peak TSV format
- [x] (A) Spin system builder with competing candidates
- [x] (A) Scoring (typing LLR + connectivity) and simulated annealing
- [x] (A) Simulator, benchmark command, regression tests
- [ ] (M) Run benchmark on the completed assignment (data/private/)
- [ ] (M) Review and set PROVISIONAL constants in specs/03 sections 3 and 4
- [ ] (A) Rebuild priors from BMRB entries (`cli priors`); record entry IDs
- [ ] (A) Assemble public benchmark set: 10 to 20 BMRB entries (specs/05)
- [ ] (M) Decide: secondary-structure-dependent priors?
- [ ] (A) Intensity columns in peak TSV; intensity-aware i / i-1 scoring
- [ ] (A) Re-test degenerate i / i-1 handling once intensities exist

## Phase 2: Performance
- [ ] (A) Julia port of scoring matrices with parity test (specs/06)
- [ ] (A) Julia annealer; `--engine julia`
- [ ] (A) CI running both engines on the benchmark suite

## Phase 3: Peak picking
- [ ] (M) Identify processed spectra behind the completed assignment
- [ ] (A) Spectrum reader (Bruker pdata, NMRPipe) with ppm axes (specs/01)
- [ ] (A) Classical picker: noise estimate, maxima, lineshape fit (specs/02)
- [ ] (A) Spectrum renderer in the simulator for synthetic training data
- [ ] (A) Neural picker (3D CNN) trained on synthetic spectra
- [ ] (A) End-to-end benchmark: spectra -> peaks -> assignment

## Phase 4: Sidechain assignment (specs/07)
- [ ] (A) Residue topology tables from the PDB Chemical Component Dictionary
- [ ] (A) Expected peaks for HCCH-TOCSY, (H)CC(CO)NH, 13C-HSQC
- [ ] (A) Joint backbone + sidechain optimization
- [ ] (A) Peak picking for 13C-edited and TOCSY spectra

## Phase 5: Structure (specs/04)
- [ ] (A) Shift-based dihedral restraint predictor
- [ ] (A) NOESY peak picking (hardest picking tier)
- [ ] (A) Ambiguous NOE restraints and iterative assignment
- [ ] (A) Structure annealing and ensemble selection
- [ ] (A) Validation against deposited PDB structures

## Phase 6: Service (specs/08)
- [ ] (A) One-command local pipeline: spectra + sequence -> report
- [ ] (A) Job API and workers
- [ ] (A) Web upload and results UI
- [ ] (A) Report with strip plots and confidence summaries

## Housekeeping
- [ ] (A) NEF import/export
- [ ] (M) Choose a public project name (avoid commercial product names)
- [ ] (M) Patent search on deep-learning peak picking and automated assignment
- [ ] (A) Publish repository; add CONTRIBUTING.md
