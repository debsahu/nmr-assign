# 00. Architecture

Open, clean-room protein NMR & chemical structure pipeline written in pure Rust + WebGPU (`wgpu` / WGSL),
with an interactive `eframe` + `wgpu` scientific GUI (native desktop & `wasm32-unknown-unknown` browser WebGPU).
Each module has its own spec; modules communicate through typed Rust data structures in `nmr-core` and open
interchange formats (`NMR-STAR v3.2`, `CCPN NEF v1.1`, `PDB`, `CDXML`/`CDX`, `MOL`/`SDF`, `SMILES`, `InChI`, `JCAMP-DX`).

## Goal: a complete open equivalent of NMRtist + interactive WebGPU & Chemical Workbench

Feature parity target, mapped to modules:

| Capability | Crate / Module | Spec | Status |
|---|---|---|---|
| Raw time-domain FID, hypercomplex IST/FISTA NUS & multidimensional DSP operators | `fid`, `process`, `ops` | 11 | Working (`nmrassign process`, `--fid`) |
| Zero-friction spectrometer folder/archive classifier & `.nef` undo/redo session state | `classify`, `session`, `nmr-gui` | 06, 12 | Working (`< 50 ms` classify, `Cmd/Ctrl+Z`) |
| Upload sequence + processed spectra, run locally or via HTTP service | `nmr-gui`, `nmr-core::service`, `nmr-cli` | 06, 08 | Working (`nmrassign pipeline` & `serve`) |
| Automated 2D/3D peak picking (classical MAD + separable residual CNN on CPU & WebGPU) | `picker`, `neural_picker`, `nmr-gpu::neural_gpu` | 02 | Working (`--method classical\|neural`) |
| Backbone chemical shift assignment (CPU `f64`, `evo-graph` & WebGPU `16 KiB` SRAM, per-engine `qc` calibration) | `solver`, `evo_graph`, `qc`, `nmr-gpu` | 03, 06, 10 | Working (12-protein BMRB + 16-entry real suite) |
| Problem-regime solvers: protonless `13C`-detected IDPs, `SSP`/`δ2D` propensity & paramagnetic `PRE`/`Δχ` `PCS` | `c13_detected`, `idp_solver`, `idp_propensity`, `paramagnetic`, `nmr-gpu::idp_gpu` | 14 | Working (`--regime idp-13c\|paramagnetic`, `propensity`) |
| Interactive 2D GPU contours, sequence ribbon, 2D covalent sidechain inspector & 3D walk strips | `nmr-gui`, `sidechain_widget` (Desktop + WASM) | 06, 07, 12 | Working |
| Aliphatic, aromatic (`Phe`/`Tyr`/`Trp`/`His`) & Bayesian stereospecific prochiral assignment | `topology`, `sidechain`, `aromatic`, `stereospec` | 07, 07b | Working (`nmrassign sidechain [--aromatic] [--stereospec]`) |
| Supramolecular `13C`-Methyl TROSY (`ILVMAT`) 2D/3D/4D picker & WebGPU/CPU contact-graph solver | `methyl_trosy`, `methyl_solver`, `nmr-gpu::methyl_gpu` | 13 | Working (`nmrassign methyl`) |
| `(phi, psi)` dihedral prediction, iterative NOESY ADR & torsion-angle 3D folding | `dihedrals`, `noesy`, `structure` | 04 | Working (`nmrassign structure`) |
| Self-contained HTML + inline SVG quality, Ramachandran, `CSI` & IDP `SSP`/`δ2D` reports | `nmr-core::report` | 08, 14 | Working (`nmrassign report [--idp-propensity]`) |
| Standards-compliant export (`NMR-STAR v3.2`, `CCPN NEF v1.1`, `TSV`, multi-model `PDB`) | `nmrstar`, `nef`, `peaks`, `structure` | 01, 04 | Working |
| 2D Chemical Structure, Reaction, Stoichiometry, IUPAC Naming & 1D ¹H/¹³C NMR Studio | `octet_mol`, `octet_analyt`, `octet_render`, `octet_canvas` | — | Working (GUI tab + `nmrassign chem`) |

## Pipeline

| Stage | Module | Input | Output | Spec |
|---|---|---|---|---|
| 0 | Raw FID, NUS & Auto-Classify | Bruker `fid`/`ser`, Varian `fid`+`procpar`, `.tar`/`.tgz` | Processed `Spectrum2D`/`3D`, detected experiment & problem regime | 11, 12 |
| 1 | Data & Spectrum I/O | Bruker `pdata`, NMRPipe, Sparky `.ucsf`, NMR-STAR, NEF, TSV | `Spectrum2D`/`3D`, `Reference`, `Peak`, `NefDocument` | 01 |
| 2 | Peak picking | Processed 2D/3D/4D spectra (`1H`- or `13C`-detected, methyl-TROSY) | Peak lists (`TSV` / `NEF`) + corroborated 3D strip rescue | 02, 13, 14 |
| 3 | Backbone & Regime assignment | Sequence + triple-resonance or `13C`-detected IDP / paramagnetic peaks | Backbone shifts + per-engine calibrated confidence (`CPU` / `WebGPU`) | 03, 10, 14 |
| 4 | Sidechain, Aromatic, Stereospec & Methyl | `13C`-HSQC / `HCCH-TOCSY` / `(HB)CB(CGCD)HD` / `3J` / Methyl-NOESY | All-atom shift table (`Ambiguity_code` `1`/`2`/`3`, `His` state, `ILVMAT`) | 07, 07b, 13 |
| 5 | IDP Propensity & `Δχ` Tensor | Assigned shifts (`idp_propensity`) or paramagnetic `PCS` + `PDB` (`paramagnetic`) | Per-residue `SSP` + 4-state `δ2D` (`TSV`/`JSON`/SVG), 5-param `Δχ` tensor | 14 |
| 6 | NOESY ADR + 3D structure | Chemical shifts + 3D NOESY peaks | Multi-conformer PDB ensemble + RMSD / Ramachandran stats | 04 |
| 7 | Reporting & HTTP service | Shifts / peaks / ensemble / browser uploads | Single-file HTML+SVG report, `.str`, `.nef`, REST job status | 08, 14 |
| 8 | Octet Chemical Studio | 2D sketch, `SMILES`/`InChI`/IUPAC, `CDXML`/`CDX`/`MOL` | Cleaned 2D coordinates, CIP stereo, ¹H/¹³C 1D NMR, `SVG`/`PDF`/`PNG`/`JDX` | — |
| x | Benchmark & simulation | Public BMRB entries (`data/benchmark/` `manifest.tsv` & `real_manifest.tsv`) | Synthetic self-consistency & non-circular real-data accuracy/HC-prec/`ECE` | 05, 11, 14 |

## Principles

1. `nmr-core` (`f64`) is the clean mathematical reference; `nmr-gpu` (`wgpu`/WGSL) accelerates hot paths (`U`/`P` scoring, `16 KiB` workgroup-SRAM multi-replica annealing for globular, `13C`-IDP & methyl-TROSY solvers, and 3D CNN inference; synchronous GPU entry points return an explicit `Err` on `wasm32`, with low-level `Future` helper `GpuEngine::read_buffer_vec_async` prepared for async browser callers) with 10-fold OOF LOPO per-engine calibrators (`STANDARD_CALIBRATION_KNOTS`, `STANDARD_GPU_CALIBRATION_KNOTS`, `EVO_GRAPH_CALIBRATION_KNOTS` in `nmr-core::qc`) parity-tested against `nmr-core` (06, 10, 13, 14).
2. Every module is benchmarked against ground truth before it is trusted (05).
3. Ambiguity is carried forward as scored candidates (`SpinSystem` clusters, `r^-6` ambiguous NOE restraints), never silently discarded.
4. Clean-room: implement from peer-reviewed papers, open BMRB/PDB data, and permissive Apache-2.0/MIT sources only (see `AGENTS.md` and `THIRD_PARTY.md`).
5. Interchange formats are open standards (`NMR-STAR v3`, `NEF v1.1`, `PDB`, `MOL`/`SDF`, `SMILES`, `InChI`, `JCAMP-DX`, `SVG`, `PDF`).
