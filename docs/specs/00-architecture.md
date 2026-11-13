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
| Backbone chemical shift assignment (CPU `f64`, `evo-graph` & WebGPU `16 KiB` SRAM, per-engine `qc` calibration) | `solver`, `evo_graph`, `qc`, `nmr-gpu` | 03, 06, 10 | Working (12-protein BMRB + 26-entry real suite) |
| Problem-regime solvers: protonless `13C`-detected IDPs, `SSP`/`δ2D` propensity & paramagnetic `PRE`/`Δχ` `PCS` | `c13_detected`, `idp_solver`, `idp_propensity`, `paramagnetic`, `nmr-gpu::idp_gpu` | 14 | Working (`--regime idp-13c\|paramagnetic`, `propensity`) |
| Interactive 2D GPU contours, sequence ribbon, 2D covalent sidechain inspector & 3D walk strips | `nmr-gui`, `sidechain_widget` (Desktop + WASM) | 06, 07, 12 | Working |
| Built-in `wgpu` 3D biomolecular `.pdb`/`.cif` ribbon viewer, putty tracks, 3D-to-strip picking & NOE/PRE cylinders | `mol3d_viewer`, `mol3d_overlay`, `shaders/mol3d.wgsl` (Desktop + WASM) | 15 | Working (`nmr-gui` 3D viewport) |
| Aliphatic, aromatic (`Phe`/`Tyr`/`Trp`/`His`) & Bayesian stereospecific prochiral assignment | `topology`, `sidechain`, `aromatic`, `stereospec` | 07, 07b | Working (`nmrassign sidechain [--aromatic] [--stereospec]`) |
| Supramolecular `13C`-Methyl TROSY (`ILVMAT`) 2D/3D/4D picker & WebGPU/CPU contact-graph solver | `methyl_trosy`, `methyl_solver`, `nmr-gpu::methyl_gpu` | 13 | Working (`nmrassign methyl`) |
| Post-assignment 3/9-mer `(phi, psi, chi1)` `TorsionDb` (`HomologFirewall`), iterative NOESY ADR & Saupe `RDC` CPU/WebGPU folding | `dihedrals`, `torsion_db`, `noesy`, `rdc`, `structure`, `nmr-gpu::rdc_gpu` | 04, 15 | Working (`nmrassign torsions`, `structure [--rdc] [--torsion-db]`) |
| Self-contained HTML + inline SVG quality, Ramachandran, `CSI`, `RDC` (`D_calc` vs `D_obs` + Cornilescu `Q`) & IDP `SSP`/`δ2D` reports | `nmr-core::report` | 04, 08, 14 | Working (`nmrassign report [--rdc] [--idp-propensity]`) |
| Standards-compliant export (`NMR-STAR v3.2`, `CCPN NEF v1.1` incl. `dihedral` & `rdc` lists, `TSV`, multi-model `PDB`/`.cif`) | `nmrstar`, `nef`, `peaks`, `structure` | 01, 04 | Working |
| 2D Chemical Structure, Reaction, Stoichiometry, IUPAC Naming & 1D ¹H/¹³C NMR Studio | `octet_mol`, `octet_analyt`, `octet_render`, `octet_canvas` | — | Working (GUI tab + `nmrassign chem`) |
| Custom `pulseprogram` physics interpreter, GPU spin relaxation ($R_1/R_2/\text{NOE}$), Palmer Model-Free ($S^2$), `CEST`, `CPMG` & ligand `CSP` $K_d$ | `pulprog`, `relax_io`, `relax_fit`, `modelfree`, `cest`, `cpmg`, `csp`, `nmr-gpu` | 11, 16 | Planned (`Phase 13`, `T13.0`–`T13.6`) |
| Analytical 1D NMR Voigt deconvolution, Hoye multiplet `J`-trees, `qNMR`, multi-trace stack/overlay & pseudo-2D `DOSY` studio | `nmr1d`, `multiplet`, `dosy`, `nmr-gpu`, `stack1d_panel`, `dosy_panel` | 17 | Planned (`Phase 14`, `T14.1`–`T14.4`) |
| Clean-room Model Context Protocol (`MCP`) server (`stdio` & Streamable HTTP) & live Undo/Redo `nmr-gui` desktop bridge | `mcp`, `mcp_workflows`, `nmr-gui::mcp_bridge`, `nmr-cli` | 18 | Planned (`Phase 15`, `T15.1`–`T15.3`) |

## Pipeline

| Stage | Module | Input | Output | Spec |
|---|---|---|---|---|
| 0 | Raw FID, `pulprog`, NUS & Auto-Classify | Bruker `fid`/`ser`/`pulseprogram`, Varian `fid`+`procpar`, `.tar`/`.tgz` | Processed `Spectrum2D`/`3D`, detected experiment & problem regime | 11, 12 |
| 1 | Data & Spectrum I/O | Bruker `pdata`, NMRPipe, Sparky `.ucsf`, NMR-STAR, NEF, TSV, `.pdb`/`.cif` | `Spectrum2D`/`3D`, `Reference`, `Peak`, `NefDocument`, `RdcMediumSet` | 01, 04, 15 |
| 2 | Peak picking | Processed 2D/3D/4D spectra (`1H`- or `13C`-detected, methyl-TROSY) | Peak lists (`TSV` / `NEF`) + corroborated 3D strip rescue | 02, 13, 14 |
| 3 | Backbone & Regime assignment | Sequence + triple-resonance or `13C`-detected IDP / paramagnetic peaks | Backbone shifts + per-engine calibrated confidence (`CPU` / `WebGPU`) | 03, 10, 14 |
| 4 | Sidechain, Aromatic, Stereospec & Methyl | `13C`-HSQC / `HCCH-TOCSY` / `(HB)CB(CGCD)HD` / `3J` / Methyl-NOESY | All-atom shift table (`Ambiguity_code` `1`/`2`/`3`, `His` state, `ILVMAT`) | 07, 07b, 13 |
| 5 | IDP Propensity & `Δχ` Tensor | Assigned shifts (`idp_propensity`) or paramagnetic `PCS` + `PDB` (`paramagnetic`) | Per-residue `SSP` + 4-state `δ2D` (`TSV`/`JSON`/SVG), 5-param `Δχ` tensor | 14 |
| 6 | Torsions, NOESY ADR, `RDC` & 3D `wgpu` Viewer | Assigned shifts + 3D NOESY peaks + multi-medium `RDC`s (`1D_NH`/`CAHA`/`CACO`/`NC'`/`HNC'`) | Firewalled 3/9-mer torsions, CPU/WebGPU `RDC`-refined PDB ensemble, Cornilescu `Q`/`Q_free` & interactive 3D ribbons | 04, 15 |
| 7 | Reporting, HTTP & `MCP` Services | Shifts / peaks / ensemble / `RDC`s / browser uploads / `MCP` JSON-RPC | Single-file HTML+SVG report, `.str`, `.nef`, REST job status, planned `MCP` tools | 04, 08, 14, 18 |
| 8 | Octet Chemical & 1D/`DOSY` Studio | 2D sketch, `SMILES`/`InChI`/IUPAC, `CDXML`/`CDX`/`MOL`, 1D/pseudo-2D `DOSY` | Cleaned 2D coordinates, CIP stereo, ¹H/¹³C 1D NMR, planned Voigt/`J`-trees/`DOSY` | 17 |
| x | Benchmark & simulation | Public BMRB entries (`data/benchmark/` `manifest.tsv` & 26-entry `real_manifest.tsv`) | Synthetic self-consistency & non-circular real-data accuracy/HC-prec/`ECE_4`/`ECE_10`/`ECE_12` | 05, 11, 14 |

## Principles

1. `nmr-core` (`f64`) is the clean mathematical reference; `nmr-gpu` (`wgpu`/WGSL) accelerates hot paths (`U`/`P` scoring, `16 KiB` workgroup-SRAM multi-replica annealing for globular, `13C`-IDP & methyl-TROSY solvers, batched multi-conformer `NeRF` + Saupe `RDC` torsion gradient evaluation, and 3D CNN inference; synchronous GPU entry points return an explicit `Err` on `wasm32`, with async `Future` helpers prepared for browser callers) with 13-fold OOF LOPO `dev` per-engine calibrators (`STANDARD_CALIBRATION_KNOTS`, CPU-shrunk `STANDARD_GPU_CALIBRATION_KNOTS` with `w = 0.50`, `EVO_GRAPH_CALIBRATION_KNOTS`, and 5-fold LOPO `IDP_CALIBRATION_KNOTS` in `nmr-core::qc`) parity-tested against `nmr-core` (04, 06, 10, 13, 14).
2. Every module is benchmarked against ground truth before it is trusted (05).
3. Ambiguity is carried forward as scored candidates (`SpinSystem` clusters, `r^-6` ambiguous NOE restraints), never silently discarded.
4. Clean-room: implement from peer-reviewed papers, open BMRB/PDB data, and permissive Apache-2.0/MIT sources only (see `AGENTS.md` and `THIRD_PARTY.md`).
5. Interchange formats are open standards (`NMR-STAR v3`, `NEF v1.1`, `PDB`, `MOL`/`SDF`, `SMILES`, `InChI`, `JCAMP-DX`, `SVG`, `PDF`).
