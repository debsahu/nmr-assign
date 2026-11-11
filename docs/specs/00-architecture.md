# 00. Architecture

Open, clean-room protein NMR & chemical structure pipeline written in pure Rust + WebGPU (`wgpu` / WGSL),
with an interactive `eframe` + `wgpu` scientific GUI (native desktop & `wasm32-unknown-unknown` browser WebGPU).
Each module has its own spec; modules communicate through typed Rust data structures in `nmr-core` and open
interchange formats (`NMR-STAR v3.2`, `CCPN NEF v1.1`, `PDB`, `CDXML`/`CDX`, `MOL`/`SDF`, `SMILES`, `InChI`, `JCAMP-DX`).

## Goal: a complete open equivalent of NMRtist + interactive WebGPU & Chemical Workbench

Feature parity target, mapped to modules:

| Capability | Crate / Module | Spec | Status |
|---|---|---|---|
| Upload sequence + processed spectra, run locally or via HTTP service | `nmr-gui`, `nmr-core::service`, `nmr-cli` | 06, 08 | Working (`nmrassign pipeline` & `serve`) |
| Automated 2D/3D peak picking (classical MAD + separable residual CNN on CPU & WebGPU) | `picker`, `neural_picker`, `nmr-gpu::neural_gpu` | 02 | Working (`--method classical\|neural`) |
| Backbone chemical shift assignment (CPU `f64` + WebGPU workgroup-SRAM annealing) | `nmr-core::solver`, `nmr-gpu` | 03, 06 | Working (12-protein BMRB suite) |
| Interactive 2D GPU contours, sequence ribbon & 3D sequential walk strips | `nmr-gui` (Desktop + WASM) | 06 | Working |
| Aliphatic & aromatic sidechain assignment (`C13HSQC`, `HCCH-TOCSY`, `(H)CC(CO)NH`) | `topology`, `sidechain` | 07 | Working (`nmrassign sidechain`) |
| `(phi, psi)` dihedral prediction, iterative NOESY ADR & torsion-angle 3D folding | `dihedrals`, `noesy`, `structure` | 04 | Working (`nmrassign structure`) |
| Self-contained HTML + inline SVG quality & validation reports | `nmr-core::report` | 08 | Working (`nmrassign report`) |
| Standards-compliant export (`NMR-STAR v3.2`, `CCPN NEF v1.1`, `TSV`, multi-model `PDB`) | `nmrstar`, `nef`, `peaks`, `structure` | 01, 04 | Working |
| 2D Chemical Structure, Reaction, Stoichiometry, IUPAC Naming & 1D ¹H/¹³C NMR Studio | `octet_mol`, `octet_analyt`, `octet_render`, `octet_canvas` | — | Working (GUI tab + `nmrassign chem`) |

## Pipeline

| Stage | Module | Input | Output | Spec |
|---|---|---|---|---|
| 1 | Data & Spectrum I/O | Bruker `pdata`, NMRPipe, NMR-STAR, NEF, TSV | `Spectrum2D`/`3D`, `Reference`, `Peak`, `NefDocument` | 01 |
| 2 | Peak picking | Processed 2D/3D spectra | Peak lists (`TSV` / `NEF`) | 02 |
| 3 | Backbone assignment | Sequence + triple-resonance peaks | Backbone shifts + consensus confidence | 03 |
| 4 | Sidechain assignment | `13C`-HSQC / `HCCH-TOCSY` / `(H)CC(CO)NH` + backbone | All-atom shift table (`H`, `N`, `C`, sidechains) | 07 |
| 5 | NOESY ADR + 3D structure | Chemical shifts + 3D NOESY peaks | Multi-conformer PDB ensemble + RMSD / Ramachandran stats | 04 |
| 6 | Reporting & HTTP service | Shifts / peaks / ensemble / browser uploads | Single-file HTML+SVG report, `.str`, `.nef`, REST job status | 08 |
| 7 | Octet Chemical Studio | 2D sketch, `SMILES`/`InChI`/IUPAC, `CDXML`/`CDX`/`MOL` | Cleaned 2D coordinates, CIP stereo, ¹H/¹³C 1D NMR, `SVG`/`PDF`/`PNG`/`JDX` | — |
| x | Benchmark & simulation | Public BMRB entries (`data/benchmark/`) | Accuracy/recall/HC-precision tables, volumetric 2D/3D cubes | 05 |

## Principles

1. `nmr-core` (`f64`) is the clean mathematical reference; `nmr-gpu` (`wgpu`/WGSL) accelerates hot paths (`U`/`P` scoring, multi-replica annealing, 3D CNN inference) and is parity-tested against `nmr-core` (06).
2. Every module is benchmarked against ground truth before it is trusted (05).
3. Ambiguity is carried forward as scored candidates (`SpinSystem` clusters, `r^-6` ambiguous NOE restraints), never silently discarded.
4. Clean-room: implement from peer-reviewed papers, open BMRB/PDB data, and permissive Apache-2.0/MIT sources only (see `AGENTS.md` and `THIRD_PARTY.md`).
5. Interchange formats are open standards (`NMR-STAR v3`, `NEF v1.1`, `PDB`, `MOL`/`SDF`, `SMILES`, `InChI`, `JCAMP-DX`, `SVG`, `PDF`).
