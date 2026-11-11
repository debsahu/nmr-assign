# 00. Architecture

Open, clean-room protein NMR pipeline written in pure Rust + WebGPU (`wgpu` / WGSL),
with an interactive `eframe` + `wgpu` scientific GUI. Each module has its own spec;
modules communicate through typed Rust data structures in `nmr-core` and the open
file formats in `01-data-io.md`.

## Goal: a complete open equivalent of NMRtist + interactive workbench

Feature parity target, mapped to modules:

| Capability | Crate / Module | Spec | Status |
|---|---|---|---|
| Upload sequence + processed spectra, run locally or in cloud | GUI + Service | 06, 08 | local GUI + CLI working |
| Automated peak picking, 2D/3D, classical + neural | `nmr-core::picker` | 02 | Stage A classical working |
| Backbone chemical shift assignment (CPU + WebGPU) | `nmr-core`, `nmr-gpu` | 03, 06 | v0.2 working |
| Interactive 2D GPU contours & 3D sequential walk strips | `nmr-gui` | 06 | v0.2 working |
| Sidechain chemical shift assignment | `nmr-core` | 07 | not started |
| NOESY peak assignment & 3D structure ensemble | Structure | 04 | not started |
| Confidence for every result | all | each | assignment working |
| Results export (NMR-STAR, TSV) | `nmr-core`, `nmr-gui` | 01 | working |

## Pipeline

| Stage | Module | Input | Output | Spec |
|---|---|---|---|---|
| 1 | Data & Spectrum I/O | Bruker `pdata`, NMRPipe, NMR-STAR, TSV | `Spectrum2D`/`3D`, `Reference`, `Peak` | 01 |
| 2 | Peak picking | processed spectra | peak lists (TSV) | 02 |
| 3 | Backbone assignment | sequence + peak lists | backbone shifts + confidence | 03 |
| 4 | Sidechain assignment | TOCSY/HCCH peaks + backbone | full shift table | 07 |
| 5 | NOESY assignment + structure | shifts + NOESY peaks | ensemble (PDB / mmCIF) | 04 |
| 6 | Service | user uploads | job results, reports | 08 |
| x | Benchmark and simulation | reference assignments | metrics, synthetic 2D/3D spectra | 05 |
| x | WebGPU compute & GUI | `nmr-core` data structures | GPU annealing, 2D contours, 3D strips | 06 |

## Principles

1. `nmr-core` (`f64`) is the clean mathematical reference; `nmr-gpu` (`wgpu`/WGSL) accelerates hot paths and is parity-tested against `nmr-core` (06).
2. Every module is benchmarked against ground truth before it is trusted (05).
3. Ambiguity is carried forward as scored candidates, never silently resolved.
4. Clean-room: implement from papers and open data only (see `AGENTS.md`).
5. Interchange formats are open standards (NMR-STAR, NEF, PDB) or the documented TSV formats in 01.

## Repository layout

```
AGENTS.md          agent rules (CLAUDE.md, GEMINI.md point here)
TODO.md            roadmap
docs/specs/        this folder
crates/nmr-core/   reference implementation, spectrum I/O, picker, simulator
crates/nmr-gpu/    wgpu + WGSL compute engine (scoring + multi-replica annealing)
crates/nmr-gui/    eframe + wgpu interactive spectrum & assignment workbench
crates/nmr-cli/    nmrassign CLI
data/fixtures/     synthetic or public test data
data/private/      unpublished data, never committed
```
