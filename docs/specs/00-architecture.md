# 00. Architecture

Open, clean-room protein NMR pipeline. Each module has its own spec; modules
talk only through the file formats in 01-data-io.md, so any module can be
replaced or rewritten (for example in Julia) without touching the others.

## Goal: a complete open equivalent of NMRtist

Feature parity target, mapped to modules:

| Capability (NMRtist) | Module | Spec | Status |
|---|---|---|---|
| Upload sequence + processed spectra, run in the cloud | Service | 08 | not started |
| Automated peak picking, 2D/3D/4D, all spectrum types | Peak picking | 02 | not started |
| Backbone chemical shift assignment | Assignment | 03 | v0.1 working (peak lists in) |
| Sidechain chemical shift assignment | Assignment | 07 | not started |
| NOESY peak assignment | Structure | 04 | not started |
| 3D structure calculation, ensemble out | Structure | 04 | not started |
| Confidence for every result | all | each | assignment only |
| Results download (peak lists, shifts, structures) | I/O + Service | 01, 08 | partial |

## Pipeline

| Stage | Module | Input | Output | Spec |
|---|---|---|---|---|
| 1 | Data I/O | Bruker / NMRPipe spectra, NMR-STAR, NEF | spectra, shift tables | 01 |
| 2 | Peak picking | processed spectra | peak lists (TSV) | 02 |
| 3 | Backbone assignment | sequence + peak lists | backbone shifts + confidence | 03 |
| 4 | Sidechain assignment | TOCSY/HCCH peaks + backbone | full shift table | 07 |
| 5 | NOESY assignment + structure | shifts + NOESY peaks | ensemble (PDB / mmCIF) | 04 |
| 6 | Service | user uploads | job results, reports | 08 |
| x | Benchmark and simulation | reference assignments | metrics | 05 |
| x | Julia core | same as Python modules | same | 06 |

## Principles

1. Python reference first, Julia for speed later, parity-tested (06).
2. Every module is benchmarked against ground truth before it is trusted (05).
3. Ambiguity is carried forward as scored candidates, never silently resolved.
4. Clean-room: implement from papers and open data only (see CLAUDE.md).
5. Interchange formats are open standards (NMR-STAR, NEF, PDB) or the
   documented TSV formats in 01.

## Repository layout

```
CLAUDE.md          agent rules
TODO.md            roadmap
docs/specs/        this folder
python/nmrassign/  reference implementation (one file per module stage)
python/tests/      regression tests
julia/             Julia core (planned)
data/fixtures/     synthetic or public test data
data/private/      unpublished data, never committed
```
