# 08. Service & Local Pipeline

Status: implemented (`crates/nmr-core/src/service.rs`, `crates/nmr-core/src/report.rs`, `crates/nmr-cli/src/main.rs`).

Goal: A user supplies a sequence and processed spectra (or peak lists), selects
experiment types and solver, and receives picked peak lists, chemical-shift
assignments (`NMR-STAR v3`, `TSV`, `NEF v1.1`), optional 3D structures, and a
self-contained HTML+SVG quality report.

## Components
- Job API (`nmrassign serve`): upload, validate (format, referencing, experiment
  type), queue, status, results download.
- Workers: run pipeline stages 2 to 5 on CPU/GPU.
- Report (`nmrassign report`): per-stage metrics, confidence plots, strip plots
  for checking assignments, structure validation summary.
- Local mode (`nmrassign pipeline`): runs the complete pipeline in a single CLI
  command with no server needed, emitting `peaks.tsv`, `shifts.str`,
  `shifts.tsv`, `result.nef`, and `report.html` into `--out-dir <dir>`:

```bash
nmrassign pipeline \
  --seq path/to/sequence.fasta \
  --hsqc path/to/hsqc \
  --triple HNCACB=path/to/hncacb \
  --triple CBCACONH=path/to/cbcaconh \
  --triple HNCO=path/to/hnco \
  --triple HNCACO=path/to/hncaco \
  --solver evo-graph \
  --out-dir ./out_assigned

nmrassign score \
  --pred ./out_assigned/shifts.str \
  --truth path/to/reference.str -v
```

Note: 3D experiments use `--triple EXP=path` (`=` separator), and `pipeline`
writes a multi-artifact directory via `--out-dir <dir>` (not `-o`).

## Principles
- Runs fully self-hosted; no data leaves the user's infrastructure unless
  they deploy it publicly.
- Every result reproducible from inputs + version + config.
