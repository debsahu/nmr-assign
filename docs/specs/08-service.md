# 08. Service & Local Pipeline

Status: implemented (`crates/nmr-core/src/service.rs`, `crates/nmr-core/src/report.rs`, `crates/nmr-cli/src/main.rs`).

Goal: A user supplies a sequence and processed spectra (or peak lists), selects
experiment types and solver, and receives picked peak lists, chemical-shift
assignments (`NMR-STAR v3`, `TSV`, `NEF v1.1`), optional 3D structures, and a
self-contained HTML+SVG quality report.

## Components
- Job & Inspection API (`nmrassign serve [--bind <addr>] [--storage <dir>] [--inspect-root <dir>] [--cors-origin <origin>]`):
  - Upload, validate (format, referencing, experiment type), queue, status, results download, and header-only workspace classification (`POST /api/v1/inspect`).
  - **Browser Security & Origin Boundary**: Responses enforce same-origin access by default (`Access-Control-Allow-Origin` is omitted unless the request `Origin` matches `--cors-origin` or same-origin `Host`, and untrusted cross-origin requests return `403 Forbidden`) with `X-Content-Type-Options: nosniff`. JSON `POST` routes require `Content-Type: application/json` (`415 Unsupported Media Type` on `text/plain` or form payloads to block preflight-free browser CSRF). Server-side `"path"` inspection on `POST /api/v1/inspect` is disabled (`403 Forbidden`) unless `--inspect-root <dir>` is explicitly supplied and the canonicalized target resides under `--inspect-root`; arbitrary local folder inspection uses `nmrassign inspect <path>` on the CLI.
- Workers: run pipeline stages 2 to 5 on CPU/GPU.
- Report (`nmrassign report`): per-stage metrics, confidence plots, strip plots
  for checking assignments, structure validation summary.
- Local mode (`nmrassign pipeline`): runs the complete pipeline in a single CLI
  command with no server needed, emitting `peaks.tsv`, `shifts.str`,
  `shifts.tsv`, `result.nef`, and `report.html` into `--out-dir <dir>`:

```bash
nmrassign pipeline \
  --seq path/to/1ubq.fasta.txt \
  --hsqc path/to/HSQC.ucsf \
  --triple HNCACB=path/to/HNCACB.ucsf \
  --triple CBCACONH=path/to/CBCACONH.ucsf \
  --triple HNCO=path/to/HNCO.ucsf \
  --triple HNCACO=path/to/HNCACO.ucsf \
  --solver evo-graph \
  --out-dir ./out_assigned

nmrassign score \
  --pred ./out_assigned/shifts.str \
  --truth path/to/reference.str -v
```

Note: 3D experiments use `--triple EXP=path` (`=` separator), `pipeline`
writes a multi-artifact directory via `--out-dir <dir>` (not `-o`), and file
paths must match exact filesystem case (`HSQC.ucsf`, `HNCACB.ucsf`, etc.) for
case-sensitive Linux filesystems and CI.

## Principles
- Runs fully self-hosted; no data leaves the user's infrastructure unless
  they deploy it publicly.
- Every result reproducible from inputs + version + config.
