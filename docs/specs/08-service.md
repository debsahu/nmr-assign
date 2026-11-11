# 08. Service (not started)

Goal: NMRtist-style workflow. A user uploads a sequence and processed
spectra, picks experiment types, and receives peak lists, assignments,
structures, and a report.

## Components (draft)
- Job API: upload, validate (format, referencing, experiment type), queue,
  status, results download.
- Workers: run pipeline stages 2 to 5 on CPU/GPU; resumable per stage.
- Report: per-stage metrics, confidence plots, strip plots for checking
  assignments, structure validation summary.
- Local mode: the same pipeline as a single CLI command, no server needed.

## Principles
- Runs fully self-hosted; no data leaves the user's infrastructure unless
  they deploy it publicly.
- Every result reproducible from inputs + version + config.
