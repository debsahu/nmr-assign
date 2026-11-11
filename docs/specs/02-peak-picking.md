# 02. Peak picking (Stage A working)

Goal: processed spectra in, peak lists (01 TSV) out, for HSQC and backbone
triple-resonance spectra first; NOESY later.

## Stage A: classical baseline (`crates/nmr-core/src/picker.rs` — working)
1. Noise estimate per spectrum: two-pass outlier-trimmed MAD (`1.4826 * median(|x - median(x)|)`) on empty regions.
2. Local extrema above `k * noise` (`k` tunable, default `5.0`), with phase sign (`+1` / `-1`) for phase-sensitive experiments (`HNCACB`).
3. Sub-voxel 3-point log-Gaussian / parabolic centre refinement along every axis.
4. Restricts 3D triple-resonance picking to `(H, N)` strips under HSQC roots.

## Stage B: neural picker (planned)
- 3D convolutional network on `wgpu` trained on synthetic spectra rendered from BMRB
  shift tables via `nmr_core::simulate` (`render_hsqc_2d`, `render_triple_3d`):
  mixed Lorentzian/Gaussian lineshapes, noise, t1 ridges, truncation artefacts, overlap, missing and weak peaks.
- Output: peak probability map plus sub-voxel offsets.
- Clean-room: DEEP Picker (GPL) may be read as a paper only, not as code.

## Evaluation
- Ground truth: expected peak positions predicted from the reference assignment (05).
- Metrics: recall, precision, position error per dimension, and downstream assignment accuracy (03).

## Open questions (maintainer)
- Which spectra are available with the completed assignment, and processed how (Bruker `pdata`, NMRPipe)?
- Acceptable false-positive rate given the assignment engine tolerates spurious peaks?
