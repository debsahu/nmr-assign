# 02. Peak picking (Stage A working)

Goal: processed spectra in, peak lists (01 TSV) out, for HSQC and backbone
triple-resonance spectra first; NOESY later.

## Stage A: classical baseline (`crates/nmr-core/src/picker.rs` — working)
1. Noise estimate per spectrum: two-pass outlier-trimmed MAD (`1.4826 * median(|x - median(x)|)`) on empty regions.
2. Spectral window gating:
   - `pick_hsqc_2d`: gates backbone `15N`-HSQC roots to the amide `1H` window (`5.5..=11.5 ppm` by default, configurable via `PickerConfig::amide_h_window`) when the `1H` sweep width extends below `4.5 ppm` (suppressing residual water ridges and aliphatic `1H` artifacts). `pick_c13_hsqc_2d` disables the amide `1H` gate (`amide_h_window = None`) so aliphatic (`0..5.5 ppm`) and aromatic `13C`-HSQC peaks are preserved.
   - `pick_triple_3d`: gates `13C` shifts by experiment (`HNCACB`, `CBCACONH`, `HNCA`, `HNCOCA`: `8.0..=76.0 ppm`; `HNCO`, `HNCACO`: `164.0..=187.0 ppm`).
3. Indirect-axis `t1` ridge and FT truncation sinc-sidelobe suppression:
   - In high-dynamic-range spectra (`max_abs > 40 * global_mad`), computes two-pass outlier-trimmed local column/row/strip MAD along indirect axes and requires `|v| >= max(snr_threshold * global_mad, ridge_snr * local_mad, sinc_frac * max_abs_in_col_or_strip)`, followed by Lorentzian-decaying near-parent sinc sidelobe suppression along F1/F2/F3.
   - Single-sign backbone 3D experiments (`HNCA`, `HNCOCA`, `CBCACONH`, `HNCO`, `HNCACO`) automatically detect `180°` inverted phase polarity (`max_neg > 2.5 * max(max_pos, threshold)`) and emit positive `sign = 1` peak heights.
   - High-resolution `15N`-HSQC spectra (`|step_ppm| <= 0.008`) coalesce resolved symmetric `^3J_HNHa` `1H` doublets (`|dN| <= 0.08 ppm`, `|dH| <= 0.019 ppm`, `min_height >= 0.40 * max_height`) into intensity-weighted centroids.
4. Local extrema detection with phase sign (`+1` / `-1` for phase-sensitive `HNCACB`), sub-voxel 3-point log-Gaussian / parabolic centre refinement along every axis, and `(H, N)` strip restriction under HSQC roots.
5. Asn/Gln sidechain `NH2` paired `1H` doublets (`|dN| <= 0.12 ppm` and `0.30 <= |dH| <= 1.20 ppm`) lacking credible intra-residue `CA(i)` support are rejected during spin-system assembly.

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
