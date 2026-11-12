# 02. Peak picking (Stages A & A2 working)

Goal: processed spectra in, peak lists (01 TSV) out, for HSQC and backbone
triple-resonance spectra first; NOESY later.

## Stage A: classical Pass-1 baseline (`crates/nmr-core/src/picker.rs` — working)
1. Noise estimate per spectrum: two-pass outlier-trimmed MAD (`1.4826 * median(|x - median(x)|)`) on empty regions.
2. Spectral window gating:
   - `pick_hsqc_2d`: gates backbone `15N`-HSQC roots to the amide `1H` window (`5.5..=11.5 ppm` by default, configurable via `PickerConfig::amide_h_window`) when the `1H` sweep width extends below `4.5 ppm` (suppressing residual water ridges and aliphatic `1H` artifacts). `pick_c13_hsqc_2d` disables the amide `1H` gate (`amide_h_window = None`) so aliphatic (`0..5.5 ppm`) and aromatic `13C`-HSQC peaks are preserved.
   - `pick_triple_3d`: gates `13C` shifts by experiment (`HNCACB`, `CBCACONH`, `HNCA`, `HNCOCA`: `8.0..=76.0 ppm`; `HNCO`, `HNCACO`: `164.0..=187.0 ppm`).
3. Indirect-axis `t1` ridge and FT truncation sinc-sidelobe suppression:
   - In high-dynamic-range spectra (`max_abs > 40 * global_mad`), computes two-pass outlier-trimmed local column/row/strip MAD along indirect axes and requires `|v| >= max(snr_threshold * global_mad, ridge_snr * local_mad, sinc_frac * max_abs_in_col_or_strip)`, followed by Lorentzian-decaying near-parent sinc sidelobe suppression along F1/F2/F3.
   - Single-sign backbone 3D experiments (`HNCA`, `HNCOCA`, `CBCACONH`, `HNCO`, `HNCACO`) automatically detect `180°` inverted phase polarity (`max_neg > 2.5 * max(max_pos, threshold)`) and emit positive `sign = 1` peak heights.
   - High-resolution `15N`-HSQC spectra (`|step_ppm| * obs_mhz <= 6.5 Hz`) coalesce resolved symmetric `^3J_HNHa` `1H` doublets (`|dN| <= 0.08 ppm`, `|dH| * obs_mhz <= max_j_hnha_hz` with default `9.5 Hz`, `min_height >= 0.40 * max_height`) into intensity-weighted centroids using the `1H` axis spectrometer frequency `obs_mhz`.
4. Local extrema detection with phase sign (`+1` / `-1` for phase-sensitive `HNCACB`), sub-voxel 3-point log-Gaussian / parabolic centre refinement along every axis, and `(H, N)` strip restriction under HSQC roots.
5. Asn/Gln sidechain `NH2` paired `1H` doublets (`|dN| <= 0.12 ppm` and `0.30 <= |dH| <= 1.20 ppm`) lacking credible intra-residue `CA(i)` support are rejected during spin-system assembly.

## Stage A2: Pass-2 experiment-corroborated 3D strip rescue (`StripBank3D` + `corroborated_re_pick_3d` — working)
Provenance: Original clean-room two-pass experiment-corroborated 3D strip rescue exploiting complementary triple-resonance coherence-transfer pathways (`HNCA`/`HN(CO)CA`, `HNCACB`/`CBCA(CO)NH`, `HNCO`/`HN(CA)CO`; Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990, **89**, 496–514; Grzesiek & Bax, *J. Am. Chem. Soc.* 1992, **114**, 6291–6293).

1. **Scientific guard against confirmation bias**:
   - Pass 2 (`corroborated_re_pick_3d` / `pick_triple_3d_two_pass`) accepts **no amino-acid sequence and no chemical-shift prior inputs**.
   - Pass 1 (`pick_hsqc_2d` + `pick_triple_3d`) runs untouched at `>= 5.0` global / `>= 5.5` local-strip MAD.
   - Pass 2 inspects raw 3D strips at a conservative **`3.8–4.2 sigma` local-strip MAD floor** (`CorroboratedRepickConfig`, `strip_mad >= global_mad`) **strictly inside narrow `+-0.15 ppm` (`CA`/`CB`) or `+-0.12 ppm` (`CO`) windows pinned by an already-observed `>= 5.5 sigma` Pass-1 experimental peak**, enforcing identical 26-neighbor `3x3x3` local extremum checks, 3-axis sub-voxel log-Gaussian/parabolic refinement, `3x3x3` signed volume integration, tight `(H, N)` root centering (`((dH / 0.028)^2 + (dN / 0.28)^2)^(1/2) <= 1.0`), `t1`-ridge / far-sinc floors, and near-parent F1/F2/cross-sign `HNCACB` sinc-sidelobe rejection.
   - Paired sidechain `NH2` doublets (`|dN| <= 0.12 ppm`, `0.30 <= |dH| <= 1.20 ppm`) and upfield non-amide roots (`1H < 6.0 ppm`) are excluded from pool donation and cross-strip rescue so spurious decoy roots are never promoted.
2. **Memory-lean streaming `StripBank3D`**:
   - Callers loading one 3D cube at a time (`StripBank3D::pick_and_record_3d`) extract a compact `StripCube3D` containing only the active `(H, N)` strip columns plus a 1-voxel `(dx, dn)` halo cropped to the active `13C` experiment window (`~0.5–8 MB` per cube vs `130–540 MB` full `Spectrum3D`), preserving bit-exact `3x3x3` stencil access and full-column outlier-trimmed `(strip_mad, strip_max_abs)`.
3. **Corroborated rescue pathways**:
   - **Intra-strip cross-experiment rescue (`>= 3.8 sigma` local MAD)**:
     * `CBCA(CO)NH` (`>= 5.5 sigma` at `c0`) -> `HNCACB` (`+CA` / `-CB` typing within `c0 +- 0.15 ppm`), and -> `HNCA` (`+1`) when `HN(CO)CA` is absent or the `HNCA` strip is empty.
     * `HN(CO)CA` (`>= 5.5 sigma` at `c0`) -> `HNCACB (+1)` within `c0 +- 0.15 ppm`, and -> `HNCA (+1)` when the `HNCA` strip has `0` peaks.
     * `HNCA` / `HNCACB (+1)` (`>= 5.5 sigma` at `c0`) -> `HN(CO)CA (+1)` at `>= 4.0 sigma` when the strip has `0` `HN(CO)CA` peaks.
     * `HNCO` (`>= 5.5 sigma` at `c0`) -> `HN(CA)CO (+1)` within `c0 +- 0.12 ppm` (corroborating `C'(i-1)`), and `HN(CA)CO` (`>= 5.5 sigma`) -> `HNCO (+1)` when the strip has `0` `HNCO` peaks.
   - **Cross-strip sequential pool rescue (`>= 4.2 sigma` local MAD, `0`-peak channels only)**:
     * Collects global Pass-1 `>= 5.5 sigma` `CA(i-1)` (`HN(CO)CA`, `CA`-range `CBCA(CO)NH`) and `C'(i-1)` (`HNCO`) shift pools.
     * Strictly on multi-experiment validated backbone strips (`>= 2` distinct Pass-1 backbone experiments at `>= 5.5 sigma`), rescues at most the single strongest local 3D extremum on completely empty (`0`-peak) `HNCA`/`HNCACB(+)` (`c_cand +- 0.15 ppm`) or `HNCO` (`c_cand +- 0.12 ppm`) strips, without splitting true `CA(i) ≈ CA(i-1)` degeneracies on single-peak strips.

## Stage B: neural picker (working / iterating in `neural_picker.rs`)
- 3D convolutional network trained on synthetic spectra rendered from BMRB
  shift tables via `nmr_core::simulate` (`render_hsqc_2d`, `render_triple_3d`):
  mixed Lorentzian/Gaussian lineshapes, noise, t1 ridges, truncation artefacts, overlap, missing and weak peaks.
- Output: peak probability map plus sub-voxel offsets.
- Clean-room:GPL pickers may be read as papers only, not as code.

## Evaluation
- Ground truth: expected peak positions predicted from the reference assignment (05).
- Metrics: recall, precision, position error per dimension, and downstream assignment accuracy (03).

## Open questions (maintainer)
- Which spectra are available with the completed assignment, and processed how (Bruker `pdata`, NMRPipe)?
- Acceptable false-positive rate given the assignment engine tolerates spurious peaks?
