# 11. Raw Time-Domain (FID) Processing and Held-Out Real-Data Benchmark

## Provenance & Clean-Room Statement
Original clean-room specification and design. Zero source code, macros, parameter tables, or binaries from NMRPipe, `nmrglue`, TopSpin, VnmrJ, CYANA, FLYA, ARTINA, NMRtist, or CcpNmr were consulted. All mathematical formulations and binary file layouts are derived strictly from peer-reviewed literature and public format documentation:

1. **Quadrature detection & hypercomplex algebra**:
   - States, D. J., Haberkorn, R. A. & Ruben, D. J. (*J. Magn. Reson.* **1982**, 48, 286–292) — hypercomplex cosine/sine indirect quadrature (`States`).
   - Marion, D., Ikura, M., Tschudin, R. & Bax, A. (*J. Magn. Reson.* **1989**, 85, 393–399) — `States-TPPI` $180^\circ$ odd-increment phase alternation moving axial peaks to the Nyquist edge.
   - Palmer, A. G. III, Cavanagh, J., Wright, P. E. & Rance, M. (*J. Magn. Reson.* **1991**, 93, 151–170); Kay, L. E., Keifer, P. & Saarinen, T. (*J. Am. Chem. Soc.* **1992**, 114, 10663–10665) — gradient sensitivity-enhanced P/N-type `Echo-Antiecho` (`Rance-Kay`) recombination.
2. **Digital oversampling & group delay**:
   - Moskau, D. (*Concepts Magn. Reson.* **2002**, 15, 164–176) — causal FIR decimation group delay (`GRPDLY`) in Bruker digital receivers.
   - Westler, W. M. & Abildgaard, F. (**1996**, NMRFAM public technical note) — legacy Bruker `(DSPFVS, DECIM)` group-delay lookup table for `DSPFVS` 10–12 when `GRPDLY` is absent or `-1`.
3. **Time-domain & frequency-domain signal processing**:
   - Otting, G., Widmer, H., Wagner, G. & Wüthrich, K. (*J. Magn. Reson.* **1986**, 66, 187–193) — first-point multiplication (`FCOR = 0.5` when $p_1 = 0^\circ$) to suppress discrete Fourier transform baseline offset.
   - Marion, D., Ikura, M. & Bax, A. (*J. Magn. Reson.* **1989**, 84, 425–430) — time-domain low-frequency Gaussian/sine-bell convolution solvent subtraction.
   - Zhu, G. & Bax, A. (*J. Magn. Reson.* **1990**, 90, 405–410) — mirror-image linear prediction (`MILP`) for constant-time / known-phase indirect interferograms.
   - Hoch, J. C. & Stern, A. S. (*NMR Data Processing*, Wiley-Liss, **1996**) — apodization windows, zero-filling, FFT sign conventions, and polynomial baseline correction.
   - Chen, L., Weng, Z., Goh, L. & Garland, M. (*J. Magn. Reson.* **2002**, 158, 164–168) — first-derivative Shannon entropy minimization with negative-absorption penalty for automatic phase correction.
4. **Chemical-shift referencing & public binary formats**:
   - Wishart, D. S. et al. (*J. Biomol. NMR* **1995**, 6, 135–140) — IUPAC/IUBMB indirect $\Xi$ referencing ratios relative to DSS.
   - Bruker JCAMP-DX v5.0 acquisition/processing parameter files (`acqus`/`acqu2s`/`acqu3s`, `procs`/`proc2s`/`proc3s`) and `fid`/`ser` 1024-byte block-aligned binary layout.
   - Varian/Agilent VnmrJ Data File Reference (`32`-byte big-endian `datafilehead`, `28`-byte `datablockhead`, and ASCII `procpar` parameter dictionary).

---

## 1. Raw FID Readers (`crates/nmr-core/src/fid.rs`)

### 1.1 Unified Hypercomplex Representation
Raw time-domain datasets are decoded into vendor-neutral structs (`Fid1D`, `Fid2D`, `Fid3D`), where every indirect dimension has already been demultiplexed and recombined from its native acquisition quadrature (`States`, `States-TPPI`, or `Echo-Antiecho`) into standard hypercomplex `(cos, sin)` pairs:
- `Fid1D`: `[n0]` complex `f64` samples `(re, im)` + `FidAxisMeta`.
- `Fid2D`: `[n1_cplx * 2 * n0]` — for each indirect increment `k1 in 0..n1_cplx`, two complex direct traces (`cos` at `0`, `sin` at `1`; for `QF`, `sin` is zero).
- `Fid3D`: `[n2_cplx * 2 * n1_cplx * 2 * n0]` — hypercomplex $2 \times 2$ direct traces `(cos2_cos1, cos2_sin1, sin2_cos1, sin2_sin1)` per `(k2, k1)` indirect increment pair, ordered so `dim 0` ($x$) is direct `${}^{1}\text{H}$, `dim 1` ($y$) corresponds to `F2` (`acqu2s` / `ni`), and `dim 2` ($z$) corresponds to `F1` (`acqu3s` / `ni2`).

Each dimension carries `FidAxisMeta`:
- `label`: nucleus string (`"1H"`, `"15N"`, `"13C"`).
- `td_raw`: raw scalar/real-plus-imaginary count from acquisition (`TD` or `np` / `2 * ni`).
- `complex_points`: complex time-domain increments (`td_raw / 2` for hypercomplex/DQD).
- `sw_hz`: spectral width in Hz (`SW * SFO1` on Bruker when `SW > 0 && SFO1 > 0`, else `SW_h`; `sw`/`sw1`/`sw2` on Varian).
- `obs_mhz`: spectrometer observation frequency (`SF` when `procs` referencing is present, else `BF1` on Bruker; `reffrq` or `sfrq`/`dfrq`/`dfrq2` on Varian).
- `carrier_mhz`: transmitter carrier frequency (`SFO1` on Bruker; `sfrq`/`dfrq`/`dfrq2` on Varian).
- `first_ppm`: downfield (index `0`) chemical shift in ppm.
- `quad_mode`: `QuadMode` (`Dqd`, `Qf`, `States`, `StatesTppi`, `EchoAntiecho`).
- `group_delay`: digital filter group delay in complex points (`0.0` on indirect axes).
- `stored_proc`: optional `StoredDimProc` extracted from Bruker `procs`/`proc2s`/`proc3s` or Varian `procpar` (`p0_deg`, `p1_deg`, `window`, `si`, `td_eff`, `fcor`, `reverse`).

### 1.2 Bruker `fid` / `ser` Reader
1. **Parameter resolution**:
   - Reads `acqus` (merging fallback keys from `acqu` when `acqu2s`/`acqu3s` write minimal TopSpin 2.1 headers), plus `acqu2s`/`acqu2` (2D/3D) and `acqu3s`/`acqu3` (3D).
   - Optionally reads `pdata/1/procs`, `proc2s`, `proc3s` (if present) strictly to populate `stored_proc` defaults and calibrated reference frequency `SF`.
2. **Binary decoding & 256-word block padding**:
   - `BYTORDA`: `0` = little-endian, `1` = big-endian.
   - `DTYPA`: `0` = 32-bit signed integer (`i32`, 4 bytes), `1` = 32-bit IEEE float (`f32`, 4 bytes), `2` = 64-bit IEEE float (`f64`, 8 bytes). Any other `DTYPA` returns an error.
   - In 2D/3D `ser` files, Bruker pads every acquired 1D FID trace of `TD` scalar words (`TD * word_bytes`) to a multiple of **1024 bytes** (`256` 32-bit words or `128` 64-bit words):
     $$\text{stride\_bytes} = \left\lceil \frac{\text{TD} \cdot \text{word\_bytes}}{1024} \right\rceil \times 1024$$
     Only the first `TD` words (`TD / 2` complex `(re, im)` pairs) of each block are retained; trailing alignment bytes are discarded.
   - In 3D `ser` files, `AQSEQ` determines loop nesting: `AQSEQ = 0` (`3-2-1`, `F2` inner / `F1` outer) vs `AQSEQ = 1` (`3-1-2`, `F1` inner / `F2` outer). Unrecognized `AQSEQ` values return an explicit error.
3. **Digital filter group delay**:
   - If `GRPDLY > 0.0` in `acqus`, `group_delay = GRPDLY`.
   - Otherwise, if `DECIM > 1` and `DSPFVS > 0`, looks up `(DSPFVS, DECIM)` in the published Westler/Moskau table (`DSPFVS = 10, 11, 12`, `DECIM = 2..2048`; plus `DSPFVS = 13` `DECIM = 2..64`). Unknown `(DSPFVS, DECIM)` pairs with `DECIM > 1` and missing `GRPDLY` return an explicit error rather than guessing.
4. **Quadrature modes (`FnMODE`)**:
   - Direct dimension (`AQ_mod`): `1` (`QF`), `3` (`DQD` simultaneous/digital quadrature), `4` (`QSIM`). Sequential `QSEQ` (`2`) or unknown modes return an explicit error.
   - Indirect dimensions (`FnMODE`):
     - `1` -> `QF` (magnitude / single trace per increment).
     - `4` -> `States` ($z_{\cos} = z_{2k}$, $z_{\sin} = z_{2k+1}$).
     - `5` -> `States-TPPI` ($z_{\cos} = (-1)^k z_{2k}$, $z_{\sin} = (-1)^k z_{2k+1}$).
     - `6` -> `Echo-Antiecho` ($z_{\cos} = z_{2k} + z_{2k+1}$, $z_{\sin} = i(z_{2k} - z_{2k+1})$, matching TopSpin `MC2 = 5` / Rance-Kay sensitivity enhancement; if `EA` gradient polarity is inverted via `FnMODE = 6` + `MC2`, handled deterministically).
     - `0` (`Undefined`), `2` (`QSEQ`), `3` (`TPPI`), or any other code returns an explicit `unsupported Bruker FnMODE` error. Never guess.
5. **Chemical-shift axis (`AxisPpm`)**:
   - With `sf_mhz = procs.SF.unwrap_or(acqu.BF1)` and `sw_hz = acqu.SW * acqu.SFO1`:
     $$\text{first\_ppm} = \frac{(\text{SFO1} - \text{sf\_mhz}) \times 10^6 + 0.5\,\text{sw\_hz}}{\text{sf\_mhz}}$$
     When `procs.OFFSET` is present and `SREF_mod != 0`, `procs.OFFSET` agrees with this formula to $< 10^{-5}\text{ ppm}$ and is used when `procs` referencing was manually calibrated (`COROFFS != 0`).

### 1.3 Varian / Agilent `fid` + `procpar` Reader
1. **Binary `fid` structure**:
   - Global 32-byte big-endian `datafilehead`: `nblocks`, `ntraces`, `np`, `ebytes`, `tbytes`, `bbytes`, `vers_id`, `status`, `nbheaders`.
   - Status flags: `S_DATA (0x01)`, `S_SPEC (0x02)` (rejected if `!= 0`, since frequency-domain data is not a raw FID), `S_32 (0x04)`, `S_FLOAT (0x08)`.
   - Each block begins with `nbheaders * 28` bytes of `datablockhead` (`scale`, `status`, `index`, `mode`, `ctcount`, `lpval`, `rpval`, `lvl`, `tlt`), followed by `ntraces` traces of `np / 2` big-endian `(re, im)` pairs (`f32` if `S_FLOAT`, `i32` if `S_32`, `i16` if `ebytes == 2`).
2. **ASCII `procpar` parser**:
   - Parses multi-line parameter records (`name subtype basictype ...` header line, count/value line(s) supporting quoted strings and multi-line arrays, and trailing enum specification line).
   - Extracts `np`, `sw`, `sfrq`, `tof`, `tn`, `rfl`, `rfp`, `reffrq`, `rp`, `lp`; indirect 1 (`ni`, `sw1`, `dfrq`/`dfrq2`, `dof`/`dof2`, `dn`/`dn2`, `phase`, `rfl1`, `rfp1`, `reffrq1`, `rp1`, `lp1`, `fcoef`/`f1coef`/`f2coef`); indirect 2 (`ni2`, `sw2`, `phase2`, `rfl2`, `rfp2`, `reffrq2`, `rp2`, `lp2`); and `array` loop order (`"phase,phase2"` vs `"phase2,phase"`).
   - Maps indirect nuclei to channels via BioPack conventions (`dn` = `C13` -> `dfrq`, `dn2` = `N15` -> `dfrq2`, matching `sw1`/`sw2` spectral ranges and explicit `axis` string such as `'pd2'` or `'p1d'` when present in `procpar`).
3. **Quadrature & `fcoef` handling**:
   - Requires `phase = [1, 2]` (or `[1]` for `QF`) and `phase2 = [1, 2]` in 3D. Any other `phase`/`phase2` array (e.g. TPPI `phase = 3`, `1,2,3`, missing `phase` when `ni > 1`) returns an explicit error.
   - When `fcoef` / `f1coef` / `f2coef` (8 floats $c_0 \dots c_7$) is present in `procpar`, applies the exact VnmrJ hypercomplex linear combination:
     $$z_{\cos} = (c_0 - i c_1)\, z_1 + (c_2 - i c_3)\, z_2, \qquad z_{\sin} = (c_4 - i c_5)\, z_1 + (c_6 - i c_7)\, z_2$$
     Otherwise uses `States` (or `Echo-Antiecho` when `etype`/`SE`/`gradtype` and Rance-Kay BioPack flags are explicitly specified, returning an error on ambiguous configurations).
4. **Chemical-shift axis**:
   - When `reffrq > 0` and `(rfl != 0 || rfp != 0)`:
     $$\text{first\_ppm} = \frac{\text{sw} - \text{rfl} + \text{rfp}}{\text{reffrq}}$$
   - Otherwise uses IUPAC indirect $\Xi$ referencing (Wishart et al. 1995) anchored to the `${}^{1}\text{H}$ water carrier (`sfrq` at temperature-compensated $\delta_{\text{H}_2\text{O}}(T)$, `temp` from `procpar`, default $25\,^\circ\text{C} \to 4.766\text{ ppm}$):
     $$\nu_0({}^{1}\text{H}) = \frac{\text{sfrq}}{1 + \delta_{\text{H}_2\text{O}} \times 10^{-6}}, \quad \nu_0(X) = \Xi_X \, \nu_0({}^{1}\text{H}), \quad \text{first\_ppm}_X = \frac{(\nu_{\text{carrier},X} - \nu_0(X)) \times 10^6 + 0.5\,\text{sw}_X}{\nu_0(X)}$$

### 1.4 Non-Uniform Sampling (NUS) Detection & Rejection
NUS reconstruction is out of scope. Both readers inspect acquisition metadata before reading signal blocks and return an explicit error (`Err("NUS (non-uniform sampling) detected: ...")`) if any of the following hold:
- Bruker: `FnTYPE == 2` (or `> 0` non-uniform mode), `NusAMOUNT > 0` with `FnTYPE != 0`, presence of a non-empty `nuslist` schedule file in the experiment folder (when `FnTYPE != 0` or `ser` size is smaller than uniform `td1 * td2 * stride_bytes`), or `ser` byte length smaller than the uniform grid size.
- Varian: `ni` / `ni2` paired with `sampling = 'sparse'` (`sparse = 'y'`), presence of `sampling.sch` / `nuslist`, or `nblocks * ntraces` smaller than the uniform hypercomplex product.

---

## 2. Deterministic Processing Pipeline (`crates/nmr-core/src/process.rs`)

All settings are captured in a serializable `ProcessingRecipe` (`serde` JSON) written as `processing_recipe.json` next to every processed output.

### 2.1 Per-Dimension Steps (Executed Strictly in Order)
For each dimension (`dim 0` direct `${}^{1}\text{H}$ first, then `dim 1` indirect, then `dim 2` indirect):

1. **Remove digital filter** (direct dimension when `group_delay > 0`):
   - Let $G = \text{group\_delay}$, $m = \text{round}(G)$, and residual $\delta G = G - m \in [-0.5, 0.5]$.
   - Left-shift the complex FID by $m$ samples ($z'[n] = z[n + m]$ for $0 \le n < M - m$, zeroing the last $m$ samples, or truncating to `td_eff` before shifting when `td_eff < M`).
   - Apply the sub-sample residual group delay $\delta G$ as an exact linear-phase term during step 5/6 ($\Delta p_1 = -360^\circ \cdot \delta G$) so $t = 0$ aligns exactly with index $0$ prior to solvent filtering and apodization.
2. **Optional solvent filter** (direct `${}^{1}\text{H}$ dimension):
   - Off or on via `SolventFilterConfig`: time-domain low-pass Gaussian convolution subtraction (Marion, Ikura & Bax 1989):
     $$z_{\text{clean}}[n] = z[n] - \frac{\sum_{j} w(n - j)\, z[j]}{\sum_{j} w(n - j)}, \qquad w(\Delta) = \exp\!\left(-\frac{\Delta^2}{2 \sigma_w^2}\right)$$
     with boundary reflection/normalization at $n = 0$ (`half_width` default `32` complex points).
3. **Apodization**:
   - Applied over active time-domain length $N_{\text{apod}} = \min(M_{\text{valid}}, \text{td\_eff})$:
     - `SineBell { offset_pi_frac }`: $w[n] = \sin\bigl(\pi[\alpha + (1 - \alpha)\, n / N_{\text{apod}}]\bigr)$, where $\alpha = 1/\text{SSB}$ for Bruker `SSB >= 2` (`0.5` for pure cosine bell `SSB = 2`, `0.0` for unshifted sine bell).
     - `SquaredSineBell { offset_pi_frac }`: $w[n] = \sin^2\bigl(\pi[\alpha + (1 - \alpha)\, n / N_{\text{apod}}]\bigr)$ (`SSB = 2` -> pure $\cos^2$ bell).
     - `Exponential { lb_hz }`: $w[n] = \exp(-\pi \cdot \text{lb\_hz} \cdot n / \text{sw\_hz})$.
     - `None`: rectangular window ($w[n] = 1$).
   - First-point scaling: $z[0] \leftarrow \text{fcor} \cdot z[0]$ (`default fcor = 0.5` when $|p_1| < 1^\circ$, else `1.0`).
   - **Optional indirect Mirror-Image Linear Prediction (`MILP`)**: gated behind `linear_prediction: bool` (**`false` by default**). When explicitly enabled (`--lp`), extends indirect interferograms of known phase (`p0 = 0, p1 = 0` or `-90, +180`) using Burg/SVD autoregressive reflection (Zhu & Bax 1990) prior to apodization.
4. **Zero-fill**:
   - Pads complex vector with zeros to `si` (`power of two >= N_apod`; defaults to `stored_proc.si` if present, else `next_power_of_two(2 * N_apod)`).
5. **FFT**:
   - Clean-room in-place radix-2 Cooley–Tukey complex `f64` FFT (with Bluestein chirp-z fallback for any non-power-of-two size), followed by Nyquist `fftshift` and frequency-sign ordering so index `0` is the downfield (`high_ppm`) edge and index `si - 1` is the upfield (`low_ppm`) edge (`step_ppm = -sw_ppm / si`), matching `AxisPpm` in `nmr_core::spectrum`.
6. **Phase correction (`p0` / `p1`)**:
   - Multiplies bin $k \in [0, N_{\text{fft}})$ by $\exp\bigl(i\,\frac{\pi}{180}[p_0 + p_1 \cdot k / N_{\text{fft}}]\bigr)$.
   - **Phase source priority**:
     1. Explicit recipe override.
     2. Stored phases in Bruker `procs`/`proc2s`/`proc3s` (`PHC0`, `PHC1`) or Varian `procpar` (`rp`, `lp`).
     3. **Automatic phase correction** (invoked **only** when neither recipe nor parameter files provide phases; on indirect axes without stored phases where acquisition starts at $t_1 = 0$, defaults to `(0°, 0°)` unless `--auto-phase-indirect` is requested; on direct `${}^{1}\text{H}$ without stored phases, minimizes Chen et al. 2002 first-derivative Shannon entropy + negative-intensity penalty):
        $$E(p_0, p_1) = -\sum_{k=0}^{N-2} h_k \ln h_k + \gamma \sum_{k=0}^{N-1} \max(0, -R_k)^2, \qquad h_k = \frac{|R_{k+1} - R_k|}{\sum_j |R_{j+1} - R_j|}$$
        evaluated on the amide region of the summed high-energy 1D traces via $5^\circ$ coarse scan in $p_0 \in [-180^\circ, 180^\circ)$ followed by coordinate refinement of $(p_0, p_1)$.
7. **Discard imaginary parts**:
   - Retains `Re(S[k])` only, halving memory before transforming subsequent indirect dimensions.
8. **Optional polynomial baseline correction**:
   - Iterative MAD-masked Legendre polynomial fit (`order = 0..=5`, default enabled on direct `${}^{1}\text{H}$ with `order = 1` or `qfil` carrier suppression when outside cropped amide window, configurable via `BaselineConfig`).
9. **Extract region of interest (ROI)**:
   - Slices axis `[idx_start..=idx_end]` covering `[roi_min_ppm, roi_max_ppm]` and updates `AxisPpm { size, first_ppm, step_ppm, obs_mhz }` exactly (`first_ppm_new = axis.index_to_ppm(idx_start as f64)`).
   - Default for direct `${}^{1}\text{H}$: `5.5..=11.5 ppm` (`--crop-h 5.5:11.5`, disable with `--no-crop` for full-sweep comparison against uncropped TopSpin `2rr`/`3rrr`).

---

## 3. CLI & Interchange Integration (`crates/nmr-cli`)

1. `nmrassign process --in <fid_dir> --out <out_path_or_dir> [--exp <EXP>] [--recipe <json>] [--no-crop] [--lp]`:
   - Processes a 2D or 3D Bruker/Varian raw FID directory into NMRPipe (`.ft2` / `.ft3`) or Bruker `pdata` directory (readable unchanged by `nmr_core::spectrum`) and writes `processing_recipe.json` alongside it.
2. `--fid` support in `pick` and `pipeline`:
   - `nmrassign pick --fid <dir> --exp <EXP> -o peaks.tsv`
   - `nmrassign pipeline --sequence seq.fasta --fid-hsqc <dir> --fid-triple EXP=<dir> ... --out-dir <dir>` (processes FIDs in memory using `ProcessingRecipe`, writes `processing_recipe.json` to `--out-dir`, and runs the existing picker + solver unchanged).

---

## 4. TopSpin Validation Protocol (Part 3 Gate)

Evaluated on the 5 local Ubiquitin raw `ser` vs TopSpin `pdata/1/{2rr,3rrr}` datasets (`Ubiquitin_090309/{2:HSQC, 3:HNCACB, 4:CBCACONH}`, `Ubiquitin_111909/{10:HNCO, 11:HNCACO}`):
1. Process each raw `ser` using `WDW`/`SSB`/`PHC0`/`PHC1`/`SI`/`TDeff` from `procs`/`proc2s`/`proc3s` (without `3rrr` binary data).
2. Compare picked peaks on our processed spectrum vs TopSpin's `2rr`/`3rrr`:
   - Median and max per-axis ppm error (`|dH|`, `|dN|`, `|dC|`) across matched peaks.
   - Pearson correlation $r$ of normalized peak heights across matched peaks.
   - Matched-peak recall relative to TopSpin `pdata` peaks.
3. Run end-to-end `pipeline` (`--solver evo-graph`) on the raw-FID-processed spectra and score against `data/ubiquitin_backbone_assignment_191104.txt` (global per-atom offset only). Gate: within $\pm 2.0$ percentage points of TopSpin `pdata` (`94.0%` accuracy / `90.0%` recall), with **zero** picker or solver tuning.

---

## 5. Public Archive Fetcher & Frozen Real-Protein Benchmark (Parts 4 & 5)

### 5.1 Fetcher (`nmrassign fetch-bmrb`)
- `nmrassign fetch-bmrb --id <bmrb_id> --dest data/private/bmrb/<id>/`:
  - Queries BMRB REST/FTP (`https://api.bmrb.io/v2/entry/<id>` / `https://bmrb.io/ftp/pub/bmrb/biomolecules/bmr<id>/` and `https://bmrb.io/ftp/pub/bmrb/timedomain/bmr<id>/`), enforcing `User-Agent: nmrassign-fetch` and $\le 1\text{ request/s}$ rate limiting.
  - Downloads the NMR-STAR entry (`bmr<id>_3.str`) and time-domain archive(s), extracts archives safely (preventing path traversal), and writes `PROVENANCE.md` with source URLs, CC0/BMRB public-domain license notice, UTC download timestamp, byte sizes, and SHA-256 digests of every downloaded file.

### 5.2 Frozen Manifest (`data/benchmark/real_manifest.tsv`)
Columns: `bmrb_id`, `protein`, `residues`, `field_mhz`, `vendor`, `experiments`, `split`.
- `split` is frozen as `dev` (~1/3, including `ubiquitin_local` and `51439` `CISD3`) or `heldout` (~2/3).
- Selection criteria for new entries (10–15 total): 50–150 residues, uniform sampling, `HSQC` + `HNCO` + (`HNCACB` or `HNCA`) + (`CBCACONH` or `HNCOCA`), $\ge 80\%$ deposited backbone completeness, spanning $\alpha$, $\beta$, and $\alpha+\beta$ folds, multiple field strengths, and both Bruker and Varian/Agilent spectrometers.

### 5.3 Runner (`nmrassign real-benchmark`)
- `nmrassign real-benchmark --manifest data/benchmark/real_manifest.tsv --split dev|heldout [--solver standard|evo-graph] [--seeds 5] [--json-out results.json]`:
  - Discovers raw FID experiments in `data/private/bmrb/<id>/` (and existing local dev paths for Ubiquitin / CISD3), runs `process -> pick -> assign -> score` using unlabelled global per-atom reference offset fitting (`fit_unlabelled_global_offsets`), and prints per-entry metrics, macro/micro averages, skipped missing-data entries (non-fatal for CI), and exact failure diagnostics for any unreadable/NUS/unsupported dataset.
  - **Strict holdout governance**: `heldout` is evaluated once for reporting; no code, default, or threshold may be modified in response to `heldout` output.
