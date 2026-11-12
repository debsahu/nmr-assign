# 11. Raw Time-Domain (FID) Processing and Held-Out Real-Data Benchmark

## Provenance & Clean-Room Statement
Original clean-room specification and design. Zero source code, macros, parameter tables, or binaries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) NMR processing or assignment software were consulted. All mathematical formulations and binary file layouts are derived strictly from peer-reviewed literature and public format documentation:

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
4. **Non-uniform sampling (NUS) compressed-sensing reconstruction**:
   - Stern, A. S., Donoho, D. L. & Hoch, J. C. (*J. Magn. Reson.* **2007**, 188, 295–300) — $L_1$-regularized Iterative Soft Thresholding (`IST`) spectrum reconstruction for non-uniformly sampled NMR data.
   - Holland, D. J., Bostock, M. J., Gladden, L. F. & Nietlispach, D. (*Angew. Chem. Int. Ed.* **2011**, 50, 6548–6551) — compressed-sensing multidimensional protein NMR reconstruction.
   - Hyberts, S. G., Milbradt, A. G., Wagner, A. B., Arthanari, H. & Wagner, G. (*J. Biomol. NMR* **2012**, 52, 315–327) — hypercomplex multidimensional Iterative Soft Thresholding (`IST` / `FISTA`) for 2D and 3D protein NMR.
5. **Chemical-shift referencing & public binary formats**:
   - Wishart, D. S. et al. (*J. Biomol. NMR* **1995**, 6, 135–140) — IUPAC/IUBMB indirect $\Xi$ referencing ratios relative to DSS.
   - Bruker JCAMP-DX v5.0 acquisition/processing parameter files (`acqus`/`acqu2s`/`acqu3s`, `procs`/`proc2s`/`proc3s`, `nuslist`) and `fid`/`ser` 1024-byte block-aligned binary layout.
   - Varian/Agilent VnmrJ Data File Reference (`32`-byte big-endian `datafilehead`, `28`-byte `datablockhead`, ASCII `procpar` parameter dictionary, and `sampling.sch` sparse schedule).

---

## 1. Raw FID Readers (`crates/nmr-core/src/fid.rs`)

### 1.1 Unified Hypercomplex Representation
Raw time-domain datasets are decoded into vendor-neutral structs (`Fid1D`, `Fid2D`, `Fid3D`), where every indirect dimension has already been demultiplexed and recombined from its native acquisition quadrature (`States`, `States-TPPI`, or `Echo-Antiecho`) into standard hypercomplex `(cos, sin)` pairs on the full Nyquist time-domain grid:
- `Fid1D`: `[n0]` complex `f64` samples `(re, im)` + `FidAxisMeta`.
- `Fid2D`: `[n1_cplx * 2 * n0]` — for each indirect increment `k1 in 0..n1_cplx`, two complex direct traces (`cos` at `0`, `sin` at `1`; for `QF`, `sin` is zero), plus `nus_mask: Option<Vec<bool>>` (`None` when uniformly sampled; `Some(mask)` of length `n1_cplx` when non-uniformly sampled, with unsampled increments zero-filled in `data`).
- `Fid3D`: `[n2_cplx * 2 * n1_cplx * 2 * n0]` — hypercomplex $2 \times 2$ direct traces `(cos2_cos1, cos2_sin1, sin2_cos1, sin2_sin1)` per `(k2, k1)` indirect increment pair, ordered so `dim 0` ($x$) is direct `${}^{1}\text{H}$, `dim 1` ($y$) corresponds to `F2` (`acqu2s` / `ni`), and `dim 2` ($z$) corresponds to `F1` (`acqu3s` / `ni2`), plus `nus_mask: Option<Vec<bool>>` (`None` when uniformly sampled; `Some(mask)` of length `n2_cplx * n1_cplx` indexed `k2 * n1_cplx + k1` when non-uniformly sampled, with unsampled increments zero-filled in `data`).

Each dimension carries `FidAxisMeta`:
- `label`: nucleus string (`"1H"`, `"15N"`, `"13C"`).
- `td_raw`: raw scalar/real-plus-imaginary full-grid count (`TD` on uniform Bruker, `NusTD` on NUS Bruker, `np` or `2 * ni` on Varian).
- `complex_points`: complex time-domain increments on the full Nyquist grid (`td_raw / 2` for hypercomplex/DQD, `td_raw` for `QF`).
- `sw_hz`: spectral width in Hz (`SW * SFO1` on Bruker indirect axes when `SW > 0 && SFO1 > 0`, else `SW_h`; `sw`/`sw1`/`sw2` on Varian).
- `obs_mhz`: spectrometer observation frequency (`SF` when `procs` referencing is present, else `BF1` on Bruker; `reffrq` or `sfrq`/`dfrq`/`dfrq2` on Varian).
- `carrier_mhz`: transmitter carrier frequency (`SFO1` on Bruker; `sfrq`/`dfrq`/`dfrq2` on Varian).
- `first_ppm`: downfield (index `0`) chemical shift in ppm of the uncropped FFT grid.
- `quad_mode`: `QuadMode` (`Dqd`, `Qf`, `States`, `StatesTppi`, `EchoAntiecho`).
- `group_delay`: digital filter group delay in complex points (`0.0` on indirect axes).
- `stored_proc`: optional `StoredDimProc` extracted from Bruker `procs`/`proc2s`/`proc3s` or Varian `procpar` (`p0_deg`, `p1_deg`, `window`, `si`, `td_eff`, `fcor`, `reverse`, `bc_mod`, `bcfw_ppm`). When TopSpin strip-transform (`STSI < FTSIZE`, `FTSIZE > 0`) is recorded in `procs`, `stored_proc.si` preserves `FTSIZE.max(SI)` (the full uncropped FFT size) and adjusts `first_ppm = OFFSET + (STSR as f64) * (sw_hz / obs_mhz) / (FTSIZE as f64)` when `STSR > 0`.

### 1.2 Bruker `fid` / `ser` Reader
1. **Parameter resolution**:
   - Reads `acqus` (merging fallback keys from `acqu` when `acqu2s`/`acqu3s` write minimal TopSpin headers), plus `acqu2s`/`acqu2` (2D/3D) and `acqu3s`/`acqu3` (3D).
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
     - `5` -> `States-TPPI` ($z_{\cos} = (-1)^k z_{2k}$, $z_{\sin} = (-1)^k z_{2k+1}$, where $k$ is the 0-based grid coordinate).
     - `6` -> `Echo-Antiecho` ($z_{\cos}^* = -i(z_{2k}^* + z_{2k+1}^*)$, $z_{\sin}^* = -(z_{2k}^* - z_{2k+1}^*)$ under Bruker's conjugate direct-FT convention, i.e. `out_cos = [-(im0 + im1), re0 + re1]` and `out_sin = [re1 - re0, im1 - im0]`, matching TopSpin `MC2 = 5` / Rance-Kay sensitivity enhancement).
     - `0` (`Undefined`), `2` (`QSEQ`), `3` (`TPPI`), or any other code returns an explicit `unsupported Bruker FnMODE` error. Never guess.
5. **Chemical-shift axis (`AxisPpm`)**:
   - With `sf_mhz = procs.SF.unwrap_or(acqu.BF1)` and `sw_hz = acqu.SW * acqu.SFO1`:
     $$\text{first\_ppm} = \frac{(\text{SFO1} - \text{sf\_mhz}) \times 10^6 + 0.5\,\text{sw\_hz}}{\text{sf\_mhz}}$$
     When `procs.OFFSET` is present and `SREF_mod != 0`, `procs.OFFSET` (adjusted by `STSR` when `FTSIZE > SI`) agrees with this formula to $< 10^{-5}\text{ ppm}$ and is used when `procs` referencing was calibrated.

### 1.3 Varian / Agilent `fid` + `procpar` Reader
1. **Binary `fid` structure**:
   - Global 32-byte big-endian `datafilehead`: `nblocks`, `ntraces`, `np`, `ebytes`, `tbytes`, `bbytes`, `vers_id`, `status`, `nbheaders`.
   - Status flags: `S_DATA (0x01)`, `S_SPEC (0x02)` (rejected if `!= 0`, since frequency-domain data is not a raw FID), `S_32 (0x04)`, `S_FLOAT (0x08)`.
   - Each block begins with `nbheaders * 28` bytes of `datablockhead` (`scale`, `status`, `index`, `mode`, `ctcount`, `lpval`, `rpval`, `lvl`, `tlt`), followed by `ntraces` traces of `np / 2` big-endian `(re, im)` pairs (`f32` if `S_FLOAT`, `i32` if `S_32`, `i16` if `ebytes == 2`).
2. **ASCII `procpar` parser**:
   - Parses multi-line parameter records (`name subtype basictype ...` header line, count/value line(s) supporting quoted strings and multi-line arrays, and trailing enum specification line).
   - Extracts `np`, `sw`, `sfrq`, `tof`, `tn`, `rfl`, `rfp`, `reffrq`, `rp`, `lp`; indirect 1 (`ni`, `nimax`, `sw1`, `dfrq`/`dfrq2`, `dof`/`dof2`, `dn`/`dn2`, `phase`, `rfl1`, `rfp1`, `reffrq1`, `rp1`, `lp1`, `fcoef`/`f1coef`/`f2coef`); indirect 2 (`ni2`, `ni2max`, `sw2`, `phase2`, `rfl2`, `rfp2`, `reffrq2`, `rp2`, `lp2`); `sampling` / `sparse`; and `array` loop order (`"phase,phase2"` vs `"phase2,phase"`).
   - Maps indirect nuclei to channels via BioPack conventions (`dn` = `C13` -> `dfrq`, `dn2` = `N15` -> `dfrq2`, matching `sw1`/`sw2` spectral ranges and explicit `axis` string such as `'pd2'` or `'p1d'` when present in `procpar`).
3. **Quadrature & `fcoef` handling**:
   - Requires `phase = [1, 2]` (or `[1]` for `QF`) and `phase2 = [1, 2]` in 3D. Any other `phase`/`phase2` array (e.g. TPPI `phase = 3`, `1,2,3`, missing `phase` when `ni > 1`) returns an explicit error.
   - When `fcoef` / `f1coef` / `f2coef` (8 floats $c_0 \dots c_7$) is present in `procpar`, applies the exact VnmrJ hypercomplex linear combination:
     $$z_{\cos} = (c_0 - i c_1)\, z_1 + (c_2 - i c_3)\, z_2, \qquad z_{\sin} = (c_4 - i c_5)\, z_1 + (c_6 - i c_7)\, z_2$$
     Otherwise uses `States`.
4. **Chemical-shift axis**:
   - When `reffrq > 0` and `(rfl != 0 || rfp != 0)`:
     $$\text{first\_ppm} = \frac{\text{sw} - \text{rfl} + \text{rfp}}{\text{reffrq}}$$
   - Otherwise uses IUPAC indirect $\Xi$ referencing (Wishart et al. 1995) anchored to the `${}^{1}\text{H}$ water carrier (`sfrq` at temperature-compensated $\delta_{\text{H}_2\text{O}}(T)$, `temp` from `procpar`, default $25\,^\circ\text{C} \to 4.766\text{ ppm}$):
     $$\nu_0({}^{1}\text{H}) = \frac{\text{sfrq}}{1 + \delta_{\text{H}_2\text{O}} \times 10^{-6}}, \quad \nu_0(X) = \Xi_X \, \nu_0({}^{1}\text{H}), \quad \text{first\_ppm}_X = \frac{(\nu_{\text{carrier},X} - \nu_0(X)) \times 10^6 + 0.5\,\text{sw}_X}{\nu_0(X)}$$

### 1.4 Non-Uniform Sampling (NUS) Auto-Detection & Schedule Decoding
Both Bruker and Varian non-uniformly sampled (NUS) 2D and 3D datasets are automatically detected and decoded onto the full Nyquist hypercomplex grid (`data`) accompanied by `nus_mask: Some(Vec<bool>)`:

1. **Bruker NUS (`FnTYPE == 2` / `NusTD` / `nuslist`)**:
   - **Auto-detection & TopSpin 4 uniform guard**: Uniform TopSpin 4 acquisitions routinely write `##$FnTYPE= 0`, `##$NUSLIST= <automatic>`, `##$NusAMOUNT= 100`, and `NusTD == TD`. When `FnTYPE == 0` in `acqus`/`acqu2s`/`acqu3s` and `ser` matches the uniform grid size (with no overriding `nuslist` file for a smaller `ser`), the dataset is decoded as uniform (`nus_mask = None`). NUS mode is activated when `FnTYPE == 2` (in `acqus` or indirect `acqu*s`), or when `FnTYPE == 0` is paired with a non-empty schedule file (`nuslist` or named `##$NUSLIST`) and a `ser` smaller than the uniform `NusTD` grid. Any unsupported `FnTYPE` (`!= 0` and `!= 2`, e.g., projection spectroscopy `FnTYPE = 1`) returns an explicit error.
   - **Full-grid sizing & flattened `ser` layout**:
     - In 2D NUS, the full scalar grid size is `acqu2s.NusTD` (falling back to `acqus.TD_INDIRECT[1]` when `NusTD <= 0`, or `acqu2s.TD` when `NusTD` exceeds acquired traces), giving `n1_cplx = td_full_f1 / q1_mult`, while `acqu2s.TD` records total acquired 1D FIDs (`n_sched * q1_mult`).
     - In 3D NUS (e.g., TopSpin 4 `FnTYPE = 2`), TopSpin flattens acquired FIDs into `acqu2s.TD = n_sched * q1_mult * q2_mult` and `acqu3s.TD = 1` (or `acqu2s.TD * acqu3s.TD` total traces), while storing the true full-grid scalar sizes in `acqu2s.NusTD` (`F2`, `n1_cplx = td_full_f2 / q1_mult`) and `acqu3s.NusTD` (`F1`, `n2_cplx = td_full_f1 / q2_mult`), with fallback to `acqus.TD_INDIRECT` (`[td_f3, td_f1, td_f2, ...]`).
   - **Schedule parsing & hypercomplex demultiplexing**:
     - Reads `nuslist` (or explicit `##$NUSLIST` file when not `automatic`/`none`). Missing or empty schedule files when NUS is active return an explicit `Err`.
     - Each non-empty, non-comment line contains 0-based integer grid coordinates: `k1` (`0..n1_cplx`) in 2D, or two integers in 3D ordered by `AQSEQ`: `k1_f2 k2_f1` (`F2` first, `F1` second) when `AQSEQ == 0` (`3-2-1`), or `k2_f1 k1_f2` (`F1` first, `F2` second) when `AQSEQ == 1` (`3-1-2`). Out-of-range coordinates or `n_rows * quad_mult` mismatching the acquired `ser` trace count return an explicit `Err`.
     - Each 3D schedule row consumes `q1_mult * q2_mult` (`4` for hypercomplex $\times$ hypercomplex) consecutive 1024-byte-padded direct traces in `ser`, with the inner acquisition axis varying fastest (`F2` quadrature fastest when `AQSEQ == 0`, `F1` quadrature fastest when `AQSEQ == 1`), and `States-TPPI` alternation `(-1)^{k_1}` / `(-1)^{k_2}` evaluated at the true grid coordinate `(k1, k2)`.
     - If a weighted NUS schedule repeats a coordinate, all repeated hypercomplex observations at that coordinate are averaged.

2. **Varian / Agilent NUS (`sampling = 'sparse'` / `sparse = 'y'` / `sampling.sch`)**:
   - **Auto-detection**: Activated when `procpar` sets `sampling = 'sparse'` or `sparse = 'y'`, or when a non-empty `sampling.sch` (or `nuslist`) schedule file accompanies a `fid` binary with fewer traces than the full uniform grid (`nimax` / `ni` and `ni2max` / `ni2`).
   - **Schedule parsing**: Reads 0-based coordinates from `sampling.sch` (or `nuslist`): `k1` (`0..n1_cplx`) in 2D, or `k1 k2` (`0..n1_cplx`, `0..n2_cplx`) in 3D. Each row consumes `q1_mult` (2D) or `q1_mult * q2_mult` (3D, respecting `array` phase nesting) consecutive traces in `fid`, applying `fcoef`/quadrature recombination and averaging repeated coordinates. Missing, empty, out-of-bounds, or trace-count-inconsistent schedules return an explicit `Err`.

---

## 2. Deterministic Processing Pipeline (`crates/nmr-core/src/process.rs`)

All settings are captured in a serializable `ProcessingRecipe` (`serde` JSON) written as `processing_recipe.json` next to every processed output.

### 2.1 Per-Dimension Steps & NUS Reconstruction (Executed Strictly in Order)
Direct `${}^{1}\text{H}$ (`dim 0`) is processed first (steps 1–9 below). When `nus_mask` is `Some(...)`, indirect compressed-sensing reconstruction (**Iterative Soft Thresholding, `IST` / `FISTA`**) is executed across the indirect time grid immediately after direct `${}^{1}\text{H}$ processing (including imaginary discard and optional `${}^{1}\text{H}$ ROI crop) and before indirect apodization, zero-filling, FFT, phase correction, imaginary discard, and baseline correction:

1. **Remove digital filter** (direct dimension when `group_delay > 0`):
   - Let $G = \text{group\_delay}$, $m = \text{round}(G)$, and residual $\delta G = G - m \in [-0.5, 0.5]$.
   - Left-shift the complex FID by $m$ samples ($z'[n] = z[n + m]$ for $0 \le n < M - m$, zeroing the last $m$ samples, or truncating to `td_eff` before shifting when `td_eff < M`).
   - Apply the sub-sample residual group delay $\delta G$ as an exact linear-phase term during step 5/6 ($\Delta p_1 = -360^\circ \cdot \delta G$) so $t = 0$ aligns exactly with index $0$ prior to solvent filtering and apodization.
2. **Optional solvent filter** (direct `${}^{1}\text{H}$ dimension):
   - Off or on via `SolventFilterConfig` (enabled automatically when `stored_proc.bc_mod == Some(6)` `qfil` carrier suppression is indicated): time-domain low-pass Gaussian convolution subtraction (Marion, Ikura & Bax 1989):
     $$z_{\text{clean}}[n] = z[n] - \frac{\sum_{j} w(n - j)\, z[j]}{\sum_{j} w(n - j)}, \qquad w(\Delta) = \exp\!\left(-\frac{\Delta^2}{2 \sigma_w^2}\right)$$
     with boundary normalization at $n = 0$ (`half_width` default `32` complex points).
3. **Indirect NUS Reconstruction (`IST` / `FISTA`, active iff `nus_mask.is_some()`)**:
   - After direct `${}^{1}\text{H}$ processing + imaginary discard + `${}^{1}\text{H}$ ROI extraction (~500–1024 amide columns), each direct `${}^{1}\text{H}$ column defines either a 1D complex indirect vector $y_{\text{obs}} \in \mathbb{C}^{N_1}$ (`Fid2D`) or a 2D hypercomplex indirect array $Y_{\text{obs}} \in \mathbb{H}^{N_2 \times N_1}$ (`Fid3D`, with 4 real components $(cc, cs, sc, ss)$ per grid point) observed on the sampled support $\Omega = \{ k : \text{nus\_mask}[k] = \text{true} \}$.
   - Solves the $L_1$-regularized inverse problem (Stern, Donoho & Hoch 2007; Holland et al. 2011; Hyberts et al. 2012):
     $$\min_{X} \; \frac{1}{2}\bigl\| \mathcal{P}_\Omega \mathcal{F}^{-1}(X) - Y_{\text{obs}} \bigr\|_2^2 + \lambda \|X\|_1$$
     via clean-room `IST` / Nesterov-accelerated `FISTA` over $K \approx 40$ iterations on a power-of-two zero-padded indirect FFT grid:
     - At iteration $t = 0 \dots K-1$, transforms the current residual-consistent time grid to the indirect frequency domain $X^{(t)} = \mathcal{F}(Z^{(t)})$, applies isotropic complex / quaternion magnitude soft-thresholding
       $$\mathcal{S}_{\tau_t}(v) = v \cdot \max\!\left(0,\; 1 - \frac{\tau_t}{\|v\|_2}\right)$$
       along a geometric threshold schedule $\tau_t$ decreasing from $0.85 \max \|X^{(0)}\|_2$ to $2.5\,\hat{\sigma}_{\text{MAD}}$, transforms back $\tilde{Z}^{(t+1)} = \mathcal{F}^{-1}(\mathcal{S}_{\tau_t}(X^{(t)}))$, and enforces exact time-domain data consistency on sampled coordinates ($Z^{(t+1)}[k] = Y_{\text{obs}}[k]$ for $k \in \Omega$, $Z^{(t+1)}[k] = \tilde{Z}^{(t+1)}[k]$ for $k \notin \Omega$).
     - Executed independently in parallel across direct `${}^{1}\text{H}$ columns (`std::thread::scope` on native targets, sequential on `wasm32`). Once `Z` is reconstructed across `0..n_cplx`, subsequent indirect steps (apodization through baseline correction) run identically to uniform data.
4. **Apodization**:
   - Applied over active time-domain length $N_{\text{apod}} = \min(M_{\text{valid}}, \text{td\_eff})$ (clamping `td_eff` to $\le \text{complex\_points}$ when `linear_prediction == false`):
     - `SineBell { offset_pi_frac }`: $w[n] = \sin\bigl(\pi[\alpha + (1 - \alpha)\, n / N_{\text{apod}}]\bigr)$, where $\alpha = 1/\text{SSB}$ for Bruker `SSB >= 2` (`0.5` for pure cosine bell `SSB = 2`, `0.0` for unshifted sine bell).
     - `SquaredSineBell { offset_pi_frac }`: $w[n] = \sin^2\bigl(\pi[\alpha + (1 - \alpha)\, n / N_{\text{apod}}]\bigr)$ (`SSB = 2` -> pure $\cos^2$ bell).
     - `Exponential { lb_hz }`: $w[n] = \exp(-\pi \cdot \text{lb\_hz} \cdot n / \text{sw\_hz})$.
     - `None`: rectangular window ($w[n] = 1$).
   - First-point scaling: $z[0] \leftarrow \text{fcor} \cdot z[0]$ (`default fcor = 0.5` when $|p_1| < 1^\circ$, else `1.0`).
   - **Optional indirect Mirror-Image Linear Prediction (`MILP`)**: gated behind `linear_prediction: bool` (**`false` by default**). When explicitly enabled (`--lp`), extends indirect interferograms of known phase (`p0 = 0, p1 = 0` or `-90, +180`) using Burg/SVD autoregressive reflection (Zhu & Bax 1990) prior to apodization.
5. **Zero-fill**:
   - Pads complex vector with zeros to `si` (`power of two >= N_apod`; defaults to `stored_proc.si` if present, else `next_power_of_two(2 * N_apod)`).
6. **FFT**:
   - Clean-room in-place radix-2 Cooley–Tukey complex `f64` FFT (with Bluestein chirp-z fallback for any non-power-of-two size), followed by Nyquist `fftshift`, `conjugate_for_fft` sign ordering, and `REVERSE = yes` bin reversal (`(N - k) % N`) when flagged so index `0` is the downfield (`high_ppm`) edge and index `si - 1` is the upfield (`low_ppm`) edge (`step_ppm = -sw_ppm / si`), matching `AxisPpm` in `nmr_core::spectrum`.
7. **Phase correction (`p0` / `p1`)**:
   - Multiplies bin $k \in [0, N_{\text{fft}})$ by $\exp\bigl(i\,\frac{\pi}{180}[p_0 + p_1 \cdot k / N_{\text{fft}}]\bigr)$.
   - **Phase source priority**:
     1. Explicit recipe override.
     2. Stored phases in Bruker `procs`/`proc2s`/`proc3s` (`PHC0`, `PHC1`) or Varian `procpar` (`rp`, `lp`).
     3. **Automatic phase correction** (invoked **only** when neither recipe nor parameter files provide phases; on indirect axes without stored phases where acquisition starts at $t_1 = 0$, defaults to `(0°, 0°)` unless `--auto-phase-indirect` is requested; on direct `${}^{1}\text{H}$ without stored phases, minimizes Chen et al. 2002 first-derivative Shannon entropy + negative-intensity penalty):
        $$E(p_0, p_1) = -\sum_{k=0}^{N-2} h_k \ln h_k + \gamma \sum_{k=0}^{N-1} \max(0, -R_k)^2, \qquad h_k = \frac{|R_{k+1} - R_k|}{\sum_j |R_{j+1} - R_j|}$$
        evaluated on the amide region of the summed high-energy 1D traces via $5^\circ$ coarse scan in $p_0 \in [-180^\circ, 180^\circ)$ followed by coordinate refinement of $(p_0, p_1)$.
8. **Discard imaginary parts**:
   - Retains `Re(S[k])` only, halving memory before transforming subsequent indirect dimensions.
9. **Optional polynomial baseline correction & ROI extraction**:
   - Iterative MAD-masked Legendre polynomial baseline fit (`order = 0..=5`, configurable via `BaselineConfig`).
   - Slices axis `[idx_start..=idx_end]` covering `[roi_min_ppm, roi_max_ppm]` and updates `AxisPpm { size, first_ppm, step_ppm, obs_mhz }` (`first_ppm_new = axis.index_to_ppm(idx_start as f64)`).
   - Default for direct `${}^{1}\text{H}$: `5.5..=11.5 ppm` (`--crop-h 5.5:11.5`, disable with `--no-crop` for full-sweep comparison against uncropped TopSpin `2rr`/`3rrr`).

---

## 3. CLI & Interchange Integration (`crates/nmr-cli`)

1. `nmrassign process --in <fid_dir> --out <out_path_or_dir> [--exp <EXP>] [--recipe <json>] [--no-crop] [--lp]`:
   - Processes a 2D or 3D Bruker/Varian raw FID directory (uniform or NUS) into NMRPipe (`.ft2` / `.ft3`) or Bruker `pdata` directory (readable unchanged by `nmr_core::spectrum`) and writes `processing_recipe.json` alongside it.
2. `--fid` support in `pick` and `pipeline`:
   - `nmrassign pick --fid <dir> --exp <EXP> -o peaks.tsv`
   - `nmrassign pipeline --sequence seq.fasta --fid-hsqc <dir> --fid-triple EXP=<dir> ... --out-dir <dir>` (processes FIDs in memory using `ProcessingRecipe`, writes `processing_recipe.json` to `--out-dir`, and runs the existing picker + solver unchanged).

---

## 4. TopSpin & NUS Validation Protocol (Part 3 Gate)

1. Evaluated on the 5 local Ubiquitin raw `ser` vs TopSpin `pdata/1/{2rr,3rrr}` datasets (`Ubiquitin_090309/{2:HSQC, 3:HNCACB, 4:CBCACONH}`, `Ubiquitin_111909/{10:HNCO, 11:HNCACO}`):
   - Process each raw `ser` using `WDW`/`SSB`/`PHC0`/`PHC1`/`SI`/`TDeff` from `procs`/`proc2s`/`proc3s` (without `3rrr` binary data).
   - Compare picked peaks on our processed spectrum vs TopSpin's `2rr`/`3rrr`: median and max per-axis ppm error (`|dH|`, `|dN|`, `|dC|`), Pearson correlation $r$ of normalized peak heights, and matched-peak recall.
   - Run end-to-end `pipeline` (`--solver evo-graph`) on the raw-FID-processed spectra and score against `data/ubiquitin_backbone_assignment_191104.txt` (global per-atom offset only). Gate: within $\pm 2.0$ percentage points of TopSpin `pdata` (`94.0%` accuracy / `90.0%` recall), with **zero** picker or solver tuning.
2. Validated on `BMRB 51439` (`CISD3`), covering TopSpin 4 uniform 2D/3D (`FnTYPE = 0`, `NusAMOUNT = 100`) and 3D NUS (`CISD3_CBCACONH`: `FnTYPE = 2`, `NusTD = 40 x 80`, `nuslist` 400 hypercomplex coordinates, `REVERSE = yes` on `F2`) reconstructed via hypercomplex `IST`/`FISTA`.

---

## 5. Public Archive Fetcher & Frozen Real-Protein Benchmark (Parts 4 & 5)

### 5.1 Fetcher (`nmrassign fetch-bmrb`)
- `nmrassign fetch-bmrb --id <bmrb_id> --dest data/private/bmrb/<id>/`:
  - Queries BMRB REST/HTTP (`https://api.bmrb.io/v2/entry/<id>` and `https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr<id>/` `bmr<id>_3.str` + `timedomain_data/`, with legacy `/biomolecules/` and `/timedomain/` fallbacks), enforcing `User-Agent: nmrassign-fetch` and $\le 1\text{ request/s}$ rate limiting.
  - Downloads the NMR-STAR entry (`bmr<id>_3.str`) and time-domain archive(s), extracts archives safely (preventing path traversal), and writes `PROVENANCE.md` with source URLs, CC0/BMRB public-domain license notice, UTC download timestamp, byte sizes, and SHA-256 digests of every downloaded file.

### 5.2 Frozen Manifest (`data/benchmark/real_manifest.tsv`)
Columns: `bmrb_id`, `protein`, `residues`, `field_mhz`, `vendor`, `experiments`, `split`.
- `split` is frozen as `dev` (~1/3, including `ubiquitin_local` and `51439` `CISD3`) or `heldout` (~2/3).
- Selection criteria for new entries (10–15 total): 50–150 residues, uniform or NUS raw time-domain data with complete schedule metadata, `HSQC` + `HNCO` + (`HNCACB` or `HNCA`) + (`CBCACONH` or `HNCOCA`), $\ge 80\%$ deposited backbone completeness, spanning $\alpha$, $\beta$, and $\alpha+\beta$ folds, multiple field strengths, and both Bruker and Varian/Agilent spectrometers.

### 5.3 Runner (`nmrassign real-benchmark`)
- `nmrassign real-benchmark --manifest data/benchmark/real_manifest.tsv --split dev|heldout [--solver standard|evo-graph] [--seeds 5] [--json-out results.json]`:
  - Discovers raw FID experiments in `data/private/bmrb/<id>/` (and existing local dev paths for Ubiquitin / CISD3), runs `process -> pick -> assign -> score` (including automatic NUS `IST`/`FISTA` reconstruction when `nus_mask` is active) using unlabelled global per-atom reference offset fitting (`fit_unlabelled_global_offsets`), and prints per-entry metrics, macro/micro averages, skipped missing-data entries (non-fatal for CI), and exact failure diagnostics for any missing-schedule NUS or unsupported quadrature dataset.
  - **Strict holdout governance**: `heldout` is evaluated once for reporting; no code, default, or threshold may be modified in response to `heldout` output.
