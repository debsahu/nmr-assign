# 14. Problem-Regime Solvers — Protonless `13C`-Detected IDPs, Ensemble Propensity & Paramagnetic Metalloproteins

> **Provenance & Clean-Room Statement**
> Original clean-room specification and mathematical design implemented in `crates/nmr-core/src/{c13_detected,idp_solver,idp_propensity,paramagnetic,report}.rs`, `crates/nmr-gpu/src/idp_gpu.rs`, and `crates/nmr-gpu/src/shaders/{idp_scoring,idp_anneal}.wgsl`. Zero source code, macros, parameter tables, residue libraries, or binaries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) NMR assignment, secondary-structure prediction, or paramagnetic tensor package were consulted. Every physical forward model, statistical prior, optimization kernel, and tensor solver derives strictly from peer-reviewed literature and public BMRB/PDB standards:
>
> 1. **Protonless `13C`-detected multidimensional NMR of intrinsically disordered proteins (IDPs)**:
>    - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C. & Pierattelli, R. (*J. Am. Chem. Soc.* **2006**, 128, 3918–3919) — protonless `2D CON` and `13C`-direct-detected biomolecular NMR.
>    - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2006**, 48, 25–45) — `13C` direct detection (`2D CACO`, `2D CON`, `IPAP` / `S3E` homonuclear `1J(C',CA)` virtual decoupling).
>    - Bermel, W., Bertini, I., Csizmok, V., Felli, I. C., Pierattelli, R. & Tompa, P. (*J. Magn. Reson.* **2009**, 198, 275–281) — `H`-start / `13C`-start `3D (H)CANCO`, `3D (H)CBCACON`, `3D (H)CC(CO)N`, and `3D (H)NCACO` experiments for unfolded and proline-rich proteins.
>    - Nováček, J., Žídek, L. & Sklenář, V. (*J. Magn. Reson.* **2014**, 241, 41–52) — narrow-dispersion multidimensional `13C`-detected correlation strategies for IDPs.
> 2. **Intrinsic amide proton solvent exchange & random-coil IDP chemical-shift statistics**:
>    - Bai, Y., Milne, J. S., Mayne, L. & Englander, S. W. (*Proteins* **1993**, 17, 75–86) — primary-structure, pH, and temperature dependence of intrinsic `1H^N`–water chemical exchange rates $k_{\text{ex}}$.
>    - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* **2010**, 132, 18000–18003) — sequence-specific random-coil chemical shifts (`ncIDP`) and nearest-neighbor $(i \pm 1)$ corrections for intrinsically disordered proteins.
>    - Kjaergaard, M., Brander, S. & Poulsen, F. M. (*J. Biomol. NMR* **2011**, 49, 139–149); Kjaergaard, M. & Poulsen, F. M. (*J. Biomol. NMR* **2011**, 50, 157–165) — temperature and neighbor dependence of disordered backbone shifts.
>    - Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* **2014**, 60, 109–129) — pH-dependent Henderson–Hasselbalch random-coil titration shifts of ionizable sidechains.
>    - Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (*J. Biomol. NMR* **1995**, 5, 67–81); Markley, J. L. et al. (*Pure Appl. Chem.* **1998**, 70, 117–142) — `DSS` referencing ratios $\Xi(^{13}\text{C})/\Xi(^{1}\text{H})$ and $\Xi(^{15}\text{N})/\Xi(^{1}\text{H})$.
> 3. **Fractional IDP secondary structure propensity (`SSP` & cooperative 4-state `δ2D`)**:
>    - Marsh, J. A., Singh, V. K., Jia, Z. & Forman-Kay, J. D. (*Protein Sci.* **2006**, 15, 2795–2804) — single-score 5-residue weighted Secondary Structure Propensity (`SSP`) combining `CA`, `CB`, `C'`, `HA`, `N`, and `H^N` secondary shifts with pre-Proline scaling.
>    - Camilloni, C., De Simone, A., Vranken, W. F. & Vendruscolo, M. (*Biochemistry* **2012**, 51, 2224–2231) — determination of per-residue $\alpha$-helix, $\beta$-strand, polyproline II (`PPII`), and coil conformational populations (`δ2D`) in disordered proteins from chemical shifts.
> 4. **Paramagnetic relaxation enhancement (`PRE`), pseudocontact shifts (`PCS`), and metalloprotein motifs**:
>    - Solomon, I. (*Phys. Rev.* **1955**, 99, 559–565) — electron–nuclear dipolar $r_{\text{M}}^{-6}$ transverse paramagnetic relaxation enhancement ($\Gamma_2$).
>    - Kurland, R. J. & McGarvey, B. R. (*J. Magn. Reson.* **1970**, 2, 286–301); Bertini, I., Luchinat, C. & Parigi, G. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2002**, 40, 249–273); Bertini, I., Luchinat, C., Parigi, G. & Ravera, E. (*NMR of Paramagnetic Molecules*, 2nd ed., **2017**) — point-dipole pseudocontact shift equation and traceless magnetic susceptibility anisotropy tensor $\Delta\boldsymbol{\chi}$.
>    - Schmitz, C., Stanton-Cook, M. J., Su, X.-C., Otting, G. & Huber, T. (*J. Biomol. NMR* **2008**, 41, 179–189) — linear SVD determination and nonlinear metal-coordinate optimization of the 5-parameter traceless $\Delta\boldsymbol{\chi}$ tensor.
>    - Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.* **1973**, 10, 413–432); Hestenes, M. R. (*J. SIAM* **1958**, 6, 51–90); Huber, P. J. (*Ann. Math. Stat.* **1964**, 35, 73–101); Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* **1998**, 120, 6836–6837) — separable variable-projection (`VarPro`) metal localization, one-sided Jacobi SVD, Huber IRLS outlier resistance, and PCS $Q$-factor.
>    - Wiley, S. E. et al. (*Proc. Natl. Acad. Sci. USA* **2007**, 104, 5318–5323); Conlan, A. R. et al. (*J. Mol. Biol.* **2009**, 392, 143–153); Cheng, H. & Markley, J. L. (*Annu. Rev. Biophys. Biomol. Struct.* **1995**, 24, 209–237) — `[2Fe-2S]` (`Cys3His` NEET/CISD, Rieske `Cys2His2`, ferredoxin `Cys4`), `[4Fe-4S]`, rubredoxin, and zinc-finger coordination motifs and $\text{NH}\cdots\text{S}$ chelate blind zones.

---

## 1. Protonless `13C`-Detected IDP Experiments, Narrow-Dispersion Picker & Volumetric Simulator (`T11.1`, `crates/nmr-core/src/c13_detected.rs`)

### 1.1 Physical Motivation: Amide Exchange Broadening vs Direct `13C'` Detection
Standard globular triple-resonance experiments (`1H–15N HSQC`, `HNCA`, `HNCACB`, `HNCO`; see `03-assignment.md`) detect amide protons $^1\text{H}^{\text{N}}$ and fail in two hallmark IDP regimes:
1. **Base-catalyzed solvent exchange at physiological pH/temperature**: Under the Bai–Englander random-coil exchange model (`intrinsic_amide_h_exchange_rate`; Bai et al., 1993), exposed disordered amides at $\text{pH } 7.4,\ T = 310\text{ K}$ experience exchange rates $k_{\text{ex}} = k_{\text{A}} 10^{-\text{pH}} + k_{\text{B}} 10^{\text{pH}-\text{p}K_w(T)} + k_{\text{W}} \sim 20\text{–}150\text{ s}^{-1}$ ($\Delta H^\ddagger \approx 71.1\text{ kJ mol}^{-1}$, $k_{\text{B,ref}}(\text{pH } 7, 293.15\text{ K}) \approx 8.5\text{ s}^{-1}$). The resulting transverse linewidth broadening and refocused-INEPT ($2\tau \approx 10.8\text{ ms}$) coherence loss
   $$\text{FWHM}_{\text{eff}}(^1\text{H}^{\text{N}}) = \text{FWHM}_0 + \frac{k_{\text{ex}}}{\pi}, \qquad A_{\text{HSQC}} = A_0\,\frac{\text{FWHM}_0}{\text{FWHM}_{\text{eff}}}\,\exp(-2\tau k_{\text{ex}})$$
   bleaches most non-Proline cross-peaks below the detection threshold (`simulate_h1_exchange_vs_c13_con` $\to$ `H1ExchangeComparison`), whereas non-exchangeable $^1\text{H}^\alpha$-start or $^{13}\text{C}$-start direct $^{13}\text{C}'$ detection remains immune to amide exchange.
2. **Proline abundance and narrow `1H` dispersion**: Disordered transactivation and signaling domains routinely contain `15–25%` Proline (`-Xaa-Pro-` and polyproline `-Pro-Pro-` tracts), which lack $^1\text{H}^{\text{N}}$ altogether and fracture `1H`-detected sequential walks into isolated short islands. Direct $^{13}\text{C}'$ detection anchored on `2D CON` observes every peptide bond $k \in \{1, \dots, L-1\}$ continuously, shifting `Pro` imino $^{15}\text{N}$ into a clean downfield window (`132–142 ppm` vs `106–129 ppm` for secondary amides; `is_proline_imino_region`, threshold `>= 131.5 ppm`).

### 1.2 Supported `13C`-Detected Experiment Catalog (`C13DetectedExp`)
`C13DetectedExp` models six IPAP/S3E homonuclear virtual-decoupled `13C'`-detected experiments (`direct_nucleus() == "13C'"`), converting spectrometer proton Larmor frequency $B_0$ (`500–950 MHz`) into $^{13}\text{C}$ and $^{15}\text{N}$ observe frequencies via IUPAC $\Xi$ ratios (`XI_RATIO_C13_OVER_H1 = 0.251_450_20`, `XI_RATIO_N15_OVER_H1 = 0.101_329_12`, `c13_obs_mhz`, `n15_obs_mhz`, `hz_to_ppm`):

| Variant | Canonical Token | Dim & Axes `(x, y[, z])` | Spectral Windows (ppm) | Correlated Spins & Sign Convention | Reference |
|---|---|---|---|---|---|
| `Con2D` | `"CON"` | 2D `(C'_{i-1}, N_i)` | `x: 166.0..183.0`, `y: 103.0..144.5` | Direct peptide-bond root $(C'_{k-1}, N_k)$ for all $k \in \{1,\dots,L-1\}$ including `Xaa-Pro` & `Pro-Pro` (`+1`) | Bermel et al. (2006) |
| `Caco2D` | `"CACO"` | 2D `(C'_i, CA_i)` | `x: 166.0..183.0`, `y: 40.0..69.0` | Intraresidue $(C'_k, CA_k)$; constant-time `1J(CA,CB)` evolution inverts `Gly` (`-1`, no `CB`) vs non-`Gly` (`+1`) | Bermel et al. (2006) |
| `Canco3D` | `"CANCO"` | 3D `(C'_{i-1}, N_i, CA)` | `x: 166.0..183.0`, `y: 103.0..144.5`, `z: 40.0..69.0` | Bidirectional sequential bridge on `CON(k)`: strong $CA_{k-1}$ (`~1.00`) and $CA_k$ (`~0.76`) | Bermel et al. (2009) |
| `Cbcaco3D` | `"CBCACON"` | 3D `(C'_{i-1}, N_i, C_ali)` | `x: 166.0..183.0`, `y: 103.0..144.5`, `z: 13.0..73.5` | Preceding-residue $(CA_{k-1}, CB_{k-1})$ on `CON(k)` (`Gly` at $k-1$ has `CA` only, no `CB`) | Bermel et al. (2009) |
| `Cccon3D` | `"CCCON"` | 3D `(C'_{i-1}, N_i, C_sc)` | `x: 166.0..183.0`, `y: 103.0..144.5`, `z: 9.5..73.5` | Isotropic FLOPSY/DIPSI aliphatic carbon ladder $C^{\text{ali}}_{k-1}$ (`CA`, `CB`, `CG*`, `CD*` including `Pro CD ~ 49.8 ppm`) on `CON(k)` | Bermel et al. (2009) |
| `Ncaco3D` | `"NCACO"` | 3D `(C'_i, N_i, CA_i)` | `x: 166.0..183.0`, `y: 103.0..144.5`, `z: 40.0..69.0` | Intraresidue $(C'_k, N_k, CA_k)$ forward link pairing $C'_k$ of `CON(k+1)` with $(N_k, CA_k)$ of `CON(k)` | Bermel et al. (2009) |

### 1.3 Narrow-Dispersion Deconvolving Picker & IPAP Ghost Suppression (`C13PickerConfig`)
In IDPs, non-`Gly`/`Pro` carbonyl shifts compress into `< 2.5 ppm` (`~173.5–176.5 ppm`) and `CA` shifts within a residue type cluster within `< 0.4 ppm`, while imperfect `IPAP` / `S3E` in-phase/antiphase sum/difference virtual decoupling leaves residual uncancelled doublet satellites along `F_direct(13C')` at full $\pm {}^{1}J_{\text{C}'\text{CA}}$ (`~ ±53.5 Hz`, i.e. `±0.304 ppm` at `700 MHz` $^{1}\text{H}$ / `176.0 MHz` $^{13}\text{C}$ and `±0.266 ppm` at `800 MHz` $^{1}\text{H}$ / `201.2 MHz` $^{13}\text{C}$ — since IPAP adds/subtracts the $\pm \frac{1}{2}{}^{1}J_{\text{C}'\text{CA}}$ doublet components after shifting both by $+\frac{1}{2}{}^{1}J_{\text{C}'\text{CA}}$ and $-\frac{1}{2}{}^{1}J_{\text{C}'\text{CA}}$, placing the cancelled partner at $\pm {}^{1}J_{\text{C}'\text{CA}}$ from the reconstructed singlet; Bermel et al., 2006). Every tolerance in `C13PickerConfig` is parameterized in physical **Hz** and dynamically converted via $\Delta\delta_{\text{ppm}}(B_0) = \Delta\nu_{\text{Hz}} / \nu_{\text{obs,MHz}}(B_0)$ across `500–950 MHz`:

1. **`pick_con_2d` $\to$ `Vec<Con2DPeak>`**:
   - Estimates thermal noise $\hat{\sigma}_{\text{MAD}}$ via two-pass outlier-trimmed median absolute deviation, extracts `3x3` local maxima above `snr_2d = 5.0` with sub-voxel parabolic refinement, and runs connected-component residual Gaussian shoulder deconvolution (`shoulder_snr_2d = 5.2`, `shoulder_min_rel_amp = 0.14`, `max_cluster_components = 16`, minimum separation `min_sep_c_prime_hz = 3.0 Hz`, `min_sep_n15_hz = 3.5 Hz`) to split blended IDP dipeptide clusters.
   - **Symmetric IPAP `1J(C',CA)` Satellite Filter**: Rejects candidate same-sign satellite peaks at $|\Delta\nu_{\text{C}'}| \in [45.0, 62.0]\text{ Hz}$ (`ipap_j_window_hz`) and $|\Delta\nu_{\text{N}}| \le 4.0\text{ Hz}$ around stronger parent `(C', N)` roots whenever candidate amplitude lies within the multi-parent residual + `2.5 σ_noise` budget (`ipap_max_residual_frac = 0.105`, capped at `0.24` of the strongest parent), or when a symmetric $\pm {}^{1}J_{\text{C}'\text{CA}}$ doublet pair flanks the parent (`<= 0.16` amplitude ratio), without suppressing genuine overlapping IDP roots.
2. **`pick_caco_2d` $\to$ `Vec<Caco2DPeak>`**:
   - Signed positive/negative extrema picker (`CacoPeakKind::{GlyOppositePhase, GlyRegion, SerThrProDownfield, GeneralAliphatic}`), preserving inverted-phase `Gly` `(C', CA ~ 45.1 ppm, sign = -1)` peaks alongside positive non-`Gly` cross-peaks and applying full-`1J(C',CA)` IPAP ghost suppression along `C'`.
3. **`pick_con_anchored_3d` $\to$ `Vec<C13Strip3DPeak>`**:
   - Auto-registers systematic `3D` vs `2D CON` `(ΔC', ΔN)` carrier offsets (`auto_register_3d_cubes = true`), extracts bilinearly interpolated 1D indirect `z`-strips at each `CON` root (`strip_tol_c_prime_hz = 5.5 Hz`, `strip_tol_n15_hz = 6.5 Hz`) for `CANCO`, `CBCACON`, and `CCC(CO)N` (`snr_3d = 4.5`, `min_sep_c_aliph_hz = 4.2 Hz`), and performs full 3D local-extrema picking on `NCACO` (linking `anchor_root_idx` when `(C'_k, N_k)` happens to coincide with an existing `CON` root and retaining `anchor_root_idx = None` otherwise so all intraresidue `(C'_k, N_k, CA_k)` peaks are preserved).

### 1.4 Volumetric Cube Simulator & Verified Recovery Gates (`IdpCubeSimConfig`)
`simulate_idp_13c_cubes` synthesizes volumetric 2D (`CON`, `CACO`) and 3D (`CANCO`, `CBCACON`, `CCC(CO)N`, `NCACO`) spectra with narrow-dispersion IDP linewidths (`lw_c_prime_hz = 4.8`, `lw_n15_hz = 4.2`, `lw_ca_hz = 6.2`, `lw_cb_hz = 7.2`), residual IPAP incomplete-cancellation satellites (`ipap_residual_frac = 0.025` at full `±53.5 Hz`), constant-time `Gly` phase inversion in `CACO`, and deterministic thermal noise:
- **Verified volumetric `2D CON` picker recovery on Proline-rich real BMRB IDPs (`800 MHz`, `3%`–`8%` IPAP satellites)**:
  - `bmr26639` (`CBP-ID4`, `207 aa`, `45 Pro = 21.74% Pro`, containing `-Xaa-Pro-` and `-Pro-Pro-` / `PPPP` tracts): **`100.00%` (`205/205`, `2.44%` spurious)** recovery (`>= 97.0%` gate).
  - `bmr51984` (`p53 TAD+PRR`, `102 aa`, `22 Pro = 21.57% Pro`): **`100.00%` (`87/87`, `0` spurious at `3%`–`5%` IPAP / `<= 2` at `8%` IPAP)** recovery (`>= 97.0%` gate).

---

## 2. Continuous Through-Proline `13C`-Detected IDP Backbone Solver (`T11.2`, `crates/nmr-core/src/idp_solver.rs`, `crates/nmr-gpu/src/idp_gpu.rs`)

### 2.1 `CON(k)` Dipeptide Spin-System Formulation (`ConDipeptideSystem`)
Unlike `1H`-detected `SpinSystem`s (which skip `Pro` slots with `FORBID`), each `13C`-detected `ConDipeptideSystem` $s \in \{0, \dots, S-1\}$ represents a **peptide bond** $(k-1, k)$ anchored at $(C'_{k-1}, N_k)$ and can map to **every sequence slot $k \in \{1, \dots, L-1\}$ continuously across `-Xaa-Pro-` and `-Pro-Pro-` boundaries** (only N-terminal $k = 0$ is `FORBID = -1.0e9` because residue `0` has no preceding peptide carbonyl $C'_{-1}$):
- Anchor shifts: `c_prev` ($C'_{k-1}$), `n_curr` ($N_k$).
- Bounded candidate arrays (`MAX_IDP_CANDS = 6`, `MAX_IDP_SC_CANDS = 6`): `ca_prev` ($CA_{k-1}$ from `CBCACON`/`CANCO`), `cb_prev` ($CB_{k-1}$ from `CBCACON`/`CCC(CO)N`), `ca_curr` ($CA_k$ from `CANCO` after subtracting `CBCACON`-tagged $CA_{k-1}$, plus `NCACO` corroboration), `c_curr` ($C'_k$ from `NCACO` + `2D CACO`), `n_next` ($N_{k+1}$ from `CON` roots sharing $C'_k$), `sidechain_c_prev` ($C^{\text{ali}}_{k-1}$ from `CCC(CO)N`), `prev_gly_opposite_phase` (from inverted `2D CACO`), and `has_cbcacon_evidence`.

### 2.2 Sequence-Specific `ncIDP` Random-Coil Predictor & Unsupervised Registration
1. **`predict_ncidp_shifts(sequence, temp_k, ph)`**:
   Evaluates sequence-specific disordered random-coil chemical shifts `(N, C', CA, CB)` across the primary sequence by combining:
   - Tamiola, Acar & Mulder (2010) `ncIDP` base random-coil shifts at `298.15 K, pH 6.5` (including imino `Pro 15N = 136.8 ppm`),
   - Zero-centered preceding-residue $(i-1)$ neighbor corrections (with explicit tertiary-imide `Xaa-Pro` $\Delta\delta_N$ modulation) and following-residue $(i+1)$ neighbor corrections (capturing the hallmark pre-Proline shifts $\Delta\delta_{\text{CA}} = -1.95\text{ ppm}$, $\Delta\delta_{\text{C}'} = -1.95\text{ ppm}$, $\Delta\delta_{\text{N}} = +1.65\text{ ppm}$, $\Delta\delta_{\text{CB}} = -0.35\text{ ppm}$ when $\text{seq}[k+1] = \text{PRO}$),
   - Platzer, Okon & McIntosh (2014) Henderson–Hasselbalch pH titration shifts for ionizable `Asp`, `Glu`, `His`, `Cys`, `Tyr`, `Lys`, and termini, and
   - Kjaergaard, Brander & Poulsen (2011) linear amide/aliphatic temperature coefficients $\frac{\partial\delta}{\partial T}(T - 298.15\text{ K})$.
2. **Unlabelled Permutation-Invariant Reference Offset Estimation (`estimate_idp_reference_offsets`)**:
   Estimates systematic referencing offsets `IdpShiftOffsets { d_n, d_c, d_ca, d_cb }` strictly from the unordered multisets of observed shifts vs sequence-predicted `ncIDP` expectations via 3.5-MAD outlier-trimmed Huber M-estimation (`0.0` permutation sensitivity; recovers synthetic differential `DSS`/`TMS`/indirect-`15N` offsets to `< 0.04 ppm`).
3. **Unsupervised Inter-Experiment Registration (`register_and_build_con_dipeptide_systems`)**:
   Aligns indirect `CA` axes of `CBCACON`/`CCC(CO)N` onto `CANCO`, aligns `2D CACO` `(C', CA)` onto `(CON, CANCO)`, and registers **both** unanchored (`anchor_root_idx == None`, matched within `|n_corr - root.n_curr_ppm| <= 0.15 ppm`) and anchored `3D NCACO` `(ΔC', ΔN, ΔCA)` carrier drifts unsupervised while retaining residual `dn_abs` inside candidate $\chi^2$ ranking.

### 2.3 Dipeptide Unary $U[s, k]$ & Sequential Pair $P[s_1, s_2, \delta]$ Scoring
1. **Dipeptide Unary Score $U[s, k]$ (`idp_unary_matrix` / WGSL `compute_idp_unary`)**:
   For peptide-bond system $s$ placed at sequence dipeptide $(k-1, k)$ ($k \ge 1$):
   $$U[s, k] = b_{\text{sys}}(s) + \sum_{\nu \in \{N_k, C'_{k-1}, CA_k, CA_{k-1}, CB_{k-1}, C'_k\}} \ln\frac{p_{\text{mix}}(\delta_\nu \mid \mu_{\nu,k}, \sigma_\nu)}{p_{\text{bg}}(\delta_\nu)} + \text{LLR}_{\text{Gly/CB}}(s, k-1) + \text{LLR}_{\text{CCCON}}(s, k-1)$$
   where $p_{\text{mix}}(\delta \mid \mu, \sigma) = 0.78\,\mathcal{N}(\delta \mid \mu, \sigma^2) + 0.22\,\mathcal{N}(\delta \mid \mu, (3.8\sigma)^2)$ absorbs transient IDP secondary-structure excursions (`sigma_ca_rc = 0.45`, `sigma_cb_rc = 0.55`, `sigma_c_rc = 0.40`, `sigma_n_rc = 0.85`, `sigma_pro_n_rc = 1.15 ppm`, floored at `llr_floor = -5.5`).
   - **3D-Corroborated Noise Rejection (`sys_assign_bonus`)**: When `multi_3d_dataset` is active ($\ge 50\%$ of `CON` roots carry 3D strip peaks), roots corroborated by $\ge 1$ 3D channel receive `b_sys = +2.65`, whereas **3D-unsupported 2D `CON` noise roots (`n_3d_ch == 0`) unconditionally receive `b_sys = -6.5`** even on incomplete sequence depositions where $S + 1 < L$, cleanly leaving spurious thermal/IPAP roots unassigned (`pos[s] = -1`) instead of forcing them into genuine vacant sequence gaps.
2. **Sequential Pair Connectivity $P[s_1, s_2]$ (`idp_pair_matrix` / WGSL `compute_idp_pair`)**:
   Links `CON(k)` ($s_1$) to `CON(k+1)` ($s_2$) across up to three independent covalent bridges:
   - Shared residue-$k$ $\text{C}\alpha$: $CA_k(s_1, \text{intra}) \leftrightarrow CA_k(s_2, \text{prev})$ ($\sigma_{\text{seq,CA}} = 0.085\text{ ppm}$),
   - Forward carbonyl link: $C'_k(s_1, \text{NCACO/CACO}) \leftrightarrow C'_k(s_2, \text{CON root})$ ($\sigma_{\text{seq,C}'} = 0.055\text{ ppm}$),
   - Forward nitrogen link: $N_{k+1}(s_1) \leftrightarrow N_{k+1}(s_2, \text{CON root})$ ($\sigma_{\text{seq,N}} = 0.14\text{ ppm}$).

### 2.4 CPU & WebGPU Multi-Replica Island Annealing (`16 KiB` SRAM Parity)
- **CPU Engine (`assign_idp_13c_cpu`, `assign_idp_13c_from_picked`)**: Multi-threaded island simulated annealing (`48` replicas across `4` islands $\times$ `3` elite-seeded generations, `2,400 * max(S, L)` steps/replica, $T: 16.0 \to 0.08$) combining single-slot swap/insert/unassign proposals (`phase < 10`), sequential bridge-guided neighbor extension (`phase < 14`), and `2..=5`-residue dipeptide block swaps (`phase >= 14`) that resolve repetitive `-Pro-Xaa-Pro-` and poly-`Gln`/`Gly`/`Ser` degeneracies.
- **WebGPU Engine (`IdpGpuSolver`, `idp_scoring.wgsl`, `idp_anneal.wgsl`)**:
  - `compute_idp_unary` and `compute_idp_pair` compute `(U, P)` on the GPU within `bounded_adapter_limits` (`<= 6` storage buffers + `1` uniform buffer; **verified parity `max |U_cpu - U_gpu| = 5.72e-6 < 1e-4` and `max |P_cpu - P_gpu| = 3.34e-6 < 1e-4`**).
  - Three size-dispatched WGSL annealing kernels mirror `anneal.wgsl`: `anneal_idp_256` (`@workgroup_size(8, 1, 1)`, stride `256` $\to$ `wg_at: array<i32, 2048>` + `wg_pos: array<i32, 2048>` = exact **`16,384 B` (`16 KiB`)** portable WebGPU workgroup SRAM), `anneal_idp_1024` (`@workgroup_size(2, 1, 1)`, stride `1024` = `16 KiB` SRAM), and `anneal_idp_global` (`> 1024` global storage fallback), chunked at `CHUNK_STEPS = 32_768` per submission to prevent OS watchdog (`TDR`) resets. Synchronous GPU entry points return an explicit `Err` on `wasm32` (`GpuEngine::read_buffer_vec_async` provides a low-level `Future` readback primitive for future async browser callers).
- **Length-Invariant Confidence Calibration & Unlabelled QC (`consensus_and_calibrate_idp`)**:
  Combines replica consensus stability $p_{\text{ens}}(k)$ (`TAU_ENS = 6.5`) with a **chain-length-invariant local Boltzmann specificity** $p_{\text{local}}(k)$ evaluated against the top `MAX_ALT_COMPETITORS = 8` strongest single-slot competitors plus the strongest `2..=5`-residue homologous motif swap (`TAU_LOCAL = 1.95`), mapped via $p_{\text{raw}} = 0.64\,p_{\text{local}} + 0.36\sqrt{p_{\text{local}}\,p_{\text{ens}}}$ through monotone `IDP_CALIBRATION_KNOTS`. Unlabelled dataset quality control (`DatasetQcVerdict`) forces `Abstain` (`conf <= 0.45`) whenever 3D strip support `< 0.25` or sequential connectivity `< 0.15`.

### 2.5 Solver-Only 5-Protein Real-BMRB IDP Benchmark (`T11.2`)
Evaluated strictly **non-circularly at the solver level** (`crates/nmr-core/tests/idp_solver_tests.rs`, `crates/nmr-gpu/tests/idp_gpu_tests.rs`; `pick_con_2d` is tested separately in §1.4) by constructing peak lists directly from experimental deposited BMRB chemical shifts across 5 real disordered proteins (`bmr6968` human $\alpha$-synuclein `3.6% Pro`, `bmr15176` `DARPP-32` `12.6% Pro`, `bmr4375` `8 M` urea-unfolded human Ubiquitin `pH 2.0` `3.9% Pro`, `bmr26639` Pro-rich `CBP-ID4` `21.7% Pro`, `bmr51984` Pro-rich `p53 TAD+PRR` `21.6% Pro`; `σ = 0.012–0.035 ppm` jitter, inter-experiment carrier drift, `10%` missing channels, `3%` spurious `CON` noise roots [minimum 2], and `8%` spurious 3D strip peaks $\to$ unlabelled `assign_idp_13c_from_picked` / `assign_idp_13c_from_picked_gpu`, zero `ncIDP` shift synthesis; `ECE` computed over assigned `CON` roots `pos[s] >= 1`):

| BMRB ID | Disordered Target | Length (`Pro %`) | CPU Acc | CPU Recall (vs Deposited Bonds) | CPU HC-Prec (`>= 0.80`) | CPU `ECE` | WebGPU Acc | WebGPU HC-Prec (`>= 0.80`) | WebGPU `ECE` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bmr6968` | $\alpha$-Synuclein | 140 aa (`3.6%`) | **`94.24%`** (`131/139`) | `94.24%` (`131/139`) | **`100.00%`** (`121/121`) | `0.0348` | **`94.24%`** | **`100.00%`** (`121/121`) | `0.0348` |
| `bmr15176` | `DARPP-32` | 119 aa (`12.6%`) | **`99.01%`** (`100/101`) | `99.01%` (`100/101`) | **`100.00%`** (`99/99`) | `0.0082` | **`99.01%`** | **`100.00%`** (`99/99`) | `0.0082` |
| `bmr4375` | `8 M`-Urea Ubiquitin (`pH 2.0`) | 76 aa (`3.9%`) | **`94.29%`** (`66/70`) | `94.29%` (`66/70`) | **`98.44%`** (`63/64`) | `0.0296` | **`94.29%`** | **`98.44%`** (`63/64`) | `0.0296` |
| `bmr26639` | `CBP-ID4` (`PPPP`) | 207 aa (**`21.7%`**) | **`98.05%`** (`201/205`) | `98.05%` (`201/205`) | **`100.00%`** (`179/179`) | `0.0490` | **`98.05%`** | **`100.00%`** (`179/179`) | `0.0480` |
| `bmr51984` | `p53 TAD+PRR` | 102 aa (**`21.6%`**) | **`98.85%`** (`86/87`) | `98.85%` (`86/87`) | **`100.00%`** (`85/85`) | `0.0174` | **`98.85%`** | **`100.00%`** (`85/85`) | `0.0175` |
| **Pooled** | **All 5 Real IDPs** | **644 aa (`602` bonds)** | **`97.01%`** (`584/602`) | **`97.01%`** (`584/602`) | **`99.82%`** (`547/548`) | **`0.0231`** | **`97.01%`** | **`99.82%`** (`547/548`) | **`0.0227`** |

Every individual protein and the pooled ensemble satisfy `Accuracy >= 90%`, `HC-Precision >= 95%`, and `ECE <= 0.06` on both CPU and WebGPU.

---

## 3. IDP Secondary Structure Propensity — `SSP` & Cooperative 4-State `δ2D` (`T11.3`, `crates/nmr-core/src/idp_propensity.rs`, `crates/nmr-core/src/report.rs`)

### 3.1 Unlabelled Reference Offset & Sample pH Sensing
Before computing secondary chemical shifts $\Delta\delta_{k,a} = (\delta^{\text{obs}}_{k,a} - \hat{\delta}^{\text{off}}_a) - \delta^{\text{RC}}_{k,a}(T, \text{pH}_{\text{eff}})$ against `ncIDP` random-coil baselines:
1. **`estimate_global_rc_offsets` (Conformation-Orthogonal Combinations)**: Because `CA` and `CB` shift in **opposite** directions in $\alpha$-helices (`+3.1` vs `-0.45 ppm`) and $\beta$-strands (`-1.5` vs `+2.15 ppm`), the linear combinations $0.48\,\Delta\delta_{\text{CA}} + 0.52\,\Delta\delta_{\text{CB}}$ and $\Delta\delta_{\text{C}'} + 0.72\,\Delta\delta_{\text{CB}}$ cancel conformational bias to first order even in fully folded proteins, recovering `DSS`/`TMS` referencing offsets over interior `[0.15, 0.85]` quantiles with exact permutation invariance (`< 1e-12`).
2. **Unsupervised Clamped Acidic-Sample pH Detector (`infer_effective_ph`)**: When sample pH is unspecified (`ph = None`), compares the unlabelled `Asp`/`Glu` vs neutral aliphatic `CB` median contrast against an acidic threshold: neutral/physiological samples return `6.5`, whereas strongly acidic samples map through a clamped protonation-fraction window (`[0.65, 0.96]`) onto $\text{pH}_{\text{eff}} \in [2.72, 3.83]$ (`auto_inferred_ph`), preventing false $\alpha$-helical drift on low-pH unfolded depositions (such as `bmr4375` `8 M`-urea Ubiquitin deposited at `pH 2.0`, which hits the `2.72` acidic clamp floor).

### 3.2 Marsh-2006-Inspired Extended 5-Residue Weighted `SSP`-Like Score
Inspired by Marsh et al. (*Protein Sci.* 2006 — which defines unweighted 5-residue `(ΔCA - ΔCB)` averaging with pre-Proline scaling), `nmr-core::idp_propensity` computes an extended multi-nucleus `SSP`-like score across $a \in \{\text{CA}, \text{CB}, \text{C}', \text{HA}, \text{N}, \text{H}^{\text{N}}\}$ at position $j$ (`Gly` omits `CB`; `Pro` omits `H`/`N`; pre-Proline $\text{seq}[j+1] = \text{PRO}$ scales `CA`, `CB`, `C'` by `0.65`). Each per-nucleus secondary shift is divided by residue-dependent $\alpha$-helical ($\Delta\delta^\alpha_{r,a}$) or $\beta$-strand ($\Delta\delta^\beta_{r,a}$) reference shifts, clamped to $[-1.35, +1.35]$, weighted by $w_a = 1 / \sigma_{\text{RC},a}$, and smoothed across a $\pm 2$-residue window (`window_half_width = 2`, positional weights `[1.00, 0.82, 0.52]` for $|j - i| = 0, 1, 2$):
$$\text{SSP}(i) = \text{clamp}\!\left(\frac{\sum_{j=i-2}^{i+2} \omega_{|j-i|} \sum_{a \in \mathcal{O}_j} w_a\, f_{j,a}}{\sum_{j=i-2}^{i+2} \omega_{|j-i|} \sum_{a \in \mathcal{O}_j} w_a},\; -1.25,\; +1.25\right)$$
Positive `SSP` (`-> +1.0`) quantifies fractional $\alpha$-helix formation; negative `SSP` (`-> -1.0`) quantifies extended $\beta$-strand propensity.

### 3.3 Camilloni-2012/`δ2D`-Inspired Cooperative 4-State Forward–Backward HMM Populations
Motivated by Camilloni et al.'s 4-state `δ2D` decomposition (*Biochemistry* 2012 — whereas Camilloni et al.'s original `δ2D` fits reference-set chemical-shift mixtures, our clean-room estimator formulates a 1D linear-chain Hidden Markov Model), each residue $i$ occupies one of four conformational basins $s_i \in \{\alpha, \beta, \text{PPII}, \text{Coil}\}$ (`ConformationalState`) inferred via `compute_idp_propensity`:
1. **Per-State Chemical-Shift Emission Potentials $\phi_i(s)$**: Evaluates Gaussian log-likelihoods of sequence-corrected secondary shifts $\Delta\delta_{i,a}$ around calibrated 4-state basin centroids ($\alpha$: `+3.10 CA, -0.45 CB, +2.05 C'`; $\beta$: `-1.50 CA, +2.15 CB, -1.55 C'`; `PPII`: `-0.38 CA, +0.28 CB, -0.32 C', +0.65 N`; `Coil`: `0.0`), combined with residue-intrinsic conformational baselines (`Pro` enriched in `PPII`; `Gly` enriched in `Coil`; unassigned residues controlled by `UnassignedPriorMode::{CoilBaseline, Uniform}`).
2. **Cooperative Nearest-Neighbor Forward–Backward Marginals**: Applies cooperative log-transition matrices (`trans_std` enforcing the $\ge 4$-residue $\alpha$-helix hydrogen-bond nucleation penalty vs local `PPII`/`Coil` interconversion, and `trans_into_pro` suppressing helix propagation across $i \to \text{Pro}_{i+1}$), solving exact Forward–Backward log-domain marginals normalized via numerically stable `softmax4`:
   $$p_\alpha(i) + p_\beta(i) + p_{\text{PPII}}(i) + p_{\text{coil}}(i) = 1.0 \quad (\text{exact within } \pm 10^{-9}\text{ for every residue } i)$$

### 3.4 Reporting & Quantitative Validation (`T11.3`)
- **Exporters & HTML Track**: `format_propensity_tsv`, `format_propensity_json`, and `render_idp_propensity_svg` (embedded automatically in `nmr-core::report` section `3b` when `ReportInput.propensity_report` is supplied via `nmrassign report --idp-propensity` / `--propensity` or `nmrassign pipeline --regime idp-13c`).
- **Verified Benchmarks**:
  - **Folded control (`bmr17769` Human Ubiquitin vs `PDB 1D3Z`)**: **`38/40 = 95.00%`** secondary-structure element agreement (`11/11` $\alpha_1$ residues `23..=33` and `27/29` $\beta_1$–$\beta_5$ core residues; `>= 85%` gate).
  - **Disordered BMRB controls**: `bmr6968` ($\alpha$-synuclein) yields `mean(p_coil + p_ppii) = 0.908` with `0` false rigid runs; `bmr4375` (`8 M` urea-unfolded Human Ubiquitin at `pH 2.0`, auto-sensed acidic `pH = 2.72`) yields `mean(p_coil + p_ppii) = 0.868` with `0` false rigid segments.

---

## 4. Paramagnetic Metalloprotein `PRE` Bleaching & 5-Parameter `Δχ` Tensor SVD (`T11.4`, `crates/nmr-core/src/paramagnetic.rs`)

### 4.1 Solomon $r_{\text{M}}^{-6}$ `PRE` Bleaching & Multi-Cluster Coordination Motifs
Unpaired electron spins ($S \ge \frac{1}{2}$ in `[2Fe-2S]`, `[4Fe-4S]`, high-spin $\text{Fe}^{2+}/\text{Fe}^{3+}$, $\text{Co}^{2+}$, $\text{Cu}^{2+}$, and paramagnetic lanthanides $\text{Ln}^{3+}$) enhance nuclear transverse relaxation via Solomon (1955) electron–nuclear dipolar coupling $\Gamma_2 \propto r_{\text{M}}^{-6}$, bleaching `1H–15N` HSQC and 3D triple-resonance signals inside a sphere of characteristic bleach radius $r_{\text{bleach}}$ (`default = 8.5 Å`):

1. **3D Multi-Cluster Coordinate Bleaching (`compute_bleach_probabilities`)**:
   - Given explicit open-shell metal coordinates (`HETATM` ions `FE`, `FES`, `SF4`, `CO`, `CU`, `NI`, `MN`, `TM`, `TB`, `DY`, `ER`, `YB`, `HO`, `EU`, `GD`, `CE`, `PR`, `ND`, `SM`) or coordinating ligand sidechains, donor atoms (`SG` on `Cys`, `ND1`/`NE2` on `His`, `OD*`/`OE*` on `Asp`/`Glu`) are partitioned into **single-linkage 3D spatial clusters (`<= 8.5 Å` donor–donor cutoff)** so multi-site metalloproteins (such as tandem `[2Fe-2S]` `CISD3`, whose two clusters sit `~16 Å` apart) place one metal center per cluster rather than collapsing onto an unphysical inter-cluster midpoint!
   - For each residue $k$ at amide distance $r_{k,m}$ from metal center $m$, the cooperative bleach probability is:
     $$p_{\text{bleach}}(k) = \frac{\sum_m (r_{\text{bleach}} / r_{k,m})^6}{1 + \sum_m (r_{\text{bleach}} / r_{k,m})^6} \in [0, 1]$$
2. **Sequence-Only Coordination Motif Detection (`detect_metal_coordination_motifs`)**:
   When no 3D `.pdb` is supplied, `paramagnetic.rs` parses explicit `--metal-ligands` (`parse_metal_ligands`, accepting `CYS42`, `C42`, or bare `42`) or automatically scans primary sequence motifs (`MetalMotifKind`):
   - `Cisd2Fe2S` (`C-X-C-X(5..16)-C-X(1..4)-H` NEET/CISD `[2Fe-2S]` `Cys3His`),
   - `Rieske2Fe2S` (`C-X-H-X(14..48)-C-X2-H`),
   - `FerredoxinFeS` (`C-X2-C-X2-C-X(3..45)-C` `[4Fe-4S]` / plant `C-X4-C-X2-C-X(20..40)-C` `[2Fe-2S]`),
   - `RubredoxinOrCys4`, `Cys2His2Finger`, and `Cys2HisCysFinger`.
   Coordinating ligands receive $p_{\text{bleach}} = 0.96$, intra-chelate loops ($\le 5$ aa) receive $0.92$, and the four post-chelate $\text{NH}\cdots\text{S}$ hydrogen-bonded turn-cap residues $(b+1 \dots b+4)$ receive $0.88, 0.78, 0.72, 0.66$ (`0.38` at $a-1$). Coordinating histidines (`coordinating_histidine_indices`) automatically set `metal_coordinated_hint = true` in `nmr-core::aromatic::HisTautomerEvidence`.

### 4.2 Paramagnetic-Aware Backbone Optimization (`u_para` Annealing & `>= 2`-Slot Rescue)
`assign_paramagnetic_backbone` and `assign_paramagnetic_from_peaks` (`ParamagneticSolverMode::{Standard, EvoGraph}`) inject $p_{\text{bleach}}(k)$ **directly into the primary unary scoring matrix $U_{\text{para}}[s, k]$ before simulated annealing or evolutionary factor-graph inference**, rather than post-filtering an already corrupted global walk:
- **Hard Blind-Zone Exclusion**: Slots with $p_{\text{bleach}}(k) \ge p_{\text{hard}}$ (`hard_bleach_cutoff = 0.85`) receive `FORBID = -1.0e9`, preventing unassignable noise/artifact roots from hijacking bleached Fe-S loop positions and cascading false sequential extensions into adjacent diamagnetic $\beta$-strands.
- **Penumbra Log-Survival Prior (`0.02 < p_bleach < 0.85`)**: Adds $\Delta U = w_{\text{bleach}} \ln(1 - p_{\text{bleach}}(k))$ (`bleach_prior_weight = 2.2`), plus a multi-competitor ambiguity guard in `[0.65, 0.85)` when $\ge 2$ alternative diamagnetic slots exist.
- **Excess `CA`-Less Artifact Guard**: When $\sum_k p_{\text{bleach}}(k) \ge 1.5$ and total spin systems exceed expected diamagnetic amides, `CA`-less partial noise roots are forbidden from displacing complete triple-resonance systems.
- **Post-Optimization Rescue (`filter_blind_zones_and_reoptimize`)**: Re-Checks any borderline blind-zone placement (`>= 0.70`) and rescues systems possessing $\ge 2$ compatible diamagnetic sequence slots onto vacant diamagnetic positions with positive sequential support.

### 4.3 5-Parameter Traceless $\Delta\boldsymbol{\chi}$ Tensor SVD, Huber IRLS & 8-Parameter `VarPro`
In anisotropic paramagnetic centers (lanthanides $\text{Tm}^{3+}, \text{Tb}^{3+}, \text{Dy}^{3+}, \text{Yb}^{3+}$, low-spin $\text{Fe}^{3+}$, high-spin $\text{Co}^{2+}$), the point-dipole pseudocontact shift (`PCS`, in ppm) at nuclear displacement $\mathbf{r} = (x, y, z) = \mathbf{x}_{\text{nuc}} - \mathbf{x}_{\text{M}}$ ($r = \|\mathbf{r}\|_2$ in Å, $\Delta\chi$ components in SI $10^{-32}\text{ m}^3$) is exact linear in the 5 independent components of the symmetric traceless tensor $\Delta\boldsymbol{\chi}$ ($\Delta\chi_{zz} = -\Delta\chi_{xx} - \Delta\chi_{yy}$):
$$\Delta\delta^{\text{PCS}}(\mathbf{r}) = \frac{10^{4}}{4\pi r^5}\Big[(x^2 - z^2)\,\Delta\chi_{xx} + (y^2 - z^2)\,\Delta\chi_{yy} + 2xy\,\Delta\chi_{xy} + 2xz\,\Delta\chi_{xz} + 2yz\,\Delta\chi_{yz}\Big]$$
1. **Linear Thin Jacobi SVD (`fit_pcs_tensor_svd`)**: Validates $N \ge 6$ (`>= 1` residual DoF), finite coordinates/PCS values, and $\sigma > 0$, solves $\min_{\boldsymbol{\theta} \in \mathbb{R}^5} \|\mathbf{W}^{1/2}(\mathbf{A}\boldsymbol{\theta} - \mathbf{y})\|_2$ via exact one-sided Hestenes Jacobi SVD (`cond <= 1e8` rank-deficiency guard), diagonalizes the $3\times 3$ traceless tensor via symmetric Jacobi rotations, and orders principal eigenvalues in the Schmitz/Otting Unique Tensor System (`UTS`) convention $|\Delta\chi'_{zz}| \ge |\Delta\chi'_{yy}| \ge |\Delta\chi'_{xx}|$ with $\Delta\chi_{\text{ax}} = \frac{3}{2}\Delta\chi'_{zz}$ and $\Delta\chi_{\text{rh}} = \Delta\chi'_{xx} - \Delta\chi'_{yy}$ ($0 \le \Delta\chi_{\text{rh}} / \Delta\chi_{\text{ax}} \le \frac{2}{3}$), reporting active right-handed `ZYZ` Euler angles $(\alpha, \beta, \gamma)$ and the Cornilescu quality factor $Q = \sqrt{\sum_i (\Delta\delta^{\text{obs}}_i - \Delta\delta^{\text{calc}}_i)^2 / \sum_i (\Delta\delta^{\text{obs}}_i)^2}$ (`f64::NAN` when $\sum_i (\Delta\delta^{\text{obs}}_i)^2 = 0$).
2. **Coordinate-Uncertainty GLS & Huber IRLS (`fit_pcs_tensor_robust`)**: Propagates isotropic 3D structural coordinate uncertainty $\sigma_{\text{coord}}$ (`Å`) analytically through $\nabla_{\mathbf{r}}\Delta\delta^{\text{PCS}}$ into per-spin effective variance $\sigma_{\text{eff},i}^2 = \sigma_{\text{pcs},i}^2 + \frac{1}{3}\sigma_{\text{coord},i}^2 \|\nabla_{\mathbf{r}}\Delta\delta^{\text{PCS}}(\mathbf{r}_i)\|_2^2$ (`effective_pcs_sigma_ppm` — downweighting $r^{-4}$-steep near-metal gradients automatically) combined with Huber IRLS outlier resistance (`huber_k = 1.65`).
3. **8-Parameter Separable Variable-Projection Metal Refinement (`refine_metal_position_and_tensor`)**: Optimizes $\mathbf{x}_{\text{M}} = (x_0, y_0, z_0) \in \mathbb{R}^3$ via coarse 3D grid search followed by multi-start Nelder–Mead simplex minimization while projecting out $\boldsymbol{\theta}(\mathbf{x}_{\text{M}}) \in \mathbb{R}^5$ exactly at every step via GLS/IRLS SVD.

### 4.4 Verified Paramagnetic & `CISD3` (`BMRB 51439`) Benchmarks (`T11.4`)
- **Synthetic `Δχ` Tensor Recovery (`Tm3+`, `Tb3+`, `Dy3+` on `PDB 1D3Z`)**:
  - Noise-free exact recovery: **`0.000000%` error** (`Q <= 1.51e-15`).
  - Realistic `0.03 ppm` thermal PCS noise + **`0.30 Å` 3D RMS structural coordinate noise**: `Tm3+` `0.11%` ($\Delta\chi_{\text{ax}}$) / `0.60%` ($\Delta\chi_{\text{rh}}$), `Tb3+` `0.27% / 0.59%`, `Dy3+` `0.01% / 0.09%` — **all `< 0.61%` (`< 1.0%` gate)**; 8-parameter `VarPro` localizes metal coordinates from a `2.6 Å`-displaced start to `0.072 Å` error (`0.22% / 0.50%` tensor error).
- **Non-Circular Real Experimental `CISD3` (`BMRB 51439`, Tandem `[2Fe-2S]` Metalloprotein, 5 Seeds `0..4`, `paramagnetic_tests::test_real_cisd3_paramagnetic_5seed_gate`)**:
  - **`Standard` solver**: `53.33%` $\to$ **`60.61%` accuracy (`+7.28 pp`)**, recall `37.65%` $\to$ **`39.22%` (`+1.57 pp`)**, **`100.00%` HC-precision**, and **`0` false `[2Fe-2S]` loop assignments across all 5 seeds** (down from `44`).
  - **`EvoGraph` solver** (post-`P1-7a` parameter-free Pass-1): `55.43%` $\to$ **`62.26%` accuracy (`+6.83 pp`)** (identical `62.26%` under sequence-auto-detected motifs and explicit `--metal-ligands`), recall **`38.04%`**, **`100.00%` HC-precision**, and **`0` false `[2Fe-2S]` loop assignments across all 5 seeds** (down from `22`).

---

## 5. CLI & Interactive Workbench Contracts (`crates/nmr-cli`, `crates/nmr-gui`)

### 5.1 Command-Line Interface (`nmrassign`)
```text
# 1. Problem-regime backbone assignment (globular default, 13C-detected IDP, or paramagnetic):
nmrassign assign (--seq <SEQ|seq.fasta> | --star <ref.str>) [--peaks <peaks.tsv>] \
  [--regime globular|idp-13c|paramagnetic] [--solver evo-graph|standard] [--engine auto|gpu|cpu] \
  [--con <path>] [--caco <path>] [--canco <path>] [--cbcacon <path>] [--cccon <path>] [--ncaco <path>] \
  [--metal-ligands CYS42,CYS44,CYS51,HIS54] [--pdb <model.pdb>] [--pcs <pcs.tsv>] \
  [--temp-k 298.15] [--ph 6.5] --out <prefix>

# 2. One-command end-to-end pipeline (supports --regime globular|idp-13c|paramagnetic):
nmrassign pipeline (--seq <SEQ|seq.fasta> (--hsqc <path> | --fid <dir> | --con <path> | --peaks <peaks.tsv>) | --star <entry.str>) \
  [--regime globular|idp-13c|paramagnetic] [--engine auto|gpu|cpu] --out-dir <out/>

# 3. Standalone IDP secondary structure propensity (Marsh-inspired SSP + Camilloni-inspired 4-state HMM):
nmrassign propensity --shifts <shifts.str|shifts.nef> [--temp-k 298.15] [--ph 6.5] \
  [--format tsv|json] [-o propensity.tsv]

# 4. HTML + inline SVG quality report with stacked 4-state d2D + SSP track:
nmrassign report --star|--shifts <shifts.str> [--peaks <peaks.tsv>] [--pdb <model.pdb>] \
  [--idp-propensity|--propensity] [--temp-k 298.15] [--ph 6.5] --out <report.html>
```
- `--regime globular` (default) preserves exact backwards-compatible `1H–15N` triple-resonance behavior (`03-assignment.md`).
- `--regime idp-13c` accepts pre-picked `TSV` peak lists or volumetric `13C`-detected spectra (`--con`, `--caco`, `--canco`, `--cbcacon`, `--cccon`, `--ncaco` or auto-classified folders via `pick_c13_idp_dataset`) and dispatches `nmr_gpu::idp_gpu::assign_idp_13c_gpu` / `assign_idp_13c_cpu`: **`--engine gpu` fails with an explicit hard error (zero silent CPU fallback)** if no WebGPU adapter is available or GPU execution errors, `--engine auto` (default) attempts WebGPU and falls back to `assign_idp_13c_cpu` with a loud `stderr` warning, and `--engine cpu` runs directly on CPU, writing continuous through-Proline NMR-STAR v3 / NEF v1.1 shifts and confidences (and automatically embedding the `SSP`/`δ2D` track in `nmrassign pipeline --regime idp-13c`).
- `--regime paramagnetic` dispatches `assign_paramagnetic_from_peaks` (`--solver standard|evo-graph`) with sequence-motif / `--metal-ligands` / `--pdb` Solomon $r_{\text{M}}^{-6}$ `PRE` bleaching, and when `--pcs <pcs.tsv>` and `--pdb <model.pdb>` are supplied, fits and prints the 5-parameter traceless $\Delta\boldsymbol{\chi}$ tensor ($\Delta\chi_{\text{ax}}$, $\Delta\chi_{\text{rh}}$, `ZYZ` Euler angles, and Cornilescu $Q$-factor).

### 5.2 Interactive WebGPU Workbench (`nmr-gui`)
- **Experiment Selection Matrix & Regime Selector**: Auto-populates problem regime (`Globular`, `IDP (13C-Detected)`, `Paramagnetic`) from `nmr-core::classify` (`2D CON` / `CACO` / `CANCO` pulse programs $\to$ `IDP (13C)`, metal/paramagnetic cues $\to$ `Paramagnetic`) with manual override in the top toolbar and guided workflow stepper.
- **2D `CON` Root Map & Continuous Proline Ribbon**: In `idp-13c` mode, renders the `2D CON` `(13C', 15N)` plane (highlighting the `>= 131.5 ppm` `Pro` imino region), walks `3D CANCO` / `CBCACON` / `NCACO` strips uninterrupted through `Pro` tiles on the sequence ribbon, and displays per-residue `(SSP, p_alpha, p_beta, p_ppii, p_coil)` hover/panel readouts; in `paramagnetic` mode, badges `PRE`-bleached coordination loops on the sequence ribbon so analysts immediately distinguish physical blind zones from missing data.
