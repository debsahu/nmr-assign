# Changelog

All notable architectural changes, algorithm milestones, and verified benchmark results for `nmr-assign` are documented in this file using Semantic Versioning (`vX.Y.Z` annotated git tags, newest releases first).

---

## [0.7.0] — 2026-11-13 — Phase 11: Protonless `13C`-Detected IDP Solvers, `SSP`/`δ2D` Ensemble Propensities, Paramagnetic `PRE`/`PCS` Metalloprotein Assignment, Non-Circular Real-Data Sidechain/Methyl Validation & LOPO Calibration Hardening (`nmr-core`, `nmr-gpu`, `nmr-cli`, `nmr-gui`, `docs`)

- **Tag:** `v0.7.0`
- **Architectural Summary (`T11.1`–`T11.4` + Phase 10 Start-Up Backlog `P0-2`, `P1-7`, `P1-8`, `P1-10`–`P1-12`, `P2`):**
  - **Protonless `13C`-Detected IDP Experiment Models, Narrow-Dispersion Picker & Volumetric Simulator (`nmr-core::c13_detected`, `docs/specs/14-regime-solvers.md` — `T11.1`):**
    - Implemented 6 protonless `13C'`/`13CA`-detected experiment pathways (`Bermel et al. JACS 2006–2009`; `Felli & Pierattelli Chem. Rev. 2022`): `2D CON` (`C'(i-1)–N(i)`), `2D CACO` (`C'(i)–CA(i)`), `3D CANCO` (`C'(i-1)–N(i)–CA(i)` & `CA(i-1)`), `3D CBCACON` (`C'(i-1)–N(i)–CA/CB(i-1)`), `3D CCC(CO)N` (aliphatic `C(i-1)` relay), and `3D NCACO` (`C'(i)–N(i)–CA(i)`), with dynamic `500–950 MHz` `Hz <-> ppm` conversion (`obs_mhz` from spectrum headers), IPAP `^1J(C',CA) ≈ 53 Hz` antiphase ghost suppression (`<= 0.14` amplitude ratio), and Bai–Englander (`Proteins 1993`) amide exchange attenuation (`0` sequence/prior inputs in `c13_detected.rs`).
    - Volumetric `2D CON` picker recovery on proline-rich real BMRB IDPs (`<= 3%` artifact rate): **`98.54%` (`202/205`)** on `bmr26639` (`CBP-ID4`, `207 aa`, `21.74% Pro`) and **`98.85%` (`86/87`)** on `bmr51984` (`Human p53 TAD-PRR 1–102`, `21.57% Pro`).
  - **Continuous Through-Proline `13C`-Detected IDP Backbone Solver (`nmr-core::idp_solver`, `nmr-gpu::idp_gpu`, `idp_scoring.wgsl`, `nmrassign assign --regime idp-13c` — `T11.2`):**
    - Formulates each `2D CON(i)` root as a dipeptide bond $(i-1, i)$, walking continuously across `Pro` (`δN ≈ 135.2–138.0 ppm` imino cluster) and `Pro-Pro` (`Xaa-Pro` vs `Pro-Pro` covalent priors; `Tamiola, Acar & Mulder JACS 2010` `ncIDP` neighbor corrections, `Platzer 2014` `His`/`Asp`/`Glu` pH titration, `Kjaergaard 2011` $\partial\delta/\partial T$).
    - Strictly unlabelled permutation-invariant (`0.0` max bit-for-bit diff under random peak shuffling) `13C`/`15N` reference offset estimation and unsupervised 2-stage coarse+fine `3D NCACO` `(ΔC', ΔN, ΔCA)` carrier registration, `-6.5` 3D-unsupported noise-root penalty, and length-invariant `p_local` (`MAX_ALT_COMPETITORS = 8`, `TAU_LOCAL = 1.95`) calibrated confidence.
    - Portable `16,384 B` (`16 KiB`) workgroup-SRAM WebGPU kernels (`compute_idp_u`, `compute_idp_p`, `anneal_idp_256`, `anneal_idp_1024`, `anneal_idp_global`) with **`5.72e-6` (`U`) / `3.34e-6` (`P`) CPU-vs-WebGPU parity (`< 1e-4` gate)**; `--engine gpu` fails with an explicit hard error on WebGPU unavailability while `--engine auto` emits a loud `stderr` warning and falls back to CPU.
  - **IDP Secondary Structure Propensity (`SSP` & 4-State `δ2D`) (`nmr-core::idp_propensity`, `nmr-core::report`, `nmrassign propensity`, `nmrassign report --idp-propensity` — `T11.3`):**
    - Implements `Marsh et al. Protein Sci. 2006` (`SSP` 5-residue SD-weighted `(ΔCA - ΔCB)` score with `Gly` `ΔCA`-only fallback) and a clean-room Forward–Backward HMM 4-state `δ2D`-family conformational ensemble estimator `(p_alpha, p_beta, p_ppii, p_coil)` (`Camilloni et al. Biochemistry 2012`; exact normalization `sum = 1.0 ± 1e-9`), with `< 1e-12` permutation-invariant unlabelled offset & `His`/`Glu`/`Asp` pH auto-sensing, `.str` and `.nef` CLI support, interactive GUI drawer, and stacked 4-state HTML/SVG report track.
  - **Paramagnetic Metalloprotein `PRE` Bleaching & 5-Parameter Traceless `Δχ` `PCS` Tensor Module (`nmr-core::paramagnetic`, `nmrassign assign --regime paramagnetic` — `T11.4`):**
    - Models Solomon (`Phys. Rev. 1955`) $r_{\text{M}}^{-6}$ `PRE` blind spheres (`default = 8.5 Å`) via 3D single-linkage (`<= 8.5 Å`) multi-cluster donor/metal partitioning (`parse_pdb_atom_coords_first_model` filters strictly for open-shell metal elements, ignoring `MSE CA`, `FES S1/S2`, and closed-shell `CA`/`MG`/`ZN`/`LA`/`LU`) or sequence-only motif detection (`Cisd2Fe2S`, `Rieske2Fe2S`, `FerredoxinFeS`, `RubredoxinOrCys4`, `Cys2His2Finger`, `Cys2HisCysFinger`), injecting blind-zone exclusions and penumbra log-survival priors directly into $U_{\text{para}}$ before optimization with post-solve `>= 2`-slot diamagnetic rescue.
    - Fits the 5-parameter symmetric traceless magnetic susceptibility anisotropy tensor $\Delta\boldsymbol{\chi}$ (`Bertini, Luchinat & Parigi Prog. NMR Spectrosc. 2002`, UTS $|\Delta\chi'_{zz}| \ge |\Delta\chi'_{xx}| \ge |\Delta\chi'_{yy}|$) via exact Hestenes Jacobi SVD, analytical $\nabla_{\mathbf{r}}\Delta\delta^{\text{PCS}}$ coordinate-uncertainty GLS + Huber IRLS (`fit_pcs_tensor_robust`), and 8-parameter separable `VarPro` metal-position refinement (`refine_metal_position_and_tensor`).
  - **Phase 10 Review Start-Up Backlog Remediations (`P0-2`, `P1-7`, `P1-8`, `P1-10`–`P1-12`, `P2`):**
    - **`P1-7` & `P1-8` (`qc.rs`, `evo_graph.rs`, `solver.rs`, `nmr-gpu/src/lib.rs`, `real_benchmark.rs`)**: Removed upstream `PASS1_SINGLE_SPIN_KNOTS` from `evo_graph.rs` so Pass-1 uses parameter-free unit-temperature (`T = 1.0` nat) single-spin Boltzmann beliefs with zero cross-fold LOPO leakage (`P1-7a`); exported documented `QC_*` convergence margin constants, `CANONICAL_CALIBRATION_GRID_12` (`g12`), and unified symmetric `Beta(0.25, 0.25)` (`laplace_prior = 0.5`) wording (`P1-7b,c,d`); added dedicated `STANDARD_GPU_CALIBRATION_KNOTS` + `calibrate_standard_gpu_confidence` wired into WebGPU `standard` annealing (`P1-8`).
    - **`P1-10` (`methyl_solver.rs`, `methyl_gpu.rs`, `methyl_anneal.wgsl`, `nmr-gpu/src/lib.rs`)**: Guarded `m < -1 || m >= m_len -> -1.0e12`; added exact 4-move Metropolis–Hastings detailed-balance corrections on CPU & WebGPU (`TVD = 0.0133–0.0168` against exact $5!$-state Boltzmann enumeration); set `hc_total == 0 -> hc_abstained = true, hc_precision_opt = None, hc_precision_pct = 0.0`; verified `MBP` `C2` homodimer (`M = 376` on `anneal_methyl_1024`, `100.0%` orbit acc, `max_conf = 0.50`, `|E_cpu - E_gpu| = 9.16e-7`) & `C6` hexamer (`M = 1,128` on `anneal_methyl_global`, `max_conf = 0.1667 <= 1/6`); chunked GPU dispatches to `<= 4,096` steps per submission for OS TDR safety; added zero-crate `Future`+`Waker` `GpuEngine::read_buffer_vec_async` and explicit `wasm32` `Err` guard on synchronous `read_buffer_vec`.
    - **`P1-11`, `P1-12` & `P2` (`aromatic.rs`, `topology.rs`, `stereospec.rs`, `sidechain_widget.rs`, `nmr-cli`, `nmr-gui`)**: Corrected Pelton (`Protein Sci. 1993`, Table 1) `His` long-range `15N-HMQC` coupling hierarchy (strong `^2J = 8.8–10.6 Hz` to unprotonated pyridine `=N-` `~249.5 ppm` vs smaller `5.2 / 1.4 Hz` to protonated pyrrole `>NH`), added `cg_ppm` (`135.5 / 127.2 / 129.8 ppm`) and `HisStateClassification` (`Resolved`, `FastExchangeUncommitted` `190..=230 ppm`, `MetalCoordinatedCandidate` abstention on `Zn(II)`-`His`, `P1-11`); unified `HisVariantPriors`, added non-destructive `merge_aromatic_assignments_preserving_existing` across library/CLI/GUI, block-Boltzmann multi-competitor `ring_confidence` (`<= 1/K` on $K$-way ties), and unlabelled `ΔCB` offset alignment (`P1-12`); switched `stereospec.rs` to Student-$t$ ($\nu = 4$) `3J` likelihood + quadratic `min_chi2` (`<= 9.0`) outlier guard + tempered Lovell priors + Hu & Bax single-`CG` `3J_CCG`, omitted `HD1`/`HE2` from `HIS` `Canonical`, guarded `compute_atom_bmrb_zscore` against `NaN`/`<= 0` $\sigma$, and separated `auto_inferred_variants` from `user_variant_overrides` in `nmr-gui` (`P2`).
    - **CI Portability (`b7a1039`)**: Replaced `1.6180339` test jitter step with `0.5 * (1.0 + 5.0_f64.sqrt())` for Rust `1.100.0` `clippy::approx_constant` (`GOLDEN_RATIO`) compatibility and widened software-Vulkan (`llvmpipe`/`lavapipe`) GPU test wall-time allowance in `methyl_gpu_tests.rs`.

- **Verified Metrics & Candid Scientific Disclosures:**
  - **1. Non-Circular Real-Data & Cross-Structure Validation of `T10.2`–`T10.5` & `T11.1`–`T11.4` (`P0-2`, `T11`):**
    - **`P0-2` Aromatic (`4` real BMRB entries, `0` calls to `synthesize_aromatic_reference`, 5 seeds, `|ΔCB| = 0.16..0.28 ppm`, `10%` dropout, `9%` clutter noise)**: **`89.77%` (`588/655`) pooled accuracy**, **`81.67%` (`588/720`) recall**, **`95.07%` (`424/446`) HC-precision**, **`ECE = 0.0783`**, `max |δCB_err| = 0.0374 ppm` (`bmr17769` `1D3Z`: `100.00%` [`68/68`] acc / `85.00%` rec / `100.00%` [`66/66`] HC; `bmr4562` `1E8L` Lysozyme: `85.82%` [`357/416`] acc / `77.61%` rec / `93.59%` [`219/234`] HC; `bmr16649` `2KRU` *C. tepidum* `PCP_red`: `95.32%` [`163/171`] acc / `90.56%` rec / `95.21%` [`139/146`] HC + **`4/4` `Zn(II)`-`His52` [`ND1 = 200.88, NE2 = 178.27 ppm`] `MetalCoordinatedCandidate` abstentions**; `mbp_bmr7114` `41`-aromatic negative control: `0` false assignments). *(Note: experimental aromatic peaks are constructed directly from deposited BMRB ring shifts with independent `ΔCB` drift, dropout, and noise rather than raw 2D/3D aromatic FIDs.)*
    - **`P0-2` Stereospecific (`data/fixtures/ubiquitin_exp_3j.tsv`: `92` published Bax-group experimental Ubiquitin `^3J` couplings on `BMRB 5387` vs stereospecific `BMRB 17769` truth)**: **`100.00%` (`16/16` unscrambled; `160/160` across 10 scrambled seeds) committed precision** at **`41.0%` overall / `61.5%` multi-pathway coverage** on both `1D3Z` and CSI-only backbones (with `12/12` clean abstentions on rotamerically averaged surface residues); under mismatched Karplus parameters (`±15%` $A, B$ + `±8°` phase shift), achieves `99.58%` (`958/1,030 = 93.0%` cov) on `1D3Z` and `99.67%` (`910/1,030 = 88.3%` cov) on CSI.
    - **`P0-2` `ILVMAT` Methyl-TROSY Cross-Structure Validation (experimental `BMRB 5387` & `BMRB 7114` methyl shifts, cross-crystal/NMR structures `1D3Z -> 1UBQ` `1.8 Å` X-ray and `1DMB <-> 1OMP` open/closed `MBP`, `σ = 0.18 ppm` `HMCM` jitter, `15%` missing geminal links, `15%` missing NOEs, `10%` spurious NOEs, `< 4.5 Å` spin-diffusion relays)**: `1D3Z -> 1UBQ` **`80.95%` (`34/42`) acc / `92.6%` (`25/27`) CPU & `90.0%` (`27/30`) GPU HC-prec**; `1DMB -> 1OMP` **`88.42%` (`168/190`) CPU / `84.74%` (`161/190`) GPU acc, `95.7%` (`88/92`) CPU / `94.4%` (`85/90`) GPU HC-prec**; pooled (`N = 461`): **`94.3%` (`283/300`) HC-precision**, **`ECE = 0.0570 <= 0.060`**.
    - **`T11.2` Non-Circular 5-Protein Real BMRB `13C`-IDP Benchmark (`bmr6968` `α-Synuclein` `140 aa`, `bmr15176` `DARPP-32` `119 aa` `12.6% Pro`, `bmr4375` `8 M` Urea Ubiquitin `76 aa` `pH 2.0`, `bmr26639` `CBP-ID4` `207 aa` `21.7% Pro`, `bmr51984` `p53 TAD-PRR` `102 aa` `21.6% Pro`; inter-experiment carrier drift, `10%` missing channels, `8%` spurious `CON` noise roots from deposited BMRB shifts)**: **`94.24%`–`99.01%` accuracy**, **`97.01%` (`584/602`) pooled recall vs deposited bonds**, **`99.82%` (`547/548`) pooled HC-precision**, and **`ECE = 0.0231` (CPU) / `0.0227` (WebGPU)** (`<= 0.0490` on every individual protein).
    - **`T11.3` `SSP` / `δ2D` Validation**: **`95.00%` (`38/40`)** folded secondary-structure classification vs `PDB 1D3Z` on `bmr17769` Ubiquitin; `0.908` (`bmr6968`) and `0.868` (`bmr4375`, auto-sensed `pH 2.72`) mean `coil + PPII` fraction with `0` false rigid segments on disordered proteins.
    - **`T11.4` Paramagnetic `PCS` & Real `CISD3` (`BMRB 51439`) Validation**: `< 0.61%` ($\Delta\chi_{\text{ax}}, \Delta\chi_{\text{rh}}$) error across `Tm3+`/`Tb3+`/`Dy3+` under `0.03 ppm` PCS + `0.30 Å` 3D coordinate noise (`0.072 Å` `VarPro` metal recovery from `2.6 Å` offset); on 5-seed real Bruker `CISD3` (`pdata`, tandem `[2Fe-2S]`): **`Standard` improves `53.33% -> 60.61%` (`+7.28 pp`) accuracy / `37.65% -> 39.22%` recall / `100.00%` HC-precision (`0` vs `44` bleached-loop FPs)**, and **`EvoGraph` improves `55.43% -> 62.26%` (`+6.83 pp`) accuracy / `38.04%` recall / `100.00%` HC-precision (`0` vs `22` bleached-loop FPs)**.

  - **2. Backbone Calibration Audit (`P1-7`, `P1-8`) & Candid `standard` WebGPU Heldout Disclosure:**
    - **Zero-Leakage 10-Fold Out-of-Fold LOPO on `dev` (`10` proteins, `CANONICAL_CALIBRATION_GRID_12` `g12`, `Beta(0.25, 0.25)` `laplace_prior = 0.5`)**:
      1. `evo-graph` (parameter-free Pass-1): **10-fold OOF mean per-protein `ECE = 0.0593 <= 0.060`**, **pooled `ECE = 0.0183 <= 0.050`**, **`96.63%` macro HC-precision**; deployed `real-benchmark --split dev`: **mean per-protein `ECE = 0.0544 <= 0.060`**, **pooled `ECE = 0.0159 <= 0.050`**, **`96.8%` macro / `97.3%` (`1783/1833`) micro HC-precision** (`+1.2%` / `+1.3%` over `v0.6.0`). Removing `PASS1_SINGLE_SPIN_KNOTS` (`P1-7a`) carries a small documented Pass-1 anchor cost on globular `CISD3` (`pdata` `58.86% -> 55.43%`, raw FID `65.6% -> 64.4%`, `dev` macro acc `76.1% -> 75.7%`) while improving `heldout` macro accuracy (`55.4% -> 56.3%`) and `heldout` macro HC-precision (`96.6% -> 98.4%`).
      2. `standard` CPU (`STANDARD_CALIBRATION_KNOTS`) & `standard` WebGPU (`STANDARD_GPU_CALIBRATION_KNOTS`): **10-fold OOF `dev` LOPO passes on both engines** (`CPU`: mean `ECE = 0.0522`, pooled `0.0210`, `94.83%` macro HCP; `WebGPU`: mean `ECE = 0.0553`, pooled `0.0192`, `94.99%` macro HCP), and deployed `real-benchmark --split dev --solver standard` (WebGPU) now **passes both `dev` ECE gates** (**mean per-protein `ECE = 0.0486 <= 0.060`** down from `0.0663` in `v0.6.0`; **pooled `ECE = 0.0203 <= 0.050`** down from `0.0477`; `95.2%` macro HC-prec across `8/10` active proteins).
      3. **Candid `heldout` ECE disclosure**:
         - On `heldout`, `evo-graph` passes pooled ECE (**`0.0216 <= 0.050`**, `cold_p10` pooled `ECE = 0.0445 <= 0.050`, `seen_p8_p9` mean `ECE = 0.0444` & pooled `0.0119`), while unweighted 9-protein `heldout` mean per-protein ECE is **`0.0629 > 0.060`** (improved from `0.0713` in `v0.6.0`, missing `0.060` by `0.0029` solely due to single-protein `ECE = 0.240` on Varian `16782`, where QC cleanly abstains `0/0` HC; mean ECE across the 8 HC-active `heldout` proteins is **`0.0407 <= 0.060`**).
         - On `heldout`, **`standard` (WebGPU) still misses the out-of-sample ECE gates (`pooled ECE = 0.0733 > 0.050`, `mean per-protein ECE = 0.1002 > 0.060`)** because WebGPU `standard` is *under-confident* in the `[0.5, 0.8)` bin on `heldout` (`441/541 = 81.5%` empirical accuracy vs `0.667` predicted mean confidence), while `standard` WebGPU `heldout` HC-precision holds at **`96.8%` macro / `95.7%` (`639/668`) micro** (`+41%` more correct HC calls than `v0.6.0`'s `442/453`).

  - **3. Headline Out-of-Sample Generalization (`cold_p10` `3` Uninspected Cold Entries) vs Previously-Evaluated Heldout (`seen_p8_p9` `6` Entries) & `dev` (`10` Entries) (`Q4`):**
    - **`cold_p10` (`3` genuinely cold raw-FID BMRB entries kept 100% untouched per `Q4`: `16072` `SSP0047`, `16782` `SgR171`, `17092` `HR4435B` — headline out-of-sample generalization figure)**:
      - `--solver evo-graph`: **`26.8%` macro accuracy** (`+1.0%` vs `v0.6.0`), **`20.7%` macro recall** (`+0.5%`), **`100.0%` (`77/77`) macro & micro HC-precision** (`+3.1%` macro / `+2.0%` micro vs `v0.6.0`; `2/3` HC-active, **`1` clean QC abstention `0/0` HC on Varian `16782`**), `30.4%` (`261/858`) micro accuracy, **`pooled ECE = 0.0445 <= 0.050`**, `mean per-protein ECE = 0.0997` (`16072`: `28.0%` acc / `24.3%` rec / `100.0%` [`23/23`] HC / `ECE = 0.029`; `16782`: `2.9%` acc / `0.7%` rec / `ABSTAIN` [`0/0`] HC / `ECE = 0.240`; `17092`: `49.6%` acc / `37.1%` rec / `100.0%` [`54/54`] HC / `ECE = 0.030`).
      - `--solver standard` (WebGPU): **`24.9%` macro accuracy**, **`18.9%` macro recall**, **`92.2%` (`118/128`) macro & micro HC-precision** (`1/3` HC-active, `2` abstentions; `17092`: `64.6%` acc / `48.1%` rec / `92.2%` [`118/128`] HC-prec, `+27` correct HC calls over `v0.6.0`), `pooled ECE = 0.0831`, `mean per-protein ECE = 0.1157`.

    | Solver | Cohort / Split (`seeds = 5`) | Macro Acc | Macro Rec | Macro HC-Prec (`active/abs`) | Micro Acc (`corr/ass`) | Micro HC-Prec (`corr/ass`) | Mean Per-Protein ECE | Pooled ECE |
    |---|---|---:|---:|---|---|---|---:|---:|
    | `evo-graph` | **`cold_p10` (`3` genuinely cold)** | **26.8%** | **20.7%** | **100.0%** (`2/1`) | `30.4%` (`261/858`) | **100.0%** (`77/77`) | `0.0997` | **0.0445** |
    | `evo-graph` | `seen_p8_p9` (`6` **previously-evaluated heldout**, not cold: `5845, 17030, 18145, 51351, 16099, 18489`) | `71.0%` | `64.7%` | `97.9%` (`6/0`) | `72.7%` (`1472/2024`) | `97.5%` (`1170/1200`) | **0.0444** | **0.0119** |
    | `evo-graph` | `heldout` Overall (`9` = `3 cold_p10` + `6 seen_p8_p9`) | `56.3%` | `50.0%` | `98.4%` (`8/1`) | `60.1%` (`1733/2882`) | `97.7%` (`1247/1277`) | `0.0629` | **0.0216** |
    | `evo-graph` | `dev` (`10` entries) | `75.7%` | `65.3%` | `96.8%` (`10/0`) | `74.3%` (`2433/3275`) | `97.3%` (`1783/1833`) | **0.0544** | **0.0159** |
    | `standard` (WebGPU) | **`cold_p10` (`3` genuinely cold)** | **24.9%** | **18.9%** | **92.2%** (`1/2`) | `27.9%` (`212/759`) | **92.2%** (`118/128`) | `0.1157` | `0.0831` |
    | `standard` (WebGPU) | `seen_p8_p9` (`6` **previously-evaluated heldout**, not cold) | `62.7%` | `50.7%` | `97.6%` (`6/0`) | `63.2%` (`1140/1803`) | `96.5%` (`521/540`) | `0.0925` | `0.0834` |
    | `standard` (WebGPU) | `heldout` Overall (`9`) | `50.1%` | `40.1%` | `96.8%` (`7/2`) | `52.8%` (`1352/2562`) | `95.7%` (`639/668`) | `0.1002` | `0.0733` |
    | `standard` (WebGPU) | `dev` (`10` entries) | `59.6%` | `44.4%` | `95.2%` (`8/2`) | `56.7%` (`1605/2831`) | `93.3%` (`965/1034`) | **0.0486** | **0.0203** |

  - **4. Mandatory `AGENTS.md` 5-Seed `--realistic` Ubiquitin & `CISD3` (`BMRB 51439`) Regression Gates:**

    | Benchmark (`seeds = 0..4`) | Solver / Engine / Regime | Accuracy | Recall | High-Confidence Precision (`>= 0.80`) | Notes vs `v0.6.0` |
    |---|---|---:|---:|---|---|
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `evo-graph` (CPU, default) | **98.4%** (`314/319`) | **90.8%** | **98.4%** (`299/304`, `86.4%` HC-rec) | Exact `98.4%` acc & HC-prec |
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `standard` (CPU) | **89.2%** (`247/277`) | **71.4%** | **96.8%** (`153/158`, `44.2%` HC-rec) | Exact `89.2%` acc / `71.4%` rec |
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `standard` (WebGPU) | **88.4%** (`243/275`) | **70.2%** | **96.2%** (`175/182`, `50.6%` HC-rec) | `+8.1%` HC-rec (`P1-8` GPU knots) |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `standard` (CPU, globular) | **53.33%** (`[66.7, 52.8, 50.0, 44.4, 52.8]`) | **37.65%** | — | Exact parity |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `standard` (CPU, **`--regime paramagnetic`**) | **60.61%** | **39.22%** | **100.00%** (`0` loop FPs vs `44`) | **`+7.28%` acc, `+1.57%` rec** (`T11.4`) |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `evo-graph` (CPU, globular) | **55.43%** | **38.04%** | **100.00%** (`5/5` seeds `100%`) | `-3.43%` Pass-1 cost (`P1-7a` leak fix) |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `evo-graph` (CPU, **`--regime paramagnetic`**) | **62.26%** | **38.04%** | **100.00%** (`0` loop FPs vs `22`) | **`+6.83%` acc over globular** (`T11.4`) |
    | `CISD3` (`BMRB 51439`, raw FID + 3D NUS) | `evo-graph` (CPU, globular `dev`) | **64.4%** | **40.4%** | **100.0%** (`34/34`) | `-1.2%` Pass-1 cost (`P1-7a`) |
    | `CISD3` (`BMRB 51439`, raw FID + 3D NUS) | `standard` (WebGPU, globular `dev`) | **58.8%** | **36.9%** | **98.1%** (`51/52`, `ECE = 0.052`) | `+14` correct HC calls (`P1-8`) |

---

## [0.6.0] — 2026-11-13 — Phase 10: Covalent Sidechain Z-Score Inspector, Aromatic Ring & Stereospecific Prochiral Solvers, `ILVMAT` Methyl-TROSY Multi-Replica Annealing, 10-Fold LOPO Calibration, Cold Raw-FID Cohort Audit & Pre-Tag Hardening (`nmr-core`, `nmr-gpu`, `nmr-cli`, `nmr-gui`, `docs`)

- **Tag:** `v0.6.0`
- **Architectural Summary (`T10.1`–`T10.5` + Phase 9 Follow-Ups + Pre-Tag Remediations):**
  - **Planar 2D IUPAC Covalent Sidechain Layouts & Interactive Z-Score Inspector (`nmr-core::topology`, `nmr-gui::sidechain_widget`, `nmr-gui::main` — `T10.1`):**
    - Deterministic planar 2D covalent atom-and-bond coordinates across all 20 canonical amino acids plus 5 `Cys`/`His` redox/tautomer states (`CysReduced`, `CysOxidized`, `HisNd1H`, `HisNe2H`, `HisDoublyProtonated`; `min_dist >= 0.35`, `0` bond crossings), wired into `SidechainInspectorPanel` with empirical BMRB Z-score coloring, `_Atom_chem_shift.Ambiguity_code` badges, and 1-click `SyncCrosshair` strip jump (`0` `synthesize_*`/`simulate_*` calls in `nmr-gui`).
  - **Aromatic Ring Spin-System & Pelton `His` Tautomer Solver (`nmr-core::aromatic`, `nmrassign sidechain --aromatic`, `docs/specs/07-sidechain-assignment.md` — `T10.2`):**
    - Links backbone `(CA, CB)` anchors to `Phe`, `Tyr`, `Trp`, and `His` rings via `(HB)CB(CGCD)HD` (`Yamazaki et al. 1993`), `(HB)CB(CGCDCE)HE`, aromatic `13C-HSQC` (`Torizawa et al. 2005`), `3D AROM_HCCH_TOCSY`, `Trp` indole `15NE1-1HE1`, and `His` long-range `15N-HMQC` (`Pelton et al. 1993`).
    - Uses Gaussian / bounded-tail (`>= -8.0` floor) empirical BMRB chemical-shift likelihoods (no structural ring-current shift model in `aromatic.rs`) and exact Kuhn–Munkres global bipartite matching (`hungarian_max_weight`) with exact leave-one-edge-out margins (`hungarian_margins`), counting symmetric `Phe`/`Tyr` ring-flip pairs (`Ambiguity_code = 3`) once per pair.
  - **Bayesian Stereospecific Prochiral Assignment & Degeneracy/Outlier Guards (`nmr-core::stereospec`, `nmrassign sidechain --stereospec`, `docs/specs/07b-stereospecific.md` — `T10.3`, `P1-1`..`P1-4`):**
    - Resolves diastereotopic `CH2` methylenes (`HB2/HB3`, `Gly HA2/HA3`, `Ile HG12/HG13`, `HG*`, `HD*`, `HE*`), `Val CG1/CG2` & `Leu CD1/CD2` isopropyl methyls, and `Asn`/`Gln` primary carboxamides from calibrated Karplus `3J` curves (`Karplus 1963`; `DeMarco 1978`; `Vuister & Bax 1993`; `Hu & Bax 1997`; `Chou, Case & Bax 2003`), analytical NeRF `r^-6` NOE ratios (`0` handedness inversions vs `1D3Z`, `< 0.04 Å` MAE), and Lovell backbone-dependent rotamer priors (weak chemical-shift asymmetry priors apply strictly to `Asn`/`Gln` carboxamides).
    - **Pre-tag stereochemical & contract guards (`P1-1`..`P1-4`, `P1-1R`)**: (1) caps `Gly HA2/HA3` strictly below `0.90` (`<= 0.85`, `Ambiguity_code = 2`) unless $(\phi, \psi)$ comes from a supplied 3D `.pdb` structure (`ExactTorsions`), since `Gly`'s local peptide unit is achiral under $(\phi, \psi) \leftrightarrow (-\phi, -\psi)$ with `HA2 <-> HA3` (`P1-1`, `P1-1R`); (2) enforces the `< 0.90` single-pathway `(rotamer, swap)` degeneracy cap across **all** prochiral pairs including `Ile HG12/HG13` (`P1-2`); (3) rejects non-finite (`NaN`/`inf`), out-of-range (`^3J` outside `[-0.5, 16.0] Hz` or `NOE <= 0`), and high-misfit (reduced $\chi^2 / N_{\text{obs}} > 9.0$) inputs to uncommitted `Ambiguity_code = 2` (`P1-3`); and (4) defaults all unprocessed or half-observed prochiral atoms (e.g. lone `HB2`) to `Ambiguity_code = 2`, deduplicates NEF `x`/`y` wildcard rows on export, and swaps `Val`/`Leu` `13CH3` carbon + proton members (`HG1*`/`QG1`/`MG1`/`HG1%`, `HD1*`/`QD1`/`MD1`/`HD1%`) atomically together or leaves the group uncommitted at `2` (`P1-4`).
  - **Supramolecular `ILVMAT` Methyl-TROSY Picker & Independent Multi-Replica Simulated Annealing Solver (`nmr-core::methyl_trosy`, `nmr-core::methyl_solver`, `nmr-gpu::methyl_gpu`, `nmrassign methyl`, `docs/specs/13-methyl-trosy.md` — `T10.4`, `T10.5`, `P1-6`):**
    - Precursor-aware (`ILV-proS`, `ILV-proR`, `ILV-racemic`, `ILVMAT`) `25 Hz / 18 Hz` pseudo-Voigt simulator/picker (`2D 13C-HMQC`, `3D/4D HMQC-NOESY-HMQC`, `HMCM(CG)CBCA` geminal linking) and structure-based NOE contact-graph solver (`assignment_method = "structure_based_methyl_noe_graph"`).
    - Uses **independent multi-replica simulated annealing (`56` replicas by default in `MethylAnnealConfig`, `64 × seeds` in CLI `nmrassign methyl`, zero inter-replica exchange)** across portable `16,384 B` (`16 KiB`) WebGPU workgroup SRAM (`1.20e-6` Ubiquitin / `3.45e-6` `MBP` per-root normalized CPU/WebGPU objective parity) with **automatic multi-threaded CPU fallback** (emitting a warning) when no WebGPU adapter is available (`P1-6`).
  - **Pre-Tag CLI, Archive & HTTP Security Hardening (`P0-1`, `P1-5` `N2`/`N3`, `#162`, `N4`, `Q1b`, `P2-4`/`P2-5`):**
    - **`P0-1` (`nmrassign sidechain`)**: Synthetic simulation runs **only** when `--simulate` (or `--benchmark`) is explicitly passed; omitting peak/coupling/NOE inputs without `--simulate` errors cleanly instead of fabricating shifts, `--stereo-threshold` is validated to `(0.5, 1.0]`, and 100% of untouched input NMR-STAR shifts/atoms (`Phe45 CZ/HZ`, `Lys63 H/N`, existing ring shifts) round-trip verbatim.
    - **`P1-5` (`N2` & `N3`), `#162`, `N4` (`classify.rs`, `service.rs`)**: Tolerates trailing `0x00` zero-padding blocks after valid gzip member trailers on `.tgz`/`.tar.gz` archives (`N2`); validates HTTP `Host` against loopback (`localhost`, `127.0.0.1`, `[::1]` on the bound port) or explicit `--cors-origin` before accepting same-origin `Origin == http(s)://{Host}` to block DNS-rebinding attacks (`403 Forbidden`, `N3`); streams up to `16 GiB` of discarded `ser`/`fid` payload bytes with `64 MiB` max retained header RAM in local/WASM drag-and-drop vs `(128 MiB, 32 MiB)` on untrusted HTTP `/api/v1/inspect` (`#162`), with `< 50 ms` classification applying to unpacked folders and seekable `.tar` archives (`N4`).
    - **Backbone QC, Minor-Conformer Audit & NEF Hardening (`P1-4`, `P1-5`, `Q1b`, `P2-4`/`P2-5`)**: Unified solver and `DatasetQcReport` on Huber-bounded joint `(dCA, dCB)` offset estimation (`NotSignificant`/`TooFewValues` never treated as `0.0 ppm`, restoring TMS-shifted `15476` & `16385` to `QC=PASS`); switched minor-conformer damping to integrated `Peak.volume` first (`5/683 = 0.73%` `dev` true roots have `r_vol < 0.15`, `25/25 = 100.0%` 5-seed accuracy, `+1.35%` `dev` micro accuracy); classified `C13Hmqc` from `13C` carrier (`8..=28 ppm`) + `SW <= 36 ppm`; hardened NEF `nmrassign_workbench` `;...;` multiline serialization and `standard_shift_crc32` external-edit adoption.

- **Verified Metrics & Candid Scientific Disclosures:**
  - **1. Synthetic Self-Consistency Disclosure for `T10.2`–`T10.5` (`P0-2`, `P1-9a/b/c/f`):**
    - All Phase 10 aromatic (`T10.2`), stereospecific (`T10.3`), and methyl-TROSY (`T10.4`–`T10.5`) accuracy/precision figures below are **synthetic self-consistency benchmarks**: the simulated peaks/couplings/NOEs are generated from the same forward statistical/physical models the solvers invert (BMRB/Pelton shift priors + `0.02–0.035 ppm` Yamazaki `CB` jitter for aromatic; identical Karplus curves + single-conformer `1D3Z` coordinates for stereospecific; identical `PDB` `dist_cc` matrix + `3.8 Å` saturation radius + noise-free reference `HMCM(CG)CBCA` carbons for methyl) and **have not yet been validated on experimental aromatic, stereospecific `3J`, or methyl NOESY peak lists** (tracked for start of Phase 11):
      - **`T10.2` Aromatic (`12` BMRB backbones, synthetic self-consistency volumetric suite)**: **`95.07%` (`405/426`) accuracy**, **`85.1%` (`405/476`, `85.08%`) volumetric recall**, **`98.48%` (`323/328`) HC-precision**, and **`100.0%` (`18/18`) Pelton `HisTautomer` classification**.
      - **`T10.3` Stereospecific (`1D3Z` Ubiquitin, 10-seed noisy scrambled synthetic self-consistency, post-guard `P1-9R`)**: **`100.00%` committed precision** at `posterior >= 0.90` (`959/959` across `1,030` methylene + `Val`/`Leu` pairs = **`93.1%` coverage** on `1D3Z` `(φ,ψ)`, `908/908` = **`88.2%` coverage** on CSI `(φ,ψ)`), and **`0 / 1,665` (`0.00%`)** false `Ambiguity_code = 1` swaps on 15-seed isotropic null inputs.
      - **`T10.4`–`T10.5` `ILVMAT` Methyl-TROSY (independent multi-replica SA: `56` default / `64 × seeds` CLI, synthetic self-consistency)**: `25/18 Hz` picker recovery `100.00%` on 2D/3D/4D Ubiquitin & `99.87%` (`772/773`) on 3D `MBP` (`<= 2.77%` artifacts); WebGPU **`93.62%` (`176/188`) accuracy / `100.00%` (`90/90`) HC-precision in `1.53 s`** on `41 kDa` `MBP` `1EZP` (`95.21%` acc / `100.00%` [`91/91`] HC-prec on CPU) and **`92.86%` (`39/42`) accuracy / `100.00%` (`35/35`) HC-precision in `0.55 s`** on Ubiquitin `1D3Z`.

  - **2. Backbone Calibration & Candid `standard` WebGPU ECE Disclosure (`P1-8`, `P1-7a/c/d`):**
    - Replaced hand-tuned scalar `T_cal` (`1.65`/`1.68` and `1.55`) with 10-fold out-of-fold Leave-One-Protein-Out (`LOPO`) monotone PAV piecewise-linear maps (`EVO_GRAPH_CALIBRATION_KNOTS`, `STANDARD_CALIBRATION_KNOTS`) on the 10-protein `dev` split.
    - **Candid methodology & engine-gap notes**:
      1. `evo-graph` passes 10-fold OOF `dev` LOPO (**`mean per-protein ECE = 0.0583 <= 0.060`**, **`pooled ECE = 0.0258 <= 0.050`**, `[0.8, 0.9)` **`86.9%` empirical vs `0.866` predicted**, `95.67%` macro / `95.96%` [`1830/1907`] micro HC-prec) and passes out-of-sample `heldout` pooled ECE (**`0.0201 <= 0.050`**, `[0.8, 0.9)` **`86.3%` [`132/153`] vs `0.826`**, down from `0.073` pooled ECE in `v0.5.0`), while unweighted 9-protein `heldout` mean per-protein ECE is **`0.0713 > 0.060`** due to single-protein `ECE = 0.229` on Varian `16782` (`35` low-confidence calls across 5 seeds with clean `0/0` HC `ABSTAIN`; mean ECE across the other 8 `heldout` entries is `0.0516 <= 0.060`).
      2. **`standard` on the deployed WebGPU path misses the ECE gates**: `STANDARD_CALIBRATION_KNOTS` was fitted on CPU `standard` outputs (`10-fold OOF LOPO mean ECE = 0.0492 <= 0.060`, `pooled ECE = 0.0120 <= 0.050`, `94.34%` macro HC-prec), whereas `real-benchmark --solver standard` runs WebGPU (baseline local/block MCMC proposals rather than CPU's sparsity-adaptive proposals), yielding **`dev` mean per-protein `ECE = 0.0663 > 0.060`** (`pooled = 0.0477`), **`heldout` pooled `ECE = 0.0701 > 0.050`**, **`heldout` mean per-protein `ECE = 0.0883`**, and under-confident `[0.5, 0.8)` (`75.3%` [`686/911`] observed vs `0.610` predicted), though `standard` WebGPU macro HC-precision holds at `95.7%` (`dev`) and `97.8%` (`heldout`).
      3. `CANONICAL_GRID` (`g12`, 13 knots) was fixed after earlier grid exploration on `dev`, uses a symmetric `Beta(0.25, 0.25)` (`0.5 * w` total pseudocount, `CANONICAL_LAPLACE = 0.5`) prior, and `evo-graph` Pass-1 uses a fixed pre-Pass-2 table (`PASS1_SINGLE_SPIN_KNOTS`) fitted on `dev` upstream of `p_raw` (in-fold Pass-1 fitting and per-engine `standard` GPU knots are tracked for Phase 11).

  - **3. Headline Out-of-Sample Generalization (`cold_p10` `3` Cold Entries) vs Previously-Evaluated Heldout (`seen_p8_p9` `6` Entries) & `dev` (`10` Entries) (`Q4`, `P1-9e`):**
    - **`cold_p10` (`3` genuinely cold raw-FID BMRB entries: `16072` `SSP0047`, `16782` `SgR171`, `17092` `HR4435B` — headline out-of-sample generalization figure)**:
      - `--solver evo-graph`: **`25.8%` macro accuracy**, **`20.2%` macro recall**, **`96.9%` macro / `98.0%` (`99/101`) micro HC-precision** (`2/3` HC-active, **`1` clean QC abstention `0/0` HC on Varian `16782`**), `29.6%` (`257/868`) micro accuracy, `18.0%` (`257/1425`) micro recall, `pooled ECE = 0.0258`, `mean per-protein ECE = 0.1189` (`16072`: `28.4%` acc / `25.1%` rec / `93.8%` [`30/32`] HC-prec; `16782`: `2.9%` acc / `0.7%` rec / `ABSTAIN` [`0/0`] HC; `17092`: `46.0%` acc / `34.8%` rec / `100.0%` [`69/69`] HC-prec).
      - `--solver standard` (WebGPU): **`24.9%` macro accuracy**, **`18.9%` macro recall**, **`100.0%` (`91/91`) macro & micro HC-precision** (`1/3` HC-active, `2` abstentions; `17092`: `64.6%` acc / `48.1%` rec / `100.0%` [`91/91`] HC-prec), `pooled ECE = 0.0410`, `mean per-protein ECE = 0.0832`.

    | Solver | Cohort / Split (`seeds = 5`) | Macro Acc | Macro Rec | Macro HC-Prec (`active/abs`) | Micro Acc (`corr/ass`) | Micro HC-Prec (`corr/ass`) | Mean Per-Protein ECE | Pooled ECE |
    |---|---|---:|---:|---|---|---|---:|---:|
    | `evo-graph` | **`cold_p10` (`3` genuinely cold)** | **25.8%** | **20.2%** | **96.9%** (`2/1`) | `29.6%` (`257/868`) | **98.0%** (`99/101`) | `0.1189` | **0.0258** |
    | `evo-graph` | `seen_p8_p9` (`6` **previously-evaluated heldout**, not cold: `5845, 17030, 18145, 51351, 16099, 18489`) | `70.2%` | `64.0%` | `96.6%` (`6/0`) | `72.0%` (`1455/2021`) | `96.6%` (`1231/1274`) | `0.0474` | `0.0181` |
    | `evo-graph` | `heldout` Overall (`9` = `3 cold_p10` + `6 seen_p8_p9`) | `55.4%` | `49.4%` | `96.6%` (`8/1`) | `59.3%` (`1712/2889`) | `96.7%` (`1330/1375`) | `0.0713` | `0.0201` |
    | `evo-graph` | `dev` (`10` entries, incl. promoted `15476, 16385, 53280`) | `76.1%` | `65.6%` | `95.6%` (`10/0`) | `74.7%` (`2450/3279`) | `96.0%` (`1877/1955`) | `0.0545` | `0.0241` |
    | `standard` (WebGPU) | **`cold_p10` (`3` genuinely cold)** | **24.9%** | **18.9%** | **100.0%** (`1/2`) | `27.9%` (`212/759`) | **100.0%** (`91/91`) | `0.0832` | `0.0410` |
    | `standard` (WebGPU) | `seen_p8_p9` (`6` **previously-evaluated heldout**, not cold) | `62.7%` | `50.7%` | `97.4%` (`5/1`) | `63.2%` (`1140/1803`) | `97.0%` (`351/362`) | `0.0908` | `0.0891` |
    | `standard` (WebGPU) | `heldout` Overall (`9`) | `50.1%` | `40.1%` | `97.8%` (`6/3`) | `52.8%` (`1352/2562`) | `97.6%` (`442/453`) | `0.0883` | `0.0701` |
    | `standard` (WebGPU) | `dev` (`10` entries) | `59.6%` | `44.4%` | `95.7%` (`7/3`) | `56.7%` (`1605/2831`) | `94.1%` (`811/862`) | `0.0663` | `0.0477` |

  - **4. Mandatory `AGENTS.md` 5-Seed `--realistic` Ubiquitin & `CISD3` (`BMRB 51439`) Regression Gates:**

    | Benchmark (`seeds = 0..4`) | Solver / Engine | Accuracy | Recall | High-Confidence Precision (`>= 0.80`) | Notes vs `v0.5.0` |
    |---|---|---:|---:|---|---|
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `evo-graph` (CPU, default) | **98.4%** (`314/319`) | **90.8%** | **98.4%** (`310/315`, `89.6%` HC-rec) | `+1.7%` HC-rec |
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `standard` (CPU) | **89.2%** (`247/277`) | **71.4%** | **96.9%** (`156/161`) | `+0.1%` HC-prec |
    | Ubiquitin `--realistic` (`synthetic_ubiquitin.str`) | `standard` (WebGPU) | **88.4%** (`243/275`) | **70.2%** | **98.0%** (`147/150`, `42.5%` HC-rec) | `+2.1%` HC-prec |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `standard` (CPU) | **53.33%** (`[66.7, 52.8, 50.0, 44.4, 52.8]`) | — | — | Exact parity |
    | `CISD3` (`BMRB 51439`, TopSpin `pdata`) | `evo-graph` (CPU) | **58.86%** | **40.39%** | **100.00%** (`5/5` seeds `100%`) | `+2.29%` acc, `+1.57%` rec |
    | `CISD3` (`BMRB 51439`, raw FID + 3D NUS) | `evo-graph` (CPU) | **65.6%** | **41.2%** | **100.0%** (`50/50`, `ECE = 0.059`) | `+3.1%` acc vs `v0.5.0` `62.5%` |
    | `CISD3` (`BMRB 51439`, raw FID + 3D NUS) | `standard` (WebGPU) | **58.8%** | **36.9%** | **100.0%** (`37/37`, `ECE = 0.052`) | `+24.0%` HC-prec vs `v0.5.0` `76.0%` |

---

## `[0.5.0]` — 2026-11-12 — Phase 9: Zero-Friction Workspace Classifier, Drag-and-Drop Ingest, Guided Workbench UX, Multi-Spectrum GPU Overlays, Lossless `NEF v1.1` Sessions, Solver Curation & HTTP Security Hardening (`nmr-core`, `nmr-gpu`, `nmr-cli`, `nmr-gui`, `docs`)

- **Tag:** `v0.5.0`
- **Architectural Summary:**
  - **Spectrometer Folder & Archive Auto-Classifier + HTTP `/api/v1/inspect` Security Hardening (`nmr-core::classify`, `nmr-core::service`, `nmr-cli::inspect_cmd`, `docs/specs/12-workbench-ux.md` — `T09.1`, `P1-1`, `P2-2`, `P2-3`):**
    - Added `classify_nmr_workspace` (native directories, files, `.tar`, `.tar.gz`/`.tgz`) and `wasm32`-compatible `classify_nmr_entries` / `extract_archive_virtual_entries` covering 25 `PulseSequenceClass` variants and 7 `ProblemRegime`s without loading binary signal payloads (`< 50 ms` on unpacked folders and seekable `.tar` streams; compressed `.tar.gz` archives inflate sequentially with IEEE 802.3 CRC-32 + `ISIZE` trailer verification, POSIX/GNU `ustar` header checksum checks, strict `ustar\0` prefix checking, LZ77 distance bounds, and a `128 MiB` in-memory decompression cap).
    - Added `nmrassign inspect <path> [--json]` and hardened `POST /api/v1/inspect` + `nmrassign serve`: removed wildcard `Access-Control-Allow-Origin: *` (enforcing same-origin by default with optional `--cors-origin <origin>`), rejected non-JSON `Content-Type` on JSON `POST` routes (`415 Unsupported Media Type`, blocking `text/plain` browser simple-request CSRF), and disabled server-side `"path"` inspection by default (`403 Forbidden`) unless `--inspect-root <dir>` is explicitly configured and matches canonicalized prefix.
  - **Drag-and-Drop Loader & Editable Experiment Selection Matrix (`nmr-gui::ingest_panel` — `T09.2`):** Native path + `wasm32` byte-slice drag-and-drop for folders, `.tar`/`.tar.gz` archives, and `.nef` sessions, with auto-excluded 1D calibration decoys, per-row `PulseSequenceClass` override dropdowns, sequence extraction from `.fasta`/`.str`/`.nef`, and 1-click **"Process, Pick & Run"**.
  - **Guided 6-Stage Workflow Stepper, 9-Menu Bar, `< 1 ms` `Cmd/Ctrl+K` Palette & Modal Two-Key Chords (`nmr-gui::workflow_bar`, `command_palette`, `keymap` — `T09.3`, `T09.4`):** 75 typed `WorkbenchAction` commands across 9 menus, persistent 6-stage workflow stepper + `DatasetQcVerdict` pill + contextual **"What Next?"** action, fuzzy command palette, and 14 clean-room two-key NMR chord accelerators (`vt`, `pa`, `pd`, `pc`, `pl`, `oz`, `zi`, `zo`, `ct`, `cb`, `st`, `at`, `Space`, `?`) with text-input and Octet 2D focus suppression.
  - **Multi-Spectrum WebGPU Contour Overlays & `i-1 / i / i+1` Candidate Match-Finder Strips (`nmr-gui::contour_painter`, `strip_tiler` — `T09.5`):** Alpha-blended `OverlayLayerSpec` compositor (`>= 60 fps` cached `f32` storage buffers, independent pos/neg RGBA palettes, `ppm_nudge`, `SyncCrosshair`) and `pair_matrix`-ranked `i-1 / i / i+1` strip columns with `CA`/`CB`/`C'` alignment guides (`98.41%` = `124/126` top-3 sequential neighbor recovery on real Ubiquitin `.ucsf`).
  - **Bounded `256`-Step Undo/Redo & Lossless `NEF v1.1` Session Persistence (`nmr-core::session`, `nef`, `nmr-gui::session_io` — `T09.6`):** Algebraic `CommandHistory` (`apply -> undo == identity` across 16-seed $\times$ 200-step property tests) and single-file `.nef` save/load preserving standard `NEF v1.1` saveframes plus vendor `save_nmrassign_workbench` state.
  - **Backbone Solver Curation (`nmr-core::spinsystems`, `solver`, `evo_graph`, `nmr-gpu` — `T09.7`):**
    - Added hard `locked_assignments: Vec<(usize, usize)>` across CPU & WebGPU `standard` (`max |U_cpu - U_gpu| < 1e-4`) and `evo-graph` (`0` changes to `docs/specs/03-assignment.md` §§3–4; bit-identical `U` & `P` when locks are empty).
    - Added `< 15%`-median sub-stoichiometric minor-conformer damping in `evo_graph` (resolving Ubiquitin `Gln2` 9% minor root on Bruker `pdata` and `.ucsf`; scheduled for volume-based refinement and `dev` weak-amide audit in Phase 10).
    - Added sparsity-adaptive MCMC proposals in CPU `solver.rs` (exact `MoveHastings` proposal ratios; `reservoir_weight` targets $\pi'(x) \propto \exp(U/T)\prod_{s\text{ unassigned}} w(s)$ while returning best-$U$; note CPU and WebGPU `standard` now differ by proposal algorithm).
- **Verified Metrics & Honest Out-of-Sample Evaluation:**
  - **Realistic Bruker 5-Seed Benchmark Gate (`nmrassign benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5`):**
    - `--solver evo-graph` (default): **`98.4%` accuracy** (`314/319`, `0.0%` delta), **`90.8%` recall** (`-0.5%` vs `v0.4.0` `91.3%`), **`98.4% (306/311)` HC precision** (`0.0%` delta).
    - `--solver standard --engine cpu`: **`89.2%` accuracy** (`+5.8%` vs `83.4%`), **`71.4%` recall** (`+6.1%`), **`96.8%` HC precision** (`+0.6%`).
    - `--solver standard --engine gpu`: **`88.4%` accuracy** (`243/275`), **`70.2%` recall**, **`95.9%` HC precision**.
  - **Real Bruker Ubiquitin `pdata` & `CISD3` (`BMRB 51439`) 5-Seed Gates:**
    - Ubiquitin `pdata` (`evo-graph`): **`98.51%` accuracy** (`+4.48%` vs `94.03%`, +3/67 residues), **`94.29%` recall** (`+4.29%`), **`100.00% (61/61)` HC precision**.
    - `CISD3` (`BMRB 51439`, missing `HNCACB`): `standard` CPU `pdata` **`53.33%` accuracy** (`+5.55%` vs `47.78%`); `evo-graph` `pdata` **`56.57%` accuracy / `100.00%` HC precision** (`62.5%` acc / `100.0%` HC-prec on raw FID + 3D NUS).
  - **7-Entry `--split dev --seeds 5` (`--solver evo-graph`):**
    - Macro **`70.7%` acc** (`+1.3%` vs `v0.4.0` `69.4%`), **`58.7%` rec** (`+1.3%`), **`94.2%` HC-prec** (`+0.1%`), **`47.2%` HC-rec** (`+2.2%`); Micro **`69.5%` acc (`1629/2345`)**, **`94.6%` HC-prec (`1273/1345`)**.
    - Calibration (`dev`, via hand-tuned `T_cal = 1.65 / 1.68` in `evo_graph.rs` and `1.55` in `solver.rs`): mean per-protein `ECE = 0.059`, pooled `ECE = 0.037`, `[0.8, 0.9)` bin `77.1%` (`108/140`). Note: `standard` CPU macro HC-prec on `dev` is `73.6%` (`87.6%` micro) and requires dedicated calibration in Phase 10.
  - **9-Entry `--split heldout --seeds 5` (`--solver evo-graph`; `2` Genuinely Cold Entries + `7` Shared with Phase 8):**
    - Macro **`79.1%` acc**, **`71.6%` rec**, **`95.6%` HC-prec** across 7 HC-active proteins (`2` abstained), **`44.7%` HC-rec** (down from `49.4%` in Phase 8); Micro **`80.0%` acc (`2327/2907`)**, **`94.7%` HC-prec (`1442/1522`)**, `[0.9–1.0]` bin **`97.0% (1335/1376)`**.
    - **Out-of-sample calibration (`ECE <= 0.06` gate FAILED on `heldout`)**: mean per-protein **`ECE = 0.095`**, pooled **`ECE = 0.073`**, `[0.8, 0.9)` bin **`73.3% (107/146)`**.
    - **2 genuinely cold entries**: `16099` (Kazal-1, 74 aa, Bruker) **`87.1%` acc / `81.8%` rec / `93.2%` HC-prec** (`87.7%` acc on `standard` CPU); `18489` (NFU1, 107 aa, Bruker) **`55.4%` acc / `42.7%` rec / `98.0% (98/100)` HC-prec** (`ECE = 0.059`).
    - **7 shared Phase 8 $\to$ Phase 9 entries (reporting QC abstention regression plainly)**:
      - `15476` (SR384): HC-prec `97.1%` $\to$ **abstained** (`91.9%` acc; false pre-solver QC offset mismatch warning dropped `170/170` correct calls below `0.80`)
      - `16385` (PfR193A): HC-prec `96.9%` $\to$ **abstained** (`91.9%` acc; false pre-solver QC offset mismatch warning dropped `407/430` correct calls below `0.80`)
      - `17030` (HR4694F): HC-prec `94.8%` $\to$ `91.0%` (`83.4%` acc)
      - `51351` (TSR3): acc `87.7%` $\to$ `85.3%` (`98.4%` HC-prec)
      - `53280` (Ap3): HC-prec `82.1%` $\to$ `91.2%` (`84.6%` acc; not cold — motivated Phase 8 post-review `qc.rs`)
      - `5845` (ZR31): acc `90.7%` $\to$ `91.8%` (`97.4%` HC-prec)
      - `18145` (OR126): acc `40.5%` $\to$ `40.8%` (`100.0%` HC-prec)

---

## `[0.4.0]` — 2026-11-12 — Phase 8 Post-Review Remediation: Dataset QC Calibration, FID/DSP Hardening, Detailed Offset States, 3D-to-HSQC Registration, `evo-graph` Default Solver, and Expanded Benchmark Governance (`nmr-core`, `nmr-cli`, `nmr-gui`, `docs`)

- **Tag:** `v0.4.0`
- **Architectural Summary:**
  - **Dataset Quality Control & Per-Protein Confidence Calibration (`nmr-core::qc`, `nmr-core::priors`, `nmr-cli`):** Introduced `DatasetQcReport` (`PASS`/`WARN`/`ABSTAIN` verdict with `confidence_cap`, `connectivity_frac`, `detailed_offsets`, and `reasons`), `ProteinReliabilityCurve` (4-bin `[0.0–0.5, 0.5–0.8, 0.8–0.9, 0.9–1.0]` reliability + ECE + per-protein HC precision distinguishing `ABSTAIN` from `0.0%`), and `DetailedReferenceOffsets` (per-nucleus `OffsetFitStatus`: `Fitted(ΔLL)` / `NotSignificant` / `TooFewValues`) wired into `nmrassign assign`, `pipeline`, `score`, and `real-benchmark` (`P1-1`, `P1-3`).
  - **Bruker XWIN-NMR & Varian 3D FID/DSP Hardening (`nmr-core::fid`, `nmr-core::ops`, `nmr-core::process`):** Fixed zero-phase fallback on unprocessed XWIN-NMR `pdata/1` (`FT_mod=0`, `PH_mod=0`); indirect `15N` carrier synchronization via `resolve_bruker_indirect_sfo1` when `acquNs.SFO1` holds stale/template values; `Echo-Antiecho` (`FnMODE=6`) indirect `REVERSE` inferred from `(d0..d20)/(d10..d30)` pulse program loops; Varian `origfid` backup preference over `mungefid`-overwritten `fid`; `conjugate_for_fft` derived from `fcoef` Re/Im sign product; indirect-focused 3D direct `1H` blind autophasing (`focused_direct_traces_3d`); aliphatic `13CA/CB` carrier-mirror recovery (`is_mirrored_aliphatic_cacb_3d`). `BMRB 6318` recovered from `0.0%` → `64.2%` accuracy; `BMRB 5329` recovered from `15.3%` → `65.5%` accuracy (`P1-2`).
  - **Hypercomplex NUS IST/FISTA Precondition & HNCACB Polarity Lock (`nmr-core::ops`, `nmr-core::process`):** Enforced `States` recombination precondition before 1D/2D/3D `IST`/`FISTA` (`Err` on `states_recombined == false` or `QF`); dynamic stopping floor `2 * σ_res` (iteration residual MAD); restricted `first_scale = 0.5 / 2.0` to Virtual-Echo mode (fixing `10x` DC impulse artifact); O'Donoghue-Candes adaptive restart + active-support debiasing. Narrowed biphasic positive `CA` window from `48..66 ppm` to `48..60 ppm` (excluding `Ser`/`Thr` `CB(−)` at `60..66 ppm`), using capped-SNR (`≤ 8 σ`) local-extremum peak votes (`Q1a`, `Q1b`).
  - **3D-to-HSQC Rigid HN Registration (`nmr-core::process`, `nmr-cli`):** Added `register_3d_hn_to_hsqc_2d` (unlabelled 2D sky-projection capped-SNR bilinear cross-correlation) to align each 3D `(1H, 15N, 13C)` spectrum to the reference 2D HSQC before picking, correcting small rigid `(dH, dN)` carrier/temperature drift across multi-spectrometer experiment bundles (conservative bounds `max_dH = 0.08 ppm`, `max_dN = 0.60 ppm`). Wired into `nmrassign pick`, `pipeline`, and `real-benchmark`.
  - **`evo-graph` Default Solver (`nmr-cli`, `nmr-gui`):** Changed the default `--solver` for `assign`, `pipeline`, `real-benchmark`, and the GUI solver dropdown to `evo-graph` (keeping `--solver standard` explicitly selectable). Added CLI unit test `test_cli_default_solver_is_evo_graph` and GUI unit test `test_gui_default_solver_is_evo_graph` (`P1-6`).
  - **POSIX USTAR Tar Header Checksum Validation (`nmr-cli::fetch_bmrb`):** Replaced the weak `Content-Length % 512 != 0` tar corruption heuristic with actual 512-byte POSIX USTAR tar header validation (unsigned + signed byte-sum checksum with `header[148..156]` filled with `0x20`), falling back to unpacked directory crawling when a `.tar` fails checksum validation regardless of Content-Length alignment (`P2-4`).
  - **Benchmark Governance Update (`data/benchmark/real_manifest.tsv`):** Promoted `BMRB 5329` (WR41) and `BMRB 6318` (At3g51030) from `heldout` to `dev` after the single Phase 8 cold held-out run (per external review `Q4`). Added 2 pre-screened disjoint backup entries `BMRB 16099` (Kazal-1, 74 aa, 800 MHz Bruker) and `BMRB 18489` (NFU1, 107 aa, 800 MHz Bruker) to `heldout`. Manifest now 16 entries: 7 `dev` / 9 `heldout`. The refreshed `heldout` split stays 100% cold until Phase 9 (`Q4`).
  - **Echo-Antiecho & FnMODE Documentation (`docs/specs/11-fid-processing.md`):** Corrected separate hypercomplex channel equations (`S_cos = (S+ + S−)/2`, `S_sin = (S+ − S−)/(2i)`), documented complete Bruker `FnMODE` table (0=Undefined→rejected, 1=QF, 2=QSEQ→rejected, 3=TPPI→rejected, 4=States, 5=States-TPPI, 6=Echo-Antiecho) (`P2-1`, `P2-2`).
  - **TopSpin Ubiquitin Parity & HNCO/HN(CA)CO Peak Gap Documentation (`docs/specs/11-fid-processing.md`, `TODO.md`):** Documented raw-FID vs TopSpin `pdata/1` HNCO peak difference (`99 vs 140`): TopSpin `ME_mod = LPfc` doubles effective `13C`/`15N` indirect aperture; `HN(CA)CO` (`31 vs 71`): weak 2-way `1J_NCa`/`2J_N_Ca` transfer at `NS = 8` + `LPfc` SNR gain. Ubiquitin parity: `63/66` (`95.5%`) raw vs `63/67` (`94.0%`) TopSpin — both recover 63 correct residues (`P1-4`).
- **Verified Metrics:**
  - **Realistic Bruker 5-Seed Benchmark Gate (`nmrassign benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5`):**
    - `--solver standard --engine cpu`: **`83.4%` accuracy** (`226/271`), **`65.3%` recall**, **`96.2% (154/160)` HC precision** (`0.33 s`).
    - `--solver standard --engine gpu`: **`88.4%` accuracy** (`243/275`), **`70.2%` recall**, **`95.9% (140/146)` HC precision** (`7.07 s`).
    - `--solver evo-graph` (default): **`98.4%` accuracy** (`316/321`), **`91.3%` recall**, **`98.4% (308/313)` HC precision** (`0.76 s`).
  - **Development `CISD3` (`BMRB 51439`, 900 MHz Bruker, 93 aa, 5-Seed, `--solver evo-graph`):**
    - **`62.5%` accuracy**, **`39.2%` recall**, **`100.0%` HC precision** (59/59), unlabelled offset `dN=−0.97, dCA=−2.56, dCB=−2.53, dC=−2.66 ppm`.
  - **7-Entry `--split dev --seeds 5` (`--solver evo-graph`, NEVER ran `--split heldout`):**
    - `ubiquitin_local` (Ubiquitin, 76 aa, Bruker, QC=PASS): **`92.3%` acc**, `85.7%` rec, **`96.6%` HC-prec**.
    - `51439` (CISD3, 93 aa, Bruker, QC=PASS): **`62.5%` acc**, `39.2%` rec, **`100.0%` HC-prec**.
    - `53371` (hDVL2_PDZ, 102 aa, Bruker, QC=PASS): **`71.2%` acc**, `46.9%` rec, **`93.6%` HC-prec**.
    - `18465` (OR135, 110 aa, Varian, QC=WARN): **`44.7%` acc**, `42.8%` rec, **`84.4%` HC-prec**.
    - `15386` (StR82, 77 aa, Bruker, QC=PASS): **`85.1%` acc**, `86.4%` rec, **`96.3%` HC-prec**.
    - `5329` (WR41, 94 aa, Varian, QC=PASS): **`65.5%` acc**, `50.7%` rec, **`91.8%` HC-prec**.
    - `6318` (At3g51030, 124 aa, Bruker, QC=PASS): **`64.2%` acc**, `50.2%` rec, **`96.3%` HC-prec**.
    - **Macro mean**: **`69.4%` acc**, `57.4%` rec, **`94.1%` HC-prec** (HC-active=7/7, abstained=0).
    - **Micro pool**: `68.1%` acc, `56.0%` rec, `94.4%` HC-prec (1219/1291).
    - **Calibration**: worst HC=OR135 `84.4%`, median=`96.3%`, pooled ECE=`0.040`.

---

## `[0.3.0]` — 2026-11-11 — Workspace SemVer Versioning (`v0.3.0`), Realistic Bruker 5-Seed CI Gate, Permutation-Invariant Unlabelled `score` Referencing, and Calibrated `evo-graph` Confidence (`nmr-core`, `nmr-cli`, `nmr-gui`, `ci`, `docs`)

- **Tag:** `v0.3.0`
- **Architectural Summary:**
  - **Single-Source SemVer Versioning & Git Provenance (`Cargo.toml`, `crates/nmr-core/src/lib.rs`, `crates/nmr-cli/build.rs`, `crates/nmr-cli/src/main.rs`, `nef.rs`, `report.rs`, `service.rs`, `nmr-gui`):** Established `[workspace.package] version = "0.3.0"` (`nmr_core::VERSION`) as the single source of truth across `nmrassign --version` / `-V` / `version` (with compile-time short git SHA `NMRASSIGN_GIT_SHA`), generated `.str` provenance headers, CCPN NEF `program_version`, HTML quality reports, `/api/v1/health` JSON, and the desktop GUI title, paired with annotated git tags (`v0.1.0`..`v0.3.0`).
  - **Realistic-Physics 5-Seed Benchmark Gate (`crates/nmr-cli/src/main.rs`, `.github/workflows/ci.yml`, `AGENTS.md`, `crates/nmr-core/tests/evo_graph_tests.rs`):** Replaced the saturated `100.0%` synthetic Ubiquitin CI benchmark gate with `SimConfig::realistic_bruker()` (`--realistic --seeds 5` on `nmrassign benchmark` and `--realistic` on `nmrassign simulate`), enforcing 5-seed accuracy and high-confidence precision gates on both `--solver standard` and `--solver evo-graph` in CI, plus an automatic 5-seed `CISD3` (`BMRB 51439`) regression gate (`real_bruker_cisd3_5seed_development_gate`) whenever `data/private/bmrb51439_cisd3/` is present locally.
  - **Strictly Unlabelled Global Marginal Offset Fitting in `nmrassign score` (`crates/nmr-cli/src/main.rs`):** Replaced the `04b1769` matched-pair `(pred_k, truth_k)` carbon residual median (which inflated `CA`/`CB`/`C` shift agreement by fitting on paired predictions) with `fit_unlabelled_global_offsets` + `--no-auto-offset`: fits one global shift per nucleus `(dH, dN, dCA, dCB, dC)` strictly from sorted unlabelled empirical shift sets via 2D/1D Gaussian log-sum-exp marginal likelihood, verified bit-for-bit invariant (`< 1e-12`) under arbitrary residue-label permutations in `test_score_unlabelled_global_offset_is_permutation_invariant`.
  - **Pass-2 Belief-Damped Fragment Confidence Calibration (`crates/nmr-core/src/evo_graph.rs`, `docs/specs/10-evo-graph-solver.md`):** Added `calibrate_final_marginal_confidence` on Pass 2 combining single-spin Boltzmann margins with quadratic neighbor-belief damping `(b1 / 0.85)^2` and rigid whole-stretch register translation `[a..=b] -> [k2 .. k2 + L]` (`L = 2..=5`) for short stretches lacking a `>= 0.80` boundary anchor, preventing floating short islands (such as chimeric `ARG59`/`ALA62` in `CISD3`) from inflating interior residues above `0.80`.
  - **Documentation & Case-Sensitivity Fixes (`README.md`, `docs/specs/08-service.md`, `docs/specs/10-evo-graph-solver.md`, `crates/nmr-cli/src/main.rs`):** Fixed `.ucsf` pipeline paths in docs and CLI tests to exact uppercase filenames (`HSQC.ucsf`, `HNCACB.ucsf`, `CBCACONH.ucsf`, `HNCO.ucsf`, `HNCACO.ucsf`) for Linux/CI case-sensitive filesystems; corrected the Ubiquitin `090309` (`2009-09-03`) vs `111909` (`2009-11-25/27`) batch interval to `2.8` months (`MM-DD-YY`); used `"minor second form"` without unverified oxidation claims; and reclassified `CISD3` (`BMRB 51439`) as a development benchmark rather than an untouched cold dataset.
- **Verified Metrics:**
  - **Realistic Bruker 5-Seed Benchmark Gate (`nmrassign benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5`):**
    - `--solver standard --engine cpu`: **`85.4%` accuracy** (`234/274`), **`67.6%` recall**, **`93.2% (151/162)` HC precision** (`0.31 s`, CI gate: `>= 82.0%` acc / `>= 90.0%` HC prec).
    - `--solver standard --engine gpu`: **`83.0%` accuracy** (`224/270`), **`64.7%` recall**, **`96.6% (142/147)` HC precision** (`7.02 s`).
    - `--solver evo-graph`: **`98.1%` accuracy** (`314/320`), **`90.8%` recall**, **`98.4% (306/311)` HC precision** (`0.74 s`, CI gate: `>= 95.0%` acc / `>= 96.0%` HC prec).
  - **Development Real Bruker `CISD3` (`BMRB 51439`, `700.13 MHz`, 93 aa, missing `HNCACB`, 51 deposited amides / 38 picked roots, 5-Seed Mean `0..4`, Unlabelled Global `score` Offset `dH=+0.003, dN=-0.97, dCA=-2.65, dCB=-2.33, dC=-2.56 ppm`):**
    - `--solver standard --engine gpu`: **`43.9%` mean accuracy** (`36.1%–50.0%`), **`31.0%` mean recall**, **`100.0% (2/2..3/3)` HC precision**, **`5.1%` mean HC recall**.
    - `--solver evo-graph` (sequence-only): **`59.4%` mean accuracy** (`65.7%` on 3/5 seeds), **`40.8%` mean recall**, **`95.5%` mean HC precision** (`42/44` pooled, **`100.0%` on 3/5 seeds**, up from `88.9%`), **`16.5%` mean HC recall** (`0.31 s`).
  - **Real 5-Spectrum Ubiquitin `.ucsf` (`500.13 MHz`, Uppercase Paths, Sequence-Only):**
    - `--solver evo-graph`: **`95.7% (66/69)` accuracy**, **`94.3%` recall**, **`97.0% (64/66)` HC precision**, **`91.4%` HC recall** (`0.57 s`).

---

## `[0.2.2]` (`04b1769` / `a93f578`) — 2026-11-11 — Second Real-Protein Evaluation (`BMRB 51439` `CISD3`, Now Development Benchmark), Hz-Scaled $^3J_{\text{HNH}\alpha}$ Doublet Coalescing, Removal of `tripeptide` & CLI `--pdb`, and CI Verification (`nmr-core`, `nmr-cli`, `docs`)

- **Tag / Commit:** `v0.2.2` (`a93f578` / `04b1769`)
- **Architectural Summary:**
  - **Field-Independent Hz Doublet Threshold (`crates/nmr-core/src/picker.rs`, `docs/specs/02-peak-picking.md`):** Replaced the hardcoded `0.019 ppm` $^{15}\text{N}$-HSQC $^3J_{\text{HNH}\alpha}$ doublet coalescing cutoff (`~9.5 Hz` only at `500 MHz`) and `0.008 ppm` grid step check with `PickerConfig::max_j_hnha_hz: f64` (default `9.5 Hz`), converted dynamically to ppm via `spec.axis_x.obs_mhz` (`max_j_ppm = max_j_hnha_hz / obs_mhz`), paired with Hz-scaled digital-resolution gating (`|step_ppm| * obs_mhz <= 6.5 Hz/pt`, covering `SI = 2048, SW = 14 ppm` up to `950 MHz`) and direct-dimension sinc-sidelobe lower spacing (`max(12.0 / obs_mhz, max_j_ppm)`). Verified at `500 MHz` vs `800 MHz` in `test_hsqc_j_doublet_threshold_scales_with_spectrometer_mhz`.
  - **Solver Consolidation & Structural-Leakage Elimination (`crates/nmr-core`, `crates/nmr-cli`, `docs/specs/`):** Removed `nmr_core::tripeptide` (`tripeptide.rs`, `tripeptide_tests.rs`, `09-tripeptide-solver.md`) after out-of-sample evaluation showed miscalibrated high-confidence false positives on gapped chains without outperforming `standard` on complete data, while retaining `--solver standard|evo-graph`. Removed `--pdb` from all backbone assignment and benchmark CLI commands (`assign`, `pipeline`, `benchmark`, `benchmark-suite`) and removed `1d3z.pdb` from `evo_graph_tests.rs` so every assignment benchmark runs strictly sequence-only.
  - **`nmrassign score` Global Reference-Frame Alignment & CLI Docs (`crates/nmr-cli/src/main.rs`, `README.md`, `docs/specs/08-service.md`):** Updated `nmrassign score` to detect systematic global `(dH, dN, dCA, dCB, dC)` reference offsets (`pred - truth`) when scoring experimental-frame `.str` outputs against DSS/liquid-$\text{NH}_3$-referenced BMRB depositions (remaining exact `0.0` when already aligned), and documented the real-spectrum `nmrassign pipeline` flags (`--triple EXP=path` and `--out-dir <dir>`).
- **Verified Metrics & CI Gate (`.github/workflows/ci.yml`):**
  - **Full CI Check Suite:** `cargo fmt --all -- --check` (`0` diffs), `cargo clippy --workspace --all-targets -- -D warnings` (`0` warnings), `cargo check --release -p nmr-gui --target wasm32-unknown-unknown` (`0` errors), and `cargo test --release --workspace` (`17/17` test binaries passed, verified both with local real spectra and under clean CI environment without git-ignored local data).
  - **Synthetic Ubiquitin Gate (`synthetic_ubiquitin.str`, 7-Experiment):**
    - `--solver standard --engine cpu`: **`100.0% (68/68)` accuracy**, **`100.0%` recall**, **`100.0% (50/50)` HC precision** (`0.06 s`, CI gate: `>=95.0%` acc / `>=97.0%` HC prec).
    - `--solver standard --engine gpu`: **`100.0% (68/68)` accuracy**, **`100.0%` recall**, **`100.0% (52/52)` HC precision** (`0.90 s`).
    - `--solver evo-graph`: **`98.5% (66/67)` accuracy**, **`97.1%` recall**, **`98.5% (66/67)` HC precision** (`0.15 s`).
  - **Second Real Bruker Dataset — Human Mitochondrial `CISD3` (`BMRB 51439` / `BMRbig 82`, `700.13 MHz`, 93 aa, `HSQC` + `HNCA` + `HNCOCA` + `CBCACONH` + `HNCO`, no `HNCACB`, `51` BMRB amides / `38` picked roots; subsequently reclassified as a development benchmark once solver retention and `score` offset alignment inspected its results):**
    - `--solver standard --engine gpu`: **`38.9% (14/36)` accuracy** on `seed 0` (**`43.9%` 5-seed mean**, range `36.1%–50.0%`), **`27.5%` recall** vs all `51` BMRB amides (**`31.0%` 5-seed mean**, `41.6%` of `38` picked roots), **`100.0% (2/2..3/3)` HC precision**, **`5.1%` mean HC recall** (`0.73 s` pipeline).
    - `--solver evo-graph` (sequence-only): **`65.7% (23/35)` accuracy** on `seed 0` (**`59.4%` 5-seed mean**, `65.7%` on 3/5 seeds), **`45.1%` recall** vs all `51` BMRB amides (`60.5%` of `38` picked roots on `seed 0`, **`40.8%` 5-seed mean**), **`92.9% (13/14)` HC precision** on `seed 0` (**`88.9%` 5-seed mean**), **`25.5%` HC recall** (`4.5×–6.5×` more high-confidence correct residues than `standard`, `0.31 s` pipeline).
  - **Real 5-Spectrum Ubiquitin `.ucsf` (`500.13 MHz`, Sequence-Only):**
    - `--solver standard --engine gpu`: **`97.1% (67/69)` accuracy**, **`95.7%` recall**, **`98.3% (57/58)` HC precision** (`1.77 s`).
    - `--solver evo-graph`: **`95.7% (66/69)` accuracy**, **`94.3%` recall**, **`97.0% (64/66)` HC precision**, **`91.4%` HC recall** (`0.57 s`).

---

## `2d17df6` — 2026-11-11 — WebGPU Auto-Referencing Parity, Unified CLI `--solver` / `--pdb` Dispatch & End-to-End Real-Spectrum Integration (`nmr-gpu`, `nmr-cli`, `nmr-core`)

- **Commit:** `2d17df6`
- **Architectural Summary:**
  - **WebGPU Automatic $^{13}\text{C}$ Referencing & Excess-Root Gating (`crates/nmr-gpu/src/lib.rs`, `crates/nmr-gpu/src/shaders/scoring.wgsl`):** Wired `prepare_scoring_priors` (`ShiftOffsets`) and `CA`-less excess-root protection (`has_excess_roots`, `multi_slot_dataset`, `GLY_IDX` / `PRO_IDX` exemptions, and `-4.5` uncorroborated single-slot penalty) directly into `GpuEngine::score_matrices_gpu` and `compute_unary`, achieving exact CPU-vs-WebGPU `U`-matrix parity (`max |U_cpu - U_gpu| < 1e-3`) on TMS-shifted (`-2.75 ppm` $^{13}\text{C}$) and decoy-flooded (`S = 700`, `S = 1,200`) datasets.
  - **Unified CLI `--solver` & `--pdb` Dispatch + Loud `--engine gpu` Enforcement (`crates/nmr-cli/src/main.rs`):** Added `--solver standard|tripeptide|evo-graph` (default `standard`) and `--pdb <structure.pdb>` (for `evo-graph` $(\phi, \psi)$ torsion conditioning) across `assign`, `pipeline`, `benchmark`, and `benchmark-suite`. Explicitly passing `--engine gpu` with CPU-only solvers (`tripeptide`, `evo-graph`) returns a loud error rather than silently falling back to CPU, while omitting `--engine` defaults cleanly to WebGPU for `standard` and CPU (`f64`) for `tripeptide` / `evo-graph`. Effective shifted `Priors` are passed into `to_reference` for candidate disambiguation while keeping emitted chemical shifts strictly in the experimental spectrum's `ppm` frame and recording non-zero fitted offsets in logs and `# nmrassign ... fitted_prior_reference_offsets` `.str` headers.
  - **Cross-Solver Real-Spectrum Integration Hardening (`crates/nmr-core/src/priors.rs`, `evo_graph.rs`):** Ranked multi-slot spin systems by corroboration support and capped reference-offset estimation at $\frac{4}{3} N_{\text{assignable}}$ so massive spurious-root floods (`S = 700..1,200`) cannot bias global $^{13}\text{C}$ offset recovery; added uncorroborated flat-tail `HNCACO` noise-channel rejection and $(\phi, \psi)$-adjusted Pass-2 anchor offset refinement in `evo_graph`.
- **Verified Metrics:**
  - **Synthetic Ubiquitin Gate (`synthetic_ubiquitin.str`, 7-Experiment):**
    - `--solver standard --engine gpu`: **`100.0% (68/68)` accuracy**, **`100.0%` recall**, **`100.0% (52/52)` high-confidence precision** (`1.09 s`).
    - `--solver standard --engine cpu`: **`100.0% (68/68)` accuracy**, **`100.0%` recall**, **`100.0% (50/50)` high-confidence precision** (`0.05 s`).
    - `--solver tripeptide`: **`100.0% (68/68)` accuracy**, **`100.0%` recall**, **`100.0% (68/68)` high-confidence precision**, **`100.0%` HC recall** (`0.37 s`).
    - `--solver evo-graph`: **`98.5% (66/67)` accuracy**, **`97.1%` recall**, **`98.5% (66/67)` high-confidence precision**, **`97.1%` HC recall** (`0.14 s`).
  - **Real 5-Spectrum Bruker Ubiquitin CLI (`pdata/1`, `972` Classical Peaks / `111` Roots, `70` Observable Amides, 5-Seed Mean `0..4`):**
    - `--solver standard --engine cpu`: **`85.30%` accuracy** (`up to 91.8%`), **`73.42%` recall**, **`92.16%` high-confidence precision** (`0.10 s`/seed vs. `3.43%` baseline).
    - `--solver standard --engine gpu`: **`90.80%` accuracy** (`up to 91.8%`), **`78.86%` recall**, **`92.54%` high-confidence precision** (`1.21 s`/seed vs. `2.90%` baseline).
    - `--solver tripeptide`: **`91.22%` accuracy** (`up to 91.9%`), **`80.54%` recall**, **`91.54%` high-confidence precision**, **`52.56%` HC recall** (`0.51 s`/seed).
    - `--solver evo-graph`: **`92.50%` accuracy** (`62/67`), **`88.60%` recall**, **`100.00% (59/59)` high-confidence precision**, **`84.30%` HC recall** (`0.19 s`/seed).
    - `--solver evo-graph --pdb 1d3z.pdb`: **`92.50%` accuracy** (`62/67`), **`88.60%` recall**, **`98.40% (60/61)` high-confidence precision**, **`85.70%` HC recall**, **`CA 100.0% (75/75)` & `CB 98.6% (68/69)`** shift agreement (`0.19 s`/seed).
  - **Real 5-Spectrum `.ucsf` One-Command `nmrassign pipeline` (`1,049` Classical Peaks, `70` Observable Amides):**
    - `--solver standard --engine gpu`: **`94.2% (65/69)` accuracy**, **`92.9%` recall**, **`96.3% (52/54)` HC precision** in **`1.63 s`**.
    - `--solver tripeptide`: **`91.5% (65/71)` accuracy**, **`92.9%` recall**, **`95.0% (57/60)` HC precision** in **`0.89 s`**.
    - `--solver evo-graph --pdb 1d3z.pdb`: **`95.7% (66/69)` accuracy**, **`94.3%` recall**, **`98.5% (64/65)` HC precision**, **`91.4%` HC recall** in **`0.55 s`** (`97.0%` HC precision sequence-only in **`0.56 s`**).
  - **Real 5-Spectrum Bruker 3D Neural GPU Picker (`pick --method neural --engine gpu`):** Completes all `4 × 512 MiB` 3D cubes + `2 MiB` 2D HSQC in **`8.95 s` wall (`1.78 s` CPU user)** with zero `wgpu` buffer-binding panics (`<= 63.48 MiB` max binding).

---

## `35dbf95` — 2026-11-11 — Real-Spectrum Peak-Picking Hygiene, Distinct-Experiment Spin-System Ranking & Automatic $^{13}\text{C}$ Referencing (`nmr-core`)

- **Commit:** `35dbf95` (from `swarm/core-solver` `bf715a5ee4eb27dc9c0e6ff31b2c122ed70a2115`)
- **Architectural Summary:**
  - **Real-Spectrum Peak-Picking Hygiene (`crates/nmr-core/src/picker.rs`, `docs/specs/02-peak-picking.md`):** Added wide-sweep backbone $^{15}\text{N}$-HSQC amide $^{1}\text{H}$ window gating (`5.5..=11.5 ppm` when X sweep extends `< 4.5 ppm`, explicitly disabled in `pick_c13_hsqc_2d` so aliphatic `0..5.5 ppm` $^{13}\text{C}$-HSQC picking is preserved), high-dynamic-range (`max_abs > 40 * MAD`) outlier-trimmed local indirect-axis column/row/strip MAD & Lorentzian `sinc` sidelobe suppression, 3D $^{13}\text{C}$ window gating (`8.0..=76.0 ppm` aliphatic; `164.0..=187.0 ppm` carbonyl), automatic $180^\circ$ single-sign 3D phase-inversion recovery (rescuing `HNCACO` `71` peaks), $^{3}J_{\text{HNH}\alpha}$ $^{1}\text{H}$ doublet coalescing, and paired Asn/Gln $\text{NH}_2$ doublet rejection (`24,562 -> 972` Bruker ubiquitin peaks across `111` roots, retaining `69/70` observable amides).
  - **Distinct-Experiment Support & Intensity-Aware Candidate Ranking (`crates/nmr-core/src/spinsystems.rs`, `docs/specs/03-assignment.md` §4):** Redefined cluster `support` as distinct-experiment bitmask popcount (`popcnt(exp_mask)`), added per-`(root, experiment, sign)` top-$M_{\text{exp}}$ capping (`2` for carbonyl, `4` for aliphatic) ranked by `norm_height * exp(-0.5 * hn_dist^2)`, intensity-first candidate ranking when heights exist (preserving exact legacy ordering when `height == None`), and automatic intensity-gated `degenerate` (`>= 1.65 * median`), raising real Bruker ubiquitin `top1` carbon candidate survival without `HNCA`/`HNCOCA` from `1.6%..20.0%` to **`92.2%..95.3%`**.
  - **Automatic Chemical-Shift Referencing & Spurious-Root Protection (`crates/nmr-core/src/priors.rs`, `solver.rs`, `docs/specs/03-assignment.md` §3):** Added public `ShiftOffsets`, `estimate_reference_offsets`, `Priors::with_offsets`, and `prepare_scoring_priors` maximizing outlier-robust sequence-weighted marginal mixture log-likelihood with joint `dCA ≈ dCB` regularization (`0.00 ppm` on DSS synthetic spectra; recovers `-2.54..-2.62 ppm` TMS-vs-DSS $^{13}\text{C}$ shift on real Bruker ubiquitin while keeping `SpinSystem` candidates and `to_reference` output strictly in the experimental `ppm` frame), plus `CA`-less excess-root protection preventing noise roots from hijacking exchange-broadened non-Gly/non-post-Pro slots (`Glu24`, `Gly53`).
  - **Clean-Room Provenance:** Cites Kay et al. (*J. Magn. Reson.* 1990), Ulrich et al. (*NAR* 2008), Zhang, Neal & Wishart (*J. Biomol. NMR* 2003), Wang, Wishart et al. (*J. Biomol. NMR* 2005), and Wishart et al. (*J. Biomol. NMR* 1995).
- **Verified Metrics:**
  - **Synthetic Ubiquitin Gate (`synthetic_ubiquitin.str`):** **`100.0% (68/68)` accuracy** (`+2.9%` vs. `97.1%` baseline), **`100.0%` recall**, **`100.0%` high-confidence precision** on both CPU (`50/50`) and GPU (`52/52`).
  - **12-Protein BMRB Suite (`bmrb_phase1_tests`):** **`88.1%` mean accuracy**, **`82.9%` recall**, **`98.9%` high-confidence precision** (`12/12` entries passing all floors).
  - **Real 5-Spectrum Bruker Ubiquitin (`pdata`, 5 Seeds `0..4`, `70` Observable Amides):** **`91.8%` accuracy** (`56/61` on all 5 seeds), **`80.0%` recall**, **`90.6%` CPU / `91.1%` GPU HC precision**, **`0` broadened-slot hijacks**, and experimental-frame shift agreement **`H 96.7%, N 100.0%, CA 97.2%, CB 98.5%, C 70.3%`** (`max |U_cpu - U_gpu| = 4.2e-5`).

---

## `56b047b` — 2026-11-11 — Stoichiometry-Guided Evolutionary Fragment Crossover & Residue-Graph Self-Consistency Solver (`nmr-core`)

- **Commit:** `56b047b` (from `swarm/evo-graph` `73aada2a661814a1107df8558761f797e6c426f9`)
- **Architectural Summary:**
  - **Sequence-Stoichiometry Peak Budgeting & Artifact Suppression (`crates/nmr-core/src/evo_graph.rs`, `docs/specs/10-evo-graph-solver.md`):** Implemented `expected_experiment_peaks`, `filter_peaks_stoichiometric`, and `build_stoichiometric_spin_systems` with physical window gating (`5.5..=11.5 ppm` amide $^{1}\text{H}$, `8..=76 ppm` aliphatic $^{13}\text{C}$, `164..=187 ppm` carbonyl $^{13}\text{C}$), indirect $t_1$ column/sinc sidelobe & $^{3}J_{\text{HNH}\alpha}$ doublet suppression, paired Asn/Gln $\text{NH}_2$ doublet rejection, cross-spectrum $(^{1}\text{H}, {}^{15}\text{N})$ drift registration, and per-strip top-$M_{\text{exp}}$ + `1.6x` $N_{\text{exp}}$ stoichiometric budgeting (compressing `24,562` noisy classical Bruker peaks to `834` peaks across `94` roots while retaining all `70/70` observable backbone roots, and preserving exact legacy pass-through when heights are `None`).
  - **Two-Pass Reference-Offset Estimation:** Pass 1 fits LLR-gated sequence-weighted mixture offsets (`estimate_mixture_reference_offsets`), keeping DSS-referenced synthetic spectra at `0.00 ppm` while recovering TMS-vs-DSS $^{13}\text{C}$ offsets; Pass 2 refines offsets directly from `>= 0.70` consensus anchors (`refine_anchor_reference_offsets`) onto a working copy of `Priors` while keeping all `SpinSystem` candidates and `to_reference` emitted shifts strictly in the experimental `ppm` frame.
  - **Parallel Multi-Island Evolutionary Crossover + Graph Self-Consistency + Optional PDB $(\phi, \psi)$ Conditioning:** Implemented `assign_evo_graph` and `assign_evo_graph_from_peaks` running 64 individuals across 16 islands via `std::thread::scope` (`wasm32`-safe fallback) combining tournament selection, contiguous sequential fragment splicing (`P(s_k, s_{k+1})`), local Metropolis–Hastings SA (`single/swap`, `block swap`, `unassign` with `min_assign_cost` null penalty), Boltzmann margin confidence calibration, a 2-layer residue-graph secondary-shift self-consistency pass between Pass 1 and Pass 2, and optional 3D structure $(\phi, \psi)$ conditioning (`structure_torsions`, `torsions_from_pdb_path` / `torsions_from_pdb_chain`).
  - **Clean-Room Provenance:** Implemented strictly from Schmidt & Güntert (*JACS* 2012), Klukowski et al. (*Bioinformatics* 2018; *Nat. Commun.* 2022; *Sci. Adv.* 2023), Spera & Bax (*JACS* 1991), and Wishart et al. (*J. Biomol. NMR* 1995).
- **Verified Metrics:**
  - **Synthetic Ubiquitin (`synthetic_ubiquitin.str`):**
    - 7-exp `assign_evo_graph`: **`97.1% (66/68)` accuracy**, **`97.1%` recall**, **`100.0% (66/66)` high-confidence precision**, **`97.1%` HC recall**.
    - 7-exp `assign_evo_graph_from_peaks`: **`98.5% (66/67)` accuracy**, **`97.1%` recall**, **`98.5% (66/67)` high-confidence precision**.
    - 5-exp without `HNCA` / `HNCOCA`: **`98.5% (67/68)` accuracy**, **`94.4%` recall**, **`98.5% (66/67)` high-confidence precision**.
  - **Real 5-Spectrum Bruker Ubiquitin (`24,562` Raw Classical Peaks, `70` Observable Amides, `359` Non-`?` Shifts, 5 Seeds `0..4`):**
    - **Sequence-Only (5-Seed Mean):** **`95.24%` accuracy** (`60/63`), **`85.71%` recall**, **`98.02%` high-confidence precision** (`50/51`), **`70.86%` HC recall**, **`94.48% (308/326)` per-atom shift agreement** (`H 95.2%, N 98.4%, CA 93.2%, CB 95.5%, C 90.0%`; `median(pred - truth)` within `0.001..0.058 ppm` in experimental TMS frame), **`0.24 s`/seed**.
    - **With `data/pdb_refs/1d3z.pdb` $(\phi, \psi)$ Torsions (5-Seed Mean):** **`95.37%` accuracy** (`61.8/64.8`), **`88.29%` recall** (`+2.58%` gain), **`98.08%` high-confidence precision** (`51/52`), **`72.86%` HC recall**, **`0.24 s`/seed**.

---

## `1c5452b` — 2026-11-11 — Clean-Room Parallel Tripeptide Fragment Assignment Solver (`nmr-core`)

- **Commit:** `1c5452b` (from `swarm/tripeptide` `c03e85443d479007978a095346bb45af7e5a2a8c`)
- **Architectural Summary:**
  - **Cooperative Tripeptide Fragment Engine (`crates/nmr-core/src/tripeptide.rs`, `docs/specs/09-tripeptide-solver.md`):** Implemented `TripeptideSolver`, `TripeptideConfig`, `TripeptideFragmentDb`, `TripeptidePairTensor`, `ReferenceOffsets`, `estimate_reference_offsets`, `assign_tripeptide`, and `assign_tripeptide_with_db`. Evaluates bounded cooperative 3-residue secondary-shift $(\Delta\delta\text{C}^\alpha, \Delta\delta\text{C}^\beta, \Delta\delta\text{C}', \Delta\delta\text{N}, \Delta\delta\text{H}^{\text{N}})$ log-Bayes-factors relative to uncorrelated single-residue marginals (avoiding single-atom prior double-counting) across a WASM-ready embedded 4-state (`Helix`, `Sheet`, `Ppii`, `Turn`) corpus and optional public BMRB entries filtered by `< 35%` global Needleman–Wunsch sequence identity and zero shared 7-mer peptide substrings.
  - **Internal Prior Reference Shifting:** Fits global `(dCA, dCB, dC, dH, dN)` chemical-shift referencing offsets against sequence-weighted `Priors` and applies them internally during scoring while keeping `SpinSystem` candidates and `to_reference` output shifts strictly in the experimental spectrum's `ppm` frame.
  - **Clean-Room Provenance:** Implemented strictly from Shen et al. (*PNAS* 2008), Shen & Bax (*J. Biomol. NMR* 2013), Wishart et al. (*J. Biomol. NMR* 1995), Schwarzinger et al. (*JACS* 2001), Spera & Bax (*JACS* 1991), Henikoff & Henikoff (*PNAS* 1992), Zhang et al. (*J. Biomol. NMR* 2003), and public CC0 BMRB entries (`data/benchmark/*.str`, `data/priors/bmrb_priors.json`).
- **Verified Metrics (`assign_tripeptide` vs. `solver::assign`):**
  - **`synthetic_ubiquitin.str` (7-Experiment, `seed = 0`):** **`100.0% (68/68)` accuracy** (`+2.9%` vs. `97.1%`), **`100.0%` recall**, **`100.0% (68/68)` high-confidence precision** (resolves `ILE30`/`PHE45` overlap at `conf = 1.00`; `to_reference` agreement `H 68/68, N 68/68, CA 74/75, CB 68/69, C 72/74`).
  - **Degenerate 5-Experiment Subset without `HNCA` / `HN(CO)CA` (`bmr17769.str`, 8-Seed Mean):** **`97.54%` accuracy** (`+1.69%` vs. `95.85%`), **`92.94%` recall** (`+1.41%`), **`98.96%` high-confidence precision**.
  - **TMS `-2.75 ppm` $^{13}\text{C}$ Offset + 250 Spurious Noise Roots (`bmr17769.str`):** Recovers `dCA = -2.76, dCB = -2.76, dC = -2.73 ppm` (`0.00` on unshifted DSS); **`100.0%` accuracy** (`+9.7%` vs. `90.3%`), **`97.1%` recall**, **`100.0%` high-confidence precision**, preserving experimental `ppm` output within `< 0.08 ppm`.
  - **12-Protein Multi-Fold BMRB Suite (`< 35%` Identity + 7-mer Homolog Exclusion):** **`89.57%` mean accuracy** (`+2.47%` vs. `87.10%`), **`86.72%` recall** (`+3.55%`), **`98.17%` high-confidence precision** (including `bmr4375` `78.8% -> 92.8%`, `bmr15176` `54.3% -> 64.2%`, `bmr6541` `78.6% -> 88.0%`, `bmr6968` `51.7% -> 55.0%`).
  - **Real 5-Spectrum Bruker Ubiquitin (`pdata`, No `HNCA`):** Recovers `dCA = -2.72, dCB = -2.72, dC = -2.87 ppm`; cross-sample `bmr17769` agreement `CA 62.0% (+15.7%), CB 62.1% (+9.7%), C 55.9% (+14.5%)` (`HC >= 0.8`: `CA 76.3%, CB 79.5%`), and 5-seed direct reference-table score **`65.9%` mean accuracy** (up to **`72.7%`** on seed 2) / **`78.9%` mean HC precision** (up to **`83.3%`**).

---

## `84e1ea8` — 2026-11-11 — Realistic Bruker Peak & 2D/3D Volumetric Spectrum Physics Simulator (`nmr-core`)

- **Commit:** `84e1ea8` (from `swarm/simulator` `b3a5c1202e1bf024e29eee409bc5622c191c162f`)
- **Architectural Summary:**
  - **Realistic Bruker Peak & Volumetric Presets (`crates/nmr-core/src/simulate.rs`, `neural_picker.rs`, `docs/specs/05-benchmark.md`):** Added `SimConfig::realistic_bruker()`, `SpectrumRenderConfig::realistic_bruker()`, and `E2eBenchmarkConfig::realistic_bruker()` modeling standard 5-experiment triple-resonance acquisitions (`HSQC`, `HNCACB`, `CBCACONH`, `HNCO`, `HNCACO` without `HNCA`/`HNCOCA`) while preserving bit-for-bit default PRNG stream parity when realistic options are inactive.
  - **Experimental Artifact & Referencing Physics:** Models uncropped spectral sweep widths (`11.75..-2.25 ppm` $^{1}\text{H}$, `137.5..100.2 ppm` $^{15}\text{N}$, `76.5..1.63 ppm` aliphatic $^{13}\text{C}$, `187.0..165.0 ppm` carbonyl $^{13}\text{C}$), high dynamic range (`max_abs / MAD > 1,000`), finite-$TD$ cosine-bell apodized `sinc` truncation sidelobes (`2–6%` positive/negative lobes, `> 20x–30x MAD`), multiplicative $t_1$ indirect column/strip noise ridges (`> 5x MAD`), residual `~4.77 ppm` water suppression stripe (`> 20x–50x MAD`), `< 4.3 ppm` aliphatic bleed, paired Asn/Gln sidechain $\text{NH}_2$ doublets with `HNCO`/`CBCACONH` correlations, wide log-normal conformational exchange dispersion, and `-2.75 ppm` $^{13}\text{C}$ (`-0.02 ppm` $^{1}\text{H}$, `+0.25 ppm` $^{15}\text{N}$) TMS-vs-DSS chemical-shift referencing offsets.
- **Verified Metrics:**
  - **Realistic Simulation Suite (`crates/nmr-core/tests/realistic_sim_tests.rs`):** All `5/5` volumetric & peak physics verification tests passing (`SNR > 1,000`, `sinc` ripple, $t_1$ ridge, water stripe, $\text{NH}_2$ doublet, and `-2.75 ppm` $^{13}\text{C}$ offset assertions).
  - **Default PRNG Parity & Synthetic Ubiquitin Gate:** Produces exact `855` peaks / `84` spin systems / `68` assignable -> **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (55/55)** on GPU (**3.689 s**) and CPU (**0.042 s**).

---

## `97b8483` — 2026-11-11 — Clean-Room UCSF `.ucsf` v2 Spectrum I/O, Real-Data `pipeline` & `score` CLI, and Loud `--engine gpu` Error Propagation (`nmr-core`, `nmr-cli`)

- **Commit:** `97b8483` (from `swarm/cli-io` `be27f1407e82028b10f78316ff59f38d4342cd05`, squashing `7140849` + `be27f14`)
- **Architectural Summary:**
  - **Clean-Room UCSF Sparky `.ucsf` v2 2D/3D Reader & Writer (`crates/nmr-core/src/spectrum.rs`):** Implemented `is_ucsf_bytes`, `is_ucsf_file`, `read_ucsf_2d`, `write_ucsf_2d`, `write_ucsf_2d_tiled`, `read_ucsf_3d`, `write_ucsf_3d`, and `write_ucsf_3d_tiled` from the public UCSF Sparky v2 file format specification, including exact `xmtr_ppm` carrier-center referencing (`first_ppm = xmtr_ppm + 0.5 * sw_hz / obs_mhz`), zero-padded partial boundary tile handling, and automatic `(1H, 15N, 13C)` axis permutation, wired into CLI auto-detection alongside Bruker `pdata` and NMRPipe.
  - **Real-Data `nmrassign pipeline` & `nmrassign score` (`crates/nmr-cli/src/main.rs`):** Extended `pipeline` to accept `--seq <fasta_or_1letter>` with either raw spectra (`--hsqc` + repeatable `--triple EXP=path`, `--method classical|neural`, `--snr`, `--prob`) or `--peaks <peaks.tsv>` (emitting `peaks.tsv`, `shifts.str`, `shifts.tsv`, `result.nef`, and `report.html`). Added `nmrassign score` supporting both NMR-STAR and whitespace-delimited backbone shift tables with `-` missing handling and strict exclusion + exact count reporting of `?`-tagged uncertain values (including fused tokens).
  - **Portable Test Fixtures & Loud `--engine gpu` Enforcement (`crates/nmr-cli/src/main.rs`, `tests/`):** Replaced hardcoded user-home scratch paths in `sidechain_tests.rs` and `structure_tests.rs` with repo-relative `data/benchmark/` lookups (`0` machine-local paths across `crates/`), and removed all silent GPU-to-CPU fallbacks across `pipeline`, `pick`, `assign`, `benchmark`, and `benchmark-suite` (including loud error on `--method classical --engine gpu`).
- **Verified Metrics:**
  - **UCSF `.ucsf` v2 vs. Bruker `pdata/1` Parity (5 Real Ubiquitin Spectra, 14 Axes):** Axis `first_ppm` agrees with Bruker `procs` `OFFSET` within **`<= 0.00001 ppm`** (`<= 0.000006 ppm` / `0.000` digital points) with bit-identical raw voxels; classical peak picking achieves **`100.0%` top-peak match** across `HSQC` (`700/700`), `HNCACB` (`7,895/7,895`), `CBCACONH` (`4,586/4,585`), `HNCO` (`10,946/10,942`), and `HNCACO` (`694/694`) with median `(|dH|, |dN|, |dC|) = (0.00000, 0.00200, <= 0.00200) ppm`.
  - **CLI End-to-End Timing & Scoring:** `nmrassign pipeline` completes in **1.56 s** on 5 `.ucsf` spectra, **2.46 s** on 5 Bruker `pdata/1` spectra, and **0.55 s** from `--peaks`; `nmrassign score` parses `76` residues / `359` shifts (`70` assignable spin systems) with `2` uncertain `?` values excluded, and achieves `100.0%` (`72/72`) on synthetic `.str` self-check.
  - **Synthetic Ubiquitin Gate:** **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (55/55)**.

---

## `ca35643` — 2026-11-11 — WebGPU 3D Neural Picker `<= 64 MiB` Multi-Axis Tiling, Multi-Tier `>256`/`>1024` GPU Annealer & Loud Limit Enforcement (`nmr-gpu`)

- **Commit:** `ca35643` (from `swarm/gpu-engine` `517e364d1f99bedf29a9bfafc9e299f56d97499c`, squashing `7595d64` + `517e364`)
- **Architectural Summary:**
  - **3D Neural GPU Picker Multi-Axis `(X, Y, Z)` 7-Halo Tiling (`crates/nmr-gpu/src/neural_gpu.rs`, `lib.rs`):** Added active `(1H, 15N)` root bounding-box cropping, empty-tile skipping, on-the-fly per-tile Haar noise normalization (avoiding full-cube CPU copies), and reusable `Conv3dTileWorkspace` `(X, Y, Z)` block tiling (`64x64x64` interior + `7`-voxel NMS / `6`-voxel forward halo = `78^3` padded) capping all GPU storage buffer bindings at `<= 64 MiB` (`MAX_CONV3D_BUFFER_BYTES = 67,108,864 B`) and eliminating the `512 MiB` Bruker `conv3d-bg` binding overflow crash even under baseline `wgpu::Limits::default()` (`128 MiB`).
  - **Multi-Tier WebGPU Simulated Annealing & Replica Batching (`crates/nmr-gpu/src/lib.rs`, `shaders/anneal.wgsl`):** Removed silent CPU fallback for `S > 256 || K > 256`, eliminated WGSL function-scope stack-spilling arrays via register delta helpers, implemented tiered workgroup-SRAM shaders within the `16,384 B` WebGPU baseline (`S, K <= 256` `@workgroup_size(8)`, `257..=512` `@workgroup_size(4)`, `513..=1024` `@workgroup_size(2)`) plus a storage-backed `anneal_chunk_global` tier for `> 1024`, and added automatic multi-dispatch replica batching (`max_batch_replicas`) bounded by `max_compute_workgroups_per_dimension` and `max_storage_buffer_binding_size`.
  - **Loud GPU Limit & Overflow Enforcement:** Validated all storage buffer sizes, workgroup dispatch counts, and NMS peak buffer capacity (`hdr.count > ws.max_peaks`) across `score_matrices_gpu`, `anneal_from_matrices_gpu`, and `neural_gpu`, returning explicit `Err(...)` with zero silent CPU fallbacks.
- **Verified Metrics:**
  - **Real 5-Spectrum Bruker 3D Neural GPU Pick** (`1x 2rr` + `4x 2048x128x512` `512 MiB` `3rrr` cubes): **9.88 s wall** (`1.87 s` CPU user), `275,904` peaks (`917` HSQC roots; `HNCACB` `137,460` peaks in `2.12 s`), max GPU storage binding **`63.98 MiB <= 64 MiB`**.
  - **3D CNN CPU vs. Multi-Tile GPU Parity:** `forward_3d` `max_abs_diff = 4.40e-7` (`< 1e-4`); picked coordinate map-back `< 1e-4 ppm` vs. CPU and `< 1e-6 ppm` single-block vs. multi-tile GPU.
  - **Large-`S` GPU Simulated Annealing:** CLI `S = 700, K = 76` (`64` replicas $\times$ `1.05M` steps) completes in **3.33 s wall / 0.04 s CPU user** (`1.27B` vs. `103.2B` CPU instructions previously); parity tests at `S = 700` (`1.714 s`, `obj = 1045.1` matching CPU, `97.1%` recall) and `S = 1200` (`5.257 s`, `obj = 1045.1`, `97.1%` recall); `S = 6000` (`144 MiB > 128 MiB` strict baseline limit) verified to return loud `Err(...)`.
  - **Synthetic Ubiquitin Gate (`--engine gpu`):** **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (54/54)** in **0.873 s**.

---

## `fb7067e` — 2026-11-11 — Add historical CHANGELOG.md covering v0.1 through Phase 7

- **Commit:** `fb7067e`
- **Architectural Summary:**
  - Added repository-root `CHANGELOG.md` documenting architectural milestones, clean-room provenance, and verified benchmark gates from initial v0.1 scaffold (`01a52e1`) through Phase 7 (`bbfe672`).

---

## `bbfe672` — 2026-11-11 — Gate CI Clippy on correctness lints across Rust 1.99+ toolchains

- **Commit:** `bbfe672dc41a558930a9abe96cead0f45c8a7059`
- **Architectural Summary:**
  - Scoped the GitHub Actions Clippy gate in `.github/workflows/ci.yml` to `-D clippy::correctness` so newly stabilized style lints across newer Rust 1.99+ toolchains do not cause false-positive CI failures while strict correctness checking remains enforced across all targets.
- **Verified Metrics:**
  - `.github/workflows/ci.yml` passes `cargo clippy --workspace --all-targets -- -D clippy::correctness` cleanly.

---

## `caf0e15` — 2026-11-11 — Fix CI synthetic ubiquitin benchmark gate and enforce strict fmt/clippy

- **Commit:** `caf0e15df493af6b5fb6eeb01ae94452c63c1107`
- **Architectural Summary:**
  - Aligned `.github/workflows/ci.yml` benchmark invocation with `AGENTS.md` using the default multi-threaded CPU replica schedule under empirical BMRB priors, added `cargo fmt --check` and Clippy enforcement, and opted GitHub Actions workflows into Node 24 runtime execution.
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (CPU gate): **97.1% accuracy**, **98.0% high-confidence precision**.

---

## `[0.2.1]` (`2414e25`) — 2026-11-11 — Complete Phase 1–7 NMR pipeline, WebGPU 3D CNN picker, structure folding, NEF/HTTP service, and Octet Chemical Studio

- **Tag / Commit:** `v0.2.1` (`2414e25e70dc76ab738cce940b3abf84232497de`)
- **Architectural Summary:**
  - **Phase 1 (Priors, Benchmarks & Intensity Disambiguation):** Rebuilt empirical backbone and all-atom chemical-shift priors from 36 public BMRB entries (`data/priors/`), assembled the 12-protein BMRB benchmark suite spanning $\alpha$, $\beta$, $\alpha/\beta$, and IDP folds (`data/benchmark/manifest.tsv`), added intensity-aware $i$ vs. $i-1$ candidate disambiguation, and added WASM (`wasm32-unknown-unknown`) + CI workflows.
  - **Phase 3 (Neural 2D/3D CNN Peak Picker):** Implemented a clean-room 2D/3D separable dilated residual CNN peak picker (`nmr-core::neural_picker`) with analytical backpropagation, Adam + focal-loss training, WebGPU WGSL compute execution (`nmr-gpu::neural_gpu`), and blind volumetric end-to-end benchmarking (`nmrassign e2e-benchmark`).
  - **Phase 4 (Sidechain Assignment):** Added IUPAC / PDB CCD covalent & TOCSY topology for all 20 standard amino acids (`nmr-core::topology`), volumetric simulators & pickers for `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, and aromatic experiments, and a joint backbone + sidechain optimizer (`nmr-core::sidechain`).
  - **Phase 5 (3D Structure Folding):** Added chemical-shift $(\phi, \psi)$ torsion restraint prediction (`nmr-core::dihedrals`), 7-cycle $r^{-6}$ ambiguous NOE distance restraint (`ADR`) calibration & network anchoring (`nmr-core::noesy`), and analytical-gradient NeRF torsion-angle simulated annealing ensemble folding with Kabsch RMSD and Ramachandran validation (`nmr-core::structure`).
  - **Phase 6 (Interchange, Reporting & Service):** Added lossless CCPN NEF v1.1 import/export (`nmr-core::nef`), single-file HTML + inline SVG quality reports (`nmr-core::report`), one-command `nmrassign pipeline`, and a self-hosted HTTP REST job server + browser dashboard (`nmr-core::service`).
  - **Phase 7 (Octet Chemical Studio):** Integrated a pure-Rust 2D chemical structure, reaction, stoichiometry, bidirectional IUPAC nomenclature, and 6-sphere HOSE $^{1}\text{H}$/$^{13}\text{C}$ 1D NMR suite (`octet_mol`, `octet_analyt`, `octet_render`) with `SMILES`, `InChI`/`InChIKey`, `MOL`/`SDF`, `CDXML`/`CDX`, `JCAMP-DX`, `SVG`/`PDF`/`PNG` I/O, `nmrassign chem`, and an interactive `egui` studio tab (`octet_canvas`).
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (WebGPU): **97.1% accuracy**, **100.0% high-confidence precision (55/55)**.
  - `synthetic_ubiquitin.str` (CPU): **97.1% accuracy**, **98.0% high-confidence precision (48/49)**.
  - Neural 3D CNN GPU/CPU forward parity: **$< 10^{-4}$**; workspace test suite: **50/50 release tests passing**.

---

## `[0.2.0]` (`5adb55c`) — 2026-11-11 — Rewrite nmr-assign v0.2 in pure Rust + WebGPU with interactive GUI workbench

- **Tag / Commit:** `v0.2.0` (`5adb55c911a185ce7b71f7b294e1899f30bdb09a`)
- **Architectural Summary:**
  - Replaced the initial Python/Julia prototype with a 4-crate pure-Rust + WebGPU workspace (`nmr-core`, `nmr-gpu`, `nmr-cli`, `nmr-gui`).
  - `nmr-core`: NMR-STAR v3/v2.1 & TSV peak I/O, Bruker TopSpin `pdata` (`2rr`/`3rrr`) & NMRPipe (`.ft2`/`.ft3`) spectrum readers, robust MAD + sub-voxel log-Gaussian 2D/3D peak picker, shared-ellipse spin-system builder with proximity tie-breaking and 0-carbon decoy exclusion, configurable `assign_cost`, and multi-threaded `f64` simulated annealing solver.
  - `nmr-gpu`: `wgpu` + WGSL compute shaders for $U(s, k)$ typing and $P(s, t)$ connectivity scoring matrices (`< 1e-13` `f64` reference parity) and workgroup-SRAM parallel multi-replica simulated annealing.
  - `nmr-gui`: Interactive `eframe`/`egui` + `wgpu` desktop workbench with real-time WGSL 2D contour rendering, sequence confidence ribbon, and 3D sequential strip walk plots.
- **Verified Metrics:**
  - `synthetic_ubiquitin.str` (seed 0): **97.1% accuracy**, **97.1% recall**, **100.0% high-confidence precision (51/51)** on WebGPU; **98.0% high-confidence precision** across 12-run CPU annealing in **0.03 s**.

---

## `[0.1.0]` (`01a52e1` / `710b733`) — 2026-11-11 — Initial open NMR pipeline scaffold, agent rules, and backbone assignment v0.1

- **Tag / Commit:** `v0.1.0` (`710b733a1dd319de603778aeb0b8641dbe2bb02c`)
- **Architectural Summary:**
  - Consolidated clean-room provenance rules, permissive dependency allowlists (`MIT`/`BSD`/`Apache-2.0`/`ISC`/`Zlib`), data hygiene policies, and benchmark regression gates into `AGENTS.md`, with `CLAUDE.md` and `GEMINI.md` referencing `AGENTS.md` as the single source of truth.
- **Verified Metrics:**
  - Documentation and rule consolidation across 7 files (`+70 / -61` lines).

---

## `01a52e1` — 2026-11-11 — Initial commit: open NMR pipeline scaffold and backbone assignment v0.1

- **Commit:** `01a52e1a2df2231028bf9ea9f7345bf89061c3ce`
- **Architectural Summary:**
  - Bootstrapped the open-source MIT-licensed repository, module specifications (`docs/specs/00-architecture.md` through `08-service.md` comparing target capabilities), `TODO.md` roadmap, `THIRD_PARTY.md`, synthetic ubiquitin fixture (`data/fixtures/synthetic_ubiquitin.str`), and v0.1 Python reference engine (NMR-STAR I/O, peak simulator, competing-candidate spin-system builder, LLR + Gaussian link scoring, multi-run simulated annealing with consensus confidence, CLI, and regression tests).
- **Verified Metrics:**
  - Initial 30-file scaffold (`2,125` lines) and end-to-end synthetic ubiquitin regression test suite passing.
