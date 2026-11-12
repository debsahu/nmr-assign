# Changelog

All notable architectural changes, algorithm milestones, and verified benchmark results for `nmr-assign` are documented in this file using Semantic Versioning (`vX.Y.Z` annotated git tags, newest releases first).

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
