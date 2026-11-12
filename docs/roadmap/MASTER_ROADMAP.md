# Master Roadmap: All-in-One GPU Biomolecular NMR Platform (`T08`–`T14`)

Clean-room, atomic swarm execution roadmap for `nmr-assign` (`v0.3.x` -> `v1.0`).
Every task is scoped with **strictly disjoint file ownership**, **peer-reviewed clean-room provenance**, an explicit **User/CLI/GUI contract**, and a **quantitative done gate** so multi-agent swarms can implement, verify, and externally audit (`scripts/review_pack.sh --task Txx.y`) each slice independently.

- **Non-negotiable rules**: `AGENTS.md` (clean-room provenance, MIT/BSD/Apache-2.0 allowlist, zero proprietary/GPL inspection, zero forbidden product names in modules/files, unlabelled-only reference offsets, sequence-only backbone benchmarks, strict `dev` vs `heldout` firewall).
- **Status tracking**: `TODO.md`.
- **External Claude review workflow**: `docs/roadmap/REVIEW_PROTOCOL.md` (`sh scripts/review_pack.sh --task Txx.y <git_range>`).

---

## Phase Summary Matrix

| Phase | ID Range | Scope | Primary Crates | Gate Summary |
|---|---|---|---|---|
| **Phase 8** | `T08.1`–`T08.7` | Raw FID readers, NUS IST/FISTA, composable DSP & autophasing, frozen BMRB benchmark | `nmr-core`, `nmr-cli` | Ubiquitin raw `ser` within `+-2.0` pts of `pdata/1`; `CISD3` 3D NUS validated; single-shot cold `heldout` |
| **Phase 9** | `T09.1`–`T09.7` | Zero-friction folder ingest, workflow stepper, two-key NMR shortcuts, overlays, undo/NEF, solver follow-ups | `nmr-core`, `nmr-gui` | `< 50 ms` folder classification; two-key modal chords + `wasm32` build green; `Gln2` / sparse-data gates |
| **Phase 10** | `T10.1`–`T10.5` | Live 2D covalent sidechain graph, aromatic rings, stereospecificity & supramolecular Methyl-TROSY | `nmr-core`, `nmr-gpu`, `nmr-gui` | Aromatic `>= 90%` acc; Methyl-TROSY GPU graph solver `>= 90%` acc / `>= 95%` HC prec on `ILVM` benchmarks |
| **Phase 11** | `T11.1`–`T11.4` | Problem-regime GPU solvers: protonless `13C`-detected IDPs & paramagnetic `[2Fe-2S]` / lanthanide `PCS`/`PRE` | `nmr-core`, `nmr-gpu`, `nmr-cli` | Continuous through-Proline IDP walk `>= 92%` acc; `CISD3` `[2Fe-2S]` bleaching model improves recall without HC loss |
| **Phase 12** | `T12.1`–`T12.4` | Built-in `wgpu` 3D biomolecular PDB viewer, 3D-to-strip picking, post-assignment torsions & GPU `RDC` refinement | `nmr-gui`, `nmr-core`, `nmr-gpu` | `60 fps` desktop+`wasm32` ribbons; Cornilescu `Q < 0.25` after GPU `RDC` refinement |
| **Phase 13** | `T13.1`–`T13.6` | GPU spin relaxation ($R_1, R_2$, hetNOE), Lipari–Szabo / Palmer Model-Free ($S^2$), `CEST`, `CPMG` & ligand `CSP` $K_d$ | `nmr-core`, `nmr-gpu`, `nmr-gui`, `nmr-cli` | `< 1e-4` parameter recovery on analytical synthetic `M1`–`M5`, 2-state `CEST`/`CPMG`, and tight-binding $K_d$ |
| **Phase 14** | `T14.1`–`T14.4` | Deterministic SVG/PNG figure exporter, slim visual `README.md`, GitHub-Pages manual & zero-install WebGPU demo | `nmr-cli`, `nmr-gui`, `docs/` | Reproducible CI-built manual + live `wasm32` demo bundle `< 15 MiB` loading in `< 2 s` |

---

## Phase 8: Raw Time-Domain (FID) Processing, NUS, DSP Operators & Frozen Real-Data Benchmark (`T08.1`–`T08.7`)

### T08.1 — Clean-Room Raw FID, Hypercomplex NUS & Benchmark Specification
- **Task ID**: `T08.1` (Status: `[x]` completed)
- **Owned Files/Modules**: `docs/specs/11-fid-processing.md`
- **Clean-Room Literature Citations**: States, Haberkorn & Ruben (*J. Magn. Reson.* 1982); Marion et al. (*J. Magn. Reson.* 1989); Palmer et al. (*J. Magn. Reson.* 1991); Kay, Keifer & Saarinen (*JACS* 1992); Moskau (*Concepts Magn. Reson.* 2002); Otting et al. (*J. Magn. Reson.* 1986); Zhu & Bax (*J. Magn. Reson.* 1990); Hoch & Stern (*NMR Data Processing*, 1996); Chen et al. (*J. Magn. Reson.* 2002); Stern, Donoho & Hoch (*J. Magn. Reson.* 2007); Hyberts et al. (*J. Biomol. NMR* 2012).
- **User/CLI/GUI Contract**: Defines `Fid1D`/`Fid2D`/`Fid3D` hypercomplex layout, Bruker `acqus`/`ser` & Varian `procpar`/`fid` decoding, NUS `nus_mask` auto-detection & IST/FISTA reconstruction, 9-step `ProcessingRecipe` JSON schema, Part 3 TopSpin parity protocol, `fetch-bmrb`, and frozen `dev`/`heldout` benchmark governance.
- **Quantitative Done Gate**: Spec committed with zero forbidden tool names and unambiguous mathematical equations for every quadrature, DSP, NUS, and referencing step.

### T08.2 — Raw Time-Domain FID & NUS Readers (`Bruker` & `Varian/Agilent`)
- **Task ID**: `T08.2` (Status: `[x]` verified on `swarm-spec-fid`, pending final merge)
- **Owned Files/Modules**: `crates/nmr-core/src/fid.rs`, `crates/nmr-core/tests/fid_tests.rs`, `docs/specs/11-fid-processing.md` (§1)
- **Clean-Room Literature Citations**: States et al. (*J. Magn. Reson.* 1982); Marion et al. (*J. Magn. Reson.* 1989); Kay, Keifer & Saarinen (*JACS* 1992); Moskau (*Concepts Magn. Reson.* 2002); Westler & Abildgaard (1996 public DSPFVS/DECIM table); Wishart et al. (*J. Biomol. NMR* 1995); public Bruker JCAMP-DX v5.0 & Varian/Agilent VnmrJ binary header specifications.
- **User/CLI/GUI Contract**: `read_fid_auto`, `read_fid_2d`, `read_fid_3d`, `Fid1D`/`Fid2D`/`Fid3D` with `nus_mask: Option<Vec<bool>>`, Linux `ext4` case-insensitive child lookup (`find_child_ci`), TopSpin 4 `FnTYPE=0` uniform guard vs `FnTYPE=2` `NusTD` + `nuslist`, Varian `sampling='s'` uniform vs `'sparse'` + `sampling.sch`, 3D `f1coef` precedence over stale 2D `fcoef`, and loud errors on unsupported `FnMODE`/`phase` or missing schedules.
- **Quantitative Done Gate**: `11/11` release `fid_tests` pass (1D/2D/3D uniform & NUS, uppercase Linux filenames, real Ubiquitin `Echo-Antiecho` & `CISD3` 3D NUS `States-TPPI`); zero new dependencies; `wasm32-unknown-unknown` clean.

### T08.3 — Deterministic Processing Engine, Hypercomplex IST/FISTA NUS & Part 3 TopSpin Validation
- **Task ID**: `T08.3` (Status: `[x]` verified on `swarm-proc-engine`, pending final merge)
- **Owned Files/Modules**: `crates/nmr-core/src/process.rs`, `crates/nmr-core/src/spectrum.rs` (3D `.ft3` / `3rrr` writers), `crates/nmr-core/tests/process_tests.rs`
- **Clean-Room Literature Citations**: Cooley & Tukey (*Math. Comput.* 1965); Bluestein (*IEEE Trans. Audio Electroacoust.* 1970); Otting et al. (*J. Magn. Reson.* 1986); Marion, Ikura & Bax (*J. Magn. Reson.* 1989); Zhu & Bax (*J. Magn. Reson.* 1990); Hoch & Stern (1996); Chen et al. (*J. Magn. Reson.* 2002); Stern, Donoho & Hoch (*J. Magn. Reson.* 2007); Beck & Teboulle (*SIAM J. Imaging Sci.* 2009); Hyberts et al. (*J. Biomol. NMR* 2012).
- **User/CLI/GUI Contract**: `ProcessingRecipe` JSON round-trip, `process_fid_1d`/`2d`/`3d`, `process_fid_dir_2d`/`3d`, 9-step deterministic pipeline (digital filter shift + sub-sample linear phase, Gaussian solvent filter, 1D/2D bicomplex quaternion FISTA/IST NUS reconstruction, apodization, opt-in `--lp` `false` by default, radix-2/Bluestein FFT + `REVERSE=yes` support, stored phases with Chen 2002 fallback only when absent, Legendre polynomial baseline, `1H` ROI crop), and `write_nmrpipe_3d` / `write_bruker_pdata_3d`.
- **Quantitative Done Gate**: Local 5-spectrum Ubiquitin raw `ser` end-to-end `evo-graph` assignment within `+-2.0` percentage points of TopSpin `pdata/1` (`95.45%` acc / `90.00%` rec achieved, `+1.42` / `+0.00` pts) with **zero** picker or solver edits; `CISD3` 3D NUS `CBCACONH` processed in `< 3 s` with `>= 59.4%` 5-seed `evo-graph` accuracy (`62.50%` / `100.0%` HC prec achieved).

### T08.4 — Composable Multidimensional DSP Operator Suite & Blind 1D/2D/3D Autophasing
- **Task ID**: `T08.4` (Status: `[ ]` in progress on `swarm-dsp-ops`)
- **Owned Files/Modules**: `crates/nmr-core/src/ops.rs`, `crates/nmr-core/tests/ops_tests.rs`
- **Clean-Room Literature Citations**: Delaglio et al. (*J. Biomol. NMR* 1995, 6, 277–293 — composable multidimensional vector-stream DSP operators); Bartholdi & Ernst (*J. Magn. Reson.* 1973 — Hilbert transform imaginary reconstruction); Ernst, Bodenhausen & Wokaun (1987 — Lorentz-to-Gauss `GM`, trapezoid `TM`, `J`-modulation windows); Zhu & Bax (*J. Magn. Reson.* 1990 — forward/backward/mirror-image LP); Friedrichs (*J. Biomol. NMR* 1995 — median baseline); Chen et al. (*J. Magn. Reson.* 2002 — entropy autophasing extended to biphasic `HNCACB` indirect axes).
- **User/CLI/GUI Contract**: Composable `DspOp` pipeline (`SineWindow`, `ExpWindow`, `LorentzToGauss`, `Trapezoid`, `JModulation`, `SolventFilter` Gaussian/SineBell/Boxcar, `LinearPredict` Forward/Backward/MirrorImage, `HilbertTransform`, `Phase`, `DeleteImaginary`, `ReverseAxis`, `CircularShiftPpm`, `PolyBaseline`, `MedianBaseline`), `.ft1` & plane-series `test%03d.ft3` readers/writers, and blind `auto_phase_direct_nd` + `auto_phase_indirect_nd` (including `Biphasic` `+-` mode for `HNCACB`).
- **Quantitative Done Gate**: `HilbertTransform -> Phase -> DeleteImaginary` matches complex phasing to `< 1e-3` relative error; blind autophase on raw Ubiquitin `ser` (ignoring `procs` `PHC0`/`PHC1`) recovers `(p0, p1)` within `<= 5 deg` of `procs` and preserves positive/negative `CA`/`CB` polarity on `HNCACB`.

### T08.5 — Two-Pass Experiment-Guided Corroborated 3D Strip Re-Picking
- **Task ID**: `T08.5` (Status: `[x]` verified on `swarm-repick-3d`, pending final merge)
- **Owned Files/Modules**: `crates/nmr-core/src/picker.rs`, `crates/nmr-core/tests/corroborated_repick_tests.rs`, `docs/specs/02-peak-picking.md`
- **Clean-Room Literature Citations**: Kay, Ikura, Tschudin & Bax (*J. Magn. Reson.* 1990); Grzesiek & Bax (*JACS* 1992 — intra/inter-residue triple-resonance pathway sharing).
- **User/CLI/GUI Contract**: Streaming `StripBank3D` (`~1–8 MB`/cube vs `500 MB` full 3D volume) + `corroborated_re_pick_3d` rescuing `3.8–4.2 sigma` local 3D maxima strictly inside narrow `+-0.15 ppm` (`CA`/`CB`) / `+-0.12 ppm` (`CO`) windows anchored by already-observed `>= 5.5 sigma` experimental peaks (zero sequence/prior inputs, zero confirmation bias).
- **Quantitative Done Gate**: Zero regression on 5-seed `--realistic` Ubiquitin, real Ubiquitin `pdata`/`.ucsf`, and 5-seed `CISD3` (`0` decoy/`NH2` false positives; rescues corroborated weak `i-1` peaks such as Ubiquitin `GLY75` `CA(i-1)`).

### T08.6 — Rate-Limited Public BMRB Fetcher & Frozen 14-Entry `dev`/`heldout` Manifest
- **Task ID**: `T08.6` (Status: `[ ]` implemented on `swarm-cli-bench`, pending final wiring)
- **Owned Files/Modules**: `crates/nmr-cli/src/fetch_bmrb.rs`, `data/benchmark/real_manifest.tsv`
- **Clean-Room Literature Citations**: Ulrich et al. (*Nucleic Acids Res.* 2008 — BMRB public CC0 archive & NMR-STAR v3.2); FIPS 180-4 (`SHA-256` digest verification).
- **User/CLI/GUI Contract**: `nmrassign fetch-bmrb --id <bmrb_id> --dest data/private/bmrb/<id>/ [--all-archives]` querying live `entry_directories/bmr<id>/` (`<= 1 req/s`, `User-Agent: nmrassign-fetch`), rejecting path traversal/symlinks across `.tar`/`.tar.gz`/`.tgz`/`.zip`, writing `PROVENANCE.md` with source URLs, CC0 notice, UTC timestamp, sizes, and SHA-256 digests; plus frozen `data/benchmark/real_manifest.tsv` (`5 dev` + `9 heldout`, 100% disjoint from `manifest.tsv` and prior training entries).
- **Quantitative Done Gate**: Unit tests pass on NIST SHA-256 vectors, traversal/symlink tar/zip rejection, multi-archive isolation (`53371`/`51351`), and manifest schema/disjointness checks.

### T08.7 — CLI Raw-FID Integration, `real-benchmark` Runner & Single-Shot Cold `heldout` Evaluation
- **Task ID**: `T08.7` (Status: `[ ]` open)
- **Owned Files/Modules**: `crates/nmr-cli/src/main.rs`, `crates/nmr-cli/src/real_benchmark.rs`
- **Clean-Room Literature Citations**: Wishart et al. (*J. Biomol. NMR* 1995); `docs/specs/11-fid-processing.md` (§3, §5).
- **User/CLI/GUI Contract**: Wires `nmrassign process` (with `ProcessingRecipe` JSON, `--no-crop`, `--lp`, `--auto-phase`), `--fid` / `--fid-hsqc` / `--fid-triple` on `pick` and `pipeline` (including `StripBank3D` Pass-2 hook), and `nmrassign real-benchmark --manifest data/benchmark/real_manifest.tsv --split dev|heldout [--solver standard|evo-graph] [--seeds 5] [--json-out <path>]` using strictly unlabelled global offset fitting (`fit_unlabelled_global_offsets`) and skipping missing `data/private/` directories cleanly with exit code `0`.
- **Quantitative Done Gate**: Full CI suite green (`fmt`, `clippy -D warnings`, `wasm32`, `cargo test --release --workspace`); `dev` split verified first; `heldout` split executed **once** for honest reporting with zero subsequent parameter/threshold/code edits.

---

## Phase 9: Zero-Friction Spectrometer Ingest, Interactive NMR Workbench UX & Solver Curation (`T09.1`–`T09.7`)

### T09.1 — Spectrometer Folder & Archive Auto-Classifier
- **Task ID**: `T09.1`
- **Owned Files/Modules**: `crates/nmr-core/src/classify.rs`, `crates/nmr-core/tests/classify_tests.rs`, `docs/specs/12-workbench-ux.md` (§1)
- **Clean-Room Literature Citations**: Kay, Ikura, Tschudin & Bax (*J. Magn. Reson.* 1990); Grzesiek & Bax (*JACS* 1992); Bermel et al. (*JACS* 2009); Tugarinov & Kay (*JACS* 2003); public Bruker JCAMP-DX (`PULPROG`, `NUC1`, `SW`, `FnTYPE`, `vdlist`, `vclist`, `fqlist`, `nuslist`) and Varian BioPack (`seqfil`, `pslabel`, `tn`/`dn`/`dn2`, `array`, `sampling`) parameter conventions.
- **User/CLI/GUI Contract**: `classify_NMR_workspace(path_or_entries) -> WorkspaceInventory` scanning nested directories or `.zip`/`.tar*` headers (case-insensitively) to identify each experiment's vendor, dimensionality (`1D`/`2D`/`3D`/`pseudo-3D`/`4D`), raw-FID vs processed status, pulse-sequence class (`15N-HSQC`, `13C-HSQC`, `13C-HMQC`, `HNCO`, `HNCACO`, `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`, `CCCONH`, `HCCCONH`, `HCCH-TOCSY`, `15N/13C-NOESY`, `CON`/`CACO`/`CANCO`/`CBCACO`, `R1`/`R2`/`hetNOE`, `CPMG`, `CEST`, titration series), and recommended problem regime (`globular-backbone`, `sidechain-noesy`, `idp-13c-detected`, `methyl-trosy`, `paramagnetic`, `relaxation-dynamics`, `titration-csp`); exposed via `nmrassign inspect <dir>`.
- **Quantitative Done Gate**: `100%` classification accuracy across synthetic + real Bruker (`Ubiquitin`, `CISD3`, `BMRB 53371`) and Varian BioPack (`BMRB 18465`, `51351`) header fixtures in `< 50 ms` without reading bulk `ser`/`fid` signal bytes; `wasm32` compatible.

### T09.2 — Drag-and-Drop Folder/Archive Loader & Editable Experiment Selection Matrix
- **Task ID**: `T09.2`
- **Owned Files/Modules**: `crates/nmr-gui/src/ingest_panel.rs`, `crates/nmr-core/src/service.rs` (`/api/v1/inspect` & archive ingest), `crates/nmr-gui/src/main.rs` (ingest hook only)
- **Clean-Room Literature Citations**: Original clean-room interactive scientific UI design built on `egui` / `eframe` and `nmr-core::classify`.
- **User/CLI/GUI Contract**: Dragging a spectrometer parent folder, `.zip`/`.tar.gz` bundle, `.fasta`, or `.nef` onto `nmr-gui` (desktop or `wasm32` browser) or the `nmrassign serve` web dashboard opens an **Experiment Selection Matrix** showing auto-detected experiment role, nuclei/axes, `MHz`, quadrature/NUS badge, override dropdowns, include/exclude checkboxes, and a 1-click **Process, Pick & Run** button with live per-experiment progress bars.
- **Quantitative Done Gate**: End-to-end drop of a multi-experiment Bruker/Varian bundle populates the matrix with zero manual path flags, handles misnamed subfolders cleanly, and compiles cleanly on native desktop and `wasm32-unknown-unknown`.

### T09.3 — Guided *"What Next?"* Workflow Stepper, Top Menu Bar & Command Palette (`Cmd/Ctrl+K`)
- **Task ID**: `T09.3`
- **Owned Files/Modules**: `crates/nmr-gui/src/workflow_bar.rs`, `crates/nmr-gui/src/command_palette.rs`, `docs/specs/12-workbench-ux.md` (§2)
- **Clean-Room Literature Citations**: Original clean-room human-computer interaction design for progressive scientific workflows.
- **User/CLI/GUI Contract**: Adds (1) a persistent 6-stage **Workflow Stepper** (*1. Ingest -> 2. Process FIDs -> 3. Pick Peaks -> 4. Assign Spins -> 5. Fold / Dynamics -> 6. Export & Report*) highlighting current readiness, missing inputs, and a single contextual primary action button (*"What Next?"*), (2) a structured **Top Menu Bar** (*File, Process, Spectrum, Peaks, Assignment, Structure, Dynamics, Chemistry, Help*), and (3) a fuzzy-scored **`Cmd/Ctrl+K` Command Palette** listing every action alongside its two-key or modifier shortcut.
- **Quantitative Done Gate**: Deterministic state-machine unit tests covering all 6 workflow transitions, `< 1 ms` fuzzy search across `>= 50` registered commands, and clean `wasm32` compilation.

### T09.4 — Interactive Two-Key NMR Spectrum Keyboard Accelerators
- **Task ID**: `T09.4`
- **Owned Files/Modules**: `crates/nmr-gui/src/keymap.rs`, `crates/nmr-gui/tests/keymap_tests.rs`, `docs/specs/12-workbench-ux.md` (§3)
- **Clean-Room Literature Citations**: Original clean-room finite-state chord input architecture implementing interactive two-key NMR spectrum accelerators comparable to modern desktop assignment suites (zero external source or keytable files consulted).
- **User/CLI/GUI Contract**: Modal two-letter chord dispatcher (`450 ms` timeout with subtle status-bar chord indicator, automatically bypassed whenever text inputs or `Octet-Draw` editors hold focus): `vt` (view/contour threshold popup), `pa` (add peak at crosshair), `pd` (delete selected peak(s)), `pc` (parabolic sub-voxel peak center snap), `pl` (toggle peak labels), `oz` (zoom out to full sweep), `zi`/`zo` (zoom in/out), `ct`/`cb` (raise/lower contour floor by `1.25x`), `st` (tile sequential strips for selected root), `at` (open residue assignment popover), `Space` (toggle crosshair lock), and `?` (searchable keyboard cheat-sheet modal).
- **Quantitative Done Gate**: `100%` state-transition unit test coverage (valid chords, timeout expiry, `Esc` cancel, text-widget focus suppression, modifier passthrough) with zero `clippy` warnings on native and `wasm32`.

### T09.5 — Synchronized 2D/3D/4D Multi-Spectrum Overlays & `i-1 / i / i+1` Match-Finding Strips
- **Task ID**: `T09.5`
- **Owned Files/Modules**: `crates/nmr-gui/src/strip_tiler.rs`, `crates/nmr-gui/src/contour_painter.rs`, `crates/nmr-gui/src/contour_shader.wgsl`
- **Clean-Room Literature Citations**: Kay et al. (*J. Magn. Reson.* 1990); Grzesiek & Bax (*JACS* 1992 — sequential `CA`/`CB`/`C'` strip walk matching).
- **User/CLI/GUI Contract**: Supports rendering an arbitrary stack of overlaid 2D spectra / 3D orthogonal planes in the GPU `wgpu` contour canvas (per-spectrum positive/negative color palette, contour floor, geometric factor, visibility, and ppm shift nudge), synchronized `(1H, 15N, 13C)` cursor crosshairs across all open views, and an interactive **Candidate Match-Finder Strip Tiler** ranking top-$K$ predecessor (`i-1`) and successor (`i+1`) spin systems around the active strip by exact `P(s, t)` connectivity log-likelihood with horizontal dashed `CA`/`CB`/`C'` alignment guides.
- **Quantitative Done Gate**: Maintains `>= 60 fps` pan/zoom with `4` simultaneous 2D/slice overlays (`2048 x 1024`) on WebGPU; top-3 match-finder recovers `>= 97%` of true sequential neighbors on Ubiquitin `.ucsf`.

### T09.6 — Bounded Undo/Redo Command Stack & Lossless NEF Session Persistence
- **Task ID**: `T09.6`
- **Owned Files/Modules**: `crates/nmr-core/src/session.rs`, `crates/nmr-core/src/nef.rs` (workbench saveframe extensions), `crates/nmr-gui/src/session_io.rs`
- **Clean-Room Literature Citations**: Gutmanas et al. (*Nat. Struct. Mol. Biol.* 2015 — CCPN NMR Exchange Format `NEF v1.1` extensibility specification).
- **User/CLI/GUI Contract**: Reversible `WorkbenchCommand` log (`AddPeak`, `DeletePeaks`, `MovePeak`, `RecenterPeak`, `AssignSpinSystem`, `LockResidue`, `UnassignResidue`, `SetContourParams`, `SetReferenceOffset`) wired to `Cmd/Ctrl+Z` / `Cmd/Ctrl+Shift+Z` (`256`-step ring buffer), plus single-file `.nef` project save/load preserving peaks, chemical shifts, user locks, `ProcessingRecipe` metadata, and active contour/view state across native files and `wasm32` browser downloads/uploads.
- **Quantitative Done Gate**: Property/round-trip tests verifying `apply -> undo == identity` and `apply -> undo -> redo == apply` across all command variants, plus bit-exact `.nef` session save -> reload parity.

### T09.7 — Backbone Solver Curation, Minor-Conformer Penalty & Sparse-Data Convergence
- **Task ID**: `T09.7`
- **Owned Files/Modules**: `crates/nmr-core/src/spinsystems.rs`, `crates/nmr-core/src/solver.rs`, `crates/nmr-core/src/evo_graph.rs`, `crates/nmr-core/tests/solver_curation_tests.rs`, `docs/specs/03-assignment.md` *(coordinate any §3/§4 constant proposals with maintainer per `AGENTS.md`)*
- **Clean-Room Literature Citations**: Schmidt & Güntert (*JACS* 2012); Klukowski, Riek & Güntert (*Nat. Commun.* 2022); Wishart et al. (*J. Biomol. NMR* 1995); Schwarzinger et al. (*JACS* 2001).
- **User/CLI/GUI Contract**: (1) Relative HSQC/3D root-intensity prior penalizing sub-stoichiometric `< 15%`-median minor-conformer peaks (such as Ubiquitin `Gln2` 9% minor second-form root) from displacing major-state assignments, (2) user-locked `(spin_system -> residue)` hard constraints honored in both `standard` (CPU/GPU) and `evo-graph` for interactive re-annealing, (3) sparse-data (`HNCACB`-absent / `CA`-only) search proposal improvements closing the `standard` vs `evo-graph` gap on `CISD3`, and (4) reliability-diagram (`ECE`) confidence calibration audit.
- **Quantitative Done Gate**: Zero regression on 5-seed `--realistic` Ubiquitin and 5-seed `CISD3` gates (`AGENTS.md` DoD); resolves Ubiquitin `Gln2` major-vs-minor conformer swap; Expected Calibration Error `ECE <= 0.06` at `conf >= 0.80` on development real-spectrum benchmarks.

---

## Phase 10: Interactive Covalent Sidechain Graph, Aromatics, Stereospecificity & Methyl-TROSY (`T10.1`–`T10.5`)

### T10.1 — Interactive Sequence Ribbon & Live 2D Covalent Amino-Acid Sidechain Graph
- **Task ID**: `T10.1`
- **Owned Files/Modules**: `crates/nmr-gui/src/sidechain_widget.rs`, `crates/nmr-core/src/topology.rs` (2D atom layout coordinates)
- **Clean-Room Literature Citations**: Markley et al. (*J. Mol. Biol.* 1998 — IUPAC-IUBMB-IUPAB biochemical NMR atom nomenclature); wwPDB Chemical Component Dictionary (`CCD` public domain).
- **User/CLI/GUI Contract**: Clicking any residue on the top sequence ribbon renders its exact IUPAC covalent 2D amino-acid atom-bond diagram (`N`, `HN`, `CA`, `HA`, `C`, `O`, `CB`..`CZ`/`NH1`/`NE1`/`SD`) color-coded by assignment status & BMRB Z-score (`|z| <= 1.5` green, `1.5..3.0` amber, `> 3.0` outlier red, unassigned hollow), displaying assigned `ppm` + uncertainty on each atom node and jumping the 2D/3D spectrum strips to that atom's frequency upon click.
- **Quantitative Done Gate**: Unit tests verifying planar non-overlapping 2D layout coordinates and exact IUPAC atom labels for all 20 canonical amino acids (plus oxidized/reduced `Cys` and protonation states of `His`); `wasm32` clean.

### T10.2 — Aromatic Ring Spin-System Assignment (`Phe`, `Tyr`, `Trp`, `His`)
- **Task ID**: `T10.2`
- **Owned Files/Modules**: `crates/nmr-core/src/aromatic.rs`, `crates/nmr-core/tests/aromatic_tests.rs`, `docs/specs/07-sidechain-assignment.md`
- **Clean-Room Literature Citations**: Yamazaki, Forman-Kay & Kay (*J. Am. Chem. Soc.* 1993, 115, 11054–11055 — `(HB)CB(CGCD)HD` & `(HB)CB(CGCDCE)HE` experiments); Pelton et al. (*Protein Sci.* 1993 — `His` `15N` tautomeric states from `15N-HMQC` shifts).
- **User/CLI/GUI Contract**: Extends `nmrassign sidechain` with `pick_aromatic_2d` / `assign_aromatic_rings`, linking assigned backbone `CB(i)` shifts to `HD*`, `HE*`, `HZ`, `CD*`, `CE*`, `CZ` (`Phe`/`Tyr`), indole `HD1`/`HE1`/`NE1`/`HZ2`/`HZ3`/`HH2` (`Trp`), and imidazole `HD2`/`HE1`/`ND1`/`NE2` (`His`, including `Nδ1-H` vs `Nε2-H` vs doubly protonated tautomer classification).
- **Quantitative Done Gate**: `>= 92%` aromatic `(1H, 13C/15N)` assignment accuracy and `>= 96%` high-confidence precision across simulated noisy spectra derived from the 12-protein BMRB benchmark suite.

### T10.3 — Bayesian Stereospecific Prochiral (`HB2`/`HB3`, `HG1`/`HG2`, `HD1`/`HD2`) Assignment
- **Task ID**: `T10.3`
- **Owned Files/Modules**: `crates/nmr-core/src/stereospec.rs`, `crates/nmr-core/tests/stereospec_tests.rs`
- **Clean-Room Literature Citations**: Güntert, Braun, Billeter & Wüthrich (*J. Biomol. NMR* 1991 — conformational database `HABAS` local scalar-coupling & intraresidue NOE analysis); Nilges (*J. Mol. Biol.* 1995 — floating chirality / ambiguous prochiral treatment); Lovell et al. (*Proteins* 2000 — penultimate rotamer library).
- **User/CLI/GUI Contract**: Computes posterior probabilities over `(g+, t, g-)` $\chi_1$ / $\chi_2$ rotamers and the two prochiral swap states (`1->2` vs `1->3`) from optional `3J_HNHB` / `3J_HAHB` couplings, intraresidue/sequential `HN-HB`, `HA-HB`, `HB-HN(i+1)` NOE peak ratios, and BMRB shift priors; emits IUPAC stereospecific labels (`HB2`/`HB3`) only when posterior `>= 0.90` and standards-compliant non-stereospecific `NEF` `HBx`/`HBy` (`_Atom_chem_shift.Ambiguity_code = 2`) otherwise.
- **Quantitative Done Gate**: `>= 95%` precision on stereospecifically committed methylene/isopropyl pairs (`posterior >= 0.90`) on synthetic Ubiquitin (`1D3Z`) NOE + coupling benchmarks, with `0` uncalibrated false swaps exported as `Ambiguity_code = 1`.

### T10.4 — Supramolecular `13C`-Methyl TROSY (`ILVMAT`) Picker & Isotopic Labeling Schemes
- **Task ID**: `T10.4`
- **Owned Files/Modules**: `crates/nmr-core/src/methyl_trosy.rs`, `crates/nmr-core/tests/methyl_trosy_tests.rs`, `docs/specs/13-methyl-trosy.md` (§1–§2)
- **Clean-Room Literature Citations**: Tugarinov, Hwang, Ollerenshaw & Kay (*J. Am. Chem. Soc.* 2003, 125, 10420–10428 — `1H-13C` methyl-TROSY HMQC in high-MW proteins); Gans et al. (*Angew. Chem. Int. Ed.* 2010 — stereospecific `pro-R`/`pro-S` `Leu`/`Val` labeling); Ruschak & Kay (*J. Biomol. NMR* 2010).
- **User/CLI/GUI Contract**: Configurable precursor labeling schemes (`ILV-racemic`, `ILV-proR`, `ILV-proS`, `ILVMAT`, custom residue/atom masks), amino-acid methyl-type classifier from `2D 1H-13C HMQC` regions (`Ile-δ1` `~11–15 ppm`, `Ala-β` `~16–22 ppm`, `Thr-γ2`, `Met-ε`, `Leu-δ`/`Val-γ` `~18–27 ppm`) + optional `HMCM(CG)CBCA` geminal pair linking, and `3D C-C-H` / `4D C-H-C-H` HMQC-NOESY-HMQC cross-peak picker + simulator.
- **Quantitative Done Gate**: Picks `>= 96%` of simulated `ILVMAT` methyl roots and `r <= 8.5 A` methyl–methyl NOE cross-peaks on `40–80 kDa` linewidth regimes (`1H` LW `25 Hz`, `13C` LW `18 Hz`) with `< 3%` artifact rate.

### T10.5 — WebGPU Methyl-TROSY Structure-Based Contact-Graph Assignment Solver
- **Task ID**: `T10.5`
- **Owned Files/Modules**: `crates/nmr-core/src/methyl_solver.rs`, `crates/nmr-gpu/src/methyl_gpu.rs`, `crates/nmr-gpu/src/shaders/methyl_anneal.wgsl`, `crates/nmr-cli/src/methyl_cmd.rs`, `docs/specs/13-methyl-trosy.md` (§3)
- **Clean-Room Literature Citations**: Xu et al. (*J. Am. Chem. Soc.* 2009, 131, 9480–9481 — automated methyl assignment from NOE networks); Chao, Shi, Masterson & Veglia (*J. Magn. Reson.* 2012, 214, 103–110); Pritišanac, Degiacomi, Alderson, Carneiro, Ab, Baldwin & Kay (*J. Am. Chem. Soc.* 2017, 139, 9523–9533 — probabilistic methyl NOE contact-graph matching).
- **User/CLI/GUI Contract**: `nmrassign methyl --pdb <model.pdb> --hmqc <2d> --noesy <3d_or_4d> --labeling ILV-proS [--engine gpu|cpu]` scoring methyl-type shift log-likelihoods, geminal `Leu`/`Val` pairing, and bidirectional $r_{\text{CC}}^{-6}$ contact-graph edge agreement via parallel multi-replica GPU simulated annealing + consensus marginal calibration.
- **Quantitative Done Gate**: CPU-vs-WebGPU objective parity `< 1e-4`; `>= 90%` methyl assignment accuracy and `>= 95%` high-confidence (`>= 0.80`) precision on `MBP` (`370 aa`, public BMRB/PDB `1EZP`) and Ubiquitin `ILVMAT` benchmarks in `< 2.0 s` on WebGPU.

---

## Phase 11: Problem-Regime GPU Solvers — IDPs (`13C`-Detected) & Paramagnetic Metalloproteins (`T11.1`–`T11.4`)

### T11.1 — Protonless `13C`-Detected IDP Experiment Models, Picker & Volumetric Simulator
- **Task ID**: `T11.1`
- **Owned Files/Modules**: `crates/nmr-core/src/c13_detected.rs`, `crates/nmr-core/tests/c13_detected_tests.rs`, `docs/specs/14-regime-solvers.md` (§1)
- **Clean-Room Literature Citations**: Bermel, Bertini, Felli, Piccioli & Pierattelli (*Prog. Nucl. Magn. Reson. Spectrosc.* 2006, 48, 25–45); Bermel et al. (*J. Am. Chem. Soc.* 2006, 128, 3918–3919 — `13C` direct detection in IDPs); Bermel et al. (*J. Am. Chem. Soc.* 2009, 131, 15339–15345 — `3D CANCO` / `CBCACON` / `CCC(CO)N`).
- **User/CLI/GUI Contract**: Defines `C13DetectedExp` (`Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D`, `Cccon3D`, `Ncaco3D`), narrow-dispersion IPAP/virtual-decoupled `2D CON` (`C'(i-1), N(i)`) root picker (`15N` `105–138 ppm` including `Pro` imino `132–142 ppm`, `13C'` `170–179 ppm`), 3D `CON`-anchored strip picker, and realistic IDP volumetric cube simulator (`1H`-exchange-broadened vs sharp `13C'-15N` lines).
- **Quantitative Done Gate**: Recovers `>= 97%` of `CON(i)` dipeptide roots — including `-Xaa-Pro-` and `-Pro-Pro-` links — on simulated proline-rich IDP cubes (`20% Pro`, e.g. public BMRB IDP depositions).

### T11.2 — Continuous Through-Proline `13C`-Detected IDP Backbone Solver (`CPU` & `WebGPU`)
- **Task ID**: `T11.2`
- **Owned Files/Modules**: `crates/nmr-core/src/idp_solver.rs`, `crates/nmr-gpu/src/shaders/scoring.wgsl` (dipeptide `(i-1, i)` prior mode), `crates/nmr-core/tests/idp_solver_tests.rs`, `docs/specs/14-regime-solvers.md` (§2)
- **Clean-Room Literature Citations**: Tamiola, Acar & Mulder (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP` sequence-neighbor random-coil chemical shift library); Kjaergaard & Poulsen (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011); Bastidas et al. (*J. Biomol. NMR* 2015).
- **User/CLI/GUI Contract**: `nmrassign assign --regime idp-13c` building `CON(i)` dipeptide spin systems (`N(i), C'(i-1), CA(i), CA(i-1), CB(i-1), C'(i), N(i+1)`) and optimizing unbroken chains directly through Prolines (`Pro` is an active spin-system node rather than a chain break!) using temperature/pH-aware `ncIDP`-class neighbor-corrected coil priors (`sigma_CA ~ 0.45 ppm`) on CPU `evo-graph` and WebGPU multi-replica annealing.
- **Quantitative Done Gate**: `>= 92%` sequence-specific accuracy and `>= 96%` high-confidence precision on public BMRB IDP benchmark entries (including alpha-synuclein / tau-repeat / proline-rich test sets) where `1H`-detected `HSQC` suffers severe overlap or exchange broadening; CPU/WebGPU `U`/`P` parity `< 1e-4`.

### T11.3 — IDP Secondary Structure Propensity (`SSP` / $\delta\text{2D}$) & Ensemble Reports
- **Task ID**: `T11.3`
- **Owned Files/Modules**: `crates/nmr-core/src/idp_propensity.rs`, `crates/nmr-core/src/report.rs` (IDP propensity SVG track), `crates/nmr-core/tests/idp_propensity_tests.rs`
- **Clean-Room Literature Citations**: Marsh, Singh, Jia & Forman-Kay (*Protein Sci.* 2006, 15, 2795–2804 — `SSP` score); Camilloni, De Simone, Vranken & Vendruscolo (*Biochemistry* 2012, 51, 2224–2231 — $\delta\text{2D}$ populations of $\alpha$-helix, $\beta$-strand, `PPII`, and coil).
- **User/CLI/GUI Contract**: Computes per-residue `SSP` scores and normalized 4-state conformational populations $(p_\alpha, p_\beta, p_{\text{PPII}}, p_{\text{coil}})$ from assigned backbone shifts relative to neighbor/T/pH-adjusted random-coil values (`nmrassign propensity`), exported as TSV/JSON and rendered as a stacked population bar track in `nmrassign report` and `nmr-gui`.
- **Quantitative Done Gate**: Populations sum to `1.0 +- 1e-9` at every residue; folded control (`Ubiquitin` `bmr17769`) correctly identifies all 5 $\beta$-strands and $\alpha_1$/$3_{10}$ helices (`>= 90%` secondary-structure state agreement with `1D3Z`), while disordered controls remain coil/PPII-dominated.

### T11.4 — Paramagnetic Metalloprotein `PCS` / `PRE` Bleaching & $\Delta\chi$-Tensor-Aware Assignment
- **Task ID**: `T11.4`
- **Owned Files/Modules**: `crates/nmr-core/src/paramagnetic.rs`, `crates/nmr-core/tests/paramagnetic_tests.rs`, `docs/specs/14-regime-solvers.md` (§3)
- **Clean-Room Literature Citations**: Solomon (*Phys. Rev.* 1955); Bertini, Luchinat & Parigi (*Prog. Nucl. Magn. Reson. Spectrosc.* 2002, 40, 249–273 — paramagnetic shifts & relaxation in metalloproteins); Schmitz, Stanton-Cook, Su, Otting & Huber (*J. Biomol. NMR* 2008 — 5-parameter $\Delta\chi$-tensor linear SVD fitting).
- **User/CLI/GUI Contract**: `nmrassign assign --regime paramagnetic [--metal-ligands CYS42,CYS45,...]` modeling sequence-local / cofactor-coord paramagnetic blind zones ($r_{\text{M}} < r_{\text{bleach}}$ expected-unobservable prior so solver penalties do not force noise roots onto bleached `[2Fe-2S]`/`[4Fe-4S]` coordination loops!) plus optional post-seed iterative 5-parameter $\Delta\chi$ pseudocontact-shift (`PCS`, $\Delta\delta^{\text{PCS}} = \frac{1}{12\pi r^3}[\Delta\chi_{\text{ax}}(3\cos^2\theta - 1) + \frac{3}{2}\Delta\chi_{\text{rh}}\sin^2\theta\cos 2\phi]$) tensor fitting when a metal scaffold is provided.
- **Quantitative Done Gate**: On development `CISD3` (`BMRB 51439`, `[2Fe-2S]` cluster coordinated by `Cys` motifs, sequence-only ligand-motif blind-zone prior), maintains `>= 95%` high-confidence precision with zero false-positive assignments inside bleached metal-binding loops and improves or preserves 5-seed accuracy; synthetic lanthanide `PCS` tensor SVD recovers $(\Delta\chi_{\text{ax}}, \Delta\chi_{\text{rh}})$ within `< 1%` error.

---

## Phase 12: Built-In Interactive `wgpu` 3D Biomolecular Structure Viewer & Dipolar Refinement (`T12.1`–`T12.4`)

### T12.1 — Native & `wasm32` `wgpu` 3D Biomolecular Ribbon, Stick & Ensemble Viewer
- **Task ID**: `T12.1`
- **Owned Files/Modules**: `crates/nmr-gui/src/mol3d_viewer.rs`, `crates/nmr-gui/src/shaders/mol3d.wgsl`, `docs/specs/15-structure-viewer.md` (§1)
- **Clean-Room Literature Citations**: Carson & Bugg (*J. Mol. Graph.* 1986 — cubic B-spline / Hermite peptide plane ribbon extrusion); Kabsch (*Acta Crystallogr. A* 1976 — optimal rotational superposition).
- **User/CLI/GUI Contract**: Built-in `wgpu` 3D viewport dockable alongside 2D/3D NMR spectra in `nmr-gui` (desktop & `wasm32`) rendering smooth secondary-structure cartoon ribbons (helical cylinders/ribbons, arrowed $\beta$-strands, coils), instanced cylinder/sphere sidechains & metal cofactors, multi-model NMR bundle sausage/ensemble opacity, trackball rotation/pan/zoom, orthographic/perspective toggle, and crisp high-DPI offscreen PNG/SVG capture.
- **Quantitative Done Gate**: Sustains `>= 60 fps` on a 20-conformer `150 aa` NMR ensemble (`~48,000` atoms) on WebGPU; compiles cleanly on both native desktop and `wasm32-unknown-unknown` with zero external C/C++ dependencies.

### T12.2 — Linked 3D-to-Strip Picking, Quantitative Putty/Color Mapping & NOE/PRE Cylinders
- **Task ID**: `T12.2`
- **Owned Files/Modules**: `crates/nmr-gui/src/mol3d_overlay.rs`, `docs/specs/15-structure-viewer.md` (§2)
- **Clean-Room Literature Citations**: Original clean-room interactive visual analytics linking 3D structural coordinates with multidimensional NMR frequency strips.
- **User/CLI/GUI Contract**: Clicking a residue/atom in the 3D `wgpu` viewer highlights it in the sequence ribbon, opens its 2D covalent sidechain graph, and centers the 2D HSQC + 3D sequential strip views on its assigned chemical shifts (and vice versa); supports quantitative ribbon radius ("putty") + Perceptually Uniform (`Viridis`/`Cividis`/`Blue-White-Red`) color mapping driven by Assignment Confidence, `CSI`, Lipari–Szabo $S^2$, $R_{\text{ex}}$, or Ligand `CSP` $\Delta\delta_{\text{comb}}$; and renders color-coded 3D restraint cylinders (green satisfied, amber `< 0.5 A` violation, red `>= 0.5 A` violation).
- **Quantitative Done Gate**: Analytical ray-sphere / ray-ribbon picking accuracy `100%` in unit tests; zero frame-time spike (`< 2 ms` update) when switching quantitative color tracks or toggling `1,500` NOE cylinders.

### T12.3 — Post-Assignment Database-Driven 3/9-mer $(\phi, \psi, \chi_1)$ Torsion Restraint Engine
- **Task ID**: `T12.3`
- **Owned Files/Modules**: `crates/nmr-core/src/torsion_db.rs`, `crates/nmr-core/tests/torsion_db_tests.rs`, `docs/specs/04-structure.md` (§2)
- **Clean-Room Literature Citations**: Shen, Vernon, Baker & Bax (*PNAS* 2008); Shen & Bax (*J. Biomol. NMR* 2013, 56, 227–241 — backbone & $\chi_1$ torsion prediction from `(HN, N, CA, CB, C', HA)` secondary shifts + sequence similarity); Lovell et al. (*Proteins* 2000).
- **User/CLI/GUI Contract**: `nmrassign torsions --shifts assigned.str -o torsions.nef` operating **strictly post-assignment** (never invoked during sequence-only backbone assignment!) over an embedded CC0 BMRB/PDB non-homologous tripeptide/9-mer corpus (enforcing `< 35%` global Needleman–Wunsch sequence identity and zero shared 7-mer peptide substrings during evaluation), classifying residues as `Good` (tight consensus Ramachandran cluster), `Dynamic` (predicted $S^2 < 0.65$ via Berjanskii & Wishart *JACS* 2005 `RCI`), or `Ambiguous`, and feeding tighter $(\phi, \psi, \chi_1)$ bounds into `nmr-core::structure`.
- **Quantitative Done Gate**: Leave-homologs-out validation on the 12-protein BMRB suite achieves `>= 88%` `Good` classification coverage with `< 2.5%` Ramachandran basin outliers and mean `(phi, psi)` circular MAE `<= 14 deg` on `Good` residues.

### T12.4 — Residual Dipolar Coupling (`RDC`) Saupe Tensor SVD & GPU Torsion-Angle Refinement
- **Task ID**: `T12.4`
- **Owned Files/Modules**: `crates/nmr-core/src/rdc.rs`, `crates/nmr-gpu/src/rdc_gpu.rs`, `crates/nmr-core/tests/rdc_tests.rs`, `docs/specs/04-structure.md` (§5)
- **Clean-Room Literature Citations**: Tjandra & Bax (*Science* 1997 — dilute liquid-crystal biomolecular RDCs); Cornilescu, Marquardt, Ottiger & Bax (*J. Am. Chem. Soc.* 1998 — RDC quality factor $Q$); Losonczi, Andrec, Fischer & Prestegard (*J. Magn. Reson.* 1999, 138, 334–342 — exact linear 5-component Saupe order matrix SVD); Clore, Gronenborn & Tjandra (*J. Magn. Reson.* 1998).
- **User/CLI/GUI Contract**: `nmrassign structure --rdc rdcs.tsv` supporting multi-medium `1D_NH`, `1D_CAHA`, `1D_CACO`, `1D_NC'` dipolar couplings (`NEF` `nef_rdc_restraint_list` & TSV I/O), analytical Saupe matrix SVD back-calculation ($D_{\text{calc}}$ vs $D_{\text{obs}}$ scatter + Cornilescu $Q$ in HTML report & GUI), and analytical torsion-angle gradient harmonic RDC energy refinement in `nmr-core::structure` & `nmr-gpu`.
- **Quantitative Done Gate**: Saupe SVD recovers synthetic axial/rhombic $(D_a, R)$ and Euler frame to `< 1e-10` on noise-free bond vectors; RDC refinement on Ubiquitin (`1D3Z` public BMRB RDCs) achieves free cross-validated $Q_{\text{free}} < 0.25$ and improves backbone RMSD vs NOE-only folding.

---

## Phase 13: GPU Spin Relaxation, Lipari–Szabo / Palmer Model-Free ($S^2$), `CEST`, `CPMG` & Ligand `CSP` $K_d$ (`T13.1`–`T13.6`)

### T13.1 — Pseudo-3D Relaxation & Exchange Plane Series Integrator
- **Task ID**: `T13.1`
- **Owned Files/Modules**: `crates/nmr-core/src/relax_io.rs`, `crates/nmr-core/tests/relax_io_tests.rs`, `docs/specs/16-dynamics-and-csp.md` (§1)
- **Clean-Room Literature Citations**: Farrow et al. (*Biochemistry* 1994 — $^{15}\text{N}$ $T_1$, $T_2$, and $\{^1\text{H}\}\text{–}^{15}\text{N}$ NOE pulse schemes & uncertainty propagation); Korzhnev et al. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2001).
- **User/CLI/GUI Contract**: Decodes Bruker interleaved/plane-queued pseudo-3D (`ser` + `vdlist`, `vclist`, `fqlist`, `cnst` / pulse length) and Varian arrayed (`d2`, `ncyc`, `tof`, `satfrq`) datasets, processes all 2D planes with identical `ProcessingRecipe` phase/window parameters, locks shared `(1H, 15N/13C)` Voigt/Gaussian peak shapes across planes to avoid low-SNR peak-wander bias, and outputs per-residue intensity decay/profile tables with per-plane MAD + duplicate-delay uncertainties (`TSV` & `NEF` relaxation saveframes).
- **Quantitative Done Gate**: Recovers true simulated plane intensities across `20`-plane synthetic pseudo-3D Bruker & Varian datasets down to `SNR = 4` with `< 1.5%` systematic amplitude bias (vs `> 8%` positive bias from naive local-maximum picking at low SNR).

### T13.2 — Batched GPU $R_1$, $R_2$ & Steady-State $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE Curve Fitter
- **Task ID**: `T13.2`
- **Owned Files/Modules**: `crates/nmr-core/src/relax_fit.rs`, `crates/nmr-gpu/src/relax_gpu.rs`, `crates/nmr-gpu/src/shaders/relax_mc.wgsl`, `crates/nmr-core/tests/relax_fit_tests.rs`
- **Clean-Room Literature Citations**: Kay, Torchia & Bax (*Biochemistry* 1989, 28, 8972–8979 — backbone dynamics & trimmed $R_2 / R_1$ $\tau_m$ estimation); Press et al. (Levenberg–Marquardt nonlinear least squares & parametric Monte Carlo error propagation).
- **User/CLI/GUI Contract**: `nmrassign relax --r1 <series> --r2 <series> --noe <sat,unsat> [--mhz 600,800] [--mc 1000] [--engine gpu|cpu]` fitting 2-parameter ($I_0 e^{-R t}$) and optional 3-parameter exponential decays, saturated/unsaturated steady-state hetNOE ratios with asymmetric error propagation, outlier-trimmed $R_2/R_1$ initial global $\tau_m$ & $D_\parallel / D_\perp$ estimate, and reduced spectral density mapping $J(0), J(\omega_N), J(0.87\omega_H)$ (Farrow et al. *J. Biomol. NMR* 1995).
- **Quantitative Done Gate**: `150` residues $\times$ `1,000` Monte Carlo bootstrap replicates complete in `< 100 ms` on WebGPU (`< 1e-5` CPU/GPU rate parity); recovers exact $R_1, R_2$, hetNOE, and `1 sigma` uncertainties within `+-3%` of analytical Fisher information bounds.

### T13.3 — Lipari–Szabo / Palmer Model-Free (`M1`–`M5`) & Anisotropic Diffusion Engine
- **Task ID**: `T13.3`
- **Owned Files/Modules**: `crates/nmr-core/src/modelfree.rs`, `crates/nmr-core/tests/modelfree_tests.rs`, `docs/specs/16-dynamics-and-csp.md` (§2)
- **Clean-Room Literature Citations**: Lipari & Szabo (*J. Am. Chem. Soc.* 1982, 104, 4546–4559 & 4559–4570 — model-free spectral density $S^2, \tau_e$); Clore, Szabo, Bax, Kay, Driscoll & Gronenborn (*J. Am. Chem. Soc.* 1990, 112, 4989–4991 — two-timescale extended model-free $S_f^2, S_s^2, \tau_s$); Mandel, Akke & Palmer (*J. Mol. Biol.* 1995, 246, 144–163 — `M1`–`M5` statistical model selection & $R_{\text{ex}}$); Woessner (*J. Chem. Phys.* 1962 — axially symmetric/anisotropic rotational diffusion); d'Auvergne & Gooley (*J. Biomol. NMR* 2003 — AIC/BIC model selection).
- **User/CLI/GUI Contract**: `nmrassign modelfree --data relax_600.tsv [--data relax_800.tsv] [--pdb model.pdb] [--criterion aic|bic|f-test]` alternating global diffusion tensor optimization (`isotropic`, `prolate`/`oblate` axial, `anisotropic` asymmetric rotor using NH bond vectors from `--pdb`) with per-residue model-free models `M1` ($S^2$), `M2` ($S^2, \tau_e$), `M3` ($S^2, R_{\text{ex}}$), `M4` ($S^2, \tau_e, R_{\text{ex}}$), and `M5` ($S_f^2, S_s^2, \tau_s$), accounting for $^{15}\text{N}$ CSA (`-172 ppm` default) and $r_{\text{NH}} = 1.02\text{ \AA}$ dipolar relaxation.
- **Quantitative Done Gate**: Recovers exact synthetic `M1`–`M5` parameters ($S^2$ MAE `< 0.005`, $\tau_m$ error `< 0.5%`, $R_{\text{ex}} \propto B_0^2$ quadratic field scaling verified across `600` & `800 MHz`) and achieves `>= 95%` correct `M1`–`M5` AIC model recovery under realistic `2%` Gaussian intensity noise.

### T13.4 — GPU Bloch–McConnell Chemical Exchange Saturation Transfer (`CEST`) & `DEST` Solver
- **Task ID**: `T13.4`
- **Owned Files/Modules**: `crates/nmr-core/src/cest.rs`, `crates/nmr-gpu/src/cest_gpu.rs`, `crates/nmr-gpu/src/shaders/bloch_cest.wgsl`, `crates/nmr-core/tests/cest_tests.rs`, `docs/specs/16-dynamics-and-csp.md` (§3)
- **Clean-Room Literature Citations**: McConnell (*J. Chem. Phys.* 1958); Vallurupalli, Bouvignies & Kay (*J. Am. Chem. Soc.* 2012, 134, 8148–8161 — $^{15}\text{N}$ CEST for invisible excited protein states); Fawzi, Ying, Ghirlando, Torchia & Clore (*Nature* 2011 — Dark-state Exchange Saturation Transfer `DEST`); Higham (*SIAM J. Matrix Anal. Appl.* 2005 — scaling-and-squaring Padé `[6/6]` matrix exponential).
- **User/CLI/GUI Contract**: `nmrassign cest --profiles cest_15hz.tsv,cest_30hz.tsv [--states 2|3] [--global-group all|cluster] [--engine gpu|cpu]` integrating the homogeneous $7 \times 7$ (2-state) or $10 \times 10$ (3-state) Bloch–McConnell propagation matrix $\exp(\mathbf{L}\, T_{\text{EX}})$ across arbitrary RF nutation fields $\omega_1 / 2\pi$ (including $B_1$ inhomogeneity quadrature) and carrier offsets, fitting global $(p_B, k_{\text{ex}})$ and per-residue excited-state chemical shifts $\Delta\varpi_{AB}$ + intrinsic $(R_{1A}, R_{2A}, R_{2B})$ with 1D/2D $\chi^2$ surface profiles.
- **Quantitative Done Gate**: Padé matrix-exponential propagator matches `f64` eigenvalue reference to `< 1e-11` (`< 1e-5` on `f32` WebGPU); recovers minor population $p_B = 1.50\% \pm 0.05\%$, $k_{\text{ex}} = 150 \pm 5\text{ s}^{-1}$, and $\Delta\varpi_{AB}$ to `< 0.05 ppm` across dual-$B_1$ (`15 Hz` & `30 Hz`) synthetic $^{15}\text{N}$ CEST datasets in `< 1.5 s` on WebGPU.

### T13.5 — GPU Relaxation Dispersion (`CPMG` & Off-Resonance $R_{1\rho}$) Solver
- **Task ID**: `T13.5`
- **Owned Files/Modules**: `crates/nmr-core/src/cpmg.rs`, `crates/nmr-gpu/src/cpmg_gpu.rs`, `crates/nmr-gpu/src/shaders/cpmg_fit.wgsl`, `crates/nmr-core/tests/cpmg_tests.rs`
- **Clean-Room Literature Citations**: Carver & Richards (*J. Magn. Reson.* 1972, 6, 89–105); Luz & Meiboom (*J. Chem. Phys.* 1963); Baldwin (*J. Magn. Reson.* 2014, 244, 114–124 — exact closed-form 2-site $R_{2,\text{eff}}(\nu_{\text{CPMG}})$ across all exchange regimes including $\Delta R_2 \ne 0$); Trott & Palmer (*J. Magn. Reson.* 2002 — off-resonance $R_{1\rho}$).
- **User/CLI/GUI Contract**: `nmrassign cpmg --series cpmg_600.tsv,cpmg_800.tsv [--model flat|lm63|cr72|baldwin2014|numerical] [--global] [--engine gpu|cpu]` computing $R_{2,\text{eff}}(\nu_{\text{CPMG}}) = -\frac{1}{T_{\text{relax}}}\ln(I(\nu_{\text{CPMG}}) / I_0)$, testing statistically for chemical exchange against the flat null model (AIC / $F$-test + $\Delta R_{2,\text{eff}} \ge 2.0\text{ s}^{-1}$), and globally fitting $(p_B, k_{\text{ex}})$ + per-residue $|\Delta\omega|$ across multiple static $B_0$ fields (resolving slow-vs-fast exchange $\alpha$-scaling).
- **Quantitative Done Gate**: `Baldwin2014` analytical formula agrees with exact numerical $2 \times 2$ spin-echo matrix propagation to `< 0.02 s^-1` across `k_ex = 50..5000 s^-1` and `p_B = 0.5%..25%`; multi-field (`600` + `800 MHz`) global fit recovers $(p_B, k_{\text{ex}}, |\Delta\omega|)$ within `< 2%` error in `< 0.5 s` on WebGPU.

### T13.6 — Ligand Titration `CSP`, Tight-Binding $K_d$ Isotherms & Interactive Dynamics Studio
- **Task ID**: `T13.6`
- **Owned Files/Modules**: `crates/nmr-core/src/csp.rs`, `crates/nmr-gui/src/dynamics_panel.rs`, `crates/nmr-core/tests/csp_tests.rs`, `docs/specs/16-dynamics-and-csp.md` (§4)
- **Clean-Room Literature Citations**: Williamson (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — using chemical shift perturbation to measure ligand binding affinities & map interfaces); Fielding (*Prog. NMR Spectrosc.* 2007).
- **User/CLI/GUI Contract**: `nmrassign csp --apo hsqc_0.ft2 --titration L1=0.2:hsqc_1.ft2,... --protein-conc-um 100` implementing greedy/Hungarian smooth trajectory tracking of fast-exchange HSQC peaks across ligand equivalents, combined shift perturbation $\Delta\delta_{\text{comb}} = \sqrt{(\Delta\delta_H)^2 + (\alpha_N \Delta\delta_N)^2}$ ($\alpha_N = 0.14$, `Gly` `0.20`), `2 sigma` iterative trimmed threshold detection of binding-site residues, global + per-residue exact quadratic tight-binding $1:1$ isotherm fitting:
  $$\Delta\delta_{\text{obs}} = \Delta\delta_{\max} \frac{([P]_t + [L]_t + K_d) - \sqrt{([P]_t + [L]_t + K_d)^2 - 4[P]_t[L]_t}}{2[P]_t}$$
  with Monte Carlo CIs, plus a unified **`📈 Dynamics & Titration Studio`** tab in `nmr-gui` plotting $R_1/R_2/\text{NOE}/S^2$, `CEST` dips, `CPMG` dispersion curves, and `CSP` binding isotherms with 1-click projection onto the 3D `wgpu` ribbon viewer (`T12.2`).
- **Quantitative Done Gate**: Recovers exact $K_d$ across both weak ($K_d = 500\text{ \mu M} \gg [P]_t$) and tight-binding ($K_d = 5\text{ \mu M} < [P]_t = 100\text{ \mu M}$) synthetic titrations within `< 2%` error (where naive Hill/Michaelis–Menten $[L]_{\text{free}} \approx [L]_t$ fails by `> 8x`); GUI studio compiles cleanly on desktop & `wasm32`.

---

## Phase 14: Automated Visual Assets, Slim README, GitHub-Pages Manual & Live WebGPU Demo (`T14.1`–`T14.4`)

### T14.1 — Deterministic CLI Figure & Vector `SVG`/`PNG` Asset Exporter
- **Task ID**: `T14.1`
- **Owned Files/Modules**: `crates/nmr-cli/src/export_fig.rs`, `crates/nmr-core/src/report.rs` (standalone figure renderers), `docs/assets/`
- **Clean-Room Literature Citations**: Original clean-room vector SVG & antialiased raster rendering engine (`nmr-core::report` & `nmr-core::octet_render`).
- **User/CLI/GUI Contract**: `nmrassign export-fig --kind <hsqc-contours|strip-walk|sidechain-graph|ribbon-3d|relax-s2|cest-dip|csp-kd|octet-scheme> ... -o docs/assets/<name>.svg|.png` generating publication-ready, deterministic (timestamp-free, reproducible byte-for-byte) dark- and light-theme visual assets directly from real Ubiquitin / simulated multi-regime benchmarks.
- **Quantitative Done Gate**: Generates all `8` canonical figure types into `docs/assets/` with valid XML/SVG syntax, crisp text/axes, `< 250 KB` per SVG, and byte-for-byte determinism across repeated runs.

### T14.2 — Slim, Visual-First `README.md` Rewrite
- **Task ID**: `T14.2`
- **Owned Files/Modules**: `README.md`
- **Clean-Room Literature Citations**: `AGENTS.md` clean-room comparison & licensing rules.
- **User/CLI/GUI Contract**: Streamlines `README.md` to `< 180` high-signal lines: one-sentence value proposition, hero workbench & strip-walk showcase (`docs/assets/`), link to zero-install live WebGPU browser demo, 60-second CLI & GUI quickstart (raw Bruker/Varian FID folder -> assignments, 3D ensemble & HTML report), concise capability & verified benchmark table (`v0.3+`), clean-room MIT provenance statement, and structured links to `docs/manual/` and `docs/specs/`.
- **Quantitative Done Gate**: `wc -l README.md <= 180`; all relative image/doc links resolve on disk; zero forbidden product claims.

### T14.3 — Hostable GitHub-Pages Documentation Manual & CI Publisher
- **Task ID**: `T14.3`
- **Owned Files/Modules**: `docs/manual/`, `.github/workflows/docs.yml`
- **Clean-Room Literature Citations**: All peer-reviewed papers across `docs/specs/01`–`16` consolidated in a master `Bibliography & Clean-Room Provenance` chapter.
- **User/CLI/GUI Contract**: Builds a searchable static documentation manual covering: (1) *Getting Started & Installation*, (2) *End-to-End Tutorial: Raw FIDs & NUS to Backbone/Sidechain Assignment*, (3) *Interactive Workbench & Two-Key Shortcuts Reference*, (4) *3D NOESY Folding, Post-Assignment Torsions & RDCs*, (5) *IDPs (`13C`-Detected), Methyl-TROSY & Paramagnetic Proteins*, (6) *Spin Relaxation, Palmer Model-Free ($S^2$), CEST, CPMG & Ligand CSP $K_d$*, (7) *Octet Chemical Draw & 1D NMR Studio*, (8) *CLI, REST Service & File Formats (`NMR-STAR`, `NEF v1.1`, `ProcessingRecipe`)*, and (9) *Swarm Roadmap & External Review Protocol*; automatically built and deployed via `.github/workflows/docs.yml`.
- **Quantitative Done Gate**: Static manual build exits `0` with zero broken internal links; every CLI subcommand in `nmrassign --help` is documented with runnable examples.

### T14.4 — Zero-Install Live In-Browser `wasm32` WebGPU Demo Bundle
- **Task ID**: `T14.4`
- **Owned Files/Modules**: `crates/nmr-gui/index.html`, `crates/nmr-gui/src/demo_bundle.rs`, `docs/demo/`
- **Clean-Room Literature Citations**: Public CC0 BMRB / wwPDB reference entries (`bmr17769` / `1D3Z` Ubiquitin).
- **User/CLI/GUI Contract**: Compiles `nmr-gui` to `wasm32-unknown-unknown` (`trunk` / `wasm-bindgen`) embedded into the GitHub Pages site (`docs/demo/`) with a 1-click **"Load Sample Ubiquitin Workspace"** button preloading compact quantized 2D `15N-HSQC`, 3D `HNCACB`/`CBCACONH` strip banks, sequence + live GPU solver, `1D3Z` 3D ensemble ribbon, sample $R_1/R_2/S^2$ & `CSP` curves, and `Octet-Draw` templates — running 100% client-side on the visitor's WebGPU adapter with graceful WebGL2/CPU fallback banner when WebGPU is unavailable.
- **Quantitative Done Gate**: `wasm32-unknown-unknown` release bundle + compressed sample assets total `< 15 MiB`, loads interactively in `< 2.0 s` locally, and runs live GPU contour rendering + simulated annealing in browser without network calls.
