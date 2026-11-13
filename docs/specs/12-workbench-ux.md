# Spec 12 — Interactive NMR Workbench UX, Auto-Classifier & Session Persistence

**Provenance**: Clean-room original specification grounded in published multidimensional NMR spectroscopy, data-interchange standards, and standard human-computer interaction principles:
- Kay, L. E., Ikura, M., Tschudin, R., & Bax, A. (*J. Magn. Reson.* 1990, 89, 496–514) — triple-resonance `HNCO` / `HNCA` / `HN(CO)CA` magnetization transfer and axis physics.
- Grzesiek, S., & Bax, A. (*J. Am. Chem. Soc.* 1992, 114, 6291–6293) — `CBCA(CO)NH` and `HNCACB` sequential `Cα`/`Cβ` strip matching.
- Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E., & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125, 10120–10121) — `¹H–¹³C` methyl-TROSY `HMQC` spectroscopy of high-molecular-weight proteins.
- Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C., & Pierattelli, R. (*J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem. Soc.* 2009, 131, 15339–15345) — protonless `¹³C`-detected `CON`, `CACO`, `CANCO`, and `CBCACO` experiments for intrinsically disordered and paramagnetic proteins.
- Wishart, D. S., Bigam, C. G., Yao, J., Abildgaard, F., Dyson, H. J., Oldfield, E., Markley, J. L., & Sykes, B. D. (*J. Biomol. NMR* 1995, 6, 135–140) — biomolecular `¹H`, `¹³C`, and `¹⁵N` chemical shift referencing.
- Delaglio, F., Grzesiek, S., Vuister, G. W., Zhu, G., Pfeifer, J., & Bax, A. (*J. Biomol. NMR* 1995, 6, 277–293) — NMRPipe multidimensional spectral header convention.
- Gutmanas, A., et al. (*Nat. Struct. Mol. Biol.* 2015, 22, 433–434) — NMR Exchange Format (`NEF v1.1`) STAR saveframe and vendor-extensibility specification.

No source code, keytables, parameter libraries, or binaries from restricted academic or commercial NMR packages were consulted.

---

## 1. Spectrometer Folder & Archive Auto-Classifier (`T09.1` & `T09.2`)

### 1.1 Design Goals
Experimentalists organize protein NMR projects as nested directories or `.tar` / `.tar.gz` / `.tgz` archives containing raw spectrometer acquisitions alongside optional processed planes, polymer sequences (`.fasta`), and shift/restraint tables (`.str`, `.nef`). Experiment folder names are frequently numeric (`1/`, `2/`, `10/`, `10003/`) or abbreviated (`b8/NHSQC`, `v6/CBCAcoNH`, `030804.3g51030/11`), making filename-only heuristics brittle.

`nmr_core::classify` provides a single zero-dependency classification engine shared across native CLI (`nmrassign inspect`), HTTP service (`POST /api/v1/inspect`), native desktop GUI (`nmr-gui`), and browser WebAssembly (`wasm32-unknown-unknown` drag-and-drop):

- `classify_nmr_workspace(path: &Path) -> Result<WorkspaceInventory, String>` (native filesystem directory, single spectrum/table file, or `.tar` / `.tar.gz` / `.tgz` archive).
- `classify_nmr_entries(entries: &[VirtualEntry]) -> WorkspaceInventory` (`wasm32`-compatible slice of `(relative_path, header_bytes, total_bytes)`).

### 1.2 Header-Only Streaming Budget (`< 50 ms` on Unpacked Folders & Seekable `.tar`)
Bulk binary time-domain (`ser`, `fid`, `origfid`, `fidc.dat`) and frequency-domain (`2rr`, `2ri`, `2ir`, `2ii`, `3rrr`, `.ucsf` tiles, `.ft2`/`.ft3` float planes) payloads can exceed hundreds of megabytes per protein bundle and **must never be read into memory during workspace classification**:

1. **Filesystem Scanner**: Checks filenames case-insensitively before opening. Signal payload files (`ser`, `fid`, `2rr`, `3rrr`, etc.) are queried solely via `fs::metadata(path).len()` to determine `DataState` (`RawFid`, `Processed`, `Both`) and byte footprint. Binary spectrum headers read only fixed prefixes (`1024 B` for UCSF v2 `180 + 128·N` headers; `2048 B` for NMRPipe 512-float headers). Parameter/schedule files (`acqus`, `acqu2s`, `acqu3s`, `acqu4s`, `procs`, `proc2s`, `proc3s`, `pulseprogram`, `pdata/1/title`, `nuslist`, `vdlist`, `vclist`, `fqlist`, Varian `procpar`, `text`, `sampling.sch`) read bounded prefixes (`<= 256 KiB`). Inside an already-recognized Bruker or Varian experiment directory, child `ft/` and `pdata/` plane stacks update `DataState::Both` without spawning duplicate per-plane experiment entries. Sibling multi-hundred-MB `.tar.gz`/`.zip` download caches inside a directory that already contains unpacked experiments are skipped rather than decompressed twice. Unpacked directories complete in `< 50 ms`.
2. **POSIX `ustar` & RFC 1952/1951 `.tar.gz` Streamer**: Validates 512-byte POSIX/GNU `ustar` header checksums (supporting GNU `typeflag = 'L'` long names, PAX `typeflag = 'x'` extended paths/sizes, and strict `ustar\0` prefix checks so GNU `ustar  \0` `atime`/`ctime` fields are never misread). On seekable uncompressed `.tar` streams, signal payload blocks are skipped in `O(1)` via `Seek::seek(SeekFrom::Current(padded_size))`, so unpacked folders and seekable `.tar` archives complete in `< 50 ms` (`N4`). On compressed `.tar.gz` / `.tgz` streams, an internal pure-Rust RFC 1952 GZIP + RFC 1951 DEFLATE decoder (with IEEE 802.3 CRC-32 + `ISIZE` trailer verification and LZ77 history bounds) inflates sequentially while discarding bulk `ser`/`fid` signal payload bytes in fixed scratch chunks without heap growth (taking time proportional to compressed archive size rather than `< 50 ms`). After verifying a gzip member's 8-byte `CRC32 + ISIZE` trailer, trailing `0x00` zero-padding bytes at the end of block-padded `.tgz` archives (accepted by `gzip -t`) terminate cleanly with `Ok(())` (`N2`), while genuine concatenated multi-member gzip streams continue when a non-zero byte follows. Trusted local/desktop/`wasm32` drag-and-drop (`extract_archive_virtual_entries`) allows up to `16 GiB` of streamed/discarded payload inflation while capping *retained* `VirtualEntry.header_bytes` RAM at `64 MiB` (`#162`), whereas untrusted HTTP `POST /api/v1/inspect` (`extract_archive_virtual_entries_bounded`) enforces strict `128 MiB` inflated and `32 MiB` retained-header limits against decompression bombs.

### 1.3 Per-Experiment Physics & Metadata Extraction
Each discovered experiment produces an `ExperimentRecord` containing:
- **`vendor`**: `Bruker`, `Varian`, `NmrPipe`, `SparkyUcsf`, `SequenceFasta`, `StarOrNef`.
- **`dimensionality`**: `1D`, `2D`, `3D`, `Pseudo3D`, `4D`.
  - `Pseudo3D` is detected when an acquisition steps a relaxation delay, CPMG counter, saturation frequency, or titration array rather than an evolved chemical-shift axis: Bruker `vdlist` / `vclist` / `vp` or `acqu3s` `FnMODE == 1 (QF)` with relaxation/exchange pulse programs (`t1`, `t2`, `r1`, `r2`, `cpmg`, `cest`, `noe`), or Varian `array` stepping non-phase variables (`d2`, `relaxT`, `ncyc`, `time_T2`, `satfrq`). Note that 3D triple-resonance sequences using a 2-entry `fq1list` for selective `¹³Cα`/`¹³C'` pulses retain hypercomplex `acqu3s` (`States`/`States-TPPI`/`Echo-Antiecho`) and are classified as true `3D`.
- **`data_state`**: `RawFid`, `Processed`, or `Both`.
- **`spectrometer_mhz`**: Proton (`¹H`) Larmor base frequency (`SFO1`/`BF1` on the `¹H` channel, Varian `sfrq`/`h1freq`, or gyromagnetic ratio conversion `|γ_H / γ_X| · f_X` for `¹³C`-detected experiments where `SFO1` is `¹³C` at `~150–226 MHz`).
- **`axes`**: Per-axis `(label, nucleus, points, sw_ppm, carrier_ppm, obs_mhz, quadrature)` with quadrature normalized across Bruker `FnMODE` and Varian `phase`/`fcoef` (`States`, `States-TPPI`, `Echo-Antiecho`, `QF`, `Dqd`, `RealProcessed`).
- **`nus`**: Non-uniform sampling badge and sampling density (`sampled_points / full_grid_points`). For Bruker TopSpin 3/4 (which writes template `NusTD`/`NusAMOUNT` keys even in uniform `FnTYPE = 0` headers), NUS is flagged strictly when `FnTYPE == 2` or a non-empty `nuslist` schedule is present; for Varian, when `sparse == "y"` or `sampling.sch` is present.
- **`pulse_class`** (`PulseSequenceClass`):
  - Heteronuclear 2D roots: `N15Hsqc` (including `¹⁵N`-TROSY / Fast-HSQC / gradient `hsgp` with `(¹H, ¹⁵N)` axes), `C13Hsqc` (constant-time or sensitivity-enhanced `(¹H, ¹³C)`), `C13Hmqc` (methyl-TROSY `¹H–¹³C` `HMQC` with methyl carrier/sweep or `hmqc` pulse program).
  - `¹H`-detected 3D backbone triple-resonance: `Hnco`, `Hncaco`, `Hnca`, `Hncoca`, `Hncacb`, `Cbcaconh`.
  - Aliphatic sidechain TOCSY & NOESY: `Ccconh` (`CC(CO)NH`, Bruker `ccconh`/`cconh`, Varian `gc_co_nh`), `Hccconh` (`H(CCO)NH`, Bruker `hccconh`/`hcconh`, Varian `ghc_co_nh`), `HcchTocsy`, `N15Noesy`, `C13Noesy`.
  - Protonless `¹³C`-detected IDP / paramagnetic suite: `Con2D` (`¹³C'` direct, `¹⁵N` indirect), `Caco2D` (`¹³C'` direct, `¹³Cα` indirect), `Canco3D` (`3D CANCO`), `Cbcaco3D` (`3D CBCACO` / `CBCANCO`).
  - Relaxation, exchange & titration: `RelaxR1`, `RelaxR2`, `HetNoe`, `Cpmg`, `Cest`, `TitrationSeries`, and `Unknown` (e.g., `1D` `pulsecal` / `zgpr`, excluded from backbone ingest by default).

### 1.4 Workspace `ProblemRegime` & Missing-Input Hints
From the ensemble of `ExperimentRecord` entries, `classify_nmr_entries` infers a primary `ProblemRegime` (plus `secondary_regimes`) and actionable `missing_input_hints`:

| `ProblemRegime` | Trigger Condition |
| :--- | :--- |
| `Paramagnetic` | Explicit paramagnetic/oxidized+reduced pair/PCS/PRE metadata or ultra-wide `¹H` sweep (`> 28 ppm`) |
| `Idp13cDetected` | Majority or primary backbone suite consists of `¹³C`-detected `Con2D` / `Caco2D` / `Canco3D` / `Cbcaco3D` without `¹H`-detected 3D triple-resonance |
| `MethylTrosy` | `C13Hmqc` methyl-TROSY spectra dominate without `¹⁵N` amide triple-resonance |
| `TitrationCsp` | `>= 4` same-nucleus 2D `HSQC`/`HMQC` series points with titration/concentration/ratio labels (or `TitrationSeries`) without 3D assignment suite |
| `RelaxationDynamics` | Workspace contains `RelaxR1` / `RelaxR2` / `HetNoe` / `Cpmg` / `Cest` dynamics experiments as primary payload |
| `SidechainNoesy` | Full backbone suite supplemented by aliphatic sidechain (`Ccconh`, `Hccconh`, `HcchTocsy`, `C13Hsqc`) and/or `N15Noesy` / `C13Noesy` |
| `GlobularBackbone` | Standard `¹H, ¹⁵N, ¹³C` backbone assignment bundle (`N15Hsqc` + 3D triple-resonance) |

Missing-input hints notify the user when:
- No polymer sequence (`.fasta`, `.str`, or `.nef`) was found in the workspace;
- 3D triple-resonance experiments exist without a `2D ¹⁵N-HSQC` root spectrum;
- Only `RawFid` time-domain data is present (prompting Stage 2 batch FID processing);
- Only `Cα`/`C'` triple-resonance experiments (`HNCA`/`HNCOCA`/`HNCO`) are present without `Cβ` (`HNCACB`/`CBCACONH`), noting reduced amino-acid type discrimination (`CA`-only sparse regime).

### 1.5 CLI (`nmrassign inspect`) & HTTP API (`POST /api/v1/inspect`)
- **CLI**: `nmrassign inspect <path> [--json]` prints either a human-readable experiment matrix + regime + hints table or pretty-printed `WorkspaceInventory` JSON (`--json`).
- **HTTP Service**: `POST /api/v1/inspect` validates the HTTP `Host` header against loopback (`localhost`, `127.0.0.1`, or `[::1]` on the bound listener port) or an explicit `--cors-origin` entry before accepting same-origin `Origin == http(s)://{Host}` (`403 Forbidden` otherwise, blocking DNS-rebinding attacks against `127.0.0.1`; `N3`) and enforces strict `Content-Type` validation (`415` on `text/plain`), accepting either:
  1. `Content-Type: application/json` with `{ "entries": [VirtualEntry, ...] }` (or `{ "path": "<dir_or_archive>" }` **only** when `--inspect-root <dir>` is configured on `nmrassign serve` and the canonicalized target lies inside `--inspect-root`; disabled `403` by default), or
  2. Raw `.tar` / `.tar.gz` / `.tgz` binary request body (`application/x-tar`, `application/gzip`, or `application/octet-stream`, bounded at `128 MiB` inflated and `32 MiB` retained header bytes, with trailing zero-padding tolerance on `.tgz`),
  returning `200 OK` with serialized `WorkspaceInventory` JSON.
- **Ingest Modal (`T09.2`)**: Renders an editable experiment selection matrix (checkbox include/exclude, dropdown override of `PulseSequenceClass`, NUS/quadrature/state badges, and sequence paste/upload box) with a single **Process & Load Selected** primary action.

---

## 2. Guided 6-Stage Workflow Stepper, Top Menu Bar & `Cmd/Ctrl+K` Command Palette (`T09.3`)

### 2.1 Top Menu Bar
The workbench replaces unstructured sidebars with a standard 9-menu horizontal bar emitting typed `WorkbenchAction` values (`>= 55` commands):
1. **`File`**: Open Workspace Folder/Archive, Import Spectra/Peaks/Sequence/STAR, Open/Save `.nef` Session, Export NMR-STAR / Peaks TSV / HTML Quality Report.
2. **`Process`**: Process All Raw FIDs, Re-process Active Spectrum, Auto-Phase (`ACME`), Toggle Zero-Filling / Apodization / Linear Prediction / IST-FISTA NUS, Edit Processing Recipe.
3. **`Spectrum`**: Contour Threshold (`vt`), Raise/Lower Contours (`ct`/`cb`), Toggle Negative Contours, Zoom Full (`oz`), Zoom In/Out (`zi`/`zo`), Manage Multi-Spectrum Overlays, Nudge Reference Offsets.
4. **`Peaks`**: Pick 2D Root Peaks, Corroborated 3D Strip Pick, Neural 3D CNN Pick, Add Peak at Crosshair (`pa`), Delete Selected Peaks (`pd`), Parabolic Sub-Voxel Snap (`pc`), Toggle Peak Labels (`pl`).
5. **`Assignment`**: Run `evo-graph` Solver, Run Multi-Replica GPU Annealing, Open `i-1 / i / i+1` Match-Finder Strips (`st`), Assign/Lock Active Root (`at`), Unlock All Locks, Inspect Dataset QC Gate.
6. **`Structure`**: Predict `φ/ψ` Dihedrals (`CSI`), Run Iterative NOESY + Torsion Folding, Export PDB Ensemble.
7. **`Dynamics`**: Fit `R₁` / `R₂` / hetNOE Relaxation Series, Model-Free `S²` Summary, Titration CSP Plot.
8. **`Chemistry`**: Open Octet 2D Chemical Studio, Peptide Builder, SMILES/IUPAC Import, Analytical `1D` NMR Predictor.
9. **`Help`**: Keyboard Accelerator Cheat-Sheet (`?`), Command Palette (`Cmd/Ctrl+K`), Load Ubiquitin Demo, About `nmrassign`.

### 2.2 Persistent 6-Stage Workflow Stepper & QC Gate Badge
Directly beneath the menu bar sits a compact 6-stage horizontal stepper computed deterministically from `WorkspaceReadiness`:
1. **`1. Ingest`** — Sequence + raw/processed experiment inventory selected.
2. **`2. Process FIDs`** — Raw time-domain FIDs transformed to phased, referenced `2D`/`3D` spectra (auto-completed if inputs are already processed).
3. **`3. Pick Peaks`** — `2D` root peaks and corroborated `3D` strip peaks picked and reviewed.
4. **`4. Assign Spins`** — Spin systems clustered, `evo-graph` / `standard` solver executed, low-confidence strips curated via match-finder.
5. **`5. Fold & Dynamics`** — Optional sidechain assignment, torsion/NOESY folding, or relaxation/CSP analysis.
6. **`6. Export & Report`** — Validated NMR-STAR v3, lossless `.nef` session, and HTML quality report exported.

Adjacent to the stepper, the calibrated `DatasetQcVerdict` pill (`PASS` emerald, `WARN` amber, `ABSTAIN` crimson) is permanently visible alongside a single contextual primary **"What Next?"** button that triggers the highest-priority uncompleted action.

### 2.3 `Cmd/Ctrl+K` Fuzzy Command Palette
Pressing `Cmd+K` (macOS) or `Ctrl+K` (Linux/Windows/Web) opens a centered modal search bar scoring all registered `WorkbenchAction` descriptors by word-boundary prefix, acronym/camelCase match, contiguous substring bonus, and subsequence gap penalty (`< 1 ms` per keystroke). Each row displays category badge, action title, subtitle, standard modifier shortcut (`Cmd/Ctrl+Z`, `Cmd/Ctrl+S`), and two-key NMR chord badge (`pa`, `st`, `ct`).

---

## 3. Modal Two-Key NMR Chord Accelerators (`T09.4`)

### 3.1 Clean-Room Finite-State Chord Dispatcher
Interactive spectrum curation relies on rapid single-handed two-letter mnemonics without modifier fatigue. `ChordDispatcher` implements a deterministic two-state (`Idle` -> `PendingFirstKey(char, Instant)`) machine with a configurable `450 ms` timeout:

| Input | Action | Description |
| :--- | :--- | :--- |
| `v` `t` | `OpenContourThresholdModal` | Open interactive contour threshold / multiplier popover |
| `p` `a` | `AddPeakAtCrosshair` | Insert new peak at current 2D/3D crosshair ppm coordinates |
| `p` `d` | `DeleteSelectedPeaks` | Delete currently selected peak(s) (undoable) |
| `p` `c` | `SnapSelectedPeaksToCenter` | 3-point parabolic sub-voxel maximum/extremum interpolation |
| `p` `l` | `TogglePeakLabels` | Toggle residue/spin-system peak annotations on canvas |
| `o` `z` | `ZoomFullSweep` | Reset viewport to full spectral sweep width |
| `z` `i` / `z` `o` | `ZoomIn` / `ZoomOut` | Symmetric viewport zoom around crosshair (`0.75x` / `1.33x`) |
| `c` `t` / `c` `b` | `RaiseContourFloor` / `LowerContourFloor` | Multiply / divide active base contour floor by `1.25x` |
| `s` `t` | `TileMatchFinderStrips` | Populate `i-1 / i / i+1` candidate strips for active root |
| `a` `t` | `OpenAssignmentPopover` | Assign or lock active spin system / peak to sequence residue |
| `Space` | `ToggleCrosshairLock` | Lock/unlock synchronized crosshair across 2D & 3D views |
| `?` | `ToggleShortcutCheatSheet` | Open searchable keyboard shortcut reference window |

### 3.2 Focus Suppression & Modifier Passthrough
- **Focus Safety**: Whenever an `egui` text-input widget (`ctx.wants_keyboard_input()`) or the Octet 2D chemical drawing canvas has keyboard focus, single/two-key NMR chords are completely suppressed so typing sequence text, SMILES strings, or atom labels (`C`, `N`, `O`, `P`, `S`) never triggers spectrum mutations.
- **Modifier Passthrough**: Events carrying `Cmd`, `Ctrl`, or `Alt` bypass the chord buffer immediately so system shortcuts (`Cmd/Ctrl+Z`, `Cmd/Ctrl+Shift+Z`, `Cmd/Ctrl+K`, `Cmd/Ctrl+S`) fire with zero latency.
- **Pending Feedback & Cancel**: While `PendingFirstKey(k)` is active (`< 450 ms`), the status bar displays `Chord: <k>_`; pressing `Esc` or letting `450 ms` elapse resets cleanly to `Idle`.

---

## 4. Multi-Spectrum GPU Overlays & `i-1 / i / i+1` Candidate Match-Finder Strips (`T09.5`)

### 4.1 Alpha-Blended Multi-Spectrum WebGPU Contour Compositor
`ContourPipelineResource` (`wgpu` + WGSL, native + `wasm32`) renders an ordered stack of visible `OverlayLayerSpec` layers into the 2D spectrum viewport using `wgpu::BlendState::ALPHA_BLENDING`:
- Each layer binds its own `f32` storage buffer (cached by `(layer_id, spec_gen)` so panning, zooming, and threshold adjustments allocate zero GPU buffers per frame), `base_level`, `level_factor`, `num_levels`, positive/negative RGBA colors (`pos_rgba`, `neg_rgba`), and `ppm_nudge: [f32; 2]` visual alignment shift.
- A shared `SyncCrosshair { h_ppm, n_ppm, c_ppm, locked }` projects cursor coordinates seamlessly across overlaid `¹⁵N-HSQC`, `¹³C-HSQC`, titration stacks, and orthogonal `3D` strip slices at `>= 60 fps`.

### 4.2 Sequential `i-1 / i / i+1` Candidate Match-Finder Strips
Given clustered `SpinSystem` records and pairwise sequential compatibility scores `P(p, s)` from `nmr_core::solver::pair_matrix` (`docs/specs/03-assignment.md` §3):
- Selecting a root spin system `s` ranks the top-$K$ (default `K = 5`) **predecessor (`i-1`)** candidates `p` descending by `P(p, s)` and **successor (`i+1`)** candidates `q` descending by `P(s, q)`.
- Each candidate column renders volumetric `3D` `(¹H, ¹³C)` strip contours (when `Spectrum3D` volumes are loaded) or Gaussian-broadened peak sticks, overlaid with horizontal dashed alignment guides across all columns:
  - **`Cα`** guide (cyan/blue),
  - **`Cβ`** guide (amber/red-orange),
  - **`C'`** guide (emerald green),
  along with per-nucleus shift residuals (`ΔCα`, `ΔCβ`, `ΔC'` in ppm) and 1-click **`Confirm Link`** / **`Lock Residue`** buttons.

---

## 5. Bounded Undo/Redo Command Stack & Lossless NEF v1.1 Session Persistence (`T09.6`)

### 5.1 Reversible `WorkbenchCommand` Ring Buffer
All interactive mutations in `nmr_core::session::WorkbenchSessionState` execute through `CommandHistory`, a bounded `256`-entry `VecDeque` supporting `Cmd/Ctrl+Z` (`undo`) and `Cmd/Ctrl+Shift+Z` (`redo`):
1. `AddPeak(Peak)`
2. `DeletePeaks(Vec<usize>)` (retaining exact original indices and `Peak` structs in the undo record)
3. `MovePeak { index, new_h, new_n, new_c }`
4. `RecenterPeak { index, fitted_h, fitted_n, fitted_c, new_height }`
5. `AssignSpinSystem { spin_system, residue_seq_idx }` (capturing any displaced assignment on both spin system and target residue so `undo` restores bijective state exactly)
6. `LockResidue { spin_system, residue_seq_idx, locked }`
7. `UnassignResidue { residue_seq_idx }`
8. `SetContourParams { layer_id, params }`
9. `SetReferenceOffset { offsets: [f64; 5] }`

Every variant satisfies the algebraic round-trip laws `undo(execute(S, cmd)) == S` and `redo(undo(execute(S, cmd))) == execute(S, cmd)`. Pushing a new command truncates any pending redo tail; exceeding `256` entries evicts the oldest undo record in `O(1)`.

### 5.2 Lossless Single-File `.nef` Session Format
Per the `NEF v1.1` vendor-extension specification (Gutmanas et al. 2015), `write_session_nef_str` and `read_session_nef_str` persist a complete workbench session in a single standards-compliant `.nef` file:
- Standard `NEF v1.1` saveframes (`nef_nmr_meta_data`, `nef_molecular_system`, `nef_chemical_shift_list`, and `nef_nmr_spectrum` peak tables) remain 100% readable by any `NEF v1.1`-compliant tool.
- A vendor-prefixed `save_nmrassign_workbench` saveframe (`_nmrassign_workbench.sf_category = nmrassign_workbench`) losslessly round-trips solver confidences, user-locked `(spin_system, residue_seq_idx)` pairs (`_nmrassign_lock`), confirmed sequential links (`_nmrassign_link`), per-layer contour thresholds/colors/visibility (`_nmrassign_contour_layer`), global reference offsets (`_nmrassign_workbench.ref_offset_*`), dataset QC verdict, and per-spectrum `ProcessingRecipe` JSON (`_nmrassign_processing_recipe`).
