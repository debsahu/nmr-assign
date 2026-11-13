# TODO

Target: an open-source, clean-room GPU biomolecular NMR platform (raw FIDs or
processed spectra in; picked peak lists, resonance assignments, 3D structures,
spin relaxation / chemical exchange parameters, and an interactive desktop/browser
WebGPU workbench out), plus an integrated 2D Chemical Structure, Reaction & 1D NMR
Studio (`Octet-Draw`).
Architecture & module map: `docs/specs/00-architecture.md`.
Atomic swarm roadmap (`T08`–`T14`): `docs/roadmap/MASTER_ROADMAP.md`.
External review workflow (`Claude` review bundles): `docs/roadmap/REVIEW_PROTOCOL.md` (`scripts/review_pack.sh`).

Status key: `[x]` done, `[ ]` open. Owner: `(M)` maintainer, `(A)` AI agents.
Specs: `docs/specs/`. Agent rules: `AGENTS.md`.

## Phase 1: Backbone assignment on real data
- [x] (A) NMR-STAR reader/writer, peak TSV format (`crates/nmr-core`)
- [x] (A) Spin system builder with competing candidates (`crates/nmr-core`)
- [x] (A) Scoring (typing LLR + connectivity) and simulated annealing (`crates/nmr-core`)
- [x] (A) Simulator, benchmark command, regression tests (`crates/nmr-core`, `crates/nmr-cli`)
- [ ] (M) Run benchmark on the completed assignment (`data/private/`)
- [ ] (M) Review and set PROVISIONAL constants in `specs/03` sections 3 and 4
- [x] (A) Rebuild priors from 36 public BMRB entries (`nmrassign priors`, `data/priors/bmrb_priors.json`, `data/priors/PROVENANCE.md`)
- [x] (A) Assemble public benchmark set: 12 BMRB entries across alpha, beta, alpha/beta & IDP folds (`data/benchmark/manifest.tsv`, `nmrassign benchmark-suite`)
- [ ] (M) Decide: secondary-structure-dependent priors?
- [x] (A) Intensity columns (`height`, `volume`) in peak TSV; intensity-aware `i` / `i-1` candidate disambiguation
- [x] (A) Re-test degenerate `i` / `i-1` handling once intensities exist
- [x] (A) Factor-graph belief propagation + evolutionary solver (`--solver evo-graph`, `nmr-core::evo_graph`, `specs/10`) with Pass-2 belief-damped fragment confidence calibration
- [x] (A) Automatic unlabelled global chemical-shift reference-offset estimation (`nmr-core::priors`, WebGPU `U`-matrix parity, permutation-invariant `nmrassign score`)
- [x] (A) Realistic Bruker 5-seed physics simulator CI gate (`--realistic --seeds 5`) and development real-spectrum benchmarks (Ubiquitin `pdata`/`.ucsf` and `BMRB 51439` `CISD3`)
- [x] (A) Deterministic covalent `(seq[k-1], seq[k], seq[k+1])` primary-sequence shift corrections (pre-Proline `CA`/`C'`/`CB` and `i-1` neighbor `15N`, Wishart 1995 / Schwarzinger 2001) (`specs/03`)
- [x] (A) Solver curation follow-ups (`T09.7`): weak-peak intensity penalty for minor conformers (`Gln2` 9% minor second-form peak), sparse-data `standard` vs `evo-graph` diagnosis when `HNCACB` is absent, and out-of-sample confidence calibration verification

## Phase 2: Rust + WebGPU performance & interactive GUI (`specs/06`)
- [x] (A) Port full reference pipeline to pure Rust (`crates/nmr-core`) with `< 1e-13` `U`/`P` parity
- [x] (A) WebGPU (`wgpu`/WGSL) scoring matrices and multi-replica GPU annealer (`--engine gpu`)
- [x] (A) Interactive `eframe` + `wgpu` GUI workbench: GPU 2D HSQC contour shader, sequence ribbon, 3D sequential walk strips (`crates/nmr-gui`)
- [x] (A) CI running CPU, WebGPU, clippy, and WASM checks on the benchmark suite (`.github/workflows/ci.yml`)
- [x] (A) Browser WASM + WebGPU build target for `crates/nmr-gui` (`wasm32-unknown-unknown`, `crates/nmr-gui/index.html`)

## Phase 3: Peak picking (`specs/01`, `specs/02`)
- [ ] (M) Identify processed spectra behind the completed assignment
- [x] (A) Spectrum reader (Bruker `pdata` `2rr`/`3rrr`, NMRPipe `.ft2`/`.ft3`) with ppm axes
- [x] (A) Sparky `.ucsf` 2D/3D spectrum reader, real-spectrum picker hygiene (local indirect-axis MAD, sinc-sidelobe & `NH2` doublet rejection, 180° phase-inversion recovery), and field-independent Hz-scaled `3J_HNHa` doublet coalescing (`specs/01`, `specs/02`)
- [x] (A) Classical picker baseline: robust MAD noise, local extrema, sub-voxel log-Gaussian fit, HSQC strip restriction
- [x] (A) Synthetic 2D/3D spectrum renderer in simulator (`render_hsqc_2d`, `render_triple_3d`)
- [x] (A) Neural picker (2D & 3D separable residual CNN in `nmr-core::neural_picker` + WebGPU WGSL compute `nmr-gpu::neural_gpu`) trained on synthetic Voigt/ridge/baseline spectra (`nmrassign pick --method neural --engine gpu`)
- [x] (A) End-to-end benchmark on volumetric spectra: `spectra -> peaks -> assignment` (`nmrassign e2e-benchmark`)
- [x] (A) Two-pass experiment-guided corroborated weak-peak rescue in raw 3D strips (`T08.5`, `nmr-core::picker::StripBank3D` / `corroborated_re_pick_3d`, `specs/02`)

## Phase 4: Sidechain assignment (`specs/07`)
- [x] (A) Residue topology tables for all 20 standard amino acids from IUPAC / PDB CCD (`nmr-core::topology`) + empirical BMRB sidechain shift priors
- [x] (A) Expected peaks & volumetric simulation for `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, and aromatic experiments (`nmr-core::sidechain`)
- [x] (A) Joint backbone + sidechain optimization (`solve_joint_backbone_sidechain`, `nmrassign sidechain`, pre-proline & prochiral handling)
- [x] (A) Peak picking for 13C-edited and 3D TOCSY spectra (`pick_c13_hsqc`, `pick_hcch_tocsy_3d`, `pick_ccconh_3d`, `pick_hccconh_3d`)

## Phase 5: Structure (`specs/04`)
- [x] (A) Shift-based backbone `(phi, psi)` dihedral restraint predictor (`nmr-core::dihedrals`)
- [x] (A) 3D `15N`-/`13C`-NOESY peak picking, $r^{-6}$ calibration, and volumetric simulator (`nmr-core::noesy`)
- [x] (A) Ambiguous NOE distance restraints ($d_{\text{eff}} = (\sum r^{-6})^{-1/6}$), network anchoring, and 7-cycle iterative ADR assignment (`nmr-core::noesy`, `nmr-core::structure`)
- [x] (A) Torsion-angle NeRF + analytical-gradient simulated annealing & multi-conformer PDB ensemble folding (`nmr-core::structure`, `nmrassign structure`)
- [x] (A) Kabsch superposition RMSD, Ramachandran basin, and restraint-violation validation against deposited PDB structures (`data/pdb_refs/`)

## Phase 6: Service (`specs/08`)
- [x] (A) One-command local pipeline: `spectra/peaks + sequence -> shifts.str + result.nef + report.html` (`nmrassign pipeline`)
- [x] (A) Self-hosted HTTP REST job API and background solver workers (`nmr-core::service`, `nmrassign serve`)
- [x] (A) Single-page browser upload & live job results dashboard (`GET /` on `nmrassign serve`) + `wasm32` WebGPU workbench
- [x] (A) Self-contained HTML + inline SVG quality report with CSI secondary shifts, Ramachandran plot, strip ladders & confidence summaries (`nmr-core::report`, `nmrassign report`)

## Phase 7: Integrated Octet-Draw Chemical Structure, Reaction & 1D NMR Suite
- [x] (A) Core 2D chemical graph, SSSR rings, Hückel aromaticity, CIP `(R)/(S)` & `(E)/(Z)` stereochemistry, Clean 2D layout, peptide/HELM builder, and multi-format I/O (`SMILES`, Reaction SMILES, `InChI`/`InChIKey`, `MOL` V2000/V3000, `SDF`, ChemDraw `CDXML` & binary `CDX`, `.oct`) (`nmr-core::octet_mol`)
- [x] (A) Analytical chemistry engine: 6-sphere HOSE + additivity ¹H/¹³C NMR shift & first-order multiplet predictor, Lorentzian 1D synthesis & `JCAMP-DX` I/O, IUPAC 2021 exact mass / isotopic envelopes / cLogP / TPSA, reaction stoichiometry solver, and offline bidirectional `IUPAC Name ↔ Structure` (`nmr-core::octet_analyt`, `nmrassign chem`)
- [x] (A) Publication-grade 2D chemical renderer (`ACS 1996`, `RSC`, `JOC`, `Angewandte`, `Nature`, `Dark`), 50+ template library, and pure-Rust `SVG`, `PDF 1.4`, and antialiased `PNG` exporters (`nmr-core::octet_render`)
- [x] (A) Interactive `egui` Octet Chemical Studio tab (`⚗️ Octet Chemical Draw & 1D NMR`) wired into `nmr-gui` (native desktop + `wasm32`) with live 1D NMR peak-to-atom highlighting and 1-click import from loaded protein sequences

## Phase 8: Raw time-domain (FID) processing, NUS, DSP operators & frozen real-data benchmark (`specs/11`)
- [x] (A) `T08.1` — Clean-room raw FID, hypercomplex IST/FISTA NUS, and real-data benchmark specification (`docs/specs/11-fid-processing.md`)
- [x] (A) `T08.2` — Raw FID & NUS readers (`nmr-core::fid`): Bruker `fid`/`ser` (`acqus`/`acqu2s`/`acqu3s`, `GRPDLY`/`DSPFVS`/`DECIM`, 256-word block padding, Linux case-insensitive lookup) and Varian/Agilent `fid` + `procpar` (`f1coef`/`f2coef`), explicit quadrature (`States`, `States-TPPI`, `Echo-Antiecho`, `QF`), and NUS (`FnTYPE=2`/`NusTD`/`nuslist`, `sampling.sch`) auto-detection
- [x] (A) `T08.3` — Deterministic processing, 1D/2D hypercomplex IST/FISTA NUS reconstruction & Part 3 TopSpin + `CISD3` 3D NUS validation (`nmr-core::process`, 3D writers in `nmr-core::spectrum`, `ProcessingRecipe` JSON; Ubiquitin raw `ser` TopSpin parity: `63/66` (`95.5%`) vs `63/67` (`94.0%`) — both recover 63 correct residues, gap due to indirect LP (`ME_mod=LPfc`) doubling effective `13C`/`15N` aperture on `HNCO`/`HN(CA)CO`; `CISD3` 3D NUS `62.5%` acc / `100.0%` HC prec)
- [x] (A) `T08.4` — Composable multidimensional DSP vector-stream operators & blind 1D/2D/3D autophasing (`nmr-core::ops`: `LorentzToGauss`, `Trapezoid`, `JModulation`, `HilbertTransform`, `Forward`/`Backward`/`MirrorImage` LP, `MedianBaseline`, `.ft1`/`test%03d.ft3` I/O, biphasic indirect autophase)
- [x] (A) `T08.5` — Two-pass experiment-guided corroborated weak-peak rescue in raw 3D strips (`nmr-core::picker::StripBank3D` / `corroborated_re_pick_3d`, `specs/02`)
- [x] (A) `T08.6` — Public archive fetcher (`nmrassign fetch-bmrb` with `<= 1 req/s`, `User-Agent: nmrassign-fetch`, `PROVENANCE.md` SHA-256 digests) and frozen 16-entry (`7 dev` / `9 heldout`) manifest (`data/benchmark/real_manifest.tsv`)
- [x] (A) `T08.7` — CLI raw-FID integration (`nmrassign process`, `--fid` on `pick`/`pipeline`, `nmrassign real-benchmark`; `dev` split verified, single-shot cold `heldout` run recorded at merge)

## Phase 9: Zero-friction spectrometer ingest, interactive NMR workbench UX & solver curation (`specs/06`, `specs/12`)
- [x] (A) `T09.1` — Spectrometer folder & archive auto-classifier (`nmr-core::classify`: Bruker `PULPROG`/`vdlist`/`fqlist`/`nuslist` & Varian `seqfil`/`pslabel` -> experiment & problem-regime detection; `< 50 ms` on unpacked folders & seekable `.tar` archives, trailing zero-padding tolerance on `.tgz`)
- [x] (A) `T09.2` — Drag-and-drop folder/archive loader & editable Experiment Selection Matrix (`nmr-gui` desktop + `wasm32` WebGPU + `nmr-core::service` with loopback `Host` DNS-rebinding protection)
- [x] (A) `T09.3` — Guided *"What Next?"* workflow stepper, structured top menu bar, and fuzzy command palette (`Cmd/Ctrl+K`) in `nmr-gui`
- [x] (A) `T09.4` — Interactive two-key NMR spectrum keyboard accelerators (`vt`, `pa`, `pd`, `pc`, `oz`, `ct`, `st`, `at`, `Space`, `?` cheat-sheet) comparable to modern desktop assignment suites (`nmr-gui::keymap`)
- [x] (A) `T09.5` — Synchronized 2D/3D/4D multi-spectrum contour overlays and ranked `i-1 / i / i+1` candidate match-finding strips (`nmr-gui`)
- [x] (A) `T09.6` — Bounded undo/redo command stack (`Cmd/Ctrl+Z`) and lossless `.nef` + workspace session persistence (`nmr-core::session`, `nmr-gui`)
- [x] (A) `T09.7` — Backbone solver curation & strict LOPO calibration follow-ups: `Gln2` 9% minor-conformer weak-peak penalty, `HNCACB`-absent sparse-data convergence, locked residue constraints, leakage-free unit-temperature (`T = 1.0 nat`) Pass-1 `evo-graph` beliefs (`CANONICAL_CALIBRATION_GRID_12`, `Beta(0.25, 0.25)` prior; `P1-7`), and engine-matched 10-fold OOF LOPO calibration (`STANDARD_CPU_CALIBRATION_KNOTS`, `STANDARD_GPU_CALIBRATION_KNOTS`, `EVO_GRAPH_CALIBRATION_KNOTS` all `<= 0.060` mean `ECE`, `<= 0.050` pooled `ECE`, `>= 94%` HC prec on `dev`; `P1-8`)

## Phase 10: Interactive covalent sidechain graph, aromatics, stereospecificity & Methyl-TROSY (`specs/07`, `specs/07b`, `specs/13`)
- [x] (A) `T10.1` — Interactive sequence ribbon & live 2D covalent amino-acid sidechain graph with per-atom assigned `ppm`, BMRB Z-scores, and strip jump (`nmr-core::topology`, `nmr-gui::sidechain_widget`; all 20 canonical AAs + 5 `Cys`/`His` variants planar with `min_dist >= 0.35`, `0` bond-node collisions, regular 5/6-rings, live `SyncCrosshair` jump, `wasm32`-clean)
- [x] (A) `T10.2` — Aromatic ring spin-system assignment (`(HB)CB(CGCD)HD`, `(HB)CB(CGCDCE)HE`, aromatic `13C-HSQC`/`HCCH-TOCSY`/`NOESY` -> `Phe`/`Tyr`/`Trp`/`His` rings via unified `HisVariantPriors`, Pelton Table 1 + `HisStateClassification` fast-exchange/metal-`His` abstention, unlabelled Yamazaki `CB` alignment, and Kuhn–Munkres multi-candidate block-Boltzmann matching; `nmr-core::aromatic`, `nmrassign sidechain --aromatic`; 12-protein **synthetic self-consistency**: `95.67%` (`442/462`) acc, `92.86%` (`442/476`) rec, `98.94%` (`374/378`) HC prec, `100.00%` (`18/18`) `HisTautomer`; **non-circular `P0-2` 4-entry real-BMRB suite** (`bmr17769`, `bmr4562`, `bmr16649` `Zn-His52` `4/4` abstention, `mbp_bmr7114` `0` FPs): `89.77%` (`588/655`) acc, `81.67%` rec, `95.07%` (`424/446`) HC prec, `ECE = 0.0783`)
- [x] (A) `T10.3` — Stereospecific prochiral (`HB2`/`HB3`, `HG1`/`HG2`, `HD1`/`HD2`) Bayesian assignment from Student-$t$ ($\nu = 4$) `3J` couplings, local NOEs & tempered Lovell rotamer priors (`nmr-core::stereospec`, `nmrassign sidechain --stereospec`; **synthetic self-consistency** 10-seed `1D3Z`: `99.58%` (`950/954`, `92.6%` cov) on `1D3Z` `(φ,ψ)` & `99.56%` (`900/904`, `87.8%` cov) on CSI `(φ,ψ)`; **non-circular `P0-2` experimental `BMRB 5387` vs `BMRB 17769` Ubiquitin `92`-coupling suite** (`data/fixtures/ubiquitin_exp_3j.tsv`): **`100.00%` (`16/16`, `61.5%` multi-pathway cov; `160/160` 10-seed scrambled) precision** on both `1D3Z` & CSI `(φ,ψ)` + `12/12` dynamic surface abstentions; perturbed-Karplus mismatch `99.58%` / `99.67%` prec)
- [x] (A) `T10.4` — Supramolecular `13C`-Methyl TROSY (`ILVMAT` `2D 13C-HMQC` + `3D/4D` Methyl-NOESY) picker & isotopic labeling schemes (Tugarinov & Kay *JACS* 2003; `nmr-core::methyl_trosy`; **synthetic self-consistency** `25/18 Hz` recovery `100.00%` on 2D/3D/4D Ubiquitin & `99.87%` (`772/773`) on 3D `MBP 1EZP`, `<= 2.77%` artifacts)
- [x] (A) `T10.5` — WebGPU/CPU Methyl-TROSY structure-based NOE contact-graph assignment solver via **exact Metropolis–Hastings independent multi-replica simulated annealing** (`TVD <= 0.0168` CPU / `<= 0.0157` WebGPU, `C_K` symmetry-orbit calibration, `<= 4,096`-step TDR chunking, `hc_abstained` `0/0` semantics; `nmr-core::methyl_solver`, `nmr-gpu::methyl_gpu`, `nmrassign methyl`; CPU/WebGPU parity `<= 3.45e-6`, `MBP C2/C6` `anneal_methyl_1024`/`global` parity `<= 9.16e-7`; **synthetic self-consistency**: `Ubiquitin` `92.86%` acc / `97.30%` HC, `MBP 1EZP` `95.21%` CPU & `92.55%` GPU acc / `98.53%` HC; **non-circular `P0-2` X-ray cross-structure suite** (`1D3Z -> 1UBQ`, `1DMB <-> 1OMP`): `80.95%`–`88.42%` acc, pooled `94.3%` (`283/300`) HC prec, `ECE = 0.0570 <= 0.06`)

## Phase 11: Problem-regime GPU solvers — IDPs (`13C`-detected) & paramagnetic metalloproteins (`specs/14`)
- [x] (A) `T11.1` — Protonless `13C`-detected IDP experiment models, narrow-dispersion picker & volumetric simulator (`2D CON`, `CACO`, `3D CANCO`, `CBCACON`, `CCCON`, `NCACO`; Bermel et al. 2006–2009; `nmr-core::c13_detected`; dynamic `500–950 MHz` `Hz -> ppm` scaling, IPAP `1J(C',CA)` ghost suppression, Bai–Englander exchange model; volumetric `2D CON` recovery `98.54%` (`202/205`) on `bmr26639` `21.7% Pro` & `98.85%` (`86/87`) on `bmr51984` `21.6% Pro`)
- [x] (A) `T11.2` — Continuous through-Proline `13C`-detected IDP backbone CPU/WebGPU solver (`nmr-core::idp_solver`, `nmr-gpu::idp_gpu`, `nmrassign assign --regime idp-13c`; `ncIDP` + Platzer pH + Kjaergaard $\partial\delta/\partial T$, unlabelled reference & `NCACO` offset registration, `(U, P)` CPU/WebGPU parity `<= 5.72e-6 < 1e-4`, `16 KiB` SRAM `anneal_idp_256`/`1024`/`global`; **non-circular 5-protein real BMRB IDP suite** (`bmr6968`, `bmr15176`, `bmr4375`, `bmr26639`, `bmr51984`): `94.24%`–`99.01%` acc, `98.44%`–`100.00%` HC prec, pooled `99.82%` HC prec, `ECE = 0.0231` CPU / `0.0227` WebGPU)
- [x] (A) `T11.3` — IDP Secondary Structure Propensity (`SSP` / `δ2D`: Marsh et al. 2006 5-residue weighted `SSP` + Camilloni et al. 2012 Forward–Backward 4-state `(p_alpha, p_beta, p_ppii, p_coil)` with unlabelled offset & pH sensing; `nmr-core::idp_propensity`, `nmrassign propensity`, `nmrassign report --idp-propensity`; `bmr17769` Ubiquitin `38/40 = 95.00%` vs `1D3Z`, disordered `bmr6968` `0.908` & `bmr4375` `0.868` `coil+PPII`)
- [x] (A) `T11.4` — Paramagnetic metalloprotein `PCS` / `PRE` $r_{\text{M}}^{-6}$ bleaching & $\Delta\chi$-tensor-aware assignment (`nmr-core::paramagnetic`, `nmrassign assign --regime paramagnetic`; `[2Fe-2S]`/`[4Fe-4S]`/Zn-finger motifs, 3D multi-cluster donor partitioning, `u_para` annealing + `>= 2`-slot rescue, 5-param Jacobi SVD + Huber IRLS + analytic $\nabla_{\mathbf{r}}\Delta\delta^{\text{PCS}}$ GLS + 8-param `VarPro`; noisy `Tm3+`/`Tb3+`/`Dy3+` `< 0.61%` error; real 5-seed `CISD3` `BMRB 51439`: `Standard` `53.33% -> 60.61%` acc, `EvoGraph` `55.43% -> 62.26%` acc, `100.00%` HC prec, `0` loop FPs)

## Phase 12: Built-in interactive `wgpu` 3D biomolecular structure viewer & RDC refinement (`specs/04`, `specs/15`)
- [ ] (A) `T12.1` — Native & `wasm32` `wgpu` 3D biomolecular cartoon ribbon, sidechain stick & multi-conformer ensemble viewer (`nmr-gui::mol3d_viewer`)
- [ ] (A) `T12.2` — Linked 3D-to-strip atom picking, quantitative putty/color mapping (confidence, `CSI`, Lipari–Szabo $S^2$, $R_{\text{ex}}$, `CSP` $\Delta\delta$), and 3D NOE/PRE violation cylinders (`nmr-gui`)
- [ ] (A) `T12.3` — Post-assignment database-driven 3/9-mer $(\phi, \psi, \chi_1)$ torsion restraint predictor (Shen et al. *PNAS* 2008; Shen & Bax *J. Biomol. NMR* 2013; `nmr-core::torsion_db`)
- [ ] (A) `T12.4` — Residual Dipolar Coupling (`RDC`) Saupe tensor SVD, Cornilescu $Q$-factor & GPU torsion-angle ensemble refinement (`nmr-core::rdc`, `nmr-gpu`, `nmrassign structure --rdc`)

## Phase 13: GPU spin relaxation, Lipari–Szabo / Palmer model-free ($S^2$), `CEST`, `CPMG` & ligand `CSP` $K_d$ (`specs/16`)
- [ ] (A) `T13.1` — Pseudo-3D relaxation & exchange plane series integrator (`vdlist`/`vclist`/`fqlist` & Varian arrayed planes with Voigt heights/volumes + uncertainties; `nmr-core::relax_io`)
- [ ] (A) `T13.2` — Batched GPU $R_1$, $R_2$ & steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE exponential fitter with Monte Carlo bootstrap & trimmed $R_2/R_1$ $\tau_m$ (`nmr-core::relax_fit`, `nmr-gpu`, `nmrassign relax`)
- [ ] (A) `T13.3` — Lipari–Szabo / Palmer Model-Free (`M1`–`M5`: $S^2, S_f^2, \tau_m, \tau_e, R_{\text{ex}}$) & anisotropic diffusion solver with AIC/BIC selection (Lipari & Szabo *JACS* 1982; Mandel, Akke & Palmer *JMB* 1995; `nmrassign modelfree`)
- [ ] (A) `T13.4` — GPU 2-/3-state Bloch–McConnell Chemical Exchange Saturation Transfer (`CEST`) & `DEST` matrix-exponential solver (Vallurupalli, Bouvignies & Kay *JACS* 2012; `nmr-core::cest`, `nmr-gpu`, `nmrassign cest`)
- [ ] (A) `T13.5` — GPU Carver–Richards / Baldwin `CPMG` & off-resonance $R_{1\rho}$ relaxation dispersion solver (`nmr-core::cpmg`, `nmr-gpu`, `nmrassign cpmg`)
- [ ] (A) `T13.6` — Ligand titration Chemical Shift Perturbation (`CSP`), tight-binding quadratic $K_d$ isotherms (Williamson 2013), interactive GUI dynamics studio & 3D pocket mapping (`nmr-core::csp`, `nmr-gui`, `nmrassign csp`)

## Phase 14: Automated visual assets, slim README, GitHub-Pages manual & live WebGPU demo
- [ ] (A) `T14.1` — Deterministic CLI figure & SVG/PNG asset exporter (`nmrassign export-fig` -> `docs/assets/`: 2D contours, 3D strips, covalent sidechains, 3D ribbons, relaxation/CEST/CSP curves)
- [ ] (A) `T14.2` — Slim, visual-first `README.md` rewrite (< 180 lines with hero figures, 60-second quickstart, benchmark table, and documentation links)
- [ ] (A) `T14.3` — Hostable GitHub-Pages documentation manual (`docs/manual/` + `.github/workflows/docs.yml` covering tutorials, CLI reference, formats, and clean-room citations)
- [ ] (A) `T14.4` — Zero-install live in-browser `wasm32` WebGPU demo bundle (`crates/nmr-gui/index.html`, `docs/demo/`) preloaded with sample 2D/3D spectra, PDB ensemble & relaxation series

## Housekeeping
- [x] (A) CCPN NEF v1.1 lossless import/export (`nmr-core::nef`, `nmrassign nef-import` / `nef-export`)
- [x] (A) Single-source SemVer versioning (`v0.7.0`, `nmr_core::VERSION`, `nmrassign --version` with short git SHA, NEF/HTML/API provenance, annotated `vX.Y.Z` git tags)
- [ ] (M) Choose a public project name (avoid commercial product names)
- [ ] (M) Patent search on deep-learning peak picking and automated assignment
- [x] (A) Add `CONTRIBUTING.md`, `THIRD_PARTY.md`, author metadata, and CI workflow
