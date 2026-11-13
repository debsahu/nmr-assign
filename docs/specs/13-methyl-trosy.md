# 13. Supramolecular `13C`-Methyl TROSY (`ILVMAT`) Spectroscopy & Contact-Graph Assignment

## Provenance & Clean-Room Statement
Original clean-room specification and mathematical design. Zero source code, macros, parameter tables, residue libraries, or binaries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) NMR assignment or processing package were consulted. All physical models, isotopic labeling topologies, lineshape simulations, peak-picking algorithms, and contact-graph optimization equations are derived strictly from peer-reviewed literature and public PDB/BMRB data standards:

1. **Methyl-TROSY physics & high-molecular-weight spectroscopy**:
   - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E. (*J. Am. Chem. Soc.* **2003**, 125, 10120–10121 & 10420–10428) — `${}^{1}\text{H}\text{–}{}^{13}\text{C}$ methyl-TROSY `HMQC` exploiting cross-correlated dipolar relaxation interference in perdeuterated high-molecular-weight proteins (`40–800 kDa`).
   - Tugarinov, V. & Kay, L. E. (*J. Am. Chem. Soc.* **2003**, 125, 13868–13878) — `HMCM(CG)CBCA` out-and-back scalar-coupled correlation of `Ile`, `Leu`, and `Val` methyls to aliphatic sidechain/backbone carbons (`CG`, `CB`, `CA`) and intraresidue geminal methyl pairing.
   - Ruschak, A. M. & Kay, L. E. (*J. Biomol. NMR* **2010**, 46, 75–87) — methyl-TROSY `3D C-C-H` and `4D C-H-C-H` `HMQC-NOESY-HMQC` spectroscopy in supramolecular assemblies.
2. **Stereospecific & multi-residue methyl precursor labeling**:
   - Gans, P. et al. (*Angew. Chem. Int. Ed.* **2010**, 49, 1958–1962) — stereospecific `pro-R` (`Leu-CD1`, `Val-CG1`) and `pro-S` (`Leu-CD2`, `Val-CG2`) `${}^{13}\text{CH}_3$ acetolactate precursor labeling in perdeuterated backgrounds.
   - Markley, J. L. et al. (*J. Mol. Biol.* **1998**, 280, 933–952) — IUPAC-IUBMB-IUPAB stereospecific biochemical NMR atom nomenclature (`VAL CG1`/`CG2`, `LEU CD1`/`CD2`).
3. **Structure-based methyl NOE contact-graph assignment**:
   - Xu, Y., Lin, Z., Ho, C. & Yang, D. (*J. Am. Chem. Soc.* **2009**, 131, 9480–9481) — automated methyl resonance assignment from methyl–methyl NOE networks and 3D coordinates.
   - Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G. (*J. Magn. Reson.* **2012**, 214, 103–110) — combinatorial/stochastic methyl NOE network matching against known 3D structures.
   - Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J. & Kay, L. E. (*J. Am. Chem. Soc.* **2017**, 139, 9523–9533) — probabilistic methyl NOE contact-graph matching incorporating amino-acid shift priors, geminal coupling, rotamer flexibility, and missing/false-peak noise floors.
4. **Local structural chemical-shift physics**:
   - Haigh, C. W. & Mallion, R. B. (*Prog. Nucl. Magn. Reson. Spectrosc.* **1979**, 13, 303–344); Perkins, S. J. & Wüthrich, K. (*Biochim. Biophys. Acta* **1979**, 576, 409–423) — aromatic ring-current dipolar `${}^{1}\text{H}$ shift perturbations near `Phe`, `Tyr`, `Trp`, and `His` rings.
   - Ulrich, E. L. et al. (*Nucleic Acids Res.* **2008**, 36, D402–D408) — public BioMagResBank (`BMRB`) empirical chemical-shift statistics.

---

## 1. Isotopic Precursor Labeling Schemes & Methyl-Type Classifier (`crates/nmr-core/src/methyl_trosy.rs`)

### 1.1 Precursor Labeling Schemes (`MethylLabelingScheme`)
Selective `${}^{13}\text{CH}_3$ protonation on a perdeuterated (`U-[2H, 12C]`) background eliminates `1H–1H` dipolar relaxation sinks and scalar `13C–13C` $J$-couplings while populating a well-defined subset of protein methyl sites (`MethylGroupKind`):

| Scheme Variant | CLI Token | Active `(Residue, Carbon, Protons)` Sites | Geminal Pairs Active? |
|---|---|---|---|
| `IlvProS` | `ILV-proS` | `ILE:CD1`, `LEU:CD2` (`pro-S`), `VAL:CG2` (`pro-S`) | No (1 methyl / `Leu`,`Val`) |
| `IlvProR` | `ILV-proR` | `ILE:CD1`, `LEU:CD1` (`pro-R`), `VAL:CG1` (`pro-R`) | No (1 methyl / `Leu`,`Val`) |
| `IlvRacemic` | `ILV-racemic` | `ILE:CD1`, `LEU:CD1,CD2`, `VAL:CG1,CG2` | Yes (`Leu`, `Val`) |
| `Ilvmat` | `ILVMAT` | `ILE:CD1`, `LEU:CD1,CD2`, `VAL:CG1,CG2`, `MET:CE`, `ALA:CB`, `THR:CG2` | Yes (`Leu`, `Val`) |
| `Custom(mask)` | `RES:ATOM,...` | Arbitrary caller-specified set of `(Comp_ID, Atom_ID)` methyls | Automatic when both `CD1,CD2` or `CG1,CG2` are active |

### 1.2 Bayesian 2D `(1H, 13C)` & `HMCM(CG)CBCA` Methyl-Type Classifier
Each observed 2D `${}^{1}\text{H}\text{–}{}^{13}\text{C}$ `HMQC` root peak $s = (H_s, C_s)$ is assigned a normalized posterior probability vector $P(\text{class}(s) = c \mid H_s, C_s, \text{HMCM}_s, \mathcal{S})$ across the six methyl-bearing residue classes $c \in \{\text{Ile}, \text{Leu}, \text{Val}, \text{Met}, \text{Ala}, \text{Thr}\}$:

1. **Labeling mask**: Any class $c$ absent from the active `MethylLabelingScheme` $\mathcal{S}$ receives exact prior weight $0$.
2. **2D `(1H, 13C)` empirical density with ring-current tail robustness**:
   - Canonical BMRB regions separate `Ile-CD1` (`~11–15 ppm`), `Met-CE` (`~14–19 ppm`, `1H ~ 1.8–2.3 ppm`), `Ala-CB` (`~16–22 ppm`, `1H ~ 1.1–1.6 ppm`), `Thr-CG2` (`~19–23 ppm`, `1H ~ 1.0–1.5 ppm`), and isopropyl `Leu-CD*` / `Val-CG*` (`~18–27 ppm`, `1H ~ 0.5–1.1 ppm`).
   - Because hydrophobic methyls packed over aromatic rings (`Phe`, `Tyr`, `Trp`, `His`) routinely experience upfield ring-current shifts of `-0.5` to `-2.0 ppm` in `${}^{1}\text{H}$ (e.g., `MBP` `Leu160-CD1` at `-1.02 ppm`, `Met336-CE` at `+0.27 ppm`), the `${}^{1}\text{H}$ likelihood uses a two-component Gaussian + heavy-tailed ring-current mixture:
     $$p_H(H_s \mid \mu_{H,c}, \sigma_{H,c}) = 0.92\,\mathcal{N}(H_s \mid \mu_{H,c}, \sigma_{H,c}^2) + 0.08\,\mathcal{N}(H_s \mid \mu_{H,c} - 0.50,\; (0.85)^2)$$
     multiplied by the `${}^{13}\text{C}$ Gaussian density $\mathcal{N}(C_s \mid \mu_{C,c}, \sigma_{C,c}^2)$.
3. **Optional `HMCM(CG)CBCA` scalar-relay discrimination & geminal linking**:
   - When `HmcmcbcaRecord` shifts `(CG, CB, CA)` are provided for root $s$, their empirical BMRB log-likelihoods cleanly separate `Leu` (`CB ~ 42.2 ppm`, `CG ~ 26.8 ppm`) from `Val` (`CB ~ 33.1 ppm`, no `CG` relay), `Ile` (`CB ~ 38.6 ppm`, `CG1 ~ 26.9 ppm`), `Ala` (`CA ~ 52.5 ppm`), and `Thr` (`CB ~ 70.0 ppm`).
   - `link_hmcmcbca_geminal_pairs` pairs two roots $(s_1, s_2)$ in `IlvRacemic` / `Ilvmat` whenever their relayed `(CB, CG, CA)` frequencies agree within experimental tolerance (`0.18 ppm`) inside the `Leu`/`Val` carbon envelope.

---

## 2. High-MW Volumetric Simulator & Multi-Dimensional Picker (`crates/nmr-core/src/methyl_trosy.rs`)

### 2.1 `40–80 kDa` Linewidth Regime & Pseudo-Voigt Lineshape
In `40–80 kDa` perdeuterated methyl-TROSY targets (`800–900 MHz`), transverse relaxation yields broader linewidths than small-protein backbone spectra: nominal full-width at half-maximum (`FWHM`) defaults to **`25.0 Hz` (`1H`)** and **`18.0 Hz` (`13C`)**.

Processed frequency-domain peaks (following cosine-bell / Lorentz-to-Gauss apodization) are modeled by a normalized convex pseudo-Voigt profile:
$$\psi(\Delta\nu; \Gamma) = (1 - \eta)\exp\!\left(-4\ln 2\,\frac{\Delta\nu^2}{\Gamma^2}\right) + \frac{\eta}{1 + 4\,\Delta\nu^2 / \Gamma^2}, \qquad \eta = 0.15$$
Because both components equal $\frac{1}{2}$ at $\Delta\nu = \pm \frac{1}{2}\Gamma$, $\psi(\Delta\nu; \Gamma)$ has **exact analytical `FWHM` $= \Gamma$** (`lw_hz / obs_mhz` in ppm) while reproducing slight Lorentzian wings.

### 2.2 `2D 1H-13C HMQC` Two-Pass Shoulder-Deconvolving Picker
1. **Pass 1 (Local Maxima & Sub-Voxel Interpolation)**: Estimates baseline thermal noise $\sigma_{\text{MAD}}$ via two-pass outlier-trimmed median absolute deviation (`1.4826 * median(|x - med|)`), identifies 2D `3x3` local maxima above `snr_hmqc * noise_mad` passing row/column `sinc_frac` skirt hygiene, and refines `(H, C)` centers via 3-point sub-voxel log-Gaussian/parabolic interpolation.
2. **Pass 2 (High-MW Blended-Pair Residual Rescue)**: Subtracts each Pass-1 peak's fitted 2D pseudo-Voigt lineshape (`FWHM` $= (\Gamma_H, \Gamma_C)$) to form a residual spectrum and rescues secondary blended shoulder roots exceeding `shoulder_snr * noise_mad` whose centers sit at least `0.48 * FWHM` from existing roots.

### 2.3 `3D C-C-H` & `4D C-H-C-H` `HMQC-NOESY-HMQC` Simulator & Reciprocal Picker
1. **NOE Intensity Transfer Model**:
   Cross-peaks between distinct active methyl sites $(m_1, m_2)$ separated by 3D carbon–carbon distance $d_{\text{CC}} = \|\mathbf{r}_C(m_1) - \mathbf{r}_C(m_2)\|_2 \le r_{\text{max}}$ follow damped $r^{-6}$ dipolar cross-relaxation accounting for fast internal methyl rotation and short-range multi-spin saturation ($d_{\text{sat}} = 3.8\text{ \AA}$):
   $$I_{\text{NOE}}(d_{\text{CC}}) = I_{\text{ref}} \left(\frac{r_{\text{ref}}}{\left(d_{\text{CC}}^6 + d_{\text{sat}}^6\right)^{1/6}}\right)^6$$
2. **Root-Anchored Reciprocal `3D C-C-H` (`(C_noe, C_root, H_root)`) Picker**:
   - Extracts gain-normalized bilinear 1D `C_noe` strips at every picked `2D HMQC` root $s = (H_s, C_s)$ (`Spectrum3D` layout: `axis_x = 1H_root`, `axis_y = 13C_root`, `axis_z = 13C_noe`), analytically subtracting both self-diagonal auto-peaks and 2D `(H_root, C_root)` point-spread bleed from neighboring roots' diagonals (`amp_diag[u] * psi_H(H_s - H_u) * psi_C(C_s - C_u)`).
   - Performs multi-pass 1D pseudo-Voigt CLEAN deconvolution along `C_noe` with sub-voxel refinement, followed by **unsupervised reciprocal `(s <-> t)` dipolar symmetry corroboration** across the picked root catalog (`hi <= 3.35 * lo`), deduplicating `C_noe`-degenerate physical cross-peaks (`h_noe_ppm = None` in 3D) while rescuing weak `7.5–8.5 Å` contacts (`>= 96%` recovery, `< 3.0%` artifact rate).
3. **`4D C-H-C-H` `HMQC-NOESY-HMQC` (`Spectrum4DMethyl`)**:
   - Full 4D volumetric simulator and Cramér–Rao `3x3` pseudo-Voigt MVUE matched-filter root-plane `(C_noe, H_noe)` extractor with joint 4D self/neighbor-diagonal subtraction, 4D Kronecker non-negative Gauss–Seidel PSF deconvolution (`(R ⊗ N) X = M`), and reciprocal dipolar corroboration resolving cross-peaks simultaneously along both `(C_noe, H_noe)` donor and `(C_root, H_root)` recipient coordinates.

---

## 3. Structure-Based Contact-Graph Solver (`methyl_solver.rs`, `methyl_gpu.rs`, `methyl_anneal.wgsl`)

### 3.1 Probabilistic Contact-Graph Objective $E(\pi)$
Given $S$ observed methyl roots and $M$ active structural methyl sites extracted from a 3D PDB coordinate model, an injective partial assignment $\pi$ (`pos[0..S] -> {-1..M-1}`, `at[0..M] -> {-1..S-1}`) maximizes the per-root normalized objective $E(\pi) = \frac{1}{\max(S, 1)} E_{\text{raw}}(\pi)$, where:

$$E_{\text{raw}}(\pi) = \sum_{s: \text{pos}[s] \ge 0} U(s, \text{pos}[s]) + \sum_{\substack{(m_1, m_2) \in \mathcal{G}_{\text{struct}} \\ \text{at}[m_1], \text{at}[m_2] \ge 0}} G_{\text{gem}}(\text{at}[m_1], \text{at}[m_2]) + \sum_{\substack{e = (s_1, s_2) \in E_{\text{obs}} \\ \text{pos}[s_1], \text{pos}[s_2] \ge 0}} W_{\text{NOE}}\!\left(e,\; d_{\text{CC}}(\text{pos}[s_1], \text{pos}[s_2])\right) - \sum_{(m_1, m_2) \in E_{\text{short}}} \Lambda_{\text{miss}}(m_1, m_2; \pi)$$

1. **Unary Site Compatibility $U(s, m)$**:
   - Incompatible amino-acid methyl classes (`|z_C| > 5.5` or conflicting `HMCM(CG)CBCA` typing) receive `FORBID = -1.0e9`, restricting proposals to sparse compatible candidate lists `slots[s]`.
   - Compatible `(s, m)` pairs combine baseline assignment reward $w_{\text{assign}}$, empirical BMRB `${}^{13}\text{C}$ log-density (with mild $\chi_1/\chi_2$ $\gamma$-gauche rotamer expectation when heavy sidechain atoms exist in the PDB), structure-informed `${}^{1}\text{H}$ log-density incorporating Haigh–Mallion / Pople aromatic ring-current dipolar shifts and proximity variance inflation $\sigma_H^2(m) = \sigma_{H,0}^2 + \sigma_{\text{rc}}^2(m)$ near `Phe`/`Tyr`/`Trp`/`His` rings, and optional `HMCM(CG)CBCA` `(CA, CB, CG)` log-likelihoods.
2. **Geminal Intraresidue Coupling $G_{\text{gem}}(s_1, s_2)$**:
   - For active structural geminal pairs $(m_1, m_2)$ (`Leu CD1/CD2`, `Val CG1/CG2`, $d_{\text{CC}} \approx 2.50\text{ \AA}$), rewards confirmed `HMCM(CG)CBCA` geminal links (`+8.0`, penalizing broken links `-8.0`) or strong intraresidue NOE / diagonal-coalesced pairs (`+2.2` vs `-1.5` when absent).
3. **Bidirectional $r_{\text{CC}}^{-6}$ NOE Edge Agreement $W_{\text{NOE}}(e, d)$**:
   - Symmetrizes directed cross-peaks `s1 -> s2` and `s2 -> s1` into undirected edges $e = (s_1, s_2)$ with reciprocity weight $w_e = 1.35$ (bidirectional) or $0.75$ (unidirectional) and unsupervised inter-residue PDB-quantile $r^{-6}$-calibrated target distance $\hat{d}_e \in [2.45, 9.2]\text{ \AA}$.
   - For $d \le d_{\text{cut}}$ (`10.0 Å`), scores flat-bottom Gaussian distance agreement with $\sigma_d = 1.15\text{ \AA}$ rotamer/spin-diffusion tolerance; for $d > d_{\text{cut}}$, applies a smooth violation penalty bounded by a false-peak noise floor (`-4.20 * w_e`).
4. **Missing Short-Contact Penalty $\Lambda_{\text{miss}}(m_1, m_2; \pi)$**:
   - For structural contacts with $d_{\text{CC}}(m_1, m_2) \le r_{\text{obs}}$ (`7.6 Å`), $\lambda_{\text{miss}}(d) = \lambda_0 \left(\frac{r_{\text{obs}} - d}{r_{\text{obs}} - 2.5}\right)^2$ penalizes unexplained missing NOEs unless masked by diagonal overlap (`|C_{s1} - C_{s2}| < 0.18 ppm`).

### 3.2 CPU & WebGPU Independent Multi-Replica Simulated Annealing (`16 KiB` Portable SRAM & Exact Detailed Balance)
Both CPU (`methyl_solver.rs`) and WebGPU (`methyl_gpu.rs` + `methyl_anneal.wgsl`) initialize independent annealing replicas (`56` replicas by default in `MethylAnnealConfig`, `64 × seeds` in CLI `nmrassign methyl`, with zero inter-replica temperature exchange — **independent multi-replica simulated annealing**, not parallel tempering) on the cardinality-$k$ injective manifold ($k = \min(S, M)$) with intact `Leu`/`Val` geminal pairs and anneal each replica independently ($T_{\text{start}} \to T_{\text{end}}$) over four **exact Metropolis–Hastings reversible move kernels** touching at most 4 sites per step:
1. **Slot-guided placement / 2-way swap (`7/16` probability)**: uniform slot proposal $m \in \text{slots}[s]$ with exact Hastings log-ratio $\ln(|\text{slots}[s]| / |\text{slots}[t]|)$ when an unassigned root $s$ displaces an occupied site $m$ (`t = at[m] >= 0`), and `0` on symmetric vacant-move or 2-way swap transitions;
2. **3D NOE-neighbor + `1/8` slot-ergodic extension (`6/16` probability)**: proposes NOE partner / compatible 3D neighbor $m$ with exact forward/reverse Hastings correction $\ln(q_{\text{rev}} / q_{\text{fwd}})$ summed strictly over transition-invariant anchor roots (`move2_prop_prob_cpu`);
3. **State-independent 4-body geminal residue involution (`2/16` probability)**: symmetric involution $(m_1, m_2) \leftrightarrow (m_3, m_4)$ across structural `Leu`/`Val` geminal pairs (`log_hastings = 0`);
4. **Intraresidue prochiral flip (`1/16` probability)**: symmetric involution $m_1 \leftrightarrow \text{gem}[m_1]$ resolving stereospecific `CD1`/`CD2` and `CG1`/`CG2` orientations (`log_hastings = 0`).

Fixed-temperature stationary Markov-chain verification (`sample_methyl_mcmc_terminal_cpu` & WebGPU) matches the exact brute-force Boltzmann distribution $\pi(\sigma) \propto \exp(E(\sigma)/T)$ across `Omega_k` to total variation distance **`TVD = 0.0133–0.0168 < 0.025` on CPU** and **`TVD = 0.0146–0.0157 < 0.025` on WebGPU**.

In `methyl_anneal.wgsl` and `methyl_gpu.rs`:
- `eval_methyl_objective_batch` enforces exact injective-mapping validity (returning `-1.0e12` identically on CPU and WGSL for any out-of-bounds `pos[s]` with `m < -1 || m >= M` or duplicate site occupancy `at[m] != -1`) and accumulates total $E(\pi)$ via `-ffast-math`-immune split fixed-point (`i32` $\times 1024$ + residual `f32`) arithmetic, guaranteeing `max |E_cpu - E_gpu| < 1e-4`.
- `anneal_methyl_256` (`@workgroup_size(8, 1, 1)`, bank-conflict-free `WG_STRIDE_256 = 255` in `wg_at: array<i32, 2048>` + `wg_pos: array<i32, 2048>` = exact **`16,384 B` (`16 KiB`)** baseline WebGPU workgroup SRAM with single-pass zero-mutation `delta_swap_root_wg`), `anneal_methyl_1024` (`@workgroup_size(2, 1, 1)`, up to `1024` roots/sites in `16 KiB` SRAM), and `anneal_methyl_global` (`@workgroup_size(8, 1, 1)` storage fallback for `> 1024`) respect `bounded_adapter_limits` with 6 storage bindings and zero dynamic stack arrays.
- Command-buffer dispatches are chunked via `chunk_steps_for_dim` (`MAX_CHUNK_STEPS = 4_096` for `<= 255`, `2_048` for `<= 1,024`, `1_024` for `> 1,024`) to prevent OS GPU watchdog (`WDDM TDR`) resets, while `wasm32` synchronous entry points return a clean `Err` (`GpuEngine::read_buffer_vec_async` serves async browser callers).

### 3.3 Consensus Marginal, Cyclic $C_K$ Symmetry Orbits & Monotone Confidence Calibration
Across completed replicas (`consensus_and_calibrate_methyl`), Boltzmann-weighted replica marginals $p_{\text{cons}}(s \to m^*)$ are combined with:
1. **Global $C_K$ Homooligomeric Symmetry-Orbit Detection (`detect_cyclic_subunit_symmetries`)**: Automatically identifies cyclic subunit automorphisms $\tau_k(m) = (m + k B) \bmod M$ ($K \in \{2, 3, 4, 5, 6, 8, 10, 12\}$) so symmetric $C_K$ homodimers/oligomers include all symmetry-equivalent subunit images in the partition denominator, naturally bounding per-chain confidence to $\le 1 / K$ (`<= 0.5000` on `C2` homodimers, `<= 0.1667` on `C6` hexamers).
2. **Local-Swap Specificity & Monotone Calibration (`calibrate_methyl_confidence_monotone`)**: Evaluates 2-way single-root swaps and 4-body geminal residue swaps on the consensus assignment $\pi^*$:
   $$p_{\text{swap}}(s \to m^*) = \frac{1}{1 + \sum_{m' \in \text{slots}[s] \setminus \{m^*\}} \exp\!\left(\frac{E(\pi^*_{s \leftrightarrow m'}) - E(\pi^*)}{T_{\text{cal}}}\right)}$$
   and maps $p_{\text{raw}}(s) = p_{\text{cons}}(s \to m^*) \cdot p_{\text{swap}}(s \to m^*)$ through the monotone piecewise-linear reliability curve `calibrate_methyl_confidence_monotone` onto `[0.0, 1.0]`.
3. **Explicit Zero-HC Abstention Semantics (`MethylEvalMetrics`, `MethylBenchmarkMetrics`)**: Whenever `hc_total == 0` (`0` roots achieve `conf >= 0.80`, e.g. under exact $C_2$ or $C_6$ symmetry), metrics explicitly set `hc_abstained = true`, `hc_precision_opt = None`, and numeric `hc_precision = 0.0` (never reporting `0/0` abstention as `100.0%` precision).

### 3.4 CLI Contract (`crates/nmr-cli/src/methyl_cmd.rs`)
```text
nmrassign methyl --pdb <model.pdb> --hmqc <2d_or_tsv> --noesy <3d_or_tsv> \
  --labeling ILV-proS|ILV-proR|ILV-racemic|ILVMAT|<custom> \
  [--hmcmcbca <tsv>] [--engine auto|gpu|cpu] [--seeds <N>] [-o <out.str>]
```
Accepts either processed 2D/3D spectra (Bruker `pdata`, NMRPipe, Sparky `.ucsf` — automatically picked via `pick_methyl_hmqc_2d` and `pick_methyl_noesy_3d`) or pre-picked `.tsv` peak lists, solves the contact-graph assignment via independent multi-replica simulated annealing (`64 × seeds` replicas in CLI; `56` default in `MethylAnnealConfig`) on WebGPU with **automatic fallback to the multi-threaded CPU solver (emitting a stderr warning) whenever no WebGPU adapter is available** (`--engine cpu` runs directly on CPU), prints an assignment summary table, and writes NMR-STAR v3 output with `assignment_method = "structure_based_methyl_noe_graph"` and per-methyl confidence annotations.

### 3.5 Quantitative Benchmarks: Synthetic Self-Consistency, Large-Assembly GPU Kernels & Non-Circular `P0-2` X-Ray Cross-Structure Validation
1. **CPU-vs-WebGPU Objective Parity & Large-Assembly Oligomer Kernels (`crates/nmr-gpu/tests/methyl_gpu_tests.rs`)**:
   - Normalized objective parity `|E_cpu - E_gpu|`: `1.20e-6` on Ubiquitin (`1D3Z`), `3.45e-6` on monomeric `MBP` (`1EZP`, `370 aa`, `41 kDa`), `9.155e-7` on **`MBP` `C2` homodimer (`S = 376, M = 376`, `anneal_methyl_1024`)**, and `8.487e-7` on **`MBP` `C6` hexamer (`S = 188, M = 1,128`, `anneal_methyl_global`)** — all well within `< 1e-4`.
   - On the `MBP` `C2` homodimer (`S = 376, M = 376`), `anneal_methyl_1024` achieves **`100.00%` (`376/376`) $C_2$-orbit accuracy** while strictly capping `max_conf = 0.5000` (`hc_abstained = true`); on the `MBP` `C6` hexamer (`M = 1,128`), `anneal_methyl_global` caps `max_conf = 0.1667 <= 1/6` (`hc_abstained = true`).
2. **Volumetric Synthetic Self-Consistency Suite (`crates/nmr-gpu/tests/methyl_gpu_tests.rs`)**:
   - Across volumetric `25 Hz / 18 Hz` pseudo-Voigt simulations (`99.87%` [`772/773`]–`100.00%` picker recovery, `<= 2.77%` artifacts; forward NOEs simulated from the target PDB coordinates), Ubiquitin `1D3Z` (`ILVMAT`) achieves **`92.86%` (`39/42`) accuracy / `97.30%` (`36/37`) HC-precision**, and `MBP` `1EZP` (`370 aa`, `41 kDa`, `ILVMAT`) achieves **`95.21%` (`179/188`) accuracy / `98.53%` (`134/136`) HC-precision on CPU** and **`92.55%` (`174/188`) accuracy / `98.53%` (`134/136`) HC-precision on WebGPU**.
3. **Non-Circular X-Ray Cross-Structure & Spin-Diffusion Benchmarks (`P0-2`, `crates/nmr-core/tests/methyl_trosy_tests.rs`)**:
   - Evaluated strictly **non-circularly** by generating noisy methyl NOE datasets from one experimental structure (`1D3Z` solution NMR vs `1UBQ` `1.80 Å` X-ray Ubiquitin; `1DMB` maltodextrin-bound holo X-ray vs `1OMP` apo domain-hinged X-ray `MBP`) with experimental BMRB methyl shifts, noisy `HMCM(CG)CBCA` relays ($\sigma = 0.18\text{ ppm}$, `15%` missing geminal links), `15%` missing short-range NOEs, `10%` false noise NOEs, and `< 4.5 Å` 3-spin relay / spin-diffusion cross-peaks, then solving blindly against the **independent** target crystal coordinates:
     - **`1D3Z -> 1UBQ` (`1.80 Å` X-ray Ubiquitin, `ILVMAT`)**: **`80.95%` (`34/42`)** accuracy, **`92.6%` / `90.0%` HC-precision**.
     - **`1DMB <-> 1OMP` (`370 aa` apo/holo X-ray `MBP`, `ILVMAT`)**: **`88.42%` / `84.74%`** accuracy, **`95.7%` / `94.4%` HC-precision**.
     - **Pooled 3-Pair X-Ray Cross-Structure Reliability (`N = 461` assigned methyls)**: **`94.3%` (`283/300`) HC-precision** and **`ECE = 0.0570 <= 0.06`**.
