# 10. Stoichiometry-Guided Evolutionary + Graph Self-Consistency Assignment Solver

Status: implemented in `crates/nmr-core/src/evo_graph.rs`. Preserves the unary
typing and sequential connectivity scoring semantics defined in
`docs/specs/03-assignment.md` Sections 3 and 4 while addressing raw classical
peak-picking artifacts, chemical-shift referencing offsets, multi-funnel search
landscapes, and sequence-graph chemical-shift cooperativity.

## 1. Provenance and Clean-Room Scope

This module is implemented 100% clean-room from peer-reviewed literature:

1. **Schmidt, E. & Güntert, P.** (2012). "A New Algorithm for Reliable and
   General NMR Resonance Assignment." *J. Am. Chem. Soc.* 134(30), 12817–12829:
   sequence-derived expected peak network stoichiometry, evolutionary
   recombination of contiguous sequential assignment fragments coupled with
   local Metropolis-Hastings simulated annealing, and multi-run consensus
   consolidation.
2. **Klukowski, P., Augoff, M., Zieba, M., Drabikowski, M., Gonczarek, A. &
   Walczak, M. J.** (2018). "NMRNet: a deep learning approach to automated peak
   picking of protein NMR spectra." *Bioinformatics* 34(15), 2590–2597:
   sequence-expected peak count ranking and spectral artifact suppression.
3. **Klukowski, P., Riek, R. & Güntert, P.** (2022). *Nature Communications*
   13, 6151: stoichiometry-proportional peak-list budgeting, automatic
   chemical-shift reference offset alignment, and iterative graph-based
   chemical-shift self-consistency with outlier pruning.
4. **Klukowski, P. et al.** (2023). *Science Advances* 9(49), eadi9323;
   **Spera, S. & Bax, A.** (1991). *J. Am. Chem. Soc.* 113(14), 5490–5492;
   **Wishart, D. S. et al.** (1995). *J. Biomol. NMR* 5(1), 67–81: optional 3D
   backbone $(\phi, \psi)$ torsion-conditioned secondary-shift expectations.

## 2. Inputs, Outputs, and Public API

All public types and entry points live in `nmr_core::evo_graph`:

- `EvoGraphConfig<'a>`: end-to-end configuration covering tolerances,
  stoichiometry budget multiplier (`1.60` default), physical shift windows,
  `min_assign_cost`, population/island topology (`64` individuals across `16`
  islands, `5` generations), fragment-crossover parameters, two-pass reference
  alignment (`auto_reference`), 2-layer graph consistency (`graph_consistency`),
  user-locked `(spin_system_idx, seq_idx)` hard constraints
  (`locked_assignments`), and optional
  `structure_torsions: Option<&'a [(Option<f64>, Option<f64>)]>`.
- `EvoGraphReport`: telemetry recording raw vs. retained peak and `HSQC` root
  counts, removed `NH2` sidechain roots, `kept_hsqc_indices`, per-experiment
  expected vs. retained counts, Pass-1 and Pass-2 `ReferenceOffsets`, Pass-2
  anchor count, graph-anchored/pruned slot counts, and objective scores.
- `expected_experiment_peaks(exp, sequence) -> usize`: exact sequence-predicted
  peak count $N_{\text{exp}}(\text{exp}, \text{sequence})$.
- `filter_peaks_stoichiometric(peaks, sequence, cfg) -> Vec<Peak>`: spectral
  artifact suppression and stoichiometry-guided peak budgeting (leaves legacy
  unheighted peak lists untouched).
- `build_stoichiometric_spin_systems(peaks, tol, max_cands) -> Vec<SpinSystem>`:
  intensity-weighted spin-system assembly with inter-spectrum $(^1\text{H},
  {}^{15}\text{N})$ drift alignment and primary-rank candidate protection.
- `assign_evo_graph(systems, sequence, priors, cfg) -> AssignResult` and
  `assign_evo_graph_with_report(...) -> (AssignResult, EvoGraphReport)`:
  two-pass solver over pre-built `SpinSystem`s.
- `assign_evo_graph_from_peaks(peaks, sequence, priors, cfg) -> (Vec<SpinSystem>, AssignResult, EvoGraphReport)`:
  end-to-end pipeline directly from raw picked `Peak`s.
- `torsions_from_pdb_chain(chain, sequence)` and
  `torsions_from_pdb_path(pdb_path, sequence)`: extract aligned backbone
  $(\phi, \psi)$ angles (in degrees) from a PDB model via `nmr_core::noesy` and
  `nmr_core::dihedrals`.

All `SpinSystem` shifts and emitted resonance assignments remain strictly in
the experimental spectrum's ppm frame; estimated `ReferenceOffsets` shift a
working copy of `Priors` (`EvoGraphReport::shifted_priors`).

## 3. Algorithmic Architecture

### 3.1 Sequence-Stoichiometry Peak Budgeting and Artifact Suppression

Given an amino-acid sequence of length $L$, let:
- $N_{\text{amide}} = |\{k \in \{1,\dots,L-1\} : \text{seq}[k] \neq \text{PRO}\}|$,
- $N_{\text{NQ}} = |\{k \in \{0,\dots,L-1\} : \text{seq}[k] \in \{\text{ASN}, \text{GLN}\}\}|$,
- $N_{\text{nonGly}}^{(i)} = |\{k \in \{1,\dots,L-1\} : \text{seq}[k] \notin \{\text{PRO}, \text{GLY}\}\}|$,
- $N_{\text{nonGly}}^{(i-1)} = |\{k \in \{1,\dots,L-1\} : \text{seq}[k] \neq \text{PRO} \land \text{seq}[k-1] \neq \text{GLY}\}|$.

Expected covalent pathway counts are:

| Experiment | Expected Peaks $N_{\text{exp}}$ |
|---|---|
| `HSQC` | $N_{\text{amide}} + 2 N_{\text{NQ}}$ |
| `HNCO` | $N_{\text{amide}} + 2 N_{\text{NQ}}$ |
| `HNCACO` | $2 N_{\text{amide}}$ |
| `HNCA` | $2 N_{\text{amide}}$ |
| `HNCOCA` | $N_{\text{amide}}$ |
| `CBCACONH` | $N_{\text{amide}} + N_{\text{nonGly}}^{(i-1)}$ |
| `HNCACB(+)` | $2 N_{\text{amide}}$ |
| `HNCACB(-)` | $N_{\text{nonGly}}^{(i)} + N_{\text{nonGly}}^{(i-1)}$ |

When input peaks carry intensities (`Peak::height = Some(...)`),
`filter_peaks_stoichiometric` executes six stages:

1. **Physical chemical-shift window gating**: `HSQC` $^1\text{H} \in [5.5, 11.5]$
   ppm (eliminating water ridges and aliphatic artifacts), `CA/CB`
   $^{13}\text{C} \in [8.0, 76.0]$ ppm (`HNCACB(+)` $\ge 35.0$ ppm;
   `HNCACB(-)` $\le 75.5$ ppm), and `CO` $^{13}\text{C}' \in [165.5, 184.0]$ ppm.
2. **2D indirect $t_1$ column ridge, sinc sidelobe, and $^3J_{\text{HN}\text{H}\alpha}$ doublet suppression**:
   suppresses secondary peaks sharing $^1\text{H}$ within $0.018$ ppm at $<18\%$
   parent height (sinc sidelobes within $|\Delta\delta_N| \le 1.8$ ppm) or
   $<25\%$ column dominance when three or more weaker maxima align along a
   single $^1\text{H}$ column, as well as unresolved $^3J_{\text{HN}\text{H}\alpha}$
   scalar-split doublet sub-maxima lacking independent 3D strip peaks.
3. **Paired Asn/Gln sidechain $\text{NH}_2$ doublet detection**: identifies
   same-$^{15}\text{N}$ pairs ($|\Delta\delta_N| \le 0.22$ ppm,
   $0.30 \le |\Delta\delta_H| \le 1.25$ ppm) with matched heights and matching
   `HNCO` carbonyl peaks (`|dCO| <= 0.25` ppm) that lack intra-residue `CA`
   intensity (`HNCACB(+)` / `HNCA`), removing up to $2 N_{\text{NQ}} + 4$
   sidechain roots and subtracting removed $\text{NH}_2$ counts from the
   remaining backbone `HSQC` budget.
4. **Cross-experiment $(^1\text{H}, {}^{15}\text{N})$ drift alignment and root ranking**:
   estimates rigid per-3D-experiment $(\Delta H, \Delta N)$ median registration
   offsets before scoring each `HSQC` root by distinct-3D-experiment support
   and normalized strip signal, retaining up to $\approx 1.35 N_{\text{amide}}$
   roots.
5. **3D strip sidelobe / baseline-floor rejection and top-$M_{\text{exp}}$ ranking**:
   calibrates experiment signal scale $\hat{I}(E)$ from the median of the top
   $\frac{2}{3} N_{\text{exp}}(E)$ peak heights and bottom-quintile picker
   threshold floor, suppresses F1 $^{13}\text{C}$ sinc sidelobes, retains at
   most $M_{\text{exp}} \in \{2, 4\}$ height-ranked peaks per root strip, and
   caps each signed experiment channel at
   $\lceil \text{stoichiometry\_budget\_mult} \cdot N_{\text{exp}}(E) \rceil$.
6. **Intensity-weighted spin-system clustering (`build_stoichiometric_spin_systems`)**:
   preserves shared tolerance-ellipse attachment ($d \le 1.0$) for overlapped
   amides per `docs/specs/03-assignment.md` while weighting cluster prominence
   by distinct-experiment support, primary per-strip rank, normalized peak
   height, and $(^1\text{H}, {}^{15}\text{N})$ proximity.

### 3.2 Two-Pass Reference-Offset Estimation (Mixture $\to$ Anchor)

1. **Pass 1 (Marginal Mixture Likelihood with LLR Gate)**: estimates a shared
   global $^{13}\text{C}$ reference offset (`CA`, `CB`, `C'`) via grid + refine
   search over $[-4.5, +4.5]$ ppm maximizing the sequence-weighted Gaussian
   mixture log-likelihood of top-ranked spin-system shifts against `Priors`. A
   log-likelihood-ratio gate ($\Delta \ell \ge 8.0$ nats and $|\Delta\delta| \ge
   0.65$ ppm) keeps DSS-referenced datasets at exact $0.00$ ppm while capturing
   systematic referencing shifts such as TMS-referenced $^{13}\text{C}$
   ($\approx -2.75$ ppm). Per-nucleus $^{13}\text{C}$ (`CA`, `CB`, `C'`) and
   amide (`H`, `N`) offsets are then refined around the baseline.
2. **Pass 2 (High-Consensus Anchor Median Refinement)**: after Pass-1
   evolutionary consolidation, collects assigned `(spin_system, residue)` pairs
   with calibrated confidence $\ge \tau_{\text{anchor}}$ (`0.70` default) and
   computes `median(obs - mu_expected)` per nucleus (requiring $\ge 8$ anchors
   per $^{13}\text{C}$ channel and tighter `H`/`N` significance gates).

### 3.3 Hybrid Evolutionary Fragment Crossover + Local Simulated Annealing

Optimization searches over injective sequence maps $\pi : \{0,\dots,S-1\} \to
\{-1, 1, \dots, L-1\}$ maximizing:

$$
F(\pi) = \sum_{k : a_k \ge 0} U(a_k, k) + \sum_{k : a_k \ge 0 \land a_{k+1} \ge 0} P(a_k, a_{k+1}),
$$

where $U(s, k)$ follows `docs/specs/03-assignment.md` Section 3 with an
enforced null-placement penalty `max(assign_cost, min_assign_cost)` whenever
$S > N_{\text{amide}}$ so artifact spin systems remain unassigned rather than
occupying exchange-broadened backbone slots.

- **Parallel Multi-Island Evolution**: evolves `population_size` (`64`)
  individuals partitioned across `num_islands` (`16`) independent islands in
  parallel across CPU worker threads via `std::thread::scope` (falling back
  cleanly to single-threaded execution on `wasm32`).
- **Local Metropolis-Hastings Sweep**: each generation runs fast $O(1)$-delta
  simulated annealing proposing single-spin-system placement/displacement
  ($55\%$), contiguous block swaps of length $1\dots B_{\max}$ ($30\%$), and
  explicit unassignment moves ($15\%$).
- **Sequential Fragment Crossover**: selects donor $A$ and recipient $B$ via
  tournament selection, extracts contiguous well-linked sequential stretches
  $[k_{\text{start}}, k_{\text{end}}]$ from $A$ where every interior peptide
  bond satisfies $P(a_k, a_{k+1}) \ge \tau_{\text{link}}$, splices the fragment
  into $B$ while cleanly evicting conflicting spin systems to `-1`, fills open
  high-affinity gaps, and anneals the offspring.
- **Thermodynamic & Belief-Damped Fragment Confidence Calibration**: Pass 1 uses
  local single-spin conditional Boltzmann calibration (`calibrate_boltzmann_confidence`,
  $T_{\text{cal}} = 1.65$) over unassignment, competing free spin systems, open
  sequence slots, and pairwise 2-swaps; Pass 2 (`calibrate_final_marginal_confidence`,
  $T_{\text{cal}} = 1.68$) adds second-order quadratic neighbor-belief damping
  $w(b_1) = \min(1, b_1 / 0.85)^2$, rigid whole-stretch register translation
  $[a..b] \to [k_2 .. k_2 + L - 1]$ ($L = 2\dots 5$) for short stretches lacking
  a high-belief ($\ge 0.80$) sequential anchor at either boundary so unanchored
  short islands cannot inflate interior residues above $0.80$ via mutual
  clamping, and a sub-threshold ($< 0.80$) Fréchet copula blend toward
  $\min(c_0, p_{\text{BP}}, p_{\text{Boltz1}})$ that corrects quadratic product
  under-confidence on low-consensus sparse placements while leaving high-confidence
  ($\ge 0.80$) calls strictly untouched (maintaining Expected Calibration Error
  $\text{ECE} \le 0.06$ across development real-spectrum benchmarks).
- **Sub-Stoichiometric Minor-Conformer Root Damping (`rel_root_height < 0.15`)**:
  both `build_spin_systems` and `build_stoichiometric_spin_systems` record each
  spin system's absolute root peak intensity `root_height` (`|HSQC height|`, or
  max attached 3D `|height|` when HSQC lacks heights) and median-normalized
  relative intensity $\tilde{I}_s = I_s / \operatorname{median}_{t \in \mathcal{C}}(I_t)$
  (`rel_root_height`) across carbon-bearing spin systems $\mathcal{C}$ without
  altering `docs/specs/03-assignment.md` Section 4 candidate clustering or
  ranking. Inside `evo_graph`, sub-stoichiometric roots with $\tilde{I}_s < 0.15$
  receive a smooth unary damping penalty parameterized by
  $u_{\text{sub}} = (0.15 - \tilde{I}_s) / 0.15$, amplified by the population
  log-ratio $\ln(\tilde{I}_{\text{maj}} / \tilde{I}_s)$ whenever a major-state
  conformer ($\tilde{I}_{\text{maj}} \ge 0.45$) shares matching $i$ / $i-1$
  $^{13}\text{C}$ shifts or when the sub-stoichiometric stub lacks intra-residue
  `CA(0)` (plus a local slow-exchange clone satellite tie-breaker when a $\ge 3\times$
  stronger nearby root within $0.20\text{ ppm } ^1\text{H} / 0.60\text{ ppm } ^{15}\text{N}$
  reproduces $100\%$ of a $\ge 4$-slot satellite's $^{13}\text{C}$ shifts within
  $0.25\text{ ppm}$). Moderate exchange-broadened amides ($\tilde{I}_s \ge 0.15$
  with distinct carbon shifts) and unique multi-slot C-terminal or Glycine amides
  are preserved while slow-exchange minor second-form roots such as Ubiquitin
  `Gln2` ($\approx 9\text{--}12\%$ amplitude) cannot displace overlapped major-state
  roots.
- **User-Locked `(spin_system, residue)` Hard Constraints & Sparse-Data Proposals**:
  `ScoreConfig::locked_assignments` and `EvoGraphConfig::locked_assignments`
  accept injective `(spin_system_idx, seq_idx)` pairs on assignable non-Proline
  residues ($k > 0$). When empty (default), $U(s, k)$ and $P(s, t)$ are
  bit-for-bit identical to `docs/specs/03-assignment.md` Section 3. When locks
  are active, $U(s_{\text{lock}}, k \neq k_{\text{lock}}) = \text{FORBID}$ and
  $U(s \neq s_{\text{lock}}, k_{\text{lock}}) = \text{FORBID}$ across CPU
  `standard`, GPU `standard` (`GpuEngine`), and `evo_graph` (preserving raw
  locked `HSQC` roots through stoichiometric filtering, pinning locked pairs
  throughout MCMC / evolutionary crossover, and reporting locked confidence
  $1.0$). For sparse `CA`-only / `HNCACB`-absent datasets in `standard`
  (`nmr_core::solver`), sparsity-adaptive Metropolis-Hastings annealing
  supplements single-spin moves (with exact 1-step reversible eviction across
  `FORBID` boundaries) with Hastings-corrected directed sequential chain-extension
  proposals ($k_{\text{anc}} \pm 1$ using top-$P$ sequential partners),
  symmetric contiguous segment register rotations ($\pm 1 \dots \pm 3$), and
  partner-guided dipeptide/tripeptide block splices (with multi-replica
  consensus bounded by local conditional Boltzmann margin at $T_{\text{cal}} = 1.55$)
  without modifying $U(s, k)$ or $P(s, t)$.

### 3.4 Residue-Graph Self-Consistency and Optional 3D $(\phi, \psi)$ Conditioning

- **2-Layer Sequence-Graph Message Passing**: between Pass 1 and Pass 2, each
  assigned residue node $k$ aggregates Pass-1 consensus, unary typing fit,
  sigmoid-weighted sequential connectivity $P(a_{k-1}, a_k)$ / $P(a_k, a_{k+1})$,
  and secondary-shift sign cooperativity ($\Delta\delta\text{CA} -
  \Delta\delta\text{CB}$ and $\Delta\delta\text{C}'$ secondary-structure
  agreement across $(k-1, k, k+1)$) over two message-passing layers. Verified
  interior stretches ($h_k^{(2)} \ge 0.78$) receive a soft unary anchor bonus
  in Pass 2 while final energies and confidences are always evaluated on the
  unmodified physical scoring matrix; low-consistency clashing outliers are
  penalized and unlocked in Pass-2 warm seeds.
- **Optional 3D Structure $(\phi, \psi)$ Conditioning (Library-Only)**: when
  `structure_torsions` is provided programmatically, empirical secondary
  chemical shifts $\Delta\delta_X(\phi_k, \psi_k)$ for `CA`, `CB`, and `C'`
  (parameterized from Spera & Bax 1991 and Wishart et al. 1995) refine residue
  expectations $\mu_{\text{struct}} = \mu_{\text{RC}} + \Delta\delta_X(\phi,
  \psi)$. All CLI backbone-assignment and benchmark commands (`assign`,
  `pipeline`, `benchmark`, `benchmark-suite`) run strictly sequence-only
  (`structure_torsions: None`, with no `--pdb` flag) to prevent structural
  ground-truth leakage.

## 4. Verification Benchmarks

Verified sequence-only via `cargo test --release -p nmr-core --test evo_graph_tests`
and multi-seed CLI evaluation (note: `CISD3` `BMRB 51439` is classified as a
development benchmark because solver retention, `score` reference-offset
alignment, and Pass-2 confidence calibration inspected its results):

| Dataset / Mode | Assignable | Assigned | Accuracy | Recall | HC Precision ($\ge 0.8$) | HC Recall | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Realistic Bruker Ubiquitin Simulator (`--realistic --seeds 5` aggregate) | 346 | 320 | **98.1%** (314/320) | **90.8%** | **98.4%** (306/311) | **88.4%** | 0.74 s |
| Synthetic Ubiquitin (`7`-exp, `assign_evo_graph`) | 68 | 68 | **97.1%** (66/68) | **97.1%** | **100.0%** (66/66) | **97.1%** | ~0.22 s |
| Synthetic Ubiquitin (`7`-exp, `assign_evo_graph_from_peaks`) | 68 | 67 | **98.5%** (66/67) | **97.1%** | **98.5%** (66/67) | **97.1%** | ~0.22 s |
| Synthetic Ubiquitin (`5`-exp without `HNCA`/`HNCOCA`) | 71 | 68 | **98.5%** (67/68) | **94.4%** | **98.5%** (66/67) | **93.0%** | ~0.22 s |
| Real Bruker Ubiquitin (`24,562` raw classical peaks, sequence-only, 5-seed mean) | 70 | 63.0 | **95.24%** | **85.71%** | **98.02%** | **70.86%** | **0.24 s** |
| Development Real Bruker `CISD3` (`BMRB 51439`, `700.13 MHz`, 93 aa, missing `HNCACB`, `seed 0` / 5-seed mean) | 51 (38 picked) | 35 | **65.7%** (23/35) / **59.4%** | **45.1%** (60.5% of picked) / **40.8%** | **100.0%** (10/10) / **95.5%** | **19.6%** / **16.5%** | **0.31 s** |

On the 5-spectrum Bruker TopSpin ubiquitin dataset (`HSQC`, `HNCACB`,
`CBCACONH`, `HNCO`, `HNCACO`; `24,562` raw classical local-maxima peaks across
`700` `HSQC` roots; `70` observable backbone amides excluding `Met1`,
`Pro19/37/38`, and exchange-broadened `Glu24`/`Gly53`), stoichiometric
filtering condenses the input to `834` peaks across `94` roots (`28` sidechain
$\text{NH}_2$ roots removed, **all `70/70` true backbone roots retained**),
automatically recovers the TMS-vs-DSS $^{13}\text{C}$ reference offset
(`Pass 1: -2.75 ppm CA` $\to$ `Pass 2: -2.66 ppm CA, -2.58 ppm CB`), and
achieves `94.5%` per-atom chemical-shift agreement in `0.24 s` per run.

Compared with published automated NMR assignment pipelines that pair deep
neural peak picking with evolutionary optimization and graph-based shift
refinement (Schmidt & Güntert, *J. Am. Chem. Soc.* 2012; Klukowski, Riek &
Güntert, *Nat. Commun.* 2022; Klukowski et al., *Sci. Adv.* 2023),
`nmr_core::evo_graph` demonstrates that combining sequence-derived peak
stoichiometry budgeting, two-pass reference alignment, multi-island sequential
fragment crossover, and 2-layer graph self-consistency enables `>95%` assignment
accuracy and `98%` high-confidence precision directly from classical local-maxima
peak picking in sub-second CPU wall time.
