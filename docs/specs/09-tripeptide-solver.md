# 09. Tripeptide Fragment & Cooperative Secondary-Shift Solver

Status: implemented in `crates/nmr-core/src/tripeptide.rs` (`TripeptideSolver`,
`TripeptideConfig`, `TripeptideFragmentDb`, `assign_tripeptide`).

## 1. Provenance and clean-room scope

This module is a clean-room implementation derived solely from peer-reviewed
literature and open BMRB data distributed in this repository:

1. Shen, Y. et al. Consistent blind protein structure generation from NMR
   chemical shift data. *Proc. Natl. Acad. Sci. USA* **2008**, 105(12),
   4685–4690 (chemical-shift $\chi^2$ + BLOSUM62 sequence-similarity fragment
   selection, Eqs. 1–3).
2. Shen, Y.; Bax, A. Protein backbone and sidechain torsion angles predicted
   from NMR chemical shifts using artificial neural networks. *J. Biomol. NMR*
   **2013**, 56, 227–241 (tripeptide `(i-1, i, i+1)` secondary chemical shifts
   and sequence-context conformational propensities).
3. Wishart, D. S.; Bigam, C. G.; Holm, A.; Hodges, R. S.; Sykes, B. D.
   $^1\text{H}$, $^{13}\text{C}$ and $^{15}\text{N}$ random coil NMR chemical
   shifts of the common amino acids. I. Investigations of nearest-neighbor
   effects. *J. Biomol. NMR* **1995**, 5, 67–81; and Schwarzinger, S. et al.
   Sequence-dependent correction of random coil NMR chemical shifts.
   *J. Am. Chem. Soc.* **2001**, 123, 2970–2978 (`i±1` sequence-neighbor
   random-coil substituent corrections, notably pre-Proline and aromatic or
   $\beta$-branched flanking residues).
4. Henikoff, S.; Henikoff, J. G. Amino acid substitution matrices from protein
   blocks. *Proc. Natl. Acad. Sci. USA* **1992**, 89(22), 10915–10919
   (symmetric BLOSUM62 log-odds substitution matrix).
5. Wang, L.; Eghbalnia, H. R.; Bahrami, A.; Markley, J. L. Linear analysis of
   carbon-13 chemical shift differences and its application to the detection
   and correction of errors in referencing and spin system identifications.
   *J. Biomol. NMR* **2005**, 32, 13–22 (global $^{13}\text{C}$ chemical-shift
   reference offset estimation prior to scoring).
6. Public BMRB entries in `data/benchmark/*.str` and residue marginal priors in
   `data/priors/bmrb_priors.json`.

Compared with published chemical-shift fragment selection methods for 3D
structure generation (Shen et al., 2008) that assume already-assigned shifts,
this solver embeds contiguous tripeptide `(k-1, k, k+1)` secondary-shift
cooperativity directly inside the **unassigned sequential spin-system
matching objective**.

## 2. Why contiguous tripeptides beat independent single-residue priors

In the standard solver (`docs/specs/03-assignment.md`), `U(s, k)` evaluates
independent single-residue Gaussians for `k` and `k-1`, while `P(s, t)` depends
only on `(s, t)` carbon shift differences without knowing *where* along the
primary sequence `(s, t)` is placed. Three failure modes arise:

1. **Sequence-neighbor chemical shift perturbations**: A residue preceding
   Proline experiences $\approx -1.95\text{ ppm}$ $\text{C}^\alpha$ and
   $\approx -2.00\text{ ppm}$ $\text{C}'$ substituent shifts (Wishart et al.,
   1995), while a residue following a $\beta$-branched (`ILE`, `VAL`, `THR`) or
   aromatic (`PHE`, `TYR`, `TRP`) neighbor experiences $+2.0$ to $+2.4\text{ ppm}$
   amide $^{15}\text{N}$ downfield shifts. Single-residue priors treat these
   predictable sequence-context shifts as noise or misfit; neighbor-corrected
   tripeptide scoring uses them to resolve tightly overlapped 2D HSQC roots
   (such as `ILE30` vs `PHE45` in ubiquitin).
2. **Cooperative secondary-structure continuity across `(k-1, k, k+1)`**:
   Placing spin system `s` at `k` and `t` at `k+1` jointly observes chemical
   shifts across the **entire tripeptide `(k-1, k, k+1)`**:
   - `s(i-1)` observes `(CA, CB, C)` of residue `k-1`,
   - `s(i)` and `t(i-1)` jointly observe consensus `(CA, CB, C)` of residue `k`
     plus `(N, H)` from `s`,
   - `t(i)` and `t`'s root observe `(CA, CB, C, N, H)` of residue `k+1`.
   Real $\alpha$-helices ($\Delta\delta\text{CA} > 0, \Delta\delta\text{CB} < 0,
   \Delta\delta\text{C}' > 0$), $\beta$-strands ($\Delta\delta\text{CA} < 0,
   \Delta\delta\text{CB} > 0, \Delta\delta\text{C}' < 0$), PPII/coil segments,
   and turns exhibit strong multi-atom, multi-residue correlation across
   `(k-1, k, k+1)`, whereas false sequential links splice incompatible
   secondary-shift regimes.
3. **Spectrometer chemical-shift referencing offsets**: Experimental Bruker
   triple-resonance datasets referenced indirectly to TMS rather than DSS carry
   a systematic $\approx -2.7\text{ ppm}$ $^{13}\text{C}$ shift across `CA`,
   `CB`, and `C'`, pushing `ALA CB`, `GLY CA`, and `SER/THR CB` ~2–3$\sigma$
   off DSS-referenced BMRB priors unless aligned prior to typing.

## 3. Neighbor-corrected secondary shifts & automatic reference alignment

### 3.1 Automatic `(dCA, dCB, dC, dH, dN)` offset estimation

Before constructing scoring tables, `estimate_reference_offsets` ranks spin
systems by distinct carbon-slot coverage and cross-experiment support (ignoring
1-slot spurious artifacts when multi-slot spin systems exist) and maximizes the
robust outlier-floored marginal sequence-mixture log-likelihood:

$$\mathcal{L}(\Delta) = \sum_{n=1}^{N_X} \ln \max\!\left(\epsilon,\; \frac{1}{L}\sum_{k=1}^{L} \mathcal{N}\!\left(v_n - \Delta \;\middle|\; \mu_{\text{BMRB}}(\text{aa}_k, X) + \tilde{\delta}_{\text{nb}}(k, X),\; \sigma_{k,X}^2\right)\right)$$

Because $\alpha$-helical and $\beta$-sheet secondary shifts have **opposite
signs** on $\text{C}^\alpha$ and $\text{C}^\beta$ while sharing a single
aliphatic $^{13}\text{C}$ indirect axis in `HNCACB`/`CBCA(CO)NH`, the joint
aliphatic objective $\mathcal{L}_{\text{CA}}(\Delta_{\text{aliph}}) + 0.35\,\mathcal{L}_{\text{CB}}(\Delta_{\text{aliph}})$
cancels secondary-structure composition bias. A deadband (`|d_aliph| < 1.15 ppm -> 0.0`)
keeps DSS-referenced datasets strictly unshifted while recovering $-2.75\text{ ppm}$
TMS offsets within $< 0.1\text{ ppm}$. Crucially, offsets shift the scoring
priors internally without mutating spin-system candidate ppm values, so
`to_reference` always emits shifts in the experimental spectrum's ppm frame.

### 3.2 Sequence-neighbor-corrected secondary shifts

For residue `k` with flanking sequence `(\text{aa}_{k-1}, \text{aa}_k, \text{aa}_{k+1})`
and atom $X \in \{\text{CA}, \text{CB}, \text{C}, \text{N}, \text{H}\}$:

$$\Delta\delta_{k,X} = (\delta_{k,X}^{\text{obs}} - \Delta_{\text{ref},X}) - \bigl(\mu_{\text{RC}}(\text{aa}_k, X) + \delta_{\text{prev}}(\text{aa}_{k-1}, X) + \delta_{\text{next}}(\text{aa}_{k+1}, X)\bigr)$$

using Wishart et al. (1995) `GGXGG` random-coil baselines (with oxidized vs
reduced `CYS` inferred from $\text{C}^\beta \gtrless 35\text{ ppm}$) and
Wishart (1995) / Schwarzinger (2001) `i±1` substituent tables.

## 4. Tripeptide fragment library (`TripeptideFragmentDb`)

`TripeptideFragmentDb` combines two complementary sources:

1. **Standalone WASM-compatible 4-state conformational basin & transition
   templates (`TripeptideFragmentDb::embedded`)**: Eight 3-residue conformational
   state trajectories across `{Helix, Sheet, Ppii, Turn}` weighted by
   amino-acid basin propensities $P(\text{basin} \mid \text{aa}_{k+m})$ plus
   20 canonical sequence-diverse secondary-structure and cap/turn motifs,
   requiring zero filesystem access on `wasm32-unknown-unknown`.
2. **Empirical BMRB tripeptides (`harvest_star_paths` / `default_cached`)**:
   Contiguous 3-residue secondary-shift windows harvested at initialization from
   `data/benchmark/*.str` (with automatic $^{13}\text{C}$ reference sanity
   check). To guarantee **zero benchmark data leakage**, `non_homologous_mask`
   excludes any source protein sharing an exact 7-mer peptide or $\ge 35\%$
   sliding-window sequence identity with the target sequence (`exclude_homologs = true`
   by default).

For each target position `k`, `build_window_models` selects the 8 propensity-weighted
basin transition modes together with the top BLOSUM62 sequence-matched empirical
fragments (requiring strict `GLY`/`PRO` compatibility at central and flanking
sites, Shen & Bax 2013), forming a normalized multi-modal Gaussian mixture
$P_{\text{coop}}(\Delta\boldsymbol{\delta}_{k-1:k+1} \mid k)$ alongside its
exact single-atom uncorrelated marginal $P_{\text{ind}}(\Delta\boldsymbol{\delta}_{k-1:k+1} \mid k)$.
The cooperative score is a bounded log-Bayes-factor:

$$B_{\text{coop}}(y; k) = \ln\!\left((1 - \lambda) + \lambda\,\frac{P_{\text{coop}}(y \mid k)}{P_{\text{ind}}(y \mid k)}\right), \qquad \lambda = \text{coop\_mix} \in [0, 0.95)$$

which rewards multi-residue secondary-shift coherence and penalizes physically
contradictory flips while remaining bounded below by $\ln(1 - \lambda)$ on
unstructured or synthetic i.i.d. shifts.

## 5. Position-specific scoring and parallel annealing

### 5.1 Unary score `U_tri(s, k)`

`U_tri(s, k)` combines:
- A neighbor-adjusted prior mixture $(1 - w_{\text{nb}})\,\mathcal{N}(\mu, \sigma^2) + w_{\text{nb}}\,\mathcal{N}(\mu + \tilde{\delta}_{\text{nb}}, \sigma^2)$
  against residue-averaged background across `(k, 0)` and `(k-1, -1)` carbon
  candidates plus `(N, H)` at `k`,
- Candidate rank tie-breaking (`cand_rank_penalty`) disabled automatically on
  2D-overlapped HSQC root pairs so noisy 2D `(H, N)` jitter never overrides
  sequence-context evidence,
- Intra-spin-system `(k-1, k)` dipeptide secondary-shift coherence
  $w_{\text{dip}}\,B_{\text{coop}}(\Delta\delta_{k-1}, \Delta\delta_k; k)$,
- Automatic spurious-root screening when observed carbon-bearing roots exceed
  $2 N_{\text{assignable}} + 20$.

### 5.2 Position-specific tripeptide pair tensor `P_tri(s, t, k)`

Unlike the position-independent 2D `P(s, t)` matrix, `TripeptidePairTensor`
stores a compact `f32` 3D table `P_tri(s, t, k)` over active spin systems for
each sequence edge `(k, k+1)`:

$$P_{\text{tri}}(s, t, k) = \sum_{X \in \{\text{CA}, \text{CB}, \text{C}\}} S_{\text{seq}}(s, t, k, X) \;+\; w_{\text{tri}}\,B_{\text{coop}}(\Delta\delta_{k-1}, \Delta\delta_k^{\text{cons}}, \Delta\delta_{k+1}; k) \;-\; w_{\text{dip}}\,B_{\text{coop}}(\Delta\delta_{k-1}, \Delta\delta_k; k)$$

where:
- $S_{\text{seq}}$ couples sequential carbon matching (`s(i) == t(i-1)`) with
  gentle candidate typing consistency so impossible two-candidate chimeras are
  disfavored,
- When `HNCA` is absent (`5-spectrum` regime) and `degenerate=false` suppresses
  a coincident `CA(i) == CA(i-1)` peak into `("CA", -1)` with `support >= 2`,
  `allow_soft_degenerate` evaluates the coincident sequential match hypothesis
  (requiring `support >= 3` when dual `HNCA + HNCACB` intra-residue support is
  detected so missing `CA(i)` peaks in 7-spectrum datasets are never confused
  with degenerate shifts),
- Consensus $\Delta\delta_k^{\text{cons}}$ uses `0.5 * (s(i) + t(i-1))` when
  matched, or falls back to `t(i-1)` (e.g., `HNCO` $\text{C}'_k$ when `HNCACO`
  misses $\text{C}'(i)$) so tripeptide scoring remains active across 5-spectrum
  links.

### 5.3 Parallel Metropolis simulated annealing (`assign_tripeptide`)

`assign_tripeptide(systems, sequence, priors, cfg) -> AssignResult` executes
`cfg.anneal.runs` independent replicas in parallel across `std::thread::scope`
using strictly symmetric Metropolis proposals (`65%` single/displace-swap,
`30%` block-swap, `5%` unassign) over `U_tri + P_tri`. Preserving exact
detailed balance down to $T_{\text{end}} = 0.05$ ensures replica agreement
`confidence[s]` remains a well-calibrated posterior probability on intrinsically
disordered proteins and degenerate repeats, returning the exact `AssignResult`
schema (`position`, `confidence`, `score`, `run_scores`) used throughout
`nmr-core`, `nmr-cli`, and `nmr-gui`.
