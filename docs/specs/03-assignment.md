# 03. Backbone assignment (v0.2 working)

Status: draft for maintainer review. Sections 3 and 4 are owned by the
maintainer; numbers marked PROVISIONAL are placeholders.

## 1. Inputs and outputs

Inputs
- Sequence: one-letter string, FASTA, or taken from an NMR-STAR file.
- Peak lists: TSV, one peak per row:

  ```
  experiment  H      N       C       sign  height   volume
  HSQC        8.312  121.45          1     540000   2100000
  HNCA        8.309  121.47  56.21   1     320000   1600000
  HNCACB      8.311  121.44  30.05   -1    210000   1100000
  ```

  `experiment` is one of HSQC, HNCA, HNCOCA, HNCACB, CBCACONH, HNCO, HNCACO.
  `C` is blank for HSQC. `sign` is the peak phase (required for HNCACB,
  where CA is positive and CB negative; optional elsewhere). `height` and
  `volume` are optional (5-column TSVs without intensities remain supported).

Outputs
- Per residue: H, N, CA, CB, C shifts with a confidence in [0, 1] (reported in
  the experimental spectrum's chemical-shift referencing frame).
- TSV and an NMR-STAR v3 `_Atom_chem_shift` loop.

## 2. Pipeline

1. Spin system building: each HSQC peak is a root. Every other peak attaches
   to all roots within the H/N tolerance ellipse (`d <= 1`), so tightly
   overlapped HSQC roots both retain their candidate carbon peaks.
2. Offset resolution: within a root, carbon peaks are labelled as residue i or
   i-1 by matching against experiments that report only i-1 (or via `1J`/`2J`
   intensity-ratio tie-breaking when no i-1-only experiment is present).
3. Scoring: automatic marginal reference offset alignment on `Priors`, unary
   typing score per (spin system, residue), and pairwise connectivity score per
   (spin system, spin system).
4. Optimization: simulated annealing over the injective map
   spin system -> residue (with an unassigned state), repeated R times.
5. Consensus: confidence = fraction of runs agreeing with the best run.

## 3. Scoring (maintainer owned)

Automatic chemical-shift reference alignment (`auto_reference = true` by
default in `ScoreConfig`): before computing `U(s, k)`,
`estimate_reference_offsets(systems, sequence, priors)` estimates global
per-nucleus offsets `(dH, dN, dCA, dCB, dC)` (`experimental_ppm - dss_prior_ppm`)
maximizing the outlier-robust marginal mixture log-likelihood of observed
multi-slot spin-system shifts against sequence-weighted `Priors`, with a joint
aliphatic carrier tether `dCA ≈ dCB` (which cancels opposite-sign secondary
structure shifts while capturing TMS vs DSS or carrier offsets) and a deadband +
log-likelihood-ratio gate so DSS-referenced spectra remain at exact `0.0 ppm`.
The offset is applied via `priors.with_offsets(&offsets)` (`prepare_scoring_priors`)
inside unary scoring only; `SpinSystem` candidates and output chemical shifts stay
in the experimental spectrum's ppm frame.

Typing score of spin system s at residue k, summed over observed atoms:

    U(s,k) = sum_X LLR(k,   X, shift_X(i))
           + sum_X LLR(k-1, X, shift_X(i-1))
           + LLR(k, N, root_N) + LLR(k, H, root_H)
           - assign_cost

    LLR(j, X, v) = max( log P_cov(v | j, X) - log B_X(v), floor )

    P_cov(v | j, X) =
      0.85 * N(v | mu_non(j,X) + d_X + Delta(j,X), sd_{type(j),X})
    + 0.15 * N(v | mu_non(j,X) + d_X,              sd_{type(j),X})   if seq[j+1] == PRO (j < K-1)
      N(v | mu_non(j,X) + d_X, sd_{type(j),X})                       if seq[j+1] != PRO (j < K-1)
      N(v | mu_{type(j),X} + d_X, sd_{type(j),X})                    otherwise

where `B_X(v)` is the sequence-slot-independent residue-averaged background
density (so `log B_X(v)` cancels identically when comparing competing sequence
positions for spin system `s`, and an atom that does not discriminate
contributes about zero), `d_X` is the global chemical-shift reference offset,
`mu_non(j, X) = mu_{type(j), X} - 0.85 * p_Pro * Delta(j, X)` removes the
`p_Pro = 91 / 2508 ≈ 0.0363` pre-Proline background already absorbed into
`bmrb_priors.json`, and `Delta(j, X)` is the deterministic primary-sequence
covalent pre-Proline shift (`delta_covalent(seq, j)`) derived strictly from
peer-reviewed random-coil host-guest peptide and proline conformational studies
(Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81; Wishart et al., *J. Biomol.
NMR* 1995, 6, 135–140; Schwarzinger et al., *J. Am. Chem. Soc.* 2001, 123,
2970–2978; MacArthur & Thornton, *J. Mol. Biol.* 1991, 218, 397–412):
- **Pre-Proline (`seq[j+1] == "PRO"`, `j < K - 1`) vs non-pre-Proline**: steric
  and tertiary-imide electronic upfield shifts `Delta = (-1.95, -1.80) ppm` on
  `(CA, C)` for standard residues (`(-1.00, -1.50) ppm` for `GLY`,
  `(-1.75, -1.70) ppm` for `PRO`), evaluated via the `0.85 / 0.15`
  coil/PPII-vs-helical-kink mixture `P_cov` (MacArthur & Thornton, 1991) and
  centered around `p_Pro = 91 / 2508` (`d_pre = 0.85 * (1 - p_Pro) * Delta`,
  `d_non = -0.85 * p_Pro * Delta`) so `p_Pro * d_pre + (1 - p_Pro) * d_non == 0`
  identically across natural sequences, coil/turn pre-Proline residues gain
  strong upfield selectivity, and kinked alpha-helical pre-Proline residues have
  bounded penalty (`<= -ln(0.15)` nats). Expected mixture shifts `delta_covalent`
  are included in sequence-mixture and anchor reference-offset estimation.
- **`CB` & `15N` (`d_cov(j, CB) = 0.0`, `d_cov(j, N) = 0.0`)**: unfolded
  pentapeptide `CB` pre-Proline (`+0.25 ppm`) and `i-1` `15N` substituent
  offsets (Braun, Wider & Wüthrich, *J. Am. Chem. Soc.* 1994, 116, 8466–8469;
  Schwarzinger et al., 2001) were empirically ablated and excluded (`0.0 ppm`)
  because folded-protein `(phi, psi)` and hydrogen-bond dispersion dominate and
  invert those random-coil shifts.

floor = -12 (PROVISIONAL).
`assign_cost` = 0.0 (PROVISIONAL, configurable via `--assign-cost`) is an
optional per-placement threshold relative to leaving a spin system unassigned.
A CB observed for a glycine position contributes log(0.02) (PROVISIONAL).
Residue k is forbidden if it is proline or the N-terminal residue. Spin
systems with zero observed carbon candidates (for example Asn/Gln sidechain
NH2 roots or 2D HSQC noise peaks) are forbidden from all backbone positions.
When `systems.len() > assignable` (`forbid_ca_less_excess_roots = true`),
spin systems with zero observed `CA(i)` and `CA(i-1)` candidates are forbidden
from non-Gly / non-post-Pro sequence slots, and sparse uncorroborated 1-peak
roots receive an excess-root placement penalty (`4.5` nats) so noise roots do
not hijack exchange-broadened sequence slots.

Connectivity score for s at k and t at k+1:

    P(s,t) = sum over X in {CA, CB, C} observed in both s(i) and t(i-1):
             max( log N(d | 0, sigma_X) + log R_X, -8 )

d = shift_X(i) of s minus shift_X(i-1) of t; R_X is the plausible shift range.
sigma: CA 0.12, CB 0.20, C 0.12 ppm (PROVISIONAL, tune to digital
resolution). R: 30 ppm (PROVISIONAL).

Total score = sum of U over assigned spin systems + sum of P over adjacent
assigned pairs. Unassigned spin systems contribute 0.

Priors: empirical `(mu, sd)` per residue type and atom shipped in
`data/priors/bmrb_priors.json` (trained on 21 DSS-referenced BMRB entries
strictly disjoint from `data/benchmark/manifest.tsv`).
`cargo run --release -p nmr-cli -- priors` recomputes them from a directory of
NMR-STAR files (BMRB).

## 4. Experiment rules (maintainer owned)

| Experiment | Peaks per root | Top-`M_exp` cap | Offset      |
|------------|----------------|-----------------|-------------|
| HNCA       | CA             | 4               | i and i-1   |
| HNCOCA     | CA             | 2               | i-1 only    |
| HNCACB     | CA (+), CB (-) | 4 (+), 4 (-)    | i and i-1   |
| CBCACONH   | CA, CB         | 4               | i-1 only    |
| HNCO       | C              | 2               | i-1 only    |
| HNCACO     | C              | 4               | i and i-1   |

CBCACONH peaks take their atom type from the HNCACB peak they match.
A carbon in an "i and i-1" experiment that matches an i-1 reference within
tolerance (CA 0.25, CB 0.35, C 0.25 ppm, PROVISIONAL) is i-1; otherwise i.
When no i-1-only reference exists for an atom, two well-centered peaks
(`hn_dist <= 0.72`) in `HNCA`, `HNCACB`, or `HNCACO` with height ratio `>= 1.60`
resolve the stronger (`1J`) peak as `i` and the weaker (`2J`) peak as `i-1`.

Candidates and distinct-experiment support. When peak heights are present,
attached peaks per `(root, experiment, sign_class)` are capped to the top
`M_exp` peaks ranked by `norm_height * exp(-0.5 * hn_dist^2)`, where
`norm_height = height / median_height(experiment, sign_class)`. Per
`(atom, offset)` the builder clusters values within tolerance and defines
cluster `support` as the count of **distinct experiments** in the cluster
(preventing multiple sinc ripples from a single experiment from inflating
`support`). Up to `max_cands` (default 3) clusters are kept as competing
candidates `(value, support)`, ranked by `(distinct_support DESC,
max_norm_height DESC, mean_hn_dist ASC)` when heights exist, or
`(distinct_support DESC, mean_hn_dist ASC)` when heights are absent. Typing
uses the best candidate minus `log(n candidates)` plus `0.7 * (support - 1)`;
connectivity uses the best candidate pair minus `log(n_s * n_t)`. A carbon from
an "i and i-1" experiment with no i-1 reference at all is a candidate for both
offsets.

Asn/Gln sidechain `NH2` doublet rejection. Paired HSQC roots sharing a common
`15N` shift (`|dN| <= 0.12 ppm`) with `0.30 <= |dH| <= 1.20 ppm` that both lack
credible intra-residue `CA(i)` support have their candidate lists cleared so
spurious 3D noise attachments on `NH2` prongs cannot enter backbone assignment.

Degenerate `i == i-1` shifts (automatic when peak heights exist; optional
`degenerate=true` otherwise). When an "i and i-1" experiment (`HNCA`, `HNCACB`,
`HNCACO`) shows only the peak matching `i-1` and its summed amplitude at that
shift is `>= 1.65 * median_height(experiment, sign_class)`, the value is also
offered as residue `i` with `support = 0` (connectivity only, never typing),
discriminating coincident `i + i-1` peaks from missing `i` peaks.

Open questions for the maintainer
- Secondary-structure-dependent priors (helix vs strand CA/CB offsets)?
- Deuteration isotope shift corrections for CA/CB?

## 5. Evaluation

Against a reference assignment (for example a deposited BMRB entry):
- Accuracy: correct / assigned.
- Recall: correct / assignable spin systems.
- High-confidence precision and recall (confidence >= 0.8).
`benchmark` simulates peak lists from a reference NMR-STAR file, runs the
pipeline blind, and reports these numbers.
