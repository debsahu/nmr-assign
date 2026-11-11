# 03. Backbone assignment (v0.1 working)

Status: draft for maintainer review. Sections 3 and 4 are owned by the
maintainer; numbers marked PROVISIONAL are placeholders.

## 1. Inputs and outputs

Inputs
- Sequence: one-letter string, FASTA, or taken from an NMR-STAR file.
- Peak lists: TSV, one peak per row:

  ```
  experiment  H      N       C       sign
  HSQC        8.312  121.45
  HNCA        8.309  121.47  56.21   1
  HNCACB      8.311  121.44  30.05   -1
  ```

  `experiment` is one of HSQC, HNCA, HNCOCA, HNCACB, CBCACONH, HNCO, HNCACO.
  `C` is blank for HSQC. `sign` is the peak phase (required for HNCACB,
  where CA is positive and CB negative; optional elsewhere).

Outputs
- Per residue: H, N, CA, CB, C shifts with a confidence in [0, 1].
- TSV and an NMR-STAR v3 `_Atom_chem_shift` loop.

## 2. Pipeline

1. Spin system building: each HSQC peak is a root. Every other peak attaches
   to all roots within the H/N tolerance ellipse (`d <= 1`), so tightly
   overlapped HSQC roots both retain their candidate carbon peaks.
2. Offset resolution: within a root, carbon peaks are labelled as residue i or
   i-1 by matching against experiments that report only i-1.
3. Scoring: unary typing score per (spin system, residue) and pairwise
   connectivity score per (spin system, spin system).
4. Optimization: simulated annealing over the injective map
   spin system -> residue (with an unassigned state), repeated R times.
5. Consensus: confidence = fraction of runs agreeing with the best run.

## 3. Scoring (maintainer owned)

Typing score of spin system s at residue k, summed over observed atoms:

    U(s,k) = sum_X LLR(type(k),   X, shift_X(i))
           + sum_X LLR(type(k-1), X, shift_X(i-1))
           + LLR(type(k), N, root_N) + LLR(type(k), H, root_H)
           - assign_cost

    LLR(t, X, v) = max( log N(v | mu_tX, sd_tX) - log B_X(v), floor )

where B_X is the residue-averaged background density, so an atom that does
not discriminate contributes about zero. floor = -12 (PROVISIONAL).
`assign_cost` = 0.0 (PROVISIONAL, configurable via `--assign-cost`) is an
optional per-placement threshold relative to leaving a spin system unassigned.
A CB observed for a glycine position contributes log(0.02) (PROVISIONAL).
Residue k is forbidden if it is proline or the N-terminal residue. Spin
systems with zero observed carbon candidates (for example Asn/Gln sidechain
NH2 roots or 2D HSQC noise peaks) are forbidden from all backbone positions.

Connectivity score for s at k and t at k+1:

    P(s,t) = sum over X in {CA, CB, C} observed in both s(i) and t(i-1):
             max( log N(d | 0, sigma_X) + log R_X, -8 )

d = shift_X(i) of s minus shift_X(i-1) of t; R_X is the plausible shift range.
sigma: CA 0.12, CB 0.20, C 0.12 ppm (PROVISIONAL, tune to digital
resolution). R: 30 ppm (PROVISIONAL).

Total score = sum of U over assigned spin systems + sum of P over adjacent
assigned pairs. Unassigned spin systems contribute 0.

Priors: mu and sd per residue type and atom. v1 ships PROVISIONAL values;
`cargo run --release -p nmr-cli -- priors` recomputes them from a directory of
NMR-STAR files (BMRB) and should replace them before release.

## 4. Experiment rules (maintainer owned)

| Experiment | Peaks per root                 | Offset      |
|------------|--------------------------------|-------------|
| HNCA       | CA                             | i and i-1   |
| HNCOCA     | CA                             | i-1 only    |
| HNCACB     | CA (+), CB (-)                 | i and i-1   |
| CBCACONH   | CA, CB                         | i-1 only    |
| HNCO       | C                              | i-1 only    |
| HNCACO     | C                              | i and i-1   |

CBCACONH peaks take their atom type from the HNCACB peak they match.
A carbon in an "i and i-1" experiment that matches an i-1 reference within
tolerance (CA 0.25, CB 0.35, C 0.25 ppm, PROVISIONAL) is i-1; otherwise i.

Candidates. Peaks are never discarded by the builder. Per (atom, offset)
the builder clusters values within tolerance and keeps up to 3 clusters as
competing candidates (value, support), ranked by descending support and then
closest mean H/N distance to the root. Typing uses the best candidate minus
log(n candidates) plus 0.7 x (support - 1); connectivity uses the best
candidate pair minus log(n_s x n_t). A carbon from an "i and i-1"
experiment with no i-1 reference at all is a candidate for both offsets.
Measured effect: HNCACB/CBCA(CO)NH-only accuracy rose from 88.6% to 98.5%.

Degenerate i / i-1 shifts (option `degenerate=True`, default off). When an
"i and i-1" experiment shows only the peak matching i-1, the value is also
offered as residue i, with support 0 (connectivity only, never typing).
Measured over three seeds it lowered recall and high-confidence recall, so it
is disabled until peak intensities can discriminate a missing peak from
coincident peaks.

Open questions for the maintainer
- Intensity ratio (i stronger than i-1 in HNCA) as a tiebreaker when no
  i-1-only experiment exists? Also the key to enabling degenerate handling.
- Secondary-structure-dependent priors (helix vs strand CA/CB offsets)?
- Handling of Asn/Gln sidechain NH2 roots: 0-carbon roots are excluded by
  default; also detect paired H doublets at shared N to reject NH2 roots that
  pick up a spurious 3D peak?
- Deuteration isotope shift corrections for CA/CB?

## 5. Evaluation

Against a reference assignment (for example a deposited BMRB entry):
- Accuracy: correct / assigned.
- Recall: correct / assignable spin systems.
- High-confidence precision and recall (confidence >= 0.8).
`benchmark` simulates peak lists from a reference NMR-STAR file, runs the
pipeline blind, and reports these numbers.
