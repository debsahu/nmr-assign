# nmr-assign: rules for AI coding agents

Open source automated protein NMR backbone resonance assignment.
Working name; rename freely before first release.

## Scope (v1)
Peak lists in, backbone chemical shifts out (H, N, CA, CB, C).
Supported experiments: HSQC, HNCA, HN(CO)CA, HNCACB, CBCA(CO)NH, HNCO, HN(CA)CO.
Out of scope for v1: peak picking from spectra, sidechains, NOESY, structure.

## Architecture
- `python/nmrassign/`: reference implementation and test oracle. Every
  algorithmic change lands here first, with a passing benchmark.
- `julia/`: performance core (planned). It must reproduce the Python
  reference on the benchmark suite before it replaces any Python module.
- Interchange: NMR-STAR v3 for shifts, the TSV peak format in docs/specs/03-assignment.md
  for peaks. NEF support is a v2 goal.

## Clean-room rules (non-negotiable)
Permitted sources:
- Peer-reviewed papers, their supplementary information, and public method
  descriptions.
- Open data: BMRB, PDB, PDB Chemical Component Dictionary.
- Code under the licence allowlist below.

Forbidden:
- Source, binaries, decompiled output, residue libraries, macros, parameter
  files, or output files from CYANA, FLYA, ARTINA, NMRtist, Xplor-NIH, ARIA,
  CcpNmr, or any tool obtained under an academic or commercial licence.
- Reading the source of GPL or AGPL tools (DEEP Picker, COMPASS and others)
  while implementing the matching module. Work from their papers instead.
- Using those product names in module, package, or file names. Compatibility
  claims in docs must be true and phrased as comparisons.

Provenance: every algorithm module starts with a docstring citing the paper
and section it implements, or stating that the method is original.

## Licence allowlist
- Allowed: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC.
- Ask the maintainer first: LGPL, MPL-2.0, EPL, anything else.
- Not allowed: GPL, AGPL, non-commercial licences, unlicensed code.
List every dependency and its licence in THIRD_PARTY.md.

## Data rules
- The maintainer's unpublished assignments never enter the public repo.
  Keep them in `data/private/` (git-ignored).
- Test fixtures are synthetic or derived from public BMRB entries, with the
  BMRB ID recorded in the fixture header.

## Definition of done for any solver change
`python -m nmrassign.cli benchmark` on the synthetic suite must not regress
accuracy or high-confidence precision. Report both numbers in the PR.

## Scientific ownership
The maintainer (an NMR scientist) owns docs/specs/03-assignment.md sections 3 and 4
(scoring and experiment rules). Agents implement the spec; they do not change
scoring semantics without the maintainer's approval.
