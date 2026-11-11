# 01. Data I/O and formats

## Shift tables: NMR-STAR
- Reader: `python/nmrassign/nmrstar.py` (own parser; v3 and v2.1).
- Handles quoted and semicolon values, multiple entities (picks the one with
  most shifts unless `entity` is given), `HN` -> `H`, `CO` -> `C`.
- Writer: `_Atom_chem_shift` loop with per-residue confidence in `Details`.

## Peak lists: TSV
```
experiment  H      N       C       sign
HSQC        8.312  121.45
HNCACB      8.311  121.44  30.05   -1
```
Experiments: HSQC, HNCA, HNCOCA, HNCACB, CBCACONH, HNCO, HNCACO.
Planned extension (v2): `height`, `volume`, `linewidth_*` columns for
intensity-aware scoring; extra dimensions for NOESY (`H2`, `C2`).

## Spectra (TODO)
- Read Bruker processed data (pdata) and NMRPipe files into an array plus
  per-axis ppm scale, nucleus, and referencing.
- Candidates: nmrglue (BSD-3) in Python; NMRTools.jl (MIT) in Julia.
- Acceptance: round-trip a known spectrum and reproduce peak positions
  within one digital point.

## Planned
- NEF reader and writer (shifts, peaks, restraints) for interoperability.
- Sparky and CcpNmr peak list importers (format conversion only).
