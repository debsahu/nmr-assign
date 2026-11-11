# 01. Data I/O and formats

## Shift tables: NMR-STAR
- Reader/writer: `crates/nmr-core/src/nmrstar.rs` (clean-room parser; v3 and v2.1).
- Handles quoted and semicolon values, multiple entities (picks the one with
  most shifts unless `entity` is given), `HN -> H`, `CO -> C`.
- Writer: `_Atom_chem_shift` loop with per-residue confidence in `Details`.

## Peak lists: TSV (`crates/nmr-core/src/peaks.rs`)
```
experiment  H      N       C       sign
HSQC        8.312  121.45
HNCACB      8.311  121.44  30.05   -1
```
Experiments: HSQC, HNCA, HNCOCA, HNCACB, CBCACONH, HNCO, HNCACO.
Planned extension: `height`, `volume`, `linewidth_*` columns for
intensity-aware scoring; extra dimensions for NOESY (`H2`, `C2`).

## Spectra (`crates/nmr-core/src/spectrum.rs` — v0.2 working)
- Reads Bruker TopSpin processed directories (`procs`, `proc2s`, `proc3s` + submatrix-tiled `2rr` and `3rrr`, `int32`/`float32`/`float64`, little/big endian, `NC_proc` scaling) into `Spectrum2D` / `Spectrum3D` with per-axis `AxisPpm`.
- Reads and writes NMRPipe 2D/3D files (`.ft2`, `.ft3`, 512-float 2048-byte header with automatic float endianness detection).
- Acceptance tested in `crates/nmr-core/tests/pipeline_tests.rs`: round-trips a spectrum through both NMRPipe and Bruker `pdata` and reproduces picked peak positions within `< 0.25` digital points.

## Planned
- Multi-file NMRPipe 3D template series (`test%03d.ft3`).
- NEF reader and writer (shifts, peaks, restraints) for interoperability.
