# nmr-assign: rules for AI coding agents

Open source automated protein NMR backbone resonance assignment and interactive WebGPU spectrum workbench.
Working name; rename freely before first release.

## Scope (v0.3)
Processed spectra (Bruker `pdata`, NMRPipe, Sparky `.ucsf`) or peak lists in; picked peaks, backbone chemical shifts (H, N, CA, CB, C), confidences, and interactive 2D/3D strip inspection out.
Supported experiments: HSQC, HNCA, HN(CO)CA, HNCACB, CBCA(CO)NH, HNCO, HN(CA)CO.
Planned next: neural 3D CNN peak picker, sidechains, NOESY, structure.

## Architecture
Pure Rust + WebGPU (`wgpu` / WGSL) workspace:
- `crates/nmr-core/`: `f64` reference algorithms, NMR-STAR & TSV I/O, Bruker `pdata` & NMRPipe spectrum readers, classical peak picker, simulator, and multi-threaded CPU solver. Every algorithmic change lands here with passing parity and benchmark tests.
- `crates/nmr-gpu/`: WebGPU (`wgpu` + WGSL) compute engine for scoring matrices (`U`, `P`) and massively parallel multi-replica simulated annealing (`--engine gpu`), parity-tested against `nmr-core`.
- `crates/nmr-gui/`: Interactive `eframe` / `egui` + `wgpu` desktop/WebGPU workbench (2D GPU contour shader, sequence confidence ribbon, 3D sequential walk strip plots, live solver).
- `crates/nmr-cli/`: Command-line interface (`nmrassign`).
- Interchange: NMR-STAR v3 for shifts, the TSV peak format in `docs/specs/03-assignment.md` for peaks. NEF support is a v2 goal.

## Clean-room rules (non-negotiable)
Permitted sources:
- Peer-reviewed papers, their supplementary information, and public method descriptions.
- Open data: BMRB, PDB, PDB Chemical Component Dictionary.
- Code under the licence allowlist below.

Forbidden:
- Source, binaries, decompiled output, residue libraries, macros, parameter files, or output files from CYANA, FLYA, ARTINA, NMRtist, Xplor-NIH, ARIA, CcpNmr, or any tool obtained under an academic or commercial licence.
- Reading the source of GPL or AGPL tools (DEEP Picker, COMPASS and others) while implementing the matching module. Work from their papers instead.
- Using those product names in module, package, or file names. Compatibility claims in docs must be true and phrased as comparisons.

Provenance: every algorithm module starts with a docstring citing the paper and section it implements, or stating that the method is original.

## Licence allowlist
- Allowed: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC, Zlib.
- Ask the maintainer first: LGPL, MPL-2.0, EPL, anything else.
- Not allowed: GPL, AGPL, non-commercial licences, unlicensed code.
List every dependency and its licence in `THIRD_PARTY.md`.

## Data rules
- The maintainer's unpublished assignments never enter the public repo. Keep them in `data/private/` (git-ignored).
- Test fixtures are synthetic or derived from public BMRB entries, with the BMRB ID recorded in the fixture header.

## Definition of done for any solver change
1. `cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5` (realistic Bruker 5-spectrum physics simulator from `84e1ea8`, not the saturated 7-spectrum synthetic preset) and `cargo test --release --workspace` must not regress accuracy or high-confidence precision.
2. When `data/private/bmrb51439_cisd3/` is present locally, run 5-seed (`0..4`) evaluation on `CISD3` (`BMRB 51439`) for both `--solver standard` and `--solver evo-graph` and verify no regression in accuracy or high-confidence precision.
3. Report both the 5-seed `--realistic` benchmark numbers and the 5-seed `CISD3` numbers in the PR / `CHANGELOG.md`.

## Scientific ownership
The maintainer (an NMR scientist) owns `docs/specs/03-assignment.md` sections 3 and 4 (scoring and experiment rules). Agents implement the spec; they do not change scoring semantics without the maintainer's approval.

## Versioning & Releases
- Single source of truth: `[workspace.package] version` in `Cargo.toml` (exposed as `nmr_core::VERSION`, `nmrassign --version` / `-V` with short git SHA via `crates/nmr-cli/build.rs`, NEF metadata, HTML reports, and `/api/v1/health`).
- Pre-1.0 Semantic Versioning (`0.MINOR.PATCH`): bump `MINOR` for new capabilities, CLI/interchange changes, or benchmark-gate upgrades; bump `PATCH` for bug fixes, calibration, and documentation fixes.
- Every release bumps `[workspace.package] version`, adds a `## [X.Y.Z] — YYYY-MM-DD` section to `CHANGELOG.md`, and creates an annotated git tag `vX.Y.Z`.
