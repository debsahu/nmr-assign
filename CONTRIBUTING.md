# Contributing to `nmr-assign`

Thank you for contributing to `nmr-assign`! Because this project is built strictly clean-room under the **MIT License**, please review and follow the guidelines in [`AGENTS.md`](AGENTS.md) before opening a pull request.

## 1. Clean-Room & Provenance Policy (Non-Negotiable)

- **Permitted sources**:
  - Peer-reviewed scientific publications, supplementary tables, and public algorithm descriptions.
  - Open public-domain scientific repositories: BMRB (<https://bmrb.io/>), wwPDB (<https://www.wwpdb.org/>), and the PDB Chemical Component Dictionary (CC0 1.0).
  - Open-source code under the permissive allowlist (`MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Apache-2.0`, `ISC`, `Zlib`), credited in `THIRD_PARTY.md` and module headers.
- **Strictly forbidden**:
  - Reading, copying, decompiling, or translating source code, residue libraries, parameter tables, or output templates from proprietary, academic-only, or GPL/AGPL NMR software (including CYANA, FLYA, ARTINA, NMRtist, Xplor-NIH, ARIA, CcpNmr, DEEP Picker, COMPASS, etc.).
- **Module provenance docstrings**:
  - Every algorithmic Rust file must begin with a `//!` docstring citing the published paper(s) and equations it implements, or stating that the method is original.

## 2. Scientific Ownership

`docs/specs/03-assignment.md` Sections 3 and 4 (scoring equations and triple-resonance experiment rules) are owned by the maintainer. Please open an issue before modifying backbone scoring semantics.

## 3. Pre-PR Verification Checklist

Run the following commands and include the ubiquitin CPU/GPU accuracy and high-confidence precision numbers in your pull request description:

```bash
# 1. Zero compiler or clippy warnings across all targets
cargo clippy --workspace --all-targets

# 2. Browser WebAssembly + WebGPU target compiles cleanly
cargo check --release -p nmr-gui --target wasm32-unknown-unknown

# 3. Full release unit & integration test suite
cargo test --release --workspace

# 4. Non-regression benchmark on synthetic ubiquitin (GPU & CPU)
cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --engine gpu -v
cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --engine cpu -v
```
