# Third-Party Dependencies & Data Provenance

`nmr-assign` is released under the **MIT License** (`LICENSE`). Every dependency in the workspace is audited via `cargo metadata --format-version 1` and `Cargo.lock` against the `AGENTS.md` licence allowlist:

- **Allowed**: `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Apache-2.0`, `ISC`, `Zlib`.
- **Forbidden**: `GPL`, `AGPL`, non-commercial licences, unlicensed code (`LGPL` / `MPL-2.0` / `EPL` require explicit maintainer approval; **none** are used).

---

## 1. Direct Workspace Dependencies

| Crate | Version | SPDX Licence | Used in | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `serde` | `1.0.229` | `MIT OR Apache-2.0` | `nmr-core` | Serialization traits for chemical shift prior tables |
| `serde_json` | `1.0.154` | `MIT OR Apache-2.0` | `nmr-core`, `nmr-gui` | JSON reader/writer for BMRB prior tables and Octet `.oct` scenes |
| `bytemuck` | `1.25.2` | `Zlib OR Apache-2.0 OR MIT` | `nmr-gpu`, `nmr-gui` | Zero-copy POD struct casting for GPU uniform & storage buffers |
| `pollster` | `0.4.0` | `Apache-2.0 OR MIT` | `nmr-gpu` | Minimal single-threaded executor for headless `wgpu` initialization |
| `wgpu` | `30.0.1` | `MIT OR Apache-2.0` | `nmr-gpu`, `nmr-gui` | Cross-platform WebGPU / Metal / Vulkan / DX12 compute & rendering |
| `eframe` | `0.37.0` | `MIT OR Apache-2.0` | `nmr-gui` | Native (`winit`) & WebAssembly (`WebRunner`) GUI shell |
| `wasm-bindgen` | `0.2.130` | `MIT OR Apache-2.0` | `nmr-gui` (`wasm32`) | Rust/WebAssembly JavaScript interop bindings |
| `wasm-bindgen-futures` | `0.4.80` | `MIT OR Apache-2.0` | `nmr-gui` (`wasm32`) | Async Rust `Future` to JavaScript `Promise` bridge |
| `web-sys` | `0.3.107` | `MIT OR Apache-2.0` | `nmr-gui` (`wasm32`) | Browser DOM, `<canvas>`, and `Performance.now()` Web APIs |

---

## 2. Key Transitive Workspace Crates (`Cargo.lock` Audit)

All 327 resolved packages in `Cargo.lock` were audited via `cargo metadata --format-version 1`. Zero `GPL-only`, `AGPL`, `LGPL-only`, or `MPL-2.0` crates are present.

| Crate Family | Version(s) | SPDX Licence | Notes |
| :--- | :--- | :--- | :--- |
| `egui`, `egui-wgpu`, `egui-winit`, `epaint`, `emath`, `ecolor` | `0.37.0` | `MIT OR Apache-2.0` | Immediate-mode GUI layout, vector tessellation, and `wgpu` render pass integration |
| `wgpu-core`, `wgpu-hal`, `wgpu-types`, `naga` | `30.0.1` | `MIT OR Apache-2.0` | WebGPU validation layer, Metal/Vulkan/DX12/GLES HAL backends, and WGSL shader compiler |
| `winit`, `raw-window-handle`, `cursor-icon` | `0.30` / `0.6` / `1.2` | `Apache-2.0` / `MIT OR Apache-2.0 OR Zlib` | Cross-platform desktop windowing & input events |
| `js-sys`, `web-time` | `0.3.107` / `1.1.0` | `MIT OR Apache-2.0` | WebAssembly JS standard objects and browser time bindings |
| `parking_lot`, `lock_api`, `smallvec`, `hashbrown`, `indexmap` | `0.12` / `0.4` / `1.16` / `0.17` / `2.14` | `MIT OR Apache-2.0` | Synchronization primitives and compact data structures |
| `bytemuck_derive`, `objc2-*`, `dispatch2`, `foldhash`, `slotmap`, `zlib-rs` | various | `Zlib` / `Zlib OR Apache-2.0 OR MIT` | macOS Metal Objective-C bindings, fast hashing, and arena maps |
| `zerocopy`, `moxcms`, `pxfm`, `num_enum` | `0.8` / `0.1` / `0.7` | `BSD-2-Clause` / `BSD-3-Clause` (with `MIT`/`Apache-2.0`) | Safe byte transmutation and color/math helpers |
| `libloading` | `0.8.9` | `ISC` | Dynamic shared library loader for Vulkan/windowing backends |

### Transitive Multi-Licence & Platform-Specific Notes

1. **Dual/Multi-licensed crates with copyleft alternatives** (`self_cell 1.3.0`: `Apache-2.0 OR GPL-2.0-only`; `r-efi 5.3.0`/`6.0.0`: `MIT OR Apache-2.0 OR LGPL-2.1-or-later`): Used strictly under their **`Apache-2.0` / `MIT`** licence election.
2. **Bundled GUI fonts (`epaint_default_fonts 0.37.0`)**: Licensed `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0` (covers the embedded Hack, NotoEmoji, and Ubuntu proportional/monospace UI font glyphs shipped by `eframe`'s `default_fonts` feature).
3. **Unicode character tables (`unicode-ident 1.0.27`, `unicode-properties 0.1.5`)**: Licensed `(MIT OR Apache-2.0) AND Unicode-3.0` (permissive OSI-approved Unicode Data Files licence).
4. **Windows-only clipboard bindings (`clipboard-win 5.4.3`, `error-code 3.4.0`)**: Licensed `BSL-1.0` (Boost Software License 1.0); target-gated to `cfg(windows)` via `arboard` and not compiled on macOS, Linux, or `wasm32-unknown-unknown`.

---

## 3. Public Scientific Data Provenance

All empirical chemical-shift priors, topology tables, and real-protein benchmark fixtures derive exclusively from open public-domain scientific repositories:

- **Biological Magnetic Resonance Data Bank (BMRB)** (<https://bmrb.io/>): Public-domain / **CC0 1.0 Universal** archival NMR-STAR depositions. Every BMRB entry used to train amino-acid chemical shift priors (`data/priors/PROVENANCE.md`) or evaluate assignment benchmarks (`data/benchmark/manifest.tsv` and fixture headers) is explicitly cited by its BMRB accession ID.
- **RCSB Protein Data Bank (PDB) & Chemical Component Dictionary (CCD)** (<https://www.wwpdb.org/>): Public-domain / **CC0 1.0 Universal** macromolecular coordinate depositions (`data/pdb_refs/1d3z.pdb`, `1enh.pdb`, `1mjc.pdb`) and standard amino-acid covalent atom nomenclature (`IUPAC` / `NMR-STAR v3.2`). Full wwPDB CC0 1.0 licence text is archived in [`.licenses/pdb_database_LICENSE.txt`](.licenses/pdb_database_LICENSE.txt).
- **Synthetic Test Fixtures (`data/fixtures/synthetic_ubiquitin.str`)**: Clean-room synthetic 76-residue ubiquitin chemical shift table generated from random-coil priors for deterministic unit, GPU parity, and CI regression testing.

---

## 4. Adapted Open-Source Algorithm Attribution

- **Octet-Draw** (<https://github.com/SamChemMan/Octet-Draw>, Copyright (c) 2026 Sam Ellis, **Apache-2.0**): Chemical graph/stereochemistry layout (`octet_mol`), 1D ¹H/¹³C NMR & exact-mass/IUPAC analytical engines (`octet_analyt`), publication-quality ACS/RSC/JOC/Angewandte vector & raster 2D chemical rendering (`octet_render`), and interactive `egui` canvas workflows (`octet_canvas`) are adapted into pure Rust under the permissive Apache License 2.0.
