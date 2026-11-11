# nmr-assign

**Open-source, GPU-accelerated protein NMR resonance assignment, 3D structure folding, and interactive WebGPU + Octet chemical workbench — written in pure Rust.**

`nmr-assign` takes processed 2D/3D NMR spectra (Bruker TopSpin `pdata`, NMRPipe `.ft2`/`.ft3`) or peak lists (`TSV`, `CCPN NEF v1.1`) and produces sequence-specific **backbone & sidechain chemical shift assignments**, **2D/3D neural CNN peak lists**, **3D NOESY torsion-angle structure ensembles**, and **self-contained HTML/SVG quality reports** — alongside a dual-mode native & browser WebGPU GUI featuring both a **Protein NMR 2D/3D Strip Workbench** and the full **Octet Chemical Draw, Reaction & 1D NMR Studio**.

Built clean-room under the **MIT License** from peer-reviewed methods, open BMRB/wwPDB data, and Apache-2.0 open-source components, with zero proprietary or copyleft dependencies.

---

## Highlights

- **Cross-Platform WebGPU Compute (`wgpu` + WGSL)**:
  - Evaluates residue-typing log-likelihood-ratio matrices $U(s, k)$, Gaussian sequential connectivity matrices $P(s, t)$, and parallel Metropolis-Hastings simulated annealing replicas directly in GPU workgroup SRAM (Apple Metal, Vulkan, DX12, WebGPU), with a deterministic multi-threaded `f64` CPU engine (`--engine cpu`).
  - Empirical BMRB priors trained on 36 curated public BMRB depositions (`data/priors/bmrb_priors.json`) and benchmarked across a 12-protein suite spanning $\alpha$, $\beta$, $\alpha/\beta$, and intrinsically disordered proteins (`data/benchmark/manifest.tsv`).
- **Classical & 2D/3D Neural CNN Peak Picking (`--method classical|neural`)**:
  - Reads Bruker `pdata` (`2rr`/`3rrr`) and NMRPipe `.ft2`/`.ft3` spectra with full `ppm` referencing.
  - Clean-room 2D & 3D separable dilated residual CNN (`nmr-core::neural_picker` + `nmr-gpu::neural_gpu` WGSL compute shaders with $< 10^{-4}$ GPU/CPU parity) with Haar-wavelet noise normalization, indirect $t_1/t_2$ ridge suppression, $3\times 3\times 3$ non-maximum suppression, and sub-voxel offset regression.
- **Aliphatic & Aromatic Sidechain Assignment (`nmrassign sidechain`)**:
  - Complete IUPAC / PDB CCD covalent atom & TOCSY spin-coupling topology for all 20 standard amino acids (`nmr-core::topology`).
  - Simulates, picks, and jointly optimizes `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, and aromatic experiments (`solve_joint_backbone_sidechain`), resolving pre-proline and double-proline spin systems.
- **Dihedral Prediction, Iterative NOESY ADR & Torsion-Angle 3D Folding (`nmrassign structure`)**:
  - Predicts $(\phi, \psi)$ torsion restraints and secondary structure from chemical shifts (`nmr-core::dihedrals`).
  - 7-cycle ARIA/CANDID-style $r^{-6}$ ambiguous distance restraint (`ADR`) calibration & network-anchoring filter (`nmr-core::noesy`) coupled to an analytical-gradient NeRF torsion-angle simulated annealing folder, Kabsch RMSD superposer, and Ramachandran validator (`nmr-core::structure`).
- **CCPN NEF v1.1, HTML+SVG Quality Reports & Self-Hosted Service (`nef-*`, `report`, `pipeline`, `serve`)**:
  - Lossless `NEF v1.1` import/export across `nef_molecular_system`, `nef_chemical_shift_list`, `nef_nmr_spectrum` (including signed `HNCACB` $-\text{C}^\beta$ & `3D NOESY`), and ambiguous OR distance/dihedral restraint lists.
  - Dependency-free single-file `HTML + inline SVG` executive reports and a zero-dependency HTTP REST job server + drag-and-drop browser dashboard (`nmrassign serve`).
- **Integrated Octet Chemical Draw, Reaction & 1D NMR Studio (`octet_*` + `nmrassign chem`)**:
  - Full 2D chemical structure & reaction sketcher (`🧬 Protein NMR & 3D Strips` | `⚗️ Octet Chemical Draw & 1D NMR` mode switcher in `nmr-gui` on both desktop and `wasm32-unknown-unknown`).
  - Supports `ChemDraw CDXML`, binary `CDX` (`VjCD0100`), `.oct`, `MOL V2000/V3000`, `SDF`, `SMILES`, Reaction SMILES, `InChI` / `InChIKey`, `JCAMP-DX`, `SVG`, `PDF 1.4`, and antialiased `PNG`.
  - Includes Clean 2D layout, CIP `(R)/(S)` & `(E)/(Z)` stereoperception, 50+ rings/amino-acids/organometallics/clipware templates, reaction stoichiometry calculator, offline bidirectional `IUPAC Name ↔ Structure`, 1-click polypeptide import from the loaded NMR protein, and live 6-sphere HOSE + additivity $^{1}\text{H}$ / $^{13}\text{C}$ 1D NMR multiplet simulation with interactive peak-to-atom highlighting.

---

## Supported Protein NMR Experiments

| Category | Experiments | Observed Nuclei |
| :--- | :--- | :--- |
| **Backbone 2D / 3D** | `HSQC`, `HNCA`, `HNCOCA`, `HNCACB` ($+\text{C}^\alpha / -\text{C}^\beta$), `CBCACONH`, `HNCO`, `HNCACO` | $^{1}\text{H}^{\text{N}}, {}^{15}\text{N}, {}^{13}\text{C}^\alpha, {}^{13}\text{C}^\beta, {}^{13}\text{C}'$ |
| **Sidechain 2D / 3D** | `C13HSQC`, `HCCHTOCSY`, `CCCONH`, `HCCCONH`, `HBHACONH`, `AROMHSQC` | Aliphatic & aromatic $^{1}\text{H}$ / $^{13}\text{C}$ |
| **Through-Space NOE** | `NOESY15N`, `NOESY13C` | 3D $^{15}\text{N}$- / $^{13}\text{C}$-edited $^{1}\text{H}$–$^{1}\text{H}$ NOEs |

---

## Quick Start & CLI Reference (`nmrassign`)

Requires **Rust 1.85+** (`cargo`). Automatically uses **Metal** on macOS, **Vulkan / DX12** on Linux & Windows, **WebGPU** in browsers (`wasm32-unknown-unknown`), or multi-threaded CPU (`--engine cpu`).

```bash
# 1. Launch the dual-mode interactive WebGPU GUI (Protein 2D/3D NMR + Octet Chemical Studio)
cargo run --release -p nmr-gui

# 2. Single-protein & 12-protein BMRB blind assignment benchmarks
cargo run --release -p nmr-cli -- benchmark --star data/fixtures/synthetic_ubiquitin.str --engine gpu -v
cargo run --release -p nmr-cli -- benchmark-suite --manifest data/benchmark/manifest.tsv --engine gpu

# 3. End-to-end blind volumetric spectra -> neural 3D CNN peak picking -> assignment benchmark
cargo run --release -p nmr-cli -- e2e-benchmark --star data/fixtures/synthetic_ubiquitin.str

# 4. Pick 2D/3D spectra with the classical or WebGPU neural 3D CNN picker
cargo run --release -p nmr-cli -- pick \
  --hsqc path/to/bruker/10/pdata/1 \
  --triple HNCACB=path/to/hncacb.ft3 \
  --out peaks.tsv --method neural --engine gpu

# 5. Joint backbone + sidechain assignment (C13-HSQC + HCCH-TOCSY + (H)CC(CO)NH)
cargo run --release -p nmr-cli -- sidechain --star data/benchmark/bmr17769.str --out ubq_all_atom.str

# 6. Chemical-shift (phi, psi) dihedrals + iterative NOESY ADR + 3D torsion-angle ensemble folding
cargo run --release -p nmr-cli -- structure \
  --star data/benchmark/bmr15536.str \
  --pdb-ref data/pdb_refs/1enh.pdb \
  --out-pdb 1enh_ensemble.pdb --conformers 10

# 7. CCPN NEF v1.1 interchange, HTML+SVG quality reports, one-command pipeline & HTTP job service
cargo run --release -p nmr-cli -- nef-export --star data/fixtures/synthetic_ubiquitin.str --out ubq.nef
cargo run --release -p nmr-cli -- nef-import --in ubq.nef --out-star roundtrip.str
cargo run --release -p nmr-cli -- report --star data/fixtures/synthetic_ubiquitin.str --out report.html
cargo run --release -p nmr-cli -- pipeline --star data/fixtures/synthetic_ubiquitin.str --out-dir ./out_ubq
cargo run --release -p nmr-cli -- serve --bind 127.0.0.1:8080 --storage ./jobs

# 8. Octet-Draw CLI: IUPAC Name <-> Structure, InChIKey, CDXML/CDX/MOL/SVG/PDF/JCAMP-DX & 1D NMR
cargo run --release -p nmr-cli -- chem info --name "2-[4-(2-methylpropyl)phenyl]propanoic acid" --out ibuprofen.svg
cargo run --release -p nmr-cli -- chem info --smiles "CC(=O)Oc1ccccc1C(=O)O" --out aspirin.cdxml

# 9. Verify browser WebAssembly + WebGPU target & run the full workspace test suite
cargo check --release -p nmr-gui --target wasm32-unknown-unknown
cargo test --release --workspace
```

---

## Representative Benchmarks

- **76-aa Ubiquitin Backbone Assignment (`data/fixtures/synthetic_ubiquitin.str`, 6% missing, 4% spurious + 18 $\text{NH}_2$ decoys)**:
  - **WebGPU (Metal, 64 replicas)**: **97.1%** accuracy, **97.1%** recall, **100.0%** high-confidence ($\ge 0.80$) precision.
  - **Multi-core CPU (`f64`, 12 replicas)**: **97.1%** accuracy, **97.1%** recall, **98.0%–100.0%** high-confidence precision in `~0.03 s`.
- **12-Protein Public BMRB Suite (`data/benchmark/manifest.tsv`, 55–148 aa across $\alpha$, $\beta$, $\alpha/\beta$, and 140-aa IDP $\alpha$-synuclein)**:
  - **99.4%** mean high-confidence ($\ge 0.80$) precision (10 of 12 proteins at **100.0%** HC precision) and **86.1%** overall mean accuracy (**95.3%** mean accuracy across globular proteins $\le 80$ aa).
- **3D Neural Peak Picker (`nmrassign e2e-benchmark`)**:
  - Beats the classical MAD picker across volumetric 2D/3D cubes containing $t_1/t_2$ ridges, baseline roll, and overlapped shoulders while matching WebGPU `conv3d.wgsl` logits within `< 1e-4`.
- **De Novo NMR Structure Folding (`nmrassign structure`)**:
  - Folds Engrailed Homeodomain (`1ENH`, 61 aa) and Human Ubiquitin (`1D3Z`, 76 aa) from chemical shifts + simulated 3D NOESY peaks to **< 3.0 Å** ordered backbone RMSD with **> 94%** Ramachandran favored+allowed geometry.

---

## Workspace Architecture

```text
crates/
  nmr-core/     f64 reference library: NMR-STAR/NEF/TSV I/O, Bruker pdata & NMRPipe readers,
                classical & separable-CNN 2D/3D peak pickers, BMRB priors, spin-system builder,
                LLR + intensity-aware solver, CCD topology & joint sidechain solver, CSI dihedrals,
                iterative r^-6 NOESY ADR & torsion-angle folder, HTML/SVG reports, HTTP job server,
                and Octet-Draw (octet_mol, octet_analyt, octet_render)
  nmr-gpu/      Headless & shared-device WebGPU (wgpu + WGSL) compute engine for U/P scoring
                matrices, workgroup-SRAM multi-replica annealing, and 3D CNN peak inference
  nmr-gui/      Dual-mode eframe/egui + wgpu desktop & WebAssembly workbench:
                  1) Protein NMR 2D GPU contour viewer, sequence ribbon & 3D sequential walk strips
                  2) Octet Chemical Draw, Reaction, Stoichiometry & Live 1D NMR Studio
  nmr-cli/      `nmrassign` CLI (benchmark, benchmark-suite, e2e-benchmark, assign, simulate,
                pick, priors, sidechain, structure, nef-import, nef-export, report, pipeline,
                serve, chem)
docs/specs/     Module specifications & scoring equations (00-architecture.md .. 08-service.md)
data/           BMRB priors (data/priors/), 12-protein benchmark set (data/benchmark/),
                PDB validation references (data/pdb_refs/), and synthetic fixtures (data/fixtures/)
```

---

## Authors & Attribution

- **Authors**: **Debashish Sahu** (`<debashish.sahu@gmail.com>`) and **Antigravity** (`<noreply@google.com>`).
- **Octet-Draw Attribution**: The `octet_mol`, `octet_analyt`, `octet_render`, and `octet_canvas` chemical structure, nomenclature, analytical 1D NMR/MS, and publication rendering suite is ported and adapted from [**Octet-Draw**](https://github.com/SamChemMan/Octet-Draw) by **Sam Ellis** (Copyright (c) 2026 Sam Ellis, licensed under **Apache-2.0**).
- **Open Scientific Data**: Empirical priors and benchmark fixtures are derived from public-domain **CC0 1.0** depositions in the [BMRB](https://bmrb.io/) (`data/priors/PROVENANCE.md`) and [wwPDB](https://www.wwpdb.org/) ([`.licenses/pdb_database_LICENSE.txt`](.licenses/pdb_database_LICENSE.txt)).

## Licence

Released under the [MIT License](LICENSE). Full third-party crate, algorithm, and dataset provenance is documented in [`THIRD_PARTY.md`](THIRD_PARTY.md). See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) for clean-room contribution rules.
