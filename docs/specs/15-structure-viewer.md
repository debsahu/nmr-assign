# 15. WebGPU 3D Biomolecular Structure Viewport, Bidirectional Spectrum Coupling & Quantitative NMR Overlays

> **Provenance & Clean-Room Statement**
> Original clean-room specification and mathematical design implemented in `crates/nmr-gui/src/mol3d_viewer.rs`, `crates/nmr-gui/src/mol3d_overlay.rs`, and `crates/nmr-gui/src/shaders/mol3d.wgsl` (docked into `crates/nmr-gui/src/main.rs`). Zero source code, shaders, residue libraries, macros, parameter files, or binaries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) molecular visualization or NMR structure package were consulted. Every geometric representation, alignment solver, secondary-structure classifier, ray-intersection primitive, colormap, and restraint metric derives strictly from peer-reviewed literature and open WWPDB/BMRB data standards:
>
> 1. **Smooth peptide-plane cartoon ribbons & extruded cross-sections**:
>    - Carson, M. & Bugg, C. E. (*J. Mol. Graph.* **1986**, 4, 121–122); Carson, M. (*J. Mol. Graph.* **1987**, 5, 103–106) — uniform cubic B-spline / Hermite backbone interpolation through $\text{C}\alpha_i$ guide points and $\text{C}\alpha_i \to \text{O}_i$ peptide-plane binormal orientation with sign-flip continuity across $\beta$-strands.
>    - Richardson, J. S. (*Adv. Protein Chem.* **1981**, 34, 167–339) — schematic secondary-structure taxonomy (flattened helical ribbons, arrowed $\beta$-strands, cylindrical coil tubes).
> 2. **Backbone hydrogen-bond & dihedral secondary structure**:
>    - Kabsch, W. & Sander, C. (*Biopolymers* **1983**, 22, 2577–2637) — electrostatic $\text{C}=\text{O}_i \cdots \text{H–N}_j$ partial-charge hydrogen-bond energy ($E < -0.5\text{ kcal mol}^{-1}$, $q_1 = 0.42\,e$, $q_2 = 0.20\,e$, $332\text{ kcal Å mol}^{-1} e^{-2}$), $n$-turn patterns ($n \in \{3, 4, 5\}$ for $3_{10}$/$\alpha$/$\pi$ helices), parallel/antiparallel $\beta$-bridges, and Ramachandran $(\phi, \psi)$ fallback when coordinate models omit `HELIX`/`SHEET` records.
> 3. **Optimal rigid-body ensemble superposition**:
>    - Kabsch, W. (*Acta Crystallogr. A* **1976**, 32, 922–923; *Acta Crystallogr. A* **1978**, 34, 827–828); Horn, B. K. P. (*J. Opt. Soc. Am. A* **1987**, 4, 629–642) — exact $4 \times 4$ symmetric Davenport/Horn quaternion eigensystem yielding proper rotation matrices $\mathbf{R} \in \text{SO}(3)$ ($\det \mathbf{R} = +1$) minimizing weighted $\text{C}\alpha$ coordinate RMSD across multi-conformer NMR ensembles.
> 4. **Interactive camera control, van der Waals radii & perceptually uniform colormaps**:
>    - Shoemake, K. (*Proc. Graphics Interface* **1992**, 151–156) — quaternion arcball 3D rotation without gimbal lock.
>    - Bondi, A. (*J. Phys. Chem.* **1964**, 68, 441–451); Alvarez, S. (*Dalton Trans.* **2013**, 42, 8617–8636) — atomic van der Waals and transition-metal covalent/ionic radii.
>    - Nuñez, J. R., Anderton, C. R. & Renslow, R. S. (*PLOS ONE* **2018**, 13, e0199239); Moreland, K. (*ISVC* **2009**, LNCS 5876, 92–103) — perceptually uniform sequential (`Viridis`, color-vision-deficiency-safe `Cividis`) and diverging (`BlueWhiteRed`) scientific colormaps.
> 5. **Quantitative NMR structural & dynamical tracks**:
>    - Wishart, D. S., Sykes, B. D. & Richards, F. M. (*Biochemistry* **1992**, 31, 1647–1651) — Chemical Shift Index (`CSI`).
>    - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982**, 104, 4546–4559) — model-free generalized order parameter $S^2$ and conformational exchange $R_{\text{ex}}$.
>    - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1–16) — combined amide chemical shift perturbation $\Delta\delta_{\text{comb}} = \sqrt{(\Delta\delta_{\text{H}})^2 + (\alpha_{\text{N}} \Delta\delta_{\text{N}})^2}$.
>    - Nilges, M. (*J. Mol. Biol.* **1995**, 245, 645–660); Solomon, I. (*Phys. Rev.* **1955**, 99, 559–565) — ambiguous $\langle r^{-6} \rangle^{-1/6}$ sum-averaged NOE/PRE effective distance restraints and upper/lower bound violation analysis.

---

## 1. WebGPU 3D Biomolecular Structure Viewport (`T12.1`, `mol3d_viewer.rs` & `shaders/mol3d.wgsl`)

### 1.1 Dockable Cross-Platform Workbench Architecture
`Mol3dViewerPanel` and `Mol3dPipelineResource` embed a hardware-accelerated 3D structure viewport directly inside the `nmr-gui` workbench alongside the 2D GPU contour canvas and 3D strip tiler on both native desktop (`Metal` / `Vulkan` / `DX12`) and browser WebAssembly (`wasm32-unknown-unknown` + WebGPU):

- **Offscreen `4x MSAA` + `Depth32Float` Pass**: Because `egui`'s parent surface render pass does not guarantee a depth buffer across platforms, `Mol3dPipelineResource` maintains a resize-aware offscreen `4x MSAA` `Rgba8Unorm` color attachment, `4x MSAA` `Depth32Float` attachment, and `1x` resolved `Rgba8Unorm` texture. During `egui_wgpu::CallbackTrait::prepare`, all opaque and alpha-blended 3D draw calls execute in a dedicated offscreen `wgpu::RenderPass` with hardware MSAA resolve; `CallbackTrait::paint` then blits the resolved texture into the `egui` panel rect via a single full-screen triangle (`vs_blit` / `fs_blit`).
- **Two-Pass Ensemble Transparency**: Multi-conformer NMR bundles render conformer $0$ (representative medoid/first conformer) through depth-writing opaque pipelines (`depth_write_enabled = true`, `CompareFunction::LessEqual`) and conformers $1 \dots M-1$ (plus translucent distance-restraint cylinders) through depth-tested alpha-blended pipelines (`depth_write_enabled = false`, `BlendState::ALPHA_BLENDING`), preserving sharp silhouette readability while revealing conformational disorder in loops and termini.
- **Zero New Dependencies**: All linear algebra (`Vec3`, `Quat`, $4 \times 4$ projection/view matrices, $4 \times 4$ Jacobi eigensolvers), `.pdb`/`.mmcif` parsers, PNG readback (`png` crate already in workspace), and painter's-algorithm vector SVG projection run in pure Rust + WGSL 1.0.

### 1.2 Hostile-Input-Safe `.pdb` & `.cif`/`.mmcif` Ingestion
`parse_pdb_ensemble`, `parse_mmcif_ensemble`, and `parse_structure_auto` ingest single-structure X-ray/AlphaFold coordinates and multi-model NMR ensembles (`MODEL` / `ENDMDL` or `_atom_site.pdbx_PDB_model_num`) via toolbar load, clipboard paste, or drag-and-drop:

1. **Multibyte UTF-8 Byte-Slice Safety**: Fixed-column `.pdb` records (`ATOM`, `HETATM`, `HELIX`, `SHEET`, `MODEL`, `ENDMDL`, `TER`) are parsed **exclusively** on `&[u8]` via `line.as_bytes().get(start..end)` followed by `.is_ascii()` validation before ASCII-to-`&str` conversion. Multibyte UTF-8 characters anywhere in a record or comment never trigger Rust char-boundary panics.
2. **Coordinate Sanity Gate**: Every parsed $(x, y, z)$ coordinate is strictly validated with `.is_finite()` (`|coord| <= 10^6\text{ Å}`); `NaN`, `Inf`, and `-Inf` coordinates return a structured `Err(...)`, while missing or non-finite `occupancy` and isotropic $B$-factor fields fall back cleanly to `1.0` and `0.0`.
3. **Chemical & Altloc Normalization**:
   - Selenomethionine `MSE` `HETATM` polymer residues are normalized to canonical `MET` (mapping `SE` $\to$ `SD`) so selenomethionine crystal references form unbroken backbone splines and bind directly to NMR sequence positions.
   - Alternate-location conformers (`altLoc` column 17 / `_atom_site.label_alt_id`) retain blank/`' '`/`'.'`/`'?'` or `'A'` when present, **and** retain `'B'`-only (first-seen altloc) atoms when a disordered residue omits `'A'` altogether, avoiding missing-atom gaps.
   - Quoted mmCIF values (`'...'`, `"..."`, and `;...;` multiline blocks) carry an explicit `is_quoted` token flag so literal strings starting with `'_'`, `'loop_'`, or `'data_'` never prematurely terminate loop skipping or trigger control keywords. Truncated mmCIF `loop_` rows (either trailing rows with fewer than `ncols` tokens or mid-row truncation before the next control tag) return an explicit `Err(...)` rather than silently dropping trailing atoms or shifting column alignment.
   - Chain, `(seq_id, ins_code)` residue, and primary-sequence index bindings in `finalize_structure_ensemble`, `bind_to_nmr_sequence`, and `superpose_ensemble_kabsch` use $O(\text{atoms} + \text{residues})$ `HashMap` / direct-slot indexing with zero quadratic $O(\text{atoms} \times \text{residues})$ scans.
   - Non-water metal ions (`ZN`, `FE`, `CU`, `MN`, `MG`, `CA`, `CO`, `NI`, `CD`, `HG`) and iron–sulfur cluster atoms are retained as `MetalSite3d` spheres with automatic coordination-bond cylinders (`d <= 2.75\text{ Å}`) to ligating `CYS SG`, `HIS ND1/NE2`, `ASP OD1/OD2`, `GLU OE1/OE2`, and `MET SD` donors.

### 1.3 Kabsch–Sander Secondary Structure & Kabsch Ensemble Superposition
1. **Intrinsic Secondary Structure (`compute_kabsch_sander_secondary_structure`)**:
   - Reconstructs ideal trans-peptide amide proton coordinates $\mathbf{r}_{\text{H}_i} = \mathbf{r}_{\text{N}_i} + 1.00\text{ Å}\cdot\widehat{(\mathbf{r}_{\text{C}_{i-1}} - \mathbf{r}_{\text{O}_{i-1}})}$ (`d(C_{i-1}, N_i) <= 2.2 Å`, Kabsch & Sander 1983, p. 2580) on non-Proline residues when explicit `H`/`HN` atoms are absent.
   - Evaluates Kabsch–Sander electrostatic backbone hydrogen-bond energies
     $$E(i, j) = 0.42 \cdot 0.20 \cdot 332 \left(\frac{1}{r_{\text{ON}}} + \frac{1}{r_{\text{CH}}} - \frac{1}{r_{\text{OH}}} - \frac{1}{r_{\text{CN}}}\right)\text{ kcal mol}^{-1}$$
     identifying H-bonds where $E(i, j) < -0.5\text{ kcal mol}^{-1}$.
   - Assigns $\alpha$-helices ($4$-turns), $3_{10}$/$5$-turn helices, and parallel/antiparallel $\beta$-sheet ladders, supplemented by contiguous $(\phi, \psi)$ Ramachandran basin classification when explicit carbonyl oxygens are sparse. Models lacking `HELIX`/`SHEET` records (common in NMR depositions and computational models) therefore render complete helical ribbons and arrowed $\beta$-strands automatically.
2. **Multi-Model Quaternion Kabsch Superposition (`kabsch_superpose_points`)**:
   - Aligns every conformer $m \in \{1, \dots, M-1\}$ onto conformer $0$ over shared $\text{C}\alpha$ atoms by constructing the $4 \times 4$ Horn/Kabsch key matrix $\mathbf{K}(\mathbf{H})$ from the cross-covariance $\mathbf{H} = \sum_k (\mathbf{p}_k - \bar{\mathbf{p}})(\mathbf{q}_k - \bar{\mathbf{q}})^{\mathsf{T}}$ and extracting its principal unit quaternion eigenvector via cyclic Jacobi sweeps ($\det \mathbf{R} = +1$ guaranteed without reflection).
   - Computes per-residue ensemble $\text{C}\alpha$ RMSD
     $$\text{RMSD}_k = \sqrt{\frac{1}{M}\sum_{m=0}^{M-1} \left\|\mathbf{r}^{(m)}_{\text{C}\alpha,k} - \bar{\mathbf{r}}_{\text{C}\alpha,k}\right\|^2}$$
     for structural putty scaling and disorder inspection. When a peptide or fragment has `< 3` shared $\text{C}\alpha$ atoms (where proper rigid-body $\text{SO}(3)$ alignment is underdetermined) or only a single conformer, `kabsch_rmsd_to_ref_opt` and `mean_ensemble_ca_rmsd_opt` return `None` (`CA RMSD n/a: <3 CAs`) rather than reporting unsuperposed coordinate spread.

### 1.4 Carson–Bugg Cubic Ribbon Extrusion, Instanced Atoms & Camera/Export Pipeline
- **Peptide-Plane Cubic Splines**: Consecutive polypeptide segments ($d(\text{C}\alpha_i, \text{C}\alpha_{i+1}) \le 4.8\text{ Å}$) are subdivided (`RIBBON_SUBDIVS = 4` subdivisions/residue, `RIBBON_RING_VERTS = 8` elliptical ring vertices) using Catmull–Rom / cubic Hermite guide tangents $\mathbf{T}(t)$ and peptide-plane normal vectors $\mathbf{B}_i = \text{normalize}((\mathbf{r}_{\text{C}\alpha_{i+1}} - \mathbf{r}_{\text{C}\alpha_{i-1}}) \times (\mathbf{r}_{\text{O}_i} - \mathbf{r}_{\text{C}\alpha_i}))$ with $\mathbf{B}_i \cdot \mathbf{B}_{i-1} < 0 \implies \mathbf{B}_i \leftarrow -\mathbf{B}_i$ sign-flip suppression across $\beta$-strands.
- **Secondary-Structure Cross-Sections**:
  - `Helix`: wide elliptical ribbon (half-axes $1.12\text{ Å} \times 0.22\text{ Å}$) with smooth terminal taper.
  - `Sheet`: flat $\beta$-ribbon (half-axes $1.18\text{ Å} \times 0.20\text{ Å}$) flaring into a $1.95\text{ Å}$ triangular arrowhead over the C-terminal residue of each strand.
  - `Coil`: circular tube ($r = 0.28\text{ Å}$), smoothly modulated by putty radius multipliers.
- **Instanced Sidechains, CPK Half-Bonds & Metal Cofactors**: Heavy/polar atoms render via 42-vertex `Subdiv-1` icosphere instances (`SphereInstance`), while covalent bonds render via unit cylinder instances (`CylinderInstance`) split at `v_end.y = 0.5` in `fs_cylinder` for crisp two-tone CPK half-bonds (`C` slate-gray, `N` royal-blue, `O` crimson, `S` gold, `Zn`/`Fe` metallic).
- **Camera, Slab & Export**:
  - Shoemake quaternion arcball rotation, right/shift-drag pan, scroll zoom, `Perspective` / `Orthographic` projection toggle, and view-space front/back Z-slab clipping (`discard` in `fs_lit` / `fs_cylinder` when $-z_{\text{view}} \notin [z_{\text{slab,near}}, z_{\text{slab,far}}]$).
  - Crisp high-DPI offscreen `PNG` readback (`export_offscreen_png`) and deterministic depth-sorted `SVG` vector export (`export_structure_svg`) passing every interpolated title, residue name, atom name, and annotation through `escape_xml` (`&`, `<`, `>`, `"`, `'`). Before GPU geometry upload or PNG readback buffer creation, `validate_scene_gpu_buffer_limits` and `validate_mol3d_buffer_limits` check `max_buffer_size` and `max_storage_buffer_binding_size` on `device.limits()`, `render_offscreen_rgba` propagates `device.poll` and `map_async` errors, and `write_mol3d_export_file` surfaces filesystem write errors directly in the status banner rather than reporting `"Exported"`.
- **Performance Gate**: Headless `wgpu` rendering sustains $\ge 60\text{ fps}$ ($\le 16.67\text{ ms}$ mean frame time over 30 frames after warm-up, synchronized via `device.poll(PollType::wait_indefinitely())`) on a 20-conformer `150 aa` all-atom NMR ensemble ($\ge 48{,}000$ atoms).

---

## 2. Bidirectional 3D–Spectrum Picking, Quantitative Putty & NOE/PRE Restraint Cylinders (`T12.2`, `mol3d_overlay.rs`)

### 2.1 Exact Analytical 3D Ray Picking & Bidirectional Workbench Coupling
Viewport clicks `(local_x, local_y)` unproject through $\mathbf{M}_{\text{VP}}^{-1}$ into world-space rays $\mathbf{r}(t) = \mathbf{o} + t\hat{\mathbf{d}}$ (`screen_to_world_ray`, valid for both `Perspective` and `Orthographic` projections) and test analytical primitives in `f64` precision (`pick_scene_analytical`):

1. **Ray–Sphere (`intersect_ray_sphere`)**: Exact quadratic discriminant for atom and metal cofactor spheres.
2. **Ray–Capsule (`intersect_ray_capsule`)**: Closest-approach distance between $\mathbf{r}(t)$ and finite segment $[\mathbf{a}, \mathbf{b}]$ across backbone ribbon segments and distance-restraint cylinders, returning the minimum positive ray parameter $t > 0$ (`100.0%` unit-test picking accuracy across both camera modes).
3. **Bidirectional `Mol3dSyncAction` Coupling**:
   - **3D $\to$ Spectra & 2D Graph**: Clicking any atom, ribbon residue, or restraint endpoint resolves `Mol3dSyncAction` (`resolve_3d_pick_sync`), highlighting the residue in the 1D sequence ribbon, opening its 2D IUPAC covalent sidechain Z-score graph (`SidechainInspectorPanel`), and firing `SyncCrosshair` (`apply_to_crosshair`) to snap the 2D `¹H–¹⁵N` HSQC crosshair and 3D `HNCACB`/`HNCO` strips onto its assigned shifts.
   - **Spectra / Sequence $\to$ 3D**: Selecting a residue in the 1D sequence ribbon, 2D HSQC canvas, 2D covalent sidechain inspector, or 3D sequential strip tiler invokes `Mol3dViewerPanel::highlight_sequence_residue(Some(k), true)`, highlighting the residue's ribbon/sidechain in gold and smoothly focusing the 3D arcball target onto $\text{C}\alpha_k$.

### 2.2 Quantitative NMR Putty Radius & Perceptually Uniform Colormaps (`< 2.0 ms`)
`QuantitativeTrackData` maps experimental or solver-derived per-residue observables onto both ribbon color and putty cross-section radius without retessellating cubic splines (`Mol3dSceneBuffers::update_quantitative_putty` caches ring centers, normals, binormals, and base half-axes per vertex for `< 2.0 ms` in-place updates):

| Quantitative Track (`QuantitativeTrack`) | Physical Domain & Normalization | Putty Radius Law | Colormaps (`ColormapKind`) |
|---|---|---|---|
| `AssignmentConfidence` | Solver posterior $p_k \in [0, 1]$ | Linear scaling $r_{\min} + p_k (r_{\max} - r_{\min})$ (unassigned residues render in slate at $r_{\min}$) | `Viridis`, `Cividis`, `BlueWhiteRed` |
| `ChemicalShiftIndex` | Wishart `CSI` consensus $\in [-1, +1]$ ($\beta \leftrightarrow \alpha$) | Diverging around $0.0$ ($r_{\min} + |\text{CSI}_k|(r_{\max} - r_{\min})$) | `BlueWhiteRed` (default), `Viridis`, `Cividis` |
| `LipariSzaboS2` | Picosecond–nanosecond order parameter $S^2_k \in [0, 1]$ | Flexible loops ($1 - S^2_k$) swell (`putty` mobility representation) | `Viridis`, `Cividis`, `BlueWhiteRed` |
| `ChemicalExchangeRex` | $\mu\text{s–ms}$ CPMG/CEST exchange $R_{\text{ex},k}\ (\text{s}^{-1})$ | Proportional to normalized $R_{\text{ex},k} / R_{\text{ex,max}}$ | `Viridis`, `Cividis`, `BlueWhiteRed` |
| `LigandCsp` | Combined titration shift $\Delta\delta_{\text{comb},k}\ (\text{ppm})$ | Proportional to binding-site $\Delta\delta_{\text{comb},k} / \Delta\delta_{\text{max}}$ | `Viridis`, `Cividis`, `BlueWhiteRed` |
| `EnsembleRmsd` | Superposed conformer $\text{C}\alpha\ \text{RMSD}_k\ (\text{Å})$ | Proportional to conformational spread | `Viridis`, `Cividis`, `BlueWhiteRed` |

### 2.3 3D NOE / PRE Distance Restraint & Violation Cylinders (`< 2.0 ms` at `1,500` Cylinders)
`AtomCoordLookup::evaluate_restraints` indexes all conformer coordinates (`0..M`) and expands degenerate prochiral/methyl/aromatic wildcards & pseudoatoms (`HB*`, `HG1*`, `HG2*`, `HD1*`, `HD2*`, `HE*`, `HZ*`, `QB`, `QG*`, `QD*`, `QE*`, `MG*`, `MD*`, `QR`, falling back to heavy parent atoms `CB`/`CG*`/`CD*`/`CE*`/`CZ` only when explicit protons are absent) so filtering or toggling `1,500+` NOE/PRE restraints completes in `< 2.0 ms`. Unresolvable restraint endpoints (unknown atom identifiers or chemically absent atoms such as `HB` on `GLY`) are **never** substituted with backbone `CA` coordinates (`0` synthetic fallback cylinders); instead, they are skipped during cylinder emission and tallied in `RestraintOverlayBuild::unresolved_count`, which `Mol3dViewerPanel` displays directly in the viewport overlay summary:

- **Ensemble $\langle r^{-6} \rangle^{-1/6}$ Effective Distance**: Across all valid conformers $m \in \{0, \dots, M-1\}$ and expanded intra-conformer atom pairs $(a \in A_{1,m}, b \in A_{2,m})$,
  $$r_{\text{eff}} = \left(\frac{1}{M_{\text{valid}}}\sum_{m}\sum_{a \in A_{1,m}}\sum_{b \in A_{2,m}} \left\|\mathbf{r}^{(m)}_a - \mathbf{r}^{(m)}_b\right\|^{-6}\right)^{-1/6}$$
- **Upper / Lower Bound Violation Classification**:
  $$\Delta_{\text{viol}} = \max\!\bigl(0,\ r_{\text{eff}} - d_{\text{upper}},\ d_{\text{lower}} - r_{\text{eff}}\bigr)$$
  - **Satisfied ($\Delta_{\text{viol}} \le 0.0\text{ Å}$)**: Emerald green cylinder (`[0.16, 0.84, 0.46, 0.90]`), thin radius (`0.075 Å`).
  - **Minor Violation ($0.0\text{ Å} < \Delta_{\text{viol}} < 0.5\text{ Å}$)**: Warm amber cylinder (`[0.96, 0.68, 0.14, 0.95]`), medium radius (`0.105 Å`).
  - **Severe Violation ($\Delta_{\text{viol}} \ge 0.5\text{ Å}$)**: Crimson red cylinder (`[0.94, 0.22, 0.24, 0.98]`), thick radius (`0.135 Å`).
- Interactive filters (`RestraintFilterConfig`, solid cylinders by default with toggleable `dashed_cylinders`) toggle all restraints, satisfied/minor/severe bands, or active-residue-only contacts. Whitespace-delimited restraint TSV ingestion (`parse_distance_restraints_tsv`) skips a leading column-header row only on the first non-comment line when `seq_id_1`, `seq_id_2`, and `upper_A` are all non-numeric labels, ensuring malformed letter-prefixed data rows (`A10 H 14 HA 4.50`) return a structured parse error instead of being silently dropped.
