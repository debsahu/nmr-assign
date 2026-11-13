# 04. Structure Calculation, Non-Homologous Torsion Prediction & Residual Dipolar Coupling Refinement

## Provenance & Clean-Room Statement

Original clean-room specification and mathematical design. Zero source code, macros, parameter files, residue libraries, fragment databases, or binaries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) structure-calculation or restraint-generation package were consulted. All polymer kinematics, ambiguous distance restraint (`ADR`) annealing protocols, chemical-shift fragment torsion algorithms, homolog-firewall filters, Saupe alignment tensor decompositions, and GPU torsion-refinement kernels are derived strictly from peer-reviewed literature and open `CC0` public `wwPDB` / `BMRB` data standards:

1. **Torsion-angle polymer kinematics & L-BFGS / Langevin folding (`noesy.rs`, `structure.rs`, `dihedrals.rs`)**:
   - Engh, R. A. & Huber, R. (*Acta Crystallogr. A* **1991**, 47, 392–400) — standard covalent peptide bond lengths and interior bond angles.
   - Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (*J. Comput. Chem.* **2005**, 26, 1063–1068) — Natural Extension Reference Frame (`NeRF`) forward internal-to-Cartesian placement.
   - Abe, H., Braun, W., Noguti, T. & Go, N. (*Comput. Chem.* **1984**, 8, 239–247) — exact $O(L)$ backward suffix-sum torque identity for analytical torsion-angle derivatives $\partial E / \partial \phi_i$ and $\partial E / \partial \psi_i$.
   - Nilges, M. (*J. Mol. Biol.* **1995**, 245, 645–660) & Herrmann, T., Güntert, P. & Wüthrich, K. (*J. Mol. Biol.* **2002**, 319, 209–227) — $r^{-6}$-summed ambiguous distance restraints $D_{\text{eff}} = (\sum_k d_k^{-6})^{-1/6}$, network anchoring, and iterative ensemble-guided restraint reweighting.
   - Nocedal, J. (*Math. Comp.* **1980**, 35, 773–782) — two-loop limited-memory quasi-Newton (`L-BFGS`) minimization.
   - Kabsch, W. (*Acta Crystallogr. A* **1976**, 32, 922–923) & Coutsias, E. A., Seok, C. & Dill, K. A. (*J. Comput. Chem.* **2004**, 25, 1849–1857) — quaternion eigensystem proper rigid-body superposition (`det(R) = +1`).
2. **Database-driven backbone & sidechain torsion prediction + 5-Layer Homolog Firewall (`torsion_db.rs`, `data/torsion_db/`)**:
   - Shen, Y., Vernon, R., Baker, D. & Bax, A. (*Proc. Natl. Acad. Sci. U.S.A.* **2008**, 105, 4685–4690) & Shen, Y. & Bax, A. (*J. Biomol. NMR* **2013**, 56, 227–241) — chemical-shift + sequence-similarity fragment matching for $(\phi, \psi, \chi_1)$ torsion prediction (published method comparison against `TALOS-N`; `TALOS` software and databases are never read or used).
   - Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (*J. Biomol. NMR* **1995**, 5, 67–81) & Wang, Y. & Wishart, D. S. (*J. Biomol. NMR* **2005**, 31, 143–148) — random-coil secondary shifts and unlabelled chemical-shift referencing offset correction.
   - Berjanskii, M. V. & Wishart, D. S. (*J. Am. Chem. Soc.* **2005**, 127, 14970–14971) — Random Coil Index (`RCI`) and chemical-shift order parameter $S^2_{\text{RCI}}$.
   - Henikoff, S. & Henikoff, J. G. (*Proc. Natl. Acad. Sci. U.S.A.* **1992**, 89, 10915–10919) — `BLOSUM62` amino-acid substitution matrix.
   - Lovell, S. C. et al. (*Proteins* **2003**, 50, 437–450) — residue-class (`General`, `Gly`, `Pro`, `Pre-Pro`) Ramachandran basin geometry and outlier boundaries.
   - Zhang, Y. & Skolnick, J. (*Proteins* **2004**, 57, 702–710) — length-normalized `TM-score` structural similarity metric with $d_0(L) = 1.24\sqrt[3]{L - 15} - 1.8$.
3. **Residual Dipolar Couplings (`RDC`), Saupe Tensor SVD & Batched WebGPU Refinement (`rdc.rs`, `rdc_gpu.rs`, `rdc_refine.wgsl`)**:
   - Tjandra, N. & Bax, A. (*Science* **1997**, 278, 1111–1114) — dipolar couplings in weakly aligned liquid-crystalline media.
   - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* **1998**, 120, 6836–6837) — bicelle alignment validation and the Cornilescu quality factor $Q$.
   - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (*J. Magn. Reson.* **1999**, 138, 334–342) — exact linear 5-parameter traceless symmetric Saupe order matrix singular value decomposition (`SVD`).
   - Clore, G. M., Gronenborn, A. M. & Tjandra, N. (*J. Magn. Reson.* **1998**, 131, 159–162) — axial ($D_a$) and rhombic ($R$) alignment tensor parameterization.

---

## 1. Torsion-Space Polymer Kinematics & Iterative ADR Folding (`noesy.rs`, `structure.rs`)

### 1.1 Rigid Covalent Geometry & `NeRF` Reconstruction
Every conformer of length $L$ is parameterized by internal backbone torsions $(\phi_i, \psi_i, \omega_i)_{i=0}^{L-1}$ with rigid Engh & Huber covalent bond lengths ($r_{\text{N–C}^\alpha} = 1.458\text{ \AA}$, $r_{\text{C}^\alpha\text{–C}'} = 1.525\text{ \AA}$, $r_{\text{C}'\text{–N}} = 1.329\text{ \AA}$, $r_{\text{C}'\text{–O}} = 1.231\text{ \AA}$, $r_{\text{N–H}} = 1.010\text{ \AA}$) and interior bond angles. Chiral L-amino acid $\text{C}^\beta$, $\text{H}^\alpha$, and $\text{Q}^\beta$ pseudoatom coordinates are reconstructed deterministically from $(\text{N}_i, \text{C}^\alpha_i, \text{C}'_i)$ so they rotate rigidly with segment $2i + 1$ around the $\text{N}_i \to \text{C}^\alpha_i$ ($\phi_i$) axis.

### 1.2 Exact $O(L)$ Analytical Torsion Gradients
Given Cartesian energy gradients $\mathbf{g}_a = \partial E / \partial \mathbf{r}_a$ accumulated onto rigid polymer segments $s \in \{0, \dots, 2L\}$ ($\text{N}_i, \text{H}_i, \text{C}^\alpha_i \in 2i$; $\text{C}'_i, \text{H}^\alpha_i, \text{C}^\beta_i \in 2i + 1$; $\text{O}_i \in 2i + 2$), the backward suffix sums

$$\mathbf{F}_t = \sum_{s > t} \sum_{a \in s} \mathbf{g}_a, \qquad \mathbf{T}_t = \sum_{s > t} \sum_{a \in s} (\mathbf{r}_a \times \mathbf{g}_a)$$

yield exact analytical torsion derivatives in $O(L)$ operations:

$$\frac{\partial E}{\partial \theta_t} = \hat{\mathbf{e}}_t \cdot \left(\mathbf{T}_t - \mathbf{p}_t \times \mathbf{F}_t\right)$$

where $(\hat{\mathbf{e}}_{2i}, \mathbf{p}_{2i}) = (\widehat{\mathbf{r}_{\text{C}^\alpha_i} - \mathbf{r}_{\text{N}_i}}, \mathbf{r}_{\text{C}^\alpha_i})$ for $\phi_i$ and $(\hat{\mathbf{e}}_{2i+1}, \mathbf{p}_{2i+1}) = (\widehat{\mathbf{r}_{\text{C}'_i} - \mathbf{r}_{\text{C}^\alpha_i}}, \mathbf{r}_{\text{C}'_i})$ for $\psi_i$. Subtracting carbonyl oxygen $\text{O}_i$'s local force and torque $(\mathbf{f}_{\text{O}_i}, \mathbf{r}_{\text{O}_i} \times \mathbf{f}_{\text{O}_i})$ from $(\mathbf{F}_{2i+1}, \mathbf{T}_{2i+1})$ simultaneously yields exact $O(L)$ peptide-bond $\partial E / \partial \omega_i$ derivatives around $(\hat{\mathbf{e}}_{\omega_i}, \mathbf{p}_{\omega_i}) = (\widehat{\mathbf{r}_{\text{N}_{i+1}} - \mathbf{r}_{\text{C}'_i}}, \mathbf{r}_{\text{N}_{i+1}})$.

### 1.3 Multi-Cycle Ambiguous NOESY Folding
1. **Initial matching & network anchoring**: 3D ${}^{15}\text{N}$-edited and ${}^{13}\text{C}$-edited NOESY crosspeaks are matched to assigned $(\text{H}^{\text{N}}, \text{H}^\alpha, \text{Q}^\beta)$ resonances and scored by covalent, sequential, reciprocal, and triangle contact consistency.
2. **Three-stage torsion annealing**: Parallel replicas undergo high-temperature soft-core Langevin dynamics (Stage 1), steric excluded-volume inflation and cooling (Stage 2), and full-potential two-loop `L-BFGS` minimization (Stage 3).
3. **Iterative `<d^-6>` ADR filtering**: Across refinement cycles, candidate weights are updated from ensemble distances via $w_k \propto \langle d_k^{-6} \rangle$, promoting dominant contacts ($\ge 85\%$ weight) to unambiguous distance restraints.

---

## 2. Post-Assignment Fragment Torsion Predictor (`torsion_db.rs`, `data/torsion_db/`)

### 2.1 Strictly Post-Assignment Boundary
`nmr_core::torsion_db` operates strictly downstream of resonance assignment (`Reference` shifts in $\to$ `TorsionPredictionReport` / `DihedralRestraint`s out). Zero PDB coordinates, fragment tables, or homology features ever enter the upstream chemical-shift assignment solver (`crates/nmr-core/src/solver.rs`).

### 2.2 Non-Redundant `CC0` Fragment Corpus (`data/torsion_db/torsion_corpus.tsv`)
`data/torsion_db/torsion_corpus.tsv` stores a compact, non-redundant `CC0` corpus of `129` monomeric globular protein chains (`< 28%` pairwise sequence identity) paired between experimental `BMRB` chemical shifts (`HN`, `N`, `CA`, `CB`, `C'`, `HA`) and high-quality `wwPDB` structures across all major fold classes (`CATH` Class 1 Mainly Alpha, Class 2 Mainly Beta, Class 3 Alpha/Beta, and Class 4 Small/Special folds), complete with `UniProt` accessions, `CATH` superfamily & topology (`C.A.T`) fold tags, native $(\phi, \psi, \chi_1)$ angles, $\text{C}^\alpha$ coordinates, and `SHA-256` provenance digests in `data/torsion_db/PROVENANCE.md`.

### 2.3 Unlabelled Global Chemical-Shift Offset Estimation
Before fragment scoring, `estimate_global_shift_offsets` separates spectrometer referencing offsets (e.g., `TMS` vs `DSS` $\approx \pm 2\text{ ppm}$ on ${}^{13}\text{C}$, liquid $\text{NH}_3$ offsets on ${}^{15}\text{N}$) from secondary-structure chemical-shift polarization using ZERO ground-truth labels:
1. A 3-state `(Coil, Helix, Sheet)` expectation-maximization (`EM`) decomposition exploits the opposite signs of $\Delta\delta_{\text{C}^\alpha}$ and $\Delta\delta_{\text{C}^\beta}$ in $\alpha$-helices versus $\beta$-sheets while fitting a shared uniform reference offset $(\delta_0^{\text{H}}, \delta_0^{\text{N}}, \delta_0^{\text{C}^\alpha}, \delta_0^{\text{C}^\beta}, \delta_0^{\text{C}'}, \delta_0^{\text{H}^\alpha})$.
2. Per-nucleus soft deadzones ($0.32\,\sigma_k$, with $\boldsymbol{\sigma} = [0.42, 3.20, 1.35, 1.45, 1.25, 0.28]\text{ ppm}$ for $(\text{H}, \text{N}, \text{C}^\alpha, \text{C}^\beta, \text{C}', \text{H}^\alpha)$) suppress spurious drift on small unbalanced single-motif domains while accurately recovering genuine mis-referencing offsets.

### 2.4 Hybrid 3-mer + 9-mer Fragment Scoring & Residue Classification
1. **Flexibility gate (`RCI` $S^2$)**: Smoothed 6-nucleus Berjanskii–Wishart Random Coil Index (`RCI`) and order parameter $S^2_{\text{RCI}}$ flag intrinsically disordered and flexible loop residues ($S^2_{\text{RCI}} < 0.65$) as `Dynamic` prior to restraint emission.
2. **Peptide-bond-aware 3-mer + 9-mer dissimilarity**: Every contiguous database window matching exact `Gly` / `Pro` / `Pre-Pro` center identity is scored by combining secondary-shift $z$-squared residuals across $(\Delta\delta_{\text{H}}, \Delta\delta_{\text{N}}, \Delta\delta_{\text{C}^\alpha}, \Delta\delta_{\text{C}^\beta}, \Delta\delta_{\text{C}'}, \Delta\delta_{\text{H}^\alpha})$, $\Delta\delta_{\text{C}^\alpha} - \Delta\delta_{\text{C}^\beta}$ polarization differences, and `BLOSUM62` log-odds substitution penalties over both the local 3-mer (`[i-1, i, i+1]`, position weights `[1.15, 2.50, 1.15]`) and flanking 9-mer (`[-4..-2, +2..+4]`, context weights `[0.4, 0.6, 1.0, 1.0, 0.6, 0.4]`, outer scale `1.15`) windows.
3. **Circular cluster consensus**: From the top $K = 10$ fragment matches, residue $i$ is classified as:
   - `Good`: $S^2_{\text{RCI}} \ge 0.65$, dominant $40^\circ$-radius $(\phi, \psi)$ cluster contains $\ge 7/10$ (`70%`) matches, combined Boltzmann-weighted circular standard deviation $\sqrt{\sigma_\phi^2 + \sigma_\psi^2} \le 25^\circ$, and circular mean $(\hat{\phi}_i, \hat{\psi}_i)$ lies strictly inside Lovell allowed Ramachandran basins. When $\ge 7$ cluster members agree within $36^\circ$ on $\chi_1$ (`non-Gly/Ala`), $\hat{\chi}_1 \pm \sigma_{\chi_1}$ is also emitted.
   - `Dynamic`: $S^2_{\text{RCI}} < 0.65$ (flexible loop / terminus / IDP).
   - `Ambiguous`: $S^2_{\text{RCI}} \ge 0.65$ with multimodal or wide fragment spread (abstains cleanly rather than emitting a wrong basin restraint).
   - `Unassigned`: chain terminus (`i = 0` or `i = L - 1`) or fewer than 2 observed shifts at residue $i$.

---

## 3. 5-Layer `HomologFirewall` (`Q5`)

Every benchmark evaluation (`predict_torsions_with_firewall`) enforces a deterministic 5-layer non-homologous firewall (`HomologFirewallQuery`) between the query protein and `TorsionDb`:

1. **Layer 1 — Protein / `UniProt` Family Exclusion**: Excludes every template matching any query `PDB` ID, `BMRB` ID, `UniProt` primary accession, or protein family / superfamily ID (eliminating point mutants, orthologs, and complexes).
2. **Layer 2 — Fold Topology Exclusion**: Excludes every template sharing the same `ECOD` H-group or `CATH` `C.A.T` topology fold group (`topology_id`, e.g., `3.10.20` $\beta$-grasp ubiquitin-like fold).
3. **Layer 3 — Global, Local & K-mer Sequence Exclusion**: Excludes any template with $> 25\%$ sequence identity over an aligned span covering $\ge 50\%$ of **either** the query length **or** the template length under **both** global Needleman–Wunsch (`BLOSUM62`, affine gap `-11/-1`) and local Smith–Waterman alignment, **plus** any template sharing a 7-mer window with $\le 1$ mismatch ($\ge 6/7$ identical residues).
4. **Layer 4 — 3D Structural `TM-score` Exclusion**: Whenever query reference $\text{C}^\alpha$ coordinates exist in benchmark evaluation, computes exact clean-room Zhang–Skolnick `TM-score` (`tm_score_ca` with multi-seed fragment/sliding-window superposition and iterative $d_0(L)$ refinement, normalized by $\min(L_{\text{query}}, L_{\text{template}})$) and excludes any template with $\text{TM-score} \ge 0.50$.
5. **Layer 5 — Cryptographic Audit & Stratified Leakage Check**: Computes a deterministic `FIPS 180-4` `SHA-256` hex digest (`exclusion_list_sha256`) over the sorted `chain_key:reasons` manifest for every run, records `max_retained_seq_identity`, and stratifies circular $(\phi, \psi)$ MAE across nearest-neighbor retained sequence identity bins (`[0–10%)`, `[10–15%)`, `[15–20%)`, `[20–25%]`) to verify zero homolog leakage slope.

---

## 4. Residual Dipolar Couplings (`RDC`), Saupe SVD & WebGPU Refinement (`rdc.rs`, `rdc_gpu.rs`)

### 4.1 Multi-Medium Dipolar Coupling Physics
For internuclear unit vector $\hat{\mathbf{u}} = (\mathbf{r}_B - \mathbf{r}_A) / \|\mathbf{r}_B - \mathbf{r}_A\|$ in alignment medium $m$, the secular dipolar coupling is linear in the traceless symmetric $3 \times 3$ Saupe order matrix $\mathbf{S}^{(m)}$ ($\text{Tr}(\mathbf{S}^{(m)}) = 0$, $\mathbf{S}^{(m)} = (\mathbf{S}^{(m)})^\top$):

$$D_{\text{calc}}(\hat{\mathbf{u}}; \mathbf{S}^{(m)}) = D_{\max}^{AB}\,\hat{\mathbf{u}}^\top \mathbf{S}^{(m)} \hat{\mathbf{u}}$$

Supported bond types (`RdcBondType`) and standard Ottiger & Bax (*J. Am. Chem. Soc.* **1998**, 120, 12334–12341) libration-adjusted dipolar prefactors $D_{\max}^{AB}$:
- `1D_NH` (`N`–`H`, $r_{\text{eff}} = 1.040\text{ \AA}$, $D_{\max}^{\text{NH}} = +21\,655.82\text{ Hz}$)
- `1D_CAHA` (`CA`–`HA`, $r_{\text{eff}} = 1.105\text{ \AA}$, $D_{\max}^{\text{CAHA}} = +44\,782.43\text{ Hz}$)
- `1D_CACO` (`CA`–`C`, $r_{\text{eff}} = 1.525\text{ \AA}$, $D_{\max}^{\text{CACO}} = +4\,284.75\text{ Hz}$)
- `1D_NC'` (`C_{i-1}`–`N_i`, $r_{\text{eff}} = 1.329\text{ \AA}$, $D_{\max}^{\text{NC}'} = +2\,610.02\text{ Hz}$)
- `1D_HNC'` (`C_{i-1}`–`H_i`, $r_{\text{eff}} = 2.100\text{ \AA}$, $D_{\max}^{\text{HNC}'} = +6\,525.10\text{ Hz}$, or explicit caller `d_max_hz`)

### 4.2 Exact 5-Parameter Losonczi Saupe Tensor SVD & Safety Invariants
Substituting $S_{xx} = -(S_{yy} + S_{zz})$ expresses $D_{\text{calc}}$ linearly in the 5 independent components $\mathbf{s} = (S_{yy}, S_{zz}, S_{xy}, S_{xz}, S_{yz})^\top$:

$$\frac{D_{\text{calc},k}}{\sigma_k} = \frac{D_{\max,k}}{\sigma_k}\Big[(u_y^2 - u_x^2)S_{yy} + (u_z^2 - u_x^2)S_{zz} + 2u_x u_y S_{xy} + 2u_x u_z S_{xz} + 2u_y u_z S_{yz}\Big]$$

`fit_saupe_tensor_svd` solves $\min_{\mathbf{s}} \|\mathbf{A}\mathbf{s} - \mathbf{b}\|_2^2$ over working (`!is_free`) couplings via one-sided Hestenes–Jacobi thin SVD (`m x 5`, without forming $\mathbf{A}^\top\mathbf{A}$), enforcing strict numerical safety invariants:
1. Requires $n_{\text{work}} \ge 6$ finite RDCs per medium with $\sigma_k > 0$ and finite $D_{\text{obs},k}, D_{\max,k}$.
2. Rejects collinear or coplanar bond vector geometries whenever $\kappa(\mathbf{A}) = \sigma_{\max} / \sigma_{\min} > 10^8$ (`Err`).
3. Diagonalizes $\mathbf{S}^{(m)} = \mathbf{R}\,\text{diag}(S_{xx}^{\text{PAS}}, S_{yy}^{\text{PAS}}, S_{zz}^{\text{PAS}})\,\mathbf{R}^\top$ in canonical principal-axis order $|S_{zz}^{\text{PAS}}| \ge |S_{yy}^{\text{PAS}}| \ge |S_{xx}^{\text{PAS}}|$, guaranteeing $D_a^{\text{NH}} = \frac{1}{2} D_{\max}^{\text{NH}} S_{zz}^{\text{PAS}}$, $0 \le R = \frac{S_{xx}^{\text{PAS}} - S_{yy}^{\text{PAS}}}{S_{zz}^{\text{PAS}}} \le \frac{2}{3}$, and canonical right-handed `SO(3)` `ZYZ` Euler angles $(\alpha, \beta, \gamma)$.
4. Computes normalized Cornilescu $Q$ (`q_work`) and held-out cross-validated $Q_{\text{free}}$ (`q_free`):

$$Q = \sqrt{\frac{\sum_k (\hat{D}_{\text{obs},k} - \hat{D}_{\text{calc},k})^2}{\sum_k \hat{D}_{\text{obs},k}^2}}, \qquad \hat{D}_k = D_k \cdot \frac{D_{\max}^{\text{NH}}}{D_{\max,k}}$$

returning `f64::NAN` whenever $\sum_k \hat{D}_{\text{obs},k}^2 \le 10^{-15}$ (and treating `NaN` as strictly worst in `rdc_q_is_better`).

### 4.3 Variable-Projection Torsion Refinement & Batched WebGPU Kernel
Under variable projection $\tilde{E}_{\text{RDC}}(\boldsymbol{\phi}, \boldsymbol{\psi}) = \min_{\{\mathbf{S}^{(m)}\}} \frac{1}{2} w_{\text{RDC}} \sum_{m, k \in \mathcal{W}_m} \left(\frac{D_{\text{calc},k}(\mathbf{S}^{(m)}) - D_{\text{obs},k}}{\sigma_k}\right)^2$, the Envelope Theorem ($\partial E / \partial \mathbf{S}^{(m)}\big|_{\mathbf{S}^{*(m)}} = \mathbf{0}$) implies exact Cartesian forces at optimal $\mathbf{S}^{*(m)}$:

$$\frac{\partial \tilde{E}_{\text{RDC}}}{\partial \mathbf{r}_{B,k}} = w_{\text{RDC}} \frac{D_{\text{calc},k} - D_{\text{obs},k}}{\sigma_k^2} \cdot \frac{2 D_{\max,k}}{\|\mathbf{v}_k\|} \left(\mathbf{S}^{*(m)}\hat{\mathbf{u}}_k - (\hat{\mathbf{u}}_k^\top \mathbf{S}^{*(m)} \hat{\mathbf{u}}_k)\hat{\mathbf{u}}_k\right), \qquad \frac{\partial \tilde{E}_{\text{RDC}}}{\partial \mathbf{r}_{A,k}} = -\frac{\partial \tilde{E}_{\text{RDC}}}{\partial \mathbf{r}_{B,k}}$$

- **CPU (`structure::refine_ensemble_with_rdcs`)**: Multi-threaded staged refinement combining full-circle compensating peptide-plane crankshaft proposals (including exact $180^\circ$ peptide-plane inversions and canonical Ramachandran turn-basin sweeps), soft-core Langevin annealing, bounded $C^1$ log-Lorentzian flat-bottom dihedral tails (preventing rare database basin-mirror outliers from overriding multi-bond RDCs), and two-loop `L-BFGS` in $(\phi, \psi, \omega)$ torsion space with harmonic Engh–Huber / Cornilescu (*JACS* **1998**) peptide-planarity restraints ($\sigma_\omega \approx 3.7^\circ$, clamped within $\pm 7.5^\circ$ of trans/cis equilibrium) over `!is_free` couplings only. In multi-cycle ADR structure calculation (`calculate_structure_with_rdcs_and_dihedrals`), initial NOESY candidate lists are re-weighted non-destructively against RDC-refined coordinates at every cycle so contacts prematurely pruned at coarse Cycle-0 resolution are recovered.
- **WebGPU (`nmr_gpu::rdc_gpu` + `shaders/rdc_refine.wgsl`)**: Evaluates `NeRF` backbone + chiral $\text{H}^\alpha/\text{C}^\beta$ forward kinematics, multi-medium harmonic RDC residuals, rigid-segment forces/torques, and $O(L)$ backward suffix-sum $(\partial E / \partial \phi_i, \partial E / \partial \psi_i)$ gradients across an entire multi-conformer ensemble in a single compute dispatch (`< 1e-5` relative parity vs `f64` CPU reference, compiling cleanly on both native and `wasm32-unknown-unknown`).
- **Interchange & Reporting (`nef.rs`, `report.rs`)**: Full `NEF v1.1` `nef_rdc_restraint_list` and `PHI`/`PSI`/`CHI1` `nef_dihedral_restraint_list` read/write round-trip, plus self-contained XML-escaped `D_calc` vs `D_obs` SVG scatter and per-medium `(D_a, R, Q, Q_free)` HTML summary tables.
