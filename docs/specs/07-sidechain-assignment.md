# 07. Sidechain & Aromatic Resonance Assignment

> **Provenance (Clean-Room Implementation)**
> All topologies, pulse-sequence magnetization-transfer models, chemical-shift likelihoods, and assignment solvers specified here and implemented in `crates/nmr-core/src/{topology,sidechain,aromatic}.rs` derive strictly from peer-reviewed literature, IUPAC-IUBMB-IUPAB biochemical nomenclature, the public PDB Chemical Component Dictionary, and public Biological Magnetic Resonance Data Bank (BMRB) statistics:
> - **Aliphatic relay experiments (`(H)CC(CO)NH-TOCSY`, `H(CCCO)NH-TOCSY`, `HCCH-TOCSY` / `HCCH-COSY`, `13C-HSQC`)**:
>   - Grzesiek, Anglister & Bax, *J. Magn. Reson. B* **101**, 114–119 (1993) — isotropic `13C` mixing correlating aliphatic sidechain carbons and protons of residue `i - 1` with `15N(i)-1HN(i)`.
>   - Montelione, Lyons, Emerson, Tashiro & Farmer, *J. Am. Chem. Soc.* **114**, 10974–10975 (1992) — `3D (H)CC(CO)NH-TOCSY` & `H(CCCO)NH-TOCSY`.
>   - Bax, Clore & Gronenborn, *J. Magn. Reson.* **88**, 425–431 (1990); Kay, Ikura & Bax, *J. Am. Chem. Soc.* **112**, 888–889 (1990) — `3D HCCH-TOCSY` & `HCCH-COSY` intra-residue `1H-13C-13C-1H` coherence transfer.
> - **Aromatic `13CB`-to-ring correlation experiments (`2D (HB)CB(CGCD)HD` & `2D (HB)CB(CGCDCE)HE`)**:
>   - Yamazaki, Forman-Kay & Kay, *J. Am. Chem. Soc.* **115**, 11054–11055 (1993) — two-dimensional scalar `1J_CC` relay correlating aliphatic `13CB` with aromatic `1HD*` and `1HE*` ring protons across quaternary `13CG`/`13CD2`.
> - **Histidine imidazole `15ND1` / `15NE2` tautomeric & protonation state determination (`2D long-range 1H-15N HMQC`)**:
>   - Pelton, Torchia, Meadow & Roseman, *Protein Sci.* **2**, 543–558 (1993) — tautomeric (`Nε2-H` $\tau$ vs `Nδ1-H` $\pi$) and imidazolium (`+1` doubly protonated) classification from long-range `2J_NH` / `3J_NH` `1H-15N` HMQC chemical shifts and cross-peak intensity patterns.
> - **Aromatic `1H-13C` HSQC / constant-time HSQC (`CT-HSQC`) & intra-ring `HCCH-TOCSY`**:
>   - Bax, Clore & Gronenborn, *J. Magn. Reson.* **88**, 425–431 (1990); Torizawa, Ono, Terauchi & Kainosho, *J. Am. Chem. Soc.* **127**, 12620–12626 (2005).
> - **IUPAC nomenclature & empirical BMRB chemical-shift priors**:
>   - Markley et al., *J. Mol. Biol.* **280**, 933–952 (1998); Ulrich et al., *Nucleic Acids Res.* **36**, D402–D408 (2008).
> - **Exact polynomial-time bipartite matching (Hungarian / Kuhn–Munkres)**:
>   - Kuhn, *Naval Res. Logist. Quart.* **2**, 83–97 (1955); Munkres, *J. Soc. Ind. Appl. Math.* **5**, 32–38 (1957).

---

## 1. Scope & Architectural Overview

Extending sequential backbone assignment (`H`, `N`, `CA`, `CB`, `C`; see `03-assignment.md`) to complete protein sidechains requires two complementary assignment engines in `crates/nmr-core`:

1. **Aliphatic Sidechain Assignment (`crates/nmr-core/src/sidechain.rs`)**:
   - Assigns aliphatic `13C` (`CA`, `CB`, `CG*`, `CD*`, `CE*`) and `1H` (`HA*`, `HB*`, `HG*`, `HD*`, `HE*`) resonances across all 20 standard amino acids, including `Pro` (`CD/HD2/HD3`, `CG/HG2/HG3`, `CB/HB2/HB3`, `CA/HA` lacking an amide `HN`), methyl groups (`Ala`, `Val`, `Leu`, `Ile`, `Thr`, `Met`), and polar sidechain `NH2` carboxyamide groups (`Asn ND2/HD21/HD22`, `Gln NE2/HE21/HE22`).
   - Supports bidirectional feedback into the `(U, P)` backbone scoring tables (`solve_joint_backbone_sidechain`) so sidechain `CC(CO)NH` / `HCCH-TOCSY` spin-system evidence disambiguates weak or degenerate `Ser`/`Thr`/`Ala`/`Gly`/`Ile`/`Val`/`Leu` backbone assignments.
2. **Aromatic Ring Spin-System Assignment (`crates/nmr-core/src/aromatic.rs`)**:
   - Assigns aromatic `(1H, 13C, 15N)` ring resonances for `Phe` (`CD*/HD*`, `CE*/HE*`, `CZ/HZ`), `Tyr` (`CD*/HD*`, `CE*/HE*`), `Trp` (`CD1/HD1`, `NE1/HE1`, `CE3/HE3`, `CZ3/HZ3`, `CH2/HH2`, `CZ2/HZ2`), and `His` (`CD2/HD2`, `CE1/HE1`, `ND1`, `NE2`) when quaternary `13CG` carbons break standard aliphatic `HCCH-TOCSY` propagation.
   - Determines `His` imidazole tautomeric and protonation states (`Ne2H`, `Nd1H`, `DoublyProtonated`) from long-range `2J_NH` `1H-15N` HMQC and `13CD2`/`13CB` chemical shifts, and solves global 1-to-1 ring-to-backbone matching via exact Kuhn–Munkres optimization with leave-one-out Boltzmann confidence calibration.

---

## 2. Covalent IUPAC Topologies & Empirical BMRB Priors (`topology.rs`)

Every standard amino acid (`ALA`..`VAL`) is represented as an immutable covalent graph `ResidueTopology`:

- **Atoms (`AtomSpec`)**: IUPAC atom name, atomic element (`H`, `C`, `N`, `O`, `S`), hybridization/multiplicity (`CH`, `CH2`, `CH3`, quaternary `C`, `NH`, `NH2`, `NH3+`), `CarbonRegion` (`AliphaticCaConnected`, `Aromatic`, `Carbonyl`), attached heavy atom, and ring-flip equivalence flag (`symmetric_equivalent` for fast-flipping `Phe`/`Tyr` `(CD1, CD2)` and `(CE1, CE2)` pairs).
- **Prochiral & Isopropyl Pairs (`ProchiralGroup`)**: explicit stereochemical pairs (`HB2`/`HB3`, `HG12`/`HG13`, `Val CG1/CG2`, `Leu CD1/CD2`, `Asn HD21/HD22`, `Gln HE21/HE22`, `Phe`/`Tyr` ring-flip pairs).
- **Empirical BMRB Priors (`SidechainPriors`)**: per-`(residue, atom)` Gaussian `(mean_ppm, sd_ppm)` parameters derived from public BMRB statistics, evaluated as log-likelihood ratios against region-specific kernel mixture background models:
  $$\text{LLR}_{\text{prior}}(\delta \mid r, a) = \max\!\Big(-8.0,\; \ln \mathcal{N}(\delta \mid \mu_{r,a}, \sigma_{r,a}^2) - \ln p_{\text{bg}}(\delta \mid \text{region}(a))\Big)$$

Key discriminating aromatic BMRB chemical-shift priors include:

| Residue | Atom Pair | $\mu(^{13}\text{C})$ (ppm) | $\sigma(^{13}\text{C})$ | $\mu(^{1}\text{H})$ (ppm) | $\sigma(^{1}\text{H})$ | Diagnostic Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `PHE` | `CD* / HD*` | 131.5 | 1.6 | 7.21 | 0.28 | Ortho CH (`2x` intensity under fast ring flip) |
| `PHE` | `CE* / HE*` | 130.7 | 1.6 | 7.28 | 0.28 | Meta CH (`~13 ppm` downfield of `TYR CE*`) |
| `PHE` | `CZ / HZ` | 129.2 | 1.8 | 7.22 | 0.32 | Para CH (`1x` intensity; absent in `TYR`) |
| `TYR` | `CD* / HD*` | 132.6 | 1.6 | 7.08 | 0.28 | Ortho CH (`2x` intensity) |
| `TYR` | `CE* / HE*` | 117.9 | 1.7 | 6.80 | 0.26 | Ortho-to-OH upfield shield (`~117.9 ppm`) |
| `TRP` | `CD1 / HD1` | 126.5 | 2.0 | 7.24 | 0.32 | Pyrrole CH coupled to `CB` (`HBCBCGCDHD`) & `HE1` |
| `TRP` | `NE1 / HE1` | 129.4 ($^{15}\text{N}$) | 2.3 | 10.12 | 0.52 | Indole `15N-1H` downfield anchor |
| `TRP` | `CE3 / HE3` | 120.5 | 1.9 | 7.56 | 0.34 | Benzenoid entry CH coupled to `CB` (`HBCBCGCDCEHE`) |
| `TRP` | `CZ3 / HZ3` | 121.6 | 1.8 | 7.06 | 0.30 | Benzenoid relay (`CE3-CZ3-CH2-CZ2`) |
| `TRP` | `CH2 / HH2` | 123.9 | 1.8 | 7.14 | 0.30 | Benzenoid relay |
| `TRP` | `CZ2 / HZ2` | 114.4 | 1.8 | 7.42 | 0.30 | Most upfield aromatic CH (`~114.4 ppm`), proximal to `HE1` |
| `HIS` | `CE1 / HE1` | 138.0 | 2.5 | 7.98 | 0.48 | Downfield C2H between `ND1` and `NE2` |
| `HIS` | `CD2 / HD2` | 119.8 | 3.2 | 7.12 | 0.35 | C4H coupled to `CB` (`HBCBCGCDHD`); tautomer-sensitive |

---

## 3. Aliphatic Sidechain Experiments & Joint `(U, P)` Solver (`sidechain.rs`)

### 3.1 Supported Aliphatic Experiments

| Experiment ID | Dimensionality & Axes | Magnetization Transfer Pathway | Sign / Multiplicity Rules |
| :--- | :--- | :--- | :--- |
| `C13HSQC` | 2D `(x = 1H, y = 13C)` | $^{1}\text{H} \xrightarrow{^{1}J_{\text{CH}}} {}^{13}\text{C} (t_1) \xrightarrow{^{1}J_{\text{CH}}} {}^{1}\text{H} (t_2)$ | Multiplicity-edited: `CH` / `CH3` `+1`, `CH2` `-1` (when enabled) |
| `CCCONH` | 3D `(x = HN_i, y = N_i, z = C_{ali,i-1})` | $^{1}\text{H}_{i-1} \to {}^{13}\text{C}_{i-1} \xrightarrow{\text{FLOPSY/DIPSI}} {}^{13}\text{CA}_{i-1} \to {}^{13}\text{C}'_{i-1} \to {}^{15}\text{N}_i \to {}^{1}\text{HN}_i$ | All aliphatic carbons of residue `i - 1` observed on amide `(HN_i, N_i)` |
| `HCCCONH` | 3D `(x = HN_i, y = N_i, z = H_{ali,i-1})` | $^{1}\text{H}_{i-1}(t_1) \to {}^{13}\text{C}_{i-1} \xrightarrow{\text{TOCSY}} {}^{13}\text{CA}_{i-1} \to {}^{13}\text{C}'_{i-1} \to {}^{15}\text{N}_i \to {}^{1}\text{HN}_i$ | All aliphatic protons of residue `i - 1` observed on amide `(HN_i, N_i)` |
| `HCCHTOCSY` | 3D `(x = H_attach, y = C_attach, z = H_relay)` | $^{1}\text{H}_m(t_1) \to {}^{13}\text{C}_m \xrightarrow{^{1}J_{\text{CC}}\;\text{TOCSY}} {}^{13}\text{C}_k(t_2) \to {}^{1}\text{H}_k(t_3)$ | Intra-residue aliphatic covalent component; blocked at quaternary `CG` / carbonyl `C'` / heteroatoms |
| `HCCHCOSY` | 3D `(x = H_attach, y = C_attach, z = H_adj)` | Single-step $^{1}J_{\text{CC}}$ transfer (`d_CC <= 1`) | Direct covalent neighbor discrimination (`CA-CB`, `CB-CG`, `CG-CD`, `CD-CE`) |

### 3.2 Aliphatic Graph Alignment & Backbone Feedback

1. **Anchor Harvest & `i - 1` Strip Extraction**:
   - High-confidence sequential amides at `i + 1` supply `(CCCONH, HCCCONH)` carbon and proton ladders for residue `i`.
   - `2D C13HSQC` peaks and `3D HCCH-TOCSY` strips are merged into intra-residue `(13C, 1H)` nodes connected by TOCSY relay edges.
2. **Covalent Tree Matching**:
   - Candidate `(13C, 1H)` nodes are matched to the residue's `AliphaticCaConnected` carbon tree (`CA -> CB -> CG* -> CD* -> CE*`) rooted at backbone `(CA, CB)`, maximizing the sum of BMRB `(13C, 1H)` LLRs, `CC(CO)NH` / `H(CCCO)NH` anchor hits, multiplicity sign compatibility, and intra-tree `HCCH-TOCSY` relay consistency.
   - `Pro` residues (`i`) lacking an amide proton are assigned via the `i + 1` amide's `(CCCONH, HCCCONH)` strips (`Pro CD ~ 50.3 ppm`, `CG ~ 27.2 ppm`, `CB ~ 31.8 ppm`) plus `HCCH-TOCSY` self-consistency.
3. **Joint Backbone–Sidechain Feedback (`solve_joint_backbone_sidechain`)**:
   - For each spin system `s` representing amide `i`, the `(i - 1)` `CC(CO)NH` carbon ladder (`n_C` distinct carbons, `Gly` absence of `CB`, `Ala` methyl `CB ~ 19 ppm`, `Ser`/`Thr` `CB ~ 63–70 ppm`, `Ile` `CD1 ~ 13.5 ppm`, `Leu` `CD* ~ 24 ppm`, `Val` `CG* ~ 21 ppm`) injects calibrated LLR adjustments into sequential `U_prev[s, k - 1]` and pairwise `P`, followed by re-annealing to rescue ambiguous backbone links.

---

## 4. Aromatic Ring Spin-System Experiments & Simulators (`aromatic.rs`)

Quaternary `13CG` carbons in `Phe`, `Tyr`, `Trp`, and `His` (plus `13CD2`/`13CE2` in `Trp`) have no attached protons and exhibit distinct aromatic-to-aliphatic chemical-shift separation (`~110–140 ppm` vs `~25–45 ppm`), truncating standard aliphatic `HCCH-TOCSY` relays at `CB`. Dedicated 2D/3D aromatic experiments bridge `CB` across `CG` into the ring and link ring spin networks internally:

### 4.1 Supported Aromatic Experiments

| Experiment ID | Axes & Spectral Window | Magnetization Transfer & Observable Spin Pairs | Reference |
| :--- | :--- | :--- | :--- |
| `AROM_C13HSQC` | 2D `x = 1H (5.5..9.0 ppm)`, `y = 13C (108..142 ppm)` | $^{1}\text{H}_{\text{arom}} \leftrightarrow {}^{13}\text{C}_{\text{arom}}$ (`CT-HSQC` suppresses $^{1}J_{\text{CC}}$ splitting): `Phe` `CD*/HD*`, `CE*/HE*`, `CZ/HZ`; `Tyr` `CD*/HD*`, `CE*/HE*`; `Trp` `CD1/HD1`, `CE3/HE3`, `CZ3/HZ3`, `CH2/HH2`, `CZ2/HZ2`; `His` `CD2/HD2`, `CE1/HE1`. Fast ring flips render `CD1=CD2` and `CE1=CE2` degenerate at `2x` intensity. | Bax et al. (1990); Torizawa et al. (2005) |
| `HBCBCGCDHD` | 2D `x = 1HD (5.8..8.8 ppm)`, `y = 13CB (24..46 ppm)` | $(^{1}\text{HB}) \to {}^{13}\text{CB}(t_1) \xrightarrow{^{1}J_{\text{CB,CG}}} {}^{13}\text{CG} \xrightarrow{^{1}J_{\text{CG,CD}}} {}^{13}\text{CD} \xrightarrow{^{1}J_{\text{CH}}} {}^{1}\text{HD}(t_2)$: correlates `Phe`/`Tyr` `HD*`, `Trp` `HD1`, and `His` `HD2` directly with backbone `CB(i)`. | Yamazaki, Forman-Kay & Kay (1993) |
| `HBCBCGCDCEHE` | 2D `x = 1HE (5.8..9.0 ppm)`, `y = 13CB (24..46 ppm)` | $(^{1}\text{HB}) \to {}^{13}\text{CB}(t_1) \to {}^{13}\text{CG} \to {}^{13}\text{CD} \to {}^{13}\text{CE} \to {}^{1}\text{HE}(t_2)$: correlates `Phe`/`Tyr` `HE*`, `Trp` `HE3` (via `CB-CG-CD2-CE3-HE3`), and weak/optional `His` `HE1` with `CB(i)`. | Yamazaki, Forman-Kay & Kay (1993) |
| `AROM_HCCH_TOCSY` | 3D `x = 1H`, `y = 13C`, `z = 1H_tocsy` | Intra-ring $^{1}\text{H}\text{-}{}^{13}\text{C}\text{-}{}^{13}\text{C}\text{-}{}^{1}\text{H}$ isotropic mixing: links `(CD*,HD*) <-> (CE*,HE*) <-> (CZ,HZ)` in `Phe`, `(CD*,HD*) <-> (CE*,HE*)` in `Tyr`, and the 4-spin benzenoid sub-ring `HE3 <-> HZ3 <-> HH2 <-> HZ2` in `Trp` (`1-step` attenuation `~0.68`, `2-step` `~0.42`, `3-step` `~0.24`). | Bax et al. (1990); Torizawa et al. (2005) |
| `HIS_LR_N15HMQC` | 2D `x = 1H (6.4..9.2 ppm)`, `y = 15N (155..265 ppm)` | Long-range $^{2}J_{\text{NH}}$ / $^{3}J_{\text{NH}}$ ($\Delta \approx 22\text{ ms}$) HMQC correlating `His` `HE1` (`~7.5–8.8 ppm`) strongly to both `ND1` (`2J`) and `NE2` (`2J`), and `HD2` (`~6.8–7.6 ppm`) strongly to `NE2` (`2J`) and weakly to `ND1` (`3J`). | Pelton et al. (1993) |
| `TRP_IND_N15HSQC` | 2D `x = 1HE1 (9.2..11.5 ppm)`, `y = 15NE1 (122..137 ppm)` | One-bond $^{1}J_{\text{NH}}$ indole `15NE1-1HE1` correlation, paired with intra-ring `HE1 -> HD1` (`~2.5 Å`) and `HE1 -> HZ2` (`~2.8 Å`) NOE/relay cross-peaks bridging the pyrrole and benzenoid halves of `Trp`. | Markley et al. (1998) |

### 4.2 Volumetric Simulation & Artifact-Suppressing Peak Picking

- **`simulate_aromatic_spectra`**: renders realistic 2D `Spectrum2D` (`AROM_C13HSQC`, `HBCBCGCDHD`, `HBCBCGCDCEHE`, `HIS_LR_N15HMQC`, `TRP_IND_N15HSQC`) and 3D `Spectrum3D` (`AROM_HCCH_TOCSY`) grids with physical Gaussian linewidths, deterministic Box–Muller thermal Gaussian noise, chemical-shift jitter, missing-peak dropout, symmetric `2x` stoichiometry for fast-flipping `Phe`/`Tyr` ortho/meta pairs, and FT window truncation sinc sidelobes (`~3%` secondary lobes).
- **`pick_aromatic_2d` & `pick_aromatic_hcch_3d`**:
  1. Estimates global thermal noise $\hat{\sigma}$ via the exact median-absolute-deviation (`MAD / 0.67449`) estimator (`estimate_noise_mad`).
  2. Extracts strict 2D (`3x3`) or 3D (`3x3x3`) local maxima above `SNR >= snr_threshold` (`6.0` in 2D, `5.5` in 3D) with 3-point parabolic sub-bin interpolation along every axis.
  3. **Artifact & Sidelobe Suppression**: suppresses axial FT truncation sinc sidelobes (`< 7.5%` of an intense parent peak within `0.18 ppm 1H` or `1.8 ppm 13C/15N` along a shared axis) and indirect-dimension `t1` thermal ridges (`< 4.5%` of column maximum along constant `F2 1H`).

---

## 5. Histidine Tautomeric & Protonation State Classification (`Pelton et al. 1993`)

Histidine's imidazole sidechain exists in three distinct chemical/protonation states near physiological pH, distinguished unambiguously by unprotonated pyridine-like nitrogen (`~249 ppm`, $\beta$-type) versus protonated pyrrole-like nitrogen (`~168–178 ppm`, $\alpha/\alpha^+$-type) chemical shifts and `2J_NH` vs `3J_NH` cross-peak topologies in `HIS_LR_N15HMQC`:

| `HisTautomer` Variant | Chemical State | `15ND1` (ppm) | `15NE2` (ppm) | `13CD2` (ppm) | `1HE1` (ppm) | `13CB` (ppm) | `HIS_LR_N15HMQC` Pattern (`HE1`, `HD2`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Ne2H` | Neutral $N^{\varepsilon 2}\text{-H}$ ($\tau$ tautomer) | `249.5 ± 4.2` | `169.5 ± 4.0` | `117.8 ± 2.1` | `7.82 ± 0.25` | `30.2 ± 1.6` | `HE1` strong to `ND1 (249)` & `NE2 (170)`; `HD2` strong `2J` to `NE2 (170)`, weak `3J` to `ND1 (249)` |
| `Nd1H` | Neutral $N^{\delta 1}\text{-H}$ ($\pi$ tautomer) | `171.0 ± 4.2` | `249.0 ± 4.2` | `125.4 ± 2.2` | `7.76 ± 0.25` | `28.1 ± 1.6` | `HE1` strong to `ND1 (171)` & `NE2 (249)`; `HD2` strong `2J` to `NE2 (249)`, weak `3J` to `ND1 (171)` |
| `DoublyProtonated` | Imidazolium `+1` cation | `177.0 ± 4.0` | `174.5 ± 4.0` | `118.2 ± 2.0` (`< 122`) | `8.58 ± 0.30` | `29.0 ± 1.6` | Both `ND1` & `NE2` in `170–182 ppm`; `HD2` stronger `2J` peak identifies `NE2` |

`classify_his_tautomer` evaluates the joint log-likelihood across `(ND1, NE2, CD2, HD2, CE1, HE1, CB)` plus Pelton's $|\delta_{\text{ND1}} - \delta_{\text{NE2}}|$ separation diagnostic (`~80 ppm` in neutral tautomers vs `< 15 ppm` in `+1` imidazolium) and normalizes via softmax to yield a calibrated tautomer posterior $P(\text{tautomer} \mid \text{data}) \in [0, 1]$. Even when one `15N` cross-peak drops out, `CD2 < 122 ppm` + downfield `HE1 ~ 8.6 ppm` (`DoublyProtonated`) versus `CD2 ~ 125.4 ppm` (`Nd1H`) preserves accurate state recovery.

---

## 6. Probabilistic Ring Clustering & Exact Bipartite Matching (`assign_aromatic_rings`)

### 6.1 Intra-Ring Spin-System Assembly (`cluster_aromatic_rings`)

1. **Node Construction**: `AROM_C13HSQC` cross-peaks form `(1H, 13C)` nodes (rescuing any dropped HSQC roots corroborated by $\ge 2$ cross-peaks in a 3D `AROM_HCCH_TOCSY` strip) and attach candidate `CB` shifts from `HBCBCGCDHD` and `HBCBCGCDCEHE` within $\Delta \delta_{\text{H}} \le \text{tol\_h}$.
2. **Topology-Specific Assembly**:
   - **`HIS`**: pairs `(HE1, CE1)` and `(HD2, CD2)` sharing `HIS_LR_N15HMQC` `(ND1, NE2)` columns (disambiguating `NE2` vs `ND1` via the stronger `2J(HD2, NE2)` vs weaker `3J(HD2, ND1)` cross-peak intensity ratio), backed up by `HBCBCGCDHD`/`HBCBCGCDCEHE` shared-`CB` pairing if `HMQC` is absent.
   - **`TRP`**: assembles the 4-spin benzenoid walk `CE3/HE3 - CZ3/HZ3 - CH2/HH2 - CZ2/HZ2` via `AROM_HCCH_TOCSY` relay edges + BMRB priors (`CZ2 ~ 114.4 ppm`), and links it to pyrrole `CD1/HD1` via shared Yamazaki `CB` (`HBCBCGCDHD` on `HD1` $\leftrightarrow$ `HBCBCGCDCEHE` on `HE3`) and/or `TRP_IND_N15HSQC` `HE1` NOE bridges (`HE1 <-> HD1` and `HE1 <-> HZ2`).
   - **`PHE` vs `TYR`**: greedily extracts mutually consistent 3-node `(CD*, CE*, CZ)` (`PHE`) and 2-node `(CD*, CE*)` (`TYR`) spin systems using intra-ring `AROM_HCCH_TOCSY` connectivity, shared `(cb_hd, cb_he)` Yamazaki agreement, and the `~13 ppm` `13CE` chemical-shift separation between `TYR` (`~117.9 ppm`) and `PHE` (`~130.7 ppm`), with partial-ring rescue when a single peak drops out.

### 6.2 Pairwise Log-Likelihood Ratio & Exact Kuhn–Munkres Matching

Given $M$ assembled ring clusters $\{C_c\}_{c=1}^M$ and $K$ aromatic sequence positions $\{r_k\}_{k=1}^K$ (`K <= 40` in typical proteins), we construct an augmented $K \times (M + K)$ bipartite weight matrix $W$ (where columns $M .. M + K$ are per-residue null/unassigned slots with baseline threshold $s_{\text{null}} = -2.5$):

$$W_{k, c} = \begin{cases} -\infty & \text{if } \text{kind}(C_c) \ne \text{aa}(r_k) \\[4pt] \displaystyle\sum_{a \in C_c} \text{LLR}_{\text{prior}}(\delta_a \mid r_k, a) + \text{LLR}_{\text{CB}}(C_c, r_k) + \text{LLR}_{\text{tie}}(C_c, r_k) + 0.25\, S_{\text{coh}}(C_c) & \text{otherwise} \end{cases}$$

where:
- $\text{LLR}_{\text{CB}}(C_c, r_k)$ scores Gaussian agreement between observed Yamazaki `CB` shifts (`cb_hd`, `cb_he`) and assigned backbone `CB(r_k)` ($\sigma_{\text{CB}} = 0.055\text{ ppm}$, gated at $|\Delta\text{CB}| \le \text{tol\_cb}$), falling back to the residue-type BMRB `CB` prior if backbone `CB(r_k)` was unassigned;
- $\text{LLR}_{\text{tie}}(C_c, r_k)$ incorporates optional `HB2`/`HB3` $\leftrightarrow$ `HD*` NOE cross-peaks (`NOESY13C` / `AROM_HB_NOE`) and pre-assigned `TRP` `NE1/HE1` shifts to break ties between sequence positions with identical or near-degenerate `CB` shifts.

### 6.3 Exact Leave-One-Out Marginal Confidence Calibration

We solve $\pi^* = \arg\max_{\pi} \sum_{k=1}^K W_{k, \pi(k)}$ exactly in $\mathcal{O}(K (M + K)^2)$ time via the primal–dual Kuhn–Munkres algorithm (`hungarian_max_weight`). For every assigned residue $r_k$ ($\pi^*(k) = c^* < M$), we forbid edge $(k, c^*)$ and re-solve global bipartite matching to obtain the exact leave-one-out global score margin:

$$\Delta S_k = S(\pi^*) - \max_{\pi : \pi(k) \ne c^*} S(\pi) \ge 0$$

The ring-to-residue marginal confidence and per-atom confidence are calibrated via:

$$P_{\text{ring}}(k) = \frac{P_{\text{backbone}}(k)}{1 + \exp(-\Delta S_k / T_{\text{cal}})}, \qquad P_{\text{atom}}(k, a) = P_{\text{ring}}(k) \cdot P_{\text{intra}}(c^*, a) \in [0, 1]$$

Crucially, whenever two homologous aromatic residues in unfolded or repetitive regions share identical `CB` shifts and lack tie-breaking `HB` NOEs, $\Delta S_k = 0 \implies P_{\text{ring}}(k) = 0.50 < 0.80$, preventing uncalibrated high-confidence swap errors while still committing unambiguous rings at $P \ge 0.95$.

---

## 7. Quantitative Evaluation & Benchmark Gates

1. **Aliphatic Sidechain Benchmark (`crates/nmr-core/tests/sidechain_tests.rs`)**:
   - Verified across all 20 standard amino-acid covalent topologies, volumetric 2D/3D pickers (`C13HSQC`, `CCCONH`, `HCCCONH`, `HCCH-TOCSY`), `Pro` assignment without amide `HN`, and joint `(U, P)` backbone rescue.
2. **12-Protein Real BMRB Aromatic Benchmark (`crates/nmr-core/tests/aromatic_tests.rs`)**:
   - Evaluated end-to-end (`simulate_aromatic_spectra` with thermal Gaussian noise, chemical-shift jitter, FT sinc sidelobes, and `3%` peak dropout $\to$ `pick_all_aromatic_spectra` $\to$ `assign_aromatic_rings`) across all 12 real BMRB proteins in `data/benchmark/manifest.tsv` (`82` aromatic residues: `36 PHE`, `23 TYR`, `5 TRP`, `18 HIS` cycling all 3 `HisTautomer` states):
   - **Aromatic `(1H, 13C/15N)` assignment accuracy**: **`>= 92.0%`** required gate.
   - **High-confidence (`>= 0.80`) precision**: **`>= 96.0%`** required gate.
   - **Histidine tautomeric state (`Ne2H`, `Nd1H`, `DoublyProtonated`) accuracy**: **`100.0%`** required gate on canonical Pelton shift regimes and across the 12-protein suite.
