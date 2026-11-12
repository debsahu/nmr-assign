# 07b. Bayesian Stereospecific Prochiral Resonance Assignment

Status: implemented in `crates/nmr-core/src/stereospec.rs`. Complements aliphatic and aromatic sidechain spin-system assignment (`docs/specs/07-sidechain-assignment.md`) by resolving diastereotopic methylene (`CH2`), prochiral isopropyl methyl (`Val CG1/CG2`, `Leu CD1/CD2`), and restricted-rotation primary carboxamide (`Asn HD21/HD22`, `Gln HE21/HE22`) pairs into IUPAC stereospecific labels when strongly supported by local scalar couplings and/or NOE intensities, while preserving non-stereospecific `Ambiguity_code = 2` / NEF `x`/`y` wildcards whenever experimental evidence is absent, dynamically averaged, or ambiguous.

---

## 1. Provenance and Clean-Room Literature Basis

This module is an original clean-room implementation derived strictly from peer-reviewed literature and open IUPAC / BMRB / NEF standards:

1. **Güntert, P., Braun, W., Billeter, M. & Wüthrich, K.** (1991). "Automated stereospecific $^1\text{H}$ NMR assignments and their impact on the precision of protein structure determinations in solution." *J. Biomol. NMR* 1, 447–456: joint conformational analysis of vicinal scalar couplings and intraresidue/sequential NOE distance constraints across $(\phi, \psi, \chi_1)$ space to distinguish stereospecific assignments from conformational averaging.
2. **Karplus, M.** (1959). "Contact electron-spin coupling of nuclear magnetic moments." *J. Chem. Phys.* 30, 11–15; and **Karplus, M.** (1963). "Vicinal proton coupling in nuclear magnetic resonance." *J. Am. Chem. Soc.* 85, 2870–2871: trigonometric $\cos^2\theta$ / $\cos\theta$ relation between three-bond scalar couplings and dihedral torsion angles.
3. **DeMarco, A., Llinás, M. & Wüthrich, K.** (1978). "$^1\text{H}$-$^{1}\text{H}$ and $^1\text{H}$-$^{13}\text{C}$ spin-spin couplings in alumichrome." *Biopolymers* 17, 637–650; **Archer, S. J., Ikura, M., Torchia, D. A. & Bax, A.** (1991). "An alternative 3D NMR technique for correlating backbone $^{15}\text{N}$ with side chain $\text{H}^\beta$ resonances in larger proteins." *J. Magn. Reson.* 95, 636–641; and **Vuister, G. W. & Bax, A.** (1993). "Quantitative $J$ correlation: a new approach for measuring homonuclear three-bond $J(\text{H}^\text{N}\text{H}^\alpha)$ coupling constants in $^{15}\text{N}$-enriched proteins." *J. Am. Chem. Soc.* 115, 7772–7777: quantitative $J$-correlation experiments and calibrated $^3J_{\text{H}\alpha\text{H}\beta}$, $^3J_{\text{NH}\beta}$, and $^3J_{\text{HNH}\alpha}$ Karplus curves.
4. **Hu, J.-S. & Bax, A.** (1997). "$\chi_1$ angle information from a simple two-dimensional NMR experiment that identifies *trans* $^3J_{\text{NC}\gamma}$ couplings in isotopically enriched proteins." *J. Biomol. NMR* 9, 323–328; **Hu, J.-S. & Bax, A.** (1997). "Determination of $\phi$ and $\chi_1$ angles in proteins from $^{13}\text{C}$-$^{13}\text{C}$ three-bond $J$ couplings measured by three-dimensional heteronuclear NMR." *J. Am. Chem. Soc.* 119, 6360–6368; and **Chou, J. J., Case, D. A. & Bax, A.** (2003). "Insights into the mobility of methyl-bearing side chains in proteins from $^3J_{\text{CC}}$ and $^3J_{\text{CN}}$ couplings." *J. Am. Chem. Soc.* 125, 8959–8966: heteronuclear $^3J_{\text{NC}\gamma}$ and $^3J_{\text{C}'\text{C}\gamma}$ / $^3J_{\text{CC}}$ Karplus parameterizations for `Val`, `Ile`, `Thr`, and `Leu`.
5. **Nilges, M.** (1995). "Calculation of protein structures with ambiguous distance restraints. Automated assignment of ambiguous NOE crosspeaks and disulphide bridges." *J. Mol. Biol.* 245, 645–660: $r^{-6}$-summed expectation and floating-chirality / ambiguous prochiral treatment when stereospecificity is unresolved.
6. **Lovell, S. C., Word, J. M., Richardson, J. S. & Richardson, D. C.** (2000). "The penultimate rotamer library." *Proteins* 40, 389–408: empirical backbone-dependent $(\chi_1, \chi_2)$ sidechain rotamer distributions and proline pyrrolidine ring-pucker states (`Cg-exo` / `Cg-endo`).
7. **Markley, J. L. et al.** (1998). "Recommendations for the presentation of NMR structures of proteins and nucleic acids." *J. Mol. Biol.* 280, 933–952 (*J. Biomol. NMR* 1998, 12, 1–23); and **Gutmanas, A. et al.** (2015). "NMR Exchange Format: a unified and open standard for representation of NMR restraint data." *Nat. Struct. Mol. Biol.* 22, 433–434 (`NEF v1.1`): IUPAC `HB2`/`HB3` vs `CG1`/`CG2` / `CD1`/`CD2` nomenclature, NMR-STAR v3 `_Atom_chem_shift.Ambiguity_code` (`1` vs `2`), and NEF v1.1 `HBx`/`HBy`, `CGx`/`CGy`, `HGx%`/`HGy%` non-stereospecific notation.

---

## 2. Exact IUPAC Prochiral Nomenclature and Dihedral Phase Offsets

Every tetrahedral dihedral offset in `stereospec.rs` follows the IUPAC-IUBMB 1998 standard (Markley et al., 1998) and is verified to `< 2°` against all 76 residues of experimental solution NMR human Ubiquitin (`PDB 1D3Z`).

### 2.1 Prochiral Site Classes

| Site Class | Residues | Governing Torsion | IUPAC Stereospecific (`Ambiguity_code = 1`) | NEF v1.1 Non-Stereospecific (`Ambiguity_code = 2`) |
|---|---|---|---|---|
| `GlyAlphaMethylene` | `GLY` | $\phi_i = \text{C}_{i-1}\text{–N}_i\text{–CA}_i\text{–C}_i$ | `HA2`, `HA3` | `HAx`, `HAy` |
| `BetaMethylene` | `ARG`, `ASN`, `ASP`, `CYS`, `GLN`, `GLU`, `HIS`, `LEU`, `LYS`, `MET`, `PHE`, `PRO`, `SER`, `TRP`, `TYR` | $\chi_1 = \text{N–CA–CB–X}^\gamma$ | `HB2`, `HB3` | `HBx`, `HBy` |
| `GammaMethylene` | `ARG`, `GLN`, `GLU`, `LYS`, `MET`, `PRO` (`CG`); `ILE` (`CG1`) | $\chi_2 = \text{CA–CB–CG–X}^\delta$ (`CD1` for `ILE`) | `HG2`/`HG3` (`HG12`/`HG13` on `ILE`) | `HGx`/`HGy` (`HG1x`/`HG1y` on `ILE`) |
| `DeltaMethylene` | `ARG`, `LYS`, `PRO` | $\chi_3 = \text{CB–CG–CD–X}^\varepsilon$ | `HD2`, `HD3` | `HDx`, `HDy` |
| `EpsilonMethylene` | `LYS` | $\chi_4 = \text{CG–CD–CE–NZ}$ | `HE2`, `HE3` | `HEx`, `HEy` |
| `ValIsopropyl` | `VAL` | $\chi_1 = \text{N–CA–CB–CG1}$ | `CG1`/`HG11..13` (`HG1%`), `CG2`/`HG21..23` (`HG2%`) | `CGx`/`HGx%`, `CGy`/`HGy%` |
| `LeuIsopropyl` | `LEU` | $\chi_1 = \text{N–CA–CB–CG}$, $\chi_2 = \text{CA–CB–CG–CD1}$ | `CD1`/`HD11..13` (`HD1%`), `CD2`/`HD21..23` (`HD2%`) | `CDx`/`HDx%`, `CDy`/`HDy%` |
| `AsnCarboxamide` | `ASN` | Planar $\text{CG(=OD1)–ND2}$ | `HD21` (*trans* to `OD1`), `HD22` (*cis* to `OD1`) | `HD2x`, `HD2y` |
| `GlnCarboxamide` | `GLN` | Planar $\text{CD(=OE1)–NE2}$ | `HE21` (*trans* to `OE1`), `HE22` (*cis* to `OE1`) | `HE2x`, `HE2y` |

*(Note: `ALA` `HB1..3` is a fast-rotating $C_3$ methyl group, not a prochiral pair; `PHE`/`TYR` `HD1/HD2` and `HE1/HE2` are symmetric aromatic ring-flip pairs with `Ambiguity_code = 3`.)*

### 2.2 Tetrahedral Phase Offsets Verified Against `1D3Z`

Under the IUPAC right-handed torsion definition (`calc_dihedral_deg(A, B, C, D)`):

1. **$\beta$-Methylene (`HB2` / `HB3`) across all 15 `CB`-methylene amino acids** (including `SER` and `CYS`, because IUPAC 1998 preserves uniform Newman-projection `+120°`/`-120°` numbering relative to $\chi_1 = \text{N–CA–CB–X}^\gamma$ regardless of CIP `O`/`S` vs `CA` priority reversal):
   $$\theta(\text{N–CA–CB–HB2}) = \chi_1 + 120^\circ, \qquad \theta(\text{N–CA–CB–HB3}) = \chi_1 - 120^\circ$$
   $$\theta(\text{HA–CA–CB–HB2}) = \chi_1 - 120^\circ, \qquad \theta(\text{HA–CA–CB–HB3}) = \chi_1$$
   $$\theta(\text{C}'\text{–CA–CB–HB2}) = \chi_1, \qquad \theta(\text{C}'\text{–CA–CB–HB3}) = \chi_1 + 120^\circ$$

2. **Valine Isopropyl Methyls (`CG1` / `CG2` and `HB` around $\chi_1 = \text{N–CA–CB–CG1}$)**:
   $$\theta(\text{N–CA–CB–CG1}) = \chi_1, \qquad \theta(\text{N–CA–CB–CG2}) = \chi_1 + 120^\circ, \qquad \theta(\text{N–CA–CB–HB}) = \chi_1 - 120^\circ$$
   $$\theta(\text{C}'\text{–CA–CB–CG1}) = \chi_1 - 120^\circ, \qquad \theta(\text{C}'\text{–CA–CB–CG2}) = \chi_1, \qquad \theta(\text{HA–CA–CB–HB}) = \chi_1$$

3. **Leucine Isopropyl Methyls (`CD1` / `CD2` and `HG` around $\chi_2 = \text{CA–CB–CG–CD1}$)**:
   $$\theta(\text{CA–CB–CG–CD1}) = \chi_2, \qquad \theta(\text{CA–CB–CG–CD2}) = \chi_2 + 120^\circ, \qquad \theta(\text{CA–CB–CG–HG}) = \chi_2 - 120^\circ$$

4. **Glycine $\alpha$-Methylene (`HA2` / `HA3` around $\phi_i = \text{C}_{i-1}\text{–N}_i\text{–CA}_i\text{–C}_i$)**:
   $$\theta(\text{C}_{i-1}\text{–N–CA–HA2}) = \phi_i + 120^\circ, \qquad \theta(\text{C}_{i-1}\text{–N–CA–HA3}) = \phi_i - 120^\circ$$
   $$\theta(\text{H–N–CA–HA2}) = \phi_i - 60^\circ, \qquad \theta(\text{H–N–CA–HA3}) = \phi_i + 60^\circ$$

5. **Distal Methylenes (`CG`, `ILE CG1`, `CD`, `CE`)**:
   - `CG` (`HG2` / `HG3` around $\chi_2 = \text{CA–CB–CG–X}^\delta$): $\theta(\text{CA–CB–CG–HG2}) = \chi_2 + 120^\circ$, $\theta(\text{CA–CB–CG–HG3}) = \chi_2 - 120^\circ$.
   - `ILE CG1` (`(2S,3S)`-Ile around $\chi_1 = \text{N–CA–CB–CG1}$, $\chi_2 = \text{CA–CB–CG1–CD1}$): at `CB`, $\theta(\text{N–CA–CB–CG2}) = \chi_1 - 120^\circ$ and $\theta(\text{N–CA–CB–HB}) = \chi_1 + 120^\circ$; at `CG1`, $\theta(\text{CA–CB–CG1–HG12}) = \chi_2 + 120^\circ$, $\theta(\text{CA–CB–CG1–HG13}) = \chi_2 - 120^\circ$, $\theta(\text{HB–CB–CG1–HG12}) = \chi_2$, $\theta(\text{HB–CB–CG1–HG13}) = \chi_2 + 120^\circ$.
   - `CD` (`HD2` / `HD3` around $\chi_3$) and `CE` (`HE2` / `HE3` around $\chi_4$): branch `2` at $\chi_k + 120^\circ$, branch `3` at $\chi_k - 120^\circ$.

6. **Primary Carboxamide $\text{NH}_2$ (`ASN HD21/HD22`, `GLN HE21/HE22`)**:
   - `ASN`: $\theta(\text{OD1–CG–ND2–HD21}) = 180^\circ$ (*trans* to `OD1`, *syn* $0^\circ$ to `CB`, downfield $^1\text{H}$), $\theta(\text{OD1–CG–ND2–HD22}) = 0^\circ$ (*cis* to `OD1`, *anti* $180^\circ$ to `CB`, upfield $^1\text{H}$).
   - `GLN`: $\theta(\text{OE1–CD–NE2–HE21}) = 180^\circ$ (*trans* to `OE1`, *syn* $0^\circ$ to `CG`, downfield $^1\text{H}$), $\theta(\text{OE1–CD–NE2–HE22}) = 0^\circ$ (*cis* to `OE1`, *anti* $180^\circ$ to `CG`, upfield $^1\text{H}$).

---

## 3. Forward Physical Models: Karplus Scalar Couplings and $r^{-6}$ NOE Expectations

### 3.1 Calibrated Karplus Equations

Each vicinal scalar coupling is evaluated via $J(\theta) = A\cos^2(\theta + \delta) + B\cos(\theta + \delta) + C$ and librationally averaged over $\mathcal{N}(0, \sigma_\chi^2)$ ($\sigma_\chi = 10^\circ$, 5-point Gauss–Hermite quadrature):

- **$^3J_{\text{H}\alpha\text{H}\beta}(\theta_{\text{HA–CA–CB–HB}})$** (DeMarco et al., 1978; Güntert et al., 1991):
  $$A = 9.5\text{ Hz},\quad B = -1.6\text{ Hz},\quad C = 1.8\text{ Hz},\quad \delta = 0^\circ \qquad (\text{trans } 12.9\text{ Hz},\ \text{gauche } 3.38\text{ Hz})$$
- **Magnitude $|^3J_{\text{NH}\beta}|(\theta_{\text{N–CA–CB–HB}})$** (DeMarco et al., 1978; Archer et al., 1991):
  $$|^3J_{\text{NH}\beta}(\theta)| = \left|-4.4\cos^2\theta + 1.2\cos\theta + 0.1\right|\text{ Hz} \qquad (\text{trans } 5.5\text{ Hz},\ \text{gauche } 0.4\text{ Hz})$$
  *(Accepts either unsigned magnitude or signed negative values via `|J_obs|`.)*
- **$^3J_{\text{HNH}\alpha}(\theta_{\text{H–N–CA–HA}})$ for `GLY HA2/HA3`** (Vuister & Bax, 1993):
  $$A = 6.51\text{ Hz},\quad B = -1.76\text{ Hz},\quad C = 1.60\text{ Hz},\quad \delta = 0^\circ$$
- **$^3J_{\text{NC}\gamma}(\theta_{\text{N–CA–CB–CG}})$** (Hu & Bax, 1997; Chou, Case & Bax, 2003):
  - `Val` / `Ile`: $A = 2.64\text{ Hz},\ B = 0.26\text{ Hz},\ C = -0.22\text{ Hz},\ \delta = 6^\circ$ ($\text{trans } \approx 2.15\text{ Hz},\ \text{gauche } \approx 0.56\text{ Hz}$).
  - `Thr`: $A = 2.01\text{ Hz},\ B = 0.21\text{ Hz},\ C = -0.12\text{ Hz},\ \delta = 7^\circ$.
- **$^3J_{\text{C}'\text{C}\gamma}(\theta_{\text{C}'\text{–CA–CB–CG}})$ and aliphatic $^3J_{\text{CC}}$ (`Leu` $^3J_{\text{CAC}\delta}$)** (Hu & Bax, 1997; Chou, Case & Bax, 2003):
  - `Val` / `Ile` / `Leu`: $A = 3.42\text{ Hz},\ B = -0.59\text{ Hz},\ C = 0.17\text{ Hz},\ \delta = 0^\circ$ ($\text{trans } 4.18\text{ Hz},\ \text{gauche } 0.73\text{ Hz}$).
  - `Thr`: $A = 2.76\text{ Hz},\ B = -0.67\text{ Hz},\ C = 0.19\text{ Hz},\ \delta = 0^\circ$.

### 3.2 Analytical NeRF Dipeptide + Sidechain $r^{-6}$ NOE Model

Given $(\phi_i, \psi_i)$, the local peptide unit $(\text{C}_{i-1}, \text{N}_i, \text{H}_i, \text{CA}_i, \text{HA}_i, \text{C}_i, \text{N}_{i+1}, \text{H}_{i+1})$ and tetrahedral sidechain atoms are placed analytically via the Natural Extension Reference Frame (Parsons et al., 2005; Engh & Huber, 1991). Validated across all 76 residues of `1D3Z`, analytical NeRF reproduces intraresidue and sequential proton-proton distances (`H_i–HB`, `HA_i–HB`, `H_{i+1}–HB`) with mean absolute error `< 0.04 Å`.

For a 3-proton methyl group $b$ under fast $C_3$ rotation, the effective $r^{-6}$ dipolar rate to anchor proton $p$ in rotamer state $r$ is:
$$\left(r_{p, b}^{(r)}\right)^{-6} = \frac{1}{3}\sum_{q=1}^{3} \left\|\mathbf{x}_p - \mathbf{x}_{b, q}^{(r)}\right\|^{-6}$$

For any rotamer population vector $\mathbf{w} = (w_{g^-}, w_t, w_{g^+}) \in \Delta^2$:
$$\langle J_m \rangle(\mathbf{w}) = \sum_{r} w_r J_m^{(r)}, \qquad \mu_{p, b}(\mathbf{w}) = \sum_{r} w_r \left(r_{p, b}^{(r)}\right)^{-6} + r_{\max}^{-6}$$
with baseline spin-diffusion / detection floor $r_{\max} = 5.5\text{ Å}$.

For each anchor proton $p$ where cross-peak intensities $(I_{p,a}, I_{p,b})$ are measured within the same strip (sharing unknown transfer/mixing scale $\kappa_p$), analytical marginalization over $\ln \kappa_p$ yields the exact scale-invariant log-ratio likelihood:
$$\ln\left(\frac{I_{p,a}}{I_{p,b}}\right) \;\sim\; \mathcal{N}\!\left(\ln\frac{\mu_{p,\pi_H(a)}(\mathbf{w})}{\mu_{p,\pi_H(b)}(\mathbf{w})},\; 2\sigma_{\ln I}^2\right)$$

---

## 4. Spike-and-Slab Simplex Prior and Joint Bayesian Inference

To accurately represent rigid core rotamers, anisotropic multi-rotamer jumps, and dynamically averaged surface sidechains without false stereospecific commitments under measurement noise:

1. **Simplex Grid**: $\Delta^2 = \{(w_{g^-}, w_t, w_{g^+}) : w_r \ge 0,\ \sum_r w_r = 1\}$ is discretized on a uniform $M = 20$ barycentric lattice ($\binom{22}{2} = 231$ points) augmented with the exact isotropic rotamer-averaged state $\mathbf{w}_{\text{iso}} = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ ($232$ states total).
2. **Three-Regime Spike-and-Slab Prior $P(\mathbf{w} \mid \phi_i, \psi_i)$**:
   - Dominant-rotamer vertices $\mathbf{e}_r$ carry total prior mass $\lambda_{\text{dom}} = 0.56$ distributed according to Lovell et al. (2000) backbone-dependent rotamer probabilities $\pi_r(\phi_i, \psi_i, \text{aa})$ (using `Cg-exo`/`Cg-endo` pyrrolidine puckers for `Pro`);
   - Multi-rotamer interior states carry total prior mass $\lambda_{\text{mix}} = 0.20$ under a regularized $\text{Dir}(\alpha_0 \boldsymbol{\pi})$ density ($\alpha_0 = 3.0$);
   - Exact branch-symmetric isotropic averaging $\mathbf{w}_{\text{iso}} = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ carries explicit spike mass $\lambda_{\text{iso}} = 0.24$, acting as a calibrated Bayesian Occam null against hallucinating swaps on noisy dynamically averaged sidechains or overlapped peaks.
3. **Swap Hypothesis Prior & Hierarchical Sidechain Conditioning**:
   - $P(H = \text{Direct}) = P(H = \text{Swapped}) = 0.5$ identically for all aliphatic methylenes and `Val`/`Leu` isopropyl methyls (preventing BMRB deposition-sorting bias). For primary carboxamides (`Asn HD21/HD22`, `Gln HE21/HE22`), a bounded BMRB shift-separation log-odds term ($|\text{LLR}_{\text{shift}}| \le 1.0$, keeping standalone shift-only posterior $\le 0.73 < 0.90$) reflects carbonyl magnetic anisotropy without ever triggering a `>= 0.90` commitment in the absence of NOE evidence.
   - Within a residue (`infer_residue_stereospec`), the marginal $\chi_1$ posterior inferred at `CB` (`HB2/HB3`) propagates forward as the $\chi_1$ prior for downstream $\chi_2$-dependent pairs (`Leu CD1/CD2`, `Ile HG12/HG13`, `CG HG2/HG3`), resolving the classic `Leu` `mt` $(-60^\circ, 180^\circ)$ vs `tp` $(180^\circ, +60^\circ)$ symmetry cleanly.

---

## 5. Strict Commitment Rule and Standards-Compliant Export

Let $P_{\text{win}} = \max(P(H = \text{Direct} \mid \mathcal{D}),\; P(H = \text{Swapped} \mid \mathcal{D})) \in [0.5, 1.0]$, with configurable threshold $\tau_{\text{commit}} = 0.90$:

- **Committed (`P_win >= 0.90`)**:
  - NMR-STAR v3: emits resolved IUPAC atom names (`HB2`/`HB3`, `CG1`/`CG2`, `HG11..13`/`HG21..23`, `CD1`/`CD2`, `HD11..13`/`HD21..23`, `HA2`/`HA3`, `HD21`/`HD22`, `HE21`/`HE22`) with `_Atom_chem_shift.Ambiguity_code = 1`.
  - NEF v1.1: emits exact stereospecific atom names (`HB2`/`HB3`, `CG1`/`CG2`, `HG1%`/`HG2%`, `CD1`/`CD2`, `HD1%`/`HD2%`, `HA2`/`HA3`, `HD21`/`HD22`, `HE21`/`HE22`).
- **Uncommitted / Ambiguous (`P_win < 0.90`)**:
  - NMR-STAR v3: emits `_Atom_chem_shift.Ambiguity_code = 2`.
  - NEF v1.1: emits standards-compliant non-stereospecific `x`/`y` prochiral wildcards (`HBx`/`HBy`, `CGx`/`CGy`, `HGx%`/`HGy%`, `CDx`/`CDy`, `HDx%`/`HDy%`, `HAx`/`HAy`, `HGx`/`HGy`, `HG1x`/`HG1y`, `HDx`/`HDy`, `HEx`/`HEy`, `HD2x`/`HD2y`, `HE2x`/`HE2y`), classified as `NefAtomWildcardKind::NonStereoProchiral` by `nmr_core::nef::classify_nef_atom_name`.

---

## 6. Quantitative Verification Gates

1. **Exact `1D3Z` IUPAC Geometry**: Zero sign or handedness inversion across all 76 residues of `PDB 1D3Z` (`CB`, `Val`, `Leu`, `Gly`, `Ile`, `Thr`, `Pro`, `CG`, `CD`, `CE`, `Asn`, `Gln`), and `< 0.05 Å` mean analytical NeRF proton-proton distance error.
2. **Noisy Scrambled `1D3Z` Precision Gate**: Across 10 independent seeds with `±0.7 Hz` Gaussian coupling noise, `±25%` multiplicative log-normal NOE intensity noise, and randomly scrambled `(a, b)` input ordering, stereospecifically committed methylene and `Val`/`Leu` isopropyl pairs (`posterior >= 0.90`) achieve **`>= 95.0%` precision**.
3. **Zero Uncalibrated False Swaps**: Absent couplings/NOEs, rotamer-averaged `(1/3, 1/3, 1/3)` observables, symmetric inputs, and ambiguous single-pathway `3J_HAHB`-only inputs produce **`0` false `Ambiguity_code = 1` swaps** (`posterior` remains `< 0.90`, emitting `Ambiguity_code = 2` and NEF `x`/`y` labels).
