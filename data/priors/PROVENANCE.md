# Empirical BMRB Chemical Shift Priors — Provenance

## Source & Licence
- **Database**: BioMagResBank (BMRB, https://bmrb.io; Ulrich et al., *Nucleic Acids Res.* 2008, 36, D402–D408).
- **Data licence**: Public domain / CC0 1.0.
- **Fallback reference** (used strictly when sample count `N < 5`, which occurs only for `PRO N` where `N = 0` in amide-detected triple-resonance entries): Wishart et al., *J. Biomol. NMR* 1995, 5, 67–81.

## Strict Train / Benchmark Disjointness
- **Training set**: **21 DSS-referenced public BMRB entries** listed below (screened for systematic TMS / perdeuteration / indirect `13CO` / `15N` referencing offsets per Zhang, Neal & Wishart, *J. Biomol. NMR* 2003, 25, 173–195).
- **Benchmark suite** (`data/benchmark/manifest.tsv`): **12 public BMRB entries** (`bmr16649`, `bmr15536`, `bmr4296`, `bmr6197`, `bmr15477`, `bmr4375`, `bmr5387`, `bmr17769`, `bmr15176`, `bmr6968`, `bmr15683`, `bmr6541`).
- **Overlap**: **0 entries** (verified programmatically and enforced in `crates/nmr-core/tests/bmrb_phase1_tests.rs`).

## Extraction & Outlier Filtering Protocol
1. **Primary shift loop selection**: Only the first `_Atom_chem_shift` loop (`save_assigned_chem_shift_list_1`) is parsed from each BMRB NMR-STAR file so multi-condition depositions never double-count or mix sample conditions.
2. **IUPAC alias normalization**: `HN -> H`, `CO -> C`.
3. **Bio-plausible chemical-shift window**:
   - `CA`: `[38.0, 72.0]` ppm
   - `CB`: `[12.0, 78.0]` ppm
   - `C`: `[165.0, 186.0]` ppm
   - `N`: `[95.0, 145.0]` ppm
   - `H`: `[5.0, 12.5]` ppm
   - Sidechain `13C`: `[5.0, 186.0]` ppm, sidechain `15N`: `[65.0, 185.0]` ppm (`ARG NE`: `[75.0, 96.0]` ppm to exclude spectral-aliasing folds, `LYS NZ`: `[25.0, 45.0]` ppm), sidechain `1H`: `[-2.5, 15.0]` ppm.
4. **3-sigma outlier filter**: Within each `(residue, atom)` pool (`N >= 5`), values with `|x - mu_0| > 3.0 * sigma_0` are removed once before computing the final `(mean, pop_std)`.
5. **Cysteine (`CYS`) redox pooling**: `CYS CB` is naturally bimodal between reduced thiol (`-SH`, `CB <= 35.0` ppm: `N = 17, mean = 29.129 ± 2.991` ppm) and disulfide-oxidized (`-S-S-`, `CB > 35.0` ppm: `N = 9, mean = 40.161 ± 3.813` ppm). In `bmrb_priors.json`, both states are pooled without 3-sigma redox clipping (`N = 26, mean = 32.948 ± 6.199` ppm) so unknown redox states are never falsely penalized; separate `CYS_REDUCED` and `CYS_OXIDIZED` entries are additionally provided in `bmrb_all_atom_priors.json`.

## Training BMRB Entries (21)
| BMRB ID | PDB ID | Construct Residues | Validated Shifts Used | Protein |
|---|---|---:|---:|---|
| `bmr4022` | `1AZM` | 260 | 1847 | Human carbonic anhydrase I |
| `bmr4267` | `1DFV` | 179 | 2190 | Human neutrophil gelatinase-associated lipocalin (NGAL) |
| `bmr4302` | `1MEK` | 115 | 1410 | Human protein disulfide isomerase a' domain |
| `bmr4472` | `1A0O` | 129 | 1350 | Escherichia coli BeFx-activated chemotaxis protein CheY |
| `bmr4562` | `193L` | 129 | 1238 | Gallus gallus hen egg-white lysozyme |
| `bmr4670` | `1J6Y` | 120 | 1144 | Arabidopsis thaliana peptidyl-prolyl cis/trans isomerase |
| `bmr4752` | `1J9I` | 68 | 879 | Bacteriophage lambda terminase gpNu1 DNA-binding domain |
| `bmr5010` | `1JAJ` | 174 | 2112 | African swine fever virus DNA polymerase X |
| `bmr5106` | `1RYJ` | 70 | 897 | Methanothermobacter thermautotrophicus MTH1743 |
| `bmr5211` | `1KA7` | 128 | 1554 | Human SLAM-associated protein SAP/SH2D1A SH2 domain |
| `bmr5505` | `1FA4` | 105 | 1201 | Anabaena variabilis reduced plastocyanin |
| `bmr5757` | `1K1C` | 93 | 438 | Bacillus subtilis catabolite repression HPr-like protein Crh |
| `bmr6032` | `1UDM` | 142 | 1797 | Mus musculus coactosin-like protein |
| `bmr6240` | `1TQ1` | 129 | 1133 | Arabidopsis thaliana protein At5g66040 |
| `bmr6338` | `1XMT` | 103 | 987 | Arabidopsis thaliana protein At1g77540 |
| `bmr6622` | `193L` | 130 | 820 | Gallus gallus S-methylated hen egg-white lysozyme |
| `bmr7021` | `1FD9` | 213 | 1810 | Legionella pneumophila macrophage infectivity potentiator Mip |
| `bmr7280` | `1GB1` | 58 | 594 | Streptococcal protein G B1 domain (GB1) |
| `bmr15250` | `1SUA` | 266 | 980 | Thermus thermophilus protein SBT70 |
| `bmr15380` | `2JSV` | 56 | 314 | Streptococcal protein G B1 domain T2Q variant |
| `bmr18025` | `3NY2` | 75 | 128 | Human UBR2 ubiquitin ligase UBR-box domain |

## Backbone Sample Counts & Empirical Parameters (`bmrb_priors.json`)
| Residue | CA (mean ± sd, N) | CB (mean ± sd, N) | C (mean ± sd, N) | N (mean ± sd, N) | H (mean ± sd, N) |
|---|---|---|---|---|---|
| `ALA` | 53.350 ± 1.925 (N=217/218) | 19.210 ± 1.752 (N=218/219) | 177.782 ± 2.135 (N=164/165) | 122.986 ± 3.565 (N=200/202) | 8.158 ± 0.601 (N=196/198) |
| `ARG` | 56.824 ± 2.194 (N=100/100) | 30.910 ± 1.838 (N=99/99) | 176.013 ± 1.787 (N=81/81) | 121.379 ± 4.128 (N=92/92) | 8.277 ± 0.715 (N=100/101) |
| `ASN` | 53.582 ± 1.635 (N=126/127) | 38.690 ± 1.570 (N=125/126) | 175.169 ± 1.509 (N=95/95) | 118.966 ± 4.208 (N=113/113) | 8.406 ± 0.613 (N=121/123) |
| `ASP` | 54.281 ± 1.879 (N=141/141) | 41.164 ± 1.664 (N=138/139) | 176.334 ± 1.641 (N=114/114) | 121.650 ± 4.244 (N=136/136) | 8.320 ± 0.657 (N=135/135) |
| `CYS` | 58.470 ± 3.371 (N=26/26) | 32.948 ± 6.199 (N=26/26) | 175.310 ± 2.099 (N=18/18) | 122.374 ± 3.936 (N=28/28) | 8.327 ± 0.914 (N=36/36) |
| `GLN` | 56.654 ± 2.025 (N=85/85) | 29.489 ± 1.526 (N=83/84) | 176.371 ± 1.931 (N=70/70) | 119.428 ± 3.748 (N=82/82) | 7.950 ± 0.700 (N=77/77) |
| `GLU` | 57.386 ± 2.030 (N=153/153) | 30.170 ± 1.732 (N=151/153) | 177.016 ± 1.961 (N=122/122) | 120.648 ± 3.826 (N=154/155) | 8.362 ± 0.633 (N=141/142) |
| `GLY` | 45.514 ± 1.144 (N=206/206) | — | 174.075 ± 1.667 (N=159/160) | 109.468 ± 3.819 (N=201/202) | 8.358 ± 0.666 (N=201/204) |
| `HIS` | 56.131 ± 2.542 (N=47/47) | 31.179 ± 2.739 (N=47/47) | 175.105 ± 2.429 (N=43/43) | 119.066 ± 4.988 (N=46/46) | 8.247 ± 0.930 (N=48/48) |
| `ILE` | 61.411 ± 2.538 (N=135/135) | 39.118 ± 1.980 (N=134/134) | 175.761 ± 1.681 (N=112/113) | 122.118 ± 4.685 (N=128/129) | 8.312 ± 0.749 (N=132/132) |
| `LEU` | 55.318 ± 1.985 (N=188/189) | 42.583 ± 1.907 (N=187/188) | 176.793 ± 2.069 (N=152/152) | 122.019 ± 4.368 (N=183/183) | 8.171 ± 0.681 (N=178/179) |
| `LYS` | 57.048 ± 2.149 (N=197/197) | 33.089 ± 1.981 (N=195/196) | 176.438 ± 2.009 (N=158/158) | 120.873 ± 4.239 (N=191/191) | 8.232 ± 0.678 (N=183/183) |
| `MET` | 55.644 ± 2.156 (N=41/41) | 33.691 ± 1.819 (N=41/41) | 175.108 ± 2.089 (N=27/27) | 120.264 ± 4.178 (N=38/38) | 8.289 ± 0.743 (N=37/37) |
| `PHE` | 57.353 ± 2.381 (N=96/96) | 40.367 ± 2.214 (N=96/96) | 175.193 ± 1.922 (N=81/81) | 121.000 ± 4.566 (N=96/96) | 8.401 ± 0.842 (N=94/94) |
| `PRO` | 63.565 ± 1.448 (N=91/91) | 32.082 ± 0.864 (N=88/90) | 177.027 ± 1.842 (N=79/79) | 134.000 ± 6.000 (fallback, N=1) | — |
| `SER` | 58.887 ± 2.105 (N=182/182) | 63.963 ± 1.320 (N=177/179) | 174.546 ± 1.671 (N=158/158) | 116.411 ± 3.938 (N=168/169) | 8.277 ± 0.627 (N=174/175) |
| `THR` | 62.265 ± 2.384 (N=157/157) | 70.027 ± 1.416 (N=155/155) | 174.451 ± 1.528 (N=128/128) | 115.331 ± 5.079 (N=151/152) | 8.280 ± 0.646 (N=144/144) |
| `TRP` | 57.703 ± 2.214 (N=38/38) | 30.391 ± 1.747 (N=39/39) | 176.304 ± 1.650 (N=30/30) | 122.569 ± 4.668 (N=33/33) | 8.177 ± 0.834 (N=37/38) |
| `TYR` | 57.770 ± 2.355 (N=80/81) | 39.978 ± 2.029 (N=81/81) | 175.106 ± 1.881 (N=68/68) | 121.411 ± 4.627 (N=80/80) | 8.558 ± 0.780 (N=78/79) |
| `VAL` | 61.975 ± 2.663 (N=202/202) | 33.133 ± 1.700 (N=202/202) | 175.359 ± 1.836 (N=162/163) | 121.466 ± 5.016 (N=195/196) | 8.264 ± 0.753 (N=189/189) |
