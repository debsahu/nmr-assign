# Torsion Fragment Corpus Provenance (`data/torsion_db/`)

## Licence & Clean-Room Statement

- **Data Licence**: CC0 1.0 Public Domain (`wwPDB` atomic coordinates & `BMRB` experimental chemical shifts) with open `CATH` / `PDBe SIFTS` fold topology & `UniProt` family identifiers.
- **Methodological Citations**:
  - Shen, Y., Vernon, R., Baker, D. & Bax, A. (2008). *Proc. Natl. Acad. Sci. U.S.A.* 105(12), 4685-4690.
  - Shen, Y. & Bax, A. (2013). *J. Biomol. NMR* 56(3), 227-241.
  - Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (1995). *J. Biomol. NMR* 5(1), 67-81.
  - Berjanskii, M. V. & Wishart, D. S. (2005). *J. Am. Chem. Soc.* 127(43), 14970-14971 (`RCI` & `S^2`).
  - Zhang, Y. & Skolnick, J. (2004). *Proteins* 57(4), 702-710 (`TM-score`).

## File Digests (SHA-256)

- `torsion_corpus.tsv`: `e34d3001d4c6989564d6bdfacff9dc401f92566aa77b499a23933b202045d88b` (`129` non-redundant monomeric protein chains)
- `benchmark_refs.tsv`: `898090b42e28b9947b2831c8e709a195da4b382cd030f4f8ad43268b2c3569a1` (`12` benchmark evaluation entries from `data/benchmark/manifest.tsv`)

## Reproducible Extraction & Verification Protocol (`scripts/build_torsion_corpus.py`)

`data/torsion_db/torsion_corpus.tsv` is generated and verified strictly from public CC0 `wwPDB` coordinate files (`https://files.rcsb.org/download/<pdb_id>.pdb`) and public CC0 `BMRB` NMR-STAR files (`https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr<bmrb_id>/bmr<bmrb_id>_3.str`) via the Python 3 standard-library script `scripts/build_torsion_corpus.py` (zero third-party Python packages; zero external structural-biology binaries):

1. **Coordinate Selection & IUPAC Dihedrals**:
   - Reads **Model 1** only (stops at the first `ENDMDL` record for multi-conformer NMR ensembles), `ATOM  ` records on `chain_id`, `altLoc in (' ', 'A')` (retaining the first conformer atom encountered), `iCode == ' '`, and standard 20 amino-acid residues with complete backbone `(N, CA, C)` coordinates.
   - Enforces covalent peptide-bond continuity `d(C_{i-1}, N_i) < 2.0 Å` before computing `phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)` and `d(C_i, N_{i+1}) < 2.0 Å` before computing `psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})`.
   - Computes sidechain `chi1_i = dihedral(N_i, CA_i, CB_i, X_gamma)` using IUPAC `X_gamma` precedence (`CYS -> SG`, `ILE -> CG1`, `SER -> OG`, `THR -> OG1`, `VAL -> CG1`, all other `chi1`-bearing residues `-> CG`).
   - Formats `(phi, psi, chi1)` to `0.1 deg` (`%.1f`) and `CA` `(x, y, z)` to `0.01 Å` (`%.2f`), normalizing `-0.0` to `0.0`.
2. **Secondary Chemical Shifts & Coil-Origin Referencing**:
   - Reads the primary `_Atom_chem_shift` loop (`Assigned_chem_shift_list_ID == 1`, first `Entity_ID`), mapping `HN -> H`, `CO -> C`, and `GLY` `HA = 0.5 * (HA2 + HA3)` when `HA` is absent, then aligns the BMRB sequence to the PDB `ATOM` sequence via Gotoh global alignment (exact gapless substring match preferred).
   - Subtracts Wishart et al. (*J. Biomol. NMR* 1995, 5, 67–81) sequence-corrected random-coil chemical shifts matching `nmr_core::dihedrals::random_coil_shifts`: oxidized `CYS` (`CB_obs > 35.0 ppm` -> `CA = 55.6, CB = 41.1, HA = 4.71 ppm`), pre-Proline corrections (`CA -= 1.95, C' -= 2.00, CB += 0.25 ppm` for non-Gly; `CA -= 1.00, C' -= 1.50 ppm` for Gly), `PRO` `(dH, dN) = None`, and `GLY` `dCB = None`.
   - Applies the self-contained Wang & Wishart (*J. Biomol. NMR* 2005, 31, 143–148) coil-origin median offset correction per chain: over residues with `|dCA - dCB| <= 1.5 ppm` (when `>= 5` coil-like residues exist), if `>= 5` observations exist for nucleus `k in (H, N, CA, CB, C', HA)` and `|median_k| > (0.20, 1.20, 1.00, 1.00, 1.00, 0.20)[k] ppm`, subtracts `median_k` from nucleus `k` across the chain and formats secondary shifts to `0.001 ppm` (`%.3f`).
3. **Offline Verification & Spot-Check Commands**:
   ```bash
   # Verify schema, 129 chains / 13,647 residues, and exact SHA-256 + offline spot-check on 1UBQ_A:
   python3 scripts/build_torsion_corpus.py \
     --verify data/torsion_db/torsion_corpus.tsv \
     --spot-check-pdb data/fixtures/ubiquitin_1ubq_xray.pdb \
     --spot-check-star data/benchmark/bmr5387.str \
     --chain 1UBQ_A

   # Rebuild all 129 chains from local or downloaded public wwPDB + BMRB caches:
   python3 scripts/build_torsion_corpus.py \
     --build --manifest data/torsion_db/PROVENANCE.md \
     --pdb-dir /path/to/wwpdb_cache --bmrb-dir /path/to/bmrb_cache \
     --download-missing --out data/torsion_db/torsion_corpus.tsv
   ```

## Corpus Chain Inventory

| Chain Key | PDB ID | Chain | BMRB ID | UniProt ID | CATH Superfamily | CATH Topology Fold | Length | Shift Coverage |
|---|---|---|---|---|---|---|---:|---:|
| `1BR0_A` | `1BR0` | `A` | `4198` | `P32851` | `1.20.58.70` | `1.20.58` | 120 | 100.0% |
| `1D1O_A` | `1D1O` | `A` | `4581` | `P02633` | `1.10.238.10` | `1.10.238` | 75 | 93.3% |
| `1GM1_A` | `1GM1` | `A` | `5131` | `Q64512` | `2.30.42.10` | `2.30.42` | 94 | 100.0% |
| `1JAS_A` | `1JAS` | `A` | `5038` | `P63146` | `3.10.110.10` | `3.10.110` | 152 | 97.4% |
| `1KGL_A` | `1KGL` | `A` | `5319` | `P02696` | `2.40.128.20` | `2.40.128` | 135 | 97.8% |
| `1SMX_A` | `1SMX` | `A` | `6122` | `P21513` | `2.40.50.140` | `2.40.50` | 87 | 100.0% |
| `1TIP_A` | `1TIP` | `A` | `5935` | `P07953` | `3.40.50.1240` | `3.40.50` | 190 | 80.5% |
| `1UBQ_A` | `1UBQ` | `A` | `5387` | `P0CG48` | `3.10.20.90` | `3.10.20` | 76 | 100.0% |
| `1WH9_A` | `1WH9` | `A` | `10095` | `P23396` | `3.30.300.20` | `3.30.300` | 92 | 87.0% |
| `1X5W_A` | `1X5W` | `A` | `11312` | `Q9NTW7` | `3.30.160.60` | `3.30.160` | 70 | 81.4% |
| `1XFL_A` | `1XFL` | `A` | `6318` | `P29448` | `3.40.30.10` | `3.40.30` | 114 | 100.0% |
| `1Z6H_A` | `1Z6H` | `A` | `6599` | `C0H419` | `2.40.50.100` | `2.40.50` | 72 | 98.6% |
| `1ZG2_A` | `1ZG2` | `A` | `6717` | `Q9KGL3` | `3.40.1440.10` | `3.40.1440` | 94 | 100.0% |
| `1ZR9_A` | `1ZR9` | `A` | `6682` | `O00488` | `3.30.160.60` | `3.30.160` | 67 | 100.0% |
| `1ZU1_A` | `1ZU1` | `A` | `6655` | `Q8AVN9` | `3.30.160.60` | `3.30.160` | 127 | 100.0% |
| `2B7E_A` | `2B7E` | `A` | `6850` | `P33203` | `1.10.10.440` | `1.10.10` | 59 | 98.3% |
| `2C0S_A` | `2C0S` | `A` | `7349` | `A0A6L7H8X9` | `4.10.280.10` | `4.10.280` | 64 | 89.1% |
| `2CDV_A` | `2CDV` | `A` | `5333` | `P00132` | `3.90.10.10` | `3.90.10` | 107 | 94.4% |
| `2E9J_A` | `2E9J` | `A` | `10143` | `O75369` | `2.60.40.10` | `2.60.40` | 119 | 92.4% |
| `2FI2_A` | `2FI2` | `A` | `6957` | `P28698` | `1.10.4020.10` | `1.10.4020` | 94 | 97.9% |
| `2HFQ_A` | `2HFQ` | `A` | `7266` | `Q82U33` | `3.10.510.10` | `3.10.510` | 85 | 100.0% |
| `2IDA_A` | `2IDA` | `A` | `7297` | `Q6NA67` | `3.30.40.10` | `3.30.40` | 102 | 100.0% |
| `2JRZ_A` | `2JRZ` | `A` | `15348` | `P41229` | `1.10.150.60` | `1.10.150` | 117 | 97.4% |
| `2JYA_A` | `2JYA` | `A` | `15585` | `A9CIL3` | `3.30.160.190` | `3.30.160` | 80 | 97.5% |
| `2JZ5_A` | `2JZ5` | `A` | `15608` | `Q87J34` | `3.30.70.860` | `3.30.70` | 91 | 94.5% |
| `2K0D_X` | `2K0D` | `X` | `15645` | `Q03708` | `1.10.1200.20` | `1.10.1200` | 93 | 100.0% |
| `2K0Z_A` | `2K0Z` | `A` | `15661` | `O25821` | `3.40.250.10` | `3.40.250` | 110 | 99.1% |
| `2K1B_A` | `2K1B` | `A` | `15674` | `O95931` | `2.40.50.40` | `2.40.50` | 55 | 100.0% |
| `2K2M_A` | `2K2M` | `A` | `15710` | `Q8TE68` | `2.30.30.40` | `2.30.30` | 68 | 95.6% |
| `2K3R_A` | `2K3R` | `A` | `15776` | `Q8U0H6` | `1.20.5.420` | `1.20.5` | 91 | 85.7% |
| `2K49_A` | `2K49` | `A` | `15791` | `Q8EAL4` | `2.30.29.80` | `2.30.29` | 118 | 94.9% |
| `2K5E_A` | `2K5E` | `A` | `15833` | `Q74DN8` | `1.10.3910.10` | `1.10.3910` | 73 | 91.8% |
| `2K5V_A` | `2K5V` | `A` | `15849` | `Q6LYF9` | `2.40.50.140` | `2.40.50` | 98 | 100.0% |
| `2K7V_A` | `2K7V` | `A` | `15931` | `P06959` | `2.40.50.100` | `2.40.50` | 85 | 75.3% |
| `2KBI_A` | `2KBI` | `A` | `16045` | `Q14524` | `1.10.238.10` | `1.10.238` | 97 | 99.0% |
| `2KCL_A` | `2KCL` | `A` | `16084` | `Q2S6C5` | `1.25.40.10` | `1.25.40` | 99 | 96.0% |
| `2KCP_A` | `2KCP` | `A` | `16091` | `O26309` | `2.40.10.310` | `2.40.10` | 98 | 95.9% |
| `2KCW_A` | `2KCW` | `A` | `16098` | `P09163` | `3.40.630.30` | `3.40.630` | 147 | 99.3% |
| `2KD1_A` | `2KD1` | `A` | `16102` | `Q81GD4` | `1.10.150.130` | `1.10.150` | 118 | 93.2% |
| `2KIF_A` | `2KIF` | `A` | `16272` | `A6B4U8` | `1.10.10.10` | `1.10.10` | 102 | 100.0% |
| `2KKY_A` | `2KKY` | `A` | `16375` | `Q8X509` | `3.30.40.80` | `3.30.40` | 102 | 100.0% |
| `2KKZ_A` | `2KKZ` | `A` | `16376` | `P03495` | `3.30.420.330` | `3.30.420` | 134 | 99.3% |
| `2KL6_A` | `2KL6` | `A` | `16385` | `Q8U1U6` | `2.60.40.10` | `2.60.40` | 108 | 100.0% |
| `2KM4_A` | `2KM4` | `A` | `16411` | `Q05543` | `1.25.40.90` | `1.25.40` | 137 | 98.5% |
| `2KNR_A` | `2KNR` | `A` | `16476` | `Q7D0C7` | `3.40.1530.20` | `3.40.1530` | 118 | 100.0% |
| `2KP6_A` | `2KP6` | `A` | `16545` | `Q7P1H6` | `1.10.10.1130` | `1.10.10` | 79 | 97.5% |
| `2KRS_A` | `2KRS` | `A` | `16647` | `Q8XMT2` | `2.30.30.40` | `2.30.30` | 74 | 93.2% |
| `2KT1_A` | `2KT1` | `A` | `16681` | `O00459` | `2.30.30.40` | `2.30.30` | 82 | 97.6% |
| `2KVC_A` | `2KVC` | `A` | `16774` | `A5TZS3` | `1.10.150.430` | `1.10.150` | 103 | 98.1% |
| `2KW9_A` | `2KW9` | `A` | `16816` | `Q969V6` | `1.10.720.30` | `1.10.720` | 75 | 85.3% |
| `2KZ5_A` | `2KZ5` | `A` | `16998` | `Q16621` | `1.10.880.10` | `1.10.880` | 91 | 89.0% |
| `2KZC_A` | `2KZC` | `A` | `17002` | `Q28P99` | `1.10.790.20` | `1.10.790` | 85 | 98.8% |
| `2KZW_A` | `2KZW` | `A` | `17021` | `Q8PSA4` | `2.60.40.10` | `2.60.40` | 145 | 84.8% |
| `2L30_A` | `2L30` | `A` | `17157` | `P09874` | `3.30.1740.10` | `3.30.1740` | 108 | 96.3% |
| `2L3B_A` | `2L3B` | `A` | `17176` | `Q8ABM6` | `2.60.40.2410` | `2.60.40` | 130 | 94.6% |
| `2L59_A` | `2L59` | `A` | `17268` | `P9WG67` | `3.40.30.10` | `3.40.30` | 116 | 89.7% |
| `2L76_A` | `2L76` | `A` | `17345` | `Q8NCF5` | `3.10.20.90` | `3.10.20` | 95 | 100.0% |
| `2LAH_A` | `2LAH` | `A` | `17524` | `O43683` | `1.25.40.430` | `1.25.40` | 160 | 92.5% |
| `2LC0_A` | `2LC0` | `A` | `17585` | `P71590` | `3.30.2320.60` | `3.30.2320` | 132 | 95.5% |
| `2LDU_A` | `2LDU` | `A` | `17683` | `Q00613` | `1.10.10.10` | `1.10.10` | 125 | 92.0% |
| `2LEK_A` | `2LEK` | `A` | `16377` | `Q6N3W8` | `3.10.20.30` | `3.10.20` | 73 | 94.5% |
| `2LGQ_A` | `2LGQ` | `A` | `17821` | `Q14061` | `1.10.287.1130` | `1.10.287` | 67 | 89.6% |
| `2LHS_A` | `2LHS` | `A` | `17160` | `P9WEM3` | `2.70.50.50` | `2.70.50` | 170 | 99.4% |
| `2LIE_A` | `2LIE` | `A` | `17890` | `B3GA02` | `2.80.10.50` | `2.80.10` | 153 | 95.4% |
| `2LIZ_A` | `2LIZ` | `A` | `17911` | `P0C6U8` | `1.10.1840.10` | `1.10.1840` | 120 | 100.0% |
| `2LJW_A` | `2LJW` | `A` | `17965` | `A0ACD6B961` | `3.30.428.40` | `3.30.428` | 104 | 100.0% |
| `2LL8_A` | `2LL8` | `A` | `18032` | `Q6N882` | `1.10.1200.10` | `1.10.1200` | 101 | 95.0% |
| `2LM4_A` | `2LM4` | `A` | `18098` | `Q08230` | `1.10.150.250` | `1.10.150` | 109 | 91.7% |
| `2LPJ_A` | `2LPJ` | `A` | `18262` | `Q05H60` | `1.10.274.70` | `1.10.274` | 137 | 97.1% |
| `2LQL_A` | `2LQL` | `A` | `18318` | `Q9BSY4` | `1.10.287.2900` | `1.10.287` | 83 | 92.8% |
| `2LQM_A` | `2LQM` | `A` | `18320` | `O58001` | `2.170.16.10` | `2.170.16` | 174 | 96.6% |
| `2LR8_A` | `2LR8` | `A` | `18352` | `Q9UKL3` | `1.10.10.60` | `1.10.10` | 70 | 92.9% |
| `2LRI_C` | `2LRI` | `C` | `18374` | `O43918` | `3.30.40.10` | `3.30.40` | 66 | 98.5% |
| `2LRN_A` | `2LRN` | `A` | `18387` | `A6L2G9` | `3.40.30.10` | `3.40.30` | 152 | 96.7% |
| `2LRS_A` | `2LRS` | `A` | `18393` | `Q9SP32` | `3.30.160.20` | `3.30.160` | 71 | 100.0% |
| `2LRV_A` | `2LRV` | `A` | `18283` | `P0AB46` | `1.10.890.30` | `1.10.890` | 84 | 98.8% |
| `2LSO_A` | `2LSO` | `A` | `18438` | `Q92522` | `1.10.10.10` | `1.10.10` | 83 | 97.6% |
| `2LSY_A` | `2LSY` | `A` | `18455` | `Q9UBZ9` | `1.20.58.1280` | `1.20.58` | 95 | 94.7% |
| `2LWY_A` | `2LWY` | `A` | `18653` | `A3DH25` | `2.170.16.10` | `2.170.16` | 138 | 99.3% |
| `2LY9_A` | `2LY9` | `A` | `18714` | `Q9UKY1` | `1.10.10.60` | `1.10.10` | 74 | 95.9% |
| `2LYV_A` | `2LYV` | `A` | `18728` | `P09651` | `3.30.70.330` | `3.30.70` | 196 | 92.3% |
| `2M0P_A` | `2M0P` | `A` | `18816` | `P98164` | `4.10.400.10` | `4.10.400` | 46 | 97.8% |
| `2M2D_A` | `2M2D` | `A` | `18908` | `Q15116` | `2.60.40.10` | `2.60.40` | 118 | 98.3% |
| `2M2I_A` | `2M2I` | `A` | `18916` | `P12689` | `3.40.50.10190` | `3.40.50` | 94 | 92.6% |
| `2M3L_A` | `2M3L` | `A` | `18967` | `P26554` | `3.30.240.40` | `3.30.240` | 76 | 97.4% |
| `2M3X_A` | `2M3X` | `A` | `18978` | `O59169` | `2.40.10.360` | `2.40.10` | 147 | 98.0% |
| `2M4F_A` | `2M4F` | `A` | `19001` | `Q45001` | `2.30.31.50` | `2.30.31` | 151 | 100.0% |
| `2M4N_A` | `2M4N` | `A` | `19014` | `Q9BIC1` | `3.10.20.90` | `3.10.20` | 108 | 100.0% |
| `2M5J_A` | `2M5J` | `A` | `18201` | `Q79AD2` | `3.55.50.30` | `3.55.50` | 107 | 99.1% |
| `2M6U_A` | `2M6U` | `A` | `19155` | `G6W2B2` | `1.20.81.20` | `1.20.81` | 82 | 97.6% |
| `2M7Q_A` | `2M7Q` | `A` | `19201` | `Q86VP1` | `6.20.250.40` | `6.20.250` | 69 | 98.6% |
| `2M9K_A` | `2M9K` | `A` | `19298` | `Q6ZRY4` | `3.30.70.330` | `3.30.70` | 93 | 98.9% |
| `2MA8_A` | `2MA8` | `A` | `15090` | `Q7CR88` | `3.30.1660.10` | `3.30.1660` | 83 | 95.2% |
| `2MC4_A` | `2MC4` | `A` | `19427` | `O52732` | `1.10.10.1930` | `1.10.10` | 89 | 95.5% |
| `2MDR_A` | `2MDR` | `A` | `19502` | `P55265` | `3.30.160.20` | `3.30.160` | 94 | 91.5% |
| `2MEY_A` | `2MEY` | `A` | `7276` | `O96870` | `1.20.58.970` | `1.20.58` | 117 | 100.0% |
| `2MF4_A` | `2MF4` | `A` | `19536` | `A4PHN0` | `1.10.1200.10` | `1.10.1200` | 85 | 98.8% |
| `2MGX_A` | `2MGX` | `A` | `19607` | `Q9HD15` | `1.20.940.10` | `1.20.940` | 131 | 98.5% |
| `2MHJ_A` | `2MHJ` | `A` | `19635` | `Q9HVN7` | `3.10.20.310` | `3.10.20` | 95 | 91.6% |
| `2MIZ_A` | `2MIZ` | `A` | `19699` | `A2Q6L0` | `2.60.40.2900` | `2.60.40` | 156 | 91.0% |
| `2MN5_A` | `2MN5` | `A` | `19880` | `D6RKI7` | `3.30.30.140` | `3.30.30` | 56 | 100.0% |
| `2MNG_A` | `2MNG` | `A` | `19891` | `Q9Y3Q4` | `2.60.120.10` | `2.60.120` | 131 | 93.1% |
| `2MRL_A` | `2MRL` | `A` | `25083` | `Q2SV23` | `1.10.150.610` | `1.10.150` | 72 | 100.0% |
| `2MVZ_A` | `2MVZ` | `A` | `18628` | `Q5KXK3` | `2.40.100.10` | `2.40.100` | 146 | 99.3% |
| `2MXD_A` | `2MXD` | `A` | `25404` | `Q9QEJ5` | `1.10.260.110` | `1.10.260` | 62 | 98.4% |
| `2MY9_A` | `2MY9` | `A` | `25448` | `Q9UL19` | `3.90.1720.10` | `3.90.1720` | 125 | 95.2% |
| `2N0M_A` | `2N0M` | `A` | `18636` | `Q5F809` | `2.60.40.420` | `2.60.40` | 130 | 100.0% |
| `2N2M_A` | `2N2M` | `A` | `25281` | `Q8D0Z7` | `2.40.160.20` | `2.40.160` | 156 | 93.6% |
| `2N7B_A` | `2N7B` | `A` | `25799` | `Q9NYZ4` | `2.60.40.10` | `2.60.40` | 145 | 100.0% |
| `2N8X_A` | `2N8X` | `A` | `25869` | `Q9HX32` | `3.30.160.150` | `3.30.160` | 166 | 97.0% |
| `2NCN_A` | `2NCN` | `A` | `26603` | `Q9BXW4` | `3.10.20.90` | `3.10.20` | 128 | 99.2% |
| `2NCP_A` | `2NCP` | `A` | `26724` | `P09226` | `1.10.238.10` | `1.10.238` | 102 | 99.0% |
| `2RUC_A` | `2RUC` | `A` | `11559` | `Q13526` | `3.10.50.40` | `3.10.50` | 117 | 97.4% |
| `2RUH_A` | `2RUH` | `A` | `11569` | `Q00987` | `1.10.245.10` | `1.10.245` | 131 | 96.2% |
| `2X8N_A` | `2X8N` | `A` | `16790` | `Q7NZQ8` | `3.30.2020.40` | `3.30.2020` | 109 | 100.0% |
| `4CRP_A` | `4CRP` | `A` | `19824` | `P04629` | `2.60.40.10` | `2.60.40` | 107 | 95.3% |
| `4CSQ_A` | `4CSQ` | `A` | `19848` | `Q9HXK4` | `2.30.29.190` | `2.30.29` | 113 | 99.1% |
| `5IEJ_A` | `5IEJ` | `A` | `30025` | `A0A1C7D1A8` | `3.40.50.2300` | `3.40.50` | 118 | 98.3% |
| `5JPX_A` | `5JPX` | `A` | `30075` | `P19474` | `3.30.160.60` | `3.30.160` | 45 | 93.3% |
| `5KQC_A` | `5KQC` | `A` | `30131` | `Q2G1F1` | `2.70.70.10` | `2.70.70` | 145 | 95.9% |
| `5NPA_A` | `5NPA` | `A` | `34123` | `Q9VJY9` | `3.30.160.20` | `3.30.160` | 67 | 82.1% |
| `5O2Y_A` | `5O2Y` | `A` | `26919` | `A0AAI9GNC7` | `3.30.700.10` | `3.30.700` | 116 | 88.8% |
| `5OMZ_A` | `5OMZ` | `A` | `34167` | `A0A0X8GJ44` | `2.60.40.350` | `2.60.40` | 115 | 99.1% |
| `5T8A_A` | `5T8A` | `A` | `30172` | `P01451` | `2.10.60.10` | `2.10.60` | 61 | 100.0% |
| `5U6H_A` | `5U6H` | `A` | `30207` | `Q9NR56` | `3.30.1370.210` | `3.30.1370` | 92 | 91.3% |
| `5WOT_A` | `5WOT` | `A` | `30322` | `P00778` | `2.40.10.10` | `2.40.10` | 198 | 98.5% |
| `5X3Y_A` | `5X3Y` | `A` | `36058` | `Q61474` | `3.30.70.330` | `3.30.70` | 96 | 93.8% |
| `5XV9_A` | `5XV9` | `A` | `36102` | `Q488P6` | `2.40.50.140` | `2.40.50` | 68 | 98.5% |
| `6DO7_A` | `6DO7` | `A` | `30478` | `P07148` | `2.40.128.20` | `2.40.128` | 135 | 94.8% |
