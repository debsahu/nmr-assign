# Real Time-Domain Benchmark Cohort — Provenance & Pre-Registration (`Q1`)

## Source & Licence
- **Database**: Biological Magnetic Resonance Data Bank (BMRB, https://bmrb.io; Hoch et al., *Nucleic Acids Res.* **2023**, 51, D368–D376; Ulrich et al., *Nucleic Acids Res.* **2008**, 36, D402–D408).
- **Data licence**: Public domain / CC0 1.0 (except `ubiquitin_local`, local 500 MHz Bruker `pdata`/FID acquisition).
- **Raw FID storage**: Time-domain archives are fetched reproducibly via `nmrassign fetch-bmrb --id <bmrb_id> --dest <entry_dir>` into git-ignored local storage (`data/real_bmrb/<bmrb_id>` or `NMR_REAL_DATA_DIR`) with per-file FIPS PUB 180-4 SHA-256 verification; multi-megabyte raw FIDs are never committed to git.

## Frozen Manifest Pre-Registration Digest (`Q1`)
- **Manifest path**: `data/benchmark/real_manifest.tsv`
- **Manifest FIPS 180-4 SHA-256**: `dc580fec6e5b9108b17abc27bfefa53da59030256f534d366502c28cdf6ea5c6`
- **Programmatic enforcement**: `real_benchmark::tests::test_committed_real_manifest_tsv_schema_splits_and_disjointness` computes the streaming SHA-256 digest of `data/benchmark/real_manifest.tsv` and asserts that this exact hex digest is recorded in `data/benchmark/PROVENANCE.md`, alongside 100% disjointness against `data/priors/PROVENANCE.md` (21 entries) and `data/benchmark/manifest.tsv` (12 entries).

## Cohort Structure & Rotation History (`26` Entries Total)
1. **`split=dev`, `cohort=dev` (`10` entries)**:
   - `ubiquitin_local` (`Ubiquitin`, 76 aa, Bruker 500 MHz), `51439` (`CISD3`, 93 aa, Bruker 900 MHz), `53371` (`hDVL2_PDZ`, 102 aa, Bruker 600 MHz), `18465` (`OR135`, 110 aa, Varian 600 MHz), `15386` (`StR82`, 77 aa, Bruker 800 MHz), `5329` (`WR41`, 94 aa, Varian 600 MHz), `6318` (`At3g51030`, 124 aa, Bruker 600 MHz), `15476` (`SR384`, 54 aa, Bruker 800 MHz), `16385` (`PfR193A`, 114 aa, Bruker 600 MHz), `53280` (`Ap3`, 60 aa, Bruker 850 MHz).
2. **`split=dev`, `cohort=dev_from_p10` (`3` entries promoted from `cold_p10` in Phase 12)**:
   - `16072` (`SSP0047`, 120 aa, Bruker 850 MHz), `16782` (`SgR171`, 120 aa, Varian 600 MHz), `17092` (`HR4435B`, 83 aa, Bruker 600 MHz).
   - Combined `dev` split = **`13` proteins** (`10` Bruker, `3` Varian; `54–124` aa).
3. **`split=heldout`, `cohort=seen_p8_p9` (`6` previously-evaluated heldout entries, retained for longitudinal tracking)**:
   - `5845` (`ZR31`, 102 aa, Varian 600 MHz), `17030` (`HR4694F`, 95 aa, Bruker 600 MHz), `18145` (`OR126`, 83 aa, Bruker 800 MHz), `51351` (`TSR3`, 75 aa, Varian 600 MHz), `16099` (`Kazal-1`, 74 aa, Bruker 800 MHz), `18489` (`NFU1`, 107 aa, Bruker 800 MHz).
4. **`split=heldout`, `cohort=cold_p12` (`7` fresh pre-screened monomeric entries, evaluated strictly once after pre-registration commit)**:

| BMRB ID | Protein | Residues | Field (MHz) | Vendor | Raw Time-Domain Backbone Experiments | Deposited Shift Coverage (`HN` / `CA` / `CB` / `C`) | Description |
|---|---|---:|---:|---|---|---|---|
| `5691` | `RS28_PYRHO` | 82 | 600 | Varian | `HSQC,HNCO,HNCACB,CBCACONH` | `92.5% / 93.9% / 84.1% / 92.7%` | *Pyrococcus horikoshii* 30S ribosomal protein S28e |
| `6585` | `At1g24000.1` | 122 | 600 | Varian | `HSQC,HNCO,HNCA,HNCACB,CBCACONH` | `94.1% / 94.3% / 87.7% / 89.3%` | *Arabidopsis thaliana* protein At1g24000.1 |
| `15491` | `VfR117` | 88 | 800 | Bruker | `HSQC,HNCO,HNCACB,CBCACONH,HNCACO` | `90.7% / 92.0% / 89.8% / 92.0%` | *Vibrio fischeri* protein Q5E7H1 (NESG VfR117) |
| `16096` | `DvR115G` | 94 | 600 | Varian | `HSQC,HNCACB,CBCACONH,HNCOCA,HNCACO` | `90.9% / 92.6% / 83.0% / 91.5%` | *Desulfovibrio vulgaris* apo CcmE OB-fold domain (NESG DvR115G) |
| `16376` | `OR8w187r` | 140 | 800 | Bruker | `HSQC,HNCO,HNCA,HNCACB,CBCACONH,HNCOCA,HNCACO` | `93.2% / 95.7% / 87.1% / 90.7%` | Monomeric W187R influenza A NS1 effector domain (NESG OR8w187r) |
| `18337` | `OR136` | 134 | 800 | Bruker | `HSQC,HNCO,HNCACB,CBCACONH` | `94.0% / 95.5% / 92.5% / 92.5%` | *De novo* P-loop NTPase fold protein (NESG OR136) |
| `18561` | `OR250` | 112 | 800 | Bruker | `HSQC,HNCO,HNCACB,CBCACONH` | `88.1% / 94.6% / 89.3% / 84.8%` | *De novo* PFK-fold protein (NESG OR250) |

## Pre-Screening & Zero-Solver Firewall Protocol (`Q1`)
1. **Disjointness & Homology Screen**: All public BMRB entries with `timedomain_data/` archives were filtered to remove every BMRB accession referenced anywhere in the repository (`21` empirical-prior entries in `data/priors/PROVENANCE.md`, `12` Phase-1 peak-list entries in `data/benchmark/manifest.tsv`, all `19` prior `real_manifest.tsv` entries, and all unit/integration test fixtures), plus entries homologous to Ubiquitin/GB1/NifU-NFU1/CISD families or marked perdeuterated (`2H`).
2. **NMR-STAR Metadata Screen**: Candidate `.str` headers were checked strictly for:
   - Single monomeric polypeptide chain (`50 <= L <= 160` canonical amino acids; oligomers, complexes, membrane/micelle samples, IDPs, and paramagnetic metalloproteins excluded),
   - `>= 80%` non-Proline `1H`/`15N` and `>= 80%` `13CA` deposited backbone chemical shifts in the primary `_Atom_chem_shift` loop (`>= 70%` `13CB` or `13C`).
3. **Header-Only Archive Verification (`ZERO` FFT / Peak-Picker / Solver Runs)**:
   - Candidate archives (`<= 650 MB`) were fetched via `nmrassign fetch-bmrb` and inspected **solely** via `nmrassign inspect --json` and `discover_entry_experiments` (`acqus` / `procpar` text headers only) to confirm the presence of a raw time-domain `2D 15N-HSQC` plus `>= 3` raw time-domain 3D triple-resonance backbone experiments (`HNCO`, `HNCA`, `HNCACB`, `CBCACONH`, `HNCOCA`, `HNCACO`). Archives containing only NOESY or 2D FIDs (`bmr25061`, `bmr16100`, `bmr16272`, `bmr25067`, `bmr17965`, `bmr18511`) or lacking discoverable 3D `HNCO` headers (`bmr5589`) were discarded at header inspection without ever invoking FFT processing, peak picking, or assignment.
4. **Pre-Registration Commit & Post-Freeze Raw Output Digests (`P1-1`)**:
   - All solver/QC/CLI code, the 13-protein `dev` isotonic calibration knot tables (`EVO_GRAPH_CALIBRATION_KNOTS`, `STANDARD_CALIBRATION_KNOTS`, and CPU-shrunk `STANDARD_GPU_CALIBRATION_KNOTS`), `data/benchmark/real_manifest.tsv`, and this `PROVENANCE.md` manifest digest were committed at **`be631d1` (`be631d1c863379c0b360e05ea8522ed05da89fb4`)** with commit timestamp **`2026-11-13T10:26:24-05:00` (`2026-11-13T15:26:24Z`)** **before** executing the single-shot cold evaluation on `split=heldout`.
   - The single-shot 3-engine `split=heldout` (`seen_p8_p9` + `cold_p12`, `seeds = 0..4`) evaluation ran from **`2026-11-13T15:27:05Z`** (start, **`+41 s`** post-commit gap after `be631d1`) to **`2026-11-13T15:30:57Z`** (end). Zero solver, QC, threshold, or calibration knot edits were made after `be631d1`.
   - Raw stdout and JSON benchmark artifacts archived locally in git-ignored `data/private/` carry the following verified FIPS PUB 180-4 SHA-256 digests:

| Local Post-Freeze Artifact (`git-ignored`) | FIPS 180-4 SHA-256 Digest |
|---|---|
| `data/private/phase12_cold_p12_results.txt` | `46ffa1bc35c6903211861488bbee8a40b764deeb6cb2f5f896c8a02d221d59dd` |
| `data/private/phase12_heldout_evo_graph.json` | `7e9d582f2096675b5304e7ab7f0e5a28638c25e0c763b3d7679a60e75e00b983` |
| `data/private/phase12_heldout_standard_cpu.json` | `7745a023070557183642cc8a4400ba4cb7bcc8f8d312429b657cc44a2c8bbfa3` |
| `data/private/phase12_heldout_standard_gpu.json` | `48b1490f7ecc7099a032d3daf821963ebfe680892b8be15993a8d3413181df62` |

## Calibration Scope & Hardware Provenance Disclosure (`P1-7`, `P1-8` & `Q2`)
- **Replica-count coupling & shrinkage selection (`P1-8`)**: `STANDARD_CALIBRATION_KNOTS` is fitted at `runs = 16` (`STANDARD_CPU_CALIBRATED_RUNS`), `STANDARD_GPU_CALIBRATION_KNOTS` at `runs = 64` (`STANDARD_GPU_CALIBRATED_RUNS`) on **`Apple M5 Pro` (`Metal` backend via `wgpu`, `f32` WGSL compute)** with empirical-Bayes shrinkage toward CPU `f64` bin counts (`CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT = 0.50`, selected on the same 13-fold `dev` OOF LOPO grid rather than nested inside each fold; nested `w` selection and `ECE_10` primary gating are tracked under `P1-8` for Phase 13), and `EVO_GRAPH_CALIBRATION_KNOTS` at `population_size = 64` (`EVO_GRAPH_CALIBRATED_RUNS`). Callers overriding `runs != fit_runs` emit a `stderr` warning and set `DatasetQcReport.confidence_calibrated = false`.
- **Candid LOPO scope & spent `cold_p12` notice (`P1-7`)**: 13-fold Leave-One-Protein-Out (LOPO) cross-validation on `dev` strictly certifies the post-solver isotonic PAV calibrator (`ECE_4`, `ECE_10`, and `ECE_12`). Upstream solver and dataset-QC structural constants were fixed with earlier `dev` entries visible; full end-to-end out-of-sample generalization at `v0.8.0` is measured by the single-shot `cold_p12` cohort. Having been observed once at `v0.8.0`, `cold_p12` is now spent for subsequent QC/gate selection: Phase 13 dataset-QC and protein-level HC gates (`P1-7`) must be fitted strictly on `dev` LOPO and validated on a newly pre-registered cold cohort.
