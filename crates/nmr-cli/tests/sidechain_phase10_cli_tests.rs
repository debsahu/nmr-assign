//! End-to-end CLI integration tests for `nmrassign sidechain` with Phase 10 aromatic ring
//! (`--hbcbgcdce`, `--aromatic-tocsy`, `--aromatic-noesy`) and Bayesian stereospecific prochiral
//! (`--stereospec`, `--j-couplings`, `--intra-noes`, `--pdb`) assignment.
//!
//! Provenance: Clean-room integration test suite citing:
//! - Yamazaki, Forman-Kay & Kay (*J. Am. Chem. Soc.* 1993, 115, 11054–11055)
//! - Pelton, Torchia, Meadow & Roseman (*Protein Sci.* 1993, 2, 543–558)
//! - Güntert, Braun, Billeter & Wüthrich (*J. Biomol. NMR* 1991, 1, 447–456)
//! - Lovell, Word, Richardson & Richardson (*Proteins* 2000, 40, 389–408)
//! - Markley et al. (*J. Mol. Biol.* 1998, 280, 933–952 — IUPAC NMR atom nomenclature & `Ambiguity_code`)

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use nmr_core::aromatic::{
    default_his_tautomer_map, simulate_aromatic_peaks, synthesize_aromatic_reference,
    AromaticSimConfig,
};
use nmr_core::nmrstar::{read_star_all_atoms, write_star};
use nmr_core::peaks::{write_peaks, Peak};
use nmr_core::sidechain::{
    simulate_sidechain_peaks, synthesize_all_atom_reference, SidechainSimConfig,
};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::stereospec::{
    parse_pdb_all_atoms_file, prochiral_pairs_for_residue, synthesize_ideal_observables_from_pdb,
    write_intra_noes_tsv, write_j_couplings_tsv, IntraNoeTsvRow, JCouplingTsvRow,
    ProchiralSiteKind,
};
use nmr_core::topology::SidechainPriors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn parse_star_ambiguity_codes(star_text: &str) -> BTreeMap<(i32, String), u8> {
    let mut map = BTreeMap::new();
    let mut in_shift_loop = false;
    for raw in star_text.lines() {
        let line = raw.trim();
        if line == "_Atom_chem_shift.Details" {
            in_shift_loop = true;
            continue;
        }
        if in_shift_loop {
            if line == "stop_" {
                break;
            }
            if line.is_empty() || line.starts_with('#') || line.starts_with('_') {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() >= 8 {
                if let (Ok(seq_id), Ok(amb)) = (cols[2].parse::<i32>(), cols[7].parse::<u8>()) {
                    map.insert((seq_id, cols[4].to_string()), amb);
                }
            }
        }
    }
    map
}

#[test]
fn sidechain_cli_aromatic_and_stereospecific_ambiguity_gating() {
    let root = repo_root();
    let star_in = root.join("data/fixtures/synthetic_ubiquitin.str");
    let pdb_in = root.join("data/pdb_refs/1d3z.pdb");
    let seed = 2026u64;

    let dir = env::temp_dir().join(format!("nmr_sidechain_p10_cli_test_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let raw_ref = read_star_all_atoms(&star_in, None).expect("read ubiquitin star");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let all_ref = synthesize_all_atom_reference(&raw_ref, &sc_priors, seed);
    let his_map = default_his_tautomer_map(&all_ref.sequence, seed);
    let truth_ref = synthesize_aromatic_reference(&all_ref, &sc_priors, &his_map, seed);

    // 1. Prepare backbone + aliphatic peaks TSV and split aromatic channel TSVs
    let (mut peaks_combined, _) = simulate(
        &truth_ref,
        &SimConfig {
            p_missing_peak: 0.02,
            spurious_rate: 0.01,
            seed,
            ..SimConfig::default()
        },
    );
    let sc_aliph = simulate_sidechain_peaks(
        &truth_ref,
        &SidechainSimConfig {
            include_aromatics: false,
            p_missing_peak: 0.02,
            spurious_rate: 0.01,
            seed,
            ..SidechainSimConfig::default()
        },
    );
    peaks_combined.extend(sc_aliph);

    let (arom_pks, trp_relays) = simulate_aromatic_peaks(
        &truth_ref,
        &his_map,
        &AromaticSimConfig {
            seed,
            ..AromaticSimConfig::default()
        },
    );
    let mut c13_hsqc = Vec::new();
    let mut hd_pks = Vec::new();
    let mut he_pks = Vec::new();
    let mut tocsy_pks = Vec::new();
    let mut his_pks = Vec::new();
    for p in arom_pks {
        match p.exp.as_str() {
            "AROM_C13HSQC" => c13_hsqc.push(p),
            "HBCBCGCDHD" => hd_pks.push(p),
            "HBCBCGCDCEHE" => he_pks.push(p),
            "AROM_HCCH_TOCSY" => tocsy_pks.push(p),
            "HIS_LR_N15HMQC" => his_pks.push(p),
            _ => {}
        }
    }
    // Synthesize HB2 -> HD1 aromatic tie-breaker NOE peaks for PHE4 & PHE45
    let mut arom_noe_pks: Vec<Peak> = trp_relays;
    for (k, rname) in truth_ref.sequence.iter().enumerate() {
        if rname == "PHE" || rname == "TYR" {
            if let (Some(&hd1), Some(&hb2)) = (
                truth_ref.shifts.get(&(k, "HD1".to_string())),
                truth_ref.shifts.get(&(k, "HB2".to_string())),
            ) {
                arom_noe_pks.push(Peak {
                    exp: "AROM_HB_NOE".to_string(),
                    h: hd1,
                    n: 0.0,
                    c: truth_ref.shifts.get(&(k, "CD1".to_string())).copied(),
                    sign: 1,
                    height: Some(60.0),
                    volume: Some(240.0),
                    h2: Some(hb2),
                });
            }
        }
    }

    let peaks_tsv = dir.join("bb_aliph.tsv");
    let arom_hsqc_tsv = dir.join("arom_c13hsqc.tsv");
    let hd_tsv = dir.join("hbcbgcdhd.tsv");
    let he_tsv = dir.join("hbcbgcdce.tsv");
    let tocsy_tsv = dir.join("arom_tocsy.tsv");
    let noe_tsv = dir.join("arom_noesy.tsv");
    let his_tsv = dir.join("his_hmqc.tsv");

    write_peaks(&peaks_tsv, &peaks_combined).unwrap();
    write_peaks(&arom_hsqc_tsv, &c13_hsqc).unwrap();
    write_peaks(&hd_tsv, &hd_pks).unwrap();
    write_peaks(&he_tsv, &he_pks).unwrap();
    write_peaks(&tocsy_tsv, &tocsy_pks).unwrap();
    write_peaks(&noe_tsv, &arom_noe_pks).unwrap();
    write_peaks(&his_tsv, &his_pks).unwrap();

    // 2. Synthesize J-coupling and intra-NOE TSVs for the first 35 residues of 1D3Z only,
    // leaving residues 36..76 without J/NOE data so BOTH Ambiguity_code=1 (committed) and
    // Ambiguity_code=2 (uncommitted prochiral) are exercised in one run!
    let pdb_res = parse_pdb_all_atoms_file(&pdb_in).expect("parse 1d3z");
    let mut j_rows = Vec::new();
    let mut intra_noe_rows = Vec::new();
    for (pi, pr) in pdb_res.iter().enumerate().take(35) {
        for spec in prochiral_pairs_for_residue(&pr.res_name) {
            let Some(obs) = synthesize_ideal_observables_from_pdb(&pdb_res, pi, spec) else {
                continue;
            };
            if let Some((ja, jb)) = obs.j_hh_vicinal_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_HAHB".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some((ja, jb)) = obs.j_nhb_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_NHB".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some((ja, jb)) = obs.j_ncg_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_NCG".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some((ja, jb)) = obs.j_cc_vicinal_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_CCG".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some(jm) = obs.j_methine_hahb_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_METHINE_HAHB".to_string(),
                    val_a_hz: jm,
                    val_b_hz: None,
                });
            }
            for np in &obs.noe_pairs {
                if spec.kind != ProchiralSiteKind::AsnCarboxamide
                    && spec.kind != ProchiralSiteKind::GlnCarboxamide
                {
                    intra_noe_rows.push(IntraNoeTsvRow {
                        seq_id: pr.seq_id,
                        res_name: pr.res_name.clone(),
                        site: spec.iupac_branch_1.to_string(),
                        anchor: np.anchor,
                        intensity_a: np.intensity_a,
                        intensity_b: np.intensity_b,
                    });
                }
            }
        }
    }
    let j_tsv = dir.join("j_couplings.tsv");
    let inoe_tsv = dir.join("intra_noes.tsv");
    write_j_couplings_tsv(&j_tsv, &j_rows).unwrap();
    write_intra_noes_tsv(&inoe_tsv, &intra_noe_rows).unwrap();

    // 3. Execute `nmrassign sidechain` with aromatic + stereospecific flags
    let out_star = dir.join("ubiquitin_phase10_all_atom.str");
    let bin = env!("CARGO_BIN_EXE_nmrassign");
    let run = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &star_in.to_string_lossy(),
            "--peaks",
            &peaks_tsv.to_string_lossy(),
            "--arom-c13hsqc",
            &arom_hsqc_tsv.to_string_lossy(),
            "--hbcbgcdhd",
            &hd_tsv.to_string_lossy(),
            "--hbcbgcdce",
            &he_tsv.to_string_lossy(),
            "--aromatic-tocsy",
            &tocsy_tsv.to_string_lossy(),
            "--aromatic-noesy",
            &noe_tsv.to_string_lossy(),
            "--his-hmqc",
            &his_tsv.to_string_lossy(),
            "--stereospec",
            "--j-couplings",
            &j_tsv.to_string_lossy(),
            "--intra-noes",
            &inoe_tsv.to_string_lossy(),
            "--pdb",
            &pdb_in.to_string_lossy(),
            "--runs",
            "8",
            "--steps",
            "1200",
            "--seed",
            "2026",
            "--out",
            &out_star.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign sidechain");

    assert!(
        run.status.success(),
        "nmrassign sidechain failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("Aromatic rings: assigned"),
        "missing aromatic summary in stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("Stereospecific prochiral: committed"),
        "missing stereospecific summary in stdout:\n{stdout}"
    );

    let star_text = fs::read_to_string(&out_star).expect("read written STAR");
    let parsed_ref = read_star_all_atoms(&out_star, None).expect("parse written STAR");
    assert_eq!(parsed_ref.sequence.len(), 76);

    // Verify aromatic ring atoms (PHE 4 / PHE 45 / TYR 59 / HIS 68) are present in output STAR
    let has_arom_atoms = parsed_ref
        .shifts
        .keys()
        .any(|(_, a)| matches!(a.as_str(), "CD1" | "CE1" | "CZ" | "HD1" | "HE1" | "ND1"));
    assert!(
        has_arom_atoms,
        "expected assigned aromatic ring atoms in STAR"
    );

    let amb_map = parse_star_ambiguity_codes(&star_text);
    let count_amb1_prochiral = amb_map
        .iter()
        .filter(|&(&(seq_id, ref atom), &code)| {
            seq_id <= 35 && matches!(atom.as_str(), "HB2" | "HB3" | "CG1" | "CG2") && code == 1
        })
        .count();
    let count_amb2_prochiral = amb_map
        .iter()
        .filter(|&(&(seq_id, ref atom), &code)| {
            seq_id > 35 && matches!(atom.as_str(), "HB2" | "HB3" | "CG1" | "CG2") && code == 2
        })
        .count();
    let count_amb3_ringflip = amb_map.values().filter(|&&code| code == 3).count();

    assert!(
        count_amb1_prochiral >= 6,
        "expected >= 6 committed stereospecific atoms (Ambiguity_code=1) in residues 1..35, got {count_amb1_prochiral}"
    );
    assert!(
        count_amb2_prochiral >= 6,
        "expected >= 6 uncommitted prochiral atoms (Ambiguity_code=2) in residues 36..76 without J/NOE, got {count_amb2_prochiral}"
    );
    assert!(
        count_amb3_ringflip >= 4,
        "expected Ambiguity_code=3 on PHE/TYR symmetric ring-flip atoms, got {count_amb3_ringflip}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sidechain_cli_real_star_without_peaks_preserves_shifts_and_resolves_stereospec() {
    let root = repo_root();
    let star_bb = root.join("data/fixtures/synthetic_ubiquitin.str");
    let pdb_in = root.join("data/pdb_refs/1d3z.pdb");
    let seed = 2026u64;

    let dir = env::temp_dir().join(format!(
        "nmr_sidechain_p10_real_star_no_peaks_test_{}",
        process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let raw_bb = read_star_all_atoms(&star_bb, None).expect("read ubiquitin star");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let ideal_ref = synthesize_all_atom_reference(&raw_bb, &sc_priors, seed);

    // Deliberately scramble `(HB2, HB3)` on residues 1..=20 in the input `.str` (non-stereospecific
    // input ordering) and swap `(ja, jb)` correspondingly so `SwapHypothesis::Swapped` restores
    // true IUPAC `(HB2, HB3)` ordering while keeping all non-prochiral shifts 100% untouched!
    let mut scrambled_ref = ideal_ref.clone();
    let mut swapped_hb_residues = Vec::new();
    for k in 0..20usize {
        if let (Some(&hb2), Some(&hb3)) = (
            ideal_ref.shifts.get(&(k, "HB2".to_string())),
            ideal_ref.shifts.get(&(k, "HB3".to_string())),
        ) {
            if (hb2 - hb3).abs() > 0.05 {
                scrambled_ref.shifts.insert((k, "HB2".to_string()), hb3);
                scrambled_ref.shifts.insert((k, "HB3".to_string()), hb2);
                swapped_hb_residues.push(k);
            }
        }
    }
    assert!(
        !swapped_hb_residues.is_empty(),
        "expected prochiral HB2/HB3 pairs in residues 1..=20"
    );

    let input_all_atom_star = dir.join("input_all_atom.str");
    write_star(&input_all_atom_star, &scrambled_ref, None).unwrap();
    // Re-read the `.3`-formatted STAR so comparisons check exact round-trip preservation
    let exact_in_ref = read_star_all_atoms(&input_all_atom_star, None).unwrap();

    let pdb_res = parse_pdb_all_atoms_file(&pdb_in).expect("parse 1d3z");
    let mut j_rows = Vec::new();
    for (pi, pr) in pdb_res.iter().enumerate().take(35) {
        let swapped_hb = swapped_hb_residues.contains(&pi);
        for spec in prochiral_pairs_for_residue(&pr.res_name) {
            let Some(obs) = synthesize_ideal_observables_from_pdb(&pdb_res, pi, spec) else {
                continue;
            };
            let flip = swapped_hb && spec.kind == ProchiralSiteKind::BetaMethylene;
            if let Some((ja, jb)) = obs.j_hh_vicinal_hz {
                let (va, vb) = if flip { (jb, ja) } else { (ja, jb) };
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_HAHB".to_string(),
                    val_a_hz: va,
                    val_b_hz: Some(vb),
                });
            }
            if let Some((ja, jb)) = obs.j_nhb_hz {
                let (va, vb) = if flip { (jb, ja) } else { (ja, jb) };
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_NHB".to_string(),
                    val_a_hz: va,
                    val_b_hz: Some(vb),
                });
            }
            if let Some((ja, jb)) = obs.j_ncg_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_NCG".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some((ja, jb)) = obs.j_cc_vicinal_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_CCG".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some(jm) = obs.j_methine_hahb_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_METHINE_HAHB".to_string(),
                    val_a_hz: jm,
                    val_b_hz: None,
                });
            }
        }
    }
    let j_tsv = dir.join("j_only.tsv");
    write_j_couplings_tsv(&j_tsv, &j_rows).unwrap();

    // 1. Run `nmrassign sidechain` with NO `--peaks`: must preserve exact input shifts up to stereospec swaps!
    let out_star = dir.join("out_stereospec_no_peaks.str");
    let bin = env!("CARGO_BIN_EXE_nmrassign");
    let run = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &input_all_atom_star.to_string_lossy(),
            "--stereospec",
            "--j-couplings",
            &j_tsv.to_string_lossy(),
            "--pdb",
            &pdb_in.to_string_lossy(),
            "--out",
            &out_star.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign sidechain without --peaks");

    assert!(
        run.status.success(),
        "nmrassign sidechain (no --peaks) failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let out_ref = read_star_all_atoms(&out_star, None).expect("parse out STAR");
    assert_eq!(
        out_ref.shifts.len(),
        exact_in_ref.shifts.len(),
        "shift count must match input STAR exactly without fabricating or dropping atoms"
    );

    const PROCHIRAL_ATOMS: &[&str] = &[
        "HA2", "HA3", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HG12", "HG13",
        "HD21", "HD22", "HE21", "HE22", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23", "HD11",
        "HD12", "HD13", "HD21", "HD22", "HD23", "CG1", "CG2", "CD1", "CD2",
    ];
    for (&(k, ref atom), &in_ppm) in &exact_in_ref.shifts {
        let out_ppm = *out_ref
            .shifts
            .get(&(k, atom.clone()))
            .unwrap_or_else(|| panic!("missing shift ({k}, {atom}) in output STAR"));
        if !PROCHIRAL_ATOMS.contains(&atom.as_str()) {
            assert!(
                (out_ppm - in_ppm).abs() <= 1e-6,
                "non-prochiral shift ({k} {}, {atom}) was altered: in={in_ppm:.3}, out={out_ppm:.3}",
                exact_in_ref.sequence[k]
            );
        }
    }

    // Verify prochiral `(HB2, HB3)` pairs preserve exact `{v1, v2}` values and restore swapped IUPAC order
    let mut restored_swaps = 0usize;
    for &k in &swapped_hb_residues {
        let in_a = exact_in_ref.shifts[&(k, "HB2".to_string())];
        let in_b = exact_in_ref.shifts[&(k, "HB3".to_string())];
        let out_a = out_ref.shifts[&(k, "HB2".to_string())];
        let out_b = out_ref.shifts[&(k, "HB3".to_string())];
        let same_direct = (out_a - in_a).abs() <= 1e-6 && (out_b - in_b).abs() <= 1e-6;
        let same_swapped = (out_a - in_b).abs() <= 1e-6 && (out_b - in_a).abs() <= 1e-6;
        assert!(
            same_direct || same_swapped,
            "prochiral HB2/HB3 values altered on residue {}: in=({in_a:.3}, {in_b:.3}), out=({out_a:.3}, {out_b:.3})",
            k + 1
        );
        let ideal_hb2 = (ideal_ref.shifts[&(k, "HB2".to_string())] * 1000.0).round() / 1000.0;
        if same_swapped && (out_a - ideal_hb2).abs() <= 1e-3 {
            restored_swaps += 1;
        }
    }
    assert!(
        restored_swaps >= 4,
        "expected >= 4 scrambled HB2/HB3 pairs restored to true IUPAC order by 3J couplings, got {restored_swaps}"
    );

    // 2. Also verify `--peaks <sidechain_only.tsv>` (no backbone peaks, `bb_peaks.is_empty()`) + `--aromatic`
    // assigns aliphatic sidechains onto `--star <backbone.str>` via `assign_sidechains` without fabricating fake aromatics!
    let sc_only_peaks = simulate_sidechain_peaks(
        &ideal_ref,
        &SidechainSimConfig {
            include_aromatics: false,
            p_missing_peak: 0.0,
            spurious_rate: 0.0,
            seed,
            ..SidechainSimConfig::default()
        },
    );
    let sc_only_tsv = dir.join("sc_only.tsv");
    write_peaks(&sc_only_tsv, &sc_only_peaks).unwrap();
    let sc_only_out = dir.join("out_sc_only.str");
    let run_sc_only = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &star_bb.to_string_lossy(),
            "--peaks",
            &sc_only_tsv.to_string_lossy(),
            "--aromatic",
            "--out",
            &sc_only_out.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign sidechain with sidechain-only peaks");
    assert!(
        run_sc_only.status.success(),
        "sidechain-only --peaks failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_sc_only.stdout),
        String::from_utf8_lossy(&run_sc_only.stderr)
    );
    let sc_only_ref = read_star_all_atoms(&sc_only_out, None).unwrap();
    assert!(
        sc_only_ref.shifts.len() > raw_bb.shifts.len() + 100,
        "expected aliphatic sidechains assigned onto backbone STAR, got {} shifts",
        sc_only_ref.shifts.len()
    );
    let fabricated_arom = sc_only_ref.shifts.keys().any(|&(k, ref a)| {
        matches!(
            sc_only_ref.sequence[k].as_str(),
            "PHE" | "TYR" | "HIS" | "TRP"
        ) && matches!(
            a.as_str(),
            "CD1" | "CE1" | "CZ" | "HD1" | "HE1" | "HZ" | "ND1"
        )
    });
    assert!(
        !fabricated_arom,
        "must not fabricate synthetic aromatic shifts when --peaks is provided without aromatic peaks"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sidechain_cli_p0_1_rejects_implicit_simulation_and_invalid_stereo_threshold() {
    let root = repo_root();
    let star_in = root.join("data/fixtures/synthetic_ubiquitin.str");
    let bin = env!("CARGO_BIN_EXE_nmrassign");

    let dir = env::temp_dir().join(format!("nmr_sidechain_p0_1_err_test_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // (a1) `--aromatic` without `--peaks`, explicit aromatic peak lists, or `--simulate` MUST fail cleanly!
    let out_arom_err = dir.join("should_not_exist_arom.str");
    let run_arom_err = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &star_in.to_string_lossy(),
            "--aromatic",
            "--out",
            &out_arom_err.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign sidechain --aromatic without inputs");
    assert!(
        !run_arom_err.status.success(),
        "sidechain --aromatic without peak inputs or --simulate must fail instead of fabricating shifts"
    );
    assert!(
        !out_arom_err.exists(),
        "failed command must not write synthetic output STAR"
    );

    // (a2) Bare `sidechain --star ... --out ...` without `--peaks`, `--stereospec`, or `--simulate` MUST fail cleanly!
    let out_bare_err = dir.join("should_not_exist_bare.str");
    let run_bare_err = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &star_in.to_string_lossy(),
            "--out",
            &out_bare_err.to_string_lossy(),
        ])
        .output()
        .expect("spawn bare nmrassign sidechain");
    assert!(!run_bare_err.status.success());
    assert!(!out_bare_err.exists());

    // (a3) `--stereo-threshold` outside `(0.5, 1.0]` or `NaN` MUST fail cleanly!
    for bad_thr in ["0.5", "0.3", "1.05", "-0.1", "NaN"] {
        let run_thr = process::Command::new(bin)
            .args([
                "sidechain",
                "--star",
                &star_in.to_string_lossy(),
                "--stereospec",
                "--stereo-threshold",
                bad_thr,
            ])
            .output()
            .expect("spawn nmrassign sidechain bad --stereo-threshold");
        assert!(
            !run_thr.status.success(),
            "--stereo-threshold {bad_thr} must be rejected"
        );
    }

    // (a4) Explicit `--simulate` opt-in still works for intentional synthetic benchmarking
    let out_sim = dir.join("explicit_sim.str");
    let run_sim = process::Command::new(bin)
        .args([
            "sidechain",
            "--star",
            &star_in.to_string_lossy(),
            "--aromatic",
            "--simulate",
            "--runs",
            "4",
            "--steps",
            "600",
            "--seed",
            "2026",
            "--out",
            &out_sim.to_string_lossy(),
        ])
        .output()
        .expect("spawn nmrassign sidechain --simulate");
    assert!(
        run_sim.status.success(),
        "explicit --simulate failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run_sim.stdout),
        String::from_utf8_lossy(&run_sim.stderr)
    );
    assert!(out_sim.exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sidechain_cli_p0_1_real_aromatic_and_stereospec_roundtrip_preserves_untouched_atoms_byte_identical(
) {
    let root = repo_root();
    let star_bb = root.join("data/fixtures/synthetic_ubiquitin.str");
    let pdb_in = root.join("data/pdb_refs/1d3z.pdb");
    let seed = 2026u64;
    let bin = env!("CARGO_BIN_EXE_nmrassign");

    let dir = env::temp_dir().join(format!(
        "nmr_sidechain_p0_1_roundtrip_test_{}",
        process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let raw_bb = read_star_all_atoms(&star_bb, None).expect("read ubiquitin star");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let mut in_ref = synthesize_all_atom_reference(&raw_bb, &sc_priors, seed);
    let his_map = default_his_tautomer_map(&in_ref.sequence, seed);
    let arom_truth = synthesize_aromatic_reference(&in_ref, &sc_priors, &his_map, seed);

    // Ensure Phe45 (k = 44) CZ/HZ and Lys63 (k = 62) H/N are present on `in.str` with sentinel values,
    // and intentionally omit Phe45's CZ/HZ peaks from the aromatic peak lists so Phe45 CZ/HZ is untouched!
    assert_eq!(in_ref.sequence[44], "PHE");
    assert_eq!(in_ref.sequence[62], "LYS");
    let phe45_cz_ppm = 129.345;
    let phe45_hz_ppm = 7.182;
    let lys63_h_ppm = *in_ref.shifts.entry((62, "H".to_string())).or_insert(8.412);
    let lys63_n_ppm = *in_ref
        .shifts
        .entry((62, "N".to_string()))
        .or_insert(121.634);
    in_ref.shifts.insert((44, "CZ".to_string()), phe45_cz_ppm);
    in_ref.shifts.insert((44, "HZ".to_string()), phe45_hz_ppm);

    let in_star = dir.join("in.str");
    write_star(&in_star, &in_ref, None).unwrap();
    let exact_in_ref = read_star_all_atoms(&in_star, None).unwrap();

    // Build real aromatic peak TSVs from `arom_truth`, removing Phe45's CZ/HZ peak so `Phe45 CZ/HZ`
    // is NOT reassigned by `assign_aromatic_rings` and must survive untouched from `in.str`!
    let truth_phe45_cz = arom_truth.shifts[&(44, "CZ".to_string())];
    let truth_phe45_hz = arom_truth.shifts[&(44, "HZ".to_string())];
    let (arom_pks, trp_relays) = simulate_aromatic_peaks(
        &arom_truth,
        &his_map,
        &AromaticSimConfig {
            seed,
            ..AromaticSimConfig::default()
        },
    );
    let is_phe45_cz_peak = |p: &Peak| -> bool {
        (p.h - truth_phe45_hz).abs() < 0.08 && p.c.is_some_and(|c| (c - truth_phe45_cz).abs() < 0.6)
    };
    let mut c13_hsqc = Vec::new();
    let mut hd_pks = Vec::new();
    let mut he_pks = Vec::new();
    let mut tocsy_pks = Vec::new();
    let mut his_pks = Vec::new();
    for p in arom_pks {
        if is_phe45_cz_peak(&p) {
            continue;
        }
        match p.exp.as_str() {
            "AROM_C13HSQC" => c13_hsqc.push(p),
            "HBCBCGCDHD" => hd_pks.push(p),
            "HBCBCGCDCEHE" => he_pks.push(p),
            "AROM_HCCH_TOCSY" => {
                // Also omit TOCSY cross-peaks whose indirect proton is Phe45 HZ
                if p.h2.is_some_and(|h2| (h2 - truth_phe45_hz).abs() < 0.05) {
                    continue;
                }
                tocsy_pks.push(p);
            }
            "HIS_LR_N15HMQC" => his_pks.push(p),
            _ => {}
        }
    }
    let arom_noe_pks: Vec<Peak> = trp_relays;

    let arom_hsqc_tsv = dir.join("arom_c13hsqc.tsv");
    let hd_tsv = dir.join("hbcbgcdhd.tsv");
    let he_tsv = dir.join("hbcbgcdce.tsv");
    let tocsy_tsv = dir.join("arom_tocsy.tsv");
    let noe_tsv = dir.join("arom_noesy.tsv");
    let his_tsv = dir.join("his_hmqc.tsv");
    write_peaks(&arom_hsqc_tsv, &c13_hsqc).unwrap();
    write_peaks(&hd_tsv, &hd_pks).unwrap();
    write_peaks(&he_tsv, &he_pks).unwrap();
    write_peaks(&tocsy_tsv, &tocsy_pks).unwrap();
    write_peaks(&noe_tsv, &arom_noe_pks).unwrap();
    write_peaks(&his_tsv, &his_pks).unwrap();

    // Build real J-coupling + intra-NOE TSVs for residues 1..=25 in canonical IUPAC order
    let pdb_res = parse_pdb_all_atoms_file(&pdb_in).expect("parse 1d3z");
    let mut j_rows = Vec::new();
    let mut intra_noe_rows = Vec::new();
    for (pi, pr) in pdb_res.iter().enumerate().take(25) {
        for spec in prochiral_pairs_for_residue(&pr.res_name) {
            let Some(obs) = synthesize_ideal_observables_from_pdb(&pdb_res, pi, spec) else {
                continue;
            };
            if let Some((ja, jb)) = obs.j_hh_vicinal_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_HAHB".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            if let Some((ja, jb)) = obs.j_nhb_hz {
                j_rows.push(JCouplingTsvRow {
                    seq_id: pr.seq_id,
                    res_name: pr.res_name.clone(),
                    site: spec.iupac_branch_1.to_string(),
                    j_kind: "3J_NHB".to_string(),
                    val_a_hz: ja,
                    val_b_hz: Some(jb),
                });
            }
            for np in &obs.noe_pairs {
                if spec.kind != ProchiralSiteKind::AsnCarboxamide
                    && spec.kind != ProchiralSiteKind::GlnCarboxamide
                {
                    intra_noe_rows.push(IntraNoeTsvRow {
                        seq_id: pr.seq_id,
                        res_name: pr.res_name.clone(),
                        site: spec.iupac_branch_1.to_string(),
                        anchor: np.anchor,
                        intensity_a: np.intensity_a,
                        intensity_b: np.intensity_b,
                    });
                }
            }
        }
    }
    let j_tsv = dir.join("j_couplings.tsv");
    let inoe_tsv = dir.join("intra_noes.tsv");
    write_j_couplings_tsv(&j_tsv, &j_rows).unwrap();
    write_intra_noes_tsv(&inoe_tsv, &intra_noe_rows).unwrap();

    let run_once = |star_arg: &PathBuf, out_arg: &PathBuf| {
        let out = process::Command::new(bin)
            .args([
                "sidechain",
                "--star",
                &star_arg.to_string_lossy(),
                "--aromatic",
                "--arom-c13hsqc",
                &arom_hsqc_tsv.to_string_lossy(),
                "--hbcbgcdhd",
                &hd_tsv.to_string_lossy(),
                "--hbcbgcdce",
                &he_tsv.to_string_lossy(),
                "--aromatic-tocsy",
                &tocsy_tsv.to_string_lossy(),
                "--aromatic-noesy",
                &noe_tsv.to_string_lossy(),
                "--his-hmqc",
                &his_tsv.to_string_lossy(),
                "--stereospec",
                "--j-couplings",
                &j_tsv.to_string_lossy(),
                "--intra-noes",
                &inoe_tsv.to_string_lossy(),
                "--pdb",
                &pdb_in.to_string_lossy(),
                "--out",
                &out_arg.to_string_lossy(),
            ])
            .output()
            .expect("spawn nmrassign sidechain real roundtrip");
        assert!(
            out.status.success(),
            "nmrassign sidechain failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    };

    let a1_star = dir.join("a1.str");
    let a2_star = dir.join("a2.str");
    run_once(&in_star, &a1_star);
    run_once(&a1_star, &a2_star);

    let a1_bytes = fs::read_to_string(&a1_star).expect("read a1.str");
    let a2_bytes = fs::read_to_string(&a2_star).expect("read a2.str");
    assert_eq!(
        a1_bytes, a2_bytes,
        "round-trip 2 (a1.str -> a2.str) must be 100% byte-for-byte identical!"
    );

    let a1_ref = read_star_all_atoms(&a1_star, None).expect("parse a1.str");
    // Verify Phe45 CZ/HZ and Lys63 H/N and every untouched backbone/aliphatic shift survived verbatim!
    assert!(
        (a1_ref.shifts[&(44, "CZ".to_string())] - phe45_cz_ppm).abs() < 1e-3,
        "Phe45 CZ was dropped or modified!"
    );
    assert!(
        (a1_ref.shifts[&(44, "HZ".to_string())] - phe45_hz_ppm).abs() < 1e-3,
        "Phe45 HZ was dropped or modified!"
    );
    assert!(
        (a1_ref.shifts[&(62, "H".to_string())] - lys63_h_ppm).abs() < 1e-3,
        "Lys63 H was dropped or modified!"
    );
    assert!(
        (a1_ref.shifts[&(62, "N".to_string())] - lys63_n_ppm).abs() < 1e-3,
        "Lys63 N was dropped or modified!"
    );

    const RING_ATOMS: &[&str] = &[
        "CD1", "CD2", "HD1", "HD2", "CE1", "CE2", "HE1", "HE2", "CZ", "HZ", "CE3", "HE3", "CZ2",
        "HZ2", "CZ3", "HZ3", "CH2", "HH2", "ND1", "NE1", "NE2",
    ];
    for (&(k, ref atom), &in_ppm) in &exact_in_ref.shifts {
        let is_arom_ring = matches!(
            exact_in_ref.sequence[k].as_str(),
            "PHE" | "TYR" | "TRP" | "HIS"
        ) && RING_ATOMS.contains(&atom.as_str());
        if !is_arom_ring {
            let out_ppm = *a1_ref
                .shifts
                .get(&(k, atom.clone()))
                .unwrap_or_else(|| panic!("untouched atom ({k}, {atom}) dropped in a1.str"));
            // Asn/Gln carboxamide or prochiral shifts could only swap within pair, but canonical Direct order stays identical
            if !matches!(atom.as_str(), "HD21" | "HD22" | "HE21" | "HE22") {
                assert!(
                    (out_ppm - in_ppm).abs() <= 1e-6,
                    "untouched non-ring shift ({k} {}, {atom}) mutated: in={in_ppm:.3}, out={out_ppm:.3}",
                    exact_in_ref.sequence[k]
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
