//! Phase 12 (`v0.8.0`) CLI integration tests for `nmrassign torsions`, `nmrassign structure --rdc`
//! (`--torsion-db`, `--rdc-weight`, `--rdc-free-frac`, `--engine cpu|auto`), `nmrassign report --rdc`
//! (including XML/XSS escaping on hostile alignment-medium labels), `--version`, and strict
//! cross-command / cross-regime flag guards.
//!
//! Provenance: Clean-room verification suite citing Shen & Bax (*J. Biomol. NMR* 2013, 56, 227–241),
//! Berjanskii & Wishart (*J. Am. Chem. Soc.* 2005, 127, 14970–14971), Cornilescu et al.
//! (*J. Am. Chem. Soc.* 1998, 120, 6836–6837), and Losonczi et al. (*J. Magn. Reson.* 1999, 138, 334–342).

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use nmr_core::nef::{read_nef_file, write_nef_file, NefDocument, NefRdcRestraintList};
use nmr_core::nmrstar::read_star_all_atoms;
use nmr_core::rdc::read_rdc_tsv;

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn test_phase12_cli_version_is_0_8_0_and_help_lists_torsions_and_rdc() {
    let bin = nmrassign_bin();

    let ver_out = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("run nmrassign --version");
    assert!(ver_out.status.success());
    let ver_stdout = String::from_utf8_lossy(&ver_out.stdout);
    assert!(
        ver_stdout.starts_with("nmrassign 0.8.0 ("),
        "expected version 0.8.0, got: {ver_stdout}"
    );
    assert_eq!(nmr_core::VERSION, "0.8.0");

    let help_out = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("run nmrassign --help");
    assert!(help_out.status.success());
    let help_stdout = String::from_utf8_lossy(&help_out.stdout);
    assert!(help_stdout.contains("nmrassign torsions"));
    assert!(help_stdout.contains("--rdc <rdcs.tsv|rdcs.nef>"));
    assert!(help_stdout.contains("--torsion-db"));
}

#[test]
fn test_phase12_torsions_cli_tsv_nef_json_and_homolog_firewall_flags() {
    let bin = nmrassign_bin();
    let root = workspace_root();
    let star_1d3z = root.join("data/benchmark/bmr17769.str");

    let tmp = env::temp_dir().join(format!("nmr_p12_torsions_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // 1. Default `torsions` -> `.tsv`
    let out_tsv = tmp.join("ubq_torsions.tsv");
    let run_tsv = Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "-o",
            out_tsv.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign torsions .tsv");
    assert!(
        run_tsv.status.success(),
        "torsions .tsv failed: {}",
        String::from_utf8_lossy(&run_tsv.stderr)
    );
    let stdout_default = String::from_utf8_lossy(&run_tsv.stdout);
    assert!(
        stdout_default.contains("Good=")
            && stdout_default.contains("Dynamic=")
            && stdout_default.contains("Ambiguous=")
            && stdout_default.contains("exclusion_list_sha256="),
        "unexpected torsions summary stdout: {stdout_default}"
    );
    let tsv_text = fs::read_to_string(&out_tsv).unwrap();
    assert!(tsv_text.contains("# exclusion_list_sha256: "));
    assert!(
        tsv_text.contains("seq_id\tres_name\tclass\tphi_deg\tphi_std_deg\tpsi_deg\tpsi_std_deg")
    );

    // 2. Explicit firewall exclusions (`--exclude-pdb`, `--exclude-bmrb`, `--exclude-topology`) -> `.json`
    let out_json = tmp.join("ubq_torsions_fw.json");
    let run_json = Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "-o",
            out_json.to_str().unwrap(),
            "--exclude-pdb",
            "1D3Z,1UBQ,2NR2",
            "--exclude-bmrb",
            "bmr17769,6457",
            "--exclude-topology",
            "3.10.20",
        ])
        .output()
        .expect("run nmrassign torsions .json with firewall");
    assert!(
        run_json.status.success(),
        "torsions .json failed: {}",
        String::from_utf8_lossy(&run_json.stderr)
    );
    let json_txt = fs::read_to_string(&out_json).expect("read torsions JSON");
    assert!(
        json_txt.contains("\"residues\": [")
            && json_txt.contains("\"seq_id\": 76")
            && json_txt.contains("\"exclusion_list_sha256\":"),
        "unexpected torsions JSON structure:\n{}",
        &json_txt[..json_txt.len().min(600)]
    );
    let sha_fw_line = json_txt
        .lines()
        .find(|l| l.contains("\"exclusion_list_sha256\":"))
        .expect("exclusion_list_sha256 line in JSON");
    let sha_fw = sha_fw_line
        .split(':')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(',')
        .trim_matches('"');
    assert_eq!(sha_fw.len(), 64);
    let max_seq_line = json_txt
        .lines()
        .find(|l| l.contains("\"max_retained_seq_identity\":"))
        .expect("max_retained_seq_identity line in JSON");
    let max_seq_id: f64 = max_seq_line
        .split(':')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(',')
        .parse()
        .expect("parse max_retained_seq_identity");
    assert!(
        max_seq_id <= 0.25 + 1e-9,
        "HomologFirewall must enforce <= 25% retained sequence identity, got {max_seq_id}"
    );
    // Adding explicit topology `3.10.20` + extra PDB exclusions must change the exclusion manifest digest
    let default_sha_line = tsv_text
        .lines()
        .find(|l| l.starts_with("# exclusion_list_sha256: "))
        .unwrap();
    let sha_default = default_sha_line
        .trim_start_matches("# exclusion_list_sha256: ")
        .trim();
    assert_ne!(
        sha_fw, sha_default,
        "explicit --exclude-topology / --exclude-pdb flags must update exclusion_list_sha256"
    );

    // 3. `.nef` output + round-trip `.nef` input into `nmrassign torsions`
    let out_nef = tmp.join("ubq_torsions.nef");
    let run_nef = Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "-o",
            out_nef.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign torsions .nef");
    assert!(
        run_nef.status.success(),
        "torsions .nef failed: {}",
        String::from_utf8_lossy(&run_nef.stderr)
    );
    let parsed_nef = read_nef_file(&out_nef).expect("parse exported NEF");
    assert_eq!(parsed_nef.dihedral_restraint_lists.len(), 1);
    assert!(
        parsed_nef.dihedral_restraint_lists[0].restraints.len() >= 80,
        "expected >= 80 PHI/PSI/CHI1 NEF dihedral restraints for Ubiquitin, got {}",
        parsed_nef.dihedral_restraint_lists[0].restraints.len()
    );

    // Feed `.nef` back in as `--shifts` input
    let out_from_nef_tsv = tmp.join("ubq_from_nef.tsv");
    let run_from_nef = Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            out_nef.to_str().unwrap(),
            "-o",
            out_from_nef_tsv.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign torsions on .nef input");
    assert!(
        run_from_nef.status.success(),
        "torsions from .nef input failed: {}",
        String::from_utf8_lossy(&run_from_nef.stderr)
    );
    assert!(out_from_nef_tsv.exists());

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_phase12_structure_cli_with_rdc_cpu_and_auto_and_nef_rdc() {
    let bin = nmrassign_bin();
    let root = workspace_root();
    let star_1d3z = root.join("data/benchmark/bmr17769.str");
    let pdb_1d3z = root.join("data/pdb_refs/1d3z.pdb");
    let rdc_1d3z = root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv");

    let tmp = env::temp_dir().join(format!("nmr_p12_struct_rdc_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // 1. `structure --rdc ... --torsion-db --rdc-weight 0.75 --rdc-free-frac 0.20 --engine cpu`
    let out_cpu_pdb = tmp.join("ubq_rdc_cpu.pdb");
    let run_cpu = Command::new(&bin)
        .args([
            "structure",
            "--star",
            star_1d3z.to_str().unwrap(),
            "--pdb-ref",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            rdc_1d3z.to_str().unwrap(),
            "--torsion-db",
            "--rdc-weight",
            "0.75",
            "--rdc-free-frac",
            "0.20",
            "--engine",
            "cpu",
            "--conformers",
            "2",
            "--seed",
            "42",
            "--out-pdb",
            out_cpu_pdb.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign structure --rdc --engine cpu");
    assert!(
        run_cpu.status.success(),
        "structure --rdc --engine cpu failed: {}",
        String::from_utf8_lossy(&run_cpu.stderr)
    );
    let stdout_cpu = String::from_utf8_lossy(&run_cpu.stdout);
    assert!(
        stdout_cpu.contains("empirical 3/9-mer TorsionDb")
            && stdout_cpu.contains("RDC validation (2 media")
            && stdout_cpu.contains("Cornilescu Q_work=")
            && stdout_cpu.contains("Q_free="),
        "unexpected structure --rdc CPU stdout:\n{stdout_cpu}"
    );
    assert!(out_cpu_pdb.exists());
    let pdb_txt = fs::read_to_string(&out_cpu_pdb).unwrap();
    assert!(pdb_txt.contains("MODEL        1") && pdb_txt.contains("MODEL        2"));

    // 2. Round-trip RDCs via NEF (`nef_rdc_restraint_list`) and run `--engine auto`
    let shifts = read_star_all_atoms(&star_1d3z, None).unwrap();
    let media = read_rdc_tsv(&rdc_1d3z).unwrap();
    let mut nef_doc = NefDocument::from_reference_and_peaks("ubq_1d3z_rdc", &shifts, None, &[]);
    for m in &media {
        nef_doc
            .rdc_restraint_lists
            .push(NefRdcRestraintList::from_rdc_medium(&m.medium_id, "A", m));
    }
    let rdc_nef_path = tmp.join("ubq_rdc.nef");
    write_nef_file(&rdc_nef_path, &nef_doc).unwrap();

    let out_auto_pdb = tmp.join("ubq_rdc_auto.pdb");
    let run_auto = Command::new(&bin)
        .args([
            "structure",
            "--star",
            star_1d3z.to_str().unwrap(),
            "--pdb-ref",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            rdc_nef_path.to_str().unwrap(),
            "--torsion-db",
            "--rdc-free-frac",
            "0.20",
            "--engine",
            "auto",
            "--conformers",
            "2",
            "--seed",
            "42",
            "--out-pdb",
            out_auto_pdb.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign structure --rdc --engine auto");
    assert!(
        run_auto.status.success(),
        "structure --rdc --engine auto failed: {}",
        String::from_utf8_lossy(&run_auto.stderr)
    );
    let stdout_auto = String::from_utf8_lossy(&run_auto.stdout);
    assert!(
        stdout_auto.contains("RDC validation (2 media") && stdout_auto.contains("Q_free="),
        "unexpected structure --rdc auto stdout:\n{stdout_auto}"
    );
    assert!(out_auto_pdb.exists());

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_phase12_report_cli_rdc_svg_and_xml_escaping_against_hostile_strings() {
    let bin = nmrassign_bin();
    let root = workspace_root();
    let star_1d3z = root.join("data/benchmark/bmr17769.str");
    let pdb_1d3z = root.join("data/pdb_refs/1d3z.pdb");
    let rdc_1d3z = root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv");

    let tmp = env::temp_dir().join(format!("nmr_p12_report_rdc_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Rewrite `bicelle_neutral` to a hostile XSS medium identifier containing `<script>`, `&`, and `"`
    let raw_rdc = fs::read_to_string(&rdc_1d3z).unwrap();
    let hostile_rdc_txt = raw_rdc.replace(
        "bicelle_neutral\t",
        "bicelle_<script>alert(\"xss\")</script>&amp_test\t",
    );
    let hostile_rdc_path = tmp.join("hostile_rdc.tsv");
    fs::write(&hostile_rdc_path, hostile_rdc_txt).unwrap();

    let out_html = tmp.join("report_rdc.html");
    let run_rep = Command::new(&bin)
        .args([
            "report",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "--pdb",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            hostile_rdc_path.to_str().unwrap(),
            "--out",
            out_html.to_str().unwrap(),
            "--title",
            "Ubiquitin <1D3Z> & RDC Validation",
        ])
        .output()
        .expect("run nmrassign report --pdb --rdc");
    assert!(
        run_rep.status.success(),
        "report --pdb --rdc failed: {}",
        String::from_utf8_lossy(&run_rep.stderr)
    );
    let html = fs::read_to_string(&out_html).unwrap();

    // Must render RDC validation card + D_calc vs D_obs SVG scatter + Cornilescu Q
    assert!(
        html.contains("Residual Dipolar Coupling")
            && html.contains("Cornilescu")
            && html.contains("<svg"),
        "expected RDC validation section + SVG scatter in report.html"
    );
    // Hostile medium ID and title must be strictly XML-escaped (`&lt;script&gt;`), never raw `<script>alert`
    assert!(
        !html.contains("<script>alert("),
        "raw unescaped <script>alert(...) found in report.html!"
    );
    assert!(
        html.contains("bicelle_&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;&amp;amp_test"),
        "expected XML-escaped medium identifier in report.html"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_phase12_cli_strict_cross_command_and_flag_guards() {
    let bin = nmrassign_bin();
    let root = workspace_root();
    let star = root.join("data/fixtures/synthetic_ubiquitin.str");
    let rdc = root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv");

    let tmp = env::temp_dir().join(format!("nmr_p12_bad_flags_{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let dummy_out = tmp.join("unused_out");
    let dummy_html = tmp.join("unused.html");
    let bad_ext = tmp.join("torsions.xyz");

    let cases: &[(&[&str], &str)] = &[
        // 1. `--rdc` and `--torsion-db` rejected on `assign` (structure-free backbone rule)
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                dummy_out.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
            ],
            "post-assignment structure/report flag",
        ),
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                dummy_out.to_str().unwrap(),
                "--torsion-db",
            ],
            "post-assignment structure/report flag",
        ),
        // 2. `--rdc` rejected on `pipeline`
        (
            &[
                "pipeline",
                "--star",
                star.to_str().unwrap(),
                "--out-dir",
                dummy_out.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
            ],
            "not supported in `nmrassign pipeline`",
        ),
        // 3. `--rdc` on `report` without `--pdb` must fail hard
        (
            &[
                "report",
                "--shifts",
                star.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
                "--out",
                dummy_html.to_str().unwrap(),
            ],
            "--rdc requires --pdb",
        ),
        // 4. `--rdc-weight`, `--rdc-free-frac`, and `--engine gpu` on `structure` without `--rdc`
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--rdc-weight",
                "0.75",
            ],
            "--rdc-weight requires --rdc",
        ),
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--rdc-free-frac",
                "0.20",
            ],
            "--rdc-free-frac requires --rdc",
        ),
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--engine",
                "gpu",
            ],
            "requires --rdc",
        ),
        // 5. Out-of-bounds `--rdc-weight` and `--rdc-free-frac`
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
                "--rdc-weight",
                "0.0",
            ],
            "invalid --rdc-weight",
        ),
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
                "--rdc-free-frac",
                "0.80",
            ],
            "invalid --rdc-free-frac",
        ),
        // 6. `--dihedrals-only` combined with `--rdc`
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--rdc",
                rdc.to_str().unwrap(),
                "--dihedrals-only",
            ],
            "--dihedrals-only cannot be combined",
        ),
        // 7. Unsupported `-o` extension on `torsions`
        (
            &[
                "torsions",
                "--shifts",
                star.to_str().unwrap(),
                "-o",
                bad_ext.to_str().unwrap(),
            ],
            "unsupported torsions output extension",
        ),
        // 8. `--include-chi1` guards across `structure` (without `--torsion-db`), `assign`, and `pipeline`
        (
            &[
                "structure",
                "--star",
                star.to_str().unwrap(),
                "--include-chi1",
            ],
            "--include-chi1 requires --torsion-db",
        ),
        (
            &[
                "assign",
                "--star",
                star.to_str().unwrap(),
                "--out",
                dummy_out.to_str().unwrap(),
                "--include-chi1",
            ],
            "post-assignment structure/report flag",
        ),
        (
            &[
                "pipeline",
                "--star",
                star.to_str().unwrap(),
                "--out-dir",
                dummy_out.to_str().unwrap(),
                "--include-chi1",
            ],
            "not supported in `nmrassign pipeline`",
        ),
        // 9. `--ph` bounds on `sidechain` (`0.0..=14.0`) and rejection on `torsions` / `structure`
        (
            &[
                "sidechain",
                "--star",
                star.to_str().unwrap(),
                "--simulate",
                "--aromatic",
                "--ph",
                "15.0",
            ],
            "invalid --ph",
        ),
        (
            &[
                "sidechain",
                "--star",
                star.to_str().unwrap(),
                "--simulate",
                "--aromatic",
                "--ph",
                "-0.5",
            ],
            "invalid --ph",
        ),
        (
            &[
                "sidechain",
                "--star",
                star.to_str().unwrap(),
                "--simulate",
                "--aromatic",
                "--ph",
                "NaN",
            ],
            "invalid --ph",
        ),
        (
            &[
                "torsions",
                "--shifts",
                star.to_str().unwrap(),
                "-o",
                dummy_out.to_str().unwrap(),
                "--ph",
                "6.5",
            ],
            "unknown torsions flag",
        ),
        (
            &["structure", "--star", star.to_str().unwrap(), "--ph", "6.5"],
            "unknown structure flag",
        ),
    ];

    for (args, expected_substr) in cases {
        let out = Command::new(&bin)
            .args(*args)
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn {:?}: {e}", args));
        assert!(
            !out.status.success(),
            "expected hard CLI failure for {:?}, but command exited 0",
            args
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(expected_substr),
            "for {:?}, expected stderr containing {:?}, got:\n{}",
            args,
            expected_substr,
            stderr
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn test_p12_pretag_sidechain_ph_include_chi1_default_rdc_free_and_dropped_rdc_warnings() {
    let bin = nmrassign_bin();
    let root = workspace_root();
    let star_1d3z = root.join("data/benchmark/bmr17769.str");
    let pdb_1d3z = root.join("data/pdb_refs/1d3z.pdb");
    let rdc_1d3z = root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv");
    let star_syn = root.join("data/fixtures/synthetic_ubiquitin.str");

    let tmp = env::temp_dir().join(format!("nmr_p12_pretag_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // 1. `P1-2`: `nmrassign sidechain --aromatic --ph 6.5` succeeds and writes STAR
    let out_sc_star = tmp.join("sidechain_ph65.str");
    let run_sc = Command::new(&bin)
        .args([
            "sidechain",
            "--star",
            star_syn.to_str().unwrap(),
            "--simulate",
            "--aromatic",
            "--ph",
            "6.5",
            "--runs",
            "2",
            "--steps",
            "200",
            "--out",
            out_sc_star.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign sidechain --aromatic --ph 6.5");
    assert!(
        run_sc.status.success(),
        "sidechain --aromatic --ph 6.5 failed: {}",
        String::from_utf8_lossy(&run_sc.stderr)
    );
    assert!(out_sc_star.exists());

    // 2. `P1-5`: `nmrassign torsions` omits CHI1 by default and emits CHI1 only with `--include-chi1`
    let tsv_no_chi1 = tmp.join("tors_default.tsv");
    let nef_no_chi1 = tmp.join("tors_default.nef");
    let tsv_with_chi1 = tmp.join("tors_chi1.tsv");
    let nef_with_chi1 = tmp.join("tors_chi1.nef");

    assert!(Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "-o",
            tsv_no_chi1.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "-o",
            nef_no_chi1.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "--include-chi1",
            "-o",
            tsv_with_chi1.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(Command::new(&bin)
        .args([
            "torsions",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "--include-chi1",
            "-o",
            nef_with_chi1.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let txt_no_chi1 = fs::read_to_string(&tsv_no_chi1).unwrap();
    let txt_with_chi1 = fs::read_to_string(&tsv_with_chi1).unwrap();
    let any_chi1_default = txt_no_chi1
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("seq_id\t"))
        .any(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            cols.get(7).copied().unwrap_or(".") != "."
        });
    let any_chi1_opt_in = txt_with_chi1
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("seq_id\t"))
        .any(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            cols.get(7).copied().unwrap_or(".") != "."
        });
    assert!(
        !any_chi1_default,
        "default torsions TSV must omit all numeric chi1_deg values (all '.')"
    );
    assert!(
        any_chi1_opt_in,
        "--include-chi1 torsions TSV must emit numeric chi1_deg values on Good residues"
    );
    let doc_no_chi1 = read_nef_file(&nef_no_chi1).unwrap();
    let doc_with_chi1 = read_nef_file(&nef_with_chi1).unwrap();
    assert!(
        doc_no_chi1.dihedral_restraint_lists[0]
            .restraints
            .iter()
            .all(|r| r.name != "CHI1"),
        "default torsions NEF must emit zero CHI1 restraints"
    );
    assert!(
        doc_with_chi1.dihedral_restraint_lists[0]
            .restraints
            .iter()
            .any(|r| r.name == "CHI1"),
        "--include-chi1 torsions NEF must emit CHI1 restraints"
    );

    // 3. `P1-5` + `P1-6`: Default `structure --rdc` (omitting `--rdc-free-frac`) holds out 20% `Q_free`,
    //    `--rdc-free-frac 0.0` explicitly disables holdout, and 22 RDC rows (20 valid + 2 unresolved)
    //    print `warning: dropped 2 RDC restraint(s)` to stderr in BOTH `structure` and `report`!
    let raw_rdc = fs::read_to_string(&rdc_1d3z).unwrap();
    let mut rdc_22_lines: Vec<String> = Vec::new();
    let mut valid_taken = 0usize;
    for line in raw_rdc.lines() {
        let tr = line.trim();
        if tr.is_empty() || tr.starts_with('#') || tr.starts_with("medium_id") {
            rdc_22_lines.push(line.to_string());
            continue;
        }
        if tr.starts_with("bicelle_neutral\t") && valid_taken < 20 {
            rdc_22_lines.push(line.to_string());
            valid_taken += 1;
        }
    }
    assert_eq!(valid_taken, 20);
    // Append 2 unresolved rows: one out-of-range seq_id (999) and one unresolvable atom (NOTANATOM)
    rdc_22_lines.push(
        "bicelle_neutral\t999\tILE\tN\t999\tILE\tH\t1D_NH\t8.5000\t0.500\t21655.82\t0".to_string(),
    );
    rdc_22_lines.push(
        "bicelle_neutral\t3\tILE\tN\t3\tILE\tNOTANATOM\t1D_NH\t-6.2000\t0.500\t21655.82\t0"
            .to_string(),
    );
    let rdc_22_path = tmp.join("rdc_22_with_2_bad.tsv");
    fs::write(&rdc_22_path, rdc_22_lines.join("\n") + "\n").unwrap();

    // 3a. `structure --rdc` WITHOUT `--rdc-free-frac` -> finite `Q_free` + stderr `dropped 2 RDC`
    let run_struct_def = Command::new(&bin)
        .args([
            "structure",
            "--star",
            star_1d3z.to_str().unwrap(),
            "--pdb-ref",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            rdc_22_path.to_str().unwrap(),
            "--torsion-db",
            "--include-chi1",
            "--engine",
            "cpu",
            "--conformers",
            "1",
            "--seed",
            "42",
        ])
        .output()
        .expect("run structure --rdc default free_frac + 22 rows");
    assert!(
        run_struct_def.status.success(),
        "structure --rdc failed: {}",
        String::from_utf8_lossy(&run_struct_def.stderr)
    );
    let stdout_def = String::from_utf8_lossy(&run_struct_def.stdout);
    let stderr_def = String::from_utf8_lossy(&run_struct_def.stderr);
    assert!(
        stdout_def.contains("20 couplings") && !stdout_def.contains("Q_free=n/a"),
        "expected 20 couplings and finite default 20% Q_free, got stdout:\n{stdout_def}"
    );
    assert!(
        stderr_def.contains("warning: dropped 2 RDC restraint(s)"),
        "expected stderr dropped-RDC warning in structure, got stderr:\n{stderr_def}"
    );

    // 3b. `structure --rdc --rdc-free-frac 0.0` explicitly disables holdout (`Q_free=n/a (0 free RDCs)`)
    let run_struct_zero = Command::new(&bin)
        .args([
            "structure",
            "--star",
            star_1d3z.to_str().unwrap(),
            "--pdb-ref",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            rdc_22_path.to_str().unwrap(),
            "--rdc-free-frac",
            "0.0",
            "--engine",
            "cpu",
            "--conformers",
            "1",
            "--seed",
            "42",
        ])
        .output()
        .expect("run structure --rdc --rdc-free-frac 0.0");
    assert!(run_struct_zero.status.success());
    let stdout_zero = String::from_utf8_lossy(&run_struct_zero.stdout);
    assert!(
        stdout_zero.contains("Q_free=n/a (0 free RDCs)"),
        "expected explicit --rdc-free-frac 0.0 to disable free holdout, got:\n{stdout_zero}"
    );

    // 3c. `report --pdb --rdc` on the same 22-row file also emits `warning: dropped 2 RDC restraint(s)` on stderr
    let rep_html = tmp.join("rep_22.html");
    let run_rep_22 = Command::new(&bin)
        .args([
            "report",
            "--shifts",
            star_1d3z.to_str().unwrap(),
            "--pdb",
            pdb_1d3z.to_str().unwrap(),
            "--rdc",
            rdc_22_path.to_str().unwrap(),
            "--out",
            rep_html.to_str().unwrap(),
        ])
        .output()
        .expect("run report --pdb --rdc with 22 rows");
    assert!(run_rep_22.status.success());
    let stderr_rep = String::from_utf8_lossy(&run_rep_22.stderr);
    assert!(
        stderr_rep.contains("warning: dropped 2 RDC restraint(s)"),
        "expected stderr dropped-RDC warning in report, got stderr:\n{stderr_rep}"
    );

    let _ = fs::remove_dir_all(&tmp);
}
