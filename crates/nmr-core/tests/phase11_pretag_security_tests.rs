//! Dedicated regression tests for Phase 11 pre-tag security & parser findings:
//! - P0-1: Multibyte UTF-8 safety across all 4 `nmr-core` fixed-column PDB parsers
//! - P0-2: XML/HTML escaping of residue/project/title names inside all SVG `<title>` tooltips
//! - P0-3: Bounded `Seq_ID` span cap & non-finite (`NaN`/`Inf`) chemical-shift rejection

use std::collections::BTreeMap;
use std::time::Instant;

use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig};
use nmr_core::methyl_trosy::{
    extract_methyl_structure_from_pdb_str, extract_multichain_methyl_structure_from_pdb_str,
    MethylLabelingScheme,
};
use nmr_core::nef::read_nef_str;
use nmr_core::nmrstar::{read_star_str, read_star_str_all_atoms, Reference};
use nmr_core::noesy::parse_pdb_str_first_model;
use nmr_core::report::{extract_dihedrals_from_pdb, generate_quality_report_html, ReportInput};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::AssignResult;
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_core::stereospec::parse_pdb_all_atoms_str;

#[test]
fn test_p0_1_all_core_pdb_parsers_survive_multibyte_utf8_at_every_column_boundary() {
    let base_lines = [
        "ATOM      1  N   ALA A   1      10.000  11.000  12.000  1.00 20.00           N  ",
        "ATOM      2  CA  ALA A   1      11.200  11.500  12.300  1.00 20.00           C  ",
        "ATOM      3  C   ALA A   1      12.400  10.800  12.100  1.00 20.00           C  ",
        "ATOM      4  CB  ALA A   1      11.300  12.900  11.800  1.00 20.00           C  ",
        "ATOM      5  N   VAL A   2      13.500  11.400  12.200  1.00 20.00           N  ",
        "ATOM      6  CA  VAL A   2      14.700  10.900  12.000  1.00 20.00           C  ",
        "ATOM      7  C   VAL A   2      15.800  11.800  12.400  1.00 20.00           C  ",
        "ATOM      8  CG1 VAL A   2      15.100  10.100  10.800  1.00 20.00           C  ",
        "ATOM      9  CG2 VAL A   2      14.900   9.900  13.200  1.00 20.00           C  ",
        "ATOM     10  N   GLY A   3      16.900  11.200  12.600  1.00 20.00           N  ",
        "ATOM     11  CA  GLY A   3      18.100  11.900  12.900  1.00 20.00           C  ",
        "ATOM     12  C   GLY A   3      19.200  11.100  12.500  1.00 20.00           C  ",
        "HETATM   99 FE   HEM A 101      25.000  25.000  25.000  1.00 15.00          FE  ",
    ];

    let scheme = MethylLabelingScheme::parse("ILVMAT").expect("ILVMAT");

    // Inject 2-byte ('é'), 3-byte ('Å'), and 4-byte ('🧬') UTF-8 scalars at every byte column 0..80
    for col in 0..80usize {
        for ch in ['é', 'Å', '🧬'] {
            let mut mutated = String::new();
            for base in &base_lines {
                mutated.push_str(base);
                mutated.push('\n');
                let mut bytes = base.as_bytes().to_vec();
                let ch_bytes = ch.to_string().into_bytes();
                if col + ch_bytes.len() <= bytes.len() {
                    bytes[col..col + ch_bytes.len()].copy_from_slice(&ch_bytes);
                }
                if let Ok(corrupted_line) = String::from_utf8(bytes) {
                    mutated.push_str(&corrupted_line);
                    mutated.push('\n');
                }
            }
            mutated.push_str("ENDMDL\n");

            // 1. noesy::parse_pdb_str_first_model
            let _ = parse_pdb_str_first_model(&mutated);
            // 2. stereospec::parse_pdb_all_atoms_str
            let _ = parse_pdb_all_atoms_str(&mutated);
            // 3. methyl_trosy::extract_methyl_structure_from_pdb_str (+ multichain)
            let _ = extract_methyl_structure_from_pdb_str(&mutated, &scheme);
            let _ = extract_multichain_methyl_structure_from_pdb_str(&mutated, &scheme);
            // 4. report::extract_dihedrals_from_pdb
            let _ = extract_dihedrals_from_pdb(&mutated);
        }
    }
}

#[test]
fn test_p0_2_report_svg_tooltips_escape_malicious_residue_and_project_names() {
    let payload_res = "<svg/onload=alert(1)>".to_string();
    let payload_res2 = "\"><script>alert(2)</script>".to_string();
    let seq = vec![
        "MET".to_string(),
        payload_res,
        payload_res2,
        "ALA".to_string(),
    ];
    let seq_ids = vec![1, 2, 3, 4];
    let mut ref_data = Reference::new(seq, seq_ids);
    for k in 0..4 {
        ref_data
            .shifts
            .insert((k, "H".to_string()), 8.20 + 0.12 * (k as f64));
        ref_data
            .shifts
            .insert((k, "N".to_string()), 120.5 + 1.2 * (k as f64));
        ref_data
            .shifts
            .insert((k, "CA".to_string()), 54.2 + 0.8 * (k as f64));
        ref_data
            .shifts
            .insert((k, "CB".to_string()), 19.1 + 1.1 * (k as f64));
        ref_data
            .shifts
            .insert((k, "C".to_string()), 176.3 + 0.6 * (k as f64));
    }

    let mut conf = BTreeMap::new();
    conf.insert(0, 0.92);
    conf.insert(1, 0.95);
    conf.insert(2, 0.90);
    conf.insert(3, 0.98);

    // Exercise BOTH branches of render_strip_walk_svg:
    // Branch A: synthetic fallback strips (systems_and_assignment = None)
    let prop = compute_idp_propensity(&ref_data, &PropensityConfig::default());
    let input_fallback = ReportInput::new(
        "<img src=x onerror=alert(9)>",
        ref_data.clone(),
        conf.clone(),
    )
    .with_propensity_report(prop.clone());
    let html_fallback = generate_quality_report_html(&input_fallback);

    assert!(
        !html_fallback.contains("<svg/onload=alert(1)>"),
        "unescaped <svg/onload> payload leaked into HTML/SVG report (fallback strip branch)"
    );
    assert!(
        !html_fallback.contains("<script>alert(2)</script>"),
        "unescaped <script> payload leaked into HTML/SVG report (fallback strip branch)"
    );
    assert!(
        !html_fallback.contains("<img src=x onerror=alert(9)>"),
        "unescaped project_name payload leaked into HTML/SVG report"
    );
    assert!(html_fallback.contains("&lt;svg/onload=alert(1)&gt;"));

    // Branch B: exact spin-system strips (systems_and_assignment = Some) with intra + inter CA/CB
    let (peaks, _) = simulate(&ref_data, &SimConfig::default());
    let sys_list = build_spin_systems(&peaks, &Tolerances::default(), 3, false)
        .expect("build 3 spin systems for XSS test");
    assert_eq!(sys_list.len(), 3);
    let assign_res = AssignResult {
        position: vec![1, 2, 3],
        confidence: vec![0.92, 0.95, 0.90, 0.98],
        score: 42.0,
        run_scores: vec![42.0],
    };

    let mut input_exact = ReportInput::new("exact_strips_xss", ref_data, conf);
    input_exact.systems_and_assignment = Some((sys_list, assign_res));
    input_exact = input_exact.with_propensity_report(prop);
    let html_exact = generate_quality_report_html(&input_exact);

    assert!(
        !html_exact.contains("<svg/onload=alert(1)>"),
        "unescaped <svg/onload> payload leaked into HTML/SVG report (exact strip branch)"
    );
    assert!(
        !html_exact.contains("<script>alert(2)</script>"),
        "unescaped <script> payload leaked into HTML/SVG report (exact strip branch)"
    );
    assert!(html_exact.contains("&lt;svg/onload=alert(1)&gt;"));
}

#[test]
fn test_p0_3_star_and_nef_reject_huge_seq_id_spans_and_non_finite_shifts_instantly() {
    // 1. Huge Seq_ID = i32::MAX must return Err in milliseconds without multi-GB allocation
    let star_huge_max = "\
data_oom_test
save_assigned_chem_shift_list_1
   _Assigned_chem_shift_list.Sf_category assigned_chemical_shifts
   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Val
      1 1 ALA CA 52.4
      2 2147483647 ALA CA 52.4
   stop_
save_
";
    let t0 = Instant::now();
    let err_max = read_star_str(star_huge_max, None).expect_err("Seq_ID=2147483647 must fail");
    assert!(
        t0.elapsed().as_millis() < 200,
        "Seq_ID span check took too long: {:?}",
        t0.elapsed()
    );
    assert!(err_max.contains("span"));

    // 2. Overflowing (i32::MIN, i32::MAX) span in i64 arithmetic must also fail cleanly without panic
    let star_min_max = "\
data_i32_overflow
save_assigned_chem_shift_list_1
   _Assigned_chem_shift_list.Sf_category assigned_chemical_shifts
   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Val
      1 -2147483648 ALA CA 52.4
      2 2147483647 ALA CA 52.4
   stop_
save_
";
    assert!(read_star_str_all_atoms(star_min_max, None).is_err());

    // 3. Non-finite shift values (NaN, inf, -inf) in NMR-STAR must be rejected at parse time
    for bad_val in ["NaN", "inf", "-inf", "Infinity"] {
        let star_nan = format!(
            "\
data_nan_test
save_assigned_chem_shift_list_1
   _Assigned_chem_shift_list.Sf_category assigned_chemical_shifts
   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Val
      1 1 ALA CA {bad_val}
   stop_
save_
"
        );
        assert!(
            read_star_str(&star_nan, None).is_err(),
            "expected NMR-STAR Val={bad_val} to be rejected"
        );
    }

    // 4. Non-finite shift values (NaN, inf) in CCPN NEF must fail cleanly
    for bad_val in ["NaN", "inf", "-inf"] {
        let nef_nan = format!(
            "\
data_nef_nan
save_nef_molecular_system
   _nef_molecular_system.sf_category nef_molecular_system
   _nef_molecular_system.sf_framecode nef_molecular_system
   loop_
      _nef_sequence.index
      _nef_sequence.chain_code
      _nef_sequence.sequence_code
      _nef_sequence.residue_name
      _nef_sequence.linking
      _nef_sequence.residue_variant
      1 A 1 ALA start .
      2 A 2 GLY end .
   stop_
save_
save_nef_chemical_shift_list_default
   _nef_chemical_shift_list.sf_category nef_chemical_shift_list
   _nef_chemical_shift_list.sf_framecode nef_chemical_shift_list_default
   loop_
      _nef_chemical_shift.chain_code
      _nef_chemical_shift.sequence_code
      _nef_chemical_shift.residue_name
      _nef_chemical_shift.atom_name
      _nef_chemical_shift.value
      A 1 ALA CA {bad_val}
   stop_
save_
"
        );
        assert!(
            read_nef_str(&nef_nan).is_err(),
            "expected NEF value={bad_val} to be rejected"
        );
    }
}
