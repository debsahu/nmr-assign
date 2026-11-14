//! Integration tests for `nmr-core::pulprog` (`T13.0` general `pulseprogram` & acquisition-physics
//! interpreter and Claude `Q1` empirical `{sign, fold}` 2D-HSQC strip-matching verification).
//!
//! Clean-room synthetic fixtures grounded in Cavanagh et al. (*Protein NMR Spectroscopy*, 2nd ed., 2007),
//! States, Haberkorn & Ruben (*JMR* 1982), Marion et al. (*JMR* 1989), Kay, Keifer & Saarinen (*JACS* 1992),
//! Moskau (*Concepts Magn. Reson.* 2002), and public Bruker JCAMP-DX v5.0 / Varian `procpar` specifications.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;
use std::time::Instant;

use nmr_core::classify::{
    classify_nmr_entries, classify_nmr_workspace, ExperimentDimensionality, PulseSequenceClass,
    QuadratureKind, VirtualEntry,
};
use nmr_core::fid::{read_bruker_pseudo3d_planes_2d, read_fid_2d, read_fid_3d, QuadMode};
use nmr_core::process::{
    process_fid_2d, process_fid_3d, register_3d_hn_to_hsqc_2d,
    register_3d_hn_to_hsqc_2d_with_audit, AxisRescueAuditRecord, ProcessingRecipe,
};
use nmr_core::pulprog::{
    apply_indirect_axis_transform_3d, decode_nuslist_str, evaluate_3d_axis_hypotheses_against_hsqc,
    interpret_bruker_experiment, interpret_experiment_dir, interpret_varian_procpar, parse_zgoptns,
    preprocess_bruker_pulseprogram, preprocess_bruker_pulseprogram_in_dir,
    validate_local_include_path, verify_3d_axis_hypotheses_against_hsqc, AcquisitionOrder,
    AxisTransformOp, BrukerExperimentInputs, DecodedQuadMode, FoldingConvention,
    NamedAxisHypothesis, NusIndexBase, Pseudo3DStorageOrder, PulprogError, PulprogSequenceFamily,
    PulprogVendor, QuadSource, MIN_AXIS_RESCUE_LN_BF,
};
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

// ============================================================================
// Helper Builders for Bruker JCAMP-DX & Varian `procpar` Fixtures
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn make_bruker_acqus(
    pulprog: &str,
    parmode: i32,
    aqseq: Option<i32>,
    zgoptns: &str,
    sfo1: f64,
    sfo2: f64,
    sfo3: f64,
    nuc2: &str,
    nuc3: &str,
    cnst20: f64,
    cnst21: f64,
    d0_s: f64,
    d8_s: f64,
    d20_s: f64,
    d28_s: f64,
    gpz1: f64,
    gpz2: f64,
    gpz3: f64,
    grpdly: f64,
    dspfvs: i32,
    decim: i32,
) -> String {
    let mut cnst = vec![0.0_f64; 32];
    cnst[20] = cnst20;
    cnst[21] = cnst21;
    let cnst_str = cnst
        .iter()
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(" ");

    let mut d = vec![0.0_f64; 32];
    d[0] = d0_s;
    d[1] = 1.0;
    d[4] = 0.00275; // 1/(4*90 Hz) 1H-15N INEPT
    d[8] = d8_s;
    d[20] = d20_s;
    d[21] = 0.014; // N-CA transfer
    d[24] = 0.016; // CO-N transfer
    d[28] = d28_s;
    let d_str = d
        .iter()
        .map(|v| format!("{v:.7}"))
        .collect::<Vec<_>>()
        .join(" ");

    let mut gpz = [0.0_f64; 16];
    gpz[1] = gpz1;
    gpz[2] = gpz2;
    gpz[3] = gpz3;
    let gpz_str = gpz
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(" ");

    let aqseq_line = aqseq
        .map(|a| format!("##$AQSEQ= {a}\n"))
        .unwrap_or_default();

    format!(
        "##TITLE= Parameter file, TopSpin 4.1\n\
         ##JCAMPDX= 5.0\n\
         ##$PULPROG= <{pulprog}>\n\
         ##$PARMODE= {parmode}\n\
         {aqseq_line}\
         ##$ZGOPTNS= <{zgoptns}>\n\
         ##$AQ_mod= 3\n\
         ##$BYTORDA= 0\n\
         ##$DTYPA= 0\n\
         ##$TD= 512\n\
         ##$SW_h= 8000.0\n\
         ##$SW= 13.3304\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= 600.130000\n\
         ##$O1= 2820.61\n\
         ##$NUC1= <1H>\n\
         ##$SFO2= {sfo2:.6}\n\
         ##$BF2= 150.902800\n\
         ##$NUC2= <{nuc2}>\n\
         ##$SFO3= {sfo3:.6}\n\
         ##$BF3= 60.817900\n\
         ##$NUC3= <{nuc3}>\n\
         ##$GRPDLY= {grpdly}\n\
         ##$DSPFVS= {dspfvs}\n\
         ##$DECIM= {decim}\n\
         ##$CNST= (0..31)\n\
         {cnst_str}\n\
         ##$D= (0..31)\n\
         {d_str}\n\
         ##$P= (0..31)\n\
         0 8.0 16.0 12.0 24.0 0 0 0 0 0 0 0 0 0 0 0 1000.0 0 0 0 0 38.0 76.0 0 0 0 0 0 0 0 0 0\n\
         ##$L= (0..15)\n\
         1 4 8 16 1 1 1 1 1 1 1 1 1 1 1 1\n\
         ##$GPZ= (0..15)\n\
         {gpz_str}\n\
         ##$SPNAM= (0..7)\n\
         <gauss> <G4.256> <Q5.1000> <Q3.1000> < Reburp.1000 > <none> <none> <none>\n\
         ##$CPDPRG= (0..4)\n\
         <none> <waltz16> <garp4> <waltz16> <mlev17>\n\
         ##END=\n"
    )
}

#[allow(clippy::too_many_arguments)]
fn make_bruker_acqu_ind(
    nuc1: &str,
    sfo1: f64,
    bf1: f64,
    sw_ppm: f64,
    td: usize,
    fnmode: i32,
    fntype: i32,
    nustd: Option<usize>,
) -> String {
    let sw_h = sw_ppm * sfo1;
    let nustd_line = nustd
        .map(|n| format!("##$NusTD= {n}\n"))
        .unwrap_or_default();
    format!(
        "##TITLE= Indirect parameter file\n\
         ##JCAMPDX= 5.0\n\
         ##$TD= {td}\n\
         {nustd_line}\
         ##$FnMODE= {fnmode}\n\
         ##$FnTYPE= {fntype}\n\
         ##$SW= {sw_ppm:.6}\n\
         ##$SW_h= {sw_h:.4}\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= {bf1:.6}\n\
         ##$NUC1= <{nuc1}>\n\
         ##END=\n"
    )
}

#[allow(clippy::too_many_arguments)]
fn make_varian_procpar(
    seqfil: &str,
    pslabel: &str,
    dn: &str,
    dn2: &str,
    dfrq: f64,
    dfrq2: f64,
    dof: f64,
    dof2: f64,
    ni: usize,
    sw1: f64,
    phase: &[i32],
    f1coef: &str,
    f1180: &str,
    ni2: Option<(usize, f64, &[i32], &str, &str)>,
    array_str: &str,
    extra_records: &str,
) -> String {
    let phase_vals = phase
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let mut s = format!(
        "seqfil 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{seqfil}\"\n\
         0\n\
         pslabel 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{pslabel}\"\n\
         0\n\
         tn 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"H1\"\n\
         0\n\
         dn 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{dn}\"\n\
         0\n\
         dn2 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{dn2}\"\n\
         0\n\
         np 7 1 131072 4 1 2 1 11 1 64\n\
         1 1024\n\
         0\n\
         sw 1 1 1000000 0 0 2 1 11 1 64\n\
         1 8000.0\n\
         0\n\
         sfrq 1 1 2000 0 0 2 1 11 1 64\n\
         1 600.132860\n\
         0\n\
         tof 1 1 100000 -100000 0 2 1 11 1 64\n\
         1 0.0\n\
         0\n\
         temp 1 1 200 -100 0 2 1 11 1 64\n\
         1 25.0\n\
         0\n\
         dfrq 1 1 2000 0 0 2 1 11 1 64\n\
         1 {dfrq:.6}\n\
         0\n\
         dof 1 1 100000 -100000 0 2 1 11 1 64\n\
         1 {dof:.2}\n\
         0\n\
         dfrq2 1 1 2000 0 0 2 1 11 1 64\n\
         1 {dfrq2:.6}\n\
         0\n\
         dof2 1 1 100000 -100000 0 2 1 11 1 64\n\
         1 {dof2:.2}\n\
         0\n\
         ni 7 1 65536 0 1 2 1 11 1 64\n\
         1 {ni}\n\
         0\n\
         sw1 1 1 500000 0 0 2 1 11 1 64\n\
         1 {sw1:.2}\n\
         0\n\
         phase 7 1 8 0 1 2 1 11 1 64\n\
         {} {phase_vals}\n\
         0\n\
         f1coef 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{f1coef}\"\n\
         0\n\
         f1180 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{f1180}\"\n\
         0\n\
         array 2 2 8192 0 0 2 1 0 1 64\n\
         1 \"{array_str}\"\n\
         0\n",
        phase.len()
    );

    if let Some((n2, sw2, ph2, f2coef, f2180)) = ni2 {
        let ph2_vals = ph2
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        s.push_str(&format!(
            "ni2 7 1 65536 0 1 2 1 11 1 64\n\
             1 {n2}\n\
             0\n\
             sw2 1 1 500000 0 0 2 1 11 1 64\n\
             1 {sw2:.2}\n\
             0\n\
             phase2 7 1 8 0 1 2 1 11 1 64\n\
             {} {ph2_vals}\n\
             0\n\
             f2coef 2 2 8192 0 0 2 1 0 1 64\n\
             1 \"{f2coef}\"\n\
             0\n\
             f2180 2 2 8192 0 0 2 1 0 1 64\n\
             1 \"{f2180}\"\n\
             0\n",
            ph2.len()
        ));
    }

    if !extra_records.is_empty() {
        s.push_str(extra_records);
    }
    s
}

// ============================================================================
// Test 1: Preprocessor Sandbox (`ZGOPTNS`, `#ifdef`/`#ifndef`/`#else`, Safe `#include`)
// ============================================================================

#[test]
fn test_preprocessor_zgoptns_ifdef_and_strict_include_sandbox() {
    let defs = parse_zgoptns("<-DTROSY -DLABEL_CN -DMODE=2 -UNOE>");
    assert_eq!(defs.get("TROSY").map(String::as_str), Some("1"));
    assert_eq!(defs.get("LABEL_CN").map(String::as_str), Some("1"));
    assert_eq!(defs.get("MODE").map(String::as_str), Some("2"));
    assert!(!defs.contains_key("NOE"));

    // Reject path traversal, absolute paths, drive prefixes, and backslashes!
    for bad in [
        "../outside.incl",
        "sub/../../secret.incl",
        "/etc/passwd",
        "/opt/topspin/exp/stan/nmr/lists/pp/Avance.incl",
        "\\windows\\system32\\cmd",
        "C:\\bruker\\Avance.incl",
        "~root/.ssh/id_rsa",
    ] {
        assert!(
            matches!(
                validate_local_include_path(bad),
                Err(PulprogError::UnsafeIncludePath(_))
            ),
            "expected UnsafeIncludePath for '{bad}'"
        );
    }
    assert_eq!(
        validate_local_include_path("custom_delays.incl").unwrap(),
        "custom_delays.incl"
    );

    // Verify in-memory `#ifdef`/`#ifndef`/`#if`/`#elif`/`#else`/`#endif` + local `#include`
    let mut local_inc = HashMap::new();
    local_inc.insert(
        "custom_delays.incl".to_string(),
        "\"d28=3.5m\"\n#define HAS_LOCAL_DELAYS 1\n".to_string(),
    );
    let src = "\
; custom pulseprogram with block comments and conditionals
/* multi-line comment containing fake directive:
#include \"../evil.incl\"
*/
#include <Avance.incl>
#include \"custom_delays.incl\"
#ifdef TROSY
  (p21 ph3):f3 ; TROSY S3E branch
#else
  (p22 ph1):f3 ; decoupled HSQC branch
#endif
#if defined(LABEL_CN) && MODE == 2 && HAS_LOCAL_DELAYS
  d28 ; HNCACB CA-CB transfer active
#elif defined(LABEL_CN)
  d21 ; HNCA only
#endif
#ifndef DEUTERATED
  ; protonated branch
  (p3 ph2):f2
#endif
";
    let pp = preprocess_bruker_pulseprogram(src, "-DTROSY -DLABEL_CN -DMODE=2", &local_inc)
        .expect("preprocess ok");
    let joined = pp.lines.join("\n");
    assert!(joined.contains("d28=3.5m"));
    assert!(joined.contains("(p21 ph3):f3"));
    assert!(!joined.contains("(p22 ph1):f3"));
    assert!(joined.contains("d28"));
    assert!(!joined.contains("d21"));
    assert!(joined.contains("(p3 ph2):f2"));
    assert_eq!(pp.resolved_includes, vec!["custom_delays.incl".to_string()]);
    assert_eq!(
        pp.unresolved_standard_includes,
        vec!["Avance.incl".to_string()]
    );

    // Reject `#include "../escape.incl"` inside an active branch!
    let evil_src = "#ifdef TROSY\n#include \"../escape.incl\"\n#endif\n";
    let err = preprocess_bruker_pulseprogram(evil_src, "-DTROSY", &local_inc).unwrap_err();
    assert!(matches!(err, PulprogError::UnsafeIncludePath(_)));

    // Reject `#include "../escape.incl"` EVEN inside an inactive `#ifdef` branch!
    let evil_inactive = "#ifdef NEVER_DEFINED\n#include \"../escape.incl\"\n#endif\n";
    let err2 = preprocess_bruker_pulseprogram(evil_inactive, "", &local_inc).unwrap_err();
    assert!(matches!(err2, PulprogError::UnsafeIncludePath(_)));

    // Reject recursive local `#include` cycles!
    let mut cyc_inc = HashMap::new();
    cyc_inc.insert("a.incl".to_string(), "#include \"b.incl\"\n".to_string());
    cyc_inc.insert("b.incl".to_string(), "#include \"a.incl\"\n".to_string());
    let cyc_err =
        preprocess_bruker_pulseprogram("#include \"a.incl\"\n", "", &cyc_inc).unwrap_err();
    assert!(matches!(cyc_err, PulprogError::IncludeCycle(_)));

    // Reject non-cyclic diamond `#include` DAG bombs (`10 x 10 = 100 > MAX_TOTAL_INCLUDES = 64`) in `< 5 ms`!
    let mut dag_inc = HashMap::new();
    dag_inc.insert(
        "lvl1.incl".to_string(),
        "#include \"lvl2.incl\"\n".repeat(10),
    );
    dag_inc.insert("lvl2.incl".to_string(), "\"d0=3u\"\n".to_string());
    let dag_root = "#include \"lvl1.incl\"\n".repeat(10);
    let t_dag = Instant::now();
    let dag_err = preprocess_bruker_pulseprogram(&dag_root, "", &dag_inc).unwrap_err();
    assert!(
        matches!(dag_err, PulprogError::ResourceLimitExceeded(_)),
        "expected ResourceLimitExceeded on diamond #include DAG, got {dag_err:?}"
    );
    assert!(t_dag.elapsed().as_secs_f64() * 1000.0 < 10.0);

    // Reject corrupt `procpar` with huge `n_vals` (`999_999_999_999`) + truncated EOF in `< 5 ms` without OOM or spin!
    let corrupt_procpar = "seqfil 2 2 8192 0 0 2 1 0 1 64\n999999999999 \"ghsqc\"\n";
    let t_pp = Instant::now();
    let pp_err = interpret_varian_procpar(corrupt_procpar, None).unwrap_err();
    assert!(
        matches!(pp_err, PulprogError::ResourceLimitExceeded(_)),
        "expected ResourceLimitExceeded on huge procpar n_vals, got {pp_err:?}"
    );
    assert!(t_pp.elapsed().as_secs_f64() * 1000.0 < 5.0);

    // Fail closed when `exp_dir` does not exist!
    let missing_dir = std::env::temp_dir().join("nmr_nonexistent_pulprog_dir_999888777");
    let _ = fs::remove_dir_all(&missing_dir);
    assert!(matches!(
        preprocess_bruker_pulseprogram_in_dir("1 ze\n", "", &missing_dir),
        Err(PulprogError::Io(_))
    ));

    // Reject symlink escaping `exp_dir` on disk (both `#include` AND top-level `pulseprogram`),
    // AND verify ISO-8859-1 / Latin-1 vendor header comments (`Wüthrich` `0xfc`, `90°` `0xb0`) parse cleanly!
    #[cfg(unix)]
    {
        let tmp = std::env::temp_dir().join(format!(
            "nmr_pulprog_sandbox_{}_{}",
            std::process::id(),
            101
        ));
        let _ = fs::remove_dir_all(&tmp);
        let exp_dir = tmp.join("exp1");
        let outside_dir = tmp.join("outside");
        fs::create_dir_all(&exp_dir).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.incl");
        fs::write(&secret, "\"d0=1u\"\n").unwrap();
        std::os::unix::fs::symlink(&secret, exp_dir.join("escape_link.incl")).unwrap();

        let res =
            preprocess_bruker_pulseprogram_in_dir("#include \"escape_link.incl\"\n", "", &exp_dir);
        assert!(matches!(res, Err(PulprogError::UnsafeIncludePath(_))));

        // Top-level `pulseprogram` symlink escaping `exp_dir` MUST also fail closed!
        let exp_top = tmp.join("exp_top_symlink");
        fs::create_dir_all(&exp_top).unwrap();
        fs::write(
            exp_top.join("acqus"),
            make_bruker_acqus(
                "hsqcetf3gpsi",
                1,
                Some(0),
                "",
                600.13,
                150.91,
                60.81,
                "13C",
                "15N",
                174.0,
                54.0,
                3e-6,
                0.0,
                0.0,
                0.0,
                80.0,
                8.1,
                0.0,
                0.0,
                0,
                1,
            ),
        )
        .unwrap();
        fs::write(
            exp_top.join("acqu2s"),
            make_bruker_acqu_ind("15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None),
        )
        .unwrap();
        std::os::unix::fs::symlink(&secret, exp_top.join("pulseprogram")).unwrap();
        assert!(matches!(
            interpret_experiment_dir(&exp_top),
            Err(PulprogError::UnsafeIncludePath(_))
        ));

        // Raw ISO-8859-1 / Latin-1 bytes (`0xfc` = `ü`, `0xb0` = `°`) in `acqus` and `pulseprogram`!
        let exp_latin1 = tmp.join("exp_latin1");
        fs::create_dir_all(&exp_latin1).unwrap();
        let mut acqus_latin1 = b"$$ Author: Kurt W\xfcthrich, Z\xfcrich 25\xb0C\n".to_vec();
        acqus_latin1.extend_from_slice(
            make_bruker_acqus(
                "hsqcetf3gpsi",
                1,
                Some(0),
                "",
                600.1328,
                150.9108,
                60.8179,
                "13C",
                "15N",
                174.0,
                54.0,
                3e-6,
                0.0,
                0.0,
                0.0,
                80.0,
                8.1,
                0.0,
                0.0,
                0,
                1,
            )
            .as_bytes(),
        );
        fs::write(exp_latin1.join("acqus"), &acqus_latin1).unwrap();
        fs::write(
            exp_latin1.join("acqu2s"),
            make_bruker_acqu_ind("15N", 60.8179, 60.8106, 32.0, 16, 6, 0, None),
        )
        .unwrap();
        let pp_latin1 = b"; Marion & W\xfcthrich 90\xb0 phase-sensitive HSQC\ndefine list<gradient> EA=<EA>\n1 ze\n2 d1\n  (p1 ph1)\n  (p21 ph3):f3\n  d0\n  p16:gp1*EA\n  (p22 ph4):f3\n  d0\n  p16:gp2\n  go=2 ph31 cpd3:f3\n  d1 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))\nexit\nph1=0\nph3=0\nph4=0\nph31=0 2\n";
        fs::write(exp_latin1.join("pulseprogram"), pp_latin1).unwrap();
        let pi_lat = interpret_experiment_dir(&exp_latin1).expect("Latin-1 acqus + pulseprogram");
        assert_eq!(
            pi_lat.transfer_graph.deduced_family,
            PulprogSequenceFamily::N15Hsqc
        );

        let _ = fs::remove_dir_all(&tmp);
    }
}

// ============================================================================
// Test 2: 35 Diverse Bruker & Varian Fixtures (`100%` Accuracy & `< 10 ms` Latency)
// ============================================================================

struct BrukerFixtureCase {
    name: &'static str,
    acqus: String,
    acqu2s: Option<String>,
    acqu3s: Option<String>,
    pulseprogram: String,
    nuslist: Option<String>,
    vdlist: Option<String>,
    vclist: Option<String>,
    fqlist: Option<String>,
    difflist: Option<String>,
    local_includes: HashMap<String, String>,
    expected_family: PulprogSequenceFamily,
    expected_aqseq: AcquisitionOrder,
    expected_pseudo: Pseudo3DStorageOrder,
    check: Box<dyn Fn(&nmr_core::pulprog::PulprogInterpretation)>,
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_35_diverse_bruker_and_varian_fixtures_100pct_accuracy_and_under_10ms() {
    let mut bruker_cases: Vec<BrukerFixtureCase> = Vec::new();

    // 1. Standard 2D 15N SE Echo-Antiecho `hsqcetf3gpsi`
    bruker_cases.push(BrukerFixtureCase {
        name: "01_bruker_hsqcetf3gpsi_2d_se_ea",
        acqus: make_bruker_acqus(
            "hsqcetf3gpsi",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcetf3gpsi
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  d4 pl3:f3
  (center (p2 ph1) (p22 ph6):f3 )
  d4
  (p21 ph3):f3
  d0
  p16:gp1*EA
  d16
  (p22 ph4):f3
  d0
  (p1 ph2)
  p16:gp2
  d16
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0 2
ph2=0
ph3=0 0 2 2
ph4=0
ph6=0
ph31=0 2 2 0
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(pi.indirect_dims.len(), 1);
            let d = &pi.indirect_dims[0];
            assert_eq!(d.quad_mode, DecodedQuadMode::EchoAntiecho);
            assert!(!d.reverse_frequency);
            let rk = d.rance_kay.expect("rance_kay");
            assert!(!rk.ea_polarity_inverted);
            assert!((rk.grad_ratio.unwrap() - 80.0 / 8.1).abs() < 0.05);
            assert!(pi.transfer_graph.sensitivity_enhanced);
            assert!(!pi.transfer_graph.trosy_selected);
        }),
    });

    // 2. Standard 2D 15N States-TPPI `hsqcf3gpph19`
    bruker_cases.push(BrukerFixtureCase {
        name: "02_bruker_hsqcf3gpph19_states_tppi",
        acqus: make_bruker_acqus(
            "hsqcf3gpph19",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            50.0,
            30.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 5, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcf3gpph19
1 ze
2 d1 do:f3
  (p1 ph1)
  d4
  (p21 ph5):f3
  d0
  (p2 ph2)
  d0
  (p21 ph6):f3
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1PH(calph(ph5, +90), caldel(d0, +in0))
exit
ph1=0
ph2=0
ph5=0 2
ph6=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert_eq!(d.quad_mode, DecodedQuadMode::StatesTppi);
            assert!(!d.reverse_frequency);
            assert_eq!(d.folding, FoldingConvention::MirrorInvertedWrap);
            assert!(!d.half_dwell_sampling);
            assert_eq!((d.predicted_phc0_deg, d.predicted_phc1_deg), (0.0, 0.0));
        }),
    });

    // 3. 2D 15N TROSY `trosyetf3gpsi`
    bruker_cases.push(BrukerFixtureCase {
        name: "03_bruker_trosyetf3gpsi_2d",
        acqus: make_bruker_acqus(
            "trosyetf3gpsi",
            1,
            Some(0),
            "",
            800.1300,
            201.1900,
            81.0760,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 81.0760, 81.0660, 34.0, 160, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;trosyetf3gpsi - S3E TROSY without f3 decoupling during acquisition
define list<gradient> EA=<EA>
1 ze
2 d1
  (p1 ph1)
  d4
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=2 ph31
  d1 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=1 3
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Trosy,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert!(pi.transfer_graph.trosy_selected);
            assert_eq!(
                pi.transfer_graph.deduced_family.to_pulse_sequence_class(),
                PulseSequenceClass::N15Hsqc
            );
        }),
    });

    // 4. 2D 13C aliphatic SE Echo-Antiecho `hsqcetgpsisp2.2`
    bruker_cases.push(BrukerFixtureCase {
        name: "04_bruker_c13_hsqcetgpsisp2_2",
        acqus: make_bruker_acqus(
            "hsqcetgpsisp2.2",
            1,
            Some(0),
            "",
            600.1328,
            150.9090,
            60.8179,
            "13C",
            "15N",
            174.0,
            42.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            20.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "13C", 150.9090, 150.9028, 70.0, 256, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcetgpsisp2.2
define list<gradient> EA=<EA>
1 ze
2 d1 do:f2
  (p1 ph1)
  (p3 ph3):f2
  d0
  p16:gp1*EA
  (p4 ph4):f2
  d0
  p16:gp2
  go=2 ph31 cpd2:f2
  d1 do:f2 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::C13Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let rk = pi.indirect_dims[0].rance_kay.unwrap();
            assert!(!rk.ea_polarity_inverted);
            assert!((rk.grad_ratio.unwrap() - 80.0 / 20.1).abs() < 0.05);
        }),
    });

    // 5. 2D 13C Methyl HMQC `hmqcgpph`
    bruker_cases.push(BrukerFixtureCase {
        name: "05_bruker_c13_hmqcgpph",
        acqus: make_bruker_acqus(
            "hmqcgpph",
            1,
            Some(0),
            "",
            600.1328,
            150.9058,
            60.8179,
            "13C",
            "15N",
            174.0,
            20.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            30.0,
            0.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "13C", 150.9058, 150.9028, 22.0, 128, 5, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hmqcgpph - methyl TROSY HMQC
1 ze
2 d1 do:f2
  (p1 ph1)
  d2
  (p3 ph3):f2
  d0
  (p2 ph2)
  d0
  (p3 ph4):f2
  d2
  go=2 ph31 cpd2:f2
  d1 do:f2 mc #0 to 2 F1PH(calph(ph3, +90), caldel(d0, +in0))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::C13Hmqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(pi.indirect_dims[0].quad_mode, DecodedQuadMode::StatesTppi);
        }),
    });

    // 6. 3D `hncagp3d`
    bruker_cases.push(BrukerFixtureCase {
        name: "06_bruker_hncagp3d",
        acqus: make_bruker_acqus(
            "hncagp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9109,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.014,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9109, 150.9028, 30.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncagp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  d4
  (p21 ph1):f3
  d21
  (p3 ph3):f2
  d0
  (p14:sp3 ph1):f2 ; selective CO 180 during CA t1
  d0
  (p3 ph4):f2
  d10
  p16:gp1*EA
  (p22 ph2):f3
  d30
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hnca,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(pi.indirect_dims.len(), 2);
            assert_eq!(pi.indirect_dims[0].quad_mode, DecodedQuadMode::EchoAntiecho);
            assert_eq!(pi.indirect_dims[1].quad_mode, DecodedQuadMode::StatesTppi);
            assert!(pi.indirect_dims[0].reverse_frequency); // d10 before d30
        }),
    });

    // 7. 3D `hncocagpwg3d` (SCT `d30` before `d10` + `cnst20`/`cnst21` carrier jumps)
    bruker_cases.push(BrukerFixtureCase {
        name: "07_bruker_hncocagpwg3d_sct_and_cnst_jump",
        acqus: make_bruker_acqus(
            "hncocagpwg3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9109,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.014,
            0.0,
            50.0,
            30.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 5, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9109, 150.9028, 30.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncocagpwg3d
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph1):f3
  d24 fq=cnst20(bf ppm):f2 ; jump to CO 174 ppm
  (p13:sp1 ph1):f2
  fq=cnst21(bf ppm):f2 ; jump to CA 54 ppm
  (p3 ph3):f2
  d0
  (p3 ph4):f2
  fq=cnst20(bf ppm):f2
  (p13:sp1 ph1):f2
  (p21 ph5):f3
  d30
  (p22 ph1):f3
  d10
  (p21 ph1):f3
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2PH(calph(ph5, +90), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph3=0 2
ph4=0
ph5=0 2
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hncoca,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            // ph5 precedes d30 ... p22 ... d10 -> has_dec_before_evo = true -> reverse_frequency = true!
            assert!(pi.indirect_dims[0].reverse_frequency);
            assert!(!pi.transfer_graph.carrier_jumps_ppm.is_empty());
        }),
    });

    // 8. 3D `cbcaconhgp3d`
    bruker_cases.push(BrukerFixtureCase {
        name: "08_bruker_cbcaconhgp3d",
        acqus: make_bruker_acqus(
            "cbcaconhgp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0035,
            0.0035,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 70.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;cbcaconhgp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1) ; 1H-13C aliphatic INEPT start
  (p3 ph3):f2
  d0
  (p14:sp3 ph1):f2
  d20
  (p3 ph4):f2
  fq=cnst20(bf ppm):f2 ; relay via CO to 15N
  (p13:sp1 ph1):f2
  d24
  (p21 ph1):f3
  d30
  p16:gp1*EA
  (p22 ph2):f3
  d10
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0) & caldel(d20, -in20))
    F2EA(calgrad(EA), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Cbcaconh,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert!(pi.indirect_dims[1].constant_time);
            assert_eq!(
                pi.indirect_dims[1].folding,
                FoldingConvention::ConstantTimeMirror
            );
            assert!(!pi.indirect_dims[0].reverse_frequency); // d30 before d10 on EA!
        }),
    });

    // 9. 3D Constant-Time `hncacbgp3d`
    bruker_cases.push(BrukerFixtureCase {
        name: "09_bruker_hncacbgp3d_constant_time",
        acqus: make_bruker_acqus(
            "hncacbgp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0035,
            0.0036,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 72.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncacbgp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph1):f3
  d21
  (p3 ph3):f2
  d28 ; 1/(2*J_CC) CA-CB transfer
  d0
  (p4 ph1):f2
  d20
  (p3 ph4):f2
  d30
  p16:gp1*EA
  (p22 ph2):f3
  d10
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0) & caldel(d20, -in20))
    F2EA(calgrad(EA), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hncacb,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert!(pi.indirect_dims[1].constant_time);
            assert_eq!(
                pi.indirect_dims[1].folding,
                FoldingConvention::ConstantTimeMirror
            );
        }),
    });

    // 10. 3D `hncogp3d`
    bruker_cases.push(BrukerFixtureCase {
        name: "10_bruker_hncogp3d",
        acqus: make_bruker_acqus(
            "hncogp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9290,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9290, 150.9028, 14.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncogp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph1):f3
  d24
  (p3 ph3):f2
  d0
  (p14:sp3 ph1):f2
  d0
  (p3 ph4):f2
  d30
  p16:gp1*EA
  (p22 ph2):f3
  d10
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hnco,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(pi.indirect_dims.len(), 2);
        }),
    });

    // 11. 3D `hncacogp3d`
    bruker_cases.push(BrukerFixtureCase {
        name: "11_bruker_hncacogp3d",
        acqus: make_bruker_acqus(
            "hncacogp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9290,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9290, 150.9028, 14.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncacogp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph1):f3
  d21 fq=cnst21(bf ppm):f2 ; relay via CA 54 ppm
  (p3 ph1):f2
  fq=cnst20(bf ppm):f2 ; evolve CO 174 ppm
  (p13:sp1 ph3):f2
  d0
  (p13:sp1 ph4):f2
  p16:gp1*EA
  d10
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10))
exit
ph1=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hncaco,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|_| {}),
    });

    // 12. 3D `ccconhgp3d` (isotropic TOCSY mixing)
    bruker_cases.push(BrukerFixtureCase {
        name: "12_bruker_ccconhgp3d_tocsy",
        acqus: make_bruker_acqus(
            "ccconhgp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 72.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;ccconhgp3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p3 ph3):f2
  d0
  ; isotropic 13C DIPSI-2 TOCSY spinlock
  (d9 cpd4:f2)
  fq=cnst20(bf ppm):f2
  d24
  (p21 ph1):f3
  d10
  p16:gp1*EA
  (p22 ph2):f3
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10))
exit
ph1=0
ph2=0
ph3=0 2
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Ccconh,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|_| {}),
    });

    // 13. 3D `noesyhsqcf3gpsi3d` (15N-edited NOESY-HSQC, `d8 = 120 ms`)
    bruker_cases.push(BrukerFixtureCase {
        name: "13_bruker_noesyhsqcf3gpsi3d",
        acqus: make_bruker_acqus(
            "noesyhsqcf3gpsi3d",
            2,
            Some(0),
            "",
            600.1328,
            600.1328,
            60.8179,
            "1H",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.120,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "1H", 600.1328, 600.1300, 12.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;noesyhsqcf3gpsi3d
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph3)
  d0
  (p1 ph4)
  d8 ; NOESY dipolar mixing time 120 ms
  (p21 ph1):f3
  d10
  p16:gp1*EA
  (p22 ph2):f3
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Noesy,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|_| {}),
    });

    // 14. Transposed `AQSEQ = 312` (`##$AQSEQ= 1`) 3D HNCO with transposed 2-col `nuslist`
    bruker_cases.push(BrukerFixtureCase {
        name: "14_bruker_aqseq312_header_and_transposed_nuslist",
        acqus: make_bruker_acqus(
            "hncogp3d",
            2,
            Some(1),
            "",
            600.1328,
            150.9290,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N",
            60.8179,
            60.8106,
            32.0,
            16,
            6,
            2,
            Some(64),
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C",
            150.9290,
            150.9028,
            14.0,
            1,
            5,
            2,
            Some(128),
        )),
        pulseprogram: "\
;hncogp3d aqseq 312
aqseq 312
1 ze
2 d1
  go=2 ph31
exit
ph31=0 2
"
        .to_string(),
        // Under AQSEQ=312: file col0 is F1 (n_cplx = 64: max 63), col1 is F2 (n_cplx = 32: max 31)!
        nuslist: Some("0 0\n63 31\n45 12\n10 29\n".to_string()),
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hnco,
        expected_aqseq: AcquisitionOrder::Aqseq312,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let nus = pi.nus_schedule.as_ref().expect("nus schedule");
            assert!(nus.aqseq_transposed);
            assert_eq!(nus.index_base, NusIndexBase::ZeroBased);
            // Normalized to `[k1_f2, k2_f1]`!
            assert_eq!(nus.points[1], vec![31, 63]);
            assert_eq!(nus.points[2], vec![12, 45]);
        }),
    });

    // 15. `AQSEQ = 312` deduced symbolically from explicit `id0` inner to `id10` loop when `##$AQSEQ` is omitted!
    bruker_cases.push(BrukerFixtureCase {
        name: "15_bruker_aqseq312_symbolic_loop_deduction",
        acqus: make_bruker_acqus(
            "custom_312_hnca",
            2,
            None,
            "",
            600.1328,
            150.9109,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            50.0,
            30.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 5, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9109, 150.9028, 30.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
; legacy XWIN-NMR 3D HNCA with inner F1 (d0/id0) loop and outer F2 (d10/id10) loop
1 ze
2 d1 do:f3
3 d1
4 (p1 ph1)
  (p21 ph5):f3
  d10
  (p3 ph3):f2
  d0
  (p3 ph4):f2
  (p21 ph1):f3
  go=4 ph31 cpd3:f3
  d1 wr #0 if #0 zd ip3
  lo to 3 times 2
  d1 id0 ip3*2 ip31*2
  lo to 3 times l1
  d1 rd0 ip5
  lo to 2 times 2
  d1 id10 ip5*2 ip31*2
  lo to 2 times l2
exit
ph1=0
ph3=0
ph4=0
ph5=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hnca,
        expected_aqseq: AcquisitionOrder::Aqseq312,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(pi.aqseq, AcquisitionOrder::Aqseq312);
        }),
    });

    // 16. Manual `ip3` + `ip31*2` `States-TPPI` inside `lo to 2 times 2` under `FnMODE = 1 (QF)`!
    bruker_cases.push(BrukerFixtureCase {
        name: "16_bruker_manual_ip_states_tppi_under_fnmode_qf",
        acqus: make_bruker_acqus(
            "custom_manual_hsqc_qf",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 1, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
; custom laboratory HSQC with manual ip3/ip31*2 States-TPPI loop while FnMODE=QF
1 ze
2 d1 do:f3
3 (p1 ph1)
  d4
  (p21 ph3):f3
  d0
  (p2 ph2)
  d0
  (p21 ph4):f3
  go=3 ph31 cpd3:f3
  d1 do:f3 wr #0 if #0 zd ip3
  lo to 3 times 2
  d1 id0 ip3*2 ip31*2
  lo to 2 times l0
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert_eq!(d.quad_mode, DecodedQuadMode::StatesTppi);
            assert_eq!(
                d.quad_source,
                QuadSource::SynthesizedManualLoop { header_fnmode: 1 }
            );
            assert!(!d.reverse_frequency);
            assert_eq!(d.complex_points, 64);
            assert_eq!(d.folding, FoldingConvention::MirrorInvertedWrap);
        }),
    });

    // 17. Manual `dp3` `States` inside `lo to 3 times 2` under `FnMODE = 0 (Undefined)` (`reverse_frequency = true`)
    bruker_cases.push(BrukerFixtureCase {
        name: "17_bruker_manual_dp_states_under_fnmode_0",
        acqus: make_bruker_acqus(
            "custom_manual_states_dp",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            50.0,
            0.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 0, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
; custom laboratory HSQC with manual dp3 States loop under FnMODE=0
1 ze
2 d1 do:f3
3 (p1 ph1)
  d4
  (p21 ph3):f3
  d0
  (p2 ph2)
  d0
  (p21 ph4):f3
  go=3 ph31 cpd3:f3
  d1 do:f3 wr #0 if #0 zd dp3
  lo to 3 times 2
  d1 id0
  lo to 2 times l0
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert_eq!(d.quad_mode, DecodedQuadMode::States);
            assert_eq!(
                d.quad_source,
                QuadSource::SynthesizedManualLoop { header_fnmode: 0 }
            );
            assert!(d.reverse_frequency);
            assert_eq!(d.folding, FoldingConvention::SameSignWrap);
        }),
    });

    // 18. Manual `igrad EA` `Echo-Antiecho` under `FnMODE = 1 (QF)`
    bruker_cases.push(BrukerFixtureCase {
        name: "18_bruker_manual_igrad_ea_under_fnmode_qf",
        acqus: make_bruker_acqus(
            "custom_manual_ea_hsqc",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 1, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
; custom laboratory Echo-Antiecho loop with igrad EA under FnMODE=QF
define list<gradient> EA={ 1.000, -1.000 }
1 ze
2 d1 do:f3
3 (p1 ph1)
  d4
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=3 ph31 cpd3:f3
  d1 do:f3 wr #0 if #0 zd igrad EA ip3*2
  lo to 3 times 2
  d1 id0 ip31*2
  lo to 2 times l0
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert_eq!(d.quad_mode, DecodedQuadMode::EchoAntiecho);
            assert_eq!(
                d.quad_source,
                QuadSource::SynthesizedManualLoop { header_fnmode: 1 }
            );
        }),
    });

    // 19. Inverted-`GPZ` `Echo-Antiecho` (`GPZ1 = +80.0, GPZ2 = -8.1` -> `ea_polarity_inverted = true`)
    bruker_cases.push(BrukerFixtureCase {
        name: "19_bruker_inverted_gpz_echo_antiecho",
        acqus: make_bruker_acqus(
            "hsqcetf3gpsi",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            -8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcetf3gpsi with negative decoding GPZ2 = -8.1%
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            let rk = d.rance_kay.expect("rance_kay");
            assert!(rk.ea_polarity_inverted);
            assert!(d.reverse_frequency);
            assert!(rk.grad_ratio.unwrap() < 0.0);
        }),
    });

    // 20. Explicit `*EA*-1` inverted `Echo-Antiecho` with positive `GPZ`
    bruker_cases.push(BrukerFixtureCase {
        name: "20_bruker_explicit_ea_minus1_inverted",
        acqus: make_bruker_acqus(
            "hsqcetf3gpsi",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcetf3gpsi with explicit gp1*EA*-1
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph3):f3
  d0
  p16:gp1*EA*-1
  (p22 ph4):f3
  d0
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert!(d.rance_kay.unwrap().ea_polarity_inverted);
            assert!(d.reverse_frequency);
        }),
    });

    // 21. Bipolar `*EA` gradient pair straddling 180° refocusing pulse (`gp2*EA ... p22:f3 ... gp2*-1*EA`, `GPZ2=80, GPZ3=16.2`)
    bruker_cases.push(BrukerFixtureCase {
        name: "21_bruker_bipolar_ea_gradient_pair_canonical",
        acqus: make_bruker_acqus(
            "hsqcetf3gpsi_bipolar",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            50.0,
            80.0,
            16.2,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 6, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
; bipolar EA encoding gradient pair straddling 15N 180 pulse
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph3):f3
  d0
  p16:gp2*EA
  d16
  (p22 ph4):f3
  p16:gp2*-1*EA
  d16
  (p1 ph2)
  p16:gp3
  d16
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph2=0
ph3=0
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let rk = pi.indirect_dims[0].rance_kay.unwrap();
            assert!(!rk.ea_polarity_inverted);
            assert!((rk.grad_ratio.unwrap() - (2.0 * 80.0 / 16.2)).abs() < 0.1);
            assert!(!pi.indirect_dims[0].reverse_frequency);
        }),
    });

    // 22. Half-dwell `t_1(0) = 1 / (2 * SW)` via `D[0]` -> `(PHC0, PHC1) = (-90°, 180°)` (Marion et al. 1989)
    let sw_hz_n = 32.0 * 60.8179;
    let half_dwell_d0 = 0.5 / sw_hz_n;
    bruker_cases.push(BrukerFixtureCase {
        name: "22_bruker_half_dwell_d0_marion1989",
        acqus: make_bruker_acqus(
            "hsqcf3gpph19",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            half_dwell_d0,
            0.0,
            0.0,
            0.0,
            50.0,
            30.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 5, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsqcf3gpph19 with half-dwell d0 = 1/(2*SW)
\"d0=in0/2\"
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph5):f3
  d0
  (p2 ph2)
  d0
  (p21 ph6):f3
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1PH(calph(ph5, +90), caldel(d0, +in0))
exit
ph1=0
ph2=0
ph5=0 2
ph6=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert!(d.half_dwell_sampling);
            assert_eq!(d.predicted_phc0_deg, -90.0);
            assert_eq!(d.predicted_phc1_deg, 180.0);
        }),
    });

    // 23. Runtime half-dwell equation `"d0=(in0/2-p1-(p21*2/3.14159))"` when `D[0] = 0` in `acqus`
    bruker_cases.push(BrukerFixtureCase {
        name: "23_bruker_runtime_half_dwell_expr_d0_zero_in_acqus",
        acqus: make_bruker_acqus(
            "hsgp19.fp2",
            1,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            30.0,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 128, 5, 0, None,
        )),
        acqu3s: None,
        pulseprogram: "\
;hsgp19.fp2 runtime half-dwell formula with D[0]=0 in acqus
\"in0=inf1/2\"
\"d0=(in0/2-p1-(p21*2/3.14159))\"
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph5):f3
  d0
  (p2 ph2)
  d0
  (p21 ph6):f3
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1PH(calph(ph5, +90), caldel(d0, +in0))
exit
ph1=0
ph2=0
ph5=0 2
ph6=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::N15Hsqc,
        expected_aqseq: AcquisitionOrder::Dim2D,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let d = &pi.indirect_dims[0];
            assert!(d.half_dwell_sampling);
            assert_eq!(d.predicted_phc0_deg, -90.0);
            assert_eq!(d.predicted_phc1_deg, 180.0);
        }),
    });

    // 24. Plane-queued pseudo-3D `hsqct1etf3gpsi3d` (`F1QF` outer `vd` loop -> `PlaneQueued`)
    bruker_cases.push(BrukerFixtureCase {
        name: "24_bruker_pseudo3d_t1_plane_queued",
        acqus: make_bruker_acqus(
            "hsqct1etf3gpsi3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9108,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 10.0, 8, 1, 0, None,
        )),
        pulseprogram: "\
;hsqct1etf3gpsi3d - plane-queued T1 relaxation (F2=t1 EA inner, F1=vd QF outer)
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph1):f3
  vd ; longitudinal T1 relaxation delay
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1QF(calclist(vd, +1))
    F2EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: Some("0.01\n0.05\n0.10\n0.20\n0.40\n0.60\n0.90\n1.30\n".to_string()),
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::RelaxR1,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::PlaneQueued,
        check: Box::new(|pi| {
            assert_eq!(pi.pseudo_3d_order, Pseudo3DStorageOrder::PlaneQueued);
        }),
    });

    // 25. `$t_1$-interleaved` pseudo-3D `my_t1_trosy_interleaved` (`F2QF` inner `ivd` loop inside `t1`!)
    bruker_cases.push(BrukerFixtureCase {
        name: "25_bruker_pseudo3d_t1_interleaved_custom",
        acqus: make_bruker_acqus(
            "my_t1_trosy_interleaved",
            2,
            Some(0),
            "",
            800.1300,
            201.1900,
            81.0760,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 81.0760, 81.0660, 10.0, 8, 1, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "15N", 81.0760, 81.0660, 34.0, 64, 6, 0, None,
        )),
        pulseprogram: "\
; custom t1-interleaved TROSY R1 relaxation (vd stepped INSIDE t1 loop!)
define list<gradient> EA={ 1.000, -1.000 }
1 ze
2 d1
3 d1
4 (p1 ph1)
  (p21 ph1):f3
  vd
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=4 ph31
  d1 wr #0 if #0 zd ivd
  lo to 4 times 8
  d1 igrad EA ip3*2
  lo to 3 times 2
  d1 id0 ip31*2
  lo to 2 times l0
exit
ph1=0
ph3=1 3
ph4=0
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: Some("0.01\n0.05\n0.10\n0.20\n0.40\n0.60\n0.90\n1.30\n".to_string()),
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::RelaxR1,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::T1Interleaved,
        check: Box::new(|pi| {
            assert_eq!(pi.pseudo_3d_order, Pseudo3DStorageOrder::T1Interleaved);
            assert!(pi.transfer_graph.trosy_selected);
        }),
    });

    // 26. Bruker `dstebpgp3s` & custom `bpp_led_dosy_custom` (`DOSY` diffusion QF)
    for (idx_name, pp_name) in [
        ("26a_bruker_dstebpgp3s_dosy", "dstebpgp3s"),
        ("26b_bruker_bpp_led_dosy_custom", "bpp_led_dosy_custom"),
    ] {
        bruker_cases.push(BrukerFixtureCase {
            name: idx_name,
            acqus: make_bruker_acqus(
                pp_name, 1, Some(0), "", 600.1328, 150.9108, 60.8179, "13C", "15N", 174.0, 54.0,
                3.0e-6, 0.0, 0.080, 0.0, 5.0, -5.0, 17.13, 67.984, 20, 2000,
            ),
            acqu2s: Some(make_bruker_acqu_ind(
                "1H", 600.1328, 600.1300, 10.0, 16, 1, 0, None,
            )),
            acqu3s: None,
            pulseprogram: format!(
                "\
;{pp_name} - bipolar gradient stimulated echo LED diffusion
define list<gradient> diff_ramp=<difflist>
1 ze
2 d1
  (p1 ph1)
  p30:gp6*diff_ramp
  (p2 ph1)
  p30:gp6*-1*diff_ramp
  (p1 ph2)
  d20 ; Big-Delta diffusion delay 80 ms
  (p1 ph3)
  p30:gp6*diff_ramp
  (p2 ph1)
  p30:gp6*-1*diff_ramp
  go=2 ph31
  d1 wr #0 if #0 zd igrad diff_ramp
  lo to 2 times td1
exit
ph1=0
ph2=0
ph3=0
ph31=0 2
"
            ),
            nuslist: None,
            vdlist: None,
            vclist: None,
            fqlist: None,
            difflist: Some(
                "2.0\n8.0\n14.0\n20.0\n26.0\n32.0\n38.0\n44.0\n50.0\n56.0\n62.0\n68.0\n74.0\n80.0\n88.0\n95.0\n"
                    .to_string(),
            ),
            local_includes: HashMap::new(),
            expected_family: PulprogSequenceFamily::Dosy,
            expected_aqseq: AcquisitionOrder::Dim2D,
            expected_pseudo: Pseudo3DStorageOrder::PlaneQueued,
            check: Box::new(|pi| {
                assert!(pi.transfer_graph.diffusion_encoded);
                assert_eq!(pi.indirect_dims[0].quad_mode, DecodedQuadMode::Qf);
                assert_eq!(pi.indirect_dims[0].folding, FoldingConvention::Magnitude);
                assert_eq!(
                    pi.transfer_graph.deduced_family.to_pulse_sequence_class(),
                    PulseSequenceClass::Unknown
                );
            }),
        });
    }

    // 27 & 28. `#ifdef` / `ZGOPTNS` custom lab sequence `custom_trhncacb_v2` WITH vs WITHOUT `-DTROSY -DCACB_TRANSFER`!
    let custom_tr_pp = "\
; custom_trhncacb_v2 - lab sequence whose PULPROG name hides whether CB transfer or TROSY is active
#include \"lab_timings.incl\"
define list<gradient> EA=<EA>
1 ze
2 d1
#ifndef TROSY
  do:f3
#endif
  (p1 ph1)
  (p21 ph1):f3
  d21
  (p3 ph3):f2
#ifdef CACB_TRANSFER
  d28 ; 3.6 ms CA-CB J_CC transfer enabled by -DCACB_TRANSFER!
#endif
  d0
  (p14:sp3 ph1):f2
  d0
  (p3 ph4):f2
  d30
  p16:gp1*EA
  (p22 ph2):f3
  d10
  p16:gp2
#ifdef TROSY
  go=2 ph31
#else
  go=2 ph31 cpd3:f3
#endif
  d1 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0))
    F2EA(calgrad(EA), caldel(d10, +in10) & caldel(d30, -in30))
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
";
    let mut lab_inc = HashMap::new();
    lab_inc.insert("lab_timings.incl".to_string(), "\"d28=3.6m\"\n".to_string());

    bruker_cases.push(BrukerFixtureCase {
        name: "27_bruker_ifdef_zgoptns_trosy_cacb_enabled",
        acqus: make_bruker_acqus(
            "custom_tr_seq_v2",
            2,
            Some(0),
            "-DTROSY -DLABEL_CN -DCACB_TRANSFER",
            800.1300,
            201.1900,
            81.0760,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 81.0760, 81.0660, 34.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 201.1900, 201.1810, 72.0, 128, 5, 0, None,
        )),
        pulseprogram: custom_tr_pp.to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: lab_inc.clone(),
        expected_family: PulprogSequenceFamily::Hncacb,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert!(pi.transfer_graph.trosy_selected);
            assert_eq!(pi.resolved_local_includes, vec!["lab_timings.incl"]);
        }),
    });

    bruker_cases.push(BrukerFixtureCase {
        name: "28_bruker_ifdef_zgoptns_hnca_branch_when_cacb_off",
        acqus: make_bruker_acqus(
            "custom_tr_seq_v2",
            2,
            Some(0),
            "-DLABEL_CN",
            600.1328,
            150.9109,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9109, 150.9028, 30.0, 128, 5, 0, None,
        )),
        pulseprogram: custom_tr_pp.to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: lab_inc,
        expected_family: PulprogSequenceFamily::Hnca,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert!(!pi.transfer_graph.trosy_selected);
        }),
    });

    // 29. Custom renamed `lab_seq_cbca_via_co` & `lab_cest_sweep` + Moskau (2002) `DSPFVS=12, DECIM=64` group delay
    bruker_cases.push(BrukerFixtureCase {
        name: "29_bruker_custom_renamed_cbca_via_co_and_dspfvs12_grpdly",
        acqus: make_bruker_acqus(
            "lab_seq_alpha_beta_07",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0035,
            0.0035,
            80.0,
            8.1,
            0.0,
            -1.0,
            12,
            64,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 70.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
; opaque custom name: 1H->13CAB->13CO->15N->1H
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p3 ph3):f2
  d0
  (p4 ph1):f2
  d20
  fq=cnst20(bf ppm):f2
  (p13:sp1 ph1):f2
  d24
  (p21 ph1):f3
  d10
  p16:gp1*EA
  (p22 ph2):f3
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2
    F1PH(calph(ph3, +90), caldel(d0, +in0) & caldel(d20, -in20))
    F2EA(calgrad(EA), caldel(d10, +in10))
exit
ph1=0
ph2=0
ph3=0 2
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Cbcaconh,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            // Moskau (2002) Table 1: DSPFVS=12, DECIM=64 -> 72.348 complex points!
            assert!((pi.direct_group_delay_pts - 72.348).abs() < 1e-4);
        }),
    });

    // 30. 1-based 3D `nuslist` auto-detection (`[1..=32, 1..=64]`)
    bruker_cases.push(BrukerFixtureCase {
        name: "30_bruker_1based_nuslist_3d",
        acqus: make_bruker_acqus(
            "hncogp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9290,
            60.8179,
            "13C",
            "15N",
            174.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N",
            60.8179,
            60.8106,
            32.0,
            32,
            6,
            2,
            Some(64),
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C",
            150.9290,
            150.9028,
            14.0,
            1,
            5,
            2,
            Some(128),
        )),
        pulseprogram: ";hncogp3d\n1 ze\n2 go=2 ph31\nexit\nph31=0 2\n".to_string(),
        nuslist: Some("1 1\n2 4\n5 12\n9 25\n14 39\n19 48\n27 55\n32 64\n".to_string()),
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hnco,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            let nus = pi.nus_schedule.as_ref().expect("nus");
            assert_eq!(nus.index_base, NusIndexBase::OneBased);
            assert_eq!(nus.points[0], vec![0, 0]);
            assert_eq!(nus.points[7], vec![31, 63]);
        }),
    });

    // 31. Bruker `hncacbgp3d` with readout `ph9 = 3 1` (`270°` initial phase offset) after `d0`:
    //     Constant phase offset `exp(+-i*phi_0)` MUST NOT invert `reverse_frequency` (`true` for readout `ip9` after `d0`).
    bruker_cases.push(BrukerFixtureCase {
        name: "31_bruker_hncacbgp3d_readout_ph9_3_1_reverse",
        acqus: make_bruker_acqus(
            "hncacbgp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            174.0,
            43.0,
            3.0e-6,
            0.0,
            0.0,
            0.0036,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 70.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncacbgp3d - readout ph9=3 1 after d0 evolution
define list<gradient> EA=<EA>
\"d0=3u\"
\"d26=2.3m\"
\"d28=3.6m\"
\"d23=12.4m\"
\"d10=d23/2\"
\"d30=d23/2\"
1 ze
2 MCWRK do:f3
LBLSTS2, MCWRK * 3
LBLF2, MCWRK * 5
LBLSTS1, MCWRK * 2
LBLF1, MCREST
3 d1 do:f3
  (p1 ph1)
  d26
  (p21 ph3):f3
  d23
  (p13:sp2 ph4):f2
  d28
  (p14:sp3 ph1):f2
  d28
  (p13:sp8 ph2):f2
  d0
  (center (p14:sp5 ph1):f2 (p22 ph8):f3)
  d0
  (p13:sp2 ph9):f2
  d28
  (p14:sp3 ph1):f2
  d28
  (p13:sp8 ph10):f2
  (p21 ph1):f3
  d30
  (p22 ph8):f3
  d10
  p16:gp1*EA
  p16:gp2
  go=2 ph31 cpd3:f3
  MCWRK wr #0 if #0 zd igrad EA ip6*2
  lo to LBLSTS2 times 2
  MCWRK id10 dd30
  lo to LBLF2 times 32
  MCWRK rd10 rd30 ip9 ip10
  lo to LBLSTS1 times 2
  MCWRK id0 dp9*2
  lo to LBLF1 times 64
exit
ph1=0
ph2=1
ph3=0
ph4=0
ph6=3 3 1 1
ph8=0 2
ph9=3 1
ph10=0 2
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Hncacb,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            // `acqu3s` (`F1`, `13C`) uses readout `ip9` after `d0` with `ph9 = 3 1`: MUST have `reverse_frequency == true`!
            assert_eq!(pi.indirect_dims[1].quad_mode, DecodedQuadMode::StatesTppi);
            assert!(
                pi.indirect_dims[1].reverse_frequency,
                "readout ip9 after d0 must set reverse_frequency=true even when ph9=3 1 starts at 270 deg"
            );
        }),
    });

    // 32. Out-and-back `hncocacbgp3d` (`HN(CO)CACB` = `CBCACONH`):
    //     Must classify as `Cbcaconh` (never hijacked by `"hncoca"` substring or single-relay `Hncacb`).
    bruker_cases.push(BrukerFixtureCase {
        name: "32_bruker_hncocacbgp3d_out_and_back_cbcaconh",
        acqus: make_bruker_acqus(
            "hncocacbgp3d",
            2,
            Some(0),
            "",
            600.1328,
            150.9092,
            60.8179,
            "13C",
            "15N",
            176.0,
            54.0,
            3.0e-6,
            0.0,
            0.0,
            0.0036,
            80.0,
            8.1,
            0.0,
            67.984,
            20,
            2000,
        ),
        acqu2s: Some(make_bruker_acqu_ind(
            "15N", 60.8179, 60.8106, 32.0, 64, 6, 0, None,
        )),
        acqu3s: Some(make_bruker_acqu_ind(
            "13C", 150.9092, 150.9028, 70.0, 128, 5, 0, None,
        )),
        pulseprogram: "\
;hncocacbgp3d - out-and-back HN(CO)CACB
define list<gradient> EA=<EA>
\"d0=3u\"
\"d22=4m\"
\"d23=12m\"
\"d26=2.3m\"
\"d28=3.6m\"
\"d10=d23/2\"
\"d30=d23/2\"
1 ze
2 d1 fq=cnst20(bf ppm):f2
  (p1 ph1)
  d26
  (p21 ph1):f3
  d23
  (p13:sp2 ph3):f2
  d22
  (p13:sp8 ph2):f2
  30u fq=cnst21(bf ppm):f2
  (p13:sp2 ph4):f2
  d28
  (p13:sp8 ph2):f2
  d0
  (center (p14:sp7 ph1):f2 (p22 ph8):f3)
  d0
  (p13:sp2 ph9):f2
  d28
  (p13:sp8 ph10):f2
  30u fq=cnst20(bf ppm):f2
  (p13:sp2 ph2):f2
  d22
  (p13:sp8 ph1):f2
  (p21 ph1):f3
  d30
  (p22 ph8):f3
  d10
  p16:gp1*EA
  p16:gp2
  go=2 ph31 cpd3:f3
  MCWRK wr #0 if #0 zd igrad EA ip6*2
  lo to 2 times 2
  MCWRK id10 dd30
  lo to 2 times 32
  MCWRK rd10 rd30 ip9 ip10
  lo to 2 times 2
  MCWRK id0 dp9*2
  lo to 2 times 64
exit
ph1=0
ph2=1
ph3=0
ph4=0
ph6=3 3 1 1
ph8=0 2
ph9=3 1
ph10=0 2
ph31=0 2
"
        .to_string(),
        nuslist: None,
        vdlist: None,
        vclist: None,
        fqlist: None,
        difflist: None,
        local_includes: HashMap::new(),
        expected_family: PulprogSequenceFamily::Cbcaconh,
        expected_aqseq: AcquisitionOrder::Aqseq321,
        expected_pseudo: Pseudo3DStorageOrder::NotPseudo3D,
        check: Box::new(|pi| {
            assert_eq!(
                pi.transfer_graph.deduced_family.to_pulse_sequence_class(),
                PulseSequenceClass::Cbcaconh
            );
            assert!(pi.indirect_dims[1].reverse_frequency);
        }),
    });

    // Execute all Bruker fixtures and assert 100% accuracy + `< 10 ms` latency per fixture!
    let mut bruker_max_ms = 0.0_f64;
    let mut bruker_sum_ms = 0.0_f64;
    for case in &bruker_cases {
        let inp = BrukerExperimentInputs {
            acqus_text: &case.acqus,
            acqu2s_text: case.acqu2s.as_deref(),
            acqu3s_text: case.acqu3s.as_deref(),
            acqu4s_text: None,
            pulseprogram_text: Some(&case.pulseprogram),
            nuslist_text: case.nuslist.as_deref(),
            vdlist_text: case.vdlist.as_deref(),
            vclist_text: case.vclist.as_deref(),
            fqlist_text: case.fqlist.as_deref(),
            difflist_text: case.difflist.as_deref(),
            local_includes: case.local_includes.clone(),
        };
        let t0 = Instant::now();
        let pi = interpret_bruker_experiment(&inp)
            .unwrap_or_else(|e| panic!("fixture {} failed: {e}", case.name));
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        bruker_max_ms = bruker_max_ms.max(elapsed_ms);
        bruker_sum_ms += elapsed_ms;

        assert_eq!(pi.vendor, PulprogVendor::Bruker, "{}", case.name);
        assert_eq!(
            pi.transfer_graph.deduced_family, case.expected_family,
            "family mismatch on {}",
            case.name
        );
        assert_eq!(
            pi.aqseq, case.expected_aqseq,
            "aqseq mismatch on {}",
            case.name
        );
        assert_eq!(
            pi.pseudo_3d_order, case.expected_pseudo,
            "pseudo3d mismatch on {}",
            case.name
        );
        (case.check)(&pi);
        assert!(
            elapsed_ms < 10.0,
            "fixture {} took {:.3} ms (>= 10 ms budget)",
            case.name,
            elapsed_ms
        );
    }

    // Varian / Agilent `procpar` fixtures (31..=36):
    // 31. Varian `gNhsqc` (2D 15N SE Rance-Kay)
    let v31 = make_varian_procpar(
        "gNhsqc",
        "gNhsqc",
        "C13",
        "N15",
        150.9028,
        60.8106,
        0.0,
        0.0,
        64,
        1946.0,
        &[1, 2],
        "1 0 -1 0 0 1 0 1",
        "n",
        None,
        "phase",
        "",
    );
    let t0 = Instant::now();
    let pi31 = interpret_varian_procpar(&v31, None).unwrap();
    assert!(t0.elapsed().as_secs_f64() * 1000.0 < 10.0);
    assert_eq!(pi31.vendor, PulprogVendor::Varian);
    assert_eq!(
        pi31.transfer_graph.deduced_family,
        PulprogSequenceFamily::N15Hsqc
    );
    assert_eq!(
        pi31.indirect_dims[0].quad_mode,
        DecodedQuadMode::EchoAntiecho
    );
    assert!(
        !pi31.indirect_dims[0]
            .rance_kay
            .unwrap()
            .ea_polarity_inverted
    );

    // 32. Varian `ghsqc` (2D 13C aliphatic HSQC)
    let v32 = make_varian_procpar(
        "ghsqc",
        "ghsqc",
        "C13",
        "N15",
        150.9028,
        60.8106,
        -2200.0,
        0.0,
        128,
        10500.0,
        &[1, 2],
        "1 0 0 0 0 0 -1 0",
        "n",
        None,
        "phase",
        "",
    );
    let t0 = Instant::now();
    let pi32 = interpret_varian_procpar(&v32, None).unwrap();
    assert!(t0.elapsed().as_secs_f64() * 1000.0 < 10.0);
    assert_eq!(
        pi32.transfer_graph.deduced_family,
        PulprogSequenceFamily::C13Hsqc
    );

    // 33. Varian `ghncacb` (3D HNCACB, `f1180 = 'y'` half-dwell `(-90°, 180°)`)
    let v33 = make_varian_procpar(
        "ghn_cacb",
        "ghn_cacb",
        "C13",
        "N15",
        150.9028,
        60.8106,
        -2100.0,
        0.0,
        64,
        10800.0,
        &[1, 2],
        "1 0 0 0 0 0 1 0",
        "y",
        Some((32, 1946.0, &[1, 2], "1 0 -1 0 0 -1 0 -1", "n")),
        "phase,phase2",
        "",
    );
    let t0 = Instant::now();
    let pi33 = interpret_varian_procpar(&v33, None).unwrap();
    assert!(t0.elapsed().as_secs_f64() * 1000.0 < 10.0);
    assert_eq!(
        pi33.transfer_graph.deduced_family,
        PulprogSequenceFamily::Hncacb
    );
    assert_eq!(pi33.aqseq, AcquisitionOrder::Aqseq321);
    assert!(pi33.indirect_dims[0].half_dwell_sampling);
    assert_eq!(
        (
            pi33.indirect_dims[0].predicted_phc0_deg,
            pi33.indirect_dims[0].predicted_phc1_deg
        ),
        (-90.0, 180.0)
    );

    // 34. Varian `gcbca_co_nh` with `array = "phase2,phase"` (`VarianPhase1Fastest`) + 1-based `sampling.sch`
    let v34 = make_varian_procpar(
        "gcbca_co_nh",
        "gcbca_co_nh",
        "C13",
        "N15",
        150.9028,
        60.8106,
        -2100.0,
        0.0,
        64,
        10800.0,
        &[1, 2],
        "1 0 0 0 0 0 -1 0",
        "n",
        Some((32, 1946.0, &[1, 2], "1 0 1 0 0 -1 0 1", "n")),
        "phase2,phase",
        "",
    );
    let t0 = Instant::now();
    let pi34 = interpret_varian_procpar(
        &v34,
        Some("1 1\n4 3\n12 9\n21 14\n33 19\n44 25\n58 30\n64 32\n"),
    )
    .unwrap();
    assert!(t0.elapsed().as_secs_f64() * 1000.0 < 10.0);
    assert_eq!(
        pi34.transfer_graph.deduced_family,
        PulprogSequenceFamily::Cbcaconh
    );
    assert_eq!(pi34.aqseq, AcquisitionOrder::VarianPhase1Fastest);
    assert_eq!(
        pi34.nus_schedule.as_ref().unwrap().index_base,
        NusIndexBase::OneBased
    );

    // 35 & 36. Varian arrayed `gNhsqc` (`"relaxT,phase"` & `"relaxT,d2,phase"` -> `PlaneQueued`;
    // `"d2,relaxT,phase"` -> `T1Interleaved` per VnmrJ `psg` physics!) & `Dbppste` DOSY
    let relax_extra = "\
relaxT 1 1 10 0 0 2 1 11 1 64\n\
6 0.01 0.05 0.1 0.2 0.5 1.0\n\
0\n";
    for pq_arr in ["relaxT,phase", "relaxT,d2,phase"] {
        let v35_pq = make_varian_procpar(
            "gNhsqc",
            "gNhsqc_T1",
            "C13",
            "N15",
            150.9028,
            60.8106,
            0.0,
            0.0,
            64,
            1946.0,
            &[1, 2],
            "1 0 -1 0 0 1 0 1",
            "n",
            None,
            pq_arr,
            relax_extra,
        );
        let pi35_pq = interpret_varian_procpar(&v35_pq, None).unwrap();
        assert_eq!(
            pi35_pq.pseudo_3d_order,
            Pseudo3DStorageOrder::PlaneQueued,
            "expected PlaneQueued for Varian array='{pq_arr}'"
        );
        assert_eq!(
            pi35_pq.transfer_graph.deduced_family,
            PulprogSequenceFamily::RelaxR1
        );
    }

    let v35_il = make_varian_procpar(
        "gNhsqc",
        "gNhsqc_T1",
        "C13",
        "N15",
        150.9028,
        60.8106,
        0.0,
        0.0,
        64,
        1946.0,
        &[1, 2],
        "1 0 -1 0 0 1 0 1",
        "n",
        None,
        "d2,relaxT,phase",
        relax_extra,
    );
    let pi35_il = interpret_varian_procpar(&v35_il, None).unwrap();
    assert_eq!(pi35_il.pseudo_3d_order, Pseudo3DStorageOrder::T1Interleaved);
    assert_eq!(pi35_il.pseudo_3d_list_kind.as_deref(), Some("relaxT"));

    let dosy_extra = "\
gzlvl1 1 1 32767 -32767 0 2 1 11 1 64\n\
8 1000 4000 8000 12000 16000 20000 24000 28000\n\
0\n";
    let v36_dosy = make_varian_procpar(
        "Dbppste",
        "Dbppste",
        "C13",
        "N15",
        150.9028,
        60.8106,
        0.0,
        0.0,
        1,
        8000.0,
        &[1],
        "",
        "n",
        None,
        "gzlvl1",
        dosy_extra,
    );
    let pi36 = interpret_varian_procpar(&v36_dosy, None).unwrap();
    assert_eq!(
        pi36.transfer_graph.deduced_family,
        PulprogSequenceFamily::Dosy
    );

    assert!(
        bruker_cases.len() + 6 >= 35,
        "verified {} total fixtures (>= 28 required)",
        bruker_cases.len() + 6
    );
    // Count 30 Bruker + 8 Varian assertions (31..34 + 35a/35b/35c + 36)
    let n_bruker = bruker_cases.len();
    let n_varian = 8usize;
    println!(
        "PULPROG_METRIC_JSON: {{\"kind\":\"fixture_suite\",\"bruker_fixtures\":{},\"varian_fixtures\":{},\"total_fixtures\":{},\"passed\":{},\"bruker_mean_ms\":{:.4},\"bruker_max_ms\":{:.4}}}",
        n_bruker,
        n_varian,
        n_bruker + n_varian,
        n_bruker + n_varian,
        bruker_sum_ms / (n_bruker as f64),
        bruker_max_ms
    );
}

// ============================================================================
// Test 3: End-to-End Binary `ser` Processing & `classify.rs` Integration
// ============================================================================

fn write_bruker_ser_traces(path: &Path, td_words: usize, traces: &[Vec<[f64; 2]>]) {
    let raw_bytes = td_words * 4;
    let stride = raw_bytes.div_ceil(1024) * 1024;
    let mut buf = vec![0u8; traces.len() * stride];
    for (ti, tr) in traces.iter().enumerate() {
        let base = ti * stride;
        for (k, &[re, im]) in tr.iter().enumerate() {
            let r_i32 = (re * 1.0e5).round() as i32;
            let i_i32 = (im * 1.0e5).round() as i32;
            buf[base + k * 8..base + k * 8 + 4].copy_from_slice(&r_i32.to_le_bytes());
            buf[base + k * 8 + 4..base + k * 8 + 8].copy_from_slice(&i_i32.to_le_bytes());
        }
    }
    fs::write(path, buf).unwrap();
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_end_to_end_manual_states_tppi_inverted_ea_pseudo3d_and_custom_classification() {
    let tmp = std::env::temp_dir().join(format!("nmr_pulprog_e2e_{}_{}", std::process::id(), 202));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // ------------------------------------------------------------------------
    // Case A: Manual `ip3` + `ip31*2` `States-TPPI` under `FnMODE = 1 (QF)`!
    // ------------------------------------------------------------------------
    let dir_manual = tmp.join("01_manual_states_tppi_qf");
    fs::create_dir_all(&dir_manual).unwrap();
    let sfo_h = 600.132864_f64;
    let bf_h = 600.130000_f64;
    let sw_h_hz = 8000.0_f64;
    let sfo_n = 60.817965_f64; // carrier ~ 121.1 ppm relative to BF1=60.8106
    let bf_n = 60.810600_f64;
    let sw_n_ppm = 32.0_f64;
    let sw_n_hz = sw_n_ppm * sfo_n;
    let n0 = 256usize;
    let n1 = 64usize;

    let acqus_manual = make_bruker_acqus(
        "lab_opaque_seq_a",
        1,
        Some(0),
        "",
        sfo_h,
        150.9108,
        sfo_n,
        "13C",
        "15N",
        174.0,
        54.0,
        0.0,
        0.0,
        0.0,
        0.0,
        50.0,
        0.0,
        0.0,
        0.0,
        0,
        1,
    );
    let acqu2s_manual = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, n1 * 2, 1, 0, None);
    let pp_manual = "\
; manual ip3 / ip31*2 States-TPPI loop while acqu2s has FnMODE = 1 (QF)
1 ze
2 d1 do:f3
3 (p1 ph1)
  d4
  (p21 ph3):f3
  d0
  (p2 ph2)
  d0
  (p21 ph4):f3
  go=3 ph31 cpd3:f3
  d1 do:f3 wr #0 if #0 zd ip3
  lo to 3 times 2
  d1 id0 ip3*2 ip31*2
  lo to 2 times l0
exit
ph1=0
ph2=0
ph3=0 2
ph4=0
ph31=0 2
";
    fs::write(dir_manual.join("acqus"), &acqus_manual).unwrap();
    fs::write(dir_manual.join("acqu2s"), &acqu2s_manual).unwrap();
    fs::write(dir_manual.join("pulseprogram"), pp_manual).unwrap();

    // Synthesize exact Bruker raw `ser` traces at `(8.35 ppm, 125.60 ppm)` with States-TPPI `(-1)^k1`!
    let target_h = 8.35_f64;
    let target_n = 125.60_f64;
    let f_h_hz = (target_h * bf_h - (sfo_h - bf_h) * 1.0e6) - 0.0; // wait: exact carrier offset `(ppm * bf - (sfo - bf)*1e6)`!
    let _ = f_h_hz;
    let nu_h = target_h * bf_h - (sfo_h - bf_h) * 1.0e6;
    let nu_n = target_n * bf_n - (sfo_n - bf_n) * 1.0e6;

    let mut traces_manual = Vec::with_capacity(n1 * 2);
    for k1 in 0..n1 {
        let t1 = (k1 as f64) / sw_n_hz;
        let env1 = (-t1 / 0.035).exp();
        let tppi_sign = if k1 % 2 == 0 { 1.0 } else { -1.0 };
        // `ip3` (+90 deg) without `REVERSE`: `cos(+2*pi*nu_n*t1)` and `sin(+2*pi*nu_n*t1)`
        let c_n = tppi_sign * env1 * (2.0 * PI * nu_n * t1).cos();
        let s_n = tppi_sign * env1 * (2.0 * PI * nu_n * t1).sin();

        for mod_val in [c_n, s_n] {
            let mut tr = Vec::with_capacity(n0);
            for k0 in 0..n0 {
                let t0 = (k0 as f64) / sw_h_hz;
                let env0 = (-t0 / 0.045).exp();
                // Bruker conjugate direct convention: `z = exp(+i * 2 * pi * nu_h * t0)` so `z_conj = exp(-i * 2 * pi * nu_h * t0)`
                let ang0 = 2.0 * PI * nu_h * t0;
                tr.push([mod_val * env0 * ang0.cos(), mod_val * env0 * ang0.sin()]);
            }
            traces_manual.push(tr);
        }
    }
    write_bruker_ser_traces(&dir_manual.join("ser"), n0 * 2, &traces_manual);

    let fid_man = read_fid_2d(&dir_manual).expect("read_fid_2d manual States-TPPI");
    assert_eq!(fid_man.axis_y.quad_mode, QuadMode::StatesTppi);
    assert_eq!(fid_man.axis_y.complex_points, n1);
    let mut rec_man = ProcessingRecipe::from_fid_2d(&fid_man);
    rec_man.dims[0].roi_ppm = None;
    rec_man.dims[0].p0_deg = Some(0.0);
    rec_man.dims[0].p1_deg = Some(0.0);
    rec_man.dims[1].p0_deg = Some(0.0);
    rec_man.dims[1].p1_deg = Some(0.0);
    let spec_man = process_fid_2d(&fid_man, &rec_man).unwrap();

    // Find global maximum in `spec_man`: must sit at `(8.35, 125.60 ppm)`, NOT half-SW shifted!
    let (mut max_v, mut max_idx) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in spec_man.data.iter().enumerate() {
        if v > max_v {
            max_v = v;
            max_idx = i;
        }
    }
    let iy_m = max_idx / spec_man.axis_x.size;
    let ix_m = max_idx % spec_man.axis_x.size;
    let obs_h = spec_man.axis_x.index_to_ppm(ix_m as f64);
    let obs_n = spec_man.axis_y.index_to_ppm(iy_m as f64);
    assert!(
        (obs_h - target_h).abs() < 0.06 && (obs_n - target_n).abs() < 0.30,
        "manual States-TPPI peak expected ({target_h:.2}, {target_n:.2}), got ({obs_h:.2}, {obs_n:.2})"
    );

    // Also verify `classify_nmr_workspace` upgrades `QuadratureKind::Qf -> StatesTppi` AND classifies `lab_opaque_seq_a` as `N15Hsqc`!
    let inv_man = classify_nmr_workspace(&dir_manual).unwrap();
    assert_eq!(inv_man.experiments.len(), 1);
    assert_eq!(
        inv_man.experiments[0].pulse_class,
        PulseSequenceClass::N15Hsqc
    );
    assert_eq!(
        inv_man.experiments[0].axes[1].quadrature,
        QuadratureKind::StatesTppi
    );

    // ------------------------------------------------------------------------
    // Case B: Inverted-`GPZ` `Echo-Antiecho` (`GPZ1 = +80, GPZ2 = -8.1`)!
    // ------------------------------------------------------------------------
    let dir_inv_ea = tmp.join("02_inverted_gpz_ea");
    fs::create_dir_all(&dir_inv_ea).unwrap();
    let acqus_inv = make_bruker_acqus(
        "hsqcetf3gpsi",
        1,
        Some(0),
        "",
        sfo_h,
        150.9108,
        sfo_n,
        "13C",
        "15N",
        174.0,
        54.0,
        0.0,
        0.0,
        0.0,
        0.0,
        80.0,
        -8.1, // Opposite-sign decoding gradient swaps P-type and N-type pathways!
        0.0,
        0.0,
        0,
        1,
    );
    let acqu2s_inv = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, n1 * 2, 6, 0, None);
    let pp_inv = "\
;hsqcetf3gpsi
define list<gradient> EA=<EA>
1 ze
2 d1 do:f3
  (p1 ph1)
  (p21 ph3):f3
  d0
  p16:gp1*EA
  (p22 ph4):f3
  d0
  p16:gp2
  go=2 ph31 cpd3:f3
  d1 do:f3 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
";
    fs::write(dir_inv_ea.join("acqus"), &acqus_inv).unwrap();
    fs::write(dir_inv_ea.join("acqu2s"), &acqu2s_inv).unwrap();
    fs::write(dir_inv_ea.join("pulseprogram"), pp_inv).unwrap();

    // Under `recombine_quad_pair` (`EchoAntiecho`):
    // Canonical `(GPZ1*GPZ2 > 0, REVERSE=false)` has `z0_conj = i * exp(-i*w1*t1) * s0`, `z1_conj = i * exp(+i*w1*t1) * s0`.
    // When `GPZ2 < 0` swaps the P/N selection pathways (`z0 <-> z1`), `z0_conj` carries `+w1*t1` and `z1_conj` carries `-w1*t1`!
    let mut traces_inv_ea = Vec::with_capacity(n1 * 2);
    for k1 in 0..n1 {
        let t1 = (k1 as f64) / sw_n_hz;
        let env1 = (-t1 / 0.035).exp();
        let w1 = 2.0 * PI * nu_n * t1;
        // Swapped P/N traces (`z0` has `-w1`, `z1` has `+w1` in `z_conj` vs canonical `+w1 / -w1`):
        for sign_w1 in [-1.0_f64, 1.0_f64] {
            let mut tr = Vec::with_capacity(n0);
            for k0 in 0..n0 {
                let t0 = (k0 as f64) / sw_h_hz;
                let env0 = (-t0 / 0.045).exp();
                let w0 = 2.0 * PI * nu_h * t0;
                let phi = w0 - 0.5 * PI - sign_w1 * w1;
                tr.push([env1 * env0 * phi.cos(), env1 * env0 * phi.sin()]);
            }
            traces_inv_ea.push(tr);
        }
    }
    write_bruker_ser_traces(&dir_inv_ea.join("ser"), n0 * 2, &traces_inv_ea);

    let fid_inv = read_fid_2d(&dir_inv_ea).expect("read_fid_2d inverted EA");
    assert!(
        fid_inv.axis_y.stored_proc.reverse,
        "pulprog must set reverse=true for inverted GPZ Echo-Antiecho"
    );
    let mut rec_inv = ProcessingRecipe::from_fid_2d(&fid_inv);
    rec_inv.dims[0].roi_ppm = None;
    rec_inv.dims[0].p0_deg = Some(0.0);
    rec_inv.dims[0].p1_deg = Some(0.0);
    rec_inv.dims[1].p0_deg = Some(0.0);
    rec_inv.dims[1].p1_deg = Some(0.0);
    let spec_inv = process_fid_2d(&fid_inv, &rec_inv).unwrap();

    let (mut max_v2, mut max_idx2) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in spec_inv.data.iter().enumerate() {
        if v > max_v2 {
            max_v2 = v;
            max_idx2 = i;
        }
    }
    let iy_inv = max_idx2 / spec_inv.axis_x.size;
    let obs_n_inv = spec_inv.axis_y.index_to_ppm(iy_inv as f64);
    assert!(
        (obs_n_inv - target_n).abs() < 0.35,
        "inverted-GPZ EA peak expected {target_n:.2} ppm, got {obs_n_inv:.2} ppm"
    );

    // ------------------------------------------------------------------------
    // Case C: Pseudo-3D `PlaneQueued` vs `T1Interleaved` de-interleaving!
    // ------------------------------------------------------------------------
    let decay_amps = [1.0_f64, 0.78_f64, 0.52_f64, 0.29_f64];
    let n_planes = decay_amps.len();
    let n1_p3d = 16usize;
    let n0_p3d = 256usize;

    for (sub_name, interleaved) in [
        ("03a_pseudo3d_plane_queued", false),
        ("03b_pseudo3d_t1_interleaved", true),
    ] {
        let pdir = tmp.join(sub_name);
        fs::create_dir_all(&pdir).unwrap();
        let acqus_p = make_bruker_acqus(
            "hsqct1etf3gpsi3d",
            2,
            Some(0),
            "",
            sfo_h,
            150.9108,
            sfo_n,
            "13C",
            "15N",
            174.0,
            54.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            0.0,
            0,
            1,
        );
        // Both have `acqu2s = States (FnMODE=4, TD=32)` and `acqu3s = QF (FnMODE=1, TD=4)`,
        // differing ONLY in whether `ivd` advances outside `t1` (`PlaneQueued`) or inside `t1` (`T1Interleaved`)!
        let a2_p = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, n1_p3d * 2, 4, 0, None);
        let a3_p = make_bruker_acqu_ind("15N", sfo_n, bf_n, 10.0, n_planes, 1, 0, None);
        let pp_p = if !interleaved {
            "\
; plane-queued: outer vd loop, inner t1 loop
1 ze
2 d1
3 d1
4 (p1 ph1)
  vd
  (p21 ph3):f3
  d0
  (p21 ph4):f3
  go=4 ph31
  d1 wr #0 if #0 zd ip3
  lo to 4 times 2
  d1 id0
  lo to 3 times 16
  d1 rd0 ivd
  lo to 2 times 4
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        } else {
            "\
; t1-interleaved: inner vd loop inside outer t1 loop!
1 ze
2 d1
3 d1
4 (p1 ph1)
  vd
  (p21 ph3):f3
  d0
  (p21 ph4):f3
  go=4 ph31
  d1 wr #0 if #0 zd ip3
  lo to 4 times 2
  d1 ivd
  lo to 3 times 4
  d1 id0
  lo to 2 times 16
exit
ph1=0
ph3=0
ph4=0
ph31=0 2
"
        };
        fs::write(pdir.join("acqus"), &acqus_p).unwrap();
        fs::write(pdir.join("acqu2s"), &a2_p).unwrap();
        fs::write(pdir.join("acqu3s"), &a3_p).unwrap();
        fs::write(pdir.join("pulseprogram"), pp_p).unwrap();
        fs::write(pdir.join("vdlist"), "0.01\n0.25\n0.65\n1.25\n").unwrap();

        let make_trace = |amp: f64, k1: usize, is_sin: bool| -> Vec<[f64; 2]> {
            let t1 = (k1 as f64) / sw_n_hz;
            let m = amp
                * (-t1 / 0.04).exp()
                * if !is_sin {
                    (-2.0 * PI * nu_n * t1).cos()
                } else {
                    (-2.0 * PI * nu_n * t1).sin()
                };
            (0..n0_p3d)
                .map(|k0| {
                    let t0 = (k0 as f64) / sw_h_hz;
                    let e0 = (-t0 / 0.04).exp();
                    let w0 = 2.0 * PI * nu_h * t0;
                    [m * e0 * w0.cos(), m * e0 * w0.sin()]
                })
                .collect()
        };

        let mut ser_traces = Vec::with_capacity(n_planes * n1_p3d * 2);
        if !interleaved {
            for &amp in &decay_amps {
                for k1 in 0..n1_p3d {
                    ser_traces.push(make_trace(amp, k1, false));
                    ser_traces.push(make_trace(amp, k1, true));
                }
            }
        } else {
            for k1 in 0..n1_p3d {
                for &amp in &decay_amps {
                    ser_traces.push(make_trace(amp, k1, false));
                    ser_traces.push(make_trace(amp, k1, true));
                }
            }
        }
        write_bruker_ser_traces(&pdir.join("ser"), n0_p3d * 2, &ser_traces);

        let (planes, detected_order) = read_bruker_pseudo3d_planes_2d(&pdir).unwrap();
        assert_eq!(
            detected_order,
            if interleaved {
                Pseudo3DStorageOrder::T1Interleaved
            } else {
                Pseudo3DStorageOrder::PlaneQueued
            }
        );
        assert_eq!(planes.len(), n_planes);
        let norm0 = planes[0].data[0][0];
        for (pi_idx, plane) in planes.iter().enumerate() {
            let ratio = plane.data[0][0] / norm0;
            assert!(
                (ratio - decay_amps[pi_idx]).abs() < 1e-3,
                "plane {pi_idx} in {sub_name}: expected ratio {:.3}, got {:.3}",
                decay_amps[pi_idx],
                ratio
            );
        }
    }

    // Also verify `03c` `(PlaneQueued, inner_is_qf=true)` AND `03d` `(T1Interleaved, list_inner_to_quad=true, inner_is_qf=false)`!
    for (sub_name, a2_fnmode, a3_fnmode, is_pq) in [
        ("03c_pseudo3d_pq_inner_qf", 1, 4, true),
        ("03d_pseudo3d_il_list_inside_quad", 4, 1, false),
    ] {
        let pdir = tmp.join(sub_name);
        fs::create_dir_all(&pdir).unwrap();
        let acqus_p = make_bruker_acqus(
            "hsqct1etf3gpsi3d",
            2,
            Some(0),
            "",
            sfo_h,
            150.9108,
            sfo_n,
            "13C",
            "15N",
            174.0,
            54.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,
            8.1,
            0.0,
            0.0,
            0,
            1,
        );
        let td2 = if a2_fnmode == 1 { n_planes } else { n1_p3d * 2 };
        let td3 = if a3_fnmode == 1 { n_planes } else { n1_p3d * 2 };
        let a2_p = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, td2, a2_fnmode, 0, None);
        let a3_p = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, td3, a3_fnmode, 0, None);
        let pp_p = if is_pq {
            // Explicit outer `ivd` loop, even though `acqu2s` has `FnMODE=1`!
            "1 ze\n2 d1\n3 d1\n4 (p1 ph1)\n  vd\n  (p21 ph3):f3\n  d0\n  (p21 ph4):f3\n  go=4 ph31\n  d1 wr #0 if #0 zd ip3\n  lo to 4 times 2\n  d1 id0\n  lo to 3 times 16\n  d1 rd0 ivd\n  lo to 2 times 4\nexit\nph1=0\nph3=0\nph4=0\nph31=0 2\n"
        } else {
            // Explicit `ivd` INNERMOST (`lo to 4 times 4` inside `lo to 3 times 2` `ip3` inside `lo to 2 times 16` `id0`),
            // even though `acqu2s = States (4)` and `acqu3s = QF (1)`!
            "1 ze\n2 d1\n3 d1\n4 (p1 ph1)\n  vd\n  (p21 ph3):f3\n  d0\n  (p21 ph4):f3\n  go=4 ph31\n  d1 wr #0 if #0 zd ivd\n  lo to 4 times 4\n  d1 ip3\n  lo to 3 times 2\n  d1 id0\n  lo to 2 times 16\nexit\nph1=0\nph3=0\nph4=0\nph31=0 2\n"
        };
        fs::write(pdir.join("acqus"), &acqus_p).unwrap();
        fs::write(pdir.join("acqu2s"), &a2_p).unwrap();
        fs::write(pdir.join("acqu3s"), &a3_p).unwrap();
        fs::write(pdir.join("pulseprogram"), pp_p).unwrap();
        fs::write(pdir.join("vdlist"), "0.01\n0.25\n0.65\n1.25\n").unwrap();

        let make_trace2 = |amp: f64, k1: usize, is_sin: bool| -> Vec<[f64; 2]> {
            let t1 = (k1 as f64) / sw_n_hz;
            let m = amp
                * (-t1 / 0.04).exp()
                * if !is_sin {
                    (-2.0 * PI * nu_n * t1).cos()
                } else {
                    (-2.0 * PI * nu_n * t1).sin()
                };
            (0..n0_p3d)
                .map(|k0| {
                    let t0 = (k0 as f64) / sw_h_hz;
                    let e0 = (-t0 / 0.04).exp();
                    let w0 = 2.0 * PI * nu_h * t0;
                    [m * e0 * w0.cos(), m * e0 * w0.sin()]
                })
                .collect()
        };

        let mut ser_traces = Vec::with_capacity(n_planes * n1_p3d * 2);
        if is_pq {
            for &amp in &decay_amps {
                for k1 in 0..n1_p3d {
                    ser_traces.push(make_trace2(amp, k1, false));
                    ser_traces.push(make_trace2(amp, k1, true));
                }
            }
        } else {
            // `k1` outermost, `q in [cos, sin]` middle, `p` (`amp`) innermost!
            for k1 in 0..n1_p3d {
                for is_sin in [false, true] {
                    for &amp in &decay_amps {
                        ser_traces.push(make_trace2(amp, k1, is_sin));
                    }
                }
            }
        }
        write_bruker_ser_traces(&pdir.join("ser"), n0_p3d * 2, &ser_traces);

        let (planes, detected_order) = read_bruker_pseudo3d_planes_2d(&pdir).unwrap();
        assert_eq!(
            detected_order,
            if is_pq {
                Pseudo3DStorageOrder::PlaneQueued
            } else {
                Pseudo3DStorageOrder::T1Interleaved
            }
        );
        let norm0 = planes[0].data[0][0];
        for (pi_idx, plane) in planes.iter().enumerate() {
            let ratio = plane.data[0][0] / norm0;
            assert!(
                (ratio - decay_amps[pi_idx]).abs() < 1e-3,
                "plane {pi_idx} in {sub_name}: expected {:.3}, got {:.3}",
                decay_amps[pi_idx],
                ratio
            );
        }
    }

    // ------------------------------------------------------------------------
    // Case D: `AQSEQ = 312` deduced from `pulseprogram` + 1-based `nuslist` in `read_fid_3d`
    // (including manual `lo to` `id0`-inside-`id10` when `acqus` has default `##$AQSEQ= 0`!),
    // AND half-dwell `FCOR = 1.0` preserved across `ProcessingRecipe::from_fid_2d`!
    // ------------------------------------------------------------------------
    let dir_hd = tmp.join("04a_half_dwell_fcor_preserved");
    fs::create_dir_all(&dir_hd).unwrap();
    let dw_n = 1.0 / sw_n_hz;
    let acqus_hd = make_bruker_acqus(
        "hsqcetf3gpsi",
        1,
        Some(0),
        "",
        sfo_h,
        150.9108,
        sfo_n,
        "13C",
        "15N",
        174.0,
        54.0,
        0.25 * dw_n, // `2 * d0 = 0.5 * DW_N` -> half-dwell sampling!
        0.0,
        0.0,
        0.0,
        80.0,
        8.1,
        0.0,
        0.0,
        0,
        1,
    );
    fs::write(dir_hd.join("acqus"), &acqus_hd).unwrap();
    fs::write(
        dir_hd.join("acqu2s"),
        make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, n1 * 2, 6, 0, None),
    )
    .unwrap();
    fs::write(
        dir_hd.join("pulseprogram"),
        "define list<gradient> EA=<EA>\n1 ze\n2 d1\n  (p1 ph1)\n  (p21 ph3):f3\n  d0\n  (p2 ph2)\n  d0\n  p16:gp1*EA\n  (p22 ph4):f3\n  p16:gp2\n  go=2 ph31\n  d1 mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))\nexit\nph1=0\nph2=0\nph3=0\nph4=0\nph31=0 2\n",
    )
    .unwrap();
    write_bruker_ser_traces(&dir_hd.join("ser"), n0 * 2, &traces_manual);
    let fid_hd = read_fid_2d(&dir_hd).unwrap();
    assert_eq!(fid_hd.axis_y.stored_proc.fcor, Some(1.0));
    let rec_hd = ProcessingRecipe::from_fid_2d(&fid_hd);
    assert!(
        (rec_hd.dims[1].fcor - 1.0).abs() < 1e-9,
        "ProcessingRecipe::from_fid_2d must preserve half-dwell FCOR=1.0 during Auto phase, got {}",
        rec_hd.dims[1].fcor
    );
    assert!(
        (rec_man.dims[1].fcor - 0.5).abs() < 1e-9,
        "Zero-delay FID must keep FCOR=0.5, got {}",
        rec_man.dims[1].fcor
    );

    // Manual `lo to` `AQSEQ = 312` (`id0` nested inside `id10`) WITH explicit `##$AQSEQ= 0` in `acqus`!
    let dir_312_man = tmp.join("04b_manual_loop_aqseq312_with_header_zero");
    fs::create_dir_all(&dir_312_man).unwrap();
    let acqus_312_man = make_bruker_acqus(
        "hncagp3d",
        2,
        Some(0), // Default `##$AQSEQ= 0` in `acqus` while manual `lo to` executes `id0` inside `id10`!
        "",
        sfo_h,
        150.9290,
        sfo_n,
        "13C",
        "15N",
        174.0,
        54.0,
        0.0,
        0.0,
        0.0,
        0.0,
        80.0,
        8.1,
        0.0,
        0.0,
        0,
        1,
    );
    let a2_312_m = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, 8 * 4, 4, 2, Some(16)); // n1 = 8
    let a3_312_m = make_bruker_acqu_ind("13C", 150.9290, 150.9028, 28.0, 1, 4, 2, Some(32)); // n2 = 16
    fs::write(dir_312_man.join("acqus"), &acqus_312_man).unwrap();
    fs::write(dir_312_man.join("acqu2s"), &a2_312_m).unwrap();
    fs::write(dir_312_man.join("acqu3s"), &a3_312_m).unwrap();
    fs::write(
        dir_312_man.join("pulseprogram"),
        "; manual lo-to 3D with inner F1 (d0/id0) and outer F2 (d10/id10)\n1 ze\n2 d1\n3 d1\n4 (p1 ph1)\n  (p21 ph5):f3\n  d10\n  (p3 ph3):f2\n  d0\n  go=4 ph31\n  d1 wr #0 if #0 zd ip3\n  lo to 4 times 2\n  d1 id0\n  lo to 3 times l0\n  d1 rd0 ip5\n  lo to 2 times 2\n  d1 id10\n  lo to 2 times l10\nexit\nph1=0\nph3=0\nph5=0\nph31=0 2\n",
    )
    .unwrap();
    fs::write(
        dir_312_man.join("nuslist"),
        "1 1\n2 3\n5 4\n7 2\n10 6\n12 5\n15 7\n16 8\n",
    )
    .unwrap();
    let dummy_traces_m: Vec<Vec<[f64; 2]>> = (0..8 * 4).map(|_| vec![[1.0, 0.0]; 256]).collect();
    write_bruker_ser_traces(&dir_312_man.join("ser"), 512, &dummy_traces_m);
    let pi_312_man = interpret_experiment_dir(&dir_312_man).unwrap();
    assert_eq!(pi_312_man.aqseq, AcquisitionOrder::Aqseq312);
    let fid_312_man = read_fid_3d(&dir_312_man).expect("read_fid_3d manual lo-to AQSEQ=312");
    let mask_312_man = fid_312_man.nus_mask.expect("nus_mask");
    assert!(mask_312_man[15 * 8 + 7]);
    let inv_312_man = classify_nmr_workspace(&dir_312_man).unwrap();
    assert!(inv_312_man.experiments[0]
        .notes
        .iter()
        .any(|n| n.contains("AQSEQ=312")));
    let dir_312 = tmp.join("04_aqseq312_1based_nus_3d");
    fs::create_dir_all(&dir_312).unwrap();
    let acqus_312 = make_bruker_acqus(
        "hncogp3d", 2,
        None, // AQSEQ omitted in acqus -> must be deduced from `aqseq 312` in pulseprogram!
        "", sfo_h, 150.9290, sfo_n, "13C", "15N", 174.0, 54.0, 0.0, 0.0, 0.0, 0.0, 80.0, 8.1, 0.0,
        0.0, 0, 1,
    );
    let a2_312 = make_bruker_acqu_ind("15N", sfo_n, bf_n, sw_n_ppm, 8 * 4, 4, 2, Some(16)); // n1 = 8
    let a3_312 = make_bruker_acqu_ind("13C", 150.9290, 150.9028, 14.0, 1, 4, 2, Some(32)); // n2 = 16
    fs::write(dir_312.join("acqus"), &acqus_312).unwrap();
    fs::write(dir_312.join("acqu2s"), &a2_312).unwrap();
    fs::write(dir_312.join("acqu3s"), &a3_312).unwrap();
    fs::write(
        dir_312.join("pulseprogram"),
        ";hncogp3d\naqseq 312\n1 ze\n2 go=2 ph31\nexit\nph31=0 2\n",
    )
    .unwrap();
    // 8 1-based rows under AQSEQ=312: col0 in `1..=16` (F1/acqu3s), col1 in `1..=8` (F2/acqu2s)!
    fs::write(
        dir_312.join("nuslist"),
        "1 1\n2 3\n5 4\n7 2\n10 6\n12 5\n15 7\n16 8\n",
    )
    .unwrap();
    let dummy_traces_312: Vec<Vec<[f64; 2]>> = (0..8 * 4).map(|_| vec![[1.0, 0.0]; 256]).collect();
    write_bruker_ser_traces(&dir_312.join("ser"), 512, &dummy_traces_312);

    let fid_312 = read_fid_3d(&dir_312).expect("read_fid_3d AQSEQ=312 + 1-based nuslist");
    let mask_312 = fid_312.nus_mask.expect("nus_mask");
    // Row 7 `(16, 8)` 1-based -> `(15, 7)` 0-based `(F1, F2)` -> `(k1_f2 = 7, k2_f1 = 15)`!
    assert!(mask_312[15 * 8 + 7]);
    assert!(mask_312[0]);
    assert_eq!(mask_312.iter().filter(|&&b| b).count(), 8);

    // ------------------------------------------------------------------------
    // Case E: `classify_nmr_entries` on custom/renamed sequences & DOSY decoys!
    // ------------------------------------------------------------------------
    let custom_entries = vec![
        VirtualEntry::new(
            "bundle/10_custom_hncacb/acqus",
            make_bruker_acqus(
                "custom_tr_seq_v2",
                2,
                Some(0),
                "-DTROSY -DLABEL_CN -DCACB_TRANSFER",
                800.1300,
                201.1900,
                81.0760,
                "13C",
                "15N",
                174.0,
                43.0,
                3.0e-6,
                0.0,
                0.0,
                0.0,
                80.0,
                8.1,
                0.0,
                67.984,
                20,
                2000,
            )
            .into_bytes(),
        ),
        VirtualEntry::new(
            "bundle/10_custom_hncacb/acqu2s",
            make_bruker_acqu_ind("15N", 81.0760, 81.0660, 34.0, 64, 6, 0, None).into_bytes(),
        ),
        VirtualEntry::new(
            "bundle/10_custom_hncacb/acqu3s",
            make_bruker_acqu_ind("13C", 201.1900, 201.1810, 72.0, 128, 5, 0, None).into_bytes(),
        ),
        VirtualEntry::new(
            "bundle/10_custom_hncacb/pulseprogram",
            b";custom\n#ifdef CACB_TRANSFER\n\"d28=3.6m\"\n#endif\n1 ze\n2 d1\n(p1 ph1)\n(p21 ph1):f3\nd21\n(p3 ph3):f2\n#ifdef CACB_TRANSFER\nd28\n#endif\nd0\n(p3 ph4):f2\nd10\ngo=2 ph31\nexit\nph1=0\nph3=0 2\nph4=0\nph31=0 2\n"
                .to_vec(),
        ),
        VirtualEntry::header_only("bundle/10_custom_hncacb/ser", Vec::new(), 4096),
        // Plus a custom DOSY decoy `bpp_led_custom`: must NOT be selected by default!
        VirtualEntry::new(
            "bundle/20_dosy_decoy/acqus",
            make_bruker_acqus(
                "bpp_led_custom",
                1,
                Some(0),
                "",
                600.1328,
                150.9108,
                60.8179,
                "13C",
                "15N",
                174.0,
                54.0,
                3.0e-6,
                0.0,
                0.080,
                0.0,
                5.0,
                -5.0,
                0.0,
                67.984,
                20,
                2000,
            )
            .into_bytes(),
        ),
        VirtualEntry::new(
            "bundle/20_dosy_decoy/acqu2s",
            make_bruker_acqu_ind("1H", 600.1328, 600.1300, 10.0, 16, 1, 0, None).into_bytes(),
        ),
        VirtualEntry::new(
            "bundle/20_dosy_decoy/difflist",
            b"2.0\n15.0\n30.0\n50.0\n75.0\n95.0\n".to_vec(),
        ),
        VirtualEntry::header_only("bundle/20_dosy_decoy/ser", Vec::new(), 4096),
    ];
    let inv_custom = classify_nmr_entries(&custom_entries);
    assert_eq!(inv_custom.experiments.len(), 2);
    let exp_tr = inv_custom
        .experiments
        .iter()
        .find(|e| e.relative_path.contains("10_custom_hncacb"))
        .unwrap();
    assert_eq!(exp_tr.pulse_class, PulseSequenceClass::Hncacb);
    assert_eq!(exp_tr.dimensionality, ExperimentDimensionality::Dim3D);
    assert!(exp_tr.selected_by_default);

    let exp_dosy = inv_custom
        .experiments
        .iter()
        .find(|e| e.relative_path.contains("20_dosy_decoy"))
        .unwrap();
    assert_eq!(exp_dosy.pulse_class, PulseSequenceClass::Unknown);
    assert!(!exp_dosy.selected_by_default);
    assert!(exp_dosy.notes.iter().any(|n| n.contains("DOSY")));

    // Reject ambiguous/corrupt `nuslist` containing BOTH `0` and `n_cplx`!
    assert!(matches!(
        decode_nuslist_str("0\n4\n64\n", &[64], AcquisitionOrder::Dim2D),
        Err(PulprogError::InvalidNusSchedule(_))
    ));

    let _ = fs::remove_dir_all(&tmp);
}

// ============================================================================
// Test 4: Volumetric `{sign, fold}` HSQC Strip-Matching Verification (`Q1`)
// ============================================================================

struct SyntheticSpinSystem {
    h: f64,
    n: f64,
    ca: f64,
    cb: f64,
    co_prev: f64,
}

fn build_synthetic_28res_systems() -> Vec<SyntheticSpinSystem> {
    // 28 well-dispersed realistic backbone spin systems spanning BMRB chemical-shift priors
    // (intentionally avoiding symmetry about the 15N carrier 118.0 ppm or 13C carrier 44.0 / 175.5 ppm!)
    let raw: [(f64, f64, f64, f64, f64); 28] = [
        (8.82, 126.4, 54.8, 38.2, 174.3), // Phe/Tyr
        (8.61, 122.7, 61.2, 32.4, 176.9), // Val
        (8.44, 119.3, 52.3, 18.6, 177.4), // Ala
        (8.19, 108.9, 45.2, 0.0, 173.2),  // Gly (no CB)
        (7.92, 115.6, 58.1, 63.4, 174.8), // Ser (downfield CB)
        (9.14, 128.8, 60.8, 69.1, 173.9), // Thr (downfield CB)
        (7.68, 114.2, 55.4, 29.8, 176.1), // Glu
        (8.35, 121.1, 56.9, 41.7, 175.6), // Leu
        (8.97, 125.2, 59.8, 37.9, 174.1), // Ile
        (7.51, 112.4, 53.6, 40.5, 172.9), // Asp
        (8.08, 117.2, 56.2, 28.9, 176.5), // Gln
        (8.73, 124.1, 55.9, 32.8, 175.2), // Lys
        (7.84, 109.8, 44.8, 0.0, 172.6),  // Gly
        (8.52, 120.4, 52.7, 19.3, 177.8), // Ala
        (9.03, 127.6, 55.1, 39.4, 173.7), // Tyr
        (7.76, 116.5, 57.8, 63.9, 175.0), // Ser
        (8.29, 123.3, 56.4, 30.6, 176.8), // Arg
        (8.66, 118.5, 62.1, 31.8, 177.1), // Val
        (7.59, 113.7, 54.2, 38.8, 174.5), // Asn
        (8.41, 125.9, 55.3, 29.1, 175.9), // Met
        (8.88, 129.6, 60.4, 38.4, 173.5), // Ile
        (8.12, 115.1, 53.1, 18.1, 176.3), // Ala
        (7.97, 121.8, 57.2, 42.1, 175.4), // Leu
        (8.58, 124.7, 56.0, 31.4, 176.6), // Lys
        (7.72, 111.3, 45.6, 0.0, 173.0),  // Gly
        (8.23, 119.9, 61.5, 68.4, 174.6), // Thr
        (8.79, 127.1, 55.7, 37.6, 174.9), // Phe
        (8.04, 116.9, 56.8, 28.4, 176.2), // Glu
    ];
    raw.into_iter()
        .map(|(h, n, ca, cb, co_prev)| SyntheticSpinSystem {
            h,
            n,
            ca,
            cb,
            co_prev,
        })
        .collect()
}

fn deterministic_noise(seed: u64, idx: usize) -> f32 {
    let mut x = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let u = ((x & 0xFFFF_FFFF) as f64) / (u32::MAX as f64) - 0.5;
    (u * 2.0) as f32
}

fn add_gaussian_2d(
    spec: &mut Spectrum2D,
    h_ppm: f64,
    n_ppm: f64,
    amp: f64,
    sig_h: f64,
    sig_n: f64,
) {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    for iy in 0..ny {
        let dn = (spec.axis_y.index_to_ppm(iy as f64) - n_ppm) / sig_n;
        if dn.abs() > 4.0 {
            continue;
        }
        let wn = (-0.5 * dn * dn).exp();
        for ix in 0..nx {
            let dh = (spec.axis_x.index_to_ppm(ix as f64) - h_ppm) / sig_h;
            if dh.abs() > 4.0 {
                continue;
            }
            spec.data[iy * nx + ix] += (amp * wn * (-0.5 * dh * dh).exp()) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_gaussian_3d(
    spec: &mut Spectrum3D,
    h_ppm: f64,
    n_ppm: f64,
    c_ppm: f64,
    amp: f64,
    sig_h: f64,
    sig_n: f64,
    sig_c: f64,
) {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let nz = spec.axis_z.size;
    for iz in 0..nz {
        let dc = (spec.axis_z.index_to_ppm(iz as f64) - c_ppm) / sig_c;
        if dc.abs() > 4.0 {
            continue;
        }
        let wc = (-0.5 * dc * dc).exp();
        for iy in 0..ny {
            let dn = (spec.axis_y.index_to_ppm(iy as f64) - n_ppm) / sig_n;
            if dn.abs() > 4.0 {
                continue;
            }
            let wnc = wc * (-0.5 * dn * dn).exp();
            let row_off = (iz * ny + iy) * nx;
            for ix in 0..nx {
                let dh = (spec.axis_x.index_to_ppm(ix as f64) - h_ppm) / sig_h;
                if dh.abs() > 4.0 {
                    continue;
                }
                spec.data[row_off + ix] += (amp * wnc * (-0.5 * dh * dh).exp()) as f32;
            }
        }
    }
}

/// Measure true 3D strip recovery fraction (`0.0..=1.0`) at exact spin-system `(H, N, C)` coordinates!
fn measure_hncacb_strip_recovery(spec: &Spectrum3D, systems: &[SyntheticSpinSystem]) -> f64 {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let nz = spec.axis_z.size;
    let mut recovered = 0usize;
    for s in systems {
        let ix = spec
            .axis_x
            .ppm_to_index(s.h)
            .round()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let iy = spec
            .axis_y
            .ppm_to_index(s.n)
            .round()
            .clamp(0.0, (ny - 1) as f64) as usize;
        let iz_ca = spec
            .axis_z
            .ppm_to_index(s.ca)
            .round()
            .clamp(0.0, (nz - 1) as f64) as usize;
        let v_ca = spec.data[(iz_ca * ny + iy) * nx + ix];
        let cb_ok = if s.cb > 10.0 {
            let iz_cb = spec
                .axis_z
                .ppm_to_index(s.cb)
                .round()
                .clamp(0.0, (nz - 1) as f64) as usize;
            spec.data[(iz_cb * ny + iy) * nx + ix] < -25.0
        } else {
            true
        };
        if v_ca > 35.0 && cb_ok {
            recovered += 1;
        }
    }
    recovered as f64 / systems.len() as f64
}

fn measure_hnco_strip_recovery(spec: &Spectrum3D, systems: &[SyntheticSpinSystem]) -> f64 {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let nz = spec.axis_z.size;
    let mut recovered = 0usize;
    for s in systems {
        let ix = spec
            .axis_x
            .ppm_to_index(s.h)
            .round()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let iy = spec
            .axis_y
            .ppm_to_index(s.n)
            .round()
            .clamp(0.0, (ny - 1) as f64) as usize;
        let iz_co = spec
            .axis_z
            .ppm_to_index(s.co_prev)
            .round()
            .clamp(0.0, (nz - 1) as f64) as usize;
        if spec.data[(iz_co * ny + iy) * nx + ix] > 35.0 {
            recovered += 1;
        }
    }
    recovered as f64 / systems.len() as f64
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_volumetric_empirical_sign_fold_hsqc_verifier_rescues_hncacb_and_hnco() {
    let systems = build_synthetic_28res_systems();

    // 2D 15N-HSQC (`X: 10.0..6.5 ppm`, `Y: 134.0..102.0 ppm`, carrier = `118.0 ppm`)
    let ax_h = AxisPpm {
        label: "1H".to_string(),
        size: 96,
        first_ppm: 10.0,
        step_ppm: -3.5 / 96.0,
        obs_mhz: 600.13,
    };
    let ax_n = AxisPpm {
        label: "15N".to_string(),
        size: 64,
        first_ppm: 134.0,
        step_ppm: -32.0 / 64.0,
        obs_mhz: 60.81,
    };
    let mut hsqc = Spectrum2D {
        name: "syn_hsqc".to_string(),
        axis_x: ax_h.clone(),
        axis_y: ax_n.clone(),
        data: (0..96 * 64)
            .map(|i| deterministic_noise(11, i) * 1.2)
            .collect(),
    };
    for s in &systems {
        add_gaussian_2d(&mut hsqc, s.h, s.n, 120.0, 0.032, 0.32);
    }

    // Canonical 3D `HNCACB` (`Z: 78.0..10.0 ppm`, carrier = `44.0 ppm`, SW = `68.0 ppm`)
    // Note: inverting around `44.0 ppm` maps `CA (56 ppm) -> 32 ppm` and `Ala CB (19 ppm) -> 69 ppm`,
    // and half-SW-shifting by `34 ppm` maps `CA (56 ppm) -> 22 ppm` and `CB (32 ppm) -> 66 ppm`!
    let ax_c_ali = AxisPpm {
        label: "13C".to_string(),
        size: 96,
        first_ppm: 78.0,
        step_ppm: -68.0 / 96.0,
        obs_mhz: 150.90,
    };
    let mut hncacb_canon = Spectrum3D {
        name: "syn_hncacb".to_string(),
        exp: "HNCACB".to_string(),
        axis_x: ax_h.clone(),
        axis_y: ax_n.clone(),
        axis_z: ax_c_ali,
        data: (0..96 * 64 * 96)
            .map(|i| deterministic_noise(22, i) * 1.5)
            .collect(),
    };
    for (idx, s) in systems.iter().enumerate() {
        add_gaussian_3d(&mut hncacb_canon, s.h, s.n, s.ca, 95.0, 0.032, 0.32, 0.55);
        if s.cb > 10.0 {
            add_gaussian_3d(&mut hncacb_canon, s.h, s.n, s.cb, -78.0, 0.032, 0.32, 0.55);
        }
        // Sequential `CA(i-1)` / `CB(i-1)` weaker cross-peaks
        let prev = &systems[(idx + systems.len() - 1) % systems.len()];
        add_gaussian_3d(
            &mut hncacb_canon,
            s.h,
            s.n,
            prev.ca,
            38.0,
            0.032,
            0.32,
            0.55,
        );
        if prev.cb > 10.0 {
            add_gaussian_3d(
                &mut hncacb_canon,
                s.h,
                s.n,
                prev.cb,
                -30.0,
                0.032,
                0.32,
                0.55,
            );
        }
    }

    // Canonical 3D `HNCO` (`Z: 181.5..169.5 ppm`, carrier = `175.5 ppm`, SW = `12.0 ppm`)
    let ax_c_co = AxisPpm {
        label: "13C".to_string(),
        size: 64,
        first_ppm: 181.5,
        step_ppm: -12.0 / 64.0,
        obs_mhz: 150.93,
    };
    let mut hnco_canon = Spectrum3D {
        name: "syn_hnco".to_string(),
        exp: "HNCO".to_string(),
        axis_x: ax_h,
        axis_y: ax_n,
        axis_z: ax_c_co,
        data: (0..96 * 64 * 64)
            .map(|i| deterministic_noise(33, i) * 1.5)
            .collect(),
    };
    for s in &systems {
        add_gaussian_3d(
            &mut hnco_canon,
            s.h,
            s.n,
            s.co_prev,
            110.0,
            0.032,
            0.32,
            0.22,
        );
    }

    // 1. Verify clean canonical cubes are NEVER modified!
    assert_eq!(measure_hncacb_strip_recovery(&hncacb_canon, &systems), 1.0);
    assert_eq!(measure_hnco_strip_recovery(&hnco_canon, &systems), 1.0);

    let mut hncacb_clean_check = hncacb_canon.clone();
    let ver_clean_cb = verify_3d_axis_hypotheses_against_hsqc(&mut hncacb_clean_check, &hsqc, None);
    assert_eq!(ver_clean_cb.selected, NamedAxisHypothesis::Canonical);
    assert!(!ver_clean_cb.applied);
    assert_eq!(hncacb_clean_check.data, hncacb_canon.data);

    let mut hnco_clean_check = hnco_canon.clone();
    let ver_clean_co = verify_3d_axis_hypotheses_against_hsqc(&mut hnco_clean_check, &hsqc, None);
    assert_eq!(ver_clean_co.selected, NamedAxisHypothesis::Canonical);
    assert!(!ver_clean_co.applied);
    assert_eq!(hnco_clean_check.data, hnco_canon.data);

    // 2. Corrupt `HNCACB` across all indirect `{sign, fold}` failure modes and rescue back to 100%!
    for (label, op_n, op_c, flip_sign) in [
        (
            "HNCACB SignFlipN",
            AxisTransformOp::SignFlip,
            AxisTransformOp::Canonical,
            false,
        ),
        (
            "HNCACB HalfSwShiftN",
            AxisTransformOp::HalfSwShift,
            AxisTransformOp::Canonical,
            false,
        ),
        (
            "HNCACB MirrorFoldN",
            AxisTransformOp::MirrorFold,
            AxisTransformOp::Canonical,
            false,
        ),
        (
            "HNCACB SignFlipC",
            AxisTransformOp::Canonical,
            AxisTransformOp::SignFlip,
            false,
        ),
        (
            "HNCACB HalfSwShiftC",
            AxisTransformOp::Canonical,
            AxisTransformOp::HalfSwShift,
            false,
        ),
        (
            "HNCACB Joint(SignFlipN, HalfSwShiftC) + inverted sign",
            AxisTransformOp::SignFlip,
            AxisTransformOp::HalfSwShift,
            true,
        ),
    ] {
        let mut corrupted = hncacb_canon.clone();
        // Every element of the Klein four-group is its own inverse (`op * op == Identity`)!
        apply_indirect_axis_transform_3d(&mut corrupted, op_n, op_c, None, false);
        if flip_sign {
            for v in &mut corrupted.data {
                *v = -*v;
            }
        }
        let pre_rec = measure_hncacb_strip_recovery(&corrupted, &systems);
        assert!(
            pre_rec < 0.15,
            "{label}: expected corrupted strip recovery < 15%, got {:.1}%",
            pre_rec * 100.0
        );

        let ver = verify_3d_axis_hypotheses_against_hsqc(&mut corrupted, &hsqc, None);
        assert!(ver.applied, "{label}: expected verifier to apply rescue");
        assert_eq!(
            ver.selected,
            NamedAxisHypothesis::from_ops(op_n, op_c),
            "{label}: wrong hypothesis selected"
        );
        assert!(
            ver.ln_bayes_factor_vs_canonical >= MIN_AXIS_RESCUE_LN_BF,
            "{label}: ln_BF={:.2}",
            ver.ln_bayes_factor_vs_canonical
        );

        let post_rec = measure_hncacb_strip_recovery(&corrupted, &systems);
        assert!(
            (post_rec - 1.0).abs() < 1e-9,
            "{label}: expected 100% strip recovery after rescue, got {:.1}%",
            post_rec * 100.0
        );
        println!(
            "PULPROG_METRIC_JSON: {{\"kind\":\"q1_rescue\",\"case\":{:?},\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2},\"pre_recovery_pct\":{:.2},\"post_recovery_pct\":{:.2}}}",
            label,
            ver.selected,
            ver.applied,
            ver.polarity_flipped,
            ver.ln_bayes_factor_vs_canonical,
            pre_rec * 100.0,
            post_rec * 100.0
        );
    }

    // 3. Corrupt `HNCO` (`SignFlipN`, `HalfSwShiftN`, `MirrorFoldN`, `HalfSwShiftC`) + test `register_3d_hn_to_hsqc_2d` integration!
    for (label, op_n, op_c) in [
        (
            "HNCO SignFlipN",
            AxisTransformOp::SignFlip,
            AxisTransformOp::Canonical,
        ),
        (
            "HNCO HalfSwShiftN",
            AxisTransformOp::HalfSwShift,
            AxisTransformOp::Canonical,
        ),
        (
            "HNCO MirrorFoldN",
            AxisTransformOp::MirrorFold,
            AxisTransformOp::Canonical,
        ),
        (
            "HNCO HalfSwShiftC",
            AxisTransformOp::Canonical,
            AxisTransformOp::HalfSwShift,
        ),
    ] {
        let mut corrupted = hnco_canon.clone();
        apply_indirect_axis_transform_3d(&mut corrupted, op_n, op_c, None, false);
        let pre_rec = measure_hnco_strip_recovery(&corrupted, &systems);
        assert!(
            pre_rec < 0.15,
            "{label}: expected corrupted strip recovery < 15%, got {:.1}%",
            pre_rec * 100.0
        );

        // Run through `register_3d_hn_to_hsqc_2d` directly to prove seamless pipeline integration!
        let ver_eval = evaluate_3d_axis_hypotheses_against_hsqc(&corrupted, &hsqc, None);
        let _shift = register_3d_hn_to_hsqc_2d(&mut corrupted, &hsqc, 0.15, 0.80);
        let post_rec = measure_hnco_strip_recovery(&corrupted, &systems);
        assert!(
            (post_rec - 1.0).abs() < 1e-9,
            "{label}: expected 100% strip recovery via register_3d_hn_to_hsqc_2d, got {:.1}%",
            post_rec * 100.0
        );
        println!(
            "PULPROG_METRIC_JSON: {{\"kind\":\"q1_rescue\",\"case\":{:?},\"selected\":\"{:?}\",\"applied\":true,\"polarity_flipped\":false,\"ln_bf\":{:.2},\"pre_recovery_pct\":{:.2},\"post_recovery_pct\":{:.2}}}",
            label,
            ver_eval.selected,
            ver_eval.ln_bayes_factor_vs_canonical,
            pre_rec * 100.0,
            post_rec * 100.0
        );
    }

    // 4. Regression guards against false `SignFlipC` on physically valid edge regimes (Verifier P1 #1):
    //    (a) `CA`-bleached paramagnetic `HNCACB` and `CBCACONH` (carrier `43.0 ppm`, `Z: 77.0..9.0 ppm`,
    //        where `CA ~ 56 ppm` is completely bleached and ONLY `CB ~ 28..36 ppm` survives — reflecting
    //        `30 -> 56 ppm` would map `CB` into the narrower `CA` Gaussian and flip negative `CB` to fake `+CA`!).
    let ax_c_para = AxisPpm {
        label: "13C".to_string(),
        size: 96,
        first_ppm: 77.0,
        step_ppm: -68.0 / 96.0,
        obs_mhz: 150.90,
    };
    for (exp_name, cb_amp) in [("HNCACB", -85.0_f64), ("CBCACONH", 85.0_f64)] {
        let mut para_cube = Spectrum3D {
            name: format!("para_{exp_name}"),
            exp: exp_name.to_string(),
            axis_x: hsqc.axis_x.clone(),
            axis_y: hsqc.axis_y.clone(),
            axis_z: ax_c_para.clone(),
            data: (0..96 * 64 * 96)
                .map(|i| deterministic_noise(77, i) * 1.4)
                .collect(),
        };
        for (idx, s) in systems.iter().enumerate() {
            // Paramagnetic bleaching of `CA`: ONLY `CB` (`28..35 ppm` cluster + occasional Ser/Thr) survives!
            if s.cb > 10.0 {
                let cb_para = if s.cb > 60.0 {
                    s.cb
                } else {
                    28.0 + ((idx % 7) as f64) * 0.95
                };
                add_gaussian_3d(&mut para_cube, s.h, s.n, cb_para, cb_amp, 0.032, 0.32, 0.55);
            }
        }
        let orig_para = para_cube.data.clone();
        let ver_para = verify_3d_axis_hypotheses_against_hsqc(&mut para_cube, &hsqc, None);
        assert_eq!(
            ver_para.selected,
            NamedAxisHypothesis::Canonical,
            "CA-less paramagnetic {exp_name} must NEVER false-SignFlip CB -> CA!"
        );
        assert!(!ver_para.applied);
        assert!(!ver_para.polarity_flipped);
        assert_eq!(para_cube.data, orig_para);
        println!(
            "PULPROG_METRIC_JSON: {{\"kind\":\"q1_guard\",\"case\":\"CA-less paramagnetic {}\",\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2}}}",
            exp_name, ver_para.selected, ver_para.applied, ver_para.polarity_flipped, ver_para.ln_bayes_factor_vs_canonical
        );

        // And if `15N` is `SignFlip`ped on a `CA`-less paramagnetic `HNCACB`, rescuing `SignFlipN`
        // must keep `op_c == Canonical` AND preserve negative `CB` polarity (`!polarity_flipped`)!
        if exp_name == "HNCACB" {
            let mut para_n_flip = para_cube.clone();
            apply_indirect_axis_transform_3d(
                &mut para_n_flip,
                AxisTransformOp::SignFlip,
                AxisTransformOp::Canonical,
                None,
                false,
            );
            let ver_n = verify_3d_axis_hypotheses_against_hsqc(&mut para_n_flip, &hsqc, None);
            assert_eq!(ver_n.selected, NamedAxisHypothesis::SignFlipN);
            assert!(ver_n.applied);
            assert!(
                !ver_n.polarity_flipped,
                "rescuing SignFlipN on CA-less HNCACB must not negate negative CB peaks!"
            );
            assert_eq!(para_n_flip.data, orig_para);
            println!(
                "PULPROG_METRIC_JSON: {{\"kind\":\"q1_guard\",\"case\":\"CA-less paramagnetic HNCACB + SignFlipN\",\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2}}}",
                ver_n.selected, ver_n.applied, ver_n.polarity_flipped, ver_n.ln_bayes_factor_vs_canonical
            );
        }
    }

    //    (b) β-sheet / Pro-rich / PCS `HNCO` (`O2P = 174.0 ppm, SW = 16.0 ppm`, `Z: 182.0..166.0 ppm`)
    //        with `48` clean roots whose `<13CO> ≈ 172.0 ppm` (`170.4..173.5 ppm`, 100% inside `166.5..184.5 ppm`):
    //        must NEVER falsely `SignFlip` `172 -> 176 ppm`!
    let mut hsqc_48 = Spectrum2D {
        name: "syn_hsqc_48".to_string(),
        axis_x: hsqc.axis_x.clone(),
        axis_y: hsqc.axis_y.clone(),
        data: (0..96 * 64)
            .map(|i| deterministic_noise(91, i) * 1.1)
            .collect(),
    };
    let ax_co_174 = AxisPpm {
        label: "13C".to_string(),
        size: 64,
        first_ppm: 182.0,
        step_ppm: -16.0 / 64.0,
        obs_mhz: 150.93,
    };
    let mut hnco_beta = Spectrum3D {
        name: "beta_pcs_hnco".to_string(),
        exp: "HNCO".to_string(),
        axis_x: hsqc.axis_x.clone(),
        axis_y: hsqc.axis_y.clone(),
        axis_z: ax_co_174,
        data: (0..96 * 64 * 64)
            .map(|i| deterministic_noise(92, i) * 1.3)
            .collect(),
    };
    for k in 0..48 {
        let h_k = 6.95 + ((k % 12) as f64) * 0.23 + ((k / 12) as f64) * 0.04;
        let n_k = 106.5 + ((k / 12) as f64) * 5.8 + ((k % 12) as f64) * 0.55;
        let co_k = 170.5 + ((k % 9) as f64) * 0.36; // mean `~171.94 ppm`!
        add_gaussian_2d(&mut hsqc_48, h_k, n_k, 125.0, 0.030, 0.30);
        add_gaussian_3d(&mut hnco_beta, h_k, n_k, co_k, 115.0, 0.030, 0.30, 0.22);
    }
    let orig_beta = hnco_beta.data.clone();
    let ver_beta = verify_3d_axis_hypotheses_against_hsqc(&mut hnco_beta, &hsqc_48, None);
    assert_eq!(
        ver_beta.selected,
        NamedAxisHypothesis::Canonical,
        "Clean beta-sheet/PCS HNCO (<13CO> ~ 172 ppm) must NEVER false-SignFlip!"
    );
    assert!(!ver_beta.applied);
    assert_eq!(hnco_beta.data, orig_beta);
    println!(
        "PULPROG_METRIC_JSON: {{\"kind\":\"q1_guard\",\"case\":\"48-root beta-sheet/PCS HNCO (<13CO>~172 ppm)\",\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2}}}",
        ver_beta.selected, ver_beta.applied, ver_beta.polarity_flipped, ver_beta.ln_bayes_factor_vs_canonical
    );

    //    (c) TMS-referenced narrow-SW `HNCO` (`O2P = 177.0 ppm, SW = 11.7 ppm`, `Z: 182.85..171.15 ppm`)
    //        where `-2.75 ppm` TMS referencing shifts `<13CO>` to `~173.5 ppm` with ~15% minor fold-in
    //        at `182.2 ppm` while `>40%` of roots remain in the central corridor: must NEVER false-`MirrorFoldC`!
    let ax_co_tms = AxisPpm {
        label: "13C".to_string(),
        size: 64,
        first_ppm: 182.85,
        step_ppm: -11.70 / 64.0,
        obs_mhz: 213.76,
    };
    let mut hnco_tms = Spectrum3D {
        name: "tms_narrow_hnco".to_string(),
        exp: "HNCO".to_string(),
        axis_x: hsqc.axis_x.clone(),
        axis_y: hsqc.axis_y.clone(),
        axis_z: ax_co_tms,
        data: (0..96 * 64 * 64)
            .map(|i| deterministic_noise(93, i) * 1.3)
            .collect(),
    };
    for k in 0..48 {
        let h_k = 6.95 + ((k % 12) as f64) * 0.23 + ((k / 12) as f64) * 0.04;
        let n_k = 106.5 + ((k / 12) as f64) * 5.8 + ((k % 12) as f64) * 0.55;
        // Beta-rich TMS-shifted `<13CO> ~ 172.8 ppm`: `2/16` (`12.5%`) alias `< 171.15` to `182.1..182.7 ppm`,
        // `7/16` (`43.8%`) sit near the lower edge `171.3..172.7 ppm`, and only `3/16` (`18.8%`) reach `174.5..176.2 ppm`.
        let bin = k % 16;
        let raw_co = 170.35 + (bin as f64) * 0.38;
        let co_k = if raw_co < 171.15 {
            raw_co + 11.70
        } else {
            raw_co
        };
        add_gaussian_3d(&mut hnco_tms, h_k, n_k, co_k, 115.0, 0.030, 0.30, 0.20);
    }
    let orig_tms = hnco_tms.data.clone();
    let ver_tms = verify_3d_axis_hypotheses_against_hsqc(&mut hnco_tms, &hsqc_48, None);
    assert_eq!(
        ver_tms.selected,
        NamedAxisHypothesis::Canonical,
        "TMS-referenced narrow-SW HNCO (<13CO> ~ 172.8 ppm) must NEVER false-MirrorFoldC or false-HalfSwShiftC!"
    );
    assert!(!ver_tms.applied);
    assert_eq!(hnco_tms.data, orig_tms);
    println!(
        "PULPROG_METRIC_JSON: {{\"kind\":\"q1_guard\",\"case\":\"48-root TMS narrow-SW HNCO (<13CO>~173.5 ppm)\",\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2}}}",
        ver_tms.selected, ver_tms.applied, ver_tms.polarity_flipped, ver_tms.ln_bayes_factor_vs_canonical
    );

    //    (d) Canonical all-positive `CBCACONH` (`O2P = 39.0 ppm, SW = 75.0 ppm`, `Z: 76.5..1.5 ppm`)
    //        where `CH2`/`CH3` `CB` INEPT cross-peaks (`20..25 ppm`, amplitude `135.0`) are brighter
    //        than `CH` `CA` (`53..59 ppm`, amplitude `75.0`): reflecting around `39.0 ppm` (`78 - c`)
    //        would swap `CB (22 ppm) <-> CA (56 ppm)` and must NEVER false-`SignFlipC`!
    let ax_cbca_39 = AxisPpm {
        label: "13C".to_string(),
        size: 96,
        first_ppm: 76.5,
        step_ppm: -75.0 / 96.0,
        obs_mhz: 125.71,
    };
    let mut cbca_39 = Spectrum3D {
        name: "canon_cbca_39ppm".to_string(),
        exp: "CBCACONH".to_string(),
        axis_x: hsqc.axis_x.clone(),
        axis_y: hsqc.axis_y.clone(),
        axis_z: ax_cbca_39,
        data: (0..96 * 64 * 96)
            .map(|i| deterministic_noise(94, i) * 1.3)
            .collect(),
    };
    for k in 0..48 {
        let h_k = 6.95 + ((k % 12) as f64) * 0.23 + ((k / 12) as f64) * 0.04;
        let n_k = 106.5 + ((k / 12) as f64) * 5.8 + ((k % 12) as f64) * 0.55;
        let ca_k = 53.5 + ((k % 7) as f64) * 0.85;
        let cb_k = if k % 9 == 0 {
            65.5 // Ser/Thr CB
        } else {
            20.5 + ((k % 6) as f64) * 0.85 // bright CH3/CH2 CB at 20.5..24.75 ppm (`78 - cb = 53.25..57.5 ppm`!)
        };
        add_gaussian_3d(&mut cbca_39, h_k, n_k, ca_k, 75.0, 0.030, 0.30, 0.55);
        add_gaussian_3d(&mut cbca_39, h_k, n_k, cb_k, 135.0, 0.030, 0.30, 0.55);
    }
    let orig_cbca_39 = cbca_39.data.clone();
    let ver_cbca_39 = verify_3d_axis_hypotheses_against_hsqc(&mut cbca_39, &hsqc_48, None);
    assert_eq!(
        ver_cbca_39.selected,
        NamedAxisHypothesis::Canonical,
        "Canonical CBCACONH (O2P=39 ppm, bright CH3/CH2 CB) must NEVER false-SignFlipC!"
    );
    assert!(!ver_cbca_39.applied);
    assert_eq!(cbca_39.data, orig_cbca_39);
    println!(
        "PULPROG_METRIC_JSON: {{\"kind\":\"q1_guard\",\"case\":\"48-root canonical CBCACONH (O2P=39 ppm, bright CB)\",\"selected\":\"{:?}\",\"applied\":{},\"polarity_flipped\":{},\"ln_bf\":{:.2}}}",
        ver_cbca_39.selected, ver_cbca_39.applied, ver_cbca_39.polarity_flipped, ver_cbca_39.ln_bayes_factor_vs_canonical
    );
}

// ============================================================================
// Test 5: Conditional Real `dev` Dataset Smoke Check (Strictly `dev` ONLY!)
// ============================================================================

#[test]
fn test_conditional_real_dev_cisd3_pulprog_and_canonical_non_corruption() {
    // STRICT ANTI-LEAKAGE: inspect ONLY `CISD3` (`BMRB 51439`, `split=dev` in `real_manifest.tsv`)!
    let Some(real_root) = std::env::var_os("NMR_REAL_DATA_DIR") else {
        return;
    };
    let cisd3_dir = Path::new(&real_root).join("private/bmrb51439_cisd3");
    if !cisd3_dir.is_dir() {
        return;
    }

    let hsqc_cand = cisd3_dir.join("CISD3_15N_HSQC_diamagnetic");
    let hsqc_dir = if hsqc_cand.is_dir() {
        hsqc_cand
    } else {
        cisd3_dir.join("CISD3_N15HSQC")
    };
    let hnco_dir = cisd3_dir.join("CISD3_HNCO");
    let cbca_dir = cisd3_dir.join("CISD3_CBCACONH");
    if !hsqc_dir.is_dir() || !hnco_dir.is_dir() || !cbca_dir.is_dir() {
        return;
    }

    let pi_hsqc = interpret_experiment_dir(&hsqc_dir).expect("interpret CISD3 HSQC");
    assert_eq!(
        pi_hsqc.transfer_graph.deduced_family,
        PulprogSequenceFamily::N15Hsqc
    );
    let pi_cbca = interpret_experiment_dir(&cbca_dir).expect("interpret CISD3 CBCACONH");
    assert_eq!(
        pi_cbca.transfer_graph.deduced_family,
        PulprogSequenceFamily::Cbcaconh
    );
    assert!(pi_cbca.nus_schedule.is_some());

    let fid_hsqc = read_fid_2d(&hsqc_dir).unwrap();
    let hsqc_spec = process_fid_2d(&fid_hsqc, &ProcessingRecipe::from_fid_2d(&fid_hsqc)).unwrap();
    let fid_hnco = read_fid_3d(&hnco_dir).unwrap();
    let hnco_spec =
        process_fid_3d(&fid_hnco, "HNCO", &ProcessingRecipe::from_fid_3d(&fid_hnco)).unwrap();

    // Canonical real `dev` spectrum MUST remain untouched (`Canonical`, `applied == false`)!
    let eval_clean = evaluate_3d_axis_hypotheses_against_hsqc(&hnco_spec, &hsqc_spec, Some("HNCO"));
    assert_eq!(eval_clean.selected, NamedAxisHypothesis::Canonical);
    assert!(!eval_clean.applied);

    // Artificially corrupting `CISD3` `HNCO` with `HalfSwShiftN` MUST be rescued back to `HalfSwShiftN`!
    let mut hnco_shifted = hnco_spec.clone();
    apply_indirect_axis_transform_3d(
        &mut hnco_shifted,
        AxisTransformOp::HalfSwShift,
        AxisTransformOp::Canonical,
        Some("HNCO"),
        false,
    );
    let ver_rescued =
        verify_3d_axis_hypotheses_against_hsqc(&mut hnco_shifted, &hsqc_spec, Some("HNCO"));
    assert_eq!(ver_rescued.selected, NamedAxisHypothesis::HalfSwShiftN);
    assert!(ver_rescued.applied);
    assert_eq!(hnco_shifted.data, hnco_spec.data);
    println!(
        "PULPROG_METRIC_JSON: {{\"kind\":\"real_cisd3\",\"canonical_selected\":\"{:?}\",\"canonical_applied\":{},\"corrupted_selected\":\"{:?}\",\"corrupted_applied\":{},\"corrupted_ln_bf\":{:.2},\"canonical_match_frac\":{:.4},\"rescued_match_frac\":{:.4}}}",
        eval_clean.selected,
        eval_clean.applied,
        ver_rescued.selected,
        ver_rescued.applied,
        ver_rescued.ln_bayes_factor_vs_canonical,
        ver_rescued.canonical_matched_fraction,
        ver_rescued.selected_matched_fraction
    );
}

// ============================================================================
// Test 6 (`13.3:P1-1`): Axis-Rescue Audit Record, Recipe/NEF/Report Provenance,
// `stderr` Warning Format, and `--no-axis-rescue` Opt-Out
// ============================================================================

#[test]
fn test_p1_1_axis_rescue_audit_record_recipe_nef_report_and_opt_out() {
    use nmr_core::nmrstar::Reference;
    use nmr_core::report::{generate_quality_report_html, ReportInput};
    use nmr_core::session::{read_session_nef_str, write_session_nef_str, WorkbenchSessionState};
    use std::collections::BTreeMap;

    let ax_h = AxisPpm {
        label: "1H".to_string(),
        size: 96,
        first_ppm: 10.5,
        step_ppm: -4.5 / 96.0,
        obs_mhz: 600.13,
    };
    let ax_n = AxisPpm {
        label: "15N".to_string(),
        size: 64,
        first_ppm: 134.0,
        step_ppm: -32.0 / 64.0,
        obs_mhz: 60.81,
    };
    let ax_co = AxisPpm {
        label: "13C".to_string(),
        size: 64,
        first_ppm: 184.0,
        step_ppm: -16.0 / 64.0,
        obs_mhz: 150.90,
    };

    let mut hsqc = Spectrum2D {
        name: "p1_1_hsqc".to_string(),
        axis_x: ax_h.clone(),
        axis_y: ax_n.clone(),
        data: (0..96 * 64)
            .map(|i| deterministic_noise(301, i) * 1.1)
            .collect(),
    };
    let mut hnco_clean = Spectrum3D {
        name: "p1_1_hnco".to_string(),
        exp: "HNCO".to_string(),
        axis_x: ax_h,
        axis_y: ax_n,
        axis_z: ax_co,
        data: (0..96 * 64 * 64)
            .map(|i| deterministic_noise(302, i) * 1.2)
            .collect(),
    };
    for k in 0..32 {
        let h_k = 6.80 + ((k % 8) as f64) * 0.36 + ((k / 8) as f64) * 0.05;
        let n_k = 107.0 + ((k / 8) as f64) * 6.0 + ((k % 8) as f64) * 0.65;
        let co_k = 172.5 + ((k % 7) as f64) * 1.1;
        add_gaussian_2d(&mut hsqc, h_k, n_k, 130.0, 0.030, 0.30);
        add_gaussian_3d(&mut hnco_clean, h_k, n_k, co_k, 120.0, 0.030, 0.30, 0.24);
    }

    // 1. Canonical cube -> audit record emitted with `Canonical`, `applied == false`, `warning_message() == None`.
    let mut hnco_canon_test = hnco_clean.clone();
    let (_, _, audit_canon_opt) =
        register_3d_hn_to_hsqc_2d_with_audit(&mut hnco_canon_test, &hsqc, 0.08, 0.60, true);
    let audit_canon = audit_canon_opt.expect("audit record emitted on canonical 3D cube");
    assert_eq!(audit_canon.selected, NamedAxisHypothesis::Canonical);
    assert!(audit_canon.enabled);
    assert!(!audit_canon.applied);
    assert!(!audit_canon.polarity_flipped);
    assert!(audit_canon.warning_message().is_none());

    // 2. Corrupt `15N` by `HalfSwShift`:
    let mut hnco_corrupted = hnco_clean.clone();
    apply_indirect_axis_transform_3d(
        &mut hnco_corrupted,
        AxisTransformOp::HalfSwShift,
        AxisTransformOp::Canonical,
        Some("HNCO"),
        false,
    );
    let corrupted_snapshot = hnco_corrupted.data.clone();

    // 2a. With `enable_axis_rescue == false` (`--no-axis-rescue`), cube is NEVER mutated!
    let mut hnco_opt_out = hnco_corrupted.clone();
    let (dh_off, dn_off, audit_off_opt) =
        register_3d_hn_to_hsqc_2d_with_audit(&mut hnco_opt_out, &hsqc, 0.08, 0.60, false);
    assert_eq!(
        hnco_opt_out.data, corrupted_snapshot,
        "opt-out (`enable_axis_rescue = false`) must not mutate 3D cube intensities"
    );
    let _ = (dh_off, dn_off);
    let audit_off = audit_off_opt.expect("audit record emitted even when opt-out is set");
    assert!(!audit_off.enabled);
    assert!(!audit_off.applied);
    assert!(audit_off.warning_message().is_none());

    // 2b. With `enable_axis_rescue == true` (default), rescue applies and emits full provenance!
    let mut hnco_rescued = hnco_corrupted;
    let (_, _, audit_on_opt) =
        register_3d_hn_to_hsqc_2d_with_audit(&mut hnco_rescued, &hsqc, 0.08, 0.60, true);
    let audit_on: AxisRescueAuditRecord =
        audit_on_opt.expect("audit record emitted when axis rescue applies");
    assert_eq!(audit_on.experiment, "HNCO");
    assert_eq!(audit_on.selected, NamedAxisHypothesis::HalfSwShiftN);
    assert!(audit_on.enabled);
    assert!(audit_on.applied);
    assert!(audit_on.ln_bayes_factor_vs_canonical >= MIN_AXIS_RESCUE_LN_BF);
    assert!(audit_on.canonical_matched_fraction < 0.25);
    assert!(audit_on.selected_matched_fraction > 0.85);
    let warn_str = audit_on
        .warning_message()
        .expect("applied rescue must produce warning_message()");
    assert!(
        warn_str.starts_with("warning: [axis-rescue]"),
        "unexpected warning format: {warn_str}"
    );
    assert!(warn_str.contains("HalfSwShiftN"));

    // 3. Round-trip provenance through `ProcessingRecipe` JSON AND NEF `WorkbenchSessionState`!
    let mut recipe = ProcessingRecipe {
        dims: Vec::new(),
        nus: None,
        enable_axis_rescue: true,
        axis_rescue_audit: Some(audit_on.clone()),
    };
    let recipe_json = recipe.to_json_pretty().expect("serialize recipe JSON");
    assert!(recipe_json.contains("\"axis_rescue_audit\""));
    assert!(recipe_json.contains("\"HalfSwShiftN\""));
    let recipe_back = ProcessingRecipe::from_json_str(&recipe_json).expect("deserialize recipe");
    assert_eq!(recipe_back.axis_rescue_audit.as_ref(), Some(&audit_on));
    recipe.set_axis_rescue(false);
    assert!(!recipe.enable_axis_rescue);

    let mut session = WorkbenchSessionState::new("p1_1_nef_test", vec!["ALA".to_string()]);
    session
        .processing_recipes
        .insert("HNCO".to_string(), recipe_back);
    let nef_txt = write_session_nef_str(&session);
    let restored = read_session_nef_str(&nef_txt).expect("decode session from NEF");
    assert_eq!(
        restored
            .processing_recipes
            .get("HNCO")
            .and_then(|r| r.axis_rescue_audit.as_ref()),
        Some(&audit_on),
        "NEF round-trip must preserve AxisRescueAuditRecord losslessly"
    );

    // 4. Verify HTML Quality Report provenance section renders the applied transform!
    let mut conf_map = BTreeMap::new();
    conf_map.insert(0, 0.95);
    conf_map.insert(1, 0.92);
    let mut rep = ReportInput::new(
        "p1_1_report",
        Reference::new(vec!["ALA".into(), "GLY".into()], vec![1, 2]),
        conf_map,
    );
    rep.axis_rescue_records = vec![audit_canon, audit_on];
    let html = generate_quality_report_html(&rep);
    assert!(html.contains("id=\"axis-rescue-provenance\""));
    assert!(html.contains("HalfSwShiftN"));
    assert!(html.contains("[Non-canonical transform applied]"));
}
