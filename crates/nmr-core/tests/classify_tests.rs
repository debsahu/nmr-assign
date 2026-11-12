//! Integration tests for `nmr_core::classify` and `POST /api/v1/inspect` (`T09.1`).
//!
//! Covers:
//! - Synthetic Bruker & Varian header trees (including misnamed folders, `pulsecal` 1D decoy,
//!   `FnTYPE=0` template false-alarm guard, NUS density from `nuslist`)
//! - `Pseudo3D` relaxation & exchange (`vdlist` -> `RelaxR1`, `vclist` -> `Cpmg`, `fqlist` -> `Cest`,
//!   Varian `array = "d2"`)
//! - Protonless `13C`-detected IDP suite (`Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D` -> `Idp13cDetected`)
//! - Methyl-TROSY `13C-HMQC` (`MethylTrosy`), paramagnetic sweep (`Paramagnetic`), and ligand
//!   titration series (`TitrationCsp` + `TitrationSeries`)
//! - Processed Sparky `.ucsf` and multi-plane NMRPipe `.ft3` binary headers
//! - POSIX `.tar` and RFC 1952 `.tar.gz` (stored & fixed-Huffman LZ77) archives with multi-MB `ser`
//!   skipping (`< 50 ms`) and `wasm32`-compatible `classify_nmr_entries` parity
//! - `POST /api/v1/inspect` HTTP endpoint (JSON `path`, JSON `entries`, and binary `.tar.gz` body)
//! - Conditional real header benchmark (`< 50 ms`, `100%` accuracy) on `Ubiquitin`, `CISD3` (`51439`),
//!   `53371`, `18465`, `5329`, and `6318` when `NMR_REAL_DATA_DIR` or local `data/` is present.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

use nmr_core::classify::{
    classify_nmr_entries, classify_nmr_workspace, extract_archive_virtual_entries, DataState,
    ExperimentDimensionality, ExperimentVendor, ProblemRegime, PulseSequenceClass, QuadratureKind,
    VirtualEntry, WorkspaceInventory,
};
use nmr_core::service::spawn_background_server;

static NEXT_TMP_ID: AtomicUsize = AtomicUsize::new(1);

fn make_temp_dir(prefix: &str) -> PathBuf {
    let n = NEXT_TMP_ID.fetch_add(1, Ordering::SeqCst);
    let dir = env::temp_dir().join(format!("nmr_classify_{prefix}_{}_{n}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_text(root: &Path, rel: &str, content: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(p, content.as_bytes()).expect("write text file");
}

fn write_bytes(root: &Path, rel: &str, bytes: &[u8]) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(p, bytes).expect("write binary file");
}

fn bruker_acqus(
    pulprog: &str,
    nuc1: &str,
    sfo1: f64,
    sw_ppm: f64,
    td: usize,
    fn_type: usize,
) -> String {
    format!(
        "##TITLE= Parameter file, TopSpin 4.1\n\
         ##$PULPROG= <{pulprog}>\n\
         ##$NUC1= <{nuc1}>\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= {sfo1:.6}\n\
         ##$SW= {sw_ppm:.4}\n\
         ##$SW_h= {:.3}\n\
         ##$TD= {td}\n\
         ##$FnMODE= 0\n\
         ##$FnTYPE= {fn_type}\n\
         ##$NusAMOUNT= 25\n\
         ##END=\n",
        sw_ppm * sfo1
    )
}

fn bruker_indirect_acqus(nuc1: &str, sfo1: f64, sw_ppm: f64, td: usize, fn_mode: usize) -> String {
    format!(
        "##TITLE= Indirect parameter file\n\
         ##$NUC1= <{nuc1}>\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= {sfo1:.6}\n\
         ##$SW= {sw_ppm:.4}\n\
         ##$SW_h= {:.3}\n\
         ##$TD= {td}\n\
         ##$FnMODE= {fn_mode}\n\
         ##END=\n",
        sw_ppm * sfo1
    )
}

#[allow(clippy::too_many_arguments)]
fn varian_procpar_3d(
    seqfil: &str,
    sfrq: f64,
    dfrq: f64,
    dfrq2: f64,
    dn: &str,
    dn2: &str,
    sw: f64,
    sw1: f64,
    sw2: f64,
    ni: usize,
    ni2: usize,
) -> String {
    format!(
        "seqfil 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{seqfil}\"\n\
         0\n\
         pslabel 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{seqfil}\"\n\
         0\n\
         tn 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"H1\"\n\
         0\n\
         dn 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{dn}\"\n\
         0\n\
         dn2 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{dn2}\"\n\
         0\n\
         sfrq 1 1 10000 0 0 2 1 1 1 64\n\
         1 {sfrq:.6}\n\
         0\n\
         dfrq 1 1 10000 0 0 2 1 1 1 64\n\
         1 {dfrq:.6}\n\
         0\n\
         dfrq2 1 1 10000 0 0 2 1 1 1 64\n\
         1 {dfrq2:.6}\n\
         0\n\
         np 1 1 100000 0 0 2 1 1 1 64\n\
         1 2048\n\
         0\n\
         ni 1 1 100000 0 0 2 1 1 1 64\n\
         1 {ni}\n\
         0\n\
         ni2 1 1 100000 0 0 2 1 1 1 64\n\
         1 {ni2}\n\
         0\n\
         sw 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw:.2}\n\
         0\n\
         sw1 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw1:.2}\n\
         0\n\
         sw2 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw2:.2}\n\
         0\n\
         phase 1 1 10 0 0 2 1 1 1 64\n\
         2 1 2\n\
         0\n\
         phase2 1 1 10 0 0 2 1 1 1 64\n\
         2 1 2\n\
         0\n"
    )
}

#[test]
fn test_synthetic_misnamed_bruker_and_varian_headers_and_nus() {
    let root = make_temp_dir("misnamed_headers");

    // 1. Misnamed Bruker folder ("10_hnco_wrong_label") that is physically a 50% NUS 3D HNCACB
    //    with BOTH raw `ser` and processed `pdata/1/3rrr`.
    write_text(
        &root,
        "10_hnco_wrong_label/acqus",
        &bruker_acqus("hncacbgp3d", "1H", 800.13, 14.0, 2048, 2),
    );
    write_text(
        &root,
        "10_hnco_wrong_label/acqu2s",
        &bruker_indirect_acqus("13C", 201.19, 72.0, 160, 5), // States-TPPI, 80 hypercomplex
    );
    write_text(
        &root,
        "10_hnco_wrong_label/acqu3s",
        &bruker_indirect_acqus("15N", 81.08, 32.0, 80, 6), // Echo-Antiecho, 40 hypercomplex => 3200 grid
    );
    let mut nuslist = String::new();
    for idx in 0..1600 {
        nuslist.push_str(&format!("{} {}\n", idx % 80, idx / 80));
    }
    write_text(&root, "10_hnco_wrong_label/nuslist", &nuslist);
    write_bytes(&root, "10_hnco_wrong_label/ser", &[0u8; 512]);
    write_text(
        &root,
        "10_hnco_wrong_label/pdata/1/procs",
        "##$SI= 1024\n##END=\n",
    );
    write_bytes(&root, "10_hnco_wrong_label/pdata/1/3rrr", &[0u8; 1024]);

    // 2. Uniform 2D 15N-HSQC where Bruker template left `NusAMOUNT=25` even though `FnTYPE=0`.
    write_text(
        &root,
        "20_uniform_hsqc/acqus",
        &bruker_acqus("fhsqcf3gpph", "1H", 800.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "20_uniform_hsqc/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 34.0, 256, 6),
    );
    write_bytes(&root, "20_uniform_hsqc/ser", &[0u8; 256]);

    // 3. 1D `pulsecal` folder `99999` with a stale cloned title "3D HNCO" — must stay 1D Unknown & unchecked!
    write_text(
        &root,
        "99999/acqus",
        &bruker_acqus("pulsecal", "1H", 800.13, 14.0, 4096, 0),
    );
    write_text(&root, "99999/pdata/1/title", "Cloned 3D HNCO template\n");
    write_bytes(&root, "99999/fid", &[0u8; 128]);

    // 4. Misnamed Varian folder ("mystery_99.fid") containing `gcbca_co_nh`
    write_text(
        &root,
        "mystery_99.fid/procpar",
        &varian_procpar_3d(
            "gcbca_co_nh",
            799.85,
            201.12,
            81.05,
            "C13",
            "N15",
            11000.0,
            14500.0,
            2500.0,
            64,
            32,
        ),
    );
    write_text(&root, "mystery_99.fid/text", "Misnamed folder test\n");
    write_bytes(&root, "mystery_99.fid/fid", &[0u8; 256]);

    // 5. Auxiliary FASTA + NMR-STAR
    write_text(
        &root,
        "target.fasta",
        ">target_protein\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n",
    );
    write_text(&root, "reference.str", "data_synthetic\n");

    let inv = classify_nmr_workspace(&root).expect("classify synthetic tree");
    assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
    assert!(
        inv.missing_input_hints.is_empty(),
        "expected zero missing hints, got {:?}",
        inv.missing_input_hints
    );
    assert_eq!(inv.auxiliary_files.len(), 2);

    let hncacb = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "10_hnco_wrong_label")
        .expect("find 10_hnco_wrong_label");
    assert_eq!(hncacb.vendor, ExperimentVendor::Bruker);
    assert_eq!(hncacb.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(hncacb.data_state, DataState::Both);
    assert_eq!(hncacb.pulse_class, PulseSequenceClass::Hncacb);
    assert!(hncacb.nus.is_nus);
    assert_eq!(hncacb.nus.sampled_points, Some(1600));
    assert_eq!(hncacb.nus.full_grid_points, Some(3200));
    assert!((hncacb.nus.sampling_density.unwrap() - 0.50).abs() < 1e-6);
    assert!(
        hncacb.nus.badge.contains("NUS") && hncacb.nus.badge.contains("50"),
        "unexpected NUS badge: {}",
        hncacb.nus.badge
    );
    assert_eq!(hncacb.axes[1].quadrature, QuadratureKind::StatesTppi);
    assert_eq!(hncacb.axes[2].quadrature, QuadratureKind::EchoAntiecho);
    assert_eq!(hncacb.raw_signal_bytes, 512);
    assert_eq!(hncacb.processed_bytes, 1024);
    assert!(hncacb.selected_by_default);

    let hsqc = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "20_uniform_hsqc")
        .expect("find 20_uniform_hsqc");
    assert_eq!(hsqc.pulse_class, PulseSequenceClass::N15Hsqc);
    assert!(
        !hsqc.nus.is_nus,
        "FnTYPE=0 must override template NusAMOUNT"
    );
    assert_eq!(hsqc.nus.badge, "Uniform");

    let pcal = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "99999")
        .expect("find pulsecal");
    assert_eq!(pcal.dimensionality, ExperimentDimensionality::Dim1D);
    assert_eq!(pcal.pulse_class, PulseSequenceClass::Unknown);
    assert!(!pcal.selected_by_default);

    let varian = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "mystery_99.fid")
        .expect("find mystery_99.fid");
    assert_eq!(varian.vendor, ExperimentVendor::Varian);
    assert_eq!(varian.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(varian.pulse_class, PulseSequenceClass::Cbcaconh);
    assert!((varian.spectrometer_mhz.unwrap() - 799.85).abs() < 0.1);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_pseudo3d_relaxation_cpmg_cest_and_varian_array() {
    let root = make_temp_dir("pseudo3d_dynamics");

    // Bruker R1 via vdlist
    write_text(
        &root,
        "10_r1/acqus",
        &bruker_acqus("hsqct1etf3gpsi3d", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "10_r1/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "10_r1/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 10, 1),
    );
    write_text(
        &root,
        "10_r1/vdlist",
        "0.01\n0.05\n0.10\n0.20\n0.40\n0.60\n0.80\n1.00\n1.20\n1.50\n",
    );
    write_bytes(&root, "10_r1/ser", &[0u8; 128]);

    // Bruker CPMG via vclist
    write_text(
        &root,
        "20_cpmg/acqus",
        &bruker_acqus("hsqccpmg", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "20_cpmg/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "20_cpmg/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 12, 1),
    );
    write_text(
        &root,
        "20_cpmg/vclist",
        "0\n2\n4\n6\n8\n10\n12\n16\n20\n24\n32\n40\n",
    );
    write_bytes(&root, "20_cpmg/ser", &[0u8; 128]);

    // Bruker CEST via fqlist
    write_text(
        &root,
        "30_cest/acqus",
        &bruker_acqus("hsqc_cest_3d", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "30_cest/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "30_cest/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 8, 1),
    );
    write_text(
        &root,
        "30_cest/fqlist",
        "105.0\n110.0\n115.0\n120.0\n125.0\n130.0\n135.0\n140.0\n",
    );
    write_bytes(&root, "30_cest/ser", &[0u8; 128]);

    // Varian R1 array = "d2"
    write_text(
        &root,
        "40_varian_r1.fid/procpar",
        "seqfil 2 2 8192 0 0 2 1 1 1 64\n1 \"gNhsqc\"\n0\n\
         tn 2 2 8192 0 0 2 1 1 1 64\n1 \"H1\"\n0\n\
         dn 2 2 8192 0 0 2 1 1 1 64\n1 \"N15\"\n0\n\
         dn2 2 2 8192 0 0 2 1 1 1 64\n1 \"\"\n0\n\
         sfrq 1 1 10000 0 0 2 1 1 1 64\n1 599.91\n0\n\
         dfrq 1 1 10000 0 0 2 1 1 1 64\n1 60.79\n0\n\
         dfrq2 1 1 10000 0 0 2 1 1 1 64\n1 60.79\n0\n\
         np 1 1 100000 0 0 2 1 1 1 64\n1 2048\n0\n\
         ni 1 1 100000 0 0 2 1 1 1 64\n1 128\n0\n\
         ni2 1 1 100000 0 0 2 1 1 1 64\n1 1\n0\n\
         sw 1 1 1000000 0 0 2 1 1 1 64\n1 8400.0\n0\n\
         sw1 1 1 1000000 0 0 2 1 1 1 64\n1 2000.0\n0\n\
         phase 1 1 10 0 0 2 1 1 1 64\n2 1 2\n0\n\
         array 2 2 8192 0 0 2 1 1 1 64\n1 \"d2\"\n0\n\
         d2 1 1 10000 0 0 2 1 1 1 64\n6 0.01 0.05 0.1 0.2 0.5 1.0\n0\n",
    );
    write_bytes(&root, "40_varian_r1.fid/fid", &[0u8; 128]);

    let inv = classify_nmr_workspace(&root).expect("classify dynamics workspace");
    assert_eq!(inv.problem_regime, ProblemRegime::RelaxationDynamics);

    let by_path = |p: &str| {
        inv.experiments
            .iter()
            .find(|e| e.relative_path == p)
            .unwrap_or_else(|| panic!("missing {p}"))
    };
    assert_eq!(
        by_path("10_r1").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("10_r1").pulse_class, PulseSequenceClass::RelaxR1);

    assert_eq!(
        by_path("20_cpmg").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("20_cpmg").pulse_class, PulseSequenceClass::Cpmg);

    assert_eq!(
        by_path("30_cest").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("30_cest").pulse_class, PulseSequenceClass::Cest);

    assert_eq!(
        by_path("40_varian_r1.fid").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(
        by_path("40_varian_r1.fid").pulse_class,
        PulseSequenceClass::RelaxR1
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_13c_detected_idp_methyl_trosy_paramagnetic_and_titration_regimes() {
    // A. Protonless 13C-detected IDP workspace (`Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D`)
    let idp_dir = make_temp_dir("idp_13c_detected");
    write_text(
        &idp_dir,
        "1_con/acqus",
        &bruker_acqus("c_con_iasq", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "1_con/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 36.0, 256, 5),
    );
    write_bytes(&idp_dir, "1_con/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "2_caco/acqus",
        &bruker_acqus("c_caco_ia", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "2_caco/acqu2s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 256, 5),
    );
    write_bytes(&idp_dir, "2_caco/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "3_canco/acqus",
        &bruker_acqus("c_canco_ia3d", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "3_canco/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 36.0, 96, 5),
    );
    write_text(
        &idp_dir,
        "3_canco/acqu3s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 128, 5),
    );
    write_bytes(&idp_dir, "3_canco/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "4_cbcaco/acqus",
        &bruker_acqus("c_cbcaco_ia3d", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "4_cbcaco/acqu2s",
        &bruker_indirect_acqus("13C", 201.20, 70.0, 128, 5),
    );
    write_text(
        &idp_dir,
        "4_cbcaco/acqu3s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 96, 5),
    );
    write_bytes(&idp_dir, "4_cbcaco/ser", &[0u8; 64]);

    let idp_inv = classify_nmr_workspace(&idp_dir).expect("classify 13C-detected IDP");
    assert_eq!(idp_inv.problem_regime, ProblemRegime::Idp13cDetected);
    let classes: Vec<PulseSequenceClass> =
        idp_inv.experiments.iter().map(|e| e.pulse_class).collect();
    assert_eq!(
        classes,
        vec![
            PulseSequenceClass::Con2D,
            PulseSequenceClass::Caco2D,
            PulseSequenceClass::Canco3D,
            PulseSequenceClass::Cbcaco3D
        ]
    );
    // Spectrometer base 1H MHz inferred from 13C 201.20 MHz (~800 MHz)
    let mhz = idp_inv.experiments[0].spectrometer_mhz.unwrap();
    assert!((mhz - 800.0).abs() < 2.0, "expected ~800 MHz, got {mhz}");
    let _ = fs::remove_dir_all(&idp_dir);

    // B. Methyl-TROSY 13C-HMQC workspace (`MethylTrosy`)
    let me_dir = make_temp_dir("methyl_trosy");
    write_text(
        &me_dir,
        "1_methyl/acqus",
        &bruker_acqus("hmqcphpr", "1H", 900.21, 12.0, 2048, 0),
    );
    write_text(
        &me_dir,
        "1_methyl/acqu2s",
        &bruker_indirect_acqus("13C", 226.38, 22.0, 256, 5),
    );
    write_bytes(&me_dir, "1_methyl/ser", &[0u8; 64]);
    let me_inv = classify_nmr_workspace(&me_dir).expect("classify methyl-TROSY");
    assert_eq!(
        me_inv.experiments[0].pulse_class,
        PulseSequenceClass::C13Hmqc
    );
    assert_eq!(me_inv.problem_regime, ProblemRegime::MethylTrosy);
    let _ = fs::remove_dir_all(&me_dir);

    // C. Paramagnetic ultra-wide 1H sweep (`SW = 45 ppm`)
    let para_dir = make_temp_dir("paramagnetic");
    write_text(
        &para_dir,
        "1_para/acqus",
        &bruker_acqus("hsqcetfpf3gpsi", "1H", 600.13, 45.0, 4096, 0),
    );
    write_text(
        &para_dir,
        "1_para/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 40.0, 256, 6),
    );
    write_bytes(&para_dir, "1_para/ser", &[0u8; 64]);
    let para_inv = classify_nmr_workspace(&para_dir).expect("classify paramagnetic");
    assert_eq!(para_inv.problem_regime, ProblemRegime::Paramagnetic);
    let _ = fs::remove_dir_all(&para_dir);

    // D. Ligand titration CSP series (`0uM`, `50uM`, `150uM`, `500uM`)
    let tit_dir = make_temp_dir("titration_csp");
    for conc in ["01_0uM", "02_50uM", "03_150uM", "04_500uM"] {
        write_text(
            &tit_dir,
            &format!("{conc}/acqus"),
            &bruker_acqus("hsqcetfpf3gpsi", "1H", 700.13, 14.0, 2048, 0),
        );
        write_text(
            &tit_dir,
            &format!("{conc}/acqu2s"),
            &bruker_indirect_acqus("15N", 70.95, 32.0, 256, 6),
        );
        write_bytes(&tit_dir, &format!("{conc}/ser"), &[0u8; 64]);
    }
    let tit_inv = classify_nmr_workspace(&tit_dir).expect("classify titration series");
    assert_eq!(tit_inv.problem_regime, ProblemRegime::TitrationCsp);
    assert!(tit_inv
        .experiments
        .iter()
        .all(|e| e.pulse_class == PulseSequenceClass::TitrationSeries));
    let _ = fs::remove_dir_all(&tit_dir);
}

fn build_synthetic_ucsf_3d(
    n0: (&str, u32, f32, f32, f32),
    n1: (&str, u32, f32, f32, f32),
    n2: (&str, u32, f32, f32, f32),
) -> Vec<u8> {
    let mut buf = vec![0u8; 180 + 3 * 128];
    buf[..8].copy_from_slice(b"UCSF NMR");
    buf[10] = 3; // ndim
    buf[13] = 2; // version 2
    for (idx, &(nuc, size, mhz, sw_hz, orig_ppm)) in [n0, n1, n2].iter().enumerate() {
        let base = 180 + idx * 128;
        let nb = nuc.as_bytes();
        buf[base..base + nb.len()].copy_from_slice(nb);
        buf[base + 8..base + 12].copy_from_slice(&size.to_be_bytes());
        buf[base + 16..base + 20].copy_from_slice(&8u32.to_be_bytes());
        buf[base + 20..base + 24].copy_from_slice(&mhz.to_be_bytes());
        buf[base + 24..base + 28].copy_from_slice(&sw_hz.to_be_bytes());
        buf[base + 28..base + 32].copy_from_slice(&orig_ppm.to_be_bytes());
    }
    buf
}

fn build_synthetic_nmrpipe_plane_header(z_planes: f32) -> Vec<u8> {
    let mut fdata = [0.0f32; 512];
    fdata[0] = 0.0;
    fdata[2] = 2.345; // IEEE-754 little-endian signature
    fdata[9] = 3.0; // FDDIMCOUNT = 3
    fdata[24] = 2.0; // DimOrder0 = F2 (direct 1H)
    fdata[25] = 1.0; // DimOrder1 = F1 (indirect 13C)
    fdata[26] = 3.0; // DimOrder2 = F3 (indirect 15N)

    // F2 (slot index 0 => X): 1H 600 MHz, 1024 pts, 8400 Hz
    let mut label_buf = [0u8; 32];
    label_buf[0..2].copy_from_slice(b"HN");
    label_buf[8..11].copy_from_slice(b"13C");
    label_buf[16..19].copy_from_slice(b"15N");
    for (k, chunk) in label_buf.chunks_exact(4).enumerate() {
        fdata[16 + k] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    fdata[99] = 1024.0; // F2 size
    fdata[100] = 8400.0; // F2 SW Hz
    fdata[119] = 600.13; // F2 OBS MHz
    fdata[56] = 1.0; // FT_FLAG

    // F1 (slot index 1): 13C 150.9 MHz, 128 pts, 10800 Hz (~71.5 ppm => HNCACB / CBCACONH window)
    fdata[219] = 128.0;
    fdata[229] = 10800.0;
    fdata[218] = 150.90;
    fdata[55] = 1.0;

    // F3 (slot index 2): 15N 60.81 MHz, z_planes pts, 1945 Hz (~32 ppm)
    fdata[15] = z_planes;
    fdata[11] = 1945.0;
    fdata[10] = 60.81;
    fdata[13] = 1.0;

    let mut bytes = Vec::with_capacity(2048);
    for v in fdata {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[test]
fn test_processed_ucsf_and_nmrpipe_plane_series_headers() {
    let root = make_temp_dir("processed_headers");

    let ucsf = build_synthetic_ucsf_3d(
        ("1H", 1024, 600.13, 8400.0, 4.7),
        ("15N", 64, 60.81, 1945.0, 118.0),
        ("13C", 96, 150.90, 4800.0, 54.0), // ~31.8 ppm 13CA window
    );
    write_bytes(&root, "spectra/HNCA.ucsf", &ucsf);

    let pipe_hdr = build_synthetic_nmrpipe_plane_header(4.0);
    for plane in 1..=4 {
        write_bytes(&root, &format!("pipe/HNCACB{plane:03}.ft3"), &pipe_hdr);
    }

    let inv = classify_nmr_workspace(&root).expect("classify processed workspace");
    assert_eq!(inv.experiments.len(), 2);

    let ucsf_exp = inv
        .experiments
        .iter()
        .find(|e| e.vendor == ExperimentVendor::SparkyUcsf)
        .expect("find UCSF experiment");
    assert_eq!(ucsf_exp.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(ucsf_exp.data_state, DataState::Processed);
    assert_eq!(ucsf_exp.pulse_class, PulseSequenceClass::Hnca);
    assert!((ucsf_exp.spectrometer_mhz.unwrap() - 600.13).abs() < 0.05);

    let pipe_exp = inv
        .experiments
        .iter()
        .find(|e| e.vendor == ExperimentVendor::NmrPipe)
        .expect("find NMRPipe series");
    assert_eq!(pipe_exp.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(pipe_exp.data_state, DataState::Processed);
    assert_eq!(pipe_exp.pulse_class, PulseSequenceClass::Hncacb);
    assert_eq!(pipe_exp.processed_bytes, 4 * 2048);

    let _ = fs::remove_dir_all(&root);
}

fn append_tar_entry(tar: &mut Vec<u8>, name: &str, payload: &[u8]) {
    let mut hdr = [0u8; 512];
    let nb = name.as_bytes();
    hdr[..nb.len()].copy_from_slice(nb);
    hdr[100..107].copy_from_slice(b"0000644");
    hdr[108..115].copy_from_slice(b"0001000");
    hdr[116..123].copy_from_slice(b"0001000");
    let size_oct = format!("{:011o}", payload.len());
    hdr[124..135].copy_from_slice(size_oct.as_bytes());
    hdr[136..147].copy_from_slice(b"14600000000");
    hdr[148..156].fill(b' ');
    hdr[156] = b'0';
    hdr[257..263].copy_from_slice(b"ustar\0");
    hdr[263..265].copy_from_slice(b"00");
    let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
    let csum_oct = format!("{csum:06o}\0 ");
    hdr[148..156].copy_from_slice(csum_oct.as_bytes());

    tar.extend_from_slice(&hdr);
    tar.extend_from_slice(payload);
    let rem = payload.len() % 512;
    if rem != 0 {
        tar.extend(std::iter::repeat_n(0u8, 512 - rem));
    }
}

/// Encode bytes as a valid RFC 1952 GZIP stream using RFC 1951 stored blocks (`BTYPE=00`).
fn gzip_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // GZIP 10-byte header: ID1=0x1f, ID2=0x8b, CM=8, FLG=0, MTIME=0, XFL=0, OS=255
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 255]);
    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[][..]]
    } else {
        data.chunks(65_535).collect()
    };
    for (idx, chunk) in chunks.iter().enumerate() {
        let is_last = idx + 1 == chunks.len();
        out.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        let len = chunk.len() as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(chunk);
    }
    // CRC32 + ISIZE (8-byte trailer)
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Encode bytes as a valid RFC 1952 GZIP stream using RFC 1951 fixed-Huffman (`BTYPE=01`)
/// with run-length LZ77 distance=1 compression for zero runs so a 4 MiB `ser` compresses to ~20 KiB!
fn gzip_fixed_huffman_rle_zeros(data: &[u8]) -> Vec<u8> {
    struct BitWriter {
        bytes: Vec<u8>,
        buf: u64,
        cnt: u8,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                buf: 0,
                cnt: 0,
            }
        }
        fn write_bits_lsb(&mut self, val: u32, n: u8) {
            self.buf |= (val as u64) << self.cnt;
            self.cnt += n;
            while self.cnt >= 8 {
                self.bytes.push((self.buf & 0xFF) as u8);
                self.buf >>= 8;
                self.cnt -= 8;
            }
        }
        fn write_huff_msb(&mut self, code: u16, len: u8) {
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as u32;
                self.write_bits_lsb(bit, 1);
            }
        }
        fn emit_fixed_lit(&mut self, b: u8) {
            if b <= 143 {
                self.write_huff_msb(0x30 + (b as u16), 8);
            } else {
                self.write_huff_msb(0x190 + ((b - 144) as u16), 9);
            }
        }
        fn emit_match_258_dist1(&mut self) {
            // Length 258 = symbol 285 -> fixed code 280..287 => 0xC0 + (285 - 280) = 0xC5 (8 bits)
            self.write_huff_msb(0xC5, 8);
            // Distance 1 = dist symbol 0 -> 5-bit MSB code 00000
            self.write_huff_msb(0, 5);
        }
        fn finish(mut self) -> Vec<u8> {
            // End-of-block symbol 256 -> 7-bit MSB code 0000000
            self.write_huff_msb(0, 7);
            if self.cnt > 0 {
                self.bytes.push((self.buf & 0xFF) as u8);
            }
            self.bytes
        }
    }

    let mut bw = BitWriter::new();
    // BFINAL=1, BTYPE=01 (fixed Huffman) -> LSB bits: 1, then 01 => 0b011 = 3 in 3 bits
    bw.write_bits_lsb(0b011, 3);
    let mut i = 0usize;
    while i < data.len() {
        if i > 0
            && data[i - 1] == 0
            && i + 258 <= data.len()
            && data[i..i + 258].iter().all(|&x| x == 0)
        {
            bw.emit_match_258_dist1();
            i += 258;
        } else {
            bw.emit_fixed_lit(data[i]);
            i += 1;
        }
    }
    let deflate = bw.finish();

    let mut out = Vec::with_capacity(18 + deflate.len());
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 255]);
    out.extend_from_slice(&deflate);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

#[test]
fn test_tar_and_targz_archive_streaming_and_wasm_entries_parity() {
    let root = make_temp_dir("archives");

    let acqus = bruker_acqus("hncogp3d", "1H", 700.13, 14.0, 2048, 0);
    let acqu2s = bruker_indirect_acqus("13C", 176.05, 14.0, 96, 5);
    let acqu3s = bruker_indirect_acqus("15N", 70.95, 32.0, 64, 6);
    let fasta =
        b">ubq\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n";
    let huge_ser = vec![0u8; 4 * 1024 * 1024]; // 4 MiB FID payload

    let mut tar_bytes = Vec::new();
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqus", acqus.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqu2s", acqu2s.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqu3s", acqu3s.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/ser", &huge_ser);
    append_tar_entry(&mut tar_bytes, "bundle/ubq.fasta", fasta);
    tar_bytes.extend(std::iter::repeat_n(0u8, 1024)); // Two 512-byte zero EOF blocks

    let tar_path = root.join("bundle.tar");
    fs::write(&tar_path, &tar_bytes).unwrap();

    let tgz_stored = gzip_stored(&tar_bytes);
    let tgz_stored_path = root.join("bundle_stored.tar.gz");
    fs::write(&tgz_stored_path, &tgz_stored).unwrap();

    let tgz_huff = gzip_fixed_huffman_rle_zeros(&tar_bytes);
    assert!(
        tgz_huff.len() < 60_000,
        "4 MiB zero-heavy tar should RLE-compress under 60 KB, got {}",
        tgz_huff.len()
    );
    let tgz_huff_path = root.join("bundle_huff.tgz");
    fs::write(&tgz_huff_path, &tgz_huff).unwrap();

    // 1. Native `.tar` scan (must seek past 4 MiB `ser` in < 50 ms)
    let inv_tar = classify_nmr_workspace(&tar_path).expect("classify .tar");
    assert!(
        inv_tar.elapsed_ms < 50.0,
        ".tar header scan took {:.2} ms (> 50 ms)",
        inv_tar.elapsed_ms
    );
    assert_eq!(inv_tar.experiments.len(), 1);
    assert_eq!(inv_tar.experiments[0].pulse_class, PulseSequenceClass::Hnco);
    assert_eq!(inv_tar.experiments[0].raw_signal_bytes, 4 * 1024 * 1024);

    // 2. Native `.tar.gz` stored + Huffman scans
    let inv_tgz_stored = classify_nmr_workspace(&tgz_stored_path).expect("classify stored .tar.gz");
    let inv_tgz_huff = classify_nmr_workspace(&tgz_huff_path).expect("classify huffman .tgz");
    assert!(
        inv_tgz_huff.elapsed_ms < 50.0,
        ".tgz header scan took {:.2} ms (> 50 ms)",
        inv_tgz_huff.elapsed_ms
    );
    assert_eq!(inv_tgz_stored.experiments, inv_tar.experiments);
    assert_eq!(inv_tgz_huff.experiments, inv_tar.experiments);
    assert_eq!(inv_tgz_huff.auxiliary_files, inv_tar.auxiliary_files);

    // 3. `wasm32`-compatible `classify_nmr_entries` in-memory parity
    let wasm_entries = vec![
        VirtualEntry::new("bundle/10_hnco/acqus", acqus.into_bytes()),
        VirtualEntry::new("bundle/10_hnco/acqu2s", acqu2s.into_bytes()),
        VirtualEntry::new("bundle/10_hnco/acqu3s", acqu3s.into_bytes()),
        VirtualEntry::header_only("bundle/10_hnco/ser", Vec::new(), 4 * 1024 * 1024),
        VirtualEntry::new("bundle/ubq.fasta", fasta.to_vec()),
    ];
    let inv_wasm = classify_nmr_entries(&wasm_entries);
    assert_eq!(inv_wasm.experiments, inv_tar.experiments);
    assert_eq!(inv_wasm.auxiliary_files, inv_tar.auxiliary_files);
    assert_eq!(inv_wasm.problem_regime, inv_tar.problem_regime);

    // 4. `extract_archive_virtual_entries` + inline `.tgz` expansion inside `classify_nmr_entries`
    let extracted =
        extract_archive_virtual_entries("bundle_huff.tgz", &tgz_huff).expect("archive bytes gz");
    let inv_mem_gz = classify_nmr_entries(&extracted);
    assert_eq!(inv_mem_gz.experiments, inv_tar.experiments);
    let inv_inline_drop = classify_nmr_entries(&[VirtualEntry::new("bundle_huff.tgz", tgz_huff)]);
    assert_eq!(inv_inline_drop.experiments, inv_tar.experiments);

    let _ = fs::remove_dir_all(&root);
}

fn send_http_post(
    addr: std::net::SocketAddr,
    path: &str,
    ctype: &str,
    body: &[u8],
) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).expect("connect to service");
    let req_hdr = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(req_hdr.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
    stream.flush().unwrap();

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let sep = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("HTTP header separator");
    let hdr_str = String::from_utf8_lossy(&raw[..sep]);
    let status: u16 = hdr_str
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (status, raw[sep + 4..].to_vec())
}

#[test]
fn test_service_post_api_v1_inspect_modes() {
    let root = make_temp_dir("service_inspect");
    let ws_dir = root.join("sample_ws");
    write_text(
        &ws_dir,
        "1_hsqc/acqus",
        &bruker_acqus("hsqcetfpf3gpsi", "1H", 800.13, 14.0, 2048, 0),
    );
    write_text(
        &ws_dir,
        "1_hsqc/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 32.0, 256, 6),
    );
    write_bytes(&ws_dir, "1_hsqc/ser", &[0u8; 128]);

    let storage_dir = root.join("srv_storage");
    let mut srv =
        spawn_background_server("127.0.0.1:0", storage_dir).expect("spawn background server");
    let addr = srv.local_addr();

    // Mode 1: JSON `{ "path": ... }`
    let body_path = serde_json::json!({
        "path": ws_dir.to_string_lossy()
    });
    let (st1, resp1) = send_http_post(
        addr,
        "/api/v1/inspect",
        "application/json",
        &serde_json::to_vec(&body_path).unwrap(),
    );
    assert_eq!(st1, 200, "got: {}", String::from_utf8_lossy(&resp1));
    let inv1: WorkspaceInventory = serde_json::from_slice(&resp1).unwrap();
    assert_eq!(inv1.experiments.len(), 1);
    assert_eq!(inv1.experiments[0].pulse_class, PulseSequenceClass::N15Hsqc);

    // Mode 2: JSON `{ "root_name": "browser_drop", "entries": [...] }`
    let entries = vec![
        VirtualEntry::new(
            "drop/acqus",
            bruker_acqus("hncagp3d", "1H", 800.13, 14.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "drop/acqu2s",
            bruker_indirect_acqus("13C", 201.19, 32.0, 96, 5).into_bytes(),
        ),
        VirtualEntry::new(
            "drop/acqu3s",
            bruker_indirect_acqus("15N", 81.08, 32.0, 64, 6).into_bytes(),
        ),
        VirtualEntry::header_only("drop/ser", Vec::new(), 65_536),
    ];
    let body_entries = serde_json::json!({
        "root_name": "browser_drop",
        "entries": entries
    });
    let (st2, resp2) = send_http_post(
        addr,
        "/api/v1/inspect",
        "application/json",
        &serde_json::to_vec(&body_entries).unwrap(),
    );
    assert_eq!(st2, 200);
    let inv2: WorkspaceInventory = serde_json::from_slice(&resp2).unwrap();
    assert_eq!(inv2.root_name, "browser_drop");
    assert_eq!(inv2.experiments.len(), 1);
    assert_eq!(inv2.experiments[0].pulse_class, PulseSequenceClass::Hnca);

    // Mode 3: Binary `.tar.gz` upload (`Content-Type: application/gzip`)
    let mut tar = Vec::new();
    append_tar_entry(
        &mut tar,
        "arch/1_con/acqus",
        bruker_acqus("c_con_iasq", "13C", 201.20, 40.0, 2048, 0).as_bytes(),
    );
    append_tar_entry(
        &mut tar,
        "arch/1_con/acqu2s",
        bruker_indirect_acqus("15N", 81.08, 36.0, 256, 5).as_bytes(),
    );
    append_tar_entry(&mut tar, "arch/1_con/ser", &[0u8; 512]);
    tar.extend(std::iter::repeat_n(0u8, 1024));
    let tgz = gzip_fixed_huffman_rle_zeros(&tar);
    let (st3, resp3) = send_http_post(addr, "/api/v1/inspect", "application/gzip", &tgz);
    assert_eq!(st3, 200);
    let inv3: WorkspaceInventory = serde_json::from_slice(&resp3).unwrap();
    assert_eq!(inv3.problem_regime, ProblemRegime::Idp13cDetected);
    assert_eq!(inv3.experiments[0].pulse_class, PulseSequenceClass::Con2D);

    srv.shutdown();
    let _ = fs::remove_dir_all(&root);
}

fn find_real_data_dir() -> Option<PathBuf> {
    if let Ok(val) = env::var("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(val);
        if p.is_dir() {
            return Some(p);
        }
    }
    let fallback = PathBuf::from("/Users/debsahu/Workspace/nmr_software/data");
    if fallback.is_dir() {
        return Some(fallback);
    }
    None
}

#[test]
fn test_conditional_real_dev_datasets_accuracy_and_speed() {
    let Some(data_dir) = find_real_data_dir() else {
        eprintln!("Skipping real dataset header tests (NMR_REAL_DATA_DIR not present)");
        return;
    };

    // 1. Ubiquitin (`data/Ubiquitin`) — multi-modal Bruker + Sparky UCSF + 13C-detected + relaxation + pulsecal decoy
    let ubq_path = data_dir.join("Ubiquitin");
    if ubq_path.is_dir() {
        let inv = classify_nmr_workspace(&ubq_path).expect("classify Ubiquitin");
        eprintln!(
            "Real dataset Ubiquitin: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "Ubiquitin header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert!(matches!(
            inv.problem_regime,
            ProblemRegime::SidechainNoesy | ProblemRegime::GlobularBackbone
        ));
        assert!(inv
            .secondary_regimes
            .contains(&ProblemRegime::Idp13cDetected));
        assert!(inv
            .secondary_regimes
            .contains(&ProblemRegime::RelaxationDynamics));

        // Check `99999` pulsecal 1D decoy is NOT misclassified as 3D HNCO
        let pcal = inv
            .experiments
            .iter()
            .find(|e| e.relative_path.ends_with("/99999") || e.relative_path == "99999")
            .expect("Ubiquitin pulsecal 99999");
        assert_eq!(pcal.dimensionality, ExperimentDimensionality::Dim1D);
        assert_eq!(pcal.pulse_class, PulseSequenceClass::Unknown);
        assert!(!pcal.selected_by_default);

        // Check Pseudo3D R1 (`Ubiquitin_apo_083109/4`)
        let pseudo_r1 = inv
            .experiments
            .iter()
            .find(|e| e.relative_path == "Ubiquitin_apo_083109/4")
            .expect("Ubiquitin_apo_083109/4");
        assert_eq!(pseudo_r1.dimensionality, ExperimentDimensionality::Pseudo3D);
        assert_eq!(pseudo_r1.pulse_class, PulseSequenceClass::RelaxR1);

        // Check 13C-detected `Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D` AND Sparky `.ucsf` files
        for expected in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Con2D,
            PulseSequenceClass::Caco2D,
            PulseSequenceClass::Canco3D,
            PulseSequenceClass::Cbcaco3D,
            PulseSequenceClass::RelaxR1,
            PulseSequenceClass::RelaxR2,
            PulseSequenceClass::HetNoe,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == expected),
                "Ubiquitin missing expected pulse class {expected:?}"
            );
        }
        assert!(inv
            .experiments
            .iter()
            .any(|e| e.vendor == ExperimentVendor::SparkyUcsf));
    }

    // 2. CISD3 (`BMRB 51439`) — Bruker 900 MHz with 50% NUS CBCACONH
    let cisd3_path = data_dir.join("private/bmrb51439_cisd3");
    if cisd3_path.is_dir() {
        let inv = classify_nmr_workspace(&cisd3_path).expect("classify CISD3 51439");
        eprintln!(
            "Real dataset CISD3 (51439): {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "CISD3 51439 header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        assert!(inv.missing_input_hints.is_empty());

        let expected_cisd3 = [
            (
                "CISD3_15N_HSQC_diamagnetic",
                PulseSequenceClass::N15Hsqc,
                false,
            ),
            ("CISD3_CBCACONH", PulseSequenceClass::Cbcaconh, true),
            ("CISD3_HNCA", PulseSequenceClass::Hnca, false),
            ("CISD3_HNCO", PulseSequenceClass::Hnco, false),
            ("CISD3_HNCOCA", PulseSequenceClass::Hncoca, false),
        ];
        for (sub, exp_cls, exp_nus) in expected_cisd3 {
            let e = inv
                .experiments
                .iter()
                .find(|x| x.relative_path.contains(sub))
                .unwrap_or_else(|| panic!("CISD3 missing {sub}"));
            assert_eq!(e.pulse_class, exp_cls, "mismatch on {sub}");
            assert_eq!(e.nus.is_nus, exp_nus, "NUS mismatch on {sub}");
            if exp_nus {
                assert!((e.nus.sampling_density.unwrap() - 0.50).abs() < 0.02);
            }
        }
    }

    // 3. BMRB 53371 — Bruker NUS triple-resonance + Uniform `fhsqc` (`FnTYPE=0` with template `NusAMOUNT=25`)
    let p53371 = data_dir.join("private/bmrb/53371");
    if p53371.is_dir() {
        let inv = classify_nmr_workspace(&p53371).expect("classify 53371");
        eprintln!(
            "Real dataset 53371: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "53371 header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        let fhsqc = inv
            .experiments
            .iter()
            .find(|e| e.pulse_class == PulseSequenceClass::N15Hsqc)
            .expect("53371 N15Hsqc");
        assert!(
            !fhsqc.nus.is_nus,
            "53371 fHSQC has FnTYPE=0 and must be Uniform"
        );
        for cls in [
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hnco,
        ] {
            let e = inv
                .experiments
                .iter()
                .find(|x| x.pulse_class == cls)
                .unwrap_or_else(|| panic!("53371 missing {cls:?}"));
            assert!(e.nus.is_nus, "53371 {cls:?} should be NUS");
        }
    }

    // 4. BMRB 18465 — Mixed Bruker (`b8/`) + Varian (`v6/` with `refsource1="dfrq2"` 15N-HSQC!)
    let p18465 = data_dir.join("private/bmrb/18465");
    if p18465.is_dir() {
        let inv = classify_nmr_workspace(&p18465).expect("classify 18465");
        eprintln!(
            "Real dataset 18465: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "18465 header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        assert!(inv
            .experiments
            .iter()
            .any(|e| e.vendor == ExperimentVendor::Bruker
                && e.pulse_class == PulseSequenceClass::N15Hsqc));
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hnca,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
        ] {
            assert!(
                inv.experiments
                    .iter()
                    .any(|e| e.vendor == ExperimentVendor::Varian && e.pulse_class == cls),
                "18465 Varian missing {cls:?}"
            );
        }
        assert!(
            inv.experiments
                .iter()
                .all(|e| e.pulse_class != PulseSequenceClass::Unknown),
            "18465 had Unknown experiments: {:?}",
            inv.experiments
        );
    }

    // 5. BMRB 5329 — Varian backbone + sidechain TOCSY + 3D/4D NOESY (`SidechainNoesy`)
    let p5329 = data_dir.join("private/bmrb/5329");
    if p5329.is_dir() {
        let inv = classify_nmr_workspace(&p5329).expect("classify 5329");
        eprintln!(
            "Real dataset 5329: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "5329 header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::SidechainNoesy);
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::C13Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Ccconh,
            PulseSequenceClass::Hccconh,
            PulseSequenceClass::HcchTocsy,
            PulseSequenceClass::N15Noesy,
            PulseSequenceClass::C13Noesy,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == cls),
                "5329 missing {cls:?}"
            );
        }
        assert!(
            inv.experiments.iter().any(|e| {
                e.pulse_class == PulseSequenceClass::C13Noesy
                    && e.dimensionality == ExperimentDimensionality::Dim4D
            }),
            "5329 missing 4D C13Noesy"
        );
        assert!(
            inv.experiments.iter().all(|e| {
                e.pulse_class != PulseSequenceClass::Unknown
                    || e.pulse_program.as_deref() == Some("ghnha")
            }),
            "5329 had unexpected Unknown experiments: {:?}",
            inv.experiments
        );
    }

    // 6. BMRB 6318 — Bruker XWIN-NMR backbone + sidechain TOCSY + NOESY (`SidechainNoesy`)
    let p6318 = data_dir.join("private/bmrb/6318");
    if p6318.is_dir() {
        let inv = classify_nmr_workspace(&p6318).expect("classify 6318");
        eprintln!(
            "Real dataset 6318: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < 50.0,
            "6318 header scan took {:.2} ms (limit < 50 ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::SidechainNoesy);
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::C13Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hnca,
            PulseSequenceClass::Hncoca,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Ccconh,
            PulseSequenceClass::Hccconh,
            PulseSequenceClass::HcchTocsy,
            PulseSequenceClass::N15Noesy,
            PulseSequenceClass::C13Noesy,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == cls),
                "6318 missing {cls:?}"
            );
        }
        assert!(
            inv.experiments.iter().all(|e| {
                e.pulse_class != PulseSequenceClass::Unknown
                    || e.dimensionality == ExperimentDimensionality::Dim1D
            }),
            "6318 had unexpected 2D/3D Unknown experiments: {:?}",
            inv.experiments
        );
    }
}
