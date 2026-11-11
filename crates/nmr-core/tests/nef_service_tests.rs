//! Integration tests for Phase 5 Interchange & Reporting (`nef.rs`, `report.rs`)
//! and Phase 6 Hosted Pipeline Service (`service.rs`).

pub use nmr_core::*;

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use nmr_core::evaluate::{compare_shifts, to_reference};
use nmr_core::nmrstar::{read_star, read_star_str};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

use crate::nef::{
    classify_nef_atom_name, normalize_nef_atom_name, read_nef_file, read_nef_str, write_nef_file,
    write_nef_str, AssignedShift, NefAtomWildcardKind, NefDihedralRestraint,
    NefDihedralRestraintList, NefDistanceRestraint, NefDistanceRestraintList, NefDocument,
};
use crate::report::{
    extract_dihedrals_from_pdb, generate_quality_report_html, predict_csi_dihedrals, ReportInput,
    StructureReportSummary,
};
use crate::service::{spawn_background_server, JobRecord, JobSubmitRequest};

fn ubiquitin_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn temp_test_dir(tag: &str) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "nmr_nef_service_{}_{}_{}",
        tag,
        std::process::id(),
        ts
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

/// Minimal raw HTTP/1.1 client helper over `std::net::TcpStream` returning `(status_code, content_type, body_string)`.
fn raw_http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    json_body: Option<&str>,
) -> (u16, String, String) {
    let mut stream = TcpStream::connect(addr).expect("connect to background HTTP server");
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let body_bytes = json_body.map(|s| s.as_bytes()).unwrap_or(&[]);
    let req_head = if json_body.is_some() {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body_bytes.len()
        )
    } else {
        format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
    };
    stream.write_all(req_head.as_bytes()).unwrap();
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes).unwrap();
    }
    stream.flush().unwrap();

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let full = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = full
        .split_once("\r\n\r\n")
        .expect("valid HTTP header/body separator");

    let mut lines = head.lines();
    let status_line = lines.next().expect("HTTP status line");
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .expect("numeric HTTP status code");

    let mut content_type = String::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-type") {
                content_type = v.trim().to_string();
            }
        }
    }
    (status_code, content_type, body.to_string())
}

#[test]
fn test_nef_v1_1_lossless_roundtrip_all_saveframes() {
    let dir = temp_test_dir("nef_roundtrip");
    let ref_data = read_star(ubiquitin_fixture_path(), None).expect("read ubiquitin fixture");
    assert_eq!(ref_data.sequence.len(), 76);

    // Check NEF v1.1 wildcard / prochiral atom helper classification
    assert_eq!(classify_nef_atom_name("CA"), NefAtomWildcardKind::Exact);
    assert_eq!(
        classify_nef_atom_name("HB%"),
        NefAtomWildcardKind::DegenerateSet
    );
    assert_eq!(
        classify_nef_atom_name("HD1%"),
        NefAtomWildcardKind::DegenerateSet
    );
    assert_eq!(
        classify_nef_atom_name("HBx"),
        NefAtomWildcardKind::NonStereoProchiral
    );
    assert_eq!(normalize_nef_atom_name("LEU", "HD1*"), "HD1%");
    assert_eq!(normalize_nef_atom_name("ALA", "HB"), "HB%");

    // Simulate 2D + 3D triple-resonance peaks (including signed HNCACB -CB) + 3D 15N-NOESY peak
    let (mut peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 7,
            ..SimConfig::default()
        },
    );
    peaks.push(Peak::noesy_3d("NOESY15N", 8.429, 121.934, 4.185, 2450.0));

    let mut conf = BTreeMap::new();
    for k in 0..ref_data.sequence.len() {
        conf.insert(k, 0.95);
    }

    let mut doc =
        NefDocument::from_reference_and_peaks("ubiquitin_v11", &ref_data, Some(&conf), &peaks);

    // Add explicit stereo / ambiguous sidechain shifts (`HB%`, `HD1%`, `HBx`) to verify lossless preservation
    doc.shift_lists[0].shifts.push(AssignedShift {
        chain_code: "A".to_string(),
        sequence_code: Some(8),
        residue_name: "LEU".to_string(),
        atom_name: "HD1%".to_string(),
        element: Some("H".to_string()),
        isotope_number: Some(1),
        value: 0.842,
        value_uncertainty: Some(0.015),
    });
    doc.shift_lists[0].shifts.push(AssignedShift {
        chain_code: "A".to_string(),
        sequence_code: Some(28),
        residue_name: "ALA".to_string(),
        atom_name: "HB%".to_string(),
        element: Some("H".to_string()),
        isotope_number: Some(1),
        value: 1.385,
        value_uncertainty: Some(0.012),
    });
    doc.shift_lists[0].shifts.push(AssignedShift {
        chain_code: "A".to_string(),
        sequence_code: Some(4),
        residue_name: "PHE".to_string(),
        atom_name: "HBx".to_string(),
        element: Some("H".to_string()),
        isotope_number: Some(1),
        value: 2.995,
        value_uncertainty: Some(0.020),
    });

    // Add unambiguous + ambiguous OR (`restraint_id = 2`, two rows) + combined (`restraint_combination_id = Some(1)`) distance restraints
    doc.distance_restraint_lists.push(NefDistanceRestraintList {
        name: "default".to_string(),
        potential_type: "upper-bound-parabolic".to_string(),
        restraints: vec![
            NefDistanceRestraint::new(
                1,
                1,
                None,
                "A",
                2,
                "GLN",
                "H",
                "A",
                16,
                "GLU",
                "H",
                Some(2.90),
                Some(1.80),
                3.60,
                1.0,
            ),
            NefDistanceRestraint::new(
                2,
                2,
                None,
                "A",
                3,
                "ILE",
                "HD1%",
                "A",
                8,
                "LEU",
                "HD1%",
                Some(3.80),
                Some(1.80),
                5.00,
                1.0,
            ),
            NefDistanceRestraint::new(
                3,
                2,
                None,
                "A",
                3,
                "ILE",
                "HD1%",
                "A",
                15,
                "LEU",
                "HD2%",
                Some(3.80),
                Some(1.80),
                5.00,
                1.0,
            ),
            NefDistanceRestraint::new(
                4,
                3,
                Some(1),
                "A",
                4,
                "PHE",
                "HBx",
                "A",
                6,
                "LYS",
                "H",
                Some(4.20),
                Some(1.80),
                5.50,
                0.75,
            ),
        ],
    });

    // Add PHI and PSI dihedral restraints
    doc.dihedral_restraint_lists.push(NefDihedralRestraintList {
        name: "default".to_string(),
        potential_type: "square-well-parabolic".to_string(),
        restraints: vec![
            NefDihedralRestraint::phi(1, 1, "A", 25, "ASN", 26, "VAL", -62.5, 18.0, 1.0),
            NefDihedralRestraint::psi(2, 2, "A", 26, "VAL", 27, "LYS", -41.0, 18.0, 1.0),
        ],
    });

    let nef_path = dir.join("ubiquitin_bundle.nef");
    write_nef_file(&nef_path, &doc).expect("write NEF file");
    let raw_nef = fs::read_to_string(&nef_path).unwrap();

    // Verify canonical NEF v1.1 saveframe headers & tags
    for expected in [
        "data_nef_ubiquitin_v11",
        "save_nef_nmr_meta_data",
        "_nef_nmr_meta_data.format_name           nmr_exchange_format",
        "_nef_nmr_meta_data.format_version        1.1",
        "save_nef_molecular_system",
        "save_nef_chemical_shift_list_default",
        "save_nef_nmr_spectrum_HNCACB",
        "save_nef_distance_restraint_list_default",
        "save_nef_dihedral_restraint_list_default",
    ] {
        assert!(
            raw_nef.contains(expected),
            "missing expected NEF block {expected:?}"
        );
    }

    let parsed = read_nef_file(&nef_path).expect("parse NEF file");
    assert_eq!(parsed.project_name, "ubiquitin_v11");
    assert_eq!(parsed.meta_data.format_name, "nmr_exchange_format");
    assert_eq!(parsed.meta_data.format_version, "1.1");
    assert_eq!(parsed.molecular_system.residues.len(), 76);
    assert_eq!(parsed.molecular_system.residues[0].linking, "start");
    assert_eq!(parsed.molecular_system.residues[75].linking, "end");

    // Verify chemical shift round-trip (including HB%, HD1%, HBx)
    assert_eq!(
        parsed.shift_lists[0].shifts.len(),
        doc.shift_lists[0].shifts.len()
    );
    assert!(parsed.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.residue_name == "LEU"
            && s.atom_name == "HD1%"
            && (s.value - 0.842).abs() < 1e-4));
    assert!(parsed.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.residue_name == "ALA"
            && s.atom_name == "HB%"
            && (s.value - 1.385).abs() < 1e-4));

    // Verify Reference conversion preserves all 76 residues and backbone shifts within 1e-3 ppm
    let back_ref = parsed.to_reference().expect("convert NEF to Reference");
    assert_eq!(back_ref.sequence, ref_data.sequence);
    assert_eq!(back_ref.seq_ids, ref_data.seq_ids);
    for (key, &orig_v) in &ref_data.shifts {
        let got_v = back_ref
            .shifts
            .get(key)
            .copied()
            .unwrap_or_else(|| panic!("missing shift {key:?} after NEF roundtrip"));
        assert!((got_v - orig_v).abs() < 1e-3);
    }

    // Verify peak list round-trip (including negative sign on HNCACB CB peaks and 3D NOESY15N)
    let back_peaks = parsed.to_peaks();
    assert_eq!(back_peaks.len(), peaks.len());
    let orig_neg_cb = peaks
        .iter()
        .filter(|p| p.exp == "HNCACB" && p.sign < 0)
        .count();
    let back_neg_cb = back_peaks
        .iter()
        .filter(|p| p.exp == "HNCACB" && p.sign < 0)
        .count();
    assert!(orig_neg_cb > 40);
    assert_eq!(orig_neg_cb, back_neg_cb);
    let back_noe = back_peaks
        .iter()
        .find(|p| p.exp == "NOESY15N")
        .expect("NOESY15N peak roundtrip");
    assert!((back_noe.h2.unwrap() - 4.185).abs() < 1e-3);

    // Verify ambiguous OR & combined distance restraints
    let d_rests = &parsed.distance_restraint_lists[0].restraints;
    assert_eq!(d_rests.len(), 4);
    assert_eq!(d_rests[1].restraint_id, 2);
    assert_eq!(d_rests[2].restraint_id, 2);
    assert_eq!(d_rests[1].restraint_combination_id, None);
    assert_eq!(d_rests[1].atom_name_1, "HD1%");
    assert_eq!(d_rests[2].atom_name_2, "HD2%");
    assert_eq!(d_rests[3].restraint_combination_id, Some(1));
    assert!((d_rests[3].weight - 0.75).abs() < 1e-3);

    // Verify PHI & PSI dihedral restraints
    let t_rests = &parsed.dihedral_restraint_lists[0].restraints;
    assert_eq!(t_rests.len(), 2);
    assert_eq!(t_rests[0].name, "PHI");
    assert!((t_rests[0].target_value - (-62.5)).abs() < 1e-2);
    assert_eq!(t_rests[1].name, "PSI");
    assert!((t_rests[1].target_value - (-41.0)).abs() < 1e-2);

    // Second write -> read idempotence check
    let rewound = read_nef_str(&write_nef_str(&parsed)).expect("re-parse written NEF");
    assert_eq!(rewound.molecular_system, parsed.molecular_system);
    assert_eq!(
        rewound.distance_restraint_lists,
        parsed.distance_restraint_lists
    );
    assert_eq!(
        rewound.dihedral_restraint_lists,
        parsed.dihedral_restraint_lists
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_quality_report_html_and_svg_generation() {
    let ref_data = read_star(ubiquitin_fixture_path(), None).unwrap();
    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let pri = Priors::default();
    let res = assign(
        &systems,
        &ref_data.sequence,
        &pri,
        &ScoreConfig::default(),
        &AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let (pred_ref, conf) = to_reference(
        &res,
        &systems,
        &ref_data.sequence,
        &ref_data.seq_ids,
        0.0,
        Some(&pri),
    );

    let csi_dihedrals = predict_csi_dihedrals(&pred_ref);
    assert!(
        csi_dihedrals.len() >= 60,
        "expected CSI dihedral estimates across ubiquitin, got {}",
        csi_dihedrals.len()
    );

    let mut input = ReportInput::new("Ubiquitin_QC", pred_ref, conf);
    input.n_peaks = Some(peaks.len());
    input.n_spin_systems = Some(systems.len());
    input.solver_score = Some(res.score);
    input.elapsed_secs = Some(0.042);
    input.engine_label = Some("CPU (12 replicas)".to_string());
    input.systems_and_assignment = Some((systems, res));
    input.dihedrals = csi_dihedrals;
    input.structure_summary = Some(StructureReportSummary {
        n_conformers: 10,
        backbone_rmsd_angstrom: 1.18,
        heavy_atom_rmsd_angstrom: Some(1.84),
        n_distance_restraints: 640,
        n_ambiguous_restraints: 185,
        n_dihedral_restraints: 104,
        distance_violations_gt_05a: 1,
        dihedral_violations_gt_5deg: 2,
        ramachandran_favored_pct: 96.2,
    });

    let html = generate_quality_report_html(&input);
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("</html>"));
    // Verify 100% self-contained: zero external stylesheets, scripts, fonts, or network URLs (only W3C SVG xmlns)
    assert!(!html.contains("<script src="));
    assert!(!html.contains("<link "));
    assert!(!html.contains("https://"));

    // Verify executive metrics, structure summary, and all 5 required inline SVG charts
    for marker in [
        "id=\"executive-metrics\"",
        "id=\"structure-summary\"",
        "1.18 &Aring;",
        "id=\"svg-confidence-ribbon\"",
        "id=\"svg-confidence-histogram\"",
        "id=\"svg-secondary-shifts\"",
        "id=\"svg-strip-connectivity\"",
        "id=\"svg-ramachandran\"",
        "class=\"shift-table\"",
    ] {
        assert!(html.contains(marker), "report HTML missing {marker}");
    }
    assert_eq!(html.matches("<svg ").count(), 5);
    assert!(
        html.matches("<title>").count() >= 150,
        "expected rich interactive SVG tooltips"
    );

    // Also verify PDB coordinate dihedral extraction for Ramachandran plotting
    let mini_pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C\n\
ATOM      4  N   GLU A   2       1.251   2.490   0.000  1.00 20.00           N\n\
ATOM      5  CA  GLU A   2       1.680   3.880   0.000  1.00 20.00           C\n\
ATOM      6  C   GLU A   2       0.920   4.620   1.090  1.00 20.00           C\n\
ATOM      7  N   VAL A   3       1.380   5.820   1.410  1.00 20.00           N\n\
ATOM      8  CA  VAL A   3       0.760   6.640   2.440  1.00 20.00           C\n\
ATOM      9  C   VAL A   3       1.720   7.740   2.860  1.00 20.00           C\n\
ENDMDL\n";
    let pdb_dih = extract_dihedrals_from_pdb(mini_pdb);
    assert_eq!(pdb_dih.len(), 1);
    assert_eq!(pdb_dih[0].seq_id, 2);
    assert_eq!(pdb_dih[0].residue, "GLU");
    assert!(pdb_dih[0].phi_deg.is_finite() && pdb_dih[0].psi_deg.is_finite());

    // Verify `generate_quality_report_html` automatically populates Ramachandran points from `pdb_coordinates`
    let mut pdb_input = ReportInput::new(
        "PDB_Coords_Report",
        input.reference.clone(),
        BTreeMap::new(),
    );
    pdb_input.pdb_coordinates = Some(mini_pdb.to_string());
    let pdb_html = generate_quality_report_html(&pdb_input);
    assert!(pdb_html.contains("1 backbone (&phi;, &psi;) torsion pairs plotted"));
}

#[test]
fn test_http_service_end_to_end_ubiquitin_job_and_dashboard() {
    let storage = temp_test_dir("http_service");
    let mut server =
        spawn_background_server("127.0.0.1:0", storage.clone()).expect("spawn background server");
    let addr = server.local_addr();
    assert_ne!(addr.port(), 0);

    // 1. Health endpoint
    let (code, ctype, body) = raw_http_request(addr, "GET", "/api/v1/health", None);
    assert_eq!(code, 200);
    assert!(ctype.contains("application/json"));
    let health: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["version"], "0.2.0");

    // 2. Browser dashboard UI (`GET /`)
    let (code_ui, ctype_ui, ui_html) = raw_http_request(addr, "GET", "/", None);
    assert_eq!(code_ui, 200);
    assert!(ctype_ui.contains("text/html"));
    assert!(ui_html.contains("Load Synthetic Ubiquitin Demo"));

    // 3. Built-in ubiquitin demo fixture endpoint
    let (code_demo, _, demo_body) = raw_http_request(addr, "GET", "/api/v1/demo/ubiquitin", None);
    assert_eq!(code_demo, 200);
    let demo_json: serde_json::Value = serde_json::from_str(&demo_body).unwrap();
    let seq_str = demo_json["sequence"].as_str().unwrap().to_string();
    let peaks_tsv = demo_json["peaks_tsv"].as_str().unwrap().to_string();
    assert_eq!(seq_str.len(), 76);

    // 4. Reject invalid empty job with 400 Bad Request
    let (code_bad, _, _) = raw_http_request(addr, "POST", "/api/v1/jobs", Some("{}"));
    assert_eq!(code_bad, 400);

    // 5. Submit real end-to-end ubiquitin assignment job via `POST /api/v1/jobs`
    let submit_payload = serde_json::to_string(&JobSubmitRequest {
        project_name: Some("ubiquitin_e2e".to_string()),
        sequence: Some(format!(">ubiquitin_1_76\n{seq_str}")),
        star_text: None,
        nef_text: None,
        peaks_tsv: Some(peaks_tsv),
        n_runs: Some(16),
        steps_per_system: Some(1500),
        seed: Some(0),
        first_seq_id: Some(1),
    })
    .unwrap();

    let (code_sub, _, sub_body) =
        raw_http_request(addr, "POST", "/api/v1/jobs", Some(&submit_payload));
    assert_eq!(code_sub, 202, "expected 202 Accepted, got body={sub_body}");
    let sub_json: serde_json::Value = serde_json::from_str(&sub_body).unwrap();
    let job_id = sub_json["job_id"].as_str().unwrap().to_string();
    assert_eq!(sub_json["status"], "queued");

    // 6. Poll `GET /api/v1/jobs/<id>` until completed
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut final_rec: Option<JobRecord> = None;
    while Instant::now() < deadline {
        let (c, _, b) = raw_http_request(addr, "GET", &format!("/api/v1/jobs/{job_id}"), None);
        assert_eq!(c, 200);
        let rec: JobRecord = serde_json::from_str(&b).unwrap();
        if rec.status == "completed" {
            final_rec = Some(rec);
            break;
        }
        assert_ne!(
            rec.status, "failed",
            "job failed unexpectedly: {:?}",
            rec.error
        );
        thread::sleep(Duration::from_millis(25));
    }
    let rec = final_rec.expect("ubiquitin job did not complete within timeout");
    let metrics = rec.metrics.expect("completed job metrics");
    assert_eq!(metrics.n_residues, 76);
    assert!(metrics.n_assigned >= 65, "metrics={metrics:?}");
    assert!(metrics.mean_confidence >= 0.85, "metrics={metrics:?}");

    // Verify `GET /api/v1/jobs` lists the completed job
    let (code_list, _, list_body) = raw_http_request(addr, "GET", "/api/v1/jobs", None);
    assert_eq!(code_list, 200);
    let list_json: serde_json::Value = serde_json::from_str(&list_body).unwrap();
    assert_eq!(list_json["jobs"].as_array().unwrap().len(), 1);

    // 7. Fetch all 3 generated artifacts over HTTP & verify Content-Type + contents
    let (c_str, ct_str, shifts_str) = raw_http_request(
        addr,
        "GET",
        &format!("/api/v1/jobs/{job_id}/shifts.str"),
        None,
    );
    assert_eq!(c_str, 200);
    assert!(ct_str.contains("text/plain"));

    let (c_nef, ct_nef, result_nef) = raw_http_request(
        addr,
        "GET",
        &format!("/api/v1/jobs/{job_id}/result.nef"),
        None,
    );
    assert_eq!(c_nef, 200);
    assert!(ct_nef.contains("text/plain"));

    let (c_rep, ct_rep, report_html) = raw_http_request(
        addr,
        "GET",
        &format!("/api/v1/jobs/{job_id}/report.html"),
        None,
    );
    assert_eq!(c_rep, 200);
    assert!(ct_rep.contains("text/html"));
    assert_eq!(report_html.matches("<svg ").count(), 5);

    // 8. Verify >= 95% chemical shift accuracy on BOTH returned shifts.str AND result.nef!
    let truth_ref = read_star(ubiquitin_fixture_path(), None).unwrap();
    let pred_from_star = read_star_str(&shifts_str, None).expect("parse returned shifts.str");
    assert_eq!(pred_from_star.sequence, truth_ref.sequence);

    let cmp_star = compare_shifts(&pred_from_star, &truth_ref);
    let (ok_total, n_total) = cmp_star
        .values()
        .fold((0usize, 0usize), |(acc_ok, acc_n), &(ok, n)| {
            (acc_ok + ok, acc_n + n)
        });
    let shift_acc = (ok_total as f64) / (n_total.max(1) as f64);
    println!(
        "E2E HTTP ubiquitin job {job_id}: assigned={}/{}, HC={}, mean_conf={:.3}, shift_acc={:.2}% ({ok_total}/{n_total}), runtime={:.3}s",
        metrics.n_assigned,
        metrics.n_assignable,
        metrics.n_high_confidence,
        metrics.mean_confidence,
        shift_acc * 100.0,
        metrics.elapsed_secs
    );
    assert!(
        shift_acc >= 0.95,
        "expected >= 95% shift accuracy on returned shifts.str, got {:.2}% ({ok_total}/{n_total}): {cmp_star:?}",
        shift_acc * 100.0
    );

    let parsed_nef = read_nef_str(&result_nef).expect("parse returned result.nef");
    let pred_from_nef = parsed_nef.to_reference().unwrap();
    let cmp_nef = compare_shifts(&pred_from_nef, &truth_ref);
    let (nef_ok, nef_n) = cmp_nef
        .values()
        .fold((0usize, 0usize), |(a, b), &(ok, n)| (a + ok, b + n));
    let nef_acc = (nef_ok as f64) / (nef_n.max(1) as f64);
    assert!(
        nef_acc >= 0.95,
        "expected >= 95% shift accuracy on returned result.nef, got {:.2}% ({nef_ok}/{nef_n})",
        nef_acc * 100.0
    );
    assert!(
        !parsed_nef.dihedral_restraint_lists.is_empty(),
        "expected CSI dihedral restraints in result.nef"
    );

    // 9. Also verify NEF-in (`nef_text` only) job submission works end-to-end
    let nef_job_payload = serde_json::to_string(&JobSubmitRequest {
        project_name: Some("ubiquitin_from_nef".to_string()),
        nef_text: Some(result_nef),
        n_runs: Some(12),
        steps_per_system: Some(1200),
        seed: Some(1),
        ..JobSubmitRequest::default()
    })
    .unwrap();
    let (c_nef_sub, _, nef_sub_body) =
        raw_http_request(addr, "POST", "/api/v1/jobs", Some(&nef_job_payload));
    assert_eq!(c_nef_sub, 202);
    let nef_job_id = serde_json::from_str::<serde_json::Value>(&nef_sub_body).unwrap()["job_id"]
        .as_str()
        .unwrap()
        .to_string();
    let mut nef_done = false;
    let deadline2 = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline2 {
        let (_, _, b) = raw_http_request(addr, "GET", &format!("/api/v1/jobs/{nef_job_id}"), None);
        let r: JobRecord = serde_json::from_str(&b).unwrap();
        if r.status == "completed" {
            nef_done = true;
            break;
        }
        assert_ne!(r.status, "failed", "NEF-in job failed: {:?}", r.error);
        thread::sleep(Duration::from_millis(25));
    }
    assert!(nef_done, "NEF-in job did not complete in time");

    // 10. Verify path-traversal rejection, clean shutdown, and storage reload across server restarts
    let (c_trav, _, _) = raw_http_request(addr, "GET", "/api/v1/jobs/..%2Fetc/shifts.str", None);
    assert_eq!(c_trav, 400);

    server.shutdown();

    let mut restarted =
        spawn_background_server("127.0.0.1:0", storage.clone()).expect("restart server on storage");
    let (c_reload, _, reload_body) =
        raw_http_request(restarted.local_addr(), "GET", "/api/v1/jobs", None);
    assert_eq!(c_reload, 200);
    let reload_json: serde_json::Value = serde_json::from_str(&reload_body).unwrap();
    assert_eq!(reload_json["jobs"].as_array().unwrap().len(), 2);
    restarted.shutdown();

    let _ = fs::remove_dir_all(&storage);
}
