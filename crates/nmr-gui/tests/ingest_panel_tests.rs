//! Integration tests for `nmr_gui::ingest_panel` (`T09.2` Editable Experiment Selection Matrix
//! and cross-platform `DroppedFile` folder/archive/`.nef` ingestion).

use nmr_core::classify::{
    ClassifiedAxis, ClassifiedExperiment, DataState, ExperimentDimensionality, ExperimentVendor,
    NusSummary, ProblemRegime, PulseSequenceClass, QuadratureKind, WorkspaceInventory,
};
use nmr_core::session::{write_session_nef_str, WorkbenchSessionState};
use nmr_gui::ingest_panel::{
    ingest_dropped_files_native, ingest_dropped_payloads, DroppedPayload, InMemoryDroppedFile,
    IngestDropOutcome, IngestPanelState,
};

fn make_exp(
    rel_path: &str,
    dim: ExperimentDimensionality,
    data_state: DataState,
    mhz: f64,
    pulse_class: PulseSequenceClass,
    selected_by_default: bool,
) -> ClassifiedExperiment {
    ClassifiedExperiment {
        relative_path: rel_path.to_string(),
        display_name: rel_path.to_string(),
        vendor: ExperimentVendor::Bruker,
        dimensionality: dim,
        data_state,
        spectrometer_mhz: Some(mhz),
        axes: vec![ClassifiedAxis {
            label: "F2".to_string(),
            nucleus: "1H".to_string(),
            points: 1024,
            sw_ppm: Some(14.0),
            carrier_ppm: Some(4.7),
            obs_mhz: Some(mhz),
            quadrature: QuadratureKind::Dqd,
        }],
        quadrature_badge: "Echo-Antiecho".to_string(),
        nus: NusSummary::default(),
        pulse_class,
        pulse_program: Some("hsqcetf3gpsi".to_string()),
        title: None,
        raw_signal_bytes: 65536,
        processed_bytes: 0,
        selected_by_default,
        notes: Vec::new(),
    }
}

/// Build an uncompressed IEEE 1003.1 `ustar` archive in memory from `(path, payload)` entries.
fn build_ustar_archive(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(name, data) in files {
        let mut hdr = [0u8; 512];
        let name_bytes = name.as_bytes();
        let copy_len = name_bytes.len().min(100);
        hdr[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        hdr[100..108].copy_from_slice(b"0000644\0");
        hdr[108..116].copy_from_slice(b"0001000\0");
        hdr[116..124].copy_from_slice(b"0001000\0");
        let size_octal = format!("{:011o}\0", data.len());
        hdr[124..136].copy_from_slice(size_octal.as_bytes());
        hdr[136..148].copy_from_slice(b"14600000000\0");
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");

        let cksum: u32 = hdr.iter().map(|&b| u32::from(b)).sum();
        let chk_str = format!("{cksum:06o}\0 ");
        hdr[148..156].copy_from_slice(chk_str.as_bytes());

        out.extend_from_slice(&hdr);
        out.extend_from_slice(data);
        let pad = (512 - (data.len() % 512)) % 512;
        if pad > 0 {
            out.resize(out.len() + pad, 0);
        }
    }
    out.resize(out.len() + 1024, 0);
    out
}

/// Wrap raw bytes in a valid RFC 1952 GZIP + RFC 1951 stored-DEFLATE stream (zero extra crates).
fn wrap_gzip_stored(raw: &[u8]) -> Vec<u8> {
    fn crc32_ieee(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg() & 0xEDB8_8320;
                crc = (crc >> 1) ^ mask;
            }
        }
        !crc
    }

    let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03];
    let chunks: Vec<&[u8]> = if raw.is_empty() {
        vec![&[]]
    } else {
        raw.chunks(32_768).collect()
    };
    for (idx, chunk) in chunks.iter().enumerate() {
        let bfinal = u8::from(idx + 1 == chunks.len());
        gz.push(bfinal); // BFINAL | BTYPE=00 (stored)
        let len = chunk.len() as u16;
        let nlen = !len;
        gz.extend_from_slice(&len.to_le_bytes());
        gz.extend_from_slice(&nlen.to_le_bytes());
        gz.extend_from_slice(chunk);
    }
    gz.extend_from_slice(&crc32_ieee(raw).to_le_bytes());
    gz.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    gz
}

#[test]
fn test_ingest_panel_initialization_decoy_exclusion_and_override_persistence() {
    let inv = WorkspaceInventory {
        root_name: "protein_bundle".to_string(),
        experiments: vec![
            // Row 0: 1D pulsecal decoy (even if selected_by_default were true, 1D must be excluded!)
            make_exp(
                "1_pulsecal",
                ExperimentDimensionality::Dim1D,
                DataState::RawFid,
                600.13,
                PulseSequenceClass::Unknown,
                true,
            ),
            // Row 1: 2D 15N-HSQC on 600 MHz magnet
            make_exp(
                "10_hsqc",
                ExperimentDimensionality::Dim2D,
                DataState::Both,
                600.13,
                PulseSequenceClass::N15Hsqc,
                true,
            ),
            // Row 2: 3D HNCACB on 800 MHz magnet (multi-magnet bundle — never penalized!)
            make_exp(
                "20_hncacb",
                ExperimentDimensionality::Dim3D,
                DataState::Processed,
                800.13,
                PulseSequenceClass::Hncacb,
                true,
            ),
            // Row 3: Ambiguous custom pulse program initially classified Unknown (2D/3D)
            make_exp(
                "30_custom_cbca",
                ExperimentDimensionality::Dim3D,
                DataState::RawFid,
                800.13,
                PulseSequenceClass::Unknown,
                false,
            ),
        ],
        auxiliary_files: Vec::new(),
        problem_regime: ProblemRegime::GlobularBackbone,
        secondary_regimes: Vec::new(),
        missing_input_hints: vec!["No polymer sequence found in workspace.".to_string()],
        elapsed_ms: 4.2,
    };

    let mut panel = IngestPanelState::from_inventory(inv);

    // 1D & Unknown decoys default to unchecked; 2D/3D backbone experiments default to checked
    assert_eq!(panel.selected, vec![false, true, true, false]);
    assert_eq!(panel.selected_count(), 2);

    // Missing sequence hint present initially, cleared once user pastes sequence
    assert!(!panel.dynamic_missing_hints().is_empty());
    // Multi-magnet (600 MHz + 800 MHz) never generates any penalty hint
    assert!(panel
        .dynamic_missing_hints()
        .iter()
        .all(|h| !h.to_ascii_lowercase().contains("magnet") && !h.contains("MHz")));

    panel.sequence_text = ">ubq_mini\nMQIFVKTLTGK".to_string();
    assert!(panel.dynamic_missing_hints().is_empty());
    assert_eq!(panel.parsed_sequence_three_letter().len(), 11);

    // Override Row 3 (`Unknown` -> `Cbcaconh`): persists override and auto-includes the 3D row
    panel.override_pulse_class(3, PulseSequenceClass::Cbcaconh);
    assert_eq!(panel.effective_classes[3], PulseSequenceClass::Cbcaconh);
    assert!(panel.user_overridden[3]);
    assert!(panel.selected[3]);

    // Override Row 2 (`Hncacb` -> `Hnca`): persists across select_all_2d_3d and plan build
    panel.override_pulse_class(2, PulseSequenceClass::Hnca);
    panel.select_all_2d_3d();
    assert_eq!(panel.selected, vec![false, true, true, true]);

    let plan = panel.build_execution_plan();
    assert_eq!(plan.selected_experiments.len(), 3);
    assert_eq!(
        plan.selected_experiments[0].effective_class,
        PulseSequenceClass::N15Hsqc
    );
    assert_eq!(
        plan.selected_experiments[1].effective_class,
        PulseSequenceClass::Hnca
    );
    assert_eq!(plan.selected_experiments[1].solver_exp_name, Some("HNCA"));
    assert_eq!(
        plan.selected_experiments[2].effective_class,
        PulseSequenceClass::Cbcaconh
    );
    assert_eq!(
        plan.selected_experiments[2].solver_exp_name,
        Some("CBCACONH")
    );
}

#[test]
fn test_nef_session_drop_detection_native_and_wasm_payload() {
    let sess = WorkbenchSessionState::from_fasta_sequence("MiniUbq", ">u\nMQIFVKTL");
    let nef_bytes = write_session_nef_str(&sess).into_bytes();

    // 1. Cross-platform `wasm32`-compatible `DroppedPayload`
    let outcome = ingest_dropped_payloads(&[DroppedPayload::new(
        "mini_ubq_session.nef",
        nef_bytes.clone(),
    )])
    .expect("NEF payload ingest");
    match outcome {
        IngestDropOutcome::NefSession {
            source_name,
            nef_bytes: out_bytes,
        } => {
            assert_eq!(source_name, "mini_ubq_session.nef");
            assert_eq!(out_bytes, nef_bytes);
        }
        other => panic!("expected NefSession, got {other:?}"),
    }

    // 2. `egui::DroppedFile` trait handle (`InMemoryDroppedFile`)
    let handle = InMemoryDroppedFile::into_handle("restored.nef", nef_bytes.clone());
    let outcome2 = ingest_dropped_files_native(&[handle]).expect("DroppedFile NEF ingest");
    assert!(matches!(outcome2, IngestDropOutcome::NefSession { .. }));
}

#[test]
fn test_wasm_compatible_tar_and_targz_bundle_byte_ingestion() {
    let fasta = b">cisd3_frag\nMQIFVKTLTGKTITLEVEPS\n";
    let pulsecal_acqus = b"##TITLE= Parameter file\n##$PULPROG= <pulsecal>\n##$TD= 16384\n##$SW= 16.0\n##$SFO1= 800.13\n##$BF1= 800.13\n##$NUC1= <1H>\n";
    let hsqc_acqus = b"##TITLE= 15N-HSQC\n##$PULPROG= <hsqcetf3gpsi>\n##$PARMODE= 1\n##$TD= 2048\n##$SW= 14.0\n##$SFO1= 800.13\n##$BF1= 800.13\n##$NUC1= <1H>\n";
    let hsqc_acqu2s =
        b"##$TD= 256\n##$SW= 32.0\n##$SFO1= 81.08\n##$BF1= 81.08\n##$NUC1= <15N>\n##$FnMODE= 6\n";
    let hncacb_acqus = b"##TITLE= 3D HNCACB\n##$PULPROG= <hncacbgpwg3d>\n##$PARMODE= 2\n##$FnTYPE= 2\n##$TD= 2048\n##$SW= 14.0\n##$SFO1= 800.13\n##$BF1= 800.13\n##$NUC1= <1H>\n";
    let hncacb_acqu2s =
        b"##$TD= 128\n##$SW= 64.0\n##$SFO1= 201.20\n##$BF1= 201.20\n##$NUC1= <13C>\n##$FnMODE= 5\n";
    let hncacb_acqu3s =
        b"##$TD= 64\n##$SW= 32.0\n##$SFO1= 81.08\n##$BF1= 81.08\n##$NUC1= <15N>\n##$FnMODE= 6\n";
    let nuslist = b"0 0\n1 2\n3 1\n5 4\n";
    let dummy_ser = vec![0u8; 8192];

    let tar_bytes = build_ustar_archive(&[
        ("bundle/protein.fasta", fasta),
        ("bundle/1/acqus", pulsecal_acqus),
        ("bundle/1/fid", &dummy_ser[..1024]),
        ("bundle/10/acqus", hsqc_acqus),
        ("bundle/10/acqu2s", hsqc_acqu2s),
        ("bundle/10/ser", &dummy_ser),
        ("bundle/20/acqus", hncacb_acqus),
        ("bundle/20/acqu2s", hncacb_acqu2s),
        ("bundle/20/acqu3s", hncacb_acqu3s),
        ("bundle/20/nuslist", nuslist),
        ("bundle/20/ser", &dummy_ser),
    ]);

    // Test 1: Uncompressed `.tar` via `egui::DroppedFile` (`InMemoryDroppedFile`)
    let tar_handle = InMemoryDroppedFile::into_handle("spectrometer_bundle.tar", tar_bytes.clone());
    let tar_outcome = ingest_dropped_files_native(&[tar_handle]).expect("tar drop ingest");
    let panel_tar = match tar_outcome {
        IngestDropOutcome::Workspace(p) => p,
        other => panic!("expected Workspace outcome for .tar, got {other:?}"),
    };

    assert_eq!(
        panel_tar.inventory.problem_regime,
        ProblemRegime::GlobularBackbone
    );
    assert_eq!(panel_tar.parsed_sequence_three_letter().len(), 20);
    assert_eq!(panel_tar.inventory.experiments.len(), 3);

    // Verify 1D `pulsecal` is unchecked while 2D `N15Hsqc` and 3D `Hncacb` are checked
    for (idx, exp) in panel_tar.inventory.experiments.iter().enumerate() {
        if exp.dimensionality == ExperimentDimensionality::Dim1D {
            assert!(
                !panel_tar.selected[idx],
                "1D pulsecal decoy must be unchecked"
            );
        } else {
            assert!(
                panel_tar.selected[idx],
                "2D/3D backbone row must be checked"
            );
        }
    }

    // Test 2: GZIP `.tar.gz` via `wasm32`-compatible `ingest_dropped_payloads`
    let targz_bytes = wrap_gzip_stored(&tar_bytes);
    let targz_outcome = ingest_dropped_payloads(&[DroppedPayload::new(
        "spectrometer_bundle.tar.gz",
        targz_bytes,
    )])
    .expect("tar.gz byte ingest");
    let panel_gz = match targz_outcome {
        IngestDropOutcome::Workspace(p) => p,
        other => panic!("expected Workspace outcome for .tar.gz, got {other:?}"),
    };

    assert_eq!(panel_gz.inventory.experiments.len(), 3);
    assert_eq!(panel_gz.selected_count(), 2);
    assert_eq!(panel_gz.parsed_sequence_three_letter().len(), 20);
    assert!(panel_gz
        .inventory
        .experiments
        .iter()
        .any(|e| e.pulse_class == PulseSequenceClass::Hncacb && e.nus.is_nus));
}
