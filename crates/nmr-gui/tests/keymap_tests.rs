//! Deterministic state-transition unit & integration tests for `nmr_gui::keymap`,
//! `nmr_gui::command_palette`, and `nmr_gui::workflow_bar` (`T09.4`).
//!
//! Provenance: Original clean-room verification suite covering all required two-key NMR spectrum
//! chords, `450 ms` timeout boundaries, invalid second-key reset, `Esc` cancellation,
//! text-widget & Octet chemical editor focus suppression, `Cmd/Ctrl` modifier passthrough,
//! `egui` event extraction (`?` / `Shift+/`), and cross-module chord badge parity.

use eframe::egui;
use nmr_core::qc::DatasetQcVerdict;
use nmr_gui::command_palette::CommandPalette;
use nmr_gui::keymap::{
    ChordDispatcher, ChordKey, ChordOutcome, ChordState, KeyChordInput, KeyFocusContext,
    CONTOUR_FLOOR_STEP_RATIO, DEFAULT_CHORD_TIMEOUT_MS, SPECTRUM_ACCELERATORS,
};
use nmr_gui::workflow_bar::{draw_workbench_chrome, WorkbenchAction, WorkspaceReadiness};

const EXPECTED_TWO_KEY_CHORDS: &[((char, char), &str, WorkbenchAction)] = &[
    (
        ('v', 't'),
        "Chord: v_",
        WorkbenchAction::OpenContourThresholdPopup,
    ),
    (('p', 'a'), "Chord: p_", WorkbenchAction::AddPeakAtCrosshair),
    (
        ('p', 'd'),
        "Chord: p_",
        WorkbenchAction::DeleteSelectedPeaks,
    ),
    (
        ('p', 'c'),
        "Chord: p_",
        WorkbenchAction::SnapPeakParabolicCenter,
    ),
    (('p', 'l'), "Chord: p_", WorkbenchAction::TogglePeakLabels),
    (('o', 'z'), "Chord: o_", WorkbenchAction::ZoomFullSweep),
    (('z', 'i'), "Chord: z_", WorkbenchAction::ZoomIn),
    (('z', 'o'), "Chord: z_", WorkbenchAction::ZoomOut),
    (('c', 't'), "Chord: c_", WorkbenchAction::RaiseContourFloor),
    (('c', 'b'), "Chord: c_", WorkbenchAction::LowerContourFloor),
    (
        ('s', 't'),
        "Chord: s_",
        WorkbenchAction::TileSequentialStrips,
    ),
    (
        ('a', 't'),
        "Chord: a_",
        WorkbenchAction::OpenResidueAssignmentPopover,
    ),
];

#[test]
fn test_all_12_two_key_chords_and_pending_indicators() {
    let mut km = ChordDispatcher::new();
    assert_eq!(km.timeout_ms(), DEFAULT_CHORD_TIMEOUT_MS);
    assert_eq!(km.timeout_ms(), 450);
    assert!((CONTOUR_FLOOR_STEP_RATIO - 1.25).abs() < f32::EPSILON);

    let focus = KeyFocusContext::default();
    let mut t = 1_000u64;

    for &((k1, k2), expected_status, expected_action) in EXPECTED_TWO_KEY_CHORDS {
        assert_eq!(km.state(), ChordState::Idle);
        assert_eq!(km.status_indicator(t), None);

        // First letter transitions to Pending and exposes "Chord: <k1>_"
        let out1 = km.feed_key(KeyChordInput::plain_char(k1), t, focus);
        assert_eq!(out1, ChordOutcome::PendingPrefix { prefix: k1 });
        assert_eq!(
            km.state(),
            ChordState::Pending {
                prefix: k1,
                started_at_ms: t
            }
        );
        assert_eq!(
            km.status_indicator(t + 120).as_deref(),
            Some(expected_status)
        );

        // Second letter within 450 ms dispatches exact WorkbenchAction and returns to Idle
        let out2 = km.feed_key(KeyChordInput::plain_char(k2), t + 180, focus);
        assert_eq!(out2, ChordOutcome::Dispatched(expected_action));
        assert_eq!(out2.action(), Some(expected_action));
        assert_eq!(km.state(), ChordState::Idle);
        assert_eq!(km.status_indicator(t + 180), None);

        t += 1_000;
    }
}

#[test]
fn test_single_key_space_crosshair_lock_and_question_cheat_sheet() {
    let mut km = ChordDispatcher::new();
    let focus = KeyFocusContext::default();

    // Space toggles crosshair lock immediately without entering Pending
    let sp = km.feed_key(KeyChordInput::space(), 100, focus);
    assert_eq!(
        sp,
        ChordOutcome::Dispatched(WorkbenchAction::ToggleCrosshairLock)
    );
    assert_eq!(km.state(), ChordState::Idle);
    assert!(!km.cheat_sheet_open);

    // `?` toggles searchable cheat-sheet modal and emits OpenKeyboardCheatSheet
    let q1 = km.feed_key(KeyChordInput::question_mark(), 200, focus);
    assert_eq!(
        q1,
        ChordOutcome::Dispatched(WorkbenchAction::OpenKeyboardCheatSheet)
    );
    assert!(km.cheat_sheet_open);
    assert_eq!(km.state(), ChordState::Idle);

    // Esc while Idle closes open cheat-sheet modal
    let esc_close = km.feed_key(KeyChordInput::escape(), 250, focus);
    assert_eq!(esc_close, ChordOutcome::ClosedCheatSheet);
    assert!(!km.cheat_sheet_open);

    // Esc while Idle with cheat-sheet already closed is a clean no-op
    assert_eq!(
        km.feed_key(KeyChordInput::escape(), 300, focus),
        ChordOutcome::IdleEscape
    );
}

#[test]
fn test_exact_450ms_timeout_boundary_and_expiry_transitions() {
    let mut km = ChordDispatcher::new();
    let focus = KeyFocusContext::default();

    // 1. Exact boundary: dt == 450 ms MUST still succeed
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('p'), 1_000, focus),
        ChordOutcome::PendingPrefix { prefix: 'p' }
    );
    assert_eq!(
        km.peek_status_indicator(1_450).as_deref(),
        Some("Chord: p_")
    );
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('a'), 1_450, focus),
        ChordOutcome::Dispatched(WorkbenchAction::AddPeakAtCrosshair)
    );

    // 2. Just past boundary: dt == 451 ms MUST expire and NOT fire `pd`
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('p'), 2_000, focus),
        ChordOutcome::PendingPrefix { prefix: 'p' }
    );
    assert_eq!(km.peek_status_indicator(2_451), None);
    let out_expired = km.feed_key(KeyChordInput::plain_char('d'), 2_451, focus);
    assert_eq!(
        out_expired,
        ChordOutcome::ExpiredToIdle {
            expired_prefix: 'p'
        }
    );
    assert_eq!(km.state(), ChordState::Idle);

    // 3. Explicit tick_timeout clears stale status indicator
    km.feed_key(KeyChordInput::plain_char('c'), 3_000, focus);
    assert_eq!(km.tick_timeout(3_450), None);
    assert_eq!(km.tick_timeout(3_451), Some('c'));
    assert_eq!(km.state(), ChordState::Idle);

    // 4. Expired prefix followed by a new valid prefix ('a' at 451 ms after 'p') starts fresh 'a_'
    //    instead of firing 'pa', and then 't' completes 'at'!
    km.feed_key(KeyChordInput::plain_char('p'), 4_000, focus);
    let fresh_a = km.feed_key(KeyChordInput::plain_char('a'), 4_451, focus);
    assert_eq!(fresh_a, ChordOutcome::PendingPrefix { prefix: 'a' });
    assert_eq!(km.status_indicator(4_500).as_deref(), Some("Chord: a_"));
    let complete_at = km.feed_key(KeyChordInput::plain_char('t'), 4_600, focus);
    assert_eq!(
        complete_at,
        ChordOutcome::Dispatched(WorkbenchAction::OpenResidueAssignmentPopover)
    );

    // 5. Configurable custom timeout
    let mut fast_km = ChordDispatcher::with_timeout_ms(200);
    assert_eq!(fast_km.timeout_ms(), 200);
    fast_km.feed_key(KeyChordInput::plain_char('v'), 0, focus);
    assert_eq!(
        fast_km.feed_key(KeyChordInput::plain_char('t'), 201, focus),
        ChordOutcome::ExpiredToIdle {
            expired_prefix: 'v'
        }
    );
}

#[test]
fn test_invalid_second_key_resets_immediately_and_esc_cancels_pending() {
    let mut km = ChordDispatcher::new();
    let focus = KeyFocusContext::default();

    // Invalid second key ('x' after 'p') resets to Idle so subsequent 'a' cannot complete stale 'pa'
    km.feed_key(KeyChordInput::plain_char('p'), 100, focus);
    let inv = km.feed_key(KeyChordInput::plain_char('x'), 150, focus);
    assert_eq!(
        inv,
        ChordOutcome::InvalidChord {
            prefix: 'p',
            second: 'x'
        }
    );
    assert_eq!(km.state(), ChordState::Idle);
    assert_eq!(km.status_indicator(160), None);

    // Now pressing 'a' starts 'a_' rather than firing AddPeakAtCrosshair
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('a'), 200, focus),
        ChordOutcome::PendingPrefix { prefix: 'a' }
    );

    // Esc cancels pending 'a_' immediately (even if cheat-sheet is open, Esc cancels pending chord first)
    km.cheat_sheet_open = true;
    let cancelled = km.feed_key(KeyChordInput::escape(), 240, focus);
    assert_eq!(cancelled, ChordOutcome::CancelledPending { prefix: 'a' });
    assert_eq!(km.state(), ChordState::Idle);
    assert_eq!(km.status_indicator(245), None);
    assert!(km.cheat_sheet_open);

    // Unbound first key while Idle stays Idle
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('q'), 300, focus),
        ChordOutcome::UnboundKey
    );
    assert_eq!(km.state(), ChordState::Idle);
}

#[test]
fn test_focus_suppression_for_text_widgets_and_octet_chemical_studio() {
    let mut km = ChordDispatcher::new();
    let normal = KeyFocusContext::default();
    let text_focus = KeyFocusContext {
        text_focus_active: true,
        octet_editor_active: false,
    };
    let octet_focus = KeyFocusContext {
        text_focus_active: false,
        octet_editor_active: true,
    };

    // Start a pending chord in normal mode, then focus moves into a TextEdit -> cleared & suppressed
    km.feed_key(KeyChordInput::plain_char('p'), 100, normal);
    assert_eq!(km.status_indicator(110).as_deref(), Some("Chord: p_"));

    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('a'), 120, text_focus),
        ChordOutcome::SuppressedFocus
    );
    assert_eq!(km.state(), ChordState::Idle);
    assert_eq!(
        km.feed_key(KeyChordInput::space(), 130, text_focus),
        ChordOutcome::SuppressedFocus
    );
    assert_eq!(
        km.feed_key(KeyChordInput::question_mark(), 140, text_focus),
        ChordOutcome::SuppressedFocus
    );

    // Octet editor active also suppresses all single-key and two-key spectrum accelerators
    km.feed_key(KeyChordInput::plain_char('s'), 200, normal);
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('t'), 220, octet_focus),
        ChordOutcome::SuppressedFocus
    );
    assert_eq!(km.state(), ChordState::Idle);
    assert_eq!(
        km.feed_key(KeyChordInput::plain_char('v'), 240, octet_focus),
        ChordOutcome::SuppressedFocus
    );
}

#[test]
fn test_modifier_passthrough_for_cmd_ctrl_z_k_s_and_alt() {
    let mut km = ChordDispatcher::new();
    let focus = KeyFocusContext::default();

    // Pending half-chord 'p_' is cleanly cleared if user hits Cmd/Ctrl+Z, Cmd/Ctrl+K, or Cmd/Ctrl+S
    km.feed_key(KeyChordInput::plain_char('p'), 100, focus);
    assert_eq!(
        km.feed_key(KeyChordInput::with_cmd_or_ctrl('z', false), 150, focus),
        ChordOutcome::PassthroughModifier
    );
    assert_eq!(km.state(), ChordState::Idle);

    for (ch, shift) in [('z', false), ('z', true), ('k', false), ('s', false)] {
        let out = km.feed_key(KeyChordInput::with_cmd_or_ctrl(ch, shift), 200, focus);
        assert_eq!(out, ChordOutcome::PassthroughModifier);
        assert_eq!(km.state(), ChordState::Idle);
    }

    // Alt+key also passes through without entering Pending
    let alt_p = KeyChordInput {
        key: ChordKey::Letter('p'),
        ctrl_or_cmd: false,
        alt: true,
        shift: false,
    };
    assert_eq!(
        km.feed_key(alt_p, 300, focus),
        ChordOutcome::PassthroughModifier
    );
    assert_eq!(km.state(), ChordState::Idle);
}

#[test]
fn test_egui_question_mark_deduplication_and_cheat_sheet_search_parity() {
    // Verify Shift+/ + Event::Text("?") in the same frame produces exactly ONE QuestionMark token
    let shift_mod = egui::Modifiers {
        shift: true,
        ..egui::Modifiers::NONE
    };
    let events = vec![
        egui::Event::Key {
            key: egui::Key::Slash,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: shift_mod,
        },
        egui::Event::Text("?".to_string()),
    ];
    let extracted = nmr_gui::keymap::extract_egui_chord_inputs(&events);
    assert_eq!(extracted, vec![KeyChordInput::question_mark()]);

    // Verify cheat-sheet filtering and 1-to-1 parity with CommandPalette chord badges
    assert_eq!(ChordDispatcher::filter_accelerators("").len(), 14);
    let para = ChordDispatcher::filter_accelerators("parabolic");
    assert_eq!(para.len(), 1);
    assert_eq!(para[0].keys_label, "pc");

    let palette = CommandPalette::new();
    for binding in SPECTRUM_ACCELERATORS {
        let meta = binding.action.metadata();
        assert_eq!(
            meta.chord_badge,
            Some(binding.keys_label),
            "Mismatch on chord badge for {:?}",
            binding.action
        );
        let hits = palette.search(binding.keys_label);
        assert!(
            hits.iter().any(|h| h.command.action == binding.action),
            "CommandPalette search('{}') did not find {:?}",
            binding.keys_label,
            binding.action
        );
    }
}

#[test]
fn test_headless_egui_chrome_palette_cheat_sheet_and_two_frame_chord() {
    let ctx = egui::Context::default();
    let mut palette = CommandPalette::new();
    let mut keymap = ChordDispatcher::new();

    palette.open_palette();
    palette.set_query("pa");
    keymap.open_cheat_sheet();
    keymap.cheat_sheet_query = "contour".to_string();

    let readiness = WorkspaceReadiness {
        sequence_loaded: true,
        sequence_len: 76,
        spectra_2d_count: 1,
        spectra_3d_count: 4,
        picked_peaks_count: 340,
        spin_systems_count: 68,
        assigned_residues_count: 63,
        high_confidence_count: 59,
        qc_verdict: Some(DatasetQcVerdict::Warn),
        qc_summary: Some("multi-conformer crowding".to_string()),
        ..WorkspaceReadiness::default()
    };

    // Frame 1: render Top Menu Bar, Workflow Stepper, Command Palette (requests TextEdit focus),
    // and Cheat-Sheet cleanly; verify focused TextEdit suppresses spectrum chords in egui!
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = draw_workbench_chrome(ui, &readiness);
        let _ = palette.show(ui.ctx());
        let _ = keymap.show_cheat_sheet_window(ui.ctx());
    });
    assert!(ctx.egui_wants_keyboard_input());

    let mut raw_suppressed = egui::RawInput {
        time: Some(0.500),
        ..egui::RawInput::default()
    };
    raw_suppressed.events.push(egui::Event::Key {
        key: egui::Key::P,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run_ui(raw_suppressed, |ui| {
        // Palette is still open and its search TextEdit holds focus -> 'p' MUST be suppressed!
        let _ = palette.show(ui.ctx());
        let suppressed_acts = keymap.handle_egui_input(ui.ctx(), false);
        assert!(suppressed_acts.is_empty());
        assert_eq!(keymap.state(), ChordState::Idle);
    });

    // Close modals and run one clean frame (releasing Key::P) so egui releases modal widget focus
    palette.close_palette();
    keymap.close_cheat_sheet();
    let mut raw_release = egui::RawInput {
        time: Some(0.600),
        ..egui::RawInput::default()
    };
    raw_release.events.push(egui::Event::Key {
        key: egui::Key::P,
        physical_key: None,
        pressed: false,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run_ui(raw_release, |ui| {
        let _ = draw_workbench_chrome(ui, &readiness);
    });
    assert!(!ctx.text_edit_focused());

    // Frame 2: inject plain 'p' at t = 1.000 s (no widget focus) -> Pending 'p_'
    let mut raw_p = egui::RawInput {
        time: Some(1.000),
        ..egui::RawInput::default()
    };
    raw_p.events.push(egui::Event::Key {
        key: egui::Key::P,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let mut frame2_actions = Vec::new();
    let _ = ctx.run_ui(raw_p, |ui| {
        frame2_actions = keymap.handle_egui_input(ui.ctx(), false);
        keymap.draw_status_bar_indicator(ui, 1_000);
    });
    assert!(frame2_actions.is_empty());
    assert_eq!(
        keymap.peek_status_indicator(1_050).as_deref(),
        Some("Chord: p_")
    );

    // Frame 3: inject plain 'a' at t = 1.180 s -> dispatches AddPeakAtCrosshair
    let mut raw_a = egui::RawInput {
        time: Some(1.180),
        ..egui::RawInput::default()
    };
    raw_a.events.push(egui::Event::Key {
        key: egui::Key::A,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let mut frame3_actions = Vec::new();
    let _ = ctx.run_ui(raw_a, |ui| {
        frame3_actions = keymap.handle_egui_input(ui.ctx(), false);
    });
    assert_eq!(frame3_actions, vec![WorkbenchAction::AddPeakAtCrosshair]);
    assert_eq!(keymap.state(), ChordState::Idle);
}
