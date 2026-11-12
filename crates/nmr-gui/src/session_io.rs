//! Cross-platform GUI session controller, `Cmd/Ctrl+Z` / `Cmd/Ctrl+Shift+Z` shortcut
//! dispatch, and lossless `.nef` session I/O (native filesystem & `wasm32` in-memory bytes).
//!
//! Provenance: Clean-room original GUI adapter built on `eframe::egui` and `nmr_core::session`,
//! persisting sessions in standards-compliant NEF v1.1 (`save_nmrassign_workbench`) per
//! Gutmanas, A. et al., *Nature Structural & Molecular Biology* 22, 433–434 (2015).

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::session::{read_session_nef_file, write_session_nef_file};
use nmr_core::session::{
    read_session_nef_str, write_session_nef_str, CommandHistory, WorkbenchCommand,
    WorkbenchSessionState,
};

/// Session-level keyboard shortcut actions recognized in the NMR workbench.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionShortcutAction {
    /// Undo most recent workbench command (`Cmd/Ctrl + Z`).
    Undo,
    /// Redo most recently undone workbench command (`Cmd/Ctrl + Shift + Z` or `Cmd/Ctrl + Y`).
    Redo,
    /// Save current workbench session as `.nef` (`Cmd/Ctrl + S`).
    SaveSessionNef,
    /// Open a `.nef` workbench session (`Cmd/Ctrl + O`).
    OpenSessionNef,
}

/// Pure, deterministic classifier for modifier + key events.
///
/// When `suppress_undo_for_text_focus` is `true` (e.g. an `egui::TextEdit` or chemical sketcher
/// holds focus), `Undo` and `Redo` return `None` so widget-local undo is preserved.
pub fn classify_key_shortcut(
    modifiers: egui::Modifiers,
    key: egui::Key,
    suppress_undo_for_text_focus: bool,
) -> Option<SessionShortcutAction> {
    let cmd_or_ctrl = modifiers.command || modifiers.ctrl;
    if !cmd_or_ctrl || modifiers.alt {
        return None;
    }

    match (key, modifiers.shift) {
        (egui::Key::Z, true) | (egui::Key::Y, false) => {
            (!suppress_undo_for_text_focus).then_some(SessionShortcutAction::Redo)
        }
        (egui::Key::Z, false) => {
            (!suppress_undo_for_text_focus).then_some(SessionShortcutAction::Undo)
        }
        (egui::Key::S, false) => Some(SessionShortcutAction::SaveSessionNef),
        (egui::Key::O, false) => Some(SessionShortcutAction::OpenSessionNef),
        _ => None,
    }
}

/// Poll and consume `Cmd/Ctrl+Z`, `Cmd/Ctrl+Shift+Z`, `Cmd/Ctrl+Y`, `Cmd/Ctrl+S`, and `Cmd/Ctrl+O`
/// from `egui::Context`.
///
/// Checks `Shift+Z` prior to plain `Z` so `Cmd/Ctrl+Shift+Z` is never shadowed by `Cmd/Ctrl+Z`.
pub fn poll_session_shortcuts(
    ctx: &egui::Context,
    external_focus_suppress: bool,
) -> Option<SessionShortcutAction> {
    let text_focus = external_focus_suppress || ctx.egui_wants_keyboard_input();

    ctx.input_mut(|input| {
        if !text_focus {
            let redo_shift_z = egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                egui::Key::Z,
            );
            if input.consume_shortcut(&redo_shift_z) {
                return Some(SessionShortcutAction::Redo);
            }
            let redo_y = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
            if input.consume_shortcut(&redo_y) {
                return Some(SessionShortcutAction::Redo);
            }
            let undo_z = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
            if input.consume_shortcut(&undo_z) {
                return Some(SessionShortcutAction::Undo);
            }
        }

        let save_s = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
        if input.consume_shortcut(&save_s) {
            return Some(SessionShortcutAction::SaveSessionNef);
        }

        let open_o = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
        if input.consume_shortcut(&open_o) {
            return Some(SessionShortcutAction::OpenSessionNef);
        }

        None
    })
}

/// Serialize a [`WorkbenchSessionState`] to UTF-8 `.nef` bytes in memory (`native` & `wasm32`).
pub fn serialize_session_nef_bytes(state: &WorkbenchSessionState) -> Vec<u8> {
    write_session_nef_str(state).into_bytes()
}

/// Deserialize a [`WorkbenchSessionState`] from in-memory UTF-8 `.nef` bytes (`native` & `wasm32`).
pub fn deserialize_session_nef_bytes(bytes: &[u8]) -> Result<WorkbenchSessionState, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("NEF session buffer is not valid UTF-8: {e}"))?;
    read_session_nef_str(text)
}

/// Return `true` if `path_or_name` carries a `.nef` extension (case-insensitive).
pub fn is_nef_filename(path_or_name: &str) -> bool {
    path_or_name.trim().to_ascii_lowercase().ends_with(".nef")
}

/// Build a clean default `.nef` download/export filename from `protein_name`.
pub fn default_session_nef_filename(protein_name: &str) -> String {
    let stem: String = protein_name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = stem.trim_matches('_');
    if trimmed.is_empty() {
        "workbench_session.nef".to_string()
    } else {
        format!("{trimmed}_session.nef")
    }
}

/// Inspect a dropped `egui::DroppedFile` on native desktop and extract `(display_name, nef_bytes)` when `.nef`.
#[cfg(not(target_arch = "wasm32"))]
pub fn extract_dropped_nef_bytes(
    dropped: &dyn egui::DroppedFile,
) -> Result<Option<(String, Vec<u8>)>, String> {
    let p = dropped.path();
    let name = p.display().to_string();
    if !is_nef_filename(&name) {
        return Ok(None);
    }
    if p.exists() {
        let bytes = std::fs::read(p)
            .map_err(|e| format!("failed to read dropped NEF file {}: {e}", p.display()))?;
        return Ok(Some((name, bytes)));
    }
    let bytes = dropped
        .bytes()
        .map_err(|e| format!("failed to read dropped NEF bytes for '{name}': {e}"))?;
    Ok(Some((name, bytes.to_vec())))
}

/// Asynchronously extract `(display_name, nef_bytes)` from a browser-dropped `.nef` file on `wasm32`.
#[cfg(target_arch = "wasm32")]
pub async fn extract_dropped_nef_bytes_async(
    dropped: &dyn egui::DroppedFile,
) -> Result<Option<(String, Vec<u8>)>, String> {
    let name = dropped.path().display().to_string();
    if !is_nef_filename(&name) {
        return Ok(None);
    }
    let bytes = dropped
        .bytes_async()
        .await
        .map_err(|e| format!("failed to read dropped NEF bytes for '{name}': {e}"))?;
    Ok(Some((name, bytes.to_vec())))
}

/// Save a [`WorkbenchSessionState`] to a native filesystem path.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_session_nef_native(
    path: impl AsRef<Path>,
    state: &WorkbenchSessionState,
) -> Result<(), String> {
    write_session_nef_file(path, state)
}

/// Load a [`WorkbenchSessionState`] from a native filesystem path.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_session_nef_native(path: impl AsRef<Path>) -> Result<WorkbenchSessionState, String> {
    read_session_nef_file(path)
}

#[cfg(target_arch = "wasm32")]
fn percent_encode_uri_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Trigger a browser download of `nef_bytes` on `wasm32-unknown-unknown` using an ephemeral `<a download>` anchor.
#[cfg(target_arch = "wasm32")]
pub fn trigger_browser_nef_download(filename: &str, nef_bytes: &[u8]) -> Result<(), String> {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        type ClickableAnchor;
        #[wasm_bindgen(method)]
        fn click(this: &ClickableAnchor);
    }

    let window = web_sys::window().ok_or_else(|| "no browser Window available".to_string())?;
    let document = window
        .document()
        .ok_or_else(|| "no browser Document available".to_string())?;
    let anchor = document
        .create_element("a")
        .map_err(|e| format!("failed to create download anchor: {e:?}"))?;
    let data_uri = format!(
        "data:text/plain;charset=utf-8,{}",
        percent_encode_uri_bytes(nef_bytes)
    );
    anchor
        .set_attribute("href", &data_uri)
        .map_err(|e| format!("failed to set href attribute: {e:?}"))?;
    anchor
        .set_attribute("download", filename)
        .map_err(|e| format!("failed to set download attribute: {e:?}"))?;
    anchor
        .set_attribute("style", "display:none")
        .map_err(|e| format!("failed to set style attribute: {e:?}"))?;
    anchor.unchecked_ref::<ClickableAnchor>().click();
    Ok(())
}

/// Stateful GUI wrapper bundling [`WorkbenchSessionState`], [`CommandHistory`] (`256`-step ring buffer),
/// dirty-tracking, and `.nef` save/load operations across native desktop and `wasm32`.
#[derive(Clone, Debug, Default)]
pub struct WorkbenchSessionController {
    pub state: WorkbenchSessionState,
    pub history: CommandHistory,
    pub dirty: bool,
    pub active_source_name: Option<String>,
    pub last_status: Option<String>,
}

impl WorkbenchSessionController {
    /// Wrap an initial [`WorkbenchSessionState`] with a fresh `256`-step [`CommandHistory`].
    pub fn new(state: WorkbenchSessionState) -> Self {
        Self {
            state,
            history: CommandHistory::new(),
            dirty: false,
            active_source_name: None,
            last_status: None,
        }
    }

    /// Execute a reversible [`WorkbenchCommand`], marking the session dirty.
    pub fn execute(&mut self, cmd: WorkbenchCommand) -> String {
        let desc = self.history.execute(&mut self.state, cmd);
        self.dirty = true;
        self.last_status = Some(desc.clone());
        desc
    }

    /// Undo the most recent command (`Cmd/Ctrl+Z`).
    pub fn undo(&mut self) -> Option<String> {
        let desc = self.history.undo(&mut self.state)?;
        self.dirty = self.history.can_undo();
        let msg = format!("Undo: {desc}");
        self.last_status = Some(msg.clone());
        Some(msg)
    }

    /// Redo the most recently undone command (`Cmd/Ctrl+Shift+Z`).
    pub fn redo(&mut self) -> Option<String> {
        let desc = self.history.redo(&mut self.state)?;
        self.dirty = true;
        let msg = format!("Redo: {desc}");
        self.last_status = Some(msg.clone());
        Some(msg)
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    #[inline]
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Serialize the current session to a canonical NEF v1.1 string.
    pub fn save_to_nef_string(&mut self) -> String {
        let s = write_session_nef_str(&self.state);
        self.dirty = false;
        s
    }

    /// Serialize the current session to UTF-8 `.nef` bytes (`native` & `wasm32`).
    pub fn save_to_nef_bytes(&mut self) -> Vec<u8> {
        let bytes = serialize_session_nef_bytes(&self.state);
        self.dirty = false;
        bytes
    }

    /// Replace the active session from a NEF v1.1 string and reset undo/redo stacks cleanly.
    pub fn load_from_nef_str(
        &mut self,
        nef_text: &str,
        source_name: Option<String>,
    ) -> Result<String, String> {
        let loaded = read_session_nef_str(nef_text)?;
        let summary = format!(
            "Loaded session '{}' ({} residues, {} peaks, {} locks)",
            loaded.protein_name,
            loaded.sequence.len(),
            loaded.peaks.len(),
            loaded.locked_residues.len()
        );
        self.state = loaded;
        self.history.clear();
        self.dirty = false;
        self.active_source_name = source_name;
        self.last_status = Some(summary.clone());
        Ok(summary)
    }

    /// Replace the active session from in-memory UTF-8 `.nef` bytes (`native` & `wasm32`).
    pub fn load_from_nef_bytes(
        &mut self,
        bytes: &[u8],
        source_name: Option<String>,
    ) -> Result<String, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| format!("NEF session buffer is not valid UTF-8: {e}"))?;
        self.load_from_nef_str(text, source_name)
    }

    /// Save the active session to disk on native platforms.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_to_path(&mut self, path: impl AsRef<Path>) -> Result<String, String> {
        let p = path.as_ref();
        save_session_nef_native(p, &self.state)?;
        self.dirty = false;
        let display = p.display().to_string();
        self.active_source_name = Some(display.clone());
        let msg = format!("Saved NEF session to {display}");
        self.last_status = Some(msg.clone());
        Ok(msg)
    }

    /// Load an active session from disk on native platforms.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_path(&mut self, path: impl AsRef<Path>) -> Result<String, String> {
        let p = path.as_ref();
        let bytes = std::fs::read(p)
            .map_err(|e| format!("failed to read NEF session file {}: {e}", p.display()))?;
        self.load_from_nef_bytes(&bytes, Some(p.display().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::peaks::Peak;
    use nmr_core::session::SessionContourParams;

    #[test]
    fn test_classify_key_shortcuts_and_focus_suppression() {
        let cmd = egui::Modifiers {
            command: true,
            ..Default::default()
        };
        let cmd_shift = egui::Modifiers {
            command: true,
            shift: true,
            ..Default::default()
        };
        let ctrl = egui::Modifiers {
            ctrl: true,
            ..Default::default()
        };

        assert_eq!(
            classify_key_shortcut(cmd, egui::Key::Z, false),
            Some(SessionShortcutAction::Undo)
        );
        assert_eq!(
            classify_key_shortcut(ctrl, egui::Key::Z, false),
            Some(SessionShortcutAction::Undo)
        );
        assert_eq!(
            classify_key_shortcut(cmd_shift, egui::Key::Z, false),
            Some(SessionShortcutAction::Redo)
        );
        assert_eq!(
            classify_key_shortcut(ctrl, egui::Key::Y, false),
            Some(SessionShortcutAction::Redo)
        );
        assert_eq!(
            classify_key_shortcut(cmd, egui::Key::S, false),
            Some(SessionShortcutAction::SaveSessionNef)
        );
        assert_eq!(
            classify_key_shortcut(cmd, egui::Key::O, false),
            Some(SessionShortcutAction::OpenSessionNef)
        );

        // Text-focus suppression suppresses Undo/Redo while allowing Save/Open
        assert_eq!(classify_key_shortcut(cmd, egui::Key::Z, true), None);
        assert_eq!(classify_key_shortcut(cmd_shift, egui::Key::Z, true), None);
        assert_eq!(classify_key_shortcut(ctrl, egui::Key::Y, true), None);
        assert_eq!(
            classify_key_shortcut(cmd, egui::Key::S, true),
            Some(SessionShortcutAction::SaveSessionNef)
        );
    }

    #[test]
    fn test_controller_undo_redo_and_in_memory_and_native_nef_roundtrip() {
        let init = WorkbenchSessionState::from_fasta_sequence("Ubiquitin Mini", "MQIFVKTL");
        let mut ctrl = WorkbenchSessionController::new(init.clone());
        assert!(!ctrl.can_undo());
        assert!(!ctrl.can_redo());
        assert!(!ctrl.dirty);

        ctrl.execute(WorkbenchCommand::AddPeak(
            Peak::hsqc(8.4125, 121.34).with_intensity(15230.5, Some(98450.25)),
        ));
        ctrl.execute(WorkbenchCommand::LockResidue {
            spin_system: 0,
            residue_seq_idx: 1,
            locked: true,
        });
        let mut layer = SessionContourParams::new("HSQC");
        layer.base_level = 2.75;
        ctrl.execute(WorkbenchCommand::SetContourParams {
            layer_id: "HSQC".to_string(),
            params: layer,
        });

        assert!(ctrl.can_undo());
        assert!(ctrl.dirty);
        assert_eq!(ctrl.state.peaks.len(), 1);
        assert_eq!(ctrl.state.locked_residue_for_spin_system(0), Some(1));

        let snapshot = ctrl.state.clone();
        let nef_bytes = ctrl.save_to_nef_bytes();
        assert!(!ctrl.dirty);

        // Undo all 3 actions back to initial state, then redo all 3
        assert!(ctrl.undo().is_some());
        assert!(ctrl.undo().is_some());
        assert!(ctrl.undo().is_some());
        assert!(ctrl.state.bitwise_eq(&init));
        assert!(!ctrl.can_undo());
        assert!(ctrl.can_redo());

        assert!(ctrl.redo().is_some());
        assert!(ctrl.redo().is_some());
        assert!(ctrl.redo().is_some());
        assert!(ctrl.state.bitwise_eq(&snapshot));

        // Reload via in-memory bytes (`wasm32`-compatible path)
        let fname = default_session_nef_filename("Ubiquitin Mini");
        assert_eq!(fname, "ubiquitin_mini_session.nef");
        assert!(is_nef_filename(&fname));
        assert!(!is_nef_filename("ubiquitin.str"));

        let mut reloaded_ctrl = WorkbenchSessionController::default();
        reloaded_ctrl
            .load_from_nef_bytes(&nef_bytes, Some(fname.clone()))
            .expect("reload from bytes");
        assert!(reloaded_ctrl.state.bitwise_eq(&snapshot));
        assert!(!reloaded_ctrl.can_undo());
        assert!(!reloaded_ctrl.can_redo());

        // Native filesystem save/load path check
        let tmp = std::env::temp_dir().join(format!("nmr_gui_{}_{fname}", std::process::id()));
        ctrl.save_to_path(&tmp).expect("save_to_path");
        let mut disk_ctrl = WorkbenchSessionController::default();
        disk_ctrl.load_from_path(&tmp).expect("load_from_path");
        let _ = std::fs::remove_file(&tmp);
        assert!(disk_ctrl.state.bitwise_eq(&snapshot));
    }
}
