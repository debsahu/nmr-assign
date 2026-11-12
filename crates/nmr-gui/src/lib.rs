//! Clean-room modular UX components for the `nmr-assign` interactive WebGPU & `egui` workbench.
//!
//! Provenance: Original clean-room human-computer interaction and state-machine architecture
//! (`docs/specs/06-gui-workbench.md`, `docs/specs/12-workbench-ux.md`). Exports decoupled
//! workflow navigation, fuzzy command-palette search, and keyboard accelerator modules for
//! native desktop and `wasm32-unknown-unknown` targets.

pub mod command_palette;
pub mod contour_painter;
pub mod ingest_panel;
pub mod keymap;
pub mod session_io;
pub mod strip_tiler;
pub mod workflow_bar;
