//! Clean-room NMR resonance assignment, spectrum I/O, peak picking, and simulation core.
//!
//! Provenance: Original implementation of the specifications in `docs/specs/01-data-io.md`,
//! `docs/specs/02-peak-picking.md`, `docs/specs/03-assignment.md`, and `docs/specs/05-benchmark.md`.

pub mod evaluate;
pub mod nmrstar;
pub mod peaks;
pub mod picker;
pub mod priors;
pub mod rng;
pub mod simulate;
pub mod solver;
pub mod spectrum;
pub mod spinsystems;
