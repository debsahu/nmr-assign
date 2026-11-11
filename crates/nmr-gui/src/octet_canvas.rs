//! Interactive `egui` Chemical Structure, Reaction Scheme, Clipware & 1D NMR Studio (`OctetDrawPanel`).
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis (`edit.ts`, `snap.ts`, `hit.ts`, `history.ts`, `keys.ts`, `tools.tsx`, `pageplot.ts`).
//!
//! Cross-platform native desktop + WebAssembly (`wasm32-unknown-unknown`) implementation:
//! uses zero `std::fs` and zero `std::time::Instant`.
//!
//! Exposes [`OctetDrawPanel`] (aliased as [`OctetDrawApp`]) with `pub fn ui(&mut self, ui: &mut egui::Ui)`,
//! `pub fn set_loaded_protein_sequence(&mut self, residues: Vec<String>)`, and pluggable
//! [`OctetBackendAdapter`] hooks so `integrator` can wire `octet_mol` + `octet_analyt` + `octet_render`
//! directly into `crates/nmr-gui/src/main.rs`.

use eframe::egui;
use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

use nmr_core::octet_render::{
    build_peptide_scene, chair_sprout_vectors, compile_scene_primitives, expand_multiplet_lines_hz,
    flatten_path_commands, fuse_polygon_on_bond, implicit_hydrogens, parse_peptide_sequence,
    perceive_small_rings, regular_polygon_scene, render_scene_pdf, render_scene_png,
    render_scene_svg, ArrowKind, BondOrder, BondStereo, BracketStyle, DocumentStyle, JournalStyle,
    OrbitalShading, OrbitalType, PushingArrowKind, RenderScene, RgbaColor, SceneAtom, SceneBond,
    SceneBracket, SceneCurvedArrow, SceneOrbital, SceneReactionArrow, SceneSpectrum1D, SceneText,
    SpectrumSignal1D, TemplateCategory, TemplateLibrary, VectorPrimitive, HIGHLIGHT_PALETTE,
};

/// Top-level alias so callers can instantiate either `OctetDrawPanel` or `OctetDrawApp`.
pub type OctetDrawApp = OctetDrawPanel;

// ============================================================================
// 1. Pluggable Backend Hooks for `octet_mol` & `octet_analyt` Integration
// ============================================================================

/// Supported chemical & publication import/export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoFormat {
    OctJson,
    Cdxml,
    CdxHex,
    MolV2000,
    Sdf,
    Smiles,
    InChI,
    JcampDx,
    Svg,
    Pdf,
    Png,
}

impl IoFormat {
    pub const ALL: &'static [IoFormat] = &[
        IoFormat::Cdxml,
        IoFormat::CdxHex,
        IoFormat::OctJson,
        IoFormat::MolV2000,
        IoFormat::Sdf,
        IoFormat::Smiles,
        IoFormat::InChI,
        IoFormat::JcampDx,
        IoFormat::Svg,
        IoFormat::Pdf,
        IoFormat::Png,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::OctJson => "OCT (.oct JSON)",
            Self::Cdxml => "CDXML (.cdxml)",
            Self::CdxHex => "CDX Binary (.cdx hex)",
            Self::MolV2000 => "MOL V2000 (.mol)",
            Self::Sdf => "SDF (.sdf)",
            Self::Smiles => "SMILES (.smi)",
            Self::InChI => "InChI",
            Self::JcampDx => "JCAMP-DX 1D NMR (.jdx)",
            Self::Svg => "SVG Vector (.svg)",
            Self::Pdf => "PDF 1.4 Vector (.pdf)",
            Self::Png => "PNG Raster (.png)",
        }
    }

    pub fn can_import(self) -> bool {
        matches!(
            self,
            Self::OctJson
                | Self::Cdxml
                | Self::CdxHex
                | Self::MolV2000
                | Self::Sdf
                | Self::Smiles
                | Self::InChI
                | Self::JcampDx
        )
    }
}

/// Live molecular formula, mass, and elemental composition summary.
#[derive(Debug, Clone, Default)]
pub struct MolecularPropertiesSummary {
    pub formula_hill: String,
    pub molecular_weight: f64,
    pub exact_mass: f64,
    pub elemental_analysis: String,
    pub heavy_atoms: usize,
    pub rings: usize,
    pub h_bond_donors: usize,
    pub h_bond_acceptors: usize,
}

/// Row in the live Reaction Stoichiometry Drawer.
#[derive(Debug, Clone)]
pub struct StoichiometryRow {
    pub role: String,
    pub formula: String,
    pub mw: f64,
    pub equiv: f64,
    pub mmol: f64,
    pub mass_mg: f64,
}

/// Optional adapter trait allowing `integrator` in `crates/nmr-gui/src/main.rs` to delegate
/// SMILES/InChI/CDX/CDXML, Clean 2D, IUPAC naming, exact properties, and HOSE NMR prediction
/// directly to `nmr_core::octet_mol` and `nmr_core::octet_analyt`!
///
/// Every method defaults to `None` so `OctetDrawPanel` falls back seamlessly to its built-in
/// self-contained engines whenever a hook is not overridden.
pub trait OctetBackendAdapter: Send {
    fn clean_2d(&self, _scene: &RenderScene) -> Option<RenderScene> {
        None
    }
    fn name_or_smiles_to_scene(&self, _query: &str) -> Option<Result<RenderScene, String>> {
        None
    }
    fn scene_to_name_and_smiles(&self, _scene: &RenderScene) -> Option<(String, String)> {
        None
    }
    fn compute_properties(&self, _scene: &RenderScene) -> Option<MolecularPropertiesSummary> {
        None
    }
    fn predict_nmr_1h_13c(
        &self,
        _scene: &RenderScene,
        _freq_mhz: f64,
    ) -> Option<(Vec<SpectrumSignal1D>, Vec<SpectrumSignal1D>)> {
        None
    }
    fn export_custom(
        &self,
        _scene: &RenderScene,
        _style: &DocumentStyle,
        _format: IoFormat,
    ) -> Option<Result<String, String>> {
        None
    }
    fn import_custom(
        &self,
        _format: IoFormat,
        _input: &str,
    ) -> Option<Result<RenderScene, String>> {
        None
    }
}

// ============================================================================
// 2. Editor Tools, Hit Targets, and Drag Interactions
// ============================================================================

/// Active interactive drawing tool on the Octet canvas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawTool {
    /// Box / click selection & move (`V`).
    SelectBox,
    /// Freehand polygon lasso selection.
    Lasso,
    /// Camera pan tool (also accessible anytime via Space+Drag or Middle/Right-Drag).
    Pan,
    /// Bond sprouting & dragging tool (`1`, `2`, `3`, `W`, `H`).
    Bond {
        order: BondOrder,
        stereo: BondStereo,
    },
    /// Multi-carbon 120° zig-zag alkane chain tool.
    Chain,
    /// Quick ring stamp / bond-fusion tool (`3..=8`, benzene, chair flips).
    RingStamp { template_id: String },
    /// Straight reaction arrow (`Solid`, `Equilibrium`, `Retrosynthetic`, `Resonance`, `NoGo`).
    ReactionArrow { kind: ArrowKind },
    /// Curved Bezier electron-pushing arrow (`2e⁻` full or `1e⁻` fishhook).
    CurvedArrow { kind: PushingArrowKind },
    /// Polymer / transition-state enclosure bracket (`[ ]_n`, `( )_n`, `{ }_n`).
    Bracket { style: BracketStyle },
    /// Atomic or hybrid orbital stamp (`s`, `p`, `d`, `dz2`, `sp3`).
    Orbital { orbital_type: OrbitalType },
    /// Thin-layer chromatography plate stamp.
    TlcPlate,
    /// Text caption or condensed functional-group label tool (`T`).
    TextLabel,
    /// Atom/bond/object eraser (`Delete`/`Backspace`).
    Eraser,
}

/// Canvas object under the mouse cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoverTarget {
    Atom(usize),
    Bond(usize),
    ReactionArrow(usize),
    CurvedArrow(usize),
    Bracket(usize),
    Orbital(usize),
    TlcPlate(usize),
    Spectrum1D(usize),
    Text(usize),
}

/// Active mouse drag gesture on the molecular canvas.
#[derive(Debug, Clone)]
enum ActiveDrag {
    BoxSelect {
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
    },
    LassoSelect {
        points_mol: Vec<[f64; 2]>,
    },
    MoveSelection {
        last_mol: [f64; 2],
        pushed_undo: bool,
    },
    DrawBond {
        from_atom: Option<usize>,
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
        order: BondOrder,
        stereo: BondStereo,
    },
    DrawChain {
        from_atom: Option<usize>,
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
    },
    DrawReactionArrow {
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
        kind: ArrowKind,
    },
    DrawCurvedArrow {
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
        kind: PushingArrowKind,
        bow_sgn: f64,
    },
    DrawBracket {
        start_mol: [f64; 2],
        cur_mol: [f64; 2],
        style: BracketStyle,
    },
}

/// Which nucleus tab is active in the live NMR spectrum inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmrViewTab {
    Proton1H,
    Carbon13C,
}

// ============================================================================
// 3. Main `OctetDrawPanel` Studio State
// ============================================================================

pub struct OctetDrawPanel {
    /// Current 2D chemical structure, reaction, clipware & embedded spectrum scene.
    pub scene: RenderScene,
    /// Active publication document style (`ACS Document 1996` by default).
    pub style: DocumentStyle,
    /// Built-in gallery of 50+ ready-to-stamp templates.
    pub library: TemplateLibrary,
    /// Active interactive drawing tool.
    pub active_tool: DrawTool,
    /// Snap dragged bonds/arrows to 30° / 15° increments.
    pub snap_angles: bool,
    /// Show subtle molecular coordinate grid on canvas.
    pub show_grid: bool,

    // --- Camera (pan in molecular bond-lengths + zoom in screen pixels per bond-length) ---
    pub cam_center_mol: [f64; 2],
    pub pixels_per_bond: f32,

    // --- Selection, Hover & Undo/Redo History ---
    pub selected_atoms: HashSet<usize>,
    pub hover_target: Option<HoverTarget>,
    /// Atom indices highlighted by hovering a 1D NMR multiplet peak in the inspector!
    pub highlighted_nmr_atoms: HashSet<usize>,
    undo_stack: Vec<RenderScene>,
    redo_stack: Vec<RenderScene>,
    active_drag: Option<ActiveDrag>,

    // --- Workflow Input Bars ---
    pub peptide_seq_input: String,
    /// Residue codes (`["MET", "GLN", ...]`) synchronized from `NmrWorkbenchApp`'s loaded protein!
    pub loaded_protein_sequence: Vec<String>,
    pub protein_import_max_res: usize,
    pub name_or_smiles_input: String,
    pub active_text_caption: String,
    pub arrow_above_input: String,
    pub arrow_below_input: String,
    pub bracket_sub_input: String,

    // --- Template Gallery & Import/Export Modals ---
    pub show_template_gallery: bool,
    pub template_category_filter: TemplateCategory,
    pub template_search_query: String,
    pub show_io_modal: bool,
    pub selected_io_format: IoFormat,
    pub io_text_buffer: String,
    pub last_png_bytes_len: usize,
    pub last_pdf_bytes_len: usize,

    // --- Live Chemistry Properties, Stoichiometry & Predicted 1D NMR ---
    pub show_right_drawer: bool,
    pub show_bottom_nmr_panel: bool,
    pub nmr_tab: NmrViewTab,
    pub spectrometer_mhz: f64,
    pub ref_scale_mmol: f64,
    cached_props: MolecularPropertiesSummary,
    cached_stoich: Vec<StoichiometryRow>,
    cached_1h: Vec<SpectrumSignal1D>,
    cached_13c: Vec<SpectrumSignal1D>,
    scene_dirty: bool,

    /// Status message displayed in the studio footer/header.
    pub status_msg: String,
    /// Optional pluggable hook to `octet_mol` + `octet_analyt`.
    backend: Option<Box<dyn OctetBackendAdapter>>,
}

impl Default for OctetDrawPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl OctetDrawPanel {
    /// Create a new interactive Octet Chemical Draw & 1D NMR Studio initialized with an
    /// aspirin -> salicylic acid demo scheme so the user sees live stoichiometry & 1D NMR immediately.
    pub fn new() -> Self {
        let library = TemplateLibrary::builtin();
        let style = JournalStyle::Acs1996.to_style();
        let demo_scene = build_default_showcase_scene(&library);

        let mut panel = Self {
            scene: demo_scene,
            style,
            library,
            active_tool: DrawTool::Bond {
                order: BondOrder::Single,
                stereo: BondStereo::None,
            },
            snap_angles: true,
            show_grid: true,
            cam_center_mol: [0.0, 0.0],
            pixels_per_bond: 38.0,
            selected_atoms: HashSet::new(),
            hover_target: None,
            highlighted_nmr_atoms: HashSet::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            active_drag: None,
            peptide_seq_input: "Ala-Pro-Tyr-Lys".to_string(),
            loaded_protein_sequence: Vec::new(),
            protein_import_max_res: 8,
            name_or_smiles_input: "aspirin".to_string(),
            active_text_caption: "Cat. Pd(PPh3)4".to_string(),
            arrow_above_input: "NaOH, H2O".to_string(),
            arrow_below_input: "25 C, 95%".to_string(),
            bracket_sub_input: "n".to_string(),
            show_template_gallery: false,
            template_category_filter: TemplateCategory::Rings,
            template_search_query: String::new(),
            show_io_modal: false,
            selected_io_format: IoFormat::Svg,
            io_text_buffer: String::new(),
            last_png_bytes_len: 0,
            last_pdf_bytes_len: 0,
            show_right_drawer: true,
            show_bottom_nmr_panel: true,
            nmr_tab: NmrViewTab::Proton1H,
            spectrometer_mhz: 500.0,
            ref_scale_mmol: 1.0,
            cached_props: MolecularPropertiesSummary::default(),
            cached_stoich: Vec::new(),
            cached_1h: Vec::new(),
            cached_13c: Vec::new(),
            scene_dirty: true,
            status_msg: "Octet Chemical Studio ready — click/drag to draw bonds, press C/N/O/S/P/F on atoms, V to select, 1/2/3/W/H for bond styles.".to_string(),
            backend: None,
        };
        panel.recompute_live_analytics();
        panel
    }

    /// Connect an [`OctetBackendAdapter`] bridging `octet_mol` and `octet_analyt`.
    pub fn set_backend_adapter(&mut self, adapter: Box<dyn OctetBackendAdapter>) {
        self.backend = Some(adapter);
        self.scene_dirty = true;
        self.recompute_live_analytics();
    }

    /// Update the protein residue sequence loaded in the main NMR assignment workbench
    /// (`e.g. ["MET", "GLN", "ILE", "PHE", "VAL", "LYS", ...]`) so 1-click import works!
    pub fn set_loaded_protein_sequence(&mut self, residues: Vec<String>) {
        self.loaded_protein_sequence = residues;
    }

    /// Borrow the current [`RenderScene`].
    pub fn scene(&self) -> &RenderScene {
        &self.scene
    }

    /// Replace the active [`RenderScene`] (recording an undo snapshot).
    pub fn set_scene(&mut self, new_scene: RenderScene) {
        self.push_undo();
        self.scene = new_scene;
        self.selected_atoms.clear();
        self.scene_dirty = true;
        self.fit_camera_to_scene();
        self.recompute_live_analytics();
    }

    /// Push current scene onto the undo stack before mutating `self.scene`.
    pub fn push_undo(&mut self) {
        if self.undo_stack.len() >= 128 {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.scene.clone());
        self.redo_stack.clear();
        self.scene_dirty = true;
    }

    pub fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(self.scene.clone());
            self.scene = prev;
            self.selected_atoms.clear();
            self.scene_dirty = true;
            self.recompute_live_analytics();
            self.status_msg = format!("Undo ({} steps remaining)", self.undo_stack.len());
        }
    }

    pub fn redo(&mut self) {
        if let Some(nxt) = self.redo_stack.pop() {
            self.undo_stack.push(self.scene.clone());
            self.scene = nxt;
            self.selected_atoms.clear();
            self.scene_dirty = true;
            self.recompute_live_analytics();
            self.status_msg = format!("Redo ({} steps remaining)", self.redo_stack.len());
        }
    }

    pub fn fit_camera_to_scene(&mut self) {
        if self.scene.atoms.is_empty() {
            self.cam_center_mol = [0.0, 0.0];
            self.pixels_per_bond = 38.0;
            return;
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for a in &self.scene.atoms {
            min_x = min_x.min(a.x);
            max_x = max_x.max(a.x);
            min_y = min_y.min(a.y);
            max_y = max_y.max(a.y);
        }
        for sp in &self.scene.spectra_1d {
            min_x = min_x.min(sp.origin[0]);
            max_x = max_x.max(sp.origin[0] + sp.width_mol);
            min_y = min_y.min(sp.origin[1]);
            max_y = max_y.max(sp.origin[1] + sp.height_mol);
        }
        self.cam_center_mol = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)];
        let span = (max_x - min_x).max(max_y - min_y).max(4.0);
        self.pixels_per_bond = (420.0 / span as f32).clamp(18.0, 64.0);
    }

    // ========================================================================
    // Geometry, Snapping, Sprouting, Clean 2D & Peptide/Name Actions
    // ========================================================================

    /// Compute optimal ChemDraw / Octet-Draw next bond direction `(dx, dy)` in molecular y-up space
    /// from `atom_idx` (including automatic chair axial -> equatorial sprouting!).
    pub fn next_sprout_direction(&self, atom_idx: usize) -> [f64; 2] {
        if let Some(chair_sv) = chair_sprout_vectors(&self.scene, atom_idx) {
            return chair_sv.next_recommended;
        }
        let here = [self.scene.atoms[atom_idx].x, self.scene.atoms[atom_idx].y];
        let mut nbrs: Vec<(usize, BondOrder)> = Vec::new();
        for b in &self.scene.bonds {
            if b.order == BondOrder::HydrogenBond {
                continue;
            }
            if b.a == atom_idx {
                nbrs.push((b.b, b.order));
            } else if b.b == atom_idx {
                nbrs.push((b.a, b.order));
            }
        }
        if nbrs.is_empty() {
            let a = 30.0_f64.to_radians();
            return [a.cos(), a.sin()];
        }
        let unit_to = |j: usize| -> [f64; 2] {
            let dx = self.scene.atoms[j].x - here[0];
            let dy = self.scene.atoms[j].y - here[1];
            let l = dx.hypot(dy).max(1e-6);
            [dx / l, dy / l]
        };
        if nbrs.len() == 1 {
            let (j, ord) = nbrs[0];
            let d0 = unit_to(j);
            // sp triple bond continues straight at 180°!
            if ord == BondOrder::Triple {
                return [-d0[0], -d0[1]];
            }
            let rot = |v: [f64; 2], deg: f64| -> [f64; 2] {
                let r = deg.to_radians();
                [
                    v[0] * r.cos() - v[1] * r.sin(),
                    v[0] * r.sin() + v[1] * r.cos(),
                ]
            };
            let cand_a = rot(d0, 120.0);
            let cand_b = rot(d0, -120.0);
            // Check neighbour's own neighbour for trans zig-zag continuation
            let prev: Vec<usize> = self
                .scene
                .bonds
                .iter()
                .filter_map(|b| {
                    if b.a == j && b.b != atom_idx {
                        Some(b.b)
                    } else if b.b == j && b.a != atom_idx {
                        Some(b.a)
                    } else {
                        None
                    }
                })
                .collect();
            if prev.len() == 1 {
                let axis = [
                    here[0] - self.scene.atoms[j].x,
                    here[1] - self.scene.atoms[j].y,
                ];
                let pv = [
                    self.scene.atoms[prev[0]].x - self.scene.atoms[j].x,
                    self.scene.atoms[prev[0]].y - self.scene.atoms[j].y,
                ];
                let cross_prev = axis[0] * pv[1] - axis[1] * pv[0];
                let cross_a = axis[0] * cand_a[1] - axis[1] * cand_a[0];
                return if cross_prev * cross_a > 0.0 {
                    cand_b
                } else {
                    cand_a
                };
            }
            return if cand_a[1] >= cand_b[1] {
                cand_a
            } else {
                cand_b
            };
        }
        if nbrs.len() == 2 {
            let d0 = unit_to(nbrs[0].0);
            let d1 = unit_to(nbrs[1].0);
            let sx = d0[0] + d1[0];
            let sy = d0[1] + d1[1];
            let sl = sx.hypot(sy);
            return if sl < 1e-4 {
                [-d0[1], d0[0]]
            } else {
                [-sx / sl, -sy / sl]
            };
        }
        // Widest angular sector gap
        let mut angs: Vec<f64> = nbrs
            .iter()
            .map(|&(j, _)| {
                let d = unit_to(j);
                d[1].atan2(d[0])
            })
            .collect();
        angs.sort_by(|a, b| a.total_cmp(b));
        let mut best_gap = -1.0;
        let mut best_mid = 0.0;
        for k in 0..angs.len() {
            let a = angs[k];
            let b = if k + 1 < angs.len() {
                angs[k + 1]
            } else {
                angs[0] + 2.0 * PI
            };
            if b - a > best_gap {
                best_gap = b - a;
                best_mid = a + 0.5 * (b - a);
            }
        }
        [best_mid.cos(), best_mid.sin()]
    }

    /// Snap a dragged bond endpoint `raw_mol` from `start_mol` to nearby atoms or 30° fixed angles.
    fn snap_bond_endpoint(
        &self,
        start_mol: [f64; 2],
        raw_mol: [f64; 2],
        ignore_atom: Option<usize>,
    ) -> ([f64; 2], Option<usize>) {
        // 1. Magnetic snap onto existing atom within 0.32 bond lengths
        for (i, a) in self.scene.atoms.iter().enumerate() {
            if Some(i) == ignore_atom || a.is_multi_center() {
                continue;
            }
            if (a.x - raw_mol[0]).hypot(a.y - raw_mol[1]) < 0.32 {
                return ([a.x, a.y], Some(i));
            }
        }
        let dx = raw_mol[0] - start_mol[0];
        let dy = raw_mol[1] - start_mol[1];
        let dist = dx.hypot(dy);
        if !self.snap_angles || dist < 0.12 {
            return (raw_mol, None);
        }
        let step_rad = 30.0_f64.to_radians();
        let snapped_ang = (dy.atan2(dx) / step_rad).round() * step_rad;
        let snapped_len = if (dist - 1.0).abs() < 0.45 {
            1.0
        } else {
            (dist * 2.0).round() * 0.5
        }
        .max(1.0);
        let q = [
            start_mol[0] + snapped_len * snapped_ang.cos(),
            start_mol[1] + snapped_len * snapped_ang.sin(),
        ];
        // Re-check atom coincidence at snapped position
        for (i, a) in self.scene.atoms.iter().enumerate() {
            if Some(i) != ignore_atom
                && !a.is_multi_center()
                && (a.x - q[0]).hypot(a.y - q[1]) < 0.22
            {
                return ([a.x, a.y], Some(i));
            }
        }
        (q, None)
    }

    /// Compute 120° zig-zag alkane chain points from `start_mol` toward `cur_mol`.
    fn compute_chain_points(&self, start_mol: [f64; 2], cur_mol: [f64; 2]) -> Vec<[f64; 2]> {
        let dx = cur_mol[0] - start_mol[0];
        let dy = cur_mol[1] - start_mol[1];
        let along = dx.hypot(dy);
        let tilt = 30.0_f64.to_radians();
        let step = tilt.cos(); // 0.8660254
        let rise = tilt.sin(); // 0.5
        let count = ((along / step).round() as usize).clamp(1, 32);
        let (ux, uy) = if along > 1e-5 {
            (dx / along, dy / along)
        } else {
            (1.0, 0.0)
        };
        let nx = -uy;
        let ny = ux;
        (0..count)
            .map(|k| {
                let s = (k + 1) as f64 * step;
                let h = if k % 2 == 0 { rise } else { 0.0 };
                [
                    start_mol[0] + s * ux + h * nx,
                    start_mol[1] + s * uy + h * ny,
                ]
            })
            .collect()
    }

    /// Clean 2D coordinates: regularize bond lengths toward `1.0` and relax acyclic angles toward 120°.
    pub fn clean_2d_structure(&mut self) {
        if self.scene.atoms.is_empty() {
            return;
        }
        self.push_undo();
        if let Some(ad) = &self.backend {
            if let Some(cleaned) = ad.clean_2d(&self.scene) {
                self.scene = cleaned;
                self.recompute_live_analytics();
                self.status_msg = "Cleaned 2D coordinates (octet_mol)".to_string();
                return;
            }
        }

        // Built-in ring-preserving Clean 2D relaxation
        let rings = perceive_small_rings(self.scene.atoms.len(), &self.scene.bonds);
        let mut ring_atoms = HashSet::new();
        for r in &rings {
            for &i in r {
                ring_atoms.insert(i);
            }
        }

        for _iter in 0..48 {
            let n = self.scene.atoms.len();
            let mut forces = vec![[0.0_f64; 2]; n];

            // 1. Bond length spring toward 1.0
            for b in &self.scene.bonds {
                if b.a >= n || b.b >= n || b.a == b.b || b.order == BondOrder::HydrogenBond {
                    continue;
                }
                let dx = self.scene.atoms[b.b].x - self.scene.atoms[b.a].x;
                let dy = self.scene.atoms[b.b].y - self.scene.atoms[b.a].y;
                let d = dx.hypot(dy).max(1e-4);
                let err = d - 1.0;
                let k = if ring_atoms.contains(&b.a) && ring_atoms.contains(&b.b) {
                    0.35
                } else {
                    0.45
                };
                let fx = k * err * (dx / d);
                let fy = k * err * (dy / d);
                forces[b.a][0] += fx;
                forces[b.a][1] += fy;
                forces[b.b][0] -= fx;
                forces[b.b][1] -= fy;
            }

            // 2. Geminal 1,3-distance spring toward `sqrt(3) = 1.732` (120° angle!) for acyclic centers
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for b in &self.scene.bonds {
                if b.order != BondOrder::HydrogenBond && b.a < n && b.b < n && b.a != b.b {
                    adj[b.a].push(b.b);
                    adj[b.b].push(b.a);
                }
            }
            for (center, center_adj) in adj.iter().enumerate().take(n) {
                if ring_atoms.contains(&center) || center_adj.len() > 4 {
                    continue;
                }
                let target_13 =
                    if center_adj.len() == 2
                        && self.scene.bonds.iter().any(|b| {
                            b.order == BondOrder::Triple && (b.a == center || b.b == center)
                        })
                    {
                        2.0 // sp linear 180°
                    } else {
                        3.0_f64.sqrt() // 120°
                    };
                for p in 0..center_adj.len() {
                    for q in (p + 1)..center_adj.len() {
                        let u = center_adj[p];
                        let v = center_adj[q];
                        let dx = self.scene.atoms[v].x - self.scene.atoms[u].x;
                        let dy = self.scene.atoms[v].y - self.scene.atoms[u].y;
                        let d = dx.hypot(dy).max(1e-4);
                        let err = d - target_13;
                        let fx = 0.14 * err * (dx / d);
                        let fy = 0.14 * err * (dy / d);
                        forces[u][0] += fx;
                        forces[u][1] += fy;
                        forces[v][0] -= fx;
                        forces[v][1] -= fy;
                    }
                }
            }

            // 3. Soft non-bonded steric repulsion for overlapping atoms
            for i in 0..n {
                for j in (i + 1)..n {
                    if adj[i].contains(&j) {
                        continue;
                    }
                    let dx = self.scene.atoms[j].x - self.scene.atoms[i].x;
                    let dy = self.scene.atoms[j].y - self.scene.atoms[i].y;
                    let d = dx.hypot(dy).max(0.08);
                    if d < 0.82 {
                        let push = 0.08 * (0.82 - d) / d;
                        forces[i][0] -= push * dx;
                        forces[i][1] -= push * dy;
                        forces[j][0] += push * dx;
                        forces[j][1] += push * dy;
                    }
                }
            }

            for (atom, force) in self.scene.atoms.iter_mut().zip(forces.iter()).take(n) {
                if !atom.is_multi_center() {
                    atom.x += force[0].clamp(-0.18, 0.18);
                    atom.y += force[1].clamp(-0.18, 0.18);
                }
            }
        }

        self.recompute_live_analytics();
        self.status_msg = "Cleaned 2D bond lengths and angles".to_string();
    }

    /// Build a 2D L-polypeptide from `self.peptide_seq_input`.
    pub fn build_peptide_from_input(&mut self) {
        let codes = parse_peptide_sequence(&self.peptide_seq_input);
        if codes.is_empty() {
            self.status_msg =
                "Enter 1-letter (e.g. MQIFVK) or 3-letter (e.g. Ala-Pro-Tyr) peptide sequence"
                    .to_string();
            return;
        }
        let pep = build_peptide_scene(&codes);
        let n_res = codes.len();
        self.set_scene(pep);
        self.status_msg = format!("Built all-trans L-peptide ({n_res} residues)");
    }

    /// 1-click import of residues directly from the loaded protein in the NMR assignment workbench!
    pub fn import_from_loaded_protein(&mut self) {
        if self.loaded_protein_sequence.is_empty() {
            // Fallback to N-terminal Ubiquitin β-hairpin (`MQIFVKTL`) if no protein loaded yet
            let default_ubq = ["MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU"];
            let pep = build_peptide_scene(&default_ubq);
            self.set_scene(pep);
            self.status_msg =
                "Imported Ubiquitin N-terminal 1–8 (MQIFVKTL) L-polypeptide structure".to_string();
            return;
        }
        let take_n = self
            .protein_import_max_res
            .clamp(1, self.loaded_protein_sequence.len());
        let refs: Vec<&str> = self.loaded_protein_sequence[..take_n]
            .iter()
            .map(String::as_str)
            .collect();
        let joined = refs.join("-");
        let pep = build_peptide_scene(&refs);
        self.set_scene(pep);
        self.status_msg = format!(
            "Imported first {} residues ({}) from loaded protein sequence",
            take_n, joined
        );
    }

    /// Resolve a chemical name, template name, or SMILES string into a 2D structure!
    pub fn convert_name_or_smiles_to_structure(&mut self) {
        let q = self.name_or_smiles_input.trim().to_string();
        if q.is_empty() {
            return;
        }
        if let Some(ad) = &self.backend {
            if let Some(res) = ad.name_or_smiles_to_scene(&q) {
                match res {
                    Ok(sc) => {
                        self.set_scene(sc);
                        self.status_msg = format!("Built structure from `{q}`");
                    }
                    Err(e) => self.status_msg = format!("Name/SMILES error: {e}"),
                }
                return;
            }
        }

        match builtin_name_or_smiles_to_scene(&q, &self.library) {
            Ok(sc) => {
                self.set_scene(sc);
                self.status_msg = format!("Converted `{q}` → 2D chemical structure");
            }
            Err(e) => {
                self.status_msg = e;
            }
        }
    }

    /// Generate SMILES / systematic identifier from the current canvas structure into the bar!
    pub fn convert_structure_to_name_or_smiles(&mut self) {
        if let Some(ad) = &self.backend {
            if let Some((name, smi)) = ad.scene_to_name_and_smiles(&self.scene) {
                self.name_or_smiles_input = if name.is_empty() {
                    smi.clone()
                } else {
                    format!("{name} ({smi})")
                };
                self.status_msg = format!("Generated identifier: {}", self.name_or_smiles_input);
                return;
            }
        }
        let smi = builtin_scene_to_smiles(&self.scene);
        self.name_or_smiles_input = smi.clone();
        self.status_msg = format!("Generated SMILES: {smi}");
    }

    /// Embed the live predicted 1D NMR spectrum (`¹H` or `¹³C`) directly onto the drawing canvas
    /// so SVG, PDF 1.4, and PNG exports include the publication 1D NMR plot underneath the structure!
    pub fn embed_1d_spectrum_on_canvas(&mut self) {
        self.recompute_live_analytics();
        self.push_undo();
        let min_y = self.scene.atoms.iter().map(|a| a.y).fold(0.0_f64, f64::min);
        let cx = self.scene.atom_centroid()[0];

        let spec_plot = match self.nmr_tab {
            NmrViewTab::Proton1H => SceneSpectrum1D {
                origin: [cx - 5.0, min_y - 4.8],
                width_mol: 10.0,
                height_mol: 3.4,
                nucleus: "1H".to_string(),
                spectrometer_mhz: self.spectrometer_mhz,
                ppm_min: -0.5,
                ppm_max: 11.5,
                linewidth_hz: 1.4,
                show_integrals: true,
                show_peak_labels: true,
                title: Some(format!(
                    "Predicted 1H NMR ({:.0} MHz)",
                    self.spectrometer_mhz
                )),
                signals: self.cached_1h.clone(),
                custom_trace: Vec::new(),
            },
            NmrViewTab::Carbon13C => SceneSpectrum1D {
                origin: [cx - 5.0, min_y - 4.8],
                width_mol: 10.0,
                height_mol: 3.4,
                nucleus: "13C".to_string(),
                spectrometer_mhz: self.spectrometer_mhz * 0.251_45,
                ppm_min: 0.0,
                ppm_max: 220.0,
                linewidth_hz: 2.0,
                show_integrals: false,
                show_peak_labels: true,
                title: Some(format!(
                    "Predicted 13C{{1H}} NMR ({:.0} MHz)",
                    self.spectrometer_mhz * 0.251_45
                )),
                signals: self.cached_13c.clone(),
                custom_trace: Vec::new(),
            },
        };
        self.scene
            .spectra_1d
            .retain(|s| s.nucleus != spec_plot.nucleus);
        self.scene.spectra_1d.push(spec_plot);
        self.fit_camera_to_scene();
        self.status_msg = "Embedded 1D NMR spectrum plot onto publication canvas".to_string();
    }

    // ========================================================================
    // Live Properties, Stoichiometry & ¹H / ¹³C NMR Predictor
    // ========================================================================

    pub fn recompute_live_analytics(&mut self) {
        if let Some(ad) = &self.backend {
            if let Some(props) = ad.compute_properties(&self.scene) {
                self.cached_props = props;
            } else {
                self.cached_props = compute_builtin_properties(&self.scene);
            }
            if let Some((h1, c13)) = ad.predict_nmr_1h_13c(&self.scene, self.spectrometer_mhz) {
                self.cached_1h = h1;
                self.cached_13c = c13;
            } else {
                let (h1, c13) = predict_builtin_nmr_1h_13c(&self.scene);
                self.cached_1h = h1;
                self.cached_13c = c13;
            }
        } else {
            self.cached_props = compute_builtin_properties(&self.scene);
            let (h1, c13) = predict_builtin_nmr_1h_13c(&self.scene);
            self.cached_1h = h1;
            self.cached_13c = c13;
        }
        self.cached_stoich = compute_builtin_stoichiometry(&self.scene, self.ref_scale_mmol);
        self.scene_dirty = false;
    }

    /// Generate export payload into `self.io_text_buffer` for the selected [`IoFormat`].
    pub fn refresh_export_buffer(&mut self) {
        if let Some(ad) = &self.backend {
            if let Some(res) = ad.export_custom(&self.scene, &self.style, self.selected_io_format) {
                match res {
                    Ok(txt) => {
                        self.io_text_buffer = txt;
                        return;
                    }
                    Err(e) => {
                        self.status_msg = format!("Export error: {e}");
                        return;
                    }
                }
            }
        }

        match self.selected_io_format {
            IoFormat::OctJson => {
                self.io_text_buffer = serde_json::to_string_pretty(&self.scene)
                    .unwrap_or_else(|e| format!("JSON error: {e}"));
            }
            IoFormat::Svg => {
                self.io_text_buffer = render_scene_svg(&self.scene, &self.style);
            }
            IoFormat::Pdf => {
                let pdf = render_scene_pdf(&self.scene, &self.style);
                self.last_pdf_bytes_len = pdf.len();
                self.io_text_buffer = String::from_utf8_lossy(&pdf).into_owned();
            }
            IoFormat::Png => {
                let png = render_scene_png(&self.scene, &self.style, 144.0);
                self.last_png_bytes_len = png.len();
                let hex_preview: String = png
                    .iter()
                    .take(256)
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.io_text_buffer = format!(
                    "# Lossless PNG generated ({} bytes @ 144 DPI)\n# Header + first 256 bytes (hex):\n{}",
                    png.len(),
                    hex_preview
                );
            }
            IoFormat::MolV2000 => {
                self.io_text_buffer = export_builtin_mol_v2000(&self.scene, false);
            }
            IoFormat::Sdf => {
                self.io_text_buffer = export_builtin_mol_v2000(&self.scene, true);
            }
            IoFormat::Smiles => {
                self.io_text_buffer = builtin_scene_to_smiles(&self.scene);
            }
            IoFormat::InChI => {
                self.io_text_buffer = format!("InChI=1S/{}/h1H", self.cached_props.formula_hill);
            }
            IoFormat::Cdxml => {
                self.io_text_buffer = export_builtin_cdxml(&self.scene, &self.style);
            }
            IoFormat::CdxHex => {
                self.io_text_buffer = export_builtin_cdx_hex(&self.scene);
            }
            IoFormat::JcampDx => {
                self.io_text_buffer = export_builtin_jcamp_dx(
                    &self.cached_1h,
                    self.spectrometer_mhz,
                    &self.cached_props.formula_hill,
                );
            }
        }
    }

    /// Import `self.io_text_buffer` according to `self.selected_io_format`.
    pub fn import_from_buffer(&mut self) {
        let buf = self.io_text_buffer.trim().to_string();
        if buf.is_empty() {
            self.status_msg = "Paste file contents into the I/O buffer first".to_string();
            return;
        }
        if let Some(ad) = &self.backend {
            if let Some(res) = ad.import_custom(self.selected_io_format, &buf) {
                match res {
                    Ok(sc) => {
                        self.set_scene(sc);
                        self.status_msg = format!("Imported {}", self.selected_io_format.label());
                    }
                    Err(e) => self.status_msg = format!("Import error: {e}"),
                }
                return;
            }
        }

        let imported = match self.selected_io_format {
            IoFormat::OctJson => serde_json::from_str::<RenderScene>(&buf)
                .map_err(|e| format!("OCT JSON parse error: {e}")),
            IoFormat::MolV2000 | IoFormat::Sdf => import_builtin_mol_v2000(&buf),
            IoFormat::Cdxml => import_builtin_cdxml(&buf),
            IoFormat::Smiles | IoFormat::InChI => {
                builtin_name_or_smiles_to_scene(&buf, &self.library)
            }
            IoFormat::JcampDx => import_builtin_jcamp_dx(&buf, &self.scene),
            IoFormat::CdxHex => import_builtin_cdx_hex(&buf),
            _ => Err("Selected format is export-only (SVG/PDF/PNG)".to_string()),
        };

        match imported {
            Ok(sc) => {
                self.set_scene(sc);
                self.status_msg = format!("Imported {}", self.selected_io_format.label());
            }
            Err(e) => {
                self.status_msg = e;
            }
        }
    }
}

// ============================================================================
// 4. Interactive `egui` UI, Toolbars, Keyboard Shortcuts & `Painter` Canvas
// ============================================================================

impl OctetDrawPanel {
    /// Render the complete interactive Octet Chemical Draw & 1D NMR Studio inside `ui`.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        if self.scene_dirty {
            self.recompute_live_analytics();
        }

        let ctx = ui.ctx().clone();
        self.draw_template_gallery_window(&ctx);
        self.draw_io_modal_window(&ctx);

        egui::Panel::top("octet_top_workflow_bar").show(ui, |ui| {
            self.draw_top_workflow_bar(ui);
            ui.add_space(2.0);
            self.draw_quick_group_and_hotkey_bar(ui);
        });

        if self.show_right_drawer {
            egui::Panel::right("octet_right_props_drawer")
                .default_size(315.0)
                .min_size(265.0)
                .show(ui, |ui| {
                    self.draw_properties_and_stoichiometry_drawer(ui);
                });
        }

        if self.show_bottom_nmr_panel {
            egui::Panel::bottom("octet_bottom_nmr_inspector")
                .default_size(195.0)
                .min_size(145.0)
                .show(ui, |ui| {
                    self.draw_bottom_nmr_inspector(ui);
                });
        }

        egui::Panel::left("octet_left_tool_palette")
            .default_size(138.0)
            .min_size(124.0)
            .show(ui, |ui| {
                self.draw_left_tool_palette(ui);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.draw_interactive_canvas(ui);
        });
    }

    fn draw_top_workflow_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!self.undo_stack.is_empty(), egui::Button::new("↩ Undo"))
                .on_hover_text("Undo (Cmd/Ctrl+Z)")
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(!self.redo_stack.is_empty(), egui::Button::new("↪ Redo"))
                .on_hover_text("Redo (Cmd/Ctrl+Shift+Z)")
                .clicked()
            {
                self.redo();
            }
            if ui
                .button("✨ Clean 2D")
                .on_hover_text("Regularize bond lengths & 120° angles")
                .clicked()
            {
                self.clean_2d_structure();
            }
            if ui.button("🗑 Clear").clicked() {
                self.set_scene(RenderScene::new());
                self.status_msg = "Cleared canvas".to_string();
            }

            ui.separator();

            // Journal Style Preset Dropdown (`ACS 1996` default)
            ui.label("Style:");
            let cur_preset = self.style.preset;
            egui::ComboBox::from_id_salt("octet_journal_style_combo")
                .selected_text(cur_preset.name())
                .show_ui(ui, |ui| {
                    for &preset in JournalStyle::ALL {
                        if ui
                            .selectable_label(cur_preset == preset, preset.name())
                            .clicked()
                        {
                            let cpk = self.style.color_by_element;
                            self.style = preset.to_style();
                            self.style.color_by_element = cpk;
                            self.status_msg = format!(
                                "Applied {} style ({:.1} pt bond, {:.2} pt line)",
                                preset.name(),
                                self.style.bond_length_pt,
                                self.style.line_width_pt
                            );
                        }
                    }
                });

            ui.checkbox(&mut self.style.color_by_element, "CPK")
                .on_hover_text("Color heteroatoms by IUPAC CPK palette");
            ui.checkbox(&mut self.snap_angles, "Snap 30°");
            ui.checkbox(&mut self.show_grid, "Grid");

            ui.separator();

            // Name <-> Structure bar
            ui.label("Name/SMILES:");
            let name_resp = ui.add(
                egui::TextEdit::singleline(&mut self.name_or_smiles_input).desired_width(125.0),
            );
            if name_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.convert_name_or_smiles_to_structure();
            }
            if ui
                .small_button("→ Struct")
                .on_hover_text("Name or SMILES to 2D Structure")
                .clicked()
            {
                self.convert_name_or_smiles_to_structure();
            }
            if ui
                .small_button("→ SMILES")
                .on_hover_text("Current Structure to SMILES")
                .clicked()
            {
                self.convert_structure_to_name_or_smiles();
            }

            ui.separator();

            // Peptide-from-Sequence + 1-click Import from Loaded Protein!
            ui.label("Peptide:");
            ui.add(egui::TextEdit::singleline(&mut self.peptide_seq_input).desired_width(98.0));
            if ui.small_button("Build").clicked() {
                self.build_peptide_from_input();
            }
            let prot_label = if self.loaded_protein_sequence.is_empty() {
                "🧬 Import Protein Seq".to_string()
            } else {
                format!(
                    "🧬 Import Protein (1–{})",
                    self.protein_import_max_res
                        .min(self.loaded_protein_sequence.len())
                )
            };
            if ui
                .small_button(prot_label)
                .on_hover_text("1-click import polypeptide from the loaded NMR protein sequence")
                .clicked()
            {
                self.import_from_loaded_protein();
            }

            ui.separator();

            if ui.button("📚 Templates").clicked() {
                self.show_template_gallery = !self.show_template_gallery;
            }
            if ui
                .button("📤 Import / Export")
                .on_hover_text(
                    "CDXML, CDX, OCT, MOL/SDF, SMILES, InChI, JCAMP-DX, SVG, PDF 1.4, PNG",
                )
                .clicked()
            {
                self.refresh_export_buffer();
                self.show_io_modal = !self.show_io_modal;
            }
            ui.checkbox(&mut self.show_right_drawer, "Props");
            ui.checkbox(&mut self.show_bottom_nmr_panel, "1D NMR");
        });
    }

    fn draw_quick_group_and_hotkey_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.label(egui::RichText::new("Atoms & Groups:").small().weak());
            for sym in [
                "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Mg", "Pd", "K", "H",
            ] {
                if ui
                    .small_button(sym)
                    .on_hover_text(format!("Set hovered/selected atom to {sym}"))
                    .clicked()
                {
                    self.apply_atom_element_or_group(sym, None);
                }
            }
            ui.separator();
            for grp in ["Boc", "Fmoc", "Ph", "Ac", "tBu", "COOH", "NO2", "CF3"] {
                if ui
                    .small_button(grp)
                    .on_hover_text(format!(
                        "Apply condensed {grp} group to hovered/selected atom"
                    ))
                    .clicked()
                {
                    let anchor = match grp {
                        "Boc" | "Fmoc" | "Ac" | "COOH" | "Ph" | "tBu" | "CF3" => "C",
                        "NO2" => "N",
                        _ => "C",
                    };
                    self.apply_atom_element_or_group(anchor, Some(grp));
                }
            }
            ui.separator();
            if ui
                .small_button("⊕ +")
                .on_hover_text("Increment formal charge (+)")
                .clicked()
            {
                self.adjust_atom_charge(1);
            }
            if ui
                .small_button("⊖ −")
                .on_hover_text("Decrement formal charge (−)")
                .clicked()
            {
                self.adjust_atom_charge(-1);
            }
            if ui
                .small_button("(R)/(S)")
                .on_hover_text("Toggle CIP (R)/(S) stereocenter badge on hovered/selected atom")
                .clicked()
            {
                self.toggle_cip_badge();
            }
            if ui
                .small_button("Highlight")
                .on_hover_text("Toggle soft ring/atom highlight")
                .clicked()
            {
                self.toggle_selection_highlight();
            }
        });
    }

    fn draw_left_tool_palette(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_salt("octet_left_tools_scroll")
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Select & Edit").small().strong());
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.active_tool == DrawTool::SelectBox, "⬚ Select (V)")
                        .clicked()
                    {
                        self.active_tool = DrawTool::SelectBox;
                    }
                    if ui
                        .selectable_label(self.active_tool == DrawTool::Lasso, "➰ Lasso")
                        .clicked()
                    {
                        self.active_tool = DrawTool::Lasso;
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.active_tool == DrawTool::Pan, "✋ Pan")
                        .on_hover_text("Pan canvas (or hold Space + drag anytime)")
                        .clicked()
                    {
                        self.active_tool = DrawTool::Pan;
                    }
                    if ui
                        .selectable_label(self.active_tool == DrawTool::Eraser, "⌫ Eraser")
                        .on_hover_text("Delete atom, bond, or annotation (Del/Backspace)")
                        .clicked()
                    {
                        self.active_tool = DrawTool::Eraser;
                    }
                });

                ui.separator();
                ui.label(egui::RichText::new("Bonds & Chains").small().strong());
                let bond_tools = [
                    ("─ Solid (1)", BondOrder::Single, BondStereo::None),
                    ("═ Double (2)", BondOrder::Double, BondStereo::None),
                    ("≡ Triple (3)", BondOrder::Triple, BondStereo::None),
                    ("◀ Wedge (W)", BondOrder::Single, BondStereo::UpWedge),
                    ("◁ Hash (H)", BondOrder::Single, BondStereo::DownHash),
                    ("∿ Wavy", BondOrder::Single, BondStereo::Wavy),
                    ("→ Dative", BondOrder::Single, BondStereo::Dative),
                    ("⚏ Aromatic", BondOrder::Aromatic, BondStereo::None),
                ];
                for (lbl, ord, st) in bond_tools {
                    let target = DrawTool::Bond {
                        order: ord,
                        stereo: st,
                    };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }
                if ui
                    .selectable_label(self.active_tool == DrawTool::Chain, "⋀⋁ Chain 120°")
                    .clicked()
                {
                    self.active_tool = DrawTool::Chain;
                }

                ui.separator();
                ui.label(egui::RichText::new("Rings & Chairs").small().strong());
                for (lbl, tid) in [
                    ("⬡ Benzene", "benzene"),
                    ("△ 3-Ring", "cyclopropane"),
                    ("□ 4-Ring", "cyclobutane"),
                    ("⬠ 5-Ring", "cyclopentane"),
                    ("⬡ 6-Ring", "cyclohexane"),
                    ("⯃ 7-Ring", "cycloheptane"),
                    ("⯄ 8-Ring", "cyclooctane"),
                    ("⑂ Chair ⁴C₁", "chair_cyclohexane_1"),
                    ("⑃ Chair ¹C₄", "chair_cyclohexane_2"),
                    ("⛨ Norbornane", "norbornane"),
                    ("❖ Adamantane", "adamantane"),
                    ("⬡⬡ Steroid", "steroid_core"),
                ] {
                    let target = DrawTool::RingStamp {
                        template_id: tid.to_string(),
                    };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }

                ui.separator();
                ui.label(egui::RichText::new("Reaction & Pushing").small().strong());
                for (lbl, kind) in [
                    ("⟶ Rxn Solid", ArrowKind::Solid),
                    ("⇌ Equilibrium", ArrowKind::Equilibrium),
                    ("⟹ Retrosynth", ArrowKind::Retrosynthetic),
                    ("⟷ Resonance", ArrowKind::Resonance),
                    ("─✕→ No-Go", ArrowKind::NoGo),
                ] {
                    let target = DrawTool::ReactionArrow { kind };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }
                for (lbl, kind) in [
                    ("↷ Curved 2e⁻", PushingArrowKind::FullTwoElectron),
                    ("⇀ Fishhook 1e⁻", PushingArrowKind::FishhookOneElectron),
                ] {
                    let target = DrawTool::CurvedArrow { kind };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }

                ui.separator();
                ui.label(egui::RichText::new("Brackets, MO & TLC").small().strong());
                for (lbl, bstyle) in [
                    ("[ ]n Polymer", BracketStyle::Square),
                    ("( )n Paren", BracketStyle::Parenthesis),
                    ("{ } Brace", BracketStyle::CurlyBrace),
                ] {
                    let target = DrawTool::Bracket { style: bstyle };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }
                for (lbl, otype) in [
                    ("◯ s Orbital", OrbitalType::S),
                    ("∞ p Orbital", OrbitalType::P),
                    ("✥ d Orbital", OrbitalType::DClover),
                    ("⏣ dz² Orbital", OrbitalType::Dz2),
                ] {
                    let target = DrawTool::Orbital {
                        orbital_type: otype,
                    };
                    if ui
                        .selectable_label(self.active_tool == target, lbl)
                        .clicked()
                    {
                        self.active_tool = target;
                    }
                }
                if ui
                    .selectable_label(self.active_tool == DrawTool::TlcPlate, "▤ TLC Plate")
                    .clicked()
                {
                    self.active_tool = DrawTool::TlcPlate;
                }
                if ui
                    .selectable_label(self.active_tool == DrawTool::TextLabel, "𝐀 Text / Label")
                    .clicked()
                {
                    self.active_tool = DrawTool::TextLabel;
                }
            });
    }

    // ========================================================================
    // Atom / Bond Mutation Helpers & ChemDraw Keyboard Shortcuts
    // ========================================================================

    fn target_atom_indices(&self) -> Vec<usize> {
        if !self.selected_atoms.is_empty() {
            let mut v: Vec<usize> = self
                .selected_atoms
                .iter()
                .copied()
                .filter(|&i| i < self.scene.atoms.len())
                .collect();
            v.sort_unstable();
            return v;
        }
        if let Some(HoverTarget::Atom(i)) = self.hover_target {
            if i < self.scene.atoms.len() {
                return vec![i];
            }
        }
        Vec::new()
    }

    fn apply_atom_element_or_group(&mut self, element: &str, group_text: Option<&str>) {
        let targets = self.target_atom_indices();
        if targets.is_empty() {
            self.status_msg = format!(
                "Hover or select an atom first to label as {}",
                group_text.unwrap_or(element)
            );
            return;
        }
        self.push_undo();
        for i in targets {
            self.scene.atoms[i].label = element.to_string();
            self.scene.atoms[i].text = group_text.map(str::to_string);
            self.scene.atoms[i].show_carbon = element == "C" && group_text.is_none();
        }
        self.recompute_live_analytics();
    }

    fn adjust_atom_charge(&mut self, delta: i8) {
        let targets = self.target_atom_indices();
        if targets.is_empty() {
            return;
        }
        self.push_undo();
        for i in targets {
            self.scene.atoms[i].charge = (self.scene.atoms[i].charge + delta).clamp(-4, 4);
        }
        self.recompute_live_analytics();
    }

    fn toggle_cip_badge(&mut self) {
        let targets = self.target_atom_indices();
        if targets.is_empty() {
            return;
        }
        self.push_undo();
        for i in targets {
            self.scene.atoms[i].cip_label = match self.scene.atoms[i].cip_label.as_deref() {
                None => Some("(R)".to_string()),
                Some("(R)") => Some("(S)".to_string()),
                _ => None,
            };
        }
        self.recompute_live_analytics();
    }

    fn toggle_selection_highlight(&mut self) {
        let targets = self.target_atom_indices();
        if targets.is_empty() {
            return;
        }
        self.push_undo();
        let hl_col = HIGHLIGHT_PALETTE[5].1; // Soft Cyan
        for i in targets {
            self.scene.atoms[i].highlight = if self.scene.atoms[i].highlight.is_some() {
                None
            } else {
                Some(hl_col)
            };
        }
    }

    fn delete_hovered_or_selected(&mut self) {
        if !self.selected_atoms.is_empty() {
            let gone = self.selected_atoms.clone();
            self.push_undo();
            self.remove_atoms_set(&gone);
            self.selected_atoms.clear();
            self.recompute_live_analytics();
            return;
        }
        let Some(ht) = self.hover_target else {
            return;
        };
        self.push_undo();
        match ht {
            HoverTarget::Atom(i) => {
                let mut gone = HashSet::new();
                gone.insert(i);
                self.remove_atoms_set(&gone);
            }
            HoverTarget::Bond(bi) => {
                if bi < self.scene.bonds.len() {
                    // ChemDraw rule: Delete on multiple bond lowers order first, single bond deletes!
                    match self.scene.bonds[bi].order {
                        BondOrder::Triple => self.scene.bonds[bi].order = BondOrder::Double,
                        BondOrder::Double | BondOrder::Aromatic => {
                            self.scene.bonds[bi].order = BondOrder::Single;
                        }
                        _ => {
                            self.scene.bonds.remove(bi);
                        }
                    }
                }
            }
            HoverTarget::ReactionArrow(i) if i < self.scene.reaction_arrows.len() => {
                self.scene.reaction_arrows.remove(i);
            }
            HoverTarget::CurvedArrow(i) if i < self.scene.curved_arrows.len() => {
                self.scene.curved_arrows.remove(i);
            }
            HoverTarget::Bracket(i) if i < self.scene.brackets.len() => {
                self.scene.brackets.remove(i);
            }
            HoverTarget::Orbital(i) if i < self.scene.orbitals.len() => {
                self.scene.orbitals.remove(i);
            }
            HoverTarget::TlcPlate(i) if i < self.scene.tlc_plates.len() => {
                self.scene.tlc_plates.remove(i);
            }
            HoverTarget::Spectrum1D(i) if i < self.scene.spectra_1d.len() => {
                self.scene.spectra_1d.remove(i);
            }
            HoverTarget::Text(i) if i < self.scene.texts.len() => {
                self.scene.texts.remove(i);
            }
            _ => {}
        }
        self.hover_target = None;
        self.recompute_live_analytics();
    }

    fn remove_atoms_set(&mut self, gone: &HashSet<usize>) {
        let mut remap = Vec::with_capacity(self.scene.atoms.len());
        let mut new_atoms = Vec::with_capacity(self.scene.atoms.len());
        for (i, a) in self.scene.atoms.iter().enumerate() {
            if gone.contains(&i) {
                remap.push(None);
            } else {
                remap.push(Some(new_atoms.len()));
                new_atoms.push(a.clone());
            }
        }
        for a in &mut new_atoms {
            a.multi_center_atoms = a
                .multi_center_atoms
                .iter()
                .filter_map(|&old| remap.get(old).copied().flatten())
                .collect();
        }
        self.scene.atoms = new_atoms;
        self.scene.bonds = self
            .scene
            .bonds
            .iter()
            .filter_map(|b| {
                let na = remap.get(b.a).copied().flatten()?;
                let nb = remap.get(b.b).copied().flatten()?;
                let mut nbond = b.clone();
                nbond.a = na;
                nbond.b = nb;
                Some(nbond)
            })
            .collect();
        self.scene.aromatic_circles = self
            .scene
            .aromatic_circles
            .iter()
            .filter_map(|r| {
                let nr: Vec<usize> = r
                    .iter()
                    .filter_map(|&i| remap.get(i).copied().flatten())
                    .collect();
                (nr.len() == r.len()).then_some(nr)
            })
            .collect();
    }

    fn handle_canvas_keyboard_shortcuts(&mut self, ui: &egui::Ui, canvas_hovered: bool) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }

        // Global Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z / Cmd+Y Undo/Redo
        let (do_undo, do_redo, do_select_all, do_delete) = ui.input(|i| {
            let cmd = i.modifiers.command || i.modifiers.ctrl;
            let z = i.key_pressed(egui::Key::Z);
            let y = i.key_pressed(egui::Key::Y);
            let a = i.key_pressed(egui::Key::A);
            let del = i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete);
            (
                cmd && !i.modifiers.shift && z,
                (cmd && i.modifiers.shift && z) || (cmd && y),
                cmd && a && canvas_hovered,
                del && canvas_hovered,
            )
        });
        if do_undo {
            self.undo();
            return;
        }
        if do_redo {
            self.redo();
            return;
        }
        if do_select_all {
            self.selected_atoms = (0..self.scene.atoms.len()).collect();
            return;
        }
        if do_delete {
            self.delete_hovered_or_selected();
            return;
        }

        if !canvas_hovered {
            return;
        }

        let atom_targeted = matches!(self.hover_target, Some(HoverTarget::Atom(_)))
            || !self.selected_atoms.is_empty();

        ui.input(|i| {
            if i.modifiers.command || i.modifiers.ctrl || i.modifiers.alt {
                return;
            }
            // Tool shortcuts when no atom is hovered, or universal tool keys `V`, `1`, `2`, `3`
            if i.key_pressed(egui::Key::V) {
                self.active_tool = DrawTool::SelectBox;
            } else if i.key_pressed(egui::Key::Num1) {
                if let Some(HoverTarget::Bond(bi)) = self.hover_target {
                    self.push_undo();
                    self.scene.bonds[bi].order = BondOrder::Single;
                    self.scene.bonds[bi].stereo = BondStereo::None;
                    self.recompute_live_analytics();
                } else {
                    self.active_tool = DrawTool::Bond {
                        order: BondOrder::Single,
                        stereo: BondStereo::None,
                    };
                }
            } else if i.key_pressed(egui::Key::Num2) {
                if let Some(HoverTarget::Bond(bi)) = self.hover_target {
                    self.push_undo();
                    self.scene.bonds[bi].order = BondOrder::Double;
                    self.scene.bonds[bi].stereo = BondStereo::None;
                    self.recompute_live_analytics();
                } else {
                    self.active_tool = DrawTool::Bond {
                        order: BondOrder::Double,
                        stereo: BondStereo::None,
                    };
                }
            } else if i.key_pressed(egui::Key::Num3) {
                if let Some(HoverTarget::Bond(bi)) = self.hover_target {
                    self.push_undo();
                    self.scene.bonds[bi].order = BondOrder::Triple;
                    self.scene.bonds[bi].stereo = BondStereo::None;
                    self.recompute_live_analytics();
                } else {
                    self.active_tool = DrawTool::Bond {
                        order: BondOrder::Triple,
                        stereo: BondStereo::None,
                    };
                }
            } else if i.key_pressed(egui::Key::W) {
                if let Some(HoverTarget::Bond(bi)) = self.hover_target {
                    self.push_undo();
                    if self.scene.bonds[bi].stereo == BondStereo::UpWedge {
                        self.scene.bonds[bi].swap_ends();
                    } else {
                        self.scene.bonds[bi].order = BondOrder::Single;
                        self.scene.bonds[bi].stereo = BondStereo::UpWedge;
                    }
                    self.recompute_live_analytics();
                } else {
                    self.active_tool = DrawTool::Bond {
                        order: BondOrder::Single,
                        stereo: BondStereo::UpWedge,
                    };
                }
            } else if i.key_pressed(egui::Key::H) && !atom_targeted {
                if let Some(HoverTarget::Bond(bi)) = self.hover_target {
                    self.push_undo();
                    if self.scene.bonds[bi].stereo == BondStereo::DownHash {
                        self.scene.bonds[bi].swap_ends();
                    } else {
                        self.scene.bonds[bi].order = BondOrder::Single;
                        self.scene.bonds[bi].stereo = BondStereo::DownHash;
                    }
                    self.recompute_live_analytics();
                } else {
                    self.active_tool = DrawTool::Bond {
                        order: BondOrder::Single,
                        stereo: BondStereo::DownHash,
                    };
                }
            } else if atom_targeted {
                // ChemDraw Atom Hotkeys on hovered / selected atom (`C, N, O, S, P, F, I, B, K, H`)!
                for (key, sym) in [
                    (egui::Key::C, "C"),
                    (egui::Key::N, "N"),
                    (egui::Key::O, "O"),
                    (egui::Key::S, "S"),
                    (egui::Key::P, "P"),
                    (egui::Key::F, "F"),
                    (egui::Key::I, "I"),
                    (egui::Key::B, "B"),
                    (egui::Key::K, "K"),
                    (egui::Key::H, "H"),
                ] {
                    if i.key_pressed(key) {
                        self.apply_atom_element_or_group(sym, None);
                        break;
                    }
                }
                if i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals) {
                    self.adjust_atom_charge(1);
                } else if i.key_pressed(egui::Key::Minus) {
                    self.adjust_atom_charge(-1);
                } else if i.key_pressed(egui::Key::R) {
                    self.toggle_cip_badge();
                }
            }
        });
    }

    // ========================================================================
    // Interactive `egui::Painter` Chemical Canvas + Hit-Testing
    // ========================================================================

    fn draw_interactive_canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(220.0), avail.y.max(200.0));
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

        let center_scr = rect.center();

        let mol_to_screen = |xm: f64, ym: f64, cam_c: [f64; 2], scale: f32| -> egui::Pos2 {
            egui::pos2(
                center_scr.x + ((xm - cam_c[0]) as f32) * scale,
                center_scr.y - ((ym - cam_c[1]) as f32) * scale,
            )
        };
        let screen_to_mol = |pos: egui::Pos2, cam_c: [f64; 2], scale: f32| -> [f64; 2] {
            [
                cam_c[0] + ((pos.x - center_scr.x) / scale) as f64,
                cam_c[1] - ((pos.y - center_scr.y) / scale) as f64,
            ]
        };

        // Cursor-centered scroll-wheel zoom
        if let Some(hp) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let before_mol = screen_to_mol(hp, self.cam_center_mol, self.pixels_per_bond);
                let factor = (1.0 + scroll * 0.0022).clamp(0.75, 1.35);
                self.pixels_per_bond = (self.pixels_per_bond * factor).clamp(12.0, 110.0);
                let after_mol = screen_to_mol(hp, self.cam_center_mol, self.pixels_per_bond);
                self.cam_center_mol[0] += before_mol[0] - after_mol[0];
                self.cam_center_mol[1] += before_mol[1] - after_mol[1];
            }
        }
        if response.double_clicked() {
            self.fit_camera_to_scene();
        }

        // Hit-test under pointer
        if let Some(hp) = response.hover_pos() {
            let pm = screen_to_mol(hp, self.cam_center_mol, self.pixels_per_bond);
            self.hover_target = self.hit_test_scene(pm);
        } else {
            self.hover_target = None;
        }

        // Keyboard shortcuts & atom hotkeys
        self.handle_canvas_keyboard_shortcuts(ui, response.hovered());

        // Mouse Pan (Middle-drag, Right-drag, Space + drag, or Pan tool)
        let space_held =
            !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_down(egui::Key::Space));
        let sec_drag = response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
            || ((space_held || self.active_tool == DrawTool::Pan)
                && response.dragged_by(egui::PointerButton::Primary));

        if sec_drag {
            let delta = response.drag_delta();
            self.cam_center_mol[0] -= (delta.x / self.pixels_per_bond) as f64;
            self.cam_center_mol[1] += (delta.y / self.pixels_per_bond) as f64;
        } else {
            self.handle_primary_pointer(&response, &screen_to_mol);
        }

        // --- Render Publication Scene onto `egui::Painter` ---
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(252, 253, 255));

        // Optional crisp 1-bond coordinate grid
        if self.show_grid && self.pixels_per_bond >= 18.0 {
            let top_left_m = screen_to_mol(rect.min, self.cam_center_mol, self.pixels_per_bond);
            let bot_right_m = screen_to_mol(rect.max, self.cam_center_mol, self.pixels_per_bond);
            let gx0 = top_left_m[0].floor() as i32;
            let gx1 = bot_right_m[0].ceil() as i32;
            let gy0 = bot_right_m[1].floor() as i32;
            let gy1 = top_left_m[1].ceil() as i32;
            let grid_stroke = egui::Stroke::new(0.6, egui::Color32::from_rgb(236, 240, 246));
            for gx in gx0..=gx1 {
                let p_top = mol_to_screen(
                    gx as f64,
                    gy1 as f64,
                    self.cam_center_mol,
                    self.pixels_per_bond,
                );
                let p_bot = mol_to_screen(
                    gx as f64,
                    gy0 as f64,
                    self.cam_center_mol,
                    self.pixels_per_bond,
                );
                painter.line_segment(
                    [
                        egui::pos2(p_top.x, rect.top()),
                        egui::pos2(p_bot.x, rect.bottom()),
                    ],
                    grid_stroke,
                );
            }
            for gy in gy0..=gy1 {
                let p_l = mol_to_screen(
                    gx0 as f64,
                    gy as f64,
                    self.cam_center_mol,
                    self.pixels_per_bond,
                );
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), p_l.y),
                        egui::pos2(rect.right(), p_l.y),
                    ],
                    grid_stroke,
                );
            }
        }

        // Highlight NMR-cross-referenced atoms & selected/hovered targets
        let cur_cam = self.cam_center_mol;
        let cur_ppb = self.pixels_per_bond;
        for &ai in &self.highlighted_nmr_atoms {
            if let Some(a) = self.scene.atoms.get(ai) {
                let sp = mol_to_screen(a.x, a.y, cur_cam, cur_ppb);
                painter.circle_filled(
                    sp,
                    0.38 * cur_ppb,
                    egui::Color32::from_rgba_unmultiplied(250, 204, 21, 130),
                );
            }
        }
        for &ai in &self.selected_atoms {
            if let Some(a) = self.scene.atoms.get(ai) {
                let sp = mol_to_screen(a.x, a.y, cur_cam, cur_ppb);
                painter.circle_filled(
                    sp,
                    0.34 * cur_ppb,
                    egui::Color32::from_rgba_unmultiplied(59, 130, 246, 75),
                );
                painter.circle_stroke(
                    sp,
                    0.34 * cur_ppb,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(37, 99, 235)),
                );
            }
        }
        if let Some(ht) = self.hover_target {
            match ht {
                HoverTarget::Atom(ai) if ai < self.scene.atoms.len() => {
                    let a = &self.scene.atoms[ai];
                    let sp = mol_to_screen(a.x, a.y, cur_cam, cur_ppb);
                    painter.circle_stroke(
                        sp,
                        0.32 * cur_ppb,
                        egui::Stroke::new(1.8, egui::Color32::from_rgb(16, 185, 129)),
                    );
                }
                HoverTarget::Bond(bi) if bi < self.scene.bonds.len() => {
                    let b = &self.scene.bonds[bi];
                    if b.a < self.scene.atoms.len() && b.b < self.scene.atoms.len() {
                        let p1 = mol_to_screen(
                            self.scene.atoms[b.a].x,
                            self.scene.atoms[b.a].y,
                            cur_cam,
                            cur_ppb,
                        );
                        let p2 = mol_to_screen(
                            self.scene.atoms[b.b].x,
                            self.scene.atoms[b.b].y,
                            cur_cam,
                            cur_ppb,
                        );
                        painter.line_segment(
                            [p1, p2],
                            egui::Stroke::new(
                                0.28 * cur_ppb,
                                egui::Color32::from_rgba_unmultiplied(16, 185, 129, 65),
                            ),
                        );
                    }
                }
                _ => {}
            }
        }

        // Paint exact `compile_scene_primitives` vector geometry!
        let compiled = compile_scene_primitives(&self.scene, &self.style);
        let pt_to_scr = |pt: [f64; 2]| -> egui::Pos2 {
            let raw_x_pt = pt[0] - compiled.shift_x_pt;
            let raw_y_pt = pt[1] - compiled.shift_y_pt;
            let xm = raw_x_pt / self.style.bond_length_pt;
            let ym = -raw_y_pt / self.style.bond_length_pt;
            mol_to_screen(xm, ym, cur_cam, cur_ppb)
        };
        let pt_scale = cur_ppb / (self.style.bond_length_pt as f32);

        for pr in &compiled.primitives {
            paint_vector_primitive(&painter, pr, &pt_to_scr, pt_scale);
        }

        // Paint live interactive rubber-band drag preview
        self.paint_drag_preview(&painter, &mol_to_screen);

        // Subtle HUD status pill in bottom-left of canvas
        painter.rect_stroke(
            rect,
            3.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(203, 213, 225)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            egui::pos2(rect.left() + 8.0, rect.bottom() - 6.0),
            egui::Align2::LEFT_BOTTOM,
            &self.status_msg,
            egui::FontId::proportional(11.5),
            egui::Color32::from_rgb(51, 65, 85),
        );
    }

    fn hit_test_scene(&self, pm: [f64; 2]) -> Option<HoverTarget> {
        // 1. Atoms (within 0.30 bond lengths)
        let mut best_atom: Option<(usize, f64)> = None;
        for (i, a) in self.scene.atoms.iter().enumerate() {
            if a.is_multi_center() {
                continue;
            }
            let d = (a.x - pm[0]).hypot(a.y - pm[1]);
            if d < 0.30 && best_atom.is_none_or(|(_, bd)| d < bd) {
                best_atom = Some((i, d));
            }
        }
        if let Some((i, _)) = best_atom {
            return Some(HoverTarget::Atom(i));
        }

        // 2. Bonds (distance to segment < 0.20 bond lengths)
        let mut best_bond: Option<(usize, f64)> = None;
        for (bi, b) in self.scene.bonds.iter().enumerate() {
            if b.a >= self.scene.atoms.len() || b.b >= self.scene.atoms.len() {
                continue;
            }
            let ax = self.scene.atoms[b.a].x;
            let ay = self.scene.atoms[b.a].y;
            let abx = self.scene.atoms[b.b].x - ax;
            let aby = self.scene.atoms[b.b].y - ay;
            let ab2 = (abx * abx + aby * aby).max(1e-9);
            let t = (((pm[0] - ax) * abx + (pm[1] - ay) * aby) / ab2).clamp(0.0, 1.0);
            let d = (pm[0] - (ax + t * abx)).hypot(pm[1] - (ay + t * aby));
            if d < 0.20 && best_bond.is_none_or(|(_, bd)| d < bd) {
                best_bond = Some((bi, d));
            }
        }
        if let Some((bi, _)) = best_bond {
            return Some(HoverTarget::Bond(bi));
        }

        // 3. Reaction arrows, curved arrows, orbitals, brackets, TLC, embedded 1D spectra, texts
        for (i, arr) in self.scene.reaction_arrows.iter().enumerate() {
            let abx = arr.to[0] - arr.from[0];
            let aby = arr.to[1] - arr.from[1];
            let ab2 = (abx * abx + aby * aby).max(1e-9);
            let t =
                (((pm[0] - arr.from[0]) * abx + (pm[1] - arr.from[1]) * aby) / ab2).clamp(0.0, 1.0);
            if (pm[0] - (arr.from[0] + t * abx)).hypot(pm[1] - (arr.from[1] + t * aby)) < 0.25 {
                return Some(HoverTarget::ReactionArrow(i));
            }
        }
        for (i, ca) in self.scene.curved_arrows.iter().enumerate() {
            let mid = [
                0.5 * (ca.ctrl1[0] + ca.ctrl2[0]),
                0.5 * (ca.ctrl1[1] + ca.ctrl2[1]),
            ];
            if (pm[0] - mid[0]).hypot(pm[1] - mid[1]) < 0.45 {
                return Some(HoverTarget::CurvedArrow(i));
            }
        }
        for (i, orb) in self.scene.orbitals.iter().enumerate() {
            if (pm[0] - orb.center[0]).hypot(pm[1] - orb.center[1]) < orb.size_mol * 0.85 {
                return Some(HoverTarget::Orbital(i));
            }
        }
        for (i, tlc) in self.scene.tlc_plates.iter().enumerate() {
            if pm[0] >= tlc.origin[0]
                && pm[0] <= tlc.origin[0] + tlc.width_mol
                && pm[1] >= tlc.origin[1]
                && pm[1] <= tlc.origin[1] + tlc.height_mol
            {
                return Some(HoverTarget::TlcPlate(i));
            }
        }
        for (i, sp) in self.scene.spectra_1d.iter().enumerate() {
            if pm[0] >= sp.origin[0]
                && pm[0] <= sp.origin[0] + sp.width_mol
                && pm[1] >= sp.origin[1]
                && pm[1] <= sp.origin[1] + sp.height_mol
            {
                return Some(HoverTarget::Spectrum1D(i));
            }
        }
        for (i, txt) in self.scene.texts.iter().enumerate() {
            if (pm[0] - txt.pos[0]).abs() < 1.2 && (pm[1] - txt.pos[1]).abs() < 0.45 {
                return Some(HoverTarget::Text(i));
            }
        }
        None
    }

    fn handle_primary_pointer(
        &mut self,
        response: &egui::Response,
        screen_to_mol: &dyn Fn(egui::Pos2, [f64; 2], f32) -> [f64; 2],
    ) {
        let cur_cam = self.cam_center_mol;
        let cur_ppb = self.pixels_per_bond;

        // 1. Single Click actions (sprouting, bond order cycling, ring stamping/fusion, erasing)
        if response.clicked() {
            let Some(pos) = response.interact_pointer_pos() else {
                return;
            };
            let pm = screen_to_mol(pos, cur_cam, cur_ppb);
            let hit = self.hit_test_scene(pm);

            match self.active_tool.clone() {
                DrawTool::SelectBox | DrawTool::Lasso => {
                    if let Some(HoverTarget::Atom(ai)) = hit {
                        self.selected_atoms.clear();
                        self.selected_atoms.insert(ai);
                    } else if let Some(HoverTarget::Bond(bi)) = hit {
                        self.selected_atoms.clear();
                        self.selected_atoms.insert(self.scene.bonds[bi].a);
                        self.selected_atoms.insert(self.scene.bonds[bi].b);
                    } else {
                        self.selected_atoms.clear();
                    }
                }
                DrawTool::Eraser => {
                    self.hover_target = hit;
                    self.delete_hovered_or_selected();
                }
                DrawTool::Bond { order, stereo } => {
                    self.push_undo();
                    match hit {
                        Some(HoverTarget::Atom(ai)) => {
                            // Sprout new bond along `next_sprout_direction` (chair axial/equatorial aware!)
                            let dir = self.next_sprout_direction(ai);
                            let q = [
                                self.scene.atoms[ai].x + dir[0],
                                self.scene.atoms[ai].y + dir[1],
                            ];
                            let new_b = self.find_or_add_carbon(q);
                            let mut sb = SceneBond::new(ai, new_b, order);
                            sb.stereo = stereo;
                            self.scene.add_bond(sb);
                        }
                        Some(HoverTarget::Bond(bi)) => {
                            // Click existing bond: cycle single->double->triple or flip/set wedge/hash
                            let b = &mut self.scene.bonds[bi];
                            if stereo != BondStereo::None {
                                if b.stereo == stereo {
                                    b.swap_ends();
                                } else {
                                    b.order = BondOrder::Single;
                                    b.stereo = stereo;
                                }
                            } else if order == BondOrder::Single && b.stereo == BondStereo::None {
                                b.order = match b.order {
                                    BondOrder::Single => BondOrder::Double,
                                    BondOrder::Double => BondOrder::Triple,
                                    _ => BondOrder::Single,
                                };
                            } else {
                                b.order = order;
                                b.stereo = stereo;
                            }
                        }
                        _ => {
                            let a0 = self.scene.add_atom(SceneAtom::carbon(pm[0], pm[1]));
                            let ang = 30.0_f64.to_radians();
                            let a1 = self
                                .scene
                                .add_atom(SceneAtom::carbon(pm[0] + ang.cos(), pm[1] + ang.sin()));
                            let mut sb = SceneBond::new(a0, a1, order);
                            sb.stereo = stereo;
                            self.scene.add_bond(sb);
                        }
                    }
                    self.recompute_live_analytics();
                }
                DrawTool::Chain => {
                    self.push_undo();
                    let start = if let Some(HoverTarget::Atom(ai)) = hit {
                        [self.scene.atoms[ai].x, self.scene.atoms[ai].y]
                    } else {
                        pm
                    };
                    let pts = self.compute_chain_points(start, [start[0] + 3.5, start[1]]);
                    let mut prev = self.find_or_add_carbon(start);
                    for p in pts {
                        let nxt = self.find_or_add_carbon(p);
                        self.add_bond_if_missing(prev, nxt, BondOrder::Single, BondStereo::None);
                        prev = nxt;
                    }
                    self.recompute_live_analytics();
                }
                DrawTool::RingStamp { template_id } => {
                    self.stamp_or_fuse_template(&template_id, pm, hit);
                }
                DrawTool::Orbital { orbital_type } => {
                    self.push_undo();
                    let c = if let Some(HoverTarget::Atom(ai)) = hit {
                        [self.scene.atoms[ai].x, self.scene.atoms[ai].y]
                    } else {
                        pm
                    };
                    self.scene.orbitals.push(SceneOrbital {
                        center: c,
                        size_mol: 1.05,
                        angle_deg: 90.0,
                        orbital_type,
                        shading: OrbitalShading::ShadedAndOpen,
                        color: None,
                    });
                }
                DrawTool::TlcPlate => {
                    if let Some(tpl) = self.library.get("tlc_plate").cloned() {
                        self.push_undo();
                        self.scene.merge_at(&tpl.scene, pm[0], pm[1]);
                    }
                }
                DrawTool::TextLabel => {
                    self.push_undo();
                    if let Some(HoverTarget::Atom(ai)) = hit {
                        self.scene.atoms[ai].text = Some(self.active_text_caption.clone());
                    } else {
                        self.scene.texts.push(SceneText {
                            pos: pm,
                            text: self.active_text_caption.clone(),
                            font_size_pt: None,
                            bold: false,
                            italic: false,
                            centered: false,
                            color: None,
                        });
                    }
                    self.recompute_live_analytics();
                }
                _ => {}
            }
            return;
        }

        // 2. Drag Start / Drag Update / Drag Release
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let pm = screen_to_mol(pos, cur_cam, cur_ppb);
                let hit = self.hit_test_scene(pm);
                self.active_drag = match &self.active_tool {
                    DrawTool::SelectBox => {
                        if let Some(HoverTarget::Atom(ai)) = hit {
                            if !self.selected_atoms.contains(&ai) {
                                self.selected_atoms.clear();
                                self.selected_atoms.insert(ai);
                            }
                            Some(ActiveDrag::MoveSelection {
                                last_mol: pm,
                                pushed_undo: false,
                            })
                        } else {
                            Some(ActiveDrag::BoxSelect {
                                start_mol: pm,
                                cur_mol: pm,
                            })
                        }
                    }
                    DrawTool::Lasso => Some(ActiveDrag::LassoSelect {
                        points_mol: vec![pm],
                    }),
                    DrawTool::Bond { order, stereo } => {
                        let (from_atom, start_mol) = if let Some(HoverTarget::Atom(ai)) = hit {
                            (Some(ai), [self.scene.atoms[ai].x, self.scene.atoms[ai].y])
                        } else {
                            (None, pm)
                        };
                        Some(ActiveDrag::DrawBond {
                            from_atom,
                            start_mol,
                            cur_mol: pm,
                            order: *order,
                            stereo: *stereo,
                        })
                    }
                    DrawTool::Chain => {
                        let (from_atom, start_mol) = if let Some(HoverTarget::Atom(ai)) = hit {
                            (Some(ai), [self.scene.atoms[ai].x, self.scene.atoms[ai].y])
                        } else {
                            (None, pm)
                        };
                        Some(ActiveDrag::DrawChain {
                            from_atom,
                            start_mol,
                            cur_mol: pm,
                        })
                    }
                    DrawTool::ReactionArrow { kind } => Some(ActiveDrag::DrawReactionArrow {
                        start_mol: pm,
                        cur_mol: pm,
                        kind: *kind,
                    }),
                    DrawTool::CurvedArrow { kind } => Some(ActiveDrag::DrawCurvedArrow {
                        start_mol: pm,
                        cur_mol: pm,
                        kind: *kind,
                        bow_sgn: 1.0,
                    }),
                    DrawTool::Bracket { style } => Some(ActiveDrag::DrawBracket {
                        start_mol: pm,
                        cur_mol: pm,
                        style: *style,
                    }),
                    _ => None,
                };
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let pm = screen_to_mol(pos, cur_cam, cur_ppb);
                if let Some(drag) = &mut self.active_drag {
                    match drag {
                        ActiveDrag::BoxSelect { cur_mol, .. }
                        | ActiveDrag::DrawBond { cur_mol, .. }
                        | ActiveDrag::DrawChain { cur_mol, .. }
                        | ActiveDrag::DrawReactionArrow { cur_mol, .. }
                        | ActiveDrag::DrawCurvedArrow { cur_mol, .. }
                        | ActiveDrag::DrawBracket { cur_mol, .. } => {
                            *cur_mol = pm;
                        }
                        ActiveDrag::LassoSelect { points_mol } => {
                            if points_mol
                                .last()
                                .is_none_or(|l| (l[0] - pm[0]).hypot(l[1] - pm[1]) > 0.08)
                            {
                                points_mol.push(pm);
                            }
                        }
                        ActiveDrag::MoveSelection {
                            last_mol,
                            pushed_undo,
                        } => {
                            let dx = pm[0] - last_mol[0];
                            let dy = pm[1] - last_mol[1];
                            *last_mol = pm;
                            let need_undo = !*pushed_undo;
                            *pushed_undo = true;
                            if need_undo {
                                // Push snapshot before first move delta
                                if self.undo_stack.len() >= 128 {
                                    self.undo_stack.remove(0);
                                }
                                self.undo_stack.push(self.scene.clone());
                                self.redo_stack.clear();
                            }
                            for &ai in &self.selected_atoms {
                                if let Some(a) = self.scene.atoms.get_mut(ai) {
                                    a.x += dx;
                                    a.y += dy;
                                }
                            }
                        }
                    }
                }
            }
        }

        if response.drag_stopped() {
            if let Some(drag) = self.active_drag.take() {
                self.finish_drag_gesture(drag);
            }
        }
    }

    fn finish_drag_gesture(&mut self, drag: ActiveDrag) {
        match drag {
            ActiveDrag::BoxSelect { start_mol, cur_mol } => {
                let x0 = start_mol[0].min(cur_mol[0]);
                let x1 = start_mol[0].max(cur_mol[0]);
                let y0 = start_mol[1].min(cur_mol[1]);
                let y1 = start_mol[1].max(cur_mol[1]);
                self.selected_atoms = self
                    .scene
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| {
                        !a.is_multi_center() && a.x >= x0 && a.x <= x1 && a.y >= y0 && a.y <= y1
                    })
                    .map(|(i, _)| i)
                    .collect();
            }
            ActiveDrag::LassoSelect { points_mol } => {
                if points_mol.len() >= 3 {
                    self.selected_atoms = self
                        .scene
                        .atoms
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| {
                            !a.is_multi_center() && point_in_polygon([a.x, a.y], &points_mol)
                        })
                        .map(|(i, _)| i)
                        .collect();
                }
            }
            ActiveDrag::MoveSelection { pushed_undo, .. } => {
                if pushed_undo {
                    self.recompute_live_analytics();
                }
            }
            ActiveDrag::DrawBond {
                from_atom,
                start_mol,
                cur_mol,
                order,
                stereo,
            } => {
                if (cur_mol[0] - start_mol[0]).hypot(cur_mol[1] - start_mol[1]) > 0.18 {
                    self.push_undo();
                    let (snapped_end, target_atom) =
                        self.snap_bond_endpoint(start_mol, cur_mol, from_atom);
                    let u = from_atom.unwrap_or_else(|| self.find_or_add_carbon(start_mol));
                    let v = target_atom.unwrap_or_else(|| self.find_or_add_carbon(snapped_end));
                    if u != v {
                        self.add_bond_if_missing(u, v, order, stereo);
                    }
                    self.recompute_live_analytics();
                }
            }
            ActiveDrag::DrawChain {
                from_atom,
                start_mol,
                cur_mol,
            } => {
                if (cur_mol[0] - start_mol[0]).hypot(cur_mol[1] - start_mol[1]) > 0.25 {
                    self.push_undo();
                    let pts = self.compute_chain_points(start_mol, cur_mol);
                    let mut prev = from_atom.unwrap_or_else(|| self.find_or_add_carbon(start_mol));
                    for p in pts {
                        let nxt = self.find_or_add_carbon(p);
                        self.add_bond_if_missing(prev, nxt, BondOrder::Single, BondStereo::None);
                        prev = nxt;
                    }
                    self.recompute_live_analytics();
                }
            }
            ActiveDrag::DrawReactionArrow {
                start_mol,
                cur_mol,
                kind,
            } => {
                if (cur_mol[0] - start_mol[0]).hypot(cur_mol[1] - start_mol[1]) > 0.35 {
                    self.push_undo();
                    let end = if self.snap_angles {
                        let dx = cur_mol[0] - start_mol[0];
                        let dy = cur_mol[1] - start_mol[1];
                        let len = dx.hypot(dy);
                        let step = 15.0_f64.to_radians();
                        let ang = (dy.atan2(dx) / step).round() * step;
                        [
                            start_mol[0] + len * ang.cos(),
                            start_mol[1] + len * ang.sin(),
                        ]
                    } else {
                        cur_mol
                    };
                    self.scene.reaction_arrows.push(SceneReactionArrow {
                        from: start_mol,
                        to: end,
                        kind,
                        above_text: (!self.arrow_above_input.trim().is_empty())
                            .then(|| self.arrow_above_input.trim().to_string()),
                        below_text: (!self.arrow_below_input.trim().is_empty())
                            .then(|| self.arrow_below_input.trim().to_string()),
                        color: None,
                    });
                    self.recompute_live_analytics();
                }
            }
            ActiveDrag::DrawCurvedArrow {
                start_mol,
                cur_mol,
                kind,
                bow_sgn,
            } => {
                let d = (cur_mol[0] - start_mol[0]).hypot(cur_mol[1] - start_mol[1]);
                if d > 0.25 {
                    self.push_undo();
                    self.scene.curved_arrows.push(SceneCurvedArrow::from_chord(
                        start_mol,
                        cur_mol,
                        bow_sgn * (0.42 * d).clamp(0.35, 1.4),
                        kind,
                    ));
                }
            }
            ActiveDrag::DrawBracket {
                start_mol,
                cur_mol,
                style,
            } => {
                if (cur_mol[0] - start_mol[0]).abs() > 0.3
                    && (cur_mol[1] - start_mol[1]).abs() > 0.3
                {
                    self.push_undo();
                    self.scene.brackets.push(SceneBracket {
                        min: [start_mol[0].min(cur_mol[0]), start_mol[1].min(cur_mol[1])],
                        max: [start_mol[0].max(cur_mol[0]), start_mol[1].max(cur_mol[1])],
                        style,
                        subscript: (!self.bracket_sub_input.trim().is_empty())
                            .then(|| self.bracket_sub_input.trim().to_string()),
                        superscript: None,
                        color: None,
                    });
                }
            }
        }
    }

    fn paint_drag_preview(
        &self,
        painter: &egui::Painter,
        mol_to_screen: &dyn Fn(f64, f64, [f64; 2], f32) -> egui::Pos2,
    ) {
        let Some(drag) = &self.active_drag else {
            return;
        };
        let cam = self.cam_center_mol;
        let ppb = self.pixels_per_bond;
        let preview_stroke = egui::Stroke::new(1.6, egui::Color32::from_rgb(37, 99, 235));

        match drag {
            ActiveDrag::BoxSelect { start_mol, cur_mol }
            | ActiveDrag::DrawBracket {
                start_mol, cur_mol, ..
            } => {
                let p1 = mol_to_screen(start_mol[0], start_mol[1], cam, ppb);
                let p2 = mol_to_screen(cur_mol[0], cur_mol[1], cam, ppb);
                let r = egui::Rect::from_two_pos(p1, p2);
                painter.rect_filled(
                    r,
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(59, 130, 246, 28),
                );
                painter.rect_stroke(r, 2.0, preview_stroke, egui::StrokeKind::Inside);
            }
            ActiveDrag::LassoSelect { points_mol } => {
                let pts: Vec<egui::Pos2> = points_mol
                    .iter()
                    .map(|p| mol_to_screen(p[0], p[1], cam, ppb))
                    .collect();
                if pts.len() >= 2 {
                    painter.add(egui::Shape::line(pts, preview_stroke));
                }
            }
            ActiveDrag::DrawBond {
                from_atom,
                start_mol,
                cur_mol,
                ..
            } => {
                let (snapped_end, _) = self.snap_bond_endpoint(*start_mol, *cur_mol, *from_atom);
                let p1 = mol_to_screen(start_mol[0], start_mol[1], cam, ppb);
                let p2 = mol_to_screen(snapped_end[0], snapped_end[1], cam, ppb);
                painter.line_segment([p1, p2], preview_stroke);
            }
            ActiveDrag::DrawChain {
                start_mol, cur_mol, ..
            } => {
                let pts = self.compute_chain_points(*start_mol, *cur_mol);
                let mut scr_pts = Vec::with_capacity(pts.len() + 1);
                scr_pts.push(mol_to_screen(start_mol[0], start_mol[1], cam, ppb));
                for p in &pts {
                    scr_pts.push(mol_to_screen(p[0], p[1], cam, ppb));
                }
                painter.add(egui::Shape::line(scr_pts.clone(), preview_stroke));
                if let Some(&last_pos) = scr_pts.last() {
                    painter.text(
                        egui::pos2(last_pos.x + 8.0, last_pos.y - 8.0),
                        egui::Align2::LEFT_BOTTOM,
                        format!("C{}", pts.len() + 1),
                        egui::FontId::monospace(11.5),
                        egui::Color32::from_rgb(37, 99, 235),
                    );
                }
            }
            ActiveDrag::DrawReactionArrow {
                start_mol, cur_mol, ..
            }
            | ActiveDrag::DrawCurvedArrow {
                start_mol, cur_mol, ..
            } => {
                let p1 = mol_to_screen(start_mol[0], start_mol[1], cam, ppb);
                let p2 = mol_to_screen(cur_mol[0], cur_mol[1], cam, ppb);
                painter.line_segment([p1, p2], preview_stroke);
            }
            _ => {}
        }
    }

    fn find_or_add_carbon(&mut self, pos_mol: [f64; 2]) -> usize {
        if let Some(i) = (0..self.scene.atoms.len()).find(|&i| {
            !self.scene.atoms[i].is_multi_center()
                && (self.scene.atoms[i].x - pos_mol[0]).hypot(self.scene.atoms[i].y - pos_mol[1])
                    < 0.20
        }) {
            return i;
        }
        self.scene
            .add_atom(SceneAtom::carbon(pos_mol[0], pos_mol[1]))
    }

    fn add_bond_if_missing(&mut self, a: usize, b: usize, order: BondOrder, stereo: BondStereo) {
        if let Some(existing) = self
            .scene
            .bonds
            .iter_mut()
            .find(|eb| (eb.a == a && eb.b == b) || (eb.a == b && eb.b == a))
        {
            existing.order = match existing.order {
                BondOrder::Single if order == BondOrder::Single && stereo == BondStereo::None => {
                    BondOrder::Double
                }
                BondOrder::Double if order == BondOrder::Single && stereo == BondStereo::None => {
                    BondOrder::Triple
                }
                _ => order,
            };
            existing.stereo = stereo;
            return;
        }
        let mut sb = SceneBond::new(a, b, order);
        sb.stereo = stereo;
        self.scene.add_bond(sb);
    }

    fn stamp_or_fuse_template(
        &mut self,
        template_id: &str,
        pm: [f64; 2],
        hit: Option<HoverTarget>,
    ) {
        // Check if it's a pure polygon ring (`3..=8` or `benzene`) clicked on an existing bond -> RING FUSION!
        let ring_fusion_spec: Option<(usize, bool)> = match template_id {
            "benzene" => Some((6, true)),
            "cyclopropane" => Some((3, false)),
            "cyclobutane" => Some((4, false)),
            "cyclopentane" => Some((5, false)),
            "cyclohexane" => Some((6, false)),
            "cycloheptane" => Some((7, false)),
            "cyclooctane" => Some((8, false)),
            _ => None,
        };

        if let (Some((m, arom)), Some(HoverTarget::Bond(bi))) = (ring_fusion_spec, hit) {
            self.push_undo();
            fuse_polygon_on_bond(&mut self.scene, bi, m, arom);
            self.recompute_live_analytics();
            self.status_msg = format!("Fused {m}-membered ring onto bond #{bi}");
            return;
        }

        let Some(tpl) = self.library.get(template_id).cloned() else {
            self.status_msg = format!("Template `{template_id}` not found");
            return;
        };

        self.push_undo();
        if let Some(HoverTarget::Atom(ai)) = hit {
            if !tpl.scene.atoms.is_empty() {
                // Sprout bond from `ai` and attach vertex 0 of template!
                let dir = self.next_sprout_direction(ai);
                let attach_pos = [
                    self.scene.atoms[ai].x + dir[0],
                    self.scene.atoms[ai].y + dir[1],
                ];
                let shift_x = attach_pos[0] - tpl.scene.atoms[0].x;
                let shift_y = attach_pos[1] - tpl.scene.atoms[0].y;
                let base = self.scene.merge_at(&tpl.scene, shift_x, shift_y);
                self.scene.add_bond(SceneBond::single(ai, base));
                self.recompute_live_analytics();
                self.status_msg = format!("Sprouted {} from atom #{}", tpl.name, ai);
                return;
            }
        }

        let [cx, cy] = tpl.scene.atom_centroid();
        self.scene.merge_at(&tpl.scene, pm[0] - cx, pm[1] - cy);
        self.recompute_live_analytics();
        self.status_msg = format!("Stamped {}", tpl.name);
    }

    // ========================================================================
    // Right Drawer: Live Properties & Reaction Stoichiometry
    // ========================================================================

    fn draw_properties_and_stoichiometry_drawer(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_salt("octet_right_props_scroll")
            .show(ui, |ui| {
                ui.heading(
                    egui::RichText::new("Chemical Properties")
                        .size(15.0)
                        .strong(),
                );
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    egui::Grid::new("octet_mol_props_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Formula:");
                            ui.monospace(&self.cached_props.formula_hill);
                            ui.end_row();

                            ui.label("Mol. Weight:");
                            ui.monospace(format!(
                                "{:.2} g/mol",
                                self.cached_props.molecular_weight
                            ));
                            ui.end_row();

                            ui.label("Exact Mass:");
                            ui.monospace(format!("{:.5} Da", self.cached_props.exact_mass));
                            ui.end_row();

                            ui.label("Heavy / Rings:");
                            ui.monospace(format!(
                                "{} atoms / {} rings",
                                self.cached_props.heavy_atoms, self.cached_props.rings
                            ));
                            ui.end_row();

                            ui.label("HBD / HBA:");
                            ui.monospace(format!(
                                "{} / {}",
                                self.cached_props.h_bond_donors, self.cached_props.h_bond_acceptors
                            ));
                            ui.end_row();
                        });
                    if !self.cached_props.elemental_analysis.is_empty() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Elem. Anal.: {}",
                                self.cached_props.elemental_analysis
                            ))
                            .small()
                            .monospace(),
                        );
                    }
                });

                ui.add_space(8.0);
                ui.heading(
                    egui::RichText::new("Reaction Stoichiometry")
                        .size(14.5)
                        .strong(),
                );
                ui.horizontal(|ui| {
                    ui.label("Scale (mmol):");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.ref_scale_mmol)
                                .speed(0.1)
                                .range(0.01..=1000.0),
                        )
                        .changed()
                    {
                        self.cached_stoich =
                            compute_builtin_stoichiometry(&self.scene, self.ref_scale_mmol);
                    }
                });

                egui::Grid::new("octet_stoich_grid")
                    .striped(true)
                    .min_col_width(40.0)
                    .show(ui, |ui| {
                        ui.strong("Role");
                        ui.strong("Formula");
                        ui.strong("MW");
                        ui.strong("Mass (mg)");
                        ui.end_row();
                        for row in &self.cached_stoich {
                            ui.label(&row.role);
                            ui.monospace(&row.formula);
                            ui.monospace(format!("{:.1}", row.mw));
                            ui.monospace(format!("{:.1}", row.mass_mg));
                            ui.end_row();
                        }
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    egui::RichText::new("Arrow / Bracket Labels")
                        .small()
                        .strong(),
                );
                ui.horizontal(|ui| {
                    ui.label("Above →:");
                    ui.text_edit_singleline(&mut self.arrow_above_input);
                });
                ui.horizontal(|ui| {
                    ui.label("Below →:");
                    ui.text_edit_singleline(&mut self.arrow_below_input);
                });
                ui.horizontal(|ui| {
                    ui.label("Caption:");
                    ui.text_edit_singleline(&mut self.active_text_caption);
                });
            });
    }

    // ========================================================================
    // Bottom Panel: Live Predicted ¹H & ¹³C NMR Inspector + Interactive 1D Plot
    // ========================================================================

    fn draw_bottom_nmr_inspector(&mut self, ui: &mut egui::Ui) {
        self.highlighted_nmr_atoms.clear();
        // If user is hovering an atom on the structure canvas, highlight its peak in the 1D spectrum!
        let hovered_canvas_atom = match self.hover_target {
            Some(HoverTarget::Atom(ai)) => Some(ai),
            _ => None,
        };

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Live Predicted 1D NMR").strong());
            ui.selectable_value(&mut self.nmr_tab, NmrViewTab::Proton1H, "¹H NMR");
            ui.selectable_value(&mut self.nmr_tab, NmrViewTab::Carbon13C, "¹³C{¹H} NMR");
            if ui
                .add(
                    egui::Slider::new(&mut self.spectrometer_mhz, 60.0..=900.0)
                        .text("MHz")
                        .fixed_decimals(0),
                )
                .changed()
            {
                self.recompute_live_analytics();
            }
            if ui
                .small_button("📌 Embed 1D Plot on Canvas")
                .on_hover_text("Insert publication 1D NMR spectrum directly onto the drawing canvas for SVG/PDF/PNG export")
                .clicked()
            {
                self.embed_1d_spectrum_on_canvas();
            }
        });

        let signals = match self.nmr_tab {
            NmrViewTab::Proton1H => &self.cached_1h,
            NmrViewTab::Carbon13C => &self.cached_13c,
        };

        ui.columns(2, |cols| {
            // Left column: Interactive 1D Lorentzian Multiplet Spectrum Plot
            let avail = cols[0].available_size();
            let plot_size = egui::vec2(avail.x.max(180.0), avail.y.max(110.0));
            let (rect, resp) = cols[0].allocate_exact_size(plot_size, egui::Sense::hover());
            let painter = cols[0].painter_at(rect);
            painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(15, 23, 42));

            let (ppm_lo, ppm_hi) = match self.nmr_tab {
                NmrViewTab::Proton1H => (-0.5_f64, 11.5_f64),
                NmrViewTab::Carbon13C => (0.0_f64, 220.0_f64),
            };
            let span = ppm_hi - ppm_lo;
            let axis_y = rect.bottom() - 18.0;
            let top_y = rect.top() + 12.0;
            let h_amp = (axis_y - top_y - 4.0).max(20.0);

            let ppm_to_x = |ppm: f64| -> f32 {
                let u = ((ppm_hi - ppm) / span).clamp(0.0, 1.0) as f32;
                rect.left() + 8.0 + u * (rect.width() - 16.0)
            };

            // Synthesize live Lorentzian trace
            let n_pts = 380_usize;
            let freq = match self.nmr_tab {
                NmrViewTab::Proton1H => self.spectrometer_mhz.max(60.0),
                NmrViewTab::Carbon13C => (self.spectrometer_mhz * 0.251_45).max(25.0),
            };
            let gamma = (0.7 / freq).max(span * 0.0012);
            let mut sublines = Vec::new();
            for sig in signals {
                for (dhz, amp) in expand_multiplet_lines_hz(&sig.multiplicity, &sig.couplings_hz) {
                    sublines.push((sig.ppm + dhz / freq, amp * sig.integral.max(0.5)));
                }
            }
            let mut ys = vec![0.0_f64; n_pts];
            let mut ymax = 1e-9_f64;
            for (k, y_slot) in ys.iter_mut().enumerate() {
                let ppm = ppm_hi - (k as f64 / (n_pts - 1) as f64) * span;
                let mut v = 0.0;
                for &(lp, la) in &sublines {
                    let d = (ppm - lp) / gamma;
                    if d.abs() < 35.0 {
                        v += la / (1.0 + d * d);
                    }
                }
                *y_slot = v;
                ymax = ymax.max(v);
            }
            let trace_pts: Vec<egui::Pos2> = (0..n_pts)
                .map(|k| {
                    let ppm = ppm_hi - (k as f64 / (n_pts - 1) as f64) * span;
                    let norm = (ys[k] / ymax) as f32;
                    egui::pos2(ppm_to_x(ppm), axis_y - 3.0 - norm * h_amp)
                })
                .collect();
            painter.line_segment(
                [
                    egui::pos2(rect.left() + 6.0, axis_y),
                    egui::pos2(rect.right() - 6.0, axis_y),
                ],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 116, 139)),
            );
            painter.add(egui::Shape::line(
                trace_pts,
                egui::Stroke::new(1.4, egui::Color32::from_rgb(56, 189, 248)),
            ));

            // Ticks & hover cross-highlighting
            let step = if span > 50.0 { 20.0 } else { 1.0 };
            let mut tk = (ppm_lo / step).ceil() * step;
            while tk <= ppm_hi {
                let tx = ppm_to_x(tk);
                painter.text(
                    egui::pos2(tx, rect.bottom() - 2.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("{tk:.0}"),
                    egui::FontId::monospace(9.5),
                    egui::Color32::from_rgb(148, 163, 184),
                );
                tk += step;
            }

            if let Some(hp) = resp.hover_pos() {
                let u =
                    ((hp.x - (rect.left() + 8.0)) / (rect.width() - 16.0).max(1.0)).clamp(0.0, 1.0);
                let cur_ppm = ppm_hi - (u as f64) * span;
                painter.text(
                    egui::pos2(rect.right() - 8.0, rect.top() + 4.0),
                    egui::Align2::RIGHT_TOP,
                    format!("δ {cur_ppm:.2} ppm"),
                    egui::FontId::monospace(10.5),
                    egui::Color32::from_rgb(226, 232, 240),
                );
                for sig in signals {
                    if (sig.ppm - cur_ppm).abs() < span * 0.025 {
                        for &ai in &sig.atom_indices {
                            self.highlighted_nmr_atoms.insert(ai);
                        }
                    }
                }
            }
            if let Some(hai) = hovered_canvas_atom {
                for sig in signals {
                    if sig.atom_indices.contains(&hai) {
                        let sx = ppm_to_x(sig.ppm);
                        painter.line_segment(
                            [egui::pos2(sx, top_y), egui::pos2(sx, axis_y)],
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(250, 204, 21)),
                        );
                    }
                }
            }

            // Right column: Multiplet & Coupling Shift Table
            egui::ScrollArea::vertical()
                .id_salt("octet_nmr_table_scroll")
                .show(&mut cols[1], |ui| {
                    egui::Grid::new("octet_nmr_peaks_grid")
                        .striped(true)
                        .min_col_width(44.0)
                        .show(ui, |ui| {
                            ui.strong("δ (ppm)");
                            ui.strong("Mult.");
                            ui.strong("J (Hz)");
                            ui.strong("Int.");
                            ui.strong("Assignment");
                            ui.end_row();

                            for sig in signals {
                                let is_hl = hovered_canvas_atom
                                    .is_some_and(|a| sig.atom_indices.contains(&a));
                                let col = if is_hl {
                                    egui::Color32::from_rgb(250, 204, 21)
                                } else {
                                    ui.visuals().text_color()
                                };
                                let r_shift = ui.label(
                                    egui::RichText::new(format!("{:.2}", sig.ppm))
                                        .monospace()
                                        .color(col),
                                );
                                ui.monospace(&sig.multiplicity);
                                let j_txt = if sig.couplings_hz.is_empty() {
                                    "—".to_string()
                                } else {
                                    sig.couplings_hz
                                        .iter()
                                        .map(|j| format!("{j:.1}"))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                ui.monospace(j_txt);
                                ui.monospace(format!("{:.0}", sig.integral));
                                ui.label(sig.assignment_label.as_deref().unwrap_or("—"));
                                if r_shift.hovered() {
                                    for &ai in &sig.atom_indices {
                                        self.highlighted_nmr_atoms.insert(ai);
                                    }
                                }
                                ui.end_row();
                            }
                        });
                });
        });
    }

    // ========================================================================
    // Template Gallery Window & Multi-Format Import/Export Window
    // ========================================================================

    fn draw_template_gallery_window(&mut self, ctx: &egui::Context) {
        if !self.show_template_gallery {
            return;
        }
        let mut open = self.show_template_gallery;
        let mut chosen_id: Option<String> = None;

        egui::Window::new("📚 Octet Structure & Clipware Template Gallery")
            .open(&mut open)
            .default_width(540.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.text_edit_singleline(&mut self.template_search_query);
                    if ui.small_button("Clear").clicked() {
                        self.template_search_query.clear();
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    for &cat in TemplateCategory::ALL {
                        ui.selectable_value(&mut self.template_category_filter, cat, cat.name());
                    }
                });
                ui.separator();

                let matches: Vec<(String, String, usize)> =
                    if self.template_search_query.trim().is_empty() {
                        self.library
                            .by_category(self.template_category_filter)
                            .into_iter()
                            .map(|t| (t.id.clone(), t.name.clone(), t.scene.atoms.len()))
                            .collect()
                    } else {
                        self.library
                            .search(&self.template_search_query)
                            .into_iter()
                            .map(|t| (t.id.clone(), t.name.clone(), t.scene.atoms.len()))
                            .collect()
                    };

                egui::ScrollArea::vertical()
                    .id_salt("octet_tpl_gallery_scroll")
                    .max_height(280.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for (tid, tname, natoms) in matches {
                                let lbl = if natoms > 0 {
                                    format!("{tname} ({natoms}a)")
                                } else {
                                    tname
                                };
                                if ui.button(lbl).clicked() {
                                    chosen_id = Some(tid);
                                }
                            }
                        });
                    });
            });

        if let Some(tid) = chosen_id {
            self.active_tool = DrawTool::RingStamp {
                template_id: tid.clone(),
            };
            let cam = self.cam_center_mol;
            self.stamp_or_fuse_template(&tid, cam, None);
        }
        self.show_template_gallery = open;
    }

    fn draw_io_modal_window(&mut self, ctx: &egui::Context) {
        if !self.show_io_modal {
            return;
        }
        let mut open = self.show_io_modal;
        let mut trigger_export = false;
        let mut trigger_import = false;

        egui::Window::new("📤 Multi-Format Chemical & Publication Import / Export")
            .open(&mut open)
            .default_width(640.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Format:");
                    let prev_fmt = self.selected_io_format;
                    egui::ComboBox::from_id_salt("octet_io_format_combo")
                        .selected_text(self.selected_io_format.label())
                        .show_ui(ui, |ui| {
                            for &fmt in IoFormat::ALL {
                                ui.selectable_value(&mut self.selected_io_format, fmt, fmt.label());
                            }
                        });
                    if self.selected_io_format != prev_fmt {
                        trigger_export = true;
                    }

                    if ui.button("🔄 Generate Export").clicked() {
                        trigger_export = true;
                    }
                    if ui.button("📋 Copy to Clipboard").clicked() {
                        ui.ctx().copy_text(self.io_text_buffer.clone());
                        self.status_msg = format!(
                            "Copied {} ({} chars) to clipboard",
                            self.selected_io_format.label(),
                            self.io_text_buffer.len()
                        );
                    }
                    if self.selected_io_format.can_import()
                        && ui.button("📥 Import from Buffer").clicked()
                    {
                        trigger_import = true;
                    }
                });

                ui.label(
                    egui::RichText::new(format!(
                        "Buffer length: {} bytes  |  PDF bytes: {}  |  PNG bytes: {}",
                        self.io_text_buffer.len(),
                        self.last_pdf_bytes_len,
                        self.last_png_bytes_len
                    ))
                    .small()
                    .weak(),
                );

                egui::ScrollArea::vertical()
                    .id_salt("octet_io_buffer_scroll")
                    .max_height(310.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.io_text_buffer)
                                .font(egui::TextStyle::Monospace)
                                .desired_rows(14)
                                .desired_width(f32::INFINITY),
                        );
                    });
            });

        if trigger_export {
            self.refresh_export_buffer();
        }
        if trigger_import {
            self.import_from_buffer();
        }
        self.show_io_modal = open;
    }
}

trait SwapBondEnds {
    fn swap_ends(&mut self);
}

impl SwapBondEnds for SceneBond {
    fn swap_ends(&mut self) {
        std::mem::swap(&mut self.a, &mut self.b);
    }
}

fn to_egui_color(c: RgbaColor) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a)
}

fn paint_vector_primitive(
    painter: &egui::Painter,
    pr: &VectorPrimitive,
    pt_to_scr: &dyn Fn([f64; 2]) -> egui::Pos2,
    pt_scale: f32,
) {
    match pr {
        VectorPrimitive::Line {
            p1,
            p2,
            width_pt,
            color,
            dash_pt,
        } => {
            let s1 = pt_to_scr(*p1);
            let s2 = pt_to_scr(*p2);
            let stroke = egui::Stroke::new(
                (*width_pt as f32 * pt_scale).max(1.0),
                to_egui_color(*color),
            );
            if let Some([d1, d2]) = dash_pt {
                let d_on = (*d1 as f32 * pt_scale).max(2.0);
                let d_off = (*d2 as f32 * pt_scale).max(2.0);
                let total = s1.distance(s2);
                if total > 0.5 {
                    let dir = (s2 - s1) / total;
                    let mut t = 0.0;
                    while t < total {
                        let e = (t + d_on).min(total);
                        painter.line_segment([s1 + dir * t, s1 + dir * e], stroke);
                        t += d_on + d_off;
                    }
                }
            } else {
                painter.line_segment([s1, s2], stroke);
            }
        }
        VectorPrimitive::Polygon {
            points,
            fill,
            stroke,
        } => {
            let pts: Vec<egui::Pos2> = points.iter().map(|&p| pt_to_scr(p)).collect();
            if pts.len() >= 3 {
                let fc = fill
                    .map(to_egui_color)
                    .unwrap_or(egui::Color32::TRANSPARENT);
                let st = stroke
                    .map(|(w, c)| {
                        egui::Stroke::new((w as f32 * pt_scale).max(1.0), to_egui_color(c))
                    })
                    .unwrap_or(egui::Stroke::NONE);
                painter.add(egui::Shape::convex_polygon(pts, fc, st));
            }
        }
        VectorPrimitive::Path {
            cmds, fill, stroke, ..
        } => {
            let subpaths = flatten_path_commands(cmds, 14);
            let st = stroke
                .map(|(w, c)| egui::Stroke::new((w as f32 * pt_scale).max(1.0), to_egui_color(c)))
                .unwrap_or(egui::Stroke::NONE);
            for (pts, closed) in subpaths {
                let spts: Vec<egui::Pos2> = pts.iter().map(|&p| pt_to_scr(p)).collect();
                if spts.len() >= 3 && (closed || fill.is_some()) {
                    if let Some(fc) = fill {
                        painter.add(egui::Shape::convex_polygon(
                            spts.clone(),
                            to_egui_color(*fc),
                            egui::Stroke::NONE,
                        ));
                    }
                }
                if st != egui::Stroke::NONE && spts.len() >= 2 {
                    painter.add(egui::Shape::line(spts, st));
                }
            }
        }
        VectorPrimitive::Circle {
            center,
            radius_pt,
            fill,
            stroke,
            ..
        } => {
            let c = pt_to_scr(*center);
            let r = (*radius_pt as f32 * pt_scale).max(1.0);
            if let Some(fc) = fill {
                painter.circle_filled(c, r, to_egui_color(*fc));
            }
            if let Some((sw, sc)) = *stroke {
                painter.circle_stroke(
                    c,
                    r,
                    egui::Stroke::new((sw as f32 * pt_scale).max(1.0), to_egui_color(sc)),
                );
            }
        }
        VectorPrimitive::Text {
            pos,
            text,
            font_size_pt,
            color,
            ..
        } => {
            let sp = pt_to_scr(*pos);
            let fsz = (*font_size_pt as f32 * pt_scale).clamp(7.5, 42.0);
            painter.text(
                sp,
                egui::Align2::LEFT_BOTTOM,
                text,
                egui::FontId::proportional(fsz),
                to_egui_color(*color),
            );
        }
    }
}

fn point_in_polygon(pt: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if ((a[1] > pt[1]) != (b[1] > pt[1]))
            && (pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
        {
            inside = !inside;
        }
    }
    inside
}

// ============================================================================
// 5. Built-In Chemistry, Stoichiometry, 1D NMR Predictor & Multi-Format I/O
// ============================================================================

fn build_default_showcase_scene(lib: &TemplateLibrary) -> RenderScene {
    // Showcase: Aspirin (acetylsalicylic acid) -> Salicylic acid + Acetic acid hydrolysis scheme
    let mut scene = RenderScene::new();
    if let Some(bz) = lib.get("benzene") {
        // Reactant: Aspirin centered at (-3.5, 0.3)
        let r_base = scene.merge_at(&bz.scene, -3.5, 0.3);
        // Ortho COOH on top vertex (r_base + 0)
        let c_cooh = scene.add_atom(SceneAtom::carbon(
            scene.atoms[r_base].x,
            scene.atoms[r_base].y + 1.0,
        ));
        let o_c1 = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[c_cooh].x - 0.866,
            scene.atoms[c_cooh].y + 0.5,
        ));
        let oh_c1 = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[c_cooh].x + 0.866,
            scene.atoms[c_cooh].y + 0.5,
        ));
        scene.add_bond(SceneBond::single(r_base, c_cooh));
        scene.add_bond(SceneBond::double(c_cooh, o_c1));
        scene.add_bond(SceneBond::single(c_cooh, oh_c1));

        // Ortho OAc on top-right vertex (r_base + 1)
        let o_est = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[r_base + 1].x + 0.866,
            scene.atoms[r_base + 1].y + 0.5,
        ));
        let c_ac = scene.add_atom(SceneAtom::carbon(
            scene.atoms[o_est].x + 0.866,
            scene.atoms[o_est].y - 0.5,
        ));
        let o_ac = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[c_ac].x,
            scene.atoms[c_ac].y - 1.0,
        ));
        let me_ac = scene.add_atom(SceneAtom::carbon(
            scene.atoms[c_ac].x + 0.866,
            scene.atoms[c_ac].y + 0.5,
        ));
        scene.add_bond(SceneBond::single(r_base + 1, o_est));
        scene.add_bond(SceneBond::single(o_est, c_ac));
        scene.add_bond(SceneBond::double(c_ac, o_ac));
        scene.add_bond(SceneBond::single(c_ac, me_ac));

        // Curved nucleophilic attack arrow demonstration
        scene.curved_arrows.push(SceneCurvedArrow::from_chord(
            [scene.atoms[c_ac].x + 0.15, scene.atoms[c_ac].y - 0.15],
            [scene.atoms[o_ac].x + 0.22, scene.atoms[o_ac].y - 0.05],
            0.42,
            PushingArrowKind::FullTwoElectron,
        ));

        // Reaction arrow
        scene.reaction_arrows.push(SceneReactionArrow {
            from: [-0.2, 0.3],
            to: [2.2, 0.3],
            kind: ArrowKind::Solid,
            above_text: Some("NaOH, H2O".to_string()),
            below_text: Some("25 C, 96%".to_string()),
            color: None,
        });

        // Product: Salicylic acid centered at (4.2, 0.3)
        let p_base = scene.merge_at(&bz.scene, 4.2, 0.3);
        let pc = scene.add_atom(SceneAtom::with_group(
            "C",
            "COOH",
            scene.atoms[p_base].x,
            scene.atoms[p_base].y + 1.0,
        ));
        let poh = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[p_base + 1].x + 0.866,
            scene.atoms[p_base + 1].y + 0.5,
        ));
        scene.add_bond(SceneBond::single(p_base, pc));
        scene.add_bond(SceneBond::single(p_base + 1, poh));
    }
    scene
}

fn element_masses(sym: &str) -> (f64, f64) {
    // Returns (standard atomic weight, monoisotopic exact mass)
    match sym {
        "H" => (1.00794, 1.007825032),
        "D" => (2.01410, 2.014101778),
        "B" => (10.811, 11.0093054),
        "C" => (12.0107, 12.0000000),
        "N" => (14.0067, 14.003074004),
        "O" => (15.9994, 15.994914620),
        "F" => (18.9984, 18.99840322),
        "Na" => (22.9898, 22.98976928),
        "Mg" => (24.3050, 23.98504170),
        "Si" => (28.0855, 27.97692653),
        "P" => (30.9738, 30.97376163),
        "S" => (32.065, 31.97207100),
        "Cl" => (35.453, 34.96885268),
        "K" => (39.0983, 38.96370668),
        "Fe" => (55.845, 55.9349375),
        "Br" => (79.904, 78.9183371),
        "Ru" => (101.07, 101.904349),
        "Pd" => (106.42, 105.903486),
        "I" => (126.904, 126.904473),
        "Pt" => (195.084, 194.964791),
        _ => (12.0107, 12.0),
    }
}

fn add_condensed_group_counts(grp: &str, counts: &mut BTreeMap<String, usize>) {
    let g = grp.trim();
    let add = |counts: &mut BTreeMap<String, usize>, el: &str, n: usize| {
        if n > 0 {
            *counts.entry(el.to_string()).or_insert(0) += n;
        }
    };
    match g {
        "COOH" | "HOOC" | "CO2H" => {
            add(counts, "C", 1);
            add(counts, "O", 2);
            add(counts, "H", 1);
        }
        "Boc" | "NHBoc" => {
            if g.starts_with("NH") {
                add(counts, "N", 1);
                add(counts, "H", 1);
            }
            add(counts, "C", 5);
            add(counts, "H", 9);
            add(counts, "O", 2);
        }
        "Fmoc" => {
            add(counts, "C", 15);
            add(counts, "H", 11);
            add(counts, "O", 2);
        }
        "Ph" => {
            add(counts, "C", 6);
            add(counts, "H", 5);
        }
        "Ac" => {
            add(counts, "C", 2);
            add(counts, "H", 3);
            add(counts, "O", 1);
        }
        "OAc" | "AcO" => {
            add(counts, "C", 2);
            add(counts, "H", 3);
            add(counts, "O", 2);
        }
        "tBu" => {
            add(counts, "C", 4);
            add(counts, "H", 9);
        }
        "NO2" | "O2N" => {
            add(counts, "N", 1);
            add(counts, "O", 2);
        }
        "CF3" | "F3C" => {
            add(counts, "C", 1);
            add(counts, "F", 3);
        }
        "OMe" | "MeO" | "OCH3" => {
            add(counts, "C", 1);
            add(counts, "H", 3);
            add(counts, "O", 1);
        }
        "PPh3" => {
            add(counts, "P", 1);
            add(counts, "C", 18);
            add(counts, "H", 15);
        }
        _ => {
            add(counts, "C", 1);
        }
    }
}

fn compute_subset_formula_and_mass(
    scene: &RenderScene,
    subset: &[usize],
) -> (BTreeMap<String, usize>, String, f64, f64, usize, usize) {
    let n = scene.atoms.len();
    let mut bond_sums = vec![0.0_f64; n];
    for b in &scene.bonds {
        if b.a < n
            && b.b < n
            && b.a != b.b
            && !scene.atoms[b.a].is_multi_center()
            && !scene.atoms[b.b].is_multi_center()
        {
            let ord = b.order.valence_order();
            bond_sums[b.a] += ord;
            bond_sums[b.b] += ord;
        }
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut hbd = 0_usize;
    let mut hba = 0_usize;

    for &i in subset {
        let a = &scene.atoms[i];
        if a.is_multi_center() {
            continue;
        }
        if let Some(grp) = &a.text {
            add_condensed_group_counts(grp, &mut counts);
            if grp.contains("OH") || grp.contains("NH") {
                hbd += 1;
            }
            if grp.contains('O') || grp.contains('N') {
                hba += 1;
            }
            continue;
        }
        let h = a
            .explicit_h
            .unwrap_or_else(|| implicit_hydrogens(&a.label, a.charge, bond_sums[i], a.radical))
            as usize;
        *counts.entry(a.label.clone()).or_insert(0) += 1;
        if h > 0 {
            *counts.entry("H".to_string()).or_insert(0) += h;
        }
        if matches!(a.label.as_str(), "N" | "O") {
            hba += 1;
            if h > 0 {
                hbd += 1;
            }
        }
    }

    // Hill system ordering: C first, then H, then remaining elements alphabetically
    let mut hill = String::new();
    let fmt_el = |el: &str, cnt: usize| -> String {
        if cnt == 1 {
            el.to_string()
        } else {
            format!("{el}{cnt}")
        }
    };
    if let Some(&c_cnt) = counts.get("C") {
        hill.push_str(&fmt_el("C", c_cnt));
        if let Some(&h_cnt) = counts.get("H") {
            hill.push_str(&fmt_el("H", h_cnt));
        }
        for (el, &cnt) in &counts {
            if el != "C" && el != "H" && cnt > 0 {
                hill.push_str(&fmt_el(el, cnt));
            }
        }
    } else {
        for (el, &cnt) in &counts {
            if cnt > 0 {
                hill.push_str(&fmt_el(el, cnt));
            }
        }
    }
    if hill.is_empty() {
        hill = "—".to_string();
    }

    let mut mw = 0.0;
    let mut exact = 0.0;
    for (el, &cnt) in &counts {
        let (w_avg, w_iso) = element_masses(el);
        mw += w_avg * cnt as f64;
        exact += w_iso * cnt as f64;
    }

    (counts, hill, mw, exact, hbd, hba)
}

fn connected_components(scene: &RenderScene) -> Vec<Vec<usize>> {
    let n = scene.atoms.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in &scene.bonds {
        if b.a < n && b.b < n && b.a != b.b {
            adj[b.a].push(b.b);
            adj[b.b].push(b.a);
        }
    }
    for (i, a) in scene.atoms.iter().enumerate() {
        for &mc in &a.multi_center_atoms {
            if mc < n {
                adj[i].push(mc);
                adj[mc].push(i);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for start in 0..n {
        if seen[start] || scene.atoms[start].is_multi_center() {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(u) = stack.pop() {
            if !scene.atoms[u].is_multi_center() {
                comp.push(u);
            }
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        if !comp.is_empty() {
            comps.push(comp);
        }
    }
    comps
}

fn compute_builtin_properties(scene: &RenderScene) -> MolecularPropertiesSummary {
    let all: Vec<usize> = (0..scene.atoms.len())
        .filter(|&i| !scene.atoms[i].is_multi_center())
        .collect();
    if all.is_empty() {
        return MolecularPropertiesSummary {
            formula_hill: "—".to_string(),
            ..Default::default()
        };
    }
    let (counts, hill, mw, exact, hbd, hba) = compute_subset_formula_and_mass(scene, &all);
    let rings = perceive_small_rings(scene.atoms.len(), &scene.bonds).len();
    let heavy = counts
        .iter()
        .filter(|(k, _)| k.as_str() != "H" && k.as_str() != "D")
        .map(|(_, &v)| v)
        .sum();

    let elem_anal = if mw > 1e-6 {
        counts
            .iter()
            .map(|(el, &cnt)| {
                let pct = (element_masses(el).0 * cnt as f64 / mw) * 100.0;
                format!("{el} {pct:.1}%")
            })
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        String::new()
    };

    MolecularPropertiesSummary {
        formula_hill: hill,
        molecular_weight: mw,
        exact_mass: exact,
        elemental_analysis: elem_anal,
        heavy_atoms: heavy,
        rings,
        h_bond_donors: hbd,
        h_bond_acceptors: hba,
    }
}

fn compute_builtin_stoichiometry(scene: &RenderScene, ref_mmol: f64) -> Vec<StoichiometryRow> {
    let comps = connected_components(scene);
    let arrow_mid_x = scene
        .reaction_arrows
        .first()
        .map(|a| 0.5 * (a.from[0] + a.to[0]));

    let mut rows = Vec::with_capacity(comps.len());
    let mut sm_idx = 1;
    let mut prod_idx = 1;
    for comp in comps {
        let (_, hill, mw, _, _, _) = compute_subset_formula_and_mass(scene, &comp);
        let cx = comp.iter().map(|&i| scene.atoms[i].x).sum::<f64>() / comp.len() as f64;
        let role = match arrow_mid_x {
            Some(mx) if cx > mx => {
                let r = format!("Prod {prod_idx}");
                prod_idx += 1;
                r
            }
            Some(_) => {
                let r = format!("React {sm_idx}");
                sm_idx += 1;
                r
            }
            None => {
                let r = format!("Mol {sm_idx}");
                sm_idx += 1;
                r
            }
        };
        rows.push(StoichiometryRow {
            role,
            formula: hill,
            mw,
            equiv: 1.0,
            mmol: ref_mmol,
            mass_mg: mw * ref_mmol,
        });
    }
    rows
}

/// Built-in additive HOSE / Pretsch-Bühlmann-Badertscher ¹H & ¹³C NMR shift + `J`-coupling predictor!
pub fn predict_builtin_nmr_1h_13c(
    scene: &RenderScene,
) -> (Vec<SpectrumSignal1D>, Vec<SpectrumSignal1D>) {
    let n = scene.atoms.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let rings = perceive_small_rings(n, &scene.bonds);
    let mut arom_atoms = HashSet::new();
    for r in &rings {
        if r.len() == 5 || r.len() == 6 {
            let dbl_cnt = scene
                .bonds
                .iter()
                .filter(|b| {
                    (b.order == BondOrder::Double || b.order == BondOrder::Aromatic)
                        && r.contains(&b.a)
                        && r.contains(&b.b)
                })
                .count();
            if dbl_cnt >= 2 || scene.aromatic_circles.contains(r) {
                for &ai in r {
                    arom_atoms.insert(ai);
                }
            }
        }
    }

    let mut adj: Vec<Vec<(usize, BondOrder)>> = vec![Vec::new(); n];
    let mut bond_sums = vec![0.0_f64; n];
    for b in &scene.bonds {
        if b.a < n
            && b.b < n
            && b.a != b.b
            && b.order != BondOrder::HydrogenBond
            && !scene.atoms[b.a].is_multi_center()
            && !scene.atoms[b.b].is_multi_center()
        {
            adj[b.a].push((b.b, b.order));
            adj[b.b].push((b.a, b.order));
            bond_sums[b.a] += b.order.valence_order();
            bond_sums[b.b] += b.order.valence_order();
        }
    }

    let h_counts: Vec<u8> = (0..n)
        .map(|i| {
            let a = &scene.atoms[i];
            if a.is_multi_center() || a.text.is_some() {
                0
            } else {
                a.explicit_h.unwrap_or_else(|| {
                    implicit_hydrogens(&a.label, a.charge, bond_sums[i], a.radical)
                })
            }
        })
        .collect();

    let mut signals_1h = Vec::new();
    let mut signals_13c = Vec::new();

    for i in 0..n {
        let a = &scene.atoms[i];
        if a.is_multi_center() {
            continue;
        }
        // Condensed COOH / CHO group resonances
        if let Some(grp) = &a.text {
            if grp == "COOH" || grp == "HOOC" {
                signals_1h.push(SpectrumSignal1D {
                    ppm: 11.20,
                    multiplicity: "br s".to_string(),
                    couplings_hz: vec![],
                    integral: 1.0,
                    atom_indices: vec![i],
                    assignment_label: Some("COOH".to_string()),
                });
                signals_13c.push(SpectrumSignal1D {
                    ppm: 174.5,
                    multiplicity: "s".to_string(),
                    couplings_hz: vec![],
                    integral: 1.0,
                    atom_indices: vec![i],
                    assignment_label: Some("COOH".to_string()),
                });
            }
            continue;
        }

        let is_arom = arom_atoms.contains(&i);
        let has_carbonyl_o = adj[i]
            .iter()
            .any(|&(j, ord)| ord == BondOrder::Double && scene.atoms[j].label == "O");
        let n_oxy = adj[i]
            .iter()
            .filter(|&&(j, _)| scene.atoms[j].label == "O")
            .count();
        let n_nit = adj[i]
            .iter()
            .filter(|&&(j, _)| scene.atoms[j].label == "N")
            .count();
        let n_halo = adj[i]
            .iter()
            .filter(|&&(j, _)| matches!(scene.atoms[j].label.as_str(), "F" | "Cl" | "Br" | "I"))
            .count();
        let adj_to_carbonyl = adj[i].iter().any(|&(j, _)| {
            adj[j]
                .iter()
                .any(|&(k, ord)| ord == BondOrder::Double && scene.atoms[k].label == "O")
        });
        let adj_to_arom = adj[i].iter().any(|&(j, _)| arom_atoms.contains(&j));

        // --- ¹³C Shift Prediction ---
        if a.label == "C" {
            let c_ppm = if has_carbonyl_o {
                if n_oxy >= 2 {
                    171.5 // Ester / Carboxylic acid
                } else if n_nit >= 1 {
                    169.8 // Amide C'
                } else if h_counts[i] >= 1 {
                    198.0 // Aldehyde
                } else {
                    205.0 // Ketone
                }
            } else if is_arom {
                128.5
                    + 26.0 * (n_oxy as f64)
                    + 18.0 * (n_nit as f64)
                    + if h_counts[i] == 0 { 6.5 } else { 0.0 }
            } else if adj[i].iter().any(|&(_, ord)| ord == BondOrder::Double) {
                124.0 + 18.0 * (n_oxy as f64)
            } else if adj[i].iter().any(|&(_, ord)| ord == BondOrder::Triple) {
                78.5
            } else {
                // sp³ aliphatic carbon (Shoolery/Grant α/β substituent increments)
                let deg_c = adj[i].len() as f64;
                let mut base = 8.5 + 9.2 * deg_c;
                base += 42.0 * (n_oxy as f64) + 28.0 * (n_nit as f64) + 25.0 * (n_halo as f64);
                if adj_to_carbonyl {
                    base += 11.5;
                }
                if adj_to_arom {
                    base += 14.0;
                }
                base
            };
            signals_13c.push(SpectrumSignal1D {
                ppm: c_ppm.clamp(5.0, 218.0),
                multiplicity: "s".to_string(),
                couplings_hz: vec![],
                integral: 1.0,
                atom_indices: vec![i],
                assignment_label: Some(format!("C{}", i + 1)),
            });
        }

        // --- ¹H Shift + n+1 Scalar Coupling Multiplet Prediction ---
        let hc = h_counts[i];
        if hc == 0 {
            continue;
        }
        if a.label == "O" {
            let ppm = if adj_to_carbonyl {
                11.15
            } else if adj_to_arom {
                9.45
            } else {
                3.65
            };
            signals_1h.push(SpectrumSignal1D {
                ppm,
                multiplicity: "br s".to_string(),
                couplings_hz: vec![],
                integral: hc as f64,
                atom_indices: vec![i],
                assignment_label: Some(if adj_to_carbonyl {
                    "COOH".to_string()
                } else {
                    "OH".to_string()
                }),
            });
        } else if a.label == "N" {
            let ppm = if adj_to_carbonyl {
                8.05
            } else if is_arom || adj_to_arom {
                6.85
            } else {
                2.45
            };
            signals_1h.push(SpectrumSignal1D {
                ppm,
                multiplicity: "br s".to_string(),
                couplings_hz: vec![],
                integral: hc as f64,
                atom_indices: vec![i],
                assignment_label: Some(format!("NH{}", if hc > 1 { "2" } else { "" })),
            });
        } else if a.label == "C" {
            let h_ppm = if has_carbonyl_o {
                9.75 // Aldehyde CHO
            } else if is_arom {
                7.26 + 0.45 * (if adj_to_carbonyl { 1.0 } else { 0.0 }) - 0.32 * (n_oxy as f64)
                    + 0.07 * ((i % 4) as f64)
            } else if adj[i].iter().any(|&(_, ord)| ord == BondOrder::Double) {
                5.55 + 0.65 * (n_oxy as f64)
            } else if adj[i].iter().any(|&(_, ord)| ord == BondOrder::Triple) {
                2.48
            } else {
                // Shoolery's rules for CH3 / CH2 / CH
                let mut base = match hc {
                    3 => 0.90,
                    2 => 1.28,
                    _ => 1.55,
                };
                base += 2.55 * (n_oxy as f64) + 1.65 * (n_nit as f64) + 2.15 * (n_halo as f64);
                if adj_to_carbonyl {
                    base += 1.22;
                }
                if adj_to_arom {
                    base += 1.42;
                }
                base
            };

            // Determine `³J_HH` vicinal proton count across adjacent carbons!
            let vicinal_h: usize = adj[i]
                .iter()
                .filter(|&&(j, _)| scene.atoms[j].label == "C")
                .map(|&(j, _)| h_counts[j] as usize)
                .sum();
            let j_val = if is_arom { 7.8 } else { 7.1 };
            let (mult, j_vec) = match vicinal_h {
                0 => ("s".to_string(), vec![]),
                1 => ("d".to_string(), vec![j_val]),
                2 => {
                    if is_arom {
                        ("dd".to_string(), vec![8.0, 7.5])
                    } else {
                        ("t".to_string(), vec![j_val])
                    }
                }
                3 => ("q".to_string(), vec![j_val]),
                4 => ("quint".to_string(), vec![j_val]),
                6 => ("sept".to_string(), vec![6.9]),
                _ => ("m".to_string(), vec![j_val, 4.2]),
            };

            signals_1h.push(SpectrumSignal1D {
                ppm: h_ppm.clamp(0.5, 11.4),
                multiplicity: mult,
                couplings_hz: j_vec,
                integral: hc as f64,
                atom_indices: vec![i],
                assignment_label: Some(format!(
                    "C{}H{}",
                    i + 1,
                    if hc > 1 {
                        hc.to_string()
                    } else {
                        String::new()
                    }
                )),
            });
        }
    }

    signals_1h.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));
    signals_13c.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));
    (signals_1h, signals_13c)
}

// ============================================================================
// Built-In Name/SMILES Parser + CDXML / CDX / MOL / SDF / SMILES / JCAMP-DX I/O
// ============================================================================

fn builtin_name_or_smiles_to_scene(
    query: &str,
    lib: &TemplateLibrary,
) -> Result<RenderScene, String> {
    let q = query.trim();
    let ql = q.to_ascii_lowercase();

    // 1. Direct template library hit (`benzene`, `indole`, `atp`, `ferrocene`, `adamantane`, etc.)
    if let Some(tpl) = lib.get(&ql) {
        return Ok(tpl.scene.clone());
    }

    // 2. Built-in common chemical names
    match ql.as_str() {
        "aspirin" | "acetylsalicylic acid" => {
            let mut full = build_default_showcase_scene(lib);
            // Keep only the aspirin molecule on left of arrow
            full.reaction_arrows.clear();
            full.curved_arrows.clear();
            let keep: HashSet<usize> = (0..full.atoms.len())
                .filter(|&i| full.atoms[i].x < 0.0)
                .collect();
            let gone: HashSet<usize> = (0..full.atoms.len())
                .filter(|i| !keep.contains(i))
                .collect();
            let mut tmp = OctetDrawPanel {
                scene: full,
                ..OctetDrawPanel::new()
            };
            tmp.remove_atoms_set(&gone);
            let [cx, cy] = tmp.scene.atom_centroid();
            tmp.scene.translate(-cx, -cy);
            return Ok(tmp.scene);
        }
        "ethanol" | "etoh" => return parse_simple_smiles("CCO"),
        "acetic acid" | "acoh" => return parse_simple_smiles("CC(=O)O"),
        "ethyl acetate" | "etoac" => return parse_simple_smiles("CCOC(=O)C"),
        "acetone" => return parse_simple_smiles("CC(=O)C"),
        "phenol" => {
            let mut bz = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
            let oh = bz.add_atom(SceneAtom::new("O", bz.atoms[0].x, bz.atoms[0].y + 1.0));
            bz.add_bond(SceneBond::single(0, oh));
            return Ok(bz);
        }
        "aniline" => {
            let mut bz = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
            let nh2 = bz.add_atom(SceneAtom::new("N", bz.atoms[0].x, bz.atoms[0].y + 1.0));
            bz.add_bond(SceneBond::single(0, nh2));
            return Ok(bz);
        }
        "toluene" => {
            let mut bz = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
            let me = bz.add_atom(SceneAtom::carbon(bz.atoms[0].x, bz.atoms[0].y + 1.0));
            bz.add_bond(SceneBond::single(0, me));
            return Ok(bz);
        }
        _ => {}
    }

    // 3. Peptide sequence shorthand (`Ala-Pro-Tyr`)
    if q.contains('-') {
        let codes = parse_peptide_sequence(q);
        if codes.len() >= 2 {
            return Ok(build_peptide_scene(&codes));
        }
    }

    // 4. SMILES parser (branches `(...)`, rings `1..9`, single/double/triple/aromatic bonds, common elements!)
    parse_simple_smiles(q)
}

fn parse_simple_smiles(smi: &str) -> Result<RenderScene, String> {
    let s = smi
        .trim()
        .strip_prefix("SMILES:")
        .unwrap_or(smi.trim())
        .trim();
    if s.is_empty() {
        return Err("Empty SMILES string".to_string());
    }
    let chars: Vec<char> = s.chars().collect();
    let mut scene = RenderScene::new();
    let mut i = 0;
    let mut cur_atom: Option<usize> = None;
    let mut branch_stack: Vec<usize> = Vec::new();
    let mut ring_open: BTreeMap<char, (usize, BondOrder)> = BTreeMap::new();
    let mut next_ord = BondOrder::Single;
    let mut next_stereo = BondStereo::None;

    while i < chars.len() {
        let c = chars[i];
        match c {
            '(' => {
                if let Some(ca) = cur_atom {
                    branch_stack.push(ca);
                }
                i += 1;
            }
            ')' => {
                cur_atom = branch_stack.pop();
                i += 1;
            }
            '=' => {
                next_ord = BondOrder::Double;
                i += 1;
            }
            '#' => {
                next_ord = BondOrder::Triple;
                i += 1;
            }
            '-' => {
                next_ord = BondOrder::Single;
                i += 1;
            }
            ':' => {
                next_ord = BondOrder::Aromatic;
                i += 1;
            }
            '@' => {
                next_stereo = if i + 1 < chars.len() && chars[i + 1] == '@' {
                    i += 1;
                    BondStereo::UpWedge
                } else {
                    BondStereo::DownHash
                };
                i += 1;
            }
            '/' | '\\' | '.' | ' ' => {
                if c == '.' {
                    cur_atom = None;
                }
                i += 1;
            }
            '1'..='9' => {
                if let Some(ca) = cur_atom {
                    if let Some((open_a, r_ord)) = ring_open.remove(&c) {
                        let ord = if next_ord != BondOrder::Single {
                            next_ord
                        } else {
                            r_ord
                        };
                        scene.add_bond(SceneBond::new(open_a, ca, ord));
                    } else {
                        ring_open.insert(c, (ca, next_ord));
                    }
                    next_ord = BondOrder::Single;
                }
                i += 1;
            }
            '[' => {
                let close = chars[i + 1..]
                    .iter()
                    .position(|&x| x == ']')
                    .ok_or_else(|| "Unclosed `[` in SMILES".to_string())?;
                let inner: String = chars[i + 1..i + 1 + close].iter().collect();
                let mut el = String::new();
                for ch in inner.chars() {
                    if ch.is_ascii_alphabetic() && el.len() < 2 {
                        if el.is_empty() {
                            el.push(ch.to_ascii_uppercase());
                        } else if ch.is_ascii_lowercase() && ch != 'h' {
                            el.push(ch);
                        } else {
                            break;
                        }
                    } else if !el.is_empty() {
                        break;
                    }
                }
                if el.is_empty() {
                    el = "C".to_string();
                }
                let charge = if inner.contains("+") {
                    1
                } else if inner.contains("-") {
                    -1
                } else {
                    0
                };
                let idx = place_next_smiles_atom(&mut scene, cur_atom, &el, charge);
                if let Some(prev) = cur_atom {
                    let mut b = SceneBond::new(prev, idx, next_ord);
                    b.stereo = next_stereo;
                    scene.add_bond(b);
                }
                cur_atom = Some(idx);
                next_ord = BondOrder::Single;
                next_stereo = BondStereo::None;
                i += close + 2;
            }
            _ if c.is_ascii_alphabetic() => {
                let is_arom_atom = c.is_ascii_lowercase();
                let (sym, step) = if i + 1 < chars.len()
                    && ((c == 'C' && chars[i + 1] == 'l') || (c == 'B' && chars[i + 1] == 'r'))
                {
                    (format!("{}{}", c, chars[i + 1]), 2)
                } else {
                    (c.to_ascii_uppercase().to_string(), 1)
                };
                let idx = place_next_smiles_atom(&mut scene, cur_atom, &sym, 0);
                if let Some(prev) = cur_atom {
                    let ord = if next_ord != BondOrder::Single {
                        next_ord
                    } else if is_arom_atom {
                        BondOrder::Aromatic
                    } else {
                        BondOrder::Single
                    };
                    let mut b = SceneBond::new(prev, idx, ord);
                    b.stereo = next_stereo;
                    scene.add_bond(b);
                }
                cur_atom = Some(idx);
                next_ord = BondOrder::Single;
                next_stereo = BondStereo::None;
                i += step;
            }
            _ => {
                i += 1;
            }
        }
    }

    if scene.atoms.is_empty() {
        return Err(format!("Could not parse `{smi}` as SMILES or known name"));
    }

    // Run Clean 2D pass so ring closures & branches lay out nicely!
    let mut panel = OctetDrawPanel {
        scene,
        ..OctetDrawPanel::new()
    };
    panel.clean_2d_structure();
    let [cx, cy] = panel.scene.atom_centroid();
    panel.scene.translate(-cx, -cy);
    Ok(panel.scene)
}

fn place_next_smiles_atom(
    scene: &mut RenderScene,
    prev: Option<usize>,
    sym: &str,
    charge: i8,
) -> usize {
    let pos = match prev {
        None => [0.0, 0.0],
        Some(p) => {
            let deg = scene.bonds.iter().filter(|b| b.a == p || b.b == p).count();
            let ang = match deg {
                0 => 30.0_f64.to_radians(),
                1 => {
                    if p % 2 == 0 {
                        -30.0_f64.to_radians()
                    } else {
                        90.0_f64.to_radians()
                    }
                }
                2 => -90.0_f64.to_radians(),
                _ => (deg as f64 * 75.0).to_radians(),
            };
            [scene.atoms[p].x + ang.cos(), scene.atoms[p].y + ang.sin()]
        }
    };
    let mut a = SceneAtom::new(sym, pos[0], pos[1]);
    a.charge = charge;
    scene.add_atom(a)
}

fn builtin_scene_to_smiles(scene: &RenderScene) -> String {
    let n = scene.atoms.len();
    if n == 0 {
        return String::new();
    }
    let mut adj: Vec<Vec<(usize, BondOrder)>> = vec![Vec::new(); n];
    for b in &scene.bonds {
        if b.a < n && b.b < n && b.a != b.b && b.order != BondOrder::HydrogenBond {
            adj[b.a].push((b.b, b.order));
            adj[b.b].push((b.a, b.order));
        }
    }
    let mut visited = vec![false; n];
    let mut ring_nums: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut next_ring = 1_usize;

    // Pre-assign ring closure numbers via DFS tree
    fn dfs_rings(
        u: usize,
        parent: Option<usize>,
        adj: &[Vec<(usize, BondOrder)>],
        visited: &mut [bool],
        ring_nums: &mut BTreeMap<(usize, usize), usize>,
        next_ring: &mut usize,
    ) {
        visited[u] = true;
        for &(v, _) in &adj[u] {
            if Some(v) == parent {
                continue;
            }
            if visited[v] {
                let key = (u.min(v), u.max(v));
                if !ring_nums.contains_key(&key) && *next_ring <= 9 {
                    ring_nums.insert(key, *next_ring);
                    *next_ring += 1;
                }
            } else {
                dfs_rings(v, Some(u), adj, visited, ring_nums, next_ring);
            }
        }
    }

    let mut frags = Vec::new();
    for start in 0..n {
        if visited[start] || scene.atoms[start].is_multi_center() {
            continue;
        }
        dfs_rings(
            start,
            None,
            &adj,
            &mut visited,
            &mut ring_nums,
            &mut next_ring,
        );
    }

    let mut emitted = vec![false; n];
    #[allow(clippy::too_many_arguments)]
    fn emit_dfs(
        u: usize,
        parent: Option<usize>,
        in_ord: Option<BondOrder>,
        scene: &RenderScene,
        adj: &[Vec<(usize, BondOrder)>],
        ring_nums: &BTreeMap<(usize, usize), usize>,
        emitted: &mut [bool],
        out: &mut String,
    ) {
        emitted[u] = true;
        if let Some(ord) = in_ord {
            match ord {
                BondOrder::Double => out.push('='),
                BondOrder::Triple => out.push('#'),
                BondOrder::Aromatic => out.push(':'),
                _ => {}
            }
        }
        let a = &scene.atoms[u];
        let el = &a.label;
        if a.charge != 0
            || !matches!(
                el.as_str(),
                "B" | "C" | "N" | "O" | "P" | "S" | "F" | "Cl" | "Br" | "I"
            )
        {
            let chg = if a.charge > 0 {
                "+".to_string()
            } else if a.charge < 0 {
                "-".to_string()
            } else {
                String::new()
            };
            out.push_str(&format!("[{el}{chg}]"));
        } else {
            out.push_str(el);
        }

        // Ring digits touching `u`
        for (&(ra, rb), &rnum) in ring_nums {
            if ra == u || rb == u {
                out.push_str(&rnum.to_string());
            }
        }

        let children: Vec<(usize, BondOrder)> = adj[u]
            .iter()
            .copied()
            .filter(|&(v, _)| {
                Some(v) != parent && !emitted[v] && !ring_nums.contains_key(&(u.min(v), u.max(v)))
            })
            .collect();
        for (ci, (v, ord)) in children.iter().enumerate() {
            let is_branch = ci + 1 < children.len();
            if is_branch {
                out.push('(');
            }
            emit_dfs(*v, Some(u), Some(*ord), scene, adj, ring_nums, emitted, out);
            if is_branch {
                out.push(')');
            }
        }
    }

    for start in 0..n {
        if emitted[start] || scene.atoms[start].is_multi_center() {
            continue;
        }
        let mut s = String::new();
        emit_dfs(
            start,
            None,
            None,
            scene,
            &adj,
            &ring_nums,
            &mut emitted,
            &mut s,
        );
        if !s.is_empty() {
            frags.push(s);
        }
    }
    frags.join(".")
}

fn export_builtin_mol_v2000(scene: &RenderScene, as_sdf: bool) -> String {
    let real_atoms: Vec<usize> = (0..scene.atoms.len())
        .filter(|&i| !scene.atoms[i].is_multi_center())
        .collect();
    let mut map_idx = vec![0_usize; scene.atoms.len()];
    for (new_1based, &old) in real_atoms.iter().enumerate() {
        map_idx[old] = new_1based + 1;
    }
    let valid_bonds: Vec<&SceneBond> = scene
        .bonds
        .iter()
        .filter(|b| {
            b.order != BondOrder::HydrogenBond
                && b.a < scene.atoms.len()
                && b.b < scene.atoms.len()
                && map_idx[b.a] > 0
                && map_idx[b.b] > 0
        })
        .collect();

    let mut out = String::new();
    out.push_str("Octet-Draw / nmr-assign\n  nmr-gui 0200262D\n\n");
    out.push_str(&format!(
        "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        real_atoms.len(),
        valid_bonds.len()
    ));
    for &i in &real_atoms {
        let a = &scene.atoms[i];
        out.push_str(&format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            a.x * 1.54,
            a.y * 1.54,
            a.z * 1.54,
            a.label
        ));
    }
    for b in &valid_bonds {
        let btype = match b.order {
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => 4,
            _ => 1,
        };
        let bstereo = match b.stereo {
            BondStereo::UpWedge => 1,
            BondStereo::Wavy => 4,
            BondStereo::DownHash => 6,
            _ => 0,
        };
        out.push_str(&format!(
            "{:>3}{:>3}{:>3}{:>3}  0  0  0\n",
            map_idx[b.a], map_idx[b.b], btype, bstereo
        ));
    }
    out.push_str("M  END\n");
    if as_sdf {
        out.push_str("$$$$\n");
    }
    out
}

fn import_builtin_mol_v2000(mol_text: &str) -> Result<RenderScene, String> {
    let lines: Vec<&str> = mol_text.lines().collect();
    let counts_idx = lines
        .iter()
        .position(|l| l.contains("V2000"))
        .ok_or_else(|| "Missing V2000 counts line in MOL/SDF".to_string())?;
    let cl = lines[counts_idx];
    if cl.len() < 6 {
        return Err("Malformed V2000 counts line".to_string());
    }
    let natoms: usize = cl[0..3].trim().parse().map_err(|_| "Invalid atom count")?;
    let nbonds: usize = cl[3..6].trim().parse().map_err(|_| "Invalid bond count")?;

    let mut scene = RenderScene::new();
    for k in 0..natoms {
        let line = lines
            .get(counts_idx + 1 + k)
            .ok_or_else(|| "Truncated MOL atom block".to_string())?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("Invalid MOL atom line".to_string());
        }
        let x: f64 = parts[0].parse().unwrap_or(0.0) / 1.54;
        let y: f64 = parts[1].parse().unwrap_or(0.0) / 1.54;
        let z: f64 = parts[2].parse().unwrap_or(0.0) / 1.54;
        let mut atom = SceneAtom::new(parts[3], x, y);
        atom.z = z;
        scene.add_atom(atom);
    }
    for k in 0..nbonds {
        let line = lines
            .get(counts_idx + 1 + natoms + k)
            .ok_or_else(|| "Truncated MOL bond block".to_string())?;
        if line.len() < 9 {
            continue;
        }
        let a1: usize = line[0..3].trim().parse().unwrap_or(1);
        let b1: usize = line[3..6].trim().parse().unwrap_or(1);
        let btype: usize = line[6..9].trim().parse().unwrap_or(1);
        let bstereo: usize = line
            .get(9..12)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        if a1 >= 1 && b1 >= 1 && a1 <= natoms && b1 <= natoms {
            let ord = match btype {
                2 => BondOrder::Double,
                3 => BondOrder::Triple,
                4 => BondOrder::Aromatic,
                _ => BondOrder::Single,
            };
            let mut sb = SceneBond::new(a1 - 1, b1 - 1, ord);
            sb.stereo = match bstereo {
                1 => BondStereo::UpWedge,
                4 => BondStereo::Wavy,
                6 => BondStereo::DownHash,
                _ => BondStereo::None,
            };
            scene.add_bond(sb);
        }
    }
    Ok(scene)
}

fn export_builtin_cdxml(scene: &RenderScene, style: &DocumentStyle) -> String {
    let bl = style.bond_length_pt;
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n\
         <!DOCTYPE CDXML SYSTEM \"http://www.cambridgesoft.com/xml/cdxml.dtd\" >\n",
    );
    xml.push_str(&format!(
        "<CDXML BondLength=\"{:.2}\" LineWidth=\"{:.2}\" BoldWidth=\"{:.2}\">\n  <page id=\"1\">\n    <fragment id=\"2\">\n",
        bl, style.line_width_pt, style.bold_width_pt
    ));
    for (i, a) in scene.atoms.iter().enumerate() {
        if a.is_multi_center() {
            continue;
        }
        let px = 200.0 + a.x * bl;
        let py = 200.0 - a.y * bl;
        xml.push_str(&format!(
            "      <n id=\"{}\" p=\"{px:.2} {py:.2}\" ElementSymbol=\"{}\"/>\n",
            10 + i,
            a.label
        ));
    }
    for (bi, b) in scene.bonds.iter().enumerate() {
        let ord = match b.order {
            BondOrder::Single => "1",
            BondOrder::Double => "2",
            BondOrder::Triple => "3",
            BondOrder::Aromatic => "1.5",
            _ => "1",
        };
        let disp = match b.stereo {
            BondStereo::UpWedge => " Display=\"WedgeBegin\"",
            BondStereo::DownHash => " Display=\"WedgedHashBegin\"",
            BondStereo::Wavy => " Display=\"Wavy\"",
            _ => "",
        };
        xml.push_str(&format!(
            "      <b id=\"{}\" B=\"{}\" E=\"{}\" Order=\"{ord}\"{disp}/>\n",
            1000 + bi,
            10 + b.a,
            10 + b.b
        ));
    }
    xml.push_str("    </fragment>\n  </page>\n</CDXML>\n");
    xml
}

fn import_builtin_cdxml(xml: &str) -> Result<RenderScene, String> {
    if !xml.contains("<CDXML") {
        return Err("Invalid CDXML document (missing <CDXML> root)".to_string());
    }
    let mut scene = RenderScene::new();
    let mut id_map: BTreeMap<String, usize> = BTreeMap::new();

    for raw_tag in xml.split('<') {
        let tag = raw_tag.trim();
        if tag.starts_with("n ") {
            let id = extract_xml_attr(tag, "id").unwrap_or_default();
            let p_str = extract_xml_attr(tag, "p").unwrap_or_default();
            let mut coords = p_str.split_whitespace();
            let px: f64 = coords.next().and_then(|s| s.parse().ok()).unwrap_or(200.0);
            let py: f64 = coords.next().and_then(|s| s.parse().ok()).unwrap_or(200.0);
            let sym = extract_xml_attr(tag, "ElementSymbol")
                .or_else(|| match extract_xml_attr(tag, "Element").as_deref() {
                    Some("7") => Some("N".to_string()),
                    Some("8") => Some("O".to_string()),
                    Some("9") => Some("F".to_string()),
                    Some("15") => Some("P".to_string()),
                    Some("16") => Some("S".to_string()),
                    Some("17") => Some("Cl".to_string()),
                    Some("35") => Some("Br".to_string()),
                    Some("53") => Some("I".to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| "C".to_string());
            let idx = scene.add_atom(SceneAtom::new(
                sym,
                (px - 200.0) / 14.4,
                (200.0 - py) / 14.4,
            ));
            if !id.is_empty() {
                id_map.insert(id, idx);
            }
        } else if tag.starts_with("b ") {
            let b_id = extract_xml_attr(tag, "B").unwrap_or_default();
            let e_id = extract_xml_attr(tag, "E").unwrap_or_default();
            if let (Some(&u), Some(&v)) = (id_map.get(&b_id), id_map.get(&e_id)) {
                let ord = match extract_xml_attr(tag, "Order").as_deref() {
                    Some("2") => BondOrder::Double,
                    Some("3") => BondOrder::Triple,
                    Some("1.5") => BondOrder::Aromatic,
                    _ => BondOrder::Single,
                };
                let mut sb = SceneBond::new(u, v, ord);
                if let Some(disp) = extract_xml_attr(tag, "Display") {
                    if disp.contains("Hash") {
                        sb.stereo = BondStereo::DownHash;
                    } else if disp.contains("Wedge") {
                        sb.stereo = BondStereo::UpWedge;
                    } else if disp.contains("Wavy") {
                        sb.stereo = BondStereo::Wavy;
                    }
                }
                scene.add_bond(sb);
            }
        }
    }
    let [cx, cy] = scene.atom_centroid();
    scene.translate(-cx, -cy);
    Ok(scene)
}

fn extract_xml_attr(tag: &str, attr_name: &str) -> Option<String> {
    let pat = format!("{attr_name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn export_builtin_cdx_hex(scene: &RenderScene) -> String {
    // Standard 22-byte ChemDraw CDX binary header (`VjCD0100` + magic bytes) + embedded OCT payload tag
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"VjCD0100\x04\x03\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    if let Ok(json) = serde_json::to_vec(scene) {
        bytes.extend_from_slice(&[0x00, 0x90]); // Custom scene tag
        let len = (json.len().min(65535)) as u16;
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&json[..len as usize]);
    }
    bytes.extend_from_slice(&[0x00, 0x00]); // End of CDX object
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn import_builtin_cdx_hex(hex_str: &str) -> Result<RenderScene, String> {
    let mut bytes = Vec::new();
    let clean: String = hex_str.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    for pair in clean.as_bytes().chunks_exact(2) {
        let s = std::str::from_utf8(pair).unwrap_or("00");
        bytes.push(u8::from_str_radix(s, 16).map_err(|_| "Invalid hex byte")?);
    }
    if bytes.len() < 24 || &bytes[0..8] != b"VjCD0100" {
        return Err("Invalid CDX binary header (expected VjCD0100)".to_string());
    }
    if bytes.len() >= 26 && bytes[22] == 0x00 && bytes[23] == 0x90 {
        let len = u16::from_le_bytes([bytes[24], bytes[25]]) as usize;
        if bytes.len() >= 26 + len {
            return serde_json::from_slice::<RenderScene>(&bytes[26..26 + len])
                .map_err(|e| format!("CDX scene decode error: {e}"));
        }
    }
    Ok(RenderScene::new())
}

fn export_builtin_jcamp_dx(
    signals_1h: &[SpectrumSignal1D],
    freq_mhz: f64,
    formula: &str,
) -> String {
    let mut jdx = String::new();
    jdx.push_str("##TITLE=Octet-Draw / nmr-assign Predicted 1D 1H NMR\n");
    jdx.push_str("##JCAMP-DX=5.01\n");
    jdx.push_str("##DATA TYPE=NMR SPECTRUM\n");
    jdx.push_str(&format!("##MOLFORM={formula}\n"));
    jdx.push_str("##.OBSERVE NUCLEUS=^1H\n");
    jdx.push_str(&format!("##.OBSERVE FREQUENCY={freq_mhz:.4}\n"));
    jdx.push_str("##XUNITS=PPM\n##YUNITS=ARBITRARY UNITS\n");
    jdx.push_str(&format!("##PEAK TABLE=(XY..XY), {}\n", signals_1h.len()));
    for sig in signals_1h {
        jdx.push_str(&format!("{:.4}, {:.2}\n", sig.ppm, sig.integral * 100.0));
    }
    jdx.push_str("##END=\n");
    jdx
}

fn import_builtin_jcamp_dx(jdx: &str, current_scene: &RenderScene) -> Result<RenderScene, String> {
    if !jdx.contains("##JCAMP-DX") && !jdx.contains("##TITLE") {
        return Err("Not a valid JCAMP-DX file (missing ##TITLE / ##JCAMP-DX)".to_string());
    }
    let mut parsed_signals = Vec::new();
    let mut in_peaks = false;
    for line in jdx.lines() {
        let t = line.trim();
        if t.starts_with("##PEAK TABLE") || t.starts_with("##XYDATA") {
            in_peaks = true;
            continue;
        }
        if t.starts_with("##") {
            in_peaks = false;
            continue;
        }
        if in_peaks && !t.is_empty() {
            let parts: Vec<&str> = t
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                if let (Ok(ppm), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                    parsed_signals.push(SpectrumSignal1D {
                        ppm,
                        multiplicity: "s".to_string(),
                        couplings_hz: vec![],
                        integral: (y / 100.0).clamp(0.5, 9.0),
                        atom_indices: vec![],
                        assignment_label: None,
                    });
                }
            }
        }
    }
    let mut next_scene = current_scene.clone();
    next_scene.spectra_1d.push(SceneSpectrum1D {
        origin: [-5.0, -5.0],
        width_mol: 10.0,
        height_mol: 3.4,
        nucleus: "1H".to_string(),
        spectrometer_mhz: 500.0,
        ppm_min: -0.5,
        ppm_max: 11.0,
        linewidth_hz: 1.5,
        show_integrals: true,
        show_peak_labels: true,
        title: Some("Imported JCAMP-DX 1D NMR".to_string()),
        signals: parsed_signals,
        custom_trace: Vec::new(),
    });
    Ok(next_scene)
}
