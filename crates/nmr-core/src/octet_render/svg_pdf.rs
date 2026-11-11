//! Publication 2D chemical scene compiler and SVG, PDF 1.4, and PNG exporters.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis (`render.ts`, `export.ts`, `pageplot.ts`).
//!
//! Converts a 2D chemical drawing scene (`RenderScene`: atoms with direction-aware `NH`/`HO` vs
//! `OH` implicit H placement, single/double/aromatic/triple/wedge/hash/wavy/dative bonds with
//! aesthetic ring-inside double bonds and heteroatom label clearance clipping, reaction arrows,
//! Bezier curved 1e⁻ fishhook & 2e⁻ full electron-pushing arrows, polymer brackets, `s`/`p`/`d`
//! orbitals, TLC plates, and embedded 1D NMR spectrum plots) into:
//! - Crisp publication **SVG 1.1** (`render_scene_svg`),
//! - Standalone vector **PDF 1.4** (`render_scene_pdf`, pure Rust, zero external crates), and
//! - Anti-aliased lossless **PNG** (`render_scene_png`, pure Rust zlib/DEFLATE + CRC-32).

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::styles::{
    char_advance_em, layout_atom_label, text_advance_pt, DocumentStyle, FormattedAtomLabel,
    LabelDirection, RgbaColor, ScriptKind,
};

// ============================================================================
// 1. Scene Data Model (coordinates in bond lengths, +x right, +y up)
// ============================================================================

/// Chemical bond order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BondOrder {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
    Quadruple,
    HydrogenBond,
}

impl BondOrder {
    /// Valence electrons contributed to implicit hydrogen calculation.
    pub fn valence_order(self) -> f64 {
        match self {
            Self::Single => 1.0,
            Self::Double => 2.0,
            Self::Triple => 3.0,
            Self::Aromatic => 1.5,
            Self::Quadruple => 4.0,
            Self::HydrogenBond => 0.0,
        }
    }
}

/// Stereochemical or specialized graphical bond style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BondStereo {
    #[default]
    None,
    /// Solid stereo wedge widening from `a` (narrow tip) to `b` (wide end).
    UpWedge,
    /// Hashed stereo wedge widening from `a` (narrow tip) to `b` (wide end).
    DownHash,
    /// Hollow (open) wedge widening from `a` to `b`.
    UpHollow,
    /// Sinusoidal wavy bond (unknown / racemic stereochemistry).
    Wavy,
    /// Coordinate / dative arrow pointing from donor `a` to acceptor `b`.
    Dative,
    /// Thick bold projection bond (e.g. Haworth / chair front edge).
    Bold,
    /// Dashed single/coordination bond.
    Dashed,
    /// Crossed double bond (`E`/`Z` unspecified).
    CrossedDouble,
}

/// Double-bond secondary line placement preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DoubleBondSide {
    #[default]
    Auto,
    Left,
    Center,
    Right,
}

/// Atom node in a 2D chemical drawing scene (`x, y` in bond-length units, `+y` up).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneAtom {
    /// Element symbol (`"C"`, `"N"`, `"O"`, `"S"`, `"P"`, `"Pd"`, etc.).
    pub label: String,
    /// Optional condensed group formula (`"COOH"`, `"Boc"`, `"Ph"`, `"NO2"`).
    pub text: Option<String>,
    /// X coordinate in bond lengths (`+x` right).
    pub x: f64,
    /// Y coordinate in bond lengths (`+y` up).
    pub y: f64,
    /// Optional 3D depth in bond lengths (`+z` toward viewer, used for conformers/perspective).
    pub z: f64,
    /// Formal integer charge (`+1`, `-1`, etc.).
    pub charge: i8,
    /// Unpaired radical electrons (`0`, `1` monoradical, `2` diradical).
    pub radical: u8,
    /// Optional mass number (`13` for `¹³C`, `2` for `²H`, `15` for `¹⁵N`).
    pub isotope: Option<u16>,
    /// Explicit hydrogen count override (`None` = standard IUPAC implicit valence).
    pub explicit_h: Option<u8>,
    /// Force carbon symbol display (`CH3` / `CH2` / `C`).
    pub show_carbon: bool,
    /// Optional stereodescriptor annotation (`"(R)"`, `"(S)"`, `"(M)"`, `"(P)"`).
    pub cip_label: Option<String>,
    /// Number of lone-pair dot pairs to draw around heteroatom.
    pub lone_pairs: u8,
    /// Optional per-atom ink override.
    pub color: Option<RgbaColor>,
    /// Optional soft circular halo highlight behind atom.
    pub highlight: Option<RgbaColor>,
    /// Multi-center attachment ring atom indices (e.g. η⁵-Cp centroid in ferrocene);
    /// when non-empty, this node is an invisible η-attachment centroid!
    pub multi_center_atoms: Vec<usize>,
}

impl SceneAtom {
    pub fn new(label: impl Into<String>, x: f64, y: f64) -> Self {
        Self {
            label: label.into(),
            text: None,
            x,
            y,
            z: 0.0,
            charge: 0,
            radical: 0,
            isotope: None,
            explicit_h: None,
            show_carbon: false,
            cip_label: None,
            lone_pairs: 0,
            color: None,
            highlight: None,
            multi_center_atoms: Vec::new(),
        }
    }

    pub fn carbon(x: f64, y: f64) -> Self {
        Self::new("C", x, y)
    }

    pub fn with_group(
        label: impl Into<String>,
        group_text: impl Into<String>,
        x: f64,
        y: f64,
    ) -> Self {
        let mut a = Self::new(label, x, y);
        a.text = Some(group_text.into());
        a
    }

    pub fn is_multi_center(&self) -> bool {
        !self.multi_center_atoms.is_empty()
    }
}

/// Chemical bond between two [`SceneAtom`] indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneBond {
    pub a: usize,
    pub b: usize,
    pub order: BondOrder,
    pub stereo: BondStereo,
    pub side: DoubleBondSide,
    pub ez_label: Option<String>,
    pub color: Option<RgbaColor>,
    pub highlight: Option<RgbaColor>,
}

impl SceneBond {
    pub fn new(a: usize, b: usize, order: BondOrder) -> Self {
        Self {
            a,
            b,
            order,
            stereo: BondStereo::None,
            side: DoubleBondSide::Auto,
            ez_label: None,
            color: None,
            highlight: None,
        }
    }

    pub fn single(a: usize, b: usize) -> Self {
        Self::new(a, b, BondOrder::Single)
    }

    pub fn double(a: usize, b: usize) -> Self {
        Self::new(a, b, BondOrder::Double)
    }

    pub fn triple(a: usize, b: usize) -> Self {
        Self::new(a, b, BondOrder::Triple)
    }

    pub fn aromatic(a: usize, b: usize) -> Self {
        Self::new(a, b, BondOrder::Aromatic)
    }

    pub fn wedge(a: usize, b: usize) -> Self {
        let mut s = Self::single(a, b);
        s.stereo = BondStereo::UpWedge;
        s
    }

    pub fn hash(a: usize, b: usize) -> Self {
        let mut s = Self::single(a, b);
        s.stereo = BondStereo::DownHash;
        s
    }

    pub fn wavy(a: usize, b: usize) -> Self {
        let mut s = Self::single(a, b);
        s.stereo = BondStereo::Wavy;
        s
    }

    pub fn dative(a: usize, b: usize) -> Self {
        let mut s = Self::single(a, b);
        s.stereo = BondStereo::Dative;
        s
    }
}

/// Reaction arrow style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArrowKind {
    /// Standard synthetic reaction arrow (`───▶`).
    Solid,
    /// Reversible equilibrium harpoons (`⇌`).
    Equilibrium,
    /// Unbalanced equilibrium harpoons (long forward, short reverse).
    UnbalancedEquilibrium,
    /// Double-headed resonance arrow (`⟷`).
    Resonance,
    /// Open double-line retrosynthetic arrow (`⟹`).
    Retrosynthetic,
    /// Crossed no-reaction arrow (`─✕─▶`).
    NoGo,
    /// Dashed theoretical / multi-step arrow (`- - ▶`).
    Dashed,
}

/// Straight reaction arrow with optional reagent/condition text above and below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneReactionArrow {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub kind: ArrowKind,
    pub above_text: Option<String>,
    pub below_text: Option<String>,
    pub color: Option<RgbaColor>,
}

/// Curved electron-pushing arrow head type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PushingArrowKind {
    /// 2-electron full double-barbed curved arrow.
    FullTwoElectron,
    /// 1-electron single-barbed radical fishhook arrow.
    FishhookOneElectron,
}

/// Cubic Bezier curved electron-pushing arrow (`start -> ctrl1 -> ctrl2 -> end`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneCurvedArrow {
    pub start: [f64; 2],
    pub ctrl1: [f64; 2],
    pub ctrl2: [f64; 2],
    pub end: [f64; 2],
    pub kind: PushingArrowKind,
    /// Which side of the tangent receives the single barb when `FishhookOneElectron`.
    pub fishhook_left: bool,
    pub dashed: bool,
    pub color: Option<RgbaColor>,
}

impl SceneCurvedArrow {
    /// Create a smooth curved electron-pushing arrow from `start` to `end` bowing by `bow_mol`
    /// bond lengths perpendicular to the chord (`bow_mol > 0` bows left of `start -> end`).
    pub fn from_chord(
        start: [f64; 2],
        end: [f64; 2],
        bow_mol: f64,
        kind: PushingArrowKind,
    ) -> Self {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let chord = (dx * dx + dy * dy).sqrt().max(1e-6);
        let nx = -dy / chord;
        let ny = dx / chord;
        let h = bow_mol * (4.0 / 3.0);
        let ctrl1 = [start[0] + 0.30 * dx + h * nx, start[1] + 0.30 * dy + h * ny];
        let ctrl2 = [start[0] + 0.70 * dx + h * nx, start[1] + 0.70 * dy + h * ny];
        Self {
            start,
            ctrl1,
            ctrl2,
            end,
            kind,
            fishhook_left: bow_mol >= 0.0,
            dashed: false,
            color: None,
        }
    }
}

/// Polymer / repeat-unit / transition-state bracket shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BracketStyle {
    Square,
    Parenthesis,
    CurlyBrace,
}

/// Enclosure bracket pair (or single bracket) with optional repeat subscript (`n`) or charge/TS (`‡`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneBracket {
    /// Minimum corner `[min_x, min_y]` in molecular bond-length coordinates.
    pub min: [f64; 2],
    /// Maximum corner `[max_x, max_y]` in molecular bond-length coordinates.
    pub max: [f64; 2],
    pub style: BracketStyle,
    pub subscript: Option<String>,
    pub superscript: Option<String>,
    pub color: Option<RgbaColor>,
}

/// Atomic / molecular orbital geometry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrbitalType {
    /// Spherical `s` orbital.
    S,
    /// Two-lobed figure-eight `p` orbital.
    P,
    /// Four-lobed cloverleaf `d` (`d_xy` / `d_x2-y2`) orbital.
    DClover,
    /// Two-lobed `d_z2` orbital with equatorial torus collar.
    Dz2,
    /// Asymmetric `sp³` / `sp²` hybrid lobe (large major lobe + small minor counter-lobe).
    HybridSp3,
}

/// Orbital phase shading style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrbitalShading {
    /// Shaded positive lobe + open white negative lobe.
    ShadedAndOpen,
    /// Blue/Red or dual-color phase fill + `+`/`−` signs.
    DualPhaseColored,
    /// Unfilled outline lobes.
    OutlineOnly,
}

/// Atomic or hybrid orbital rendered in the chemical scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneOrbital {
    pub center: [f64; 2],
    /// Lobe length (or `s` radius) in molecular bond lengths.
    pub size_mol: f64,
    /// Orientation angle in degrees (`0°` = `+x`, `90°` = `+y` upright).
    pub angle_deg: f64,
    pub orbital_type: OrbitalType,
    pub shading: OrbitalShading,
    pub color: Option<RgbaColor>,
}

/// TLC spot visual appearance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TlcSpotStyle {
    Solid,
    Hollow,
    UvActive,
    Faint,
}

/// Single spot on a [`SceneTlcPlate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TlcSpot {
    pub lane: usize,
    /// Retention factor `Rf` in `[0.0, 1.0]`.
    pub rf: f64,
    /// Spot radius in bond-length units (default `~0.18`).
    pub radius_mol: f64,
    /// Vertical streaking/tailing factor (`1.0` = circular, `> 1.0` = elongated).
    pub tailing: f64,
    pub style: TlcSpotStyle,
    pub label: Option<String>,
    pub color: Option<RgbaColor>,
}

/// Thin-layer chromatography (TLC) plate diagram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneTlcPlate {
    /// Bottom-left corner `[x, y]` in molecular bond lengths.
    pub origin: [f64; 2],
    pub width_mol: f64,
    pub height_mol: f64,
    pub lane_labels: Vec<String>,
    /// Baseline height fraction from bottom (`0.12` typical).
    pub baseline_frac: f64,
    /// Solvent-front height fraction from bottom (`0.88` typical).
    pub solvent_front_frac: f64,
    pub show_rf_scale: bool,
    pub spots: Vec<TlcSpot>,
}

/// Predicted or experimental 1D NMR multiplet signal for embedded spectrum plots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectrumSignal1D {
    pub ppm: f64,
    /// Multiplicity code (`"s"`, `"d"`, `"t"`, `"q"`, `"dd"`, `"dt"`, `"td"`, `"sept"`, `"m"`, `"br s"`).
    pub multiplicity: String,
    /// Scalar coupling constants `J` in Hz (`e.g. vec![7.1]` for a quartet/triplet).
    pub couplings_hz: Vec<f64>,
    /// Relative proton / carbon integral (`e.g. 3.0` for `CH3`).
    pub integral: f64,
    /// Indices of atoms in `RenderScene::atoms` producing this resonance.
    pub atom_indices: Vec<usize>,
    pub assignment_label: Option<String>,
}

/// Embedded 1D ¹H or ¹³C NMR spectrum plot on the chemical drawing canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneSpectrum1D {
    /// Bottom-left corner `[x, y]` in molecular bond lengths.
    pub origin: [f64; 2],
    pub width_mol: f64,
    pub height_mol: f64,
    /// `"1H"` or `"13C"`.
    pub nucleus: String,
    /// Spectrometer frequency in MHz (`400.0`, `500.0`, `600.0`).
    pub spectrometer_mhz: f64,
    /// Right-hand (upfield, low ppm) axis bound (`e.g. -0.5` for ¹H, `0.0` for ¹³C).
    pub ppm_min: f64,
    /// Left-hand (downfield, high ppm) axis bound (`e.g. 10.5` for ¹H, `220.0` for ¹³C).
    pub ppm_max: f64,
    /// Lorentzian half-height linewidth in Hz (`1.2` Hz typical for ¹H).
    pub linewidth_hz: f64,
    pub show_integrals: bool,
    pub show_peak_labels: bool,
    pub title: Option<String>,
    pub signals: Vec<SpectrumSignal1D>,
    /// Optional raw `(ppm, normalized_intensity)` experimental trace points; when empty,
    /// a high-resolution Lorentzian multiplet trace is synthesized directly from `signals`.
    pub custom_trace: Vec<[f64; 2]>,
}

/// Free text caption / compound label / reaction condition block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneText {
    pub pos: [f64; 2],
    pub text: String,
    pub font_size_pt: Option<f64>,
    pub bold: bool,
    pub italic: bool,
    pub centered: bool,
    pub color: Option<RgbaColor>,
}

/// Soft polygon fill inside an aromatic or highlighted ring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneRingFill {
    pub atoms: Vec<usize>,
    pub color: RgbaColor,
}

/// Complete 2D publication chemical drawing, reaction scheme, and 1D NMR scene.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderScene {
    pub atoms: Vec<SceneAtom>,
    pub bonds: Vec<SceneBond>,
    pub ring_fills: Vec<SceneRingFill>,
    pub reaction_arrows: Vec<SceneReactionArrow>,
    pub curved_arrows: Vec<SceneCurvedArrow>,
    pub brackets: Vec<SceneBracket>,
    pub orbitals: Vec<SceneOrbital>,
    pub tlc_plates: Vec<SceneTlcPlate>,
    pub spectra_1d: Vec<SceneSpectrum1D>,
    pub texts: Vec<SceneText>,
    /// Aromatic ring indices drawn with an inner delocalized circle (Robinson style) rather than Kekulé lines.
    pub aromatic_circles: Vec<Vec<usize>>,
}

impl RenderScene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
            && self.reaction_arrows.is_empty()
            && self.curved_arrows.is_empty()
            && self.brackets.is_empty()
            && self.orbitals.is_empty()
            && self.tlc_plates.is_empty()
            && self.spectra_1d.is_empty()
            && self.texts.is_empty()
    }

    pub fn add_atom(&mut self, atom: SceneAtom) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(atom);
        idx
    }

    pub fn add_bond(&mut self, bond: SceneBond) -> usize {
        let idx = self.bonds.len();
        self.bonds.push(bond);
        idx
    }

    /// Translate every element in the scene by `(dx, dy)` in bond-length units.
    pub fn translate(&mut self, dx: f64, dy: f64) {
        for a in &mut self.atoms {
            a.x += dx;
            a.y += dy;
        }
        for arr in &mut self.reaction_arrows {
            arr.from[0] += dx;
            arr.from[1] += dy;
            arr.to[0] += dx;
            arr.to[1] += dy;
        }
        for ca in &mut self.curved_arrows {
            for p in [&mut ca.start, &mut ca.ctrl1, &mut ca.ctrl2, &mut ca.end] {
                p[0] += dx;
                p[1] += dy;
            }
        }
        for br in &mut self.brackets {
            br.min[0] += dx;
            br.min[1] += dy;
            br.max[0] += dx;
            br.max[1] += dy;
        }
        for orb in &mut self.orbitals {
            orb.center[0] += dx;
            orb.center[1] += dy;
        }
        for tlc in &mut self.tlc_plates {
            tlc.origin[0] += dx;
            tlc.origin[1] += dy;
        }
        for sp in &mut self.spectra_1d {
            sp.origin[0] += dx;
            sp.origin[1] += dy;
        }
        for t in &mut self.texts {
            t.pos[0] += dx;
            t.pos[1] += dy;
        }
    }

    /// Merge another `RenderScene` shifted by `(offset_x, offset_y)` bond lengths, remapping atom indices.
    pub fn merge_at(&mut self, other: &RenderScene, offset_x: f64, offset_y: f64) -> usize {
        let base = self.atoms.len();
        for a in &other.atoms {
            let mut na = a.clone();
            na.x += offset_x;
            na.y += offset_y;
            for mc in &mut na.multi_center_atoms {
                *mc += base;
            }
            self.atoms.push(na);
        }
        for b in &other.bonds {
            let mut nb = b.clone();
            nb.a += base;
            nb.b += base;
            self.bonds.push(nb);
        }
        for rf in &other.ring_fills {
            self.ring_fills.push(SceneRingFill {
                atoms: rf.atoms.iter().map(|i| i + base).collect(),
                color: rf.color,
            });
        }
        for c in &other.aromatic_circles {
            self.aromatic_circles
                .push(c.iter().map(|i| i + base).collect());
        }
        for arr in &other.reaction_arrows {
            let mut na = arr.clone();
            na.from[0] += offset_x;
            na.from[1] += offset_y;
            na.to[0] += offset_x;
            na.to[1] += offset_y;
            self.reaction_arrows.push(na);
        }
        for ca in &other.curved_arrows {
            let mut nca = ca.clone();
            for p in [&mut nca.start, &mut nca.ctrl1, &mut nca.ctrl2, &mut nca.end] {
                p[0] += offset_x;
                p[1] += offset_y;
            }
            self.curved_arrows.push(nca);
        }
        for br in &other.brackets {
            let mut nbr = br.clone();
            nbr.min[0] += offset_x;
            nbr.min[1] += offset_y;
            nbr.max[0] += offset_x;
            nbr.max[1] += offset_y;
            self.brackets.push(nbr);
        }
        for orb in &other.orbitals {
            let mut no = orb.clone();
            no.center[0] += offset_x;
            no.center[1] += offset_y;
            self.orbitals.push(no);
        }
        for tlc in &other.tlc_plates {
            let mut nt = tlc.clone();
            nt.origin[0] += offset_x;
            nt.origin[1] += offset_y;
            self.tlc_plates.push(nt);
        }
        for sp in &other.spectra_1d {
            let mut ns = sp.clone();
            ns.origin[0] += offset_x;
            ns.origin[1] += offset_y;
            for sig in &mut ns.signals {
                for idx in &mut sig.atom_indices {
                    *idx += base;
                }
            }
            self.spectra_1d.push(ns);
        }
        for t in &other.texts {
            let mut nt = t.clone();
            nt.pos[0] += offset_x;
            nt.pos[1] += offset_y;
            self.texts.push(nt);
        }
        base
    }

    /// Molecular-coordinate centroid `(cx, cy)` of all atoms, or `(0.0, 0.0)` if empty.
    pub fn atom_centroid(&self) -> [f64; 2] {
        if self.atoms.is_empty() {
            return [0.0, 0.0];
        }
        let n = self.atoms.len() as f64;
        let sx: f64 = self.atoms.iter().map(|a| a.x).sum();
        let sy: f64 = self.atoms.iter().map(|a| a.y).sum();
        [sx / n, sy / n]
    }
}

// ============================================================================
// 2. Device-Independent 2D Vector Primitives (in page points, +x right, +y down)
// ============================================================================

/// Cubic Bezier or polyline path segment in page points (`+x` right, `+y` down).
#[derive(Debug, Clone, PartialEq)]
pub enum PathCmd {
    MoveTo(f64, f64),
    LineTo(f64, f64),
    CubicTo(f64, f64, f64, f64, f64, f64),
    Close,
}

/// Device-independent 2D vector primitive compiled from a [`RenderScene`].
///
/// Exposed publicly so interactive UIs (`egui::Painter` in `nmr-gui`) and vector/raster
/// exporters (`SVG`, `PDF 1.4`, `PNG`) share 100% identical publication geometry!
#[derive(Debug, Clone, PartialEq)]
pub enum VectorPrimitive {
    Line {
        p1: [f64; 2],
        p2: [f64; 2],
        width_pt: f64,
        color: RgbaColor,
        dash_pt: Option<[f64; 2]>,
    },
    Polygon {
        points: Vec<[f64; 2]>,
        fill: Option<RgbaColor>,
        stroke: Option<(f64, RgbaColor)>,
    },
    Path {
        cmds: Vec<PathCmd>,
        fill: Option<RgbaColor>,
        stroke: Option<(f64, RgbaColor)>,
        dash_pt: Option<[f64; 2]>,
    },
    Circle {
        center: [f64; 2],
        radius_pt: f64,
        fill: Option<RgbaColor>,
        stroke: Option<(f64, RgbaColor)>,
        dash_pt: Option<[f64; 2]>,
    },
    Text {
        /// Left baseline origin `(x_pt, y_pt)` in screen-space points (`+y` down).
        pos: [f64; 2],
        text: String,
        font_size_pt: f64,
        bold: bool,
        italic: bool,
        color: RgbaColor,
    },
}

/// Compiled publication scene with normalized bounding box starting at `(0, 0)` and size `(width_pt, height_pt)`.
#[derive(Debug, Clone)]
pub struct CompiledScene {
    pub width_pt: f64,
    pub height_pt: f64,
    /// Offset added to raw `(x_mol * bond_len, -y_mol * bond_len)` to yield `[0..width_pt, 0..height_pt]`.
    pub shift_x_pt: f64,
    pub shift_y_pt: f64,
    pub atom_labels: Vec<Option<FormattedAtomLabel>>,
    pub primitives: Vec<VectorPrimitive>,
}

// ============================================================================
// 3. Ring Perception & Chemical Scene Geometry Compiler
// ============================================================================

/// Find small elementary rings (size 3..=10) in the molecular graph for aesthetic inside
/// double-bond placement and aromatic circles.
pub fn perceive_small_rings(atoms_len: usize, bonds: &[SceneBond]) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); atoms_len];
    for b in bonds {
        if b.order == BondOrder::HydrogenBond || b.a >= atoms_len || b.b >= atoms_len || b.a == b.b
        {
            continue;
        }
        adj[b.a].push(b.b);
        adj[b.b].push(b.a);
    }

    let mut rings: Vec<Vec<usize>> = Vec::new();
    let mut seen_keys: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    // For each bond (u, v), find shortest path between u and v not using direct edge (u, v)
    for b in bonds {
        if b.order == BondOrder::HydrogenBond || b.a >= atoms_len || b.b >= atoms_len || b.a == b.b
        {
            continue;
        }
        let u = b.a;
        let v = b.b;
        let mut prev = vec![usize::MAX; atoms_len];
        let mut dist = vec![usize::MAX; atoms_len];
        let mut queue = std::collections::VecDeque::new();
        dist[u] = 0;
        queue.push_back(u);

        while let Some(cur) = queue.pop_front() {
            if dist[cur] >= 9 {
                continue;
            }
            if cur == v {
                break;
            }
            for &nxt in &adj[cur] {
                if (cur == u && nxt == v) || (cur == v && nxt == u) {
                    continue;
                }
                if dist[nxt] == usize::MAX {
                    dist[nxt] = dist[cur] + 1;
                    prev[nxt] = cur;
                    queue.push_back(nxt);
                }
            }
        }

        if dist[v] != usize::MAX && dist[v] + 1 >= 3 && dist[v] < 10 {
            let mut cycle = Vec::with_capacity(dist[v] + 1);
            let mut cur = v;
            while cur != u && cur != usize::MAX {
                cycle.push(cur);
                cur = prev[cur];
            }
            cycle.push(u);
            let mut key = cycle.clone();
            key.sort_unstable();
            if seen_keys.insert(key) {
                rings.push(cycle);
            }
        }
    }

    rings
}

/// Clip point `p1` toward `p2` against axis-aligned box `[min_x, min_y, max_x, max_y]` around `p1`
/// so a bond starting inside an atom label's clearance box begins cleanly on its boundary!
fn clip_endpoint_outside_box(p1: [f64; 2], p2: [f64; 2], rel_box: [f64; 4]) -> [f64; 2] {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
        return p1;
    }
    // Box around p1: [p1.x + min_x, p1.y + min_y, p1.x + max_x, p1.y + max_y]
    // Find ray exit parameter t in (0, 1) where p1 + t*(p2 - p1) exits the box (containing 0,0).
    if rel_box[0] > 0.0 || rel_box[2] < 0.0 || rel_box[1] > 0.0 || rel_box[3] < 0.0 {
        return p1;
    }
    let tx = if dx > 1e-9 {
        rel_box[2] / dx
    } else if dx < -1e-9 {
        rel_box[0] / dx
    } else {
        f64::INFINITY
    };
    let ty = if dy > 1e-9 {
        rel_box[3] / dy
    } else if dy < -1e-9 {
        rel_box[1] / dy
    } else {
        f64::INFINITY
    };
    let t_exit = tx.min(ty);
    if t_exit > 0.0 && t_exit < 0.85 {
        [p1[0] + t_exit * dx, p1[1] + t_exit * dy]
    } else {
        p1
    }
}

/// Segment-segment intersection parameter `(t, u)` strictly inside `(0.15..0.85)` for crossing-bond gaps.
fn segment_crossing(a1: [f64; 2], a2: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> Option<f64> {
    let r = [a2[0] - a1[0], a2[1] - a1[1]];
    let s = [b2[0] - b1[0], b2[1] - b1[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() < 1e-6 {
        return None;
    }
    let qp = [b1[0] - a1[0], b1[1] - a1[1]];
    let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
    let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
    if (0.14..=0.86).contains(&t) && (0.14..=0.86).contains(&u) {
        Some(t)
    } else {
        None
    }
}

/// Compile a [`RenderScene`] + [`DocumentStyle`] into normalized [`CompiledScene`] vector primitives.
pub fn compile_scene_primitives(scene: &RenderScene, style: &DocumentStyle) -> CompiledScene {
    let bl = style.bond_length_pt.max(2.0);
    let lw = style.line_width_pt.max(0.2);
    let to_pt = |xm: f64, ym: f64| -> [f64; 2] { [xm * bl, -ym * bl] };

    let mut raw_prims: Vec<VectorPrimitive> = Vec::new();
    let n_atoms = scene.atoms.len();

    // Update multi-center centroid positions from their ring atoms if present
    let mut atom_pos_mol: Vec<[f64; 2]> = scene.atoms.iter().map(|a| [a.x, a.y]).collect();
    for i in 0..n_atoms {
        if scene.atoms[i].is_multi_center() {
            let mc = &scene.atoms[i].multi_center_atoms;
            let valid: Vec<[f64; 2]> = mc
                .iter()
                .filter_map(|&idx| atom_pos_mol.get(idx).copied())
                .collect();
            if !valid.is_empty() {
                let cx = valid.iter().map(|p| p[0]).sum::<f64>() / valid.len() as f64;
                let cy = valid.iter().map(|p| p[1]).sum::<f64>() / valid.len() as f64;
                atom_pos_mol[i] = [cx, cy];
            }
        }
    }

    let atom_pts: Vec<[f64; 2]> = atom_pos_mol.iter().map(|p| to_pt(p[0], p[1])).collect();

    // 0. Ring fills (drawn behind everything else)
    for rf in &scene.ring_fills {
        let pts: Vec<[f64; 2]> = rf
            .atoms
            .iter()
            .filter_map(|&idx| atom_pts.get(idx).copied())
            .collect();
        if pts.len() >= 3 {
            raw_prims.push(VectorPrimitive::Polygon {
                points: pts,
                fill: Some(rf.color),
                stroke: None,
            });
        }
    }

    // Atom & bond soft highlights (drawn under bonds and labels)
    let halo_r = 0.34 * bl;
    for (i, atom) in scene.atoms.iter().enumerate() {
        if let Some(hl) = atom.highlight {
            raw_prims.push(VectorPrimitive::Circle {
                center: atom_pts[i],
                radius_pt: halo_r,
                fill: Some(hl),
                stroke: None,
                dash_pt: None,
            });
        }
    }
    for b in &scene.bonds {
        if let Some(hl) = b.highlight {
            if b.a < n_atoms && b.b < n_atoms {
                raw_prims.push(VectorPrimitive::Line {
                    p1: atom_pts[b.a],
                    p2: atom_pts[b.b],
                    width_pt: 0.55 * bl,
                    color: hl,
                    dash_pt: None,
                });
            }
        }
    }

    // 1. Precompute atom valence & outgoing bond vectors in molecular y-up space
    let mut bond_order_sums = vec![0.0_f64; n_atoms];
    let mut bond_dirs_mol: Vec<Vec<[f64; 2]>> = vec![Vec::new(); n_atoms];
    let mut atom_degree = vec![0_usize; n_atoms];

    for b in &scene.bonds {
        if b.a >= n_atoms || b.b >= n_atoms || b.a == b.b {
            continue;
        }
        let ord = if scene.atoms[b.a].is_multi_center() || scene.atoms[b.b].is_multi_center() {
            0.0
        } else {
            b.order.valence_order()
        };
        bond_order_sums[b.a] += ord;
        bond_order_sums[b.b] += ord;
        if b.order != BondOrder::HydrogenBond {
            atom_degree[b.a] += 1;
            atom_degree[b.b] += 1;
        }
        let dx = atom_pos_mol[b.b][0] - atom_pos_mol[b.a][0];
        let dy = atom_pos_mol[b.b][1] - atom_pos_mol[b.a][1];
        let d = (dx * dx + dy * dy).sqrt();
        if d > 1e-9 {
            bond_dirs_mol[b.a].push([dx / d, dy / d]);
            bond_dirs_mol[b.b].push([-dx / d, -dy / d]);
        }
    }

    // Linear cumulated carbon detection (e.g. allene center `=C=` at ~180° gets `C` shown or dot)
    let mut atom_labels: Vec<Option<FormattedAtomLabel>> = Vec::with_capacity(n_atoms);
    for (i, atom) in scene.atoms.iter().enumerate() {
        if atom.is_multi_center() {
            atom_labels.push(None);
            continue;
        }
        let is_linear_cumulene = atom.label == "C"
            && atom.text.is_none()
            && bond_dirs_mol[i].len() == 2
            && bond_order_sums[i] >= 3.9
            && (bond_dirs_mol[i][0][0] * bond_dirs_mol[i][1][0]
                + bond_dirs_mol[i][0][1] * bond_dirs_mol[i][1][1])
                < -0.94;
        let lbl = layout_atom_label(
            &atom.label,
            atom.text.as_deref(),
            atom.charge,
            atom.radical,
            atom.isotope,
            atom.explicit_h,
            atom.show_carbon || is_linear_cumulene,
            bond_order_sums[i],
            &bond_dirs_mol[i],
            style,
        );
        atom_labels.push(lbl);
    }

    // 2. Perceive rings for aesthetic double-bond inside placement
    let rings = perceive_small_rings(n_atoms, &scene.bonds);
    let circle_ring_bond_set: std::collections::HashSet<(usize, usize)> = {
        let mut s = std::collections::HashSet::new();
        for r in &scene.aromatic_circles {
            for k in 0..r.len() {
                let u = r[k];
                let v = r[(k + 1) % r.len()];
                s.insert((u.min(v), u.max(v)));
            }
        }
        s
    };

    // Precompute clipped endpoints for all bonds
    let clipped_ends: Vec<([f64; 2], [f64; 2])> = scene
        .bonds
        .iter()
        .map(|b| {
            if b.a >= n_atoms || b.b >= n_atoms {
                return ([0.0, 0.0], [0.0, 0.0]);
            }
            let mut p1 = atom_pts[b.a];
            let mut p2 = atom_pts[b.b];
            if let Some(lbl_a) = &atom_labels[b.a] {
                p1 = clip_endpoint_outside_box(p1, p2, lbl_a.clearance_box_pt(style));
            }
            if let Some(lbl_b) = &atom_labels[b.b] {
                p2 = clip_endpoint_outside_box(p2, p1, lbl_b.clearance_box_pt(style));
            }
            (p1, p2)
        })
        .collect();

    // 3. Emit bonds
    for (bi, b) in scene.bonds.iter().enumerate() {
        if b.a >= n_atoms || b.b >= n_atoms || b.a == b.b {
            continue;
        }
        let (p1, p2) = clipped_ends[bi];
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < 1e-4 {
            continue;
        }
        let ux = dx / seg_len;
        let uy = dy / seg_len;
        // Screen-space left normal (`+x` right, `+y` down: rotating `(ux, uy)` -90° is `(uy, -ux)`)
        let nx = uy;
        let ny = -ux;

        let bond_color = b.color.unwrap_or(style.foreground);

        // Hydrogen bond (`- - -`)
        if b.order == BondOrder::HydrogenBond {
            raw_prims.push(VectorPrimitive::Line {
                p1,
                p2,
                width_pt: lw,
                color: bond_color,
                dash_pt: Some([2.2 * lw, 2.2 * lw]),
            });
            continue;
        }

        // Specialized stereo / display styles on single bonds
        match b.stereo {
            BondStereo::UpWedge => {
                let half_tip = 0.5 * style.wedge_tip_width_pt();
                let half_wide = 0.5 * style.wedge_wide_width_pt();
                raw_prims.push(VectorPrimitive::Polygon {
                    points: vec![
                        [p1[0] + nx * half_tip, p1[1] + ny * half_tip],
                        [p2[0] + nx * half_wide, p2[1] + ny * half_wide],
                        [p2[0] - nx * half_wide, p2[1] - ny * half_wide],
                        [p1[0] - nx * half_tip, p1[1] - ny * half_tip],
                    ],
                    fill: Some(bond_color),
                    stroke: None,
                });
                continue;
            }
            BondStereo::UpHollow => {
                let half_tip = 0.5 * style.wedge_tip_width_pt();
                let half_wide = 0.5 * style.wedge_wide_width_pt();
                raw_prims.push(VectorPrimitive::Polygon {
                    points: vec![
                        [p1[0] + nx * half_tip, p1[1] + ny * half_tip],
                        [p2[0] + nx * half_wide, p2[1] + ny * half_wide],
                        [p2[0] - nx * half_wide, p2[1] - ny * half_wide],
                        [p1[0] - nx * half_tip, p1[1] - ny * half_tip],
                    ],
                    fill: style.background,
                    stroke: Some((lw, bond_color)),
                });
                continue;
            }
            BondStereo::DownHash => {
                let n_bars = style.hash_count(seg_len);
                let half_tip = 0.5 * lw.max(style.wedge_tip_width_pt());
                let half_wide = 0.5 * style.wedge_wide_width_pt();
                for k in 0..n_bars {
                    let t = if n_bars <= 1 {
                        0.5
                    } else {
                        k as f64 / (n_bars - 1) as f64
                    };
                    let cx = p1[0] + t * dx;
                    let cy = p1[1] + t * dy;
                    let hw = half_tip + t * (half_wide - half_tip);
                    raw_prims.push(VectorPrimitive::Line {
                        p1: [cx - nx * hw, cy - ny * hw],
                        p2: [cx + nx * hw, cy + ny * hw],
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });
                }
                continue;
            }
            BondStereo::Wavy => {
                let amp = 0.5 * style.bold_width_pt.max(1.6 * lw);
                let half_waves = ((seg_len / (2.2 * amp)).round() as usize).max(4);
                let step = seg_len / half_waves as f64;
                let mut cmds = vec![PathCmd::MoveTo(p1[0], p1[1])];
                for k in 0..half_waves {
                    let sgn = if k % 2 == 0 { 1.0 } else { -1.0 };
                    let s0 = k as f64 * step;
                    let s1 = (k + 1) as f64 * step;
                    let c1x = p1[0] + (s0 + 0.35 * step) * ux + sgn * amp * nx;
                    let c1y = p1[1] + (s0 + 0.35 * step) * uy + sgn * amp * ny;
                    let c2x = p1[0] + (s0 + 0.65 * step) * ux + sgn * amp * nx;
                    let c2y = p1[1] + (s0 + 0.65 * step) * uy + sgn * amp * ny;
                    let ex = p1[0] + s1 * ux;
                    let ey = p1[1] + s1 * uy;
                    cmds.push(PathCmd::CubicTo(c1x, c1y, c2x, c2y, ex, ey));
                }
                raw_prims.push(VectorPrimitive::Path {
                    cmds,
                    fill: None,
                    stroke: Some((lw, bond_color)),
                    dash_pt: None,
                });
                continue;
            }
            BondStereo::Dative => {
                let head_len = (4.6 * lw).min(0.38 * seg_len);
                let head_hw = 2.1 * lw;
                let shaft_end = [
                    p2[0] - ux * (0.65 * head_len),
                    p2[1] - uy * (0.65 * head_len),
                ];
                raw_prims.push(VectorPrimitive::Line {
                    p1,
                    p2: shaft_end,
                    width_pt: lw,
                    color: bond_color,
                    dash_pt: None,
                });
                raw_prims.push(VectorPrimitive::Polygon {
                    points: vec![
                        p2,
                        [
                            p2[0] - ux * head_len + nx * head_hw,
                            p2[1] - uy * head_len + ny * head_hw,
                        ],
                        [
                            p2[0] - ux * (0.72 * head_len),
                            p2[1] - uy * (0.72 * head_len),
                        ],
                        [
                            p2[0] - ux * head_len - nx * head_hw,
                            p2[1] - uy * head_len - ny * head_hw,
                        ],
                    ],
                    fill: Some(bond_color),
                    stroke: None,
                });
                continue;
            }
            BondStereo::Bold if b.order == BondOrder::Single => {
                raw_prims.push(VectorPrimitive::Line {
                    p1,
                    p2,
                    width_pt: style.bold_width_pt,
                    color: bond_color,
                    dash_pt: None,
                });
                continue;
            }
            BondStereo::Dashed if b.order == BondOrder::Single => {
                raw_prims.push(VectorPrimitive::Line {
                    p1,
                    p2,
                    width_pt: lw,
                    color: bond_color,
                    dash_pt: Some([2.6 * lw, 2.0 * lw]),
                });
                continue;
            }
            BondStereo::CrossedDouble => {
                let half_sp = 0.5 * style.double_bond_spacing_pt();
                raw_prims.push(VectorPrimitive::Line {
                    p1: [p1[0] + nx * half_sp, p1[1] + ny * half_sp],
                    p2: [p2[0] - nx * half_sp, p2[1] - ny * half_sp],
                    width_pt: lw,
                    color: bond_color,
                    dash_pt: None,
                });
                raw_prims.push(VectorPrimitive::Line {
                    p1: [p1[0] - nx * half_sp, p1[1] - ny * half_sp],
                    p2: [p2[0] + nx * half_sp, p2[1] + ny * half_sp],
                    width_pt: lw,
                    color: bond_color,
                    dash_pt: None,
                });
                continue;
            }
            _ => {}
        }

        // If this bond belongs to an aromatic-circle ring, draw only the single outer frame bond
        let edge_key = (b.a.min(b.b), b.a.max(b.b));
        let effective_order = if circle_ring_bond_set.contains(&edge_key) {
            BondOrder::Single
        } else {
            b.order
        };

        match effective_order {
            BondOrder::Single => {
                // Check if any LATER single bond crosses over this bond -> leave a clean bridge gap!
                let mut cuts: Vec<f64> = Vec::new();
                for (lb, &(lp1, lp2)) in scene.bonds.iter().zip(clipped_ends.iter()).skip(bi + 1) {
                    if lb.a == b.a || lb.a == b.b || lb.b == b.a || lb.b == b.b {
                        continue;
                    }
                    if let Some(t_cross) = segment_crossing(p1, p2, lp1, lp2) {
                        cuts.push(t_cross);
                    }
                }
                if cuts.is_empty() {
                    raw_prims.push(VectorPrimitive::Line {
                        p1,
                        p2,
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });
                } else {
                    cuts.sort_by(|x, y| x.total_cmp(y));
                    let gap_dt = ((2.4 * style.margin_width_pt) / seg_len).min(0.18);
                    let mut cur_t = 0.0;
                    for tc in cuts {
                        let t_end = (tc - gap_dt).max(cur_t);
                        if t_end > cur_t + 0.02 {
                            raw_prims.push(VectorPrimitive::Line {
                                p1: [p1[0] + cur_t * dx, p1[1] + cur_t * dy],
                                p2: [p1[0] + t_end * dx, p1[1] + t_end * dy],
                                width_pt: lw,
                                color: bond_color,
                                dash_pt: None,
                            });
                        }
                        cur_t = (tc + gap_dt).min(1.0);
                    }
                    if 1.0 > cur_t + 0.02 {
                        raw_prims.push(VectorPrimitive::Line {
                            p1: [p1[0] + cur_t * dx, p1[1] + cur_t * dy],
                            p2,
                            width_pt: lw,
                            color: bond_color,
                            dash_pt: None,
                        });
                    }
                }
            }
            BondOrder::Double | BondOrder::Aromatic => {
                let sp = style.double_bond_spacing_pt();
                let is_aromatic_dashed = effective_order == BondOrder::Aromatic;

                // Determine whether centered (`C=O`, `C=C` terminal/allene) or offset inside a ring/chain!
                let mut chosen_side = b.side;
                let mut best_ring: Option<&Vec<usize>> = None;

                if chosen_side == DoubleBondSide::Auto {
                    // Find best ring containing edge (b.a, b.b): prefer 6-membered rings, then smaller rings
                    for r in &rings {
                        let has_edge = (0..r.len()).any(|k| {
                            let u = r[k];
                            let v = r[(k + 1) % r.len()];
                            (u == b.a && v == b.b) || (u == b.b && v == b.a)
                        });
                        if has_edge {
                            let score_new = (r.len() as isize - 6).unsigned_abs();
                            let replace = match best_ring {
                                None => true,
                                Some(old) => score_new < (old.len() as isize - 6).unsigned_abs(),
                            };
                            if replace {
                                best_ring = Some(r);
                            }
                        }
                    }

                    if let Some(ring) = best_ring {
                        // Project ring centroid onto normal `(nx, ny)` in screen points
                        let rcx =
                            ring.iter().map(|&i| atom_pts[i][0]).sum::<f64>() / ring.len() as f64;
                        let rcy =
                            ring.iter().map(|&i| atom_pts[i][1]).sum::<f64>() / ring.len() as f64;
                        let mid = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])];
                        let dot_n = (rcx - mid[0]) * nx + (rcy - mid[1]) * ny;
                        chosen_side = if dot_n >= 0.0 {
                            DoubleBondSide::Left
                        } else {
                            DoubleBondSide::Right
                        };
                    } else if (atom_degree[b.a] <= 1
                        || atom_degree[b.b] <= 1
                        || atom_labels[b.a].is_some()
                        || atom_labels[b.b].is_some())
                        && !is_aromatic_dashed
                    {
                        // Terminal carbonyl / imine / alkene (`R2C=O`, `H2C=CH2`) -> symmetric centered double bond!
                        chosen_side = DoubleBondSide::Center;
                    } else {
                        // Acyclic internal alkene -> place secondary line on side of more substituents
                        let mut side_pull = 0.0;
                        for sb in &scene.bonds {
                            let other = if sb.a == b.a && sb.b != b.b {
                                Some((atom_pts[b.a], atom_pts[sb.b]))
                            } else if sb.b == b.a && sb.a != b.b {
                                Some((atom_pts[b.a], atom_pts[sb.a]))
                            } else if sb.a == b.b && sb.b != b.a {
                                Some((atom_pts[b.b], atom_pts[sb.b]))
                            } else if sb.b == b.b && sb.a != b.a {
                                Some((atom_pts[b.b], atom_pts[sb.a]))
                            } else {
                                None
                            };
                            if let Some((base_p, nbr_p)) = other {
                                side_pull +=
                                    (nbr_p[0] - base_p[0]) * nx + (nbr_p[1] - base_p[1]) * ny;
                            }
                        }
                        chosen_side = if side_pull >= 0.0 {
                            DoubleBondSide::Left
                        } else {
                            DoubleBondSide::Right
                        };
                    }
                }

                if chosen_side == DoubleBondSide::Center {
                    let half = 0.5 * sp;
                    let inset = if atom_labels[b.a].is_none() && atom_labels[b.b].is_none() {
                        style.terminal_double_inset_pt
                    } else {
                        0.0
                    };
                    let q1 = [p1[0] + inset * ux, p1[1] + inset * uy];
                    let q2 = [p2[0] - inset * ux, p2[1] - inset * uy];
                    raw_prims.push(VectorPrimitive::Line {
                        p1: [q1[0] + nx * half, q1[1] + ny * half],
                        p2: [q2[0] + nx * half, q2[1] + ny * half],
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });
                    raw_prims.push(VectorPrimitive::Line {
                        p1: [q1[0] - nx * half, q1[1] - ny * half],
                        p2: [q2[0] - nx * half, q2[1] - ny * half],
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: if is_aromatic_dashed {
                            Some([2.4 * lw, 2.0 * lw])
                        } else {
                            None
                        },
                    });
                } else {
                    let sgn = if chosen_side == DoubleBondSide::Left {
                        1.0
                    } else {
                        -1.0
                    };
                    // Primary outer bond along atom-to-atom axis
                    raw_prims.push(VectorPrimitive::Line {
                        p1,
                        p2,
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });

                    // Compute aesthetic inset for inner bond (`spacing / tan(theta/2)` when unlabeled)
                    let trim_a = if atom_labels[b.a].is_some() {
                        0.0
                    } else {
                        style.ring_inner_trim_pt(2.0 * PI / 3.0).min(0.28 * seg_len)
                    };
                    let trim_b = if atom_labels[b.b].is_some() {
                        0.0
                    } else {
                        style.ring_inner_trim_pt(2.0 * PI / 3.0).min(0.28 * seg_len)
                    };
                    let ip1 = [
                        p1[0] + trim_a * ux + sgn * sp * nx,
                        p1[1] + trim_a * uy + sgn * sp * ny,
                    ];
                    let ip2 = [
                        p2[0] - trim_b * ux + sgn * sp * nx,
                        p2[1] - trim_b * uy + sgn * sp * ny,
                    ];
                    raw_prims.push(VectorPrimitive::Line {
                        p1: ip1,
                        p2: ip2,
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: if is_aromatic_dashed {
                            Some([2.4 * lw, 2.0 * lw])
                        } else {
                            None
                        },
                    });
                }
            }
            BondOrder::Triple => {
                let sp = style.double_bond_spacing_pt();
                raw_prims.push(VectorPrimitive::Line {
                    p1,
                    p2,
                    width_pt: lw,
                    color: bond_color,
                    dash_pt: None,
                });
                for sgn in [-1.0, 1.0] {
                    raw_prims.push(VectorPrimitive::Line {
                        p1: [p1[0] + sgn * sp * nx, p1[1] + sgn * sp * ny],
                        p2: [p2[0] + sgn * sp * nx, p2[1] + sgn * sp * ny],
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });
                }
            }
            BondOrder::Quadruple => {
                let sp = 0.85 * style.double_bond_spacing_pt();
                for mult in [-1.5, -0.5, 0.5, 1.5] {
                    raw_prims.push(VectorPrimitive::Line {
                        p1: [p1[0] + mult * sp * nx, p1[1] + mult * sp * ny],
                        p2: [p2[0] + mult * sp * nx, p2[1] + mult * sp * ny],
                        width_pt: lw,
                        color: bond_color,
                        dash_pt: None,
                    });
                }
            }
            BondOrder::HydrogenBond => {}
        }

        if let Some(ez) = &b.ez_label {
            let mid = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]) - 0.35 * bl];
            let sz = 0.78 * style.atom_label_size_pt;
            let w = text_advance_pt(ez, sz);
            raw_prims.push(VectorPrimitive::Text {
                pos: [mid[0] - 0.5 * w, mid[1]],
                text: ez.clone(),
                font_size_pt: sz,
                bold: false,
                italic: true,
                color: bond_color,
            });
        }
    }

    // Aromatic inner Robinson circles
    for r in &scene.aromatic_circles {
        let pts: Vec<[f64; 2]> = r.iter().filter_map(|&i| atom_pts.get(i).copied()).collect();
        if pts.len() >= 3 {
            let cx = pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64;
            let cy = pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64;
            let min_vertex_r = pts
                .iter()
                .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min);
            let apothem = min_vertex_r * (PI / pts.len() as f64).cos();
            let rad = (apothem - 1.35 * style.double_bond_spacing_pt()).max(0.25 * bl);
            raw_prims.push(VectorPrimitive::Circle {
                center: [cx, cy],
                radius_pt: rad,
                fill: None,
                stroke: Some((lw, style.foreground)),
                dash_pt: None,
            });
        }
    }

    // 4. Emit formatted atom labels (with subscripts/superscripts), lone pairs, radicals, CIP badges
    for (i, atom) in scene.atoms.iter().enumerate() {
        if atom.is_multi_center() {
            continue;
        }
        let anchor = atom_pts[i];
        let col = style.color_for_element(&atom.label, atom.color);

        if let Some(lbl) = &atom_labels[i] {
            let cap = style.cap_height_pt();
            let base_y = anchor[1] + 0.5 * cap;

            match lbl.direction {
                LabelDirection::StackedAbove | LabelDirection::StackedBelow => {
                    let pitch = style.stacked_pitch_pt();
                    for (ti, tok) in lbl.tokens.iter().enumerate() {
                        let dy_stack =
                            (ti as isize - lbl.anchor_token_index as isize) as f64 * pitch;
                        let tw = text_advance_pt(&tok.text, style.atom_label_size_pt);
                        raw_prims.push(VectorPrimitive::Text {
                            pos: [anchor[0] - 0.5 * tw, base_y + dy_stack],
                            text: tok.text.clone(),
                            font_size_pt: style.atom_label_size_pt,
                            bold: style.atom_label_bold,
                            italic: false,
                            color: col,
                        });
                    }
                }
                _ => {
                    let mut cursor_x = anchor[0] + lbl.min_x_pt;
                    for tok in &lbl.tokens {
                        let (sz, y_shift) = match tok.script {
                            ScriptKind::Normal => (style.atom_label_size_pt, 0.0),
                            ScriptKind::Subscript => {
                                (style.subscript_size_pt(), style.subscript_drop_pt())
                            }
                            ScriptKind::Superscript => {
                                (style.superscript_size_pt(), -style.superscript_rise_pt())
                            }
                        };
                        let adv = text_advance_pt(&tok.text, sz);
                        raw_prims.push(VectorPrimitive::Text {
                            pos: [cursor_x, base_y + y_shift],
                            text: tok.text.clone(),
                            font_size_pt: sz,
                            bold: style.atom_label_bold,
                            italic: false,
                            color: col,
                        });
                        cursor_x += adv;
                    }
                }
            }
        } else if atom.charge != 0 {
            // Skeletal carbon carbocation / carbanion (`+` or `−` near vertex)
            let ch_txt = if atom.charge > 0 { "+" } else { "−" };
            raw_prims.push(VectorPrimitive::Text {
                pos: [anchor[0] + 0.14 * bl, anchor[1] - 0.14 * bl],
                text: ch_txt.to_string(),
                font_size_pt: 0.85 * style.atom_label_size_pt,
                bold: true,
                italic: false,
                color: col,
            });
        }

        // Lone pairs (pairs of crisp dots placed into open angular sectors around the atom)
        if atom.lone_pairs > 0 {
            let mut sum_vx = 0.0;
            let mut sum_vy = 0.0;
            for d in &bond_dirs_mol[i] {
                sum_vx += d[0];
                sum_vy -= d[1]; // screen y-down
            }
            let base_ang = if sum_vx.hypot(sum_vy) > 1e-4 {
                (-sum_vy).atan2(-sum_vx)
            } else {
                -PI / 2.0
            };
            let lp_dist = 0.44 * bl;
            let dot_sep = 1.45 * lw.max(0.7);
            let dot_r = 0.95 * lw.max(0.65);
            for k in 0..atom.lone_pairs {
                let spread = if atom.lone_pairs == 1 {
                    0.0
                } else {
                    (k as f64 - 0.5 * (atom.lone_pairs - 1) as f64) * (PI / 2.2)
                };
                let ang = base_ang + spread;
                let cx = anchor[0] + lp_dist * ang.cos();
                let cy = anchor[1] + lp_dist * ang.sin();
                let tx = -ang.sin();
                let ty = ang.cos();
                for sgn in [-1.0, 1.0] {
                    raw_prims.push(VectorPrimitive::Circle {
                        center: [cx + sgn * dot_sep * tx, cy + sgn * dot_sep * ty],
                        radius_pt: dot_r,
                        fill: Some(col),
                        stroke: None,
                        dash_pt: None,
                    });
                }
            }
        }

        // Radical dot (`•`)
        if atom.radical > 0 {
            raw_prims.push(VectorPrimitive::Circle {
                center: [anchor[0] + 0.32 * bl, anchor[1] - 0.30 * bl],
                radius_pt: 1.25 * lw.max(0.7),
                fill: Some(col),
                stroke: None,
                dash_pt: None,
            });
        }

        // Stereochemical CIP descriptor `(R)` / `(S)` / `(M)` / `(P)`
        if let Some(cip) = &atom.cip_label {
            let mut sum_vx = 0.0;
            let mut sum_vy = 0.0;
            for d in &bond_dirs_mol[i] {
                sum_vx += d[0];
                sum_vy -= d[1];
            }
            let open_ang = if sum_vx.hypot(sum_vy) > 1e-4 {
                (-sum_vy).atan2(-sum_vx)
            } else {
                -PI / 2.0
            };
            let sz = 0.76 * style.atom_label_size_pt;
            let w = text_advance_pt(cip, sz);
            let cx = anchor[0] + 0.44 * bl * open_ang.cos();
            let cy = anchor[1] + 0.44 * bl * open_ang.sin() + 0.34 * sz;
            raw_prims.push(VectorPrimitive::Text {
                pos: [cx - 0.5 * w, cy],
                text: cip.clone(),
                font_size_pt: sz,
                bold: false,
                italic: true,
                color: RgbaColor::rgb(0x37, 0x41, 0x51),
            });
        }
    }

    // 5. Reaction Arrows (Solid, Equilibrium, Unbalanced, Resonance, Retrosynthetic, NoGo, Dashed)
    for arr in &scene.reaction_arrows {
        emit_reaction_arrow(&mut raw_prims, arr, style);
    }

    // 6. Bezier Curved 2e⁻ Full & 1e⁻ Fishhook Electron-Pushing Arrows
    for ca in &scene.curved_arrows {
        emit_curved_arrow(&mut raw_prims, ca, style);
    }

    // 7. Polymer / Enclosure Brackets (`[ ]_n`, `( )_n`, `{ }_n`)
    for br in &scene.brackets {
        emit_bracket(&mut raw_prims, br, style);
    }

    // 8. Atomic & Hybrid Orbitals (`s`, `p`, `d`, `dz2`, `sp3`)
    for orb in &scene.orbitals {
        emit_orbital(&mut raw_prims, orb, style);
    }

    // 9. Thin-Layer Chromatography (TLC) Plates
    for tlc in &scene.tlc_plates {
        emit_tlc_plate(&mut raw_prims, tlc, style);
    }

    // 10. Embedded 1D NMR Spectrum Plots
    for spec in &scene.spectra_1d {
        emit_spectrum_1d(&mut raw_prims, spec, style);
    }

    // 11. Free Text Labels & Captions
    for txt in &scene.texts {
        let p = to_pt(txt.pos[0], txt.pos[1]);
        let sz = txt.font_size_pt.unwrap_or(style.caption_size_pt);
        let col = txt.color.unwrap_or(style.foreground);
        let line_h = 1.25 * sz;
        for (li, line) in txt.text.lines().enumerate() {
            let x = if txt.centered {
                p[0] - 0.5 * text_advance_pt(line, sz)
            } else {
                p[0]
            };
            raw_prims.push(VectorPrimitive::Text {
                pos: [x, p[1] + li as f64 * line_h],
                text: line.to_string(),
                font_size_pt: sz,
                bold: txt.bold,
                italic: txt.italic,
                color: col,
            });
        }
    }

    // Normalize bounding box to `[0..width_pt, 0..height_pt]` with `style.export_padding_pt`
    normalize_compiled_scene(raw_prims, atom_labels, style)
}

fn emit_reaction_arrow(
    out: &mut Vec<VectorPrimitive>,
    arr: &SceneReactionArrow,
    style: &DocumentStyle,
) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let p1 = [arr.from[0] * bl, -arr.from[1] * bl];
    let p2 = [arr.to[0] * bl, -arr.to[1] * bl];
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let len = dx.hypot(dy);
    if len < 1e-3 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    let nx = uy;
    let ny = -ux;
    let col = arr.color.unwrap_or(style.foreground);

    let head_len = (6.2 * lw).clamp(3.2, 0.35 * len);
    let head_hw = 2.4 * lw.max(0.5);

    let full_head = |tip: [f64; 2], dir_x: f64, dir_y: f64| -> VectorPrimitive {
        let hnx = dir_y;
        let hny = -dir_x;
        VectorPrimitive::Polygon {
            points: vec![
                tip,
                [
                    tip[0] - dir_x * head_len + hnx * head_hw,
                    tip[1] - dir_y * head_len + hny * head_hw,
                ],
                [
                    tip[0] - dir_x * (0.76 * head_len),
                    tip[1] - dir_y * (0.76 * head_len),
                ],
                [
                    tip[0] - dir_x * head_len - hnx * head_hw,
                    tip[1] - dir_y * head_len - hny * head_hw,
                ],
            ],
            fill: Some(col),
            stroke: None,
        }
    };

    match arr.kind {
        ArrowKind::Solid | ArrowKind::Dashed | ArrowKind::NoGo => {
            let dash = if arr.kind == ArrowKind::Dashed {
                Some([3.5 * lw, 2.5 * lw])
            } else {
                None
            };
            out.push(VectorPrimitive::Line {
                p1,
                p2: [
                    p2[0] - ux * (0.70 * head_len),
                    p2[1] - uy * (0.70 * head_len),
                ],
                width_pt: lw,
                color: col,
                dash_pt: dash,
            });
            out.push(full_head(p2, ux, uy));

            if arr.kind == ArrowKind::NoGo {
                let mid = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])];
                let s = 3.6 * lw.max(0.6);
                for sgn in [-1.0, 1.0] {
                    out.push(VectorPrimitive::Line {
                        p1: [
                            mid[0] - s * ux - sgn * s * nx,
                            mid[1] - s * uy - sgn * s * ny,
                        ],
                        p2: [
                            mid[0] + s * ux + sgn * s * nx,
                            mid[1] + s * uy + sgn * s * ny,
                        ],
                        width_pt: 1.25 * lw,
                        color: col,
                        dash_pt: None,
                    });
                }
            }
        }
        ArrowKind::Resonance => {
            out.push(VectorPrimitive::Line {
                p1: [
                    p1[0] + ux * (0.70 * head_len),
                    p1[1] + uy * (0.70 * head_len),
                ],
                p2: [
                    p2[0] - ux * (0.70 * head_len),
                    p2[1] - uy * (0.70 * head_len),
                ],
                width_pt: lw,
                color: col,
                dash_pt: None,
            });
            out.push(full_head(p2, ux, uy));
            out.push(full_head(p1, -ux, -uy));
        }
        ArrowKind::Equilibrium | ArrowKind::UnbalancedEquilibrium => {
            let sep = 1.65 * lw.max(0.6);
            // Top forward harpoon (barb on +n outer side)
            let f1 = [p1[0] + nx * sep, p1[1] + ny * sep];
            let f2 = [p2[0] + nx * sep, p2[1] + ny * sep];
            out.push(VectorPrimitive::Line {
                p1: f1,
                p2: [f2[0] - ux * (0.5 * head_len), f2[1] - uy * (0.5 * head_len)],
                width_pt: lw,
                color: col,
                dash_pt: None,
            });
            out.push(VectorPrimitive::Polygon {
                points: vec![
                    f2,
                    [
                        f2[0] - ux * head_len + nx * (1.25 * head_hw),
                        f2[1] - uy * head_len + ny * (1.25 * head_hw),
                    ],
                    [
                        f2[0] - ux * (0.75 * head_len),
                        f2[1] - uy * (0.75 * head_len),
                    ],
                ],
                fill: Some(col),
                stroke: None,
            });

            // Bottom reverse harpoon (shortened when UnbalancedEquilibrium)
            let inset = if arr.kind == ArrowKind::UnbalancedEquilibrium {
                0.22 * len
            } else {
                0.0
            };
            let r_tip = [p1[0] + inset * ux - nx * sep, p1[1] + inset * uy - ny * sep];
            let r_tail = [p2[0] - inset * ux - nx * sep, p2[1] - inset * uy - ny * sep];
            out.push(VectorPrimitive::Line {
                p1: r_tail,
                p2: [
                    r_tip[0] + ux * (0.5 * head_len),
                    r_tip[1] + uy * (0.5 * head_len),
                ],
                width_pt: lw,
                color: col,
                dash_pt: None,
            });
            out.push(VectorPrimitive::Polygon {
                points: vec![
                    r_tip,
                    [
                        r_tip[0] + ux * head_len - nx * (1.25 * head_hw),
                        r_tip[1] + uy * head_len - ny * (1.25 * head_hw),
                    ],
                    [
                        r_tip[0] + ux * (0.75 * head_len),
                        r_tip[1] + uy * (0.75 * head_len),
                    ],
                ],
                fill: Some(col),
                stroke: None,
            });
        }
        ArrowKind::Retrosynthetic => {
            let sep = 2.0 * lw.max(0.6);
            let hw = 2.1 * sep;
            let hl = 2.1 * sep;
            let shaft_end_t = (len - 0.6 * hl).max(0.2 * len);
            for sgn in [-1.0, 1.0] {
                out.push(VectorPrimitive::Line {
                    p1: [p1[0] + sgn * sep * nx, p1[1] + sgn * sep * ny],
                    p2: [
                        p1[0] + shaft_end_t * ux + sgn * sep * nx,
                        p1[1] + shaft_end_t * uy + sgn * sep * ny,
                    ],
                    width_pt: lw,
                    color: col,
                    dash_pt: None,
                });
                // Open chevron head (`=>`)
                out.push(VectorPrimitive::Line {
                    p1: [
                        p2[0] - hl * ux + sgn * hw * nx,
                        p2[1] - hl * uy + sgn * hw * ny,
                    ],
                    p2,
                    width_pt: lw,
                    color: col,
                    dash_pt: None,
                });
            }
        }
    }

    // Reagent / condition captions above and below the reaction arrow
    let mid = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])];
    let sz = style.caption_size_pt;
    if let Some(above) = &arr.above_text {
        let w = text_advance_pt(above, sz);
        out.push(VectorPrimitive::Text {
            pos: [mid[0] - 0.5 * w, mid[1] - 4.5 * lw - 0.25 * sz],
            text: above.clone(),
            font_size_pt: sz,
            bold: false,
            italic: false,
            color: col,
        });
    }
    if let Some(below) = &arr.below_text {
        let w = text_advance_pt(below, sz);
        out.push(VectorPrimitive::Text {
            pos: [mid[0] - 0.5 * w, mid[1] + 5.0 * lw + 0.85 * sz],
            text: below.clone(),
            font_size_pt: sz,
            bold: false,
            italic: false,
            color: col,
        });
    }
}

fn emit_curved_arrow(out: &mut Vec<VectorPrimitive>, ca: &SceneCurvedArrow, style: &DocumentStyle) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let col = ca.color.unwrap_or(style.foreground);

    let s = [ca.start[0] * bl, -ca.start[1] * bl];
    let c1 = [ca.ctrl1[0] * bl, -ca.ctrl1[1] * bl];
    let c2 = [ca.ctrl2[0] * bl, -ca.ctrl2[1] * bl];
    let e = [ca.end[0] * bl, -ca.end[1] * bl];

    // End tangent vector at `t = 1` of cubic Bezier (`3*(e - c2)`, fallback to `e - s`)
    let mut tx = e[0] - c2[0];
    let mut ty = e[1] - c2[1];
    if tx.hypot(ty) < 1e-4 {
        tx = e[0] - s[0];
        ty = e[1] - s[1];
    }
    let tlen = tx.hypot(ty).max(1e-6);
    let ux = tx / tlen;
    let uy = ty / tlen;
    let nx = uy;
    let ny = -ux;

    let head_len = 5.4 * lw.max(0.55);
    let head_hw = 2.25 * lw.max(0.55);

    // Pull cubic Bezier endpoint slightly back into the arrowhead throat for a razor-sharp tip
    let setback = 0.68 * head_len;
    let stem_end = [e[0] - ux * setback, e[1] - uy * setback];

    out.push(VectorPrimitive::Path {
        cmds: vec![
            PathCmd::MoveTo(s[0], s[1]),
            PathCmd::CubicTo(c1[0], c1[1], c2[0], c2[1], stem_end[0], stem_end[1]),
        ],
        fill: None,
        stroke: Some((lw, col)),
        dash_pt: if ca.dashed {
            Some([2.8 * lw, 2.0 * lw])
        } else {
            None
        },
    });

    match ca.kind {
        PushingArrowKind::FullTwoElectron => {
            out.push(VectorPrimitive::Polygon {
                points: vec![
                    e,
                    [
                        e[0] - ux * head_len + nx * head_hw,
                        e[1] - uy * head_len + ny * head_hw,
                    ],
                    [e[0] - ux * (0.74 * head_len), e[1] - uy * (0.74 * head_len)],
                    [
                        e[0] - ux * head_len - nx * head_hw,
                        e[1] - uy * head_len - ny * head_hw,
                    ],
                ],
                fill: Some(col),
                stroke: None,
            });
        }
        PushingArrowKind::FishhookOneElectron => {
            let sgn = if ca.fishhook_left { 1.0 } else { -1.0 };
            out.push(VectorPrimitive::Polygon {
                points: vec![
                    e,
                    [
                        e[0] - ux * head_len + sgn * nx * (1.28 * head_hw),
                        e[1] - uy * head_len + sgn * ny * (1.28 * head_hw),
                    ],
                    [e[0] - ux * (0.74 * head_len), e[1] - uy * (0.74 * head_len)],
                ],
                fill: Some(col),
                stroke: None,
            });
        }
    }
}

fn emit_bracket(out: &mut Vec<VectorPrimitive>, br: &SceneBracket, style: &DocumentStyle) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let col = br.color.unwrap_or(style.foreground);

    let x0 = br.min[0].min(br.max[0]) * bl;
    let x1 = br.min[0].max(br.max[0]) * bl;
    let y0 = -(br.min[1].max(br.max[1])) * bl;
    let y1 = -(br.min[1].min(br.max[1])) * bl;
    let h = (y1 - y0).max(4.0);
    let lip = (0.14 * h).clamp(2.5, 0.45 * bl);

    match br.style {
        BracketStyle::Square => {
            // Left `[`
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(x0 + lip, y0),
                    PathCmd::LineTo(x0, y0),
                    PathCmd::LineTo(x0, y1),
                    PathCmd::LineTo(x0 + lip, y1),
                ],
                fill: None,
                stroke: Some((lw, col)),
                dash_pt: None,
            });
            // Right `]`
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(x1 - lip, y0),
                    PathCmd::LineTo(x1, y0),
                    PathCmd::LineTo(x1, y1),
                    PathCmd::LineTo(x1 - lip, y1),
                ],
                fill: None,
                stroke: Some((lw, col)),
                dash_pt: None,
            });
        }
        BracketStyle::Parenthesis => {
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(x0 + lip, y0),
                    PathCmd::CubicTo(
                        x0 - 0.35 * lip,
                        y0 + 0.25 * h,
                        x0 - 0.35 * lip,
                        y1 - 0.25 * h,
                        x0 + lip,
                        y1,
                    ),
                ],
                fill: None,
                stroke: Some((lw, col)),
                dash_pt: None,
            });
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(x1 - lip, y0),
                    PathCmd::CubicTo(
                        x1 + 0.35 * lip,
                        y0 + 0.25 * h,
                        x1 + 0.35 * lip,
                        y1 - 0.25 * h,
                        x1 - lip,
                        y1,
                    ),
                ],
                fill: None,
                stroke: Some((lw, col)),
                dash_pt: None,
            });
        }
        BracketStyle::CurlyBrace => {
            let ym = 0.5 * (y0 + y1);
            for (bx, sgn) in [(x0, 1.0), (x1, -1.0)] {
                out.push(VectorPrimitive::Path {
                    cmds: vec![
                        PathCmd::MoveTo(bx + sgn * lip, y0),
                        PathCmd::CubicTo(
                            bx,
                            y0,
                            bx + sgn * 0.4 * lip,
                            ym - 0.1 * h,
                            bx - sgn * 0.4 * lip,
                            ym,
                        ),
                        PathCmd::CubicTo(
                            bx + sgn * 0.4 * lip,
                            ym + 0.1 * h,
                            bx,
                            y1,
                            bx + sgn * lip,
                            y1,
                        ),
                    ],
                    fill: None,
                    stroke: Some((lw, col)),
                    dash_pt: None,
                });
            }
        }
    }

    if let Some(sub) = &br.subscript {
        out.push(VectorPrimitive::Text {
            pos: [x1 + 1.8 * lw, y1 + 0.30 * style.subscript_size_pt()],
            text: sub.clone(),
            font_size_pt: style.subscript_size_pt(),
            bold: false,
            italic: sub.chars().all(|c| c.is_ascii_alphabetic()),
            color: col,
        });
    }
    if let Some(sup) = &br.superscript {
        out.push(VectorPrimitive::Text {
            pos: [x1 + 1.8 * lw, y0 + 0.45 * style.superscript_size_pt()],
            text: sup.clone(),
            font_size_pt: style.superscript_size_pt(),
            bold: false,
            italic: false,
            color: col,
        });
    }
}

fn emit_orbital(out: &mut Vec<VectorPrimitive>, orb: &SceneOrbital, style: &DocumentStyle) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let cx = orb.center[0] * bl;
    let cy = -orb.center[1] * bl;
    let size = (orb.size_mol * bl).max(4.0);
    let stroke_col = orb.color.unwrap_or(style.foreground);

    let (pos_fill, neg_fill) = match orb.shading {
        OrbitalShading::ShadedAndOpen => (
            Some(RgbaColor::rgb(0x9c, 0xa3, 0xaf)),
            Some(RgbaColor::WHITE),
        ),
        OrbitalShading::DualPhaseColored => (
            Some(RgbaColor::rgb(0x93, 0xc5, 0xfd)),
            Some(RgbaColor::rgb(0xfc, 0xa5, 0xa5)),
        ),
        OrbitalShading::OutlineOnly => (None, None),
    };

    // Helper: teardrop lobe starting at `(cx, cy)` pointing along screen angle `ang_rad`
    let emit_lobe =
        |out: &mut Vec<VectorPrimitive>, len: f64, ang_rad: f64, fill: Option<RgbaColor>| {
            let ux = ang_rad.cos();
            let uy = ang_rad.sin();
            let nx = -uy;
            let ny = ux;
            let w = 0.56 * len;
            let tip = [cx + len * ux, cy + len * uy];
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(cx, cy),
                    PathCmd::CubicTo(
                        cx + 0.35 * len * ux + w * nx,
                        cy + 0.35 * len * uy + w * ny,
                        cx + 0.95 * len * ux + 0.65 * w * nx,
                        cy + 0.95 * len * uy + 0.65 * w * ny,
                        tip[0],
                        tip[1],
                    ),
                    PathCmd::CubicTo(
                        cx + 0.95 * len * ux - 0.65 * w * nx,
                        cy + 0.95 * len * uy - 0.65 * w * ny,
                        cx + 0.35 * len * ux - w * nx,
                        cy + 0.35 * len * uy - w * ny,
                        cx,
                        cy,
                    ),
                    PathCmd::Close,
                ],
                fill,
                stroke: Some((lw, stroke_col)),
                dash_pt: None,
            });
        };

    // Screen angle (`+y` down -> negate molecular `angle_deg`)
    let ang = -(orb.angle_deg.to_radians());

    match orb.orbital_type {
        OrbitalType::S => {
            out.push(VectorPrimitive::Circle {
                center: [cx, cy],
                radius_pt: 0.65 * size,
                fill: pos_fill,
                stroke: Some((lw, stroke_col)),
                dash_pt: None,
            });
        }
        OrbitalType::P => {
            emit_lobe(out, size, ang, pos_fill);
            emit_lobe(out, size, ang + PI, neg_fill);
        }
        OrbitalType::DClover => {
            emit_lobe(out, size, ang, pos_fill);
            emit_lobe(out, size, ang + PI, pos_fill);
            emit_lobe(out, size, ang + 0.5 * PI, neg_fill);
            emit_lobe(out, size, ang + 1.5 * PI, neg_fill);
        }
        OrbitalType::Dz2 => {
            emit_lobe(out, size, ang, pos_fill);
            emit_lobe(out, size, ang + PI, pos_fill);
            // Equatorial torus collar perpendicular to lobe axis
            let px = -(ang.sin());
            let py = ang.cos();
            let rx = 0.46 * size;
            let ry = 0.16 * size;
            let ux = ang.cos();
            let uy = ang.sin();
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(cx + rx * px, cy + rx * py),
                    PathCmd::CubicTo(
                        cx + rx * px + ry * ux,
                        cy + rx * py + ry * uy,
                        cx - rx * px + ry * ux,
                        cy - rx * py + ry * uy,
                        cx - rx * px,
                        cy - rx * py,
                    ),
                    PathCmd::CubicTo(
                        cx - rx * px - ry * ux,
                        cy - rx * py - ry * uy,
                        cx + rx * px - ry * ux,
                        cy + rx * py - ry * uy,
                        cx + rx * px,
                        cy + rx * py,
                    ),
                    PathCmd::Close,
                ],
                fill: neg_fill,
                stroke: Some((lw, stroke_col)),
                dash_pt: None,
            });
        }
        OrbitalType::HybridSp3 => {
            emit_lobe(out, 0.36 * size, ang + PI, neg_fill);
            emit_lobe(out, 1.08 * size, ang, pos_fill);
        }
    }
}

fn emit_tlc_plate(out: &mut Vec<VectorPrimitive>, tlc: &SceneTlcPlate, style: &DocumentStyle) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let x0 = tlc.origin[0] * bl;
    let y1 = -tlc.origin[1] * bl; // bottom in screen points
    let w = (tlc.width_mol * bl).max(20.0);
    let h = (tlc.height_mol * bl).max(30.0);
    let x1 = x0 + w;
    let y0 = y1 - h; // top in screen points

    // Silica plate rectangle
    out.push(VectorPrimitive::Polygon {
        points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        fill: Some(RgbaColor::WHITE),
        stroke: Some((lw, style.foreground)),
    });

    let base_y = y1 - tlc.baseline_frac.clamp(0.05, 0.35) * h;
    let front_y = y1 - tlc.solvent_front_frac.clamp(0.65, 0.96) * h;
    let run_span = (base_y - front_y).max(1.0);

    // Baseline (pencil origin) and solvent front dashed lines
    for y_line in [base_y, front_y] {
        out.push(VectorPrimitive::Line {
            p1: [x0 + 2.0 * lw, y_line],
            p2: [x1 - 2.0 * lw, y_line],
            width_pt: 0.85 * lw,
            color: RgbaColor::rgb(0x4b, 0x55, 0x63),
            dash_pt: Some([2.6 * lw, 2.0 * lw]),
        });
    }

    let n_lanes = tlc
        .lane_labels
        .len()
        .max(tlc.spots.iter().map(|s| s.lane + 1).max().unwrap_or(1));
    let lane_x = |lane: usize| -> f64 { x0 + (lane as f64 + 0.5) / (n_lanes as f64) * w };

    let sz = (0.78 * style.caption_size_pt).max(5.5);
    for lane in 0..n_lanes {
        let lx = lane_x(lane);
        // Origin tick
        out.push(VectorPrimitive::Line {
            p1: [lx, base_y - 1.6 * lw],
            p2: [lx, base_y + 1.6 * lw],
            width_pt: lw,
            color: style.foreground,
            dash_pt: None,
        });
        if let Some(lbl) = tlc.lane_labels.get(lane) {
            if !lbl.is_empty() {
                let tw = text_advance_pt(lbl, sz);
                out.push(VectorPrimitive::Text {
                    pos: [lx - 0.5 * tw, y1 + 1.15 * sz],
                    text: lbl.clone(),
                    font_size_pt: sz,
                    bold: false,
                    italic: false,
                    color: style.foreground,
                });
            }
        }
    }

    for spot in &tlc.spots {
        let lx = lane_x(spot.lane.min(n_lanes.saturating_sub(1)));
        let sy = base_y - spot.rf.clamp(0.0, 1.0) * run_span;
        let rx = (spot.radius_mol * bl).max(1.8);
        let ry = rx * spot.tailing.clamp(0.6, 3.5);
        let col = spot.color.unwrap_or(style.foreground);

        let (fill, stroke, dash) = match spot.style {
            TlcSpotStyle::Solid => (Some(col), Some((0.8 * lw, col)), None),
            TlcSpotStyle::Hollow => (Some(RgbaColor::WHITE), Some((lw, col)), None),
            TlcSpotStyle::UvActive => (
                Some(RgbaColor::rgb(0xd8, 0xb4, 0xfe)),
                Some((lw, RgbaColor::rgb(0x6b, 0x21, 0xa8))),
                None,
            ),
            TlcSpotStyle::Faint => (None, Some((0.85 * lw, col)), Some([1.8 * lw, 1.6 * lw])),
        };

        if (ry - rx).abs() < 0.15 {
            out.push(VectorPrimitive::Circle {
                center: [lx, sy],
                radius_pt: rx,
                fill,
                stroke,
                dash_pt: dash,
            });
        } else {
            let k = 0.55228475;
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(lx + rx, sy),
                    PathCmd::CubicTo(lx + rx, sy + k * ry, lx + k * rx, sy + ry, lx, sy + ry),
                    PathCmd::CubicTo(lx - k * rx, sy + ry, lx - rx, sy + k * ry, lx - rx, sy),
                    PathCmd::CubicTo(lx - rx, sy - k * ry, lx - k * rx, sy - ry, lx, sy - ry),
                    PathCmd::CubicTo(lx + k * rx, sy - ry, lx + rx, sy - k * ry, lx + rx, sy),
                    PathCmd::Close,
                ],
                fill,
                stroke,
                dash_pt: dash,
            });
        }

        if let Some(lbl) = &spot.label {
            out.push(VectorPrimitive::Text {
                pos: [lx + rx + 2.0, sy + 0.35 * sz],
                text: lbl.clone(),
                font_size_pt: 0.9 * sz,
                bold: false,
                italic: false,
                color: col,
            });
        }
    }

    if tlc.show_rf_scale {
        for rf_tick in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let ty = base_y - rf_tick * run_span;
            out.push(VectorPrimitive::Line {
                p1: [x1, ty],
                p2: [x1 + 2.5 * lw, ty],
                width_pt: 0.8 * lw,
                color: style.foreground,
                dash_pt: None,
            });
            out.push(VectorPrimitive::Text {
                pos: [x1 + 3.5 * lw, ty + 0.35 * sz],
                text: format!("{:.2}", rf_tick),
                font_size_pt: 0.85 * sz,
                bold: false,
                italic: false,
                color: RgbaColor::rgb(0x4b, 0x55, 0x63),
            });
        }
    }
}

/// Synthesize discrete multiplet sub-lines `(delta_ppm_offset_hz, rel_amplitude)` from `couplings_hz`.
pub fn expand_multiplet_lines_hz(multiplicity: &str, couplings_hz: &[f64]) -> Vec<(f64, f64)> {
    let mut lines = vec![(0.0_f64, 1.0_f64)];
    let split_once = |cur: Vec<(f64, f64)>, j_hz: f64| -> Vec<(f64, f64)> {
        let mut next = Vec::with_capacity(cur.len() * 2);
        for (pos, amp) in cur {
            next.push((pos - 0.5 * j_hz, 0.5 * amp));
            next.push((pos + 0.5 * j_hz, 0.5 * amp));
        }
        next
    };

    if !couplings_hz.is_empty() {
        // Explicit J list: interpret `d`/`t`/`q`/`dd`/`dt`/`td`/`sept` or apply each J
        let m = multiplicity.trim().to_ascii_lowercase();
        let j0 = couplings_hz[0].abs().max(1.5);
        match m.as_str() {
            "s" | "br s" | "br" => {}
            "d" => lines = split_once(lines, j0),
            "t" => {
                lines = split_once(split_once(lines, j0), j0);
            }
            "q" => {
                for _ in 0..3 {
                    lines = split_once(lines, j0);
                }
            }
            "quint" | "p" => {
                for _ in 0..4 {
                    lines = split_once(lines, j0);
                }
            }
            "sext" => {
                for _ in 0..5 {
                    lines = split_once(lines, j0);
                }
            }
            "sept" | "hept" => {
                for _ in 0..6 {
                    lines = split_once(lines, j0);
                }
            }
            "dd" => {
                let j1 = couplings_hz.get(1).copied().unwrap_or(0.45 * j0).abs();
                lines = split_once(split_once(lines, j0), j1);
            }
            "dt" => {
                let j1 = couplings_hz.get(1).copied().unwrap_or(0.5 * j0).abs();
                lines = split_once(split_once(split_once(lines, j0), j1), j1);
            }
            "td" => {
                let j1 = couplings_hz.get(1).copied().unwrap_or(0.4 * j0).abs();
                lines = split_once(split_once(split_once(lines, j0), j0), j1);
            }
            _ => {
                for &j in couplings_hz.iter().take(6) {
                    lines = split_once(lines, j.abs());
                }
            }
        }
    } else {
        // Default textbook coupling (`7.0 Hz`) from multiplicity string alone
        let j0 = 7.0;
        match multiplicity.trim().to_ascii_lowercase().as_str() {
            "d" => lines = split_once(lines, j0),
            "t" => lines = split_once(split_once(lines, j0), j0),
            "q" => {
                for _ in 0..3 {
                    lines = split_once(lines, j0);
                }
            }
            "dd" => lines = split_once(split_once(lines, 8.2), 3.6),
            "m" => {
                for &j in &[7.2, 5.4, 3.1] {
                    lines = split_once(lines, j);
                }
            }
            _ => {}
        }
    }
    lines
}

fn emit_spectrum_1d(out: &mut Vec<VectorPrimitive>, spec: &SceneSpectrum1D, style: &DocumentStyle) {
    let bl = style.bond_length_pt;
    let lw = style.line_width_pt;
    let x0 = spec.origin[0] * bl;
    let y1 = -spec.origin[1] * bl; // bottom frame
    let w = (spec.width_mol * bl).max(60.0);
    let h = (spec.height_mol * bl).max(36.0);
    let x1 = x0 + w;
    let y0 = y1 - h;

    let ppm_hi = spec.ppm_max.max(spec.ppm_min + 0.5);
    let ppm_lo = spec.ppm_min;
    let ppm_span = ppm_hi - ppm_lo;

    // Subtle white plot card + bottom chemical shift axis (IUPAC NMR: high ppm on LEFT, low ppm on RIGHT)
    out.push(VectorPrimitive::Polygon {
        points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        fill: Some(RgbaColor::WHITE),
        stroke: Some((0.75 * lw, RgbaColor::rgb(0xd1, 0xd5, 0xdb))),
    });

    let axis_y = y1 - 12.0;
    let baseline_y = axis_y - 4.0;
    let peak_top_y = y0 + 13.0;
    let max_peak_h = (baseline_y - peak_top_y).max(10.0);

    let ppm_to_x = |ppm: f64| -> f64 {
        let u = ((ppm_hi - ppm) / ppm_span).clamp(0.0, 1.0);
        x0 + 6.0 + u * (w - 12.0)
    };

    // Axis line + ppm ticks
    out.push(VectorPrimitive::Line {
        p1: [x0 + 4.0, axis_y],
        p2: [x1 - 4.0, axis_y],
        width_pt: lw,
        color: style.foreground,
        dash_pt: None,
    });

    let major_step = if ppm_span > 120.0 {
        20.0
    } else if ppm_span > 30.0 {
        10.0
    } else if ppm_span > 6.0 {
        1.0
    } else {
        0.5
    };
    let tick_sz = (0.72 * style.caption_size_pt).max(5.2);
    let mut tick_ppm = (ppm_lo / major_step).ceil() * major_step;
    while tick_ppm <= ppm_hi + 1e-6 {
        let tx = ppm_to_x(tick_ppm);
        out.push(VectorPrimitive::Line {
            p1: [tx, axis_y],
            p2: [tx, axis_y + 2.6],
            width_pt: 0.85 * lw,
            color: style.foreground,
            dash_pt: None,
        });
        let lbl = if major_step >= 1.0 {
            format!("{:.0}", tick_ppm)
        } else {
            format!("{:.1}", tick_ppm)
        };
        let tw = text_advance_pt(&lbl, tick_sz);
        out.push(VectorPrimitive::Text {
            pos: [tx - 0.5 * tw, y1 - 2.2],
            text: lbl,
            font_size_pt: tick_sz,
            bold: false,
            italic: false,
            color: style.foreground,
        });
        tick_ppm += major_step;
    }

    // Synthesize or plot 1D spectral trace
    let freq = spec.spectrometer_mhz.max(60.0);
    let gamma_ppm = (0.5 * spec.linewidth_hz.max(0.6) / freq).max(ppm_span * 0.0008);

    let samples: Vec<[f64; 2]> = if !spec.custom_trace.is_empty() {
        spec.custom_trace.clone()
    } else {
        let n_pts = 520_usize;
        let mut sublines: Vec<(f64, f64)> = Vec::new();
        for sig in &spec.signals {
            let mult_hz = expand_multiplet_lines_hz(&sig.multiplicity, &sig.couplings_hz);
            let scale = sig.integral.max(0.35);
            for (dhz, amp) in mult_hz {
                sublines.push((sig.ppm + dhz / freq, amp * scale));
            }
        }
        let mut raw = Vec::with_capacity(n_pts);
        let mut ymax = 1e-9_f64;
        for k in 0..n_pts {
            let ppm = ppm_hi - (k as f64 / (n_pts - 1) as f64) * ppm_span;
            let mut inten = 0.0;
            for &(l_ppm, l_amp) in &sublines {
                let d = (ppm - l_ppm) / gamma_ppm;
                if d.abs() < 40.0 {
                    inten += l_amp / (1.0 + d * d);
                }
            }
            ymax = ymax.max(inten);
            raw.push([ppm, inten]);
        }
        for p in &mut raw {
            p[1] /= ymax;
        }
        raw
    };

    if samples.len() >= 2 {
        let mut cmds = Vec::with_capacity(samples.len());
        for (k, pt) in samples.iter().enumerate() {
            let sx = ppm_to_x(pt[0]);
            let sy = baseline_y - pt[1].clamp(-0.05, 1.0) * max_peak_h;
            if k == 0 {
                cmds.push(PathCmd::MoveTo(sx, sy));
            } else {
                cmds.push(PathCmd::LineTo(sx, sy));
            }
        }
        out.push(VectorPrimitive::Path {
            cmds,
            fill: None,
            stroke: Some((0.9 * lw, RgbaColor::rgb(0x11, 0x18, 0x27))),
            dash_pt: None,
        });
    }

    // Peak shift labels & integral curves
    for sig in &spec.signals {
        if sig.ppm < ppm_lo || sig.ppm > ppm_hi {
            continue;
        }
        let sx = ppm_to_x(sig.ppm);
        if spec.show_peak_labels {
            let txt = match &sig.assignment_label {
                Some(a) => format!("{:.2} ({}, {})", sig.ppm, sig.multiplicity, a),
                None => format!("{:.2} ({})", sig.ppm, sig.multiplicity),
            };
            let tw = text_advance_pt(&txt, tick_sz);
            out.push(VectorPrimitive::Text {
                pos: [sx - 0.5 * tw, y0 + 9.5],
                text: txt,
                font_size_pt: 0.92 * tick_sz,
                bold: false,
                italic: false,
                color: RgbaColor::rgb(0x1d, 0x4e, 0xd8),
            });
        }
        if spec.show_integrals && sig.integral > 0.05 && spec.nucleus.contains('H') {
            let int_w = 5.5;
            let step_h = (sig.integral.min(6.0) / 6.0) * 0.28 * max_peak_h + 3.0;
            out.push(VectorPrimitive::Path {
                cmds: vec![
                    PathCmd::MoveTo(sx - int_w, baseline_y - 2.0),
                    PathCmd::CubicTo(
                        sx - 0.25 * int_w,
                        baseline_y - 2.0,
                        sx + 0.25 * int_w,
                        baseline_y - 2.0 - step_h,
                        sx + int_w,
                        baseline_y - 2.0 - step_h,
                    ),
                ],
                fill: None,
                stroke: Some((0.75 * lw, RgbaColor::rgb(0x05, 0x96, 0x69))),
                dash_pt: None,
            });
            let ilbl = format!("{:.0}H", sig.integral.round().max(1.0));
            out.push(VectorPrimitive::Text {
                pos: [
                    sx - 0.5 * text_advance_pt(&ilbl, 0.85 * tick_sz),
                    axis_y - 0.8,
                ],
                text: ilbl,
                font_size_pt: 0.85 * tick_sz,
                bold: false,
                italic: false,
                color: RgbaColor::rgb(0x04, 0x78, 0x57),
            });
        }
    }

    if let Some(title) = &spec.title {
        out.push(VectorPrimitive::Text {
            pos: [x0 + 5.0, y0 + 9.0],
            text: title.clone(),
            font_size_pt: tick_sz,
            bold: true,
            italic: false,
            color: style.foreground,
        });
    }
}

fn normalize_compiled_scene(
    mut prims: Vec<VectorPrimitive>,
    atom_labels: Vec<Option<FormattedAtomLabel>>,
    style: &DocumentStyle,
) -> CompiledScene {
    if prims.is_empty() {
        let s = 2.0 * style.export_padding_pt.max(8.0) + style.bond_length_pt;
        return CompiledScene {
            width_pt: s,
            height_pt: s,
            shift_x_pt: 0.5 * s,
            shift_y_pt: 0.5 * s,
            atom_labels,
            primitives: prims,
        };
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut expand = |x: f64, y: f64, pad: f64| {
        min_x = min_x.min(x - pad);
        min_y = min_y.min(y - pad);
        max_x = max_x.max(x + pad);
        max_y = max_y.max(y + pad);
    };

    for pr in &prims {
        match pr {
            VectorPrimitive::Line {
                p1, p2, width_pt, ..
            } => {
                expand(p1[0], p1[1], 0.6 * width_pt);
                expand(p2[0], p2[1], 0.6 * width_pt);
            }
            VectorPrimitive::Polygon { points, stroke, .. } => {
                let pad = stroke.map(|(w, _)| 0.6 * w).unwrap_or(0.0);
                for p in points {
                    expand(p[0], p[1], pad);
                }
            }
            VectorPrimitive::Path { cmds, stroke, .. } => {
                let pad = stroke.map(|(w, _)| 0.6 * w).unwrap_or(0.0);
                for c in cmds {
                    match *c {
                        PathCmd::MoveTo(x, y) | PathCmd::LineTo(x, y) => expand(x, y, pad),
                        PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                            expand(x1, y1, pad);
                            expand(x2, y2, pad);
                            expand(x, y, pad);
                        }
                        PathCmd::Close => {}
                    }
                }
            }
            VectorPrimitive::Circle {
                center,
                radius_pt,
                stroke,
                ..
            } => {
                let pad = radius_pt + stroke.map(|(w, _)| 0.6 * w).unwrap_or(0.0);
                expand(center[0], center[1], pad);
            }
            VectorPrimitive::Text {
                pos,
                text,
                font_size_pt,
                ..
            } => {
                let w = text_advance_pt(text, *font_size_pt);
                expand(pos[0], pos[1] - 0.85 * font_size_pt, 0.5);
                expand(pos[0] + w, pos[1] + 0.28 * font_size_pt, 0.5);
            }
        }
    }

    let pad = style.export_padding_pt.max(4.0);
    let shift_x = -min_x + pad;
    let shift_y = -min_y + pad;
    let width_pt = ((max_x - min_x) + 2.0 * pad).max(16.0);
    let height_pt = ((max_y - min_y) + 2.0 * pad).max(16.0);

    for pr in &mut prims {
        match pr {
            VectorPrimitive::Line { p1, p2, .. } => {
                p1[0] += shift_x;
                p1[1] += shift_y;
                p2[0] += shift_x;
                p2[1] += shift_y;
            }
            VectorPrimitive::Polygon { points, .. } => {
                for p in points {
                    p[0] += shift_x;
                    p[1] += shift_y;
                }
            }
            VectorPrimitive::Path { cmds, .. } => {
                for c in cmds {
                    match c {
                        PathCmd::MoveTo(x, y) | PathCmd::LineTo(x, y) => {
                            *x += shift_x;
                            *y += shift_y;
                        }
                        PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                            *x1 += shift_x;
                            *y1 += shift_y;
                            *x2 += shift_x;
                            *y2 += shift_y;
                            *x += shift_x;
                            *y += shift_y;
                        }
                        PathCmd::Close => {}
                    }
                }
            }
            VectorPrimitive::Circle { center, .. } => {
                center[0] += shift_x;
                center[1] += shift_y;
            }
            VectorPrimitive::Text { pos, .. } => {
                pos[0] += shift_x;
                pos[1] += shift_y;
            }
        }
    }

    CompiledScene {
        width_pt,
        height_pt,
        shift_x_pt: shift_x,
        shift_y_pt: shift_y,
        atom_labels,
        primitives: prims,
    }
}

// ============================================================================
// 4. Publication SVG 1.1 Exporter
// ============================================================================

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render a [`RenderScene`] into a standalone publication-grade **SVG 1.1** document string.
pub fn render_scene_svg(scene: &RenderScene, style: &DocumentStyle) -> String {
    let compiled = compile_scene_primitives(scene, style);
    let w = compiled.width_pt;
    let h = compiled.height_pt;

    let mut svg = String::with_capacity(4096 + compiled.primitives.len() * 120);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"{w:.2}pt\" height=\"{h:.2}pt\" viewBox=\"0 0 {w:.2} {h:.2}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n"
    ));

    if let Some(bg) = style.background {
        if bg.a > 0 {
            svg.push_str(&format!(
                "  <rect x=\"0\" y=\"0\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"{}\"/>\n",
                bg.to_svg_hex()
            ));
        }
    }

    for pr in &compiled.primitives {
        match pr {
            VectorPrimitive::Line {
                p1,
                p2,
                width_pt,
                color,
                dash_pt,
            } => {
                let dash_attr = match dash_pt {
                    Some([d1, d2]) => format!(" stroke-dasharray=\"{d1:.2},{d2:.2}\""),
                    None => String::new(),
                };
                svg.push_str(&format!(
                    "  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{}\" stroke-width=\"{:.2}\"{dash_attr}/>\n",
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1],
                    color.to_svg_hex(),
                    width_pt
                ));
            }
            VectorPrimitive::Polygon {
                points,
                fill,
                stroke,
            } => {
                let pts_str = points
                    .iter()
                    .map(|p| format!("{:.2},{:.2}", p[0], p[1]))
                    .collect::<Vec<_>>()
                    .join(" ");
                let fill_attr = match fill {
                    Some(c) if c.a > 0 => c.to_svg_hex(),
                    _ => "none".to_string(),
                };
                let stroke_attr = match stroke {
                    Some((sw, sc)) => {
                        format!(" stroke=\"{}\" stroke-width=\"{sw:.2}\"", sc.to_svg_hex())
                    }
                    None => " stroke=\"none\"".to_string(),
                };
                svg.push_str(&format!(
                    "  <polygon points=\"{pts_str}\" fill=\"{fill_attr}\"{stroke_attr}/>\n"
                ));
            }
            VectorPrimitive::Path {
                cmds,
                fill,
                stroke,
                dash_pt,
            } => {
                let mut d = String::new();
                for cmd in cmds {
                    match *cmd {
                        PathCmd::MoveTo(x, y) => d.push_str(&format!("M {x:.2} {y:.2} ")),
                        PathCmd::LineTo(x, y) => d.push_str(&format!("L {x:.2} {y:.2} ")),
                        PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                            d.push_str(&format!(
                                "C {x1:.2} {y1:.2} {x2:.2} {y2:.2} {x:.2} {y:.2} "
                            ));
                        }
                        PathCmd::Close => d.push_str("Z "),
                    }
                }
                let fill_attr = match fill {
                    Some(c) if c.a > 0 => c.to_svg_hex(),
                    _ => "none".to_string(),
                };
                let stroke_attr = match stroke {
                    Some((sw, sc)) => {
                        format!(" stroke=\"{}\" stroke-width=\"{sw:.2}\"", sc.to_svg_hex())
                    }
                    None => " stroke=\"none\"".to_string(),
                };
                let dash_attr = match dash_pt {
                    Some([d1, d2]) => format!(" stroke-dasharray=\"{d1:.2},{d2:.2}\""),
                    None => String::new(),
                };
                svg.push_str(&format!(
                    "  <path d=\"{}\" fill=\"{fill_attr}\"{stroke_attr}{dash_attr}/>\n",
                    d.trim()
                ));
            }
            VectorPrimitive::Circle {
                center,
                radius_pt,
                fill,
                stroke,
                dash_pt,
            } => {
                let fill_attr = match fill {
                    Some(c) if c.a > 0 => c.to_svg_hex(),
                    _ => "none".to_string(),
                };
                let stroke_attr = match stroke {
                    Some((sw, sc)) => {
                        format!(" stroke=\"{}\" stroke-width=\"{sw:.2}\"", sc.to_svg_hex())
                    }
                    None => " stroke=\"none\"".to_string(),
                };
                let dash_attr = match dash_pt {
                    Some([d1, d2]) => format!(" stroke-dasharray=\"{d1:.2},{d2:.2}\""),
                    None => String::new(),
                };
                svg.push_str(&format!(
                    "  <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{fill_attr}\"{stroke_attr}{dash_attr}/>\n",
                    center[0], center[1], radius_pt
                ));
            }
            VectorPrimitive::Text {
                pos,
                text,
                font_size_pt,
                bold,
                italic,
                color,
            } => {
                let weight = if *bold { " font-weight=\"bold\"" } else { "" };
                let fstyle = if *italic {
                    " font-style=\"italic\""
                } else {
                    ""
                };
                svg.push_str(&format!(
                    "  <text x=\"{:.2}\" y=\"{:.2}\" font-family=\"{}, Helvetica, Arial, sans-serif\" font-size=\"{:.2}\" fill=\"{}\"{weight}{fstyle}>{}</text>\n",
                    pos[0],
                    pos[1],
                    xml_escape(&style.atom_label_font),
                    font_size_pt,
                    color.to_svg_hex(),
                    xml_escape(text)
                ));
            }
        }
    }

    svg.push_str("</svg>\n");
    svg
}

// ============================================================================
// 5. Pure-Rust Vector PDF 1.4 Exporter (zero external dependencies)
// ============================================================================

/// Encode text into PDF WinAnsi literal string `(...)`, mapping common chemical symbols
/// (`−`, `–`, `•`, `°`, `±`, `‡`) into standard WinAnsi bytes and escaping `\`, `(`, `)`.
fn pdf_encode_literal(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 4);
    out.push(b'(');
    for ch in s.chars() {
        let byte: u8 = match ch {
            '\\' => {
                out.extend_from_slice(b"\\\\");
                continue;
            }
            '(' => {
                out.extend_from_slice(b"\\(");
                continue;
            }
            ')' => {
                out.extend_from_slice(b"\\)");
                continue;
            }
            '−' | '–' => 0x96, // WinAnsi en-dash / minus
            '—' => 0x97,
            '•' => 0x95, // WinAnsi bullet
            '‡' => 0x87, // WinAnsi double dagger (transition state)
            '†' => 0x86,
            '°' => 0xB0,
            '±' => 0xB1,
            'µ' | 'μ' => 0xB5,
            '·' => 0xB7,
            '¹' => 0xB9,
            '²' => 0xB2,
            '³' => 0xB3,
            'α' => b'a',
            'β' => b'b',
            'δ' => b'd',
            c if (0x20..=0x7E).contains(&(c as u32)) => c as u8,
            _ => b'?',
        };
        out.push(byte);
    }
    out.push(b')');
    out
}

/// Render a [`RenderScene`] into standalone vector **PDF 1.4** bytes (`%PDF-1.4`).
pub fn render_scene_pdf(scene: &RenderScene, style: &DocumentStyle) -> Vec<u8> {
    let compiled = compile_scene_primitives(scene, style);
    let w = compiled.width_pt;
    let h = compiled.height_pt;
    // PDF user space has `+y` UP from bottom-left `(0, 0)`!
    let py = |screen_y: f64| -> f64 { h - screen_y };

    let mut stream: Vec<u8> = Vec::with_capacity(4096 + compiled.primitives.len() * 110);
    stream.extend_from_slice(b"q\n1 J 1 j\n");

    if let Some(bg) = style.background {
        if bg.a > 0 {
            let [r, g, b] = bg.to_pdf_rgb();
            stream.extend_from_slice(
                format!("{r:.4} {g:.4} {b:.4} rg\n0 0 {w:.2} {h:.2} re f\n").as_bytes(),
            );
        }
    }

    for pr in &compiled.primitives {
        match pr {
            VectorPrimitive::Line {
                p1,
                p2,
                width_pt,
                color,
                dash_pt,
            } => {
                let [r, g, b] = color.to_pdf_rgb();
                let dash_cmd = match dash_pt {
                    Some([d1, d2]) => format!("[{d1:.2} {d2:.2}] 0 d\n"),
                    None => "[] 0 d\n".to_string(),
                };
                stream.extend_from_slice(
                    format!(
                        "{dash_cmd}{r:.4} {g:.4} {b:.4} RG\n{width_pt:.3} w\n{:.2} {:.2} m {:.2} {:.2} l S\n",
                        p1[0],
                        py(p1[1]),
                        p2[0],
                        py(p2[1])
                    )
                    .as_bytes(),
                );
            }
            VectorPrimitive::Polygon {
                points,
                fill,
                stroke,
            } => {
                if points.len() < 3 {
                    continue;
                }
                stream.extend_from_slice(b"[] 0 d\n");
                if let Some(fc) = fill {
                    let [r, g, b] = fc.to_pdf_rgb();
                    stream.extend_from_slice(format!("{r:.4} {g:.4} {b:.4} rg\n").as_bytes());
                }
                if let Some((sw, sc)) = stroke {
                    let [r, g, b] = sc.to_pdf_rgb();
                    stream.extend_from_slice(
                        format!("{r:.4} {g:.4} {b:.4} RG\n{sw:.3} w\n").as_bytes(),
                    );
                }
                stream.extend_from_slice(
                    format!("{:.2} {:.2} m\n", points[0][0], py(points[0][1])).as_bytes(),
                );
                for pt in &points[1..] {
                    stream
                        .extend_from_slice(format!("{:.2} {:.2} l\n", pt[0], py(pt[1])).as_bytes());
                }
                let op = match (fill.is_some(), stroke.is_some()) {
                    (true, true) => "h B\n",
                    (true, false) => "h f\n",
                    (false, true) => "h S\n",
                    (false, false) => "n\n",
                };
                stream.extend_from_slice(op.as_bytes());
            }
            VectorPrimitive::Path {
                cmds,
                fill,
                stroke,
                dash_pt,
            } => {
                if cmds.is_empty() {
                    continue;
                }
                let dash_cmd = match dash_pt {
                    Some([d1, d2]) => format!("[{d1:.2} {d2:.2}] 0 d\n"),
                    None => "[] 0 d\n".to_string(),
                };
                stream.extend_from_slice(dash_cmd.as_bytes());
                if let Some(fc) = fill {
                    let [r, g, b] = fc.to_pdf_rgb();
                    stream.extend_from_slice(format!("{r:.4} {g:.4} {b:.4} rg\n").as_bytes());
                }
                if let Some((sw, sc)) = stroke {
                    let [r, g, b] = sc.to_pdf_rgb();
                    stream.extend_from_slice(
                        format!("{r:.4} {g:.4} {b:.4} RG\n{sw:.3} w\n").as_bytes(),
                    );
                }
                let mut closed = false;
                for cmd in cmds {
                    match *cmd {
                        PathCmd::MoveTo(x, y) => {
                            stream.extend_from_slice(format!("{x:.2} {:.2} m\n", py(y)).as_bytes());
                        }
                        PathCmd::LineTo(x, y) => {
                            stream.extend_from_slice(format!("{x:.2} {:.2} l\n", py(y)).as_bytes());
                        }
                        PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                            stream.extend_from_slice(
                                format!(
                                    "{x1:.2} {:.2} {x2:.2} {:.2} {x:.2} {:.2} c\n",
                                    py(y1),
                                    py(y2),
                                    py(y)
                                )
                                .as_bytes(),
                            );
                        }
                        PathCmd::Close => {
                            stream.extend_from_slice(b"h\n");
                            closed = true;
                        }
                    }
                }
                let op = match (fill.is_some(), stroke.is_some(), closed) {
                    (true, true, _) => "B\n",
                    (true, false, _) => "f\n",
                    (false, true, _) => "S\n",
                    (false, false, _) => "n\n",
                };
                stream.extend_from_slice(op.as_bytes());
            }
            VectorPrimitive::Circle {
                center,
                radius_pt,
                fill,
                stroke,
                dash_pt,
            } => {
                let k = 0.55228475 * radius_pt;
                let cx = center[0];
                let cy = py(center[1]);
                let r = *radius_pt;
                let dash_cmd = match dash_pt {
                    Some([d1, d2]) => format!("[{d1:.2} {d2:.2}] 0 d\n"),
                    None => "[] 0 d\n".to_string(),
                };
                stream.extend_from_slice(dash_cmd.as_bytes());
                if let Some(fc) = fill {
                    let [cr, cg, cb] = fc.to_pdf_rgb();
                    stream.extend_from_slice(format!("{cr:.4} {cg:.4} {cb:.4} rg\n").as_bytes());
                }
                if let Some((sw, sc)) = stroke {
                    let [cr, cg, cb] = sc.to_pdf_rgb();
                    stream.extend_from_slice(
                        format!("{cr:.4} {cg:.4} {cb:.4} RG\n{sw:.3} w\n").as_bytes(),
                    );
                }
                stream.extend_from_slice(
                    format!(
                        "{:.2} {:.2} m\n\
                         {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c\n\
                         {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c\n\
                         {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c\n\
                         {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} c\nh\n",
                        cx + r,
                        cy,
                        cx + r,
                        cy + k,
                        cx + k,
                        cy + r,
                        cx,
                        cy + r,
                        cx - k,
                        cy + r,
                        cx - r,
                        cy + k,
                        cx - r,
                        cy,
                        cx - r,
                        cy - k,
                        cx - k,
                        cy - r,
                        cx,
                        cy - r,
                        cx + k,
                        cy - r,
                        cx + r,
                        cy - k,
                        cx + r,
                        cy
                    )
                    .as_bytes(),
                );
                let op = match (fill.is_some(), stroke.is_some()) {
                    (true, true) => "B\n",
                    (true, false) => "f\n",
                    (false, true) => "S\n",
                    (false, false) => "n\n",
                };
                stream.extend_from_slice(op.as_bytes());
            }
            VectorPrimitive::Text {
                pos,
                text,
                font_size_pt,
                bold,
                italic,
                color,
            } => {
                let font_tag = match (*bold, *italic) {
                    (false, false) => "/F1",
                    (true, false) => "/F2",
                    (false, true) => "/F3",
                    (true, true) => "/F4",
                };
                let [r, g, b] = color.to_pdf_rgb();
                stream.extend_from_slice(
                    format!(
                        "BT\n{r:.4} {g:.4} {b:.4} rg\n{font_tag} {font_size_pt:.2} Tf\n{:.2} {:.2} Td\n",
                        pos[0],
                        py(pos[1])
                    )
                    .as_bytes(),
                );
                stream.extend_from_slice(&pdf_encode_literal(text));
                stream.extend_from_slice(b" Tj\nET\n");
            }
        }
    }

    stream.extend_from_slice(b"Q\n");

    // Assemble standalone PDF 1.4 file with exact byte-offset xref table
    let mut pdf: Vec<u8> = Vec::with_capacity(stream.len() + 1024);
    pdf.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let mut offsets = Vec::with_capacity(8);
    let mut push_obj = |pdf: &mut Vec<u8>, id: usize, body: &[u8]| {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    };

    // 1: Catalog
    push_obj(&mut pdf, 1, b"<< /Type /Catalog /Pages 2 0 R >>");
    // 2: Pages
    push_obj(&mut pdf, 2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    // 3: Page
    let page_dict = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.2} {h:.2}] \
         /Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> >> \
         /Contents 8 0 R >>"
    );
    push_obj(&mut pdf, 3, page_dict.as_bytes());
    // 4..7: Standard Type1 PDF Fonts (WinAnsiEncoding)
    push_obj(
        &mut pdf,
        4,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    push_obj(
        &mut pdf,
        5,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    );
    push_obj(
        &mut pdf,
        6,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>",
    );
    push_obj(
        &mut pdf,
        7,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-BoldOblique /Encoding /WinAnsiEncoding >>",
    );

    // 8: Content Stream
    offsets.push(pdf.len());
    pdf.extend_from_slice(format!("8 0 obj\n<< /Length {} >>\nstream\n", stream.len()).as_bytes());
    pdf.extend_from_slice(&stream);
    pdf.extend_from_slice(b"endstream\nendobj\n");

    let xref_pos = pdf.len();
    let total_objs = offsets.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {total_objs}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n")
            .as_bytes(),
    );

    pdf
}

// ============================================================================
// 6. Pure-Rust Anti-Aliased Rasterizer + Lossless PNG & PPM Exporters
// ============================================================================

struct SoftCanvas {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

impl SoftCanvas {
    fn new(w: usize, h: usize, bg: RgbaColor) -> Self {
        let mut rgba = vec![0_u8; w * h * 4];
        for px in rgba.chunks_exact_mut(4) {
            px[0] = bg.r;
            px[1] = bg.g;
            px[2] = bg.b;
            px[3] = bg.a;
        }
        Self { w, h, rgba }
    }

    #[inline]
    fn blend_pixel(&mut self, x: isize, y: isize, col: RgbaColor, coverage: f64) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return;
        }
        let alpha = ((col.a as f64 / 255.0) * coverage.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        if alpha <= 1e-4 {
            return;
        }
        let idx = (y as usize * self.w + x as usize) * 4;
        let dst_r = self.rgba[idx] as f64;
        let dst_g = self.rgba[idx + 1] as f64;
        let dst_b = self.rgba[idx + 2] as f64;
        let dst_a = self.rgba[idx + 3] as f64 / 255.0;

        let out_a = alpha + dst_a * (1.0 - alpha);
        if out_a > 1e-6 {
            let r = (col.r as f64 * alpha + dst_r * dst_a * (1.0 - alpha)) / out_a;
            let g = (col.g as f64 * alpha + dst_g * dst_a * (1.0 - alpha)) / out_a;
            let b = (col.b as f64 * alpha + dst_b * dst_a * (1.0 - alpha)) / out_a;
            self.rgba[idx] = r.round().clamp(0.0, 255.0) as u8;
            self.rgba[idx + 1] = g.round().clamp(0.0, 255.0) as u8;
            self.rgba[idx + 2] = b.round().clamp(0.0, 255.0) as u8;
            self.rgba[idx + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Draw an anti-aliased round-cap capsule segment from `p1` to `p2` of thickness `width_px`.
    fn draw_capsule(&mut self, p1: [f64; 2], p2: [f64; 2], width_px: f64, col: RgbaColor) {
        let r = (0.5 * width_px).max(0.55);
        let pad = r + 1.25;
        let min_x = (p1[0].min(p2[0]) - pad).floor() as isize;
        let max_x = (p1[0].max(p2[0]) + pad).ceil() as isize;
        let min_y = (p1[1].min(p2[1]) - pad).floor() as isize;
        let max_y = (p1[1].max(p2[1]) + pad).ceil() as isize;

        let abx = p2[0] - p1[0];
        let aby = p2[1] - p1[1];
        let ab2 = abx * abx + aby * aby;

        for py in min_y.max(0)..=max_y.min(self.h as isize - 1) {
            let cy = py as f64 + 0.5;
            for px in min_x.max(0)..=max_x.min(self.w as isize - 1) {
                let cx = px as f64 + 0.5;
                let t = if ab2 > 1e-9 {
                    (((cx - p1[0]) * abx + (cy - p1[1]) * aby) / ab2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let qx = p1[0] + t * abx;
                let qy = p1[1] + t * aby;
                let dist = (cx - qx).hypot(cy - qy);
                let cov = (r + 0.5 - dist).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, col, cov);
                }
            }
        }
    }

    fn draw_dashed_line(
        &mut self,
        p1: [f64; 2],
        p2: [f64; 2],
        width_px: f64,
        col: RgbaColor,
        dash: Option<[f64; 2]>,
    ) {
        let Some([d_on, d_off]) = dash else {
            self.draw_capsule(p1, p2, width_px, col);
            return;
        };
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let total = dx.hypot(dy);
        if total < 1e-4 {
            return;
        }
        let ux = dx / total;
        let uy = dy / total;
        let period = (d_on + d_off).max(1.0);
        let mut s = 0.0;
        while s < total {
            let e = (s + d_on).min(total);
            self.draw_capsule(
                [p1[0] + s * ux, p1[1] + s * uy],
                [p1[0] + e * ux, p1[1] + e * uy],
                width_px,
                col,
            );
            s += period;
        }
    }

    /// Scanline fill a polygon (even-odd / non-zero rule).
    fn fill_polygon(&mut self, pts: &[[f64; 2]], col: RgbaColor) {
        if pts.len() < 3 {
            return;
        }
        let min_y = pts
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
            .floor() as isize;
        let max_y = pts
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as isize;

        let mut xs = Vec::with_capacity(pts.len());
        for py in min_y.max(0)..=max_y.min(self.h as isize - 1) {
            let scan_y = py as f64 + 0.5;
            xs.clear();
            for k in 0..pts.len() {
                let a = pts[k];
                let b = pts[(k + 1) % pts.len()];
                if (a[1] <= scan_y && b[1] > scan_y) || (b[1] <= scan_y && a[1] > scan_y) {
                    let t = (scan_y - a[1]) / (b[1] - a[1]);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            for pair in xs.chunks_exact(2) {
                let x_start = pair[0].floor() as isize;
                let x_end = pair[1].ceil() as isize;
                for px in x_start.max(0)..=x_end.min(self.w as isize - 1) {
                    let cx = px as f64 + 0.5;
                    let left_cov = (cx - pair[0] + 0.5).clamp(0.0, 1.0);
                    let right_cov = (pair[1] - cx + 0.5).clamp(0.0, 1.0);
                    self.blend_pixel(px, py, col, left_cov.min(right_cov));
                }
            }
        }
    }

    fn draw_circle(
        &mut self,
        center: [f64; 2],
        r_px: f64,
        fill: Option<RgbaColor>,
        stroke: Option<(f64, RgbaColor)>,
    ) {
        let sw = stroke.map(|(w, _)| w).unwrap_or(0.0);
        let outer = r_px + 0.5 * sw + 1.2;
        let min_x = (center[0] - outer).floor() as isize;
        let max_x = (center[0] + outer).ceil() as isize;
        let min_y = (center[1] - outer).floor() as isize;
        let max_y = (center[1] + outer).ceil() as isize;

        for py in min_y.max(0)..=max_y.min(self.h as isize - 1) {
            let cy = py as f64 + 0.5;
            for px in min_x.max(0)..=max_x.min(self.w as isize - 1) {
                let cx = px as f64 + 0.5;
                let d = (cx - center[0]).hypot(cy - center[1]);
                if let Some(fc) = fill {
                    let cov = (r_px + 0.5 - d).clamp(0.0, 1.0);
                    if cov > 0.0 {
                        self.blend_pixel(px, py, fc, cov);
                    }
                }
                if let Some((w, sc)) = stroke {
                    let half_w = 0.5 * w.max(1.0);
                    let cov = (half_w + 0.5 - (d - r_px).abs()).clamp(0.0, 1.0);
                    if cov > 0.0 {
                        self.blend_pixel(px, py, sc, cov);
                    }
                }
            }
        }
    }

    /// Crisp built-in vector stroke glyph renderer (`[0..1, 0..1]` glyph box -> capsule strokes)
    /// so chemical element symbols, subscripts, numbers, and +/- charges look sharp at any DPI!
    fn draw_vector_text(
        &mut self,
        origin_px: [f64; 2],
        text: &str,
        font_px: f64,
        bold: bool,
        italic: bool,
        col: RgbaColor,
    ) {
        let cap_h = 0.72 * font_px;
        let stroke_w = (if bold { 0.125 } else { 0.092 } * font_px).max(1.05);
        let slant = if italic { 0.18 } else { 0.0 };
        let mut cursor_x = origin_px[0];
        let base_y = origin_px[1];

        for ch in text.chars() {
            let adv = char_advance_em(ch) * font_px;
            let gw = (adv * 0.74).max(1.5);
            let x0 = cursor_x + 0.13 * adv;
            let map_pt = |u: f64, v: f64| -> [f64; 2] {
                // u in [0, 1] left-to-right, v in [0, 1] bottom-to-top (0 = baseline, 1 = cap top)
                [x0 + u * gw + slant * v * cap_h, base_y - v * cap_h]
            };

            for seg in glyph_strokes(ch) {
                self.draw_capsule(
                    map_pt(seg[0], seg[1]),
                    map_pt(seg[2], seg[3]),
                    stroke_w,
                    col,
                );
            }
            cursor_x += adv;
        }
    }
}

/// Normalized `[u1, v1, u2, v2]` stroke segments (`v = 0` baseline, `v = 1` cap height) for ASCII + chem glyphs.
fn glyph_strokes(ch: char) -> &'static [[f64; 4]] {
    match ch {
        'A' => &[
            [0.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 1.0, 0.0],
            [0.2, 0.35, 0.8, 0.35],
        ],
        'B' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.75, 1.0],
            [0.75, 1.0, 0.95, 0.78],
            [0.95, 0.78, 0.75, 0.52],
            [0.0, 0.52, 0.75, 0.52],
            [0.75, 0.52, 1.0, 0.25],
            [1.0, 0.25, 0.75, 0.0],
            [0.75, 0.0, 0.0, 0.0],
        ],
        'C' => &[
            [0.95, 0.82, 0.65, 1.0],
            [0.65, 1.0, 0.25, 1.0],
            [0.25, 1.0, 0.0, 0.72],
            [0.0, 0.72, 0.0, 0.28],
            [0.0, 0.28, 0.25, 0.0],
            [0.25, 0.0, 0.65, 0.0],
            [0.65, 0.0, 0.95, 0.18],
        ],
        'D' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.65, 1.0],
            [0.65, 1.0, 1.0, 0.70],
            [1.0, 0.70, 1.0, 0.30],
            [1.0, 0.30, 0.65, 0.0],
            [0.65, 0.0, 0.0, 0.0],
        ],
        'E' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.95, 1.0],
            [0.0, 0.52, 0.80, 0.52],
            [0.0, 0.0, 0.95, 0.0],
        ],
        'F' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.95, 1.0],
            [0.0, 0.52, 0.78, 0.52],
        ],
        'G' => &[
            [0.95, 0.82, 0.65, 1.0],
            [0.65, 1.0, 0.25, 1.0],
            [0.25, 1.0, 0.0, 0.72],
            [0.0, 0.72, 0.0, 0.28],
            [0.0, 0.28, 0.25, 0.0],
            [0.25, 0.0, 0.85, 0.0],
            [0.85, 0.0, 0.95, 0.48],
            [0.52, 0.48, 0.95, 0.48],
        ],
        'H' => &[
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 0.52, 1.0, 0.52],
        ],
        'I' => &[
            [0.2, 1.0, 0.8, 1.0],
            [0.5, 0.0, 0.5, 1.0],
            [0.2, 0.0, 0.8, 0.0],
        ],
        'J' => &[
            [0.85, 1.0, 0.85, 0.25],
            [0.85, 0.25, 0.55, 0.0],
            [0.55, 0.0, 0.20, 0.0],
            [0.20, 0.0, 0.0, 0.22],
        ],
        'K' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.95, 1.0, 0.0, 0.45],
            [0.28, 0.60, 1.0, 0.0],
        ],
        'L' => &[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.92, 0.0]],
        'M' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.5, 0.15],
            [0.5, 0.15, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        'N' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        'O' => &[
            [0.25, 1.0, 0.75, 1.0],
            [0.75, 1.0, 1.0, 0.72],
            [1.0, 0.72, 1.0, 0.28],
            [1.0, 0.28, 0.75, 0.0],
            [0.75, 0.0, 0.25, 0.0],
            [0.25, 0.0, 0.0, 0.28],
            [0.0, 0.28, 0.0, 0.72],
            [0.0, 0.72, 0.25, 1.0],
        ],
        'P' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.75, 1.0],
            [0.75, 1.0, 1.0, 0.75],
            [1.0, 0.75, 0.75, 0.50],
            [0.75, 0.50, 0.0, 0.50],
        ],
        'Q' => &[
            [0.25, 1.0, 0.75, 1.0],
            [0.75, 1.0, 1.0, 0.72],
            [1.0, 0.72, 1.0, 0.28],
            [1.0, 0.28, 0.75, 0.0],
            [0.75, 0.0, 0.25, 0.0],
            [0.25, 0.0, 0.0, 0.28],
            [0.0, 0.28, 0.0, 0.72],
            [0.0, 0.72, 0.25, 1.0],
            [0.55, 0.28, 1.05, -0.12],
        ],
        'R' => &[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.75, 1.0],
            [0.75, 1.0, 1.0, 0.75],
            [1.0, 0.75, 0.75, 0.50],
            [0.75, 0.50, 0.0, 0.50],
            [0.52, 0.50, 1.0, 0.0],
        ],
        'S' => &[
            [0.92, 0.85, 0.68, 1.0],
            [0.68, 1.0, 0.22, 1.0],
            [0.22, 1.0, 0.02, 0.78],
            [0.02, 0.78, 0.22, 0.54],
            [0.22, 0.54, 0.78, 0.46],
            [0.78, 0.46, 0.98, 0.22],
            [0.98, 0.22, 0.75, 0.0],
            [0.75, 0.0, 0.22, 0.0],
            [0.22, 0.0, 0.02, 0.16],
        ],
        'T' => &[[0.0, 1.0, 1.0, 1.0], [0.5, 0.0, 0.5, 1.0]],
        'U' => &[
            [0.0, 1.0, 0.0, 0.26],
            [0.0, 0.26, 0.28, 0.0],
            [0.28, 0.0, 0.72, 0.0],
            [0.72, 0.0, 1.0, 0.26],
            [1.0, 0.26, 1.0, 1.0],
        ],
        'V' => &[[0.0, 1.0, 0.5, 0.0], [0.5, 0.0, 1.0, 1.0]],
        'W' => &[
            [0.0, 1.0, 0.24, 0.0],
            [0.24, 0.0, 0.5, 0.72],
            [0.5, 0.72, 0.76, 0.0],
            [0.76, 0.0, 1.0, 1.0],
        ],
        'X' => &[[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
        'Y' => &[
            [0.0, 1.0, 0.5, 0.48],
            [1.0, 1.0, 0.5, 0.48],
            [0.5, 0.48, 0.5, 0.0],
        ],
        'Z' => &[
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        'a' => &[
            [0.15, 0.62, 0.75, 0.68],
            [0.75, 0.68, 0.90, 0.50],
            [0.90, 0.50, 0.90, 0.0],
            [0.90, 0.36, 0.25, 0.36],
            [0.25, 0.36, 0.05, 0.18],
            [0.05, 0.18, 0.25, 0.0],
            [0.25, 0.0, 0.90, 0.0],
        ],
        'b' => &[
            [0.08, 1.0, 0.08, 0.0],
            [0.08, 0.58, 0.68, 0.66],
            [0.68, 0.66, 0.95, 0.34],
            [0.95, 0.34, 0.68, 0.0],
            [0.68, 0.0, 0.08, 0.0],
        ],
        'c' => &[
            [0.90, 0.58, 0.30, 0.66],
            [0.30, 0.66, 0.05, 0.34],
            [0.05, 0.34, 0.30, 0.0],
            [0.30, 0.0, 0.90, 0.08],
        ],
        'd' => &[
            [0.92, 1.0, 0.92, 0.0],
            [0.92, 0.58, 0.32, 0.66],
            [0.32, 0.66, 0.05, 0.34],
            [0.05, 0.34, 0.32, 0.0],
            [0.32, 0.0, 0.92, 0.0],
        ],
        'e' => &[
            [0.05, 0.34, 0.92, 0.34],
            [0.92, 0.34, 0.72, 0.66],
            [0.72, 0.66, 0.28, 0.66],
            [0.28, 0.66, 0.05, 0.34],
            [0.05, 0.34, 0.28, 0.0],
            [0.28, 0.0, 0.85, 0.06],
        ],
        'f' => &[
            [0.85, 0.98, 0.50, 1.0],
            [0.50, 1.0, 0.35, 0.82],
            [0.35, 0.82, 0.35, 0.0],
            [0.08, 0.64, 0.75, 0.64],
        ],
        'g' => &[
            [0.90, 0.66, 0.28, 0.66],
            [0.28, 0.66, 0.05, 0.34],
            [0.05, 0.34, 0.28, 0.04],
            [0.28, 0.04, 0.90, 0.04],
            [0.90, 0.66, 0.90, -0.22],
            [0.90, -0.22, 0.25, -0.26],
        ],
        'h' => &[
            [0.08, 1.0, 0.08, 0.0],
            [0.08, 0.55, 0.65, 0.66],
            [0.65, 0.66, 0.92, 0.48],
            [0.92, 0.48, 0.92, 0.0],
        ],
        'i' => &[[0.5, 0.0, 0.5, 0.66], [0.5, 0.88, 0.5, 0.96]],
        'k' => &[
            [0.1, 1.0, 0.1, 0.0],
            [0.88, 0.66, 0.1, 0.30],
            [0.35, 0.42, 0.92, 0.0],
        ],
        'l' => &[[0.5, 1.0, 0.5, 0.0]],
        'm' => &[
            [0.0, 0.0, 0.0, 0.66],
            [0.0, 0.60, 0.5, 0.66],
            [0.5, 0.66, 0.5, 0.0],
            [0.5, 0.60, 1.0, 0.66],
            [1.0, 0.66, 1.0, 0.0],
        ],
        'n' => &[
            [0.08, 0.0, 0.08, 0.66],
            [0.08, 0.56, 0.68, 0.66],
            [0.68, 0.66, 0.92, 0.48],
            [0.92, 0.48, 0.92, 0.0],
        ],
        'o' => &[
            [0.28, 0.66, 0.72, 0.66],
            [0.72, 0.66, 0.95, 0.33],
            [0.95, 0.33, 0.72, 0.0],
            [0.72, 0.0, 0.28, 0.0],
            [0.28, 0.0, 0.05, 0.33],
            [0.05, 0.33, 0.28, 0.66],
        ],
        'p' => &[
            [0.08, -0.26, 0.08, 0.66],
            [0.08, 0.66, 0.72, 0.66],
            [0.72, 0.66, 0.95, 0.34],
            [0.95, 0.34, 0.72, 0.0],
            [0.72, 0.0, 0.08, 0.0],
        ],
        'q' => &[
            [0.92, -0.26, 0.92, 0.66],
            [0.92, 0.66, 0.28, 0.66],
            [0.28, 0.66, 0.05, 0.34],
            [0.05, 0.34, 0.28, 0.0],
            [0.28, 0.0, 0.92, 0.0],
        ],
        'r' => &[
            [0.12, 0.0, 0.12, 0.66],
            [0.12, 0.48, 0.55, 0.66],
            [0.55, 0.66, 0.88, 0.62],
        ],
        's' => &[
            [0.88, 0.58, 0.30, 0.66],
            [0.30, 0.66, 0.08, 0.48],
            [0.08, 0.48, 0.90, 0.22],
            [0.90, 0.22, 0.65, 0.0],
            [0.65, 0.0, 0.10, 0.08],
        ],
        't' => &[
            [0.42, 0.92, 0.42, 0.08],
            [0.42, 0.08, 0.78, 0.0],
            [0.10, 0.66, 0.82, 0.66],
        ],
        'u' => &[
            [0.08, 0.66, 0.08, 0.18],
            [0.08, 0.18, 0.35, 0.0],
            [0.35, 0.0, 0.90, 0.0],
            [0.90, 0.66, 0.90, 0.0],
        ],
        'v' => &[[0.05, 0.66, 0.5, 0.0], [0.5, 0.0, 0.95, 0.66]],
        'x' => &[[0.08, 0.0, 0.92, 0.66], [0.08, 0.66, 0.92, 0.0]],
        'y' => &[[0.05, 0.66, 0.5, 0.0], [0.95, 0.66, 0.22, -0.26]],
        'z' => &[
            [0.08, 0.66, 0.92, 0.66],
            [0.92, 0.66, 0.08, 0.0],
            [0.08, 0.0, 0.92, 0.0],
        ],
        '0' => &[
            [0.25, 1.0, 0.75, 1.0],
            [0.75, 1.0, 0.96, 0.72],
            [0.96, 0.72, 0.96, 0.28],
            [0.96, 0.28, 0.75, 0.0],
            [0.75, 0.0, 0.25, 0.0],
            [0.25, 0.0, 0.04, 0.28],
            [0.04, 0.28, 0.04, 0.72],
            [0.04, 0.72, 0.25, 1.0],
        ],
        '1' => &[
            [0.20, 0.78, 0.55, 1.0],
            [0.55, 1.0, 0.55, 0.0],
            [0.20, 0.0, 0.88, 0.0],
        ],
        '2' => &[
            [0.08, 0.78, 0.32, 1.0],
            [0.32, 1.0, 0.75, 1.0],
            [0.75, 1.0, 0.95, 0.75],
            [0.95, 0.75, 0.05, 0.0],
            [0.05, 0.0, 0.96, 0.0],
        ],
        '3' => &[
            [0.10, 0.85, 0.35, 1.0],
            [0.35, 1.0, 0.78, 1.0],
            [0.78, 1.0, 0.95, 0.75],
            [0.95, 0.75, 0.65, 0.52],
            [0.42, 0.52, 0.65, 0.52],
            [0.65, 0.52, 0.98, 0.25],
            [0.98, 0.25, 0.72, 0.0],
            [0.72, 0.0, 0.25, 0.0],
            [0.25, 0.0, 0.06, 0.16],
        ],
        '4' => &[
            [0.75, 0.0, 0.75, 1.0],
            [0.75, 1.0, 0.04, 0.32],
            [0.04, 0.32, 0.98, 0.32],
        ],
        '5' => &[
            [0.92, 1.0, 0.12, 1.0],
            [0.12, 1.0, 0.08, 0.54],
            [0.08, 0.54, 0.75, 0.58],
            [0.75, 0.58, 0.96, 0.28],
            [0.96, 0.28, 0.72, 0.0],
            [0.72, 0.0, 0.12, 0.06],
        ],
        '6' => &[
            [0.85, 0.92, 0.32, 1.0],
            [0.32, 1.0, 0.05, 0.60],
            [0.05, 0.60, 0.05, 0.22],
            [0.05, 0.22, 0.32, 0.0],
            [0.32, 0.0, 0.75, 0.0],
            [0.75, 0.0, 0.95, 0.28],
            [0.95, 0.28, 0.72, 0.54],
            [0.72, 0.54, 0.08, 0.48],
        ],
        '7' => &[[0.05, 1.0, 0.95, 1.0], [0.95, 1.0, 0.32, 0.0]],
        '8' => &[
            [0.28, 1.0, 0.72, 1.0],
            [0.72, 1.0, 0.92, 0.76],
            [0.92, 0.76, 0.72, 0.52],
            [0.28, 0.52, 0.72, 0.52],
            [0.28, 1.0, 0.08, 0.76],
            [0.08, 0.76, 0.28, 0.52],
            [0.72, 0.52, 0.96, 0.25],
            [0.96, 0.25, 0.72, 0.0],
            [0.72, 0.0, 0.28, 0.0],
            [0.28, 0.0, 0.04, 0.25],
            [0.04, 0.25, 0.28, 0.52],
        ],
        '9' => &[
            [0.92, 0.52, 0.28, 0.46],
            [0.28, 0.46, 0.05, 0.72],
            [0.05, 0.72, 0.28, 1.0],
            [0.28, 1.0, 0.72, 1.0],
            [0.72, 1.0, 0.94, 0.72],
            [0.94, 0.72, 0.94, 0.26],
            [0.94, 0.26, 0.65, 0.0],
            [0.65, 0.0, 0.18, 0.04],
        ],
        '+' => &[[0.12, 0.50, 0.88, 0.50], [0.50, 0.15, 0.50, 0.85]],
        '-' | '−' | '–' => &[[0.12, 0.50, 0.88, 0.50]],
        '=' => &[[0.12, 0.65, 0.88, 0.65], [0.12, 0.35, 0.88, 0.35]],
        '(' => &[
            [0.75, 1.05, 0.32, 0.68],
            [0.32, 0.68, 0.32, 0.22],
            [0.32, 0.22, 0.75, -0.12],
        ],
        ')' => &[
            [0.25, 1.05, 0.68, 0.68],
            [0.68, 0.68, 0.68, 0.22],
            [0.68, 0.22, 0.25, -0.12],
        ],
        '[' => &[
            [0.75, 1.05, 0.25, 1.05],
            [0.25, 1.05, 0.25, -0.10],
            [0.25, -0.10, 0.75, -0.10],
        ],
        ']' => &[
            [0.25, 1.05, 0.75, 1.05],
            [0.75, 1.05, 0.75, -0.10],
            [0.75, -0.10, 0.25, -0.10],
        ],
        '.' => &[[0.45, 0.0, 0.55, 0.08]],
        ',' => &[[0.55, 0.10, 0.35, -0.16]],
        ':' => &[[0.45, 0.62, 0.55, 0.68], [0.45, 0.08, 0.55, 0.14]],
        '/' => &[[0.10, -0.08, 0.90, 1.05]],
        '‡' => &[
            [0.50, 0.0, 0.50, 1.0],
            [0.15, 0.74, 0.85, 0.74],
            [0.15, 0.28, 0.85, 0.28],
        ],
        ' ' => &[],
        _ => &[
            [0.15, 0.15, 0.85, 0.15],
            [0.85, 0.15, 0.85, 0.85],
            [0.85, 0.85, 0.15, 0.85],
            [0.15, 0.85, 0.15, 0.15],
        ],
    }
}

/// Flatten a [`PathCmd`] list into dense polyline subpaths by subdividing cubic Beziers.
pub fn flatten_path_commands(
    cmds: &[PathCmd],
    steps_per_cubic: usize,
) -> Vec<(Vec<[f64; 2]>, bool)> {
    let mut subpaths: Vec<(Vec<[f64; 2]>, bool)> = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    let steps = steps_per_cubic.max(8);

    for cmd in cmds {
        match *cmd {
            PathCmd::MoveTo(x, y) => {
                if !cur.is_empty() {
                    subpaths.push((std::mem::take(&mut cur), false));
                }
                cur.push([x, y]);
            }
            PathCmd::LineTo(x, y) => {
                cur.push([x, y]);
            }
            PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                let p0 = cur.last().copied().unwrap_or([x1, y1]);
                for s in 1..=steps {
                    let t = s as f64 / steps as f64;
                    let mt = 1.0 - t;
                    let bx = mt * mt * mt * p0[0]
                        + 3.0 * mt * mt * t * x1
                        + 3.0 * mt * t * t * x2
                        + t * t * t * x;
                    let by = mt * mt * mt * p0[1]
                        + 3.0 * mt * mt * t * y1
                        + 3.0 * mt * t * t * y2
                        + t * t * t * y;
                    cur.push([bx, by]);
                }
            }
            PathCmd::Close => {
                if let Some(&first) = cur.first() {
                    cur.push(first);
                }
                subpaths.push((std::mem::take(&mut cur), true));
            }
        }
    }
    if !cur.is_empty() {
        subpaths.push((cur, false));
    }
    subpaths
}

fn rasterize_compiled_rgba(
    compiled: &CompiledScene,
    style: &DocumentStyle,
    dpi: f64,
) -> (usize, usize, Vec<u8>) {
    let scale = (dpi.clamp(36.0, 600.0)) / 72.0;
    let out_w = ((compiled.width_pt * scale).ceil() as usize).clamp(16, 4096);
    let out_h = ((compiled.height_pt * scale).ceil() as usize).clamp(16, 4096);

    let bg = style.background.unwrap_or(RgbaColor::WHITE);
    let mut canvas = SoftCanvas::new(out_w, out_h, bg);

    let sp = |p: [f64; 2]| -> [f64; 2] { [p[0] * scale, p[1] * scale] };

    for pr in &compiled.primitives {
        match pr {
            VectorPrimitive::Line {
                p1,
                p2,
                width_pt,
                color,
                dash_pt,
            } => {
                let dash_px = dash_pt.map(|[a, b]| [a * scale, b * scale]);
                canvas.draw_dashed_line(sp(*p1), sp(*p2), width_pt * scale, *color, dash_px);
            }
            VectorPrimitive::Polygon {
                points,
                fill,
                stroke,
            } => {
                let pts_px: Vec<[f64; 2]> = points.iter().map(|&p| sp(p)).collect();
                if let Some(fc) = fill {
                    canvas.fill_polygon(&pts_px, *fc);
                }
                if let Some((sw, sc)) = stroke {
                    for k in 0..pts_px.len() {
                        canvas.draw_capsule(
                            pts_px[k],
                            pts_px[(k + 1) % pts_px.len()],
                            sw * scale,
                            *sc,
                        );
                    }
                }
            }
            VectorPrimitive::Path {
                cmds,
                fill,
                stroke,
                dash_pt,
            } => {
                let subpaths = flatten_path_commands(cmds, 18);
                let dash_px = dash_pt.map(|[a, b]| [a * scale, b * scale]);
                for (pts, _closed) in subpaths {
                    let pts_px: Vec<[f64; 2]> = pts.iter().map(|&p| sp(p)).collect();
                    if let Some(fc) = fill {
                        canvas.fill_polygon(&pts_px, *fc);
                    }
                    if let Some((sw, sc)) = stroke {
                        for seg in pts_px.windows(2) {
                            canvas.draw_dashed_line(seg[0], seg[1], sw * scale, *sc, dash_px);
                        }
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
                canvas.draw_circle(
                    sp(*center),
                    radius_pt * scale,
                    *fill,
                    stroke.map(|(w, c)| (w * scale, c)),
                );
            }
            VectorPrimitive::Text {
                pos,
                text,
                font_size_pt,
                bold,
                italic,
                color,
            } => {
                canvas.draw_vector_text(
                    sp(*pos),
                    text,
                    font_size_pt * scale,
                    *bold,
                    *italic,
                    *color,
                );
            }
        }
    }

    (out_w, out_h, canvas.rgba)
}

/// IEEE 802.3 CRC-32 used by PNG chunks.
fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// RFC 1950 Adler-32 checksum used by zlib streams inside PNG `IDAT`.
fn zlib_adler32(bytes: &[u8]) -> u32 {
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in bytes {
        s1 = (s1 + b as u32) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    (s2 << 16) | s1
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = out.len();
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = png_crc32(&out[crc_start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode 8-bit RGBA pixels `(width, height)` into valid lossless **PNG** bytes (RFC 2083).
pub fn encode_rgba_png(width: usize, height: usize, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);

    // Filter type 0 (None) scanlines: 1 filter byte + 4 * width bytes per row
    let row_bytes = width * 4;
    let mut raw_scanlines = Vec::with_capacity(height * (1 + row_bytes));
    for y in 0..height {
        raw_scanlines.push(0_u8); // Filter 0 (None)
        let start = y * row_bytes;
        raw_scanlines.extend_from_slice(&rgba[start..start + row_bytes]);
    }

    // Wrap in RFC 1950 zlib stream (`CMF = 0x78`, `FLG = 0x01`) using stored DEFLATE blocks (RFC 1951 BTYPE=00)
    let mut zlib = Vec::with_capacity(raw_scanlines.len() + 64);
    zlib.extend_from_slice(&[0x78, 0x01]);

    let max_block = 65_535_usize;
    let chunks: Vec<&[u8]> = if raw_scanlines.is_empty() {
        vec![&[]]
    } else {
        raw_scanlines.chunks(max_block).collect()
    };
    for (ci, block) in chunks.iter().enumerate() {
        let is_final = if ci + 1 == chunks.len() {
            0x01_u8
        } else {
            0x00_u8
        };
        zlib.push(is_final);
        let len = block.len() as u16;
        let nlen = !len;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&nlen.to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&zlib_adler32(&raw_scanlines).to_be_bytes());

    let mut png = Vec::with_capacity(zlib.len() + 128);
    // 8-byte PNG file signature
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    // IHDR: width (4B), height (4B), bit_depth=8, color_type=6 (RGBA), comp=0, filter=0, interlace=0
    let mut ihdr = [0_u8; 13];
    ihdr[0..4].copy_from_slice(&(width as u32).to_be_bytes());
    ihdr[4..8].copy_from_slice(&(height as u32).to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 6;
    write_png_chunk(&mut png, b"IHDR", &ihdr);
    write_png_chunk(&mut png, b"IDAT", &zlib);
    write_png_chunk(&mut png, b"IEND", &[]);
    png
}

/// Render a [`RenderScene`] into lossless anti-aliased **PNG** bytes at `dpi` (`144.0` or `300.0` recommended).
pub fn render_scene_png(scene: &RenderScene, style: &DocumentStyle, dpi: f64) -> Vec<u8> {
    let compiled = compile_scene_primitives(scene, style);
    let (w, h, rgba) = rasterize_compiled_rgba(&compiled, style, dpi);
    encode_rgba_png(w, h, &rgba)
}

/// Render a [`RenderScene`] into lossless **PNG** bytes at default `144.0` Retina DPI (`2 px/pt`).
pub fn render_scene_png_default(scene: &RenderScene, style: &DocumentStyle) -> Vec<u8> {
    render_scene_png(scene, style, 144.0)
}

/// Render a [`RenderScene`] into binary **PPM (`P6`)** raster bytes at `dpi`.
pub fn render_scene_ppm(scene: &RenderScene, style: &DocumentStyle, dpi: f64) -> Vec<u8> {
    let compiled = compile_scene_primitives(scene, style);
    let (w, h, rgba) = rasterize_compiled_rgba(&compiled, style, dpi);
    let header = format!("P6\n{w} {h}\n255\n");
    let mut ppm = Vec::with_capacity(header.len() + w * h * 3);
    ppm.extend_from_slice(header.as_bytes());
    for px in rgba.chunks_exact(4) {
        ppm.push(px[0]);
        ppm.push(px[1]);
        ppm.push(px[2]);
    }
    ppm
}
