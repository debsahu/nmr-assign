//! Publication journal style sheets, color palettes, and chemical label typography.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis (`docstyle.ts` and `render.ts` label metrics).
//!
//! Provides exact publication document style presets (`ACS Document 1996`, `RSC`, `JACS`,
//! `Wiley`, `Angewandte`, `Nature`, `Presentation`), page dimensions, CPK element palettes,
//! highlight palettes, hash-wedge spacing & bar counts, wedge widths, double-bond offsets,
//! margin clearances around heteroatom labels, and direction-aware chemical formula
//! subscript/superscript layout (`NH`/`HO` left vs `OH` right vs stacked `NH`).

use serde::{Deserialize, Serialize};

/// Standard publication journal style presets matching publisher author guidelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum JournalStyle {
    /// ACS Document 1996 (`14.4 pt` bond, `0.6 pt` line, `2.0 pt` bold, `10 pt` Helvetica).
    #[default]
    Acs1996,
    /// Royal Society of Chemistry (`12.2 pt` bond, `0.45 pt` line, `1.6 pt` bold, `7 pt` Arial).
    Rsc,
    /// Journal of the American Chemical Society (`14.4 pt` bond, `0.6 pt` line, `18%` spacing).
    Jacs,
    /// Wiley (`17.0 pt` bond, `0.75 pt` line, `2.6 pt` bold, `12 pt` Arial).
    Wiley,
    /// Angewandte Chemie (`17.0 pt` bond, `0.75 pt` line, `2.6 pt` bold, `12 pt` bold labels).
    Angewandte,
    /// Nature Chemistry / Nature Portfolio (`10.8 pt` bond, `0.6 pt` line, `6 pt` Arial).
    Nature,
    /// Slide / Poster Presentation (`30.0 pt` bond, `1.6 pt` line, `4.0 pt` bold, `15%` spacing).
    Presentation,
}

impl JournalStyle {
    /// All built-in journal style presets in UI display order.
    pub const ALL: &'static [JournalStyle] = &[
        JournalStyle::Acs1996,
        JournalStyle::Rsc,
        JournalStyle::Jacs,
        JournalStyle::Wiley,
        JournalStyle::Angewandte,
        JournalStyle::Nature,
        JournalStyle::Presentation,
    ];

    /// Human-readable display name matching ChemDraw / Octet-Draw style sheets.
    pub fn name(self) -> &'static str {
        match self {
            Self::Acs1996 => "ACS Document 1996",
            Self::Rsc => "RSC",
            Self::Jacs => "JACS",
            Self::Wiley => "Wiley",
            Self::Angewandte => "Angewandte",
            Self::Nature => "Nature",
            Self::Presentation => "Presentation",
        }
    }

    /// Short identifier for CLI / serialization.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Acs1996 => "acs1996",
            Self::Rsc => "rsc",
            Self::Jacs => "jacs",
            Self::Wiley => "wiley",
            Self::Angewandte => "angewandte",
            Self::Nature => "nature",
            Self::Presentation => "presentation",
        }
    }

    /// Parse a style preset by name or slug (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        let norm = s.trim().to_ascii_lowercase();
        match norm.as_str() {
            "acs" | "acs 1996" | "acs1996" | "acs document 1996" => Some(Self::Acs1996),
            "rsc" | "royal society of chemistry" => Some(Self::Rsc),
            "jacs" => Some(Self::Jacs),
            "wiley" => Some(Self::Wiley),
            "angewandte" | "angew" | "angewandte chemie" => Some(Self::Angewandte),
            "nature" | "nature chemistry" => Some(Self::Nature),
            "presentation" | "slide" | "poster" => Some(Self::Presentation),
            _ => None,
        }
    }

    /// Construct the full [`DocumentStyle`] for this preset.
    pub fn to_style(self) -> DocumentStyle {
        DocumentStyle::from_preset(self)
    }
}

/// RGBA color value in 8-bit sRGB channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RgbaColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RgbaColor {
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const TRANSPARENT: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 0,
    };

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Parse `#RRGGBB` or `#RGB` hex color string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let s = hex.trim().strip_prefix('#').unwrap_or(hex.trim());
        match s.len() {
            6 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                Some(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                let a = u8::from_str_radix(&s[6..8], 16).ok()?;
                Some(Self::rgba(r, g, b, a))
            }
            3 => {
                let r = u8::from_str_radix(&s[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&s[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&s[2..3], 16).ok()? * 17;
                Some(Self::rgb(r, g, b))
            }
            _ => None,
        }
    }

    /// Format as CSS/SVG `#rrggbb` string.
    pub fn to_svg_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Convert to normalized `[0.0, 1.0]` RGB floats for PDF content streams.
    pub fn to_pdf_rgb(self) -> [f64; 3] {
        [
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
        ]
    }
}

/// Standard ink palette from Octet-Draw (`docstyle.ts` `COLOURS`).
pub const INK_PALETTE: [(&str, RgbaColor); 9] = [
    ("Black", RgbaColor::rgb(0x00, 0x00, 0x00)),
    ("Red", RgbaColor::rgb(0xc8, 0x1e, 0x1e)),
    ("Orange", RgbaColor::rgb(0xd9, 0x5f, 0x02)),
    ("Green", RgbaColor::rgb(0x1b, 0x7f, 0x3b)),
    ("Teal", RgbaColor::rgb(0x0d, 0x7c, 0x8c)),
    ("Blue", RgbaColor::rgb(0x1d, 0x4e, 0xd8)),
    ("Purple", RgbaColor::rgb(0x7c, 0x2d, 0x92)),
    ("Magenta", RgbaColor::rgb(0xb8, 0x1d, 0x6f)),
    ("Grey", RgbaColor::rgb(0x6b, 0x72, 0x80)),
];

/// Soft ring-fill and atom/bond halo highlight palette (`docstyle.ts` `HIGHLIGHT_COLOURS`).
pub const HIGHLIGHT_PALETTE: [(&str, RgbaColor); 8] = [
    ("Yellow", RgbaColor::rgb(0xfd, 0xe6, 0x8a)),
    ("Orange", RgbaColor::rgb(0xfd, 0xba, 0x74)),
    ("Pink", RgbaColor::rgb(0xf9, 0xa8, 0xd4)),
    ("Red", RgbaColor::rgb(0xfc, 0xa5, 0xa5)),
    ("Green", RgbaColor::rgb(0x86, 0xef, 0xac)),
    ("Cyan", RgbaColor::rgb(0x67, 0xe8, 0xf9)),
    ("Blue", RgbaColor::rgb(0x93, 0xc5, 0xfd)),
    ("Lavender", RgbaColor::rgb(0xc4, 0xb5, 0xfd)),
];

/// Publication CPK heteroatom color lookup (`docstyle.ts` / IUPAC CPK colors).
pub fn element_cpk_color(symbol: &str) -> RgbaColor {
    match symbol.trim() {
        "N" => RgbaColor::rgb(0x1d, 0x4e, 0xd8), // Deep nitrogen blue
        "O" => RgbaColor::rgb(0xdc, 0x26, 0x26), // Oxygen crimson
        "S" => RgbaColor::rgb(0xb4, 0x8a, 0x04), // Sulfur warm ochre (crisp on white)
        "P" => RgbaColor::rgb(0xd9, 0x5f, 0x02), // Phosphorus orange
        "F" => RgbaColor::rgb(0x16, 0xa3, 0x4a), // Fluorine green
        "Cl" => RgbaColor::rgb(0x15, 0x80, 0x3d), // Chlorine forest green
        "Br" => RgbaColor::rgb(0x99, 0x1b, 0x1b), // Bromine dark red-brown
        "I" => RgbaColor::rgb(0x6b, 0x21, 0xa8), // Iodine violet
        "B" => RgbaColor::rgb(0xb4, 0x53, 0x09), // Boron warm amber
        "Si" => RgbaColor::rgb(0x92, 0x40, 0x0e), // Silicon sienna
        "Fe" => RgbaColor::rgb(0xc2, 0x41, 0x0c), // Iron rust
        "Pd" | "Pt" | "Ru" | "Rh" | "Ir" | "Ni" => RgbaColor::rgb(0x0f, 0x76, 0x6e),
        "Mg" | "Ca" | "Zn" | "Cu" => RgbaColor::rgb(0x43, 0x38, 0xca),
        "Li" | "Na" | "K" => RgbaColor::rgb(0x7e, 0x22, 0xce),
        "H" | "D" | "T" => RgbaColor::rgb(0x37, 0x41, 0x51),
        _ => RgbaColor::BLACK,
    }
}

/// Standard document page dimensions in points (`1 pt = 1/72 in`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PageSetup {
    pub width_pt: f64,
    pub height_pt: f64,
    pub margin_pt: f64,
}

impl PageSetup {
    pub const US_LETTER: Self = Self {
        width_pt: 612.0,
        height_pt: 792.0,
        margin_pt: 36.0,
    };
    pub const A4: Self = Self {
        width_pt: 595.0,
        height_pt: 842.0,
        margin_pt: 36.0,
    };
    pub const SLIDE_16_9: Self = Self {
        width_pt: 960.0,
        height_pt: 540.0,
        margin_pt: 36.0,
    };
}

/// Complete publication style sheet governing 2D chemical rendering and export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentStyle {
    /// Active preset identifier.
    pub preset: JournalStyle,
    /// Fixed bond length in typographic points (`1 pt = 1/72 inch`).
    pub bond_length_pt: f64,
    /// Primary bond & frame stroke width in points.
    pub line_width_pt: f64,
    /// Bold bond width in points (wedge wide end is `1.5 * bold_width_pt`).
    pub bold_width_pt: f64,
    /// Centre-to-centre double-bond line spacing as `%` of `bond_length_pt`.
    pub double_bond_spacing_pct: f64,
    /// Additional terminal double-bond shortening in points (`0.0` for ACS full-length).
    pub terminal_double_inset_pt: f64,
    /// Additional ring-inside double-bond inset factor beyond angle-bisector trim.
    pub ring_double_inset_ratio: f64,
    /// Centre-to-centre spacing of hashed-wedge bars in points.
    pub hash_spacing_pt: f64,
    /// Clearance margin around atom labels in points.
    pub margin_width_pt: f64,
    /// Atom label font family.
    pub atom_label_font: String,
    /// Atom label font size in points.
    pub atom_label_size_pt: f64,
    /// Whether atom labels are rendered bold by default (e.g. Angewandte / Slide schemes).
    pub atom_label_bold: bool,
    /// Caption / condition / spectrum label font family.
    pub caption_font: String,
    /// Caption font size in points.
    pub caption_size_pt: f64,
    /// Default chain tool interior angle in degrees (`120.0`).
    pub chain_angle_deg: f64,
    /// Outer export canvas padding in points.
    pub export_padding_pt: f64,
    /// Optional CPK heteroatom coloring (off by default for monochrome journal print).
    pub color_by_element: bool,
    /// Primary ink color.
    pub foreground: RgbaColor,
    /// Optional solid page/canvas background (`None` = transparent SVG, white PNG/PDF).
    pub background: Option<RgbaColor>,
    /// Page setup for multi-structure / full-page PDF export.
    pub page: PageSetup,
}

impl Default for DocumentStyle {
    fn default() -> Self {
        Self::from_preset(JournalStyle::Acs1996)
    }
}

impl DocumentStyle {
    /// Build exact style parameters from a [`JournalStyle`] preset (`docstyle.ts`).
    pub fn from_preset(preset: JournalStyle) -> Self {
        match preset {
            JournalStyle::Acs1996 => Self {
                preset,
                bond_length_pt: 14.4,
                line_width_pt: 0.6,
                bold_width_pt: 2.0,
                double_bond_spacing_pct: 18.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 2.5,
                margin_width_pt: 1.6,
                atom_label_font: "Helvetica".to_string(),
                atom_label_size_pt: 10.0,
                atom_label_bold: false,
                caption_font: "Helvetica".to_string(),
                caption_size_pt: 10.0,
                chain_angle_deg: 120.0,
                export_padding_pt: 8.0,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::US_LETTER,
            },
            JournalStyle::Jacs => Self {
                // JACS uses ACS Document 1996 dimensions with slightly tighter single-column margin
                preset,
                bond_length_pt: 14.4,
                line_width_pt: 0.6,
                bold_width_pt: 2.0,
                double_bond_spacing_pct: 18.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 2.5,
                margin_width_pt: 1.6,
                atom_label_font: "Helvetica".to_string(),
                atom_label_size_pt: 10.0,
                atom_label_bold: false,
                caption_font: "Helvetica".to_string(),
                caption_size_pt: 9.5,
                chain_angle_deg: 120.0,
                export_padding_pt: 7.2,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::US_LETTER,
            },
            JournalStyle::Rsc => Self {
                preset,
                bond_length_pt: 12.2,
                line_width_pt: 0.45,
                bold_width_pt: 1.6,
                double_bond_spacing_pct: 20.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.09,
                hash_spacing_pt: 1.75,
                margin_width_pt: 1.25,
                atom_label_font: "Arial".to_string(),
                atom_label_size_pt: 7.0,
                atom_label_bold: false,
                caption_font: "Arial".to_string(),
                caption_size_pt: 7.0,
                chain_angle_deg: 120.0,
                export_padding_pt: 6.5,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::A4,
            },
            JournalStyle::Wiley => Self {
                preset,
                bond_length_pt: 17.0,
                line_width_pt: 0.75,
                bold_width_pt: 2.6,
                double_bond_spacing_pct: 18.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 2.6,
                margin_width_pt: 2.0,
                atom_label_font: "Arial".to_string(),
                atom_label_size_pt: 12.0,
                atom_label_bold: false,
                caption_font: "Arial".to_string(),
                caption_size_pt: 12.0,
                chain_angle_deg: 120.0,
                export_padding_pt: 8.5,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::A4,
            },
            JournalStyle::Angewandte => Self {
                preset,
                bond_length_pt: 17.0,
                line_width_pt: 0.80,
                bold_width_pt: 2.6,
                double_bond_spacing_pct: 18.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 2.6,
                margin_width_pt: 2.0,
                atom_label_font: "Arial".to_string(),
                atom_label_size_pt: 12.0,
                atom_label_bold: true,
                caption_font: "Arial".to_string(),
                caption_size_pt: 11.5,
                chain_angle_deg: 120.0,
                export_padding_pt: 8.5,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::A4,
            },
            JournalStyle::Nature => Self {
                preset,
                bond_length_pt: 10.8,
                line_width_pt: 0.6,
                bold_width_pt: 1.56,
                double_bond_spacing_pct: 18.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 1.7,
                margin_width_pt: 1.19,
                atom_label_font: "Arial".to_string(),
                atom_label_size_pt: 6.0,
                atom_label_bold: false,
                caption_font: "Arial".to_string(),
                caption_size_pt: 6.0,
                chain_angle_deg: 120.0,
                export_padding_pt: 6.0,
                color_by_element: false,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::A4,
            },
            JournalStyle::Presentation => Self {
                preset,
                bond_length_pt: 30.0,
                line_width_pt: 1.6,
                bold_width_pt: 4.0,
                double_bond_spacing_pct: 15.0,
                terminal_double_inset_pt: 0.0,
                ring_double_inset_ratio: 0.08,
                hash_spacing_pt: 2.7,
                margin_width_pt: 2.0,
                atom_label_font: "Arial".to_string(),
                atom_label_size_pt: 14.0,
                atom_label_bold: true,
                caption_font: "Arial".to_string(),
                caption_size_pt: 16.0,
                chain_angle_deg: 120.0,
                export_padding_pt: 14.0,
                color_by_element: true,
                foreground: RgbaColor::BLACK,
                background: Some(RgbaColor::WHITE),
                page: PageSetup::SLIDE_16_9,
            },
        }
    }

    /// Uniformly scale every point-valued dimension (`docstyle.ts` `withScale`).
    pub fn with_scale(&self, k: f64) -> Self {
        let mut s = self.clone();
        s.bond_length_pt *= k;
        s.line_width_pt *= k;
        s.bold_width_pt *= k;
        s.terminal_double_inset_pt *= k;
        s.hash_spacing_pt *= k;
        s.margin_width_pt *= k;
        s.atom_label_size_pt *= k;
        s.caption_size_pt *= k;
        s.export_padding_pt *= k;
        s
    }

    /// Centre-to-centre double-bond spacing in points (`spacing_pct / 100 * bond_length_pt`).
    pub fn double_bond_spacing_pt(&self) -> f64 {
        self.bond_length_pt * (self.double_bond_spacing_pct / 100.0)
    }

    /// Full width of the wide end of a solid or hashed stereo wedge (`1.5 * bold_width_pt`).
    pub fn wedge_wide_width_pt(&self) -> f64 {
        1.5 * self.bold_width_pt
    }

    /// Narrow tip width of a stereo wedge (`0.033 * bond_length_pt`, matching ACS 1996).
    pub fn wedge_tip_width_pt(&self) -> f64 {
        (0.033 * self.bond_length_pt).max(self.line_width_pt)
    }

    /// Number of discrete parallel bars in a hashed wedge of length `bond_len_pt`.
    pub fn hash_count(&self, bond_len_pt: f64) -> usize {
        if self.hash_spacing_pt <= 1e-6 {
            return 5;
        }
        let usable = (bond_len_pt - self.line_width_pt).max(0.0);
        ((usable / self.hash_spacing_pt).round() as usize).max(2) + 1
    }

    /// Margin clearance in points around atom glyphs where incoming bonds stop (`render.ts` `CLEAR`).
    pub fn label_clearance_pt(&self) -> f64 {
        self.margin_width_pt * (1.69 / 1.6)
    }

    /// Trim distance in points for an inside ring double-bond line meeting an adjacent ring bond
    /// at interior angle `theta_rad` (`spacing / tan(theta/2)` plus aesthetic inset).
    pub fn ring_inner_trim_pt(&self, theta_rad: f64) -> f64 {
        let half = (0.5 * theta_rad).clamp(0.25, std::f64::consts::FRAC_PI_2 - 0.05);
        let bisector_trim = self.double_bond_spacing_pt() / half.tan();
        bisector_trim + self.ring_double_inset_ratio * self.bond_length_pt
    }

    /// Resolve ink color for an atom symbol when `override_color` is `None`.
    pub fn color_for_element(&self, symbol: &str, override_color: Option<RgbaColor>) -> RgbaColor {
        if let Some(c) = override_color {
            return c;
        }
        if self.color_by_element {
            let cpk = element_cpk_color(symbol);
            if cpk != RgbaColor::BLACK {
                return cpk;
            }
        }
        self.foreground
    }

    /// Cap-height of the atom label font in points (`0.718 * em` for Helvetica/Arial).
    pub fn cap_height_pt(&self) -> f64 {
        0.718 * self.atom_label_size_pt
    }

    /// Subscript font size in points (`0.75 * em`).
    pub fn subscript_size_pt(&self) -> f64 {
        0.75 * self.atom_label_size_pt
    }

    /// Subscript downward baseline shift in points (`0.212 * em`).
    pub fn subscript_drop_pt(&self) -> f64 {
        0.212 * self.atom_label_size_pt
    }

    /// Superscript font size in points (`0.75 * em`).
    pub fn superscript_size_pt(&self) -> f64 {
        0.75 * self.atom_label_size_pt
    }

    /// Superscript upward baseline shift in points (`0.33 * em`).
    pub fn superscript_rise_pt(&self) -> f64 {
        0.33 * self.atom_label_size_pt
    }

    /// Stacked vertical atom-label pitch (`NH` stacked) in points (`1.167 * cap_height`).
    pub fn stacked_pitch_pt(&self) -> f64 {
        1.167 * self.cap_height_pt()
    }
}

// ============================================================================
// Implicit Valence & Chemical Formula / Heteroatom Label Typography
// ============================================================================

/// Elements that write hydrogen before the element symbol when unbonded (`H2O`, `HCl`, `H2S`).
pub fn is_h_first_element(el: &str) -> bool {
    matches!(el, "O" | "S" | "Se" | "Te" | "F" | "Cl" | "Br" | "I" | "At")
}

/// Standard neutral/charged implicit hydrogen count matching IUPAC / Octet-Draw (`render.ts` `hydrogens`).
///
/// `bond_order_sum` may be fractional (`1.5` per aromatic bond); it is rounded up to integer
/// valence electrons consumed.
pub fn implicit_hydrogens(
    element: &str,
    charge: i8,
    bond_order_sum: f64,
    radical_electrons: u8,
) -> u8 {
    let used = bond_order_sum.round() as i32 + radical_electrons as i32;
    let c = charge as i32;
    let target_valence: i32 = match element.trim() {
        "B" => 3 - c,
        "C" | "Si" | "Ge" => 4 - c.abs(),
        "N" | "P" | "As" => {
            if c > 0 {
                4 // e.g. NH4+
            } else if used <= 3 + c {
                3 + c
            } else if element == "P" && used <= 5 {
                5
            } else {
                0
            }
        }
        "O" | "S" | "Se" | "Te" => {
            if c > 0 {
                3 // e.g. H3O+
            } else if c < 0 {
                1 // e.g. O- has 0 H when 1 bond
            } else if used <= 2 {
                2
            } else {
                0
            }
        }
        "F" | "Cl" | "Br" | "I" => {
            if c == 0 && used <= 1 {
                1
            } else {
                0
            }
        }
        "H" | "D" | "T" if c == 0 && used == 0 => 1,
        _ => 0,
    };
    (target_valence - used).max(0) as u8
}

/// Reading/placement direction of a heteroatom or condensed-group label around its anchor atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelDirection {
    /// Anchor on first glyph, label extends right (`OH`, `NH2`, `COOH`).
    LeftToRight,
    /// Anchor on rightmost heteroatom glyph, H/prefix extends left (`HO`, `H2N`, `HOOC`).
    RightToLeft,
    /// Vertically stacked (`H` above `N` or `H` below `N`) at steep ring/chain apices.
    StackedAbove,
    /// Vertically stacked with `H` below anchor atom.
    StackedBelow,
    /// Centered single or multi-port group.
    Centered,
}

/// Script positioning of a tokenized piece inside a chemical label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScriptKind {
    Normal,
    Subscript,
    Superscript,
}

/// A single typographic run in a formatted chemical formula or heteroatom label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelToken {
    pub text: String,
    pub script: ScriptKind,
}

/// Fully laid-out atom label with subscript/superscript tokens and clearance bounding box
/// relative to the atom's `(0, 0)` anchor in **page points** (`+x` right, `+y` down in screen/SVG).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormattedAtomLabel {
    /// Direction chosen from outgoing bond vectors.
    pub direction: LabelDirection,
    /// Plain chemical string (`"OH"`, `"HO"`, `"COOH"`, `"NH3+"`).
    pub plain: String,
    /// Ordered visual tokens from left to right (or top to bottom when stacked).
    pub tokens: Vec<LabelToken>,
    /// Index in `tokens` of the anchor atom's element glyph (`0` for `OH`, last non-script token for `HO`).
    pub anchor_token_index: usize,
    /// Left edge of rendered text relative to atom anchor in points (`<= 0`).
    pub min_x_pt: f64,
    /// Right edge of rendered text relative to atom anchor in points (`>= 0`).
    pub max_x_pt: f64,
    /// Top edge relative to atom anchor in points (`<= 0`, screen y-down).
    pub min_y_pt: f64,
    /// Bottom edge relative to atom anchor in points (`>= 0`, screen y-down).
    pub max_y_pt: f64,
}

impl FormattedAtomLabel {
    /// Padded clearance box `[min_x, min_y, max_x, max_y]` in points around the atom `(0, 0)`
    /// where incoming bonds are clipped.
    pub fn clearance_box_pt(&self, style: &DocumentStyle) -> [f64; 4] {
        let pad = style.label_clearance_pt();
        [
            self.min_x_pt - pad,
            self.min_y_pt - pad,
            self.max_x_pt + pad,
            self.max_y_pt + pad,
        ]
    }
}

/// Approximate proportional Helvetica/Arial character width in `em` units (`render.ts` metrics).
pub fn char_advance_em(ch: char) -> f64 {
    match ch {
        'I' | 'i' | 'l' | '1' | '|' | '!' | '.' | ',' | ':' | ';' | '\'' => 0.28,
        '(' | ')' | '[' | ']' | '{' | '}' | '-' | 't' | 'f' | 'j' | 'r' => 0.35,
        's' | 'c' | 'z' | 'v' | 'x' | 'y' | 'k' | 'e' | 'a' => 0.52,
        'C' | 'G' | 'O' | 'Q' | 'N' | 'H' | 'U' | 'D' | 'R' | 'B' | 'P' | 'S' | 'A' | 'V' | 'X'
        | 'Y' | 'Z' => 0.68,
        'M' | 'W' | 'm' => 0.84,
        '+' | '=' | '−' | '–' => 0.56,
        _ => 0.58,
    }
}

/// Measure string width in points at a given `font_size_pt`.
pub fn text_advance_pt(s: &str, font_size_pt: f64) -> f64 {
    s.chars().map(char_advance_em).sum::<f64>() * font_size_pt
}

/// Tokenize a chemical label string (`"COOH"`, `"NH2"`, `"H2N"`, `"NH3+"`, `"SO4^2-"`, `"[13C]H3"`, `"CF3"`)
/// into normal, subscript, and superscript runs.
pub fn tokenize_chemical_formula(text: &str) -> Vec<LabelToken> {
    let s = text.trim();
    if s.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<LabelToken> = Vec::new();
    let push_tok = |out: &mut Vec<LabelToken>, ch_str: &str, script: ScriptKind| {
        if ch_str.is_empty() {
            return;
        }
        if let Some(last) = out.last_mut() {
            if last.script == script {
                last.text.push_str(ch_str);
                return;
            }
        }
        out.push(LabelToken {
            text: ch_str.to_string(),
            script,
        });
    };

    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;

    // Leading isotope in brackets e.g. `[13C]` or `13CH3`
    if i < n && chars[i] == '[' {
        if let Some(close) = chars[i + 1..].iter().position(|&c| c == ']') {
            let inner: String = chars[i + 1..i + 1 + close].iter().collect();
            let digits_end = inner
                .char_indices()
                .take_while(|&(_, c)| c.is_ascii_digit())
                .last()
                .map(|(idx, c)| idx + c.len_utf8())
                .unwrap_or(0);
            if digits_end > 0 && digits_end < inner.len() {
                push_tok(&mut out, &inner[..digits_end], ScriptKind::Superscript);
                push_tok(&mut out, &inner[digits_end..], ScriptKind::Normal);
                i = i + 2 + close;
            }
        }
    }

    while i < n {
        let ch = chars[i];
        // Explicit caret superscript `^2+` or `^+`
        if ch == '^' && i + 1 < n {
            i += 1;
            let start = i;
            while i < n
                && (chars[i].is_ascii_digit()
                    || chars[i] == '+'
                    || chars[i] == '-'
                    || chars[i] == '−'
                    || chars[i] == '•')
            {
                i += 1;
            }
            let sup: String = chars[start..i].iter().collect();
            push_tok(&mut out, &sup.replace('-', "−"), ScriptKind::Superscript);
            continue;
        }

        // Trailing charge like `NH3+`, `O-`, `Fe2+`, `SO42-` (when at end of string)
        if (ch == '+' || ch == '-' || ch == '−' || ch == '⊕' || ch == '⊖') && i + 1 == n && i > 0
        {
            let sign = if ch == '-' {
                "−".to_string()
            } else {
                ch.to_string()
            };
            push_tok(&mut out, &sign, ScriptKind::Superscript);
            i += 1;
            continue;
        }
        if ch.is_ascii_digit()
            && i + 2 == n
            && (chars[i + 1] == '+' || chars[i + 1] == '-' || chars[i + 1] == '−')
            && i > 0
        {
            // Check if previous char is a letter or closing paren -> e.g. `Fe2+` or `(SO4)2-`
            // Note: for `NH3+`, `3` is subscript and `+` is superscript!
            let prev = chars[i - 1];
            let sign = if chars[i + 1] == '-' {
                "−".to_string()
            } else {
                chars[i + 1].to_string()
            };
            if prev == 'H' || prev == 'O' || prev == 'F' || prev == 'l' || prev == 'r' {
                // Subscript stoichiometric digit followed by single charge (`NH3+`, `OH2+`)
                push_tok(&mut out, &ch.to_string(), ScriptKind::Subscript);
                push_tok(&mut out, &sign, ScriptKind::Superscript);
                i += 2;
                continue;
            } else {
                let mut sup = ch.to_string();
                sup.push_str(&sign);
                push_tok(&mut out, &sup, ScriptKind::Superscript);
                i += 2;
                continue;
            }
        }

        // Digits after an element letter or `)` become subscripts (`NH2`, `CF3`, `CH2`, `(CH3)3`)
        if ch.is_ascii_digit() && i > 0 {
            let prev = chars[i - 1];
            if prev.is_ascii_alphabetic() || prev == ')' || prev == ']' {
                let start = i;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let sub: String = chars[start..i].iter().collect();
                push_tok(&mut out, &sub, ScriptKind::Subscript);
                continue;
            }
        }

        push_tok(&mut out, &ch.to_string(), ScriptKind::Normal);
        i += 1;
    }

    out
}

/// Reverse a simple condensed functional group when bonds point right (`dx > 0`) so the
/// anchor heteroatom/carbon sits directly on the bond terminus (`OH` -> `HO`, `NH2` -> `H2N`,
/// `COOH` -> `HOOC`, `SH` -> `HS`, `OCH3` -> `H3CO`, `OMe` -> `MeO`, `NHBoc` -> `BocHN`).
fn reverse_condensed_for_rtl(label: &str) -> Option<(String, usize)> {
    match label {
        "OH" => Some(("HO".to_string(), 1)),
        "SH" => Some(("HS".to_string(), 1)),
        "SeH" => Some(("HSe".to_string(), 1)),
        "NH" => Some(("HN".to_string(), 1)),
        "NH2" => Some(("H2N".to_string(), 2)),
        "NH3+" => Some(("H3N+".to_string(), 2)),
        "PH2" => Some(("H2P".to_string(), 2)),
        "COOH" => Some(("HOOC".to_string(), 3)),
        "CO2H" => Some(("HO2C".to_string(), 3)),
        "OCH3" => Some(("H3CO".to_string(), 3)),
        "OMe" => Some(("MeO".to_string(), 1)),
        "OEt" => Some(("EtO".to_string(), 1)),
        "OAc" => Some(("AcO".to_string(), 1)),
        "OTs" => Some(("TsO".to_string(), 1)),
        "OTBS" => Some(("TBSO".to_string(), 1)),
        "NHAc" => Some(("AcHN".to_string(), 2)),
        "NHBoc" => Some(("BocHN".to_string(), 2)),
        "NHFmoc" => Some(("FmocHN".to_string(), 2)),
        "CH3" => Some(("H3C".to_string(), 2)),
        "CF3" => Some(("F3C".to_string(), 2)),
        "CN" => Some(("NC".to_string(), 1)),
        "NO2" => Some(("O2N".to_string(), 2)),
        "SO3H" => Some(("HO3S".to_string(), 3)),
        _ => None,
    }
}

/// Format an atom's visible label (or return `None` for an implicit skeletal carbon!) given
/// outgoing bond unit vectors `bond_dirs_mol` in **molecular y-up coordinates** (`+x` right, `+y` up).
#[allow(clippy::too_many_arguments)]
pub fn layout_atom_label(
    element: &str,
    custom_text: Option<&str>,
    charge: i8,
    radical: u8,
    isotope: Option<u16>,
    explicit_h: Option<u8>,
    show_carbon: bool,
    bond_order_sum: f64,
    bond_dirs_mol: &[[f64; 2]],
    style: &DocumentStyle,
) -> Option<FormattedAtomLabel> {
    let custom_clean = custom_text.map(str::trim).filter(|s| !s.is_empty());

    // Skeletal carbon hiding rule: plain neutral 12C with >= 1 bond and `show_carbon == false`
    // (and not a linear cumulated allene center if desired, though callers can set show_carbon)
    if custom_clean.is_none()
        && element == "C"
        && !show_carbon
        && charge == 0
        && radical == 0
        && isotope.is_none()
        && explicit_h.is_none()
        && !bond_dirs_mol.is_empty()
    {
        return None;
    }

    // Sum outgoing bond unit vectors in molecular space (+x right, +y up)
    let mut sum_dx = 0.0_f64;
    let mut sum_dy = 0.0_f64;
    for d in bond_dirs_mol {
        sum_dx += d[0];
        sum_dy += d[1];
    }

    let h_count = if custom_clean.is_some() {
        0
    } else {
        explicit_h.unwrap_or_else(|| implicit_hydrogens(element, charge, bond_order_sum, radical))
    };

    let charge_suffix = if custom_clean.is_some() {
        String::new()
    } else if charge == 1 {
        "+".to_string()
    } else if charge == -1 {
        "-".to_string()
    } else if charge > 1 {
        format!("^{}+", charge)
    } else if charge < -1 {
        format!("^{}-", -charge)
    } else {
        String::new()
    };

    let iso_prefix = match (custom_clean, isotope) {
        (None, Some(iso)) => format!("[{}{}]", iso, element),
        _ => String::new(),
    };

    // Determine direction from outgoing bond vectors:
    // - `sum_dx > 0.12`: bonds leave toward the RIGHT -> open space is on the LEFT (`RightToLeft`: `HO`, `H2N`)
    // - `sum_dx < -0.12`: bonds leave toward the LEFT -> open space is on the RIGHT (`LeftToRight`: `OH`, `NH2`)
    // - `|sum_dx| <= 0.12` and >= 2 bonds with single H (secondary amine `NH`, `PH` at ring apex): stack `H` vertically!
    if custom_clean.is_none()
        && isotope.is_none()
        && charge == 0
        && h_count == 1
        && bond_dirs_mol.len() >= 2
        && sum_dx.abs() <= 0.15
        && sum_dy.abs() > 0.20
    {
        // Stacked vertical `NH`: if bonds point down (`sum_dy < 0`), `H` goes ABOVE `N`;
        // if bonds point up (`sum_dy > 0`), `H` goes BELOW `N`.
        let dir = if sum_dy < 0.0 {
            LabelDirection::StackedAbove
        } else {
            LabelDirection::StackedBelow
        };
        let cap = style.cap_height_pt();
        let pitch = style.stacked_pitch_pt();
        let w_el = text_advance_pt(element, style.atom_label_size_pt);
        let w_h = text_advance_pt("H", style.atom_label_size_pt);
        let half_w = 0.5 * w_el.max(w_h);

        let (tokens, anchor_idx, min_y, max_y) = match dir {
            LabelDirection::StackedAbove => (
                vec![
                    LabelToken {
                        text: "H".to_string(),
                        script: ScriptKind::Normal,
                    },
                    LabelToken {
                        text: element.to_string(),
                        script: ScriptKind::Normal,
                    },
                ],
                1,
                -0.5 * cap - pitch,
                0.5 * cap,
            ),
            _ => (
                vec![
                    LabelToken {
                        text: element.to_string(),
                        script: ScriptKind::Normal,
                    },
                    LabelToken {
                        text: "H".to_string(),
                        script: ScriptKind::Normal,
                    },
                ],
                0,
                -0.5 * cap,
                0.5 * cap + pitch,
            ),
        };
        return Some(FormattedAtomLabel {
            direction: dir,
            plain: format!("{}H", element),
            tokens,
            anchor_token_index: anchor_idx,
            min_x_pt: -half_w,
            max_x_pt: half_w,
            min_y_pt: min_y,
            max_y_pt: max_y,
        });
    }

    let wants_rtl = if bond_dirs_mol.is_empty() {
        false
    } else {
        sum_dx > 0.12
    };

    let (direction, formula_str, explicit_anchor_tok) = if let Some(txt) = custom_clean {
        if wants_rtl {
            if let Some((rev, anchor_i)) = reverse_condensed_for_rtl(txt) {
                (LabelDirection::RightToLeft, rev, Some(anchor_i))
            } else {
                (LabelDirection::LeftToRight, txt.to_string(), Some(0))
            }
        } else {
            (LabelDirection::LeftToRight, txt.to_string(), Some(0))
        }
    } else {
        let base_el = if iso_prefix.is_empty() {
            element.to_string()
        } else {
            iso_prefix
        };
        let hs = if h_count == 0 {
            String::new()
        } else if h_count == 1 {
            "H".to_string()
        } else {
            format!("H{}", h_count)
        };

        if bond_dirs_mol.is_empty() && h_count > 0 && is_h_first_element(element) && charge == 0 {
            // Unbonded `H2O`, `HCl`, `HF`, `H2S`: H-first, anchor on heteroatom
            let anchor_tok = if h_count > 1 { 2 } else { 1 };
            (
                LabelDirection::RightToLeft,
                format!("{}{}", hs, base_el),
                Some(anchor_tok),
            )
        } else if wants_rtl && h_count > 0 {
            // Bonds go right -> `HO`, `H2N`, `H3C` extending left from anchor atom!
            let anchor_tok = if h_count > 1 { 2 } else { 1 };
            (
                LabelDirection::RightToLeft,
                format!("{}{}{}", hs, base_el, charge_suffix),
                Some(anchor_tok),
            )
        } else {
            // Bonds go left (or vertical with no H flip) -> `OH`, `NH2`, `CH3` extending right!
            let anchor_tok = if isotope.is_some() { 1 } else { 0 };
            (
                LabelDirection::LeftToRight,
                format!("{}{}{}", base_el, hs, charge_suffix),
                Some(anchor_tok),
            )
        }
    };

    let tokens = tokenize_chemical_formula(&formula_str);
    if tokens.is_empty() {
        return None;
    }

    let anchor_idx = explicit_anchor_tok
        .unwrap_or(0)
        .min(tokens.len().saturating_sub(1));

    // Compute exact horizontal advances for each token so the anchor token's center sits at x = 0!
    let widths: Vec<f64> = tokens
        .iter()
        .map(|t| {
            let sz = match t.script {
                ScriptKind::Normal => style.atom_label_size_pt,
                ScriptKind::Subscript => style.subscript_size_pt(),
                ScriptKind::Superscript => style.superscript_size_pt(),
            };
            text_advance_pt(&t.text, sz)
        })
        .collect();

    // Anchor on the first character of `tokens[anchor_idx]` when LTR, or last character of `tokens[anchor_idx]` when RTL
    let prefix_w: f64 = widths[..anchor_idx].iter().sum();
    let anchor_tok_text = &tokens[anchor_idx].text;
    let anchor_char_half = if direction == LabelDirection::RightToLeft {
        let last_ch = anchor_tok_text.chars().last().unwrap_or('C');
        let last_w = char_advance_em(last_ch) * style.atom_label_size_pt;
        widths[anchor_idx] - 0.5 * last_w
    } else {
        let first_ch = anchor_tok_text.chars().next().unwrap_or('C');
        0.5 * char_advance_em(first_ch) * style.atom_label_size_pt
    };

    let anchor_center_from_left = prefix_w + anchor_char_half;
    let total_w: f64 = widths.iter().sum();

    let cap = style.cap_height_pt();
    let has_sub = tokens.iter().any(|t| t.script == ScriptKind::Subscript);
    let has_sup = tokens.iter().any(|t| t.script == ScriptKind::Superscript);

    let min_y = -0.5 * cap
        - if has_sup {
            style.superscript_rise_pt() * 0.65
        } else {
            0.0
        };
    let max_y = 0.5 * cap
        + if has_sub {
            style.subscript_drop_pt()
        } else {
            0.0
        };

    Some(FormattedAtomLabel {
        direction,
        plain: formula_str,
        tokens,
        anchor_token_index: anchor_idx,
        min_x_pt: -anchor_center_from_left,
        max_x_pt: total_w - anchor_center_from_left,
        min_y_pt: min_y,
        max_y_pt: max_y,
    })
}
