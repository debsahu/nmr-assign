//! Publication chemical structure, reaction scheme, clipware, and 1D NMR rendering engine.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis.
//!
//! Submodules:
//! - [`styles`]: Journal style sheets (`ACS Document 1996`, `RSC`, `JACS`, `Wiley`, `Angewandte`,
//!   `Nature`, `Presentation`), CPK & highlight palettes, and direction-aware heteroatom
//!   subscript/superscript typography (`NH`/`HO` left vs `OH` right vs stacked `NH`).
//! - [`svg_pdf`]: 2D scene graph (`RenderScene`), ring-aware vector compiler, W3C SVG 1.1 exporter
//!   (`render_scene_svg`), standalone pure-Rust PDF 1.4 exporter (`render_scene_pdf`), and
//!   anti-aliased lossless PNG/PPM software rasterizer (`render_scene_png`, `render_scene_ppm`).
//! - [`templates`]: Built-in gallery of 50+ ready-to-stamp 2D templates (rings, heterocycles,
//!   chair conformers with axial/equatorial sprouting, bicyclics, steroid core, 20 L-amino acids &
//!   polypeptide builder, Haworth/chair sugars, ATP/nucleosides, peptide β-turn, ferrocene,
//!   square-planar & octahedral complexes, TLC plates, and `s`/`p`/`d` orbital lobes).

pub mod styles;
pub mod svg_pdf;
pub mod templates;

pub use styles::{
    char_advance_em, element_cpk_color, implicit_hydrogens, is_h_first_element, layout_atom_label,
    text_advance_pt, tokenize_chemical_formula, DocumentStyle, FormattedAtomLabel, JournalStyle,
    LabelDirection, LabelToken, PageSetup, RgbaColor, ScriptKind, HIGHLIGHT_PALETTE, INK_PALETTE,
};
pub use svg_pdf::{
    compile_scene_primitives, encode_rgba_png, expand_multiplet_lines_hz, flatten_path_commands,
    perceive_small_rings, render_scene_pdf, render_scene_png, render_scene_png_default,
    render_scene_ppm, render_scene_svg, ArrowKind, BondOrder, BondStereo, BracketStyle,
    CompiledScene, DoubleBondSide, OrbitalShading, OrbitalType, PathCmd, PushingArrowKind,
    RenderScene, SceneAtom, SceneBond, SceneBracket, SceneCurvedArrow, SceneOrbital,
    SceneReactionArrow, SceneRingFill, SceneSpectrum1D, SceneText, SceneTlcPlate, SpectrumSignal1D,
    TlcSpot, TlcSpotStyle, VectorPrimitive,
};
pub use templates::{
    build_peptide_scene, chair_sprout_vectors, fuse_polygon_on_bond, make_chair_cyclohexane_scene,
    normalize_amino_acid_code, parse_peptide_sequence, regular_polygon_scene,
    ChairAtomSproutAngles, ChairSproutVectors, StructureTemplate, TemplateCategory,
    TemplateLibrary,
};
