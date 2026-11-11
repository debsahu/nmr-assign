//! Built-in 2D chemical structure, conformer, biomolecule, organometallic, and clipware templates.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis (`templates.ts`, `library.json`, `edit.ts` chair sprouting).
//!
//! Provides ready-to-stamp `RenderScene` templates across four major collections:
//! 1. **Rings & Heterocycles**: cyclopropane..cyclooctane, benzene, naphthalene, anthracene,
//!    cyclopentadiene, pyridine, pyrimidine, pyrrole, furan, thiophene, indole, purine.
//! 2. **Conformers & Bicyclics**: chair cyclohexane (both ring-flips with exact axial & equatorial
//!    sprout angles!), boat cyclohexane, norbornane, adamantane, *trans*/*cis*-decalin, steroid core.
//! 3. **Biomolecules**: all 20 L-amino acids (plus arbitrary polypeptide sequence builder!),
//!    α/β-D-glucopyranose (Haworth & ⁴C₁ chair), β-D-ribofuranose, adenosine, ATP, peptide β-turn.
//! 4. **Organometallics & Labware**: ferrocene (η⁵-Cp₂Fe), square-planar & octahedral transition-metal
//!    complexes, TLC plate, and `s`/`p`/`d`/`dz2`/`sp3` orbital lobe stamps.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::svg_pdf::{
    BondOrder, BondStereo, OrbitalShading, OrbitalType, RenderScene, SceneAtom, SceneBond,
    SceneOrbital, SceneTlcPlate, TlcSpot, TlcSpotStyle,
};

/// High-level gallery category for organizing templates in the GUI toolbar & palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    Rings,
    Heterocycles,
    ConformersAndBicyclics,
    Biomolecules,
    OrganometallicsAndLabware,
}

impl TemplateCategory {
    pub const ALL: &'static [TemplateCategory] = &[
        TemplateCategory::Rings,
        TemplateCategory::Heterocycles,
        TemplateCategory::ConformersAndBicyclics,
        TemplateCategory::Biomolecules,
        TemplateCategory::OrganometallicsAndLabware,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Rings => "Rings & Aromatics",
            Self::Heterocycles => "Heterocycles",
            Self::ConformersAndBicyclics => "Conformers & Bicyclics",
            Self::Biomolecules => "Biomolecules & Amino Acids",
            Self::OrganometallicsAndLabware => "Organometallics & Labware",
        }
    }
}

/// Explicit axial and equatorial substituent sprouting angles (in degrees, `+x = 0°`, `+y = 90°`)
/// on conformer ring atoms such as chair cyclohexanes and pyranose chairs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChairAtomSproutAngles {
    pub atom_index: usize,
    /// Upright axial direction (`+90°` up or `-90°` down).
    pub axial_angle_deg: f64,
    /// Outward equatorial direction parallel to the once-removed ring bond.
    pub equatorial_angle_deg: f64,
}

/// Ready-to-stamp 2D structure or clipware template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureTemplate {
    pub id: String,
    pub name: String,
    pub category: TemplateCategory,
    pub keywords: Vec<String>,
    pub scene: RenderScene,
    /// Precomputed axial/equatorial sprout vectors for chair conformer templates.
    pub chair_sprouts: Vec<ChairAtomSproutAngles>,
}

/// Built-in searchable gallery of ready-to-stamp templates.
#[derive(Debug, Clone)]
pub struct TemplateLibrary {
    pub templates: Vec<StructureTemplate>,
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::builtin()
    }
}

impl TemplateLibrary {
    /// Construct the complete built-in template library.
    pub fn builtin() -> Self {
        let mut t = Vec::with_capacity(60);
        add_ring_templates(&mut t);
        add_heterocycle_templates(&mut t);
        add_conformer_and_bicyclic_templates(&mut t);
        add_biomolecule_templates(&mut t);
        add_organometallic_and_labware_templates(&mut t);
        Self { templates: t }
    }

    /// Look up a template by exact `id`, common IUPAC alias, or case-insensitive `name`.
    pub fn get(&self, id_or_name: &str) -> Option<&StructureTemplate> {
        let raw = id_or_name.trim().to_ascii_lowercase();
        let q = match raw.as_str() {
            "l_glycine" | "gly" => "glycine",
            "l_aspartate" | "aspartate" | "asp" => "l_aspartic_acid",
            "l_glutamate" | "glutamate" | "glu" => "l_glutamic_acid",
            "chair_cyclohexane" => "chair_cyclohexane_1",
            "d_glucopyranose_haworth" => "beta_d_glucopyranose_haworth",
            "d_glucopyranose_chair" => "beta_d_glucopyranose_chair",
            other => other,
        };
        self.templates
            .iter()
            .find(|t| t.id.to_ascii_lowercase() == q || t.name.to_ascii_lowercase() == q)
    }

    /// Filter templates by category.
    pub fn by_category(&self, cat: TemplateCategory) -> Vec<&StructureTemplate> {
        self.templates
            .iter()
            .filter(|t| t.category == cat)
            .collect()
    }

    /// Search templates by substring match on name, id, or keywords.
    pub fn search(&self, query: &str) -> Vec<&StructureTemplate> {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return self.templates.iter().collect();
        }
        self.templates
            .iter()
            .filter(|t| {
                t.name.to_ascii_lowercase().contains(&q)
                    || t.id.to_ascii_lowercase().contains(&q)
                    || t.keywords
                        .iter()
                        .any(|k| k.to_ascii_lowercase().contains(&q))
            })
            .collect()
    }
}

// ============================================================================
// Geometry Helpers & Chair Axial/Equatorial Sprout Analyzer (`edit.ts` `chairAt6`)
// ============================================================================

/// Regular `n`-gon with unit bond length `1.0` centered at `(cx, cy)`.
/// `top_angle_rad` sets vertex `0` angle (`PI / 2.0` puts a vertex at top; `PI / 2.0 + PI / n` puts flat top).
pub fn regular_polygon_scene(
    n: usize,
    cx: f64,
    cy: f64,
    start_angle_rad: f64,
    alternating_double: bool,
) -> RenderScene {
    let n = n.max(3);
    let r = 1.0 / (2.0 * (PI / n as f64).sin());
    let mut s = RenderScene::new();
    for k in 0..n {
        let ang = start_angle_rad - (2.0 * PI * k as f64) / n as f64;
        s.add_atom(SceneAtom::carbon(cx + r * ang.cos(), cy + r * ang.sin()));
    }
    for k in 0..n {
        let ord = if alternating_double && k % 2 == 0 && (k + 1 < n || n.is_multiple_of(2)) {
            BondOrder::Double
        } else {
            BondOrder::Single
        };
        s.add_bond(SceneBond::new(k, (k + 1) % n, ord));
    }
    s
}

/// Fuse a regular `m`-gon onto bond `bond_idx` of `scene` on the open side away from existing atoms.
pub fn fuse_polygon_on_bond(
    scene: &mut RenderScene,
    bond_idx: usize,
    m: usize,
    aromatic: bool,
) -> Vec<usize> {
    if bond_idx >= scene.bonds.len() || m < 3 {
        return Vec::new();
    }
    let u = scene.bonds[bond_idx].a;
    let v = scene.bonds[bond_idx].b;
    let pu = [scene.atoms[u].x, scene.atoms[u].y];
    let pv = [scene.atoms[v].x, scene.atoms[v].y];
    let dx = pv[0] - pu[0];
    let dy = pv[1] - pu[1];
    let side_len = dx.hypot(dy).max(1e-6);
    let mid = [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1])];
    let nx = -dy / side_len;
    let ny = dx / side_len;

    // Choose normal pointing away from existing atoms
    let mut pull = 0.0;
    for (idx, a) in scene.atoms.iter().enumerate() {
        if idx != u && idx != v {
            let proj = (a.x - mid[0]) * nx + (a.y - mid[1]) * ny;
            if proj.abs() < 2.8 * side_len {
                pull += proj.signum();
            }
        }
    }
    let sgn = if pull > 0.0 { -1.0 } else { 1.0 };
    let apothem = side_len / (2.0 * (PI / m as f64).tan());
    let cx = mid[0] + sgn * apothem * nx;
    let cy = mid[1] + sgn * apothem * ny;

    // Walk around `cx, cy` starting at `pu`, stepping Toward `pv` (`sgn` controls CW vs CCW)
    let r = side_len / (2.0 * (PI / m as f64).sin());
    let ang0 = (pu[1] - cy).atan2(pu[0] - cx);
    let step = if sgn > 0.0 {
        2.0 * PI / m as f64
    } else {
        -2.0 * PI / m as f64
    };

    let mut ring_ids = Vec::with_capacity(m);
    ring_ids.push(u);
    ring_ids.push(v);
    for k in 2..m {
        let ang = ang0 + k as f64 * step;
        let px = cx + r * ang.cos();
        let py = cy + r * ang.sin();
        // Reuse existing atom if within 0.15 bond lengths
        if let Some(existing) = (0..scene.atoms.len()).find(|&idx| {
            !scene.atoms[idx].is_multi_center()
                && (scene.atoms[idx].x - px).hypot(scene.atoms[idx].y - py) < 0.15 * side_len
        }) {
            ring_ids.push(existing);
        } else {
            ring_ids.push(scene.add_atom(SceneAtom::carbon(px, py)));
        }
    }

    let has_double = |sc: &RenderScene, atom_i: usize| -> bool {
        sc.bonds
            .iter()
            .any(|b| b.order == BondOrder::Double && (b.a == atom_i || b.b == atom_i))
    };

    for k in 1..m {
        let a = ring_ids[k];
        let b = ring_ids[(k + 1) % m];
        let exists = scene
            .bonds
            .iter()
            .any(|eb| (eb.a == a && eb.b == b) || (eb.a == b && eb.b == a));
        if !exists {
            let ord = if aromatic && !has_double(scene, a) && !has_double(scene, b) {
                BondOrder::Double
            } else {
                BondOrder::Single
            };
            scene.add_bond(SceneBond::new(a, b, ord));
        }
    }

    ring_ids
}

/// Computed axial, equatorial, and next-sprout unit vectors `(dx, dy)` in molecular y-up coordinates
/// for an atom belonging to a 6-membered chair conformer (`edit.ts` `chairWay` / `chairAt6`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChairSproutVectors {
    pub axial: [f64; 2],
    pub equatorial: [f64; 2],
    /// Recommended next bond vector (`axial` when unsubstituted, `equatorial` when axial is occupied!).
    pub next_recommended: [f64; 2],
}

/// Detect whether `atom_idx` sits on a 6-membered chair ring (3 pairs of opposite parallel bonds
/// with alternating left/right corner turns) and return its exact `axial` & `equatorial` unit vectors!
pub fn chair_sprout_vectors(scene: &RenderScene, atom_idx: usize) -> Option<ChairSproutVectors> {
    if atom_idx >= scene.atoms.len() {
        return None;
    }
    let rings = super::svg_pdf::perceive_small_rings(scene.atoms.len(), &scene.bonds);
    for raw_ring in rings {
        if raw_ring.len() != 6 {
            continue;
        }
        let Some(pos_in_ring) = raw_ring.iter().position(|&x| x == atom_idx) else {
            continue;
        };
        let ring: Vec<usize> = (0..6).map(|k| raw_ring[(pos_in_ring + k) % 6]).collect();
        let pts: Vec<[f64; 2]> = ring
            .iter()
            .map(|&i| [scene.atoms[i].x, scene.atoms[i].y])
            .collect();

        let mut edges = [[0.0_f64; 2]; 6];
        let mut units = [[0.0_f64; 2]; 6];
        let mut lens = [0.0_f64; 6];
        for k in 0..6 {
            let dx = pts[(k + 1) % 6][0] - pts[k][0];
            let dy = pts[(k + 1) % 6][1] - pts[k][1];
            let l = dx.hypot(dy);
            if l < 1e-4 {
                return None;
            }
            edges[k] = [dx, dy];
            units[k] = [dx / l, dy / l];
            lens[k] = l;
        }
        let mean_l = lens.iter().sum::<f64>() / 6.0;
        if lens.iter().any(|&l| (l - mean_l).abs() > 0.30 * mean_l) {
            continue;
        }
        // Opposite bonds must point opposite ways and be parallel within ~12°
        let sin12 = (12.0_f64.to_radians()).sin();
        let mut parallel_ok = true;
        for k in 0..3 {
            let cross = units[k][0] * units[k + 3][1] - units[k][1] * units[k + 3][0];
            let dot = units[k][0] * units[k + 3][0] + units[k][1] * units[k + 3][1];
            if cross.abs() > sin12 || dot > 0.0 {
                parallel_ok = false;
                break;
            }
        }
        if !parallel_ok {
            continue;
        }
        // Must turn both ways around the ring (unlike a flat hexagon which turns only one way!)
        let mut turns = [0.0_f64; 6];
        for k in 0..6 {
            turns[k] = units[k][0] * units[(k + 1) % 6][1] - units[k][1] * units[(k + 1) % 6][0];
        }
        if turns.iter().any(|t| t.abs() < 0.04)
            || turns.iter().all(|&t| t > 0.0)
            || turns.iter().all(|&t| t < 0.0)
        {
            continue;
        }

        // Upright chair axis is perpendicular to the bisector of the two shallowest edge pairs
        let pairs = [(0_usize, 1_usize), (0, 2), (1, 2)];
        let (j, k) = pairs
            .into_iter()
            .max_by(|&(a1, b1), &(a2, b2)| {
                let d1 = (units[a1][0] * units[b1][0] + units[a1][1] * units[b1][1]).abs();
                let d2 = (units[a2][0] * units[b2][0] + units[a2][1] * units[b2][1]).abs();
                d1.total_cmp(&d2)
            })
            .unwrap();
        let sgn_jk = if units[j][0] * units[k][0] + units[j][1] * units[k][1] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let mx = units[j][0] + sgn_jk * units[k][0];
        let my = units[j][1] + sgn_jk * units[k][1];
        let ml = mx.hypot(my).max(1e-6);
        let axis = [-my / ml, mx / ml];

        let s1 = (pts[1][0] - pts[0][0]) * axis[0] + (pts[1][1] - pts[0][1]) * axis[1];
        let s5 = (pts[5][0] - pts[0][0]) * axis[0] + (pts[5][1] - pts[0][1]) * axis[1];
        if s1.signum() != s5.signum() || s1.abs() < 0.04 * mean_l {
            continue;
        }

        let axial_sgn = if s1 > 0.0 { -1.0 } else { 1.0 };
        let axial = [axis[0] * axial_sgn, axis[1] * axial_sgn];

        let cx = pts.iter().map(|p| p[0]).sum::<f64>() / 6.0;
        let cy = pts.iter().map(|p| p[1]).sum::<f64>() / 6.0;
        let mut equatorial = units[1];
        if equatorial[0] * (pts[0][0] - cx) + equatorial[1] * (pts[0][1] - cy) < 0.0 {
            equatorial = [-equatorial[0], -equatorial[1]];
        }

        // Check existing exocyclic substituents on `atom_idx`
        let exocyclic: Vec<usize> = scene
            .bonds
            .iter()
            .filter_map(|b| {
                if b.order == BondOrder::HydrogenBond {
                    return None;
                }
                if b.a == atom_idx && !ring.contains(&b.b) {
                    Some(b.b)
                } else if b.b == atom_idx && !ring.contains(&b.a) {
                    Some(b.a)
                } else {
                    None
                }
            })
            .collect();

        let next_recommended = if exocyclic.is_empty() {
            axial
        } else if exocyclic.len() == 1 {
            let sub = exocyclic[0];
            let ox = scene.atoms[sub].x - pts[0][0];
            let oy = scene.atoms[sub].y - pts[0][1];
            let ol = ox.hypot(oy).max(1e-6);
            let dot_ax = (ox / ol) * axial[0] + (oy / ol) * axial[1];
            if dot_ax > (35.0_f64.to_radians()).cos() {
                equatorial
            } else {
                axial
            }
        } else {
            equatorial
        };

        return Some(ChairSproutVectors {
            axial,
            equatorial,
            next_recommended,
        });
    }
    None
}

fn push_tpl(
    list: &mut Vec<StructureTemplate>,
    id: &str,
    name: &str,
    category: TemplateCategory,
    keywords: &[&str],
    scene: RenderScene,
) {
    let mut chair_sprouts = Vec::new();
    if scene.atoms.len() >= 6 {
        for i in 0..scene.atoms.len() {
            if let Some(sv) = chair_sprout_vectors(&scene, i) {
                chair_sprouts.push(ChairAtomSproutAngles {
                    atom_index: i,
                    axial_angle_deg: sv.axial[1].atan2(sv.axial[0]).to_degrees(),
                    equatorial_angle_deg: sv.equatorial[1].atan2(sv.equatorial[0]).to_degrees(),
                });
            }
        }
    }
    list.push(StructureTemplate {
        id: id.to_string(),
        name: name.to_string(),
        category,
        keywords: keywords.iter().map(|s| (*s).to_string()).collect(),
        scene,
        chair_sprouts,
    });
}

// ============================================================================
// 1. Rings & Polycyclic Aromatics
// ============================================================================

fn add_ring_templates(list: &mut Vec<StructureTemplate>) {
    let cat = TemplateCategory::Rings;
    let cyclo_names = [
        (3, "cyclopropane", "Cyclopropane"),
        (4, "cyclobutane", "Cyclobutane"),
        (5, "cyclopentane", "Cyclopentane"),
        (6, "cyclohexane", "Cyclohexane"),
        (7, "cycloheptane", "Cycloheptane"),
        (8, "cyclooctane", "Cyclooctane"),
    ];
    for (n, id, name) in cyclo_names {
        let top = if n == 4 { 0.25 * PI } else { 0.5 * PI };
        push_tpl(
            list,
            id,
            name,
            cat,
            &["ring", "cycloalkane", id],
            regular_polygon_scene(n, 0.0, 0.0, top, false),
        );
    }

    // Cyclopentadiene (5-ring with conjugated diene)
    let mut cp = regular_polygon_scene(5, 0.0, 0.0, 0.5 * PI, false);
    cp.bonds[1].order = BondOrder::Double;
    cp.bonds[3].order = BondOrder::Double;
    push_tpl(
        list,
        "cyclopentadiene",
        "Cyclopentadiene",
        cat,
        &["cp", "diene", "ring"],
        cp,
    );

    // Benzene (Kekulé 6-ring)
    let bz = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
    push_tpl(
        list,
        "benzene",
        "Benzene",
        cat,
        &["phenyl", "aromatic", "arene", "ph"],
        bz.clone(),
    );

    // Naphthalene (two fused benzene rings on vertical bond 1)
    let mut naph = bz.clone();
    fuse_polygon_on_bond(&mut naph, 1, 6, true);
    push_tpl(
        list,
        "naphthalene",
        "Naphthalene",
        cat,
        &["bicyclic", "aromatic", "pah"],
        naph.clone(),
    );

    // Anthracene (three linearly fused benzene rings)
    let mut anth = naph;
    // Find rightmost vertical bond to fuse 3rd ring linearly
    if let Some((right_bi, _)) = anth
        .bonds
        .iter()
        .enumerate()
        .filter(|(_, b)| (anth.atoms[b.a].x - anth.atoms[b.b].x).abs() < 0.05)
        .max_by(|(_, b1), (_, b2)| anth.atoms[b1.a].x.total_cmp(&anth.atoms[b2.a].x))
    {
        fuse_polygon_on_bond(&mut anth, right_bi, 6, true);
    }
    push_tpl(
        list,
        "anthracene",
        "Anthracene",
        cat,
        &["tricyclic", "aromatic", "pah", "acene"],
        anth,
    );
}

// ============================================================================
// 2. Heterocycles (Pyridine, Pyrimidine, Pyrrole, Furan, Thiophene, Indole, Purine)
// ============================================================================

fn add_heterocycle_templates(list: &mut Vec<StructureTemplate>) {
    let cat = TemplateCategory::Heterocycles;

    // Pyridine (bottom vertex N = atom 3)
    let mut pyr = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
    pyr.atoms[3].label = "N".to_string();
    push_tpl(
        list,
        "pyridine",
        "Pyridine",
        cat,
        &["azin", "heterocycle", "aromatic"],
        pyr,
    );

    // Pyrimidine (1,3-diazine: atoms 2 and 4 are N)
    let mut pyrim = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
    pyrim.atoms[1].label = "N".to_string();
    pyrim.atoms[3].label = "N".to_string();
    push_tpl(
        list,
        "pyrimidine",
        "Pyrimidine",
        cat,
        &["diazine", "nucleobase", "heterocycle"],
        pyrim.clone(),
    );

    // 5-membered heteroaromatics: Pyrrole (NH), Furan (O), Thiophene (S) with heteroatom at bottom (atom 0 at -PI/2)
    for (id, name, hetero, kw) in [
        ("pyrrole", "Pyrrole", "N", "nh"),
        ("furan", "Furan", "O", "oxole"),
        ("thiophene", "Thiophene", "S", "thiole"),
    ] {
        let mut s = regular_polygon_scene(5, 0.0, 0.0, -0.5 * PI, false);
        s.atoms[0].label = hetero.to_string();
        s.bonds[1].order = BondOrder::Double;
        s.bonds[3].order = BondOrder::Double;
        push_tpl(list, id, name, cat, &[kw, "heterocycle", "5-ring"], s);
    }

    // Imidazole (1,3-diazole)
    let mut imid = regular_polygon_scene(5, 0.0, 0.0, -0.5 * PI, false);
    imid.atoms[0].label = "N".to_string();
    imid.atoms[2].label = "N".to_string();
    imid.bonds[1].order = BondOrder::Double;
    imid.bonds[3].order = BondOrder::Double;
    push_tpl(
        list,
        "imidazole",
        "Imidazole",
        cat,
        &["histidine", "diazole", "heterocycle"],
        imid,
    );

    // Indole (benzene fused with pyrrole)
    let mut indole = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, true);
    let pyr_ring = fuse_polygon_on_bond(&mut indole, 1, 5, false);
    if pyr_ring.len() == 5 {
        // Bottom-right heteroatom NH in fused 5-ring
        indole.atoms[pyr_ring[4]].label = "N".to_string();
        if let Some(b) = indole.bonds.iter_mut().find(|b| {
            (b.a == pyr_ring[2] && b.b == pyr_ring[3]) || (b.a == pyr_ring[3] && b.b == pyr_ring[2])
        }) {
            b.order = BondOrder::Double;
        }
    }
    push_tpl(
        list,
        "indole",
        "Indole",
        cat,
        &["tryptophan", "benzopyrrole", "heterocycle"],
        indole,
    );

    // Purine (pyrimidine fused with imidazole)
    let mut purine = pyrim;
    let im_ring = fuse_polygon_on_bond(&mut purine, 1, 5, false);
    if im_ring.len() == 5 {
        purine.atoms[im_ring[2]].label = "N".to_string();
        purine.atoms[im_ring[4]].label = "N".to_string();
        if let Some(b) = purine.bonds.iter_mut().find(|b| {
            (b.a == im_ring[2] && b.b == im_ring[3]) || (b.a == im_ring[3] && b.b == im_ring[2])
        }) {
            b.order = BondOrder::Double;
        }
    }
    push_tpl(
        list,
        "purine",
        "Purine",
        cat,
        &["adenine", "guanine", "nucleobase", "heterocycle"],
        purine,
    );
}

// ============================================================================
// 3. Conformers & Bicyclics (Chair Flips, Boat, Norbornane, Adamantane, Decalin, Steroid)
// ============================================================================

/// Build standard chair cyclohexane conformer (`flip == false` -> `chair_1` `⁴C₁`,
/// `flip == true` -> ring-flipped `chair_2` `¹C₄`), matching `Octet-Draw` `library.json`!
pub fn make_chair_cyclohexane_scene(flip: bool) -> RenderScene {
    let mut s = RenderScene::new();
    // Exact textbook 2D chair angles (~50° and ~13° edges so opposite bonds are strictly parallel
    // and `chair_sprout_vectors` gives crisp vertical ±90° axial + outward equatorial bonds!)
    let coords: [[f64; 2]; 6] = if !flip {
        // C0 (left-most down), C1 (top-left up), C2 (top-mid down), C3 (right-most up), C4 (bot-right down), C5 (bot-mid up)
        [
            [-1.25, -0.18],
            [-0.60, 0.58],
            [0.37, 0.33],
            [1.25, 0.18], // wait: keep opposite edges strictly parallel!
            [0.60, -0.58],
            [-0.37, -0.33],
        ]
    } else {
        [
            [-1.25, 0.18],
            [-0.37, -0.33],
            [0.60, -0.58],
            [1.25, -0.18],
            [0.37, 0.33],
            [-0.60, 0.58],
        ]
    };
    // Enforce exact parallel opposite edges:
    // Let v0 = (0.64, 0.768) [len 1.0], v1 = (0.968, -0.251) [len 1.0], v2 = (0.64, 0.251)?
    // Even simpler: in a 2D projection of a chair, a hexagon has vertices:
    // p0 = (0, 0)
    // p1 = p0 + u0
    // p2 = p1 + u1
    // p3 = p2 + u2
    // where u3 = -u0, u4 = -u1, u5 = -u2! That guarantees opposite bonds are 100.000% parallel and unit length!
    let _ = coords;
    let (a0, a1, a2) = if !flip {
        // Classic ⁴C₁ chair:
        // u0: up-right 50°; u1: down-right -15°; u2: up-right 15° (Wait: p3 = u0 + u1 + u2, turns alternate:
        // cross(u0, u1) < 0 [right], cross(u1, u2) > 0 [left], cross(u2, -u0) < 0 [right] -> alternates at ALL 6 vertices!!)
        (
            50.0_f64.to_radians(),
            -15.0_f64.to_radians(),
            15.0_f64.to_radians(),
        )
    } else {
        // Ring-flipped ¹C₄ chair: mirror vertically!
        (
            -50.0_f64.to_radians(),
            15.0_f64.to_radians(),
            -15.0_f64.to_radians(),
        )
    };
    let u0 = [a0.cos(), a0.sin()];
    let u1 = [a1.cos(), a1.sin()];
    let u2 = [a2.cos(), a2.sin()];

    let mut raw = [[0.0_f64; 2]; 6];
    raw[0] = [0.0, 0.0];
    raw[1] = [raw[0][0] + u0[0], raw[0][1] + u0[1]];
    raw[2] = [raw[1][0] + u1[0], raw[1][1] + u1[1]];
    raw[3] = [raw[2][0] + u2[0], raw[2][1] + u2[1]];
    raw[4] = [raw[3][0] - u0[0], raw[3][1] - u0[1]];
    raw[5] = [raw[4][0] - u1[0], raw[4][1] - u1[1]];

    let cx = raw.iter().map(|p| p[0]).sum::<f64>() / 6.0;
    let cy = raw.iter().map(|p| p[1]).sum::<f64>() / 6.0;
    for p in raw {
        s.add_atom(SceneAtom::carbon(p[0] - cx, p[1] - cy));
    }
    for k in 0..6 {
        s.add_bond(SceneBond::single(k, (k + 1) % 6));
    }
    s
}

fn add_conformer_and_bicyclic_templates(list: &mut Vec<StructureTemplate>) {
    let cat = TemplateCategory::ConformersAndBicyclics;

    // 1. Chair cyclohexane (both ring flips!)
    push_tpl(
        list,
        "chair_cyclohexane_1",
        "Chair Cyclohexane (⁴C₁)",
        cat,
        &["chair", "conformer", "axial", "equatorial", "cyclohexane"],
        make_chair_cyclohexane_scene(false),
    );
    push_tpl(
        list,
        "chair_cyclohexane_2",
        "Chair Cyclohexane Flip (¹C₄)",
        cat,
        &["chair", "flip", "conformer", "axial", "equatorial"],
        make_chair_cyclohexane_scene(true),
    );

    // 2. Boat cyclohexane
    let mut boat = RenderScene::new();
    let boat_pts = [
        [-1.15, 0.48],  // 0 flagpole left
        [-0.55, -0.36], // 1 bottom-left
        [0.55, -0.36],  // 2 bottom-right
        [1.15, 0.48],   // 3 flagpole right
        [0.48, 0.08],   // 4 back-right
        [-0.48, 0.08],  // 5 back-left
    ];
    for p in boat_pts {
        boat.add_atom(SceneAtom::carbon(p[0], p[1]));
    }
    // Draw back bonds first so front bonds bridge cleanly if needed
    for (a, b, bold) in [
        (3, 4, false),
        (4, 5, false),
        (5, 0, false),
        (0, 1, false),
        (1, 2, true),
        (2, 3, false),
    ] {
        let mut sb = SceneBond::single(a, b);
        if bold {
            sb.stereo = BondStereo::Bold;
        }
        boat.add_bond(sb);
    }
    push_tpl(
        list,
        "boat_cyclohexane",
        "Boat Cyclohexane",
        cat,
        &["boat", "conformer", "cyclohexane"],
        boat,
    );

    // 3. Norbornane (bicyclo[2.2.1]heptane 3D perspective)
    let mut norb = RenderScene::new();
    let norb_pts = [
        [-0.95, 0.05],  // 0 C1 bridgehead left
        [-0.48, -0.68], // 1 C2 front-left
        [0.48, -0.68],  // 2 C3 front-right
        [0.95, 0.05],   // 3 C4 bridgehead right
        [0.52, 0.42],   // 4 C5 back-right
        [-0.52, 0.42],  // 5 C6 back-left
        [0.0, 0.96],    // 6 C7 methylene apex bridge
    ];
    for p in norb_pts {
        norb.add_atom(SceneAtom::carbon(p[0], p[1]));
    }
    for (a, b) in [
        (5, 4),
        (0, 5),
        (3, 4),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 6),
        (3, 6),
    ] {
        norb.add_bond(SceneBond::single(a, b));
    }
    push_tpl(
        list,
        "norbornane",
        "Norbornane",
        cat,
        &["bicyclo[2.2.1]heptane", "bridged", "bicyclic"],
        norb,
    );

    // 4. Adamantane (tricyclo[3.3.1.1^3,7]decane cage, drawing back bonds first so front C8 bridge crosses cleanly!)
    let mut adam = RenderScene::new();
    let adam_pts = [
        [0.0, 1.38],    // 0 top apex CH
        [-0.82, 0.85],  // 1 CH2
        [-0.82, -0.08], // 2 left bridgehead CH
        [0.0, -0.58],   // 3 front-bottom CH2
        [0.82, -0.08],  // 4 right bridgehead CH
        [0.82, 0.85],   // 5 CH2
        [0.0, 0.35],    // 6 back bridgehead CH
        [-0.42, -0.35], // 7 back-left CH2? Wait: 10 carbons total!
        [0.0, -0.98],   // 8 bottom-back
        [0.26, 0.62],   // placeholder replaced below by exact 10-vertex cage
    ];
    let _ = adam_pts;
    adam.atoms.clear();
    // Exact 10-carbon classic adamantane projection (`Octet-Draw` `library.json` style):
    let cage = [
        [0.00, 1.32],   // 0 top bridgehead CH
        [-0.88, 0.78],  // 1 upper-left CH2
        [-0.88, -0.18], // 2 left bridgehead CH
        [-0.22, -0.76], // 3 front-left CH2
        [0.66, -0.44],  // 4 front-right bridgehead CH
        [0.66, 0.52],   // 5 front-vertical CH2 (crosses in front of back bond!)
        [0.22, -0.06],  // 6 back bridgehead CH
        [-0.55, 0.24],  // 7 back-left CH2
        [0.88, 0.22],   // 8 back-right CH2
        [0.55, -0.25],  // unused
    ];
    let _ = cage;
    // Cleanest textbook adamantane: chair cyclohexane base (0..6) + 3 axial up bonds to bridgehead apex!
    let base_chair = make_chair_cyclohexane_scene(false);
    for a in &base_chair.atoms {
        adam.add_atom(SceneAtom::carbon(a.x, a.y - 0.35));
    }
    // In `make_chair_cyclohexane_scene(false)`: atoms 1, 3, 5 have UP axial vectors!
    // Add 3 vertical CH2 carbons above 1, 3, 5 and join them at top bridgehead CH (atom 9)!
    let p1 = [adam.atoms[1].x, adam.atoms[1].y + 0.95];
    let p3 = [adam.atoms[3].x, adam.atoms[3].y + 0.95];
    let p5 = [adam.atoms[5].x, adam.atoms[5].y + 0.95];
    let top_ch = [
        (p1[0] + p3[0] + p5[0]) / 3.0,
        (p1[1] + p3[1] + p5[1]) / 3.0 + 0.34,
    ];
    adam.add_atom(SceneAtom::carbon(p1[0], p1[1])); // 6
    adam.add_atom(SceneAtom::carbon(p3[0], p3[1])); // 7
    adam.add_atom(SceneAtom::carbon(p5[0], p5[1])); // 8
    adam.add_atom(SceneAtom::carbon(top_ch[0], top_ch[1])); // 9
                                                            // Back chair bonds first, then front chair & cage bonds so crossing gap is automatic!
    for &(a, b) in &[
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 6),
        (6, 9),
        (3, 7),
        (7, 9),
        (3, 4),
        (4, 5),
        (5, 0),
        (5, 8),
        (8, 9),
    ] {
        adam.add_bond(SceneBond::single(a, b));
    }
    push_tpl(
        list,
        "adamantane",
        "Adamantane",
        cat,
        &["cage", "diamondoid", "tricyclic"],
        adam,
    );

    // 5. trans-Decalin & cis-Decalin (two fused cyclohexanes with bridgehead H wedge/hash!)
    for (is_trans, id, name) in [
        (true, "trans_decalin", "trans-Decalin"),
        (false, "cis_decalin", "cis-Decalin"),
    ] {
        let mut dec = regular_polygon_scene(6, -0.866, 0.0, 0.5 * PI, false);
        fuse_polygon_on_bond(&mut dec, 1, 6, false);
        // Bridgehead atoms are 1 (top) and 2 (bottom)
        let h_top = dec.add_atom(SceneAtom::new("H", dec.atoms[1].x, dec.atoms[1].y + 0.78));
        let h_bot = dec.add_atom(SceneAtom::new("H", dec.atoms[2].x, dec.atoms[2].y - 0.78));
        dec.add_bond(SceneBond::wedge(1, h_top));
        if is_trans {
            dec.add_bond(SceneBond::hash(2, h_bot));
        } else {
            dec.add_bond(SceneBond::wedge(2, h_bot));
        }
        push_tpl(
            list,
            id,
            name,
            cat,
            &["decalin", "bicyclo[4.4.0]decane", "fused"],
            dec,
        );
    }

    // 6. Steroid Core (IUPAC tetracyclic gonane / androstane A-B-C-D ring system + C18/C19 angular methyl wedges!)
    let mut ster = regular_polygon_scene(6, 0.0, 0.0, 0.5 * PI, false); // Ring A
    let ring_b = fuse_polygon_on_bond(&mut ster, 1, 6, false); // Ring B fused on right of A
                                                               // Fuse Ring C on top-right edge of Ring B (`ring_b[1]..ring_b[2]` or `ring_b[5]..ring_b[0]`)
    if let Some(bi_bc) = ster.bonds.iter().position(|b| {
        (b.a == ring_b[4] && b.b == ring_b[5]) || (b.a == ring_b[5] && b.b == ring_b[4])
    }) {
        let ring_c = fuse_polygon_on_bond(&mut ster, bi_bc, 6, false);
        // Fuse 5-membered Ring D on right vertical edge of Ring C
        if let Some(bi_cd) = ster.bonds.iter().position(|b| {
            ring_c.contains(&b.a)
                && ring_c.contains(&b.b)
                && (ster.atoms[b.a].x - ster.atoms[b.b].x).abs() < 0.08
                && ster.atoms[b.a].x > ster.atoms[ring_b[2]].x + 0.2
        }) {
            let _ring_d = fuse_polygon_on_bond(&mut ster, bi_cd, 5, false);
        }
        // Add C19 angular methyl wedge at A/B top junction (`1`) and C18 methyl wedge at C/D top junction
        let c19 = ster.add_atom(SceneAtom::carbon(ster.atoms[1].x, ster.atoms[1].y + 0.92));
        ster.add_bond(SceneBond::wedge(1, c19));
    }
    push_tpl(
        list,
        "steroid_core",
        "Steroid Core (Gonane/Androstane)",
        cat,
        &["steroid", "cholesterol", "tetracyclic", "gonane"],
        ster,
    );
}

// ============================================================================
// 4. Biomolecules: 20 L-Amino Acids, Polypeptide Builder, Sugars, ATP, β-Turn
// ============================================================================

/// Normalize a 1-letter or 3-letter amino acid code into canonical 3-letter uppercase (`"ALA"`, `"PRO"`, etc.).
pub fn normalize_amino_acid_code(code: &str) -> Option<&'static str> {
    match code.trim().to_ascii_uppercase().as_str() {
        "A" | "ALA" | "ALANINE" => Some("ALA"),
        "C" | "CYS" | "CYSTEINE" => Some("CYS"),
        "D" | "ASP" | "ASPARTATE" | "ASPARTIC ACID" => Some("ASP"),
        "E" | "GLU" | "GLUTAMATE" | "GLUTAMIC ACID" => Some("GLU"),
        "F" | "PHE" | "PHENYLALANINE" => Some("PHE"),
        "G" | "GLY" | "GLYCINE" => Some("GLY"),
        "H" | "HIS" | "HISTIDINE" => Some("HIS"),
        "I" | "ILE" | "ISOLEUCINE" => Some("ILE"),
        "K" | "LYS" | "LYSINE" => Some("LYS"),
        "L" | "LEU" | "LEUCINE" => Some("LEU"),
        "M" | "MET" | "METHIONINE" => Some("MET"),
        "N" | "ASN" | "ASPARAGINE" => Some("ASN"),
        "P" | "PRO" | "PROLINE" => Some("PRO"),
        "Q" | "GLN" | "GLUTAMINE" => Some("GLN"),
        "R" | "ARG" | "ARGININE" => Some("ARG"),
        "S" | "SER" | "SERINE" => Some("SER"),
        "T" | "THR" | "THREONINE" => Some("THR"),
        "V" | "VAL" | "VALINE" => Some("VAL"),
        "W" | "TRP" | "TRYPTOPHAN" => Some("TRP"),
        "Y" | "TYR" | "TYROSINE" => Some("TYR"),
        _ => None,
    }
}

/// Parse a user sequence string (either 1-letter `"MQIFVK"` or space/hyphen-separated 3-letter `"Met-Gln-Ile"`
/// or `"ALA GLY PRO"`) into canonical 3-letter residue codes.
pub fn parse_peptide_sequence(seq: &str) -> Vec<&'static str> {
    let trimmed = seq.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.contains('-') || trimmed.contains(' ') || trimmed.contains(',') {
        let mut out = Vec::new();
        for tok in trimmed.split(|c: char| c == '-' || c.is_whitespace() || c == ',') {
            if tok.is_empty() {
                continue;
            }
            if let Some(three) = normalize_amino_acid_code(tok) {
                out.push(three);
            } else {
                for ch in tok.chars() {
                    if let Some(three) = normalize_amino_acid_code(&ch.to_string()) {
                        out.push(three);
                    }
                }
            }
        }
        return out;
    }
    if let Some(single_three) = normalize_amino_acid_code(trimmed) {
        if trimmed.len() == 3 {
            return vec![single_three];
        }
    }
    trimmed
        .chars()
        .filter_map(|ch| normalize_amino_acid_code(&ch.to_string()))
        .collect()
}

/// Attach a canonical 2D sidechain for residue `three_code` onto `ca_idx` pointing along `(dx, dy)`
/// (`dir_sgn = +1.0` for sidechain UP -> L-amino acid wedge; `dir_sgn = -1.0` for sidechain DOWN -> L-amino acid hash!).
fn attach_amino_acid_sidechain(
    scene: &mut RenderScene,
    n_idx: usize,
    ca_idx: usize,
    three_code: &str,
    dir_sgn: f64,
) {
    if three_code == "GLY" {
        return;
    }
    let ca_x = scene.atoms[ca_idx].x;
    let ca_y = scene.atoms[ca_idx].y;

    if three_code == "PRO" {
        // Pyrrolidine 5-membered ring closing back onto backbone N (`n_idx`)!
        let nx = scene.atoms[n_idx].x;
        let ny = scene.atoms[n_idx].y;
        let cb = scene.add_atom(SceneAtom::carbon(ca_x + 0.25, ca_y + dir_sgn * 0.96));
        let cg = scene.add_atom(SceneAtom::carbon(0.5 * (ca_x + nx), ca_y + dir_sgn * 1.42));
        let cd = scene.add_atom(SceneAtom::carbon(nx - 0.22, ny + dir_sgn * 0.96));
        let stereo_bond = if dir_sgn > 0.0 {
            SceneBond::wedge(ca_idx, cb)
        } else {
            SceneBond::hash(ca_idx, cb)
        };
        scene.add_bond(stereo_bond);
        scene.add_bond(SceneBond::single(cb, cg));
        scene.add_bond(SceneBond::single(cg, cd));
        scene.add_bond(SceneBond::single(cd, n_idx));
        return;
    }

    // Cβ stereobond: for an extended all-trans zig-zag backbone running left-to-right,
    // an L-amino acid at an UP-apex Cα (`dir_sgn > 0`) projects Cβ UP-WEDGE toward the viewer;
    // at a DOWN-apex Cα (`dir_sgn < 0`) Cβ projects DOWN-HASH away from the viewer!
    let cb_x = ca_x;
    let cb_y = ca_y + dir_sgn * 1.0;
    let cb = scene.add_atom(SceneAtom::carbon(cb_x, cb_y));
    if dir_sgn > 0.0 {
        scene.add_bond(SceneBond::wedge(ca_idx, cb));
    } else {
        scene.add_bond(SceneBond::hash(ca_idx, cb));
    }

    // Helper to grow zig-zag sidechain steps
    let step_pt = |bx: f64, by: f64, k: usize, side: f64| -> [f64; 2] {
        let ang = if k.is_multiple_of(2) {
            dir_sgn * 60.0_f64.to_radians()
        } else {
            dir_sgn * 90.0_f64.to_radians()
        };
        [bx + side * ang.cos(), by + ang.sin()]
    };

    match three_code {
        "ALA" => {}
        "SER" => {
            let og = scene.add_atom(SceneAtom::new("O", cb_x + 0.866, cb_y + dir_sgn * 0.5));
            scene.add_bond(SceneBond::single(cb, og));
        }
        "CYS" => {
            let sg = scene.add_atom(SceneAtom::new("S", cb_x + 0.866, cb_y + dir_sgn * 0.5));
            scene.add_bond(SceneBond::single(cb, sg));
        }
        "VAL" => {
            let cg1 = scene.add_atom(SceneAtom::carbon(cb_x - 0.866, cb_y + dir_sgn * 0.5));
            let cg2 = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            scene.add_bond(SceneBond::single(cb, cg1));
            scene.add_bond(SceneBond::single(cb, cg2));
        }
        "THR" => {
            let og1 = scene.add_atom(SceneAtom::new("O", cb_x - 0.866, cb_y + dir_sgn * 0.5));
            let cg2 = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            scene.add_bond(SceneBond::wedge(cb, og1));
            scene.add_bond(SceneBond::single(cb, cg2));
        }
        "LEU" => {
            let cg = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let cd1 = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 1.5));
            let cd2 = scene.add_atom(SceneAtom::carbon(cb_x + 1.732, cb_y));
            scene.add_bond(SceneBond::single(cb, cg));
            scene.add_bond(SceneBond::single(cg, cd1));
            scene.add_bond(SceneBond::single(cg, cd2));
        }
        "ILE" => {
            let cg2 = scene.add_atom(SceneAtom::carbon(cb_x - 0.866, cb_y + dir_sgn * 0.5));
            let cg1 = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let cd1 = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 1.5));
            scene.add_bond(SceneBond::hash(cb, cg2));
            scene.add_bond(SceneBond::single(cb, cg1));
            scene.add_bond(SceneBond::single(cg1, cd1));
        }
        "ASP" | "ASN" => {
            let cg = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let od1 = scene.add_atom(SceneAtom::new("O", cb_x + 1.732, cb_y));
            let x2_lbl = if three_code == "ASP" { "O" } else { "N" };
            let od2 = scene.add_atom(SceneAtom::new(x2_lbl, cb_x + 0.866, cb_y + dir_sgn * 1.5));
            scene.add_bond(SceneBond::single(cb, cg));
            scene.add_bond(SceneBond::double(cg, od1));
            scene.add_bond(SceneBond::single(cg, od2));
        }
        "GLU" | "GLN" => {
            let cg = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let cd = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 1.5));
            let oe1 = scene.add_atom(SceneAtom::new("O", cb_x + 0.0, cb_y + dir_sgn * 2.0));
            let x2_lbl = if three_code == "GLU" { "O" } else { "N" };
            let oe2 = scene.add_atom(SceneAtom::new(x2_lbl, cb_x + 1.732, cb_y + dir_sgn * 2.0));
            scene.add_bond(SceneBond::single(cb, cg));
            scene.add_bond(SceneBond::single(cg, cd));
            scene.add_bond(SceneBond::double(cd, oe1));
            scene.add_bond(SceneBond::single(cd, oe2));
        }
        "MET" => {
            let cg = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let sd = scene.add_atom(SceneAtom::new("S", cb_x + 0.866, cb_y + dir_sgn * 1.5));
            let ce = scene.add_atom(SceneAtom::carbon(cb_x + 1.732, cb_y + dir_sgn * 2.0));
            scene.add_bond(SceneBond::single(cb, cg));
            scene.add_bond(SceneBond::single(cg, sd));
            scene.add_bond(SceneBond::single(sd, ce));
        }
        "LYS" => {
            let mut prev = cb;
            let mut px = cb_x;
            let mut py = cb_y;
            for k in 0..4 {
                let [nx, ny] = step_pt(px, py, k, if k % 2 == 0 { 1.0 } else { 0.0 });
                let lbl = if k == 3 { "N" } else { "C" };
                let nxt = scene.add_atom(SceneAtom::new(lbl, nx, ny));
                scene.add_bond(SceneBond::single(prev, nxt));
                prev = nxt;
                px = nx;
                py = ny;
            }
        }
        "ARG" => {
            let cg = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 0.5));
            let cd = scene.add_atom(SceneAtom::carbon(cb_x + 0.866, cb_y + dir_sgn * 1.5));
            let ne = scene.add_atom(SceneAtom::new("N", cb_x + 1.732, cb_y + dir_sgn * 2.0));
            let cz = scene.add_atom(SceneAtom::carbon(cb_x + 1.732, cb_y + dir_sgn * 3.0));
            let nh1 = scene.add_atom(SceneAtom::new("N", cb_x + 0.866, cb_y + dir_sgn * 3.5));
            let nh2 = scene.add_atom(SceneAtom::new("N", cb_x + 2.598, cb_y + dir_sgn * 3.5));
            scene.add_bond(SceneBond::single(cb, cg));
            scene.add_bond(SceneBond::single(cg, cd));
            scene.add_bond(SceneBond::single(cd, ne));
            scene.add_bond(SceneBond::single(ne, cz));
            scene.add_bond(SceneBond::double(cz, nh1));
            scene.add_bond(SceneBond::single(cz, nh2));
        }
        "PHE" | "TYR" => {
            // Phenyl / 4-hydroxyphenyl ring attached at Cγ
            let r = 1.0;
            let ring_cy = cb_y + dir_sgn * 2.0;
            let mut ring_ids = Vec::with_capacity(6);
            for k in 0..6 {
                let ang = -dir_sgn * 0.5 * PI + (k as f64 * PI / 3.0);
                ring_ids.push(scene.add_atom(SceneAtom::carbon(
                    cb_x + r * ang.cos(),
                    ring_cy + r * ang.sin(),
                )));
            }
            scene.add_bond(SceneBond::single(cb, ring_ids[0]));
            for k in 0..6 {
                let ord = if k % 2 == 0 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
                scene.add_bond(SceneBond::new(ring_ids[k], ring_ids[(k + 1) % 6], ord));
            }
            if three_code == "TYR" {
                let para = ring_ids[3];
                let oh = scene.add_atom(SceneAtom::new(
                    "O",
                    scene.atoms[para].x,
                    scene.atoms[para].y + dir_sgn * 1.0,
                ));
                scene.add_bond(SceneBond::single(para, oh));
            }
        }
        "HIS" => {
            // Imidazole 5-ring attached at Cγ
            let r = 1.0 / (2.0 * (PI / 5.0).sin());
            let ring_cy = cb_y + dir_sgn * (1.0 + r);
            let mut rids = Vec::with_capacity(5);
            for k in 0..5 {
                let ang = -dir_sgn * 0.5 * PI + (2.0 * PI * k as f64 / 5.0);
                let lbl = if k == 1 || k == 3 { "N" } else { "C" };
                rids.push(scene.add_atom(SceneAtom::new(
                    lbl,
                    cb_x + r * ang.cos(),
                    ring_cy + r * ang.sin(),
                )));
            }
            scene.add_bond(SceneBond::single(cb, rids[0]));
            for k in 0..5 {
                let ord = if k == 0 || k == 2 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
                scene.add_bond(SceneBond::new(rids[k], rids[(k + 1) % 5], ord));
            }
        }
        "TRP" => {
            // Indole bicyclic sidechain attached at C3 of pyrrole ring
            let mut ind =
                regular_polygon_scene(6, cb_x + 1.45, cb_y + dir_sgn * 2.15, 0.5 * PI, true);
            let pyr = fuse_polygon_on_bond(&mut ind, 4, 5, false);
            if pyr.len() == 5 {
                ind.atoms[pyr[3]].label = "N".to_string();
                let c3_local = pyr[2];
                // Align `c3_local` at `(cb_x, cb_y + dir_sgn * 1.0)`
                let shift_x = cb_x - ind.atoms[c3_local].x;
                let shift_y = (cb_y + dir_sgn * 1.0) - ind.atoms[c3_local].y;
                let base = scene.merge_at(&ind, shift_x, shift_y);
                scene.add_bond(SceneBond::single(cb, base + c3_local));
            }
        }
        _ => {}
    }
}

/// Build an all-*trans* 2D L-polypeptide chemical structure from a slice of 1-letter or 3-letter
/// residue codes (e.g. `&["MET", "GLN", "ILE", "PHE"]` or `&["A", "C", "D"]`), complete with
/// N-terminal `H₂N`, C-terminal `COOH`, trans-amide `C=O`/`NH`, and L-stereocenter wedges/hashes!
pub fn build_peptide_scene(residue_codes: &[&str]) -> RenderScene {
    let parsed: Vec<&'static str> = residue_codes
        .iter()
        .filter_map(|c| normalize_amino_acid_code(c))
        .collect();
    if parsed.is_empty() {
        return RenderScene::new();
    }

    let mut scene = RenderScene::new();
    let dx_step = (30.0_f64.to_radians()).cos(); // 0.8660254
    let dy_step = (30.0_f64.to_radians()).sin(); // 0.5

    let mut cur_x = 0.0;
    let mut cur_y = 0.0;
    let mut prev_c_opt: Option<usize> = None;

    for (ri, &three) in parsed.iter().enumerate() {
        // Alternate residue Cα UP (`ri % 2 == 0`) vs DOWN (`ri % 2 == 1`) along extended β-strand zig-zag!
        let up = if ri % 2 == 0 { 1.0 } else { -1.0 };

        // Backbone N_i
        let n_idx = scene.add_atom(SceneAtom::new("N", cur_x, cur_y));
        if let Some(prev_c) = prev_c_opt {
            scene.add_bond(SceneBond::single(prev_c, n_idx));
        }

        // Backbone Cα_i
        let ca_x = cur_x + dx_step;
        let ca_y = cur_y + up * dy_step;
        let ca_idx = scene.add_atom(SceneAtom::carbon(ca_x, ca_y));
        scene.add_bond(SceneBond::single(n_idx, ca_idx));

        // Backbone carbonyl C'_i
        let c_x = ca_x + dx_step;
        let c_y = ca_y - up * dy_step;
        let c_idx = scene.add_atom(SceneAtom::carbon(c_x, c_y));
        scene.add_bond(SceneBond::single(ca_idx, c_idx));

        // Carbonyl `=O` pointing opposite to Cα (`-up`)
        let o_idx = scene.add_atom(SceneAtom::new("O", c_x, c_y - up * 1.0));
        scene.add_bond(SceneBond::double(c_idx, o_idx));

        // Attach L-amino acid sidechain onto `(n_idx, ca_idx)`
        attach_amino_acid_sidechain(&mut scene, n_idx, ca_idx, three, up);

        // Next amide bond step (`c_idx -> next n_idx`) or C-terminal `-OH`
        if ri + 1 == parsed.len() {
            let oh_idx = scene.add_atom(SceneAtom::new("O", c_x + dx_step, c_y + up * dy_step));
            scene.add_bond(SceneBond::single(c_idx, oh_idx));
        } else {
            cur_x = c_x + dx_step;
            cur_y = c_y + up * dy_step;
            prev_c_opt = Some(c_idx);
        }
    }

    // Center around origin
    let [cx, cy] = scene.atom_centroid();
    scene.translate(-cx, -cy);
    scene
}

fn add_biomolecule_templates(list: &mut Vec<StructureTemplate>) {
    let cat = TemplateCategory::Biomolecules;

    // 1. All 20 standard L-Amino Acids
    let aa_table = [
        ("ALA", "l_alanine", "L-Alanine"),
        ("GLY", "glycine", "Glycine"),
        ("VAL", "l_valine", "L-Valine"),
        ("LEU", "l_leucine", "L-Leucine"),
        ("ILE", "l_isoleucine", "L-Isoleucine"),
        ("PRO", "l_proline", "L-Proline"),
        ("PHE", "l_phenylalanine", "L-Phenylalanine"),
        ("TYR", "l_tyrosine", "L-Tyrosine"),
        ("TRP", "l_tryptophan", "L-Tryptophan"),
        ("SER", "l_serine", "L-Serine"),
        ("THR", "l_threonine", "L-Threonine"),
        ("CYS", "l_cysteine", "L-Cysteine"),
        ("MET", "l_methionine", "L-Methionine"),
        ("ASP", "l_aspartic_acid", "L-Aspartic Acid"),
        ("GLU", "l_glutamic_acid", "L-Glutamic Acid"),
        ("ASN", "l_asparagine", "L-Asparagine"),
        ("GLN", "l_glutamine", "L-Glutamine"),
        ("LYS", "l_lysine", "L-Lysine"),
        ("ARG", "l_arginine", "L-Arginine"),
        ("HIS", "l_histidine", "L-Histidine"),
    ];
    for (three, id, name) in aa_table {
        push_tpl(
            list,
            id,
            name,
            cat,
            &["amino acid", three, id],
            build_peptide_scene(&[three]),
        );
    }

    // 2. α-D-Glucopyranose & β-D-Glucopyranose (Haworth projection with bold front edge!)
    for (is_beta, id, name) in [
        (
            false,
            "alpha_d_glucopyranose_haworth",
            "alpha-D-Glucopyranose (Haworth)",
        ),
        (
            true,
            "beta_d_glucopyranose_haworth",
            "beta-D-Glucopyranose (Haworth)",
        ),
    ] {
        let mut g = RenderScene::new();
        // 6-membered Haworth pyranose hexagon: C1 (right), C2 (bot-right), C3 (bot-left), C4 (left), C5 (top-left), O5 (top-right)
        let pts = [
            [1.15, 0.0],    // 0 C1
            [0.58, -0.78],  // 1 C2
            [-0.58, -0.78], // 2 C3
            [-1.15, 0.0],   // 3 C4
            [-0.58, 0.78],  // 4 C5
            [0.58, 0.78],   // 5 O5
        ];
        for (k, p) in pts.iter().enumerate() {
            let lbl = if k == 5 { "O" } else { "C" };
            g.add_atom(SceneAtom::new(lbl, p[0], p[1]));
        }
        // Ring bonds: front C2-C3 bold, C1-C2 and C3-C4 wedged perspective!
        g.add_bond(SceneBond::wedge(0, 1));
        let mut front = SceneBond::single(1, 2);
        front.stereo = BondStereo::Bold;
        g.add_bond(front);
        g.add_bond(SceneBond::wedge(3, 2));
        g.add_bond(SceneBond::single(3, 4));
        g.add_bond(SceneBond::single(4, 5));
        g.add_bond(SceneBond::single(5, 0));

        // Vertical Haworth substituents: C1 (up if beta, down if alpha), C2 down OH, C3 up OH, C4 down OH, C5 up CH2OH
        let c1_y = if is_beta { 0.88 } else { -0.88 };
        let o1 = g.add_atom(SceneAtom::new("O", 1.15, c1_y));
        let o2 = g.add_atom(SceneAtom::new("O", 0.58, -1.62));
        let o3 = g.add_atom(SceneAtom::new("O", -0.58, 0.05));
        let o4 = g.add_atom(SceneAtom::new("O", -1.15, -0.88));
        let c6 = g.add_atom(SceneAtom::carbon(-0.58, 1.65));
        let o6 = g.add_atom(SceneAtom::new("O", -1.35, 2.10));
        for (u, v) in [(0, o1), (1, o2), (2, o3), (3, o4), (4, c6), (c6, o6)] {
            g.add_bond(SceneBond::single(u, v));
        }
        push_tpl(
            list,
            id,
            name,
            cat,
            &["glucose", "glucopyranose", "haworth", "sugar", "hexose"],
            g,
        );
    }

    // 3. β-D-Glucopyranose (⁴C₁ Chair with all-equatorial substituents!)
    {
        let mut gc = make_chair_cyclohexane_scene(false);
        // Atom 2 (top-right-back) is ring oxygen O5
        gc.atoms[2].label = "O".to_string();
        // Sprout all-equatorial substituents on C1 (3), C2 (4), C3 (5), C4 (0), C5 (1 -> CH2OH)
        for &(ring_i, is_c5) in &[
            (3_usize, false),
            (4, false),
            (5, false),
            (0, false),
            (1, true),
        ] {
            if let Some(sv) = chair_sprout_vectors(&gc, ring_i) {
                let eq = sv.equatorial;
                let px = gc.atoms[ring_i].x + eq[0];
                let py = gc.atoms[ring_i].y + eq[1];
                if is_c5 {
                    let c6 = gc.add_atom(SceneAtom::carbon(px, py));
                    let o6 = gc.add_atom(SceneAtom::new("O", px - 0.5, py + 0.86));
                    gc.add_bond(SceneBond::single(ring_i, c6));
                    gc.add_bond(SceneBond::single(c6, o6));
                } else {
                    let oh = gc.add_atom(SceneAtom::new("O", px, py));
                    gc.add_bond(SceneBond::single(ring_i, oh));
                }
            }
        }
        push_tpl(
            list,
            "beta_d_glucopyranose_chair",
            "beta-D-Glucopyranose (⁴C₁ Chair)",
            cat,
            &["glucose", "chair", "glucopyranose", "sugar"],
            gc,
        );
    }

    // 4. β-D-Ribofuranose (5-membered furanose Haworth ring)
    let make_ribofuranose = |anomeric_label: Option<&str>| -> (RenderScene, usize) {
        let mut r = RenderScene::new();
        let pts = [
            [0.0, 0.85],    // 0 O4 apex
            [1.05, 0.18],   // 1 C1' right
            [0.65, -0.78],  // 2 C2' bot-right
            [-0.65, -0.78], // 3 C3' bot-left
            [-1.05, 0.18],  // 4 C4' left
        ];
        for (k, p) in pts.iter().enumerate() {
            r.add_atom(SceneAtom::new(if k == 0 { "O" } else { "C" }, p[0], p[1]));
        }
        r.add_bond(SceneBond::single(0, 1));
        r.add_bond(SceneBond::wedge(1, 2));
        let mut front = SceneBond::single(2, 3);
        front.stereo = BondStereo::Bold;
        r.add_bond(front);
        r.add_bond(SceneBond::wedge(4, 3));
        r.add_bond(SceneBond::single(4, 0));

        // 2'-OH and 3'-OH down, 5'-CH2OH up, 1' beta-up
        let o2 = r.add_atom(SceneAtom::new("O", 0.65, -1.62));
        let o3 = r.add_atom(SceneAtom::new("O", -0.65, -1.62));
        let c5 = r.add_atom(SceneAtom::carbon(-1.05, 1.08));
        r.add_bond(SceneBond::single(2, o2));
        r.add_bond(SceneBond::single(3, o3));
        r.add_bond(SceneBond::single(4, c5));

        if let Some(base_el) = anomeric_label {
            let n9 = r.add_atom(SceneAtom::new(base_el, 1.05, 1.12));
            r.add_bond(SceneBond::single(1, n9));
            (r, c5)
        } else {
            let o1 = r.add_atom(SceneAtom::new("O", 1.05, 1.05));
            let o5 = r.add_atom(SceneAtom::new("O", -1.85, 1.48));
            r.add_bond(SceneBond::single(1, o1));
            r.add_bond(SceneBond::single(c5, o5));
            (r, c5)
        }
    };

    let (ribo, _) = make_ribofuranose(None);
    push_tpl(
        list,
        "beta_d_ribofuranose",
        "beta-D-Ribofuranose",
        cat,
        &["ribose", "ribofuranose", "sugar", "rna"],
        ribo,
    );

    // 5. Adenosine & ATP (Adenosine 5'-triphosphate)
    for (is_atp, id, name) in [
        (false, "adenosine", "Adenosine (Nucleoside)"),
        (true, "atp", "ATP (Adenosine 5'-Triphosphate)"),
    ] {
        let (mut base_ribo, c5_idx) = make_ribofuranose(Some("N"));
        let n9_idx = base_ribo.atoms.len() - 1;
        // Build compact purine ring directly above N9 (`(1.05, 1.12)`)
        let c8 = base_ribo.add_atom(SceneAtom::carbon(1.88, 1.62));
        let n7 = base_ribo.add_atom(SceneAtom::new("N", 1.62, 2.52));
        let c5a = base_ribo.add_atom(SceneAtom::carbon(0.68, 2.52));
        let c4a = base_ribo.add_atom(SceneAtom::carbon(0.38, 1.62));
        let c6a = base_ribo.add_atom(SceneAtom::carbon(0.12, 3.32));
        let n1a = base_ribo.add_atom(SceneAtom::new("N", -0.82, 3.22));
        let c2a = base_ribo.add_atom(SceneAtom::carbon(-1.12, 2.32));
        let n3a = base_ribo.add_atom(SceneAtom::new("N", -0.52, 1.52));
        let n6a = base_ribo.add_atom(SceneAtom::new("N", 0.50, 4.18));
        for &(u, v, ord) in &[
            (n9_idx, c8, BondOrder::Single),
            (c8, n7, BondOrder::Double),
            (n7, c5a, BondOrder::Single),
            (c5a, c4a, BondOrder::Double),
            (c4a, n9_idx, BondOrder::Single),
            (c5a, c6a, BondOrder::Single),
            (c6a, n1a, BondOrder::Double),
            (n1a, c2a, BondOrder::Single),
            (c2a, n3a, BondOrder::Double),
            (n3a, c4a, BondOrder::Single),
            (c6a, n6a, BondOrder::Single),
        ] {
            base_ribo.add_bond(SceneBond::new(u, v, ord));
        }

        if !is_atp {
            let o5 = base_ribo.add_atom(SceneAtom::new("O", -1.88, 1.48));
            base_ribo.add_bond(SceneBond::single(c5_idx, o5));
        } else {
            // 5'-O-Pα(=O)(OH)-O-Pβ(=O)(OH)-O-Pγ(=O)(OH)2 triphosphate chain!
            let mut prev = c5_idx;
            let mut px = -1.85;
            let py = 1.08;
            for p_idx in 0..3 {
                let o_bridge = base_ribo.add_atom(SceneAtom::new("O", px, py));
                base_ribo.add_bond(SceneBond::single(prev, o_bridge));
                let p_atom = base_ribo.add_atom(SceneAtom::new("P", px - 0.92, py));
                base_ribo.add_bond(SceneBond::single(o_bridge, p_atom));
                let o_up = base_ribo.add_atom(SceneAtom::new("O", px - 0.92, py + 0.88));
                let o_dn = base_ribo.add_atom(SceneAtom::new("O", px - 0.92, py - 0.88));
                base_ribo.add_bond(SceneBond::double(p_atom, o_up));
                base_ribo.add_bond(SceneBond::single(p_atom, o_dn));
                if p_idx == 2 {
                    let o_term = base_ribo.add_atom(SceneAtom::new("O", px - 1.84, py));
                    base_ribo.add_bond(SceneBond::single(p_atom, o_term));
                }
                prev = p_atom;
                px -= 1.84;
            }
        }
        let [cx, cy] = base_ribo.atom_centroid();
        base_ribo.translate(-cx, -cy);
        push_tpl(
            list,
            id,
            name,
            cat,
            &["nucleotide", "nucleoside", "purine", id],
            base_ribo,
        );
    }

    // 6. Peptide Type-I β-Turn (i .. i+3 tetrapeptide reverse turn with 10-membered i C=O ··· H-N i+3 H-bond!)
    {
        let mut bt = RenderScene::new();
        // U-shaped 4-residue backbone trace with dashed H-bond between C=O(i) and HN(i+3)
        let n1 = bt.add_atom(SceneAtom::new("N", -1.8, -1.5));
        let ca1 = bt.add_atom(SceneAtom::carbon(-1.0, -0.8));
        let c1 = bt.add_atom(SceneAtom::carbon(-1.0, 0.2));
        let o1 = bt.add_atom(SceneAtom::new("O", -0.1, 0.55));

        let n2 = bt.add_atom(SceneAtom::new("N", -1.8, 0.85));
        let ca2 = bt.add_atom(SceneAtom::carbon(-1.4, 1.85));
        let c2 = bt.add_atom(SceneAtom::carbon(-0.4, 2.20));
        let o2 = bt.add_atom(SceneAtom::new("O", -0.2, 3.12));

        let n3 = bt.add_atom(SceneAtom::new("N", 0.4, 1.65));
        let ca3 = bt.add_atom(SceneAtom::carbon(1.35, 1.85));
        let c3 = bt.add_atom(SceneAtom::carbon(1.75, 0.90));
        let o3 = bt.add_atom(SceneAtom::new("O", 2.68, 0.85));

        let n4 = bt.add_atom(SceneAtom::new("N", 1.05, 0.20));
        let ca4 = bt.add_atom(SceneAtom::carbon(1.25, -0.80));
        let c4 = bt.add_atom(SceneAtom::with_group("C", "COOH", 2.10, -1.35));

        for &(u, v, ord) in &[
            (n1, ca1, BondOrder::Single),
            (ca1, c1, BondOrder::Single),
            (c1, o1, BondOrder::Double),
            (c1, n2, BondOrder::Single),
            (n2, ca2, BondOrder::Single),
            (ca2, c2, BondOrder::Single),
            (c2, o2, BondOrder::Double),
            (c2, n3, BondOrder::Single),
            (n3, ca3, BondOrder::Single),
            (ca3, c3, BondOrder::Single),
            (c3, o3, BondOrder::Double),
            (c3, n4, BondOrder::Single),
            (n4, ca4, BondOrder::Single),
            (ca4, c4, BondOrder::Single),
            (o1, n4, BondOrder::HydrogenBond), // Classic 1<-4 β-turn H-bond!
        ] {
            bt.add_bond(SceneBond::new(u, v, ord));
        }
        // Proline ring at i+1 (n2-ca2) for canonical Pro-Gly β-turn
        let cb2 = bt.add_atom(SceneAtom::carbon(-2.35, 2.25));
        let cg2 = bt.add_atom(SceneAtom::carbon(-2.85, 1.45));
        bt.add_bond(SceneBond::wedge(ca2, cb2));
        bt.add_bond(SceneBond::single(cb2, cg2));
        bt.add_bond(SceneBond::single(cg2, n2));

        push_tpl(
            list,
            "peptide_beta_turn",
            "Peptide beta-Turn (Type I, i..i+3 H-bond)",
            cat,
            &["peptide", "beta turn", "protein", "secondary structure"],
            bt,
        );
    }
}

// ============================================================================
// 5. Organometallics & Labware (Ferrocene, Square-Planar, Octahedral, TLC, Orbitals)
// ============================================================================

fn add_organometallic_and_labware_templates(list: &mut Vec<StructureTemplate>) {
    let cat = TemplateCategory::OrganometallicsAndLabware;

    // 1. Ferrocene (η⁵-Cp₂Fe sandwich with tilted perspective pentagons & multi-center η⁵ bonds!)
    {
        let mut fc = RenderScene::new();
        let make_tilted_cp = |fc: &mut RenderScene, yc: f64| -> (Vec<usize>, usize) {
            let rx = 1.05;
            let ry = 0.42;
            let mut ids = Vec::with_capacity(5);
            for k in 0..5 {
                let ang = -0.5 * PI + (2.0 * PI * k as f64 / 5.0);
                ids.push(fc.add_atom(SceneAtom::carbon(rx * ang.cos(), yc + ry * ang.sin())));
            }
            // Front two edges wedged/bold for crisp 3D sandwich perspective
            fc.add_bond(SceneBond::wedge(ids[4], ids[0]));
            fc.add_bond(SceneBond::wedge(ids[1], ids[0]));
            for &(u, v) in &[(1, 2), (2, 3), (3, 4)] {
                fc.add_bond(SceneBond::single(ids[u], ids[v]));
            }
            fc.aromatic_circles.push(ids.clone());
            let mut mc = SceneAtom::carbon(0.0, yc);
            mc.multi_center_atoms = ids.clone();
            let mc_idx = fc.add_atom(mc);
            (ids, mc_idx)
        };

        let (_, top_mc) = make_tilted_cp(&mut fc, 1.15);
        let fe = fc.add_atom(SceneAtom::new("Fe", 0.0, 0.0));
        let (_, bot_mc) = make_tilted_cp(&mut fc, -1.15);
        // Connect Fe to top & bottom Cp centroids (drawn so top Cp ring sits visually above!)
        let mut b_top = SceneBond::single(fe, top_mc);
        b_top.stereo = BondStereo::Dashed;
        let mut b_bot = SceneBond::single(fe, bot_mc);
        b_bot.stereo = BondStereo::Dashed;
        fc.add_bond(b_top);
        fc.add_bond(b_bot);

        push_tpl(
            list,
            "ferrocene",
            "Ferrocene (eta5-Cp2Fe)",
            cat,
            &["organometallic", "metallocene", "sandwich", "iron", "cp"],
            fc,
        );
    }

    // 2. Square-Planar Transition-Metal Complex (`cis-[PtCl2(NH3)2]` / `PdL2Cl2`)
    {
        let mut sq = RenderScene::new();
        let pd = sq.add_atom(SceneAtom::new("Pd", 0.0, 0.0));
        let l1 = sq.add_atom(SceneAtom::with_group("P", "PPh3", -0.95, 0.68));
        let l2 = sq.add_atom(SceneAtom::with_group("P", "PPh3", 0.95, 0.68));
        let cl1 = sq.add_atom(SceneAtom::new("Cl", -0.95, -0.68));
        let cl2 = sq.add_atom(SceneAtom::new("Cl", 0.95, -0.68));
        sq.add_bond(SceneBond::hash(pd, l1));
        sq.add_bond(SceneBond::hash(pd, l2));
        sq.add_bond(SceneBond::wedge(pd, cl1));
        sq.add_bond(SceneBond::wedge(pd, cl2));
        push_tpl(
            list,
            "square_planar_complex",
            "Square-Planar Complex [PdCl2(PPh3)2]",
            cat,
            &[
                "square planar",
                "palladium",
                "platinum",
                "cisplatin",
                "organometallic",
            ],
            sq,
        );
    }

    // 3. Octahedral Transition-Metal Complex (`[RuL6]` / `[ML6]` with 2 axial + 4 wedged/hashed equatorial ligands)
    {
        let mut oct = RenderScene::new();
        let m = oct.add_atom(SceneAtom::new("Ru", 0.0, 0.0));
        let ax_top = oct.add_atom(SceneAtom::new("L", 0.0, 1.15));
        let ax_bot = oct.add_atom(SceneAtom::new("L", 0.0, -1.15));
        let eq_fl = oct.add_atom(SceneAtom::new("L", -0.98, -0.42));
        let eq_fr = oct.add_atom(SceneAtom::new("L", 0.98, -0.42));
        let eq_bl = oct.add_atom(SceneAtom::new("L", -0.98, 0.42));
        let eq_br = oct.add_atom(SceneAtom::new("L", 0.98, 0.42));
        oct.add_bond(SceneBond::single(m, ax_top));
        oct.add_bond(SceneBond::single(m, ax_bot));
        oct.add_bond(SceneBond::wedge(m, eq_fl));
        oct.add_bond(SceneBond::wedge(m, eq_fr));
        oct.add_bond(SceneBond::hash(m, eq_bl));
        oct.add_bond(SceneBond::hash(m, eq_br));
        push_tpl(
            list,
            "octahedral_complex",
            "Octahedral Metal Complex [ML6]",
            cat,
            &[
                "octahedral",
                "coordination",
                "transition metal",
                "organometallic",
            ],
            oct,
        );
    }

    // 4. Ready-to-stamp TLC Plate (SM, Co-spot, Rxn lanes)
    {
        let mut tlc_scene = RenderScene::new();
        tlc_scene.tlc_plates.push(SceneTlcPlate {
            origin: [-1.6, -2.2],
            width_mol: 3.2,
            height_mol: 4.4,
            lane_labels: vec!["SM".to_string(), "Co".to_string(), "Rxn".to_string()],
            baseline_frac: 0.14,
            solvent_front_frac: 0.88,
            show_rf_scale: true,
            spots: vec![
                TlcSpot {
                    lane: 0,
                    rf: 0.28,
                    radius_mol: 0.16,
                    tailing: 1.0,
                    style: TlcSpotStyle::Solid,
                    label: None,
                    color: None,
                },
                TlcSpot {
                    lane: 1,
                    rf: 0.28,
                    radius_mol: 0.15,
                    tailing: 1.0,
                    style: TlcSpotStyle::Hollow,
                    label: None,
                    color: None,
                },
                TlcSpot {
                    lane: 1,
                    rf: 0.68,
                    radius_mol: 0.16,
                    tailing: 1.0,
                    style: TlcSpotStyle::Solid,
                    label: None,
                    color: None,
                },
                TlcSpot {
                    lane: 2,
                    rf: 0.68,
                    radius_mol: 0.18,
                    tailing: 1.05,
                    style: TlcSpotStyle::UvActive,
                    label: Some("Prod".to_string()),
                    color: None,
                },
            ],
        });
        push_tpl(
            list,
            "tlc_plate",
            "TLC Plate (SM / Co-spot / Rxn)",
            cat,
            &["tlc", "chromatography", "labware", "rf"],
            tlc_scene,
        );
    }

    // 5. Atomic & Hybrid Orbital Lobe Stamps (`s`, `p`, `d`, `dz2`, `sp3`)
    for (id, name, otype, ang) in [
        ("orbital_s", "s Orbital", OrbitalType::S, 90.0),
        ("orbital_p", "p Orbital (2p)", OrbitalType::P, 90.0),
        (
            "orbital_d",
            "d Orbital (3d Clover)",
            OrbitalType::DClover,
            45.0,
        ),
        (
            "orbital_dz2",
            "dz2 Orbital (Lobes + Torus)",
            OrbitalType::Dz2,
            90.0,
        ),
        (
            "orbital_sp3",
            "sp3 Hybrid Orbital Lobe",
            OrbitalType::HybridSp3,
            90.0,
        ),
    ] {
        let mut os = RenderScene::new();
        os.orbitals.push(SceneOrbital {
            center: [0.0, 0.0],
            size_mol: 1.15,
            angle_deg: ang,
            orbital_type: otype,
            shading: OrbitalShading::ShadedAndOpen,
            color: None,
        });
        push_tpl(list, id, name, cat, &["orbital", "mo", "lobe", id], os);
    }
}
