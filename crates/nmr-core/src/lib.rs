//! Clean-room NMR resonance assignment, raw FID & spectrum I/O, classical & neural peak picking,
//! sidechain assignment, 3D NOESY / torsion structure folding, NEF v1.1 interchange,
//! HTML/SVG quality reporting, HTTP job service, and Octet-Draw 2D chemical structure,
//! reaction scheme & 1D NMR cheminformatics core.
//!
//! Provenance: Original implementation of `docs/specs/00-architecture.md` through
//! `docs/specs/11-fid-processing.md`, plus small-molecule cheminformatics, analytical 1D NMR,
//! and publication vector rendering ported/adapted from Octet-Draw by Sam Ellis
//! (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0).

/// Canonical SemVer release version inherited from `[workspace.package]` in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod aromatic;
pub mod c13_detected;
pub mod classify;
pub mod dihedrals;
pub mod evaluate;
pub mod evo_graph;
pub mod fid;
pub mod idp_propensity;
pub mod idp_solver;
pub mod methyl_solver;
pub mod methyl_trosy;
pub mod nef;
pub mod neural_picker;
pub mod nmrstar;
pub mod noesy;
pub mod octet_analyt;
pub mod octet_mol;
pub mod octet_render;
pub mod ops;
pub mod paramagnetic;
pub mod peaks;
pub mod picker;
pub mod priors;
pub mod process;
pub mod qc;
pub mod report;
pub mod rng;
pub mod service;
pub mod session;
pub mod sidechain;
pub mod simulate;
pub mod solver;
pub mod spectrum;
pub mod spinsystems;
pub mod stereospec;
pub mod structure;
pub mod topology;

// ============================================================================
// Seamless Interoperability Bridges: `octet_mol` <-> `octet_analyt` <-> `octet_render`
// ============================================================================

impl From<&octet_mol::Molecule> for octet_analyt::AnalytMol {
    fn from(mol: &octet_mol::Molecule) -> Self {
        let mut work = mol.clone();
        if work.atoms.iter().any(|a| a.superatom_label.is_some()) {
            let _ = octet_mol::expand_abbreviations(&mut work);
        }

        let implicit_hs: Vec<u8> = (0..work.atoms.len())
            .map(|idx| work.implicit_hydrogens(idx))
            .collect();
        let orig_arom_bonds: Vec<bool> = work
            .bonds
            .iter()
            .map(|b| b.kind == octet_mol::BondKind::Aromatic)
            .collect();
        let _ = octet_mol::kekulize(&mut work);
        let arom = octet_mol::perceive_aromaticity(&work);
        let cip = octet_mol::perceive_cip(&work);

        let mut atoms = Vec::with_capacity(work.atoms.len());
        for (idx, a) in work.atoms.iter().enumerate() {
            let stereo_tag = match cip.atoms.get(&idx) {
                Some(octet_mol::CipAtomDescriptor::R | octet_mol::CipAtomDescriptor::PseudoR) => {
                    octet_analyt::AtomStereoTag::R
                }
                Some(octet_mol::CipAtomDescriptor::S | octet_mol::CipAtomDescriptor::PseudoS) => {
                    octet_analyt::AtomStereoTag::S
                }
                None => octet_analyt::AtomStereoTag::None,
            };
            let el = if a.z == 0 {
                "*".to_string()
            } else {
                octet_analyt::normalize_element_symbol(&a.symbol)
            };
            atoms.push(octet_analyt::AnalytAtom {
                element: el,
                isotope: a.isotope,
                formal_charge: a.formal_charge,
                implicit_h: Some(implicit_hs[idx]),
                x: a.x,
                y: a.y,
                stereo_tag,
                smiles_neighbor_order: Vec::new(),
            });
        }

        let mut bonds = Vec::with_capacity(work.bonds.len());
        for (bi, b) in work.bonds.iter().enumerate() {
            let is_arom = orig_arom_bonds.get(bi).copied().unwrap_or(false)
                || arom.aromatic_bonds.get(bi).copied().unwrap_or(false);
            let order = b.kind.order_u8();
            let stereo = match b.kind {
                octet_mol::BondKind::WedgeUp | octet_mol::BondKind::HollowWedge => {
                    octet_analyt::BondStereoTag::Up
                }
                octet_mol::BondKind::HashDown => octet_analyt::BondStereoTag::Down,
                octet_mol::BondKind::Wavy => octet_analyt::BondStereoTag::Either,
                octet_mol::BondKind::Double => match b.double_geometry {
                    octet_mol::DoubleBondGeometry::Either => octet_analyt::BondStereoTag::Either,
                    octet_mol::DoubleBondGeometry::Cis { .. } => octet_analyt::BondStereoTag::Cis,
                    octet_mol::DoubleBondGeometry::Trans { .. } => {
                        octet_analyt::BondStereoTag::Trans
                    }
                    octet_mol::DoubleBondGeometry::Auto => match cip.bonds.get(&bi) {
                        Some(octet_mol::CipBondDescriptor::Z) => octet_analyt::BondStereoTag::Cis,
                        Some(octet_mol::CipBondDescriptor::E) => octet_analyt::BondStereoTag::Trans,
                        None => octet_analyt::BondStereoTag::None,
                    },
                },
                _ => octet_analyt::BondStereoTag::None,
            };
            bonds.push(octet_analyt::AnalytBond {
                a: b.a,
                b: b.b,
                order,
                aromatic: is_arom,
                stereo,
            });
        }

        octet_analyt::AnalytMol {
            atoms,
            bonds,
            name: (!work.name.is_empty()).then(|| work.name.clone()),
        }
    }
}

impl From<octet_mol::Molecule> for octet_analyt::AnalytMol {
    fn from(mol: octet_mol::Molecule) -> Self {
        Self::from(&mol)
    }
}

impl From<&octet_mol::ChemDocument> for octet_analyt::AnalytMol {
    fn from(doc: &octet_mol::ChemDocument) -> Self {
        let merged = doc.merged_molecule();
        Self::from(&merged)
    }
}

impl From<&octet_mol::Molecule> for octet_render::RenderScene {
    fn from(mol: &octet_mol::Molecule) -> Self {
        let scale = octet_mol::ACS_BOND_LENGTH_ANGSTROM;
        let cip = octet_mol::perceive_cip(mol);
        let mut scene = octet_render::RenderScene::new();

        for (idx, a) in mol.atoms.iter().enumerate() {
            let rad_u8 = match a.radical {
                octet_mol::Radical::None => 0,
                octet_mol::Radical::Doublet => 1,
                octet_mol::Radical::Singlet | octet_mol::Radical::Triplet => 2,
            };
            let cip_label = cip.atoms.get(&idx).map(|d| format!("({})", d.as_str()));
            let label = if a.z == 0 && a.superatom_label.is_some() {
                "C".to_string()
            } else {
                a.symbol.clone()
            };
            scene.atoms.push(octet_render::SceneAtom {
                label,
                text: a.superatom_label.clone(),
                x: a.x / scale,
                y: a.y / scale,
                z: a.z_coord.unwrap_or(0.0) / scale,
                charge: a.formal_charge,
                radical: rad_u8,
                isotope: a.isotope,
                explicit_h: a.explicit_hydrogens,
                show_carbon: false,
                cip_label,
                lone_pairs: 0,
                color: None,
                highlight: None,
                multi_center_atoms: a.multi_center_atoms.clone(),
            });
        }

        for (bi, b) in mol.bonds.iter().enumerate() {
            let (order, stereo) = match b.kind {
                octet_mol::BondKind::Single => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::None,
                ),
                octet_mol::BondKind::Double => {
                    let st = if b.double_geometry == octet_mol::DoubleBondGeometry::Either {
                        octet_render::BondStereo::CrossedDouble
                    } else {
                        octet_render::BondStereo::None
                    };
                    (octet_render::BondOrder::Double, st)
                }
                octet_mol::BondKind::Triple => (
                    octet_render::BondOrder::Triple,
                    octet_render::BondStereo::None,
                ),
                octet_mol::BondKind::Aromatic => (
                    octet_render::BondOrder::Aromatic,
                    octet_render::BondStereo::None,
                ),
                octet_mol::BondKind::WedgeUp => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::UpWedge,
                ),
                octet_mol::BondKind::HashDown => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::DownHash,
                ),
                octet_mol::BondKind::HollowWedge => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::UpHollow,
                ),
                octet_mol::BondKind::Wavy => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::Wavy,
                ),
                octet_mol::BondKind::Bold => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::Bold,
                ),
                octet_mol::BondKind::Dashed | octet_mol::BondKind::Ionic => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::Dashed,
                ),
                octet_mol::BondKind::Dative => (
                    octet_render::BondOrder::Single,
                    octet_render::BondStereo::Dative,
                ),
                octet_mol::BondKind::Hydrogen => (
                    octet_render::BondOrder::HydrogenBond,
                    octet_render::BondStereo::Dashed,
                ),
            };
            let ez_label = cip.bonds.get(&bi).map(|d| format!("({})", d.as_str()));
            scene.bonds.push(octet_render::SceneBond {
                a: b.a,
                b: b.b,
                order,
                stereo,
                side: octet_render::DoubleBondSide::Auto,
                ez_label,
                color: None,
                highlight: None,
            });
        }

        scene
    }
}

impl From<&octet_render::RenderScene> for octet_mol::Molecule {
    fn from(scene: &octet_render::RenderScene) -> Self {
        let scale = octet_mol::ACS_BOND_LENGTH_ANGSTROM;
        let mut mol = octet_mol::Molecule::new();

        for sa in &scene.atoms {
            let mut atom = if let Some(group_txt) = &sa.text {
                octet_mol::Atom::new_superatom(group_txt)
            } else if !sa.multi_center_atoms.is_empty() {
                let mut mc = octet_mol::Atom::new("*");
                mc.multi_center_atoms = sa.multi_center_atoms.clone();
                mc
            } else {
                octet_mol::Atom::new(&sa.label)
            };
            atom.x = sa.x * scale;
            atom.y = sa.y * scale;
            if sa.z.abs() > 1e-9 {
                atom.z_coord = Some(sa.z * scale);
            }
            atom.formal_charge = sa.charge;
            atom.radical = match sa.radical {
                0 => octet_mol::Radical::None,
                1 => octet_mol::Radical::Doublet,
                _ => octet_mol::Radical::Triplet,
            };
            atom.isotope = sa.isotope;
            atom.explicit_hydrogens = sa.explicit_h;
            mol.atoms.push(atom);
        }

        for sb in &scene.bonds {
            let (kind, double_geometry) = match (sb.order, sb.stereo) {
                (octet_render::BondOrder::HydrogenBond, _) => (
                    octet_mol::BondKind::Hydrogen,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Double, octet_render::BondStereo::CrossedDouble) => (
                    octet_mol::BondKind::Double,
                    octet_mol::DoubleBondGeometry::Either,
                ),
                (octet_render::BondOrder::Double, _) => (
                    octet_mol::BondKind::Double,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Triple | octet_render::BondOrder::Quadruple, _) => (
                    octet_mol::BondKind::Triple,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Aromatic, _) => (
                    octet_mol::BondKind::Aromatic,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::UpWedge) => (
                    octet_mol::BondKind::WedgeUp,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::DownHash) => (
                    octet_mol::BondKind::HashDown,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::UpHollow) => (
                    octet_mol::BondKind::HollowWedge,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::Wavy) => (
                    octet_mol::BondKind::Wavy,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::Dative) => (
                    octet_mol::BondKind::Dative,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::Bold) => (
                    octet_mol::BondKind::Bold,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, octet_render::BondStereo::Dashed) => (
                    octet_mol::BondKind::Dashed,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
                (octet_render::BondOrder::Single, _) => (
                    octet_mol::BondKind::Single,
                    octet_mol::DoubleBondGeometry::Auto,
                ),
            };
            mol.bonds.push(octet_mol::Bond {
                a: sb.a,
                b: sb.b,
                kind,
                double_geometry,
            });
        }

        mol
    }
}

impl From<&octet_mol::ChemDocument> for octet_render::RenderScene {
    fn from(doc: &octet_mol::ChemDocument) -> Self {
        let scale = octet_mol::ACS_BOND_LENGTH_ANGSTROM;
        let merged = doc.merged_molecule();
        let mut scene = octet_render::RenderScene::from(&merged);

        for arr in &doc.reaction_arrows {
            let kind = match arr.kind {
                octet_mol::ReactionArrowKind::Solid => octet_render::ArrowKind::Solid,
                octet_mol::ReactionArrowKind::Equilibrium => octet_render::ArrowKind::Equilibrium,
                octet_mol::ReactionArrowKind::UnbalancedEquilibrium => {
                    octet_render::ArrowKind::UnbalancedEquilibrium
                }
                octet_mol::ReactionArrowKind::Resonance => octet_render::ArrowKind::Resonance,
                octet_mol::ReactionArrowKind::Retrosynthetic => {
                    octet_render::ArrowKind::Retrosynthetic
                }
                octet_mol::ReactionArrowKind::NoGoCross
                | octet_mol::ReactionArrowKind::NoGoHash => octet_render::ArrowKind::NoGo,
                octet_mol::ReactionArrowKind::Dashed => octet_render::ArrowKind::Dashed,
            };
            scene
                .reaction_arrows
                .push(octet_render::SceneReactionArrow {
                    from: [arr.tail.0 / scale, arr.tail.1 / scale],
                    to: [arr.head.0 / scale, arr.head.1 / scale],
                    kind,
                    above_text: arr.above_text.clone(),
                    below_text: arr.below_text.clone(),
                    color: None,
                });
        }

        for ca in &doc.curved_arrows {
            let s = [ca.start.0 / scale, ca.start.1 / scale];
            let c = [ca.control.0 / scale, ca.control.1 / scale];
            let e = [ca.end.0 / scale, ca.end.1 / scale];
            let ctrl1 = [
                s[0] + (2.0 / 3.0) * (c[0] - s[0]),
                s[1] + (2.0 / 3.0) * (c[1] - s[1]),
            ];
            let ctrl2 = [
                e[0] + (2.0 / 3.0) * (c[0] - e[0]),
                e[1] + (2.0 / 3.0) * (c[1] - e[1]),
            ];
            let kind = if ca.electrons == 1 {
                octet_render::PushingArrowKind::FishhookOneElectron
            } else {
                octet_render::PushingArrowKind::FullTwoElectron
            };
            scene.curved_arrows.push(octet_render::SceneCurvedArrow {
                start: s,
                ctrl1,
                ctrl2,
                end: e,
                kind,
                fishhook_left: true,
                dashed: false,
                color: None,
            });
        }

        for br in &doc.brackets {
            let style = match br.style {
                octet_mol::BracketStyle::Square => octet_render::BracketStyle::Square,
                octet_mol::BracketStyle::Round => octet_render::BracketStyle::Parenthesis,
                octet_mol::BracketStyle::Brace => octet_render::BracketStyle::CurlyBrace,
            };
            scene.brackets.push(octet_render::SceneBracket {
                min: [br.bounds.0 / scale, br.bounds.1 / scale],
                max: [br.bounds.2 / scale, br.bounds.3 / scale],
                style,
                subscript: (!br.subscript.is_empty()).then(|| br.subscript.clone()),
                superscript: br.superscript.clone(),
                color: None,
            });
        }

        for orb in &doc.orbitals {
            let cx = orb.center.0 / scale;
            let cy = orb.center.1 / scale;
            let ex = orb.axis_end.0 / scale;
            let ey = orb.axis_end.1 / scale;
            let dx = ex - cx;
            let dy = ey - cy;
            let size_mol = dx.hypot(dy).max(0.5);
            let angle_deg = dy.atan2(dx).to_degrees();
            let orbital_type = match orb.kind {
                octet_mol::OrbitalKind::S => octet_render::OrbitalType::S,
                octet_mol::OrbitalKind::P => octet_render::OrbitalType::P,
                octet_mol::OrbitalKind::Sp3 => octet_render::OrbitalType::HybridSp3,
                octet_mol::OrbitalKind::Dz2 => octet_render::OrbitalType::Dz2,
                octet_mol::OrbitalKind::Dxy => octet_render::OrbitalType::DClover,
            };
            let shading = match orb.phase {
                octet_mol::OrbitalPhase::FilledPlus => octet_render::OrbitalShading::ShadedAndOpen,
                octet_mol::OrbitalPhase::ShadedMinus => {
                    octet_render::OrbitalShading::DualPhaseColored
                }
                octet_mol::OrbitalPhase::Hollow => octet_render::OrbitalShading::OutlineOnly,
            };
            scene.orbitals.push(octet_render::SceneOrbital {
                center: [cx, cy],
                size_mol,
                angle_deg,
                orbital_type,
                shading,
                color: None,
            });
        }

        for tlc in &doc.tlc_plates {
            let min_x = tlc.bounds.0 / scale;
            let min_y = tlc.bounds.1 / scale;
            let max_x = tlc.bounds.2 / scale;
            let max_y = tlc.bounds.3 / scale;
            let spots = tlc
                .spots
                .iter()
                .map(|sp| octet_render::TlcSpot {
                    lane: sp.lane,
                    rf: sp.rf,
                    radius_mol: 0.18,
                    tailing: 1.0,
                    style: match sp.style {
                        octet_mol::TlcSpotStyle::Solid => octet_render::TlcSpotStyle::Solid,
                        octet_mol::TlcSpotStyle::Hollow => octet_render::TlcSpotStyle::Hollow,
                        octet_mol::TlcSpotStyle::Shaded => octet_render::TlcSpotStyle::Faint,
                        octet_mol::TlcSpotStyle::UvActive => octet_render::TlcSpotStyle::UvActive,
                    },
                    label: sp.label.clone(),
                    color: None,
                })
                .collect();
            scene.tlc_plates.push(octet_render::SceneTlcPlate {
                origin: [min_x, min_y],
                width_mol: (max_x - min_x).max(1.0),
                height_mol: (max_y - min_y).max(2.0),
                lane_labels: tlc.lane_labels.clone(),
                baseline_frac: tlc.origin_fraction,
                solvent_front_frac: (1.0 - tlc.solvent_front_fraction).clamp(0.5, 0.95),
                show_rf_scale: true,
                spots,
            });
        }

        for tb in &doc.text_boxes {
            let bold = tb.spans.iter().any(|s| s.bold);
            let italic = tb.spans.iter().any(|s| s.italic);
            let font_size_pt = tb.spans.iter().find_map(|s| s.font_size_pt);
            scene.texts.push(octet_render::SceneText {
                pos: [tb.position.0 / scale, tb.position.1 / scale],
                text: tb.plain_text(),
                font_size_pt,
                bold,
                italic,
                centered: false,
                color: None,
            });
        }

        scene
    }
}

impl From<&octet_render::RenderScene> for octet_mol::ChemDocument {
    fn from(scene: &octet_render::RenderScene) -> Self {
        let scale = octet_mol::ACS_BOND_LENGTH_ANGSTROM;
        let merged = octet_mol::Molecule::from(scene);
        let molecules = if merged.atoms.is_empty() {
            Vec::new()
        } else {
            merged.split_fragments()
        };

        let reaction_arrows = scene
            .reaction_arrows
            .iter()
            .map(|a| octet_mol::ReactionArrow {
                kind: match a.kind {
                    octet_render::ArrowKind::Solid => octet_mol::ReactionArrowKind::Solid,
                    octet_render::ArrowKind::Equilibrium => {
                        octet_mol::ReactionArrowKind::Equilibrium
                    }
                    octet_render::ArrowKind::UnbalancedEquilibrium => {
                        octet_mol::ReactionArrowKind::UnbalancedEquilibrium
                    }
                    octet_render::ArrowKind::Resonance => octet_mol::ReactionArrowKind::Resonance,
                    octet_render::ArrowKind::Retrosynthetic => {
                        octet_mol::ReactionArrowKind::Retrosynthetic
                    }
                    octet_render::ArrowKind::NoGo => octet_mol::ReactionArrowKind::NoGoCross,
                    octet_render::ArrowKind::Dashed => octet_mol::ReactionArrowKind::Dashed,
                },
                tail: (a.from[0] * scale, a.from[1] * scale),
                head: (a.to[0] * scale, a.to[1] * scale),
                above_text: a.above_text.clone(),
                below_text: a.below_text.clone(),
            })
            .collect();

        let curved_arrows = scene
            .curved_arrows
            .iter()
            .map(|ca| {
                let ctrl_mid_x = 0.5 * (ca.ctrl1[0] + ca.ctrl2[0]) * scale;
                let ctrl_mid_y = 0.5 * (ca.ctrl1[1] + ca.ctrl2[1]) * scale;
                octet_mol::CurvedArrow {
                    electrons: match ca.kind {
                        octet_render::PushingArrowKind::FishhookOneElectron => 1,
                        octet_render::PushingArrowKind::FullTwoElectron => 2,
                    },
                    start: (ca.start[0] * scale, ca.start[1] * scale),
                    control: (ctrl_mid_x, ctrl_mid_y),
                    end: (ca.end[0] * scale, ca.end[1] * scale),
                    source_anchor: None,
                    target_anchor: None,
                }
            })
            .collect();

        let brackets = scene
            .brackets
            .iter()
            .map(|b| octet_mol::Bracket {
                style: match b.style {
                    octet_render::BracketStyle::Square => octet_mol::BracketStyle::Square,
                    octet_render::BracketStyle::Parenthesis => octet_mol::BracketStyle::Round,
                    octet_render::BracketStyle::CurlyBrace => octet_mol::BracketStyle::Brace,
                },
                usage: octet_mol::BracketUsage::Sru,
                bounds: (
                    b.min[0] * scale,
                    b.min[1] * scale,
                    b.max[0] * scale,
                    b.max[1] * scale,
                ),
                subscript: b.subscript.clone().unwrap_or_default(),
                superscript: b.superscript.clone(),
                enclosed_atoms: Vec::new(),
            })
            .collect();

        let orbitals = scene
            .orbitals
            .iter()
            .map(|o| {
                let rad = o.angle_deg.to_radians();
                let cx = o.center[0] * scale;
                let cy = o.center[1] * scale;
                let r = o.size_mol * scale;
                octet_mol::Orbital {
                    kind: match o.orbital_type {
                        octet_render::OrbitalType::S => octet_mol::OrbitalKind::S,
                        octet_render::OrbitalType::P => octet_mol::OrbitalKind::P,
                        octet_render::OrbitalType::HybridSp3 => octet_mol::OrbitalKind::Sp3,
                        octet_render::OrbitalType::Dz2 => octet_mol::OrbitalKind::Dz2,
                        octet_render::OrbitalType::DClover => octet_mol::OrbitalKind::Dxy,
                    },
                    phase: match o.shading {
                        octet_render::OrbitalShading::ShadedAndOpen => {
                            octet_mol::OrbitalPhase::FilledPlus
                        }
                        octet_render::OrbitalShading::DualPhaseColored => {
                            octet_mol::OrbitalPhase::ShadedMinus
                        }
                        octet_render::OrbitalShading::OutlineOnly => {
                            octet_mol::OrbitalPhase::Hollow
                        }
                    },
                    center: (cx, cy),
                    axis_end: (cx + r * rad.cos(), cy + r * rad.sin()),
                    atom_anchor: None,
                }
            })
            .collect();

        let tlc_plates = scene
            .tlc_plates
            .iter()
            .map(|p| {
                let min_x = p.origin[0] * scale;
                let min_y = p.origin[1] * scale;
                let max_x = (p.origin[0] + p.width_mol) * scale;
                let max_y = (p.origin[1] + p.height_mol) * scale;
                octet_mol::TlcPlate {
                    bounds: (min_x, min_y, max_x, max_y),
                    origin_fraction: p.baseline_frac,
                    solvent_front_fraction: (1.0 - p.solvent_front_frac).clamp(0.05, 0.5),
                    lane_labels: p.lane_labels.clone(),
                    spots: p
                        .spots
                        .iter()
                        .map(|sp| octet_mol::TlcSpot {
                            lane: sp.lane,
                            rf: sp.rf,
                            style: match sp.style {
                                octet_render::TlcSpotStyle::Solid => octet_mol::TlcSpotStyle::Solid,
                                octet_render::TlcSpotStyle::Hollow => {
                                    octet_mol::TlcSpotStyle::Hollow
                                }
                                octet_render::TlcSpotStyle::UvActive => {
                                    octet_mol::TlcSpotStyle::UvActive
                                }
                                octet_render::TlcSpotStyle::Faint => {
                                    octet_mol::TlcSpotStyle::Shaded
                                }
                            },
                            label: sp.label.clone(),
                        })
                        .collect(),
                }
            })
            .collect();

        let text_boxes = scene
            .texts
            .iter()
            .map(|t| octet_mol::RichTextBox {
                position: (t.pos[0] * scale, t.pos[1] * scale),
                spans: vec![octet_mol::TextSpan {
                    text: t.text.clone(),
                    bold: t.bold,
                    italic: t.italic,
                    subscript: false,
                    superscript: false,
                    font_size_pt: t.font_size_pt,
                    color_rgb: None,
                }],
            })
            .collect();

        octet_mol::ChemDocument {
            title: String::new(),
            molecules,
            reaction_arrows,
            curved_arrows,
            brackets,
            orbitals,
            tlc_plates,
            text_boxes,
            shapes: Vec::new(),
        }
    }
}
