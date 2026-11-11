//! Comprehensive integration & verification tests for `octet_render` (`styles`, `svg_pdf`, `templates`).
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! Copyright (c) 2026 Sam Ellis.

pub use nmr_core::*;

use octet_render::*;

#[test]
fn test_journal_style_presets_and_scaling() {
    assert_eq!(JournalStyle::ALL.len(), 7);
    for &preset in JournalStyle::ALL {
        let style = preset.to_style();
        assert_eq!(style.preset, preset);
        assert!(style.bond_length_pt > 8.0);
        assert!(style.line_width_pt > 0.3);
        assert!(style.bold_width_pt > style.line_width_pt);
        assert!(style.double_bond_spacing_pt() > 1.0);
        assert!(style.wedge_wide_width_pt() > style.wedge_tip_width_pt());
        assert!(style.label_clearance_pt() > style.margin_width_pt);
        assert!(style.hash_count(style.bond_length_pt) >= 4);

        let parsed = JournalStyle::from_name(preset.name()).expect("roundtrip style name");
        assert_eq!(parsed, preset);
        let parsed_slug = JournalStyle::from_name(preset.slug()).expect("roundtrip style slug");
        assert_eq!(parsed_slug, preset);

        let doubled = style.with_scale(2.0);
        assert!((doubled.bond_length_pt - 2.0 * style.bond_length_pt).abs() < 1e-9);
        assert!((doubled.atom_label_size_pt - 2.0 * style.atom_label_size_pt).abs() < 1e-9);
    }

    // Verify exact ACS Document 1996 house numbers (`14.4 pt` bond, `0.6 pt` line, `2.0 pt` bold, `10 pt` label)
    let acs = JournalStyle::Acs1996.to_style();
    assert!((acs.bond_length_pt - 14.4).abs() < 1e-9);
    assert!((acs.line_width_pt - 0.6).abs() < 1e-9);
    assert!((acs.bold_width_pt - 2.0).abs() < 1e-9);
    assert!((acs.atom_label_size_pt - 10.0).abs() < 1e-9);

    let rsc = JournalStyle::Rsc.to_style();
    assert!((rsc.bond_length_pt - 12.2).abs() < 1e-9);
    assert!((rsc.atom_label_size_pt - 7.0).abs() < 1e-9);

    let wiley = JournalStyle::Wiley.to_style();
    assert!((wiley.bond_length_pt - 17.0).abs() < 1e-9);

    let nature = JournalStyle::Nature.to_style();
    assert!((nature.bond_length_pt - 10.8).abs() < 1e-9);
}

#[test]
fn test_direction_aware_heteroatom_labels_and_subscripts() {
    let style = DocumentStyle::default();

    // 1. Alcohol oxygen with bond going LEFT (`dx = -1.0`) -> `OH` extending right!
    let oh_right = layout_atom_label(
        "O",
        None,
        0,
        0,
        None,
        None,
        false,
        1.0,
        &[[-1.0, 0.0]],
        &style,
    )
    .expect("oxygen label");
    assert_eq!(oh_right.plain, "OH");
    assert_eq!(oh_right.direction, LabelDirection::LeftToRight);
    assert!(oh_right.max_x_pt > -oh_right.min_x_pt);

    // 2. Alcohol oxygen with bond going RIGHT (`dx = +1.0`) -> `HO` extending left!
    let ho_left = layout_atom_label(
        "O",
        None,
        0,
        0,
        None,
        None,
        false,
        1.0,
        &[[1.0, 0.0]],
        &style,
    )
    .expect("left oxygen label");
    assert_eq!(ho_left.plain, "HO");
    assert_eq!(ho_left.direction, LabelDirection::RightToLeft);
    assert!(-ho_left.min_x_pt > ho_left.max_x_pt);

    // 3. Primary amine with bond going RIGHT -> `H2N` with subscript `2`!
    let h2n = layout_atom_label(
        "N",
        None,
        0,
        0,
        None,
        None,
        false,
        1.0,
        &[[0.866, 0.5]],
        &style,
    )
    .expect("amine label");
    assert_eq!(h2n.plain, "H2N");
    assert!(h2n
        .tokens
        .iter()
        .any(|t| t.text == "2" && t.script == ScriptKind::Subscript));

    // 4. Secondary amine at apex with symmetric downward bonds -> stacked `NH` (`H` above `N`)!
    let nh_top = layout_atom_label(
        "N",
        None,
        0,
        0,
        None,
        None,
        false,
        2.0,
        &[[-0.707, -0.707], [0.707, -0.707]],
        &style,
    )
    .expect("stacked NH");
    assert_eq!(nh_top.direction, LabelDirection::StackedAbove);
    assert_eq!(nh_top.tokens[0].text, "H");
    assert_eq!(nh_top.tokens[1].text, "N");

    // 5. Condensed `COOH` on left terminus (`HOOC` RTL) vs right terminus (`COOH` LTR) + `NH3+` sub+super
    let hooc = layout_atom_label(
        "C",
        Some("COOH"),
        0,
        0,
        None,
        None,
        false,
        1.0,
        &[[1.0, 0.0]],
        &style,
    )
    .expect("HOOC");
    assert_eq!(hooc.plain, "HOOC");

    let nh3_plus = tokenize_chemical_formula("NH3+");
    assert_eq!(nh3_plus.len(), 3);
    assert_eq!(nh3_plus[1].script, ScriptKind::Subscript);
    assert_eq!(nh3_plus[2].script, ScriptKind::Superscript);

    // 6. Skeletal carbon with >= 1 bond and no charge/label is hidden (`None`)
    assert!(layout_atom_label(
        "C",
        None,
        0,
        0,
        None,
        None,
        false,
        2.0,
        &[[-0.866, -0.5], [0.866, -0.5]],
        &style
    )
    .is_none());
}

#[test]
fn test_template_library_completeness_and_chair_sprouting() {
    let lib = TemplateLibrary::builtin();
    assert!(
        lib.templates.len() >= 50,
        "expected >= 50 templates, got {}",
        lib.templates.len()
    );

    // Every required template from the spec must exist and be non-empty
    let required_ids = [
        "cyclopropane",
        "cyclobutane",
        "cyclopentane",
        "cyclohexane",
        "cycloheptane",
        "cyclooctane",
        "cyclopentadiene",
        "benzene",
        "naphthalene",
        "anthracene",
        "pyridine",
        "pyrimidine",
        "pyrrole",
        "furan",
        "thiophene",
        "indole",
        "purine",
        "chair_cyclohexane_1",
        "chair_cyclohexane_2",
        "boat_cyclohexane",
        "norbornane",
        "adamantane",
        "trans_decalin",
        "cis_decalin",
        "steroid_core",
        "l_alanine",
        "l_proline",
        "l_tryptophan",
        "l_arginine",
        "alpha_d_glucopyranose_haworth",
        "beta_d_glucopyranose_haworth",
        "beta_d_glucopyranose_chair",
        "beta_d_ribofuranose",
        "adenosine",
        "atp",
        "peptide_beta_turn",
        "ferrocene",
        "square_planar_complex",
        "octahedral_complex",
        "tlc_plate",
        "orbital_s",
        "orbital_p",
        "orbital_d",
    ];
    for id in required_ids {
        let tpl = lib
            .get(id)
            .unwrap_or_else(|| panic!("missing template `{id}`"));
        assert!(!tpl.scene.is_empty(), "template `{id}` scene is empty");
    }

    // Both chair cyclohexane flips must report 6 ring atoms with alternating up/down axial sprout angles!
    for chair_id in ["chair_cyclohexane_1", "chair_cyclohexane_2"] {
        let tpl = lib.get(chair_id).unwrap();
        assert_eq!(
            tpl.chair_sprouts.len(),
            6,
            "`{chair_id}` must have 6 chair sprout vectors"
        );
        let mut up_count = 0;
        let mut down_count = 0;
        for sp in &tpl.chair_sprouts {
            if sp.axial_angle_deg > 45.0 {
                up_count += 1;
            } else if sp.axial_angle_deg < -45.0 {
                down_count += 1;
            }
        }
        assert_eq!(up_count, 3);
        assert_eq!(down_count, 3);

        // Verify interactive sprouting switches from axial (1st substituent) to equatorial (2nd substituent)!
        let mut scene = tpl.scene.clone();
        let sv1 = chair_sprout_vectors(&scene, 0).expect("chair vector before substituent");
        assert_eq!(sv1.next_recommended, sv1.axial);
        let ax_atom = scene.add_atom(SceneAtom::new(
            "O",
            scene.atoms[0].x + sv1.axial[0],
            scene.atoms[0].y + sv1.axial[1],
        ));
        scene.add_bond(SceneBond::single(0, ax_atom));
        let sv2 = chair_sprout_vectors(&scene, 0).expect("chair vector after axial substituent");
        assert_eq!(sv2.next_recommended, sv2.equatorial);
    }

    // Polypeptide sequence builder from both 1-letter and 3-letter codes
    let pep1 = build_peptide_scene(&parse_peptide_sequence("MQIFVK"));
    let pep2 = build_peptide_scene(&["MET", "GLN", "ILE", "PHE", "VAL", "LYS"]);
    assert_eq!(pep1.atoms.len(), pep2.atoms.len());
    assert!(pep1.atoms.len() > 35);
}

#[test]
fn test_svg_pdf_and_png_rendering_full_scene() {
    let lib = TemplateLibrary::builtin();
    let mut scene = RenderScene::new();

    // 1. Indole + wedge/hash/wavy/dative substituents
    let indole = &lib.get("indole").unwrap().scene;
    scene.merge_at(indole, -4.0, 2.0);
    let c0 = scene.add_atom(SceneAtom::carbon(-1.0, 2.0));
    let oh = scene.add_atom(SceneAtom::new("O", -1.866, 2.5));
    let nh2 = scene.add_atom(SceneAtom::new("N", -0.134, 2.5));
    let cooh = scene.add_atom(SceneAtom::with_group("C", "COOH", -1.0, 1.0));
    let f_atom = scene.add_atom(SceneAtom::new("F", -1.7, 1.2));
    scene.atoms[c0].cip_label = Some("(S)".to_string());
    scene.atoms[oh].lone_pairs = 2;
    scene.add_bond(SceneBond::wedge(c0, oh));
    scene.add_bond(SceneBond::hash(c0, nh2));
    scene.add_bond(SceneBond::wavy(c0, cooh));
    scene.add_bond(SceneBond::dative(nh2, f_atom));

    // 2. Reaction arrows (Solid + Equilibrium + Retrosynthetic)
    scene.reaction_arrows.push(SceneReactionArrow {
        from: [1.2, 2.0],
        to: [3.8, 2.0],
        kind: ArrowKind::Solid,
        above_text: Some("Pd(PPh3)4, K2CO3".to_string()),
        below_text: Some("THF/H2O, 80 C, 94%".to_string()),
        color: None,
    });
    scene.reaction_arrows.push(SceneReactionArrow {
        from: [1.2, 0.4],
        to: [3.8, 0.4],
        kind: ArrowKind::Equilibrium,
        above_text: None,
        below_text: None,
        color: None,
    });

    // 3. Curved 2e⁻ full & 1e⁻ fishhook electron-pushing arrows
    scene.curved_arrows.push(SceneCurvedArrow::from_chord(
        [-1.8, 2.6],
        [-0.2, 2.6],
        0.75,
        PushingArrowKind::FullTwoElectron,
    ));
    scene.curved_arrows.push(SceneCurvedArrow::from_chord(
        [-0.2, 1.8],
        [0.8, 1.8],
        -0.55,
        PushingArrowKind::FishhookOneElectron,
    ));

    // 4. Polymer bracket `[ ]_n`
    scene.brackets.push(SceneBracket {
        min: [-2.4, 0.6],
        max: [0.5, 3.2],
        style: BracketStyle::Square,
        subscript: Some("n".to_string()),
        superscript: Some("‡".to_string()),
        color: None,
    });

    // 5. s / p / d orbitals
    scene.orbitals.push(SceneOrbital {
        center: [5.5, 2.8],
        size_mol: 0.9,
        angle_deg: 0.0,
        orbital_type: OrbitalType::S,
        shading: OrbitalShading::ShadedAndOpen,
        color: None,
    });
    scene.orbitals.push(SceneOrbital {
        center: [7.2, 2.8],
        size_mol: 1.0,
        angle_deg: 90.0,
        orbital_type: OrbitalType::P,
        shading: OrbitalShading::DualPhaseColored,
        color: None,
    });
    scene.orbitals.push(SceneOrbital {
        center: [9.2, 2.8],
        size_mol: 1.0,
        angle_deg: 45.0,
        orbital_type: OrbitalType::DClover,
        shading: OrbitalShading::ShadedAndOpen,
        color: None,
    });

    // 6. TLC plate
    scene.merge_at(&lib.get("tlc_plate").unwrap().scene, 7.0, -1.5);

    // 7. Embedded 1D ¹H NMR spectrum plot
    scene.spectra_1d.push(SceneSpectrum1D {
        origin: [-5.0, -4.2],
        width_mol: 9.5,
        height_mol: 3.4,
        nucleus: "1H".to_string(),
        spectrometer_mhz: 500.0,
        ppm_min: 0.0,
        ppm_max: 10.0,
        linewidth_hz: 1.4,
        show_integrals: true,
        show_peak_labels: true,
        title: Some("Predicted 1H NMR (500 MHz, CDCl3)".to_string()),
        signals: vec![
            SpectrumSignal1D {
                ppm: 7.42,
                multiplicity: "dd".to_string(),
                couplings_hz: vec![8.1, 1.4],
                integral: 2.0,
                atom_indices: vec![0, 1],
                assignment_label: Some("Ar-H".to_string()),
            },
            SpectrumSignal1D {
                ppm: 4.12,
                multiplicity: "q".to_string(),
                couplings_hz: vec![7.1],
                integral: 2.0,
                atom_indices: vec![c0],
                assignment_label: Some("OCH2".to_string()),
            },
            SpectrumSignal1D {
                ppm: 1.26,
                multiplicity: "t".to_string(),
                couplings_hz: vec![7.1],
                integral: 3.0,
                atom_indices: vec![],
                assignment_label: Some("CH3".to_string()),
            },
        ],
        custom_trace: Vec::new(),
    });

    let style = JournalStyle::Acs1996.to_style();

    // --- Verify SVG ---
    let svg = render_scene_svg(&scene, &style);
    assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(svg.ends_with("</svg>\n"));
    assert!(svg.contains("<polygon"));
    assert!(svg.contains("<path"));
    assert!(svg.contains("<line"));
    assert!(svg.contains("<circle"));
    assert!(svg.contains("Pd(PPh3)4"));
    assert!(svg.contains("Predicted 1H NMR"));

    // --- Verify Standalone Vector PDF 1.4 ---
    let pdf = render_scene_pdf(&scene, &style);
    assert!(pdf.starts_with(b"%PDF-1.4\n"));
    let pdf_str = String::from_utf8_lossy(&pdf);
    assert!(pdf_str.contains("/Type /Catalog"));
    assert!(pdf_str.contains("/Type /Page"));
    assert!(pdf_str.contains("/BaseFont /Helvetica"));
    assert!(pdf_str.contains("\nxref\n"));
    assert!(pdf_str.contains("\ntrailer\n"));
    assert!(pdf_str.ends_with("%%EOF\n"));
    assert!(pdf.len() > 2500);

    // --- Verify Pure-Rust Lossless PNG ---
    let png = render_scene_png(&scene, &style, 96.0);
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[12..16], b"IHDR");
    let png_w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let png_h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    assert!(png_w >= 120 && png_h >= 80);
    // Verify IDAT and trailing IEND chunk
    assert!(png.windows(4).any(|w| w == b"IDAT"));
    assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

    // --- Verify PPM (`P6`) has non-white ink pixels rendered ---
    let ppm = render_scene_ppm(&scene, &style, 72.0);
    assert!(ppm.starts_with(b"P6\n"));
    let non_white_bytes = ppm.iter().skip(20).filter(|&&b| b < 240).count();
    assert!(
        non_white_bytes > 400,
        "expected crisp rendered ink pixels in raster output, got {non_white_bytes}"
    );
}

#[test]
fn test_pdf_xref_byte_offsets_and_png_crc32_integrity() {
    let lib = TemplateLibrary::builtin();
    let scene = lib.get("atp").unwrap().scene.clone();

    for &preset in JournalStyle::ALL {
        let style = preset.to_style();
        let pdf = render_scene_pdf(&scene, &style);
        assert!(pdf.starts_with(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"));
        // Everything after the 15-byte `%PDF-1.4\n%\xE2\xE3\xCF\xD3\n` header is pure ASCII!
        let ascii_tail =
            std::str::from_utf8(&pdf[15..]).expect("PDF body after binary marker must be ASCII");

        // Verify `startxref` points directly to `xref\n`
        let startxref_rel = ascii_tail.rfind("startxref\n").expect("missing startxref");
        let after = &ascii_tail[startxref_rel + "startxref\n".len()..];
        let xref_offset: usize = after
            .lines()
            .next()
            .unwrap()
            .trim()
            .parse()
            .expect("parse startxref offset");
        assert!(
            pdf[xref_offset..].starts_with(b"xref\n"),
            "startxref offset {xref_offset} did not point to `xref\\n`"
        );

        // Verify every in-use (`n`) xref entry points directly to `<id> 0 obj\n`
        let xref_section = std::str::from_utf8(&pdf[xref_offset..]).unwrap();
        let mut lines = xref_section.lines();
        assert_eq!(lines.next(), Some("xref"));
        let header = lines.next().expect("xref subsection header");
        let count: usize = header.split_whitespace().nth(1).unwrap().parse().unwrap();
        for obj_id in 0..count {
            let entry = lines.next().expect("xref entry line");
            assert_eq!(
                entry.len(),
                19,
                "PDF 1.4 xref line + trailing space must be 19 chars (20 with LF): `{entry}`"
            );
            let off: usize = entry[0..10].parse().expect("10-digit offset");
            let flag = &entry[17..18];
            if obj_id == 0 {
                assert_eq!(flag, "f");
            } else {
                assert_eq!(flag, "n");
                let expected_prefix = format!("{obj_id} 0 obj\n");
                assert!(
                    pdf[off..].starts_with(expected_prefix.as_bytes()),
                    "xref object {obj_id} offset {off} did not point to `{expected_prefix:?}`"
                );
            }
        }

        // Verify PNG chunk framing and IEEE CRC-32 checksum on every chunk
        let png = render_scene_png(&scene, &style, 96.0);
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
        let mut cursor = 8usize;
        let mut saw_ihdr = false;
        let mut saw_idat = false;
        let mut saw_iend = false;
        while cursor < png.len() {
            assert!(cursor + 12 <= png.len(), "truncated PNG chunk header");
            let len = u32::from_be_bytes([
                png[cursor],
                png[cursor + 1],
                png[cursor + 2],
                png[cursor + 3],
            ]) as usize;
            let ctype = &png[cursor + 4..cursor + 8];
            let data_end = cursor + 8 + len;
            assert!(data_end + 4 <= png.len(), "truncated PNG chunk payload");
            let expected_crc = u32::from_be_bytes([
                png[data_end],
                png[data_end + 1],
                png[data_end + 2],
                png[data_end + 3],
            ]);
            let mut crc = 0xFFFF_FFFFu32;
            for &b in &png[cursor + 4..data_end] {
                crc ^= b as u32;
                for _ in 0..8 {
                    if crc & 1 != 0 {
                        crc = (crc >> 1) ^ 0xEDB8_8320;
                    } else {
                        crc >>= 1;
                    }
                }
            }
            let computed_crc = !crc;
            assert_eq!(
                computed_crc,
                expected_crc,
                "CRC-32 mismatch in PNG chunk {:?}",
                std::str::from_utf8(ctype)
            );
            match ctype {
                b"IHDR" => saw_ihdr = true,
                b"IDAT" => saw_idat = true,
                b"IEND" => saw_iend = true,
                _ => {}
            }
            cursor = data_end + 4;
        }
        assert!(saw_ihdr && saw_idat && saw_iend);
    }
}

#[test]
fn test_all_arrows_brackets_orbitals_aromatic_circles_and_bridge_gaps() {
    let mut scene = RenderScene::new();

    // Benzene with aromatic Robinson circle (`BondOrder::Aromatic`)
    for k in 0..6 {
        let ang = std::f64::consts::FRAC_PI_2 + (k as f64) * std::f64::consts::FRAC_PI_3;
        scene.add_atom(SceneAtom::carbon(ang.cos(), ang.sin()));
    }
    for k in 0..6 {
        let mut b = SceneBond::single(k, (k + 1) % 6);
        b.order = BondOrder::Aromatic;
        scene.add_bond(b);
    }
    scene.aromatic_circles.push((0..6).collect());
    // Also fuse a 5-membered ring onto bond 0 (`indane`-style ring fusion)
    let fused = fuse_polygon_on_bond(&mut scene, 0, 5, false);
    assert_eq!(fused.len(), 5);
    assert_eq!(scene.atoms.len(), 9);

    // Crossing-bond pair (automatic bridge-gap clipping on the background bond!)
    let a0 = scene.add_atom(SceneAtom::carbon(3.0, 0.0));
    let a1 = scene.add_atom(SceneAtom::carbon(5.0, 2.0));
    let mut front0 = SceneAtom::carbon(3.0, 2.0);
    front0.z = 0.5;
    let mut front1 = SceneAtom::carbon(5.0, 0.0);
    front1.z = 0.5;
    let a2 = scene.add_atom(front0);
    let a3 = scene.add_atom(front1);
    scene.add_bond(SceneBond::single(a0, a1));
    scene.add_bond(SceneBond::single(a2, a3));

    // Hollow wedge + crossed E/Z double bond + triple bond
    let a4 = scene.add_atom(SceneAtom::carbon(6.5, 0.0));
    let a5 = scene.add_atom(SceneAtom::carbon(7.5, 0.8));
    let a6 = scene.add_atom(SceneAtom::carbon(8.5, 0.0));
    let a7 = scene.add_atom(SceneAtom::new("N", 9.5, 0.0));
    let mut hw = SceneBond::single(a4, a5);
    hw.stereo = BondStereo::UpHollow;
    scene.add_bond(hw);
    let mut cr = SceneBond::double(a5, a6);
    cr.stereo = BondStereo::CrossedDouble;
    scene.add_bond(cr);
    scene.add_bond(SceneBond::triple(a6, a7));

    // All 7 reaction arrow kinds
    let kinds = [
        ArrowKind::Solid,
        ArrowKind::Equilibrium,
        ArrowKind::UnbalancedEquilibrium,
        ArrowKind::Retrosynthetic,
        ArrowKind::Resonance,
        ArrowKind::NoGo,
        ArrowKind::Dashed,
    ];
    for (idx, kind) in kinds.into_iter().enumerate() {
        let y = -1.5 - (idx as f64) * 0.65;
        scene.reaction_arrows.push(SceneReactionArrow {
            from: [0.0, y],
            to: [2.5, y],
            kind,
            above_text: None,
            below_text: None,
            color: None,
        });
    }

    // All 3 bracket styles (`Square`, `Parenthesis`, `CurlyBrace`)
    for (idx, bs) in [
        BracketStyle::Square,
        BracketStyle::Parenthesis,
        BracketStyle::CurlyBrace,
    ]
    .into_iter()
    .enumerate()
    {
        let x = 4.0 + (idx as f64) * 2.2;
        scene.brackets.push(SceneBracket {
            min: [x, -3.5],
            max: [x + 1.4, -1.5],
            style: bs,
            subscript: Some("m".to_string()),
            superscript: None,
            color: None,
        });
    }

    // Remaining orbital types (`Dz2`, `HybridSp3`)
    scene.orbitals.push(SceneOrbital {
        center: [11.0, 1.0],
        size_mol: 1.0,
        angle_deg: 90.0,
        orbital_type: OrbitalType::Dz2,
        shading: OrbitalShading::ShadedAndOpen,
        color: None,
    });
    scene.orbitals.push(SceneOrbital {
        center: [13.0, 1.0],
        size_mol: 1.0,
        angle_deg: 30.0,
        orbital_type: OrbitalType::HybridSp3,
        shading: OrbitalShading::DualPhaseColored,
        color: None,
    });

    let style = JournalStyle::Jacs.to_style();
    let compiled = compile_scene_primitives(&scene, &style);
    assert!(!compiled.primitives.is_empty());

    // Confirm the aromatic Robinson inner circle was emitted
    assert!(compiled.primitives.iter().any(|p| matches!(
        p,
        VectorPrimitive::Circle {
            fill: None,
            stroke: Some(_),
            ..
        }
    )));

    let svg = render_scene_svg(&scene, &style);
    let pdf = render_scene_pdf(&scene, &style);
    let png = render_scene_png(&scene, &style, 96.0);
    assert!(svg.len() > 2000);
    assert!(pdf.starts_with(b"%PDF-1.4\n"));
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
}

#[test]
fn test_all_twenty_canonical_l_amino_acids_present() {
    let lib = TemplateLibrary::builtin();
    let aa_ids = [
        "l_alanine",
        "l_arginine",
        "l_asparagine",
        "l_aspartate",
        "l_cysteine",
        "l_glutamate",
        "l_glutamine",
        "l_glycine",
        "l_histidine",
        "l_isoleucine",
        "l_leucine",
        "l_lysine",
        "l_methionine",
        "l_phenylalanine",
        "l_proline",
        "l_serine",
        "l_threonine",
        "l_tryptophan",
        "l_tyrosine",
        "l_valine",
    ];
    for id in aa_ids {
        let tpl = lib
            .get(id)
            .unwrap_or_else(|| panic!("missing canonical L-amino acid template `{id}`"));
        assert!(
            tpl.scene.atoms.len() >= 5,
            "`{id}` must contain backbone + sidechain heavy atoms"
        );
    }
}
