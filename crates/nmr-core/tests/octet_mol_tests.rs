//! Comprehensive integration tests for `octet_mol`: Core Molecular Graph, Ring & Aromaticity Perception,
//! IUPAC CIP `(R)/(S)` & `(E)/(Z)`, Bridged `endo`/`exo`, Chair `axial`/`equatorial`, Axial `(M)/(P)`,
//! 2D/3D Coordinate Cleanup, Superatom Abbreviations, Peptide & Oligonucleotide Sequences, and
//! Chemical File Formats (SMILES, InChI/InChIKey, MOL V2000/V3000, SDF, CDXML, Binary CDX `VjCD0100`, `.oct`).
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! and open cheminformatics specifications.

pub use nmr_core::*;

use octet_mol::*;

#[test]
fn test_periodic_table_valence_and_rings_aromaticity() {
    // Full periodic table Z = 1..=118 + D/T
    for z in 1u8..=118u8 {
        let sym = symbol_from_atomic_number(z);
        assert_ne!(sym, "*", "Z={} should have real symbol", z);
        assert_eq!(atomic_number_from_symbol(sym), Some(z));
    }
    let deut = Atom::new("D");
    assert_eq!(deut.z, 1);
    assert_eq!(deut.isotope, Some(2));
    let trit = Atom::new("T");
    assert_eq!(trit.z, 1);
    assert_eq!(trit.isotope, Some(3));
    assert!(is_metal(26)); // Fe
    assert!(is_metal(46)); // Pd
    assert!(!is_metal(6)); // C

    // SSSR & Hückel 4n+2 aromaticity + Kekulization across diverse heterocycles & fused rings
    let cases = [
        ("c1ccccc1", 1, 6, "C6H6"),                        // benzene
        ("c1ccncc1", 1, 6, "C5H5N"),                       // pyridine
        ("c1cc[nH]c1", 1, 5, "C4H5N"),                     // pyrrole
        ("c1ccoc1", 1, 5, "C4H4O"),                        // furan
        ("c1ccsc1", 1, 5, "C4H4S"),                        // thiophene
        ("c1ccc2[nH]ccc2c1", 2, 9, "C8H7N"),               // indole
        ("c1ccc2ccccc2c1", 2, 10, "C10H8"),                // naphthalene
        ("c1ncnc2[nH]cnc12", 2, 9, "C5H4N4"),              // purine
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 2, 9, "C8H10N4O2"), // caffeine (exocyclic carbonyl 6+5 aromatic)
    ];

    for (smi, expected_rings, expected_arom_atoms, expected_formula) in cases {
        let mut mol = parse_smiles(smi).expect(smi);
        assert_eq!(mol.hill_formula(), expected_formula, "formula for {}", smi);
        let rings = perceive_sssr(&mol);
        assert_eq!(rings.len(), expected_rings, "SSSR count for {}", smi);
        let arom = perceive_aromaticity(&mol);
        let arom_count = arom.aromatic_atoms.iter().filter(|&&b| b).count();
        assert_eq!(
            arom_count, expected_arom_atoms,
            "aromatic atom count for {}",
            smi
        );

        // Kekulize and verify aromatic perception recovers the exact same aromatic system
        kekulize(&mut mol).expect("kekulize should succeed");
        assert!(
            mol.bonds.iter().all(|b| b.kind != BondKind::Aromatic),
            "all aromatic bonds should be Kekulé single/double after kekulize for {}",
            smi
        );
        assert_eq!(
            mol.hill_formula(),
            expected_formula,
            "Kekulized formula must stay identical for {}",
            smi
        );
        let arom_after = perceive_aromaticity(&mol);
        let arom_count_after = arom_after.aromatic_atoms.iter().filter(|&&b| b).count();
        assert_eq!(
            arom_count_after, expected_arom_atoms,
            "Kekulé form must re-perceive identical aromatic atoms for {}",
            smi
        );
    }

    // Norbornane (bicyclo[2.2.1]heptane): E=8, V=7 => SSSR = 2 5-membered rings
    let norbornane = parse_smiles("C1CC2CCC1C2").unwrap();
    let nb_rings = perceive_sssr(&norbornane);
    assert_eq!(nb_rings.len(), 2);
    assert!(nb_rings.iter().all(|r| r.len() == 5));
}

#[test]
fn test_cip_rs_ez_pseudoasymmetric_endo_exo_chair_and_allenes() {
    // 1. Chiral amino acids & chiral pharmaceuticals
    // L-Alanine: N[C@@H](C)C(=O)O -> (2S)
    let mut l_ala = parse_smiles("N[C@@H](C)C(=O)O").unwrap();
    clean_2d(&mut l_ala);
    let cip_l_ala = perceive_cip(&l_ala);
    assert_eq!(cip_l_ala.atoms.get(&1), Some(&CipAtomDescriptor::S));

    // D-Alanine: N[C@H](C)C(=O)O -> (2R)
    let mut d_ala = parse_smiles("N[C@H](C)C(=O)O").unwrap();
    clean_2d(&mut d_ala);
    let cip_d_ala = perceive_cip(&d_ala);
    assert_eq!(cip_d_ala.atoms.get(&1), Some(&CipAtomDescriptor::R));

    // L-Cysteine: N[C@@H](CS)C(=O)O -> (2R) because -CH2SH outranks -COOH (S > O)!
    let mut l_cys = parse_smiles("N[C@@H](CS)C(=O)O").unwrap();
    clean_2d(&mut l_cys);
    let cip_l_cys = perceive_cip(&l_cys);
    assert_eq!(cip_l_cys.atoms.get(&1), Some(&CipAtomDescriptor::R));

    // L-Threonine: N[C@@H]([C@H](O)C)C(=O)O -> (2S, 3R)
    let mut l_thr = parse_smiles("N[C@@H]([C@H](O)C)C(=O)O").unwrap();
    clean_2d(&mut l_thr);
    let cip_l_thr = perceive_cip(&l_thr);
    assert_eq!(cip_l_thr.atoms.get(&1), Some(&CipAtomDescriptor::S));
    assert_eq!(cip_l_thr.atoms.get(&2), Some(&CipAtomDescriptor::R));

    // L-Isoleucine: N[C@@H]([C@@H](C)CC)C(=O)O -> (2S, 3S)
    let mut l_ile = parse_smiles("N[C@@H]([C@@H](C)CC)C(=O)O").unwrap();
    clean_2d(&mut l_ile);
    let cip_l_ile = perceive_cip(&l_ile);
    assert_eq!(cip_l_ile.atoms.get(&1), Some(&CipAtomDescriptor::S));
    assert_eq!(cip_l_ile.atoms.get(&2), Some(&CipAtomDescriptor::S));

    // (S)-Ibuprofen (`[C@H]`) and (R)-Ibuprofen (`[C@@H]`): phenyl is CIP rank 2 < COOH rank 1
    let mut s_ibu = parse_smiles("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1").unwrap();
    clean_2d(&mut s_ibu);
    let cip_ibu = perceive_cip(&s_ibu);
    assert_eq!(
        cip_ibu.atoms.values().copied().collect::<Vec<_>>(),
        vec![CipAtomDescriptor::S]
    );

    let mut r_ibu = parse_smiles("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1").unwrap();
    clean_2d(&mut r_ibu);
    assert_eq!(
        perceive_cip(&r_ibu)
            .atoms
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![CipAtomDescriptor::R]
    );

    // (R)-Thalidomide (`[C@@H]`) vs (S)-Thalidomide (`[C@H]`)
    let mut r_thal = parse_smiles("O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1").unwrap();
    clean_2d(&mut r_thal);
    assert_eq!(
        perceive_cip(&r_thal).atoms.get(&4),
        Some(&CipAtomDescriptor::R)
    );

    let mut s_thal = parse_smiles("O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1").unwrap();
    clean_2d(&mut s_thal);
    assert_eq!(
        perceive_cip(&s_thal).atoms.get(&4),
        Some(&CipAtomDescriptor::S)
    );

    // CIP Rule 2 isotopic chirality: (1R)-[1-2H]ethanol: C[C@H]([2H])O -> O(1) > C(2) > 2H(3) > 1H(4)
    let mut iso_etoh = parse_smiles("C[C@@H]([2H])O").unwrap();
    clean_2d(&mut iso_etoh);
    let cip_iso = perceive_cip(&iso_etoh);
    assert!(
        cip_iso.atoms.contains_key(&1),
        "isotopic chiral center must be perceived"
    );

    // CIP Rule 5 pseudoasymmetric center (r)/(s):
    // In linear SMILES `C[C@H](O)[C@H](O)[C@H](O)C`, C2 is (S) and C4 is (R) (meso), so C3 is pseudoasymmetric (r)/(s);
    // whereas in homochiral `C[C@H](O)[C@H](O)[C@@H](O)C` (`(2S,4S)`), C3 has two identical (S) arms and is achiral (`None`)!
    let mut homochiral = parse_smiles("C[C@H](O)[C@H](O)[C@@H](O)C").unwrap();
    clean_2d(&mut homochiral);
    assert_eq!(perceive_cip(&homochiral).atoms.get(&3), None);

    let mut meso_1 = parse_smiles("C[C@H](O)[C@H](O)[C@H](O)C").unwrap();
    clean_2d(&mut meso_1);
    let cip_meso_1 = perceive_cip(&meso_1);
    let c3_desc_1 = cip_meso_1.atoms.get(&3).copied();
    assert!(
        matches!(
            c3_desc_1,
            Some(CipAtomDescriptor::PseudoR | CipAtomDescriptor::PseudoS)
        ),
        "C3 of meso (2S,4R)-pentane-2,3,4-triol should be pseudoasymmetric (r)/(s), got {:?}",
        c3_desc_1
    );
    let mut meso_2 = parse_smiles("C[C@H](O)[C@@H](O)[C@H](O)C").unwrap();
    clean_2d(&mut meso_2);
    let cip_meso_2 = perceive_cip(&meso_2);
    let c3_desc_2 = cip_meso_2.atoms.get(&3).copied();
    assert!(matches!(
        c3_desc_2,
        Some(CipAtomDescriptor::PseudoR | CipAtomDescriptor::PseudoS)
    ));
    assert_ne!(
        c3_desc_1, c3_desc_2,
        "epimeric pseudoasymmetric centers must flip r <-> s"
    );

    // 2. Alkene (E)/(Z)
    let mut e_butene = parse_smiles("C/C=C/C").unwrap();
    clean_2d(&mut e_butene);
    assert_eq!(
        perceive_cip(&e_butene).bonds.get(&1),
        Some(&CipBondDescriptor::E)
    );

    let mut z_butene = parse_smiles("C/C=C\\C").unwrap();
    clean_2d(&mut z_butene);
    assert_eq!(
        perceive_cip(&z_butene).bonds.get(&1),
        Some(&CipBondDescriptor::Z)
    );

    // Higher-priority halogen alkene: Br(35) > Cl(17) on C1, F(9) > H(1) on C2
    // Br and F trans => (E); Br and F cis => (Z)
    let mut z_halo = parse_smiles("Br/C(Cl)=C(\\F)[H]").unwrap();
    clean_2d(&mut z_halo);
    let halo_db = z_halo
        .bonds
        .iter()
        .position(|b| b.kind == BondKind::Double)
        .unwrap();
    assert_eq!(
        perceive_cip(&z_halo).bonds.get(&halo_db),
        Some(&CipBondDescriptor::Z)
    );

    // 3. Bridged bicyclic endo/exo on 2-norbornanol (bicyclo[2.2.1]heptan-2-ol) in 3D!
    // Construct explicit 3D norbornane skeleton:
    // Bridgeheads C1(0), C4(3); 2-carbon bridge C2(1), C3(2) at -X; 2-carbon bridge C5(4), C6(5) at +X;
    // 1-carbon methylene bridge C7(6) at +Z (the shortest z=1 bridge!).
    let make_3d_norbornanol = |oh_z: f64| -> Molecule {
        let mut m = Molecule::new();
        let coords = [
            (0.0, 1.15, 0.0),      // 0: C1 bridgehead
            (-1.25, 0.77, -0.45),  // 1: C2 (on x=2 bridge)
            (-1.25, -0.77, -0.45), // 2: C3
            (0.0, -1.15, 0.0),     // 3: C4 bridgehead
            (1.25, -0.77, -0.45),  // 4: C5 (on y=2 bridge)
            (1.25, 0.77, -0.45),   // 5: C6
            (0.0, 0.0, 1.10),      // 6: C7 shortest bridge (+Z)
        ];
        for &(x, y, z) in &coords {
            let mut a = Atom::from_z_xy(6, x, y);
            a.z_coord = Some(z);
            m.add_atom(a);
        }
        let mut oh = Atom::from_z_xy(8, -2.15, 1.15);
        oh.z_coord = Some(oh_z);
        m.add_atom(oh); // 7: O attached to C2(1)
        for &(u, v) in &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 6),
            (3, 6),
            (1, 7),
        ] {
            m.add_bond(u, v, BondKind::Single);
        }
        m
    };
    // Exo-2-norbornanol: OH points toward C7 (+Z)
    let exo_nb = make_3d_norbornanol(0.55);
    let ee_exo = perceive_endo_exo(&exo_nb);
    assert_eq!(ee_exo.get(&1), Some(&EndoExoDescriptor::Exo));

    // Endo-2-norbornanol: OH points away from C7 (-Z)
    let endo_nb = make_3d_norbornanol(-1.55);
    let ee_endo = perceive_endo_exo(&endo_nb);
    assert_eq!(ee_endo.get(&1), Some(&EndoExoDescriptor::Endo));

    // 4. Chair cyclohexane axial vs equatorial (both 3D chair and 2D perspective chair)
    let mut chair_3d = Molecule::new();
    for k in 0..6 {
        let ang = (k as f64) * std::f64::consts::PI / 3.0;
        let z = if k % 2 == 0 { 0.25 } else { -0.25 };
        let mut c = Atom::from_z_xy(6, 1.45 * ang.cos(), 1.45 * ang.sin());
        c.z_coord = Some(z);
        chair_3d.add_atom(c);
    }
    for k in 0..6 {
        chair_3d.add_bond(k, (k + 1) % 6, BondKind::Single);
    }
    // Axial substituent at C0 (z = +0.25) goes straight up (+Z)
    let mut ax_f = Atom::from_z_xy(9, 1.45, 0.0);
    ax_f.z_coord = Some(1.65);
    let ax_idx = chair_3d.add_atom(ax_f);
    chair_3d.add_bond(0, ax_idx, BondKind::Single);
    // Equatorial substituent at C1 (z = -0.25) goes outward radially
    let ang1 = std::f64::consts::PI / 3.0;
    let mut eq_cl = Atom::from_z_xy(17, 2.85 * ang1.cos(), 2.85 * ang1.sin());
    eq_cl.z_coord = Some(0.10);
    let eq_idx = chair_3d.add_atom(eq_cl);
    chair_3d.add_bond(1, eq_idx, BondKind::Single);

    let chair_orient_3d = perceive_chair_axial_equatorial(&chair_3d);
    assert_eq!(chair_orient_3d.get(&ax_idx), Some(&ChairOrientation::Axial));
    assert_eq!(
        chair_orient_3d.get(&eq_idx),
        Some(&ChairOrientation::Equatorial)
    );

    // 2D perspective chair drawing with chair_sprout_vectors
    let mut chair_2d = Molecule::new();
    let chair_pts = [
        (-1.30, 0.40),
        (0.00, 0.00),
        (1.30, 0.40),
        (0.80, -0.50),
        (-0.50, -0.10),
        (-1.80, -0.50),
    ];
    for &(x, y) in &chair_pts {
        chair_2d.add_atom(Atom::from_z_xy(6, x, y));
    }
    for k in 0..6 {
        chair_2d.add_bond(k, (k + 1) % 6, BondKind::Single);
    }
    let ring_indices = vec![0, 1, 2, 3, 4, 5];
    let (ax_v, _eq_v) = chair_sprout_vectors(&chair_2d, &ring_indices, 0)
        .expect("2D chair ring must yield axial & equatorial sprout vectors");
    let ax_2d_atom = chair_2d.add_atom(Atom::from_z_xy(
        8,
        chair_pts[0].0 + 1.54 * ax_v[0],
        chair_pts[0].1 + 1.54 * ax_v[1],
    ));
    chair_2d.add_bond(0, ax_2d_atom, BondKind::Single);
    let (_, eq_v2) = chair_sprout_vectors(&chair_2d, &ring_indices, 2).unwrap();
    let eq_2d_atom = chair_2d.add_atom(Atom::from_z_xy(
        6,
        chair_pts[2].0 + 1.54 * eq_v2[0],
        chair_pts[2].1 + 1.54 * eq_v2[1],
    ));
    chair_2d.add_bond(2, eq_2d_atom, BondKind::Single);

    let chair_orient_2d = perceive_chair_axial_equatorial(&chair_2d);
    assert_eq!(
        chair_orient_2d.get(&ax_2d_atom),
        Some(&ChairOrientation::Axial)
    );
    assert_eq!(
        chair_orient_2d.get(&eq_2d_atom),
        Some(&ChairOrientation::Equatorial)
    );

    // 5. Axial chirality (M)/(P) on 2,3-pentadiene (CH3-CH=C=CH-CH3)
    let make_allene = |right_me_bond: BondKind| -> Molecule {
        let mut m = Molecule::new();
        let c1 = m.add_atom(Atom::from_z_xy(6, -1.30, 0.0));
        let c2 = m.add_atom(Atom::from_z_xy(6, 0.00, 0.0));
        let c3 = m.add_atom(Atom::from_z_xy(6, 1.30, 0.0));
        let me_l = m.add_atom(Atom::from_z_xy(6, -2.05, 1.10));
        let h_l = m.add_atom(Atom::from_z_xy(1, -2.05, -1.10));
        let me_r = m.add_atom(Atom::from_z_xy(6, 2.05, 1.10));
        let h_r = m.add_atom(Atom::from_z_xy(1, 2.05, -1.10));
        m.add_bond(c1, c2, BondKind::Double);
        m.add_bond(c2, c3, BondKind::Double);
        m.add_bond(c1, me_l, BondKind::Single);
        m.add_bond(c1, h_l, BondKind::Single);
        m.add_bond(c3, me_r, right_me_bond);
        m.add_bond(
            c3,
            h_r,
            if right_me_bond == BondKind::WedgeUp {
                BondKind::HashDown
            } else {
                BondKind::WedgeUp
            },
        );
        m
    };
    let allene_up = make_allene(BondKind::WedgeUp);
    let allene_dn = make_allene(BondKind::HashDown);
    let ax_up = perceive_axial_chirality(&allene_up);
    let ax_dn = perceive_axial_chirality(&allene_dn);
    assert!(ax_up.contains_key(&1));
    assert!(ax_dn.contains_key(&1));
    assert_ne!(
        ax_up.get(&1),
        ax_dn.get(&1),
        "enantiomeric allenes must have opposite M/P"
    );
}

#[test]
fn test_clean_2d_and_clean_3d_layout_and_organometallics() {
    // 1. Clean 2D on chains, fused rings, spiro rings, bridged rings, and macrocycles
    let smiles_cases = [
        "CCCCCCCC",                 // linear octane zig-zag
        "c1ccc2ccccc2c1",           // fused naphthalene
        "C1CCC2(CC1)CCCC2",         // spiro[4.5]decane
        "C1CC2CCC1C2",              // bridged norbornane
        "C1CCCCCCCCCCC1",           // cyclododecane 12-ring macrocycle
        "N[C@@H](Cc1ccccc1)C(=O)O", // L-Phenylalanine (must synthesize wedge/hash & preserve S)
        "CC/C=C/CC",                // (E)-hex-3-ene (must preserve E)
    ];

    for smi in smiles_cases {
        let mut mol = parse_smiles(smi).expect(smi);
        clean_2d(&mut mol);
        assert!(
            !mol.needs_2d_coords(),
            "{} must have non-zero 2D coords",
            smi
        );

        // Check covalent bond lengths are close to 1.54 Å (bridged rings allow slight projection foreshortening)
        for b in &mol.bonds {
            let dx = mol.atoms[b.a].x - mol.atoms[b.b].x;
            let dy = mol.atoms[b.a].y - mol.atoms[b.b].y;
            let len = dx.hypot(dy);
            assert!(
                (0.75..=2.40).contains(&len),
                "2D bond length {} out of range for {}",
                len,
                smi
            );
        }

        // Non-bonded atoms must not collide on top of each other
        for i in 0..mol.atoms.len() {
            for j in (i + 1)..mol.atoms.len() {
                let d = (mol.atoms[i].x - mol.atoms[j].x).hypot(mol.atoms[i].y - mol.atoms[j].y);
                assert!(d > 0.25, "atom clash ({}, {}) d={} in {}", i, j, d, smi);
            }
        }
    }

    // Verify regular benzene bond lengths are exact ACS_BOND_LENGTH_ANGSTROM (1.54 Å)
    let mut bz = parse_smiles("c1ccccc1").unwrap();
    clean_2d(&mut bz);
    for b in &bz.bonds {
        let d = (bz.atoms[b.a].x - bz.atoms[b.b].x).hypot(bz.atoms[b.a].y - bz.atoms[b.b].y);
        assert!(
            (d - ACS_BOND_LENGTH_ANGSTROM).abs() < 0.05,
            "benzene bond len = {}",
            d
        );
    }

    // Verify ferrocene (η5-Cp2Fe) organometallic hapticity layout
    let mut ferrocene = Molecule::with_name("Ferrocene");
    let fe = ferrocene.add_atom(Atom::new("Fe"));
    let mut cp1_atoms = Vec::new();
    let mut cp2_atoms = Vec::new();
    for _ in 0..5 {
        cp1_atoms.push(ferrocene.add_atom(Atom::new("C")));
    }
    for _ in 0..5 {
        cp2_atoms.push(ferrocene.add_atom(Atom::new("C")));
    }
    for k in 0..5 {
        ferrocene.add_bond(cp1_atoms[k], cp1_atoms[(k + 1) % 5], BondKind::Aromatic);
        ferrocene.add_bond(cp2_atoms[k], cp2_atoms[(k + 1) % 5], BondKind::Aromatic);
    }
    let mut dummy1 = Atom::new("*");
    dummy1.multi_center_atoms = cp1_atoms.clone();
    let d1 = ferrocene.add_atom(dummy1);
    let mut dummy2 = Atom::new("*");
    dummy2.multi_center_atoms = cp2_atoms.clone();
    let d2 = ferrocene.add_atom(dummy2);
    ferrocene.add_bond(d1, fe, BondKind::Dative);
    ferrocene.add_bond(d2, fe, BondKind::Dative);

    clean_2d(&mut ferrocene);
    // Both Cp centroids should flank Fe on opposite sides!
    let c1x: f64 = cp1_atoms.iter().map(|&u| ferrocene.atoms[u].x).sum::<f64>() / 5.0;
    let c1y: f64 = cp1_atoms.iter().map(|&u| ferrocene.atoms[u].y).sum::<f64>() / 5.0;
    let c2x: f64 = cp2_atoms.iter().map(|&u| ferrocene.atoms[u].x).sum::<f64>() / 5.0;
    let c2y: f64 = cp2_atoms.iter().map(|&u| ferrocene.atoms[u].y).sum::<f64>() / 5.0;
    let sep = (c1x - c2x).hypot(c1y - c2y);
    assert!(
        sep > 2.0,
        "Ferrocene η5-Cp rings must be separated around Fe, got {}",
        sep
    );

    // 2. Clean 3D: planar benzene, puckered chair cyclohexane, and preserved (R)/(S) 3D stereocenters
    clean_3d(&mut bz);
    assert!(bz.has_3d_coords());
    // Benzene must be coplanar: check normal distance of all 6 atoms from plane of (0, 1, 2)
    let p0 = [bz.atoms[0].x, bz.atoms[0].y, bz.atoms[0].z_coord.unwrap()];
    let p1 = [bz.atoms[1].x, bz.atoms[1].y, bz.atoms[1].z_coord.unwrap()];
    let p2 = [bz.atoms[2].x, bz.atoms[2].y, bz.atoms[2].z_coord.unwrap()];
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let nvec = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let nlen = (nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2]).sqrt();
    for i in 0..6 {
        let pi = [bz.atoms[i].x, bz.atoms[i].y, bz.atoms[i].z_coord.unwrap()];
        let dist_plane =
            ((pi[0] - p0[0]) * nvec[0] + (pi[1] - p0[1]) * nvec[1] + (pi[2] - p0[2]) * nvec[2])
                .abs()
                / nlen;
        assert!(
            dist_plane < 0.08,
            "benzene atom {} out-of-plane by {}",
            i,
            dist_plane
        );
    }

    // L-Alanine vs D-Alanine in 3D: strip parity & wedges AFTER clean_3d so perceive_cip tests pure 3D coords!
    let mut l_ala_3d = parse_smiles("N[C@@H](C)C(=O)O").unwrap();
    clean_3d(&mut l_ala_3d);
    l_ala_3d.atoms[1].parity = None;
    for b in &mut l_ala_3d.bonds {
        if b.kind.is_stereo_display() {
            b.kind = BondKind::Single;
        }
    }
    assert_eq!(
        perceive_cip(&l_ala_3d).atoms.get(&1),
        Some(&CipAtomDescriptor::S)
    );

    let mut d_ala_3d = parse_smiles("N[C@H](C)C(=O)O").unwrap();
    clean_3d(&mut d_ala_3d);
    d_ala_3d.atoms[1].parity = None;
    for b in &mut d_ala_3d.bonds {
        if b.kind.is_stereo_display() {
            b.kind = BondKind::Single;
        }
    }
    assert_eq!(
        perceive_cip(&d_ala_3d).atoms.get(&1),
        Some(&CipAtomDescriptor::R)
    );
}

#[test]
fn test_all_32_superatom_abbreviations_and_condensed_formulas() {
    let required_32 = [
        "Boc", "Fmoc", "Cbz", "Ac", "Bz", "Ts", "Ms", "Tf", "TBS", "TBDPS", "TMS", "TIPS", "THP",
        "MOM", "Bn", "PMB", "Ph", "Mes", "tBu", "iPr", "nBu", "Et", "Me", "CF3", "NO2", "CN", "N3",
        "COOH", "COOMe", "COOEt", "SO3H", "PO3H2",
    ];

    for &abbr in &required_32 {
        let tmpl_smi = lookup_abbreviation_smiles(abbr)
            .unwrap_or_else(|| panic!("Missing required abbreviation {}", abbr));
        assert_eq!(canonical_abbreviation_name(abbr), Some(abbr));

        // Build CH3-CH2-<Abbr> with a superatom node and expand it!
        let mut mol = Molecule::new();
        let c0 = mol.add_atom(Atom::from_z_xy(6, 0.0, 0.0));
        let c1 = mol.add_atom(Atom::from_z_xy(6, 1.54, 0.0));
        let sa = mol.add_atom(Atom::new_superatom(abbr));
        mol.atoms[sa].x = 3.08;
        mol.atoms[sa].y = 0.0;
        mol.add_bond(c0, c1, BondKind::Single);
        mol.add_bond(c1, sa, BondKind::Single);

        let expanded_n = expand_abbreviations(&mut mol).expect(abbr);
        assert_eq!(expanded_n, 1, "should expand {}", abbr);
        assert!(
            mol.atoms.iter().all(|a| a.superatom_label.is_none()),
            "no superatoms should remain after expanding {}",
            abbr
        );

        // Compare formula against direct SMILES reference `CC` + template
        let ref_smi = tmpl_smi.replacen('*', "CC", 1);
        let ref_mol = parse_smiles(&ref_smi).expect(&ref_smi);
        assert_eq!(
            mol.hill_formula(),
            ref_mol.hill_formula(),
            "Hill formula mismatch for expanded {}",
            abbr
        );

        // Collapse the group atoms (all atoms >= 2) back into `<Abbr>` and re-expand losslessly!
        let group_indices: Vec<usize> = (2..mol.atoms.len()).collect();
        let collapsed_idx = collapse_abbreviation(&mut mol, &group_indices, abbr).unwrap();
        assert_eq!(
            mol.atoms[collapsed_idx].superatom_label.as_deref(),
            Some(abbr)
        );
        assert!(mol.atoms[collapsed_idx].inner_fragment.is_some());
        assert_eq!(mol.atoms.len(), 3);

        let re_expanded = expand_abbreviations(&mut mol).unwrap();
        assert_eq!(re_expanded, 1);
        assert_eq!(mol.hill_formula(), ref_mol.hill_formula());
    }

    // Condensed formula parser checks
    let condensed_cases = [
        ("CH3CH2OH", "CCO"),
        ("(CH3)3COH", "CC(C)(C)O"),
        ("CH3COOEt", "CCOC(=O)C"),
        ("CF3COOH", "O=C(O)C(F)(F)F"),
        ("CH3CN", "CC#N"),
        ("PhB(OH)2", "OB(O)c1ccccc1"),
    ];
    for (formula, ref_smiles) in condensed_cases {
        let parsed = parse_condensed_formula(formula).expect(formula);
        let expected = parse_smiles(ref_smiles).expect(ref_smiles);
        assert_eq!(
            parsed.hill_formula(),
            expected.hill_formula(),
            "condensed formula {} mismatch",
            formula
        );
        assert_eq!(
            write_canonical_smiles(&parsed),
            write_canonical_smiles(&expected),
            "canonical SMILES mismatch for condensed formula {}",
            formula
        );
    }
}

#[test]
fn test_peptide_and_oligonucleotide_sequence_builders() {
    // 1. 3-letter peptide: H-Ala-Gly-Pro-Phe-OH (C19H26N4O5)
    let pep3 = build_peptide_2d("H-Ala-Gly-Pro-Phe-OH").unwrap();
    assert_eq!(pep3.hill_formula(), "C19H26N4O5");
    assert!(!pep3.needs_2d_coords());

    // Verify all 3 chiral alpha-carbons (Ala, Pro, Phe; Gly is achiral) have (S) CIP configuration from their 2D wedges!
    let cip_pep3 = perceive_cip(&pep3);
    assert_eq!(
        cip_pep3.atoms.len(),
        3,
        "Ala, Pro, Phe have 3 chiral alpha-carbons"
    );
    for (&atom_idx, &desc) in &cip_pep3.atoms {
        assert_eq!(
            desc,
            CipAtomDescriptor::S,
            "L-amino acid alpha-carbon {} in H-Ala-Gly-Pro-Phe-OH must be (S)",
            atom_idx
        );
    }

    // 2. Capped peptide with L-Cys (whose CIP descriptor is R!): Ac-Ala-Cys-NH2
    let pep_cys = build_peptide_2d("Ac-Ala-Cys-NH2").unwrap();
    let cip_cys = perceive_cip(&pep_cys);
    let mut descs: Vec<&str> = cip_cys.atoms.values().map(|d| d.as_str()).collect();
    descs.sort_unstable();
    assert_eq!(
        descs,
        vec!["R", "S"],
        "Ac-Ala-Cys-NH2 must contain one (S)-Ala and one (R)-Cys"
    );

    // 3. Ubiquitin N-terminal hexapeptide 1-letter: MQIFVK (Met-Gln-Ile-Phe-Val-Lys; Ile also has 3S beta-carbon -> 7 centers!)
    let ubi6 = build_peptide_2d("MQIFVK").unwrap();
    assert_eq!(ubi6.hill_formula(), "C36H60N8O8S");
    let cip_ubi6 = perceive_cip(&ubi6);
    assert_eq!(
        cip_ubi6.atoms.len(),
        7,
        "6 L-alpha carbons + Ile C-beta = 7 (S) stereocenters"
    );
    assert!(cip_ubi6.atoms.values().all(|&d| d == CipAtomDescriptor::S));

    // 4. DNA (`5'-pATGC-3'`) and RNA (`rna: AACGUU`) oligonucleotides -> 2D structures
    let dna = build_oligonucleotide_2d("5'-pATGC-3'", true).unwrap();
    assert!(!dna.needs_2d_coords());
    // 4 phosphodiester/5'-phosphate P atoms in 5'-pATGC-3'
    let p_count_dna = dna.atoms.iter().filter(|a| a.z == 15).count();
    assert_eq!(p_count_dna, 4);
    // Every 2'-deoxy-beta-D-ribofuranose has 3 stereocenters (C1', C3', C4') -> 12 chiral centers
    assert_eq!(perceive_cip(&dna).atoms.len(), 12);

    let rna = parse_biopolymer_sequence("rna: ACGU").unwrap();
    assert!(!rna.needs_2d_coords());
    // Internal 3 phosphodiester linkages in uncapped 4-mer ACGU -> 3 P atoms
    let p_count_rna = rna.atoms.iter().filter(|a| a.z == 15).count();
    assert_eq!(p_count_rna, 3);
    // Every beta-D-ribofuranose has 4 stereocenters (C1', C2', C3', C4') -> 16 chiral centers
    assert_eq!(perceive_cip(&rna).atoms.len(), 16);
}

#[test]
fn test_smiles_and_inchi_inchikey_roundtrips() {
    // 1. Canonical SMILES invariance across atom orderings & Kekulé vs aromatic input
    let pairs = [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("OCC", "CCO"),
        ("CC(=O)O", "OC(C)=O"),
        ("N[C@@H](C)C(=O)O", "C([C@H](C)N)(=O)O"),
        ("C/C=C/C", "C(\\C)=C/C"),
        ("C/C=C\\C", "C(\\C)=C\\C"),
        ("[Na+].[Cl-]", "[Cl-].[Na+]"),
        ("[13CH3]C(=O)[15NH2]", "[15NH2]C(=O)[13CH3]"),
    ];
    for (s1, s2) in pairs {
        let m1 = parse_smiles(s1).expect(s1);
        let m2 = parse_smiles(s2).expect(s2);
        let c1 = write_canonical_smiles(&m1);
        let c2 = write_canonical_smiles(&m2);
        assert_eq!(c1, c2, "Canonical SMILES must match for {} vs {}", s1, s2);

        // Re-parsing canonical SMILES must be idempotent
        let m_rt = parse_smiles(&c1).expect(&c1);
        assert_eq!(write_canonical_smiles(&m_rt), c1);
    }

    // 2. Reaction SMILES (Reactants > Agents > Products)
    let rxn = parse_reaction_smiles("CC(=O)O.CCO>O=S(=O)(O)O>CC(=O)OCC.O").unwrap();
    assert_eq!(rxn.reaction_arrows.len(), 1);
    assert_eq!(rxn.molecules.len(), 4);
    let rxn_out = write_reaction_smiles(&rxn);
    assert!(rxn_out.contains('>'), "reaction SMILES output: {}", rxn_out);
    let rxn_rt = parse_reaction_smiles(&rxn_out).unwrap();
    assert_eq!(rxn_rt.molecules.len(), 4);

    // 3. InChI & InChIKey standard verification + round-trip
    let water = parse_smiles("O").unwrap();
    let water_inchi = write_inchi(&water);
    assert_eq!(water_inchi, "InChI=1S/H2O/h1H2");
    let water_key = write_inchikey(&water);
    assert_eq!(water_key, "XLYOFNOQVPJJNP-UHFFFAOYSA-N");

    assert_eq!(
        inchi_to_inchikey("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H").unwrap(),
        "UHOVQNZJYSORNB-UHFFFAOYSA-N"
    );
    assert_eq!(
        inchi_to_inchikey("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3").unwrap(),
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    );
    let parsed_mckay_bz = parse_inchi("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H").unwrap();
    assert_eq!(
        write_canonical_smiles(&parsed_mckay_bz),
        write_canonical_smiles(&parse_smiles("c1ccccc1").unwrap())
    );

    let inchi_cases = [
        "CCO",              // ethanol
        "CC(=O)O",          // acetic acid
        "c1ccccc1",         // benzene
        "c1ccncc1",         // pyridine
        "N[C@@H](C)C(=O)O", // L-alanine (tetrahedral /t + /m + /s1)
        "C/C=C/C",          // (E)-but-2-ene (/b2-1+)
        "C/C=C\\C",         // (Z)-but-2-ene (/b2-1-)
        "[13CH3]CO",        // isotopic ethanol (/i)
    ];

    for smi in inchi_cases {
        let mol = parse_smiles(smi).expect(smi);
        let inchi = write_inchi(&mol);
        assert!(
            inchi.starts_with("InChI=1S/"),
            "InChI for {}: {}",
            smi,
            inchi
        );

        let key = write_inchikey(&mol);
        assert_eq!(key.len(), 27, "InChIKey must be 27 chars: {}", key);
        assert_eq!(&key[14..15], "-");
        assert_eq!(&key[25..26], "-");
        assert_eq!(inchi_to_inchikey(&inchi).unwrap(), key);

        let parsed =
            parse_inchi(&inchi).unwrap_or_else(|e| panic!("parse_inchi({}): {}", inchi, e));
        assert_eq!(
            parsed.hill_formula(),
            mol.hill_formula(),
            "formula mismatch for {}",
            inchi
        );
        assert_eq!(
            write_canonical_smiles(&parsed),
            write_canonical_smiles(&mol),
            "connectivity/stereo mismatch after InChI round-trip for {} ({})",
            smi,
            inchi
        );
    }

    // Opposite enantiomers and E/Z isomers must produce distinct InChIs and distinct 2nd-block InChIKeys
    let l_ala = parse_smiles("N[C@@H](C)C(=O)O").unwrap();
    let d_ala = parse_smiles("N[C@H](C)C(=O)O").unwrap();
    assert_ne!(write_inchi(&l_ala), write_inchi(&d_ala));
    let k_l = write_inchikey(&l_ala);
    let k_d = write_inchikey(&d_ala);
    assert_eq!(
        &k_l[..14],
        &k_d[..14],
        "enantiomers share connectivity block"
    );
    assert_ne!(
        &k_l[15..25],
        &k_d[15..25],
        "enantiomers differ in stereo block"
    );
}

#[test]
fn test_mol_v2000_v3000_and_sdf_roundtrips() {
    // Rich chiral + isotopic + charged + radical molecule
    let mut mol = parse_smiles("[13CH3][C@@H](O)C(=O)[O-]").unwrap();
    mol.name = "Isotopic-(R)-Lactate".to_string();
    clean_2d(&mut mol);
    let cip_orig = perceive_cip(&mol);
    assert_eq!(cip_orig.atoms.len(), 1);

    // V2000 round-trip
    let v2000_txt = write_mol_v2000(&mol).unwrap();
    assert!(v2000_txt.contains("V2000"));
    assert!(v2000_txt.contains("M  CHG"));
    assert!(v2000_txt.contains("M  ISO"));
    let mol_v2 = read_mol(&v2000_txt).unwrap();
    assert_eq!(mol_v2.name, "Isotopic-(R)-Lactate");
    assert_eq!(mol_v2.atoms[0].isotope, Some(13));
    assert_eq!(mol_v2.total_charge(), -1);
    assert_eq!(
        perceive_cip(&mol_v2).atoms,
        cip_orig.atoms,
        "V2000 must preserve (R)/(S)"
    );

    // V3000 round-trip (including superatom SUP Sgroup and multi-center ENDPTS η5 bond!)
    let mut v3_src = mol.clone();
    let boc_idx = v3_src.add_atom(Atom::new("Boc"));
    v3_src.atoms[boc_idx].x = 4.2;
    v3_src.atoms[boc_idx].y = 1.5;
    v3_src.add_bond(2, boc_idx, BondKind::Single);

    let v3000_txt = write_mol_v3000(&v3_src);
    assert!(v3000_txt.contains("V3000"));
    assert!(v3000_txt.contains("BEGIN SGROUP"));
    let mol_v3 = read_mol(&v3000_txt).unwrap();
    assert_eq!(mol_v3.atoms[0].isotope, Some(13));
    assert_eq!(mol_v3.total_charge(), -1);
    assert_eq!(
        mol_v3.atoms[boc_idx].superatom_label.as_deref(),
        Some("Boc")
    );
    assert_eq!(
        perceive_cip(&mol_v3).atoms,
        cip_orig.atoms,
        "V3000 must preserve (R)/(S)"
    );

    // V3000 multi-center ENDPTS round-trip (η5-Cp- Fe)
    let mut mc = parse_smiles("[cH-]1cccc1").unwrap();
    clean_2d(&mut mc);
    let fe = mc.add_atom(Atom::from_z_xy(26, 0.0, 2.5));
    let mut star = Atom::new("*");
    star.multi_center_atoms = vec![0, 1, 2, 3, 4];
    let st_idx = mc.add_atom(star);
    mc.add_bond(st_idx, fe, BondKind::Dative);
    let mc_v3_txt = write_mol_v3000(&mc);
    assert!(mc_v3_txt.contains("ENDPTS=(5 1 2 3 4 5)"));
    let mc_rt = read_mol(&mc_v3_txt).unwrap();
    assert!(
        mc_rt.atoms.iter().any(|a| a.multi_center_atoms.len() == 5),
        "V3000 ENDPTS multi-center attachment must round-trip"
    );

    // Multi-record SDF round-trip with multi-line properties
    let mut rec1 = SdfRecord::new(mol.clone(), std::collections::BTreeMap::new());
    rec1.properties
        .insert("BMRB_ID".to_string(), "bmr17769".to_string());
    rec1.properties
        .insert("NOTES".to_string(), "Line 1\nLine 2".to_string());

    let mut m2 = parse_smiles("c1ccccc1").unwrap();
    m2.name = "Benzene".to_string();
    clean_2d(&mut m2);
    let mut rec2 = SdfRecord::new(m2, std::collections::BTreeMap::new());
    rec2.properties
        .insert("SOLVENT".to_string(), "CDCl3".to_string());

    let sdf_txt = write_sdf(&[rec1, rec2]);
    assert_eq!(sdf_txt.matches("$$$$").count(), 2);
    let recs_rt = read_sdf(&sdf_txt).unwrap();
    assert_eq!(recs_rt.len(), 2);
    assert_eq!(
        recs_rt[0].properties.get("BMRB_ID").map(String::as_str),
        Some("bmr17769")
    );
    assert_eq!(
        recs_rt[0].properties.get("NOTES").map(String::as_str),
        Some("Line 1\nLine 2")
    );
    assert_eq!(recs_rt[1].molecule.hill_formula(), "C6H6");
    assert_eq!(
        recs_rt[1].properties.get("SOLVENT").map(String::as_str),
        Some("CDCl3")
    );
}

#[test]
fn test_cdxml_binary_cdx_and_oct_full_document_roundtrips() {
    // Build a full reaction scheme ChemDocument with:
    // - Chiral molecule 0: L-Alanine (tests Y-axis handedness inversion between Y-up Å and Y-down CDXML/CDX pt!)
    // - Alkene molecule 1: (E)-but-2-ene + Boc superatom
    // - ReactionArrow (Equilibrium with above/below reagent text)
    // - 2-electron CurvedArrow (Bond -> Atom) & 1-electron radical fishhook CurvedArrow (Atom -> Bond)
    // - Polymer SRU Bracket ([...]_n^HT)
    // - Atomic Orbitals (P and Dz2)
    // - TLC Plate with 3 lanes and Rf spots
    // - Styled RichTextBox & CanvasShape
    let mut l_ala = parse_smiles("N[C@@H](C)C(=O)O").unwrap();
    clean_2d(&mut l_ala);
    assert_eq!(
        perceive_cip(&l_ala).atoms.get(&1),
        Some(&CipAtomDescriptor::S)
    );

    let mut e_alk = parse_smiles("C/C=C/C").unwrap();
    clean_2d(&mut e_alk);
    for a in &mut e_alk.atoms {
        a.x += 8.0;
    }

    let doc = ChemDocument {
        title: "SN2 & Polymerization Scheme".to_string(),
        molecules: vec![l_ala, e_alk],
        reaction_arrows: vec![ReactionArrow {
            kind: ReactionArrowKind::Equilibrium,
            tail: (3.0, 0.0),
            head: (6.0, 0.0),
            above_text: Some("Pd(PPh3)4, K2CO3".to_string()),
            below_text: Some("THF/H2O, 80 °C".to_string()),
        }],
        curved_arrows: vec![
            CurvedArrow {
                electrons: 2,
                start: (0.0, 1.0),
                control: (0.8, 2.2),
                end: (1.5, 0.8),
                source_anchor: Some(MechanismAnchor::Atom { mol: 0, atom: 0 }),
                target_anchor: Some(MechanismAnchor::Bond { mol: 0, bond: 0 }),
            },
            CurvedArrow {
                electrons: 1,
                start: (8.0, 0.5),
                control: (8.5, 1.8),
                end: (9.2, 0.5),
                source_anchor: Some(MechanismAnchor::Bond { mol: 1, bond: 1 }),
                target_anchor: Some(MechanismAnchor::Atom { mol: 1, atom: 2 }),
            },
        ],
        brackets: vec![Bracket {
            style: BracketStyle::Square,
            usage: BracketUsage::Sru,
            bounds: (6.5, -1.5, 11.5, 1.5),
            subscript: "n".to_string(),
            superscript: Some("HT".to_string()),
            enclosed_atoms: vec![(1, 0), (1, 1), (1, 2), (1, 3)],
        }],
        orbitals: vec![
            Orbital {
                kind: OrbitalKind::P,
                phase: OrbitalPhase::FilledPlus,
                center: (0.0, 0.0),
                axis_end: (0.0, 1.2),
                atom_anchor: Some((0, 1)),
            },
            Orbital {
                kind: OrbitalKind::Dz2,
                phase: OrbitalPhase::ShadedMinus,
                center: (4.5, -2.5),
                axis_end: (4.5, -1.1),
                atom_anchor: None,
            },
        ],
        tlc_plates: vec![TlcPlate {
            bounds: (13.0, -3.0, 17.0, 3.0),
            origin_fraction: 0.12,
            solvent_front_fraction: 0.10,
            lane_labels: vec!["SM".to_string(), "Co".to_string(), "Rxn".to_string()],
            spots: vec![
                TlcSpot {
                    lane: 0,
                    rf: 0.25,
                    style: TlcSpotStyle::Solid,
                    label: None,
                },
                TlcSpot {
                    lane: 1,
                    rf: 0.25,
                    style: TlcSpotStyle::Hollow,
                    label: None,
                },
                TlcSpot {
                    lane: 1,
                    rf: 0.68,
                    style: TlcSpotStyle::UvActive,
                    label: None,
                },
                TlcSpot {
                    lane: 2,
                    rf: 0.68,
                    style: TlcSpotStyle::Shaded,
                    label: Some("Prod".to_string()),
                },
            ],
        }],
        text_boxes: vec![RichTextBox {
            position: (2.0, 4.0),
            spans: vec![
                TextSpan {
                    text: "Scheme 1. ".to_string(),
                    bold: true,
                    italic: false,
                    subscript: false,
                    superscript: false,
                    font_size_pt: Some(10.0),
                    color_rgb: Some((20, 40, 120)),
                },
                TextSpan::plain("Catalytic coupling (94% yield)"),
            ],
        }],
        shapes: vec![CanvasShape {
            kind: ShapeKind::RoundedRectangle,
            bounds: (-2.5, -4.0, 18.0, 5.0),
            dashed: true,
            filled: false,
        }],
    };

    // 1. CDXML round-trip
    let cdxml_str = write_cdxml(&doc);
    assert!(cdxml_str.contains("<CDXML"));
    assert!(cdxml_str.contains("<fragment"));
    assert!(cdxml_str.contains("<tlcplate"));
    let doc_from_cdxml = read_cdxml(&cdxml_str).unwrap();
    assert_eq!(doc_from_cdxml.molecules.len(), 2);
    assert_eq!(
        perceive_cip(&doc_from_cdxml.molecules[0]).atoms.get(&1),
        Some(&CipAtomDescriptor::S),
        "CDXML round-trip must preserve (S)-Alanine wedge handedness!"
    );
    assert_eq!(
        perceive_cip(&doc_from_cdxml.molecules[1]).bonds.get(&1),
        Some(&CipBondDescriptor::E),
        "CDXML round-trip must preserve (E)-but-2-ene geometry!"
    );
    assert_eq!(doc_from_cdxml.reaction_arrows.len(), 1);
    assert_eq!(
        doc_from_cdxml.reaction_arrows[0].kind,
        ReactionArrowKind::Equilibrium
    );
    assert_eq!(doc_from_cdxml.curved_arrows.len(), 2);
    assert_eq!(doc_from_cdxml.curved_arrows[0].electrons, 2);
    assert_eq!(doc_from_cdxml.curved_arrows[1].electrons, 1);
    assert_eq!(doc_from_cdxml.brackets.len(), 1);
    assert_eq!(doc_from_cdxml.brackets[0].subscript, "n");
    assert_eq!(doc_from_cdxml.orbitals.len(), 2);
    assert_eq!(doc_from_cdxml.tlc_plates.len(), 1);
    assert_eq!(doc_from_cdxml.tlc_plates[0].spots.len(), 4);
    assert!((doc_from_cdxml.tlc_plates[0].spots[2].rf - 0.68).abs() < 1e-2);

    // 2. Binary CDX (`VjCD0100`) round-trip
    let cdx_bytes = write_cdx(&doc);
    assert!(cdx_bytes.len() > 28);
    assert_eq!(
        &cdx_bytes[0..8],
        CDX_MAGIC,
        "Binary CDX must start with VjCD0100"
    );
    let doc_from_cdx = read_cdx(&cdx_bytes).unwrap();
    assert_eq!(doc_from_cdx.molecules.len(), 2);
    assert_eq!(
        perceive_cip(&doc_from_cdx.molecules[0]).atoms.get(&1),
        Some(&CipAtomDescriptor::S),
        "Binary CDX round-trip must preserve (S)-Alanine wedge handedness!"
    );
    assert_eq!(
        perceive_cip(&doc_from_cdx.molecules[1]).bonds.get(&1),
        Some(&CipBondDescriptor::E),
        "Binary CDX round-trip must preserve (E)-but-2-ene geometry!"
    );
    assert_eq!(doc_from_cdx.reaction_arrows.len(), 1);
    assert_eq!(doc_from_cdx.curved_arrows.len(), 2);
    assert_eq!(doc_from_cdx.curved_arrows[1].electrons, 1);
    assert_eq!(doc_from_cdx.brackets.len(), 1);
    assert_eq!(doc_from_cdx.orbitals.len(), 2);
    assert_eq!(doc_from_cdx.tlc_plates.len(), 1);
    assert_eq!(doc_from_cdx.tlc_plates[0].spots.len(), 4);

    // 3. Lossless Octet `.oct` round-trip (intact FNV-1a fast path + external-edit fallback path)
    let oct_str = write_oct(&doc).unwrap();
    assert!(oct_str.contains("<!--Octet 1 "));
    let doc_from_oct = read_oct(&oct_str).unwrap();
    assert_eq!(doc_from_oct.title, doc.title);
    assert_eq!(doc_from_oct.molecules.len(), doc.molecules.len());
    assert_eq!(
        doc_from_oct.molecules[0].atoms[1].parity,
        doc.molecules[0].atoms[1].parity
    );
    assert_eq!(
        doc_from_oct.molecules[1].bonds[1].double_geometry,
        doc.molecules[1].bonds[1].double_geometry
    );
    assert_eq!(doc_from_oct.reaction_arrows, doc.reaction_arrows);
    assert_eq!(doc_from_oct.curved_arrows, doc.curved_arrows);
    assert_eq!(doc_from_oct.brackets, doc.brackets);
    assert_eq!(doc_from_oct.orbitals, doc.orbitals);
    assert_eq!(doc_from_oct.tlc_plates, doc.tlc_plates);
    assert_eq!(doc_from_oct.text_boxes, doc.text_boxes);
    assert_eq!(doc_from_oct.shapes, doc.shapes);
    for (m_a, m_b) in doc_from_oct.molecules.iter().zip(doc.molecules.iter()) {
        for (a, b) in m_a.atoms.iter().zip(m_b.atoms.iter()) {
            assert!((a.x - b.x).abs() < 1e-14 && (a.y - b.y).abs() < 1e-14);
        }
    }

    // Simulate external ChemDraw edit by modifying the CDXML body BEFORE `<!--Octet 1 ` -> FNV hash mismatch -> graceful CDXML fallback
    let edited_oct = oct_str.replacen(
        "</CDXML>",
        "<!-- edited in external CDXML tool --></CDXML>",
        1,
    );
    let doc_fallback = read_oct(&edited_oct).unwrap();
    assert_eq!(doc_fallback.molecules.len(), 2);
    assert_eq!(
        perceive_cip(&doc_fallback.molecules[0]).atoms.get(&1),
        Some(&CipAtomDescriptor::S)
    );
}
