//! Comprehensive topological, geometric, statistical, and headless `egui` integration tests for
//! `T10.1` (`nmr_core::topology` 2D layouts & `nmr_gui::sidechain_widget`).
//!
//! Provenance: Clean-room verification suite grounded in:
//! - Markley et al. (*J. Mol. Biol.* 1998, 280, 933–952 — IUPAC-IUBMB-IUPAB biochemical NMR atom
//!   nomenclature);
//! - wwPDB Chemical Component Dictionary (`CCD`, public domain / CC0 1.0);
//! - Ulrich et al. (*Nucleic Acids Res.* 2008, 36, D402–D408 — BMRB empirical chemical shift
//!   statistics).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use eframe::egui;
use nmr_core::priors::Priors;
use nmr_core::topology::{
    all_residue_topologies, compute_atom_bmrb_zscore, residue_layout_2d, residue_topology,
    CovalentBondOrder, Element, ResidueLayout2D, ResidueVariant, SidechainPriors,
};
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::sidechain_widget::{
    evaluate_atom_node, AssignedAtomInfo, AtomZScoreClass, SequenceRibbonState,
    SidechainInspectorPanel, SidechainWidgetOutput, SidechainWorkspaceInput,
};

/// Euclidean distance from point `p` to line segment `a-b`.
fn point_to_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let vx = b.0 - a.0;
    let vy = b.1 - a.1;
    let wx = p.0 - a.0;
    let wy = p.1 - a.1;
    let len_sq = vx * vx + vy * vy;
    if len_sq <= 1e-12 {
        return (wx * wx + wy * wy).sqrt();
    }
    let t = ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0);
    let proj_x = a.0 + t * vx;
    let proj_y = a.1 + t * vy;
    ((p.0 - proj_x).powi(2) + (p.1 - proj_y).powi(2)).sqrt()
}

fn orient_2d(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// True if closed segments `a1-a2` and `b1-b2` (with disjoint endpoints) intersect.
fn non_incident_segments_intersect(
    a1: (f64, f64),
    a2: (f64, f64),
    b1: (f64, f64),
    b2: (f64, f64),
) -> bool {
    let eps = 1e-9;
    let o1 = orient_2d(a1, a2, b1);
    let o2 = orient_2d(a1, a2, b2);
    let o3 = orient_2d(b1, b2, a1);
    let o4 = orient_2d(b1, b2, a2);

    if ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps))
        && ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))
    {
        return true;
    }

    // Collinear overlap guard
    let on_seg = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| -> bool {
        q.0 >= p.0.min(r.0) - eps
            && q.0 <= p.0.max(r.0) + eps
            && q.1 >= p.1.min(r.1) - eps
            && q.1 <= p.1.max(r.1) + eps
    };
    (o1.abs() <= eps && on_seg(a1, b1, a2))
        || (o2.abs() <= eps && on_seg(a1, b2, a2))
        || (o3.abs() <= eps && on_seg(b1, a1, b2))
        || (o4.abs() <= eps && on_seg(b1, a2, b2))
}

fn verify_layout_invariants(layout: &ResidueLayout2D) {
    let tag = format!("{}({:?})", layout.three_letter, layout.variant);
    assert!(!layout.atoms.is_empty(), "{tag}: empty atoms");
    assert!(!layout.bonds.is_empty(), "{tag}: empty bonds");

    // 1. Unique atom names & finite coordinates
    let mut names = BTreeSet::new();
    for a in &layout.atoms {
        assert!(
            names.insert(a.name),
            "{tag}: duplicate atom label {}",
            a.name
        );
        assert!(
            a.x.is_finite() && a.y.is_finite(),
            "{tag}: non-finite coords on {}",
            a.name
        );
    }

    // 2. Every bond endpoint exists & graph is connected (single connected component)
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for a in &layout.atoms {
        adj.insert(a.name, Vec::new());
    }
    for b in &layout.bonds {
        assert_ne!(b.atom_a, b.atom_b, "{tag}: self-bond on {}", b.atom_a);
        assert!(
            names.contains(b.atom_a),
            "{tag}: missing bond atom_a {}",
            b.atom_a
        );
        assert!(
            names.contains(b.atom_b),
            "{tag}: missing bond atom_b {}",
            b.atom_b
        );
        adj.get_mut(b.atom_a).unwrap().push(b.atom_b);
        adj.get_mut(b.atom_b).unwrap().push(b.atom_a);
    }

    let mut visited = BTreeSet::new();
    let mut q = VecDeque::new();
    let start = layout.atoms[0].name;
    visited.insert(start);
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if visited.insert(v) {
                q.push_back(v);
            }
        }
    }
    assert_eq!(
        visited.len(),
        layout.atoms.len(),
        "{tag}: disconnected covalent graph ({}/{} reachable)",
        visited.len(),
        layout.atoms.len()
    );

    // 3. Pairwise Euclidean distance >= 0.35 normalized bond units for ALL atom pairs
    let mut min_pair_dist = f64::INFINITY;
    for i in 0..layout.atoms.len() {
        for j in (i + 1)..layout.atoms.len() {
            let ai = &layout.atoms[i];
            let aj = &layout.atoms[j];
            let d = ((ai.x - aj.x).powi(2) + (ai.y - aj.y).powi(2)).sqrt();
            if d < min_pair_dist {
                min_pair_dist = d;
            }
            assert!(
                d >= 0.35,
                "{tag}: overlapping atoms {} and {} at distance {d:.4} (< 0.35)",
                ai.name,
                aj.name
            );
        }
    }
    assert!(min_pair_dist >= 0.35);

    // 4. Zero bond-node collisions (every non-incident atom stays well clear of every bond segment)
    for node in &layout.atoms {
        for bond in &layout.bonds {
            if node.name == bond.atom_a || node.name == bond.atom_b {
                continue;
            }
            let a = layout.atom(bond.atom_a).unwrap();
            let b = layout.atom(bond.atom_b).unwrap();
            let d_seg = point_to_segment_distance((node.x, node.y), (a.x, a.y), (b.x, b.y));
            assert!(
                d_seg >= 0.25,
                "{tag}: bond-node collision between atom {} and bond {}-{} (dist = {d_seg:.4})",
                node.name,
                bond.atom_a,
                bond.atom_b
            );
        }
    }

    // 5. Planarity: zero crossings between non-adjacent covalent bond segments
    for i in 0..layout.bonds.len() {
        for j in (i + 1)..layout.bonds.len() {
            let b1 = &layout.bonds[i];
            let b2 = &layout.bonds[j];
            if b1.atom_a == b2.atom_a
                || b1.atom_a == b2.atom_b
                || b1.atom_b == b2.atom_a
                || b1.atom_b == b2.atom_b
            {
                continue;
            }
            let p1 = layout.atom(b1.atom_a).unwrap();
            let p2 = layout.atom(b1.atom_b).unwrap();
            let q1 = layout.atom(b2.atom_a).unwrap();
            let q2 = layout.atom(b2.atom_b).unwrap();
            assert!(
                !non_incident_segments_intersect(
                    (p1.x, p1.y),
                    (p2.x, p2.y),
                    (q1.x, q1.y),
                    (q2.x, q2.y)
                ),
                "{tag}: non-planar crossing between bonds {}-{} and {}-{}",
                b1.atom_a,
                b1.atom_b,
                b2.atom_a,
                b2.atom_b
            );
        }
    }
}

#[test]
fn test_all_20_canonical_and_5_variant_layouts_planar_connected_and_non_overlapping() {
    let topos = all_residue_topologies();
    assert_eq!(topos.len(), 20);

    let mut observed_all_atoms: BTreeSet<&'static str> = BTreeSet::new();

    for topo in topos {
        let layout = topo.layout_2d(ResidueVariant::Canonical);
        verify_layout_invariants(&layout);

        // Exact IUPAC atom and bond parity with canonical `ResidueTopology`, except `HIS`
        // `Canonical` intentionally omits exchangeable `HD1` and `HE2` so neutral uncommitted
        // histidine never falsely depicts cationic imidazolium `HisH+`.
        let mut expected_atoms: BTreeSet<&str> = topo.atoms.iter().map(|a| a.name).collect();
        let expected_bond_len = if topo.three_letter == "HIS" {
            expected_atoms.remove("HD1");
            expected_atoms.remove("HE2");
            topo.bonds.len() - 2
        } else {
            topo.bonds.len()
        };
        let actual_atoms: BTreeSet<&str> = layout.atoms.iter().map(|a| a.name).collect();
        assert_eq!(
            actual_atoms, expected_atoms,
            "{}: IUPAC atom set mismatch",
            topo.three_letter
        );
        assert_eq!(
            layout.bonds.len(),
            expected_bond_len,
            "{}: covalent bond count mismatch",
            topo.three_letter
        );

        // Peptide carbonyl C=O is always a Double bond
        assert_eq!(
            layout.bond_order("C", "O"),
            Some(CovalentBondOrder::Double),
            "{}: C=O must be Double",
            topo.three_letter
        );

        for a in &layout.atoms {
            observed_all_atoms.insert(a.name);
        }
    }

    // Verify full IUPAC coverage across backbone & sidechain classes
    for required_atom in [
        "N", "H", "CA", "HA", "HA2", "HA3", "C", "O", "CB", "HB", "HB1", "HB2", "HB3", "CG", "CG1",
        "CG2", "HG", "HG1", "HG2", "HG3", "HG12", "HG13", "HG21", "HG22", "HG23", "CD", "CD1",
        "CD2", "HD1", "HD2", "HD3", "HD11", "HD21", "CE", "CE1", "CE2", "CE3", "HE", "HE1", "HE2",
        "HE3", "HE21", "HE22", "CZ", "CZ2", "CZ3", "CH2", "HZ", "HZ1", "HZ2", "HZ3", "HH", "HH2",
        "HH11", "HH12", "HH21", "HH22", "NH1", "NH2", "NE", "NE1", "NE2", "ND1", "ND2", "NZ", "OG",
        "OG1", "OH", "OD1", "OD2", "OE1", "OE2", "SD", "SG",
    ] {
        assert!(
            observed_all_atoms.contains(required_atom),
            "Missing required IUPAC atom {required_atom} across 20 canonical layouts"
        );
    }

    // Verify all 5 Cys / His chemical state variants
    let cys_red = residue_layout_2d("CYS", ResidueVariant::CysReduced).unwrap();
    verify_layout_invariants(&cys_red);
    assert!(cys_red.atom("HG").is_some());
    assert!(cys_red.are_bonded("SG", "HG"));

    let cys_ox = residue_layout_2d("CYS", ResidueVariant::CysOxidized).unwrap();
    verify_layout_invariants(&cys_ox);
    assert!(
        cys_ox.atom("HG").is_none(),
        "CysOxidized must omit thiol HG"
    );
    assert!(!cys_ox.are_bonded("SG", "HG"));

    let his_nd1 = residue_layout_2d("HIS", ResidueVariant::HisNd1H).unwrap();
    verify_layout_invariants(&his_nd1);
    assert!(his_nd1.atom("HD1").is_some());
    assert!(his_nd1.atom("HE2").is_none());

    let his_ne2 = residue_layout_2d("HIS", ResidueVariant::HisNe2H).unwrap();
    verify_layout_invariants(&his_ne2);
    assert!(his_ne2.atom("HD1").is_none());
    assert!(his_ne2.atom("HE2").is_some());

    let his_2h = residue_layout_2d("HIS", ResidueVariant::HisDoublyProtonated).unwrap();
    verify_layout_invariants(&his_2h);
    assert!(his_2h.atom("HD1").is_some());
    assert!(his_2h.atom("HE2").is_some());
}

#[test]
fn test_regular_5_and_6_membered_rings_and_bond_orders() {
    for (res_code, expected_ring_count) in [
        ("PRO", 1usize),
        ("HIS", 1),
        ("PHE", 1),
        ("TYR", 1),
        ("TRP", 2),
    ] {
        let layout = residue_layout_2d(res_code, ResidueVariant::Canonical).unwrap();
        assert_eq!(layout.rings.len(), expected_ring_count, "{res_code} rings");

        for ring in &layout.rings {
            let n = ring.atoms.len();
            assert!(n == 5 || n == 6, "{res_code}: unexpected ring size {n}");
            let expected_interior_deg = ((n - 2) as f64) * 180.0 / (n as f64);

            for k in 0..n {
                let prev = layout.atom(ring.atoms[(k + n - 1) % n]).unwrap();
                let curr = layout.atom(ring.atoms[k]).unwrap();
                let next = layout.atom(ring.atoms[(k + 1) % n]).unwrap();

                let edge_len = ((next.x - curr.x).powi(2) + (next.y - curr.y).powi(2)).sqrt();
                assert!(
                    (edge_len - 1.0).abs() < 1e-6,
                    "{res_code}: ring edge {}-{} length {edge_len:.6} != 1.0",
                    curr.name,
                    next.name
                );

                let v1 = (prev.x - curr.x, prev.y - curr.y);
                let v2 = (next.x - curr.x, next.y - curr.y);
                let cos_ang = ((v1.0 * v2.0 + v1.1 * v2.1)
                    / ((v1.0.hypot(v1.1)) * (v2.0.hypot(v2.1))))
                .clamp(-1.0, 1.0);
                let ang_deg = cos_ang.acos().to_degrees();
                assert!(
                    (ang_deg - expected_interior_deg).abs() < 1e-4,
                    "{res_code}: ring interior angle at {} = {ang_deg:.4} != {expected_interior_deg:.4}",
                    curr.name
                );

                let expected_order = if ring.is_aromatic {
                    CovalentBondOrder::Aromatic
                } else {
                    CovalentBondOrder::Single
                };
                assert_eq!(
                    layout.bond_order(curr.name, next.name),
                    Some(expected_order),
                    "{res_code}: unexpected bond order on {}-{}",
                    curr.name,
                    next.name
                );
            }
        }
    }

    // Delocalized resonance bonds on ASP, GLU, ARG and amide double bonds on ASN, GLN
    let asp = residue_layout_2d("ASP", ResidueVariant::Canonical).unwrap();
    assert_eq!(
        asp.bond_order("CG", "OD1"),
        Some(CovalentBondOrder::Resonance)
    );
    assert_eq!(
        asp.bond_order("CG", "OD2"),
        Some(CovalentBondOrder::Resonance)
    );

    let arg = residue_layout_2d("ARG", ResidueVariant::Canonical).unwrap();
    assert_eq!(
        arg.bond_order("CZ", "NH1"),
        Some(CovalentBondOrder::Resonance)
    );
    assert_eq!(
        arg.bond_order("CZ", "NH2"),
        Some(CovalentBondOrder::Resonance)
    );

    let asn = residue_layout_2d("ASN", ResidueVariant::Canonical).unwrap();
    assert_eq!(asn.bond_order("CG", "OD1"), Some(CovalentBondOrder::Double));
}

#[test]
fn test_bmrb_zscore_bands_and_cys_his_state_variants() {
    let priors = Priors::bmrb();
    let sc_priors = SidechainPriors::empirical_bmrb();

    // Exact boundary checks on AtomZScoreClass
    assert_eq!(
        AtomZScoreClass::from_zscore(0.0),
        AtomZScoreClass::NormalGreen
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(1.5),
        AtomZScoreClass::NormalGreen
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(-1.5),
        AtomZScoreClass::NormalGreen
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(1.5001),
        AtomZScoreClass::WarningAmber
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(3.0),
        AtomZScoreClass::WarningAmber
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(-3.0),
        AtomZScoreClass::WarningAmber
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(3.0001),
        AtomZScoreClass::OutlierCrimson
    );
    assert_eq!(
        AtomZScoreClass::from_zscore(-4.2),
        AtomZScoreClass::OutlierCrimson
    );

    // ALA CA normal, warning, and outlier shifts
    let (mu_ca, sd_ca, z0) = compute_atom_bmrb_zscore(
        "ALA",
        ResidueVariant::Canonical,
        "CA",
        53.35,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(z0.abs() <= 1.5);

    let ala_layout = residue_layout_2d("ALA", ResidueVariant::Canonical).unwrap();
    let ca_node = ala_layout.atom("CA").unwrap();
    let o_node = ala_layout.atom("O").unwrap();
    let cb_node = ala_layout.atom("CB").unwrap();

    // 1. |z| <= 1.5 -> NormalGreen
    let eval_green = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        ca_node,
        Some(&AssignedAtomInfo::with_metadata(
            mu_ca + 1.0 * sd_ca,
            Some(0.08),
            Some(1),
        )),
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval_green.class, AtomZScoreClass::NormalGreen);
    assert!(eval_green.shift_caption().unwrap().contains("±0.08"));
    let tip = eval_green.build_tooltip("ALA", 5, ResidueVariant::Canonical);
    assert!(tip.atom.contains("ALA 5 — CA"));
    assert!(tip.assigned_ppm.contains("+- 0.080 ppm"));
    assert!(tip.bmrb_prior.contains("+-"));
    assert!(tip.z_score.contains("Normal"));
    assert!(tip.ambiguity_code.starts_with('1'));

    // 2. 1.5 < |z| <= 3.0 -> WarningAmber
    let eval_amber = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        ca_node,
        Some(&AssignedAtomInfo::new(mu_ca - 2.2 * sd_ca)),
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval_amber.class, AtomZScoreClass::WarningAmber);

    // 3. |z| > 3.0 -> OutlierCrimson
    let eval_red = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        ca_node,
        Some(&AssignedAtomInfo::new(mu_ca + 3.8 * sd_ca)),
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval_red.class, AtomZScoreClass::OutlierCrimson);

    // 4. Unassigned NMR-active vs non-observable carbonyl/quaternary
    let eval_unassigned = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        cb_node,
        None,
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval_unassigned.class, AtomZScoreClass::UnassignedHollow);

    let eval_oxygen = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        o_node,
        None,
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval_oxygen.class, AtomZScoreClass::MutedNonObservable);

    for arom in ["PHE", "TYR", "TRP", "HIS"] {
        let lay = residue_layout_2d(arom, ResidueVariant::Canonical).unwrap();
        let cg = lay.atom("CG").unwrap();
        assert!(
            !cg.is_nmr_observable,
            "{arom} CG should be muted quaternary"
        );
        let ev = evaluate_atom_node(
            arom,
            ResidueVariant::Canonical,
            cg,
            None,
            None,
            &priors,
            &sc_priors,
        );
        assert_eq!(ev.class, AtomZScoreClass::MutedNonObservable);
    }

    // 5. CysReduced (CB ~ 28 ppm) vs CysOxidized (CB ~ 41 ppm, HG absent)
    let (_, _, z_cys_red_ok) = compute_atom_bmrb_zscore(
        "CYS",
        ResidueVariant::CysReduced,
        "CB",
        28.1,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(z_cys_red_ok.abs() <= 1.5);
    assert!(compute_atom_bmrb_zscore(
        "CYS",
        ResidueVariant::CysReduced,
        "HG",
        2.05,
        &priors,
        &sc_priors
    )
    .is_some());

    let (_, _, z_cys_red_mismatch) = compute_atom_bmrb_zscore(
        "CYS",
        ResidueVariant::CysReduced,
        "CB",
        41.2,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(
        z_cys_red_mismatch.abs() > 3.0,
        "Disulfide CB=41.2 ppm under CysReduced must be |z| > 3.0 outlier"
    );

    let (_, _, z_cys_ox_ok) = compute_atom_bmrb_zscore(
        "CYS",
        ResidueVariant::CysOxidized,
        "CB",
        41.1,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(z_cys_ox_ok.abs() <= 1.5);
    assert!(
        compute_atom_bmrb_zscore(
            "CYS",
            ResidueVariant::CysOxidized,
            "HG",
            2.05,
            &priors,
            &sc_priors
        )
        .is_none(),
        "CysOxidized has no HG prior"
    );

    // 6. His tautomers (Pelton et al. 1993 15N chemical shift Inversion)
    let (_, _, z_nd1_prot) = compute_atom_bmrb_zscore(
        "HIS",
        ResidueVariant::HisNd1H,
        "ND1",
        168.0,
        &priors,
        &sc_priors,
    )
    .unwrap();
    let (_, _, z_ne2_unprot) = compute_atom_bmrb_zscore(
        "HIS",
        ResidueVariant::HisNd1H,
        "NE2",
        249.0,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(z_nd1_prot.abs() <= 1.5 && z_ne2_unprot.abs() <= 1.5);

    let (_, _, z_ne2_prot) = compute_atom_bmrb_zscore(
        "HIS",
        ResidueVariant::HisNe2H,
        "NE2",
        168.0,
        &priors,
        &sc_priors,
    )
    .unwrap();
    let (_, _, z_nd1_unprot) = compute_atom_bmrb_zscore(
        "HIS",
        ResidueVariant::HisNe2H,
        "ND1",
        249.0,
        &priors,
        &sc_priors,
    )
    .unwrap();
    assert!(z_ne2_prot.abs() <= 1.5 && z_nd1_unprot.abs() <= 1.5);
}

fn simulate_click(
    ctx: &egui::Context,
    pos: egui::Pos2,
    mut run_frame: impl FnMut(&mut egui::Ui) -> SidechainWidgetOutput,
) -> SidechainWidgetOutput {
    let screen_rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(960.0, 640.0));

    let mut press_in = egui::RawInput {
        screen_rect: Some(screen_rect),
        ..egui::RawInput::default()
    };
    press_in.events.push(egui::Event::PointerMoved(pos));
    press_in.events.push(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run_ui(press_in, |ui| {
        let _ = run_frame(ui);
    });

    let mut release_in = egui::RawInput {
        screen_rect: Some(screen_rect),
        ..egui::RawInput::default()
    };
    release_in.events.push(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    let mut out = SidechainWidgetOutput {
        selected_residue: 0,
        jump_to_atom: None,
        variant_override: None,
    };
    let _ = ctx.run_ui(release_in, |ui| {
        out = run_frame(ui);
    });
    out
}

#[test]
fn test_headless_egui_ribbon_selection_atom_jump_and_variant_toggle() {
    let priors = Priors::bmrb();
    let sc_priors = SidechainPriors::empirical_bmrb();

    let sequence = vec![
        "ALA".to_string(),
        "CYS".to_string(),
        "HIS".to_string(),
        "TRP".to_string(),
        "ILE".to_string(),
    ];
    let seq_ids = vec![10, 11, 12, 13, 14];

    let mut assigned = BTreeMap::new();
    // Residue 1 (CYS 11)
    assigned.insert((1, "H".to_string()), AssignedAtomInfo::new(8.35));
    assigned.insert((1, "N".to_string()), AssignedAtomInfo::new(121.90));
    assigned.insert((1, "CB".to_string()), AssignedAtomInfo::new(41.05));
    // Residue 4 (ILE 14)
    assigned.insert((4, "H".to_string()), AssignedAtomInfo::new(8.29));
    assigned.insert((4, "N".to_string()), AssignedAtomInfo::new(122.10));
    assigned.insert(
        (4, "CA".to_string()),
        AssignedAtomInfo::with_metadata(61.40, Some(0.05), Some(1)),
    );
    assigned.insert(
        (4, "CB".to_string()),
        AssignedAtomInfo::with_metadata(39.12, Some(0.04), Some(1)),
    );
    assigned.insert((4, "HB".to_string()), AssignedAtomInfo::new(1.76));
    assigned.insert(
        (4, "CD1".to_string()),
        AssignedAtomInfo::with_metadata(13.80, Some(0.06), Some(1)),
    );
    assigned.insert((4, "HD11".to_string()), AssignedAtomInfo::new(0.71));

    let mut confs = BTreeMap::new();
    confs.insert(1, 0.92);
    confs.insert(4, 0.96);

    let input = SidechainWorkspaceInput::from_assigned_map(
        &sequence, &seq_ids, assigned, &priors, &sc_priors,
    )
    .with_backbone_confidences(confs);

    let ctx = egui::Context::default();
    let mut state = SequenceRibbonState::new();
    let screen_rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(960.0, 640.0));

    // Frame 0: initial layout pass (selected_residue == 0 -> ALA 10)
    let raw0 = egui::RawInput {
        screen_rect: Some(screen_rect),
        ..egui::RawInput::default()
    };
    let full_out0 = ctx.run_ui(raw0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let out = SidechainInspectorPanel::show(ui, &mut state, &input);
            assert_eq!(out.selected_residue, 0);
            assert!(out.jump_to_atom.is_none());
        });
    });
    assert!(
        !full_out0.shapes.is_empty(),
        "Painter must emit tessellatable shapes"
    );
    assert_eq!(state.last_ribbon_cell_rects.len(), 5);
    assert!(state.last_atom_hit_rects.contains_key("HB1"));

    // Click ribbon cell #4 (ILE 14) -> selects residue 4 and renders ILE sidechain graph
    let ile_ribbon_center = state.last_ribbon_cell_rects[4].center();
    let out_select_ile = simulate_click(&ctx, ile_ribbon_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(out_select_ile.selected_residue, 4);
    assert_eq!(state.selected_residue, 4);
    assert!(state.last_atom_hit_rects.contains_key("CD1"));
    assert!(state.last_atom_hit_rects.contains_key("CG2"));

    // Click unassigned CG2 node on ILE 14 -> must NOT emit AtomJumpTarget
    let cg2_center = state.last_atom_hit_rects["CG2"].center();
    let out_unassigned_click = simulate_click(&ctx, cg2_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert!(out_unassigned_click.jump_to_atom.is_none());

    // Click assigned CB atom node on ILE 14 -> emits AtomJumpTarget with exact shift & crosshair sync
    let cb_center = state.last_atom_hit_rects["CB"].center();
    let SidechainWidgetOutput {
        selected_residue,
        jump_to_atom,
        variant_override,
    } = simulate_click(&ctx, cb_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(selected_residue, 4);
    assert!(variant_override.is_none());

    let jump = jump_to_atom.expect("clicking assigned CB node must emit AtomJumpTarget");
    assert_eq!(jump.residue_idx, 4);
    assert_eq!(jump.seq_id, 14);
    assert_eq!(jump.residue_name, "ILE");
    assert_eq!(jump.atom_name, "CB");
    assert_eq!(jump.element, Element::C);
    assert!((jump.ppm - 39.12).abs() < 1e-9);
    assert_eq!(jump.c13_plane_coords(), Some((1.76, 39.12)));
    assert_eq!(jump.n15_plane_coords(), Some((8.29, 122.10)));

    let mut crosshair = SyncCrosshair::default();
    jump.apply_to_crosshair(&mut crosshair);
    assert!((crosshair.h_ppm - 8.29).abs() < 1e-9);
    assert!((crosshair.n_ppm - 122.10).abs() < 1e-9);
    assert_eq!(crosshair.c_ppm, Some(39.12));

    // Select CYS 11 (index 1) and click the CysOxidized UI variant toggle button
    let cys_ribbon_center = state.last_ribbon_cell_rects[1].center();
    let _ = simulate_click(&ctx, cys_ribbon_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(state.selected_residue, 1);
    assert!(state.last_atom_hit_rects.contains_key("HG"));

    let cys_ox_btn_center = state.last_variant_button_rects[&ResidueVariant::CysOxidized].center();
    let out_cys_ox = simulate_click(&ctx, cys_ox_btn_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(
        out_cys_ox.variant_override,
        Some((1, ResidueVariant::CysOxidized))
    );
    assert_eq!(state.variant_for(1), ResidueVariant::CysOxidized);

    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(screen_rect),
            ..egui::RawInput::default()
        },
        |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = SidechainInspectorPanel::show(ui, &mut state, &input);
            });
        },
    );
    assert!(
        !state.last_atom_hit_rects.contains_key("HG"),
        "CysOxidized toggle must remove HG from 2D canvas"
    );

    // Select HIS 12 (index 2) and click the HisNe2H UI variant toggle button
    let his_ribbon_center = state.last_ribbon_cell_rects[2].center();
    let _ = simulate_click(&ctx, his_ribbon_center, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(state.selected_residue, 2);
    let his_ne2_btn = state.last_variant_button_rects[&ResidueVariant::HisNe2H].center();
    let out_his_ne2 = simulate_click(&ctx, his_ne2_btn, |ui| {
        let mut out = SidechainWidgetOutput {
            selected_residue: 0,
            jump_to_atom: None,
            variant_override: None,
        };
        egui::CentralPanel::default().show(ui, |ui| {
            out = SidechainInspectorPanel::show(ui, &mut state, &input);
        });
        out
    });
    assert_eq!(
        out_his_ne2.variant_override,
        Some((2, ResidueVariant::HisNe2H))
    );

    // Render all 20 canonical amino acids headlessly without panics or missing rects
    let all_seq: Vec<String> = all_residue_topologies()
        .iter()
        .map(|t| t.three_letter.to_string())
        .collect();
    let all_ids: Vec<i32> = (1..=20).collect();
    let all_input = SidechainWorkspaceInput::from_assigned_map(
        &all_seq,
        &all_ids,
        BTreeMap::new(),
        &priors,
        &sc_priors,
    );
    for (idx, res_name) in all_seq.iter().enumerate() {
        state.selected_residue = idx;
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(screen_rect),
                ..egui::RawInput::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let out = SidechainInspectorPanel::show(ui, &mut state, &all_input);
                    assert_eq!(out.selected_residue, idx);
                });
            },
        );
        let raw_len = residue_topology(res_name).unwrap().atoms.len();
        let expected_len = if res_name == "HIS" {
            raw_len - 2
        } else {
            raw_len
        };
        assert_eq!(state.last_atom_hit_rects.len(), expected_len);
    }
}

#[test]
fn test_his_canonical_omits_exchangeable_hd1_and_he2() {
    let his_canon = residue_layout_2d("HIS", ResidueVariant::Canonical).unwrap();
    assert!(
        his_canon.atom("HD1").is_none() && his_canon.atom("HE2").is_none(),
        "HIS Canonical must omit both exchangeable HD1 and HE2"
    );
    assert!(his_canon.bond_order("ND1", "HD1").is_none());
    assert!(his_canon.bond_order("NE2", "HE2").is_none());
    assert!(his_canon.atom("HD2").is_some() && his_canon.atom("HE1").is_some());

    let his_nd1h = residue_layout_2d("HIS", ResidueVariant::HisNd1H).unwrap();
    assert!(his_nd1h.atom("HD1").is_some() && his_nd1h.atom("HE2").is_none());

    let his_ne2h = residue_layout_2d("HIS", ResidueVariant::HisNe2H).unwrap();
    assert!(his_ne2h.atom("HD1").is_none() && his_ne2h.atom("HE2").is_some());

    let his_prot = residue_layout_2d("HIS", ResidueVariant::HisDoublyProtonated).unwrap();
    assert!(his_prot.atom("HD1").is_some() && his_prot.atom("HE2").is_some());
}

#[test]
fn test_zero_or_nan_prior_sigma_returns_none_and_renders_no_bmrb_prior() {
    let mut priors = Priors::default();
    let sc_priors = SidechainPriors::empirical_bmrb();

    // Inject degenerate sd = 0.0, negative sd, and NaN into backbone priors for ALA CA / N / H
    let ala_map = priors.table.entry("ALA".to_string()).or_default();
    ala_map.insert("CA".to_string(), (53.35, 0.0));
    ala_map.insert("N".to_string(), (123.5, -1.0));
    ala_map.insert("H".to_string(), (f64::NAN, 0.45));

    assert_eq!(
        compute_atom_bmrb_zscore(
            "ALA",
            ResidueVariant::Canonical,
            "CA",
            55.0,
            &priors,
            &sc_priors
        ),
        None
    );
    assert_eq!(
        compute_atom_bmrb_zscore(
            "ALA",
            ResidueVariant::Canonical,
            "N",
            124.0,
            &priors,
            &sc_priors
        ),
        None
    );
    assert_eq!(
        compute_atom_bmrb_zscore(
            "ALA",
            ResidueVariant::Canonical,
            "H",
            8.2,
            &priors,
            &sc_priors
        ),
        None
    );

    let ala_layout = residue_layout_2d("ALA", ResidueVariant::Canonical).unwrap();
    let ca_node = ala_layout.atom("CA").unwrap();
    assert!(ca_node.is_nmr_observable);

    let eval = evaluate_atom_node(
        "ALA",
        ResidueVariant::Canonical,
        ca_node,
        Some(&AssignedAtomInfo::new(55.0)),
        None,
        &priors,
        &sc_priors,
    );
    assert_eq!(eval.class, AtomZScoreClass::NoPriorNeutral);
    assert_ne!(eval.class, AtomZScoreClass::OutlierCrimson);
    assert!(eval.z_score.is_none());
    assert!(eval.bmrb_mu_sd.is_none());

    let tip = eval.build_tooltip("ALA", 1, ResidueVariant::Canonical);
    assert_eq!(tip.bmrb_prior, "No BMRB prior");
    assert_eq!(tip.z_score, "No BMRB prior");

    // Also verify observable atom node on a non-standard residue lacking BMRB priors (e.g. MSE CA)
    let eval_mse = evaluate_atom_node(
        "MSE",
        ResidueVariant::Canonical,
        ca_node,
        Some(&AssignedAtomInfo::new(55.4)),
        None,
        &Priors::default(),
        &sc_priors,
    );
    assert_eq!(eval_mse.class, AtomZScoreClass::NoPriorNeutral);
    assert_ne!(eval_mse.class, AtomZScoreClass::OutlierCrimson);
    assert_eq!(
        eval_mse
            .build_tooltip("MSE", 7, ResidueVariant::Canonical)
            .z_score,
        "No BMRB prior"
    );
}

#[test]
fn test_sidechain_inspector_state_sync_inferred_variants_preserves_user_overrides() {
    use nmr_core::nmrstar::Reference;
    use nmr_gui::sidechain_widget::SidechainInspectorState;

    let mut state = SidechainInspectorState::new();
    let mut ref_data = Reference {
        sequence: vec![
            "CYS".to_string(),
            "CYS".to_string(),
            "HIS".to_string(),
            "HIS".to_string(),
        ],
        seq_ids: vec![1, 2, 3, 4],
        ..Default::default()
    };
    // CYS 0: reduced CB = 27.9 ppm; CYS 1: oxidized disulfide CB = 41.4 ppm
    ref_data.shifts.insert((0, "CB".to_string()), 27.9);
    ref_data.shifts.insert((1, "CB".to_string()), 41.4);
    // HIS 2: canonical Pelton Ne2H (ND1=250.1, NE2=168.8, CD2=118.0, HE1=7.80)
    ref_data.shifts.insert((2, "ND1".to_string()), 250.1);
    ref_data.shifts.insert((2, "NE2".to_string()), 168.8);
    ref_data.shifts.insert((2, "CD2".to_string()), 118.0);
    ref_data.shifts.insert((2, "HE1".to_string()), 7.80);
    // HIS 3: Pelton Nd1H (ND1=170.2, NE2=249.4, CD2=125.2, HE1=7.84)
    ref_data.shifts.insert((3, "ND1".to_string()), 170.2);
    ref_data.shifts.insert((3, "NE2".to_string()), 249.4);
    ref_data.shifts.insert((3, "CD2".to_string()), 125.2);
    ref_data.shifts.insert((3, "HE1".to_string()), 7.84);

    // User manually overrides CYS 1 to CysReduced before auto-sync
    state.set_variant(1, ResidueVariant::CysReduced);
    assert_eq!(
        state.user_variant_overrides.get(&1),
        Some(&ResidueVariant::CysReduced)
    );

    state.sync_inferred_variants_from_reference(&ref_data);
    assert_eq!(state.variant_for(0), ResidueVariant::CysReduced);
    // User override on CYS 1 MUST win over auto-inferred CysOxidized!
    assert_eq!(
        state.auto_inferred_variants.get(&1),
        Some(&ResidueVariant::CysOxidized)
    );
    assert_eq!(state.variant_for(1), ResidueVariant::CysReduced);
    assert_eq!(state.variant_for(2), ResidueVariant::HisNe2H);
    assert_eq!(state.variant_for(3), ResidueVariant::HisNd1H);

    // Now mutate shifts: HIS 2 enters fast tautomer exchange (ND1=208, NE2=211) and CYS 0 loses CB:
    // stale auto-inferred states MUST be cleared while user override on CYS 1 stays intact!
    ref_data.shifts.remove(&(0, "CB".to_string()));
    ref_data.shifts.insert((2, "ND1".to_string()), 208.0);
    ref_data.shifts.insert((2, "NE2".to_string()), 211.0);
    // HIS 3 shifts to Zn(II)-coordinated mimicry (ND1=200.88, NE2=178.27, HE1=7.72) -> must abstain!
    ref_data.shifts.insert((3, "ND1".to_string()), 200.88);
    ref_data.shifts.insert((3, "NE2".to_string()), 178.27);
    ref_data.shifts.insert((3, "CD2".to_string()), 117.70);
    ref_data.shifts.insert((3, "HE1".to_string()), 7.72);

    state.sync_inferred_variants_from_reference(&ref_data);
    assert_eq!(state.variant_for(0), ResidueVariant::Canonical);
    assert_eq!(state.variant_for(1), ResidueVariant::CysReduced);
    assert_eq!(state.variant_for(2), ResidueVariant::Canonical);
    assert_eq!(state.variant_for(3), ResidueVariant::Canonical);

    // Clearing user override on CYS 1 falls back to auto-inferred CysOxidized
    state.clear_user_variant_override(1);
    assert_eq!(state.variant_for(1), ResidueVariant::CysOxidized);
}
