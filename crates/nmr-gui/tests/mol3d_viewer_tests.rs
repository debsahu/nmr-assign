//! Integration, safety, picking-accuracy, and headless WebGPU (`wgpu`) performance gate tests
//! for the Phase 12 3D Biomolecular Structure Viewport & NMR Overlays (`T12.1`, `T12.2`, `P2`).
//!
//! Provenance: Clean-room test suite validating `docs/specs/15-structure-viewer.md` against:
//! - Carson, M. & Bugg, C. E. (*J. Mol. Graph.* 1986, 4, 121–122 — cubic peptide-plane ribbons).
//! - Kabsch, W. (*Acta Crystallogr. A* 1976, 32, 922–923 — proper `SO(3)` ensemble superposition).
//! - Kabsch, W. & Sander, C. (*Biopolymers* 1983, 22, 2577–2637 — H-bond secondary structure).
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* 1982, 104, 4546–4559 — `S^2` & `R_ex` tracks).
//! - Nilges, M. (*J. Mol. Biol.* 1995, 245, 645–660 — `r^-6` NOE/PRE restraint evaluation).

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use eframe::wgpu;
use nmr_core::nmrstar::{read_star_str, Reference};
use nmr_core::priors::Priors;
use nmr_core::topology::ResidueVariant;
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::mol3d_overlay::{
    ColormapKind, DistanceRestraint3d, QuantitativeTrack, QuantitativeTrackData,
    RestraintAtomSelector, RestraintFilterConfig, RestraintKind, RestraintStatus,
};
use nmr_gui::mol3d_viewer::{
    build_mol3d_scene, escape_xml, export_structure_svg, kabsch_superpose_points,
    parse_mmcif_ensemble, parse_pdb_ensemble, parse_structure_auto, ArcballCamera,
    Mol3dPipelineResource, Mol3dViewerPanel, ProjectionMode, RepresentationStyle, SceneBuildConfig,
    SecondaryStructureKind,
};
use nmr_gui::sidechain_widget::SequenceRibbonState;

const UBIQUITIN_1UBQ_PDB: &str = include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");
const SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

/// Build a synthetic multi-conformer all-atom `n_res`-residue helical/sheet/coil NMR ensemble in
/// standard PDB format (`17` atoms per residue -> `20 * 150 * 17 = 51,000` atoms for `20 x 150 aa`).
fn synthesize_nmr_ensemble_pdb(n_models: usize, n_res: usize) -> String {
    const RES_CYCLE: &[&str] = &["LEU", "ILE", "PHE", "GLU", "LYS", "ASP", "VAL", "HIS"];
    const ATOM_TEMPLATES: &[(&str, &str, [f64; 3])] = &[
        ("N", "N", [-1.458, 0.000, 0.000]),
        ("H", "H", [-1.960, 0.820, 0.280]),
        ("CA", "C", [0.000, 0.000, 0.000]),
        ("HA", "H", [0.360, -0.890, -0.480]),
        ("C", "C", [0.540, 1.420, 0.000]),
        ("O", "O", [0.020, 2.390, -0.540]),
        ("CB", "C", [0.530, -0.760, 1.220]),
        ("HB2", "H", [0.180, -1.780, 1.210]),
        ("HB3", "H", [1.610, -0.770, 1.190]),
        ("CG", "C", [0.080, -0.120, 2.540]),
        ("HG2", "H", [-0.990, -0.100, 2.580]),
        ("HG3", "H", [0.460, 0.890, 2.610]),
        ("CD1", "C", [0.620, -0.920, 3.720]),
        ("HD11", "H", [0.290, -0.470, 4.640]),
        ("HD12", "H", [1.700, -0.940, 3.690]),
        ("CD2", "C", [-1.440, -0.180, 2.660]),
        ("HD21", "H", [-1.760, 0.280, 3.580]),
    ];

    let mut pdb = String::with_capacity(n_models * n_res * ATOM_TEMPLATES.len() * 82 + 512);
    pdb.push_str("HEADER    SYNTHETIC 20-CONFORMER 150AA NMR ENSEMBLE\n");

    for m in 0..n_models {
        pdb.push_str(&format!("MODEL     {:>4}\n", m + 1));
        // Rigid-body offset/rotation per conformer to exercise Kabsch superposition too
        let theta = (m as f64) * 0.09;
        let (st, ct) = theta.sin_cos();
        let shift = [(m as f64) * 2.5, -(m as f64) * 1.7, (m as f64) * 1.1];
        let mut serial = 1u32;

        for i in 0..n_res {
            let res_name = RES_CYCLE[i % RES_CYCLE.len()];
            let seq_id = (i + 1) as i32;
            // Alternate helical turns and extended beta-like segments
            let (base_x, base_y, base_z) = if (i / 25) % 2 == 0 {
                let ang = (i as f64) * 1.745; // ~100 deg/res alpha helix
                (2.3 * ang.cos(), 2.3 * ang.sin(), (i as f64) * 1.50)
            } else {
                let zig = if i % 2 == 0 { 0.45 } else { -0.45 };
                (zig, 6.0 + zig * 0.5, (i as f64) * 3.25)
            };

            // Mild loop flexibility on conformers > 0
            let loop_amp = if i % 25 < 4 { 0.35 } else { 0.06 };
            let jitter = ((m * 31 + i * 17) as f64).sin() * loop_amp;

            for &(atom_name, elem, off) in ATOM_TEMPLATES {
                let lx = base_x + off[0] + jitter;
                let ly = base_y + off[1] - 0.5 * jitter;
                let lz = base_z + off[2] + 0.3 * jitter;
                let rx = ct * lx - st * ly + shift[0];
                let ry = st * lx + ct * ly + shift[1];
                let rz = lz + shift[2];

                let atom_field = if atom_name.len() == 4 {
                    atom_name.to_string()
                } else {
                    format!(" {atom_name:<3}")
                };
                pdb.push_str(&format!(
                    "ATOM  {:>5} {} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 20.00          {:>2}\n",
                    serial % 100_000,
                    atom_field,
                    res_name,
                    seq_id,
                    rx,
                    ry,
                    rz,
                    elem
                ));
                serial += 1;
            }
        }
        pdb.push_str("ENDMDL\n");
    }
    pdb.push_str("END\n");
    pdb
}

fn block_on_future<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn create_headless_wgpu_device() -> Option<(wgpu::Device, wgpu::Queue, String)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    let label = format!("{} ({:?})", info.name, info.backend);
    let (device, queue) = block_on_future(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("mol3d-headless-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits:
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue, label))
}

#[test]
fn test_pdb_and_mmcif_parsing_mse_altloc_and_hostile_utf8_safety() {
    // 1. Real 1UBQ X-ray PDB fixture
    let ubiq = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("1UBQ PDB should parse cleanly");
    assert_eq!(ubiq.models.len(), 1);
    assert_eq!(ubiq.primary_sequence.len(), 76);
    assert_eq!(ubiq.primary_sequence[0], "MET");
    assert_eq!(ubiq.primary_sequence[75], "GLY");
    assert!(ubiq.total_atom_count() >= 600);
    assert!(ubiq.bounding_radius > 10.0 && ubiq.bounding_radius < 40.0);

    // 2. Multi-model mmCIF with quoted-token CIF control-keyword traps ('_...', 'data_...', "loop_", ';...;'),
    //    MSE -> MET normalization, B-only altloc retention, A+B altloc deduplication, and ZN cofactor
    let mmcif = "\
data_test_mse_altloc_zn
_entry.id 'TEST_MMCIF'
loop_
_audit_author.name
_audit_author.pdbx_ordinal
_audit_author.note
'_quoted_underscore_not_a_tag' 1 'data_fake_block_inside_quotes'
\"loop_\" 2
; _multiline_quoted_note
data_fake_multiline
loop_
;
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM   1 N  N   . MSE A 1  0.000  0.000  0.000 1.00 15.0 1
ATOM   2 C  CA  . MSE A 1  1.458  0.000  0.000 1.00 15.0 1
ATOM   3 C  C   . MSE A 1  2.010  1.420  0.000 1.00 15.0 1
ATOM   4 O  O   . MSE A 1  1.300  2.420  0.000 1.00 15.0 1
ATOM   5 SE SE  . MSE A 1  2.100 -1.500  1.200 1.00 18.0 1
ATOM   6 N  N   B CYS A 2  3.320  1.520  0.000 0.70 16.0 1
ATOM   7 C  CA  B CYS A 2  3.980  2.820  0.000 0.70 16.0 1
ATOM   8 C  C   A CYS A 2  5.490  2.710  0.000 0.60 16.0 1
ATOM   9 C  C   B CYS A 2  5.550  2.790  0.120 0.40 16.0 1
ATOM  10 O  O   . CYS A 2  6.080  1.630  0.000 1.00 16.0 1
ATOM  11 S  SG  B CYS A 2  3.420  3.820  1.420 1.00 17.0 1
ATOM  12 N  N   . HIS A 3  6.120  3.880  0.000 1.00 14.0 1
ATOM  13 C  CA  . HIS A 3  7.570  3.980  0.000 1.00 14.0 1
ATOM  14 C  C   . HIS A 3  8.120  5.400  0.000 1.00 14.0 1
ATOM  15 O  O   . HIS A 3  7.380  6.380  0.000 1.00 14.0 1
ATOM  16 N  ND1 . HIS A 3  5.100  4.850  1.800 1.00 15.0 1
HETATM 17 ZN ZN . ZN  A 201 4.200 4.350  1.650 1.00 12.0 1
ATOM  18 N  N   . MSE A 1  10.000 10.000  5.000 1.00 15.0 2
ATOM  19 C  CA  . MSE A 1  11.458 10.000  5.000 1.00 15.0 2
ATOM  20 C  C   . MSE A 1  12.010 11.420  5.000 1.00 15.0 2
ATOM  21 N  N   B CYS A 2  13.320 11.520  5.000 0.70 16.0 2
ATOM  22 C  CA  B CYS A 2  13.980 12.820  5.000 0.70 16.0 2
ATOM  23 C  C   A CYS A 2  15.490 12.710  5.000 0.60 16.0 2
ATOM  24 N  N   . HIS A 3  16.120 13.880  5.000 1.00 14.0 2
ATOM  25 C  CA  . HIS A 3  17.570 13.980  5.000 1.00 14.0 2
ATOM  26 C  C   . HIS A 3  18.120 15.400  5.000 1.00 14.0 2
#
";
    let cif_ens = parse_mmcif_ensemble(mmcif).expect("mmCIF ensemble should parse");
    assert_eq!(cif_ens.models.len(), 2);
    // MSE normalized to MET!
    assert_eq!(cif_ens.primary_sequence, vec!["MET", "CYS", "HIS"]);
    let m0_res = &cif_ens.models[0].chains[0].residues;
    assert!(
        m0_res[0].atom_pos("SD").is_some(),
        "MSE SE must normalize to MET SD"
    );
    // B-only altloc retained on CYS N, CA, SG; A+B altloc on CYS C deduplicated to A (5.490)!
    assert!(
        m0_res[1].atom_pos("CA").is_some() && m0_res[1].atom_pos("SG").is_some(),
        "B-only altloc atoms must be retained"
    );
    let cys_c = m0_res[1].atom_pos("C").expect("CYS C present");
    assert!((cys_c[0] - 5.490).abs() < 1e-3);
    // Metal ZN retained as HETATM with is_metal == true
    assert_eq!(cif_ens.models[0].het_atoms.len(), 1);
    assert!(cif_ens.models[0].het_atoms[0].is_metal);
    // Model 2 was translated by (+10, +10, +5) with identical internal geometry -> Kabsch RMSD ~ 0!
    assert!(cif_ens.models[1].kabsch_rmsd_to_ref < 1e-4);

    // 3. Hostile multibyte UTF-8 in fixed-width PDB columns, short lines, and headers must NEVER panic!
    let hostile_utf8_pdb = "\
HEADER    🔥🧪αβγ🦀 HOSTILE MULTIBYTE UTF-8 HEADER
REMARK   1 日本語コメント — Ω≈ç√∫˜µ≤≥÷
ATOM  🔥🧪🦀αβγδεζηθικλμνξοπρστυφχψω
ATOM      1  N   ALA A   1      🔥.000   1.000   2.000  1.00 10.00           N
HETATM
TER
END
";
    let res_utf8 = parse_structure_auto(hostile_utf8_pdb);
    assert!(
        res_utf8.is_err(),
        "Hostile multibyte UTF-8 in numeric columns must return clean Err without panicking"
    );

    // 4. NaN / Inf / -Inf coordinate rejection in both PDB and mmCIF
    for bad_coord in ["     NaN", "     inf", "    -inf"] {
        let bad_pdb = format!(
            "ATOM      1  CA  ALA A   1    {bad_coord}   1.000   2.000  1.00 10.00           C\nEND\n"
        );
        assert!(
            parse_pdb_ensemble(&bad_pdb).is_err(),
            "PDB coordinate '{bad_coord}' must be rejected"
        );
    }
    for bad_cif_val in ["NaN", "inf", "-inf", "Infinity"] {
        let bad_cif = format!(
            "data_bad\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\nATOM CA ALA A 1 {bad_cif_val} 1.0 2.0\n#\n"
        );
        assert!(
            parse_mmcif_ensemble(&bad_cif).is_err(),
            "mmCIF coordinate '{bad_cif_val}' must be rejected"
        );
    }
}

#[test]
fn test_kabsch_sander_ss_without_headers_and_kabsch_superposition() {
    // Strip every HELIX and SHEET record from 1UBQ so secondary structure relies 100% on
    // Kabsch–Sander backbone H-bonds + Ramachandran (phi, psi) basins!
    let stripped_pdb: String = UBIQUITIN_1UBQ_PDB
        .lines()
        .filter(|ln| !(ln.starts_with("HELIX") || ln.starts_with("SHEET")))
        .map(|ln| format!("{ln}\n"))
        .collect();

    let ens = parse_pdb_ensemble(&stripped_pdb).expect("Headerless 1UBQ should parse");
    let residues = &ens.models[0].chains[0].residues;
    let helix_count = residues
        .iter()
        .filter(|r| r.ss == SecondaryStructureKind::Helix)
        .count();
    let sheet_count = residues
        .iter()
        .filter(|r| r.ss == SecondaryStructureKind::Sheet)
        .count();
    let coil_count = residues
        .iter()
        .filter(|r| r.ss == SecondaryStructureKind::Coil)
        .count();

    assert!(
        helix_count >= 10,
        "Expected Ubiquitin alpha-helix (>= 10 residues) without HELIX records, got {helix_count}"
    );
    assert!(
        sheet_count >= 18,
        "Expected Ubiquitin beta-sheet (>= 18 residues) without SHEET records, got {sheet_count}"
    );
    assert!(coil_count >= 10);

    // Build cartoon ribbon mesh on headerless 1UBQ and verify non-empty smooth ribbon triangles
    let scene = build_mol3d_scene(
        &ens,
        &SceneBuildConfig::default(),
        &QuantitativeTrackData::default(),
        None,
        1,
    );
    assert!(scene.opaque_ribbon_verts.len() > 800);
    assert!(scene.opaque_ribbon_indices.len() > 2_000);

    // Kabsch quaternion proper-rotation (`det(R) == +1`) verification including reflection immunity
    let ref_pts = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.2, 0.1],
        [6.1, 2.9, -1.2],
        [8.4, 1.1, 2.7],
        [11.0, -1.5, 1.9],
    ];
    let angle = 1.13_f64;
    let (sa, ca) = angle.sin_cos();
    let mov_pts: Vec<[f64; 3]> = ref_pts
        .iter()
        .map(|p| {
            [
                ca * p[0] - sa * p[1] + 42.0,
                sa * p[0] + ca * p[1] - 19.5,
                p[2] + 8.25,
            ]
        })
        .collect();
    let (rot, _, _, rmsd) = kabsch_superpose_points(&ref_pts, &mov_pts);
    assert!(rmsd < 1e-9, "Exact rigid transform RMSD = {rmsd}");

    let det = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
        - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
        + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]);
    assert!(
        (det - 1.0).abs() < 1e-9,
        "Kabsch rotation must belong to SO(3) with det(R)=+1, got {det}"
    );

    // Mirror-reflected input must STILL yield det(R) = +1 (never an improper reflection!)
    let mirrored: Vec<[f64; 3]> = ref_pts.iter().map(|p| [-p[0], p[1], p[2]]).collect();
    let (rot_m, _, _, _) = kabsch_superpose_points(&ref_pts, &mirrored);
    let det_m = rot_m[0][0] * (rot_m[1][1] * rot_m[2][2] - rot_m[1][2] * rot_m[2][1])
        - rot_m[0][1] * (rot_m[1][0] * rot_m[2][2] - rot_m[1][2] * rot_m[2][0])
        + rot_m[0][2] * (rot_m[1][0] * rot_m[1][2] - rot_m[1][1] * rot_m[2][0]);
    // Recompute exact determinant for rot_m:
    let det_m_exact = rot_m[0][0] * (rot_m[1][1] * rot_m[2][2] - rot_m[1][2] * rot_m[2][1])
        - rot_m[0][1] * (rot_m[1][0] * rot_m[2][2] - rot_m[1][2] * rot_m[2][0])
        + rot_m[0][2] * (rot_m[1][0] * rot_m[2][1] - rot_m[1][1] * rot_m[2][0]);
    let _ = det_m;
    assert!(
        (det_m_exact - 1.0).abs() < 1e-9,
        "Mirrored input must still produce proper rotation det(R)=+1, got {det_m_exact}"
    );
}

#[test]
fn test_analytical_3d_ray_picking_100_percent_perspective_and_orthographic_and_sync() {
    // Construct a planar non-self-occluding 16-residue polypeptide + ZN metal cofactor in XY plane (`z = 0`)
    // so every single residue/atom/metal has a clear line-of-sight from the camera along +Z!
    let mut pdb = String::new();
    let aa_names = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "ILE",
        "THR", "LEU", "GLU",
    ];
    let mut serial = 1;
    for (k, &aa) in aa_names.iter().enumerate() {
        let seq_id = (k + 1) as i32;
        let x0 = (k as f64) * 3.6;
        let y0 = if k % 2 == 0 { 0.8 } else { -0.8 };
        for &(aname, elem, dx, dy) in &[
            ("N", "N", -1.20, -0.35),
            ("CA", "C", 0.00, 0.00),
            ("C", "C", 1.25, -0.30),
            ("O", "O", 1.35, -1.50),
            ("CB", "C", 0.00, 1.52),
        ] {
            if aa == "GLY" && aname == "CB" {
                continue;
            }
            pdb.push_str(&format!(
                "ATOM  {:>5}  {:<3} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 15.00          {:>2}\n",
                serial,
                aname,
                aa,
                seq_id,
                x0 + dx,
                y0 + dy,
                0.0,
                elem
            ));
            serial += 1;
        }
    }
    pdb.push_str(&format!(
        "HETATM{:>5} ZN    ZN A 201    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          ZN\nEND\n",
        serial, 25.0, 7.5, 0.0
    ));

    let seq: Vec<String> = aa_names.iter().map(|s| (*s).to_string()).collect();
    let seq_ids: Vec<i32> = (1..=16).collect();
    let mut ref_shifts = Reference::new(seq.clone(), seq_ids.clone());
    for (k, &aa_name) in aa_names.iter().enumerate() {
        ref_shifts
            .shifts
            .insert((k, "H".to_string()), 8.10 + (k as f64) * 0.05);
        ref_shifts
            .shifts
            .insert((k, "N".to_string()), 118.0 + (k as f64) * 0.6);
        ref_shifts
            .shifts
            .insert((k, "CA".to_string()), 54.0 + (k as f64) * 0.4);
        if aa_name != "GLY" {
            ref_shifts
                .shifts
                .insert((k, "CB".to_string()), 32.0 + (k as f64) * 0.7);
        }
    }

    let vp_w = 1280.0;
    let vp_h = 800.0;

    for proj_mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
        // A. Test BallAndStick atom + metal sphere picking across all residues
        let mut panel = Mol3dViewerPanel::new();
        panel.scene_cfg.style = RepresentationStyle::BallAndStick;
        panel
            .load_structure_text(&pdb, &seq, &seq_ids)
            .expect("planar structure loads");
        panel.camera.projection = proj_mode;
        panel.rebuild_for_test();

        let mut total_targets = 0usize;
        let mut exact_hits = 0usize;

        let spheres = panel.scene.pick_spheres.clone();
        assert!(spheres.len() >= 75);
        for sph in &spheres {
            total_targets += 1;
            let world = sph.center;
            let (px, py, _) = panel
                .camera
                .project_world_to_screen(world, vp_w, vp_h)
                .expect("sphere projects on screen");

            if sph.residue_idx.is_some() {
                let sync = panel
                    .pick_at_viewport_pixel(px, py, vp_w, vp_h, &seq, &seq_ids, Some(&ref_shifts))
                    .expect("polymer sphere pick must resolve Mol3dSyncAction");
                if Some(sync.residue_idx) == sph.residue_idx && sync.clicked_atom == sph.atom_name {
                    exact_hits += 1;
                }
                // Verify T12.2 crosshair & covalent graph coupling on every hit!
                assert!(sync.open_sidechain_graph);
                let mut ch = SyncCrosshair::default();
                sync.apply_to_crosshair(&mut ch);
                let exp_h = 8.10 + (sync.residue_idx as f64) * 0.05;
                let exp_n = 118.0 + (sync.residue_idx as f64) * 0.6;
                assert!((ch.h_ppm - exp_h).abs() < 1e-6);
                assert!((ch.n_ppm - exp_n).abs() < 1e-6);
            } else {
                // Metal cofactor HETATM sphere: verify raw hit matches ZN!
                let _ = panel.pick_at_viewport_pixel(
                    px,
                    py,
                    vp_w,
                    vp_h,
                    &seq,
                    &seq_ids,
                    Some(&ref_shifts),
                );
                let hit = panel.last_picked_hit.as_ref().expect("ZN sphere hit");
                if hit.atom_name == "ZN" && hit.seq_id == 201 {
                    exact_hits += 1;
                }
            }
        }
        assert_eq!(
            exact_hits, total_targets,
            "Expected 100.0% analytical sphere/metal picking accuracy in {proj_mode:?}, got {exact_hits}/{total_targets}"
        );

        // B. Test Cartoon ribbon-segment capsule picking across all residues
        panel.scene_cfg.style = RepresentationStyle::Cartoon;
        panel.rebuild_for_test();
        let mut ribbon_targets = 0usize;
        let mut ribbon_hits = 0usize;
        for k in 0..16 {
            let ca = panel.ensemble.as_ref().unwrap().models[0].chains[0].residues[k]
                .atom_pos("CA")
                .unwrap();
            let (px, py, _) = panel
                .camera
                .project_world_to_screen([ca[0] as f64, ca[1] as f64, ca[2] as f64], vp_w, vp_h)
                .unwrap();
            ribbon_targets += 1;
            let sync = panel
                .pick_at_viewport_pixel(px, py, vp_w, vp_h, &seq, &seq_ids, Some(&ref_shifts))
                .expect("cartoon ribbon pick resolves");
            if sync.residue_idx == k {
                ribbon_hits += 1;
            }
        }
        assert_eq!(
            ribbon_hits, ribbon_targets,
            "Expected 100.0% cartoon ribbon picking accuracy in {proj_mode:?}, got {ribbon_hits}/{ribbon_targets}"
        );
    }

    // C. Reverse coupling: selecting residue in sequence ribbon / 3D strip centers 3D camera on CA_k
    let mut panel = Mol3dViewerPanel::new();
    panel
        .load_structure_text(UBIQUITIN_1UBQ_PDB, &[], &[])
        .expect("1UBQ loads");
    panel.highlight_sequence_residue(Some(28), true);
    let ca28 = panel.ensemble.as_ref().unwrap().models[0].chains[0].residues[28]
        .atom_pos("CA")
        .unwrap();
    assert!((panel.camera.target[0] - ca28[0] as f64).abs() < 1e-4);
    assert!((panel.camera.target[1] - ca28[1] as f64).abs() < 1e-4);
    assert!((panel.camera.target[2] - ca28[2] as f64).abs() < 1e-4);
}

#[test]
fn test_quantitative_putty_and_colormaps_under_2ms() {
    // Verify ColormapKind endpoints & finite bounds
    for cmap in ColormapKind::ALL {
        let c0 = cmap.sample_rgba(0.0, 1.0);
        let cm = cmap.sample_rgba(0.5, 1.0);
        let c1 = cmap.sample_rgba(1.0, 1.0);
        assert!(c0.iter().chain(&cm).chain(&c1).all(|v| v.is_finite()));
        assert!(
            (c0[0] - c1[0]).abs() + (c0[1] - c1[1]).abs() + (c0[2] - c1[2]).abs() > 0.25,
            "{cmap:?} endpoints must be perceptually distinct"
        );
    }
    // Diverging BlueWhiteRed midpoint is crisp neutral white
    let bwr_mid = ColormapKind::BlueWhiteRed.sample_rgba(0.5, 1.0);
    assert!(bwr_mid[0] > 0.94 && bwr_mid[1] > 0.94 && bwr_mid[2] > 0.94);

    // Benchmark `< 2.0 ms` putty + colormap updates on a 20-conformer 150 aa ensemble!
    let pdb_20x150 = synthesize_nmr_ensemble_pdb(20, 150);
    let ens = parse_pdb_ensemble(&pdb_20x150).expect("20x150 ensemble parses");
    assert!(ens.total_atom_count() >= 48_000);

    let mut track_data = QuantitativeTrackData {
        putty_enabled: true,
        ..QuantitativeTrackData::default()
    };
    let ubiq_ref = read_star_str(SYNTHETIC_UBIQUITIN_STR, None).expect("ubiquitin star");
    track_data.populate_csi_from_reference(&ubiq_ref, &Priors::provisional());
    for k in 0..150 {
        let f = (k as f64) / 149.0;
        track_data.confidence.insert(k, 0.35 + 0.65 * f);
        track_data.csi.entry(k).or_insert((f * 2.0) - 1.0);
        track_data
            .lipari_szabo_s2
            .insert(k, 0.45 + 0.48 * (1.0 - f));
        track_data
            .rex_s_inv
            .insert(k, 18.0 * (f * std::f64::consts::TAU).sin().abs());
        track_data
            .csp_ppm
            .insert(k, 0.42 * (f * std::f64::consts::PI).sin().abs());
    }

    let mut scene = build_mol3d_scene(&ens, &SceneBuildConfig::default(), &track_data, None, 1);
    let baseline_pos = scene.opaque_ribbon_verts[10].pos;

    let tracks_and_maps = [
        (
            QuantitativeTrack::AssignmentConfidence,
            ColormapKind::Viridis,
        ),
        (
            QuantitativeTrack::ChemicalShiftIndex,
            ColormapKind::BlueWhiteRed,
        ),
        (QuantitativeTrack::LipariSzaboS2, ColormapKind::Cividis),
        (
            QuantitativeTrack::ChemicalExchangeRex,
            ColormapKind::Viridis,
        ),
        (QuantitativeTrack::LigandCsp, ColormapKind::Cividis),
    ];

    for (trk, cmap) in tracks_and_maps {
        track_data.active_track = trk;
        track_data.colormap = cmap;
        let t0 = Instant::now();
        scene.update_quantitative_putty(&track_data);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("[mol3d putty bench] track={trk:?} colormap={cmap:?} update={elapsed_ms:.3} ms");
        assert!(
            elapsed_ms < 2.0,
            "Putty/colormap update for {trk:?}/{cmap:?} took {elapsed_ms:.3} ms (gate: < 2.0 ms)"
        );
    }

    let updated_pos = scene.opaque_ribbon_verts[10].pos;
    assert_ne!(
        baseline_pos, updated_pos,
        "Putty radius update must displace ribbon cross-section vertices"
    );
}

#[test]
fn test_1500_noe_pre_restraint_cylinders_under_2ms() {
    let ens = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("1UBQ parses");
    let seq_ids = &ens.primary_seq_ids;
    assert_eq!(seq_ids.len(), 76);

    // Construct 1,500 NOE & PRE restraints spanning green Satisfied, amber Minor (< 0.5 A), and red Major (>= 0.5 A)
    let mut restraints = Vec::with_capacity(1_500);
    for idx in 0..1_500 {
        let i = idx % 70;
        let step = 1 + (idx % 6);
        let j = (i + step).min(75);
        let seq_a = seq_ids[i];
        let seq_b = seq_ids[j];
        let ca_a = ens.models[0].chains[0].residues[i].atom_pos("CA").unwrap();
        let ca_b = ens.models[0].chains[0].residues[j].atom_pos("CA").unwrap();
        let true_d = (((ca_a[0] - ca_b[0]).powi(2)
            + (ca_a[1] - ca_b[1]).powi(2)
            + (ca_a[2] - ca_b[2]).powi(2)) as f64)
            .sqrt();

        let upper = match idx % 3 {
            0 => true_d + 0.60,            // Satisfied (green)
            1 => (true_d - 0.30).max(1.8), // Minor violation ~0.30 A < 0.50 A (amber)
            _ => (true_d - 1.25).max(1.8), // Major violation >= 0.50 A (crimson)
        };
        let kind = if idx % 5 == 0 {
            RestraintKind::Pre
        } else {
            RestraintKind::Noe
        };
        restraints.push(DistanceRestraint3d::pair(
            idx + 1,
            kind,
            RestraintAtomSelector::new(Some('A'), seq_a, "CA"),
            RestraintAtomSelector::new(Some('A'), seq_b, "CA"),
            1.8,
            upper,
        ));
    }

    let mut scene = build_mol3d_scene(
        &ens,
        &SceneBuildConfig::default(),
        &QuantitativeTrackData::default(),
        None,
        1,
    );

    // Warm-up + timed evaluation & scene cylinder toggle across 1,500 restraints
    let filter_all = RestraintFilterConfig::default();
    let _ = ens
        .coord_lookup
        .evaluate_restraints(&restraints, &filter_all);

    let t0 = Instant::now();
    let overlay_all = ens
        .coord_lookup
        .evaluate_restraints(&restraints, &filter_all);
    scene.set_restraint_overlay(&overlay_all);
    let eval_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[mol3d restraint bench] n=1500 eval+upload={eval_ms:.3} ms (sat={} minor={} major={})",
        overlay_all.satisfied_count,
        overlay_all.minor_violation_count,
        overlay_all.major_violation_count
    );

    assert_eq!(overlay_all.cylinders.len(), 1_500);
    assert!(overlay_all.satisfied_count >= 450);
    assert!(overlay_all.minor_violation_count >= 450);
    assert!(overlay_all.major_violation_count >= 450);
    assert!(
        eval_ms < 2.0,
        "Evaluating & updating 1,500 NOE/PRE restraint cylinders took {eval_ms:.3} ms (gate: < 2.0 ms)"
    );

    // Verify exact green / amber / red cylinder colors
    for (ev, cyl) in overlay_all
        .evaluated
        .iter()
        .zip(overlay_all.cylinders.iter())
    {
        assert_eq!(cyl.color0, ev.status.rgba());
        match ev.status {
            RestraintStatus::Satisfied => assert!(ev.violation_angstrom <= 1e-6),
            RestraintStatus::MinorViolation => {
                assert!(ev.violation_angstrom > 1e-6 && ev.violation_angstrom < 0.5)
            }
            RestraintStatus::MajorViolation => assert!(ev.violation_angstrom >= 0.5),
        }
    }

    // Toggle violations-only filter in `< 2.0 ms`
    let filter_viol_only = RestraintFilterConfig {
        show_satisfied: false,
        dashed_cylinders: true,
        ..RestraintFilterConfig::default()
    };
    let t1 = Instant::now();
    let overlay_viol = ens
        .coord_lookup
        .evaluate_restraints(&restraints, &filter_viol_only);
    scene.set_restraint_overlay(&overlay_viol);
    let toggle_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        scene.restraint_cylinders.len(),
        overlay_all.minor_violation_count + overlay_all.major_violation_count
    );
    assert!(
        toggle_ms < 2.0,
        "Toggling 1,500 restraint cylinders took {toggle_ms:.3} ms (gate: < 2.0 ms)"
    );

    // Also verify 1,500 restraints across a full 20-conformer 150-residue ensemble stay < 2.0 ms!
    let pdb_20x150 = synthesize_nmr_ensemble_pdb(20, 150);
    let ens_20 = parse_pdb_ensemble(&pdb_20x150).expect("20x150 parses");
    assert_eq!(ens_20.coord_lookup.model_count(), 20);
    let _ = ens_20
        .coord_lookup
        .evaluate_restraints(&restraints, &filter_all);
    let t_ens = Instant::now();
    let ov_20 = ens_20
        .coord_lookup
        .evaluate_restraints(&restraints, &filter_all);
    let ens_ms = t_ens.elapsed().as_secs_f64() * 1000.0;
    println!("[mol3d 20-model restraint bench] models=20 n=1500 eval={ens_ms:.3} ms");
    assert_eq!(ov_20.evaluated.len(), 1_500);
    assert!(
        ens_ms < 2.0,
        "20-conformer 1,500-restraint <r^-6> evaluation took {ens_ms:.3} ms (gate: < 2.0 ms)"
    );
}

#[test]
fn test_multi_conformer_r_inv6_and_pseudoatom_hd1_hg1_qr_expansion() {
    // 1. Build a 4-conformer PDB with identical CA backbone (so Kabsch superposition is exact identity),
    //    where conformer 1 has LEU 1 HD1* far from VAL 2 HG1* (~5.8 A), while conformers 2..4 pack
    //    LEU 1 HD1* against VAL 2 HG1* (~3.2 A) and PHE 3 QR (~3.4 A), while CA-CA distances are >= 9.2 A!
    let mut pdb = String::new();
    for model_num in 1..=4 {
        pdb.push_str(&format!("MODEL     {model_num:>4}\n"));
        // LEU 1 (CA at x = 0.0, y = 0.0, z = 0.0; methyl HD1* flips x = 0.2 in model 1 vs x = 3.2 in models 2..4)
        let hd1_x = if model_num == 1 { 0.20 } else { 3.20 };
        let leu_atoms: &[(&str, &str, f64, f64, f64)] = &[
            ("N", "N", -1.20, 0.40, 0.00),
            ("CA", "C", 0.00, 0.00, 0.00),
            ("C", "C", 1.25, 0.60, 0.00),
            ("O", "O", 1.35, 1.82, 0.00),
            ("CB", "C", 0.30, -1.50, 0.00),
            ("CG", "C", 1.60, -2.10, 0.00),
            ("CD1", "C", hd1_x - 0.35, -2.50, 0.00),
            ("HD11", "H", hd1_x, -2.30, 0.25),
            ("HD12", "H", hd1_x, -2.70, 0.00),
            ("HD13", "H", hd1_x, -2.50, -0.25),
        ];
        let mut serial = 1;
        for &(aname, elem, x, y, z) in leu_atoms {
            pdb.push_str(&format!(
                "ATOM  {:>5} {:<4} LEU A   1    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          {:>2}\n",
                serial, aname, x, y, z, elem
            ));
            serial += 1;
        }

        // VAL 2 (CA at x = 9.20 -> CA(1)-CA(2) = 9.20 A! Methyl HG1* at x = 6.10, y = -2.50)
        // In model 1: |6.10 - 0.20| = 5.90 A; in models 2..4: |6.10 - 3.20| = 2.90 A!
        let val_atoms: &[(&str, &str, f64, f64, f64)] = &[
            ("N", "N", 8.00, 0.40, 0.00),
            ("CA", "C", 9.20, 0.00, 0.00),
            ("C", "C", 10.45, 0.60, 0.00),
            ("O", "O", 10.55, 1.82, 0.00),
            ("CB", "C", 8.60, -1.50, 0.00),
            ("CG1", "C", 7.10, -2.30, 0.00),
            ("HG11", "H", 6.10, -2.30, 0.25),
            ("HG12", "H", 6.10, -2.70, 0.00),
            ("HG13", "H", 6.10, -2.50, -0.25),
        ];
        for &(aname, elem, x, y, z) in val_atoms {
            pdb.push_str(&format!(
                "ATOM  {:>5} {:<4} VAL A   2    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          {:>2}\n",
                serial, aname, x, y, z, elem
            ));
            serial += 1;
        }

        // PHE 3 (CA at x = 3.20, y = 7.40 -> CA(1)-CA(3) = 8.06 A; aromatic ring protons near y = 0.70..1.20,
        // so ~3.3..3.8 A from LEU 1 HD1* in models 2..4!)
        let phe_atoms: &[(&str, &str, f64, f64, f64)] = &[
            ("N", "N", 2.10, 7.80, 0.00),
            ("CA", "C", 3.20, 7.40, 0.00),
            ("C", "C", 4.45, 7.95, 0.00),
            ("O", "O", 4.60, 9.15, 0.00),
            ("CB", "C", 3.20, 5.88, 0.00),
            ("CG", "C", 3.20, 4.38, 0.00),
            ("CD1", "C", 2.05, 3.62, 0.00),
            ("CD2", "C", 4.35, 3.62, 0.00),
            ("CE1", "C", 2.05, 2.24, 0.00),
            ("CE2", "C", 4.35, 2.24, 0.00),
            ("CZ", "C", 3.20, 1.54, 0.00),
            ("HD1", "H", 1.12, 4.15, 0.00),
            ("HD2", "H", 5.28, 4.15, 0.00),
            ("HE1", "H", 1.12, 1.70, 0.00),
            ("HE2", "H", 5.28, 1.70, 0.00),
            ("HZ", "H", 3.20, 0.46, 0.00),
        ];
        for &(aname, elem, x, y, z) in phe_atoms {
            pdb.push_str(&format!(
                "ATOM  {:>5} {:<4} PHE A   3    {:>8.3}{:>8.3}{:>8.3}  1.00 12.00          {:>2}\n",
                serial, aname, x, y, z, elem
            ));
            serial += 1;
        }
        pdb.push_str("ENDMDL\n");
    }
    pdb.push_str("END\n");

    let ens = parse_pdb_ensemble(&pdb).expect("4-model methyl/aromatic ensemble parses");
    assert_eq!(ens.models.len(), 4);
    assert_eq!(ens.coord_lookup.model_count(), 4);

    // Unambiguous single-atom check across 4 conformers: LEU 1 CD1 to VAL 2 CG1
    // Model 1: |(0.20 - 0.35) - 7.10| = 7.25 A, dy = 0.20 -> d1 = sqrt(7.25^2 + 0.04) = 7.252758 A
    // Models 2..4: |(3.20 - 0.35) - 7.10| = 4.25 A, dy = 0.20 -> d2 = sqrt(4.25^2 + 0.04) = 4.254703 A
    // Conformer-0-only would give 7.253 A; true 4-conformer <r^-6>^(-1/6) gives ~4.456 A!
    let d1_sq: f64 = 7.25_f64.powi(2) + 0.20_f64.powi(2);
    let d2_sq: f64 = 4.25_f64.powi(2) + 0.20_f64.powi(2);
    let expected_cd1_cg1 = (0.25 * d1_sq.powi(-3) + 0.75 * d2_sq.powi(-3)).powf(-1.0 / 6.0);

    let restraints = vec![
        DistanceRestraint3d::pair(
            1,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "CD1"),
            RestraintAtomSelector::new(Some('A'), 2, "CG1"),
            1.8,
            4.80,
        ),
        // Methyl wildcard `HD1*` to `HG1*`: satisfied (~2.1-2.9 A), whereas CA-CA fallback (9.20 A) would be MajorViolation!
        DistanceRestraint3d::pair(
            2,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "HD1*"),
            RestraintAtomSelector::new(Some('A'), 2, "HG1*"),
            1.5,
            3.50,
        ),
        // Methyl pseudoatom `MD1` to aromatic ring pseudoatom `QR`: satisfied (< 4.0 A), whereas CA-CA (8.06 A) would be MajorViolation!
        DistanceRestraint3d::pair(
            3,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "MD1"),
            RestraintAtomSelector::new(Some('A'), 3, "QR"),
            1.5,
            4.00,
        ),
    ];

    let ov = ens
        .coord_lookup
        .evaluate_restraints(&restraints, &RestraintFilterConfig::default());
    assert_eq!(ov.evaluated.len(), 3);
    assert!(
        (ov.evaluated[0].effective_distance - expected_cd1_cg1).abs() < 1e-3,
        "Expected 4-conformer <r^-6>^(-1/6) = {expected_cd1_cg1:.4} A, got {:.4} A",
        ov.evaluated[0].effective_distance
    );
    assert_eq!(ov.evaluated[0].status, RestraintStatus::Satisfied);
    assert_eq!(
        ov.evaluated[1].status,
        RestraintStatus::Satisfied,
        "HD1* - HG1* methyl NOE must expand methyl protons (not fall back to CA=9.2A): got d_eff={:.3}",
        ov.evaluated[1].effective_distance
    );
    assert_eq!(
        ov.evaluated[2].status,
        RestraintStatus::Satisfied,
        "MD1 - QR methyl-aromatic NOE must expand ring protons (not fall back to CA=8.06A): got d_eff={:.3}",
        ov.evaluated[2].effective_distance
    );

    // 2. Heavy-atom-only X-ray structure (1UBQ lacks protons): verify `HD1*` / `HG1*` / `QR` fall back
    //    to sidechain heavy parents (`CD1`, `CG1`, aromatic carbons `CD1..CZ`) rather than `CA`!
    let ubiq = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("1UBQ parses");
    let ca_ile3 = ubiq.models[0].chains[0].residues[2].atom_pos("CA").unwrap();
    let cg1_ile3 = ubiq.models[0].chains[0].residues[2]
        .atom_pos("CG1")
        .unwrap();
    let ca_phe4 = ubiq.models[0].chains[0].residues[3].atom_pos("CA").unwrap();
    let r_heavy = vec![DistanceRestraint3d::pair(
        10,
        RestraintKind::Noe,
        RestraintAtomSelector::new(Some('A'), 3, "HG1*"),
        RestraintAtomSelector::new(Some('A'), 4, "QR"),
        1.5,
        8.0,
    )];
    let ov_heavy = ubiq
        .coord_lookup
        .evaluate_restraints(&r_heavy, &RestraintFilterConfig::default());
    assert_eq!(ov_heavy.evaluated.len(), 1);
    let ev_h = &ov_heavy.evaluated[0];
    // Endpoint p0 must equal ILE 3 CG1 (NOT ILE 3 CA!) and p1 must NOT equal PHE 4 CA!
    assert!((ev_h.p0[0] - cg1_ile3[0]).abs() < 1e-4 && (ev_h.p0[1] - cg1_ile3[1]).abs() < 1e-4);
    assert!(
        (ev_h.p0[0] - ca_ile3[0]).abs() + (ev_h.p0[1] - ca_ile3[1]).abs() > 0.5,
        "HG1* without protons must fall back to CG1, not CA"
    );
    assert!(
        (ev_h.p1[0] - ca_phe4[0]).abs() + (ev_h.p1[1] - ca_phe4[1]).abs() > 0.5,
        "QR without protons must fall back to aromatic ring carbons, not CA"
    );
}

#[test]
fn test_deterministic_xml_escaped_svg_and_offscreen_wgpu_png_export() {
    assert_eq!(
        escape_xml(r#"<script>alert("xss"&'bad')</script>"#),
        "&lt;script&gt;alert(&quot;xss&quot;&amp;&apos;bad&apos;)&lt;/script&gt;"
    );

    let mut ens = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("1UBQ parses");
    ens.title = r#"Ubiquitin <script>alert("xss"&'1ubq')</script>"#.to_string();

    let mut cam = ArcballCamera::default();
    cam.fit_ensemble(&ens);
    let cfg = SceneBuildConfig::default();
    let tracks = QuantitativeTrackData::default();

    let mut r_hostile = DistanceRestraint3d::pair(
        1,
        RestraintKind::Noe,
        RestraintAtomSelector::new(Some('A'), 2, "CA"),
        RestraintAtomSelector::new(Some('A'), 16, "CA"),
        1.8,
        4.0,
    );
    r_hostile.label = r#"NOE <img src=x onerror="evil()"> & 'tag'"#.to_string();
    let ov = ens
        .coord_lookup
        .evaluate_restraints(&[r_hostile], &RestraintFilterConfig::default());

    let svg1 = export_structure_svg(&ens, &cam, &cfg, &tracks, Some(&ov), 960, 720);
    let svg2 = export_structure_svg(&ens, &cam, &cfg, &tracks, Some(&ov), 960, 720);
    assert_eq!(svg1, svg2, "SVG export must be 100% deterministic");
    assert!(svg1.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(svg1.contains("&lt;script&gt;"));
    assert!(svg1.contains("&lt;img src=x onerror=&quot;evil()&quot;&gt;"));
    assert!(!svg1.contains("<script>"));
    assert!(!svg1.contains("<img "));

    // Offscreen high-DPI PNG export via headless wgpu
    if let Some((device, queue, _label)) = create_headless_wgpu_device() {
        let mut gpu_res = Mol3dPipelineResource::new(&device, None);
        let scene = build_mol3d_scene(&ens, &cfg, &tracks, Some(&ov), 1);
        let uniforms = cam.to_gpu_uniforms(640.0, 480.0, [0.055, 0.068, 0.095, 1.0]);
        let rgba = gpu_res
            .render_offscreen_rgba(&device, &queue, 640, 480, &uniforms, &scene)
            .expect("offscreen RGBA render succeeds");
        assert_eq!(rgba.len(), 640 * 480 * 4);
        // Verify protein geometry rendered non-background pixels
        let non_bg_pixels = rgba
            .chunks_exact(4)
            .filter(|px| px[0] > 35 || px[1] > 45 || px[2] > 60)
            .count();
        assert!(
            non_bg_pixels > 2_000,
            "Expected > 2,000 shaded structure pixels in offscreen capture, got {non_bg_pixels}"
        );

        let png_bytes = gpu_res
            .export_offscreen_png(&device, &queue, 640, 480, &uniforms, &scene)
            .expect("PNG encode succeeds");
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert!(png_bytes.len() > 4_096);
    }
}

#[test]
fn test_headless_wgpu_20_conformer_150aa_48k_atoms_sustains_60fps() {
    let pdb_20x150 = synthesize_nmr_ensemble_pdb(20, 150);
    let ens = parse_pdb_ensemble(&pdb_20x150).expect("20-conformer 150aa ensemble parses");
    let total_atoms = ens.total_atom_count();
    assert_eq!(ens.models.len(), 20);
    assert_eq!(ens.primary_sequence.len(), 150);
    assert!(
        total_atoms >= 48_000,
        "Ensemble must contain >= 48,000 atoms, got {total_atoms}"
    );

    let Some((device, queue, adapter_label)) = create_headless_wgpu_device() else {
        eprintln!("Skipping headless wgpu 60fps gate: no WebGPU adapter available");
        return;
    };

    let mut gpu_res = Mol3dPipelineResource::new(&device, Some(wgpu::TextureFormat::Bgra8Unorm));
    let mut cam = ArcballCamera::default();
    cam.fit_ensemble(&ens);
    let tracks = QuantitativeTrackData::default();

    // Benchmark BOTH default CartoonAndSidechains ensemble bundle AND full 51,000-atom BallAndStick!
    for style in [
        RepresentationStyle::CartoonAndSidechains,
        RepresentationStyle::BallAndStick,
    ] {
        let cfg = SceneBuildConfig {
            style,
            show_all_ensemble_models: true,
            show_hydrogens: true,
            ..SceneBuildConfig::default()
        };
        let scene = build_mol3d_scene(&ens, &cfg, &tracks, None, 1);

        // 5 warm-up frames (pipeline + target + VBO upload warm-up)
        for w_i in 0..5 {
            cam.orbit_angles((w_i as f64) * 0.03, 0.015);
            let uniforms = cam.to_gpu_uniforms(960.0, 720.0, [0.055, 0.068, 0.095, 1.0]);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mol3d-warmup-enc"),
            });
            gpu_res
                .record_offscreen_3d_pass(&device, &queue, &mut enc, 960, 720, &uniforms, &scene);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Timed 30-frame benchmark with GPU fence wait every frame
        let n_frames = 30usize;
        let t_start = Instant::now();
        for f_i in 0..n_frames {
            cam.orbit_angles(0.04, if f_i % 2 == 0 { 0.01 } else { -0.01 });
            let uniforms = cam.to_gpu_uniforms(960.0, 720.0, [0.055, 0.068, 0.095, 1.0]);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mol3d-bench-enc"),
            });
            gpu_res
                .record_offscreen_3d_pass(&device, &queue, &mut enc, 960, 720, &uniforms, &scene);
            queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let mean_ms = total_ms / (n_frames as f64);
        let fps = 1000.0 / mean_ms.max(1e-6);
        println!(
            "[mol3d bench] adapter={adapter_label} style={style:?} atoms={total_atoms} mean_frame={mean_ms:.3} ms ({fps:.1} fps)"
        );
        assert!(
            mean_ms <= 16.67,
            "Headless wgpu {style:?} on {total_atoms} atoms averaged {mean_ms:.3} ms/frame (< 60 fps)"
        );
    }
}

#[test]
fn test_p2_explicit_canonical_variant_override_persists() {
    let mut ref_data = Reference::new(vec!["CYS".to_string(), "HIS".to_string()], vec![10, 11]);
    ref_data.shifts.insert((0, "CB".to_string()), 41.5); // Oxidized Cys
    ref_data.shifts.insert((1, "ND1".to_string()), 249.2); // His Ne2H
    ref_data.shifts.insert((1, "NE2".to_string()), 167.4);

    let mut ribbon = SequenceRibbonState::new();
    ribbon.sync_inferred_variants_from_reference(&ref_data);
    assert_eq!(ribbon.variant_for(0), ResidueVariant::CysOxidized);
    assert_eq!(ribbon.variant_for(1), ResidueVariant::HisNe2H);

    // Explicit Canonical choice MUST record ResidueVariant::Canonical in user_variant_overrides
    ribbon.set_variant(0, ResidueVariant::Canonical);
    ribbon.set_variant(1, ResidueVariant::Canonical);
    assert_eq!(
        ribbon.user_variant_overrides.get(&0),
        Some(&ResidueVariant::Canonical)
    );
    assert_eq!(
        ribbon.user_variant_overrides.get(&1),
        Some(&ResidueVariant::Canonical)
    );

    ribbon.sync_inferred_variants_from_reference(&ref_data);
    assert_eq!(ribbon.variant_for(0), ResidueVariant::Canonical);
    assert_eq!(ribbon.variant_for(1), ResidueVariant::Canonical);
}
