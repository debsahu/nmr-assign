//! Integration and quantitative gate tests for `nmr_core::paramagnetic` (`T11.4`):
//! - `--metal-ligands` parser and canonical metalloprotein coordination motif auto-detection
//! - Solomon–Bloembergen $r_{\text{M}}^{-6}$ 3D & sequence-shell `PRE` bleaching probabilities
//! - 5-parameter traceless symmetric $\Delta\boldsymbol{\chi}$-tensor linear SVD (`Tm3+`, `Tb3+`, `Dy3+`),
//!   Huber IRLS outlier rejection, and 3D metal-position refinement
//! - Conditional real experimental `CISD3` (`BMRB 51439`, `[2Fe-2S]` metalloprotein) 5-seed gate

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nmr_core::evaluate::score;
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::{parse_pdb_str_first_model, v_dist, Vec3};
use nmr_core::paramagnetic::{
    assign_paramagnetic_backbone, assign_paramagnetic_from_peaks,
    compute_bleaching_probabilities_3d, compute_bleaching_probabilities_from_pdb,
    compute_bleaching_probabilities_sequence, coordinating_histidine_indices,
    detect_metal_coordination_motifs, fit_pcs_tensor_robust, fit_pcs_tensor_svd,
    parse_metal_ligands, predict_pcs_from_tensor, predict_pcs_spherical_ppm,
    refine_metal_position_and_tensor, rotation_matrix_from_zyz, MetalClusterSite, MetalMotifKind,
    ParamagneticAssignConfig, ParamagneticSolverMode, PcsObservation,
};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::rng::Rng;
use nmr_core::solver::{prepare_scoring_priors, AnnealConfig, ScoreConfig};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// Resolve local git-ignored `data/` directory (via `NMR_REAL_DATA_DIR`, workspace `data/`,
/// or the parent repository of `--git-common-dir` when running inside a git worktree).
fn resolve_local_real_data_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(custom);
        return p.is_dir().then_some(p);
    }
    let ws_data = workspace_root().join("data");
    if ws_data.join("private/bmrb51439_cisd3").is_dir() {
        return Some(ws_data);
    }
    let out = Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .current_dir(workspace_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let git_common = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let common_data = git_common.parent()?.join("data");
    if common_data.join("private/bmrb51439_cisd3").is_dir() {
        Some(common_data)
    } else {
        None
    }
}

fn seq_from_one_letter(s: &str) -> Vec<String> {
    s.chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| one_to_three(c).to_string())
        .collect()
}

fn parse_fasta_sequence(path: &Path) -> Vec<String> {
    let raw = fs::read_to_string(path).expect("read FASTA");
    let mut seq = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') {
            continue;
        }
        for ch in trimmed.chars() {
            if ch.is_ascii_alphabetic() {
                let three = one_to_three(ch);
                assert_ne!(three, "UNK", "unexpected residue '{ch}' in FASTA");
                seq.push(three.to_string());
            }
        }
    }
    seq
}

#[test]
fn test_metal_ligand_parser_and_canonical_motif_detection() {
    // Build an 80-residue test sequence containing CYS42, CYS45, HIS68, CYS71 (1-based)
    let mut seq = vec!["ALA".to_string(); 80];
    seq[41] = "CYS".to_string();
    seq[44] = "CYS".to_string();
    seq[67] = "HIS".to_string();
    seq[70] = "CYS".to_string();
    let seq_ids: Vec<i32> = (1..=80).collect();

    let parsed_3 =
        parse_metal_ligands("CYS42,CYS45,HIS68,CYS71", &seq, &seq_ids).expect("parse 3-letter");
    assert_eq!(parsed_3, vec![41, 44, 67, 70]);

    let parsed_1 =
        parse_metal_ligands("C42; C45, H68 C71", &seq, &seq_ids).expect("parse 1-letter");
    assert_eq!(parsed_1, vec![41, 44, 67, 70]);

    let parsed_num = parse_metal_ligands("71, 42, 68, 45", &seq, &[]).expect("parse numeric");
    assert_eq!(parsed_num, vec![41, 44, 67, 70]);

    // Non-1-based seq_ids offset check
    let shifted_ids: Vec<i32> = (101..=180).collect();
    let parsed_shifted = parse_metal_ligands("CYS142,HIS168", &seq, &shifted_ids)
        .expect("parse shifted Author_seq_ID");
    assert_eq!(parsed_shifted, vec![41, 67]);

    // Error handling checks
    assert!(parse_metal_ligands("HIS42", &seq, &seq_ids).is_err());
    assert!(parse_metal_ligands("CYS999", &seq, &seq_ids).is_err());
    assert!(parse_metal_ligands("", &seq, &seq_ids).is_err());
    assert!(parse_metal_ligands("CYS", &seq, &seq_ids).is_err());

    // Canonical motif detection checks:
    // 1. CISD [2Fe-2S] C-X-C-X8-C-X3-H
    let cisd_seq = seq_from_one_letter("AAAAACRCAAAAAAAACDGSHAAAAA");
    let cisd_sites = detect_metal_coordination_motifs(&cisd_seq);
    assert_eq!(cisd_sites.len(), 1);
    assert_eq!(cisd_sites[0].kind, MetalMotifKind::Cisd2Fe2S);
    assert_eq!(cisd_sites[0].ligand_indices, vec![5, 7, 16, 20]);
    assert_eq!(
        coordinating_histidine_indices(&cisd_sites, &cisd_seq),
        vec![20]
    );

    // 2. Bacterial [4Fe-4S] ferredoxin C-X2-C-X2-C-X3-C
    let fd4_seq = seq_from_one_letter("AAAACXXCXXCXXXCAAAA");
    let fd4_sites = detect_metal_coordination_motifs(&fd4_seq);
    assert_eq!(fd4_sites.len(), 1);
    assert_eq!(fd4_sites[0].kind, MetalMotifKind::FerredoxinFeS);
    assert_eq!(fd4_sites[0].ligand_indices, vec![4, 7, 10, 14]);

    // 3. Plant [2Fe-2S] ferredoxin C-X4-C-X2-C-X22-C
    let fd2_seq = seq_from_one_letter("AAACAAAACAACAAAAAAAAAAAAAAAAAAAAAACAAA");
    let fd2_sites = detect_metal_coordination_motifs(&fd2_seq);
    assert_eq!(fd2_sites.len(), 1);
    assert_eq!(fd2_sites[0].kind, MetalMotifKind::FerredoxinFeS);
    assert_eq!(fd2_sites[0].ligand_indices, vec![3, 8, 11, 34]);

    // 4. Rieske [2Fe-2S] C-X-H-X18-C-X2-H
    let rieske_seq = seq_from_one_letter("AAACAHAAAAAAAAAAAAAAAAAACAWHAAA");
    let rieske_sites = detect_metal_coordination_motifs(&rieske_seq);
    assert_eq!(rieske_sites.len(), 1);
    assert_eq!(rieske_sites[0].kind, MetalMotifKind::Rieske2Fe2S);
    assert_eq!(
        coordinating_histidine_indices(&rieske_sites, &rieske_seq),
        vec![5, 27]
    );

    // 5. Classical Cys2His2 zinc/cobalt finger C-X2-C-X12-H-X3-H
    let zf_seq = seq_from_one_letter("AAACPGCAAAAAAAAAAAAHAAAHAAA");
    let zf_sites = detect_metal_coordination_motifs(&zf_seq);
    assert_eq!(zf_sites.len(), 1);
    assert_eq!(zf_sites[0].kind, MetalMotifKind::Cys2His2Finger);
    assert_eq!(zf_sites[0].ligand_indices, vec![3, 6, 19, 23]);
    assert_eq!(
        coordinating_histidine_indices(&zf_sites, &zf_seq),
        vec![19, 23]
    );

    // 6. CCHC retroviral knuckle C-X2-C-X4-H-X4-C
    let cchc_seq = seq_from_one_letter("AAACAACAAAAHAAAACAAA");
    let cchc_sites = detect_metal_coordination_motifs(&cchc_seq);
    assert_eq!(cchc_sites.len(), 1);
    assert_eq!(cchc_sites[0].kind, MetalMotifKind::Cys2HisCysFinger);
    assert_eq!(cchc_sites[0].ligand_indices, vec![3, 6, 11, 16]);

    // 7. Rubredoxin / Cys4 C-X2-C-X20-C-X2-C
    let rub_seq = seq_from_one_letter("AAACPGCAAAAAAAAAAAAAAAAAAAACPGCAAA");
    let rub_sites = detect_metal_coordination_motifs(&rub_seq);
    assert_eq!(rub_sites.len(), 1);
    assert_eq!(rub_sites[0].kind, MetalMotifKind::RubredoxinOrCys4);
    assert_eq!(rub_sites[0].ligand_indices, vec![3, 6, 27, 30]);
}

#[test]
fn test_solomon_3d_and_sequence_bleaching_probabilities() {
    let r_b = 9.0_f64;
    let amides = vec![
        Some([4.5, 0.0, 0.0]),  // r = 0.5 * r_b => 64 / 65 = 0.984615...
        Some([9.0, 0.0, 0.0]),  // r = 1.0 * r_b => 0.5
        Some([18.0, 0.0, 0.0]), // r = 2.0 * r_b => 1 / 65 = 0.0153846...
        None,
    ];
    let p3d = compute_bleaching_probabilities_3d(&amides, &[[0.0, 0.0, 0.0]], r_b);
    assert!((p3d[0] - (64.0 / 65.0)).abs() < 1e-12);
    assert!((p3d[1] - 0.5).abs() < 1e-12);
    assert!((p3d[2] - (1.0 / 65.0)).abs() < 1e-12);
    assert_eq!(p3d[3], 0.0);
    assert!(p3d[0] > p3d[1] && p3d[1] > p3d[2]);

    // Verify PDB-based 3D bleaching extraction on 1D3Z
    let pdb_txt =
        fs::read_to_string(workspace_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let chain = parse_pdb_str_first_model(&pdb_txt).expect("parse 1d3z");
    let seq: Vec<String> = chain.residues.iter().map(|r| r.res_name.clone()).collect();
    let cluster = MetalClusterSite {
        kind: MetalMotifKind::ExplicitLigands,
        ligand_indices: vec![6, 9], // Thr7, Gly10 beta-turn region
        metal_coord: None,
    };
    let p_pdb = compute_bleaching_probabilities_from_pdb(&seq, &chain, &[cluster], 8.5);
    assert_eq!(p_pdb.len(), 76);
    assert!(
        p_pdb[7] > 0.80 && p_pdb[8] > 0.80,
        "residues inside the coordinated beta-turn must lie in the 3D blind sphere"
    );
    assert!(
        p_pdb[30] < 0.05,
        "distal alpha-helix residue must remain diamagnetic (got {:.4})",
        p_pdb[30]
    );

    // Sequence-only shell probabilities around CISD-like [2Fe-2S] cluster
    let cisd_cluster = MetalClusterSite {
        kind: MetalMotifKind::Cisd2Fe2S,
        ligand_indices: vec![25, 27, 36, 40],
        metal_coord: None,
    };
    let p_seq = compute_bleaching_probabilities_sequence(60, &[cisd_cluster]);
    assert!(p_seq[25] >= 0.95 && p_seq[27] >= 0.95 && p_seq[36] >= 0.95 && p_seq[40] >= 0.95);
    assert!(
        p_seq[26] >= 0.90 && p_seq[38] >= 0.90,
        "short intra-chelate linkers must be strongly bleached"
    );
    assert!(
        p_seq[41] >= 0.85 && p_seq[44] >= 0.65,
        "C-terminal covalent amide + NH...S turn cap must be inside blind zone (>= 0.65)"
    );
    assert_eq!(p_seq[5], 0.0);
    assert_eq!(p_seq[55], 0.0);
}

#[test]
fn test_synthetic_lanthanide_pcs_svd_robust_irls_and_metal_refinement() {
    let pdb_txt =
        fs::read_to_string(workspace_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let chain = parse_pdb_str_first_model(&pdb_txt).expect("parse 1d3z");

    // Place a lanthanide coordination site inside the 1D3Z metalloprotein frame (centroid ~ [51.5, -81.0, -2.4])
    // and collect all first-model 1D3Z nuclear sites outside the 10.5 Å Curie PRE blind sphere
    // (Schmitz et al. 2008 NUMBAT multi-nuclear PCS benchmark).
    let metal_true: Vec3 = [51.5, -81.0, -2.4];
    let mut amides: Vec<(i32, Vec3)> = Vec::new();
    for line in pdb_txt.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        if line.starts_with("ATOM  ") && line.len() >= 54 {
            let sid: i32 = line[22..26].trim().parse().unwrap_or(1);
            let x: f64 = line[30..38].trim().parse().expect("pdb x");
            let y: f64 = line[38..46].trim().parse().expect("pdb y");
            let z: f64 = line[46..54].trim().parse().expect("pdb z");
            let pos = [x, y, z];
            if v_dist(pos, metal_true) >= 10.5 {
                amides.push((sid, pos));
            }
        }
    }
    let _ = chain;
    assert!(
        amides.len() >= 500,
        "expected >= 500 non-bleached 1D3Z nuclear sites (r >= 10.5 Å) for lanthanide PCS test, got {}",
        amides.len()
    );

    // Canonical lanthanide susceptibility anisotropy tensors in 10^-32 m^3 (UTS ordering |zz| >= |xx| >= |yy|)
    let lanthanides = [
        ("Tm3+", 22.4_f64, -5.8_f64, [0.65_f64, 1.05, 2.15]),
        ("Tb3+", -34.2_f64, 8.6_f64, [1.20_f64, 0.82, 0.45]),
        ("Dy3+", -38.5_f64, 11.2_f64, [2.35_f64, 1.38, 1.70]),
    ];

    for (ion_name, true_ax, true_rh, true_euler) in lanthanides {
        // 1. Noise-free exact recovery gate (< 0.01% error)
        let clean_obs: Vec<PcsObservation> = amides
            .iter()
            .map(|&(seq_id, pos)| {
                let pcs = predict_pcs_spherical_ppm(pos, metal_true, true_ax, true_rh, true_euler);
                PcsObservation::new(seq_id, pos, pcs).with_uncertainty(0.03)
            })
            .collect();

        let fit_clean = fit_pcs_tensor_svd(&clean_obs, metal_true).expect("SVD fit clean");

        // Verify forward spherical PAF == Cartesian design-row prediction to machine precision
        for o in &clean_obs {
            let pred_cart = predict_pcs_from_tensor(o.coord_angstrom, &fit_clean);
            assert!(
                (pred_cart - o.pcs_ppm).abs() < 1e-12,
                "{ion_name}: Cartesian vs spherical PCS mismatch: {} vs {}",
                pred_cart,
                o.pcs_ppm
            );
        }

        // Verify UTS eigenvalue ordering, tracelessness, right-handedness, and rhombicity <= 2/3
        let [xx_p, yy_p, zz_p] = fit_clean.eigenvalues_paf;
        assert!((xx_p + yy_p + zz_p).abs() < 1e-11, "PAF trace must vanish");
        assert!(
            zz_p.abs() >= xx_p.abs() - 1e-12 && xx_p.abs() >= yy_p.abs() - 1e-12,
            "{ion_name}: UTS ordering violated: {:?}",
            fit_clean.eigenvalues_paf
        );
        assert!(fit_clean.rhombicity <= 2.0 / 3.0 + 1e-12);
        let r_recon = rotation_matrix_from_zyz(
            fit_clean.euler_zyz_rad[0],
            fit_clean.euler_zyz_rad[1],
            fit_clean.euler_zyz_rad[2],
        );
        for (row_rec, row_fit) in r_recon.iter().zip(&fit_clean.rotation_mol_to_paf) {
            for (&val_rec, &val_fit) in row_rec.iter().zip(row_fit) {
                assert!(
                    (val_rec - val_fit).abs() < 1e-10,
                    "{ion_name}: ZYZ Euler reconstruction mismatch"
                );
            }
        }

        let rel_ax_clean = ((fit_clean.dchi_ax - true_ax) / true_ax).abs() * 100.0;
        let rel_rh_clean = ((fit_clean.dchi_rh - true_rh) / true_rh).abs() * 100.0;
        println!(
            "{ion_name} noise-free SVD: dchi_ax={:.4} (err {:.6}%), dchi_rh={:.4} (err {:.6}%), Q={:.3e}, cond={:.2}",
            fit_clean.dchi_ax, rel_ax_clean, fit_clean.dchi_rh, rel_rh_clean, fit_clean.q_factor, fit_clean.condition_number
        );
        assert!(
            rel_ax_clean < 0.01 && rel_rh_clean < 0.01,
            "{ion_name} noise-free error exceeded 0.01%: ax={rel_ax_clean:.6}%, rh={rel_rh_clean:.6}%"
        );
        assert!(fit_clean.q_factor < 1e-12 && fit_clean.rmsd_ppm < 1e-12);

        // 2. Realistic noisy recovery gate: 0.03 ppm shift noise + 0.3 Å 3D RMS coordinate noise
        //    (sigma_xyz = 0.3 / sqrt(3) Å per Cartesian axis => E[||dr||^2]^1/2 = 0.30 Å)
        let sigma_xyz = 0.30_f64 / 3.0_f64.sqrt();
        let mut rng = Rng::new(2026_1104 ^ (true_ax.to_bits() & 0xFFFF));
        let noisy_obs: Vec<PcsObservation> = amides
            .iter()
            .map(|&(seq_id, pos)| {
                let exact_pcs =
                    predict_pcs_spherical_ppm(pos, metal_true, true_ax, true_rh, true_euler);
                let noisy_coord = [
                    pos[0] + rng.normal(0.0, sigma_xyz),
                    pos[1] + rng.normal(0.0, sigma_xyz),
                    pos[2] + rng.normal(0.0, sigma_xyz),
                ];
                let noisy_pcs = exact_pcs + rng.normal(0.0, 0.03);
                PcsObservation::new(seq_id, noisy_coord, noisy_pcs)
                    .with_uncertainty(0.03)
                    .with_coord_uncertainty(0.30)
            })
            .collect();

        let fit_noisy = fit_pcs_tensor_svd(&noisy_obs, metal_true).expect("SVD fit noisy");
        let rel_ax_noisy = ((fit_noisy.dchi_ax - true_ax) / true_ax).abs() * 100.0;
        let rel_rh_noisy = ((fit_noisy.dchi_rh - true_rh) / true_rh).abs() * 100.0;
        println!(
            "{ion_name} noisy (0.03 ppm + 0.3 A) SVD: dchi_ax={:.3} (err {:.3}%), dchi_rh={:.3} (err {:.3}%), RMSD={:.3} ppm, Q={:.4}",
            fit_noisy.dchi_ax, rel_ax_noisy, fit_noisy.dchi_rh, rel_rh_noisy, fit_noisy.rmsd_ppm, fit_noisy.q_factor
        );
        assert!(
            rel_ax_noisy < 1.0 && rel_rh_noisy < 1.0,
            "{ion_name} noisy (0.03 ppm + 0.3 A) error exceeded 1.0%: ax={rel_ax_noisy:.3}%, rh={rel_rh_noisy:.3}%"
        );
    }

    // 3. Robust Huber IRLS test against contact-shift / misassignment outliers
    let (true_ax, true_rh, true_euler) = (-34.2_f64, 8.6_f64, [1.20_f64, 0.82, 0.45]);
    let mut outlier_obs: Vec<PcsObservation> = amides
        .iter()
        .map(|&(seq_id, pos)| {
            let pcs = predict_pcs_spherical_ppm(pos, metal_true, true_ax, true_rh, true_euler);
            PcsObservation::new(seq_id, pos, pcs).with_uncertainty(0.03)
        })
        .collect();
    outlier_obs[3].pcs_ppm += 5.2;
    outlier_obs[17].pcs_ppm -= 4.4;
    outlier_obs[41].pcs_ppm += 4.9;

    let ols_contaminated = fit_pcs_tensor_svd(&outlier_obs, metal_true).expect("OLS");
    let huber_fit = fit_pcs_tensor_robust(&outlier_obs, metal_true, 0.08, 25).expect("Huber IRLS");
    let ols_ax_err = ((ols_contaminated.dchi_ax - true_ax) / true_ax).abs() * 100.0;
    let huber_ax_err = ((huber_fit.dchi_ax - true_ax) / true_ax).abs() * 100.0;
    let huber_rh_err = ((huber_fit.dchi_rh - true_rh) / true_rh).abs() * 100.0;
    assert!(
        huber_ax_err < ols_ax_err && huber_ax_err < 0.6 && huber_rh_err < 0.8,
        "Huber IRLS must suppress PCS outliers: ols_ax_err={ols_ax_err:.3}%, huber_ax_err={huber_ax_err:.3}%, huber_rh_err={huber_rh_err:.3}%"
    );

    // 4. Joint 8-parameter 3D metal-position + 5-component tensor VarPro refinement
    let clean_tb_obs: Vec<PcsObservation> = amides
        .iter()
        .map(|&(seq_id, pos)| {
            let pcs = predict_pcs_spherical_ppm(pos, metal_true, true_ax, true_rh, true_euler);
            PcsObservation::new(seq_id, pos, pcs).with_uncertainty(0.03)
        })
        .collect();
    let displaced_guess = [
        metal_true[0] + 1.35,
        metal_true[1] - 1.10,
        metal_true[2] + 0.85,
    ];
    let refined = refine_metal_position_and_tensor(&clean_tb_obs, displaced_guess, 2.5, 0.75)
        .expect("refine metal position");
    let pos_err = v_dist(refined.metal_pos_angstrom, metal_true);
    let ref_ax_err = ((refined.dchi_ax - true_ax) / true_ax).abs() * 100.0;
    let ref_rh_err = ((refined.dchi_rh - true_rh) / true_rh).abs() * 100.0;
    println!(
        "VarPro metal refinement from {:.2} A offset -> pos_err={:.5} A, ax_err={:.5}%, rh_err={:.5}%",
        v_dist(displaced_guess, metal_true),
        pos_err,
        ref_ax_err,
        ref_rh_err
    );
    assert!(
        pos_err < 0.02 && ref_ax_err < 0.05 && ref_rh_err < 0.05,
        "metal position refinement failed: pos_err={pos_err:.5} A, ax_err={ref_ax_err:.5}%, rh_err={ref_rh_err:.5}%"
    );

    // Verify coord_uncertainty_angstrom propagation in both fit_pcs_tensor_robust and refine_metal_position_and_tensor
    let mut coord_unc_obs: Vec<PcsObservation> = clean_tb_obs
        .iter()
        .map(|o| o.with_coord_uncertainty(0.25))
        .collect();
    coord_unc_obs[2].pcs_ppm += 3.50; // outlier suppressed by Huber IRLS
    let rob_coord =
        fit_pcs_tensor_robust(&coord_unc_obs, metal_true, 0.20, 15).expect("robust + coord unc");
    assert!(
        ((rob_coord.dchi_ax - true_ax) / true_ax).abs() < 0.02,
        "robust IRLS with coord_uncertainty_angstrom failed: dchi_ax={:.4}",
        rob_coord.dchi_ax
    );
    let clean_coord_unc: Vec<PcsObservation> = clean_tb_obs
        .iter()
        .map(|o| o.with_coord_uncertainty(0.25))
        .collect();
    let ref_coord = refine_metal_position_and_tensor(&clean_coord_unc, displaced_guess, 2.5, 0.75)
        .expect("refine + coord unc");
    assert!(v_dist(ref_coord.metal_pos_angstrom, metal_true) < 0.05);
}

fn map_spin_systems_to_truth(systems: &[SpinSystem], truth_ref: &Reference) -> Vec<i32> {
    let mut out = vec![-1i32; systems.len()];
    for (s, ss) in systems.iter().enumerate() {
        let mut best_k = -1i32;
        let mut best_d = 1.0_f64;
        for k in 1..truth_ref.sequence.len() {
            let Some(&th) = truth_ref.shifts.get(&(k, "H".to_string())) else {
                continue;
            };
            let Some(&tn) = truth_ref.shifts.get(&(k, "N".to_string())) else {
                continue;
            };
            let d = ((ss.h - th) / 0.045).hypot((ss.n - tn) / 0.40);
            if d <= 1.0 && d < best_d {
                best_d = d;
                best_k = k as i32;
            }
        }
        out[s] = best_k;
    }
    out
}

fn fit_unlabelled_hn_offset(systems: &[SpinSystem], truth_ref: &Reference) -> (f64, f64) {
    let mut t_hn = Vec::new();
    for k in 1..truth_ref.sequence.len() {
        if let (Some(&h), Some(&n)) = (
            truth_ref.shifts.get(&(k, "H".to_string())),
            truth_ref.shifts.get(&(k, "N".to_string())),
        ) {
            t_hn.push((h, n));
        }
    }
    if systems.len() < 5 || t_hn.len() < 5 {
        return (0.0, 0.0);
    }
    let eval_ll = |dh: f64, dn: f64| -> f64 {
        let mut ll = -0.5 * ((dh / 0.35).powi(2) + (dn / 2.50).powi(2));
        for ss in systems {
            let mut max_lp = -4.5_f64;
            for &(th, tn) in &t_hn {
                let lp =
                    -0.5 * (((ss.h - dh - th) / 0.045).powi(2) + ((ss.n - dn - tn) / 0.40).powi(2));
                if lp > max_lp {
                    max_lp = lp;
                }
            }
            let mut sum_exp = (-4.5 - max_lp).exp();
            for &(th, tn) in &t_hn {
                let lp =
                    -0.5 * (((ss.h - dh - th) / 0.045).powi(2) + ((ss.n - dn - tn) / 0.40).powi(2));
                sum_exp += (lp - max_lp).exp();
            }
            ll += max_lp + sum_exp.ln();
        }
        ll
    };
    let ll0 = eval_ll(0.0, 0.0);
    let mut best = (0.0_f64, 0.0_f64, ll0);
    for ih in -20..=20 {
        let dh = (ih as f64) * 0.025;
        for jn in -30..=30 {
            let dn = (jn as f64) * 0.10;
            let v = eval_ll(dh, dn);
            if v > best.2 {
                best = (dh, dn, v);
            }
        }
    }
    let (c_dh, c_dn, _) = best;
    for ih in -10..=10 {
        let dh = c_dh + (ih as f64) * 0.003;
        for jn in -10..=10 {
            let dn = c_dn + (jn as f64) * 0.015;
            let v = eval_ll(dh, dn);
            if v > best.2 {
                best = (dh, dn, v);
            }
        }
    }
    if best.2 - ll0 >= 4.0 && (best.0.abs() >= 0.05 || best.1.abs() >= 0.35) {
        (best.0, best.1)
    } else {
        (0.0, 0.0)
    }
}

/// Count false-positive placements inside the experimentally bleached `[2Fe-2S]` coordination
/// loops of `CISD3` (`BMRB 51439`: residues inside the two `[2Fe-2S]` loop shells lacking
/// observable `(H, N)` shifts in `bmr51439_3.str`, i.e. 1-based `26..=31, 37..=45, 64..=70, 75..=80`).
fn count_bleached_loop_false_positives(
    positions: &[i32],
    sequence: &[String],
    truth_ref: &Reference,
) -> usize {
    // Coordination-loop windows spanning both [2Fe-2S] domains (C26-H41+turn and C64-H79+turn)
    let in_coord_domain = |k: usize| -> bool { (25..=44).contains(&k) || (63..=79).contains(&k) };
    let mut fp = 0usize;
    for &pos in positions {
        if pos < 0 {
            continue;
        }
        let k = pos as usize;
        if k < sequence.len() && in_coord_domain(k) {
            let has_truth_hn = truth_ref.shifts.contains_key(&(k, "H".to_string()))
                && truth_ref.shifts.contains_key(&(k, "N".to_string()));
            if !has_truth_hn {
                fp += 1;
            }
        }
    }
    fp
}

#[test]
fn test_real_cisd3_paramagnetic_5seed_gate() {
    let Some(data_dir) = resolve_local_real_data_dir() else {
        eprintln!("Skipping CISD3 paramagnetic 5-seed gate: local data directory not present");
        return;
    };
    let cisd3_dir = data_dir.join("private/bmrb51439_cisd3");
    let fasta_path = cisd3_dir.join("cisd3.fasta");
    let star_path = cisd3_dir.join("bmr51439_3.str");
    if !fasta_path.is_file() || !star_path.is_file() {
        eprintln!("Skipping CISD3 paramagnetic 5-seed gate: private/bmrb51439_cisd3 files missing");
        return;
    }

    let sequence = parse_fasta_sequence(&fasta_path);
    assert_eq!(sequence.len(), 93);

    // Verify sequence-only [2Fe-2S] CISD motif detection & coordinating His identification
    let detected = detect_metal_coordination_motifs(&sequence);
    assert_eq!(
        detected.len(),
        2,
        "CISD3 must auto-detect both tandem [2Fe-2S] 3Cys-1His clusters"
    );
    assert_eq!(detected[0].kind, MetalMotifKind::Cisd2Fe2S);
    assert_eq!(detected[0].ligand_indices, vec![25, 27, 36, 40]); // C26, C28, C37, H41
    assert_eq!(detected[1].kind, MetalMotifKind::Cisd2Fe2S);
    assert_eq!(detected[1].ligand_indices, vec![63, 65, 74, 78]); // C64, C66, C75, H79
    assert_eq!(
        coordinating_histidine_indices(&detected, &sequence),
        vec![40, 78]
    );

    let explicit_ligs = parse_metal_ligands(
        "CYS26,CYS28,CYS37,HIS41,CYS64,CYS66,CYS75,HIS79",
        &sequence,
        &[],
    )
    .expect("parse explicit CISD3 ligands");
    assert_eq!(explicit_ligs, vec![25, 27, 36, 40, 63, 65, 74, 78]);

    let mut truth_ref = read_star(&star_path, None).expect("read BMRB 51439 STAR");
    let assignable_amides = (1..sequence.len())
        .filter(|&k| {
            sequence[k] != "PRO"
                && truth_ref.shifts.contains_key(&(k, "H".to_string()))
                && truth_ref.shifts.contains_key(&(k, "N".to_string()))
        })
        .count();
    assert_eq!(assignable_amides, 51);

    // Pick 2D HSQC + 4 3D Bruker spectra identically to the standard evaluation pipeline
    let pcfg = PickerConfig::default();
    let hsqc_spec = read_bruker_pdata_2d(cisd3_dir.join("CISD3_15N_HSQC_diamagnetic/pdata/1"))
        .expect("read CISD3 HSQC");
    let hsqc_roots = pick_hsqc_2d(&hsqc_spec, &pcfg);
    let mut raw_peaks = hsqc_roots.clone();
    for (rel_path, exp_name) in [
        ("CISD3_HNCA/pdata/1", "HNCA"),
        ("CISD3_HNCOCA/pdata/1", "HNCOCA"),
        ("CISD3_CBCACONH/pdata/1", "CBCACONH"),
        ("CISD3_HNCO/pdata/1", "HNCO"),
    ] {
        let spec3d = read_bruker_pdata_3d(cisd3_dir.join(rel_path), exp_name).expect("read 3D");
        raw_peaks.extend(pick_triple_3d(&spec3d, &hsqc_roots, &pcfg));
    }

    let priors = Priors::bmrb();
    let scfg = ScoreConfig::default();
    let probe_systems =
        build_spin_systems(&raw_peaks, &Tolerances::default(), 3, false).expect("spin systems");
    let (dh, dn) = fit_unlabelled_hn_offset(&probe_systems, &truth_ref);
    for ((_, atom), val) in truth_ref.shifts.iter_mut() {
        if atom == "H" {
            *val += dh;
        } else if atom == "N" {
            *val += dn;
        }
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);

    // -------------------------------------------------------------------------
    // 1. Standard Solver: Globular Baseline vs Paramagnetic PRE Blind-Zone Prior
    // -------------------------------------------------------------------------
    let _ = prepare_scoring_priors;
    let std_truth = map_spin_systems_to_truth(&probe_systems, &truth_ref);
    let mut glob_std_accs = Vec::new();
    let mut glob_std_recs = Vec::new();
    let mut glob_std_loop_fps = 0usize;
    let mut para_std_accs = Vec::new();
    let mut para_std_recs = Vec::new();
    let mut para_std_hcps = Vec::new();
    let mut para_std_loop_fps = 0usize;

    for seed in 0..5u64 {
        let acfg = AnnealConfig {
            seed,
            ..Default::default()
        };
        let glob_res = nmr_core::solver::assign(&probe_systems, &sequence, &priors, &scfg, &acfg);
        glob_std_loop_fps +=
            count_bleached_loop_false_positives(&glob_res.position, &sequence, &truth_ref);
        let mut gm = score(&glob_res, &std_truth, 0.8);
        gm.assignable = assignable_amides;
        glob_std_accs.push(gm.accuracy() * 100.0);
        glob_std_recs.push(gm.recall() * 100.0);

        let p_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::Standard,
            seed,
            ..Default::default()
        };
        let (para_res, para_rep) =
            assign_paramagnetic_backbone(&probe_systems, &sequence, &priors, &p_cfg);
        assert_eq!(para_rep.detected_clusters.len(), 2);
        let fp = count_bleached_loop_false_positives(&para_res.position, &sequence, &truth_ref);
        para_std_loop_fps += fp;
        let mut pm = score(&para_res, &std_truth, 0.8);
        pm.assignable = assignable_amides;
        para_std_accs.push(pm.accuracy() * 100.0);
        para_std_recs.push(pm.recall() * 100.0);
        para_std_hcps.push(pm.hc_precision() * 100.0);
    }

    println!(
        "CISD3 5-seed Standard: globular_acc={:.2}%/rec={:.2}% (bleached_loop_FP={}), paramagnetic_acc={:.2}%/rec={:.2}%, HC_prec={:.2}% (bleached_loop_FP={})",
        mean(&glob_std_accs),
        mean(&glob_std_recs),
        glob_std_loop_fps,
        mean(&para_std_accs),
        mean(&para_std_recs),
        mean(&para_std_hcps),
        para_std_loop_fps
    );
    assert_eq!(
        para_std_loop_fps, 0,
        "Standard paramagnetic solver must emit ZERO false-positive assignments inside bleached [2Fe-2S] coordination loops"
    );
    assert!(
        mean(&para_std_hcps) >= 95.0,
        "CISD3 5-seed Standard paramagnetic HC precision fell below 95.0%: {:.2}%",
        mean(&para_std_hcps)
    );
    assert!(
        mean(&para_std_accs) >= mean(&glob_std_accs) - 1e-9
            && mean(&para_std_recs) >= mean(&glob_std_recs) - 1e-9,
        "Paramagnetic Standard 5-seed accuracy/recall ({:.2}%/{:.2}%) must preserve or beat globular Standard ({:.2}%/{:.2}%)",
        mean(&para_std_accs),
        mean(&para_std_recs),
        mean(&glob_std_accs),
        mean(&glob_std_recs)
    );

    // -------------------------------------------------------------------------
    // 2. EvoGraph Solver: Globular Baseline vs Paramagnetic (Auto & Explicit)
    // -------------------------------------------------------------------------
    let mut glob_evo_accs = Vec::new();
    let mut glob_evo_recs = Vec::new();
    let mut glob_evo_loop_fps = 0usize;
    let mut para_evo_accs = Vec::new();
    let mut para_evo_recs = Vec::new();
    let mut para_evo_hcps = Vec::new();
    let mut para_evo_loop_fps = 0usize;
    let mut explicit_evo_accs = Vec::new();
    let mut explicit_evo_recs = Vec::new();
    let mut explicit_evo_hcps = Vec::new();
    let mut explicit_evo_loop_fps = 0usize;

    for seed in 0..5u64 {
        let evo_cfg = EvoGraphConfig {
            structure_torsions: None,
            seed,
            ..Default::default()
        };
        let (g_sys, g_res, _) =
            assign_evo_graph_from_peaks(&raw_peaks, &sequence, &priors, &evo_cfg);
        glob_evo_loop_fps +=
            count_bleached_loop_false_positives(&g_res.position, &sequence, &truth_ref);
        let g_truth = map_spin_systems_to_truth(&g_sys, &truth_ref);
        let mut gm = score(&g_res, &g_truth, 0.8);
        gm.assignable = assignable_amides;
        glob_evo_accs.push(gm.accuracy() * 100.0);
        glob_evo_recs.push(gm.recall() * 100.0);

        // Sequence-only auto-detected [2Fe-2S] motifs
        let p_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            seed,
            ..Default::default()
        };
        let (p_sys, p_res, p_rep) =
            assign_paramagnetic_from_peaks(&raw_peaks, &sequence, &priors, &p_cfg);
        assert_eq!(p_rep.detected_clusters.len(), 2);
        para_evo_loop_fps +=
            count_bleached_loop_false_positives(&p_res.position, &sequence, &truth_ref);
        let p_truth = map_spin_systems_to_truth(&p_sys, &truth_ref);
        let mut pm = score(&p_res, &p_truth, 0.8);
        pm.assignable = assignable_amides;
        para_evo_accs.push(pm.accuracy() * 100.0);
        para_evo_recs.push(pm.recall() * 100.0);
        para_evo_hcps.push(pm.hc_precision() * 100.0);

        // Explicit --metal-ligands mode
        let exp_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            explicit_ligand_indices: explicit_ligs.clone(),
            auto_detect_motifs: false,
            seed,
            ..Default::default()
        };
        let (e_sys, e_res, _) =
            assign_paramagnetic_from_peaks(&raw_peaks, &sequence, &priors, &exp_cfg);
        explicit_evo_loop_fps +=
            count_bleached_loop_false_positives(&e_res.position, &sequence, &truth_ref);
        let e_truth = map_spin_systems_to_truth(&e_sys, &truth_ref);
        let mut em = score(&e_res, &e_truth, 0.8);
        em.assignable = assignable_amides;
        explicit_evo_accs.push(em.accuracy() * 100.0);
        explicit_evo_recs.push(em.recall() * 100.0);
        explicit_evo_hcps.push(em.hc_precision() * 100.0);
    }

    println!(
        "CISD3 5-seed EvoGraph: globular_acc={:.2}%/rec={:.2}% (bleached_loop_FP={}), para_auto_acc={:.2}%/rec={:.2}% / HC_prec={:.2}% (bleached_loop_FP={}), para_explicit_acc={:.2}%/rec={:.2}% / HC_prec={:.2}% (bleached_loop_FP={})",
        mean(&glob_evo_accs),
        mean(&glob_evo_recs),
        glob_evo_loop_fps,
        mean(&para_evo_accs),
        mean(&para_evo_recs),
        mean(&para_evo_hcps),
        para_evo_loop_fps,
        mean(&explicit_evo_accs),
        mean(&explicit_evo_recs),
        mean(&explicit_evo_hcps),
        explicit_evo_loop_fps
    );

    assert_eq!(
        para_evo_loop_fps, 0,
        "EvoGraph paramagnetic solver (auto) must emit ZERO false-positive assignments inside bleached [2Fe-2S] coordination loops"
    );
    assert_eq!(
        explicit_evo_loop_fps, 0,
        "EvoGraph paramagnetic solver (--metal-ligands) must emit ZERO false-positive assignments inside bleached [2Fe-2S] coordination loops"
    );
    assert!(
        mean(&para_evo_hcps) >= 95.0 && mean(&explicit_evo_hcps) >= 95.0,
        "CISD3 5-seed paramagnetic HC precision fell below 95.0%: auto={:.2}%, explicit={:.2}%",
        mean(&para_evo_hcps),
        mean(&explicit_evo_hcps)
    );
    assert!(
        mean(&para_evo_accs) >= mean(&glob_evo_accs) - 1e-9
            && mean(&explicit_evo_accs) >= mean(&glob_evo_accs) - 1e-9
            && mean(&para_evo_recs) >= mean(&glob_evo_recs) - 1e-9
            && mean(&explicit_evo_recs) >= mean(&glob_evo_recs) - 1e-9,
        "CISD3 5-seed paramagnetic EvoGraph acc/rec (auto={:.2}%/{:.2}%, explicit={:.2}%/{:.2}%) must preserve or beat globular ({:.2}%/{:.2}%)",
        mean(&para_evo_accs),
        mean(&para_evo_recs),
        mean(&explicit_evo_accs),
        mean(&explicit_evo_recs),
        mean(&glob_evo_accs),
        mean(&glob_evo_recs)
    );
}

#[test]
fn test_two_slot_hnca_hncoca_diamagnetic_spin_system_not_stolen_by_bleached_distractor() {
    use nmr_core::solver::{assign_from_matrices_with_locks, pair_matrix, unary_matrix};
    use std::collections::BTreeMap;

    let priors = Priors::bmrb();
    // Sequence:
    // 0:M 1:T 2:A(true diamag target, prev=THR, CA~61.6) 3:P ... 6:P 7:C(lig) 8:A(bleached distractor, prev=CYS, CA~58.2, p_bleach>=0.90) 9:C(lig) 10:G 11:H(lig) 12:G 13:C(lig) 14:G
    let seq_1 = "MTAPGGPCACGHGCG";
    let sequence: Vec<String> = seq_1
        .chars()
        .map(|ch| one_to_three(ch).to_string())
        .collect();
    let ligs =
        parse_metal_ligands("CYS8,CYS10,HIS12,CYS14", &sequence, &[]).expect("parse ligands");

    // 2-slot HNCA + HNCOCA spin system (count_positive_carbon_slots == 2 < 3):
    // True origin is k_true = 2 (ALA2, preceded by THR1), but CA(-1) = 59.5 ppm sits slightly closer
    // to CYS7 (~58.2 ppm) than THR1 (~61.6 ppm) so diamagnetic u_base prefers bleached ALA8 (k=8)
    // while a weaker competitor s_comp squats on k_true = 2 under globular annealing!
    let mut cands_main = BTreeMap::new();
    cands_main.insert(("CA".to_string(), 0), vec![(52.5, 2)]);
    cands_main.insert(("CA".to_string(), -1), vec![(59.5, 2)]);
    let ss_main = SpinSystem {
        h: 8.20,
        n: 123.5,
        cands: cands_main,
        root_height: Some(1.0),
        rel_root_height: Some(1.0),
        rel_root_intensity: Some(1.0),
    };

    let mut cands_comp = BTreeMap::new();
    cands_comp.insert(("CA".to_string(), 0), vec![(54.8, 1)]);
    cands_comp.insert(("CA".to_string(), -1), vec![(61.5, 1)]);
    let ss_comp = SpinSystem {
        h: 8.45,
        n: 125.8,
        cands: cands_comp,
        root_height: Some(1.0),
        rel_root_height: Some(1.0),
        rel_root_intensity: Some(1.0),
    };
    let systems = vec![ss_main, ss_comp];

    // Confirm u_base attracts ss_main (s=0) to bleached k=8 over diamagnetic k=2 under unmodified globular annealing
    let scfg = ScoreConfig {
        auto_reference: false,
        ..ScoreConfig::default()
    };
    let u_base = unary_matrix(&systems, &sequence, &priors, &scfg);
    let p_mat = pair_matrix(&systems, &scfg);
    assert!(
        u_base.get(0, 8) > u_base.get(0, 2),
        "setup check: u_base(0, 8)={:.2} must exceed u_base(0, 2)={:.2}",
        u_base.get(0, 8),
        u_base.get(0, 2)
    );
    let glob_res = assign_from_matrices_with_locks(&u_base, &p_mat, &AnnealConfig::default(), &[]);
    assert_eq!(
        glob_res.position[0], 8,
        "globular baseline should place s=0 onto bleached distractor k=8"
    );

    // Under assign_paramagnetic_backbone, u_para forbids k=8 (p_bleach >= 0.90) DURING annealing,
    // so 2-slot s=0 wins its true diamagnetic slot k=2 instead of being evicted to -1!
    let p_cfg = ParamagneticAssignConfig {
        mode: ParamagneticSolverMode::Standard,
        explicit_ligand_indices: ligs,
        auto_detect_motifs: false,
        score_cfg: scfg,
        seed: 0,
        ..Default::default()
    };
    let (para_res, para_rep) = assign_paramagnetic_backbone(&systems, &sequence, &priors, &p_cfg);
    assert!(
        para_rep.p_bleach[8] >= 0.90,
        "distractor k=8 must have p_bleach >= 0.90, got {:.3}",
        para_rep.p_bleach[8]
    );
    assert_eq!(
        para_res.position[0], 2,
        "2-slot HNCA+HNCOCA diamagnetic spin system must be assigned to true slot k=2 (not evicted to -1), got {}",
        para_res.position[0]
    );
}

#[test]
fn test_tandem_two_cluster_explicit_ligands_with_3d_pdb_preserves_bimodal_bleaching() {
    // Construct a 25-residue synthetic PDB with TWO spatially separated 4-ligand metal sites:
    // - Cluster 1 at x ~ 0 Å: CYS2, CYS4, HIS6, CYS8 (cluster center ~ (0, 0, 0) Å)
    // - Diamagnetic core midpoint at k = 12 (ALA13) placed at (13.0, 0, 0) Å (~13 Å from both clusters!)
    // - Cluster 2 at x ~ 26 Å: CYS17, CYS19, HIS21, CYS23 (cluster center ~ (26, 0, 0) Å)
    let seq_1 = "ACACAHACAGGGAAAGACACAHACA";
    let sequence: Vec<String> = seq_1
        .chars()
        .map(|ch| one_to_three(ch).to_string())
        .collect();
    assert_eq!(sequence.len(), 25);

    let mut pdb = String::new();
    let mut serial = 1usize;
    for (k, res) in sequence.iter().enumerate() {
        let seq_id = k + 1;
        let (cx, cy, cz): (f64, f64, f64) = if k <= 8 {
            // Cluster 1 shell around (0, 0, 0) Å (radius ~3.2 Å)
            let ang = (k as f64) * 0.75;
            (3.2 * ang.cos(), 3.2 * ang.sin(), 0.5)
        } else if k >= 16 {
            // Cluster 2 shell around (26, 0, 0) Å (radius ~3.2 Å)
            let ang = ((k - 16) as f64) * 0.75;
            (26.0 + 3.2 * ang.cos(), 3.2 * ang.sin(), -0.5)
        } else {
            // Diamagnetic bridge around midpoint (13, 0, 0) Å (13 Å from both clusters!)
            let offset = ((k as isize) - 12) as f64 * 0.8;
            (13.0 + offset, 0.0, 0.0)
        };

        for (atom, dx, dy, dz) in [
            ("N", -0.4, 0.0, 0.0),
            ("H", -0.8, 0.2, 0.0),
            ("CA", 0.0, 0.0, 0.0),
            ("C", 0.5, 0.3, 0.0),
        ] {
            pdb.push_str(&format!(
                "ATOM  {:5} {:<4} {:>3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           {}\n",
                serial,
                atom,
                res,
                seq_id,
                cx + dx,
                cy + dy,
                cz + dz,
                &atom[..1]
            ));
            serial += 1;
        }
        if res != "GLY" {
            pdb.push_str(&format!(
                "ATOM  {:5} {:<4} {:>3} A{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C\n",
                serial,
                "CB",
                res,
                seq_id,
                cx * 0.65 + if k >= 16 { 26.0 * 0.35 } else { 0.0 },
                cy * 0.65,
                cz * 0.65
            ));
            serial += 1;
        }
    }
    pdb.push_str("END\n");

    let all_8_ligs = parse_metal_ligands(
        "CYS2,CYS4,HIS6,CYS8,CYS18,CYS20,HIS22,CYS24",
        &sequence,
        &[],
    )
    .expect("parse 8 tandem ligands");
    let chain = parse_pdb_str_first_model(&pdb).expect("parse synthetic tandem PDB");

    // 1. Direct call with a single combined ExplicitLigands cluster (metal_coord: None)
    let single_site = vec![MetalClusterSite {
        kind: MetalMotifKind::ExplicitLigands,
        ligand_indices: all_8_ligs.clone(),
        metal_coord: None,
    }];
    let pb_direct = compute_bleaching_probabilities_from_pdb(&sequence, &chain, &single_site, 8.5);
    assert!(
        pb_direct[1] > 0.80 && pb_direct[5] > 0.80 && pb_direct[17] > 0.80 && pb_direct[21] > 0.80,
        "both tandem clusters must have p_bleach > 0.80: site1=({:.3},{:.3}), site2=({:.3},{:.3})",
        pb_direct[1],
        pb_direct[5],
        pb_direct[17],
        pb_direct[21]
    );
    assert!(
        pb_direct[12] < 0.20,
        "diamagnetic midpoint (ALA13) must NOT be falsely bleached by phantom centroid: got p_bleach={:.3}",
        pb_direct[12]
    );

    // 2. End-to-end assign_paramagnetic_backbone with explicit_ligand_indices + pdb_coordinates
    let priors = Priors::bmrb();
    let cfg = ParamagneticAssignConfig {
        explicit_ligand_indices: all_8_ligs.clone(),
        auto_detect_motifs: false,
        pdb_coordinates: Some(pdb.clone()),
        ..Default::default()
    };
    let (_, rep) = assign_paramagnetic_backbone(&[], &sequence, &priors, &cfg);
    assert_eq!(
        rep.detected_clusters.len(),
        2,
        "8 tandem ligands must partition into 2 3D donor clusters"
    );
    assert!(rep.p_bleach[1] > 0.80 && rep.p_bleach[17] > 0.80 && rep.p_bleach[12] < 0.20);

    // 3. Verify all entries of explicit_metal_coords[0..2] are preserved when supplied explicitly
    let cfg_two_metals = ParamagneticAssignConfig {
        explicit_ligand_indices: all_8_ligs,
        auto_detect_motifs: false,
        pdb_coordinates: Some(pdb),
        explicit_metal_coords: vec![[0.0, 0.0, 0.0], [26.0, 0.0, 0.0]],
        ..Default::default()
    };
    let (_, rep_two) = assign_paramagnetic_backbone(&[], &sequence, &priors, &cfg_two_metals);
    assert_eq!(rep_two.detected_clusters.len(), 2);
    assert!(
        rep_two.p_bleach[1] > 0.80 && rep_two.p_bleach[17] > 0.80 && rep_two.p_bleach[12] < 0.20
    );
}
