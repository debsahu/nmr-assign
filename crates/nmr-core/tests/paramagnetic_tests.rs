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
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::{parse_pdb_str_first_model, v_dist, Vec3};
use nmr_core::paramagnetic::{
    assign_paramagnetic_backbone, assign_paramagnetic_from_peaks,
    compute_bleaching_probabilities_3d, compute_bleaching_probabilities_3d_per_site,
    compute_bleaching_probabilities_from_pdb, compute_bleaching_probabilities_sequence,
    coordinating_histidine_indices, detect_metal_coordination_motifs, effective_pcs_sigma_ppm,
    filter_pcs_observations_blind_zone, fit_pcs_tensor_paramagnetic, fit_pcs_tensor_robust,
    fit_pcs_tensor_svd, fit_pcs_tensor_with_options, parse_metal_ligands,
    parse_paramagnetic_pdb_sites, pcs_gaussian_two_nll, predict_pcs_from_tensor,
    predict_pcs_spherical_ppm, refine_metal_position_and_tensor,
    refine_metal_position_and_tensor_with_options, rotation_matrix_from_zyz, MetalClusterSite,
    MetalMotifKind, MetalParamagneticClass, ParamagneticAssignConfig, ParamagneticSolverMode,
    PcsFitOptions, PcsObservation,
};
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::rng::Rng;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{AnnealConfig, ScoreConfig};
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
        metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
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
        metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
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
        let bytes = line.as_bytes();
        if line.starts_with("ATOM  ") && bytes.len() >= 54 && bytes[..54].is_ascii() {
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

    // Canonical lanthanide susceptibility anisotropy tensors in 10^-32 m^3
    // (Schmitz & Otting 2008 UTS ordering |zz'| >= |yy'| >= |xx'| => 0 <= dchi_rh / dchi_ax <= 2/3)
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let lanthanides = [
        ("Tm3+", 22.4_f64, 5.8_f64, [0.65_f64, 1.05, 2.15 + pi_2]),
        ("Tb3+", -34.2_f64, -8.6_f64, [1.20_f64, 0.82, 0.45 + pi_2]),
        ("Dy3+", -38.5_f64, -11.2_f64, [2.35_f64, 1.38, 1.70 + pi_2]),
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

        // Verify Schmitz/Otting UTS eigenvalue ordering |zz'| >= |yy'| >= |xx'|,
        // tracelessness, right-handedness, and 0 <= dchi_rh / dchi_ax <= 2/3
        let [xx_p, yy_p, zz_p] = fit_clean.eigenvalues_paf;
        assert!((xx_p + yy_p + zz_p).abs() < 1e-11, "PAF trace must vanish");
        assert!(
            zz_p.abs() >= yy_p.abs() - 1e-12
                && yy_p.abs() >= xx_p.abs() - 1e-12
                && fit_clean.dchi_rh * fit_clean.dchi_ax >= -1e-12,
            "{ion_name}: Schmitz/Otting UTS ordering violated: eig={:?}, ax={}, rh={}",
            fit_clean.eigenvalues_paf,
            fit_clean.dchi_ax,
            fit_clean.dchi_rh
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
    let (true_ax, true_rh, true_euler) = (-34.2_f64, -8.6_f64, [1.20_f64, 0.82, 0.45 + pi_2]);
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
    let huber_fit = fit_pcs_tensor_robust(&outlier_obs, metal_true, 1.65, 25).expect("Huber IRLS");
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
        fit_pcs_tensor_robust(&coord_unc_obs, metal_true, 1.65, 15).expect("robust + coord unc");
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

/// Count false-positive placements at sequence positions with `p_bleach[k] >= blind_zone_threshold`
/// lacking observable `(H, N)` shifts in `truth_ref` (computed strictly from `p_bleach` with zero
/// hardcoded residue-range slices).
fn count_blind_zone_false_positives(
    positions: &[i32],
    p_bleach: &[f64],
    blind_zone_threshold: f64,
    truth_ref: &Reference,
) -> usize {
    let mut fp = 0usize;
    for &pos in positions {
        if pos < 0 {
            continue;
        }
        let k = pos as usize;
        if p_bleach
            .get(k)
            .is_some_and(|&pb| pb >= blind_zone_threshold)
        {
            let has_truth_hn = truth_ref.shifts.contains_key(&(k, "H".to_string()))
                && truth_ref.shifts.contains_key(&(k, "N".to_string()));
            if !has_truth_hn {
                fp += 1;
            }
        }
    }
    fp
}

#[derive(Clone, Debug)]
struct AblationSummary {
    label: &'static str,
    mean_recall_pct: f64,
    mean_precision_pct: f64,
    mean_hc_prec_pct: f64,
    mean_n_assigned: f64,
    total_blind_zone_fp: usize,
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

    let full_p_bleach = compute_bleaching_probabilities_sequence(sequence.len(), &detected);
    let blind_thresh = ParamagneticAssignConfig::default().blind_zone_threshold;

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
    // (a) 4-Way 5-Seed Ablation Table for BOTH Standard & EvoGraph Solvers
    // -------------------------------------------------------------------------
    let ablation_specs: [(&str, bool, bool, bool); 4] = [
        ("1. Globular baseline", false, false, false),
        ("2. Generic CA-less filter only", false, true, false),
        ("3. u_para bleach prior only", true, false, true),
        ("4. Full Paramagnetic", true, true, true),
    ];

    for mode in [
        ParamagneticSolverMode::Standard,
        ParamagneticSolverMode::EvoGraph,
    ] {
        let mode_name = match mode {
            ParamagneticSolverMode::Standard => "Standard",
            ParamagneticSolverMode::EvoGraph => "EvoGraph",
        };
        let mut rows: Vec<AblationSummary> = Vec::new();
        for &(label, auto_motifs, ca_less_filter, u_para_prior) in &ablation_specs {
            let mut recs = Vec::new();
            let mut precs = Vec::new();
            let mut hcps = Vec::new();
            let mut n_ass = Vec::new();
            let mut fp_sum = 0usize;

            for seed in 0..5u64 {
                let cfg = ParamagneticAssignConfig {
                    mode,
                    auto_detect_motifs: auto_motifs,
                    enable_generic_ca_less_filter: ca_less_filter,
                    enable_u_para_bleach_prior: u_para_prior,
                    seed,
                    ..Default::default()
                };
                let (sys, res, rep) = match mode {
                    ParamagneticSolverMode::Standard => {
                        let (r, rp) =
                            assign_paramagnetic_backbone(&probe_systems, &sequence, &priors, &cfg);
                        (probe_systems.clone(), r, rp)
                    }
                    ParamagneticSolverMode::EvoGraph => {
                        assign_paramagnetic_from_peaks(&raw_peaks, &sequence, &priors, &cfg)
                    }
                };
                if auto_motifs {
                    assert_eq!(rep.detected_clusters.len(), 2);
                }
                fp_sum += count_blind_zone_false_positives(
                    &res.position,
                    &full_p_bleach,
                    blind_thresh,
                    &truth_ref,
                );
                let sys_truth = map_spin_systems_to_truth(&sys, &truth_ref);
                let mut m = score(&res, &sys_truth, 0.8);
                m.assignable = assignable_amides;
                recs.push(m.recall() * 100.0);
                precs.push(m.accuracy() * 100.0);
                hcps.push(m.hc_precision() * 100.0);
                n_ass.push(m.assigned as f64);
            }
            let summary = AblationSummary {
                label,
                mean_recall_pct: mean(&recs),
                mean_precision_pct: mean(&precs),
                mean_hc_prec_pct: mean(&hcps),
                mean_n_assigned: mean(&n_ass),
                total_blind_zone_fp: fp_sum,
            };
            println!(
                "CISD3 5-seed Ablation [{mode_name}] {:<32} | recall={:5.2}% | precision={:5.2}% (N_ass={:4.1}) | HC_prec={:6.2}% | p_bleach>=0.65_FP={}",
                summary.label,
                summary.mean_recall_pct,
                summary.mean_precision_pct,
                summary.mean_n_assigned,
                summary.mean_hc_prec_pct,
                summary.total_blind_zone_fp
            );
            rows.push(summary);
        }

        let glob = &rows[0];
        let u_only = &rows[2];
        let full = &rows[3];
        assert_eq!(
            u_only.total_blind_zone_fp, 0,
            "{mode_name}: u_para bleach prior must eliminate all p_bleach >= 0.65 false positives"
        );
        assert_eq!(
            full.total_blind_zone_fp, 0,
            "{mode_name}: Full Paramagnetic must emit ZERO p_bleach >= 0.65 false positives"
        );
        assert!(
            glob.total_blind_zone_fp > 0,
            "{mode_name}: Globular baseline expected > 0 blind-zone FPs on CISD3"
        );
        assert!(
            full.mean_hc_prec_pct >= 95.0,
            "{mode_name}: Full Paramagnetic HC precision fell below 95.0%: {:.2}%",
            full.mean_hc_prec_pct
        );
        assert!(
            full.mean_precision_pct >= glob.mean_precision_pct - 1e-9
                && full.mean_recall_pct >= glob.mean_recall_pct - 1e-9,
            "{mode_name}: Full Paramagnetic recall/precision ({:.2}%/{:.2}%) must preserve or beat Globular baseline ({:.2}%/{:.2}%)",
            full.mean_recall_pct,
            full.mean_precision_pct,
            glob.mean_recall_pct,
            glob.mean_precision_pct
        );
    }

    // Explicit --metal-ligands 5-seed EvoGraph parity check
    let mut explicit_evo_recs = Vec::new();
    let mut explicit_evo_precs = Vec::new();
    let mut explicit_evo_hcps = Vec::new();
    let mut explicit_evo_fps = 0usize;
    for seed in 0..5u64 {
        let exp_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            explicit_ligand_indices: explicit_ligs.clone(),
            auto_detect_motifs: false,
            seed,
            ..Default::default()
        };
        let (e_sys, e_res, _) =
            assign_paramagnetic_from_peaks(&raw_peaks, &sequence, &priors, &exp_cfg);
        explicit_evo_fps += count_blind_zone_false_positives(
            &e_res.position,
            &full_p_bleach,
            blind_thresh,
            &truth_ref,
        );
        let e_truth = map_spin_systems_to_truth(&e_sys, &truth_ref);
        let mut em = score(&e_res, &e_truth, 0.8);
        em.assignable = assignable_amides;
        explicit_evo_recs.push(em.recall() * 100.0);
        explicit_evo_precs.push(em.accuracy() * 100.0);
        explicit_evo_hcps.push(em.hc_precision() * 100.0);
    }
    println!(
        "CISD3 5-seed Explicit --metal-ligands [EvoGraph]: recall={:.2}% | precision={:.2}% | HC_prec={:.2}% | p_bleach>=0.65_FP={}",
        mean(&explicit_evo_recs),
        mean(&explicit_evo_precs),
        mean(&explicit_evo_hcps),
        explicit_evo_fps
    );
    assert_eq!(explicit_evo_fps, 0);
    assert!(mean(&explicit_evo_hcps) >= 95.0);

    // -------------------------------------------------------------------------
    // (b) Leave-One-Site-Out (LOSO) Cross-Site Validation on Tandem [2Fe-2S] CISD3
    //     (zero threshold retuning: fixed default blind_zone_threshold=0.65,
    //      hard_bleach_cutoff=0.85, bleach_prior_weight=2.2)
    // -------------------------------------------------------------------------
    let pb_site1 = compute_bleaching_probabilities_sequence(
        sequence.len(),
        std::slice::from_ref(&detected[0]),
    );
    let pb_site2 = compute_bleaching_probabilities_sequence(
        sequence.len(),
        std::slice::from_ref(&detected[1]),
    );

    for (fold_name, known_ligs, pb_known, pb_heldout) in [
        (
            "LOSO Fold 1 (Site 1 known, Site 2 held-out)",
            detected[0].ligand_indices.clone(),
            &pb_site1,
            &pb_site2,
        ),
        (
            "LOSO Fold 2 (Site 2 known, Site 1 held-out)",
            detected[1].ligand_indices.clone(),
            &pb_site2,
            &pb_site1,
        ),
    ] {
        let mut fp_known_only = 0usize;
        let mut fp_heldout_unmasked = 0usize;
        let mut recs_partial = Vec::new();
        let mut precs_partial = Vec::new();
        for seed in 0..5u64 {
            let partial_cfg = ParamagneticAssignConfig {
                mode: ParamagneticSolverMode::Standard,
                explicit_ligand_indices: known_ligs.clone(),
                auto_detect_motifs: false,
                seed,
                ..Default::default()
            };
            let (res_p, _) =
                assign_paramagnetic_backbone(&probe_systems, &sequence, &priors, &partial_cfg);
            fp_known_only += count_blind_zone_false_positives(
                &res_p.position,
                pb_known,
                blind_thresh,
                &truth_ref,
            );
            fp_heldout_unmasked += count_blind_zone_false_positives(
                &res_p.position,
                pb_heldout,
                blind_thresh,
                &truth_ref,
            );
            let std_truth = map_spin_systems_to_truth(&probe_systems, &truth_ref);
            let mut m = score(&res_p, &std_truth, 0.8);
            m.assignable = assignable_amides;
            recs_partial.push(m.recall() * 100.0);
            precs_partial.push(m.accuracy() * 100.0);
        }
        // Transfer motif detector + fixed hyperparameters onto the held-out site without retuning
        let mut fp_transferred = 0usize;
        for seed in 0..5u64 {
            let transfer_cfg = ParamagneticAssignConfig {
                mode: ParamagneticSolverMode::Standard,
                auto_detect_motifs: true,
                seed,
                ..Default::default()
            };
            let (res_t, _) =
                assign_paramagnetic_backbone(&probe_systems, &sequence, &priors, &transfer_cfg);
            fp_transferred += count_blind_zone_false_positives(
                &res_t.position,
                pb_heldout,
                blind_thresh,
                &truth_ref,
            );
        }
        println!(
            "CISD3 5-seed {fold_name}: partial recall={:.2}%/prec={:.2}% | known_site_FP={} | heldout_unmasked_FP={} -> heldout_transferred_FP={}",
            mean(&recs_partial),
            mean(&precs_partial),
            fp_known_only,
            fp_heldout_unmasked,
            fp_transferred
        );
        assert_eq!(
            fp_known_only, 0,
            "{fold_name}: supplied site must have ZERO blind-zone FPs with un-retuned thresholds"
        );
        assert_eq!(
            fp_transferred, 0,
            "{fold_name}: cross-site transferred motif prior must have ZERO blind-zone FPs on held-out site"
        );
    }
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
        metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
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

#[test]
fn test_p1_1_pcs_uts_sign_input_validation_rank_check_and_nan_q_factor() {
    use nmr_core::paramagnetic::pcs_q_is_better;

    let metal: Vec3 = [0.0, 0.0, 0.0];
    let sites: [Vec3; 8] = [
        [12.0, 3.0, -2.0],
        [-11.0, 5.5, 4.0],
        [4.0, -13.0, 3.5],
        [-3.5, 6.0, 14.0],
        [8.5, 9.0, -10.5],
        [-9.5, -8.0, -9.0],
        [14.5, -4.0, 7.5],
        [-6.0, 12.5, -8.5],
    ];

    // (a) Canonical Schmitz/Otting UTS (+30.0, +8.0) recovers dchi_rh = +8.0 (not -8.0)
    let euler = [0.55_f64, 1.15, 2.05];
    let valid_obs: Vec<PcsObservation> = sites
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let pcs = predict_pcs_spherical_ppm(pos, metal, 30.0, 8.0, euler);
            PcsObservation::new((i + 1) as i32, pos, pcs).with_uncertainty(0.03)
        })
        .collect();
    let t = fit_pcs_tensor_svd(&valid_obs, metal).expect("valid 8-site UTS fit");
    assert!(
        (t.dchi_ax - 30.0).abs() < 1e-9 && (t.dchi_rh - 8.0).abs() < 1e-9,
        "expected exact (+30.0, +8.0) UTS recovery, got ax={}, rh={}",
        t.dchi_ax,
        t.dchi_rh
    );

    // (b) NaN / Inf PCS or coordinates must fail cleanly
    let mut nan_obs = valid_obs.clone();
    nan_obs[2].pcs_ppm = f64::NAN;
    assert!(fit_pcs_tensor_svd(&nan_obs, metal).is_err());
    assert!(fit_pcs_tensor_robust(&nan_obs, metal, 1.65, 10).is_err());
    assert!(refine_metal_position_and_tensor(&nan_obs, metal, 2.0, 1.0).is_err());

    // (c) Non-positive or non-finite uncertainty_ppm / negative coord_uncertainty must fail cleanly
    let mut zero_sig = valid_obs.clone();
    zero_sig[0] = zero_sig[0].with_uncertainty(0.0);
    assert!(fit_pcs_tensor_svd(&zero_sig, metal).is_err());
    let mut neg_sig = valid_obs.clone();
    neg_sig[1] = neg_sig[1].with_uncertainty(-0.02);
    assert!(fit_pcs_tensor_svd(&neg_sig, metal).is_err());
    let mut neg_coord_sig = valid_obs.clone();
    neg_coord_sig[0] = neg_coord_sig[0].with_coord_uncertainty(-0.1);
    assert!(fit_pcs_tensor_svd(&neg_coord_sig, metal).is_err());

    // (d) Underdetermined n = 5 (0 residual DoF) must fail cleanly
    assert!(fit_pcs_tensor_svd(&valid_obs[..5], metal).is_err());
    assert!(refine_metal_position_and_tensor(&valid_obs[..7], metal, 2.0, 1.0).is_err());

    // (e) Rank-deficient collinear nuclear coordinates (cond > 1e8) must fail cleanly
    let collinear_obs: Vec<PcsObservation> = (0..7)
        .map(|i| {
            let r = 12.0 + f64::from(i) * 2.0;
            PcsObservation::new(i + 1, [r, 0.0, 0.0], 0.4 - 0.03 * f64::from(i))
                .with_uncertainty(0.03)
        })
        .collect();
    assert!(
        fit_pcs_tensor_svd(&collinear_obs, metal).is_err(),
        "collinear nuclear coordinates have rank-1 dipolar design matrix and must return Err"
    );

    // (f) Diamagnetic / all-zero PCS vector has undefined Q = 0/0 => q_factor.is_nan(),
    //     and pcs_q_is_better ranks any finite Q strictly ahead of NaN!
    let zero_pcs_obs: Vec<PcsObservation> = sites
        .iter()
        .enumerate()
        .map(|(i, &pos)| PcsObservation::new((i + 1) as i32, pos, 0.0).with_uncertainty(0.03))
        .collect();
    let zero_fit = fit_pcs_tensor_svd(&zero_pcs_obs, metal).expect("all-zero PCS SVD fit");
    assert!(
        zero_fit.q_factor.is_nan(),
        "expected NaN Q-factor when sum(pcs_obs^2) == 0, got {}",
        zero_fit.q_factor
    );
    assert!(zero_fit.rmsd_ppm.abs() < 1e-12);
    assert!(!pcs_q_is_better(f64::NAN, 0.25));
    assert!(pcs_q_is_better(0.25, f64::NAN));
    assert!(!pcs_q_is_better(f64::NAN, f64::NAN));
}

#[test]
fn test_metal_paramagnetic_class_per_site_r_bleach_and_diamagnetic_zero() {
    assert!(
        (MetalParamagneticClass::FeSClusterOrHighSpin3d.default_r_bleach_angstrom() - 8.5).abs()
            < 1e-12
    );
    assert!(
        (MetalParamagneticClass::IsotropicBroadener.default_r_bleach_angstrom() - 13.5).abs()
            < 1e-12
    );
    assert!(
        (MetalParamagneticClass::StrongLanthanide.default_r_bleach_angstrom() - 9.0).abs() < 1e-12
    );
    assert!(
        (MetalParamagneticClass::ModerateLanthanide.default_r_bleach_angstrom() - 5.5).abs()
            < 1e-12
    );
    assert_eq!(
        MetalParamagneticClass::Diamagnetic.default_r_bleach_angstrom(),
        0.0
    );

    // Mixed two-site metalloprotein: Ca2+ (Diamagnetic, 0.0 Å) at (0,0,0) + Yb3+ (ModerateLanthanide, 5.5 Å) at (30,0,0)
    let amides = vec![
        Some([3.0, 0.0, 0.0]),  // 3.0 Å from Ca2+ => must NOT be bleached!
        Some([24.5, 0.0, 0.0]), // 5.5 Å from Yb3+ => p_bleach == 0.50!
        Some([27.0, 0.0, 0.0]), // 3.0 Å from Yb3+ => strongly bleached!
    ];
    let p_mixed = compute_bleaching_probabilities_3d_per_site(
        &amides,
        &[([0.0, 0.0, 0.0], 0.0), ([30.0, 0.0, 0.0], 5.5)],
    );
    assert!(
        p_mixed[0] < 1e-4,
        "amide 3.0 Å from closed-shell Ca2+ must remain unbleached, got {}",
        p_mixed[0]
    );
    assert!((p_mixed[1] - 0.5).abs() < 1e-5);
    assert!(p_mixed[2] > 0.95);

    // Diamagnetic Zn2+ Cys2His2 zinc finger in BOTH sequence and 3D modes must yield p_bleach == 0.0 everywhere
    let zf_seq = seq_from_one_letter("MKPYKCPDCGKSFSQSSNLQKHQRTHTGEKP");
    let priors = Priors::bmrb();
    let cfg_zn = ParamagneticAssignConfig {
        metal_class: Some(MetalParamagneticClass::Diamagnetic),
        ..Default::default()
    };
    let (_, rep_zn) = assign_paramagnetic_backbone(&[], &zf_seq, &priors, &cfg_zn);
    assert_eq!(rep_zn.detected_clusters.len(), 1);
    assert!(
        rep_zn.p_bleach.iter().all(|&p| p == 0.0),
        "Diamagnetic Zn2+ finger must have p_bleach == 0.0 across all residues"
    );

    // Explicit r_bleach_angstrom override (e.g. Mn2+ 13.5 Å vs Some(0.0))
    let pdb_txt =
        fs::read_to_string(workspace_root().join("data/pdb_refs/1d3z.pdb")).expect("1d3z.pdb");
    let chain = parse_pdb_str_first_model(&pdb_txt).expect("parse 1d3z");
    let ub_seq: Vec<String> = chain.residues.iter().map(|r| r.res_name.clone()).collect();
    let m_pos = chain.residues[0].ca;
    let cfg_yb = ParamagneticAssignConfig {
        auto_detect_motifs: false,
        pdb_coordinates: Some(pdb_txt.clone()),
        explicit_metal_coords: vec![m_pos],
        metal_class: Some(MetalParamagneticClass::ModerateLanthanide),
        ..Default::default()
    };
    let cfg_mn = ParamagneticAssignConfig {
        metal_class: Some(MetalParamagneticClass::IsotropicBroadener),
        ..cfg_yb.clone()
    };
    let cfg_zero = ParamagneticAssignConfig {
        r_bleach_angstrom: Some(0.0),
        ..cfg_mn.clone()
    };
    let (_, rep_yb) = assign_paramagnetic_backbone(&[], &ub_seq, &priors, &cfg_yb);
    let (_, rep_mn) = assign_paramagnetic_backbone(&[], &ub_seq, &priors, &cfg_mn);
    let (_, rep_zero) = assign_paramagnetic_backbone(&[], &ub_seq, &priors, &cfg_zero);
    let sum_yb: f64 = rep_yb.p_bleach.iter().sum();
    let sum_mn: f64 = rep_mn.p_bleach.iter().sum();
    assert!(
        sum_mn > sum_yb * 2.5,
        "Mn2+/Gd3+ (13.5 Å) must bleach a much wider sphere than Yb3+ (5.5 Å): mn={sum_mn:.2}, yb={sum_yb:.2}"
    );
    assert!(rep_zero.p_bleach.iter().all(|&p| p == 0.0));
}

#[test]
fn test_huber_irls_standardized_residual_no_double_downweighting() {
    let metal: Vec3 = [0.0, 0.0, 0.0];
    let sites: [Vec3; 8] = [
        [11.5, 2.5, -1.5],
        [-10.5, 5.0, 3.5],
        [3.5, -12.5, 3.0],
        [-3.0, 5.5, 13.5],
        [8.0, 8.5, -10.0],
        [-9.0, -7.5, -8.5],
        [14.0, -3.5, 7.0],
        [-5.5, 12.0, -8.0],
    ];
    let euler = [0.48_f64, 1.08, 1.92];
    let mut obs: Vec<PcsObservation> = sites
        .iter()
        .enumerate()
        .map(|(i, &pos)| {
            let pcs = predict_pcs_spherical_ppm(pos, metal, 26.0, 6.5, euler);
            PcsObservation::new((i + 1) as i32, pos, pcs).with_uncertainty(0.03)
        })
        .collect();

    // Site 0 has large spectral uncertainty sigma = 0.30 ppm and residual +0.25 ppm:
    // standardized z_0 = 0.25 / 0.30 = 0.833 <= huber_k (1.65), so Huber weight MUST be 1.0
    // (exact parity with GLS SVD, zero double-downweighting!).
    obs[0] = obs[0].with_uncertainty(0.30);
    obs[0].pcs_ppm += 0.25;

    let gls_svd = fit_pcs_tensor_svd(&obs, metal).expect("GLS SVD");
    let huber = fit_pcs_tensor_robust(&obs, metal, 1.65, 25).expect("Standardized Huber IRLS");
    for j in 0..5 {
        assert!(
            (gls_svd.components_5[j] - huber.components_5[j]).abs() < 1e-11,
            "high-sigma inlier (z=0.833 <= 1.65) was falsely downweighted at component {j}: svd={} vs huber={}",
            gls_svd.components_5[j],
            huber.components_5[j]
        );
    }

    // Now inject a true low-sigma 25-sigma outlier on site 5 (sigma=0.03 ppm, residual +0.75 ppm, z=25.0 > 1.65)
    let mut with_outlier = obs;
    with_outlier[5].pcs_ppm += 0.75;
    let svd_bad = fit_pcs_tensor_svd(&with_outlier, metal).expect("contaminated SVD");
    let huber_good = fit_pcs_tensor_robust(&with_outlier, metal, 1.65, 25).expect("Huber outlier");
    assert!(
        (huber_good.dchi_ax - 26.0).abs() < (svd_bad.dchi_ax - 26.0).abs() * 0.35,
        "standardized Huber IRLS must suppress z=25 outlier: svd_ax={:.3}, huber_ax={:.3}",
        svd_bad.dchi_ax,
        huber_good.dchi_ax
    );
}

#[test]
fn test_pdb_hetatm_chain_mse_b_altloc_2x4fe4s_non_chaining_and_pcs_blind_zone_filter() {
    // 1. Hostile multi-byte UTF-8, NaN/Inf coords, HETATM MSE ('SE' donor), 'B'-only altlocs,
    //    and oxidation-state charge columns (CU 1+ vs CU 2+, CO 3+ vs CO 2+).
    let hostile_pdb = "\
ATOM      1 🔥🚀🔥 MET A   1       0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  N   ALA A   1         NaN   0.000   0.000  1.00  0.00           N\n\
HETATM   10  N  BMSE A   1       0.000   0.000   0.000  0.50 10.00           N\n\
HETATM   11  CA BMSE A   1       1.458   0.000   0.000  0.50 10.00           C\n\
HETATM   12  C  BMSE A   1       2.009   1.420   0.000  0.50 10.00           C\n\
HETATM   13  O  BMSE A   1       1.251   2.390   0.000  0.50 10.00           O\n\
HETATM   14  SE BMSE A   1       2.200  -1.800   0.500  0.50 10.00          SE\n\
ATOM     20  N  BCYS A   2       3.300   1.550   0.000  1.00 10.00           N\n\
ATOM     21  CA BCYS A   2       3.950   2.850   0.000  1.00 10.00           C\n\
ATOM     22  C  BCYS A   2       5.460   2.750   0.000  1.00 10.00           C\n\
ATOM     23  SG BCYS A   2       3.400   3.850   1.450  1.00 10.00           S\n\
HETATM   30 CU    CU A 101       3.500   4.100   3.550  1.00 10.00          CU2+\n\
HETATM   31 CU    CU A 102       2.200  -1.900   2.600  1.00 10.00          CU1+\n\
HETATM   32 CO    CO A 103       1.300   2.500   2.100  1.00 10.00          CO3+\n\
END\n";
    let seq_mc = vec!["MET".to_string(), "CYS".to_string()];
    let (chain_mc, sites_mc) =
        parse_paramagnetic_pdb_sites(hostile_pdb, &seq_mc).expect("parse hostile MSE/altLocB PDB");
    assert_eq!(chain_mc.residues.len(), 2);
    assert_eq!(chain_mc.residues[0].res_name, "MET"); // MSE normalized to MET
    assert_eq!(chain_mc.residues[1].res_name, "CYS"); // B-only altloc retained
    assert_eq!(sites_mc.len(), 3);
    assert_eq!(
        sites_mc[0].metal_class,
        MetalParamagneticClass::IsotropicBroadener
    ); // Cu2+
    assert_eq!(sites_mc[1].metal_class, MetalParamagneticClass::Diamagnetic); // Cu1+
    assert_eq!(sites_mc[2].metal_class, MetalParamagneticClass::Diamagnetic); // Co3+

    // 2. Real public 2x[4Fe-4S] bacterial ferredoxin 1BWE.pdb (~11.55 Å inter-cluster distance):
    //    must detect 2 separate clusters BOTH with SF4 HETATMs AND when HETATMs are stripped!
    let pdb_1bwe =
        fs::read_to_string(workspace_root().join("data/fixtures/ferredoxin_1bwe_2x4fe4s.pdb"))
            .expect("read ferredoxin_1bwe_2x4fe4s.pdb");
    let (chain_1bwe, het_sites_1bwe) =
        parse_paramagnetic_pdb_sites(&pdb_1bwe, &[]).expect("parse 1BWE");
    assert_eq!(chain_1bwe.residues.len(), 77);
    assert_eq!(
        het_sites_1bwe.len(),
        2,
        "1BWE must resolve into 2 distinct [4Fe-4S] SF4 clusters (not chained)"
    );
    let inter_sf4_dist = v_dist(het_sites_1bwe[0].centroid, het_sites_1bwe[1].centroid);
    assert!(
        (11.0..=12.2).contains(&inter_sf4_dist),
        "expected ~11.55 Å between 1BWE SF4 cubanes, got {inter_sf4_dist:.3} Å"
    );

    let seq_1bwe: Vec<String> = chain_1bwe
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let atom_only_1bwe: String = pdb_1bwe
        .lines()
        .filter(|l| !l.starts_with("HETATM"))
        .collect::<Vec<_>>()
        .join("\n");
    let priors = Priors::bmrb();
    let (_, rep_with_het) = assign_paramagnetic_backbone(
        &[],
        &seq_1bwe,
        &priors,
        &ParamagneticAssignConfig {
            pdb_coordinates: Some(pdb_1bwe),
            ..Default::default()
        },
    );
    let (_, rep_no_het) = assign_paramagnetic_backbone(
        &[],
        &seq_1bwe,
        &priors,
        &ParamagneticAssignConfig {
            pdb_coordinates: Some(atom_only_1bwe),
            ..Default::default()
        },
    );
    assert_eq!(rep_with_het.detected_clusters.len(), 2);
    assert_eq!(
        rep_no_het.detected_clusters.len(),
        2,
        "ATOM-only 1BWE must partition via true CYS SG donors into 2 clusters (not chain via 4.1 Å CB-CB)"
    );
    // Verify ligand-swapped 3D cubane centroids match X-ray SF4 cubanes within < 0.30 Å!
    for cl in &rep_no_het.detected_clusters {
        let mc = cl.metal_coord.expect("3D SG centroid");
        let min_err = het_sites_1bwe
            .iter()
            .map(|hs| v_dist(mc, hs.centroid))
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_err < 0.30,
            "1BWE SG-donor centroid deviated from X-ray SF4 cubane center by {min_err:.3} Å"
        );
    }

    // 3. Q4: Blind-zone & < 8.0 Å [Fe-S] Fermi-contact PCS observation auto-exclusion
    let m_fes = het_sites_1bwe[0].centroid;
    let mut pcs_obs = Vec::new();
    for r in &chain_1bwe.residues {
        let pos = r.h.unwrap_or(r.n);
        let dist = v_dist(pos, m_fes);
        if dist >= 10.0 {
            let clean_pcs = predict_pcs_spherical_ppm(pos, m_fes, 8.5, 2.2, [0.4, 0.9, 1.3]);
            pcs_obs.push(PcsObservation::new(r.seq_id, pos, clean_pcs).with_uncertainty(0.03));
        }
    }
    // Inject two hyperfine Fermi-contact shifted points inside r < 8.0 Å / p_bleach >= 0.85
    pcs_obs.push(
        PcsObservation::new(901, [m_fes[0] + 5.4, m_fes[1], m_fes[2]], 7.80).with_uncertainty(0.05),
    );
    pcs_obs.push(
        PcsObservation::new(902, [m_fes[0], m_fes[1] - 7.3, m_fes[2]], -5.60)
            .with_uncertainty(0.05),
    );
    let unfiltered_fit = fit_pcs_tensor_svd(&pcs_obs, m_fes).expect("unfiltered SVD");
    let (filtered_fit, retained) = fit_pcs_tensor_paramagnetic(
        &pcs_obs,
        m_fes,
        MetalParamagneticClass::FeSClusterOrHighSpin3d,
        true,
        None,
        0.85,
        Some(1.65),
    )
    .expect("blind-zone filtered PCS fit");
    assert_eq!(retained.len(), pcs_obs.len() - 2);
    assert!(
        unfiltered_fit.q_factor > 0.15
            && (unfiltered_fit.dchi_ax - 8.5).abs() > 0.5
            && filtered_fit.q_factor < 1e-12
            && (filtered_fit.dchi_ax - 8.5).abs() < 1e-9,
        "blind-zone/Fermi-contact filter must rescue PCS fit: unfiltered Q={:.4} (ax={:.3}), filtered Q={:.6e} (ax={:.6})",
        unfiltered_fit.q_factor,
        unfiltered_fit.dchi_ax,
        filtered_fit.q_factor,
        filtered_fit.dchi_ax
    );
}

#[test]
fn test_real_ecoli_2fe2s_ferredoxin_bmr19273_and_1i7h_multichain_validation() {
    let fixtures = workspace_root().join("data/fixtures");
    let ferr_ref = read_star(fixtures.join("bmr19273_ferredoxin_2fe2s.str"), None)
        .expect("read BMRB 19273 E. coli [2Fe-2S] ferredoxin");
    let pdb_1i7h =
        fs::read_to_string(fixtures.join("ferredoxin_1i7h_xray.pdb")).expect("read 1I7H.pdb");

    // 1. Multi-chain X-ray 1I7H has chains A, B, C with FES A 1112, FES B 2112, FES C 3112:
    //    must select ONLY target-chain FES A 1112!
    let (chain_a, sites_1i7h) =
        parse_paramagnetic_pdb_sites(&pdb_1i7h, &ferr_ref.sequence).expect("parse 1I7H");
    assert_eq!(
        sites_1i7h.len(),
        1,
        "must associate only chain A FES A 1112"
    );
    assert!(sites_1i7h[0].is_fes_cluster);
    assert_eq!(
        sites_1i7h[0].metal_class,
        MetalParamagneticClass::FeSClusterOrHighSpin3d
    );

    // Adversarial check: relabel FES B 2112 and FES C 3112 to chain 'A' -> donor distance check
    // (coordinated <= 3.8 Å to chains B/C, > 15 Å from chain A donors) MUST still reject them!
    let spoofed_1i7h: String = pdb_1i7h
        .lines()
        .map(|l| {
            if l.starts_with("HETATM") && l.contains("FES ") {
                let mut s = l.to_string();
                s.replace_range(21..22, "A");
                s
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (_, spoofed_sites) = parse_paramagnetic_pdb_sites(&spoofed_1i7h, &ferr_ref.sequence)
        .expect("parse spoofed 1I7H");
    assert_eq!(
        spoofed_sites.len(),
        1,
        "donor-distance coordination check must reject chain B/C FES clusters even when chain ID is spoofed"
    );

    // 2. Verify experimental BMRB 19273 missing-amide PRE blind zone aligns strongly with 3D Solomon p_bleach
    let priors = Priors::bmrb();
    let cfg_3d = ParamagneticAssignConfig {
        pdb_coordinates: Some(pdb_1i7h.clone()),
        ..Default::default()
    };
    let (_, rep_3d) = assign_paramagnetic_backbone(&[], &ferr_ref.sequence, &priors, &cfg_3d);
    assert_eq!(rep_3d.detected_clusters.len(), 1);

    let mut bleached_missing = 0usize;
    let mut bleached_total_non_pro = 0usize;
    for k in 1..ferr_ref.sequence.len() {
        if ferr_ref.sequence[k] == "PRO" {
            continue;
        }
        if rep_3d.p_bleach[k] >= cfg_3d.blind_zone_threshold {
            bleached_total_non_pro += 1;
            let has_hn = ferr_ref.shifts.contains_key(&(k, "H".to_string()))
                && ferr_ref.shifts.contains_key(&(k, "N".to_string()));
            if !has_hn {
                bleached_missing += 1;
            }
        }
    }
    let exp_bleach_frac = (bleached_missing as f64) / (bleached_total_non_pro.max(1) as f64);
    println!(
        "E. coli [2Fe-2S] Ferredoxin (BMRB 19273 + 1I7H): 3D blind-zone non-Pro residues={}, experimentally bleached in BMRB 19273={} ({:.1}%), chain_len={}",
        bleached_total_non_pro,
        bleached_missing,
        exp_bleach_frac * 100.0,
        chain_a.residues.len()
    );
    assert!(
        bleached_total_non_pro >= 12 && exp_bleach_frac >= 0.80,
        "expected >= 80% of 3D Solomon blind-zone amides to be experimentally bleached in BMRB 19273, got {bleached_missing}/{bleached_total_non_pro} ({:.1}%)",
        exp_bleach_frac * 100.0
    );

    // 3. Cross-protein 5-seed paramagnetic assignment validation (with fixed un-retuned CISD3 hyperparameters!)
    let assignable = (1..ferr_ref.sequence.len())
        .filter(|&k| {
            ferr_ref.sequence[k] != "PRO"
                && ferr_ref.shifts.contains_key(&(k, "H".to_string()))
                && ferr_ref.shifts.contains_key(&(k, "N".to_string()))
        })
        .count();
    let mut shifted_truth = ferr_ref.clone();
    let sim_base = SimConfig::realistic_bruker();
    for ((_, atom), val) in shifted_truth.shifts.iter_mut() {
        match atom.as_str() {
            "H" => *val += sim_base.ref_offset_h,
            "N" => *val += sim_base.ref_offset_n,
            "CA" => *val += sim_base.ref_offset_ca,
            "CB" => *val += sim_base.ref_offset_cb,
            "C" => *val += sim_base.ref_offset_c,
            _ => {}
        }
    }

    let mut glob_fps = 0usize;
    let mut para_fps = 0usize;
    let mut glob_precs = Vec::new();
    let mut para_precs = Vec::new();
    let mut para_recs = Vec::new();
    let mut para_hcps = Vec::new();
    for seed in 0..5u64 {
        let mut sc = SimConfig::realistic_bruker();
        sc.seed = seed;
        let (peaks, _) = simulate(&ferr_ref, &sc);

        let glob_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            auto_detect_motifs: false,
            enable_generic_ca_less_filter: false,
            enable_u_para_bleach_prior: false,
            seed,
            ..Default::default()
        };
        let (g_sys, g_res, _) =
            assign_paramagnetic_from_peaks(&peaks, &ferr_ref.sequence, &priors, &glob_cfg);
        glob_fps += count_blind_zone_false_positives(
            &g_res.position,
            &rep_3d.p_bleach,
            cfg_3d.blind_zone_threshold,
            &shifted_truth,
        );
        let g_truth = map_spin_systems_to_truth(&g_sys, &shifted_truth);
        let mut gm = score(&g_res, &g_truth, 0.8);
        gm.assignable = assignable;
        glob_precs.push(gm.accuracy() * 100.0);

        let p_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            pdb_coordinates: Some(pdb_1i7h.clone()),
            seed,
            ..Default::default()
        };
        let (p_sys, p_res, _) =
            assign_paramagnetic_from_peaks(&peaks, &ferr_ref.sequence, &priors, &p_cfg);
        para_fps += count_blind_zone_false_positives(
            &p_res.position,
            &rep_3d.p_bleach,
            cfg_3d.blind_zone_threshold,
            &shifted_truth,
        );
        let p_truth = map_spin_systems_to_truth(&p_sys, &shifted_truth);
        let mut pm = score(&p_res, &p_truth, 0.8);
        pm.assignable = assignable;
        para_precs.push(pm.accuracy() * 100.0);
        para_recs.push(pm.recall() * 100.0);
        para_hcps.push(pm.hc_precision() * 100.0);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);
    println!(
        "E. coli [2Fe-2S] Ferredoxin 5-seed EvoGraph (BMRB 19273 + 1I7H, un-retuned): glob_prec={:.2}% (blind_FP={}) -> para_rec={:.2}% | para_prec={:.2}% | HC_prec={:.2}% (blind_FP={})",
        mean(&glob_precs),
        glob_fps,
        mean(&para_recs),
        mean(&para_precs),
        mean(&para_hcps),
        para_fps
    );
    assert_eq!(
        para_fps, 0,
        "un-retuned paramagnetic solver must emit ZERO blind-zone FPs on E. coli [2Fe-2S] ferredoxin"
    );
    assert!(mean(&para_precs) >= mean(&glob_precs) && mean(&para_hcps) >= 92.0);
}

#[test]
fn test_real_calbindin_d9k_yb3_la3_experimental_pcs_bmr15594_bmr6699_1ksm() {
    let fixtures = workspace_root().join("data/fixtures");
    let yb_ref = read_star(fixtures.join("bmr15594_calbindin_yb3.str"), None)
        .expect("read BMRB 15594 Calbindin D9k Yb3+");
    let la_ref = read_star(fixtures.join("bmr6699_calbindin_la3.str"), None)
        .expect("read BMRB 6699 Calbindin D9k La3+");
    let pdb_1ksm =
        fs::read_to_string(fixtures.join("calbindin_1ksm_nmr.pdb")).expect("read 1KSM.pdb");

    let (chain_1ksm, sites_1ksm) =
        parse_paramagnetic_pdb_sites(&pdb_1ksm, &yb_ref.sequence).expect("parse 1KSM");
    assert_eq!(chain_1ksm.residues.len(), 75);
    assert!(
        !sites_1ksm.is_empty(),
        "1KSM must detect EF-hand lanthanide HETATM LA A 105"
    );
    // LA in 1KSM is closed-shell diamagnetic La3+ (r_bleach = 0.0 Å); substituting Yb3+ gives ModerateLanthanide (5.5 Å)
    assert_eq!(
        sites_1ksm[0].metal_class,
        MetalParamagneticClass::Diamagnetic
    );
    let la_metal_pos = sites_1ksm[0].centroid;

    // Extract real experimental amide 1H PCS = delta(Yb3+, BMRB 15594) - delta(La3+, BMRB 6699)
    let mut raw_pcs_obs: Vec<PcsObservation> = Vec::new();
    for res in &chain_1ksm.residues {
        if res.res_name == "PRO" {
            continue;
        }
        let k = (res.seq_id - 1) as usize;
        let (Some(&h_yb), Some(&h_la), Some(h_coord)) = (
            yb_ref.shifts.get(&(k, "H".to_string())),
            la_ref.shifts.get(&(k, "H".to_string())),
            res.h,
        ) else {
            continue;
        };
        let pcs_ppm = h_yb - h_la;
        raw_pcs_obs.push(
            PcsObservation::new(res.seq_id, h_coord, pcs_ppm)
                .with_uncertainty(0.04)
                .with_coord_uncertainty(0.25),
        );
    }
    assert_eq!(
        raw_pcs_obs.len(),
        51,
        "expected all 51 matched experimental Yb3+ - La3+ amide 1H PCSs in Calbindin D9k, got {}",
        raw_pcs_obs.len()
    );

    // 1. Full-dataset evaluation (all N = 51 experimental Yb3+ - La3+ amide 1H PCSs):
    //    Confirm that the default ModerateLanthanide Yb3+ blind-sphere gate (`r_bleach = 5.5 Å`,
    //    `p_bleach >= 0.85`) removes 0/51 points (`pre_filtered.len() == 51`), and evaluate both
    //    fixed-metal SVD (`Q ~ 0.297`) and 1.5 Å trust-radius VarPro (`Q ~ 0.286`) on all 51 points
    //    without hiding any inner-shell exclusions.
    let pre_filtered = filter_pcs_observations_blind_zone(
        &raw_pcs_obs,
        &[la_metal_pos],
        MetalParamagneticClass::ModerateLanthanide.default_r_bleach_angstrom(),
        false,
        0.85,
    );
    assert_eq!(
        pre_filtered.len(),
        51,
        "Yb3+ blind-sphere filter (r_bleach = 5.5 Å, p_bleach >= 0.85) must remove 0 of 51 experimental points"
    );

    let opts_off_svd = PcsFitOptions {
        fit_offset: true,
        huber_k: None,
        max_irls_iters: 25,
    };
    let opts_off_huber = PcsFitOptions {
        fit_offset: true,
        huber_k: Some(1.65),
        max_irls_iters: 25,
    };

    let full_svd = fit_pcs_tensor_svd(&pre_filtered, la_metal_pos).expect("Calbindin full-51 SVD");
    let full_svd_off = fit_pcs_tensor_with_options(&pre_filtered, la_metal_pos, &opts_off_svd)
        .expect("Calbindin full-51 SVD + offset");
    let full_huber = fit_pcs_tensor_robust(&pre_filtered, la_metal_pos, 1.65, 25)
        .expect("Calbindin full-51 Huber IRLS");
    let full_huber_off = fit_pcs_tensor_with_options(&pre_filtered, la_metal_pos, &opts_off_huber)
        .expect("Calbindin full-51 Huber IRLS + offset");
    let full_varpro_1p5 = refine_metal_position_and_tensor(&pre_filtered, la_metal_pos, 1.5, 0.5)
        .expect("Calbindin full-51 VarPro r_max=1.5 Å");
    let full_varpro_1p5_off = refine_metal_position_and_tensor_with_options(
        &pre_filtered,
        la_metal_pos,
        1.5,
        0.5,
        &opts_off_svd,
    )
    .expect("Calbindin full-51 VarPro + offset r_max=1.5 Å");
    let full_shift_1p5 = v_dist(full_varpro_1p5.metal_pos_angstrom, la_metal_pos);
    let full_shift_1p5_off = v_dist(full_varpro_1p5_off.metal_pos_angstrom, la_metal_pos);

    assert!(
        full_svd.q_factor < 0.35
            && full_svd_off.q_factor < 0.35
            && full_varpro_1p5.q_factor < 0.32
            && full_varpro_1p5_off.q_factor < 0.32
            && full_svd_off.pcs_offset_ppm.abs() < 0.25,
        "expected full-51 Calbindin Yb3+-La3+ SVD Q < 0.35 and VarPro(1.5 Å) Q < 0.32, got svd={:.4}, svd_off={:.4}, varpro={:.4}, varpro_off={:.4}",
        full_svd.q_factor,
        full_svd_off.q_factor,
        full_varpro_1p5.q_factor,
        full_varpro_1p5_off.q_factor
    );

    // 2. Outer-sphere dipolar subset (`r >= 10.8 Å`, N = 47, excluding inner-shell EF-hand II
    //    coordination residues `[25, 51, 67, 68]` where Yb3+ vs La3+ ionic-radius contraction
    //    0.86 Å vs 1.03 Å and local reorganization perturb amide geometry; Bertini et al. 2001):
    //    Note also the two-lab experimental caveat: paramagnetic Yb3+ shifts come from BMRB 15594
    //    (298 K) while diamagnetic La3+ shifts come from BMRB 6699 (300 K). Because of this slight
    //    inter-entry temperature/referencing offset + 1KSM crystal-vs-solution coordinates, an
    //    unconstrained `r_max = 5.0 Å` search drifts `~3.08 Å` out of the EF-hand II Ca2+/Ln3+
    //    site (`Q ~ 0.139`, inflated `dchi_ax ~ +12.4`), whereas the physical `r_max = 1.5 Å`
    //    trust radius pins at the boundary (`|dx_M| = 1.500 Å`) and preserves physical literature
    //    `dchi_ax ~ +8.9 x 10^-32 m^3` (`Q ~ 0.1578`).
    let mut excluded_inner_seq_ids: Vec<i32> = pre_filtered
        .iter()
        .filter(|o| v_dist(o.coord_angstrom, la_metal_pos) < 10.8)
        .map(|o| o.seq_id)
        .collect();
    excluded_inner_seq_ids.sort_unstable();
    assert_eq!(
        excluded_inner_seq_ids,
        vec![25, 51, 67, 68],
        "expected inner-shell r < 10.8 Å cut to exclude exactly EF-hand II residues [25, 51, 67, 68]"
    );

    let filtered_obs: Vec<PcsObservation> = pre_filtered
        .iter()
        .copied()
        .filter(|o| v_dist(o.coord_angstrom, la_metal_pos) >= 10.8)
        .collect();
    assert_eq!(filtered_obs.len(), 47);

    let svd_at_la = fit_pcs_tensor_svd(&filtered_obs, la_metal_pos).expect("Calbindin SVD at LA");
    let svd_at_la_off = fit_pcs_tensor_with_options(&filtered_obs, la_metal_pos, &opts_off_svd)
        .expect("Calbindin SVD + offset at LA");
    let huber_at_la =
        fit_pcs_tensor_robust(&filtered_obs, la_metal_pos, 1.65, 25).expect("Calbindin Huber IRLS");
    let huber_at_la_off = fit_pcs_tensor_with_options(&filtered_obs, la_metal_pos, &opts_off_huber)
        .expect("Calbindin Huber IRLS + offset");
    let zero_rad = refine_metal_position_and_tensor(&filtered_obs, la_metal_pos, 0.0, 0.5)
        .expect("Calbindin zero-radius VarPro must pin initial metal position");
    assert!(
        v_dist(zero_rad.metal_pos_angstrom, la_metal_pos) <= 1e-12
            && (zero_rad.dchi_ax - svd_at_la.dchi_ax).abs() <= 1e-12,
        "search_radius_angstrom=0.0 must keep metal_pos exact (got shift={:.6} Å)",
        v_dist(zero_rad.metal_pos_angstrom, la_metal_pos)
    );
    assert!(
        refine_metal_position_and_tensor(&filtered_obs, la_metal_pos, -0.5, 0.5).is_err(),
        "negative search_radius_angstrom must return Err"
    );
    let refined_yb = refine_metal_position_and_tensor(&filtered_obs, la_metal_pos, 1.5, 0.5)
        .expect("Calbindin VarPro metal + DeltaChi refinement (r_max=1.5 Å)");
    let refined_yb_off = refine_metal_position_and_tensor_with_options(
        &filtered_obs,
        la_metal_pos,
        1.5,
        0.5,
        &opts_off_svd,
    )
    .expect("Calbindin VarPro + offset refinement (r_max=1.5 Å)");
    let metal_shift_ang = v_dist(refined_yb.metal_pos_angstrom, la_metal_pos);
    let metal_shift_ang_off = v_dist(refined_yb_off.metal_pos_angstrom, la_metal_pos);

    let unconstrained_yb = refine_metal_position_and_tensor(&filtered_obs, la_metal_pos, 5.0, 0.5)
        .expect("Calbindin unconstrained VarPro diagnostic (r_max=5.0 Å)");
    let unconstrained_yb_off = refine_metal_position_and_tensor_with_options(
        &filtered_obs,
        la_metal_pos,
        5.0,
        0.5,
        &opts_off_svd,
    )
    .expect("Calbindin unconstrained VarPro + offset diagnostic (r_max=5.0 Å)");
    let unconstrained_shift_ang = v_dist(unconstrained_yb.metal_pos_angstrom, la_metal_pos);
    let unconstrained_shift_ang_off = v_dist(unconstrained_yb_off.metal_pos_angstrom, la_metal_pos);

    println!(
        "Calbindin D9k Real Experimental Yb3+-La3+ 1H PCS (BMRB 15594 [298 K] - BMRB 6699 [300 K] vs 1KSM):\n  [Full Unfiltered Dataset N={} (r_bleach=5.5 Å removes 0 pts)]:\n    SVD@LA105 (off=0):       dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    SVD@LA105 (+fit_off):    dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    Huber@LA105 (off=0):     dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    Huber@LA105 (+fit_off):  dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    VarPro<=1.5Å (off=0):    dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å\n    VarPro<=1.5Å (+fit_off): dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å\n  [Outer-Sphere r>=10.8 Å Subset N={} (excludes inner-shell seq_ids {:?})]:\n    SVD@LA105 (off=0):       dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    SVD@LA105 (+fit_off):    dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    Huber@LA105 (off=0):     dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    Huber@LA105 (+fit_off):  dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm\n    VarPro<=1.5Å (off=0):    dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å\n    VarPro<=1.5Å (+fit_off): dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å\n    VarPro<=5.0Å (off=0):    dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å\n    VarPro<=5.0Å (+fit_off): dchi_ax={:+.3}, dchi_rh={:+.3}, off={:+.4}, Q={:.4}, RMSD={:.4} ppm, |dx_M|={:.3} Å",
        pre_filtered.len(),
        full_svd.dchi_ax,
        full_svd.dchi_rh,
        full_svd.pcs_offset_ppm,
        full_svd.q_factor,
        full_svd.rmsd_ppm,
        full_svd_off.dchi_ax,
        full_svd_off.dchi_rh,
        full_svd_off.pcs_offset_ppm,
        full_svd_off.q_factor,
        full_svd_off.rmsd_ppm,
        full_huber.dchi_ax,
        full_huber.dchi_rh,
        full_huber.pcs_offset_ppm,
        full_huber.q_factor,
        full_huber.rmsd_ppm,
        full_huber_off.dchi_ax,
        full_huber_off.dchi_rh,
        full_huber_off.pcs_offset_ppm,
        full_huber_off.q_factor,
        full_huber_off.rmsd_ppm,
        full_varpro_1p5.dchi_ax,
        full_varpro_1p5.dchi_rh,
        full_varpro_1p5.pcs_offset_ppm,
        full_varpro_1p5.q_factor,
        full_varpro_1p5.rmsd_ppm,
        full_shift_1p5,
        full_varpro_1p5_off.dchi_ax,
        full_varpro_1p5_off.dchi_rh,
        full_varpro_1p5_off.pcs_offset_ppm,
        full_varpro_1p5_off.q_factor,
        full_varpro_1p5_off.rmsd_ppm,
        full_shift_1p5_off,
        filtered_obs.len(),
        excluded_inner_seq_ids,
        svd_at_la.dchi_ax,
        svd_at_la.dchi_rh,
        svd_at_la.pcs_offset_ppm,
        svd_at_la.q_factor,
        svd_at_la.rmsd_ppm,
        svd_at_la_off.dchi_ax,
        svd_at_la_off.dchi_rh,
        svd_at_la_off.pcs_offset_ppm,
        svd_at_la_off.q_factor,
        svd_at_la_off.rmsd_ppm,
        huber_at_la.dchi_ax,
        huber_at_la.dchi_rh,
        huber_at_la.pcs_offset_ppm,
        huber_at_la.q_factor,
        huber_at_la.rmsd_ppm,
        huber_at_la_off.dchi_ax,
        huber_at_la_off.dchi_rh,
        huber_at_la_off.pcs_offset_ppm,
        huber_at_la_off.q_factor,
        huber_at_la_off.rmsd_ppm,
        refined_yb.dchi_ax,
        refined_yb.dchi_rh,
        refined_yb.pcs_offset_ppm,
        refined_yb.q_factor,
        refined_yb.rmsd_ppm,
        metal_shift_ang,
        refined_yb_off.dchi_ax,
        refined_yb_off.dchi_rh,
        refined_yb_off.pcs_offset_ppm,
        refined_yb_off.q_factor,
        refined_yb_off.rmsd_ppm,
        metal_shift_ang_off,
        unconstrained_yb.dchi_ax,
        unconstrained_yb.dchi_rh,
        unconstrained_yb.pcs_offset_ppm,
        unconstrained_yb.q_factor,
        unconstrained_yb.rmsd_ppm,
        unconstrained_shift_ang,
        unconstrained_yb_off.dchi_ax,
        unconstrained_yb_off.dchi_rh,
        unconstrained_yb_off.pcs_offset_ppm,
        unconstrained_yb_off.q_factor,
        unconstrained_yb_off.rmsd_ppm,
        unconstrained_shift_ang_off,
    );

    assert!(
        svd_at_la.q_factor < 0.25
            && svd_at_la_off.q_factor < 0.25
            && refined_yb.q_factor < 0.25
            && refined_yb_off.q_factor < 0.25
            && huber_at_la.q_factor < 0.25
            && huber_at_la_off.q_factor < 0.25,
        "real experimental Calbindin D9k outer-sphere Yb3+ PCS Q-factor exceeded 0.25: svd Q={:.4}, svd_off Q={:.4}, refined Q={:.4}, huber Q={:.4}",
        svd_at_la.q_factor,
        svd_at_la_off.q_factor,
        refined_yb.q_factor,
        huber_at_la.q_factor
    );
    assert!(
        unconstrained_yb.q_factor < refined_yb.q_factor && unconstrained_shift_ang > 2.0,
        "expected unconstrained r_max=5.0 Å search to drift > 2.0 Å (got |dx_M|={unconstrained_shift_ang:.3} Å, Q={:.4})",
        unconstrained_yb.q_factor
    );
    assert!(
        svd_at_la.dchi_ax > 7.5
            && svd_at_la.dchi_ax < 10.5
            && refined_yb.dchi_ax > 7.0
            && refined_yb.dchi_ax < 11.5
            && metal_shift_ang <= 1.5001
            && metal_shift_ang_off <= 1.5001,
        "expected literature Yb3+ dchi_ax ~ +8.5..9.0e-32 m^3 and strictly bounded |d_metal| <= 1.5001 Å (got svd_ax={:.3}, varpro_ax={:.3}, shift={metal_shift_ang:.4} Å)",
        svd_at_la.dchi_ax,
        refined_yb.dchi_ax
    );
}

#[test]
fn test_varpro_gaussian_two_nll_penalizes_variance_inflation_and_fits_pcs_offset() {
    let true_metal = [1.5_f64, -2.0, 0.8];
    let true_ax = 24.0_f64;
    let true_rh = 6.2_f64;
    let true_euler = [0.65_f64, 1.05, 2.15];
    let true_offset = 0.185_f64;

    let mut rng = Rng::new(0x516A_2411_u64);
    let mut coords = Vec::with_capacity(28);
    while coords.len() < 28 {
        let c = [
            true_metal[0] + rng.uniform(-18.0, 18.0),
            true_metal[1] + rng.uniform(-18.0, 18.0),
            true_metal[2] + rng.uniform(-18.0, 18.0),
        ];
        let r = v_dist(c, true_metal);
        if (9.5..=22.0).contains(&r) {
            coords.push(c);
        }
    }

    // 1. Prove `pcs_gaussian_two_nll` strictly increases (worsens) when `sigma_eff` is artificially
    //    inflated at small/moderate residuals, whereas bare unweighted `sum (r_i / sigma_eff_i)^2`
    //    spuriously decreases!
    let mut obs_tight = Vec::new();
    let mut obs_inflated = Vec::new();
    for (idx, &c) in coords.iter().enumerate() {
        let exact = predict_pcs_spherical_ppm(c, true_metal, true_ax, true_rh, true_euler);
        let noisy = exact + rng.normal(0.0, 0.035);
        obs_tight.push(
            PcsObservation::new((idx + 1) as i32, c, noisy)
                .with_uncertainty(0.04)
                .with_coord_uncertainty(0.15),
        );
        obs_inflated.push(
            PcsObservation::new((idx + 1) as i32, c, noisy)
                .with_uncertainty(0.35)
                .with_coord_uncertainty(0.95),
        );
    }
    let t_tight = fit_pcs_tensor_svd(&obs_tight, true_metal).expect("tight SVD");
    let bare_chi2 = |obs_slice: &[PcsObservation]| -> f64 {
        obs_slice
            .iter()
            .map(|o| {
                let pred = predict_pcs_from_tensor(o.coord_angstrom, &t_tight);
                let sig =
                    effective_pcs_sigma_ppm(o, t_tight.metal_pos_angstrom, &t_tight.components_5);
                ((o.pcs_ppm - pred) / sig).powi(2)
            })
            .sum()
    };
    let chi2_tight = bare_chi2(&obs_tight);
    let chi2_inflated = bare_chi2(&obs_inflated);
    let nll_tight = pcs_gaussian_two_nll(&obs_tight, &t_tight);
    let nll_inflated = pcs_gaussian_two_nll(&obs_inflated, &t_tight);
    assert!(
        chi2_inflated < chi2_tight,
        "bare chi2 shrinks under sigma_eff inflation ({chi2_inflated:.3} vs {chi2_tight:.3})"
    );
    assert!(
        nll_inflated > nll_tight + 50.0,
        "Gaussian 2*NLL must strictly penalize inflated sigma_eff via +2*ln(sigma_eff) ({nll_inflated:.3} vs {nll_tight:.3})"
    );

    // Also place an uncertain coordinate near a decoy site (`r ~ 2.3 Å` from decoy, `11.5 Å` from true_metal)
    // and verify VarPro with `2*NLL` recovers `true_metal` (< 0.15 Å) instead of collapsing onto the decoy!
    let mut obs_with_probe = obs_tight.clone();
    let probe_coord = [
        true_metal[0] + 7.5,
        true_metal[1] + 6.5,
        true_metal[2] - 5.5,
    ];
    let probe_pcs =
        predict_pcs_spherical_ppm(probe_coord, true_metal, true_ax, true_rh, true_euler);
    obs_with_probe.push(
        PcsObservation::new(99, probe_coord, probe_pcs)
            .with_uncertainty(0.04)
            .with_coord_uncertainty(0.65),
    );
    let decoy_init = [probe_coord[0] - 2.3, probe_coord[1], probe_coord[2]];
    let search_r = v_dist(decoy_init, true_metal) + 1.2;
    let recovered = refine_metal_position_and_tensor(&obs_with_probe, decoy_init, search_r, 1.0)
        .expect("VarPro 2*NLL recovery past high-coord-uncertainty probe");
    assert!(
        v_dist(recovered.metal_pos_angstrom, true_metal) < 0.20,
        "VarPro 2*NLL must converge to true_metal rather than decoy gradient-inflation trap (err={:.4} Å)",
        v_dist(recovered.metal_pos_angstrom, true_metal)
    );

    // 2. Verify `PcsFitOptions { fit_offset: true }` recovers exact constant referencing offset
    //    in both 6-param fixed-metal SVD, Huber IRLS, and 9-param VarPro (and enforces DoF bounds N>=7 / N>=9)
    let obs_shifted: Vec<PcsObservation> = coords
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let pcs = predict_pcs_spherical_ppm(c, true_metal, true_ax, true_rh, true_euler)
                + true_offset;
            PcsObservation::new((i + 1) as i32, c, pcs).with_uncertainty(0.03)
        })
        .collect();

    assert!(
        fit_pcs_tensor_with_options(
            &obs_shifted[..6],
            true_metal,
            &PcsFitOptions {
                fit_offset: true,
                ..Default::default()
            }
        )
        .is_err(),
        "6-param fixed-metal PCS fit + offset must reject N=6 (< 7)"
    );
    assert!(
        refine_metal_position_and_tensor_with_options(
            &obs_shifted[..8],
            true_metal,
            1.5,
            0.5,
            &PcsFitOptions {
                fit_offset: true,
                ..Default::default()
            }
        )
        .is_err(),
        "9-param VarPro PCS fit + offset must reject N=8 (< 9)"
    );

    let fit_no_off = fit_pcs_tensor_svd(&obs_shifted, true_metal).expect("shifted without offset");
    let fit_with_off = fit_pcs_tensor_with_options(
        &obs_shifted,
        true_metal,
        &PcsFitOptions {
            fit_offset: true,
            ..Default::default()
        },
    )
    .expect("shifted with offset");
    assert!(
        fit_no_off.q_factor > 0.10 && fit_with_off.q_factor < 1e-12,
        "expected un-corrected offset Q > 0.10 and 6-param SVD Q < 1e-12 (got {:.4} vs {:.3e})",
        fit_no_off.q_factor,
        fit_with_off.q_factor
    );
    assert!(
        (fit_with_off.pcs_offset_ppm - true_offset).abs() < 1e-11
            && (fit_with_off.dchi_ax - true_ax).abs() < 1e-10
            && (fit_with_off.dchi_rh - true_rh).abs() < 1e-10,
        "exact 6-param SVD offset/tensor recovery mismatch: off={:.6}, ax={:.6}, rh={:.6}",
        fit_with_off.pcs_offset_ppm,
        fit_with_off.dchi_ax,
        fit_with_off.dchi_rh
    );

    let init_guess = [
        true_metal[0] + 1.1,
        true_metal[1] - 0.9,
        true_metal[2] + 0.8,
    ];
    let vp_off = refine_metal_position_and_tensor_with_options(
        &obs_shifted,
        init_guess,
        2.5,
        0.6,
        &PcsFitOptions {
            fit_offset: true,
            ..Default::default()
        },
    )
    .expect("9-param VarPro with offset");
    assert!(
        v_dist(vp_off.metal_pos_angstrom, true_metal) < 1e-4
            && (vp_off.pcs_offset_ppm - true_offset).abs() < 1e-4
            && (vp_off.dchi_ax - true_ax).abs() / true_ax < 1e-4,
        "9-param VarPro recovery mismatch: pos_err={:.6} Å, off={:.6}, ax={:.4}",
        v_dist(vp_off.metal_pos_angstrom, true_metal),
        vp_off.pcs_offset_ppm,
        vp_off.dchi_ax
    );
}

#[test]
fn test_volumetric_solomon_pre_attenuated_2d_3d_picked_paramagnetic_benchmark() {
    let fixtures = workspace_root().join("data/fixtures");
    let ferr_bmrb = read_star(fixtures.join("bmr19273_ferredoxin_2fe2s.str"), None)
        .expect("read BMRB 19273 ferredoxin");
    let pdb_1i7h =
        fs::read_to_string(fixtures.join("ferredoxin_1i7h_xray.pdb")).expect("read 1I7H.pdb");
    let priors = Priors::bmrb();

    let (chain_a, sites) =
        parse_paramagnetic_pdb_sites(&pdb_1i7h, &ferr_bmrb.sequence).expect("parse 1I7H");
    assert_eq!(sites.len(), 1);
    let metal_center = sites[0].centroid;

    // 1. Build a COMPLETE pre-PRE shift table covering ALL 110 residues (including the 16
    //    inner-sphere blind-zone residues absent from BMRB 19273!) by filling any unrecorded
    //    shifts from `Priors::bmrb()` + deterministic scatter BEFORE physical PRE volume rendering.
    //    Zero blind-zone residues are pre-deleted from the input shift table!
    let mut full_ref = ferr_bmrb.clone();
    let mut rng_fill = Rng::new(0xFE25_1174_u64);
    for k in 0..full_ref.sequence.len() {
        let res_name = &full_ref.sequence[k];
        for &atom in &["H", "N", "CA", "CB", "C"] {
            if (res_name == "PRO" && atom == "H") || (res_name == "GLY" && atom == "CB") {
                continue;
            }
            let key = (k, atom.to_string());
            if let std::collections::btree_map::Entry::Vacant(e) = full_ref.shifts.entry(key) {
                if let Some((mu, sd)) = priors.get(res_name, atom) {
                    e.insert(mu + rng_fill.normal(0.0, 0.35 * sd));
                }
            }
        }
    }

    let total_non_pro_amides = (1..full_ref.sequence.len())
        .filter(|&k| {
            full_ref.sequence[k] != "PRO"
                && full_ref.shifts.contains_key(&(k, "H".to_string()))
                && full_ref.shifts.contains_key(&(k, "N".to_string()))
        })
        .count();
    assert!(
        total_non_pro_amides >= 100,
        "complete pre-bleach ferredoxin shift table must contain all >=100 non-Pro amides (got {total_non_pro_amides})"
    );

    // Compute exact 3D amide-to-metal distance `r_M[k]` (Å) for every residue in 1I7H
    let mut r_metal_ang = vec![25.0_f64; full_ref.sequence.len()];
    for res in &chain_a.residues {
        let k = (res.seq_id - 1) as usize;
        if k < r_metal_ang.len() {
            let coord = res.h.unwrap_or(res.n);
            r_metal_ang[k] = v_dist(coord, metal_center);
        }
    }

    let cfg_3d = ParamagneticAssignConfig {
        pdb_coordinates: Some(pdb_1i7h.clone()),
        ..Default::default()
    };
    let (_, rep_3d) = assign_paramagnetic_backbone(&[], &full_ref.sequence, &priors, &cfg_3d);

    // Physical Solomon-Bloembergen transverse PRE relaxation rate & INEPT amplitude transfer:
    //   R2_PRE(r_M) = R2_ref * (r_bleach / r_M)^6
    //   I(r_M) / I_0 = exp(-R2_PRE * tau_INEPT) * R2_0 / (R2_0 + R2_PRE)
    // (`Solomon, Phys. Rev. 1955, 99, 559`; `Battiste & Wagner, Biochemistry 2000, 39, 5355`)
    let r_bleach = MetalParamagneticClass::FeSClusterOrHighSpin3d.default_r_bleach_angstrom(); // 8.5 Å
    let r2_0 = 20.0_f64; // s^-1 diamagnetic transverse relaxation
    let r2_pre_at_bleach = 95.0_f64; // s^-1 at r_M = 8.5 Å
    let tau_2d_s = 0.0108_f64; // 10.8 ms 2D 15N-HSQC INEPT + reverse-INEPT
    let tau_3d_s = 0.0250_f64; // 25.0 ms 3D out-and-back triple-resonance transfer

    let pre_atten = |r_m: f64, tau_s: f64| -> f64 {
        let r_eff = r_m.max(2.0);
        let r2_pre = r2_pre_at_bleach * (r_bleach / r_eff).powi(6);
        (-r2_pre * tau_s).exp() * (r2_0 / (r2_0 + r2_pre))
    };

    // Verify physical regime separation before rendering
    assert!(
        pre_atten(7.2, tau_2d_s) * 58.0 < 2.0,
        "core blind-zone amide (r_M = 7.2 Å) must physically decay below 2.0x thermal noise"
    );
    assert!(
        pre_atten(9.2, tau_2d_s) * 58.0 > 6.5 && pre_atten(9.2, tau_3d_s) * 28.0 < 4.8,
        "twilight-zone amide (r_M = 9.2 Å) must survive 2D HSQC while attenuating in 3D triple-resonance"
    );

    let pcfg = PickerConfig {
        snr_threshold: 5.5,
        ..PickerConfig::default()
    };

    let mut glob_fps = 0usize;
    let mut para_fps = 0usize;
    let mut glob_precs = Vec::new();
    let mut para_precs = Vec::new();
    let mut para_hcps = Vec::new();
    let mut picked_core_blind = 0usize;
    let mut picked_twilight = 0usize;
    let mut picked_outer = 0usize;

    for seed in 0..3u64 {
        // Synthesize full backbone peak table for ALL residues (including inner-sphere blind zone!)
        let mut all_peaks: Vec<Peak> = Vec::new();
        for (k, &r_ang) in r_metal_ang.iter().enumerate().skip(1) {
            if full_ref.sequence[k] == "PRO" {
                continue;
            }
            let h = full_ref.shifts[&(k, "H".to_string())];
            let n = full_ref.shifts[&(k, "N".to_string())];
            let f2d = pre_atten(r_ang, tau_2d_s);
            let f3d = pre_atten(r_ang, tau_3d_s);

            let snr_hsqc = 58.0 * f2d;
            all_peaks.push(Peak::hsqc(h, n).with_intensity(snr_hsqc, Some(snr_hsqc)));

            if let Some(&ca_i) = full_ref.shifts.get(&(k, "CA".to_string())) {
                all_peaks.push(
                    Peak::triple("HNCA", h, n, ca_i, 1)
                        .with_intensity(34.0 * f3d, Some(34.0 * f3d)),
                );
                all_peaks.push(
                    Peak::triple("HNCACB", h, n, ca_i, 1)
                        .with_intensity(28.0 * f3d, Some(28.0 * f3d)),
                );
            }
            if let Some(&cb_i) = full_ref.shifts.get(&(k, "CB".to_string())) {
                all_peaks.push(
                    Peak::triple("HNCACB", h, n, cb_i, -1)
                        .with_intensity(25.0 * f3d, Some(25.0 * f3d)),
                );
            }
            if let Some(&ca_im1) = full_ref.shifts.get(&(k - 1, "CA".to_string())) {
                all_peaks.push(
                    Peak::triple("HNCA", h, n, ca_im1, 1)
                        .with_intensity(20.0 * f3d, Some(20.0 * f3d)),
                );
                all_peaks.push(
                    Peak::triple("CBCACONH", h, n, ca_im1, 1)
                        .with_intensity(32.0 * f3d, Some(32.0 * f3d)),
                );
                all_peaks.push(
                    Peak::triple("HNCACB", h, n, ca_im1, 1)
                        .with_intensity(16.0 * f3d, Some(16.0 * f3d)),
                );
            }
            if let Some(&cb_im1) = full_ref.shifts.get(&(k - 1, "CB".to_string())) {
                all_peaks.push(
                    Peak::triple("CBCACONH", h, n, cb_im1, 1)
                        .with_intensity(30.0 * f3d, Some(30.0 * f3d)),
                );
                all_peaks.push(
                    Peak::triple("HNCACB", h, n, cb_im1, -1)
                        .with_intensity(15.0 * f3d, Some(15.0 * f3d)),
                );
            }
            if let Some(&c_im1) = full_ref.shifts.get(&(k - 1, "C".to_string())) {
                all_peaks.push(
                    Peak::triple("HNCO", h, n, c_im1, 1)
                        .with_intensity(38.0 * f3d, Some(38.0 * f3d)),
                );
            }
        }

        // Render genuine noisy 2D HSQC (including Asn/Gln sidechain NH2 doublets lacking 3D
        // backbone carbon relays) + four 3D triple-resonance volumes (`noise_level = 1.0`)
        let mut hsqc_render_peaks = all_peaks.clone();
        for &(nh_a, nh_b, n_sc) in &[
            (6.92_f64, 7.56_f64, 112.1_f64),
            (7.04, 7.64, 112.8),
            (6.86, 7.48, 111.6),
            (7.12, 7.71, 113.4),
        ] {
            hsqc_render_peaks.push(Peak::hsqc(nh_a, n_sc).with_intensity(26.0, Some(26.0)));
            hsqc_render_peaks.push(Peak::hsqc(nh_b, n_sc).with_intensity(26.0, Some(26.0)));
        }
        let rcfg_base = SpectrumRenderConfig {
            nx_h: 512,
            ny_n: 256,
            nz_c: 128,
            noise_level: 1.0,
            seed: 4200 + seed * 17,
            ..SpectrumRenderConfig::default()
        };
        let hsqc_vol = render_hsqc_2d(&hsqc_render_peaks, &rcfg_base);
        let picked_roots = pick_hsqc_2d(&hsqc_vol, &pcfg);

        if seed == 0 {
            for r in &picked_roots {
                if let Some(best_k) = (1..full_ref.sequence.len())
                    .filter(|&k| full_ref.sequence[k] != "PRO")
                    .min_by(|&a, &b| {
                        let da = ((r.h - full_ref.shifts[&(a, "H".to_string())]) / 0.03).powi(2)
                            + ((r.n - full_ref.shifts[&(a, "N".to_string())]) / 0.25).powi(2);
                        let db = ((r.h - full_ref.shifts[&(b, "H".to_string())]) / 0.03).powi(2)
                            + ((r.n - full_ref.shifts[&(b, "N".to_string())]) / 0.25).powi(2);
                        da.total_cmp(&db)
                    })
                {
                    let dh = (r.h - full_ref.shifts[&(best_k, "H".to_string())]).abs();
                    let dn = (r.n - full_ref.shifts[&(best_k, "N".to_string())]).abs();
                    if dh <= 0.04 && dn <= 0.30 {
                        let rm = r_metal_ang[best_k];
                        if rm < 7.6 {
                            picked_core_blind += 1;
                        } else if rm <= 10.0 {
                            picked_twilight += 1;
                        } else {
                            picked_outer += 1;
                        }
                    }
                }
            }
        }

        let mut picked_all = picked_roots.clone();
        for (e_idx, exp_name) in ["HNCA", "HNCACB", "CBCACONH", "HNCO"].iter().enumerate() {
            let mut rcfg_3d = rcfg_base.clone();
            rcfg_3d.seed = 5200 + seed * 31 + (e_idx as u64);
            let vol_3d = render_triple_3d(exp_name, &all_peaks, &rcfg_3d);
            picked_all.extend(pick_triple_3d(&vol_3d, &picked_roots, &pcfg));
        }

        let count_wrong_blind_zone = |positions: &[i32], truth: &[i32]| -> usize {
            positions
                .iter()
                .zip(truth.iter())
                .filter(|&(&pos, &tr)| {
                    pos >= 0
                        && pos != tr
                        && rep_3d
                            .p_bleach
                            .get(pos as usize)
                            .is_some_and(|&pb| pb >= cfg_3d.blind_zone_threshold)
                })
                .count()
        };

        // Globular baseline vs Paramagnetic-aware solver on identical volumetrically picked peaks
        let glob_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            auto_detect_motifs: false,
            enable_generic_ca_less_filter: false,
            enable_u_para_bleach_prior: false,
            seed,
            ..Default::default()
        };
        let (g_sys, g_res, _) =
            assign_paramagnetic_from_peaks(&picked_all, &full_ref.sequence, &priors, &glob_cfg);
        let g_truth = map_spin_systems_to_truth(&g_sys, &full_ref);
        glob_fps += count_wrong_blind_zone(&g_res.position, &g_truth);
        let gm = score(&g_res, &g_truth, 0.80);
        glob_precs.push(gm.accuracy() * 100.0);

        let para_cfg = ParamagneticAssignConfig {
            mode: ParamagneticSolverMode::EvoGraph,
            pdb_coordinates: Some(pdb_1i7h.clone()),
            seed,
            ..Default::default()
        };
        let (p_sys, p_res, _) =
            assign_paramagnetic_from_peaks(&picked_all, &full_ref.sequence, &priors, &para_cfg);
        let p_truth = map_spin_systems_to_truth(&p_sys, &full_ref);
        para_fps += count_wrong_blind_zone(&p_res.position, &p_truth);
        let pm = score(&p_res, &p_truth, 0.80);
        para_precs.push(pm.accuracy() * 100.0);
        para_hcps.push(pm.hc_precision() * 100.0);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);
    println!(
        "[Volumetric Solomon-PRE 2D+3D Picked Ferredoxin (3 seeds, 1I7H)]:\n  Seed-0 2D HSQC picked shells: core_blind(r<7.6Å)={picked_core_blind}, twilight(7.6..10Å)={picked_twilight}, outer(r>10Å)={picked_outer}\n  Globular EvoGraph: prec={:.2}% | blind_zone_FP={glob_fps}\n  Paramagnetic EvoGraph: prec={:.2}% | HC_prec={:.2}% | blind_zone_FP={para_fps}",
        mean(&glob_precs),
        mean(&para_precs),
        mean(&para_hcps),
    );
    assert_eq!(
        picked_core_blind, 0,
        "core blind-zone amides (r_M < 7.6 Å) must naturally attenuate below the volumetric 2D HSQC picker threshold"
    );
    assert!(
        picked_twilight >= 2 && picked_outer >= 55,
        "expected volumetric picker to recover twilight-zone and outer-shell amides (twilight={picked_twilight}, outer={picked_outer})"
    );
    assert!(
        glob_fps > 0 && para_fps == 0,
        "volumetrically picked paramagnetic benchmark must show globular blind-zone FPs ({glob_fps} > 0) eliminated to 0 by paramagnetic solver ({para_fps})"
    );
    assert!(
        mean(&para_precs) >= mean(&glob_precs) && mean(&para_hcps) >= 95.0,
        "volumetric paramagnetic solver must outperform globular precision and maintain >=95% HC precision (glob={:.2}%, para={:.2}%, HC={:.2}%)",
        mean(&glob_precs),
        mean(&para_precs),
        mean(&para_hcps)
    );
}
