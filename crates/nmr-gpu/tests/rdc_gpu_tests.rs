//! WebGPU (`wgpu` + WGSL) batched multi-conformer RDC torsion energy & analytical gradient parity
//! (`< 1e-5` vs `f64` CPU reference), baseline browser `wgpu::Limits::default()` portability, and
//! cross-validated experimental `1D3Z` Ubiquitin GPU ensemble refinement tests (`Phase 12`, `T12.4`).
//!
//! Provenance: Clean-room verification suite citing:
//! - Tjandra, N. & Bax, A. (*Science* 1997, 278, 1111–1114).
//! - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* 1998, 120, 6836–6837).
//! - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (*J. Magn. Reson.* 1999, 138, 334–342).
//! - Parsons, J. et al. (*J. Comput. Chem.* 2005, 26, 1063–1068 — `NeRF`).
//! - Abe, H., Braun, W., Noguti, T. & Go, N. (*Comput. Chem.* 1984, 8, 239–247).

use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::noesy::{
    compute_chain_torsions, parse_pdb_first_model, v_norm, v_scale, v_sub, PdbChain,
};
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, evaluate_rdc_validation, partition_rdc_free_set, read_rdc_tsv,
    saupe_matrix_from_pas, RdcBondType, RdcMediumSet, RdcRestraint,
};
use nmr_core::rng::Rng;
use nmr_core::structure::{
    build_backbone_from_torsions, evaluate_rdc_torsion_energy_and_gradients_fixed_saupe,
    refine_ensemble_with_rdcs, superpose_chain_onto, StructureConformer,
};
use nmr_gpu::rdc_gpu::{
    evaluate_rdc_batch_gpu, refine_ensemble_with_rdcs_gpu, RdcGpuConformerInput,
};
use nmr_gpu::GpuEngine;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn build_all5_medium(chain: &PdbChain, medium_id: &str, s_mat: &[[f64; 3]; 3]) -> RdcMediumSet {
    let mut restraints = Vec::new();
    let l = chain.residues.len();
    for i in 0..l {
        let r = &chain.residues[i];
        if let Some(h) = r.h {
            let u = v_scale(v_sub(h, r.n), 1.0 / v_norm(v_sub(h, r.n)));
            let dmax = RdcBondType::NH.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "N".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "H".to_string(),
                bond_type: RdcBondType::NH,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 0.5,
                d_max_hz: dmax,
                is_free: i % 5 == 4,
            });
        }
        if r.res_name.eq_ignore_ascii_case("GLY") {
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            for (aname, pos_opt) in [("HA2", r.ha2), ("HA3", r.ha3)] {
                let ha_pos = pos_opt.expect("GLY must carry explicit HA2 and HA3 coordinates");
                let u = v_scale(v_sub(ha_pos, r.ca), 1.0 / v_norm(v_sub(ha_pos, r.ca)));
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: r.seq_id,
                    res_name_1: r.res_name.clone(),
                    atom_name_1: "CA".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: aname.to_string(),
                    bond_type: RdcBondType::CAHA,
                    d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                    sigma_hz: 1.0,
                    d_max_hz: dmax,
                    is_free: false,
                });
            }
        } else {
            let u = v_scale(v_sub(r.ha, r.ca), 1.0 / v_norm(v_sub(r.ha, r.ca)));
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "HA".to_string(),
                bond_type: RdcBondType::CAHA,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 1.0,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        {
            let u = v_scale(v_sub(r.c, r.ca), 1.0 / v_norm(v_sub(r.c, r.ca)));
            let dmax = RdcBondType::CACO.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "C".to_string(),
                bond_type: RdcBondType::CACO,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 0.25,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        if i > 0 {
            let prev = &chain.residues[i - 1];
            let u_nc = v_scale(v_sub(r.n, prev.c), 1.0 / v_norm(v_sub(r.n, prev.c)));
            let dmax_nc = RdcBondType::NCPrime.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: prev.seq_id,
                res_name_1: prev.res_name.clone(),
                atom_name_1: "C".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "N".to_string(),
                bond_type: RdcBondType::NCPrime,
                d_obs_hz: calc_rdc_from_unit_vec(u_nc, dmax_nc, s_mat),
                sigma_hz: 0.20,
                d_max_hz: dmax_nc,
                is_free: false,
            });
            if let Some(h) = r.h {
                let u_hc = v_scale(v_sub(h, prev.c), 1.0 / v_norm(v_sub(h, prev.c)));
                let dmax_hc = RdcBondType::HNCPrime.default_d_max_hz();
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: prev.seq_id,
                    res_name_1: prev.res_name.clone(),
                    atom_name_1: "C".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: "H".to_string(),
                    bond_type: RdcBondType::HNCPrime,
                    d_obs_hz: calc_rdc_from_unit_vec(u_hc, dmax_hc, s_mat),
                    sigma_hz: 0.35,
                    d_max_hz: dmax_hc,
                    is_free: false,
                });
            }
        }
    }
    RdcMediumSet {
        medium_id: medium_id.to_string(),
        restraints,
    }
}

#[test]
fn test_rdc_gpu_vs_cpu_batch_energy_and_gradient_parity_and_default_limits() {
    let engine = match GpuEngine::new_headless_with_limits(wgpu::Limits::default()) {
        Ok(e) => e,
        Err(err) => {
            println!("Skipping WebGPU RDC parity test (no headless GPU adapter): {err}");
            return;
        }
    };

    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "ILE",
        "THR", "LEU", "GLU", "VAL", "GLU", "PRO", "SER", "ASP", "THR", "ILE", "GLU",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let l = seq.len();
    let seq_ids: Vec<i32> = (1..=(l as i32)).collect();

    let mut rng = Rng::new(987654);
    let ref_phi: Vec<f64> = (0..l).map(|_| rng.uniform(-140.0, -55.0)).collect();
    let ref_psi: Vec<f64> = (0..l).map(|_| rng.uniform(-50.0, 150.0)).collect();
    let ref_omg = vec![180.0; l];
    let ref_chain = build_backbone_from_torsions(&seq, &seq_ids, &ref_phi, &ref_psi, &ref_omg);

    let s_obs_1 = saupe_matrix_from_pas(16.0, 0.26, [32.0, 51.0, 114.0]);
    let s_obs_2 = saupe_matrix_from_pas(-13.0, 0.48, [128.0, 42.0, 230.0]);
    let media = vec![
        build_all5_medium(&ref_chain, "m1", &s_obs_1),
        build_all5_medium(&ref_chain, "m2", &s_obs_2),
    ];

    let n_conf = 16usize;
    let w_rdc = 0.42;
    let mut batch_in = Vec::with_capacity(n_conf);
    for c in 0..n_conf {
        let phi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.6, 2.2)).collect();
        let psi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.6, 2.6)).collect();
        let omega_rad = vec![PI; l];
        let sm1 = saupe_matrix_from_pas(
            14.0 + 0.5 * (c as f64),
            0.20 + 0.02 * (c as f64),
            [20.0 + 7.0 * (c as f64), 48.0, 95.0],
        );
        let sm2 = saupe_matrix_from_pas(
            -11.5 - 0.4 * (c as f64),
            0.35 + 0.015 * (c as f64),
            [110.0, 39.0 + 3.0 * (c as f64), 210.0],
        );
        batch_in.push(RdcGpuConformerInput {
            phi_rad,
            psi_rad,
            omega_rad,
            saupe_matrices: vec![sm1, sm2],
        });
    }

    let gpu_outs = evaluate_rdc_batch_gpu(&engine, &seq, &seq_ids, &media, &batch_in, w_rdc)
        .expect("GPU batch RDC evaluation");
    assert_eq!(gpu_outs.len(), n_conf);

    let mut max_rel_e = 0.0f64;
    let mut max_rel_g = 0.0f64;

    for (ci, (inp, gout)) in batch_in.iter().zip(gpu_outs.iter()).enumerate() {
        let mut cpu_gp = vec![0.0; l];
        let mut cpu_gs = vec![0.0; l];
        let cpu_e = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
            &seq,
            &seq_ids,
            &inp.phi_rad,
            &inp.psi_rad,
            &inp.omega_rad,
            &media,
            &inp.saupe_matrices,
            w_rdc,
            &mut cpu_gp,
            &mut cpu_gs,
        );

        let rel_e = (gout.rdc_energy - cpu_e).abs() / cpu_e.abs().max(1.0);
        max_rel_e = max_rel_e.max(rel_e);
        assert!(
            rel_e < 1e-5,
            "conformer {ci} CPU/GPU energy parity exceeded 1e-5: cpu={cpu_e}, gpu={}, rel={rel_e}",
            gout.rdc_energy
        );

        let g_scale = cpu_gp
            .iter()
            .chain(cpu_gs.iter())
            .fold(1.0f64, |m, v| m.max(v.abs()));
        for i in 0..l {
            let err_p = (gout.grad_phi[i] - cpu_gp[i]).abs() / g_scale;
            let err_s = (gout.grad_psi[i] - cpu_gs[i]).abs() / g_scale;
            max_rel_g = max_rel_g.max(err_p).max(err_s);
            assert!(
                err_p < 1e-5 && err_s < 1e-5,
                "conformer {ci} res {i} CPU/GPU grad parity exceeded 1e-5: err_phi={err_p}, err_psi={err_s}"
            );
        }
    }

    println!(
        "[RDC GPU PARITY] adapter='{}' ({}) | 16 conformers x 2 media x 5 bond types: max_rel_energy_err = {:.3e}, max_rel_grad_err = {:.3e}",
        engine.adapter_name, engine.backend_name, max_rel_e, max_rel_g
    );
}

#[test]
fn test_rdc_gpu_ensemble_refinement_ubiquitin_1d3z() {
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            println!("Skipping WebGPU 1D3Z refinement test (no headless GPU adapter): {err}");
            return;
        }
    };

    let root = workspace_root();
    let pdb_1d3z = parse_pdb_first_model(root.join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_1d3z =
        nmr_core::nmrstar::read_star_all_atoms(root.join("data/benchmark/bmr17769.str"), None)
            .unwrap();
    let fw_ubq = nmr_core::torsion_db::HomologFirewallQuery {
        query_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string(), "1G6J".to_string()],
        query_bmrb_ids: vec!["bmr17769".to_string(), "bmr5387".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: Some("3.10.20.90".to_string()),
        query_topology_id: Some("3.10.20".to_string()),
        query_ca_coords: Some(pdb_1d3z.residues.iter().map(|r| r.ca).collect()),
    };
    let fw_dihs = nmr_core::torsion_db::predict_torsions_with_firewall(&shifts_1d3z, &fw_ubq)
        .to_dihedral_restraints();
    let peaks_1d3z = nmr_core::noesy::simulate_noesy_peaks(
        &pdb_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::NoesySimConfig::default(),
    );
    let noe_adrs = nmr_core::noesy::match_initial_adrs(
        &peaks_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::AdrConfig::default(),
    );
    let mut media = read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).unwrap();
    partition_rdc_free_set(&mut media, 0.20, 4242);

    let native_tors = compute_chain_torsions(&pdb_1d3z);
    let seq: Vec<String> = pdb_1d3z
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let seq_ids: Vec<i32> = pdb_1d3z.residues.iter().map(|r| r.seq_id).collect();
    let l = seq.len();

    // Create 6 perturbed conformers around the folded basin (~9 deg torsion noise -> ~2.0-2.8 A BB RMSD)
    let mut rng = Rng::new(31415);
    let mut init_confs = Vec::new();
    for _ in 0..6 {
        let mut phi_deg = Vec::with_capacity(l);
        let mut psi_deg = Vec::with_capacity(l);
        for &(_, _, p_opt, s_opt) in &native_tors {
            phi_deg.push(p_opt.unwrap_or(-63.0) + rng.normal(0.0, 8.5));
            psi_deg.push(s_opt.unwrap_or(135.0) + rng.normal(0.0, 8.5));
        }
        let omega_deg = vec![180.0; l];
        let ch = build_backbone_from_torsions(&seq, &seq_ids, &phi_deg, &psi_deg, &omega_deg);
        init_confs.push(StructureConformer {
            chain: ch,
            phi_deg,
            psi_deg,
            omega_deg,
            total_energy: 100.0,
            noe_energy: 0.0,
            dihedral_energy: 0.0,
            vdw_energy: 0.0,
            rama_energy: 0.0,
        });
    }

    let val_before = evaluate_rdc_validation(&init_confs[0].chain, &media).unwrap();
    let mut c0_before = init_confs[0].chain.clone();
    let ordered_mask: Vec<bool> = (0..l).map(|i| i < 71).collect();
    let rmsd_before = superpose_chain_onto(&mut c0_before, &pdb_1d3z, Some(&ordered_mask));

    // Matched-input, matched-`free_seed = 4242` CPU vs WebGPU comparison on the exact same 6 starting conformers
    let (cpu_refined, cpu_val_after) = refine_ensemble_with_rdcs(
        &init_confs,
        &seq,
        &seq_ids,
        &noe_adrs,
        &fw_dihs,
        &media,
        0.55,
        110,
    )
    .expect("CPU 1D3Z RDC ensemble refinement on matched inputs");
    let mut c0_cpu_after = cpu_refined[0].chain.clone();
    let cpu_rmsd_after = superpose_chain_onto(&mut c0_cpu_after, &pdb_1d3z, Some(&ordered_mask));

    let (refined, val_after) = refine_ensemble_with_rdcs_gpu(
        &engine,
        &init_confs,
        &seq,
        &seq_ids,
        &noe_adrs,
        &fw_dihs,
        &media,
        0.55,
        110,
    )
    .expect("WebGPU 1D3Z RDC ensemble refinement");

    let mut c0_after = refined[0].chain.clone();
    let rmsd_after = superpose_chain_onto(&mut c0_after, &pdb_1d3z, Some(&ordered_mask));

    println!(
        "[RDC MATCHED-INPUT 1D3Z REFINEMENT (free_seed=4242, 6 conformers, w_rdc=0.55, iters=110)]"
    );
    println!(
        "  Initial           : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        val_before.overall_q_work, val_before.overall_q_free, rmsd_before
    );
    println!(
        "  CPU (f64 L-BFGS)  : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        cpu_val_after.overall_q_work, cpu_val_after.overall_q_free, cpu_rmsd_after
    );
    println!(
        "  WebGPU + polish   : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        val_after.overall_q_work, val_after.overall_q_free, rmsd_after
    );

    assert!(
        cpu_val_after.overall_q_free < 0.25 && cpu_rmsd_after < rmsd_before,
        "CPU matched-input RDC refinement failed gate: Q_free={:.4}, RMSD={cpu_rmsd_after:.3} A",
        cpu_val_after.overall_q_free
    );
    assert!(
        val_after.overall_q_free < 0.25 && val_after.overall_q_free < val_before.overall_q_free,
        "GPU RDC refinement failed Q_free gate: {:.4} -> {:.4}",
        val_before.overall_q_free,
        val_after.overall_q_free
    );
    assert!(
        rmsd_after < rmsd_before,
        "GPU RDC refinement must improve ordered BB RMSD ({rmsd_before:.3} -> {rmsd_after:.3} A)"
    );
}
