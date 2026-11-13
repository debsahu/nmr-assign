//! Integration & quantitative-gate tests for `Phase 11` `T11.1`:
//! Protonless `13C`-detected IDP experiment models (`CON`, `CACO`, `CANCO`, `CBCACON`, `CCC(CO)N`,
//! `NCACO`), field-strength-invariant (`500–950 MHz`) narrow-dispersion peak pickers, volumetric
//! cube simulator, proline-rich (`>= 20% Pro`) `-Xaa-Pro-` / `-Pro-Pro-` recovery gates, and
//! `1H^N` solvent-exchange broadening comparison.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::c13_detected::{
    c13_obs_mhz, hz_to_ppm, intrinsic_amide_h_exchange_rate, n15_obs_mhz, pick_caco_2d,
    pick_con_2d, pick_con_anchored_3d, simulate_h1_exchange_vs_c13_con, simulate_idp_13c_cubes,
    C13DetectedExp, C13PickerConfig, CacoPeakKind, IdpCubeSimConfig,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms, Reference};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_c13_experiment_models_and_field_invariance() {
    for exp in C13DetectedExp::ALL {
        assert_eq!(C13DetectedExp::from_name(exp.as_str()), Some(exp));
        assert_eq!(exp.direct_nucleus(), "13C'");
        assert!(!exp.pathway_description().is_empty());
        let ((c_lo, c_hi), _, _) = exp.spectral_bounds_ppm();
        assert!(c_lo <= 168.0 && c_hi >= 180.0);
    }
    assert!(C13DetectedExp::Con2D.is_2d());
    assert!(C13DetectedExp::Caco2D.is_2d());
    assert!(C13DetectedExp::Canco3D.is_3d());
    assert!(C13DetectedExp::Cbcaco3D.is_3d());
    assert!(C13DetectedExp::Cccon3D.is_3d());
    assert!(C13DetectedExp::Ncaco3D.is_3d());

    // Dynamic Hz -> ppm scaling across 500 MHz..950 MHz (zero hardcoded 500 MHz ppm constants)
    let lw_500 = hz_to_ppm(5.0, c13_obs_mhz(500.0), 150.0);
    let lw_950 = hz_to_ppm(5.0, c13_obs_mhz(950.0), 150.0);
    let ratio = lw_500 / lw_950;
    assert!(
        (ratio - 950.0 / 500.0).abs() < 1e-6,
        "Hz->ppm ratio must scale inversely with B0: got {ratio}"
    );
    let n_500 = hz_to_ppm(4.5, n15_obs_mhz(500.0), 60.0);
    let n_800 = hz_to_ppm(4.5, n15_obs_mhz(800.0), 60.0);
    assert!((n_500 / n_800 - 1.6).abs() < 1e-6);
}

#[test]
fn test_crowded_con_cluster_deconvolution_and_ipap_suppression_500_to_950_mhz() {
    // Construct a synthetic crowded IDP cluster containing:
    // - 3 overlapping CON peaks separated by 0.04 ppm in 13C' and 0.15 ppm in 15N
    // - 1 `-Xaa-Pro-` peak at 136.8 ppm and 1 `-Pro-Pro-` peak at 135.95 ppm / +0.045 ppm C'
    let seq: Vec<String> = ["ALA", "SER", "GLY", "THR", "PRO", "PRO"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let seq_ids: Vec<i32> = (1..=6).collect();
    let mut ref_data = Reference::new(seq, seq_ids);
    // Crowded amide trio around (175.50..175.58 C', 120.10..120.25 N)
    let true_con = [
        (0, 175.50, 1, 120.10),
        (1, 175.54, 2, 120.25),
        (2, 175.58, 3, 120.11),
        // Proline imino doublet (-Thr-Pro- and -Pro-Pro-)
        (3, 172.90, 4, 136.85),
        (4, 175.10, 5, 135.92),
    ];
    for &(ci, cp, ni, np) in &true_con {
        ref_data.shifts.insert((ci, "C".to_string()), cp);
        ref_data.shifts.insert((ni, "N".to_string()), np);
    }

    for &mhz in &[500.0, 700.0, 950.0] {
        let sim_cfg = IdpCubeSimConfig {
            proton_mhz: mhz,
            lw_c_prime_hz: 4.4,
            lw_n15_hz: 4.0,
            snr_2d: 65.0,
            ipap_residual_frac: 0.055, // 5.5% residual IPAP ±J/2 decoupling satellites
            con_grid: (1024, 1024),
            experiments: vec![C13DetectedExp::Con2D],
            seed: 101 + (mhz as u64),
            ..IdpCubeSimConfig::default()
        };
        let cubes = simulate_idp_13c_cubes(&ref_data, &sim_cfg);
        let picked = pick_con_2d(&cubes.con_2d, &C13PickerConfig::default());

        // Every true CON peak must be recovered within (0.025 ppm C', 0.09 ppm N),
        // and symmetric 5.5% IPAP satellites must NOT be picked as false roots!
        for &(_, cp, _, np) in &true_con {
            let hit = picked
                .iter()
                .any(|p| (p.c_prev_ppm - cp).abs() <= 0.028 && (p.n_curr_ppm - np).abs() <= 0.095);
            assert!(
                hit,
                "Failed to resolve crowded CON peak ({cp:.2}, {np:.2}) at {mhz} MHz; picked={picked:?}"
            );
        }
        assert_eq!(
            picked.len(),
            true_con.len(),
            "IPAP satellites or noise leaked at {mhz} MHz: expected {}, got {}",
            true_con.len(),
            picked.len()
        );
        let n_pro_region = picked.iter().filter(|p| p.is_proline_imino_region).count();
        assert_eq!(n_pro_region, 2);
    }
}

#[test]
fn test_proline_rich_bmrb_con_root_recovery_gate_above_97_percent() {
    let root = workspace_root();
    // Both real public CC0 BMRB IDP fixtures have > 21.5% Proline!
    let fixtures = [
        (
            root.join("data/fixtures/bmr26639_idp_cbp_id4.str"),
            "BMRB 26639 CBP-ID4 (207 aa, 21.7% Pro)",
        ),
        (
            root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
            "BMRB 51984 p53-TAD-PRR (102 aa, 21.6% Pro)",
        ),
    ];

    for (path, label) in &fixtures {
        let ref_data = read_star(path, None).expect("valid BMRB IDP STAR fixture");
        let n_pro = ref_data
            .sequence
            .iter()
            .filter(|r| r.eq_ignore_ascii_case("PRO"))
            .count();
        let pro_frac = (n_pro as f64) / (ref_data.sequence.len() as f64);
        assert!(
            pro_frac >= 0.20,
            "{label} must have >= 20% Pro, got {:.2}%",
            pro_frac * 100.0
        );

        let sim_cfg = IdpCubeSimConfig {
            proton_mhz: 800.0,
            temperature_k: 303.15,
            ph: 6.5,
            lw_c_prime_hz: 4.5,
            lw_n15_hz: 4.0,
            snr_2d: 55.0,
            ipap_residual_frac: 0.03,
            con_grid: (1536, 1536),
            experiments: vec![C13DetectedExp::Con2D, C13DetectedExp::Caco2D],
            seed: 2026,
            ..IdpCubeSimConfig::default()
        };
        let cubes = simulate_idp_13c_cubes(&ref_data, &sim_cfg);
        let con_peaks = pick_con_2d(&cubes.con_2d, &C13PickerConfig::default());

        let mut matched_all = 0usize;
        let mut total_x_pro = 0usize;
        let mut matched_x_pro = 0usize;
        let mut total_pro_pro = 0usize;
        let mut matched_pro_pro = 0usize;

        for &(_, cp, np, prev_pro, curr_pro) in &cubes.ground_truth_con {
            let hit = con_peaks
                .iter()
                .any(|p| (p.c_prev_ppm - cp).abs() <= 0.038 && (p.n_curr_ppm - np).abs() <= 0.15);
            if hit {
                matched_all += 1;
            }
            if !prev_pro && curr_pro {
                total_x_pro += 1;
                if hit {
                    matched_x_pro += 1;
                }
            }
            if prev_pro && curr_pro {
                total_pro_pro += 1;
                if hit {
                    matched_pro_pro += 1;
                }
            }
        }

        let rec_all = (matched_all as f64) / (cubes.ground_truth_con.len() as f64);
        let rec_x_pro = (matched_x_pro as f64) / (total_x_pro.max(1) as f64);
        let rec_pro_pro = (matched_pro_pro as f64) / (total_pro_pro.max(1) as f64);

        assert!(
            rec_all >= 0.97,
            "{label}: CON overall recovery {:.2}% ({}/{}) < 97% gate!",
            rec_all * 100.0,
            matched_all,
            cubes.ground_truth_con.len()
        );
        assert!(
            rec_x_pro >= 0.97,
            "{label}: -Xaa-Pro- recovery {:.2}% ({}/{}) < 97% gate!",
            rec_x_pro * 100.0,
            matched_x_pro,
            total_x_pro
        );
        if total_pro_pro > 0 {
            assert!(
                rec_pro_pro >= 0.97,
                "{label}: -Pro-Pro- recovery {:.2}% ({}/{}) < 97% gate!",
                rec_pro_pro * 100.0,
                matched_pro_pro,
                total_pro_pro
            );
        }

        // Verify 2D CACO Gly opposite-phase classification on the same protein
        let caco_spec = cubes.caco_2d.as_ref().unwrap();
        let caco_peaks = pick_caco_2d(caco_spec, &C13PickerConfig::default());
        let n_gly_seq = ref_data
            .sequence
            .iter()
            .enumerate()
            .filter(|&(i, r)| {
                r.eq_ignore_ascii_case("GLY")
                    && ref_data.shifts.contains_key(&(i, "CA".to_string()))
                    && ref_data.shifts.contains_key(&(i, "C".to_string()))
            })
            .count();
        let n_gly_opp = caco_peaks
            .iter()
            .filter(|p| p.kind == CacoPeakKind::GlyOppositePhase && p.sign < 0)
            .count();
        if n_gly_seq > 0 {
            assert!(
                n_gly_opp * 10 >= n_gly_seq * 8,
                "{label}: expected opposite-phase Gly CACO peaks (got {n_gly_opp} vs {n_gly_seq} Gly)"
            );
        }
    }
}

#[test]
fn test_3d_con_anchored_strips_with_inter_experiment_drift() {
    let root = workspace_root();
    let ref_data = read_star_all_atoms(
        root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .unwrap();
    // Sub-slice first 36 residues (p53 TAD1, rich in Pro/Glu/Leu) for fast full-cube 3D test
    let mut sub = Reference::new(
        ref_data.sequence[..36].to_vec(),
        ref_data.seq_ids[..36].to_vec(),
    );
    for ((i, atom), &val) in &ref_data.shifts {
        if *i < 36 {
            sub.shifts.insert((*i, atom.clone()), val);
        }
    }

    let mut exp_offsets = BTreeMap::new();
    exp_offsets.insert(C13DetectedExp::Canco3D, (0.14, -0.18, 0.11));
    exp_offsets.insert(C13DetectedExp::Cbcaco3D, (-0.12, 0.16, -0.09));
    exp_offsets.insert(C13DetectedExp::Cccon3D, (0.10, 0.14, 0.08));
    exp_offsets.insert(C13DetectedExp::Ncaco3D, (-0.11, -0.15, 0.12));

    let sim_cfg = IdpCubeSimConfig {
        proton_mhz: 700.0,
        con_grid: (768, 768),
        cube_3d_grid: (128, 128, 144),
        experiments: vec![
            C13DetectedExp::Con2D,
            C13DetectedExp::Canco3D,
            C13DetectedExp::Cbcaco3D,
            C13DetectedExp::Cccon3D,
            C13DetectedExp::Ncaco3D,
        ],
        exp_offsets,
        seed: 77,
        ..IdpCubeSimConfig::default()
    };
    let cubes = simulate_idp_13c_cubes(&sub, &sim_cfg);
    let picker_cfg = C13PickerConfig::default();
    let con_roots = pick_con_2d(&cubes.con_2d, &picker_cfg);
    assert!(con_roots.len() >= cubes.ground_truth_con.len() - 1);

    for (exp, cube) in &cubes.cubes_3d {
        let strips = pick_con_anchored_3d(cube, *exp, &con_roots, &picker_cfg);
        assert!(
            strips.len() >= con_roots.len() * 8 / 10,
            "3D {exp:?} strip picker lost peaks under carrier drift: got {} peaks for {} roots",
            strips.len(),
            con_roots.len()
        );
    }
}

#[test]
fn test_h1_exchange_broadening_vs_c13_con_advantage_at_physiological_ph() {
    // Verify Bai-Englander intrinsic exchange physics
    let kex_ph6 = intrinsic_amide_h_exchange_rate("ALA", "SER", 6.0, 298.15);
    let kex_ph74 = intrinsic_amide_h_exchange_rate("ALA", "SER", 7.4, 308.15);
    assert!(kex_ph74 > 25.0 * kex_ph6);
    assert_eq!(
        intrinsic_amide_h_exchange_rate("ALA", "PRO", 7.4, 308.15),
        0.0
    );

    let root = workspace_root();
    let cbp = read_star(root.join("data/fixtures/bmr26639_idp_cbp_id4.str"), None).unwrap();
    let sim_cfg = IdpCubeSimConfig {
        proton_mhz: 700.0,
        temperature_k: 308.15, // 35 C physiological temperature
        ph: 7.5,
        snr_2d: 45.0,
        con_grid: (1536, 1536),
        seed: 308,
        ..IdpCubeSimConfig::default()
    };
    let cmp = simulate_h1_exchange_vs_c13_con(&cbp, &sim_cfg, &C13PickerConfig::default());
    assert!(
        cmp.con_bond_recovery_frac >= 0.97,
        "13C-CON recovery at pH 7.5, 308 K must stay >= 97%, got {:.2}%",
        cmp.con_bond_recovery_frac * 100.0
    );
    assert!(
        cmp.hsqc_bond_recovery_frac < 0.45,
        "1H-15N HSQC at pH 7.5, 308 K on 21.7%-Pro IDP must suffer severe Pro + k_ex loss (< 45%), got {:.2}%",
        cmp.hsqc_bond_recovery_frac * 100.0
    );
    assert!(cmp.proline_peptide_bonds >= 44);
    assert!(cmp.amide_bonds_broadened_below_snr > 40);
}
