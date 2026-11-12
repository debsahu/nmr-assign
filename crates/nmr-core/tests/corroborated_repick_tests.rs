//! Integration and real-spectrum tests for Stage A2 two-pass experiment-corroborated
//! 3D strip peak rescue (`StripCube3D`, `StripBank3D`, `corroborated_re_pick_3d`,
//! `pick_triple_3d_two_pass`).
//!
//! Provenance: Clean-room verification of `docs/specs/02-peak-picking.md` Stage A2.

use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::evaluate::score;
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::picker::{
    corroborated_re_pick_3d, estimate_noise_mad, pick_hsqc_2d, pick_triple_3d_two_pass,
    CorroboratedRepickConfig, PickerConfig, StripBank3D,
};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_ucsf_2d, read_ucsf_3d, AxisPpm, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn find_local_real_data_dir() -> Option<PathBuf> {
    if let Some(env_dir) = std::env::var_os("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(env_dir);
        if p.exists() {
            return Some(p);
        }
    }
    let local_data = workspace_root().join("data");
    if local_data.join("ubiquitin").exists() {
        return Some(local_data);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let cand = PathBuf::from(home).join("Workspace/nmr_software/data");
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn make_test_cube(
    exp: &str,
    c_hi: f64,
    c_lo: f64,
    gaussians_sigma: &[(f64, f64, f64, f32)],
) -> Spectrum3D {
    let (nx, ny, nz) = (160usize, 64usize, 384usize);
    let axis_x = AxisPpm::from_bounds("1H", nx, 10.2, 5.8, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 130.0, 106.0, 60.81);
    let axis_z = AxisPpm::from_bounds("13C", nz, c_hi, c_lo, 150.90);
    let mut data = vec![0.0f32; nx * ny * nz];
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (exp.len() as u64 * 0x517C_C1B7_2722_0A95);
    for v in &mut data {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (((state >> 40) & 0xFFFF) as f32) / 65536.0 - 0.5;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (((state >> 40) & 0xFFFF) as f32) / 65536.0 - 0.5;
        *v = u1 + u2;
    }
    let sigma = estimate_noise_mad(&data);

    for &(h0, n0, c0, snr_mult) in gaussians_sigma {
        let amp = snr_mult * sigma;
        let cx = axis_x.ppm_to_index(h0);
        let cy = axis_y.ppm_to_index(n0);
        let cz = axis_z.ppm_to_index(c0);
        for iz in (cz.floor() as isize - 4).max(0) as usize
            ..=(cz.ceil() as isize + 4).min(nz as isize - 1) as usize
        {
            for iy in (cy.floor() as isize - 3).max(0) as usize
                ..=(cy.ceil() as isize + 3).min(ny as isize - 1) as usize
            {
                for ix in (cx.floor() as isize - 3).max(0) as usize
                    ..=(cx.ceil() as isize + 3).min(nx as isize - 1) as usize
                {
                    if (ix as f64 - cx).abs() <= 1.3
                        && (iy as f64 - cy).abs() <= 1.3
                        && (iz as f64 - cz).abs() <= 1.3
                    {
                        data[(iz * ny + iy) * nx + ix] = 0.0;
                    }
                    let dx = (ix as f64 - cx) / 1.45;
                    let dy = (iy as f64 - cy) / 1.45;
                    let dz = (iz as f64 - cz) / 1.55;
                    let g = (-0.5 * (dx * dx + dy * dy + dz * dz)) as f32;
                    data[(iz * ny + iy) * nx + ix] += amp * g.exp();
                }
            }
        }
    }

    Spectrum3D {
        name: exp.to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

#[test]
fn test_intra_strip_corroborated_rescue_and_uncorroborated_rejection() {
    let pcfg = PickerConfig::default();
    let rcfg = CorroboratedRepickConfig::default();

    // Root 0: true backbone amide (e.g. Gly following Arg); Roots 1 & 2: paired sidechain NH2 doublet
    let roots = vec![
        Peak {
            exp: "HSQC".into(),
            h: 8.42,
            n: 111.10,
            c: None,
            sign: 1,
            height: Some(50.0),
            volume: Some(200.0),
            h2: None,
        },
        Peak {
            exp: "HSQC".into(),
            h: 7.52,
            n: 112.40,
            c: None,
            sign: 1,
            height: Some(40.0),
            volume: Some(160.0),
            h2: None,
        },
        Peak {
            exp: "HSQC".into(),
            h: 6.84,
            n: 112.42,
            c: None,
            sign: 1,
            height: Some(39.0),
            volume: Some(155.0),
            h2: None,
        },
    ];

    // CBCACONH: strong 14-sigma CA(i-1)=53.75 and CB(i-1)=28.00 on Root 0
    let cbcaconh = make_test_cube(
        "CBCACONH",
        76.0,
        10.0,
        &[(8.42, 111.10, 53.75, 16.0), (8.42, 111.10, 28.00, 15.0)],
    );
    // HNCACB: strong CA(i)=42.50 (+18), strong CB(i-1)=28.00 (-8), WEAK corroborated CA(i-1)=53.75 (+4.4),
    // UNCORROBORATED weak bump at 61.20 (+4.5, must be rejected!), and NH2 decoy bump at 53.75 (+4.5, must be rejected!)
    let hncacb = make_test_cube(
        "HNCACB",
        76.0,
        10.0,
        &[
            (8.42, 111.10, 42.50, 18.0),
            (8.42, 111.10, 28.00, -8.5),
            (8.42, 111.10, 53.75, 4.4),
            (8.42, 111.10, 61.20, 4.5),
            (7.52, 112.40, 53.75, 4.5),
        ],
    );
    // HNCO: strong C'(i-1)=174.10 (+18)
    let hnco = make_test_cube("HNCO", 186.0, 165.0, &[(8.42, 111.10, 174.10, 18.0)]);
    // HNCACO: strong C'(i)=171.80 (+14), WEAK corroborated C'(i-1)=174.10 (+4.3), UNCORROBORATED bump at 179.50 (+4.5)
    let hncaco = make_test_cube(
        "HNCACO",
        186.0,
        165.0,
        &[
            (8.42, 111.10, 171.80, 14.0),
            (8.42, 111.10, 174.10, 4.3),
            (8.42, 111.10, 179.50, 4.5),
        ],
    );

    let raw_bytes =
        (cbcaconh.data.len() + hncacb.data.len() + hnco.data.len() + hncaco.data.len()) * 4;
    let mut bank = StripBank3D::new();
    let mut pass1 = roots.clone();
    for spec in [&cbcaconh, &hncacb, &hnco, &hncaco] {
        pass1.extend(bank.pick_and_record_3d(spec, &roots, &pcfg));
    }
    assert!(
        bank.byte_size() * 5 < raw_bytes,
        "StripBank3D ({}) must be < 20% of raw 3D cubes ({})",
        bank.byte_size(),
        raw_bytes
    );

    // Pass 1 misses the 4.3-4.4 sigma peaks at 53.75 (HNCACB) and 174.10 (HNCACO)
    assert!(
        !pass1
            .iter()
            .any(|p| p.exp == "HNCACB" && (p.c.unwrap_or(0.0) - 53.75).abs() < 0.30),
        "Pass 1 should not pick 4.4-sigma peak before corroboration"
    );
    assert!(
        !pass1
            .iter()
            .any(|p| p.exp == "HNCACO" && (p.c.unwrap_or(0.0) - 174.10).abs() < 0.25),
        "Pass 1 should not pick 4.3-sigma HNCACO peak before corroboration"
    );

    let rescued = corroborated_re_pick_3d(&bank, &roots, &pass1, &pcfg, &rcfg);
    assert_eq!(
        rescued.len(),
        2,
        "Expected exactly 2 corroborated rescues (HNCACB 53.75 & HNCACO 174.10), got {rescued:?}"
    );
    assert!(rescued
        .iter()
        .any(|p| p.exp == "HNCACB" && p.sign == 1 && (p.c.unwrap() - 53.75).abs() < 0.15));
    assert!(rescued
        .iter()
        .any(|p| p.exp == "HNCACO" && p.sign == 1 && (p.c.unwrap() - 174.10).abs() < 0.12));
    // Verify uncorroborated 61.20 / 179.50 bumps and NH2 decoy bumps were NOT rescued
    assert!(!rescued.iter().any(|p| (p.h - 7.52).abs() < 0.10));

    // Convenience wrapper produces identical 3D peak set
    let two_pass =
        pick_triple_3d_two_pass(&[&cbcaconh, &hncacb, &hnco, &hncaco], &roots, &pcfg, &rcfg);
    assert_eq!(two_pass.len(), (pass1.len() - roots.len()) + rescued.len());

    // Downstream spin-system assembly recovers CA(-1)=53.75 with 2-experiment support
    let mut combined = pass1;
    combined.extend(rescued);
    let systems = build_spin_systems(&combined, &Tolerances::default(), 3, false).unwrap();
    let ca_im1 = systems[0]
        .cands
        .get(&("CA".to_string(), -1))
        .expect("CA(-1) typed after Pass-2 rescue");
    assert!((ca_im1[0].0 - 53.75).abs() < 0.25 && ca_im1[0].1 == 2);
}

#[test]
fn test_cross_strip_zero_peak_rescue_preserves_single_peak_degeneracy() {
    let pcfg = PickerConfig::default();
    let rcfg = CorroboratedRepickConfig::default();

    // Root 0: empty HNCACB(+) strip (weak 4.5-sigma CA(i)=56.40), validated by strong HNCO + CBCACONH
    // Root 1: successor residue with strong 15-sigma CBCACONH CA(i-1)=56.40
    // Root 2: degenerate CA(i)≈CA(i-1)=55.40 (1 strong +16 peak) + 4.5-sigma noise bump at 56.40 (must NOT split degeneracy!)
    let roots = vec![
        Peak {
            exp: "HSQC".into(),
            h: 8.20,
            n: 119.50,
            c: None,
            sign: 1,
            height: Some(50.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HSQC".into(),
            h: 8.55,
            n: 122.30,
            c: None,
            sign: 1,
            height: Some(50.0),
            volume: None,
            h2: None,
        },
        Peak {
            exp: "HSQC".into(),
            h: 9.05,
            n: 115.00,
            c: None,
            sign: 1,
            height: Some(50.0),
            volume: None,
            h2: None,
        },
    ];

    let cbcaconh = make_test_cube(
        "CBCACONH",
        76.0,
        10.0,
        &[
            (8.20, 119.50, 29.50, 15.0),
            (8.55, 122.30, 56.40, 16.0),
            (9.05, 115.00, 55.40, 16.0),
        ],
    );
    let hncacb = make_test_cube(
        "HNCACB",
        76.0,
        10.0,
        &[
            (8.20, 119.50, 29.50, -14.0),
            (8.20, 119.50, 56.40, 4.95), // 0-peak +CA on Root 0 -> rescued via Root 1's 56.40!
            (8.55, 122.30, 56.40, 9.0),
            (8.55, 122.30, 61.00, 16.0),
            (9.05, 115.00, 55.40, 16.0), // 1-peak degenerate CA on Root 2
            (9.05, 115.00, 56.40, 4.95), // must NOT be rescued on Root 2!
        ],
    );
    let hnco = make_test_cube(
        "HNCO",
        186.0,
        165.0,
        &[
            (8.20, 119.50, 175.20, 16.0),
            (8.55, 122.30, 173.80, 16.0),
            (9.05, 115.00, 176.10, 16.0),
        ],
    );

    let mut bank = StripBank3D::new();
    let mut pass1 = roots.clone();
    for spec in [&cbcaconh, &hncacb, &hnco] {
        pass1.extend(bank.pick_and_record_3d(spec, &roots, &pcfg));
    }

    let rescued = corroborated_re_pick_3d(&bank, &roots, &pass1, &pcfg, &rcfg);
    assert_eq!(
        rescued.len(),
        1,
        "Expected only Root 0 0-peak rescue, got {rescued:?}"
    );
    assert!((rescued[0].h - 8.20).abs() < 0.03 && (rescued[0].c.unwrap() - 56.40).abs() < 0.15);
}

fn map_systems_1to1(
    systems: &[SpinSystem],
    truth_ref: &Reference,
    tol_h: f64,
    tol_n: f64,
) -> Vec<i32> {
    let mut out = vec![-1i32; systems.len()];
    for k in 1..truth_ref.sequence.len() {
        if truth_ref.sequence[k] == "PRO" {
            continue;
        }
        let (Some(&th), Some(&tn)) = (
            truth_ref.shifts.get(&(k, "H".to_string())),
            truth_ref.shifts.get(&(k, "N".to_string())),
        ) else {
            continue;
        };
        let mut best_s = None;
        let mut best_d = 1.0_f64;
        for (s, ss) in systems.iter().enumerate() {
            let d = ((ss.h - th) / tol_h).hypot((ss.n - tn) / tol_n);
            if d <= 1.0 && d < best_d && out[s] < 0 {
                best_d = d;
                best_s = Some(s);
            }
        }
        if let Some(s) = best_s {
            out[s] = k as i32;
        }
    }
    out
}

#[test]
fn test_synthetic_realistic_3d_two_pass_non_regression() {
    let syn_ref = read_star(
        workspace_root().join("data/fixtures/synthetic_ubiquitin.str"),
        None,
    )
    .unwrap();
    let (syn_peaks, _) = simulate(
        &syn_ref,
        &SimConfig {
            seed: 3,
            p_missing_peak: 0.0,
            p_missing_root: 0.0,
            spurious_rate: 0.0,
            noise_h: 0.0,
            noise_n: 0.0,
            noise_c: 0.0,
            exchange_sigma: 0.95,
            ..Default::default()
        },
    );

    let pcfg = PickerConfig::default();
    let rcfg_repick = CorroboratedRepickConfig::default();
    let mut rcfg = SpectrumRenderConfig::realistic_bruker();
    rcfg.nx_h = 512;
    rcfg.ny_n = 96;
    rcfg.nz_c = 192;
    rcfg.seed = 293;
    rcfg.noise_level = 0.16;

    let hsqc = render_hsqc_2d(&syn_peaks, &rcfg);
    let roots = pick_hsqc_2d(&hsqc, &pcfg);
    let mut pass1 = roots.clone();
    let mut bank = StripBank3D::new();
    for exp in ["HNCACB", "CBCACONH", "HNCO", "HNCACO"] {
        rcfg.seed += 13;
        let s3 = render_triple_3d(exp, &syn_peaks, &rcfg);
        pass1.extend(bank.pick_and_record_3d(&s3, &roots, &pcfg));
    }
    let rescued = corroborated_re_pick_3d(&bank, &roots, &pass1, &pcfg, &rcfg_repick);
    assert!(
        !rescued.is_empty(),
        "Expected corroborated rescues on low-SNR exchange-broadened 3D cubes"
    );
    let mut pass2 = pass1.clone();
    pass2.extend(rescued);

    let (es1, er1, _) = assign_evo_graph_from_peaks(
        &pass1,
        &syn_ref.sequence,
        &Priors::provisional(),
        &EvoGraphConfig {
            seed: 3,
            ..Default::default()
        },
    );
    let et1 = map_systems_1to1(&es1, &syn_ref, 0.04, 0.35);
    let m1 = score(&er1, &et1, 0.8);

    let (es2, er2, _) = assign_evo_graph_from_peaks(
        &pass2,
        &syn_ref.sequence,
        &Priors::provisional(),
        &EvoGraphConfig {
            seed: 3,
            ..Default::default()
        },
    );
    let et2 = map_systems_1to1(&es2, &syn_ref, 0.04, 0.35);
    let m2 = score(&er2, &et2, 0.8);

    assert!(
        m2.accuracy() >= m1.accuracy() && m2.hc_precision() >= m1.hc_precision(),
        "Pass-2 regressed synthetic 3D evo-graph: pass1=({:.2}%, {:.2}%), pass2=({:.2}%, {:.2}%)",
        100.0 * m1.accuracy(),
        100.0 * m1.hc_precision(),
        100.0 * m2.accuracy(),
        100.0 * m2.hc_precision()
    );
}

fn parse_fasta_three(path: &Path) -> Vec<String> {
    let txt = fs::read_to_string(path).unwrap();
    let mut seq = Vec::new();
    for line in txt.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('>') {
            continue;
        }
        for ch in t.chars() {
            if ch.is_ascii_alphabetic() {
                seq.push(one_to_three(ch).to_string());
            }
        }
    }
    seq
}

#[test]
fn test_real_ubiquitin_and_cisd3_two_pass_repick() {
    let Some(droot) = find_local_real_data_dir() else {
        eprintln!("Skipping real-data Pass-2 test (local real NMR data directory not present)");
        return;
    };
    let pcfg = PickerConfig::default();
    let repick_cfg = CorroboratedRepickConfig::default();

    let ubq_dir = droot.join("ubiquitin");
    let ubq_table = droot.join("ubiquitin_backbone_assignment_191104.txt");
    if ubq_dir.join("Ubiquitin_090309/2/pdata/1").exists() && ubq_table.exists() {
        let u_seq = parse_fasta_three(&ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt"));
        let txt = fs::read_to_string(&ubq_table).unwrap();
        let mut u_truth = Reference::new(u_seq.clone(), (1..=76).collect());
        for line in txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("Index") {
                continue;
            }
            let norm = t.replace("?1", "? 1").replace(":H* ?", ":H*?");
            let cols: Vec<&str> = norm.split_whitespace().collect();
            let Ok(sn) = cols[0].parse::<usize>() else {
                continue;
            };
            let k = sn - 1;
            let pv = |s: &str| {
                (!s.contains('?') && s != "-")
                    .then(|| s.split(':').next().unwrap().parse::<f64>().ok())
                    .flatten()
            };
            if let Some(v) = cols.get(2).and_then(|s| pv(s)) {
                u_truth.shifts.insert((k, "H".into()), v);
            }
            if let Some(v) = cols.get(3).and_then(|s| pv(s)) {
                u_truth.shifts.insert((k, "N".into()), v);
            }
            if let Some(v) = cols.get(4).and_then(|s| pv(s)) {
                u_truth.shifts.insert((k, "C".into()), v);
            }
            if let Some(v) = cols.get(5).and_then(|s| pv(s)) {
                u_truth.shifts.insert((k, "CA".into()), v);
            }
            if let Some(v) = cols
                .iter()
                .skip(6)
                .find(|s| s.ends_with(":Cb"))
                .and_then(|s| pv(s))
            {
                u_truth.shifts.insert((k, "CB".into()), v);
            }
        }
        let n_obs = 70usize;

        let u_hsqc = read_bruker_pdata_2d(ubq_dir.join("Ubiquitin_090309/2/pdata/1")).unwrap();
        let u_roots = pick_hsqc_2d(&u_hsqc, &pcfg);
        let mut u_p1 = u_roots.clone();
        let mut u_bank = StripBank3D::new();
        for (rel, exp) in [
            ("Ubiquitin_090309/3/pdata/1", "HNCACB"),
            ("Ubiquitin_090309/4/pdata/1", "CBCACONH"),
            ("Ubiquitin_111909/10/pdata/1", "HNCO"),
            ("Ubiquitin_111909/11/pdata/1", "HNCACO"),
        ] {
            let s3 = read_bruker_pdata_3d(ubq_dir.join(rel), exp).unwrap();
            u_p1.extend(u_bank.pick_and_record_3d(&s3, &u_roots, &pcfg));
        }
        assert_eq!(u_p1.len(), 972);
        let u_rescued = corroborated_re_pick_3d(&u_bank, &u_roots, &u_p1, &pcfg, &repick_cfg);
        assert!(
            u_rescued
                .iter()
                .any(|p| p.exp == "HNCACB" && p.sign == 1 && (p.c.unwrap() - 53.75).abs() < 0.15),
            "Must rescue GLY75's weak 4.3-sigma CA(i-1)=53.75 peak in Ubiquitin HNCACB"
        );
        let mut u_p2 = u_p1.clone();
        u_p2.extend(u_rescued);

        let sys1 = build_spin_systems(&u_p1, &Tolerances::default(), 3, false).unwrap();
        let sys2 = build_spin_systems(&u_p2, &Tolerances::default(), 3, false).unwrap();
        let t1 = map_systems_1to1(&sys1, &u_truth, 0.08, 0.65);
        let t2 = map_systems_1to1(&sys2, &u_truth, 0.08, 0.65);

        let mut std_rec_1 = 0.0;
        let mut std_rec_2 = 0.0;
        let mut std_hcp_1 = 0.0;
        let mut std_hcp_2 = 0.0;
        for seed in 0..5u64 {
            let acfg = AnnealConfig {
                runs: 64,
                steps_per_system: 4000,
                seed,
                ..Default::default()
            };
            let r1 = assign(
                &sys1,
                &u_seq,
                &Priors::bmrb(),
                &ScoreConfig::default(),
                &acfg,
            );
            let r2 = assign(
                &sys2,
                &u_seq,
                &Priors::bmrb(),
                &ScoreConfig::default(),
                &acfg,
            );
            let mut m1 = score(&r1, &t1, 0.8);
            let mut m2 = score(&r2, &t2, 0.8);
            m1.assignable = n_obs;
            m2.assignable = n_obs;
            std_rec_1 += m1.recall() / 5.0;
            std_rec_2 += m2.recall() / 5.0;
            std_hcp_1 += m1.hc_precision() / 5.0;
            std_hcp_2 += m2.hc_precision() / 5.0;
        }
        assert!(
            std_rec_2 > std_rec_1 && std_hcp_2 >= std_hcp_1 - 1e-6,
            "Ubiquitin pdata standard solver must improve recall without HC precision loss: rec {:.2}%->{:.2}%, HCp {:.2}%->{:.2}%",
            100.0 * std_rec_1,
            100.0 * std_rec_2,
            100.0 * std_hcp_1,
            100.0 * std_hcp_2
        );

        // Sparky UCSF parity check on seed 0
        let ucsf_dir = ubq_dir.join("sparky_files/ubiquitin_data");
        if ucsf_dir.join("HSQC.ucsf").exists() {
            let s_hsqc = read_ucsf_2d(ucsf_dir.join("HSQC.ucsf")).unwrap();
            let s_roots = pick_hsqc_2d(&s_hsqc, &pcfg);
            let mut s_p1 = s_roots.clone();
            let mut s_bank = StripBank3D::new();
            for (fname, exp) in [
                ("HNCACB.ucsf", "HNCACB"),
                ("CBCACONH.ucsf", "CBCACONH"),
                ("HNCO.ucsf", "HNCO"),
                ("HNCACO.ucsf", "HNCACO"),
            ] {
                let s3 = read_ucsf_3d(ucsf_dir.join(fname), exp).unwrap();
                s_p1.extend(s_bank.pick_and_record_3d(&s3, &s_roots, &pcfg));
            }
            let s_rescued = corroborated_re_pick_3d(&s_bank, &s_roots, &s_p1, &pcfg, &repick_cfg);
            assert!(!s_rescued.is_empty());
        }
    }

    // Private CISD3 (BMRB 51439) 5-seed non-regression check when present locally
    let cisd3_dir = droot.join("private/bmrb51439_cisd3");
    if cisd3_dir.join("bmr51439_3.str").exists() {
        let c_seq = parse_fasta_three(&cisd3_dir.join("cisd3.fasta"));
        let mut c_truth = read_star(cisd3_dir.join("bmr51439_3.str"), None).unwrap();
        for ((_, atom), val) in c_truth.shifts.iter_mut() {
            match atom.as_str() {
                "H" => *val += 0.003,
                "N" => *val -= 0.955,
                "CA" | "CB" | "C" => *val -= 2.55,
                _ => {}
            }
        }
        let c_hsqc =
            read_bruker_pdata_2d(cisd3_dir.join("CISD3_15N_HSQC_diamagnetic/pdata/1")).unwrap();
        let c_roots = pick_hsqc_2d(&c_hsqc, &pcfg);
        let mut c_p1 = c_roots.clone();
        let mut c_bank = StripBank3D::new();
        for (rel, exp) in [
            ("CISD3_HNCA/pdata/1", "HNCA"),
            ("CISD3_HNCOCA/pdata/1", "HNCOCA"),
            ("CISD3_CBCACONH/pdata/1", "CBCACONH"),
            ("CISD3_HNCO/pdata/1", "HNCO"),
        ] {
            let s3 = read_bruker_pdata_3d(cisd3_dir.join(rel), exp).unwrap();
            c_p1.extend(c_bank.pick_and_record_3d(&s3, &c_roots, &pcfg));
        }
        let c_rescued = corroborated_re_pick_3d(&c_bank, &c_roots, &c_p1, &pcfg, &repick_cfg);
        let mut c_p2 = c_p1.clone();
        c_p2.extend(c_rescued);

        for seed in 0..5u64 {
            let (es1, er1, _) = assign_evo_graph_from_peaks(
                &c_p1,
                &c_seq,
                &Priors::provisional(),
                &EvoGraphConfig {
                    seed,
                    ..Default::default()
                },
            );
            let (es2, er2, _) = assign_evo_graph_from_peaks(
                &c_p2,
                &c_seq,
                &Priors::provisional(),
                &EvoGraphConfig {
                    seed,
                    ..Default::default()
                },
            );
            let et1 = map_systems_1to1(&es1, &c_truth, 0.045, 0.40);
            let et2 = map_systems_1to1(&es2, &c_truth, 0.045, 0.40);
            let m1 = score(&er1, &et1, 0.8);
            let m2 = score(&er2, &et2, 0.8);
            assert!(
                m2.accuracy() >= m1.accuracy() - 1e-9
                    && m2.hc_precision() >= m1.hc_precision() - 1e-9,
                "CISD3 seed {seed} regressed: pass1=({:.2}%, {:.2}%), pass2=({:.2}%, {:.2}%)",
                100.0 * m1.accuracy(),
                100.0 * m1.hc_precision(),
                100.0 * m2.accuracy(),
                100.0 * m2.hc_precision()
            );
        }
    }
}
