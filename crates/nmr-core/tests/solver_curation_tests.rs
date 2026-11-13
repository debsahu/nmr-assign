//! Integration and regression tests for Task T09.7 (Backbone Solver Curation):
//! 1. User-locked `(spin_system, residue)` hard constraints (`locked_assignments`), verifying
//!    bit-for-bit identical `U` and `P` when empty, safe filtering of non-assignable (`k=0` / `PRO`)
//!    or duplicate locks, and strict pinning (`confidence == 1.0`) across `standard` and `evo_graph`
//!    (including raw HSQC root protection in `assign_evo_graph_from_peaks`).
//! 2. Sub-stoichiometric minor-conformer peak damping (`rel_root_height < 0.15`) in `evo_graph`
//!    without modifying `docs/specs/03-assignment.md` Section 4 spin-system clustering, tested on
//!    both synthetic `Gln2`-like `9%` slow-exchange duplicate roots (preserving `>= 15%`-median
//!    exchange-broadened amides and unique weak C-terminal Glycine roots) and real Ubiquitin
//!    Bruker `pdata` + Sparky `.ucsf` spectra when `NMR_REAL_DATA_DIR` is set.
//! 3. Sparse-data (`CA`-only: `HSQC` + `HNCA` + `HNCOCA` + `HNCO`, absent `HNCACB`/`CBCACONH`)
//!    Metropolis-Hastings chain-extension & segment-shift convergence in `standard` (`solver.rs`).
//! 4. Expected Calibration Error (`ECE <= 0.06`) audit via `ProteinReliabilityCurve`.
//!
//! Provenance: Clean-room test suite grounded in Schmidt & Güntert (*J. Am. Chem. Soc.* 2012, 134,
//! 12817–12829), Klukowski, Riek & Güntert (*Nat. Commun.* 2022, 13, 6151), Wishart et al.
//! (*J. Biomol. NMR* 1995, 5, 67–81), Metropolis et al. (*J. Chem. Phys.* 1953, 21, 1087–1092),
//! Hastings (*Biometrika* 1970, 57, 97–109), and Guo et al. (*Proc. 34th ICML* 2017, PMLR 70,
//! 1321–1330).

use std::path::PathBuf;

use nmr_core::evaluate::score;
use nmr_core::evo_graph::{assign_evo_graph, assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::Peak;
use nmr_core::picker::{
    corroborated_re_pick_3d, pick_hsqc_2d, pick_triple_3d, CorroboratedRepickConfig, PickerConfig,
    StripBank3D,
};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::qc::ProteinReliabilityCurve;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{
    assign, pair_matrix, resolve_locked_assignments, unary_matrix, AnnealConfig, ScoreConfig,
    FORBID,
};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d, read_ucsf_2d, read_ucsf_3d};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};

fn fixture_ubiquitin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn hsqc_with_height(h: f64, n: f64, height: f64) -> Peak {
    let mut p = Peak::hsqc(h, n);
    p.height = Some(height);
    p
}

fn triple_with_height(exp: &str, h: f64, n: f64, c: f64, sign: i32, height: f64) -> Peak {
    let mut p = Peak::triple(exp, h, n, c, sign);
    p.height = Some(height);
    p
}

#[test]
fn empty_locks_preserve_bit_exact_u_and_p_and_filter_invalid_locks() {
    let ref_data =
        read_star(fixture_ubiquitin_path(), None).expect("parse synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let sim_cfg = SimConfig {
        seed: 7,
        ..SimConfig::realistic_bruker()
    };
    let (peaks, _) = simulate(&ref_data, &sim_cfg);
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 4, false).expect("systems");

    let cfg_default = ScoreConfig::default();
    assert!(cfg_default.locked_assignments.is_empty());

    let u_a = unary_matrix(&systems, &ref_data.sequence, &priors, &cfg_default);
    let p_a = pair_matrix(&systems, &cfg_default);

    // Invalid locks ONLY (k=0 N-terminus, Pro19 at k=18, out-of-bounds s/k) must leave U & P bit-identical
    assert_eq!(ref_data.sequence[18], "PRO");
    let cfg_only_invalid = ScoreConfig {
        locked_assignments: vec![(0, 0), (1, 18), (9999, 5), (2, 9999)],
        ..ScoreConfig::default()
    };
    assert!(resolve_locked_assignments(
        systems.len(),
        &ref_data.sequence,
        &cfg_only_invalid.locked_assignments
    )
    .is_empty());
    let u_b = unary_matrix(&systems, &ref_data.sequence, &priors, &cfg_only_invalid);
    let p_b = pair_matrix(&systems, &cfg_only_invalid);

    assert_eq!(u_a.data.len(), u_b.data.len());
    for (i, (&va, &vb)) in u_a.data.iter().zip(&u_b.data).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "U bit-exact mismatch at index {i}: {va} vs {vb}"
        );
    }
    for (i, (&va, &vb)) in p_a.data.iter().zip(&p_b.data).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "P bit-exact mismatch at index {i}: {va} vs {vb}"
        );
    }

    // Duplicate spin or duplicate residue locks keep only the first injective assignable pair
    let resolved = resolve_locked_assignments(
        systems.len(),
        &ref_data.sequence,
        &[(3, 4), (3, 5), (6, 4), (7, 18), (8, 9)],
    );
    assert_eq!(resolved, vec![(3, 4), (8, 9)]);
}

#[test]
fn locked_constraints_enforced_across_standard_and_evo_graph() {
    let ref_data =
        read_star(fixture_ubiquitin_path(), None).expect("parse synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let sim_cfg = SimConfig {
        seed: 3,
        ..SimConfig::default()
    };
    let (mut peaks, truth) = simulate(&ref_data, &sim_cfg);

    // Append an extra weak HSQC root with a single carbon peak at the end of the HSQC list so
    // stoichiometric excess-root pruning would ordinarily drop it unless protected by a user lock.
    let raw_hsqc_count = peaks.iter().filter(|p| p.exp == "HSQC").count();
    let locked_raw_hsqc_idx = raw_hsqc_count;
    peaks.push(hsqc_with_height(9.65, 129.4, 12.0));
    peaks.push(triple_with_height("HNCA", 9.65, 129.4, 54.2, 1, 8.0));

    let systems = build_spin_systems(&peaks, &Tolerances::default(), 4, false).expect("systems");
    assert_eq!(systems.len(), raw_hsqc_count + 1);

    // Choose spin system 0 (normally truth[0]) and force it onto a different assignable non-PRO
    // slot k_force_0, and force `locked_raw_hsqc_idx` onto assignable non-PRO slot k_force_extra.
    let k_force_0 = (1..ref_data.sequence.len())
        .find(|&k| ref_data.sequence[k] != "PRO" && truth[0] != k as i32 && k != 18)
        .unwrap();
    let k_force_extra = (1..ref_data.sequence.len())
        .rev()
        .find(|&k| ref_data.sequence[k] != "PRO" && k != k_force_0)
        .unwrap();
    let raw_locks = vec![
        (0, k_force_0),
        (locked_raw_hsqc_idx, k_force_extra),
        (2, 18), // PRO -> ignored
    ];

    let score_cfg = ScoreConfig {
        locked_assignments: raw_locks.clone(),
        ..ScoreConfig::default()
    };
    let u = unary_matrix(&systems, &ref_data.sequence, &priors, &score_cfg);
    assert!(u.get(0, k_force_0) > FORBID * 0.5 && u.get(0, k_force_0).is_finite());
    assert!(
        u.get(locked_raw_hsqc_idx, k_force_extra) > FORBID * 0.5
            && u.get(locked_raw_hsqc_idx, k_force_extra).is_finite()
    );
    for k in 0..ref_data.sequence.len() {
        if k != k_force_0 {
            assert_eq!(u.get(0, k), FORBID);
        }
        if k != k_force_extra {
            assert_eq!(u.get(locked_raw_hsqc_idx, k), FORBID);
        }
    }
    for s in 0..systems.len() {
        if s != 0 {
            assert_eq!(u.get(s, k_force_0), FORBID);
        }
        if s != locked_raw_hsqc_idx {
            assert_eq!(u.get(s, k_force_extra), FORBID);
        }
    }

    // 1. Standard CPU solver
    let anneal_cfg = AnnealConfig {
        runs: 8,
        steps_per_system: 1200,
        seed: 11,
        ..AnnealConfig::default()
    };
    let std_res = assign(
        &systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        &anneal_cfg,
    );
    assert_eq!(std_res.position[0], k_force_0 as i32);
    assert_eq!(std_res.confidence[0], 1.0);
    assert_eq!(std_res.position[locked_raw_hsqc_idx], k_force_extra as i32);
    assert_eq!(std_res.confidence[locked_raw_hsqc_idx], 1.0);

    // 2. Evo-Graph solver on pre-built spin systems
    let evo_cfg = EvoGraphConfig {
        population_size: 24,
        num_islands: 4,
        locked_assignments: raw_locks.clone(),
        seed: 11,
        ..EvoGraphConfig::default()
    };
    let evo_res = assign_evo_graph(&systems, &ref_data.sequence, &priors, &evo_cfg);
    assert_eq!(evo_res.position[0], k_force_0 as i32);
    assert_eq!(evo_res.confidence[0], 1.0);
    assert_eq!(evo_res.position[locked_raw_hsqc_idx], k_force_extra as i32);
    assert_eq!(evo_res.confidence[locked_raw_hsqc_idx], 1.0);

    // 3. Evo-Graph solver from raw peaks (verifying stoichiometric pruning protection + remapping)
    let (_, peak_res, peak_rep) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);
    let kept_0 = peak_rep
        .kept_hsqc_indices
        .iter()
        .position(|&orig| orig == 0)
        .expect("raw HSQC root 0 retained");
    let kept_extra = peak_rep
        .kept_hsqc_indices
        .iter()
        .position(|&orig| orig == locked_raw_hsqc_idx)
        .expect("locked weak raw HSQC root protected from stoichiometric pruning");
    assert_eq!(peak_res.position[kept_0], k_force_0 as i32);
    assert_eq!(peak_res.confidence[kept_0], 1.0);
    assert_eq!(peak_res.position[kept_extra], k_force_extra as i32);
    assert_eq!(peak_res.confidence[kept_extra], 1.0);
}

#[test]
fn minor_conformer_gln2_competition_synthetic_and_real_ubiquitin() {
    let priors = Priors::bmrb();
    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "GLY",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let shifts: [(f64, f64, f64, Option<f64>, f64); 9] = [
        (0.00, 0.00, 55.4, Some(32.8), 175.8),  // M1 (k=0, N-term)
        (8.45, 121.2, 55.9, Some(29.2), 175.9), // Q2 (k=1)
        (8.29, 115.3, 60.8, Some(38.6), 175.1), // I3 (k=2)
        (8.56, 118.9, 57.3, Some(39.8), 174.9), // F4 (k=3)
        (8.18, 121.6, 62.1, Some(32.4), 175.4), // V5 (k=4)
        (8.37, 122.8, 56.4, Some(32.9), 176.2), // K6 (k=5)
        (8.22, 114.6, 61.7, Some(69.5), 174.6), // T7 (k=6, moderate 0.22x exchange-broadened!)
        (8.11, 123.4, 55.1, Some(42.2), 176.8), // L8 (k=7)
        (7.92, 109.6, 45.2, None, 173.9),       // G9 (k=8, unique weak 0.06x C-terminal Gly!)
    ];

    let mut peaks = Vec::new();
    for k in 1..seq.len() {
        let (h, n, ca0, cb0, c0) = shifts[k];
        let (_, _, cam1, cbm1, cm1) = shifts[k - 1];
        let root_h = match k {
            6 => 220.0, // 0.22 * median: genuine moderate exchange-broadened amide (must NOT be penalized)
            8 => 60.0, // 0.06 * median: unique 4-slot C-terminal Glycine (must still assign cleanly)
            _ => 1000.0,
        };
        peaks.push(hsqc_with_height(h, n, root_h));
        for (exp, c_val, sign) in [
            ("HNCACB", ca0, 1),
            ("HNCACB", cam1, 1),
            ("CBCACONH", cam1, 1),
            ("HNCO", cm1, 1),
            ("HNCACO", c0, 1),
            ("HNCACO", cm1, 1),
        ] {
            peaks.push(triple_with_height(exp, h, n, c_val, sign, root_h * 0.5));
        }
        if let Some(cb) = cb0 {
            peaks.push(triple_with_height("HNCACB", h, n, cb, -1, -root_h * 0.4));
        }
        if let Some(cb_prev) = cbm1 {
            peaks.push(triple_with_height(
                "HNCACB",
                h,
                n,
                cb_prev,
                -1,
                -root_h * 0.3,
            ));
            peaks.push(triple_with_height(
                "CBCACONH",
                h,
                n,
                cb_prev,
                1,
                root_h * 0.4,
            ));
        }
    }

    // Give major Gln2 root (s=0, H=8.45, N=121.2) crowded overlap extra candidates so its raw
    // -ln(n) candidate multiplicity penalty would lose to a clean 1-candidate minor clone without
    // sub-stoichiometric minor-conformer damping.
    for (exp, extra_c, sign) in [
        ("HNCACB", 61.9, 1),
        ("CBCACONH", 61.9, 1),
        ("HNCO", 173.4, 1),
    ] {
        peaks.push(triple_with_height(exp, 8.45, 121.2, extra_c, sign, 380.0));
    }

    // Add a 9%-median slow-exchange minor second-form Gln2 root (s=8, rel_root_height = 0.09 < 0.15)
    // with clean 1-candidate Gln2/Met1 carbon shifts!
    let (h_min, n_min) = (8.31, 120.4);
    peaks.push(hsqc_with_height(h_min, n_min, 90.0));
    for (exp, c_val, sign) in [
        ("HNCACB", 55.92, 1),
        ("HNCACB", 29.18, -1),
        ("CBCACONH", 55.38, 1),
        ("CBCACONH", 32.82, 1),
        ("HNCO", 175.81, 1),
        ("HNCACO", 175.88, 1),
    ] {
        peaks.push(triple_with_height(
            exp,
            h_min,
            n_min,
            c_val,
            sign,
            (sign as f64) * 45.0,
        ));
    }

    let built = build_spin_systems(&peaks, &Tolerances::default(), 4, false).expect("systems");
    assert_eq!(built.len(), 9);
    assert!((built[0].rel_root_height.unwrap() - 1.0).abs() < 1e-6);
    assert!((built[5].rel_root_height.unwrap() - 0.22).abs() < 1e-6);
    assert!((built[7].rel_root_height.unwrap() - 0.06).abs() < 1e-6);
    assert!((built[8].rel_root_height.unwrap() - 0.09).abs() < 1e-6);

    // Also verify 3D-peak-height fallback when HSQC heights are `None` preserves §4 `cands` identically
    let mut peaks_no_hsqc_h = peaks.clone();
    for p in &mut peaks_no_hsqc_h {
        if p.exp == "HSQC" {
            p.height = None;
        }
    }
    let built_fallback =
        build_spin_systems(&peaks_no_hsqc_h, &Tolerances::default(), 4, false).expect("fallback");
    for (a, b) in built.iter().zip(&built_fallback) {
        assert_eq!(a.cands, b.cands);
        assert!(b.root_height.is_some() && b.rel_root_height.is_some());
    }
    assert!((built_fallback[8].rel_root_height.unwrap() - 0.09).abs() < 0.02);

    let evo_cfg = EvoGraphConfig {
        seed: 0,
        ..EvoGraphConfig::default()
    };
    let (systems_out, res, rep) = assign_evo_graph_from_peaks(&peaks, &seq, &priors, &evo_cfg);
    let idx_of = |orig: usize| {
        rep.kept_hsqc_indices
            .iter()
            .position(|&i| i == orig)
            .expect("root retained")
    };
    let s_gln2_major = idx_of(0);
    let s_thr7_mod = idx_of(5);
    let s_gly9_weak = idx_of(7);
    assert_eq!(
        res.position[s_gln2_major], 1,
        "Major Gln2 root must win residue k=1; got positions={:?}",
        res.position
    );
    assert_eq!(
        res.position[s_thr7_mod], 6,
        "Moderate 0.22x exchange-broadened Thr7 root must win residue k=6"
    );
    assert_eq!(
        res.position[s_gly9_weak], 8,
        "Unique weak 0.06x C-terminal Gly9 root must still win residue k=8"
    );
    if let Some(s_gln2_minor) = rep.kept_hsqc_indices.iter().position(|&i| i == 8) {
        assert_eq!(
            res.position[s_gln2_minor], -1,
            "9% minor second-form Gln2 root must remain unassigned (-1), not displace major Gln2"
        );
    }
    let _ = systems_out;

    // Optional real Ubiquitin Bruker `pdata` + Sparky `.ucsf` verification when NMR_REAL_DATA_DIR is set
    let Some(droot) = std::env::var_os("NMR_REAL_DATA_DIR").map(PathBuf::from) else {
        return;
    };
    let ubq_dir = droot.join("ubiquitin");
    let fasta_path = ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt");
    let truth_path = droot.join("ubiquitin_backbone_assignment_191104.txt");
    if !fasta_path.exists() || !truth_path.exists() {
        return;
    }
    let fasta_txt = std::fs::read_to_string(&fasta_path).unwrap();
    let ubq_seq: Vec<String> = fasta_txt
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.trim().chars())
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| one_to_three(c).to_string())
        .collect();
    let truth_txt = std::fs::read_to_string(&truth_path).unwrap();
    let mut truth_hn = std::collections::BTreeMap::<usize, (f64, f64)>::new();
    let parse_tok = |tok: &str| -> Option<f64> {
        if tok == "-" || tok.contains('?') {
            return None;
        }
        tok.split(':').next().unwrap_or(tok).parse::<f64>().ok()
    };
    for line in truth_txt.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("Index") {
            continue;
        }
        let cols: Vec<&str> = t.split_whitespace().collect();
        if cols.len() >= 4 {
            if let (Ok(seq_num), Some(h_v), Some(n_v)) = (
                cols[0].parse::<usize>(),
                parse_tok(cols[2]),
                parse_tok(cols[3]),
            ) {
                if seq_num >= 1 && seq_num <= ubq_seq.len() {
                    truth_hn.insert(seq_num - 1, (h_v, n_v));
                }
            }
        }
    }
    let pcfg = PickerConfig::default();

    let hsqc_p = read_bruker_pdata_2d(ubq_dir.join("Ubiquitin_090309/2/pdata/1")).unwrap();
    let roots_p = pick_hsqc_2d(&hsqc_p, &pcfg);
    let mut peaks_p = roots_p.clone();
    for (rel, exp) in [
        ("Ubiquitin_090309/3/pdata/1", "HNCACB"),
        ("Ubiquitin_090309/4/pdata/1", "CBCACONH"),
        ("Ubiquitin_111909/10/pdata/1", "HNCO"),
        ("Ubiquitin_111909/11/pdata/1", "HNCACO"),
    ] {
        let s3 = read_bruker_pdata_3d(ubq_dir.join(rel), exp).unwrap();
        peaks_p.extend(pick_triple_3d(&s3, &roots_p, &pcfg));
    }

    let ucsf_dir = ubq_dir.join("sparky_files/ubiquitin_data");
    let hsqc_u = read_ucsf_2d(ucsf_dir.join("HSQC.ucsf")).unwrap();
    let roots_u = pick_hsqc_2d(&hsqc_u, &pcfg);
    let mut peaks_u = roots_u.clone();
    let mut bank_u = StripBank3D::new();
    for (fname, exp) in [
        ("HNCACB.ucsf", "HNCACB"),
        ("CBCACONH.ucsf", "CBCACONH"),
        ("HNCO.ucsf", "HNCO"),
        ("HNCACO.ucsf", "HNCACO"),
    ] {
        let s3 = read_ucsf_3d(ucsf_dir.join(fname), exp).unwrap();
        peaks_u.extend(bank_u.pick_and_record_3d(&s3, &roots_u, &pcfg));
    }
    peaks_u.extend(corroborated_re_pick_3d(
        &bank_u,
        &roots_u,
        &peaks_u,
        &pcfg,
        &CorroboratedRepickConfig::default(),
    ));

    for (label, real_peaks) in [("pdata", &peaks_p), ("ucsf", &peaks_u)] {
        let (real_sys, real_res, _) =
            assign_evo_graph_from_peaks(real_peaks, &ubq_seq, &priors, &evo_cfg);
        for &(target_k, res_label) in &[(1usize, "Gln2"), (2, "Ile3"), (16, "Val17"), (74, "Gly75")]
        {
            let &(th, tn) = truth_hn.get(&target_k).expect("truth H/N present");
            let s_assigned = real_res
                .position
                .iter()
                .position(|&p| p == target_k as i32)
                .unwrap_or_else(|| {
                    panic!("{label}: residue {res_label} (k={target_k}) must be assigned")
                });
            let d_hn =
                ((real_sys[s_assigned].h - th) / 0.045).hypot((real_sys[s_assigned].n - tn) / 0.40);
            assert!(
                d_hn <= 1.0,
                "{label}: spin s={s_assigned} assigned to {res_label} (k={target_k}) must match ground-truth (H,N), got d={d_hn:.3}"
            );
            if target_k == 1 {
                assert!(
                    real_sys[s_assigned]
                        .rel_root_intensity
                        .or(real_sys[s_assigned].rel_root_height)
                        .unwrap_or(1.0)
                        >= 0.45,
                    "{label}: Gln2 winner must be the major-state conformer"
                );
            }
        }
        for (s, ss) in real_sys.iter().enumerate() {
            if ss.rel_root_intensity.or(ss.rel_root_height).unwrap_or(1.0) < 0.15 {
                assert_ne!(
                    real_res.position[s], 1,
                    "{label}: sub-stoichiometric root s={s} must not occupy Gln2 (k=1)"
                );
            }
        }
    }
}

#[test]
fn sparse_ca_only_convergence_standard_solver() {
    let ref_data =
        read_star(fixture_ubiquitin_path(), None).expect("parse synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let score_cfg = ScoreConfig::default();

    // 1. Pure single-nucleus CA-only connectivity (`HSQC + HNCA + HNCOCA + HNCO`, zero CB and zero C(0))
    let mut ca_assigned = 0usize;
    let mut ca_correct = 0usize;
    let mut ca_hc_assigned = 0usize;
    let mut ca_hc_correct = 0usize;
    let mut evo_ca_assigned = 0usize;
    let mut evo_ca_correct = 0usize;

    // 2. CB-absent (`HNCACB`/`CBCACONH`-absent) sparse suite (`HSQC + HNCA + HNCOCA + HNCO + HNCACO`)
    let mut ncb_assigned = 0usize;
    let mut ncb_correct = 0usize;
    let mut ncb_hc_assigned = 0usize;
    let mut ncb_hc_correct = 0usize;

    for seed in 0..3u64 {
        let ca_only_cfg = SimConfig {
            experiments: vec![
                "HSQC".to_string(),
                "HNCA".to_string(),
                "HNCOCA".to_string(),
                "HNCO".to_string(),
            ],
            noise_h: 0.008,
            noise_n: 0.06,
            noise_c: 0.05,
            p_missing_peak: 0.05,
            p_missing_root: 0.02,
            spurious_rate: 0.03,
            sidechain_nh2: true,
            seed,
            ..SimConfig::default()
        };
        let (peaks, truth) = simulate(&ref_data, &ca_only_cfg);
        let systems: Vec<SpinSystem> =
            build_spin_systems(&peaks, &Tolerances::default(), 4, false).expect("systems");

        let pm = pair_matrix(&systems, &score_cfg);
        let max_p = pm
            .data
            .iter()
            .copied()
            .filter(|&v| v > FORBID * 0.5)
            .fold(0.0_f64, f64::max);
        assert!(
            max_p <= 4.65,
            "CA-only pair_matrix ceiling should be single-nucleus (~4.60 nats), got {max_p}"
        );

        let anneal_cfg = AnnealConfig {
            runs: 16,
            steps_per_system: 2500,
            seed: 100 + seed,
            ..AnnealConfig::default()
        };
        let std_res = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        );
        let m = score(&std_res, &truth, 0.80);
        ca_assigned += m.assigned;
        ca_correct += m.correct;
        ca_hc_assigned += m.hc_assigned;
        ca_hc_correct += m.hc_correct;

        // HNCACB/CBCACONH-absent 2-nucleus CA+C' sparse suite (HSQC + HNCA + HNCOCA + HNCO + HNCACO)
        let no_cb_cfg = SimConfig {
            experiments: vec![
                "HSQC".to_string(),
                "HNCA".to_string(),
                "HNCOCA".to_string(),
                "HNCO".to_string(),
                "HNCACO".to_string(),
            ],
            seed,
            ..ca_only_cfg
        };
        let (peaks_ncb, truth_ncb) = simulate(&ref_data, &no_cb_cfg);
        let systems_ncb: Vec<SpinSystem> =
            build_spin_systems(&peaks_ncb, &Tolerances::default(), 4, false).expect("systems");
        let pm_ncb = pair_matrix(&systems_ncb, &score_cfg);
        let max_p_ncb = pm_ncb
            .data
            .iter()
            .copied()
            .filter(|&v| v > FORBID * 0.5)
            .fold(0.0_f64, f64::max);
        assert!(
            max_p_ncb <= 9.25,
            "HNCACB/CBCACONH-absent pair_matrix ceiling should be 2-nucleus CA+C' (~9.21 nats), got {max_p_ncb}"
        );
        let std_ncb = assign(
            &systems_ncb,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        );
        let m_ncb = score(&std_ncb, &truth_ncb, 0.80);
        ncb_assigned += m_ncb.assigned;
        ncb_correct += m_ncb.correct;
        ncb_hc_assigned += m_ncb.hc_assigned;
        ncb_hc_correct += m_ncb.hc_correct;

        let (_, evo_res, evo_rep) = assign_evo_graph_from_peaks(
            &peaks_ncb,
            &ref_data.sequence,
            &priors,
            &EvoGraphConfig {
                seed: 100 + seed,
                ..EvoGraphConfig::default()
            },
        );
        let evo_truth: Vec<i32> = evo_rep
            .kept_hsqc_indices
            .iter()
            .map(|&i| truth_ncb[i])
            .collect();
        let m_evo = score(&evo_res, &evo_truth, 0.80);
        evo_ca_assigned += m_evo.assigned;
        evo_ca_correct += m_evo.correct;
    }

    let ca_acc = (ca_correct as f64) / (ca_assigned.max(1) as f64);
    let ca_hc_prec = (ca_hc_correct as f64) / (ca_hc_assigned.max(1) as f64);
    let evo_ca_acc = (evo_ca_correct as f64) / (evo_ca_assigned.max(1) as f64);
    let ncb_acc = (ncb_correct as f64) / (ncb_assigned.max(1) as f64);
    let ncb_hc_prec = (ncb_hc_correct as f64) / (ncb_hc_assigned.max(1) as f64);

    assert!(
        ca_acc >= 0.40 && ca_hc_prec >= 0.85,
        "Pure CA-only (HSQC+HNCA+HNCOCA+HNCO) standard solver 3-seed gate failed: acc={:.1}% ({}/{}), hc_prec={:.1}% ({}/{})",
        ca_acc * 100.0,
        ca_correct,
        ca_assigned,
        ca_hc_prec * 100.0,
        ca_hc_correct,
        ca_hc_assigned
    );
    assert!(
        ncb_acc >= 0.70 && ncb_hc_prec >= 0.90,
        "HNCACB-absent CA-only-connectivity standard solver 3-seed gate failed: acc={:.1}% ({}/{}), hc_prec={:.1}% ({}/{})",
        ncb_acc * 100.0,
        ncb_correct,
        ncb_assigned,
        ncb_hc_prec * 100.0,
        ncb_hc_correct,
        ncb_hc_assigned
    );
    assert!(
        evo_ca_acc >= 0.80,
        "HNCACB-absent CA-only-connectivity evo-graph 3-seed accuracy should be >= 80%, got {:.1}% ({}/{})",
        evo_ca_acc * 100.0,
        evo_ca_correct,
        evo_ca_assigned
    );
}

#[test]
fn evo_graph_confidence_calibration_ece_gate() {
    let ref_data =
        read_star(fixture_ubiquitin_path(), None).expect("parse synthetic_ubiquitin.str");
    let priors = Priors::bmrb();
    let mut pooled_preds: Vec<(f64, bool)> = Vec::new();

    for seed in 0..5u64 {
        let realistic_cfg = SimConfig {
            seed,
            ..SimConfig::realistic_bruker()
        };
        let (peaks, truth) = simulate(&ref_data, &realistic_cfg);
        let evo_cfg = EvoGraphConfig {
            seed,
            ..EvoGraphConfig::default()
        };
        let (_, res, rep) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);
        let kept_truth: Vec<i32> = rep
            .kept_hsqc_indices
            .iter()
            .map(|&orig| truth[orig])
            .collect();
        for (s, &pos) in res.position.iter().enumerate() {
            if pos >= 0 {
                pooled_preds.push((res.confidence[s], kept_truth[s] == pos));
            }
        }
    }

    let curve = ProteinReliabilityCurve::from_predictions(&pooled_preds, 0.80);
    assert!(
        curve.ece <= 0.06,
        "Expected Calibration Error must satisfy ECE <= 0.06, got ECE={:.4} (bins={:?})",
        curve.ece,
        curve.bins
    );
    assert!(
        curve.hc_precision.unwrap_or(0.0) >= 0.96,
        "Pooled HC precision must remain >= 96%, got {}",
        curve.hc_status
    );
}

#[test]
fn p1_5_volume_preferred_minor_conformer_over_broadened_height() {
    let priors = Priors::bmrb();
    let seq: Vec<String> = ["MET", "GLN", "ILE", "PHE", "VAL", "LYS"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let shifts: [(f64, f64, f64, Option<f64>, f64); 6] = [
        (8.30, 120.0, 55.4, Some(32.8), 175.8), // Met1
        (8.45, 121.2, 55.9, Some(29.2), 175.9), // Gln2
        (8.18, 119.5, 61.2, Some(38.9), 175.2), // Ile3
        (8.62, 122.4, 57.6, Some(40.1), 174.9), // Phe4
        (8.05, 118.2, 62.1, Some(33.0), 175.1), // Val5
        (8.39, 121.8, 56.8, Some(33.1), 176.2), // Lys6
    ];

    let mut peaks = Vec::new();
    for k in 1..seq.len() {
        let (h, n, ca0, cb0, c0) = shifts[k];
        let (_, _, cam1, cbm1, cm1) = shifts[k - 1];
        // Exchange-broadened major Gln2 (k=1) has lower peak height (140) than sharp minor (190),
        // but full stoichiometric integrated volume (5000 vs 400 -> r = 0.08 < 0.15).
        let (ht, vol) = if k == 1 {
            (140.0, 5000.0)
        } else {
            (1000.0, 5000.0)
        };
        let mut root = hsqc_with_height(h, n, ht);
        root.volume = Some(vol);
        peaks.push(root);
        for (exp, c_val, sign) in [
            ("HNCACB", ca0, 1),
            ("HNCACB", cam1, 1),
            ("CBCACONH", cam1, 1),
            ("HNCO", cm1, 1),
            ("HNCACO", c0, 1),
            ("HNCACO", cm1, 1),
        ] {
            peaks.push(triple_with_height(exp, h, n, c_val, sign, 400.0));
        }
        if let Some(cb) = cb0 {
            peaks.push(triple_with_height("HNCACB", h, n, cb, -1, -350.0));
        }
        if let Some(cb_prev) = cbm1 {
            peaks.push(triple_with_height("HNCACB", h, n, cb_prev, -1, -300.0));
            peaks.push(triple_with_height("CBCACONH", h, n, cb_prev, 1, 350.0));
        }
    }
    // Add overlap candidates on major Gln2 (s=0) so multiplicity penalty favors minor unless damped
    for (exp, extra_c, sign) in [
        ("HNCACB", 61.9, 1),
        ("CBCACONH", 61.9, 1),
        ("HNCO", 173.4, 1),
    ] {
        peaks.push(triple_with_height(exp, 8.45, 121.2, extra_c, sign, 300.0));
    }
    // Sharp minor slow-exchange Gln2 root (s=5): height 190 > major 140, but volume 400 = 0.08 * median!
    let (h_min, n_min) = (8.33, 120.5);
    let mut minor_root = hsqc_with_height(h_min, n_min, 190.0);
    minor_root.volume = Some(400.0);
    peaks.push(minor_root);
    for (exp, c_val, sign) in [
        ("HNCACB", 55.92, 1),
        ("HNCACB", 29.18, -1),
        ("CBCACONH", 55.38, 1),
        ("CBCACONH", 32.82, 1),
        ("HNCO", 175.81, 1),
        ("HNCACO", 175.88, 1),
    ] {
        peaks.push(triple_with_height(
            exp,
            h_min,
            n_min,
            c_val,
            sign,
            (sign as f64) * 80.0,
        ));
    }

    let built = build_spin_systems(&peaks, &Tolerances::default(), 4, false).expect("systems");
    assert!((built[0].rel_root_intensity.unwrap() - 1.0).abs() < 1e-6);
    assert!((built[5].rel_root_intensity.unwrap() - 0.08).abs() < 1e-6);

    let evo_cfg = EvoGraphConfig {
        seed: 0,
        ..EvoGraphConfig::default()
    };
    let (_, res, rep) = assign_evo_graph_from_peaks(&peaks, &seq, &priors, &evo_cfg);
    let s_major = rep.kept_hsqc_indices.iter().position(|&i| i == 0).unwrap();
    assert_eq!(
        res.position[s_major], 1,
        "Broadened full-volume major Gln2 root must win residue k=1"
    );
}
