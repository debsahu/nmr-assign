//! Integration, physics, thermodynamic, and accuracy benchmark tests for `nmr_core::bayes_sa`.
//!
//! Provenance: Clean-room verification of the Bayesian pseudo-energy and specific-heat ($C_v(T)$)
//! adaptive simulated annealing backbone solver inspired by Bishop, Torres-Montalvo, Kotaru,
//! Mimun & Wand, *Nature Communications* **14**, 1556 (2023), DOI `10.1038/s41467-023-37219-z`;
//! Kirkpatrick, Gelatt & Vecchi, *Science* **220**, 671–680 (1983); and Wishart et al.,
//! *J. Biomol. NMR* **5**, 67–81 (1995).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nmr_core::bayes_sa::{
    compute_bayes_energy_matrices, evaluate_state_energy, solve_bayes_sa,
    solve_bayes_sa_from_peaks, BayesSaConfig, CoolingScheduleKind, LockedAssignment, FORBID_ENERGY,
};
use nmr_core::evaluate::{score, Metrics};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::priors::{delta_covalent, Priors};
use nmr_core::rng::Rng;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::spinsystems::SpinSystem;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

type SlotSpec<'a> = (&'a str, i32, &'a [(f64, usize)]);

fn make_spin_system(_id: usize, h: f64, n: f64, slots: &[SlotSpec<'_>]) -> SpinSystem {
    let mut cands = BTreeMap::new();
    for &(atom, off, vals) in slots {
        cands.insert((atom.to_string(), off), vals.to_vec());
    }
    SpinSystem {
        h,
        n,
        cands,
        root_height: Some(1000.0),
        rel_root_height: Some(1.0),
        rel_root_intensity: Some(1.0),
    }
}

/// Build a clean-room synthetic protein `Reference` with realistic secondary structure
/// ($\alpha$-helix, $\beta$-strand, random coil), Wishart covalent neighbor shifts, and thermal jitter.
fn build_synthetic_protein_reference(
    length: usize,
    proline_rich_disordered: bool,
    seed: u64,
) -> Reference {
    let priors = Priors::default();
    let mut rng = Rng::new(seed);

    let folded_alphabet = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let idp_alphabet = [
        "PRO", "PRO", "PRO", "PRO", "PRO", "GLY", "GLY", "SER", "SER", "THR", "ALA", "GLU", "GLN",
        "ASP", "ASN", "LYS", "ARG", "VAL", "LEU", "ILE",
    ];

    let mut sequence = Vec::with_capacity(length);
    let mut seq_ids = Vec::with_capacity(length);
    sequence.push("MET".to_string());
    seq_ids.push(1);

    for k in 1..length {
        let aa = if proline_rich_disordered {
            // Guarantee >= 24% Proline while avoiding trivial all-Pro homopolymer stretches > 3
            if k % 4 == 2 {
                "PRO"
            } else {
                idp_alphabet[rng.usize(idp_alphabet.len())]
            }
        } else {
            folded_alphabet[rng.usize(folded_alphabet.len())]
        };
        sequence.push(aa.to_string());
        seq_ids.push((k + 1) as i32);
    }

    // Assign secondary-structure segments: 0 = alpha, 1 = beta, 2 = coil
    let mut ss_state = vec![2usize; length];
    if !proline_rich_disordered {
        let mut pos = 3usize;
        let mut motif = 0usize;
        while pos + 8 < length {
            let seg_len = 7 + rng.usize(8);
            let end = (pos + seg_len).min(length - 2);
            let conf = if motif.is_multiple_of(2) { 0 } else { 1 };
            for slot in &mut ss_state[pos..end] {
                if *slot == 2 {
                    *slot = conf;
                }
            }
            // Prolines break alpha-helices into coil/turn
            for k in pos..end {
                if sequence[k] == "PRO" && ss_state[k] == 0 {
                    ss_state[k] = 2;
                }
            }
            pos = end + 3 + rng.usize(5);
            motif += 1;
        }
    }

    let mut ref_data = Reference::new(sequence.clone(), seq_ids);
    for k in 0..length {
        let res = &sequence[k];
        let conf = ss_state[k];
        let (cov_n, cov_ca, cov_cb, cov_c) = delta_covalent(&sequence, k);

        let (d_ca, d_cb, d_c, d_n, d_h) = match (res.as_str(), conf) {
            ("GLY", 0) => (1.5, 0.0, 1.4, -1.2, -0.15),
            ("GLY", 1) => (-0.9, 0.0, -1.1, 1.0, 0.20),
            (_, 0) => (2.6, -0.4, 1.7, -1.5, -0.25),
            (_, 1) => (-1.4, 1.5, -1.3, 1.4, 0.32),
            _ => (-0.2, 0.0, -0.15, 0.0, 0.0),
        };

        if let Some((m, sd)) = priors.get(res, "CA") {
            let val = m + d_ca + cov_ca + rng.normal(0.0, (sd * 0.65).min(1.25));
            ref_data.shifts.insert((k, "CA".to_string()), val);
        }
        if res != "GLY" {
            if let Some((m, sd)) = priors.get(res, "CB") {
                let val = m + d_cb + cov_cb + rng.normal(0.0, (sd * 0.65).min(1.35));
                ref_data.shifts.insert((k, "CB".to_string()), val);
            }
        }
        if let Some((m, sd)) = priors.get(res, "C") {
            let val = m + d_c + cov_c + rng.normal(0.0, (sd * 0.65).min(1.20));
            ref_data.shifts.insert((k, "C".to_string()), val);
        }
        if k > 0 && res != "PRO" {
            if let Some((m, sd)) = priors.get(res, "N") {
                let val = m + d_n + cov_n + rng.normal(0.0, (sd * 0.80).min(3.2));
                ref_data.shifts.insert((k, "N".to_string()), val);
            }
            if let Some((m, sd)) = priors.get(res, "H") {
                let val = (m + d_h + rng.normal(0.0, (sd * 0.80).min(0.58))).clamp(5.9, 10.6);
                ref_data.shifts.insert((k, "H".to_string()), val);
            }
        }
    }

    ref_data
}

#[test]
fn missing_channels_yield_exact_zero_log_bayes_factor_and_hnca_only_works() {
    let priors = Priors::default();
    let seq: Vec<String> = ["MET", "GLY", "ALA", "SER", "PRO", "VAL"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    // s0: completely empty spin system (with use_hn = false so ALL 10 channels are missing)
    // s1: HNCA-only spin system matching Gly2(i): CA(0)=45.2, CA(-1)=55.6 (Met1)
    // s2: CA-less paramagnetic spin system: only C(-1)=176.2 (Ser4->Pro5 skipped, say Ala3: C(-1)=173.8 from Gly2)
    let s_empty = make_spin_system(0, 8.20, 120.0, &[]);
    let s_hnca_gly = make_spin_system(
        1,
        8.30,
        109.2,
        &[("CA", 0, &[(45.2, 1)]), ("CA", -1, &[(55.6, 1)])],
    );
    let s_caless = make_spin_system(2, 8.15, 123.5, &[("C", -1, &[(173.8, 1)])]);

    let cfg_no_hn = BayesSaConfig {
        use_hn: false,
        auto_reference: false,
        forbid_zero_carbons: false,
        forbid_ca_less_excess_roots: false,
        ..BayesSaConfig::default()
    };
    let mats =
        compute_bayes_energy_matrices(&[s_empty, s_hnca_gly, s_caless], &seq, &priors, &cfg_no_hn);

    // 1. All missing channels on s_empty must yield exact 0.0 log-Bayes-factor E_type on assignable non-Pro slots
    for k in [1usize, 2, 3, 5] {
        let et = mats.e_type.get(0, k);
        assert!(
            et.abs() < 1e-12,
            "empty spin system must have exact 0.0 E_type at k={k}, got {et}"
        );
    }
    // N-terminus (k=0) and Proline (k=4) must be strictly forbidden
    assert!(mats.e_type.get(0, 0) >= FORBID_ENERGY * 0.5);
    assert!(mats.e_type.get(0, 4) >= FORBID_ENERGY * 0.5);

    // 2. HNCA-only Gly2 spin system must strongly favor k=1 (Gly2) over Ala3/Ser4/Val6
    let e_gly = mats.e_type.get(1, 1);
    let e_ala = mats.e_type.get(1, 2);
    let e_val = mats.e_type.get(1, 5);
    assert!(
        e_gly < -2.0 && e_gly + 2.5 < e_ala && e_gly + 2.5 < e_val,
        "HNCA-only Gly spin system should favor Gly2 ({e_gly:.3}) over Ala3 ({e_ala:.3}) and Val6 ({e_val:.3})"
    );

    // 3. CA-less paramagnetic root: finite informative E_type when `forbid_ca_less_excess_roots = false`,
    //    and forbidden on non-Gly/non-post-Pro slots when excess roots exist and `forbid_ca_less_excess_roots = true`.
    assert!(mats.e_type.get(2, 2).is_finite() && mats.e_type.get(2, 2) < 10.0);

    let cfg_forbid_caless = BayesSaConfig {
        forbid_ca_less_excess_roots: true,
        auto_reference: false,
        ..BayesSaConfig::default()
    };
    let mats_forbid = compute_bayes_energy_matrices(
        &[
            make_spin_system(0, 8.30, 109.2, &[("CA", 0, &[(45.2, 1)])]),
            make_spin_system(1, 8.10, 121.0, &[("CA", 0, &[(53.1, 1)])]),
            make_spin_system(2, 8.25, 116.5, &[("CA", 0, &[(58.4, 1)])]),
            make_spin_system(3, 8.05, 119.8, &[("CA", 0, &[(61.2, 1)])]),
            make_spin_system(4, 8.15, 123.5, &[("C", -1, &[(173.8, 1)])]),
        ],
        &seq,
        &priors,
        &cfg_forbid_caless,
    );
    assert!(
        mats_forbid.e_unary.get(4, 2) >= FORBID_ENERGY * 0.5,
        "CA-less excess artifact strip should be gated on Ala3 when S > N_assignable and forbid_ca_less_excess_roots=true"
    );
}

#[test]
fn student_t_heavy_tail_outlier_never_vetoes_true_residue() {
    let priors = Priors::default();
    let seq: Vec<String> = ["MET", "ALA", "SER", "VAL"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    // True Ser3 (k=2, preceded by Ala2) with an extreme 16 ppm CB ring-current/mispick outlier (47.5 instead of ~63.8)
    let s_ala = make_spin_system(
        0,
        8.22,
        123.4,
        &[
            ("CA", 0, &[(52.8, 2)]),
            ("CB", 0, &[(19.1, 1)]),
            ("CA", -1, &[(55.6, 1)]),
            ("CB", -1, &[(32.8, 1)]),
            ("C", -1, &[(176.1, 1)]),
        ],
    );
    let s_ser_outlier = make_spin_system(
        1,
        8.31,
        116.2,
        &[
            ("CA", 0, &[(58.3, 2)]),
            ("CB", 0, &[(47.5, 1)]), // ~16 ppm outlier from Ser CB (~63.8 ppm)!
            ("CA", -1, &[(52.8, 2)]),
            ("CB", -1, &[(19.1, 2)]),
            ("C", -1, &[(177.4, 1)]),
        ],
    );
    let s_val = make_spin_system(
        2,
        8.08,
        121.0,
        &[
            ("CA", 0, &[(62.2, 2)]),
            ("CB", 0, &[(32.6, 1)]),
            ("CA", -1, &[(58.3, 2)]),
            ("CB", -1, &[(47.5, 2)]),
            ("C", -1, &[(174.6, 1)]),
        ],
    );

    let cfg = BayesSaConfig {
        auto_reference: false,
        replicas: 16,
        seed: 7,
        ..BayesSaConfig::default()
    };
    let systems = vec![s_ala, s_ser_outlier, s_val];
    let mats = compute_bayes_energy_matrices(&systems, &seq, &priors, &cfg);
    let e_ser = mats.e_type.get(1, 2);
    assert!(
        e_ser < 8.0,
        "Student-t + outlier floor must keep extreme single-channel outlier bounded, got E_type={e_ser:.3}"
    );

    let (res, _rep) = solve_bayes_sa(&systems, &seq, &priors, &cfg);
    assert_eq!(
        res.position,
        vec![1, 2, 3],
        "Connected chain must still assign outlier Ser3 cleanly without veto"
    );
}

#[test]
fn rare_shifts_give_stronger_sequential_link_bayes_factor_than_crowded_band() {
    let priors = Priors::default();
    let seq: Vec<String> = ["MET", "GLY", "ALA", "SER", "VAL"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    // Pair 1 (0 -> 1): rare Gly CA overlap around 44.8 ppm (delta = 0.02 ppm)
    // Pair 2 (2 -> 3): crowded middle-band CA overlap around 56.0 ppm (exact same delta = 0.02 ppm)
    // Pair 3 (4 -> 5): rare Ser/Thr CB overlap around 68.5 ppm (delta = 0.03 ppm)
    // Pair 4 (6 -> 7): crowded middle-band CB overlap around 32.0 ppm (exact same delta = 0.03 ppm)
    let systems = vec![
        make_spin_system(0, 8.30, 109.5, &[("CA", 0, &[(44.80, 2)])]),
        make_spin_system(
            1,
            8.15,
            122.0,
            &[("CA", -1, &[(44.82, 2)]), ("CA", 0, &[(53.0, 1)])],
        ),
        make_spin_system(2, 8.20, 121.0, &[("CA", 0, &[(56.00, 2)])]),
        make_spin_system(
            3,
            8.05,
            119.5,
            &[("CA", -1, &[(56.02, 2)]), ("CA", 0, &[(60.5, 1)])],
        ),
        make_spin_system(
            4,
            8.28,
            115.8,
            &[("CA", 0, &[(61.2, 1)]), ("CB", 0, &[(68.50, 2)])],
        ),
        make_spin_system(
            5,
            7.98,
            121.5,
            &[("CA", 0, &[(54.1, 1)]), ("CB", -1, &[(68.53, 2)])],
        ),
        make_spin_system(
            6,
            8.12,
            120.4,
            &[("CA", 0, &[(55.2, 1)]), ("CB", 0, &[(32.00, 2)])],
        ),
        make_spin_system(
            7,
            8.02,
            118.9,
            &[("CA", 0, &[(57.1, 1)]), ("CB", -1, &[(32.03, 2)])],
        ),
    ];

    let cfg = BayesSaConfig {
        auto_reference: false,
        ..BayesSaConfig::default()
    };
    let mats = compute_bayes_energy_matrices(&systems, &seq, &priors, &cfg);

    let e_link_gly_ca = mats.e_link.get(0, 1);
    let e_link_mid_ca = mats.e_link.get(2, 3);
    assert!(
        e_link_gly_ca < e_link_mid_ca - 0.8,
        "Rare Gly CA match ({e_link_gly_ca:.3}) must give stronger negative E_link than crowded 56 ppm CA ({e_link_mid_ca:.3})"
    );

    let e_link_thr_cb = mats.e_link.get(4, 5);
    let e_link_mid_cb = mats.e_link.get(6, 7);
    assert!(
        e_link_thr_cb < e_link_mid_cb - 0.8,
        "Rare Ser/Thr CB match ({e_link_thr_cb:.3}) must give stronger negative E_link than crowded 32 ppm CB ({e_link_mid_cb:.3})"
    );
}

#[test]
fn locked_assignments_and_state_energy_invariants() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::default();
    let (peaks, truth) = simulate(&ref_data, &SimConfig::realistic_bruker());

    // Pick two valid ground-truth (s, k) pairs to lock
    let locked_pairs: Vec<(usize, usize)> = truth
        .iter()
        .enumerate()
        .filter_map(|(s, &k)| (k >= 1).then_some((s, k as usize)))
        .take(2)
        .collect();
    assert_eq!(locked_pairs.len(), 2);

    let locks: Vec<LockedAssignment> = locked_pairs
        .iter()
        .map(|&(s, k)| LockedAssignment::new(s, k))
        .collect();

    let cfg = BayesSaConfig {
        replicas: 16,
        locked_assignments: locks,
        seed: 42,
        ..BayesSaConfig::default()
    };
    let (systems, res, rep) =
        solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &cfg).expect("solve");

    for &(s_lock, k_lock) in &locked_pairs {
        assert_eq!(res.position[s_lock], k_lock as i32);
        assert!(
            (res.confidence[s_lock] - 1.0).abs() < 1e-12,
            "Locked assignment must retain confidence == 1.0"
        );
    }
    for (s, &p) in res.position.iter().enumerate() {
        if p >= 0 {
            let k = p as usize;
            assert!(
                k >= 1,
                "N-terminus k=0 cannot hold an amide root (spin {s})"
            );
            assert_ne!(
                ref_data.sequence[k], "PRO",
                "Proline at k={k} cannot hold an amide root (spin {s})"
            );
        }
    }

    // Check exact state energy reconstruction matches best_energy
    let mats = compute_bayes_energy_matrices(&systems, &ref_data.sequence, &priors, &cfg);
    let mut at = vec![-1i32; ref_data.sequence.len()];
    for (s, &p) in res.position.iter().enumerate() {
        if p >= 0 {
            at[p as usize] = s as i32;
        }
    }
    let e_check = evaluate_state_energy(&at, &mats).expect("valid assignment state");
    assert!(
        (e_check - rep.best_energy).abs() < 1e-9,
        "evaluate_state_energy ({e_check:.6}) must match report.best_energy ({:.6})",
        rep.best_energy
    );
}

#[test]
fn thermodynamic_cv_schedule_peaks_decelerates_and_beats_fixed_geometric() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::default();
    let (peaks, _truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 3,
            ..SimConfig::realistic_bruker()
        },
    );

    let total_steps = 48 * 1100;
    let adaptive_cfg = BayesSaConfig {
        replicas: 32,
        max_stages: 48,
        steps_per_stage_per_active: 15,
        fixed_total_steps: Some(total_steps),
        cooling_schedule: CoolingScheduleKind::AdaptiveSpecificHeat,
        seed: 11,
        ..BayesSaConfig::default()
    };
    let fixed_cfg = BayesSaConfig {
        cooling_schedule: CoolingScheduleKind::FixedGeometric,
        ..adaptive_cfg.clone()
    };

    let (_sys_a, _res_a, rep_a) =
        solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &adaptive_cfg)
            .expect("adaptive solve");
    let (_sys_f, _res_f, rep_f) =
        solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &fixed_cfg)
            .expect("fixed solve");

    assert_eq!(
        rep_a.total_steps_per_replica, rep_f.total_steps_per_replica,
        "Matched total MCMC step budget required for fair schedule comparison"
    );

    // 1. Non-negative specific heat C_v(T) everywhere and clear interior phase-transition peak T_crit
    for step in &rep_a.cooling_trajectory {
        assert!(
            step.specific_heat_cv >= 0.0 && step.specific_heat_cv.is_finite(),
            "C_v(T) must be finite and non-negative at stage {}",
            step.stage
        );
    }
    assert!(
        rep_a.max_cv > 1.0,
        "Cooperative assignment phase transition must produce clear C_v peak, got {}",
        rep_a.max_cv
    );
    assert!(
        rep_a.t_crit < adaptive_cfg.t_start * 0.90 && rep_a.t_crit > adaptive_cfg.t_min * 1.25,
        "T_crit ({:.3}) must lie strictly inside (t_min={}, t_start={})",
        rep_a.t_crit,
        adaptive_cfg.t_min,
        adaptive_cfg.t_start
    );

    // 2. Verify cooling decelerates across T_crit relative to high-T paramagnetic regime
    let peak_stage = rep_a
        .cooling_trajectory
        .iter()
        .enumerate()
        .find(|(_, s)| (s.temperature - rep_a.t_crit).abs() <= 1e-9)
        .map(|(i, _)| i)
        .unwrap();
    assert!(peak_stage > 0 && peak_stage + 1 < rep_a.cooling_trajectory.len());
    let ratio_high_t =
        rep_a.cooling_trajectory[1].temperature / rep_a.cooling_trajectory[0].temperature;
    let ratio_at_peak = rep_a.cooling_trajectory[peak_stage + 1].temperature
        / rep_a.cooling_trajectory[peak_stage].temperature;
    assert!(
        ratio_at_peak > ratio_high_t,
        "Adaptive schedule must cool more slowly across T_crit (ratio={ratio_at_peak:.4}) than at high T ({ratio_high_t:.4})"
    );

    // 3. Lower or equal final ensemble energy vs fixed geometric cooling at matched total steps
    eprintln!(
        "[thermodynamic C_v(T)] adaptive: mean_E={:.2} best_E={:.2} T_crit={:.2} max_Cv={:.2} | fixed: mean_E={:.2} best_E={:.2}",
        rep_a.final_mean_energy,
        rep_a.best_energy,
        rep_a.t_crit,
        rep_a.max_cv,
        rep_f.final_mean_energy,
        rep_f.best_energy
    );
    assert!(
        rep_a.final_mean_energy >= rep_a.best_energy - 1e-6
            && rep_f.final_mean_energy >= rep_f.best_energy - 1e-6,
        "Mean final energy cannot fall below minimum best energy (adaptive: mean={:.4} best={:.4}, fixed: mean={:.4} best={:.4})",
        rep_a.final_mean_energy,
        rep_a.best_energy,
        rep_f.final_mean_energy,
        rep_f.best_energy
    );
    assert!(
        rep_a.final_mean_energy <= rep_f.final_mean_energy + 1e-6
            && rep_a.best_energy <= rep_f.best_energy + 2.5,
        "Adaptive C_v(T) cooling (mean={:.2}, best={:.2}) must achieve lower or equal final ensemble energy vs FixedGeometric (mean={:.2}, best={:.2})",
        rep_a.final_mean_energy,
        rep_a.best_energy,
        rep_f.final_mean_energy,
        rep_f.best_energy
    );
}

#[test]
fn ubiquitin_realistic_bruker_5_seeds_accuracy_gate() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::default();
    let base_sim = SimConfig::realistic_bruker();

    let mut total_assignable = 0usize;
    let mut total_assigned = 0usize;
    let mut total_correct = 0usize;
    let mut total_hc_assigned = 0usize;
    let mut total_hc_correct = 0usize;

    for seed in 0..5u64 {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                ..base_sim.clone()
            },
        );
        let cfg = BayesSaConfig {
            replicas: 32,
            seed,
            ..BayesSaConfig::default()
        };
        let (_systems, res, rep) =
            solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &cfg).expect("solve");

        // Check automatic 13C offset registration recovers the -2.75 ppm TMS shift
        let d_ca = rep.applied_offsets.get("CA");
        assert!(
            (d_ca - (-2.75)).abs() < 0.75,
            "seed {seed}: expected ~ -2.75 ppm CA offset, got {d_ca:.3}"
        );

        for &p in &res.position {
            if p >= 0 {
                assert_ne!(ref_data.sequence[p as usize], "PRO");
            }
        }

        let m = Metrics {
            assignable: truth.iter().filter(|&&t| t >= 0).count(),
            ..score(&res, &truth, 0.80)
        };
        eprintln!(
            "[ubiquitin realistic seed={seed}] acc={:.2}% ({}/{}) rec={:.2}% ({}/{}) hc_prec={:.2}% ({}/{}) best_E={:.2} T_crit={:.2} stages={}",
            100.0 * m.accuracy(),
            m.correct,
            m.assigned,
            100.0 * m.recall(),
            m.correct,
            m.assignable,
            100.0 * m.hc_precision(),
            m.hc_correct,
            m.hc_assigned,
            rep.best_energy,
            rep.t_crit,
            rep.total_stages
        );
        total_assignable += m.assignable;
        total_assigned += m.assigned;
        total_correct += m.correct;
        total_hc_assigned += m.hc_assigned;
        total_hc_correct += m.hc_correct;
    }

    let overall_acc = (total_correct as f64) / (total_assigned.max(1) as f64);
    let overall_rec = (total_correct as f64) / (total_assignable.max(1) as f64);
    let overall_hc_prec = if total_hc_assigned > 0 {
        (total_hc_correct as f64) / (total_hc_assigned as f64)
    } else {
        1.0
    };
    eprintln!(
        "[ubiquitin realistic 5-seed aggregate] accuracy={:.2}% ({total_correct}/{total_assigned}) recall={:.2}% ({total_correct}/{total_assignable}) hc_precision={:.2}% ({total_hc_correct}/{total_hc_assigned})",
        100.0 * overall_acc,
        100.0 * overall_rec,
        100.0 * overall_hc_prec
    );

    assert!(
        overall_acc >= 0.93,
        "Ubiquitin 5-seed realistic accuracy {:.2}% fell below 93% gate",
        100.0 * overall_acc
    );
    assert!(
        overall_hc_prec >= 0.96,
        "Ubiquitin 5-seed realistic HC precision {:.2}% fell below 96% gate",
        100.0 * overall_hc_prec
    );
}

#[test]
fn sparse_30pct_missing_plus_20pct_noise_and_proline_rich_disordered_gates() {
    let priors = Priors::default();

    // Part A: Sparse 30%-missing-root + 20%-noise-root stress test on Ubiquitin
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ubq = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let sparse_sim = SimConfig {
        p_missing_root: 0.30,
        p_missing_peak: 0.08,
        spurious_rate: 0.10,
        seed: 21,
        ..SimConfig::default()
    };
    let (mut peaks, mut truth) = simulate(&ubq, &sparse_sim);

    // Inject +20% extra spurious noise HSQC roots with random 3D noise strips
    let mut noise_rng = Rng::new(9001);
    let extra_roots = ((truth.len() as f64) * 0.20).ceil() as usize;
    for _ in 0..extra_roots {
        let h = noise_rng.uniform(6.6, 9.6);
        let n = noise_rng.uniform(107.0, 131.0);
        peaks.push(Peak::hsqc(h, n).with_intensity(220.0, None));
        peaks.push(
            Peak::triple("HNCA", h, n, noise_rng.uniform(44.0, 65.0), 1)
                .with_intensity(140.0, None),
        );
        peaks.push(
            Peak::triple("HNCACB", h, n, noise_rng.uniform(18.0, 44.0), -1)
                .with_intensity(110.0, None),
        );
        peaks.push(
            Peak::triple("HNCO", h, n, noise_rng.uniform(171.0, 179.0), 1)
                .with_intensity(160.0, None),
        );
        truth.push(-1);
    }

    let cfg_sparse = BayesSaConfig {
        replicas: 32,
        seed: 21,
        ..BayesSaConfig::default()
    };
    let (_sys_sp, res_sp, _rep_sp) =
        solve_bayes_sa_from_peaks(&peaks, &ubq.sequence, &priors, &cfg_sparse)
            .expect("sparse solve");

    for &p in &res_sp.position {
        if p >= 0 {
            assert_ne!(ubq.sequence[p as usize], "PRO");
        }
    }
    let m_sp = Metrics {
        assignable: truth.iter().filter(|&&t| t >= 0).count(),
        ..score(&res_sp, &truth, 0.80)
    };
    eprintln!(
        "[sparse 30% missing + 20% noise roots] accuracy={:.2}% ({}/{}) hc_prec={:.2}% ({}/{})",
        100.0 * m_sp.accuracy(),
        m_sp.correct,
        m_sp.assigned,
        100.0 * m_sp.hc_precision(),
        m_sp.hc_correct,
        m_sp.hc_assigned
    );
    assert!(
        m_sp.accuracy() >= 0.85,
        "Sparse 30%-missing + 20%-noise accuracy {:.2}% fell below 85% gate",
        100.0 * m_sp.accuracy()
    );

    // Part B: Proline-rich (>= 24% Pro) intrinsically disordered 84-residue sequence
    let idp_ref = build_synthetic_protein_reference(84, true, 77);
    let pro_count = idp_ref
        .sequence
        .iter()
        .filter(|r| r.as_str() == "PRO")
        .count();
    assert!(
        pro_count >= 20,
        "Expected >= 20 Prolines in 84-aa disordered sequence, got {pro_count}"
    );
    let (idp_peaks, idp_truth) = simulate(
        &idp_ref,
        &SimConfig {
            seed: 77,
            ..SimConfig::realistic_bruker()
        },
    );
    let cfg_idp = BayesSaConfig {
        replicas: 32,
        seed: 77,
        ..BayesSaConfig::default()
    };
    let (_sys_idp, res_idp, _rep_idp) =
        solve_bayes_sa_from_peaks(&idp_peaks, &idp_ref.sequence, &priors, &cfg_idp)
            .expect("proline-rich IDP solve");

    let pro_assigned = res_idp
        .position
        .iter()
        .filter(|&&p| p >= 0 && idp_ref.sequence[p as usize] == "PRO")
        .count();
    assert_eq!(pro_assigned, 0, "Zero Proline amide assignments allowed");

    let m_idp = Metrics {
        assignable: idp_truth.iter().filter(|&&t| t >= 0).count(),
        ..score(&res_idp, &idp_truth, 0.80)
    };
    eprintln!(
        "[proline-rich IDP 84aa ({pro_count} PRO)] accuracy={:.2}% ({}/{}) hc_prec={:.2}%",
        100.0 * m_idp.accuracy(),
        m_idp.correct,
        m_idp.assigned,
        100.0 * m_idp.hc_precision()
    );
    assert!(
        m_idp.accuracy() >= 0.85,
        "Proline-rich disordered accuracy {:.2}% fell below 85% gate",
        100.0 * m_idp.accuracy()
    );
}

#[test]
fn large_480aa_protein_cpu_accuracy_gate() {
    let priors = Priors::default();
    let ref_480 = build_synthetic_protein_reference(480, false, 101);
    assert_eq!(ref_480.sequence.len(), 480);

    let (peaks, truth) = simulate(
        &ref_480,
        &SimConfig {
            seed: 101,
            p_missing_root: 0.04,
            p_missing_peak: 0.05,
            spurious_rate: 0.03,
            ..SimConfig::default()
        },
    );

    let t0 = std::time::Instant::now();
    let cfg = BayesSaConfig {
        replicas: 32,
        seed: 101,
        ..BayesSaConfig::default()
    };
    let (_sys, res, rep) =
        solve_bayes_sa_from_peaks(&peaks, &ref_480.sequence, &priors, &cfg).expect("480aa solve");
    let elapsed_s = t0.elapsed().as_secs_f64();

    for &p in &res.position {
        if p >= 0 {
            assert_ne!(ref_480.sequence[p as usize], "PRO");
        }
    }
    let m = Metrics {
        assignable: truth.iter().filter(|&&t| t >= 0).count(),
        ..score(&res, &truth, 0.80)
    };
    eprintln!(
        "[480aa CPU] accuracy={:.2}% ({}/{}) recall={:.2}% ({}/{}) hc_prec={:.2}% ({}/{}) elapsed={elapsed_s:.2}s T_crit={:.2} stages={}",
        100.0 * m.accuracy(),
        m.correct,
        m.assigned,
        100.0 * m.recall(),
        m.correct,
        m.assignable,
        100.0 * m.hc_precision(),
        m.hc_correct,
        m.hc_assigned,
        rep.t_crit,
        rep.total_stages
    );
    assert!(
        m.accuracy() >= 0.88,
        "480aa CPU accuracy {:.2}% fell below 88% gate",
        100.0 * m.accuracy()
    );
    assert!(
        m.recall() >= 0.65,
        "480aa CPU recall {:.2}% ({}/{}) fell below 65% gate",
        100.0 * m.recall(),
        m.correct,
        m.assignable
    );
    assert!(
        m.hc_assigned >= 40 && m.hc_precision() >= 0.95,
        "480aa CPU high-confidence output ({}/{} = {:.2}%) fell below >=40 HC & >=95% HC precision gate",
        m.hc_correct,
        m.hc_assigned,
        100.0 * m.hc_precision()
    );
}
