//! Regression and mathematical verification suite for `crates/nmr-core/src/methyl_solver.rs` (`P1-6`):
//! - Exact maximum-cardinality / maximum-weight bipartite matching initialization (`max_cardinality_max_weight_matching`
//!   and `build_initial_replica_states`) against brute-force enumeration on greedy-trap and Hall-deficient graphs.
//! - Multi-seed (`>= 5` seeds) per-move and combined-move exact stationary Boltzmann distribution
//!   verification (`TVD <= 0.025`) over variable-cardinality $\Omega = \bigcup_{k=0}^{k_{\max}} \Omega_k$
//!   for both $S < M$ (vacant sites) and $S > M$ (artifact / spurious roots).

use std::collections::BTreeMap;

use nmr_core::methyl_solver::{
    assign_methyl_cpu, build_initial_replica_states, evaluate_methyl_raw_score,
    max_cardinality_max_weight_matching, sample_methyl_mcmc_terminal_cpu_with_mask,
    MethylAnnealConfig, MethylContactProblem, ObsAdjEntry, ObsNoeEdge, StructShortAdjEntry,
    StructShortEdge, METHYL_FORBID, METHYL_FORBID_HALF, METHYL_INIT_MIN_UNARY, METHYL_MOVE_ALL,
    METHYL_MOVE_BIRTH_DEATH, METHYL_MOVE_CONTACT, METHYL_MOVE_GEMINAL, METHYL_MOVE_SLOT,
};
use nmr_core::methyl_trosy::MethylClassPosterior;
use nmr_core::rng::Rng;

fn build_variable_cardinality_test_problem(s_len: usize, m_len: usize) -> MethylContactProblem {
    let d_cutoff = 8.5f32;
    let sigma_d = 1.1f32;
    let flat_bottom_d = 0.6f32;
    let coords: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [2.5, 1.7, 0.0],
        [5.2, 0.4, 0.0],
        [5.8, 3.3, 0.0],
        [10.8, 0.2, 0.0],
        [3.1, 4.9, 0.0],
    ];
    let mut d_mat = vec![0.0f32; m_len * m_len];
    for m1 in 0..m_len {
        for m2 in 0..m_len {
            if m1 == m2 {
                continue;
            }
            let dx = coords[m1][0] - coords[m2][0];
            let dy = coords[m1][1] - coords[m2][1];
            let dz = coords[m1][2] - coords[m2][2];
            d_mat[m1 * m_len + m2] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    let mut gem_site = vec![-1i32; m_len];
    if m_len >= 2 {
        gem_site[0] = 1;
        gem_site[1] = 0;
    }
    if m_len >= 4 {
        gem_site[2] = 3;
        gem_site[3] = 2;
    }

    let mut u_mat = vec![METHYL_FORBID; s_len * m_len];
    let mut slots = vec![Vec::new(); s_len];
    for s in 0..s_len {
        for m in 0..m_len {
            let allowed = match s {
                0 => true,
                1 => m != (m_len - 1),
                2 => true,
                3 => m < 3.min(m_len),
                // Excess / artifact root `s >= 4` has unequal sparse slots and negative unary
                // preference so both assigned and unassigned cardinalities carry Boltzmann weight.
                _ => m == 0 || m == (m_len - 1),
            };
            if allowed {
                let u = if s >= 4 {
                    -0.65 + 0.25 * (m as f32)
                } else {
                    0.55 + 0.37 * ((s * 3 + m * 5) % 7) as f32 - 0.28 * (m as f32)
                };
                u_mat[s * m_len + m] = u;
                slots[s].push(m as u32);
            }
        }
    }

    let mut gem_obs_mat = vec![0.0f32; s_len * s_len];
    if s_len >= 2 {
        gem_obs_mat[1] = 1.65;
        gem_obs_mat[s_len] = 1.65;
    }
    if s_len >= 4 {
        gem_obs_mat[2 * s_len + 3] = 1.25;
        gem_obs_mat[3 * s_len + 2] = 1.25;
    }

    let raw_edges: &[(usize, usize, f32, f32)] = &[
        (0, 1, 3.2, 0.90),
        (0, 2, 5.4, 0.75),
        (1, 3, 4.8, 0.65),
        (2, 4, 9.8, 0.80),
    ];
    let mut obs_edges = Vec::new();
    let mut adj_lists: Vec<Vec<ObsAdjEntry>> = vec![Vec::new(); s_len];
    let mut obs_explained_mask = vec![0u32; s_len * s_len];
    for &(s1, s2, d_hat, w) in raw_edges {
        if s1 < s_len && s2 < s_len {
            obs_edges.push(ObsNoeEdge {
                s1: s1 as u32,
                s2: s2 as u32,
                d_hat,
                weight: w,
            });
            obs_explained_mask[s1 * s_len + s2] = 1;
            obs_explained_mask[s2 * s_len + s1] = 1;
            adj_lists[s1].push(ObsAdjEntry {
                peer_s: s2 as u32,
                d_hat,
                weight: w,
            });
            adj_lists[s2].push(ObsAdjEntry {
                peer_s: s1 as u32,
                d_hat,
                weight: w,
            });
        }
    }
    let mut obs_adj_offsets = Vec::with_capacity(s_len + 1);
    let mut obs_adj_entries = Vec::new();
    obs_adj_offsets.push(0);
    for list in &adj_lists {
        obs_adj_entries.extend_from_slice(list);
        obs_adj_offsets.push(obs_adj_entries.len() as u32);
    }

    let mut struct_short_edges = Vec::new();
    let mut struct_short_lists: Vec<Vec<StructShortAdjEntry>> = vec![Vec::new(); m_len];
    let mut struct_cut_lists: Vec<Vec<u32>> = vec![Vec::new(); m_len];
    for m1 in 0..m_len {
        for m2 in (m1 + 1)..m_len {
            let d = d_mat[m1 * m_len + m2];
            if d <= d_cutoff {
                struct_cut_lists[m1].push(m2 as u32);
                struct_cut_lists[m2].push(m1 as u32);
            }
            if d <= 5.5 {
                let lam = 0.65f32;
                struct_short_edges.push(StructShortEdge {
                    m1: m1 as u32,
                    m2: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m1].push(StructShortAdjEntry {
                    peer_m: m2 as u32,
                    lambda_miss: lam,
                });
                struct_short_lists[m2].push(StructShortAdjEntry {
                    peer_m: m1 as u32,
                    lambda_miss: lam,
                });
            }
        }
    }
    let mut struct_short_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_short_entries = Vec::new();
    struct_short_offsets.push(0);
    for list in &struct_short_lists {
        struct_short_entries.extend_from_slice(list);
        struct_short_offsets.push(struct_short_entries.len() as u32);
    }
    let mut struct_cut_offsets = Vec::with_capacity(m_len + 1);
    let mut struct_cut_peers = Vec::new();
    struct_cut_offsets.push(0);
    for list in &struct_cut_lists {
        struct_cut_peers.extend_from_slice(list);
        struct_cut_offsets.push(struct_cut_peers.len() as u32);
    }

    MethylContactProblem {
        s_len,
        m_len,
        d_cutoff,
        sigma_d,
        flat_bottom_d,
        norm_inv: 1.0 / (s_len.max(1) as f64),
        u_mat,
        slots,
        d_mat,
        gem_site,
        gem_obs_mat,
        obs_explained_mask,
        obs_edges,
        obs_adj_offsets,
        obs_adj_entries,
        struct_short_edges,
        struct_short_offsets,
        struct_short_entries,
        struct_cut_offsets,
        struct_cut_peers,
        class_posteriors: vec![
            MethylClassPosterior {
                probs: [1.0 / 6.0; 6]
            };
            s_len
        ],
    }
}

fn enumerate_boltzmann_omega(
    prob: &MethylContactProblem,
    fixed_k: Option<usize>,
    temp: f64,
) -> BTreeMap<Vec<i32>, f64> {
    fn rec(
        s: usize,
        assigned_so_far: usize,
        fixed_k: Option<usize>,
        prob: &MethylContactProblem,
        used_m: &mut [bool],
        cur: &mut Vec<i32>,
        states: &mut Vec<Vec<i32>>,
    ) {
        if s == prob.s_len {
            if fixed_k.is_none_or(|k| assigned_so_far == k) {
                states.push(cur.clone());
            }
            return;
        }
        let rem = prob.s_len - s;
        if let Some(k) = fixed_k {
            if assigned_so_far + rem < k {
                return;
            }
        }
        if fixed_k.is_none_or(|k| assigned_so_far < k) {
            for &m_u32 in &prob.slots[s] {
                let m = m_u32 as usize;
                if !used_m[m] {
                    used_m[m] = true;
                    cur.push(m as i32);
                    rec(
                        s + 1,
                        assigned_so_far + 1,
                        fixed_k,
                        prob,
                        used_m,
                        cur,
                        states,
                    );
                    cur.pop();
                    used_m[m] = false;
                }
            }
        }
        if fixed_k.is_none_or(|k| assigned_so_far + (rem - 1) >= k) {
            cur.push(-1);
            rec(s + 1, assigned_so_far, fixed_k, prob, used_m, cur, states);
            cur.pop();
        }
    }

    let mut states = Vec::new();
    let mut used_m = vec![false; prob.m_len];
    let mut cur = Vec::with_capacity(prob.s_len);
    rec(0, 0, fixed_k, prob, &mut used_m, &mut cur, &mut states);
    assert!(!states.is_empty());

    let energies: Vec<f64> = states
        .iter()
        .map(|st| evaluate_methyl_raw_score(prob, st))
        .collect();
    let max_e = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = energies
        .iter()
        .map(|&e| ((e - max_e) / temp).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    states
        .into_iter()
        .zip(weights)
        .map(|(st, w)| (st, w / z))
        .collect()
}

fn compute_tvd(exact: &BTreeMap<Vec<i32>, f64>, samples: &[Vec<i32>]) -> f64 {
    let mut counts: BTreeMap<&[i32], usize> = BTreeMap::new();
    for st in samples {
        assert!(
            exact.contains_key(st),
            "MCMC sampled invalid or out-of-manifold state: {st:?}"
        );
        *counts.entry(st.as_slice()).or_insert(0) += 1;
    }
    let inv_n = 1.0 / (samples.len() as f64);
    0.5 * exact
        .iter()
        .map(|(st, &p_true)| {
            let p_emp = (*counts.get(st.as_slice()).unwrap_or(&0) as f64) * inv_n;
            (p_emp - p_true).abs()
        })
        .sum::<f64>()
}

fn brute_force_max_card_max_weight(
    s_len: usize,
    m_len: usize,
    row_edges: &[Vec<(usize, f64)>],
) -> (usize, f64) {
    fn dfs(
        s: usize,
        s_len: usize,
        row_edges: &[Vec<(usize, f64)>],
        used_m: &mut [bool],
        card: usize,
        weight: f64,
        best: &mut (usize, f64),
    ) {
        if s == s_len {
            if card > best.0 || (card == best.0 && weight > best.1 + 1e-12) {
                *best = (card, weight);
            }
            return;
        }
        for &(m, w) in &row_edges[s] {
            if !used_m[m] {
                used_m[m] = true;
                dfs(s + 1, s_len, row_edges, used_m, card + 1, weight + w, best);
                used_m[m] = false;
            }
        }
        dfs(s + 1, s_len, row_edges, used_m, card, weight, best);
    }
    let mut used = vec![false; m_len];
    let mut best = (0usize, 0.0f64);
    dfs(0, s_len, row_edges, &mut used, 0, 0.0, &mut best);
    best
}

#[test]
fn p1_6_max_cardinality_max_weight_matching_augmenting_path_optimality() {
    // 1. Canonical greedy-trap bipartite graph:
    // Root 0 prefers site 0 (+100.0) over site 1 (+1.0); Root 1 can ONLY match site 0 (+5.0).
    // Greedy maximal matching assigns 0 -> 0 and leaves 1 unassigned (cardinality 1, weight 100),
    // whereas maximum-cardinality matching assigns 0 -> 1, 1 -> 0 (cardinality 2, weight 6.0).
    let trap_edges = vec![vec![(0, 100.0), (1, 1.0)], vec![(0, 5.0)]];
    let (at, pos) = max_cardinality_max_weight_matching(2, 2, &trap_edges);
    assert_eq!(pos, vec![1, 0]);
    assert_eq!(at, vec![1, 0]);

    // 2. Multi-hop augmenting-path chain (`0 -> {0,1}`, `1 -> {1,2}`, `2 -> {2,3}`, `3 -> {0}`)
    let chain_edges = vec![
        vec![(0, 20.0), (1, 2.0)],
        vec![(1, 20.0), (2, 2.0)],
        vec![(2, 20.0), (3, 2.0)],
        vec![(0, 3.0)],
    ];
    let (_, pos_chain) = max_cardinality_max_weight_matching(4, 4, &chain_edges);
    assert_eq!(pos_chain, vec![1, 2, 3, 0]);

    // 3. 64 randomized sparse instances (`S < M`, `S == M`, `S > M`, and Hall-deficient subsets)
    // verified against exact brute-force search.
    let mut rng = Rng::new(20261113);
    for trial in 0..64usize {
        let s_len = 3 + (trial % 5);
        let m_len = 3 + ((trial * 3 + 1) % 5);
        let mut row_edges = vec![Vec::new(); s_len];
        for edges in &mut row_edges {
            for m in 0..m_len {
                if rng.f64() < 0.45 {
                    let w = -8.0 + 20.0 * rng.f64();
                    edges.push((m, w));
                }
            }
        }
        let (exp_card, exp_w) = brute_force_max_card_max_weight(s_len, m_len, &row_edges);
        let (at, pos) = max_cardinality_max_weight_matching(s_len, m_len, &row_edges);
        let mut got_card = 0usize;
        let mut got_w = 0.0f64;
        for (s, &m_i) in pos.iter().enumerate() {
            if m_i >= 0 {
                let m = m_i as usize;
                assert_eq!(at[m], s as i32, "at/pos bijection broken on trial {trial}");
                let &(_, w) = row_edges[s]
                    .iter()
                    .find(|&&(em, _)| em == m)
                    .expect("matched disallowed edge");
                got_card += 1;
                got_w += w;
            }
        }
        assert_eq!(
            got_card, exp_card,
            "trial {trial}: cardinality {got_card} != k_max {exp_card}"
        );
        assert!(
            (got_w - exp_w).abs() < 1.0e-8,
            "trial {trial}: weight {got_w:.9} != max weight {exp_w:.9}"
        );
    }

    // 4. Verify `build_initial_replica_states` always initializes at exact `k_max` even when
    // greedy order would strand constrained roots.
    let mut prob = build_variable_cardinality_test_problem(3, 3);
    prob.u_mat = vec![
        50.0,
        0.5,
        METHYL_FORBID,
        0.5,
        50.0,
        0.5,
        50.0,
        METHYL_FORBID,
        METHYL_FORBID,
    ];
    prob.slots = vec![vec![0, 1], vec![0, 1, 2], vec![0]];
    let reps = build_initial_replica_states(&prob, 32, 999);
    for (idx, (_, pos, _)) in reps.iter().enumerate() {
        let k = pos.iter().filter(|&&m| m >= 0).count();
        assert_eq!(
            k, 3,
            "replica {idx} failed to reach exact maximum cardinality k_max=3: {pos:?}"
        );
    }
}

#[test]
fn p1_6_cpu_variable_cardinality_multi_seed_per_move_and_combined_boltzmann_tvd() {
    let temp = 2.6;
    let seeds: [u64; 5] = [101_001, 202_002, 303_003, 404_004, 505_005];
    let n_samples = 48_000usize;

    let apply_slots = |prob: &mut MethylContactProblem, allowed: &[&[u32]]| {
        for (s, &sl) in allowed.iter().enumerate() {
            prob.slots[s] = sl.to_vec();
            for m in 0..prob.m_len {
                if !sl.contains(&(m as u32)) {
                    prob.u_mat[s * prob.m_len + m] = METHYL_FORBID;
                } else if prob.u_mat[s * prob.m_len + m] <= METHYL_FORBID_HALF {
                    prob.u_mat[s * prob.m_len + m] =
                        0.45 + 0.31 * ((s * 3 + m * 5) % 7) as f32 - 0.22 * (m as f32);
                }
            }
        }
    };

    let mut prob_vacant = build_variable_cardinality_test_problem(4, 5); // S < M (vacant sites)
    apply_slots(
        &mut prob_vacant,
        &[&[0, 1, 2, 4], &[0, 1, 3], &[0, 2, 3, 4], &[1, 3]],
    );
    let mut prob_artifact = build_variable_cardinality_test_problem(5, 4); // S > M (artifact root)
    apply_slots(
        &mut prob_artifact,
        &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3], &[0, 3]],
    );
    for u in &mut prob_vacant.u_mat {
        if *u > METHYL_FORBID * 0.5 {
            *u -= 0.85;
        }
    }
    for s in 0..prob_artifact.s_len {
        for m in 0..prob_artifact.m_len {
            let v = &mut prob_artifact.u_mat[s * prob_artifact.m_len + m];
            if *v > METHYL_FORBID * 0.5 {
                *v -= if s == 4 { 1.60 } else { 0.75 };
            }
        }
    }

    let exact_vacant_var = enumerate_boltzmann_omega(&prob_vacant, None, temp);
    let exact_artifact_var = enumerate_boltzmann_omega(&prob_artifact, None, temp);
    let exact_vacant_fixed = enumerate_boltzmann_omega(&prob_vacant, Some(4), temp);
    let exact_artifact_fixed = enumerate_boltzmann_omega(&prob_artifact, Some(4), temp);

    // Verify multiple cardinalities carry non-trivial probability mass in exact_vacant_var & exact_artifact_var
    let mut card_mass_vacant = [0.0f64; 5];
    for (st, &p) in &exact_vacant_var {
        let k = st.iter().filter(|&&m| m >= 0).count();
        card_mass_vacant[k] += p;
    }
    assert!(
        card_mass_vacant[3] > 0.02 && card_mass_vacant[4] > 0.05,
        "test problem must exercise multiple cardinalities in Omega = U_k Omega_k: {card_mass_vacant:?}"
    );

    // 1. Per-move and paired-move variable-cardinality Boltzmann TVD <= 0.025 across 5 seeds
    for (label, prob, exact, mask, steps) in [
        (
            "Move4_BirthDeath_Only (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_var,
            METHYL_MOVE_BIRTH_DEATH,
            2_000usize,
        ),
        (
            "Move4_BirthDeath_Only (S=5>M=4)",
            &prob_artifact,
            &exact_artifact_var,
            METHYL_MOVE_BIRTH_DEATH,
            2_200usize,
        ),
        (
            "Move1+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_var,
            METHYL_MOVE_SLOT | METHYL_MOVE_BIRTH_DEATH,
            900usize,
        ),
        (
            "Move2+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_var,
            METHYL_MOVE_CONTACT | METHYL_MOVE_BIRTH_DEATH,
            900usize,
        ),
        (
            "Move3_Geminal+Move4 (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_var,
            METHYL_MOVE_GEMINAL | METHYL_MOVE_BIRTH_DEATH,
            2_000usize,
        ),
        (
            "Move1_Slot_FixedK (S=5>M=4)",
            &prob_artifact,
            &exact_artifact_fixed,
            METHYL_MOVE_SLOT,
            800usize,
        ),
        (
            "Move2_Contact_FixedK (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_fixed,
            METHYL_MOVE_CONTACT,
            800usize,
        ),
        (
            "Combined_AllMoves_0xF (S=4<M=5)",
            &prob_vacant,
            &exact_vacant_var,
            METHYL_MOVE_ALL,
            900usize,
        ),
        (
            "Combined_AllMoves_0xF (S=5>M=4)",
            &prob_artifact,
            &exact_artifact_var,
            METHYL_MOVE_ALL,
            900usize,
        ),
    ] {
        let mut max_tvd = 0.0f64;
        let mut mean_tvd = 0.0f64;
        for &sd in &seeds {
            let samples =
                sample_methyl_mcmc_terminal_cpu_with_mask(prob, temp, steps, n_samples, sd, mask);
            let tvd = compute_tvd(exact, &samples);
            assert!(
                tvd <= 0.025,
                "[CPU {label} seed={sd}] Boltzmann TVD {tvd:.4} > 0.025 (|Omega|={})",
                exact.len()
            );
            max_tvd = max_tvd.max(tvd);
            mean_tvd += tvd / (seeds.len() as f64);
        }
        println!(
            "[P1-6 CPU 5-Seed Boltzmann {label}] |Omega|={} | mean_TVD={mean_tvd:.4} | max_TVD={max_tvd:.4}",
            exact.len()
        );
    }
}

#[test]
fn p1_6_artifact_root_leaves_via_birth_death_without_displacing_true_sites() {
    // S = 4 roots (3 true well-matched roots + 1 spurious artifact root) on M = 4 sites (1 vacant true site):
    // Under fixed-k MCMC, `k = min(4, 4) = 4` forced the artifact root onto the 4th vacant site
    // despite strongly negative unary/NOE log-odds. Variable-cardinality Move 4 unassigns the artifact root!
    let mut prob = build_variable_cardinality_test_problem(4, 4);
    prob.u_mat.fill(METHYL_FORBID);
    for s in 0..3 {
        prob.slots[s] = vec![0, 1, 2, 3];
        for m in 0..4 {
            prob.u_mat[s * 4 + m] = if s == m { 9.5 } else { -2.5 };
        }
    }
    // Root 3 is a spurious artifact peak with negative log-odds on every site
    prob.slots[3] = vec![0, 1, 2, 3];
    for m in 0..4 {
        prob.u_mat[3 * 4 + m] = -6.0;
    }

    let res = assign_methyl_cpu(
        &prob,
        &MethylAnnealConfig {
            replicas: 32,
            steps_per_root: 180,
            seed: 42,
            ..MethylAnnealConfig::default()
        },
    );
    assert_eq!(res.position[0], 0);
    assert_eq!(res.position[1], 1);
    assert_eq!(res.position[2], 2);
    assert_eq!(
        res.position[3], -1,
        "spurious negative-log-odds artifact root must unassign via Move 4 even when a vacant site exists: {:?}",
        res.position
    );
}

fn ln_gamma_lanczos(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut acc = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// Regularized upper incomplete gamma `Q(a, x) = P(Chi2_{2a} >= 2x)` via series / Lentz continued fraction.
fn chi2_survival_prob(stat: f64, df: usize) -> f64 {
    if stat <= 0.0 {
        return 1.0;
    }
    let a = 0.5 * (df as f64);
    let x = 0.5 * stat;
    let gln = ln_gamma_lanczos(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..512 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * 1.0e-14 {
                break;
            }
        }
        let p_lower = (sum * (-x + a * x.ln() - gln).exp()).clamp(0.0, 1.0);
        (1.0 - p_lower).clamp(0.0, 1.0)
    } else {
        let tiny = 1.0e-300_f64;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b.max(tiny);
        let mut h = d;
        for i in 1..=512 {
            let an = -(i as f64) * ((i as f64) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= 1.0e-14 {
                break;
            }
        }
        ((-x + a * x.ln() - gln).exp() * h).clamp(0.0, 1.0)
    }
}

/// Compute pooled multinomial log-likelihood ratio `G = 2 * sum O_j ln(O_j / E_j)` and Pearson
/// `chi2 = sum (O_j - E_j)^2 / E_j` (pooling only rare tail states with `E < min_exp` so Cochran's
/// `E_j >= 5` multinomial asymptotic validity condition holds strictly).
fn multinomial_g_and_chi2(
    exact: &BTreeMap<Vec<i32>, f64>,
    samples: &[Vec<i32>],
    min_exp: f64,
) -> (f64, f64, usize) {
    let mut counts: BTreeMap<&[i32], usize> = BTreeMap::new();
    for st in samples {
        assert!(
            exact.contains_key(st),
            "MCMC sampled invalid or out-of-manifold state: {st:?}"
        );
        *counts.entry(st.as_slice()).or_insert(0) += 1;
    }

    let n = samples.len() as f64;
    let mut ordered: Vec<(usize, f64)> = exact
        .iter()
        .map(|(st, &p)| (*counts.get(st.as_slice()).unwrap_or(&0), n * p))
        .collect();
    // Sort ascending by expected count so any rare tail cells pool deterministically first
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut bins: Vec<(f64, f64)> = Vec::with_capacity(ordered.len());
    let mut pool_o = 0.0f64;
    let mut pool_e = 0.0f64;
    for (o_u, e) in ordered {
        pool_o += o_u as f64;
        pool_e += e;
        if pool_e >= min_exp {
            bins.push((pool_o, pool_e));
            pool_o = 0.0;
            pool_e = 0.0;
        }
    }
    if pool_e > 0.0 {
        if let Some(last) = bins.last_mut() {
            last.0 += pool_o;
            last.1 += pool_e;
        } else {
            bins.push((pool_o, pool_e));
        }
    }
    assert!(bins.len() >= 2, "require >= 2 multinomial bins");

    let mut g_stat = 0.0f64;
    let mut chi2_stat = 0.0f64;
    for &(o, e) in &bins {
        if o > 0.0 {
            g_stat += 2.0 * o * (o / e).ln();
        }
        let diff = o - e;
        chi2_stat += (diff * diff) / e;
    }
    (g_stat, chi2_stat, bins.len() - 1)
}

#[test]
fn p2_near_forbidden_init_guard_and_s4_m4_209_state_multinomial_g_test() {
    // 1. Verify `build_initial_replica_states` NEVER forces near-forbidden edges
    //    (`METHYL_FORBID_HALF < u <= METHYL_INIT_MIN_UNARY`, e.g. `-25.0` or `-5_000.0`)
    //    onto vacant sites via `max_cardinality_max_weight_matching`.
    let mut prob_near_forbid = build_variable_cardinality_test_problem(3, 3);
    prob_near_forbid.u_mat.fill(METHYL_FORBID);
    for s in 0..2 {
        prob_near_forbid.slots[s] = vec![0, 1, 2];
        for m in 0..3 {
            prob_near_forbid.u_mat[s * 3 + m] = if s == m { 8.0 } else { 0.5 };
        }
    }
    // Root 2 only has near-forbidden outlier edges (`u = -25.0 <= METHYL_INIT_MIN_UNARY`,
    // still `> METHYL_FORBID_HALF` so present in `slots[2]`).
    prob_near_forbid.slots[2] = vec![0, 1, 2];
    for m in 0..3 {
        prob_near_forbid.u_mat[2 * 3 + m] = METHYL_INIT_MIN_UNARY - 10.0;
    }
    let init_reps = build_initial_replica_states(&prob_near_forbid, 32, 2026_1113);
    for (ri, (_, pos, _)) in init_reps.iter().enumerate() {
        assert!(
            pos[0] >= 0 && pos[1] >= 0,
            "replica {ri} must still match compatible roots 0 & 1: {pos:?}"
        );
        assert_eq!(
            pos[2],
            -1,
            "replica {ri} must NOT force near-forbidden edge (u = {:.1}) onto vacant site: {pos:?}",
            METHYL_INIT_MIN_UNARY - 10.0
        );
    }

    // 2. Exact `S = 4, M = 4` (`|Omega| = sum_{k=0}^4 C(4,k)^2 k! = 209` states) Boltzmann
    //    multinomial G-test & Pearson chi-squared verification across 3 independent seeds.
    let mut prob_209 = build_variable_cardinality_test_problem(4, 4);
    for s in 0..4 {
        prob_209.slots[s] = vec![0, 1, 2, 3];
        for m in 0..4 {
            prob_209.u_mat[s * 4 + m] =
                -0.45 + 0.29 * ((s * 3 + m * 5) % 7) as f32 - 0.21 * (m as f32);
        }
    }
    let temp = 3.2;
    let exact_209 = enumerate_boltzmann_omega(&prob_209, None, temp);
    assert_eq!(
        exact_209.len(),
        209,
        "complete S=4, M=4 partial-matching space must have exact cardinality 209"
    );

    let n_samples = 100_000usize;
    let steps = 950usize;
    for &seed in &[314_159u64, 271_828, 161_803] {
        let samples = sample_methyl_mcmc_terminal_cpu_with_mask(
            &prob_209,
            temp,
            steps,
            n_samples,
            seed,
            METHYL_MOVE_ALL,
        );
        let tvd = compute_tvd(&exact_209, &samples);
        let (g_stat, chi2_stat, df) = multinomial_g_and_chi2(&exact_209, &samples, 8.0);
        let p_g = chi2_survival_prob(g_stat, df);
        let p_chi2 = chi2_survival_prob(chi2_stat, df);
        println!(
            "[P2 S=4,M=4 209-State Boltzmann seed={seed}] |Omega|={} df={df} G={g_stat:.2} (p={p_g:.4}) chi2={chi2_stat:.2} (p={p_chi2:.4}) TVD={tvd:.4}",
            exact_209.len()
        );
        assert!(
            df >= 195,
            "S=4, M=4 Boltzmann test should exercise >= 196 unpooled state bins out of 209, got df={df}"
        );
        assert!(
            p_g > 0.005 && p_chi2 > 0.005,
            "S=4, M=4 209-state Boltzmann G-test/chi2 failed at seed={seed}: G={g_stat:.2} (p={p_g:.5}), chi2={chi2_stat:.2} (p={p_chi2:.5}), df={df}"
        );
        assert!(
            tvd <= 0.025,
            "S=4, M=4 209-state Boltzmann TVD {tvd:.4} > 0.025 at seed={seed}"
        );

        // Negative control: verify G-test rejects a slightly skewed reference distribution (`p < 1e-6`)
        if seed == 314_159 {
            let mut skewed = exact_209.clone();
            let first_key = skewed.keys().next().unwrap().clone();
            *skewed.get_mut(&first_key).unwrap() += 0.025;
            let z_skew: f64 = skewed.values().sum();
            for v in skewed.values_mut() {
                *v /= z_skew;
            }
            let (g_bad, _, df_bad) = multinomial_g_and_chi2(&skewed, &samples, 8.0);
            let p_bad = chi2_survival_prob(g_bad, df_bad);
            assert!(
                p_bad < 1.0e-6,
                "G-test power check must strongly reject skewed Boltzmann null: G={g_bad:.2}, p={p_bad:.3e}"
            );
        }
    }
}
