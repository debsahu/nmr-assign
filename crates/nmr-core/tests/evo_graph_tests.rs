//! Integration and benchmark tests for `nmr_core::evo_graph`:
//! - Sequence-derived stoichiometry peak budgeting and spectral artifact suppression
//! - Two-pass chemical-shift reference offset estimation (Mixture -> Anchor)
//! - Hybrid evolutionary sequential-fragment crossover + local simulated annealing
//! - 2-layer sequence-graph self-consistency
//! - Synthetic ubiquitin (7-exp and 5-exp Bruker-like subset without `HNCA`)
//! - Optional local real Bruker 5-spectrum ubiquitin benchmark (5 seeds, sequence-only)

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use nmr_core::evaluate::{compare_shifts, score, to_reference, Metrics};
use nmr_core::evo_graph::{
    apply_offsets_to_priors, assign_evo_graph, assign_evo_graph_from_peaks,
    build_stoichiometric_spin_systems, estimate_mixture_reference_offsets,
    expected_experiment_peaks, filter_peaks_stoichiometric, refine_anchor_reference_offsets,
    EvoGraphConfig, ReferenceOffsets,
};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

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
        if p.join("ubiquitin").is_dir() {
            return Some(p);
        }
    }
    let ws_data = workspace_root().join("data");
    if ws_data.join("ubiquitin").is_dir() {
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
    if common_data.join("ubiquitin").is_dir() {
        Some(common_data)
    } else {
        None
    }
}

#[test]
fn sequence_stoichiometry_counts_match_pathway_rules() {
    // Sequence: M A P G N K (6 residues: Met1, Ala2, Pro3, Gly4, Asn5, Lys6)
    // Assignable non-Pro amides k>=1: Ala2, Gly4, Asn5, Lys6 => n_amide = 4
    // Asn/Gln count = 1 (Asn5)
    // Non-Gly intra among assignable amides: Ala2, Asn5, Lys6 => 3
    // Non-Gly i-1 among assignable amides:
    //   k=1(Ala2, prev=Met1 non-Gly), k=3(Gly4, prev=Pro3 non-Gly),
    //   k=4(Asn5, prev=Gly4 GLY), k=5(Lys6, prev=Asn5 non-Gly) => 3
    let seq: Vec<String> = ["MET", "ALA", "PRO", "GLY", "ASN", "LYS"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    assert_eq!(expected_experiment_peaks("HSQC", &seq), 4 + 2);
    assert_eq!(expected_experiment_peaks("HNCO", &seq), 4 + 2);
    assert_eq!(expected_experiment_peaks("HNCACO", &seq), 8);
    assert_eq!(expected_experiment_peaks("CBCACONH", &seq), 4 + 3);
    assert_eq!(expected_experiment_peaks("HNCACB+", &seq), 8);
    assert_eq!(expected_experiment_peaks("HNCACB-", &seq), 3 + 3);
    assert_eq!(expected_experiment_peaks("HNCACB", &seq), 8 + 6);
}

#[test]
fn filter_peaks_stoichiometric_suppresses_nh2_sidelobes_and_preserves_unheighted() {
    let seq: Vec<String> = ["MET", "ALA", "ASN", "VAL"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let cfg = EvoGraphConfig::default();

    // Legacy peaks without heights must pass through untouched
    let legacy = vec![
        Peak::hsqc(8.20, 121.0),
        Peak::hsqc(4.75, 110.0),
        Peak::triple("HNCO", 8.20, 121.0, 175.2, 1),
    ];
    let kept_legacy = filter_peaks_stoichiometric(&legacy, &seq, &cfg);
    assert_eq!(kept_legacy.len(), legacy.len());

    // Heighted peaks with water artifact, F1 sinc sidelobes, and paired Asn NH2 doublet
    let noisy = vec![
        // True backbone amides
        Peak::hsqc(8.35, 122.4).with_intensity(1000.0, None),
        Peak::hsqc(8.10, 118.6).with_intensity(950.0, None),
        Peak::hsqc(7.92, 120.8).with_intensity(900.0, None),
        // Water-region artifact (outside 5.5..=11.5 ppm)
        Peak::hsqc(4.76, 115.0).with_intensity(5000.0, None),
        // Indirect 15N sinc sidelobe of (8.35, 122.4)
        Peak::hsqc(8.352, 123.1).with_intensity(60.0, None),
        // Paired Asn NH2 sidechain doublet (same 15N = 112.4, dH = 0.68, no intra CA)
        Peak::hsqc(7.55, 112.40).with_intensity(700.0, None),
        Peak::hsqc(6.87, 112.43).with_intensity(680.0, None),
        // True 3D peaks on backbone strips
        Peak::triple("HNCACB", 8.35, 122.4, 53.2, 1).with_intensity(500.0, None),
        Peak::triple("HNCACB", 8.35, 122.4, 19.1, -1).with_intensity(420.0, None),
        Peak::triple("CBCACONH", 8.35, 122.4, 55.4, 1).with_intensity(480.0, None),
        Peak::triple("HNCO", 8.35, 122.4, 176.1, 1).with_intensity(800.0, None),
        // F1 sinc sidelobe on HNCO strip
        Peak::triple("HNCO", 8.35, 122.4, 176.8, 1).with_intensity(75.0, None),
        // Unphysical 13C artifact in HNCACB(+) < 35 ppm
        Peak::triple("HNCACB", 8.35, 122.4, 19.2, 1).with_intensity(55.0, None),
        Peak::triple("HNCACB", 8.10, 118.6, 54.0, 1).with_intensity(480.0, None),
        Peak::triple("CBCACONH", 8.10, 118.6, 53.2, 1).with_intensity(450.0, None),
        Peak::triple("HNCO", 8.10, 118.6, 177.2, 1).with_intensity(760.0, None),
        Peak::triple("HNCACB", 7.92, 120.8, 62.1, 1).with_intensity(510.0, None),
        Peak::triple("CBCACONH", 7.92, 120.8, 54.0, 1).with_intensity(460.0, None),
        Peak::triple("HNCO", 7.92, 120.8, 174.8, 1).with_intensity(790.0, None),
        // NH2 HNCO peaks (no intra CA!)
        Peak::triple("HNCO", 7.55, 112.40, 176.9, 1).with_intensity(600.0, None),
        Peak::triple("HNCO", 6.87, 112.43, 176.9, 1).with_intensity(580.0, None),
    ];

    let filtered = filter_peaks_stoichiometric(&noisy, &seq, &cfg);
    let kept_hsqc: Vec<&Peak> = filtered.iter().filter(|p| p.exp == "HSQC").collect();
    assert_eq!(
        kept_hsqc.len(),
        3,
        "only the 3 true backbone roots should survive"
    );
    assert!(kept_hsqc.iter().all(|p| p.h > 7.8 && p.n > 118.0));
    let kept_hnco: Vec<&Peak> = filtered.iter().filter(|p| p.exp == "HNCO").collect();
    assert_eq!(
        kept_hnco.len(),
        3,
        "HNCO sidelobe and NH2 strips must be removed"
    );
}

#[test]
fn reference_offset_estimation_recovers_tms_shift_and_preserves_dss_zero() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::provisional();

    let (peaks_dss, _truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 7,
            ..Default::default()
        },
    );
    let systems_dss = build_stoichiometric_spin_systems(&peaks_dss, &Tolerances::default(), 3);
    let off_dss = estimate_mixture_reference_offsets(&systems_dss, &ref_data.sequence, &priors);
    assert!(
        off_dss.d_ca.abs() <= 0.35 && off_dss.d_cb.abs() <= 0.35 && off_dss.d_c.abs() <= 0.35,
        "DSS-referenced synthetic data should stay near 0 ppm, got {off_dss:?}"
    );

    // Shift all 13C peaks by -2.75 ppm (TMS-like offset) and 15N by +0.45 ppm
    let shifted_peaks: Vec<Peak> = peaks_dss
        .into_iter()
        .map(|mut p| {
            p.n += 0.45;
            if let Some(ref mut c) = p.c {
                *c -= 2.75;
            }
            p
        })
        .collect();

    let cfg = EvoGraphConfig {
        seed: 7,
        ..Default::default()
    };
    let (systems_tms, res_tms, rep_tms) =
        assign_evo_graph_from_peaks(&shifted_peaks, &ref_data.sequence, &priors, &cfg);

    assert!(
        (rep_tms.pass1_offsets.d_ca - (-2.75)).abs() <= 0.55,
        "Pass-1 mixture offset should capture ~-2.75 ppm 13C shift, got {:?}",
        rep_tms.pass1_offsets
    );
    assert!(
        (rep_tms.final_offsets.d_ca - (-2.75)).abs() <= 0.35
            && (rep_tms.final_offsets.d_cb - (-2.75)).abs() <= 0.35
            && (rep_tms.final_offsets.d_c - (-2.75)).abs() <= 0.35,
        "Pass-2 anchor offset should refine near -2.75 ppm, got {:?}",
        rep_tms.final_offsets
    );

    // Verify emitted shifts remain strictly in the experimental (-2.75 ppm) frame!
    let shifted_pri = rep_tms.shifted_priors(&priors);
    let (pred_ref, _) = to_reference(
        &res_tms,
        &systems_tms,
        &ref_data.sequence,
        &ref_data.seq_ids,
        0.0,
        Some(&shifted_pri),
    );
    let mut ca_diffs = Vec::new();
    for (&(k, ref atom), &v_pred) in &pred_ref.shifts {
        if atom == "CA" {
            if let Some(&v_dss) = ref_data.shifts.get(&(k, "CA".to_string())) {
                ca_diffs.push(v_pred - v_dss);
            }
        }
    }
    ca_diffs.sort_by(|a, b| a.total_cmp(b));
    let med_diff = ca_diffs[ca_diffs.len() / 2];
    assert!(
        (med_diff - (-2.75)).abs() <= 0.15,
        "emitted shifts must remain in experimental ppm frame (-2.75 ppm), median diff = {med_diff:.3}"
    );
}

#[test]
fn synthetic_ubiquitin_7exp_and_5exp_meet_accuracy_targets() {
    let star_path = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&star_path, None).expect("read synthetic_ubiquitin.str");
    let priors = Priors::provisional();

    // 1. Standard 7-experiment synthetic benchmark (seed 0) via pre-built spin systems
    let (peaks7, truth7) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..Default::default()
        },
    );
    let systems_std =
        build_spin_systems(&peaks7, &Tolerances::default(), 3, true).expect("build_spin_systems");
    let cfg = EvoGraphConfig {
        seed: 0,
        ..Default::default()
    };
    let res_std = assign_evo_graph(&systems_std, &ref_data.sequence, &priors, &cfg);
    let m_std = score(&res_std, &truth7, 0.8);
    println!(
        "=== Synthetic Ubiquitin (7-exp, assign_evo_graph) ===\n{}",
        m_std.report()
    );
    assert!(
        m_std.accuracy() >= 66.0 / 68.0 - 1e-6,
        "7-exp accuracy {:.2}% < 97.1% (66/68)",
        m_std.accuracy() * 100.0
    );
    assert!(
        m_std.hc_precision() >= 0.981,
        "7-exp HC precision {:.2}% < 98.1%",
        m_std.hc_precision() * 100.0
    );

    // Also verify end-to-end `assign_evo_graph_from_peaks` on 7-exp peak list (mapping filtered roots)
    let (sys_e2e, res_e2e, rep_e2e) =
        assign_evo_graph_from_peaks(&peaks7, &ref_data.sequence, &priors, &cfg);
    let truth_e2e: Vec<i32> = rep_e2e
        .kept_hsqc_indices
        .iter()
        .map(|&i| truth7[i])
        .collect();
    let m_e2e = Metrics {
        assignable: truth7.iter().filter(|&&t| t >= 0).count(),
        ..score(&res_e2e, &truth_e2e, 0.8)
    };
    println!(
        "=== Synthetic Ubiquitin (7-exp, assign_evo_graph_from_peaks) ===\n{}",
        m_e2e.report()
    );
    assert!(
        m_e2e.accuracy() >= 66.0 / 68.0 - 1e-6,
        "7-exp e2e accuracy {:.2}% < 97.1% (66/68)",
        m_e2e.accuracy() * 100.0
    );
    assert!(
        m_e2e.hc_precision() >= 0.981,
        "7-exp e2e HC precision {:.2}% < 98.1%",
        m_e2e.hc_precision() * 100.0
    );
    let _ = sys_e2e;

    // 2. 5-experiment subset (`HSQC, HNCACB, CBCACONH, HNCO, HNCACO` — WITHOUT `HNCA` or `HNCOCA`)
    let sim5 = SimConfig {
        experiments: vec![
            "HNCACB".to_string(),
            "CBCACONH".to_string(),
            "HNCO".to_string(),
            "HNCACO".to_string(),
        ],
        seed: 0,
        ..Default::default()
    };
    let (peaks5, truth5) = simulate(&ref_data, &sim5);
    let (_sys5, res5, rep5) =
        assign_evo_graph_from_peaks(&peaks5, &ref_data.sequence, &priors, &cfg);
    let truth5_kept: Vec<i32> = rep5.kept_hsqc_indices.iter().map(|&i| truth5[i]).collect();
    let m5 = Metrics {
        assignable: truth5.iter().filter(|&&t| t >= 0).count(),
        ..score(&res5, &truth5_kept, 0.8)
    };
    println!(
        "=== Synthetic Ubiquitin (5-exp without HNCA/HNCOCA) ===\n{}",
        m5.report()
    );
    assert!(
        m5.accuracy() >= 0.94,
        "5-exp accuracy {:.2}% < 94.0%",
        m5.accuracy() * 100.0
    );
    assert!(
        m5.hc_precision() >= 0.97,
        "5-exp HC precision {:.2}% < 97.0%",
        m5.hc_precision() * 100.0
    );
}

fn parse_fasta_sequence(path: &Path) -> Vec<String> {
    let txt = fs::read_to_string(path).expect("read FASTA");
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

/// Parse external 8-column backbone shift table (`IndexRes H N C CA Ha Cb`), excluding `-` and `?`
/// entries cleanly (including non-whitespace-delimited `?` markers such as `119.24?170.15`).
fn parse_external_shift_table(path: &Path, sequence: &[String]) -> Reference {
    let txt = fs::read_to_string(path).expect("read external shift table");
    let seq_ids: Vec<i32> = (1..=sequence.len() as i32).collect();
    let mut reference = Reference::new(sequence.to_vec(), seq_ids);

    for line in txt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("Index") {
            continue;
        }
        // Replace `?<digit>` adjacency (e.g. `119.24?170.15`) and `:H* ?` with single-token uncertain markers
        let normalized = trimmed
            .replace("?1", "? 1")
            .replace("?2", "? 2")
            .replace("?3", "? 3")
            .replace("?4", "? 4")
            .replace("?5", "? 5")
            .replace("?6", "? 6")
            .replace("?7", "? 7")
            .replace("?8", "? 8")
            .replace("?9", "? 9")
            .replace(":H* ?", ":H*?");
        let cols: Vec<&str> = normalized.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let Ok(seq_num) = cols[0].parse::<usize>() else {
            continue;
        };
        if seq_num == 0 || seq_num > sequence.len() {
            continue;
        }
        let k = seq_num - 1;

        let parse_val = |tok: &str| -> Option<f64> {
            if tok == "-" || tok.contains('?') {
                return None;
            }
            let head = tok.split(':').next().unwrap_or(tok);
            head.parse::<f64>().ok()
        };

        // Columns: 0=Index, 1=Res, 2=H, 3=N, 4=C, 5=CA, followed by optional `:Cb` token
        if let Some(v) = cols.get(2).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "H".to_string()), v);
        }
        if let Some(v) = cols.get(3).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "N".to_string()), v);
        }
        if let Some(v) = cols.get(4).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "C".to_string()), v);
        }
        if let Some(v) = cols.get(5).and_then(|t| parse_val(t)) {
            reference.shifts.insert((k, "CA".to_string()), v);
        }
        if let Some(v) = cols
            .iter()
            .skip(6)
            .find(|t| t.ends_with(":Cb"))
            .and_then(|t| parse_val(t))
        {
            reference.shifts.insert((k, "CB".to_string()), v);
        }
    }
    reference
}

/// Match each retained `SpinSystem` `(H, N)` root against ground-truth `(H, N)` shifts within
/// `(0.045 ppm 1H, 0.40 ppm 15N)` so spin-system accuracy & recall can be scored directly.
fn map_spin_systems_to_truth(
    systems: &[nmr_core::spinsystems::SpinSystem],
    truth_ref: &Reference,
) -> Vec<i32> {
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

#[test]
fn real_bruker_ubiquitin_5seed_classical_pick_sequence_only() {
    let Some(data_dir) = resolve_local_real_data_dir() else {
        eprintln!(
            "Skipping real Bruker ubiquitin test: local data/ubiquitin directory not present"
        );
        return;
    };
    let ubq_dir = data_dir.join("ubiquitin");
    let fasta_path = ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt");
    let truth_path = data_dir.join("ubiquitin_backbone_assignment_191104.txt");
    if !fasta_path.is_file() || !truth_path.is_file() {
        eprintln!("Skipping real Bruker ubiquitin test: FASTA or shift table missing");
        return;
    }

    let sequence = parse_fasta_sequence(&fasta_path);
    assert_eq!(sequence.len(), 76);
    let truth_ref = parse_external_shift_table(&truth_path, &sequence);
    let observable_amides = (1..sequence.len())
        .filter(|&k| {
            sequence[k] != "PRO"
                && truth_ref.shifts.contains_key(&(k, "H".to_string()))
                && truth_ref.shifts.contains_key(&(k, "N".to_string()))
        })
        .count();
    assert_eq!(
        observable_amides, 70,
        "70 observable backbone amides expected after excluding Pro19/37/38, Met1, and exchange-broadened Glu24/Gly53"
    );
    assert_eq!(
        truth_ref.shifts.len(),
        359,
        "359 non-uncertain backbone shifts expected after excluding the 2 `?` values on Met1"
    );

    // Load the 5 real Bruker TopSpin processed spectra and run default classical peak picker
    let t_pick = Instant::now();
    let pcfg = PickerConfig::default();
    let hsqc_spec =
        read_bruker_pdata_2d(ubq_dir.join("Ubiquitin_090309/2/pdata/1")).expect("read HSQC");
    let hsqc_roots = pick_hsqc_2d(&hsqc_spec, &pcfg);

    let mut raw_peaks = hsqc_roots.clone();
    for (rel_path, exp_name) in [
        ("Ubiquitin_090309/3/pdata/1", "HNCACB"),
        ("Ubiquitin_090309/4/pdata/1", "CBCACONH"),
        ("Ubiquitin_111909/10/pdata/1", "HNCO"),
        ("Ubiquitin_111909/11/pdata/1", "HNCACO"),
    ] {
        let spec3d = read_bruker_pdata_3d(ubq_dir.join(rel_path), exp_name).expect("read 3D");
        let p3 = pick_triple_3d(&spec3d, &hsqc_roots, &pcfg);
        raw_peaks.extend(p3);
    }
    let pick_secs = t_pick.elapsed().as_secs_f64();
    println!(
        "Classical pick complete in {:.2}s: {} HSQC roots, {} total raw peaks",
        pick_secs,
        hsqc_roots.len(),
        raw_peaks.len()
    );
    assert_eq!(hsqc_roots.len(), 111);
    assert_eq!(raw_peaks.len(), 972);

    let priors = Priors::provisional();
    let mode_label = "Sequence-Only (from raw classical peaks)";
    {
        println!("\n==================== {mode_label} ====================");
        let mut accs = Vec::new();
        let mut recs = Vec::new();
        let mut hc_precs = Vec::new();
        let mut hc_recs = Vec::new();
        let mut shift_accs = Vec::new();
        let mut runtimes = Vec::new();

        for seed in 0..5u64 {
            let t0 = Instant::now();
            let cfg = EvoGraphConfig {
                structure_torsions: None,
                seed,
                ..Default::default()
            };
            let (systems, res, rep) =
                assign_evo_graph_from_peaks(&raw_peaks, &sequence, &priors, &cfg);
            let dt = t0.elapsed().as_secs_f64();

            let sys_truth = map_spin_systems_to_truth(&systems, &truth_ref);
            let distinct_true_roots: std::collections::BTreeSet<i32> =
                sys_truth.iter().copied().filter(|&k| k >= 0).collect();

            let mut m = score(&res, &sys_truth, 0.8);
            m.assignable = observable_amides;

            let shifted_pri = rep.shifted_priors(&priors);
            let (pred_ref, _) = to_reference(
                &res,
                &systems,
                &sequence,
                &truth_ref.seq_ids,
                0.0,
                Some(&shifted_pri),
            );
            let atom_cmp: BTreeMap<String, (usize, usize)> = compare_shifts(&pred_ref, &truth_ref);
            let tot_ok: usize = atom_cmp.values().map(|x| x.0).sum();
            let tot_cmp: usize = atom_cmp.values().map(|x| x.1).sum();
            let shift_pct = if tot_cmp > 0 {
                100.0 * (tot_ok as f64) / (tot_cmp as f64)
            } else {
                0.0
            };

            if seed == 0 {
                println!(
                    "  Filter: {} -> {} peaks ({} -> {} HSQC roots, {} NH2 removed, {}/70 true roots retained)",
                    rep.raw_peak_count,
                    rep.filtered_peak_count,
                    rep.raw_hsqc_count,
                    rep.filtered_hsqc_count,
                    rep.nh2_removed_count,
                    distinct_true_roots.len()
                );
                println!(
                    "  Offsets: pass1=(CA {:+.2}, CB {:+.2}, C {:+.2}, H {:+.2}, N {:+.2}) -> pass2[{} anchors]=(CA {:+.2}, CB {:+.2}, C {:+.2}, H {:+.2}, N {:+.2}); graph(anchored={}, pruned={})",
                    rep.pass1_offsets.d_ca,
                    rep.pass1_offsets.d_cb,
                    rep.pass1_offsets.d_c,
                    rep.pass1_offsets.d_h,
                    rep.pass1_offsets.d_n,
                    rep.pass2_anchor_count,
                    rep.final_offsets.d_ca,
                    rep.final_offsets.d_cb,
                    rep.final_offsets.d_c,
                    rep.final_offsets.d_h,
                    rep.final_offsets.d_n,
                    rep.graph_anchored_count,
                    rep.graph_pruned_count,
                );
                println!("  Per-atom shift match (seed 0): {atom_cmp:?}");
            }

            println!(
                "  seed {seed}: assigned={}/{}, correct={}, acc={:.1}%, recall={:.1}%, HC_prec={:.1}% ({}/{}), HC_rec={:.1}%, shift_acc={:.1}% ({}/{}), time={:.2}s",
                m.assigned,
                m.assignable,
                m.correct,
                m.accuracy() * 100.0,
                m.recall() * 100.0,
                m.hc_precision() * 100.0,
                m.hc_correct,
                m.hc_assigned,
                m.hc_recall() * 100.0,
                shift_pct,
                tot_ok,
                tot_cmp,
                dt
            );

            accs.push(m.accuracy() * 100.0);
            recs.push(m.recall() * 100.0);
            hc_precs.push(m.hc_precision() * 100.0);
            hc_recs.push(m.hc_recall() * 100.0);
            shift_accs.push(shift_pct);
            runtimes.push(dt);
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64);
        println!(
            "  5-SEED MEAN ({mode_label}): acc={:.2}%, recall={:.2}%, HC_prec={:.2}%, HC_rec={:.2}%, shift_acc={:.2}%, runtime={:.2}s",
            mean(&accs),
            mean(&recs),
            mean(&hc_precs),
            mean(&hc_recs),
            mean(&shift_accs),
            mean(&runtimes),
        );

        assert!(
            mean(&accs) >= 90.0,
            "{mode_label} 5-seed mean accuracy {:.2}% < 90.0%",
            mean(&accs)
        );
        assert!(
            mean(&hc_precs) >= 95.0,
            "{mode_label} 5-seed mean HC precision {:.2}% < 95.0%",
            mean(&hc_precs)
        );
        let _ = ReferenceOffsets::default();
        let _ = apply_offsets_to_priors;
        let _ = refine_anchor_reference_offsets;
    }
}
