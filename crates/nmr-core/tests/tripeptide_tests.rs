//! Integration & benchmark tests for the clean-room Shen–Baker–Bax parallel tripeptide solver
//! (`nmr_core::tripeptide`): neighbor-corrected secondary shifts, automatic TMS/DSS reference
//! alignment, blind homolog-excluded BMRB fragment scoring, `synthetic_ubiquitin.str` (`ILE30`/`PHE45`
//! overlap & 5-spectrum no-`HNCA` subsets), 12-protein multi-fold BMRB benchmark suite, and real
//! Bruker ubiquitin spectra (when present on disk).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use nmr_core::evaluate::{compare_shifts, score, to_reference};
use nmr_core::nmrstar::read_star;
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::tripeptide::{
    assign_tripeptide, blosum62_score, estimate_reference_offsets, neighbor_corrected_random_coil,
    secondary_shift, ReferenceOffsets, TripeptideConfig, TripeptideFragmentDb, TripeptideSolver,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    bmrb_id: String,
    _pdb_id: String,
    name: String,
    fold_class: String,
    n_residues: usize,
    path: PathBuf,
}

fn load_manifest() -> Vec<ManifestEntry> {
    let root = workspace_root();
    let txt = fs::read_to_string(root.join("data/benchmark/manifest.tsv"))
        .expect("read data/benchmark/manifest.tsv");
    let mut out = Vec::new();
    for (idx, line) in txt.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || idx == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        out.push(ManifestEntry {
            bmrb_id: cols[0].to_string(),
            _pdb_id: cols[1].to_string(),
            name: cols[2].to_string(),
            fold_class: cols[3].to_string(),
            n_residues: cols[4].parse().unwrap(),
            path: root.join(cols[5]),
        });
    }
    out
}

#[test]
fn test_neighbor_corrected_secondary_shifts_and_blosum62() {
    // 1. Symmetric BLOSUM62 matrix sanity
    for a in [
        "ALA", "ARG", "CYS", "GLY", "ILE", "PHE", "PRO", "TRP", "TYR", "VAL",
    ] {
        assert!(blosum62_score(a, a) >= 4);
        for b in ["ALA", "GLY", "ILE", "PHE", "PRO", "TYR", "VAL"] {
            assert_eq!(blosum62_score(a, b), blosum62_score(b, a));
        }
    }
    assert!(blosum62_score("PHE", "TYR") > blosum62_score("PHE", "GLY"));
    assert!(blosum62_score("ILE", "VAL") > blosum62_score("ILE", "ASP"));

    // 2. Pre-Proline neighbor shift (Wishart et al., JBNMR 1995 Table 3)
    let glu_gg =
        neighbor_corrected_random_coil(Some("GLY"), "GLU", Some("GLY"), "CA", false).unwrap();
    let glu_pro =
        neighbor_corrected_random_coil(Some("GLY"), "GLU", Some("PRO"), "CA", false).unwrap();
    assert!(
        (glu_pro - glu_gg - (-1.95)).abs() < 1e-6,
        "expected -1.95 ppm pre-Pro CA shift, got {} vs {}",
        glu_pro,
        glu_gg
    );
    let glu_c_gg =
        neighbor_corrected_random_coil(Some("GLY"), "GLU", Some("GLY"), "C", false).unwrap();
    let glu_c_pro =
        neighbor_corrected_random_coil(Some("GLY"), "GLU", Some("PRO"), "C", false).unwrap();
    assert!((glu_c_pro - glu_c_gg - (-2.00)).abs() < 1e-6);

    // 3. Preceding beta-branched substituent effect on 15N (Schwarzinger et al., JACS 2001)
    let ala_after_gly =
        neighbor_corrected_random_coil(Some("GLY"), "ALA", Some("GLY"), "N", false).unwrap();
    let ala_after_ile =
        neighbor_corrected_random_coil(Some("ILE"), "ALA", Some("GLY"), "N", false).unwrap();
    assert!(
        ala_after_ile - ala_after_gly > 2.0,
        "expected > +2.0 ppm downfield 15N shift after Ile: {} vs {}",
        ala_after_ile,
        ala_after_gly
    );

    // 4. Cysteine redox state & Gly/Pro absent nuclei
    let cys_red = neighbor_corrected_random_coil(None, "CYS", None, "CB", false).unwrap();
    let cys_ox = neighbor_corrected_random_coil(None, "CYS", None, "CB", true).unwrap();
    assert!(cys_ox - cys_red > 12.0);
    assert!(neighbor_corrected_random_coil(None, "GLY", None, "CB", false).is_none());
    assert!(neighbor_corrected_random_coil(None, "PRO", None, "H", false).is_none());

    // 5. Secondary shift with TMS -2.75 ppm reference offset
    let dca = secondary_shift(56.90, -2.75, Some("LYS"), "ALA", Some("LYS"), "CA", false).unwrap();
    // (56.90 - (-2.75)) - ~52.36 = +7.29 ppm (or for 52.90 -> +3.29 ppm helical)
    assert!(dca > 6.5);
}

#[test]
fn test_tripeptide_db_homolog_exclusion_and_cooperative_basin_scoring() {
    // 1. Standalone embedded DB works with zero external files (WASM-compatible)
    let emb_db = TripeptideFragmentDb::embedded();
    assert!(emb_db.sources.is_empty());
    assert!(emb_db.fragments.len() >= 16);

    let solver = TripeptideSolver::with_db(TripeptideConfig::default(), emb_db);
    let seq: Vec<String> = ["ALA", "LYS", "ILE", "GLN", "ASP"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    // Cooperative helical secondary shifts across (LYS, ILE, GLN): dCA > 0, dCB < 0, dC > 0
    let coherent_helix = [
        [Some(3.2), Some(-0.6), Some(2.1), None, None],
        [Some(3.6), Some(-0.8), Some(2.3), Some(-1.5), Some(-0.3)],
        [Some(3.1), Some(-0.5), Some(2.0), Some(-1.4), Some(-0.25)],
    ];
    // Physically contradictory secondary-shift flip: Helix -> Sheet -> Helix with mismatched CA/CB/C'
    let contradictory_flip = [
        [Some(3.2), Some(-0.6), Some(2.1), None, None],
        [Some(-2.2), Some(2.6), Some(2.4), Some(2.0), Some(0.5)],
        [Some(3.1), Some(-0.5), Some(2.0), Some(-1.4), Some(-0.25)],
    ];
    let sc_coherent = solver.score_window_secondary_shifts(&seq, 2, &coherent_helix);
    let sc_flip = solver.score_window_secondary_shifts(&seq, 2, &contradictory_flip);
    println!(
        "[Embedded basin cooperativity] coherent_helix={:.3} vs contradictory_flip={:.3}",
        sc_coherent, sc_flip
    );
    assert!(
        sc_coherent > sc_flip + 1.0,
        "cooperative helical tripeptide must outscore contradictory secondary-shift flip: {sc_coherent} vs {sc_flip}"
    );

    // 2. Default DB harvests BMRB entries and strictly excludes ubiquitin homologs on ubiquitin query
    let full_db = TripeptideFragmentDb::default();
    assert!(
        full_db.sources.len() >= 12 && full_db.fragments.len() > 500,
        "expected >= 12 harvested BMRB sources and > 500 fragments, got {} sources / {} fragments",
        full_db.sources.len(),
        full_db.fragments.len()
    );
    let ubq_ref = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None).unwrap();
    let mask = full_db.non_homologous_mask(&ubq_ref.sequence, 0.35);
    for (src, &allowed) in full_db.sources.iter().zip(mask.iter()) {
        if src.id == "bmr17769" || src.id == "bmr5387" || src.id == "bmr4375" {
            assert!(
                !allowed,
                "ubiquitin homolog {} MUST be excluded during blind evaluation!",
                src.id
            );
        } else {
            assert!(
                allowed,
                "unrelated fold {} should remain available in fragment library",
                src.id
            );
        }
    }
}

#[test]
fn test_auto_reference_offset_recovery_and_tms_shifted_ubiquitin() {
    let root = workspace_root();
    let priors = Priors::bmrb();
    let ubq_ref = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();

    // 1. DSS-referenced peaks must yield ~0.0 ppm offsets
    let (dss_peaks, truth) = simulate(
        &ubq_ref,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let dss_sys = build_spin_systems(&dss_peaks, &Tolerances::default(), 3, true).unwrap();
    let dss_off = estimate_reference_offsets(&dss_sys, &ubq_ref.sequence, &priors);
    assert_eq!(
        dss_off,
        ReferenceOffsets::zero(),
        "DSS-referenced ubiquitin should trigger zero offset deadband, got {:?}",
        dss_off
    );

    // 2. Apply realistic Bruker TMS -2.75 ppm 13C offset + 250 spurious roots
    let tms_peaks: Vec<Peak> = dss_peaks
        .iter()
        .cloned()
        .map(|mut p| {
            if let Some(c) = p.c {
                p.c = Some(c - 2.75);
            }
            p
        })
        .collect();
    let mut tms_sys = build_spin_systems(&tms_peaks, &Tolerances::default(), 3, true).unwrap();
    let mut tms_truth = truth.clone();
    // Inject 250 spurious 1-peak HNCO noise roots (mimicking t1/sinc artifacts)
    for i in 0..250 {
        let mut junk = SpinSystem::new(6.1 + (i as f64) * 0.015, 105.0 + (i % 25) as f64);
        junk.cands
            .insert(("C".to_string(), -1), vec![(171.0 + (i % 9) as f64, 1)]);
        tms_sys.push(junk);
        tms_truth.push(-1);
    }

    let est = estimate_reference_offsets(&tms_sys, &ubq_ref.sequence, &priors);
    println!(
        "[TMS -2.75 ppm + 250 noise roots] estimated offsets: {:?}",
        est
    );
    assert!(
        (est.dca - (-2.75)).abs() < 0.35
            && (est.dcb - (-2.75)).abs() < 0.35
            && (est.dc - (-2.75)).abs() < 0.40,
        "expected ~ -2.75 ppm 13C offsets, got {:?}",
        est
    );

    // Compare pre-alignment standard `assign` baseline vs `assign_tripeptide`
    let std_res = assign(
        &tms_sys,
        &ubq_ref.sequence,
        &priors,
        &ScoreConfig {
            auto_reference: false,
            forbid_ca_less_excess_roots: false,
            ..ScoreConfig::default()
        },
        &AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let tri_res = assign_tripeptide(
        &tms_sys,
        &ubq_ref.sequence,
        &priors,
        &TripeptideConfig::default(),
    );
    let m_std = score(&std_res, &tms_truth, 0.8);
    let m_tri = score(&tri_res, &tms_truth, 0.8);
    println!(
        "[TMS -2.75 ppm + 250 noise roots] standard acc={:.1}% rec={:.1}% vs tripeptide acc={:.1}% rec={:.1}% hc_prec={:.1}%",
        m_std.accuracy() * 100.0,
        m_std.recall() * 100.0,
        m_tri.accuracy() * 100.0,
        m_tri.recall() * 100.0,
        m_tri.hc_precision() * 100.0
    );
    assert!(
        m_tri.accuracy() >= 0.95
            && m_tri.recall() >= 0.95
            && m_tri.hc_precision() >= 0.98
            && m_tri.accuracy() >= m_std.accuracy() + 0.05,
        "tripeptide solver must recover TMS-shifted spectra under spurious root flood: tri={:.3}, std={:.3}",
        m_tri.accuracy(),
        m_std.accuracy()
    );

    // Ensure `to_reference` preserves the experimental TMS ppm frame (not shifted by +2.75 ppm)
    let (pred_ref, _) = to_reference(
        &tri_res,
        &tms_sys,
        &ubq_ref.sequence,
        &ubq_ref.seq_ids,
        0.0,
        Some(&priors),
    );
    let mut diffs = Vec::new();
    for (&(k, ref atom), &v_pred) in &pred_ref.shifts {
        if matches!(atom.as_str(), "CA" | "CB" | "C") {
            if let Some(&v_dss) = ubq_ref.shifts.get(&(k, atom.clone())) {
                diffs.push(v_pred - (v_dss - 2.75));
            }
        }
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    let med_diff = diffs[diffs.len() / 2];
    assert!(
        med_diff.abs() < 0.08,
        "predicted carbon shifts must stay in experimental TMS ppm frame, median diff={med_diff}"
    );
}

#[test]
fn test_synthetic_ubiquitin_ile30_phe45_and_5spec_no_hnca() {
    let root = workspace_root();
    let priors = Priors::bmrb();
    let syn_ref = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None).unwrap();

    // 1. Standard 7-spectrum synthetic_ubiquitin.str (seed = 0, matching CLI benchmark)
    let (syn_peaks, syn_truth) = simulate(
        &syn_ref,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let syn_sys = build_spin_systems(&syn_peaks, &Tolerances::default(), 3, false).unwrap();

    let std_res = assign(
        &syn_sys,
        &syn_ref.sequence,
        &priors,
        &ScoreConfig::default(),
        &AnnealConfig {
            runs: 16,
            steps_per_system: 2500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let tri_res = assign_tripeptide(
        &syn_sys,
        &syn_ref.sequence,
        &priors,
        &TripeptideConfig {
            anneal: AnnealConfig {
                runs: 20,
                steps_per_system: 2500,
                seed: 0,
                ..AnnealConfig::default()
            },
            ..TripeptideConfig::default()
        },
    );
    let m_std = score(&std_res, &syn_truth, 0.8);
    let m_tri = score(&tri_res, &syn_truth, 0.8);

    // Check ILE30 (k=29) and PHE45 (k=44) specifically!
    let ile30_ss = syn_truth.iter().position(|&t| t == 29).unwrap();
    let phe45_ss = syn_truth.iter().position(|&t| t == 44).unwrap();
    println!(
        "[synthetic_ubiquitin seed=0]\n  standard  : acc={:.1}% ({}/{}) rec={:.1}% hc_prec={:.1}% ({}/{}) | ILE30->{} (conf={:.2}) PHE45->{} (conf={:.2})\n  tripeptide: acc={:.1}% ({}/{}) rec={:.1}% hc_prec={:.1}% ({}/{}) | ILE30->{} (conf={:.2}) PHE45->{} (conf={:.2})",
        m_std.accuracy() * 100.0,
        m_std.correct,
        m_std.assigned,
        m_std.recall() * 100.0,
        m_std.hc_precision() * 100.0,
        m_std.hc_correct,
        m_std.hc_assigned,
        std_res.position[ile30_ss],
        std_res.confidence[ile30_ss],
        std_res.position[phe45_ss],
        std_res.confidence[phe45_ss],
        m_tri.accuracy() * 100.0,
        m_tri.correct,
        m_tri.assigned,
        m_tri.recall() * 100.0,
        m_tri.hc_precision() * 100.0,
        m_tri.hc_correct,
        m_tri.hc_assigned,
        tri_res.position[ile30_ss],
        tri_res.confidence[ile30_ss],
        tri_res.position[phe45_ss],
        tri_res.confidence[phe45_ss],
    );

    assert_eq!(
        tri_res.position[ile30_ss], 29,
        "tripeptide solver should resolve overlapped ILE30 (k=29)"
    );
    assert_eq!(
        tri_res.position[phe45_ss], 44,
        "tripeptide solver should resolve overlapped PHE45 (k=44)"
    );
    assert!(
        m_tri.accuracy() >= m_std.accuracy()
            && m_tri.recall() >= m_std.recall()
            && m_tri.hc_precision() >= 0.98,
        "tripeptide must match or beat standard solver on synthetic_ubiquitin.str"
    );

    let (pred_ref, _) = to_reference(
        &tri_res,
        &syn_sys,
        &syn_ref.sequence,
        &syn_ref.seq_ids,
        0.0,
        Some(&priors),
    );
    let shift_ag = compare_shifts(&pred_ref, &syn_ref);
    println!("[synthetic_ubiquitin shift agreement] {:?}", shift_ag);

    // 2. Degenerate 5-spectrum subset WITHOUT HNCA & WITHOUT HNCOCA:
    //    `HSQC + HNCACB + CBCACONH + HNCO + HNCACO` (with `degenerate = false` default!)
    let bmr_ubq = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let exps_5 = vec![
        "HSQC".to_string(),
        "HNCACB".to_string(),
        "CBCACONH".to_string(),
        "HNCO".to_string(),
        "HNCACO".to_string(),
    ];
    let mut sum_std_5 = 0.0;
    let mut sum_tri_5 = 0.0;
    let mut sum_std_rec = 0.0;
    let mut sum_tri_rec = 0.0;
    let mut sum_tri_hcp = 0.0;
    let seeds = 8u64;

    for seed in 0..seeds {
        let (p5, t5) = simulate(
            &bmr_ubq,
            &SimConfig {
                experiments: exps_5.clone(),
                p_missing_peak: 0.08,
                spurious_rate: 0.05,
                seed,
                ..SimConfig::default()
            },
        );
        let sys5 = build_spin_systems(&p5, &Tolerances::default(), 3, false).unwrap();
        let r_std = assign(
            &sys5,
            &bmr_ubq.sequence,
            &priors,
            &ScoreConfig::default(),
            &AnnealConfig {
                runs: 16,
                steps_per_system: 2000,
                seed,
                ..AnnealConfig::default()
            },
        );
        let r_tri = assign_tripeptide(
            &sys5,
            &bmr_ubq.sequence,
            &priors,
            &TripeptideConfig {
                anneal: AnnealConfig {
                    runs: 16,
                    steps_per_system: 2000,
                    seed,
                    ..AnnealConfig::default()
                },
                ..TripeptideConfig::default()
            },
        );
        let ms = score(&r_std, &t5, 0.8);
        let mt = score(&r_tri, &t5, 0.8);
        sum_std_5 += ms.accuracy();
        sum_tri_5 += mt.accuracy();
        sum_std_rec += ms.recall();
        sum_tri_rec += mt.recall();
        sum_tri_hcp += mt.hc_precision();
    }
    let mean_std_5 = sum_std_5 / (seeds as f64);
    let mean_tri_5 = sum_tri_5 / (seeds as f64);
    println!(
        "[5-spectrum no-HNCA bmr17769 8-seed mean (degenerate=false)]\n  standard  : acc={:.2}% rec={:.2}%\n  tripeptide: acc={:.2}% rec={:.2}% hc_prec={:.2}%",
        mean_std_5 * 100.0,
        (sum_std_rec / seeds as f64) * 100.0,
        mean_tri_5 * 100.0,
        (sum_tri_rec / seeds as f64) * 100.0,
        (sum_tri_hcp / seeds as f64) * 100.0,
    );
    assert!(
        mean_tri_5 >= mean_std_5 && mean_tri_5 >= 0.94 && (sum_tri_hcp / seeds as f64) >= 0.97,
        "tripeptide solver must match or beat standard solver on 5-spectrum no-HNCA ubiquitin: tri={mean_tri_5:.4}, std={mean_std_5:.4}"
    );

    // 3. Also verify 5-spectrum no-HNCA subset directly on `synthetic_ubiquitin.str` (8-seed mean)
    let mut syn_std_5 = 0.0;
    let mut syn_tri_5 = 0.0;
    let mut syn_tri_hcp_5 = 0.0;
    for seed in 0..seeds {
        let (sp5, st5) = simulate(
            &syn_ref,
            &SimConfig {
                experiments: exps_5.clone(),
                p_missing_peak: 0.08,
                spurious_rate: 0.05,
                seed,
                ..SimConfig::default()
            },
        );
        let ssys5 = build_spin_systems(&sp5, &Tolerances::default(), 3, false).unwrap();
        let rs = assign(
            &ssys5,
            &syn_ref.sequence,
            &priors,
            &ScoreConfig::default(),
            &AnnealConfig {
                runs: 16,
                steps_per_system: 2000,
                seed,
                ..AnnealConfig::default()
            },
        );
        let rt = assign_tripeptide(
            &ssys5,
            &syn_ref.sequence,
            &priors,
            &TripeptideConfig {
                anneal: AnnealConfig {
                    runs: 16,
                    steps_per_system: 2000,
                    seed,
                    ..AnnealConfig::default()
                },
                ..TripeptideConfig::default()
            },
        );
        syn_std_5 += score(&rs, &st5, 0.8).accuracy();
        let mrt = score(&rt, &st5, 0.8);
        syn_tri_5 += mrt.accuracy();
        syn_tri_hcp_5 += mrt.hc_precision();
    }
    println!(
        "[5-spectrum no-HNCA synthetic_ubiquitin 8-seed mean (degenerate=false)]\n  standard  : acc={:.2}%\n  tripeptide: acc={:.2}% hc_prec={:.2}%",
        (syn_std_5 / seeds as f64) * 100.0,
        (syn_tri_5 / seeds as f64) * 100.0,
        (syn_tri_hcp_5 / seeds as f64) * 100.0,
    );
    assert!(
        syn_tri_5 / (seeds as f64) >= syn_std_5 / (seeds as f64)
            && syn_tri_hcp_5 / (seeds as f64) >= 0.97,
        "tripeptide must match or beat standard on 5-spectrum synthetic_ubiquitin"
    );
}

#[test]
fn test_benchmark_suite_12_folds_tripeptide_vs_standard() {
    let priors = Priors::bmrb();
    let manifest = load_manifest();
    assert_eq!(manifest.len(), 12);

    let mut std_acc_sum = 0.0;
    let mut std_rec_sum = 0.0;
    let mut std_hcp_sum = 0.0;
    let mut tri_acc_sum = 0.0;
    let mut tri_rec_sum = 0.0;
    let mut tri_hcp_sum = 0.0;

    println!("\nBMRB_ID   FOLD        LEN  STD_ACC  TRI_ACC  STD_REC  TRI_REC  STD_HCP  TRI_HCP");
    for entry in &manifest {
        let ref_data = read_star(&entry.path, None).unwrap();
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed: 0,
                ..SimConfig::default()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, true).unwrap();

        let r_std = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &ScoreConfig::default(),
            &AnnealConfig {
                runs: 12,
                steps_per_system: 1500,
                seed: 0,
                ..AnnealConfig::default()
            },
        );
        let r_tri = assign_tripeptide(
            &systems,
            &ref_data.sequence,
            &priors,
            &TripeptideConfig {
                anneal: AnnealConfig {
                    runs: 16,
                    steps_per_system: 2000,
                    seed: 0,
                    ..AnnealConfig::default()
                },
                ..TripeptideConfig::default()
            },
        );
        let ms = score(&r_std, &truth, 0.8);
        let mt = score(&r_tri, &truth, 0.8);

        println!(
            "{:<9} {:<10} {:>4}  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>6.1}%  {}",
            entry.bmrb_id,
            entry.fold_class,
            entry.n_residues,
            ms.accuracy() * 100.0,
            mt.accuracy() * 100.0,
            ms.recall() * 100.0,
            mt.recall() * 100.0,
            ms.hc_precision() * 100.0,
            mt.hc_precision() * 100.0,
            entry.name
        );

        std_acc_sum += ms.accuracy();
        std_rec_sum += ms.recall();
        std_hcp_sum += ms.hc_precision();
        tri_acc_sum += mt.accuracy();
        tri_rec_sum += mt.recall();
        tri_hcp_sum += mt.hc_precision();

        let is_disordered =
            entry.fold_class == "idp" || entry.bmrb_id == "bmr4375" || entry.bmrb_id == "bmr15176";
        let min_acc = if is_disordered {
            0.50
        } else if entry.n_residues >= 145 && entry.fold_class == "alpha" {
            0.75
        } else {
            0.90
        };
        assert!(
            mt.accuracy() >= min_acc,
            "{} tripeptide accuracy {:.1}% below floor {:.1}%",
            entry.bmrb_id,
            mt.accuracy() * 100.0,
            min_acc * 100.0
        );
        assert!(
            mt.hc_precision() >= 0.90,
            "{} tripeptide HC precision {:.1}% below 90%",
            entry.bmrb_id,
            mt.hc_precision() * 100.0
        );
    }

    let n = manifest.len() as f64;
    println!(
        "MEAN (12 BMRB entries): STD acc={:.2}% rec={:.2}% hcp={:.2}% | TRI acc={:.2}% rec={:.2}% hcp={:.2}%",
        (std_acc_sum / n) * 100.0,
        (std_rec_sum / n) * 100.0,
        (std_hcp_sum / n) * 100.0,
        (tri_acc_sum / n) * 100.0,
        (tri_rec_sum / n) * 100.0,
        (tri_hcp_sum / n) * 100.0,
    );
    assert!(tri_acc_sum / n >= 0.85 && tri_hcp_sum / n >= 0.975);
}

fn find_real_ubiquitin_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NMR_UBIQUITIN_DIR") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    let direct = workspace_root().join("data/ubiquitin");
    if direct.is_dir() {
        return Some(direct);
    }
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .current_dir(workspace_root())
        .output()
    {
        if out.status.success() {
            let common = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let common_path = workspace_root().join(common);
            if let Some(repo_parent) = common_path.parent() {
                let shared = repo_parent.join("data/ubiquitin");
                if shared.is_dir() {
                    return Some(shared);
                }
            }
        }
    }
    None
}

#[test]
fn test_real_ubiquitin_bruker_spectra_if_present() {
    let Some(ubq_dir) = find_real_ubiquitin_dir() else {
        println!("Skipping real Bruker ubiquitin test: data/ubiquitin not present on disk");
        return;
    };
    let hsqc_path = ubq_dir.join("Ubiquitin_090309/2/pdata/1");
    let hncacb_path = ubq_dir.join("Ubiquitin_090309/3/pdata/1");
    let cbcaconh_path = ubq_dir.join("Ubiquitin_090309/4/pdata/1");
    let hnco_path = ubq_dir.join("Ubiquitin_111909/10/pdata/1");
    let hncaco_path = ubq_dir.join("Ubiquitin_111909/11/pdata/1");
    let fasta_path = ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt");
    if !hsqc_path.is_dir() || !fasta_path.is_file() {
        println!("Skipping real Bruker ubiquitin test: pdata/FASTA subpaths missing");
        return;
    }

    let fasta_txt = fs::read_to_string(&fasta_path).unwrap();
    let seq_1: String = fasta_txt
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.trim().chars())
        .collect();
    let sequence: Vec<String> = seq_1.chars().map(|c| one_to_three(c).to_string()).collect();
    assert_eq!(sequence.len(), 76);

    let pcfg = PickerConfig::default();
    let hsqc_2d = read_bruker_pdata_2d(&hsqc_path).unwrap();
    let raw_roots = pick_hsqc_2d(&hsqc_2d, &pcfg);

    // Filter HSQC roots to amide 1H window & top intensity (suppressing F1/t1 sinc ripples in test)
    let mut roots_sorted: Vec<Peak> = raw_roots
        .iter()
        .filter(|p| (5.8..=10.8).contains(&p.h) && p.sign > 0)
        .cloned()
        .collect();
    roots_sorted.sort_by(|a, b| b.height.unwrap_or(0.0).total_cmp(&a.height.unwrap_or(0.0)));
    // Suppress F1 t1-ridge / sinc sidelobes at virtually identical 1H (`|dH| <= 0.018 ppm`) < 18% parent height
    let mut clean_roots: Vec<Peak> = Vec::new();
    for cand in roots_sorted {
        let ch = cand.height.unwrap_or(0.0);
        let is_sinc = clean_roots.iter().any(|parent| {
            let ph = parent.height.unwrap_or(0.0);
            ((cand.h - parent.h).abs() <= 0.016 && ch < 0.22 * ph)
                || ((cand.n - parent.n).abs() <= 0.14
                    && (0.32..=1.15).contains(&(cand.h - parent.h).abs())
                    && (ch - ph).abs() < 0.35 * ph.max(ch)
                    && cand.h < 7.85
                    && parent.h < 7.85)
        });
        if !is_sinc {
            clean_roots.push(cand);
        }
        if clean_roots.len() >= 92 {
            break;
        }
    }

    let real_tol = Tolerances {
        h: 0.045,
        n: 0.42,
        ..Tolerances::default()
    };
    let mut all_peaks = clean_roots.clone();
    for (exp_name, path) in [
        ("HNCACB", &hncacb_path),
        ("CBCACONH", &cbcaconh_path),
        ("HNCO", &hnco_path),
        ("HNCACO", &hncaco_path),
    ] {
        let s3d = read_bruker_pdata_3d(path, exp_name).unwrap();
        let raw_3d = pick_triple_3d(&s3d, &clean_roots, &pcfg);
        let mut exp_kept: Vec<Peak> = Vec::new();

        // Per-(root, sign) stoichiometric selection within physical 13C bounds + F1 sinc sidelobe gate
        for r in &clean_roots {
            let mut pos_parents: Vec<(f64, f64)> = Vec::new();
            for sign_dir in [1i32, -1i32] {
                if exp_name != "HNCACB" && sign_dir < 0 {
                    continue;
                }
                let (c_lo, c_hi, cap, rel_floor) = match (exp_name, sign_dir) {
                    ("HNCACB", 1) => (38.0, 66.5, 2usize, 0.12),
                    ("HNCACB", _) => (10.0, 72.5, 2usize, 0.10),
                    ("CBCACONH", _) => (10.0, 72.5, 2usize, 0.10),
                    ("HNCO", _) => (165.0, 185.0, 1usize, 0.20),
                    ("HNCACO", _) => (165.0, 185.0, 2usize, 0.15),
                    _ => (10.0, 185.0, 2usize, 0.12),
                };
                let mut strip: Vec<Peak> = raw_3d
                    .iter()
                    .filter(|p| {
                        p.sign.signum() == sign_dir
                            && p.c.is_some_and(|c| c >= c_lo && c <= c_hi)
                            && ((r.h - p.h) / real_tol.h).hypot((r.n - p.n) / real_tol.n) <= 1.0
                    })
                    .cloned()
                    .collect();
                strip.sort_by(|a, b| {
                    let da = ((r.h - a.h) / real_tol.h).hypot((r.n - a.n) / real_tol.n);
                    let db = ((r.h - b.h) / real_tol.h).hypot((r.n - b.n) / real_tol.n);
                    let sa = a.height.unwrap_or(0.0).abs() * (-0.5 * da * da).exp();
                    let sb = b.height.unwrap_or(0.0).abs() * (-0.5 * db * db).exp();
                    sb.total_cmp(&sa)
                });
                let top_amp = strip
                    .first()
                    .and_then(|p| p.height)
                    .map(f64::abs)
                    .unwrap_or(0.0);
                let mut kept: Vec<Peak> = Vec::new();
                for pk in strip {
                    let c_val = pk.c.unwrap();
                    let amp = pk.height.unwrap_or(0.0).abs();
                    if amp < rel_floor * top_amp {
                        continue;
                    }
                    // Reject negative F1 sinc truncation sidelobes adjacent to intense positive CA peaks
                    if exp_name == "HNCACB"
                        && sign_dir < 0
                        && pos_parents
                            .iter()
                            .any(|&(pc, pamp)| (pc - c_val).abs() <= 0.90 && amp < 0.18 * pamp)
                    {
                        continue;
                    }
                    if !kept
                        .iter()
                        .any(|k: &Peak| (k.c.unwrap() - c_val).abs() <= 0.28)
                    {
                        if exp_name == "HNCACB" && sign_dir > 0 {
                            pos_parents.push((c_val, amp));
                        }
                        kept.push(pk);
                    }
                    if kept.len() >= cap {
                        break;
                    }
                }
                for pk in kept {
                    let dup = exp_kept.iter().any(|q| {
                        q.sign == pk.sign
                            && (q.h - pk.h).abs() < 1e-5
                            && (q.n - pk.n).abs() < 1e-4
                            && (q.c.unwrap() - pk.c.unwrap()).abs() < 1e-4
                    });
                    if !dup {
                        exp_kept.push(pk);
                    }
                }
            }
        }
        all_peaks.extend(exp_kept);
    }

    let systems = build_spin_systems(&all_peaks, &real_tol, 3, false).unwrap();
    let priors = Priors::bmrb();
    let est_off = estimate_reference_offsets(&systems, &sequence, &priors);
    println!(
        "[Real Bruker ubiquitin] roots={} | estimated offsets: dCA={:.2} dCB={:.2} dC={:.2} dH={:.2} dN={:.2}",
        systems.len(),
        est_off.dca,
        est_off.dcb,
        est_off.dc,
        est_off.dh,
        est_off.dn
    );
    assert!(
        est_off.dca < -1.8 && est_off.dcb < -1.8 && est_off.dc < -1.8,
        "expected negative ~ -2.7 ppm TMS-vs-DSS 13C offset on real Bruker ubiquitin, got {:?}",
        est_off
    );

    let r_std = assign(
        &systems,
        &sequence,
        &priors,
        &ScoreConfig {
            auto_reference: false,
            ..ScoreConfig::default()
        },
        &AnnealConfig {
            runs: 16,
            steps_per_system: 2000,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let r_tri = assign_tripeptide(&systems, &sequence, &priors, &TripeptideConfig::default());

    // Compare carbon chemical shifts against public CC0 BMRB 17769 (after shifting BMRB DSS -> TMS by est_off)
    let bmr17769 = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None).unwrap();
    let mut bmr_tms = bmr17769.clone();
    for ((_k, atom), val) in &mut bmr_tms.shifts {
        *val += est_off.for_atom(atom);
    }
    let seq_ids: Vec<i32> = (1..=76).collect();
    let (pred_std, _) = to_reference(&r_std, &systems, &sequence, &seq_ids, 0.0, Some(&priors));
    let (pred_tri, _) = to_reference(&r_tri, &systems, &sequence, &seq_ids, 0.0, Some(&priors));
    let (pred_tri_hc, _) = to_reference(&r_tri, &systems, &sequence, &seq_ids, 0.8, Some(&priors));

    let ag_std = compare_shifts(&pred_std, &bmr_tms);
    let ag_tri = compare_shifts(&pred_tri, &bmr_tms);
    let ag_tri_hc = compare_shifts(&pred_tri_hc, &bmr_tms);

    let pct = |m: &BTreeMap<String, (usize, usize)>, k: &str| -> f64 {
        let (ok, tot) = m.get(k).copied().unwrap_or((0, 0));
        if tot > 0 {
            100.0 * (ok as f64) / (tot as f64)
        } else {
            0.0
        }
    };
    println!(
        "[Real Bruker ubiquitin vs public BMRB 17769 (TMS-aligned)]\n  standard   all : CA={:.1}% ({:?}) CB={:.1}% ({:?}) C={:.1}% ({:?})\n  tripeptide all : CA={:.1}% ({:?}) CB={:.1}% ({:?}) C={:.1}% ({:?})\n  tripeptide HC  : CA={:.1}% ({:?}) CB={:.1}% ({:?}) C={:.1}% ({:?})",
        pct(&ag_std, "CA"),
        ag_std.get("CA"),
        pct(&ag_std, "CB"),
        ag_std.get("CB"),
        pct(&ag_std, "C"),
        ag_std.get("C"),
        pct(&ag_tri, "CA"),
        ag_tri.get("CA"),
        pct(&ag_tri, "CB"),
        ag_tri.get("CB"),
        pct(&ag_tri, "C"),
        ag_tri.get("C"),
        pct(&ag_tri_hc, "CA"),
        ag_tri_hc.get("CA"),
        pct(&ag_tri_hc, "CB"),
        ag_tri_hc.get("CB"),
        pct(&ag_tri_hc, "C"),
        ag_tri_hc.get("C"),
    );

    assert!(
        pct(&ag_tri, "CA") >= pct(&ag_std, "CA") + 12.0
            && pct(&ag_tri, "CB") >= pct(&ag_std, "CB") + 7.5
            && pct(&ag_tri, "C") >= pct(&ag_std, "C") + 5.0
            && pct(&ag_tri, "CA") >= 80.0,
        "tripeptide solver should substantially outperform unaligned standard solver across CA/CB/C on real Bruker ubiquitin: tri CA={:.1}% CB={:.1}% C={:.1}% vs std CA={:.1}% CB={:.1}% C={:.1}%",
        pct(&ag_tri, "CA"),
        pct(&ag_tri, "CB"),
        pct(&ag_tri, "C"),
        pct(&ag_std, "CA"),
        pct(&ag_std, "CB"),
        pct(&ag_std, "C")
    );
    assert!(
        pct(&ag_tri_hc, "CA") >= 74.0 && pct(&ag_tri_hc, "CB") >= 74.0,
        "high-confidence real ubiquitin CA/CB cross-sample BMRB 17769 agreement should be >= 74%, got CA={:.1}% CB={:.1}%",
        pct(&ag_tri_hc, "CA"),
        pct(&ag_tri_hc, "CB")
    );
}
