//! Phase 1 tests: empirical BMRB priors, strict train/benchmark disjointness,
//! peak intensity TSV & picker roundtrip, `HNCA` `1J`/`2J` intensity tie-breaking,
//! 10-seed intensity-gated `degenerate` benchmark, and 12-protein real BMRB benchmark suite.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use nmr_core::evaluate::score;
use nmr_core::nmrstar::{read_star, write_star_with_header};
use nmr_core::peaks::{read_peaks, read_peaks_str, write_peaks, Peak};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{
    aa_all_atom_prior, aa_prior, is_standard_residue, Priors, ATOMS, STANDARD_RESIDUES,
    TRAINING_BMRB_IDS,
};
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("nmrassign_phase1_{}_{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, Clone)]
struct ManifestEntry {
    bmrb_id: String,
    pdb_id: String,
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
        if trimmed.is_empty() {
            continue;
        }
        if idx == 0 {
            assert_eq!(
                trimmed,
                "bmrb_id\tpdb_id\tname\tfold_class\tn_residues\tpath"
            );
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 6, "manifest line {}: {:?}", idx + 1, line);
        out.push(ManifestEntry {
            bmrb_id: cols[0].to_string(),
            pdb_id: cols[1].to_string(),
            name: cols[2].to_string(),
            fold_class: cols[3].to_string(),
            n_residues: cols[4].parse().expect("valid n_residues"),
            path: root.join(cols[5]),
        });
    }
    out
}

#[test]
fn test_priors_provenance_and_zero_benchmark_overlap() {
    let root = workspace_root();
    let manifest = load_manifest();
    assert_eq!(manifest.len(), 12, "expected 12 curated benchmark proteins");
    assert!(
        TRAINING_BMRB_IDS.len() >= 20,
        "expected >= 20 training BMRB entries, got {}",
        TRAINING_BMRB_IDS.len()
    );

    let train_set: BTreeSet<String> = TRAINING_BMRB_IDS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        train_set.len(),
        TRAINING_BMRB_IDS.len(),
        "duplicate training IDs"
    );

    let bench_set: BTreeSet<String> = manifest.iter().map(|m| m.bmrb_id.clone()).collect();
    assert_eq!(bench_set.len(), 12, "duplicate benchmark IDs");

    let overlap: Vec<&String> = train_set.intersection(&bench_set).collect();
    assert!(
        overlap.is_empty(),
        "NON-ZERO overlap between training priors and benchmark suite: {:?}",
        overlap
    );

    // Check PROVENANCE.md documents all training IDs
    let prov = fs::read_to_string(root.join("data/priors/PROVENANCE.md"))
        .expect("read data/priors/PROVENANCE.md");
    for tid in &TRAINING_BMRB_IDS {
        assert!(
            prov.contains(tid),
            "PROVENANCE.md missing training BMRB ID {tid}"
        );
    }

    // Check empirical backbone & all-atom JSON files match embedded priors
    let disk_bb = Priors::from_json(root.join("data/priors/bmrb_priors.json")).unwrap();
    let emb_bb = Priors::bmrb();
    for &(res, _) in &STANDARD_RESIDUES {
        for &atom in &ATOMS {
            if (res == "GLY" && atom == "CB") || (res == "PRO" && atom == "H") {
                assert!(!emb_bb.has(res, atom));
                assert!(aa_prior(res, atom).is_none());
            } else {
                let (mu, sd) = aa_prior(res, atom)
                    .unwrap_or_else(|| panic!("missing empirical prior for {res} {atom}"));
                assert_eq!(disk_bb.get(res, atom), Some((mu, sd)));
                assert!(
                    sd > 0.1 && sd < 10.0,
                    "implausible sd for {res} {atom}: {sd}"
                );
            }
        }
    }

    // Verify key sidechain atoms exist in all-atom priors for Phase 3 sidechain work
    let all_pri = Priors::all_atoms();
    for (res, atom) in [
        ("ALA", "HA"),
        ("ALA", "HB1"),
        ("GLY", "HA2"),
        ("GLY", "HA3"),
        ("ILE", "CG1"),
        ("ILE", "CG2"),
        ("ILE", "CD1"),
        ("ILE", "HD11"),
        ("LEU", "CG"),
        ("LEU", "CD1"),
        ("LEU", "CD2"),
        ("LYS", "CE"),
        ("LYS", "NZ"),
        ("LYS", "HE2"),
        ("PHE", "CD1"),
        ("PHE", "HD1"),
        ("THR", "CG2"),
        ("THR", "HG21"),
        ("VAL", "CG1"),
        ("VAL", "HG11"),
    ] {
        assert!(
            all_pri.has(res, atom) && aa_all_atom_prior(res, atom).is_some(),
            "missing all-atom prior for {res} {atom}"
        );
    }
    assert!(all_pri.table.contains_key("CYS_REDUCED"));
    assert!(all_pri.table.contains_key("CYS_OXIDIZED"));
}

#[test]
fn test_peak_intensities_tsv_picker_and_rng_invariance() {
    let dir = temp_dir("tsv_intensities");
    let fpath = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&fpath, None).unwrap();

    // 1. Verify simulator coordinate & truth invariance when include_intensities toggles
    let (peaks_with, truth_with) = simulate(
        &ref_data,
        &SimConfig {
            include_intensities: true,
            seed: 0,
            ..SimConfig::default()
        },
    );
    let (peaks_no, truth_no) = simulate(
        &ref_data,
        &SimConfig {
            include_intensities: false,
            seed: 0,
            ..SimConfig::default()
        },
    );
    assert_eq!(truth_with, truth_no);
    assert_eq!(peaks_with.len(), peaks_no.len());
    for (a, b) in peaks_with.iter().zip(peaks_no.iter()) {
        assert_eq!(a.exp, b.exp);
        assert_eq!(a.h, b.h);
        assert_eq!(a.n, b.n);
        assert_eq!(a.c, b.c);
        assert_eq!(a.sign, b.sign);
        assert!(a.height.is_some() && a.volume.is_some());
        assert!(b.height.is_none() && b.volume.is_none());
    }

    // 2. Verify backward-compatible 5-column TSV and 7-column intensity TSV roundtrips
    let tsv5 =
        "experiment\tH\tN\tC\tsign\nHSQC\t8.3120\t121.450\t\t1\nHNCA\t8.3090\t121.470\t56.210\t1\n";
    let parsed5 = read_peaks_str(tsv5).unwrap();
    assert_eq!(parsed5.len(), 2);
    assert!(parsed5[0].height.is_none() && parsed5[1].volume.is_none());

    let out5 = dir.join("peaks5.tsv");
    write_peaks(&out5, &peaks_no).unwrap();
    let head5 = fs::read_to_string(&out5).unwrap();
    assert!(head5.starts_with("experiment\tH\tN\tC\tsign\n"));

    let out7 = dir.join("peaks7.tsv");
    write_peaks(&out7, &peaks_with).unwrap();
    let head7 = fs::read_to_string(&out7).unwrap();
    assert!(head7.starts_with("experiment\tH\tN\tC\tsign\theight\tvolume\n"));
    let back7 = read_peaks(&out7).unwrap();
    assert_eq!(back7.len(), peaks_with.len());
    for (orig, got) in peaks_with.iter().zip(back7.iter()) {
        assert!((orig.height.unwrap() - got.height.unwrap()).abs() < 0.02);
        assert!((orig.volume.unwrap() - got.volume.unwrap()).abs() < 0.02);
    }

    // 3. Verify classical 2D/3D picker populates positive height and volume
    let rcfg = SpectrumRenderConfig {
        nx_h: 256,
        ny_n: 128,
        nz_c: 96,
        noise_level: 1.0,
        seed: 7,
        ..SpectrumRenderConfig::default()
    };
    let hsqc_2d = render_hsqc_2d(&peaks_with, &rcfg);
    let pcfg = PickerConfig {
        snr_threshold: 6.0,
        ..PickerConfig::default()
    };
    let picked_roots = pick_hsqc_2d(&hsqc_2d, &pcfg);
    assert!(!picked_roots.is_empty());
    assert!(picked_roots
        .iter()
        .all(|p| p.height.unwrap_or(0.0) > 0.0 && p.volume.unwrap_or(0.0) >= p.height.unwrap()));

    let hnca_3d = render_triple_3d("HNCA", &peaks_with, &rcfg);
    let picked_hnca = pick_triple_3d(&hnca_3d, &picked_roots, &pcfg);
    assert!(picked_hnca.len() >= 80);
    assert!(picked_hnca
        .iter()
        .all(|p| p.height.unwrap_or(0.0) > 0.0 && p.volume.unwrap_or(0.0) >= p.height.unwrap()));

    // Verify write_star_with_header preserves BMRB/PDB header comments and exact sequence
    let star_hdr_path = dir.join("with_hdr.str");
    write_star_with_header(
        &star_hdr_path,
        &ref_data,
        None,
        Some("BMRB ID: bmr17769\nPDB ID: 1D3Z"),
    )
    .unwrap();
    let star_txt = fs::read_to_string(&star_hdr_path).unwrap();
    assert!(star_txt.contains("# BMRB ID: bmr17769") && star_txt.contains("# PDB ID: 1D3Z"));
    let round_ref = read_star(&star_hdr_path, None).unwrap();
    assert_eq!(round_ref.sequence, ref_data.sequence);
    assert_eq!(round_ref.shifts.len(), ref_data.shifts.len());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_hnca_only_intensity_disambiguation_gain() {
    let fpath = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(&fpath, None).unwrap();
    let priors = Priors::bmrb();
    let scfg = ScoreConfig::default();

    let mut acc_with_sum = 0.0;
    let mut acc_no_sum = 0.0;
    let mut resolved_with = 0usize;
    let mut resolved_no = 0usize;
    let seeds = 5u64;

    for seed in 0..seeds {
        let exps = vec!["HSQC".to_string(), "HNCA".to_string()];
        let (peaks_with, truth) = simulate(
            &ref_data,
            &SimConfig {
                experiments: exps.clone(),
                p_missing_peak: 0.02,
                spurious_rate: 0.02,
                include_intensities: true,
                seed,
                ..SimConfig::default()
            },
        );
        let peaks_no: Vec<Peak> = peaks_with
            .iter()
            .cloned()
            .map(|mut p| {
                p.height = None;
                p.volume = None;
                p
            })
            .collect();

        let sys_with = build_spin_systems(&peaks_with, &Tolerances::default(), 3, false).unwrap();
        let sys_no = build_spin_systems(&peaks_no, &Tolerances::default(), 3, false).unwrap();

        // Count roots where CA(0) and CA(-1) are distinctly disambiguated instead of symmetric
        resolved_with += sys_with
            .iter()
            .filter(|ss| {
                ss.cands.get(&("CA".to_string(), 0)) != ss.cands.get(&("CA".to_string(), -1))
            })
            .count();
        resolved_no += sys_no
            .iter()
            .filter(|ss| {
                ss.cands.get(&("CA".to_string(), 0)) != ss.cands.get(&("CA".to_string(), -1))
            })
            .count();

        let acfg = AnnealConfig {
            runs: 16,
            steps_per_system: 2000,
            seed,
            ..AnnealConfig::default()
        };
        let m_with = score(
            &assign(&sys_with, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        );
        let m_no = score(
            &assign(&sys_no, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        );
        acc_with_sum += m_with.accuracy();
        acc_no_sum += m_no.accuracy();
    }

    let mean_with = acc_with_sum / (seeds as f64);
    let mean_no = acc_no_sum / (seeds as f64);
    println!(
        "[HNCA-only 5-seed mean] disambiguated roots: with_intensities={} vs without={} | accuracy: {:.1}% vs {:.1}%",
        resolved_with,
        resolved_no,
        mean_with * 100.0,
        mean_no * 100.0
    );
    assert!(
        resolved_with > 200 && resolved_no == 0,
        "expected 1J/2J intensity ratio to disambiguate HNCA i vs i-1 roots: with={resolved_with}, no={resolved_no}"
    );
    assert!(
        mean_with > mean_no + 0.14,
        "expected substantial HNCA-only accuracy gain with intensities: with={:.3}, without={:.3}",
        mean_with,
        mean_no
    );

    // Also test triple-resonance set without HNCOCA (`HSQC + HNCA + HNCO + HNCACO`):
    // `CA(i)` vs `CA(i-1)` relies exclusively on `HNCA` 1J/2J intensity ratios while `CO` links normally.
    let mut min_with_sum = 0.0;
    let mut min_no_sum = 0.0;
    for seed in 0..seeds {
        let exps = vec![
            "HSQC".to_string(),
            "HNCA".to_string(),
            "HNCO".to_string(),
            "HNCACO".to_string(),
        ];
        let (peaks_with, truth) = simulate(
            &ref_data,
            &SimConfig {
                experiments: exps,
                p_missing_peak: 0.02,
                spurious_rate: 0.02,
                include_intensities: true,
                seed,
                ..SimConfig::default()
            },
        );
        let peaks_no: Vec<Peak> = peaks_with
            .iter()
            .cloned()
            .map(|mut p| {
                p.height = None;
                p.volume = None;
                p
            })
            .collect();
        let sys_with = build_spin_systems(&peaks_with, &Tolerances::default(), 3, false).unwrap();
        let sys_no = build_spin_systems(&peaks_no, &Tolerances::default(), 3, false).unwrap();
        let acfg = AnnealConfig {
            runs: 16,
            steps_per_system: 2000,
            seed,
            ..AnnealConfig::default()
        };
        min_with_sum += score(
            &assign(&sys_with, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        )
        .accuracy();
        min_no_sum += score(
            &assign(&sys_no, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        )
        .accuracy();
    }
    let min_mean_with = min_with_sum / (seeds as f64);
    let min_mean_no = min_no_sum / (seeds as f64);
    println!(
        "[HSQC+HNCA+HNCO+HNCACO (no HNCOCA) 5-seed mean] accuracy: with_intensities={:.1}% vs without={:.1}%",
        min_mean_with * 100.0,
        min_mean_no * 100.0
    );
    assert!(
        min_mean_with > min_mean_no + 0.08,
        "expected accuracy gain without HNCOCA when HNCA intensities are used: with={:.3}, without={:.3}",
        min_mean_with,
        min_mean_no
    );
}

#[test]
fn test_10_seed_intensity_gated_degenerate_benchmark() {
    let fpath = workspace_root().join("data/benchmark/bmr17769.str");
    let ref_data = read_star(&fpath, None).unwrap();
    let priors = Priors::bmrb();
    let scfg = ScoreConfig::default();

    let mut acc_off = 0.0;
    let mut rec_off = 0.0;
    let mut hcp_off = 0.0;

    let mut acc_ungated = 0.0;
    let mut rec_ungated = 0.0;
    let mut hcp_ungated = 0.0;
    let mut deg_cands_ungated = 0usize;

    let mut acc_gated = 0.0;
    let mut rec_gated = 0.0;
    let mut hcp_gated = 0.0;
    let mut deg_cands_gated = 0usize;

    for seed in 0..10u64 {
        let (peaks_int, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: 0.08,
                spurious_rate: 0.04,
                include_intensities: true,
                seed,
                ..SimConfig::default()
            },
        );
        let peaks_noint: Vec<Peak> = peaks_int
            .iter()
            .cloned()
            .map(|mut p| {
                p.height = None;
                p.volume = None;
                p
            })
            .collect();

        let sys_off = build_spin_systems(&peaks_int, &Tolerances::default(), 3, false).unwrap();
        let sys_ungated =
            build_spin_systems(&peaks_noint, &Tolerances::default(), 3, true).unwrap();
        let sys_gated = build_spin_systems(&peaks_int, &Tolerances::default(), 3, true).unwrap();

        deg_cands_ungated += sys_ungated
            .iter()
            .flat_map(|ss| ss.cands.values())
            .flatten()
            .filter(|&&(_, sup)| sup == 0)
            .count();
        deg_cands_gated += sys_gated
            .iter()
            .flat_map(|ss| ss.cands.values())
            .flatten()
            .filter(|&&(_, sup)| sup == 0)
            .count();

        let acfg = AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed,
            ..AnnealConfig::default()
        };

        let m_off = score(
            &assign(&sys_off, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        );
        let m_un = score(
            &assign(&sys_ungated, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        );
        let m_ga = score(
            &assign(&sys_gated, &ref_data.sequence, &priors, &scfg, &acfg),
            &truth,
            0.8,
        );

        acc_off += m_off.accuracy();
        rec_off += m_off.recall();
        hcp_off += m_off.hc_precision();

        acc_ungated += m_un.accuracy();
        rec_ungated += m_un.recall();
        hcp_ungated += m_un.hc_precision();

        acc_gated += m_ga.accuracy();
        rec_gated += m_ga.recall();
        hcp_gated += m_ga.hc_precision();
    }

    println!(
        "[10-seed degenerate benchmark on bmr17769]\n\
         - degenerate=false           : acc={:.2}%  rec={:.2}%  hc_prec={:.2}%\n\
         - degenerate=true (ungated)  : acc={:.2}%  rec={:.2}%  hc_prec={:.2}%  (deg_cands={})\n\
         - degenerate=true (gated)    : acc={:.2}%  rec={:.2}%  hc_prec={:.2}%  (deg_cands={})",
        acc_off * 10.0,
        rec_off * 10.0,
        hcp_off * 10.0,
        acc_ungated * 10.0,
        rec_ungated * 10.0,
        hcp_ungated * 10.0,
        deg_cands_ungated,
        acc_gated * 10.0,
        rec_gated * 10.0,
        hcp_gated * 10.0,
        deg_cands_gated
    );

    // Intensity gating (>= 1.65x median) must reject missing-peak false positives while keeping real superposed peaks
    assert!(
        deg_cands_gated > 0 && deg_cands_gated < deg_cands_ungated,
        "expected intensity gating to filter missing-peak false degenerates: gated={deg_cands_gated}, ungated={deg_cands_ungated}"
    );
    assert!(
        acc_gated >= acc_ungated && rec_gated >= rec_ungated && acc_gated / 10.0 >= 0.96,
        "intensity-gated degenerate must outperform or match ungated and exceed 96% accuracy"
    );
}

#[test]
fn test_benchmark_suite_12_real_bmrb_proteins_and_synthetic_ubiquitin() {
    let root = workspace_root();
    let priors = Priors::bmrb();
    let scfg = ScoreConfig::default();

    // 1. Strict synthetic_ubiquitin.str non-regression gate
    let syn_ref = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None).unwrap();
    let (syn_peaks, syn_truth) = simulate(
        &syn_ref,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let syn_sys = build_spin_systems(&syn_peaks, &Tolerances::default(), 3, false).unwrap();
    let syn_res = assign(
        &syn_sys,
        &syn_ref.sequence,
        &priors,
        &scfg,
        &AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let syn_m = score(&syn_res, &syn_truth, 0.8);
    println!(
        "[synthetic_ubiquitin] acc={:.1}% rec={:.1}% hc_prec={:.1}% ({}/{})",
        syn_m.accuracy() * 100.0,
        syn_m.recall() * 100.0,
        syn_m.hc_precision() * 100.0,
        syn_m.hc_correct,
        syn_m.hc_assigned
    );
    assert!(
        syn_m.accuracy() >= 0.970 && syn_m.hc_precision() >= 0.979,
        "synthetic_ubiquitin regression: {}",
        syn_m.report()
    );

    // 2. Run across all 12 curated real BMRB benchmark entries
    let manifest = load_manifest();
    let mut sum_acc = 0.0;
    let mut sum_rec = 0.0;
    let mut sum_hcp = 0.0;

    println!("\nBMRB_ID   PDB   FOLD        LEN  ASSIGNABLE  ACCURACY  RECALL  HC_PREC");
    let mut per_entry_metrics = Vec::new();
    for entry in &manifest {
        let header_txt = fs::read_to_string(&entry.path).expect("read benchmark .str");
        assert!(
            header_txt.contains(&format!("BMRB ID: {}", entry.bmrb_id)),
            "{} missing BMRB ID header",
            entry.bmrb_id
        );
        assert!(
            header_txt.contains(&format!("PDB ID: {}", entry.pdb_id)),
            "{} missing PDB ID header",
            entry.bmrb_id
        );

        let ref_data = read_star(&entry.path, None).expect("parse benchmark .str");
        assert_eq!(
            ref_data.sequence.len(),
            entry.n_residues,
            "{} sequence length mismatch vs manifest.tsv",
            entry.bmrb_id
        );
        assert!(
            ref_data.sequence.iter().all(|r| is_standard_residue(r)),
            "{} contains non-standard/UNK residue: {:?}",
            entry.bmrb_id,
            ref_data.sequence
        );
        for atom in ATOMS {
            let count = ref_data.shifts.keys().filter(|(_, a)| a == atom).count();
            assert!(
                count >= entry.n_residues / 2,
                "{} incomplete {atom} shifts ({count}/{})",
                entry.bmrb_id,
                entry.n_residues
            );
        }

        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed: 0,
                ..SimConfig::default()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, true).unwrap();
        let res = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &scfg,
            &AnnealConfig {
                runs: 12,
                steps_per_system: 1500,
                seed: 0,
                ..AnnealConfig::default()
            },
        );
        let m = score(&res, &truth, 0.8);
        println!(
            "{:<9} {:<5} {:<10} {:>4}  {:>10}  {:>7.1}%  {:>5.1}%  {:>6.1}% ({}/{})  {}",
            entry.bmrb_id,
            entry.pdb_id,
            entry.fold_class,
            entry.n_residues,
            m.assignable,
            m.accuracy() * 100.0,
            m.recall() * 100.0,
            m.hc_precision() * 100.0,
            m.hc_correct,
            m.hc_assigned,
            entry.name
        );

        sum_acc += m.accuracy();
        sum_rec += m.recall();
        sum_hcp += m.hc_precision();
        per_entry_metrics.push((entry, m));
    }

    let n = manifest.len() as f64;
    let mean_acc = sum_acc / n;
    let mean_rec = sum_rec / n;
    let mean_hcp = sum_hcp / n;
    println!(
        "MEAN (12 BMRB entries): accuracy={:.1}%  recall={:.1}%  hc_precision={:.1}%",
        mean_acc * 100.0,
        mean_rec * 100.0,
        mean_hcp * 100.0
    );

    for (entry, m) in &per_entry_metrics {
        // Unfolded / intrinsically disordered proteins (`bmr4375` 8M-urea unfolded ubiquitin,
        // `bmr15176` unstructured DARPP-32, `bmr6968` IDP alpha-synuclein) have collapsed ~0.6 ppm
        // 1H dispersion; 148-aa all-helical EF-hand calmodulin (`bmr6541`) has pseudo-symmetric lobes.
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
            m.accuracy() >= min_acc,
            "{} accuracy {:.1}% below floor {:.1}%",
            entry.bmrb_id,
            m.accuracy() * 100.0,
            min_acc * 100.0
        );
        assert!(
            m.hc_precision() >= 0.90,
            "{} HC precision {:.1}% below 90%",
            entry.bmrb_id,
            m.hc_precision() * 100.0
        );
    }
    assert!(mean_acc >= 0.85 && mean_hcp >= 0.98);
}
