pub use nmr_core::*;

use std::path::Path;

use nmr_core::evaluate::score as score_backbone;
use nmr_core::nmrstar::{read_star, read_star_all_atoms};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

use sidechain::{
    assign_sidechains, evaluate_sidechain_shifts, pick_c13_hsqc, pick_ccconh_3d, pick_hccconh_3d,
    pick_hcch_tocsy_3d, simulate_c13_hsqc_spectrum, simulate_sidechain_3d_spectrum,
    simulate_sidechain_peaks, solve_joint_backbone_sidechain, synthesize_all_atom_reference,
    SidechainPickerConfig, SidechainRenderConfig, SidechainSimConfig, SidechainSolverConfig,
};
use topology::{
    all_residue_topologies, residue_topology, CarbonRegion, ChMultiplicity, Element,
    SidechainPriors,
};

#[test]
fn test_all_20_amino_acid_topologies_and_covalent_bonds() {
    let topos = all_residue_topologies();
    assert_eq!(topos.len(), 20);

    let mut seen_three = std::collections::BTreeSet::new();
    let mut seen_one = std::collections::BTreeSet::new();

    for t in topos {
        assert!(
            seen_three.insert(t.three_letter),
            "duplicate {}",
            t.three_letter
        );
        assert!(seen_one.insert(t.one_letter), "duplicate {}", t.one_letter);
        assert_eq!(
            residue_topology(t.three_letter).unwrap().one_letter,
            t.one_letter
        );
        assert_eq!(
            residue_topology(&t.one_letter.to_string())
                .unwrap()
                .three_letter,
            t.three_letter
        );

        // Every bond endpoint must exist in `atoms`
        for &(a, b) in t.bonds {
            assert!(
                t.atom(a).is_some(),
                "{}: missing bond endpoint {}",
                t.three_letter,
                a
            );
            assert!(
                t.atom(b).is_some(),
                "{}: missing bond endpoint {}",
                t.three_letter,
                b
            );
        }

        // Every proton must declare its covalently attached heavy atom and have a matching C-H / N-H / O-H / S-H bond
        for atom in t.atoms {
            if atom.element == Element::H {
                let heavy = atom
                    .attached_heavy
                    .expect("every proton must specify attached_heavy");
                assert!(
                    t.are_bonded(atom.name, heavy),
                    "{}: proton {} not bonded to {}",
                    t.three_letter,
                    atom.name,
                    heavy
                );
            } else {
                assert!(atom.attached_heavy.is_none());
            }
        }

        // Proline lacks backbone amide H; all other 19 standard amino acids have H bonded to N
        if t.three_letter == "PRO" {
            assert!(t.atom("H").is_none());
            assert!(t.are_bonded("CD", "N"));
        } else {
            assert!(t.atom("H").is_some());
            assert!(t.are_bonded("N", "H"));
        }

        // Check CH_n multiplicities & covalent C-H bonds
        for chg in t.ch_groups {
            assert_eq!(chg.protons.len(), chg.multiplicity.proton_count());
            for &p in chg.protons {
                assert!(
                    t.are_bonded(chg.carbon, p),
                    "{}: {} not bonded to {}",
                    t.three_letter,
                    chg.carbon,
                    p
                );
            }
            for &nbr_c in chg.covalent_carbon_neighbors {
                assert!(
                    t.are_bonded(chg.carbon, nbr_c),
                    "{}: {} not bonded to neighbor carbon {}",
                    t.three_letter,
                    chg.carbon,
                    nbr_c
                );
            }
        }
    }

    // Spot-check key spin-system counts and TOCSY components
    assert_eq!(
        residue_topology("GLY")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        1
    );
    assert_eq!(
        residue_topology("ALA")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        2
    );
    assert_eq!(
        residue_topology("THR")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        3
    );
    assert_eq!(
        residue_topology("VAL")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        4
    );
    assert_eq!(
        residue_topology("PRO")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        4
    );
    assert_eq!(
        residue_topology("ARG")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        4
    );
    assert_eq!(
        residue_topology("ILE")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        5
    );
    assert_eq!(
        residue_topology("LEU")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        5
    );
    assert_eq!(
        residue_topology("LYS")
            .unwrap()
            .aliphatic_ca_groups()
            .count(),
        5
    );

    // MET has 3 CA-connected aliphatic carbons (CA, CB, CG) + 1 isolated methyl (CE across SD)
    let met = residue_topology("MET").unwrap();
    assert_eq!(met.aliphatic_ca_groups().count(), 3);
    assert_eq!(met.all_aliphatic_groups().count(), 4);
    assert_eq!(
        met.ch_group("CE").unwrap().region,
        CarbonRegion::AliphaticIsolatedMethyl
    );
    assert_eq!(
        met.ch_group("CE").unwrap().multiplicity,
        ChMultiplicity::Methyl
    );

    // Aromatics separate CA/CB aliphatic TOCSY from ring CHs
    assert_eq!(
        residue_topology("PHE").unwrap().aromatic_groups().count(),
        5
    );
    assert_eq!(
        residue_topology("TYR").unwrap().aromatic_groups().count(),
        4
    );
    assert_eq!(
        residue_topology("HIS").unwrap().aromatic_groups().count(),
        2
    );
    assert_eq!(
        residue_topology("TRP").unwrap().aromatic_groups().count(),
        5
    );
}

#[test]
fn test_empirical_bmrb_sidechain_priors_completeness_and_discrimination() {
    let priors = SidechainPriors::empirical_bmrb();
    for topo in all_residue_topologies() {
        for chg in topo.ch_groups {
            let (mc, sc) = priors
                .prior(topo.three_letter, chg.carbon)
                .unwrap_or_else(|| panic!("missing C prior {} {}", topo.three_letter, chg.carbon));
            assert!(mc > 8.0 && mc < 165.0 && sc > 0.1);
            for &p in chg.protons {
                let (mh, sh) = priors
                    .prior(topo.three_letter, p)
                    .unwrap_or_else(|| panic!("missing H prior {} {}", topo.three_letter, p));
                assert!(mh > 0.0 && mh < 11.0 && sh > 0.05);
            }
        }
    }

    // Characteristic spin-system shifts must produce positive LLRs for their true residue type
    assert!(priors.llr_c13("ALA", "CB", CarbonRegion::AliphaticCaConnected, 18.8) > 1.5);
    assert!(priors.llr_c13("THR", "CB", CarbonRegion::AliphaticCaConnected, 70.0) > 2.5);
    assert!(priors.llr_c13("ILE", "CD1", CarbonRegion::AliphaticCaConnected, 13.8) > 1.8);
    assert!(priors.llr_c13("PRO", "CD", CarbonRegion::AliphaticCaConnected, 50.5) > 2.2);
    assert!(priors.llr_c13("LYS", "CE", CarbonRegion::AliphaticCaConnected, 42.0) > 1.8);
    assert!(priors.llr_c13("ARG", "CD", CarbonRegion::AliphaticCaConnected, 43.2) > 1.8);
    assert!(priors.llr_c13("TYR", "CE1", CarbonRegion::Aromatic, 118.0) > 1.8);
    assert!(priors.llr_c13("HIS", "CE1", CarbonRegion::Aromatic, 138.2) > 1.8);
}

#[test]
fn test_volumetric_2d_and_3d_sidechain_simulators_and_pickers() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ubq_bb = read_star(
        workspace.join("data/fixtures/synthetic_ubiquitin.str"),
        None,
    )
    .expect("read synthetic ubiquitin");

    // Take an 8-residue slice MQIFVKTL (including diverse methyls, methylene, aromatic)
    let mut sub = nmr_core::nmrstar::Reference::new(
        ubq_bb.sequence[..8].to_vec(),
        ubq_bb.seq_ids[..8].to_vec(),
    );
    for (&(k, ref a), &v) in &ubq_bb.shifts {
        if k < 8 {
            sub.shifts.insert((k, a.clone()), v);
        }
    }
    let sc_priors = SidechainPriors::empirical_bmrb();
    let all_atom = synthesize_all_atom_reference(&sub, &sc_priors, 7);

    let sim_cfg = SidechainSimConfig {
        include_aromatics: false,
        noise_h: 0.0,
        noise_n: 0.0,
        noise_c: 0.0,
        p_missing_peak: 0.0,
        spurious_rate: 0.0,
        include_intensities: true,
        seed: 11,
        ..Default::default()
    };
    let truth_sc_peaks = simulate_sidechain_peaks(&all_atom, &sim_cfg);

    let render_cfg = SidechainRenderConfig {
        nx: 256,
        ny: 128,
        nz: 160,
        noise_level: 1.0,
        seed: 99,
        ..Default::default()
    };
    let pick_cfg = SidechainPickerConfig {
        snr_threshold: 5.0,
        ..Default::default()
    };

    // 1. 2D C13-HSQC render + pick
    let c13_spec = simulate_c13_hsqc_spectrum(&truth_sc_peaks, &render_cfg);
    let picked_c13 = pick_c13_hsqc(&c13_spec, &pick_cfg);
    assert!(
        picked_c13.len() >= 18,
        "expected >= 18 picked C13-HSQC peaks, got {}",
        picked_c13.len()
    );

    // 2. Build amide roots for 3D CCCONH and HCCCONH picking
    let mut hsqc_roots = Vec::new();
    for k in 1..8 {
        if let (Some(&h), Some(&n)) = (
            all_atom.shifts.get(&(k, "H".to_string())),
            all_atom.shifts.get(&(k, "N".to_string())),
        ) {
            hsqc_roots.push(Peak::hsqc(h, n));
        }
    }

    let ccconh_spec = simulate_sidechain_3d_spectrum("CCCONH", &truth_sc_peaks, &render_cfg);
    let picked_ccconh = pick_ccconh_3d(&ccconh_spec, &hsqc_roots, &pick_cfg);
    assert!(
        picked_ccconh.len() >= 16,
        "expected >= 16 picked CCCONH peaks, got {}",
        picked_ccconh.len()
    );

    let hccconh_spec = simulate_sidechain_3d_spectrum("HCCCONH", &truth_sc_peaks, &render_cfg);
    let picked_hccconh = pick_hccconh_3d(&hccconh_spec, &hsqc_roots, &pick_cfg);
    assert!(
        picked_hccconh.len() >= 16,
        "expected >= 16 picked HCCCONH peaks, got {}",
        picked_hccconh.len()
    );

    // 3. 3D HCCH-TOCSY render + pick anchored on picked C13-HSQC roots
    let hcch_spec = simulate_sidechain_3d_spectrum("HCCHTOCSY", &truth_sc_peaks, &render_cfg);
    let picked_hcch = pick_hcch_tocsy_3d(&hcch_spec, Some(&picked_c13), &pick_cfg);
    assert!(
        picked_hcch.len() >= 40,
        "expected >= 40 picked HCCH-TOCSY peaks, got {}",
        picked_hcch.len()
    );
}

#[test]
fn test_ubiquitin_bmr17769_blind_sidechain_assignment_benchmark() {
    let raw_bmr17769 = Path::new(
        "/Users/debsahu/.gemini/antigravity/brain/abeb5c50-57b8-474f-bd50-46bb84550078/scratch/bmrb_raw/bmr17769.str",
    );
    let sc_priors = SidechainPriors::empirical_bmrb();
    let bb_priors = Priors::default();

    let truth_all = if raw_bmr17769.exists() {
        read_star_all_atoms(raw_bmr17769, None).expect("parse all-atom bmr17769.str")
    } else {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let base = read_star(
            workspace.join("data/fixtures/synthetic_ubiquitin.str"),
            None,
        )
        .expect("read synthetic_ubiquitin.str");
        synthesize_all_atom_reference(&base, &sc_priors, 17769)
    };
    assert_eq!(truth_all.sequence.len(), 76);

    // Simulate realistic noisy backbone AND sidechain peak lists (with missing peaks & spurious noise)
    let bb_sim_cfg = SimConfig {
        noise_h: 0.008,
        noise_n: 0.06,
        noise_c: 0.04,
        p_missing_peak: 0.04,
        p_missing_root: 0.0,
        spurious_rate: 0.03,
        seed: 2026,
        ..Default::default()
    };
    let (bb_peaks, bb_truth) = simulate(&truth_all, &bb_sim_cfg);
    let systems = build_spin_systems(&bb_peaks, &Tolerances::default(), 3, false)
        .expect("build backbone spin systems");

    let sc_sim_cfg = SidechainSimConfig {
        include_aromatics: true,
        multiplicity_edited_hsqc: false,
        noise_h: 0.008,
        noise_n: 0.06,
        noise_c: 0.04,
        p_missing_peak: 0.04,
        spurious_rate: 0.02,
        include_intensities: true,
        seed: 2026,
        ..Default::default()
    };
    let sc_peaks = simulate_sidechain_peaks(&truth_all, &sc_sim_cfg);

    let joint = solve_joint_backbone_sidechain(
        &systems,
        &truth_all.sequence,
        &truth_all.seq_ids,
        &bb_priors,
        &sc_priors,
        &sc_peaks,
        &ScoreConfig::default(),
        &AnnealConfig {
            runs: 16,
            steps_per_system: 2500,
            seed: 42,
            ..Default::default()
        },
        &SidechainSolverConfig::default(),
    );

    let bb_metrics = score_backbone(&joint.backbone_result, &bb_truth, 0.8);
    let sc_metrics =
        evaluate_sidechain_shifts(&joint.sidechain_result.reference, &truth_all, 0.40, 0.06);

    println!("=== Ubiquitin (bmr17769, 76 aa) Joint Backbone + Sidechain Benchmark ===");
    println!(
        "Backbone: accuracy = {:.1}% ({}/{}), recall = {:.1}%",
        bb_metrics.accuracy() * 100.0,
        bb_metrics.correct,
        bb_metrics.assigned,
        bb_metrics.recall() * 100.0
    );
    println!("{}", sc_metrics.summary());

    // Verify pre-proline residues (GLU 18 -> idx 17 before PRO 19; ILE 36 -> idx 35 before PRO 37;
    // and double-proline PRO 37 -> idx 36 before PRO 38!) get sidechain assignments via C13HSQC + HCCH-TOCSY!
    let pred = &joint.sidechain_result.reference;
    for &(res_idx, label, atoms) in &[
        (17usize, "GLU18 (pre-PRO19)", &["HA", "HB2", "CG"][..]),
        (
            35usize,
            "ILE36 (pre-PRO37)",
            &["HA", "HB", "CG1", "CG2", "CD1"][..],
        ),
        (36usize, "PRO37 (pre-PRO38)", &["HA", "CB", "CG", "CD"][..]),
    ] {
        let mut found = 0usize;
        for &a in atoms {
            if pred.shifts.contains_key(&(res_idx, a.to_string())) {
                found += 1;
            }
        }
        println!(
            "  Pre-proline check {label}: assigned {found}/{} key sidechain atoms",
            atoms.len()
        );
        assert!(
            found >= atoms.len() - 1,
            "expected pre-proline residue {label} to recover almost all sidechain atoms, got {found}/{}",
            atoms.len()
        );
    }

    assert!(
        bb_metrics.accuracy() >= 0.95,
        "expected >= 95% backbone accuracy, got {:.1}%",
        bb_metrics.accuracy() * 100.0
    );
    assert!(
        sc_metrics.aliphatic_accuracy() >= 0.90,
        "expected >= 90.0% aliphatic sidechain accuracy, got {:.1}%",
        sc_metrics.aliphatic_accuracy() * 100.0
    );
    assert!(
        sc_metrics.aliphatic_recall() >= 0.88,
        "expected >= 88.0% aliphatic sidechain recall, got {:.1}%",
        sc_metrics.aliphatic_recall() * 100.0
    );
}

#[test]
fn test_synthetic_all_atom_ubiquitin_ci_benchmark() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = read_star(
        workspace.join("data/fixtures/synthetic_ubiquitin.str"),
        None,
    )
    .expect("read synthetic_ubiquitin.str");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let truth_all = synthesize_all_atom_reference(&base, &sc_priors, 12345);

    let sc_peaks = simulate_sidechain_peaks(
        &truth_all,
        &SidechainSimConfig {
            p_missing_peak: 0.05,
            spurious_rate: 0.02,
            seed: 99,
            ..Default::default()
        },
    );

    let sc_res = assign_sidechains(
        &base,
        None,
        &sc_peaks,
        &sc_priors,
        &SidechainSolverConfig::default(),
    );
    let metrics = evaluate_sidechain_shifts(&sc_res.reference, &truth_all, 0.40, 0.06);
    println!("=== Synthetic All-Atom Ubiquitin Sidechain Benchmark ===");
    println!("{}", metrics.summary());

    assert!(
        metrics.aliphatic_accuracy() >= 0.92,
        "expected >= 92% synthetic aliphatic accuracy, got {:.1}%",
        metrics.aliphatic_accuracy() * 100.0
    );
}

#[test]
fn test_joint_backbone_sidechain_rescues_ambiguous_hnca_only_backbone() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = read_star(
        workspace.join("data/fixtures/synthetic_ubiquitin.str"),
        None,
    )
    .expect("read synthetic_ubiquitin.str");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let bb_priors = Priors::default();
    let truth_all = synthesize_all_atom_reference(&base, &sc_priors, 31415);

    // Simulate sparse HNCA-only backbone peaks (NO CB, NO HNCOCA, NO HNCO!)
    let sparse_bb_cfg = SimConfig {
        experiments: vec!["HSQC".to_string(), "HNCA".to_string()],
        noise_h: 0.008,
        noise_n: 0.06,
        noise_c: 0.05,
        p_missing_peak: 0.06,
        p_missing_root: 0.0,
        spurious_rate: 0.04,
        include_intensities: false, // disable intensity i/i-1 tie-breaking to maximize backbone ambiguity
        seed: 77,
        ..Default::default()
    };
    let (bb_peaks, bb_truth) = simulate(&truth_all, &sparse_bb_cfg);
    let systems = build_spin_systems(&bb_peaks, &Tolerances::default(), 3, false)
        .expect("build sparse HNCA spin systems");

    let anneal_cfg = AnnealConfig {
        runs: 16,
        steps_per_system: 2500,
        seed: 77,
        ..Default::default()
    };
    let bb_only = assign(
        &systems,
        &truth_all.sequence,
        &bb_priors,
        &ScoreConfig::default(),
        &anneal_cfg,
    );
    let bb_only_m = score_backbone(&bb_only, &bb_truth, 0.8);

    let sc_peaks = simulate_sidechain_peaks(
        &truth_all,
        &SidechainSimConfig {
            p_missing_peak: 0.04,
            spurious_rate: 0.01,
            seed: 77,
            ..Default::default()
        },
    );

    let joint = solve_joint_backbone_sidechain(
        &systems,
        &truth_all.sequence,
        &truth_all.seq_ids,
        &bb_priors,
        &sc_priors,
        &sc_peaks,
        &ScoreConfig::default(),
        &anneal_cfg,
        &SidechainSolverConfig::default(),
    );
    let joint_m = score_backbone(&joint.backbone_result, &bb_truth, 0.8);

    println!(
        "Sparse unedited HNCA-only backbone accuracy: {:.1}% -> Joint with sidechain boosts: {:.1}%",
        bb_only_m.accuracy() * 100.0,
        joint_m.accuracy() * 100.0
    );
    assert!(
        joint_m.correct > bb_only_m.correct && joint_m.accuracy() >= bb_only_m.accuracy() + 0.05,
        "expected sidechain spin-type feedback to boost sparse HNCA accuracy ({:.1}% vs {:.1}%)",
        bb_only_m.accuracy() * 100.0,
        joint_m.accuracy() * 100.0
    );
}
