//! Non-circular unit and real-BMRB integration tests for `nmr_core::idp_propensity` (`T11.3`)
//! and the `report.rs` stacked 4-state `δ2D` + `SSP` vector SVG track.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::idp_propensity::{
    compute_idp_propensity, format_propensity_json, format_propensity_tsv, ConformationalState,
    PropensityConfig, PropensityReport, UnassignedPriorMode,
};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::report::{
    generate_quality_report_html, render_idp_propensity_svg, ReportInput, SecondaryStructureClass,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn test_simplex_normalization_and_unassigned_residue_handling() {
    let ubq = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None)
        .expect("read bmr17769.str");
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let urea = read_star(workspace_root().join("data/benchmark/bmr4375.str"), None)
        .expect("read bmr4375.str");

    for (label, ref_data) in [("ubq", &ubq), ("asyn", &asyn), ("urea", &urea)] {
        let rep = compute_idp_propensity(ref_data, &PropensityConfig::default());
        assert_eq!(rep.residues.len(), ref_data.sequence.len());
        for r in &rep.residues {
            let s = r.population_sum();
            assert!(
                (s - 1.0).abs() <= 1e-9,
                "{label} seq_id {} sum={s:.12} deviated from 1.0",
                r.seq_id
            );
            assert!(
                (0.0..=1.0).contains(&r.p_alpha)
                    && (0.0..=1.0).contains(&r.p_beta)
                    && (0.0..=1.0).contains(&r.p_ppii)
                    && (0.0..=1.0).contains(&r.p_coil),
                "{label} seq_id {} probability out of [0,1]",
                r.seq_id
            );
            if r.residue == "GLY" {
                assert!(
                    !r.secondary_shifts.contains_key("CB"),
                    "GLY {} must never carry a CB secondary shift",
                    r.seq_id
                );
            }
        }
    }

    // Create a sparse copy with completely unassigned gap (seq_index 25..=32)
    let mut gapped: Reference = asyn.clone();
    gapped.shifts.retain(|&(k, _), _| !(25..=32).contains(&k));

    let rep_coil = compute_idp_propensity(
        &gapped,
        &PropensityConfig::default().with_unassigned_prior(UnassignedPriorMode::CoilBaseline),
    );
    let rep_unif = compute_idp_propensity(
        &gapped,
        &PropensityConfig::default().with_unassigned_prior(UnassignedPriorMode::Uniform),
    );

    for k in 25..=32 {
        let rc = &rep_coil.residues[k];
        assert!(!rc.has_data, "gap residue k={k} must set has_data = false");
        assert_eq!(rc.n_observed_nuclei, 0);
        assert!(rc.ssp_score.is_none());
        assert!((rc.population_sum() - 1.0).abs() <= 1e-9);
        assert!((rc.p_coil - 0.65).abs() < 1e-9 && (rc.p_ppii - 0.25).abs() < 1e-9);

        let ru = &rep_unif.residues[k];
        assert!(!ru.has_data);
        assert!((ru.population_sum() - 1.0).abs() <= 1e-9);
        assert!((ru.p_alpha - 0.25).abs() < 1e-9 && (ru.p_coil - 0.25).abs() < 1e-9);
    }
}

#[test]
fn test_folded_ubiquitin_bmr17769_vs_1d3z_secondary_structure() {
    let ubq = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None)
        .expect("read bmr17769.str");
    let pdb_txt =
        fs::read_to_string(workspace_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    assert!(
        pdb_txt.contains("HELIX") && pdb_txt.contains("SHEET"),
        "1d3z.pdb must contain secondary-structure records"
    );

    let rep = compute_idp_propensity(&ubq, &PropensityConfig::default());
    assert!(
        !rep.is_predominantly_disordered,
        "folded Ubiquitin must not be classified as disordered"
    );

    // Canonical PDB 1D3Z secondary-structure elements (Seq_ID 1..=76):
    // 5 beta-strands: 2-7, 12-16, 41-45, 48-49, 66-71
    // alpha1 helix: 23-34; 3_10 helix: 56-59
    let beta_ranges = [(2, 7), (12, 16), (41, 45), (48, 49), (66, 71)];
    let helix_ranges = [(23, 34), (56, 59)];

    let mut total_structured = 0usize;
    let mut matched_structured = 0usize;
    let mut mismatches = Vec::new();

    for &(lo, hi) in &beta_ranges {
        for sid in lo..=hi {
            let r = &rep.residues[(sid - 1) as usize];
            total_structured += 1;
            if r.dominant_state.to_three_state() == SecondaryStructureClass::Strand {
                matched_structured += 1;
            } else {
                mismatches.push(format!(
                    "{}{}: expected Beta, got {:?} (p_a={:.2}, p_b={:.2}, p_p={:.2}, p_c={:.2}, ssp={:?})",
                    r.residue, sid, r.dominant_state, r.p_alpha, r.p_beta, r.p_ppii, r.p_coil, r.ssp_score
                ));
            }
        }
    }

    for &(lo, hi) in &helix_ranges {
        for sid in lo..=hi {
            let r = &rep.residues[(sid - 1) as usize];
            total_structured += 1;
            if r.dominant_state.to_three_state() == SecondaryStructureClass::Helix {
                matched_structured += 1;
            } else {
                mismatches.push(format!(
                    "{}{}: expected Helix, got {:?} (p_a={:.2}, p_b={:.2}, p_p={:.2}, p_c={:.2}, ssp={:?})",
                    r.residue, sid, r.dominant_state, r.p_alpha, r.p_beta, r.p_ppii, r.p_coil, r.ssp_score
                ));
            }
        }
    }

    let agreement = (matched_structured as f64) / (total_structured as f64);
    println!(
        "Ubiquitin (bmr17769 vs 1D3Z) structured-element state agreement: {matched_structured}/{total_structured} = {:.2}% (mismatches: {mismatches:?})",
        agreement * 100.0
    );
    assert!(
        agreement >= 0.90,
        "Ubiquitin secondary-structure agreement {:.2}% < 90.0% ({mismatches:?})",
        agreement * 100.0
    );

    // Check Marsh (2006) SSP sign consistency across alpha1 (23..=34) and core strands
    let alpha1_ssp_mean: f64 = (23..=34)
        .filter_map(|sid| rep.residues[sid - 1].ssp_score)
        .sum::<f64>()
        / 12.0;
    let beta1_ssp_mean: f64 = (2..=7)
        .filter_map(|sid| rep.residues[sid - 1].ssp_score)
        .sum::<f64>()
        / 6.0;
    assert!(
        alpha1_ssp_mean >= 0.45,
        "alpha1 mean SSP {alpha1_ssp_mean:.3} should be strongly positive"
    );
    assert!(
        beta1_ssp_mean <= -0.35,
        "beta1 mean SSP {beta1_ssp_mean:.3} should be strongly negative"
    );
}

#[test]
fn test_disordered_alpha_synuclein_bmr6968_and_urea_ubiquitin_bmr4375() {
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let urea = read_star(workspace_root().join("data/benchmark/bmr4375.str"), None)
        .expect("read bmr4375.str");

    let rep_asyn = compute_idp_propensity(&asyn, &PropensityConfig::default());
    let asyn_coil_ppii = rep_asyn.mean_p_coil + rep_asyn.mean_p_ppii;
    println!(
        "alpha-synuclein (bmr6968): mean(p_coil+p_ppii)={:.3} (coil={:.3}, ppii={:.3}, alpha={:.3}, beta={:.3}), mean|SSP|={:.3}",
        asyn_coil_ppii,
        rep_asyn.mean_p_coil,
        rep_asyn.mean_p_ppii,
        rep_asyn.mean_p_alpha,
        rep_asyn.mean_p_beta,
        rep_asyn.mean_abs_ssp
    );
    assert!(
        asyn_coil_ppii >= 0.70,
        "alpha-synuclein mean(p_coil + p_ppii) = {asyn_coil_ppii:.3} < 0.70"
    );
    assert!(
        rep_asyn.is_predominantly_disordered,
        "alpha-synuclein must be flagged predominantly disordered"
    );

    // Proline-rich C-terminal tail (105..=140) should exhibit higher mean p_ppii than N-terminal repeats (1..=60)
    let ppii_nterm: f64 = rep_asyn.residues[0..60]
        .iter()
        .map(|r| r.p_ppii)
        .sum::<f64>()
        / 60.0;
    let ppii_cterm: f64 = rep_asyn.residues[104..140]
        .iter()
        .map(|r| r.p_ppii)
        .sum::<f64>()
        / 36.0;
    assert!(
        ppii_cterm > ppii_nterm,
        "C-terminal Pro-rich tail p_ppii ({ppii_cterm:.3}) should exceed N-terminal ({ppii_nterm:.3})"
    );

    // 8M urea-unfolded Ubiquitin (bmr4375, pH 2): test both unlabelled auto-sensing and explicit pH 2.0
    for (mode, cfg) in [
        ("auto_ph", PropensityConfig::default()),
        (
            "explicit_ph2",
            PropensityConfig::default().with_sample_ph(2.0),
        ),
    ] {
        let rep_urea = compute_idp_propensity(&urea, &cfg);
        let urea_coil_ppii = rep_urea.mean_p_coil + rep_urea.mean_p_ppii;
        println!(
            "8M urea Ubiquitin (bmr4375, {mode}, eff_pH={:.2}): mean(p_coil+p_ppii)={:.3}, mean|SSP|={:.3}",
            rep_urea.effective_ph, urea_coil_ppii, rep_urea.mean_abs_ssp
        );
        assert!(
            rep_urea.effective_ph <= 3.2,
            "{mode}: acidic protonation on bmr4375 should yield eff_pH <= 3.2, got {:.2}",
            rep_urea.effective_ph
        );
        assert!(
            urea_coil_ppii >= 0.70,
            "{mode}: 8M urea Ubiquitin mean(p_coil + p_ppii) = {urea_coil_ppii:.3} < 0.70"
        );
        assert!(
            rep_urea.is_predominantly_disordered,
            "{mode}: 8M urea Ubiquitin must have zero false rigid secondary-structure elements"
        );

        // Explicitly check zero contiguous >=4 rigid helix/strand stretches
        let mut run = 0usize;
        for r in &rep_urea.residues {
            if r.has_data
                && matches!(
                    r.dominant_state,
                    ConformationalState::Alpha | ConformationalState::Beta
                )
                && (r.p_alpha + r.p_beta) >= 0.55
            {
                run += 1;
                assert!(
                    run < 4,
                    "{mode}: false rigid element detected ending at {}{}",
                    r.residue,
                    r.seq_id
                );
            } else {
                run = 0;
            }
        }
    }
}

#[test]
fn test_global_offset_correction_temperature_exports_and_report_svg() {
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let base_rep = compute_idp_propensity(&asyn, &PropensityConfig::default());
    let base_auto = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_auto_offset_correction(true),
    );

    // Apply synthetic TMS-like 13C offset (-2.70 ppm), 15N (+1.45 ppm), and 1H (-0.18 ppm)
    let mut shifted = asyn.clone();
    for ((_, atom), val) in shifted.shifts.iter_mut() {
        match atom.as_str() {
            "CA" | "CB" | "C" => *val -= 2.70,
            "N" => *val += 1.45,
            "H" => *val -= 0.18,
            _ => {}
        }
    }
    let corr_rep = compute_idp_propensity(
        &shifted,
        &PropensityConfig::default().with_auto_offset_correction(true),
    );
    assert!(
        ((corr_rep.applied_offsets.d_ca - base_auto.applied_offsets.d_ca) - (-2.70)).abs() <= 0.30
    );
    assert!(
        ((corr_rep.applied_offsets.d_c - base_auto.applied_offsets.d_c) - (-2.70)).abs() <= 0.35
    );
    assert!(((corr_rep.applied_offsets.d_n - base_auto.applied_offsets.d_n) - 1.45).abs() <= 0.45);
    assert!(
        ((corr_rep.applied_offsets.d_h - base_auto.applied_offsets.d_h) - (-0.18)).abs() <= 0.08
    );

    let max_pop_diff = base_auto
        .residues
        .iter()
        .zip(&corr_rep.residues)
        .map(|(a, b)| (a.p_coil - b.p_coil).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_pop_diff <= 0.08,
        "offset-corrected populations should match DSS baseline, max diff={max_pop_diff:.4}"
    );

    // Temperature adjustment check (278.15 K vs 318.15 K shifts amide H random coil by ~ -0.18 ppm)
    let rep_cold = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_temperature_kelvin(278.15),
    );
    let rep_warm = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_temperature_kelvin(318.15),
    );
    let dh_cold = rep_cold.residues[10].secondary_shifts["H"];
    let dh_warm = rep_warm.residues[10].secondary_shifts["H"];
    assert!(
        dh_warm > dh_cold + 0.12,
        "warming by +40 K lowers RC amide 1H, raising dH ({dh_cold:.3} -> {dh_warm:.3})"
    );

    // TSV & JSON round-trip
    let tsv = format_propensity_tsv(&base_rep);
    assert!(
        tsv.contains("seq_id\tresidue\thas_data\tn_nuclei\tssp\tp_alpha\tp_beta\tp_ppii\tp_coil")
    );
    assert_eq!(tsv.lines().count(), 142);

    let json = format_propensity_json(&base_rep).expect("serialize propensity JSON");
    let decoded: PropensityReport = serde_json::from_str(&json).expect("deserialize JSON");
    assert_eq!(decoded.residues.len(), 140);

    // Standalone SVG + HTML quality report integration (preserving default 5-SVG invariant)
    let svg_standalone = render_idp_propensity_svg(&base_rep);
    assert!(svg_standalone.contains("id=\"svg-idp-propensity\""));
    assert!(svg_standalone.contains("<polyline"));

    let conf: BTreeMap<usize, f64> = (1..asyn.sequence.len()).map(|k| (k, 0.90)).collect();
    let html_default =
        generate_quality_report_html(&ReportInput::new("asyn", asyn.clone(), conf.clone()));
    assert_eq!(
        html_default.matches("<svg ").count(),
        5,
        "default ReportInput::new must preserve exact 5-<svg> count"
    );
    assert!(html_default.contains("id=\"svg-idp-propensity-track\""));

    let html_explicit = generate_quality_report_html(
        &ReportInput::new("asyn", asyn, conf).with_propensity_report(base_rep),
    );
    assert_eq!(html_explicit.matches("<svg ").count(), 6);
    assert!(html_explicit.contains("id=\"section-idp-propensity\""));
    assert!(html_explicit.contains("id=\"svg-idp-propensity\""));
}

#[test]
fn test_estimate_global_rc_offsets_permutation_invariance() {
    use nmr_core::idp_propensity::estimate_global_rc_offsets;

    for star_rel in [
        "data/benchmark/bmr17769.str",
        "data/benchmark/bmr6968.str",
        "data/benchmark/bmr4375.str",
    ] {
        let orig = read_star(workspace_root().join(star_rel), None).expect("read star");
        let cfg = PropensityConfig::default().with_auto_offset_correction(true);
        let off_orig = estimate_global_rc_offsets(&orig, &cfg);

        // Construct deterministic Fisher-Yates permutation pi over 0..n (including termini & Prolines)
        let n = orig.sequence.len();
        let mut pi: Vec<usize> = (0..n).collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((state >> 33) as usize) % (i + 1);
            pi.swap(i, j);
        }

        let mut permuted = orig.clone();
        permuted.shifts.clear();
        for ((k, atom), val) in &orig.shifts {
            permuted.shifts.insert((pi[*k], atom.clone()), *val);
        }

        let off_perm = estimate_global_rc_offsets(&permuted, &cfg);
        assert!(
            (off_orig.d_ca - off_perm.d_ca).abs() < 1e-12
                && (off_orig.d_cb - off_perm.d_cb).abs() < 1e-12
                && (off_orig.d_c - off_perm.d_c).abs() < 1e-12
                && (off_orig.d_ha - off_perm.d_ha).abs() < 1e-12
                && (off_orig.d_h - off_perm.d_h).abs() < 1e-12
                && (off_orig.d_n - off_perm.d_n).abs() < 1e-12,
            "{star_rel}: estimate_global_rc_offsets must be invariant (< 1e-12) under k -> pi(k), got orig={off_orig:?} vs perm={off_perm:?}"
        );
    }
}
