//! Real-data Bruker ubiquitin validation and core solver hygiene unit tests.
//!
//! Provenance: Clean-room verification of `docs/specs/02-peak-picking.md` and
//! `docs/specs/03-assignment.md` §§3–4 on experimental Bruker `pdata` 2D/3D spectra
//! (skipping gracefully in CI when local experimental files are absent) plus synthetic
//! unit tests for `13C`-HSQC aliphatic preservation, `180°` single-phase 3D inversion,
//! paired `NH2` doublet rejection, distinct-experiment support, and automatic `13C`
//! chemical-shift referencing alignment.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::evaluate::{score, to_reference};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_c13_hsqc_2d, pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{estimate_reference_offsets, one_to_three, Priors, ShiftOffsets};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{assign, prepare_scoring_priors, AnnealConfig, ScoreConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, AxisPpm, Spectrum2D, Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn find_local_real_data_dir() -> Option<PathBuf> {
    if let Some(env_dir) = std::env::var_os("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(env_dir);
        return p.exists().then_some(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join("Workspace/nmr_software/data");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn load_fasta_sequence(path: &Path) -> Vec<String> {
    let txt = fs::read_to_string(path).expect("read FASTA");
    let mut seq = Vec::new();
    for line in txt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') {
            continue;
        }
        for ch in trimmed.chars() {
            if ch.is_ascii_alphabetic() {
                seq.push(one_to_three(ch).to_string());
            }
        }
    }
    seq
}

/// Parse whitespace-delimited reference shift table while strictly excluding `-` and any `?`-tagged
/// uncertain values (including fused tokens such as `119.24?170.15` or trailing `?` markers).
fn parse_reference_shift_table(path: &Path, expected_len: usize) -> (Reference, usize) {
    let txt = fs::read_to_string(path).expect("read reference shift table");
    let seq_ids: Vec<i32> = (1..=expected_len as i32).collect();
    let mut sequence = vec!["UNK".to_string(); expected_len];
    let mut shifts: BTreeMap<(usize, String), f64> = BTreeMap::new();
    let mut excluded_uncertain = 0usize;

    for raw_line in txt.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('!') || line.starts_with('#') {
            continue;
        }
        let raw_tokens: Vec<&str> = line.split_whitespace().collect();
        if raw_tokens.len() < 3 {
            continue;
        }
        let Ok(seq_num) = raw_tokens[0].parse::<usize>() else {
            continue;
        };
        if seq_num == 0 || seq_num > expected_len {
            continue;
        }
        let k = seq_num - 1;
        sequence[k] = raw_tokens[1].to_ascii_uppercase();

        // Merge standalone "?" suffix onto the immediately preceding token, and split fused "a?b"
        let mut norm_tokens: Vec<String> = Vec::new();
        for &tok in &raw_tokens[2..] {
            if tok == "?" {
                if let Some(prev) = norm_tokens.last_mut() {
                    prev.push('?');
                }
                continue;
            }
            if let Some(qpos) = tok.find('?') {
                let left = &tok[..=qpos];
                let right = &tok[qpos + 1..];
                norm_tokens.push(left.to_string());
                if !right.is_empty() {
                    norm_tokens.push(right.to_string());
                }
            } else {
                norm_tokens.push(tok.to_string());
            }
        }

        // Fixed positional columns 0..4 correspond to H, N, C, CA; tagged tokens carry `:Cb` etc.
        let pos_atoms = ["H", "N", "C", "CA"];
        let mut pos_idx = 0usize;
        for tok in norm_tokens {
            if let Some((val_part, tag_part)) = tok.split_once(':') {
                let tag_up = tag_part.to_ascii_uppercase();
                if tok.contains('?') {
                    excluded_uncertain += 1;
                    continue;
                }
                if tag_up == "CB" {
                    if let Ok(v) = val_part.parse::<f64>() {
                        shifts.insert((k, "CB".to_string()), v);
                    }
                }
            } else {
                let atom_opt = pos_atoms.get(pos_idx).copied();
                pos_idx += 1;
                if tok == "-" {
                    continue;
                }
                if tok.contains('?') {
                    excluded_uncertain += 1;
                    continue;
                }
                if let (Some(atom), Ok(v)) = (atom_opt, tok.parse::<f64>()) {
                    shifts.insert((k, atom.to_string()), v);
                }
            }
        }
    }

    let mut out = Reference::new(sequence, seq_ids);
    out.shifts = shifts;
    (out, excluded_uncertain)
}

#[test]
fn test_picker_amide_gate_c13_exemption_and_inverted_3d_polarity() {
    // 1. Wide-sweep 2D spectrum spanning [-1.0, 11.0] ppm with an aliphatic peak at 1.25 ppm,
    //    water ridge at 4.75 ppm, and an amide peak at 8.35 ppm (high dynamic range > 40x MAD).
    let nx = 256usize;
    let ny = 128usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 11.0, -1.0, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 132.0, 104.0, 60.81);
    let mut data = vec![0.0f32; nx * ny];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (((i * 73 + 19) % 97) as f32 - 48.0) * 0.01;
    }
    let add_blob_2d = |buf: &mut [f32], h_ppm: f64, y_ppm: f64, amp: f32| {
        let cx = axis_x.ppm_to_index(h_ppm).round() as isize;
        let cy = axis_y.ppm_to_index(y_ppm).round() as isize;
        for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let ix = (cx + dx).clamp(0, nx as isize - 1) as usize;
                let iy = (cy + dy).clamp(0, ny as isize - 1) as usize;
                let w = (-0.6 * ((dx * dx + dy * dy) as f32)).exp();
                buf[iy * nx + ix] += amp * w;
            }
        }
    };
    add_blob_2d(&mut data, 1.25, 118.0, 500.0); // aliphatic
    add_blob_2d(&mut data, 4.75, 115.0, 800.0); // water artifact
    add_blob_2d(&mut data, 8.35, 121.0, 600.0); // true amide
                                                // Add 1.5% F1 sinc sidelobe along the same 1H column at 8.35 ppm
    add_blob_2d(&mut data, 8.35, 125.0, 9.0);

    let spec2d = Spectrum2D {
        name: "HSQC".to_string(),
        axis_x,
        axis_y,
        data,
    };
    let pcfg = PickerConfig::default();

    // 15N-HSQC must gate out 1.25 ppm and 4.75 ppm AND suppress the 1.5% sinc sidelobe
    let hsqc_peaks = pick_hsqc_2d(&spec2d, &pcfg);
    assert_eq!(
        hsqc_peaks.len(),
        1,
        "expected only 1 clean amide peak: {hsqc_peaks:?}"
    );
    assert!((hsqc_peaks[0].h - 8.35).abs() < 0.06);

    // 13C-HSQC must disable amide 1H gating so aliphatic 1.25 ppm and 4.75 ppm peaks ARE picked!
    let c13_peaks = pick_c13_hsqc_2d(&spec2d, &pcfg);
    assert!(
        c13_peaks.iter().any(|p| (p.h - 1.25).abs() < 0.08),
        "pick_c13_hsqc_2d must preserve aliphatic 0..5.5 ppm peaks, got {c13_peaks:?}"
    );

    // 2. 180-degree phase-inverted HNCACO 3D cube + out-of-band carbon artifact at 120 ppm
    let (nx3, ny3, nz3) = (64usize, 32usize, 64usize);
    let ax3_x = AxisPpm::from_bounds("1H", nx3, 9.5, 7.0, 600.13);
    let ax3_y = AxisPpm::from_bounds("15N", ny3, 128.0, 112.0, 60.81);
    let ax3_z = AxisPpm::from_bounds("13C", nz3, 195.0, 115.0, 150.90);
    let mut data3 = vec![0.0f32; nx3 * ny3 * nz3];
    for (i, v) in data3.iter_mut().enumerate() {
        *v = (((i * 37 + 11) % 53) as f32 - 26.0) * 0.02;
    }
    let cx = ax3_x.ppm_to_index(8.35).round() as isize;
    let cy = ax3_y.ppm_to_index(121.0).round() as isize;
    for &(c_ppm, amp) in &[(174.5, -400.0f32), (122.0, -450.0f32)] {
        let cz = ax3_z.ppm_to_index(c_ppm).round() as isize;
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let ix = (cx + dx) as usize;
                    let iy = (cy + dy) as usize;
                    let iz = (cz + dz) as usize;
                    let w = (-0.5 * ((dx * dx + dy * dy + dz * dz) as f32)).exp();
                    data3[(iz * ny3 + iy) * nx3 + ix] += amp * w;
                }
            }
        }
    }
    let spec3d = Spectrum3D {
        name: "HNCACO".to_string(),
        exp: "HNCACO".to_string(),
        axis_x: ax3_x,
        axis_y: ax3_y,
        axis_z: ax3_z,
        data: data3,
    };
    let p3 = pick_triple_3d(&spec3d, &hsqc_peaks, &pcfg);
    assert_eq!(
        p3.len(),
        1,
        "expected inverted 174.5 ppm peak recovered and 122.0 ppm gated: {p3:?}"
    );
    assert_eq!(p3[0].sign, 1);
    assert!((p3[0].c.unwrap() - 174.5).abs() < 1.5);
}

#[test]
fn test_auto_reference_offset_recovery_and_experimental_frame_preservation() {
    let fpath = workspace_root().join("data/fixtures/synthetic_ubiquitin.str");
    let dss_ref = read_star(&fpath, None).unwrap();
    let priors = Priors::bmrb();

    // 1. DSS-referenced synthetic ubiquitin must stay at exact 0.0 ppm within deadband
    let (dss_peaks, _) = simulate(
        &dss_ref,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let dss_sys = build_spin_systems(&dss_peaks, &Tolerances::default(), 3, false).unwrap();
    let dss_off = estimate_reference_offsets(&dss_sys, &dss_ref.sequence, &priors);
    assert_eq!(
        dss_off,
        ShiftOffsets::default(),
        "DSS-referenced spectra must produce exact zero offsets, got {dss_off}"
    );

    // 2. Shift all 13C peaks by -2.75 ppm (TMS referencing) and add NH2 doublet + 2 sinc ripples
    let tms_shift = -2.75;
    let mut tms_peaks: Vec<Peak> = dss_peaks
        .iter()
        .cloned()
        .map(|mut p| {
            if let Some(c) = p.c.as_mut() {
                *c += tms_shift;
            }
            p
        })
        .collect();
    let mut tms_truth_ref = dss_ref.clone();
    for ((_, atom), val) in tms_truth_ref.shifts.iter_mut() {
        if matches!(atom.as_str(), "CA" | "CB" | "C") {
            *val += tms_shift;
        }
    }

    // Add paired NH2 doublet at shared N=112.40 ppm with weak 3D noise attachment on one prong
    tms_peaks.push(Peak::hsqc(6.88, 112.40).with_intensity(110.0, Some(440.0)));
    tms_peaks.push(Peak::hsqc(7.56, 112.43).with_intensity(105.0, Some(420.0)));
    tms_peaks.push(Peak::triple("HNCACB", 6.88, 112.40, 52.0, 1).with_intensity(8.0, Some(30.0)));
    tms_peaks.push(Peak::triple("HNCO", 6.88, 112.40, 174.0, 1).with_intensity(95.0, Some(380.0)));

    // Add two HNCO sinc ripples around the first root to verify distinct-experiment support == 1
    let r0 = tms_peaks.iter().find(|p| p.exp == "HSQC").unwrap().clone();
    tms_peaks.push(Peak::triple("HNCO", r0.h, r0.n, 169.10, 1).with_intensity(6.0, Some(24.0)));
    tms_peaks.push(Peak::triple("HNCO", r0.h, r0.n, 169.22, 1).with_intensity(5.5, Some(22.0)));

    let tms_sys = build_spin_systems(&tms_peaks, &Tolerances::default(), 3, false).unwrap();
    // Both NH2 prongs at the end must have had their candidates cleared
    let n_sys = tms_sys.len();
    assert!(!tms_sys[n_sys - 2].has_observed_carbons());
    assert!(!tms_sys[n_sys - 1].has_observed_carbons());

    let est = estimate_reference_offsets(&tms_sys, &dss_ref.sequence, &priors);
    println!("Estimated TMS offsets: {est}");
    assert!(
        (est.ca - tms_shift).abs() < 0.60
            && (est.cb - tms_shift).abs() < 0.60
            && (est.c - tms_shift).abs() < 0.60,
        "expected ~ -2.75 ppm 13C offsets (within protein secondary-structure bias), got {est}"
    );

    let scfg = ScoreConfig::default();
    let (_, prep_off) = prepare_scoring_priors(&tms_sys, &dss_ref.sequence, &priors, &scfg);
    assert_eq!(prep_off, est);

    let mut truth = vec![-1; n_sys];
    let (_, base_truth) = simulate(
        &dss_ref,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    truth[..base_truth.len()].copy_from_slice(&base_truth);

    let res = assign(
        &tms_sys,
        &dss_ref.sequence,
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
    assert!(
        m.accuracy() >= 0.97 && m.hc_precision() >= 0.98,
        "TMS-shifted ubiquitin accuracy regressed: {}",
        m.report()
    );

    // Output chemical shifts from to_reference MUST stay in the experimental (TMS) ppm frame!
    let (pred_ref, _) = to_reference(
        &res,
        &tms_sys,
        &dss_ref.sequence,
        &dss_ref.seq_ids,
        0.0,
        Some(&priors),
    );
    let mut ca_diffs = Vec::new();
    for (&(k, ref atom), &pred_v) in &pred_ref.shifts {
        if atom == "CA" {
            if let Some(&exp_true_v) = tms_truth_ref.shifts.get(&(k, "CA".to_string())) {
                ca_diffs.push((pred_v - exp_true_v).abs());
            }
        }
    }
    ca_diffs.sort_by(|a, b| a.total_cmp(b));
    let med_ca_err = ca_diffs[ca_diffs.len() / 2];
    assert!(
        med_ca_err < 0.10,
        "to_reference shifts must remain in experimental ppm frame (median |dCA|={med_ca_err:.3})"
    );
}

#[test]
fn test_real_bruker_ubiquitin_5_seed_end_to_end() {
    let Some(data_dir) = find_local_real_data_dir() else {
        eprintln!("Skipping test_real_bruker_ubiquitin_5_seed_end_to_end: local dataset absent");
        return;
    };
    let ubq_dir = data_dir.join("ubiquitin");
    let truth_path = data_dir.join("ubiquitin_backbone_assignment_191104.txt");
    let fasta_path = ubq_dir.join("sparky_files/ubiquitin_data/1ubq.fasta.txt");
    let hsqc_dir = ubq_dir.join("Ubiquitin_090309/2/pdata/1");
    let hncacb_dir = ubq_dir.join("Ubiquitin_090309/3/pdata/1");
    let cbcaconh_dir = ubq_dir.join("Ubiquitin_090309/4/pdata/1");
    let hnco_dir = ubq_dir.join("Ubiquitin_111909/10/pdata/1");
    let hncaco_dir = ubq_dir.join("Ubiquitin_111909/11/pdata/1");

    for req in [
        &truth_path,
        &fasta_path,
        &hsqc_dir,
        &hncacb_dir,
        &cbcaconh_dir,
        &hnco_dir,
        &hncaco_dir,
    ] {
        if !req.exists() {
            eprintln!(
                "Skipping test_real_bruker_ubiquitin_5_seed_end_to_end: missing {:?}",
                req
            );
            return;
        }
    }

    let sequence = load_fasta_sequence(&fasta_path);
    assert_eq!(sequence.len(), 76);
    let seq_ids: Vec<i32> = (1..=76).collect();
    let (truth_ref, excluded_q) = parse_reference_shift_table(&truth_path, 76);
    assert_eq!(
        excluded_q, 2,
        "expected 2 uncertain '?' tokens excluded from truth"
    );

    // 1. Classical peak picking across all 5 processed Bruker spectra
    let pcfg = PickerConfig::default();
    let hsqc_spec = read_bruker_pdata_2d(&hsqc_dir).expect("read Bruker HSQC");
    let roots = pick_hsqc_2d(&hsqc_spec, &pcfg);
    let mut all_peaks = roots.clone();
    for (exp_name, dir) in [
        ("HNCACB", &hncacb_dir),
        ("CBCACONH", &cbcaconh_dir),
        ("HNCO", &hnco_dir),
        ("HNCACO", &hncaco_dir),
    ] {
        let s3 = read_bruker_pdata_3d(dir, exp_name).expect("read Bruker 3D");
        let p3 = pick_triple_3d(&s3, &roots, &pcfg);
        println!("[Bruker picker] {:<9}: {:>5} peaks", exp_name, p3.len());
        assert!(
            !p3.is_empty(),
            "{exp_name} produced 0 peaks (check phase/bounds)"
        );
        all_peaks.extend(p3);
    }
    println!(
        "[Bruker picker] HSQC roots: {} | total peaks: {}",
        roots.len(),
        all_peaks.len()
    );
    assert!(
        roots.len() <= 180,
        "expected clean HSQC root count <= 180, got {}",
        roots.len()
    );

    // 2. Build spin systems and check true root matching against reference (H, N)
    let tol = Tolerances::default();
    let systems = build_spin_systems(&all_peaks, &tol, 3, false).expect("build spin systems");

    // Map each reference backbone (H, N) residue (k >= 1, non-Pro) to its closest picked root
    let mut truth_for_sys = vec![-1i32; systems.len()];
    let mut matched_true_roots = 0usize;
    let mut total_true_amides = 0usize;
    for (k, aa) in sequence.iter().enumerate().skip(1) {
        if aa == "PRO" {
            continue;
        }
        let (Some(&th), Some(&tn)) = (
            truth_ref.shifts.get(&(k, "H".to_string())),
            truth_ref.shifts.get(&(k, "N".to_string())),
        ) else {
            continue;
        };
        total_true_amides += 1;
        // Account for systematic H/N offset (~0.02 ppm H, ~0.2 ppm N) via normalized tolerance
        let mut best_s: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        for (s, ss) in systems.iter().enumerate() {
            let d = ((ss.h - th) / 0.08).hypot((ss.n - tn) / 0.65);
            if d <= 1.0 && d < best_d && truth_for_sys[s] < 0 {
                best_d = d;
                best_s = Some(s);
            }
        }
        if let Some(s) = best_s {
            truth_for_sys[s] = k as i32;
            matched_true_roots += 1;
        }
    }
    println!(
        "[Bruker roots] matched true amides: {}/{}",
        matched_true_roots, total_true_amides
    );
    assert_eq!(total_true_amides, 70);
    assert!(
        matched_true_roots >= 67,
        "expected >= 67/70 true ubiquitin amides picked, got {matched_true_roots}/70"
    );

    // 3. Verify auto reference offset estimation (~ -2.75 ppm 13C offset)
    let priors = Priors::bmrb();
    let offsets = estimate_reference_offsets(&systems, &sequence, &priors);
    println!("[Bruker auto-reference] {offsets}");
    assert!(
        offsets.ca < -2.0 && offsets.cb < -2.0 && offsets.c < -2.0,
        "expected ~ -2.75 ppm 13C offsets on TMS-referenced Bruker ubiquitin, got {offsets}"
    );

    // 4. Run 5-seed assignment and evaluate spin-system accuracy, recall, HC precision, & shifts
    let scfg = ScoreConfig::default();
    let (eff_priors, _) = prepare_scoring_priors(&systems, &sequence, &priors, &scfg);
    let mut acc_sum = 0.0;
    let mut rec_sum = 0.0;
    let mut hcp_sum = 0.0;
    let mut hcr_sum = 0.0;
    let mut broadened_hijacks = 0usize;
    let mut shift_agree_sum = [0.0f64; 5]; // H, N, CA, CB, C

    for seed in 0..5u64 {
        let res = assign(
            &systems,
            &sequence,
            &priors,
            &scfg,
            &AnnealConfig {
                runs: 64,
                steps_per_system: 4000,
                seed,
                ..AnnealConfig::default()
            },
        );

        // Count correct placements among assignable truth-matched roots + false placements
        let mut assigned_eval = 0usize;
        let mut correct_eval = 0usize;
        let mut hc_assigned_eval = 0usize;
        let mut hc_correct_eval = 0usize;

        for (s, &k_pred) in res.position.iter().enumerate() {
            if k_pred < 0 {
                continue;
            }
            let conf = res.confidence[s];
            // Exchange-broadened Glu24 (k=23) and Gly53 (k=52) have no true amide root
            if k_pred == 23 || k_pred == 52 {
                broadened_hijacks += 1;
            }
            assigned_eval += 1;
            let is_ok = truth_for_sys[s] == k_pred;
            if is_ok {
                correct_eval += 1;
            }
            if conf >= 0.8 {
                hc_assigned_eval += 1;
                if is_ok {
                    hc_correct_eval += 1;
                }
            }
        }
        let acc = (correct_eval as f64) / (assigned_eval.max(1) as f64);
        let rec = (correct_eval as f64) / (total_true_amides as f64);
        let hcp = if hc_assigned_eval > 0 {
            (hc_correct_eval as f64) / (hc_assigned_eval as f64)
        } else {
            1.0
        };
        let hcr = (hc_correct_eval as f64) / (total_true_amides as f64);
        acc_sum += acc;
        rec_sum += rec;
        hcp_sum += hcp;
        hcr_sum += hcr;

        let (pred_ref, _) =
            to_reference(&res, &systems, &sequence, &seq_ids, 0.0, Some(&eff_priors));
        let atom_tols = [
            ("H", 0.08),
            ("N", 0.65),
            ("CA", 0.40),
            ("CB", 0.50),
            ("C", 0.40),
        ];
        for (ai, &(atom, atol)) in atom_tols.iter().enumerate() {
            let mut n_cmp = 0usize;
            let mut n_hit = 0usize;
            for (&(k, ref a), &tv) in &truth_ref.shifts {
                if a != atom {
                    continue;
                }
                if let Some(&pv) = pred_ref.shifts.get(&(k, atom.to_string())) {
                    n_cmp += 1;
                    if (pv - tv).abs() <= atol {
                        n_hit += 1;
                    }
                }
            }
            shift_agree_sum[ai] += (n_hit as f64) / (n_cmp.max(1) as f64);
        }

        println!(
            "[seed {seed}] assigned={assigned_eval} correct={correct_eval}/70 | acc={:.1}% rec={:.1}% hc_prec={:.1}% ({hc_correct_eval}/{hc_assigned_eval}) hc_rec={:.1}%",
            acc * 100.0,
            rec * 100.0,
            hcp * 100.0,
            hcr * 100.0
        );
    }

    let mean_acc = acc_sum / 5.0;
    let mean_rec = rec_sum / 5.0;
    let mean_hcp = hcp_sum / 5.0;
    let mean_hcr = hcr_sum / 5.0;
    println!(
        "[5-seed Bruker Ubiquitin MEAN] acc={:.1}% rec={:.1}% hc_prec={:.1}% hc_rec={:.1}% broadened_hijacks={} | shift_agree: H={:.1}% N={:.1}% CA={:.1}% CB={:.1}% C={:.1}%",
        mean_acc * 100.0,
        mean_rec * 100.0,
        mean_hcp * 100.0,
        mean_hcr * 100.0,
        broadened_hijacks,
        shift_agree_sum[0] * 20.0,
        shift_agree_sum[1] * 20.0,
        shift_agree_sum[2] * 20.0,
        shift_agree_sum[3] * 20.0,
        shift_agree_sum[4] * 20.0,
    );

    assert!(
        mean_acc >= 0.80 && mean_rec >= 0.80 && mean_hcp >= 0.90,
        "real Bruker ubiquitin 5-seed metrics below target: acc={mean_acc:.3}, rec={mean_rec:.3}, hcp={mean_hcp:.3}"
    );
}
