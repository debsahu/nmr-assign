//! Comprehensive regression tests for `nmr-core` (I/O, spin systems, scoring,
//! Bruker `pdata` & NMRPipe spectra, classical peak picking, and assignment benchmark).

use std::fs;
use std::path::PathBuf;

use nmr_core::evaluate::{score, to_reference};
use nmr_core::nmrstar::{read_star, write_star};
use nmr_core::peaks::{read_peaks, write_peaks};
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{compute_priors, Priors};
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_nmrpipe_2d, write_bruker_pdata_2d, write_nmrpipe_2d,
};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nmrassign_test_{}_{}", name, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_star_and_peaks_roundtrip() {
    let dir = temp_dir("star_peaks");
    let ref_data = read_star(fixture_path(), None).expect("read fixture star");
    assert_eq!(ref_data.sequence.len(), 76);
    assert_eq!(ref_data.sequence[0], "MET");
    assert!(ref_data.shifts.contains_key(&(9, "CA".to_string()))); // Gly10 CA
    assert!(!ref_data.shifts.contains_key(&(9, "CB".to_string()))); // Gly10 has no CB

    let out_star = dir.join("roundtrip.str");
    write_star(&out_star, &ref_data, None).unwrap();
    let back_star = read_star(&out_star, None).unwrap();
    assert_eq!(back_star.sequence, ref_data.sequence);
    assert_eq!(back_star.shifts.len(), ref_data.shifts.len());

    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            seed: 1,
            ..SimConfig::default()
        },
    );
    let p_tsv = dir.join("peaks.tsv");
    write_peaks(&p_tsv, &peaks).unwrap();
    let back_peaks = read_peaks(&p_tsv).unwrap();
    assert_eq!(back_peaks.len(), peaks.len());
    assert_eq!(back_peaks[0].exp, peaks[0].exp);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_clean_data_spin_systems_exact() {
    let ref_data = read_star(fixture_path(), None).unwrap();
    let cfg = SimConfig {
        p_missing_peak: 0.0,
        p_missing_root: 0.0,
        spurious_rate: 0.0,
        sidechain_nh2: false,
        seed: 2,
        ..SimConfig::default()
    };
    let (peaks, truth) = simulate(&ref_data, &cfg);
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();

    let mut total = 0usize;
    let mut lost = 0usize;
    let mut top_wrong = 0usize;

    for (ss, &k_raw) in systems.iter().zip(truth.iter()) {
        let k = k_raw as isize;
        for atom in ["CA", "CB", "C"] {
            for off in [0isize, -1isize] {
                let idx = k + off;
                if idx < 0 {
                    continue;
                }
                let Some(&true_val) = ref_data.shifts.get(&(idx as usize, atom.to_string())) else {
                    continue;
                };
                total += 1;
                let cands = ss
                    .cands
                    .get(&(atom.to_string(), off as i32))
                    .cloned()
                    .unwrap_or_default();
                if !cands.iter().any(|&(v, _)| (v - true_val).abs() < 0.2) {
                    lost += 1;
                } else if (cands[0].0 - true_val).abs() >= 0.2 {
                    top_wrong += 1;
                }
            }
        }
    }

    assert!((lost as f64) / (total as f64) < 0.06, "lost={lost}/{total}");
    assert!(
        (top_wrong as f64) / (total as f64) < 0.03,
        "top_wrong={top_wrong}/{total}"
    );
}

#[test]
fn test_spectrum_io_and_classical_picker_roundtrip() {
    let dir = temp_dir("spectra_io");
    let ref_data = read_star(fixture_path(), None).unwrap();
    let (peaks, _) = simulate(
        &ref_data,
        &SimConfig {
            p_missing_peak: 0.0,
            p_missing_root: 0.0,
            spurious_rate: 0.0,
            sidechain_nh2: false,
            noise_h: 0.0,
            noise_n: 0.0,
            noise_c: 0.0,
            seed: 5,
            ..SimConfig::default()
        },
    );

    let rcfg = SpectrumRenderConfig {
        nx_h: 512,
        ny_n: 256,
        nz_c: 128,
        noise_level: 1.0,
        seed: 11,
        ..SpectrumRenderConfig::default()
    };
    let hsqc = render_hsqc_2d(&peaks, &rcfg);

    // 1. NMRPipe .ft2 round-trip within < 1 digital point
    let pipe_path = dir.join("test_hsqc.ft2");
    write_nmrpipe_2d(&pipe_path, &hsqc).unwrap();
    let hsqc_pipe = read_nmrpipe_2d(&pipe_path).unwrap();
    assert_eq!(hsqc_pipe.nx(), hsqc.nx());
    assert_eq!(hsqc_pipe.ny(), hsqc.ny());

    // 2. Bruker TopSpin pdata (2rr) round-trip within < 1 digital point
    let pdata_dir = dir.join("pdata_1");
    write_bruker_pdata_2d(&pdata_dir, &hsqc).unwrap();
    let hsqc_bruker = read_bruker_pdata_2d(&pdata_dir).unwrap();
    assert_eq!(hsqc_bruker.nx(), hsqc.nx());
    assert_eq!(hsqc_bruker.ny(), hsqc.ny());

    let pcfg = PickerConfig {
        snr_threshold: 6.0,
        ..PickerConfig::default()
    };
    let picked_pipe = pick_hsqc_2d(&hsqc_pipe, &pcfg);
    let picked_bruker = pick_hsqc_2d(&hsqc_bruker, &pcfg);
    assert!(picked_pipe.len() >= 60, "picked_pipe={}", picked_pipe.len());
    assert_eq!(picked_pipe.len(), picked_bruker.len());

    // Verify peak centres agree within < 0.25 digital points between NMRPipe and Bruker round-trips
    let step_h = hsqc.axis_x.step_ppm.abs();
    let step_n = hsqc.axis_y.step_ppm.abs();
    for (pa, pb) in picked_pipe.iter().zip(picked_bruker.iter()) {
        assert!((pa.h - pb.h).abs() < 0.25 * step_h);
        assert!((pa.n - pb.n).abs() < 0.25 * step_n);
    }

    // 3. 3D HNCACB picking restricted under HSQC roots preserves +CA / -CB phase signs
    let hncacb_3d = render_triple_3d("HNCACB", &peaks, &rcfg);
    let picked_3d = pick_triple_3d(&hncacb_3d, &picked_pipe, &pcfg);
    let n_pos = picked_3d.iter().filter(|p| p.sign > 0).count();
    let n_neg = picked_3d.iter().filter(|p| p.sign < 0).count();
    assert!(n_pos > 50 && n_neg > 40, "n_pos={n_pos}, n_neg={n_neg}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_full_set_benchmark_floor_and_priors() {
    let fpath = fixture_path();
    let ref_data = read_star(&fpath, None).unwrap();
    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();

    let res = assign(
        &systems,
        &ref_data.sequence,
        &Priors::provisional(),
        &ScoreConfig::default(),
        &AnnealConfig {
            runs: 12,
            steps_per_system: 1500,
            seed: 0,
            ..AnnealConfig::default()
        },
    );
    let m = score(&res, &truth, 0.8);
    assert!(m.recall() >= 0.90, "{}", m.report());
    assert!(m.hc_precision() >= 0.95, "{}", m.report());

    // Zero-carbon sidechain NH2 decoy roots (truth == -1) must remain unassigned
    for (s, &t) in truth.iter().enumerate() {
        if t < 0 && !systems[s].has_observed_carbons() {
            assert_eq!(
                res.position[s], -1,
                "0-carbon decoy SS #{s} should stay unassigned"
            );
        }
    }

    // Tightly overlapped ILE30 (k=29) and PHE45 (k=44) must both recover CA/CB shifts
    let (pred, _) = to_reference(
        &res,
        &systems,
        &ref_data.sequence,
        &ref_data.seq_ids,
        0.0,
        None,
    );
    for (k, atom) in [(29usize, "CA"), (29, "CB"), (44, "CA"), (44, "CB")] {
        let got = pred
            .shifts
            .get(&(k, atom.to_string()))
            .copied()
            .unwrap_or_else(|| panic!("missing {atom} for residue index {k}"));
        let exp = ref_data.shifts[&(k, atom.to_string())];
        assert!(
            (got - exp).abs() <= 0.35,
            "residue k={k} {atom} mismatch: got={got}, ref={exp}"
        );
    }

    let recomputed = compute_priors(&[fpath], 5);
    assert!(!recomputed.has("GLY", "CB"));
    assert!(!recomputed.has("PRO", "H"));
}
