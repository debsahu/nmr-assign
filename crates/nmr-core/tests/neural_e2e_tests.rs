//! End-to-end tests, finite-difference gradient checks, weight training & serialization,
//! and blind `Spectra -> Picked Peaks -> Assignment` benchmark for `neural_picker`.

pub use nmr_core::*;

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nmr_core::neural_picker::{
    evaluate_picked_peaks, generate_synthetic_spectrum_2d, generate_synthetic_spectrum_3d,
    normalize_spectrum_data, pick_hsqc_2d_neural, pick_triple_3d_neural, run_e2e_blind_benchmark,
    train_weights_2d, train_weights_3d, E2eBenchmarkConfig, NeuralPicker, NeuralPickerConfig,
    SeparableConvNet2D, SeparableConvNet3D, SyntheticGenConfig2D, SyntheticGenConfig3D,
    TrainerConfig, CONV2D_NUM_WEIGHTS, CONV3D_NUM_WEIGHTS,
};
use nmr_core::nmrstar::read_star;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};

fn fixture_star_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

#[test]
fn test_analytical_vs_numerical_gradients_2d_and_3d() {
    // 1. 2D finite-difference gradient check across every layer group
    let mut net2 = SeparableConvNet2D::init_seeded(77);
    let syn2 = generate_synthetic_spectrum_2d(&SyntheticGenConfig2D {
        nx: 12,
        ny: 12,
        n_peaks: 3,
        seed: 19,
        ..SyntheticGenConfig2D::default()
    });
    let (in2, _, _) = normalize_spectrum_data(&syn2.spectrum.data);
    let loss_fn_2d = |net: &SeparableConvNet2D| -> (f64, Vec<f32>) {
        net.forward_backward_raw_2d(&in2, 12, 12, |p, raw_h, g_raw| {
            let mut l = 0.0f64;
            for k in 0..5 {
                let target = (((p * 7 + k * 3) % 11) as f32 - 5.0) * 0.15;
                let diff = raw_h[k] - target;
                l += 0.5 * (diff as f64) * (diff as f64) / 144.0;
                g_raw[k] = diff / 144.0;
            }
            l
        })
    };
    let (_, grad2) = loss_fn_2d(&net2);
    let eps = 1e-3f32;
    for idx in (0..CONV2D_NUM_WEIGHTS).step_by(29) {
        let orig = net2.weights[idx];
        net2.weights[idx] = orig + eps;
        let (lp, _) = loss_fn_2d(&net2);
        net2.weights[idx] = orig - eps;
        let (lm, _) = loss_fn_2d(&net2);
        net2.weights[idx] = orig;
        let num_g = ((lp - lm) / (2.0 * eps as f64)) as f32;
        let err = (grad2[idx] - num_g).abs();
        assert!(
            err < 1.5e-3,
            "2D grad mismatch at weight {idx}: analytical={}, numerical={num_g}, err={err}",
            grad2[idx]
        );
    }

    // 2. 3D finite-difference gradient check across every layer group
    let mut net3 = SeparableConvNet3D::init_seeded(88);
    let syn3 = generate_synthetic_spectrum_3d(&SyntheticGenConfig3D {
        nx: 8,
        ny: 8,
        nz: 8,
        n_peaks: 2,
        seed: 23,
        ..SyntheticGenConfig3D::default()
    });
    let (in3, _, _) = normalize_spectrum_data(&syn3.spectrum.data);
    let loss_fn_3d = |net: &SeparableConvNet3D| -> (f64, Vec<f32>) {
        net.forward_backward_raw_3d(&in3, 8, 8, 8, |v, raw_h, g_raw| {
            let mut l = 0.0f64;
            for k in 0..6 {
                let target = (((v * 5 + k * 13) % 9) as f32 - 4.0) * 0.12;
                let diff = raw_h[k] - target;
                l += 0.5 * (diff as f64) * (diff as f64) / 512.0;
                g_raw[k] = diff / 512.0;
            }
            l
        })
    };
    let (_, grad3) = loss_fn_3d(&net3);
    for idx in (0..CONV3D_NUM_WEIGHTS).step_by(31) {
        let orig = net3.weights[idx];
        net3.weights[idx] = orig + eps;
        let (lp, _) = loss_fn_3d(&net3);
        net3.weights[idx] = orig - eps;
        let (lm, _) = loss_fn_3d(&net3);
        net3.weights[idx] = orig;
        let num_g = ((lp - lm) / (2.0 * eps as f64)) as f32;
        let err = (grad3[idx] - num_g).abs();
        assert!(
            err < 1.5e-3,
            "3D grad mismatch at weight {idx}: analytical={}, numerical={num_g}, err={err}",
            grad3[idx]
        );
    }
}

#[test]
fn test_trainer_convergence_serialization_and_e2e_benchmark() {
    let t_train = Instant::now();
    let (net2, loss2_start, loss2_end) = train_weights_2d(&TrainerConfig::default());
    let (net3, loss3_start, loss3_end) = train_weights_3d(&TrainerConfig::default());
    println!(
        "Trained 2D ({loss2_start:.4} -> {loss2_end:.4}) & 3D ({loss3_start:.4} -> {loss3_end:.4}) in {:.2}s",
        t_train.elapsed().as_secs_f64()
    );
    assert!(
        loss2_end < loss2_start * 0.65,
        "2D trainer did not converge"
    );
    assert!(
        loss3_end < loss3_start * 0.65,
        "3D trainer did not converge"
    );

    // Dump converged constant arrays to scratch if requested via env var
    if let Ok(dump_path) = std::env::var("NMR_DUMP_WEIGHTS") {
        let mut s = String::new();
        s.push_str("pub const PRETRAINED_WEIGHTS_2D: [f32; CONV2D_NUM_WEIGHTS] = [\n");
        for chunk in net2.weights.chunks(8) {
            s.push_str("    ");
            for &v in chunk {
                s.push_str(&format!("{v:.7}_f32, "));
            }
            s.push('\n');
        }
        s.push_str("];\n\npub const PRETRAINED_WEIGHTS_3D: [f32; CONV3D_NUM_WEIGHTS] = [\n");
        for chunk in net3.weights.chunks(8) {
            s.push_str("    ");
            for &v in chunk {
                s.push_str(&format!("{v:.7}_f32, "));
            }
            s.push('\n');
        }
        s.push_str("];\n");
        fs::write(&dump_path, s).expect("write dumped weights");
    }

    // Verify JSON & compact binary round-trip
    let j2 = net2.to_json().unwrap();
    let back_j2 = SeparableConvNet2D::from_json(&j2).unwrap();
    assert_eq!(net2.weights.len(), back_j2.weights.len());
    let b3 = net3.to_bytes();
    let back_b3 = SeparableConvNet3D::from_bytes(&b3).unwrap();
    assert_eq!(net3.weights, back_b3.weights);

    // Evaluate synthetic noisy/ridge/shoulder 2D & 3D patch F1 vs classical
    let picker = NeuralPicker {
        net2d: net2,
        net3d: net3,
    };

    let (mut cls2_tp, mut cls2_p, mut cls2_t) = (0usize, 0usize, 0usize);
    let (mut neu2_tp, mut neu2_p, mut neu2_t) = (0usize, 0usize, 0usize);
    for i in 0..16u64 {
        let syn2 = generate_synthetic_spectrum_2d(&SyntheticGenConfig2D {
            nx: 64,
            ny: 64,
            n_peaks: 14,
            allow_negative: true,
            min_snr: 3.8,
            max_snr: 24.0,
            ridge_prob: 0.85,
            ridge_rel_amp: 0.18,
            baseline_roll_sigma: 2.5,
            shoulder_prob: 0.40,
            seed: 8000 + i,
            ..SyntheticGenConfig2D::default()
        });
        let c_pks = pick_hsqc_2d(
            &syn2.spectrum,
            &PickerConfig {
                snr_threshold: 5.0,
                allow_negative: true,
                ..PickerConfig::default()
            },
        );
        let n_pks = picker.pick_2d(
            &syn2.spectrum,
            &NeuralPickerConfig {
                allow_negative: true,
                ..NeuralPickerConfig::default()
            },
        );
        let mc = evaluate_picked_peaks("HSQC", &c_pks, &syn2.peaks);
        let mn = evaluate_picked_peaks("HSQC", &n_pks, &syn2.peaks);
        cls2_tp += mc.matched_tp;
        cls2_p += mc.picked_peaks;
        cls2_t += mc.true_peaks;
        neu2_tp += mn.matched_tp;
        neu2_p += mn.picked_peaks;
        neu2_t += mn.true_peaks;
    }
    let cls2_f1 = 2.0 * (cls2_tp as f64) / ((cls2_p + cls2_t) as f64);
    let neu2_f1 = 2.0 * (neu2_tp as f64) / ((neu2_p + neu2_t) as f64);
    println!("2D Patch Validation F1: Classical={cls2_f1:.3} vs Neural={neu2_f1:.3}");
    assert!(
        neu2_f1 > cls2_f1 + 0.05,
        "Expected 2D neural picker ({neu2_f1:.3}) to beat classical ({cls2_f1:.3})"
    );

    let mut cls_tp = 0usize;
    let mut cls_p = 0usize;
    let mut cls_t = 0usize;
    let mut neu_tp = 0usize;
    let mut neu_p = 0usize;
    let mut neu_t = 0usize;

    for i in 0..12u64 {
        let syn3 = generate_synthetic_spectrum_3d(&SyntheticGenConfig3D {
            nx: 32,
            ny: 32,
            nz: 32,
            exp: "HNCACB".to_string(),
            n_peaks: 12,
            allow_negative: true,
            min_snr: 4.0,
            max_snr: 22.0,
            ridge_prob: 0.85,
            ridge_rel_amp: 0.16,
            baseline_roll_sigma: 2.3,
            shoulder_prob: 0.40,
            seed: 9000 + i,
            ..SyntheticGenConfig3D::default()
        });
        let c_pks = pick_triple_3d(
            &syn3.spectrum,
            &[],
            &PickerConfig {
                snr_threshold: 5.0,
                allow_negative: true,
                ..PickerConfig::default()
            },
        );
        let n_pks = picker.pick_3d(
            &syn3.spectrum,
            "HNCACB",
            &NeuralPickerConfig {
                allow_negative: true,
                ..NeuralPickerConfig::default()
            },
        );
        let mc = evaluate_picked_peaks("HNCACB", &c_pks, &syn3.peaks);
        let mn = evaluate_picked_peaks("HNCACB", &n_pks, &syn3.peaks);
        cls_tp += mc.matched_tp;
        cls_p += mc.picked_peaks;
        cls_t += mc.true_peaks;
        neu_tp += mn.matched_tp;
        neu_p += mn.picked_peaks;
        neu_t += mn.true_peaks;
    }
    let cls_f1 = 2.0 * (cls_tp as f64) / ((cls_p + cls_t) as f64);
    let neu_f1 = 2.0 * (neu_tp as f64) / ((neu_p + neu_t) as f64);
    println!("3D Patch Validation F1: Classical={cls_f1:.3} vs Neural={neu_f1:.3}");
    assert!(
        neu_f1 > cls_f1 + 0.05,
        "Expected neural picker ({neu_f1:.3}) to beat classical ({cls_f1:.3}) on noisy/ridge/shoulder 3D cubes"
    );

    // Verify public entry points & end-to-end ubiquitin benchmark
    let ref_data = read_star(fixture_star_path(), None).expect("read ubiquitin fixture");
    let cmp = run_e2e_blind_benchmark(&ref_data, &E2eBenchmarkConfig::default())
        .expect("run blind e2e benchmark");
    println!("\n{}", cmp.format_table());
    for m in &cmp.classical.per_experiment {
        let nm = cmp
            .neural
            .per_experiment
            .iter()
            .find(|x| x.experiment == m.experiment)
            .unwrap();
        println!(
            "  {:10} | Classical F1={:.3} (Prec={:.1}%, Rec={:.1}%, FDR={:.1}%) | Neural F1={:.3} (Prec={:.1}%, Rec={:.1}%, FDR={:.1}%)",
            m.experiment,
            m.f1,
            m.precision * 100.0,
            m.recall * 100.0,
            m.fdr * 100.0,
            nm.f1,
            nm.precision * 100.0,
            nm.recall * 100.0,
            nm.fdr * 100.0
        );
    }
    assert!(
        cmp.neural.overall_peaks.f1 > cmp.classical.overall_peaks.f1,
        "Expected neural overall peak F1 ({:.3}) > classical ({:.3})",
        cmp.neural.overall_peaks.f1,
        cmp.classical.overall_peaks.f1
    );
    assert!(
        cmp.neural.assignment_recall >= cmp.classical.assignment_recall,
        "Expected neural downstream assignment recall ({:.3}) >= classical ({:.3})",
        cmp.neural.assignment_recall,
        cmp.classical.assignment_recall
    );

    // Sanity check public standalone functions
    let tiny2 = generate_synthetic_spectrum_2d(&SyntheticGenConfig2D {
        nx: 64,
        ny: 64,
        n_peaks: 6,
        allow_negative: false,
        seed: 777,
        ..SyntheticGenConfig2D::default()
    });
    let t_fast = Instant::now();
    let p2 = pick_hsqc_2d_neural(&tiny2.spectrum, &NeuralPickerConfig::default());
    let dt_ms = t_fast.elapsed().as_secs_f64() * 1000.0;
    println!(
        "64x64 2D neural inference + NMS latency: {dt_ms:.3} ms ({} peaks)",
        p2.len()
    );
    let _ = pick_triple_3d_neural(
        &generate_synthetic_spectrum_3d(&SyntheticGenConfig3D::default()).spectrum,
        "HNCACB",
        &NeuralPickerConfig::default(),
    );
    let _ = pick_hsqc_2d(&tiny2.spectrum, &PickerConfig::default());
}
