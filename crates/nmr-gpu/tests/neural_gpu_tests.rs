//! WebGPU (`wgpu` + WGSL) 3D convolution & NMS numerical parity tests against `nmr-core` CPU `neural_picker`.

pub use nmr_core::*;
pub use nmr_gpu::*;

use nmr_core::neural_picker::{
    generate_synthetic_spectrum_3d, normalize_spectrum_data_with_nx, NeuralPicker,
    NeuralPickerConfig, SyntheticGenConfig3D, CONV3D_OUT_CH, PRETRAINED_WEIGHTS_3D,
};
use nmr_gpu::neural_gpu::{
    normalize_spectrum_3d_for_gpu, GpuNeuralPicker, GpuNeuralPickerConfig,
    PRETRAINED_WEIGHTS_3D_GPU,
};

#[test]
fn test_conv3d_cpu_vs_webgpu_forward_and_nms_parity() {
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping WebGPU test (no GPU adapter available): {err}");
            return;
        }
    };
    println!(
        "Testing WebGPU 3D Neural Picker on {} ({})",
        engine.adapter_name, engine.backend_name
    );

    assert_eq!(
        PRETRAINED_WEIGHTS_3D.as_slice(),
        PRETRAINED_WEIGHTS_3D_GPU.as_slice()
    );

    let cpu_picker = NeuralPicker::pretrained();
    let gpu_picker = GpuNeuralPicker::pretrained(&engine).expect("init GpuNeuralPicker");

    // 1. Generate realistic noisy 3D HNCACB cube (+CA / -CB, baseline roll, t1 ridges, shoulders)
    let syn = generate_synthetic_spectrum_3d(&SyntheticGenConfig3D {
        nx: 28,
        ny: 24,
        nz: 32,
        exp: "HNCACB".to_string(),
        n_peaks: 12,
        allow_negative: true,
        min_snr: 4.0,
        max_snr: 25.0,
        ridge_prob: 1.0,
        ridge_rel_amp: 0.18,
        baseline_roll_sigma: 2.3,
        shoulder_prob: 0.45,
        j_split_prob: 0.25,
        seed: 20261111,
        ..SyntheticGenConfig3D::default()
    });
    let (nx, ny, nz) = (syn.spectrum.nx(), syn.spectrum.ny(), syn.spectrum.nz());

    let (norm_cpu, med_cpu, noise_cpu) = normalize_spectrum_data_with_nx(&syn.spectrum.data, nx);
    let (norm_gpu, med_gpu, noise_gpu) = normalize_spectrum_3d_for_gpu(&syn.spectrum.data, nx);

    assert!((med_cpu - med_gpu).abs() < 1e-6);
    assert!((noise_cpu - noise_gpu).abs() < 1e-6);
    assert_eq!(norm_cpu.len(), norm_gpu.len());

    // 2. Compare dense 3D separable CNN forward pass across all 6 channels on all 21,504 voxels
    let out_cpu = cpu_picker.net3d.forward_3d(&norm_cpu, nx, ny, nz);
    let out_gpu = gpu_picker
        .forward_3d_gpu(&norm_gpu, nx, ny, nz)
        .expect("GPU forward_3d_gpu");

    assert_eq!(out_cpu.len(), out_gpu.len());
    let mut max_abs_diff = 0.0f32;
    let mut max_ch = 0usize;
    for (idx, (&c, &g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_ch = idx % CONV3D_OUT_CH;
        }
    }
    println!(
        "CPU vs WebGPU 3D CNN forward max_abs_diff = {max_abs_diff:.3e} (worst channel {max_ch})"
    );
    assert!(
        max_abs_diff < 1e-4,
        "WebGPU conv3d forward max abs diff {max_abs_diff:.3e} exceeded 1e-4"
    );

    // 3. Compare full 3D NMS peak picking (+CA and -CB) between CPU and WebGPU
    let cpu_cfg = NeuralPickerConfig {
        prob_threshold: 0.48,
        min_snr_norm: 0.08,
        allow_negative: true,
        ..NeuralPickerConfig::default()
    };
    let gpu_cfg = GpuNeuralPickerConfig {
        prob_threshold: 0.48,
        min_snr_norm: 0.08,
        allow_negative: true,
        ..GpuNeuralPickerConfig::default()
    };

    let cpu_peaks = cpu_picker.pick_3d(&syn.spectrum, "HNCACB", &cpu_cfg);
    let gpu_peaks = gpu_picker
        .pick_triple_3d_gpu(&syn.spectrum, "HNCACB", &[], &gpu_cfg)
        .expect("GPU pick_triple_3d_gpu");

    println!(
        "Picked 3D HNCACB peaks: CPU={} vs WebGPU={} (has +CA={}, -CB={})",
        cpu_peaks.len(),
        gpu_peaks.len(),
        gpu_peaks.iter().filter(|p| p.sign > 0).count(),
        gpu_peaks.iter().filter(|p| p.sign < 0).count()
    );
    assert!(!cpu_peaks.is_empty(), "expected picked 3D peaks");
    assert!(cpu_peaks.iter().any(|p| p.sign > 0), "expected +CA peaks");
    assert!(cpu_peaks.iter().any(|p| p.sign < 0), "expected -CB peaks");
    assert_eq!(
        cpu_peaks.len(),
        gpu_peaks.len(),
        "CPU vs GPU NMS peak count mismatch"
    );

    for (i, (cp, gp)) in cpu_peaks.iter().zip(gpu_peaks.iter()).enumerate() {
        assert_eq!(cp.sign, gp.sign, "sign mismatch at peak {i}");
        assert!(
            (cp.h - gp.h).abs() < 1e-4,
            "1H ppm mismatch at peak {i}: {} vs {}",
            cp.h,
            gp.h
        );
        assert!(
            (cp.n - gp.n).abs() < 1e-4,
            "15N ppm mismatch at peak {i}: {} vs {}",
            cp.n,
            gp.n
        );
        assert!(
            (cp.c.unwrap() - gp.c.unwrap()).abs() < 1e-4,
            "13C ppm mismatch at peak {i}: {:?} vs {:?}",
            cp.c,
            gp.c
        );
    }

    // 4. Verify HSQC-strip-restricted 3D picking parity between CPU subregion halo and GPU NMS
    let roots: Vec<peaks::Peak> = syn
        .peaks
        .iter()
        .take(5)
        .map(|p| peaks::Peak::hsqc(p.h, p.n))
        .collect();
    let cpu_strip_peaks =
        cpu_picker.pick_3d(&syn.spectrum, "HNCACB", &cpu_cfg.clone().with_roots(&roots));
    let gpu_strip_peaks = gpu_picker
        .pick_triple_3d_gpu(&syn.spectrum, "HNCACB", &roots, &gpu_cfg)
        .expect("GPU strip pick_triple_3d_gpu");
    assert_eq!(cpu_strip_peaks.len(), gpu_strip_peaks.len());
    for (cp, gp) in cpu_strip_peaks.iter().zip(gpu_strip_peaks.iter()) {
        assert_eq!(cp.sign, gp.sign);
        assert!((cp.h - gp.h).abs() < 1e-4);
        assert!((cp.n - gp.n).abs() < 1e-4);
        assert!((cp.c.unwrap() - gp.c.unwrap()).abs() < 1e-4);
    }
}
