//! Parity tests between `nmr-core` (`f64` reference) and `nmr-gpu` (`wgpu` WGSL compute shaders).

use std::path::PathBuf;

use nmr_core::evaluate::score;
use nmr_core::nmrstar::read_star;
use nmr_core::priors::Priors;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{pair_matrix, unary_matrix, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_gpu::GpuEngine;

#[test]
fn test_gpu_scoring_and_annealing_parity() {
    let star_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/fixtures/synthetic_ubiquitin.str");
    let ref_data = read_star(star_path, None).unwrap();
    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::default()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::provisional();
    let scfg = ScoreConfig::default();

    let gpu = GpuEngine::new_headless().expect("headless wgpu adapter");
    for assign_cost in [0.0, 1.25] {
        let scfg_test = ScoreConfig {
            assign_cost,
            ..ScoreConfig::default()
        };
        let u_cpu = unary_matrix(&systems, &ref_data.sequence, &priors, &scfg_test);
        let p_cpu = pair_matrix(&systems, &scfg_test);

        let (u_gpu, p_gpu) = gpu
            .score_matrices_gpu(&systems, &ref_data.sequence, &priors, &scfg_test)
            .expect("GPU scoring matrices");

        assert_eq!(u_cpu.rows, u_gpu.rows);
        assert_eq!(u_cpu.cols, u_gpu.cols);
        for i in 0..u_cpu.data.len() {
            let a = u_cpu.data[i];
            let b = u_gpu.data[i];
            // f32 has ~1.5e-5 ppm ULP near 176 ppm (C'), giving ~3e-4 ULP sensitivity in (d / 0.12)^2
            if a < -1e8 {
                assert!(b < -1e8, "expected FORBID at {i}: cpu={a}, gpu={b}");
            } else {
                assert!(
                    (a - b).abs() < 1e-3,
                    "U mismatch (assign_cost={assign_cost}) at {i}: cpu={a}, gpu={b}"
                );
            }
        }

        for i in 0..p_cpu.data.len() {
            let a = p_cpu.data[i];
            let b = p_gpu.data[i];
            if a < -1e8 {
                assert!(b < -1e8);
            } else {
                assert!((a - b).abs() < 1e-3, "P mismatch at {i}: cpu={a}, gpu={b}");
            }
        }
    }

    let res_gpu = gpu
        .assign_gpu(
            &systems,
            &ref_data.sequence,
            &priors,
            &scfg,
            &AnnealConfig {
                runs: 32,
                steps_per_system: 1200,
                seed: 0,
                ..AnnealConfig::default()
            },
        )
        .expect("GPU annealing");
    let m = score(&res_gpu, &truth, 0.8);
    assert!(m.recall() >= 0.90, "{}", m.report());
    assert!(m.hc_precision() >= 0.95, "{}", m.report());
}
