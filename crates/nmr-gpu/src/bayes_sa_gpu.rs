//! Clean-room WebGPU (`wgpu` + WGSL) compute engine for Bayesian pseudo-energy evaluation
//! (`E_type`, `E_unary`, `E_link`, and `Q27.36` 64-bit integer state energy) and massively parallel
//! multi-replica thermodynamic specific-heat ($C_v(T)$) adaptive simulated annealing
//! (`docs/specs/19-bayesian-sa-solver.md`).
//!
//! # Provenance (100% Clean-Room)
//! Derived strictly from peer-reviewed statistical mechanics and biomolecular NMR publications,
//! using our CC0 empirical BMRB prior corpus (`data/priors/bmrb_priors.json`):
//! 1. **Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J.** (2023).
//!    "Robust automated backbone triple resonance NMR assignments of proteins using
//!    Bayesian-based simulated annealing." *Nature Communications* **14**, 1556.
//!    DOI: `10.1038/s41467-023-37219-z`.
//! 2. **Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P.** (1983). "Optimization by Simulated
//!    Annealing." *Science* **220**(4598), 671–680.
//! 3. **Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.** (1995).
//!    *J. Biomol. NMR* **5**, 67–81 & **6**, 135–140 (primary-sequence covalent neighbor shifts).
//!
//! No external source code or external parameter tables were consulted.

use bytemuck::{Pod, Zeroable};
use nmr_core::bayes_sa::{
    build_bayes_move_tables, build_initial_bayes_replicas, condition_peaks_preserving_hsqc_roots,
    evaluate_state_energy, finalize_bayes_sa_from_replicas, pack_bayes_scoring_inputs,
    BayesMoveTables, BayesPackedScoringInputs, BayesSaConfig, BayesSaMatrices, BayesSaReport,
    BayesScheduleDriver, FORBID_ENERGY,
};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::rng::Rng;
use nmr_core::solver::{AssignResult, Matrix2D};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum MCMC proposals executed per single WebGPU command-buffer submission (`<= 4,096`
/// to prevent OS GPU watchdog / Windows WDDM TDR resets on long stages).
pub const MAX_BAYES_GPU_CHUNK_STEPS: u32 = 4_096;

/// Select per-submit MCMC step chunk cap (`<= 4,096`) based on sequence/spin dimension.
#[inline]
pub const fn bayes_gpu_chunk_steps_for_dim(max_dim: usize) -> u32 {
    if max_dim <= 1024 {
        4_096
    } else {
        1_024
    }
}

/// Raw output of a single WebGPU annealing stage (used to test bit-for-bit TDR multi-chunk
/// equivalence and global-buffer fallback execution).
#[derive(Clone, Debug, PartialEq)]
pub struct BayesGpuStageTrajectory {
    pub cur_ats: Vec<Vec<i32>>,
    pub best_ats: Vec<Vec<i32>>,
    pub cur_positions: Vec<Vec<i32>>,
    /// Per-replica `[cur_e, best_e, mean_e, m2_e]` in `f32`.
    pub stats_float: Vec<[f32; 4]>,
    /// Per-replica `[n_samples, n_accepted, n_proposed, 0]`.
    pub stats_uint: Vec<[u32; 4]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BayesGpuParams {
    s_len: u32,
    k_len: u32,
    replicas: u32,
    n_fk: u32,

    n_fs: u32,
    cand_m: u32,
    max_w: u32,
    stage_steps: u32,

    step_offset: u32,
    chunk_steps: u32,
    burn_in: u32,
    use_hn: u32,

    outlier_eps: f32,
    p_leak_std: f32,
    p_leak_deg: f32,
    mu_null: f32,

    sigma_ca: f32,
    sigma_cb: f32,
    sigma_c: f32,
    sigma_ha: f32,

    inv_temp: f32,
    r_glob: f32,
    r_cand: f32,
    r_ext: f32,

    off_f0: u32,
    off_f1: u32,
    off_f2: u32,
    off_f3: u32,

    off_f4: u32,
    off_f5: u32,
    off_f6: u32,
    off_f7: u32,

    off_f8: u32,
    off_f9: u32,
    off_f10: u32,
    off_f11: u32,

    off_f12: u32,
    off_f13: u32,
    off_f14: u32,
    off_f15: u32,

    off_i0: u32,
    off_i1: u32,
    off_i2: u32,
    off_i3: u32,

    off_i4: u32,
    off_i5: u32,
    off_i6: u32,
    off_i7: u32,

    off_i8: u32,
    off_mat_link_out: u32,
    _pad_u0: u32,
    _pad_u1: u32,
}

const _: () = assert!(std::mem::size_of::<BayesGpuParams>() == 208);

fn check_storage_buffer_limits(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
) -> Result<(), String> {
    let lim = device.limits();
    let max_buf = lim.max_buffer_size;
    let max_bind = lim.max_storage_buffer_binding_size;
    if bytes > max_buf {
        return Err(format!(
            "bayes_sa_gpu buffer '{label}' ({bytes} B) exceeds device max_buffer_size ({max_buf} B)"
        ));
    }
    if bytes > max_bind {
        return Err(format!(
            "bayes_sa_gpu buffer '{label}' ({bytes} B) exceeds device max_storage_buffer_binding_size ({max_bind} B)"
        ));
    }
    Ok(())
}

/// Validate that all scoring and multi-replica state storage buffers for `(s_len, k_len, replicas)`
/// fit inside `device.limits().max_buffer_size` and `max_storage_buffer_binding_size`.
pub fn validate_bayes_gpu_buffer_budget(
    device: &wgpu::Device,
    s_len: usize,
    k_len: usize,
    replicas: usize,
) -> Result<(), String> {
    let r = replicas.max(1);
    let sk = s_len
        .checked_mul(k_len)
        .ok_or_else(|| "s_len * k_len overflow".to_string())?;
    let ss = s_len
        .checked_mul(s_len)
        .ok_or_else(|| "s_len * s_len overflow".to_string())?;
    let mat_ab_elems = sk
        .checked_add(ss.max(s_len))
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let mat_bytes = (mat_ab_elems as u64).saturating_mul(4).max(16);
    check_storage_buffer_limits(device, "energy-matrices", mat_bytes)?;

    let at_elems = r
        .checked_mul(k_len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| "replica at_buf element overflow".to_string())?;
    let at_bytes = (at_elems as u64).saturating_mul(4).max(16);
    check_storage_buffer_limits(device, "replica-at-state", at_bytes)?;

    let pos_elems = r
        .checked_mul(s_len)
        .ok_or_else(|| "replica pos_buf element overflow".to_string())?;
    let pos_bytes = (pos_elems as u64).saturating_mul(4).max(16);
    check_storage_buffer_limits(device, "replica-pos-state", pos_bytes)?;

    let state_u_bytes = (r as u64).saturating_mul(2 * 16).max(32);
    check_storage_buffer_limits(device, "replica-rng-telemetry", state_u_bytes)?;

    let max_wg = device.limits().max_compute_workgroups_per_dimension as usize;
    let wg_s = s_len.div_ceil(16);
    let wg_k = k_len.div_ceil(16);
    if wg_s > max_wg || wg_k > max_wg {
        return Err(format!(
            "bayes_sa_gpu scoring workgroups ({wg_s}x{wg_k}) exceed max_compute_workgroups_per_dimension ({max_wg})"
        ));
    }
    let max_dim = s_len.max(k_len);
    let wg_threads = if max_dim <= 512 {
        8usize
    } else if max_dim <= 1024 {
        4usize
    } else {
        32usize
    };
    let anneal_wgs = r.div_ceil(wg_threads);
    if anneal_wgs > max_wg {
        return Err(format!(
            "bayes_sa_gpu replica workgroups ({anneal_wgs}) exceed max_compute_workgroups_per_dimension ({max_wg})"
        ));
    }
    Ok(())
}

fn seed_xoshiro128(seed: u64) -> [u32; 4] {
    let mut rng = Rng::new(seed ^ 0xA076_1D64_78BD_642F);
    let a = rng.next_u64();
    let b = rng.next_u64();
    let mut st = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
    if st == [0, 0, 0, 0] {
        st[0] = 1;
    }
    st
}

fn pack_scoring_buffers(inp: &BayesPackedScoringInputs) -> (BayesGpuParams, Vec<f32>, Vec<i32>) {
    let mut ro_f = Vec::new();
    let mut ro_i = Vec::new();

    let push_f64s = |dst: &mut Vec<f32>, src: &[f64]| -> u32 {
        let off = dst.len() as u32;
        dst.extend(src.iter().map(|&v| v as f32));
        off
    };

    let off_f0 = push_f64s(&mut ro_f, &inp.bg_mu);
    let off_f1 = push_f64s(&mut ro_f, &inp.bg_sig_t);
    let off_f2 = push_f64s(&mut ro_f, &inp.res_conf_w);
    let off_f3 = push_f64s(&mut ro_f, &inp.pi_bg_chan);

    let off_f4 = ro_f.len() as u32;
    for &p in &inp.pi_seq_i {
        ro_f.push(if p > 0.0 { p.ln() as f32 } else { -1.0e30 });
    }
    let off_f5 = ro_f.len() as u32;
    for &p in &inp.pi_seq_im1 {
        ro_f.push(if p > 0.0 { p.ln() as f32 } else { -1.0e30 });
    }
    let off_f6 = push_f64s(&mut ro_f, &inp.log_p0_joint);
    let off_f7 = push_f64s(&mut ro_f, &inp.seq_mu);
    let off_f8 = push_f64s(&mut ro_f, &inp.seq_sig_t);
    let off_f9 = push_f64s(&mut ro_f, &inp.seq_log_p_joint);
    let off_f10 = push_f64s(&mut ro_f, &inp.unary_cand_val);
    let off_f11 = push_f64s(&mut ro_f, &inp.unary_cand_w);
    let off_f12 = push_f64s(&mut ro_f, &inp.link_cand_val);
    let off_f13 = push_f64s(&mut ro_f, &inp.link_cand_w);
    let off_f14 = push_f64s(&mut ro_f, &inp.sys_extra_unary);
    while ro_f.len() < 4 {
        ro_f.push(0.0);
    }

    let push_u32s_as_i32 = |dst: &mut Vec<i32>, src: &[u32]| -> u32 {
        let off = dst.len() as u32;
        dst.extend(src.iter().map(|&v| v as i32));
        off
    };
    let push_i32s = |dst: &mut Vec<i32>, src: &[i32]| -> u32 {
        let off = dst.len() as u32;
        dst.extend_from_slice(src);
        off
    };

    let off_i0 = push_u32s_as_i32(&mut ro_i, &inp.res_atom_present);
    let off_i1 = push_u32s_as_i32(&mut ro_i, &inp.seq_res_idx);
    let off_i2 = push_u32s_as_i32(&mut ro_i, &inp.unary_cand_cnt);
    let off_i3 = push_u32s_as_i32(&mut ro_i, &inp.link_cand_cnt);
    let off_i4 = push_u32s_as_i32(&mut ro_i, &inp.link_only_deg);
    let off_i5 = push_u32s_as_i32(&mut ro_i, &inp.sys_forbidden);
    let off_i6 = push_u32s_as_i32(&mut ro_i, &inp.sys_ca_less_forbid);
    let off_i7 = push_i32s(&mut ro_i, &inp.lock_k_for_s);
    let off_i8 = push_i32s(&mut ro_i, &inp.lock_s_for_k);
    while ro_i.len() < 4 {
        ro_i.push(0);
    }

    let p_leak_std = inp.link_floor.exp() as f32;
    let p_leak_deg = ((-2.5_f64).exp() as f32).max(p_leak_std);

    let params = BayesGpuParams {
        s_len: inp.s_len as u32,
        k_len: inp.k_len as u32,
        replicas: 0,
        n_fk: 0,
        n_fs: 0,
        cand_m: 0,
        max_w: 0,
        stage_steps: 0,
        step_offset: 0,
        chunk_steps: 0,
        burn_in: 0,
        use_hn: u32::from(inp.use_hn),
        outlier_eps: inp.outlier_floor_eps as f32,
        p_leak_std,
        p_leak_deg,
        mu_null: inp.mu_null as f32,
        sigma_ca: inp.sigmas_link[0] as f32,
        sigma_cb: inp.sigmas_link[1] as f32,
        sigma_c: inp.sigmas_link[2] as f32,
        sigma_ha: inp.sigmas_link[3] as f32,
        inv_temp: 1.0,
        r_glob: 0.0,
        r_cand: 0.0,
        r_ext: 0.0,
        off_f0,
        off_f1,
        off_f2,
        off_f3,
        off_f4,
        off_f5,
        off_f6,
        off_f7,
        off_f8,
        off_f9,
        off_f10,
        off_f11,
        off_f12,
        off_f13,
        off_f14,
        off_f15: inp.s_len as u32,
        off_i0,
        off_i1,
        off_i2,
        off_i3,
        off_i4,
        off_i5,
        off_i6,
        off_i7,
        off_i8,
        off_mat_link_out: (inp.s_len * inp.k_len) as u32,
        _pad_u0: 0,
        _pad_u1: 0,
    };

    (params, ro_f, ro_i)
}

fn pack_move_tables_buffer(tables: &BayesMoveTables) -> ([u32; 8], Vec<i32>) {
    let mut buf = Vec::new();
    let off_free_k = buf.len() as u32;
    buf.extend(tables.free_k.iter().map(|&v| v as i32));

    let off_free_k_idx = buf.len() as u32;
    buf.extend_from_slice(&tables.free_k_idx);

    let off_free_s = buf.len() as u32;
    buf.extend(tables.free_s.iter().map(|&v| v as i32));

    let off_unlocked_k = buf.len() as u32;
    buf.extend(tables.unlocked_k_flag.iter().map(|&v| v as i32));

    let off_cand_spins = buf.len() as u32;
    let mut cand_sorted = tables.cand_spins.clone();
    if tables.cand_m > 1 {
        for row in cand_sorted.chunks_exact_mut(tables.cand_m) {
            row.sort_unstable_by(|a, b| b.cmp(a));
        }
    }
    buf.extend_from_slice(&cand_sorted);

    let sort_rows8_desc = |src: &[i32]| -> Vec<i32> {
        let mut copy = src.to_vec();
        for row in copy.chunks_exact_mut(8) {
            row.sort_unstable_by(|a, b| b.cmp(a));
        }
        copy
    };

    let off_top_k8 = buf.len() as u32;
    buf.extend_from_slice(&sort_rows8_desc(&tables.top_k8));

    let off_succ8 = buf.len() as u32;
    buf.extend_from_slice(&sort_rows8_desc(&tables.succ8));

    let off_pred8 = buf.len() as u32;
    buf.extend_from_slice(&sort_rows8_desc(&tables.pred8));

    while buf.len() < 4 {
        buf.push(0);
    }

    (
        [
            off_free_k,
            off_free_k_idx,
            off_free_s,
            off_unlocked_k,
            off_cand_spins,
            off_top_k8,
            off_succ8,
            off_pred8,
        ],
        buf,
    )
}

fn pack_matrices_f32(matrices: &BayesSaMatrices) -> Vec<f32> {
    let sk = matrices.e_unary.data.len();
    let ss = matrices.e_link.data.len();
    let mut out = Vec::with_capacity((sk + ss).max(4));
    out.extend(matrices.e_unary.data.iter().map(|&v| v as f32));
    out.extend(matrices.e_link.data.iter().map(|&v| v as f32));
    while out.len() < 4 {
        out.push(0.0);
    }
    out
}

const Q36_SCALE: f64 = 68_719_476_736.0; // 2^36 (~1.45e-11 nats resolution)

fn pack_matrices_q36(matrices: &BayesSaMatrices) -> Vec<i32> {
    let total = (matrices.e_unary.data.len() + matrices.e_link.data.len()).max(2);
    let mut out = Vec::with_capacity(2 * total);
    for &v in matrices
        .e_unary
        .data
        .iter()
        .chain(matrices.e_link.data.iter())
    {
        if !v.is_finite() || v >= FORBID_ENERGY * 0.5 {
            out.push(-1i32); // 0xFFFF_FFFFu32 sentinel lo
            out.push(i32::MAX); // 0x7FFF_FFFFi32 sentinel hi
        } else {
            let q = (v * Q36_SCALE).round() as i64;
            let u = q as u64;
            out.push((u as u32) as i32);
            out.push(((u >> 32) as u32) as i32);
        }
    }
    while out.len() < 4 {
        out.push(0);
    }
    out
}

/// WebGPU compute pipelines and context for Bayesian pseudo-energy evaluation and specific-heat
/// $C_v(T)$ adaptive simulated annealing.
pub struct BayesSaGpuSolver {
    pub engine: GpuEngine,
    bgl: wgpu::BindGroupLayout,
    ln_z_bg_pipeline: wgpu::ComputePipeline,
    e_type_pipeline: wgpu::ComputePipeline,
    e_link_pipeline: wgpu::ComputePipeline,
    eval_states_pipeline: wgpu::ComputePipeline,
    anneal_256_pipeline: wgpu::ComputePipeline,
    anneal_1024_pipeline: wgpu::ComputePipeline,
    anneal_global_pipeline: wgpu::ComputePipeline,
}

impl BayesSaGpuSolver {
    /// Initialize standalone headless WebGPU Bayesian SA solver with hardware buffer limits.
    pub fn new_headless() -> Result<Self, String> {
        Ok(Self::new(GpuEngine::new_headless()?))
    }

    /// Initialize headless WebGPU Bayesian SA solver with caller-supplied `wgpu::Limits`
    /// (for portable browser-cap and buffer-limit rejection tests).
    pub fn new_headless_with_limits(required_limits: wgpu::Limits) -> Result<Self, String> {
        Ok(Self::new(GpuEngine::new_headless_with_limits(
            required_limits,
        )?))
    }

    /// Build `BayesSaGpuSolver` pipelines atop an existing `GpuEngine`.
    pub fn new(engine: GpuEngine) -> Self {
        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nmr-bayes-sa-wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bayes_sa.wgsl").into()),
            });

        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = engine
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("nmr-bayes-sa-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw(3),
                    storage_rw(4),
                    storage_rw(5),
                    storage_rw(6),
                    storage_rw(7),
                    storage_rw(8),
                ],
            });

        let pl = engine
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nmr-bayes-sa-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let make_pipe = |label: &str, entry: &str| {
            engine
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Self {
            ln_z_bg_pipeline: make_pipe("nmr-bayes-ln-z-bg-pipe", "compute_ln_z_bg"),
            e_type_pipeline: make_pipe("nmr-bayes-e-type-pipe", "compute_e_type"),
            e_link_pipeline: make_pipe("nmr-bayes-e-link-pipe", "compute_e_link"),
            eval_states_pipeline: make_pipe("nmr-bayes-eval-states-pipe", "eval_state_energies"),
            anneal_256_pipeline: make_pipe("nmr-bayes-anneal-256-pipe", "anneal_bayes_256"),
            anneal_1024_pipeline: make_pipe("nmr-bayes-anneal-1024-pipe", "anneal_bayes_1024"),
            anneal_global_pipeline: make_pipe(
                "nmr-bayes-anneal-global-pipe",
                "anneal_bayes_global",
            ),
            engine,
            bgl,
        }
    }

    fn compute_energy_matrices_with_gpu_buf(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        cfg: &BayesSaConfig,
    ) -> Result<(BayesSaMatrices, Option<wgpu::Buffer>), String> {
        let inp = pack_bayes_scoring_inputs(systems, sequence, priors, cfg);
        let s_len = inp.s_len;
        let k_len = inp.k_len;
        if s_len == 0 || k_len == 0 {
            let empty = BayesSaMatrices {
                e_type: Matrix2D::filled(s_len, k_len, FORBID_ENERGY),
                e_unary: Matrix2D::filled(s_len, k_len, FORBID_ENERGY),
                e_link: Matrix2D::filled(s_len, s_len, 0.0),
                weak_penalties: inp.weak_penalties,
                mu_null: inp.mu_null,
                applied_offsets: inp.applied_offsets,
                detailed_offsets: inp.detailed_offsets,
                effective_priors: inp.effective_priors,
                resolved_locks: inp.resolved_locks,
                active_slots: inp.active_slots,
            };
            return Ok((empty, None));
        }

        validate_bayes_gpu_buffer_budget(&self.engine.device, s_len, k_len, 1)?;
        let (params, ro_f, ro_i) = pack_scoring_buffers(&inp);
        check_storage_buffer_limits(
            &self.engine.device,
            "score-ro-floats",
            (ro_f.len() * 4) as u64,
        )?;
        check_storage_buffer_limits(
            &self.engine.device,
            "score-ro-ints",
            (ro_i.len() * 4) as u64,
        )?;

        let mat_a_len = (s_len + s_len * k_len).max(4);
        let mat_b_len = (s_len * k_len + s_len * s_len).max(4);
        let mat_a_bytes = (mat_a_len * 4) as u64;
        let mat_b_bytes = (mat_b_len * 4) as u64;
        check_storage_buffer_limits(&self.engine.device, "score-rw-mat-a", mat_a_bytes)?;
        check_storage_buffer_limits(&self.engine.device, "score-rw-mat-b", mat_b_bytes)?;

        let dev = &self.engine.device;
        let u_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_ro_f = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-ro-f"),
            contents: bytemuck::cast_slice(&ro_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_ro_i = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-ro-i"),
            contents: bytemuck::cast_slice(&ro_i),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_mat_a = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bayes-score-mat-a"),
            size: mat_a_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_mat_b = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bayes-score-mat-b"),
            size: mat_b_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dummy_i32 = [0i32; 4];
        let dummy_f4 = [[0.0f32; 4]; 1];
        let dummy_u4 = [[0u32; 4]; 2];
        let b_at = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-dummy-at"),
            contents: bytemuck::cast_slice(&dummy_i32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_pos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-dummy-pos"),
            contents: bytemuck::cast_slice(&dummy_i32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_sf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-dummy-sf"),
            contents: bytemuck::cast_slice(&dummy_f4),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_su = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-score-dummy-su"),
            contents: bytemuck::cast_slice(&dummy_u4),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayes-score-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_ro_f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_ro_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_mat_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_mat_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_sf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_su.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bayes-score-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bayes-score-pass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bg, &[]);

            cpass.set_pipeline(&self.ln_z_bg_pipeline);
            cpass.dispatch_workgroups((s_len as u32).div_ceil(64), 1, 1);

            cpass.set_pipeline(&self.e_type_pipeline);
            cpass.dispatch_workgroups((s_len as u32).div_ceil(16), (k_len as u32).div_ceil(16), 1);

            cpass.set_pipeline(&self.e_link_pipeline);
            cpass.dispatch_workgroups((s_len as u32).div_ceil(16), (s_len as u32).div_ceil(16), 1);
        }
        self.engine.queue.submit(Some(enc.finish()));

        let raw_a: Vec<f32> = self.engine.read_buffer_vec(&b_mat_a, mat_a_bytes)?;
        let raw_b: Vec<f32> = self.engine.read_buffer_vec(&b_mat_b, mat_b_bytes)?;

        let sk = s_len * k_len;
        let ss = s_len * s_len;
        let e_type_f64: Vec<f64> = raw_a[s_len..s_len + sk]
            .iter()
            .map(|&v| if v >= 5.0e8 { FORBID_ENERGY } else { v as f64 })
            .collect();
        let e_unary_f64: Vec<f64> = raw_b[0..sk]
            .iter()
            .map(|&v| if v >= 5.0e8 { FORBID_ENERGY } else { v as f64 })
            .collect();
        let e_link_f64: Vec<f64> = raw_b[sk..sk + ss]
            .iter()
            .map(|&v| if v >= 5.0e8 { FORBID_ENERGY } else { v as f64 })
            .collect();

        let matrices = BayesSaMatrices {
            e_type: Matrix2D {
                rows: s_len,
                cols: k_len,
                data: e_type_f64,
            },
            e_unary: Matrix2D {
                rows: s_len,
                cols: k_len,
                data: e_unary_f64,
            },
            e_link: Matrix2D {
                rows: s_len,
                cols: s_len,
                data: e_link_f64,
            },
            weak_penalties: inp.weak_penalties,
            mu_null: inp.mu_null,
            applied_offsets: inp.applied_offsets,
            detailed_offsets: inp.detailed_offsets,
            effective_priors: inp.effective_priors,
            resolved_locks: inp.resolved_locks,
            active_slots: inp.active_slots,
        };

        Ok((matrices, Some(b_mat_b)))
    }

    /// Compute `E_type`, `E_unary`, and `E_link` on WebGPU (`< 1e-4` parity vs CPU `f64`).
    pub fn compute_energy_matrices(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        cfg: &BayesSaConfig,
    ) -> Result<BayesSaMatrices, String> {
        let (matrices, _) =
            self.compute_energy_matrices_with_gpu_buf(systems, sequence, priors, cfg)?;
        Ok(matrices)
    }

    /// Evaluate exact total Bayesian pseudo-energy $E(\mathbf{x})$ for a batch of sequence-slot
    /// assignment states `state_ats` on WebGPU using fast-math-immune `Q27.36` 64-bit integer
    /// accumulation (`< 1e-4` parity vs `nmr_core::bayes_sa::evaluate_state_energy`).
    pub fn evaluate_states_energy(
        &self,
        matrices: &BayesSaMatrices,
        state_ats: &[Vec<i32>],
    ) -> Result<Vec<f64>, String> {
        if state_ats.is_empty() {
            return Ok(Vec::new());
        }
        let s_len = matrices.e_unary.rows;
        let k_len = matrices.e_unary.cols;
        let n_states = state_ats.len();

        // Validate state structural invariants identically to CPU evaluate_state_energy
        for (idx, at) in state_ats.iter().enumerate() {
            if at.len() != k_len {
                return Err(format!("state[{idx}] len {} != k_len {k_len}", at.len()));
            }
            let mut seen = vec![false; s_len];
            for (k, &s_raw) in at.iter().enumerate() {
                if s_raw >= 0 {
                    let s = s_raw as usize;
                    if s >= s_len {
                        return Err(format!(
                            "state[{idx}] invalid spin {s} >= {s_len} at slot {k}"
                        ));
                    }
                    if seen[s] {
                        return Err(format!("state[{idx}] duplicate spin {s}"));
                    }
                    seen[s] = true;
                    if matrices.e_unary.get(s, k) >= FORBID_ENERGY * 0.5 {
                        return Err(format!("state[{idx}] forbidden placement (s={s}, k={k})"));
                    }
                }
            }
        }
        if s_len == 0 || k_len == 0 {
            return Ok(vec![0.0; n_states]);
        }

        let q36_ints = pack_matrices_q36(matrices);
        check_storage_buffer_limits(
            &self.engine.device,
            "eval-matrices-q36",
            (q36_ints.len() * 4) as u64,
        )?;
        let at_bytes = ((n_states * k_len).max(4) * 4) as u64;
        check_storage_buffer_limits(&self.engine.device, "eval-states-at", at_bytes)?;
        let out_bytes = ((2 * n_states).max(4) * 4) as u64;
        check_storage_buffer_limits(&self.engine.device, "eval-states-out", out_bytes)?;
        let max_wg = self
            .engine
            .device
            .limits()
            .max_compute_workgroups_per_dimension as usize;
        let eval_wgs = n_states.div_ceil(64);
        if eval_wgs > max_wg {
            return Err(format!(
                "bayes_sa_gpu eval_states workgroups ({eval_wgs}) exceed max_compute_workgroups_per_dimension ({max_wg})"
            ));
        }

        let mut at_flat = Vec::with_capacity((n_states * k_len).max(4));
        for at in state_ats {
            at_flat.extend_from_slice(at);
        }
        while at_flat.len() < 4 {
            at_flat.push(-1);
        }

        let mut params = BayesGpuParams::zeroed();
        params.s_len = s_len as u32;
        params.k_len = k_len as u32;
        params.replicas = n_states as u32;
        params.off_f0 = 0;
        params.off_f1 = (s_len * k_len) as u32;

        let dev = &self.engine.device;
        let u_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dummy_f = [0.0f32; 4];
        let dummy_i = [0i32; 4];
        let dummy_f4 = [[0.0f32; 4]; 1];
        let dummy_u4 = [[0u32; 4]; 2];
        let b_ro_f = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-dummy-ro-f"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_ro_i = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-matrices-q36"),
            contents: bytemuck::cast_slice(&q36_ints),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_out = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bayes-eval-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_mat_b = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-dummy-mat-b"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_at = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-at"),
            contents: bytemuck::cast_slice(&at_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_pos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-dummy-pos"),
            contents: bytemuck::cast_slice(&dummy_i),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_sf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-dummy-sf"),
            contents: bytemuck::cast_slice(&dummy_f4),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_su = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-eval-dummy-su"),
            contents: bytemuck::cast_slice(&dummy_u4),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayes-eval-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_ro_f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_ro_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_mat_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_sf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_su.as_entire_binding(),
                },
            ],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bayes-eval-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bayes-eval-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.eval_states_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_states as u32).div_ceil(64), 1, 1);
        }
        self.engine.queue.submit(Some(enc.finish()));

        let raw_out: Vec<u32> = self.engine.read_buffer_vec(&b_out, out_bytes)?;
        Ok((0..n_states)
            .map(|i| {
                let lo = raw_out[2 * i];
                let hi = raw_out[2 * i + 1];
                if hi == 0x7FFF_FFFF && lo == 0xFFFF_FFFF {
                    FORBID_ENERGY
                } else {
                    let q = (((hi as u64) << 32) | (lo as u64)) as i64;
                    (q as f64) * (1.0 / Q36_SCALE)
                }
            })
            .collect())
    }

    /// Execute a single stage of WebGPU MCMC with caller-specified `max_chunk_steps` (`<= 4,096`)
    /// and optional `force_global_kernel` to verify bit-for-bit TDR chunking equivalence and
    /// storage-buffer global kernel execution.
    #[allow(clippy::too_many_arguments)]
    pub fn run_stage_raw_with_chunk_cap(
        &self,
        matrices: &BayesSaMatrices,
        tables: &BayesMoveTables,
        cur_ats: &[Vec<i32>],
        cur_poss: &[Vec<i32>],
        cur_energies: &[f64],
        sub_seeds: &[u64],
        temp: f64,
        stage_steps: usize,
        max_fragment_len: usize,
        max_chunk_steps: u32,
        force_global_kernel: bool,
    ) -> Result<BayesGpuStageTrajectory, String> {
        let r_cnt = cur_ats.len();
        if r_cnt == 0 {
            return Err("replicas must be >= 1".to_string());
        }
        let s_len = matrices.e_unary.rows;
        let k_len = matrices.e_unary.cols;
        validate_bayes_gpu_buffer_budget(&self.engine.device, s_len, k_len, r_cnt)?;

        let ro_f = pack_matrices_f32(matrices);
        let (tbl_offs, ro_i) = pack_move_tables_buffer(tables);
        check_storage_buffer_limits(&self.engine.device, "stage-ro-f", (ro_f.len() * 4) as u64)?;
        check_storage_buffer_limits(&self.engine.device, "stage-ro-i", (ro_i.len() * 4) as u64)?;

        let mut at_flat = Vec::with_capacity(2 * r_cnt * k_len);
        for at in cur_ats {
            at_flat.extend_from_slice(at);
        }
        for at in cur_ats {
            at_flat.extend_from_slice(at);
        }
        let mut pos_flat = Vec::with_capacity(r_cnt * s_len);
        for p in cur_poss {
            pos_flat.extend_from_slice(p);
        }
        while at_flat.len() < 4 {
            at_flat.push(-1);
        }
        while pos_flat.len() < 4 {
            pos_flat.push(-1);
        }

        let stats_f_init: Vec<[f32; 4]> = (0..r_cnt)
            .map(|r| {
                let e = cur_energies[r] as f32;
                [e, e, 0.0, 0.0]
            })
            .collect();
        let mut state_u_init = vec![[0u32; 4]; 2 * r_cnt];
        for r in 0..r_cnt {
            state_u_init[r_cnt + r] = seed_xoshiro128(sub_seeds[r]);
        }

        let n_fk = tables.free_k.len();
        let n_fs = tables.free_s.len();
        let has_cands = tables.cand_m > 0;
        let r_glob = if n_fs > 0 {
            (if has_cands { 0.22 } else { 0.62 }) / (n_fs as f32)
        } else {
            0.0
        };
        let r_cand = if has_cands {
            0.40 / (tables.cand_m as f32)
        } else {
            0.0
        };
        let r_ext = 0.19_f32 / 8.0;

        let mut base_params = BayesGpuParams::zeroed();
        base_params.s_len = s_len as u32;
        base_params.k_len = k_len as u32;
        base_params.replicas = r_cnt as u32;
        base_params.n_fk = n_fk as u32;
        base_params.n_fs = n_fs as u32;
        base_params.cand_m = tables.cand_m as u32;
        base_params.max_w = max_fragment_len.clamp(2, 32) as u32;
        base_params.stage_steps = stage_steps as u32;
        base_params.burn_in = (stage_steps / 2) as u32;
        base_params.inv_temp = (1.0 / temp.max(1e-6)) as f32;
        base_params.r_glob = r_glob;
        base_params.r_cand = r_cand;
        base_params.r_ext = r_ext;
        base_params.off_f0 = 0;
        base_params.off_f1 = (s_len * k_len) as u32;
        base_params.off_i0 = tbl_offs[0];
        base_params.off_i1 = tbl_offs[1];
        base_params.off_i2 = tbl_offs[2];
        base_params.off_i3 = tbl_offs[3];
        base_params.off_i4 = tbl_offs[4];
        base_params.off_i5 = tbl_offs[5];
        base_params.off_i6 = tbl_offs[6];
        base_params.off_i7 = tbl_offs[7];

        let dev = &self.engine.device;
        let u_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-params"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let b_ro_f = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-ro-f"),
            contents: bytemuck::cast_slice(&ro_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_ro_i = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-ro-i"),
            contents: bytemuck::cast_slice(&ro_i),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_f = [0.0f32; 4];
        let b_mat_a = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-dummy-a"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_mat_b = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-dummy-b"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_at = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-at"),
            contents: bytemuck::cast_slice(&at_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_pos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-pos"),
            contents: bytemuck::cast_slice(&pos_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_sf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-sf"),
            contents: bytemuck::cast_slice(&stats_f_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_su = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-raw-su"),
            contents: bytemuck::cast_slice(&state_u_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayes-raw-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_ro_f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_ro_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_mat_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_mat_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_sf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_su.as_entire_binding(),
                },
            ],
        });

        let max_dim = s_len.max(k_len);
        let (pipe, wg_size) = if force_global_kernel || max_dim > 1024 {
            (&self.anneal_global_pipeline, 32u32)
        } else if max_dim <= 512 {
            (&self.anneal_256_pipeline, 8u32)
        } else {
            (&self.anneal_1024_pipeline, 4u32)
        };
        let n_wg = (r_cnt as u32).div_ceil(wg_size);
        let chunk_cap = max_chunk_steps.clamp(1, MAX_BAYES_GPU_CHUNK_STEPS) as usize;

        let mut step_offset = 0usize;
        loop {
            let chunk = (stage_steps - step_offset).min(chunk_cap);
            base_params.step_offset = step_offset as u32;
            base_params.chunk_steps = chunk as u32;
            self.engine
                .queue
                .write_buffer(&u_params, 0, bytemuck::bytes_of(&base_params));

            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bayes-raw-chunk-enc"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bayes-raw-chunk-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipe);
                cpass.set_bind_group(0, &bg, &[]);
                cpass.dispatch_workgroups(n_wg, 1, 1);
            }
            self.engine.queue.submit(Some(enc.finish()));

            step_offset += chunk;
            if step_offset >= stage_steps {
                break;
            }
        }

        let at_out: Vec<i32> = self
            .engine
            .read_buffer_vec(&b_at, (at_flat.len() * 4) as u64)?;
        let pos_out: Vec<i32> = self
            .engine
            .read_buffer_vec(&b_pos, (pos_flat.len() * 4) as u64)?;
        let sf_out: Vec<[f32; 4]> = self.engine.read_buffer_vec(&b_sf, (r_cnt * 16) as u64)?;
        let su_all: Vec<[u32; 4]> = self
            .engine
            .read_buffer_vec(&b_su, (2 * r_cnt * 16) as u64)?;

        let mut out_cur_ats = Vec::with_capacity(r_cnt);
        let mut out_best_ats = Vec::with_capacity(r_cnt);
        let mut out_cur_poss = Vec::with_capacity(r_cnt);
        for r in 0..r_cnt {
            out_cur_ats.push(at_out[r * k_len..(r + 1) * k_len].to_vec());
            out_best_ats.push(at_out[(r_cnt + r) * k_len..(r_cnt + r + 1) * k_len].to_vec());
            out_cur_poss.push(pos_out[r * s_len..(r + 1) * s_len].to_vec());
        }

        Ok(BayesGpuStageTrajectory {
            cur_ats: out_cur_ats,
            best_ats: out_best_ats,
            cur_positions: out_cur_poss,
            stats_float: sf_out,
            stats_uint: su_all[..r_cnt].to_vec(),
        })
    }

    /// Run full multi-replica specific-heat ($C_v(T)$) adaptive simulated annealing on WebGPU,
    /// returning `(AssignResult, BayesSaReport)` compatible with `nmr-core::bayes_sa`.
    pub fn solve_with_peaks(
        &self,
        peaks: Option<&[Peak]>,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        cfg: &BayesSaConfig,
    ) -> Result<(AssignResult, BayesSaReport), String> {
        let (matrices, gpu_mat_b_opt) =
            self.compute_energy_matrices_with_gpu_buf(systems, sequence, priors, cfg)?;
        if systems.is_empty() || sequence.is_empty() {
            return Ok(finalize_bayes_sa_from_replicas(
                peaks,
                systems,
                sequence,
                priors,
                cfg,
                &matrices,
                &[],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                false,
            ));
        }

        let s_len = matrices.e_unary.rows;
        let k_len = matrices.e_unary.cols;
        let r_cnt = cfg.replicas.max(1);
        validate_bayes_gpu_buffer_budget(&self.engine.device, s_len, k_len, r_cnt)?;

        let tables = build_bayes_move_tables(&matrices, cfg);
        let (tbl_offs, ro_i) = pack_move_tables_buffer(&tables);
        check_storage_buffer_limits(
            &self.engine.device,
            "anneal-move-tables",
            (ro_i.len() * 4) as u64,
        )?;

        let mut master_rng = Rng::new(cfg.seed);
        let (init_ats, init_poss, init_es, sub_seeds) =
            build_initial_bayes_replicas(&matrices, &tables, r_cnt, &mut master_rng);

        let mut at_flat = Vec::with_capacity(2 * r_cnt * k_len);
        for at in &init_ats {
            at_flat.extend_from_slice(at);
        }
        for at in &init_ats {
            at_flat.extend_from_slice(at);
        }
        let mut pos_flat = Vec::with_capacity(r_cnt * s_len);
        for p in &init_poss {
            pos_flat.extend_from_slice(p);
        }
        while at_flat.len() < 4 {
            at_flat.push(-1);
        }
        while pos_flat.len() < 4 {
            pos_flat.push(-1);
        }

        let stats_f_init: Vec<[f32; 4]> = (0..r_cnt)
            .map(|r| {
                let e = init_es[r] as f32;
                [e, e, 0.0, 0.0]
            })
            .collect();
        let mut state_u_init = vec![[0u32; 4]; 2 * r_cnt];
        for r in 0..r_cnt {
            state_u_init[r_cnt + r] = seed_xoshiro128(sub_seeds[r]);
        }

        let n_fk = tables.free_k.len();
        let n_fs = tables.free_s.len();
        let has_cands = tables.cand_m > 0;
        let r_glob = if n_fs > 0 {
            (if has_cands { 0.22 } else { 0.62 }) / (n_fs as f32)
        } else {
            0.0
        };
        let r_cand = if has_cands {
            0.40 / (tables.cand_m as f32)
        } else {
            0.0
        };
        let r_ext = 0.19_f32 / 8.0;

        let mut base_params = BayesGpuParams::zeroed();
        base_params.s_len = s_len as u32;
        base_params.k_len = k_len as u32;
        base_params.replicas = r_cnt as u32;
        base_params.n_fk = n_fk as u32;
        base_params.n_fs = n_fs as u32;
        base_params.cand_m = tables.cand_m as u32;
        base_params.max_w = cfg.max_fragment_len.clamp(2, 32) as u32;
        base_params.r_glob = r_glob;
        base_params.r_cand = r_cand;
        base_params.r_ext = r_ext;
        base_params.off_f0 = 0;
        base_params.off_f1 = (s_len * k_len) as u32;
        base_params.off_i0 = tbl_offs[0];
        base_params.off_i1 = tbl_offs[1];
        base_params.off_i2 = tbl_offs[2];
        base_params.off_i3 = tbl_offs[3];
        base_params.off_i4 = tbl_offs[4];
        base_params.off_i5 = tbl_offs[5];
        base_params.off_i6 = tbl_offs[6];
        base_params.off_i7 = tbl_offs[7];

        let dev = &self.engine.device;
        let u_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-params"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let b_ro_f = match gpu_mat_b_opt {
            Some(buf) => buf,
            None => {
                let ro_f = pack_matrices_f32(&matrices);
                dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bayes-anneal-ro-f"),
                    contents: bytemuck::cast_slice(&ro_f),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
        };
        let b_ro_i = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-ro-i"),
            contents: bytemuck::cast_slice(&ro_i),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_f = [0.0f32; 4];
        let b_mat_a = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-dummy-a"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_mat_b = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-dummy-b"),
            contents: bytemuck::cast_slice(&dummy_f),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_at = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-at"),
            contents: bytemuck::cast_slice(&at_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_pos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-pos"),
            contents: bytemuck::cast_slice(&pos_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_sf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-sf"),
            contents: bytemuck::cast_slice(&stats_f_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let b_su = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bayes-anneal-su"),
            contents: bytemuck::cast_slice(&state_u_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bayes-anneal-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_ro_f.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_ro_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_mat_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_mat_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_sf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: b_su.as_entire_binding(),
                },
            ],
        });

        let max_dim = s_len.max(k_len);
        let (pipe, wg_size) = if max_dim <= 512 {
            (&self.anneal_256_pipeline, 8u32)
        } else if max_dim <= 1024 {
            (&self.anneal_1024_pipeline, 4u32)
        } else {
            (&self.anneal_global_pipeline, 32u32)
        };
        let n_wg = (r_cnt as u32).div_ceil(wg_size);
        let chunk_cap = bayes_gpu_chunk_steps_for_dim(max_dim) as usize;

        let sf_bytes = (r_cnt * 16) as u64;
        let su_head_bytes = (r_cnt * 16) as u64;
        let telem_bytes = sf_bytes + su_head_bytes;
        let b_telem_staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bayes-anneal-telem-staging"),
            size: telem_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut telem_u32 = vec![0u32; r_cnt * 8];
        let mut driver = BayesScheduleDriver::new(cfg, matrices.active_slots, r_cnt);

        while !driver.done {
            let stage_steps = driver.stage_steps.max(1);
            base_params.stage_steps = stage_steps as u32;
            base_params.burn_in = (if driver.stage == 0 {
                (3 * stage_steps) / 4
            } else {
                stage_steps / 2
            }) as u32;
            base_params.inv_temp = (1.0 / driver.temp.max(1e-6)) as f32;

            let mut step_offset = 0usize;
            let mut last_sub_idx: Option<wgpu::SubmissionIndex> = None;
            while step_offset < stage_steps {
                let chunk = (stage_steps - step_offset).min(chunk_cap);
                let is_last_chunk = step_offset + chunk >= stage_steps;
                base_params.step_offset = step_offset as u32;
                base_params.chunk_steps = chunk as u32;
                self.engine
                    .queue
                    .write_buffer(&u_params, 0, bytemuck::bytes_of(&base_params));

                let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bayes-anneal-chunk-enc"),
                });
                {
                    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bayes-anneal-chunk-pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(pipe);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups(n_wg, 1, 1);
                }
                if is_last_chunk {
                    enc.copy_buffer_to_buffer(&b_sf, 0, &b_telem_staging, 0, sf_bytes);
                    enc.copy_buffer_to_buffer(&b_su, 0, &b_telem_staging, sf_bytes, su_head_bytes);
                }
                let sub = self.engine.queue.submit(Some(enc.finish()));
                if is_last_chunk {
                    last_sub_idx = Some(sub);
                }
                step_offset += chunk;
            }

            {
                let slice = b_telem_staging.slice(0..telem_bytes);
                let res_cell = std::sync::Arc::new(std::sync::Mutex::new(None));
                let cb_cell = std::sync::Arc::clone(&res_cell);
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    if let Ok(mut g) = cb_cell.lock() {
                        *g = Some(res);
                    }
                });
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let _ = dev.poll(wgpu::PollType::Wait {
                        submission_index: last_sub_idx,
                        timeout: None,
                    });
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = last_sub_idx;
                }
                let map_res = res_cell
                    .lock()
                    .map_err(|_| "telem staging map mutex poisoned".to_string())?
                    .take()
                    .ok_or_else(|| "telem staging map_async did not complete".to_string())?;
                map_res.map_err(|e| format!("telem staging map_async failed: {e}"))?;
                {
                    let mapped = slice
                        .get_mapped_range()
                        .map_err(|e| format!("telem staging mapped range failed: {e}"))?;
                    let words: &[u32] = bytemuck::cast_slice(&mapped[..(telem_u32.len() * 4)]);
                    telem_u32.copy_from_slice(words);
                }
                b_telem_staging.unmap();
            }
            let sf_u32_len = r_cnt * 4;

            for r in 0..r_cnt {
                let sf_base = r * 4;
                let su_base = sf_u32_len + r * 4;
                let best_r = f32::from_bits(telem_u32[sf_base + 1]);
                let mean_r = f32::from_bits(telem_u32[sf_base + 2]);
                let m2_r = f32::from_bits(telem_u32[sf_base + 3]);
                driver.rep_stats[r] = (
                    mean_r as f64,
                    (m2_r as f64).max(0.0),
                    telem_u32[su_base] as usize,
                    telem_u32[su_base + 1] as usize,
                    telem_u32[su_base + 2] as usize,
                    best_r as f64,
                );
            }
            driver.advance_after_stage(cfg);
        }

        let at_out: Vec<i32> = self
            .engine
            .read_buffer_vec(&b_at, (at_flat.len() * 4) as u64)?;
        let sf_final: Vec<[f32; 4]> = self.engine.read_buffer_vec(&b_sf, sf_bytes)?;

        let mut best_ats = Vec::with_capacity(r_cnt);
        let mut best_es = Vec::with_capacity(r_cnt);
        let mut final_es = Vec::with_capacity(r_cnt);
        for r in 0..r_cnt {
            let cur_at = &at_out[r * k_len..(r + 1) * k_len];
            let best_at = at_out[(r_cnt + r) * k_len..(r_cnt + r + 1) * k_len].to_vec();
            let exact_fin =
                evaluate_state_energy(cur_at, &matrices).unwrap_or(sf_final[r][0] as f64);
            best_ats.push(best_at);
            best_es.push(sf_final[r][1] as f64);
            final_es.push(exact_fin);
        }

        Ok(finalize_bayes_sa_from_replicas(
            peaks,
            systems,
            sequence,
            priors,
            cfg,
            &matrices,
            &best_ats,
            best_es,
            final_es,
            driver.trajectory,
            driver.converged_early,
        ))
    }

    /// Run WebGPU Bayesian SA on assembled `SpinSystem`s.
    pub fn solve(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        cfg: &BayesSaConfig,
    ) -> Result<(AssignResult, BayesSaReport), String> {
        self.solve_with_peaks(None, systems, sequence, priors, cfg)
    }

    /// End-to-end WebGPU Bayesian SA directly from raw picked `Peak`s.
    pub fn solve_from_peaks(
        &self,
        peaks: &[Peak],
        sequence: &[String],
        priors: &Priors,
        cfg: &BayesSaConfig,
    ) -> Result<(Vec<SpinSystem>, AssignResult, BayesSaReport), String> {
        let conditioned = if cfg.enable_weak_root_penalty {
            condition_peaks_preserving_hsqc_roots(peaks, &cfg.tolerances)
        } else {
            peaks.to_vec()
        };
        let systems = build_spin_systems(&conditioned, &cfg.tolerances, cfg.max_cands, false)?;
        let (res, rep) = self.solve_with_peaks(Some(peaks), &systems, sequence, priors, cfg)?;
        Ok((systems, res, rep))
    }
}

/// Convenience entry point computing `BayesSaMatrices` on WebGPU via a headless context.
pub fn compute_bayes_energy_matrices_gpu(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> Result<BayesSaMatrices, String> {
    BayesSaGpuSolver::new_headless()?.compute_energy_matrices(systems, sequence, priors, cfg)
}

/// Convenience entry point evaluating batch state energies $E(\mathbf{x})$ on WebGPU.
pub fn evaluate_states_energy_gpu(
    matrices: &BayesSaMatrices,
    state_ats: &[Vec<i32>],
) -> Result<Vec<f64>, String> {
    BayesSaGpuSolver::new_headless()?.evaluate_states_energy(matrices, state_ats)
}

/// Convenience entry point running the WebGPU Bayesian pseudo-energy and specific-heat ($C_v(T)$)
/// adaptive simulated annealing solver (`R = 128` replicas default via `BayesSaConfig::default_gpu`).
pub fn solve_bayes_sa_gpu(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> Result<(AssignResult, BayesSaReport), String> {
    BayesSaGpuSolver::new_headless()?.solve(systems, sequence, priors, cfg)
}

/// Variant of [`solve_bayes_sa_gpu`] forwarding optional raw `peaks` into `DatasetQcReport`.
pub fn solve_bayes_sa_gpu_with_peaks(
    peaks: Option<&[Peak]>,
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> Result<(AssignResult, BayesSaReport), String> {
    BayesSaGpuSolver::new_headless()?.solve_with_peaks(peaks, systems, sequence, priors, cfg)
}

/// End-to-end WebGPU Bayesian SA solver directly from picked `Peak`s.
pub fn solve_bayes_sa_gpu_from_peaks(
    peaks: &[Peak],
    sequence: &[String],
    priors: &Priors,
    cfg: &BayesSaConfig,
) -> Result<(Vec<SpinSystem>, AssignResult, BayesSaReport), String> {
    BayesSaGpuSolver::new_headless()?.solve_from_peaks(peaks, sequence, priors, cfg)
}
