//! WebGPU (`wgpu` + WGSL) compute solver for continuous through-Proline `13C`-direct-detected
//! IDP backbone scoring (`U`, `P` matrices) and massively parallel multi-replica simulated
//! annealing (`Phase 11`, `T11.2`).
//!
//! Provenance: Clean-room GPU compute implementation derived strictly from peer-reviewed literature:
//! - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn.
//!   Reson. Spectrosc.* 2006, 48, 25–45; *J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem.
//!   Soc.* 2009, 131, 15339–15345 — protonless `2D CON`, `2D CACO`, `3D CANCO`, `CBCACON`,
//!   `CCC(CO)N`, and `NCACO` continuous through-Proline IDP backbone assignment).
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP`
//!   sequence-neighbor random-coil chemical shift library for intrinsically disordered proteins).
//! - Kjaergaard, M. & Poulsen, F. M. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57 —
//!   temperature and neighbor effects on disordered protein chemical shifts).

use bytemuck::{Pod, Zeroable};
use nmr_core::c13_detected::C13PickedDataset;
use nmr_core::idp_solver::{
    build_idp_initial_replicas, build_idp_valid_slots, check_idp_matrix_bounds,
    consensus_and_calibrate_idp, crossover_idp_states, estimate_idp_reference_offsets,
    evaluate_idp_total_score, polish_idp_state, predict_ncidp_shifts,
    register_and_build_con_dipeptide_systems, ConDipeptideSystem, IdpAssignResult, IdpReplicaState,
    IdpShiftOffsets, IdpSolverConfig, IDP_BG_CA_MUS, IDP_BG_CB_MUS, IDP_BG_C_MUS, IDP_BG_N_MUS,
    IDP_BG_SC_MUS, MAX_IDP_CANDS, MAX_IDP_SC_CANDS,
};
use nmr_core::rng::Rng;
use nmr_core::solver::{Matrix2D, FORBID};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

const CHUNK_STEPS: u32 = 32_768;
const SYS_FLOAT_STRIDE: usize = 32;
const SEQ_FLOAT_STRIDE: usize = 12;
const SYS_INT_STRIDE: usize = 7;
const SEQ_INT_STRIDE: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct IdpParamsGpu {
    s_len: u32,
    k_len: u32,
    replicas: u32,
    max_block: u32,

    total_steps: u32,
    step_offset: u32,
    chunk_steps: u32,
    multi_3d_dataset: u32,

    off_bg_floats: u32,
    off_sys_floats: u32,
    off_seq_floats: u32,
    off_u_mat: u32,

    off_p_mat: u32,
    off_sys_ints: u32,
    off_seq_ints: u32,
    off_slots_packed: u32,

    d_n: f32,
    d_c: f32,
    d_ca: f32,
    d_cb: f32,

    llr_floor: f32,
    pair_floor: f32,
    sigma_seq_ca: f32,
    sigma_seq_c: f32,

    w_mult: f32,
    t_start: f32,
    t_end: f32,
    score_out_mode: f32,
}

struct PackedIdpScoringInputs {
    params: IdpParamsGpu,
    ro_floats: Vec<f32>,
    ro_ints: Vec<u32>,
}

fn pack_idp_scoring_inputs(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> PackedIdpScoringInputs {
    let s_len = systems.len();
    let k_len = sequence.len();
    let preds = predict_ncidp_shifts(sequence, cfg);

    let dn = offsets.d_n as f32;
    let dc = offsets.d_c as f32;
    let dca = offsets.d_ca as f32;
    let dcb = offsets.d_cb as f32;

    let n_with_3d = systems.iter().filter(|s| s.has_3d_support()).count();
    let multi_3d_dataset = u32::from(n_with_3d * 2 >= s_len);

    let mut ro_floats =
        Vec::with_capacity(91 + s_len * SYS_FLOAT_STRIDE + k_len * SEQ_FLOAT_STRIDE);
    let mut ro_ints = Vec::with_capacity(s_len * SYS_INT_STRIDE + k_len * SEQ_INT_STRIDE);

    // 1. Pre-shifted background arrays (`20 + 20 + 20 + 19 + 12 = 91` floats)
    let off_bg_floats = ro_floats.len() as u32;
    for &m in &IDP_BG_N_MUS {
        ro_floats.push(m + dn);
    }
    for &m in &IDP_BG_C_MUS {
        ro_floats.push(m + dc);
    }
    for &m in &IDP_BG_CA_MUS {
        ro_floats.push(m + dca);
    }
    for &m in &IDP_BG_CB_MUS {
        ro_floats.push(m + dcb);
    }
    for &m in &IDP_BG_SC_MUS {
        ro_floats.push(m + dcb);
    }
    debug_assert_eq!(ro_floats.len() as u32 - off_bg_floats, 91);

    // 2. Per-system floats (`SYS_FLOAT_STRIDE = 38`) and ints (`SYS_INT_STRIDE = 8`)
    let off_sys_floats = ro_floats.len() as u32;
    let off_sys_ints = ro_ints.len() as u32;
    for sys in systems {
        let base_f = ro_floats.len();
        ro_floats.push(sys.n_curr as f32);
        ro_floats.push(sys.c_prev as f32);

        let push_padded = |buf: &mut Vec<f32>, src: &[f64], cap: usize| {
            let n = src.len().min(cap);
            for &v in &src[..n] {
                buf.push(v as f32);
            }
            for _ in n..cap {
                buf.push(0.0);
            }
            n as u32
        };

        let n_ca_curr = push_padded(&mut ro_floats, &sys.ca_curr, MAX_IDP_CANDS);
        let n_ca_prev = push_padded(&mut ro_floats, &sys.ca_prev, MAX_IDP_CANDS);
        let n_cb_prev = push_padded(&mut ro_floats, &sys.cb_prev, MAX_IDP_CANDS);
        let n_c_curr = push_padded(&mut ro_floats, &sys.c_curr, MAX_IDP_CANDS);
        let n_sc_prev = push_padded(&mut ro_floats, &sys.sidechain_c_prev, MAX_IDP_SC_CANDS);
        debug_assert_eq!(ro_floats.len() - base_f, SYS_FLOAT_STRIDE);

        let flags =
            u32::from(sys.has_cbcacon_evidence) | (u32::from(sys.prev_gly_opposite_phase) << 1);
        let n_3d_ch = u32::from(n_ca_curr > 0)
            + u32::from(n_ca_prev > 0)
            + u32::from(n_cb_prev > 0)
            + u32::from(n_c_curr > 0)
            + u32::from(n_sc_prev > 0);
        ro_ints.extend_from_slice(&[
            n_ca_curr, n_ca_prev, n_cb_prev, n_c_curr, n_sc_prev, flags, n_3d_ch,
        ]);
    }

    // 3. Per-sequence residue predictions (`SEQ_FLOAT_STRIDE = 12`, `SEQ_INT_STRIDE = 2`)
    let off_seq_floats = ro_floats.len() as u32;
    let off_seq_ints = ro_ints.len() as u32;
    for p in &preds {
        let base_q = ro_floats.len();
        ro_floats.push((p.n_ppm as f32) + dn);
        ro_floats.push(p.sigma_n as f32);
        ro_floats.push((p.c_ppm as f32) + dc);
        ro_floats.push(p.sigma_c as f32);
        ro_floats.push((p.ca_ppm as f32) + dca);
        ro_floats.push(p.sigma_ca as f32);
        ro_floats.push((p.cb_ppm.unwrap_or(0.0) as f32) + dcb);
        ro_floats.push(p.sigma_cb as f32);
        let n_sc = p.sidechain_c.len().min(4);
        for i in 0..4 {
            if i < n_sc {
                ro_floats.push((p.sidechain_c[i] as f32) + dcb);
            } else {
                ro_floats.push(0.0);
            }
        }
        debug_assert_eq!(ro_floats.len() - base_q, SEQ_FLOAT_STRIDE);
        ro_ints.push(u32::from(p.cb_ppm.is_some()));
        ro_ints.push(n_sc as u32);
    }

    let params = IdpParamsGpu {
        s_len: s_len as u32,
        k_len: k_len as u32,
        replicas: 0,
        max_block: 16,
        total_steps: 0,
        step_offset: 0,
        chunk_steps: CHUNK_STEPS,
        multi_3d_dataset,
        off_bg_floats,
        off_sys_floats,
        off_seq_floats,
        off_u_mat: 0,
        off_p_mat: 0,
        off_sys_ints,
        off_seq_ints,
        off_slots_packed: 0,
        d_n: dn,
        d_c: dc,
        d_ca: dca,
        d_cb: dcb,
        llr_floor: cfg.llr_floor as f32,
        pair_floor: cfg.pair_floor as f32,
        sigma_seq_ca: cfg.sigma_seq_ca as f32,
        sigma_seq_c: cfg.sigma_seq_c as f32,
        w_mult: (cfg.w_mult as f32).max(0.0),
        t_start: cfg.t_start as f32,
        t_end: cfg.t_end as f32,
        score_out_mode: 1.0,
    };

    PackedIdpScoringInputs {
        params,
        ro_floats,
        ro_ints,
    }
}
/// Compiled WebGPU pipelines for `13C`-detected IDP `U`/`P` scoring and multi-replica annealing.
pub struct IdpGpuSolver<'a> {
    engine: &'a GpuEngine,
    bgl: wgpu::BindGroupLayout,
    unary_pipeline: wgpu::ComputePipeline,
    pair_pipeline: wgpu::ComputePipeline,
    anneal_256_pipeline: wgpu::ComputePipeline,
    anneal_1024_pipeline: wgpu::ComputePipeline,
    anneal_global_pipeline: wgpu::ComputePipeline,
}

impl<'a> IdpGpuSolver<'a> {
    pub fn new(engine: &'a GpuEngine) -> Self {
        let device = &engine.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("idp-scoring-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/idp_scoring.wgsl").into()),
        });

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("idp-gpu-bgl"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("idp-gpu-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let mk_pipe = |label: &str, ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            engine,
            bgl,
            unary_pipeline: mk_pipe("idp-unary-pipe", "compute_idp_unary"),
            pair_pipeline: mk_pipe("idp-pair-pipe", "compute_idp_pair"),
            anneal_256_pipeline: mk_pipe("idp-anneal-256-pipe", "anneal_idp_256"),
            anneal_1024_pipeline: mk_pipe("idp-anneal-1024-pipe", "anneal_idp_1024"),
            anneal_global_pipeline: mk_pipe("idp-anneal-global-pipe", "anneal_idp_global"),
        }
    }

    fn read_vec<T: Pod>(&self, src: &wgpu::Buffer, byte_len: u64) -> Result<Vec<T>, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (src, byte_len);
            Err(
                "synchronous WebGPU buffer readback is unsupported on wasm32; use CPU solver or async WebGPU pipeline"
                    .to_string(),
            )
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.engine.read_buffer_vec(src, byte_len)
        }
    }

    /// Compute continuous through-Proline IDP unary `U` `(S x K)` and sequential `P` `(S x S)`
    /// matrices on WebGPU, matching `idp_unary_matrix` & `idp_pair_matrix` to `< 1e-4`.
    pub fn score_matrices(
        &self,
        systems: &[ConDipeptideSystem],
        sequence: &[String],
        cfg: &IdpSolverConfig,
        offsets: &IdpShiftOffsets,
    ) -> Result<(Matrix2D, Matrix2D), String> {
        if cfg!(target_arch = "wasm32") {
            return Err(
                "synchronous WebGPU buffer readback is unsupported on wasm32; use CPU solver or async WebGPU pipeline"
                    .to_string(),
            );
        }

        let s_len = systems.len();
        let k_len = sequence.len();
        check_idp_matrix_bounds(s_len, k_len)?;
        if s_len == 0 || k_len < 2 {
            return Ok((
                Matrix2D::filled(s_len, k_len, FORBID),
                Matrix2D::filled(s_len, s_len, 0.0),
            ));
        }

        let packed = pack_idp_scoring_inputs(systems, sequence, cfg, offsets);
        let u_elems = s_len * k_len;
        let p_elems = s_len * s_len;
        let out_elems = u_elems + p_elems;
        let out_bytes = (out_elems * 4) as u64;
        let ro_f_bytes = (packed.ro_floats.len().max(1) * 4) as u64;
        let ro_i_bytes = (packed.ro_ints.len().max(1) * 4) as u64;

        let max_bind = self.engine.max_storage_binding_bytes();
        let max_req = out_bytes.max(ro_f_bytes).max(ro_i_bytes);
        if max_req > max_bind {
            return Err(format!(
                "IDP GPU scoring buffer ({max_req} B) exceeds WebGPU storage binding limit ({max_bind} B) for S={s_len}, K={k_len}"
            ));
        }

        let max_wg = self
            .engine
            .device
            .limits()
            .max_compute_workgroups_per_dimension as u64;
        let wg_u = (u_elems as u64).div_ceil(64);
        let wg_p = (p_elems as u64).div_ceil(64);
        if wg_u > max_wg || wg_p > max_wg {
            return Err(format!(
                "IDP GPU scoring workgroups ({}, {}) exceed device limit ({max_wg})",
                wg_u, wg_p
            ));
        }

        let device = &self.engine.device;
        let u_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-score-params"),
            contents: bytemuck::bytes_of(&packed.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let b_ro_f = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-score-ro-floats"),
            contents: bytemuck::cast_slice(if packed.ro_floats.is_empty() {
                &[0.0f32]
            } else {
                &packed.ro_floats
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_ro_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-score-ro-ints"),
            contents: bytemuck::cast_slice(if packed.ro_ints.is_empty() {
                &[0u32]
            } else {
                &packed.ro_ints
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_dummy_at = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-dummy-at"),
            contents: bytemuck::cast_slice(&[0i32]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_dummy_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-dummy-pos"),
            contents: bytemuck::cast_slice(&[0i32]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idp-score-out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_dummy_rng = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-dummy-rng"),
            contents: bytemuck::cast_slice(&[0u32; 4]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("idp-score-bg"),
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
                    resource: b_dummy_at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_dummy_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_dummy_rng.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idp-score-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idp-score-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.unary_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(wg_u as u32, 1, 1);

            cpass.set_pipeline(&self.pair_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(wg_p as u32, 1, 1);
        }
        self.engine.queue.submit(Some(enc.finish()));

        let out_f32: Vec<f32> = self.read_vec(&b_out, out_bytes)?;
        let mut u_mat = Matrix2D::filled(s_len, k_len, FORBID);
        for (i, &v) in out_f32[..u_elems].iter().enumerate() {
            u_mat.data[i] = v as f64;
        }
        let mut p_mat = Matrix2D::filled(s_len, s_len, 0.0);
        for (i, &v) in out_f32[u_elems..out_elems].iter().enumerate() {
            p_mat.data[i] = v as f64;
        }
        Ok((u_mat, p_mat))
    }

    /// Run full WebGPU `13C`-detected IDP assignment (`U`/`P` scoring + multi-replica GPU
    /// annealing + island crossover + calibrated `ProteinReliabilityCurve` QC).
    pub fn assign(
        &self,
        systems: &[ConDipeptideSystem],
        sequence: &[String],
        cfg: &IdpSolverConfig,
    ) -> Result<IdpAssignResult, String> {
        if cfg!(target_arch = "wasm32") {
            return Err(
                "synchronous WebGPU buffer readback is unsupported on wasm32; use CPU solver or async WebGPU pipeline"
                    .to_string(),
            );
        }

        let s_len = systems.len();
        let k_len = sequence.len();
        check_idp_matrix_bounds(s_len, k_len)?;
        let fitted_offsets = if cfg.auto_reference_offset {
            estimate_idp_reference_offsets(systems, sequence, cfg)
        } else {
            cfg.manual_offsets
        };

        if s_len == 0 || k_len < 2 {
            return Ok(consensus_and_calibrate_idp(
                systems,
                sequence,
                &Matrix2D::filled(0, 0, 0.0),
                &Matrix2D::filled(0, 0, 0.0),
                &[],
                &[],
                &[],
                fitted_offsets,
                cfg,
            ));
        }

        let (u_mat, p_mat) = self.score_matrices(systems, sequence, cfg, &fitted_offsets)?;
        let slots = build_idp_valid_slots(&u_mat);

        // Pack read-only `U` + `P` (`ro_floats`) and CSR `slots` (`ro_ints`) for GPU annealing
        let mut ro_floats: Vec<f32> = Vec::with_capacity(s_len * k_len + s_len * s_len);
        let off_u_mat = 0u32;
        ro_floats.extend(u_mat.data.iter().map(|&v| v as f32));
        let off_p_mat = ro_floats.len() as u32;
        ro_floats.extend(p_mat.data.iter().map(|&v| v as f32));

        let off_slots_packed = 0u32;
        let total_slot_vals: usize = slots.iter().map(|v| v.len()).sum();
        let mut ro_ints: Vec<u32> = Vec::with_capacity(s_len + 1 + total_slot_vals);
        let mut prefix = 0u32;
        for sl in &slots {
            ro_ints.push(prefix);
            prefix += sl.len() as u32;
        }
        ro_ints.push(prefix);
        for sl in &slots {
            ro_ints.extend_from_slice(sl);
        }

        let max_bind = self.engine.max_storage_binding_bytes();
        let ro_f_bytes = (ro_floats.len().max(1) * 4) as u64;
        let ro_i_bytes = (ro_ints.len().max(1) * 4) as u64;
        if ro_f_bytes.max(ro_i_bytes) > max_bind {
            return Err(format!(
                "IDP GPU annealing read-only buffer exceeds storage binding limit ({max_bind} B)"
            ));
        }

        let device = &self.engine.device;
        let b_ro_f = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-anneal-ro-floats"),
            contents: bytemuck::cast_slice(&ro_floats),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_ro_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idp-anneal-ro-ints"),
            contents: bytemuck::cast_slice(&ro_ints),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let max_dim = s_len.max(k_len);
        let (pipeline, wg_threads): (&wgpu::ComputePipeline, u32) = if max_dim <= 256 {
            (&self.anneal_256_pipeline, 8)
        } else if max_dim <= 1024 {
            (&self.anneal_1024_pipeline, 2)
        } else {
            (&self.anneal_global_pipeline, 8)
        };

        let n_islands = cfg.islands.max(1);
        let total_replicas = cfg.replicas.max(n_islands).max(4);
        let gens = cfg.generations.max(1);
        let total_steps = cfg
            .steps_per_system
            .saturating_mul(s_len.max(24))
            .max(20_000);
        let steps_per_gen = (total_steps / gens).max(4_000).min(u32::MAX as usize) as u32;

        let mut population: Vec<IdpReplicaState> =
            build_idp_initial_replicas(&u_mat, &p_mat, &slots, total_replicas, cfg.seed);
        let mut master_rng = Rng::new(cfg.seed ^ 0x4750_555F_4944_5031);

        let per_rep_at_bytes = (2 * k_len * 4) as u64;
        let per_rep_pos_bytes = (2 * s_len * 4) as u64;
        let max_per_rep = per_rep_at_bytes.max(per_rep_pos_bytes).max(16);
        let max_wg = self
            .engine
            .device
            .limits()
            .max_compute_workgroups_per_dimension as usize;
        let max_reps_per_batch = ((max_bind / max_per_rep) as usize)
            .min(max_wg.saturating_mul(wg_threads as usize))
            .max(1);

        for gen_idx in 0..gens {
            let gen_t_start = if gen_idx == 0 {
                cfg.t_start
            } else {
                (cfg.t_start * 0.38_f64.powi(gen_idx as i32)).max(cfg.t_end * 4.0)
            };

            let mut next_pop: Vec<IdpReplicaState> = Vec::with_capacity(total_replicas);
            let mut rep_offset = 0usize;
            while rep_offset < total_replicas {
                let batch_reps = (total_replicas - rep_offset).min(max_reps_per_batch);

                let mut at_both = vec![-1i32; 2 * batch_reps * k_len];
                let mut pos_both = vec![-1i32; 2 * batch_reps * s_len];
                let mut scores_both = vec![0.0f32; 2 * batch_reps];
                let mut rng_words = vec![0u32; 4 * batch_reps];

                for r in 0..batch_reps {
                    let (ref at_r, ref pos_r, sc_r) = population[rep_offset + r];
                    at_both[r * k_len..(r + 1) * k_len].copy_from_slice(at_r);
                    at_both[(batch_reps + r) * k_len..(batch_reps + r + 1) * k_len]
                        .copy_from_slice(at_r);
                    pos_both[r * s_len..(r + 1) * s_len].copy_from_slice(pos_r);
                    pos_both[(batch_reps + r) * s_len..(batch_reps + r + 1) * s_len]
                        .copy_from_slice(pos_r);
                    scores_both[2 * r] = sc_r as f32;
                    scores_both[2 * r + 1] = sc_r as f32;

                    for w in 0..4 {
                        let mut v = master_rng.next_u64() as u32;
                        if v == 0 {
                            v = 0x9E37_79B9u32.wrapping_add(w as u32);
                        }
                        rng_words[4 * r + w] = v;
                    }
                }

                let p_base = IdpParamsGpu {
                    s_len: s_len as u32,
                    k_len: k_len as u32,
                    replicas: batch_reps as u32,
                    max_block: 16,
                    total_steps: steps_per_gen,
                    step_offset: 0,
                    chunk_steps: CHUNK_STEPS,
                    multi_3d_dataset: 1,
                    off_bg_floats: 0,
                    off_sys_floats: 0,
                    off_seq_floats: 0,
                    off_u_mat,
                    off_p_mat,
                    off_sys_ints: 0,
                    off_seq_ints: 0,
                    off_slots_packed,
                    d_n: fitted_offsets.d_n as f32,
                    d_c: fitted_offsets.d_c as f32,
                    d_ca: fitted_offsets.d_ca as f32,
                    d_cb: fitted_offsets.d_cb as f32,
                    llr_floor: cfg.llr_floor as f32,
                    pair_floor: cfg.pair_floor as f32,
                    sigma_seq_ca: cfg.sigma_seq_ca as f32,
                    sigma_seq_c: cfg.sigma_seq_c as f32,
                    w_mult: (cfg.w_mult as f32).max(0.0),
                    t_start: gen_t_start as f32,
                    t_end: cfg.t_end as f32,
                    score_out_mode: 0.0,
                };

                let u_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("idp-anneal-params"),
                    contents: bytemuck::bytes_of(&p_base),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let b_at = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("idp-anneal-at"),
                    contents: bytemuck::cast_slice(&at_both),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let b_pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("idp-anneal-pos"),
                    contents: bytemuck::cast_slice(&pos_both),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
                let b_scores = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("idp-anneal-scores"),
                    contents: bytemuck::cast_slice(&scores_both),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let b_rng = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("idp-anneal-rng"),
                    contents: bytemuck::cast_slice(&rng_words),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("idp-anneal-bg"),
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
                            resource: b_at.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: b_pos.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: b_scores.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: b_rng.as_entire_binding(),
                        },
                    ],
                });

                let workgroups = (batch_reps as u32).div_ceil(wg_threads);
                let mut step_offset = 0u32;
                while step_offset < steps_per_gen {
                    let mut p_chunk = p_base;
                    p_chunk.step_offset = step_offset;
                    self.engine
                        .queue
                        .write_buffer(&u_params, 0, bytemuck::bytes_of(&p_chunk));

                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("idp-anneal-chunk-enc"),
                    });
                    {
                        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("idp-anneal-chunk-pass"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(pipeline);
                        cpass.set_bind_group(0, &bg, &[]);
                        cpass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    self.engine.queue.submit(Some(enc.finish()));
                    step_offset = step_offset.saturating_add(CHUNK_STEPS);
                }

                let pos_bytes = (2 * batch_reps * s_len * 4) as u64;
                let pos_out: Vec<i32> = self.read_vec(&b_pos, pos_bytes)?;
                let best_pos_slice = &pos_out[batch_reps * s_len..2 * batch_reps * s_len];

                for r in 0..batch_reps {
                    let raw_pos = &best_pos_slice[r * s_len..(r + 1) * s_len];
                    let mut at_r = vec![-1i32; k_len];
                    let mut pos_r = vec![-1i32; s_len];
                    for (s, &k_i) in raw_pos.iter().enumerate() {
                        if k_i >= 1 && (k_i as usize) < k_len && at_r[k_i as usize] < 0 {
                            at_r[k_i as usize] = s as i32;
                            pos_r[s] = k_i;
                        }
                    }
                    let exact_sc = evaluate_idp_total_score(&at_r, &u_mat, &p_mat);
                    next_pop.push((at_r, pos_r, exact_sc));
                }

                rep_offset += batch_reps;
            }

            population = next_pop;

            // Island elite polish + segment crossover between GPU annealing generations
            if gen_idx + 1 < gens && total_replicas >= 4 {
                let per_island = (total_replicas / n_islands).max(2);
                for isl in 0..n_islands {
                    let lo = isl * per_island;
                    let hi = if isl + 1 == n_islands {
                        total_replicas
                    } else {
                        ((isl + 1) * per_island).min(total_replicas)
                    };
                    if hi <= lo + 1 {
                        continue;
                    }
                    population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                    for elite in &mut population[lo..(lo + 2).min(hi)] {
                        elite.2 =
                            polish_idp_state(&mut elite.0, &mut elite.1, &u_mat, &p_mat, &slots);
                    }
                    population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                    let elite_clone = population[lo].clone();
                    let second_clone = population[lo + 1].clone();
                    let mid = lo + (hi - lo) / 2;
                    for idx in mid..hi {
                        let donor = if idx % 2 == 0 {
                            &elite_clone
                        } else {
                            &second_clone
                        };
                        population[idx] = crossover_idp_states(
                            &population[lo + (idx - mid) % (mid - lo)],
                            donor,
                            &u_mat,
                            &p_mat,
                            &slots,
                            &mut master_rng,
                        );
                    }
                }
                if n_islands >= 2 && per_island >= 3 {
                    let elites: Vec<IdpReplicaState> = (0..n_islands)
                        .map(|isl| population[isl * per_island].clone())
                        .collect();
                    for (isl, elite_st) in elites.into_iter().enumerate() {
                        let target_isl = (isl + 1) % n_islands;
                        let replace_idx =
                            (target_isl * per_island + per_island - 1).min(total_replicas - 1);
                        population[replace_idx] = elite_st;
                    }
                }
            }
        }

        let rep_positions: Vec<Vec<i32>> = population.iter().map(|st| st.1.clone()).collect();
        let rep_scores: Vec<f64> = population.iter().map(|st| st.2).collect();

        Ok(consensus_and_calibrate_idp(
            systems,
            sequence,
            &u_mat,
            &p_mat,
            &slots,
            &rep_positions,
            &rep_scores,
            fitted_offsets,
            cfg,
        ))
    }
}

/// Compute continuous through-Proline IDP `(U, P)` scoring matrices on WebGPU.
pub fn score_idp_matrices_gpu(
    engine: &GpuEngine,
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> Result<(Matrix2D, Matrix2D), String> {
    IdpGpuSolver::new(engine).score_matrices(systems, sequence, cfg, offsets)
}

/// Run continuous through-Proline `13C`-detected IDP backbone assignment on WebGPU (`T11.2`).
pub fn assign_idp_13c_gpu(
    engine: &GpuEngine,
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Result<IdpAssignResult, String> {
    IdpGpuSolver::new(engine).assign(systems, sequence, cfg)
}

/// End-to-end WebGPU convenience pipeline from picked `13C`-detected spectra (`C13PickedDataset`).
pub fn assign_idp_13c_from_picked_gpu(
    engine: &GpuEngine,
    picked: &C13PickedDataset,
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Result<(Vec<ConDipeptideSystem>, IdpAssignResult), String> {
    let systems = register_and_build_con_dipeptide_systems(picked);
    let res = assign_idp_13c_gpu(engine, &systems, sequence, cfg)?;
    Ok((systems, res))
}

impl GpuEngine {
    /// Compute continuous through-Proline IDP `(U, P)` scoring matrices on WebGPU.
    pub fn score_idp_matrices_gpu(
        &self,
        systems: &[ConDipeptideSystem],
        sequence: &[String],
        cfg: &IdpSolverConfig,
        offsets: &IdpShiftOffsets,
    ) -> Result<(Matrix2D, Matrix2D), String> {
        score_idp_matrices_gpu(self, systems, sequence, cfg, offsets)
    }

    /// Solve continuous through-Proline `13C`-detected IDP assignment on WebGPU.
    pub fn assign_idp_13c_gpu(
        &self,
        systems: &[ConDipeptideSystem],
        sequence: &[String],
        cfg: &IdpSolverConfig,
    ) -> Result<IdpAssignResult, String> {
        assign_idp_13c_gpu(self, systems, sequence, cfg)
    }
}
