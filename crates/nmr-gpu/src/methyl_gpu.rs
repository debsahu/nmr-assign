//! WebGPU (`wgpu` + WGSL) compute solver for supramolecular `13C`-methyl TROSY (`ILVMAT`)
//! structure-based NOE contact-graph objective evaluation and massively parallel multi-replica
//! Metropolis–Hastings simulated annealing (`docs/specs/13-methyl-trosy.md` §3).
//!
//! Provenance: Clean-room GPU compute implementation from peer-reviewed biomolecular NMR literature:
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125,
//!   10120–10121 & 10420–10428 — `1H-13C` methyl-TROSY HMQC in high-MW proteins).
//! - Gans, P. et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962 — stereospecific `pro-R` / `pro-S`
//!   `Leu` / `Val` precursor labeling).
//! - Ruschak, A. M. & Kay, L. E. (*J. Biomol. NMR* 2010, 46, 75–87).
//! - Xu, Y., Lin, Z., Ho, C. & Yang, D. (*J. Am. Chem. Soc.* 2009, 131, 9480–9481 — automated
//!   methyl resonance assignment from NOE networks).
//! - Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G. (*J. Magn. Reson.* 2012, 214, 103–110).
//! - Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J.
//!   & Kay, L. E. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533 — probabilistic methyl NOE contact-graph
//!   matching).

#[cfg(not(target_arch = "wasm32"))]
use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::methyl_solver::{
    build_geminal_move_table, build_initial_replica_states, consensus_and_calibrate_methyl,
    evaluate_methyl_raw_score,
};
use nmr_core::methyl_solver::{MethylAnnealConfig, MethylAssignResult, MethylContactProblem};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::rng::Rng;
#[cfg(not(target_arch = "wasm32"))]
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum MCMC steps per single WebGPU command-buffer submission (`<= 4,096` to prevent WDDM TDR).
pub const MAX_CHUNK_STEPS: u32 = 4_096;

/// Adaptive per-submit step chunk size (`<= 4,096`) scaled inversely with problem dimension so
/// single dispatches stay well below OS GPU watchdog timeouts (`2 s` Windows TDR) even above 1,024 sites.
#[inline]
pub const fn chunk_steps_for_dim(max_dim: usize) -> u32 {
    if max_dim <= 255 {
        4_096
    } else if max_dim <= 1024 {
        2_048
    } else {
        1_024
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MethylParamsGpu {
    replicas: u32,
    s_len: u32,
    m_len: u32,
    n_obs_edges: u32,

    n_short_edges: u32,
    total_steps: u32,
    step_offset: u32,
    chunk_steps: u32,

    off_u_mat: u32,
    off_d_mat: u32,
    off_gem_obs_mat: u32,
    off_obs_edge_floats: u32,

    off_obs_adj_floats: u32,
    off_short_edge_lam: u32,
    off_short_adj_lam: u32,
    n_gem_pairs: u32,

    off_slots_offs: u32,
    off_slots_vals: u32,
    off_gem_site: u32,
    off_obs_explained: u32,

    off_obs_edge_pairs: u32,
    off_obs_adj_offs: u32,
    off_obs_adj_peers: u32,
    off_short_edge_pairs: u32,

    off_short_adj_offs: u32,
    off_short_adj_peers: u32,
    off_cut_adj_offs: u32,
    off_cut_adj_peers: u32,

    off_gem_pairs: u32,
    off_gem_nbr_offs: u32,
    off_gem_nbr_vals: u32,
    _pad_u1: u32,

    d_cutoff: f32,
    sigma_d: f32,
    flat_bottom_d: f32,
    norm_inv: f32,

    t_start: f32,
    t_end: f32,
    _pad_f0: f32,
    _pad_f1: f32,
}

#[cfg(not(target_arch = "wasm32"))]
struct PackedMethylProblemGpu {
    base_params: MethylParamsGpu,
    ro_floats: Vec<f32>,
    ro_ints: Vec<u32>,
}

#[cfg(not(target_arch = "wasm32"))]
fn pack_methyl_problem(prob: &MethylContactProblem) -> PackedMethylProblemGpu {
    let mut ro_floats = Vec::new();
    let mut ro_ints = Vec::new();

    // 1. Pack read-only float arrays
    let off_u_mat = ro_floats.len() as u32;
    ro_floats.extend_from_slice(&prob.u_mat);

    let off_d_mat = ro_floats.len() as u32;
    ro_floats.extend_from_slice(&prob.d_mat);

    let off_gem_obs_mat = ro_floats.len() as u32;
    ro_floats.extend_from_slice(&prob.gem_obs_mat);

    let off_obs_edge_floats = ro_floats.len() as u32;
    for e in &prob.obs_edges {
        ro_floats.push(e.d_hat);
        ro_floats.push(e.weight);
    }

    let off_obs_adj_floats = ro_floats.len() as u32;
    for e in &prob.obs_adj_entries {
        ro_floats.push(e.d_hat);
        ro_floats.push(e.weight);
    }

    let off_short_edge_lam = ro_floats.len() as u32;
    for se in &prob.struct_short_edges {
        ro_floats.push(se.lambda_miss);
    }

    let off_short_adj_lam = ro_floats.len() as u32;
    for se in &prob.struct_short_entries {
        ro_floats.push(se.lambda_miss);
    }

    while ro_floats.len() < 4 {
        ro_floats.push(0.0);
    }

    // 2. Pack read-only integer / CSR arrays
    let off_slots_offs = ro_ints.len() as u32;
    let mut cur_slot_off = 0u32;
    ro_ints.push(0);
    for sl in &prob.slots {
        cur_slot_off += sl.len() as u32;
        ro_ints.push(cur_slot_off);
    }

    let off_slots_vals = ro_ints.len() as u32;
    for sl in &prob.slots {
        ro_ints.extend_from_slice(sl);
    }

    let off_gem_site = ro_ints.len() as u32;
    for &g in &prob.gem_site {
        ro_ints.push(g as u32);
    }

    let off_obs_explained = ro_ints.len() as u32;
    let total_bits = prob.obs_explained_mask.len();
    let n_words = total_bits.div_ceil(32);
    for w in 0..n_words {
        let mut bits = 0u32;
        let base = w * 32;
        let end = (base + 32).min(total_bits);
        for (bit_pos, &flag) in prob.obs_explained_mask[base..end].iter().enumerate() {
            if flag != 0 {
                bits |= 1u32 << bit_pos;
            }
        }
        ro_ints.push(bits);
    }

    let off_obs_edge_pairs = ro_ints.len() as u32;
    for e in &prob.obs_edges {
        ro_ints.push(e.s1);
        ro_ints.push(e.s2);
    }

    let off_obs_adj_offs = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&prob.obs_adj_offsets);

    let off_obs_adj_peers = ro_ints.len() as u32;
    for e in &prob.obs_adj_entries {
        ro_ints.push(e.peer_s);
    }

    let off_short_edge_pairs = ro_ints.len() as u32;
    for se in &prob.struct_short_edges {
        ro_ints.push(se.m1);
        ro_ints.push(se.m2);
    }

    let off_short_adj_offs = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&prob.struct_short_offsets);

    let off_short_adj_peers = ro_ints.len() as u32;
    for se in &prob.struct_short_entries {
        ro_ints.push(se.peer_m);
    }

    let off_cut_adj_offs = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&prob.struct_cut_offsets);

    let off_cut_adj_peers = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&prob.struct_cut_peers);

    let gem_tbl = build_geminal_move_table(prob);
    let n_gem_pairs = gem_tbl.pairs.len() as u32;
    let off_gem_pairs = ro_ints.len() as u32;
    for &[m1, m2] in &gem_tbl.pairs {
        ro_ints.push(m1);
        ro_ints.push(m2);
    }
    let off_gem_nbr_offs = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&gem_tbl.nbr_offsets);
    let off_gem_nbr_vals = ro_ints.len() as u32;
    ro_ints.extend_from_slice(&gem_tbl.nbr_indices);

    while ro_ints.len() < 4 {
        ro_ints.push(0);
    }

    let base_params = MethylParamsGpu {
        replicas: 0,
        s_len: prob.s_len as u32,
        m_len: prob.m_len as u32,
        n_obs_edges: prob.obs_edges.len() as u32,
        n_short_edges: prob.struct_short_edges.len() as u32,
        total_steps: 0,
        step_offset: 0,
        chunk_steps: 0,
        off_u_mat,
        off_d_mat,
        off_gem_obs_mat,
        off_obs_edge_floats,
        off_obs_adj_floats,
        off_short_edge_lam,
        off_short_adj_lam,
        n_gem_pairs,
        off_slots_offs,
        off_slots_vals,
        off_gem_site,
        off_obs_explained,
        off_obs_edge_pairs,
        off_obs_adj_offs,
        off_obs_adj_peers,
        off_short_edge_pairs,
        off_short_adj_offs,
        off_short_adj_peers,
        off_cut_adj_offs,
        off_cut_adj_peers,
        off_gem_pairs,
        off_gem_nbr_offs,
        off_gem_nbr_vals,
        _pad_u1: 0,
        d_cutoff: prob.d_cutoff,
        sigma_d: prob.sigma_d,
        flat_bottom_d: prob.flat_bottom_d,
        norm_inv: prob.norm_inv as f32,
        t_start: 28.0,
        t_end: 0.15,
        _pad_f0: 0.0,
        _pad_f1: 0.0,
    };

    PackedMethylProblemGpu {
        base_params,
        ro_floats,
        ro_ints,
    }
}

/// Reusable WebGPU compute pipelines for methyl NOE contact-graph objective evaluation and
/// multi-replica simulated annealing (`16 KiB` workgroup SRAM + strict `<= 6` storage bindings).
pub struct MethylGpuSolver<'a> {
    engine: &'a GpuEngine,
    bgl: wgpu::BindGroupLayout,
    eval_pipeline: wgpu::ComputePipeline,
    anneal_256_pipeline: wgpu::ComputePipeline,
    anneal_1024_pipeline: wgpu::ComputePipeline,
    anneal_global_pipeline: wgpu::ComputePipeline,
}

impl<'a> MethylGpuSolver<'a> {
    /// Compile the `methyl_anneal.wgsl` compute pipelines on `engine`.
    pub fn new(engine: &'a GpuEngine) -> Self {
        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nmr-methyl-anneal-wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/methyl_anneal.wgsl").into()),
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
                label: Some("nmr-methyl-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw(3),
                    storage_rw(4),
                    storage_rw(5),
                    storage_rw(6),
                ],
            });

        let pl = engine
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nmr-methyl-pl"),
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
            engine,
            bgl,
            eval_pipeline: make_pipe("nmr-methyl-eval-pipe", "eval_methyl_objective_batch"),
            anneal_256_pipeline: make_pipe("nmr-methyl-anneal-256-pipe", "anneal_methyl_256"),
            anneal_1024_pipeline: make_pipe("nmr-methyl-anneal-1024-pipe", "anneal_methyl_1024"),
            anneal_global_pipeline: make_pipe(
                "nmr-methyl-anneal-global-pipe",
                "anneal_methyl_global",
            ),
        }
    }

    /// Evaluate exact per-root normalized objective $E(\pi)$ on WebGPU across a batch of
    /// assignments `positions` (matching `nmr_core::methyl_solver::evaluate_methyl_objective` to `< 1e-4`).
    pub fn evaluate_objectives(
        &self,
        prob: &MethylContactProblem,
        positions: &[Vec<i32>],
    ) -> Result<Vec<f64>, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                &self.engine,
                &self.bgl,
                &self.eval_pipeline,
                &self.anneal_256_pipeline,
                &self.anneal_1024_pipeline,
                &self.anneal_global_pipeline,
                prob,
                positions,
            );
            Err(
                "synchronous WebGPU methyl objective readback is unsupported on wasm32; use async readback or CPU solver"
                    .to_string(),
            )
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if positions.is_empty() {
                return Ok(Vec::new());
            }
            for p in positions {
                if p.len() != prob.s_len {
                    return Err(format!(
                        "methyl assignment vector length {} != problem s_len {}",
                        p.len(),
                        prob.s_len
                    ));
                }
            }
            if prob.s_len == 0 || prob.m_len == 0 {
                return Ok(vec![0.0; positions.len()]);
            }

            let packed = pack_methyl_problem(prob);
            let max_bind = self.engine.max_storage_binding_bytes();
            let ro_f_bytes = (packed.ro_floats.len() * 4) as u64;
            let ro_i_bytes = (packed.ro_ints.len() * 4) as u64;
            if ro_f_bytes > max_bind || ro_i_bytes > max_bind {
                return Err(format!(
                    "methyl problem read-only buffer ({} B) exceeds adapter limit ({} B)",
                    ro_f_bytes.max(ro_i_bytes),
                    max_bind
                ));
            }

            let b_floats =
                self.engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("methyl-eval-ro-floats"),
                        contents: bytemuck::cast_slice(&packed.ro_floats),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let b_ints = self
                .engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("methyl-eval-ro-ints"),
                    contents: bytemuck::cast_slice(&packed.ro_ints),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let per_rep_bytes = ((prob.m_len.max(prob.s_len) * 4) as u64).max(16);
            let max_wg = self
                .engine
                .device
                .limits()
                .max_compute_workgroups_per_dimension as usize;
            let max_batch = ((max_bind / per_rep_bytes) as usize)
                .min(max_wg.saturating_mul(64))
                .max(1);

            let mut out = Vec::with_capacity(positions.len());
            let mut offset = 0usize;
            while offset < positions.len() {
                let batch_n = (positions.len() - offset).min(max_batch);
                let mut p_val = packed.base_params;
                p_val.replicas = batch_n as u32;

                let mut pos_flat = Vec::with_capacity(batch_n * prob.s_len);
                for p in &positions[offset..offset + batch_n] {
                    pos_flat.extend_from_slice(p);
                }
                if pos_flat.is_empty() {
                    pos_flat.push(-1);
                }

                let at_dummy = vec![-1i32; (batch_n * prob.m_len).max(1)];
                let scores_init = vec![[0.0f32; 2]; batch_n.max(1)];
                let rng_dummy = vec![[1u32; 4]; batch_n.max(1)];

                let u_params =
                    self.engine
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("methyl-eval-params"),
                            contents: bytemuck::bytes_of(&p_val),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                let b_at =
                    self.engine
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("methyl-eval-at"),
                            contents: bytemuck::cast_slice(&at_dummy),
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                let b_pos =
                    self.engine
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("methyl-eval-pos"),
                            contents: bytemuck::cast_slice(&pos_flat),
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                let b_scores =
                    self.engine
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("methyl-eval-scores"),
                            contents: bytemuck::cast_slice(&scores_init),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                let b_rng =
                    self.engine
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("methyl-eval-rng"),
                            contents: bytemuck::cast_slice(&rng_dummy),
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                let bg = self
                    .engine
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("methyl-eval-bg"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_params.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_floats.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: b_ints.as_entire_binding(),
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

                let mut enc =
                    self.engine
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("methyl-eval-enc"),
                        });
                {
                    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("methyl-eval-pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.eval_pipeline);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups((batch_n as u32).div_ceil(64), 1, 1);
                }
                self.engine.queue.submit(Some(enc.finish()));

                let scores_gpu: Vec<[f32; 2]> = self
                    .engine
                    .read_buffer_vec(&b_scores, (batch_n * 8) as u64)?;
                for sc in scores_gpu {
                    out.push(sc[0] as f64);
                }
                offset += batch_n;
            }

            Ok(out)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    fn run_replica_batches(
        &self,
        prob: &MethylContactProblem,
        replicas: usize,
        total_steps: u32,
        t_start: f64,
        t_end: f64,
        seed: u64,
        return_terminal_cur: bool,
    ) -> Result<Vec<Vec<i32>>, String> {
        let packed = pack_methyl_problem(prob);
        let max_bind = self.engine.max_storage_binding_bytes();
        let ro_f_bytes = (packed.ro_floats.len() * 4) as u64;
        let ro_i_bytes = (packed.ro_ints.len() * 4) as u64;
        if ro_f_bytes > max_bind || ro_i_bytes > max_bind {
            return Err(format!(
                "methyl problem read-only buffer ({} B) exceeds adapter limit ({} B)",
                ro_f_bytes.max(ro_i_bytes),
                max_bind
            ));
        }

        let b_floats = self
            .engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("methyl-anneal-ro-floats"),
                contents: bytemuck::cast_slice(&packed.ro_floats),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_ints = self
            .engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("methyl-anneal-ro-ints"),
                contents: bytemuck::cast_slice(&packed.ro_ints),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let max_dim = prob.s_len.max(prob.m_len);
        let chunk_steps = chunk_steps_for_dim(max_dim);
        debug_assert!(chunk_steps <= MAX_CHUNK_STEPS);

        let (pipeline, wg_threads) = if max_dim <= 255 {
            (&self.anneal_256_pipeline, 8u32)
        } else if max_dim <= 1024 {
            (&self.anneal_1024_pipeline, 2u32)
        } else {
            (&self.anneal_global_pipeline, 8u32)
        };

        let init_states = build_initial_replica_states(prob, replicas, seed);
        let mut seeder = Rng::new(seed ^ 0x4750_555F_4D45_5448);
        let mut all_rng_seeds: Vec<[u32; 4]> = Vec::with_capacity(replicas);
        for _ in 0..replicas {
            let a = seeder.next_u64();
            let b = seeder.next_u64();
            let mut st = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
            if (st[0] | st[1] | st[2] | st[3]) == 0 {
                st[0] = 0x9E37_79B9;
            }
            all_rng_seeds.push(st);
        }

        let per_rep_state_bytes = ((2 * max_dim * 4) as u64).max(16);
        let max_wg = self
            .engine
            .device
            .limits()
            .max_compute_workgroups_per_dimension as usize;
        let max_batch_reps = ((max_bind / per_rep_state_bytes) as usize)
            .min(max_wg.saturating_mul(wg_threads as usize))
            .max(1);

        let mut all_positions: Vec<Vec<i32>> = Vec::with_capacity(replicas);
        let mut rep_offset = 0usize;
        while rep_offset < replicas {
            let batch_reps = (replicas - rep_offset).min(max_batch_reps);
            let mut at_both_init = Vec::with_capacity(2 * batch_reps * prob.m_len);
            let mut pos_both_init = Vec::with_capacity(2 * batch_reps * prob.s_len);
            let mut scores_init = Vec::with_capacity(batch_reps);

            for st in &init_states[rep_offset..rep_offset + batch_reps] {
                at_both_init.extend_from_slice(&st.0);
                pos_both_init.extend_from_slice(&st.1);
                let sc_f32 = st.2 as f32;
                scores_init.push([sc_f32, sc_f32]);
            }
            for st in &init_states[rep_offset..rep_offset + batch_reps] {
                at_both_init.extend_from_slice(&st.0);
                pos_both_init.extend_from_slice(&st.1);
            }

            let rng_slice = &all_rng_seeds[rep_offset..rep_offset + batch_reps];

            let mut p_init = packed.base_params;
            p_init.replicas = batch_reps as u32;
            p_init.total_steps = total_steps;
            p_init.chunk_steps = chunk_steps;
            p_init.t_start = t_start as f32;
            p_init.t_end = t_end as f32;

            let u_params =
                self.engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("methyl-anneal-params"),
                        contents: bytemuck::bytes_of(&p_init),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
            let b_at = self
                .engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("methyl-anneal-at-both"),
                    contents: bytemuck::cast_slice(&at_both_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let b_pos = self
                .engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("methyl-anneal-pos-both"),
                    contents: bytemuck::cast_slice(&pos_both_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let b_scores =
                self.engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("methyl-anneal-scores"),
                        contents: bytemuck::cast_slice(&scores_init),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    });
            let b_rng = self
                .engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("methyl-anneal-rng"),
                    contents: bytemuck::cast_slice(rng_slice),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let bg = self
                .engine
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("methyl-anneal-bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: b_floats.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b_ints.as_entire_binding(),
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
            while step_offset < total_steps {
                let mut p_chunk = p_init;
                p_chunk.step_offset = step_offset;
                self.engine
                    .queue
                    .write_buffer(&u_params, 0, bytemuck::bytes_of(&p_chunk));

                let mut enc =
                    self.engine
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("methyl-anneal-chunk-enc"),
                        });
                {
                    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("methyl-anneal-chunk-pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(pipeline);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }
                self.engine.queue.submit(Some(enc.finish()));
                step_offset = step_offset.saturating_add(chunk_steps);
            }

            let pos_bytes = (2 * batch_reps * prob.s_len * 4) as u64;
            let pos_both_out: Vec<i32> = self.engine.read_buffer_vec(&b_pos, pos_bytes)?;
            let slice_out = if return_terminal_cur {
                &pos_both_out[0..batch_reps * prob.s_len]
            } else {
                &pos_both_out[batch_reps * prob.s_len..2 * batch_reps * prob.s_len]
            };

            for r in 0..batch_reps {
                all_positions.push(slice_out[r * prob.s_len..(r + 1) * prob.s_len].to_vec());
            }

            rep_offset += batch_reps;
        }

        Ok(all_positions)
    }

    /// Run fixed-temperature WebGPU Metropolis–Hastings MCMC and return terminal chain states
    /// `pos_cur` (for stationary Boltzmann detailed-balance verification).
    pub fn sample_mcmc_terminal_states(
        &self,
        prob: &MethylContactProblem,
        temperature: f64,
        steps: u32,
        replicas: usize,
        seed: u64,
    ) -> Result<Vec<Vec<i32>>, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (prob, temperature, steps, replicas, seed);
            Err(
                "synchronous WebGPU methyl MCMC readback is unsupported on wasm32; use async readback or CPU solver"
                    .to_string(),
            )
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let reps = replicas.max(1);
            if prob.s_len == 0 || prob.m_len == 0 {
                return Ok(vec![vec![-1; prob.s_len]; reps]);
            }
            self.run_replica_batches(
                prob,
                reps,
                steps.max(1),
                temperature,
                temperature,
                seed,
                true,
            )
        }
    }

    /// Run massively parallel WebGPU multi-replica simulated annealing + Boltzmann consensus
    /// and local-swap confidence calibration on `prob`.
    pub fn assign(
        &self,
        prob: &MethylContactProblem,
        cfg: &MethylAnnealConfig,
    ) -> Result<MethylAssignResult, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (prob, cfg);
            Err(
                "synchronous WebGPU methyl assignment readback is unsupported on wasm32; use async readback or CPU solver"
                    .to_string(),
            )
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let replicas = cfg.replicas.max(1);
            if prob.s_len == 0 || prob.m_len == 0 {
                return Ok(MethylAssignResult {
                    position: vec![-1; prob.s_len],
                    confidence: vec![0.0; prob.s_len],
                    class_posteriors: prob.class_posteriors.clone(),
                    score: 0.0,
                    raw_score: 0.0,
                    run_scores: vec![0.0; replicas],
                });
            }

            let min_floor = cfg.steps_per_root.saturating_mul(85);
            let total_steps = cfg
                .steps_per_root
                .saturating_mul(prob.s_len)
                .max(min_floor)
                .clamp(1, u32::MAX as usize) as u32;

            let all_positions = self.run_replica_batches(
                prob,
                replicas,
                total_steps,
                cfg.t_start,
                cfg.t_end,
                cfg.seed,
                false,
            )?;
            let all_raw_scores: Vec<f64> = all_positions
                .iter()
                .map(|p_vec| evaluate_methyl_raw_score(prob, p_vec))
                .collect();

            Ok(consensus_and_calibrate_methyl(
                prob,
                &all_positions,
                &all_raw_scores,
            ))
        }
    }
}

/// Evaluate exact per-root normalized objective $E(\pi)$ on WebGPU across `positions`.
pub fn evaluate_methyl_objectives_gpu(
    engine: &GpuEngine,
    prob: &MethylContactProblem,
    positions: &[Vec<i32>],
) -> Result<Vec<f64>, String> {
    MethylGpuSolver::new(engine).evaluate_objectives(prob, positions)
}

/// Run WebGPU multi-replica simulated annealing + Boltzmann confidence calibration on `prob`.
pub fn assign_methyl_gpu(
    engine: &GpuEngine,
    prob: &MethylContactProblem,
    cfg: &MethylAnnealConfig,
) -> Result<MethylAssignResult, String> {
    MethylGpuSolver::new(engine).assign(prob, cfg)
}

impl GpuEngine {
    /// Evaluate exact per-root normalized methyl contact-graph objectives $E(\pi)$ on WebGPU.
    pub fn evaluate_methyl_objectives_gpu(
        &self,
        prob: &MethylContactProblem,
        positions: &[Vec<i32>],
    ) -> Result<Vec<f64>, String> {
        evaluate_methyl_objectives_gpu(self, prob, positions)
    }

    /// Solve methyl NOE contact-graph assignment on WebGPU (`anneal_methyl_256` / `1024` / `global`).
    pub fn assign_methyl_gpu(
        &self,
        prob: &MethylContactProblem,
        cfg: &MethylAnnealConfig,
    ) -> Result<MethylAssignResult, String> {
        assign_methyl_gpu(self, prob, cfg)
    }
}
