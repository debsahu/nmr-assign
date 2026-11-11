//! WebGPU (`wgpu` + WGSL) compute engine for NMR scoring matrices and multi-replica annealing.
//!
//! Provenance: Original GPU compute implementation of `docs/specs/03-assignment.md` section 3,
//! executing unary typing `U(s, k)`, sequential connectivity `P(s, t)`, and massively parallel
//! Metropolis-Hastings simulated annealing replicas on Metal / Vulkan / DX12 / WebGPU.

use std::sync::mpsc;

pub mod neural_gpu;

use bytemuck::{Pod, Zeroable};
use nmr_core::priors::{Priors, ATOMS, STANDARD_RESIDUES};
use nmr_core::rng::Rng;
use nmr_core::solver::{
    consensus_from_runs, AnnealConfig, AssignResult, Matrix2D, ScoreConfig, FORBID,
};
use nmr_core::spinsystems::SpinSystem;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScoreParamsGpu {
    s_len: u32,
    k_len: u32,
    max_cands: u32,
    use_hn: u32,
    sigma_ca: f32,
    sigma_cb: f32,
    sigma_c: f32,
    range_ppm: f32,
    pair_floor: f32,
    support_bonus: f32,
    llr_floor: f32,
    gly_cb: f32,
    assign_cost: f32,
    forbid_zero_carbons: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PriorEntryGpu {
    mu: f32,
    sd: f32,
    present: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CandEntryGpu {
    val: f32,
    support: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnnealParamsGpu {
    replicas: u32,
    s_len: u32,
    k_len: u32,
    max_block: u32,
    total_steps: u32,
    step_offset: u32,
    chunk_steps: u32,
    _pad: u32,
    t_start: f32,
    t_end: f32,
    _pad2: [f32; 2],
}

/// Safely upgrade baseline `wgpu::Limits` with the hardware adapter's buffer and binding
/// capacities while preserving portable baseline workgroup SRAM (`16 KiB`) and avoiding
/// optional-feature-gated limits.
fn bounded_adapter_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let hw = adapter.limits();
    let mut lim = wgpu::Limits::default()
        .using_resolution(hw.clone())
        .using_alignment(hw.clone());
    let max_buf_cap: u64 = 2_147_483_648; // 2 GiB
    let max_bind_cap: u64 = 1_073_741_824; // 1 GiB
    lim.max_buffer_size = lim.max_buffer_size.max(hw.max_buffer_size.min(max_buf_cap));
    lim.max_storage_buffer_binding_size = lim
        .max_storage_buffer_binding_size
        .max(hw.max_storage_buffer_binding_size.min(max_bind_cap));
    lim.max_uniform_buffer_binding_size = lim
        .max_uniform_buffer_binding_size
        .max(hw.max_uniform_buffer_binding_size.min(65_536));
    lim.max_compute_workgroups_per_dimension = lim
        .max_compute_workgroups_per_dimension
        .max(hw.max_compute_workgroups_per_dimension);
    lim
}

pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend_name: String,
    unary_pipeline: wgpu::ComputePipeline,
    pair_pipeline: wgpu::ComputePipeline,
    score_bgl: wgpu::BindGroupLayout,
    anneal_pipeline: wgpu::ComputePipeline,
    anneal_1024_pipeline: wgpu::ComputePipeline,
    anneal_global_pipeline: wgpu::ComputePipeline,
    anneal_bgl: wgpu::BindGroupLayout,
}

impl GpuEngine {
    /// Initialize a standalone high-performance headless WebGPU compute context
    /// exposing bounded hardware buffer limits on native Metal / Vulkan / DX12 adapters.
    pub fn new_headless() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|e| format!("no WebGPU adapter available: {e}"))?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);
        let required_limits = bounded_adapter_limits(&adapter);

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("nmr-gpu-headless-device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                ..Default::default()
            })) {
                Ok(dq) => dq,
                Err(_) => pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("nmr-gpu-headless-device-default"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .map_err(|e| format!("failed to request WebGPU device: {e}"))?,
            };

        Ok(Self::from_device(device, queue, adapter_name, backend_name))
    }

    /// Initialize a headless WebGPU compute context with caller-specified `wgpu::Limits`
    /// (e.g. strict `wgpu::Limits::default()` to validate baseline browser WebGPU caps).
    pub fn new_headless_with_limits(required_limits: wgpu::Limits) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|e| format!("no WebGPU adapter available: {e}"))?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nmr-gpu-headless-custom-limits"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        }))
        .map_err(|e| format!("failed to request WebGPU device with custom limits: {e}"))?;

        Ok(Self::from_device(device, queue, adapter_name, backend_name))
    }

    /// Share an existing `wgpu::Device` and `wgpu::Queue` (e.g. from `eframe` GUI).
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        backend_name: String,
    ) -> Self {
        let score_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-scoring-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scoring.wgsl").into()),
        });
        let anneal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-anneal-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/anneal.wgsl").into()),
        });

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

        let score_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-score-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let score_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-score-pl"),
            bind_group_layouts: &[Some(&score_bgl)],
            immediate_size: 0,
        });

        let unary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-unary-pipeline"),
            layout: Some(&score_pl),
            module: &score_shader,
            entry_point: Some("compute_unary"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pair_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-pair-pipeline"),
            layout: Some(&score_pl),
            module: &score_shader,
            entry_point: Some("compute_pair"),
            compilation_options: Default::default(),
            cache: None,
        });

        let anneal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-anneal-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_rw(4),
                storage_rw(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let anneal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-anneal-pl"),
            bind_group_layouts: &[Some(&anneal_bgl)],
            immediate_size: 0,
        });
        let anneal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-anneal-pipeline"),
            layout: Some(&anneal_pl),
            module: &anneal_shader,
            entry_point: Some("anneal_chunk"),
            compilation_options: Default::default(),
            cache: None,
        });
        let anneal_1024_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nmr-anneal-1024-pipeline"),
                layout: Some(&anneal_pl),
                module: &anneal_shader,
                entry_point: Some("anneal_chunk_1024"),
                compilation_options: Default::default(),
                cache: None,
            });
        let anneal_global_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nmr-anneal-global-pipeline"),
                layout: Some(&anneal_pl),
                module: &anneal_shader,
                entry_point: Some("anneal_chunk_global"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            adapter_name,
            backend_name,
            unary_pipeline,
            pair_pipeline,
            score_bgl,
            anneal_pipeline,
            anneal_1024_pipeline,
            anneal_global_pipeline,
            anneal_bgl,
        }
    }

    fn max_storage_binding_bytes(&self) -> u64 {
        let lim = self.device.limits();
        lim.max_storage_buffer_binding_size.min(lim.max_buffer_size)
    }

    fn read_buffer_vec<T: Pod>(&self, src: &wgpu::Buffer, byte_len: u64) -> Result<Vec<T>, String> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-staging-read"),
            size: byte_len.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-readback-enc"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..byte_len);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| format!("GPU map channel closed: {e}"))?
            .map_err(|e| format!("GPU buffer map failed: {e}"))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("GPU mapped range failed: {e}"))?;
        let out: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// Compute `U` `(S x K)` and `P` `(S x S)` scoring matrices on the GPU.
    pub fn score_matrices_gpu(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        cfg: &ScoreConfig,
    ) -> Result<(Matrix2D, Matrix2D), String> {
        let s_len = systems.len();
        let k_len = sequence.len();
        if s_len == 0 || k_len == 0 {
            return Ok((
                Matrix2D::filled(s_len, k_len, FORBID),
                Matrix2D::filled(s_len, s_len, FORBID),
            ));
        }

        let max_cands = systems
            .iter()
            .flat_map(|ss| ss.cands.values().map(|c| c.len()))
            .max()
            .unwrap_or(1)
            .max(1);

        let u_bytes = (s_len as u64) * (k_len as u64) * 4;
        let p_bytes = (s_len as u64) * (s_len as u64) * 4;
        let cands_bytes = (s_len as u64) * 6 * (max_cands as u64) * 8;
        let roots_bytes = (s_len as u64) * 8;
        let counts_bytes = (s_len as u64) * 6 * 4;
        let seq_bytes = (k_len as u64) * 4;
        let max_req = u_bytes
            .max(p_bytes)
            .max(cands_bytes)
            .max(roots_bytes)
            .max(counts_bytes)
            .max(seq_bytes);
        let max_bind = self.max_storage_binding_bytes();
        if max_req > max_bind {
            return Err(format!(
                "GPU scoring matrix buffer ({max_req} B) exceeds device storage binding limit ({max_bind} B) for S={s_len}, K={k_len}"
            ));
        }
        let max_wg_dim = self.device.limits().max_compute_workgroups_per_dimension;
        let wg_s = (s_len as u64).div_ceil(16);
        let wg_k = (k_len as u64).div_ceil(16);
        if wg_s > (max_wg_dim as u64) || wg_k > (max_wg_dim as u64) {
            return Err(format!(
                "GPU scoring workgroup count ({}x{}) exceeds device limit ({max_wg_dim}) for S={s_len}, K={k_len}",
                wg_s,
                wg_s.max(wg_k)
            ));
        }

        let params = ScoreParamsGpu {
            s_len: s_len as u32,
            k_len: k_len as u32,
            max_cands: max_cands as u32,
            use_hn: u32::from(cfg.use_hn),
            sigma_ca: cfg.sigma_ca as f32,
            sigma_cb: cfg.sigma_cb as f32,
            sigma_c: cfg.sigma_c as f32,
            range_ppm: cfg.range_ppm as f32,
            pair_floor: cfg.pair_floor as f32,
            support_bonus: cfg.support_bonus as f32,
            llr_floor: -12.0,
            gly_cb: (0.02_f64.ln()) as f32,
            assign_cost: cfg.assign_cost as f32,
            forbid_zero_carbons: u32::from(cfg.forbid_zero_carbons),
            _pad: [0; 2],
        };

        let mut prior_table = Vec::with_capacity(20 * 5);
        for &(res, _) in &STANDARD_RESIDUES {
            for &atom in &ATOMS {
                if let Some(&(mu, sd)) = priors.table.get(res).and_then(|m| m.get(atom)) {
                    prior_table.push(PriorEntryGpu {
                        mu: mu as f32,
                        sd: sd as f32,
                        present: 1,
                        _pad: 0,
                    });
                } else {
                    prior_table.push(PriorEntryGpu {
                        mu: 0.0,
                        sd: 1.0,
                        present: 0,
                        _pad: 0,
                    });
                }
            }
        }

        let seq_res: Vec<u32> = sequence
            .iter()
            .map(|r| {
                STANDARD_RESIDUES
                    .iter()
                    .position(|&(three, _)| three == r.as_str())
                    .map(|i| i as u32)
                    .unwrap_or(255)
            })
            .collect();

        let roots_hn: Vec<[f32; 2]> = systems
            .iter()
            .map(|ss| [ss.h as f32, ss.n as f32])
            .collect();

        let total_slots = s_len * 3 * 2;
        let mut cand_counts = vec![0u32; total_slots];
        let mut cands_flat = vec![
            CandEntryGpu {
                val: 0.0,
                support: 0
            };
            total_slots * max_cands
        ];

        for (s, ss) in systems.iter().enumerate() {
            for (atom_c, atom_name) in ["CA", "CB", "C"].iter().enumerate() {
                for (off_idx, off) in [0i32, -1i32].iter().enumerate() {
                    let slot = (s * 3 + atom_c) * 2 + off_idx;
                    if let Some(list) = ss.cands.get(&(atom_name.to_string(), *off)) {
                        let cnt = list.len().min(max_cands);
                        cand_counts[slot] = cnt as u32;
                        for (c_idx, &(val, sup)) in list[..cnt].iter().enumerate() {
                            cands_flat[slot * max_cands + c_idx] = CandEntryGpu {
                                val: val as f32,
                                support: sup as u32,
                            };
                        }
                    }
                }
            }
        }

        let mk_ro = |label: &str, bytes: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let u_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("score-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let b_priors = mk_ro("priors", bytemuck::cast_slice(&prior_table));
        let b_seq = mk_ro("seq", bytemuck::cast_slice(&seq_res));
        let b_roots = mk_ro("roots", bytemuck::cast_slice(&roots_hn));
        let b_cands = mk_ro("cands", bytemuck::cast_slice(&cands_flat));
        let b_counts = mk_ro("counts", bytemuck::cast_slice(&cand_counts));

        let b_out_u = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out-u"),
            size: u_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_out_p = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out-p"),
            size: p_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("score-bg"),
            layout: &self.score_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_priors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_seq.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_roots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_cands.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_out_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_out_p.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("score-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("score-pass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bg, &[]);
            cpass.set_pipeline(&self.unary_pipeline);
            cpass.dispatch_workgroups((s_len as u32).div_ceil(16), (k_len as u32).div_ceil(16), 1);

            cpass.set_pipeline(&self.pair_pipeline);
            cpass.dispatch_workgroups((s_len as u32).div_ceil(16), (s_len as u32).div_ceil(16), 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let u_f32: Vec<f32> = self.read_buffer_vec(&b_out_u, u_bytes)?;
        let p_f32: Vec<f32> = self.read_buffer_vec(&b_out_p, p_bytes)?;

        Ok((
            Matrix2D {
                rows: s_len,
                cols: k_len,
                data: u_f32.into_iter().map(|v| v as f64).collect(),
            },
            Matrix2D {
                rows: s_len,
                cols: s_len,
                data: p_f32.into_iter().map(|v| v as f64).collect(),
            },
        ))
    }

    /// Run multi-replica Simulated Annealing on the GPU across `cfg.runs` parallel replicas.
    pub fn assign_gpu(
        &self,
        systems: &[SpinSystem],
        sequence: &[String],
        priors: &Priors,
        score_cfg: &ScoreConfig,
        cfg: &AnnealConfig,
    ) -> Result<AssignResult, String> {
        let (u, p) = self.score_matrices_gpu(systems, sequence, priors, score_cfg)?;
        self.anneal_from_matrices_gpu(&u, &p, cfg)
    }

    /// Run GPU simulated annealing directly from precomputed `(U, P)` matrices.
    ///
    /// Automatically selects:
    /// - `anneal_chunk` (`8` threads/workgroup, `256` slots in `16 KiB` workgroup SRAM) for `S, K <= 256`,
    /// - `anneal_chunk_1024` (`2` threads/workgroup, `1024` slots in `16 KiB` workgroup SRAM) for `S, K <= 1024`,
    /// - `anneal_chunk_global` (`8` threads/workgroup, storage-buffer state) for `S > 1024` or `K > 1024`.
    ///
    /// Never silently falls back to CPU execution; returns an explicit `Err` if GPU buffer limits are exceeded.
    pub fn anneal_from_matrices_gpu(
        &self,
        u: &Matrix2D,
        p: &Matrix2D,
        cfg: &AnnealConfig,
    ) -> Result<AssignResult, String> {
        let s_len = u.rows;
        let k_len = u.cols;
        if p.rows != s_len || p.cols != s_len {
            return Err(format!(
                "invalid matrix dimensions: U is {s_len}x{k_len}, P is {}x{}",
                p.rows, p.cols
            ));
        }
        let replicas = cfg.runs.max(1);
        if s_len == 0 || k_len == 0 {
            return Ok(AssignResult {
                position: vec![-1; s_len],
                confidence: vec![0.0; s_len],
                score: 0.0,
                run_scores: vec![0.0; replicas],
            });
        }

        let mut slots_packed = Vec::with_capacity(s_len + 1 + s_len * k_len / 2);
        let mut slot_indices = Vec::new();
        slots_packed.push(0u32);
        for s in 0..s_len {
            for k in 0..k_len {
                if u.get(s, k) > FORBID * 0.5 {
                    slot_indices.push(k as u32);
                }
            }
            slots_packed.push(slot_indices.len() as u32);
        }
        if slot_indices.is_empty() {
            slot_indices.push(0);
        }
        slots_packed.extend_from_slice(&slot_indices);

        let u_bytes = (s_len as u64) * (k_len as u64) * 4;
        let p_bytes = (s_len as u64) * (s_len as u64) * 4;
        let slots_bytes = (slots_packed.len() as u64) * 4;
        let per_rep_at_bytes = 2 * (k_len as u64) * 4;
        let per_rep_pos_bytes = (s_len as u64) * 4;
        let per_rep_max_bytes = per_rep_at_bytes.max(per_rep_pos_bytes).max(16);
        let max_static_req = u_bytes.max(p_bytes).max(slots_bytes).max(per_rep_max_bytes);
        let max_bind = self.max_storage_binding_bytes();
        if max_static_req > max_bind {
            return Err(format!(
                "GPU annealer buffer ({max_static_req} B) exceeds device storage binding limit ({max_bind} B) for S={s_len}, K={k_len}"
            ));
        }

        let (pipeline, wg_threads) = if s_len <= 256 && k_len <= 256 {
            (&self.anneal_pipeline, 8u32)
        } else if s_len <= 1024 && k_len <= 1024 {
            (&self.anneal_1024_pipeline, 2u32)
        } else {
            (&self.anneal_global_pipeline, 8u32)
        };
        let max_wg_dim = self.device.limits().max_compute_workgroups_per_dimension as usize;
        let max_reps_by_wg = max_wg_dim.saturating_mul(wg_threads as usize).max(1);
        let max_reps_by_bind = (max_bind / per_rep_max_bytes) as usize;
        let max_batch_replicas = max_reps_by_wg.min(max_reps_by_bind).min(replicas).max(1);

        let u_f32: Vec<f32> = u.data.iter().map(|&v| v as f32).collect();
        let p_f32: Vec<f32> = p.data.iter().map(|&v| v as f32).collect();

        let mut seeder = Rng::new(cfg.seed);
        let mut rng_init_all = Vec::with_capacity(replicas);
        for _ in 0..replicas {
            let a = seeder.next_u64();
            let b = seeder.next_u64();
            rng_init_all.push([
                (a as u32) | 1,
                (a >> 32) as u32,
                (b as u32) | 1,
                (b >> 32) as u32,
            ]);
        }

        let mk_buf = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage,
                })
        };

        let total_steps = cfg
            .steps_per_system
            .saturating_mul(s_len)
            .max(1)
            .min(u32::MAX as usize) as u32;
        let chunk_steps: u32 = 32_768;

        let u_params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anneal-params"),
            size: std::mem::size_of::<AnnealParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_u = mk_buf(
            "anneal-u",
            bytemuck::cast_slice(&u_f32),
            wgpu::BufferUsages::STORAGE,
        );
        let b_p = mk_buf(
            "anneal-p",
            bytemuck::cast_slice(&p_f32),
            wgpu::BufferUsages::STORAGE,
        );
        let b_slots = mk_buf(
            "anneal-slots",
            bytemuck::cast_slice(&slots_packed),
            wgpu::BufferUsages::STORAGE,
        );

        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let mut positions: Vec<Vec<i32>> = Vec::with_capacity(replicas);
        let mut exact_scores: Vec<f64> = Vec::with_capacity(replicas);

        let mut rep_offset = 0usize;
        while rep_offset < replicas {
            let batch_reps = (replicas - rep_offset).min(max_batch_replicas);
            let at_bytes = 2 * (batch_reps as u64) * (k_len as u64) * 4;
            // [0 .. batch_reps*k_len) = current `at`, [batch_reps*k_len .. 2*batch_reps*k_len) = `best_at`
            let init_at_both = vec![-1i32; 2 * batch_reps * k_len];
            let init_pos = vec![-1i32; batch_reps * s_len];
            let init_scores = vec![[0.0f32, 0.0f32]; batch_reps];
            let rng_slice = &rng_init_all[rep_offset..rep_offset + batch_reps];

            let b_at = mk_buf("anneal-at-both", bytemuck::cast_slice(&init_at_both), rw);
            let b_pos = mk_buf("anneal-pos", bytemuck::cast_slice(&init_pos), rw);
            let b_scores = mk_buf("anneal-scores", bytemuck::cast_slice(&init_scores), rw);
            let b_rng = mk_buf("anneal-rng", bytemuck::cast_slice(rng_slice), rw);

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("anneal-bg"),
                layout: &self.anneal_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_p.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_slots.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: b_at.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b_pos.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: b_scores.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: b_rng.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (batch_reps as u32).div_ceil(wg_threads);
            let mut step_offset = 0u32;
            while step_offset < total_steps {
                let p_val = AnnealParamsGpu {
                    replicas: batch_reps as u32,
                    s_len: s_len as u32,
                    k_len: k_len as u32,
                    max_block: cfg.max_block as u32,
                    total_steps,
                    step_offset,
                    chunk_steps,
                    _pad: 0,
                    t_start: cfg.t_start as f32,
                    t_end: cfg.t_end as f32,
                    _pad2: [0.0; 2],
                };
                self.queue
                    .write_buffer(&u_params, 0, bytemuck::bytes_of(&p_val));
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("anneal-chunk-enc"),
                    });
                {
                    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("anneal-chunk-pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(pipeline);
                    cpass.set_bind_group(0, &bg, &[]);
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }
                self.queue.submit(Some(enc.finish()));
                step_offset = step_offset.saturating_add(chunk_steps);
            }

            let at_both_out: Vec<i32> = self.read_buffer_vec(&b_at, at_bytes)?;
            let best_at_all = &at_both_out[batch_reps * k_len..2 * batch_reps * k_len];
            let scores_out: Vec<[f32; 2]> =
                self.read_buffer_vec(&b_scores, (batch_reps * 8) as u64)?;

            // Reconstruct `pos` per replica and re-evaluate exact f64 total score
            for r in 0..batch_reps {
                let at_slice = &best_at_all[r * k_len..(r + 1) * k_len];
                let mut pos = vec![-1i32; s_len];
                let mut sc = 0.0f64;
                for (k, &s) in at_slice.iter().enumerate() {
                    if s >= 0 && (s as usize) < s_len {
                        pos[s as usize] = k as i32;
                        sc += u.get(s as usize, k);
                    }
                }
                for e in 0..k_len.saturating_sub(1) {
                    let a = at_slice[e];
                    let b = at_slice[e + 1];
                    if a >= 0 && b >= 0 {
                        sc += p.get(a as usize, b as usize);
                    }
                }
                let _gpu_best = scores_out[r][1];
                positions.push(pos);
                exact_scores.push(sc);
            }

            rep_offset += batch_reps;
        }

        Ok(consensus_from_runs(&positions, exact_scores))
    }
}
