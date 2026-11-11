//! WebGPU (`wgpu` + WGSL) 3D separable convolutional neural peak picker (`GpuNeuralPicker`).
//!
//! Provenance: Clean-room original GPU compute implementation designed from public signal-processing
//! and deep-learning literature, matching `nmr_core::neural_picker::SeparableConvNet3D` to `< 1e-4`:
//! - Li, D.-W., Hansen, A. L., Yuan, C., Bruschweiler-Li, L. & Brüschweiler, R. (2021),
//!   *J. Biomol. NMR* 75, 79–89 — methodological reference for convolutional NMR peak detection
//!   and sub-voxel coordinate regression.
//! - Klukowski, P., Augoff, M., Zięba, M., Drwal, M., Gonczarek, A. & Walczak, M. J. (2018),
//!   *Bioinformatics* 34(15), 2590–2597 — methodological reference for indirect-ridge suppression.
//! - Chollet, F. (2017), *CVPR*, 1251–1258 & Yu, F. & Koltun, V. (2016), *ICLR* — multi-scale
//!   depthwise/separable dilated residual 3D convolutions.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use nmr_core::peaks::Peak;
use nmr_core::picker::estimate_noise_mad;
use nmr_core::spectrum::Spectrum3D;
use wgpu::util::DeviceExt;

use crate::GpuEngine;

pub const CONV3D_GPU_HIDDEN_CH: usize = 16;
pub const CONV3D_GPU_OUT_CH: usize = 6;
pub const CONV3D_GPU_NUM_WEIGHTS: usize = 1030;

const NORM_SIGMA_DIV: f32 = 10.0;
const NORM_CLAMP: f32 = 8.0;
/// Maximum voxels per single GPU slab so `n_vox * 16 * 4 <= 64 MiB` (well within WebGPU's 128 MiB default binding limit).
const MAX_SLAB_VOXELS: usize = 1_048_576;
const MAX_NMS_PEAKS_PER_SLAB: u32 = 32_768;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Conv3DParamsGpu {
    nx: u32,
    ny: u32,
    nz: u32,
    allow_neg: u32,
    prob_thresh: f32,
    min_snr_norm: f32,
    valid_z0: u32,
    valid_z1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NmsHeaderGpu {
    count: u32,
    max_peaks: u32,
    z_global_offset: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuRawPeak3D {
    pub ix: u32,
    pub iy: u32,
    pub iz: u32,
    pub sign: i32,
    pub prob: f32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub amp_norm: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

#[derive(Clone, Debug)]
pub struct GpuNeuralPickerConfig {
    pub prob_threshold: f32,
    pub min_snr_norm: f32,
    pub allow_negative: bool,
    pub strip_tol_h: f64,
    pub strip_tol_n: f64,
}

impl Default for GpuNeuralPickerConfig {
    fn default() -> Self {
        Self {
            prob_threshold: 0.45,
            min_snr_norm: 0.08,
            allow_negative: false,
            strip_tol_h: 0.035,
            strip_tol_n: 0.35,
        }
    }
}

fn median_f32_gpu(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Haar wavelet first-difference thermal noise estimator matching `nmr_core::neural_picker::estimate_thermal_noise_haar`.
pub fn estimate_thermal_noise_haar_gpu(data: &[f32], nx: usize) -> f32 {
    if data.len() < 4 || nx < 2 {
        return estimate_noise_mad(data).max(1e-6);
    }
    let n_rows = data.len() / nx;
    let row_stride = (n_rows / 512).max(1);
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut diffs = Vec::with_capacity((n_rows / row_stride + 1) * (nx - 1));
    for r in (0..n_rows).step_by(row_stride) {
        let row = &data[r * nx..(r + 1) * nx];
        let x_stride = (nx / 128).max(1);
        let mut ix = 0usize;
        while ix + 1 < nx {
            diffs.push((row[ix + 1] - row[ix]).abs() * inv_sqrt2);
            ix += x_stride;
        }
    }
    if diffs.len() < 8 {
        return estimate_noise_mad(data).max(1e-6);
    }
    let mad0 = (1.4826 * median_f32_gpu(&mut diffs)).max(1e-6);
    let mut bg: Vec<f32> = diffs.into_iter().filter(|&d| d <= 2.5 * mad0).collect();
    if bg.len() < 8 {
        return mad0;
    }
    (1.4826 * 1.014 * median_f32_gpu(&mut bg)).max(1e-6)
}

/// Normalize 3D spectrum intensities identically to `nmr_core::neural_picker::normalize_spectrum_data_with_nx`.
pub fn normalize_spectrum_3d_for_gpu(data: &[f32], nx: usize) -> (Vec<f32>, f32, f32) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let stride = (data.len() / 32_768).max(1);
    let mut sample: Vec<f32> = data.iter().step_by(stride).copied().collect();
    let med = median_f32_gpu(&mut sample);
    let noise = estimate_thermal_noise_haar_gpu(data, nx);
    let inv = 1.0 / (NORM_SIGMA_DIV * noise);
    let norm = data
        .iter()
        .map(|&v| ((v - med) * inv).clamp(-NORM_CLAMP, NORM_CLAMP))
        .collect();
    (norm, med, noise)
}

pub struct GpuNeuralPicker {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend_name: String,
    bgl: wgpu::BindGroupLayout,
    p_stem_x: wgpu::ComputePipeline,
    p_stem_y: wgpu::ComputePipeline,
    p_stem_z_pw1: wgpu::ComputePipeline,
    p_block2_x: wgpu::ComputePipeline,
    p_block2_y: wgpu::ComputePipeline,
    p_block2_z_pw2_head: wgpu::ComputePipeline,
    p_nms_3d: wgpu::ComputePipeline,
    weights_buf: wgpu::Buffer,
    weights: Vec<f32>,
}

impl GpuNeuralPicker {
    /// Create a WebGPU 3D neural peak picker sharing an existing `GpuEngine` device and queue.
    pub fn from_engine(engine: &GpuEngine, weights: &[f32]) -> Result<Self, String> {
        Self::from_device(
            engine.device.clone(),
            engine.queue.clone(),
            engine.adapter_name.clone(),
            engine.backend_name.clone(),
            weights,
        )
    }

    /// Create a standalone headless WebGPU 3D neural peak picker with explicit `1,030` `f32` weights.
    pub fn new_headless(weights: &[f32]) -> Result<Self, String> {
        let engine = GpuEngine::new_headless()?;
        Self::from_engine(&engine, weights)
    }

    /// Create a WebGPU 3D neural peak picker loaded with converged clean-room pretrained weights.
    pub fn pretrained(engine: &GpuEngine) -> Result<Self, String> {
        Self::from_engine(engine, &PRETRAINED_WEIGHTS_3D_GPU)
    }

    /// Create a standalone headless WebGPU 3D neural peak picker with converged pretrained weights.
    pub fn pretrained_headless() -> Result<Self, String> {
        let engine = GpuEngine::new_headless()?;
        Self::pretrained(&engine)
    }

    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        backend_name: String,
        weights: &[f32],
    ) -> Result<Self, String> {
        if weights.len() != CONV3D_GPU_NUM_WEIGHTS {
            return Err(format!(
                "expected {CONV3D_GPU_NUM_WEIGHTS} 3D CNN weights, got {}",
                weights.len()
            ));
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-conv3d-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/conv3d.wgsl").into()),
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-conv3d-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_rw(3),
                storage_rw(4),
                storage_rw(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-conv3d-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let make_pipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let p_stem_x = make_pipe("conv3d-stem-x", "stem_pass_x");
        let p_stem_y = make_pipe("conv3d-stem-y", "stem_pass_y");
        let p_stem_z_pw1 = make_pipe("conv3d-stem-z-pw1", "stem_pass_z_pw1");
        let p_block2_x = make_pipe("conv3d-block2-x", "block2_pass_x");
        let p_block2_y = make_pipe("conv3d-block2-y", "block2_pass_y");
        let p_block2_z_pw2_head = make_pipe("conv3d-block2-z-pw2-head", "block2_pass_z_pw2_head");
        let p_nms_3d = make_pipe("conv3d-nms-3d", "nms_3d_pass");

        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv3d-weights-buf"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend_name,
            bgl,
            p_stem_x,
            p_stem_y,
            p_stem_z_pw1,
            p_block2_x,
            p_block2_y,
            p_block2_z_pw2_head,
            p_nms_3d,
            weights_buf,
            weights: weights.to_vec(),
        })
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        if weights.len() != CONV3D_GPU_NUM_WEIGHTS {
            return Err(format!(
                "expected {CONV3D_GPU_NUM_WEIGHTS} weights, got {}",
                weights.len()
            ));
        }
        self.queue
            .write_buffer(&self.weights_buf, 0, bytemuck::cast_slice(weights));
        self.weights.copy_from_slice(weights);
        Ok(())
    }

    fn read_buffer_bytes(
        &self,
        src: &wgpu::Buffer,
        offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>, String> {
        if byte_len == 0 {
            return Ok(Vec::new());
        }
        let aligned_len = (byte_len + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv3d-staging-read"),
            size: aligned_len.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv3d-readback-enc"),
            });
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, aligned_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..aligned_len);
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
        let out = mapped[..byte_len as usize].to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// Run a single Z-slab (or full cube) on WebGPU, optionally executing `nms_3d_pass`.
    #[allow(clippy::too_many_arguments)]
    fn execute_slab_gpu(
        &self,
        slab_input: &[f32],
        nx: usize,
        ny: usize,
        slab_nz: usize,
        valid_z0: u32,
        valid_z1: u32,
        z_global_offset: u32,
        prob_thresh: f32,
        min_snr_norm: f32,
        allow_neg: bool,
        read_dense_out: bool,
        run_nms: bool,
    ) -> Result<(Vec<f32>, Vec<GpuRawPeak3D>), String> {
        let n_vox = nx * ny * slab_nz;
        if slab_input.len() != n_vox {
            return Err(format!(
                "slab_input length {} != nx*ny*slab_nz ({n_vox})",
                slab_input.len()
            ));
        }

        let params = Conv3DParamsGpu {
            nx: nx as u32,
            ny: ny as u32,
            nz: slab_nz as u32,
            allow_neg: u32::from(allow_neg),
            prob_thresh,
            min_snr_norm,
            valid_z0,
            valid_z1,
        };
        let u_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv3d-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let b_in = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv3d-input"),
                contents: bytemuck::cast_slice(slab_input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let hidden_bytes = (n_vox * CONV3D_GPU_HIDDEN_CH * 4).max(16) as u64;
        let out_bytes = (n_vox * CONV3D_GPU_OUT_CH * 4).max(16) as u64;
        let make_rw = |label: &str, size: u64, copy_src: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if copy_src {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let buf_a = make_rw("conv3d-buf-a", hidden_bytes, false);
        let buf_b = make_rw("conv3d-buf-b", hidden_bytes, false);
        let buf_skip = make_rw("conv3d-buf-skip", hidden_bytes, false);
        let buf_out = make_rw("conv3d-buf-out", out_bytes, read_dense_out);

        let max_peaks = if run_nms { MAX_NMS_PEAKS_PER_SLAB } else { 1 };
        let header = NmsHeaderGpu {
            count: 0,
            max_peaks,
            z_global_offset,
            pad0: 0,
        };
        let nms_total_bytes = std::mem::size_of::<NmsHeaderGpu>() as u64
            + (max_peaks as u64) * (std::mem::size_of::<GpuRawPeak3D>() as u64);
        let b_nms = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv3d-nms-out"),
            size: nms_total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&b_nms, 0, bytemuck::bytes_of(&header));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv3d-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_skip.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: b_nms.as_entire_binding(),
                },
            ],
        });

        let wg_x = (nx as u32).div_ceil(8);
        let wg_y = (ny as u32).div_ceil(8);
        let wg_z = (slab_nz as u32).div_ceil(4);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv3d-forward-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv3d-forward-pass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bg, &[]);

            for pipe in [
                &self.p_stem_x,
                &self.p_stem_y,
                &self.p_stem_z_pw1,
                &self.p_block2_x,
                &self.p_block2_y,
                &self.p_block2_z_pw2_head,
            ] {
                cpass.set_pipeline(pipe);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }
            if run_nms {
                cpass.set_pipeline(&self.p_nms_3d);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }
        }
        self.queue.submit(Some(encoder.finish()));

        let dense = if read_dense_out {
            let valid_planes = (valid_z1 - valid_z0) as usize;
            let plane_floats = nx * ny * CONV3D_GPU_OUT_CH;
            let byte_off = (valid_z0 as usize * plane_floats * 4) as u64;
            let byte_len = (valid_planes * plane_floats * 4) as u64;
            let raw_bytes = self.read_buffer_bytes(&buf_out, byte_off, byte_len)?;
            bytemuck::cast_slice::<u8, f32>(&raw_bytes).to_vec()
        } else {
            Vec::new()
        };

        let raw_peaks = if run_nms {
            let hdr_bytes =
                self.read_buffer_bytes(&b_nms, 0, std::mem::size_of::<NmsHeaderGpu>() as u64)?;
            let hdr: NmsHeaderGpu = *bytemuck::from_bytes(&hdr_bytes);
            let cnt = hdr.count.min(max_peaks) as usize;
            if cnt > 0 {
                let pk_off = std::mem::size_of::<NmsHeaderGpu>() as u64;
                let pk_bytes = (cnt * std::mem::size_of::<GpuRawPeak3D>()) as u64;
                let raw_pk_bytes = self.read_buffer_bytes(&b_nms, pk_off, pk_bytes)?;
                bytemuck::cast_slice::<u8, GpuRawPeak3D>(&raw_pk_bytes).to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Ok((dense, raw_peaks))
    }

    /// Execute the 3D separable CNN forward pass on WebGPU (`[nz * ny * nx * 6]`), automatically
    /// tiling large 3D volumes along Z with exact 6-plane receptive-field halos when needed.
    pub fn forward_3d_gpu(
        &self,
        input_norm: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Vec<f32>, String> {
        let plane = nx * ny;
        if input_norm.len() != plane * nz {
            return Err(format!(
                "input_norm length {} != nx*ny*nz ({})",
                input_norm.len(),
                plane * nz
            ));
        }
        if nx == 0 || ny == 0 || nz == 0 {
            return Ok(Vec::new());
        }

        if plane * nz <= MAX_SLAB_VOXELS {
            let (dense, _) = self.execute_slab_gpu(
                input_norm, nx, ny, nz, 0, nz as u32, 0, 0.5, 0.0, true, true, false,
            )?;
            return Ok(dense);
        }

        let halo = 6usize;
        let max_slab_z = (MAX_SLAB_VOXELS / plane.max(1)).max(halo * 2 + 4);
        let core_step_z = max_slab_z - 2 * halo;
        let mut full_out = Vec::with_capacity(plane * nz * CONV3D_GPU_OUT_CH);

        let mut z0 = 0usize;
        while z0 < nz {
            let z1 = (z0 + core_step_z).min(nz);
            let sub_z0 = z0.saturating_sub(halo);
            let sub_z1 = (z1 + halo).min(nz);
            let slab_nz = sub_z1 - sub_z0;
            let slab_slice = &input_norm[sub_z0 * plane..sub_z1 * plane];
            let (slab_dense, _) = self.execute_slab_gpu(
                slab_slice,
                nx,
                ny,
                slab_nz,
                (z0 - sub_z0) as u32,
                (z1 - sub_z0) as u32,
                sub_z0 as u32,
                0.5,
                0.0,
                true,
                true,
                false,
            )?;
            full_out.extend_from_slice(&slab_dense);
            z0 = z1;
        }

        Ok(full_out)
    }

    /// Pick 3D triple-resonance peaks (`HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`, `HNCO`, `HNCACO`)
    /// entirely on the GPU (separable 3D conv + residual + GELU + 3x3x3 GPU NMS + sub-voxel offset).
    pub fn pick_triple_3d_gpu(
        &self,
        spec: &Spectrum3D,
        experiment: &str,
        hsqc_roots: &[Peak],
        cfg: &GpuNeuralPickerConfig,
    ) -> Result<Vec<Peak>, String> {
        let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
        if nx < 3 || ny < 3 || nz < 3 {
            return Ok(Vec::new());
        }
        let exp_name = if experiment.is_empty() {
            spec.exp.as_str()
        } else {
            experiment
        };
        let allow_neg = cfg.allow_negative || exp_name == "HNCACB";
        let y_is_n = spec.axis_y.label.contains('N')
            || (spec.axis_y.max_ppm() > 95.0
                && spec.axis_y.min_ppm() < 145.0
                && !spec.axis_z.label.contains('N'));

        let (norm_in, med, noise) = normalize_spectrum_3d_for_gpu(&spec.data, nx);
        let eff_prob_thresh = cfg.prob_threshold;
        let plane = nx * ny;

        let mut all_raw_peaks: Vec<GpuRawPeak3D> = Vec::new();
        if plane * nz <= MAX_SLAB_VOXELS {
            let (_, raws) = self.execute_slab_gpu(
                &norm_in,
                nx,
                ny,
                nz,
                1,
                (nz - 1) as u32,
                0,
                eff_prob_thresh,
                cfg.min_snr_norm,
                allow_neg,
                false,
                true,
            )?;
            all_raw_peaks = raws;
        } else {
            // 7-plane Z-halo (6 conv receptive field + 1 NMS neighbor plane) guarantees exact 3x3x3 NMS
            let halo = 7usize;
            let max_slab_z = (MAX_SLAB_VOXELS / plane.max(1)).max(halo * 2 + 4);
            let core_step_z = max_slab_z - 2 * halo;
            let mut z0 = 1usize;
            let z_limit = nz - 1;
            while z0 < z_limit {
                let z1 = (z0 + core_step_z).min(z_limit);
                let sub_z0 = z0.saturating_sub(halo);
                let sub_z1 = (z1 + halo).min(nz);
                let slab_nz = sub_z1 - sub_z0;
                let slab_slice = &norm_in[sub_z0 * plane..sub_z1 * plane];
                let (_, raws) = self.execute_slab_gpu(
                    slab_slice,
                    nx,
                    ny,
                    slab_nz,
                    (z0 - sub_z0) as u32,
                    (z1 - sub_z0) as u32,
                    sub_z0 as u32,
                    eff_prob_thresh,
                    cfg.min_snr_norm,
                    allow_neg,
                    false,
                    true,
                )?;
                all_raw_peaks.extend(raws);
                z0 = z1;
            }
        }

        // Deterministic sort matching CPU scan order (iz, iy, ix, ch)
        all_raw_peaks.sort_by_key(|a| (a.iz, a.iy, a.ix, -a.sign));

        let mut peaks = Vec::with_capacity(all_raw_peaks.len());
        for rp in all_raw_peaks {
            let (ix, iy, iz) = (rp.ix as usize, rp.iy as usize, rp.iz as usize);
            if !hsqc_roots.is_empty() {
                let h_grid = spec.axis_x.index_to_ppm(ix as f64);
                let n_grid = if y_is_n {
                    spec.axis_y.index_to_ppm(iy as f64)
                } else {
                    spec.axis_z.index_to_ppm(iz as f64)
                };
                if !hsqc_roots.iter().any(|r| {
                    (r.h - h_grid).abs() <= cfg.strip_tol_h
                        && (r.n - n_grid).abs() <= cfg.strip_tol_n
                }) {
                    continue;
                }
            }

            let h = spec.axis_x.index_to_ppm((ix as f64) + (rp.dx as f64));
            let (n, c) = if y_is_n {
                (
                    spec.axis_y.index_to_ppm((iy as f64) + (rp.dy as f64)),
                    spec.axis_z.index_to_ppm((iz as f64) + (rp.dz as f64)),
                )
            } else {
                (
                    spec.axis_z.index_to_ppm((iz as f64) + (rp.dz as f64)),
                    spec.axis_y.index_to_ppm((iy as f64) + (rp.dy as f64)),
                )
            };

            let cnn_height = (rp.amp_norm * NORM_SIGMA_DIV * noise) as f64;
            let raw_height = ((spec.get(ix, iy, iz) - med).abs()) as f64;
            let height = (0.85 * cnn_height + 0.15 * raw_height).max(1e-6);

            let mut vol = 0.0f64;
            for wz in -1isize..=1 {
                for wy in -1isize..=1 {
                    for wx in -1isize..=1 {
                        let sv = spec.get(
                            (ix as isize + wx) as usize,
                            (iy as isize + wy) as usize,
                            (iz as isize + wz) as usize,
                        ) - med;
                        if (rp.sign > 0 && sv > 0.0) || (rp.sign < 0 && sv < 0.0) {
                            vol += (sv.abs()) as f64;
                        }
                    }
                }
            }

            peaks.push(Peak {
                exp: exp_name.to_string(),
                h,
                n,
                c: Some(c),
                sign: rp.sign,
                height: Some(height),
                volume: Some(vol.max(height)),
                h2: None,
            });
        }

        Ok(peaks)
    }
}

/// Converged clean-room pretrained 3D CNN weights (`1,030` `f32` parameters, shared 1:1 with `nmr_core::neural_picker`).
#[allow(clippy::excessive_precision)]
pub const PRETRAINED_WEIGHTS_3D_GPU: [f32; CONV3D_GPU_NUM_WEIGHTS] = [
    -0.0976156_f32,
    0.5865893_f32,
    0.8234336_f32,
    0.2408835_f32,
    -0.2319170_f32,
    -0.2722685_f32,
    0.6327428_f32,
    0.0787241_f32,
    0.3480960_f32,
    -0.3548083_f32,
    -0.5219154_f32,
    0.0518068_f32,
    0.7113128_f32,
    -0.1180177_f32,
    -0.3362499_f32,
    -0.3052263_f32,
    0.6335716_f32,
    0.2855617_f32,
    0.4766995_f32,
    -0.3466614_f32,
    -0.3927867_f32,
    -0.5134649_f32,
    0.3798009_f32,
    0.8324884_f32,
    0.1745117_f32,
    -0.2205904_f32,
    0.4338102_f32,
    0.9855352_f32,
    0.4009320_f32,
    -0.1842854_f32,
    -0.1493649_f32,
    0.4415535_f32,
    0.8436815_f32,
    0.2740238_f32,
    -0.0615668_f32,
    0.1511941_f32,
    0.4939968_f32,
    0.8286315_f32,
    0.2815271_f32,
    0.0447646_f32,
    0.2338325_f32,
    0.3033748_f32,
    0.4804524_f32,
    0.2842765_f32,
    0.1340881_f32,
    0.1129794_f32,
    -0.0064097_f32,
    0.6412024_f32,
    0.2260302_f32,
    0.0010504_f32,
    -0.0082433_f32,
    -0.0678198_f32,
    0.7106975_f32,
    0.0693463_f32,
    -0.0517540_f32,
    -0.3900950_f32,
    0.5274752_f32,
    -0.0000433_f32,
    0.6177756_f32,
    -0.3107450_f32,
    -0.2382209_f32,
    -0.2905028_f32,
    0.1979266_f32,
    0.5679480_f32,
    0.1051923_f32,
    -0.4432760_f32,
    0.4213407_f32,
    0.4768032_f32,
    0.3656780_f32,
    -0.2262964_f32,
    -0.2029778_f32,
    0.5645167_f32,
    -0.1208719_f32,
    0.6428103_f32,
    -0.1205166_f32,
    -0.0461320_f32,
    -0.1601584_f32,
    0.7572544_f32,
    0.1403813_f32,
    -0.0153732_f32,
    -0.3437098_f32,
    0.1340311_f32,
    0.8786735_f32,
    0.2098457_f32,
    -0.2447941_f32,
    -0.2215833_f32,
    0.6158163_f32,
    0.2351247_f32,
    0.5362384_f32,
    -0.1726981_f32,
    -0.1634198_f32,
    0.2523359_f32,
    0.7854782_f32,
    0.4353702_f32,
    -0.1381640_f32,
    -0.1386765_f32,
    0.5758839_f32,
    0.3000023_f32,
    0.5665652_f32,
    -0.0761982_f32,
    -0.1201167_f32,
    0.3217549_f32,
    0.8152915_f32,
    0.1893322_f32,
    -0.2406658_f32,
    -0.1896923_f32,
    -0.8737372_f32,
    -0.1065906_f32,
    0.8102896_f32,
    0.0703356_f32,
    -0.1620375_f32,
    0.2782011_f32,
    0.8909982_f32,
    0.2599207_f32,
    -0.2571162_f32,
    0.0016676_f32,
    0.3060810_f32,
    0.8802185_f32,
    0.4253758_f32,
    0.0465740_f32,
    0.3486545_f32,
    0.4238868_f32,
    0.4213827_f32,
    -0.1044520_f32,
    -0.3038613_f32,
    -0.3072591_f32,
    -0.0746676_f32,
    0.6192414_f32,
    0.0558429_f32,
    -0.0516612_f32,
    0.1684330_f32,
    0.1895531_f32,
    0.5129414_f32,
    0.1392495_f32,
    0.1197830_f32,
    -0.1272034_f32,
    0.7765920_f32,
    0.0053432_f32,
    0.5507826_f32,
    -0.1876212_f32,
    0.0622656_f32,
    0.4971855_f32,
    0.1337866_f32,
    0.3911787_f32,
    -0.2688237_f32,
    -0.3814436_f32,
    -0.1303153_f32,
    0.4369792_f32,
    0.7616761_f32,
    0.1425049_f32,
    -0.1626107_f32,
    0.5860379_f32,
    0.1345080_f32,
    0.6731585_f32,
    -0.0675683_f32,
    -0.1192611_f32,
    0.0700530_f32,
    0.7379736_f32,
    0.1812315_f32,
    -0.2239110_f32,
    -0.2289306_f32,
    0.0667208_f32,
    0.9463195_f32,
    0.4623603_f32,
    -0.2334162_f32,
    -0.2841312_f32,
    0.6144496_f32,
    0.1782960_f32,
    0.4873763_f32,
    0.0348867_f32,
    0.0065047_f32,
    0.5175835_f32,
    0.6589286_f32,
    0.7973474_f32,
    -0.1398593_f32,
    -0.2602083_f32,
    -0.0386915_f32,
    0.4439891_f32,
    0.1089773_f32,
    -0.4623799_f32,
    -0.2631274_f32,
    0.3415903_f32,
    0.7526602_f32,
    0.0694701_f32,
    -0.2168191_f32,
    -0.1620105_f32,
    0.4868574_f32,
    1.0822198_f32,
    0.4511348_f32,
    -0.1894724_f32,
    -0.0686353_f32,
    -0.9254017_f32,
    -0.3919806_f32,
    0.6672061_f32,
    -0.0583248_f32,
    -0.2564615_f32,
    0.0127624_f32,
    0.5679355_f32,
    0.1263388_f32,
    -0.3435050_f32,
    0.2367501_f32,
    0.3234884_f32,
    0.4205286_f32,
    0.3900122_f32,
    0.2552914_f32,
    0.2912583_f32,
    0.2267634_f32,
    0.4627742_f32,
    -0.0386288_f32,
    0.1283334_f32,
    -0.3029073_f32,
    -0.0437350_f32,
    0.7341929_f32,
    -0.1067808_f32,
    -0.2245992_f32,
    -0.0739648_f32,
    0.0167732_f32,
    0.5925903_f32,
    0.1866255_f32,
    0.0552297_f32,
    -0.2081706_f32,
    0.5078833_f32,
    0.1198902_f32,
    0.5085612_f32,
    -0.1255057_f32,
    -0.2017633_f32,
    0.6756870_f32,
    0.1850348_f32,
    0.6249605_f32,
    -0.1539433_f32,
    -0.1262822_f32,
    -0.2154289_f32,
    -0.5459570_f32,
    0.3874536_f32,
    0.1423837_f32,
    -0.3235873_f32,
    -0.0605569_f32,
    0.6992477_f32,
    -0.1169560_f32,
    -0.2522258_f32,
    0.3295299_f32,
    -0.3683593_f32,
    0.1404348_f32,
    0.1555665_f32,
    -0.1742305_f32,
    0.6005692_f32,
    -0.0992569_f32,
    0.2098382_f32,
    0.0954537_f32,
    -0.0306030_f32,
    -0.2254187_f32,
    -0.0183872_f32,
    0.2550589_f32,
    -0.2957007_f32,
    -0.0606671_f32,
    0.5447669_f32,
    -0.9808239_f32,
    0.1788936_f32,
    -0.4684070_f32,
    -0.3323285_f32,
    -0.0977209_f32,
    0.7490460_f32,
    -0.3596503_f32,
    -0.4497676_f32,
    0.2939382_f32,
    -0.5171903_f32,
    -0.4038871_f32,
    0.5890323_f32,
    0.1678278_f32,
    -0.1270953_f32,
    -0.0669885_f32,
    -0.2784708_f32,
    0.3654064_f32,
    0.6405777_f32,
    0.3173735_f32,
    0.3852752_f32,
    -0.3027392_f32,
    -0.3359242_f32,
    -0.4470619_f32,
    0.0698214_f32,
    -0.1243540_f32,
    0.1074519_f32,
    0.3111192_f32,
    -0.2466942_f32,
    -0.1169045_f32,
    0.1794460_f32,
    0.2515127_f32,
    0.3134565_f32,
    0.4485229_f32,
    -0.8794997_f32,
    0.2827999_f32,
    -0.0194973_f32,
    0.0856703_f32,
    0.5510954_f32,
    -0.1493742_f32,
    0.2606383_f32,
    0.4092523_f32,
    0.0689814_f32,
    0.1591952_f32,
    -0.0210568_f32,
    -0.4427639_f32,
    -0.0302895_f32,
    0.0433559_f32,
    0.0388531_f32,
    0.2595844_f32,
    -0.2038149_f32,
    0.9508591_f32,
    0.0999254_f32,
    0.4772177_f32,
    0.3338193_f32,
    0.1679785_f32,
    0.0742715_f32,
    0.3292704_f32,
    0.6394325_f32,
    0.3857028_f32,
    -0.4370784_f32,
    0.0670566_f32,
    -0.2150397_f32,
    0.1300691_f32,
    0.3379788_f32,
    -0.1362541_f32,
    -0.4149824_f32,
    -0.8558764_f32,
    -0.4167784_f32,
    -0.6275173_f32,
    -0.0568384_f32,
    0.4070663_f32,
    0.1882032_f32,
    -0.0056704_f32,
    -0.0432768_f32,
    -0.3740177_f32,
    0.1730565_f32,
    0.2446702_f32,
    0.2627119_f32,
    -0.2891131_f32,
    -0.8225421_f32,
    0.5479049_f32,
    0.0044434_f32,
    0.4826599_f32,
    0.8075740_f32,
    -0.6497988_f32,
    -0.3197050_f32,
    0.1088915_f32,
    0.1556042_f32,
    0.0067669_f32,
    0.6280903_f32,
    0.3950166_f32,
    -0.0274145_f32,
    0.3299229_f32,
    -0.0922509_f32,
    0.1307300_f32,
    0.2303542_f32,
    0.6658716_f32,
    0.2861496_f32,
    0.2928735_f32,
    -0.0015334_f32,
    0.5825924_f32,
    0.6548495_f32,
    -0.4409434_f32,
    0.2050192_f32,
    0.1693599_f32,
    -0.0897386_f32,
    0.2834574_f32,
    -0.3524189_f32,
    -0.2532374_f32,
    -0.5332743_f32,
    0.5541626_f32,
    0.2111126_f32,
    0.1639944_f32,
    -0.0504616_f32,
    0.2012189_f32,
    0.2970550_f32,
    0.8520983_f32,
    -0.4731808_f32,
    -0.2962967_f32,
    0.1936362_f32,
    -0.0864200_f32,
    0.2232141_f32,
    -0.0017433_f32,
    -0.1040500_f32,
    -0.3220062_f32,
    0.1639892_f32,
    0.1717254_f32,
    0.2288501_f32,
    0.0568804_f32,
    0.0806665_f32,
    0.0706644_f32,
    0.0936192_f32,
    -0.5810406_f32,
    0.3389675_f32,
    0.6937408_f32,
    -0.1861841_f32,
    -0.1289704_f32,
    -0.0061798_f32,
    -0.1203282_f32,
    -0.0997128_f32,
    0.3518953_f32,
    -0.2008733_f32,
    -0.2065027_f32,
    0.4065139_f32,
    0.3805785_f32,
    -0.0245618_f32,
    0.3401976_f32,
    0.2946540_f32,
    -0.4101917_f32,
    0.5233698_f32,
    -0.1983267_f32,
    -0.0206596_f32,
    0.3854667_f32,
    -0.2092257_f32,
    0.1022226_f32,
    0.1209470_f32,
    0.1731847_f32,
    -0.2183768_f32,
    -0.0077827_f32,
    0.2340731_f32,
    0.1125831_f32,
    0.1515527_f32,
    -0.0629936_f32,
    -0.0611408_f32,
    -0.2723598_f32,
    -0.4103260_f32,
    -0.6316774_f32,
    -0.2907023_f32,
    -0.4240564_f32,
    -0.2788835_f32,
    -0.0724356_f32,
    0.0131696_f32,
    -0.1515085_f32,
    0.1750832_f32,
    -0.0209173_f32,
    0.0671740_f32,
    -0.6108940_f32,
    0.1488806_f32,
    -0.0303117_f32,
    -0.0237769_f32,
    -0.3380297_f32,
    0.1213495_f32,
    1.0656165_f32,
    -0.5583879_f32,
    -0.0406271_f32,
    -0.2514542_f32,
    -0.0514522_f32,
    0.1313706_f32,
    -0.0971740_f32,
    -0.0539927_f32,
    -0.0893971_f32,
    -0.2700705_f32,
    -0.0160460_f32,
    -0.1110049_f32,
    0.3340858_f32,
    0.0117367_f32,
    -0.0104371_f32,
    0.1511707_f32,
    -1.3839163_f32,
    0.1453659_f32,
    0.0415906_f32,
    0.0681526_f32,
    0.0577984_f32,
    -0.1572631_f32,
    0.0018150_f32,
    -0.1501347_f32,
    0.4057209_f32,
    0.0725810_f32,
    0.0487302_f32,
    -0.1111329_f32,
    0.0595756_f32,
    0.1608834_f32,
    0.0848768_f32,
    -0.4654472_f32,
    -0.0619625_f32,
    0.9370192_f32,
    0.0030927_f32,
    -0.0252680_f32,
    0.3937866_f32,
    -0.0868804_f32,
    -0.0888709_f32,
    0.0342504_f32,
    0.2865414_f32,
    0.4626017_f32,
    -0.0664817_f32,
    -0.2966340_f32,
    -0.5574431_f32,
    -0.1566837_f32,
    0.5652665_f32,
    -0.8642877_f32,
    0.1088534_f32,
    -0.8720545_f32,
    -0.3298033_f32,
    -0.0193032_f32,
    -0.2575068_f32,
    0.0013762_f32,
    -0.2579965_f32,
    0.0500300_f32,
    -0.0332654_f32,
    -0.3436339_f32,
    0.0961687_f32,
    0.3507096_f32,
    0.1452819_f32,
    0.4173290_f32,
    0.2038103_f32,
    0.0421715_f32,
    0.0803799_f32,
    -0.4728044_f32,
    0.2023124_f32,
    0.4054636_f32,
    0.3039753_f32,
    0.1126815_f32,
    0.4041777_f32,
    0.4002688_f32,
    0.2833641_f32,
    0.1227788_f32,
    0.1290662_f32,
    0.8913703_f32,
    -0.0819709_f32,
    0.3683386_f32,
    1.1088820_f32,
    0.1723389_f32,
    0.1418617_f32,
    0.8664063_f32,
    -0.0926033_f32,
    0.2200187_f32,
    0.9676481_f32,
    -0.0868889_f32,
    -0.0287097_f32,
    1.0583351_f32,
    0.6951579_f32,
    -0.2709865_f32,
    0.3163892_f32,
    -1.2430345_f32,
    -0.0445386_f32,
    1.0022820_f32,
    -0.0017845_f32,
    0.5329359_f32,
    1.1495661_f32,
    0.6924145_f32,
    -0.0907409_f32,
    0.8943833_f32,
    -0.2234907_f32,
    0.2558363_f32,
    0.8782756_f32,
    -0.2773184_f32,
    0.1716597_f32,
    1.1096650_f32,
    -0.0169593_f32,
    0.1608080_f32,
    1.0827579_f32,
    0.2286979_f32,
    0.3645399_f32,
    0.8390363_f32,
    0.1331834_f32,
    -0.0719916_f32,
    0.9933913_f32,
    -0.0044286_f32,
    0.0583530_f32,
    0.9611533_f32,
    -0.1725870_f32,
    0.1505117_f32,
    1.0512412_f32,
    0.0215237_f32,
    0.1491656_f32,
    1.0468957_f32,
    -0.1678671_f32,
    0.0649399_f32,
    1.0711814_f32,
    0.2711818_f32,
    -0.1086336_f32,
    0.8506011_f32,
    0.1267437_f32,
    0.0378131_f32,
    0.9945403_f32,
    -0.2168279_f32,
    0.2839020_f32,
    1.1097300_f32,
    -0.0229765_f32,
    0.3212028_f32,
    1.0971128_f32,
    0.1764092_f32,
    0.0193826_f32,
    0.8760764_f32,
    0.1285975_f32,
    0.4890027_f32,
    1.1460613_f32,
    0.7786202_f32,
    -0.0626174_f32,
    0.8631148_f32,
    -0.0611514_f32,
    -0.2474871_f32,
    0.9575630_f32,
    0.0698243_f32,
    -0.1563008_f32,
    1.0598850_f32,
    -0.0022975_f32,
    -0.3100439_f32,
    1.0589292_f32,
    -0.5999134_f32,
    -0.5350822_f32,
    0.8955259_f32,
    -0.3289533_f32,
    0.0320311_f32,
    0.9984024_f32,
    0.0836964_f32,
    -0.0767166_f32,
    0.8861030_f32,
    0.1190345_f32,
    0.1941680_f32,
    1.1190816_f32,
    0.0339921_f32,
    0.0122775_f32,
    0.9701337_f32,
    0.2830221_f32,
    0.2137495_f32,
    0.9747545_f32,
    0.4714881_f32,
    0.0635100_f32,
    0.7753871_f32,
    0.1939915_f32,
    0.0953922_f32,
    0.9393375_f32,
    0.3287779_f32,
    0.3957763_f32,
    1.0679222_f32,
    0.5971149_f32,
    0.2515192_f32,
    1.0563650_f32,
    0.5500914_f32,
    0.0454987_f32,
    0.9670420_f32,
    0.4008161_f32,
    0.6394243_f32,
    1.1475577_f32,
    0.5707502_f32,
    0.0835870_f32,
    0.9485882_f32,
    0.0248572_f32,
    -0.3437831_f32,
    0.8994003_f32,
    -0.1906862_f32,
    0.1739060_f32,
    1.1674629_f32,
    0.2980099_f32,
    -0.0545066_f32,
    1.0888529_f32,
    -0.1651553_f32,
    0.2676092_f32,
    0.7580549_f32,
    0.5553530_f32,
    -0.3227307_f32,
    0.9823096_f32,
    -0.1327482_f32,
    -0.1396336_f32,
    0.9205542_f32,
    -0.0759557_f32,
    0.2259490_f32,
    1.0587139_f32,
    0.2116114_f32,
    -0.2150586_f32,
    -0.4815771_f32,
    -0.2345996_f32,
    0.0715430_f32,
    0.1694490_f32,
    0.6277893_f32,
    -0.2980029_f32,
    -0.3116676_f32,
    -0.2534522_f32,
    -0.0138473_f32,
    0.8790613_f32,
    -0.6601876_f32,
    -0.3273432_f32,
    -0.0320922_f32,
    0.2296826_f32,
    -0.7639822_f32,
    -0.2126144_f32,
    0.3112076_f32,
    0.1890715_f32,
    -0.4402646_f32,
    -0.1989510_f32,
    0.3819954_f32,
    0.1802935_f32,
    -0.4343880_f32,
    0.1462527_f32,
    0.0455597_f32,
    0.0330499_f32,
    -0.3809013_f32,
    0.0917431_f32,
    -0.0742168_f32,
    0.1854619_f32,
    -0.3023599_f32,
    0.0821447_f32,
    -0.4640500_f32,
    -0.0656499_f32,
    0.3148834_f32,
    -0.0142385_f32,
    -0.1224921_f32,
    0.2085187_f32,
    -0.3357277_f32,
    0.0071951_f32,
    -0.3801588_f32,
    0.4073377_f32,
    -0.2737229_f32,
    -0.6480386_f32,
    0.1416543_f32,
    0.2305917_f32,
    -0.4069017_f32,
    0.0930619_f32,
    -0.3361472_f32,
    0.0197008_f32,
    0.2601102_f32,
    0.4998631_f32,
    0.7458162_f32,
    -0.3030633_f32,
    -0.3106394_f32,
    0.1557774_f32,
    -0.1393241_f32,
    0.3388757_f32,
    -0.7485161_f32,
    -0.2825306_f32,
    0.3942816_f32,
    -0.0023692_f32,
    -0.8545148_f32,
    0.0209450_f32,
    -0.7245241_f32,
    0.3064304_f32,
    0.2281581_f32,
    -0.1104448_f32,
    0.2878368_f32,
    0.0564405_f32,
    -0.0034715_f32,
    0.0213236_f32,
    -0.3463767_f32,
    -0.1118070_f32,
    -0.3310691_f32,
    -0.2943667_f32,
    0.1948387_f32,
    0.1440035_f32,
    -0.4818816_f32,
    -0.0789752_f32,
    -0.0475191_f32,
    -0.1980652_f32,
    -0.5627871_f32,
    -1.1011760_f32,
    -0.2705984_f32,
    -0.6317919_f32,
    -0.0953277_f32,
    -0.2671058_f32,
    0.3563909_f32,
    -0.0501725_f32,
    0.1251185_f32,
    -0.0024093_f32,
    -0.5946230_f32,
    -0.2770158_f32,
    0.0868469_f32,
    0.4372998_f32,
    -1.0561750_f32,
    0.2019041_f32,
    0.4024456_f32,
    0.2089194_f32,
    0.0663882_f32,
    0.1930386_f32,
    -0.2509571_f32,
    -0.0155761_f32,
    -0.1245154_f32,
    0.2880098_f32,
    -0.4039198_f32,
    -0.4449839_f32,
    0.0360742_f32,
    0.0785074_f32,
    -0.7333779_f32,
    0.0374439_f32,
    -0.6991382_f32,
    -0.0062087_f32,
    0.1952604_f32,
    -0.0794176_f32,
    0.6325591_f32,
    0.1258151_f32,
    -0.2446987_f32,
    0.0741554_f32,
    -0.7450882_f32,
    -0.0141401_f32,
    -0.1925968_f32,
    -0.2425244_f32,
    0.4810871_f32,
    0.1200433_f32,
    -0.5432531_f32,
    0.3180644_f32,
    -0.8195585_f32,
    0.0018904_f32,
    0.2279391_f32,
    0.3819911_f32,
    0.7607157_f32,
    0.0227216_f32,
    -0.2518524_f32,
    0.4207601_f32,
    -0.4938162_f32,
    -0.3393400_f32,
    -0.1476648_f32,
    -0.5783122_f32,
    0.1096253_f32,
    0.0586957_f32,
    -0.2941149_f32,
    -0.2041496_f32,
    -0.2823389_f32,
    -0.7590166_f32,
    0.0984770_f32,
    -0.2234334_f32,
    -0.0723187_f32,
    -0.6239209_f32,
    -0.6325903_f32,
    -0.2222456_f32,
    -0.0244784_f32,
    0.2966291_f32,
    0.2093343_f32,
    -0.2688833_f32,
    -0.0791881_f32,
    -0.8672296_f32,
    -0.0247865_f32,
    0.0127916_f32,
    -0.3957820_f32,
    0.4069268_f32,
    0.0186687_f32,
    -0.1345360_f32,
    0.0709840_f32,
    0.1383424_f32,
    -0.5773191_f32,
    0.0646583_f32,
    -0.4039402_f32,
    0.2468977_f32,
    0.0070791_f32,
    -0.4549448_f32,
    0.2705908_f32,
    -0.0364471_f32,
    -0.1968273_f32,
    -0.4171826_f32,
    0.1853591_f32,
    -0.3599790_f32,
    -0.3924109_f32,
    -0.9873305_f32,
    -0.0236451_f32,
    -0.4369704_f32,
    -0.3435808_f32,
    -0.3827062_f32,
    -0.3815695_f32,
    -0.2316054_f32,
    0.2810324_f32,
    -0.0667219_f32,
    0.1555770_f32,
    0.0738921_f32,
    0.0186481_f32,
    -0.5534586_f32,
    -0.0189945_f32,
    -0.6178082_f32,
    -0.6310515_f32,
    -1.0482125_f32,
    0.6715516_f32,
    -0.6172658_f32,
    -0.7239385_f32,
    -0.4223104_f32,
    0.3821667_f32,
    -0.3781609_f32,
    -0.4107505_f32,
    -0.0463715_f32,
    -0.4596776_f32,
    -0.4936227_f32,
    -0.3267912_f32,
    0.1723617_f32,
    -0.1801065_f32,
    0.1414004_f32,
    0.2754991_f32,
    -0.0445292_f32,
    0.1011325_f32,
    -0.1429617_f32,
    -0.1663917_f32,
    0.1718503_f32,
    -0.4029871_f32,
    0.1995195_f32,
    0.0507600_f32,
    -0.1068991_f32,
    0.6494417_f32,
    0.2228922_f32,
    -0.4733713_f32,
    -0.2036799_f32,
    -0.3645424_f32,
    0.0340878_f32,
    0.0463455_f32,
    0.1899979_f32,
    0.4913444_f32,
    0.1001289_f32,
    -0.6297565_f32,
    0.2036585_f32,
    -0.2957630_f32,
    -0.1077577_f32,
    0.1951456_f32,
    -0.6914741_f32,
    0.0110312_f32,
    -0.0962586_f32,
    -0.0358197_f32,
    -0.4012583_f32,
    0.0231706_f32,
    -0.1281997_f32,
    -0.8421248_f32,
    0.1666217_f32,
    0.3005572_f32,
    -0.1027572_f32,
    -0.1166396_f32,
    -0.0727212_f32,
    -0.0180009_f32,
    -0.3706886_f32,
    0.3216677_f32,
    0.3744043_f32,
    -0.2427881_f32,
    0.0365849_f32,
    0.0474909_f32,
    0.1623246_f32,
    0.2930891_f32,
    0.2488571_f32,
    0.4307851_f32,
    -0.0908276_f32,
    -0.1336632_f32,
    -0.0631434_f32,
    0.1080624_f32,
    0.0419672_f32,
    0.3945646_f32,
    0.3952877_f32,
    -0.1056918_f32,
    0.2305445_f32,
    0.1911329_f32,
    0.3260361_f32,
    0.1858367_f32,
    0.0440293_f32,
    -0.6557166_f32,
    0.1156181_f32,
    -0.0017963_f32,
    0.3244708_f32,
    0.1014456_f32,
    0.2468570_f32,
    0.3840420_f32,
    0.2116503_f32,
    -0.8463494_f32,
    0.0452271_f32,
    0.0122542_f32,
    -0.6610509_f32,
    0.1553790_f32,
    -0.0081018_f32,
    -0.3137966_f32,
    -0.3779987_f32,
    0.1755779_f32,
    -0.5357634_f32,
    -0.5429783_f32,
    -0.0517890_f32,
    0.1552144_f32,
    -0.2603098_f32,
    -0.0602923_f32,
    -0.1586821_f32,
    -0.1995614_f32,
    -0.6059815_f32,
    0.2292956_f32,
    0.4575188_f32,
    -0.4609511_f32,
    -0.7025276_f32,
    -0.0529122_f32,
    0.0936021_f32,
    -0.0662461_f32,
    -0.3161530_f32,
    0.1855838_f32,
    -0.0045533_f32,
    0.3807741_f32,
    -0.3142389_f32,
    0.1079214_f32,
    0.3304892_f32,
    0.1348565_f32,
    -0.1116893_f32,
    0.2480519_f32,
    -0.1326695_f32,
    -0.1445942_f32,
    0.1271062_f32,
    -0.3148151_f32,
    0.3086464_f32,
    -0.2716174_f32,
    -0.1267714_f32,
    0.2351022_f32,
    0.1043204_f32,
    0.0513768_f32,
    -0.0031377_f32,
    0.0185488_f32,
    -0.2646657_f32,
    0.0786427_f32,
    0.0316442_f32,
    0.2512519_f32,
    -0.0425524_f32,
    0.0146502_f32,
    -0.2090448_f32,
    0.3714326_f32,
    0.1231930_f32,
    0.1179512_f32,
    0.0037010_f32,
    -0.0021364_f32,
    0.0855395_f32,
    -0.2811730_f32,
    0.2779074_f32,
    -0.2193812_f32,
    0.0505008_f32,
    -0.1677244_f32,
    -0.0521841_f32,
    0.6761827_f32,
    -0.2363487_f32,
    -0.3807964_f32,
    0.0283764_f32,
    -0.0172157_f32,
    -0.0192487_f32,
    0.1193792_f32,
    -0.1898157_f32,
    -0.1632355_f32,
    0.3336088_f32,
    0.2504867_f32,
    0.2001165_f32,
    -0.1812013_f32,
    0.0634891_f32,
    -0.3439041_f32,
    -0.3072314_f32,
    -0.2081489_f32,
    -0.0836721_f32,
    0.1509564_f32,
    0.0470818_f32,
    0.1288994_f32,
    -2.7429733_f32,
    -2.6276667_f32,
    -0.0047758_f32,
    0.0667859_f32,
    0.0743652_f32,
    -0.2614106_f32,
];
