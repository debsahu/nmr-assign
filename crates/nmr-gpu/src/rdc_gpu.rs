//! Batched multi-conformer WebGPU (`wgpu` + WGSL) torsion-space Residual Dipolar Coupling (`RDC`)
//! energy, analytical torque gradient, and ensemble refinement engine (`docs/specs/04-structure.md` §5).
//!
//! Provenance: Clean-room GPU compute implementation from peer-reviewed biomolecular NMR and
//! polymer kinematics literature:
//! - Tjandra, N. & Bax, A. (*Science* 1997, 278, 1111–1114 — dipolar couplings in weakly aligned
//!   macromolecules).
//! - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* 1998, 120,
//!   6836–6837 — `Q`-factor validation in bicelle media).
//! - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (*J. Magn. Reson.* 1999,
//!   138, 334–342 — linear 5-component traceless symmetric Saupe alignment tensor SVD).
//! - Clore, G. M., Gronenborn, A. M. & Tjandra, N. (*J. Magn. Reson.* 1998, 131, 159–162 — axial
//!   and rhombic dipolar alignment components).
//! - Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (*J. Comput. Chem.*
//!   2005, 26, 1063–1068 — Natural Extension Reference Frame forward kinematics).
//! - Abe, H., Braun, W., Noguti, T. & Go, N. (*Comput. Chem.* 1984, 8, 239–247 — `O(L)` backward
//!   suffix-sum torque identity for exact torsion-angle derivatives).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use bytemuck::{Pod, Zeroable};
use nmr_core::dihedrals::DihedralRestraint;
use nmr_core::noesy::AmbiguousDistanceRestraint;
use nmr_core::rdc::{
    evaluate_rdc_validation, fit_saupe_tensor_svd, normalize_rdc_atom_name, RdcMediumSet,
    RdcValidationSummary,
};
use nmr_core::structure::{
    build_backbone_from_torsions_rad, evaluate_energy_and_gradients, EnergyWeights,
    StructureConformer,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RdcRefineParamsGpu {
    n_conf: u32,
    l_res: u32,
    n_media: u32,
    n_restraints: u32,
    w_rdc: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RdcTermGpu {
    medium_idx: u32,
    atom_a: u32,
    atom_b: u32,
    seg_a: u32,
    seg_b: u32,
    d_obs: f32,
    d_max: f32,
    inv_sig2: f32,
}

/// Per-conformer input state (`(phi, psi, omega)` in radians + per-medium `3x3` Saupe tensors)
/// for batched WebGPU RDC energy & analytical torsion-gradient evaluation.
#[derive(Clone, Debug)]
pub struct RdcGpuConformerInput {
    pub phi_rad: Vec<f64>,
    pub psi_rad: Vec<f64>,
    pub omega_rad: Vec<f64>,
    pub saupe_matrices: Vec<[[f64; 3]; 3]>,
}

/// Per-conformer WebGPU RDC energy and analytical `(dE_rdc/dphi, dE_rdc/dpsi)` torsion gradients.
#[derive(Clone, Debug)]
pub struct RdcGpuConformerOutput {
    pub rdc_energy: f64,
    pub grad_phi: Vec<f64>,
    pub grad_psi: Vec<f64>,
}

/// Pre-compiled WebGPU compute pipeline for batched multi-conformer RDC torsion evaluation.
pub struct RdcGpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl RdcGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-rdc-refine-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rdc_refine.wgsl").into()),
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
            label: Some("nmr-rdc-refine-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_rw(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-rdc-refine-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-rdc-refine-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eval_rdc_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }
}

fn resolve_gpu_atom_slot_and_seg(
    sequence: &[String],
    seq_to_idx: &BTreeMap<i32, usize>,
    seq_id: i32,
    atom_name: &str,
) -> Option<(u32, u32)> {
    let &res_idx = seq_to_idx.get(&seq_id)?;
    let res_name = sequence.get(res_idx)?;
    let norm = normalize_rdc_atom_name(atom_name);
    let i = res_idx as u32;
    match norm {
        "N" => Some((7 * i, 2 * i)),
        "H" => {
            if res_name.eq_ignore_ascii_case("PRO") {
                None
            } else {
                Some((7 * i + 1, 2 * i))
            }
        }
        "CA" => Some((7 * i + 2, 2 * i)),
        "HA" => Some((7 * i + 3, 2 * i + 1)),
        "C" => Some((7 * i + 4, 2 * i + 1)),
        "O" => Some((7 * i + 5, 2 * i + 2)),
        "CB" => {
            if res_name.eq_ignore_ascii_case("GLY") {
                None
            } else {
                Some((7 * i + 6, 2 * i + 1))
            }
        }
        _ => None,
    }
}

fn pack_rdc_terms_gpu(
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
) -> Result<Vec<RdcTermGpu>, String> {
    let mut seq_to_idx = BTreeMap::new();
    for (idx, &sid) in seq_ids.iter().enumerate() {
        seq_to_idx.insert(sid, idx);
    }

    let mut terms = Vec::new();
    for (m_idx, medium) in rdc_media.iter().enumerate() {
        for r in &medium.restraints {
            r.validate()?;
            if r.is_free {
                continue;
            }
            let (Some((atom_a, seg_a)), Some((atom_b, seg_b))) = (
                resolve_gpu_atom_slot_and_seg(sequence, &seq_to_idx, r.seq_id_1, &r.atom_name_1),
                resolve_gpu_atom_slot_and_seg(sequence, &seq_to_idx, r.seq_id_2, &r.atom_name_2),
            ) else {
                continue;
            };
            let inv_sig2 = 1.0 / (r.sigma_hz * r.sigma_hz);
            terms.push(RdcTermGpu {
                medium_idx: m_idx as u32,
                atom_a,
                atom_b,
                seg_a,
                seg_b,
                d_obs: r.d_obs_hz as f32,
                d_max: r.d_max_hz as f32,
                inv_sig2: inv_sig2 as f32,
            });
        }
    }
    Ok(terms)
}

/// Evaluate multi-conformer harmonic RDC energies and analytical `(dE_rdc/dphi, dE_rdc/dpsi)`
/// torsion gradients on WebGPU asynchronously (`wasm32` + native compatible).
pub async fn evaluate_rdc_batch_gpu_async(
    engine: &GpuEngine,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    let rdc_pipe = RdcGpuPipeline::new(&engine.device);
    evaluate_rdc_batch_with_pipeline_async(
        engine, &rdc_pipe, sequence, seq_ids, rdc_media, conformers, w_rdc,
    )
    .await
}

async fn evaluate_rdc_batch_with_pipeline_async(
    engine: &GpuEngine,
    rdc_pipe: &RdcGpuPipeline,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    let l = sequence.len();
    if seq_ids.len() != l {
        return Err("sequence and seq_ids length mismatch".to_string());
    }
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!("invalid w_rdc={w_rdc}: must be finite and >= 0"));
    }
    let n_conf = conformers.len();
    if n_conf == 0 {
        return Ok(Vec::new());
    }
    if l == 0 || rdc_media.is_empty() || w_rdc == 0.0 {
        return Ok((0..n_conf)
            .map(|_| RdcGpuConformerOutput {
                rdc_energy: 0.0,
                grad_phi: vec![0.0; l],
                grad_psi: vec![0.0; l],
            })
            .collect());
    }

    let n_media = rdc_media.len();
    for (ci, c) in conformers.iter().enumerate() {
        if c.phi_rad.len() != l || c.psi_rad.len() != l || c.omega_rad.len() != l {
            return Err(format!(
                "conformer {ci} torsion array length mismatch (expected {l})"
            ));
        }
        if c.saupe_matrices.len() != n_media {
            return Err(format!(
                "conformer {ci} saupe_matrices length {} != n_media {n_media}",
                c.saupe_matrices.len()
            ));
        }
    }

    let terms = pack_rdc_terms_gpu(sequence, seq_ids, rdc_media)?;
    let n_restraints = terms.len();
    if n_restraints == 0 {
        return Ok((0..n_conf)
            .map(|_| RdcGpuConformerOutput {
                rdc_energy: 0.0,
                grad_phi: vec![0.0; l],
                grad_psi: vec![0.0; l],
            })
            .collect());
    }

    let res_flags: Vec<u32> = sequence
        .iter()
        .map(|r| {
            if r.eq_ignore_ascii_case("GLY") {
                1u32
            } else if r.eq_ignore_ascii_case("PRO") {
                2u32
            } else {
                0u32
            }
        })
        .collect();

    let mut torsions_flat = Vec::with_capacity(n_conf * l);
    let mut saupe_flat = Vec::with_capacity(n_conf * n_media * 3);
    for c in conformers {
        for i in 0..l {
            torsions_flat.push([
                c.phi_rad[i] as f32,
                c.psi_rad[i] as f32,
                c.omega_rad[i] as f32,
                0.0f32,
            ]);
        }
        for s in &c.saupe_matrices {
            saupe_flat.push([s[0][0] as f32, s[0][1] as f32, s[0][2] as f32, 0.0f32]);
            saupe_flat.push([s[1][0] as f32, s[1][1] as f32, s[1][2] as f32, 0.0f32]);
            saupe_flat.push([s[2][0] as f32, s[2][1] as f32, s[2][2] as f32, 0.0f32]);
        }
    }

    let params = RdcRefineParamsGpu {
        n_conf: n_conf as u32,
        l_res: l as u32,
        n_media: n_media as u32,
        n_restraints: n_restraints as u32,
        w_rdc: w_rdc as f32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let n_segs = 2 * l + 1;
    let scratch_vec4_count = n_conf
        .checked_mul(7 * l + 2 * n_segs)
        .ok_or_else(|| "GPU RDC scratch buffer size overflow".to_string())?;
    let scratch_bytes = (scratch_vec4_count as u64)
        .checked_mul(16)
        .ok_or_else(|| "GPU RDC scratch byte size overflow".to_string())?;

    let lim = engine.device.limits();
    let max_bind = lim.max_storage_buffer_binding_size.min(lim.max_buffer_size);
    if scratch_bytes > max_bind {
        return Err(format!(
            "GPU RDC scratch buffer ({scratch_bytes} B) exceeds device max storage binding ({max_bind} B)"
        ));
    }

    let params_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let flags_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-res-flags"),
            contents: bytemuck::cast_slice(&res_flags),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let torsions_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-torsions"),
            contents: bytemuck::cast_slice(&torsions_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let saupe_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-saupe"),
            contents: bytemuck::cast_slice(&saupe_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let terms_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-terms"),
            contents: bytemuck::cast_slice(&terms),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let scratch_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-rdc-scratch"),
        size: scratch_bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let energy_bytes = ((n_conf * 4) as u64)
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        * wgpu::COPY_BUFFER_ALIGNMENT;
    let energies_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-rdc-energies"),
        size: energy_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let grad_elems = n_conf * l;
    let grad_bytes = ((grad_elems * 8) as u64)
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        * wgpu::COPY_BUFFER_ALIGNMENT;
    let grads_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-rdc-grads"),
        size: grad_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nmr-rdc-bg"),
        layout: &rdc_pipe.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: flags_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: torsions_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: saupe_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: terms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scratch_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: energies_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: grads_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = engine
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nmr-rdc-batch-enc"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nmr-rdc-batch-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&rdc_pipe.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        let n_wg = (n_conf as u32).div_ceil(64);
        cpass.dispatch_workgroups(n_wg, 1, 1);
    }
    engine.queue.submit(Some(encoder.finish()));

    let energies_f32: Vec<f32> = engine.read_buffer_vec_async(&energies_buf, n_conf).await?;
    let grads_f32: Vec<[f32; 2]> = engine.read_buffer_vec_async(&grads_buf, grad_elems).await?;

    let mut outputs = Vec::with_capacity(n_conf);
    for (ci, &e_f32) in energies_f32.iter().enumerate().take(n_conf) {
        let mut gp = Vec::with_capacity(l);
        let mut gs = Vec::with_capacity(l);
        let off = ci * l;
        for i in 0..l {
            gp.push(grads_f32[off + i][0] as f64);
            gs.push(grads_f32[off + i][1] as f64);
        }
        outputs.push(RdcGpuConformerOutput {
            rdc_energy: e_f32 as f64,
            grad_phi: gp,
            grad_psi: gs,
        });
    }
    Ok(outputs)
}

/// Synchronous native wrapper around [`evaluate_rdc_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn evaluate_rdc_batch_gpu(
    engine: &GpuEngine,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    pollster::block_on(evaluate_rdc_batch_gpu_async(
        engine, sequence, seq_ids, rdc_media, conformers, w_rdc,
    ))
}

#[inline]
fn wrap_rad(rad: f64) -> f64 {
    let mut r = (rad + PI) % (2.0 * PI);
    if r <= 0.0 {
        r += 2.0 * PI;
    }
    r - PI
}

#[inline]
fn wrap_deg(deg: f64) -> f64 {
    let mut d = (deg + 180.0) % 360.0;
    if d <= 0.0 {
        d += 360.0;
    }
    d - 180.0
}

/// Refine a conformer ensemble asynchronously using exact CPU `f64` Losonczi SVD Saupe tensor
/// updates paired with batched WebGPU (`rdc_refine.wgsl`) multi-conformer RDC torsion gradient
/// dispatches (`wasm32` + native compatible).
#[allow(clippy::too_many_arguments)]
pub async fn refine_ensemble_with_rdcs_gpu_async(
    engine: &GpuEngine,
    conformers: &[StructureConformer],
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[RdcMediumSet],
    w_rdc: f64,
    n_iters: usize,
) -> Result<(Vec<StructureConformer>, RdcValidationSummary), String> {
    if conformers.is_empty() {
        return Err("refine_ensemble_with_rdcs_gpu requires at least one conformer".to_string());
    }
    let l = sequence.len();
    if seq_ids.len() != l {
        return Err("sequence and seq_ids length mismatch".to_string());
    }
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!("invalid w_rdc={w_rdc}: must be finite and >= 0"));
    }
    if rdc_media.is_empty() {
        return Err("refine_ensemble_with_rdcs_gpu requires at least one RDC medium".to_string());
    }

    // Validate all media up front on first conformer
    let _ = evaluate_rdc_validation(&conformers[0].chain, rdc_media)?;

    let rdc_pipe = RdcGpuPipeline::new(&engine.device);
    let n_conf = conformers.len();

    struct ConfState {
        phi: Vec<f64>,
        psi: Vec<f64>,
        omega: Vec<f64>,
        vel_phi: Vec<f64>,
        vel_psi: Vec<f64>,
        best_phi: Vec<f64>,
        best_psi: Vec<f64>,
        best_total: f64,
        best_noe: f64,
        best_dih: f64,
        best_vdw: f64,
        best_rama: f64,
    }

    let mut states: Vec<ConfState> = conformers
        .iter()
        .map(|c| {
            let phi: Vec<f64> = c.phi_deg.iter().map(|d| d.to_radians()).collect();
            let psi: Vec<f64> = c.psi_deg.iter().map(|d| d.to_radians()).collect();
            let omega: Vec<f64> = if c.omega_deg.len() == l {
                c.omega_deg.iter().map(|d| d.to_radians()).collect()
            } else {
                vec![PI; l]
            };
            ConfState {
                best_phi: phi.clone(),
                best_psi: psi.clone(),
                phi,
                psi,
                omega,
                vel_phi: vec![0.0; l],
                vel_psi: vec![0.0; l],
                best_total: f64::INFINITY,
                best_noe: c.noe_energy,
                best_dih: c.dihedral_energy,
                best_vdw: c.vdw_energy,
                best_rama: c.rama_energy,
            }
        })
        .collect();

    // Flat-bottom dihedral tolerances (`>= 18 deg`) prevent Saupe mirror-image degeneracies
    // without pulling `(phi, psi)` away from sub-Hz experimental RDCs inside the native basin.
    let reg_dihs: Vec<DihedralRestraint> = if dihedrals.len() == l {
        dihedrals
            .iter()
            .map(|dr| {
                let mut d = dr.clone();
                if d.restrained_phi {
                    d.phi_err_deg = d.phi_err_deg.max(18.0);
                }
                if d.restrained_psi {
                    d.psi_err_deg = d.psi_err_deg.max(18.0);
                }
                d
            })
            .collect()
    } else {
        Vec::new()
    };

    let n_work_tot: usize = rdc_media
        .iter()
        .map(|m| m.restraints.iter().filter(|r| !r.is_free).count())
        .sum();
    let eff_w_rdc = w_rdc * (185.0 / (n_work_tot.max(60) as f64)).clamp(0.34, 1.50);
    let weights = EnergyWeights {
        w_noe: 0.90,
        noe_soft_window: 2.0,
        w_dihedral: 2.0,
        w_vdw: 1.8,
        w_rama: 8.0,
        ..EnergyWeights::default()
    };
    let total_steps = n_iters.max(1);

    for step in 0..total_steps {
        let prog = if total_steps > 1 {
            (step as f64) / ((total_steps - 1) as f64)
        } else {
            1.0
        };
        // Ramp RDC weight smoothly over the first 30% of steps, then hold full eff_w_rdc
        let cur_w_rdc = if prog < 0.30 {
            eff_w_rdc * (0.25 + 0.75 * (prog / 0.30))
        } else {
            eff_w_rdc
        };

        // 1. Exact CPU f64 SVD Saupe tensor fit per conformer + physical force evaluation
        let mut gpu_inputs = Vec::with_capacity(n_conf);
        let mut phys_gp = vec![vec![0.0f64; l]; n_conf];
        let mut phys_gs = vec![vec![0.0f64; l]; n_conf];
        let mut phys_eb = Vec::with_capacity(n_conf);

        for (ci, st) in states.iter().enumerate() {
            let (chain, eb) = evaluate_energy_and_gradients(
                sequence,
                seq_ids,
                &st.phi,
                &st.psi,
                &st.omega,
                adrs,
                &reg_dihs,
                &weights,
                &mut phys_gp[ci],
                &mut phys_gs[ci],
            );
            let mut mats = Vec::with_capacity(rdc_media.len());
            for m in rdc_media {
                let fit = fit_saupe_tensor_svd(&chain, m)?;
                mats.push(fit.s_matrix);
            }
            gpu_inputs.push(RdcGpuConformerInput {
                phi_rad: st.phi.clone(),
                psi_rad: st.psi.clone(),
                omega_rad: st.omega.clone(),
                saupe_matrices: mats,
            });
            phys_eb.push(eb);
        }

        // 2. Single batched WebGPU dispatch across all conformers for RDC energy + torsion gradients
        let gpu_outs = evaluate_rdc_batch_with_pipeline_async(
            engine,
            &rdc_pipe,
            sequence,
            seq_ids,
            rdc_media,
            &gpu_inputs,
            cur_w_rdc,
        )
        .await?;

        // 3. Record lowest full-weight total energy and apply clipped Nesterov/heavy-ball step
        for ci in 0..n_conf {
            let st = &mut states[ci];
            let eb = &phys_eb[ci];
            let rdc_out = &gpu_outs[ci];
            let full_rdc_e = if cur_w_rdc > 0.0 {
                rdc_out.rdc_energy * (eff_w_rdc / cur_w_rdc)
            } else {
                0.0
            };
            let full_total = eb.total + full_rdc_e;
            if full_total < st.best_total {
                st.best_total = full_total;
                st.best_noe = eb.noe;
                st.best_dih = eb.dihedral;
                st.best_vdw = eb.vdw;
                st.best_rama = eb.rama;
                st.best_phi.copy_from_slice(&st.phi);
                st.best_psi.copy_from_slice(&st.psi);
            }

            let mut norm2 = 0.0f64;
            for i in 0..l {
                let gp = if i > 0 {
                    phys_gp[ci][i] + rdc_out.grad_phi[i]
                } else {
                    0.0
                };
                let gs = phys_gs[ci][i] + rdc_out.grad_psi[i];
                norm2 += gp * gp + gs * gs;
            }
            let g_norm = norm2.sqrt().max(1.0);
            let lr = (0.045 * (1.0 - 0.65 * prog)) / (1.0 + 0.12 * g_norm);
            let mom = 0.55 * (1.0 - 0.4 * prog);

            for i in 0..l {
                if i > 0 {
                    let gp = phys_gp[ci][i] + rdc_out.grad_phi[i];
                    st.vel_phi[i] = mom * st.vel_phi[i] - lr * gp;
                    st.phi[i] = wrap_rad(st.phi[i] + st.vel_phi[i].clamp(-0.04, 0.04));
                }
                let gs = phys_gs[ci][i] + rdc_out.grad_psi[i];
                st.vel_psi[i] = mom * st.vel_psi[i] - lr * gs;
                st.psi[i] = wrap_rad(st.psi[i] + st.vel_psi[i].clamp(-0.04, 0.04));
            }
        }
    }

    // Final exact evaluation at best torsion state per conformer + final L-BFGS polish
    let mut best_seeds = Vec::with_capacity(n_conf);
    for st in states {
        let chain = build_backbone_from_torsions_rad(
            sequence,
            seq_ids,
            &st.best_phi,
            &st.best_psi,
            &st.omega,
        );
        best_seeds.push(StructureConformer {
            chain,
            phi_deg: st
                .best_phi
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            psi_deg: st
                .best_psi
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            omega_deg: st
                .omega
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            total_energy: st.best_total,
            noe_energy: st.best_noe,
            dihedral_energy: st.best_dih,
            vdw_energy: st.best_vdw,
            rama_energy: st.best_rama,
        });
    }

    let polish_iters = n_iters.clamp(40, 140);
    nmr_core::structure::refine_ensemble_with_rdcs(
        &best_seeds,
        sequence,
        seq_ids,
        adrs,
        dihedrals,
        rdc_media,
        w_rdc,
        polish_iters,
    )
}

/// Synchronous native wrapper around [`refine_ensemble_with_rdcs_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn refine_ensemble_with_rdcs_gpu(
    engine: &GpuEngine,
    conformers: &[StructureConformer],
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[RdcMediumSet],
    w_rdc: f64,
    n_iters: usize,
) -> Result<(Vec<StructureConformer>, RdcValidationSummary), String> {
    pollster::block_on(refine_ensemble_with_rdcs_gpu_async(
        engine, conformers, sequence, seq_ids, adrs, dihedrals, rdc_media, w_rdc, n_iters,
    ))
}
