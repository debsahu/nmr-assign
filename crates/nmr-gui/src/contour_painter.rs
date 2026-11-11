//! Custom `egui_wgpu` GPU callback resource for real-time 2D NMR contour rendering.
//!
//! Provenance: Original `wgpu` rendering pipeline driving `contour_shader.wgsl`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu, wgpu};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContourUniforms {
    /// `(h_left_ppm, h_right_ppm, n_top_ppm, n_bottom_ppm)`
    pub view_ppm: [f32; 4],
    /// `(first_x_ppm, step_x_ppm, first_y_ppm, step_y_ppm)`
    pub axis_ppm: [f32; 4],
    /// `(nx, ny, canvas_width_px, canvas_height_px)`
    pub dims: [f32; 4],
    /// `(base_level, level_factor, num_levels, show_negative)`
    pub levels: [f32; 4],
}

pub struct ContourPipelineResource {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    spec_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uploaded_spec_gen: u64,
}

impl ContourPipelineResource {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-contour-2d-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("contour_shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-contour-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-contour-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nmr-contour-pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-contour-uniform"),
            size: std::mem::size_of::<ContourUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy = [0.0f32; 4];
        let spec_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-contour-spec-init"),
            contents: bytemuck::cast_slice(&dummy),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-contour-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spec_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bgl,
            uniform_buf,
            spec_buf,
            bind_group,
            uploaded_spec_gen: 0,
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &ContourUniforms,
        spec_gen: u64,
        spec_data: &[f32],
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
        if self.uploaded_spec_gen != spec_gen && !spec_data.is_empty() {
            self.spec_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-contour-spec"),
                contents: bytemuck::cast_slice(spec_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nmr-contour-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.spec_buf.as_entire_binding(),
                    },
                ],
            });
            self.uploaded_spec_gen = spec_gen;
        }
    }
}

pub struct ContourCallback {
    pub uniforms: ContourUniforms,
    pub spec_gen: u64,
    pub spec_data: Arc<Vec<f32>>,
}

impl egui_wgpu::CallbackTrait for ContourCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<ContourPipelineResource>() {
            res.update(
                device,
                queue,
                &self.uniforms,
                self.spec_gen,
                &self.spec_data,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<ContourPipelineResource>() else {
            return;
        };
        let vp = info.viewport_in_pixels();
        if vp.width_px <= 0 || vp.height_px <= 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
