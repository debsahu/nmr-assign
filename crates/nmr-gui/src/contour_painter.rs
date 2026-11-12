//! Custom `egui_wgpu` GPU callback resource for real-time multi-spectrum 2D NMR contour and 3D
//! orthogonal slice overlays, plus synchronized `(1H, 15N, 13C)` cursor crosshairs.
//!
//! Provenance: Original clean-room `wgpu` rendering pipeline driving `contour_shader.wgsl` with
//! per-layer VRAM buffer caching, chemical-shift nudge offsets, hardware alpha compositing
//! (`BlendState::ALPHA_BLENDING`), and shared 2D/3D coordinate projection helpers.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::{egui, egui_wgpu, wgpu};
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use wgpu::util::DeviceExt;

/// Maximum cached spectrum GPU storage buffers retained across frames before LRU eviction.
const MAX_CACHED_SPEC_BUFFERS: usize = 32;

/// Default apex highlight whitening blend factor in `[0.0, 1.0]`.
const DEFAULT_APEX_WHITEN: f32 = 0.62;

static NEXT_MULTI_CALLBACK_BATCH_ID: AtomicU64 = AtomicU64::new(1);

/// Curated dark-theme `(positive_rgba, negative_rgba)` overlay color palettes so stacked
/// `15N-HSQC`, `13C-HSQC`, titration points, and 3D orthogonal slices remain visually distinct.
#[allow(dead_code)]
pub const OVERLAY_PALETTES: [([f32; 4], [f32; 4]); 6] = [
    // 0: Sky-cyan (+) / Coral-rose (-) — primary reference spectrum
    ([0.12, 0.68, 0.96, 1.0], [0.95, 0.36, 0.42, 0.95]),
    // 1: Emerald-mint (+) / Warm-amber (-) — overlaid 13C-HSQC or 1st titration point
    ([0.18, 0.88, 0.56, 0.92], [0.98, 0.72, 0.24, 0.90]),
    // 2: Orchid-violet (+) / Soft-tangerine (-) — 2nd overlay / HNCO plane
    ([0.74, 0.44, 0.98, 0.90], [0.98, 0.54, 0.28, 0.88]),
    // 3: Solar-gold (+) / Crimson-red (-) — 3rd overlay / HNCACB slice
    ([0.98, 0.84, 0.24, 0.90], [0.92, 0.24, 0.34, 0.88]),
    // 4: Aqua-teal (+) / Magenta-pink (-)
    ([0.16, 0.86, 0.82, 0.88], [0.94, 0.34, 0.74, 0.86]),
    // 5: Periwinkle-blue (+) / Peach-coral (-)
    ([0.46, 0.62, 1.00, 0.88], [1.00, 0.64, 0.48, 0.86]),
];

/// Return `(pos_rgba, neg_rgba)` from [`OVERLAY_PALETTES`] cycling cleanly by `layer_idx`.
#[allow(dead_code)]
#[inline]
pub fn default_overlay_palette(layer_idx: usize) -> ([f32; 4], [f32; 4]) {
    OVERLAY_PALETTES[layer_idx % OVERLAY_PALETTES.len()]
}

/// Legacy single-layer uniform descriptor preserved with exact field layout for backward
/// compatibility with single-spectrum `ContourCallback` call sites.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
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

impl ContourUniforms {
    /// Construct single-spectrum uniforms from a [`Spectrum2D`] and viewport parameters.
    #[allow(dead_code)]
    pub fn from_spectrum_2d(
        spec: &Spectrum2D,
        view_ppm: [f32; 4],
        canvas_size_px: [f32; 2],
        base_level: f32,
        level_factor: f32,
        num_levels: u32,
        show_negative: bool,
    ) -> Self {
        Self {
            view_ppm,
            axis_ppm: [
                spec.axis_x.first_ppm as f32,
                spec.axis_x.step_ppm as f32,
                spec.axis_y.first_ppm as f32,
                spec.axis_y.step_ppm as f32,
            ],
            dims: [
                spec.nx() as f32,
                spec.ny() as f32,
                canvas_size_px[0],
                canvas_size_px[1],
            ],
            levels: [
                base_level,
                level_factor,
                num_levels as f32,
                if show_negative { 1.0 } else { 0.0 },
            ],
        }
    }

    /// Promote legacy single-spectrum uniforms into a 112-byte [`GpuContourLayerUniforms`] block.
    #[inline]
    pub fn to_gpu_layer(self, draw_background: bool) -> GpuContourLayerUniforms {
        let (pos_rgba, neg_rgba) = OVERLAY_PALETTES[0];
        GpuContourLayerUniforms {
            view_ppm: self.view_ppm,
            axis_ppm: self.axis_ppm,
            dims: self.dims,
            levels: self.levels,
            pos_rgba,
            neg_rgba,
            layer_meta: [
                0.0,
                0.0,
                if draw_background { 1.0 } else { 0.0 },
                DEFAULT_APEX_WHITEN,
            ],
        }
    }
}

/// Hardware-aligned (`7 * 16 = 112` bytes) per-pass uniform block matching `ContourUniforms` in
/// `contour_shader.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuContourLayerUniforms {
    /// `(h_left_ppm, h_right_ppm, n_top_ppm, n_bottom_ppm)`
    pub view_ppm: [f32; 4],
    /// `(first_x_ppm, step_x_ppm, first_y_ppm, step_y_ppm)`
    pub axis_ppm: [f32; 4],
    /// `(nx, ny, canvas_width_px, canvas_height_px)`
    pub dims: [f32; 4],
    /// `(base_level, level_factor, num_levels, show_negative)`
    pub levels: [f32; 4],
    /// Positive contour base color and opacity `(r, g, b, a)`
    pub pos_rgba: [f32; 4],
    /// Negative contour base color and opacity `(r, g, b, a)`
    pub neg_rgba: [f32; 4],
    /// `(nudge_x_ppm, nudge_y_ppm, draw_background, apex_whiten)`
    pub layer_meta: [f32; 4],
}

/// Per-layer specification for an ordered stack of overlaid 2D NMR spectra or 3D orthogonal slices.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct OverlayLayerSpec {
    pub layer_id: u64,
    pub spec_gen: u64,
    pub spec_data: Arc<Vec<f32>>,
    /// `(first_x_ppm, step_x_ppm, first_y_ppm, step_y_ppm)`
    pub axis_ppm: [f32; 4],
    /// `(nx, ny)` grid resolution of `spec_data`
    pub dims: [u32; 2],
    pub base_level: f32,
    pub level_factor: f32,
    pub num_levels: u32,
    pub show_negative: bool,
    pub pos_rgba: [f32; 4],
    pub neg_rgba: [f32; 4],
    /// Display chemical-shift nudge `(dh_ppm, dn_ppm)` applied to this layer
    pub ppm_nudge: [f32; 2],
    pub visible: bool,
}

#[allow(dead_code)]
impl OverlayLayerSpec {
    /// Build an overlay layer from a [`Spectrum2D`] using [`default_overlay_palette`].
    pub fn from_spectrum_2d(
        layer_id: u64,
        spec_gen: u64,
        spec: &Spectrum2D,
        spec_data: Arc<Vec<f32>>,
        base_level: f32,
        palette_index: usize,
    ) -> Self {
        let (pos_rgba, neg_rgba) = default_overlay_palette(palette_index);
        Self {
            layer_id,
            spec_gen,
            spec_data,
            axis_ppm: [
                spec.axis_x.first_ppm as f32,
                spec.axis_x.step_ppm as f32,
                spec.axis_y.first_ppm as f32,
                spec.axis_y.step_ppm as f32,
            ],
            dims: [spec.nx() as u32, spec.ny() as u32],
            base_level: base_level.max(1e-4),
            level_factor: 1.42,
            num_levels: 10,
            show_negative: true,
            pos_rgba,
            neg_rgba,
            ppm_nudge: [0.0, 0.0],
            visible: true,
        }
    }

    /// Extract an orthogonal `(1H, 15N)` plane at `c_ppm` from a [`Spectrum3D`] cube for direct
    /// multi-spectrum contour overlay onto a 2D `15N-HSQC` viewport.
    pub fn from_spectrum_3d_hn_plane(
        layer_id: u64,
        spec_gen: u64,
        spec3d: &Spectrum3D,
        c_ppm: f64,
        base_level: f32,
        palette_index: usize,
    ) -> Self {
        let y_is_n = spec3d.axis_y.label.contains('N')
            || (spec3d.axis_y.max_ppm() > 95.0
                && spec3d.axis_y.min_ppm() < 145.0
                && !spec3d.axis_z.label.contains('N'));

        let nx = spec3d.nx();
        let (n_axis, c_axis, plane_data) = if y_is_n {
            let iz = spec3d
                .axis_z
                .ppm_to_index(c_ppm)
                .round()
                .clamp(0.0, (spec3d.nz().saturating_sub(1)) as f64) as usize;
            let ny = spec3d.ny();
            let mut plane = Vec::with_capacity(nx * ny);
            for iy in 0..ny {
                for ix in 0..nx {
                    plane.push(spec3d.get(ix, iy, iz));
                }
            }
            (&spec3d.axis_y, &spec3d.axis_z, plane)
        } else {
            let iy = spec3d
                .axis_y
                .ppm_to_index(c_ppm)
                .round()
                .clamp(0.0, (spec3d.ny().saturating_sub(1)) as f64) as usize;
            let nz = spec3d.nz();
            let mut plane = Vec::with_capacity(nx * nz);
            for iz in 0..nz {
                for ix in 0..nx {
                    plane.push(spec3d.get(ix, iy, iz));
                }
            }
            (&spec3d.axis_z, &spec3d.axis_y, plane)
        };
        let _ = c_axis;

        let (pos_rgba, neg_rgba) = default_overlay_palette(palette_index);
        Self {
            layer_id,
            spec_gen,
            spec_data: Arc::new(plane_data),
            axis_ppm: [
                spec3d.axis_x.first_ppm as f32,
                spec3d.axis_x.step_ppm as f32,
                n_axis.first_ppm as f32,
                n_axis.step_ppm as f32,
            ],
            dims: [nx as u32, n_axis.size as u32],
            base_level: base_level.max(1e-4),
            level_factor: 1.42,
            num_levels: 10,
            show_negative: true,
            pos_rgba,
            neg_rgba,
            ppm_nudge: [0.0, 0.0],
            visible: true,
        }
    }

    /// Extract a 2D `(axis_x, axis_y)` orthogonal plane at `(w_ppm, z_ppm)` from a row-major 4D
    /// hypercube `[iw * (nz * ny * nx) + iz * (ny * nx) + iy * nx + ix]`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_4d_plane(
        layer_id: u64,
        spec_gen: u64,
        axes: [&nmr_core::spectrum::AxisPpm; 4],
        cube_4d: &[f32],
        z_ppm: f64,
        w_ppm: f64,
        base_level: f32,
        palette_index: usize,
    ) -> Self {
        let (nx, ny, nz, nw) = (axes[0].size, axes[1].size, axes[2].size, axes[3].size);
        let iz = axes[2]
            .ppm_to_index(z_ppm)
            .round()
            .clamp(0.0, (nz.saturating_sub(1)) as f64) as usize;
        let iw = axes[3]
            .ppm_to_index(w_ppm)
            .round()
            .clamp(0.0, (nw.saturating_sub(1)) as f64) as usize;

        let plane_len = nx * ny;
        let base_off = (iw * nz + iz) * plane_len;
        let plane = if base_off + plane_len <= cube_4d.len() {
            cube_4d[base_off..base_off + plane_len].to_vec()
        } else {
            vec![0.0f32; plane_len]
        };

        let (pos_rgba, neg_rgba) = default_overlay_palette(palette_index);
        Self {
            layer_id,
            spec_gen,
            spec_data: Arc::new(plane),
            axis_ppm: [
                axes[0].first_ppm as f32,
                axes[0].step_ppm as f32,
                axes[1].first_ppm as f32,
                axes[1].step_ppm as f32,
            ],
            dims: [nx as u32, ny as u32],
            base_level: base_level.max(1e-4),
            level_factor: 1.42,
            num_levels: 10,
            show_negative: true,
            pos_rgba,
            neg_rgba,
            ppm_nudge: [0.0, 0.0],
            visible: true,
        }
    }

    /// Map a viewport coordinate `(view_x_ppm, view_y_ppm)` into this layer's native spectrum
    /// chemical-shift frame after subtracting `ppm_nudge`.
    #[inline]
    pub fn sample_ppm_for_view(&self, view_x_ppm: f32, view_y_ppm: f32) -> (f32, f32) {
        (
            view_x_ppm - self.ppm_nudge[0],
            view_y_ppm - self.ppm_nudge[1],
        )
    }

    /// Pack this layer and shared viewport bounds into a [`GpuContourLayerUniforms`] descriptor.
    #[inline]
    pub fn to_gpu_uniforms(
        &self,
        view_ppm: [f32; 4],
        canvas_size_px: [f32; 2],
        draw_background: bool,
    ) -> GpuContourLayerUniforms {
        GpuContourLayerUniforms {
            view_ppm,
            axis_ppm: self.axis_ppm,
            dims: [
                self.dims[0] as f32,
                self.dims[1] as f32,
                canvas_size_px[0],
                canvas_size_px[1],
            ],
            levels: [
                self.base_level,
                self.level_factor,
                self.num_levels as f32,
                if self.show_negative { 1.0 } else { 0.0 },
            ],
            pos_rgba: self.pos_rgba,
            neg_rgba: self.neg_rgba,
            layer_meta: [
                self.ppm_nudge[0],
                self.ppm_nudge[1],
                if draw_background { 1.0 } else { 0.0 },
                DEFAULT_APEX_WHITEN,
            ],
        }
    }
}

/// Synchronized `(1H, 15N, 13C)` cursor crosshair shared across 2D contour viewports and 3D strips.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyncCrosshair {
    pub h_ppm: f64,
    pub n_ppm: f64,
    pub c_ppm: Option<f64>,
    pub locked: bool,
}

impl Default for SyncCrosshair {
    fn default() -> Self {
        Self {
            h_ppm: 8.25,
            n_ppm: 120.0,
            c_ppm: None,
            locked: false,
        }
    }
}

#[allow(dead_code)]
impl SyncCrosshair {
    pub fn new(h_ppm: f64, n_ppm: f64, c_ppm: Option<f64>) -> Self {
        Self {
            h_ppm,
            n_ppm,
            c_ppm,
            locked: false,
        }
    }

    /// Toggle crosshair lock (`Space` accelerator) and return the new `locked` state.
    #[inline]
    pub fn toggle_lock(&mut self) -> bool {
        self.locked = !self.locked;
        self.locked
    }

    /// Update `(h_ppm, n_ppm)` from 2D viewport pointer hover unless locked.
    #[inline]
    pub fn update_from_2d_hover(&mut self, h_ppm: f64, n_ppm: f64) {
        if !self.locked {
            self.h_ppm = h_ppm;
            self.n_ppm = n_ppm;
        }
    }

    /// Update `(h_ppm, n_ppm, c_ppm)` from a 3D strip hover unless locked.
    #[inline]
    pub fn update_from_strip_hover(&mut self, h_ppm: Option<f64>, n_ppm: Option<f64>, c_ppm: f64) {
        if !self.locked {
            if let Some(h) = h_ppm {
                self.h_ppm = h;
            }
            if let Some(n) = n_ppm {
                self.n_ppm = n;
            }
            self.c_ppm = Some(c_ppm);
        }
    }

    /// Explicitly jump/snap crosshair coordinates (even when locked, e.g. on root selection).
    #[inline]
    pub fn snap_to(&mut self, h_ppm: f64, n_ppm: f64, c_ppm: Option<f64>) {
        self.h_ppm = h_ppm;
        self.n_ppm = n_ppm;
        self.c_ppm = c_ppm;
    }

    /// Project `(x_ppm, y_ppm)` onto screen pixels inside `rect` given `view_ppm = [left, right, top, bottom]`.
    #[inline]
    pub fn ppm_to_screen_2d(
        view_ppm: [f32; 4],
        rect: egui::Rect,
        x_ppm: f64,
        y_ppm: f64,
    ) -> egui::Pos2 {
        let dx = (view_ppm[1] - view_ppm[0]) as f64;
        let dy = (view_ppm[3] - view_ppm[2]) as f64;
        let u = if dx.abs() > 1e-12 {
            ((x_ppm - view_ppm[0] as f64) / dx) as f32
        } else {
            0.5
        };
        let v = if dy.abs() > 1e-12 {
            ((y_ppm - view_ppm[2] as f64) / dy) as f32
        } else {
            0.5
        };
        egui::pos2(
            rect.left() + u * rect.width(),
            rect.top() + v * rect.height(),
        )
    }

    /// Inverse-project a screen pixel `pos` inside `rect` back into `(x_ppm, y_ppm)`.
    #[inline]
    pub fn screen_to_ppm_2d(view_ppm: [f32; 4], rect: egui::Rect, pos: egui::Pos2) -> (f64, f64) {
        let u = if rect.width().abs() > 1e-6 {
            ((pos.x - rect.left()) / rect.width()) as f64
        } else {
            0.5
        };
        let v = if rect.height().abs() > 1e-6 {
            ((pos.y - rect.top()) / rect.height()) as f64
        } else {
            0.5
        };
        let x_ppm = view_ppm[0] as f64 + u * (view_ppm[1] as f64 - view_ppm[0] as f64);
        let y_ppm = view_ppm[2] as f64 + v * (view_ppm[3] as f64 - view_ppm[2] as f64);
        (x_ppm, y_ppm)
    }

    /// Project this crosshair's `(h_ppm, n_ppm)` onto a 2D `(1H, 15N)` viewport.
    #[inline]
    pub fn project_to_2d(&self, view_ppm: [f32; 4], rect: egui::Rect) -> egui::Pos2 {
        Self::ppm_to_screen_2d(view_ppm, rect, self.h_ppm, self.n_ppm)
    }

    /// Project a `13C` chemical shift onto a vertical strip axis `[plot_top_px, plot_bottom_px]`.
    #[inline]
    pub fn ppm_to_strip_y(
        c_top_ppm: f64,
        c_bottom_ppm: f64,
        plot_top_px: f32,
        plot_bottom_px: f32,
        c_ppm: f64,
    ) -> f32 {
        let span = c_bottom_ppm - c_top_ppm;
        let t = if span.abs() > 1e-12 {
            ((c_ppm - c_top_ppm) / span) as f32
        } else {
            0.5
        };
        plot_top_px + t.clamp(0.0, 1.0) * (plot_bottom_px - plot_top_px)
    }

    /// Inverse-project a vertical strip pixel `y_px` back into `13C` ppm.
    #[inline]
    pub fn strip_y_to_ppm(
        c_top_ppm: f64,
        c_bottom_ppm: f64,
        plot_top_px: f32,
        plot_bottom_px: f32,
        y_px: f32,
    ) -> f64 {
        let h = plot_bottom_px - plot_top_px;
        let t = if h.abs() > 1e-6 {
            ((y_px - plot_top_px) / h) as f64
        } else {
            0.5
        };
        c_top_ppm + t * (c_bottom_ppm - c_top_ppm)
    }

    /// Project `self.c_ppm` onto a vertical `13C` strip axis when `c_ppm` is present and inside bounds.
    #[inline]
    pub fn project_c_to_strip_y(
        &self,
        c_top_ppm: f64,
        c_bottom_ppm: f64,
        plot_top_px: f32,
        plot_bottom_px: f32,
    ) -> Option<f32> {
        let c = self.c_ppm?;
        let lo = c_top_ppm.min(c_bottom_ppm);
        let hi = c_top_ppm.max(c_bottom_ppm);
        if (lo..=hi).contains(&c) {
            Some(Self::ppm_to_strip_y(
                c_top_ppm,
                c_bottom_ppm,
                plot_top_px,
                plot_bottom_px,
                c,
            ))
        } else {
            None
        }
    }

    /// Render full-canvas synchronized hairlines and lock indicator on a 2D spectrum viewport.
    pub fn paint_on_2d(&self, painter: &egui::Painter, view_ppm: [f32; 4], rect: egui::Rect) {
        let pt = self.project_to_2d(view_ppm, rect);
        let col = if self.locked {
            egui::Color32::from_rgba_unmultiplied(255, 215, 90, 210)
        } else {
            egui::Color32::from_rgba_unmultiplied(165, 215, 255, 130)
        };
        let stroke = egui::Stroke::new(if self.locked { 1.3 } else { 0.9 }, col);
        if (rect.left()..=rect.right()).contains(&pt.x) {
            painter.line_segment(
                [
                    egui::pos2(pt.x, rect.top()),
                    egui::pos2(pt.x, rect.bottom()),
                ],
                stroke,
            );
        }
        if (rect.top()..=rect.bottom()).contains(&pt.y) {
            painter.line_segment(
                [
                    egui::pos2(rect.left(), pt.y),
                    egui::pos2(rect.right(), pt.y),
                ],
                stroke,
            );
        }
    }
}

struct CachedSpecBuffer {
    spec_gen: u64,
    spec_len: usize,
    last_used_tick: u64,
    buffer: wgpu::Buffer,
}

struct PassDrawSlot {
    uniform_buf: wgpu::Buffer,
    bound_key: Option<(u64, u64, usize)>,
    bind_group: wgpu::BindGroup,
}

impl PassDrawSlot {
    fn new(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        dummy_spec_buf: &wgpu::Buffer,
    ) -> Self {
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-contour-layer-uniform"),
            size: std::mem::size_of::<GpuContourLayerUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-contour-layer-bg-init"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_spec_buf.as_entire_binding(),
                },
            ],
        });
        Self {
            uniform_buf,
            bound_key: None,
            bind_group,
        }
    }
}

struct BatchState {
    pass_slots: Vec<PassDrawSlot>,
    active_passes: usize,
}

pub struct ContourPipelineResource {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    dummy_spec_buf: wgpu::Buffer,
    spec_cache: BTreeMap<u64, CachedSpecBuffer>,
    batches: BTreeMap<u64, BatchState>,
    tick: u64,
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
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            GpuContourLayerUniforms,
                        >()
                            as u64),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let dummy = [0.0f32; 4];
        let dummy_spec_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-contour-spec-dummy"),
            contents: bytemuck::cast_slice(&dummy),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            pipeline,
            bgl,
            dummy_spec_buf,
            spec_cache: BTreeMap::new(),
            batches: BTreeMap::new(),
            tick: 0,
        }
    }

    fn ensure_spec_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer_id: u64,
        spec_gen: u64,
        spec_data: &[f32],
    ) {
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;

        if spec_data.is_empty() {
            return;
        }

        if let Some(entry) = self.spec_cache.get_mut(&layer_id) {
            entry.last_used_tick = tick;
            if entry.spec_gen == spec_gen && entry.spec_len == spec_data.len() {
                return;
            }
            if entry.spec_len == spec_data.len() {
                queue.write_buffer(&entry.buffer, 0, bytemuck::cast_slice(spec_data));
                entry.spec_gen = spec_gen;
                return;
            }
        }

        if !self.spec_cache.contains_key(&layer_id)
            && self.spec_cache.len() >= MAX_CACHED_SPEC_BUFFERS
        {
            if let Some((&evict_id, _)) =
                self.spec_cache.iter().min_by_key(|(_, v)| v.last_used_tick)
            {
                self.spec_cache.remove(&evict_id);
                for batch in self.batches.values_mut() {
                    for slot in &mut batch.pass_slots {
                        if slot.bound_key.is_some_and(|(id, _, _)| id == evict_id) {
                            slot.bound_key = None;
                        }
                    }
                }
            }
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-contour-spec-layer"),
            contents: bytemuck::cast_slice(spec_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.spec_cache.insert(
            layer_id,
            CachedSpecBuffer {
                spec_gen,
                spec_len: spec_data.len(),
                last_used_tick: tick,
                buffer,
            },
        );
    }

    /// Backward-compatible single-layer update used by [`ContourCallback`] (batch `0`).
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &ContourUniforms,
        spec_gen: u64,
        spec_data: &[f32],
    ) {
        const LEGACY_BATCH_ID: u64 = 0;
        const LEGACY_LAYER_ID: u64 = 0;

        self.ensure_spec_buffer(device, queue, LEGACY_LAYER_ID, spec_gen, spec_data);
        let gpu_uni = uniforms.to_gpu_layer(true);

        let batch = self
            .batches
            .entry(LEGACY_BATCH_ID)
            .or_insert_with(|| BatchState {
                pass_slots: Vec::new(),
                active_passes: 0,
            });
        if batch.pass_slots.is_empty() {
            batch
                .pass_slots
                .push(PassDrawSlot::new(device, &self.bgl, &self.dummy_spec_buf));
        }

        let slot = &mut batch.pass_slots[0];
        queue.write_buffer(&slot.uniform_buf, 0, bytemuck::bytes_of(&gpu_uni));

        let desired_key = if spec_data.is_empty() {
            (u64::MAX, 0, 0)
        } else {
            (LEGACY_LAYER_ID, spec_gen, spec_data.len())
        };
        if slot.bound_key != Some(desired_key) {
            let spec_buf_ref = self
                .spec_cache
                .get(&LEGACY_LAYER_ID)
                .filter(|_| !spec_data.is_empty())
                .map(|c| &c.buffer)
                .unwrap_or(&self.dummy_spec_buf);
            slot.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nmr-contour-bg-legacy"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: slot.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: spec_buf_ref.as_entire_binding(),
                    },
                ],
            });
            slot.bound_key = Some(desired_key);
        }
        batch.active_passes = 1;
    }

    /// Multi-layer ordered overlay preparation: uploads only modified spectrum buffers and writes
    /// compact 112-byte per-pass uniforms in painter's order (`0..visible_layers`).
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn update_layers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_id: u64,
        view_ppm: [f32; 4],
        canvas_size_px: [f32; 2],
        layers: &[OverlayLayerSpec],
        draw_opaque_bg: bool,
    ) {
        for layer in layers.iter().filter(|l| l.visible) {
            self.ensure_spec_buffer(
                device,
                queue,
                layer.layer_id,
                layer.spec_gen,
                &layer.spec_data,
            );
        }

        let visible_count = layers.iter().filter(|l| l.visible).count();
        let needed_passes = if visible_count == 0 && draw_opaque_bg {
            1
        } else {
            visible_count
        };

        let batch = self.batches.entry(batch_id).or_insert_with(|| BatchState {
            pass_slots: Vec::new(),
            active_passes: 0,
        });
        while batch.pass_slots.len() < needed_passes {
            batch
                .pass_slots
                .push(PassDrawSlot::new(device, &self.bgl, &self.dummy_spec_buf));
        }

        if visible_count == 0 {
            if draw_opaque_bg {
                let bg_uni = GpuContourLayerUniforms {
                    view_ppm,
                    axis_ppm: [10.0, -0.01, 100.0, 0.1],
                    dims: [2.0, 2.0, canvas_size_px[0], canvas_size_px[1]],
                    levels: [1.0, 1.4, 0.0, 0.0],
                    pos_rgba: OVERLAY_PALETTES[0].0,
                    neg_rgba: OVERLAY_PALETTES[0].1,
                    layer_meta: [0.0, 0.0, 1.0, DEFAULT_APEX_WHITEN],
                };
                let slot = &mut batch.pass_slots[0];
                queue.write_buffer(&slot.uniform_buf, 0, bytemuck::bytes_of(&bg_uni));
                let dummy_key = (u64::MAX, 0, 0);
                if slot.bound_key != Some(dummy_key) {
                    slot.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("nmr-contour-bg-empty"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: slot.uniform_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.dummy_spec_buf.as_entire_binding(),
                            },
                        ],
                    });
                    slot.bound_key = Some(dummy_key);
                }
                batch.active_passes = 1;
            } else {
                batch.active_passes = 0;
            }
            return;
        }

        for (pass_idx, layer) in layers.iter().filter(|l| l.visible).enumerate() {
            let draw_bg = draw_opaque_bg && pass_idx == 0;
            let gpu_uni = layer.to_gpu_uniforms(view_ppm, canvas_size_px, draw_bg);
            let slot = &mut batch.pass_slots[pass_idx];
            queue.write_buffer(&slot.uniform_buf, 0, bytemuck::bytes_of(&gpu_uni));

            let desired_key = if layer.spec_data.is_empty() {
                (u64::MAX, 0, 0)
            } else {
                (layer.layer_id, layer.spec_gen, layer.spec_data.len())
            };
            if slot.bound_key != Some(desired_key) {
                let spec_buf_ref = self
                    .spec_cache
                    .get(&layer.layer_id)
                    .filter(|_| !layer.spec_data.is_empty())
                    .map(|c| &c.buffer)
                    .unwrap_or(&self.dummy_spec_buf);
                slot.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("nmr-contour-bg-overlay"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: slot.uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: spec_buf_ref.as_entire_binding(),
                        },
                    ],
                });
                slot.bound_key = Some(desired_key);
            }
        }
        batch.active_passes = visible_count;
    }

    /// Issue draw calls for `batch_id` onto an active `wgpu::RenderPass`.
    pub fn draw_batch(&self, render_pass: &mut wgpu::RenderPass<'_>, batch_id: u64) {
        let Some(batch) = self.batches.get(&batch_id) else {
            return;
        };
        if batch.active_passes == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        for slot in batch.pass_slots.iter().take(batch.active_passes) {
            render_pass.set_bind_group(0, &slot.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }

    /// Number of cached GPU spectrum storage buffers currently resident in VRAM.
    #[allow(dead_code)]
    pub fn cached_layer_count(&self) -> usize {
        self.spec_cache.len()
    }
}

/// Backward-compatible single-spectrum `egui_wgpu` paint callback.
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
        res.draw_batch(render_pass, 0);
    }
}

/// Ordered multi-spectrum 2D / 3D-slice GPU contour paint callback composited via hardware
/// `BlendState::ALPHA_BLENDING`.
#[allow(dead_code)]
pub struct MultiContourCallback {
    batch_id: u64,
    pub view_ppm: [f32; 4],
    pub canvas_size_px: [f32; 2],
    pub layers: Vec<OverlayLayerSpec>,
    pub draw_opaque_bg: bool,
}

#[allow(dead_code)]
impl MultiContourCallback {
    pub fn new(
        view_ppm: [f32; 4],
        canvas_size_px: [f32; 2],
        layers: Vec<OverlayLayerSpec>,
    ) -> Self {
        let mut id = NEXT_MULTI_CALLBACK_BATCH_ID.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            id = NEXT_MULTI_CALLBACK_BATCH_ID.fetch_add(1, Ordering::Relaxed);
        }
        Self {
            batch_id: id,
            view_ppm,
            canvas_size_px,
            layers,
            draw_opaque_bg: true,
        }
    }

    pub fn with_opaque_background(mut self, draw_opaque_bg: bool) -> Self {
        self.draw_opaque_bg = draw_opaque_bg;
        self
    }

    #[inline]
    pub fn batch_id(&self) -> u64 {
        self.batch_id
    }
}

impl egui_wgpu::CallbackTrait for MultiContourCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<ContourPipelineResource>() {
            res.update_layers(
                device,
                queue,
                self.batch_id,
                self.view_ppm,
                self.canvas_size_px,
                &self.layers,
                self.draw_opaque_bg,
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
        res.draw_batch(render_pass, self.batch_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::spectrum::AxisPpm;

    #[test]
    fn test_gpu_uniform_packing_and_overlay_layer_invariants() {
        assert_eq!(std::mem::size_of::<ContourUniforms>(), 64);
        assert_eq!(std::mem::size_of::<GpuContourLayerUniforms>(), 112);
        assert_eq!(std::mem::size_of::<GpuContourLayerUniforms>() % 16, 0);

        let ax_x = AxisPpm::from_bounds("1H", 128, 10.5, 6.0, 600.13);
        let ax_y = AxisPpm::from_bounds("15N", 64, 134.0, 104.0, 60.81);
        let spec2d = Spectrum2D {
            name: "15N-HSQC".to_string(),
            axis_x: ax_x,
            axis_y: ax_y,
            data: vec![0.0f32; 128 * 64],
        };

        let mut layer = OverlayLayerSpec::from_spectrum_2d(
            7,
            42,
            &spec2d,
            Arc::new(spec2d.data.clone()),
            12.5,
            1,
        );
        layer.ppm_nudge = [0.035, -0.25];
        layer.show_negative = false;

        let (sx, sy) = layer.sample_ppm_for_view(8.235, 119.75);
        assert!((sx - 8.20).abs() < 1e-5);
        assert!((sy - 120.00).abs() < 1e-5);

        let view_ppm = [10.5, 6.0, 104.0, 134.0];
        let gpu_u = layer.to_gpu_uniforms(view_ppm, [1200.0, 800.0], false);
        assert_eq!(gpu_u.view_ppm, view_ppm);
        assert_eq!(gpu_u.dims, [128.0, 64.0, 1200.0, 800.0]);
        assert_eq!(gpu_u.levels, [12.5, 1.42, 10.0, 0.0]);
        assert_eq!(gpu_u.pos_rgba, OVERLAY_PALETTES[1].0);
        assert_eq!(gpu_u.neg_rgba, OVERLAY_PALETTES[1].1);
        assert_eq!(gpu_u.layer_meta[0..3], [0.035, -0.25, 0.0]);

        let raw_bytes = bytemuck::bytes_of(&gpu_u);
        assert_eq!(raw_bytes.len(), 112);
        let round_trip: &GpuContourLayerUniforms = bytemuck::from_bytes(raw_bytes);
        assert_eq!(*round_trip, gpu_u);
    }

    #[test]
    fn test_overlay_3d_hn_plane_extraction() {
        let ax_x = AxisPpm::from_bounds("1H", 16, 10.0, 7.0, 600.0);
        let ax_y = AxisPpm::from_bounds("15N", 12, 130.0, 108.0, 60.8);
        let ax_z = AxisPpm::from_bounds("13C", 20, 182.0, 168.0, 150.9);
        let mut data = vec![0.0f32; 16 * 12 * 20];
        let target_iz = ax_z.ppm_to_index(175.0).round() as usize;
        data[(target_iz * 12 + 5) * 16 + 9] = 321.5;

        let spec3d = Spectrum3D {
            name: "HNCO".to_string(),
            exp: "HNCO".to_string(),
            axis_x: ax_x,
            axis_y: ax_y,
            axis_z: ax_z,
            data,
        };

        let plane_layer = OverlayLayerSpec::from_spectrum_3d_hn_plane(3, 1, &spec3d, 175.0, 8.0, 2);
        assert_eq!(plane_layer.dims, [16, 12]);
        assert_eq!(plane_layer.spec_data.len(), 16 * 12);
        assert!((plane_layer.spec_data[5 * 16 + 9] - 321.5).abs() < 1e-5);
        assert_eq!(plane_layer.pos_rgba, OVERLAY_PALETTES[2].0);

        // 4D hypercube orthogonal plane extraction
        let ax_w = AxisPpm::from_bounds("1H_ind", 6, 9.5, 1.0, 600.0);
        let mut data4 = vec![0.0f32; 16 * 12 * 20 * 6];
        let target_iw = ax_w.ppm_to_index(4.4).round() as usize;
        data4[((target_iw * 20 + target_iz) * 12 + 3) * 16 + 7] = -188.0;
        let plane4 = OverlayLayerSpec::from_4d_plane(
            4,
            1,
            [&spec3d.axis_x, &spec3d.axis_y, &spec3d.axis_z, &ax_w],
            &data4,
            175.0,
            4.4,
            8.0,
            3,
        );
        assert_eq!(plane4.dims, [16, 12]);
        assert!((plane4.spec_data[3 * 16 + 7] - (-188.0)).abs() < 1e-5);
    }

    #[test]
    fn test_sync_crosshair_lock_and_coordinate_transforms() {
        let mut xh = SyncCrosshair::new(8.42, 118.6, Some(54.3));
        assert!(!xh.locked);

        let view = [10.4f32, 6.2, 104.0, 134.0];
        let rect = egui::Rect::from_min_size(egui::pos2(40.0, 25.0), egui::vec2(840.0, 600.0));
        let screen_pt = xh.project_to_2d(view, rect);
        let (h_back, n_back) = SyncCrosshair::screen_to_ppm_2d(view, rect, screen_pt);
        assert!((h_back - 8.42).abs() < 1e-5);
        assert!((n_back - 118.6).abs() < 1e-4);

        let y_strip = xh
            .project_c_to_strip_y(13.0, 74.0, 50.0, 355.0)
            .expect("54.3 ppm inside aliphatic strip");
        let c_back = SyncCrosshair::strip_y_to_ppm(13.0, 74.0, 50.0, 355.0, y_strip);
        assert!((c_back - 54.3).abs() < 1e-4);
        assert!(xh.project_c_to_strip_y(168.0, 182.0, 10.0, 45.0).is_none());

        assert!(xh.toggle_lock());
        xh.update_from_2d_hover(7.10, 110.0);
        xh.update_from_strip_hover(Some(7.10), Some(110.0), 31.2);
        assert!((xh.h_ppm - 8.42).abs() < 1e-9);
        assert!((xh.n_ppm - 118.6).abs() < 1e-9);
        assert_eq!(xh.c_ppm, Some(54.3));

        assert!(!xh.toggle_lock());
        xh.update_from_2d_hover(7.85, 122.4);
        xh.update_from_strip_hover(None, None, 31.2);
        assert!((xh.h_ppm - 7.85).abs() < 1e-9);
        assert!((xh.n_ppm - 122.4).abs() < 1e-9);
        assert_eq!(xh.c_ppm, Some(31.2));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_wgpu_4_layer_2048x1024_alpha_overlay_and_60fps_pan_zoom() {
        let Ok(gpu) = nmr_gpu::GpuEngine::new_headless() else {
            eprintln!("Skipping headless wgpu overlay render test: no GPU adapter available");
            return;
        };

        let target_fmt = wgpu::TextureFormat::Rgba8Unorm;
        let mut res = ContourPipelineResource::new(&gpu.device, target_fmt);

        let (nx, ny) = (2048usize, 1024usize);
        let ax_x = AxisPpm::from_bounds("1H", nx, 10.5, 6.0, 600.13);
        let ax_y = AxisPpm::from_bounds("15N", ny, 134.0, 104.0, 60.81);

        // Synthesize 4 distinct 2048x1024 2D spectra with non-overlapping Gaussian peaks
        let peak_coords = [(8.80, 114.0), (8.20, 118.0), (7.60, 122.0), (7.10, 126.0)];
        let layers: Vec<OverlayLayerSpec> = peak_coords
            .iter()
            .enumerate()
            .map(|(idx, &(ph, pn))| {
                let mut buf = vec![0.0f32; nx * ny];
                let cx = ax_x.ppm_to_index(ph).round() as isize;
                let cy = ax_y.ppm_to_index(pn).round() as isize;
                for dy in -8isize..=8 {
                    for dx in -8isize..=8 {
                        let ix = (cx + dx).clamp(0, nx as isize - 1) as usize;
                        let iy = (cy + dy).clamp(0, ny as isize - 1) as usize;
                        let r2 = ((dx * dx + dy * dy) as f32) / 10.0;
                        buf[iy * nx + ix] = 120.0 * (-0.5 * r2).exp();
                    }
                }
                let spec = Spectrum2D {
                    name: format!("Layer-{idx}"),
                    axis_x: ax_x.clone(),
                    axis_y: ax_y.clone(),
                    data: Vec::new(),
                };
                OverlayLayerSpec::from_spectrum_2d(
                    (idx + 1) as u64,
                    1,
                    &spec,
                    Arc::new(buf),
                    10.0,
                    idx,
                )
            })
            .collect();

        let (vp_w, vp_h) = (640u32, 480u32);
        let target_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-overlay-target"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let base_view = [10.5f32, 6.0, 104.0, 134.0];

        let bytes_per_row = (vp_w * 4).div_ceil(256) * 256;
        let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-overlay-readback"),
            size: (bytes_per_row * vp_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let render_and_read = |res: &mut ContourPipelineResource,
                               cur_layers: &[OverlayLayerSpec]| {
            res.update_layers(
                &gpu.device,
                &gpu.queue,
                99,
                base_view,
                [vp_w as f32, vp_h as f32],
                cur_layers,
                true,
            );
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("readback-overlay-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                res.draw_batch(&mut rpass, 99);
            }
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(vp_h),
                    },
                },
                wgpu::Extent3d {
                    width: vp_w,
                    height: vp_h,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(Some(enc.finish()));
            let slice = readback_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let data = slice.get_mapped_range().expect("mapped readback").to_vec();
            readback_buf.unmap();
            data
        };

        let patch_max_luma = |img: &[u8], h_ppm: f64, n_ppm: f64| -> u8 {
            let rect =
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(vp_w as f32, vp_h as f32));
            let pt = SyncCrosshair::ppm_to_screen_2d(base_view, rect, h_ppm, n_ppm);
            let cx = pt.x.round() as i32;
            let cy = pt.y.round() as i32;
            let mut best = 0u8;
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let px = (cx + dx).clamp(0, vp_w as i32 - 1) as u32;
                    let py = (cy + dy).clamp(0, vp_h as i32 - 1) as u32;
                    let off = (py * bytes_per_row + px * 4) as usize;
                    let m = img[off].max(img[off + 1]).max(img[off + 2]);
                    best = best.max(m);
                }
            }
            best
        };

        // 1. All 4 layers visible -> all 4 non-overlapping peak patches illuminate via alpha blend
        let img_all = render_and_read(&mut res, &layers);
        assert_eq!(res.cached_layer_count(), 4);
        for &(ph, pn) in &peak_coords {
            let luma = patch_max_luma(&img_all, ph, pn);
            assert!(
                luma >= 90,
                "Expected contour illumination >= 90 near ({ph}, {pn}), got {luma}"
            );
        }

        // 2. Hide layer 1 & nudge layer 0 by (+0.30 H, +3.0 N) -> verify visibility & shift nudge!
        let mut tweaked = layers.clone();
        tweaked[1].visible = false;
        tweaked[0].ppm_nudge = [0.30, 3.0];
        let img_tweaked = render_and_read(&mut res, &tweaked);
        assert!(
            patch_max_luma(&img_tweaked, peak_coords[1].0, peak_coords[1].1) <= 40,
            "Hidden layer 1 must revert to dark slate background"
        );
        assert!(
            patch_max_luma(
                &img_tweaked,
                peak_coords[0].0 + 0.30,
                peak_coords[0].1 + 3.0
            ) >= 90,
            "Nudged layer 0 peak must appear at shifted (+0.30 H, +3.0 N) coordinates"
        );
        assert!(
            patch_max_luma(&img_tweaked, peak_coords[2].0, peak_coords[2].1) >= 90,
            "Remaining visible layers must remain composited"
        );

        // Measure 24 frames of animated pan/zoom across all 4 simultaneous 2048x1024 overlays
        let frames = 24usize;
        let t0 = std::time::Instant::now();
        for f in 0..frames {
            let pan = (f as f32) * 0.01;
            let zoom_view = [10.5 - pan, 6.0 + pan * 0.5, 104.0 + pan * 2.0, 134.0 - pan];
            res.update_layers(
                &gpu.device,
                &gpu.queue,
                99,
                zoom_view,
                [vp_w as f32, vp_h as f32],
                &layers,
                true,
            );
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("timed-overlay-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                res.draw_batch(&mut rpass, 99);
            }
            gpu.queue.submit(Some(enc.finish()));
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let mean_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (frames as f64);
        assert!(
            mean_ms < 16.67,
            "Expected >= 60 fps (< 16.67 ms/frame) for 4x 2048x1024 overlays, got {mean_ms:.3} ms/frame"
        );
    }
}
