// Clean-room WGSL 2D NMR contour renderer for eframe / egui_wgpu.
// Provenance: Original GPU contour shader computing bilinear spectrum interpolation
// and screen-space gradient anti-aliased positive/negative geometric contour lines.

struct ContourUniforms {
    // (h_left_ppm, h_right_ppm, n_top_ppm, n_bottom_ppm)
    view_ppm: vec4<f32>,
    // (first_x_ppm, step_x_ppm, first_y_ppm, step_y_ppm)
    axis_ppm: vec4<f32>,
    // (nx, ny, canvas_width_px, canvas_height_px)
    dims: vec4<f32>,
    // (base_level, level_factor, num_levels, show_negative)
    levels: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uni: ContourUniforms;
@group(0) @binding(1) var<storage, read> spec_data: array<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle covering clip space [-1, 1]^2
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    // uv.x: 0 (left) -> 1 (right), uv.y: 0 (top) -> 1 (bottom)
    out.uv = vec2<f32>(0.5 * (p.x + 1.0), 0.5 * (1.0 - p.y));
    return out;
}

fn sample_bilinear(uv: vec2<f32>) -> f32 {
    let h_ppm = mix(uni.view_ppm.x, uni.view_ppm.y, uv.x);
    let n_ppm = mix(uni.view_ppm.z, uni.view_ppm.w, uv.y);

    let fx = (h_ppm - uni.axis_ppm.x) / uni.axis_ppm.y;
    let fy = (n_ppm - uni.axis_ppm.z) / uni.axis_ppm.w;

    let nx = i32(uni.dims.x);
    let ny = i32(uni.dims.y);
    if (fx < 0.0 || fy < 0.0 || fx > f32(nx - 1) || fy > f32(ny - 1)) {
        return 0.0;
    }

    let x0 = clamp(i32(floor(fx)), 0, nx - 1);
    let y0 = clamp(i32(floor(fy)), 0, ny - 1);
    let x1 = min(x0 + 1, nx - 1);
    let y1 = min(y0 + 1, ny - 1);
    let tx = fx - floor(fx);
    let ty = fy - floor(fy);

    let v00 = spec_data[y0 * nx + x0];
    let v10 = spec_data[y0 * nx + x1];
    let v01 = spec_data[y1 * nx + x0];
    let v11 = spec_data[y1 * nx + x1];

    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Deep slate NMR canvas background with subtle grid
    var bg = vec3<f32>(0.055, 0.070, 0.098);

    let h_ppm = mix(uni.view_ppm.x, uni.view_ppm.y, in.uv.x);
    let n_ppm = mix(uni.view_ppm.z, uni.view_ppm.w, in.uv.y);

    let dh_px = abs(uni.view_ppm.x - uni.view_ppm.y) / max(uni.dims.z, 1.0);
    let dn_px = abs(uni.view_ppm.z - uni.view_ppm.w) / max(uni.dims.w, 1.0);
    let grid_h = abs(fract(h_ppm * 2.0 + 0.5) - 0.5) / (2.0 * max(dh_px, 1e-6));
    let grid_n = abs(fract(n_ppm * 0.2 + 0.5) - 0.5) / (0.2 * max(dn_px, 1e-6));
    if (min(grid_h, grid_n) < 0.75) {
        bg = bg + vec3<f32>(0.025, 0.032, 0.045);
    }

    let val = sample_bilinear(in.uv);
    let du = vec2<f32>(1.0 / max(uni.dims.z, 1.0), 0.0);
    let dv = vec2<f32>(0.0, 1.0 / max(uni.dims.w, 1.0));
    let gx = sample_bilinear(in.uv + du) - sample_bilinear(in.uv - du);
    let gy = sample_bilinear(in.uv + dv) - sample_bilinear(in.uv - dv);
    let grad = max(0.5 * length(vec2<f32>(gx, gy)), 1e-4);

    let base = max(uni.levels.x, 1e-4);
    let factor = max(uni.levels.y, 1.05);
    let n_levels = i32(uni.levels.z);
    let show_neg = uni.levels.w > 0.5;

    let abs_v = abs(val);
    if (abs_v < base * 0.65 || (val < 0.0 && !show_neg)) {
        return vec4<f32>(bg, 1.0);
    }

    var line_alpha: f32 = 0.0;
    var level_norm: f32 = 0.0;
    var lvl = base;
    for (var i: i32 = 0; i < n_levels; i = i + 1) {
        let dist_px = abs(abs_v - lvl) / grad;
        let a = 1.0 - smoothstep(0.45, 1.45, dist_px);
        if (a > line_alpha) {
            line_alpha = a;
            level_norm = f32(i) / max(f32(n_levels - 1), 1.0);
        }
        lvl = lvl * factor;
    }

    // Subtle soft glow inside high contours so dense peaks remain readable when zoomed out
    let fill_glow = clamp((log(max(abs_v / base, 1.0)) / log(pow(factor, f32(n_levels)))) * 0.22, 0.0, 0.25);

    var contour_col: vec3<f32>;
    if (val >= 0.0) {
        // Positive contours: deep sky cyan -> bright ice white-cyan at peak apex
        contour_col = mix(vec3<f32>(0.12, 0.68, 0.96), vec3<f32>(0.68, 0.96, 1.0), level_norm);
    } else {
        // Negative contours: warm coral-rose -> bright amber-coral
        contour_col = mix(vec3<f32>(0.95, 0.36, 0.42), vec3<f32>(1.0, 0.72, 0.52), level_norm);
    }

    let alpha = clamp(max(line_alpha, fill_glow), 0.0, 1.0);
    let final_rgb = mix(bg, contour_col, alpha);
    return vec4<f32>(final_rgb, 1.0);
}
