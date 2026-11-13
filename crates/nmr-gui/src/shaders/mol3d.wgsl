// Clean-room WebGPU (WGSL 1.0) 3D biomolecular structure shader (`T12.1` & `T12.2`).
//
// Provenance: Original clean-room shader implementing hardware-instanced Blinn-Phong + hemisphere
// rim-lit biomolecular cartoon ribbons (Carson & Bugg, *J. Mol. Graph.* 1986, 4, 121-122),
// van-der-Waals / ball-and-stick spheres, half-bond CPK & NOE/PRE restraint cylinders, view-space
// Z-slab clipping, depth-cued atmospheric fog, and offscreen MSAA resolve blitting.

struct Mol3dUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    bg_color: vec4<f32>,
    // x = slab_near_dist (> 0), y = slab_far_dist (> slab_near), z = fog_start_dist, w = fog_end_dist
    slab_fog: vec4<f32>,
    // xyz = normalized view-space key light direction, w = specular shininess exponent
    light_dir: vec4<f32>,
    // x = ambient_weight, y = diffuse_weight, z = specular_weight, w = is_orthographic (0 or 1)
    params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uni: Mol3dUniforms;

struct LitVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) cyl_param: vec2<f32>,
};

@vertex
fn vs_ribbon(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> LitVarying {
    let vp = uni.view * vec4<f32>(pos, 1.0);
    let vn = (uni.view * vec4<f32>(normal, 0.0)).xyz;
    var out: LitVarying;
    out.clip_pos = uni.proj * vp;
    out.view_pos = vp.xyz;
    out.view_normal = vn;
    out.color = color;
    out.cyl_param = vec2<f32>(0.0, 0.0);
    return out;
}

@vertex
fn vs_sphere(
    @location(0) local_pos: vec3<f32>,
    @location(1) center_radius: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> LitVarying {
    let r = center_radius.w;
    if (r <= 0.0 || color.a <= 0.001) {
        var deg: LitVarying;
        deg.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        deg.view_pos = vec3<f32>(0.0, 0.0, 0.0);
        deg.view_normal = vec3<f32>(0.0, 0.0, 1.0);
        deg.color = vec4<f32>(0.0);
        deg.cyl_param = vec2<f32>(0.0, 0.0);
        return deg;
    }
    let n_world = normalize(local_pos);
    let p_world = center_radius.xyz + n_world * r;
    let vp = uni.view * vec4<f32>(p_world, 1.0);
    let vn = (uni.view * vec4<f32>(n_world, 0.0)).xyz;
    var out: LitVarying;
    out.clip_pos = uni.proj * vp;
    out.view_pos = vp.xyz;
    out.view_normal = vn;
    out.color = color;
    out.cyl_param = vec2<f32>(0.0, 0.0);
    return out;
}

@vertex
fn vs_cylinder(
    @location(0) radial_t: vec3<f32>,
    @location(1) p0_rad: vec4<f32>,
    @location(2) p1_flags: vec4<f32>,
    @location(3) color0: vec4<f32>,
    @location(4) color1: vec4<f32>,
) -> LitVarying {
    let delta = p1_flags.xyz - p0_rad.xyz;
    let len = length(delta);
    let r = p0_rad.w;
    if (len < 1e-5 || r <= 0.0) {
        var deg: LitVarying;
        deg.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        deg.view_pos = vec3<f32>(0.0, 0.0, 0.0);
        deg.view_normal = vec3<f32>(0.0, 0.0, 1.0);
        deg.color = vec4<f32>(0.0);
        deg.cyl_param = vec2<f32>(0.0, 0.0);
        return deg;
    }
    let w_axis = delta / len;
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(w_axis.x) > 0.85);
    let u_axis = normalize(cross(helper, w_axis));
    let v_axis = cross(w_axis, u_axis);
    let n_world = normalize(radial_t.x * u_axis + radial_t.y * v_axis);
    let p_world = mix(p0_rad.xyz, p1_flags.xyz, radial_t.z) + n_world * r;

    let vp = uni.view * vec4<f32>(p_world, 1.0);
    let vn = (uni.view * vec4<f32>(n_world, 0.0)).xyz;
    var out: LitVarying;
    out.clip_pos = uni.proj * vp;
    out.view_pos = vp.xyz;
    out.view_normal = vn;
    // Pass color0 in out.color and pack color1 via fragment interpolation or split-vertex:
    // radial_t.z is strictly 0.0 or 1.0 on cylinder caps, so we store color0 at z <= 0.5 and color1 at z > 0.5
    // using duplicate mid-ring vertices at t = 0.50 on the unit cylinder template for crisp CPK half-bonds!
    out.color = select(color0, color1, radial_t.z > 0.5);
    out.cyl_param = vec2<f32>(abs(radial_t.z), p1_flags.w);
    return out;
}

fn shade_blinn_phong(
    view_pos: vec3<f32>,
    raw_normal: vec3<f32>,
    base_rgba: vec4<f32>,
    front_facing: bool,
) -> vec4<f32> {
    let depth = -view_pos.z;
    if (depth < uni.slab_fog.x || depth > uni.slab_fog.y) {
        discard;
    }

    var n = normalize(raw_normal);
    if (!front_facing) {
        n = -n;
    }

    let v_dir = select(normalize(-view_pos), vec3<f32>(0.0, 0.0, 1.0), uni.params.w > 0.5);
    let l_key = normalize(uni.light_dir.xyz);
    let l_fill = normalize(vec3<f32>(-0.48, -0.32, 0.82));

    let ndotl_key = max(dot(n, l_key), 0.0);
    let ndotl_fill = max(dot(n, l_fill), 0.0) * 0.28;
    let hemi = (0.5 * n.y + 0.5) * 0.14;

    let h_vec = normalize(l_key + v_dir);
    let spec = pow(max(dot(n, h_vec), 0.0), max(uni.light_dir.w, 4.0)) * uni.params.z;
    let rim = pow(1.0 - max(dot(n, v_dir), 0.0), 3.0) * 0.12;

    let lit_rgb = base_rgba.rgb * (uni.params.x + uni.params.y * (ndotl_key + ndotl_fill) + hemi)
        + vec3<f32>(spec + rim);

    let fog_span = max(uni.slab_fog.w - uni.slab_fog.z, 1e-3);
    let fog_t = clamp((depth - uni.slab_fog.z) / fog_span, 0.0, 0.68);
    let final_rgb = clamp(mix(lit_rgb, uni.bg_color.rgb, fog_t), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(final_rgb, base_rgba.a);
}

@fragment
fn fs_lit(
    in: LitVarying,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    return shade_blinn_phong(in.view_pos, in.view_normal, in.color, front_facing);
}

@fragment
fn fs_cylinder(
    in: LitVarying,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    if (in.cyl_param.y >= 2.0) {
        let phase = fract(in.cyl_param.x * in.cyl_param.y);
        if (phase > 0.64) {
            discard;
        }
    }
    return shade_blinn_phong(in.view_pos, in.view_normal, in.color, front_facing);
}

// ============================================================================
// Offscreen Rgba8Unorm -> egui RenderPass Blit Shader
// ============================================================================

@group(0) @binding(0)
var resolved_tex: texture_2d<f32>;
@group(0) @binding(1)
var resolved_sampler: sampler;

struct BlitVarying {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vid: u32) -> BlitVarying {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid & 2u) * 2 - 1);
    var out: BlitVarying;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (1.0 - y));
    return out;
}

@fragment
fn fs_blit(in: BlitVarying) -> @location(0) vec4<f32> {
    return textureSample(resolved_tex, resolved_sampler, in.uv);
}
