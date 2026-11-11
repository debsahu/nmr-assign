// Clean-room WGSL compute shader for multi-scale separable 3D convolutional NMR peak
// detection, sub-voxel offset regression, and 3D non-maximum suppression (NMS).
//
// Provenance: Original GPU compute implementation matching `nmr_core::neural_picker::SeparableConvNet3D`
// (depthwise/separable dilated 1D convolutions along 1H, 15N, 13C + pointwise 1x1x1 mixing + GELU).

struct Conv3DParams {
    nx: u32,
    ny: u32,
    nz: u32,
    allow_neg: u32,
    prob_thresh: f32,
    min_snr_norm: f32,
    valid_z0: u32,
    valid_z1: u32,
};

struct GpuRawPeak3D {
    ix: u32,
    iy: u32,
    iz: u32,
    sign: i32,
    prob: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    amp_norm: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

struct NmsOutputBuffer {
    count: atomic<u32>,
    max_peaks: u32,
    z_global_offset: u32,
    pad0: u32,
    peaks: array<GpuRawPeak3D>,
};

@group(0) @binding(0) var<uniform> params: Conv3DParams;
@group(0) @binding(1) var<storage, read> input_vol: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> buf_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> buf_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> buf_skip: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_voxels: array<f32>;
@group(0) @binding(7) var<storage, read_write> nms_out: NmsOutputBuffer;

const HIDDEN_CH: u32 = 16u;
const OUT_CH: u32 = 6u;

const OFF3_STEM_KX: u32 = 0u;
const OFF3_STEM_KY: u32 = 80u;
const OFF3_STEM_KZ: u32 = 160u;
const OFF3_PW1_W: u32 = 240u;
const OFF3_PW1_B: u32 = 496u;
const OFF3_DW2_KX: u32 = 512u;
const OFF3_DW2_KY: u32 = 560u;
const OFF3_DW2_KZ: u32 = 608u;
const OFF3_PW2_W: u32 = 656u;
const OFF3_PW2_B: u32 = 912u;
const OFF3_HEAD_W: u32 = 928u;
const OFF3_HEAD_B: u32 = 1024u;

fn chan_dilation(c: u32) -> i32 {
    return select(2, 1, c < 8u);
}

fn clamp_idx(v: i32, dim: u32) -> u32 {
    return u32(clamp(v, 0, i32(dim) - 1));
}

fn safe_tanh(x: f32) -> f32 {
    let xc = clamp(x, -9.5, 9.5);
    let e2 = exp(2.0 * xc);
    return (e2 - 1.0) / (e2 + 1.0);
}

fn gelu_f32(x: f32) -> f32 {
    let inner = clamp(0.7978846 * (x + 0.044715 * x * x * x), -9.0, 9.0);
    return 0.5 * x * (1.0 + safe_tanh(inner));
}

fn sigmoid_f32(x: f32) -> f32 {
    let xc = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-xc));
}

fn softplus_f32(x: f32) -> f32 {
    if (x > 18.0) {
        return x;
    }
    if (x < -18.0) {
        return exp(x);
    }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(8, 8, 4)
fn stem_pass_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let row_base = (gid.z * params.ny + gid.y) * params.nx;
    let v_idx = row_base + gid.x;
    let dst_base = v_idx * HIDDEN_CH;

    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_STEM_KX + c * 5u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 5; m = m + 1) {
            let gx = clamp_idx(i32(gid.x) + d * (m - 2), params.nx);
            acc = acc + weights[kw_base + u32(m)] * input_vol[row_base + gx];
        }
        buf_a[dst_base + c] = acc;
    }
}

@compute @workgroup_size(8, 8, 4)
fn stem_pass_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let plane_base = gid.z * params.ny * params.nx;
    let v_idx = plane_base + gid.y * params.nx + gid.x;
    let dst_base = v_idx * HIDDEN_CH;

    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_STEM_KY + c * 5u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 5; m = m + 1) {
            let gy = clamp_idx(i32(gid.y) + d * (m - 2), params.ny);
            let src_v = plane_base + gy * params.nx + gid.x;
            acc = acc + weights[kw_base + u32(m)] * buf_a[src_v * HIDDEN_CH + c];
        }
        buf_b[dst_base + c] = acc;
    }
}

@compute @workgroup_size(8, 8, 4)
fn stem_pass_z_pw1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let plane = params.nx * params.ny;
    let p_off = gid.y * params.nx + gid.x;
    let v_idx = gid.z * plane + p_off;
    let dst_base = v_idx * HIDDEN_CH;

    var sz: array<f32, 16>;
    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_STEM_KZ + c * 5u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 5; m = m + 1) {
            let gz = clamp_idx(i32(gid.z) + d * (m - 2), params.nz);
            acc = acc + weights[kw_base + u32(m)] * buf_b[(gz * plane + p_off) * HIDDEN_CH + c];
        }
        sz[c] = acc;
    }

    for (var co: u32 = 0u; co < HIDDEN_CH; co = co + 1u) {
        let pw_base = OFF3_PW1_W + co * HIDDEN_CH;
        var sum: f32 = weights[OFF3_PW1_B + co];
        for (var ci: u32 = 0u; ci < HIDDEN_CH; ci = ci + 1u) {
            sum = sum + weights[pw_base + ci] * sz[ci];
        }
        buf_skip[dst_base + co] = gelu_f32(sum);
    }
}

@compute @workgroup_size(8, 8, 4)
fn block2_pass_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let row_base = (gid.z * params.ny + gid.y) * params.nx;
    let v_idx = row_base + gid.x;
    let dst_base = v_idx * HIDDEN_CH;

    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_DW2_KX + c * 3u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 3; m = m + 1) {
            let gx = clamp_idx(i32(gid.x) + d * (m - 1), params.nx);
            acc = acc + weights[kw_base + u32(m)] * buf_skip[(row_base + gx) * HIDDEN_CH + c];
        }
        buf_a[dst_base + c] = acc;
    }
}

@compute @workgroup_size(8, 8, 4)
fn block2_pass_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let plane_base = gid.z * params.ny * params.nx;
    let v_idx = plane_base + gid.y * params.nx + gid.x;
    let dst_base = v_idx * HIDDEN_CH;

    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_DW2_KY + c * 3u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 3; m = m + 1) {
            let gy = clamp_idx(i32(gid.y) + d * (m - 1), params.ny);
            let src_v = plane_base + gy * params.nx + gid.x;
            acc = acc + weights[kw_base + u32(m)] * buf_a[src_v * HIDDEN_CH + c];
        }
        buf_b[dst_base + c] = acc;
    }
}

@compute @workgroup_size(8, 8, 4)
fn block2_pass_z_pw2_head(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.ny || gid.z >= params.nz) {
        return;
    }
    let plane = params.nx * params.ny;
    let p_off = gid.y * params.nx + gid.x;
    let v_idx = gid.z * plane + p_off;
    let skip_base = v_idx * HIDDEN_CH;

    var rz: array<f32, 16>;
    for (var c: u32 = 0u; c < HIDDEN_CH; c = c + 1u) {
        let d = chan_dilation(c);
        let kw_base = OFF3_DW2_KZ + c * 3u;
        var acc: f32 = 0.0;
        for (var m: i32 = 0; m < 3; m = m + 1) {
            let gz = clamp_idx(i32(gid.z) + d * (m - 1), params.nz);
            acc = acc + weights[kw_base + u32(m)] * buf_b[(gz * plane + p_off) * HIDDEN_CH + c];
        }
        rz[c] = acc;
    }

    var a2: array<f32, 16>;
    for (var co: u32 = 0u; co < HIDDEN_CH; co = co + 1u) {
        let pw_base = OFF3_PW2_W + co * HIDDEN_CH;
        var sum: f32 = buf_skip[skip_base + co] + weights[OFF3_PW2_B + co];
        for (var ci: u32 = 0u; ci < HIDDEN_CH; ci = ci + 1u) {
            sum = sum + weights[pw_base + ci] * rz[ci];
        }
        a2[co] = gelu_f32(sum);
    }

    var raw_h: array<f32, 6>;
    for (var k: u32 = 0u; k < OUT_CH; k = k + 1u) {
        let hw_base = OFF3_HEAD_W + k * HIDDEN_CH;
        var sum: f32 = weights[OFF3_HEAD_B + k];
        for (var ci: u32 = 0u; ci < HIDDEN_CH; ci = ci + 1u) {
            sum = sum + weights[hw_base + ci] * a2[ci];
        }
        raw_h[k] = sum;
    }

    let out_base = v_idx * OUT_CH;
    out_voxels[out_base + 0u] = sigmoid_f32(raw_h[0]);
    out_voxels[out_base + 1u] = sigmoid_f32(raw_h[1]);
    out_voxels[out_base + 2u] = 0.5 * safe_tanh(clamp(raw_h[2], -12.0, 12.0));
    out_voxels[out_base + 3u] = 0.5 * safe_tanh(clamp(raw_h[3], -12.0, 12.0));
    out_voxels[out_base + 4u] = 0.5 * safe_tanh(clamp(raw_h[4], -12.0, 12.0));
    out_voxels[out_base + 5u] = softplus_f32(raw_h[5]);
}

@compute @workgroup_size(8, 8, 4)
fn nms_3d_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 1u || gid.x + 1u >= params.nx ||
        gid.y < 1u || gid.y + 1u >= params.ny ||
        gid.z < params.valid_z0 || gid.z >= params.valid_z1 ||
        gid.z < 1u || gid.z + 1u >= params.nz) {
        return;
    }

    let plane = params.nx * params.ny;
    let v0 = gid.z * plane + gid.y * params.nx + gid.x;
    let base0 = v0 * OUT_CH;
    let z_val = input_vol[v0];

    let n_ch = select(1u, 2u, params.allow_neg != 0u);
    for (var ch: u32 = 0u; ch < n_ch; ch = ch + 1u) {
        let sign_val = select(-1, 1, ch == 0u);
        let prob = out_voxels[base0 + ch];
        if (prob < params.prob_thresh) {
            continue;
        }
        if ((sign_val > 0 && z_val < params.min_snr_norm) ||
            (sign_val < 0 && z_val > -params.min_snr_norm)) {
            continue;
        }

        var is_max: bool = true;
        for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    let v1 = u32(i32(gid.z) + dz) * plane
                        + u32(i32(gid.y) + dy) * params.nx
                        + u32(i32(gid.x) + dx);
                    let np = out_voxels[v1 * OUT_CH + ch];
                    if (np > prob || (np == prob && v1 < v0)) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) {
                    break;
                }
            }
            if (!is_max) {
                break;
            }
        }

        if (is_max) {
            let slot = atomicAdd(&nms_out.count, 1u);
            if (slot < nms_out.max_peaks) {
                nms_out.peaks[slot] = GpuRawPeak3D(
                    gid.x,
                    gid.y,
                    gid.z + nms_out.z_global_offset,
                    sign_val,
                    prob,
                    out_voxels[base0 + 2u],
                    out_voxels[base0 + 3u],
                    out_voxels[base0 + 4u],
                    out_voxels[base0 + 5u],
                    0.0,
                    0.0,
                    0.0,
                );
            }
        }
    }
}
