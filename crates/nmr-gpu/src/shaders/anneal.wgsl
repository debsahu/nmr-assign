// Clean-room WGSL compute shader for massively parallel multi-replica
// Simulated Annealing of backbone NMR spin-system assignment.
// Provenance: Parallel GPU execution of the Metropolis-Hastings move set
// (single/swap, block swap, unassign) in docs/specs/03-assignment.md section 3.
// Stores per-replica `at` and `pos` arrays in on-chip workgroup SRAM (`16 KB` total:
// 8 replicas/workgroup x 256 i32 x 2 arrays = 16,384 bytes, within WebGPU baseline limits).

struct AnnealParams {
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
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: AnnealParams;
@group(0) @binding(1) var<storage, read> u_mat: array<f32>;
@group(0) @binding(2) var<storage, read> p_mat: array<f32>;
@group(0) @binding(3) var<storage, read> slots_packed: array<u32>;
@group(0) @binding(4) var<storage, read_write> at_state: array<i32>;
@group(0) @binding(5) var<storage, read_write> pos_state: array<i32>;
@group(0) @binding(6) var<storage, read_write> scores_state: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> rng_state: array<vec4<u32>>;

const FORBID_HALF: f32 = -5e8;
const WG_SLOTS: u32 = 256u;

var<workgroup> wg_at: array<i32, 2048>;
var<workgroup> wg_pos: array<i32, 2048>;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next_u32(st: ptr<function, vec4<u32>>) -> u32 {
    let s0 = (*st).x;
    var s1 = (*st).y;
    var s2 = (*st).z;
    var s3 = (*st).w;
    let res = rotl(s0 + s3, 7u) + s0;
    let t = s1 << 9u;
    s2 = s2 ^ s0;
    s3 = s3 ^ s1;
    s1 = s1 ^ s2;
    let new_s0 = s0 ^ s3;
    s2 = s2 ^ t;
    s3 = rotl(s3, 11u);
    *st = vec4<u32>(new_s0, s1, s2, s3);
    return res;
}

fn rand_f32(st: ptr<function, vec4<u32>>) -> f32 {
    return f32(next_u32(st) >> 8u) * (1.0 / 16777216.0);
}

fn rand_below(st: ptr<function, vec4<u32>>, upper: u32) -> u32 {
    return next_u32(st) % max(upper, 1u);
}

fn eval_local_wg(
    wg_base: u32,
    k_len: u32,
    s_len: u32,
    ch_k: ptr<function, array<u32, 32>>,
    n_ch: u32,
) -> f32 {
    var sc: f32 = 0.0;
    var edges: array<u32, 64>;
    var n_edges: u32 = 0u;

    for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
        let k = (*ch_k)[i];
        let s = wg_at[wg_base + k];
        if (s >= 0) {
            sc = sc + u_mat[u32(s) * k_len + k];
        }
        if (k > 0u && (k - 1u) < (k_len - 1u)) {
            let e1 = k - 1u;
            var dup1 = false;
            for (var j: u32 = 0u; j < n_edges; j = j + 1u) {
                if (edges[j] == e1) {
                    dup1 = true;
                    break;
                }
            }
            if (!dup1) {
                edges[n_edges] = e1;
                n_edges = n_edges + 1u;
            }
        }
        if (k < (k_len - 1u)) {
            var dup2 = false;
            for (var j: u32 = 0u; j < n_edges; j = j + 1u) {
                if (edges[j] == k) {
                    dup2 = true;
                    break;
                }
            }
            if (!dup2) {
                edges[n_edges] = k;
                n_edges = n_edges + 1u;
            }
        }
    }

    for (var j: u32 = 0u; j < n_edges; j = j + 1u) {
        let e = edges[j];
        let a = wg_at[wg_base + e];
        let b = wg_at[wg_base + e + 1u];
        if (a >= 0 && b >= 0) {
            sc = sc + p_mat[u32(a) * s_len + u32(b)];
        }
    }
    return sc;
}

@compute @workgroup_size(8, 1, 1)
fn anneal_chunk(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let s_len = params.s_len;
    let k_len = params.k_len;
    if (s_len == 0u || k_len == 0u || s_len > WG_SLOTS || k_len > WG_SLOTS) {
        return;
    }

    let wg_base = lid.x * WG_SLOTS;
    let at_base = r * k_len;
    let best_at_base = (params.replicas + r) * k_len;
    let pos_base = r * s_len;
    let slot_idx_base = s_len + 1u;

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        wg_at[wg_base + k] = at_state[at_base + k];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        wg_pos[wg_base + s] = pos_state[pos_base + s];
    }

    var rng = rng_state[r];
    var sc_pair = scores_state[r];
    var cur = sc_pair.x;
    var best = sc_pair.y;

    let denom = f32(max(params.total_steps - 1u, 1u));
    let log_ratio = log(params.t_end / params.t_start);
    let max_blk = min(min(params.max_block, 16u), k_len / 2u);

    var ch_k: array<u32, 32>;
    var ch_new: array<i32, 32>;
    var ch_old: array<i32, 32>;

    let end_step = min(params.step_offset + params.chunk_steps, params.total_steps);
    for (var step: u32 = params.step_offset; step < end_step; step = step + 1u) {
        let temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        var n_ch: u32 = 0u;
        let draw = rand_f32(&rng);

        if (draw < 0.65) {
            let s = rand_below(&rng, s_len);
            let sl_start = slots_packed[s];
            let sl_count = slots_packed[s + 1u] - sl_start;
            if (sl_count == 0u) {
                continue;
            }
            let k = slots_packed[slot_idx_base + sl_start + rand_below(&rng, sl_count)];
            let ks = wg_pos[wg_base + s];
            let t = wg_at[wg_base + k];
            if (ks == i32(k)) {
                ch_k[0] = k;
                ch_new[0] = -1;
                n_ch = 1u;
            } else {
                ch_k[0] = k;
                ch_new[0] = i32(s);
                n_ch = 1u;
                if (ks >= 0) {
                    let ks_u = u32(ks);
                    var repl: i32 = -1;
                    if (t >= 0 && u_mat[u32(t) * k_len + ks_u] > FORBID_HALF) {
                        repl = t;
                    }
                    ch_k[1] = ks_u;
                    ch_new[1] = repl;
                    n_ch = 2u;
                }
            }
        } else if (draw < 0.95) {
            if (max_blk == 0u) {
                continue;
            }
            let l = 1u + rand_below(&rng, max_blk);
            let span = k_len - l + 1u;
            let k1 = rand_below(&rng, span);
            let k2 = rand_below(&rng, span);
            let diff = select(k2 - k1, k1 - k2, k1 >= k2);
            if (diff < l) {
                continue;
            }
            var valid = true;
            for (var i: u32 = 0u; i < l; i = i + 1u) {
                let a = wg_at[wg_base + k1 + i];
                let b = wg_at[wg_base + k2 + i];
                if ((b >= 0 && u_mat[u32(b) * k_len + (k1 + i)] < FORBID_HALF)
                    || (a >= 0 && u_mat[u32(a) * k_len + (k2 + i)] < FORBID_HALF)) {
                    valid = false;
                    break;
                }
                ch_k[2u * i] = k1 + i;
                ch_new[2u * i] = b;
                ch_k[2u * i + 1u] = k2 + i;
                ch_new[2u * i + 1u] = a;
            }
            if (!valid) {
                continue;
            }
            n_ch = 2u * l;
        } else {
            let s = rand_below(&rng, s_len);
            let ks = wg_pos[wg_base + s];
            if (ks < 0) {
                continue;
            }
            ch_k[0] = u32(ks);
            ch_new[0] = -1;
            n_ch = 1u;
        }

        if (n_ch == 0u) {
            continue;
        }

        let before = eval_local_wg(wg_base, k_len, s_len, &ch_k, n_ch);

        for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
            let k = ch_k[i];
            let old_s = wg_at[wg_base + k];
            ch_old[i] = old_s;
            if (old_s >= 0) {
                wg_pos[wg_base + u32(old_s)] = -1;
            }
        }
        for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
            wg_at[wg_base + ch_k[i]] = ch_new[i];
        }
        for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
            let ns = ch_new[i];
            if (ns >= 0) {
                wg_pos[wg_base + u32(ns)] = i32(ch_k[i]);
            }
        }

        let delta = eval_local_wg(wg_base, k_len, s_len, &ch_k, n_ch) - before;
        var accept = false;
        if (delta >= 0.0) {
            accept = true;
        } else if (rand_f32(&rng) < exp(delta / temp)) {
            accept = true;
        }

        if (accept) {
            cur = cur + delta;
            if (cur > best) {
                best = cur;
                for (var k: u32 = 0u; k < k_len; k = k + 1u) {
                    at_state[best_at_base + k] = wg_at[wg_base + k];
                }
            }
        } else {
            for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
                let ns = ch_new[i];
                if (ns >= 0) {
                    wg_pos[wg_base + u32(ns)] = -1;
                }
            }
            for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
                wg_at[wg_base + ch_k[i]] = ch_old[i];
            }
            for (var i: u32 = 0u; i < n_ch; i = i + 1u) {
                let os = ch_old[i];
                if (os >= 0) {
                    wg_pos[wg_base + u32(os)] = i32(ch_k[i]);
                }
            }
        }
    }

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        at_state[at_base + k] = wg_at[wg_base + k];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        pos_state[pos_base + s] = wg_pos[wg_base + s];
    }
    scores_state[r] = vec2<f32>(cur, best);
    rng_state[r] = rng;
}
