// Clean-room WGSL compute shader for massively parallel multi-replica
// Simulated Annealing of backbone NMR spin-system assignment.
// Provenance: Parallel GPU execution of the Metropolis-Hastings move set
// (single/swap, block swap, unassign) in docs/specs/03-assignment.md section 3.
//
// Zero function-scope dynamic arrays are used (all local energy deltas and state
// rollbacks execute in pure GPU registers without stack spills), and all three
// compute entry points share the exact 16,384-byte baseline WebGPU workgroup
// SRAM budget (`wg_at` + `wg_pos` = 2 x 2048 x 4 B = 16 KiB):
// - `anneal_chunk` (`@workgroup_size(8, 1, 1)`): 8 replicas/workgroup, up to 256 spin systems/residues in SRAM.
// - `anneal_chunk_1024` (`@workgroup_size(2, 1, 1)`): 2 replicas/workgroup, up to 1024 spin systems/residues in SRAM.
// - `anneal_chunk_global` (`@workgroup_size(8, 1, 1)`): storage-backed state for >1024 spin systems/residues.

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
const WG_SLOTS_1024: u32 = 1024u;

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

fn eval_1_wg(wg_base: u32, k_len: u32, s_len: u32, k0: u32) -> f32 {
    var sc: f32 = 0.0;
    let s0 = wg_at[wg_base + k0];
    if (s0 >= 0) {
        sc = sc + u_mat[u32(s0) * k_len + k0];
        if (k0 > 0u) {
            let a = wg_at[wg_base + k0 - 1u];
            if (a >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(s0)];
            }
        }
        if (k0 + 1u < k_len) {
            let b = wg_at[wg_base + k0 + 1u];
            if (b >= 0) {
                sc = sc + p_mat[u32(s0) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn eval_2_wg(wg_base: u32, k_len: u32, s_len: u32, k_a: u32, k_b: u32) -> f32 {
    let lo = min(k_a, k_b);
    let hi = max(k_a, k_b);
    var sc: f32 = 0.0;
    let s_lo = wg_at[wg_base + lo];
    let s_hi = wg_at[wg_base + hi];
    if (s_lo >= 0) {
        sc = sc + u_mat[u32(s_lo) * k_len + lo];
        if (lo > 0u) {
            let a = wg_at[wg_base + lo - 1u];
            if (a >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(s_lo)];
            }
        }
        let b_lo = wg_at[wg_base + lo + 1u];
        if (b_lo >= 0) {
            sc = sc + p_mat[u32(s_lo) * s_len + u32(b_lo)];
        }
    }
    if (s_hi >= 0) {
        sc = sc + u_mat[u32(s_hi) * k_len + hi];
        if (hi > lo + 1u) {
            let a_hi = wg_at[wg_base + hi - 1u];
            if (a_hi >= 0) {
                sc = sc + p_mat[u32(a_hi) * s_len + u32(s_hi)];
            }
        }
        if (hi + 1u < k_len) {
            let b_hi = wg_at[wg_base + hi + 1u];
            if (b_hi >= 0) {
                sc = sc + p_mat[u32(s_hi) * s_len + u32(b_hi)];
            }
        }
    }
    return sc;
}

fn eval_block_wg(wg_base: u32, k_len: u32, s_len: u32, k1: u32, k2: u32, l: u32) -> f32 {
    let b0 = min(k1, k2);
    let b1 = max(k1, k2);
    var sc: f32 = 0.0;
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let s0 = wg_at[wg_base + b0 + i];
        if (s0 >= 0) {
            sc = sc + u_mat[u32(s0) * k_len + (b0 + i)];
        }
        let s1 = wg_at[wg_base + b1 + i];
        if (s1 >= 0) {
            sc = sc + u_mat[u32(s1) * k_len + (b1 + i)];
        }
    }
    if (k_len > 1u) {
        let max_e = k_len - 1u;
        let e0_start = select(0u, b0 - 1u, b0 > 0u);
        let e0_end = min(b0 + l, max_e);
        for (var e: u32 = e0_start; e < e0_end; e = e + 1u) {
            let a = wg_at[wg_base + e];
            let b = wg_at[wg_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
        let e1_start = max(b1 - 1u, e0_end);
        let e1_end = min(b1 + l, max_e);
        for (var e: u32 = e1_start; e < e1_end; e = e + 1u) {
            let a = wg_at[wg_base + e];
            let b = wg_at[wg_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn eval_total_wg(wg_base: u32, k_len: u32, s_len: u32) -> f32 {
    var sc: f32 = 0.0;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = wg_at[wg_base + k];
        if (s >= 0) {
            sc = sc + u_mat[u32(s) * k_len + k];
        }
    }
    if (k_len > 1u) {
        for (var e: u32 = 0u; e + 1u < k_len; e = e + 1u) {
            let a = wg_at[wg_base + e];
            let b = wg_at[wg_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn swap_blocks_wg(wg_base: u32, k1: u32, k2: u32, l: u32) {
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let a = wg_at[wg_base + k1 + i];
        let b = wg_at[wg_base + k2 + i];
        wg_at[wg_base + k1 + i] = b;
        wg_at[wg_base + k2 + i] = a;
        if (a >= 0) {
            wg_pos[wg_base + u32(a)] = i32(k2 + i);
        }
        if (b >= 0) {
            wg_pos[wg_base + u32(b)] = i32(k1 + i);
        }
    }
}

fn run_anneal_wg(r: u32, wg_base: u32, check_exact_best: bool) {
    let s_len = params.s_len;
    let k_len = params.k_len;
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
    if (check_exact_best && params.step_offset > 0u) {
        cur = eval_total_wg(wg_base, k_len, s_len);
    }
    var best = sc_pair.y;

    let denom = f32(max(params.total_steps - 1u, 1u));
    let log_ratio = log(params.t_end / params.t_start);
    let max_blk = min(min(params.max_block, 16u), k_len / 2u);

    let end_step = min(params.step_offset + params.chunk_steps, params.total_steps);
    for (var step: u32 = params.step_offset; step < end_step; step = step + 1u) {
        let temp = params.t_start * exp(log_ratio * (f32(step) / denom));
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
                let before = eval_1_wg(wg_base, k_len, s_len, k);
                wg_at[wg_base + k] = -1;
                wg_pos[wg_base + s] = -1;
                let delta = -before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        if (check_exact_best) {
                            cur = eval_total_wg(wg_base, k_len, s_len);
                        }
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = wg_at[wg_base + ki];
                            }
                        }
                    }
                } else {
                    wg_at[wg_base + k] = i32(s);
                    wg_pos[wg_base + s] = i32(k);
                }
            } else if (ks < 0) {
                let before = eval_1_wg(wg_base, k_len, s_len, k);
                if (t >= 0) {
                    wg_pos[wg_base + u32(t)] = -1;
                }
                wg_at[wg_base + k] = i32(s);
                wg_pos[wg_base + s] = i32(k);
                let delta = eval_1_wg(wg_base, k_len, s_len, k) - before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        if (check_exact_best) {
                            cur = eval_total_wg(wg_base, k_len, s_len);
                        }
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = wg_at[wg_base + ki];
                            }
                        }
                    }
                } else {
                    wg_pos[wg_base + s] = -1;
                    wg_at[wg_base + k] = t;
                    if (t >= 0) {
                        wg_pos[wg_base + u32(t)] = i32(k);
                    }
                }
            } else {
                let ks_u = u32(ks);
                var repl: i32 = -1;
                if (t >= 0 && u_mat[u32(t) * k_len + ks_u] > FORBID_HALF) {
                    repl = t;
                }
                let before = eval_2_wg(wg_base, k_len, s_len, k, ks_u);
                if (t >= 0) {
                    wg_pos[wg_base + u32(t)] = -1;
                }
                wg_at[wg_base + k] = i32(s);
                wg_at[wg_base + ks_u] = repl;
                wg_pos[wg_base + s] = i32(k);
                if (repl >= 0) {
                    wg_pos[wg_base + u32(repl)] = i32(ks_u);
                }
                let delta = eval_2_wg(wg_base, k_len, s_len, k, ks_u) - before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        if (check_exact_best) {
                            cur = eval_total_wg(wg_base, k_len, s_len);
                        }
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = wg_at[wg_base + ki];
                            }
                        }
                    }
                } else {
                    if (repl >= 0) {
                        wg_pos[wg_base + u32(repl)] = -1;
                    }
                    wg_at[wg_base + k] = t;
                    wg_at[wg_base + ks_u] = i32(s);
                    if (t >= 0) {
                        wg_pos[wg_base + u32(t)] = i32(k);
                    }
                    wg_pos[wg_base + s] = i32(ks_u);
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
            }
            if (!valid) {
                continue;
            }
            let before = eval_block_wg(wg_base, k_len, s_len, k1, k2, l);
            swap_blocks_wg(wg_base, k1, k2, l);
            let delta = eval_block_wg(wg_base, k_len, s_len, k1, k2, l) - before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    if (check_exact_best) {
                        cur = eval_total_wg(wg_base, k_len, s_len);
                    }
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            at_state[best_at_base + ki] = wg_at[wg_base + ki];
                        }
                    }
                }
            } else {
                swap_blocks_wg(wg_base, k1, k2, l);
            }
        } else {
            let s = rand_below(&rng, s_len);
            let ks = wg_pos[wg_base + s];
            if (ks < 0) {
                continue;
            }
            let k = u32(ks);
            let before = eval_1_wg(wg_base, k_len, s_len, k);
            wg_at[wg_base + k] = -1;
            wg_pos[wg_base + s] = -1;
            let delta = -before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    if (check_exact_best) {
                        cur = eval_total_wg(wg_base, k_len, s_len);
                    }
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            at_state[best_at_base + ki] = wg_at[wg_base + ki];
                        }
                    }
                }
            } else {
                wg_at[wg_base + k] = i32(s);
                wg_pos[wg_base + s] = i32(k);
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

fn eval_1_global(at_base: u32, k_len: u32, s_len: u32, k0: u32) -> f32 {
    var sc: f32 = 0.0;
    let s0 = at_state[at_base + k0];
    if (s0 >= 0) {
        sc = sc + u_mat[u32(s0) * k_len + k0];
        if (k0 > 0u) {
            let a = at_state[at_base + k0 - 1u];
            if (a >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(s0)];
            }
        }
        if (k0 + 1u < k_len) {
            let b = at_state[at_base + k0 + 1u];
            if (b >= 0) {
                sc = sc + p_mat[u32(s0) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn eval_2_global(at_base: u32, k_len: u32, s_len: u32, k_a: u32, k_b: u32) -> f32 {
    let lo = min(k_a, k_b);
    let hi = max(k_a, k_b);
    var sc: f32 = 0.0;
    let s_lo = at_state[at_base + lo];
    let s_hi = at_state[at_base + hi];
    if (s_lo >= 0) {
        sc = sc + u_mat[u32(s_lo) * k_len + lo];
        if (lo > 0u) {
            let a = at_state[at_base + lo - 1u];
            if (a >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(s_lo)];
            }
        }
        let b_lo = at_state[at_base + lo + 1u];
        if (b_lo >= 0) {
            sc = sc + p_mat[u32(s_lo) * s_len + u32(b_lo)];
        }
    }
    if (s_hi >= 0) {
        sc = sc + u_mat[u32(s_hi) * k_len + hi];
        if (hi > lo + 1u) {
            let a_hi = at_state[at_base + hi - 1u];
            if (a_hi >= 0) {
                sc = sc + p_mat[u32(a_hi) * s_len + u32(s_hi)];
            }
        }
        if (hi + 1u < k_len) {
            let b_hi = at_state[at_base + hi + 1u];
            if (b_hi >= 0) {
                sc = sc + p_mat[u32(s_hi) * s_len + u32(b_hi)];
            }
        }
    }
    return sc;
}

fn eval_block_global(at_base: u32, k_len: u32, s_len: u32, k1: u32, k2: u32, l: u32) -> f32 {
    let b0 = min(k1, k2);
    let b1 = max(k1, k2);
    var sc: f32 = 0.0;
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let s0 = at_state[at_base + b0 + i];
        if (s0 >= 0) {
            sc = sc + u_mat[u32(s0) * k_len + (b0 + i)];
        }
        let s1 = at_state[at_base + b1 + i];
        if (s1 >= 0) {
            sc = sc + u_mat[u32(s1) * k_len + (b1 + i)];
        }
    }
    if (k_len > 1u) {
        let max_e = k_len - 1u;
        let e0_start = select(0u, b0 - 1u, b0 > 0u);
        let e0_end = min(b0 + l, max_e);
        for (var e: u32 = e0_start; e < e0_end; e = e + 1u) {
            let a = at_state[at_base + e];
            let b = at_state[at_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
        let e1_start = max(b1 - 1u, e0_end);
        let e1_end = min(b1 + l, max_e);
        for (var e: u32 = e1_start; e < e1_end; e = e + 1u) {
            let a = at_state[at_base + e];
            let b = at_state[at_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn eval_total_global(at_base: u32, k_len: u32, s_len: u32) -> f32 {
    var sc: f32 = 0.0;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = at_state[at_base + k];
        if (s >= 0) {
            sc = sc + u_mat[u32(s) * k_len + k];
        }
    }
    if (k_len > 1u) {
        for (var e: u32 = 0u; e + 1u < k_len; e = e + 1u) {
            let a = at_state[at_base + e];
            let b = at_state[at_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + p_mat[u32(a) * s_len + u32(b)];
            }
        }
    }
    return sc;
}

fn swap_blocks_global(at_base: u32, pos_base: u32, k1: u32, k2: u32, l: u32) {
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let a = at_state[at_base + k1 + i];
        let b = at_state[at_base + k2 + i];
        at_state[at_base + k1 + i] = b;
        at_state[at_base + k2 + i] = a;
        if (a >= 0) {
            pos_state[pos_base + u32(a)] = i32(k2 + i);
        }
        if (b >= 0) {
            pos_state[pos_base + u32(b)] = i32(k1 + i);
        }
    }
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
    if (params.s_len == 0u || params.k_len == 0u || params.s_len > WG_SLOTS || params.k_len > WG_SLOTS) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_SLOTS, false);
}

@compute @workgroup_size(2, 1, 1)
fn anneal_chunk_1024(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    if (params.s_len == 0u || params.k_len == 0u || params.s_len > WG_SLOTS_1024 || params.k_len > WG_SLOTS_1024) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_SLOTS_1024, true);
}

@compute @workgroup_size(8, 1, 1)
fn anneal_chunk_global(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let s_len = params.s_len;
    let k_len = params.k_len;
    if (s_len == 0u || k_len == 0u) {
        return;
    }

    let at_base = r * k_len;
    let best_at_base = (params.replicas + r) * k_len;
    let pos_base = r * s_len;
    let slot_idx_base = s_len + 1u;

    var rng = rng_state[r];
    var sc_pair = scores_state[r];
    var cur = select(sc_pair.x, eval_total_global(at_base, k_len, s_len), params.step_offset > 0u);
    var best = sc_pair.y;

    let denom = f32(max(params.total_steps - 1u, 1u));
    let log_ratio = log(params.t_end / params.t_start);
    let max_blk = min(min(params.max_block, 16u), k_len / 2u);

    let end_step = min(params.step_offset + params.chunk_steps, params.total_steps);
    for (var step: u32 = params.step_offset; step < end_step; step = step + 1u) {
        let temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        let draw = rand_f32(&rng);

        if (draw < 0.65) {
            let s = rand_below(&rng, s_len);
            let sl_start = slots_packed[s];
            let sl_count = slots_packed[s + 1u] - sl_start;
            if (sl_count == 0u) {
                continue;
            }
            let k = slots_packed[slot_idx_base + sl_start + rand_below(&rng, sl_count)];
            let ks = pos_state[pos_base + s];
            let t = at_state[at_base + k];

            if (ks == i32(k)) {
                let before = eval_1_global(at_base, k_len, s_len, k);
                at_state[at_base + k] = -1;
                pos_state[pos_base + s] = -1;
                let delta = -before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_global(at_base, k_len, s_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = at_state[at_base + ki];
                            }
                        }
                    }
                } else {
                    at_state[at_base + k] = i32(s);
                    pos_state[pos_base + s] = i32(k);
                }
            } else if (ks < 0) {
                let before = eval_1_global(at_base, k_len, s_len, k);
                if (t >= 0) {
                    pos_state[pos_base + u32(t)] = -1;
                }
                at_state[at_base + k] = i32(s);
                pos_state[pos_base + s] = i32(k);
                let delta = eval_1_global(at_base, k_len, s_len, k) - before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_global(at_base, k_len, s_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = at_state[at_base + ki];
                            }
                        }
                    }
                } else {
                    pos_state[pos_base + s] = -1;
                    at_state[at_base + k] = t;
                    if (t >= 0) {
                        pos_state[pos_base + u32(t)] = i32(k);
                    }
                }
            } else {
                let ks_u = u32(ks);
                var repl: i32 = -1;
                if (t >= 0 && u_mat[u32(t) * k_len + ks_u] > FORBID_HALF) {
                    repl = t;
                }
                let before = eval_2_global(at_base, k_len, s_len, k, ks_u);
                if (t >= 0) {
                    pos_state[pos_base + u32(t)] = -1;
                }
                at_state[at_base + k] = i32(s);
                at_state[at_base + ks_u] = repl;
                pos_state[pos_base + s] = i32(k);
                if (repl >= 0) {
                    pos_state[pos_base + u32(repl)] = i32(ks_u);
                }
                let delta = eval_2_global(at_base, k_len, s_len, k, ks_u) - before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_global(at_base, k_len, s_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                at_state[best_at_base + ki] = at_state[at_base + ki];
                            }
                        }
                    }
                } else {
                    if (repl >= 0) {
                        pos_state[pos_base + u32(repl)] = -1;
                    }
                    at_state[at_base + k] = t;
                    at_state[at_base + ks_u] = i32(s);
                    if (t >= 0) {
                        pos_state[pos_base + u32(t)] = i32(k);
                    }
                    pos_state[pos_base + s] = i32(ks_u);
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
                let a = at_state[at_base + k1 + i];
                let b = at_state[at_base + k2 + i];
                if ((b >= 0 && u_mat[u32(b) * k_len + (k1 + i)] < FORBID_HALF)
                    || (a >= 0 && u_mat[u32(a) * k_len + (k2 + i)] < FORBID_HALF)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }
            let before = eval_block_global(at_base, k_len, s_len, k1, k2, l);
            swap_blocks_global(at_base, pos_base, k1, k2, l);
            let delta = eval_block_global(at_base, k_len, s_len, k1, k2, l) - before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    cur = eval_total_global(at_base, k_len, s_len);
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            at_state[best_at_base + ki] = at_state[at_base + ki];
                        }
                    }
                }
            } else {
                swap_blocks_global(at_base, pos_base, k1, k2, l);
            }
        } else {
            let s = rand_below(&rng, s_len);
            let ks = pos_state[pos_base + s];
            if (ks < 0) {
                continue;
            }
            let k = u32(ks);
            let before = eval_1_global(at_base, k_len, s_len, k);
            at_state[at_base + k] = -1;
            pos_state[pos_base + s] = -1;
            let delta = -before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    cur = eval_total_global(at_base, k_len, s_len);
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            at_state[best_at_base + ki] = at_state[at_base + ki];
                        }
                    }
                }
            } else {
                at_state[at_base + k] = i32(s);
                pos_state[pos_base + s] = i32(k);
            }
        }
    }

    scores_state[r] = vec2<f32>(cur, best);
    rng_state[r] = rng;
}
