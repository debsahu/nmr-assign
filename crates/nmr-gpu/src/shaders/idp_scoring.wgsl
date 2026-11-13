// Clean-room WGSL compute shaders for protonless 13C-detected IDP backbone scoring
// (`compute_idp_unary`, `compute_idp_pair`) and massively parallel multi-replica
// detailed-balance Metropolis-Hastings simulated annealing (`anneal_idp_256`,
// `anneal_idp_1024`, `anneal_idp_global`) continuously across Prolines (`Phase 11`, `T11.2`).
//
// Provenance: Clean-room GPU compute implementation derived from peer-reviewed literature:
// - Bermel, Bertini, Felli, Piccioli & Pierattelli (Prog. Nucl. Magn. Reson. Spectrosc.
//   2006, 48, 25-45; J. Am. Chem. Soc. 2006, 128, 3918-3919; J. Am. Chem. Soc. 2009,
//   131, 15339-15345 -- 2D CON, CACO, 3D CANCO, CBCACON, CCC(CO)N, NCACO).
// - Tamiola, Acar & Mulder (J. Am. Chem. Soc. 2010, 132, 18000-18003 -- ncIDP library).
// - Kjaergaard & Poulsen (Prog. Nucl. Magn. Reson. Spectrosc. 2011, 59, 43-57).
//
// Portability & Limits:
// - Uses 1 uniform buffer + 6 storage buffers (<= 8 baseline WebGPU limit).
// - Uses exact 16,384-byte workgroup SRAM (`wg_at` + `wg_pos` = 2 x 2048 x 4 B = 16 KiB).
// - Zero function-scope dynamic array indexing (zero GPU stack spills).

struct IdpGpuParams {
    s_len: u32,
    k_len: u32,
    replicas: u32,
    max_block: u32,

    total_steps: u32,
    step_offset: u32,
    chunk_steps: u32,
    multi_3d_dataset: u32,

    off_bg_floats: u32,
    off_sys_floats: u32,
    off_seq_floats: u32,
    off_u_mat: u32,

    off_p_mat: u32,
    off_sys_ints: u32,
    off_seq_ints: u32,
    off_slots_packed: u32,

    d_n: f32,
    d_c: f32,
    d_ca: f32,
    d_cb: f32,

    llr_floor: f32,
    pair_floor: f32,
    sigma_seq_ca: f32,
    sigma_seq_c: f32,

    w_mult: f32,
    t_start: f32,
    t_end: f32,
    score_out_mode: f32,
};

@group(0) @binding(0) var<uniform> params: IdpGpuParams;
@group(0) @binding(1) var<storage, read> ro_floats: array<f32>;
@group(0) @binding(2) var<storage, read> ro_ints: array<u32>;
@group(0) @binding(3) var<storage, read_write> rw_at: array<i32>;
@group(0) @binding(4) var<storage, read_write> rw_pos: array<i32>;
@group(0) @binding(5) var<storage, read_write> rw_floats: array<f32>;
@group(0) @binding(6) var<storage, read_write> rw_rng: array<vec4<u32>>;

const FORBID: f32 = -1.0e9;
const FORBID_HALF: f32 = -5.0e8;
const INV_SQRT_2PI: f32 = 0.3989423;
const SYS_FLOAT_STRIDE: u32 = 32u;
const SEQ_FLOAT_STRIDE: u32 = 12u;
const SYS_INT_STRIDE: u32 = 7u;
const SEQ_INT_STRIDE: u32 = 2u;

const WG_SLOTS_256: u32 = 256u;
const WG_SLOTS_1024: u32 = 1024u;

var<workgroup> wg_at: array<i32, 2048>;
var<workgroup> wg_pos: array<i32, 2048>;

fn logpdf_2comp(v: f32, mu: f32, sd: f32) -> f32 {
    let s1 = max(sd, 0.15);
    let s2 = 3.8 * s1;
    let z1 = (v - mu) / s1;
    let z2 = (v - mu) / s2;
    let p1 = 0.78 * INV_SQRT_2PI * exp(-0.5 * z1 * z1) / s1;
    let p2 = 0.22 * INV_SQRT_2PI * exp(-0.5 * z2 * z2) / s2;
    return log(max(p1 + p2, 1.0e-30));
}

fn bg_logpdf(v: f32, bg_off: u32, bg_len: u32, sd: f32) -> f32 {
    var sum: f32 = 0.0;
    let inv_n = 1.0 / f32(bg_len);
    let s1 = max(sd * 1.25, 0.25);
    for (var i: u32 = 0u; i < bg_len; i = i + 1u) {
        let m = ro_floats[bg_off + i];
        let z = (v - m) / s1;
        sum = sum + inv_n * INV_SQRT_2PI * exp(-0.5 * z * z) / s1;
    }
    return log(max(0.95 * sum + 0.05 / 45.0, 1.0e-30));
}

fn best_cand_llr(
    cand_off: u32,
    cand_len: u32,
    mu: f32,
    sd: f32,
    bg_off: u32,
    bg_len: u32,
    bg_sd: f32,
    llr_floor: f32,
) -> f32 {
    if (cand_len == 0u) {
        return 0.0;
    }
    var best: f32 = -1.0e30;
    for (var idx: u32 = 0u; idx < cand_len; idx = idx + 1u) {
        let v = ro_floats[cand_off + idx];
        let rank_pen = 0.25 * f32(idx);
        let llr = max(
            logpdf_2comp(v, mu, sd) - bg_logpdf(v, bg_off, bg_len, bg_sd),
            llr_floor,
        ) - rank_pen;
        if (llr > best) {
            best = llr;
        }
    }
    let mult_gate = clamp(best / 1.5, 0.0, 1.0);
    return best - 0.12 * log(f32(cand_len)) * mult_gate;
}

@compute @workgroup_size(64, 1, 1)
fn compute_idp_unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_cells = params.s_len * params.k_len;
    let flat_idx = gid.x;
    if (flat_idx >= total_cells) {
        return;
    }
    let s = flat_idx / params.k_len;
    let k = flat_idx - s * params.k_len;
    if (k == 0u) {
        rw_floats[flat_idx] = FORBID;
        return;
    }

    let bg_n_off = params.off_bg_floats;
    let bg_c_off = bg_n_off + 20u;
    let bg_ca_off = bg_c_off + 20u;
    let bg_cb_off = bg_ca_off + 20u;
    let bg_sc_off = bg_cb_off + 19u;

    let sf = params.off_sys_floats + s * SYS_FLOAT_STRIDE;
    let si = params.off_sys_ints + s * SYS_INT_STRIDE;

    let n_curr = ro_floats[sf + 0u];
    let c_prev = ro_floats[sf + 1u];
    let ca_curr_off = sf + 2u;
    let ca_prev_off = sf + 8u;
    let cb_prev_off = sf + 14u;
    let c_curr_off = sf + 20u;
    let sc_prev_off = sf + 26u;

    let n_ca_curr = ro_ints[si + 0u];
    let n_ca_prev = ro_ints[si + 1u];
    let n_cb_prev = ro_ints[si + 2u];
    let n_c_curr = ro_ints[si + 3u];
    let n_sc_prev = ro_ints[si + 4u];
    let flags = ro_ints[si + 5u];
    let n_3d_ch = ro_ints[si + 6u];

    let has_cbcacon_ev = (flags & 1u) != 0u;
    let prev_gly_opp = (flags & 2u) != 0u;

    var sys_assign_bonus: f32 = 0.5;
    if (n_3d_ch >= 1u) {
        sys_assign_bonus = 2.65;
    } else if (params.multi_3d_dataset != 0u) {
        sys_assign_bonus = -6.5;
    }

    let cur_qf = params.off_seq_floats + k * SEQ_FLOAT_STRIDE;
    let prv_qf = params.off_seq_floats + (k - 1u) * SEQ_FLOAT_STRIDE;
    let prv_qi = params.off_seq_ints + (k - 1u) * SEQ_INT_STRIDE;

    let cur_n_mu = ro_floats[cur_qf + 0u];
    let cur_n_sd = ro_floats[cur_qf + 1u];
    let cur_c_mu = ro_floats[cur_qf + 2u];
    let cur_c_sd = ro_floats[cur_qf + 3u];
    let cur_ca_mu = ro_floats[cur_qf + 4u];
    let cur_ca_sd = ro_floats[cur_qf + 5u];

    let prv_c_mu = ro_floats[prv_qf + 2u];
    let prv_c_sd = ro_floats[prv_qf + 3u];
    let prv_ca_mu = ro_floats[prv_qf + 4u];
    let prv_ca_sd = ro_floats[prv_qf + 5u];
    let prv_cb_mu = ro_floats[prv_qf + 6u];
    let prv_cb_sd = ro_floats[prv_qf + 7u];

    let prv_has_cb = ro_ints[prv_qi + 0u] != 0u;
    let prv_n_sc = ro_ints[prv_qi + 1u];
    let llr_floor = params.llr_floor;

    // 1. Root 15N(k) and 13C'(k-1) (active continuously across Prolines!)
    var total: f32 = sys_assign_bonus;
    total = total + max(
        logpdf_2comp(n_curr, cur_n_mu, cur_n_sd) - bg_logpdf(n_curr, bg_n_off, 20u, 1.4),
        llr_floor,
    );
    total = total + max(
        logpdf_2comp(c_prev, prv_c_mu, prv_c_sd) - bg_logpdf(c_prev, bg_c_off, 20u, 0.7),
        llr_floor,
    );

    // 2. CA(k) & CA(k-1)
    total = total + best_cand_llr(
        ca_curr_off, n_ca_curr, cur_ca_mu, cur_ca_sd, bg_ca_off, 20u, 1.2, llr_floor,
    );
    total = total + best_cand_llr(
        ca_prev_off, n_ca_prev, prv_ca_mu, prv_ca_sd, bg_ca_off, 20u, 1.2, llr_floor,
    );

    // 3. CB(k-1) and Gly(k-1) opposite-phase / missing-CB signatures
    if (prv_has_cb) {
        if (n_cb_prev > 0u) {
            total = total + best_cand_llr(
                cb_prev_off, n_cb_prev, prv_cb_mu, prv_cb_sd, bg_cb_off, 19u, 2.5, llr_floor,
            );
        } else if (has_cbcacon_ev) {
            total = total - 1.35;
        }
        if (prev_gly_opp) {
            total = total - 4.50;
        }
    } else {
        if (n_cb_prev > 0u) {
            total = total - 4.20;
        } else if (has_cbcacon_ev && n_ca_prev > 0u) {
            total = total + 2.10;
        }
        if (prev_gly_opp) {
            total = total + 3.60;
        }
    }

    // 4. Intraresidue C'(k)
    if (n_c_curr > 0u) {
        total = total + 0.85 * best_cand_llr(
            c_curr_off, n_c_curr, cur_c_mu, cur_c_sd, bg_c_off, 20u, 0.7, llr_floor,
        );
    }

    // 5. Aliphatic sidechain CCC(CO)N peaks of k - 1
    if (n_sc_prev > 0u) {
        if (prv_n_sc == 0u) {
            total = total - 1.25 * f32(min(n_sc_prev, 3u));
        } else {
            var sc_sum: f32 = 0.0;
            for (var isc: u32 = 0u; isc < n_sc_prev; isc = isc + 1u) {
                let v = ro_floats[sc_prev_off + isc];
                var best_comp: f32 = -1.0e30;
                for (var m: u32 = 0u; m < prv_n_sc; m = m + 1u) {
                    let mu_s = ro_floats[prv_qf + 8u + m];
                    let lp = logpdf_2comp(v, mu_s, 0.75);
                    if (lp > best_comp) {
                        best_comp = lp;
                    }
                }
                let llr = max(best_comp - bg_logpdf(v, bg_sc_off, 12u, 1.8), -4.0);
                sc_sum = sc_sum + 0.55 * llr;
            }
            total = total + sc_sum;
        }
    }

    rw_floats[flat_idx] = total;
}

fn pair_channel_llr(
    a_off: u32,
    a_len: u32,
    b_off: u32,
    b_len: u32,
    sigma: f32,
    range_ppm: f32,
    pair_floor: f32,
) -> f32 {
    if (a_len == 0u || b_len == 0u) {
        return 0.0;
    }
    let s = max(sigma, 0.02);
    let bg_log = log(1.0 / max(range_ppm, 2.0));
    let norm_log = log(INV_SQRT_2PI / s);
    var best: f32 = -1.0e30;
    for (var ia: u32 = 0u; ia < a_len; ia = ia + 1u) {
        let va = ro_floats[a_off + ia];
        for (var ib: u32 = 0u; ib < b_len; ib = ib + 1u) {
            let vb = ro_floats[b_off + ib];
            let z = (va - vb) / s;
            let lp = norm_log - 0.5 * z * z - 0.14 * f32(ia + ib);
            let llr = max(lp - bg_log, pair_floor);
            if (llr > best) {
                best = llr;
            }
        }
    }
    let mult_gate = clamp(best / 1.5, 0.0, 1.0);
    return best - 0.18 * log(f32(a_len * b_len)) * mult_gate;
}

fn raw_seq_pair(a: u32, b: u32) -> f32 {
    let sf_a = params.off_sys_floats + a * SYS_FLOAT_STRIDE;
    let si_a = params.off_sys_ints + a * SYS_INT_STRIDE;
    let sf_b = params.off_sys_floats + b * SYS_FLOAT_STRIDE;
    let si_b = params.off_sys_ints + b * SYS_INT_STRIDE;

    let sc_ca = pair_channel_llr(
        sf_a + 2u,
        ro_ints[si_a + 0u],
        sf_b + 8u,
        ro_ints[si_b + 1u],
        params.sigma_seq_ca,
        28.0,
        params.pair_floor,
    );
    let sc_c = pair_channel_llr(
        sf_a + 20u,
        ro_ints[si_a + 3u],
        sf_b + 1u,
        1u,
        params.sigma_seq_c,
        13.0,
        params.pair_floor,
    );
    return sc_ca + sc_c;
}

fn pair_compat_weight(raw: f32) -> f32 {
    let u = clamp((raw - 2.10) / 2.40, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

@compute @workgroup_size(64, 1, 1)
fn compute_idp_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_cells = params.s_len * params.s_len;
    let flat_idx = gid.x;
    if (flat_idx >= total_cells) {
        return;
    }
    let s = flat_idx / params.s_len;
    let t = flat_idx - s * params.s_len;
    let out_base = select(0u, params.s_len * params.k_len, params.score_out_mode > 0.5);
    if (s == t) {
        rw_floats[out_base + flat_idx] = FORBID;
        return;
    }

    let raw_st = raw_seq_pair(s, t);
    var final_st = raw_st;
    if (raw_st > 0.0 && params.w_mult > 0.0) {
        var sum_out: f32 = 0.0;
        var sum_in: f32 = 0.0;
        for (var u: u32 = 0u; u < params.s_len; u = u + 1u) {
            if (u != s) {
                sum_out = sum_out + pair_compat_weight(raw_seq_pair(s, u));
            }
            if (u != t) {
                sum_in = sum_in + pair_compat_weight(raw_seq_pair(u, t));
            }
        }
        let n_out = max(sum_out, 1.0);
        let n_in = max(sum_in, 1.0);
        let gate_mult = clamp(raw_st / 2.0, 0.0, 1.0);
        final_st = raw_st - gate_mult * params.w_mult * 0.5 * (log(n_out) + log(n_in));
    }

    rw_floats[out_base + flat_idx] = final_st;
}

// ============================================================================
// Multi-Replica Detailed-Balance Simulated Annealing (`256`, `1024`, `global`)
// ============================================================================

fn rotl_u32(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next_u32(st: ptr<function, vec4<u32>>) -> u32 {
    let s0 = (*st).x;
    var s1 = (*st).y;
    var s2 = (*st).z;
    var s3 = (*st).w;
    let res = rotl_u32(s0 + s3, 7u) + s0;
    let t = s1 << 9u;
    s2 = s2 ^ s0;
    s3 = s3 ^ s1;
    s1 = s1 ^ s2;
    let new_s0 = s0 ^ s3;
    s2 = s2 ^ t;
    s3 = rotl_u32(s3, 11u);
    *st = vec4<u32>(new_s0, s1, s2, s3);
    return res;
}

fn rand_f32(st: ptr<function, vec4<u32>>) -> f32 {
    return f32(next_u32(st) >> 8u) * (1.0 / 16777216.0);
}

fn rand_below(st: ptr<function, vec4<u32>>, upper: u32) -> u32 {
    return next_u32(st) % max(upper, 1u);
}

fn read_u(s: u32, k: u32) -> f32 {
    return ro_floats[params.off_u_mat + s * params.k_len + k];
}

fn read_p(a: u32, b: u32) -> f32 {
    return ro_floats[params.off_p_mat + a * params.s_len + b];
}

fn slots_start(s: u32) -> u32 {
    return ro_ints[params.off_slots_packed + s];
}

fn slots_count(s: u32) -> u32 {
    return ro_ints[params.off_slots_packed + s + 1u] - ro_ints[params.off_slots_packed + s];
}

fn slot_value(s: u32, opt: u32) -> u32 {
    let vals_base = params.off_slots_packed + params.s_len + 1u;
    return ro_ints[vals_base + slots_start(s) + opt];
}

fn eval_1_wg(wg_base: u32, k_len: u32, k0: u32) -> f32 {
    let s0 = wg_at[wg_base + k0];
    if (s0 < 0) {
        return 0.0;
    }
    let su = u32(s0);
    var sc = read_u(su, k0);
    if (k0 > 0u) {
        let a = wg_at[wg_base + k0 - 1u];
        if (a >= 0) {
            sc = sc + read_p(u32(a), su);
        }
    }
    if (k0 + 1u < k_len) {
        let b = wg_at[wg_base + k0 + 1u];
        if (b >= 0) {
            sc = sc + read_p(su, u32(b));
        }
    }
    return sc;
}

fn eval_2_wg(wg_base: u32, k_len: u32, k_a: u32, k_b: u32) -> f32 {
    let lo = min(k_a, k_b);
    let hi = max(k_a, k_b);
    var sc: f32 = 0.0;
    let s_lo = wg_at[wg_base + lo];
    let s_hi = wg_at[wg_base + hi];
    if (s_lo >= 0) {
        let u_lo = u32(s_lo);
        sc = sc + read_u(u_lo, lo);
        if (lo > 0u) {
            let a = wg_at[wg_base + lo - 1u];
            if (a >= 0) {
                sc = sc + read_p(u32(a), u_lo);
            }
        }
        let b_lo = wg_at[wg_base + lo + 1u];
        if (b_lo >= 0) {
            sc = sc + read_p(u_lo, u32(b_lo));
        }
    }
    if (s_hi >= 0) {
        let u_hi = u32(s_hi);
        sc = sc + read_u(u_hi, hi);
        if (hi > lo + 1u) {
            let a_hi = wg_at[wg_base + hi - 1u];
            if (a_hi >= 0) {
                sc = sc + read_p(u32(a_hi), u_hi);
            }
        }
        if (hi + 1u < k_len) {
            let b_hi = wg_at[wg_base + hi + 1u];
            if (b_hi >= 0) {
                sc = sc + read_p(u_hi, u32(b_hi));
            }
        }
    }
    return sc;
}

fn eval_block_wg(wg_base: u32, k_len: u32, k1: u32, k2: u32, l: u32) -> f32 {
    let b0 = min(k1, k2);
    let b1 = max(k1, k2);
    var sc: f32 = 0.0;
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let s0 = wg_at[wg_base + b0 + i];
        if (s0 >= 0) {
            sc = sc + read_u(u32(s0), b0 + i);
        }
        let s1 = wg_at[wg_base + b1 + i];
        if (s1 >= 0) {
            sc = sc + read_u(u32(s1), b1 + i);
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
                sc = sc + read_p(u32(a), u32(b));
            }
        }
        let e1_start = max(b1 - 1u, e0_end);
        let e1_end = min(b1 + l, max_e);
        for (var e: u32 = e1_start; e < e1_end; e = e + 1u) {
            let a = wg_at[wg_base + e];
            let b = wg_at[wg_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + read_p(u32(a), u32(b));
            }
        }
    }
    return sc;
}

fn eval_total_wg(wg_base: u32, k_len: u32) -> f32 {
    var sc: f32 = 0.0;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = wg_at[wg_base + k];
        if (s >= 0) {
            sc = sc + read_u(u32(s), k);
        }
    }
    if (k_len > 1u) {
        for (var e: u32 = 0u; e + 1u < k_len; e = e + 1u) {
            let a = wg_at[wg_base + e];
            let b = wg_at[wg_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + read_p(u32(a), u32(b));
            }
        }
    }
    return sc;
}

fn run_anneal_wg(r: u32, wg_base: u32) {
    let s_len = params.s_len;
    let k_len = params.k_len;
    let at_base = r * k_len;
    let best_at_base = (params.replicas + r) * k_len;
    let pos_base = r * s_len;
    let best_pos_base = (params.replicas + r) * s_len;

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        wg_at[wg_base + k] = rw_at[at_base + k];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        wg_pos[wg_base + s] = rw_pos[pos_base + s];
    }

    var rng = rw_rng[r];
    var cur = eval_total_wg(wg_base, k_len);
    var best = rw_floats[2u * r + 1u];

    let denom = f32(max(params.total_steps - 1u, 1u));
    let log_ratio = log(max(params.t_end, 1.0e-4) / max(params.t_start, 1.0e-3));
    let max_blk = min(min(params.max_block, 16u), (k_len - 1u) / 2u);
    let end_step = min(params.step_offset + params.chunk_steps, params.total_steps);

    for (var step: u32 = params.step_offset; step < end_step; step = step + 1u) {
        let temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        let draw = rand_f32(&rng);

        if (draw < 0.70 || max_blk == 0u) {
            let s = rand_below(&rng, s_len);
            let sl_s = slots_count(s);
            if (sl_s == 0u) {
                continue;
            }
            let opt = rand_below(&rng, sl_s + 1u);
            let ks = wg_pos[wg_base + s];

            if (opt == sl_s) {
                if (ks < 0) {
                    continue;
                }
                let ksu = u32(ks);
                let before = eval_1_wg(wg_base, k_len, ksu);
                wg_at[wg_base + ksu] = -1;
                wg_pos[wg_base + s] = -1;
                let delta = -before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_wg(wg_base, k_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                rw_at[best_at_base + ki] = wg_at[wg_base + ki];
                            }
                            for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                                rw_pos[best_pos_base + si] = wg_pos[wg_base + si];
                            }
                        }
                    }
                } else {
                    wg_at[wg_base + ksu] = i32(s);
                    wg_pos[wg_base + s] = ks;
                }
            } else {
                let k = slot_value(s, opt);
                if (ks == i32(k)) {
                    continue;
                }
                let t = wg_at[wg_base + k];
                if (ks < 0) {
                    let before = eval_1_wg(wg_base, k_len, k);
                    var hastings_log: f32 = 0.0;
                    if (t >= 0) {
                        let sl_t = slots_count(u32(t));
                        hastings_log = log(f32(sl_s + 1u) / f32(sl_t + 1u));
                        wg_pos[wg_base + u32(t)] = -1;
                    }
                    wg_at[wg_base + k] = i32(s);
                    wg_pos[wg_base + s] = i32(k);
                    let delta = eval_1_wg(wg_base, k_len, k) - before;
                    let log_acc = (delta / temp) + hastings_log;
                    if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                        cur = cur + delta;
                        if (cur > best) {
                            cur = eval_total_wg(wg_base, k_len);
                            if (cur > best) {
                                best = cur;
                                for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                    rw_at[best_at_base + ki] = wg_at[wg_base + ki];
                                }
                                for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                                    rw_pos[best_pos_base + si] = wg_pos[wg_base + si];
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
                    let ksu = u32(ks);
                    if (t >= 0 && read_u(u32(t), ksu) <= FORBID_HALF) {
                        continue;
                    }
                    let before = eval_2_wg(wg_base, k_len, k, ksu);
                    wg_at[wg_base + k] = i32(s);
                    wg_at[wg_base + ksu] = t;
                    wg_pos[wg_base + s] = i32(k);
                    if (t >= 0) {
                        wg_pos[wg_base + u32(t)] = ks;
                    }
                    let delta = eval_2_wg(wg_base, k_len, k, ksu) - before;
                    if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                        cur = cur + delta;
                        if (cur > best) {
                            cur = eval_total_wg(wg_base, k_len);
                            if (cur > best) {
                                best = cur;
                                for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                    rw_at[best_at_base + ki] = wg_at[wg_base + ki];
                                }
                                for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                                    rw_pos[best_pos_base + si] = wg_pos[wg_base + si];
                                }
                            }
                        }
                    } else {
                        wg_at[wg_base + k] = t;
                        wg_at[wg_base + ksu] = i32(s);
                        wg_pos[wg_base + s] = ks;
                        if (t >= 0) {
                            wg_pos[wg_base + u32(t)] = i32(k);
                        }
                    }
                }
            }
        } else {
            let l = 1u + rand_below(&rng, max_blk);
            if (k_len < l + 2u) {
                continue;
            }
            let span = (k_len - 1u) - l + 1u;
            let k1 = 1u + rand_below(&rng, span);
            let k2 = 1u + rand_below(&rng, span);
            let diff = select(k2 - k1, k1 - k2, k1 >= k2);
            if (diff < l) {
                continue;
            }
            var valid = true;
            for (var i: u32 = 0u; i < l; i = i + 1u) {
                let a = wg_at[wg_base + k1 + i];
                let b = wg_at[wg_base + k2 + i];
                if ((b >= 0 && read_u(u32(b), k1 + i) <= FORBID_HALF)
                    || (a >= 0 && read_u(u32(a), k2 + i) <= FORBID_HALF))
                {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }
            let before = eval_block_wg(wg_base, k_len, k1, k2, l);
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
            let delta = eval_block_wg(wg_base, k_len, k1, k2, l) - before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    cur = eval_total_wg(wg_base, k_len);
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            rw_at[best_at_base + ki] = wg_at[wg_base + ki];
                        }
                        for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                            rw_pos[best_pos_base + si] = wg_pos[wg_base + si];
                        }
                    }
                }
            } else {
                for (var i: u32 = 0u; i < l; i = i + 1u) {
                    let b = wg_at[wg_base + k1 + i];
                    let a = wg_at[wg_base + k2 + i];
                    wg_at[wg_base + k1 + i] = a;
                    wg_at[wg_base + k2 + i] = b;
                    if (a >= 0) {
                        wg_pos[wg_base + u32(a)] = i32(k1 + i);
                    }
                    if (b >= 0) {
                        wg_pos[wg_base + u32(b)] = i32(k2 + i);
                    }
                }
            }
        }
    }

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        rw_at[at_base + k] = wg_at[wg_base + k];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        rw_pos[pos_base + s] = wg_pos[wg_base + s];
    }
    rw_floats[2u * r] = cur;
    rw_floats[2u * r + 1u] = best;
    rw_rng[r] = rng;
}

fn eval_1_global(at_base: u32, k_len: u32, k0: u32) -> f32 {
    let s0 = rw_at[at_base + k0];
    if (s0 < 0) {
        return 0.0;
    }
    let su = u32(s0);
    var sc = read_u(su, k0);
    if (k0 > 0u) {
        let a = rw_at[at_base + k0 - 1u];
        if (a >= 0) {
            sc = sc + read_p(u32(a), su);
        }
    }
    if (k0 + 1u < k_len) {
        let b = rw_at[at_base + k0 + 1u];
        if (b >= 0) {
            sc = sc + read_p(su, u32(b));
        }
    }
    return sc;
}

fn eval_2_global(at_base: u32, k_len: u32, k_a: u32, k_b: u32) -> f32 {
    let lo = min(k_a, k_b);
    let hi = max(k_a, k_b);
    var sc: f32 = 0.0;
    let s_lo = rw_at[at_base + lo];
    let s_hi = rw_at[at_base + hi];
    if (s_lo >= 0) {
        let u_lo = u32(s_lo);
        sc = sc + read_u(u_lo, lo);
        if (lo > 0u) {
            let a = rw_at[at_base + lo - 1u];
            if (a >= 0) {
                sc = sc + read_p(u32(a), u_lo);
            }
        }
        let b_lo = rw_at[at_base + lo + 1u];
        if (b_lo >= 0) {
            sc = sc + read_p(u_lo, u32(b_lo));
        }
    }
    if (s_hi >= 0) {
        let u_hi = u32(s_hi);
        sc = sc + read_u(u_hi, hi);
        if (hi > lo + 1u) {
            let a_hi = rw_at[at_base + hi - 1u];
            if (a_hi >= 0) {
                sc = sc + read_p(u32(a_hi), u_hi);
            }
        }
        if (hi + 1u < k_len) {
            let b_hi = rw_at[at_base + hi + 1u];
            if (b_hi >= 0) {
                sc = sc + read_p(u_hi, u32(b_hi));
            }
        }
    }
    return sc;
}

fn eval_total_global(at_base: u32, k_len: u32) -> f32 {
    var sc: f32 = 0.0;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = rw_at[at_base + k];
        if (s >= 0) {
            sc = sc + read_u(u32(s), k);
        }
    }
    if (k_len > 1u) {
        for (var e: u32 = 0u; e + 1u < k_len; e = e + 1u) {
            let a = rw_at[at_base + e];
            let b = rw_at[at_base + e + 1u];
            if (a >= 0 && b >= 0) {
                sc = sc + read_p(u32(a), u32(b));
            }
        }
    }
    return sc;
}

@compute @workgroup_size(8, 1, 1)
fn anneal_idp_256(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_SLOTS_256);
}

@compute @workgroup_size(2, 1, 1)
fn anneal_idp_1024(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_SLOTS_1024);
}

@compute @workgroup_size(8, 1, 1)
fn anneal_idp_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let s_len = params.s_len;
    let k_len = params.k_len;
    let at_base = r * k_len;
    let best_at_base = (params.replicas + r) * k_len;
    let pos_base = r * s_len;
    let best_pos_base = (params.replicas + r) * s_len;

    var rng = rw_rng[r];
    var cur = eval_total_global(at_base, k_len);
    var best = rw_floats[2u * r + 1u];

    let denom = f32(max(params.total_steps - 1u, 1u));
    let log_ratio = log(max(params.t_end, 1.0e-4) / max(params.t_start, 1.0e-3));
    let end_step = min(params.step_offset + params.chunk_steps, params.total_steps);

    for (var step: u32 = params.step_offset; step < end_step; step = step + 1u) {
        let temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        let s = rand_below(&rng, s_len);
        let sl_s = slots_count(s);
        if (sl_s == 0u) {
            continue;
        }
        let opt = rand_below(&rng, sl_s + 1u);
        let ks = rw_pos[pos_base + s];

        if (opt == sl_s) {
            if (ks < 0) {
                continue;
            }
            let ksu = u32(ks);
            let before = eval_1_global(at_base, k_len, ksu);
            rw_at[at_base + ksu] = -1;
            rw_pos[pos_base + s] = -1;
            let delta = -before;
            if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                cur = cur + delta;
                if (cur > best) {
                    cur = eval_total_global(at_base, k_len);
                    if (cur > best) {
                        best = cur;
                        for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                            rw_at[best_at_base + ki] = rw_at[at_base + ki];
                        }
                        for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                            rw_pos[best_pos_base + si] = rw_pos[pos_base + si];
                        }
                    }
                }
            } else {
                rw_at[at_base + ksu] = i32(s);
                rw_pos[pos_base + s] = ks;
            }
        } else {
            let k = slot_value(s, opt);
            if (ks == i32(k)) {
                continue;
            }
            let t = rw_at[at_base + k];
            if (ks < 0) {
                let before = eval_1_global(at_base, k_len, k);
                var hastings_log: f32 = 0.0;
                if (t >= 0) {
                    let sl_t = slots_count(u32(t));
                    hastings_log = log(f32(sl_s + 1u) / f32(sl_t + 1u));
                    rw_pos[pos_base + u32(t)] = -1;
                }
                rw_at[at_base + k] = i32(s);
                rw_pos[pos_base + s] = i32(k);
                let delta = eval_1_global(at_base, k_len, k) - before;
                let log_acc = (delta / temp) + hastings_log;
                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_global(at_base, k_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                rw_at[best_at_base + ki] = rw_at[at_base + ki];
                            }
                            for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                                rw_pos[best_pos_base + si] = rw_pos[pos_base + si];
                            }
                        }
                    }
                } else {
                    rw_pos[pos_base + s] = -1;
                    rw_at[at_base + k] = t;
                    if (t >= 0) {
                        rw_pos[pos_base + u32(t)] = i32(k);
                    }
                }
            } else {
                let ksu = u32(ks);
                if (t >= 0 && read_u(u32(t), ksu) <= FORBID_HALF) {
                    continue;
                }
                let before = eval_2_global(at_base, k_len, k, ksu);
                rw_at[at_base + k] = i32(s);
                rw_at[at_base + ksu] = t;
                rw_pos[pos_base + s] = i32(k);
                if (t >= 0) {
                    rw_pos[pos_base + u32(t)] = ks;
                }
                let delta = eval_2_global(at_base, k_len, k, ksu) - before;
                if (delta >= 0.0 || rand_f32(&rng) < exp(delta / temp)) {
                    cur = cur + delta;
                    if (cur > best) {
                        cur = eval_total_global(at_base, k_len);
                        if (cur > best) {
                            best = cur;
                            for (var ki: u32 = 0u; ki < k_len; ki = ki + 1u) {
                                rw_at[best_at_base + ki] = rw_at[at_base + ki];
                            }
                            for (var si: u32 = 0u; si < s_len; si = si + 1u) {
                                rw_pos[best_pos_base + si] = rw_pos[pos_base + si];
                            }
                        }
                    }
                } else {
                    rw_at[at_base + k] = t;
                    rw_at[at_base + ksu] = i32(s);
                    rw_pos[pos_base + s] = ks;
                    if (t >= 0) {
                        rw_pos[pos_base + u32(t)] = i32(k);
                    }
                }
            }
        }
    }

    rw_floats[2u * r] = cur;
    rw_floats[2u * r + 1u] = best;
    rw_rng[r] = rng;
}
