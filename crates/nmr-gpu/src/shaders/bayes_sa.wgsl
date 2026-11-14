// Clean-room WebGPU (WGSL) compute shaders for Bayesian pseudo-energy evaluation
// (`E_type`, `E_unary`, `E_link`, and `Q27.36` 64-bit integer state energy) and massively
// parallel multi-replica specific-heat C_v(T) adaptive simulated annealing.
//
// Provenance (100% Clean-Room):
// 1. Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J. (2023).
//    "Robust automated backbone triple resonance NMR assignments of proteins using
//    Bayesian-based simulated annealing." Nature Communications 14, 1556.
//    DOI: 10.1038/s41467-023-37219-z.
// 2. Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983). "Optimization by Simulated
//    Annealing." Science 220(4598), 671-680.
// 3. Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (1995).
//    J. Biomol. NMR 5, 67-81 & 6, 135-140 (primary-sequence covalent neighbor shifts).
//
// Workgroup SRAM budget: `wg_packed: array<u32, 4096>` = 16,384 B (exact portable WebGPU
// baseline limit, packing `at[i]` in low signed `i16` and `pos[i]` in high signed `i16`):
// - `anneal_bayes_256`  (`@workgroup_size(8, 1, 1)`): 8 replicas/workgroup, <= 512 residues/spins (incl. 480 aa).
// - `anneal_bayes_1024` (`@workgroup_size(4, 1, 1)`): 4 replicas/workgroup, <= 1024 residues/spins (513-1024 aa).
// - `anneal_bayes_global` (`@workgroup_size(32, 1, 1)`): storage-backed state for > 1024 residues/spins.

struct BayesGpuParams {
    s_len: u32,
    k_len: u32,
    replicas: u32,
    n_fk: u32,

    n_fs: u32,
    cand_m: u32,
    max_w: u32,
    stage_steps: u32,

    step_offset: u32,
    chunk_steps: u32,
    burn_in: u32,
    use_hn: u32,

    outlier_eps: f32,
    p_leak_std: f32,
    p_leak_deg: f32,
    mu_null: f32,

    sigma_ca: f32,
    sigma_cb: f32,
    sigma_c: f32,
    sigma_ha: f32,

    inv_temp: f32,
    r_glob: f32,
    r_cand: f32,
    r_ext: f32,

    off_f0: u32,
    off_f1: u32,
    off_f2: u32,
    off_f3: u32,

    off_f4: u32,
    off_f5: u32,
    off_f6: u32,
    off_f7: u32,

    off_f8: u32,
    off_f9: u32,
    off_f10: u32,
    off_f11: u32,

    off_f12: u32,
    off_f13: u32,
    off_f14: u32,
    off_f15: u32,

    off_i0: u32,
    off_i1: u32,
    off_i2: u32,
    off_i3: u32,

    off_i4: u32,
    off_i5: u32,
    off_i6: u32,
    off_i7: u32,

    off_i8: u32,
    off_mat_link_out: u32,
    _pad_u0: u32,
    _pad_u1: u32,
};

@group(0) @binding(0) var<uniform> params: BayesGpuParams;
@group(0) @binding(1) var<storage, read> ro_floats: array<f32>;
@group(0) @binding(2) var<storage, read> ro_ints: array<i32>;
@group(0) @binding(3) var<storage, read_write> rw_mat_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> rw_mat_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> at_buf: array<i32>;
@group(0) @binding(6) var<storage, read_write> pos_buf: array<i32>;
@group(0) @binding(7) var<storage, read_write> stats_f_buf: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> state_u_buf: array<vec4<u32>>;

const FORBID_ENERGY: f32 = 1.0e9;
const FORBID_HALF: f32 = 5.0e8;
const STUDENT_T5_NORM: f32 = 0.37994380;
const GLY_OBSERVED_CB_LOG_BF: f32 = -3.9120230;
const SQRT_TAU: f32 = 2.50662827;
const GLY_RES_IDX: i32 = 7;
const PRO_RES_IDX: i32 = 14;
const MAX_SLOT_CANDS: u32 = 4u;
const P_RESTORE_EVICT: f32 = 0.35;
const P_EVICT_WHEN_SWAP_OK: f32 = 0.18;

var<workgroup> wg_packed: array<u32, 4096>;

fn wg_get_at(base: u32, k: u32) -> i32 {
    return bitcast<i32>(wg_packed[base + k] << 16u) >> 16u;
}

fn wg_set_at(base: u32, k: u32, val: i32) {
    let slot = base + k;
    wg_packed[slot] = (wg_packed[slot] & 0xFFFF0000u) | (bitcast<u32>(val) & 0x0000FFFFu);
}

fn wg_get_pos(base: u32, s: u32) -> i32 {
    return bitcast<i32>(wg_packed[base + s]) >> 16u;
}

fn wg_set_pos(base: u32, s: u32, val: i32) {
    let slot = base + s;
    wg_packed[slot] = (wg_packed[slot] & 0x0000FFFFu) | ((bitcast<u32>(val) & 0x0000FFFFu) << 16u);
}

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

fn chan_range(a_idx: u32) -> f32 {
    switch (a_idx) {
        case 0u: { return 36.0; }
        case 1u: { return 64.0; }
        case 2u: { return 24.0; }
        case 3u: { return 3.2; }
        case 4u: { return 38.0; }
        default: { return 4.5; }
    }
}

fn link_sigma(a_idx: u32) -> f32 {
    switch (a_idx) {
        case 0u: { return params.sigma_ca; }
        case 1u: { return params.sigma_cb; }
        case 2u: { return params.sigma_c; }
        default: { return params.sigma_ha; }
    }
}

fn student_t5_robust_pdf(x_c: f32, mu_c: f32, sig_t: f32, range_ppm: f32, eps: f32) -> f32 {
    let z = (x_c - mu_c) / sig_t;
    let u = 1.0 + 0.2 * z * z;
    let t_pdf = STUDENT_T5_NORM / (sig_t * u * u * u);
    return (1.0 - eps) * t_pdf + eps / range_ppm;
}

fn eval_channel_bg_density(a_idx: u32, x_c: f32) -> f32 {
    let range_ppm = chan_range(a_idx);
    let eps = params.outlier_eps;
    var sum: f32 = 0.0;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let pi_r = ro_floats[params.off_f3 + r * 6u + a_idx];
        if (pi_r <= 0.0) {
            continue;
        }
        var r_pdf: f32 = 0.0;
        for (var c: u32 = 0u; c < 3u; c = c + 1u) {
            let w_c = ro_floats[params.off_f2 + r * 3u + c];
            let base = (r * 3u + c) * 6u + a_idx;
            let mu = ro_floats[params.off_f0 + base];
            let sig_t = ro_floats[params.off_f1 + base];
            r_pdf = r_pdf + w_c * student_t5_robust_pdf(x_c, mu, sig_t, range_ppm, eps);
        }
        sum = sum + pi_r * r_pdf;
    }
    return max(0.985 * sum + 0.015 / range_ppm, 1.0e-12);
}

fn eval_slot_log_bf(
    s: u32,
    u_slot: u32,
    a_idx: u32,
    res_idx: u32,
    conf_c: u32,
    use_seq: bool,
    k_pos: u32,
) -> f32 {
    let cnt = u32(ro_ints[params.off_i2 + s * 10u + u_slot]);
    if (cnt == 0u) {
        return 0.0;
    }
    let present = ro_ints[params.off_i0 + res_idx * 6u + a_idx];
    if (present == 0) {
        if (a_idx == 1u && i32(res_idx) == GLY_RES_IDX) {
            return GLY_OBSERVED_CB_LOG_BF;
        }
        return 0.0;
    }
    var mu: f32 = 0.0;
    var sig_t: f32 = 1.0;
    if (use_seq) {
        let idx = (k_pos * 3u + conf_c) * 6u + a_idx;
        mu = ro_floats[params.off_f7 + idx];
        sig_t = ro_floats[params.off_f8 + idx];
    } else {
        let idx = (res_idx * 3u + conf_c) * 6u + a_idx;
        mu = ro_floats[params.off_f0 + idx];
        sig_t = ro_floats[params.off_f1 + idx];
    }
    let range_ppm = chan_range(a_idx);
    let eps = params.outlier_eps;
    let base_cell = (s * 10u + u_slot) * MAX_SLOT_CANDS;
    var bf_sum: f32 = 0.0;
    for (var j: u32 = 0u; j < cnt; j = j + 1u) {
        let x = ro_floats[params.off_f10 + base_cell + j];
        let w = ro_floats[params.off_f11 + base_cell + j];
        let p_fg = student_t5_robust_pdf(x, mu, sig_t, range_ppm, eps);
        let p_bg = eval_channel_bg_density(a_idx, x);
        bf_sum = bf_sum + w * (p_fg / p_bg);
    }
    return log(max(bf_sum, 1.0e-12));
}

fn logsumexp9(v: array<f32, 9>) -> f32 {
    var m: f32 = -1.0e30;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        m = max(m, v[i]);
    }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        s = s + exp(v[i] - m);
    }
    return m + log(s);
}

fn logsumexp20(v: array<f32, 20>) -> f32 {
    var m: f32 = -1.0e30;
    for (var i: u32 = 0u; i < 20u; i = i + 1u) {
        m = max(m, v[i]);
    }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < 20u; i = i + 1u) {
        if (v[i] > -1.0e20) {
            s = s + exp(v[i] - m);
        }
    }
    return m + log(s);
}

// ============================================================================
// Entry Points 1-4: Bayesian Pseudo-Energy Matrices & State Energy Evaluation
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn compute_ln_z_bg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.s_len) {
        return;
    }

    var ln_bi: array<f32, 3>;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        var terms_i: array<f32, 20>;
        for (var r: u32 = 0u; r < 20u; r = r + 1u) {
            let ln_pi = ro_floats[params.off_f4 + r];
            if (ln_pi <= -1.0e20) {
                terms_i[r] = -1.0e30;
            } else {
                var ll: f32 = 0.0;
                for (var a_idx: u32 = 0u; a_idx < 6u; a_idx = a_idx + 1u) {
                    ll = ll + eval_slot_log_bf(s, a_idx, a_idx, r, c, false, 0u);
                }
                terms_i[r] = ln_pi + ll;
            }
        }
        ln_bi[c] = logsumexp20(terms_i);
    }

    var ln_bim1: array<f32, 3>;
    for (var cp: u32 = 0u; cp < 3u; cp = cp + 1u) {
        var terms_im1: array<f32, 20>;
        for (var rp: u32 = 0u; rp < 20u; rp = rp + 1u) {
            var ll: f32 = 0.0;
            for (var a_idx: u32 = 0u; a_idx < 4u; a_idx = a_idx + 1u) {
                ll = ll + eval_slot_log_bf(s, 6u + a_idx, a_idx, rp, cp, false, 0u);
            }
            terms_im1[rp] = ro_floats[params.off_f5 + rp] + ll;
        }
        ln_bim1[cp] = logsumexp20(terms_im1);
    }

    var joint9: array<f32, 9>;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        for (var cp: u32 = 0u; cp < 3u; cp = cp + 1u) {
            let j_idx = 3u * c + cp;
            joint9[j_idx] = ro_floats[params.off_f6 + j_idx] + ln_bi[c] + ln_bim1[cp];
        }
    }
    rw_mat_a[s] = logsumexp9(joint9);
}

@compute @workgroup_size(16, 16, 1)
fn compute_e_type(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let k = gid.y;
    if (s >= params.s_len || k >= params.k_len) {
        return;
    }
    let cell = s * params.k_len + k;
    let et_off = params.off_f15 + cell;
    if (k == 0u) {
        rw_mat_a[et_off] = FORBID_ENERGY;
        rw_mat_b[cell] = FORBID_ENERGY;
        return;
    }

    let rk = ro_ints[params.off_i1 + k];
    if (rk == PRO_RES_IDX) {
        rw_mat_a[et_off] = FORBID_ENERGY;
        rw_mat_b[cell] = FORBID_ENERGY;
        return;
    }

    let lock_ks = ro_ints[params.off_i7 + s];
    let lock_sk = ro_ints[params.off_i8 + k];
    let is_exact_lock = (lock_ks == i32(k)) && (lock_sk == i32(s));
    if (!is_exact_lock) {
        if (lock_ks >= 0 || lock_sk >= 0 || ro_ints[params.off_i5 + s] != 0) {
            rw_mat_a[et_off] = FORBID_ENERGY;
            rw_mat_b[cell] = FORBID_ENERGY;
            return;
        }
        let rkm1_chk = ro_ints[params.off_i1 + (k - 1u)];
        if (ro_ints[params.off_i6 + s] != 0 && rk != GLY_RES_IDX && rkm1_chk != PRO_RES_IDX) {
            rw_mat_a[et_off] = FORBID_ENERGY;
            rw_mat_b[cell] = FORBID_ENERGY;
            return;
        }
    }

    let rkm1 = ro_ints[params.off_i1 + (k - 1u)];
    var raw_et: f32 = 0.0;
    if (rk >= 0 && rk < 20 && rkm1 >= 0 && rkm1 < 20) {
        let rk_u = u32(rk);
        let rkm1_u = u32(rkm1);
        var ll_i: array<f32, 3>;
        for (var c: u32 = 0u; c < 3u; c = c + 1u) {
            var acc: f32 = 0.0;
            for (var a_idx: u32 = 0u; a_idx < 6u; a_idx = a_idx + 1u) {
                acc = acc + eval_slot_log_bf(s, a_idx, a_idx, rk_u, c, true, k);
            }
            ll_i[c] = acc;
        }
        var ll_im1: array<f32, 3>;
        for (var cp: u32 = 0u; cp < 3u; cp = cp + 1u) {
            var acc: f32 = 0.0;
            for (var a_idx: u32 = 0u; a_idx < 4u; a_idx = a_idx + 1u) {
                acc = acc + eval_slot_log_bf(s, 6u + a_idx, a_idx, rkm1_u, cp, true, k - 1u);
            }
            ll_im1[cp] = acc;
        }
        var joint9: array<f32, 9>;
        for (var c: u32 = 0u; c < 3u; c = c + 1u) {
            for (var cp: u32 = 0u; cp < 3u; cp = cp + 1u) {
                let jp = ro_floats[params.off_f9 + k * 9u + 3u * c + cp];
                joint9[3u * c + cp] = jp + ll_i[c] + ll_im1[cp];
            }
        }
        let ln_z_fg = logsumexp9(joint9);
        raw_et = -(ln_z_fg - rw_mat_a[s]);
    }

    if (is_exact_lock) {
        let clamped = clamp(raw_et, -500.0, 500.0);
        rw_mat_a[et_off] = clamped;
        rw_mat_b[cell] = clamped;
    } else {
        rw_mat_a[et_off] = raw_et;
        rw_mat_b[cell] = raw_et + ro_floats[params.off_f14 + s];
    }
}

@compute @workgroup_size(16, 16, 1)
fn compute_e_link(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let t = gid.y;
    if (s >= params.s_len || t >= params.s_len) {
        return;
    }
    let out_idx = params.off_mat_link_out + s * params.s_len + t;
    if (s == t) {
        rw_mat_b[out_idx] = FORBID_ENERGY;
        return;
    }

    var link_e: f32 = 0.0;
    for (var a_idx: u32 = 0u; a_idx < 4u; a_idx = a_idx + 1u) {
        let s_slot = s * 8u + a_idx;
        let t_slot = t * 8u + 4u + a_idx;
        let na = u32(ro_ints[params.off_i3 + s_slot]);
        let nb = u32(ro_ints[params.off_i3 + t_slot]);
        if (na == 0u || nb == 0u) {
            continue;
        }
        let sig = link_sigma(a_idx);
        let norm_gauss = 1.0 / (SQRT_TAU * sig);
        let only_deg = (ro_ints[params.off_i4 + s_slot] != 0) || (ro_ints[params.off_i4 + t_slot] != 0);
        let p_leak = select(params.p_leak_std, params.p_leak_deg, only_deg);

        let s_base = s_slot * MAX_SLOT_CANDS;
        let t_base = t_slot * MAX_SLOT_CANDS;
        var overlap_sum: f32 = 0.0;
        for (var ia: u32 = 0u; ia < na; ia = ia + 1u) {
            let xa = ro_floats[params.off_f12 + s_base + ia];
            let wa = ro_floats[params.off_f13 + s_base + ia];
            for (var ib: u32 = 0u; ib < nb; ib = ib + 1u) {
                let yb = ro_floats[params.off_f12 + t_base + ib];
                let wb = ro_floats[params.off_f13 + t_base + ib];
                let d = (xa - yb) / sig;
                let g = norm_gauss * exp(-0.5 * d * d);
                let x_mid = 0.5 * (xa + yb);
                let p_rand = eval_channel_bg_density(a_idx, x_mid);
                overlap_sum = overlap_sum + (wa * wb) * (g / p_rand);
            }
        }
        let bf = (1.0 - p_leak) * overlap_sum + p_leak;
        link_e = link_e - log(max(bf, 1.0e-12));
    }
    rw_mat_b[out_idx] = link_e;
}

fn i64_add(acc: vec2<u32>, term: vec2<u32>) -> vec2<u32> {
    let lo = acc.x + term.x;
    let carry = select(0u, 1u, lo < acc.x);
    let hi = acc.y + term.y + carry;
    return vec2<u32>(lo, hi);
}

@compute @workgroup_size(64, 1, 1)
fn eval_state_energies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.replicas) {
        return;
    }
    let k_len = params.k_len;
    let s_len = params.s_len;
    let base_at = idx * k_len;
    let eu_off = params.off_f0;
    let el_off = params.off_f1;

    var acc = vec2<u32>(0u, 0u);

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = at_buf[base_at + k];
        if (s >= 0) {
            let cell2 = 2u * (eu_off + u32(s) * k_len + k);
            let lo = bitcast<u32>(ro_ints[cell2]);
            let hi = ro_ints[cell2 + 1u];
            if (hi == 2147483647 && lo == 4294967295u) {
                rw_mat_a[2u * idx] = bitcast<f32>(4294967295u);
                rw_mat_a[2u * idx + 1u] = bitcast<f32>(2147483647u);
                return;
            }
            acc = i64_add(acc, vec2<u32>(lo, bitcast<u32>(hi)));
        }
    }
    for (var k: u32 = 1u; k < k_len; k = k + 1u) {
        let sa = at_buf[base_at + k - 1u];
        let sb = at_buf[base_at + k];
        if (sa >= 0 && sb >= 0) {
            let cell2 = 2u * (el_off + u32(sa) * s_len + u32(sb));
            let lo = bitcast<u32>(ro_ints[cell2]);
            let hi = ro_ints[cell2 + 1u];
            acc = i64_add(acc, vec2<u32>(lo, bitcast<u32>(hi)));
        }
    }
    rw_mat_a[2u * idx] = bitcast<f32>(acc.x);
    rw_mat_a[2u * idx + 1u] = bitcast<f32>(acc.y);
}

// ============================================================================
// Move Proposal Helpers & Exact O(1) Local Energy Deltas (SRAM & Global)
// ============================================================================

fn in_cand_spins(k: u32, s: i32) -> bool {
    if (s < 0 || params.cand_m == 0u) {
        return false;
    }
    let base = params.off_i4 + k * params.cand_m;
    var lo: u32 = 0u;
    var hi: u32 = params.cand_m;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let v = ro_ints[base + mid];
        if (v == s) {
            return true;
        }
        if (v > s) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return false;
}

fn in_list8(off_tbl: u32, row: i32, needle: i32) -> bool {
    if (row < 0 || needle < 0) {
        return false;
    }
    let base = off_tbl + u32(row) * 8u;
    var lo: u32 = 0u;
    var hi: u32 = 8u;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let v = ro_ints[base + mid];
        if (v == needle) {
            return true;
        }
        if (v > needle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return false;
}

fn evict_slot_weight(displaced_spin: i32, slot_m: u32, n_fk_f: f32) -> f32 {
    var w: f32 = 0.50 / n_fk_f;
    if (in_list8(params.off_i5, displaced_spin, i32(slot_m))) {
        w = w + (0.50 / 8.0);
    }
    return w;
}

fn get_eu(s: u32, k: u32) -> f32 {
    return ro_floats[params.off_f0 + s * params.k_len + k];
}

fn get_el(sa: u32, sb: u32) -> f32 {
    return ro_floats[params.off_f1 + sa * params.s_len + sb];
}

// --- Workgroup SRAM state accessors (`wg_packed` i16-paired 16 KiB SRAM) ---

fn pair_prop_w_wg(base_k: u32, k: u32, s: i32) -> f32 {
    var w = params.r_glob;
    if (params.r_cand > 0.0 && in_cand_spins(k, s)) {
        w = w + params.r_cand;
    }
    if (k > 0u && in_list8(params.off_i6, wg_get_at(base_k, k - 1u), s)) {
        w = w + params.r_ext;
    }
    if (k + 1u < params.k_len && in_list8(params.off_i7, wg_get_at(base_k, k + 1u), s)) {
        w = w + params.r_ext;
    }
    return w;
}

fn block_swap_prop_w_wg(base_k: u32, ia: u32, ib: u32, l: u32, m_start_f: f32) -> f32 {
    var w: f32 = 0.80 / m_start_f;
    let r_guided: f32 = 0.15 / 8.0;
    let ka0 = u32(ro_ints[params.off_i0 + ia]);
    let kb0 = u32(ro_ints[params.off_i0 + ib]);
    let ka_end = u32(ro_ints[params.off_i0 + ia + l - 1u]);
    let kb_end = u32(ro_ints[params.off_i0 + ib + l - 1u]);

    let sa0 = wg_get_at(base_k, ka0);
    let sb0 = wg_get_at(base_k, kb0);
    let sa_e = wg_get_at(base_k, ka_end);
    let sb_e = wg_get_at(base_k, kb_end);

    if (ka0 > 1u && in_list8(params.off_i6, wg_get_at(base_k, ka0 - 1u), sb0)) { w = w + r_guided; }
    if (kb0 > 1u && in_list8(params.off_i6, wg_get_at(base_k, kb0 - 1u), sa0)) { w = w + r_guided; }
    if (ka_end + 1u < params.k_len && in_list8(params.off_i7, wg_get_at(base_k, ka_end + 1u), sb_e)) { w = w + r_guided; }
    if (kb_end + 1u < params.k_len && in_list8(params.off_i7, wg_get_at(base_k, kb_end + 1u), sa_e)) { w = w + r_guided; }

    if (in_list8(params.off_i5, sa0, i32(kb0))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sb0, i32(ka0))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sa_e, i32(kb_end))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sb_e, i32(ka_end))) { w = w + r_guided; }
    return w;
}

fn eval_full_wg(base_k: u32) -> f32 {
    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    let k_len = params.k_len;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = wg_get_at(base_k, k);
        if (s >= 0) {
            let y = get_eu(u32(s), k) - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }
    for (var k: u32 = 1u; k < k_len; k = k + 1u) {
        let sa = wg_get_at(base_k, k - 1u);
        let sb = wg_get_at(base_k, k);
        if (sa >= 0 && sb >= 0) {
            let y = get_el(u32(sa), u32(sb)) - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

fn site_e_wg(base_k: u32, k: u32) -> f32 {
    let s = wg_get_at(base_k, k);
    if (s < 0) {
        return 0.0;
    }
    let su = u32(s);
    var e = get_eu(su, k);
    if (k > 0u) {
        let sl = wg_get_at(base_k, k - 1u);
        if (sl >= 0) {
            e = e + get_el(u32(sl), su);
        }
    }
    if (k + 1u < params.k_len) {
        let sr = wg_get_at(base_k, k + 1u);
        if (sr >= 0) {
            e = e + get_el(su, u32(sr));
        }
    }
    return e;
}

fn two_site_e_wg(base_k: u32, ka: u32, kb: u32) -> f32 {
    let lo = min(ka, kb);
    let hi = max(ka, kb);
    let slo = wg_get_at(base_k, lo);
    let shi = wg_get_at(base_k, hi);
    var e: f32 = 0.0;
    if (slo >= 0) {
        let su = u32(slo);
        e = e + get_eu(su, lo);
        if (lo > 0u) {
            let sl = wg_get_at(base_k, lo - 1u);
            if (sl >= 0) { e = e + get_el(u32(sl), su); }
        }
        let sr = wg_get_at(base_k, lo + 1u);
        if (sr >= 0) { e = e + get_el(su, u32(sr)); }
    }
    if (shi >= 0) {
        let su = u32(shi);
        e = e + get_eu(su, hi);
        if (hi > lo + 1u) {
            let sl = wg_get_at(base_k, hi - 1u);
            if (sl >= 0) { e = e + get_el(u32(sl), su); }
        }
        if (hi + 1u < params.k_len) {
            let sr = wg_get_at(base_k, hi + 1u);
            if (sr >= 0) { e = e + get_el(su, u32(sr)); }
        }
    }
    return e;
}

fn interval_e_wg(base_k: u32, start: u32, len: u32) -> f32 {
    let end_k = start + len;
    var e: f32 = 0.0;
    for (var k: u32 = start; k < end_k; k = k + 1u) {
        let s = wg_get_at(base_k, k);
        if (s >= 0) {
            e = e + get_eu(u32(s), k);
        }
    }
    let edge_lo = max(start, 1u);
    let edge_hi = min(end_k, params.k_len - 1u);
    for (var k: u32 = edge_lo; k <= edge_hi; k = k + 1u) {
        let sa = wg_get_at(base_k, k - 1u);
        let sb = wg_get_at(base_k, k);
        if (sa >= 0 && sb >= 0) {
            e = e + get_el(u32(sa), u32(sb));
        }
    }
    return e;
}

fn two_spans_e_wg(base_k: u32, a_start: u32, a_len: u32, b_start: u32, b_len: u32) -> f32 {
    var lo_s = a_start;
    var lo_len = a_len;
    var hi_s = b_start;
    var hi_len = b_len;
    if (b_start < a_start) {
        lo_s = b_start;
        lo_len = b_len;
        hi_s = a_start;
        hi_len = a_len;
    }
    var e = interval_e_wg(base_k, lo_s, lo_len) + interval_e_wg(base_k, hi_s, hi_len);
    if (lo_s + lo_len == hi_s) {
        let sa = wg_get_at(base_k, hi_s - 1u);
        let sb = wg_get_at(base_k, hi_s);
        if (sa >= 0 && sb >= 0) {
            e = e - get_el(u32(sa), u32(sb));
        }
    }
    return e;
}

fn run_anneal_wg_impl(r: u32, base: u32) {
    let k_len = params.k_len;
    let s_len = params.s_len;
    let max_dim = max(k_len, s_len);
    let n_fk = params.n_fk;
    let n_fs = params.n_fs;
    let n_fk_f = f32(n_fk);
    let best_base = (params.replicas + r) * k_len;
    let cur_base = r * k_len;
    let pos_base = r * s_len;

    for (var i: u32 = 0u; i < max_dim; i = i + 1u) {
        var av: i32 = -1;
        var pv: i32 = -1;
        if (i < k_len) { av = at_buf[cur_base + i]; }
        if (i < s_len) { pv = pos_buf[pos_base + i]; }
        wg_packed[base + i] = (bitcast<u32>(av) & 0x0000FFFFu) | ((bitcast<u32>(pv) & 0x0000FFFFu) << 16u);
    }

    var sf = stats_f_buf[r];
    var su = state_u_buf[r];
    var rng = state_u_buf[params.replicas + r];

    var cur_e = sf.x;
    var best_e = sf.y;
    var mean_e = sf.z;
    var m2_e = sf.w;
    var n_samples = su.x;
    var n_accepted = su.y;
    var n_proposed = su.z;

    if (params.step_offset == 0u) {
        cur_e = eval_full_wg(base);
        mean_e = 0.0;
        m2_e = 0.0;
        n_samples = 0u;
        n_accepted = 0u;
        n_proposed = 0u;
        if (cur_e < best_e) {
            best_e = cur_e;
            for (var k: u32 = 0u; k < k_len; k = k + 1u) {
                at_buf[best_base + k] = wg_get_at(base, k);
            }
        }
    }

    if (n_fk == 0u || n_fs == 0u || params.chunk_steps == 0u) {
        stats_f_buf[r] = vec4<f32>(cur_e, best_e, cur_e, 0.0);
        state_u_buf[r] = vec4<u32>(1u, 0u, 0u, 0u);
        return;
    }

    let inv_t = params.inv_temp;
    let max_w = clamp(params.max_w, 2u, 32u);
    let has_cands = params.cand_m > 0u;

    for (var step_local: u32 = 0u; step_local < params.chunk_steps; step_local = step_local + 1u) {
        let global_step = params.step_offset + step_local;
        n_proposed = n_proposed + 1u;
        let u_move = rand_f32(&rng);

        if (u_move < 0.16) {
            if (n_fk >= 2u) {
                var k1: u32 = 0u;
                var k2: u32 = 0u;
                if (rand_f32(&rng) < 0.55 && n_fk > 2u) {
                    let i1 = rand_below(&rng, n_fk - 1u);
                    let max_d = min(12u, n_fk - 1u - i1);
                    let d = 1u + rand_below(&rng, max_d);
                    k1 = u32(ro_ints[params.off_i0 + i1]);
                    k2 = u32(ro_ints[params.off_i0 + i1 + d]);
                } else {
                    let i1 = rand_below(&rng, n_fk);
                    var i2 = rand_below(&rng, n_fk - 1u);
                    if (i2 >= i1) { i2 = i2 + 1u; }
                    k1 = u32(ro_ints[params.off_i0 + i1]);
                    k2 = u32(ro_ints[params.off_i0 + i2]);
                }
                let s1 = wg_get_at(base, k1);
                let s2 = wg_get_at(base, k2);
                if (s1 != s2
                    && (s1 < 0 || get_eu(u32(s1), k2) < FORBID_HALF)
                    && (s2 < 0 || get_eu(u32(s2), k1) < FORBID_HALF)) {
                    let before = two_site_e_wg(base, k1, k2);
                    wg_set_at(base, k1, s2);
                    wg_set_at(base, k2, s1);
                    let after = two_site_e_wg(base, k1, k2);
                    let de = after - before;
                    if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                        if (s1 >= 0) { wg_set_pos(base, u32(s1), i32(k2)); }
                        if (s2 >= 0) { wg_set_pos(base, u32(s2), i32(k1)); }
                        cur_e = cur_e + de;
                        n_accepted = n_accepted + 1u;
                    } else {
                        wg_set_at(base, k1, s1);
                        wg_set_at(base, k2, s2);
                    }
                }
            }
        } else if (u_move < 0.72) {
            let k = u32(ro_ints[params.off_i0 + rand_below(&rng, n_fk)]);
            let sub = rand_f32(&rng);
            var s_i32: i32 = -1;
            if (sub >= 0.62) {
                let use_left = rand_f32(&rng) < 0.5;
                var anchor: i32 = -1;
                if (use_left) {
                    if (k > 0u) { anchor = wg_get_at(base, k - 1u); }
                } else {
                    if (k + 1u < k_len) { anchor = wg_get_at(base, k + 1u); }
                }
                if (anchor >= 0) {
                    let j = rand_below(&rng, 8u);
                    if (use_left) {
                        s_i32 = ro_ints[params.off_i6 + u32(anchor) * 8u + j];
                    } else {
                        s_i32 = ro_ints[params.off_i7 + u32(anchor) * 8u + j];
                    }
                }
            } else if (sub >= 0.22 && has_cands) {
                let j = rand_below(&rng, params.cand_m);
                s_i32 = ro_ints[params.off_i4 + k * params.cand_m + j];
            } else {
                s_i32 = ro_ints[params.off_i2 + rand_below(&rng, n_fs)];
            }

            if (s_i32 >= 0) {
                let s = u32(s_i32);
                let t = wg_get_at(base, k);
                let ks = wg_get_pos(base, s);

                if (t == s_i32) {
                    let before = site_e_wg(base, k);
                    wg_set_at(base, k, -1);
                    let de = -before;
                    if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                        wg_set_pos(base, s, -1);
                        cur_e = cur_e + de;
                        n_accepted = n_accepted + 1u;
                    } else {
                        wg_set_at(base, k, s_i32);
                    }
                } else if (t == -1 && ks == -1) {
                    if (get_eu(s, k) < FORBID_HALF) {
                        let before = site_e_wg(base, k);
                        wg_set_at(base, k, s_i32);
                        let after = site_e_wg(base, k);
                        let de = after - before;
                        if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                            wg_set_pos(base, s, i32(k));
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else {
                            wg_set_at(base, k, -1);
                        }
                    }
                } else if (t >= 0 && ks == -1) {
                    if (get_eu(s, k) < FORBID_HALF) {
                        let t_u = u32(t);
                        if (n_fk >= 2u && rand_f32(&rng) < P_RESTORE_EVICT) {
                            var m_i32: i32 = -1;
                            if (rand_f32(&rng) < 0.50) {
                                m_i32 = ro_ints[params.off_i5 + t_u * 8u + rand_below(&rng, 8u)];
                            } else {
                                m_i32 = ro_ints[params.off_i0 + rand_below(&rng, n_fk)];
                            }
                            if (m_i32 >= 0) {
                                let m = u32(m_i32);
                                if (m != k
                                    && ro_ints[params.off_i3 + m] != 0
                                    && wg_get_at(base, m) < 0
                                    && get_eu(t_u, m) < FORBID_HALF) {
                                    let w_fwd = pair_prop_w_wg(base, k, s_i32)
                                        * P_RESTORE_EVICT
                                        * evict_slot_weight(t, m, n_fk_f);
                                    let before = two_site_e_wg(base, k, m);
                                    wg_set_at(base, k, s_i32);
                                    wg_set_at(base, m, t);
                                    let after = two_site_e_wg(base, k, m);
                                    let p_ev_rev = select(1.0, P_EVICT_WHEN_SWAP_OK, get_eu(s, m) < FORBID_HALF);
                                    let w_rev = pair_prop_w_wg(base, k, t) * p_ev_rev;
                                    let de = after - before;
                                    let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                    if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                        wg_set_pos(base, s, i32(k));
                                        wg_set_pos(base, t_u, i32(m));
                                        cur_e = cur_e + de;
                                        n_accepted = n_accepted + 1u;
                                    } else {
                                        wg_set_at(base, k, t);
                                        wg_set_at(base, m, -1);
                                    }
                                }
                            }
                        } else {
                            let w_fwd = pair_prop_w_wg(base, k, s_i32);
                            let w_rev = pair_prop_w_wg(base, k, t);
                            let before = site_e_wg(base, k);
                            wg_set_at(base, k, s_i32);
                            let after = site_e_wg(base, k);
                            let de = after - before;
                            let log_acc = -de * inv_t + log(w_rev / w_fwd);
                            if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                wg_set_pos(base, t_u, -1);
                                wg_set_pos(base, s, i32(k));
                                cur_e = cur_e + de;
                                n_accepted = n_accepted + 1u;
                            } else {
                                wg_set_at(base, k, t);
                            }
                        }
                    }
                } else if (t == -1 && ks >= 0) {
                    let ks_u = u32(ks);
                    if (ro_ints[params.off_i3 + ks_u] != 0 && get_eu(s, k) < FORBID_HALF) {
                        let w_fwd = pair_prop_w_wg(base, k, s_i32);
                        let before = two_site_e_wg(base, k, ks_u);
                        wg_set_at(base, ks_u, -1);
                        wg_set_at(base, k, s_i32);
                        let after = two_site_e_wg(base, k, ks_u);
                        let w_rev = pair_prop_w_wg(base, ks_u, s_i32);
                        let de = after - before;
                        let log_acc = -de * inv_t + log(w_rev / w_fwd);
                        if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                            wg_set_pos(base, s, i32(k));
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else {
                            wg_set_at(base, ks_u, s_i32);
                            wg_set_at(base, k, -1);
                        }
                    }
                } else if (t >= 0 && ks >= 0) {
                    let ks_u = u32(ks);
                    let t_u = u32(t);
                    if (ks_u != k && ro_ints[params.off_i3 + ks_u] != 0 && get_eu(s, k) < FORBID_HALF) {
                        let swap_ok = get_eu(t_u, ks_u) < FORBID_HALF;
                        let do_evict = (!swap_ok) || (n_fk >= 2u && rand_f32(&rng) < P_EVICT_WHEN_SWAP_OK);
                        if (do_evict) {
                            if (n_fk >= 2u) {
                                let p_ev = select(1.0, P_EVICT_WHEN_SWAP_OK, swap_ok);
                                let w_fwd = pair_prop_w_wg(base, k, s_i32) * p_ev;
                                let before = two_site_e_wg(base, k, ks_u);
                                wg_set_at(base, ks_u, -1);
                                wg_set_at(base, k, s_i32);
                                let after = two_site_e_wg(base, k, ks_u);
                                let w_rev = pair_prop_w_wg(base, k, t)
                                    * P_RESTORE_EVICT
                                    * evict_slot_weight(s_i32, ks_u, n_fk_f);
                                let de = after - before;
                                let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                    wg_set_pos(base, s, i32(k));
                                    wg_set_pos(base, t_u, -1);
                                    cur_e = cur_e + de;
                                    n_accepted = n_accepted + 1u;
                                } else {
                                    wg_set_at(base, ks_u, s_i32);
                                    wg_set_at(base, k, t);
                                }
                            }
                        } else {
                            let w_fwd = pair_prop_w_wg(base, k, s_i32) + pair_prop_w_wg(base, ks_u, t);
                            let before = two_site_e_wg(base, k, ks_u);
                            wg_set_at(base, k, s_i32);
                            wg_set_at(base, ks_u, t);
                            let after = two_site_e_wg(base, k, ks_u);
                            let w_rev = pair_prop_w_wg(base, k, t) + pair_prop_w_wg(base, ks_u, s_i32);
                            let de = after - before;
                            let log_acc = -de * inv_t + log(w_rev / w_fwd);
                            if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                wg_set_pos(base, s, i32(k));
                                wg_set_pos(base, t_u, ks);
                                cur_e = cur_e + de;
                                n_accepted = n_accepted + 1u;
                            } else {
                                wg_set_at(base, k, t);
                                wg_set_at(base, ks_u, s_i32);
                            }
                        }
                    }
                }
            }
        } else {
            if (n_fk >= 4u) {
                if (rand_f32(&rng) < 0.34) {
                    let max_cw = min(max_w, n_fk);
                    var w: u32 = 2u;
                    if (rand_f32(&rng) < 0.55) {
                        w = 2u + rand_below(&rng, max(min(max_cw, 8u) - 1u, 1u));
                    } else {
                        w = 2u + rand_below(&rng, max(max_cw - 1u, 1u));
                    }
                    let ia = rand_below(&rng, n_fk + 1u - w);
                    let shift_right = rand_f32(&rng) < 0.5;
                    var valid_u = true;
                    var any_change = false;
                    for (var off: u32 = 0u; off < w; off = off + 1u) {
                        var src_off: u32 = 0u;
                        if (shift_right) {
                            src_off = select(off - 1u, w - 1u, off == 0u);
                        } else {
                            src_off = select(off + 1u, 0u, off + 1u == w);
                        }
                        let k_dst = u32(ro_ints[params.off_i0 + ia + off]);
                        let k_src = u32(ro_ints[params.off_i0 + ia + src_off]);
                        let s_prop = wg_get_at(base, k_src);
                        if (s_prop != wg_get_at(base, k_dst)) {
                            any_change = true;
                        }
                        if (s_prop >= 0 && get_eu(u32(s_prop), k_dst) >= FORBID_HALF) {
                            valid_u = false;
                            break;
                        }
                    }
                    if (valid_u && any_change) {
                        let k_lo = u32(ro_ints[params.off_i0 + ia]);
                        let k_hi = u32(ro_ints[params.off_i0 + ia + w - 1u]);
                        let span = k_hi - k_lo + 1u;
                        let before = interval_e_wg(base, k_lo, span);
                        if (shift_right) {
                            let last_s = wg_get_at(base, u32(ro_ints[params.off_i0 + ia + w - 1u]));
                            for (var rev_i: u32 = 1u; rev_i < w; rev_i = rev_i + 1u) {
                                let off = w - rev_i;
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off - 1u]);
                                wg_set_at(base, kd, wg_get_at(base, ks0));
                            }
                            wg_set_at(base, u32(ro_ints[params.off_i0 + ia]), last_s);
                        } else {
                            let first_s = wg_get_at(base, u32(ro_ints[params.off_i0 + ia]));
                            for (var off: u32 = 0u; off + 1u < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off + 1u]);
                                wg_set_at(base, kd, wg_get_at(base, ks0));
                            }
                            wg_set_at(base, u32(ro_ints[params.off_i0 + ia + w - 1u]), first_s);
                        }
                        let after = interval_e_wg(base, k_lo, span);
                        let de = after - before;
                        if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                            for (var off: u32 = 0u; off < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let sp = wg_get_at(base, kd);
                                if (sp >= 0) { wg_set_pos(base, u32(sp), i32(kd)); }
                            }
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else if (shift_right) {
                            let first_s = wg_get_at(base, u32(ro_ints[params.off_i0 + ia]));
                            for (var off: u32 = 0u; off + 1u < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off + 1u]);
                                wg_set_at(base, kd, wg_get_at(base, ks0));
                            }
                            wg_set_at(base, u32(ro_ints[params.off_i0 + ia + w - 1u]), first_s);
                        } else {
                            let last_s = wg_get_at(base, u32(ro_ints[params.off_i0 + ia + w - 1u]));
                            for (var rev_i: u32 = 1u; rev_i < w; rev_i = rev_i + 1u) {
                                let off = w - rev_i;
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off - 1u]);
                                wg_set_at(base, kd, wg_get_at(base, ks0));
                            }
                            wg_set_at(base, u32(ro_ints[params.off_i0 + ia]), last_s);
                        }
                    }
                } else {
                    let max_l = min(max_w, n_fk / 2u);
                    if (max_l >= 2u) {
                        var l: u32 = 2u;
                        if (rand_f32(&rng) < 0.55) {
                            l = 2u + rand_below(&rng, max(min(max_l, 5u) - 1u, 1u));
                        } else {
                            l = 2u + rand_below(&rng, max_l - 1u);
                        }
                        let m_start = n_fk + 1u - l;
                        let ia = rand_below(&rng, m_start);
                        let ka0 = u32(ro_ints[params.off_i0 + ia]);
                        let ka_end = u32(ro_ints[params.off_i0 + ia + l - 1u]);
                        let u_mode = rand_f32(&rng);

                        var ib_valid = false;
                        var ib: u32 = 0u;
                        if (u_mode < 0.40) {
                            ib = rand_below(&rng, m_start);
                            ib_valid = true;
                        } else if (u_mode < 0.70) {
                            let stitch_left = rand_f32(&rng) < 0.5;
                            let j = rand_below(&rng, 8u);
                            if (stitch_left) {
                                if (ka0 > 1u) {
                                    let anc = wg_get_at(base, ka0 - 1u);
                                    if (anc >= 0) {
                                        let sp = ro_ints[params.off_i6 + u32(anc) * 8u + j];
                                        if (sp >= 0) {
                                            let kb = wg_get_pos(base, u32(sp));
                                            if (kb >= 0) {
                                                let ib_c = ro_ints[params.off_i1 + u32(kb)];
                                                if (ib_c >= 0 && u32(ib_c) < m_start) {
                                                    ib = u32(ib_c);
                                                    ib_valid = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (ka_end + 1u < k_len) {
                                    let anc = wg_get_at(base, ka_end + 1u);
                                    if (anc >= 0) {
                                        let sp = ro_ints[params.off_i7 + u32(anc) * 8u + j];
                                        if (sp >= 0) {
                                            let kb_e = wg_get_pos(base, u32(sp));
                                            if (kb_e >= 0) {
                                                let ib_end = ro_ints[params.off_i1 + u32(kb_e)];
                                                if (ib_end >= i32(l) - 1) {
                                                    let ib_c = u32(ib_end) + 1u - l;
                                                    if (ib_c < m_start) {
                                                        ib = ib_c;
                                                        ib_valid = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            let use_head = rand_f32(&rng) < 0.5;
                            let j = rand_below(&rng, 8u);
                            if (use_head) {
                                let sh = wg_get_at(base, ka0);
                                if (sh >= 0) {
                                    let kh = ro_ints[params.off_i5 + u32(sh) * 8u + j];
                                    if (kh >= 0) {
                                        let ib_c = ro_ints[params.off_i1 + u32(kh)];
                                        if (ib_c >= 0 && u32(ib_c) < m_start) {
                                            ib = u32(ib_c);
                                            ib_valid = true;
                                        }
                                    }
                                }
                            } else {
                                let st = wg_get_at(base, ka_end);
                                if (st >= 0) {
                                    let kt = ro_ints[params.off_i5 + u32(st) * 8u + j];
                                    if (kt >= 0) {
                                        let ib_end = ro_ints[params.off_i1 + u32(kt)];
                                        if (ib_end >= i32(l) - 1) {
                                            let ib_c = u32(ib_end) + 1u - l;
                                            if (ib_c < m_start) {
                                                ib = ib_c;
                                                ib_valid = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let dist_ab = select(ib - ia, ia - ib, ia > ib);
                        if (ib_valid && dist_ab >= l) {
                            var ok = true;
                            var any_nonempty = false;
                            for (var off: u32 = 0u; off < l; off = off + 1u) {
                                let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                let sa = wg_get_at(base, ka);
                                let sb = wg_get_at(base, kb);
                                if (sa != sb) { any_nonempty = true; }
                                if ((sa >= 0 && get_eu(u32(sa), kb) >= FORBID_HALF)
                                    || (sb >= 0 && get_eu(u32(sb), ka) >= FORBID_HALF)) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok && any_nonempty) {
                                let m_start_f = f32(m_start);
                                let kb0 = u32(ro_ints[params.off_i0 + ib]);
                                let kb_end = u32(ro_ints[params.off_i0 + ib + l - 1u]);
                                let a_span = ka_end - ka0 + 1u;
                                let b_span = kb_end - kb0 + 1u;

                                let w_fwd = block_swap_prop_w_wg(base, ia, ib, l, m_start_f);
                                let before = two_spans_e_wg(base, ka0, a_span, kb0, b_span);
                                for (var off: u32 = 0u; off < l; off = off + 1u) {
                                    let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                    let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                    let sa_tmp = wg_get_at(base, ka);
                                    let sb_tmp = wg_get_at(base, kb);
                                    wg_set_at(base, ka, sb_tmp);
                                    wg_set_at(base, kb, sa_tmp);
                                }
                                let after = two_spans_e_wg(base, ka0, a_span, kb0, b_span);
                                let w_rev = block_swap_prop_w_wg(base, ia, ib, l, m_start_f);
                                let de = after - before;
                                let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                    for (var off: u32 = 0u; off < l; off = off + 1u) {
                                        let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                        let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                        let sa_n = wg_get_at(base, ka);
                                        let sb_n = wg_get_at(base, kb);
                                        if (sa_n >= 0) { wg_set_pos(base, u32(sa_n), i32(ka)); }
                                        if (sb_n >= 0) { wg_set_pos(base, u32(sb_n), i32(kb)); }
                                    }
                                    cur_e = cur_e + de;
                                    n_accepted = n_accepted + 1u;
                                } else {
                                    for (var off: u32 = 0u; off < l; off = off + 1u) {
                                        let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                        let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                        let sa_tmp = wg_get_at(base, ka);
                                        let sb_tmp = wg_get_at(base, kb);
                                        wg_set_at(base, ka, sb_tmp);
                                        wg_set_at(base, kb, sa_tmp);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (cur_e < best_e) {
            best_e = cur_e;
            for (var k: u32 = 0u; k < k_len; k = k + 1u) {
                at_buf[best_base + k] = wg_get_at(base, k);
            }
        }

        if (global_step >= params.burn_in) {
            n_samples = n_samples + 1u;
            let delta = cur_e - mean_e;
            mean_e = mean_e + delta / f32(n_samples);
            let delta2 = cur_e - mean_e;
            m2_e = m2_e + delta * delta2;
        }
    }

    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        at_buf[cur_base + k] = wg_get_at(base, k);
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        pos_buf[pos_base + s] = wg_get_pos(base, s);
    }
    stats_f_buf[r] = vec4<f32>(cur_e, best_e, mean_e, m2_e);
    state_u_buf[r] = vec4<u32>(n_samples, n_accepted, n_proposed, 0u);
    state_u_buf[params.replicas + r] = rng;
}

// --- Global Storage-Buffer Fallback (> 1024 residues / spin systems) ---

fn pair_prop_w_gl(cur_base: u32, k: u32, s: i32) -> f32 {
    var w = params.r_glob;
    if (params.r_cand > 0.0 && in_cand_spins(k, s)) {
        w = w + params.r_cand;
    }
    if (k > 0u && in_list8(params.off_i6, at_buf[cur_base + k - 1u], s)) {
        w = w + params.r_ext;
    }
    if (k + 1u < params.k_len && in_list8(params.off_i7, at_buf[cur_base + k + 1u], s)) {
        w = w + params.r_ext;
    }
    return w;
}

fn eval_full_gl(cur_base: u32) -> f32 {
    var sum: f32 = 0.0;
    var comp: f32 = 0.0;
    let k_len = params.k_len;
    for (var k: u32 = 0u; k < k_len; k = k + 1u) {
        let s = at_buf[cur_base + k];
        if (s >= 0) {
            let y = get_eu(u32(s), k) - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }
    for (var k: u32 = 1u; k < k_len; k = k + 1u) {
        let sa = at_buf[cur_base + k - 1u];
        let sb = at_buf[cur_base + k];
        if (sa >= 0 && sb >= 0) {
            let y = get_el(u32(sa), u32(sb)) - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }
    return sum;
}

fn site_e_gl(cur_base: u32, k: u32) -> f32 {
    let s = at_buf[cur_base + k];
    if (s < 0) { return 0.0; }
    let su = u32(s);
    var e = get_eu(su, k);
    if (k > 0u) {
        let sl = at_buf[cur_base + k - 1u];
        if (sl >= 0) { e = e + get_el(u32(sl), su); }
    }
    if (k + 1u < params.k_len) {
        let sr = at_buf[cur_base + k + 1u];
        if (sr >= 0) { e = e + get_el(su, u32(sr)); }
    }
    return e;
}

fn two_site_e_gl(cur_base: u32, ka: u32, kb: u32) -> f32 {
    let lo = min(ka, kb);
    let hi = max(ka, kb);
    let slo = at_buf[cur_base + lo];
    let shi = at_buf[cur_base + hi];
    var e: f32 = 0.0;
    if (slo >= 0) {
        let su = u32(slo);
        e = e + get_eu(su, lo);
        if (lo > 0u) {
            let sl = at_buf[cur_base + lo - 1u];
            if (sl >= 0) { e = e + get_el(u32(sl), su); }
        }
        let sr = at_buf[cur_base + lo + 1u];
        if (sr >= 0) { e = e + get_el(su, u32(sr)); }
    }
    if (shi >= 0) {
        let su = u32(shi);
        e = e + get_eu(su, hi);
        if (hi > lo + 1u) {
            let sl = at_buf[cur_base + hi - 1u];
            if (sl >= 0) { e = e + get_el(u32(sl), su); }
        }
        if (hi + 1u < params.k_len) {
            let sr = at_buf[cur_base + hi + 1u];
            if (sr >= 0) { e = e + get_el(su, u32(sr)); }
        }
    }
    return e;
}

fn block_swap_prop_w_gl(cur_base: u32, ia: u32, ib: u32, l: u32, m_start_f: f32) -> f32 {
    var w: f32 = 0.80 / m_start_f;
    let r_guided: f32 = 0.15 / 8.0;
    let ka0 = u32(ro_ints[params.off_i0 + ia]);
    let kb0 = u32(ro_ints[params.off_i0 + ib]);
    let ka_end = u32(ro_ints[params.off_i0 + ia + l - 1u]);
    let kb_end = u32(ro_ints[params.off_i0 + ib + l - 1u]);

    let sa0 = at_buf[cur_base + ka0];
    let sb0 = at_buf[cur_base + kb0];
    let sa_e = at_buf[cur_base + ka_end];
    let sb_e = at_buf[cur_base + kb_end];

    if (ka0 > 1u && in_list8(params.off_i6, at_buf[cur_base + ka0 - 1u], sb0)) { w = w + r_guided; }
    if (kb0 > 1u && in_list8(params.off_i6, at_buf[cur_base + kb0 - 1u], sa0)) { w = w + r_guided; }
    if (ka_end + 1u < params.k_len && in_list8(params.off_i7, at_buf[cur_base + ka_end + 1u], sb_e)) { w = w + r_guided; }
    if (kb_end + 1u < params.k_len && in_list8(params.off_i7, at_buf[cur_base + kb_end + 1u], sa_e)) { w = w + r_guided; }

    if (in_list8(params.off_i5, sa0, i32(kb0))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sb0, i32(ka0))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sa_e, i32(kb_end))) { w = w + r_guided; }
    if (in_list8(params.off_i5, sb_e, i32(ka_end))) { w = w + r_guided; }
    return w;
}

fn interval_e_gl(cur_base: u32, start: u32, len: u32) -> f32 {
    let end_k = start + len;
    var e: f32 = 0.0;
    for (var k: u32 = start; k < end_k; k = k + 1u) {
        let s = at_buf[cur_base + k];
        if (s >= 0) {
            e = e + get_eu(u32(s), k);
        }
    }
    let edge_lo = max(start, 1u);
    let edge_hi = min(end_k, params.k_len - 1u);
    for (var k: u32 = edge_lo; k <= edge_hi; k = k + 1u) {
        let sa = at_buf[cur_base + k - 1u];
        let sb = at_buf[cur_base + k];
        if (sa >= 0 && sb >= 0) {
            e = e + get_el(u32(sa), u32(sb));
        }
    }
    return e;
}

fn two_spans_e_gl(cur_base: u32, a_start: u32, a_len: u32, b_start: u32, b_len: u32) -> f32 {
    var lo_s = a_start;
    var lo_len = a_len;
    var hi_s = b_start;
    var hi_len = b_len;
    if (b_start < a_start) {
        lo_s = b_start;
        lo_len = b_len;
        hi_s = a_start;
        hi_len = a_len;
    }
    var e = interval_e_gl(cur_base, lo_s, lo_len) + interval_e_gl(cur_base, hi_s, hi_len);
    if (lo_s + lo_len == hi_s) {
        let sa = at_buf[cur_base + hi_s - 1u];
        let sb = at_buf[cur_base + hi_s];
        if (sa >= 0 && sb >= 0) {
            e = e - get_el(u32(sa), u32(sb));
        }
    }
    return e;
}

// ============================================================================
// Entry Points 5-7: Multi-Replica Stage MCMC Kernels
// ============================================================================

@compute @workgroup_size(8, 1, 1)
fn anneal_bayes_256(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let base = lid.x * 512u;
    run_anneal_wg_impl(r, base);
}

@compute @workgroup_size(4, 1, 1)
fn anneal_bayes_1024(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let base = lid.x * 1024u;
    run_anneal_wg_impl(r, base);
}

@compute @workgroup_size(32, 1, 1)
fn anneal_bayes_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let k_len = params.k_len;
    let s_len = params.s_len;
    let n_fk = params.n_fk;
    let n_fs = params.n_fs;
    let n_fk_f = f32(n_fk);
    let max_w = clamp(params.max_w, 2u, 32u);
    let cur_base = r * k_len;
    let best_base = (params.replicas + r) * k_len;
    let pos_base = r * s_len;

    var sf = stats_f_buf[r];
    var su = state_u_buf[r];
    var rng = state_u_buf[params.replicas + r];

    var cur_e = sf.x;
    var best_e = sf.y;
    var mean_e = sf.z;
    var m2_e = sf.w;
    var n_samples = su.x;
    var n_accepted = su.y;
    var n_proposed = su.z;

    if (params.step_offset == 0u) {
        cur_e = eval_full_gl(cur_base);
        mean_e = 0.0;
        m2_e = 0.0;
        n_samples = 0u;
        n_accepted = 0u;
        n_proposed = 0u;
        if (cur_e < best_e) {
            best_e = cur_e;
            for (var k: u32 = 0u; k < k_len; k = k + 1u) {
                at_buf[best_base + k] = at_buf[cur_base + k];
            }
        }
    }

    if (n_fk == 0u || n_fs == 0u || params.chunk_steps == 0u) {
        stats_f_buf[r] = vec4<f32>(cur_e, best_e, cur_e, 0.0);
        state_u_buf[r] = vec4<u32>(1u, 0u, 0u, 0u);
        return;
    }

    let inv_t = params.inv_temp;
    let has_cands = params.cand_m > 0u;

    for (var step_local: u32 = 0u; step_local < params.chunk_steps; step_local = step_local + 1u) {
        let global_step = params.step_offset + step_local;
        n_proposed = n_proposed + 1u;
        let u_move = rand_f32(&rng);

        if (u_move < 0.16) {
            if (n_fk >= 2u) {
                var i1: u32 = 0u;
                var i2: u32 = 1u;
                if (rand_f32(&rng) < 0.55 && n_fk > 2u) {
                    i1 = rand_below(&rng, n_fk - 1u);
                    let max_d = min(12u, n_fk - 1u - i1);
                    i2 = i1 + 1u + rand_below(&rng, max_d);
                } else {
                    i1 = rand_below(&rng, n_fk);
                    i2 = rand_below(&rng, n_fk - 1u);
                    if (i2 >= i1) { i2 = i2 + 1u; }
                }
                let k1 = u32(ro_ints[params.off_i0 + i1]);
                let k2 = u32(ro_ints[params.off_i0 + i2]);
                let s1 = at_buf[cur_base + k1];
                let s2 = at_buf[cur_base + k2];
                if (s1 != s2
                    && (s1 < 0 || get_eu(u32(s1), k2) < FORBID_HALF)
                    && (s2 < 0 || get_eu(u32(s2), k1) < FORBID_HALF)) {
                    let before = two_site_e_gl(cur_base, k1, k2);
                    at_buf[cur_base + k1] = s2;
                    at_buf[cur_base + k2] = s1;
                    let after = two_site_e_gl(cur_base, k1, k2);
                    let de = after - before;
                    if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                        if (s1 >= 0) { pos_buf[pos_base + u32(s1)] = i32(k2); }
                        if (s2 >= 0) { pos_buf[pos_base + u32(s2)] = i32(k1); }
                        cur_e = cur_e + de;
                        n_accepted = n_accepted + 1u;
                    } else {
                        at_buf[cur_base + k1] = s1;
                        at_buf[cur_base + k2] = s2;
                    }
                }
            }
        } else if (u_move < 0.72) {
            let k = u32(ro_ints[params.off_i0 + rand_below(&rng, n_fk)]);
            let sub = rand_f32(&rng);
            var s_i32: i32 = -1;
            if (sub >= 0.62) {
                let use_left = rand_f32(&rng) < 0.5;
                var anchor: i32 = -1;
                if (use_left && k > 0u) { anchor = at_buf[cur_base + k - 1u]; }
                if (!use_left && k + 1u < k_len) { anchor = at_buf[cur_base + k + 1u]; }
                if (anchor >= 0) {
                    let j = rand_below(&rng, 8u);
                    s_i32 = select(
                        ro_ints[params.off_i7 + u32(anchor) * 8u + j],
                        ro_ints[params.off_i6 + u32(anchor) * 8u + j],
                        use_left
                    );
                }
            } else if (sub >= 0.22 && has_cands) {
                s_i32 = ro_ints[params.off_i4 + k * params.cand_m + rand_below(&rng, params.cand_m)];
            } else {
                s_i32 = ro_ints[params.off_i2 + rand_below(&rng, n_fs)];
            }

            if (s_i32 >= 0) {
                let s = u32(s_i32);
                let t = at_buf[cur_base + k];
                let ks = pos_buf[pos_base + s];
                if (t == s_i32) {
                    let before = site_e_gl(cur_base, k);
                    at_buf[cur_base + k] = -1;
                    let de = -before;
                    if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                        pos_buf[pos_base + s] = -1;
                        cur_e = cur_e + de;
                        n_accepted = n_accepted + 1u;
                    } else {
                        at_buf[cur_base + k] = s_i32;
                    }
                } else if (t == -1 && ks == -1 && get_eu(s, k) < FORBID_HALF) {
                    let before = site_e_gl(cur_base, k);
                    at_buf[cur_base + k] = s_i32;
                    let after = site_e_gl(cur_base, k);
                    let de = after - before;
                    if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                        pos_buf[pos_base + s] = i32(k);
                        cur_e = cur_e + de;
                        n_accepted = n_accepted + 1u;
                    } else {
                        at_buf[cur_base + k] = -1;
                    }
                } else if (t >= 0 && ks == -1 && get_eu(s, k) < FORBID_HALF) {
                    let t_u = u32(t);
                    if (n_fk >= 2u && rand_f32(&rng) < P_RESTORE_EVICT) {
                        var m_i32: i32 = -1;
                        if (rand_f32(&rng) < 0.50) {
                            m_i32 = ro_ints[params.off_i5 + t_u * 8u + rand_below(&rng, 8u)];
                        } else {
                            m_i32 = ro_ints[params.off_i0 + rand_below(&rng, n_fk)];
                        }
                        if (m_i32 >= 0) {
                            let m = u32(m_i32);
                            if (m != k && ro_ints[params.off_i3 + m] != 0 && at_buf[cur_base + m] < 0 && get_eu(t_u, m) < FORBID_HALF) {
                                let w_fwd = pair_prop_w_gl(cur_base, k, s_i32) * P_RESTORE_EVICT * evict_slot_weight(t, m, n_fk_f);
                                let before = two_site_e_gl(cur_base, k, m);
                                at_buf[cur_base + k] = s_i32;
                                at_buf[cur_base + m] = t;
                                let after = two_site_e_gl(cur_base, k, m);
                                let p_ev_rev = select(1.0, P_EVICT_WHEN_SWAP_OK, get_eu(s, m) < FORBID_HALF);
                                let w_rev = pair_prop_w_gl(cur_base, k, t) * p_ev_rev;
                                let de = after - before;
                                let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                    pos_buf[pos_base + s] = i32(k);
                                    pos_buf[pos_base + t_u] = i32(m);
                                    cur_e = cur_e + de;
                                    n_accepted = n_accepted + 1u;
                                } else {
                                    at_buf[cur_base + k] = t;
                                    at_buf[cur_base + m] = -1;
                                }
                            }
                        }
                    } else {
                        let w_fwd = pair_prop_w_gl(cur_base, k, s_i32);
                        let w_rev = pair_prop_w_gl(cur_base, k, t);
                        let before = site_e_gl(cur_base, k);
                        at_buf[cur_base + k] = s_i32;
                        let after = site_e_gl(cur_base, k);
                        let de = after - before;
                        let log_acc = -de * inv_t + log(w_rev / w_fwd);
                        if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                            pos_buf[pos_base + t_u] = -1;
                            pos_buf[pos_base + s] = i32(k);
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else {
                            at_buf[cur_base + k] = t;
                        }
                    }
                } else if (t == -1 && ks >= 0) {
                    let ks_u = u32(ks);
                    if (ro_ints[params.off_i3 + ks_u] != 0 && get_eu(s, k) < FORBID_HALF) {
                        let w_fwd = pair_prop_w_gl(cur_base, k, s_i32);
                        let before = two_site_e_gl(cur_base, k, ks_u);
                        at_buf[cur_base + ks_u] = -1;
                        at_buf[cur_base + k] = s_i32;
                        let after = two_site_e_gl(cur_base, k, ks_u);
                        let w_rev = pair_prop_w_gl(cur_base, ks_u, s_i32);
                        let de = after - before;
                        let log_acc = -de * inv_t + log(w_rev / w_fwd);
                        if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                            pos_buf[pos_base + s] = i32(k);
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else {
                            at_buf[cur_base + ks_u] = s_i32;
                            at_buf[cur_base + k] = -1;
                        }
                    }
                } else if (t >= 0 && ks >= 0) {
                    let ks_u = u32(ks);
                    let t_u = u32(t);
                    if (ks_u != k && ro_ints[params.off_i3 + ks_u] != 0 && get_eu(s, k) < FORBID_HALF) {
                        let swap_ok = get_eu(t_u, ks_u) < FORBID_HALF;
                        let do_evict = (!swap_ok) || (n_fk >= 2u && rand_f32(&rng) < P_EVICT_WHEN_SWAP_OK);
                        if (do_evict) {
                            if (n_fk >= 2u) {
                                let p_ev = select(1.0, P_EVICT_WHEN_SWAP_OK, swap_ok);
                                let w_fwd = pair_prop_w_gl(cur_base, k, s_i32) * p_ev;
                                let before = two_site_e_gl(cur_base, k, ks_u);
                                at_buf[cur_base + ks_u] = -1;
                                at_buf[cur_base + k] = s_i32;
                                let after = two_site_e_gl(cur_base, k, ks_u);
                                let w_rev = pair_prop_w_gl(cur_base, k, t) * P_RESTORE_EVICT * evict_slot_weight(s_i32, ks_u, n_fk_f);
                                let de = after - before;
                                let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                    pos_buf[pos_base + s] = i32(k);
                                    pos_buf[pos_base + t_u] = -1;
                                    cur_e = cur_e + de;
                                    n_accepted = n_accepted + 1u;
                                } else {
                                    at_buf[cur_base + ks_u] = s_i32;
                                    at_buf[cur_base + k] = t;
                                }
                            }
                        } else {
                            let w_fwd = pair_prop_w_gl(cur_base, k, s_i32) + pair_prop_w_gl(cur_base, ks_u, t);
                            let before = two_site_e_gl(cur_base, k, ks_u);
                            at_buf[cur_base + k] = s_i32;
                            at_buf[cur_base + ks_u] = t;
                            let after = two_site_e_gl(cur_base, k, ks_u);
                            let w_rev = pair_prop_w_gl(cur_base, k, t) + pair_prop_w_gl(cur_base, ks_u, s_i32);
                            let de = after - before;
                            let log_acc = -de * inv_t + log(w_rev / w_fwd);
                            if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                pos_buf[pos_base + s] = i32(k);
                                pos_buf[pos_base + t_u] = ks;
                                cur_e = cur_e + de;
                                n_accepted = n_accepted + 1u;
                            } else {
                                at_buf[cur_base + k] = t;
                                at_buf[cur_base + ks_u] = s_i32;
                            }
                        }
                    }
                }
            }
        } else {
            if (n_fk >= 4u) {
                if (rand_f32(&rng) < 0.34) {
                    let max_cw = min(max_w, n_fk);
                    var w: u32 = 2u;
                    if (rand_f32(&rng) < 0.55) {
                        w = 2u + rand_below(&rng, max(min(max_cw, 8u) - 1u, 1u));
                    } else {
                        w = 2u + rand_below(&rng, max(max_cw - 1u, 1u));
                    }
                    let ia = rand_below(&rng, n_fk + 1u - w);
                    let shift_right = rand_f32(&rng) < 0.5;
                    var valid_u = true;
                    var any_change = false;
                    for (var off: u32 = 0u; off < w; off = off + 1u) {
                        var src_off: u32 = 0u;
                        if (shift_right) {
                            src_off = select(off - 1u, w - 1u, off == 0u);
                        } else {
                            src_off = select(off + 1u, 0u, off + 1u == w);
                        }
                        let k_dst = u32(ro_ints[params.off_i0 + ia + off]);
                        let k_src = u32(ro_ints[params.off_i0 + ia + src_off]);
                        let s_prop = at_buf[cur_base + k_src];
                        if (s_prop != at_buf[cur_base + k_dst]) {
                            any_change = true;
                        }
                        if (s_prop >= 0 && get_eu(u32(s_prop), k_dst) >= FORBID_HALF) {
                            valid_u = false;
                            break;
                        }
                    }
                    if (valid_u && any_change) {
                        let k_lo = u32(ro_ints[params.off_i0 + ia]);
                        let k_hi = u32(ro_ints[params.off_i0 + ia + w - 1u]);
                        let span = k_hi - k_lo + 1u;
                        let before = interval_e_gl(cur_base, k_lo, span);
                        if (shift_right) {
                            let last_s = at_buf[cur_base + u32(ro_ints[params.off_i0 + ia + w - 1u])];
                            for (var rev_i: u32 = 1u; rev_i < w; rev_i = rev_i + 1u) {
                                let off = w - rev_i;
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off - 1u]);
                                at_buf[cur_base + kd] = at_buf[cur_base + ks0];
                            }
                            at_buf[cur_base + u32(ro_ints[params.off_i0 + ia])] = last_s;
                        } else {
                            let first_s = at_buf[cur_base + u32(ro_ints[params.off_i0 + ia])];
                            for (var off: u32 = 0u; off + 1u < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off + 1u]);
                                at_buf[cur_base + kd] = at_buf[cur_base + ks0];
                            }
                            at_buf[cur_base + u32(ro_ints[params.off_i0 + ia + w - 1u])] = first_s;
                        }
                        let after = interval_e_gl(cur_base, k_lo, span);
                        let de = after - before;
                        if (de <= 0.0 || rand_f32(&rng) < exp(-de * inv_t)) {
                            for (var off: u32 = 0u; off < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let sp = at_buf[cur_base + kd];
                                if (sp >= 0) { pos_buf[pos_base + u32(sp)] = i32(kd); }
                            }
                            cur_e = cur_e + de;
                            n_accepted = n_accepted + 1u;
                        } else if (shift_right) {
                            let first_s = at_buf[cur_base + u32(ro_ints[params.off_i0 + ia])];
                            for (var off: u32 = 0u; off + 1u < w; off = off + 1u) {
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off + 1u]);
                                at_buf[cur_base + kd] = at_buf[cur_base + ks0];
                            }
                            at_buf[cur_base + u32(ro_ints[params.off_i0 + ia + w - 1u])] = first_s;
                        } else {
                            let last_s = at_buf[cur_base + u32(ro_ints[params.off_i0 + ia + w - 1u])];
                            for (var rev_i: u32 = 1u; rev_i < w; rev_i = rev_i + 1u) {
                                let off = w - rev_i;
                                let kd = u32(ro_ints[params.off_i0 + ia + off]);
                                let ks0 = u32(ro_ints[params.off_i0 + ia + off - 1u]);
                                at_buf[cur_base + kd] = at_buf[cur_base + ks0];
                            }
                            at_buf[cur_base + u32(ro_ints[params.off_i0 + ia])] = last_s;
                        }
                    }
                } else {
                    let max_l = min(max_w, n_fk / 2u);
                    if (max_l >= 2u) {
                        var l: u32 = 2u;
                        if (rand_f32(&rng) < 0.55) {
                            l = 2u + rand_below(&rng, max(min(max_l, 5u) - 1u, 1u));
                        } else {
                            l = 2u + rand_below(&rng, max_l - 1u);
                        }
                        let m_start = n_fk + 1u - l;
                        let ia = rand_below(&rng, m_start);
                        let ka0 = u32(ro_ints[params.off_i0 + ia]);
                        let ka_end = u32(ro_ints[params.off_i0 + ia + l - 1u]);
                        let u_mode = rand_f32(&rng);

                        var ib_valid = false;
                        var ib: u32 = 0u;
                        if (u_mode < 0.40) {
                            ib = rand_below(&rng, m_start);
                            ib_valid = true;
                        } else if (u_mode < 0.70) {
                            let stitch_left = rand_f32(&rng) < 0.5;
                            let j = rand_below(&rng, 8u);
                            if (stitch_left) {
                                if (ka0 > 1u) {
                                    let anc = at_buf[cur_base + ka0 - 1u];
                                    if (anc >= 0) {
                                        let sp = ro_ints[params.off_i6 + u32(anc) * 8u + j];
                                        if (sp >= 0) {
                                            let kb = pos_buf[pos_base + u32(sp)];
                                            if (kb >= 0) {
                                                let ib_c = ro_ints[params.off_i1 + u32(kb)];
                                                if (ib_c >= 0 && u32(ib_c) < m_start) {
                                                    ib = u32(ib_c);
                                                    ib_valid = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (ka_end + 1u < k_len) {
                                    let anc = at_buf[cur_base + ka_end + 1u];
                                    if (anc >= 0) {
                                        let sp = ro_ints[params.off_i7 + u32(anc) * 8u + j];
                                        if (sp >= 0) {
                                            let kb_e = pos_buf[pos_base + u32(sp)];
                                            if (kb_e >= 0) {
                                                let ib_end = ro_ints[params.off_i1 + u32(kb_e)];
                                                if (ib_end >= i32(l) - 1) {
                                                    let ib_c = u32(ib_end) + 1u - l;
                                                    if (ib_c < m_start) {
                                                        ib = ib_c;
                                                        ib_valid = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            let use_head = rand_f32(&rng) < 0.5;
                            let j = rand_below(&rng, 8u);
                            if (use_head) {
                                let sh = at_buf[cur_base + ka0];
                                if (sh >= 0) {
                                    let kh = ro_ints[params.off_i5 + u32(sh) * 8u + j];
                                    if (kh >= 0) {
                                        let ib_c = ro_ints[params.off_i1 + u32(kh)];
                                        if (ib_c >= 0 && u32(ib_c) < m_start) {
                                            ib = u32(ib_c);
                                            ib_valid = true;
                                        }
                                    }
                                }
                            } else {
                                let st = at_buf[cur_base + ka_end];
                                if (st >= 0) {
                                    let kt = ro_ints[params.off_i5 + u32(st) * 8u + j];
                                    if (kt >= 0) {
                                        let ib_end = ro_ints[params.off_i1 + u32(kt)];
                                        if (ib_end >= i32(l) - 1) {
                                            let ib_c = u32(ib_end) + 1u - l;
                                            if (ib_c < m_start) {
                                                ib = ib_c;
                                                ib_valid = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let dist_ab = select(ib - ia, ia - ib, ia > ib);
                        if (ib_valid && dist_ab >= l) {
                            var ok = true;
                            var any_nonempty = false;
                            for (var off: u32 = 0u; off < l; off = off + 1u) {
                                let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                let sa = at_buf[cur_base + ka];
                                let sb = at_buf[cur_base + kb];
                                if (sa != sb) { any_nonempty = true; }
                                if ((sa >= 0 && get_eu(u32(sa), kb) >= FORBID_HALF)
                                    || (sb >= 0 && get_eu(u32(sb), ka) >= FORBID_HALF)) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok && any_nonempty) {
                                let m_start_f = f32(m_start);
                                let kb0 = u32(ro_ints[params.off_i0 + ib]);
                                let kb_end = u32(ro_ints[params.off_i0 + ib + l - 1u]);
                                let a_span = ka_end - ka0 + 1u;
                                let b_span = kb_end - kb0 + 1u;

                                let w_fwd = block_swap_prop_w_gl(cur_base, ia, ib, l, m_start_f);
                                let before = two_spans_e_gl(cur_base, ka0, a_span, kb0, b_span);
                                for (var off: u32 = 0u; off < l; off = off + 1u) {
                                    let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                    let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                    let tmp = at_buf[cur_base + ka];
                                    at_buf[cur_base + ka] = at_buf[cur_base + kb];
                                    at_buf[cur_base + kb] = tmp;
                                }
                                let after = two_spans_e_gl(cur_base, ka0, a_span, kb0, b_span);
                                let w_rev = block_swap_prop_w_gl(cur_base, ia, ib, l, m_start_f);
                                let de = after - before;
                                let log_acc = -de * inv_t + log(w_rev / w_fwd);
                                if (log_acc >= 0.0 || rand_f32(&rng) < exp(log_acc)) {
                                    for (var off: u32 = 0u; off < l; off = off + 1u) {
                                        let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                        let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                        let sa_n = at_buf[cur_base + ka];
                                        let sb_n = at_buf[cur_base + kb];
                                        if (sa_n >= 0) { pos_buf[pos_base + u32(sa_n)] = i32(ka); }
                                        if (sb_n >= 0) { pos_buf[pos_base + u32(sb_n)] = i32(kb); }
                                    }
                                    cur_e = cur_e + de;
                                    n_accepted = n_accepted + 1u;
                                } else {
                                    for (var off: u32 = 0u; off < l; off = off + 1u) {
                                        let ka = u32(ro_ints[params.off_i0 + ia + off]);
                                        let kb = u32(ro_ints[params.off_i0 + ib + off]);
                                        let tmp = at_buf[cur_base + ka];
                                        at_buf[cur_base + ka] = at_buf[cur_base + kb];
                                        at_buf[cur_base + kb] = tmp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (cur_e < best_e) {
            best_e = cur_e;
            for (var k: u32 = 0u; k < k_len; k = k + 1u) {
                at_buf[best_base + k] = at_buf[cur_base + k];
            }
        }

        if (global_step >= params.burn_in) {
            n_samples = n_samples + 1u;
            let delta = cur_e - mean_e;
            mean_e = mean_e + delta / f32(n_samples);
            let delta2 = cur_e - mean_e;
            m2_e = m2_e + delta * delta2;
        }
    }

    stats_f_buf[r] = vec4<f32>(cur_e, best_e, mean_e, m2_e);
    state_u_buf[r] = vec4<u32>(n_samples, n_accepted, n_proposed, 0u);
    state_u_buf[params.replicas + r] = rng;
}
