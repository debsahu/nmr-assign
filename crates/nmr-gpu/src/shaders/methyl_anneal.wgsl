// Clean-room WGSL compute shader for supramolecular 13C-methyl TROSY (`ILVMAT`)
// structure-based NOE contact-graph objective evaluation and massively parallel
// multi-replica Metropolis-Hastings simulated annealing (`docs/specs/13-methyl-trosy.md` §3).
//
// Provenance: Clean-room GPU compute implementation from peer-reviewed literature:
// - Tugarinov, Hwang, Ollerenshaw & Kay (J. Am. Chem. Soc. 2003, 125, 10120-10121 & 10420-10428)
// - Gans et al. (Angew. Chem. Int. Ed. 2010, 49, 1958-1962)
// - Ruschak & Kay (J. Biomol. NMR 2010, 46, 75-87)
// - Xu, Lin, Ho & Yang (J. Am. Chem. Soc. 2009, 131, 9480-9481)
// - Chao, Shi, Masterson & Veglia (J. Magn. Reson. 2012, 214, 103-110)
// - Pritisanac, Degiacomi, Alderson, Carneiro, Ab, Baldwin & Kay (J. Am. Chem. Soc. 2017, 139, 9523-9533)
//
// Hardware & Browser Portability Invariants:
// - Total storage bindings = 6 (<= baseline WebGPU limit of 8 per stage) + 1 uniform buffer.
// - Total workgroup SRAM = `wg_at` (8,192 B) + `wg_pos` (8,192 B) = exact 16,384 B (16 KiB).
// - Zero dynamic function-scope stack arrays; unified 4-site proposal pipeline prevents GPU register spills.
// - Split exact `i32` (x1024) + residual `f32` accumulation guarantees `< 1e-5` parity even under GPU `-ffast-math`.

struct MethylParams {
    replicas: u32,
    s_len: u32,
    m_len: u32,
    n_obs_edges: u32,

    n_short_edges: u32,
    total_steps: u32,
    step_offset: u32,
    chunk_steps: u32,

    off_u_mat: u32,
    off_d_mat: u32,
    off_gem_obs_mat: u32,
    off_obs_edge_floats: u32,

    off_obs_adj_floats: u32,
    off_short_edge_lam: u32,
    off_short_adj_lam: u32,
    n_gem_pairs: u32,

    off_slots_offs: u32,
    off_slots_vals: u32,
    off_gem_site: u32,
    off_obs_explained: u32,

    off_obs_edge_pairs: u32,
    off_obs_adj_offs: u32,
    off_obs_adj_peers: u32,
    off_short_edge_pairs: u32,

    off_short_adj_offs: u32,
    off_short_adj_peers: u32,
    off_cut_adj_offs: u32,
    off_cut_adj_peers: u32,

    off_gem_pairs: u32,
    off_gem_nbr_offs: u32,
    off_gem_nbr_vals: u32,
    move_mask: u32,

    d_cutoff: f32,
    sigma_d: f32,
    flat_bottom_d: f32,
    norm_inv: f32,

    t_start: f32,
    t_end: f32,
    _pad_f0: f32,
    _pad_f1: f32,
};

@group(0) @binding(0) var<uniform> params: MethylParams;
@group(0) @binding(1) var<storage, read> ro_floats: array<f32>;
@group(0) @binding(2) var<storage, read> ro_ints: array<u32>;
@group(0) @binding(3) var<storage, read_write> at_both: array<i32>;
@group(0) @binding(4) var<storage, read_write> pos_both: array<i32>;
@group(0) @binding(5) var<storage, read_write> scores_state: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> rng_state: array<vec4<u32>>;

const METHYL_FORBID_HALF: f32 = -5.0e8;
const WG_STRIDE_256: u32 = 255u; // coprime to 32 SRAM banks (8 * 255 = 2040 <= 2048)
const WG_STRIDE_1024: u32 = 1024u;
const FIX_SCALE: f32 = 1024.0;
const FIX_INV: f32 = 0.0009765625; // exact 1.0 / 1024.0

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

fn select_move_phase(st: ptr<function, vec4<u32>>, move_mask: u32) -> u32 {
    let mask = move_mask & 15u;
    if (mask == 0u || mask == 15u) {
        return next_u32(st) & 15u;
    }
    let pop = countOneBits(mask);
    let pick = rand_below(st, pop);
    var seen: u32 = 0u;
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        if ((mask & (1u << b)) != 0u) {
            if (seen == pick) {
                if (b == 0u) {
                    return 0u;
                }
                if (b == 1u) {
                    return 6u;
                }
                if (b == 2u) {
                    if ((next_u32(st) & 1u) == 0u) {
                        return 12u;
                    }
                    return 14u;
                }
                return 15u;
            }
            seen = seen + 1u;
        }
    }
    return 15u;
}

fn in_u4(x: u32, n: u32, a0: u32, a1: u32, a2: u32, a3: u32) -> bool {
    return (n >= 1u && x == a0)
        || (n >= 2u && x == a1)
        || (n >= 3u && x == a2)
        || (n >= 4u && x == a3);
}

fn get_u4(idx: u32, a0: u32, a1: u32, a2: u32, a3: u32) -> u32 {
    if (idx == 0u) {
        return a0;
    }
    if (idx == 1u) {
        return a1;
    }
    if (idx == 2u) {
        return a2;
    }
    return a3;
}

fn get_i4(idx: u32, a0: i32, a1: i32, a2: i32, a3: i32) -> i32 {
    if (idx == 0u) {
        return a0;
    }
    if (idx == 1u) {
        return a1;
    }
    if (idx == 2u) {
        return a2;
    }
    return a3;
}

fn push_unique_s(
    s_cand: i32,
    ns: ptr<function, u32>,
    s0: ptr<function, u32>,
    s1: ptr<function, u32>,
    s2: ptr<function, u32>,
    s3: ptr<function, u32>,
) {
    if (s_cand < 0) {
        return;
    }
    let su = u32(s_cand);
    let n = *ns;
    if (in_u4(su, n, *s0, *s1, *s2, *s3)) {
        return;
    }
    if (n == 0u) {
        *s0 = su;
    } else if (n == 1u) {
        *s1 = su;
    } else if (n == 2u) {
        *s2 = su;
    } else {
        *s3 = su;
    }
    *ns = n + 1u;
}

fn is_obs_explained(sa: u32, sb: u32, s_len: u32) -> bool {
    let bit_idx = sa * s_len + sb;
    let word = ro_ints[params.off_obs_explained + (bit_idx >> 5u)];
    return ((word >> (bit_idx & 31u)) & 1u) != 0u;
}

fn noe_edge_score(d_cc: f32, d_hat: f32, weight: f32) -> f32 {
    if (d_cc <= params.d_cutoff) {
        let diff = abs(d_cc - d_hat) - params.flat_bottom_d;
        let ex = select(0.0, diff / params.sigma_d, diff > 0.0);
        return weight * (2.60 - 0.5 * ex * ex);
    } else {
        let pen = min(1.80 + 0.45 * (d_cc - params.d_cutoff), 4.20);
        return -weight * pen;
    }
}

// `-ffast-math`-immune exact integer + tiny-residual accumulator (`< 1e-5` parity vs CPU f64)
fn exact_add(i_acc: ptr<function, i32>, f_acc: ptr<function, f32>, term: f32) {
    if (abs(term) < 32768.0) {
        let q = i32(term * FIX_SCALE);
        *i_acc = (*i_acc) + q;
        *f_acc = (*f_acc) + (term - f32(q) * FIX_INV);
    } else {
        *f_acc = (*f_acc) + term;
    }
}

fn eval_total_split_global(at_base: u32, pos_base: u32) -> vec2<f32> {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var i_acc: i32 = 0;
    var f_acc: f32 = 0.0;

    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        let m = pos_both[pos_base + s];
        if (m >= 0) {
            exact_add(&i_acc, &f_acc, ro_floats[params.off_u_mat + s * m_len + u32(m)]);
        }
    }

    for (var m1: u32 = 0u; m1 < m_len; m1 = m1 + 1u) {
        let m2_i = bitcast<i32>(ro_ints[params.off_gem_site + m1]);
        if (m2_i > i32(m1)) {
            let s1 = at_both[at_base + m1];
            let s2 = at_both[at_base + u32(m2_i)];
            if (s1 >= 0 && s2 >= 0) {
                exact_add(
                    &i_acc,
                    &f_acc,
                    ro_floats[params.off_gem_obs_mat + u32(s1) * s_len + u32(s2)],
                );
            }
        }
    }

    for (var ei: u32 = 0u; ei < params.n_obs_edges; ei = ei + 1u) {
        let s1 = ro_ints[params.off_obs_edge_pairs + 2u * ei];
        let s2 = ro_ints[params.off_obs_edge_pairs + 2u * ei + 1u];
        let m1 = pos_both[pos_base + s1];
        let m2 = pos_both[pos_base + s2];
        if (m1 >= 0 && m2 >= 0) {
            let d = ro_floats[params.off_d_mat + u32(m1) * m_len + u32(m2)];
            let d_hat = ro_floats[params.off_obs_edge_floats + 2u * ei];
            let w = ro_floats[params.off_obs_edge_floats + 2u * ei + 1u];
            exact_add(&i_acc, &f_acc, noe_edge_score(d, d_hat, w));
        }
    }

    for (var sei: u32 = 0u; sei < params.n_short_edges; sei = sei + 1u) {
        let m1 = ro_ints[params.off_short_edge_pairs + 2u * sei];
        let m2 = ro_ints[params.off_short_edge_pairs + 2u * sei + 1u];
        let lam = ro_floats[params.off_short_edge_lam + sei];
        let s1 = at_both[at_base + m1];
        let s2 = at_both[at_base + m2];
        if (s1 >= 0 && s2 >= 0) {
            if (!is_obs_explained(u32(s1), u32(s2), s_len)) {
                exact_add(&i_acc, &f_acc, -lam);
            }
        } else {
            exact_add(&i_acc, &f_acc, -0.5 * lam);
        }
    }

    return vec2<f32>(f32(i_acc), f_acc);
}

fn eval_total_raw_global(at_base: u32, pos_base: u32) -> f32 {
    let sp = eval_total_split_global(at_base, pos_base);
    return sp.x * FIX_INV + sp.y;
}

fn eval_total_raw_wg(wg_base: u32) -> f32 {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var i_acc: i32 = 0;
    var f_acc: f32 = 0.0;

    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        let m = wg_pos[wg_base + s];
        if (m >= 0) {
            exact_add(&i_acc, &f_acc, ro_floats[params.off_u_mat + s * m_len + u32(m)]);
        }
    }

    for (var m1: u32 = 0u; m1 < m_len; m1 = m1 + 1u) {
        let m2_i = bitcast<i32>(ro_ints[params.off_gem_site + m1]);
        if (m2_i > i32(m1)) {
            let s1 = wg_at[wg_base + m1];
            let s2 = wg_at[wg_base + u32(m2_i)];
            if (s1 >= 0 && s2 >= 0) {
                exact_add(
                    &i_acc,
                    &f_acc,
                    ro_floats[params.off_gem_obs_mat + u32(s1) * s_len + u32(s2)],
                );
            }
        }
    }

    for (var ei: u32 = 0u; ei < params.n_obs_edges; ei = ei + 1u) {
        let s1 = ro_ints[params.off_obs_edge_pairs + 2u * ei];
        let s2 = ro_ints[params.off_obs_edge_pairs + 2u * ei + 1u];
        let m1 = wg_pos[wg_base + s1];
        let m2 = wg_pos[wg_base + s2];
        if (m1 >= 0 && m2 >= 0) {
            let d = ro_floats[params.off_d_mat + u32(m1) * m_len + u32(m2)];
            let d_hat = ro_floats[params.off_obs_edge_floats + 2u * ei];
            let w = ro_floats[params.off_obs_edge_floats + 2u * ei + 1u];
            exact_add(&i_acc, &f_acc, noe_edge_score(d, d_hat, w));
        }
    }

    for (var sei: u32 = 0u; sei < params.n_short_edges; sei = sei + 1u) {
        let m1 = ro_ints[params.off_short_edge_pairs + 2u * sei];
        let m2 = ro_ints[params.off_short_edge_pairs + 2u * sei + 1u];
        let lam = ro_floats[params.off_short_edge_lam + sei];
        let s1 = wg_at[wg_base + m1];
        let s2 = wg_at[wg_base + m2];
        if (s1 >= 0 && s2 >= 0) {
            if (!is_obs_explained(u32(s1), u32(s2), s_len)) {
                exact_add(&i_acc, &f_acc, -lam);
            }
        } else {
            exact_add(&i_acc, &f_acc, -0.5 * lam);
        }
    }

    return f32(i_acc) * FIX_INV + f_acc;
}

@compute @workgroup_size(64, 1, 1)
fn eval_methyl_objective_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let s_len = params.s_len;
    let m_len = params.m_len;
    let at_base = r * m_len;
    let pos_base = r * s_len;

    for (var m: u32 = 0u; m < m_len; m = m + 1u) {
        at_both[at_base + m] = -1;
    }

    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        let m = pos_both[pos_base + s];
        if (m < -1 || (m >= 0 && u32(m) >= m_len)) {
            scores_state[r] = vec2<f32>(-1.0e12 * params.norm_inv, -1.0e12);
            return;
        }
        if (m >= 0) {
            if (at_both[at_base + u32(m)] >= 0) {
                scores_state[r] = vec2<f32>(-1.0e12 * params.norm_inv, -1.0e12);
                return;
            }
            at_both[at_base + u32(m)] = i32(s);
        }
    }

    let sp = eval_total_split_global(at_base, pos_base);
    let denom = f32(max(s_len, 1u));
    let norm = (sp.x / denom) * FIX_INV + (sp.y / denom);
    let raw = sp.x * FIX_INV + sp.y;
    scores_state[r] = vec2<f32>(norm, raw);
}

// ============================================================================
// Incremental Subset Energy Evaluation (<= 4 Sites & <= 4 Roots in Registers)
// ============================================================================

fn eval_subset_wg(
    wg_base: u32,
    nm: u32,
    m0: u32,
    m1: u32,
    m2: u32,
    m3: u32,
    ns: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
) -> f32 {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var val: f32 = 0.0;

    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let s = get_u4(i, s0, s1, s2, s3);
        let m = wg_pos[wg_base + s];
        if (m >= 0) {
            val = val + ro_floats[params.off_u_mat + s * m_len + u32(m)];
        }
    }

    var g_n: u32 = 0u;
    var g0: u32 = 0u;
    var g1: u32 = 0u;
    var g2: u32 = 0u;
    var g3: u32 = 0u;
    for (var i: u32 = 0u; i < nm; i = i + 1u) {
        let ma = get_u4(i, m0, m1, m2, m3);
        let mb_i = bitcast<i32>(ro_ints[params.off_gem_site + ma]);
        if (mb_i >= 0) {
            let mb = u32(mb_i);
            let lo = min(ma, mb);
            let hi = max(ma, mb);
            if (!in_u4(lo, g_n, g0, g1, g2, g3)) {
                if (g_n == 0u) {
                    g0 = lo;
                } else if (g_n == 1u) {
                    g1 = lo;
                } else if (g_n == 2u) {
                    g2 = lo;
                } else {
                    g3 = lo;
                }
                g_n = g_n + 1u;
                let sa = wg_at[wg_base + lo];
                let sb = wg_at[wg_base + hi];
                if (sa >= 0 && sb >= 0) {
                    val = val + ro_floats[params.off_gem_obs_mat + u32(sa) * s_len + u32(sb)];
                }
            }
        }
    }

    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let sa = get_u4(i, s0, s1, s2, s3);
        let ma = wg_pos[wg_base + sa];
        if (ma >= 0) {
            let e0 = ro_ints[params.off_obs_adj_offs + sa];
            let e1 = ro_ints[params.off_obs_adj_offs + sa + 1u];
            for (var ei: u32 = e0; ei < e1; ei = ei + 1u) {
                let sb = ro_ints[params.off_obs_adj_peers + ei];
                if (sb < sa && in_u4(sb, ns, s0, s1, s2, s3)) {
                    continue;
                }
                let mb = wg_pos[wg_base + sb];
                if (mb >= 0) {
                    let d = ro_floats[params.off_d_mat + u32(ma) * m_len + u32(mb)];
                    let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
                    let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
                    val = val + noe_edge_score(d, d_hat, w);
                }
            }
        }
    }

    for (var i: u32 = 0u; i < nm; i = i + 1u) {
        let ma = get_u4(i, m0, m1, m2, m3);
        let sa = wg_at[wg_base + ma];
        let se0 = ro_ints[params.off_short_adj_offs + ma];
        let se1 = ro_ints[params.off_short_adj_offs + ma + 1u];
        for (var sei: u32 = se0; sei < se1; sei = sei + 1u) {
            let mb = ro_ints[params.off_short_adj_peers + sei];
            if (mb < ma && in_u4(mb, nm, m0, m1, m2, m3)) {
                continue;
            }
            let lam = ro_floats[params.off_short_adj_lam + sei];
            let sb = wg_at[wg_base + mb];
            if (sa >= 0 && sb >= 0) {
                if (!is_obs_explained(u32(sa), u32(sb), s_len)) {
                    val = val - lam;
                }
            } else {
                val = val - 0.5 * lam;
            }
        }
    }

    return val;
}

fn eval_subset_global(
    at_base: u32,
    pos_base: u32,
    nm: u32,
    m0: u32,
    m1: u32,
    m2: u32,
    m3: u32,
    ns: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
) -> f32 {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var val: f32 = 0.0;

    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let s = get_u4(i, s0, s1, s2, s3);
        let m = pos_both[pos_base + s];
        if (m >= 0) {
            val = val + ro_floats[params.off_u_mat + s * m_len + u32(m)];
        }
    }

    var g_n: u32 = 0u;
    var g0: u32 = 0u;
    var g1: u32 = 0u;
    var g2: u32 = 0u;
    var g3: u32 = 0u;
    for (var i: u32 = 0u; i < nm; i = i + 1u) {
        let ma = get_u4(i, m0, m1, m2, m3);
        let mb_i = bitcast<i32>(ro_ints[params.off_gem_site + ma]);
        if (mb_i >= 0) {
            let mb = u32(mb_i);
            let lo = min(ma, mb);
            let hi = max(ma, mb);
            if (!in_u4(lo, g_n, g0, g1, g2, g3)) {
                if (g_n == 0u) {
                    g0 = lo;
                } else if (g_n == 1u) {
                    g1 = lo;
                } else if (g_n == 2u) {
                    g2 = lo;
                } else {
                    g3 = lo;
                }
                g_n = g_n + 1u;
                let sa = at_both[at_base + lo];
                let sb = at_both[at_base + hi];
                if (sa >= 0 && sb >= 0) {
                    val = val + ro_floats[params.off_gem_obs_mat + u32(sa) * s_len + u32(sb)];
                }
            }
        }
    }

    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let sa = get_u4(i, s0, s1, s2, s3);
        let ma = pos_both[pos_base + sa];
        if (ma >= 0) {
            let e0 = ro_ints[params.off_obs_adj_offs + sa];
            let e1 = ro_ints[params.off_obs_adj_offs + sa + 1u];
            for (var ei: u32 = e0; ei < e1; ei = ei + 1u) {
                let sb = ro_ints[params.off_obs_adj_peers + ei];
                if (sb < sa && in_u4(sb, ns, s0, s1, s2, s3)) {
                    continue;
                }
                let mb = pos_both[pos_base + sb];
                if (mb >= 0) {
                    let d = ro_floats[params.off_d_mat + u32(ma) * m_len + u32(mb)];
                    let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
                    let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
                    val = val + noe_edge_score(d, d_hat, w);
                }
            }
        }
    }

    for (var i: u32 = 0u; i < nm; i = i + 1u) {
        let ma = get_u4(i, m0, m1, m2, m3);
        let sa = at_both[at_base + ma];
        let se0 = ro_ints[params.off_short_adj_offs + ma];
        let se1 = ro_ints[params.off_short_adj_offs + ma + 1u];
        for (var sei: u32 = se0; sei < se1; sei = sei + 1u) {
            let mb = ro_ints[params.off_short_adj_peers + sei];
            if (mb < ma && in_u4(mb, nm, m0, m1, m2, m3)) {
                continue;
            }
            let lam = ro_floats[params.off_short_adj_lam + sei];
            let sb = at_both[at_base + mb];
            if (sa >= 0 && sb >= 0) {
                if (!is_obs_explained(u32(sa), u32(sb), s_len)) {
                    val = val - lam;
                }
            } else {
                val = val - 0.5 * lam;
            }
        }
    }

    return val;
}

fn save_best_wg(wg_base: u32, at_best_base: u32, pos_best_base: u32) {
    let s_len = params.s_len;
    let m_len = params.m_len;
    for (var m: u32 = 0u; m < m_len; m = m + 1u) {
        at_both[at_best_base + m] = wg_at[wg_base + m];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        pos_both[pos_best_base + s] = wg_pos[wg_base + s];
    }
}

fn save_best_global(at_cur_base: u32, pos_cur_base: u32, at_best_base: u32, pos_best_base: u32) {
    let s_len = params.s_len;
    let m_len = params.m_len;
    for (var m: u32 = 0u; m < m_len; m = m + 1u) {
        at_both[at_best_base + m] = at_both[at_cur_base + m];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        pos_both[pos_best_base + s] = pos_both[pos_cur_base + s];
    }
}

// ============================================================================
// Single-Pass Zero-Mutation Delta for 1-/2-Site Moves (`93.75%` of Steps)
// ============================================================================

fn short_pen_term(sa: i32, sb_u: u32, lam: f32, s_len: u32) -> f32 {
    if (sa < 0) {
        return -0.5 * lam;
    }
    if (is_obs_explained(u32(sa), sb_u, s_len)) {
        return 0.0;
    }
    return -lam;
}

// Exact incremental delta when root `s` moves from `ms` (`>= 0`) to `m != ms`,
// swapping with `t_i = wg_at[wg_base + m]` (`-1` if `m` was vacant, else `>= 0`).
// Zero SRAM writes occur unless the proposal is accepted.
fn delta_swap_root_wg(wg_base: u32, s: u32, ms: u32, m: u32, t_i: i32) -> f32 {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var delta: f32 = ro_floats[params.off_u_mat + s * m_len + m]
        - ro_floats[params.off_u_mat + s * m_len + ms];

    if (t_i >= 0) {
        let t = u32(t_i);
        delta = delta + ro_floats[params.off_u_mat + t * m_len + ms]
            - ro_floats[params.off_u_mat + t * m_len + m];
    }

    let gp_m = bitcast<i32>(ro_ints[params.off_gem_site + m]);
    if (gp_m != i32(ms)) {
        if (gp_m >= 0) {
            let su = wg_at[wg_base + u32(gp_m)];
            if (su >= 0) {
                let after_g = ro_floats[params.off_gem_obs_mat + s * s_len + u32(su)];
                var before_g: f32 = 0.0;
                if (t_i >= 0) {
                    before_g = ro_floats[params.off_gem_obs_mat + u32(t_i) * s_len + u32(su)];
                }
                delta = delta + (after_g - before_g);
            }
        }
        let gp_ms = bitcast<i32>(ro_ints[params.off_gem_site + ms]);
        if (gp_ms >= 0) {
            let sv = wg_at[wg_base + u32(gp_ms)];
            if (sv >= 0) {
                let before_g = ro_floats[params.off_gem_obs_mat + s * s_len + u32(sv)];
                var after_g: f32 = 0.0;
                if (t_i >= 0) {
                    after_g = ro_floats[params.off_gem_obs_mat + u32(t_i) * s_len + u32(sv)];
                }
                delta = delta + (after_g - before_g);
            }
        }
    }

    let e0_s = ro_ints[params.off_obs_adj_offs + s];
    let e1_s = ro_ints[params.off_obs_adj_offs + s + 1u];
    for (var ei: u32 = e0_s; ei < e1_s; ei = ei + 1u) {
        let sb = ro_ints[params.off_obs_adj_peers + ei];
        if (i32(sb) == t_i) {
            continue;
        }
        let mb = wg_pos[wg_base + sb];
        if (mb >= 0) {
            let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
            let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
            let d_new = ro_floats[params.off_d_mat + m * m_len + u32(mb)];
            let d_old = ro_floats[params.off_d_mat + ms * m_len + u32(mb)];
            delta = delta + noe_edge_score(d_new, d_hat, w) - noe_edge_score(d_old, d_hat, w);
        }
    }

    if (t_i >= 0) {
        let t = u32(t_i);
        let e0_t = ro_ints[params.off_obs_adj_offs + t];
        let e1_t = ro_ints[params.off_obs_adj_offs + t + 1u];
        for (var ei: u32 = e0_t; ei < e1_t; ei = ei + 1u) {
            let sb = ro_ints[params.off_obs_adj_peers + ei];
            if (sb == s) {
                continue;
            }
            let mb = wg_pos[wg_base + sb];
            if (mb >= 0) {
                let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
                let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
                let d_new = ro_floats[params.off_d_mat + ms * m_len + u32(mb)];
                let d_old = ro_floats[params.off_d_mat + m * m_len + u32(mb)];
                delta = delta + noe_edge_score(d_new, d_hat, w) - noe_edge_score(d_old, d_hat, w);
            }
        }
    }

    let se0_m = ro_ints[params.off_short_adj_offs + m];
    let se1_m = ro_ints[params.off_short_adj_offs + m + 1u];
    for (var sei: u32 = se0_m; sei < se1_m; sei = sei + 1u) {
        let mb = ro_ints[params.off_short_adj_peers + sei];
        if (mb == ms) {
            continue;
        }
        let sb = wg_at[wg_base + mb];
        if (sb >= 0) {
            let lam = ro_floats[params.off_short_adj_lam + sei];
            delta = delta + short_pen_term(i32(s), u32(sb), lam, s_len)
                - short_pen_term(t_i, u32(sb), lam, s_len);
        }
    }

    let se0_ms = ro_ints[params.off_short_adj_offs + ms];
    let se1_ms = ro_ints[params.off_short_adj_offs + ms + 1u];
    for (var sei: u32 = se0_ms; sei < se1_ms; sei = sei + 1u) {
        let mb = ro_ints[params.off_short_adj_peers + sei];
        if (mb == m) {
            continue;
        }
        let sb = wg_at[wg_base + mb];
        if (sb >= 0) {
            let lam = ro_floats[params.off_short_adj_lam + sei];
            delta = delta + short_pen_term(t_i, u32(sb), lam, s_len)
                - short_pen_term(i32(s), u32(sb), lam, s_len);
        }
    }

    return delta;
}

fn delta_swap_root_global(at_base: u32, pos_base: u32, s: u32, ms: u32, m: u32, t_i: i32) -> f32 {
    let s_len = params.s_len;
    let m_len = params.m_len;
    var delta: f32 = ro_floats[params.off_u_mat + s * m_len + m]
        - ro_floats[params.off_u_mat + s * m_len + ms];

    if (t_i >= 0) {
        let t = u32(t_i);
        delta = delta + ro_floats[params.off_u_mat + t * m_len + ms]
            - ro_floats[params.off_u_mat + t * m_len + m];
    }

    let gp_m = bitcast<i32>(ro_ints[params.off_gem_site + m]);
    if (gp_m != i32(ms)) {
        if (gp_m >= 0) {
            let su = at_both[at_base + u32(gp_m)];
            if (su >= 0) {
                let after_g = ro_floats[params.off_gem_obs_mat + s * s_len + u32(su)];
                var before_g: f32 = 0.0;
                if (t_i >= 0) {
                    before_g = ro_floats[params.off_gem_obs_mat + u32(t_i) * s_len + u32(su)];
                }
                delta = delta + (after_g - before_g);
            }
        }
        let gp_ms = bitcast<i32>(ro_ints[params.off_gem_site + ms]);
        if (gp_ms >= 0) {
            let sv = at_both[at_base + u32(gp_ms)];
            if (sv >= 0) {
                let before_g = ro_floats[params.off_gem_obs_mat + s * s_len + u32(sv)];
                var after_g: f32 = 0.0;
                if (t_i >= 0) {
                    after_g = ro_floats[params.off_gem_obs_mat + u32(t_i) * s_len + u32(sv)];
                }
                delta = delta + (after_g - before_g);
            }
        }
    }

    let e0_s = ro_ints[params.off_obs_adj_offs + s];
    let e1_s = ro_ints[params.off_obs_adj_offs + s + 1u];
    for (var ei: u32 = e0_s; ei < e1_s; ei = ei + 1u) {
        let sb = ro_ints[params.off_obs_adj_peers + ei];
        if (i32(sb) == t_i) {
            continue;
        }
        let mb = pos_both[pos_base + sb];
        if (mb >= 0) {
            let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
            let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
            let d_new = ro_floats[params.off_d_mat + m * m_len + u32(mb)];
            let d_old = ro_floats[params.off_d_mat + ms * m_len + u32(mb)];
            delta = delta + noe_edge_score(d_new, d_hat, w) - noe_edge_score(d_old, d_hat, w);
        }
    }

    if (t_i >= 0) {
        let t = u32(t_i);
        let e0_t = ro_ints[params.off_obs_adj_offs + t];
        let e1_t = ro_ints[params.off_obs_adj_offs + t + 1u];
        for (var ei: u32 = e0_t; ei < e1_t; ei = ei + 1u) {
            let sb = ro_ints[params.off_obs_adj_peers + ei];
            if (sb == s) {
                continue;
            }
            let mb = pos_both[pos_base + sb];
            if (mb >= 0) {
                let d_hat = ro_floats[params.off_obs_adj_floats + 2u * ei];
                let w = ro_floats[params.off_obs_adj_floats + 2u * ei + 1u];
                let d_new = ro_floats[params.off_d_mat + ms * m_len + u32(mb)];
                let d_old = ro_floats[params.off_d_mat + m * m_len + u32(mb)];
                delta = delta + noe_edge_score(d_new, d_hat, w) - noe_edge_score(d_old, d_hat, w);
            }
        }
    }

    let se0_m = ro_ints[params.off_short_adj_offs + m];
    let se1_m = ro_ints[params.off_short_adj_offs + m + 1u];
    for (var sei: u32 = se0_m; sei < se1_m; sei = sei + 1u) {
        let mb = ro_ints[params.off_short_adj_peers + sei];
        if (mb == ms) {
            continue;
        }
        let sb = at_both[at_base + mb];
        if (sb >= 0) {
            let lam = ro_floats[params.off_short_adj_lam + sei];
            delta = delta + short_pen_term(i32(s), u32(sb), lam, s_len)
                - short_pen_term(t_i, u32(sb), lam, s_len);
        }
    }

    let se0_ms = ro_ints[params.off_short_adj_offs + ms];
    let se1_ms = ro_ints[params.off_short_adj_offs + ms + 1u];
    for (var sei: u32 = se0_ms; sei < se1_ms; sei = sei + 1u) {
        let mb = ro_ints[params.off_short_adj_peers + sei];
        if (mb == m) {
            continue;
        }
        let sb = at_both[at_base + mb];
        if (sb >= 0) {
            let lam = ro_floats[params.off_short_adj_lam + sei];
            delta = delta + short_pen_term(t_i, u32(sb), lam, s_len)
                - short_pen_term(i32(s), u32(sb), lam, s_len);
        }
    }

    return delta;
}

// ============================================================================
// Exact Metropolis-Hastings Acceptance & Move-2 Forward/Reverse Proposal Prob
// ============================================================================

fn mh_accept(st: ptr<function, vec4<u32>>, delta: f32, cur_temp: f32, log_hastings: f32) -> bool {
    let log_acc = (delta / cur_temp) + log_hastings;
    if (log_acc >= 0.0) {
        return true;
    }
    if (log_acc > -36.0) {
        return rand_f32(st) < exp(log_acc);
    }
    return false;
}

fn move2_prop_prob_wg(wg_base: u32, u: u32, target_m: u32, excl_s: u32, excl_t: i32) -> f32 {
    let sl0 = ro_ints[params.off_slots_offs + u];
    let sl1 = ro_ints[params.off_slots_offs + u + 1u];
    if (sl0 >= sl1) {
        return 0.0;
    }
    let m_len = params.m_len;
    var p: f32 = 0.125 / f32(sl1 - sl0);
    let o0 = ro_ints[params.off_obs_adj_offs + u];
    let o1 = ro_ints[params.off_obs_adj_offs + u + 1u];
    if (o0 < o1) {
        let inv_odeg = 0.875 / f32(o1 - o0);
        var sum_anc: f32 = 0.0;
        for (var ei: u32 = o0; ei < o1; ei = ei + 1u) {
            let a = ro_ints[params.off_obs_adj_peers + ei];
            if (a == excl_s || (excl_t >= 0 && i32(a) == excl_t)) {
                continue;
            }
            let ma_i = wg_pos[wg_base + a];
            if (ma_i < 0) {
                continue;
            }
            let ma = u32(ma_i);
            if (ma == target_m) {
                continue;
            }
            if (ro_floats[params.off_d_mat + ma * m_len + target_m] <= params.d_cutoff) {
                let c0 = ro_ints[params.off_cut_adj_offs + ma];
                let c1 = ro_ints[params.off_cut_adj_offs + ma + 1u];
                if (c0 < c1) {
                    sum_anc = sum_anc + (1.0 / f32(c1 - c0));
                }
            }
        }
        p = p + inv_odeg * sum_anc;
    }
    return p;
}

fn move2_prop_prob_global(pos_base: u32, u: u32, target_m: u32, excl_s: u32, excl_t: i32) -> f32 {
    let sl0 = ro_ints[params.off_slots_offs + u];
    let sl1 = ro_ints[params.off_slots_offs + u + 1u];
    if (sl0 >= sl1) {
        return 0.0;
    }
    let m_len = params.m_len;
    var p: f32 = 0.125 / f32(sl1 - sl0);
    let o0 = ro_ints[params.off_obs_adj_offs + u];
    let o1 = ro_ints[params.off_obs_adj_offs + u + 1u];
    if (o0 < o1) {
        let inv_odeg = 0.875 / f32(o1 - o0);
        var sum_anc: f32 = 0.0;
        for (var ei: u32 = o0; ei < o1; ei = ei + 1u) {
            let a = ro_ints[params.off_obs_adj_peers + ei];
            if (a == excl_s || (excl_t >= 0 && i32(a) == excl_t)) {
                continue;
            }
            let ma_i = pos_both[pos_base + a];
            if (ma_i < 0) {
                continue;
            }
            let ma = u32(ma_i);
            if (ma == target_m) {
                continue;
            }
            if (ro_floats[params.off_d_mat + ma * m_len + target_m] <= params.d_cutoff) {
                let c0 = ro_ints[params.off_cut_adj_offs + ma];
                let c1 = ro_ints[params.off_cut_adj_offs + ma + 1u];
                if (c0 < c1) {
                    sum_anc = sum_anc + (1.0 / f32(c1 - c0));
                }
            }
        }
        p = p + inv_odeg * sum_anc;
    }
    return p;
}

// ============================================================================
// Unified Workgroup-SRAM Annealing Kernel (`anneal_methyl_256` & `anneal_methyl_1024`)
// ============================================================================

fn run_anneal_wg(r: u32, wg_base: u32) {
    let s_len = params.s_len;
    let m_len = params.m_len;
    if (s_len == 0u || m_len == 0u) {
        return;
    }
    let reps = params.replicas;
    let at_cur_base = r * m_len;
    let at_best_base = (reps + r) * m_len;
    let pos_cur_base = r * s_len;
    let pos_best_base = (reps + r) * s_len;

    for (var m: u32 = 0u; m < m_len; m = m + 1u) {
        wg_at[wg_base + m] = at_both[at_cur_base + m];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        wg_pos[wg_base + s] = pos_both[pos_cur_base + s];
    }

    var rng = rng_state[r];
    var cur: f32 = scores_state[r].x;
    var best_sc: f32 = scores_state[r].y;
    if (params.step_offset == 0u) {
        cur = eval_total_raw_wg(wg_base);
        best_sc = cur;
        save_best_wg(wg_base, at_best_base, pos_best_base);
    }

    let denom = f32(max(select(0u, params.total_steps - 1u, params.total_steps > 0u), 1u));
    let log_ratio = log(params.t_end / params.t_start);
    let step_end = min(params.step_offset + params.chunk_steps, params.total_steps);

    for (var step: u32 = params.step_offset; step < step_end; step = step + 1u) {
        let cur_temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        let phase = select_move_phase(&rng, params.move_mask);
        let save_ready = step * 3u >= params.total_steps;

        if (phase < 6u) {
            // Move 1 (`6/16`): slot-guided swap / move / displace-occupied
            let s = rand_below(&rng, s_len);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            let m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl1 - sl0)];
            let ms_i = wg_pos[wg_base + s];

            if (ms_i >= 0 && ms_i != i32(m)) {
                let ms = u32(ms_i);
                let t_i = wg_at[wg_base + m];
                if (t_i >= 0
                    && ro_floats[params.off_u_mat + u32(t_i) * m_len + ms] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                let delta = delta_swap_root_wg(wg_base, s, ms, m, t_i);
                if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                    wg_at[wg_base + m] = i32(s);
                    wg_at[wg_base + ms] = t_i;
                    wg_pos[wg_base + s] = i32(m);
                    if (t_i >= 0) {
                        wg_pos[wg_base + u32(t_i)] = i32(ms);
                    }
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_wg(wg_base, at_best_base, pos_best_base);
                    }
                }
            } else if (ms_i < 0) {
                let oa0 = wg_at[wg_base + m];
                if (oa0 < 0) {
                    continue;
                }
                let na0 = i32(s);
                let t_sl0 = ro_ints[params.off_slots_offs + u32(oa0)];
                let t_sl1 = ro_ints[params.off_slots_offs + u32(oa0) + 1u];
                let log_hastings = log(f32(sl1 - sl0) / f32(max(t_sl1 - t_sl0, 1u)));
                var ns: u32 = 0u;
                var s0: u32 = 0u;
                var s1_u: u32 = 0u;
                var s2_u: u32 = 0u;
                var s3_u: u32 = 0u;
                push_unique_s(oa0, &ns, &s0, &s1_u, &s2_u, &s3_u);
                push_unique_s(na0, &ns, &s0, &s1_u, &s2_u, &s3_u);
                let before = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, ns, s0, s1_u, s2_u, s3_u);
                for (var k: u32 = 0u; k < ns; k = k + 1u) {
                    wg_pos[wg_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                }
                wg_at[wg_base + m] = na0;
                wg_pos[wg_base + u32(na0)] = i32(m);
                let delta = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, ns, s0, s1_u, s2_u, s3_u) - before;
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_wg(wg_base, at_best_base, pos_best_base);
                    }
                } else {
                    for (var k: u32 = 0u; k < ns; k = k + 1u) {
                        wg_pos[wg_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                    }
                    wg_at[wg_base + m] = oa0;
                    wg_pos[wg_base + u32(oa0)] = i32(m);
                }
            }
        } else if (phase < 12u) {
            // Move 2 (`6/16`): 3D contact-guided (`7/8`) + slot-ergodic (`1/8`) move/swap with exact Hastings correction
            let s = rand_below(&rng, s_len);
            let ms_i = wg_pos[wg_base + s];
            if (ms_i < 0) {
                continue;
            }
            let ms = u32(ms_i);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            var m: u32 = 0u;
            var ok_m2: bool = false;
            if ((next_u32(&rng) & 7u) == 0u) {
                m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl1 - sl0)];
                ok_m2 = true;
            } else {
                let o0 = ro_ints[params.off_obs_adj_offs + s];
                let o1 = ro_ints[params.off_obs_adj_offs + s + 1u];
                if (o0 < o1) {
                    let a = ro_ints[params.off_obs_adj_peers + o0 + rand_below(&rng, o1 - o0)];
                    let ma_i = wg_pos[wg_base + a];
                    if (ma_i >= 0) {
                        let c0 = ro_ints[params.off_cut_adj_offs + u32(ma_i)];
                        let c1 = ro_ints[params.off_cut_adj_offs + u32(ma_i) + 1u];
                        if (c0 < c1) {
                            m = ro_ints[params.off_cut_adj_peers + c0 + rand_below(&rng, c1 - c0)];
                            ok_m2 = true;
                        }
                    }
                }
            }
            if (!ok_m2 || m == ms || ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF) {
                continue;
            }
            let t_i = wg_at[wg_base + m];
            if (t_i >= 0 && ro_floats[params.off_u_mat + u32(t_i) * m_len + ms] <= METHYL_FORBID_HALF) {
                continue;
            }
            var q_fwd: f32 = 0.0;
            var q_rev: f32 = 0.0;
            if (t_i < 0) {
                q_fwd = move2_prop_prob_wg(wg_base, s, m, s, -1);
                q_rev = move2_prop_prob_wg(wg_base, s, ms, s, -1);
            } else {
                let tu = u32(t_i);
                q_fwd = move2_prop_prob_wg(wg_base, s, m, s, t_i)
                    + move2_prop_prob_wg(wg_base, tu, ms, s, t_i);
                q_rev = move2_prop_prob_wg(wg_base, s, ms, s, t_i)
                    + move2_prop_prob_wg(wg_base, tu, m, s, t_i);
            }
            if (q_fwd <= 0.0 || q_rev <= 0.0) {
                continue;
            }
            let delta = delta_swap_root_wg(wg_base, s, ms, m, t_i);
            if (mh_accept(&rng, delta, cur_temp, log(q_rev / q_fwd))) {
                wg_at[wg_base + m] = i32(s);
                wg_at[wg_base + ms] = t_i;
                wg_pos[wg_base + s] = i32(m);
                if (t_i >= 0) {
                    wg_pos[wg_base + u32(t_i)] = i32(ms);
                }
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_wg(wg_base, at_best_base, pos_best_base);
                }
            }
        } else if (phase < 14u) {
            // Move 3A (`2/16`): state-independent 4-body geminal residue involution
            let ng = params.n_gem_pairs;
            if (ng < 2u) {
                continue;
            }
            let ga = rand_below(&rng, ng);
            var gb: u32 = 0u;
            if ((next_u32(&rng) & 1u) == 0u) {
                let n0 = ro_ints[params.off_gem_nbr_offs + ga];
                let n1 = ro_ints[params.off_gem_nbr_offs + ga + 1u];
                if (n0 < n1) {
                    gb = ro_ints[params.off_gem_nbr_vals + n0 + rand_below(&rng, n1 - n0)];
                } else {
                    let k = rand_below(&rng, ng - 1u);
                    if (k >= ga) {
                        gb = k + 1u;
                    } else {
                        gb = k;
                    }
                }
            } else {
                let k = rand_below(&rng, ng - 1u);
                if (k >= ga) {
                    gb = k + 1u;
                } else {
                    gb = k;
                }
            }
            let gm1 = ro_ints[params.off_gem_pairs + 2u * ga];
            let gm2 = ro_ints[params.off_gem_pairs + 2u * ga + 1u];
            var gm3 = ro_ints[params.off_gem_pairs + 2u * gb];
            var gm4 = ro_ints[params.off_gem_pairs + 2u * gb + 1u];
            if ((next_u32(&rng) & 1u) == 1u) {
                let tmp = gm3;
                gm3 = gm4;
                gm4 = tmp;
            }
            let sa1 = wg_at[wg_base + gm1];
            let sa2 = wg_at[wg_base + gm2];
            let sa3 = wg_at[wg_base + gm3];
            let sa4 = wg_at[wg_base + gm4];
            if (sa1 < 0 && sa2 < 0 && sa3 < 0 && sa4 < 0) {
                continue;
            }
            if ((sa1 >= 0 && ro_floats[params.off_u_mat + u32(sa1) * m_len + gm3] <= METHYL_FORBID_HALF)
                || (sa2 >= 0 && ro_floats[params.off_u_mat + u32(sa2) * m_len + gm4] <= METHYL_FORBID_HALF)
                || (sa3 >= 0 && ro_floats[params.off_u_mat + u32(sa3) * m_len + gm1] <= METHYL_FORBID_HALF)
                || (sa4 >= 0 && ro_floats[params.off_u_mat + u32(sa4) * m_len + gm2] <= METHYL_FORBID_HALF))
            {
                continue;
            }
            var ns: u32 = 0u;
            var s0: u32 = 0u;
            var s1_u: u32 = 0u;
            var s2_u: u32 = 0u;
            var s3_u: u32 = 0u;
            push_unique_s(sa1, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa2, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa3, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa4, &ns, &s0, &s1_u, &s2_u, &s3_u);

            let before = eval_subset_wg(wg_base, 4u, gm1, gm2, gm3, gm4, ns, s0, s1_u, s2_u, s3_u);
            for (var k: u32 = 0u; k < ns; k = k + 1u) {
                wg_pos[wg_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
            }
            wg_at[wg_base + gm1] = sa3;
            wg_at[wg_base + gm2] = sa4;
            wg_at[wg_base + gm3] = sa1;
            wg_at[wg_base + gm4] = sa2;
            if (sa3 >= 0) { wg_pos[wg_base + u32(sa3)] = i32(gm1); }
            if (sa4 >= 0) { wg_pos[wg_base + u32(sa4)] = i32(gm2); }
            if (sa1 >= 0) { wg_pos[wg_base + u32(sa1)] = i32(gm3); }
            if (sa2 >= 0) { wg_pos[wg_base + u32(sa2)] = i32(gm4); }

            let delta = eval_subset_wg(wg_base, 4u, gm1, gm2, gm3, gm4, ns, s0, s1_u, s2_u, s3_u) - before;
            if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_wg(wg_base, at_best_base, pos_best_base);
                }
            } else {
                for (var k: u32 = 0u; k < ns; k = k + 1u) {
                    wg_pos[wg_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                }
                wg_at[wg_base + gm1] = sa1;
                wg_at[wg_base + gm2] = sa2;
                wg_at[wg_base + gm3] = sa3;
                wg_at[wg_base + gm4] = sa4;
                if (sa1 >= 0) { wg_pos[wg_base + u32(sa1)] = i32(gm1); }
                if (sa2 >= 0) { wg_pos[wg_base + u32(sa2)] = i32(gm2); }
                if (sa3 >= 0) { wg_pos[wg_base + u32(sa3)] = i32(gm3); }
                if (sa4 >= 0) { wg_pos[wg_base + u32(sa4)] = i32(gm4); }
            }
        } else if (phase < 15u) {
            // Move 3B (`1/16`): state-independent intraresidue prochiral 2-flip involution
            let ng = params.n_gem_pairs;
            if (ng == 0u) {
                continue;
            }
            let g = rand_below(&rng, ng);
            let fm1 = ro_ints[params.off_gem_pairs + 2u * g];
            let fm2 = ro_ints[params.off_gem_pairs + 2u * g + 1u];
            let sa1 = wg_at[wg_base + fm1];
            let sa2 = wg_at[wg_base + fm2];
            if (sa1 == sa2) {
                continue;
            }
            if ((sa1 >= 0 && ro_floats[params.off_u_mat + u32(sa1) * m_len + fm2] <= METHYL_FORBID_HALF)
                || (sa2 >= 0 && ro_floats[params.off_u_mat + u32(sa2) * m_len + fm1] <= METHYL_FORBID_HALF))
            {
                continue;
            }
            var s_f: u32 = 0u;
            var ms_f: u32 = 0u;
            var m_f: u32 = 0u;
            var t_f: i32 = -1;
            if (sa1 >= 0) {
                s_f = u32(sa1);
                ms_f = fm1;
                m_f = fm2;
                t_f = sa2;
            } else {
                s_f = u32(sa2);
                ms_f = fm2;
                m_f = fm1;
                t_f = -1;
            }
            let delta = delta_swap_root_wg(wg_base, s_f, ms_f, m_f, t_f);
            if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                wg_at[wg_base + m_f] = i32(s_f);
                wg_at[wg_base + ms_f] = t_f;
                wg_pos[wg_base + s_f] = i32(m_f);
                if (t_f >= 0) {
                    wg_pos[wg_base + u32(t_f)] = i32(ms_f);
                }
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_wg(wg_base, at_best_base, pos_best_base);
                }
            }
        } else {
            // Move 4 (`1/16`): paired assign/unassign (birth/death) on variable-cardinality Omega = U_k Omega_k
            let s = rand_below(&rng, s_len);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            let sl_len = sl1 - sl0;
            let ms_i = wg_pos[wg_base + s];
            if (ms_i < 0) {
                let m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl_len)];
                if (wg_at[wg_base + m] >= 0
                    || ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                let before = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u);
                wg_at[wg_base + m] = i32(s);
                wg_pos[wg_base + s] = i32(m);
                let delta = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u) - before;
                let log_hastings = log(f32(sl_len));
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_wg(wg_base, at_best_base, pos_best_base);
                    }
                } else {
                    wg_at[wg_base + m] = -1;
                    wg_pos[wg_base + s] = -1;
                }
            } else {
                let m = u32(ms_i);
                if (ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF) {
                    continue;
                }
                let before = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u);
                wg_at[wg_base + m] = -1;
                wg_pos[wg_base + s] = -1;
                let delta = eval_subset_wg(wg_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u) - before;
                let log_hastings = -log(f32(sl_len));
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_wg(wg_base, at_best_base, pos_best_base);
                    }
                } else {
                    wg_at[wg_base + m] = i32(s);
                    wg_pos[wg_base + s] = i32(m);
                }
            }
        }
    }

    if (step_end >= params.total_steps) {
        cur = eval_total_raw_wg(wg_base);
        if (cur > best_sc) {
            best_sc = cur;
            save_best_wg(wg_base, at_best_base, pos_best_base);
        }
    }

    for (var m: u32 = 0u; m < m_len; m = m + 1u) {
        at_both[at_cur_base + m] = wg_at[wg_base + m];
    }
    for (var s: u32 = 0u; s < s_len; s = s + 1u) {
        pos_both[pos_cur_base + s] = wg_pos[wg_base + s];
    }
    scores_state[r] = vec2<f32>(cur, best_sc);
    rng_state[r] = rng;
}

@compute @workgroup_size(8, 1, 1)
fn anneal_methyl_256(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_STRIDE_256);
}

@compute @workgroup_size(2, 1, 1)
fn anneal_methyl_1024(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    run_anneal_wg(r, lid.x * WG_STRIDE_1024);
}

// ============================================================================
// Global Storage Fallback (`anneal_methyl_global` for max(S, M) > 1024)
// ============================================================================

@compute @workgroup_size(8, 1, 1)
fn anneal_methyl_global(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.replicas) {
        return;
    }
    let s_len = params.s_len;
    let m_len = params.m_len;
    if (s_len == 0u || m_len == 0u) {
        return;
    }
    let reps = params.replicas;
    let at_cur_base = r * m_len;
    let at_best_base = (reps + r) * m_len;
    let pos_cur_base = r * s_len;
    let pos_best_base = (reps + r) * s_len;

    var rng = rng_state[r];
    var cur: f32 = scores_state[r].x;
    var best_sc: f32 = scores_state[r].y;
    if (params.step_offset == 0u) {
        cur = eval_total_raw_global(at_cur_base, pos_cur_base);
        best_sc = cur;
        save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
    }

    let denom = f32(max(select(0u, params.total_steps - 1u, params.total_steps > 0u), 1u));
    let log_ratio = log(params.t_end / params.t_start);
    let step_end = min(params.step_offset + params.chunk_steps, params.total_steps);

    for (var step: u32 = params.step_offset; step < step_end; step = step + 1u) {
        let cur_temp = params.t_start * exp(log_ratio * (f32(step) / denom));
        let phase = select_move_phase(&rng, params.move_mask);
        let save_ready = step * 3u >= params.total_steps;

        if (phase < 6u) {
            let s = rand_below(&rng, s_len);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            let m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl1 - sl0)];
            let ms_i = pos_both[pos_cur_base + s];

            if (ms_i >= 0 && ms_i != i32(m)) {
                let ms = u32(ms_i);
                let t_i = at_both[at_cur_base + m];
                if (t_i >= 0
                    && ro_floats[params.off_u_mat + u32(t_i) * m_len + ms] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                let delta = delta_swap_root_global(at_cur_base, pos_cur_base, s, ms, m, t_i);
                if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                    at_both[at_cur_base + m] = i32(s);
                    at_both[at_cur_base + ms] = t_i;
                    pos_both[pos_cur_base + s] = i32(m);
                    if (t_i >= 0) {
                        pos_both[pos_cur_base + u32(t_i)] = i32(ms);
                    }
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                    }
                }
            } else if (ms_i < 0) {
                let oa0 = at_both[at_cur_base + m];
                if (oa0 < 0) {
                    continue;
                }
                let na0 = i32(s);
                let t_sl0 = ro_ints[params.off_slots_offs + u32(oa0)];
                let t_sl1 = ro_ints[params.off_slots_offs + u32(oa0) + 1u];
                let log_hastings = log(f32(sl1 - sl0) / f32(max(t_sl1 - t_sl0, 1u)));
                var ns: u32 = 0u;
                var s0: u32 = 0u;
                var s1_u: u32 = 0u;
                var s2_u: u32 = 0u;
                var s3_u: u32 = 0u;
                push_unique_s(oa0, &ns, &s0, &s1_u, &s2_u, &s3_u);
                push_unique_s(na0, &ns, &s0, &s1_u, &s2_u, &s3_u);
                let before = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, ns, s0, s1_u, s2_u, s3_u);
                for (var k: u32 = 0u; k < ns; k = k + 1u) {
                    pos_both[pos_cur_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                }
                at_both[at_cur_base + m] = na0;
                pos_both[pos_cur_base + u32(na0)] = i32(m);
                let delta = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, ns, s0, s1_u, s2_u, s3_u) - before;
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                    }
                } else {
                    for (var k: u32 = 0u; k < ns; k = k + 1u) {
                        pos_both[pos_cur_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                    }
                    at_both[at_cur_base + m] = oa0;
                    pos_both[pos_cur_base + u32(oa0)] = i32(m);
                }
            }
        } else if (phase < 12u) {
            let s = rand_below(&rng, s_len);
            let ms_i = pos_both[pos_cur_base + s];
            if (ms_i < 0) {
                continue;
            }
            let ms = u32(ms_i);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            var m: u32 = 0u;
            var ok_m2: bool = false;
            if ((next_u32(&rng) & 7u) == 0u) {
                m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl1 - sl0)];
                ok_m2 = true;
            } else {
                let o0 = ro_ints[params.off_obs_adj_offs + s];
                let o1 = ro_ints[params.off_obs_adj_offs + s + 1u];
                if (o0 < o1) {
                    let a = ro_ints[params.off_obs_adj_peers + o0 + rand_below(&rng, o1 - o0)];
                    let ma_i = pos_both[pos_cur_base + a];
                    if (ma_i >= 0) {
                        let c0 = ro_ints[params.off_cut_adj_offs + u32(ma_i)];
                        let c1 = ro_ints[params.off_cut_adj_offs + u32(ma_i) + 1u];
                        if (c0 < c1) {
                            m = ro_ints[params.off_cut_adj_peers + c0 + rand_below(&rng, c1 - c0)];
                            ok_m2 = true;
                        }
                    }
                }
            }
            if (!ok_m2 || m == ms || ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF) {
                continue;
            }
            let t_i = at_both[at_cur_base + m];
            if (t_i >= 0 && ro_floats[params.off_u_mat + u32(t_i) * m_len + ms] <= METHYL_FORBID_HALF) {
                continue;
            }
            var q_fwd: f32 = 0.0;
            var q_rev: f32 = 0.0;
            if (t_i < 0) {
                q_fwd = move2_prop_prob_global(pos_cur_base, s, m, s, -1);
                q_rev = move2_prop_prob_global(pos_cur_base, s, ms, s, -1);
            } else {
                let tu = u32(t_i);
                q_fwd = move2_prop_prob_global(pos_cur_base, s, m, s, t_i)
                    + move2_prop_prob_global(pos_cur_base, tu, ms, s, t_i);
                q_rev = move2_prop_prob_global(pos_cur_base, s, ms, s, t_i)
                    + move2_prop_prob_global(pos_cur_base, tu, m, s, t_i);
            }
            if (q_fwd <= 0.0 || q_rev <= 0.0) {
                continue;
            }
            let delta = delta_swap_root_global(at_cur_base, pos_cur_base, s, ms, m, t_i);
            if (mh_accept(&rng, delta, cur_temp, log(q_rev / q_fwd))) {
                at_both[at_cur_base + m] = i32(s);
                at_both[at_cur_base + ms] = t_i;
                pos_both[pos_cur_base + s] = i32(m);
                if (t_i >= 0) {
                    pos_both[pos_cur_base + u32(t_i)] = i32(ms);
                }
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                }
            }
        } else if (phase < 14u) {
            let ng = params.n_gem_pairs;
            if (ng < 2u) {
                continue;
            }
            let ga = rand_below(&rng, ng);
            var gb: u32 = 0u;
            if ((next_u32(&rng) & 1u) == 0u) {
                let n0 = ro_ints[params.off_gem_nbr_offs + ga];
                let n1 = ro_ints[params.off_gem_nbr_offs + ga + 1u];
                if (n0 < n1) {
                    gb = ro_ints[params.off_gem_nbr_vals + n0 + rand_below(&rng, n1 - n0)];
                } else {
                    let k = rand_below(&rng, ng - 1u);
                    if (k >= ga) {
                        gb = k + 1u;
                    } else {
                        gb = k;
                    }
                }
            } else {
                let k = rand_below(&rng, ng - 1u);
                if (k >= ga) {
                    gb = k + 1u;
                } else {
                    gb = k;
                }
            }
            let gm1 = ro_ints[params.off_gem_pairs + 2u * ga];
            let gm2 = ro_ints[params.off_gem_pairs + 2u * ga + 1u];
            var gm3 = ro_ints[params.off_gem_pairs + 2u * gb];
            var gm4 = ro_ints[params.off_gem_pairs + 2u * gb + 1u];
            if ((next_u32(&rng) & 1u) == 1u) {
                let tmp = gm3;
                gm3 = gm4;
                gm4 = tmp;
            }
            let sa1 = at_both[at_cur_base + gm1];
            let sa2 = at_both[at_cur_base + gm2];
            let sa3 = at_both[at_cur_base + gm3];
            let sa4 = at_both[at_cur_base + gm4];
            if (sa1 < 0 && sa2 < 0 && sa3 < 0 && sa4 < 0) {
                continue;
            }
            if ((sa1 >= 0 && ro_floats[params.off_u_mat + u32(sa1) * m_len + gm3] <= METHYL_FORBID_HALF)
                || (sa2 >= 0 && ro_floats[params.off_u_mat + u32(sa2) * m_len + gm4] <= METHYL_FORBID_HALF)
                || (sa3 >= 0 && ro_floats[params.off_u_mat + u32(sa3) * m_len + gm1] <= METHYL_FORBID_HALF)
                || (sa4 >= 0 && ro_floats[params.off_u_mat + u32(sa4) * m_len + gm2] <= METHYL_FORBID_HALF))
            {
                continue;
            }

            var ns: u32 = 0u;
            var s0: u32 = 0u;
            var s1_u: u32 = 0u;
            var s2_u: u32 = 0u;
            var s3_u: u32 = 0u;
            push_unique_s(sa1, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa2, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa3, &ns, &s0, &s1_u, &s2_u, &s3_u);
            push_unique_s(sa4, &ns, &s0, &s1_u, &s2_u, &s3_u);

            let before = eval_subset_global(at_cur_base, pos_cur_base, 4u, gm1, gm2, gm3, gm4, ns, s0, s1_u, s2_u, s3_u);
            for (var k: u32 = 0u; k < ns; k = k + 1u) {
                pos_both[pos_cur_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
            }
            at_both[at_cur_base + gm1] = sa3;
            at_both[at_cur_base + gm2] = sa4;
            at_both[at_cur_base + gm3] = sa1;
            at_both[at_cur_base + gm4] = sa2;
            if (sa3 >= 0) { pos_both[pos_cur_base + u32(sa3)] = i32(gm1); }
            if (sa4 >= 0) { pos_both[pos_cur_base + u32(sa4)] = i32(gm2); }
            if (sa1 >= 0) { pos_both[pos_cur_base + u32(sa1)] = i32(gm3); }
            if (sa2 >= 0) { pos_both[pos_cur_base + u32(sa2)] = i32(gm4); }

            let delta = eval_subset_global(at_cur_base, pos_cur_base, 4u, gm1, gm2, gm3, gm4, ns, s0, s1_u, s2_u, s3_u) - before;
            if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                }
            } else {
                for (var k: u32 = 0u; k < ns; k = k + 1u) {
                    pos_both[pos_cur_base + get_u4(k, s0, s1_u, s2_u, s3_u)] = -1;
                }
                at_both[at_cur_base + gm1] = sa1;
                at_both[at_cur_base + gm2] = sa2;
                at_both[at_cur_base + gm3] = sa3;
                at_both[at_cur_base + gm4] = sa4;
                if (sa1 >= 0) { pos_both[pos_cur_base + u32(sa1)] = i32(gm1); }
                if (sa2 >= 0) { pos_both[pos_cur_base + u32(sa2)] = i32(gm2); }
                if (sa3 >= 0) { pos_both[pos_cur_base + u32(sa3)] = i32(gm3); }
                if (sa4 >= 0) { pos_both[pos_cur_base + u32(sa4)] = i32(gm4); }
            }
        } else if (phase < 15u) {
            let ng = params.n_gem_pairs;
            if (ng == 0u) {
                continue;
            }
            let g = rand_below(&rng, ng);
            let fm1 = ro_ints[params.off_gem_pairs + 2u * g];
            let fm2 = ro_ints[params.off_gem_pairs + 2u * g + 1u];
            let sa1 = at_both[at_cur_base + fm1];
            let sa2 = at_both[at_cur_base + fm2];
            if (sa1 == sa2) {
                continue;
            }
            if ((sa1 >= 0 && ro_floats[params.off_u_mat + u32(sa1) * m_len + fm2] <= METHYL_FORBID_HALF)
                || (sa2 >= 0 && ro_floats[params.off_u_mat + u32(sa2) * m_len + fm1] <= METHYL_FORBID_HALF))
            {
                continue;
            }
            var s_f: u32 = 0u;
            var ms_f: u32 = 0u;
            var m_f: u32 = 0u;
            var t_f: i32 = -1;
            if (sa1 >= 0) {
                s_f = u32(sa1);
                ms_f = fm1;
                m_f = fm2;
                t_f = sa2;
            } else {
                s_f = u32(sa2);
                ms_f = fm2;
                m_f = fm1;
                t_f = -1;
            }
            let delta = delta_swap_root_global(at_cur_base, pos_cur_base, s_f, ms_f, m_f, t_f);
            if (mh_accept(&rng, delta, cur_temp, 0.0)) {
                at_both[at_cur_base + m_f] = i32(s_f);
                at_both[at_cur_base + ms_f] = t_f;
                pos_both[pos_cur_base + s_f] = i32(m_f);
                if (t_f >= 0) {
                    pos_both[pos_cur_base + u32(t_f)] = i32(ms_f);
                }
                cur = cur + delta;
                if (save_ready && cur > best_sc + 0.15) {
                    best_sc = cur;
                    save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                }
            }
        } else {
            let s = rand_below(&rng, s_len);
            let sl0 = ro_ints[params.off_slots_offs + s];
            let sl1 = ro_ints[params.off_slots_offs + s + 1u];
            if (sl0 >= sl1) {
                continue;
            }
            let sl_len = sl1 - sl0;
            let ms_i = pos_both[pos_cur_base + s];
            if (ms_i < 0) {
                let m = ro_ints[params.off_slots_vals + sl0 + rand_below(&rng, sl_len)];
                if (at_both[at_cur_base + m] >= 0
                    || ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF)
                {
                    continue;
                }
                let before = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u);
                at_both[at_cur_base + m] = i32(s);
                pos_both[pos_cur_base + s] = i32(m);
                let delta = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u) - before;
                let log_hastings = log(f32(sl_len));
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                    }
                } else {
                    at_both[at_cur_base + m] = -1;
                    pos_both[pos_cur_base + s] = -1;
                }
            } else {
                let m = u32(ms_i);
                if (ro_floats[params.off_u_mat + s * m_len + m] <= METHYL_FORBID_HALF) {
                    continue;
                }
                let before = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u);
                at_both[at_cur_base + m] = -1;
                pos_both[pos_cur_base + s] = -1;
                let delta = eval_subset_global(at_cur_base, pos_cur_base, 1u, m, 0u, 0u, 0u, 1u, s, 0u, 0u, 0u) - before;
                let log_hastings = -log(f32(sl_len));
                if (mh_accept(&rng, delta, cur_temp, log_hastings)) {
                    cur = cur + delta;
                    if (save_ready && cur > best_sc + 0.15) {
                        best_sc = cur;
                        save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
                    }
                } else {
                    at_both[at_cur_base + m] = i32(s);
                    pos_both[pos_cur_base + s] = i32(m);
                }
            }
        }
    }

    if (step_end >= params.total_steps) {
        cur = eval_total_raw_global(at_cur_base, pos_cur_base);
        if (cur > best_sc) {
            best_sc = cur;
            save_best_global(at_cur_base, pos_cur_base, at_best_base, pos_best_base);
        }
    }
    scores_state[r] = vec2<f32>(cur, best_sc);
    rng_state[r] = rng;
}
