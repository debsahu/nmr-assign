// Clean-room WGSL compute shaders for backbone NMR unary typing U(s, k)
// and pairwise sequential connectivity P(s, t) matrices.
// Provenance: Implements docs/specs/03-assignment.md section 3.

struct ScoreParams {
    s_len: u32,
    k_len: u32,
    max_cands: u32,
    use_hn: u32,
    sigma_ca: f32,
    sigma_cb: f32,
    sigma_c: f32,
    range_ppm: f32,
    pair_floor: f32,
    support_bonus: f32,
    llr_floor: f32,
    gly_cb: f32,
    assign_cost: f32,
    forbid_zero_carbons: u32,
    pad0: u32,
    pad1: u32,
};

struct PriorEntry {
    mu: f32,
    sd: f32,
    present: u32,
    _pad: u32,
};

struct CandEntry {
    val: f32,
    support: u32,
};

@group(0) @binding(0) var<uniform> params: ScoreParams;
// 20 residues x 5 atoms (H=0, N=1, CA=2, CB=3, C=4)
@group(0) @binding(1) var<storage, read> priors: array<PriorEntry>;
// Sequence residue type index (0..19, 14 = PRO, 255 = UNK)
@group(0) @binding(2) var<storage, read> seq_res: array<u32>;
// Root (H, N) per spin system: vec2<f32>(H, N)
@group(0) @binding(3) var<storage, read> roots_hn: array<vec2<f32>>;
// Candidates indexed [((s * 3 + atom_c) * 2 + off_idx) * max_cands + c]
// atom_c: 0=CA, 1=CB, 2=C; off_idx: 0=(i), 1=(i-1)
@group(0) @binding(4) var<storage, read> cands: array<CandEntry>;
// Counts indexed [(s * 3 + atom_c) * 2 + off_idx]
@group(0) @binding(5) var<storage, read> cand_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_p: array<f32>;

const TAU: f32 = 6.283185307179586;
const FORBID: f32 = -1e9;
const PRO_IDX: u32 = 14u;

fn logpdf(v: f32, mu: f32, sd: f32) -> f32 {
    let z = (v - mu) / sd;
    return -0.5 * z * z - log(sd * sqrt(TAU));
}

fn llr(res_idx: u32, atom_idx: u32, v: f32) -> f32 {
    if (res_idx >= 20u) {
        return 0.0;
    }
    let pe = priors[res_idx * 5u + atom_idx];
    if (pe.present == 0u) {
        return params.gly_cb;
    }
    let lp_res = logpdf(v, pe.mu, pe.sd);

    var max_lp: f32 = -1e30;
    var count: u32 = 0u;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            let lp = logpdf(v, p_r.mu, p_r.sd);
            if (lp > max_lp) {
                max_lp = lp;
            }
            count = count + 1u;
        }
    }
    var sum_exp: f32 = 0.0;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            sum_exp = sum_exp + exp(logpdf(v, p_r.mu, p_r.sd) - max_lp);
        }
    }
    let bg = max_lp + log(sum_exp / f32(count));
    return max(lp_res - bg, params.llr_floor);
}

@compute @workgroup_size(16, 16, 1)
fn compute_unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let k = gid.y;
    if (s >= params.s_len || k >= params.k_len) {
        return;
    }
    let out_idx = s * params.k_len + k;
    if (k == 0u) {
        out_u[out_idx] = FORBID;
        return;
    }
    let res = seq_res[k];
    if (res == PRO_IDX) {
        out_u[out_idx] = FORBID;
        return;
    }
    let prev = seq_res[k - 1u];

    var sc: f32 = 0.0;
    var total_pos_cands: u32 = 0u;
    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let atom_idx = atom_c + 2u; // 2=CA, 3=CB, 4=C
        for (var off_idx: u32 = 0u; off_idx < 2u; off_idx = off_idx + 1u) {
            let slot = (s * 3u + atom_c) * 2u + off_idx;
            let n_raw = cand_counts[slot];
            if (n_raw == 0u) {
                continue;
            }
            let base = slot * params.max_cands;
            let target_res = select(prev, res, off_idx == 0u);
            var n_pos: u32 = 0u;
            var best: f32 = -1e30;
            for (var c: u32 = 0u; c < n_raw; c = c + 1u) {
                let ce = cands[base + c];
                if (ce.support > 0u) {
                    n_pos = n_pos + 1u;
                    let term = llr(target_res, atom_idx, ce.val)
                        + params.support_bonus * (f32(ce.support) - 1.0);
                    if (term > best) {
                        best = term;
                    }
                }
            }
            if (n_pos > 0u) {
                total_pos_cands = total_pos_cands + n_pos;
                sc = sc + best - log(f32(n_pos));
            }
        }
    }

    if (params.forbid_zero_carbons != 0u && total_pos_cands == 0u) {
        out_u[out_idx] = FORBID;
        return;
    }

    if (params.use_hn != 0u) {
        let hn = roots_hn[s];
        sc = sc + llr(res, 1u, hn.y) + llr(res, 0u, hn.x);
    }
    sc = sc - params.assign_cost;
    out_u[out_idx] = sc;
}

@compute @workgroup_size(16, 16, 1)
fn compute_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let t = gid.y;
    if (s >= params.s_len || t >= params.s_len) {
        return;
    }
    let out_idx = s * params.s_len + t;
    if (s == t) {
        out_p[out_idx] = FORBID;
        return;
    }

    let sigmas = array<f32, 3>(params.sigma_ca, params.sigma_cb, params.sigma_c);
    let log_range = log(params.range_ppm);
    var total_p: f32 = 0.0;

    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let slot_a = (s * 3u + atom_c) * 2u + 0u;
        let slot_b = (t * 3u + atom_c) * 2u + 1u;
        let na = cand_counts[slot_a];
        let nb = cand_counts[slot_b];
        if (na == 0u || nb == 0u) {
            continue;
        }
        let sig = sigmas[atom_c];
        let const_term = -log(sig * sqrt(TAU)) + log_range;
        let base_a = slot_a * params.max_cands;
        let base_b = slot_b * params.max_cands;

        var best: f32 = -1e30;
        for (var ia: u32 = 0u; ia < na; ia = ia + 1u) {
            let va = cands[base_a + ia].val;
            for (var ib: u32 = 0u; ib < nb; ib = ib + 1u) {
                let vb = cands[base_b + ib].val;
                let d = (va - vb) / sig;
                let term = max(-0.5 * d * d + const_term, params.pair_floor);
                if (term > best) {
                    best = term;
                }
            }
        }
        let pen = log(f32(na * nb));
        total_p = total_p + (best - pen);
    }
    out_p[out_idx] = total_p;
}
