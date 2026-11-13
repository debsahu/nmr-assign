// Batched multi-conformer WebGPU torsion-space RDC energy & analytical gradient kernel.
//
// Provenance: Clean-room WGSL compute implementation of:
// - Tjandra, N. & Bax, A. (1997), Science 278, 1111-1114 (macromolecular RDC formulation).
// - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (1999), J. Magn. Reson.
//   138, 334-342 (Saupe alignment tensor back-calculation).
// - Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (2005),
//   J. Comput. Chem. 26, 1063-1068 (NeRF internal-to-Cartesian backbone placement).
// - Abe, H., Braun, W., Noguti, T. & Go, N. (1984), Comput. Chem. 8, 239-247 (O(L) backward
//   suffix-sum torque identity for exact torsion-angle gradients).

struct RdcRefineParams {
    n_conf: u32,
    l_res: u32,
    n_media: u32,
    n_restraints: u32,
    w_rdc: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct RdcTermGpu {
    medium_idx: u32,
    atom_a: u32,
    atom_b: u32,
    seg_a: u32,
    seg_b: u32,
    d_obs: f32,
    d_max: f32,
    inv_sig2: f32,
}

@group(0) @binding(0) var<uniform> params: RdcRefineParams;
@group(0) @binding(1) var<storage, read> res_flags: array<u32>;
@group(0) @binding(2) var<storage, read> torsions_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> saupe_mats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rdc_terms: array<RdcTermGpu>;
@group(0) @binding(5) var<storage, read_write> scratch: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> energies_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> grads_out: array<vec2<f32>>;

const BOND_N_CA: f32 = 1.458;
const BOND_CA_C: f32 = 1.525;
const BOND_C_N: f32 = 1.329;
const BOND_C_O: f32 = 1.231;
const BOND_N_H: f32 = 1.010;

const ANG_C_N_CA: f32 = 2.12406570;
const ANG_N_CA_C: f32 = 1.94080613;
const ANG_N_CA_C_GLY: f32 = 1.96349541;
const ANG_CA_C_N: f32 = 2.02807259;
const ANG_CA_C_O: f32 = 2.10835774;
const ANG_C_N_H: f32 = 2.08043247;

fn safe_unit(v: vec3<f32>) -> vec3<f32> {
    let n2 = dot(v, v);
    if (n2 > 1e-20) {
        return v * inverseSqrt(n2);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn nerf_place(
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    r_cd: f32,
    ang_bcd: f32,
    sin_ch: f32,
    cos_ch: f32,
) -> vec3<f32> {
    let ex = safe_unit(c - b);
    let ab_rev = a - b;
    let perp = ab_rev - ex * dot(ab_rev, ex);
    let ey = safe_unit(perp);
    let ez = cross(ex, ey);

    let sin_th = sin(ang_bcd);
    let cos_th = cos(ang_bcd);

    let dx = -r_cd * cos_th;
    let dy = r_cd * sin_th * cos_ch;
    let dz = r_cd * sin_th * sin_ch;

    return c + ex * dx + ey * dy + ez * dz;
}

// Atom slot offsets within residue `i`:
// 0 = N, 1 = H, 2 = CA, 3 = HA, 4 = C, 5 = O, 6 = CB
fn seg_pivot_pos(atom_base: u32, seg: u32) -> vec3<f32> {
    if (seg == 0u) {
        return scratch[atom_base + 0u].xyz;
    }
    if ((seg & 1u) == 1u) {
        let i = (seg - 1u) / 2u;
        return scratch[atom_base + 7u * i + 2u].xyz;
    }
    let i = (seg - 2u) / 2u;
    return scratch[atom_base + 7u * i + 4u].xyz;
}

@compute @workgroup_size(64)
fn eval_rdc_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let conf_idx = gid.x;
    if (conf_idx >= params.n_conf) {
        return;
    }

    let l = params.l_res;
    if (l == 0u) {
        energies_out[conf_idx] = 0.0;
        return;
    }

    let n_segs = 2u * l + 1u;
    let stride_scratch = 7u * l + 2u * n_segs;
    let atom_base = conf_idx * stride_scratch;
    let force_base = atom_base + 7u * l;
    let torque_base = force_base + n_segs;
    let tors_base = conf_idx * l;

    for (var s: u32 = 0u; s < n_segs; s = s + 1u) {
        scratch[force_base + s] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        scratch[torque_base + s] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // 1. Seed residue 0 in canonical Engh-Huber frame
    let flag0 = res_flags[0u];
    let ang_ca0 = select(ANG_N_CA_C, ANG_N_CA_C_GLY, flag0 == 1u);
    let c_prev = vec3<f32>(-BOND_C_N * cos(ANG_C_N_CA), BOND_C_N * sin(ANG_C_N_CA), 0.0);
    let n0 = vec3<f32>(0.0, 0.0, 0.0);
    let ca0 = vec3<f32>(BOND_N_CA, 0.0, 0.0);
    scratch[atom_base + 0u] = vec4<f32>(n0, 1.0);
    scratch[atom_base + 2u] = vec4<f32>(ca0, 1.0);
    if (flag0 != 2u) {
        let u1 = safe_unit(n0 - c_prev);
        let u2 = safe_unit(n0 - ca0);
        let h0 = n0 + safe_unit(u1 + u2) * BOND_N_H;
        scratch[atom_base + 1u] = vec4<f32>(h0, 1.0);
    } else {
        scratch[atom_base + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let t0 = torsions_in[tors_base + 0u];
    let c0 = nerf_place(c_prev, n0, ca0, BOND_CA_C, ang_ca0, sin(t0.x), cos(t0.x));
    scratch[atom_base + 4u] = vec4<f32>(c0, 1.0);

    // 2. Propagate NeRF backbone along residues 0 .. l-1
    for (var i: u32 = 0u; i < l; i = i + 1u) {
        let r_off = atom_base + 7u * i;
        let ni = scratch[r_off + 0u].xyz;
        let cai = scratch[r_off + 2u].xyz;
        let ci = scratch[r_off + 4u].xyz;
        let ti = torsions_in[tors_base + i];
        let psi_i = ti.y;
        let omg_i = ti.z;

        // O_i at torsion = psi_i + pi => sin = -sin(psi_i), cos = -cos(psi_i)
        let oi = nerf_place(ni, cai, ci, BOND_C_O, ANG_CA_C_O, -sin(psi_i), -cos(psi_i));
        scratch[r_off + 5u] = vec4<f32>(oi, 1.0);

        // Chiral L-amino acid HA / CB reconstruction matching `reconstruct_ca_substituents`
        let b1 = safe_unit(ni - cai);
        let b2 = safe_unit(ci - cai);
        let u_bis = safe_unit(b1 + b2);
        let nrm = safe_unit(cross(b1, b2));
        let dir_cb = safe_unit(u_bis * (-0.5810) + nrm * 0.8139);
        let dir_ha = safe_unit(u_bis * (-0.5810) + nrm * (-0.8139));
        let flg = res_flags[i];
        if (flg == 1u) {
            // Explicit tetrahedral Gly pro-R HA2 (slot +3) and pro-S HA3 (slot +6, unused CB slot on Gly)
            let ha2 = cai + dir_ha * 1.090;
            let ha3 = cai + dir_cb * 1.090;
            scratch[r_off + 3u] = vec4<f32>(ha2, 1.0);
            scratch[r_off + 6u] = vec4<f32>(ha3, 1.0);
        } else {
            scratch[r_off + 3u] = vec4<f32>(cai + dir_ha * 1.090, 1.0);
            scratch[r_off + 6u] = vec4<f32>(cai + dir_cb * 1.526, 1.0);
        }

        if (i + 1u < l) {
            let next_off = atom_base + 7u * (i + 1u);
            let next_flag = res_flags[i + 1u];
            let ang_ca_next = select(ANG_N_CA_C, ANG_N_CA_C_GLY, next_flag == 1u);

            let n_next = nerf_place(ni, cai, ci, BOND_C_N, ANG_CA_C_N, sin(psi_i), cos(psi_i));
            let ca_next = nerf_place(cai, ci, n_next, BOND_N_CA, ANG_C_N_CA, sin(omg_i), cos(omg_i));
            scratch[next_off + 0u] = vec4<f32>(n_next, 1.0);
            scratch[next_off + 2u] = vec4<f32>(ca_next, 1.0);

            if (next_flag != 2u) {
                let h_next = nerf_place(cai, ci, n_next, BOND_N_H, ANG_C_N_H, -sin(omg_i), -cos(omg_i));
                scratch[next_off + 1u] = vec4<f32>(h_next, 1.0);
            } else {
                scratch[next_off + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }

            let t_next = torsions_in[tors_base + i + 1u];
            let c_next = nerf_place(ci, n_next, ca_next, BOND_CA_C, ang_ca_next, sin(t_next.x), cos(t_next.x));
            scratch[next_off + 4u] = vec4<f32>(c_next, 1.0);
        }
    }

    // 3. Evaluate multi-medium harmonic RDC potential & accumulate rigid-segment forces/torques
    var total_e: f32 = 0.0;
    let w_rdc = params.w_rdc;
    if (w_rdc > 0.0) {
        let saupe_conf_base = conf_idx * params.n_media * 3u;
        for (var r_idx: u32 = 0u; r_idx < params.n_restraints; r_idx = r_idx + 1u) {
            let term = rdc_terms[r_idx];
            let a4 = scratch[atom_base + term.atom_a];
            let b4 = scratch[atom_base + term.atom_b];
            if (a4.w < 0.5 || b4.w < 0.5) {
                continue;
            }
            let pa = a4.xyz;
            let pb = b4.xyz;
            let dv = pb - pa;
            let d2 = dot(dv, dv);
            if (d2 < 1e-12) {
                continue;
            }
            let inv_r = inverseSqrt(d2);
            let u = dv * inv_r;

            let m_base = saupe_conf_base + term.medium_idx * 3u;
            let s0 = saupe_mats[m_base + 0u].xyz;
            let s1 = saupe_mats[m_base + 1u].xyz;
            let s2 = saupe_mats[m_base + 2u].xyz;

            let su = vec3<f32>(dot(s0, u), dot(s1, u), dot(s2, u));
            let ut_s_u = dot(u, su);
            let d_calc = term.d_max * ut_s_u;
            let resid = d_calc - term.d_obs;

            total_e = total_e + 0.5 * w_rdc * resid * resid * term.inv_sig2;

            let pref = w_rdc * resid * term.inv_sig2 * (2.0 * term.d_max * inv_r);
            let grad_pb = (su - u * ut_s_u) * pref;
            let grad_pa = -grad_pb;

            let piv_b = seg_pivot_pos(atom_base, term.seg_b);
            let fb = scratch[force_base + term.seg_b].xyz + grad_pb;
            let tb = scratch[torque_base + term.seg_b].xyz + cross(pb - piv_b, grad_pb);
            scratch[force_base + term.seg_b] = vec4<f32>(fb, 0.0);
            scratch[torque_base + term.seg_b] = vec4<f32>(tb, 0.0);

            let piv_a = seg_pivot_pos(atom_base, term.seg_a);
            let fa = scratch[force_base + term.seg_a].xyz + grad_pa;
            let ta = scratch[torque_base + term.seg_a].xyz + cross(pa - piv_a, grad_pa);
            scratch[force_base + term.seg_a] = vec4<f32>(fa, 0.0);
            scratch[torque_base + term.seg_a] = vec4<f32>(ta, 0.0);
        }
    }

    energies_out[conf_idx] = total_e;

    // 4. O(L) local-pivot backward suffix-sum torque propagation -> (grad_phi, grad_psi)
    var suf_f = vec3<f32>(0.0, 0.0, 0.0);
    var suf_t = vec3<f32>(0.0, 0.0, 0.0);
    var cur_q = scratch[atom_base + 7u * (l - 1u) + 4u].xyz;
    var i_rev: u32 = l;
    while (i_rev > 0u) {
        i_rev = i_rev - 1u;
        let r_off = atom_base + 7u * i_rev;
        let ni = scratch[r_off + 0u].xyz;
        let cai = scratch[r_off + 2u].xyz;
        let ci = scratch[r_off + 4u].xyz;

        // Shift downstream torque reference from cur_q to ci, then add local segment 2*i + 2
        suf_t = suf_t + cross(cur_q - ci, suf_f);
        cur_q = ci;
        let seg_psi = 2u * i_rev + 2u;
        suf_f = suf_f + scratch[force_base + seg_psi].xyz;
        suf_t = suf_t + scratch[torque_base + seg_psi].xyz;
        let axis_psi = safe_unit(ci - cai);
        let g_psi = dot(axis_psi, suf_t);

        // Shift downstream torque reference from ci to cai, then add local segment 2*i + 1
        suf_t = suf_t + cross(cur_q - cai, suf_f);
        cur_q = cai;
        let seg_phi = 2u * i_rev + 1u;
        suf_f = suf_f + scratch[force_base + seg_phi].xyz;
        suf_t = suf_t + scratch[torque_base + seg_phi].xyz;
        var g_phi: f32 = 0.0;
        if (i_rev > 0u) {
            let axis_phi = safe_unit(cai - ni);
            g_phi = dot(axis_phi, suf_t);
        }

        grads_out[tors_base + i_rev] = vec2<f32>(g_phi, g_psi);
    }
}
