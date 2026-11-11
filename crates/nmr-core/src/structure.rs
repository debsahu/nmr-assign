//! Clean-room torsion-angle polymer builder (NeRF), analytical torque-gradient simulated annealing
//! folding engine, Kabsch superposition, structure validation, and multi-model PDB ensemble writer.
//!
//! Provenance: Clean-room implementation from peer-reviewed structural biology and optimization literature:
//! - Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (2005). "Practical
//!   conversion from torsion space to Cartesian space for in silico protein synthesis."
//!   *J. Comput. Chem.* 26(10), 1063-1068 (Natural Extension Reference Frame, NeRF).
//! - Engh, R. A. & Huber, R. (1991). "Accurate bond and angle parameters for X-ray protein
//!   structure refinement." *Acta Crystallogr. A* 47(4), 392-400 (ideal backbone covalent geometry).
//! - Abe, H., Braun, W., Noguti, T. & Go, N. (1984). "Rapid calculation of first and second
//!   derivatives of conformational energy with respect to dihedral angles for proteins."
//!   *Comput. Chem.* 8(4), 239-247 (`O(L)` suffix-sum torque gradient in torsion space).
//! - Nilges, M. (1995). *J. Mol. Biol.* 245(5), 645-660 (soft-asymptote `r^-6` ambiguous NOE potential).
//! - Stein, E. G., Rice, L. M. & Brunger, A. T. (1997). "Torsion-angle molecular dynamics as a new
//!   efficient tool for NMR structure calculation." *J. Magn. Reson.* 124(1), 154-164.
//! - Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
//!   *Math. Comp.* 35(151), 773-782 (two-loop L-BFGS refinement).
//! - Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors."
//!   *Acta Crystallogr. A* 32(5), 922-923; and Coutsias, E. A., Seok, C. & Dill, K. A. (2004).
//!   "Using quaternions to calculate RMSD." *J. Comput. Chem.* 25(15), 1849-1857.
//! - Lovell, S. C. et al. (2003). "Structure validation by Calpha geometry: phi, psi and Cbeta
//!   deviation." *Proteins* 50(3), 437-450 (Ramachandran favored/allowed basins).

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use crate::dihedrals::{
    align_by_res_names, periodic_angle_abs_diff_deg, predict_dihedrals, wrap_deg,
    DihedralRestraint, SecondaryStructure,
};
use crate::nmrstar::Reference;
use crate::noesy::{
    match_initial_adrs, reconstruct_ca_substituents, reweight_adrs_with_coords, v_add, v_cross,
    v_dist_sq, v_dot, v_scale, v_sub, v_unit, AdrConfig, AmbiguousDistanceRestraint, NoeProton,
    PdbChain, PdbResidue, Vec3,
};
use crate::peaks::Peak;
use crate::rng::Rng;

// Engh & Huber (1991) standard covalent backbone bond lengths (Angstroms) and interior angles (degrees)
pub const BOND_N_CA: f64 = 1.458;
pub const BOND_CA_C: f64 = 1.525;
pub const BOND_C_N: f64 = 1.329;
pub const BOND_C_O: f64 = 1.231;
pub const BOND_N_H: f64 = 1.010;

pub const ANGLE_C_N_CA_DEG: f64 = 121.7;
pub const ANGLE_N_CA_C_DEG: f64 = 111.2;
pub const ANGLE_N_CA_C_GLY_DEG: f64 = 112.5;
pub const ANGLE_CA_C_N_DEG: f64 = 116.2;
pub const ANGLE_CA_C_O_DEG: f64 = 120.8;
pub const ANGLE_C_N_H_DEG: f64 = 119.2;

#[inline]
fn wrap_rad(rad: f64) -> f64 {
    let mut r = (rad + PI) % (2.0 * PI);
    if r <= 0.0 {
        r += 2.0 * PI;
    }
    r - PI
}

/// Natural Extension Reference Frame (Parsons et al., *J. Comput. Chem.* 2005):
/// Given atoms `a, b, c`, place atom `d` at bond length `r_cd = ||d - c||`, interior bond angle
/// `angle_bcd_rad = angle(b - c - d)`, and IUPAC right-handed torsion `torsion_abcd_rad = dihedral(a, b, c, d)`.
#[inline]
pub fn nerf_place_atom(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    r_cd: f64,
    angle_bcd_rad: f64,
    torsion_abcd_rad: f64,
) -> Vec3 {
    let ex = v_unit(v_sub(c, b));
    let ab_rev = v_sub(a, b);
    let perp = v_sub(ab_rev, v_scale(ex, v_dot(ab_rev, ex)));
    let ey = v_unit(perp);
    let ez = v_cross(ex, ey);

    let (sin_th, cos_th) = angle_bcd_rad.sin_cos();
    let (sin_ch, cos_ch) = torsion_abcd_rad.sin_cos();

    let dx = -r_cd * cos_th;
    let dy = r_cd * sin_th * cos_ch;
    let dz = r_cd * sin_th * sin_ch;

    [
        c[0] + dx * ex[0] + dy * ey[0] + dz * ez[0],
        c[1] + dx * ex[1] + dy * ey[1] + dz * ez[1],
        c[2] + dx * ex[2] + dy * ey[2] + dz * ez[2],
    ]
}

/// A single folded 3D protein conformer with backbone + `CB`/`QB` coordinates and energy breakdown.
#[derive(Clone, Debug)]
pub struct StructureConformer {
    pub chain: PdbChain,
    pub phi_deg: Vec<f64>,
    pub psi_deg: Vec<f64>,
    pub omega_deg: Vec<f64>,
    pub total_energy: f64,
    pub noe_energy: f64,
    pub dihedral_energy: f64,
    pub vdw_energy: f64,
    pub rama_energy: f64,
}

/// Construct an exact Engh-Huber backbone + `CB`/`HA`/`QB` polymer chain from `(phi, psi, omega)`
/// torsion angles in degrees via NeRF.
pub fn build_backbone_from_torsions(
    sequence: &[String],
    seq_ids: &[i32],
    phi_deg: &[f64],
    psi_deg: &[f64],
    omega_deg: &[f64],
) -> PdbChain {
    let l = sequence.len();
    assert_eq!(seq_ids.len(), l);
    assert_eq!(phi_deg.len(), l);
    assert_eq!(psi_deg.len(), l);
    assert_eq!(omega_deg.len(), l);

    let phi_rad: Vec<f64> = phi_deg.iter().map(|d| d.to_radians()).collect();
    let psi_rad: Vec<f64> = psi_deg.iter().map(|d| d.to_radians()).collect();
    let omg_rad: Vec<f64> = omega_deg.iter().map(|d| d.to_radians()).collect();

    build_backbone_from_torsions_rad(sequence, seq_ids, &phi_rad, &psi_rad, &omg_rad)
}

pub fn build_backbone_from_torsions_rad(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
) -> PdbChain {
    let l = sequence.len();
    if l == 0 {
        return PdbChain {
            residues: Vec::new(),
        };
    }

    let ang_c_n_ca = ANGLE_C_N_CA_DEG.to_radians();
    let ang_ca_c_n = ANGLE_CA_C_N_DEG.to_radians();
    let ang_ca_c_o = ANGLE_CA_C_O_DEG.to_radians();
    let ang_c_n_h = ANGLE_C_N_H_DEG.to_radians();

    let mut n_pos = vec![[0.0; 3]; l];
    let mut h_pos: Vec<Option<Vec3>> = vec![None; l];
    let mut ca_pos = vec![[0.0; 3]; l];
    let mut c_pos = vec![[0.0; 3]; l];
    let mut o_pos = vec![[0.0; 3]; l];

    // Seed residue 0 in canonical frame (virtual C_{-1} placed so phi_0 is well-defined if inspected)
    let ang_ca0 = if sequence[0].eq_ignore_ascii_case("GLY") {
        ANGLE_N_CA_C_GLY_DEG.to_radians()
    } else {
        ANGLE_N_CA_C_DEG.to_radians()
    };
    let c_prev_virtual = [
        -BOND_C_N * ang_c_n_ca.cos(),
        BOND_C_N * ang_c_n_ca.sin(),
        0.0,
    ];
    n_pos[0] = [0.0, 0.0, 0.0];
    ca_pos[0] = [BOND_N_CA, 0.0, 0.0];
    if !sequence[0].eq_ignore_ascii_case("PRO") {
        let u1 = v_unit(v_sub(n_pos[0], c_prev_virtual));
        let u2 = v_unit(v_sub(n_pos[0], ca_pos[0]));
        h_pos[0] = Some(v_add(n_pos[0], v_scale(v_unit(v_add(u1, u2)), BOND_N_H)));
    }
    c_pos[0] = nerf_place_atom(
        c_prev_virtual,
        n_pos[0],
        ca_pos[0],
        BOND_CA_C,
        ang_ca0,
        phi_rad[0],
    );

    for i in 0..l {
        // Carbonyl oxygen O_i rotates rigidly with psi_i around CA_i -> C_i
        o_pos[i] = nerf_place_atom(
            n_pos[i],
            ca_pos[i],
            c_pos[i],
            BOND_C_O,
            ang_ca_c_o,
            psi_rad[i] + PI,
        );

        if i + 1 < l {
            let next_is_gly = sequence[i + 1].eq_ignore_ascii_case("GLY");
            let next_is_pro = sequence[i + 1].eq_ignore_ascii_case("PRO");
            let ang_ca_next = if next_is_gly {
                ANGLE_N_CA_C_GLY_DEG.to_radians()
            } else {
                ANGLE_N_CA_C_DEG.to_radians()
            };

            n_pos[i + 1] = nerf_place_atom(
                n_pos[i], ca_pos[i], c_pos[i], BOND_C_N, ang_ca_c_n, psi_rad[i],
            );
            ca_pos[i + 1] = nerf_place_atom(
                ca_pos[i],
                c_pos[i],
                n_pos[i + 1],
                BOND_N_CA,
                ang_c_n_ca,
                omega_rad[i],
            );
            if !next_is_pro {
                h_pos[i + 1] = Some(nerf_place_atom(
                    ca_pos[i],
                    c_pos[i],
                    n_pos[i + 1],
                    BOND_N_H,
                    ang_c_n_h,
                    omega_rad[i] + PI,
                ));
            }
            c_pos[i + 1] = nerf_place_atom(
                c_pos[i],
                n_pos[i + 1],
                ca_pos[i + 1],
                BOND_CA_C,
                ang_ca_next,
                phi_rad[i + 1],
            );
        }
    }

    let mut residues = Vec::with_capacity(l);
    for i in 0..l {
        let (cb, ha, ha3, qb) =
            reconstruct_ca_substituents(&sequence[i], n_pos[i], ca_pos[i], c_pos[i]);
        residues.push(PdbResidue {
            seq_id: seq_ids[i],
            res_name: sequence[i].to_ascii_uppercase(),
            chain_id: 'A',
            n: n_pos[i],
            h: h_pos[i],
            ca: ca_pos[i],
            ha,
            ha3,
            c: c_pos[i],
            o: o_pos[i],
            cb,
            qb,
        });
    }

    PdbChain { residues }
}

/// Potential energy weights for a given simulated annealing or refinement stage.
#[derive(Clone, Copy, Debug)]
pub struct EnergyWeights {
    pub w_noe: f64,
    pub noe_soft_window: f64,
    pub w_dihedral: f64,
    pub w_vdw: f64,
    pub vdw_scale: f64,
    pub w_rama: f64,
    pub w_rg: f64,
}

impl Default for EnergyWeights {
    fn default() -> Self {
        Self {
            w_noe: 2.5,
            noe_soft_window: 3.0,
            w_dihedral: 28.0,
            w_vdw: 1.8,
            vdw_scale: 1.0,
            w_rama: 14.0,
            w_rg: 0.0,
        }
    }
}

/// Breakdown of conformational potential energy terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyBreakdown {
    pub total: f64,
    pub noe: f64,
    pub dihedral: f64,
    pub vdw: f64,
    pub rama: f64,
    pub rg: f64,
}

#[derive(Clone, Copy)]
enum ResClass {
    General,
    Gly,
    Pro,
}

/// Differentiable log-sum-exp Gaussian Ramachandran potential and gradient `(V, dV/dphi, dV/dpsi)`
/// parameterized after Lovell et al. (*Proteins* 2003) favored regions.
fn ramachandran_energy_and_grad(rc: ResClass, phi: f64, psi: f64) -> (f64, f64, f64) {
    // Each basin: (phi0_deg, psi0_deg, sig_phi_deg, sig_psi_deg, amp)
    const GEN_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-63.0, -42.0, 22.0, 22.0, 1.00),  // alpha_R
        (-122.0, 135.0, 28.0, 28.0, 1.00), // beta
        (-138.0, 168.0, 26.0, 24.0, 0.92), // twisted high-psi beta
        (-75.0, 145.0, 22.0, 24.0, 0.85),  // PPII
        (-85.0, -12.0, 20.0, 22.0, 0.40),  // gamma / bridge
        (56.0, 38.0, 16.0, 18.0, 0.25),    // alpha_L turn
    ];
    const PRO_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-63.0, 145.0, 13.0, 24.0, 1.00), // PPII Pro
        (-61.0, -38.0, 13.0, 22.0, 0.95), // alpha Pro
    ];
    const GLY_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-65.0, -41.0, 30.0, 30.0, 1.00),
        (65.0, 41.0, 30.0, 30.0, 1.00),
        (-80.0, 155.0, 34.0, 34.0, 0.95),
        (80.0, -155.0, 34.0, 34.0, 0.95),
        (82.0, 12.0, 30.0, 30.0, 0.90),
        (-82.0, -12.0, 30.0, 30.0, 0.90),
        (180.0, 180.0, 36.0, 36.0, 0.85),
    ];

    let basins = match rc {
        ResClass::General => GEN_BASINS,
        ResClass::Pro => PRO_BASINS,
        ResClass::Gly => GLY_BASINS,
    };

    // Stabilized log-sum-exp with tiny uniform floor so gradients stay smooth everywhere
    let floor_log = -9.0f64; // caps max basin penalty at ~9.0
    let mut max_log = floor_log;
    let mut logs = [0.0f64; 8];
    let mut dlog_dphi = [0.0f64; 8];
    let mut dlog_dpsi = [0.0f64; 8];

    for (bi, &(p0_d, s0_d, sp_d, ss_d, amp)) in basins.iter().enumerate() {
        let dphi = wrap_rad(phi - p0_d.to_radians());
        let dpsi = wrap_rad(psi - s0_d.to_radians());
        let inv_vp = 1.0 / (sp_d.to_radians().powi(2));
        let inv_vs = 1.0 / (ss_d.to_radians().powi(2));

        let lg = amp.ln() - 0.5 * (dphi * dphi * inv_vp + dpsi * dpsi * inv_vs);
        logs[bi] = lg;
        dlog_dphi[bi] = -dphi * inv_vp;
        dlog_dpsi[bi] = -dpsi * inv_vs;
        if lg > max_log {
            max_log = lg;
        }
    }

    let mut sum_exp = (floor_log - max_log).exp();
    let mut num_phi = 0.0;
    let mut num_psi = 0.0;

    for bi in 0..basins.len() {
        let w = (logs[bi] - max_log).exp();
        sum_exp += w;
        num_phi += w * dlog_dphi[bi];
        num_psi += w * dlog_dpsi[bi];
    }

    let log_p = max_log + sum_exp.ln();
    // Flat-bottom zero inside favored basin cores (-log_p <= 0.35) so alpha_R, beta, and PPII
    // carry zero relative energetic bias while strongly penalizing Ramachandran outliers!
    let raw_v = -log_p - 0.35;
    if raw_v <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (raw_v, -num_phi / sum_exp, -num_psi / sum_exp)
    }
}

/// Evaluate total conformational energy AND exact analytical `(dE/dphi_i, dE/dpsi_i)` torsion gradients
/// in `O(L + N_contacts)` time via the Abe, Braun, Noguti & Go (1984) suffix-sum torque identity.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_energy_and_gradients(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    weights: &EnergyWeights,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
) -> (PdbChain, EnergyBreakdown) {
    let l = sequence.len();
    grad_phi.fill(0.0);
    grad_psi.fill(0.0);

    let chain = build_backbone_from_torsions_rad(sequence, seq_ids, phi_rad, psi_rad, omega_rad);
    if l == 0 {
        return (chain, EnergyBreakdown::default());
    }

    // Rigid kinematic segments: 0 .. 2*l
    // Residue i:
    //   N_i, H_i, CA_i -> seg = 2*i (for i=0: seg 0)
    //   phi_i (torsion t = 2*i) rotates segments >= 2*i + 1 around axis unit(CA_i - N_i) at pivot CA_i
    //   C_i, CB_i, HA_i, QB_i -> seg = 2*i + 1
    //   psi_i (torsion t = 2*i + 1) rotates segments >= 2*i + 2 around axis unit(C_i - CA_i) at pivot C_i
    //   O_i -> seg = 2*i + 2 (same rigid link as N_{i+1}, H_{i+1}, CA_{i+1}!)
    let n_segs = 2 * l + 1;
    let mut seg_force = vec![[0.0f64; 3]; n_segs];
    let mut seg_torque = vec![[0.0f64; 3]; n_segs];

    let mut add_atom_grad = |seg: usize, pos: Vec3, g: Vec3| {
        seg_force[seg][0] += g[0];
        seg_force[seg][1] += g[1];
        seg_force[seg][2] += g[2];
        let tq = v_cross(pos, g);
        seg_torque[seg][0] += tq[0];
        seg_torque[seg][1] += tq[1];
        seg_torque[seg][2] += tq[2];
    };

    #[inline]
    fn noe_site_seg_and_pos(
        res: &PdbResidue,
        res_idx: usize,
        site: NoeProton,
    ) -> Option<(usize, Vec3)> {
        match site {
            NoeProton::AmideH => res.h.map(|p| (2 * res_idx, p)),
            NoeProton::AlphaHA => Some((2 * res_idx + 1, res.ha)),
            NoeProton::BetaQB => res.qb.map(|p| (2 * res_idx + 1, p)),
        }
    }

    let mut breakdown = EnergyBreakdown::default();

    // 1. Nilges (1995) r^-6 sum-averaged Ambiguous Distance Restraints
    if weights.w_noe > 0.0 && !adrs.is_empty() {
        let w_noe = weights.w_noe;
        let soft = weights.noe_soft_window;

        for adr in adrs {
            if adr.candidates.is_empty() {
                continue;
            }

            if adr.candidates.len() == 1 {
                let c = &adr.candidates[0];
                let (Some((seg_a, pa)), Some((seg_b, pb))) = (
                    noe_site_seg_and_pos(&chain.residues[c.res_a], c.res_a, c.atom_a),
                    noe_site_seg_and_pos(&chain.residues[c.res_b], c.res_b, c.atom_b),
                ) else {
                    continue;
                };
                let diff = v_sub(pa, pb);
                let d2 = v_dot(diff, diff) + 1e-6;
                let d = d2.sqrt();

                let (v_e, dv_dd) = if d < adr.lower_bound {
                    let u = adr.lower_bound - d;
                    (w_noe * u * u, -2.0 * w_noe * u)
                } else if d <= adr.upper_bound {
                    (0.0, 0.0)
                } else {
                    let u = d - adr.upper_bound;
                    if u <= soft {
                        (w_noe * u * u, 2.0 * w_noe * u)
                    } else {
                        (
                            w_noe * (soft * soft + 2.0 * soft * (u - soft)),
                            2.0 * w_noe * soft,
                        )
                    }
                };
                breakdown.noe += v_e;
                if dv_dd != 0.0 {
                    let g = v_scale(diff, dv_dd / d);
                    add_atom_grad(seg_a, pa, g);
                    add_atom_grad(seg_b, pb, [-g[0], -g[1], -g[2]]);
                }
            } else {
                // Multi-candidate Nilges r^-6 sum: D_eff = (sum_c d_c^-6)^(-1/6)
                let mut sum_inv_r6 = 0.0f64;
                for c in &adr.candidates {
                    if let (Some((_, pa)), Some((_, pb))) = (
                        noe_site_seg_and_pos(&chain.residues[c.res_a], c.res_a, c.atom_a),
                        noe_site_seg_and_pos(&chain.residues[c.res_b], c.res_b, c.atom_b),
                    ) {
                        let d2 = v_dist_sq(pa, pb) + 1e-4;
                        sum_inv_r6 += 1.0 / (d2 * d2 * d2);
                    }
                }
                if sum_inv_r6 <= 0.0 {
                    continue;
                }
                let d_eff = 1.0 / sum_inv_r6.cbrt().sqrt();

                let (v_e, dv_ddeff) = if d_eff < adr.lower_bound {
                    let u = adr.lower_bound - d_eff;
                    (w_noe * u * u, -2.0 * w_noe * u)
                } else if d_eff <= adr.upper_bound {
                    (0.0, 0.0)
                } else {
                    let u = d_eff - adr.upper_bound;
                    if u <= soft {
                        (w_noe * u * u, 2.0 * w_noe * u)
                    } else {
                        (
                            w_noe * (soft * soft + 2.0 * soft * (u - soft)),
                            2.0 * w_noe * soft,
                        )
                    }
                };
                breakdown.noe += v_e;
                if dv_ddeff != 0.0 {
                    let d_eff_7 = d_eff.powi(7);
                    let pref = dv_ddeff * d_eff_7;
                    for c in &adr.candidates {
                        if let (Some((seg_a, pa)), Some((seg_b, pb))) = (
                            noe_site_seg_and_pos(&chain.residues[c.res_a], c.res_a, c.atom_a),
                            noe_site_seg_and_pos(&chain.residues[c.res_b], c.res_b, c.atom_b),
                        ) {
                            let diff = v_sub(pa, pb);
                            let d2 = v_dot(diff, diff) + 1e-4;
                            let inv_d8 = 1.0 / ((d2 * d2) * (d2 * d2));
                            let g = v_scale(diff, pref * inv_d8);
                            add_atom_grad(seg_a, pa, g);
                            add_atom_grad(seg_b, pb, [-g[0], -g[1], -g[2]]);
                        }
                    }
                }
            }
        }
    }

    // 2. Quartic soft-core repulsive steric van der Waals clashes (with CA-CA bounding-sphere cull)
    if weights.w_vdw > 0.0 {
        let w_vdw = weights.w_vdw;
        let s = weights.vdw_scale;
        let s2 = s * s;

        // Atom kind: 0 = N, 1 = C/CA, 2 = O, 3 = CB
        #[inline]
        fn min_r2_base(ka: u8, kb: u8) -> f64 {
            match (ka.min(kb), ka.max(kb)) {
                (0, 2) => 2.65 * 2.65, // N..O hydrogen bond donor-acceptor allowance!
                (2, 2) => 2.80 * 2.80, // O..O
                (3, 3) => 3.25 * 3.25, // CB..CB hydrophobic contact
                (_, 3) => 3.00 * 3.00, // heavy..CB
                _ => 2.85 * 2.85,      // backbone heavy..heavy
            }
        }

        let max_ca_dist2 = (8.4 * s.max(0.75)).powi(2);

        for i in 0..l {
            let ri = &chain.residues[i];

            // Local i, i+1 Ramachandran steric guard: (O_i .. C_{i+1}), (O_i .. O_{i+1}), (O_i .. CB_{i+1}), (CB_i .. N_{i+1})
            if i + 1 < l {
                let rj = &chain.residues[i + 1];
                let local_pairs: &[(usize, Vec3, u8, usize, Vec3, u8)] = &[
                    (2 * i + 2, ri.o, 2, 2 * (i + 1) + 1, rj.c, 1),
                    (2 * i + 2, ri.o, 2, 2 * (i + 1) + 2, rj.o, 2),
                ];
                for &(sa, pa, ka, sb, pb, kb) in local_pairs {
                    let r2_min = s2 * min_r2_base(ka, kb);
                    let diff = v_sub(pa, pb);
                    let d2 = v_dot(diff, diff);
                    if d2 < r2_min {
                        let delta = r2_min - d2;
                        breakdown.vdw += w_vdw * delta * delta;
                        let g = v_scale(diff, -4.0 * w_vdw * delta);
                        add_atom_grad(sa, pa, g);
                        add_atom_grad(sb, pb, [-g[0], -g[1], -g[2]]);
                    }
                }
                if let Some(cb_j) = rj.cb {
                    let r2_min = s2 * (2.90 * 2.90);
                    let diff = v_sub(ri.o, cb_j);
                    let d2 = v_dot(diff, diff);
                    if d2 < r2_min {
                        let delta = r2_min - d2;
                        breakdown.vdw += w_vdw * delta * delta;
                        let g = v_scale(diff, -4.0 * w_vdw * delta);
                        add_atom_grad(2 * i + 2, ri.o, g);
                        add_atom_grad(2 * (i + 1) + 1, cb_j, [-g[0], -g[1], -g[2]]);
                    }
                }
            }

            // Non-local pairs j >= i + 2 with fast CA-CA sphere cull
            let mut atoms_i: [(usize, Vec3, u8); 5] = [
                (2 * i, ri.n, 0),
                (2 * i, ri.ca, 1),
                (2 * i + 1, ri.c, 1),
                (2 * i + 2, ri.o, 2),
                (2 * i + 1, [0.0; 3], 3),
            ];
            let ni = if let Some(cb) = ri.cb {
                atoms_i[4].1 = cb;
                5
            } else {
                4
            };

            for j in (i + 2)..l {
                let rj = &chain.residues[j];
                if v_dist_sq(ri.ca, rj.ca) > max_ca_dist2 {
                    continue;
                }
                let mut atoms_j: [(usize, Vec3, u8); 5] = [
                    (2 * j, rj.n, 0),
                    (2 * j, rj.ca, 1),
                    (2 * j + 1, rj.c, 1),
                    (2 * j + 2, rj.o, 2),
                    (2 * j + 1, [0.0; 3], 3),
                ];
                let nj = if let Some(cb) = rj.cb {
                    atoms_j[4].1 = cb;
                    5
                } else {
                    4
                };

                for &(sa, pa, ka) in &atoms_i[..ni] {
                    for &(sb, pb, kb) in &atoms_j[..nj] {
                        // Skip 1-4 peptide-plane neighbor (C_i / O_i .. N_{i+2}) if within normal angle
                        if j == i + 2 && ka == 1 && kb == 0 && sa == 2 * i + 1 {
                            continue;
                        }
                        let r2_min = s2 * min_r2_base(ka, kb);
                        let diff = v_sub(pa, pb);
                        let d2 = v_dot(diff, diff);
                        if d2 < r2_min {
                            let delta = r2_min - d2;
                            breakdown.vdw += w_vdw * delta * delta;
                            let g = v_scale(diff, -4.0 * w_vdw * delta);
                            add_atom_grad(sa, pa, g);
                            add_atom_grad(sb, pb, [-g[0], -g[1], -g[2]]);
                        }
                    }
                }
            }
        }
    }

    // 3. Flat-bottom Radius-of-Gyration compactness potential
    if weights.w_rg > 0.0 && l >= 6 {
        let lf = l as f64;
        let mut com = [0.0; 3];
        for r in &chain.residues {
            com = v_add(com, r.ca);
        }
        com = v_scale(com, 1.0 / lf);

        let mut rg2 = 0.0;
        for r in &chain.residues {
            rg2 += v_dist_sq(r.ca, com);
        }
        rg2 /= lf;

        let rg_target = 2.20 * lf.powf(0.38) + 1.5;
        let rg_max2 = rg_target * rg_target;
        if rg2 > rg_max2 {
            let ex = rg2 - rg_max2;
            breakdown.rg = weights.w_rg * ex * ex;
            let pref = 4.0 * weights.w_rg * ex / lf;
            for (i, r) in chain.residues.iter().enumerate() {
                let g = v_scale(v_sub(r.ca, com), pref);
                add_atom_grad(2 * i, r.ca, g);
            }
        }
    }

    // Backward suffix sum of forces and torques (Abe, Braun, Noguti & Go 1984)
    let mut suf_f = [0.0f64; 3];
    let mut suf_t = [0.0f64; 3];

    for t in (0..(2 * l)).rev() {
        let next_seg = t + 1;
        suf_f = v_add(suf_f, seg_force[next_seg]);
        suf_t = v_add(suf_t, seg_torque[next_seg]);

        let res_i = t / 2;
        if t % 2 == 1 {
            // Torsion t = 2*res_i + 1 is psi[res_i]: axis = unit(C_i - CA_i), pivot = C_i
            let r = &chain.residues[res_i];
            let axis = v_unit(v_sub(r.c, r.ca));
            let lever_t = v_sub(suf_t, v_cross(r.c, suf_f));
            grad_psi[res_i] += v_dot(axis, lever_t);
        } else {
            // Torsion t = 2*res_i is phi[res_i]: axis = unit(CA_i - N_i), pivot = CA_i
            let r = &chain.residues[res_i];
            let axis = v_unit(v_sub(r.ca, r.n));
            let lever_t = v_sub(suf_t, v_cross(r.ca, suf_f));
            grad_phi[res_i] += v_dot(axis, lever_t);
        }
    }

    // 4. Flat-bottom periodic (phi, psi) dihedral restraints (active only on confident secondary-structure / Pro torsions)
    if weights.w_dihedral > 0.0 && dihedrals.len() == l {
        let w_dih = weights.w_dihedral;
        for i in 0..l {
            let dr = &dihedrals[i];
            let wc = w_dih * dr.confidence;
            if i > 0 && dr.restrained_phi {
                let dphi = wrap_rad(phi_rad[i] - dr.phi_deg.to_radians());
                let tol = dr.phi_err_deg.to_radians();
                if dphi.abs() > tol {
                    let u = dphi.abs() - tol;
                    breakdown.dihedral += wc * u * u;
                    grad_phi[i] += 2.0 * wc * u * dphi.signum();
                }
            }
            if i + 1 < l && dr.restrained_psi {
                let dpsi = wrap_rad(psi_rad[i] - dr.psi_deg.to_radians());
                let tol = dr.psi_err_deg.to_radians();
                if dpsi.abs() > tol {
                    let u = dpsi.abs() - tol;
                    breakdown.dihedral += wc * u * u;
                    grad_psi[i] += 2.0 * wc * u * dpsi.signum();
                }
            }
        }
    }

    // 5. Smooth knowledge-based Ramachandran basin potential (internal residues 1 .. l-1)
    if weights.w_rama > 0.0 && l >= 3 {
        let w_r = weights.w_rama;
        for i in 1..(l - 1) {
            let rc = if sequence[i].eq_ignore_ascii_case("GLY") {
                ResClass::Gly
            } else if sequence[i].eq_ignore_ascii_case("PRO") {
                ResClass::Pro
            } else {
                ResClass::General
            };
            let (vr, gp, gs) = ramachandran_energy_and_grad(rc, phi_rad[i], psi_rad[i]);
            breakdown.rama += w_r * vr;
            grad_phi[i] += w_r * gp;
            grad_psi[i] += w_r * gs;
        }
    }

    // N-terminal phi_0 has no preceding residue C_{-1}; keep fixed
    grad_phi[0] = 0.0;

    breakdown.total =
        breakdown.noe + breakdown.dihedral + breakdown.vdw + breakdown.rama + breakdown.rg;

    (chain, breakdown)
}

/// Configuration for multi-replica torsion-angle simulated annealing and iterative ADR structure calculation.
#[derive(Clone, Debug)]
pub struct StructureCalcConfig {
    /// Number of independent parallel conformer replicas annealed per cycle (`default = 20`).
    pub n_replicas: usize,
    /// Number of lowest-energy conformers selected into the final bundle (`default = 10`).
    pub n_select: usize,
    /// Number of iterative ADR structure-filtering cycles after Cycle 0 (`default = 2`, total 3 folds).
    pub n_refinement_cycles: usize,
    /// Torsion Langevin / basin-hopping steps in soft-core high-T Stage 1.
    pub stage1_steps: usize,
    /// Torsion Langevin cooling steps in steric-inflation Stage 2.
    pub stage2_steps: usize,
    /// Two-loop L-BFGS quasi-Newton minimization iterations in Stage 3.
    pub lbfgs_iters: usize,
    pub adr_cfg: AdrConfig,
    pub seed: u64,
}

impl Default for StructureCalcConfig {
    fn default() -> Self {
        Self {
            n_replicas: 56,
            n_select: 10,
            n_refinement_cycles: 2,
            stage1_steps: 1500,
            stage2_steps: 800,
            lbfgs_iters: 320,
            adr_cfg: AdrConfig::default(),
            seed: 2026,
        }
    }
}

/// Two-loop L-BFGS quasi-Newton minimizer (Nocedal 1980) in `2*L` `(phi, psi)` torsion space.
#[allow(clippy::too_many_arguments)]
fn lbfgs_polish(
    sequence: &[String],
    seq_ids: &[i32],
    phi: &mut [f64],
    psi: &mut [f64],
    omega: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    weights: &EnergyWeights,
    max_iters: usize,
) -> (PdbChain, EnergyBreakdown) {
    let l = sequence.len();
    let dim = 2 * l;
    let m_hist = 7usize;

    let mut x = vec![0.0f64; dim];
    for i in 0..l {
        x[2 * i] = phi[i];
        x[2 * i + 1] = psi[i];
    }

    let mut gp = vec![0.0f64; l];
    let mut gs = vec![0.0f64; l];
    let (mut best_chain, mut cur_eb) = evaluate_energy_and_gradients(
        sequence,
        seq_ids,
        &x_phi(&x, l),
        &x_psi(&x, l),
        omega,
        adrs,
        dihedrals,
        weights,
        &mut gp,
        &mut gs,
    );

    let mut g = pack_grad(&gp, &gs);
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m_hist);

    let mut x_trial = vec![0.0f64; dim];
    let mut g_trial = vec![0.0f64; dim];

    for _iter in 0..max_iters {
        let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 1e-4 {
            break;
        }

        // Two-loop L-BFGS search direction d = -H_k * g
        let mut q = g.clone();
        let k_len = s_hist.len();
        let mut alpha = vec![0.0f64; k_len];
        for idx in (0..k_len).rev() {
            let a = rho_hist[idx] * dot_slice(&s_hist[idx], &q);
            alpha[idx] = a;
            for d_i in 0..dim {
                q[d_i] -= a * y_hist[idx][d_i];
            }
        }

        let gamma = if let (Some(sl), Some(yl)) = (s_hist.last(), y_hist.last()) {
            let yy = dot_slice(yl, yl);
            if yy > 1e-16 {
                (dot_slice(sl, yl) / yy).clamp(1e-4, 10.0)
            } else {
                0.05
            }
        } else {
            (0.15 / g_norm.max(1.0)).clamp(1e-3, 0.2)
        };

        let mut r_dir: Vec<f64> = q.into_iter().map(|v| v * gamma).collect();
        for idx in 0..k_len {
            let b = rho_hist[idx] * dot_slice(&y_hist[idx], &r_dir);
            let diff = alpha[idx] - b;
            for d_i in 0..dim {
                r_dir[d_i] += s_hist[idx][d_i] * diff;
            }
        }

        let mut dir: Vec<f64> = r_dir.into_iter().map(|v| -v).collect();
        dir[0] = 0.0; // phi_0 inactive

        let mut dg = dot_slice(&dir, &g);
        if dg >= 0.0 {
            for d_i in 0..dim {
                dir[d_i] = -g[d_i];
            }
            dir[0] = 0.0;
            dg = dot_slice(&dir, &g);
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        // Cap max single torsion step to 20 degrees (0.35 rad) for line-search stability
        let max_step = dir.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let mut step = if max_step > 0.35 {
            0.35 / max_step
        } else {
            1.0
        };

        let mut accepted = false;
        let mut next_eb = cur_eb;
        let mut next_chain = None;

        for _ls in 0..8 {
            for d_i in 0..dim {
                x_trial[d_i] = wrap_rad(x[d_i] + step * dir[d_i]);
            }
            let (ch_t, eb_t) = evaluate_energy_and_gradients(
                sequence,
                seq_ids,
                &x_phi(&x_trial, l),
                &x_psi(&x_trial, l),
                omega,
                adrs,
                dihedrals,
                weights,
                &mut gp,
                &mut gs,
            );
            if eb_t.total <= cur_eb.total + 1e-4 * step * dg {
                g_trial = pack_grad(&gp, &gs);
                next_eb = eb_t;
                next_chain = Some(ch_t);
                accepted = true;
                break;
            }
            step *= 0.45;
        }

        if !accepted {
            break;
        }

        let mut s_vec = vec![0.0f64; dim];
        let mut y_vec = vec![0.0f64; dim];
        for d_i in 0..dim {
            s_vec[d_i] = wrap_rad(x_trial[d_i] - x[d_i]);
            y_vec[d_i] = g_trial[d_i] - g[d_i];
        }
        let sy = dot_slice(&s_vec, &y_vec);
        if sy > 1e-12 {
            if s_hist.len() == m_hist {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        x.copy_from_slice(&x_trial);
        g.copy_from_slice(&g_trial);
        cur_eb = next_eb;
        if let Some(ch) = next_chain {
            best_chain = ch;
        }
    }

    for i in 0..l {
        phi[i] = x[2 * i];
        psi[i] = x[2 * i + 1];
    }

    (best_chain, cur_eb)
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn x_phi(x: &[f64], l: usize) -> Vec<f64> {
    (0..l).map(|i| x[2 * i]).collect()
}

#[inline]
fn x_psi(x: &[f64], l: usize) -> Vec<f64> {
    (0..l).map(|i| x[2 * i + 1]).collect()
}

#[inline]
fn pack_grad(gp: &[f64], gs: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * gp.len());
    for i in 0..gp.len() {
        out.push(gp[i]);
        out.push(gs[i]);
    }
    out
}

/// Anneal a single replica from randomized loop/turn torsions (or a perturbed prior conformer)
/// + chemical-shift secondary structure nucleation.
#[allow(clippy::too_many_arguments)]
fn anneal_single_replica(
    sequence: &[String],
    seq_ids: &[i32],
    local_adrs: &[AmbiguousDistanceRestraint],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    cfg: &StructureCalcConfig,
    warm_start: Option<(&[f64], &[f64])>,
    seed: u64,
) -> StructureConformer {
    let l = sequence.len();
    let mut rng = Rng::new(seed);

    let mut phi = vec![-1.10f64; l];
    let mut psi = vec![2.35f64; l];
    let omega = vec![PI; l];

    if let Some((w_phi_deg, w_psi_deg)) = warm_start {
        for i in 0..l {
            let jit = if i < dihedrals.len() && dihedrals[i].ss_class != SecondaryStructure::Coil {
                0.06
            } else {
                0.18
            };
            phi[i] = if sequence[i].eq_ignore_ascii_case("PRO") {
                wrap_rad(w_phi_deg[i].to_radians() + rng.normal(0.0, 0.05))
            } else {
                wrap_rad(w_phi_deg[i].to_radians() + rng.normal(0.0, jit))
            };
            psi[i] = wrap_rad(w_psi_deg[i].to_radians() + rng.normal(0.0, jit));
        }
    } else {
        // Initialize high-confidence Helix/Sheet cores near predicted torsions, and initialize Coil/Turn
        // linkers in diverse conformations so independent replicas sample all loop/bundle topologies
        for i in 0..l {
            if sequence[i].eq_ignore_ascii_case("PRO") {
                phi[i] = (-63.0f64).to_radians() + rng.normal(0.0, 0.12);
                psi[i] =
                    if i < dihedrals.len() && dihedrals[i].ss_class == SecondaryStructure::Helix {
                        (-40.0f64).to_radians() + rng.normal(0.0, 0.20)
                    } else {
                        (142.0f64).to_radians() + rng.normal(0.0, 0.30)
                    };
            } else if i < dihedrals.len()
                && dihedrals[i].ss_class != SecondaryStructure::Coil
                && dihedrals[i].confidence >= 0.60
            {
                let dr = &dihedrals[i];
                phi[i] = wrap_rad(dr.phi_deg.to_radians() + rng.normal(0.0, 0.18));
                psi[i] = wrap_rad(dr.psi_deg.to_radians() + rng.normal(0.0, 0.18));
            } else {
                let u = rng.f64();
                let is_gly = sequence[i].eq_ignore_ascii_case("GLY");
                if is_gly && u < 0.32 {
                    phi[i] = (80.0f64).to_radians() + rng.normal(0.0, 0.55);
                    psi[i] = (15.0f64).to_radians() + rng.normal(0.0, 0.55);
                } else if u < 0.68 {
                    phi[i] = (-105.0f64).to_radians() + rng.normal(0.0, 0.55);
                    psi[i] = (138.0f64).to_radians() + rng.normal(0.0, 0.50);
                } else {
                    phi[i] = (-66.0f64).to_radians() + rng.normal(0.0, 0.45);
                    psi[i] = (-35.0f64).to_radians() + rng.normal(0.0, 0.45);
                }
            }
        }
    }
    phi[0] = (-63.0f64).to_radians();

    let mut gp = vec![0.0f64; l];
    let mut gs = vec![0.0f64; l];
    let mut vel_phi = vec![0.0f64; l];
    let mut vel_psi = vec![0.0f64; l];

    // Stage 1: Soft-core high-T damped Langevin torsion dynamics
    // First 20% shapes local/medium-range turns & hairpins (|i - j| <= 5) cleanly before
    // ramping full long-range tertiary contacts through a ghost-chain soft core!
    let mut best_phi = phi.clone();
    let mut best_psi = psi.clone();
    let mut best_stage1_e = f64::INFINITY;

    let is_warm = warm_start.is_some();
    let s1 = if is_warm {
        (cfg.stage1_steps / 2).max(1)
    } else {
        cfg.stage1_steps.max(1)
    };
    let local_cutoff = if is_warm { 0.0 } else { 0.20 };
    let t_init = if is_warm { 3.5 } else { 14.0 };

    for step in 0..s1 {
        let prog = (step as f64) / (s1 as f64);
        let temp = t_init * (0.05f64).powf(prog);
        let active_adrs = if prog < local_cutoff && !local_adrs.is_empty() {
            local_adrs
        } else {
            adrs
        };
        let w = EnergyWeights {
            w_noe: if is_warm {
                2.0 + 0.6 * prog
            } else {
                1.2 + 1.5 * prog
            },
            noe_soft_window: if is_warm {
                5.0 - 1.5 * prog
            } else {
                7.5 - 3.5 * prog
            },
            w_dihedral: 22.0,
            w_vdw: if is_warm {
                0.04 + 0.12 * prog * prog
            } else {
                0.002 + 0.05 * prog * prog
            },
            vdw_scale: if is_warm {
                0.72 + 0.08 * prog
            } else {
                0.58 + 0.20 * prog
            },
            w_rama: if is_warm { 4.0 } else { 2.5 },
            w_rg: if prog < local_cutoff || is_warm {
                0.0
            } else {
                0.025 * (1.0 - 0.6 * prog)
            },
        };

        let (_, eb) = evaluate_energy_and_gradients(
            sequence,
            seq_ids,
            &phi,
            &psi,
            &omega,
            active_adrs,
            dihedrals,
            &w,
            &mut gp,
            &mut gs,
        );
        if prog > 0.72 && eb.total < best_stage1_e {
            best_stage1_e = eb.total;
            best_phi.copy_from_slice(&phi);
            best_psi.copy_from_slice(&psi);
        }

        // Clipped momentum + thermal Langevin step
        let g_norm = gp
            .iter()
            .chain(gs.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let lr = (0.105 * (1.0 - 0.45 * prog)) / (1.0 + 0.065 * g_norm);
        let noise_amp = (0.060 * (temp / 14.0).sqrt()).clamp(0.004, 0.075);
        let mom = 0.70;

        for i in 0..l {
            let flex = if i < dihedrals.len() && dihedrals[i].ss_class != SecondaryStructure::Coil {
                0.45
            } else {
                1.40
            };
            if i > 0 && !sequence[i].eq_ignore_ascii_case("PRO") {
                vel_phi[i] = mom * vel_phi[i] - lr * gp[i] + flex * rng.normal(0.0, noise_amp);
                phi[i] = wrap_rad(phi[i] + vel_phi[i].clamp(-0.26, 0.26));
            } else if i > 0 {
                vel_phi[i] = 0.3 * vel_phi[i] - 0.5 * lr * gp[i];
                phi[i] = wrap_rad(phi[i] + vel_phi[i].clamp(-0.05, 0.05));
            }
            vel_psi[i] = mom * vel_psi[i] - lr * gs[i] + flex * rng.normal(0.0, noise_amp);
            psi[i] = wrap_rad(psi[i] + vel_psi[i].clamp(-0.26, 0.26));
        }

        // Compensating loop pivot / crankshaft Metropolis jump during early Stage 1 (de novo search only)
        if !is_warm && step % 30 == 10 && prog >= local_cutoff && prog < 0.65 && l > 5 {
            let idx = rng.range_usize(1, l - 2);
            if idx < dihedrals.len()
                && dihedrals[idx].ss_class == SecondaryStructure::Coil
                && !sequence[idx].eq_ignore_ascii_case("PRO")
            {
                let old_p = phi[idx];
                let old_s = psi[idx];
                let old_p2 = phi[idx + 1];
                let (_, e_before) = evaluate_energy_and_gradients(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    active_adrs,
                    dihedrals,
                    &w,
                    &mut gp,
                    &mut gs,
                );
                let delta = rng.normal(0.0, 1.15);
                phi[idx] = wrap_rad(phi[idx] + rng.normal(0.0, 0.90));
                psi[idx] = wrap_rad(psi[idx] + delta);
                if !sequence[idx + 1].eq_ignore_ascii_case("PRO") {
                    phi[idx + 1] = wrap_rad(phi[idx + 1] - 0.85 * delta);
                }
                let (_, e_after) = evaluate_energy_and_gradients(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    active_adrs,
                    dihedrals,
                    &w,
                    &mut gp,
                    &mut gs,
                );
                let de = e_after.total - e_before.total;
                if de > 0.0 && rng.f64() >= (-de / temp.max(0.8)).exp() {
                    phi[idx] = old_p;
                    psi[idx] = old_s;
                    phi[idx + 1] = old_p2;
                }
            }
        }
    }

    if best_stage1_e.is_finite() {
        phi.copy_from_slice(&best_phi);
        psi.copy_from_slice(&best_psi);
    }

    // Soft-core L-BFGS pre-polish: snap secondary-structure elements into exact long-range NOE
    // packing while van der Waals radii are still soft (so topological crossings resolve cleanly!)
    let soft_polish_weights = EnergyWeights {
        w_noe: 2.6,
        noe_soft_window: 5.0,
        w_dihedral: 24.0,
        w_vdw: 0.08,
        vdw_scale: 0.74,
        w_rama: 5.0,
        w_rg: 0.0,
    };
    lbfgs_polish(
        sequence,
        seq_ids,
        &mut phi,
        &mut psi,
        &omega,
        adrs,
        dihedrals,
        &soft_polish_weights,
        90,
    );

    // Stage 2: Steric excluded-volume inflation & cooling
    vel_phi.fill(0.0);
    vel_psi.fill(0.0);
    let s2 = cfg.stage2_steps.max(1);
    for step in 0..s2 {
        let prog = (step as f64) / (s2 as f64);
        let w = EnergyWeights {
            w_noe: 2.5,
            noe_soft_window: 4.0 - 1.0 * prog,
            w_dihedral: 26.0 + 2.0 * prog,
            w_vdw: 0.12 + 1.68 * prog * prog,
            vdw_scale: 0.76 + 0.24 * prog,
            w_rama: 6.0 + 8.0 * prog,
            w_rg: 0.0,
        };

        evaluate_energy_and_gradients(
            sequence, seq_ids, &phi, &psi, &omega, adrs, dihedrals, &w, &mut gp, &mut gs,
        );

        let g_norm = gp
            .iter()
            .chain(gs.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let lr = (0.045 * (1.0 - 0.5 * prog)) / (1.0 + 0.10 * g_norm);
        let noise_amp = 0.012 * (1.0 - prog);

        for i in 0..l {
            if i > 0 {
                vel_phi[i] = 0.55 * vel_phi[i] - lr * gp[i] + rng.normal(0.0, noise_amp);
                phi[i] = wrap_rad(phi[i] + vel_phi[i].clamp(-0.10, 0.10));
            }
            vel_psi[i] = 0.55 * vel_psi[i] - lr * gs[i] + rng.normal(0.0, noise_amp);
            psi[i] = wrap_rad(psi[i] + vel_psi[i].clamp(-0.10, 0.10));
        }
    }

    // Stage 3: Full physical potential L-BFGS quasi-Newton minimization (zero Rg bias!)
    let final_weights = EnergyWeights::default();
    let (final_chain, final_eb) = lbfgs_polish(
        sequence,
        seq_ids,
        &mut phi,
        &mut psi,
        &omega,
        adrs,
        dihedrals,
        &final_weights,
        cfg.lbfgs_iters,
    );

    StructureConformer {
        chain: final_chain,
        phi_deg: phi.into_iter().map(|r| wrap_deg(r.to_degrees())).collect(),
        psi_deg: psi.into_iter().map(|r| wrap_deg(r.to_degrees())).collect(),
        omega_deg: omega
            .into_iter()
            .map(|r| wrap_deg(r.to_degrees()))
            .collect(),
        total_energy: final_eb.total,
        noe_energy: final_eb.noe,
        dihedral_energy: final_eb.dihedral,
        vdw_energy: final_eb.vdw,
        rama_energy: final_eb.rama,
    }
}

/// Fold a multi-replica conformer ensemble in parallel across CPU threads (`std::thread::scope`)
/// and return the `n_select` lowest-energy conformers sorted by ascending target restraint energy.
pub fn fold_ensemble(
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    cfg: &StructureCalcConfig,
    cycle_seed: u64,
) -> Vec<StructureConformer> {
    fold_ensemble_seeded(
        sequence,
        seq_ids,
        adrs,
        dihedrals,
        cfg,
        &[],
        0,
        1,
        cycle_seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn fold_ensemble_seeded(
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    cfg: &StructureCalcConfig,
    prior_conformers: &[StructureConformer],
    warm_fraction_num: usize,
    warm_fraction_den: usize,
    cycle_seed: u64,
) -> Vec<StructureConformer> {
    let n_rep = cfg.n_replicas.max(1);
    let mut master_rng = Rng::new(cycle_seed);
    let n_warm = if prior_conformers.is_empty() || warm_fraction_den == 0 {
        0
    } else {
        (n_rep * warm_fraction_num) / warm_fraction_den
    };
    let tasks: Vec<(usize, u64)> = (0..n_rep).map(|idx| (idx, master_rng.next_u64())).collect();

    // Short/medium-range subset (|i - j| <= 5) for early Stage 1 turn & hairpin nucleation
    let local_adrs: Vec<AmbiguousDistanceRestraint> = adrs
        .iter()
        .filter(|r| r.candidates.iter().any(|c| c.res_a.abs_diff(c.res_b) <= 5))
        .cloned()
        .collect();
    let local_ref = local_adrs.as_slice();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_rep);

    let mut all_conformers = Vec::with_capacity(n_rep);

    if n_threads <= 1 || n_rep <= 1 {
        for (idx, s) in tasks {
            let warm = if idx < n_warm {
                let pc = &prior_conformers[idx % prior_conformers.len()];
                Some((pc.phi_deg.as_slice(), pc.psi_deg.as_slice()))
            } else {
                None
            };
            all_conformers.push(anneal_single_replica(
                sequence, seq_ids, local_ref, adrs, dihedrals, cfg, warm, s,
            ));
        }
    } else {
        let chunk_len = n_rep.div_ceil(n_threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in tasks.chunks(chunk_len) {
                handles.push(scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&(idx, s)| {
                            let warm = if idx < n_warm {
                                let pc = &prior_conformers[idx % prior_conformers.len()];
                                Some((pc.phi_deg.as_slice(), pc.psi_deg.as_slice()))
                            } else {
                                None
                            };
                            anneal_single_replica(
                                sequence, seq_ids, local_ref, adrs, dihedrals, cfg, warm, s,
                            )
                        })
                        .collect::<Vec<_>>()
                }));
            }
            for h in handles {
                if let Ok(part) = h.join() {
                    all_conformers.extend(part);
                }
            }
        });
    }

    all_conformers.sort_by(|a, b| {
        let ta = a.noe_energy + a.dihedral_energy + a.vdw_energy;
        let tb = b.noe_energy + b.dihedral_energy + b.vdw_energy;
        ta.total_cmp(&tb)
    });
    all_conformers.truncate(cfg.n_select.max(1).min(all_conformers.len()));
    all_conformers
}

/// Complete output of iterative chemical-shift + 3D NOESY structure calculation.
#[derive(Clone, Debug)]
pub struct StructureEnsembleResult {
    pub conformers: Vec<StructureConformer>,
    pub dihedral_restraints: Vec<DihedralRestraint>,
    pub initial_unambiguous_adrs: usize,
    pub final_adrs: Vec<AmbiguousDistanceRestraint>,
}

/// Full end-to-end iterative Nilges (1995) ADR assignment + torsion-angle simulated annealing pipeline:
/// - Predicts `(phi, psi)` dihedral restraints from assigned chemical shifts,
/// - Cycle 0: Shift matching + covalent/reciprocal/triangle network anchoring -> folds initial ensemble,
/// - Cycles `1..K`: Re-weights candidate pairs by `<d_c^-6>` across current conformers, promoting
///   candidates with `>= 85%` weight to unambiguous restraints -> refolds final 10-conformer bundle!
pub fn calculate_structure_iterative(
    shifts: &Reference,
    noesy_peaks: &[Peak],
    cfg: &StructureCalcConfig,
) -> StructureEnsembleResult {
    let dihedral_restraints = predict_dihedrals(shifts);
    let mut adrs = match_initial_adrs(noesy_peaks, shifts, &cfg.adr_cfg);
    let initial_unambiguous_adrs = adrs
        .iter()
        .filter(|r| r.is_unambiguous || r.candidates.len() == 1)
        .count();

    let mut master = Rng::new(cfg.seed);

    // Cycle 0 fold (slightly faster intermediate budget if followed by refinement cycles)
    let mut cycle0_cfg = cfg.clone();
    if cfg.n_refinement_cycles > 0 {
        cycle0_cfg.lbfgs_iters = (cfg.lbfgs_iters * 2 / 3).max(120);
    }
    let mut ensemble = fold_ensemble_seeded(
        &shifts.sequence,
        &shifts.seq_ids,
        &adrs,
        &dihedral_restraints,
        &cycle0_cfg,
        &[],
        0,
        1,
        master.next_u64(),
    );

    // Iterative Cycles 1..=K: structure-guided <d^-6> ADR reweighting on lowest-energy conformers + refolding
    for cyc in 0..cfg.n_refinement_cycles {
        let is_last = cyc + 1 == cfg.n_refinement_cycles;
        let min_keep = if is_last { 0.07 } else { 0.035 };
        let cum_target = if is_last { 0.90 } else { 0.95 };
        let n_guide = if is_last {
            ensemble.len().clamp(1, 4)
        } else {
            ensemble.len().clamp(1, 5)
        };
        reweight_adrs_with_coords(
            &mut adrs,
            n_guide,
            min_keep,
            cum_target,
            cfg.adr_cfg.unambiguous_threshold,
            |m, res_idx, site| ensemble[m].chain.residues.get(res_idx)?.noe_coord(site),
        );

        let cyc_cfg = if is_last {
            cfg.clone()
        } else {
            cycle0_cfg.clone()
        };
        let n_seed_models = if is_last { 2 } else { 3 };
        let (wf_num, wf_den) = if is_last { (3, 4) } else { (1, 2) };
        let top_seeds = &ensemble[..ensemble.len().min(n_seed_models)];
        ensemble = fold_ensemble_seeded(
            &shifts.sequence,
            &shifts.seq_ids,
            &adrs,
            &dihedral_restraints,
            &cyc_cfg,
            top_seeds,
            wf_num,
            wf_den,
            master.next_u64(),
        );
    }

    StructureEnsembleResult {
        conformers: ensemble,
        dihedral_restraints,
        initial_unambiguous_adrs,
        final_adrs: adrs,
    }
}

// ============================================================================
// Kabsch Superposition (Quaternion Eigensystem, Kabsch 1976 / Coutsias et al. 2004)
// ============================================================================

/// Exact Jacobi diagonalization of a real symmetric 4x4 matrix, returning the unit eigenvector
/// corresponding to the largest eigenvalue.
#[allow(clippy::needless_range_loop)]
fn max_eigenvector_sym4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for _sweep in 0..25 {
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if a[i][j].abs() > max_off {
                    max_off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-15 {
            break;
        }

        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-30 {
            1.0
        } else {
            let phi = diff / (2.0 * a[p][q]);
            let sgn = if phi >= 0.0 { 1.0 } else { -1.0 };
            sgn / (phi.abs() + (1.0 + phi * phi).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..4 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for r in 0..4 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    let mut best_col = 0;
    for col in 1..4 {
        if a[col][col] > a[best_col][best_col] {
            best_col = col;
        }
    }
    [
        v[0][best_col],
        v[1][best_col],
        v[2][best_col],
        v[3][best_col],
    ]
}

/// Optimal proper rigid-body rotation `R` (`det(R) = +1`), centroids `(c_mov, c_ref)`, and Kabsch RMSD
/// aligning `moving` onto `reference` (`x_aligned = R * (x - c_mov) + c_ref`).
pub fn kabsch_transform(moving: &[Vec3], reference: &[Vec3]) -> ([[f64; 3]; 3], Vec3, Vec3, f64) {
    assert_eq!(moving.len(), reference.len());
    let n = moving.len();
    if n == 0 {
        return (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            [0.0; 3],
            0.0,
        );
    }

    let nf = n as f64;
    let mut c_mov = [0.0; 3];
    let mut c_ref = [0.0; 3];
    for i in 0..n {
        c_mov = v_add(c_mov, moving[i]);
        c_ref = v_add(c_ref, reference[i]);
    }
    c_mov = v_scale(c_mov, 1.0 / nf);
    c_ref = v_scale(c_ref, 1.0 / nf);

    let mut cov = [[0.0f64; 3]; 3];
    for i in 0..n {
        let pm = v_sub(moving[i], c_mov);
        let pr = v_sub(reference[i], c_ref);
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += pm[r] * pr[c];
            }
        }
    }

    // Coutsias, Seok & Dill (2004) symmetric 4x4 quaternion matrix F
    let (rxx, rxy, rxz) = (cov[0][0], cov[0][1], cov[0][2]);
    let (ryx, ryy, ryz) = (cov[1][0], cov[1][1], cov[1][2]);
    let (rzx, rzy, rzz) = (cov[2][0], cov[2][1], cov[2][2]);

    let f = [
        [rxx + ryy + rzz, ryz - rzy, rzx - rxz, rxy - ryx],
        [ryz - rzy, rxx - ryy - rzz, rxy + ryx, rxz + rzx],
        [rzx - rxz, rxy + ryx, -rxx + ryy - rzz, ryz + rzy],
        [rxy - ryx, rxz + rzx, ryz + rzy, -rxx - ryy + rzz],
    ];

    let q = max_eigenvector_sym4(f);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);

    let rot = [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q2 * q1 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q3 * q1 - q0 * q2),
            2.0 * (q3 * q2 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ];

    let mut sum_sq = 0.0;
    for i in 0..n {
        let pm = v_sub(moving[i], c_mov);
        let pr = v_sub(reference[i], c_ref);
        let aligned = [
            rot[0][0] * pm[0] + rot[0][1] * pm[1] + rot[0][2] * pm[2],
            rot[1][0] * pm[0] + rot[1][1] * pm[1] + rot[1][2] * pm[2],
            rot[2][0] * pm[0] + rot[2][1] * pm[1] + rot[2][2] * pm[2],
        ];
        sum_sq += v_dist_sq(aligned, pr);
    }

    let rmsd = (sum_sq / nf).max(0.0).sqrt();
    (rot, c_mov, c_ref, rmsd)
}

/// Optimal proper rigid-body superposition RMSD (Angstroms) between two equal-length 3D coordinate sets.
pub fn kabsch_rmsd(moving: &[Vec3], reference: &[Vec3]) -> f64 {
    kabsch_transform(moving, reference).3
}

/// Superpose a `PdbChain` in-place onto a `target` `PdbChain` using `N, CA, C` atoms of the aligned residues.
pub fn superpose_chain_onto(
    moving: &mut PdbChain,
    target: &PdbChain,
    ordered_mask: Option<&[bool]>,
) -> f64 {
    let m_keys: Vec<(i32, &str)> = moving
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let t_keys: Vec<(i32, &str)> = target
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let pairs = align_by_res_names(&m_keys, &t_keys);

    let mut pts_m = Vec::with_capacity(pairs.len() * 3);
    let mut pts_t = Vec::with_capacity(pairs.len() * 3);

    for &(mi, ti) in &pairs {
        if let Some(mask) = ordered_mask {
            if mi < mask.len() && !mask[mi] {
                continue;
            }
        }
        pts_m.extend([
            moving.residues[mi].n,
            moving.residues[mi].ca,
            moving.residues[mi].c,
        ]);
        pts_t.extend([
            target.residues[ti].n,
            target.residues[ti].ca,
            target.residues[ti].c,
        ]);
    }
    if pts_m.len() < 9 {
        pts_m.clear();
        pts_t.clear();
        for &(mi, ti) in &pairs {
            pts_m.extend([
                moving.residues[mi].n,
                moving.residues[mi].ca,
                moving.residues[mi].c,
            ]);
            pts_t.extend([
                target.residues[ti].n,
                target.residues[ti].ca,
                target.residues[ti].c,
            ]);
        }
    }
    if pts_m.is_empty() {
        return 0.0;
    }

    let (rot, c_mov, c_ref, rmsd) = kabsch_transform(&pts_m, &pts_t);
    let xform = |p: Vec3| -> Vec3 {
        let d = v_sub(p, c_mov);
        [
            rot[0][0] * d[0] + rot[0][1] * d[1] + rot[0][2] * d[2] + c_ref[0],
            rot[1][0] * d[0] + rot[1][1] * d[1] + rot[1][2] * d[2] + c_ref[1],
            rot[2][0] * d[0] + rot[2][1] * d[1] + rot[2][2] * d[2] + c_ref[2],
        ]
    };

    for r in &mut moving.residues {
        r.n = xform(r.n);
        r.h = r.h.map(xform);
        r.ca = xform(r.ca);
        r.ha = xform(r.ha);
        r.ha3 = r.ha3.map(xform);
        r.c = xform(r.c);
        r.o = xform(r.o);
        r.cb = r.cb.map(xform);
        r.qb = r.qb.map(xform);
    }

    rmsd
}

/// Backbone (`N, CA, C`) RMSD between two `PdbChain`s over aligned residues (`ordered_only` selects
/// residues marked `true` in `ordered_mask`).
pub fn backbone_rmsd_between_chains(
    chain_a: &PdbChain,
    chain_b: &PdbChain,
    ordered_mask_a: Option<&[bool]>,
) -> f64 {
    let a_keys: Vec<(i32, &str)> = chain_a
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let b_keys: Vec<(i32, &str)> = chain_b
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let pairs = align_by_res_names(&a_keys, &b_keys);

    let mut pts_a = Vec::with_capacity(pairs.len() * 3);
    let mut pts_b = Vec::with_capacity(pairs.len() * 3);

    for &(ai, bi) in &pairs {
        if let Some(mask) = ordered_mask_a {
            if ai < mask.len() && !mask[ai] {
                continue;
            }
        }
        pts_a.extend([
            chain_a.residues[ai].n,
            chain_a.residues[ai].ca,
            chain_a.residues[ai].c,
        ]);
        pts_b.extend([
            chain_b.residues[bi].n,
            chain_b.residues[bi].ca,
            chain_b.residues[bi].c,
        ]);
    }
    if pts_a.is_empty() {
        return 0.0;
    }
    kabsch_rmsd(&pts_a, &pts_b)
}

/// Ramachandran quality classification per Lovell et al. (*Proteins* 2003).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamachandranClass {
    Favored,
    Allowed,
    Outlier,
}

/// Classify a residue's `(phi_deg, psi_deg)` into `Favored`, `Allowed`, or `Outlier`.
pub fn classify_ramachandran(res_name: &str, phi_deg: f64, psi_deg: f64) -> RamachandranClass {
    let up = res_name.trim().to_ascii_uppercase();
    let rc = if up == "GLY" {
        ResClass::Gly
    } else if up == "PRO" {
        ResClass::Pro
    } else {
        ResClass::General
    };
    let (v, _, _) = ramachandran_energy_and_grad(rc, phi_deg.to_radians(), psi_deg.to_radians());
    // v = -ln(mixture) where peak ~ 0.0, 2-sigma contour ~ 2.0, 3.2-sigma allowed contour ~ 4.8
    if v <= 2.35 {
        RamachandranClass::Favored
    } else if v <= 5.10 {
        RamachandranClass::Allowed
    } else {
        RamachandranClass::Outlier
    }
}

/// Validation report for a folded NMR structure ensemble.
#[derive(Clone, Debug)]
pub struct EnsembleValidationReport {
    pub n_conformers: usize,
    pub n_residues: usize,
    pub n_ordered_residues: usize,
    pub ensemble_pairwise_rmsd_ordered: f64,
    pub ensemble_pairwise_rmsd_all: f64,
    pub mean_backbone_rmsd_ordered_to_ref: Option<f64>,
    pub lowest_energy_rmsd_ordered_to_ref: Option<f64>,
    pub best_conformer_rmsd_ordered_to_ref: Option<f64>,
    pub mean_backbone_rmsd_all_to_ref: Option<f64>,
    pub ramachandran_favored_pct: f64,
    pub ramachandran_allowed_pct: f64,
    pub ramachandran_outlier_pct: f64,
    pub mean_distance_violations_gt_0_5a: f64,
    pub max_distance_violation_a: f64,
    pub mean_dihedral_violations_gt_5deg: f64,
}

/// Compute comprehensive ensemble precision, reference PDB accuracy, Ramachandran quality, and
/// distance/dihedral restraint violation statistics.
pub fn validate_ensemble(
    conformers: &[StructureConformer],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    reference_pdb: Option<&PdbChain>,
) -> EnsembleValidationReport {
    let m = conformers.len();
    let l = conformers
        .first()
        .map(|c| c.chain.residues.len())
        .unwrap_or(0);

    let ordered_mask: Vec<bool> = (0..l)
        .map(|i| {
            i < dihedrals.len()
                && dihedrals[i].ss_class != SecondaryStructure::Coil
                && i > 1
                && i + 2 < l
        })
        .collect();
    let n_ordered = ordered_mask.iter().filter(|&&b| b).count();
    let mask_opt = if n_ordered >= 6 {
        Some(ordered_mask.as_slice())
    } else {
        None
    };

    // Ensemble pairwise RMSDs
    let mut pw_ord_sum = 0.0;
    let mut pw_all_sum = 0.0;
    let mut pw_cnt = 0usize;
    for a in 0..m {
        for b in (a + 1)..m {
            pw_ord_sum +=
                backbone_rmsd_between_chains(&conformers[a].chain, &conformers[b].chain, mask_opt);
            pw_all_sum +=
                backbone_rmsd_between_chains(&conformers[a].chain, &conformers[b].chain, None);
            pw_cnt += 1;
        }
    }
    let pw_denom = (pw_cnt.max(1)) as f64;

    // Reference PDB RMSDs
    let (mean_ord_ref, low_e_ord_ref, best_ord_ref, mean_all_ref) =
        if let Some(ref_pdb) = reference_pdb {
            if m > 0 {
                let mut sum_ord = 0.0;
                let mut sum_all = 0.0;
                let mut best_ord = f64::INFINITY;
                let mut low_e_ord = 0.0;
                for (ci, conf) in conformers.iter().enumerate() {
                    let r_ord = backbone_rmsd_between_chains(&conf.chain, ref_pdb, mask_opt);
                    let r_all = backbone_rmsd_between_chains(&conf.chain, ref_pdb, None);
                    if ci == 0 {
                        low_e_ord = r_ord;
                    }
                    if r_ord < best_ord {
                        best_ord = r_ord;
                    }
                    sum_ord += r_ord;
                    sum_all += r_all;
                }
                (
                    Some(sum_ord / (m as f64)),
                    Some(low_e_ord),
                    Some(best_ord),
                    Some(sum_all / (m as f64)),
                )
            } else {
                (None, None, None, None)
            }
        } else {
            (None, None, None, None)
        };

    // Ramachandran statistics across internal residues 1 .. l-1 of all conformers
    let mut n_fav = 0usize;
    let mut n_allw = 0usize;
    let mut n_out = 0usize;
    for conf in conformers {
        for i in 1..l.saturating_sub(1) {
            match classify_ramachandran(
                &conf.chain.residues[i].res_name,
                conf.phi_deg[i],
                conf.psi_deg[i],
            ) {
                RamachandranClass::Favored => n_fav += 1,
                RamachandranClass::Allowed => n_allw += 1,
                RamachandranClass::Outlier => n_out += 1,
            }
        }
    }
    let rama_tot = ((n_fav + n_allw + n_out).max(1)) as f64;

    // Distance restraint violations (> 0.5 A above upper_bound or below lower_bound)
    let mut dist_viol_total = 0usize;
    let mut max_dist_viol = 0.0f64;
    for conf in conformers {
        for adr in adrs {
            if adr.candidates.is_empty() {
                continue;
            }
            let mut sum_inv_r6 = 0.0f64;
            for c in &adr.candidates {
                if let (Some(pa), Some(pb)) = (
                    conf.chain
                        .residues
                        .get(c.res_a)
                        .and_then(|r| r.noe_coord(c.atom_a)),
                    conf.chain
                        .residues
                        .get(c.res_b)
                        .and_then(|r| r.noe_coord(c.atom_b)),
                ) {
                    let d2 = v_dist_sq(pa, pb).max(1e-4);
                    sum_inv_r6 += 1.0 / (d2 * d2 * d2);
                }
            }
            if sum_inv_r6 <= 0.0 {
                continue;
            }
            let d_eff = 1.0 / sum_inv_r6.cbrt().sqrt();
            let viol = if d_eff > adr.upper_bound {
                d_eff - adr.upper_bound
            } else if d_eff < adr.lower_bound {
                adr.lower_bound - d_eff
            } else {
                0.0
            };
            if viol > max_dist_viol {
                max_dist_viol = viol;
            }
            if viol > 0.50 {
                dist_viol_total += 1;
            }
        }
    }

    // Dihedral restraint violations (> 5.0 deg outside target +/- err on active dihedral restraints)
    let mut dih_viol_total = 0usize;
    for conf in conformers {
        for (i, dr) in dihedrals.iter().enumerate() {
            if i == 0 || i + 1 >= l {
                continue;
            }
            if dr.restrained_phi {
                let v_phi = (periodic_angle_abs_diff_deg(conf.phi_deg[i], dr.phi_deg)
                    - dr.phi_err_deg)
                    .max(0.0);
                if v_phi > 5.0 {
                    dih_viol_total += 1;
                }
            }
            if dr.restrained_psi {
                let v_psi = (periodic_angle_abs_diff_deg(conf.psi_deg[i], dr.psi_deg)
                    - dr.psi_err_deg)
                    .max(0.0);
                if v_psi > 5.0 {
                    dih_viol_total += 1;
                }
            }
        }
    }

    let mf = (m.max(1)) as f64;
    EnsembleValidationReport {
        n_conformers: m,
        n_residues: l,
        n_ordered_residues: n_ordered,
        ensemble_pairwise_rmsd_ordered: pw_ord_sum / pw_denom,
        ensemble_pairwise_rmsd_all: pw_all_sum / pw_denom,
        mean_backbone_rmsd_ordered_to_ref: mean_ord_ref,
        lowest_energy_rmsd_ordered_to_ref: low_e_ord_ref,
        best_conformer_rmsd_ordered_to_ref: best_ord_ref,
        mean_backbone_rmsd_all_to_ref: mean_all_ref,
        ramachandran_favored_pct: 100.0 * (n_fav as f64) / rama_tot,
        ramachandran_allowed_pct: 100.0 * (n_allw as f64) / rama_tot,
        ramachandran_outlier_pct: 100.0 * (n_out as f64) / rama_tot,
        mean_distance_violations_gt_0_5a: (dist_viol_total as f64) / mf,
        max_distance_violation_a: max_dist_viol,
        mean_dihedral_violations_gt_5deg: (dih_viol_total as f64) / mf,
    }
}

/// Format a superposed multi-model NMR structure ensemble in standard wwPDB `MODEL ... ENDMDL` format.
pub fn format_pdb_ensemble(conformers: &[StructureConformer], title: &str) -> String {
    if conformers.is_empty() {
        return "END\n".to_string();
    }

    // Superpose all conformers onto the lowest-energy conformer (model 1) for clean bundle visualization
    let ref_chain = &conformers[0].chain;
    let mut aligned_chains: Vec<PdbChain> = conformers.iter().map(|c| c.chain.clone()).collect();
    for ch in aligned_chains.iter_mut().skip(1) {
        superpose_chain_onto(ch, ref_chain, None);
    }

    let mut out = String::with_capacity(conformers.len() * ref_chain.residues.len() * 520);
    out.push_str(&format!("TITLE     {title}\n"));
    out.push_str(&format!(
        "REMARK   1 GENERATED BY NMR-CORE CLEAN-ROOM TORSION-ANGLE ENSEMBLE ENGINE ({} MODELS)\n",
        conformers.len()
    ));

    for (mi, (conf, ch)) in conformers.iter().zip(aligned_chains.iter()).enumerate() {
        out.push_str(&format!("MODEL     {:>4}\n", mi + 1));
        out.push_str(&format!(
            "REMARK   2 ENERGY TOTAL={:.2} NOE={:.2} DIHED={:.2} VDW={:.2} RAMA={:.2}\n",
            conf.total_energy,
            conf.noe_energy,
            conf.dihedral_energy,
            conf.vdw_energy,
            conf.rama_energy
        ));

        let mut serial = 1usize;
        let emit_atom = |buf: &mut String,
                         ser: &mut usize,
                         at_name: &str,
                         res_name: &str,
                         chain_id: char,
                         seq_id: i32,
                         xyz: Vec3,
                         elem: &str| {
            let name_col = if at_name.len() < 4 {
                format!(" {at_name:<3}")
            } else {
                format!("{at_name:<4}")
            };
            buf.push_str(&format!(
                "ATOM  {:>5} {} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
                *ser, name_col, res_name, chain_id, seq_id, xyz[0], xyz[1], xyz[2], elem
            ));
            *ser += 1;
        };

        for r in &ch.residues {
            emit_atom(
                &mut out,
                &mut serial,
                "N",
                &r.res_name,
                r.chain_id,
                r.seq_id,
                r.n,
                "N",
            );
            if let Some(h) = r.h {
                emit_atom(
                    &mut out,
                    &mut serial,
                    "H",
                    &r.res_name,
                    r.chain_id,
                    r.seq_id,
                    h,
                    "H",
                );
            }
            emit_atom(
                &mut out,
                &mut serial,
                "CA",
                &r.res_name,
                r.chain_id,
                r.seq_id,
                r.ca,
                "C",
            );
            if r.res_name == "GLY" {
                emit_atom(
                    &mut out,
                    &mut serial,
                    "HA2",
                    &r.res_name,
                    r.chain_id,
                    r.seq_id,
                    r.ha,
                    "H",
                );
                if let Some(ha3) = r.ha3 {
                    emit_atom(
                        &mut out,
                        &mut serial,
                        "HA3",
                        &r.res_name,
                        r.chain_id,
                        r.seq_id,
                        ha3,
                        "H",
                    );
                }
            } else {
                emit_atom(
                    &mut out,
                    &mut serial,
                    "HA",
                    &r.res_name,
                    r.chain_id,
                    r.seq_id,
                    r.ha,
                    "H",
                );
            }
            if let Some(cb) = r.cb {
                emit_atom(
                    &mut out,
                    &mut serial,
                    "CB",
                    &r.res_name,
                    r.chain_id,
                    r.seq_id,
                    cb,
                    "C",
                );
            }
            emit_atom(
                &mut out,
                &mut serial,
                "C",
                &r.res_name,
                r.chain_id,
                r.seq_id,
                r.c,
                "C",
            );
            emit_atom(
                &mut out,
                &mut serial,
                "O",
                &r.res_name,
                r.chain_id,
                r.seq_id,
                r.o,
                "O",
            );
        }

        if let Some(last) = ch.residues.last() {
            out.push_str(&format!(
                "TER   {:>5}      {:>3} {}{:>4}\n",
                serial, last.res_name, last.chain_id, last.seq_id
            ));
        }
        out.push_str("ENDMDL\n");
    }

    out.push_str("END\n");
    out
}

/// Write a multi-model NMR structure ensemble to disk in wwPDB `MODEL ... ENDMDL` format.
pub fn write_pdb_ensemble(
    path: impl AsRef<Path>,
    conformers: &[StructureConformer],
    title: &str,
) -> Result<(), String> {
    let text = format_pdb_ensemble(conformers, title);
    fs::write(path.as_ref(), text).map_err(|e| {
        format!(
            "failed to write PDB ensemble {}: {e}",
            path.as_ref().display()
        )
    })
}
