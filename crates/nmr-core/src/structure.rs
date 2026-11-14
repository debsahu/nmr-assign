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

// Engh & Huber (1991) standard covalent backbone bond lengths (Angstroms) and interior angles (degrees).
// Note on RDC dipolar scaling vs static polymer geometry (Ottiger & Bax, J. Am. Chem. Soc. 1998, 120, 12334-12341):
// `BOND_N_H = 1.010 A` (`rdc::R_NH_GEOM_A`) and `BOND_CA_HA = 1.090 A` (`rdc::R_CAHA_GEOM_A`, used in
// `noesy::reconstruct_ca_substituents` for chiral `HA` and `GLY` `HA2`/`HA3`) orient unit vectors `u`
// (`||u|| = 1`), whereas vibrationally/librationally averaged `<r^-3>^(-1/3)` lengths `R_NH_EFF_A = 1.040 A`
// and `R_CAHA_EFF_A = 1.105 A` enter exclusively via `DMAX_NH_HZ` and `DMAX_CAHA_HZ`.
pub const BOND_N_CA: f64 = 1.458;
pub const BOND_CA_C: f64 = 1.525;
pub const BOND_C_N: f64 = 1.329;
pub const BOND_C_O: f64 = 1.231;
pub const BOND_N_H: f64 = 1.010;
pub const BOND_CA_HA: f64 = 1.090;

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
        let (cb, ha, ha2, ha3, qb) =
            reconstruct_ca_substituents(&sequence[i], n_pos[i], ca_pos[i], c_pos[i]);
        residues.push(PdbResidue {
            seq_id: seq_ids[i],
            res_name: sequence[i].to_ascii_uppercase(),
            chain_id: 'A',
            n: n_pos[i],
            h: h_pos[i],
            ca: ca_pos[i],
            ha,
            ha2,
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
    pub rdc: f64,
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
    evaluate_energy_and_gradients_with_rdcs(
        sequence,
        seq_ids,
        phi_rad,
        psi_rad,
        omega_rad,
        adrs,
        dihedrals,
        &[],
        0.0,
        weights,
        grad_phi,
        grad_psi,
    )
}

/// Evaluate total conformational energy including multi-medium variable-projection harmonic RDC
/// chi-squared (`0.5 * w_rdc * sum ((D_calc - D_obs)/sigma)^2` over `!is_free` working RDCs)
/// AND exact analytical `(dE/dphi_i, dE/dpsi_i)` torsion gradients via the Envelope Theorem +
/// Abe-Braun-Noguti-Go suffix-sum torque identity.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_energy_and_gradients_with_rdcs(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    weights: &EnergyWeights,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
) -> (PdbChain, EnergyBreakdown) {
    evaluate_energy_and_gradients_with_rdcs_and_omega(
        sequence, seq_ids, phi_rad, psi_rad, omega_rad, adrs, dihedrals, rdc_media, w_rdc, 0.0,
        weights, grad_phi, grad_psi, None,
    )
}

/// Evaluate total conformational energy including multi-medium variable-projection harmonic RDCs
/// and optional harmonic peptide-planarity `omega` restraint (`w_omega * wrap_rad(omega_i - omega_0)^2`),
/// computing exact `O(L)` analytical `(grad_phi, grad_psi, grad_omega)` gradients.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_energy_and_gradients_with_rdcs_and_omega(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    w_omega: f64,
    weights: &EnergyWeights,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
    grad_omega: Option<&mut [f64]>,
) -> (PdbChain, EnergyBreakdown) {
    evaluate_energy_and_gradients_with_rdcs_omega_hbond(
        sequence, seq_ids, phi_rad, psi_rad, omega_rad, adrs, dihedrals, rdc_media, w_rdc, w_omega,
        0.0, weights, grad_phi, grad_psi, grad_omega,
    )
}

/// Evaluate total conformational energy including multi-medium variable-projection harmonic RDCs,
/// optional harmonic peptide-planarity `omega` restraint, and smooth Kabsch & Sander (*Biopolymers*
/// 1983, 22, 2577-2637, Eq. 1) cooperative backbone `C'=O ... H^N-N` dipole-dipole electrostatics
/// (`w_hbond`), computing exact `O(L)` analytical `(grad_phi, grad_psi, grad_omega)` gradients.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_energy_and_gradients_with_rdcs_omega_hbond(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    w_omega: f64,
    w_hbond: f64,
    weights: &EnergyWeights,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
    grad_omega: Option<&mut [f64]>,
) -> (PdbChain, EnergyBreakdown) {
    evaluate_energy_and_gradients_internal(
        sequence, seq_ids, phi_rad, psi_rad, omega_rad, adrs, dihedrals, rdc_media, w_rdc, w_omega,
        w_hbond, weights, None, grad_phi, grad_psi, grad_omega,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_energy_and_gradients_internal(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    w_omega: f64,
    w_hbond: f64,
    weights: &EnergyWeights,
    plane_has_work: Option<&[bool]>,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
    mut grad_omega: Option<&mut [f64]>,
) -> (PdbChain, EnergyBreakdown) {
    let l = sequence.len();
    grad_phi.fill(0.0);
    grad_psi.fill(0.0);
    if let Some(gomg) = grad_omega.as_deref_mut() {
        gomg.fill(0.0);
    }

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
    //   O_i -> seg = 2*i + 2 (rotates with psi_i, NOT with omega_i; tracked separately in o_force/o_torque
    //          so omega_i rotates all atoms of residues j >= i + 1 around unit(N_{i+1} - C_i) at pivot N_{i+1})
    let n_segs = 2 * l + 1;
    let mut seg_force = vec![[0.0f64; 3]; n_segs];
    let mut seg_torque = vec![[0.0f64; 3]; n_segs];
    let mut o_force = vec![[0.0f64; 3]; l];
    let mut o_torque = vec![[0.0f64; 3]; l];

    let mut add_atom_grad = |seg: usize, pos: Vec3, g: Vec3| {
        seg_force[seg][0] += g[0];
        seg_force[seg][1] += g[1];
        seg_force[seg][2] += g[2];
        let tq = v_cross(pos, g);
        seg_torque[seg][0] += tq[0];
        seg_torque[seg][1] += tq[1];
        seg_torque[seg][2] += tq[2];
    };
    let mut add_o_grad = |res_idx: usize, pos: Vec3, g: Vec3| {
        o_force[res_idx][0] += g[0];
        o_force[res_idx][1] += g[1];
        o_force[res_idx][2] += g[2];
        let tq = v_cross(pos, g);
        o_torque[res_idx][0] += tq[0];
        o_torque[res_idx][1] += tq[1];
        o_torque[res_idx][2] += tq[2];
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
                        let neg_g = [-g[0], -g[1], -g[2]];
                        add_atom_grad(sa, pa, g);
                        add_atom_grad(sb, pb, neg_g);
                        if ka == 2 {
                            add_o_grad((sa - 2) / 2, pa, g);
                        }
                        if kb == 2 {
                            add_o_grad((sb - 2) / 2, pb, neg_g);
                        }
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
                        add_o_grad(i, ri.o, g);
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
                            let neg_g = [-g[0], -g[1], -g[2]];
                            add_atom_grad(sa, pa, g);
                            add_atom_grad(sb, pb, neg_g);
                            if ka == 2 {
                                add_o_grad((sa - 2) / 2, pa, g);
                            }
                            if kb == 2 {
                                add_o_grad((sb - 2) / 2, pb, neg_g);
                            }
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

    // 3b. Kabsch & Sander (Biopolymers 1983, 22, 2577-2637, Eq. 1) smooth 4-site backbone
    //     C'=O ... H^N-N dipole-dipole electrostatics (`|i - j| >= 3`), locking cooperative
    //     alpha-helical and beta-sheet peptide-plane orientations without H-bond tables.
    if w_hbond > 0.0 && l >= 4 {
        const KS_KAPPA: f64 = 27.888; // 332 * 0.42 * 0.20 kcal/mol * A
        const KS_EPS2: f64 = 0.20;
        const U0_ON: f64 = 3.5 * 3.5;
        const U1_ON: f64 = 5.2 * 5.2;
        const INV_DU_ON: f64 = 1.0 / (U1_ON - U0_ON);
        const CA_CULL2: f64 = 9.5 * 9.5; // Exact triangle-inequality upper bound for ||O_i - N_j|| < 5.2 A

        for i in 0..(l - 1) {
            let ri = &chain.residues[i];
            for j in 1..l {
                if i.abs_diff(j) < 3 {
                    continue;
                }
                let rj = &chain.residues[j];
                let Some(hj) = rj.h else {
                    continue;
                };
                if v_dist_sq(ri.ca, rj.ca) >= CA_CULL2 {
                    continue;
                }
                let d_on = v_sub(ri.o, rj.n);
                let u_on = v_dot(d_on, d_on);
                if u_on >= U1_ON {
                    continue;
                }
                let (s_sw, ds_du) = if u_on <= U0_ON {
                    (1.0, 0.0)
                } else {
                    let t = (u_on - U0_ON) * INV_DU_ON;
                    (
                        1.0 - t * t * (3.0 - 2.0 * t),
                        6.0 * t * (t - 1.0) * INV_DU_ON,
                    )
                };

                let d_ch = v_sub(ri.c, hj);
                let d_oh = v_sub(ri.o, hj);
                let d_cn = v_sub(ri.c, rj.n);

                let r2_on = u_on + KS_EPS2;
                let inv_on = 1.0 / r2_on.sqrt();
                let inv3_on = inv_on / r2_on;

                let r2_ch = v_dot(d_ch, d_ch) + KS_EPS2;
                let inv_ch = 1.0 / r2_ch.sqrt();
                let inv3_ch = inv_ch / r2_ch;

                let r2_oh = v_dot(d_oh, d_oh) + KS_EPS2;
                let inv_oh = 1.0 / r2_oh.sqrt();
                let inv3_oh = inv_oh / r2_oh;

                let r2_cn = v_dot(d_cn, d_cn) + KS_EPS2;
                let inv_cn = 1.0 / r2_cn.sqrt();
                let inv3_cn = inv_cn / r2_cn;

                let e_raw = KS_KAPPA * ((inv_on + inv_ch) - (inv_oh + inv_cn));
                breakdown.rama += w_hbond * s_sw * e_raw;

                let scale = w_hbond * s_sw * KS_KAPPA;
                let g_on = v_scale(d_on, -scale * inv3_on + 2.0 * w_hbond * ds_du * e_raw);
                let g_ch = v_scale(d_ch, -scale * inv3_ch);
                let g_oh = v_scale(d_oh, scale * inv3_oh);
                let g_cn = v_scale(d_cn, scale * inv3_cn);

                let g_c = v_add(g_ch, g_cn);
                let g_o = v_add(g_on, g_oh);
                let g_n = [-g_on[0] - g_cn[0], -g_on[1] - g_cn[1], -g_on[2] - g_cn[2]];
                let g_h = [-g_ch[0] - g_oh[0], -g_ch[1] - g_oh[1], -g_ch[2] - g_oh[2]];

                add_atom_grad(2 * i + 1, ri.c, g_c);
                add_atom_grad(2 * i + 2, ri.o, g_o);
                add_o_grad(i, ri.o, g_o);
                add_atom_grad(2 * j, rj.n, g_n);
                add_atom_grad(2 * j, hj, g_h);
            }
        }
    }

    // 3c. Multi-medium harmonic RDC potential (variable-projection SVD on working `!is_free` subset)
    if w_rdc > 0.0 && !rdc_media.is_empty() {
        breakdown.rdc = crate::rdc::accumulate_rdc_energy_and_forces(
            &chain,
            rdc_media,
            w_rdc,
            &mut add_atom_grad,
        );
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
            if let Some(gomg) = grad_omega.as_deref_mut() {
                if res_i + 1 < l {
                    let omg_f = v_sub(suf_f, o_force[res_i]);
                    let omg_t = v_sub(suf_t, o_torque[res_i]);
                    let n_next = chain.residues[res_i + 1].n;
                    let axis_omg = v_unit(v_sub(n_next, chain.residues[res_i].c));
                    let lever_omg = v_sub(omg_t, v_cross(n_next, omg_f));
                    gomg[res_i] += v_dot(axis_omg, lever_omg);
                }
            }
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

    // 4. Flat-bottom periodic (phi, psi) dihedral restraints + optional harmonic peptide-planarity omega restraint
    if w_omega > 0.0 {
        for i in 0..(l.saturating_sub(1)) {
            let target_omg = if omega_rad[i].cos() > 0.0 { 0.0 } else { PI };
            let domg = wrap_rad(omega_rad[i] - target_omg);
            breakdown.dihedral += w_omega * domg * domg;
            if let Some(gomg) = grad_omega.as_deref_mut() {
                gomg[i] += 2.0 * w_omega * domg;
            }
        }
    }
    if let Some(gomg) = grad_omega {
        gomg[l - 1] = 0.0;
    }
    if weights.w_dihedral > 0.0 && dihedrals.len() == l {
        let w_dih = weights.w_dihedral;
        let u_soft = 0.55f64;
        for i in 0..l {
            let dr = &dihedrals[i];
            if i > 0 && dr.restrained_phi {
                let unmeasured = plane_has_work.is_some_and(|hw| i < hw.len() && !hw[i]);
                let (wc, tol) = if unmeasured {
                    (
                        w_dih.max(28.0) * dr.confidence,
                        (dr.phi_err_deg * 0.25).clamp(1.5, 4.0).to_radians(),
                    )
                } else {
                    (
                        w_dih * dr.confidence,
                        if plane_has_work.is_some() {
                            dr.phi_err_deg.max(15.0).to_radians()
                        } else {
                            dr.phi_err_deg.to_radians()
                        },
                    )
                };
                let dphi = wrap_rad(phi_rad[i] - dr.phi_deg.to_radians());
                if dphi.abs() > tol {
                    let u = dphi.abs() - tol;
                    if !unmeasured && w_rdc > 0.0 && u > u_soft {
                        let s2 = u_soft * u_soft;
                        breakdown.dihedral += wc * s2 * (1.0 + 2.0 * (u / u_soft).ln());
                        grad_phi[i] += 2.0 * wc * (s2 / u) * dphi.signum();
                    } else {
                        breakdown.dihedral += wc * u * u;
                        grad_phi[i] += 2.0 * wc * u * dphi.signum();
                    }
                }
            }
            if i + 1 < l && dr.restrained_psi {
                let unmeasured = plane_has_work.is_some_and(|hw| i + 1 < hw.len() && !hw[i + 1]);
                let (wc, tol) = if unmeasured {
                    (
                        w_dih.max(28.0) * dr.confidence,
                        (dr.psi_err_deg * 0.25).clamp(1.5, 4.0).to_radians(),
                    )
                } else {
                    (
                        w_dih * dr.confidence,
                        if plane_has_work.is_some() {
                            dr.psi_err_deg.max(15.0).to_radians()
                        } else {
                            dr.psi_err_deg.to_radians()
                        },
                    )
                };
                let dpsi = wrap_rad(psi_rad[i] - dr.psi_deg.to_radians());
                if dpsi.abs() > tol {
                    let u = dpsi.abs() - tol;
                    if !unmeasured && w_rdc > 0.0 && u > u_soft {
                        let s2 = u_soft * u_soft;
                        breakdown.dihedral += wc * s2 * (1.0 + 2.0 * (u / u_soft).ln());
                        grad_psi[i] += 2.0 * wc * (s2 / u) * dpsi.signum();
                    } else {
                        breakdown.dihedral += wc * u * u;
                        grad_psi[i] += 2.0 * wc * u * dpsi.signum();
                    }
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

    breakdown.total = breakdown.noe
        + breakdown.dihedral
        + breakdown.vdw
        + breakdown.rama
        + breakdown.rg
        + breakdown.rdc;

    (chain, breakdown)
}

/// Evaluate pure RDC torsion-space energy and `(grad_phi, grad_psi)` gradients for fixed Saupe
/// tensors `fixed_s_matrices` (reference oracle for `< 1e-5` CPU vs WebGPU kernel parity).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
    sequence: &[String],
    seq_ids: &[i32],
    phi_rad: &[f64],
    psi_rad: &[f64],
    omega_rad: &[f64],
    rdc_media: &[crate::rdc::RdcMediumSet],
    fixed_s_matrices: &[[[f64; 3]; 3]],
    w_rdc: f64,
    grad_phi: &mut [f64],
    grad_psi: &mut [f64],
) -> f64 {
    let l = sequence.len();
    grad_phi.fill(0.0);
    grad_psi.fill(0.0);
    if l == 0 || w_rdc <= 0.0 || rdc_media.is_empty() {
        return 0.0;
    }

    let chain = build_backbone_from_torsions_rad(sequence, seq_ids, phi_rad, psi_rad, omega_rad);
    let n_segs = 2 * l + 1;
    let mut seg_force = vec![[0.0f64; 3]; n_segs];
    let mut seg_torque = vec![[0.0f64; 3]; n_segs];

    let e_rdc = crate::rdc::accumulate_rdc_energy_and_forces_fixed_saupe(
        &chain,
        rdc_media,
        fixed_s_matrices,
        w_rdc,
        |seg, pos, g| {
            seg_force[seg][0] += g[0];
            seg_force[seg][1] += g[1];
            seg_force[seg][2] += g[2];
            let tq = v_cross(pos, g);
            seg_torque[seg][0] += tq[0];
            seg_torque[seg][1] += tq[1];
            seg_torque[seg][2] += tq[2];
        },
    );

    let mut suf_f = [0.0f64; 3];
    let mut suf_t = [0.0f64; 3];
    for t in (0..(2 * l)).rev() {
        let next_seg = t + 1;
        suf_f = v_add(suf_f, seg_force[next_seg]);
        suf_t = v_add(suf_t, seg_torque[next_seg]);
        let res_i = t / 2;
        if t % 2 == 1 {
            let r = &chain.residues[res_i];
            let axis = v_unit(v_sub(r.c, r.ca));
            let lever_t = v_sub(suf_t, v_cross(r.c, suf_f));
            grad_psi[res_i] += v_dot(axis, lever_t);
        } else {
            let r = &chain.residues[res_i];
            let axis = v_unit(v_sub(r.ca, r.n));
            let lever_t = v_sub(suf_t, v_cross(r.ca, suf_f));
            grad_phi[res_i] += v_dot(axis, lever_t);
        }
    }
    grad_phi[0] = 0.0;
    e_rdc
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

#[allow(clippy::too_many_arguments)]
fn lbfgs_polish_with_rdcs_hbond(
    sequence: &[String],
    seq_ids: &[i32],
    phi: &mut [f64],
    psi: &mut [f64],
    omega: &[f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    w_hbond: f64,
    weights: &EnergyWeights,
    plane_has_work: Option<&[bool]>,
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
    let (mut best_chain, mut cur_eb) = evaluate_energy_and_gradients_internal(
        sequence,
        seq_ids,
        &x_phi(&x, l),
        &x_psi(&x, l),
        omega,
        adrs,
        dihedrals,
        rdc_media,
        w_rdc,
        0.0,
        w_hbond,
        weights,
        plane_has_work,
        &mut gp,
        &mut gs,
        None,
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
            let (ch_t, eb_t) = evaluate_energy_and_gradients_internal(
                sequence,
                seq_ids,
                &x_phi(&x_trial, l),
                &x_psi(&x_trial, l),
                omega,
                adrs,
                dihedrals,
                rdc_media,
                w_rdc,
                0.0,
                w_hbond,
                weights,
                plane_has_work,
                &mut gp,
                &mut gs,
                None,
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

/// Two-loop L-BFGS quasi-Newton minimizer in `3*L` `(phi, psi, omega)` torsion space with
/// harmonic Engh-Huber / Cornilescu (*JACS* 1998) peptide-planarity restraint (`w_omega`).
#[allow(clippy::too_many_arguments)]
fn lbfgs_polish_with_rdcs_and_omega(
    sequence: &[String],
    seq_ids: &[i32],
    phi: &mut [f64],
    psi: &mut [f64],
    omega: &mut [f64],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    w_omega: f64,
    w_hbond: f64,
    weights: &EnergyWeights,
    plane_has_work: Option<&[bool]>,
    max_iters: usize,
) -> (PdbChain, EnergyBreakdown) {
    let l = sequence.len();
    let dim = 3 * l;
    let m_hist = 7usize;
    let max_omg_dev = (7.5f64).to_radians();

    let mut target_omgs = vec![PI; l];
    for i in 0..l {
        target_omgs[i] = if omega[i].cos() > 0.0 { 0.0 } else { PI };
    }

    let mut x = vec![0.0f64; dim];
    for i in 0..l {
        x[3 * i] = phi[i];
        x[3 * i + 1] = psi[i];
        x[3 * i + 2] = omega[i];
    }

    let unpack3 = |xv: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut p = Vec::with_capacity(l);
        let mut s = Vec::with_capacity(l);
        let mut o = Vec::with_capacity(l);
        for i in 0..l {
            p.push(xv[3 * i]);
            s.push(xv[3 * i + 1]);
            o.push(xv[3 * i + 2]);
        }
        (p, s, o)
    };
    let pack3 = |gp: &[f64], gs: &[f64], go: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(3 * l);
        for i in 0..l {
            out.push(gp[i]);
            out.push(gs[i]);
            out.push(go[i]);
        }
        out
    };

    let mut gp = vec![0.0f64; l];
    let mut gs = vec![0.0f64; l];
    let mut go = vec![0.0f64; l];
    let (p0, s0, o0) = unpack3(&x);
    let (mut best_chain, mut cur_eb) = evaluate_energy_and_gradients_internal(
        sequence,
        seq_ids,
        &p0,
        &s0,
        &o0,
        adrs,
        dihedrals,
        rdc_media,
        w_rdc,
        w_omega,
        w_hbond,
        weights,
        plane_has_work,
        &mut gp,
        &mut gs,
        Some(&mut go),
    );

    let mut g = pack3(&gp, &gs, &go);
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
            (0.12 / g_norm.max(1.0)).clamp(1e-3, 0.18)
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
        if l > 0 {
            dir[3 * (l - 1) + 2] = 0.0; // omega_{l-1} inactive
        }

        let mut dg = dot_slice(&dir, &g);
        if dg >= 0.0 {
            for d_i in 0..dim {
                dir[d_i] = -g[d_i];
            }
            dir[0] = 0.0;
            if l > 0 {
                dir[3 * (l - 1) + 2] = 0.0;
            }
            dg = dot_slice(&dir, &g);
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        let max_step = dir.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let mut step = if max_step > 0.30 {
            0.30 / max_step
        } else {
            1.0
        };

        let mut accepted = false;
        let mut next_eb = cur_eb;
        let mut next_chain = None;

        for _ls in 0..8 {
            for i in 0..l {
                x_trial[3 * i] = wrap_rad(x[3 * i] + step * dir[3 * i]);
                x_trial[3 * i + 1] = wrap_rad(x[3 * i + 1] + step * dir[3 * i + 1]);
                let raw_omg = wrap_rad(x[3 * i + 2] + step * dir[3 * i + 2]);
                let dev = wrap_rad(raw_omg - target_omgs[i]).clamp(-max_omg_dev, max_omg_dev);
                x_trial[3 * i + 2] = wrap_rad(target_omgs[i] + dev);
            }
            let (pt, st, ot) = unpack3(&x_trial);
            let (ch_t, eb_t) = evaluate_energy_and_gradients_internal(
                sequence,
                seq_ids,
                &pt,
                &st,
                &ot,
                adrs,
                dihedrals,
                rdc_media,
                w_rdc,
                w_omega,
                w_hbond,
                weights,
                plane_has_work,
                &mut gp,
                &mut gs,
                Some(&mut go),
            );
            if eb_t.total <= cur_eb.total + 1e-4 * step * dg {
                g_trial = pack3(&gp, &gs, &go);
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
        phi[i] = x[3 * i];
        psi[i] = x[3 * i + 1];
        omega[i] = x[3 * i + 2];
    }

    (best_chain, cur_eb)
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

    let pure_noe_mode = dihedrals.is_empty();

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
    } else if pure_noe_mode {
        // Pure experimental-NOE de novo folding (`dihedrals.is_empty()`): infer coarse local
        // secondary-structure propensity strictly from Wuthrich (1986, Ch. 7) medium-range
        // `(i, i+3)/(i, i+4)` helical NOEs vs long-range `|i - j| >= 5` backbone beta-sheet NOEs.
        let mut med_helix = vec![0usize; l];
        let mut lr_sheet = vec![0usize; l];
        for adr in adrs {
            if adr.candidates.len() > 2 || adr.upper_bound > 5.6 {
                continue;
            }
            for c in &adr.candidates {
                let sep = c.res_a.abs_diff(c.res_b);
                let lo = c.res_a.min(c.res_b);
                let hi = c.res_a.max(c.res_b);
                let bb_pair = matches!(c.atom_a, NoeProton::AmideH | NoeProton::AlphaHA)
                    && matches!(c.atom_b, NoeProton::AmideH | NoeProton::AlphaHA);
                if (sep == 3 || sep == 4)
                    && (bb_pair
                        || matches!(c.atom_a, NoeProton::AlphaHA)
                        || matches!(c.atom_b, NoeProton::AlphaHA))
                    && adr.upper_bound <= 5.3
                {
                    for mh in &mut med_helix[lo..=hi] {
                        *mh += 1;
                    }
                } else if sep >= 5 && bb_pair && adr.upper_bound <= 5.2 {
                    lr_sheet[c.res_a] += 1;
                    lr_sheet[c.res_b] += 1;
                }
            }
        }
        for i in 0..l {
            let is_pro = sequence[i].eq_ignore_ascii_case("PRO");
            let is_gly = sequence[i].eq_ignore_ascii_case("GLY");
            let u = rng.f64();
            if is_pro {
                phi[i] = (-63.0f64).to_radians() + rng.normal(0.0, 0.12);
                psi[i] = if med_helix[i] >= 2 && lr_sheet[i] == 0 {
                    (-40.0f64).to_radians() + rng.normal(0.0, 0.22)
                } else {
                    (142.0f64).to_radians() + rng.normal(0.0, 0.30)
                };
            } else if med_helix[i] >= 3 && lr_sheet[i] == 0 && u < 0.80 {
                phi[i] = (-63.0f64).to_radians() + rng.normal(0.0, 0.28);
                psi[i] = (-42.0f64).to_radians() + rng.normal(0.0, 0.28);
            } else if lr_sheet[i] >= 2 && med_helix[i] <= 1 && u < 0.78 {
                phi[i] = (-118.0f64).to_radians() + rng.normal(0.0, 0.35);
                psi[i] = (136.0f64).to_radians() + rng.normal(0.0, 0.32);
            } else if is_gly && u < 0.34 {
                phi[i] = (82.0f64).to_radians() + rng.normal(0.0, 0.55);
                psi[i] = (14.0f64).to_radians() + rng.normal(0.0, 0.55);
            } else if u < 0.66 {
                phi[i] = (-108.0f64).to_radians() + rng.normal(0.0, 0.55);
                psi[i] = (138.0f64).to_radians() + rng.normal(0.0, 0.50);
            } else {
                phi[i] = (-65.0f64).to_radians() + rng.normal(0.0, 0.45);
                psi[i] = (-38.0f64).to_radians() + rng.normal(0.0, 0.45);
            }
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
                && (dihedrals[i].ss_class != SecondaryStructure::Coil
                    || (dihedrals[i].restrained_phi && dihedrals[i].restrained_psi))
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

        let s1_hb = if pure_noe_mode && prog >= 0.40 {
            0.55 * ((prog - 0.40) / 0.60)
        } else {
            0.0
        };
        let (_, eb) = evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            sequence,
            seq_ids,
            &phi,
            &psi,
            &omega,
            active_adrs,
            dihedrals,
            &[],
            0.0,
            0.0,
            s1_hb,
            &w,
            &mut gp,
            &mut gs,
            None,
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
            if (dihedrals.is_empty()
                || (idx < dihedrals.len() && dihedrals[idx].ss_class == SecondaryStructure::Coil))
                && !sequence[idx].eq_ignore_ascii_case("PRO")
            {
                let old_p = phi[idx];
                let old_s = psi[idx];
                let old_p2 = phi[idx + 1];
                let (_, e_before) = evaluate_energy_and_gradients_with_rdcs_omega_hbond(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    active_adrs,
                    dihedrals,
                    &[],
                    0.0,
                    0.0,
                    s1_hb,
                    &w,
                    &mut gp,
                    &mut gs,
                    None,
                );
                let delta = rng.normal(0.0, 1.15);
                phi[idx] = wrap_rad(phi[idx] + rng.normal(0.0, 0.90));
                psi[idx] = wrap_rad(psi[idx] + delta);
                if !sequence[idx + 1].eq_ignore_ascii_case("PRO") {
                    phi[idx + 1] = wrap_rad(phi[idx + 1] - 0.85 * delta);
                }
                let (_, e_after) = evaluate_energy_and_gradients_with_rdcs_omega_hbond(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    active_adrs,
                    dihedrals,
                    &[],
                    0.0,
                    0.0,
                    s1_hb,
                    &w,
                    &mut gp,
                    &mut gs,
                    None,
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
    lbfgs_polish_with_rdcs_hbond(
        sequence,
        seq_ids,
        &mut phi,
        &mut psi,
        &omega,
        adrs,
        dihedrals,
        &[],
        0.0,
        if pure_noe_mode { 0.95 } else { 0.0 },
        &soft_polish_weights,
        None,
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

        evaluate_energy_and_gradients_with_rdcs_omega_hbond(
            sequence,
            seq_ids,
            &phi,
            &psi,
            &omega,
            adrs,
            dihedrals,
            &[],
            0.0,
            0.0,
            if pure_noe_mode {
                1.15 + 0.35 * prog
            } else {
                0.0
            },
            &w,
            &mut gp,
            &mut gs,
            None,
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
    let (final_chain, final_eb) = lbfgs_polish_with_rdcs_hbond(
        sequence,
        seq_ids,
        &mut phi,
        &mut psi,
        &omega,
        adrs,
        dihedrals,
        &[],
        0.0,
        if pure_noe_mode { 1.55 } else { 0.0 },
        &final_weights,
        None,
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

    let rama_sort_w = if dihedrals.is_empty() { 0.06 } else { 0.0 };
    all_conformers.sort_by(|a, b| {
        let ta = a.noe_energy + a.dihedral_energy + a.vdw_energy + rama_sort_w * a.rama_energy;
        let tb = b.noe_energy + b.dihedral_energy + b.vdw_energy + rama_sort_w * b.rama_energy;
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

fn align_dihedrals_to_seq_ids(
    sequence: &[String],
    seq_ids: &[i32],
    dihedrals: &[DihedralRestraint],
) -> Vec<DihedralRestraint> {
    if dihedrals.is_empty() || dihedrals.len() == sequence.len() {
        return dihedrals.to_vec();
    }
    if !dihedrals.iter().any(|d| seq_ids.contains(&d.seq_id)) {
        return Vec::new();
    }
    (0..sequence.len())
        .map(|k| {
            if let Some(d) = dihedrals.iter().find(|d| d.seq_id == seq_ids[k]) {
                d.clone()
            } else {
                DihedralRestraint {
                    seq_id: seq_ids[k],
                    res_name: sequence[k].clone(),
                    phi_deg: -65.0,
                    phi_err_deg: 35.0,
                    psi_deg: 130.0,
                    psi_err_deg: 35.0,
                    ss_class: SecondaryStructure::Coil,
                    confidence: 0.0,
                    secondary_shift_score: 0.0,
                    restrained_phi: false,
                    restrained_psi: false,
                    chi1_deg: None,
                    chi1_err_deg: None,
                    restrained_chi1: false,
                }
            }
        })
        .collect()
}

/// Refine a folded conformer ensemble against multi-medium residual dipolar couplings (`RDC`s)
/// using exact variable-projection Losonczi SVD Saupe tensor updates + staged analytical
/// torsion-angle L-BFGS minimization over working (`!is_free`) RDCs.
///
/// Evaluates and returns `RdcValidationSummary` (including Cornilescu `Q` and held-out `Q_free`)
/// on the lowest-energy refined conformer.
#[allow(clippy::too_many_arguments)]
pub fn refine_ensemble_with_rdcs(
    conformers: &[StructureConformer],
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    lbfgs_iters: usize,
) -> Result<(Vec<StructureConformer>, crate::rdc::RdcValidationSummary), String> {
    if conformers.is_empty() {
        return Err("refine_ensemble_with_rdcs requires at least one conformer".to_string());
    }
    if sequence.len() != seq_ids.len() {
        return Err("sequence and seq_ids lengths must match".to_string());
    }
    let aligned_dihs;
    let dihedrals: &[DihedralRestraint] =
        if dihedrals.is_empty() || dihedrals.len() == sequence.len() {
            dihedrals
        } else {
            aligned_dihs = align_dihedrals_to_seq_ids(sequence, seq_ids, dihedrals);
            &aligned_dihs
        };
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!(
            "invalid RDC weight w_rdc={w_rdc}: must be finite and >= 0"
        ));
    }
    if rdc_media.is_empty() {
        return Err(
            "refine_ensemble_with_rdcs requires at least one RDC alignment medium".to_string(),
        );
    }

    // Up-front validation of every alignment medium (n_work >= 6, sigma > 0, cond <= 1e8)
    let init_val = crate::rdc::evaluate_rdc_validation(&conformers[0].chain, rdc_media)?;
    let n_work_tot: usize = init_val.media.iter().map(|m| m.n_work).sum();
    let eff_w_rdc = w_rdc * (185.0 / (n_work_tot.max(60) as f64)).clamp(0.34, 1.50);

    let refine_one = |(conf_idx, conf): (usize, &StructureConformer)| -> StructureConformer {
        let l = sequence.len();
        let mut phi: Vec<f64> = conf.phi_deg.iter().map(|d| d.to_radians()).collect();
        let mut psi: Vec<f64> = conf.psi_deg.iter().map(|d| d.to_radians()).collect();
        let mut omega: Vec<f64> = if conf.omega_deg.len() == l {
            conf.omega_deg.iter().map(|d| d.to_radians()).collect()
        } else {
            vec![PI; l]
        };

        let mut plane_has_work_rdc = vec![false; l];
        for m in rdc_media {
            for r in &m.restraints {
                if !r.is_free {
                    let pid = crate::rdc::rdc_peptide_plane_id(r);
                    if let Some(pos) = seq_ids.iter().position(|&sid| sid == pid) {
                        plane_has_work_rdc[pos] = true;
                    }
                }
            }
        }

        // Keep flat-bottom dihedral tolerances (`>= 15 deg`) so non-homologous chemical-shift
        // predictions prevent mirror-image Saupe degeneracies and stabilize unmeasured peptide planes
        // without biasing `(phi, psi)` away from sub-Hz experimental RDCs inside the native basin.
        let reg_dihs: Vec<DihedralRestraint> = if dihedrals.len() == l {
            dihedrals
                .iter()
                .map(|dr| {
                    let mut d = dr.clone();
                    if d.restrained_phi {
                        d.phi_err_deg = d.phi_err_deg.max(15.0);
                    }
                    if d.restrained_psi {
                        d.psi_err_deg = d.psi_err_deg.max(15.0);
                    }
                    d
                })
                .collect()
        } else if dihedrals.is_empty() {
            // Pure NOE + RDC refinement without chemical-shift dihedrals: do not self-tether to
            // unrefined NOE-only `(phi, psi)` so mis-oriented peptide planes can crankshaft freely!
            Vec::new()
        } else {
            (0..l)
                .map(|k| DihedralRestraint {
                    seq_id: seq_ids[k],
                    res_name: sequence[k].clone(),
                    phi_deg: conf.phi_deg[k],
                    phi_err_deg: 22.0,
                    psi_deg: conf.psi_deg[k],
                    psi_err_deg: 22.0,
                    ss_class: SecondaryStructure::Coil,
                    confidence: 0.25,
                    secondary_shift_score: 0.0,
                    restrained_phi: k > 0,
                    restrained_psi: k + 1 < l,
                    chi1_deg: None,
                    chi1_err_deg: None,
                    restrained_chi1: false,
                })
                .collect()
        };

        let pure_noe_mode = dihedrals.is_empty();
        let hbond_early = if pure_noe_mode { 0.85 } else { 0.0 };
        let hbond_post = if pure_noe_mode { 1.65 } else { 0.35 };
        let post_dihs: &[DihedralRestraint] = if dihedrals.len() == l {
            dihedrals
        } else {
            &reg_dihs
        };

        if eff_w_rdc > 0.0 && lbfgs_iters > 0 {
            let mut rng = Rng::new(0x5244_4331u64 ^ ((conf_idx as u64).wrapping_mul(0x9E37_79B9)));
            let mut gp = vec![0.0f64; l];
            let mut gs = vec![0.0f64; l];
            let mut vp = vec![0.0f64; l];
            let mut vs = vec![0.0f64; l];

            // Pre-polish at soft steric core (vdw_scale = 0.70) to align secondary-structure elements
            // with the initial Saupe tensor frame before discrete crankshaft & turn-basin hops.
            let w_soft = EnergyWeights {
                w_noe: 1.4,
                noe_soft_window: 2.4,
                w_dihedral: 2.0,
                w_vdw: 0.20,
                vdw_scale: 0.70,
                w_rama: 6.5,
                w_rg: 0.0,
            };
            let _ = lbfgs_polish_with_rdcs_hbond(
                sequence,
                seq_ids,
                &mut phi,
                &mut psi,
                &omega,
                adrs,
                &reg_dihs,
                rdc_media,
                0.35 * eff_w_rdc,
                hbond_early,
                &w_soft,
                None,
                55,
            );

            // Two passes of full-circle compensating peptide-plane crankshaft proposals
            // (including exact 180-deg `PI` peptide-plane inversions) plus canonical Ramachandran
            // basin jumps across all non-Proline internal residues.
            const CRANK_DELTAS: &[f64] = &[
                -2.95, -2.45, -1.90, -1.35, -0.80, -0.40, 0.40, 0.80, 1.35, 1.90, 2.45, 2.95, PI,
            ];
            const TURN_BASINS_DEG: &[(f64, f64)] = &[
                (90.0, 12.0),
                (64.0, 26.0),
                (-63.0, -42.0),
                (-88.0, -8.0),
                (-94.0, 4.0),
                (-82.0, 152.0),
                (-118.0, 136.0),
            ];
            for pass in 0..2 {
                let w_crank = EnergyWeights {
                    w_noe: 1.1,
                    noe_soft_window: 2.2,
                    w_dihedral: 1.2,
                    w_vdw: 0.30 + 0.25 * (pass as f64),
                    vdw_scale: 0.72 + 0.06 * (pass as f64),
                    w_rama: 7.2,
                    w_rg: 0.0,
                };
                let crank_rdc_w = (0.50 + 0.18 * (pass as f64)) * eff_w_rdc;
                let (_, mut cur_eb) = evaluate_energy_and_gradients_internal(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    adrs,
                    &reg_dihs,
                    rdc_media,
                    crank_rdc_w,
                    0.0,
                    hbond_early,
                    &w_crank,
                    None,
                    &mut gp,
                    &mut gs,
                    None,
                );

                for k in 0..(l.saturating_sub(1)) {
                    let next_is_pro = sequence[k + 1].eq_ignore_ascii_case("PRO");
                    let saved_psi_k = psi[k];
                    let saved_phi_k1 = phi[k + 1];
                    let mut best_psi_k = saved_psi_k;
                    let mut best_phi_k1 = saved_phi_k1;

                    for &d in CRANK_DELTAS {
                        let comp = if d.abs() > 2.6 { 1.0 } else { 0.95 };
                        psi[k] = wrap_rad(saved_psi_k + d);
                        if !next_is_pro {
                            phi[k + 1] = wrap_rad(saved_phi_k1 - comp * d);
                        }
                        let (_, cand_eb) = evaluate_energy_and_gradients_internal(
                            sequence,
                            seq_ids,
                            &phi,
                            &psi,
                            &omega,
                            adrs,
                            &reg_dihs,
                            rdc_media,
                            crank_rdc_w,
                            0.0,
                            hbond_early,
                            &w_crank,
                            None,
                            &mut gp,
                            &mut gs,
                            None,
                        );
                        if cand_eb.total < cur_eb.total {
                            cur_eb = cand_eb;
                            best_psi_k = psi[k];
                            best_phi_k1 = phi[k + 1];
                        }
                    }
                    psi[k] = best_psi_k;
                    phi[k + 1] = best_phi_k1;

                    let is_pro = sequence[k].eq_ignore_ascii_case("PRO");
                    if k > 0 && !is_pro {
                        let s_pk = phi[k];
                        let s_sk = psi[k];
                        let s_sm1 = psi[k - 1];
                        let s_pk1 = phi[k + 1];
                        let mut b_pk = s_pk;
                        let mut b_sk = s_sk;
                        let mut b_sm1 = s_sm1;
                        let mut b_pk1 = s_pk1;
                        for &(bp_deg, bs_deg) in TURN_BASINS_DEG {
                            let tp = bp_deg.to_radians();
                            let ts = bs_deg.to_radians();
                            let dp = wrap_rad(tp - s_pk);
                            let ds = wrap_rad(ts - s_sk);
                            phi[k] = tp;
                            psi[k] = ts;
                            psi[k - 1] = wrap_rad(s_sm1 - 0.92 * dp);
                            if !next_is_pro {
                                phi[k + 1] = wrap_rad(s_pk1 - 0.92 * ds);
                            }
                            let (_, cand_eb) = evaluate_energy_and_gradients_internal(
                                sequence,
                                seq_ids,
                                &phi,
                                &psi,
                                &omega,
                                adrs,
                                &reg_dihs,
                                rdc_media,
                                crank_rdc_w,
                                0.0,
                                hbond_early,
                                &w_crank,
                                None,
                                &mut gp,
                                &mut gs,
                                None,
                            );
                            if cand_eb.total < cur_eb.total {
                                cur_eb = cand_eb;
                                b_pk = phi[k];
                                b_sk = psi[k];
                                b_sm1 = psi[k - 1];
                                b_pk1 = phi[k + 1];
                            }
                        }
                        phi[k] = b_pk;
                        psi[k] = b_sk;
                        psi[k - 1] = b_sm1;
                        phi[k + 1] = b_pk1;
                    }
                }

                let _ = lbfgs_polish_with_rdcs_hbond(
                    sequence,
                    seq_ids,
                    &mut phi,
                    &mut psi,
                    &omega,
                    adrs,
                    &reg_dihs,
                    rdc_media,
                    crank_rdc_w,
                    hbond_early,
                    &w_crank,
                    None,
                    65,
                );
            }

            // Soft-core RDC simulated annealing cool-down so secondary-structure elements pivot
            // cleanly across loop hinges into the global multi-medium Saupe tensor minimum.
            let mut best_phi = phi.clone();
            let mut best_psi = psi.clone();
            let w_ref = EnergyWeights {
                w_noe: 1.0,
                noe_soft_window: 2.0,
                w_dihedral: 2.0,
                w_vdw: 1.5,
                vdw_scale: 0.92,
                w_rama: 7.5,
                ..EnergyWeights::default()
            };
            let (_, init_eb) = evaluate_energy_and_gradients_internal(
                sequence,
                seq_ids,
                &phi,
                &psi,
                &omega,
                adrs,
                &reg_dihs,
                rdc_media,
                0.65 * eff_w_rdc,
                0.0,
                hbond_early,
                &w_ref,
                None,
                &mut gp,
                &mut gs,
                None,
            );
            let mut best_e = init_eb.total;
            let n_relax = if lbfgs_iters >= 120 {
                360usize
            } else {
                150usize
            };
            let t_start = if lbfgs_iters >= 120 { 1.20 } else { 0.25 };
            for st in 0..n_relax {
                let prog = (st as f64) / (n_relax as f64);
                let temp = t_start * (0.015f64).powf(prog);
                let w_s = EnergyWeights {
                    w_noe: 1.1 - 0.3 * prog,
                    noe_soft_window: 2.3 - 0.5 * prog,
                    w_dihedral: 1.8 + 0.4 * prog,
                    w_vdw: 0.25 + 1.35 * prog * prog,
                    vdw_scale: 0.68 + 0.26 * prog,
                    w_rama: 7.0 + 1.0 * prog,
                    w_rg: 0.0,
                };
                let (_, eb_s) = evaluate_energy_and_gradients_internal(
                    sequence,
                    seq_ids,
                    &phi,
                    &psi,
                    &omega,
                    adrs,
                    &reg_dihs,
                    rdc_media,
                    (0.45 + 0.35 * prog) * eff_w_rdc,
                    0.0,
                    hbond_early,
                    &w_s,
                    None,
                    &mut gp,
                    &mut gs,
                    None,
                );
                if prog > 0.62 && eb_s.total < best_e {
                    best_e = eb_s.total;
                    best_phi.copy_from_slice(&phi);
                    best_psi.copy_from_slice(&psi);
                }
                let mut g2 = 0.0f64;
                for k in 0..l {
                    g2 += gp[k] * gp[k] + gs[k] * gs[k];
                }
                let lr = (0.042 * (1.0 - 0.35 * prog)) / (1.0 + 0.09 * g2.sqrt());
                let noise = (2.0 * lr * temp).sqrt() * 0.30;
                let step_cap = 0.10 * (1.0 - 0.65 * prog) + 0.02;
                for k in 0..l {
                    let flex = if k < reg_dihs.len()
                        && (reg_dihs[k].restrained_phi || reg_dihs[k].restrained_psi)
                    {
                        0.65
                    } else {
                        1.35
                    };
                    if k > 0 && !sequence[k].eq_ignore_ascii_case("PRO") {
                        vp[k] = 0.58 * vp[k] - lr * gp[k] + flex * rng.normal(0.0, noise);
                        phi[k] = wrap_rad(phi[k] + vp[k].clamp(-step_cap, step_cap));
                    } else if k > 0 {
                        vp[k] = 0.35 * vp[k] - 0.5 * lr * gp[k];
                        phi[k] = wrap_rad(phi[k] + vp[k].clamp(-0.04, 0.04));
                    }
                    vs[k] = 0.58 * vs[k] - lr * gs[k] + flex * rng.normal(0.0, noise);
                    psi[k] = wrap_rad(psi[k] + vs[k].clamp(-step_cap, step_cap));
                }
            }
            phi.copy_from_slice(&best_phi);
            psi.copy_from_slice(&best_psi);

            // Post-annealing 180-deg peptide-plane & tripeptide basin sweep before final L-BFGS
            let w_post = EnergyWeights {
                w_noe: 0.90,
                noe_soft_window: 2.0,
                w_dihedral: 1.5,
                w_vdw: 1.2,
                vdw_scale: 0.88,
                w_rama: 7.5,
                w_rg: 0.0,
            };
            let (_, mut post_eb) = evaluate_energy_and_gradients_internal(
                sequence,
                seq_ids,
                &phi,
                &psi,
                &omega,
                adrs,
                &reg_dihs,
                rdc_media,
                0.75 * eff_w_rdc,
                0.0,
                hbond_early,
                &w_post,
                None,
                &mut gp,
                &mut gs,
                None,
            );
            for k in 0..(l.saturating_sub(1)) {
                let next_is_pro = sequence[k + 1].eq_ignore_ascii_case("PRO");
                let saved_psi_k = psi[k];
                let saved_phi_k1 = phi[k + 1];
                let mut best_psi_k = saved_psi_k;
                let mut best_phi_k1 = saved_phi_k1;
                for &d in CRANK_DELTAS {
                    let comp = if d.abs() > 2.6 { 1.0 } else { 0.95 };
                    psi[k] = wrap_rad(saved_psi_k + d);
                    if !next_is_pro {
                        phi[k + 1] = wrap_rad(saved_phi_k1 - comp * d);
                    }
                    let (_, cand_eb) = evaluate_energy_and_gradients_internal(
                        sequence,
                        seq_ids,
                        &phi,
                        &psi,
                        &omega,
                        adrs,
                        &reg_dihs,
                        rdc_media,
                        0.75 * eff_w_rdc,
                        0.0,
                        hbond_early,
                        &w_post,
                        None,
                        &mut gp,
                        &mut gs,
                        None,
                    );
                    if cand_eb.total < post_eb.total {
                        post_eb = cand_eb;
                        best_psi_k = psi[k];
                        best_phi_k1 = phi[k + 1];
                    }
                }
                psi[k] = best_psi_k;
                phi[k + 1] = best_phi_k1;

                if k > 0 && !sequence[k].eq_ignore_ascii_case("PRO") {
                    let s_pk = phi[k];
                    let s_sk = psi[k];
                    let s_sm1 = psi[k - 1];
                    let s_pk1 = phi[k + 1];
                    let mut b_pk = s_pk;
                    let mut b_sk = s_sk;
                    let mut b_sm1 = s_sm1;
                    let mut b_pk1 = s_pk1;
                    for &(bp_deg, bs_deg) in TURN_BASINS_DEG {
                        let tp = bp_deg.to_radians();
                        let ts = bs_deg.to_radians();
                        let dp = wrap_rad(tp - s_pk);
                        let ds = wrap_rad(ts - s_sk);
                        phi[k] = tp;
                        psi[k] = ts;
                        psi[k - 1] = wrap_rad(s_sm1 - 0.92 * dp);
                        if !next_is_pro {
                            phi[k + 1] = wrap_rad(s_pk1 - 0.92 * ds);
                        }
                        let (_, cand_eb) = evaluate_energy_and_gradients_internal(
                            sequence,
                            seq_ids,
                            &phi,
                            &psi,
                            &omega,
                            adrs,
                            &reg_dihs,
                            rdc_media,
                            0.75 * eff_w_rdc,
                            0.0,
                            hbond_early,
                            &w_post,
                            None,
                            &mut gp,
                            &mut gs,
                            None,
                        );
                        if cand_eb.total < post_eb.total {
                            post_eb = cand_eb;
                            b_pk = phi[k];
                            b_sk = psi[k];
                            b_sm1 = psi[k - 1];
                            b_pk1 = phi[k + 1];
                        }
                    }
                    phi[k] = b_pk;
                    psi[k] = b_sk;
                    psi[k - 1] = b_sm1;
                    phi[k + 1] = b_pk1;
                }
            }

            let w_a = EnergyWeights {
                w_noe: 0.90,
                noe_soft_window: 2.0,
                w_dihedral: 2.0,
                w_vdw: 1.6,
                vdw_scale: 0.92,
                w_rama: 7.5,
                ..EnergyWeights::default()
            };
            let (_, mut cur_wa_eb) = lbfgs_polish_with_rdcs_hbond(
                sequence,
                seq_ids,
                &mut phi,
                &mut psi,
                &omega,
                adrs,
                &reg_dihs,
                rdc_media,
                0.70 * eff_w_rdc,
                hbond_early,
                &w_a,
                None,
                (lbfgs_iters / 3).max(45),
            );

            // Relaxed 180-deg peptide-plane flip escape:
            let w_flip_soft = EnergyWeights {
                w_noe: 0.85,
                noe_soft_window: 2.2,
                w_dihedral: 1.5,
                w_vdw: 0.45,
                vdw_scale: 0.76,
                w_rama: 7.5,
                w_rg: 0.0,
            };
            for _flip_pass in 0..2 {
                for k in 0..(l.saturating_sub(1)) {
                    if sequence[k + 1].eq_ignore_ascii_case("PRO") {
                        continue;
                    }
                    let rc_k = if sequence[k].eq_ignore_ascii_case("GLY") {
                        ResClass::Gly
                    } else if sequence[k].eq_ignore_ascii_case("PRO") {
                        ResClass::Pro
                    } else {
                        ResClass::General
                    };
                    let rc_k1 = if sequence[k + 1].eq_ignore_ascii_case("GLY") {
                        ResClass::Gly
                    } else {
                        ResClass::General
                    };
                    let vr_k = if k > 0 {
                        ramachandran_energy_and_grad(rc_k, phi[k], psi[k]).0
                    } else {
                        0.0
                    };
                    let vr_k1 = if k + 2 < l {
                        ramachandran_energy_and_grad(rc_k1, phi[k + 1], psi[k + 1]).0
                    } else {
                        0.0
                    };
                    let k_is_gly = sequence[k].eq_ignore_ascii_case("GLY");
                    let k1_is_gly = sequence[k + 1].eq_ignore_ascii_case("GLY");
                    let pos_or_eclipsed_phi =
                        (!k_is_gly && k > 0 && phi[k] > (-28.0f64).to_radians())
                            || (!k1_is_gly && phi[k + 1] > (-28.0f64).to_radians());
                    let dih_viol = (k < reg_dihs.len()
                        && !k_is_gly
                        && reg_dihs[k].restrained_psi
                        && wrap_rad(psi[k] - reg_dihs[k].psi_deg.to_radians()).abs() > 0.95)
                        || (k + 1 < reg_dihs.len()
                            && !k1_is_gly
                            && reg_dihs[k + 1].restrained_phi
                            && wrap_rad(phi[k + 1] - reg_dihs[k + 1].phi_deg.to_radians()).abs()
                                > 0.95);
                    if !pos_or_eclipsed_phi && !dih_viol && vr_k < 1.8 && vr_k1 < 1.8 {
                        continue;
                    }

                    let mut proposals: Vec<(f64, f64, f64, f64)> = vec![
                        (
                            phi[k],
                            wrap_rad(psi[k] + PI),
                            wrap_rad(phi[k + 1] - PI),
                            psi[k + 1],
                        ),
                        (
                            phi[k],
                            wrap_rad(psi[k] + 2.15),
                            wrap_rad(phi[k + 1] - 2.15),
                            psi[k + 1],
                        ),
                        (
                            phi[k],
                            wrap_rad(psi[k] - 2.15),
                            wrap_rad(phi[k + 1] + 2.15),
                            psi[k + 1],
                        ),
                    ];
                    if k > 0 && !sequence[k].eq_ignore_ascii_case("PRO") {
                        proposals.push((
                            (-96.0f64).to_radians(),
                            (138.0f64).to_radians(),
                            (-102.0f64).to_radians(),
                            (138.0f64).to_radians(),
                        ));
                        proposals.push((
                            (-92.0f64).to_radians(),
                            (142.0f64).to_radians(),
                            (-68.0f64).to_radians(),
                            (-24.0f64).to_radians(),
                        ));
                        proposals.push((
                            (88.0f64).to_radians(),
                            (12.0f64).to_radians(),
                            (-96.0f64).to_radians(),
                            (140.0f64).to_radians(),
                        ));
                    }
                    if dih_viol && k + 1 < reg_dihs.len() {
                        let pk = if k > 0 && !k_is_gly && reg_dihs[k].restrained_phi {
                            reg_dihs[k].phi_deg.to_radians()
                        } else {
                            phi[k]
                        };
                        let sk = if !k_is_gly && reg_dihs[k].restrained_psi {
                            reg_dihs[k].psi_deg.to_radians()
                        } else {
                            wrap_rad(psi[k] + PI)
                        };
                        let pk1 = if !k1_is_gly && reg_dihs[k + 1].restrained_phi {
                            reg_dihs[k + 1].phi_deg.to_radians()
                        } else {
                            wrap_rad(phi[k + 1] - PI)
                        };
                        let sk1 = if k + 2 < l && !k1_is_gly && reg_dihs[k + 1].restrained_psi {
                            reg_dihs[k + 1].psi_deg.to_radians()
                        } else {
                            psi[k + 1]
                        };
                        proposals.push((pk, sk, pk1, sk1));
                    }

                    for (pk, sk, pk1, sk1) in proposals {
                        let mut trial_phi = phi.clone();
                        let mut trial_psi = psi.clone();
                        trial_phi[k] = pk;
                        trial_psi[k] = sk;
                        trial_phi[k + 1] = pk1;
                        trial_psi[k + 1] = sk1;
                        let _ = lbfgs_polish_with_rdcs_hbond(
                            sequence,
                            seq_ids,
                            &mut trial_phi,
                            &mut trial_psi,
                            &omega,
                            adrs,
                            &reg_dihs,
                            rdc_media,
                            0.70 * eff_w_rdc,
                            hbond_early,
                            &w_flip_soft,
                            None,
                            18,
                        );
                        let (_, cand_eb) = lbfgs_polish_with_rdcs_hbond(
                            sequence,
                            seq_ids,
                            &mut trial_phi,
                            &mut trial_psi,
                            &omega,
                            adrs,
                            &reg_dihs,
                            rdc_media,
                            0.70 * eff_w_rdc,
                            hbond_early,
                            &w_a,
                            None,
                            14,
                        );
                        if cand_eb.total + 1e-3 < cur_wa_eb.total {
                            cur_wa_eb = cand_eb;
                            phi.copy_from_slice(&trial_phi);
                            psi.copy_from_slice(&trial_psi);
                            break;
                        }
                    }
                }
            }

            // Post-convergence equilibration of peptide planes with ZERO working RDCs
            // (`!plane_has_work_rdc[k + 1]`), e.g. whole planes held out under `ByPeptidePlane`
            // `RdcFree` or unmeasured residues: keep all working-RDC planes locked to the Saupe
            // tensor while re-centering unmeasured planes via strictly compensating crankshaft
            // rotations (`psi[k] + d, phi[k+1] - d`) on their physical NOE + Kabsch-Sander
            // H-bond + non-homologous chemical-shift dihedral minimum.
            if lbfgs_iters >= 140 {
                let w_unmeas = EnergyWeights {
                    w_noe: if pure_noe_mode { 2.25 } else { 1.25 },
                    noe_soft_window: 2.0,
                    w_dihedral: 2.0,
                    w_vdw: 1.8,
                    vdw_scale: 0.95,
                    w_rama: if pure_noe_mode { 11.5 } else { 8.5 },
                    w_rg: 0.0,
                };
                let hbond_unmeas = if pure_noe_mode { 2.20 } else { 0.85 };
                let (_, mut cur_unmeas_eb) = lbfgs_polish_with_rdcs_hbond(
                    sequence,
                    seq_ids,
                    &mut phi,
                    &mut psi,
                    &omega,
                    adrs,
                    post_dihs,
                    rdc_media,
                    0.85 * eff_w_rdc,
                    hbond_unmeas,
                    &w_unmeas,
                    Some(&plane_has_work_rdc),
                    28,
                );
                for k in 0..(l.saturating_sub(1)) {
                    if plane_has_work_rdc[k + 1] || sequence[k + 1].eq_ignore_ascii_case("PRO") {
                        continue;
                    }
                    let mut plane_cands: Vec<(f64, f64)> = Vec::with_capacity(10);
                    if dihedrals.len() == l
                        && dihedrals[k].restrained_psi
                        && dihedrals[k + 1].restrained_phi
                    {
                        let sk_t = dihedrals[k].psi_deg.to_radians();
                        let pk1_t = dihedrals[k + 1].phi_deg.to_radians();
                        if wrap_rad((sk_t + pk1_t) - (psi[k] + phi[k + 1])).abs() < 0.65 {
                            plane_cands.push((sk_t, pk1_t));
                        }
                    }
                    for &d in &[PI, 2.05, -2.05, 1.15, -1.15, 0.55, -0.55, 0.28, -0.28] {
                        plane_cands.push((wrap_rad(psi[k] + d), wrap_rad(phi[k + 1] - d)));
                    }
                    for (sk, pk1) in plane_cands {
                        let mut trial_phi = phi.clone();
                        let mut trial_psi = psi.clone();
                        trial_psi[k] = sk;
                        trial_phi[k + 1] = pk1;
                        let (_, cand_eb) = lbfgs_polish_with_rdcs_hbond(
                            sequence,
                            seq_ids,
                            &mut trial_phi,
                            &mut trial_psi,
                            &omega,
                            adrs,
                            post_dihs,
                            rdc_media,
                            0.85 * eff_w_rdc,
                            hbond_unmeas,
                            &w_unmeas,
                            Some(&plane_has_work_rdc),
                            16,
                        );
                        if cand_eb.total + 1e-3 < cur_unmeas_eb.total {
                            cur_unmeas_eb = cand_eb;
                            phi.copy_from_slice(&trial_phi);
                            psi.copy_from_slice(&trial_psi);
                        }
                    }
                }
            }

            // Enable physical peptide-planarity omega relaxation (sigma_omega ~ 3.7 deg;
            // Cornilescu et al., JACS 1998) alongside (phi, psi) in the final two L-BFGS stages.
            let final_plane_mask = if lbfgs_iters >= 140 {
                Some(plane_has_work_rdc.as_slice())
            } else {
                None
            };
            let active_post_dihs = if lbfgs_iters >= 140 {
                post_dihs
            } else {
                &reg_dihs
            };
            let w_b = EnergyWeights {
                w_noe: if pure_noe_mode { 1.85 } else { 0.80 },
                noe_soft_window: 1.8,
                w_dihedral: 2.0,
                w_vdw: 2.0,
                vdw_scale: 0.98,
                w_rama: if pure_noe_mode { 11.0 } else { 8.0 },
                ..EnergyWeights::default()
            };
            let _ = lbfgs_polish_with_rdcs_and_omega(
                sequence,
                seq_ids,
                &mut phi,
                &mut psi,
                &mut omega,
                adrs,
                active_post_dihs,
                rdc_media,
                0.85 * eff_w_rdc,
                110.0,
                if lbfgs_iters >= 140 {
                    hbond_post
                } else {
                    hbond_early
                },
                &w_b,
                final_plane_mask,
                (lbfgs_iters / 2).max(65),
            );
        }

        let final_plane_mask = if lbfgs_iters >= 140 {
            Some(plane_has_work_rdc.as_slice())
        } else {
            None
        };
        let active_post_dihs = if lbfgs_iters >= 140 {
            post_dihs
        } else {
            &reg_dihs
        };
        let final_weights = EnergyWeights {
            w_noe: if pure_noe_mode { 1.65 } else { 0.70 },
            noe_soft_window: 1.8,
            w_dihedral: 2.0,
            w_vdw: 2.2,
            w_rama: if pure_noe_mode { 11.5 } else { 8.5 },
            ..EnergyWeights::default()
        };
        let (final_chain, final_eb) = lbfgs_polish_with_rdcs_and_omega(
            sequence,
            seq_ids,
            &mut phi,
            &mut psi,
            &mut omega,
            adrs,
            active_post_dihs,
            rdc_media,
            eff_w_rdc,
            115.0,
            if lbfgs_iters >= 140 {
                hbond_post
            } else {
                hbond_early
            },
            &final_weights,
            final_plane_mask,
            lbfgs_iters,
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
    };

    let n_conf = conformers.len();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_conf);

    let mut refined = Vec::with_capacity(n_conf);
    if n_threads <= 1 || n_conf <= 1 {
        for (idx, c) in conformers.iter().enumerate() {
            refined.push(refine_one((idx, c)));
        }
    } else {
        let chunk_len = n_conf.div_ceil(n_threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_idx, chunk) in conformers.chunks(chunk_len).enumerate() {
                let base_idx = chunk_idx * chunk_len;
                handles.push(scope.spawn(move || {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(k, c)| refine_one((base_idx + k, c)))
                        .collect::<Vec<_>>()
                }));
            }
            for h in handles {
                if let Ok(part) = h.join() {
                    refined.extend(part);
                }
            }
        });
    }

    refined.sort_by(|a, b| a.total_energy.total_cmp(&b.total_energy));
    let summary = crate::rdc::evaluate_rdc_validation(&refined[0].chain, rdc_media)?;
    Ok((refined, summary))
}

/// Full end-to-end iterative ADR NOESY folding + optional `TorsionDb` `(phi, psi)` restraints
/// + optional multi-medium `RDC` torsion-space refinement and Cornilescu `Q` / `Q_free` validation.
pub fn calculate_structure_with_rdcs(
    shifts: &Reference,
    noesy_peaks: &[Peak],
    rdc_media: &[crate::rdc::RdcMediumSet],
    use_torsion_db: bool,
    w_rdc: f64,
    cfg: &StructureCalcConfig,
) -> Result<
    (
        StructureEnsembleResult,
        Option<crate::rdc::RdcValidationSummary>,
    ),
    String,
> {
    let dihedral_restraints = if use_torsion_db {
        crate::torsion_db::predict_torsions(shifts).to_dihedral_restraints()
    } else {
        predict_dihedrals(shifts)
    };
    calculate_structure_with_rdcs_and_dihedrals(
        shifts,
        noesy_peaks,
        dihedral_restraints,
        rdc_media,
        w_rdc,
        cfg,
    )
}

/// Iterative ADR NOESY folding + caller-supplied `DihedralRestraint`s (e.g. `HomologFirewall`
/// benchmark predictions) + optional multi-medium `RDC` refinement across ADR cycles.
pub fn calculate_structure_with_rdcs_and_dihedrals(
    shifts: &Reference,
    noesy_peaks: &[Peak],
    dihedral_restraints: Vec<DihedralRestraint>,
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    cfg: &StructureCalcConfig,
) -> Result<
    (
        StructureEnsembleResult,
        Option<crate::rdc::RdcValidationSummary>,
    ),
    String,
> {
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!(
            "invalid RDC weight w_rdc={w_rdc}: must be finite and >= 0"
        ));
    }

    let initial_adrs = match_initial_adrs(noesy_peaks, shifts, &cfg.adr_cfg);
    let mut adrs = initial_adrs.clone();
    let initial_unambiguous_adrs = adrs
        .iter()
        .filter(|r| r.is_unambiguous || r.candidates.len() == 1)
        .count();

    let use_rdc = !rdc_media.is_empty() && w_rdc > 0.0;
    let mut master = Rng::new(cfg.seed);
    let mut cycle0_cfg = cfg.clone();
    if cfg.n_refinement_cycles > 0 {
        cycle0_cfg.lbfgs_iters = (cfg.lbfgs_iters * 2 / 3).max(120);
    }
    if use_rdc {
        cycle0_cfg.n_select = cfg.n_select.max(16).min(cfg.n_replicas.max(1));
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

    if use_rdc && cfg.n_refinement_cycles > 0 {
        let (ens_rdc, _) = refine_ensemble_with_rdcs(
            &ensemble,
            &shifts.sequence,
            &shifts.seq_ids,
            &adrs,
            &dihedral_restraints,
            rdc_media,
            0.60 * w_rdc,
            110,
        )?;
        ensemble = ens_rdc;
    }

    for cyc in 0..cfg.n_refinement_cycles {
        let is_last = cyc + 1 == cfg.n_refinement_cycles;
        let min_keep = if is_last { 0.07 } else { 0.035 };
        let cum_target = if is_last { 0.90 } else { 0.95 };
        let n_guide = if is_last {
            ensemble.len().clamp(1, 4)
        } else {
            ensemble.len().clamp(1, 5)
        };
        if use_rdc {
            // Re-evaluate all initial NOESY candidates against the RDC-refined coordinates so
            // true contacts prematurely pruned in an early coarse cycle are recovered cleanly.
            adrs = initial_adrs.clone();
        }
        reweight_adrs_with_coords(
            &mut adrs,
            n_guide,
            min_keep,
            cum_target,
            cfg.adr_cfg.unambiguous_threshold,
            |m, res_idx, site| ensemble[m].chain.residues.get(res_idx)?.noe_coord(site),
        );

        let mut cyc_cfg = if is_last {
            cfg.clone()
        } else {
            cycle0_cfg.clone()
        };
        if use_rdc {
            cyc_cfg.n_select = cfg.n_select.max(16).min(cfg.n_replicas.max(1));
        }
        let n_seed_models = if is_last { 2 } else { 3 };
        let (wf_num, wf_den) = if is_last { (3, 4) } else { (1, 2) };
        let top_seeds = &ensemble[..ensemble.len().min(n_seed_models)];
        let mut folded = fold_ensemble_seeded(
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

        if use_rdc {
            // Preserve the top RDC-refined conformers alongside newly folded conformers so
            // high-temperature NOE annealing never discards an already-converged Saupe basin!
            let keep_prev = ensemble.len().min(8);
            folded.truncate(cyc_cfg.n_select.saturating_sub(keep_prev).max(4));
            folded.extend_from_slice(&ensemble[..keep_prev]);
            let (ens_rdc, _) = refine_ensemble_with_rdcs(
                &folded,
                &shifts.sequence,
                &shifts.seq_ids,
                &adrs,
                &dihedral_restraints,
                rdc_media,
                0.85 * w_rdc,
                130,
            )?;
            ensemble = ens_rdc;
            if is_last {
                adrs = initial_adrs.clone();
                let n_g_final = ensemble.len().clamp(1, 4);
                reweight_adrs_with_coords(
                    &mut adrs,
                    n_g_final,
                    0.07,
                    0.90,
                    cfg.adr_cfg.unambiguous_threshold,
                    |m, res_idx, site| ensemble[m].chain.residues.get(res_idx)?.noe_coord(site),
                );
            }
        } else {
            ensemble = folded;
        }
    }

    if rdc_media.is_empty() {
        return Ok((
            StructureEnsembleResult {
                conformers: ensemble,
                dihedral_restraints,
                initial_unambiguous_adrs,
                final_adrs: adrs,
            },
            None,
        ));
    }

    let (mut refined_conformers, rdc_summary) = refine_ensemble_with_rdcs(
        &ensemble,
        &shifts.sequence,
        &shifts.seq_ids,
        &adrs,
        &dihedral_restraints,
        rdc_media,
        w_rdc,
        cfg.lbfgs_iters.max(200),
    )?;
    refined_conformers.truncate(cfg.n_select.max(1).min(refined_conformers.len()));

    Ok((
        StructureEnsembleResult {
            conformers: refined_conformers,
            dihedral_restraints,
            initial_unambiguous_adrs,
            final_adrs: adrs,
        },
        Some(rdc_summary),
    ))
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
        r.ha2 = r.ha2.map(xform);
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
                    r.ha2.unwrap_or(r.ha),
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

/// Compute Kabsch backbone (`N, CA, C'`) RMSD (in Angstroms) between `chain_a` and `chain_b`
/// restricted to residues with `seq_min <= seq_id <= seq_max`.
pub fn backbone_rmsd_seq_range(
    chain_a: &PdbChain,
    chain_b: &PdbChain,
    seq_min: i32,
    seq_max: i32,
) -> f64 {
    let mask: Vec<bool> = chain_a
        .residues
        .iter()
        .map(|r| r.seq_id >= seq_min && r.seq_id <= seq_max)
        .collect();
    backbone_rmsd_between_chains(chain_a, chain_b, Some(&mask))
}

/// Map an experimental proton/pseudoatom name on `res_name` (`seq_idx`) to `(NoeProton, upper_allowance_a)`
/// following Wuthrich, Billeter & Braun (*J. Mol. Biol.* 1983, 169, 949-961) covalent pseudoatom geometry
/// relative to backbone `(AmideH, AlphaHA, BetaQB)`.
fn map_exp_proton_to_noe_site(
    res_name: &str,
    is_n_term: bool,
    atom_pat: &str,
) -> Option<(NoeProton, f64)> {
    let r = res_name.trim().to_ascii_uppercase();
    let a = atom_pat.trim().to_ascii_uppercase();
    let is_gly = r == "GLY";
    let is_pro = r == "PRO";

    if matches!(
        a.as_str(),
        "H" | "HN" | "H1" | "H2" | "H3" | "HT1" | "HT2" | "HT3"
    ) {
        if is_pro {
            return None;
        }
        let allow = if is_n_term && a != "H" && a != "HN" {
            0.35
        } else {
            0.0
        };
        return Some((NoeProton::AmideH, allow));
    }
    if a.starts_with("HA") || a == "1HA" || a == "2HA" || a == "QA" {
        let allow = if is_gly { 0.90 } else { 0.0 };
        return Some((NoeProton::AlphaHA, allow));
    }
    if is_gly {
        return None;
    }
    if a.starts_with("HB") || a == "1HB" || a == "2HB" || a == "3HB" || a == "QB" {
        let has_wild = a.contains('*') || a.contains('#') || a == "QB";
        let is_beta_methine = matches!(r.as_str(), "ILE" | "THR" | "VAL") && a == "HB";
        let allow = if has_wild || is_beta_methine {
            0.30
        } else {
            0.65
        };
        return Some((NoeProton::BetaQB, allow));
    }
    if a.starts_with("HG") || a.starts_with("QG") || a == "1HG" || a == "2HG" {
        return Some((NoeProton::BetaQB, 1.55));
    }
    if a.starts_with("HD") || a.starts_with("QD") || a == "1HD" || a == "2HD" {
        return Some((NoeProton::BetaQB, 2.45));
    }
    if a.starts_with("HE") || a.starts_with("QE") || a == "1HE" || a == "2HE" {
        return Some((NoeProton::BetaQB, 3.35));
    }
    if a.starts_with("HZ") || a.starts_with("QZ") {
        return Some((NoeProton::BetaQB, 4.10));
    }
    if a.starts_with("HH") || a.starts_with("QH") {
        return Some((NoeProton::BetaQB, 4.55));
    }
    None
}

/// Read experimental proton-proton NOE distance restraints from a clean ASCII TSV file
/// (`restraint_id  lower_a  upper_a  group_1  group_2`) and map onto [`AmbiguousDistanceRestraint`]s.
pub fn read_exp_noe_tsv(
    path: impl AsRef<Path>,
    sequence: &[String],
    seq_ids: &[i32],
) -> Result<Vec<AmbiguousDistanceRestraint>, String> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        format!(
            "failed to read experimental NOE TSV '{}': {e}",
            path.as_ref().display()
        )
    })?;
    if !bytes.is_ascii() {
        return Err(format!(
            "experimental NOE TSV '{}' contains non-ASCII bytes",
            path.as_ref().display()
        ));
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    parse_exp_noe_tsv_str(text, sequence, seq_ids)
}

/// Parse experimental proton-proton NOE distance restraints from ASCII TSV text.
pub fn parse_exp_noe_tsv_str(
    text: &str,
    sequence: &[String],
    seq_ids: &[i32],
) -> Result<Vec<AmbiguousDistanceRestraint>, String> {
    if !text.as_bytes().is_ascii() {
        return Err("experimental NOE TSV contains non-ASCII characters".to_string());
    }
    if sequence.len() != seq_ids.len() {
        return Err("sequence and seq_ids length mismatch".to_string());
    }
    let mut seq_to_idx = std::collections::BTreeMap::new();
    for (idx, &sid) in seq_ids.iter().enumerate() {
        seq_to_idx.insert(sid, idx);
    }

    let parse_group = |grp: &str, line_no: usize| -> Result<Vec<(usize, NoeProton, f64)>, String> {
        let mut out = Vec::new();
        for tok in grp.split('|') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let (s_str, a_str) = t.split_once(':').ok_or_else(|| {
                format!("line {line_no}: malformed group token '{t}' (expected seq_id:atom)")
            })?;
            let sid: i32 = s_str
                .trim()
                .parse()
                .map_err(|e| format!("line {line_no}: invalid seq_id '{s_str}': {e}"))?;
            let Some(&r_idx) = seq_to_idx.get(&sid) else {
                continue;
            };
            if let Some((site, allow)) =
                map_exp_proton_to_noe_site(&sequence[r_idx], r_idx == 0, a_str)
            {
                out.push((r_idx, site, allow));
            }
        }
        Ok(out)
    };

    let mut adrs = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
        if cols.first().copied() == Some("restraint_id") {
            continue;
        }
        if cols.len() < 5 {
            return Err(format!(
                "line {line_no}: expected 5 TSV columns, got {}",
                cols.len()
            ));
        }
        let rid: usize = cols[0]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid restraint_id '{}': {e}", cols[0]))?;
        let lower_a: f64 = cols[1]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid lower_a '{}': {e}", cols[1]))?;
        let upper_a: f64 = cols[2]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid upper_a '{}': {e}", cols[2]))?;
        if !lower_a.is_finite() || !upper_a.is_finite() || lower_a <= 0.0 || upper_a < lower_a {
            return Err(format!(
                "line {line_no}: invalid distance bounds [{lower_a}, {upper_a}]"
            ));
        }

        let g1 = parse_group(cols[3], line_no)?;
        let g2 = parse_group(cols[4], line_no)?;
        if g1.is_empty() || g2.is_empty() {
            continue;
        }

        // If any pair in `g1 x g2` resolves to the same residue and coarse proton/pseudoatom site
        // (e.g. an intra-residue `29:HB2 -- 29:HD1` alternative inside an `OR`-ambiguous peak),
        // the peak is trivially explained intra-residually (`d ~ 0 <= upper_a`) and must NOT force
        // the remaining inter-residue candidates into spurious cross-links!
        if g1
            .iter()
            .any(|&(ra, sa, _)| g2.iter().any(|&(rb, sb, _)| ra == rb && sa == sb))
        {
            continue;
        }

        let mut pair_map: std::collections::BTreeMap<
            (usize, NoeProton, usize, NoeProton),
            (f64, f64),
        > = std::collections::BTreeMap::new();
        for &(ra, sa, aa) in &g1 {
            for &(rb, sb, ab) in &g2 {
                let key = if (ra, sa) <= (rb, sb) {
                    (ra, sa, rb, sb)
                } else {
                    (rb, sb, ra, sa)
                };
                let allow_up = aa.max(ab) + 0.55 * aa.min(ab);
                let allow_lo = aa + ab;
                pair_map
                    .entry(key)
                    .and_modify(|cur| {
                        cur.0 = cur.0.max(allow_up);
                        cur.1 = cur.1.max(allow_lo);
                    })
                    .or_insert((allow_up, allow_lo));
            }
        }
        if pair_map.is_empty() {
            continue;
        }

        let max_allow_up = pair_map.values().map(|v| v.0).fold(0.0f64, f64::max);
        let max_allow_lo = pair_map.values().map(|v| v.1).fold(0.0f64, f64::max);
        let w = 1.0 / (pair_map.len() as f64);
        let candidates: Vec<crate::noesy::NoeCandidatePair> = pair_map
            .keys()
            .map(|&(ra, sa, rb, sb)| crate::noesy::NoeCandidatePair {
                res_a: ra,
                atom_a: sa,
                res_b: rb,
                atom_b: sb,
                shift_score: 1.0,
                network_score: 1.0,
                weight: w,
            })
            .collect();
        let is_unambiguous = candidates.len() == 1;
        adrs.push(AmbiguousDistanceRestraint {
            peak_index: rid,
            exp: "EXP_NOE".to_string(),
            lower_bound: (lower_a - max_allow_lo).max(1.80),
            upper_bound: upper_a + max_allow_up,
            all_candidates: candidates.clone(),
            candidates,
            is_unambiguous,
        });
    }

    if adrs.is_empty() {
        return Err(
            "experimental NOE TSV produced zero valid proton-proton restraints".to_string(),
        );
    }
    Ok(adrs)
}

/// Fold a protein ensemble de novo starting strictly from sequence + experimental NOE restraints
/// (`adrs`) and optional `dihedrals` (with zero reference PDB coordinates), followed by optional
/// multi-medium variable-projection RDC refinement over working (`!is_free`) couplings.
#[allow(clippy::too_many_arguments)]
pub fn fold_from_experimental_noes_and_rdcs(
    sequence: &[String],
    seq_ids: &[i32],
    exp_adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[crate::rdc::RdcMediumSet],
    w_rdc: f64,
    cfg: &StructureCalcConfig,
) -> Result<
    (
        StructureEnsembleResult,
        Option<crate::rdc::RdcValidationSummary>,
    ),
    String,
> {
    if sequence.is_empty() || sequence.len() != seq_ids.len() {
        return Err("non-empty matching sequence and seq_ids required".to_string());
    }
    if exp_adrs.is_empty() {
        return Err("at least one experimental NOE restraint is required".to_string());
    }
    let aligned_dihs;
    let dihedrals: &[DihedralRestraint] =
        if dihedrals.is_empty() || dihedrals.len() == sequence.len() {
            dihedrals
        } else {
            aligned_dihs = align_dihedrals_to_seq_ids(sequence, seq_ids, dihedrals);
            &aligned_dihs
        };

    let mut adrs = exp_adrs.to_vec();
    let initial_unambiguous_adrs = adrs
        .iter()
        .filter(|r| r.is_unambiguous || r.candidates.len() == 1)
        .count();
    let use_rdc = !rdc_media.is_empty() && w_rdc > 0.0;
    let mut master = Rng::new(cfg.seed);

    let mut cycle0_cfg = cfg.clone();
    if cfg.n_refinement_cycles > 0 {
        cycle0_cfg.lbfgs_iters = (cfg.lbfgs_iters * 2 / 3).max(120);
    }
    if use_rdc {
        cycle0_cfg.n_select = cfg.n_select.max(16).min(cfg.n_replicas.max(1));
    }
    let mut ensemble = fold_ensemble_seeded(
        sequence,
        seq_ids,
        &adrs,
        dihedrals,
        &cycle0_cfg,
        &[],
        0,
        1,
        master.next_u64(),
    );

    if use_rdc && cfg.n_refinement_cycles > 0 {
        let (ens_rdc, _) = refine_ensemble_with_rdcs(
            &ensemble,
            sequence,
            seq_ids,
            &adrs,
            dihedrals,
            rdc_media,
            0.60 * w_rdc,
            110,
        )?;
        ensemble = ens_rdc;
    }

    for cyc in 0..cfg.n_refinement_cycles {
        let is_last = cyc + 1 == cfg.n_refinement_cycles;
        let min_keep = if is_last { 0.07 } else { 0.035 };
        let cum_target = if is_last { 0.90 } else { 0.95 };
        let n_guide = if is_last {
            ensemble.len().clamp(1, 4)
        } else {
            ensemble.len().clamp(1, 5)
        };
        if use_rdc {
            adrs = exp_adrs.to_vec();
        }
        reweight_adrs_with_coords(
            &mut adrs,
            n_guide,
            min_keep,
            cum_target,
            cfg.adr_cfg.unambiguous_threshold,
            |m, res_idx, site| ensemble[m].chain.residues.get(res_idx)?.noe_coord(site),
        );

        let mut cyc_cfg = if is_last {
            cfg.clone()
        } else {
            cycle0_cfg.clone()
        };
        if use_rdc {
            cyc_cfg.n_select = cfg.n_select.max(16).min(cfg.n_replicas.max(1));
        }
        let n_seed_models = if is_last { 2 } else { 3 };
        let (wf_num, wf_den) = if is_last { (3, 4) } else { (1, 2) };
        let top_seeds = &ensemble[..ensemble.len().min(n_seed_models)];
        let mut folded = fold_ensemble_seeded(
            sequence,
            seq_ids,
            &adrs,
            dihedrals,
            &cyc_cfg,
            top_seeds,
            wf_num,
            wf_den,
            master.next_u64(),
        );

        if use_rdc {
            let keep_prev = ensemble.len().min(8);
            folded.truncate(cyc_cfg.n_select.saturating_sub(keep_prev).max(4));
            folded.extend_from_slice(&ensemble[..keep_prev]);
            let (ens_rdc, _) = refine_ensemble_with_rdcs(
                &folded,
                sequence,
                seq_ids,
                &adrs,
                dihedrals,
                rdc_media,
                0.85 * w_rdc,
                130,
            )?;
            ensemble = ens_rdc;
            if is_last {
                adrs = exp_adrs.to_vec();
                let n_g_final = ensemble.len().clamp(1, 4);
                reweight_adrs_with_coords(
                    &mut adrs,
                    n_g_final,
                    0.07,
                    0.90,
                    cfg.adr_cfg.unambiguous_threshold,
                    |m, res_idx, site| ensemble[m].chain.residues.get(res_idx)?.noe_coord(site),
                );
            }
        } else {
            ensemble = folded;
        }
    }

    let mut res = StructureEnsembleResult {
        conformers: ensemble,
        dihedral_restraints: dihedrals.to_vec(),
        initial_unambiguous_adrs,
        final_adrs: adrs,
    };

    if !use_rdc {
        return Ok((res, None));
    }

    let (mut refined, _) = refine_ensemble_with_rdcs(
        &res.conformers,
        sequence,
        seq_ids,
        &res.final_adrs,
        dihedrals,
        rdc_media,
        w_rdc,
        cfg.lbfgs_iters.max(200),
    )?;
    refined.truncate(cfg.n_select.max(1));
    let rdc_val = crate::rdc::evaluate_rdc_validation(&refined[0].chain, rdc_media)?;
    res.conformers = refined;
    Ok((res, Some(rdc_val)))
}
