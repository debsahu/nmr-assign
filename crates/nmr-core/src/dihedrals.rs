//! Backbone `(phi, psi)` dihedral angle and secondary-structure prediction from chemical shifts.
//!
//! Provenance: Clean-room implementation from peer-reviewed NMR chemical-shift literature:
//! - Spera, S. & Bax, A. (1991). "Empirical correlation between protein backbone conformation
//!   and Calpha and Cbeta 13C nuclear magnetic resonance chemical shifts."
//!   *J. Am. Chem. Soc.* 113(14), 5490-5492.
//! - Wishart, D. S., Sykes, B. D. & Richards, F. M. (1991). "Relationship between nuclear
//!   magnetic resonance chemical shift and protein secondary structure." *J. Mol. Biol.* 222(2), 311-333;
//!   and Wishart, D. S., Sykes, B. D. & Richards, F. M. (1992). "The chemical shift index: a fast
//!   and simple method for the assignment of protein secondary structure through NMR spectroscopy."
//!   *Biochemistry* 31(6), 1647-1651.
//! - Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (1995). "1H, 13C and 15N
//!   random coil NMR chemical shifts of the common amino acids. I. Investigations of nearest-neighbor
//!   effects." *J. Biomol. NMR* 5(1), 67-81.

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;

/// Three-state backbone secondary structure class derived from chemical shifts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

/// Predicted backbone `(phi, psi)` torsion restraint and secondary-structure propensity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DihedralRestraint {
    pub seq_id: i32,
    pub res_name: String,
    pub phi_deg: f64,
    pub phi_err_deg: f64,
    pub psi_deg: f64,
    pub psi_err_deg: f64,
    pub ss_class: SecondaryStructure,
    pub confidence: f64,
    /// Composite smoothed secondary chemical shift index (positive = helical, negative = extended).
    pub secondary_shift_score: f64,
    /// Whether `phi` is confident enough to be actively restrained during 3D structure folding.
    pub restrained_phi: bool,
    /// Whether `psi` is confident enough to be actively restrained during 3D structure folding.
    pub restrained_psi: bool,
    /// Optional sidechain `chi1` target angle in degrees (populated only when `--include-chi1` is enabled).
    #[serde(default)]
    pub chi1_deg: Option<f64>,
    /// Optional sidechain `chi1` restraint half-width in degrees.
    #[serde(default)]
    pub chi1_err_deg: Option<f64>,
    /// Whether `chi1` is actively exported/restrained (default `false`).
    #[serde(default)]
    pub restrained_chi1: bool,
}

/// DSS-referenced random-coil chemical shifts (ppm) per standard residue type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomCoilShift {
    pub ca: f64,
    pub cb: f64,
    pub c: f64,
    pub n: f64,
    pub h: f64,
    pub ha: f64,
    pub hb: f64,
}

/// Random-coil chemical shifts (ppm, DSS reference at pH 5, 25 C) from Wishart et al. (*J. Biomol. NMR* 1995),
/// including oxidized cysteine (`CB > 35 ppm`) and pre-proline (`i+1 == PRO`) nearest-neighbor corrections.
pub fn random_coil_shifts(
    res_name: &str,
    cys_oxidized: bool,
    next_is_pro: bool,
) -> RandomCoilShift {
    let up = res_name.trim().to_ascii_uppercase();
    let (mut ca, mut cb, mut c, n, h, ha, hb) = match up.as_str() {
        "ALA" => (52.5, 19.1, 177.8, 123.8, 8.24, 4.32, 1.39),
        "ARG" => (56.0, 30.9, 176.3, 121.3, 8.23, 4.34, 1.84),
        "ASN" => (53.1, 38.9, 175.2, 118.7, 8.40, 4.74, 2.81),
        "ASP" => (54.2, 41.1, 176.3, 120.4, 8.34, 4.64, 2.72),
        "CYS" => {
            if cys_oxidized {
                (55.6, 41.1, 174.6, 118.8, 8.32, 4.71, 3.18)
            } else {
                (58.2, 28.0, 174.6, 118.8, 8.32, 4.55, 2.96)
            }
        }
        "GLN" => (55.7, 29.4, 176.0, 119.8, 8.32, 4.34, 2.08),
        "GLU" => (56.6, 29.9, 176.6, 120.2, 8.42, 4.35, 2.04),
        "GLY" => (45.1, 0.0, 173.9, 108.8, 8.33, 3.96, 0.00),
        "HIS" => (55.0, 29.0, 174.1, 118.2, 8.42, 4.73, 3.20),
        "ILE" => (61.1, 38.8, 176.4, 119.9, 8.00, 4.17, 1.90),
        "LEU" => (55.1, 42.4, 177.6, 121.8, 8.16, 4.34, 1.65),
        "LYS" => (56.2, 33.1, 176.6, 120.4, 8.29, 4.32, 1.81),
        "MET" => (55.4, 32.9, 176.3, 119.6, 8.28, 4.48, 2.08),
        "PHE" => (57.7, 39.6, 175.8, 120.3, 8.30, 4.62, 3.08),
        "PRO" => (63.3, 32.1, 177.3, 135.0, 0.00, 4.42, 2.15),
        "SER" => (58.3, 63.8, 174.6, 115.7, 8.31, 4.47, 3.88),
        "THR" => (61.8, 69.8, 174.7, 113.6, 8.15, 4.35, 4.22),
        "TRP" => (57.3, 29.6, 176.1, 121.3, 8.25, 4.66, 3.26),
        "TYR" => (57.9, 38.8, 175.9, 120.3, 8.12, 4.55, 3.01),
        "VAL" => (62.2, 32.9, 176.3, 119.2, 8.03, 4.12, 2.08),
        _ => (56.0, 33.0, 176.0, 120.0, 8.25, 4.35, 2.10),
    };

    if next_is_pro && up != "GLY" {
        // Wishart et al. (1995) Table 3: residues preceding Pro exhibit systematic upfield 13CA and 13C' shifts.
        ca -= 1.95;
        c -= 2.00;
        cb += 0.25;
    } else if next_is_pro && up == "GLY" {
        ca -= 1.00;
        c -= 1.50;
    }

    RandomCoilShift {
        ca,
        cb,
        c,
        n,
        h,
        ha,
        hb,
    }
}

/// Wrap an angle in degrees into `(-180.0, 180.0]`.
#[inline]
pub fn wrap_deg(deg: f64) -> f64 {
    let mut d = (deg + 180.0) % 360.0;
    if d <= 0.0 {
        d += 360.0;
    }
    d - 180.0
}

/// Signed periodic difference `a_deg - b_deg` wrapped into `(-180.0, 180.0]`.
#[inline]
pub fn periodic_angle_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    wrap_deg(a_deg - b_deg)
}

/// Unsigned shortest angular separation in `[0.0, 180.0]` degrees.
#[inline]
pub fn periodic_angle_abs_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    periodic_angle_diff_deg(a_deg, b_deg).abs()
}

/// Detect systematic uniform `13C` chemical-shift referencing offset (e.g. TMS vs DSS ~ -2.0 ppm).
/// Because helix shifts `CA` positive and `CB` negative while sheet shifts `CA` negative and `CB`
/// positive, the average `0.5 * (median(dCA) + median(dCB))` cancels secondary-structure bias and
/// isolates true uniform `13C` referencing offsets when exceeding `1.4 ppm`.
fn detect_uniform_c13_offset(reference: &Reference) -> f64 {
    let n = reference.sequence.len();
    let mut dca_vals = Vec::with_capacity(n);
    let mut dcb_vals = Vec::with_capacity(n);

    for i in 0..n {
        let res = &reference.sequence[i];
        let next_pro = i + 1 < n && reference.sequence[i + 1].eq_ignore_ascii_case("PRO");
        let cb_obs = reference.shifts.get(&(i, "CB".to_string())).copied();
        let cys_ox = res.eq_ignore_ascii_case("CYS") && cb_obs.unwrap_or(28.0) > 35.0;
        let rc = random_coil_shifts(res, cys_ox, next_pro);

        if let Some(&ca) = reference.shifts.get(&(i, "CA".to_string())) {
            dca_vals.push(ca - rc.ca);
        }
        if !res.eq_ignore_ascii_case("GLY") {
            if let Some(cb) = cb_obs {
                dcb_vals.push(cb - rc.cb);
            }
        }
    }

    if dca_vals.len() < 8 || dcb_vals.len() < 8 {
        return 0.0;
    }
    dca_vals.sort_by(|a, b| a.total_cmp(b));
    dcb_vals.sort_by(|a, b| a.total_cmp(b));
    let med_ca = dca_vals[dca_vals.len() / 2];
    let med_cb = dcb_vals[dcb_vals.len() / 2];

    if med_ca.signum() == med_cb.signum() && med_ca.abs() > 1.2 && med_cb.abs() > 1.2 {
        let est = 0.5 * (med_ca + med_cb);
        if est.abs() > 1.4 {
            return est;
        }
    }
    0.0
}

/// Predict smoothed 3-state secondary structure and continuous `(phi, psi)` torsion restraints
/// with uncertainty bounds from assigned chemical shifts.
pub fn predict_dihedrals(reference: &Reference) -> Vec<DihedralRestraint> {
    let n = reference.sequence.len();
    if n == 0 {
        return Vec::new();
    }

    let c13_off = detect_uniform_c13_offset(reference);

    let mut raw_score = vec![0.0f64; n];
    let mut dca_arr = vec![0.0f64; n];
    let mut dcb_arr = vec![0.0f64; n];
    let mut dc_arr = vec![0.0f64; n];
    let mut dh_arr = vec![0.0f64; n];
    let mut n_obs_shifts = vec![0usize; n];

    for i in 0..n {
        let res = &reference.sequence[i];
        let is_gly = res.eq_ignore_ascii_case("GLY");
        let is_pro = res.eq_ignore_ascii_case("PRO");
        let next_pro = i + 1 < n && reference.sequence[i + 1].eq_ignore_ascii_case("PRO");
        let cb_obs = reference
            .shifts
            .get(&(i, "CB".to_string()))
            .map(|&v| v - c13_off);
        let cys_ox = res.eq_ignore_ascii_case("CYS") && cb_obs.unwrap_or(28.0) > 35.0;
        let rc = random_coil_shifts(res, cys_ox, next_pro);

        let mut count = 0usize;
        let dca = if let Some(&ca) = reference.shifts.get(&(i, "CA".to_string())) {
            count += 1;
            (ca - c13_off) - rc.ca
        } else {
            0.0
        };
        let dcb = if !is_gly {
            if let Some(cb) = cb_obs {
                count += 1;
                cb - rc.cb
            } else {
                0.0
            }
        } else {
            // Glycine lacks CB; scale dCA slightly to match non-Gly (dCA - dCB) dynamic range.
            -0.35 * dca
        };
        let dc = if let Some(&c_val) = reference.shifts.get(&(i, "C".to_string())) {
            count += 1;
            (c_val - c13_off) - rc.c
        } else {
            0.0
        };
        let dh = if !is_pro {
            if let Some(&h_val) = reference.shifts.get(&(i, "H".to_string())) {
                count += 1;
                h_val - rc.h
            } else {
                0.0
            }
        } else {
            0.0
        };
        let dha = if let Some(&ha_val) = reference
            .shifts
            .get(&(i, "HA".to_string()))
            .or_else(|| reference.shifts.get(&(i, "HA2".to_string())))
        {
            count += 1;
            ha_val - rc.ha
        } else {
            0.0
        };

        dca_arr[i] = dca;
        dcb_arr[i] = dcb;
        dc_arr[i] = dc;
        dh_arr[i] = dh;
        n_obs_shifts[i] = count;

        // Spera & Bax (1991) + Wishart CSI (1992):
        // Helix: dCA > 0, dCB < 0, dC' > 0, dHN < 0, dHA < 0 -> positive composite index.
        // Sheet: dCA < 0, dCB > 0, dC' < 0, dHN > 0, dHA > 0 -> negative composite index.
        raw_score[i] = 0.55 * (dca - dcb) + 0.45 * dc - 0.45 * dh - 0.85 * dha;
    }

    // 3-residue triplet smoothing with [0.22, 0.56, 0.22] kernel
    let mut smooth_score = vec![0.0f64; n];
    for i in 0..n {
        let s_m1 = if i > 0 {
            raw_score[i - 1]
        } else {
            raw_score[i]
        };
        let s_0 = raw_score[i];
        let s_p1 = if i + 1 < n {
            raw_score[i + 1]
        } else {
            raw_score[i]
        };
        smooth_score[i] = 0.22 * s_m1 + 0.56 * s_0 + 0.22 * s_p1;
    }

    // Initial CSI secondary-structure classification with Wishart et al. (1992) cooperativity cleanup
    let mut ss_vec = vec![SecondaryStructure::Coil; n];
    for i in 0..n {
        if smooth_score[i] >= 0.95 && raw_score[i] >= 0.15 {
            ss_vec[i] = SecondaryStructure::Helix;
        } else if smooth_score[i] <= -0.85 && raw_score[i] <= -0.10 {
            ss_vec[i] = SecondaryStructure::Sheet;
        }
    }

    // Bridge single-residue gaps inside helices/strands and demote isolated 1-residue islands
    let prev_ss = ss_vec.clone();
    for i in 0..n {
        if prev_ss[i] == SecondaryStructure::Coil && i > 0 && i + 1 < n {
            if prev_ss[i - 1] == SecondaryStructure::Helix
                && prev_ss[i + 1] == SecondaryStructure::Helix
                && raw_score[i] > -0.2
            {
                ss_vec[i] = SecondaryStructure::Helix;
            } else if prev_ss[i - 1] == SecondaryStructure::Sheet
                && prev_ss[i + 1] == SecondaryStructure::Sheet
                && raw_score[i] < 0.2
            {
                ss_vec[i] = SecondaryStructure::Sheet;
            }
        } else if prev_ss[i] == SecondaryStructure::Helix {
            let left = i > 0 && prev_ss[i - 1] == SecondaryStructure::Helix;
            let right = i + 1 < n && prev_ss[i + 1] == SecondaryStructure::Helix;
            if !left && !right && smooth_score[i] < 1.5 {
                ss_vec[i] = SecondaryStructure::Coil;
            }
        } else if prev_ss[i] == SecondaryStructure::Sheet {
            let left = i > 0 && prev_ss[i - 1] == SecondaryStructure::Sheet;
            let right = i + 1 < n && prev_ss[i + 1] == SecondaryStructure::Sheet;
            if !left && !right && smooth_score[i] > -1.4 {
                ss_vec[i] = SecondaryStructure::Coil;
            }
        }
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let res = reference.sequence[i].to_ascii_uppercase();
        let is_gly = res == "GLY";
        let is_pro = res == "PRO";
        let next_pro = i + 1 < n && reference.sequence[i + 1].eq_ignore_ascii_case("PRO");
        let is_terminal = i == 0 || i + 1 == n;

        let sc = smooth_score[i];
        let r_sc = raw_score[i];
        let ss = if is_terminal {
            SecondaryStructure::Coil
        } else {
            ss_vec[i]
        };

        let obs_factor = (n_obs_shifts[i] as f64 / 3.0).clamp(0.25, 1.0);

        let (phi_deg, phi_err_deg, psi_deg, psi_err_deg, confidence) = if is_pro {
            // Proline pyrrolidine ring covalently restricts phi around -63 +/- 12 deg.
            let phi = -63.0;
            let phi_err = 12.0;
            let (psi, psi_err, conf) =
                if ss == SecondaryStructure::Helix || sc > 0.65 || dc_arr[i] > 0.7 {
                    (-40.0, 20.0, 0.85 * obs_factor)
                } else if ss == SecondaryStructure::Sheet || sc < -0.5 {
                    (144.0, 22.0, 0.82 * obs_factor)
                } else {
                    (142.0, 38.0, 0.65 * obs_factor)
                };
            (phi, phi_err, psi, psi_err, conf)
        } else if is_gly {
            // Glycine is achiral and frequently populates alpha_L (+80, +15) at helix C-caps and beta-turns.
            if ss == SecondaryStructure::Helix && r_sc > 0.6 {
                (-63.0, 22.0, -42.0, 22.0, 0.72 * obs_factor)
            } else if ss == SecondaryStructure::Sheet && r_sc < -0.6 {
                (-125.0, 35.0, 140.0, 35.0, 0.60 * obs_factor)
            } else if i > 0 && ss_vec[i - 1] == SecondaryStructure::Helix && r_sc < 0.3 {
                // Classic Schellman / alpha_L helix C-cap Glycine
                (78.0, 55.0, 18.0, 55.0, 0.30 * obs_factor)
            } else {
                // Unrestrained / broad Glycine turn region so NOEs freely determine turn handedness
                (75.0, 110.0, 15.0, 110.0, 0.20 * obs_factor)
            }
        } else if ss == SecondaryStructure::Helix {
            // Continuous modulation around canonical (-63.5, -42.0) alpha-helix basin
            let strength = ((sc - 0.9).max(0.0) / 2.2).clamp(0.0, 1.0);
            let phi = -63.5 + (-2.0 * (dc_arr[i] - 1.8)).clamp(-5.0, 5.0) * 0.4;
            let psi = -42.0 + (-1.5 * (dca_arr[i] - dcb_arr[i] - 3.0)).clamp(-5.0, 5.0) * 0.4;
            let err = (22.0 - 8.0 * strength).clamp(13.0, 24.0);
            let conf = ((0.68 + 0.28 * strength) * obs_factor).clamp(0.50, 0.96);
            (phi, err, psi, err, conf)
        } else if ss == SecondaryStructure::Sheet {
            // Continuous modulation around canonical (-122.0, +135.0) beta-strand basin (Spera & Bax 1991)
            let strength = ((-sc - 0.8).max(0.0) / 2.0).clamp(0.0, 1.0);
            let beta_branched = matches!(res.as_str(), "VAL" | "ILE" | "THR");
            let base_phi = if beta_branched { -124.0 } else { -120.0 };
            let base_psi = if next_pro { 138.0 } else { 134.0 };
            let d_ext = (dcb_arr[i] - dca_arr[i] - 2.5).clamp(-2.5, 2.5);
            let phi = base_phi - 2.0 * d_ext;
            let psi = base_psi + 2.2 * d_ext;
            let err = (26.0 - 8.0 * strength).clamp(16.0, 28.0);
            let conf = ((0.65 + 0.27 * strength) * obs_factor).clamp(0.50, 0.94);
            (phi, err, psi, err, conf)
        } else {
            // Turn / Coil: keep soft, generous tolerance bounds so NOE distance restraints shape loops cleanly!
            if next_pro {
                (-95.0, 45.0, 136.0, 40.0, 0.40 * obs_factor)
            } else if r_sc >= 0.45 || dc_arr[i] >= 0.55 {
                // Helical / Type-I turn propensity
                (-66.0, 42.0, -34.0, 45.0, 0.38 * obs_factor)
            } else if r_sc <= -0.35 {
                // Extended / beta-like coil
                (-115.0, 45.0, 134.0, 45.0, 0.38 * obs_factor)
            } else {
                // Polyproline-II / general coil basin
                (-82.0, 55.0, 132.0, 60.0, 0.28 * obs_factor)
            }
        };

        let is_ordered_non_gly =
            !is_terminal && !is_gly && ss != SecondaryStructure::Coil && confidence >= 0.55;
        let restrained_phi = !is_terminal && (is_pro || is_ordered_non_gly);
        let restrained_psi = !is_terminal && is_ordered_non_gly;

        out.push(DihedralRestraint {
            seq_id: reference.seq_ids[i],
            res_name: res,
            phi_deg,
            phi_err_deg,
            psi_deg,
            psi_err_deg,
            ss_class: ss,
            confidence,
            secondary_shift_score: sc,
            restrained_phi,
            restrained_psi,
            chi1_deg: None,
            chi1_err_deg: None,
            restrained_chi1: false,
        });
    }

    out
}

/// Summary accuracy metrics comparing shift-predicted `(phi, psi)` against experimental PDB torsions.
#[derive(Clone, Debug, PartialEq)]
pub struct DihedralValidationMetrics {
    pub n_compared_all: usize,
    pub n_compared_ordered: usize,
    pub phi_mae_all_deg: f64,
    pub psi_mae_all_deg: f64,
    pub phi_mae_ordered_deg: f64,
    pub psi_mae_ordered_deg: f64,
    pub within_bounds_ordered_pct: f64,
    pub q3_accuracy: f64,
}

fn torsion_ss_class(phi: f64, psi: f64) -> SecondaryStructure {
    if (-115.0..=-30.0).contains(&phi) && (-80.0..=-5.0).contains(&psi) {
        SecondaryStructure::Helix
    } else if ((-180.0..=-70.0).contains(&phi) || phi >= 155.0)
        && ((80.0..=180.0).contains(&psi) || psi <= -150.0)
    {
        SecondaryStructure::Sheet
    } else {
        SecondaryStructure::Coil
    }
}

/// Find the ungapped index mapping `(pred_idx, exp_idx)` that maximizes 3-letter residue sequence identity
/// (handling author sequence offsets such as BMRB 15536 vs PDB `1ENH` offset `+2`).
pub fn align_by_res_names(pred: &[(i32, &str)], exp: &[(i32, &str)]) -> Vec<(usize, usize)> {
    if pred.is_empty() || exp.is_empty() {
        return Vec::new();
    }

    // Try candidate integer shifts delta = pred_seq_id - exp_seq_id across [-20..=20]
    let min_delta = (pred.first().unwrap().0 - exp.last().unwrap().0) - 2;
    let max_delta = (pred.last().unwrap().0 - exp.first().unwrap().0) + 2;

    let mut best_pairs = Vec::new();
    let mut best_matches = 0usize;

    for delta in min_delta..=max_delta {
        let mut pairs = Vec::new();
        let mut matches = 0usize;
        for (ei, &(e_sid, e_res)) in exp.iter().enumerate() {
            let target_sid = e_sid + delta;
            if let Ok(pi) = pred.binary_search_by_key(&target_sid, |&(sid, _)| sid) {
                pairs.push((pi, ei));
                if pred[pi].1.eq_ignore_ascii_case(e_res) {
                    matches += 1;
                }
            }
        }
        // Prefer exact seq_id match (delta == 0) on ties
        if matches > best_matches || (matches == best_matches && matches > 0 && delta == 0) {
            best_matches = matches;
            best_pairs = pairs;
        }
    }

    best_pairs
}

/// Compare predicted backbone dihedral restraints against experimental `(seq_id, res_name, phi, psi)` torsions.
pub fn validate_dihedrals_against_torsions(
    predicted: &[DihedralRestraint],
    exp_torsions: &[(i32, String, Option<f64>, Option<f64>)],
) -> DihedralValidationMetrics {
    let pred_keys: Vec<(i32, &str)> = predicted
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let exp_keys: Vec<(i32, &str)> = exp_torsions
        .iter()
        .map(|(sid, rn, _, _)| (*sid, rn.as_str()))
        .collect();

    let aligned = align_by_res_names(&pred_keys, &exp_keys);

    let mut sum_phi_all = 0.0;
    let mut sum_psi_all = 0.0;
    let mut n_all = 0usize;

    let mut sum_phi_ord = 0.0;
    let mut sum_psi_ord = 0.0;
    let mut within_ord = 0usize;
    let mut n_ord = 0usize;

    let mut q3_hits = 0usize;

    for (pi, ei) in aligned {
        let pr = &predicted[pi];
        let (_, _, Some(exp_phi), Some(exp_psi)) = &exp_torsions[ei] else {
            continue;
        };
        let d_phi = periodic_angle_abs_diff_deg(pr.phi_deg, *exp_phi);
        let d_psi = periodic_angle_abs_diff_deg(pr.psi_deg, *exp_psi);

        sum_phi_all += d_phi;
        sum_psi_all += d_psi;
        n_all += 1;

        let exp_ss = torsion_ss_class(*exp_phi, *exp_psi);
        if pr.ss_class == exp_ss {
            q3_hits += 1;
        }

        if pr.ss_class != SecondaryStructure::Coil && pr.res_name != "GLY" {
            sum_phi_ord += d_phi;
            sum_psi_ord += d_psi;
            n_ord += 1;
            if d_phi <= pr.phi_err_deg + 10.0 && d_psi <= pr.psi_err_deg + 10.0 {
                within_ord += 1;
            }
        }
    }

    let denom_all = (n_all.max(1)) as f64;
    let denom_ord = (n_ord.max(1)) as f64;

    DihedralValidationMetrics {
        n_compared_all: n_all,
        n_compared_ordered: n_ord,
        phi_mae_all_deg: sum_phi_all / denom_all,
        psi_mae_all_deg: sum_psi_all / denom_all,
        phi_mae_ordered_deg: sum_phi_ord / denom_ord,
        psi_mae_ordered_deg: sum_psi_ord / denom_ord,
        within_bounds_ordered_pct: 100.0 * (within_ord as f64) / denom_ord,
        q3_accuracy: (q3_hits as f64) / denom_all,
    }
}
