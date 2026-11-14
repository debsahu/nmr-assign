//! Aromatic ring spin-system models (`PHE`, `TYR`, `TRP`, `HIS`), volumetric 2D/3D simulators
//! & artifact-suppressing peak pickers, histidine tautomeric state classifier, and exact
//! probabilistic bipartite ring-to-backbone matcher with calibrated confidences.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed
//! pulse-sequence physics, chemical-shift distributions, and covalent IUPAC topologies:
//! - `2D (HB)CB(CGCD)HD` & `2D (HB)CB(CGCDCE)HE` (scalar `1J_CC` relay correlating aliphatic
//!   `13CB` with aromatic `1HD*` and `1HE*` protons across quaternary `13CG`/`13CD2`):
//!   Yamazaki, Forman-Kay & Kay, *J. Am. Chem. Soc.* 115, 11054–11055 (1993).
//! - Histidine imidazole `15ND1` / `15NE2` tautomeric and protonation state determination
//!   from long-range `2J_NH` / `3J_NH` `1H-15N` HMQC chemical shifts and cross-peak patterns:
//!   Pelton, Torchia, Meadow & Roseman, *Protein Sci.* 2, 543–558 (1993).
//! - Aromatic `1H-13C` HSQC / constant-time HSQC (`CT-HSQC`) and intra-ring `1H-13C-13C-1H`
//!   `HCCH-TOCSY` / `HCCH-COSY` coherence transfer:
//!   Bax, Clore & Gronenborn, *J. Magn. Reson.* 88, 425–431 (1990);
//!   Torizawa, Ono, Terauchi & Kainosho, *J. Am. Chem. Soc.* 127, 12620–12626 (2005).
//! - IUPAC-IUBMB-IUPAB biochemical NMR nomenclature & public BMRB statistics:
//!   Markley et al., *J. Mol. Biol.* 280, 933–952 (1998);
//!   Ulrich et al., *Nucleic Acids Res.* 36, D402–D408 (2008).
//! - Exact polynomial-time maximum-weight bipartite matching (Hungarian / Kuhn–Munkres):
//!   Kuhn, *Naval Res. Logist. Quart.* 2, 83–97 (1955); Munkres, *J. Soc. Ind. Appl. Math.* 5, 32–38 (1957).

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::{LN_2, PI, TAU};

use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::picker::estimate_noise_mad;
use crate::rng::Rng;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use crate::topology::{
    residue_topology, CarbonRegion, HisVariantPriors, ProchiralGroup, SidechainPriors,
    HIS_DOUBLY_PROTONATED_PRIORS, HIS_ND1H_PRIORS, HIS_NE2H_PRIORS,
};

/// Canonical experiment identifiers for aromatic sidechain assignment.
pub const AROMATIC_EXPERIMENTS: [&str; 6] = [
    "AROM_C13HSQC",
    "HBCBCGCDHD",
    "HBCBCGCDCEHE",
    "AROM_HCCH_TOCSY",
    "HIS_LR_N15HMQC",
    "TRP_IND_N15HSQC",
];

/// Supported 2D and 3D NMR experiments for aromatic spin-system assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AromaticExperiment {
    /// 2D aromatic `1H-13C` HSQC / CT-HSQC (`1H` `5.5..9.0 ppm`, `13C` `108..142 ppm`).
    AromC13Hsqc,
    /// 2D `(HB)CB(CGCD)HD` (`x = HD*`, `y = CB` at `25..46 ppm`; Yamazaki et al. 1993).
    Hbcbcgcdhd,
    /// 2D `(HB)CB(CGCDCE)HE` (`x = HE*`, `y = CB` at `25..46 ppm`; Yamazaki et al. 1993).
    Hbcbcgcdcehe,
    /// 3D intra-ring `1H-13C-13C-1H` aromatic `HCCH-TOCSY` / `COSY` (`x = H`, `y = C`, `z = H_tocsy`).
    AromHcchTocsy,
    /// 2D long-range `2J_NH` `1H-15N` HMQC correlating `His` `HE1`/`HD2` with `ND1`/`NE2` (Pelton et al. 1993).
    HisLrN15Hmqc,
    /// 2D `1H-15N` HSQC capturing the downfield `Trp` indole `15NE1-1HE1` pair (`NE1 ~ 129 ppm, HE1 ~ 10.1 ppm`).
    TrpIndN15Hsqc,
}

impl AromaticExperiment {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AromC13Hsqc => "AROM_C13HSQC",
            Self::Hbcbcgcdhd => "HBCBCGCDHD",
            Self::Hbcbcgcdcehe => "HBCBCGCDCEHE",
            Self::AromHcchTocsy => "AROM_HCCH_TOCSY",
            Self::HisLrN15Hmqc => "HIS_LR_N15HMQC",
            Self::TrpIndN15Hsqc => "TRP_IND_N15HSQC",
        }
    }

    pub fn parse_exp(name: &str) -> Option<Self> {
        let up = name
            .trim()
            .to_ascii_uppercase()
            .replace(['(', ')', '-', ' '], "");
        match up.as_str() {
            "AROM_C13HSQC" | "AROMC13HSQC" => Some(Self::AromC13Hsqc),
            "HBCBCGCDHD" => Some(Self::Hbcbcgcdhd),
            "HBCBCGCDCEHE" => Some(Self::Hbcbcgcdcehe),
            "AROM_HCCH_TOCSY" | "AROMHCCHTOCSY" | "AROMHCCHCOSY" => Some(Self::AromHcchTocsy),
            "HIS_LR_N15HMQC" | "HISLRN15HMQC" | "LRN15HMQC" => Some(Self::HisLrN15Hmqc),
            "TRP_IND_N15HSQC" | "TRPINDN15HSQC" => Some(Self::TrpIndN15Hsqc),
            _ => None,
        }
    }

    /// Physical `(1H_min..1H_max, indirect_min..indirect_max)` chemical shift window in ppm.
    #[inline]
    pub const fn spectral_window(self) -> ((f64, f64), (f64, f64)) {
        match self {
            Self::AromC13Hsqc | Self::AromHcchTocsy => ((4.8, 9.45), (108.0, 143.5)),
            Self::Hbcbcgcdhd | Self::Hbcbcgcdcehe => ((4.8, 9.45), (24.0, 48.0)),
            Self::HisLrN15Hmqc => ((6.0, 9.35), (155.0, 265.0)),
            Self::TrpIndN15Hsqc => ((8.8, 12.5), (120.0, 140.0)),
        }
    }
}

// ============================================================================
// 1. Histidine Imidazole Tautomeric, Exchange & Metal-Coordination State Model
// ============================================================================

/// Tautomeric and protonation state of a histidine imidazole ring in solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HisTautomer {
    /// Neutral $\tau$ tautomer (`Nε2-H` protonated, `Nδ1` unprotonated):
    /// `ND1 ~ 245–255 ppm` ($\beta$-type `=N-`), `NE2 ~ 165–175 ppm` ($\alpha$-type `>NH`), `CD2 < 122 ppm`.
    Ne2H,
    /// Neutral $\pi$ tautomer (`Nδ1-H` protonated, `Nε2` unprotonated):
    /// `ND1 ~ 165–178 ppm` ($\alpha$-type `>NH`), `NE2 ~ 245–255 ppm` ($\beta$-type `=N-`), `CD2 ~ 124–128 ppm`.
    Nd1H,
    /// Cationic imidazolium `+1` state (both `Nδ1-H` and `Nε2-H` protonated):
    /// `ND1 ~ 173–182 ppm`, `NE2 ~ 170–180 ppm`, `CD2 < 122 ppm`, `HE1 ~ 8.5–8.8 ppm`, upfield `CB`.
    DoublyProtonated,
}

impl HisTautomer {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ne2H => "Ne2H",
            Self::Nd1H => "Nd1H",
            Self::DoublyProtonated => "DoublyProtonated",
        }
    }

    pub const ALL: [Self; 3] = [Self::Ne2H, Self::Nd1H, Self::DoublyProtonated];

    /// Authoritative unified Pelton et al. (1993) + BMRB prior record shared with `crate::topology`.
    #[inline]
    pub const fn variant_priors(self) -> HisVariantPriors {
        match self {
            Self::Ne2H => HIS_NE2H_PRIORS,
            Self::Nd1H => HIS_ND1H_PRIORS,
            Self::DoublyProtonated => HIS_DOUBLY_PROTONATED_PRIORS,
        }
    }

    /// Canonical `(mean_ppm, std_ppm)` priors `(ND1, NE2, CD2, HD2, CE1, HE1, CB)` per Pelton et al. (1993),
    /// read directly from the single authoritative `HisVariantPriors` table in `crate::topology`.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub const fn shift_priors(
        self,
    ) -> (
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64),
        (f64, f64),
    ) {
        let vp = self.variant_priors();
        (vp.nd1, vp.ne2, vp.cd2, vp.hd2, vp.ce1, vp.he1, vp.cb)
    }
}

/// High-level chemical regime of a histidine imidazole ring:
/// - `Resolved(HisTautomer)`: well-resolved Pelton $\tau$ (`Ne2H`), $\pi$ (`Nd1H`), or cationic (`DoublyProtonated`) state anchored by $\ge 1$ imidazole `15N` observation (`ND1` or `NE2`) and passing the $\chi^2_{0.95}$ goodness-of-fit gate on both the winning pure state and the Pelton titration manifold.
/// - `FastExchangeUncommitted`: dynamic neutral $\tau \leftrightarrow \pi$ or neutral $\leftrightarrow$ cationic fast exchange (`ND1`/`NE2` coalesced into intermediate positions), `15N`-unanchored (`CH`/`CG`-only) ring evidence, or $\chi^2_{0.95}$ out-of-distribution misfit (`conf < 0.50`).
/// - `MetalCoordinatedCandidate`: `Zn(II)`/`Cu(I)`/`Cu(II)`-ligated imidazole detected either by `15N` shifts outside Pelton's neutral$\leftrightarrow$cation simplex envelope with orthogonal residual `d_orth_shift^2 >= 5.5`, or by an unprotonated `=N–M` nitrogen (`>= 194.0 ppm`, `f_prot_n >= 0.35`) paired with an unprotonated `HE1`/`CE1` `CH`-contradiction and large orthogonal Mahalanobis residual off `[0, 1]^2` (`d_orth_joint^2 >= 12.0` or `d_orth_shift^2 >= 8.0`; Barraud et al., *J. Biomol. NMR* 53, 95–108, 2012).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HisStateClassification {
    Resolved(HisTautomer),
    FastExchangeUncommitted,
    MetalCoordinatedCandidate,
}

impl HisStateClassification {
    #[inline]
    pub const fn resolved_tautomer(self) -> Option<HisTautomer> {
        match self {
            Self::Resolved(t) => Some(t),
            Self::FastExchangeUncommitted | Self::MetalCoordinatedCandidate => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resolved(HisTautomer::Ne2H) => "Ne2H",
            Self::Resolved(HisTautomer::Nd1H) => "Nd1H",
            Self::Resolved(HisTautomer::DoublyProtonated) => "DoublyProtonated",
            Self::FastExchangeUncommitted => "FastExchangeUncommitted",
            Self::MetalCoordinatedCandidate => "MetalCoordinatedCandidate",
        }
    }
}

/// Experimental shifts, sample conditions, and cross-peak evidence for classifying a histidine ring.
#[derive(Clone, Debug, Default)]
pub struct HisTautomerEvidence {
    pub nd1_ppm: Option<f64>,
    pub ne2_ppm: Option<f64>,
    /// Quaternary `13CG` shift (`~135.5 ppm` `Ne2H` $\tau$, `~127.2 ppm` `Nd1H` $\pi$, `~129.8 ppm` `DoublyProtonated`;
    /// Reynolds et al., *J. Am. Chem. Soc.* 95, 328–331, 1973; Sudmeier et al., *J. Magn. Reson.* 163, 194–198, 2003).
    pub cg_ppm: Option<f64>,
    pub cd2_ppm: Option<f64>,
    pub hd2_ppm: Option<f64>,
    pub ce1_ppm: Option<f64>,
    pub he1_ppm: Option<f64>,
    pub cb_ppm: Option<f64>,
    pub sample_ph: Option<f64>,
    pub metal_coordinated_hint: bool,
}

#[inline]
fn gauss_logpdf(x: f64, mu: f64, sd: f64) -> f64 {
    let z = (x - mu) / sd;
    -0.5 * z * z - (sd * TAU.sqrt()).ln()
}

fn hh_pka_zscore(f_prot: f64, sample_ph: f64) -> f64 {
    // Weak Henderson-Hasselbalch prior around average protein His pKa = 6.3 +- 0.80
    // (Pelton et al., Protein Sci. 1993; Edgcomb & Murphy, Proteins 2002: protein His pKa ~ 4.8..8.0).
    // Evaluate over a +-0.015 protonation-fraction tolerance band so boundary states
    // f_prot = 0 (high pH) and f_prot = 1 (low pH) incur zero artificial log-odds blowup.
    const PKA_MEAN: f64 = 6.30;
    const PKA_SD: f64 = 0.80;
    const BAND: f64 = 0.015;

    let f_lo = (f_prot - BAND).clamp(1.0e-4, 1.0 - 1.0e-4);
    let f_hi = (f_prot + BAND).clamp(1.0e-4, 1.0 - 1.0e-4);
    let pka_at = |f: f64| sample_ph - ((1.0 - f) / f).log10();

    let pka_min = if f_prot <= BAND {
        f64::NEG_INFINITY
    } else {
        pka_at(f_lo)
    };
    let pka_max = if f_prot >= 1.0 - BAND {
        f64::INFINITY
    } else {
        pka_at(f_hi)
    };

    if PKA_MEAN < pka_min {
        (pka_min - PKA_MEAN) / PKA_SD
    } else if PKA_MEAN > pka_max {
        (PKA_MEAN - pka_max) / PKA_SD
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct PeltonManifoldFit {
    /// MAP protonated imidazolium fraction `f_prot in [0.0, 1.0]` including optional pH prior.
    f_prot_map: f64,
    /// MAP neutral tau (`Ne2H`) fraction `p_tau in [0.0, 1.0]` within the `(1 - f_prot)` neutral pool.
    p_tau_map: f64,
    /// Shift-only minimizer `(f_prot, p_tau)` on `[0, 1]^2`.
    f_prot_shift: f64,
    p_tau_shift: f64,
    /// Minimum orthogonal Mahalanobis residual `d_orth^2` off the pure shift manifold `(f_prot, p_tau) in [0, 1]^2`.
    d_orth_sq_shift: f64,
    /// Minimum joint `(shift + Henderson-Hasselbalch pH)` residual `d_orth_joint^2` off `[0, 1]^2`.
    d_orth_sq_joint: f64,
    /// Minimum orthogonal Mahalanobis residual `d_orth_n^2` of `(ND1, NE2)` off the `15N` Pelton simplex.
    d_orth_sq_n: Option<f64>,
    /// Protonated fraction `f_prot_n` implied by `(ND1, NE2)` alone (`None` if neither `15N` is observed).
    f_prot_n_only: Option<f64>,
    /// Number of observed ring shift channels among `(ND1, NE2, CG, CD2, HD2, CE1, HE1)`.
    n_ring_obs: usize,
}

/// Fit Pelton's two-site neutral (`tau <-> pi`) <-> cationic (`HisH+`) fast-exchange manifold:
/// `mu(f_prot, p_tau) = (1 - f_prot) * (p_tau * mu_tau + (1 - p_tau) * mu_pi) + f_prot * mu_cat`
/// across `(ND1, NE2, CG, CD2, HD2, CE1, HE1)` (plus weak `CB` tie-breaking in the MAP objective)
/// and optional Henderson-Hasselbalch `sample_ph` prior (`pKa = 6.3 +- 0.8`).
fn fit_pelton_titration_manifold(ev: &HisTautomerEvidence) -> PeltonManifoldFit {
    let vp_tau = HIS_NE2H_PRIORS;
    let vp_pi = HIS_ND1H_PRIORS;
    let vp_cat = HIS_DOUBLY_PROTONATED_PRIORS;
    let ph_opt = ev.sample_ph.filter(|v| v.is_finite());

    type HisRingChannel = (Option<f64>, (f64, f64), (f64, f64), (f64, f64));
    let ring_channels: [HisRingChannel; 7] = [
        (ev.nd1_ppm, vp_tau.nd1, vp_pi.nd1, vp_cat.nd1),
        (ev.ne2_ppm, vp_tau.ne2, vp_pi.ne2, vp_cat.ne2),
        (ev.cg_ppm, vp_tau.cg, vp_pi.cg, vp_cat.cg),
        (ev.cd2_ppm, vp_tau.cd2, vp_pi.cd2, vp_cat.cd2),
        (ev.hd2_ppm, vp_tau.hd2, vp_pi.hd2, vp_cat.hd2),
        (ev.ce1_ppm, vp_tau.ce1, vp_pi.ce1, vp_cat.ce1),
        (ev.he1_ppm, vp_tau.he1, vp_pi.he1, vp_cat.he1),
    ];
    let n_ring_obs = ring_channels.iter().filter(|c| c.0.is_some()).count();

    let eval_point = |f_prot: f64, p_tau: f64| -> (f64, f64, f64, f64) {
        let w_tau = (1.0 - f_prot) * p_tau;
        let w_pi = (1.0 - f_prot) * (1.0 - p_tau);
        let w_cat = f_prot;

        let mut d_sq_ring = 0.0;
        let mut d_sq_n = 0.0;
        for (idx, &(obs_opt, t_p, p_p, c_p)) in ring_channels.iter().enumerate() {
            if let Some(obs) = obs_opt {
                let mu = w_tau * t_p.0 + w_pi * p_p.0 + w_cat * c_p.0;
                let var = w_tau * t_p.1 * t_p.1 + w_pi * p_p.1 * p_p.1 + w_cat * c_p.1 * c_p.1;
                let z2 = (obs - mu) * (obs - mu) / var.max(1.0e-6);
                d_sq_ring += z2;
                if idx < 2 {
                    d_sq_n += z2;
                }
            }
        }

        let cb_term = if let Some(cb) = ev.cb_ppm {
            let mu_cb = w_tau * vp_tau.cb.0 + w_pi * vp_pi.cb.0 + w_cat * vp_cat.cb.0;
            let var_cb = w_tau * vp_tau.cb.1 * vp_tau.cb.1
                + w_pi * vp_pi.cb.1 * vp_pi.cb.1
                + w_cat * vp_cat.cb.1 * vp_cat.cb.1;
            0.25 * (cb - mu_cb) * (cb - mu_cb) / var_cb.max(1.0e-6)
        } else {
            0.0
        };

        // Weak neutral tau:pi ~ 0.64:0.36 preference from Pelton/BMRB when neutral tautomer is underdetermined
        let z_tau_prior = (p_tau - 0.64) / 0.45;
        let map_prior_pen = (1.0 - f_prot) * 0.12 * z_tau_prior * z_tau_prior;

        let (ph_joint_pen, ph_map_pen) = if let Some(ph) = ph_opt {
            let z_ph = hh_pka_zscore(f_prot, ph);
            (0.65 * z_ph * z_ph, 0.45 * z_ph * z_ph)
        } else {
            (0.0, 0.0)
        };

        let d_joint = d_sq_ring + ph_joint_pen;
        let map_obj = d_sq_ring + cb_term + ph_map_pen + map_prior_pen;
        (d_sq_ring, d_joint, map_obj, d_sq_n)
    };

    let mut d_orth_sq_shift = f64::INFINITY;
    let mut f_prot_shift = 0.0;
    let mut p_tau_shift = 0.64;
    let mut d_orth_sq_joint = f64::INFINITY;
    let mut best_map_obj = f64::INFINITY;
    let mut f_prot_map = 0.0;
    let mut p_tau_map = 0.64;

    let has_n = ev.nd1_ppm.is_some() || ev.ne2_ppm.is_some();
    let mut best_n_sq = f64::INFINITY;
    let mut best_f_n = 0.0;
    let mut best_p_n = 0.64;

    // Coarse 41 x 41 grid (`0.025` step) + local coordinate refinement (`0.005` step) on `[0, 1]^2`
    const N_GRID: usize = 40;
    for i in 0..=N_GRID {
        let f = (i as f64) / (N_GRID as f64);
        for j in 0..=N_GRID {
            let p = (j as f64) / (N_GRID as f64);
            let (d_shift, d_joint, map_obj, d_n) = eval_point(f, p);
            if d_shift < d_orth_sq_shift {
                d_orth_sq_shift = d_shift;
                f_prot_shift = f;
                p_tau_shift = p;
            }
            if d_joint < d_orth_sq_joint {
                d_orth_sq_joint = d_joint;
            }
            if map_obj < best_map_obj {
                best_map_obj = map_obj;
                f_prot_map = f;
                p_tau_map = p;
            }
            if has_n && d_n < best_n_sq {
                best_n_sq = d_n;
                best_f_n = f;
                best_p_n = p;
            }
        }
    }

    for &(f_c, p_c) in &[
        (f_prot_map, p_tau_map),
        (f_prot_shift, p_tau_shift),
        (best_f_n, best_p_n),
    ] {
        for di in -5..=5_i32 {
            let f = (f_c + (di as f64) * 0.005).clamp(0.0, 1.0);
            for dj in -5..=5_i32 {
                let p = (p_c + (dj as f64) * 0.005).clamp(0.0, 1.0);
                let (d_shift, d_joint, map_obj, d_n) = eval_point(f, p);
                if d_shift < d_orth_sq_shift {
                    d_orth_sq_shift = d_shift;
                    f_prot_shift = f;
                    p_tau_shift = p;
                }
                d_orth_sq_joint = d_orth_sq_joint.min(d_joint);
                if map_obj < best_map_obj {
                    best_map_obj = map_obj;
                    f_prot_map = f;
                    p_tau_map = p;
                }
                if has_n && d_n < best_n_sq {
                    best_n_sq = d_n;
                    best_f_n = f;
                }
            }
        }
    }

    PeltonManifoldFit {
        f_prot_map,
        p_tau_map,
        f_prot_shift,
        p_tau_shift,
        d_orth_sq_shift,
        d_orth_sq_joint,
        d_orth_sq_n: has_n.then_some(best_n_sq),
        f_prot_n_only: has_n.then_some(best_f_n),
        n_ring_obs,
    }
}

/// Squared Mahalanobis distance `d_M^2`, maximum per-channel `|z|`, and observed channel count `m`
/// to a single pure Pelton state (`Ne2H`, `Nd1H`, or `DoublyProtonated`).
fn pure_his_state_mahalanobis(ev: &HisTautomerEvidence, state: HisTautomer) -> (f64, f64, usize) {
    let vp = state.variant_priors();
    let channels: [(Option<f64>, (f64, f64)); 7] = [
        (ev.nd1_ppm, vp.nd1),
        (ev.ne2_ppm, vp.ne2),
        (ev.cg_ppm, vp.cg),
        (ev.cd2_ppm, vp.cd2),
        (ev.hd2_ppm, vp.hd2),
        (ev.ce1_ppm, vp.ce1),
        (ev.he1_ppm, vp.he1),
    ];
    let mut d_m_sq = 0.0;
    let mut z_max = 0.0_f64;
    let mut m = 0usize;
    for &(obs_opt, (mu, sd)) in &channels {
        if let Some(obs) = obs_opt {
            let z = ((obs - mu) / sd).abs();
            d_m_sq += z * z;
            z_max = z_max.max(z);
            m += 1;
        }
    }
    (d_m_sq, z_max, m)
}

fn raw_pelton_three_state_posterior(ev: &HisTautomerEvidence) -> ([f64; 3], HisTautomer, f64) {
    let log_priors = if let Some(ph) = ev.sample_ph.filter(|v| v.is_finite()) {
        // Weak Henderson-Hasselbalch prior around pKa = 6.3 +- 0.80 so basic/acidic pH informs
        // neutral vs cationic odds without overriding genuine salt-bridged or buried pKa shifts.
        let z_neut = hh_pka_zscore(0.0, ph);
        let z_cat = hh_pka_zscore(1.0, ph);
        let lp_neut = -0.225 * z_neut * z_neut;
        let lp_cat = -0.225 * z_cat * z_cat;
        [0.64_f64.ln() + lp_neut, 0.36_f64.ln() + lp_neut, lp_cat]
    } else {
        [0.45_f64.ln(), 0.25_f64.ln(), 0.30_f64.ln()]
    };

    let mut log_post = [0.0; 3];
    for (idx, &state) in HisTautomer::ALL.iter().enumerate() {
        let vp = state.variant_priors();
        let mut lp = log_priors[idx];

        if let Some(v) = ev.nd1_ppm {
            lp += gauss_logpdf(v, vp.nd1.0, vp.nd1.1);
        }
        if let Some(v) = ev.ne2_ppm {
            lp += gauss_logpdf(v, vp.ne2.0, vp.ne2.1);
        }
        if let Some(v) = ev.cg_ppm {
            lp += gauss_logpdf(v, vp.cg.0, vp.cg.1);
        }
        if let Some(v) = ev.cd2_ppm {
            lp += gauss_logpdf(v, vp.cd2.0, vp.cd2.1);
        }
        if let Some(v) = ev.hd2_ppm {
            lp += gauss_logpdf(v, vp.hd2.0, vp.hd2.1);
        }
        if let Some(v) = ev.ce1_ppm {
            lp += gauss_logpdf(v, vp.ce1.0, vp.ce1.1);
        }
        if let Some(v) = ev.he1_ppm {
            lp += gauss_logpdf(v, vp.he1.0, vp.he1.1);
        }
        if let Some(v) = ev.cb_ppm {
            lp += 0.35 * gauss_logpdf(v, vp.cb.0, vp.cb.1);
        }
        log_post[idx] = lp;
    }

    let max_lp = log_post.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs = [0.0; 3];
    let mut sum = 0.0;
    for i in 0..3 {
        probs[i] = (log_post[i] - max_lp).exp();
        sum += probs[i];
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p = (*p / sum).clamp(0.0, 1.0);
        }
    } else {
        probs = [1.0 / 3.0; 3];
    }
    let mut best_i = 0;
    for i in 1..3 {
        if probs[i] > probs[best_i] {
            best_i = i;
        }
    }
    (probs, HisTautomer::ALL[best_i], probs[best_i])
}

/// Classify histidine imidazole state (`Resolved(HisTautomer)`, `FastExchangeUncommitted`, or
/// `MetalCoordinatedCandidate`) and return the calibrated static-tautomer commitment confidence
/// in `[0.0, 1.0]` (`< 0.50` on fast exchange or out-of-distribution states, `< 0.60` on metal
/// coordination, so callers and auto-apply widgets never over-commit non-static states).
///
/// Provenance:
/// - Pelton, Torchia, Meadow & Roseman, *Protein Sci.* 2, 543–558 (1993): two-site fast-exchange
///   titration surface `mu(f_prot, p_tau)` across `(ND1, NE2, CG, CD2, HD2, CE1, HE1)`.
/// - Barraud, Schubert, Allain & Nietlispach, *J. Biomol. NMR* 53, 95–108 (2012): metal-ligated
///   histidine `15N`/`1H`/`13C` orthogonal electronic departure off the protonation manifold.
pub fn classify_his_state(ev: &HisTautomerEvidence) -> (HisStateClassification, f64) {
    let (_probs, best_taut, raw_post) = raw_pelton_three_state_posterior(ev);

    // 1. Explicit metal-coordination hint from sample/domain metadata
    if ev.metal_coordinated_hint {
        return (
            HisStateClassification::MetalCoordinatedCandidate,
            raw_post.min(0.58),
        );
    }

    let fit = fit_pelton_titration_manifold(ev);

    // 1b. Zero observed imidazole ring shifts (`ND1, NE2, CG, CD2, HD2, CE1, HE1` all `None`):
    //     never commit purely from `sample_ph` or `CB` priors!
    if fit.n_ring_obs == 0 {
        return (HisStateClassification::FastExchangeUncommitted, 0.0);
    }

    // 1c. Require >= 1 imidazole `15N` observation (`ND1` or `NE2`) to emit `Resolved`:
    //     `CH`-only (`CD2`, `HD2`, `CE1`, `HE1`) or `CG`-only ring shifts without a direct `15N`
    //     anchor cannot separate fast tautomer/protonation exchange or ring-current shifts from a
    //     static Pelton tautomer -> abstain (`FastExchangeUncommitted`, `conf <= 0.45 < 0.50`).
    if ev.nd1_ppm.is_none() && ev.ne2_ppm.is_none() {
        return (
            HisStateClassification::FastExchangeUncommitted,
            raw_post.min(0.45),
        );
    }

    // 2. Metal-coordination detection conditioned strictly on statistically significant orthogonal
    //    Mahalanobis residual `d_orth^2` off the entire two-site neutral<->cation titration manifold
    //    (zero ad-hoc 15N split windows):
    //    (a) 15N shift outside Pelton's neutral<->cation simplex envelope (`> 258.5 ppm` downfield
    //        beyond neutral `=N-` `249.5 +- 3.8`, `< 156.0 ppm`, or `ND1 + NE2 > 438.0 ppm` with `d_orth^2 >= 5.5`), OR
    //    (b) Significant multi-channel orthogonal Mahalanobis residual off `[0, 1]^2` (`d_orth_joint^2 >= 12.0`
    //        or `d_orth_shift^2 >= 8.0`) where at least one unprotonated metal-bound `=N-M` nitrogen sits in
    //        `>= 194.0 ppm` (`f_prot_n >= 0.35`; Barraud et al. 2012, *J. Biomol. NMR* 53:95-108) WITHOUT
    //        the accompanying cationic protonation shift on `HE1`/`CE1`. Symmetric protonated `-NH+` pairs
    //        (`both < 194.0 ppm`, e.g. canonical `HisH2+` `~176/174 ppm` with aromatic ring-current `HE1`)
    //        never trigger `MetalCoordinatedCandidate`.
    let n_envelope_exceeded = [ev.nd1_ppm, ev.ne2_ppm]
        .into_iter()
        .flatten()
        .any(|n| !(156.0..=258.5).contains(&n))
        || matches!((ev.nd1_ppm, ev.ne2_ppm), (Some(a), Some(b)) if a + b > 438.0 || a + b < 326.0);

    if n_envelope_exceeded && fit.d_orth_sq_shift >= 5.5 {
        return (
            HisStateClassification::MetalCoordinatedCandidate,
            raw_post.min(0.55),
        );
    }

    let has_nm_coord_nitrogen = [ev.nd1_ppm, ev.ne2_ppm]
        .into_iter()
        .flatten()
        .any(|n| n >= 194.0);
    if let Some(f_n) = fit.f_prot_n_only {
        let exp_he1_at_fn = (1.0 - f_n) * 7.815 + f_n * HIS_DOUBLY_PROTONATED_PRIORS.he1.0;
        let exp_ce1_at_fn = (1.0 - f_n) * 137.92 + f_n * HIS_DOUBLY_PROTONATED_PRIORS.ce1.0;
        let ce1_corroborates_protonation = ev.ce1_ppm.is_some_and(|c| c <= exp_ce1_at_fn + 0.45);
        let unprotonated_ch_contradiction = has_nm_coord_nitrogen
            && (f_n >= 0.35)
            && !ce1_corroborates_protonation
            && (ev
                .he1_ppm
                .is_some_and(|h| (h - exp_he1_at_fn) / 0.28 <= -2.35)
                || ev
                    .ce1_ppm
                    .is_some_and(|c| (c - exp_ce1_at_fn) / 1.60 >= 2.10));

        if unprotonated_ch_contradiction
            && (fit.d_orth_sq_joint >= 12.0 || fit.d_orth_sq_shift >= 8.0)
        {
            return (
                HisStateClassification::MetalCoordinatedCandidate,
                raw_post.min(0.56),
            );
        }
    }

    // 3. Intermediate fast-exchange abstention zone on Pelton's titration surface:
    //    - Protonation exchange (`neutral <-> HisH+`): `0.18 < f_prot_map < 0.82`
    //    - Neutral tautomer exchange (`Nd1H <-> Ne2H` when predominantly neutral): `f_prot_map <= 0.18 && 0.25 < p_tau_map < 0.75`
    let in_prot_exchange = fit.f_prot_map > 0.18 && fit.f_prot_map < 0.82;
    let in_tau_pi_exchange = fit.f_prot_map <= 0.18 && fit.p_tau_map > 0.25 && fit.p_tau_map < 0.75;
    if in_prot_exchange || in_tau_pi_exchange {
        let dist_edge = if in_prot_exchange {
            ((fit.f_prot_map - 0.50).abs() / 0.32).clamp(0.0, 1.0)
        } else {
            ((fit.p_tau_map - 0.50).abs() / 0.25).clamp(0.0, 1.0)
        };
        let conf = (0.30 + 0.16 * dist_edge).clamp(0.25, 0.48);
        return (HisStateClassification::FastExchangeUncommitted, conf);
    }

    // 4. Pure-state & Pelton-chord chi2_0.95 goodness-of-fit gate (`d_M^2` against winning pure state
    //    `Ne2H`, `Nd1H`, or `DoublyProtonated`, plus orthogonal chord residual `d_orth_sq_shift` and
    //    `15N` simplex residual `d_orth_sq_n`): any statistically significant misfit (`p < 0.05`) or
    //    uncorroborated strong `sample_ph` discordance abstains as `FastExchangeUncommitted` (`conf < 0.50`).
    #[inline]
    const fn chi2_95_critical(df: usize) -> f64 {
        match df {
            0 | 1 => 3.8415,
            2 => 5.9915,
            3 => 7.8147,
            4 => 9.4877,
            5 => 11.0705,
            6 => 12.5916,
            _ => 14.0671,
        }
    }

    let (d_m_sq, z_max, m_obs) = pure_his_state_mahalanobis(ev, best_taut);
    let chi2_95_pure = chi2_95_critical(m_obs);

    let active_chord_params = if fit.n_ring_obs <= 2 {
        1usize
    } else {
        let f_interior = usize::from(fit.f_prot_shift > 0.05 && fit.f_prot_shift < 0.95);
        let p_interior = usize::from(
            fit.f_prot_shift < 0.95 && fit.p_tau_shift > 0.05 && fit.p_tau_shift < 0.95,
        );
        f_interior + p_interior
    };
    let df_orth = fit.n_ring_obs.saturating_sub(active_chord_params).max(1);
    let chi2_95_orth = chi2_95_critical(df_orth);
    let n_pair_orth_exceeded = ev.nd1_ppm.is_some()
        && ev.ne2_ppm.is_some()
        && fit.d_orth_sq_n.is_some_and(|dn| dn > 3.8415);

    let pure_f_prot = if best_taut == HisTautomer::DoublyProtonated {
        1.0
    } else {
        0.0
    };
    let z_ph_pure = ev
        .sample_ph
        .filter(|v| v.is_finite())
        .map_or(0.0, |ph| hh_pka_zscore(pure_f_prot, ph));
    let vp_best = best_taut.variant_priors();
    let protonation_diagnostic_c = match best_taut {
        HisTautomer::DoublyProtonated => {
            ev.ce1_ppm.is_some_and(|c| c <= 136.5)
                || ev.cg_ppm.is_some_and(|g| (128.3..=131.5).contains(&g))
        }
        HisTautomer::Ne2H | HisTautomer::Nd1H => {
            ev.cg_ppm
                .is_some_and(|v| ((v - vp_best.cg.0) / vp_best.cg.1).abs() <= 1.5)
                || ev
                    .cd2_ppm
                    .is_some_and(|v| ((v - vp_best.cd2.0) / vp_best.cd2.1).abs() <= 1.5)
                || ev
                    .ce1_ppm
                    .is_some_and(|v| ((v - vp_best.ce1.0) / vp_best.ce1.1).abs() <= 1.5)
        }
    };
    let corroborated_pure_n_and_c = ev
        .nd1_ppm
        .is_some_and(|v| ((v - vp_best.nd1.0) / vp_best.nd1.1).abs() <= 1.5)
        && ev
            .ne2_ppm
            .is_some_and(|v| ((v - vp_best.ne2.0) / vp_best.ne2.1).abs() <= 1.5)
        && protonation_diagnostic_c;
    let ph_shift_contradiction = !corroborated_pure_n_and_c && z_ph_pure > 2.5 && d_m_sq >= 2.5;

    if m_obs > 0
        && (z_max > 3.50
            || d_m_sq > chi2_95_pure
            || fit.d_orth_sq_shift > chi2_95_orth
            || n_pair_orth_exceeded
            || ph_shift_contradiction)
    {
        let excess = (d_m_sq - chi2_95_pure)
            .max(fit.d_orth_sq_shift - chi2_95_orth)
            .max(fit.d_orth_sq_n.map_or(0.0, |dn| dn - 3.8415))
            .max((z_max - 3.50) * 4.0)
            .max(if ph_shift_contradiction {
                z_ph_pure - 2.5
            } else {
                0.0
            })
            .max(0.0);
        let conf = (0.45 * (-0.08 * excess).exp()).clamp(0.18, 0.46);
        return (HisStateClassification::FastExchangeUncommitted, conf);
    }

    let calibrated = raw_post.clamp(0.0, 1.0);
    (HisStateClassification::Resolved(best_taut), calibrated)
}

/// Classify histidine `HisTautomer` and compute its calibrated static-tautomer posterior in `[0.0, 1.0]`
/// from Pelton et al. (1993) `(ND1, NE2, CG)` shifts and corroborating `CD2`/`CE1`/`HE1`/`CB` shifts.
/// When `classify_his_state` detects `FastExchangeUncommitted` (`conf < 0.50`) or
/// `MetalCoordinatedCandidate` (`conf < 0.60`), callers never over-commit non-static states.
pub fn classify_his_tautomer(ev: &HisTautomerEvidence) -> (HisTautomer, f64) {
    let (_probs, best_taut, _raw_post) = raw_pelton_three_state_posterior(ev);
    let (_state, calibrated_conf) = classify_his_state(ev);
    (best_taut, calibrated_conf)
}

// ============================================================================
// 2. All-Atom Aromatic Reference Synthesizer & Volumetric 2D/3D Simulators
// ============================================================================

/// Deterministically assign canonical `HisTautomer` states across `HIS` residues in `sequence`
/// so all three Pelton states (`Ne2H`, `Nd1H`, `DoublyProtonated`) are exercised.
pub fn default_his_tautomer_map(sequence: &[String], seed: u64) -> BTreeMap<usize, HisTautomer> {
    let mut map = BTreeMap::new();
    let mut his_count = 0usize;
    for (k, r) in sequence.iter().enumerate() {
        if r == "HIS" {
            let idx = (his_count.wrapping_add((seed as usize) % 3)) % 3;
            map.insert(k, HisTautomer::ALL[idx]);
            his_count += 1;
        }
    }
    map
}

/// Complete any missing aromatic `(1H, 13C, 15N)` and `CB` chemical shifts in `base` from
/// empirical BMRB distributions (`SidechainPriors`) and Pelton `HisTautomer` shift regimes,
/// enforcing symmetric `PHE`/`TYR` fast-flip degeneracy (`CD1=CD2`, `CE1=CE2`).
pub fn synthesize_aromatic_reference(
    base: &Reference,
    priors: &SidechainPriors,
    his_tautomers: &BTreeMap<usize, HisTautomer>,
    seed: u64,
) -> Reference {
    let mut rng = Rng::new(seed ^ 0xA204_A71C_9E37_79B9);
    let mut out = base.clone();

    for (k, res) in base.sequence.iter().enumerate() {
        if !matches!(res.as_str(), "PHE" | "TYR" | "TRP" | "HIS") {
            continue;
        }
        let Some(topo) = residue_topology(res) else {
            continue;
        };

        // Ensure CB exists on the physical ground-truth molecule if absent from partial STAR file
        out.shifts.entry((k, "CB".to_string())).or_insert_with(|| {
            let (mu_cb, sd_cb) = if res == "HIS" {
                let st = his_tautomers.get(&k).copied().unwrap_or(HisTautomer::Ne2H);
                st.shift_priors().6
            } else {
                priors.prior(res, "CB").unwrap_or((38.0, 1.8))
            };
            mu_cb + rng.normal(0.0, (sd_cb * 0.55).min(1.1))
        });

        if res == "HIS" {
            let st = his_tautomers.get(&k).copied().unwrap_or(HisTautomer::Ne2H);
            let (p_nd1, p_ne2, p_cd2, p_hd2, p_ce1, p_he1, _) = st.shift_priors();
            let sample_or_keep = |out_map: &mut BTreeMap<(usize, String), f64>,
                                  rng_ref: &mut Rng,
                                  atom: &str,
                                  (mu, sd): (f64, f64),
                                  scale: f64| {
                out_map
                    .entry((k, atom.to_string()))
                    .or_insert_with(|| mu + rng_ref.normal(0.0, sd * scale));
            };
            sample_or_keep(&mut out.shifts, &mut rng, "ND1", p_nd1, 0.45);
            sample_or_keep(&mut out.shifts, &mut rng, "NE2", p_ne2, 0.45);
            // For DoublyProtonated, ensure ND1 remains > NE2 by at least ~1.8 ppm so HMQC pair is realistic
            if st == HisTautomer::DoublyProtonated {
                if let (Some(&nd1), Some(&ne2)) = (
                    out.shifts.get(&(k, "ND1".to_string())),
                    out.shifts.get(&(k, "NE2".to_string())),
                ) {
                    if (nd1 - ne2).abs() < 1.8 {
                        let mid = (0.5 * (nd1 + ne2)).clamp(173.0, 178.5);
                        out.shifts.insert((k, "ND1".to_string()), mid + 1.6);
                        out.shifts.insert((k, "NE2".to_string()), mid - 1.6);
                    }
                }
            }
            sample_or_keep(&mut out.shifts, &mut rng, "CD2", p_cd2, 0.62);
            sample_or_keep(&mut out.shifts, &mut rng, "HD2", p_hd2, 0.62);
            sample_or_keep(&mut out.shifts, &mut rng, "CE1", p_ce1, 0.62);
            sample_or_keep(&mut out.shifts, &mut rng, "HE1", p_he1, 0.62);
            continue;
        }

        // PHE, TYR, TRP aromatic CH groups
        for chg in topo.aromatic_groups() {
            if let std::collections::btree_map::Entry::Vacant(e) =
                out.shifts.entry((k, chg.carbon.to_string()))
            {
                if let Some((mu_c, sd_c)) = priors.prior(res, chg.carbon) {
                    let cv = mu_c + rng.normal(0.0, (sd_c * 0.65).min(1.10));
                    e.insert(cv);
                }
            }
            let p_name = chg.protons[0];
            if let std::collections::btree_map::Entry::Vacant(e) =
                out.shifts.entry((k, p_name.to_string()))
            {
                if let Some((mu_h, sd_h)) = priors.prior(res, p_name) {
                    let hv = mu_h + rng.normal(0.0, (sd_h * 0.65).min(0.28));
                    e.insert(hv);
                }
            }
        }

        // Enforce fast-flip aromatic symmetry on PHE and TYR (CD1=CD2, HD1=HD2, CE1=CE2, HE1=HE2)
        for pg in topo.prochiral_groups {
            if let ProchiralGroup::SymmetricAromatic {
                carbon_a,
                proton_a,
                carbon_b,
                proton_b,
            } = pg
            {
                if let Some(&ca) = out.shifts.get(&(k, (*carbon_a).to_string())) {
                    out.shifts.insert((k, (*carbon_b).to_string()), ca);
                }
                if let Some(&ha) = out.shifts.get(&(k, (*proton_a).to_string())) {
                    out.shifts.insert((k, (*proton_b).to_string()), ha);
                }
            }
        }

        // Ensure PHE CD*, CE*, CZ are not accidentally generated within < 0.012 ppm H AND < 0.12 ppm C simultaneously
        if res == "PHE" {
            let hd = out
                .shifts
                .get(&(k, "HD1".to_string()))
                .copied()
                .unwrap_or(7.10);
            let he = out
                .shifts
                .get(&(k, "HE1".to_string()))
                .copied()
                .unwrap_or(7.02);
            let hz = out
                .shifts
                .get(&(k, "HZ".to_string()))
                .copied()
                .unwrap_or(6.93);
            if (hd - he).abs() < 0.03 {
                let nudged = he - 0.055;
                out.shifts.insert((k, "HE1".to_string()), nudged);
                out.shifts.insert((k, "HE2".to_string()), nudged);
            }
            let he_now = out.shifts[&(k, "HE1".to_string())];
            if (hz - he_now).abs() < 0.03 || (hz - hd).abs() < 0.03 {
                out.shifts.insert((k, "HZ".to_string()), hz - 0.075);
            }
        }

        // TRP indole NH (NE1, HE1)
        if res == "TRP" {
            out.shifts.entry((k, "NE1".to_string())).or_insert_with(|| {
                let (mu_n, sd_n) = priors.prior("TRP", "NE1").unwrap_or((129.08, 1.85));
                mu_n + rng.normal(0.0, sd_n * 0.55)
            });
            out.shifts.entry((k, "HE1".to_string())).or_insert_with(|| {
                let (mu_h, sd_h) = priors.prior("TRP", "HE1").unwrap_or((10.05, 0.35));
                mu_h + rng.normal(0.0, sd_h * 0.55)
            });
        }
    }

    out
}

/// Configuration for simulating noisy volumetric 2D/3D aromatic spectra and peak lists.
#[derive(Clone, Debug)]
pub struct AromaticSimConfig {
    pub include_c13_hsqc: bool,
    pub include_hbcbcgcdhd: bool,
    pub include_hbcbcgcdcehe: bool,
    pub include_hcch_tocsy: bool,
    pub include_his_lr_hmqc: bool,
    pub include_trp_indole: bool,
    /// Optional weak `His` `HE1` relay in `(HB)CB(CGCDCE)HE`.
    pub include_weak_his_he1_yamazaki: bool,
    /// Per-experiment chemical-shift jitter (ppm) modeling sample/referencing drift.
    pub jitter_h_ppm: f64,
    pub jitter_c_ppm: f64,
    pub jitter_n_ppm: f64,
    /// Bernoulli missing-peak dropout probability on relay cross-peaks.
    pub p_missing_peak: f64,
    /// Volumetric grid dimensions & linewidths.
    pub nx_2d: usize,
    pub ny_2d: usize,
    pub ny_hmqc: usize,
    pub nx_3d: usize,
    pub ny_3d: usize,
    pub nz_3d: usize,
    pub lw_h_ppm: f64,
    pub lw_c_arom_ppm: f64,
    pub lw_cb_ppm: f64,
    pub lw_n_hmqc_ppm: f64,
    /// Thermal white Gaussian noise standard deviation per voxel (`1x` aromatic peak height ~ `100.0`).
    pub noise_level: f32,
    /// Pseudo-Voigt Gaussian fraction in `[0.0, 1.0]`.
    pub gaussian_mix: f64,
    /// Fractional FT truncation sinc-sidelobe amplitude along indirect `F1` axis (tests picker suppression).
    pub sinc_sidelobe_frac: f64,
    pub seed: u64,
}

impl Default for AromaticSimConfig {
    fn default() -> Self {
        Self {
            include_c13_hsqc: true,
            include_hbcbcgcdhd: true,
            include_hbcbcgcdcehe: true,
            include_hcch_tocsy: true,
            include_his_lr_hmqc: true,
            include_trp_indole: true,
            include_weak_his_he1_yamazaki: true,
            jitter_h_ppm: 0.004,
            jitter_c_ppm: 0.022,
            jitter_n_ppm: 0.06,
            p_missing_peak: 0.02,
            nx_2d: 512,
            ny_2d: 256,
            ny_hmqc: 384,
            nx_3d: 192,
            ny_3d: 128,
            nz_3d: 192,
            lw_h_ppm: 0.022,
            lw_c_arom_ppm: 0.22,
            lw_cb_ppm: 0.16,
            lw_n_hmqc_ppm: 0.48,
            noise_level: 1.5,
            gaussian_mix: 0.72,
            sinc_sidelobe_frac: 0.045,
            seed: 2026,
        }
    }
}

/// Complete simulated volumetric 2D/3D aromatic dataset and ground-truth peak list.
#[derive(Clone, Debug)]
pub struct AromaticSpectraSet {
    pub arom_c13_hsqc: Option<Spectrum2D>,
    pub hbcbcgcdhd: Option<Spectrum2D>,
    pub hbcbcgcdcehe: Option<Spectrum2D>,
    pub his_lr_n15_hmqc: Option<Spectrum2D>,
    pub trp_ind_n15_hsqc: Option<Spectrum2D>,
    pub arom_hcch_tocsy: Option<Spectrum3D>,
    /// Intra-ring `TRP` indole `15N`-edited NOE/relay cross-peaks `(HE1, NE1) -> HD1 / HZ2`.
    pub trp_indole_relays: Vec<Peak>,
    /// Underlying simulated peak list prior to volumetric voxelization.
    pub ground_truth_peaks: Vec<Peak>,
}

#[derive(Clone, Debug)]
struct RingChSpin {
    carbon: &'static str,
    c_ppm: f64,
    h_ppm: f64,
    equiv_protons: usize,
}

fn collect_residue_aromatic_chs(ref_data: &Reference, k: usize) -> Vec<RingChSpin> {
    let res = ref_data.sequence[k].as_str();
    let mut out = Vec::new();
    let get_sym = |a1: &str, a2: &str| -> Option<f64> {
        ref_data
            .shifts
            .get(&(k, a1.to_string()))
            .or_else(|| ref_data.shifts.get(&(k, a2.to_string())))
            .copied()
    };
    match res {
        "PHE" => {
            if let (Some(c), Some(h)) = (get_sym("CD1", "CD2"), get_sym("HD1", "HD2")) {
                out.push(RingChSpin {
                    carbon: "CD1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
            if let (Some(c), Some(h)) = (get_sym("CE1", "CE2"), get_sym("HE1", "HE2")) {
                out.push(RingChSpin {
                    carbon: "CE1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
            if let (Some(&c), Some(&h)) = (
                ref_data.shifts.get(&(k, "CZ".to_string())),
                ref_data.shifts.get(&(k, "HZ".to_string())),
            ) {
                out.push(RingChSpin {
                    carbon: "CZ",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 1,
                });
            }
        }
        "TYR" => {
            if let (Some(c), Some(h)) = (get_sym("CD1", "CD2"), get_sym("HD1", "HD2")) {
                out.push(RingChSpin {
                    carbon: "CD1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
            if let (Some(c), Some(h)) = (get_sym("CE1", "CE2"), get_sym("HE1", "HE2")) {
                out.push(RingChSpin {
                    carbon: "CE1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
        }
        "TRP" => {
            for &(ca, ha) in &[
                ("CD1", "HD1"),
                ("CE3", "HE3"),
                ("CZ3", "HZ3"),
                ("CH2", "HH2"),
                ("CZ2", "HZ2"),
            ] {
                if let (Some(&c), Some(&h)) = (
                    ref_data.shifts.get(&(k, ca.to_string())),
                    ref_data.shifts.get(&(k, ha.to_string())),
                ) {
                    out.push(RingChSpin {
                        carbon: ca,
                        c_ppm: c,
                        h_ppm: h,
                        equiv_protons: 1,
                    });
                }
            }
        }
        "HIS" => {
            for &(ca, ha) in &[("CD2", "HD2"), ("CE1", "HE1")] {
                if let (Some(&c), Some(&h)) = (
                    ref_data.shifts.get(&(k, ca.to_string())),
                    ref_data.shifts.get(&(k, ha.to_string())),
                ) {
                    out.push(RingChSpin {
                        carbon: ca,
                        c_ppm: c,
                        h_ppm: h,
                        equiv_protons: 1,
                    });
                }
            }
        }
        _ => {}
    }
    out
}

/// Topological `13C-13C` bond steps inside an aromatic ring for `AROM_HCCH_TOCSY` attenuation.
fn intra_ring_tocsy_steps(res: &str, c_a: &str, c_b: &str) -> Option<u8> {
    if c_a == c_b {
        return Some(0);
    }
    match res {
        "PHE" => match (c_a, c_b) {
            ("CD1", "CE1") | ("CE1", "CD1") | ("CE1", "CZ") | ("CZ", "CE1") => Some(1),
            ("CD1", "CZ") | ("CZ", "CD1") => Some(2),
            _ => None,
        },
        "TYR" => match (c_a, c_b) {
            ("CD1", "CE1") | ("CE1", "CD1") => Some(1),
            _ => None,
        },
        "TRP" => {
            let pos = |c: &str| -> Option<i32> {
                match c {
                    "CE3" => Some(0),
                    "CZ3" => Some(1),
                    "CH2" => Some(2),
                    "CZ2" => Some(3),
                    _ => None,
                }
            };
            let (pa, pb) = (pos(c_a)?, pos(c_b)?);
            Some((pa - pb).unsigned_abs() as u8)
        }
        _ => None,
    }
}

/// Generate underlying jittered aromatic peak lists (with fast-flip degeneracy and dropout).
pub fn simulate_aromatic_peaks(
    ref_data: &Reference,
    his_tautomers: &BTreeMap<usize, HisTautomer>,
    cfg: &AromaticSimConfig,
) -> (Vec<Peak>, Vec<Peak>) {
    let mut rng = Rng::new(cfg.seed);
    let mut peaks = Vec::new();
    let mut trp_relays = Vec::new();

    for (k, res) in ref_data.sequence.iter().enumerate() {
        if !matches!(res.as_str(), "PHE" | "TYR" | "TRP" | "HIS") {
            continue;
        }
        let chs = collect_residue_aromatic_chs(ref_data, k);
        let cb_opt = ref_data.shifts.get(&(k, "CB".to_string())).copied();

        // 1. 2D AROM_C13HSQC
        if cfg.include_c13_hsqc {
            for ch in &chs {
                if rng.f64() < cfg.p_missing_peak * 0.25 {
                    continue;
                }
                let h = ch.h_ppm + rng.normal(0.0, cfg.jitter_h_ppm);
                let c = ch.c_ppm + rng.normal(0.0, cfg.jitter_c_ppm);
                let mult = if ch.equiv_protons == 2 { 1.95 } else { 1.0 };
                let ht = 110.0 * mult * (rng.normal(0.0, 0.10)).exp();
                peaks.push(Peak {
                    exp: "AROM_C13HSQC".to_string(),
                    h,
                    n: 0.0,
                    c: Some(c),
                    sign: 1,
                    height: Some(ht),
                    volume: Some(ht * 4.8),
                    h2: None,
                });
            }
        }

        // 2. Yamazaki 2D (HB)CB(CGCD)HD & (HB)CB(CGCDCE)HE
        if let Some(cb) = cb_opt {
            if cfg.include_hbcbcgcdhd {
                let hd_val = match res.as_str() {
                    "PHE" | "TYR" => ref_data
                        .shifts
                        .get(&(k, "HD1".to_string()))
                        .or_else(|| ref_data.shifts.get(&(k, "HD2".to_string())))
                        .copied(),
                    "TRP" => ref_data.shifts.get(&(k, "HD1".to_string())).copied(),
                    "HIS" => ref_data.shifts.get(&(k, "HD2".to_string())).copied(),
                    _ => None,
                };
                if let Some(hd) = hd_val {
                    if rng.f64() >= cfg.p_missing_peak * 0.5 {
                        let mult = if matches!(res.as_str(), "PHE" | "TYR") {
                            1.85
                        } else {
                            1.0
                        };
                        let ht = 95.0 * mult * (rng.normal(0.0, 0.12)).exp();
                        peaks.push(Peak {
                            exp: "HBCBCGCDHD".to_string(),
                            h: hd + rng.normal(0.0, cfg.jitter_h_ppm),
                            n: 0.0,
                            c: Some(cb + rng.normal(0.0, cfg.jitter_c_ppm)),
                            sign: 1,
                            height: Some(ht),
                            volume: Some(ht * 4.5),
                            h2: None,
                        });
                    }
                }
            }

            if cfg.include_hbcbcgcdcehe {
                let he_spec: Option<(Option<f64>, f64, f64)> = match res.as_str() {
                    "PHE" | "TYR" => Some((
                        ref_data
                            .shifts
                            .get(&(k, "HE1".to_string()))
                            .or_else(|| ref_data.shifts.get(&(k, "HE2".to_string())))
                            .copied(),
                        1.70,
                        cfg.p_missing_peak * 0.6,
                    )),
                    "TRP" => Some((
                        ref_data.shifts.get(&(k, "HE3".to_string())).copied(),
                        0.78,
                        cfg.p_missing_peak,
                    )),
                    "HIS" if cfg.include_weak_his_he1_yamazaki => Some((
                        ref_data.shifts.get(&(k, "HE1".to_string())).copied(),
                        0.42,
                        (cfg.p_missing_peak * 2.0).min(0.35),
                    )),
                    _ => None,
                };
                if let Some((Some(he), amp_scale, miss_p)) = he_spec {
                    if rng.f64() >= miss_p {
                        let ht = 90.0 * amp_scale * (rng.normal(0.0, 0.14)).exp();
                        peaks.push(Peak {
                            exp: "HBCBCGCDCEHE".to_string(),
                            h: he + rng.normal(0.0, cfg.jitter_h_ppm),
                            n: 0.0,
                            c: Some(cb + rng.normal(0.0, cfg.jitter_c_ppm)),
                            sign: 1,
                            height: Some(ht),
                            volume: Some(ht * 4.5),
                            h2: None,
                        });
                    }
                }
            }
        }

        // 3. 3D Intra-Ring AROM_HCCH_TOCSY
        if cfg.include_hcch_tocsy {
            for u in &chs {
                for v in &chs {
                    let Some(steps) = intra_ring_tocsy_steps(res, u.carbon, v.carbon) else {
                        continue;
                    };
                    let miss_p = if steps == 0 {
                        cfg.p_missing_peak * 0.25
                    } else if steps == 1 {
                        cfg.p_missing_peak * 0.65
                    } else {
                        cfg.p_missing_peak * 1.2
                    };
                    if rng.f64() < miss_p {
                        continue;
                    }
                    let atten = match steps {
                        0 => 1.25,
                        1 => 0.88,
                        2 => 0.58,
                        _ => 0.40,
                    };
                    let mult = ((u.equiv_protons * v.equiv_protons) as f64).sqrt();
                    let ht = 85.0 * atten * mult * (rng.normal(0.0, 0.12)).exp();
                    peaks.push(Peak {
                        exp: "AROM_HCCH_TOCSY".to_string(),
                        h: u.h_ppm + rng.normal(0.0, cfg.jitter_h_ppm),
                        n: 0.0,
                        c: Some(u.c_ppm + rng.normal(0.0, cfg.jitter_c_ppm)),
                        sign: 1,
                        height: Some(ht),
                        volume: Some(ht * 5.0),
                        h2: Some(v.h_ppm + rng.normal(0.0, cfg.jitter_h_ppm)),
                    });
                }
            }
        }

        // 4. Pelton 2D HIS_LR_N15HMQC (Pelton et al., Protein Sci. 1993, 2, 543-558, Table 1):
        //    Large long-range J_NH (~8.8-10.6 Hz) occurs to the unprotonated pyridine-type =N-
        //    (~249.5 ppm), whereas coupling to protonated pyrrole >NH (~168-177 ppm) is smaller
        //    (~5.2-5.4 Hz, and 3J(HD2, ND1H) ~ 1.4-2.0 Hz).
        if cfg.include_his_lr_hmqc && res == "HIS" {
            let st = his_tautomers.get(&k).copied().unwrap_or(HisTautomer::Ne2H);
            let nd1 = ref_data.shifts.get(&(k, "ND1".to_string())).copied();
            let ne2 = ref_data.shifts.get(&(k, "NE2".to_string())).copied();
            let he1 = ref_data.shifts.get(&(k, "HE1".to_string())).copied();
            let hd2 = ref_data.shifts.get(&(k, "HD2".to_string())).copied();

            if let (Some(nd1_v), Some(ne2_v)) = (nd1, ne2) {
                let j_nd1 = nd1_v + rng.normal(0.0, cfg.jitter_n_ppm);
                let j_ne2 = ne2_v + rng.normal(0.0, cfg.jitter_n_ppm);

                if let Some(he1_v) = he1 {
                    let j_he1 = he1_v + rng.normal(0.0, cfg.jitter_h_ppm);
                    // 2J(HE1, =N-) ~ 10.6 Hz vs 2J(HE1, >NH) ~ 5.2-5.4 Hz (Pelton 1993 Table 1)
                    let (a_nd1, a_ne2) = match st {
                        HisTautomer::Ne2H => (1.22, 0.65),
                        HisTautomer::Nd1H => (0.65, 1.22),
                        HisTautomer::DoublyProtonated => (0.85, 0.85),
                    };
                    for &(n_t, amp) in &[(j_nd1, a_nd1), (j_ne2, a_ne2)] {
                        if rng.f64() >= cfg.p_missing_peak * 0.3 {
                            let ht = 95.0 * amp * (rng.normal(0.0, 0.12)).exp();
                            peaks.push(Peak {
                                exp: "HIS_LR_N15HMQC".to_string(),
                                h: j_he1,
                                n: n_t,
                                c: None,
                                sign: 1,
                                height: Some(ht),
                                volume: Some(ht * 4.5),
                                h2: None,
                            });
                        }
                    }
                }

                if let Some(hd2_v) = hd2 {
                    let j_hd2 = hd2_v + rng.normal(0.0, cfg.jitter_h_ppm);
                    // Pelton 1993 Table 1 HD2 couplings:
                    // - Ne2H (tau): 3J(HD2, ND1) ~ 8.8 Hz > 2J(HD2, NE2H) ~ 5.2 Hz
                    // - Nd1H (pi):  2J(HD2, NE2) ~ 9.8 Hz >> 3J(HD2, ND1H) ~ 1.4 Hz
                    // - DoublyProtonated (+1): 2J(HD2, NE2H) ~ 5.4 Hz > 3J(HD2, ND1H) ~ 1.4 Hz
                    let (amp_nd1, miss_nd1, amp_ne2, miss_ne2) = match st {
                        HisTautomer::Ne2H => (
                            1.02,
                            cfg.p_missing_peak * 0.30,
                            0.56,
                            cfg.p_missing_peak * 0.45,
                        ),
                        HisTautomer::Nd1H => (
                            0.22,
                            (cfg.p_missing_peak * 2.5).min(0.42),
                            1.14,
                            cfg.p_missing_peak * 0.30,
                        ),
                        HisTautomer::DoublyProtonated => (
                            0.24,
                            (cfg.p_missing_peak * 2.5).min(0.42),
                            0.78,
                            cfg.p_missing_peak * 0.35,
                        ),
                    };
                    if rng.f64() >= miss_ne2 {
                        let ht_ne2 = 92.0 * amp_ne2 * (rng.normal(0.0, 0.12)).exp();
                        peaks.push(Peak {
                            exp: "HIS_LR_N15HMQC".to_string(),
                            h: j_hd2,
                            n: j_ne2,
                            c: None,
                            sign: 1,
                            height: Some(ht_ne2),
                            volume: Some(ht_ne2 * 4.5),
                            h2: None,
                        });
                    }
                    if rng.f64() >= miss_nd1 {
                        let ht_nd1 = 92.0 * amp_nd1 * (rng.normal(0.0, 0.14)).exp();
                        peaks.push(Peak {
                            exp: "HIS_LR_N15HMQC".to_string(),
                            h: j_hd2,
                            n: j_nd1,
                            c: None,
                            sign: 1,
                            height: Some(ht_nd1),
                            volume: Some(ht_nd1 * 4.5),
                            h2: None,
                        });
                    }
                }
            }
        }

        // 5. Optional TRP Indole 15NE1-1HE1 HSQC root + intra-ring NOE/relay to HD1 & HZ2
        if cfg.include_trp_indole && res == "TRP" {
            if let (Some(&ne1), Some(&he1)) = (
                ref_data.shifts.get(&(k, "NE1".to_string())),
                ref_data.shifts.get(&(k, "HE1".to_string())),
            ) {
                let j_he1 = he1 + rng.normal(0.0, cfg.jitter_h_ppm);
                let j_ne1 = ne1 + rng.normal(0.0, cfg.jitter_n_ppm);
                let ht = 115.0 * (rng.normal(0.0, 0.10)).exp();
                peaks.push(Peak {
                    exp: "TRP_IND_N15HSQC".to_string(),
                    h: j_he1,
                    n: j_ne1,
                    c: None,
                    sign: 1,
                    height: Some(ht),
                    volume: Some(ht * 4.6),
                    h2: None,
                });

                for &(target_h_atom, rel_amp) in &[("HD1", 1.0), ("HZ2", 0.85)] {
                    if let Some(&th) = ref_data.shifts.get(&(k, target_h_atom.to_string())) {
                        if rng.f64() >= cfg.p_missing_peak {
                            trp_relays.push(Peak {
                                exp: "NOESY15N".to_string(),
                                h: j_he1,
                                n: j_ne1,
                                c: None,
                                sign: 1,
                                height: Some(80.0 * rel_amp),
                                volume: Some(360.0 * rel_amp),
                                h2: Some(th + rng.normal(0.0, cfg.jitter_h_ppm)),
                            });
                        }
                    }
                }
            }
        }
    }

    (peaks, trp_relays)
}

#[inline]
fn pseudo_voigt(delta_ppm: f64, lw_ppm: f64, gauss_mix: f64) -> f64 {
    let u = delta_ppm / (0.5 * lw_ppm.max(1e-6));
    let g = (-LN_2 * u * u).exp();
    let l = 1.0 / (1.0 + u * u);
    gauss_mix * g + (1.0 - gauss_mix) * l
}

#[allow(clippy::too_many_arguments)]
fn render_2d_aromatic(
    exp: AromaticExperiment,
    peaks: &[Peak],
    nx: usize,
    ny: usize,
    lw_x: f64,
    lw_y: f64,
    cfg: &AromaticSimConfig,
    seed_offset: u64,
) -> Spectrum2D {
    let ((h_lo, h_hi), (y_lo, y_hi)) = exp.spectral_window();
    let (y_label, y_mhz, use_n_field) = match exp {
        AromaticExperiment::HisLrN15Hmqc | AromaticExperiment::TrpIndN15Hsqc => {
            ("15N", 60.81, true)
        }
        _ => ("13C", 150.90, false),
    };
    let axis_x = AxisPpm::from_bounds("1H", nx, h_hi, h_lo, 600.13);
    let axis_y = AxisPpm::from_bounds(y_label, ny, y_hi, y_lo, y_mhz);
    let mut rng = Rng::new(cfg.seed.wrapping_add(seed_offset));

    let mut data = vec![0.0f32; ny * nx];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.8 * lw_x / axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let rad_y = ((3.8 * lw_y / axis_y.step_ppm.abs()).ceil() as isize).max(4);
    let exp_name = exp.as_str();

    for p in peaks.iter().filter(|p| p.exp == exp_name) {
        let vy = if use_n_field {
            p.n
        } else {
            let Some(c) = p.c else { continue };
            c
        };
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(vy).round() as isize;
        let amp = p.height.unwrap_or(95.0) * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(nx as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y - 4).max(0) as usize;
        let y1 = (cy + rad_y + 4).min(ny as isize - 1).max(0) as usize;

        for iy in y0..=y1 {
            let py = axis_y.index_to_ppm(iy as f64);
            let dy_pts = (iy as f64) - axis_y.ppm_to_index(vy);
            let mut wy = pseudo_voigt(py - vy, lw_y, cfg.gaussian_mix);
            // Add realistic FT truncation sinc sidelobes along indirect axis F1
            if cfg.sinc_sidelobe_frac > 0.0 && dy_pts.abs() >= 1.75 && dy_pts.abs() <= 5.0 {
                let arg = PI * dy_pts / 1.8;
                wy += cfg.sinc_sidelobe_frac * (arg.sin() / arg);
            }
            for ix in x0..=x1 {
                let px = axis_x.index_to_ppm(ix as f64);
                let wx = pseudo_voigt(px - p.h, lw_x, cfg.gaussian_mix);
                data[iy * nx + ix] += (amp * wx * wy) as f32;
            }
        }
    }

    Spectrum2D {
        name: format!("Synthetic 2D {exp_name}"),
        axis_x,
        axis_y,
        data,
    }
}

fn render_3d_arom_hcch(peaks: &[Peak], cfg: &AromaticSimConfig) -> Spectrum3D {
    let ((h_lo, h_hi), (c_lo, c_hi)) = AromaticExperiment::AromHcchTocsy.spectral_window();
    let axis_x = AxisPpm::from_bounds("1H", cfg.nx_3d, h_hi, h_lo, 600.13);
    let axis_y = AxisPpm::from_bounds("13C", cfg.ny_3d, c_hi, c_lo, 150.90);
    let axis_z = AxisPpm::from_bounds("1H_tocsy", cfg.nz_3d, h_hi, h_lo, 600.13);
    let (nx, ny, nz) = (axis_x.size, axis_y.size, axis_z.size);

    let mut rng = Rng::new(cfg.seed.wrapping_add(5501));
    let mut data = vec![0.0f32; nz * ny * nx];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.0 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(2);
    let rad_y = ((3.0 * cfg.lw_c_arom_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(2);
    let rad_z = ((3.0 * cfg.lw_h_ppm / axis_z.step_ppm.abs()).ceil() as isize).max(2);

    for p in peaks.iter().filter(|p| p.exp == "AROM_HCCH_TOCSY") {
        let (Some(vc), Some(vh2)) = (p.c, p.h2) else {
            continue;
        };
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(vc).round() as isize;
        let cz = axis_z.ppm_to_index(vh2).round() as isize;
        let amp = p.height.unwrap_or(80.0) * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(nx as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(ny as isize - 1).max(0) as usize;
        let z0 = (cz - rad_z).max(0) as usize;
        let z1 = (cz + rad_z).min(nz as isize - 1).max(0) as usize;

        for iz in z0..=z1 {
            let pz = axis_z.index_to_ppm(iz as f64);
            let wz = pseudo_voigt(pz - vh2, cfg.lw_h_ppm, cfg.gaussian_mix);
            for iy in y0..=y1 {
                let py = axis_y.index_to_ppm(iy as f64);
                let wy = pseudo_voigt(py - vc, cfg.lw_c_arom_ppm, cfg.gaussian_mix);
                for ix in x0..=x1 {
                    let px = axis_x.index_to_ppm(ix as f64);
                    let wx = pseudo_voigt(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix);
                    data[(iz * ny + iy) * nx + ix] += (amp * wx * wy * wz) as f32;
                }
            }
        }
    }

    Spectrum3D {
        name: "Synthetic 3D AROM_HCCH_TOCSY".to_string(),
        exp: "AROM_HCCH_TOCSY".to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

/// Simulate realistic noisy volumetric 2D/3D aromatic spectra (`AROM_C13HSQC`, `HBCBCGCDHD`,
/// `HBCBCGCDCEHE`, `HIS_LR_N15HMQC`, `TRP_IND_N15HSQC`, `AROM_HCCH_TOCSY`) from `ref_data`.
pub fn simulate_aromatic_spectra(
    ref_data: &Reference,
    his_tautomers: &BTreeMap<usize, HisTautomer>,
    cfg: &AromaticSimConfig,
) -> AromaticSpectraSet {
    let (gt_peaks, trp_indole_relays) = simulate_aromatic_peaks(ref_data, his_tautomers, cfg);

    let arom_c13_hsqc = cfg.include_c13_hsqc.then(|| {
        render_2d_aromatic(
            AromaticExperiment::AromC13Hsqc,
            &gt_peaks,
            cfg.nx_2d,
            cfg.ny_2d,
            cfg.lw_h_ppm,
            cfg.lw_c_arom_ppm,
            cfg,
            101,
        )
    });
    let hbcbcgcdhd = cfg.include_hbcbcgcdhd.then(|| {
        render_2d_aromatic(
            AromaticExperiment::Hbcbcgcdhd,
            &gt_peaks,
            cfg.nx_2d,
            cfg.ny_2d,
            cfg.lw_h_ppm,
            cfg.lw_cb_ppm,
            cfg,
            202,
        )
    });
    let hbcbcgcdcehe = cfg.include_hbcbcgcdcehe.then(|| {
        render_2d_aromatic(
            AromaticExperiment::Hbcbcgcdcehe,
            &gt_peaks,
            cfg.nx_2d,
            cfg.ny_2d,
            cfg.lw_h_ppm,
            cfg.lw_cb_ppm,
            cfg,
            303,
        )
    });
    let his_lr_n15_hmqc = cfg.include_his_lr_hmqc.then(|| {
        render_2d_aromatic(
            AromaticExperiment::HisLrN15Hmqc,
            &gt_peaks,
            cfg.nx_2d,
            cfg.ny_hmqc,
            cfg.lw_h_ppm,
            cfg.lw_n_hmqc_ppm,
            cfg,
            404,
        )
    });
    let trp_ind_n15_hsqc = cfg.include_trp_indole.then(|| {
        render_2d_aromatic(
            AromaticExperiment::TrpIndN15Hsqc,
            &gt_peaks,
            256,
            128,
            cfg.lw_h_ppm,
            0.28,
            cfg,
            505,
        )
    });
    let arom_hcch_tocsy = cfg
        .include_hcch_tocsy
        .then(|| render_3d_arom_hcch(&gt_peaks, cfg));

    AromaticSpectraSet {
        arom_c13_hsqc,
        hbcbcgcdhd,
        hbcbcgcdcehe,
        his_lr_n15_hmqc,
        trp_ind_n15_hsqc,
        arom_hcch_tocsy,
        trp_indole_relays,
        ground_truth_peaks: gt_peaks,
    }
}

// ============================================================================
// 3. 2D & 3D Aromatic Peak Pickers with Ridge & Sinc-Sidelobe Suppression
// ============================================================================

/// Configuration for `pick_aromatic_2d` and `pick_aromatic_hcch_3d`.
#[derive(Clone, Debug)]
pub struct AromaticPickerConfig {
    /// Global MAD signal-to-noise threshold (`k * noise`, default `5.0`).
    pub snr_threshold: f32,
    /// Allow negative cross-peaks (default `false` for magnitude/absorption aromatic spectra).
    pub allow_negative: bool,
    /// Indirect-axis (`F1`) column MAD multiplier for `t1`-ridge suppression (default `5.2`).
    pub ridge_snr: f32,
    /// Minimum fraction of column max amplitude to suppress `t1` baseline wiggles (default `0.04`).
    pub col_sinc_frac: f32,
    /// Near-parent FT truncation sinc-sidelobe amplitude ratio within `+-5.5` `F1` points (default `0.15`).
    pub sidelobe_ratio: f64,
    /// Strip tolerances for anchored 3D `AROM_HCCH_TOCSY` picking.
    pub strip_tol_h: f64,
    pub strip_tol_c: f64,
}

impl Default for AromaticPickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.0,
            allow_negative: false,
            ridge_snr: 5.2,
            col_sinc_frac: 0.04,
            sidelobe_ratio: 0.15,
            strip_tol_h: 0.035,
            strip_tol_c: 0.40,
        }
    }
}

fn median_f32_local(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

fn column_trimmed_mad_and_max(
    spec: &Spectrum2D,
    ix: usize,
    scratch: &mut Vec<f32>,
    fallback_mad: f32,
) -> (f32, f32) {
    let ny = spec.ny();
    scratch.clear();
    let mut max_abs = 0.0f32;
    for iy in 0..ny {
        let v = spec.get(ix, iy);
        max_abs = max_abs.max(v.abs());
        scratch.push(v);
    }
    if scratch.len() < 16 {
        return (fallback_mad, max_abs);
    }
    let med0 = median_f32_local(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med0).abs();
    }
    let mad0 = (1.4826 * median_f32_local(scratch)).max(1e-12);
    scratch.clear();
    for iy in 0..ny {
        let v = spec.get(ix, iy);
        if (v - med0).abs() <= 3.5 * mad0 {
            scratch.push((v - med0).abs());
        }
    }
    if scratch.len() < 12 {
        return (mad0.max(fallback_mad), max_abs);
    }
    let mad1 = (1.4826 * median_f32_local(scratch)).max(fallback_mad);
    (mad1, max_abs)
}

#[inline]
fn subvoxel_3pt_fit(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.abs()) as f64;
    let b = (y0.abs()) as f64;
    let c = (yp1.abs()) as f64;
    if a > 0.0
        && b > a
        && b > c
        && c > 0.0
        && ym1.signum() == y0.signum()
        && yp1.signum() == y0.signum()
    {
        let la = a.ln();
        let lb = b.ln();
        let lc = c.ln();
        let denom = la - 2.0 * lb + lc;
        if denom.abs() > 1e-12 {
            return (0.5 * (la - lc) / denom).clamp(-0.5, 0.5);
        }
    }
    let denom = a - 2.0 * b + c;
    if denom.abs() > 1e-12 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Pick 2D aromatic peaks (`AROM_C13HSQC`, `HBCBCGCDHD`, `HBCBCGCDCEHE`, `HIS_LR_N15HMQC`,
/// `TRP_IND_N15HSQC`) with spectral-window gating, `t1`-ridge MAD thresholding, and near-parent
/// FT truncation sinc-sidelobe suppression.
pub fn pick_aromatic_2d(
    spec: &Spectrum2D,
    exp: AromaticExperiment,
    cfg: &AromaticPickerConfig,
) -> Vec<Peak> {
    let (nx, ny) = (spec.nx(), spec.ny());
    if nx < 3 || ny < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let global_thresh = cfg.snr_threshold * noise;
    let ((h_lo, h_hi), (y_lo, y_hi)) = exp.spectral_window();
    let use_n_field = matches!(
        exp,
        AromaticExperiment::HisLrN15Hmqc | AromaticExperiment::TrpIndN15Hsqc
    );

    let mut scratch = Vec::with_capacity(ny);
    let mut col_thresh = vec![global_thresh; nx];
    for (ix, ct) in col_thresh.iter_mut().enumerate().take(nx - 1).skip(1) {
        let hp = spec.axis_x.index_to_ppm(ix as f64);
        if hp < h_lo || hp > h_hi {
            *ct = f32::INFINITY;
            continue;
        }
        let (cmad, cmax) = column_trimmed_mad_and_max(spec, ix, &mut scratch, noise);
        *ct = global_thresh
            .max(cfg.ridge_snr * cmad)
            .max(cfg.col_sinc_frac * cmax);
    }

    let mut raw: Vec<(Peak, f64, f64)> = Vec::new();
    let exp_str = exp.as_str().to_string();

    for iy in 1..ny - 1 {
        let yp0 = spec.axis_y.index_to_ppm(iy as f64);
        if yp0 < y_lo || yp0 > y_hi {
            continue;
        }
        for (ix, &cth) in col_thresh.iter().enumerate().take(nx - 1).skip(1) {
            if !cth.is_finite() {
                continue;
            }
            let v = spec.get(ix, iy);
            if v.abs() < cth || (!cfg.allow_negative && v < 0.0) {
                continue;
            }
            let sign = if v >= 0.0 { 1 } else { -1 };
            let mut is_ext = true;
            'nbr: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = spec.get((ix as isize + dx) as usize, (iy as isize + dy) as usize);
                    if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                        is_ext = false;
                        break 'nbr;
                    }
                }
            }
            if !is_ext {
                continue;
            }

            let dx = subvoxel_3pt_fit(spec.get(ix - 1, iy), v, spec.get(ix + 1, iy));
            let dy = subvoxel_3pt_fit(spec.get(ix, iy - 1), v, spec.get(ix, iy + 1));
            let fx = (ix as f64) + dx;
            let fy = (iy as f64) + dy;
            let h = spec.axis_x.index_to_ppm(fx);
            let y_ppm = spec.axis_y.index_to_ppm(fy);

            let mut vol = 0.0f64;
            for wy in -1isize..=1 {
                for wx in -1isize..=1 {
                    let sv = spec.get((ix as isize + wx) as usize, (iy as isize + wy) as usize);
                    if sv.signum() == v.signum() {
                        vol += (sv.abs()) as f64;
                    }
                }
            }
            let pk = Peak {
                exp: exp_str.clone(),
                h,
                n: if use_n_field { y_ppm } else { 0.0 },
                c: if use_n_field { None } else { Some(y_ppm) },
                sign,
                height: Some((v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            };
            raw.push((pk, fx, fy));
        }
    }

    // Sort descending by amplitude and suppress indirect-axis FT sinc truncation wiggles
    raw.sort_by(|a, b| {
        b.0.height
            .unwrap_or(0.0)
            .total_cmp(&a.0.height.unwrap_or(0.0))
    });
    let mut kept: Vec<(Peak, f64, f64)> = Vec::with_capacity(raw.len());
    for (cand, fx, fy) in raw {
        let ch = cand.height.unwrap_or(0.0);
        let is_sidelobe = kept.iter().any(|(parent, pfx, pfy)| {
            let ph = parent.height.unwrap_or(0.0);
            let dx_pts = (fx - pfx).abs();
            let dy_pts = (fy - pfy).abs();
            // Immediate sinc sidelobe halo in same 1H column or shoulder within 1.2 voxels
            (dx_pts <= 1.6 && dy_pts <= 5.8 && ch < cfg.sidelobe_ratio * ph)
                || (dx_pts <= 1.2 && dy_pts <= 1.2)
        });
        if !is_sidelobe {
            kept.push((cand, fx, fy));
        }
    }

    kept.into_iter().map(|(p, _, _)| p).collect()
}

/// Pick 3D `AROM_HCCH_TOCSY` peaks optionally anchored on picked 2D `AROM_C13HSQC` roots.
pub fn pick_aromatic_hcch_3d(
    spec: &Spectrum3D,
    c13_roots: Option<&[Peak]>,
    cfg: &AromaticPickerConfig,
) -> Vec<Peak> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let roots = c13_roots.unwrap_or(&[]);

    let mut out = Vec::new();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            let y_grid = spec.axis_y.index_to_ppm(iy as f64);
            if !roots.is_empty()
                && !roots.iter().any(|r| {
                    r.c.is_some_and(|rc| (rc - y_grid).abs() <= cfg.strip_tol_c * 1.4)
                })
            {
                continue;
            }
            for ix in 1..nx - 1 {
                let v = spec.get(ix, iy, iz);
                if v.abs() < thresh || (!cfg.allow_negative && v < 0.0) {
                    continue;
                }
                let x_grid = spec.axis_x.index_to_ppm(ix as f64);
                if !roots.is_empty()
                    && !roots.iter().any(|r| {
                        (r.h - x_grid).abs() <= cfg.strip_tol_h
                            && r.c.is_some_and(|rc| (rc - y_grid).abs() <= cfg.strip_tol_c)
                    })
                {
                    continue;
                }
                let sign = if v >= 0.0 { 1 } else { -1 };
                let mut is_ext = true;
                'nbr: for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nv = spec.get(
                                (ix as isize + dx) as usize,
                                (iy as isize + dy) as usize,
                                (iz as isize + dz) as usize,
                            );
                            if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                                is_ext = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if !is_ext {
                    continue;
                }
                let dx = subvoxel_3pt_fit(spec.get(ix - 1, iy, iz), v, spec.get(ix + 1, iy, iz));
                let dy = subvoxel_3pt_fit(spec.get(ix, iy - 1, iz), v, spec.get(ix, iy + 1, iz));
                let dz = subvoxel_3pt_fit(spec.get(ix, iy, iz - 1), v, spec.get(ix, iy, iz + 1));

                let mut vol = 0.0f64;
                for wz in -1isize..=1 {
                    for wy in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = spec.get(
                                (ix as isize + wx) as usize,
                                (iy as isize + wy) as usize,
                                (iz as isize + wz) as usize,
                            );
                            if sv.signum() == v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                out.push(Peak {
                    exp: "AROM_HCCH_TOCSY".to_string(),
                    h: spec.axis_x.index_to_ppm(ix as f64 + dx),
                    n: 0.0,
                    c: Some(spec.axis_y.index_to_ppm(iy as f64 + dy)),
                    sign,
                    height: Some((v.abs()) as f64),
                    volume: Some(vol),
                    h2: Some(spec.axis_z.index_to_ppm(iz as f64 + dz)),
                });
            }
        }
    }
    out
}

/// Convenience helper picking all populated 2D and 3D spectra in an `AromaticSpectraSet`.
pub fn pick_all_aromatic_spectra(
    spectra: &AromaticSpectraSet,
    cfg: &AromaticPickerConfig,
) -> Vec<Peak> {
    let mut all = Vec::new();
    let c13_roots = spectra
        .arom_c13_hsqc
        .as_ref()
        .map(|s| pick_aromatic_2d(s, AromaticExperiment::AromC13Hsqc, cfg));
    if let Some(ref r) = c13_roots {
        all.extend(r.iter().cloned());
    }
    if let Some(ref s) = spectra.hbcbcgcdhd {
        all.extend(pick_aromatic_2d(s, AromaticExperiment::Hbcbcgcdhd, cfg));
    }
    if let Some(ref s) = spectra.hbcbcgcdcehe {
        all.extend(pick_aromatic_2d(s, AromaticExperiment::Hbcbcgcdcehe, cfg));
    }
    if let Some(ref s) = spectra.his_lr_n15_hmqc {
        all.extend(pick_aromatic_2d(s, AromaticExperiment::HisLrN15Hmqc, cfg));
    }
    if let Some(ref s) = spectra.trp_ind_n15_hsqc {
        all.extend(pick_aromatic_2d(s, AromaticExperiment::TrpIndN15Hsqc, cfg));
    }
    if let Some(ref s3) = spectra.arom_hcch_tocsy {
        all.extend(pick_aromatic_hcch_3d(s3, c13_roots.as_deref(), cfg));
    }
    all.extend(spectra.trp_indole_relays.iter().cloned());
    all
}

// ============================================================================
// 4. Exact Polynomial-Time Hungarian (Kuhn–Munkres) Bipartite Matcher
// ============================================================================

const FORBID_SCORE: f64 = -1.0e6;

/// Exact `O(N^3)` Kuhn–Munkres (Hungarian) maximum-weight 1-to-1 bipartite matching.
/// Given an `n_rows x n_cols` score matrix (`rows <= cols` handled automatically by transposition),
/// returns `(row_to_col, total_weight)` maximizing `sum_r weights[r][row_to_col[r]]`.
pub fn hungarian_max_weight(weights: &[Vec<f64>]) -> (Vec<Option<usize>>, f64) {
    let n_rows = weights.len();
    if n_rows == 0 {
        return (Vec::new(), 0.0);
    }
    let n_cols = weights[0].len();
    if n_cols == 0 {
        return (vec![None; n_rows], 0.0);
    }

    let n = n_rows.max(n_cols);
    // Shift weights so dummy padded entries have weight 0.0 and valid finite entries are well-conditioned
    let mut w = vec![vec![0.0f64; n]; n];
    for (r, row) in weights.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            w[r][c] = val.max(FORBID_SCORE * 2.0);
        }
    }

    // Convert max-weight to min-cost: cost[i][j] = max_val - w[i][j] >= 0
    let mut max_val = 0.0f64;
    for row in &w {
        for &v in row {
            if v > max_val {
                max_val = v;
            }
        }
    }

    // 1-indexed potentials u[0..=n], v[0..=n], matching p[0..=n] (col -> row), way[0..=n]
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cost = max_val - w[i0 - 1][j - 1];
                    let cur = cost - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![None; n_rows];
    let mut total = 0.0;
    for j in 1..=n {
        let r = p[j];
        if r >= 1 && r <= n_rows && (j - 1) < n_cols {
            let val = weights[r - 1][j - 1];
            if val > FORBID_SCORE * 0.5 {
                row_to_col[r - 1] = Some(j - 1);
                total += val;
            }
        }
    }
    (row_to_col, total)
}

/// Compute per-row exact leave-one-out margin `S* - max_{alt != win} S(alt)` using Hungarian re-solves.
fn hungarian_margins(
    weights: &[Vec<f64>],
    win_cols: &[Option<usize>],
    best_total: f64,
) -> Vec<f64> {
    let n_rows = weights.len();
    let mut margins = vec![0.0; n_rows];
    let mut scratch = weights.to_vec();

    for r in 0..n_rows {
        let Some(c_win) = win_cols[r] else {
            margins[r] = 0.0;
            continue;
        };
        let saved = scratch[r][c_win];
        scratch[r][c_win] = FORBID_SCORE * 2.0;
        let (_, alt_total) = hungarian_max_weight(&scratch);
        scratch[r][c_win] = saved;
        margins[r] = (best_total - alt_total).max(0.0);
    }
    margins
}

// ============================================================================
// 5. Aromatic Ring Spin-System Clustering (`PHE`, `TYR`, `TRP`, `HIS`)
// ============================================================================

/// Inferred chemical ring class of an assembled `AromaticRingCluster`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AromaticRingKind {
    Phe,
    Tyr,
    Trp,
    His,
}

impl AromaticRingKind {
    #[inline]
    pub const fn res_name(self) -> &'static str {
        match self {
            Self::Phe => "PHE",
            Self::Tyr => "TYR",
            Self::Trp => "TRP",
            Self::His => "HIS",
        }
    }
}

/// Assigned `(13C_ppm, 1H_ppm, intra_cluster_confidence)` slot inside an `AromaticRingCluster`.
#[derive(Clone, Copy, Debug)]
pub struct RingChSlot {
    pub c_ppm: f64,
    pub h_ppm: f64,
    pub intra_conf: f64,
}

/// Candidate aromatic ring spin system assembled from 2D/3D experimental cross-peaks.
#[derive(Clone, Debug)]
pub struct AromaticRingCluster {
    pub kind: AromaticRingKind,
    /// Keyed by canonical carbon name (`"CD1"`, `"CE1"`, `"CZ"`, `"CD2"`, `"CE3"`, `"CZ3"`, `"CH2"`, `"CZ2"`).
    pub ch_slots: BTreeMap<&'static str, RingChSlot>,
    /// Yamazaki `(HB)CB(CGCD)HD` observed `13CB` shift(s) linked to this ring's `HD*`/`HD1`/`HD2`.
    pub cb_hd: Option<f64>,
    /// Yamazaki `(HB)CB(CGCDCE)HE` observed `13CB` shift(s) linked to this ring's `HE*`/`HE3`/`HE1`.
    pub cb_he: Option<f64>,
    /// Imidazole `15ND1` and `15NE2` shifts + classified `HisTautomer` / `HisStateClassification` (`HIS` clusters).
    pub nd1_ppm: Option<(f64, f64)>,
    pub ne2_ppm: Option<(f64, f64)>,
    pub his_tautomer: Option<(HisTautomer, f64)>,
    pub his_state: Option<(HisStateClassification, f64)>,
    /// Indole `(15NE1, 1HE1, intra_conf)` root (`TRP` clusters).
    pub trp_indole: Option<(f64, f64, f64)>,
    /// Internal spin-network consistency log-score (TOCSY + Yamazaki triad agreement).
    pub internal_coherence: f64,
}

/// Configuration for `cluster_aromatic_rings` and `assign_aromatic_rings`.
#[derive(Clone, Debug)]
pub struct AromaticSolverConfig {
    /// Aromatic `1H` matching tolerance across experiments (ppm, default `0.028`).
    pub tol_h: f64,
    /// Aromatic `13C` matching tolerance across `C13HSQC` and `HCCH-TOCSY` (ppm, default `0.28`).
    pub tol_c: f64,
    /// `15N` matching tolerance for `HIS_LR_N15HMQC` & `TRP_IND_N15HSQC` (ppm, default `0.38`).
    pub tol_n: f64,
    /// Maximum `|CB_yamazaki - CB_backbone(k)|` tolerance (ppm, default `0.55`).
    pub tol_cb: f64,
    /// Gaussian standard deviation `sigma_cb` (ppm) for `|CB_yamazaki - CB_backbone|` LLR (default `0.055`).
    pub sigma_cb: f64,
    /// Minimum backbone confidence to trust `backbone_ref` `CB(k)` as a hard anchor (default `0.50`).
    pub min_backbone_conf: f64,
    /// Calibration temperature mapping bipartite Boltzmann energies to marginal posterior probability.
    pub calibration_temp: f64,
    /// Optional sample pH for Pelton / Henderson-Hasselbalch histidine state disambiguation.
    pub sample_ph: Option<f64>,
    /// Optional hint that the protein binds `Zn(II)` / transition metal ions via histidine sidechains.
    pub metal_coordinated_hint: bool,
}

impl Default for AromaticSolverConfig {
    fn default() -> Self {
        Self {
            tol_h: 0.028,
            tol_c: 0.28,
            tol_n: 0.38,
            tol_cb: 0.55,
            sigma_cb: 0.055,
            min_backbone_conf: 0.50,
            calibration_temp: 1.85,
            sample_ph: None,
            metal_coordinated_hint: false,
        }
    }
}

#[derive(Clone, Debug)]
struct AromHcNode {
    h: f64,
    c: f64,
    height: f64,
    tocsy_h: Vec<f64>,
    tocsy_relays: Vec<(f64, f64)>, // (h2_ppm, max_height)
    cb_hd_cands: Vec<(f64, f64)>,  // (cb_ppm, dh_abs)
    cb_he_cands: Vec<(f64, f64)>,  // (cb_ppm, dh_abs)
}

impl AromHcNode {
    fn off_diag_tocsy_count(&self, tol_h: f64) -> usize {
        self.tocsy_h
            .iter()
            .filter(|&&th| (th - self.h).abs() > tol_h * 1.05)
            .count()
    }

    fn tocsy_amp_to(&self, target_h: f64, tol_h: f64) -> f64 {
        self.tocsy_relays
            .iter()
            .filter(|&&(h2, _)| (h2 - target_h).abs() <= tol_h * 1.25)
            .map(|&(_, ht)| ht)
            .fold(0.0_f64, f64::max)
    }
}

fn cluster_1d_vals(mut vals: Vec<f64>, tol: f64) -> Vec<f64> {
    if vals.is_empty() {
        return Vec::new();
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for v in vals {
        if let Some(last) = groups.last_mut() {
            let mean = last.iter().sum::<f64>() / (last.len() as f64);
            if (v - mean).abs() <= tol {
                last.push(v);
                continue;
            }
        }
        groups.push(vec![v]);
    }
    groups
        .into_iter()
        .map(|g| g.iter().sum::<f64>() / (g.len() as f64))
        .collect()
}

fn build_arom_hc_nodes(peaks: &[Peak], cfg: &AromaticSolverConfig) -> Vec<AromHcNode> {
    let mut nodes: Vec<AromHcNode> = Vec::new();

    for p in peaks {
        let is_arom_c13 = p.exp == "AROM_C13HSQC"
            || (p.exp == "C13HSQC" && p.c.is_some_and(|c| (105.0..=146.0).contains(&c)));
        if !is_arom_c13 {
            continue;
        }
        let Some(c) = p.c else { continue };
        let merge_h = (cfg.tol_h * 0.55).max(0.012);
        let merge_c = (cfg.tol_c * 0.50).max(0.12);
        if let Some(ex) = nodes
            .iter_mut()
            .find(|n| ((n.h - p.h) / merge_h).hypot((n.c - c) / merge_c) <= 1.0)
        {
            let w1 = ex.height.max(1.0);
            let w2 = p.height.unwrap_or(80.0).max(1.0);
            ex.h = (ex.h * w1 + p.h * w2) / (w1 + w2);
            ex.c = (ex.c * w1 + c * w2) / (w1 + w2);
            ex.height += p.height.unwrap_or(0.0);
        } else {
            nodes.push(AromHcNode {
                h: p.h,
                c,
                height: p.height.unwrap_or(100.0),
                tocsy_h: Vec::new(),
                tocsy_relays: Vec::new(),
                cb_hd_cands: Vec::new(),
                cb_he_cands: Vec::new(),
            });
        }
    }

    // Attach 3D AROM_HCCH_TOCSY (and HCCHTOCSY aromatic strips), rescuing missing HSQC roots
    #[allow(clippy::type_complexity)]
    let mut orphans: Vec<(f64, f64, Vec<f64>, Vec<(f64, f64)>, f64)> = Vec::new();
    for p in peaks {
        let is_tocsy = p.exp == "AROM_HCCH_TOCSY"
            || (p.exp == "HCCHTOCSY" && p.c.is_some_and(|c| (105.0..=146.0).contains(&c)));
        if !is_tocsy {
            continue;
        }
        let (Some(c), Some(h2)) = (p.c, p.h2) else {
            continue;
        };
        let p_ht = p.height.unwrap_or(70.0).max(1.0);
        let mut attached = false;
        for n in &mut nodes {
            if ((n.h - p.h) / cfg.tol_h).hypot((n.c - c) / cfg.tol_c) <= 1.15 {
                n.tocsy_h.push(h2);
                n.tocsy_relays.push((h2, p_ht));
                attached = true;
            }
        }
        if !attached {
            if let Some(o) = orphans
                .iter_mut()
                .find(|o| ((o.0 - p.h) / cfg.tol_h).hypot((o.1 - c) / cfg.tol_c) <= 1.0)
            {
                let cnt = o.2.len() as f64;
                o.0 = (o.0 * cnt + p.h) / (cnt + 1.0);
                o.1 = (o.1 * cnt + c) / (cnt + 1.0);
                o.2.push(h2);
                o.3.push((h2, p_ht));
                o.4 = o.4.max(p_ht);
            } else {
                orphans.push((p.h, c, vec![h2], vec![(h2, p_ht)], p_ht));
            }
        }
    }
    for (oh, oc, h2s, relays, ht) in orphans {
        let reciprocated_single = h2s.len() == 1
            && (oh - h2s[0]).abs() > cfg.tol_h * 1.4
            && nodes.iter().any(|n| {
                (n.h - h2s[0]).abs() <= cfg.tol_h * 1.1
                    && has_h_in_ladder(&n.tocsy_h, oh, cfg.tol_h * 1.15).is_some()
            });
        if h2s.len() >= 2 || reciprocated_single {
            nodes.push(AromHcNode {
                h: oh,
                c: oc,
                height: ht,
                tocsy_h: h2s,
                tocsy_relays: relays,
                cb_hd_cands: Vec::new(),
                cb_he_cands: Vec::new(),
            });
        }
    }

    for n in &mut nodes {
        n.tocsy_h = cluster_1d_vals(std::mem::take(&mut n.tocsy_h), cfg.tol_h * 0.85);
    }

    // Attach Yamazaki HBCBCGCDHD and HBCBCGCDCEHE CB candidates
    for p in peaks {
        if p.exp == "HBCBCGCDHD" {
            let Some(cb) = p.c else { continue };
            for n in &mut nodes {
                let dh = (n.h - p.h).abs();
                if dh <= cfg.tol_h * 1.15 {
                    n.cb_hd_cands.push((cb, dh));
                }
            }
        } else if p.exp == "HBCBCGCDCEHE" {
            let Some(cb) = p.c else { continue };
            for n in &mut nodes {
                let dh = (n.h - p.h).abs();
                if dh <= cfg.tol_h * 1.15 {
                    n.cb_he_cands.push((cb, dh));
                }
            }
        }
    }

    for n in &mut nodes {
        n.cb_hd_cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        n.cb_he_cands.sort_by(|a, b| a.1.total_cmp(&b.1));
    }

    nodes
}

#[inline]
fn has_h_in_ladder(ladder: &[f64], target: f64, tol: f64) -> Option<f64> {
    ladder
        .iter()
        .map(|&x| (x - target).abs())
        .filter(|&d| d <= tol)
        .min_by(|a, b| a.total_cmp(b))
}

#[inline]
fn count_third_party_tocsy_relays(u: &AromHcNode, v: &AromHcNode, tol_h: f64) -> usize {
    let mut cnt = 0usize;
    for &hu in &u.tocsy_h {
        if (hu - u.h).abs() > tol_h * 1.1
            && (hu - v.h).abs() > tol_h * 1.1
            && has_h_in_ladder(&v.tocsy_h, hu, tol_h * 1.15).is_some()
        {
            cnt += 1;
        }
    }
    cnt
}

#[inline]
fn is_verified_tocsy_link(u: &AromHcNode, v: &AromHcNode, tol_h: f64) -> bool {
    let uv = has_h_in_ladder(&u.tocsy_h, v.h, tol_h * 1.2);
    let vu = has_h_in_ladder(&v.tocsy_h, u.h, tol_h * 1.2);
    if uv.is_some() && vu.is_some() {
        return true;
    }
    if uv.is_none() && vu.is_none() {
        return false;
    }
    if count_third_party_tocsy_relays(u, v, tol_h) >= 1 {
        return true;
    }
    // Allow 1-sided TOCSY when the unreciprocated strip has <= 1 off-diagonal peak
    // (e.g. 2-spin TYR or weak 3-spin PHE CZ strip near contour threshold), while
    // rejecting multi-relay (>= 2 off-diagonal peaks) strips from another ring that
    // share zero third-party relays.
    if uv.is_some() && v.off_diag_tocsy_count(tol_h) <= 1 {
        return true;
    }
    if vu.is_some() && u.off_diag_tocsy_count(tol_h) <= 1 {
        return true;
    }
    false
}

fn arom_tocsy_edge_score(u: &AromHcNode, v: &AromHcNode, directly_bonded: bool, tol_h: f64) -> f64 {
    let uv = has_h_in_ladder(&u.tocsy_h, v.h, tol_h * 1.2);
    let vu = has_h_in_ladder(&v.tocsy_h, u.h, tol_h * 1.2);

    let mut shared = 0usize;
    for &hu in &u.tocsy_h {
        if has_h_in_ladder(&v.tocsy_h, hu, tol_h * 1.2).is_some() {
            shared += 1;
        }
    }
    let union = (u.tocsy_h.len() + v.tocsy_h.len()).saturating_sub(shared);
    let jaccard = if union > 0 {
        (shared as f64) / (union as f64)
    } else {
        0.0
    };

    let verified = is_verified_tocsy_link(u, v, tol_h);
    let w = if directly_bonded { 3.2 } else { 1.8 };
    let mut sc = if verified { 2.4 * jaccard } else { 0.0 };
    if verified {
        if let Some(d) = uv {
            sc += w * (1.0 - 0.35 * (d / tol_h).min(1.2));
        }
        if let Some(d) = vu {
            sc += w * (1.0 - 0.35 * (d / tol_h).min(1.2));
        }
    } else if uv.is_some() ^ vu.is_some() {
        // Unverified 1-sided 1D 1H collision whose partner strip sees unrelated off-diagonal protons
        sc -= if directly_bonded { 2.6 } else { 1.5 };
    }
    if uv.is_none()
        && vu.is_none()
        && u.off_diag_tocsy_count(tol_h) >= 1
        && v.off_diag_tocsy_count(tol_h) >= 1
        && jaccard < 0.15
    {
        sc -= if directly_bonded { 5.5 } else { 2.8 };
    }
    sc
}

fn pick_best_single_yamazaki_cb(
    cands: &[(f64, f64)],
    known_cbs: &[f64],
    base_ok: f64,
    base_miss: f64,
    cfg: &AromaticSolverConfig,
) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for &(cb, dh) in cands {
        let bb_ok =
            known_cbs.is_empty() || known_cbs.iter().any(|&kcb| (kcb - cb).abs() <= cfg.tol_cb);
        let sc = if bb_ok {
            base_ok - dh / cfg.tol_h
        } else {
            base_miss - 0.5 * (dh / cfg.tol_h)
        };
        if best.is_none_or(|(_, b)| sc > b) {
            best = Some((cb, sc));
        }
    }
    best
}

/// Evaluate best `(cb_hd, cb_he)` Yamazaki agreement between an `HD` node and an `HE` node,
/// optionally guided by known protein backbone `CB` shifts so accidental `1H` overlaps pick
/// the physically matching `CB` pair.
fn best_shared_yamazaki_cb(
    hd_node: &AromHcNode,
    he_node: &AromHcNode,
    known_cbs: &[f64],
    cfg: &AromaticSolverConfig,
) -> (Option<f64>, Option<f64>, f64) {
    let mut best: Option<(f64, f64, f64)> = None;
    for &(cb1, dh1) in &hd_node.cb_hd_cands {
        for &(cb2, dh2) in &he_node.cb_he_cands {
            let dcb = (cb1 - cb2).abs();
            if dcb <= cfg.tol_cb * 0.75 {
                let z_cb = dcb / (cfg.sigma_cb * 1.45);
                let z_h = (dh1 + dh2) / cfg.tol_h;
                let cb_mean = 0.5 * (cb1 + cb2);
                let bb_bonus = if known_cbs
                    .iter()
                    .any(|&kcb| (kcb - cb_mean).abs() <= cfg.tol_cb)
                {
                    1.5
                } else {
                    0.0
                };
                let sc = 8.5 - 0.5 * z_cb * z_cb - 2.2 * z_h + bb_bonus;
                if best.is_none_or(|(_, _, b)| sc > b) {
                    best = Some((cb1, cb2, sc));
                }
            }
        }
    }
    if let Some((c1, c2, sc)) = best {
        return (Some(c1), Some(c2), sc);
    }

    // Single-sided fallback if either HD or HE Yamazaki cross-peak dropped out
    let best_hd = pick_best_single_yamazaki_cb(&hd_node.cb_hd_cands, known_cbs, 2.5, 0.5, cfg);
    let best_he = pick_best_single_yamazaki_cb(&he_node.cb_he_cands, known_cbs, 2.2, 0.4, cfg);

    // If BOTH have Yamazaki CB candidates and NO pair agrees within 1.25 * tol_cb, strongly penalize pairing them!
    if let (Some((c1, _)), Some((c2, _))) = (best_hd, best_he) {
        let any_close = hd_node.cb_hd_cands.iter().any(|&(cb1, _)| {
            he_node
                .cb_he_cands
                .iter()
                .any(|&(cb2, _)| (cb1 - cb2).abs() <= cfg.tol_cb * 1.25)
        });
        if !any_close {
            return (Some(c1), Some(c2), -9.5);
        }
    }
    if let Some((c1, s1)) = best_hd {
        return (Some(c1), None, s1);
    }
    if let Some((c2, s2)) = best_he {
        return (None, Some(c2), s2);
    }
    (None, None, 0.0)
}

#[inline]
fn margin_to_intra_conf(margin: f64) -> f64 {
    (1.0 / (1.0 + (-margin / 1.8).exp())).clamp(0.35, 1.0)
}

/// Score long-range `1H-15N` HMQC cross-peak intensity pattern `(hd_d1, hd_e2, he_d1, he_e2)`
/// under a specific `HisTautomer` hypothesis per Pelton et al. (*Protein Sci.* 1993, 2, 543–558, Table 1):
/// - `Ne2H` ($\tau$, `ND1` unprotonated `~249.5`, `NE2H` protonated `~168.5`):
///   `3J(HD2, ND1) ~ 8.8 Hz` > `2J(HD2, NE2H) ~ 5.2 Hz`, and `2J(HE1, ND1) ~ 10.6 Hz` >= `2J(HE1, NE2H) ~ 5.2 Hz`.
/// - `Nd1H` ($\pi$, `ND1H` protonated `~169.5`, `NE2` unprotonated `~249.5`):
///   `2J(HD2, NE2) ~ 9.8 Hz` >> `3J(HD2, ND1H) ~ 1.4 Hz`, and `2J(HE1, NE2) ~ 10.6 Hz` >= `2J(HE1, ND1H) ~ 5.2 Hz`.
/// - `DoublyProtonated` (`+1`, both `ND1H` and `NE2H` protonated):
///   `2J(HE1, ND1H) ~ 5.4 Hz` ≈ `2J(HE1, NE2H) ~ 5.4 Hz`, and `2J(HD2, NE2H) ~ 5.4 Hz` > `3J(HD2, ND1H) ~ 1.4 Hz`.
fn pelton_hmqc_coupling_llr(
    st: HisTautomer,
    hd_d1: f64,
    hd_e2: f64,
    he_d1: f64,
    he_e2: f64,
) -> f64 {
    let mut lp = 0.0;
    if he_d1 > 0.0 {
        lp += 1.1;
    }
    if he_e2 > 0.0 {
        lp += 1.1;
    }
    match st {
        HisTautomer::Ne2H => {
            if hd_d1 > 0.0 && (hd_e2 == 0.0 || hd_d1 >= 1.12 * hd_e2) {
                lp += 3.4;
            } else if hd_e2 > 1.35 * hd_d1 && hd_e2 > 0.0 {
                lp -= 2.8;
            }
            if he_d1 > 0.0 && he_e2 > 0.0 {
                if he_d1 >= 1.08 * he_e2 {
                    lp += 1.4;
                } else if he_e2 > 1.35 * he_d1 {
                    lp -= 1.2;
                }
            }
        }
        HisTautomer::Nd1H => {
            if hd_e2 > 0.0 && (hd_d1 == 0.0 || hd_e2 >= 1.45 * hd_d1) {
                lp += 3.6;
            } else if hd_d1 > 1.15 * hd_e2 && hd_d1 > 0.0 {
                lp -= 3.6;
            }
            if he_d1 > 0.0 && he_e2 > 0.0 {
                if he_e2 >= 1.08 * he_d1 {
                    lp += 1.4;
                } else if he_d1 > 1.35 * he_e2 {
                    lp -= 1.2;
                }
            }
        }
        HisTautomer::DoublyProtonated => {
            if hd_e2 > 0.0 && (hd_d1 == 0.0 || hd_e2 >= 1.35 * hd_d1) {
                lp += 3.4;
            } else if hd_d1 > 1.20 * hd_e2 && hd_d1 > 0.0 {
                lp -= 3.4;
            }
            if he_d1 > 0.0 && he_e2 > 0.0 {
                let ratio = he_d1.max(he_e2) / he_d1.min(he_e2).max(1e-6);
                if ratio <= 1.45 {
                    lp += 1.0;
                } else {
                    lp -= 0.6;
                }
            }
        }
    }
    lp
}

/// Infer `(ND1, NE2, (HisStateClassification, confidence), shared_link_score)` for a paired
/// `(opt_cd2, opt_ce1)` histidine ring from `HIS_LR_N15HMQC` cross-peaks and `(CD2, CE1, HE1, HD2, CB)` shifts.
#[allow(clippy::type_complexity)]
fn infer_his_nitrogens_and_tautomer(
    cd2_node: Option<&AromHcNode>,
    ce1_node: Option<&AromHcNode>,
    cb_hint: Option<f64>,
    hmqc_peaks: &[Peak],
    cfg: &AromaticSolverConfig,
) -> (
    Option<(f64, f64)>,
    Option<(f64, f64)>,
    (HisStateClassification, f64),
    f64,
) {
    // Gather HMQC (n_ppm, height) near HE1 and near HD2
    let mut he1_ns: Vec<(f64, f64)> = Vec::new();
    if let Some(ce1) = ce1_node {
        for p in hmqc_peaks {
            if (p.h - ce1.h).abs() <= cfg.tol_h * 1.25 {
                he1_ns.push((p.n, p.height.unwrap_or(60.0)));
            }
        }
    }
    let mut hd2_ns: Vec<(f64, f64)> = Vec::new();
    if let Some(cd2) = cd2_node {
        for p in hmqc_peaks {
            if (p.h - cd2.h).abs() <= cfg.tol_h * 1.25 {
                hd2_ns.push((p.n, p.height.unwrap_or(50.0)));
            }
        }
    }

    // Consolidate distinct 15N values seen by this ring
    let mut all_ns: Vec<(f64, f64, f64)> = Vec::new(); // (n_ppm, he1_ht, hd2_ht)
    for &(n, ht) in &he1_ns {
        if let Some(ex) = all_ns.iter_mut().find(|x| (x.0 - n).abs() <= cfg.tol_n) {
            ex.0 = 0.5 * (ex.0 + n);
            ex.1 = ex.1.max(ht);
        } else {
            all_ns.push((n, ht, 0.0));
        }
    }
    for &(n, ht) in &hd2_ns {
        if let Some(ex) = all_ns.iter_mut().find(|x| (x.0 - n).abs() <= cfg.tol_n) {
            let w_prev = ex.1.max(1.0);
            ex.0 = (ex.0 * w_prev + n * ht.max(1.0)) / (w_prev + ht.max(1.0));
            ex.2 = ex.2.max(ht);
        } else {
            all_ns.push((n, 0.0, ht));
        }
    }

    // Score how many nitrogens are shared between HE1 and HD2 (linking score!)
    let mut shared_link_score = 0.0;
    for &(_, h_e1, h_d2) in &all_ns {
        if h_e1 > 0.0 && h_d2 > 0.0 {
            shared_link_score += 6.5;
        }
    }

    let cd2_ppm = cd2_node.map(|n| n.c);
    let hd2_ppm = cd2_node.map(|n| n.h);
    let ce1_ppm = ce1_node.map(|n| n.c);
    let he1_ppm = ce1_node.map(|n| n.h);

    if all_ns.len() >= 2 {
        // Evaluate all ordered pairs (i_nd1, i_ne2) with i_nd1 != i_ne2 across canonical Pelton
        // states AND Zn(II)-ligated / fast-exchange 15N sub-hypotheses.
        let mut best_pair: Option<(usize, usize, f64)> = None;
        let mut second_lp = f64::NEG_INFINITY;

        for (i_d, &(n_d1, he_d1, hd_d1)) in all_ns.iter().enumerate() {
            for (i_e, &(n_e2, he_e2, hd_e2)) in all_ns.iter().enumerate() {
                if i_d == i_e {
                    continue;
                }
                let mut lp_best_sub = f64::NEG_INFINITY;
                let log_priors = [0.45_f64.ln(), 0.25_f64.ln(), 0.30_f64.ln()];
                for (s_idx, &st) in HisTautomer::ALL.iter().enumerate() {
                    let vp = st.variant_priors();
                    let mut lp = log_priors[s_idx]
                        + gauss_logpdf(n_d1, vp.nd1.0, vp.nd1.1)
                        + gauss_logpdf(n_e2, vp.ne2.0, vp.ne2.1);
                    if let Some(v) = cd2_ppm {
                        lp += gauss_logpdf(v, vp.cd2.0, vp.cd2.1);
                    }
                    if let Some(v) = ce1_ppm {
                        lp += gauss_logpdf(v, vp.ce1.0, vp.ce1.1);
                    }
                    if let Some(v) = he1_ppm {
                        lp += gauss_logpdf(v, vp.he1.0, vp.he1.1);
                    }
                    lp += pelton_hmqc_coupling_llr(st, hd_d1, hd_e2, he_d1, he_e2);
                    lp_best_sub = lp_best_sub.max(lp);
                }

                // Zn(II)-ligated / fast-exchange sub-hypotheses so intermediate 185..230 ppm
                // coordinating =N-M nitrogens still map accurately to ND1 vs NE2:
                // (a) Zn-ND1 (tau-like, NE2-H protonated ~172 ppm, ND1-Zn ~208 ppm, CD2 ~118.5 ppm)
                {
                    let mut lp_zn_d1 = (0.08_f64).ln()
                        + gauss_logpdf(n_d1, 208.0, 12.0)
                        + gauss_logpdf(n_e2, 172.0, 5.5);
                    if let Some(v) = cd2_ppm {
                        lp_zn_d1 += gauss_logpdf(v, 118.5, 2.2);
                    }
                    if let Some(v) = he1_ppm {
                        lp_zn_d1 += gauss_logpdf(v, 7.82, 0.35);
                    }
                    lp_zn_d1 +=
                        pelton_hmqc_coupling_llr(HisTautomer::Ne2H, hd_d1, hd_e2, he_d1, he_e2);
                    lp_best_sub = lp_best_sub.max(lp_zn_d1);
                }
                // (b) Zn-NE2 (pi-like, ND1-H protonated ~172 ppm, NE2-Zn ~208 ppm, CD2 ~124.5 ppm)
                {
                    let mut lp_zn_e2 = (0.06_f64).ln()
                        + gauss_logpdf(n_d1, 172.0, 5.5)
                        + gauss_logpdf(n_e2, 208.0, 12.0);
                    if let Some(v) = cd2_ppm {
                        lp_zn_e2 += gauss_logpdf(v, 124.5, 2.4);
                    }
                    if let Some(v) = he1_ppm {
                        lp_zn_e2 += gauss_logpdf(v, 7.88, 0.35);
                    }
                    lp_zn_e2 +=
                        pelton_hmqc_coupling_llr(HisTautomer::Nd1H, hd_d1, hd_e2, he_d1, he_e2);
                    lp_best_sub = lp_best_sub.max(lp_zn_e2);
                }

                if best_pair.is_none_or(|(_, _, blp)| lp_best_sub > blp) {
                    if let Some((_, _, blp)) = best_pair {
                        second_lp = second_lp.max(blp);
                    }
                    best_pair = Some((i_d, i_e, lp_best_sub));
                } else if lp_best_sub > second_lp {
                    second_lp = lp_best_sub;
                }
            }
        }

        if let Some((i_d, i_e, blp)) = best_pair {
            let n_conf = margin_to_intra_conf((blp - second_lp).max(0.0));
            let ev = HisTautomerEvidence {
                nd1_ppm: Some(all_ns[i_d].0),
                ne2_ppm: Some(all_ns[i_e].0),
                cg_ppm: None,
                cd2_ppm,
                hd2_ppm,
                ce1_ppm,
                he1_ppm,
                cb_ppm: cb_hint,
                sample_ph: cfg.sample_ph,
                metal_coordinated_hint: cfg.metal_coordinated_hint,
            };
            let state = classify_his_state(&ev);
            return (
                Some((all_ns[i_d].0, n_conf)),
                Some((all_ns[i_e].0, n_conf)),
                state,
                shared_link_score,
            );
        }
    } else if all_ns.len() == 1 {
        let (n_only, he_ht, hd_ht) = all_ns[0];
        let score_single = |as_nd1: bool| -> f64 {
            let mut best = f64::NEG_INFINITY;
            for &st in &HisTautomer::ALL {
                let vp = st.variant_priors();
                let mut lp = if as_nd1 {
                    gauss_logpdf(n_only, vp.nd1.0, vp.nd1.1)
                } else {
                    gauss_logpdf(n_only, vp.ne2.0, vp.ne2.1)
                };
                if let Some(v) = cd2_ppm {
                    lp += gauss_logpdf(v, vp.cd2.0, vp.cd2.1);
                }
                if let Some(v) = ce1_ppm {
                    lp += gauss_logpdf(v, vp.ce1.0, vp.ce1.1);
                }
                if let Some(v) = he1_ppm {
                    lp += gauss_logpdf(v, vp.he1.0, vp.he1.1);
                }
                if hd_ht > 0.0 {
                    lp += match (as_nd1, st) {
                        (true, HisTautomer::Ne2H) => 2.0,
                        (true, HisTautomer::Nd1H | HisTautomer::DoublyProtonated) => {
                            if he_ht == 0.0 || hd_ht >= 0.75 * he_ht {
                                -2.2
                            } else {
                                -0.6
                            }
                        }
                        (false, HisTautomer::Nd1H | HisTautomer::DoublyProtonated) => 2.2,
                        (false, HisTautomer::Ne2H) => 0.8,
                    };
                }
                best = best.max(lp);
            }
            best
        };
        let lp_d1 = score_single(true);
        let lp_e2 = score_single(false);
        if lp_e2 >= lp_d1 {
            let ev_as_ne2 = HisTautomerEvidence {
                nd1_ppm: None,
                ne2_ppm: Some(n_only),
                cg_ppm: None,
                cd2_ppm,
                hd2_ppm,
                ce1_ppm,
                he1_ppm,
                cb_ppm: cb_hint,
                sample_ph: cfg.sample_ph,
                metal_coordinated_hint: cfg.metal_coordinated_hint,
            };
            let n_conf = margin_to_intra_conf((lp_e2 - lp_d1).max(0.4)).min(0.88);
            return (
                None,
                Some((n_only, n_conf)),
                classify_his_state(&ev_as_ne2),
                shared_link_score,
            );
        } else {
            let ev_as_nd1 = HisTautomerEvidence {
                nd1_ppm: Some(n_only),
                ne2_ppm: None,
                cg_ppm: None,
                cd2_ppm,
                hd2_ppm,
                ce1_ppm,
                he1_ppm,
                cb_ppm: cb_hint,
                sample_ph: cfg.sample_ph,
                metal_coordinated_hint: cfg.metal_coordinated_hint,
            };
            let n_conf = margin_to_intra_conf((lp_d1 - lp_e2).max(0.4)).min(0.88);
            return (
                Some((n_only, n_conf)),
                None,
                classify_his_state(&ev_as_nd1),
                shared_link_score,
            );
        }
    }

    let ev_fallback = HisTautomerEvidence {
        nd1_ppm: None,
        ne2_ppm: None,
        cg_ppm: None,
        cd2_ppm,
        hd2_ppm,
        ce1_ppm,
        he1_ppm,
        cb_ppm: cb_hint,
        sample_ph: cfg.sample_ph,
        metal_coordinated_hint: cfg.metal_coordinated_hint,
    };
    let state = classify_his_state(&ev_fallback);
    (None, None, state, shared_link_score)
}

/// Cluster observed 2D/3D aromatic peaks into candidate `PHE`, `TYR`, `TRP`, and `HIS`
/// ring spin systems (`AromaticRingCluster`).
pub fn cluster_aromatic_rings(
    aromatic_peaks: &[Peak],
    backbone_ref: Option<&Reference>,
    priors: &SidechainPriors,
    cfg: &AromaticSolverConfig,
) -> Vec<AromaticRingCluster> {
    let nodes = build_arom_hc_nodes(aromatic_peaks, cfg);
    if nodes.is_empty() && !aromatic_peaks.iter().any(|p| p.exp == "HIS_LR_N15HMQC") {
        return Vec::new();
    }

    // Collect known aromatic CB shifts from backbone_ref (only as unlabelled set of valid CB frequencies!)
    let mut known_arom_cbs: Vec<f64> = Vec::new();
    let mut known_his_cbs: Vec<f64> = Vec::new();
    let mut known_trp_cbs: Vec<f64> = Vec::new();
    let mut known_tyr_cbs: Vec<f64> = Vec::new();
    let mut known_phe_cbs: Vec<f64> = Vec::new();
    let (mut n_his_seq, mut n_trp_seq, mut n_tyr_seq, mut n_phe_seq) =
        (0usize, 0usize, 0usize, 0usize);
    let mut n_his_anchored = 0usize;
    if let Some(bb) = backbone_ref {
        let any_bb_global = !bb.shifts.is_empty();
        for (k, r) in bb.sequence.iter().enumerate() {
            match r.as_str() {
                "HIS" => {
                    n_his_seq += 1;
                    let has_bb_k = !any_bb_global
                        || ["H", "N", "CA", "CB", "C"]
                            .iter()
                            .any(|a| bb.shifts.contains_key(&(k, (*a).to_string())));
                    if has_bb_k {
                        n_his_anchored += 1;
                    }
                }
                "TRP" => n_trp_seq += 1,
                "TYR" => n_tyr_seq += 1,
                "PHE" => n_phe_seq += 1,
                _ => continue,
            }
            if let Some(&cb) = bb.shifts.get(&(k, "CB".to_string())) {
                known_arom_cbs.push(cb);
                match r.as_str() {
                    "HIS" => known_his_cbs.push(cb),
                    "TRP" => known_trp_cbs.push(cb),
                    "TYR" => known_tyr_cbs.push(cb),
                    "PHE" => known_phe_cbs.push(cb),
                    _ => {}
                }
            }
        }
    }

    // Estimate 1-proton baseline amplitude (median height adjusted for 2x PHE/TYR abundance)
    let mut heights: Vec<f64> = nodes
        .iter()
        .map(|n| n.height)
        .filter(|&h| h > 0.0)
        .collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let raw_med = if heights.is_empty() {
        110.0
    } else {
        heights[heights.len() / 2]
    };
    // If PHE/TYR dominate, median is closer to ~2.0x; use sequence multiplicity balance or 25th percentile
    let n_1x_exp = n_phe_seq + 5 * n_trp_seq + 2 * n_his_seq;
    let n_2x_exp = 2 * n_phe_seq + 2 * n_tyr_seq;
    let h_1x = if backbone_ref.is_some() && n_2x_exp > n_1x_exp {
        let low_vals: Vec<f64> = heights
            .iter()
            .copied()
            .filter(|&x| x < raw_med * 0.72)
            .collect();
        if !low_vals.is_empty() {
            low_vals[low_vals.len() / 2].max(raw_med * 0.45)
        } else {
            raw_med * 0.53
        }
    } else if heights.len() >= 4 {
        heights[heights.len() / 4].max(raw_med * 0.52)
    } else {
        raw_med * 0.65
    };

    let hmqc_peaks: Vec<Peak> = aromatic_peaks
        .iter()
        .filter(|p| p.exp == "HIS_LR_N15HMQC")
        .cloned()
        .collect();
    let trp_ind_roots: Vec<Peak> = aromatic_peaks
        .iter()
        .filter(|p| {
            p.exp == "TRP_IND_N15HSQC"
                || (p.exp == "HSQC" && p.h > 8.8 && (120.0..=140.0).contains(&p.n))
        })
        .cloned()
        .collect();
    let trp_noes: Vec<Peak> = aromatic_peaks
        .iter()
        .filter(|p| (p.exp == "NOESY15N" || p.exp == "TRP_IND_NOE") && p.h > 8.8 && p.h2.is_some())
        .cloned()
        .collect();

    let mut used = vec![false; nodes.len()];
    let mut clusters: Vec<AromaticRingCluster> = Vec::new();

    // ------------------------------------------------------------------------
    // Family 1: Histidine Imidazole Rings (CD2/HD2, CE1/HE1, ND1, NE2, State)
    // ------------------------------------------------------------------------
    if backbone_ref.is_none() || n_his_seq > 0 {
        let is_trp_noe_proton = |h_val: f64| -> bool {
            trp_noes.iter().any(|np| {
                np.h2
                    .is_some_and(|h2| (h2 - h_val).abs() <= cfg.tol_h * 1.15)
            })
        };

        let mut ce1_cands: Vec<usize> = (0..nodes.len())
            .filter(|&u| {
                let n = &nodes[u];
                let hmqc_hit = hmqc_peaks
                    .iter()
                    .any(|p| (p.h - n.h).abs() <= cfg.tol_h * 1.35);
                let singlet_ce1_rescue = (134.8..=142.5).contains(&n.c)
                    && (7.05..=9.45).contains(&n.h)
                    && n.height < 1.45 * h_1x;
                n.off_diag_tocsy_count(cfg.tol_h) == 0
                    && !is_trp_noe_proton(n.h)
                    && ((hmqc_peaks.is_empty()
                        && (133.5..=144.0).contains(&n.c)
                        && (7.05..=9.45).contains(&n.h))
                        || (hmqc_hit
                            && (131.5..=144.5).contains(&n.c)
                            && (6.80..=9.50).contains(&n.h))
                        || singlet_ce1_rescue)
            })
            .collect();
        if backbone_ref.is_some() && ce1_cands.len() < n_his_anchored {
            let mut by_ht = ce1_cands.clone();
            by_ht.sort_by(|&a, &b| nodes[b].height.total_cmp(&nodes[a].height));
            for u in by_ht {
                if ce1_cands.len() < n_his_anchored && nodes[u].height >= 1.75 * h_1x {
                    ce1_cands.push(u);
                }
            }
        }

        let mut cd2_cands: Vec<usize> = (0..nodes.len())
            .filter(|&u| {
                if ce1_cands.contains(&u) {
                    return false;
                }
                let n = &nodes[u];
                let hmqc_hit = hmqc_peaks
                    .iter()
                    .any(|p| (p.h - n.h).abs() <= cfg.tol_h * 1.35);
                let his_cb_rescue = n.height < 1.45 * h_1x
                    && (113.5..=126.6).contains(&n.c)
                    && n.cb_hd_cands.iter().any(|&(cb, _)| {
                        known_his_cbs
                            .iter()
                            .any(|&hcb| (hcb - cb).abs() <= cfg.tol_cb)
                    });
                let c_ok = if hmqc_hit {
                    (112.5..=131.2).contains(&n.c)
                } else {
                    (113.5..=126.6).contains(&n.c)
                };
                n.off_diag_tocsy_count(cfg.tol_h) == 0
                    && !is_trp_noe_proton(n.h)
                    && (6.15..=8.35).contains(&n.h)
                    && c_ok
                    && (hmqc_hit || hmqc_peaks.is_empty() || his_cb_rescue)
            })
            .collect();
        if backbone_ref.is_some() && cd2_cands.len() < n_his_anchored {
            let mut by_ht = cd2_cands.clone();
            by_ht.sort_by(|&a, &b| nodes[b].height.total_cmp(&nodes[a].height));
            for u in by_ht {
                if cd2_cands.len() < n_his_anchored && nodes[u].height >= 1.75 * h_1x {
                    cd2_cands.push(u);
                }
            }
        }

        let make_his_taut_and_state = |st_pair: (HisStateClassification, f64), scale: f64| {
            let conf = (st_pair.1 * scale).clamp(0.0, 1.0);
            let resolved = st_pair.0.resolved_tautomer().map(|t| (t, conf));
            (resolved, Some((st_pair.0, conf)))
        };

        if !cd2_cands.is_empty() || !ce1_cands.is_empty() {
            let nr = cd2_cands.len();
            let nc = ce1_cands.len();
            if nr > 0 && nc > 0 {
                let mut w = vec![vec![FORBID_SCORE; nc]; nr];
                for (ir, &u_cd2) in cd2_cands.iter().enumerate() {
                    for (ic, &u_ce1) in ce1_cands.iter().enumerate() {
                        let cb_hint = pick_best_single_yamazaki_cb(
                            &nodes[u_cd2].cb_hd_cands,
                            &known_arom_cbs,
                            2.5,
                            0.5,
                            cfg,
                        )
                        .map(|x| x.0);
                        let (_, _, (_, t_post), link_sc) = infer_his_nitrogens_and_tautomer(
                            Some(&nodes[u_cd2]),
                            Some(&nodes[u_ce1]),
                            cb_hint,
                            &hmqc_peaks,
                            cfg,
                        );
                        let (_, _, cb_sc) = best_shared_yamazaki_cb(
                            &nodes[u_cd2],
                            &nodes[u_ce1],
                            &known_arom_cbs,
                            cfg,
                        );
                        let prior_sc =
                            priors.llr_c13("HIS", "CE1", CarbonRegion::Aromatic, nodes[u_ce1].c)
                                + priors.llr_h1(
                                    "HIS",
                                    "HE1",
                                    CarbonRegion::Aromatic,
                                    nodes[u_ce1].h,
                                )
                                + priors.llr_c13(
                                    "HIS",
                                    "CD2",
                                    CarbonRegion::Aromatic,
                                    nodes[u_cd2].c,
                                );
                        let his_cb_bonus = if !known_his_cbs.is_empty() {
                            let matches_his = nodes[u_cd2].cb_hd_cands.iter().any(|&(cb, _)| {
                                known_his_cbs.iter().any(|&h| (h - cb).abs() <= cfg.tol_cb)
                            });
                            let matches_only_trp = !matches_his
                                && !known_trp_cbs.is_empty()
                                && nodes[u_cd2].cb_hd_cands.iter().any(|&(cb, _)| {
                                    known_trp_cbs.iter().any(|&w| (w - cb).abs() <= cfg.tol_cb)
                                });
                            if matches_his {
                                4.5
                            } else if matches_only_trp {
                                -4.5
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let prior_w = if hmqc_peaks.is_empty() { 0.65 } else { 0.35 };
                        let mut pair_sc =
                            link_sc + 0.5 * cb_sc + prior_w * prior_sc + his_cb_bonus + t_post;
                        if !hmqc_peaks.is_empty()
                            && link_sc == 0.0
                            && cb_sc <= 0.0
                            && !(nr == 1 && nc == 1 && n_his_seq == 1)
                        {
                            pair_sc -= 5.0;
                        }
                        w[ir][ic] = pair_sc;
                    }
                }
                let (r2c, best_tot) = hungarian_max_weight(&w);
                let pair_margins = hungarian_margins(&w, &r2c, best_tot);
                let mut matched_ce1 = vec![false; nc];
                let min_pair_sc = if hmqc_peaks.is_empty() { 1.0 } else { -3.5 };

                for (ir, &u_cd2) in cd2_cands.iter().enumerate() {
                    let paired_ic = r2c[ir].filter(|&ic| w[ir][ic] > min_pair_sc);
                    if let Some(ic) = paired_ic {
                        matched_ce1[ic] = true;
                        let u_ce1 = ce1_cands[ic];
                        used[u_cd2] = true;
                        used[u_ce1] = true;
                        let (cb_hd, cb_he, cb_sc) = best_shared_yamazaki_cb(
                            &nodes[u_cd2],
                            &nodes[u_ce1],
                            &known_arom_cbs,
                            cfg,
                        );
                        let (nd1, ne2, st_pair, link_sc) = infer_his_nitrogens_and_tautomer(
                            Some(&nodes[u_cd2]),
                            Some(&nodes[u_ce1]),
                            cb_hd.or(cb_he),
                            &hmqc_peaks,
                            cfg,
                        );
                        let pair_conf = if nr == 1 && nc == 1 {
                            0.98
                        } else {
                            margin_to_intra_conf(pair_margins[ir])
                        };
                        let (his_tautomer, his_state) =
                            make_his_taut_and_state(st_pair, pair_conf.max(0.85));
                        let mut slots = BTreeMap::new();
                        slots.insert(
                            "CD2",
                            RingChSlot {
                                c_ppm: nodes[u_cd2].c,
                                h_ppm: nodes[u_cd2].h,
                                intra_conf: 0.98,
                            },
                        );
                        slots.insert(
                            "CE1",
                            RingChSlot {
                                c_ppm: nodes[u_ce1].c,
                                h_ppm: nodes[u_ce1].h,
                                intra_conf: pair_conf,
                            },
                        );
                        clusters.push(AromaticRingCluster {
                            kind: AromaticRingKind::His,
                            ch_slots: slots,
                            cb_hd,
                            cb_he,
                            nd1_ppm: nd1.map(|(v, c)| (v, c * pair_conf)),
                            ne2_ppm: ne2.map(|(v, c)| (v, c * pair_conf)),
                            his_tautomer,
                            his_state,
                            trp_indole: None,
                            internal_coherence: link_sc + cb_sc.max(0.0),
                        });
                    } else if !used[u_cd2]
                        && (backbone_ref.is_none()
                            || clusters
                                .iter()
                                .filter(|c| c.kind == AromaticRingKind::His)
                                .count()
                                < n_his_anchored)
                        && (!hmqc_peaks.is_empty()
                            || (n_trp_seq == 0 && n_phe_seq == 0)
                            || nodes[u_cd2].cb_hd_cands.iter().any(|&(cb, _)| {
                                known_his_cbs.iter().any(|&h| (h - cb).abs() <= cfg.tol_cb)
                                    && ((113.5..=121.8).contains(&nodes[u_cd2].c)
                                        || !known_trp_cbs
                                            .iter()
                                            .any(|&w| (w - cb).abs() <= cfg.tol_cb))
                            }))
                    {
                        used[u_cd2] = true;
                        let cb_hd = pick_best_single_yamazaki_cb(
                            &nodes[u_cd2].cb_hd_cands,
                            &known_arom_cbs,
                            2.5,
                            0.5,
                            cfg,
                        )
                        .map(|x| x.0);
                        let (nd1, ne2, st_pair, link_sc) = infer_his_nitrogens_and_tautomer(
                            Some(&nodes[u_cd2]),
                            None,
                            cb_hd,
                            &hmqc_peaks,
                            cfg,
                        );
                        let (his_tautomer, his_state) = make_his_taut_and_state(st_pair, 1.0);
                        let mut slots = BTreeMap::new();
                        slots.insert(
                            "CD2",
                            RingChSlot {
                                c_ppm: nodes[u_cd2].c,
                                h_ppm: nodes[u_cd2].h,
                                intra_conf: 0.92,
                            },
                        );
                        clusters.push(AromaticRingCluster {
                            kind: AromaticRingKind::His,
                            ch_slots: slots,
                            cb_hd,
                            cb_he: None,
                            nd1_ppm: nd1,
                            ne2_ppm: ne2,
                            his_tautomer,
                            his_state,
                            trp_indole: None,
                            internal_coherence: link_sc,
                        });
                    }
                }
                let bridge_cb_hd_from_hmqc = |he1_h: f64,
                                              nd1_opt: Option<(f64, f64)>,
                                              ne2_opt: Option<(f64, f64)>|
                 -> Option<f64> {
                    let ns: Vec<f64> = [nd1_opt.map(|x| x.0), ne2_opt.map(|x| x.0)]
                        .into_iter()
                        .flatten()
                        .collect();
                    if ns.is_empty() {
                        return None;
                    }
                    let mut hd2_hs: Vec<f64> = Vec::new();
                    for p in &hmqc_peaks {
                        if (p.h - he1_h).abs() > cfg.tol_h * 1.5
                            && ns.iter().any(|&nt| (p.n - nt).abs() <= cfg.tol_n)
                        {
                            hd2_hs.push(p.h);
                        }
                    }
                    let mut cands: Vec<(f64, f64)> = Vec::new();
                    for p in aromatic_peaks {
                        if p.exp == "HBCBCGCDHD" {
                            if let Some(cb) = p.c {
                                if let Some(dh) = hd2_hs
                                    .iter()
                                    .map(|&hh| (hh - p.h).abs())
                                    .filter(|&d| d <= cfg.tol_h * 1.25)
                                    .min_by(|a, b| a.total_cmp(b))
                                {
                                    cands.push((cb, dh));
                                }
                            }
                        }
                    }
                    pick_best_single_yamazaki_cb(&cands, &known_arom_cbs, 2.5, 0.5, cfg)
                        .map(|x| x.0)
                };

                for (ic, &u_ce1) in ce1_cands.iter().enumerate() {
                    if !matched_ce1[ic]
                        && !used[u_ce1]
                        && (!hmqc_peaks.is_empty()
                            || !nodes[u_ce1].cb_he_cands.is_empty()
                            || ((7.35..=9.25).contains(&nodes[u_ce1].h)
                                && (135.0..=140.5).contains(&nodes[u_ce1].c)))
                    {
                        used[u_ce1] = true;
                        let cb_he = pick_best_single_yamazaki_cb(
                            &nodes[u_ce1].cb_he_cands,
                            &known_arom_cbs,
                            2.2,
                            0.4,
                            cfg,
                        )
                        .map(|x| x.0);
                        let (nd1, ne2, st_pair, link_sc) = infer_his_nitrogens_and_tautomer(
                            None,
                            Some(&nodes[u_ce1]),
                            cb_he,
                            &hmqc_peaks,
                            cfg,
                        );
                        let cb_hd = bridge_cb_hd_from_hmqc(nodes[u_ce1].h, nd1, ne2);
                        let (his_tautomer, his_state) = make_his_taut_and_state(st_pair, 1.0);
                        let mut slots = BTreeMap::new();
                        slots.insert(
                            "CE1",
                            RingChSlot {
                                c_ppm: nodes[u_ce1].c,
                                h_ppm: nodes[u_ce1].h,
                                intra_conf: 0.90,
                            },
                        );
                        clusters.push(AromaticRingCluster {
                            kind: AromaticRingKind::His,
                            ch_slots: slots,
                            cb_hd,
                            cb_he,
                            nd1_ppm: nd1,
                            ne2_ppm: ne2,
                            his_tautomer,
                            his_state,
                            trp_indole: None,
                            internal_coherence: link_sc,
                        });
                    }
                }
            } else {
                for &u_cd2 in &cd2_cands {
                    if used[u_cd2]
                        || (backbone_ref.is_some()
                            && clusters
                                .iter()
                                .filter(|c| c.kind == AromaticRingKind::His)
                                .count()
                                >= n_his_anchored)
                        || (hmqc_peaks.is_empty()
                            && (n_trp_seq > 0 || n_phe_seq > 0)
                            && !nodes[u_cd2].cb_hd_cands.iter().any(|&(cb, _)| {
                                known_his_cbs.iter().any(|&h| (h - cb).abs() <= cfg.tol_cb)
                                    && ((113.5..=121.8).contains(&nodes[u_cd2].c)
                                        || !known_trp_cbs
                                            .iter()
                                            .any(|&w| (w - cb).abs() <= cfg.tol_cb))
                            }))
                    {
                        continue;
                    }
                    used[u_cd2] = true;
                    let cb_hd = pick_best_single_yamazaki_cb(
                        &nodes[u_cd2].cb_hd_cands,
                        &known_arom_cbs,
                        2.5,
                        0.5,
                        cfg,
                    )
                    .map(|x| x.0);
                    let (nd1, ne2, st_pair, link_sc) = infer_his_nitrogens_and_tautomer(
                        Some(&nodes[u_cd2]),
                        None,
                        cb_hd,
                        &hmqc_peaks,
                        cfg,
                    );
                    let (his_tautomer, his_state) = make_his_taut_and_state(st_pair, 1.0);
                    let mut slots = BTreeMap::new();
                    slots.insert(
                        "CD2",
                        RingChSlot {
                            c_ppm: nodes[u_cd2].c,
                            h_ppm: nodes[u_cd2].h,
                            intra_conf: 0.92,
                        },
                    );
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::His,
                        ch_slots: slots,
                        cb_hd,
                        cb_he: None,
                        nd1_ppm: nd1,
                        ne2_ppm: ne2,
                        his_tautomer,
                        his_state,
                        trp_indole: None,
                        internal_coherence: link_sc,
                    });
                }
                for &u_ce1 in &ce1_cands {
                    used[u_ce1] = true;
                    let cb_he = pick_best_single_yamazaki_cb(
                        &nodes[u_ce1].cb_he_cands,
                        &known_arom_cbs,
                        2.2,
                        0.4,
                        cfg,
                    )
                    .map(|x| x.0);
                    let (nd1, ne2, st_pair, link_sc) = infer_his_nitrogens_and_tautomer(
                        None,
                        Some(&nodes[u_ce1]),
                        cb_he,
                        &hmqc_peaks,
                        cfg,
                    );
                    let mut cb_hd = None;
                    if let Some(n1) = nd1.or(ne2).map(|x| x.0) {
                        let hd2_hs: Vec<f64> = hmqc_peaks
                            .iter()
                            .filter(|p| {
                                (p.h - nodes[u_ce1].h).abs() > cfg.tol_h * 1.5
                                    && (p.n - n1).abs() <= cfg.tol_n
                            })
                            .map(|p| p.h)
                            .collect();
                        let mut cands: Vec<(f64, f64)> = Vec::new();
                        for p in aromatic_peaks {
                            if p.exp == "HBCBCGCDHD" {
                                if let Some(cb) = p.c {
                                    if let Some(dh) = hd2_hs
                                        .iter()
                                        .map(|&hh| (hh - p.h).abs())
                                        .filter(|&d| d <= cfg.tol_h * 1.25)
                                        .min_by(|a, b| a.total_cmp(b))
                                    {
                                        cands.push((cb, dh));
                                    }
                                }
                            }
                        }
                        cb_hd =
                            pick_best_single_yamazaki_cb(&cands, &known_arom_cbs, 2.5, 0.5, cfg)
                                .map(|x| x.0);
                    }
                    let (his_tautomer, his_state) = make_his_taut_and_state(st_pair, 1.0);
                    let mut slots = BTreeMap::new();
                    slots.insert(
                        "CE1",
                        RingChSlot {
                            c_ppm: nodes[u_ce1].c,
                            h_ppm: nodes[u_ce1].h,
                            intra_conf: 0.90,
                        },
                    );
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::His,
                        ch_slots: slots,
                        cb_hd,
                        cb_he,
                        nd1_ppm: nd1,
                        ne2_ppm: ne2,
                        his_tautomer,
                        his_state,
                        trp_indole: None,
                        internal_coherence: link_sc,
                    });
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Family 2: Tryptophan Bicyclic Indole Rings (CD1 + CE3-CZ3-CH2-CZ2 + NE1/HE1)
    // ------------------------------------------------------------------------
    if backbone_ref.is_none() || n_trp_seq > 0 {
        let is_likely_tyr_ce = |u: usize| -> bool {
            n_tyr_seq > 0 && (114.5..=120.8).contains(&nodes[u].c) && {
                nodes[u].cb_he_cands.iter().any(|&(cb, _)| {
                    if !known_tyr_cbs.is_empty() || !known_trp_cbs.is_empty() {
                        known_tyr_cbs
                            .iter()
                            .any(|&tcb| (tcb - cb).abs() <= cfg.tol_cb)
                            && !known_trp_cbs
                                .iter()
                                .any(|&wcb| (wcb - cb).abs() <= cfg.tol_cb)
                    } else {
                        cb >= 36.2
                    }
                })
            }
        };

        let has_tocsy_exp = aromatic_peaks
            .iter()
            .any(|p| p.exp == "AROM_HCCH_TOCSY" || p.exp == "HCCHTOCSY");

        // Seed benzenoid 4-spin chains from unclaimed CZ2 (~114.4 ppm, 108.5..119.8 ppm)
        let mut cz2_seeds: Vec<usize> = (0..nodes.len())
            .filter(|&u| {
                !used[u]
                    && !is_likely_tyr_ce(u)
                    && (108.5..=119.8).contains(&nodes[u].c)
                    && (nodes[u].off_diag_tocsy_count(cfg.tol_h) >= 1
                        || (!has_tocsy_exp && nodes[u].height < 1.55 * h_1x))
            })
            .collect();
        cz2_seeds.sort_by(|&a, &b| {
            let multi_a = u8::from(nodes[a].off_diag_tocsy_count(cfg.tol_h) >= 2);
            let multi_b = u8::from(nodes[b].off_diag_tocsy_count(cfg.tol_h) >= 2);
            let one_x_a = u8::from(nodes[a].height < 1.55 * h_1x);
            let one_x_b = u8::from(nodes[b].height < 1.55 * h_1x);
            let core_cz2_a = u8::from((110.5..=117.6).contains(&nodes[a].c));
            let core_cz2_b = u8::from((110.5..=117.6).contains(&nodes[b].c));
            let conn_a = u8::from(nodes[a].off_diag_tocsy_count(cfg.tol_h) >= 1)
                + u8::from(trp_noes.iter().any(|np| {
                    np.h2
                        .is_some_and(|h2| (h2 - nodes[a].h).abs() <= cfg.tol_h * 1.25)
                }));
            let conn_b = u8::from(nodes[b].off_diag_tocsy_count(cfg.tol_h) >= 1)
                + u8::from(trp_noes.iter().any(|np| {
                    np.h2
                        .is_some_and(|h2| (h2 - nodes[b].h).abs() <= cfg.tol_h * 1.25)
                }));
            multi_b
                .cmp(&multi_a)
                .then_with(|| one_x_b.cmp(&one_x_a))
                .then_with(|| core_cz2_b.cmp(&core_cz2_a))
                .then_with(|| conn_b.cmp(&conn_a))
                .then_with(|| nodes[a].c.total_cmp(&nodes[b].c))
        });

        let active_cz2_seeds: Vec<usize> = if backbone_ref.is_some() {
            cz2_seeds.iter().copied().take(n_trp_seq).collect()
        } else {
            cz2_seeds.clone()
        };

        let has_trp_cb_he = |u: usize| -> bool {
            nodes[u].cb_he_cands.iter().any(|&(cb, _)| {
                if !known_trp_cbs.is_empty() {
                    known_trp_cbs
                        .iter()
                        .any(|&wcb| (wcb - cb).abs() <= cfg.tol_cb)
                } else {
                    cb < 36.2
                }
            })
        };

        let tocsy_pair_amp = |u: usize, v: usize| -> f64 {
            let a1 = nodes[u].tocsy_amp_to(nodes[v].h, cfg.tol_h);
            let a2 = nodes[v].tocsy_amp_to(nodes[u].h, cfg.tol_h);
            if a1 > 0.0 && a2 > 0.0 {
                0.5 * (a1 + a2)
            } else {
                a1.max(a2)
            }
        };

        let is_reserved_other_cz2 = |v: usize, u_self: usize| -> bool {
            v != u_self
                && nodes[v].c <= 117.2
                && active_cz2_seeds.contains(&v)
                && (backbone_ref.is_none() || active_cz2_seeds.len() <= n_trp_seq)
        };

        #[allow(clippy::type_complexity)]
        let mut benz_chains: Vec<([Option<usize>; 4], f64, [f64; 4])> = Vec::new();
        let mut trp_node_uses = vec![0usize; nodes.len()];
        for &u_cz2 in &cz2_seeds {
            if used[u_cz2] || (backbone_ref.is_some() && benz_chains.len() >= n_trp_seq) {
                continue;
            }
            // Candidate pool for (CE3, CZ3, CH2) connected to u_cz2 in 114.0..130.5 ppm
            let mut pool: Vec<usize> = (0..nodes.len())
                .filter(|&v| {
                    !used[v]
                        && v != u_cz2
                        && !is_reserved_other_cz2(v, u_cz2)
                        && !is_likely_tyr_ce(v)
                        && (114.0..=130.5).contains(&nodes[v].c)
                        && (is_verified_tocsy_link(&nodes[u_cz2], &nodes[v], cfg.tol_h)
                            || (!has_tocsy_exp
                                && nodes[u_cz2].off_diag_tocsy_count(cfg.tol_h) == 0
                                && nodes[v].height < 1.62 * h_1x))
                })
                .collect();
            if has_tocsy_exp && pool.is_empty() {
                continue;
            }
            // Expand by up to 2 TOCSY hops (CZ2 -> CH2 -> CZ3 -> CE3 when 2-/3-bond relays drop out)
            for _ in 0..2 {
                let cur = pool.clone();
                for &m in &cur {
                    for v in 0..nodes.len() {
                        if !used[v]
                            && v != u_cz2
                            && !is_reserved_other_cz2(v, u_cz2)
                            && !is_likely_tyr_ce(v)
                            && !pool.contains(&v)
                            && (114.0..=130.5).contains(&nodes[v].c)
                            && is_verified_tocsy_link(&nodes[m], &nodes[v], cfg.tol_h)
                            && arom_tocsy_edge_score(&nodes[m], &nodes[v], true, cfg.tol_h) > 1.8
                        {
                            pool.push(v);
                        }
                    }
                }
            }

            // Enumerate all injective assignments to (CE3, CZ3, CH2) from pool U {None}
            let mut opts: Vec<Option<usize>> = pool.iter().copied().map(Some).collect();
            opts.push(None);

            let mut best_arr = [None, None, None, Some(u_cz2)];
            let mut best_sc = f64::NEG_INFINITY;
            let mut hyps: Vec<([Option<usize>; 4], f64)> = Vec::new();

            for &o_ce3 in &opts {
                for &o_cz3 in &opts {
                    if o_ce3.is_some() && o_ce3 == o_cz3 {
                        continue;
                    }
                    for &o_ch2 in &opts {
                        if (o_ch2.is_some() && (o_ch2 == o_ce3 || o_ch2 == o_cz3))
                            || (o_ce3.is_none() && o_cz3.is_none() && o_ch2.is_none())
                        {
                            continue;
                        }
                        let mut sc =
                            priors.llr_c13("TRP", "CZ2", CarbonRegion::Aromatic, nodes[u_cz2].c)
                                + priors.llr_h1(
                                    "TRP",
                                    "HZ2",
                                    CarbonRegion::Aromatic,
                                    nodes[u_cz2].h,
                                );
                        if let Some(u) = o_ce3 {
                            sc += priors.llr_c13("TRP", "CE3", CarbonRegion::Aromatic, nodes[u].c)
                                + priors.llr_h1("TRP", "HE3", CarbonRegion::Aromatic, nodes[u].h);
                            if has_trp_cb_he(u) {
                                sc += 5.6;
                            } else if !nodes[u].cb_he_cands.is_empty() {
                                sc += 2.2;
                            }
                        } else {
                            sc -= 2.2;
                        }
                        if let Some(u) = o_cz3 {
                            sc += priors.llr_c13("TRP", "CZ3", CarbonRegion::Aromatic, nodes[u].c)
                                + priors.llr_h1("TRP", "HZ3", CarbonRegion::Aromatic, nodes[u].h);
                        } else {
                            sc -= 2.2;
                        }
                        if let Some(u) = o_ch2 {
                            sc += priors.llr_c13("TRP", "CH2", CarbonRegion::Aromatic, nodes[u].c)
                                + priors.llr_h1("TRP", "HH2", CarbonRegion::Aromatic, nodes[u].h);
                            sc += arom_tocsy_edge_score(&nodes[u], &nodes[u_cz2], true, cfg.tol_h);
                        } else {
                            sc -= 2.2;
                        }
                        if let (Some(u1), Some(u2)) = (o_ce3, o_cz3) {
                            sc += arom_tocsy_edge_score(&nodes[u1], &nodes[u2], true, cfg.tol_h);
                        }
                        if let (Some(u1), Some(u3)) = (o_ce3, o_ch2) {
                            let w23 = if o_cz3.is_none() { 0.90 } else { 0.55 };
                            sc += w23
                                * arom_tocsy_edge_score(&nodes[u1], &nodes[u3], false, cfg.tol_h);
                        }
                        if let (Some(u2), Some(u3)) = (o_cz3, o_ch2) {
                            sc += arom_tocsy_edge_score(&nodes[u2], &nodes[u3], true, cfg.tol_h);
                        }
                        if let Some(u2) = o_cz3 {
                            let w24 = if o_ch2.is_none() { 0.90 } else { 0.55 };
                            sc += w24
                                * arom_tocsy_edge_score(
                                    &nodes[u2],
                                    &nodes[u_cz2],
                                    false,
                                    cfg.tol_h,
                                );
                        }
                        if let Some(u1) = o_ce3 {
                            sc += 0.35
                                * arom_tocsy_edge_score(
                                    &nodes[u1],
                                    &nodes[u_cz2],
                                    false,
                                    cfg.tol_h,
                                );
                        }
                        // Mild isotropic-mixing 13C-13C TOCSY step-attenuation bonus along CE3(0)-CZ3(1)-CH2(2)-CZ2(3)
                        if let (Some(u_ch2), Some(u_cz3)) = (o_ch2, o_cz3) {
                            let i1 = tocsy_pair_amp(u_cz2, u_ch2);
                            let i2 = tocsy_pair_amp(u_cz2, u_cz3);
                            if i1 > 0.0 && i2 > 0.0 && i1 >= 1.28 * i2 {
                                sc += 0.9;
                            }
                        }
                        if let (Some(u_ch2), Some(u_ce3)) = (o_ch2, o_ce3) {
                            let i1 = tocsy_pair_amp(u_cz2, u_ch2);
                            let i3 = tocsy_pair_amp(u_cz2, u_ce3);
                            if i1 > 0.0 && i3 > 0.0 && i1 >= 1.35 * i3 {
                                sc += 1.1;
                            }
                        }
                        if let (Some(u_ce3), Some(u_cz3), Some(u_ch2)) = (o_ce3, o_cz3, o_ch2) {
                            let j1 = tocsy_pair_amp(u_ce3, u_cz3);
                            let j2 = tocsy_pair_amp(u_ce3, u_ch2);
                            if j1 > 0.0 && j2 > 0.0 && j1 >= 1.28 * j2 {
                                sc += 0.9;
                            }
                        }
                        let arr = [o_ce3, o_cz3, o_ch2, Some(u_cz2)];
                        if sc > best_sc {
                            best_sc = sc;
                            best_arr = arr;
                        }
                        hyps.push((arr, sc));
                    }
                }
            }

            let mut slot_confs = [0.95; 4];
            for s_idx in 0..3 {
                let alt_s = hyps
                    .iter()
                    .filter(|&&(a, _)| a[s_idx] != best_arr[s_idx])
                    .map(|&(_, s)| s)
                    .fold(f64::NEG_INFINITY, f64::max);
                slot_confs[s_idx] = if alt_s.is_finite() {
                    margin_to_intra_conf((best_sc - alt_s).max(0.0))
                } else {
                    0.98
                };
            }
            for u in best_arr.into_iter().flatten() {
                trp_node_uses[u] += 1;
                let cap = if u != u_cz2 && nodes[u].height >= 1.72 * h_1x {
                    2
                } else {
                    1
                };
                if trp_node_uses[u] >= cap {
                    used[u] = true;
                }
            }
            benz_chains.push((best_arr, best_sc, slot_confs));
        }

        // Rescue partial TRP benzenoid chains when CZ2 dropped out in AROM_C13HSQC
        if backbone_ref.is_none() || benz_chains.len() < n_trp_seq {
            let mut ce3_rescue_seeds: Vec<usize> = (0..nodes.len())
                .filter(|&u| {
                    !used[u]
                        && !is_likely_tyr_ce(u)
                        && (117.5..=125.8).contains(&nodes[u].c)
                        && has_trp_cb_he(u)
                        && nodes[u].height < 1.58 * h_1x
                })
                .collect();
            ce3_rescue_seeds.sort_by(|&a, &b| {
                nodes[b]
                    .off_diag_tocsy_count(cfg.tol_h)
                    .cmp(&nodes[a].off_diag_tocsy_count(cfg.tol_h))
            });
            for u_ce3 in ce3_rescue_seeds {
                if used[u_ce3] || (backbone_ref.is_some() && benz_chains.len() >= n_trp_seq) {
                    continue;
                }
                let mut pool: Vec<usize> = (0..nodes.len())
                    .filter(|&v| {
                        !used[v]
                            && v != u_ce3
                            && !is_likely_tyr_ce(v)
                            && (118.0..=130.2).contains(&nodes[v].c)
                            && is_verified_tocsy_link(&nodes[u_ce3], &nodes[v], cfg.tol_h)
                    })
                    .collect();
                let cur = pool.clone();
                for &m in &cur {
                    for v in 0..nodes.len() {
                        if !used[v]
                            && v != u_ce3
                            && !is_likely_tyr_ce(v)
                            && !pool.contains(&v)
                            && (118.0..=130.2).contains(&nodes[v].c)
                            && is_verified_tocsy_link(&nodes[m], &nodes[v], cfg.tol_h)
                            && arom_tocsy_edge_score(&nodes[m], &nodes[v], true, cfg.tol_h) > 1.8
                        {
                            pool.push(v);
                        }
                    }
                }
                let mut opts: Vec<Option<usize>> = pool.iter().copied().map(Some).collect();
                opts.push(None);
                let mut best_arr = [Some(u_ce3), None, None, None];
                let mut best_sc =
                    priors.llr_c13("TRP", "CE3", CarbonRegion::Aromatic, nodes[u_ce3].c)
                        + priors.llr_h1("TRP", "HE3", CarbonRegion::Aromatic, nodes[u_ce3].h)
                        + 2.0;
                for &o_cz3 in &opts {
                    for &o_ch2 in &opts {
                        if o_cz3.is_some() && o_cz3 == o_ch2 {
                            continue;
                        }
                        let mut sc =
                            priors.llr_c13("TRP", "CE3", CarbonRegion::Aromatic, nodes[u_ce3].c)
                                + priors.llr_h1(
                                    "TRP",
                                    "HE3",
                                    CarbonRegion::Aromatic,
                                    nodes[u_ce3].h,
                                )
                                + 5.6;
                        if let Some(u) = o_cz3 {
                            sc += priors.llr_c13("TRP", "CZ3", CarbonRegion::Aromatic, nodes[u].c)
                                + priors.llr_h1("TRP", "HZ3", CarbonRegion::Aromatic, nodes[u].h)
                                + arom_tocsy_edge_score(&nodes[u_ce3], &nodes[u], true, cfg.tol_h);
                        } else {
                            sc -= 2.2;
                        }
                        if let Some(u) = o_ch2 {
                            sc += priors.llr_c13("TRP", "CH2", CarbonRegion::Aromatic, nodes[u].c)
                                + priors.llr_h1("TRP", "HH2", CarbonRegion::Aromatic, nodes[u].h);
                        } else {
                            sc -= 2.2;
                        }
                        if let (Some(u2), Some(u3)) = (o_cz3, o_ch2) {
                            sc += arom_tocsy_edge_score(&nodes[u2], &nodes[u3], true, cfg.tol_h);
                        }
                        if sc > best_sc {
                            best_sc = sc;
                            best_arr = [Some(u_ce3), o_cz3, o_ch2, None];
                        }
                    }
                }
                for u in best_arr.into_iter().flatten() {
                    used[u] = true;
                }
                benz_chains.push((best_arr, best_sc, [0.92, 0.88, 0.88, 0.0]));
            }
        }

        // Pair each benzenoid chain with pyrrole CD1/HD1 and TRP_IND_N15HSQC (NE1, HE1)
        let mut cd1_cands: Vec<usize> = Vec::new();
        for u in 0..nodes.len() {
            if !used[u]
                && (119.5..=133.2).contains(&nodes[u].c)
                && nodes[u].off_diag_tocsy_count(cfg.tol_h) == 0
                && (!nodes[u].cb_hd_cands.is_empty() || nodes[u].height < 1.65 * h_1x)
            {
                cd1_cands.push(u);
                if nodes[u].height >= 1.75 * h_1x {
                    cd1_cands.push(u);
                }
            }
        }

        let nb = benz_chains.len();
        if nb > 0 {
            let ce3_h_overlapped: Vec<bool> = (0..nb)
                .map(|ib| {
                    nb > 1
                        && benz_chains[ib].0[0].is_some_and(|u_ce3| {
                            benz_chains
                                .iter()
                                .enumerate()
                                .any(|(jb, &(other_arr, _, _))| {
                                    jb != ib
                                        && other_arr[0].is_some_and(|u2| {
                                            (nodes[u_ce3].h - nodes[u2].h).abs() <= cfg.tol_h * 1.25
                                        })
                                })
                        })
                })
                .collect();

            let mut w_cd1 = vec![vec![FORBID_SCORE; cd1_cands.len().max(1)]; nb];
            if !cd1_cands.is_empty() {
                for (ib, &(arr, _, _)) in benz_chains.iter().enumerate() {
                    for (id, &u_cd1) in cd1_cands.iter().enumerate() {
                        let mut sc =
                            priors.llr_c13("TRP", "CD1", CarbonRegion::Aromatic, nodes[u_cd1].c)
                                + priors.llr_h1(
                                    "TRP",
                                    "HD1",
                                    CarbonRegion::Aromatic,
                                    nodes[u_cd1].h,
                                );
                        if let Some(u_ce3) = arr[0] {
                            let (_, _, cb_sc) = best_shared_yamazaki_cb(
                                &nodes[u_cd1],
                                &nodes[u_ce3],
                                &known_arom_cbs,
                                cfg,
                            );
                            if cb_sc <= -8.0 && ce3_h_overlapped[ib] {
                                // u_ce3's true cb_he may have dropped out while a 1H-degenerate neighbor CE3's cb_he bled onto u_ce3
                                if let Some((_, hd_sc)) = pick_best_single_yamazaki_cb(
                                    &nodes[u_cd1].cb_hd_cands,
                                    &known_trp_cbs,
                                    2.5,
                                    0.5,
                                    cfg,
                                ) {
                                    sc += hd_sc - 0.5;
                                }
                            } else {
                                sc += cb_sc;
                            }
                        }
                        // Bonus if an indole NH root sees BOTH HD1 and HZ2 via NOE!
                        if let Some(u_cz2) = arr[3] {
                            let bridged = trp_ind_roots.iter().any(|r| {
                                let sees_hd1 = trp_noes.iter().any(|np| {
                                    (np.h - r.h).abs() <= cfg.tol_h
                                        && (np.n - r.n).abs() <= cfg.tol_n
                                        && np.h2.is_some_and(|h2| {
                                            (h2 - nodes[u_cd1].h).abs() <= cfg.tol_h * 1.2
                                        })
                                });
                                let sees_hz2 = trp_noes.iter().any(|np| {
                                    (np.h - r.h).abs() <= cfg.tol_h
                                        && (np.n - r.n).abs() <= cfg.tol_n
                                        && np.h2.is_some_and(|h2| {
                                            (h2 - nodes[u_cz2].h).abs() <= cfg.tol_h * 1.2
                                        })
                                });
                                sees_hd1 && sees_hz2
                            });
                            if bridged {
                                sc += 7.5;
                            }
                        }
                        w_cd1[ib][id] = sc;
                    }
                }
            }
            let (b2d, tot_d) = hungarian_max_weight(&w_cd1);
            let d_margins = hungarian_margins(&w_cd1, &b2d, tot_d);

            // Match indole (NE1, HE1) roots to benzenoid+CD1 rings
            let mut used_ind = vec![false; trp_ind_roots.len()];

            for (ib, &(arr, b_sc, s_confs)) in benz_chains.iter().enumerate() {
                let ce3_cb_ambig = if nb <= 1 {
                    false
                } else if let Some(u_ce3) = arr[0] {
                    let cands = &nodes[u_ce3].cb_he_cands;
                    let h_overlapped_other_ce3 =
                        benz_chains
                            .iter()
                            .enumerate()
                            .any(|(jb, &(other_arr, _, _))| {
                                jb != ib
                                    && other_arr[0].is_some_and(|u2| {
                                        (nodes[u_ce3].h - nodes[u2].h).abs() <= cfg.tol_h * 1.25
                                    })
                            });
                    if cands.is_empty() || h_overlapped_other_ce3 {
                        trp_noes.is_empty()
                    } else {
                        cands.iter().enumerate().any(|(i, &(cb1, _))| {
                            cands[(i + 1)..]
                                .iter()
                                .any(|&(cb2, _)| (cb1 - cb2).abs() > cfg.tol_cb * 0.65)
                        }) && trp_noes.is_empty()
                    }
                } else {
                    trp_noes.is_empty()
                };
                let benz_link_scale = if ce3_cb_ambig {
                    margin_to_intra_conf(d_margins[ib])
                } else {
                    1.0
                };

                let mut slots = BTreeMap::new();
                let names = ["CE3", "CZ3", "CH2", "CZ2"];
                for (si, &c_name) in names.iter().enumerate() {
                    if let Some(u) = arr[si] {
                        slots.insert(
                            c_name,
                            RingChSlot {
                                c_ppm: nodes[u].c,
                                h_ppm: nodes[u].h,
                                intra_conf: s_confs[si] * benz_link_scale,
                            },
                        );
                    }
                }

                let mut cb_hd = None;
                let mut cb_he = arr[0].and_then(|u| {
                    pick_best_single_yamazaki_cb(
                        &nodes[u].cb_he_cands,
                        &known_arom_cbs,
                        2.2,
                        0.4,
                        cfg,
                    )
                    .map(|x| x.0)
                });

                if !cd1_cands.is_empty() {
                    if let Some(id) = b2d[ib].filter(|&id| w_cd1[ib][id] > -4.0) {
                        let u_cd1 = cd1_cands[id];
                        used[u_cd1] = true;
                        let cd1_conf = if nb == 1 && cd1_cands.len() == 1 {
                            0.98
                        } else {
                            margin_to_intra_conf(d_margins[ib])
                        };
                        slots.insert(
                            "CD1",
                            RingChSlot {
                                c_ppm: nodes[u_cd1].c,
                                h_ppm: nodes[u_cd1].h,
                                intra_conf: cd1_conf,
                            },
                        );
                        if let Some(u_ce3) = arr[0] {
                            let (c1, c2, sh_sc) = best_shared_yamazaki_cb(
                                &nodes[u_cd1],
                                &nodes[u_ce3],
                                &known_arom_cbs,
                                cfg,
                            );
                            cb_hd = c1.or_else(|| {
                                pick_best_single_yamazaki_cb(
                                    &nodes[u_cd1].cb_hd_cands,
                                    &known_arom_cbs,
                                    2.5,
                                    0.5,
                                    cfg,
                                )
                                .map(|x| x.0)
                            });
                            if sh_sc <= -8.0 && ce3_h_overlapped[ib] {
                                cb_he = None;
                            } else {
                                cb_he = c2.or(cb_he);
                            }
                        } else {
                            cb_hd = pick_best_single_yamazaki_cb(
                                &nodes[u_cd1].cb_hd_cands,
                                &known_arom_cbs,
                                2.5,
                                0.5,
                                cfg,
                            )
                            .map(|x| x.0);
                        }
                    } else if ce3_h_overlapped[ib] {
                        if let Some(c_spill) = cb_he {
                            let claimed_by_other = (0..nb).any(|jb| {
                                jb != ib
                                    && benz_chains[jb].0[0].is_some_and(|u2| {
                                        nodes[u2].cb_he_cands.iter().any(|&(cb2, _)| {
                                            (cb2 - c_spill).abs() <= cfg.tol_cb * 0.75
                                        })
                                    })
                            });
                            if claimed_by_other {
                                cb_he = None;
                            }
                        }
                    }
                }

                // Match indole (NE1, HE1) root
                let mut best_ind: Option<(usize, f64)> = None;
                let mut second_ind = f64::NEG_INFINITY;
                for (ir, r) in trp_ind_roots.iter().enumerate() {
                    if used_ind[ir] {
                        continue;
                    }
                    let mut isc = 1.5;
                    if let Some(cd1_s) = slots.get("CD1") {
                        if trp_noes.iter().any(|np| {
                            (np.h - r.h).abs() <= cfg.tol_h
                                && (np.n - r.n).abs() <= cfg.tol_n
                                && np
                                    .h2
                                    .is_some_and(|h2| (h2 - cd1_s.h_ppm).abs() <= cfg.tol_h * 1.25)
                        }) {
                            isc += 5.5;
                        }
                    }
                    if let Some(cz2_s) = slots.get("CZ2") {
                        if trp_noes.iter().any(|np| {
                            (np.h - r.h).abs() <= cfg.tol_h
                                && (np.n - r.n).abs() <= cfg.tol_n
                                && np
                                    .h2
                                    .is_some_and(|h2| (h2 - cz2_s.h_ppm).abs() <= cfg.tol_h * 1.25)
                        }) {
                            isc += 5.5;
                        }
                    }
                    if best_ind.is_none_or(|(_, bs)| isc > bs) {
                        if let Some((_, bs)) = best_ind {
                            second_ind = second_ind.max(bs);
                        }
                        best_ind = Some((ir, isc));
                    } else if isc > second_ind {
                        second_ind = isc;
                    }
                }
                let trp_indole = best_ind.and_then(|(ir, isc)| {
                    if isc > 2.0 || (nb == 1 && trp_ind_roots.len() == 1) {
                        used_ind[ir] = true;
                        let icf = if nb == 1 && trp_ind_roots.len() == 1 {
                            0.98
                        } else {
                            margin_to_intra_conf((isc - second_ind).max(0.0))
                        };
                        Some((trp_ind_roots[ir].n, trp_ind_roots[ir].h, icf))
                    } else {
                        None
                    }
                });

                clusters.push(AromaticRingCluster {
                    kind: AromaticRingKind::Trp,
                    ch_slots: slots,
                    cb_hd,
                    cb_he,
                    nd1_ppm: None,
                    ne2_ppm: None,
                    his_tautomer: None,
                    his_state: None,
                    trp_indole,
                    internal_coherence: b_sc,
                });
            }
        }

        // Rescue remaining unambiguous TRP CD1 singlets when benzenoid CZ2/CE3 dropped out
        for &u_cd1 in &cd1_cands {
            if used[u_cd1]
                || (backbone_ref.is_some()
                    && clusters
                        .iter()
                        .filter(|c| c.kind == AromaticRingKind::Trp)
                        .count()
                        >= n_trp_seq)
            {
                continue;
            }
            let matches_trp_cb = !known_trp_cbs.is_empty()
                && nodes[u_cd1].cb_hd_cands.iter().any(|&(cb, _)| {
                    known_trp_cbs.iter().any(|&w| (w - cb).abs() <= cfg.tol_cb)
                        && !known_phe_cbs.iter().any(|&f| (f - cb).abs() <= cfg.tol_cb)
                        && !known_tyr_cbs.iter().any(|&y| (y - cb).abs() <= cfg.tol_cb)
                });
            if (121.5..=130.5).contains(&nodes[u_cd1].c)
                && nodes[u_cd1].height < 1.55 * h_1x
                && matches_trp_cb
            {
                used[u_cd1] = true;
                let cb_hd = pick_best_single_yamazaki_cb(
                    &nodes[u_cd1].cb_hd_cands,
                    &known_trp_cbs,
                    2.5,
                    0.5,
                    cfg,
                )
                .map(|x| x.0);
                let mut slots = BTreeMap::new();
                slots.insert(
                    "CD1",
                    RingChSlot {
                        c_ppm: nodes[u_cd1].c,
                        h_ppm: nodes[u_cd1].h,
                        intra_conf: 0.90,
                    },
                );
                clusters.push(AromaticRingCluster {
                    kind: AromaticRingKind::Trp,
                    ch_slots: slots,
                    cb_hd,
                    cb_he: None,
                    nd1_ppm: None,
                    ne2_ppm: None,
                    his_tautomer: None,
                    his_state: None,
                    trp_indole: None,
                    internal_coherence: 3.5,
                });
            }
        }
    }

    // ------------------------------------------------------------------------
    // Family 3: Tyrosine Phenol Rings (CE* ~ 117.9 ppm <-> CD* ~ 132.8 ppm)
    // ------------------------------------------------------------------------
    if backbone_ref.is_none() || n_tyr_seq > 0 {
        let mut tyr_ce_slots: Vec<usize> = Vec::new();
        for (u, n) in nodes.iter().enumerate() {
            if used[u] || !(111.0..=124.5).contains(&n.c) {
                continue;
            }
            let plausible_tyr_ce = n.height >= 1.32 * h_1x
                || ((112.5..=121.8).contains(&n.c)
                    && (n.off_diag_tocsy_count(cfg.tol_h) >= 1 || !n.cb_he_cands.is_empty()));
            if !plausible_tyr_ce {
                continue;
            }
            tyr_ce_slots.push(u);
        }
        if backbone_ref.is_some() && tyr_ce_slots.len() < n_tyr_seq {
            let mut by_ht = tyr_ce_slots.clone();
            by_ht.sort_by(|&a, &b| nodes[b].height.total_cmp(&nodes[a].height));
            for u in by_ht {
                if tyr_ce_slots.len() < n_tyr_seq && nodes[u].height >= 2.85 * h_1x {
                    tyr_ce_slots.push(u);
                }
            }
        }
        let mut tyr_cd_slots: Vec<usize> = Vec::new();
        for (u, n) in nodes.iter().enumerate() {
            if used[u] || !(125.5..=138.8).contains(&n.c) {
                continue;
            }
            let linked_to_tyr_ce = tyr_ce_slots.iter().any(|&u_ce| {
                arom_tocsy_edge_score(n, &nodes[u_ce], true, cfg.tol_h) > 0.4
                    || best_shared_yamazaki_cb(n, &nodes[u_ce], &known_arom_cbs, cfg).2 > 2.8
            });
            if !linked_to_tyr_ce && n_phe_seq > 0 {
                continue;
            }
            tyr_cd_slots.push(u);
            if n.height >= 3.10 * h_1x {
                tyr_cd_slots.push(u);
            }
        }

        if !tyr_ce_slots.is_empty() && !tyr_cd_slots.is_empty() {
            let nr = tyr_ce_slots.len();
            let nc = tyr_cd_slots.len();
            let mut w = vec![vec![FORBID_SCORE; nc]; nr];
            for (ir, &u_ce) in tyr_ce_slots.iter().enumerate() {
                for (ic, &u_cd) in tyr_cd_slots.iter().enumerate() {
                    let tocsy = arom_tocsy_edge_score(&nodes[u_cd], &nodes[u_ce], true, cfg.tol_h);
                    let (_, _, cb_sc) =
                        best_shared_yamazaki_cb(&nodes[u_cd], &nodes[u_ce], &known_arom_cbs, cfg);
                    let pr = priors.llr_c13("TYR", "CD1", CarbonRegion::Aromatic, nodes[u_cd].c)
                        + priors.llr_h1("TYR", "HD1", CarbonRegion::Aromatic, nodes[u_cd].h)
                        + priors.llr_c13("TYR", "CE1", CarbonRegion::Aromatic, nodes[u_ce].c)
                        + priors.llr_h1("TYR", "HE1", CarbonRegion::Aromatic, nodes[u_ce].h);
                    let mult_bonus = if nodes[u_cd].height >= 1.40 * h_1x {
                        1.4
                    } else {
                        -0.8
                    };
                    w[ir][ic] = tocsy + cb_sc + 0.55 * pr + mult_bonus;
                }
            }
            let (r2c, best_tot) = hungarian_max_weight(&w);
            let margins = hungarian_margins(&w, &r2c, best_tot);

            for (ir, &u_ce) in tyr_ce_slots.iter().enumerate() {
                used[u_ce] = true;
                let mut slots = BTreeMap::new();
                slots.insert(
                    "CE1",
                    RingChSlot {
                        c_ppm: nodes[u_ce].c,
                        h_ppm: nodes[u_ce].h,
                        intra_conf: 0.98,
                    },
                );
                if let Some(ic) = r2c[ir].filter(|&ic| w[ir][ic] > -3.0) {
                    let u_cd = tyr_cd_slots[ic];
                    used[u_cd] = true;
                    let (cb_hd, cb_he, cb_sc) =
                        best_shared_yamazaki_cb(&nodes[u_cd], &nodes[u_ce], &known_arom_cbs, cfg);
                    let cd_conf = if nr == 1 {
                        0.98
                    } else {
                        margin_to_intra_conf(margins[ir].max(cb_sc * 0.5))
                    };
                    slots.insert(
                        "CD1",
                        RingChSlot {
                            c_ppm: nodes[u_cd].c,
                            h_ppm: nodes[u_cd].h,
                            intra_conf: cd_conf,
                        },
                    );
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::Tyr,
                        ch_slots: slots,
                        cb_hd,
                        cb_he,
                        nd1_ppm: None,
                        ne2_ppm: None,
                        his_tautomer: None,
                        his_state: None,
                        trp_indole: None,
                        internal_coherence: w[ir][ic],
                    });
                } else {
                    let cb_he = pick_best_single_yamazaki_cb(
                        &nodes[u_ce].cb_he_cands,
                        &known_arom_cbs,
                        2.2,
                        0.4,
                        cfg,
                    )
                    .map(|x| x.0);
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::Tyr,
                        ch_slots: slots,
                        cb_hd: None,
                        cb_he,
                        nd1_ppm: None,
                        ne2_ppm: None,
                        his_tautomer: None,
                        his_state: None,
                        trp_indole: None,
                        internal_coherence: 0.0,
                    });
                }
            }
        }

        // Rescue unclaimed TYR CD* nodes whose CE* partner dropped out in AROM_C13HSQC
        if backbone_ref.is_some()
            && clusters
                .iter()
                .filter(|c| c.kind == AromaticRingKind::Tyr)
                .count()
                < n_tyr_seq
        {
            for u in 0..nodes.len() {
                if used[u] || !(129.0..=137.5).contains(&nodes[u].c) {
                    continue;
                }
                let matches_tyr_cb_uniquely = nodes[u].cb_hd_cands.iter().any(|&(cb, _)| {
                    known_tyr_cbs
                        .iter()
                        .any(|&tcb| (tcb - cb).abs() <= cfg.tol_cb * 0.75)
                        && !known_phe_cbs
                            .iter()
                            .any(|&pcb| (pcb - cb).abs() <= cfg.tol_cb * 0.75)
                });
                if matches_tyr_cb_uniquely {
                    used[u] = true;
                    let cb_hd = pick_best_single_yamazaki_cb(
                        &nodes[u].cb_hd_cands,
                        &known_tyr_cbs,
                        2.5,
                        0.5,
                        cfg,
                    )
                    .map(|x| x.0);
                    let mut slots = BTreeMap::new();
                    slots.insert(
                        "CD1",
                        RingChSlot {
                            c_ppm: nodes[u].c,
                            h_ppm: nodes[u].h,
                            intra_conf: 0.90,
                        },
                    );
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::Tyr,
                        ch_slots: slots,
                        cb_hd,
                        cb_he: None,
                        nd1_ppm: None,
                        ne2_ppm: None,
                        his_tautomer: None,
                        his_state: None,
                        trp_indole: None,
                        internal_coherence: 1.0,
                    });
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Family 4: Phenylalanine Rings (CD* <-> CE* <-> CZ) via Two-Stage Hungarian
    // ------------------------------------------------------------------------
    if backbone_ref.is_none() || n_phe_seq > 0 {
        let rem: Vec<usize> = (0..nodes.len())
            .filter(|&u| !used[u] && (122.5..=138.8).contains(&nodes[u].c))
            .collect();

        if !rem.is_empty() {
            // Classify candidate CD*, CE*, and CZ pools from Yamazaki cross-peaks, 2x vs 1x height, and priors
            let mut cd_pool: Vec<usize> = Vec::new();
            let mut ce_pool: Vec<usize> = Vec::new();

            for &u in &rem {
                let n = &nodes[u];
                let is_2x = n.height >= 1.38 * h_1x;
                let shadowed_by_2x = !is_2x
                    && rem.iter().any(|&v| {
                        v != u
                            && nodes[v].height >= 1.38 * h_1x
                            && (nodes[v].h - n.h).abs() <= cfg.tol_h * 1.25
                    });
                let has_hd = !shadowed_by_2x
                    && (is_2x || n.c >= 129.4)
                    && n.cb_hd_cands.iter().any(|&(cb, _)| {
                        known_phe_cbs.is_empty()
                            || known_phe_cbs
                                .iter()
                                .any(|&pcb| (pcb - cb).abs() <= cfg.tol_cb)
                    });
                let has_he = !shadowed_by_2x
                    && (is_2x || n.c >= 127.8)
                    && n.cb_he_cands.iter().any(|&(cb, _)| {
                        known_phe_cbs.is_empty()
                            || known_phe_cbs
                                .iter()
                                .any(|&pcb| (pcb - cb).abs() <= cfg.tol_cb)
                    });
                if has_hd || is_2x {
                    cd_pool.push(u);
                    if n.height >= 2.90 * h_1x {
                        cd_pool.push(u);
                    }
                }
                if has_he || is_2x {
                    ce_pool.push(u);
                    if n.height >= 2.90 * h_1x {
                        ce_pool.push(u);
                    }
                }
            }

            // Stage 4A: Match CD* <-> CE* stems via Hungarian (avoiding u_cd == u_ce unless >=2.55x coalesced)
            #[allow(clippy::type_complexity)]
            let mut stems: Vec<(
                Option<usize>,
                Option<usize>,
                Option<f64>,
                Option<f64>,
                f64,
                f64,
            )> = Vec::new();
            let mut stem_used: BTreeMap<usize, usize> = BTreeMap::new();

            if !cd_pool.is_empty() && !ce_pool.is_empty() {
                let nr = cd_pool.len();
                let nc = ce_pool.len();
                let mut w_stem = vec![vec![FORBID_SCORE; nc]; nr];
                for (ir, &u_cd) in cd_pool.iter().enumerate() {
                    for (ic, &u_ce) in ce_pool.iter().enumerate() {
                        if u_cd == u_ce && nodes[u_cd].height < 2.55 * h_1x {
                            continue;
                        }
                        if nodes[u_cd].height < 1.38 * h_1x && nodes[u_ce].height < 1.38 * h_1x {
                            continue;
                        }
                        let tocsy =
                            arom_tocsy_edge_score(&nodes[u_cd], &nodes[u_ce], true, cfg.tol_h);
                        let (_, _, cb_sc) = best_shared_yamazaki_cb(
                            &nodes[u_cd],
                            &nodes[u_ce],
                            &known_arom_cbs,
                            cfg,
                        );
                        if cb_sc <= -8.0 && tocsy < 1.0 {
                            continue;
                        }
                        let pr =
                            priors.llr_c13("PHE", "CD1", CarbonRegion::Aromatic, nodes[u_cd].c)
                                + priors.llr_h1(
                                    "PHE",
                                    "HD1",
                                    CarbonRegion::Aromatic,
                                    nodes[u_cd].h,
                                )
                                + priors.llr_c13(
                                    "PHE",
                                    "CE1",
                                    CarbonRegion::Aromatic,
                                    nodes[u_ce].c,
                                )
                                + priors.llr_h1(
                                    "PHE",
                                    "HE1",
                                    CarbonRegion::Aromatic,
                                    nodes[u_ce].h,
                                );
                        let role_bonus =
                            (if !nodes[u_cd].cb_hd_cands.is_empty() {
                                2.2
                            } else {
                                0.0
                            }) + (if !nodes[u_ce].cb_he_cands.is_empty() && cb_sc > -2.0 {
                                2.2
                            } else {
                                0.0
                            });
                        let mult_bonus = (if nodes[u_cd].height >= 1.38 * h_1x {
                            2.6
                        } else {
                            -2.6
                        }) + (if nodes[u_ce].height >= 1.38 * h_1x {
                            2.6
                        } else {
                            -2.6
                        });
                        w_stem[ir][ic] = tocsy + cb_sc + 0.45 * pr + role_bonus + mult_bonus;
                    }
                }
                let (r2c, tot_s) = hungarian_max_weight(&w_stem);
                let s_margins = hungarian_margins(&w_stem, &r2c, tot_s);

                // Sort accepted stems by score and enforce physical node capacity (1, or 2 if >=3.1x)
                let mut raw_stems: Vec<(usize, usize, f64, f64)> = Vec::new();
                for (ir, &u_cd) in cd_pool.iter().enumerate() {
                    if let Some(ic) = r2c[ir].filter(|&ic| w_stem[ir][ic] > -2.5) {
                        raw_stems.push((u_cd, ce_pool[ic], w_stem[ir][ic], s_margins[ir]));
                    }
                }
                raw_stems.sort_by(|a, b| b.2.total_cmp(&a.2));

                for (u_cd, u_ce, sc, mrg) in raw_stems {
                    let cap_cd = if nodes[u_cd].height >= 2.55 * h_1x {
                        2
                    } else {
                        1
                    };
                    let cap_ce = if nodes[u_ce].height >= 2.55 * h_1x {
                        2
                    } else {
                        1
                    };
                    let used_cd = stem_used.get(&u_cd).copied().unwrap_or(0);
                    let used_ce = stem_used.get(&u_ce).copied().unwrap_or(0);
                    if used_cd >= cap_cd
                        || used_ce >= cap_ce
                        || (u_cd == u_ce && used_cd + 2 > cap_cd)
                    {
                        continue;
                    }
                    *stem_used.entry(u_cd).or_insert(0) += 1;
                    *stem_used.entry(u_ce).or_insert(0) += 1;
                    let (cb_hd, cb_he, _) =
                        best_shared_yamazaki_cb(&nodes[u_cd], &nodes[u_ce], &known_arom_cbs, cfg);
                    let st_conf = if nr == 1 && nc == 1 {
                        0.98
                    } else {
                        margin_to_intra_conf(mrg)
                    };
                    stems.push((Some(u_cd), Some(u_ce), cb_hd, cb_he, sc, st_conf));
                }
            }

            // Rescue any remaining single-sided 2x / Yamazaki-anchored PHE nodes whose partner dropped out
            if backbone_ref.is_none() || stems.len() < n_phe_seq {
                for &u in &rem {
                    if stem_used.contains_key(&u)
                        || (backbone_ref.is_some() && stems.len() >= n_phe_seq)
                    {
                        continue;
                    }
                    let is_2x = nodes[u].height >= 1.38 * h_1x;
                    let shadowed_by_stem = !is_2x
                        && stem_used.keys().any(|&v| {
                            v != u && (nodes[v].h - nodes[u].h).abs() <= cfg.tol_h * 1.25
                        });
                    if shadowed_by_stem {
                        continue;
                    }
                    let hd_phe_ok = (is_2x || nodes[u].c >= 129.4)
                        && nodes[u].cb_hd_cands.iter().any(|&(cb, _)| {
                            known_phe_cbs.is_empty()
                                || known_phe_cbs
                                    .iter()
                                    .any(|&pcb| (pcb - cb).abs() <= cfg.tol_cb)
                        });
                    let he_phe_ok = (is_2x || nodes[u].c >= 127.8)
                        && nodes[u].cb_he_cands.iter().any(|&(cb, _)| {
                            known_phe_cbs.is_empty()
                                || known_phe_cbs
                                    .iter()
                                    .any(|&pcb| (pcb - cb).abs() <= cfg.tol_cb)
                        });
                    if hd_phe_ok {
                        *stem_used.entry(u).or_insert(0) += 1;
                        let cb_hd = pick_best_single_yamazaki_cb(
                            &nodes[u].cb_hd_cands,
                            &known_phe_cbs,
                            2.5,
                            0.5,
                            cfg,
                        )
                        .map(|x| x.0);
                        stems.push((Some(u), None, cb_hd, None, 1.0, 0.88));
                    } else if he_phe_ok {
                        *stem_used.entry(u).or_insert(0) += 1;
                        let cb_he = pick_best_single_yamazaki_cb(
                            &nodes[u].cb_he_cands,
                            &known_phe_cbs,
                            2.2,
                            0.4,
                            cfg,
                        )
                        .map(|x| x.0);
                        stems.push((None, Some(u), None, cb_he, 1.0, 0.88));
                    }
                }
            }

            // Stage 4B: Match unclaimed 1x CZ/HZ nodes to PHE stems via Hungarian
            let mut cz_pool: Vec<usize> = Vec::new();
            for &u in &rem {
                let u_cnt = stem_used.get(&u).copied().unwrap_or(0);
                if u_cnt == 0 {
                    cz_pool.push(u);
                    if nodes[u].height >= 1.75 * h_1x {
                        cz_pool.push(u);
                    }
                }
            }

            let ns = stems.len();
            if ns > 0 {
                let mut w_cz = vec![vec![FORBID_SCORE; cz_pool.len().max(1)]; ns];
                if !cz_pool.is_empty() {
                    for (is, &(o_cd, o_ce, _, _, _, _)) in stems.iter().enumerate() {
                        for (iz, &u_cz) in cz_pool.iter().enumerate() {
                            let mut sc =
                                priors.llr_c13("PHE", "CZ", CarbonRegion::Aromatic, nodes[u_cz].c)
                                    + priors.llr_h1(
                                        "PHE",
                                        "HZ",
                                        CarbonRegion::Aromatic,
                                        nodes[u_cz].h,
                                    );
                            if let Some(u_ce) = o_ce {
                                sc += 1.35
                                    * arom_tocsy_edge_score(
                                        &nodes[u_ce],
                                        &nodes[u_cz],
                                        true,
                                        cfg.tol_h,
                                    );
                            }
                            if let Some(u_cd) = o_cd {
                                sc += 0.85
                                    * arom_tocsy_edge_score(
                                        &nodes[u_cd],
                                        &nodes[u_cz],
                                        false,
                                        cfg.tol_h,
                                    );
                            }
                            if nodes[u_cz].cb_hd_cands.is_empty()
                                && nodes[u_cz].cb_he_cands.is_empty()
                            {
                                sc += 1.2;
                            }
                            w_cz[is][iz] = sc;
                        }
                    }
                }
                let (s2z, tot_z) = hungarian_max_weight(&w_cz);
                let z_margins = hungarian_margins(&w_cz, &s2z, tot_z);

                for (is, &(o_cd, o_ce, cb_hd, cb_he, st_sc, st_conf)) in stems.iter().enumerate() {
                    let mut slots = BTreeMap::new();
                    if let Some(u_cd) = o_cd {
                        used[u_cd] = true;
                        slots.insert(
                            "CD1",
                            RingChSlot {
                                c_ppm: nodes[u_cd].c,
                                h_ppm: nodes[u_cd].h,
                                intra_conf: 0.98,
                            },
                        );
                    }
                    if let Some(u_ce) = o_ce {
                        used[u_ce] = true;
                        slots.insert(
                            "CE1",
                            RingChSlot {
                                c_ppm: nodes[u_ce].c,
                                h_ppm: nodes[u_ce].h,
                                intra_conf: st_conf,
                            },
                        );
                    }
                    if !cz_pool.is_empty() {
                        if let Some(iz) = s2z[is].filter(|&iz| w_cz[is][iz] > -2.5) {
                            let u_cz = cz_pool[iz];
                            used[u_cz] = true;
                            let cz_conf = if ns == 1 && cz_pool.len() == 1 {
                                0.98
                            } else {
                                (st_conf * margin_to_intra_conf(z_margins[is])).clamp(0.35, 1.0)
                            };
                            slots.insert(
                                "CZ",
                                RingChSlot {
                                    c_ppm: nodes[u_cz].c,
                                    h_ppm: nodes[u_cz].h,
                                    intra_conf: cz_conf,
                                },
                            );
                        }
                    }
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::Phe,
                        ch_slots: slots,
                        cb_hd,
                        cb_he,
                        nd1_ppm: None,
                        ne2_ppm: None,
                        his_tautomer: None,
                        his_state: None,
                        trp_indole: None,
                        internal_coherence: st_sc,
                    });
                }
            }
        }
    }

    clusters
}

// ============================================================================
// 6. Probabilistic Bipartite Ring Matcher & Calibrated Marginal Confidence
// ============================================================================

/// Assigned aromatic ring output for a single `PHE`, `TYR`, `TRP`, or `HIS` residue.
#[derive(Clone, Debug)]
pub struct AssignedAromaticRing {
    pub residue_idx: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub cluster_idx: Option<usize>,
    /// Multi-candidate Boltzmann bipartite ring-to-residue posterior confidence in `[0.0, 1.0]`.
    pub ring_confidence: f64,
    /// Log-likelihood ratio score of `(cluster_idx, residue_idx)`.
    pub score: f64,
    /// Leave-one-out bipartite margin `S* - S_alt(k)` against the best competing ring/residue swap.
    pub margin: f64,
    /// Classified static histidine tautomer and posterior confidence (`None` if non-`HIS`,
    /// unassigned, `FastExchangeUncommitted`, or `MetalCoordinatedCandidate`).
    pub his_tautomer: Option<(HisTautomer, f64)>,
    /// Full histidine state classification (`Resolved`, `FastExchangeUncommitted`, or `MetalCoordinatedCandidate`).
    pub his_state: Option<(HisStateClassification, f64)>,
}

/// Complete output of `assign_aromatic_rings`.
#[derive(Clone, Debug)]
pub struct AromaticAssignResult {
    /// Updated `Reference` combining input shifts with newly assigned aromatic `1H`/`13C`/`15N` shifts
    /// while preserving pre-existing shifts and ambiguity codes on untouched atoms.
    pub reference: Reference,
    /// Calibrated per-atom confidence in `[0.0, 1.0]` keyed ONLY by newly assigned `(residue_idx, iupac_atom_name)` atoms.
    pub atom_confidence: BTreeMap<(usize, String), f64>,
    /// Per-aromatic-residue assignment details, `HisTautomer` calls, and `HisStateClassification`.
    pub rings: Vec<AssignedAromaticRing>,
    /// Assembled experimental ring clusters.
    pub clusters: Vec<AromaticRingCluster>,
}

fn is_aromatic_ring_atom(res_name: &str, atom: &str) -> bool {
    match res_name {
        "PHE" | "TYR" => matches!(
            atom,
            "CD1" | "CD2" | "HD1" | "HD2" | "CE1" | "CE2" | "HE1" | "HE2" | "CZ" | "HZ" | "CG"
        ),
        "TRP" => matches!(
            atom,
            "CD1"
                | "HD1"
                | "NE1"
                | "HE1"
                | "CE3"
                | "HE3"
                | "CZ3"
                | "HZ3"
                | "CH2"
                | "HH2"
                | "CZ2"
                | "HZ2"
                | "CG"
                | "CD2"
                | "CE2"
        ),
        "HIS" => matches!(
            atom,
            "CD2" | "HD2" | "CE1" | "HE1" | "ND1" | "NE2" | "CG" | "HD1" | "HE2"
        ),
        _ => false,
    }
}

fn aromatic_ring_position_key(res_name: &str, atom: &str) -> Option<&'static str> {
    match (res_name, atom) {
        ("PHE" | "TYR", "CD1" | "CD2" | "HD1" | "HD2") => Some("CD"),
        ("PHE" | "TYR", "CE1" | "CE2" | "HE1" | "HE2") => Some("CE"),
        ("PHE" | "TYR", "CZ" | "HZ" | "HH") => Some("CZ"),
        ("PHE" | "TYR", "CG") => Some("CG"),
        ("TRP", "CD1" | "HD1") => Some("CD1"),
        ("TRP", "NE1" | "HE1") => Some("NE1"),
        ("TRP", "CE3" | "HE3") => Some("CE3"),
        ("TRP", "CZ3" | "HZ3") => Some("CZ3"),
        ("TRP", "CH2" | "HH2") => Some("CH2"),
        ("TRP", "CZ2" | "HZ2") => Some("CZ2"),
        ("TRP", "CG") => Some("CG"),
        ("TRP", "CD2") => Some("CD2"),
        ("TRP", "CE2") => Some("CE2"),
        ("HIS", "CD2" | "HD2") => Some("CD2"),
        ("HIS", "CE1" | "HE1") => Some("CE1"),
        ("HIS", "ND1" | "HD1") => Some("ND1"),
        ("HIS", "NE2" | "HE2") => Some("NE2"),
        ("HIS", "CG") => Some("CG"),
        _ => None,
    }
}

#[inline]
fn curated_aromatic_atom_tol(atom: &str) -> f64 {
    if atom.starts_with('H') {
        0.06
    } else if atom.starts_with('C') {
        0.55
    } else {
        0.85
    }
}

fn residue_curated_ring_atoms(
    existing_ref: &Reference,
    k: usize,
    res_name: &str,
) -> Vec<(String, f64)> {
    existing_ref
        .shifts
        .iter()
        .filter(|&(&(rk, ref a), &v)| {
            rk == k && is_aromatic_ring_atom(res_name, a) && v.is_finite()
        })
        .map(|((_, a), &v)| (a.clone(), v))
        .collect()
}

fn cluster_atom_shifts_for_residue(
    clu: &AromaticRingCluster,
    res_name: &str,
) -> Vec<(&'static str, f64)> {
    let mut out = Vec::new();
    for (&c_atom, slot) in &clu.ch_slots {
        match (res_name, c_atom) {
            ("PHE" | "TYR", "CD1") => {
                out.extend_from_slice(&[
                    ("CD1", slot.c_ppm),
                    ("CD2", slot.c_ppm),
                    ("HD1", slot.h_ppm),
                    ("HD2", slot.h_ppm),
                ]);
            }
            ("PHE" | "TYR", "CE1") => {
                out.extend_from_slice(&[
                    ("CE1", slot.c_ppm),
                    ("CE2", slot.c_ppm),
                    ("HE1", slot.h_ppm),
                    ("HE2", slot.h_ppm),
                ]);
            }
            ("PHE", "CZ") => {
                out.extend_from_slice(&[("CZ", slot.c_ppm), ("HZ", slot.h_ppm)]);
            }
            ("TRP", "CD1") => out.extend_from_slice(&[("CD1", slot.c_ppm), ("HD1", slot.h_ppm)]),
            ("TRP", "CE3") => out.extend_from_slice(&[("CE3", slot.c_ppm), ("HE3", slot.h_ppm)]),
            ("TRP", "CZ3") => out.extend_from_slice(&[("CZ3", slot.c_ppm), ("HZ3", slot.h_ppm)]),
            ("TRP", "CH2") => out.extend_from_slice(&[("CH2", slot.c_ppm), ("HH2", slot.h_ppm)]),
            ("TRP", "CZ2") => out.extend_from_slice(&[("CZ2", slot.c_ppm), ("HZ2", slot.h_ppm)]),
            ("HIS", "CD2") => out.extend_from_slice(&[("CD2", slot.c_ppm), ("HD2", slot.h_ppm)]),
            ("HIS", "CE1") => out.extend_from_slice(&[("CE1", slot.c_ppm), ("HE1", slot.h_ppm)]),
            _ => {}
        }
    }
    if res_name == "TRP" {
        if let Some((ne1, he1, _)) = clu.trp_indole {
            out.extend_from_slice(&[("NE1", ne1), ("HE1", he1)]);
        }
    }
    if res_name == "HIS" {
        if let Some((nd1, _)) = clu.nd1_ppm {
            out.push(("ND1", nd1));
        }
        if let Some((ne2, _)) = clu.ne2_ppm {
            out.push(("NE2", ne2));
        }
    }
    out
}

fn his_evidence_from_ring_map(
    atoms: &BTreeMap<String, f64>,
    cb_ppm: Option<f64>,
) -> HisTautomerEvidence {
    HisTautomerEvidence {
        nd1_ppm: atoms.get("ND1").copied(),
        ne2_ppm: atoms.get("NE2").copied(),
        cg_ppm: atoms.get("CG").copied(),
        cd2_ppm: atoms.get("CD2").copied(),
        hd2_ppm: atoms.get("HD2").copied(),
        ce1_ppm: atoms.get("CE1").copied(),
        he1_ppm: atoms.get("HE1").copied(),
        cb_ppm,
        sample_ph: None,
        metal_coordinated_hint: false,
    }
}

fn merge_aromatic_into_shifts_internal(
    existing_ref: &Reference,
    newly_assigned: &Reference,
    newly_assigned_atoms: &BTreeMap<(usize, String), f64>,
    solver_corroborated_positions: Option<&BTreeMap<usize, BTreeSet<&'static str>>>,
    overwrite_existing: bool,
) -> Reference {
    let mut merged = existing_ref.clone();
    if overwrite_existing {
        // Selective per-atom overwrite: updates ONLY finite shifts explicitly present in
        // `newly_assigned_atoms`, and NEVER strips curated ring atoms that are absent from
        // a partial solver cluster!
        for key in newly_assigned_atoms.keys() {
            let Some(&val) = newly_assigned.shifts.get(key) else {
                continue;
            };
            if !val.is_finite() {
                continue;
            }
            merged.shifts.insert(key.clone(), val);
            if let Some(&code) = newly_assigned.ambiguity_codes.get(key) {
                merged.ambiguity_codes.insert(key.clone(), code);
            } else {
                let res_name = existing_ref
                    .sequence
                    .get(key.0)
                    .or_else(|| newly_assigned.sequence.get(key.0))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                merged.ambiguity_codes.insert(
                    key.clone(),
                    default_aromatic_atom_ambiguity_code(res_name, &key.1),
                );
            }
        }
        return merged;
    }

    let mut residues: BTreeSet<usize> = BTreeSet::new();
    for &(k, _) in newly_assigned_atoms.keys() {
        residues.insert(k);
    }

    for k in residues {
        let res_name = existing_ref
            .sequence
            .get(k)
            .or_else(|| newly_assigned.sequence.get(k))
            .map(|s| s.as_str())
            .unwrap_or("");
        let curated_ring = residue_curated_ring_atoms(existing_ref, k, res_name);
        if !curated_ring.is_empty() {
            // (a) Reject if any overlapping ring shift in newly_assigned is non-finite or conflicts with curated_ring
            let has_conflict = curated_ring.iter().any(|(atom, cur_v)| {
                newly_assigned
                    .shifts
                    .get(&(k, atom.clone()))
                    .is_some_and(|&new_v| {
                        !new_v.is_finite()
                            || (new_v - *cur_v).abs() > curated_aromatic_atom_tol(atom)
                    })
            });
            if has_conflict {
                continue;
            }

            // (b) Count distinct newly proposed CH/NH/N ring positions on residue k (`newly_assigned_atoms` only;
            //     bare cloned shifts absent from `newly_assigned_atoms` are NEVER treated as solver-corroborated!).
            let mut new_positions: BTreeSet<&'static str> = BTreeSet::new();
            for (rk, atom) in newly_assigned_atoms.keys() {
                if *rk == k
                    && newly_assigned
                        .shifts
                        .get(&(k, atom.clone()))
                        .is_some_and(|v| v.is_finite())
                {
                    if let Some(pos) = aromatic_ring_position_key(res_name, atom) {
                        new_positions.insert(pos);
                    }
                }
            }

            let mut curated_positions: BTreeSet<&'static str> = BTreeSet::new();
            for (atom, _) in &curated_ring {
                if let Some(pos) = aromatic_ring_position_key(res_name, atom) {
                    curated_positions.insert(pos);
                }
            }
            let mut effective_solver_positions = new_positions.clone();
            if let Some(corrob) = solver_corroborated_positions.and_then(|m| m.get(&k)) {
                for &p in corrob {
                    effective_solver_positions.insert(p);
                }
            }

            // (c) Mandatory overlap corroboration for ALL merges onto a residue with curated ring shifts
            //     (regardless of whether `new_positions.len()` is 1, 2, or more!). This blocks chimeric
            //     merges such as an uncorroborated 2-slot `CD+CE` cluster merging onto curated `CZ` on `PHE`,
            //     or a cationic `CD2/CE1` cluster merging onto curated `tau` `ND1` on `HIS`.
            let corroborates_curated = !curated_positions.is_empty()
                && curated_positions
                    .iter()
                    .any(|p| effective_solver_positions.contains(p));
            if !corroborates_curated || effective_solver_positions.len() < 2 {
                continue;
            }

            // (d) For `HIS` with curated imidazole `15N` (`ND1` or `NE2`), verify that adding the
            //     proposed CH/N shifts does not create a Pelton neutral-15N vs cationic-CH contradiction!
            if res_name == "HIS" {
                let curated_has_n = curated_ring.iter().any(|(a, _)| a == "ND1" || a == "NE2");
                if curated_has_n {
                    let cb_ppm = existing_ref
                        .shifts
                        .get(&(k, "CB".to_string()))
                        .copied()
                        .or_else(|| newly_assigned.shifts.get(&(k, "CB".to_string())).copied());
                    let cur_map: BTreeMap<String, f64> = curated_ring.iter().cloned().collect();
                    let mut comb_map = cur_map.clone();
                    for (rk, atom) in newly_assigned_atoms.keys() {
                        if *rk == k && !comb_map.contains_key(atom) {
                            if let Some(&v) = newly_assigned.shifts.get(&(k, atom.clone())) {
                                if v.is_finite() && is_aromatic_ring_atom("HIS", atom) {
                                    comb_map.insert(atom.clone(), v);
                                }
                            }
                        }
                    }
                    let ev_cur = his_evidence_from_ring_map(&cur_map, cb_ppm);
                    let ev_comb = his_evidence_from_ring_map(&comb_map, cb_ppm);
                    let fit_cur = fit_pelton_titration_manifold(&ev_cur);
                    let fit_comb = fit_pelton_titration_manifold(&ev_comb);
                    let cls_cur = classify_his_state(&ev_cur).0;
                    let cls_comb = classify_his_state(&ev_comb).0;
                    if (fit_cur.d_orth_sq_shift <= 5.991 && fit_comb.d_orth_sq_shift > 7.815)
                        || (cls_cur != HisStateClassification::MetalCoordinatedCandidate
                            && cls_comb == HisStateClassification::MetalCoordinatedCandidate)
                    {
                        continue;
                    }
                }
            }

            if new_positions.len() < 2 {
                // Single-slot (or CB-only) completion onto a pre-curated ringadditionally requires
                // covering the residue's complete aromatic ring topology (`{"CD","CE"}` on `TYR`,
                // `{"CD","CE","CZ"}` on `PHE`, `{"CD2","CE1"}` or `>= 3` positions on `HIS`,
                // `>= 5` indole positions on `TRP`).
                let has_complete_ring = match res_name {
                    "TYR" => {
                        effective_solver_positions.contains("CD")
                            && effective_solver_positions.contains("CE")
                    }
                    "PHE" => {
                        effective_solver_positions.contains("CD")
                            && effective_solver_positions.contains("CE")
                            && effective_solver_positions.contains("CZ")
                    }
                    "HIS" => {
                        (effective_solver_positions.contains("CD2")
                            && effective_solver_positions.contains("CE1"))
                            || effective_solver_positions.len() >= 3
                    }
                    "TRP" => effective_solver_positions.len() >= 5,
                    _ => false,
                };
                if !has_complete_ring {
                    continue;
                }
            }
        }

        for key in newly_assigned_atoms.keys().filter(|(rk, _)| *rk == k) {
            if merged.shifts.contains_key(key) {
                continue;
            }
            if let Some(&val) = newly_assigned.shifts.get(key) {
                if val.is_finite() {
                    merged.shifts.insert(key.clone(), val);
                } else {
                    continue;
                }
            } else {
                continue;
            }
            if !merged.ambiguity_codes.contains_key(key) {
                if let Some(&code) = newly_assigned.ambiguity_codes.get(key) {
                    merged.ambiguity_codes.insert(key.clone(), code);
                }
            }
        }
    }

    merged
}

/// Merge newly assigned aromatic chemical shifts into `existing_ref` with explicit overwrite policy.
///
/// When `overwrite_existing == false` (the safe default):
/// 1. Keys already present in `existing_ref.shifts` are **never** overwritten (`shifts` and `ambiguity_codes` preserved).
/// 2. If residue `k` already carries curated aromatic ring shifts in `existing_ref`, any solver ring
///    that conflicts on an overlapping ring atom, fails to overlap and corroborate at least one curated
///    ring position on `k` (e.g. a non-overlapping `CD+CE` cluster onto curated `CZ` on `PHE`, or cationic
///    `CE1/HE1` onto curated $\tau$ `ND1` on `HIS`), or proposes `< 2` corroborated ring positions without
///    full-ring topology is rejected cleanly rather than producing a chimeric mixed ring.
///
/// When `overwrite_existing == true`:
/// - Overwrites only finite shift keys explicitly present in `newly_assigned_atoms` and **never** strips
///   pre-existing curated ring atoms that are absent from a partial solver cluster.
pub fn merge_aromatic_into_shifts_with_options(
    existing_ref: &Reference,
    newly_assigned: &Reference,
    newly_assigned_atoms: &BTreeMap<(usize, String), f64>,
    overwrite_existing: bool,
) -> Reference {
    merge_aromatic_into_shifts_internal(
        existing_ref,
        newly_assigned,
        newly_assigned_atoms,
        None,
        overwrite_existing,
    )
}

/// Merge newly assigned aromatic shifts into `existing_ref` without overwriting any pre-existing shifts
/// (`overwrite_existing = false`) and without mixing partial solver rings onto curated residues.
#[inline]
pub fn merge_aromatic_into_shifts(
    existing_ref: &Reference,
    newly_assigned: &Reference,
    newly_assigned_atoms: &BTreeMap<(usize, String), f64>,
) -> Reference {
    merge_aromatic_into_shifts_with_options(
        existing_ref,
        newly_assigned,
        newly_assigned_atoms,
        false,
    )
}

/// Merge newly assigned aromatic chemical shifts into `existing_ref` while strictly preserving
/// all pre-existing shifts and ambiguity codes (`overwrite_existing = false`).
#[inline]
pub fn merge_aromatic_assignments_preserving_existing(
    existing_ref: &Reference,
    newly_assigned: &Reference,
    newly_assigned_atoms: &BTreeMap<(usize, String), f64>,
) -> Reference {
    merge_aromatic_into_shifts_with_options(
        existing_ref,
        newly_assigned,
        newly_assigned_atoms,
        false,
    )
}

/// Estimate an unlabelled systematic `13CB` referencing offset `δ` (`obs_cb - bb_cb`, in ppm)
/// between 2D Yamazaki `(HB)CB(CGCD)HD` / `(HB)CB(CGCDCE)HE` cross-peaks and `backbone_ref`
/// aromatic `CB` shifts via 1-to-1 injective Gaussian kernel consensus + median refinement.
/// Returns `0.0` unless at least 2 independent 1-to-1 pairs agree on `|δ| >= 0.04 ppm` and
/// strictly improve over `δ = 0.0`.
pub fn estimate_unlabelled_yamazaki_cb_offset(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    aromatic_peaks: &[Peak],
    exp_filter: Option<&str>,
    min_backbone_conf: f64,
) -> f64 {
    let mut bb_cbs: Vec<f64> = Vec::new();
    for (k, r) in backbone_ref.sequence.iter().enumerate() {
        if !matches!(r.as_str(), "PHE" | "TYR" | "TRP" | "HIS") {
            continue;
        }
        let a_conf = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0);
        if a_conf < min_backbone_conf {
            continue;
        }
        if let Some(&cb) = backbone_ref.shifts.get(&(k, "CB".to_string())) {
            if cb.is_finite() {
                bb_cbs.push(cb);
            }
        }
    }
    if bb_cbs.len() < 2 {
        return 0.0;
    }

    let mut obs_cbs: Vec<f64> = Vec::new();
    for p in aromatic_peaks {
        let ok_exp = match exp_filter {
            Some(e) => p.exp == e,
            None => p.exp == "HBCBCGCDHD" || p.exp == "HBCBCGCDCEHE",
        };
        if !ok_exp {
            continue;
        }
        if let Some(cb) = p.c.filter(|v| v.is_finite()) {
            obs_cbs.push(cb);
        }
    }
    if obs_cbs.len() < 2 {
        return 0.0;
    }

    let eval_delta = |delta: f64| -> (f64, Vec<f64>) {
        // Build all candidate (residual_abs, obs_idx, bb_idx, raw_diff) pairs within 0.095 ppm
        let mut pairs: Vec<(f64, usize, usize, f64)> = Vec::new();
        for (oi, &oy) in obs_cbs.iter().enumerate() {
            for (bi, &bc) in bb_cbs.iter().enumerate() {
                let diff = oy - bc;
                let res = (diff - delta).abs();
                if res <= 0.095 {
                    pairs.push((res, oi, bi, diff));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_o = vec![false; obs_cbs.len()];
        let mut used_b = vec![false; bb_cbs.len()];
        let mut score = 0.0;
        let mut matched_diffs = Vec::new();
        for (res, oi, bi, diff) in pairs {
            if used_o[oi] || used_b[bi] {
                continue;
            }
            used_o[oi] = true;
            used_b[bi] = true;
            let z = res / 0.045;
            score += (-0.5 * z * z).exp();
            matched_diffs.push(diff);
        }
        (score, matched_diffs)
    };

    let (score_zero, _) = eval_delta(0.0);
    let mut cands: Vec<f64> = vec![0.0];
    for &oy in &obs_cbs {
        for &bc in &bb_cbs {
            let d = oy - bc;
            if d.abs() <= 0.45 {
                cands.push(d);
            }
        }
    }
    cands.sort_by(|a, b| a.total_cmp(b));
    cands.dedup_by(|a, b| (*a - *b).abs() < 0.002);

    let mut best_delta = 0.0;
    let mut best_score = score_zero;
    let mut best_diffs: Vec<f64> = Vec::new();
    for d in cands {
        let (sc, diffs) = eval_delta(d);
        if sc > best_score {
            best_score = sc;
            best_delta = d;
            best_diffs = diffs;
        }
    }

    if best_diffs.len() < 2 {
        return 0.0;
    }
    best_diffs.sort_by(|a, b| a.total_cmp(b));
    let n = best_diffs.len();
    let refined = if n % 2 == 1 {
        best_diffs[n / 2]
    } else {
        0.5 * (best_diffs[n / 2 - 1] + best_diffs[n / 2])
    };
    let (refined_score, refined_diffs) = eval_delta(refined);
    let final_delta = if refined_score >= best_score - 1e-6 && refined_diffs.len() >= 2 {
        refined
    } else {
        best_delta
    };
    let final_inliers = refined_diffs.len().max(n);
    let min_cap = obs_cbs.len().min(bb_cbs.len());

    if final_inliers >= 2
        && final_inliers * 3 >= min_cap
        && final_delta.abs() >= 0.04
        && refined_score.max(best_score) > score_zero + 0.65
    {
        final_delta
    } else {
        0.0
    }
}

fn align_yamazaki_cb_peaks(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    aromatic_peaks: &[Peak],
    cfg: &AromaticSolverConfig,
) -> Vec<Peak> {
    let d_joint = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        None,
        cfg.min_backbone_conf,
    );
    let d_hd_raw = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        Some("HBCBCGCDHD"),
        cfg.min_backbone_conf,
    );
    let d_he_raw = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        Some("HBCBCGCDCEHE"),
        cfg.min_backbone_conf,
    );
    let d_hd = if d_hd_raw != 0.0 { d_hd_raw } else { d_joint };
    let d_he = if d_he_raw != 0.0 { d_he_raw } else { d_joint };
    if d_hd == 0.0 && d_he == 0.0 {
        return aromatic_peaks.to_vec();
    }
    aromatic_peaks
        .iter()
        .map(|p| {
            let mut q = p.clone();
            if p.exp == "HBCBCGCDHD" && d_hd != 0.0 {
                if let Some(cb) = q.c {
                    q.c = Some(cb - d_hd);
                }
            } else if p.exp == "HBCBCGCDCEHE" && d_he != 0.0 {
                if let Some(cb) = q.c {
                    q.c = Some(cb - d_he);
                }
            }
            q
        })
        .collect()
}

/// Compute pairwise log-likelihood ratio `S(cluster, residue_k)` combining Yamazaki
/// `|CB_yamazaki - CB_backbone(k)|`, optional `HB2/HB3` or `NE1/HE1` anchors, intra-ring TOCSY
/// coherence, and BMRB per-atom `(1H, 13C, 15N)` priors.
#[allow(clippy::too_many_arguments)]
fn score_cluster_for_residue(
    ci: usize,
    c: &AromaticRingCluster,
    all_clusters: &[AromaticRingCluster],
    k: usize,
    res_name: &str,
    backbone_ref: &Reference,
    anchor_conf: f64,
    aromatic_peaks: &[Peak],
    has_link_exps: bool,
    priors: &SidechainPriors,
    cfg: &AromaticSolverConfig,
) -> f64 {
    // Hard cross-family exclusions when unambiguous heteronuclear / topological signatures conflict
    if c.kind.res_name() != res_name {
        return FORBID_SCORE;
    }

    let mut llr = 0.0;

    // 1. Per-atom BMRB (13C, 1H) ring chemical-shift log-likelihood ratios
    for (&c_atom, slot) in &c.ch_slots {
        let p_atom = match (res_name, c_atom) {
            ("PHE" | "TYR" | "TRP", "CD1") => "HD1",
            ("PHE" | "TYR" | "HIS", "CE1") => "HE1",
            ("PHE", "CZ") => "HZ",
            ("HIS", "CD2") => "HD2",
            ("TRP", "CE3") => "HE3",
            ("TRP", "CZ3") => "HZ3",
            ("TRP", "CH2") => "HH2",
            ("TRP", "CZ2") => "HZ2",
            _ => continue,
        };
        if res_name == "HIS" {
            if let Some((st, _)) = c.his_tautomer {
                let (_, _, p_cd2, p_hd2, p_ce1, p_he1, _) = st.shift_priors();
                if c_atom == "CD2" {
                    llr += (gauss_logpdf(slot.c_ppm, p_cd2.0, p_cd2.1)
                        - priors.background_c13(CarbonRegion::Aromatic, slot.c_ppm))
                    .max(-8.0);
                    llr += (gauss_logpdf(slot.h_ppm, p_hd2.0, p_hd2.1)
                        - priors.background_h1(CarbonRegion::Aromatic, slot.h_ppm))
                    .max(-8.0);
                } else {
                    llr += (gauss_logpdf(slot.c_ppm, p_ce1.0, p_ce1.1)
                        - priors.background_c13(CarbonRegion::Aromatic, slot.c_ppm))
                    .max(-8.0);
                    llr += (gauss_logpdf(slot.h_ppm, p_he1.0, p_he1.1)
                        - priors.background_h1(CarbonRegion::Aromatic, slot.h_ppm))
                    .max(-8.0);
                }
            } else {
                llr += priors.llr_c13("HIS", c_atom, CarbonRegion::Aromatic, slot.c_ppm);
                llr += priors.llr_h1("HIS", p_atom, CarbonRegion::Aromatic, slot.h_ppm);
            }
        } else {
            llr += priors.llr_c13(res_name, c_atom, CarbonRegion::Aromatic, slot.c_ppm);
            llr += priors.llr_h1(res_name, p_atom, CarbonRegion::Aromatic, slot.h_ppm);
        }
    }

    // 2. Yamazaki |CB_yamazaki - CB_backbone(k)| Gaussian log-likelihood ratio
    let bb_cb = (anchor_conf >= cfg.min_backbone_conf)
        .then(|| backbone_ref.shifts.get(&(k, "CB".to_string())).copied())
        .flatten();

    if let Some(target_cb) = bb_cb {
        let same_aa_cbs: Vec<f64> = backbone_ref
            .sequence
            .iter()
            .enumerate()
            .filter_map(|(j, r)| {
                (r == res_name)
                    .then(|| backbone_ref.shifts.get(&(j, "CB".to_string())).copied())
                    .flatten()
                    .filter(|v| v.is_finite())
            })
            .collect();
        let cb_plausible_for_aa = |cb: f64| -> bool {
            same_aa_cbs.is_empty()
                || same_aa_cbs
                    .iter()
                    .any(|&scb| (scb - cb).abs() <= cfg.tol_cb)
        };

        let hd_h = c
            .ch_slots
            .get("CD1")
            .or_else(|| c.ch_slots.get("CD2"))
            .map(|s| s.h_ppm);
        let he_h = match res_name {
            "TRP" => c.ch_slots.get("CE3").map(|s| s.h_ppm),
            _ => c.ch_slots.get("CE1").map(|s| s.h_ppm),
        };
        let owned_by_closer_cluster = |exp_name: &str, h0: f64, p_h: f64, p_cb: f64| -> bool {
            let dh0 = (p_h - h0).abs();
            all_clusters.iter().enumerate().any(|(oj, oc)| {
                if oj == ci {
                    return false;
                }
                let oh_opt = if exp_name == "HBCBCGCDHD" {
                    oc.ch_slots
                        .get("CD1")
                        .or_else(|| oc.ch_slots.get("CD2"))
                        .map(|s| s.h_ppm)
                } else if oc.kind == AromaticRingKind::Trp {
                    oc.ch_slots.get("CE3").map(|s| s.h_ppm)
                } else {
                    oc.ch_slots.get("CE1").map(|s| s.h_ppm)
                };
                let Some(oh) = oh_opt else { return false };
                let claims_cb = oc
                    .cb_hd
                    .is_some_and(|v| (v - p_cb).abs() <= cfg.tol_cb * 0.85)
                    || oc
                        .cb_he
                        .is_some_and(|v| (v - p_cb).abs() <= cfg.tol_cb * 0.85);
                claims_cb && (p_h - oh).abs() + 0.004 < dh0
            })
        };
        let best_yamazaki_for_h = |exp_name: &str, h_opt: Option<f64>, fallback: Option<f64>| {
            let mut best = fallback.filter(|&fb| cb_plausible_for_aa(fb));
            if let Some(h0) = h_opt {
                for p in aromatic_peaks {
                    if p.exp == exp_name && (p.h - h0).abs() <= cfg.tol_h * 1.20 {
                        if let Some(cb) = p.c {
                            if !cb_plausible_for_aa(cb)
                                || owned_by_closer_cluster(exp_name, h0, p.h, cb)
                            {
                                continue;
                            }
                            if best
                                .is_none_or(|cur| (cb - target_cb).abs() < (cur - target_cb).abs())
                            {
                                best = Some(cb);
                            }
                        }
                    }
                }
            }
            best
        };
        // Only search alternative 1H-overlapped Yamazaki peaks when the cluster does NOT already
        // possess a self-consistent two-channel (cb_hd, cb_he) Yamazaki pair plausible for this AA!
        let corroborated_pair = matches!(
            (c.cb_hd, c.cb_he),
            (Some(a), Some(b)) if (a - b).abs() <= cfg.tol_cb * 0.85 && cb_plausible_for_aa(a)
        );
        let (eff_cb_hd, eff_cb_he) = if corroborated_pair {
            (c.cb_hd, c.cb_he)
        } else {
            (
                best_yamazaki_for_h("HBCBCGCDHD", hd_h, c.cb_hd),
                best_yamazaki_for_h("HBCBCGCDCEHE", he_h, c.cb_he),
            )
        };

        let score_one_cb = |obs_cb: f64, base_w: f64, miss_pen: f64| -> f64 {
            let dcb = (obs_cb - target_cb).abs();
            if dcb <= cfg.tol_cb {
                let z = dcb / cfg.sigma_cb;
                base_w - 0.5 * z * z
            } else {
                miss_pen - 4.0 * ((dcb - cfg.tol_cb).min(3.0))
            }
        };
        match (eff_cb_hd, eff_cb_he) {
            (Some(c1), Some(c2)) => {
                if (c1 - c2).abs() <= cfg.tol_cb * 0.85 {
                    // Both HD and HE Yamazaki channels corroborate the same CB -> strong joint constraint
                    llr += score_one_cb(c1, 10.5, -14.0) + score_one_cb(c2, 9.5, -14.0);
                } else {
                    // Discrepant HD vs HE CBs indicates one channel picked up an overlapped 1H peak
                    let s1 = score_one_cb(c1, 9.5, -16.0);
                    let s2 = score_one_cb(c2, 8.5, -16.0);
                    llr += s1.max(s2) - 1.5;
                }
            }
            (Some(c1), None) => {
                llr += score_one_cb(c1, 11.5, -16.0);
            }
            (None, Some(c2)) => {
                llr += score_one_cb(c2, 10.5, -16.0);
            }
            (None, None) => {
                llr -= 2.0;
                // Reject bare 1-peak noise blips lacking TOCSY, Yamazaki CB, and 15N links
                if has_link_exps
                    && c.ch_slots.len() <= 1
                    && c.nd1_ppm.is_none()
                    && c.ne2_ppm.is_none()
                    && c.trp_indole.is_none()
                    && c.internal_coherence <= 0.1
                {
                    llr -= 6.5;
                }
            }
        }
    } else {
        if !backbone_ref.shifts.is_empty() {
            let has_any_bb = ["H", "N", "CA", "CB", "C"]
                .iter()
                .any(|a| backbone_ref.shifts.contains_key(&(k, (*a).to_string())));
            if !has_any_bb {
                // Unassigned backbone residue (e.g. disordered terminal His6 purification tag)
                llr -= 4.0;
            }
        }
        if let Some(obs_cb) = c.cb_hd.or(c.cb_he) {
            // Missing backbone CB(k): score Yamazaki CB against amino-acid BMRB CB prior!
            llr += priors.llr_c13(res_name, "CB", CarbonRegion::AliphaticCaConnected, obs_cb);
            if res_name == "HIS" {
                if let Some((st, _)) = c.his_tautomer {
                    let p_cb = st.shift_priors().6;
                    llr += 0.5 * gauss_logpdf(obs_cb, p_cb.0, p_cb.1);
                }
            }
        } else {
            llr -= 2.0;
        }
    }

    // 3. Optional TRP NE1/HE1 or aliphatic HB2/HB3 NOE tie-breakers when present on backbone_ref
    if res_name == "TRP" {
        if let (Some((c_ne1, c_he1, _)), Some(&b_ne1), Some(&b_he1)) = (
            c.trp_indole,
            backbone_ref.shifts.get(&(k, "NE1".to_string())),
            backbone_ref.shifts.get(&(k, "HE1".to_string())),
        ) {
            let zn = (c_ne1 - b_ne1) / (cfg.tol_n * 0.45);
            let zh = (c_he1 - b_he1) / (cfg.tol_h * 0.6);
            llr += 9.0 - 0.5 * (zn * zn + zh * zh).min(30.0);
        }
    }

    let hb2 = backbone_ref.shifts.get(&(k, "HB2".to_string())).copied();
    let hb3 = backbone_ref.shifts.get(&(k, "HB3".to_string())).copied();
    let hd_slot = c.ch_slots.get("CD1").or_else(|| c.ch_slots.get("CD2"));
    if let (Some(hd_s), true) = (hd_slot, hb2.is_some() || hb3.is_some()) {
        for p in aromatic_peaks {
            if (p.exp == "NOESY13C" || p.exp == "AROM_HB_NOE")
                && (p.h - hd_s.h_ppm).abs() <= cfg.tol_h * 1.25
            {
                if let Some(h2) = p.h2 {
                    for hb_v in [hb2, hb3].into_iter().flatten() {
                        let d = (h2 - hb_v).abs();
                        if d <= cfg.tol_h * 1.5 {
                            let z = d / (cfg.tol_h * 0.6);
                            llr += 4.5 - 0.5 * z * z;
                        }
                    }
                }
            }
        }
    }

    // 4. Modest completeness / intra-ring coherence bonus
    llr += 0.25 * c.internal_coherence.clamp(-4.0, 12.0);
    llr
}

#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a >= b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

/// Exact bitmask-DP Boltzmann permanent marginal `P(π(target_r_local) = target_c_local)` over all
/// valid 1-to-1 matchings + per-residue null slots within an amino-acid sub-block (`m <= 11`).
fn exact_block_boltzmann_marginal(
    rows: &[usize],
    cols: &[usize],
    n_clu: usize,
    weights: &[Vec<f64>],
    temp: f64,
    target_r_local: usize,
    target_c_local: usize,
) -> Option<f64> {
    let k_len = rows.len();
    let m_len = cols.len();
    if k_len == 0 || m_len == 0 || k_len > 12 || m_len > 11 {
        return None;
    }
    let inv_t = 1.0 / temp.max(0.15);
    let n_masks = 1usize << m_len;

    let run_dp = |forced: Option<(usize, usize)>| -> f64 {
        let mut dp = vec![f64::NEG_INFINITY; n_masks];
        dp[0] = 0.0;
        for (ri_loc, &r_glob) in rows.iter().enumerate() {
            let mut next = vec![f64::NEG_INFINITY; n_masks];
            if let Some((f_r, f_c)) = forced {
                if ri_loc == f_r {
                    let c_glob = cols[f_c];
                    let w = weights[r_glob][c_glob];
                    if w > FORBID_SCORE * 0.5 {
                        let bit = 1usize << f_c;
                        let wt = w * inv_t;
                        for mask in 0..n_masks {
                            if (mask & bit) == 0 && dp[mask] > f64::NEG_INFINITY {
                                next[mask | bit] = log_add_exp(next[mask | bit], dp[mask] + wt);
                            }
                        }
                    }
                    dp = next;
                    continue;
                }
            }

            let w_null = weights[r_glob][n_clu + r_glob] * inv_t;
            for mask in 0..n_masks {
                let base = dp[mask];
                if base == f64::NEG_INFINITY {
                    continue;
                }
                next[mask] = log_add_exp(next[mask], base + w_null);
                for (ci_loc, &c_glob) in cols.iter().enumerate() {
                    let bit = 1usize << ci_loc;
                    if (mask & bit) != 0 {
                        continue;
                    }
                    if let Some((_, f_c)) = forced {
                        if ci_loc == f_c {
                            continue;
                        }
                    }
                    let w = weights[r_glob][c_glob];
                    if w > FORBID_SCORE * 0.5 {
                        next[mask | bit] = log_add_exp(next[mask | bit], base + w * inv_t);
                    }
                }
            }
            dp = next;
        }
        dp.into_iter().fold(f64::NEG_INFINITY, log_add_exp)
    };

    let log_z_total = run_dp(None);
    if !log_z_total.is_finite() {
        return None;
    }
    let log_z_num = run_dp(Some((target_r_local, target_c_local)));
    if !log_z_num.is_finite() {
        return Some(0.0);
    }
    Some((log_z_num - log_z_total).exp().clamp(0.0, 1.0))
}

/// Compute calibrated multi-candidate Boltzmann posterior confidences in `[0.0, 1.0]` combining
/// exact amino-acid block Boltzmann permanent (`m <= 11`) and global swap/free/null/LOO softmax
/// so any symmetric `K`-way tie yields `<= 1 / K` while unique high-margin matches achieve `>= 0.95`.
fn compute_calibrated_ring_confidences(
    arom_positions: &[usize],
    backbone_ref: &Reference,
    clusters: &[AromaticRingCluster],
    weights: &[Vec<f64>],
    win_cols: &[Option<usize>],
    margins: &[f64],
    temp: f64,
) -> Vec<f64> {
    let n_res = arom_positions.len();
    let n_clu = clusters.len();
    let t = temp.max(0.15);
    let mut confs = vec![0.0; n_res];

    for kind in [
        AromaticRingKind::Phe,
        AromaticRingKind::Tyr,
        AromaticRingKind::Trp,
        AromaticRingKind::His,
    ] {
        let aa = kind.res_name();
        let rows_aa: Vec<usize> = (0..n_res)
            .filter(|&ri| backbone_ref.sequence[arom_positions[ri]] == aa)
            .collect();
        if rows_aa.is_empty() {
            continue;
        }
        let cols_aa: Vec<usize> = (0..n_clu).filter(|&ci| clusters[ci].kind == kind).collect();

        let mut matched_clusters: BTreeSet<usize> = BTreeSet::new();
        for &ri in &rows_aa {
            if let Some(ci) = win_cols[ri].filter(|&c| c < n_clu) {
                matched_clusters.insert(ci);
            }
        }

        for (r_loc, &ri) in rows_aa.iter().enumerate() {
            let Some(ci) = win_cols[ri].filter(|&c| c < n_clu) else {
                continue;
            };
            let w_win = weights[ri][ci];
            let w_null = weights[ri][n_clu + ri];

            // Multi-competitor partition across null + all 2-swaps + all free clusters + global LOO
            let mut z_denom = 1.0 + (-(w_win - w_null).max(0.0) / t).exp();

            for &rj in &rows_aa {
                if rj == ri {
                    continue;
                }
                if let Some(cj) = win_cols[rj].filter(|&c| c < n_clu) {
                    let delta_swap =
                        ((w_win + weights[rj][cj]) - (weights[ri][cj] + weights[rj][ci])).max(0.0);
                    z_denom += (-delta_swap / t).exp();
                } else {
                    let w_rj_null = weights[rj][n_clu + rj];
                    let delta_shift = ((w_win + w_rj_null) - (w_null + weights[rj][ci])).max(0.0);
                    z_denom += (-delta_shift / t).exp();
                }
            }

            for &c_alt in &cols_aa {
                if c_alt != ci && !matched_clusters.contains(&c_alt) {
                    let delta_free = (w_win - weights[ri][c_alt]).max(0.0);
                    z_denom += (-delta_free / t).exp();
                }
            }

            let z_loo = 1.0 + (-margins[ri].max(0.0) / t).exp();
            let p_multi = (1.0 / z_denom.max(z_loo)).clamp(0.0, 1.0);

            let p_perm = cols_aa.iter().position(|&c| c == ci).and_then(|c_loc| {
                exact_block_boltzmann_marginal(&rows_aa, &cols_aa, n_clu, weights, t, r_loc, c_loc)
            });

            confs[ri] = p_perm.map_or(p_multi, |pp| pp.min(p_multi));
        }
    }

    confs
}

fn default_aromatic_atom_ambiguity_code(res_name: &str, atom: &str) -> u8 {
    match (res_name, atom) {
        ("PHE" | "TYR", "CD1" | "CD2" | "HD1" | "HD2" | "CE1" | "CE2" | "HE1" | "HE2") => 3,
        _ => 1,
    }
}

/// Assign aromatic ring spin systems (`PHE`, `TYR`, `TRP`, `HIS`) to sequence positions via
/// exact maximum-weight bipartite matching (Kuhn–Munkres) with multi-candidate Boltzmann
/// marginal confidence calibration (`[0.0, 1.0]`), unlabelled inter-experiment `CB` offset
/// alignment, and strict preservation of pre-existing user shifts on untouched atoms.
pub fn assign_aromatic_rings(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    aromatic_peaks: &[Peak],
    priors: &SidechainPriors,
    cfg: &AromaticSolverConfig,
) -> AromaticAssignResult {
    let aligned_peaks = align_yamazaki_cb_peaks(backbone_ref, backbone_conf, aromatic_peaks, cfg);
    let clusters = cluster_aromatic_rings(&aligned_peaks, Some(backbone_ref), priors, cfg);
    let arom_positions: Vec<usize> = backbone_ref
        .sequence
        .iter()
        .enumerate()
        .filter_map(|(k, r)| matches!(r.as_str(), "PHE" | "TYR" | "TRP" | "HIS").then_some(k))
        .collect();

    let mut newly_assigned_ref = backbone_ref.clone();
    let mut atom_confidence: BTreeMap<(usize, String), f64> = BTreeMap::new();

    let n_res = arom_positions.len();
    let n_clu = clusters.len();
    if n_res == 0 {
        return AromaticAssignResult {
            reference: newly_assigned_ref,
            atom_confidence,
            rings: Vec::new(),
            clusters,
        };
    }

    let has_link_exps = aligned_peaks.iter().any(|p| {
        matches!(
            p.exp.as_str(),
            "AROM_HCCH_TOCSY"
                | "HCCHTOCSY"
                | "HBCBCGCDHD"
                | "HBCBCGCDCEHE"
                | "HIS_LR_N15HMQC"
                | "TRP_IND_N15HSQC"
        )
    });

    // Build n_res x (n_clu + n_res) bipartite weight matrix:
    // Columns 0..n_clu are observed ring clusters; columns n_clu..(n_clu + n_res) are per-residue
    // null/unassigned slots with baseline score -2.5 so poor/spurious clusters are cleanly rejected.
    let n_cols = n_clu + n_res;
    let null_score = -2.5;
    let mut weights = vec![vec![FORBID_SCORE; n_cols]; n_res];

    for (ri, &k) in arom_positions.iter().enumerate() {
        let res_name = backbone_ref.sequence[k].as_str();
        let a_conf = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0);
        for (ci, clu) in clusters.iter().enumerate() {
            weights[ri][ci] = score_cluster_for_residue(
                ci,
                clu,
                &clusters,
                k,
                res_name,
                backbone_ref,
                a_conf,
                &aligned_peaks,
                has_link_exps,
                priors,
                cfg,
            );
        }
        // Dedicated null option for residue ri
        weights[ri][n_clu + ri] = null_score;
    }

    // Pre-curated ring locks & conflict guards (P1-10):
    // 1. Any cluster conflicting with curated ring shifts on residue `ri` is forbidden on `ri`.
    // 2. Any cluster whose observed `(C, H)` / `(N, H)` shifts uniquely corroborate a pre-curated
    //    residue `ri` (`>= 2` curated ring scalars matched within tolerance, zero conflicts, and
    //    compatible `CB` when present) is locked exclusively to `ri` (`weights[rj][ci] = FORBID_SCORE`
    //    for all `rj != ri`), preventing duplicate assignment of a curated ring onto a second residue!
    let curated_per_ri: Vec<Vec<(String, f64)>> = arom_positions
        .iter()
        .map(|&k| residue_curated_ring_atoms(backbone_ref, k, &backbone_ref.sequence[k]))
        .collect();

    for (ci, clu) in clusters.iter().enumerate() {
        let mut best_lock_ri: Option<(usize, usize, f64)> = None;
        let mut quarantine_partial_curated = false;
        for (ri, &k) in arom_positions.iter().enumerate() {
            let res_name = backbone_ref.sequence[k].as_str();
            if clu.kind.res_name() != res_name {
                continue;
            }
            let cur_atoms = &curated_per_ri[ri];
            if cur_atoms.is_empty() {
                continue;
            }
            let clu_atoms = cluster_atom_shifts_for_residue(clu, res_name);
            let mut n_matched_positions: BTreeSet<&'static str> = BTreeSet::new();
            let mut n_conflict_positions: BTreeSet<&'static str> = BTreeSet::new();
            let mut n_matched_scalars = 0usize;
            let mut sum_norm_err = 0.0_f64;
            let mut conflict = false;

            for &(c_atom, c_val) in &clu_atoms {
                if let Some(&(_, cur_val)) = cur_atoms.iter().find(|(a, _)| a == c_atom) {
                    let tol = curated_aromatic_atom_tol(c_atom);
                    let diff = (c_val - cur_val).abs();
                    if diff > tol {
                        conflict = true;
                        if let Some(pos) = aromatic_ring_position_key(res_name, c_atom) {
                            n_conflict_positions.insert(pos);
                        }
                    } else {
                        n_matched_scalars += 1;
                        sum_norm_err += diff / tol;
                        if let Some(pos) = aromatic_ring_position_key(res_name, c_atom) {
                            n_matched_positions.insert(pos);
                        }
                    }
                }
            }

            // Also verify Yamazaki CB agreement if both backbone CB and cluster CB exist
            let cb_conflict = if let (Some(&bb_cb), Some(clu_cb)) = (
                backbone_ref.shifts.get(&(k, "CB".to_string())),
                clu.cb_hd.or(clu.cb_he),
            ) {
                (clu_cb - bb_cb).abs() > 0.65
            } else {
                false
            };

            if conflict {
                weights[ri][ci] = FORBID_SCORE;
                // If >= 2 ring positions (e.g. CD + CE) and CB match curated residue `ri` while
                // only 1 user-adjusted slot sits outside tolerance, quarantine `ci` so `ri`'s
                // ring cluster cannot leak onto a second uncurated residue!
                if !cb_conflict
                    && n_matched_positions.len() >= 2
                    && n_matched_positions.len() > n_conflict_positions.len()
                {
                    quarantine_partial_curated = true;
                }
                continue;
            }

            if !cb_conflict && (!n_matched_positions.is_empty() && n_matched_scalars >= 2) {
                let mean_err = sum_norm_err / (n_matched_scalars as f64);
                let better = match best_lock_ri {
                    None => true,
                    Some((_, prev_n, prev_err)) => {
                        n_matched_scalars > prev_n
                            || (n_matched_scalars == prev_n && mean_err < prev_err)
                    }
                };
                if better {
                    best_lock_ri = Some((ri, n_matched_scalars, mean_err));
                }
            } else if n_matched_positions.is_empty() {
                // Cluster has zero overlap with curated ring atoms on `ri`: forbid uncorroborated
                // clusters (even 2-slot `CD+CE` on curated `CZ` `PHE` or cationic `CD2/CE1` on
                // curated `tau` `ND1` `HIS`) from attaching to a curated residue!
                weights[ri][ci] = FORBID_SCORE;
            }
        }

        if let Some((lock_ri, _, _)) = best_lock_ri {
            for (rj, row) in weights.iter_mut().enumerate() {
                if rj != lock_ri {
                    row[ci] = FORBID_SCORE;
                }
            }
            weights[lock_ri][ci] = weights[lock_ri][ci].max(null_score + 12.0) + 25.0;
        } else if quarantine_partial_curated {
            for row in &mut weights {
                row[ci] = FORBID_SCORE;
            }
        }
    }

    let (win_cols, best_total) = hungarian_max_weight(&weights);
    let margins = hungarian_margins(&weights, &win_cols, best_total);
    let raw_bip_confs = compute_calibrated_ring_confidences(
        &arom_positions,
        backbone_ref,
        &clusters,
        &weights,
        &win_cols,
        &margins,
        cfg.calibration_temp,
    );

    let mut rings = Vec::with_capacity(n_res);
    let mut clu_positions_by_k: BTreeMap<usize, BTreeSet<&'static str>> = BTreeMap::new();
    for (ri, &k) in arom_positions.iter().enumerate() {
        let res_name = backbone_ref.sequence[k].clone();
        let seq_id = backbone_ref
            .seq_ids
            .get(k)
            .copied()
            .unwrap_or((k + 1) as i32);
        let a_conf = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        let chosen_clu = win_cols[ri].filter(|&c| c < n_clu);
        let Some(ci) = chosen_clu else {
            rings.push(AssignedAromaticRing {
                residue_idx: k,
                seq_id,
                res_name,
                cluster_idx: None,
                ring_confidence: 0.0,
                score: null_score,
                margin: 0.0,
                his_tautomer: None,
                his_state: None,
            });
            continue;
        };

        let mrg = margins[ri];
        let win_sc = weights[ri][ci];
        let bip_conf = (raw_bip_confs[ri] * a_conf).clamp(0.0, 1.0);
        let clu = &clusters[ci];

        // Determine how many distinct aromatic ring positions this cluster actually observed
        let clu_atoms_preview = cluster_atom_shifts_for_residue(clu, &res_name);
        let mut clu_positions: BTreeSet<&'static str> = BTreeSet::new();
        for &(ca, _) in &clu_atoms_preview {
            if let Some(pos) = aromatic_ring_position_key(&res_name, ca) {
                clu_positions.insert(pos);
            }
        }
        let clu_conflicts_curated = curated_per_ri[ri].iter().any(|(cur_a, cur_v)| {
            clu_atoms_preview
                .iter()
                .find(|&&(ca, _)| ca == cur_a.as_str())
                .is_some_and(|&(_, cv)| (cv - *cur_v).abs() > curated_aromatic_atom_tol(cur_a))
        });
        let clu_corroborates_curated = curated_per_ri[ri].is_empty()
            || curated_per_ri[ri]
                .iter()
                .filter_map(|(cur_a, _)| aromatic_ring_position_key(&res_name, cur_a))
                .any(|p| clu_positions.contains(p));
        let allow_ring_atom_writes = !clu_conflicts_curated
            && clu_corroborates_curated
            && (curated_per_ri[ri].is_empty() || clu_positions.len() >= 2);
        if !clu_conflicts_curated && clu_corroborates_curated {
            clu_positions_by_k.insert(k, clu_positions);
        }

        let mut record_new_atom = |atom_name: &str, val: f64, conf: f64| {
            let key = (k, atom_name.to_string());
            // Never overwrite or claim confidence on shifts already curated in `backbone_ref`!
            if backbone_ref.shifts.contains_key(&key) {
                return;
            }
            if !allow_ring_atom_writes && is_aromatic_ring_atom(&res_name, atom_name) {
                return;
            }
            newly_assigned_ref.shifts.insert(key.clone(), val);
            newly_assigned_ref.ambiguity_codes.insert(
                key.clone(),
                default_aromatic_atom_ambiguity_code(&res_name, atom_name),
            );
            atom_confidence.insert(key, conf);
        };

        // Populate newly assigned IUPAC shifts & calibrated per-atom confidences
        for (&c_atom, slot) in &clu.ch_slots {
            let at_conf = (bip_conf * slot.intra_conf).clamp(0.0, 1.0);
            match (res_name.as_str(), c_atom) {
                ("PHE" | "TYR", "CD1") => {
                    for &(ca, pa) in &[("CD1", "HD1"), ("CD2", "HD2")] {
                        record_new_atom(ca, slot.c_ppm, at_conf);
                        record_new_atom(pa, slot.h_ppm, at_conf);
                    }
                }
                ("PHE" | "TYR", "CE1") => {
                    for &(ca, pa) in &[("CE1", "HE1"), ("CE2", "HE2")] {
                        record_new_atom(ca, slot.c_ppm, at_conf);
                        record_new_atom(pa, slot.h_ppm, at_conf);
                    }
                }
                ("PHE", "CZ") => {
                    record_new_atom("CZ", slot.c_ppm, at_conf);
                    record_new_atom("HZ", slot.h_ppm, at_conf);
                }
                ("TRP", ca) => {
                    let pa = match ca {
                        "CD1" => "HD1",
                        "CE3" => "HE3",
                        "CZ3" => "HZ3",
                        "CH2" => "HH2",
                        "CZ2" => "HZ2",
                        _ => continue,
                    };
                    record_new_atom(ca, slot.c_ppm, at_conf);
                    record_new_atom(pa, slot.h_ppm, at_conf);
                }
                ("HIS", ca) => {
                    let pa = if ca == "CD2" { "HD2" } else { "HE1" };
                    record_new_atom(ca, slot.c_ppm, at_conf);
                    record_new_atom(pa, slot.h_ppm, at_conf);
                }
                _ => {}
            }
        }

        if res_name == "TRP" {
            if let Some((ne1, he1, icf)) = clu.trp_indole {
                let c_ind = (bip_conf * icf).clamp(0.0, 1.0);
                record_new_atom("NE1", ne1, c_ind);
                record_new_atom("HE1", he1, c_ind);
            }
        }

        if res_name == "HIS" {
            if let Some((nd1, ncf)) = clu.nd1_ppm {
                let c_n = (bip_conf * ncf).clamp(0.0, 1.0);
                record_new_atom("ND1", nd1, c_n);
            }
            if let Some((ne2, ncf)) = clu.ne2_ppm {
                let c_n = (bip_conf * ncf).clamp(0.0, 1.0);
                record_new_atom("NE2", ne2, c_n);
            }
        }

        // Populate CB ONLY if missing on backbone_ref
        if !backbone_ref.shifts.contains_key(&(k, "CB".to_string())) {
            if let Some(cb_y) = clu.cb_hd.or(clu.cb_he) {
                record_new_atom("CB", cb_y, bip_conf);
            }
        }

        rings.push(AssignedAromaticRing {
            residue_idx: k,
            seq_id,
            res_name,
            cluster_idx: Some(ci),
            ring_confidence: bip_conf,
            score: win_sc,
            margin: mrg,
            his_tautomer: clu.his_tautomer,
            his_state: clu.his_state,
        });
    }

    let merged_ref = merge_aromatic_into_shifts_internal(
        backbone_ref,
        &newly_assigned_ref,
        &atom_confidence,
        Some(&clu_positions_by_k),
        false,
    );
    // Strict invariant: `atom_confidence` is always a key-subset of `reference.shifts` and never
    // contains pre-existing curated shifts from `backbone_ref.shifts` or rejected merge entries.
    atom_confidence.retain(|key, _| {
        merged_ref.shifts.contains_key(key) && !backbone_ref.shifts.contains_key(key)
    });

    AromaticAssignResult {
        reference: merged_ref,
        atom_confidence,
        rings,
        clusters,
    }
}

// ============================================================================
// 7. Quantitative Aromatic Evaluation Metrics (`1H`, `13C`, `15N` & `HisTautomer`)
// ============================================================================

/// Quantitative accuracy, recall, high-confidence (`>= 0.80`) precision, and `HisTautomer`
/// metrics across aromatic `(1H, 13C, 15N)` ring resonances.
#[derive(Clone, Debug, Default)]
pub struct AromaticMetrics {
    pub assignable_atoms: usize,
    pub assigned_atoms: usize,
    pub correct_atoms: usize,
    pub hc_assigned_atoms: usize,
    pub hc_correct_atoms: usize,
    pub his_total: usize,
    pub his_classified: usize,
    pub his_correct: usize,
}

impl AromaticMetrics {
    pub fn accumulate(&mut self, other: &Self) {
        self.assignable_atoms += other.assignable_atoms;
        self.assigned_atoms += other.assigned_atoms;
        self.correct_atoms += other.correct_atoms;
        self.hc_assigned_atoms += other.hc_assigned_atoms;
        self.hc_correct_atoms += other.hc_correct_atoms;
        self.his_total += other.his_total;
        self.his_classified += other.his_classified;
        self.his_correct += other.his_correct;
    }

    /// Fraction of assigned aromatic `(1H, 13C/15N)` resonances within tolerance of ground truth.
    pub fn accuracy(&self) -> f64 {
        if self.assigned_atoms > 0 {
            (self.correct_atoms as f64) / (self.assigned_atoms as f64)
        } else {
            0.0
        }
    }

    /// Overall completeness/recall (`correct_atoms / assignable_atoms`).
    pub fn recall(&self) -> f64 {
        if self.assignable_atoms > 0 {
            (self.correct_atoms as f64) / (self.assignable_atoms as f64)
        } else {
            0.0
        }
    }

    /// Precision on aromatic resonances committed with confidence `>= 0.80`.
    pub fn high_confidence_precision(&self) -> f64 {
        if self.hc_assigned_atoms > 0 {
            (self.hc_correct_atoms as f64) / (self.hc_assigned_atoms as f64)
        } else {
            0.0
        }
    }

    /// Histidine tautomeric state classification accuracy (`Ne2H` vs `Nd1H` vs `DoublyProtonated`).
    pub fn his_tautomer_accuracy(&self) -> f64 {
        if self.his_classified > 0 {
            (self.his_correct as f64) / (self.his_classified as f64)
        } else {
            1.0
        }
    }
}

/// Evaluate `AromaticAssignResult` against ground-truth `truth` and `truth_his_tautomers`
/// across all non-redundant observable aromatic `(1H, 13C, 15N)` resonances (`CD1/HD1`, `CE1/HE1`,
/// `CZ/HZ` for `PHE`; `CD1/HD1`, `CE1/HE1` for `TYR`; `CD1/HD1`, `NE1/HE1`, `CE3/HE3`, `CZ3/HZ3`,
/// `CH2/HH2`, `CZ2/HZ2` for `TRP`; `CD2/HD2`, `CE1/HE1`, `ND1`, `NE2` for `HIS`).
pub fn evaluate_aromatic_assignment(
    res: &AromaticAssignResult,
    truth: &Reference,
    truth_his_tautomers: &BTreeMap<usize, HisTautomer>,
    tol_h: f64,
    tol_c: f64,
    tol_n: f64,
) -> AromaticMetrics {
    let mut m = AromaticMetrics::default();
    let k_len = truth.sequence.len().min(res.reference.sequence.len());

    // Build residue -> HisTautomer lookup from result
    let mut pred_his: BTreeMap<usize, HisTautomer> = BTreeMap::new();
    for r in &res.rings {
        if let Some((st, _)) = r.his_tautomer {
            pred_his.insert(r.residue_idx, st);
        }
    }

    for k in 0..k_len {
        let aa = truth.sequence[k].as_str();
        // Distinct observable aromatic (atom_name, tolerance) pairs (counting symmetric PHE/TYR pairs once so 2x atoms do not double-count!)
        let eval_atoms: &[(&str, f64)] = match aa {
            "PHE" => &[
                ("CD1", tol_c),
                ("HD1", tol_h),
                ("CE1", tol_c),
                ("HE1", tol_h),
                ("CZ", tol_c),
                ("HZ", tol_h),
            ],
            "TYR" => &[
                ("CD1", tol_c),
                ("HD1", tol_h),
                ("CE1", tol_c),
                ("HE1", tol_h),
            ],
            "TRP" => &[
                ("CD1", tol_c),
                ("HD1", tol_h),
                ("NE1", tol_n),
                ("HE1", tol_h),
                ("CE3", tol_c),
                ("HE3", tol_h),
                ("CZ3", tol_c),
                ("HZ3", tol_h),
                ("CH2", tol_c),
                ("HH2", tol_h),
                ("CZ2", tol_c),
                ("HZ2", tol_h),
            ],
            "HIS" => &[
                ("CD2", tol_c),
                ("HD2", tol_h),
                ("CE1", tol_c),
                ("HE1", tol_h),
                ("ND1", tol_n),
                ("NE2", tol_n),
            ],
            _ => continue,
        };

        let mut handled: BTreeSet<&str> = BTreeSet::new();
        for &(atom, tol) in eval_atoms {
            if !handled.insert(atom) {
                continue;
            }
            // In 1H/13C-only depositions lacking 15N indole shifts (e.g. BMRB 4562), TRP HE1 has no
            // heteronuclear (1H, 15N/13C) correlation peak unless NE1 is also present.
            if aa == "TRP"
                && atom == "HE1"
                && !truth.shifts.contains_key(&(k, "NE1".to_string()))
                && !res.reference.shifts.contains_key(&(k, "HE1".to_string()))
            {
                continue;
            }
            let sym_alt = match (aa, atom) {
                ("PHE" | "TYR", "CD1") => Some("CD2"),
                ("PHE" | "TYR", "HD1") => Some("HD2"),
                ("PHE" | "TYR", "CE1") => Some("CE2"),
                ("PHE" | "TYR", "HE1") => Some("HE2"),
                _ => None,
            };
            let Some(&tv) = truth
                .shifts
                .get(&(k, atom.to_string()))
                .or_else(|| sym_alt.and_then(|alt| truth.shifts.get(&(k, alt.to_string()))))
            else {
                continue;
            };
            m.assignable_atoms += 1;
            if let Some(&pv) = res.reference.shifts.get(&(k, atom.to_string())) {
                m.assigned_atoms += 1;
                let ok = (tv - pv).abs() <= tol;
                if ok {
                    m.correct_atoms += 1;
                }
                let conf = res
                    .atom_confidence
                    .get(&(k, atom.to_string()))
                    .copied()
                    .unwrap_or(0.0);
                if conf >= 0.80 {
                    m.hc_assigned_atoms += 1;
                    if ok {
                        m.hc_correct_atoms += 1;
                    }
                }
            }
        }

        if aa == "HIS" {
            if let Some(&true_st) = truth_his_tautomers.get(&k) {
                m.his_total += 1;
                if let Some(&pred_st) = pred_his.get(&k) {
                    m.his_classified += 1;
                    if pred_st == true_st {
                        m.his_correct += 1;
                    }
                }
            }
        }
    }

    m
}
