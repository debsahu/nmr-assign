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
use crate::topology::{residue_topology, CarbonRegion, ProchiralGroup, SidechainPriors};

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
            Self::AromC13Hsqc | Self::AromHcchTocsy => ((5.4, 9.25), (108.0, 143.5)),
            Self::Hbcbcgcdhd | Self::Hbcbcgcdcehe => ((5.4, 9.25), (24.0, 48.0)),
            Self::HisLrN15Hmqc => ((6.2, 9.25), (155.0, 265.0)),
            Self::TrpIndN15Hsqc => ((9.0, 11.6), (122.0, 136.5)),
        }
    }
}

// ============================================================================
// 1. Histidine Imidazole Tautomeric & Protonation State Model (Pelton et al. 1993)
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

    /// Canonical `(mean_ppm, std_ppm)` priors `(ND1, NE2, CD2, HD2, CE1, HE1, CB)` per Pelton et al. (1993).
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
        match self {
            Self::Ne2H => (
                (249.5, 3.2), // ND1
                (168.5, 3.2), // NE2
                (118.2, 1.8), // CD2 (< 122 ppm)
                (7.02, 0.25), // HD2
                (138.2, 1.5), // CE1
                (7.80, 0.28), // HE1
                (31.2, 2.2),  // CB
            ),
            Self::Nd1H => (
                (169.5, 3.5), // ND1
                (249.5, 3.2), // NE2
                (125.0, 2.0), // CD2 (downfield > 122.5 ppm)
                (7.14, 0.25), // HD2
                (137.8, 1.5), // CE1
                (7.88, 0.28), // HE1
                (30.4, 2.2),  // CB
            ),
            Self::DoublyProtonated => (
                (177.0, 2.8), // ND1 (173..182 ppm)
                (173.5, 2.8), // NE2 (170..180 ppm)
                (119.2, 1.8), // CD2 (< 122 ppm)
                (7.32, 0.25), // HD2
                (136.0, 1.5), // CE1
                (8.56, 0.28), // HE1 (downfield imidazolium shift)
                (29.2, 2.0),  // CB (upfield imidazolium shift)
            ),
        }
    }
}

/// Experimental shifts and cross-peak evidence for classifying a histidine's tautomeric state.
#[derive(Clone, Debug, Default)]
pub struct HisTautomerEvidence {
    pub nd1_ppm: Option<f64>,
    pub ne2_ppm: Option<f64>,
    pub cd2_ppm: Option<f64>,
    pub hd2_ppm: Option<f64>,
    pub ce1_ppm: Option<f64>,
    pub he1_ppm: Option<f64>,
    pub cb_ppm: Option<f64>,
}

#[inline]
fn gauss_logpdf(x: f64, mu: f64, sd: f64) -> f64 {
    let z = (x - mu) / sd;
    -0.5 * z * z - (sd * TAU.sqrt()).ln()
}

/// Classify histidine `HisTautomer` and compute its calibrated posterior probability in `[0.0, 1.0]`
/// from Pelton et al. (1993) `(ND1, NE2)` `15N-HMQC` shifts and corroborating `CD2`/`CE1`/`HE1`/`CB` shifts.
pub fn classify_his_tautomer(ev: &HisTautomerEvidence) -> (HisTautomer, f64) {
    // Mild biological prior (~0.45 Ne2H, ~0.25 Nd1H, ~0.30 DoublyProtonated across pH 5..7.5)
    let log_priors = [0.45_f64.ln(), 0.25_f64.ln(), 0.30_f64.ln()];
    let mut log_post = [0.0; 3];

    for (idx, &state) in HisTautomer::ALL.iter().enumerate() {
        let (p_nd1, p_ne2, p_cd2, p_hd2, p_ce1, p_he1, p_cb) = state.shift_priors();
        let mut lp = log_priors[idx];

        if let Some(v) = ev.nd1_ppm {
            lp += gauss_logpdf(v, p_nd1.0, p_nd1.1);
        }
        if let Some(v) = ev.ne2_ppm {
            lp += gauss_logpdf(v, p_ne2.0, p_ne2.1);
        }
        if let Some(v) = ev.cd2_ppm {
            lp += gauss_logpdf(v, p_cd2.0, p_cd2.1);
        }
        if let Some(v) = ev.hd2_ppm {
            lp += gauss_logpdf(v, p_hd2.0, p_hd2.1);
        }
        if let Some(v) = ev.ce1_ppm {
            lp += gauss_logpdf(v, p_ce1.0, p_ce1.1);
        }
        if let Some(v) = ev.he1_ppm {
            lp += gauss_logpdf(v, p_he1.0, p_he1.1);
        }
        if let Some(v) = ev.cb_ppm {
            lp += 0.35 * gauss_logpdf(v, p_cb.0, p_cb.1);
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
    let mut best_i = 0;
    for i in 1..3 {
        if probs[i] > probs[best_i] {
            best_i = i;
        }
    }
    let post = if sum > 0.0 {
        (probs[best_i] / sum).clamp(0.0, 1.0)
    } else {
        1.0 / 3.0
    };
    (HisTautomer::ALL[best_i], post)
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
    match res {
        "PHE" => {
            if let (Some(&c), Some(&h)) = (
                ref_data.shifts.get(&(k, "CD1".to_string())),
                ref_data.shifts.get(&(k, "HD1".to_string())),
            ) {
                out.push(RingChSpin {
                    carbon: "CD1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
            if let (Some(&c), Some(&h)) = (
                ref_data.shifts.get(&(k, "CE1".to_string())),
                ref_data.shifts.get(&(k, "HE1".to_string())),
            ) {
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
            if let (Some(&c), Some(&h)) = (
                ref_data.shifts.get(&(k, "CD1".to_string())),
                ref_data.shifts.get(&(k, "HD1".to_string())),
            ) {
                out.push(RingChSpin {
                    carbon: "CD1",
                    c_ppm: c,
                    h_ppm: h,
                    equiv_protons: 2,
                });
            }
            if let (Some(&c), Some(&h)) = (
                ref_data.shifts.get(&(k, "CE1".to_string())),
                ref_data.shifts.get(&(k, "HE1".to_string())),
            ) {
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
                let hd_atom = match res.as_str() {
                    "PHE" | "TYR" | "TRP" => "HD1",
                    "HIS" => "HD2",
                    _ => "",
                };
                if let Some(&hd) = ref_data.shifts.get(&(k, hd_atom.to_string())) {
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
                let he_spec: Option<(&str, f64, f64)> = match res.as_str() {
                    "PHE" | "TYR" => Some(("HE1", 1.70, cfg.p_missing_peak * 0.6)),
                    "TRP" => Some(("HE3", 0.78, cfg.p_missing_peak)),
                    "HIS" if cfg.include_weak_his_he1_yamazaki => {
                        Some(("HE1", 0.42, (cfg.p_missing_peak * 2.0).min(0.35)))
                    }
                    _ => None,
                };
                if let Some((he_atom, amp_scale, miss_p)) = he_spec {
                    if let Some(&he) = ref_data.shifts.get(&(k, he_atom.to_string())) {
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

        // 4. Pelton 2D HIS_LR_N15HMQC (HE1 -> ND1 & NE2; HD2 -> NE2 strong & ND1 weak)
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
                    // Unprotonated =N- has slightly larger 2J_NH (~10 Hz) than protonated >NH (~7 Hz)
                    let (a_nd1, a_ne2) = match st {
                        HisTautomer::Ne2H => (1.15, 0.82),
                        HisTautomer::Nd1H => (0.82, 1.15),
                        HisTautomer::DoublyProtonated => (0.90, 0.88),
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
                    // HD2 -> NE2 is strong 2J_NH (~8.5 Hz); HD2 -> ND1 is weak 3J_NH (~2 Hz)
                    if rng.f64() >= cfg.p_missing_peak * 0.3 {
                        let ht_ne2 = 92.0 * (rng.normal(0.0, 0.12)).exp();
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
                    if rng.f64() >= (cfg.p_missing_peak * 2.5).min(0.40) {
                        let ht_nd1 = 24.0 * (rng.normal(0.0, 0.15)).exp();
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
    /// Imidazole `15ND1` and `15NE2` shifts + classified `HisTautomer` (`HIS` clusters).
    pub nd1_ppm: Option<(f64, f64)>,
    pub ne2_ppm: Option<(f64, f64)>,
    pub his_tautomer: Option<(HisTautomer, f64)>,
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
    /// Calibration temperature mapping Hungarian leave-one-out margin `ΔS` to posterior probability.
    pub calibration_temp: f64,
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
        }
    }
}

#[derive(Clone, Debug)]
struct AromHcNode {
    h: f64,
    c: f64,
    height: f64,
    tocsy_h: Vec<f64>,
    cb_hd_cands: Vec<(f64, f64)>, // (cb_ppm, dh_abs)
    cb_he_cands: Vec<(f64, f64)>, // (cb_ppm, dh_abs)
}

impl AromHcNode {
    fn off_diag_tocsy_count(&self, tol_h: f64) -> usize {
        self.tocsy_h
            .iter()
            .filter(|&&th| (th - self.h).abs() > tol_h * 1.05)
            .count()
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
                cb_hd_cands: Vec::new(),
                cb_he_cands: Vec::new(),
            });
        }
    }

    // Attach 3D AROM_HCCH_TOCSY (and HCCHTOCSY aromatic strips), rescuing missing HSQC roots
    let mut orphans: Vec<(f64, f64, Vec<f64>, f64)> = Vec::new();
    for p in peaks {
        let is_tocsy = p.exp == "AROM_HCCH_TOCSY"
            || (p.exp == "HCCHTOCSY" && p.c.is_some_and(|c| (105.0..=146.0).contains(&c)));
        if !is_tocsy {
            continue;
        }
        let (Some(c), Some(h2)) = (p.c, p.h2) else {
            continue;
        };
        let mut attached = false;
        for n in &mut nodes {
            if ((n.h - p.h) / cfg.tol_h).hypot((n.c - c) / cfg.tol_c) <= 1.15 {
                n.tocsy_h.push(h2);
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
                o.3 = o.3.max(p.height.unwrap_or(70.0));
            } else {
                orphans.push((p.h, c, vec![h2], p.height.unwrap_or(70.0)));
            }
        }
    }
    for (oh, oc, h2s, ht) in orphans {
        if h2s.len() >= 2 {
            nodes.push(AromHcNode {
                h: oh,
                c: oc,
                height: ht,
                tocsy_h: h2s,
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

    let w = if directly_bonded { 3.2 } else { 1.8 };
    let mut sc = 2.4 * jaccard;
    if let Some(d) = uv {
        sc += w * (1.0 - 0.35 * (d / tol_h).min(1.2));
    }
    if let Some(d) = vu {
        sc += w * (1.0 - 0.35 * (d / tol_h).min(1.2));
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
                let sc = 8.5 - 0.5 * z_cb * z_cb - 0.6 * z_h + bb_bonus;
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
    let best_hd = hd_node.cb_hd_cands.first().map(|&(cb, dh)| {
        let bb_ok =
            known_cbs.is_empty() || known_cbs.iter().any(|&kcb| (kcb - cb).abs() <= cfg.tol_cb);
        (cb, if bb_ok { 2.5 - dh / cfg.tol_h } else { 0.5 })
    });
    let best_he = he_node.cb_he_cands.first().map(|&(cb, dh)| {
        let bb_ok =
            known_cbs.is_empty() || known_cbs.iter().any(|&kcb| (kcb - cb).abs() <= cfg.tol_cb);
        (cb, if bb_ok { 2.2 - dh / cfg.tol_h } else { 0.4 })
    });

    // If BOTH have unambiguous distinct CBs that disagree strongly, penalize pairing them!
    if let (Some((c1, _)), Some((c2, _))) = (best_hd, best_he) {
        if hd_node.cb_hd_cands.len() == 1
            && he_node.cb_he_cands.len() == 1
            && (c1 - c2).abs() > cfg.tol_cb
        {
            return (Some(c1), Some(c2), -7.5);
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

/// Infer `(ND1, NE2, HisTautomer, posterior)` for a paired `(opt_cd2, opt_ce1)` histidine ring
/// from `HIS_LR_N15HMQC` cross-peaks and `(CD2, CE1, HE1, HD2, CB)` shifts.
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
    (HisTautomer, f64),
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
        // Evaluate all ordered pairs (i_nd1, i_ne2) with i_nd1 != i_ne2
        let mut best_hyp: Option<(usize, usize, HisTautomer, f64, f64)> = None;
        let mut second_lp = f64::NEG_INFINITY;

        for (i_d, &(n_d1, he_d1, hd_d1)) in all_ns.iter().enumerate() {
            for (i_e, &(n_e2, he_e2, hd_e2)) in all_ns.iter().enumerate() {
                if i_d == i_e {
                    continue;
                }
                let ev = HisTautomerEvidence {
                    nd1_ppm: Some(n_d1),
                    ne2_ppm: Some(n_e2),
                    cd2_ppm,
                    hd2_ppm,
                    ce1_ppm,
                    he1_ppm,
                    cb_ppm: cb_hint,
                };
                let (st, post) = classify_his_tautomer(&ev);
                let (p_nd1, p_ne2, p_cd2, _, p_ce1, p_he1, _) = st.shift_priors();
                let mut lp =
                    gauss_logpdf(n_d1, p_nd1.0, p_nd1.1) + gauss_logpdf(n_e2, p_ne2.0, p_ne2.1);
                if let Some(v) = cd2_ppm {
                    lp += gauss_logpdf(v, p_cd2.0, p_cd2.1);
                }
                if let Some(v) = ce1_ppm {
                    lp += gauss_logpdf(v, p_ce1.0, p_ce1.1);
                }
                if let Some(v) = he1_ppm {
                    lp += gauss_logpdf(v, p_he1.0, p_he1.1);
                }
                // Pelton 2J vs 3J rule: HD2 correlates much more strongly to NE2 (2J ~ 8.5 Hz) than ND1 (3J ~ 2 Hz)!
                if hd_e2 > 0.0 && (hd_d1 == 0.0 || hd_e2 >= 1.5 * hd_d1) {
                    lp += 3.6;
                } else if hd_d1 > 1.5 * hd_e2 && hd_d1 > 0.0 {
                    lp -= 3.6;
                }
                if he_d1 > 0.0 {
                    lp += 1.2;
                }
                if he_e2 > 0.0 {
                    lp += 1.2;
                }

                if best_hyp.is_none_or(|(_, _, _, _, blp)| lp > blp) {
                    if let Some((_, _, _, _, blp)) = best_hyp {
                        second_lp = second_lp.max(blp);
                    }
                    best_hyp = Some((i_d, i_e, st, post, lp));
                } else if lp > second_lp {
                    second_lp = lp;
                }
            }
        }

        if let Some((i_d, i_e, st, post, blp)) = best_hyp {
            let n_conf = margin_to_intra_conf((blp - second_lp).max(0.0));
            return (
                Some((all_ns[i_d].0, n_conf)),
                Some((all_ns[i_e].0, n_conf)),
                (st, post),
                shared_link_score,
            );
        }
    } else if all_ns.len() == 1 {
        let (n_only, _, hd_ht) = all_ns[0];
        // Test whether n_only is NE2 (preferred if HD2 sees it strongly) or ND1
        let ev_as_ne2 = HisTautomerEvidence {
            nd1_ppm: None,
            ne2_ppm: Some(n_only),
            cd2_ppm,
            hd2_ppm,
            ce1_ppm,
            he1_ppm,
            cb_ppm: cb_hint,
        };
        let ev_as_nd1 = HisTautomerEvidence {
            nd1_ppm: Some(n_only),
            ne2_ppm: None,
            cd2_ppm,
            hd2_ppm,
            ce1_ppm,
            he1_ppm,
            cb_ppm: cb_hint,
        };
        let (st_e2, p_e2) = classify_his_tautomer(&ev_as_ne2);
        let (st_d1, p_d1) = classify_his_tautomer(&ev_as_nd1);
        if hd_ht > 0.0 || p_e2 >= p_d1 {
            return (None, Some((n_only, 0.82)), (st_e2, p_e2), shared_link_score);
        } else {
            return (Some((n_only, 0.78)), None, (st_d1, p_d1), shared_link_score);
        }
    }

    let ev_fallback = HisTautomerEvidence {
        nd1_ppm: None,
        ne2_ppm: None,
        cd2_ppm,
        hd2_ppm,
        ce1_ppm,
        he1_ppm,
        cb_ppm: cb_hint,
    };
    let (st, post) = classify_his_tautomer(&ev_fallback);
    (None, None, (st, post), shared_link_score)
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
    let (mut n_his_seq, mut n_trp_seq, mut n_tyr_seq, mut n_phe_seq) =
        (0usize, 0usize, 0usize, 0usize);
    if let Some(bb) = backbone_ref {
        for (k, r) in bb.sequence.iter().enumerate() {
            match r.as_str() {
                "HIS" => n_his_seq += 1,
                "TRP" => n_trp_seq += 1,
                "TYR" => n_tyr_seq += 1,
                "PHE" => n_phe_seq += 1,
                _ => continue,
            }
            if let Some(&cb) = bb.shifts.get(&(k, "CB".to_string())) {
                known_arom_cbs.push(cb);
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
    // If PHE/TYR dominate, median is closer to 1.6x; use 25th percentile as robust 1-proton scale
    let h_1x = if heights.len() >= 4 {
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
                || (p.exp == "HSQC" && p.h > 9.3 && (124.0..=136.0).contains(&p.n))
        })
        .cloned()
        .collect();
    let trp_noes: Vec<Peak> = aromatic_peaks
        .iter()
        .filter(|p| (p.exp == "NOESY15N" || p.exp == "TRP_IND_NOE") && p.h > 9.3 && p.h2.is_some())
        .cloned()
        .collect();

    let mut used = vec![false; nodes.len()];
    let mut clusters: Vec<AromaticRingCluster> = Vec::new();

    // ------------------------------------------------------------------------
    // Family 1: Histidine Imidazole Rings (CD2/HD2, CE1/HE1, ND1, NE2, Tautomer)
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
                n.off_diag_tocsy_count(cfg.tol_h) == 0
                    && !is_trp_noe_proton(n.h)
                    && ((hmqc_peaks.is_empty() && (133.5..=144.0).contains(&n.c))
                        || (hmqc_hit && (131.5..=144.5).contains(&n.c)))
            })
            .collect();
        if backbone_ref.is_some() && ce1_cands.len() < n_his_seq {
            let mut by_ht = ce1_cands.clone();
            by_ht.sort_by(|&a, &b| nodes[b].height.total_cmp(&nodes[a].height));
            for u in by_ht {
                if ce1_cands.len() < n_his_seq && nodes[u].height >= 1.32 * h_1x {
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
                n.off_diag_tocsy_count(cfg.tol_h) == 0
                    && !is_trp_noe_proton(n.h)
                    && (112.5..=131.2).contains(&n.c)
                    && (hmqc_hit || hmqc_peaks.is_empty())
            })
            .collect();
        if backbone_ref.is_some() && cd2_cands.len() < n_his_seq {
            let mut by_ht = cd2_cands.clone();
            by_ht.sort_by(|&a, &b| nodes[b].height.total_cmp(&nodes[a].height));
            for u in by_ht {
                if cd2_cands.len() < n_his_seq && nodes[u].height >= 1.32 * h_1x {
                    cd2_cands.push(u);
                }
            }
        }

        if !cd2_cands.is_empty() || !ce1_cands.is_empty() {
            let nr = cd2_cands.len();
            let nc = ce1_cands.len();
            if nr > 0 && nc > 0 {
                let mut w = vec![vec![FORBID_SCORE; nc]; nr];
                for (ir, &u_cd2) in cd2_cands.iter().enumerate() {
                    for (ic, &u_ce1) in ce1_cands.iter().enumerate() {
                        let cb_hint = nodes[u_cd2].cb_hd_cands.first().map(|x| x.0);
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
                                );
                        let mut pair_sc = link_sc + 0.5 * cb_sc + 0.35 * prior_sc + t_post;
                        if !hmqc_peaks.is_empty() && link_sc == 0.0 && cb_sc <= 0.0 {
                            pair_sc -= 5.0;
                        }
                        w[ir][ic] = pair_sc;
                    }
                }
                let (r2c, best_tot) = hungarian_max_weight(&w);
                let pair_margins = hungarian_margins(&w, &r2c, best_tot);
                let mut matched_ce1 = vec![false; nc];

                for (ir, &u_cd2) in cd2_cands.iter().enumerate() {
                    let paired_ic = r2c[ir].filter(|&ic| w[ir][ic] > -3.5);
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
                        let (nd1, ne2, taut, link_sc) = infer_his_nitrogens_and_tautomer(
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
                            his_tautomer: Some((taut.0, taut.1 * pair_conf.max(0.85))),
                            trp_indole: None,
                            internal_coherence: link_sc + cb_sc.max(0.0),
                        });
                    } else if !used[u_cd2] {
                        used[u_cd2] = true;
                        let cb_hd = nodes[u_cd2].cb_hd_cands.first().map(|x| x.0);
                        let (nd1, ne2, taut, link_sc) = infer_his_nitrogens_and_tautomer(
                            Some(&nodes[u_cd2]),
                            None,
                            cb_hd,
                            &hmqc_peaks,
                            cfg,
                        );
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
                            his_tautomer: Some(taut),
                            trp_indole: None,
                            internal_coherence: link_sc,
                        });
                    }
                }
                for (ic, &u_ce1) in ce1_cands.iter().enumerate() {
                    if !matched_ce1[ic] && !used[u_ce1] {
                        used[u_ce1] = true;
                        let cb_he = nodes[u_ce1].cb_he_cands.first().map(|x| x.0);
                        let (nd1, ne2, taut, link_sc) = infer_his_nitrogens_and_tautomer(
                            None,
                            Some(&nodes[u_ce1]),
                            cb_he,
                            &hmqc_peaks,
                            cfg,
                        );
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
                            cb_hd: None,
                            cb_he,
                            nd1_ppm: nd1,
                            ne2_ppm: ne2,
                            his_tautomer: Some(taut),
                            trp_indole: None,
                            internal_coherence: link_sc,
                        });
                    }
                }
            } else {
                for &u_cd2 in &cd2_cands {
                    used[u_cd2] = true;
                    let cb_hd = nodes[u_cd2].cb_hd_cands.first().map(|x| x.0);
                    let (nd1, ne2, taut, link_sc) = infer_his_nitrogens_and_tautomer(
                        Some(&nodes[u_cd2]),
                        None,
                        cb_hd,
                        &hmqc_peaks,
                        cfg,
                    );
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
                        his_tautomer: Some(taut),
                        trp_indole: None,
                        internal_coherence: link_sc,
                    });
                }
                for &u_ce1 in &ce1_cands {
                    used[u_ce1] = true;
                    let cb_he = nodes[u_ce1].cb_he_cands.first().map(|x| x.0);
                    let (nd1, ne2, taut, link_sc) = infer_his_nitrogens_and_tautomer(
                        None,
                        Some(&nodes[u_ce1]),
                        cb_he,
                        &hmqc_peaks,
                        cfg,
                    );
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
                        cb_hd: None,
                        cb_he,
                        nd1_ppm: nd1,
                        ne2_ppm: ne2,
                        his_tautomer: Some(taut),
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
        // Seed benzenoid 4-spin chains from unclaimed CZ2 (~114.4 ppm) or multi-TOCSY nodes in 108.5..119.5 ppm
        let mut cz2_seeds: Vec<usize> = (0..nodes.len())
            .filter(|&u| {
                !used[u]
                    && (108.5..=119.5).contains(&nodes[u].c)
                    && (nodes[u].off_diag_tocsy_count(cfg.tol_h) >= 1
                        || nodes[u].height < 1.55 * h_1x)
            })
            .collect();
        cz2_seeds.sort_by(|&a, &b| nodes[a].c.total_cmp(&nodes[b].c));
        if backbone_ref.is_some() && cz2_seeds.len() > n_trp_seq {
            cz2_seeds.truncate(n_trp_seq);
        }

        #[allow(clippy::type_complexity)]
        let mut benz_chains: Vec<([Option<usize>; 4], f64, [f64; 4])> = Vec::new();
        for &u_cz2 in &cz2_seeds {
            if used[u_cz2] {
                continue;
            }
            // Candidate pool for (CE3, CZ3, CH2) connected to u_cz2 in 114.5..129.5 ppm
            let mut pool: Vec<usize> = (0..nodes.len())
                .filter(|&v| {
                    !used[v]
                        && v != u_cz2
                        && (114.5..=129.5).contains(&nodes[v].c)
                        && (arom_tocsy_edge_score(&nodes[u_cz2], &nodes[v], false, cfg.tol_h) > 0.5
                            || nodes[u_cz2].off_diag_tocsy_count(cfg.tol_h) == 0)
                })
                .collect();
            // Expand by 1 more TOCSY hop (for CE3 which is 3 bonds from CZ2)
            let hop1 = pool.clone();
            for &m in &hop1 {
                for v in 0..nodes.len() {
                    if !used[v]
                        && v != u_cz2
                        && !pool.contains(&v)
                        && (114.5..=129.5).contains(&nodes[v].c)
                        && arom_tocsy_edge_score(&nodes[m], &nodes[v], true, cfg.tol_h) > 1.8
                    {
                        pool.push(v);
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
                            if !nodes[u].cb_he_cands.is_empty() {
                                sc += 3.8;
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
                        if let (Some(u2), Some(u3)) = (o_cz3, o_ch2) {
                            sc += arom_tocsy_edge_score(&nodes[u2], &nodes[u3], true, cfg.tol_h);
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
                used[u] = true;
            }
            benz_chains.push((best_arr, best_sc, slot_confs));
        }

        // Pair each benzenoid chain with pyrrole CD1/HD1 and TRP_IND_N15HSQC (NE1, HE1)
        let cd1_cands: Vec<usize> = (0..nodes.len())
            .filter(|&u| {
                !used[u]
                    && (119.5..=132.5).contains(&nodes[u].c)
                    && nodes[u].off_diag_tocsy_count(cfg.tol_h) == 0
                    && (!nodes[u].cb_hd_cands.is_empty() || nodes[u].height < 1.65 * h_1x)
            })
            .collect();

        let nb = benz_chains.len();
        if nb > 0 {
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
                            sc += cb_sc;
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
                let mut slots = BTreeMap::new();
                let names = ["CE3", "CZ3", "CH2", "CZ2"];
                for (si, &c_name) in names.iter().enumerate() {
                    if let Some(u) = arr[si] {
                        slots.insert(
                            c_name,
                            RingChSlot {
                                c_ppm: nodes[u].c,
                                h_ppm: nodes[u].h,
                                intra_conf: s_confs[si],
                            },
                        );
                    }
                }

                let mut cb_hd = None;
                let mut cb_he = arr[0].and_then(|u| nodes[u].cb_he_cands.first().map(|x| x.0));

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
                            let (c1, c2, _) = best_shared_yamazaki_cb(
                                &nodes[u_cd1],
                                &nodes[u_ce3],
                                &known_arom_cbs,
                                cfg,
                            );
                            cb_hd = c1.or_else(|| nodes[u_cd1].cb_hd_cands.first().map(|x| x.0));
                            cb_he = c2.or(cb_he);
                        } else {
                            cb_hd = nodes[u_cd1].cb_hd_cands.first().map(|x| x.0);
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
                    trp_indole,
                    internal_coherence: b_sc,
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
            if used[u] || !(112.0..=124.2).contains(&n.c) {
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
            if used[u] || !(126.0..=138.2).contains(&n.c) {
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
                        trp_indole: None,
                        internal_coherence: w[ir][ic],
                    });
                } else {
                    let cb_he = nodes[u_ce].cb_he_cands.first().map(|x| x.0);
                    clusters.push(AromaticRingCluster {
                        kind: AromaticRingKind::Tyr,
                        ch_slots: slots,
                        cb_hd: None,
                        cb_he,
                        nd1_ppm: None,
                        ne2_ppm: None,
                        his_tautomer: None,
                        trp_indole: None,
                        internal_coherence: 0.0,
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
            .filter(|&u| !used[u] && (123.0..=138.0).contains(&nodes[u].c))
            .collect();

        if !rem.is_empty() {
            // Classify candidate CD*, CE*, and CZ pools from Yamazaki cross-peaks, 2x vs 1x height, and priors
            let mut cd_pool: Vec<usize> = Vec::new();
            let mut ce_pool: Vec<usize> = Vec::new();

            for &u in &rem {
                let n = &nodes[u];
                let has_hd = !n.cb_hd_cands.is_empty();
                let has_he = !n.cb_he_cands.is_empty();
                let is_2x = n.height >= 1.38 * h_1x;
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
                        let tocsy =
                            arom_tocsy_edge_score(&nodes[u_cd], &nodes[u_ce], true, cfg.tol_h);
                        let (_, _, cb_sc) = best_shared_yamazaki_cb(
                            &nodes[u_cd],
                            &nodes[u_ce],
                            &known_arom_cbs,
                            cfg,
                        );
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
            for &u in &rem {
                if stem_used.contains_key(&u) {
                    continue;
                }
                if !nodes[u].cb_hd_cands.is_empty() {
                    *stem_used.entry(u).or_insert(0) += 1;
                    let cb_hd = nodes[u].cb_hd_cands.first().map(|x| x.0);
                    stems.push((Some(u), None, cb_hd, None, 1.0, 0.88));
                } else if !nodes[u].cb_he_cands.is_empty() {
                    *stem_used.entry(u).or_insert(0) += 1;
                    let cb_he = nodes[u].cb_he_cands.first().map(|x| x.0);
                    stems.push((None, Some(u), None, cb_he, 1.0, 0.88));
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
    /// Bipartite ring-to-residue posterior confidence in `[0.0, 1.0]`.
    pub ring_confidence: f64,
    /// Log-likelihood ratio score of `(cluster_idx, residue_idx)`.
    pub score: f64,
    /// Leave-one-out bipartite margin `S* - S_alt(k)` against the best competing ring/residue swap.
    pub margin: f64,
    /// Classified histidine tautomeric state and posterior confidence (`Some` for assigned `HIS`).
    pub his_tautomer: Option<(HisTautomer, f64)>,
}

/// Complete output of `assign_aromatic_rings`.
#[derive(Clone, Debug)]
pub struct AromaticAssignResult {
    /// Updated `Reference` combining input backbone shifts with assigned aromatic `1H`/`13C`/`15N` shifts.
    pub reference: Reference,
    /// Calibrated per-atom confidence in `[0.0, 1.0]` keyed by `(residue_idx, iupac_atom_name)`.
    pub atom_confidence: BTreeMap<(usize, String), f64>,
    /// Per-aromatic-residue assignment details and `HisTautomer` calls.
    pub rings: Vec<AssignedAromaticRing>,
    /// Assembled experimental ring clusters.
    pub clusters: Vec<AromaticRingCluster>,
}

/// Compute pairwise log-likelihood ratio `S(cluster, residue_k)` combining Yamazaki
/// `|CB_yamazaki - CB_backbone(k)|`, optional `HB2/HB3` or `NE1/HE1` anchors, intra-ring TOCSY
/// coherence, and BMRB per-atom `(1H, 13C, 15N)` priors.
#[allow(clippy::too_many_arguments)]
fn score_cluster_for_residue(
    c: &AromaticRingCluster,
    k: usize,
    res_name: &str,
    backbone_ref: &Reference,
    anchor_conf: f64,
    aromatic_peaks: &[Peak],
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
        let hd_h = c
            .ch_slots
            .get("CD1")
            .or_else(|| c.ch_slots.get("CD2"))
            .map(|s| s.h_ppm);
        let he_h = match res_name {
            "TRP" => c.ch_slots.get("CE3").map(|s| s.h_ppm),
            _ => c.ch_slots.get("CE1").map(|s| s.h_ppm),
        };
        let best_yamazaki_for_h = |exp_name: &str, h_opt: Option<f64>, fallback: Option<f64>| {
            let mut best = fallback;
            if let Some(h0) = h_opt {
                for p in aromatic_peaks {
                    if p.exp == exp_name && (p.h - h0).abs() <= cfg.tol_h * 1.20 {
                        if let Some(cb) = p.c {
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
        // possess a self-consistent two-channel (cb_hd, cb_he) Yamazaki pair!
        let corroborated_pair = matches!(
            (c.cb_hd, c.cb_he),
            (Some(a), Some(b)) if (a - b).abs() <= cfg.tol_cb * 0.85
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
            }
        }
    } else if let Some(obs_cb) = c.cb_hd.or(c.cb_he) {
        // Missing backbone CB(k): score Yamazaki CB against amino-acid BMRB CB prior!
        llr += priors.llr_c13(res_name, "CB", CarbonRegion::AliphaticCaConnected, obs_cb);
        if res_name == "HIS" {
            if let Some((st, _)) = c.his_tautomer {
                let p_cb = st.shift_priors().6;
                llr += 0.5 * gauss_logpdf(obs_cb, p_cb.0, p_cb.1);
            }
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

/// Assign aromatic ring spin systems (`PHE`, `TYR`, `TRP`, `HIS`) to sequence positions via
/// exact maximum-weight bipartite matching (Kuhn–Munkres) with leave-one-out marginal confidence
/// calibration (`[0.0, 1.0]`).
pub fn assign_aromatic_rings(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    aromatic_peaks: &[Peak],
    priors: &SidechainPriors,
    cfg: &AromaticSolverConfig,
) -> AromaticAssignResult {
    let clusters = cluster_aromatic_rings(aromatic_peaks, Some(backbone_ref), priors, cfg);
    let arom_positions: Vec<usize> = backbone_ref
        .sequence
        .iter()
        .enumerate()
        .filter_map(|(k, r)| matches!(r.as_str(), "PHE" | "TYR" | "TRP" | "HIS").then_some(k))
        .collect();

    let mut out_ref = backbone_ref.clone();
    let mut atom_confidence: BTreeMap<(usize, String), f64> = BTreeMap::new();
    for &(k, ref a) in backbone_ref.shifts.keys() {
        let bc = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0);
        atom_confidence.insert((k, a.clone()), bc);
    }

    let n_res = arom_positions.len();
    let n_clu = clusters.len();
    if n_res == 0 {
        return AromaticAssignResult {
            reference: out_ref,
            atom_confidence,
            rings: Vec::new(),
            clusters,
        };
    }

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
                clu,
                k,
                res_name,
                backbone_ref,
                a_conf,
                aromatic_peaks,
                priors,
                cfg,
            );
        }
        // Dedicated null option for residue ri
        weights[ri][n_clu + ri] = null_score;
    }

    let (win_cols, best_total) = hungarian_max_weight(&weights);
    let margins = hungarian_margins(&weights, &win_cols, best_total);

    let mut rings = Vec::with_capacity(n_res);
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
            });
            continue;
        };

        let mrg = margins[ri];
        let win_sc = weights[ri][ci];
        // Exact leave-one-out Boltzmann marginal probability P(π(k) = ci) = 1 / (1 + exp(-ΔS / T_cal))
        // Note: when ΔS == 0 (e.g. identical CB on two PHEs in unfolded ubiquitin), P = 0.50 (< 0.80)!
        let bip_conf = (1.0 / (1.0 + (-mrg / cfg.calibration_temp).exp())) * a_conf;
        let clu = &clusters[ci];

        // Populate assigned IUPAC shifts & calibrated per-atom confidences into out_ref
        for (&c_atom, slot) in &clu.ch_slots {
            let at_conf = (bip_conf * slot.intra_conf).clamp(0.0, 1.0);
            match (res_name.as_str(), c_atom) {
                ("PHE" | "TYR", "CD1") => {
                    for &(ca, pa) in &[("CD1", "HD1"), ("CD2", "HD2")] {
                        out_ref.shifts.insert((k, ca.to_string()), slot.c_ppm);
                        out_ref.shifts.insert((k, pa.to_string()), slot.h_ppm);
                        atom_confidence.insert((k, ca.to_string()), at_conf);
                        atom_confidence.insert((k, pa.to_string()), at_conf);
                    }
                }
                ("PHE" | "TYR", "CE1") => {
                    for &(ca, pa) in &[("CE1", "HE1"), ("CE2", "HE2")] {
                        out_ref.shifts.insert((k, ca.to_string()), slot.c_ppm);
                        out_ref.shifts.insert((k, pa.to_string()), slot.h_ppm);
                        atom_confidence.insert((k, ca.to_string()), at_conf);
                        atom_confidence.insert((k, pa.to_string()), at_conf);
                    }
                }
                ("PHE", "CZ") => {
                    out_ref.shifts.insert((k, "CZ".to_string()), slot.c_ppm);
                    out_ref.shifts.insert((k, "HZ".to_string()), slot.h_ppm);
                    atom_confidence.insert((k, "CZ".to_string()), at_conf);
                    atom_confidence.insert((k, "HZ".to_string()), at_conf);
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
                    out_ref.shifts.insert((k, ca.to_string()), slot.c_ppm);
                    out_ref.shifts.insert((k, pa.to_string()), slot.h_ppm);
                    atom_confidence.insert((k, ca.to_string()), at_conf);
                    atom_confidence.insert((k, pa.to_string()), at_conf);
                }
                ("HIS", ca) => {
                    let pa = if ca == "CD2" { "HD2" } else { "HE1" };
                    out_ref.shifts.insert((k, ca.to_string()), slot.c_ppm);
                    out_ref.shifts.insert((k, pa.to_string()), slot.h_ppm);
                    atom_confidence.insert((k, ca.to_string()), at_conf);
                    atom_confidence.insert((k, pa.to_string()), at_conf);
                }
                _ => {}
            }
        }

        if res_name == "TRP" {
            if let Some((ne1, he1, icf)) = clu.trp_indole {
                let c_ind = (bip_conf * icf).clamp(0.0, 1.0);
                out_ref.shifts.insert((k, "NE1".to_string()), ne1);
                out_ref.shifts.insert((k, "HE1".to_string()), he1);
                atom_confidence.insert((k, "NE1".to_string()), c_ind);
                atom_confidence.insert((k, "HE1".to_string()), c_ind);
            }
        }

        if res_name == "HIS" {
            if let Some((nd1, ncf)) = clu.nd1_ppm {
                let c_n = (bip_conf * ncf).clamp(0.0, 1.0);
                out_ref.shifts.insert((k, "ND1".to_string()), nd1);
                atom_confidence.insert((k, "ND1".to_string()), c_n);
            }
            if let Some((ne2, ncf)) = clu.ne2_ppm {
                let c_n = (bip_conf * ncf).clamp(0.0, 1.0);
                out_ref.shifts.insert((k, "NE2".to_string()), ne2);
                atom_confidence.insert((k, "NE2".to_string()), c_n);
            }
        }

        // Populate CB if missing on backbone_ref
        if let std::collections::btree_map::Entry::Vacant(e) =
            out_ref.shifts.entry((k, "CB".to_string()))
        {
            if let Some(cb_y) = clu.cb_hd.or(clu.cb_he) {
                e.insert(cb_y);
                atom_confidence.insert((k, "CB".to_string()), bip_conf);
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
        });
    }

    AromaticAssignResult {
        reference: out_ref,
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
            let Some(&tv) = truth.shifts.get(&(k, atom.to_string())) else {
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
