//! Supramolecular `13C`-Methyl TROSY (`ILVMAT`) isotopic labeling schemes, Bayesian 2D `(1H, 13C)`
//! and `HMCM(CG)CBCA` methyl-type classifier, 3D PDB methyl/aromatic environment extractor, and
//! high-MW (`40–80 kDa`, `25 Hz / 18 Hz`) `2D HMQC`, `3D C-C-H`, and `4D C-H-C-H` volumetric
//! simulators and peak pickers.
//!
//! Provenance: Clean-room implementation from peer-reviewed biomolecular NMR literature:
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125,
//!   10120–10121 & 10420–10428 — `1H-13C` methyl-TROSY HMQC in high-MW proteins).
//! - Tugarinov, V. & Kay, L. E. (*J. Am. Chem. Soc.* 2003, 125, 13868–13878 — `HMCM(CG)CBCA`
//!   methyl-to-backbone scalar-coupled out-and-back correlation and geminal pairing).
//! - Gans, P. et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962 — stereospecific `pro-R` / `pro-S`
//!   `Leu` / `Val` acetolactate precursor labeling).
//! - Ruschak, A. M. & Kay, L. E. (*J. Biomol. NMR* 2010, 46, 75–87 — `3D C-C-H` and `4D C-H-C-H`
//!   methyl-TROSY NOESY spectroscopy of supramolecular systems).
//! - Xu, Y., Lin, Z., Ho, C. & Yang, D. (*J. Am. Chem. Soc.* 2009, 131, 9480–9481 — automated
//!   methyl resonance assignment from NOE networks).
//! - Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G. (*J. Magn. Reson.* 2012, 214, 103–110).
//! - Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J.
//!   & Kay, L. E. (*J. Am. Chem. Soc.* 2017, 139, 9523–9533 — probabilistic methyl NOE contact-graph
//!   matching).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::noesy::{
    calc_dihedral_deg, v_add, v_cross, v_dist, v_dot, v_scale, v_sub, v_unit, Vec3,
};
use crate::picker::estimate_noise_mad;
use crate::rng::Rng;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use crate::structure::nerf_place_atom;

// ============================================================================
// 1. Methyl Residue Classes, IUPAC Methyl Sites & Precursor Labeling Schemes
// ============================================================================

/// Six canonical methyl-bearing amino-acid classes in `ILVMAT` methyl-TROSY spectroscopy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MethylResidueClass {
    Ile = 0,
    Leu = 1,
    Val = 2,
    Met = 3,
    Ala = 4,
    Thr = 5,
}

impl MethylResidueClass {
    pub const ALL: [Self; 6] = [
        Self::Ile,
        Self::Leu,
        Self::Val,
        Self::Met,
        Self::Ala,
        Self::Thr,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn three_letter(self) -> &'static str {
        match self {
            Self::Ile => "ILE",
            Self::Leu => "LEU",
            Self::Val => "VAL",
            Self::Met => "MET",
            Self::Ala => "ALA",
            Self::Thr => "THR",
        }
    }

    pub fn from_three_letter(res: &str) -> Option<Self> {
        match res.trim().to_ascii_uppercase().as_str() {
            "ILE" => Some(Self::Ile),
            "LEU" => Some(Self::Leu),
            "VAL" => Some(Self::Val),
            "MET" => Some(Self::Met),
            "ALA" => Some(Self::Ala),
            "THR" => Some(Self::Thr),
            _ => None,
        }
    }
}

/// Specific IUPAC `13CH3` methyl group observable in high-MW `1H-13C` methyl-TROSY experiments.
///
/// Stereospecificity follows IUPAC-IUBMB-IUPAB 1998 recommendations (Markley et al., *J. Mol. Biol.*
/// 1998, 280, 933–952) and Gans et al. (*Angew. Chem. Int. Ed.* 2010, 49, 1958–1962):
/// - `LeuCd1` and `ValCg1` are stereospecifically `pro-R`,
/// - `LeuCd2` and `ValCg2` are stereospecifically `pro-S`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MethylGroupKind {
    IleCd1,
    LeuCd1,
    LeuCd2,
    ValCg1,
    ValCg2,
    MetCe,
    AlaCb,
    ThrCg2,
}

impl MethylGroupKind {
    pub const ALL: [Self; 8] = [
        Self::IleCd1,
        Self::LeuCd1,
        Self::LeuCd2,
        Self::ValCg1,
        Self::ValCg2,
        Self::MetCe,
        Self::AlaCb,
        Self::ThrCg2,
    ];

    #[inline]
    pub const fn residue_class(self) -> MethylResidueClass {
        match self {
            Self::IleCd1 => MethylResidueClass::Ile,
            Self::LeuCd1 | Self::LeuCd2 => MethylResidueClass::Leu,
            Self::ValCg1 | Self::ValCg2 => MethylResidueClass::Val,
            Self::MetCe => MethylResidueClass::Met,
            Self::AlaCb => MethylResidueClass::Ala,
            Self::ThrCg2 => MethylResidueClass::Thr,
        }
    }

    #[inline]
    pub const fn res_name(self) -> &'static str {
        self.residue_class().three_letter()
    }

    #[inline]
    pub const fn carbon_atom(self) -> &'static str {
        match self {
            Self::IleCd1 | Self::LeuCd1 => "CD1",
            Self::LeuCd2 => "CD2",
            Self::ValCg1 => "CG1",
            Self::ValCg2 | Self::ThrCg2 => "CG2",
            Self::MetCe => "CE",
            Self::AlaCb => "CB",
        }
    }

    #[inline]
    pub const fn proton_atoms(self) -> [&'static str; 3] {
        match self {
            Self::IleCd1 | Self::LeuCd1 => ["HD11", "HD12", "HD13"],
            Self::LeuCd2 => ["HD21", "HD22", "HD23"],
            Self::ValCg1 => ["HG11", "HG12", "HG13"],
            Self::ValCg2 | Self::ThrCg2 => ["HG21", "HG22", "HG23"],
            Self::MetCe => ["HE1", "HE2", "HE3"],
            Self::AlaCb => ["HB1", "HB2", "HB3"],
        }
    }

    #[inline]
    pub const fn parent_heavy_atom(self) -> &'static str {
        match self {
            Self::IleCd1 => "CG1",
            Self::LeuCd1 | Self::LeuCd2 => "CG",
            Self::ValCg1 | Self::ValCg2 | Self::ThrCg2 => "CB",
            Self::MetCe => "SD",
            Self::AlaCb => "CA",
        }
    }

    #[inline]
    pub const fn is_pro_r(self) -> bool {
        matches!(self, Self::LeuCd1 | Self::ValCg1)
    }

    #[inline]
    pub const fn is_pro_s(self) -> bool {
        matches!(self, Self::LeuCd2 | Self::ValCg2)
    }

    /// Empirical BMRB `(mu_c, sd_c, mu_h, sd_h)` prior in ppm for this methyl group.
    #[inline]
    pub const fn bmrb_prior(self) -> (f64, f64, f64, f64) {
        match self {
            Self::IleCd1 => (13.42, 1.65, 0.78, 0.28),
            Self::LeuCd1 => (24.35, 1.68, 0.82, 0.27),
            Self::LeuCd2 => (23.85, 1.68, 0.80, 0.27),
            Self::ValCg1 => (21.52, 1.48, 0.83, 0.26),
            Self::ValCg2 => (21.22, 1.48, 0.81, 0.26),
            Self::MetCe => (17.00, 1.50, 1.96, 0.29),
            Self::AlaCb => (19.10, 1.75, 1.36, 0.24),
            Self::ThrCg2 => (21.58, 1.22, 1.15, 0.22),
        }
    }

    pub fn from_res_atom(res_name: &str, carbon_atom: &str) -> Option<Self> {
        match (
            res_name.trim().to_ascii_uppercase().as_str(),
            carbon_atom.trim().to_ascii_uppercase().as_str(),
        ) {
            ("ILE", "CD1") => Some(Self::IleCd1),
            ("LEU", "CD1") => Some(Self::LeuCd1),
            ("LEU", "CD2") => Some(Self::LeuCd2),
            ("VAL", "CG1") => Some(Self::ValCg1),
            ("VAL", "CG2") => Some(Self::ValCg2),
            ("MET", "CE") => Some(Self::MetCe),
            ("ALA", "CB") => Some(Self::AlaCb),
            ("THR", "CG2") => Some(Self::ThrCg2),
            _ => None,
        }
    }
}

/// Selective `13CH3` precursor isotopic labeling scheme on a perdeuterated background.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MethylLabelingScheme {
    /// Non-stereospecific alpha-ketoisovalerate + alpha-ketobutyrate (`Ile-CD1`, `Leu-CD1/CD2`, `Val-CG1/CG2`).
    IlvRacemic,
    /// Stereospecific `pro-R` acetolactate + alpha-ketobutyrate (`Ile-CD1`, `Leu-CD1`, `Val-CG1`).
    IlvProR,
    /// Stereospecific `pro-S` acetolactate + alpha-ketobutyrate (`Ile-CD1`, `Leu-CD2`, `Val-CG2`).
    IlvProS,
    /// Six-residue methyl labeling (`Ile-CD1`, `Leu-CD1/CD2`, `Val-CG1/CG2`, `Met-CE`, `Ala-CB`, `Thr-CG2`).
    Ilvmat,
    /// Custom `(RES_3LETTER, CARBON_ATOM)` mask, e.g. `[("ILE", "CD1"), ("MET", "CE"), ("ALA", "CB")]`.
    Custom(Vec<(String, String)>),
}

impl MethylLabelingScheme {
    /// Parse a CLI or configuration labeling token (`ILV-proS`, `ILV-proR`, `ILV-racemic`, `ILVMAT`,
    /// or comma-separated `RES:ATOM` pairs such as `ILE:CD1,ALA:CB,MET:CE`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let trimmed = spec.trim();
        let norm = trimmed.to_ascii_lowercase().replace('_', "-");
        match norm.as_str() {
            "ilv-racemic" | "ilv" => Ok(Self::IlvRacemic),
            "ilv-pror" | "pror" => Ok(Self::IlvProR),
            "ilv-pros" | "pros" => Ok(Self::IlvProS),
            "ilvmat" => Ok(Self::Ilvmat),
            _ if trimmed.contains(':') => {
                let mut pairs = Vec::new();
                for item in trimmed.split(',') {
                    let (r, a) = item.trim().split_once(':').ok_or_else(|| {
                        format!(
                            "invalid custom methyl mask entry {item:?}; expected RES:ATOM (e.g. ILE:CD1)"
                        )
                    })?;
                    let ru = r.trim().to_ascii_uppercase();
                    let au = a.trim().to_ascii_uppercase();
                    if MethylGroupKind::from_res_atom(&ru, &au).is_none() {
                        return Err(format!(
                            "unsupported methyl site {ru}:{au} in custom labeling mask"
                        ));
                    }
                    if !pairs.iter().any(|(er, ea)| er == &ru && ea == &au) {
                        pairs.push((ru, au));
                    }
                }
                if pairs.is_empty() {
                    return Err("empty custom methyl labeling mask".to_string());
                }
                Ok(Self::Custom(pairs))
            }
            _ => Err(format!(
                "unknown methyl labeling scheme {trimmed:?}; expected ILV-proS, ILV-proR, ILV-racemic, ILVMAT, or RES:ATOM,..."
            )),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::IlvRacemic => "ILV-racemic".to_string(),
            Self::IlvProR => "ILV-proR".to_string(),
            Self::IlvProS => "ILV-proS".to_string(),
            Self::Ilvmat => "ILVMAT".to_string(),
            Self::Custom(pairs) => pairs
                .iter()
                .map(|(r, a)| format!("{r}:{a}"))
                .collect::<Vec<_>>()
                .join(","),
        }
    }

    #[inline]
    pub fn includes_kind(&self, kind: MethylGroupKind) -> bool {
        match self {
            Self::IlvRacemic => matches!(
                kind,
                MethylGroupKind::IleCd1
                    | MethylGroupKind::LeuCd1
                    | MethylGroupKind::LeuCd2
                    | MethylGroupKind::ValCg1
                    | MethylGroupKind::ValCg2
            ),
            Self::IlvProR => matches!(
                kind,
                MethylGroupKind::IleCd1 | MethylGroupKind::LeuCd1 | MethylGroupKind::ValCg1
            ),
            Self::IlvProS => matches!(
                kind,
                MethylGroupKind::IleCd1 | MethylGroupKind::LeuCd2 | MethylGroupKind::ValCg2
            ),
            Self::Ilvmat => true,
            Self::Custom(pairs) => {
                let r = kind.res_name();
                let c = kind.carbon_atom();
                pairs
                    .iter()
                    .any(|(pr, pc)| pr.eq_ignore_ascii_case(r) && pc.eq_ignore_ascii_case(c))
            }
        }
    }

    #[inline]
    pub fn includes(&self, res_name: &str, carbon_atom: &str) -> bool {
        MethylGroupKind::from_res_atom(res_name, carbon_atom).is_some_and(|k| self.includes_kind(k))
    }

    /// Active methyl-bearing residue classes permitted by this scheme.
    pub fn active_classes(&self) -> Vec<MethylResidueClass> {
        let mut out = Vec::new();
        for cls in MethylResidueClass::ALL {
            if MethylGroupKind::ALL
                .iter()
                .any(|&k| k.residue_class() == cls && self.includes_kind(k))
            {
                out.push(cls);
            }
        }
        out
    }

    /// True if both geminal methyls on `LEU` (`CD1` & `CD2`) or `VAL` (`CG1` & `CG2`) are active.
    pub fn has_geminal_pair(&self, res_name: &str) -> bool {
        match res_name.trim().to_ascii_uppercase().as_str() {
            "LEU" => {
                self.includes_kind(MethylGroupKind::LeuCd1)
                    && self.includes_kind(MethylGroupKind::LeuCd2)
            }
            "VAL" => {
                self.includes_kind(MethylGroupKind::ValCg1)
                    && self.includes_kind(MethylGroupKind::ValCg2)
            }
            _ => false,
        }
    }
}

// ============================================================================
// 2. HMCM(CG)CBCA Out-and-Back Scalar Correlation & Bayesian 2D Classifier
// ============================================================================

/// Out-and-back scalar-relayed aliphatic carbon frequencies (`CG`, `CB`, `CA`) correlated to a
/// methyl `(1H, 13C)` root in 3D `HMCM(CG)CBCA` (Tugarinov & Kay, *JACS* 2003, 125, 13868–13878).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HmcmcbcaRecord {
    pub h_ppm: f64,
    pub c_methyl_ppm: f64,
    pub cg_ppm: Option<f64>,
    pub cb_ppm: Option<f64>,
    pub ca_ppm: Option<f64>,
}

/// Posterior probability distribution across `[Ile, Leu, Val, Met, Ala, Thr]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethylClassPosterior {
    pub probs: [f64; 6],
}

impl MethylClassPosterior {
    #[inline]
    pub fn prob(&self, cls: MethylResidueClass) -> f64 {
        self.probs[cls.index()]
    }

    /// Maximum-a-posteriori (`MAP`) methyl residue class and its probability.
    pub fn map_class(&self) -> (MethylResidueClass, f64) {
        let mut best_c = MethylResidueClass::Ile;
        let mut best_p = -1.0;
        for cls in MethylResidueClass::ALL {
            let p = self.probs[cls.index()];
            if p > best_p {
                best_p = p;
                best_c = cls;
            }
        }
        (best_c, best_p.max(0.0))
    }
}

#[inline]
fn gaussian_pdf(x: f64, mu: f64, sd: f64) -> f64 {
    let z = (x - mu) / sd;
    (-0.5 * z * z).exp() / (sd * TAU.sqrt())
}

#[inline]
fn gaussian_logpdf(x: f64, mu: f64, sd: f64) -> f64 {
    let z = (x - mu) / sd;
    -0.5 * z * z - (sd * TAU.sqrt()).ln()
}

/// Empirical `(CA, CB, CG)` BMRB `(mean, sd)` tuple in ppm.
pub type AliphaticShiftPriors = (Option<(f64, f64)>, Option<(f64, f64)>, Option<(f64, f64)>);

/// Empirical `(CA, CB, CG)` BMRB `(mean, sd)` in ppm for `HMCM(CG)CBCA` amino-acid discrimination.
pub fn hmcmcbca_aliphatic_priors(cls: MethylResidueClass) -> AliphaticShiftPriors {
    // Returns (CA, CB, CG/CG1)
    match cls {
        MethylResidueClass::Ile => (
            Some((61.55, 2.55)),
            Some((38.60, 1.95)),
            Some((26.85, 1.65)),
        ),
        MethylResidueClass::Leu => (
            Some((55.15, 2.20)),
            Some((42.25, 1.90)),
            Some((26.75, 1.30)),
        ),
        MethylResidueClass::Val => (Some((62.10, 2.65)), Some((32.85, 1.80)), None),
        MethylResidueClass::Met => (None, None, None), // Thioether SD blocks 13C-13C relay
        MethylResidueClass::Ala => (Some((52.50, 2.15)), None, None),
        MethylResidueClass::Thr => (Some((61.95, 2.50)), Some((69.85, 1.65)), None),
    }
}

/// Compute `HMCM(CG)CBCA` log-likelihood contribution for `cls` given relayed `(CA, CB, CG)` shifts.
pub fn hmcmcbca_log_likelihood(rec: &HmcmcbcaRecord, cls: MethylResidueClass) -> f64 {
    let (ca_p, cb_p, cg_p) = hmcmcbca_aliphatic_priors(cls);
    let mut ll = 0.0;
    if let Some(ca_obs) = rec.ca_ppm {
        if let Some((mu, sd)) = ca_p {
            ll += gaussian_logpdf(ca_obs, mu, sd).max(-14.0);
        } else {
            ll -= 10.0;
        }
    }
    if let Some(cb_obs) = rec.cb_ppm {
        if let Some((mu, sd)) = cb_p {
            ll += gaussian_logpdf(cb_obs, mu, sd).max(-16.0);
        } else {
            ll -= 12.0;
        }
    }
    if let Some(cg_obs) = rec.cg_ppm {
        if let Some((mu, sd)) = cg_p {
            ll += gaussian_logpdf(cg_obs, mu, sd).max(-14.0);
        } else {
            // Val, Ala, Thr, Met have no relayed non-methyl CG in HMCM(CG)CBCA
            ll -= 12.0;
        }
    }
    ll
}

/// Classify an observed 2D `(1H, 13C)` methyl peak (plus optional `HMCM(CG)CBCA` shifts) into
/// normalized Bayesian posteriors over `[Ile, Leu, Val, Met, Ala, Thr]`, masked to `scheme`.
pub fn classify_methyl_peak(
    h_ppm: f64,
    c_ppm: f64,
    hmcm: Option<&HmcmcbcaRecord>,
    scheme: &MethylLabelingScheme,
) -> MethylClassPosterior {
    let mut log_w = [f64::NEG_INFINITY; 6];
    let mut kind_count = [0usize; 6];

    for kind in MethylGroupKind::ALL {
        if !scheme.includes_kind(kind) {
            continue;
        }
        let cls = kind.residue_class();
        let (mu_c, sd_c, mu_h, sd_h) = kind.bmrb_prior();

        // 13C density + two-component 1H density (92% core BMRB + 8% ring-current tail)
        let lc = gaussian_logpdf(c_ppm, mu_c, sd_c);
        let ph_core = 0.92 * gaussian_pdf(h_ppm, mu_h, sd_h);
        let ph_rc = 0.08 * gaussian_pdf(h_ppm, mu_h - 0.50, 0.85);
        let lh = (ph_core + ph_rc).max(1e-30).ln();

        let l_hmcm = hmcm.map(|r| hmcmcbca_log_likelihood(r, cls)).unwrap_or(0.0);
        let total = lc + lh + l_hmcm;

        // Equal-weight mixture across active methyl sites of that residue class (so each class integrates to 1)
        let idx = cls.index();
        kind_count[idx] += 1;
        let prev = log_w[idx];
        if prev == f64::NEG_INFINITY {
            log_w[idx] = total;
        } else {
            let hi = prev.max(total);
            log_w[idx] = hi + ((prev - hi).exp() + (total - hi).exp()).ln();
        }
    }

    for (lw, &cnt) in log_w.iter_mut().zip(&kind_count) {
        if cnt > 1 && lw.is_finite() {
            *lw -= (cnt as f64).ln();
        }
    }

    let max_lw = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return MethylClassPosterior { probs: [0.0; 6] };
    }

    let mut probs = [0.0; 6];
    let mut sum = 0.0;
    for i in 0..6 {
        if log_w[i].is_finite() {
            let w = (log_w[i] - max_lw).exp();
            probs[i] = w;
            sum += w;
        }
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    MethylClassPosterior { probs }
}

/// Identify geminal methyl root pairs `(root_a, root_b)` (`root_a < root_b`) for `Leu` (`CD1/CD2`)
/// and `Val` (`CG1/CG2`) from matched `HMCM(CG)CBCA` aliphatic carbon shifts.
pub fn link_hmcmcbca_geminal_pairs(
    roots: &[MethylRootPeak],
    tol_aliph_ppm: f64,
) -> Vec<(usize, usize)> {
    let n = roots.len();
    let mut cands: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        let Some(ref hi) = roots[i].hmcm else {
            continue;
        };
        let Some(cbi) = hi.cb_ppm else {
            continue;
        };
        // Only Leu (CB ~ 36..49) and Val (CB ~ 27..38) carry geminal methyl pairs; Thr CB ~ 70
        if !(26.5..=49.5).contains(&cbi) {
            continue;
        }
        for (j, rj) in roots.iter().enumerate().take(n).skip(i + 1) {
            let Some(ref hj) = rj.hmcm else {
                continue;
            };
            let Some(cbj) = hj.cb_ppm else {
                continue;
            };
            let d_cb = (cbi - cbj).abs();
            if d_cb > tol_aliph_ppm {
                continue;
            }
            let d_ca = match (hi.ca_ppm, hj.ca_ppm) {
                (Some(a), Some(b)) => {
                    let d = (a - b).abs();
                    if d > tol_aliph_ppm * 1.25 {
                        continue;
                    }
                    d
                }
                _ => 0.04,
            };
            let d_cg = match (hi.cg_ppm, hj.cg_ppm) {
                (Some(a), Some(b)) => {
                    let d = (a - b).abs();
                    if d > tol_aliph_ppm {
                        continue;
                    }
                    d
                }
                (None, None) => 0.0,
                _ => continue,
            };
            let dist = (d_cb * d_cb + d_ca * d_ca + d_cg * d_cg).sqrt();
            cands.push((dist, i, j));
        }
    }

    // Greedy 1-to-1 minimum-deviation matching so each methyl participates in at most 1 geminal pair
    cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used = vec![false; n];
    let mut pairs = Vec::new();
    for (_, i, j) in cands {
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            pairs.push((i, j));
        }
    }
    pairs.sort_unstable();
    pairs
}

// ============================================================================
// 3. 3D PDB Methyl & Aromatic Ring Environment Extractor
// ============================================================================

/// Aromatic ring geometric descriptor for Haigh–Mallion / Pople point-dipole ring-current modeling.
#[derive(Clone, Debug, PartialEq)]
pub struct AromaticRing {
    pub seq_id: i32,
    pub res_name: String,
    pub center: Vec3,
    pub normal: Vec3,
    /// Ring-current intensity factor relative to benzene (`Phe = 1.00`, `Tyr = 0.88`,
    /// `His = 0.53`, `Trp-5 = 0.90`, `Trp-6 = 1.05`; Perkins & Wuthrich 1979).
    pub ring_factor: f64,
}

/// A structural `13CH3` methyl site extracted from a 3D PDB model.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralMethylSite {
    pub site_idx: usize,
    pub residue_idx: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub carbon_atom: &'static str,
    pub kind: MethylGroupKind,
    pub carbon_xyz: Vec3,
    pub proton_centroid_xyz: Vec3,
    /// Sidechain $\chi_1$ (`Val`) or $\chi_2$ (`Ile`, `Leu`) dihedral in degrees when heavy atoms exist.
    pub chi_deg: Option<f64>,
    /// Index of the intraresidue geminal partner site (`Leu CD1<->CD2`, `Val CG1<->CG2`) if also active.
    pub geminal_partner: Option<usize>,
    /// Structure-informed expected aromatic ring-current `1H` shift offset in ppm (damped for rotamer motion).
    pub ring_current_h_offset_ppm: f64,
    /// Extra `1H` standard deviation (in ppm) induced by nearby aromatic ring-current gradients.
    pub ring_current_h_extra_sd_ppm: f64,
    /// Backbone $(\phi, \psi)$ angles in degrees when surrounding peptide atoms are present.
    pub phi_psi_deg: Option<(f64, f64)>,
}

/// Complete 3D structural methyl contact model extracted from a PDB file under a `MethylLabelingScheme`.
#[derive(Clone, Debug)]
pub struct MethylStructureModel {
    pub sequence: Vec<String>,
    pub seq_ids: Vec<i32>,
    pub scheme: MethylLabelingScheme,
    pub sites: Vec<StructuralMethylSite>,
    pub aromatic_rings: Vec<AromaticRing>,
    /// Row-major `M x M` carbon–carbon Euclidean distance matrix in Angstroms.
    pub d_cc: Vec<f64>,
}

impl MethylStructureModel {
    #[inline]
    pub fn dist_cc(&self, m1: usize, m2: usize) -> f64 {
        let m = self.sites.len();
        self.d_cc[m1 * m + m2]
    }
}

/// Extract active `StructuralMethylSite`s, aromatic rings, and the exact `M x M` methyl `C-C`
/// distance matrix from the first model of a PDB file.
pub fn extract_methyl_structure_from_pdb_file(
    path: impl AsRef<Path>,
    scheme: &MethylLabelingScheme,
) -> Result<MethylStructureModel, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read PDB {}: {e}", path.as_ref().display()))?;
    extract_methyl_structure_from_pdb_str(&text, scheme)
}

/// Extract active `StructuralMethylSite`s, aromatic rings, and `d_cc` from PDB text (first model,
/// first polypeptide chain).
pub fn extract_methyl_structure_from_pdb_str(
    pdb_text: &str,
    scheme: &MethylLabelingScheme,
) -> Result<MethylStructureModel, String> {
    extract_methyl_structure_inner(pdb_text, scheme, false)
}

/// Extract active `StructuralMethylSite`s, aromatic rings, and `d_cc` across ALL chains in the
/// first model of a multi-chain homooligomer / heterooligomer PDB file.
pub fn extract_multichain_methyl_structure_from_pdb_file(
    path: impl AsRef<Path>,
    scheme: &MethylLabelingScheme,
) -> Result<MethylStructureModel, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read PDB {}: {e}", path.as_ref().display()))?;
    extract_multichain_methyl_structure_from_pdb_str(&text, scheme)
}

/// Extract active `StructuralMethylSite`s, aromatic rings, and `d_cc` across ALL chains in the
/// first model of multi-chain PDB text.
pub fn extract_multichain_methyl_structure_from_pdb_str(
    pdb_text: &str,
    scheme: &MethylLabelingScheme,
) -> Result<MethylStructureModel, String> {
    extract_methyl_structure_inner(pdb_text, scheme, true)
}

fn extract_methyl_structure_inner(
    pdb_text: &str,
    scheme: &MethylLabelingScheme,
    all_chains: bool,
) -> Result<MethylStructureModel, String> {
    #[derive(Default)]
    struct RawResidue {
        chain_id: char,
        seq_id: i32,
        res_name: String,
        atoms: BTreeMap<String, Vec3>,
    }

    let mut residues: Vec<RawResidue> = Vec::new();
    let mut idx_by_seq: BTreeMap<(char, i32), usize> = BTreeMap::new();
    let mut selected_chain: Option<char> = None;

    #[inline]
    fn pdb_ascii_col(bytes: &[u8], start: usize, end: usize) -> Option<&str> {
        let slice = bytes.get(start..end)?;
        if !slice.is_ascii() {
            return None;
        }
        std::str::from_utf8(slice).ok()
    }

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        let bytes = line.as_bytes();
        if !line.starts_with("ATOM  ") || bytes.len() < 54 {
            continue;
        }
        let alt = bytes[16];
        if alt != b' ' && alt != b'A' && alt != b'1' {
            continue;
        }
        let chain_byte = bytes[21];
        if !chain_byte.is_ascii() {
            continue;
        }
        let chain_id = chain_byte as char;
        if !all_chains {
            if let Some(ch) = selected_chain {
                if chain_id != ch {
                    continue;
                }
            } else {
                selected_chain = Some(chain_id);
            }
        }
        if bytes[26] != b' ' {
            continue;
        }

        let (Some(atom_col), Some(res_col), Some(seq_col), Some(x_col), Some(y_col), Some(z_col)) = (
            pdb_ascii_col(bytes, 12, 16),
            pdb_ascii_col(bytes, 17, 20),
            pdb_ascii_col(bytes, 22, 26),
            pdb_ascii_col(bytes, 30, 38),
            pdb_ascii_col(bytes, 38, 46),
            pdb_ascii_col(bytes, 46, 54),
        ) else {
            continue;
        };
        let atom_name = atom_col.trim().to_ascii_uppercase();
        let res_name = res_col.trim().to_ascii_uppercase();
        let Ok(seq_id) = seq_col.trim().parse::<i32>() else {
            continue;
        };
        let Ok(x) = x_col.trim().parse::<f64>() else {
            continue;
        };
        let Ok(y) = y_col.trim().parse::<f64>() else {
            continue;
        };
        let Ok(z) = z_col.trim().parse::<f64>() else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }

        let r_idx = *idx_by_seq.entry((chain_id, seq_id)).or_insert_with(|| {
            let id = residues.len();
            residues.push(RawResidue {
                chain_id,
                seq_id,
                res_name: res_name.clone(),
                atoms: BTreeMap::new(),
            });
            id
        });
        residues[r_idx].atoms.entry(atom_name).or_insert([x, y, z]);
    }

    if residues.is_empty() {
        return Err("no ATOM records found in PDB model".to_string());
    }

    let sequence: Vec<String> = residues.iter().map(|r| r.res_name.clone()).collect();
    let seq_ids: Vec<i32> = residues.iter().map(|r| r.seq_id).collect();

    // 1. Extract Phe / Tyr / His / Trp aromatic rings
    let mut aromatic_rings = Vec::new();
    const RING_SPECS: &[(&str, &[&str], f64)] = &[
        ("PHE", &["CG", "CD1", "CE1", "CZ", "CE2", "CD2"], 1.00),
        ("TYR", &["CG", "CD1", "CE1", "CZ", "CE2", "CD2"], 0.88),
        ("HIS", &["CG", "ND1", "CE1", "NE2", "CD2"], 0.53),
        ("TRP", &["CG", "CD1", "NE1", "CE2", "CD2"], 0.90),
        ("TRP", &["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"], 1.05),
    ];

    for raw in &residues {
        for &(rname, ring_atoms, factor) in RING_SPECS {
            if raw.res_name == rname {
                let pts: Vec<Vec3> = ring_atoms
                    .iter()
                    .filter_map(|&a| raw.atoms.get(a).copied())
                    .collect();
                if pts.len() >= 5 {
                    let inv = 1.0 / (pts.len() as f64);
                    let mut cen = [0.0; 3];
                    for p in &pts {
                        cen = v_add(cen, *p);
                    }
                    cen = v_scale(cen, inv);
                    let normal = v_unit(v_cross(v_sub(pts[1], pts[0]), v_sub(pts[2], pts[0])));
                    aromatic_rings.push(AromaticRing {
                        seq_id: raw.seq_id,
                        res_name: raw.res_name.clone(),
                        center: cen,
                        normal,
                        ring_factor: factor,
                    });
                }
            }
        }
    }

    // 2. Extract active methyl sites
    let mut sites: Vec<StructuralMethylSite> = Vec::new();
    for (r_idx, raw) in residues.iter().enumerate() {
        let phi_psi = {
            let prev_c = if r_idx > 0 && residues[r_idx - 1].chain_id == raw.chain_id {
                residues[r_idx - 1].atoms.get("C").copied()
            } else {
                None
            };
            let next_n =
                if r_idx + 1 < residues.len() && residues[r_idx + 1].chain_id == raw.chain_id {
                    residues[r_idx + 1].atoms.get("N").copied()
                } else {
                    None
                };
            match (
                prev_c,
                raw.atoms.get("N").copied(),
                raw.atoms.get("CA").copied(),
                raw.atoms.get("C").copied(),
                next_n,
            ) {
                (Some(cp), Some(n), Some(ca), Some(c), Some(nn)) => {
                    let phi = calc_dihedral_deg(cp, n, ca, c);
                    let psi = calc_dihedral_deg(n, ca, c, nn);
                    Some((phi, psi))
                }
                _ => None,
            }
        };

        for kind in MethylGroupKind::ALL {
            if kind.res_name() != raw.res_name || !scheme.includes_kind(kind) {
                continue;
            }
            let c_name = kind.carbon_atom();
            let parent_name = kind.parent_heavy_atom();

            // Resolve carbon coordinate (or fallback ideal trans-rotamer placement if absent in incomplete X-ray PDB)
            let Some((c_xyz, parent_xyz)) = resolve_methyl_and_parent_xyz(&raw.atoms, kind) else {
                continue;
            };

            // Methyl proton pseudoatom centroid: average explicit protons if present, else tetrahedral +0.363 A along parent->C
            let h_pts: Vec<Vec3> = kind
                .proton_atoms()
                .iter()
                .filter_map(|&hn| raw.atoms.get(hn).copied())
                .collect();
            let qh_xyz = if !h_pts.is_empty() {
                let mut s = [0.0; 3];
                for p in &h_pts {
                    s = v_add(s, *p);
                }
                v_scale(s, 1.0 / (h_pts.len() as f64))
            } else {
                let dir = v_unit(v_sub(c_xyz, parent_xyz));
                v_add(c_xyz, v_scale(dir, 0.3633))
            };

            // Sidechain dihedral for gamma-gauche 13C shift expectation
            let chi_deg = match kind {
                MethylGroupKind::IleCd1 => match (
                    raw.atoms.get("CA"),
                    raw.atoms.get("CB"),
                    raw.atoms.get("CG1"),
                ) {
                    (Some(&ca), Some(&cb), Some(&cg1)) => {
                        Some(calc_dihedral_deg(ca, cb, cg1, c_xyz))
                    }
                    _ => None,
                },
                MethylGroupKind::LeuCd1 | MethylGroupKind::LeuCd2 => {
                    match (
                        raw.atoms.get("CA"),
                        raw.atoms.get("CB"),
                        raw.atoms.get("CG"),
                    ) {
                        (Some(&ca), Some(&cb), Some(&cg)) => {
                            Some(calc_dihedral_deg(ca, cb, cg, c_xyz))
                        }
                        _ => None,
                    }
                }
                MethylGroupKind::ValCg1 | MethylGroupKind::ValCg2 => {
                    match (raw.atoms.get("N"), raw.atoms.get("CA"), raw.atoms.get("CB")) {
                        (Some(&n), Some(&ca), Some(&cb)) => {
                            Some(calc_dihedral_deg(n, ca, cb, c_xyz))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };

            // Point-dipole aromatic ring-current expectation + proximity variance inflation
            let mut rc_shift = 0.0;
            let mut rc_var = 0.0;
            for ring in &aromatic_rings {
                let dv = v_sub(qh_xyz, ring.center);
                let d = v_dist(qh_xyz, ring.center);
                if (1.8..10.5).contains(&d) {
                    let cos_th = v_dot(dv, ring.normal) / d;
                    // Damped Haigh-Mallion/Pople dipole (12.0 ppm*A^3 accounting for rotamer averaging)
                    rc_shift +=
                        12.0 * ring.ring_factor * (1.0 - 3.0 * cos_th * cos_th) / (d * d * d);
                    let prox = 0.85 * ring.ring_factor * (5.0 / d.max(3.2)).powi(3);
                    rc_var += prox * prox;
                }
            }
            let rc_offset = rc_shift.clamp(-1.25, 0.65);
            let rc_extra_sd = rc_var.sqrt().min(1.10);

            let _ = parent_name;
            let site_idx = sites.len();
            sites.push(StructuralMethylSite {
                site_idx,
                residue_idx: r_idx,
                seq_id: raw.seq_id,
                res_name: raw.res_name.clone(),
                carbon_atom: c_name,
                kind,
                carbon_xyz: c_xyz,
                proton_centroid_xyz: qh_xyz,
                chi_deg,
                geminal_partner: None,
                ring_current_h_offset_ppm: rc_offset,
                ring_current_h_extra_sd_ppm: rc_extra_sd,
                phi_psi_deg: phi_psi,
            });
        }
    }

    // Link intraresidue geminal partners (Leu CD1<->CD2, Val CG1<->CG2)
    let m_len = sites.len();
    for i in 0..m_len {
        for j in (i + 1)..m_len {
            if sites[i].residue_idx == sites[j].residue_idx {
                sites[i].geminal_partner = Some(j);
                sites[j].geminal_partner = Some(i);
            }
        }
    }

    let mut d_cc = vec![0.0f64; m_len * m_len];
    for i in 0..m_len {
        for j in (i + 1)..m_len {
            let d = v_dist(sites[i].carbon_xyz, sites[j].carbon_xyz);
            d_cc[i * m_len + j] = d;
            d_cc[j * m_len + i] = d;
        }
    }

    Ok(MethylStructureModel {
        sequence,
        seq_ids,
        scheme: scheme.clone(),
        sites,
        aromatic_rings,
        d_cc,
    })
}

/// Construct an exact $C_K$-symmetric homooligomer (`K >= 2`, e.g. $C_2$ homodimer or $C_6$
/// homohexamer) from a monomer `MethylStructureModel` by applying cyclic $z$-axis rotations
/// $R_z(2\pi k / K)$ after radial translation chosen so the minimum inter-subunit methyl `C-C`
/// contact distance equals `min_inter_subunit_cc_angstrom`.
pub fn build_cn_homooligomer_methyl_model(
    monomer: &MethylStructureModel,
    n_subunits: usize,
    min_inter_subunit_cc_angstrom: f64,
) -> MethylStructureModel {
    assert!(
        n_subunits >= 2,
        "homooligomer requires n_subunits >= 2, got {n_subunits}"
    );
    let m0 = monomer.sites.len();
    let r0 = monomer.sequence.len();
    if m0 == 0 {
        return MethylStructureModel {
            sequence: (0..n_subunits)
                .flat_map(|_| monomer.sequence.iter().cloned())
                .collect(),
            seq_ids: monomer.seq_ids.repeat(n_subunits),
            scheme: monomer.scheme.clone(),
            sites: Vec::new(),
            aromatic_rings: Vec::new(),
            d_cc: Vec::new(),
        };
    }

    // Center monomer at its methyl carbon centroid
    let mut cen = [0.0f64; 3];
    for st in &monomer.sites {
        cen = v_add(cen, st.carbon_xyz);
    }
    cen = v_scale(cen, 1.0 / (m0 as f64));

    let rot_z = |p: Vec3, theta: f64| -> Vec3 {
        let (s, c) = theta.sin_cos();
        [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
    };

    let min_inter_at_radius = |radius: f64| -> f64 {
        let mut min_d = f64::INFINITY;
        for k in 1..n_subunits {
            let th = TAU * (k as f64) / (n_subunits as f64);
            for a in &monomer.sites {
                let pa = [
                    a.carbon_xyz[0] - cen[0] + radius,
                    a.carbon_xyz[1] - cen[1],
                    a.carbon_xyz[2] - cen[2],
                ];
                for b in &monomer.sites {
                    let pb0 = [
                        b.carbon_xyz[0] - cen[0] + radius,
                        b.carbon_xyz[1] - cen[1],
                        b.carbon_xyz[2] - cen[2],
                    ];
                    let pb = rot_z(pb0, th);
                    let d = v_dist(pa, pb);
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
        }
        min_d
    };

    let target_gap = min_inter_subunit_cc_angstrom.max(3.5);
    let mut lo = 0.0f64;
    let mut hi = 60.0f64;
    while min_inter_at_radius(hi) < target_gap && hi < 500.0 {
        hi *= 1.8;
    }
    for _ in 0..44 {
        let mid = 0.5 * (lo + hi);
        if min_inter_at_radius(mid) < target_gap {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let radius = 0.5 * (lo + hi);

    let mut sequence = Vec::with_capacity(r0 * n_subunits);
    let mut seq_ids = Vec::with_capacity(r0 * n_subunits);
    let mut aromatic_rings = Vec::with_capacity(monomer.aromatic_rings.len() * n_subunits);
    let mut sites = Vec::with_capacity(m0 * n_subunits);

    for k in 0..n_subunits {
        let th = TAU * (k as f64) / (n_subunits as f64);
        sequence.extend_from_slice(&monomer.sequence);
        seq_ids.extend_from_slice(&monomer.seq_ids);

        for ring in &monomer.aromatic_rings {
            let rc0 = [
                ring.center[0] - cen[0] + radius,
                ring.center[1] - cen[1],
                ring.center[2] - cen[2],
            ];
            aromatic_rings.push(AromaticRing {
                seq_id: ring.seq_id,
                res_name: ring.res_name.clone(),
                center: rot_z(rc0, th),
                normal: rot_z(ring.normal, th),
                ring_factor: ring.ring_factor,
            });
        }
    }

    for k in 0..n_subunits {
        let th = TAU * (k as f64) / (n_subunits as f64);
        let site_base = k * m0;
        let res_base = k * r0;
        for st in &monomer.sites {
            let c0 = [
                st.carbon_xyz[0] - cen[0] + radius,
                st.carbon_xyz[1] - cen[1],
                st.carbon_xyz[2] - cen[2],
            ];
            let h0 = [
                st.proton_centroid_xyz[0] - cen[0] + radius,
                st.proton_centroid_xyz[1] - cen[1],
                st.proton_centroid_xyz[2] - cen[2],
            ];
            let c_xyz = rot_z(c0, th);
            let qh_xyz = rot_z(h0, th);

            let mut rc_shift = 0.0;
            let mut rc_var = 0.0;
            for ring in &aromatic_rings {
                let dv = v_sub(qh_xyz, ring.center);
                let d = v_dist(qh_xyz, ring.center);
                if (1.8..10.5).contains(&d) {
                    let cos_th = v_dot(dv, ring.normal) / d;
                    rc_shift +=
                        12.0 * ring.ring_factor * (1.0 - 3.0 * cos_th * cos_th) / (d * d * d);
                    let prox = 0.85 * ring.ring_factor * (5.0 / d.max(3.2)).powi(3);
                    rc_var += prox * prox;
                }
            }

            sites.push(StructuralMethylSite {
                site_idx: site_base + st.site_idx,
                residue_idx: res_base + st.residue_idx,
                seq_id: st.seq_id,
                res_name: st.res_name.clone(),
                carbon_atom: st.carbon_atom,
                kind: st.kind,
                carbon_xyz: c_xyz,
                proton_centroid_xyz: qh_xyz,
                chi_deg: st.chi_deg,
                geminal_partner: st.geminal_partner.map(|p| site_base + p),
                ring_current_h_offset_ppm: rc_shift.clamp(-1.25, 0.65),
                ring_current_h_extra_sd_ppm: rc_var.sqrt().min(1.10),
                phi_psi_deg: st.phi_psi_deg,
            });
        }
    }

    let m_len = sites.len();
    let mut d_cc = vec![0.0f64; m_len * m_len];
    for i in 0..m_len {
        for j in (i + 1)..m_len {
            let d = v_dist(sites[i].carbon_xyz, sites[j].carbon_xyz);
            d_cc[i * m_len + j] = d;
            d_cc[j * m_len + i] = d;
        }
    }

    MethylStructureModel {
        sequence,
        seq_ids,
        scheme: monomer.scheme.clone(),
        sites,
        aromatic_rings,
        d_cc,
    }
}

/// Convenience builder for an exact $C_2$-symmetric homodimer (`2 * M` sites).
pub fn build_c2_homodimer_methyl_model(
    monomer: &MethylStructureModel,
    min_inter_subunit_cc_angstrom: f64,
) -> MethylStructureModel {
    build_cn_homooligomer_methyl_model(monomer, 2, min_inter_subunit_cc_angstrom)
}

fn resolve_methyl_and_parent_xyz(
    atoms: &BTreeMap<String, Vec3>,
    kind: MethylGroupKind,
) -> Option<(Vec3, Vec3)> {
    let c_name = kind.carbon_atom();
    let p_name = kind.parent_heavy_atom();
    if let (Some(&c), Some(&p)) = (atoms.get(c_name), atoms.get(p_name)) {
        return Some((c, p));
    }
    // Fallback reconstruction from (N, CA, C) when sidechain atoms are absent in backbone-only models
    let (&n, &ca, &c_bb) = (atoms.get("N")?, atoms.get("CA")?, atoms.get("C")?);
    let b1 = v_unit(v_sub(n, ca));
    let b2 = v_unit(v_sub(c_bb, ca));
    let u = v_unit(v_add(b1, b2));
    let normal = v_unit(v_cross(b1, b2));
    let dir_cb = v_unit(v_add(v_scale(u, -0.5810), v_scale(normal, 0.8139)));
    let cb = atoms
        .get("CB")
        .copied()
        .unwrap_or_else(|| v_add(ca, v_scale(dir_cb, 1.526)));

    let tet = 109.47_f64.to_radians();
    let trans = std::f64::consts::PI;
    let gauche_p = 60.0_f64.to_radians();
    let gauche_m = -60.0_f64.to_radians();

    match kind {
        MethylGroupKind::AlaCb => Some((cb, ca)),
        MethylGroupKind::ThrCg2 => {
            let cg2 = nerf_place_atom(n, ca, cb, 1.520, tet, gauche_m);
            Some((cg2, cb))
        }
        MethylGroupKind::ValCg1 => {
            let cg1 = nerf_place_atom(n, ca, cb, 1.520, tet, trans);
            Some((cg1, cb))
        }
        MethylGroupKind::ValCg2 => {
            let cg2 = nerf_place_atom(n, ca, cb, 1.520, tet, gauche_p);
            Some((cg2, cb))
        }
        MethylGroupKind::IleCd1 => {
            let cg1 = atoms
                .get("CG1")
                .copied()
                .unwrap_or_else(|| nerf_place_atom(n, ca, cb, 1.530, tet, -60.0_f64.to_radians()));
            let cd1 = nerf_place_atom(ca, cb, cg1, 1.520, tet, trans);
            Some((cd1, cg1))
        }
        MethylGroupKind::LeuCd1 => {
            let cg = atoms
                .get("CG")
                .copied()
                .unwrap_or_else(|| nerf_place_atom(n, ca, cb, 1.530, tet, -60.0_f64.to_radians()));
            let cd1 = nerf_place_atom(ca, cb, cg, 1.520, tet, trans);
            Some((cd1, cg))
        }
        MethylGroupKind::LeuCd2 => {
            let cg = atoms
                .get("CG")
                .copied()
                .unwrap_or_else(|| nerf_place_atom(n, ca, cb, 1.530, tet, -60.0_f64.to_radians()));
            let cd2 = nerf_place_atom(ca, cb, cg, 1.520, tet, gauche_p);
            Some((cd2, cg))
        }
        MethylGroupKind::MetCe => {
            let cg = atoms
                .get("CG")
                .copied()
                .unwrap_or_else(|| nerf_place_atom(n, ca, cb, 1.520, tet, -65.0_f64.to_radians()));
            let sd = atoms
                .get("SD")
                .copied()
                .unwrap_or_else(|| nerf_place_atom(ca, cb, cg, 1.810, tet, trans));
            let ce = nerf_place_atom(cb, cg, sd, 1.790, 100.9_f64.to_radians(), gauche_p);
            Some((ce, sd))
        }
    }
}

/// Look up experimental `(1H, 13C)` methyl shifts for a `StructuralMethylSite` from a `Reference`.
pub fn lookup_site_methyl_shifts(
    ref_data: &Reference,
    site: &StructuralMethylSite,
) -> Option<(f64, f64)> {
    // Match by Seq_ID first, falling back to residue_idx
    let r_idx = ref_data
        .seq_ids
        .iter()
        .position(|&sid| sid == site.seq_id)
        .unwrap_or(site.residue_idx);
    let c_ppm = *ref_data
        .shifts
        .get(&(r_idx, site.carbon_atom.to_string()))?;
    let mut h_sum = 0.0;
    let mut h_cnt = 0usize;
    for &hp in &site.kind.proton_atoms() {
        if let Some(&hv) = ref_data.shifts.get(&(r_idx, hp.to_string())) {
            h_sum += hv;
            h_cnt += 1;
        }
    }
    if h_cnt == 0 {
        // Also check collapsed pseudoatom prefix e.g. HD1 / HG1 / HB / HE
        let prefix = &site.kind.proton_atoms()[0][..site.kind.proton_atoms()[0].len() - 1];
        if let Some(&hv) = ref_data.shifts.get(&(r_idx, prefix.to_string())) {
            return Some((hv, c_ppm));
        }
        return None;
    }
    Some((h_sum / (h_cnt as f64), c_ppm))
}

/// Extract ordered `(1H, 13C)` shifts for every site in `model` from `ref_data`
/// (falling back to BMRB mean shifts if an individual atom is absent).
pub fn extract_true_site_shifts(
    model: &MethylStructureModel,
    ref_data: &Reference,
) -> Vec<(f64, f64)> {
    model
        .sites
        .iter()
        .map(|st| {
            lookup_site_methyl_shifts(ref_data, st).unwrap_or_else(|| {
                let (mc, _, mh, _) = st.kind.bmrb_prior();
                (mh + st.ring_current_h_offset_ppm, mc)
            })
        })
        .collect()
}

/// Extract `HMCM(CG)CBCA` scalar-relayed `(CG, CB, CA)` records from `ref_data` for all active
/// sites in `model` (`Ile` `CG1/CB/CA`, `Leu` `CG/CB/CA`, `Val` `CB/CA`, `Ala` `CA`, `Thr` `CB/CA`).
pub fn extract_hmcmcbca_from_reference(
    model: &MethylStructureModel,
    ref_data: &Reference,
) -> Vec<HmcmcbcaRecord> {
    let mut out = Vec::with_capacity(model.sites.len());
    for st in &model.sites {
        let Some((h_ppm, c_methyl_ppm)) = lookup_site_methyl_shifts(ref_data, st) else {
            continue;
        };
        let r_idx = ref_data
            .seq_ids
            .iter()
            .position(|&sid| sid == st.seq_id)
            .unwrap_or(st.residue_idx);
        let (cg_ppm, cb_ppm, ca_ppm) = match st.kind.residue_class() {
            MethylResidueClass::Ile => (
                ref_data.shifts.get(&(r_idx, "CG1".to_string())).copied(),
                ref_data.shifts.get(&(r_idx, "CB".to_string())).copied(),
                ref_data.shifts.get(&(r_idx, "CA".to_string())).copied(),
            ),
            MethylResidueClass::Leu => (
                ref_data.shifts.get(&(r_idx, "CG".to_string())).copied(),
                ref_data.shifts.get(&(r_idx, "CB".to_string())).copied(),
                ref_data.shifts.get(&(r_idx, "CA".to_string())).copied(),
            ),
            MethylResidueClass::Val | MethylResidueClass::Thr => (
                None,
                ref_data.shifts.get(&(r_idx, "CB".to_string())).copied(),
                ref_data.shifts.get(&(r_idx, "CA".to_string())).copied(),
            ),
            MethylResidueClass::Ala => (
                None,
                None,
                ref_data.shifts.get(&(r_idx, "CA".to_string())).copied(),
            ),
            MethylResidueClass::Met => continue,
        };
        if cg_ppm.is_some() || cb_ppm.is_some() || ca_ppm.is_some() {
            out.push(HmcmcbcaRecord {
                h_ppm,
                c_methyl_ppm,
                cg_ppm,
                cb_ppm,
                ca_ppm,
            });
        }
    }
    out
}

// ============================================================================
// 4. Peak Data Models, TSV Interchange & High-MW Volumetric Simulators/Pickers
// ============================================================================

/// Observed `2D 1H-13C HMQC` methyl root peak.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethylRootPeak {
    pub id: usize,
    pub h_ppm: f64,
    pub c_ppm: f64,
    pub height: f64,
    pub volume: f64,
    pub hmcm: Option<HmcmcbcaRecord>,
}

/// Observed cross-peak in `3D C-C-H` (`(C_noe, C_root, H_root)`) or `4D C-H-C-H` `HMQC-NOESY-HMQC`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethylNoePeak {
    pub c_noe_ppm: f64,
    pub c_root_ppm: f64,
    pub h_root_ppm: f64,
    /// Present for `4D C-H-C-H` cross-peaks (`None` in `3D C-C-H` unless uniquely matched).
    pub h_noe_ppm: Option<f64>,
    pub intensity: f64,
    pub root_idx: Option<usize>,
    pub donor_root_idx: Option<usize>,
    pub bidirectional: bool,
}

/// 4D `C-H-C-H` `HMQC-NOESY-HMQC` volumetric cube stored `[ic_noe, ih_noe, ic_root, ih_root]`.
#[derive(Clone, Debug)]
pub struct Spectrum4DMethyl {
    pub name: String,
    pub axis_h_root: AxisPpm,
    pub axis_c_root: AxisPpm,
    pub axis_h_noe: AxisPpm,
    pub axis_c_noe: AxisPpm,
    pub data: Vec<f32>,
}

impl Spectrum4DMethyl {
    #[inline]
    pub fn get(&self, ih_r: usize, ic_r: usize, ih_n: usize, ic_n: usize) -> f32 {
        let nh_r = self.axis_h_root.size;
        let nc_r = self.axis_c_root.size;
        let nh_n = self.axis_h_noe.size;
        self.data[((ic_n * nh_n + ih_n) * nc_r + ic_r) * nh_r + ih_r]
    }
}

/// High-molecular-weight (`40–80 kDa`) methyl-TROSY simulation parameters.
#[derive(Clone, Debug)]
pub struct MethylSimConfig {
    /// Proton Larmor frequency in MHz (`default = 800.0`, `13C = 201.16 MHz`).
    pub obs_h_mhz: f64,
    /// `1H` full-width at half-maximum in Hz (`default = 25.0 Hz`, `40–80 kDa` regime).
    pub lw_h_hz: f64,
    /// `13C` full-width at half-maximum in Hz (`default = 18.0 Hz`, `40–80 kDa` regime).
    pub lw_c_hz: f64,
    /// Lorentzian mixing weight in convex pseudo-Voigt lineshape (`default = 0.15`, exact FWHM preserved).
    pub lorentz_frac: f64,
    pub hmqc_nx: usize,
    pub hmqc_ny: usize,
    pub noesy_nx: usize,
    pub noesy_ny: usize,
    pub noesy_nz: usize,
    /// Signal-to-noise ratio of a single `13CH3` HMQC peak (`default = 65.0`).
    pub snr_hmqc: f64,
    /// Signal-to-noise ratio of a `d_CC = 5.0 A` NOESY cross-peak (`default = 185.0`, giving ~9.1 intrinsic SNR at `8.5 A`).
    pub snr_noe_5a: f64,
    /// Maximum methyl `C-C` distance simulated in NOESY (`default = 9.5 A`).
    pub r_max_noe_angstrom: f64,
    /// Short-range multi-spin saturation damping length (`default = 3.8 A`).
    pub r_sat_angstrom: f64,
    pub seed: u64,
}

impl Default for MethylSimConfig {
    fn default() -> Self {
        Self {
            obs_h_mhz: 800.0,
            lw_h_hz: 25.0,
            lw_c_hz: 18.0,
            lorentz_frac: 0.15,
            hmqc_nx: 512,
            hmqc_ny: 384,
            noesy_nx: 256,
            noesy_ny: 256,
            noesy_nz: 256,
            snr_hmqc: 65.0,
            snr_noe_5a: 185.0,
            r_max_noe_angstrom: 9.5,
            r_sat_angstrom: 3.8,
            seed: 2026,
        }
    }
}

impl MethylSimConfig {
    #[inline]
    pub fn obs_c_mhz(&self) -> f64 {
        self.obs_h_mhz * 0.251_450_20
    }

    #[inline]
    pub fn fwhm_h_ppm(&self) -> f64 {
        self.lw_h_hz / self.obs_h_mhz
    }

    #[inline]
    pub fn fwhm_c_ppm(&self) -> f64 {
        self.lw_c_hz / self.obs_c_mhz()
    }
}

/// Peak-picking configuration for `2D 1H-13C HMQC`, `3D C-C-H`, and `4D C-H-C-H` methyl-TROSY spectra.
#[derive(Clone, Debug)]
pub struct MethylPickerConfig {
    pub snr_hmqc: f32,
    pub shoulder_snr_hmqc: f32,
    pub nominal_lw_h_hz: f64,
    pub nominal_lw_c_hz: f64,
    pub lorentz_frac: f64,
    pub snr_noe_primary: f32,
    pub snr_noe_reciprocal: f32,
    pub diag_tol_c_ppm: f64,
    pub diag_tol_h_ppm: f64,
    pub match_tol_c_ppm: f64,
    pub match_tol_h_ppm: f64,
}

impl Default for MethylPickerConfig {
    fn default() -> Self {
        Self {
            snr_hmqc: 6.0,
            shoulder_snr_hmqc: 6.5,
            nominal_lw_h_hz: 25.0,
            nominal_lw_c_hz: 18.0,
            lorentz_frac: 0.15,
            snr_noe_primary: 5.3,
            snr_noe_reciprocal: 3.7,
            diag_tol_c_ppm: 0.14,
            diag_tol_h_ppm: 0.035,
            match_tol_c_ppm: 0.12,
            match_tol_h_ppm: 0.04,
        }
    }
}

/// Normalized 1D convex pseudo-Voigt profile with exact `FWHM = fwhm` (`psi(+-0.5 * fwhm) == 0.5`).
#[inline]
pub fn pseudo_voigt_1d(delta: f64, fwhm: f64, lorentz_frac: f64) -> f64 {
    let u = delta / fwhm.max(1e-9);
    let u2 = u * u;
    let g = (-4.0 * std::f64::consts::LN_2 * u2).exp();
    let l = 1.0 / (1.0 + 4.0 * u2);
    (1.0 - lorentz_frac) * g + lorentz_frac * l
}

#[inline]
fn subvoxel_3pt(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.max(0.0)) as f64;
    let b = (y0.max(0.0)) as f64;
    let c = (yp1.max(0.0)) as f64;
    if a > 0.0 && b > a && b > c && c > 0.0 {
        let la = a.ln();
        let lb = b.ln();
        let lc = c.ln();
        let den = la - 2.0 * lb + lc;
        if den.abs() > 1e-12 {
            return (0.5 * (la - lc) / den).clamp(-0.5, 0.5);
        }
    }
    let den = a - 2.0 * b + c;
    if den.abs() > 1e-12 {
        (0.5 * (a - c) / den).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Simulate a volumetric `2D 1H-13C HMQC` spectrum at high-MW methyl-TROSY linewidths (`25 Hz / 18 Hz`).
pub fn simulate_methyl_hmqc_2d(true_roots_hc: &[(f64, f64)], cfg: &MethylSimConfig) -> Spectrum2D {
    let nx = cfg.hmqc_nx.max(64);
    let ny = cfg.hmqc_ny.max(64);
    let axis_x = AxisPpm::from_bounds("1H", nx, 2.80, -1.40, cfg.obs_h_mhz);
    let axis_y = AxisPpm::from_bounds("13C", ny, 28.50, 8.50, cfg.obs_c_mhz());

    let fwhm_h = cfg.fwhm_h_ppm();
    let fwhm_c = cfg.fwhm_c_ppm();
    let rad_x = ((4.5 * fwhm_h / axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let rad_y = ((4.5 * fwhm_c / axis_y.step_ppm.abs()).ceil() as isize).max(3);

    let mut rng = Rng::new(cfg.seed ^ 0x4D45_5448_3244_0001);
    let noise_sd = (1.0 / cfg.snr_hmqc.max(1.0)) as f32;
    let mut data = vec![0.0f32; nx * ny];
    for v in &mut data {
        *v = rng.normal(0.0, noise_sd as f64) as f32;
    }

    for &(h0, c0) in true_roots_hc {
        // Modest log-normal dynamic range across methyls (order parameters S^2_axis ~ 0.85..1.15)
        let amp = (1.0 + rng.normal(0.0, 0.08)).clamp(0.75, 1.25);
        let cx = axis_x.ppm_to_index(h0).round() as isize;
        let cy = axis_y.ppm_to_index(c0).round() as isize;
        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min((nx - 1) as isize).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min((ny - 1) as isize).max(0) as usize;

        for iy in y0..=y1 {
            let py = pseudo_voigt_1d(
                axis_y.index_to_ppm(iy as f64) - c0,
                fwhm_c,
                cfg.lorentz_frac,
            );
            for ix in x0..=x1 {
                let px = pseudo_voigt_1d(
                    axis_x.index_to_ppm(ix as f64) - h0,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                data[iy * nx + ix] += (amp * px * py) as f32;
            }
        }
    }

    Spectrum2D {
        name: "Methyl_13C_HMQC_2D".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

/// Pick methyl roots from a `2D 1H-13C HMQC` spectrum using Pass-1 local maxima + sub-voxel
/// refinement and Pass-2 high-MW pseudo-Voigt residual deconvolution for shoulder-blended pairs.
pub fn pick_methyl_hmqc_2d(spec: &Spectrum2D, cfg: &MethylPickerConfig) -> Vec<MethylRootPeak> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 5 || ny < 5 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh1 = cfg.snr_hmqc * noise;

    let fwhm_h = cfg.nominal_lw_h_hz / spec.axis_x.obs_mhz.max(100.0);
    let fwhm_c = cfg.nominal_lw_c_hz / spec.axis_y.obs_mhz.max(25.0);

    // Compute per-column and per-row max to suppress Lorentzian/truncation sinc skirts (< 3% of parent)
    let mut col_max = vec![0.0f32; nx];
    let mut row_max = vec![0.0f32; ny];
    for (iy, rmax) in row_max.iter_mut().enumerate().take(ny) {
        for (ix, cmax) in col_max.iter_mut().enumerate().take(nx) {
            let v = spec.data[iy * nx + ix];
            if v > *cmax {
                *cmax = v;
            }
            if v > *rmax {
                *rmax = v;
            }
        }
    }

    let mut roots: Vec<MethylRootPeak> = Vec::new();

    // Pass 1: strict 3x3 2D local maxima
    for (iy, &rmax) in row_max.iter().enumerate().take(ny - 1).skip(1) {
        for (ix, &cmax) in col_max.iter().enumerate().take(nx - 1).skip(1) {
            let v0 = spec.data[iy * nx + ix];
            if v0 < thresh1 {
                continue;
            }
            if v0 < 0.032 * cmax && v0 < 0.032 * rmax {
                continue;
            }
            let mut is_max = true;
            'nb: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let vn =
                        spec.data[((iy as isize + dy) as usize) * nx + (ix as isize + dx) as usize];
                    if vn > v0 {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let ox = subvoxel_3pt(spec.data[iy * nx + ix - 1], v0, spec.data[iy * nx + ix + 1]);
            let oy = subvoxel_3pt(
                spec.data[(iy - 1) * nx + ix],
                v0,
                spec.data[(iy + 1) * nx + ix],
            );
            let h_ppm = spec.axis_x.index_to_ppm(ix as f64 + ox);
            let c_ppm = spec.axis_y.index_to_ppm(iy as f64 + oy);
            let id = roots.len();
            roots.push(MethylRootPeak {
                id,
                h_ppm,
                c_ppm,
                height: v0 as f64,
                volume: (v0 as f64) * fwhm_h * fwhm_c,
                hmcm: None,
            });
        }
    }

    // Pass 2: subtract Pass-1 fitted pseudo-Voigt profiles and rescue blended high-MW shoulders
    let mut residual = spec.data.clone();
    let rad_x = ((4.5 * fwhm_h / spec.axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let rad_y = ((4.5 * fwhm_c / spec.axis_y.step_ppm.abs()).ceil() as isize).max(3);

    for pk in &roots {
        let cx = spec.axis_x.ppm_to_index(pk.h_ppm).round() as isize;
        let cy = spec.axis_y.ppm_to_index(pk.c_ppm).round() as isize;
        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min((nx - 1) as isize).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min((ny - 1) as isize).max(0) as usize;
        for iy in y0..=y1 {
            let py = pseudo_voigt_1d(
                spec.axis_y.index_to_ppm(iy as f64) - pk.c_ppm,
                fwhm_c,
                cfg.lorentz_frac,
            );
            for ix in x0..=x1 {
                let px = pseudo_voigt_1d(
                    spec.axis_x.index_to_ppm(ix as f64) - pk.h_ppm,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                residual[iy * nx + ix] -= (pk.height * px * py) as f32;
            }
        }
    }

    let thresh2 = cfg.shoulder_snr_hmqc * noise;
    let pass1_count = roots.len();
    let mut shoulder_cands: Vec<(f32, usize, usize)> = Vec::new();
    for iy in 1..(ny - 1) {
        for ix in 1..(nx - 1) {
            let r0 = residual[iy * nx + ix];
            let raw0 = spec.data[iy * nx + ix];
            if r0 < thresh2 || raw0 < thresh1 {
                continue;
            }
            let mut is_max = true;
            'nb2: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if residual[((iy as isize + dy) as usize) * nx + (ix as isize + dx) as usize]
                        > r0
                    {
                        is_max = false;
                        break 'nb2;
                    }
                }
            }
            if is_max {
                shoulder_cands.push((r0, ix, iy));
            }
        }
    }
    shoulder_cands.sort_by(|a, b| b.0.total_cmp(&a.0));

    for (r0, ix, iy) in shoulder_cands {
        let ox = subvoxel_3pt(residual[iy * nx + ix - 1], r0, residual[iy * nx + ix + 1]);
        let oy = subvoxel_3pt(
            residual[(iy - 1) * nx + ix],
            r0,
            residual[(iy + 1) * nx + ix],
        );
        let h_ppm = spec.axis_x.index_to_ppm(ix as f64 + ox);
        let c_ppm = spec.axis_y.index_to_ppm(iy as f64 + oy);

        // Must be a genuine blended shoulder near a Pass-1 peak (0.48..1.85 FWHM), not an isolated skirt ripple
        let mut min_norm_d = f64::INFINITY;
        let mut parent_ht = 0.0f64;
        for pk in &roots[..pass1_count] {
            let dh = (h_ppm - pk.h_ppm) / fwhm_h;
            let dc = (c_ppm - pk.c_ppm) / fwhm_c;
            let nd = (dh * dh + dc * dc).sqrt();
            if nd < min_norm_d {
                min_norm_d = nd;
                parent_ht = pk.height;
            }
        }
        let too_close_any = roots.iter().any(|pk| {
            let dh = (h_ppm - pk.h_ppm) / fwhm_h;
            let dc = (c_ppm - pk.c_ppm) / fwhm_c;
            (dh * dh + dc * dc).sqrt() < 0.48
        });
        if too_close_any || !(0.48..=1.85).contains(&min_norm_d) || (r0 as f64) < 0.25 * parent_ht {
            continue;
        }

        let id = roots.len();
        roots.push(MethylRootPeak {
            id,
            h_ppm,
            c_ppm,
            height: r0 as f64,
            volume: (r0 as f64) * fwhm_h * fwhm_c,
            hmcm: None,
        });
    }

    roots
}

/// Compute damped $r_{\text{CC}}^{-6}$ dipolar cross-peak amplitude relative to a `5.0 A` reference.
#[inline]
pub fn methyl_noe_relative_amplitude(d_cc: f64, r_sat: f64) -> f64 {
    let d_eff6 = d_cc.powi(6) + r_sat.powi(6);
    let ref_eff6 = 5.0_f64.powi(6) + r_sat.powi(6);
    ref_eff6 / d_eff6
}

/// Simulate a volumetric `3D C-C-H` (`axis_x = 1H`, `axis_y = 13C` root, `axis_z = 13CM` NOE)
/// `HMQC-NOESY-HMQC` cube at `40–80 kDa` linewidths (`25 Hz / 18 Hz`).
pub fn simulate_methyl_noesy_3d(
    model: &MethylStructureModel,
    true_roots_hc: &[(f64, f64)],
    cfg: &MethylSimConfig,
) -> Spectrum3D {
    assert_eq!(model.sites.len(), true_roots_hc.len());
    let m_len = model.sites.len();
    let nx = cfg.noesy_nx.max(64);
    let ny = cfg.noesy_ny.max(64);
    let nz = cfg.noesy_nz.max(64);

    // Avoid 'N' in 13C axis labels so Sparky UCSF 5-char nucleus classification preserves (1H, 13C, 13CM)
    let axis_x = AxisPpm::from_bounds("1H", nx, 2.80, -1.40, cfg.obs_h_mhz);
    let axis_y = AxisPpm::from_bounds("13C", ny, 28.50, 8.50, cfg.obs_c_mhz());
    let axis_z = AxisPpm::from_bounds("13CM", nz, 28.50, 8.50, cfg.obs_c_mhz());

    let fwhm_h = cfg.fwhm_h_ppm().hypot(0.85 * axis_x.step_ppm.abs());
    let fwhm_cy = cfg.fwhm_c_ppm().hypot(0.85 * axis_y.step_ppm.abs());
    let fwhm_cz = cfg.fwhm_c_ppm().hypot(0.85 * axis_z.step_ppm.abs());
    let rad_x = ((3.8 * fwhm_h / axis_x.step_ppm.abs()).ceil() as isize).max(2);
    let rad_y = ((3.8 * fwhm_cy / axis_y.step_ppm.abs()).ceil() as isize).max(2);
    let rad_z = ((3.8 * fwhm_cz / axis_z.step_ppm.abs()).ceil() as isize).max(2);

    let mut rng = Rng::new(cfg.seed ^ 0x4D45_5448_3344_0002);
    let noise_sd = (1.0 / cfg.snr_noe_5a.max(1.0)) as f32;
    let mut data = vec![0.0f32; nx * ny * nz];
    for v in &mut data {
        *v = rng.normal(0.0, noise_sd as f64) as f32;
    }

    let mut add_peak_3d = |h_r: f64, c_r: f64, c_n: f64, amp: f64| {
        let cx = axis_x.ppm_to_index(h_r).round() as isize;
        let cy = axis_y.ppm_to_index(c_r).round() as isize;
        let cz = axis_z.ppm_to_index(c_n).round() as isize;
        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min((nx - 1) as isize).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min((ny - 1) as isize).max(0) as usize;
        let z0 = (cz - rad_z).max(0) as usize;
        let z1 = (cz + rad_z).min((nz - 1) as isize).max(0) as usize;

        for iz in z0..=z1 {
            let pz = pseudo_voigt_1d(
                axis_z.index_to_ppm(iz as f64) - c_n,
                fwhm_cz,
                cfg.lorentz_frac,
            );
            for iy in y0..=y1 {
                let py = pseudo_voigt_1d(
                    axis_y.index_to_ppm(iy as f64) - c_r,
                    fwhm_cy,
                    cfg.lorentz_frac,
                );
                let pyz = amp * py * pz;
                let row_off = (iz * ny + iy) * nx;
                for ix in x0..=x1 {
                    let px = pseudo_voigt_1d(
                        axis_x.index_to_ppm(ix as f64) - h_r,
                        fwhm_h,
                        cfg.lorentz_frac,
                    );
                    data[row_off + ix] += (pyz * px) as f32;
                }
            }
        }
    };

    // Diagonal auto-peaks + symmetric methyl-methyl cross-peaks for d_CC <= r_max_noe_angstrom
    for (m1, &(h1, c1)) in true_roots_hc.iter().enumerate().take(m_len) {
        add_peak_3d(h1, c1, c1, 8.5);
        for (m2, &(h2, c2)) in true_roots_hc.iter().enumerate().take(m_len).skip(m1 + 1) {
            let d = model.dist_cc(m1, m2);
            if d <= cfg.r_max_noe_angstrom {
                let base_amp = methyl_noe_relative_amplitude(d, cfg.r_sat_angstrom);
                let jitter12 = (1.0 + rng.normal(0.0, 0.10)).clamp(0.70, 1.35);
                let jitter21 = (1.0 + rng.normal(0.0, 0.10)).clamp(0.70, 1.35);
                add_peak_3d(h1, c1, c2, base_amp * jitter12);
                add_peak_3d(h2, c2, c1, base_amp * jitter21);
            }
        }
    }

    Spectrum3D {
        name: "Methyl_3D_CCH_NOESY".to_string(),
        exp: "METHYL_NOESY_3D".to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

/// Pick `3D C-C-H` (`(C_noe, C_root, H_root)`) methyl NOESY cross-peaks anchored on picked 2D HMQC
/// roots using self-diagonal lineshape subtraction, multi-pass 1D pseudo-Voigt strip deconvolution,
/// and unsupervised `(s <-> t)` reciprocal dipolar symmetry corroboration.
pub fn pick_methyl_noesy_3d(
    cube: &Spectrum3D,
    roots: &[MethylRootPeak],
    cfg: &MethylPickerConfig,
) -> Vec<MethylNoePeak> {
    let nx = cube.nx();
    let ny = cube.ny();
    let nz = cube.nz();
    let s_len = roots.len();
    if nx < 4 || ny < 4 || nz < 5 || s_len == 0 {
        return Vec::new();
    }

    let noise = estimate_noise_mad(&cube.data);
    let fwhm_hx = (cfg.nominal_lw_h_hz / cube.axis_x.obs_mhz.max(100.0))
        .hypot(0.85 * cube.axis_x.step_ppm.abs());
    let fwhm_cy = (cfg.nominal_lw_c_hz / cube.axis_y.obs_mhz.max(25.0))
        .hypot(0.85 * cube.axis_y.step_ppm.abs());
    let fwhm_cz = (cfg.nominal_lw_c_hz / cube.axis_z.obs_mhz.max(25.0))
        .hypot(0.85 * cube.axis_z.step_ppm.abs());

    // 1. Extract bilinearly interpolated, gain-normalized 1D C_noe strip per HMQC root and
    //    estimate each root's 3D self-diagonal amplitude so both self-diagonal skirts AND
    //    2D (H_root, C_root) neighbor-diagonal PSF bleed are analytically subtracted.
    let mut raw_strips: Vec<Vec<f32>> = Vec::with_capacity(s_len);
    let mut strip_sigmas: Vec<f32> = Vec::with_capacity(s_len);
    let mut diag_info: Vec<(f64, f64)> = Vec::with_capacity(s_len); // (c_diag, amp_diag)

    for rt in roots {
        let fx = cube
            .axis_x
            .ppm_to_index(rt.h_ppm)
            .clamp(0.0, (nx - 1) as f64);
        let fy = cube
            .axis_y
            .ppm_to_index(rt.c_ppm)
            .clamp(0.0, (ny - 1) as f64);
        let x0 = (fx.floor() as usize).min(nx - 1);
        let x1 = (x0 + 1).min(nx - 1);
        let y0 = (fy.floor() as usize).min(ny - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let wx = (fx - x0 as f64) as f32;
        let wy = (fy - y0 as f64) as f32;

        let gx_r = (1.0 - wx as f64)
            * pseudo_voigt_1d(
                cube.axis_x.index_to_ppm(x0 as f64) - rt.h_ppm,
                fwhm_hx,
                cfg.lorentz_frac,
            )
            + (wx as f64)
                * pseudo_voigt_1d(
                    cube.axis_x.index_to_ppm(x1 as f64) - rt.h_ppm,
                    fwhm_hx,
                    cfg.lorentz_frac,
                );
        let gy_r = (1.0 - wy as f64)
            * pseudo_voigt_1d(
                cube.axis_y.index_to_ppm(y0 as f64) - rt.c_ppm,
                fwhm_cy,
                cfg.lorentz_frac,
            )
            + (wy as f64)
                * pseudo_voigt_1d(
                    cube.axis_y.index_to_ppm(y1 as f64) - rt.c_ppm,
                    fwhm_cy,
                    cfg.lorentz_frac,
                );
        let g_root = (gx_r * gy_r).max(0.45) as f32;
        let interp_var_x = if x0 == x1 {
            1.0
        } else {
            (1.0 - wx) * (1.0 - wx) + wx * wx
        };
        let interp_var_y = if y0 == y1 {
            1.0
        } else {
            (1.0 - wy) * (1.0 - wy) + wy * wy
        };
        let sigma_s = noise * (interp_var_x * interp_var_y).sqrt() / g_root;
        let primary_s = cfg.snr_noe_primary * sigma_s;

        let mut tr = vec![0.0f32; nz];
        for (iz, tr_val) in tr.iter_mut().enumerate().take(nz) {
            let v00 = cube.get(x0, y0, iz);
            let v10 = cube.get(x1, y0, iz);
            let v01 = cube.get(x0, y1, iz);
            let v11 = cube.get(x1, y1, iz);
            let vx0 = v00 * (1.0 - wx) + v10 * wx;
            let vx1 = v01 * (1.0 - wx) + v11 * wx;
            *tr_val = (vx0 * (1.0 - wy) + vx1 * wy) / g_root;
        }

        let mut best_diag: Option<(usize, f32)> = None;
        for (iz, &v) in tr.iter().enumerate().take(nz - 1).skip(1) {
            let c_iz = cube.axis_z.index_to_ppm(iz as f64);
            if (c_iz - rt.c_ppm).abs() <= 0.16
                && v >= tr[iz - 1]
                && v >= tr[iz + 1]
                && best_diag.is_none_or(|(_, bv)| v > bv)
            {
                best_diag = Some((iz, v));
            }
        }
        let d_pair = if let Some((iz_d, vd)) = best_diag {
            if vd > primary_s {
                let oz = subvoxel_3pt(tr[iz_d - 1], vd, tr[iz_d + 1]);
                let c_diag = cube.axis_z.index_to_ppm(iz_d as f64 + oz);
                let u_off = (oz * cube.axis_z.step_ppm.abs()) / fwhm_cz.max(1e-6);
                let peak_scale =
                    pseudo_voigt_1d(u_off * fwhm_cz, fwhm_cz, cfg.lorentz_frac).max(0.7);
                (c_diag, (vd as f64) / peak_scale)
            } else {
                (rt.c_ppm, 0.0)
            }
        } else {
            (rt.c_ppm, 0.0)
        };

        raw_strips.push(tr);
        strip_sigmas.push(sigma_s);
        diag_info.push(d_pair);
    }

    let root_psf_bleed = |from_u: usize, to_s: usize| -> f64 {
        pseudo_voigt_1d(
            roots[to_s].h_ppm - roots[from_u].h_ppm,
            fwhm_hx,
            cfg.lorentz_frac,
        ) * pseudo_voigt_1d(
            roots[to_s].c_ppm - roots[from_u].c_ppm,
            fwhm_cy,
            cfg.lorentz_frac,
        )
    };

    let mut strip_comps: Vec<Vec<(f64, f32)>> = Vec::with_capacity(s_len);
    let mut diag_free_traces: Vec<Vec<f32>> = Vec::with_capacity(s_len);
    let mut res_traces: Vec<Vec<f32>> = Vec::with_capacity(s_len);
    let step_cz = cube.axis_z.step_ppm.abs();

    for (s, rt) in roots.iter().enumerate() {
        let mut tr = raw_strips[s].clone();
        let sigma_s = strip_sigmas[s];
        let recip_s = cfg.snr_noe_reciprocal * sigma_s;

        // Subtract both self-diagonal (u == s) and 2D (H_root, C_root) bleed of neighboring roots' diagonals
        for (u, &(c_du, amp_du)) in diag_info.iter().enumerate() {
            if amp_du <= 0.0 {
                continue;
            }
            let eff_amp = if u == s {
                amp_du
            } else {
                let b = root_psf_bleed(u, s);
                if amp_du * b < 0.25 * (sigma_s as f64) {
                    continue;
                }
                amp_du * b
            };
            for (iz, tv) in tr.iter_mut().enumerate().take(nz) {
                let c_z = cube.axis_z.index_to_ppm(iz as f64);
                *tv -= (eff_amp * pseudo_voigt_1d(c_z - c_du, fwhm_cz, cfg.lorentz_frac)) as f32;
            }
        }

        let diag_free = tr.clone();
        let mut res = tr;
        let mut comps: Vec<(f64, f32)> = Vec::new();

        for pass in 0..4 {
            let pass_thr = if pass == 0 { recip_s } else { recip_s * 1.15 };
            let mut cands: Vec<(f32, usize)> = Vec::new();
            for iz in 1..(nz - 1) {
                let r0 = res[iz];
                if r0 < pass_thr || diag_free[iz] < recip_s || r0 < res[iz - 1] || r0 < res[iz + 1]
                {
                    continue;
                }
                let oz = subvoxel_3pt(res[iz - 1], r0, res[iz + 1]);
                let c_pk = cube.axis_z.index_to_ppm(iz as f64 + oz);
                if (c_pk - rt.c_ppm).abs() <= cfg.diag_tol_c_ppm {
                    continue;
                }
                if comps
                    .iter()
                    .any(|&(ec, _)| (ec - c_pk).abs() < 0.38 * fwhm_cz)
                {
                    continue;
                }
                cands.push((r0, iz));
            }
            if cands.is_empty() {
                break;
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut accepted_in_pass = Vec::new();
            for (r0, iz) in cands {
                let oz = subvoxel_3pt(res[iz - 1], r0, res[iz + 1]);
                let c_pk = cube.axis_z.index_to_ppm(iz as f64 + oz);
                if accepted_in_pass
                    .iter()
                    .chain(comps.iter())
                    .any(|&(ec, _): &(f64, f32)| (ec - c_pk).abs() < 0.40 * fwhm_cz)
                {
                    continue;
                }
                let sub_scale = pseudo_voigt_1d(oz * step_cz, fwhm_cz, cfg.lorentz_frac).max(0.72);
                let amp_true = ((r0 as f64) / sub_scale) as f32;
                accepted_in_pass.push((c_pk, amp_true));
            }
            for &(c_pk, amp_true) in &accepted_in_pass {
                for (iz, rv) in res.iter_mut().enumerate().take(nz) {
                    let c_z = cube.axis_z.index_to_ppm(iz as f64);
                    *rv -= ((amp_true as f64)
                        * pseudo_voigt_1d(c_z - c_pk, fwhm_cz, cfg.lorentz_frac))
                        as f32;
                }
                comps.push((c_pk, amp_true));
            }
        }

        strip_comps.push(comps);
        diag_free_traces.push(diag_free);
        res_traces.push(res);
    }

    let diag_free_at_ppm = |s: usize, c_ppm: f64| -> f32 {
        let fz = cube.axis_z.ppm_to_index(c_ppm).clamp(0.0, (nz - 1) as f64);
        let z0 = (fz.floor() as usize).min(nz - 1);
        let z1 = (z0 + 1).min(nz - 1);
        let wz = (fz - z0 as f64) as f32;
        diag_free_traces[s][z0] * (1.0 - wz) + diag_free_traces[s][z1] * wz
    };

    let res_at_ppm = |s: usize, c_ppm: f64| -> f32 {
        let fz = cube.axis_z.ppm_to_index(c_ppm).clamp(0.0, (nz - 1) as f64);
        let z0 = (fz.floor() as usize).min(nz - 1);
        let z1 = (z0 + 1).min(nz - 1);
        let wz = (fz - z0 as f64) as f32;
        let raw = res_traces[s][z0] * (1.0 - wz) + res_traces[s][z1] * wz;
        let gz = (1.0 - wz as f64)
            * pseudo_voigt_1d((wz as f64) * step_cz, fwhm_cz, cfg.lorentz_frac)
            + (wz as f64) * pseudo_voigt_1d((1.0 - wz as f64) * step_cz, fwhm_cz, cfg.lorentz_frac);
        (raw as f64 / gz.max(0.70)) as f32
    };

    // Suppress cross-peaks on `s` that are purely 2D (H_root, C_root) PSF bleed from a nearby HMQC root `u`:
    // by linear superposition, reject iff `amp_s - b(u->s)*amp_u < 0.90 * recip_sigma`
    let is_cross_root_bleed = |s: usize, c_target: f64, amp_s: f32, partner_t: usize| -> bool {
        let min_net = 0.90 * cfg.snr_noe_reciprocal * strip_sigmas[s];
        for u in 0..s_len {
            if u == s || u == partner_t {
                continue;
            }
            let b = root_psf_bleed(u, s);
            if b > 0.03 {
                let amp_u = diag_free_at_ppm(u, c_target);
                if amp_u > 1.25 * amp_s && (amp_s - (b as f32) * amp_u) < min_net {
                    return true;
                }
            }
        }
        false
    };

    let best_comp_near = |s: usize, target_c: f64, tol_c: f64| -> Option<(f64, f32)> {
        strip_comps[s]
            .iter()
            .copied()
            .filter(|&(c_pk, _)| (c_pk - target_c).abs() <= tol_c)
            .min_by(|a, b| (a.0 - target_c).abs().total_cmp(&(b.0 - target_c).abs()))
    };

    let match_tol = cfg.match_tol_c_ppm.min(0.105);
    let mut explained_comp: Vec<Vec<bool>> =
        strip_comps.iter().map(|c| vec![false; c.len()]).collect();
    // Per-strip emitted 3D cross-peaks `(c_noe, intensity, donor_idx_or_usize_max_if_ambiguous, bidir)`
    // deduplicated by physical `(recipient_strip, c_noe)` coordinate so C_noe-degenerate donors
    // never emit duplicate peaks or hallucinated 4D H_noe coordinates in 3D C-C-H spectra.
    type Emitted3DPeak = (f64, f32, Option<usize>, bool);
    let mut strip_emitted: Vec<Vec<Emitted3DPeak>> = (0..s_len).map(|_| Vec::new()).collect();

    let mut record_3d_peak =
        |r_idx: usize, c_val: f64, amp_val: f32, donor_idx: usize, is_bidir: bool| {
            for entry in &mut strip_emitted[r_idx] {
                if (entry.0 - c_val).abs() <= 0.048 {
                    if amp_val > entry.1 {
                        entry.0 = c_val;
                        entry.1 = amp_val;
                    }
                    if entry.2 != Some(donor_idx) {
                        entry.2 = None;
                    }
                    entry.3 |= is_bidir;
                    return;
                }
            }
            strip_emitted[r_idx].push((c_val, amp_val, Some(donor_idx), is_bidir));
        };

    // 2. Pairwise reciprocal corroboration with dipolar symmetry consistency check
    for s in 0..s_len {
        for t in (s + 1)..s_len {
            if (roots[s].c_ppm - roots[t].c_ppm).abs() <= cfg.diag_tol_c_ppm {
                continue;
            }
            let sig_pair = 0.5 * (strip_sigmas[s] + strip_sigmas[t]);
            let prim_pair = cfg.snr_noe_primary * sig_pair;
            let rec_pair = cfg.snr_noe_reciprocal * sig_pair;

            let comp_st = best_comp_near(s, roots[t].c_ppm, match_tol);
            let comp_ts = best_comp_near(t, roots[s].c_ppm, match_tol);

            let pair_match = match (comp_st, comp_ts) {
                (Some((c_st, a_st)), Some((c_ts, a_ts))) => {
                    let hi = a_st.max(a_ts);
                    let lo = a_st.min(a_ts).max(1e-9);
                    if hi >= prim_pair * 0.97 && lo >= rec_pair * 1.00 && hi <= 3.05 * lo {
                        Some((c_st, a_st, c_ts, a_ts))
                    } else {
                        None
                    }
                }
                (Some((c_st, a_st)), None) => {
                    let amp_ts = res_at_ppm(t, roots[s].c_ppm);
                    let hi = a_st.max(amp_ts);
                    let lo = a_st.min(amp_ts).max(1e-9);
                    if a_st >= prim_pair * 1.12 && amp_ts >= rec_pair * 1.40 && hi <= 2.55 * lo {
                        Some((c_st, a_st, roots[s].c_ppm, amp_ts))
                    } else {
                        None
                    }
                }
                (None, Some((c_ts, a_ts))) => {
                    let amp_st = res_at_ppm(s, roots[t].c_ppm);
                    let hi = a_ts.max(amp_st);
                    let lo = a_ts.min(amp_st).max(1e-9);
                    if a_ts >= prim_pair * 1.12 && amp_st >= rec_pair * 1.40 && hi <= 2.55 * lo {
                        Some((roots[t].c_ppm, amp_st, c_ts, a_ts))
                    } else {
                        None
                    }
                }
                (None, None) => None,
            };

            if let Some((c_st, a_st, c_ts, a_ts)) = pair_match {
                if is_cross_root_bleed(s, roots[t].c_ppm, a_st, t)
                    || is_cross_root_bleed(t, roots[s].c_ppm, a_ts, s)
                {
                    continue;
                }
                for (k, &(cpk, _)) in strip_comps[s].iter().enumerate() {
                    if (cpk - c_st).abs() <= match_tol {
                        explained_comp[s][k] = true;
                    }
                }
                for (k, &(cpk, _)) in strip_comps[t].iter().enumerate() {
                    if (cpk - c_ts).abs() <= match_tol {
                        explained_comp[t][k] = true;
                    }
                }
                record_3d_peak(s, c_st, a_st, t, true);
                record_3d_peak(t, c_ts, a_ts, s, true);
            }
        }
    }

    let mut out_peaks: Vec<MethylNoePeak> = Vec::new();
    for s in 0..s_len {
        for &(c_noe, amp, donor_opt, bidir) in &strip_emitted[s] {
            out_peaks.push(MethylNoePeak {
                c_noe_ppm: c_noe,
                c_root_ppm: roots[s].c_ppm,
                h_root_ppm: roots[s].h_ppm,
                h_noe_ppm: None,
                intensity: amp as f64,
                root_idx: Some(s),
                donor_root_idx: donor_opt,
                bidirectional: bidir,
            });
        }
    }

    // 3. Retain only extreme (>= 45 sigma) unexplained primary components (e.g. if a geminal
    //    partner root was completely overlapped in 2D HMQC), never residual subtraction wings
    for s in 0..s_len {
        let solo_thr = 45.0 * strip_sigmas[s];
        for (k, &(c_pk, a_pk)) in strip_comps[s].iter().enumerate() {
            if k > 1 || explained_comp[s][k] || a_pk < solo_thr {
                continue;
            }
            if (c_pk - roots[s].c_ppm).abs() <= cfg.diag_tol_c_ppm * 1.8 {
                continue;
            }
            if strip_emitted[s]
                .iter()
                .any(|&(ce, _, _, _)| (ce - c_pk).abs() <= 0.06)
            {
                continue;
            }
            out_peaks.push(MethylNoePeak {
                c_noe_ppm: c_pk,
                c_root_ppm: roots[s].c_ppm,
                h_root_ppm: roots[s].h_ppm,
                h_noe_ppm: None,
                intensity: a_pk as f64,
                root_idx: Some(s),
                donor_root_idx: None,
                bidirectional: false,
            });
        }
    }

    out_peaks
}

/// Simulate a volumetric `4D C-H-C-H` `HMQC-NOESY-HMQC` cube (`n_grid^4`) at `40–80 kDa` linewidths,
/// incorporating Nyquist voxel acquisition broadening on coarse 4D grids.
pub fn simulate_methyl_noesy_4d(
    model: &MethylStructureModel,
    true_roots_hc: &[(f64, f64)],
    cfg: &MethylSimConfig,
    n_h_grid: usize,
    n_c_grid: usize,
) -> Spectrum4DMethyl {
    assert_eq!(model.sites.len(), true_roots_hc.len());
    let m_len = model.sites.len();
    let nh = n_h_grid.max(32);
    let nc = n_c_grid.max(32);

    let axis_h_root = AxisPpm::from_bounds("1H", nh, 2.80, -1.40, cfg.obs_h_mhz);
    let axis_c_root = AxisPpm::from_bounds("13C", nc, 28.50, 8.50, cfg.obs_c_mhz());
    let axis_h_noe = AxisPpm::from_bounds("1HM", nh, 2.80, -1.40, cfg.obs_h_mhz);
    let axis_c_noe = AxisPpm::from_bounds("13CM", nc, 28.50, 8.50, cfg.obs_c_mhz());

    let fwhm_h = cfg.fwhm_h_ppm().hypot(0.90 * axis_h_root.step_ppm.abs());
    let fwhm_c = cfg.fwhm_c_ppm().hypot(0.90 * axis_c_root.step_ppm.abs());
    let rad_h = ((3.2 * fwhm_h / axis_h_root.step_ppm.abs()).ceil() as isize).max(2);
    let rad_c = ((3.2 * fwhm_c / axis_c_root.step_ppm.abs()).ceil() as isize).max(2);

    let mut rng = Rng::new(cfg.seed ^ 0x4D45_5448_3444_0004);
    let noise_sd = (1.0 / cfg.snr_noe_5a.max(1.0)) as f32;
    let total_vox = nh * nc * nh * nc;
    let mut data = vec![0.0f32; total_vox];
    for v in &mut data {
        *v = rng.normal(0.0, noise_sd as f64) as f32;
    }

    let mut add_peak_4d = |h_r: f64, c_r: f64, h_n: f64, c_n: f64, amp: f64| {
        let ihr0 = (axis_h_root.ppm_to_index(h_r).round() as isize - rad_h).max(0) as usize;
        let ihr1 = (axis_h_root.ppm_to_index(h_r).round() as isize + rad_h)
            .min((nh - 1) as isize)
            .max(0) as usize;
        let icr0 = (axis_c_root.ppm_to_index(c_r).round() as isize - rad_c).max(0) as usize;
        let icr1 = (axis_c_root.ppm_to_index(c_r).round() as isize + rad_c)
            .min((nc - 1) as isize)
            .max(0) as usize;
        let ihn0 = (axis_h_noe.ppm_to_index(h_n).round() as isize - rad_h).max(0) as usize;
        let ihn1 = (axis_h_noe.ppm_to_index(h_n).round() as isize + rad_h)
            .min((nh - 1) as isize)
            .max(0) as usize;
        let icn0 = (axis_c_noe.ppm_to_index(c_n).round() as isize - rad_c).max(0) as usize;
        let icn1 = (axis_c_noe.ppm_to_index(c_n).round() as isize + rad_c)
            .min((nc - 1) as isize)
            .max(0) as usize;

        for icn in icn0..=icn1 {
            let pcn = pseudo_voigt_1d(
                axis_c_noe.index_to_ppm(icn as f64) - c_n,
                fwhm_c,
                cfg.lorentz_frac,
            );
            for ihn in ihn0..=ihn1 {
                let phn = pseudo_voigt_1d(
                    axis_h_noe.index_to_ppm(ihn as f64) - h_n,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                let p_noe = amp * pcn * phn;
                for icr in icr0..=icr1 {
                    let pcr = pseudo_voigt_1d(
                        axis_c_root.index_to_ppm(icr as f64) - c_r,
                        fwhm_c,
                        cfg.lorentz_frac,
                    );
                    let base = ((icn * nh + ihn) * nc + icr) * nh;
                    for ihr in ihr0..=ihr1 {
                        let phr = pseudo_voigt_1d(
                            axis_h_root.index_to_ppm(ihr as f64) - h_r,
                            fwhm_h,
                            cfg.lorentz_frac,
                        );
                        data[base + ihr] += (p_noe * pcr * phr) as f32;
                    }
                }
            }
        }
    };

    for (m1, &(h1, c1)) in true_roots_hc.iter().enumerate().take(m_len) {
        add_peak_4d(h1, c1, h1, c1, 8.5);
        for (m2, &(h2, c2)) in true_roots_hc.iter().enumerate().take(m_len).skip(m1 + 1) {
            let d = model.dist_cc(m1, m2);
            if d <= cfg.r_max_noe_angstrom {
                let base_amp = methyl_noe_relative_amplitude(d, cfg.r_sat_angstrom);
                add_peak_4d(h1, c1, h2, c2, base_amp);
                add_peak_4d(h2, c2, h1, c1, base_amp);
            }
        }
    }

    Spectrum4DMethyl {
        name: "Methyl_4D_CHCH_NOESY".to_string(),
        axis_h_root,
        axis_c_root,
        axis_h_noe,
        axis_c_noe,
        data,
    }
}

/// Pick cross-peaks from a `4D C-H-C-H` `HMQC-NOESY-HMQC` cube anchored at picked 2D HMQC roots
/// via bilinear 2D `(H_noe, C_noe)` plane extraction, analytically un-attenuated 2D self-diagonal
/// subtraction, 4-pass 2D pseudo-Voigt CLEAN deconvolution, and reciprocal corroboration.
pub fn pick_methyl_noesy_4d(
    cube: &Spectrum4DMethyl,
    roots: &[MethylRootPeak],
    cfg: &MethylPickerConfig,
) -> Vec<MethylNoePeak> {
    let nh = cube.axis_h_root.size;
    let nc = cube.axis_c_root.size;
    let s_len = roots.len();
    if nh < 5 || nc < 5 || s_len == 0 {
        return Vec::new();
    }

    let noise = estimate_noise_mad(&cube.data);
    let step_h = cube.axis_h_noe.step_ppm.abs();
    let step_c = cube.axis_c_noe.step_ppm.abs();
    let fwhm_h = (cfg.nominal_lw_h_hz / cube.axis_h_noe.obs_mhz.max(100.0)).hypot(0.90 * step_h);
    let fwhm_c = (cfg.nominal_lw_c_hz / cube.axis_c_noe.obs_mhz.max(25.0)).hypot(0.90 * step_c);
    let rad_h = ((3.2 * fwhm_h / step_h).ceil() as isize).max(2);
    let rad_c = ((3.2 * fwhm_c / step_c).ceil() as isize).max(2);

    let mut raw_planes: Vec<Vec<f32>> = Vec::with_capacity(s_len);
    let mut plane_sigmas: Vec<f32> = Vec::with_capacity(s_len);
    let mut root_wx_all: Vec<Vec<(usize, f32)>> = Vec::with_capacity(s_len);
    let mut root_wy_all: Vec<Vec<(usize, f32)>> = Vec::with_capacity(s_len);
    let mut root_w2_all: Vec<f64> = Vec::with_capacity(s_len);

    // Cramér–Rao 3x3 pseudo-Voigt MVUE matched-filter 2D plane extraction at each root `(H_s, C_s)`
    for rt in roots {
        let cx_r = cube.axis_h_root.ppm_to_index(rt.h_ppm).round() as isize;
        let cy_r = cube.axis_c_root.ppm_to_index(rt.c_ppm).round() as isize;

        let mut wx_list: Vec<(usize, f32)> = Vec::with_capacity(3);
        let mut sum_wx2 = 0.0f64;
        for dx in -1isize..=1 {
            let ix = cx_r + dx;
            if ix >= 0 && (ix as usize) < nh {
                let w = pseudo_voigt_1d(
                    cube.axis_h_root.index_to_ppm(ix as f64) - rt.h_ppm,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                sum_wx2 += w * w;
                wx_list.push((ix as usize, w as f32));
            }
        }

        let mut wy_list: Vec<(usize, f32)> = Vec::with_capacity(3);
        let mut sum_wy2 = 0.0f64;
        for dy in -1isize..=1 {
            let iy = cy_r + dy;
            if iy >= 0 && (iy as usize) < nc {
                let w = pseudo_voigt_1d(
                    cube.axis_c_root.index_to_ppm(iy as f64) - rt.c_ppm,
                    fwhm_c,
                    cfg.lorentz_frac,
                );
                sum_wy2 += w * w;
                wy_list.push((iy as usize, w as f32));
            }
        }

        let w2_root = (sum_wx2 * sum_wy2).max(1e-6);
        let inv_w2 = (1.0 / w2_root) as f32;
        let sigma_s = noise / (w2_root.sqrt() as f32);

        let mut pl = vec![0.0f32; nc * nh];
        for icn in 0..nc {
            for ihn in 0..nh {
                let mut acc = 0.0f32;
                for &(iy, wy) in &wy_list {
                    for &(ix, wx) in &wx_list {
                        acc += (wx * wy) * cube.get(ix, iy, ihn, icn);
                    }
                }
                pl[icn * nh + ihn] = acc * inv_w2;
            }
        }

        raw_planes.push(pl);
        plane_sigmas.push(sigma_s);
        root_wx_all.push(wx_list);
        root_wy_all.push(wy_list);
        root_w2_all.push(w2_root);
    }

    // Exact linear-functional coupling `b_root(u -> s)` of root `u`'s 2D plane into `raw_planes[s]`
    let mut root_bleed = vec![0.0f64; s_len * s_len];
    for s in 0..s_len {
        for u in 0..s_len {
            if u == s {
                root_bleed[u * s_len + s] = 1.0;
                continue;
            }
            let mut bx = 0.0f64;
            for &(ix, wx) in &root_wx_all[s] {
                bx += (wx as f64)
                    * pseudo_voigt_1d(
                        cube.axis_h_root.index_to_ppm(ix as f64) - roots[u].h_ppm,
                        fwhm_h,
                        cfg.lorentz_frac,
                    );
            }
            let mut by = 0.0f64;
            for &(iy, wy) in &root_wy_all[s] {
                by += (wy as f64)
                    * pseudo_voigt_1d(
                        cube.axis_c_root.index_to_ppm(iy as f64) - roots[u].c_ppm,
                        fwhm_c,
                        cfg.lorentz_frac,
                    );
            }
            root_bleed[u * s_len + s] = (bx * by) / root_w2_all[s];
        }
    }

    // Cramér–Rao 3x3 MVUE matched filter on a 2D `(H_noe, C_noe)` plane returning `(amp_hat, sigma_mf)`
    let sample_plane_mf = |plane: &[f32], sigma_base: f32, h_n: f64, c_n: f64| -> (f32, f32) {
        let cx = cube.axis_h_noe.ppm_to_index(h_n).round() as isize;
        let cy = cube.axis_c_noe.ppm_to_index(c_n).round() as isize;
        let mut wx_n = [(0usize, 0.0f32); 3];
        let mut nx_n = 0usize;
        let mut sx2 = 0.0f64;
        for dx in -1isize..=1 {
            let ix = cx + dx;
            if ix >= 0 && (ix as usize) < nh {
                let w = pseudo_voigt_1d(
                    cube.axis_h_noe.index_to_ppm(ix as f64) - h_n,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                sx2 += w * w;
                wx_n[nx_n] = (ix as usize, w as f32);
                nx_n += 1;
            }
        }
        let mut wy_n = [(0usize, 0.0f32); 3];
        let mut ny_n = 0usize;
        let mut sy2 = 0.0f64;
        for dy in -1isize..=1 {
            let iy = cy + dy;
            if iy >= 0 && (iy as usize) < nc {
                let w = pseudo_voigt_1d(
                    cube.axis_c_noe.index_to_ppm(iy as f64) - c_n,
                    fwhm_c,
                    cfg.lorentz_frac,
                );
                sy2 += w * w;
                wy_n[ny_n] = (iy as usize, w as f32);
                ny_n += 1;
            }
        }
        let w2 = (sx2 * sy2).max(1e-6) as f32;
        let mut acc = 0.0f32;
        for &(iy, wy) in &wy_n[..ny_n] {
            let row = iy * nh;
            for &(ix, wx) in &wx_n[..nx_n] {
                acc += (wx * wy) * plane[row + ix];
            }
        }
        (acc / w2, sigma_base / w2.sqrt())
    };

    // Jointly solve un-attenuated 4D self-diagonal amplitudes `diag_amps_4d` (4 Gauss–Seidel passes)
    let mut diag_obs = vec![0.0f64; s_len];
    for (s, rt) in roots.iter().enumerate() {
        let (amp_d, sig_d) = sample_plane_mf(&raw_planes[s], plane_sigmas[s], rt.h_ppm, rt.c_ppm);
        if amp_d > cfg.snr_noe_primary * sig_d {
            diag_obs[s] = amp_d as f64;
        }
    }
    let mut diag_amps_4d = diag_obs.clone();
    for _ in 0..4 {
        for s in 0..s_len {
            if diag_obs[s] <= 0.0 {
                diag_amps_4d[s] = 0.0;
                continue;
            }
            let mut rhs = diag_obs[s];
            for u in 0..s_len {
                if u != s && diag_amps_4d[u] > 0.0 {
                    let b_r = root_bleed[u * s_len + s];
                    if b_r > 1e-3 {
                        // On symmetric 4D grids `b_noe(u -> s) == b_root(u -> s)`
                        rhs -= b_r * b_r * diag_amps_4d[u];
                    }
                }
            }
            diag_amps_4d[s] = rhs.max(0.0);
        }
    }

    let mut clean_planes: Vec<Vec<f32>> = Vec::with_capacity(s_len);
    for s in 0..s_len {
        let mut pl = raw_planes[s].clone();
        let sigma_s = plane_sigmas[s];

        // Subtract both self-diagonal (u == s) and exact matched-filter 2D bleed of neighboring roots' 4D diagonals
        for (u, &amp_du) in diag_amps_4d.iter().enumerate() {
            if amp_du <= 0.0 {
                continue;
            }
            let eff_amp = if u == s {
                amp_du
            } else {
                let b = root_bleed[u * s_len + s];
                if amp_du * b < 0.15 * (sigma_s as f64) {
                    continue;
                }
                amp_du * b
            };
            let cx = cube.axis_h_noe.ppm_to_index(roots[u].h_ppm).round() as isize;
            let cy = cube.axis_c_noe.ppm_to_index(roots[u].c_ppm).round() as isize;
            let ix0 = (cx - rad_h).max(0) as usize;
            let ix1 = (cx + rad_h).min((nh - 1) as isize).max(0) as usize;
            let iy0 = (cy - rad_c).max(0) as usize;
            let iy1 = (cy + rad_c).min((nc - 1) as isize).max(0) as usize;
            for icn in iy0..=iy1 {
                let py = pseudo_voigt_1d(
                    cube.axis_c_noe.index_to_ppm(icn as f64) - roots[u].c_ppm,
                    fwhm_c,
                    cfg.lorentz_frac,
                );
                for ihn in ix0..=ix1 {
                    let px = pseudo_voigt_1d(
                        cube.axis_h_noe.index_to_ppm(ihn as f64) - roots[u].h_ppm,
                        fwhm_h,
                        cfg.lorentz_frac,
                    );
                    pl[icn * nh + ihn] -= (eff_amp * px * py) as f32;
                }
            }
        }
        clean_planes.push(pl);
    }

    // Joint 4D Kronecker Cramér–Rao matched-filter observation matrix `m_obs[s, t]`,
    // exact noise standard deviation `sig_mat[s, t]`, and sparse PSF coupling `(root_nbs, noe_nbs)`.
    let mut m_obs = vec![0.0f32; s_len * s_len];
    let mut sig_mat = vec![1.0f32; s_len * s_len];
    let mut root_nbs: Vec<Vec<(usize, f32)>> = vec![Vec::new(); s_len];
    let mut noe_nbs: Vec<Vec<(usize, f32)>> = vec![Vec::new(); s_len];

    for s in 0..s_len {
        root_nbs[s].push((s, 1.0));
        for u in 0..s_len {
            if u != s {
                let b = root_bleed[u * s_len + s] as f32;
                if b > 0.015 {
                    root_nbs[s].push((u, b));
                }
            }
        }
    }

    for t in 0..s_len {
        let h_t = roots[t].h_ppm;
        let c_t = roots[t].c_ppm;
        let cx = cube.axis_h_noe.ppm_to_index(h_t).round() as isize;
        let cy = cube.axis_c_noe.ppm_to_index(c_t).round() as isize;
        let mut wx_n = [(0usize, 0.0f32); 3];
        let mut nx_n = 0usize;
        let mut sx2 = 0.0f64;
        for dx in -1isize..=1 {
            let ix = cx + dx;
            if ix >= 0 && (ix as usize) < nh {
                let w = pseudo_voigt_1d(
                    cube.axis_h_noe.index_to_ppm(ix as f64) - h_t,
                    fwhm_h,
                    cfg.lorentz_frac,
                );
                sx2 += w * w;
                wx_n[nx_n] = (ix as usize, w as f32);
                nx_n += 1;
            }
        }
        let mut wy_n = [(0usize, 0.0f32); 3];
        let mut ny_n = 0usize;
        let mut sy2 = 0.0f64;
        for dy in -1isize..=1 {
            let iy = cy + dy;
            if iy >= 0 && (iy as usize) < nc {
                let w = pseudo_voigt_1d(
                    cube.axis_c_noe.index_to_ppm(iy as f64) - c_t,
                    fwhm_c,
                    cfg.lorentz_frac,
                );
                sy2 += w * w;
                wy_n[ny_n] = (iy as usize, w as f32);
                ny_n += 1;
            }
        }
        let w2 = (sx2 * sy2).max(1e-6);
        let inv_w2 = (1.0 / w2) as f32;
        let inv_sqrt_w2 = (1.0 / w2.sqrt()) as f32;

        noe_nbs[t].push((t, 1.0));
        for (v, rv) in roots.iter().enumerate() {
            if v == t {
                continue;
            }
            let mut bx = 0.0f64;
            for &(ix, wx) in &wx_n[..nx_n] {
                bx += (wx as f64)
                    * pseudo_voigt_1d(
                        cube.axis_h_noe.index_to_ppm(ix as f64) - rv.h_ppm,
                        fwhm_h,
                        cfg.lorentz_frac,
                    );
            }
            let mut by = 0.0f64;
            for &(iy, wy) in &wy_n[..ny_n] {
                by += (wy as f64)
                    * pseudo_voigt_1d(
                        cube.axis_c_noe.index_to_ppm(iy as f64) - rv.c_ppm,
                        fwhm_c,
                        cfg.lorentz_frac,
                    );
            }
            let b_vt = ((bx * by) / w2) as f32;
            if b_vt > 0.015 {
                noe_nbs[t].push((v, b_vt));
            }
        }

        for s in 0..s_len {
            sig_mat[s * s_len + t] = plane_sigmas[s] * inv_sqrt_w2;
            if s == t {
                continue;
            }
            let plane = &clean_planes[s];
            let mut acc = 0.0f32;
            for &(iy, wy) in &wy_n[..ny_n] {
                let row = iy * nh;
                for &(ix, wx) in &wx_n[..nx_n] {
                    acc += (wx * wy) * plane[row + ix];
                }
            }
            m_obs[s * s_len + t] = acc * inv_w2;
        }
    }

    // Solve `M = (R (x) N) X` via 12 damped non-negative Gauss–Seidel sweeps over active cross-peaks
    let mut x_est: Vec<f32> = m_obs.iter().map(|&v| v.max(0.0)).collect();
    for s in 0..s_len {
        x_est[s * s_len + s] = 0.0;
    }

    let compute_interf = |s: usize, t: usize, x_cur: &[f32]| -> f32 {
        let mut interf = 0.0f32;
        for &(u, r_us) in &root_nbs[s] {
            for &(v, n_vt) in &noe_nbs[t] {
                if u == v || (u == s && v == t) {
                    continue;
                }
                let x_uv = x_cur[u * s_len + v];
                if x_uv > 2.0 * sig_mat[u * s_len + v] {
                    interf += r_us * n_vt * x_uv;
                }
            }
        }
        interf
    };

    for _ in 0..12 {
        for s in 0..s_len {
            for t in 0..s_len {
                if s == t {
                    continue;
                }
                let idx = s * s_len + t;
                let target = (m_obs[idx] - compute_interf(s, t, &x_est)).max(0.0);
                x_est[idx] = 0.30 * x_est[idx] + 0.70 * target;
            }
        }
    }

    let mut out = Vec::new();
    for s in 0..s_len {
        for t in (s + 1)..s_len {
            if (roots[s].c_ppm - roots[t].c_ppm).abs() <= cfg.diag_tol_c_ppm
                && (roots[s].h_ppm - roots[t].h_ppm).abs() <= cfg.diag_tol_h_ppm
            {
                continue;
            }
            let amp_st = m_obs[s * s_len + t] - compute_interf(s, t, &x_est);
            let amp_ts = m_obs[t * s_len + s] - compute_interf(t, s, &x_est);
            let sig_st = sig_mat[s * s_len + t].max(1e-9);
            let sig_ts = sig_mat[t * s_len + s].max(1e-9);
            let z_st = amp_st / sig_st;
            let z_ts = amp_ts / sig_ts;
            let hi = amp_st.max(amp_ts);
            let lo = amp_st.min(amp_ts).max(1e-9);

            if z_st >= 3.05
                && z_ts >= 3.05
                && z_st.hypot(z_ts) >= 5.05
                && m_obs[s * s_len + t] >= 3.05 * sig_st
                && m_obs[t * s_len + s] >= 3.05 * sig_ts
                && hi <= 2.85 * lo
            {
                out.push(MethylNoePeak {
                    c_noe_ppm: roots[t].c_ppm,
                    c_root_ppm: roots[s].c_ppm,
                    h_root_ppm: roots[s].h_ppm,
                    h_noe_ppm: Some(roots[t].h_ppm),
                    intensity: amp_st as f64,
                    root_idx: Some(s),
                    donor_root_idx: Some(t),
                    bidirectional: true,
                });
                out.push(MethylNoePeak {
                    c_noe_ppm: roots[s].c_ppm,
                    c_root_ppm: roots[t].c_ppm,
                    h_root_ppm: roots[t].h_ppm,
                    h_noe_ppm: Some(roots[s].h_ppm),
                    intensity: amp_ts as f64,
                    root_idx: Some(t),
                    donor_root_idx: Some(s),
                    bidirectional: true,
                });
            }
        }
    }
    out
}

// ============================================================================
// 5. TSV Readers & Writers for Methyl HMQC, NOESY & HMCM(CG)CBCA Peak Lists
// ============================================================================

pub fn write_methyl_hmqc_tsv(
    path: impl AsRef<Path>,
    roots: &[MethylRootPeak],
) -> Result<(), String> {
    let mut out = String::from("id\tH\tC\theight\tvolume\n");
    for r in roots {
        out.push_str(&format!(
            "{}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\n",
            r.id, r.h_ppm, r.c_ppm, r.height, r.volume
        ));
    }
    fs::write(path.as_ref(), out)
        .map_err(|e| format!("failed to write HMQC TSV {}: {e}", path.as_ref().display()))
}

pub fn read_methyl_hmqc_tsv(path: impl AsRef<Path>) -> Result<Vec<MethylRootPeak>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read HMQC TSV {}: {e}", path.as_ref().display()))?;
    let mut out = Vec::new();
    let mut header_cols: Option<Vec<String>> = None;

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        if header_cols.is_none()
            && cols
                .iter()
                .any(|c| c.eq_ignore_ascii_case("H") || c.eq_ignore_ascii_case("experiment"))
        {
            header_cols = Some(cols.iter().map(|s| s.to_ascii_lowercase()).collect());
            continue;
        }
        let (h_ppm, c_ppm, height, volume) = if let Some(ref hdr) = header_cols {
            let find = |names: &[&str]| {
                hdr.iter()
                    .position(|h| names.iter().any(|n| h == n))
                    .and_then(|idx| cols.get(idx).copied())
            };
            let h_s = find(&["h", "h_root", "h1"]).ok_or_else(|| {
                format!(
                    "missing H column at {}:{}",
                    path.as_ref().display(),
                    line_no + 1
                )
            })?;
            let c_s = find(&["c", "c_root", "c13"]).ok_or_else(|| {
                format!(
                    "missing C column at {}:{}",
                    path.as_ref().display(),
                    line_no + 1
                )
            })?;
            let h_v: f64 = h_s
                .parse()
                .map_err(|e| format!("invalid H at line {}: {e}", line_no + 1))?;
            let c_v: f64 = c_s
                .parse()
                .map_err(|e| format!("invalid C at line {}: {e}", line_no + 1))?;
            let ht = find(&["height", "intensity"])
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.0);
            let vol = find(&["volume"])
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(ht);
            (h_v, c_v, ht, vol)
        } else if cols.len() >= 2 {
            let h_v: f64 = cols[0]
                .parse()
                .map_err(|e| format!("invalid H at line {}: {e}", line_no + 1))?;
            let c_v: f64 = cols[1]
                .parse()
                .map_err(|e| format!("invalid C at line {}: {e}", line_no + 1))?;
            let ht = cols.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let vol = cols.get(3).and_then(|s| s.parse().ok()).unwrap_or(ht);
            (h_v, c_v, ht, vol)
        } else {
            return Err(format!(
                "expected at least 2 tab-separated columns at {}:{}",
                path.as_ref().display(),
                line_no + 1
            ));
        };

        let id = out.len();
        out.push(MethylRootPeak {
            id,
            h_ppm,
            c_ppm,
            height,
            volume,
            hmcm: None,
        });
    }
    Ok(out)
}

pub fn write_methyl_noesy_tsv(
    path: impl AsRef<Path>,
    peaks: &[MethylNoePeak],
) -> Result<(), String> {
    let mut out = String::from("C_noe\tC_root\tH_root\tH_noe\tintensity\tbidirectional\n");
    for p in peaks {
        let hn = p.h_noe_ppm.map(|v| format!("{v:.4}")).unwrap_or_default();
        out.push_str(&format!(
            "{:.3}\t{:.3}\t{:.4}\t{}\t{:.4}\t{}\n",
            p.c_noe_ppm,
            p.c_root_ppm,
            p.h_root_ppm,
            hn,
            p.intensity,
            u8::from(p.bidirectional)
        ));
    }
    fs::write(path.as_ref(), out)
        .map_err(|e| format!("failed to write NOESY TSV {}: {e}", path.as_ref().display()))
}

pub fn read_methyl_noesy_tsv(path: impl AsRef<Path>) -> Result<Vec<MethylNoePeak>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read NOESY TSV {}: {e}", path.as_ref().display()))?;
    let mut out = Vec::new();
    let mut header_cols: Option<Vec<String>> = None;

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        if header_cols.is_none()
            && cols
                .iter()
                .any(|c| c.eq_ignore_ascii_case("C_noe") || c.eq_ignore_ascii_case("H_root"))
        {
            header_cols = Some(cols.iter().map(|s| s.to_ascii_lowercase()).collect());
            continue;
        }

        if let Some(ref hdr) = header_cols {
            let find = |names: &[&str]| {
                hdr.iter()
                    .position(|h| names.iter().any(|n| h == n))
                    .and_then(|idx| cols.get(idx).copied())
            };
            let c_noe: f64 = find(&["c_noe", "c_donor"])
                .ok_or_else(|| format!("missing C_noe at line {}", line_no + 1))?
                .parse()
                .map_err(|e| format!("invalid C_noe at line {}: {e}", line_no + 1))?;
            let c_root: f64 = find(&["c_root", "c"])
                .ok_or_else(|| format!("missing C_root at line {}", line_no + 1))?
                .parse()
                .map_err(|e| format!("invalid C_root at line {}: {e}", line_no + 1))?;
            let h_root: f64 = find(&["h_root", "h"])
                .ok_or_else(|| format!("missing H_root at line {}", line_no + 1))?
                .parse()
                .map_err(|e| format!("invalid H_root at line {}: {e}", line_no + 1))?;
            let h_noe = find(&["h_noe", "h2"]).and_then(|s| {
                if s.is_empty() || s == "." {
                    None
                } else {
                    s.parse::<f64>().ok()
                }
            });
            let intensity = find(&["intensity", "volume", "height"])
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.0);
            let bidir = find(&["bidirectional", "bidir"])
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            out.push(MethylNoePeak {
                c_noe_ppm: c_noe,
                c_root_ppm: c_root,
                h_root_ppm: h_root,
                h_noe_ppm: h_noe,
                intensity,
                root_idx: None,
                donor_root_idx: None,
                bidirectional: bidir,
            });
        } else if cols.len() >= 3 {
            let c_noe: f64 = cols[0]
                .parse()
                .map_err(|e| format!("invalid C_noe at line {}: {e}", line_no + 1))?;
            let c_root: f64 = cols[1]
                .parse()
                .map_err(|e| format!("invalid C_root at line {}: {e}", line_no + 1))?;
            let h_root: f64 = cols[2]
                .parse()
                .map_err(|e| format!("invalid H_root at line {}: {e}", line_no + 1))?;
            let intensity = cols.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            out.push(MethylNoePeak {
                c_noe_ppm: c_noe,
                c_root_ppm: c_root,
                h_root_ppm: h_root,
                h_noe_ppm: None,
                intensity,
                root_idx: None,
                donor_root_idx: None,
                bidirectional: false,
            });
        } else {
            return Err(format!(
                "expected at least 3 columns (C_noe, C_root, H_root) at {}:{}",
                path.as_ref().display(),
                line_no + 1
            ));
        }
    }
    Ok(out)
}

pub fn write_hmcmcbca_tsv(
    path: impl AsRef<Path>,
    records: &[HmcmcbcaRecord],
) -> Result<(), String> {
    let mut out = String::from("H\tC_methyl\tCG\tCB\tCA\n");
    for r in records {
        let cg = r.cg_ppm.map(|v| format!("{v:.3}")).unwrap_or_default();
        let cb = r.cb_ppm.map(|v| format!("{v:.3}")).unwrap_or_default();
        let ca = r.ca_ppm.map(|v| format!("{v:.3}")).unwrap_or_default();
        out.push_str(&format!(
            "{:.4}\t{:.3}\t{}\t{}\t{}\n",
            r.h_ppm, r.c_methyl_ppm, cg, cb, ca
        ));
    }
    fs::write(path.as_ref(), out).map_err(|e| {
        format!(
            "failed to write HMCMCBCA TSV {}: {e}",
            path.as_ref().display()
        )
    })
}

pub fn read_hmcmcbca_tsv(path: impl AsRef<Path>) -> Result<Vec<HmcmcbcaRecord>, String> {
    let text = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read HMCMCBCA TSV {}: {e}",
            path.as_ref().display()
        )
    })?;
    let mut out = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        if line_no == 0 && cols.first().is_some_and(|c| c.eq_ignore_ascii_case("H")) {
            continue;
        }
        if cols.len() < 2 {
            return Err(format!(
                "expected at least H and C_methyl columns at {}:{}",
                path.as_ref().display(),
                line_no + 1
            ));
        }
        let parse_opt = |idx: usize| -> Option<f64> {
            cols.get(idx).and_then(|s| {
                if s.is_empty() || *s == "." {
                    None
                } else {
                    s.parse::<f64>().ok()
                }
            })
        };
        let h_ppm: f64 = cols[0]
            .parse()
            .map_err(|e| format!("invalid H at line {}: {e}", line_no + 1))?;
        let c_methyl_ppm: f64 = cols[1]
            .parse()
            .map_err(|e| format!("invalid C_methyl at line {}: {e}", line_no + 1))?;
        out.push(HmcmcbcaRecord {
            h_ppm,
            c_methyl_ppm,
            cg_ppm: parse_opt(2),
            cb_ppm: parse_opt(3),
            ca_ppm: parse_opt(4),
        });
    }
    Ok(out)
}

/// Attach `HMCM(CG)CBCA` records to the closest matching `MethylRootPeak` within `(tol_h, tol_c)`.
pub fn attach_hmcmcbca_records(
    roots: &mut [MethylRootPeak],
    records: &[HmcmcbcaRecord],
    tol_h_ppm: f64,
    tol_c_ppm: f64,
) {
    for rt in roots.iter_mut() {
        let mut best: Option<(f64, &HmcmcbcaRecord)> = None;
        for rec in records {
            let dh = (rt.h_ppm - rec.h_ppm) / tol_h_ppm.max(1e-6);
            let dc = (rt.c_ppm - rec.c_methyl_ppm) / tol_c_ppm.max(1e-6);
            if dh.abs() <= 1.0 && dc.abs() <= 1.0 {
                let d2 = dh * dh + dc * dc;
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, rec));
                }
            }
        }
        if let Some((_, rec)) = best {
            rt.hmcm = Some(rec.clone());
        }
    }
}
