//! Residual Dipolar Coupling (`RDC`) multi-medium I/O, exact 5-parameter traceless symmetric
//! Saupe alignment tensor SVD, Cornilescu `Q` / `Q_free` cross-validation, and variable-projection
//! analytical Cartesian / torsion-space energy & force evaluation.
//!
//! Provenance: Clean-room implementation from peer-reviewed biomolecular NMR literature:
//! - Tjandra, N. & Bax, A. (1997). "Direct measurement of distances and angles in biomolecules
//!   by NMR in a dilute liquid crystalline medium." *Science* 278(5340), 1111-1114.
//! - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (1998). "Validation of protein
//!   structure from anisotropic carbonyl chemical shifts in a dilute liquid crystalline phase."
//!   *J. Am. Chem. Soc.* 120(27), 6836-6837 (Cornilescu `Q`-factor and bicelle multi-bond RDCs).
//! - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (1999). "Order matrix
//!   analysis of residual dipolar couplings using singular value decomposition."
//!   *J. Magn. Reson.* 138(2), 334-342 (linear 5-component traceless symmetric Saupe matrix SVD).
//! - Clore, G. M., Gronenborn, A. M. & Tjandra, N. (1998). "Direct structure refinement against
//!   residual dipolar couplings in the presence of rhombicity of unknown magnitude."
//!   *J. Magn. Reson.* 131(1), 159-162.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::noesy::{v_cross, v_dot, v_norm, v_scale, v_sub, PdbChain, PdbResidue, Vec3};
use crate::rng::Rng;

/// Standard libration-adjusted `1D_NH` dipolar prefactor (Hz) at `r_eff(N-H) = 1.04 A`
/// (Ottiger & Bax, *J. Am. Chem. Soc.* 1998, 120, 12334-12341).
pub const DMAX_NH_HZ: f64 = 21655.824475221038;
/// Standard libration-adjusted `1D_CAHA` dipolar prefactor (Hz) at `r_eff(CA-HA) = 1.105 A`.
pub const DMAX_CAHA_HZ: f64 = 44782.43289695711;
/// Standard `1D_CACO` dipolar prefactor (Hz) at Engh-Huber `r(CA-C') = 1.525 A`.
pub const DMAX_CACO_HZ: f64 = 4284.749003486518;
/// Standard `1D_NC'` dipolar prefactor (Hz) at Engh-Huber `r(N-C') = 1.329 A`.
pub const DMAX_NCPRIME_HZ: f64 = 2610.0228269156532;
/// Standard two-bond `2D_HNC'` dipolar prefactor (Hz) across trans-peptide `H_i - C'_{i-1}` (`r_eff = 2.10 A`).
pub const DMAX_HNCPRIME_HZ: f64 = 6525.095452421265;

/// Maximum permitted condition number on the weighted `(n_work x 5)` Losonczi SVD design matrix.
pub const MAX_SAUPE_SVD_COND: f64 = 1.0e8;

/// Supported backbone internuclear dipolar coupling vector types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RdcBondType {
    NH,
    CAHA,
    CACO,
    NCPrime,
    HNCPrime,
}

impl RdcBondType {
    pub fn canonical_label(self) -> &'static str {
        match self {
            Self::NH => "1D_NH",
            Self::CAHA => "1D_CAHA",
            Self::CACO => "1D_CACO",
            Self::NCPrime => "1D_NC'",
            Self::HNCPrime => "1D_HNC'",
        }
    }

    pub fn default_d_max_hz(self) -> f64 {
        match self {
            Self::NH => DMAX_NH_HZ,
            Self::CAHA => DMAX_CAHA_HZ,
            Self::CACO => DMAX_CACO_HZ,
            Self::NCPrime => DMAX_NCPRIME_HZ,
            Self::HNCPrime => DMAX_HNCPRIME_HZ,
        }
    }

    pub fn default_sigma_hz(self) -> f64 {
        match self {
            Self::NH => 0.50,
            Self::CAHA => 1.00,
            Self::CACO => 0.20,
            Self::NCPrime => 0.15,
            Self::HNCPrime => 0.35,
        }
    }

    pub fn parse_label(s: &str) -> Result<Self, String> {
        let up = s.trim().to_ascii_uppercase();
        match up.as_str() {
            "1D_NH" | "NH" | "N-H" | "HN-N" => Ok(Self::NH),
            "1D_CAHA" | "CAHA" | "CA-HA" | "CA-HA2" | "CA-HA3" | "CAHA2" | "CAHA3" => {
                Ok(Self::CAHA)
            }
            "1D_CACO" | "CACO" | "CA-C" | "CA-CO" | "CAC'" => Ok(Self::CACO),
            "1D_NC'" | "NC'" | "NC" | "N-C" | "N-C'" | "NCPRIME" | "1D_NCPRIME" => {
                Ok(Self::NCPrime)
            }
            "1D_HNC'" | "2D_HNC'" | "HNC'" | "HNC" | "H-C'" | "HN-C'" | "HNCPRIME" => {
                Ok(Self::HNCPrime)
            }
            _ => Err(format!("unrecognized RDC bond_type '{s}'")),
        }
    }

    pub fn infer_from_atoms(
        seq_1: i32,
        atom_1: &str,
        seq_2: i32,
        atom_2: &str,
    ) -> Result<Self, String> {
        let a1 = normalize_rdc_atom_name(atom_1);
        let a2 = normalize_rdc_atom_name(atom_2);
        match (a1, a2, seq_2 - seq_1) {
            ("N", "H", 0) | ("H", "N", 0) => Ok(Self::NH),
            ("CA", "HA" | "HA2" | "HA3", 0) | ("HA" | "HA2" | "HA3", "CA", 0) => Ok(Self::CAHA),
            ("CA", "C", 0) | ("C", "CA", 0) => Ok(Self::CACO),
            ("N", "C", -1) | ("C", "N", 1) => Ok(Self::NCPrime),
            ("H", "C", -1) | ("C", "H", 1) => Ok(Self::HNCPrime),
            _ => Err(format!(
                "unsupported RDC atom pair ({seq_1}:{atom_1} - {seq_2}:{atom_2})"
            )),
        }
    }
}

#[inline]
pub fn normalize_rdc_atom_name(atom: &str) -> &str {
    let t = atom.trim();
    if t.eq_ignore_ascii_case("HN") || t.eq_ignore_ascii_case("H") {
        "H"
    } else if t.eq_ignore_ascii_case("N") {
        "N"
    } else if t.eq_ignore_ascii_case("CA") {
        "CA"
    } else if t.eq_ignore_ascii_case("HA") {
        "HA"
    } else if t.eq_ignore_ascii_case("HA2") || t.eq_ignore_ascii_case("1HA") {
        "HA2"
    } else if t.eq_ignore_ascii_case("HA3") || t.eq_ignore_ascii_case("2HA") {
        "HA3"
    } else if t.eq_ignore_ascii_case("C")
        || t.eq_ignore_ascii_case("CO")
        || t.eq_ignore_ascii_case("C'")
    {
        "C"
    } else {
        t
    }
}

/// Single experimental residual dipolar coupling restraint in an alignment medium.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RdcRestraint {
    pub medium_id: String,
    pub seq_id_1: i32,
    pub res_name_1: String,
    pub atom_name_1: String,
    pub seq_id_2: i32,
    pub res_name_2: String,
    pub atom_name_2: String,
    pub bond_type: RdcBondType,
    pub d_obs_hz: f64,
    pub sigma_hz: f64,
    pub d_max_hz: f64,
    /// When `true`, held out from Saupe SVD fitting and energy/gradient refinement for `Q_free`.
    pub is_free: bool,
}

impl RdcRestraint {
    pub fn validate(&self) -> Result<(), String> {
        if self.medium_id.trim().is_empty() {
            return Err("RDC medium_id must not be empty".to_string());
        }
        if !self.d_obs_hz.is_finite() {
            return Err(format!(
                "non-finite d_obs_hz ({}) on {}:{}",
                self.d_obs_hz, self.seq_id_1, self.atom_name_1
            ));
        }
        if !self.sigma_hz.is_finite() || self.sigma_hz <= 0.0 {
            return Err(format!(
                "invalid sigma_hz ({}) on {}:{}; must be finite and > 0",
                self.sigma_hz, self.seq_id_1, self.atom_name_1
            ));
        }
        if !self.d_max_hz.is_finite() || self.d_max_hz <= 0.0 {
            return Err(format!(
                "invalid d_max_hz ({}) on {}:{}; must be finite and > 0",
                self.d_max_hz, self.seq_id_1, self.atom_name_1
            ));
        }
        for (seq_id, res_name, atom_name) in [
            (
                self.seq_id_1,
                self.res_name_1.trim(),
                self.atom_name_1.as_str(),
            ),
            (
                self.seq_id_2,
                self.res_name_2.trim(),
                self.atom_name_2.as_str(),
            ),
        ] {
            if !res_name.is_empty() && res_name != "." {
                let norm = normalize_rdc_atom_name(atom_name);
                if (norm == "HA2" || norm == "HA3") && !res_name.eq_ignore_ascii_case("GLY") {
                    return Err(format!(
                        "atom '{norm}' on {seq_id}:{res_name} is only valid on GLY"
                    ));
                }
                if norm == "HA" && res_name.eq_ignore_ascii_case("GLY") {
                    return Err(format!(
                        "ambiguous 'HA' on {seq_id}:GLY; specify prochiral HA2 or HA3"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Collection of RDC restraints measured in one independent anisotropic alignment medium.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RdcMediumSet {
    pub medium_id: String,
    pub restraints: Vec<RdcRestraint>,
}

/// Fitted 5-parameter traceless symmetric Saupe order matrix and cross-validation metrics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SaupeTensorFit {
    pub medium_id: String,
    /// Losonczi 5-vector `[S_yy, S_zz, S_xy, S_xz, S_yz]` with `S_xx = -(S_yy + S_zz)`.
    pub s_5: [f64; 5],
    /// Symmetric traceless `3x3` Saupe matrix in the molecular coordinate frame.
    pub s_matrix: [[f64; 3]; 3],
    /// Principal order parameters `[S_xx, S_yy, S_zz]` ordered `|S_zz| >= |S_yy| >= |S_xx|`.
    pub principal_s: [f64; 3],
    /// Axial alignment magnitude normalized to `1D_NH` (`0.5 * D_max(NH) * S_zz`, in Hz).
    pub da_nh_hz: f64,
    /// Rhombicity `R = (2/3) * (S_xx - S_yy) / S_zz` (`0.0 <= R <= 2/3`).
    pub rhombicity: f64,
    /// Right-handed ZYZ Euler angles `[alpha_deg, beta_deg, gamma_deg]` rotating molecular -> PAS.
    pub euler_zyz_deg: [f64; 3],
    /// Condition number `sigma_max / sigma_min` of the weighted `(n_work x 5)` Losonczi SVD matrix.
    pub condition_number: f64,
    pub n_work: usize,
    pub n_free: usize,
    /// Cornilescu `Q` on working restraints (`f64::NAN` when `sum(D_obs^2) <= 1e-15`).
    pub q_work: f64,
    /// Held-out cross-validated Cornilescu `Q_free` (`f64::NAN` when `n_free == 0` or `sum(D_obs^2) <= 1e-15`).
    pub q_free: f64,
    pub rmsd_work_hz: f64,
    pub rmsd_free_hz: f64,
}

/// Single observed vs back-calculated RDC point for residual analysis and SVG scatter plotting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RdcBackCalcPoint {
    pub medium_id: String,
    pub seq_id: i32,
    pub res_name: String,
    pub bond_label: String,
    pub d_obs_hz: f64,
    pub d_calc_hz: f64,
    /// Normalized to `1D_NH` dipolar scale (`D * D_max(NH) / D_max`) for multi-bond scatter plotting.
    pub d_obs_norm_nh_hz: f64,
    pub d_calc_norm_nh_hz: f64,
    pub sigma_hz: f64,
    pub is_free: bool,
}

/// Multi-medium RDC validation report across all alignment media.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RdcValidationSummary {
    pub media: Vec<SaupeTensorFit>,
    pub points: Vec<RdcBackCalcPoint>,
    pub overall_q_work: f64,
    pub overall_q_free: f64,
}

/// Compare two Cornilescu `Q` values, treating non-finite (`NaN`/`Inf`) as strictly worst.
#[inline]
pub fn rdc_q_is_better(candidate_q: f64, incumbent_q: f64) -> bool {
    match (candidate_q.is_finite(), incumbent_q.is_finite()) {
        (true, true) => candidate_q < incumbent_q,
        (true, false) => true,
        (false, _) => false,
    }
}

/// Read multi-medium RDC restraints from a TSV file (ASCII-safe).
pub fn read_rdc_tsv(path: impl AsRef<Path>) -> Result<Vec<RdcMediumSet>, String> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| format!("failed to read RDC TSV '{}': {e}", path.as_ref().display()))?;
    if !bytes.is_ascii() {
        return Err(format!(
            "RDC TSV '{}' contains non-ASCII bytes; expected pure ASCII TSV",
            path.as_ref().display()
        ));
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| {
        format!(
            "invalid UTF-8 in RDC TSV '{}': {e}",
            path.as_ref().display()
        )
    })?;
    parse_rdc_tsv_str(text)
}

/// Parse multi-medium RDC restraints from ASCII TSV text.
///
/// Expected columns (tab- or whitespace-separated after header):
/// `medium_id  seq_id_1  res_name_1  atom_name_1  seq_id_2  res_name_2  atom_name_2  bond_type  d_obs_hz  sigma_hz  [d_max_hz]  [is_free]`
pub fn parse_rdc_tsv_str(text: &str) -> Result<Vec<RdcMediumSet>, String> {
    let raw_bytes = text.as_bytes();
    if !raw_bytes.is_ascii() {
        return Err("RDC TSV input contains non-ASCII characters".to_string());
    }

    let mut media_map: BTreeMap<String, Vec<RdcRestraint>> = BTreeMap::new();
    let mut media_order: Vec<String> = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let lb = raw_line.as_bytes();
        if lb.get(0..1).is_none() {
            continue;
        }
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').map(str::trim).collect()
        } else {
            trimmed.split_whitespace().collect()
        };
        if cols.first().copied() == Some("medium_id") {
            continue;
        }
        if cols.len() < 10 {
            return Err(format!(
                "line {line_no}: expected at least 10 RDC TSV columns, found {}",
                cols.len()
            ));
        }

        let medium_id = cols[0].to_string();
        let seq_id_1: i32 = cols[1]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid seq_id_1 '{}': {e}", cols[1]))?;
        let res_name_1 = cols[2].to_ascii_uppercase();
        let atom_name_1 = normalize_rdc_atom_name(cols[3]).to_string();
        let seq_id_2: i32 = cols[4]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid seq_id_2 '{}': {e}", cols[4]))?;
        let res_name_2 = cols[5].to_ascii_uppercase();
        let atom_name_2 = normalize_rdc_atom_name(cols[6]).to_string();
        let bond_type =
            RdcBondType::parse_label(cols[7]).map_err(|e| format!("line {line_no}: {e}"))?;
        let d_obs_hz: f64 = cols[8]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid d_obs_hz '{}': {e}", cols[8]))?;
        let sigma_hz: f64 = cols[9]
            .parse()
            .map_err(|e| format!("line {line_no}: invalid sigma_hz '{}': {e}", cols[9]))?;
        let d_max_hz: f64 = if let Some(c10) = cols.get(10) {
            if c10.is_empty() || *c10 == "." {
                bond_type.default_d_max_hz()
            } else {
                c10.parse()
                    .map_err(|e| format!("line {line_no}: invalid d_max_hz '{c10}': {e}"))?
            }
        } else {
            bond_type.default_d_max_hz()
        };
        let is_free = if let Some(c11) = cols.get(11) {
            matches!(
                c11.to_ascii_lowercase().as_str(),
                "1" | "true" | "free" | "yes"
            )
        } else {
            false
        };

        let r = RdcRestraint {
            medium_id: medium_id.clone(),
            seq_id_1,
            res_name_1,
            atom_name_1,
            seq_id_2,
            res_name_2,
            atom_name_2,
            bond_type,
            d_obs_hz,
            sigma_hz,
            d_max_hz,
            is_free,
        };
        r.validate().map_err(|e| format!("line {line_no}: {e}"))?;

        if !media_map.contains_key(&medium_id) {
            media_order.push(medium_id.clone());
        }
        media_map.entry(medium_id).or_default().push(r);
    }

    if media_order.is_empty() {
        return Err("RDC TSV contains zero restraints".to_string());
    }

    let mut out = Vec::with_capacity(media_order.len());
    for mid in media_order {
        let restraints = media_map.remove(&mid).unwrap_or_default();
        let n_work = restraints.iter().filter(|r| !r.is_free).count();
        if n_work < 6 {
            return Err(format!(
                "alignment medium '{mid}' has only {n_work} working RDCs; require n >= 6 for 5-parameter Saupe SVD"
            ));
        }
        out.push(RdcMediumSet {
            medium_id: mid,
            restraints,
        });
    }
    Ok(out)
}

/// Format multi-medium RDC sets to canonical ASCII TSV text.
pub fn format_rdc_tsv(media: &[RdcMediumSet]) -> String {
    let mut out = String::from(
        "medium_id\tseq_id_1\tres_name_1\tatom_name_1\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\td_max_hz\tis_free\n",
    );
    for m in media {
        for r in &m.restraints {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.2}\t{}\n",
                r.medium_id,
                r.seq_id_1,
                r.res_name_1,
                r.atom_name_1,
                r.seq_id_2,
                r.res_name_2,
                r.atom_name_2,
                r.bond_type.canonical_label(),
                r.d_obs_hz,
                r.sigma_hz,
                r.d_max_hz,
                if r.is_free { 1 } else { 0 }
            ));
        }
    }
    out
}

/// Write multi-medium RDC sets to an ASCII TSV file.
pub fn write_rdc_tsv(path: impl AsRef<Path>, media: &[RdcMediumSet]) -> Result<(), String> {
    fs::write(path.as_ref(), format_rdc_tsv(media))
        .map_err(|e| format!("failed to write RDC TSV '{}': {e}", path.as_ref().display()))
}

/// Deterministically partition a fraction (`free_fraction`, e.g. `0.15` or `0.20`) of RDCs per
/// alignment medium (stratified by `RdcBondType`) into the held-out `is_free = true` set while
/// guaranteeing at least 6 working restraints remain per medium.
pub fn partition_rdc_free_set(media: &mut [RdcMediumSet], free_fraction: f64, seed: u64) {
    let frac = free_fraction.clamp(0.0, 0.5);
    let mut rng = Rng::new(seed ^ 0x5244_435F_4652_4545);

    for m in media {
        for r in &mut m.restraints {
            r.is_free = false;
        }
        let total = m.restraints.len();
        if frac <= 0.0 || total <= 6 {
            continue;
        }
        let max_free = total - 6;
        let target_free = ((total as f64 * frac).round() as usize).clamp(1, max_free);

        // Stratified reservoir shuffle by bond_type
        let mut by_bt: BTreeMap<RdcBondType, Vec<usize>> = BTreeMap::new();
        for (idx, r) in m.restraints.iter().enumerate() {
            by_bt.entry(r.bond_type).or_default().push(idx);
        }

        let mut picked = Vec::new();
        for indices in by_bt.values_mut() {
            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                indices.swap(i, j);
            }
            let k = ((indices.len() as f64 * frac).round() as usize).min(indices.len());
            picked.extend(indices.iter().copied().take(k));
        }
        picked.sort_unstable();
        picked.dedup();
        if picked.len() > max_free {
            picked.truncate(max_free);
        } else if picked.len() < target_free {
            for idx in 0..total {
                if picked.len() >= target_free {
                    break;
                }
                if !picked.contains(&idx) {
                    picked.push(idx);
                }
            }
        }
        for idx in picked {
            m.restraints[idx].is_free = true;
        }
    }
}

/// Resolve an RDC atom coordinate and rigid-segment index `(res_idx, seg_idx, pos)` on `PdbChain`.
#[inline]
pub fn resolve_chain_rdc_atom(
    chain: &PdbChain,
    seq_id_to_idx: &BTreeMap<i32, usize>,
    seq_id: i32,
    atom_name: &str,
) -> Option<(usize, usize, Vec3)> {
    let &res_idx = seq_id_to_idx.get(&seq_id)?;
    let res: &PdbResidue = chain.residues.get(res_idx)?;
    let norm = normalize_rdc_atom_name(atom_name);
    let is_gly = res.res_name.eq_ignore_ascii_case("GLY");
    match norm {
        "N" => Some((res_idx, 2 * res_idx, res.n)),
        "H" => res.h.map(|p| (res_idx, 2 * res_idx, p)),
        "CA" => Some((res_idx, 2 * res_idx, res.ca)),
        "HA" => {
            if is_gly {
                None
            } else {
                Some((res_idx, 2 * res_idx + 1, res.ha))
            }
        }
        "HA2" => {
            if is_gly {
                res.ha2.map(|p| (res_idx, 2 * res_idx + 1, p))
            } else {
                None
            }
        }
        "HA3" => {
            if is_gly {
                res.ha3.map(|p| (res_idx, 2 * res_idx + 1, p))
            } else {
                None
            }
        }
        "C" => Some((res_idx, 2 * res_idx + 1, res.c)),
        "O" => Some((res_idx, 2 * res_idx + 2, res.o)),
        "CB" => res.cb.map(|p| (res_idx, 2 * res_idx + 1, p)),
        _ => None,
    }
}

/// Count RDC restraints across `media` whose endpoints fail to resolve to finite 3D coordinates on `chain`.
pub fn count_unresolved_rdc_restraints(chain: &PdbChain, media: &[RdcMediumSet]) -> usize {
    let mut seq_to_idx = BTreeMap::new();
    for (idx, r) in chain.residues.iter().enumerate() {
        seq_to_idx.insert(r.seq_id, idx);
    }
    let mut unresolved = 0usize;
    for m in media {
        for r in &m.restraints {
            let ok_1 =
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_1, &r.atom_name_1).is_some();
            let ok_2 =
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_2, &r.atom_name_2).is_some();
            if !ok_1 || !ok_2 {
                unresolved += 1;
            }
        }
    }
    unresolved
}

type ThinSvdM5 = (Vec<[f64; 5]>, [f64; 5], [[f64; 5]; 5]);

/// One-sided Hestenes-Jacobi thin SVD on an `(m x 5)` matrix `A` (`m >= 6`).
/// Computes `A = U * diag(sigma) * V^T` without forming `A^T A`, preserving full `f64` dynamic range.
fn jacobi_thin_svd_m5(a: &[[f64; 5]]) -> Result<ThinSvdM5, String> {
    let m = a.len();
    if m < 6 {
        return Err(format!("Saupe SVD requires m >= 6 rows, got {m}"));
    }

    let mut w: Vec<[f64; 5]> = a.to_vec();
    let mut v = [[0.0f64; 5]; 5];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _sweep in 0..45 {
        let mut max_off = 0.0f64;
        for p in 0..4 {
            for q in (p + 1)..5 {
                let mut app = 0.0f64;
                let mut aqq = 0.0f64;
                let mut apq = 0.0f64;
                for row in &w {
                    app += row[p] * row[p];
                    aqq += row[q] * row[q];
                    apq += row[p] * row[q];
                }
                let scale = (app * aqq).sqrt();
                if scale <= 1e-300 {
                    continue;
                }
                let rel = apq.abs() / scale;
                if rel > max_off {
                    max_off = rel;
                }
                if rel <= 1e-15 {
                    continue;
                }
                let zeta = (aqq - app) / (2.0 * apq);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for row in &mut w {
                    let wp = row[p];
                    let wq = row[q];
                    row[p] = c * wp - s * wq;
                    row[q] = s * wp + c * wq;
                }
                for row in &mut v {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        if max_off <= 1e-15 {
            break;
        }
    }

    let mut sigma = [0.0f64; 5];
    for j in 0..5 {
        let mut n2 = 0.0f64;
        for row in &w {
            n2 += row[j] * row[j];
        }
        sigma[j] = n2.sqrt();
    }

    // Sort singular values descending
    let mut order = [0usize, 1, 2, 3, 4];
    order.sort_by(|&a_idx, &b_idx| sigma[b_idx].total_cmp(&sigma[a_idx]));

    let s_max = sigma[order[0]];
    let s_min = sigma[order[4]];
    if !s_max.is_finite() || !s_min.is_finite() || s_max <= 1e-30 || s_min <= 0.0 {
        return Err("degenerate Saupe SVD design matrix (zero singular value; collinear/coplanar bond vectors)".to_string());
    }
    let cond = s_max / s_min;
    if !cond.is_finite() || cond > MAX_SAUPE_SVD_COND {
        return Err(format!(
            "ill-conditioned Saupe SVD design matrix (cond = {cond:.3e} > {:.1e}; collinear/coplanar bond vectors)",
            MAX_SAUPE_SVD_COND
        ));
    }

    let mut u = vec![[0.0f64; 5]; m];
    let mut sig_sorted = [0.0f64; 5];
    let mut v_sorted = [[0.0f64; 5]; 5];
    for (new_j, &old_j) in order.iter().enumerate() {
        let sj = sigma[old_j];
        sig_sorted[new_j] = sj;
        for i in 0..m {
            u[i][new_j] = w[i][old_j] / sj;
        }
        for i in 0..5 {
            v_sorted[i][new_j] = v[i][old_j];
        }
    }

    Ok((u, sig_sorted, v_sorted))
}

/// Exact `3x3` symmetric Jacobi eigensolver: `s_mat = rot * diag(evals) * rot^T`
/// with canonical ordering `|evals[2]| >= |evals[1]| >= |evals[0]|` and right-handed `det(rot) = +1`.
#[allow(clippy::needless_range_loop)]
pub fn decompose_saupe_matrix(s_mat: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    let mut a = s_mat;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..40 {
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        if a[0][2].abs() > max_val {
            p = 0;
            q = 2;
            max_val = a[0][2].abs();
        }
        if a[1][2].abs() > max_val {
            p = 1;
            q = 2;
            max_val = a[1][2].abs();
        }
        if max_val <= 1e-16 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs() + 1e-30) {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let zeta = (aqq - app) / (2.0 * apq);
        let t = if zeta >= 0.0 {
            1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
        } else {
            -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;

        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for row in &mut v {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }

    let raw_evals = [a[0][0], a[1][1], a[2][2]];
    // Order [S_xx, S_yy, S_zz] such that |S_zz| >= |S_yy| >= |S_xx|
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&i, &j| raw_evals[i].abs().total_cmp(&raw_evals[j].abs()));

    let principal_s = [raw_evals[ord[0]], raw_evals[ord[1]], raw_evals[ord[2]]];
    // Columns of `rot` are principal axes e_x, e_y, e_z in molecular frame (`s_mat = rot * diag * rot^T`)
    let mut rot = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rot[r][c] = v[r][ord[c]];
        }
    }

    // Canonicalize principal axis signs cleanly in SO(3):
    // 1. Ensure e_z (`col 2`) lies in upper hemisphere (`z >= 0`, breaking ties by `y`, `x`)
    let ez = [rot[0][2], rot[1][2], rot[2][2]];
    let flip_z = ez[2] < -1e-12
        || (ez[2].abs() <= 1e-12 && (ez[1] < -1e-12 || (ez[1].abs() <= 1e-12 && ez[0] < 0.0)));
    if flip_z {
        for row in &mut rot {
            row[2] = -row[2];
        }
    }
    // 2. Ensure e_x (`col 0`) has canonical sign (`x >= 0`, breaking ties by `y`, `z`)
    let ex = [rot[0][0], rot[1][0], rot[2][0]];
    let flip_x = ex[0] < -1e-12
        || (ex[0].abs() <= 1e-12 && (ex[1] < -1e-12 || (ex[1].abs() <= 1e-12 && ex[2] < 0.0)));
    if flip_x {
        for row in &mut rot {
            row[0] = -row[0];
        }
    }
    // 3. Enforce strictly right-handed frame `e_y = e_z x e_x` (`det(rot) = +1`)
    let ez_f = [rot[0][2], rot[1][2], rot[2][2]];
    let ex_f = [rot[0][0], rot[1][0], rot[2][0]];
    let ey_rh = v_cross(ez_f, ex_f);
    let dot_y = ey_rh[0] * rot[0][1] + ey_rh[1] * rot[1][1] + ey_rh[2] * rot[2][1];
    if dot_y < 0.0 {
        for row in &mut rot {
            row[1] = -row[1];
        }
    }

    // Extract active ZYZ Euler angles `[alpha, beta, gamma]` in degrees for `R_mol_to_pas = rot^T`:
    // Convention: `rot = Rz(alpha) * Ry(beta) * Rz(gamma)` (`s_mat = rot * diag(principal_s) * rot^T`),
    // where `beta in [0, 180]`, `alpha in [0, 360)`, `gamma in [0, 360)`.
    let cos_beta = rot[2][2].clamp(-1.0, 1.0);
    let beta = cos_beta.acos();
    let sin_beta = beta.sin();
    let (alpha, gamma) = if sin_beta > 1e-8 {
        let a_rad = rot[1][2].atan2(rot[0][2]);
        let g_rad = rot[2][1].atan2(-rot[2][0]);
        (a_rad, g_rad)
    } else {
        let a_rad = rot[1][0].atan2(rot[0][0]);
        (a_rad, 0.0)
    };
    let wrap_360 = |deg: f64| -> f64 {
        let mut d = deg % 360.0;
        if d < 0.0 {
            d += 360.0;
        }
        if (d - 360.0).abs() < 1e-12 {
            0.0
        } else {
            d
        }
    };
    let euler_zyz_deg = [
        wrap_360(alpha.to_degrees()),
        beta.to_degrees(),
        wrap_360(gamma.to_degrees()),
    ];

    (principal_s, rot, euler_zyz_deg)
}

/// Construct a molecular-frame `3x3` traceless symmetric Saupe matrix from `(D_a^NH, R, euler_zyz_deg)`
/// using `rot = Rz(alpha) * Ry(beta) * Rz(gamma)` and `s_mat = rot * diag(S_xx, S_yy, S_zz) * rot^T`.
pub fn saupe_matrix_from_pas(
    da_nh_hz: f64,
    rhombicity: f64,
    euler_zyz_deg: [f64; 3],
) -> [[f64; 3]; 3] {
    let szz = 2.0 * da_nh_hz / DMAX_NH_HZ;
    let r = rhombicity.clamp(0.0, 2.0 / 3.0);
    let sxx = -0.5 * szz * (1.0 - 1.5 * r);
    let syy = -0.5 * szz * (1.0 + 1.5 * r);
    let diag = [sxx, syy, szz];

    let (sa, ca) = euler_zyz_deg[0].to_radians().sin_cos();
    let (sb, cb) = euler_zyz_deg[1].to_radians().sin_cos();
    let (sg, cg) = euler_zyz_deg[2].to_radians().sin_cos();

    let rot = [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ];

    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += rot[i][k] * diag[k] * rot[j][k];
            }
            out[i][j] = sum;
        }
    }
    out
}

/// Back-calculate `D_calc` (Hz) for unit vector `u` given `d_max_hz` and `3x3` Saupe matrix `s`.
#[inline]
pub fn calc_rdc_from_unit_vec(u: Vec3, d_max_hz: f64, s: &[[f64; 3]; 3]) -> f64 {
    let su0 = s[0][0] * u[0] + s[0][1] * u[1] + s[0][2] * u[2];
    let su1 = s[1][0] * u[0] + s[1][1] * u[1] + s[1][2] * u[2];
    let su2 = s[2][0] * u[0] + s[2][1] * u[1] + s[2][2] * u[2];
    d_max_hz * (u[0] * su0 + u[1] * su1 + u[2] * su2)
}

/// Compute Cornilescu `Q`-factor over `(d_obs_norm, d_calc_norm)` pairs.
/// Returns `f64::NAN` whenever `pairs.is_empty()` or `sum(d_obs^2) <= 1e-15` (never `0.0`).
#[inline]
pub fn cornilescu_q(pairs: &[(f64, f64)]) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for &(obs, calc) in pairs {
        if !obs.is_finite() || !calc.is_finite() {
            return f64::NAN;
        }
        let diff = obs - calc;
        num += diff * diff;
        den += obs * obs;
    }
    if den <= 1e-15 {
        f64::NAN
    } else {
        (num / den).sqrt()
    }
}

/// Fit the exact weighted 5-parameter Losonczi Saupe tensor via thin Jacobi SVD on working
/// (`!is_free`) RDCs of `medium`, and evaluate both working `Q` and held-out `Q_free`.
pub fn fit_saupe_tensor_svd(
    chain: &PdbChain,
    medium: &RdcMediumSet,
) -> Result<SaupeTensorFit, String> {
    let (fit, _) = fit_saupe_tensor_with_points(chain, medium)?;
    Ok(fit)
}

pub fn fit_saupe_tensor_with_points(
    chain: &PdbChain,
    medium: &RdcMediumSet,
) -> Result<(SaupeTensorFit, Vec<RdcBackCalcPoint>), String> {
    if chain.residues.is_empty() {
        return Err("cannot fit Saupe tensor on an empty PdbChain".to_string());
    }
    let mut seq_to_idx = BTreeMap::new();
    for (idx, r) in chain.residues.iter().enumerate() {
        seq_to_idx.insert(r.seq_id, idx);
    }

    let mut a_rows: Vec<[f64; 5]> = Vec::new();
    let mut b_vec: Vec<f64> = Vec::new();
    let mut resolved: Vec<(&RdcRestraint, Vec3)> = Vec::new();

    for r in &medium.restraints {
        r.validate()?;
        let (Some((_, _, p1)), Some((_, _, p2))) = (
            resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_1, &r.atom_name_1),
            resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_2, &r.atom_name_2),
        ) else {
            continue;
        };
        let dv = v_sub(p2, p1);
        let dist = v_norm(dv);
        if !dist.is_finite() || dist < 1e-6 {
            return Err(format!(
                "zero/degenerate internuclear distance ({dist}) on {}:{} - {}:{}",
                r.seq_id_1, r.atom_name_1, r.seq_id_2, r.atom_name_2
            ));
        }
        let u = v_scale(dv, 1.0 / dist);
        resolved.push((r, u));

        if !r.is_free {
            let w_scale = r.d_max_hz / r.sigma_hz;
            let ux = u[0];
            let uy = u[1];
            let uz = u[2];
            a_rows.push([
                w_scale * (uy * uy - ux * ux),
                w_scale * (uz * uz - ux * ux),
                w_scale * (2.0 * ux * uy),
                w_scale * (2.0 * ux * uz),
                w_scale * (2.0 * uy * uz),
            ]);
            b_vec.push(r.d_obs_hz / r.sigma_hz);
        }
    }

    let n_work = a_rows.len();
    if n_work < 6 {
        return Err(format!(
            "alignment medium '{}' resolved only {n_work} finite working RDCs on chain; require n >= 6",
            medium.medium_id
        ));
    }

    let (u_mat, sigmas, v_mat) = jacobi_thin_svd_m5(&a_rows)?;
    let cond = sigmas[0] / sigmas[4];

    // Solve s_5 = V * Sigma^-1 * U^T * b
    let mut ut_b = [0.0f64; 5];
    for (row_u, &bi) in u_mat.iter().zip(b_vec.iter()) {
        for j in 0..5 {
            ut_b[j] += row_u[j] * bi;
        }
    }
    let mut s_5 = [0.0f64; 5];
    for i in 0..5 {
        let mut acc = 0.0f64;
        for j in 0..5 {
            acc += v_mat[i][j] * (ut_b[j] / sigmas[j]);
        }
        s_5[i] = acc;
    }

    let syy = s_5[0];
    let szz = s_5[1];
    let sxy = s_5[2];
    let sxz = s_5[3];
    let syz = s_5[4];
    let sxx = -(syy + szz);
    let s_matrix = [[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]];

    let (principal_s, _rot, euler_zyz_deg) = decompose_saupe_matrix(s_matrix);
    let da_nh_hz = 0.5 * DMAX_NH_HZ * principal_s[2];
    let rhombicity = if principal_s[2].abs() > 1e-18 {
        ((2.0 / 3.0) * (principal_s[0] - principal_s[1]) / principal_s[2]).clamp(0.0, 2.0 / 3.0)
    } else {
        0.0
    };

    let mut work_norm_pairs = Vec::with_capacity(n_work);
    let mut free_norm_pairs = Vec::new();
    let mut work_sq_hz = 0.0f64;
    let mut free_sq_hz = 0.0f64;
    let mut points = Vec::with_capacity(resolved.len());

    for (r, u) in resolved {
        let d_calc = calc_rdc_from_unit_vec(u, r.d_max_hz, &s_matrix);
        let scale_nh = DMAX_NH_HZ / r.d_max_hz;
        let obs_norm = r.d_obs_hz * scale_nh;
        let calc_norm = d_calc * scale_nh;
        let diff_hz = r.d_obs_hz - d_calc;
        if r.is_free {
            free_norm_pairs.push((obs_norm, calc_norm));
            free_sq_hz += diff_hz * diff_hz;
        } else {
            work_norm_pairs.push((obs_norm, calc_norm));
            work_sq_hz += diff_hz * diff_hz;
        }
        points.push(RdcBackCalcPoint {
            medium_id: medium.medium_id.clone(),
            seq_id: r.seq_id_1,
            res_name: r.res_name_1.clone(),
            bond_label: r.bond_type.canonical_label().to_string(),
            d_obs_hz: r.d_obs_hz,
            d_calc_hz: d_calc,
            d_obs_norm_nh_hz: obs_norm,
            d_calc_norm_nh_hz: calc_norm,
            sigma_hz: r.sigma_hz,
            is_free: r.is_free,
        });
    }

    let n_free = free_norm_pairs.len();
    let q_work = cornilescu_q(&work_norm_pairs);
    let q_free = cornilescu_q(&free_norm_pairs);
    let rmsd_work_hz = (work_sq_hz / (n_work as f64)).sqrt();
    let rmsd_free_hz = if n_free > 0 {
        (free_sq_hz / (n_free as f64)).sqrt()
    } else {
        f64::NAN
    };

    Ok((
        SaupeTensorFit {
            medium_id: medium.medium_id.clone(),
            s_5,
            s_matrix,
            principal_s,
            da_nh_hz,
            rhombicity,
            euler_zyz_deg,
            condition_number: cond,
            n_work,
            n_free,
            q_work,
            q_free,
            rmsd_work_hz,
            rmsd_free_hz,
        },
        points,
    ))
}

/// Evaluate multi-medium Saupe SVD fits, back-calculated points, and overall `Q` / `Q_free`.
pub fn evaluate_rdc_validation(
    chain: &PdbChain,
    media: &[RdcMediumSet],
) -> Result<RdcValidationSummary, String> {
    if media.is_empty() {
        return Err("evaluate_rdc_validation called with empty media slice".to_string());
    }
    let mut fits = Vec::with_capacity(media.len());
    let mut all_points = Vec::new();
    let mut work_pairs = Vec::new();
    let mut free_pairs = Vec::new();

    for m in media {
        let (fit, pts) = fit_saupe_tensor_with_points(chain, m)?;
        for p in &pts {
            if p.is_free {
                free_pairs.push((p.d_obs_norm_nh_hz, p.d_calc_norm_nh_hz));
            } else {
                work_pairs.push((p.d_obs_norm_nh_hz, p.d_calc_norm_nh_hz));
            }
        }
        fits.push(fit);
        all_points.extend(pts);
    }

    Ok(RdcValidationSummary {
        media: fits,
        points: all_points,
        overall_q_work: cornilescu_q(&work_pairs),
        overall_q_free: cornilescu_q(&free_pairs),
    })
}

/// Evaluate the variable-projection harmonic RDC energy across working (`!is_free`) restraints
/// and accumulate exact Envelope-Theorem Cartesian gradients into `add_atom_grad(seg, pos, force)`.
///
/// By the Envelope Theorem, since `S*_m(theta)` minimizes the weighted harmonic RDC chi-squared
/// `0.5 * w_rdc * sum_{i in work} ((D_calc_i(theta, S) - D_obs_i) / sigma_i)^2` at each conformer
/// step, `dE / dS |_{S*} = 0`, so `dE(theta, S*(theta)) / dtheta = partial E / partial theta |_{S*}`.
pub fn accumulate_rdc_energy_and_forces<F>(
    chain: &PdbChain,
    media: &[RdcMediumSet],
    w_rdc: f64,
    mut add_atom_grad: F,
) -> f64
where
    F: FnMut(usize, Vec3, Vec3),
{
    if w_rdc <= 0.0 || media.is_empty() || chain.residues.is_empty() {
        return 0.0;
    }

    let mut seq_to_idx = BTreeMap::new();
    for (idx, r) in chain.residues.iter().enumerate() {
        seq_to_idx.insert(r.seq_id, idx);
    }

    let mut total_e = 0.0f64;

    for m in media {
        let Ok((fit, _)) = fit_saupe_tensor_with_points(chain, m) else {
            continue;
        };
        let s = &fit.s_matrix;

        for r in &m.restraints {
            if r.is_free {
                continue;
            }
            let (Some((_, seg_a, pa)), Some((_, seg_b, pb))) = (
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_1, &r.atom_name_1),
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_2, &r.atom_name_2),
            ) else {
                continue;
            };
            let v = v_sub(pb, pa);
            let dist = v_norm(v);
            if dist < 1e-6 {
                continue;
            }
            let inv_r = 1.0 / dist;
            let u = v_scale(v, inv_r);
            let su = [
                s[0][0] * u[0] + s[0][1] * u[1] + s[0][2] * u[2],
                s[1][0] * u[0] + s[1][1] * u[1] + s[1][2] * u[2],
                s[2][0] * u[0] + s[2][1] * u[1] + s[2][2] * u[2],
            ];
            let ut_s_u = v_dot(u, su);
            let d_calc = r.d_max_hz * ut_s_u;
            let inv_sig2 = 1.0 / (r.sigma_hz * r.sigma_hz);
            let resid = d_calc - r.d_obs_hz;

            total_e += 0.5 * w_rdc * resid * resid * inv_sig2;

            // d(u^T S u)/dv = (2 / r) * (S u - (u^T S u) u), with v = pb - pa
            let pref = w_rdc * resid * inv_sig2 * (2.0 * r.d_max_hz * inv_r);
            let grad_pb = [
                pref * (su[0] - ut_s_u * u[0]),
                pref * (su[1] - ut_s_u * u[1]),
                pref * (su[2] - ut_s_u * u[2]),
            ];
            add_atom_grad(seg_b, pb, grad_pb);
            add_atom_grad(seg_a, pa, [-grad_pb[0], -grad_pb[1], -grad_pb[2]]);
        }
    }

    total_e
}

/// Evaluate RDC energy and Forces with FIXED pre-computed Saupe matrices `fixed_s_matrices`
/// (used for exact CPU vs WebGPU single-step kernel parity verification and alternating-projection steps).
pub fn accumulate_rdc_energy_and_forces_fixed_saupe<F>(
    chain: &PdbChain,
    media: &[RdcMediumSet],
    fixed_s_matrices: &[[[f64; 3]; 3]],
    w_rdc: f64,
    mut add_atom_grad: F,
) -> f64
where
    F: FnMut(usize, Vec3, Vec3),
{
    if w_rdc <= 0.0
        || media.is_empty()
        || fixed_s_matrices.len() != media.len()
        || chain.residues.is_empty()
    {
        return 0.0;
    }

    let mut seq_to_idx = BTreeMap::new();
    for (idx, r) in chain.residues.iter().enumerate() {
        seq_to_idx.insert(r.seq_id, idx);
    }

    let mut total_e = 0.0f64;
    for (m, s) in media.iter().zip(fixed_s_matrices.iter()) {
        for r in &m.restraints {
            if r.is_free {
                continue;
            }
            let (Some((_, seg_a, pa)), Some((_, seg_b, pb))) = (
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_1, &r.atom_name_1),
                resolve_chain_rdc_atom(chain, &seq_to_idx, r.seq_id_2, &r.atom_name_2),
            ) else {
                continue;
            };
            let v = v_sub(pb, pa);
            let dist = v_norm(v);
            if dist < 1e-6 {
                continue;
            }
            let inv_r = 1.0 / dist;
            let u = v_scale(v, inv_r);
            let su = [
                s[0][0] * u[0] + s[0][1] * u[1] + s[0][2] * u[2],
                s[1][0] * u[0] + s[1][1] * u[1] + s[1][2] * u[2],
                s[2][0] * u[0] + s[2][1] * u[1] + s[2][2] * u[2],
            ];
            let ut_s_u = v_dot(u, su);
            let d_calc = r.d_max_hz * ut_s_u;
            let inv_sig2 = 1.0 / (r.sigma_hz * r.sigma_hz);
            let resid = d_calc - r.d_obs_hz;

            total_e += 0.5 * w_rdc * resid * resid * inv_sig2;

            let pref = w_rdc * resid * inv_sig2 * (2.0 * r.d_max_hz * inv_r);
            let grad_pb = [
                pref * (su[0] - ut_s_u * u[0]),
                pref * (su[1] - ut_s_u * u[1]),
                pref * (su[2] - ut_s_u * u[2]),
            ];
            add_atom_grad(seg_b, pb, grad_pb);
            add_atom_grad(seg_a, pa, [-grad_pb[0], -grad_pb[1], -grad_pb[2]]);
        }
    }
    total_e
}
